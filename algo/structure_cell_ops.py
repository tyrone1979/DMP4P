"""
Cell-wise food_id crossover / mutate for structure-cell GA.

Chromosomes stay bitmask over the pool; operators act on food_id sets per (meal, major).
CUDA hot path stays on-device (vectorized per cell over the population).
CPU numpy path remains for device=cpu / tests.
"""
from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
import torch

from algo.structure_cell_flags import MAJOR_NUTRIENT_RESPONSIBILITY
from algo.structure_features import MAJOR_CODES, MEAL_NAMES


MEAL_ID = {"breakfast": 1, "lunch": 2, "dinner": 3, "snack": 4}


def ensure_cell_tensors_on_device(
    cells: list[dict[str, Any]], device: torch.device
) -> list[dict[str, Any]]:
    """Attach rows/grams/nut torch tensors to each cell (idempotent per device)."""
    for cell in cells:
        if cell.get("end", 0) <= cell.get("start", 0):
            continue
        if cell.get("_device") == device and "rows_t" in cell:
            continue
        cell["rows_t"] = torch.as_tensor(cell["rows"], device=device, dtype=torch.long)
        cell["grams_t"] = torch.as_tensor(cell["grams"], device=device, dtype=torch.float32)
        cell["nut_t"] = torch.as_tensor(cell["nut"], device=device, dtype=torch.float32)
        cell["_device"] = device
    return cells


@torch.no_grad()
def _cap_true_mask(
    mask: torch.Tensor, max_k: int, gen: torch.Generator
) -> torch.Tensor:
    """mask [B,C] → at most max_k True bits/row (random keep among True)."""
    if max_k <= 0:
        return torch.zeros_like(mask)
    b, c = mask.shape
    if c <= max_k:
        return mask
    cnt = mask.sum(dim=1)
    if bool((cnt <= max_k).all().item()):
        return mask
    rnd = torch.rand((b, c), device=mask.device, generator=gen)
    scores = torch.where(mask, rnd, torch.full_like(rnd, -1.0))
    topv, topi = scores.topk(max_k, dim=1)
    out = torch.zeros_like(mask)
    out.scatter_(1, topi, topv >= 0)
    under = cnt <= max_k
    if bool(under.any().item()):
        out[under] = mask[under]
    return out


@torch.no_grad()
def _gpu_pick_index(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    *,
    topk: int,
    explore: float,
    gen: torch.Generator,
) -> torch.Tensor:
    """
    Per-row pick among eligible columns.
    scores/eligible: [B, C]; returns index in 0..C-1 (0 if none eligible).
    """
    b, c = scores.shape
    device = scores.device
    neg = torch.finfo(scores.dtype).min
    sc = torch.where(eligible, scores, torch.full_like(scores, neg))
    if explore > 0:
        use_rand = torch.rand(b, device=device, generator=gen) < float(explore)
        rnd = torch.rand((b, c), device=device, generator=gen)
        rnd = torch.where(eligible, rnd, torch.full_like(rnd, neg))
        sc = torch.where(use_rand.unsqueeze(1), rnd, sc)
    k = max(1, min(int(topk), c))
    topv, topi = sc.topk(k, dim=1)
    # random among top-k that are finite/eligible
    valid = topv > (neg / 2)
    # pick slot
    slot = torch.randint(0, k, (b,), device=device, generator=gen)
    # if chosen slot invalid, fall back to best (slot 0)
    slot = torch.where(valid.gather(1, slot.unsqueeze(1)).squeeze(1), slot, torch.zeros_like(slot))
    return topi.gather(1, slot.unsqueeze(1)).squeeze(1)


def build_band_tensors(
    personal: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, set[int]]:
    """Return band_lo, band_hi shaped [4, 9] and upper_only major set."""
    bands = personal.get("bands") or {}
    upper = set(int(x) for x in (personal.get("upper_only_majors") or (8, 9)))
    lo = torch.zeros(4, 9, dtype=torch.float32, device=device)
    hi = torch.full((4, 9), 1e9, dtype=torch.float32, device=device)
    for i, meal in enumerate(MEAL_NAMES):
        for j, mi in enumerate(MAJOR_CODES):
            key = f"g_major_{mi}_{meal}"
            b = bands.get(key) or {}
            lo[i, j] = float(b.get("lo", 0.0) or 0.0)
            if b.get("hi") is not None:
                hi[i, j] = float(b["hi"])
    return lo, hi, upper


def build_cell_index_cache(
    cell_ranges: dict[str, dict[int, tuple[int, int]]],
    food_ids_np: np.ndarray,
    grams_np: np.ndarray,
    nut_np: np.ndarray,
) -> list[dict[str, Any]]:
    """Precompute per-cell row indices + food_id→row map (CPU)."""
    cells: list[dict[str, Any]] = []
    for mi, meal in enumerate(MEAL_NAMES):
        for ji, maj in enumerate(MAJOR_CODES):
            start, end = cell_ranges.get(meal, {}).get(maj, (0, 0))
            if end <= start:
                cells.append(
                    {
                        "mi": mi,
                        "ji": ji,
                        "maj": maj,
                        "start": start,
                        "end": end,
                        "rows": np.asarray([], dtype=np.int64),
                        "fids": np.asarray([], dtype=np.int64),
                        "fid_to_row": {},
                        "grams": np.asarray([], dtype=np.float32),
                        "nut": np.zeros((0, nut_np.shape[1]), dtype=np.float32),
                    }
                )
                continue
            rows = np.arange(start, end, dtype=np.int64)
            fids = food_ids_np[start:end].astype(np.int64, copy=False)
            fid_to_row: dict[int, int] = {}
            for loc, fid in enumerate(fids):
                fi = int(fid)
                if fi >= 0 and fi not in fid_to_row:
                    fid_to_row[fi] = int(start + loc)
            cells.append(
                {
                    "mi": mi,
                    "ji": ji,
                    "maj": maj,
                    "start": start,
                    "end": end,
                    "rows": rows,
                    "fids": fids,
                    "fid_to_row": fid_to_row,
                    "grams": grams_np[start:end].astype(np.float32, copy=False),
                    "nut": nut_np[start:end].astype(np.float32, copy=False),
                }
            )
    return cells


def cell_grams_batch(
    sel: torch.Tensor,
    grams: torch.Tensor,
    meal_id: torch.Tensor,
    major_id: torch.Tensor,
) -> torch.Tensor:
    """sel [P,N] float/bool → cell_g [P,4,9]."""
    p = sel.size(0)
    device = sel.device
    out = torch.zeros(p, 4, 9, dtype=torch.float32, device=device)
    g = sel.float() * grams
    for i, meal in enumerate(MEAL_NAMES):
        mid = MEAL_ID[meal]
        m_mask = meal_id == mid
        for j, mi in enumerate(MAJOR_CODES):
            out[:, i, j] = (g * (m_mask & (major_id == mi)).float()).sum(dim=1)
    return out


def structure_ok_batch(
    cell_g: torch.Tensor,
    band_lo: torch.Tensor,
    band_hi: torch.Tensor,
    upper_only: set[int],
) -> torch.Tensor:
    """S_ok [P,4,9]."""
    ok = cell_g <= (band_hi + 1e-6)
    for j, mi in enumerate(MAJOR_CODES):
        if mi in upper_only:
            continue
        lo_j = band_lo[:, j]  # [4]
        ge = cell_g[:, :, j] >= (lo_j.unsqueeze(0) - 1e-6)
        mask = (lo_j > 0).unsqueeze(0).expand_as(ge)
        ok[:, :, j] = torch.where(mask, ok[:, :, j] & ge, ok[:, :, j])
    return ok


def target_ok_batch(
    nut_sums: torch.Tensor,
    tgt_lo: torch.Tensor,
    tgt_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """T_ok [P,U], below [P,U], above [P,U]."""
    below = nut_sums < (tgt_lo - 1e-6)
    above = nut_sums > (tgt_hi + 1e-6)
    ok = ~(below | above)
    return ok, below, above


def evolve_mask_batch(
    s_ok: torch.Tensor,
    t_ok: torch.Tensor,
    nutrient_names: list[str],
    *,
    responsibility: Optional[dict[str, tuple[int, ...]]] = None,
    band_hi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    evolve [P,4,9]: True = may change food_ids in that cell.
    frozen = S_ok AND (all T ok OR major unrelated to unmet nutrients).
    hi≤0 cells are never evolvable (pool hard-locked empty).
    """
    resp = responsibility or MAJOR_NUTRIENT_RESPONSIBILITY
    p = s_ok.size(0)
    device = s_ok.device
    all_t = t_ok.all(dim=1)  # [P]
    resp_maj = torch.zeros(p, 9, dtype=torch.bool, device=device)
    unmet = ~t_ok
    for u, name in enumerate(nutrient_names):
        majors = resp.get(name, ())
        if not majors:
            continue
        col = unmet[:, u]
        for mi in majors:
            if 1 <= int(mi) <= 9:
                resp_maj[:, int(mi) - 1] = resp_maj[:, int(mi) - 1] | col
    evolve = ~s_ok
    for j in range(9):
        need = (~all_t) & resp_maj[:, j]
        evolve[:, :, j] = evolve[:, :, j] | (s_ok[:, :, j] & need.unsqueeze(1))
    if band_hi is not None:
        # [4,9] → broadcast: hi≤0 ⇒ never evolve
        locked = band_hi <= 0
        evolve = evolve & (~locked.unsqueeze(0))
    return evolve


def selected_food_ids_in_cell(
    pop_row: torch.Tensor,
    food_ids_tensor: torch.Tensor,
    start: int,
    end: int,
) -> set[int]:
    if end <= start:
        return set()
    on = torch.where(pop_row[start:end])[0]
    if on.numel() == 0:
        return set()
    fids = food_ids_tensor[start:end][on]
    return {int(x) for x in fids.tolist() if int(x) >= 0}


def write_cell_food_ids(
    pop_row: torch.Tensor,
    food_ids_tensor: torch.Tensor,
    start: int,
    end: int,
    food_ids: set[int],
    max_k: int,
) -> None:
    if end <= start:
        return
    pop_row[start:end] = False
    if not food_ids:
        return
    slice_ids = food_ids_tensor[start:end]
    n = 0
    for local in range(end - start):
        fid = int(slice_ids[local].item())
        if fid in food_ids:
            pop_row[start + local] = True
            n += 1
            if n >= max_k:
                break


def uniform_mix_food_ids(sa: set[int], sb: set[int], rng: random.Random) -> tuple[set[int], set[int]]:
    shared = sa & sb
    only = list(sa ^ sb)
    rng.shuffle(only)
    ca = set(shared)
    cb = set(shared)
    for fid in only:
        if rng.random() < 0.5:
            ca.add(fid)
        else:
            cb.add(fid)
    return ca, cb


def _selected_fids_np(pop_row: np.ndarray, cell: dict[str, Any]) -> set[int]:
    start, end = cell["start"], cell["end"]
    if end <= start:
        return set()
    on = np.flatnonzero(pop_row[start:end])
    if on.size == 0:
        return set()
    return {int(x) for x in cell["fids"][on] if int(x) >= 0}


def _write_fids_np(
    pop_row: np.ndarray,
    cell: dict[str, Any],
    food_ids: set[int],
    max_k: int,
    rng: Optional[random.Random] = None,
) -> None:
    start, end = cell["start"], cell["end"]
    if end <= start:
        return
    pop_row[start:end] = False
    fids = [int(f) for f in food_ids if int(f) in cell["fid_to_row"]]
    if not fids:
        return
    if len(fids) > max_k:
        rng = rng or random.Random()
        fids = rng.sample(fids, max_k)
    else:
        rng = rng or random.Random()
        rng.shuffle(fids)
    for fid in fids:
        pop_row[cell["fid_to_row"][fid]] = True


def _pick_topk(
    scored: list[tuple[Any, float]],
    rng: random.Random,
    topk: int,
    *,
    explore_frac: float,
) -> Any:
    """Guided pick: explore_frac → uniform; else random among top-k by score."""
    if not scored:
        raise ValueError("empty scored list")
    if len(scored) == 1:
        return scored[0][0]
    if explore_frac > 0 and rng.random() < float(explore_frac):
        return rng.choice([x[0] for x in scored])
    scored = sorted(scored, key=lambda t: t[1], reverse=True)
    k = max(1, min(int(topk), len(scored)))
    return rng.choice([x[0] for x in scored[:k]])


def cell_crossover_mutate_fast(
    pop: torch.Tensor,
    cells: list[dict[str, Any]],
    evolve: torch.Tensor,
    s_ok: torch.Tensor,
    cell_g: torch.Tensor,
    band_lo: torch.Tensor,
    band_hi: torch.Tensor,
    t_below: torch.Tensor,
    t_above: torch.Tensor,
    nutrient_names: list[str],
    *,
    crossover_rate: float,
    mutation_rate: float,
    max_foods_per_cell: int,
    elite_size: int = 0,
    responsibility: Optional[dict[str, tuple[int, ...]]] = None,
    rng: Optional[random.Random] = None,
    mutate_topk: int = 5,
    mutate_explore_frac: float = 0.25,
    protect_elites: bool = True,
) -> None:
    """In-place crossover+mutate. CUDA → on-device path; else CPU numpy."""
    if pop.is_cuda:
        cell_crossover_mutate_gpu(
            pop,
            cells,
            evolve,
            s_ok,
            cell_g,
            band_lo,
            band_hi,
            t_below,
            t_above,
            nutrient_names,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            max_foods_per_cell=max_foods_per_cell,
            elite_size=elite_size,
            responsibility=responsibility,
            rng=rng,
            mutate_topk=mutate_topk,
            mutate_explore_frac=mutate_explore_frac,
            protect_elites=protect_elites,
        )
    else:
        _cell_crossover_mutate_cpu(
            pop,
            cells,
            evolve,
            s_ok,
            cell_g,
            band_lo,
            band_hi,
            t_below,
            t_above,
            nutrient_names,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            max_foods_per_cell=max_foods_per_cell,
            elite_size=elite_size,
            responsibility=responsibility,
            rng=rng,
            mutate_topk=mutate_topk,
            mutate_explore_frac=mutate_explore_frac,
            protect_elites=protect_elites,
        )


@torch.no_grad()
def cell_crossover_mutate_gpu(
    pop: torch.Tensor,
    cells: list[dict[str, Any]],
    evolve: torch.Tensor,
    s_ok: torch.Tensor,
    cell_g: torch.Tensor,
    band_lo: torch.Tensor,
    band_hi: torch.Tensor,
    t_below: torch.Tensor,
    t_above: torch.Tensor,
    nutrient_names: list[str],
    *,
    crossover_rate: float,
    mutation_rate: float,
    max_foods_per_cell: int,
    elite_size: int = 0,
    responsibility: Optional[dict[str, tuple[int, ...]]] = None,
    rng: Optional[random.Random] = None,
    mutate_topk: int = 5,
    mutate_explore_frac: float = 0.25,
    protect_elites: bool = True,
) -> None:
    """
    On-device cell operators (same drop/add/replace + top-k/explore semantics).

    Loops over ≤36 cells; vectorizes across the population on GPU.
    """
    rng = rng or random.Random()
    device = pop.device
    ensure_cell_tensors_on_device(cells, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng.randint(0, 2**31 - 1)))

    topk = max(1, int(mutate_topk))
    explore = float(np.clip(mutate_explore_frac, 0.0, 1.0))
    max_k = int(max_foods_per_cell)
    resp = responsibility or MAJOR_NUTRIENT_RESPONSIBILITY
    name_to_u = {n: i for i, n in enumerate(nutrient_names)}
    maj_to_nuts: dict[int, list[int]] = {m: [] for m in MAJOR_CODES}
    for nut, majors in resp.items():
        u = name_to_u.get(nut)
        if u is None:
            continue
        for m in majors:
            if int(m) in maj_to_nuts:
                maj_to_nuts[int(m)].append(u)

    p = pop.size(0)
    n_pairs = p // 2
    n_elite = int(elite_size) if protect_elites else 0
    start_pair = min((n_elite + 1) // 2, n_pairs)
    mut_gate2 = min(1.0, float(mutation_rate) * 2.0)

    # --- crossover (per cell, batched pairs) ---
    if n_pairs > start_pair and crossover_rate > 0:
        n_do = n_pairs - start_pair
        pair_idx = torch.arange(start_pair, n_pairs, device=device)
        ia = pair_idx * 2
        ib = ia + 1
        do_pair = torch.rand(n_do, device=device, generator=gen) < float(crossover_rate)
        for cell in cells:
            start, end = int(cell["start"]), int(cell["end"])
            if end <= start:
                continue
            mi, ji = int(cell["mi"]), int(cell["ji"])
            ev = evolve[:, mi, ji]
            active = do_pair & (ev[ia] | ev[ib])
            if not bool(active.any().item()):
                continue
            sa = pop[ia, start:end]
            sb = pop[ib, start:end]
            shared = sa & sb
            xor = sa ^ sb
            flip = torch.rand(sa.shape, device=device, generator=gen) < 0.5
            ca = shared | (xor & flip)
            cb = shared | (xor & ~flip)
            ca = _cap_true_mask(ca, max_k, gen)
            cb = _cap_true_mask(cb, max_k, gen)
            # write only active pairs
            if bool(active.all().item()):
                pop[ia, start:end] = ca
                pop[ib, start:end] = cb
            else:
                pop[ia[active], start:end] = ca[active]
                pop[ib[active], start:end] = cb[active]

    # --- mutate (per cell, batched individuals) ---
    if mutation_rate > 0 or mut_gate2 > 0:
        r1 = torch.rand(p, device=device, generator=gen)
        r2 = torch.rand(p, device=device, generator=gen)
        mut_gate = (r1 < float(mutation_rate)) | (r2 < mut_gate2)
        if n_elite > 0:
            mut_gate[:n_elite] = False
        for cell in cells:
            start, end = int(cell["start"]), int(cell["end"])
            if end <= start:
                continue
            mi, ji, maj = int(cell["mi"]), int(cell["ji"]), int(cell["maj"])
            c = end - start
            grams = cell["grams_t"]
            nut = cell["nut_t"]
            active = mut_gate & evolve[:, mi, ji]
            if not bool(active.any().item()):
                continue

            g_cell = cell_g[:, mi, ji]
            hi = band_hi[mi, ji]
            lo = band_lo[mi, ji]
            struct_fail = ~s_ok[:, mi, ji]
            # mode: 0=replace, 1=drop, 2=add
            mode = torch.zeros(p, dtype=torch.long, device=device)
            mode = torch.where(struct_fail & (g_cell > hi), torch.ones_like(mode), mode)
            mode = torch.where(
                (mode == 0) & struct_fail & (lo > 0) & (g_cell < lo),
                torch.full_like(mode, 2),
                mode,
            )
            focus_u = torch.full((p,), -1, dtype=torch.long, device=device)
            for u in maj_to_nuts.get(maj, ()):
                can = mode == 0
                add_m = can & t_below[:, u]
                mode = torch.where(add_m, torch.full_like(mode, 2), mode)
                focus_u = torch.where(add_m, torch.full_like(focus_u, int(u)), focus_u)
                can = mode == 0
                drop_m = can & t_above[:, u]
                mode = torch.where(drop_m, torch.ones_like(mode), mode)
                focus_u = torch.where(drop_m, torch.full_like(focus_u, int(u)), focus_u)

            slice_on = pop[:, start:end]
            # base scores = grams; nutrient-weighted when focus_u >= 0
            score = grams.unsqueeze(0).expand(p, -1).contiguous()
            has_f = focus_u >= 0
            if bool(has_f.any().item()) and nut.numel() > 0 and nut.size(1) > 0:
                # nut: [C,U] → score_n[p,c] = nut[c, focus_u[p]] * grams[c]
                fu = focus_u.clamp(min=0)
                score_n = nut.T.index_select(0, fu) * grams.unsqueeze(0)
                score = torch.where(has_f.unsqueeze(1), score_n, score)

            # DROP
            drop_m = active & (mode == 1) & slice_on.any(dim=1)
            if bool(drop_m.any().item()):
                idx = _gpu_pick_index(
                    score, slice_on, topk=topk, explore=explore, gen=gen
                )
                rows = torch.arange(p, device=device)[drop_m]
                pop[rows, start + idx[drop_m]] = False

            # refresh after drops
            slice_on = pop[:, start:end]
            cnt = slice_on.sum(dim=1)
            avail = ~slice_on

            # ADD
            add_m = active & (mode == 2) & avail.any(dim=1) & (cnt < max_k)
            if bool(add_m.any().item()):
                idx = _gpu_pick_index(
                    score, avail, topk=topk, explore=explore, gen=gen
                )
                rows = torch.arange(p, device=device)[add_m]
                pop[rows, start + idx[add_m]] = True

            # REPLACE (drop one + add one)
            slice_on = pop[:, start:end]
            avail = ~slice_on
            rep_m = active & (mode == 0) & avail.any(dim=1)
            if bool(rep_m.any().item()):
                has_on = slice_on.any(dim=1)
                # drop if any on
                drop_rep = rep_m & has_on
                if bool(drop_rep.any().item()):
                    idx_d = _gpu_pick_index(
                        score, slice_on, topk=topk, explore=explore, gen=gen
                    )
                    rows = torch.arange(p, device=device)[drop_rep]
                    pop[rows, start + idx_d[drop_rep]] = False
                slice_on = pop[:, start:end]
                avail = ~slice_on
                add_rep = rep_m & avail.any(dim=1) & (slice_on.sum(dim=1) < max_k)
                if bool(add_rep.any().item()):
                    # replace scoring prefers grams (CPU used grams-only for replace add)
                    score_g = grams.unsqueeze(0).expand(p, -1)
                    idx_a = _gpu_pick_index(
                        score_g, avail, topk=topk, explore=explore, gen=gen
                    )
                    rows = torch.arange(p, device=device)[add_rep]
                    pop[rows, start + idx_a[add_rep]] = True

            # final cap
            slice_on = pop[:, start:end]
            capped = _cap_true_mask(slice_on, max_k, gen)
            pop[:, start:end] = capped


def _cell_crossover_mutate_cpu(
    pop: torch.Tensor,
    cells: list[dict[str, Any]],
    evolve: torch.Tensor,
    s_ok: torch.Tensor,
    cell_g: torch.Tensor,
    band_lo: torch.Tensor,
    band_hi: torch.Tensor,
    t_below: torch.Tensor,
    t_above: torch.Tensor,
    nutrient_names: list[str],
    *,
    crossover_rate: float,
    mutation_rate: float,
    max_foods_per_cell: int,
    elite_size: int = 0,
    responsibility: Optional[dict[str, tuple[int, ...]]] = None,
    rng: Optional[random.Random] = None,
    mutate_topk: int = 5,
    mutate_explore_frac: float = 0.25,
    protect_elites: bool = True,
) -> None:
    """CPU numpy fallback (device=cpu / ablation)."""
    rng = rng or random.Random()
    topk = max(1, int(mutate_topk))
    explore = float(np.clip(mutate_explore_frac, 0.0, 1.0))
    resp = responsibility or MAJOR_NUTRIENT_RESPONSIBILITY
    name_to_u = {n: i for i, n in enumerate(nutrient_names)}
    maj_to_nuts: dict[int, list[int]] = {m: [] for m in MAJOR_CODES}
    for nut, majors in resp.items():
        u = name_to_u.get(nut)
        if u is None:
            continue
        for m in majors:
            if int(m) in maj_to_nuts:
                maj_to_nuts[int(m)].append(u)

    pop_np = pop.detach().cpu().numpy()
    evolve_np = evolve.detach().cpu().numpy()
    s_ok_np = s_ok.detach().cpu().numpy()
    cell_g_np = cell_g.detach().cpu().numpy()
    band_lo_np = band_lo.detach().cpu().numpy()
    band_hi_np = band_hi.detach().cpu().numpy()
    t_below_np = t_below.detach().cpu().numpy()
    t_above_np = t_above.detach().cpu().numpy()

    p = pop_np.shape[0]
    n_pairs = p // 2
    n_elite = int(elite_size) if protect_elites else 0
    start_pair = min((n_elite + 1) // 2, n_pairs)

    for pi in range(start_pair, n_pairs):
        if rng.random() >= crossover_rate:
            continue
        ia, ib = pi * 2, pi * 2 + 1
        for cell in cells:
            if cell["end"] <= cell["start"]:
                continue
            mi, ji = cell["mi"], cell["ji"]
            if not (evolve_np[ia, mi, ji] or evolve_np[ib, mi, ji]):
                continue
            sa = _selected_fids_np(pop_np[ia], cell)
            sb = _selected_fids_np(pop_np[ib], cell)
            ca, cb = uniform_mix_food_ids(sa, sb, rng)
            _write_fids_np(pop_np[ia], cell, ca, max_foods_per_cell, rng)
            _write_fids_np(pop_np[ib], cell, cb, max_foods_per_cell, rng)

    for i in range(n_elite, p):
        if not bool(evolve_np[i].any()):
            continue
        if rng.random() >= mutation_rate and rng.random() >= min(1.0, mutation_rate * 2):
            continue
        for cell in cells:
            if cell["end"] <= cell["start"]:
                continue
            mi, ji, maj = cell["mi"], cell["ji"], cell["maj"]
            if not evolve_np[i, mi, ji]:
                continue
            start, end = cell["start"], cell["end"]
            cur = _selected_fids_np(pop_np[i], cell)
            g_cell = float(cell_g_np[i, mi, ji])
            hi = float(band_hi_np[mi, ji])
            lo = float(band_lo_np[mi, ji])
            struct_fail = not bool(s_ok_np[i, mi, ji])

            mode = "replace"
            focus_u: Optional[int] = None
            if struct_fail and g_cell > hi:
                mode = "drop"
            elif struct_fail and lo > 0 and g_cell < lo:
                mode = "add"
            else:
                for u in maj_to_nuts.get(maj, ()):
                    if t_below_np[i, u]:
                        mode = "add"
                        focus_u = u
                        break
                    if t_above_np[i, u]:
                        mode = "drop"
                        focus_u = u
                        break

            avail = set(cell["fid_to_row"].keys()) - cur
            new_set = set(cur)
            on = np.flatnonzero(pop_np[i, start:end])

            if mode == "drop" and on.size:
                scored = []
                for loc in on:
                    gg = float(cell["grams"][loc])
                    score = (
                        float(cell["nut"][loc, focus_u]) * gg
                        if focus_u is not None
                        else gg
                    )
                    scored.append((int(loc), score))
                loc = int(_pick_topk(scored, rng, topk, explore_frac=explore))
                new_set.discard(int(cell["fids"][loc]))
            elif mode == "add" and avail and len(new_set) < max_foods_per_cell:
                scored = []
                for fid in avail:
                    row = cell["fid_to_row"][fid]
                    loc = row - start
                    gg = float(cell["grams"][loc])
                    score = (
                        float(cell["nut"][loc, focus_u]) * gg
                        if focus_u is not None
                        else gg
                    )
                    scored.append((fid, score))
                new_set.add(_pick_topk(scored, rng, topk, explore_frac=explore))
            elif avail:
                if new_set:
                    new_set.discard(rng.choice(tuple(new_set)))
                scored = [
                    (fid, float(cell["grams"][cell["fid_to_row"][fid] - start]))
                    for fid in avail
                ]
                new_set.add(_pick_topk(scored, rng, topk, explore_frac=explore))

            _write_fids_np(pop_np[i], cell, new_set, max_foods_per_cell, rng)

    pop.copy_(torch.from_numpy(pop_np).to(device=pop.device, dtype=torch.bool))


def build_dedupe_fid_column_groups(
    food_ids: np.ndarray | torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    """Column-index groups for food_ids that appear more than once (day-level dedupe)."""
    if isinstance(food_ids, torch.Tensor):
        fids = food_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        fids = np.asarray(food_ids, dtype=np.int64)
    from collections import defaultdict

    buckets: dict[int, list[int]] = defaultdict(list)
    for j, fid in enumerate(fids):
        if int(fid) >= 0:
            buckets[int(fid)].append(j)
    groups = []
    for cols in buckets.values():
        if len(cols) > 1:
            groups.append(torch.tensor(cols, dtype=torch.long, device=device))
    return groups


@torch.no_grad()
def dedupe_pop_food_ids_gpu(
    pop: torch.Tensor,
    grams: torch.Tensor,
    fid_groups: list[torch.Tensor],
) -> None:
    """
    In-place: at most one selected column per duplicated food_id (keep max grams).
    ``fid_groups`` from ``build_dedupe_fid_column_groups`` (only len>1 groups).
    """
    if not fid_groups:
        return
    p = pop.size(0)
    rows = torch.arange(p, device=pop.device)
    for cols in fid_groups:
        sub = pop[:, cols]
        if not bool(sub.any().item()):
            continue
        g = grams[cols]
        scores = torch.where(
            sub, g.unsqueeze(0).expand_as(sub), torch.full_like(sub, -1.0, dtype=g.dtype)
        )
        best = scores.argmax(dim=1)
        any_on = sub.any(dim=1)
        pop[:, cols] = False
        pop[rows, cols[best]] = any_on


# Legacy names used by unit tests / older call sites
def cell_crossover_population(
    pop: torch.Tensor,
    food_ids_tensor: torch.Tensor,
    cell_ranges: dict[str, dict[int, tuple[int, int]]],
    evolve: torch.Tensor,
    *,
    crossover_rate: float,
    max_foods_per_cell: int,
    elite_size: int = 0,
    rng: Optional[random.Random] = None,
) -> None:
    rng = rng or random.Random()
    food_ids_np = food_ids_tensor.detach().cpu().numpy()
    grams_np = np.ones(food_ids_np.shape[0], dtype=np.float32)
    nut_np = np.zeros((food_ids_np.shape[0], 1), dtype=np.float32)
    cells = build_cell_index_cache(cell_ranges, food_ids_np, grams_np, nut_np)
    # dummy tensors for mutate skip path — only run crossover via fast path
    p = pop.size(0)
    device = pop.device
    zeros_s = torch.ones(p, 4, 9, dtype=torch.bool, device=device)
    zeros_g = torch.zeros(p, 4, 9, device=device)
    zeros_b = torch.zeros(4, 9, device=device)
    zeros_t = torch.zeros(p, 1, dtype=torch.bool, device=device)
    cell_crossover_mutate_fast(
        pop,
        cells,
        evolve,
        zeros_s,
        zeros_g,
        zeros_b,
        zeros_b + 1e9,
        zeros_t,
        zeros_t,
        ["_"],
        crossover_rate=crossover_rate,
        mutation_rate=0.0,  # crossover only
        max_foods_per_cell=max_foods_per_cell,
        elite_size=elite_size,
        rng=rng,
    )


def cell_mutate_population(*_args, **_kwargs) -> None:
    """Deprecated: use cell_crossover_mutate_fast."""
    raise RuntimeError("cell_mutate_population deprecated; use cell_crossover_mutate_fast")
