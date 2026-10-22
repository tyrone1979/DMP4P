"""
Shared Generic / Nutrient-GA operators that are NOT V2-specific.

Allowed: cardinality clamp to [min_foods, max_foods], food_id dedupe (keep first).
Forbidden: structure priors, nutrient-responsibility maps, cell masks, V2 mutate.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def init_population_in_range(
    p: int,
    n_rows: int,
    allowed_idx: np.ndarray,
    *,
    min_foods: int,
    max_foods: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pop = np.zeros((p, n_rows), dtype=bool)
    if len(allowed_idx) == 0:
        return pop
    lo = max(1, int(min_foods))
    hi = max(lo, int(max_foods))
    for i in range(p):
        k = int(rng.integers(lo, hi + 1))
        k = min(k, len(allowed_idx))
        chosen = rng.choice(allowed_idx, size=k, replace=False)
        pop[i, chosen] = True
    return pop


def repair_cardinality(
    mask: np.ndarray,
    allowed_idx: np.ndarray,
    *,
    min_foods: int,
    max_foods: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    """Generic add/drop to bring selected count into [min, max]."""
    out = mask.copy()
    stats = {"card_drops": 0, "card_adds": 0}
    on = np.flatnonzero(out)
    if len(on) > max_foods:
        drop = rng.choice(on, size=len(on) - max_foods, replace=False)
        out[drop] = False
        stats["card_drops"] = int(len(drop))
        on = np.flatnonzero(out)
    if len(on) < min_foods:
        off = np.array([i for i in allowed_idx if not out[i]], dtype=int)
        need = min_foods - len(on)
        if len(off) and need > 0:
            add = rng.choice(off, size=min(need, len(off)), replace=False)
            out[add] = True
            stats["card_adds"] = int(len(add))
    return out, stats


def repair_food_id_dedupe(
    mask: np.ndarray,
    food_ids: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Keep one random row per food_id among selected; clear the rest."""
    out = mask.copy()
    on = np.flatnonzero(out)
    if len(on) == 0:
        return out, 0
    fids = food_ids[on]
    dropped = 0
    for fid in np.unique(fids):
        rows = on[fids == fid]
        if len(rows) <= 1:
            continue
        keep = int(rng.choice(rows))
        for r in rows:
            if r != keep:
                out[r] = False
                dropped += 1
    return out, dropped


def apply_shared_repairs(
    pop: np.ndarray,
    *,
    food_ids: np.ndarray,
    allowed_mask: np.ndarray,
    min_foods: int,
    max_foods: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply disallowed-row mask, food_id dedupe, then cardinality repair to each individual.
    """
    allowed_idx = np.flatnonzero(allowed_mask)
    totals = {"card_drops": 0, "card_adds": 0, "dup_drops": 0, "disallowed_cleared": 0}
    out = pop.copy()
    for i in range(len(out)):
        before = int(out[i].sum())
        out[i] = out[i] & allowed_mask
        totals["disallowed_cleared"] += before - int(out[i].sum())
        out[i], n_dup = repair_food_id_dedupe(out[i], food_ids, rng=rng)
        totals["dup_drops"] += n_dup
        out[i], st = repair_cardinality(
            out[i], allowed_idx, min_foods=min_foods, max_foods=max_foods, rng=rng
        )
        # dedupe again if adds reintroduced same food_id
        out[i], n_dup2 = repair_food_id_dedupe(out[i], food_ids, rng=rng)
        totals["dup_drops"] += n_dup2
        if int(out[i].sum()) > max_foods or int(out[i].sum()) < min_foods:
            out[i], st2 = repair_cardinality(
                out[i], allowed_idx, min_foods=min_foods, max_foods=max_foods, rng=rng
            )
            totals["card_drops"] += st2["card_drops"]
            totals["card_adds"] += st2["card_adds"]
        totals["card_drops"] += st["card_drops"]
        totals["card_adds"] += st["card_adds"]
    return out, totals


def population_diagnostics(
    pop: np.ndarray,
    *,
    food_ids: np.ndarray,
    nut: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    grams: np.ndarray,
    max_daily_g: float,
    min_foods: int,
    max_foods: int,
) -> dict[str, Any]:
    """Cheap per-generation diagnostics (no full evaluator)."""
    p = len(pop)
    n_sel = pop.sum(axis=1).astype(float)
    dup_frac = []
    nut_ok = 0
    card_ok = 0
    wt_ok = 0
    for i in range(p):
        on = np.flatnonzero(pop[i])
        if len(on) == 0:
            dup_frac.append(0.0)
            continue
        f = food_ids[on]
        dup_frac.append(1.0 - (len(np.unique(f)) / len(f)))
        if min_foods <= len(np.unique(f)) <= max_foods:
            card_ok += 1
        totals = nut[on].sum(axis=0)
        if np.all(totals >= lo - 1e-9) and np.all(totals <= hi + 1e-9):
            nut_ok += 1
        if float(grams[on].sum()) <= max_daily_g + 1e-9:
            wt_ok += 1
    return {
        "n_sel_mean": float(np.mean(n_sel)),
        "n_sel_min": float(np.min(n_sel)),
        "n_sel_max": float(np.max(n_sel)),
        "dup_food_frac_mean": float(np.mean(dup_frac)) if dup_frac else 0.0,
        "pass_card_frac": card_ok / max(p, 1),
        "pass_nutrient_frac": nut_ok / max(p, 1),
        "pass_weight_frac": wt_ok / max(p, 1),
    }
