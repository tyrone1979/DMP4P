"""
GPU Genetic Algorithm v2 for multi-objective daily meal composition.

Layer A (hard): nutrient ranges in [L,U] + B/L/D minimum meal grams + cardinality
+ condiment/oil/sweet-beverage weight-share (same thresholds as Silver)
+ learned day-structure envelope (Full/Golden prior; critical axes).
Layer B (soft, among feasible): positive-negative, food-group diversity, meal diversity,
structure_fit to NHANES priors, uncommon-food penalty, learned iforest bonus.
Archive: unique frozenset(food_id) fingerprints; keep best Layer-B score; up to N solutions.

Reuses selection/crossover/mutate from ga_gpu.GeneticAlgorithm.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from algo.diet_structure_priors import (
    check_bld_hard,
    check_condiment_oil_hard,
    daily_weight_hard_max,
    day_structure_metrics,
    food_id_fingerprint,
    fingerprint_hash,
    get_common_uncommon_sets,
    layer_b_score,
    load_ga_config,
    load_structure_prior,
)
from ga_gpu import GeneticAlgorithm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class FoodCombinerV2:
    """Food combination optimizer with structure + diversity + anti-repetition archive."""

    # feature layout:
    # 0 positive, 1 negative, 2 eating_type, 3 grams, 4 meal_id, 5 food_group, 6 is_common,
    # 7: nutrients...
    META_DIM = 7

    def __init__(
        self,
        food_df: pd.DataFrame,
        target_dict: dict,
        max_combination_size: int = 30,
        population_size: int = 300,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.01,
        device: str = "cuda",
        debug: bool = False,
        structure_prior: Optional[dict] = None,
        config: Optional[dict] = None,
        n_solutions: int = 7,
        archive_target: Optional[int] = None,
        require_bld: bool = True,
        participant_profile: Optional[dict] = None,
        personal_category_structure: Optional[dict] = None,
        prebuilt_cell_ranges: Optional[dict] = None,
        prebuilt_forbidden_cells: Optional[set] = None,
    ):
        """
        Parameters
        ----------
        prebuilt_cell_ranges / prebuilt_forbidden_cells :
            When both pool and ranges come from a benchmark snapshot, skip
            ``build_structure_cell_pool`` so all algorithms share the same rows.
        """
        self.cfg = config or load_ga_config()
        # Benchmark: never re-sample/quota-truncate the provided pool
        self.benchmark_freeze_pool = bool(self.cfg.get("benchmark_freeze_pool", False))
        # Benchmark-only ablations (never set by production YAML)
        self.benchmark_zero_cell_soft = bool(
            (self.cfg.get("structure") or {}).get("cell_ga", {}).get("benchmark_zero_cell_soft", False)
        )
        self.benchmark_uniform_archive = bool(self.cfg.get("benchmark_uniform_archive", False))
        self.benchmark_disable_nutrient_guidance = bool(
            self.cfg.get("benchmark_disable_nutrient_guidance", False)
        )
        self.benchmark_disable_freeze = bool(
            (self.cfg.get("structure") or {}).get("cell_ga", {}).get("benchmark_disable_freeze", False)
        )
        self.benchmark_ignore_structure_evolve = bool(
            (self.cfg.get("structure") or {}).get("cell_ga", {}).get(
                "benchmark_ignore_structure_evolve", False
            )
        )
        self.benchmark_quality_continuation = bool(
            self.cfg.get("benchmark_quality_continuation", False)
        )
        self.food_df = food_df.reset_index(drop=True)
        self.target_dict = target_dict
        self.nutrient_columns = list(target_dict.keys())
        self.max_combination_size = max_combination_size or int(self.cfg.get("max_combination_size", 15))
        self.min_foods = int(self.cfg.get("min_foods", 5))
        self.device = device
        self.debug = debug
        self.require_bld = require_bld
        self.n_solutions = n_solutions or int(self.cfg.get("n_solutions_per_user", 7))
        self.archive_target = archive_target or int(self.cfg.get("archive_target", 7))
        self.structure_prior = structure_prior or load_structure_prior(config=self.cfg)
        self.participant_profile = dict(participant_profile or {})
        self._common, self._uncommon = get_common_uncommon_sets(self.cfg)
        st = self.cfg.get("structure") or {}
        self.hard_daily_weight = bool(st.get("hard_daily_weight", True))
        self.max_daily_g = daily_weight_hard_max(self.structure_prior, self.cfg)
        # Learned structure prior (Full by default) — archive Hard on critical axes
        self.learned_structure = None
        self.category_prior = None
        self.personal_category_structure = None
        # Gram bands Hard off by default (Bronze allows OOR; Silver may repair).
        # Presence Hard: hi≤0 meal×major must be empty.
        self.category_grams_hard = bool(st.get("category_grams_hard", False))
        self.category_presence_hard = bool(st.get("category_presence_hard", True))
        self.learned_structure_hard = bool(st.get("learned_structure_hard", True))
        self.learned_require_iforest = bool(st.get("learned_require_iforest", False))
        self.w_learned_iforest = float((self.cfg.get("weights") or {}).get("w_learned_iforest", 2.0))
        # Prefer Bronze-preselected personal structure; else resolve once from demo profile
        try:
            from algo.structure_category import (
                format_personal_structure_brief,
                get_active_category_prior,
                select_personal_category_for_bronze,
            )

            self.category_prior = get_active_category_prior((self.cfg.get("paths") or {}).get("structure_learn_config", ""))
            if personal_category_structure is not None:
                self.personal_category_structure = personal_category_structure
            elif self.category_prior is not None:
                # Resolve even with empty demo → overall envelope (needed for cell_ga bands)
                self.personal_category_structure = select_personal_category_for_bronze(
                    self.participant_profile or {},
                    self.target_dict,
                    model=self.category_prior,
                )
            use_cat = self.category_grams_hard or self.category_presence_hard
            if self.personal_category_structure is not None and use_cat:
                self.learned_structure_hard = False
                print(format_personal_structure_brief(self.personal_category_structure), flush=True)
            elif self.category_prior is not None and use_cat:
                self.learned_structure_hard = False
                print(
                    f"category structure prior (no personal resolve): "
                    f"path={self.category_prior.get('_active_path')} "
                    f"bmi={self.participant_profile.get('bmi')}",
                    flush=True,
                )
        except Exception as e:
            print(f"category-gram prior unavailable: {e}", flush=True)
            self.category_prior = None
            self.personal_category_structure = None
        if self.learned_structure_hard or self.w_learned_iforest:
            try:
                from algo.structure_learn import get_active_structure_prior

                self.learned_structure = get_active_structure_prior((self.cfg.get("paths") or {}).get("structure_learn_config", ""))
                if self.learned_structure is not None:
                    corp = self.learned_structure.get("_active_corpus", "?")
                    print(
                        f"learned structure prior: corpus={corp} "
                        f"hard={self.learned_structure_hard} "
                        f"iforest_hard={self.learned_require_iforest} "
                        f"bmi={self.participant_profile.get('bmi')} "
                        f"total_g_ref={self.participant_profile.get('total_g_ref')}",
                        flush=True,
                    )
            except Exception as e:
                print(f"learned structure prior unavailable: {e}", flush=True)
                self.learned_structure = None

        # cell_ga + type presence (hi≤0 forbidden). Gram bands Soft / Silver repair.
        st_cfg = dict(self.cfg.get("structure") or {})
        cg = dict(st_cfg.get("cell_ga") or {})
        if self.personal_category_structure is not None and bool(cg.get("enabled", False)):
            self.learned_structure_hard = False
            cg["enabled"] = True
            st_cfg["cell_ga"] = cg
            self.cfg = dict(self.cfg)
            self.cfg["structure"] = st_cfg
        self.cell_ga_enabled = bool(cg.get("enabled", False)) and (
            self.personal_category_structure is not None
        )
        self.cell_ga_cfg = dict(cg)
        self.cell_ranges: dict[str, dict[int, tuple[int, int]]] = {}
        self._cell_band_lo = None
        self._cell_band_hi = None
        self._cell_upper_only: set[int] = set()
        self._forbidden_cells: set[tuple[str, int]] = set()
        if self.cell_ga_enabled and prebuilt_cell_ranges is not None:
            # Benchmark / snapshot path: pool already ordered; do not re-sample.
            self.cell_ranges = {
                m: {int(k): (int(a), int(b)) for k, (a, b) in cells.items()}
                for m, cells in prebuilt_cell_ranges.items()
            }
            self._forbidden_cells = set(prebuilt_forbidden_cells or [])
            n_cells = sum(
                1 for m in self.cell_ranges for _mi, (a, b) in self.cell_ranges[m].items() if b > a
            )
            print(
                f"cell_ga: prebuilt pool={len(self.food_df)} nonempty={n_cells}/36 "
                f"(snapshot reuse)",
                flush=True,
            )
        elif self.cell_ga_enabled and self.benchmark_freeze_pool:
            from algo.structure_cell_pool import rebuild_cell_ranges_preserve_rows

            ordered, ranges, meta = rebuild_cell_ranges_preserve_rows(
                self.food_df,
                personal=self.personal_category_structure,
            )
            self._forbidden_cells = set(meta.get("forbidden_cells") or [])
            if len(ordered) == 0:
                print("cell_ga: empty freeze pool → falling back to flat pool", flush=True)
                self.cell_ga_enabled = False
            else:
                self.food_df = ordered
                self.cell_ranges = ranges
                n_cells = int(meta.get("n_nonempty_cells") or 0)
                print(
                    f"cell_ga: freeze-preserve pool={len(ordered)} nonempty={n_cells}/36 "
                    f"(no quota sample)",
                    flush=True,
                )
        elif self.cell_ga_enabled:
            from algo.structure_cell_pool import build_structure_cell_pool

            quota = int(cg.get("per_cell_quota", 80))
            ordered, ranges, meta = build_structure_cell_pool(
                self.food_df,
                personal=self.personal_category_structure,
                per_cell_quota=quota,
            )
            self._forbidden_cells = set(meta.get("forbidden_cells") or [])
            if len(ordered) == 0:
                print("cell_ga: empty cell pool → falling back to flat pool", flush=True)
                self.cell_ga_enabled = False
            else:
                self.food_df = ordered
                self.cell_ranges = ranges
                n_cells = int(meta.get("n_nonempty_cells") or 0)
                n_drop = int(meta.get("n_dropped_forbidden") or 0)
                forb = meta.get("forbidden_cells") or []
                forb_s = ",".join(f"{m[:1]}{mi}" for m, mi in forb[:12])
                suffix = f" ({forb_s})" if forb_s else ""
                print(
                    f"cell_ga: pool={len(ordered)} nonempty={n_cells}/36 "
                    f"quota={quota} hi0_locked={len(forb)} dropped={n_drop}{suffix}",
                    flush=True,
                )
        elif self.personal_category_structure is not None:
            # Even without cell_ga ops: never let hi=0 meal×major rows into the pool
            from algo.structure_cell_pool import drop_forbidden_structure_cells

            kept, n_drop, forb = drop_forbidden_structure_cells(
                self.food_df, self.personal_category_structure
            )
            self._forbidden_cells = forb
            if n_drop:
                self.food_df = kept
                print(
                    f"structure hi0 lock: dropped={n_drop} rows cells={len(forb)}",
                    flush=True,
                )

        self._prepare_features()
        self.num_foods = len(self.food_df)
        if self.num_foods == 0:
            raise ValueError("Dataset is empty. No foods found for the given target.")

        self.ga = GeneticAlgorithm(
            population_size=population_size,
            chromosome_length=self.num_foods,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            device=self.device,
            debug=debug,
        )
        self.target_min = torch.tensor([v[0] for v in target_dict.values()], dtype=torch.float32, device=device)
        self.target_max = torch.tensor([v[1] for v in target_dict.values()], dtype=torch.float32, device=device)
        mins = self.cfg["min_meal_weight_g"]
        self.min_b = float(mins.get("breakfast", 30))
        self.min_l = float(mins.get("lunch", 50))
        self.min_d = float(mins.get("dinner", 50))

        self._cell_cache = None
        if self.cell_ga_enabled:
            from algo.structure_cell_ops import (
                build_band_tensors,
                build_cell_index_cache,
                ensure_cell_tensors_on_device,
            )

            self._cell_band_lo, self._cell_band_hi, self._cell_upper_only = build_band_tensors(
                self.personal_category_structure, torch.device(self.device)
            )
            food_ids_np = self.food_ids_tensor.detach().cpu().numpy()
            grams_np = self.features[:, 3].detach().cpu().numpy()
            nut_np = self.features[:, self.META_DIM :].detach().cpu().numpy()
            self._cell_cache = build_cell_index_cache(
                self.cell_ranges, food_ids_np, grams_np, nut_np
            )
            ensure_cell_tensors_on_device(self._cell_cache, torch.device(self.device))

        # fingerprint_hash -> {score, indices_np, meta}
        self.archive: dict[str, dict[str, Any]] = {}
        # Population is seeded in optimize / optimize_adaptive (avoids double greedy cost)

    def _indices_tensor(self, positions: list[int]) -> torch.Tensor:
        pos = [int(p) for p in positions if 0 <= int(p) < self.num_foods][: self.max_combination_size]
        if not pos:
            return torch.full((1,), -1, dtype=torch.long, device=self.device)
        return torch.tensor(pos, dtype=torch.long, device=self.device)

    def _pop_rng(self) -> tuple[np.random.Generator, int]:
        """
        RNG for population seeding.

        - ``ga_seed`` set: reproducible ``ga_seed + pop_seed_offset``
        - else: fresh entropy each call (sequential sols must not share one seed)
        """
        offset = int(self.cfg.get("pop_seed_offset", 0) or 0)
        raw = self.cfg.get("ga_seed", None)
        if raw is not None and str(raw).strip() != "":
            seed = int(raw) + offset
        else:
            seed = secrets.randbits(31)
        return np.random.default_rng(seed), int(seed)

    def _seed_bld_population(self) -> None:
        """V1-style random seed with B/L/D present + nut>0 coverage (no greedy packer)."""
        if self.cell_ga_enabled and self.cell_ranges:
            self._seed_cell_population()
            return
        meal = self.food_df["meal_id"].to_numpy()
        b_idx = np.where(meal == 1)[0]
        l_idx = np.where(meal == 2)[0]
        d_idx = np.where(meal == 3)[0]
        s_idx = np.where(meal == 4)[0]
        if len(b_idx) == 0 or len(l_idx) == 0 or len(d_idx) == 0:
            print(
                f"WARNING: pool missing meal(s) B={len(b_idx)} L={len(l_idx)} D={len(d_idx)} "
                f"size={self.num_foods}",
                flush=True,
            )

        fd = self.food_df
        rng, seed_used = self._pop_rng()
        # Coverage only: foods with amount > 0 for nutrients that have lo > 0 (no density top-k)
        boost: list[int] = []
        for col, bounds in self.target_dict.items():
            if col not in fd.columns:
                continue
            try:
                lo_f = float(bounds[0])
            except (TypeError, ValueError, IndexError):
                continue
            if lo_f <= 0:
                continue
            amt = pd.to_numeric(fd[col], errors="coerce").fillna(0.0)
            pos = np.flatnonzero(amt.to_numpy() > 0)
            if len(pos) == 0:
                continue
            # random subset for diversity (not densest bran/cheat foods)
            take = min(40, len(pos))
            boost.extend(pos[rng.choice(len(pos), size=take, replace=False)].tolist())
        boost_arr = np.asarray(sorted(set(int(i) for i in boost)), dtype=np.int64)

        # BP-like: inject low-Na / high-K rows into every seed chromosome
        nak_idx = np.asarray([], dtype=np.int64)
        nak_cfg = dict((self.cfg.get("pool") or {}).get("low_na_high_k") or {})
        if bool(nak_cfg.get("enabled", True)):
            from algo.diet_structure_priors import target_needs_low_na_high_k, low_na_high_k_mask

            if target_needs_low_na_high_k(self.target_dict):
                mask = low_na_high_k_mask(
                    fd,
                    sodium_per_100g_max=float(nak_cfg.get("sodium_per_100g_max", 140)),
                    potassium_per_100g_min=float(nak_cfg.get("potassium_per_100g_min", 200)),
                    grams_min=float(nak_cfg.get("grams_min", 20)),
                    potassium_abs_min=float(nak_cfg.get("potassium_abs_min", 150)),
                )
                nak_idx = np.asarray(np.where(mask.to_numpy())[0], dtype=np.int64)
                if len(nak_idx):
                    nak_idx = nak_idx[rng.permutation(len(nak_idx))]

        pop = self.ga.population
        n = self.num_foods
        p = self.ga.population_size

        def _meal_pick(all_idx: np.ndarray) -> Optional[int]:
            if len(all_idx) == 0:
                return None
            if len(nak_idx):
                inter = np.intersect1d(all_idx, nak_idx, assume_unique=False)
                use = inter if len(inter) else all_idx
            else:
                use = all_idx
            return int(use[rng.integers(0, len(use))])

        seed_nak = int(nak_cfg.get("seed_picks", 8)) if len(nak_idx) else 0
        max_n = int(self.max_combination_size)
        seed_target = min(
            int(self.cfg.get("weights", {}).get("w_n_foods_target", 11)),
            max_n,
        )
        pop[:] = False
        for i in range(p):
            # Build an ordered pick list (BLD first), then hard-cap to max_n.
            picks: list[int] = []

            def _add(pid: Optional[int]):
                if pid is None:
                    return
                pid = int(pid)
                if pid not in picks:
                    picks.append(pid)

            for idx_arr in (b_idx, l_idx, d_idx):
                _add(_meal_pick(idx_arr))
            if len(s_idx) and rng.random() < 0.7:
                _add(_meal_pick(s_idx))

            # NaK seeds before nut>0 coverage picks (BP targets)
            if seed_nak and len(nak_idx):
                top = nak_idx[: min(120, len(nak_idx))]
                n_pick = min(seed_nak, len(top), max(0, max_n - len(picks)))
                if n_pick > 0:
                    for pid in top[rng.choice(len(top), size=n_pick, replace=False)]:
                        _add(int(pid))
                        if len(picks) >= max_n:
                            break

            if len(boost_arr) and len(picks) < max_n:
                boost_use = (
                    np.intersect1d(boost_arr, nak_idx, assume_unique=False)
                    if len(nak_idx)
                    else boost_arr
                )
                if len(boost_use) == 0:
                    boost_use = boost_arr
                n_boost = min(8, len(boost_use), max_n - len(picks))
                if n_boost > 0:
                    for pid in boost_use[rng.choice(len(boost_use), size=n_boost, replace=False)]:
                        _add(int(pid))
                        if len(picks) >= max_n:
                            break

            # Fill toward seed_target (never above max_n)
            need = max(0, min(seed_target, max_n) - len(picks))
            if need > 0:
                if len(nak_idx):
                    fill_from = nak_idx
                    # exclude already picked
                    picked = set(picks)
                    fill_from = np.asarray([x for x in fill_from if int(x) not in picked], dtype=np.int64)
                    if len(fill_from):
                        for pid in fill_from[
                            rng.choice(len(fill_from), size=min(need, len(fill_from)), replace=False)
                        ]:
                            _add(int(pid))
                            if len(picks) >= max_n:
                                break
                else:
                    while need > 0 and len(picks) < max_n:
                        pid = int(rng.integers(0, n))
                        before = len(picks)
                        _add(pid)
                        if len(picks) > before:
                            need -= 1
                        else:
                            # avoid infinite loop on tiny pools
                            need -= 1

            picks = picks[:max_n]
            if picks:
                pop[i, torch.tensor(picks, device=self.device, dtype=torch.long)] = True
        msg = (
            f"seed random BLD+nut>0 coverage p={p} (no dens boosters) "
            f"cap≤{max_n} pop_seed={seed_used}"
        )
        if seed_nak:
            msg += f" +lowNa/highK×{seed_nak} fill=nak-only (pool={len(nak_idx)})"
        print(msg, flush=True)
        self._dedupe_pop_one_food_per_day()

    def _seed_cell_population(self) -> None:
        """
        Seed by sampling 0–2 food_ids per meal×major cell (plus B/L/D guarantee).

        Vectorized over the population (same Bernoulli / lo>0 rules as the
        previous per-individual Python loop) — much cheaper at p≈1000.
        """
        from algo.structure_features import MAJOR_CODES, MEAL_NAMES

        pop = self.ga.population
        p = self.ga.population_size
        n = self.num_foods
        max_n = int(self.max_combination_size)
        max_k = int(self.cell_ga_cfg.get("max_foods_per_cell", 3))
        rng, seed_used = self._pop_rng()
        pop[:] = False

        bands = (self.personal_category_structure or {}).get("bands") or {}
        # Precompute nonempty evolvable cells
        cells_meta: list[tuple[int, int, str, float]] = []
        for meal in MEAL_NAMES:
            for maj in MAJOR_CODES:
                start, end = self.cell_ranges.get(meal, {}).get(maj, (0, 0))
                if end <= start:
                    continue
                b = bands.get(f"g_major_{maj}_{meal}") or {}
                hi = b.get("hi")
                if hi is not None and float(hi) <= 0:
                    continue
                lo = float(b.get("lo", 0.0) or 0.0)
                cells_meta.append((start, end, meal, lo))

        # Accumulate picks as list-of-lists then cap — still far fewer Python ops
        # than scanning 36 cells × nested meal checks with iloc.
        picks_per: list[list[int]] = [[] for _ in range(p)]
        for start, end, meal, lo in cells_meta:
            span = end - start
            if meal in ("breakfast", "lunch", "dinner") and lo > 0:
                base = rng.random(p) < 0.85
            else:
                base = rng.random(p) < 0.25
            extra = rng.random(p) < 0.15
            n_pick = base.astype(np.int8) + extra.astype(np.int8)
            n_pick = np.minimum(n_pick, max_k)
            loc1 = rng.integers(0, span, size=p)
            need1 = n_pick >= 1
            for i in np.flatnonzero(need1):
                if len(picks_per[i]) < max_n:
                    picks_per[i].append(int(start + loc1[i]))
            # second pick (different loc when possible)
            need2 = n_pick >= 2
            if need2.any() and span > 1:
                loc2 = rng.integers(0, span, size=p)
                for i in np.flatnonzero(need2):
                    if len(picks_per[i]) >= max_n:
                        continue
                    if int(loc2[i]) == int(loc1[i]):
                        loc2[i] = (int(loc2[i]) + 1) % span
                    picks_per[i].append(int(start + int(loc2[i])))

        meal_ids = (
            self.food_df["meal_id"].to_numpy()
            if "meal_id" in self.food_df.columns
            else None
        )
        meal_pools = {}
        if meal_ids is not None:
            for mid in (1, 2, 3):
                meal_pools[mid] = np.flatnonzero(meal_ids == mid)

        for i in range(p):
            picks = picks_per[i]
            if meal_ids is not None:
                have = {int(meal_ids[j]) for j in picks} if picks else set()
                for mid in (1, 2, 3):
                    if mid in have:
                        continue
                    idx = meal_pools.get(mid)
                    if idx is not None and len(idx):
                        picks.append(int(idx[rng.integers(0, len(idx))]))
            picks = picks[:max_n]
            if picks:
                pop[i, torch.tensor(picks, device=self.device, dtype=torch.long)] = True

        print(
            f"seed cell-GA p={p} cap≤{max_n} max_per_cell={max_k} pop_seed={seed_used} "
            f"cells={len(cells_meta)} n={n}",
            flush=True,
        )
        self._dedupe_pop_one_food_per_day()

    def _pool_fingerprint(self, pool_df: pd.DataFrame) -> tuple[int, str]:
        fids = (
            pd.to_numeric(pool_df["food_id"], errors="coerce")
            .fillna(-1)
            .astype(np.int64)
            .to_numpy()
        )
        return (int(len(fids)), hashlib.sha1(fids.tobytes()).hexdigest()[:16])

    def rebind_pool(
        self,
        pool_df: pd.DataFrame,
        *,
        config: Optional[dict] = None,
        population_size: Optional[int] = None,
        crossover_rate: Optional[float] = None,
        mutation_rate: Optional[float] = None,
        quiet: bool = True,
        freeze_no_resample: bool = False,
    ) -> None:
        """
        Reuse personal structure / priors; rebuild tensors + GA for a new pool.

        Used by sequential N+1 ban so we do not re-resolve category structure
        or reload priors on every sol.

        ``freeze_no_resample=True`` (benchmark): only re-index remaining rows into
        cell slices — no per-cell quota sampling / truncation / new candidates.
        """
        if config is not None:
            self.cfg = dict(config)
            st = dict(self.cfg.get("structure") or {})
            cg = dict(st.get("cell_ga") or {})
            self.cell_ga_cfg = dict(cg)
            self.cell_ga_enabled = bool(cg.get("enabled", False)) and (
                self.personal_category_structure is not None
            )
        self.food_df = pool_df.reset_index(drop=True)
        if self.cell_ga_enabled and self.personal_category_structure is not None:
            if freeze_no_resample or bool(getattr(self, "benchmark_freeze_pool", False)):
                from algo.structure_cell_pool import rebuild_cell_ranges_preserve_rows

                ordered, ranges, meta = rebuild_cell_ranges_preserve_rows(
                    self.food_df,
                    personal=self.personal_category_structure,
                )
                self._forbidden_cells = set(meta.get("forbidden_cells") or [])
                if len(ordered) == 0:
                    raise ValueError("cell_ga: empty cell pool after freeze rebind")
                self.food_df = ordered
                self.cell_ranges = ranges
                if not quiet:
                    n_cells = int(meta.get("n_nonempty_cells") or 0)
                    print(
                        f"cell_ga freeze-rebind: pool={len(ordered)} nonempty={n_cells}/36 "
                        f"(no resample)",
                        flush=True,
                    )
            else:
                from algo.structure_cell_pool import build_structure_cell_pool

                quota = int(self.cell_ga_cfg.get("per_cell_quota", 80))
                ordered, ranges, meta = build_structure_cell_pool(
                    self.food_df,
                    personal=self.personal_category_structure,
                    per_cell_quota=quota,
                )
                self._forbidden_cells = set(meta.get("forbidden_cells") or [])
                if len(ordered) == 0:
                    raise ValueError("cell_ga: empty cell pool after rebind")
                self.food_df = ordered
                self.cell_ranges = ranges
                if not quiet:
                    n_cells = int(meta.get("n_nonempty_cells") or 0)
                    print(
                        f"cell_ga rebind: pool={len(ordered)} nonempty={n_cells}/36 "
                        f"quota={quota}",
                        flush=True,
                    )
        self._prepare_features()
        self.num_foods = len(self.food_df)
        if self.num_foods == 0:
            raise ValueError("rebind_pool: empty pool")
        p = int(population_size or self.ga.population_size)
        cr = float(
            crossover_rate if crossover_rate is not None else self.ga.crossover_rate
        )
        mr = float(
            mutation_rate if mutation_rate is not None else self.ga.mutation_rate
        )
        self.ga = GeneticAlgorithm(
            population_size=p,
            chromosome_length=self.num_foods,
            crossover_rate=cr,
            mutation_rate=mr,
            device=self.device,
            debug=self.debug,
        )
        if self.cell_ga_enabled:
            from algo.structure_cell_ops import (
                build_band_tensors,
                build_cell_index_cache,
                ensure_cell_tensors_on_device,
            )

            self._cell_band_lo, self._cell_band_hi, self._cell_upper_only = build_band_tensors(
                self.personal_category_structure, torch.device(self.device)
            )
            food_ids_np = self.food_ids_tensor.detach().cpu().numpy()
            grams_np = self.features[:, 3].detach().cpu().numpy()
            nut_np = self.features[:, self.META_DIM :].detach().cpu().numpy()
            self._cell_cache = build_cell_index_cache(
                self.cell_ranges, food_ids_np, grams_np, nut_np
            )
            ensure_cell_tensors_on_device(self._cell_cache, torch.device(self.device))
        self.archive = {}
        self.last_stage_log = []
        self._pool_fp = self._pool_fingerprint(self.food_df)

    def configure_search_slot(
        self,
        *,
        pop_seed_offset: int,
        adaptive: dict,
        n_solutions: int = 1,
        archive_target: int = 1,
    ) -> None:
        """Update seed offset / adaptive stages without rebuilding the pool."""
        self.cfg = dict(self.cfg)
        self.cfg["pop_seed_offset"] = int(pop_seed_offset)
        self.cfg["adaptive"] = dict(adaptive)
        st = dict(self.cfg.get("structure") or {})
        cg = dict(st.get("cell_ga") or {})
        # Keep cell_ga stall aligned with sequential full-g budget
        z = int((adaptive or {}).get("zero_archive_stall_gens", cg.get("zero_archive_stall_gens", 12)))
        cg["zero_archive_stall_gens"] = z
        st["cell_ga"] = cg
        self.cfg["structure"] = st
        self.cell_ga_cfg = dict(cg)
        self.n_solutions = int(n_solutions)
        self.archive_target = int(archive_target)
        self.archive = {}
        self.last_stage_log = []

    def _prepare_features(self) -> None:
        df = self.food_df
        for col in ("positive", "negative", "eating_type", "grams", "meal_id", "food_group_code", "is_common"):
            if col not in df.columns:
                if col == "meal_id":
                    from algo.diet_structure_priors import meal_id_from_eating_type
                    df["meal_id"] = df["eating_type"].map(meal_id_from_eating_type).fillna(0).astype(int)
                elif col == "food_group_code":
                    from algo.diet_structure_priors import food_group_code
                    df["food_group_code"] = df["food_id"].map(food_group_code).astype(int)
                elif col == "is_common":
                    df["is_common"] = df["food_id"].map(
                        lambda x: 1 if pd.notna(x) and int(float(x)) in self._common else 0
                    ).astype(int)
                else:
                    df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        for nut in self.nutrient_columns:
            if nut not in df.columns:
                df[nut] = 0.0
            df[nut] = pd.to_numeric(df[nut], errors="coerce").fillna(0.0)

        self.feature_columns = [
            "positive", "negative", "eating_type", "grams", "meal_id", "food_group_code", "is_common",
        ] + self.nutrient_columns
        self.food_df = df
        self.features = torch.tensor(df[self.feature_columns].values, dtype=torch.float32, device=self.device)
        self.food_ids_tensor = torch.tensor(
            pd.to_numeric(df["food_id"], errors="coerce").fillna(-1).astype(np.int64).values,
            dtype=torch.long,
            device=self.device,
        )
        if torch.any(torch.isnan(self.features)):
            raise ValueError("NaN values detected in self.features")

    def check_nutrient_range_batch(self, selected_foods: torch.Tensor) -> torch.Tensor:
        nutrients = selected_foods[:, :, self.META_DIM:]
        nutrient_sums = nutrients.sum(dim=1)
        below_min = torch.clamp(self.target_min - nutrient_sums, min=0)
        above_max = torch.clamp(nutrient_sums - self.target_max, min=0)
        return below_min.sum(dim=1) + above_max.sum(dim=1)

    def check_structure_hard_batch(self, selected_foods: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Penalty for missing B/L/D minimum grams + oil/condiment meal dominance. 0 => OK."""
        grams = selected_foods[:, :, 3] * mask.float()
        meal_id = selected_foods[:, :, 4]
        fgroup = selected_foods[:, :, 5]
        # meal_id 1/2/3
        b = torch.where(meal_id == 1, grams, torch.zeros_like(grams)).sum(dim=1)
        l = torch.where(meal_id == 2, grams, torch.zeros_like(grams)).sum(dim=1)
        d = torch.where(meal_id == 3, grams, torch.zeros_like(grams)).sum(dim=1)
        penalty = (
            torch.clamp(self.min_b - b, min=0)
            + torch.clamp(self.min_l - l, min=0)
            + torch.clamp(self.min_d - d, min=0)
        )

        # Condiment/oil share hard (groups 8 & 9 by default) — steer search before archive
        hard = self.cfg.get("hard_operability") or {}
        if hard:
            codes = [int(x) for x in hard.get("condiment_oil_food_group_codes", [8, 9])]
            ch = float(hard.get("condiment_oil_weight_share_hard", 0.85))
            need_n = int(hard.get("condiment_fail_n_meals", 2))
            # proxy min energy: meal grams >= 80g (~condiment_min_meal_kcal without cal feature)
            min_g = 80.0
            is_cond = torch.zeros_like(fgroup, dtype=torch.bool)
            for c in codes:
                is_cond = is_cond | (fgroup == c)
            bad_count = torch.zeros(grams.size(0), device=grams.device)
            for mid, meal_tot in ((1, b), (2, l), (3, d)):
                m_mask = (meal_id == mid) & mask
                m_g = torch.where(m_mask, grams, torch.zeros_like(grams)).sum(dim=1)
                cond_g = torch.where(m_mask & is_cond, grams, torch.zeros_like(grams)).sum(dim=1)
                share = torch.where(m_g > 0, cond_g / m_g.clamp(min=1e-6), torch.zeros_like(m_g))
                bad = (share >= ch) & (m_g >= min_g)
                bad_count = bad_count + bad.float()
            # large penalty if too many dominated meals (same spirit as Silver hard)
            over = torch.clamp(bad_count - float(need_n - 1), min=0.0)
            penalty = penalty + over * 500.0
        return penalty

    def fitness_function(self):
        pop = self.ga.population
        n_sel = pop.sum(dim=1)
        ok_card = (n_sel >= self.min_foods) & (n_sel <= self.max_combination_size)
        sel = pop.float()
        nut_sums = sel @ self.features[:, self.META_DIM :]
        below = torch.clamp(self.target_min - nut_sums, min=0)
        above = torch.clamp(nut_sums - self.target_max, min=0)
        nut_pen = below.sum(dim=1) + above.sum(dim=1)

        grams = self.features[:, 3]
        meal_id = self.features[:, 4]
        fgroup = self.features[:, 5]
        gmat = sel * grams
        b = (gmat * (meal_id == 1).float()).sum(dim=1)
        l = (gmat * (meal_id == 2).float()).sum(dim=1)
        d = (gmat * (meal_id == 3).float()).sum(dim=1)
        struct_pen = (
            torch.clamp(self.min_b - b, min=0)
            + torch.clamp(self.min_l - l, min=0)
            + torch.clamp(self.min_d - d, min=0)
        )
        hard = self.cfg.get("hard_operability") or {}
        if hard:
            codes = [int(x) for x in hard.get("condiment_oil_food_group_codes", [8, 9])]
            ch = float(hard.get("condiment_oil_weight_share_hard", 0.85))
            need_n = int(hard.get("condiment_fail_n_meals", 2))
            min_g = 80.0
            is_cond = torch.zeros_like(fgroup, dtype=torch.bool)
            for c in codes:
                is_cond = is_cond | (fgroup == c)
            cond_w = is_cond.float() * grams
            bad_count = torch.zeros(sel.size(0), device=sel.device)
            for mid, meal_tot in ((1, b), (2, l), (3, d)):
                m_g = meal_tot
                cond_g = (sel * cond_w * (meal_id == mid).float()).sum(dim=1)
                share = torch.where(m_g > 0, cond_g / m_g.clamp(min=1e-6), torch.zeros_like(m_g))
                bad_count = bad_count + ((share >= ch) & (m_g >= min_g)).float()
            struct_pen = struct_pen + torch.clamp(bad_count - float(need_n - 1), min=0.0) * 500.0

        pen = nut_pen + struct_pen
        total_g = gmat.sum(dim=1)
        if self.hard_daily_weight:
            # soft pressure toward max, hard gate below
            over_g = torch.clamp(total_g - self.max_daily_g, min=0.0)
            pen = pen + over_g * 0.01
            ok_wt = total_g <= self.max_daily_g
        else:
            ok_wt = torch.ones_like(ok_card)

        # Cell-GA: gram-band violations are soft in fitness; archive Hard is
        # presence-only (hi≤0 forbidden types) unless category_grams_hard is on.
        cell_soft = torch.zeros(pop.size(0), device=self.device)
        if self.cell_ga_enabled and self._cell_band_lo is not None:
            from algo.structure_cell_ops import cell_grams_batch, structure_ok_batch

            major_id = fgroup  # food_group_code == FNDDS major 1–9
            cell_g = cell_grams_batch(sel, grams, meal_id, major_id)
            s_ok = structure_ok_batch(
                cell_g, self._cell_band_lo, self._cell_band_hi, self._cell_upper_only
            )
            over = torch.clamp(cell_g - self._cell_band_hi, min=0.0)
            under = torch.zeros_like(over)
            for j in range(9):
                mi = j + 1
                if mi in self._cell_upper_only:
                    continue
                lo = self._cell_band_lo[:, j]
                under[:, :, j] = torch.where(
                    lo.unsqueeze(0) > 0,
                    torch.clamp(lo.unsqueeze(0) - cell_g[:, :, j], min=0.0),
                    under[:, :, j],
                )
            cell_soft = (over + under).sum(dim=(1, 2))
            self._last_cell_g = cell_g
            self._last_s_ok = s_ok
        else:
            self._last_cell_g = None
            self._last_s_ok = None

        if self.benchmark_zero_cell_soft:
            cell_soft = torch.zeros_like(cell_soft)

        # Gram bands are soft (cell_soft). fitness==1 = nutrient feasible (archive
        # still Hard-rejects forbidden food types hi≤0 via presence check).
        fitness_scores = torch.zeros(self.ga.population_size, device=self.device)
        if self.require_bld:
            good = ok_card & ok_wt & (pen <= 0)
            soft_den = 1.0 + pen + 0.02 * cell_soft
            fitness_scores = torch.where(ok_card & ok_wt, 1.0 / soft_den, fitness_scores)
            fitness_scores = torch.where(good, torch.ones_like(fitness_scores), fitness_scores)
        else:
            good = ok_card & ok_wt & (nut_pen <= 0)
            soft = 1.0 / (
                1.0 + torch.log1p(nut_pen + 0.05 * struct_pen + 0.02 * cell_soft)
            )
            fitness_scores = torch.where(ok_card & ok_wt, soft, fitness_scores)
            fitness_scores = torch.where(good, torch.ones_like(fitness_scores), fitness_scores)

        valid_indices: list[torch.Tensor] = []
        n_good = int(good.sum().item())
        if n_good:
            k = int(self.max_combination_size)
            pop_g = pop[good]
            vals, idx = pop_g.float().topk(k, dim=1)
            idx = idx.masked_fill(vals < 0.5, -1)
            valid_indices = list(idx)
        return valid_indices, fitness_scores

    def _solution_from_indices(self, selected_indices: torch.Tensor) -> Optional[pd.DataFrame]:
        from algo.meal_merge import keep_one_food_per_day, keep_one_food_per_eating_type

        valid = selected_indices != -1
        idx = selected_indices[valid].detach().cpu().numpy()
        if len(idx) == 0:
            return None
        sol = self.food_df.iloc[idx].copy()
        # One food_id per meal, and at most once in the whole day
        return keep_one_food_per_day(keep_one_food_per_eating_type(sol))

    def _repair_bld(self, sol: pd.DataFrame) -> pd.DataFrame:
        """If a major meal is missing, append one moderate food from that meal in the pool."""
        from algo.meal_merge import keep_one_food_per_day, keep_one_food_per_eating_type

        metrics = day_structure_metrics(sol)
        ok, _ = check_bld_hard(metrics, self.cfg)
        if ok:
            return sol
        out = sol.copy()
        used = set(
            pd.to_numeric(out["food_id"], errors="coerce").dropna().astype(int).tolist()
        ) if "food_id" in out.columns else set()
        mins = self.cfg["min_meal_weight_g"]
        for meal, mid in (("breakfast", 1), ("lunch", 2), ("dinner", 3)):
            need = float(mins.get(meal, 0))
            have = float(metrics["meal_g"].get(meal, 0.0))
            if have >= need:
                continue
            cand = self.food_df[self.food_df["meal_id"] == mid]
            if cand.empty:
                continue
            if used:
                fids = pd.to_numeric(cand["food_id"], errors="coerce")
                cand = cand[~fids.isin(used)]
            if cand.empty:
                continue
            # pick densest calorie row that still fits alone under nutrient uppers if possible
            row = cand.nlargest(1, "calorie").iloc[0]
            fid = int(pd.to_numeric(row["food_id"], errors="coerce") or -1)
            if fid >= 0:
                used.add(fid)
            out = pd.concat([out, row.to_frame().T], ignore_index=True)
        return keep_one_food_per_day(keep_one_food_per_eating_type(out))

    def _update_archive(self, valid_indices) -> None:
        for selected_indices in valid_indices:
            sol = self._solution_from_indices(selected_indices)
            if sol is None or sol.empty:
                continue
            # Prefer B/L/D-complete days; try a light repair if missing a meal
            metrics = day_structure_metrics(sol)
            ok, _ = check_bld_hard(metrics, self.cfg)
            if not ok:
                sol = self._repair_bld(sol)
                metrics = day_structure_metrics(sol)
                ok, _ = check_bld_hard(metrics, self.cfg)
            # Re-check nutrients after repair
            nut_ok = True
            for nut, (lo, hi) in self.target_dict.items():
                if nut not in sol.columns:
                    continue
                val = float(pd.to_numeric(sol[nut], errors="coerce").fillna(0).sum())
                if not (float(lo) <= val <= float(hi)):
                    nut_ok = False
                    break
            if not nut_ok:
                continue
            if self.require_bld and not ok:
                continue
            # Hard: each food_id at most once in the day (no cross-meal repeats)
            if "food_id" in sol.columns:
                fids = pd.to_numeric(sol["food_id"], errors="coerce").dropna().astype(int)
                if int(fids.nunique()) < int(len(fids)):
                    from algo.meal_merge import keep_one_food_per_day

                    sol = keep_one_food_per_day(sol)
                    metrics = day_structure_metrics(sol)
                    # re-check nutrients after drop
                    nut_ok2 = True
                    for nut, (lo, hi) in self.target_dict.items():
                        if nut not in sol.columns:
                            continue
                        val = float(pd.to_numeric(sol[nut], errors="coerce").fillna(0).sum())
                        if not (float(lo) <= val <= float(hi)):
                            nut_ok2 = False
                            break
                    if not nut_ok2:
                        continue
                    fids = pd.to_numeric(sol["food_id"], errors="coerce").dropna().astype(int)
                    if int(fids.nunique()) < int(len(fids)):
                        continue
            # Hard: daily total weight + food cardinality (NHANES-like)
            n_foods = int(metrics.get("n_unique_foods", sol["food_id"].nunique() if "food_id" in sol.columns else len(sol)))
            total_g = float(metrics.get("total_g", 0.0))
            if n_foods > self.max_combination_size or n_foods < self.min_foods:
                continue
            if self.hard_daily_weight and total_g > self.max_daily_g:
                continue
            # Hard: oil/condiment/sweet-beverage weight share (Bronze = Silver rule)
            oil_ok, _oil_fails = check_condiment_oil_hard(sol, self.cfg)
            if not oil_ok:
                continue
            # Hard: no RTE cereal/crackers on lunch/dinner
            try:
                from algo.bronze_food_filter import is_breakfast_snack_only, is_breakfast_only

                if "meal" not in sol.columns and "meal_id" in sol.columns:
                    from algo.diet_structure_priors import ID_TO_MEAL

                    sol = sol.copy()
                    sol["meal"] = sol["meal_id"].map(ID_TO_MEAL)
                meal = sol["meal"].astype(str).str.lower() if "meal" in sol.columns else None
                if meal is not None and "food_id" in sol.columns:
                    desc_map = getattr(self, "_food_desc_cache", None)
                    if desc_map is None:
                        try:
                            fc = pd.read_csv(
                                "data/nhanes/data/food_code.csv",
                                usecols=["food_id", "food_desc"],
                            )
                            desc_map = dict(zip(fc.food_id.astype(int), fc.food_desc.astype(str)))
                        except Exception:
                            desc_map = {}
                        self._food_desc_cache = desc_map
                    bad_ld = False
                    for i in sol.index:
                        m = str(meal.at[i])
                        fid = int(pd.to_numeric(sol.at[i, "food_id"], errors="coerce") or 0)
                        name = desc_map.get(fid, "")
                        if not name:
                            continue
                        if is_breakfast_only(name) and m != "breakfast":
                            bad_ld = True
                            break
                        if m in ("lunch", "dinner") and is_breakfast_snack_only(name):
                            bad_ld = True
                            break
                    if bad_ld:
                        continue
            except Exception:
                pass
            # Hard: category-gram major envelopes; optional plate soft/iforest
            learned_score = None
            category_score = None
            feat = None
            cal_lo = cal_hi = None
            if "calorie" in self.target_dict:
                cal_lo = float(self.target_dict["calorie"][0])
                cal_hi = float(self.target_dict["calorie"][1])
            pp = self.participant_profile or {}
            need_feat = (
                self.personal_category_structure is not None
                or self.category_prior is not None
                or self.learned_structure is not None
            )
            if need_feat:
                try:
                    from algo.structure_features import day_structure_feature_dict

                    if "meal" not in sol.columns and "meal_id" in sol.columns:
                        from algo.diet_structure_priors import ID_TO_MEAL

                        sol = sol.copy()
                        sol["meal"] = sol["meal_id"].map(ID_TO_MEAL)
                    feat = day_structure_feature_dict(sol)
                except Exception:
                    feat = None
            if feat is not None and (
                self.personal_category_structure is not None or self.category_prior is not None
            ):
                try:
                    if self.personal_category_structure is not None:
                        from algo.structure_category import score_personal_category_day

                        # Default: presence-only (hi≤0 forbidden). Full gram bands optional.
                        category_score = score_personal_category_day(
                            feat,
                            self.personal_category_structure,
                            presence_only=not self.category_grams_hard,
                        )
                        if self.category_grams_hard and not category_score["ok"]:
                            continue
                        if (
                            self.category_presence_hard
                            and not self.category_grams_hard
                            and not category_score["ok"]
                        ):
                            continue
                    else:
                        from algo.structure_category import score_category_day

                        category_score = score_category_day(
                            feat,
                            self.category_prior,
                            bmi=pp.get("bmi"),
                            sex=pp.get("sex", pp.get("gender")),
                            age=pp.get("age"),
                            total_g_ref=pp.get("total_g_ref"),
                            calorie_lo=cal_lo,
                            calorie_hi=cal_hi,
                        )
                        if self.category_grams_hard and not category_score["ok"]:
                            continue
                except Exception:
                    category_score = None
            if self.learned_structure is not None and feat is not None:
                try:
                    from algo.structure_learn import score_day_features

                    learned_score = score_day_features(
                        feat,
                        self.learned_structure,
                        require_iforest=self.learned_require_iforest,
                        calorie_lo=cal_lo,
                        calorie_hi=cal_hi,
                        bmi=pp.get("bmi"),
                        sex=pp.get("sex", pp.get("gender")),
                        age=pp.get("age"),
                        total_g_ref=pp.get("total_g_ref"),
                    )
                    if self.learned_structure_hard and not learned_score["ok"]:
                        continue
                except Exception:
                    learned_score = None
            scored = layer_b_score(sol, self.structure_prior, self._uncommon, self.cfg)
            # bonus if B/L/D complete
            if ok:
                scored["score"] = float(scored["score"]) + 5.0
            if category_score is not None:
                scored["category_structure_ok"] = bool(category_score.get("ok", False))
                scored["category_g_scale"] = float(category_score.get("g_scale", 1.0))
                if category_score.get("critical_fails"):
                    scored["category_fails"] = list(category_score["critical_fails"][:6])
            if learned_score is not None and self.w_learned_iforest:
                # higher iforest decision_function → more in-distribution
                scored["score"] = float(scored["score"]) + self.w_learned_iforest * float(
                    learned_score.get("iforest_score", 0.0)
                )
                scored["learned_iforest_score"] = float(learned_score.get("iforest_score", 0.0))
                scored["learned_structure_ok"] = bool(learned_score.get("ok", False))
            fp = scored["fingerprint"]
            if not fp:
                continue
            h = scored["fingerprint_hash"]
            prev = self.archive.get(h)
            # Uniform archive: first-seen fingerprint wins (no Layer-B replace).
            if self.benchmark_uniform_archive:
                admit = prev is None
            else:
                admit = prev is None or scored["score"] > prev["score"]
            if admit:
                same_fp = prev is not None
                self.archive[h] = {
                    "score": scored["score"] if not self.benchmark_uniform_archive else float(
                        len(self.archive) if prev is None else prev["score"]
                    ),
                    "structure_score": scored["structure_score"],
                    "diversity_score": scored["diversity_score"],
                    "fingerprint": fp,
                    "fingerprint_hash": h,
                    "solution": sol,
                    "has_bld": bool(ok),
                    "layer_b_raw": float(scored["score"]),
                    "uniform_archive": bool(self.benchmark_uniform_archive),
                }
                # Benchmark observer hooks (no effect unless set)
                ctr = getattr(self, "_bench_archive_counters", None)
                if ctr is not None:
                    if same_fp:
                        ctr["n_same_fp_replace"] = int(ctr.get("n_same_fp_replace", 0)) + 1
                    else:
                        ctr["n_new_fp"] = int(ctr.get("n_new_fp", 0)) + 1
                cb = getattr(self, "benchmark_archive_callback", None)
                if callable(cb):
                    try:
                        cb(sol, scored, same_fp=same_fp)
                    except Exception:
                        pass

    def _rebuild_ga(self, population_size: int, crossover_rate: float = None, mutation_rate: float = None):
        """Recreate GA population (used when escalating p)."""
        cr = crossover_rate if crossover_rate is not None else self.ga.crossover_rate
        mr = mutation_rate if mutation_rate is not None else self.ga.mutation_rate
        self.ga = GeneticAlgorithm(
            population_size=population_size,
            chromosome_length=self.num_foods,
            crossover_rate=cr,
            mutation_rate=mr,
            device=self.device,
            debug=self.debug,
        )
        self._seed_bld_population()

    def _run_generations(
        self,
        generations: int,
        stop_at: int,
        target_idx=0,
        batch_index=0,
        stage_name: str = "",
        escalate_after_no_feasible: Optional[int] = None,
    ) -> str:
        """
        Returns: 'ok' | 'escalate' | 'done'
        """
        no_feas_streak = 0
        no_growth_streak = 0
        prev_archive = len(self.archive)
        # Stop wasting gens when archive is stuck below target
        ad = self.cfg.get("adaptive") or {}
        no_growth_limit = int(ad.get("no_growth_stall_gens", 25))
        # feasible>0 but archive stays 0 (structure Hard reject) used to burn full g
        zero_arch_limit = int(ad.get("zero_archive_stall_gens", 20))
        if self.cell_ga_enabled:
            zero_arch_limit = int(
                self.cell_ga_cfg.get("zero_archive_stall_gens", min(zero_arch_limit, 15))
            )
        desc = f"B{batch_index} t{target_idx} {stage_name or 'GA-v2'}"
        pbar = tqdm(range(generations), desc=desc, leave=False)
        for _ in pbar:
            valid_indices, fitness_scores = self.fitness_function()
            if valid_indices:
                self._update_archive(valid_indices)
                no_feas_streak = 0
            else:
                no_feas_streak += 1
            cur_archive = len(self.archive)
            if cur_archive > prev_archive:
                no_growth_streak = 0
                prev_archive = cur_archive
            else:
                no_growth_streak += 1
            pbar.set_postfix(
                archive=cur_archive,
                feasible=len(valid_indices),
                p=self.ga.population_size,
            )
            if cur_archive >= stop_at:
                return "done"
            # Optional wall-clock deadline (benchmark / fair time budgets)
            deadline = getattr(self, "wall_deadline", None)
            if deadline is not None and time.perf_counter() >= float(deadline):
                return "timeout"
            if (
                escalate_after_no_feasible is not None
                and no_feas_streak >= escalate_after_no_feasible
                and cur_archive == 0
            ):
                return "escalate"
            # Zero archive stuck (often nutrient-OK / structure-fail) → don't burn remaining gens
            if cur_archive == 0 and no_growth_streak >= zero_arch_limit:
                return "stalled"
            # Partial archive but not growing → move on / stop burning GPU
            if cur_archive > 0 and no_growth_streak >= no_growth_limit and cur_archive < stop_at:
                return "stalled"
            self.ga.select_next_generation(fitness_scores)
            if self.cell_ga_enabled:
                self._structure_cell_crossover_mutate()
            else:
                self.ga.crossover()
                self.ga.mutate()
            self._clamp_pop_cardinality()
            self._dedupe_pop_one_food_per_day()
        return "ok"

    def _structure_cell_crossover_mutate(self) -> None:
        """food_id mix on evolve cells; CUDA path stays on-device."""
        import random as _random

        from algo.structure_cell_ops import (
            cell_crossover_mutate_fast,
            cell_grams_batch,
            evolve_mask_batch,
            structure_ok_batch,
            target_ok_batch,
        )

        pop = self.ga.population
        sel = pop.float()
        grams = self.features[:, 3]
        meal_id = self.features[:, 4]
        major_id = self.features[:, 5]
        nut = self.features[:, self.META_DIM :]
        nut_sums = sel @ nut
        cell_g = cell_grams_batch(sel, grams, meal_id, major_id)
        s_ok = structure_ok_batch(
            cell_g, self._cell_band_lo, self._cell_band_hi, self._cell_upper_only
        )
        t_ok, t_below, t_above = target_ok_batch(nut_sums, self.target_min, self.target_max)
        # Ablation: structure vs nutrient contributions to evolve mask
        s_ok_for_evolve = s_ok
        t_ok_for_evolve = t_ok
        resp_for_evolve = None
        if self.benchmark_ignore_structure_evolve:
            s_ok_for_evolve = torch.ones_like(s_ok)
        if self.benchmark_disable_nutrient_guidance:
            resp_for_evolve = {}
            t_ok_for_evolve = torch.ones_like(t_ok)
        evolve = evolve_mask_batch(
            s_ok_for_evolve,
            t_ok_for_evolve,
            self.nutrient_columns,
            responsibility=resp_for_evolve,
            band_hi=self._cell_band_hi,
        )
        if self.benchmark_disable_freeze:
            # All non-hi0 cells evolvable (keep hi≤0 lock)
            evolve = torch.ones_like(evolve, dtype=torch.bool)
            if self._cell_band_hi is not None:
                locked = self._cell_band_hi <= 0
                evolve = evolve & (~locked.unsqueeze(0))
        max_k = int(self.cell_ga_cfg.get("max_foods_per_cell", 3))
        # Cap effective mutation: cell ops are heavier than bit-flip
        mut = min(
            float(self.ga.mutation_rate),
            float(self.cell_ga_cfg.get("mutation_rate_cap", 0.15)),
        )
        # Optional experimental stream; production retains fresh-entropy behavior.
        cell_rng = getattr(self, "benchmark_cell_rng", None)
        if cell_rng is None:
            cell_rng = _random.Random(secrets.randbits(31))
        s_ok_for_mutate = (
            torch.ones_like(s_ok) if self.benchmark_ignore_structure_evolve else s_ok
        )
        t_below_m = (
            torch.zeros_like(t_below)
            if self.benchmark_disable_nutrient_guidance
            else t_below
        )
        t_above_m = (
            torch.zeros_like(t_above)
            if self.benchmark_disable_nutrient_guidance
            else t_above
        )
        protect = bool(self.cell_ga_cfg.get("protect_elites", True))
        if self.benchmark_disable_freeze:
            protect = False
        cell_crossover_mutate_fast(
            pop,
            self._cell_cache or [],
            evolve,
            s_ok_for_mutate,
            cell_g,
            self._cell_band_lo,
            self._cell_band_hi,
            t_below_m,
            t_above_m,
            self.nutrient_columns,
            crossover_rate=float(self.ga.crossover_rate),
            mutation_rate=mut,
            max_foods_per_cell=max_k,
            elite_size=int(getattr(self.ga, "elite_size", 0) or 0),
            responsibility=resp_for_evolve,
            rng=cell_rng,
            mutate_topk=int(self.cell_ga_cfg.get("mutate_topk", 5)),
            mutate_explore_frac=float(self.cell_ga_cfg.get("mutate_explore_frac", 0.25)),
            protect_elites=protect,
        )

    def _clamp_pop_cardinality(self) -> None:
        """Ensure no chromosome exceeds max_combination_size after mutate/crossover."""
        pop = self.ga.population
        n_sel = pop.sum(dim=1)
        over = torch.where(n_sel > self.max_combination_size)[0]
        if len(over) == 0:
            return
        max_n = int(self.max_combination_size)
        for i in over.tolist():
            on = torch.where(pop[i])[0]
            # randomly drop extras
            perm = on[torch.randperm(len(on), device=pop.device)]
            drop = perm[max_n:]
            pop[i, drop] = False

    def _dedupe_pop_one_food_per_eating_type(self) -> None:
        """At most one pool row per (eating_type, food_id); keep largest grams."""
        self._dedupe_pop_cpu(by_eating_type=True)

    def _dedupe_pop_one_food_per_day(self) -> None:
        """At most one pool row per food_id / day (keep max grams).

        Still runs every generation (quality). Uses CPU numpy — a naïve
        per-food_id GPU group loop was slower (~8×) than one host transfer.
        """
        self._dedupe_pop_cpu(by_eating_type=False)

    def _dedupe_pop_cpu(self, *, by_eating_type: bool) -> None:
        """CPU numpy dedupe fallback."""
        pop = self.ga.population
        pop_np = pop.detach().cpu().numpy()
        fids = self.food_ids_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
        grams = self.features[:, 3].detach().cpu().numpy()
        et = self.features[:, 2].detach().cpu().numpy().astype(np.int64, copy=False) if by_eating_type else None
        p = pop_np.shape[0]
        for i in range(p):
            on = np.flatnonzero(pop_np[i])
            if on.size <= 1:
                continue
            order = np.argsort(-grams[on])
            on_s = on[order]
            seen: set[int] = set()
            drop: list[int] = []
            for j in on_s:
                fid = int(fids[j])
                if fid < 0:
                    continue
                key = (int(et[j]) * 10_000_000_000 + fid) if et is not None else fid
                if key in seen:
                    drop.append(int(j))
                else:
                    seen.add(key)
            if drop:
                pop_np[i, drop] = False
        pop.copy_(torch.from_numpy(pop_np).to(device=pop.device, dtype=torch.bool))

    def _pack_solutions(self):
        from algo.meal_merge import keep_one_food_per_day, keep_one_food_per_eating_type

        ranked = sorted(self.archive.values(), key=lambda x: x["score"], reverse=True)
        top = ranked[: self.n_solutions]
        solutions = []
        for item in top:
            sol = item["solution"].copy()
            # One food_id per meal, and at most once in the whole day
            sol = keep_one_food_per_day(keep_one_food_per_eating_type(sol))
            # Contract: grams_raw = grams = NHANES; FNDDS serve_* are labels only
            drop = [c for c in ("grams_snapped", "_merged_n") if c in sol.columns]
            if drop:
                sol = sol.drop(columns=drop)
            try:
                from algo.operable_units import attach_fndds_serves

                sol = attach_fndds_serves(sol, prefer_grams_raw=True)
            except Exception:
                pass
            sol["fingerprint_hash"] = item["fingerprint_hash"]
            sol["structure_score"] = item["structure_score"]
            sol["diversity_score"] = item["diversity_score"]
            sol["layer_b_score"] = item["score"]
            solutions.append(sol)
        return solutions

    def optimize(self, target_idx=0, batch_index=0, generations=500):
        """Single-stage run (legacy). Prefer optimize_adaptive for production."""
        self.archive = {}
        self._seed_bld_population()
        self._run_generations(
            generations=generations,
            stop_at=self.archive_target,
            target_idx=target_idx,
            batch_index=batch_index,
            stage_name="GA-v2",
        )
        valid_indices, _ = self.fitness_function()
        if valid_indices:
            self._update_archive(valid_indices)
        sols = self._pack_solutions()
        print(f"GA-v2 archive={len(self.archive)} returning={len(sols)}", flush=True)
        return sols

    def optimize_adaptive(
        self,
        target_idx=0,
        batch_index=0,
        crossover_rate: float = 0.2,
        mutation_rate: float = 0.9,
    ):
        """
        Run configured adaptive stages (often a single p/g for one escalation pass).
        Early-exit a mid-stage if no feasible for N gens (last stage always uses full g).
        Sets self.last_stage_log: list of {stage,p,g,time_s,n_solutions,status}.
        """
        self.archive = {}
        self.last_stage_log = []
        ad = self.cfg.get("adaptive") or {}
        if not ad.get("enabled", True):
            return self.optimize(target_idx=target_idx, batch_index=batch_index, generations=100)

        stages = ad.get("stages") or [
            {"name": "easy", "population_size": 1000, "generations": 50, "stop_at": 7},
            {"name": "hard", "population_size": 5000, "generations": 300, "stop_at": 7},
        ]
        escalate_n = int(ad.get("no_feasible_escalate_gens", 40))
        need = self.n_solutions
        # Benchmark-only: quality continuation honors stage stop_at even if > n_solutions.
        # Emit count remains n_solutions via _pack_solutions. Production path unchanged.
        qc = bool(getattr(self, "benchmark_quality_continuation", False))

        for si, stage in enumerate(stages):
            p = int(stage["population_size"])
            g = int(stage["generations"])
            stage_stop = int(stage.get("stop_at", need))
            if qc:
                stop_at = stage_stop
            else:
                # Cap by n_solutions (default 7); stage may request fewer.
                stop_at = min(stage_stop, need)
            name = stage.get("name", f"p{p}")
            is_last = si == len(stages) - 1
            if self.ga.population_size != p:
                self._rebuild_ga(p, crossover_rate=crossover_rate, mutation_rate=mutation_rate)
            elif si == 0 or len(self.archive) == 0:
                # First stage or empty archive: fast random BLD seed (GA finds feasibility).
                self._seed_bld_population()

            print(
                f"[tgt {target_idx}] stage={name} p={p} g={g} pool={self.num_foods} "
                f"archive={len(self.archive)} stop_at={stop_at}",
                flush=True,
            )
            # Production: enough returned sols. QC: enough distinct fingerprints to stop early.
            if (not qc) and len(self.archive) >= need:
                break
            if qc and len(self.archive) >= stop_at:
                break
            # Zero-archive stages may escalate early; partial archive uses no-growth stall
            esc = None if (is_last or len(self.archive) > 0) else escalate_n
            t0 = time.perf_counter()
            status = self._run_generations(
                generations=g,
                stop_at=stop_at,
                target_idx=target_idx,
                batch_index=batch_index,
                stage_name=name,
                escalate_after_no_feasible=esc,
            )
            elapsed = time.perf_counter() - t0
            n_now = len(self.archive)
            self.last_stage_log.append(
                {
                    "stage": name,
                    "p": p,
                    "g": g,
                    "time_s": round(elapsed, 3),
                    "n_solutions": int(n_now),
                    "run_status": status,
                    "stop_at": int(stop_at),
                    "quality_continuation": qc,
                }
            )
            print(
                f"[tgt {target_idx}] stage={name} done status={status} "
                f"n_sol={n_now} time={elapsed:.2f}s",
                flush=True,
            )
            if n_now >= stop_at:
                break
            if (not qc) and n_now >= need:
                break
            if status == "escalate":
                print(f"[tgt {target_idx}] no feasible in {escalate_n} gens → next stage", flush=True)
                continue
            if status == "stalled":
                print(
                    f"[tgt {target_idx}] archive stalled at {n_now}<{need} → next stage",
                    flush=True,
                )
                continue

        sols = self._pack_solutions()
        print(
            f"GA-v2 adaptive archive={len(self.archive)} returning={len(sols)}",
            flush=True,
        )
        return sols

    def optimize_sequential_ban_mains(
        self,
        target_idx=0,
        batch_index=0,
        crossover_rate: float = 0.2,
        mutation_rate: float = 0.9,
        *,
        generations_per_sol: Optional[int] = None,
        population_size: Optional[int] = None,
        seed_solutions: Optional[list[pd.DataFrame]] = None,
        ban_plugin=None,
    ) -> list[pd.DataFrame]:
        """
        Find solutions one-by-one. After solution N, optionally shrink the pool
        for N+1 via a pluggable ban plugin (``algo/sequential_ban_plugin.py``):

          - ``none`` (default): no pool ban — ablation baseline
          - ``fndds``: ban day food_ids / FNDDS policy families

        ``seed_solutions``: keep incomplete sols from a prior pass and continue
        searching from N+1 (bans from seeds are applied first). Never drops seeds.
        """
        from algo.sequential_ban_plugin import build_sequential_ban_plugin

        plugin = ban_plugin or build_sequential_ban_plugin(self.cfg)

        ad = self.cfg.get("adaptive") or {}
        stages = ad.get("stages") or []
        if stages:
            p = int(stages[0].get("population_size", self.ga.population_size))
            g = int(stages[0].get("generations", 50))
        else:
            p = int(self.ga.population_size)
            g = 50
        if population_size is not None:
            p = int(population_size)
        if generations_per_sol is not None:
            g = int(generations_per_sol)

        need = int(self.n_solutions)
        base_pool = self.food_df.copy()
        solutions: list[pd.DataFrame] = []
        banned_ids: set[int] = set()
        self.last_stage_log = []
        self.archive = {}

        # optional food_id → desc map for ban matching when sol lacks food_desc
        desc_map: dict[int, str] = {}
        if "food_desc" in base_pool.columns:
            for fid, desc in zip(
                pd.to_numeric(base_pool["food_id"], errors="coerce").fillna(-1).astype(int),
                base_pool["food_desc"].astype(str),
            ):
                if fid > 0 and fid not in desc_map:
                    desc_map[fid] = desc
        if not desc_map:
            try:
                fc = pd.read_csv("data/nhanes/data/food_code.csv", usecols=["food_id", "food_desc"])
                desc_map = {
                    int(a): str(b)
                    for a, b in zip(
                        pd.to_numeric(fc["food_id"], errors="coerce").fillna(-1).astype(int),
                        fc["food_desc"].astype(str),
                    )
                    if int(a) > 0
                }
            except Exception:
                desc_map = {}

        pool_id_list = (
            list(desc_map.keys())
            if desc_map
            else pd.to_numeric(base_pool["food_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        def _apply_ban(sol: pd.DataFrame) -> set[int]:
            return plugin.ban_ids_after(sol, pool_food_ids=pool_id_list)

        # Restore prior incomplete solutions + their sequential bans
        seen_fp: set[str] = set()
        for sol0 in seed_solutions or []:
            if sol0 is None or len(sol0) == 0:
                continue
            sol = sol0.copy()
            if "fingerprint_hash" in sol.columns and pd.notna(sol["fingerprint_hash"].iloc[0]):
                h = str(sol["fingerprint_hash"].iloc[0])
            else:
                h = fingerprint_hash(food_id_fingerprint(sol["food_id"]))
                sol["fingerprint_hash"] = h
            if h in seen_fp:
                continue
            seen_fp.add(h)
            k = len(solutions)
            sol["seq_sol_index"] = k + 1
            solutions.append(sol)
            newly = _apply_ban(sol)
            banned_ids |= newly
            self.archive[h] = {
                "score": float(sol["layer_b_score"].iloc[0]) if "layer_b_score" in sol.columns else 0.0,
                "structure_score": float(sol["structure_score"].iloc[0]) if "structure_score" in sol.columns else 0.0,
                "diversity_score": float(sol["diversity_score"].iloc[0]) if "diversity_score" in sol.columns else 0.0,
                "fingerprint": food_id_fingerprint(sol["food_id"]),
                "fingerprint_hash": h,
                "solution": sol,
                "has_bld": True,
                "banned_ids_after": sorted(banned_ids),
                "ban_plugin": plugin.name,
                "from_seed": True,
            }
        n_seed = len(solutions)
        if n_seed:
            print(
                f"[tgt {target_idx}] sequential resume seed={n_seed}/{need} "
                f"ban_plugin={plugin.name} banned={len(banned_ids)} — continue from sol {n_seed + 1}",
                flush=True,
            )
        if n_seed >= need:
            print(
                f"GA-v2 sequential returning seed-complete {n_seed}/{need} "
                f"ban_plugin={plugin.name}",
                flush=True,
            )
            return solutions[:need]

        print(
            f"[tgt {target_idx}] sequential start need={need} ban_plugin={plugin.name} "
            f"(enabled={plugin.enabled})",
            flush=True,
        )

        # Reuse one Combiner across sols: rebind pool on ban shrink; retry same pool = reseeds only.
        sub: Optional[FoodCombinerV2] = None
        bound_fp: Optional[tuple[int, str]] = None

        for k in range(n_seed, need):
            deadline = getattr(self, "wall_deadline", None)
            if deadline is not None and time.perf_counter() >= float(deadline):
                print(
                    f"[tgt {target_idx}] sequential wall deadline before sol {k+1}/{need}",
                    flush=True,
                )
                break
            if banned_ids:
                fids = pd.to_numeric(base_pool["food_id"], errors="coerce").fillna(-1).astype(int)
                pool_k = base_pool[~fids.isin(banned_ids)].reset_index(drop=True)
            else:
                pool_k = base_pool.reset_index(drop=True)

            if "meal_id" not in pool_k.columns and "eating_type" in pool_k.columns:
                from algo.diet_structure_priors import meal_id_from_eating_type

                pool_k = pool_k.copy()
                pool_k["meal_id"] = (
                    pool_k["eating_type"].map(meal_id_from_eating_type).fillna(0).astype(int)
                )
            meal = pool_k["meal_id"].to_numpy() if "meal_id" in pool_k.columns else np.array([])
            n_b = int((meal == 1).sum()) if len(meal) else 0
            n_l = int((meal == 2).sum()) if len(meal) else 0
            n_d = int((meal == 3).sum()) if len(meal) else 0
            if len(pool_k) < self.min_foods or n_b == 0 or n_l == 0 or n_d == 0:
                print(
                    f"[tgt {target_idx}] sequential stop at sol {k+1}/{need}: "
                    f"pool={len(pool_k)} B/L/D={n_b}/{n_l}/{n_d} banned={len(banned_ids)}",
                    flush=True,
                )
                break

            seq_cfg = dict(self.cfg.get("sequential_ban") or {})
            retries = max(0, int(seq_cfg.get("retries_per_sol", 3)))
            sols = []
            elapsed = 0.0
            pool_fp = self._pool_fingerprint(pool_k)
            for attempt in range(retries + 1):
                adaptive = {
                    "enabled": True,
                    "no_feasible_escalate_gens": max(g, 1),
                    "zero_archive_stall_gens": max(g, 1),
                    "stages": [
                        {
                            "name": f"seq_sol{k+1}"
                            + (f"_r{attempt}" if attempt else ""),
                            "population_size": p,
                            "generations": g,
                            "stop_at": 1,
                        }
                    ],
                }
                pop_seed_offset = int(k) * 17 + int(attempt) + 1
                try:
                    if sub is None:
                        cfg_k = dict(self.cfg)
                        st_k = dict(cfg_k.get("structure") or {})
                        cg_k = dict(st_k.get("cell_ga") or {})
                        cg_k["zero_archive_stall_gens"] = max(g, 1)
                        st_k["cell_ga"] = cg_k
                        cfg_k["structure"] = st_k
                        cfg_k["adaptive"] = adaptive
                        cfg_k["pop_seed_offset"] = pop_seed_offset
                        sub = FoodCombinerV2(
                            pool_k,
                            self.target_dict,
                            max_combination_size=self.max_combination_size,
                            population_size=p,
                            crossover_rate=crossover_rate,
                            mutation_rate=mutation_rate,
                            device=self.device,
                            debug=self.debug,
                            structure_prior=self.structure_prior,
                            config=cfg_k,
                            n_solutions=1,
                            archive_target=1,
                            require_bld=self.require_bld,
                            participant_profile=self.participant_profile,
                            personal_category_structure=self.personal_category_structure,
                        )
                        if getattr(self, "wall_deadline", None) is not None:
                            sub.wall_deadline = self.wall_deadline
                        bound_fp = pool_fp
                        reuse_mode = "new"
                    elif bound_fp != pool_fp:
                        cfg_k = dict(sub.cfg)
                        st_k = dict(cfg_k.get("structure") or {})
                        cg_k = dict(st_k.get("cell_ga") or {})
                        cg_k["zero_archive_stall_gens"] = max(g, 1)
                        st_k["cell_ga"] = cg_k
                        cfg_k["structure"] = st_k
                        cfg_k["adaptive"] = adaptive
                        cfg_k["pop_seed_offset"] = pop_seed_offset
                        sub.rebind_pool(
                            pool_k,
                            config=cfg_k,
                            population_size=p,
                            crossover_rate=crossover_rate,
                            mutation_rate=mutation_rate,
                            quiet=True,
                            freeze_no_resample=bool(
                                getattr(self, "benchmark_freeze_pool", False)
                            ),
                        )
                        sub.n_solutions = 1
                        sub.archive_target = 1
                        if getattr(self, "wall_deadline", None) is not None:
                            sub.wall_deadline = self.wall_deadline
                        bound_fp = pool_fp
                        reuse_mode = "rebind"
                    else:
                        # Same pool (retry): only new seed offset + adaptive stage name
                        sub.configure_search_slot(
                            pop_seed_offset=pop_seed_offset,
                            adaptive=adaptive,
                            n_solutions=1,
                            archive_target=1,
                        )
                        if getattr(self, "wall_deadline", None) is not None:
                            sub.wall_deadline = self.wall_deadline
                        reuse_mode = "reseed"
                except ValueError as e:
                    print(
                        f"[tgt {target_idx}] sequential rebuild failed at sol {k+1}: {e}",
                        flush=True,
                    )
                    sols = []
                    break

                print(
                    f"[tgt {target_idx}] sequential sol={k+1}/{need} "
                    f"attempt={attempt + 1}/{retries + 1} pool={len(pool_k)} "
                    f"ban_plugin={plugin.name} banned={len(banned_ids)} p={p} g={g} "
                    f"reuse={reuse_mode}",
                    flush=True,
                )
                t0 = time.perf_counter()
                sols = sub.optimize_adaptive(
                    target_idx=target_idx,
                    batch_index=batch_index,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                )
                elapsed = time.perf_counter() - t0
                for sl in getattr(sub, "last_stage_log", None) or []:
                    row = dict(sl)
                    row["seq_sol"] = k + 1
                    row["seq_attempt"] = attempt + 1
                    row["reuse"] = reuse_mode
                    self.last_stage_log.append(row)
                if sols:
                    break
                print(
                    f"[tgt {target_idx}] sequential sol={k+1} attempt={attempt + 1} "
                    f"no feasible ({elapsed:.1f}s)"
                    + (" — retry" if attempt < retries else " — give up slot"),
                    flush=True,
                )

            if not sols:
                print(
                    f"[tgt {target_idx}] sequential no feasible for sol {k+1} "
                    f"after {retries + 1} attempts — stop "
                    f"(keep seed={n_seed} total={len(solutions)})",
                    flush=True,
                )
                break

            sol = sols[0].copy()
            if "fingerprint_hash" in sol.columns and pd.notna(sol["fingerprint_hash"].iloc[0]):
                h = str(sol["fingerprint_hash"].iloc[0])
            else:
                h = fingerprint_hash(food_id_fingerprint(sol["food_id"]))
                sol["fingerprint_hash"] = h
            if h in seen_fp:
                print(
                    f"[tgt {target_idx}] sequential sol={k+1} duplicate fp={h[:12]} — skip/retry stop",
                    flush=True,
                )
                # still apply ban plugin so we don't loop the same niche forever
                newly = _apply_ban(sol)
                banned_ids |= newly
                continue
            seen_fp.add(h)
            sol["seq_sol_index"] = k + 1
            solutions.append(sol)
            newly = _apply_ban(sol)
            banned_ids |= newly
            self.archive[h] = {
                "score": float(sol["layer_b_score"].iloc[0]) if "layer_b_score" in sol.columns else 0.0,
                "structure_score": float(sol["structure_score"].iloc[0]) if "structure_score" in sol.columns else 0.0,
                "diversity_score": float(sol["diversity_score"].iloc[0]) if "diversity_score" in sol.columns else 0.0,
                "fingerprint": food_id_fingerprint(sol["food_id"]),
                "fingerprint_hash": h,
                "solution": sol,
                "has_bld": True,
                "banned_ids_after": sorted(banned_ids),
                "ban_plugin": plugin.name,
            }
            print(
                f"[tgt {target_idx}] sequential accepted sol={k+1} "
                f"ban_plugin={plugin.name} ban_this={sorted(newly)} "
                f"banned_total={len(banned_ids)} time={elapsed:.1f}s",
                flush=True,
            )

        print(
            f"GA-v2 sequential ban_plugin={plugin.name} returning={len(solutions)}/{need} "
            f"(seed={n_seed} new={len(solutions) - n_seed}) banned={len(banned_ids)}",
            flush=True,
        )
        return solutions

    def optimize_with_meta(self, target_idx=0, batch_index=0, generations=500):
        """Same as optimize but also returns list of meta dicts aligned with solutions."""
        sols = self.optimize(target_idx=target_idx, batch_index=batch_index, generations=generations)
        metas = []
        for sol in sols:
            metas.append({
                "fingerprint_hash": sol["fingerprint_hash"].iloc[0] if "fingerprint_hash" in sol.columns else "",
                "structure_score": float(sol["structure_score"].iloc[0]) if "structure_score" in sol.columns else 0.0,
                "diversity_score": float(sol["diversity_score"].iloc[0]) if "diversity_score" in sol.columns else 0.0,
                "layer_b_score": float(sol["layer_b_score"].iloc[0]) if "layer_b_score" in sol.columns else 0.0,
                "fingerprint": food_id_fingerprint(sol["food_id"]),
            })
        return sols, metas


def assign_solutions_to_users(
    solutions: list[pd.DataFrame],
    user_ids: list,
    n_per_user: int = 7,
) -> list[pd.DataFrame]:
    """
    Greedy per-user assignment: each user gets up to n_per_user days with
    distinct food_id fingerprints (shared archive, independent used-sets).
    """
    if not solutions:
        return []
    # Precompute fingerprints
    packed = []
    for sol in solutions:
        fp = food_id_fingerprint(sol["food_id"])
        h = fingerprint_hash(fp)
        score = float(sol["layer_b_score"].iloc[0]) if "layer_b_score" in sol.columns else 0.0
        packed.append((score, h, fp, sol))
    packed.sort(key=lambda x: x[0], reverse=True)

    out = []
    for uid in user_ids:
        used = set()
        day = 3
        for score, h, fp, sol in packed:
            if h in used:
                continue
            used.add(h)
            s = sol.copy()
            s["user_id"] = uid
            s["day"] = day
            day += 1
            out.append(s)
            if len(used) >= n_per_user:
                break
    return out
