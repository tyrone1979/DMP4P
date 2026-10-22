"""
V2 production method wrapper (FoodCombinerV2).

Keeps cell-GA, structure-guided operators, nutrient responsibility, and object
reuse. Exposes unified seed / pool snapshot / wall-clock budget / result packing
for the fair benchmark. Solutions are re-validated by UnifiedEvaluator.
"""
from __future__ import annotations

import copy
from typing import Optional

from dmp4p_runtime.algorithms.base import AlgoResult, BaseAlgorithm, SearchBudget
from dmp4p_runtime.budget_clock import BudgetClock
from dmp4p_runtime.rng_util import seed_all


class V2Production(BaseAlgorithm):
    name = "v2"

    def __init__(
        self,
        *args,
        population_size: int = 1000,
        crossover_rate: float = 0.2,
        mutation_rate: float = 0.9,
        quality_continuation: bool = False,
        ablation_variant: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.population_size = int(population_size)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.quality_continuation = bool(quality_continuation)
        self.ablation_variant = ablation_variant
        self._combiner = None
        self.budget_clock: Optional[BudgetClock] = None
        self.dual_observers = None
        if self.quality_continuation:
            self.name = "v2_quality_continuation"
        elif self.ablation_variant:
            self.name = str(self.ablation_variant)

    def _build_combiner(self, n_solutions: int = 1):
        from ga_gpu_v2 import FoodCombinerV2
        from dmp4p_runtime.v2_ablation_flags import apply_ablation_to_cfg

        cfg = copy.deepcopy(self.config)
        if self.ablation_variant:
            cfg, _spec = apply_ablation_to_cfg(cfg, self.ablation_variant)
        if self.quality_continuation:
            cfg["benchmark_quality_continuation"] = True
        cfg["ga_seed"] = int(self.seed)
        cfg["pop_seed_offset"] = 0
        cfg["benchmark_freeze_pool"] = True
        seq = dict(cfg.get("sequential_ban") or {})
        seq["enabled"] = False
        cfg["sequential_ban"] = seq
        device = self.device
        try:
            import torch

            if str(device).startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

        demo = {}
        if self.personal:
            demo = {
                "sex": (self.personal.get("stratum") or "").split("|")[0]
                if self.personal.get("stratum")
                else None,
                "bmi": None,
                "age": None,
            }
        combiner_class = FoodCombinerV2
        if getattr(self, "quality_search", False):
            from dmp4p_runtime.v2_quality_search import FoodCombinerV2Quality
            combiner_class = FoodCombinerV2Quality
            if getattr(self, "quality_search_version", 1) == 2:
                from dmp4p_runtime.v2_quality_recovery import FoodCombinerV2QualityRecovery
                combiner_class = FoodCombinerV2QualityRecovery
        combiner = combiner_class(
            self.pool,
            self.target_dict,
            max_combination_size=int(cfg.get("max_combination_size", 15)),
            population_size=self.population_size,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            device=device,
            structure_prior=self.evaluator.structure_prior,
            config=cfg,
            n_solutions=n_solutions,
            archive_target=n_solutions,
            require_bld=bool(self.evaluator.spec.require_bld),
            participant_profile=demo,
            personal_category_structure=self.personal,
            prebuilt_cell_ranges=self.cell_ranges,
            prebuilt_forbidden_cells=set(
                tuple(x) if not isinstance(x, tuple) else x
                for x in (getattr(self, "_forbidden", None) or [])
            )
            or None,
        )
        if getattr(self, "quality_search_version", 1) == 2:
            combiner.quality_evaluator = self.evaluator
        return combiner, device

    def solve(self, budget: SearchBudget, *, banned_food_ids: Optional[set[int]] = None) -> AlgoResult:
        try:
            return self._solve_impl(budget, banned_food_ids=banned_food_ids or set())
        except Exception as e:
            return AlgoResult(
                algorithm=self.name,
                status="error",
                failure_reason=f"exception:{type(e).__name__}:{e}",
            )

    def _solve_impl(self, budget: SearchBudget, *, banned_food_ids: set[int]) -> AlgoResult:
        from algo.structure_cell_pool import rebuild_cell_ranges_preserve_rows
        from dmp4p_runtime.candidate_ids import pool_candidate_set_hash

        seed_info = seed_all(self.seed, device=self.device)
        clock = getattr(self, "budget_clock", None) or BudgetClock(wall_s=float(budget.wall_s))
        self.budget_clock = clock

        t_init0 = clock.now()
        pool = self.pool
        self_ranges = self.cell_ranges
        remaining_hash = None
        if banned_food_ids:
            pool = self.pool[~self.pool["food_id"].astype(int).isin(banned_food_ids)].reset_index(
                drop=True
            )
            if len(pool) < self.evaluator.spec.min_foods:
                clock.mark_finalization_end()
                return AlgoResult(
                    algorithm=self.name,
                    status="incomplete",
                    failure_reason="insufficient_unbanned_rows",
                    timings={
                        "init_s": clock.now() - t_init0,
                        "budget_overflow_s": clock.budget_overflow_s(),
                    },
                    extra={"budget_clock": clock.to_dict()},
                )
            ordered, ranges, _meta = rebuild_cell_ranges_preserve_rows(
                pool, personal=self.personal
            )
            pool = ordered
            self_ranges = ranges
            if "candidate_id" in pool.columns:
                remaining_hash = pool_candidate_set_hash(pool)

        orig_pool, orig_ranges = self.pool, self.cell_ranges
        self.pool = pool
        self.cell_ranges = self_ranges
        combiner, device = self._build_combiner(n_solutions=int(budget.n_solutions_target))
        combiner.benchmark_freeze_pool = True
        self.pool, self.cell_ranges = orig_pool, orig_ranges
        init_pool_hash = None
        if "candidate_id" in combiner.food_df.columns:
            init_pool_hash = pool_candidate_set_hash(combiner.food_df)
        t_init = clock.now() - t_init0

        stop_at = int(budget.n_solutions_target)
        if self.quality_continuation:
            stop_at = max(stop_at, 10**6)
        ad = {
            "enabled": True,
            "no_feasible_escalate_gens": 10**9,
            "zero_archive_stall_gens": 10**9,
            "no_growth_stall_gens": 10**9,
            "stages": [
                {
                    "name": "bench_qc" if self.quality_continuation else "bench",
                    "population_size": self.population_size,
                    "generations": 100000,
                    "stop_at": stop_at,
                }
            ],
        }
        combiner.cfg = dict(combiner.cfg)
        combiner.cfg["adaptive"] = ad
        if self.quality_continuation:
            combiner.cfg["benchmark_quality_continuation"] = True
            combiner.benchmark_quality_continuation = True
        st = dict(combiner.cfg.get("structure") or {})
        cg = dict(st.get("cell_ga") or {})
        cg["zero_archive_stall_gens"] = 10**9
        st["cell_ga"] = cg
        combiner.cfg["structure"] = st
        combiner.cell_ga_cfg = dict(cg)
        combiner.cfg["ga_seed"] = int(self.seed)
        combiner.n_solutions = int(budget.n_solutions_target)
        combiner.archive_target = stop_at
        combiner.wall_deadline = clock.deadline
        combiner._bench_archive_counters = {"n_same_fp_replace": 0, "n_new_fp": 0}

        dual = getattr(self, "dual_observers", None)
        first_archive_abs = {"t": None}
        online_fps = set()
        online_in_budget: list = []  # packed sols validated before deadline

        def _on_archive(sol, scored, *, same_fp: bool):
            """
            Observe every archive admit/replace while in budget.
            Dedup for packing uses food fingerprint; DualObservers log by plan_uid
            so gram/meal variants of the same food set are not silently dropped.
            """
            if first_archive_abs["t"] is None:
                first_archive_abs["t"] = clock.now()
            if not clock.in_budget():
                return
            h = str(scored.get("fingerprint_hash") or "")
            ev = self.evaluator.evaluate_solution(sol)
            t_done = clock.now()
            if dual is not None and ev.solution is not None:
                dual.observe_verified(
                    ev.solution,
                    feasible=bool(ev.feasible),
                    fingerprint_hash=ev.fingerprint_hash,
                    eval_completed_at=t_done,
                )
            if h:
                online_fps.add(h)
            if clock.in_budget(t_done) and ev.feasible:
                packed = self._pack_eval(
                    ev, found_at_s=float(clock.rel(t_done) or 0.0), eval_count=-1
                )
                packed["in_budget"] = True
                packed["validated_online"] = True
                packed["raw_layer_b"] = getattr(ev, "raw_layer_b", None)
                packed["legacy_selection_score"] = getattr(ev, "legacy_selection_score", None)
                packed["bld_min_weight_pass"] = getattr(ev, "bld_min_weight_pass", None)
                online_in_budget.append(packed)

        combiner.benchmark_archive_callback = _on_archive

        t_search0 = clock.now()
        try:
            import torch

            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        sols_df = combiner.optimize_adaptive(
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
        )
        try:
            import torch

            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        clock.mark_search_end()
        search_s = float(clock.last_search_end) - t_search0

        t_dec0 = clock.now()
        solutions_in = []
        solutions_diag = []
        n_archive_eval_reject = 0
        max_abs_nutrient_dev = 0.0
        nut_cols = list(self.nutrient_cols)
        archive_items = list((getattr(combiner, "archive", {}) or {}).values())
        cand_sols = [it["solution"] for it in archive_items] if archive_items else list(sols_df or [])
        best_lb = None
        best_packed = None
        for sol in cand_sols:
            h = None
            try:
                from algo.diet_structure_priors import fingerprint_hash, food_id_fingerprint

                h = fingerprint_hash(food_id_fingerprint(sol))
            except Exception:
                pass
            # Skip duplicate end-of-run eval if already observed online in-budget
            if h and h in online_fps and clock.in_budget():
                # Still need packed return candidate from last online? Re-eval only if needed.
                pass
            ev = self.evaluator.evaluate_solution(sol)
            t_done = clock.now()
            packed = self._pack_eval(
                ev, found_at_s=float(clock.rel(t_done) or -1.0), eval_count=-1
            )
            in_b = clock.in_budget(t_done)
            packed["in_budget"] = in_b
            if dual is not None and ev.solution is not None and h not in online_fps:
                dual.observe_verified(
                    ev.solution,
                    feasible=bool(ev.feasible),
                    fingerprint_hash=ev.fingerprint_hash,
                    eval_completed_at=t_done,
                )
            if not in_b:
                solutions_diag.append(packed)
                continue
            if ev.feasible:
                lb = ev.layer_b_score
                if lb is None:
                    lb = -1e18
                if best_lb is None or float(lb) > float(best_lb):
                    best_lb = float(lb)
                    best_packed = packed
                solutions_in.append(packed)
            else:
                n_archive_eval_reject += 1
                totals = getattr(ev, "nutrient_totals", None) or {}
                for c in nut_cols:
                    lo_hi = self.target_dict.get(c)
                    if not lo_hi or c not in totals:
                        continue
                    lo, hi = float(lo_hi[0]), float(lo_hi[1])
                    val = float(totals[c])
                    if val < lo:
                        max_abs_nutrient_dev = max(max_abs_nutrient_dev, lo - val)
                    elif val > hi:
                        max_abs_nutrient_dev = max(max_abs_nutrient_dev, val - hi)
        if best_packed is not None:
            solutions_in = [best_packed] + [s for s in solutions_in if s is not best_packed]
        # Prefer online in-budget validations when finalize exhausted the deadline.
        if not solutions_in and online_in_budget:
            best_on = None
            best_on_lb = None
            for p in online_in_budget:
                lb = p.get("layer_b_score")
                if lb is None:
                    continue
                if best_on_lb is None or float(lb) > float(best_on_lb):
                    best_on_lb = float(lb)
                    best_on = p
            if best_on is not None:
                solutions_in = [best_on]
        decode_s = clock.now() - t_dec0
        clock.mark_finalization_end()

        if not solutions_in:
            status, term_reason = "timeout", "no_feasible_in_budget"
        elif self.quality_continuation:
            status, term_reason = "ok", "wall_budget"
        elif len(getattr(combiner, "archive", {}) or {}) >= int(budget.n_solutions_target):
            status, term_reason = "ok", "archive_target_reached"
        else:
            status, term_reason = "ok", "wall_or_stall"

        eval_count = 0
        for row in getattr(combiner, "last_stage_log", []) or []:
            eval_count += int(row.get("p", self.population_size)) * max(1, int(row.get("g", 0)))
        if eval_count <= 0:
            eval_count = self.population_size

        ctr = getattr(combiner, "_bench_archive_counters", {}) or {}
        return AlgoResult(
            algorithm=self.name,
            status=status,
            failure_reason=None if solutions_in else "no_feasible_in_budget",
            solutions=solutions_in[: budget.n_solutions_target],
            timings={
                "init_s": t_init,
                "search_s": search_s,
                "decode_eval_s": decode_s,
                "init_search_decode_merged_note": (
                    "init+search+decode+validate share one deadline from solve start; "
                    "phase fields are approximate wall slices, not exclusive partitions"
                ),
                "budget_overflow_s": clock.budget_overflow_s(),
            },
            evaluation_count=eval_count,
            extra={
                "seed_info": seed_info,
                "device": device,
                "stage_log": getattr(combiner, "last_stage_log", []),
                "archive_size_raw": len(getattr(combiner, "archive", {}) or {}),
                "n_evaluator_accepted_in_budget": len(solutions_in),
                "n_post_deadline_diagnostic": len(solutions_diag),
                "post_deadline_diagnostic_solutions": solutions_diag[:20],
                "n_archive_evaluator_reject": n_archive_eval_reject,
                "max_abs_nutrient_dev_archive_reject": max_abs_nutrient_dev,
                "candidate_set_hash": init_pool_hash,
                "remaining_candidate_set_hash": remaining_hash,
                "benchmark_freeze_pool": True,
                "sequential_ban_disabled": True,
                "quality_continuation": bool(self.quality_continuation),
                "ablation_variant": self.ablation_variant,
                "stop_at_configured": stop_at,
                "termination_reason": term_reason,
                "first_archive_admit_rel_s": clock.rel(first_archive_abs["t"]),
                "budget_clock": clock.to_dict(),
                "archive_counters": dict(ctr),
                "quality_search_stats": dict(getattr(combiner, "quality_stats", {})),
                "quality_rejection_reasons": dict(getattr(combiner, "quality_rejection_reasons", {})),
                "dual_observers": dual.summary() if dual is not None else None,
                "benchmark_flags": {
                    "zero_cell_soft": bool(getattr(combiner, "benchmark_zero_cell_soft", False)),
                    "uniform_archive": bool(getattr(combiner, "benchmark_uniform_archive", False)),
                    "disable_nutrient_guidance": bool(
                        getattr(combiner, "benchmark_disable_nutrient_guidance", False)
                    ),
                    "disable_freeze": bool(getattr(combiner, "benchmark_disable_freeze", False)),
                    "ignore_structure_evolve": bool(
                        getattr(combiner, "benchmark_ignore_structure_evolve", False)
                    ),
                },
            },
        )

    def solve_sequential(
        self,
        budget: SearchBudget,
        *,
        ban_plugin=None,
        n_solutions: int = 7,
    ) -> AlgoResult:
        """Fair multi_sol: ban-delete + preserve_rows re-index via solve()."""
        res = super().solve_sequential(
            budget, ban_plugin=ban_plugin, n_solutions=n_solutions
        )
        extra = dict(res.extra or {})
        extra["benchmark_freeze_pool"] = True
        extra["multi_sol_path"] = "ban_delete_only_preserve_rows"
        res.extra = extra
        return res


class V2QualityContinuation(V2Production):
    """Named E1a variant: continue post-feasibility until wall (not production)."""

    name = "v2_quality_continuation"

    def __init__(self, *args, **kwargs):
        kwargs["quality_continuation"] = True
        kwargs.pop("ablation_variant", None)
        super().__init__(*args, **kwargs)


def make_v2_ablation(variant: str):
    """Factory for E2 named ablation algorithms."""

    class _V2Ablation(V2Production):
        name = variant

        def __init__(self, *args, **kwargs):
            kwargs["ablation_variant"] = variant
            kwargs["quality_continuation"] = False
            super().__init__(*args, **kwargs)

    _V2Ablation.__name__ = f"V2Ablation_{variant}"
    return _V2Ablation


class V2Quality(V2QualityContinuation):
    name = "v2_quality"
    quality_search = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "v2_quality"


class V2QualityRecovery(V2Quality):
    name = "v2_quality_recovery"
    quality_search_version = 2
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.name = "v2_quality_recovery"
