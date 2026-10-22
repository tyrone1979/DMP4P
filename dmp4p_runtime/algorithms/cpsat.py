"""
CP-SAT discrete feasibility baseline (OR-Tools).

Hard constraints aim to match UnifiedEvaluator. Conservative integer scaling may
shrink the feasible region; integer INFEASIBLE ≠ original float infeasible.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd

from dmp4p_runtime.algorithms.base import AlgoResult, BaseAlgorithm, SearchBudget

DEFAULT_NUTRIENT_SCALES: dict[str, int] = {
    "calorie": 1,
    "protein": 10,
    "carb": 10,
    "sugar": 10,
    "fiber": 10,
    "saturated_fat": 100,
    "cholesterol": 10,
    "folic_acid": 10,
    "vitamin_b12": 100,
    "vitamin_c": 10,
    "vitamin_d": 100,
    "calcium": 1,
    "phosphorus": 1,
    "potassium": 1,
    "iron": 100,
    "sodium": 1,
}

SHARE_DEN = 100  # ch=0.85 → 85/100; bh=0.95 → 95/100

# Daily weight upper bound: integer model must be conservative w.r.t. float Σgrams.
# OLD (incorrect): sum(floor(g_i)) <= floor(G)  — underestimates weight → float_reject.
# NEW: sum(ceil(g_i * S)) <= floor(G * S)  with S=WEIGHT_SCALE.
# Proof: Σ g_i ≤ Σ ceil(g_i S)/S ≤ floor(G S)/S ≤ G.
WEIGHT_SCALE = 10  # deci-grams; finer than whole grams, still integer-safe


def weight_int_coeffs(grams: np.ndarray, *, scale: int = WEIGHT_SCALE) -> np.ndarray:
    """Per-row ceil coefficients for a conservative weight upper bound."""
    g = np.asarray(grams, dtype=np.float64)
    # Plain ceil — never underestimate grams. Tiny FP over-ceil is acceptable miss risk.
    return np.ceil(g * float(scale)).astype(np.int64)


def weight_int_rhs(max_daily_g: float, *, scale: int = WEIGHT_SCALE) -> int:
    # floor(G*S): never overestimate the allowed integer RHS
    return int(np.floor(float(max_daily_g) * float(scale)))


def diagnose_weight_selection(
    grams: np.ndarray,
    max_daily_g: float,
    *,
    scale: int = WEIGHT_SCALE,
) -> dict[str, Any]:
    """Dump raw vs integer weight totals for selected rows (float_reject diagnosis)."""
    g = np.asarray(grams, dtype=np.float64)
    coeffs = weight_int_coeffs(g, scale=scale)
    rhs = weight_int_rhs(max_daily_g, scale=scale)
    raw = float(g.sum())
    floor_sum = int(np.floor(g).sum())  # legacy (non-conservative) integer total
    return {
        "n_selected": int(len(g)),
        "weight_scale": int(scale),
        "raw_grams": [float(x) for x in g.tolist()],
        "int_ceil_coeffs": [int(x) for x in coeffs.tolist()],
        "legacy_floor_coeffs": [int(x) for x in np.floor(g).astype(int).tolist()],
        "raw_total_g": raw,
        "int_ceil_total": int(coeffs.sum()),
        "legacy_floor_total": floor_sum,
        "max_daily_g": float(max_daily_g),
        "int_rhs_floor_scaled": rhs,
        "legacy_rhs_floor_g": int(np.floor(float(max_daily_g))),
        "float_ok": raw <= float(max_daily_g) + 1e-9,
        "new_int_ok": int(coeffs.sum()) <= rhs,
        "legacy_int_ok": floor_sum <= int(np.floor(float(max_daily_g))),
        "overshoot_g": raw - float(max_daily_g),
    }


def scaling_report(pool: pd.DataFrame, nutrient_cols: list[str], scales: dict[str, int]) -> dict:
    out = {}
    for nut in nutrient_cols:
        scale = int(scales.get(nut, 10))
        vals = (
            pd.to_numeric(pool[nut], errors="coerce").fillna(0.0).to_numpy()
            if nut in pool.columns
            else np.zeros(len(pool))
        )
        scaled = vals * scale
        out[nut] = {
            "scale": scale,
            "max_abs_round_err": float(np.max(np.abs(scaled - np.round(scaled)))) if len(vals) else 0.0,
            "policy": "lo uses floor(v*s) >= ceil(lo*s); hi uses ceil(v*s) <= floor(hi*s)",
        }
    return out


class CPSATSolver(BaseAlgorithm):
    name = "cpsat"

    def __init__(
        self,
        *args,
        num_search_workers: int = 1,
        nutrient_scales: Optional[dict] = None,
        weight_scale: int = WEIGHT_SCALE,
        weight_rounding: str = "ceil",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_search_workers = int(num_search_workers)
        self.weight_scale = int(weight_scale)
        # "ceil" = conservative (default); "floor" = legacy non-conservative (diagnosis only)
        self.weight_rounding = str(weight_rounding or "ceil").lower()
        self.nutrient_scales = dict(DEFAULT_NUTRIENT_SCALES)
        if nutrient_scales:
            self.nutrient_scales.update({str(k): int(v) for k, v in nutrient_scales.items()})

    def solve(self, budget: SearchBudget, *, banned_food_ids: Optional[set[int]] = None) -> AlgoResult:
        try:
            from ortools.sat.python import cp_model
        except Exception as e:
            return AlgoResult(
                algorithm=self.name,
                status="error",
                failure_reason=f"ortools_unavailable:{e}",
            )
        try:
            return self._solve_impl(
                cp_model, budget, banned_food_ids=banned_food_ids or set(), nogoods=[]
            )
        except Exception as e:
            return AlgoResult(
                algorithm=self.name,
                status="error",
                failure_reason=f"exception:{type(e).__name__}:{e}",
            )

    def _active_row_indices(self, banned_food_ids: set[int]) -> np.ndarray:
        food_ids = pd.to_numeric(self.pool["food_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        ok = np.ones(len(self.pool), dtype=bool)
        if "bench_placement_allowed" in self.pool.columns:
            ok &= self.pool["bench_placement_allowed"].astype(bool).to_numpy()
        if banned_food_ids:
            ok &= ~np.isin(food_ids, list(banned_food_ids))
        return np.flatnonzero(ok)

    def _build_model(
        self,
        cp_model,
        *,
        active: np.ndarray,
        nogoods: list[frozenset],
    ):
        """Build CP model on ``active`` row indices only (fixed bans pre-excluded)."""
        model = cp_model.CpModel()
        n_all = len(self.pool)
        # Map global index -> var; inactive rows have no var (pre-excluded)
        x: dict[int, Any] = {int(i): model.NewBoolVar(f"r{i}") for i in active}
        food_ids = pd.to_numeric(self.pool["food_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        grams = pd.to_numeric(self.pool["grams"], errors="coerce").fillna(0.0).to_numpy()
        meal = (
            self.pool["meal"].astype(str).str.lower().to_numpy()
            if "meal" in self.pool.columns
            else np.array(["snack"] * n_all)
        )
        if "food_group_code" in self.pool.columns:
            majors = pd.to_numeric(self.pool["food_group_code"], errors="coerce").fillna(0).astype(int).to_numpy()
        else:
            majors = np.array([int(str(int(f)).zfill(8)[0]) if f > 0 else 0 for f in food_ids])
        kcal = (
            pd.to_numeric(self.pool["calorie"], errors="coerce").fillna(0.0).to_numpy()
            if "calorie" in self.pool.columns
            else np.zeros(n_all)
        )

        by_fid: dict[int, list[int]] = defaultdict(list)
        for i in active:
            by_fid[int(food_ids[i])].append(int(i))
        y: dict[int, Any] = {}
        for fid, idxs in by_fid.items():
            if len(idxs) == 1:
                y[fid] = x[idxs[0]]
            else:
                yf = model.NewBoolVar(f"y{fid}")
                model.AddMaxEquality(yf, [x[i] for i in idxs])
                model.Add(sum(x[i] for i in idxs) <= 1)
                y[fid] = yf

        # Cardinality on selected rows (== unique foods after per-fid constraint)
        model.Add(sum(x[i] for i in active) >= int(self.evaluator.spec.min_foods))
        model.Add(sum(x[i] for i in active) <= int(self.evaluator.spec.max_combination_size))

        weight_scale = int(getattr(self, "weight_scale", WEIGHT_SCALE))
        weight_rounding = str(getattr(self, "weight_rounding", "ceil")).lower()
        if self.evaluator.spec.hard_daily_weight:
            # Conservative upper bound (default ceil). Legacy floor underestimates Σg.
            if weight_rounding == "floor":
                g_coeff = np.floor(grams * float(weight_scale)).astype(np.int64)
            else:
                g_coeff = weight_int_coeffs(grams, scale=weight_scale)
            g_int = {i: int(g_coeff[i]) for i in active}
            model.Add(
                sum(g_int[i] * x[i] for i in active)
                <= weight_int_rhs(self.evaluator.spec.max_daily_g, scale=weight_scale)
            )

        scale_used = {}
        round_err = {}
        for nut in self.nutrient_cols:
            scale = int(self.nutrient_scales.get(nut, 10))
            scale_used[nut] = scale
            lo, hi = float(self.target_dict[nut][0]), float(self.target_dict[nut][1])
            vals = (
                pd.to_numeric(self.pool[nut], errors="coerce").fillna(0.0).to_numpy()
                if nut in self.pool.columns
                else np.zeros(n_all)
            )
            v_floor = {i: int(np.floor(vals[i] * scale)) for i in active}
            v_ceil = {i: int(np.ceil(vals[i] * scale + 1e-12)) for i in active}
            model.Add(sum(v_floor[i] * x[i] for i in active) >= int(np.ceil(lo * scale - 1e-12)))
            model.Add(sum(v_ceil[i] * x[i] for i in active) <= int(np.floor(hi * scale + 1e-12)))
            round_err[nut] = float(np.max(np.abs(vals * scale - np.round(vals * scale))))

        # hi0 presence
        if self.evaluator.spec.category_presence_hard and self.personal is not None:
            bands = self.personal.get("bands") or {}
            for key, band in bands.items():
                if band.get("hi") is None or float(band["hi"]) > 0:
                    continue
                parts = key.split("_")
                try:
                    mi = int(parts[2])
                    mname = parts[3]
                except (IndexError, ValueError):
                    continue
                for i in active:
                    if meal[i] == mname and int(majors[i]) == mi:
                        model.Add(x[i] == 0)

        # Condiment / beverage hard (same AND/OR / >= boundaries as check_condiment_oil_hard)
        unsupported: list[str] = []
        if self.evaluator.spec.check_condiment:
            hard = self.config.get("hard_operability") or {}
            codes = set(int(c) for c in hard.get("condiment_oil_food_group_codes", [8, 9]))
            ch = float(hard.get("condiment_oil_weight_share_hard", 0.85))
            bh = float(hard.get("beverage_weight_share_hard", 0.95))
            min_k = float(hard.get("condiment_min_meal_kcal", 200))
            need_n = int(hard.get("condiment_fail_n_meals", 2))
            ch_num = int(round(ch * SHARE_DEN))
            bh_num = int(round(bh * SHARE_DEN))
            min_k_i = int(np.floor(min_k))  # conservative: harder to trigger?
            # For forbidding float-infeasible: use ceil(kcal) for threshold side
            # dominated when share>=ch AND kcal>=min_k
            # Cross-multiply conservative (prefer false dominated → may miss float-feasible):
            # use ceil(cond_g), floor(meal_g) so share looks larger.
            dominated = []
            for mname in ("breakfast", "lunch", "dinner"):
                idxs = [int(i) for i in active if meal[i] == mname]
                if not idxs:
                    # empty meal → share 0, not dominated
                    continue
                # cond grams: ceil for condiment rows, 0 else
                cond_ceil = []
                meal_floor = []
                kcal_ceil = []
                for i in idxs:
                    g = float(grams[i])
                    k = float(kcal[i])
                    meal_floor.append(int(np.floor(g)) * x[i])
                    kcal_ceil.append(int(np.ceil(k + 1e-12)) * x[i])
                    if int(majors[i]) in codes:
                        cond_ceil.append(int(np.ceil(g + 1e-12)) * x[i])
                    else:
                        cond_ceil.append(0 * x[i])
                # Linear sums
                C = sum(cond_ceil) if cond_ceil else 0
                G = sum(meal_floor) if meal_floor else 0
                K = sum(kcal_ceil) if kcal_ceil else 0
                # share >= ch  <=>  C * SHARE_DEN >= ch_num * G
                # When G==0, share=0 → not dominated. Encode:
                # high_share => C*DEN >= ch_num*G; if G==0, force high_share false unless C>0 weird
                high_share = model.NewBoolVar(f"hs_{mname}")
                # C*DEN - ch_num*G >= 0  when high_share
                # Use reification with big-M
                # Max grams ~ 5000*15; use M large
                M = 10_000_000
                # high_share => C*DEN >= ch_num * G
                model.Add(C * SHARE_DEN - ch_num * G >= 0).OnlyEnforceIf(high_share)
                # not high_share => C*DEN <= ch_num * G - 1  (strict) OR G==0 handled:
                # When G=0 and C=0: 0 <= -1 fails for not high_share...
                # Better: high_share <=> (C*DEN >= ch_num*G) AND (G >= 1)
                g_pos = model.NewBoolVar(f"gpos_{mname}")
                model.Add(G >= 1).OnlyEnforceIf(g_pos)
                model.Add(G == 0).OnlyEnforceIf(g_pos.Not())
                ge = model.NewBoolVar(f"ge_{mname}")
                model.Add(C * SHARE_DEN >= ch_num * G).OnlyEnforceIf(ge)
                model.Add(C * SHARE_DEN <= ch_num * G - 1).OnlyEnforceIf(ge.Not())
                # high_share = ge AND g_pos
                model.AddBoolAnd([ge, g_pos]).OnlyEnforceIf(high_share)
                model.AddBoolOr([ge.Not(), g_pos.Not()]).OnlyEnforceIf(high_share.Not())

                kcal_ok = model.NewBoolVar(f"k_{mname}")
                model.Add(K >= min_k_i).OnlyEnforceIf(kcal_ok)
                model.Add(K <= min_k_i - 1).OnlyEnforceIf(kcal_ok.Not())

                dom = model.NewBoolVar(f"dom_{mname}")
                model.AddBoolAnd([high_share, kcal_ok]).OnlyEnforceIf(dom)
                model.AddBoolOr([high_share.Not(), kcal_ok.Not()]).OnlyEnforceIf(dom.Not())
                dominated.append(dom)

                # beverage: share >= bh AND kcal >= min_k  → forbid
                bev_ge = model.NewBoolVar(f"bevge_{mname}")
                model.Add(C * SHARE_DEN >= bh_num * G).OnlyEnforceIf(bev_ge)
                model.Add(C * SHARE_DEN <= bh_num * G - 1).OnlyEnforceIf(bev_ge.Not())
                bev_hs = model.NewBoolVar(f"bevhs_{mname}")
                model.AddBoolAnd([bev_ge, g_pos]).OnlyEnforceIf(bev_hs)
                model.AddBoolOr([bev_ge.Not(), g_pos.Not()]).OnlyEnforceIf(bev_hs.Not())
                bev = model.NewBoolVar(f"bev_{mname}")
                model.AddBoolAnd([bev_hs, kcal_ok]).OnlyEnforceIf(bev)
                model.AddBoolOr([bev_hs.Not(), kcal_ok.Not()]).OnlyEnforceIf(bev.Not())
                model.Add(bev == 0)

            if dominated:
                model.Add(sum(dominated) <= need_n - 1)
        else:
            unsupported.append("condiment_disabled_in_spec")

        # Food-id set no-goods
        all_fids = list(y.keys())
        for F in nogoods:
            F = set(int(f) for f in F)
            inside = [y[f] for f in F if f in y]
            outside = [y[f] for f in all_fids if f not in F]
            if not inside:
                continue
            if outside:
                model.Add(sum(inside) - sum(outside) <= len(inside) - 1)
            else:
                model.Add(sum(inside) <= len(inside) - 1)

        meta = {
            "scale_used": scale_used,
            "weight_scale": weight_scale if self.evaluator.spec.hard_daily_weight else None,
            "weight_policy": (
                "sum(ceil(g_i*S)) <= floor(G*S)  [conservative upper]; "
                "NOT sum(floor(g_i))<=floor(G)"
                if self.evaluator.spec.hard_daily_weight
                else None
            ),
            "round_err": round_err,
            "unsupported": unsupported,
            "n_active_rows": int(len(active)),
            "n_nogoods": len(nogoods),
            "scaling": scaling_report(self.pool.iloc[list(active)], self.nutrient_cols, self.nutrient_scales),
        }
        return model, x, meta

    def _solve_impl(
        self,
        cp_model,
        budget: SearchBudget,
        *,
        banned_food_ids: set[int],
        nogoods: list[frozenset],
    ) -> AlgoResult:
        from dmp4p_runtime.budget_clock import BudgetClock

        clock = getattr(self, "budget_clock", None) or BudgetClock(wall_s=float(budget.wall_s))
        self.budget_clock = clock
        dual = getattr(self, "dual_observers", None)

        t_init0 = clock.now()
        active = self._active_row_indices(banned_food_ids)
        if len(active) < self.evaluator.spec.min_foods:
            clock.mark_finalization_end()
            return AlgoResult(
                algorithm=self.name,
                status="incomplete",
                failure_reason="insufficient_active_rows",
                timings={"init_s": clock.now() - t_init0, "budget_overflow_s": clock.budget_overflow_s()},
                extra={"budget_clock": clock.to_dict()},
            )
        model, x, meta = self._build_model(cp_model, active=active, nogoods=nogoods)
        solver = cp_model.CpSolver()
        # Remaining time after init (shared deadline); never extend beyond wall.
        rem = max(0.01, clock.deadline - clock.now())
        solver.parameters.max_time_in_seconds = float(rem)
        solver.parameters.num_search_workers = int(self.num_search_workers)
        solver.parameters.random_seed = int(self.seed)
        solver.parameters.stop_after_first_solution = True
        t_init = clock.now() - t_init0

        t_search0 = clock.now()
        status = solver.Solve(model)
        t_solver_end = clock.now()
        clock.mark_search_end(t_solver_end)
        search_s = t_solver_end - t_search0
        status_name = solver.StatusName(status)
        first_solver_feasible_rel = None
        first_evaluator_feasible_rel = None
        try:
            from importlib.metadata import version

            ver = version("ortools")
        except Exception:
            ver = "unknown"

        solutions = []
        solutions_diag = []
        fail = None
        out_status = "unknown"
        decode_s = 0.0
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            first_solver_feasible_rel = clock.rel(t_solver_end)
            mask = np.zeros(len(self.pool), dtype=bool)
            for i, var in x.items():
                if solver.Value(var) == 1:
                    mask[i] = True
            t_dec0 = clock.now()
            ev = self.evaluator.evaluate_mask(self.pool, mask)
            t_eval_done = clock.now()
            decode_s = t_eval_done - t_dec0
            packed = self._pack_eval(
                ev, found_at_s=float(clock.rel(t_eval_done) or -1.0), eval_count=1
            )
            packed["in_budget"] = clock.in_budget(t_eval_done)
            packed["cpsat_solver_status"] = status_name
            packed["note"] = (
                "OPTIMAL/FEASIBLE means integer model status only; not structure quality optimum"
            )
            if dual is not None and ev.solution is not None:
                dual.observe_verified(
                    ev.solution,
                    feasible=bool(ev.feasible),
                    fingerprint_hash=ev.fingerprint_hash,
                    eval_completed_at=t_eval_done,
                )
            if not packed["in_budget"]:
                out_status = "timeout"
                fail = "budget_overflow_before_evaluator_accept"
                solutions_diag = [packed]
            elif ev.feasible:
                solutions = [packed]
                out_status = "ok"
                first_evaluator_feasible_rel = clock.rel(t_eval_done)
            else:
                solutions = [packed]
                out_status = "float_reject"
                fail = "cpsat_integer_feasible_float_evaluator_reject:" + ",".join(
                    ev.failure_reasons[:6]
                )
                sel_idx = np.flatnonzero(mask)
                sel_g = (
                    pd.to_numeric(self.pool.iloc[sel_idx]["grams"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
                wdiag = diagnose_weight_selection(
                    sel_g,
                    float(self.evaluator.spec.max_daily_g),
                    scale=int(getattr(self, "weight_scale", WEIGHT_SCALE)),
                )
                wdiag["selected_food_ids"] = [
                    int(self.pool.iloc[int(i)]["food_id"]) for i in sel_idx
                ]
                wdiag["selected_row_indices"] = [int(i) for i in sel_idx.tolist()]
                meta["weight_float_reject_diag"] = wdiag
        elif status == cp_model.INFEASIBLE:
            out_status = "infeasible_proved"
            fail = (
                "integer_model_infeasible_not_original_float_proof;"
                "conservative_scaling_may_shrink_feasible_region"
            )
        else:
            out_status = "timeout"
            fail = f"solver_status:{status_name}"

        clock.mark_finalization_end()
        return AlgoResult(
            algorithm=self.name,
            status=out_status,
            failure_reason=fail,
            solutions=solutions,
            timings={
                "init_s": t_init,
                "search_s": search_s,
                "decode_eval_s": decode_s,
                "init_search_decode_merged_note": (
                    "init+solver+evaluator share one deadline from solve start; "
                    "solver max_time = remaining after init"
                ),
                "budget_overflow_s": clock.budget_overflow_s(),
            },
            evaluation_count=1 if (solutions or solutions_diag) else 0,
            solver_status=status_name,
            solver_version=ver,
            extra={
                **meta,
                "role": "feasibility_only_reference",
                "first_solver_feasible_rel_s": first_solver_feasible_rel,
                "first_evaluator_feasible_rel_s": first_evaluator_feasible_rel,
                "primary_ttf_metric": "first_evaluator_feasible_rel_s",
                "budget_clock": clock.to_dict(),
                "n_post_deadline_diagnostic": len(solutions_diag),
                "post_deadline_diagnostic_solutions": solutions_diag,
                "dual_observers": dual.summary() if dual is not None else None,
                "note_infeasible": (
                    "Integer INFEASIBLE ≠ original float infeasible under conservative scaling."
                ),
                "note_optimal": (
                    "CP-SAT OPTIMAL is integer feasibility status, not Layer-B or band quality optimum."
                ),
                "status_taxonomy": {
                    "ok": "integer feasible + evaluator feasible in budget",
                    "float_reject": "integer feasible but original-precision evaluator fails",
                    "infeasible_proved": "integer model proved infeasible",
                    "timeout": "budget exhausted / UNKNOWN / eval after deadline",
                    "error": "program exception",
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
        from algo.sequential_ban_plugin import build_sequential_ban_plugin
        from dmp4p_runtime.evaluator import pairwise_jaccard

        try:
            from ortools.sat.python import cp_model
        except Exception as e:
            return AlgoResult(
                algorithm=self.name,
                status="error",
                failure_reason=f"ortools_unavailable:{e}",
            )

        plugin = ban_plugin or build_sequential_ban_plugin(self.config)
        banned: set[int] = set()
        accepted: list[dict[str, Any]] = []
        nogoods: list[frozenset] = []
        t0 = time.perf_counter()
        deadline = t0 + float(budget.wall_s)
        term = None
        per_sol_times: list[float] = []

        while len(accepted) < n_solutions:
            rem = deadline - time.perf_counter()
            if rem <= 0:
                term = "budget_exhausted"
                break
            sub = SearchBudget(
                wall_s=rem,
                checkpoints_s=budget.checkpoints_s,
                n_solutions_target=1,
                stop_at_first_feasible=True,
            )
            res = self._solve_impl(cp_model, sub, banned_food_ids=set(banned), nogoods=list(nogoods))
            if res.status == "error":
                return AlgoResult(
                    algorithm=self.name,
                    status="error",
                    failure_reason=res.failure_reason,
                    solutions=accepted,
                    timings={"search_s": time.perf_counter() - t0},
                    extra={"child": res.to_dict()},
                )
            feas = [
                s
                for s in res.solutions
                if s.get("feasible") and float(s.get("found_at_s", 0)) <= rem + 1e-9
            ]
            # found_at is relative to sub-budget start; accept if child ok
            if res.status != "ok" or not [s for s in res.solutions if s.get("feasible")]:
                term = res.failure_reason or res.status
                break
            sol = next(s for s in res.solutions if s.get("feasible"))
            # stamp cumulative time
            cum = time.perf_counter() - t0
            if cum > float(budget.wall_s):
                term = "budget_overflow_before_accept"
                break
            sol = dict(sol)
            sol["found_at_s"] = cum
            accepted.append(sol)
            per_sol_times.append(cum)
            fp = frozenset(int(x) for x in (sol.get("fingerprint_food_ids") or []))
            nogoods.append(fp)
            day = pd.DataFrame(sol.get("solution_records") or [])
            new_ban = plugin.ban_ids_after(day, pool_food_ids=self.pool["food_id"]) if len(day) else set(fp)
            banned |= set(int(x) for x in new_ban)

        status = "ok" if len(accepted) >= n_solutions else "incomplete"
        fps = [frozenset(s.get("fingerprint_food_ids") or []) for s in accepted]
        return AlgoResult(
            algorithm=self.name,
            status=status,
            failure_reason=None if status == "ok" else term,
            solutions=accepted,
            timings={"search_s": time.perf_counter() - t0},
            evaluation_count=len(accepted),
            checkpoint_metrics=self._checkpoint_counts(accepted, per_sol_times, budget),
            extra={
                "n_accepted": len(accepted),
                "time_to_k": {
                    str(k): per_sol_times[k - 1] for k in (1, 3, 7) if len(per_sol_times) >= k
                },
                "pairwise_jaccard": pairwise_jaccard(fps),
                "nogood_food_sets": [sorted(x) for x in nogoods],
                "ban_plugin": getattr(plugin, "name", None),
            },
        )
