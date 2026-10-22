"""Conservative hard constraints plus a rounded linear personal-band objective.
All improving incumbents are independently verified online within the budget.
"""
import time
import numpy as np
import pandas as pd
from typing import Any
from dmp4p_runtime.algorithms.cpsat import CPSATSolver, diagnose_weight_selection
from dmp4p_runtime.algorithms.base import AlgoResult,SearchBudget
from algo.structure_category import UPPER_ONLY_MAJORS

class CPSATBand(CPSATSolver):
    name="cpsat_band"
    def _build_model(self,cp_model,*,active,nogoods):
        model,x,meta=super()._build_model(cp_model,active=active,nogoods=nogoods)
        scale=10
        grams=self.pool.grams.to_numpy(dtype=float)
        meals=self.pool.meal.astype(str).str.lower().to_numpy()
        majors=self.pool.food_group_code.astype(int).to_numpy()
        # BLD is the common delivery gate; imposing it directly searches only
        # deliverable plans. Other methods may traverse nondeliverable intermediates.
        for meal,minimum in (("breakfast",30.0),("lunch",50.0),("dinner",50.0)):
            model.Add(sum(int(np.floor(grams[i]*scale))*x[int(i)] for i in active if meals[i]==meal)>=int(np.ceil(minimum*scale)))
        terms=[];upper_only=set((self.personal or {}).get("upper_only_majors") or UPPER_ONLY_MAJORS)
        rounding_error=0.0
        for key,bounds in ((self.personal or {}).get("bands") or {}).items():
            _,_,major,meal=key.split("_",3);major=int(major)
            hi=float(bounds["hi"]);lo=0.0 if major in upper_only else float(bounds["lo"])
            if hi<=0:continue
            ids=[int(i) for i in active if meals[i]==meal and majors[i]==major]
            g=sum(int(np.rint(grams[i]*scale))*x[i] for i in ids)
            upper=int(np.rint(hi*scale));lower=int(np.rint(lo*scale))
            cap=max(1,sum(abs(int(np.rint(grams[i]*scale))) for i in ids)+abs(upper)+abs(lower))
            over=model.NewIntVar(0,cap,"band_over_"+key);model.Add(over>=g-upper);terms.append(over)
            if lo>0:
                under=model.NewIntVar(0,cap,"band_under_"+key);model.Add(under>=lower-g);terms.append(under)
            rounding_error+=0.5/scale*(1+(lo>0))
        model.Minimize(sum(terms))
        # Lipschitz bound on the sum of cell deviations: each row belongs to
        # one cell; a two-sided interval distance remains 1-Lipschitz.
        self.objective_meta={"objective":"sum_scaled_cell_interval_deviation",
            "objective_scale":scale,"objective_terms":len(terms),
            "max_abs_objective_error_g":self.evaluator.spec.max_combination_size*0.5/scale+rounding_error,
            "raw_layer_b_optimized":False,"bld_modeled_as_delivery_gate":True}
        return model,x,meta

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
        solver.parameters.stop_after_first_solution = False
        t_init = clock.now() - t_init0

        t_search0 = clock.now()
        outer = self
        class Incumbent(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                if not clock.in_budget():
                    self.StopSearch(); return
                mask = np.zeros(len(outer.pool),dtype=bool)
                for i,var in x.items(): mask[int(i)]=bool(self.Value(var))
                ev=outer.evaluator.evaluate_mask(outer.pool,mask)
                done=clock.now()
                if dual is not None and ev.feasible and ev.solution is not None:
                    dual.observe_verified(ev.solution,feasible=True,fingerprint_hash=ev.fingerprint_hash,eval_completed_at=done)
                if not clock.in_budget(): self.StopSearch()
        status = solver.Solve(model, Incumbent())
        meta.update(self.objective_meta)
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            val=float(solver.ObjectiveValue());bound=float(solver.BestObjectiveBound())
            meta.update(integer_band_objective=val,integer_band_best_bound=bound,
                        relative_gap=abs(val-bound)/max(1.0,abs(val)))
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
                "role": "band_surrogate_optimization_reference",
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
                    "OPTIMAL certifies the scaled integer band surrogate only; not float band or Layer-B optimality."
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
