"""Shared algorithm interfaces for the Bronze search benchmark."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from algo.sequential_ban_plugin import SequentialBanPlugin, build_sequential_ban_plugin
from dmp4p_runtime.evaluator import EvalResult, UnifiedEvaluator, pairwise_jaccard


@dataclass
class SearchBudget:
    """Wall-clock search budget (seconds) for one unit×seed×task run."""

    wall_s: float
    checkpoints_s: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0, 300.0)
    n_solutions_target: int = 1  # task1=1, task2=7
    stop_at_first_feasible: bool = True
    max_evaluations: Optional[int] = None  # optional GA aux budget


@dataclass
class AlgoResult:
    algorithm: str
    status: str  # ok | timeout | infeasible_proved | unknown | error | incomplete
    failure_reason: Optional[str]
    solutions: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    evaluation_count: int = 0
    solver_status: Optional[str] = None
    solver_version: Optional[str] = None
    checkpoint_metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "solutions": self.solutions,
            "timings": self.timings,
            "evaluation_count": int(self.evaluation_count),
            "solver_status": self.solver_status,
            "solver_version": self.solver_version,
            "checkpoint_metrics": self.checkpoint_metrics,
            "extra": self.extra,
        }


class BaseAlgorithm(ABC):
    name: str = "base"

    def __init__(
        self,
        pool: pd.DataFrame,
        target_dict: dict,
        evaluator: UnifiedEvaluator,
        *,
        config: Optional[dict] = None,
        cell_ranges: Optional[dict] = None,
        personal: Optional[dict] = None,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.pool = pool.reset_index(drop=True)
        self.target_dict = target_dict
        self.evaluator = evaluator
        self.config = config or {}
        self.cell_ranges = cell_ranges
        self.personal = personal
        self.device = device
        self.seed = int(seed)
        self.n_rows = len(self.pool)
        self.nutrient_cols = list(target_dict.keys())

    @abstractmethod
    def solve(self, budget: SearchBudget, *, banned_food_ids: Optional[set[int]] = None) -> AlgoResult:
        """Task-1 style: find up to n_solutions under cumulative wall budget."""

    def solve_sequential(
        self,
        budget: SearchBudget,
        *,
        ban_plugin: Optional[SequentialBanPlugin] = None,
        n_solutions: int = 7,
    ) -> AlgoResult:
        """
        Task-2: accumulate up to n_solutions with shared ban plugin.

        Budget is cumulative across the whole unit (not reset per solution).
        Generic methods may restart; ban set grows after each accepted solution.
        """
        plugin = ban_plugin or build_sequential_ban_plugin(self.config)
        banned: set[int] = set()
        accepted: list[dict[str, Any]] = []
        t_search0 = time.perf_counter()
        eval_count = 0
        deadline = t_search0 + float(budget.wall_s)
        term_reason = None
        per_sol_times: list[float] = []
        remaining_hashes: list[str] = []

        while len(accepted) < n_solutions:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                term_reason = "budget_exhausted"
                break
            sub = SearchBudget(
                wall_s=remaining,
                checkpoints_s=budget.checkpoints_s,
                n_solutions_target=1,
                stop_at_first_feasible=True,
                max_evaluations=budget.max_evaluations,
            )
            res = self.solve(sub, banned_food_ids=set(banned))
            eval_count += int(res.evaluation_count)
            if res.status == "error":
                return AlgoResult(
                    algorithm=self.name,
                    status="error",
                    failure_reason=res.failure_reason,
                    solutions=accepted,
                    timings={"search_s": time.perf_counter() - t_search0},
                    evaluation_count=eval_count,
                    extra={"sequential_stop": "child_error", "child": res.to_dict()},
                )
            feas = [s for s in res.solutions if s.get("feasible")]
            if not feas:
                term_reason = res.failure_reason or res.status or "no_feasible"
                break
            sol = feas[0]
            accepted.append(sol)
            per_sol_times.append(time.perf_counter() - t_search0)
            day = pd.DataFrame(sol.get("solution_records") or [])
            if day.empty and sol.get("fingerprint_food_ids"):
                # ban by fingerprint ids even without full records
                banned |= set(int(x) for x in sol["fingerprint_food_ids"])
            else:
                new_ban = plugin.ban_ids_after(day, pool_food_ids=self.pool["food_id"])
                banned |= set(int(x) for x in new_ban)
            # pool emptiness check — delete banned from fixed snapshot only
            remain_rows = self.pool[~self.pool["food_id"].astype(int).isin(banned)]
            rem_hash = None
            if "candidate_id" in remain_rows.columns:
                from dmp4p_runtime.candidate_ids import pool_candidate_set_hash

                rem_hash = pool_candidate_set_hash(remain_rows)
                remaining_hashes.append(rem_hash)
            if len(remain_rows) < self.evaluator.spec.min_foods:
                term_reason = "pool_too_small_after_ban"
                break
            meals = set(remain_rows.get("meal", pd.Series(dtype=str)).astype(str).str.lower())
            if not {"breakfast", "lunch", "dinner"}.issubset(meals):
                term_reason = "bld_meals_empty_after_ban"
                break

        search_s = time.perf_counter() - t_search0
        fps = [frozenset(s.get("fingerprint_food_ids") or []) for s in accepted]
        status = "ok" if len(accepted) >= n_solutions else "incomplete"
        if not accepted and term_reason:
            status = "timeout" if "budget" in str(term_reason) else "incomplete"
        init_hash = None
        if "candidate_id" in self.pool.columns:
            from dmp4p_runtime.candidate_ids import pool_candidate_set_hash

            init_hash = pool_candidate_set_hash(self.pool)
        return AlgoResult(
            algorithm=self.name,
            status=status,
            failure_reason=None if status == "ok" else term_reason,
            solutions=accepted,
            timings={"search_s": search_s},
            evaluation_count=eval_count,
            checkpoint_metrics=self._checkpoint_counts(accepted, per_sol_times, budget),
            extra={
                "n_accepted": len(accepted),
                "time_to_k": {str(k): per_sol_times[k - 1] for k in (1, 3, 7) if len(per_sol_times) >= k},
                "pairwise_jaccard": pairwise_jaccard(fps),
                "banned_food_ids_final": sorted(banned),
                "ban_plugin": getattr(plugin, "name", None),
                "candidate_set_hash_initial": init_hash,
                "remaining_candidate_set_hashes": remaining_hashes,
            },
        )

    @staticmethod
    def _checkpoint_counts(
        accepted: list[dict],
        per_sol_times: list[float],
        budget: SearchBudget,
    ) -> dict[str, Any]:
        """
        Checkpoint counts. If the run finished early with K solutions, later
        checkpoints still report K (carry-forward), not missing/timeout.
        """
        out = {}
        finished_early = len(accepted) >= int(budget.n_solutions_target)
        final_n = len(per_sol_times)
        for cp in budget.checkpoints_s:
            n = sum(1 for t in per_sol_times if t <= float(cp))
            if finished_early and float(cp) >= (per_sol_times[-1] if per_sol_times else 0):
                n = final_n
            out[f"n_unique_feasible_at_{int(cp)}s"] = n
        out["time_to_k"] = {
            str(k): per_sol_times[k - 1] for k in (1, 3, 7) if len(per_sol_times) >= k
        }
        return out

    def _pack_eval(self, ev, *, found_at_s: float, eval_count: int) -> dict[str, Any]:
        d = ev.to_dict(include_solution=True)
        d["found_at_s"] = float(found_at_s)
        d["evaluation_count_at_find"] = int(eval_count)
        return d
