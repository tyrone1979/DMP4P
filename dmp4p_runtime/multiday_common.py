"""
Multi-day (up to 7) system comparison harness — shared pool + food_id ban.

Protocol multiday_dev_v3 (benchmark-only online ready_plan):
  - task wall = 210s; slot cap = 30s; no first-select finalize reserve
  - ready_plan updated during observe when BLD+metrics finish before deadlines
  - commit uses immutable pre-deadline ready_plan after solve returns
  - solve overrun may delay commit but does not discard an already-ready plan
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from algo.fndds_code_policy import is_main_cuisine_by_code
from algo.sequential_ban_plugin import build_sequential_ban_plugin
from dmp4p_runtime.algorithms.base import SearchBudget
from dmp4p_runtime.best_so_far import DualObservers, OBSERVE_CACHE_VERSION
from dmp4p_runtime.budget_clock import BudgetClock
from dmp4p_runtime.candidate_ids import pool_candidate_set_hash
from dmp4p_runtime.factory import make_e1a_algorithm
from dmp4p_runtime.online_ready_plan import OnlineReadyPlan
from dmp4p_runtime.personal_band_metrics import PERSONAL_BAND_METRIC_VERSION
from dmp4p_runtime.score_fields import (
    SCORE_FIELD_VERSION,
    plan_uid_from_day,
    selected_rows_for_log,
)

METHODS = ("v2_production", "v2_quality_continuation", "nutrient_ga")
from dmp4p_runtime.factory import SUPPORTED_METHODS
DEV_TASK_WALL_S = 210.0
DEV_SLOT_CAP_S = 30.0
DEV_MAX_DAYS = 7
# v3: no arbitrary first-select reserve; online ready_plan covers selection in-slot.
DEV_SLOT_FINALIZE_RESERVE_S = 0.0
MULTIDAY_PROTOCOL_ID = "multiday_dev_v3"

FAILURE_REASONS = (
    "no_feasible_observed_in_budget",
    "feasible_but_no_bld_candidate",
    "candidate_not_ready_before_deadline",
    "ready_plan_context_mismatch",
    "task_budget_exhausted",
    "solver_error",
    "completed_target",
    "pool_too_small_after_ban",
    "commit_rebuild_failed",
)


@dataclass
class TaskBudget:
    task_wall_s: float = DEV_TASK_WALL_S
    slot_cap_s: float = DEV_SLOT_CAP_S
    t_start: float = field(default_factory=time.perf_counter)

    @property
    def task_deadline(self) -> float:
        return self.t_start + float(self.task_wall_s)

    def remaining_task_s(self, at: Optional[float] = None) -> float:
        t = time.perf_counter() if at is None else float(at)
        return max(0.0, self.task_deadline - t)

    def next_slot_wall_s(self) -> float:
        return min(float(self.slot_cap_s), self.remaining_task_s())

    def overflow_s(self, at: Optional[float] = None) -> float:
        t = time.perf_counter() if at is None else float(at)
        return max(0.0, t - self.task_deadline)


def _day_food_ids(day: pd.DataFrame) -> set[int]:
    if day is None or day.empty or "food_id" not in day.columns:
        return set()
    return set(int(x) for x in day["food_id"].tolist())


def _main_cuisine_ids(day: pd.DataFrame) -> list[int]:
    ids = []
    for fid in _day_food_ids(day):
        try:
            if is_main_cuisine_by_code(fid):
                ids.append(int(fid))
        except Exception:
            continue
    return sorted(set(ids))


def _context_fingerprint(
    *,
    structure_prior,
    config,
    personal,
    uncommon_ids,
    pool_hash: Optional[str],
) -> str:
    payload = {
        "score": SCORE_FIELD_VERSION,
        "band": PERSONAL_BAND_METRIC_VERSION,
        "observe_cache": OBSERVE_CACHE_VERSION,
        "pool_hash": pool_hash,
        "prior_id": id(structure_prior),
        "config_id": id(config),
        "personal_id": id(personal),
        "uncommon_id": id(uncommon_ids),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _rebuild_day_from_rows(pool0: pd.DataFrame, rows: list) -> Optional[pd.DataFrame]:
    if not rows:
        return None
    try:
        by = pool0.drop_duplicates("candidate_id").set_index("candidate_id", drop=False)
        day = pd.DataFrame([by.loc[str(r["candidate_id"])] for r in rows]).reset_index(
            drop=True
        )
        return day
    except Exception:
        try:
            return pd.DataFrame(rows)
        except Exception:
            return None


def _classify_slot_failure(
    *,
    dual: DualObservers,
    ready: OnlineReadyPlan,
    solve_status: Optional[str],
    solve_failure: Optional[str],
) -> str:
    if solve_status == "error" or (
        solve_failure and str(solve_failure).startswith("exception")
    ):
        return "solver_error"
    n_feas = int(dual.n_in_budget_feasible)
    if n_feas <= 0:
        return "no_feasible_observed_in_budget"
    # Feasible observed but never became ready before deadline
    if ready.ready is None:
        if ready.n_rejected_after_deadline > 0 and ready.n_ready_updates == 0:
            return "candidate_not_ready_before_deadline"
        # Observed feasible but no BLD-qualified ready (or hook never updated)
        return "feasible_but_no_bld_candidate"
    return "candidate_not_ready_before_deadline"


def run_multiday_system(
    *,
    method: str,
    snap: dict,
    evaluator,
    cfg: dict,
    device: str,
    seed: int,
    task_wall_s: float = DEV_TASK_WALL_S,
    slot_cap_s: float = DEV_SLOT_CAP_S,
    max_days: int = DEV_MAX_DAYS,
    checkpoint=None,
    resume_state: Optional[dict] = None,
    stop_requested=None,
) -> dict[str, Any]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method: {method}")

    task = TaskBudget(task_wall_s=task_wall_s, slot_cap_s=slot_cap_s)
    pool0 = snap["pool"]
    init_hash = pool_candidate_set_hash(pool0) if "candidate_id" in pool0.columns else None
    plugin = build_sequential_ban_plugin(
        {**(cfg or {}), "sequential_ban": {"plugin": "fndds", "enabled": True}}
    )

    banned: set[int] = set()
    days_out: list[dict] = []
    slot_logs: list[dict] = []
    remaining_hashes: list[str] = []
    stop_reason = None
    mains_seen: set[int] = set()
    n_main_day_repeat = 0
    n_days_with_main = 0

    resumed = resume_state or {}
    if resumed:
        if resumed.get("initial_pool_hash") != init_hash:
            raise ValueError("checkpoint pool identity mismatch")
        days_out = resumed.get("days", [])
        slot_logs = resumed.get("slots", [])
        banned = set(int(x) for x in resumed.get("banned_food_ids", []))
        expected_ban = set()
        for day_record in days_out:
            expected_ban |= set(int(x) for x in day_record["food_ids"])
        if not expected_ban.issubset(banned):
            raise ValueError("checkpoint missing committed food bans")
        remaining_hashes = resumed.get("remaining_hashes", [])
        mains_seen = set(x for d in days_out for x in d.get("main_cuisine_food_ids", []))
        n_days_with_main = sum(bool(d.get("has_main_cuisine")) for d in days_out)
        consumed = float(resumed.get("consumed_s", 0))
        if consumed < 0: raise ValueError("negative checkpoint budget")
        task.t_start = time.perf_counter() - consumed

    def save_boundary(phase):
        if checkpoint is None: return
        checkpoint.save({"phase":phase,"initial_pool_hash":init_hash,
            "days":days_out,"slots":slot_logs,"banned_food_ids":sorted(banned),
            "remaining_hashes":remaining_hashes,"consumed_s":time.perf_counter()-task.t_start,
            "task_wall_s":task_wall_s,"slot_cap_s":slot_cap_s,
            "next_slot":len(days_out)+1,"stop_reason":stop_reason,
            "recovery_note":resumed.get("recovery_note")})

    if resumed.get("phase") == "terminal":
        stop_reason = resumed.get("stop_reason")
    resume_slot = int(max_days) + 1 if resumed.get("phase") == "terminal" else len(days_out) + 1
    for slot_i in range(resume_slot, int(max_days) + 1):
        if stop_requested is not None and stop_requested():
            save_boundary("boundary")
            raise InterruptedError("graceful_pause_at_slot_boundary")
        rem = task.remaining_task_s()
        if rem <= 1e-6:
            stop_reason = "task_budget_exhausted"
            break
        slot_wall = task.next_slot_wall_s()
        if slot_wall <= 1e-6:
            stop_reason = "task_budget_exhausted"
            break

        remain_pool = pool0[~pool0["food_id"].astype(int).isin(banned)].reset_index(drop=True)
        rem_hash = (
            pool_candidate_set_hash(remain_pool)
            if "candidate_id" in remain_pool.columns
            else None
        )
        if rem_hash:
            remaining_hashes.append(rem_hash)
        if len(remain_pool) < getattr(evaluator.spec, "min_foods", 3):
            stop_reason = "pool_too_small_after_ban"
            slot_logs.append(
                {
                    "slot": slot_i,
                    "status": "stop",
                    "failure_class": "pool_too_small_after_ban",
                    "remaining_candidate_set_hash": rem_hash,
                    "n_banned": len(banned),
                }
            )
            break

        t_slot0 = time.perf_counter()
        slot_deadline_abs = min(t_slot0 + float(slot_wall), task.task_deadline)
        # Full slot wall for search+observe (no first-select reserve in v3)
        slot_clock = BudgetClock(wall_s=float(slot_wall))
        # Clamp if task ends sooner
        if slot_clock.deadline > task.task_deadline:
            slot_clock = BudgetClock(
                wall_s=max(0.5, task.task_deadline - time.perf_counter())
            )
            slot_deadline_abs = min(slot_deadline_abs, task.task_deadline)

        ctx_fp = _context_fingerprint(
            structure_prior=snap.get("structure_prior") or {},
            config=cfg,
            personal=snap.get("personal"),
            uncommon_ids=getattr(evaluator, "uncommon_ids", None),
            pool_hash=rem_hash,
        )
        ready = OnlineReadyPlan(
            slot_deadline_abs=float(slot_deadline_abs),
            task_deadline_abs=float(task.task_deadline),
            context_fingerprint=ctx_fp,
            pool_candidate_set_hash=rem_hash,
            clock_now=slot_clock.now,
        )

        def _on_quality_ready(
            plan_uid,
            pack,
            rows,
            *,
            eval_completed_at,
            quality_completed_at,
            fingerprint_hash=None,
            quality_from_cache=False,
        ):
            ready.consider(
                plan_uid=str(plan_uid),
                metrics=pack,
                selected_rows=rows,
                fingerprint_hash=fingerprint_hash,
                eval_completed_at=float(eval_completed_at),
                quality_completed_at=float(quality_completed_at),
                evaluator_feasible=True,
                quality_from_cache=bool(quality_from_cache),
            )

        dual = DualObservers(
            clock=slot_clock,
            structure_prior=snap.get("structure_prior") or {},
            uncommon_ids=getattr(evaluator, "uncommon_ids", None),
            config=cfg,
            personal=snap.get("personal"),
            on_quality_ready=_on_quality_ready,
        )
        algo = make_e1a_algorithm(
            method,
            snap=snap,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            seed=int(seed),
            v2_population_size=1000,
            ga_population_size=200,
        )
        algo.budget_clock = slot_clock
        algo.dual_observers = dual

        budget = SearchBudget(
            wall_s=float(slot_clock.wall_s),
            n_solutions_target=1,
            stop_at_first_feasible=(method == "v2_production"),
        )
        save_boundary("inflight")
        res = algo.solve(budget, banned_food_ids=set(banned))
        t_solve_return = time.perf_counter()
        slot_clock.mark_search_end(t_solve_return)

        ready_snap = ready.snapshot()
        slot_rec: dict[str, Any] = {
            "slot": slot_i,
            "slot_wall_s_allocated": float(slot_wall),
            "finalize_reserve_s": 0.0,
            "slot_elapsed_s": time.perf_counter() - t_slot0,
            "task_elapsed_s": time.perf_counter() - task.t_start,
            "task_remaining_s_after": task.remaining_task_s(),
            "solver_status": res.status,
            "solver_diagnostics": {key:res.extra.get(key) for key in ("objective", "objective_scale", "integer_band_objective", "integer_band_best_bound", "relative_gap", "max_abs_objective_error_g", "raw_layer_b_optimized", "observation_scope", "common_observe_generations", "common_observe_evaluations") if key in res.extra},
            "quality_search_stats": (res.extra or {}).get("quality_search_stats"),
            "quality_rejection_reasons": (res.extra or {}).get("quality_rejection_reasons"),
            "solver_failure_reason": res.failure_reason,
            "termination_reason": (res.extra or {}).get("termination_reason"),
            "n_observe_calls": dual.n_observe_calls,
            "n_unique_plans_logged": len(dual.unique_plan_uids_in_budget),
            "n_in_budget_feasible_observations": dual.n_in_budget_feasible,
            "observe_overhead_s": dual.observe_overhead_s,
            "observe_cache": dual.cache_stats(),
            "budget_clock_slot": slot_clock.to_dict(),
            "remaining_candidate_set_hash_before": rem_hash,
            "n_banned_before": len(banned),
            "ready_tracker": ready.summary(),
            "solve_return_at_rel_task_s": t_solve_return - task.t_start,
            "solve_return_at_rel_slot_s": t_solve_return - t_slot0,
            "slot_deadline_abs": slot_deadline_abs,
            "task_deadline_abs": task.task_deadline,
        }

        if ready_snap is None:
            fail = _classify_slot_failure(
                dual=dual,
                ready=ready,
                solve_status=res.status,
                solve_failure=res.failure_reason,
            )
            slot_rec["status"] = "no_commit"
            slot_rec["failure_class"] = fail
            slot_rec["reason"] = fail
            slot_logs.append(slot_rec)
            stop_reason = fail
            break

        # Context must still match this slot's pool/config
        if ready_snap.context_fingerprint != ctx_fp or (
            rem_hash
            and ready_snap.pool_candidate_set_hash
            and ready_snap.pool_candidate_set_hash != rem_hash
        ):
            slot_rec["status"] = "no_commit"
            slot_rec["failure_class"] = "ready_plan_context_mismatch"
            slot_rec["reason"] = "ready_plan_context_mismatch"
            slot_logs.append(slot_rec)
            stop_reason = "ready_plan_context_mismatch"
            break

        # ready_at must have been before deadlines (enforced at consider time)
        if ready_snap.ready_at > slot_deadline_abs + 1e-12 or ready_snap.ready_at > task.task_deadline + 1e-12:
            slot_rec["status"] = "no_commit"
            slot_rec["failure_class"] = "candidate_not_ready_before_deadline"
            slot_rec["reason"] = "ready_at_after_deadline"
            slot_logs.append(slot_rec)
            stop_reason = "candidate_not_ready_before_deadline"
            break

        day = _rebuild_day_from_rows(pool0, ready_snap.selected_rows)
        if day is None or day.empty:
            slot_rec["status"] = "no_commit"
            slot_rec["failure_class"] = "commit_rebuild_failed"
            slot_rec["reason"] = "empty_day"
            slot_logs.append(slot_rec)
            stop_reason = "commit_rebuild_failed"
            break

        uid_now = plan_uid_from_day(day)
        if uid_now != ready_snap.plan_uid:
            slot_rec["status"] = "no_commit"
            slot_rec["failure_class"] = "ready_plan_context_mismatch"
            slot_rec["reason"] = "plan_uid_mismatch_on_rebuild"
            slot_rec["ready_plan_uid"] = ready_snap.plan_uid
            slot_rec["rebuilt_plan_uid"] = uid_now
            slot_logs.append(slot_rec)
            stop_reason = "ready_plan_context_mismatch"
            break

        t_commit = time.perf_counter()
        # Commit/ban/log may overflow slot/task; that is delivery delay, not loss of ready.
        delivery_delay_s = max(0.0, t_commit - ready_snap.ready_at)
        slot_overflow_at_commit = max(0.0, t_commit - slot_deadline_abs)
        task_overflow_at_commit = max(0.0, t_commit - task.task_deadline)

        # Do not start ban update after task deadline for a NEW day if task already
        # exhausted — but ready was before deadline, so we still accept this day.
        # Task deadline gate for *starting new slots* is at loop top.

        mains = _main_cuisine_ids(day)
        if mains:
            n_days_with_main += 1
            if set(mains) & mains_seen:
                n_main_day_repeat += 1
            mains_seen |= set(mains)

        day_foods = _day_food_ids(day)
        new_ban = plugin.ban_ids_after(day, pool_food_ids=pool0["food_id"])
        banned |= set(int(x) for x in new_ban) | day_foods
        t_ban_done = time.perf_counter()
        slot_clock.mark_finalization_end(t_ban_done)

        metrics = ready_snap.metrics
        day_rec = {
            "day_index": slot_i,
            "plan_uid": ready_snap.plan_uid,
            "fingerprint_hash": ready_snap.fingerprint_hash,
            "eval_completed_at": ready_snap.eval_completed_at,
            "quality_completed_at": ready_snap.quality_completed_at,
            "selection_completed_at": ready_snap.selection_completed_at,
            "ready_at": ready_snap.ready_at,
            "ready_at_rel_slot_s": ready_snap.ready_at - t_slot0,
            "ready_at_rel_task_s": ready_snap.ready_at - task.t_start,
            "solve_return_at_rel_task_s": t_solve_return - task.t_start,
            "commit_at": t_commit,
            "commit_t_rel_task_s": t_commit - task.t_start,
            "commit_t_rel_slot_s": t_commit - t_slot0,
            "delivery_delay_after_ready_s": delivery_delay_s,
            "slot_overflow_at_commit_s": slot_overflow_at_commit,
            "task_overflow_at_commit_s": task_overflow_at_commit,
            "prepared_in_budget": True,
            "response_within_slot_cap": t_commit <= slot_deadline_abs + 1e-12,
            "raw_layer_b": ready_snap.raw_layer_b,
            "layer_b_components": (metrics or {}).get("layer_b_components"),
            "personal_band": (metrics or {}).get("personal_band"),
            "all_bands_simultaneous_pass": (metrics or {}).get(
                "all_bands_simultaneous_pass"
            ),
            "bld_min_weight_pass": True,
            "meal_grams": (metrics or {}).get("meal_grams"),
            "selected_rows": ready_snap.selected_rows,
            "food_ids": sorted(day_foods),
            "main_cuisine_food_ids": mains,
            "has_main_cuisine": bool(mains),
            "quality_from_cache": ready_snap.quality_from_cache,
            "context_fingerprint": ready_snap.context_fingerprint,
        }
        days_out.append(day_rec)
        slot_rec.update(
            {
                "status": "committed",
                "failure_class": None,
                "committed_plan_uid": ready_snap.plan_uid,
                "ready_at_rel_slot_s": day_rec["ready_at_rel_slot_s"],
                "commit_at_rel_slot_s": day_rec["commit_t_rel_slot_s"],
                "delivery_delay_after_ready_s": delivery_delay_s,
                "response_within_slot_cap": day_rec["response_within_slot_cap"],
                "n_banned_after": len(banned),
                "finalization_end_rel_slot_s": t_ban_done - t_slot0,
            }
        )
        slot_logs.append(slot_rec)
        save_boundary("boundary")

        # After commit work, if task exhausted, do not start another slot
        if task.remaining_task_s() <= 1e-6 and slot_i < max_days:
            # Will exit at next loop iteration via task_budget_exhausted
            pass

    save_boundary("terminal")
    t_end = time.perf_counter()
    n_delivered = len(days_out)
    n_all_bands = sum(1 for d in days_out if d.get("all_bands_simultaneous_pass") is True)
    food_sets = [set(d.get("food_ids") or []) for d in days_out]
    cross_food_dup = any(
        food_sets[i] & food_sets[j]
        for i in range(len(food_sets))
        for j in range(i + 1, len(food_sets))
    )
    commit_times = [d["commit_t_rel_task_s"] for d in days_out]

    def t_k(k: int):
        return commit_times[k - 1] if len(commit_times) >= k else None

    if n_delivered >= max_days:
        stop_reason = "completed_target"

    # Process audit: ready existed but lost to post-solve select? should be impossible in v3
    lost_ready = []
    for s in slot_logs:
        rt = s.get("ready_tracker") or {}
        if rt.get("has_ready_plan") and s.get("status") == "no_commit":
            if s.get("failure_class") not in (
                "ready_plan_context_mismatch",
                "commit_rebuild_failed",
            ):
                lost_ready.append(
                    {
                        "slot": s.get("slot"),
                        "failure_class": s.get("failure_class"),
                        "ready_uid": rt.get("ready_plan_uid"),
                    }
                )

    return {
        "checkpoint_recovery_note": resumed.get("recovery_note"),
        "method": method,
        "seed": int(seed),
        "protocol_id": MULTIDAY_PROTOCOL_ID,
        "task_wall_s": float(task_wall_s),
        "slot_cap_s": float(slot_cap_s),
        "max_days": int(max_days),
        "status": "ok" if n_delivered >= max_days else "incomplete",
        "stop_reason": stop_reason,
        "n_days_delivered": n_delivered,
        "ge1": n_delivered >= 1,
        "ge3": n_delivered >= 3,
        "ge7": n_delivered >= 7,
        "n_all_bands_days": n_all_bands,
        "all_bands_rate_among_delivered_days": (
            n_all_bands / n_delivered if n_delivered else None
        ),
        "seven_all_delivered_and_all_bands_every_day": bool(
            n_delivered >= 7 and n_all_bands >= 7
        ),
        "cross_day_food_id_overlap": cross_food_dup,
        "n_days_with_main_cuisine": n_days_with_main,
        "n_days_with_main_cuisine_repeat_vs_prior": n_main_day_repeat,
        "t1_s": t_k(1),
        "t3_s": t_k(3),
        "t7_s": t_k(7),
        "t1_censored": t_k(1) is None,
        "t3_censored": t_k(3) is None,
        "t7_censored": t_k(7) is None,
        "wall_total_s": t_end - task.t_start,
        "task_budget_overflow_s": task.overflow_s(t_end),
        "candidate_set_hash_initial": init_hash,
        "remaining_candidate_set_hashes": remaining_hashes,
        "banned_food_ids_final": sorted(banned),
        "ban_plugin": getattr(plugin, "name", None),
        "days": days_out,
        "slots": slot_logs,
        "primary_return_strategy": "bld_band_lex_v1",
        "online_ready_plan": True,
        "process_audit": {
            "ready_but_lost_on_commit": lost_ready,
            "note": (
                "prepared_in_budget vs response_within_slot_cap are distinct; "
                "solve overrun may delay commit without discarding ready_plan"
            ),
        },
        "budget_policy": {
            "task_wall_s": task_wall_s,
            "slot_cap_s": slot_cap_s,
            "slot_finalize_reserve_s": 0.0,
            "online_ready_plan": True,
            "unused_time_does_not_raise_slot_cap": True,
            "no_backtrack": True,
            "no_extra_retry": True,
            "stop_on_no_ready_plan": True,
        },
    }
