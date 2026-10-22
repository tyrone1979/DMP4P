"""
Benchmark-only online ready_plan for multi-day slots.

Maintains the best BLD-qualified observed plan whose eval + quality + selection
all finished before slot and task deadlines. Immutable snapshots for commit.
Does not change production search / breeding.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd


def _now(clock_now: Optional[Callable[[], float]] = None) -> float:
    return float(clock_now() if clock_now is not None else time.perf_counter())


def _band_dev(pack: dict) -> Optional[float]:
    b = pack.get("personal_band") or {}
    v = pack.get("total_band_deviation_g")
    if v is None:
        v = b.get("total_band_deviation_g")
    return None if v is None else float(v)


def _raw_lb(pack: dict) -> Optional[float]:
    v = pack.get("raw_layer_b")
    if v is None:
        v = pack.get("layer_b_total")
    return None if v is None else float(v)


def _lex_key(plan_uid: str, pack: dict) -> tuple:
    """bld_band_lex_v1: band asc, raw_lb desc, plan_uid lex."""
    dev = _band_dev(pack)
    lb = _raw_lb(pack)
    if dev is None or lb is None:
        return (float("inf"), 0.0, str(plan_uid))
    return (float(dev), -float(lb), str(plan_uid))


@dataclass
class ReadyPlanSnapshot:
    """Immutable ready plan (value copy; safe after later observes)."""

    plan_uid: str
    selected_rows: list
    metrics: dict
    fingerprint_hash: Optional[str]
    eval_completed_at: float
    quality_completed_at: float
    selection_completed_at: float
    ready_at: float
    context_fingerprint: str
    pool_candidate_set_hash: Optional[str]
    quality_from_cache: bool
    evaluator_feasible: bool
    bld_min_weight_pass: bool
    raw_layer_b: Optional[float]
    total_band_deviation_g: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_uid": self.plan_uid,
            "selected_rows": copy.deepcopy(self.selected_rows),
            "metrics": copy.deepcopy(self.metrics),
            "fingerprint_hash": self.fingerprint_hash,
            "eval_completed_at": self.eval_completed_at,
            "quality_completed_at": self.quality_completed_at,
            "selection_completed_at": self.selection_completed_at,
            "ready_at": self.ready_at,
            "context_fingerprint": self.context_fingerprint,
            "pool_candidate_set_hash": self.pool_candidate_set_hash,
            "quality_from_cache": self.quality_from_cache,
            "evaluator_feasible": self.evaluator_feasible,
            "bld_min_weight_pass": self.bld_min_weight_pass,
            "raw_layer_b": self.raw_layer_b,
            "total_band_deviation_g": self.total_band_deviation_g,
        }


@dataclass
class OnlineReadyPlan:
    """
    Updates only when quality+selection finish before deadlines.
    Post-deadline better candidates → diagnostics only.
    """

    slot_deadline_abs: float
    task_deadline_abs: float
    context_fingerprint: str
    pool_candidate_set_hash: Optional[str] = None
    clock_now: Optional[Callable[[], float]] = None

    ready: Optional[ReadyPlanSnapshot] = None
    n_consider_calls: int = 0
    n_ready_updates: int = 0
    n_rejected_after_deadline: int = 0
    n_rejected_not_bld: int = 0
    n_rejected_incomplete_metrics: int = 0
    post_deadline_diagnostics: list[dict] = field(default_factory=list)

    def in_deadlines(self, at: Optional[float] = None) -> bool:
        t = _now(self.clock_now) if at is None else float(at)
        return t <= self.slot_deadline_abs + 1e-12 and t <= self.task_deadline_abs + 1e-12

    def consider(
        self,
        *,
        plan_uid: str,
        metrics: dict,
        selected_rows: Optional[list],
        fingerprint_hash: Optional[str],
        eval_completed_at: float,
        quality_completed_at: float,
        evaluator_feasible: bool = True,
        quality_from_cache: bool = False,
    ) -> dict[str, Any]:
        """
        Call after quality pack is fully available.
        ``quality_completed_at`` / selection time must be actual wall moments — never backdated.
        """
        self.n_consider_calls += 1
        t_sel0 = _now(self.clock_now)
        # Selection work is the lex compare + snapshot copy below
        pack = metrics or {}
        bld = bool(pack.get("bld_min_weight_pass"))
        if not evaluator_feasible:
            self.n_rejected_not_bld += 1
            return {"updated": False, "reason": "not_evaluator_feasible"}
        if not bld:
            self.n_rejected_not_bld += 1
            return {"updated": False, "reason": "feasible_but_no_bld"}
        if _band_dev(pack) is None or _raw_lb(pack) is None:
            self.n_rejected_incomplete_metrics += 1
            return {"updated": False, "reason": "incomplete_metrics"}
        if selected_rows is None:
            self.n_rejected_incomplete_metrics += 1
            return {"updated": False, "reason": "missing_selected_rows"}

        # Gate on eval, quality, and selection completion times (actual wall moments).
        if not self.in_deadlines(eval_completed_at):
            self.n_rejected_after_deadline += 1
            self.post_deadline_diagnostics.append(
                {
                    "plan_uid": plan_uid,
                    "eval_completed_at": eval_completed_at,
                    "reason": "eval_after_deadline",
                    "raw_layer_b": _raw_lb(pack),
                    "band_dev": _band_dev(pack),
                }
            )
            return {"updated": False, "reason": "eval_after_deadline"}

        if not self.in_deadlines(quality_completed_at):
            self.n_rejected_after_deadline += 1
            self.post_deadline_diagnostics.append(
                {
                    "plan_uid": plan_uid,
                    "quality_completed_at": quality_completed_at,
                    "reason": "quality_after_deadline",
                    "raw_layer_b": _raw_lb(pack),
                    "band_dev": _band_dev(pack),
                }
            )
            return {"updated": False, "reason": "quality_after_deadline"}

        t_sel_done = _now(self.clock_now)
        if not self.in_deadlines(t_sel_done):
            self.n_rejected_after_deadline += 1
            self.post_deadline_diagnostics.append(
                {
                    "plan_uid": plan_uid,
                    "selection_completed_at": t_sel_done,
                    "reason": "selection_after_deadline",
                    "raw_layer_b": _raw_lb(pack),
                    "band_dev": _band_dev(pack),
                }
            )
            return {"updated": False, "reason": "selection_after_deadline"}

        cand_key = _lex_key(plan_uid, pack)
        if self.ready is not None:
            cur_key = _lex_key(self.ready.plan_uid, self.ready.metrics)
            if cand_key >= cur_key:
                return {"updated": False, "reason": "not_better_than_ready"}

        snap = ReadyPlanSnapshot(
            plan_uid=str(plan_uid),
            selected_rows=copy.deepcopy(list(selected_rows)),
            metrics=copy.deepcopy(dict(pack)),
            fingerprint_hash=fingerprint_hash,
            eval_completed_at=float(eval_completed_at),
            quality_completed_at=float(quality_completed_at),
            selection_completed_at=float(t_sel_done),
            ready_at=float(t_sel_done),  # actual moment this plan became ready
            context_fingerprint=self.context_fingerprint,
            pool_candidate_set_hash=self.pool_candidate_set_hash,
            quality_from_cache=bool(quality_from_cache),
            evaluator_feasible=True,
            bld_min_weight_pass=True,
            raw_layer_b=_raw_lb(pack),
            total_band_deviation_g=_band_dev(pack),
        )
        # Copying rows/metrics is part of preparing the immutable delivery.
        # Do not backdate readiness to the pre-copy deadline check.
        completed_at = _now(self.clock_now)
        if not self.in_deadlines(completed_at):
            self.n_rejected_after_deadline += 1
            self.post_deadline_diagnostics.append({
                "plan_uid": plan_uid,
                "selection_completed_at": completed_at,
                "reason": "snapshot_after_deadline",
            })
            return {"updated": False, "reason": "snapshot_after_deadline"}
        snap.selection_completed_at = completed_at
        snap.ready_at = completed_at
        self.ready = snap
        self.n_ready_updates += 1
        return {
            "updated": True,
            "plan_uid": snap.plan_uid,
            "ready_at": snap.ready_at,
            "selection_elapsed_s": completed_at - t_sel0,
        }

    def snapshot(self) -> Optional[ReadyPlanSnapshot]:
        if self.ready is None:
            return None
        # Return a deep copy so later observes cannot mutate committed state
        return ReadyPlanSnapshot(**self.ready.to_dict())

    def summary(self) -> dict[str, Any]:
        r = self.snapshot()
        return {
            "has_ready_plan": r is not None,
            "ready_plan_uid": None if r is None else r.plan_uid,
            "ready_at": None if r is None else r.ready_at,
            "eval_completed_at": None if r is None else r.eval_completed_at,
            "quality_completed_at": None if r is None else r.quality_completed_at,
            "selection_completed_at": None if r is None else r.selection_completed_at,
            "n_consider_calls": self.n_consider_calls,
            "n_ready_updates": self.n_ready_updates,
            "n_rejected_after_deadline": self.n_rejected_after_deadline,
            "n_rejected_not_bld": self.n_rejected_not_bld,
            "n_rejected_incomplete_metrics": self.n_rejected_incomplete_metrics,
            "n_post_deadline_diagnostics": len(self.post_deadline_diagnostics),
            "context_fingerprint": self.context_fingerprint,
            "pool_candidate_set_hash": self.pool_candidate_set_hash,
        }
