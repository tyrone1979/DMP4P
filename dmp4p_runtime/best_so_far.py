"""
External dual observers for E1a (recording / post-selection only).

A) raw Layer-B-best-so-far (maximize)
B) personal-band-deviation-best-so-far (minimize total_band_deviation_g)

Observers never change breeding, CX, mutate, archive stop, or selection.
Only sols whose UnifiedEvaluator validation finished while BudgetClock.in_budget()
may update best-so-far / enter the observed candidate log.
Post-deadline evaluations go to diagnostics only.

Candidate log key = ``plan_uid`` (food+meal+grams / candidate_id), **not**
food-id fingerprint alone (same foods ≠ same plan).

Quality packs are cached per ``plan_uid`` within a DualObservers instance
(bound to structure_prior / config / personal / score-field version). Cache hits
reuse deterministic metrics; they do not forge new verification times and do not
admit post-deadline first-verifications into in-budget scores.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from dmp4p_runtime.budget_clock import BudgetClock
from dmp4p_runtime.personal_band_metrics import PERSONAL_BAND_METRIC_VERSION
from dmp4p_runtime.score_fields import (
    SCORE_FIELD_VERSION,
    plan_uid_from_day,
    selected_rows_for_log,
    unified_quality_pack,
)

# Bump when cached pack schema / identity rules change.
OBSERVE_CACHE_VERSION = "observe_cache_v1_plan_uid_context"


@dataclass
class DualObservers:
    clock: BudgetClock
    structure_prior: Optional[dict] = None
    uncommon_ids: Optional[set[int]] = None
    config: Optional[dict] = None
    personal: Optional[dict] = None

    # A: raw Layer-B maximize
    layer_b_best: Optional[float] = None
    layer_b_best_fp: Optional[str] = None
    layer_b_best_plan_uid: Optional[str] = None
    layer_b_best_day: Optional[pd.DataFrame] = None
    layer_b_best_t_rel: Optional[float] = None
    layer_b_components: Optional[dict] = None
    n_layer_b_updates: int = 0
    layer_b_events: list[dict] = field(default_factory=list)

    # B: personal band deviation minimize
    band_best_dev: Optional[float] = None
    band_best_fp: Optional[str] = None
    band_best_plan_uid: Optional[str] = None
    band_best_day: Optional[pd.DataFrame] = None
    band_best_t_rel: Optional[float] = None
    band_best_detail: Optional[dict] = None
    n_band_updates: int = 0
    band_events: list[dict] = field(default_factory=list)

    n_in_budget_feasible: int = 0
    unique_fps_in_budget: set[str] = field(default_factory=set)
    unique_plan_uids_in_budget: set[str] = field(default_factory=set)
    n_post_deadline_eval: int = 0
    post_deadline_diagnostics: list[dict] = field(default_factory=list)
    # plan_uid → candidate record (observed path only; unique actual plans)
    observed_candidates: dict[str, dict] = field(default_factory=dict)
    observe_overhead_s: float = 0.0
    n_observe_calls: int = 0

    # Instance-local quality cache (never global; never shared across DualObservers)
    _quality_cache: dict[str, dict] = field(default_factory=dict, repr=False)
    n_cache_hits: int = 0
    n_cache_misses: int = 0
    n_duplicate_plan_callbacks: int = 0

    # Exclusive wall segments inside observe (do NOT nest-add into observe_overhead_s)
    t_plan_uid_s: float = 0.0
    t_cache_lookup_s: float = 0.0
    t_quality_pack_s: float = 0.0
    t_selected_rows_s: float = 0.0
    t_log_write_s: float = 0.0
    t_df_copy_s: float = 0.0
    t_observer_update_s: float = 0.0

    # Optional benchmark hook (multi-day online ready_plan). Production search unchanged.
    # Signature: (plan_uid, pack, rows, *, eval_completed_at, quality_completed_at,
    #             fingerprint_hash, quality_from_cache) -> None
    on_quality_ready: Optional[Callable[..., Any]] = field(default=None, repr=False)

    def _context_token(self) -> str:
        """Bind cache entries to this observer's scoring/context versions."""
        return "|".join(
            (
                OBSERVE_CACHE_VERSION,
                SCORE_FIELD_VERSION,
                PERSONAL_BAND_METRIC_VERSION,
                str(id(self.structure_prior)),
                str(id(self.config)),
                str(id(self.personal)),
                str(id(self.uncommon_ids)),
            )
        )

    def _get_or_compute_pack(
        self, day: pd.DataFrame, plan_uid: str
    ) -> tuple[dict[str, Any], Optional[list], bool]:
        """
        Return (pack, selected_rows, from_cache).

        Cache key = plan_uid (content hash of food|meal|grams) under this
        DualObservers instance context. Packs are value-dicts, not live DataFrames.
        """
        t0 = time.perf_counter()
        hit = self._quality_cache.get(plan_uid)
        self.t_cache_lookup_s += time.perf_counter() - t0
        ctx = self._context_token()
        if hit is not None and hit.get("context_token") == ctx:
            self.n_cache_hits += 1
            return hit["pack"], hit.get("selected_rows"), True

        self.n_cache_misses += 1
        t1 = time.perf_counter()
        pack = unified_quality_pack(
            day,
            structure_prior=self.structure_prior,
            uncommon_ids=self.uncommon_ids,
            config=self.config,
            personal=self.personal,
        )
        self.t_quality_pack_s += time.perf_counter() - t1
        t2 = time.perf_counter()
        rows = selected_rows_for_log(day)
        self.t_selected_rows_s += time.perf_counter() - t2
        self._quality_cache[plan_uid] = {
            "context_token": ctx,
            "pack": pack,
            "selected_rows": rows,
        }
        return pack, rows, False

    def observe_verified(
        self,
        day: pd.DataFrame,
        *,
        feasible: bool,
        fingerprint_hash: Optional[str] = None,
        eval_completed_at: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Call only after UnifiedEvaluator finished.
        ``eval_completed_at`` = absolute perf_counter when validation completed.
        """
        t_obs0 = time.perf_counter()
        self.n_observe_calls += 1
        try:
            return self._observe_impl(
                day,
                feasible=feasible,
                fingerprint_hash=fingerprint_hash,
                eval_completed_at=eval_completed_at,
            )
        finally:
            # Inclusive total for this call (exclusive segments are recorded inside).
            self.observe_overhead_s += time.perf_counter() - t_obs0

    def _observe_impl(
        self,
        day: pd.DataFrame,
        *,
        feasible: bool,
        fingerprint_hash: Optional[str],
        eval_completed_at: Optional[float],
    ) -> dict[str, Any]:
        t_abs = float(eval_completed_at if eval_completed_at is not None else self.clock.now())
        t_rel = t_abs - self.clock.t_start
        fp = str(fingerprint_hash or "")
        t0 = time.perf_counter()
        plan_uid = plan_uid_from_day(day)
        self.t_plan_uid_s += time.perf_counter() - t0

        if not self.clock.in_budget(t_abs):
            self.n_post_deadline_eval += 1
            diag: dict[str, Any] = {
                "t_rel_s": t_rel,
                "fingerprint_hash": fp,
                "plan_uid": plan_uid,
                "feasible": bool(feasible),
                "timing_status": "post_deadline",
            }
            if feasible:
                pack, _rows, from_cache = self._get_or_compute_pack(day, plan_uid)
                diag["raw_layer_b"] = pack.get("raw_layer_b")
                diag["total_band_deviation_g"] = (pack.get("personal_band") or {}).get(
                    "total_band_deviation_g"
                )
                diag["quality_from_cache"] = from_cache
            self.post_deadline_diagnostics.append(diag)
            return {"accepted_in_budget": False, "reason": "eval_after_deadline"}

        if not feasible:
            return {"accepted_in_budget": False, "reason": "infeasible"}

        # First verification + every actual observe call stay on BudgetClock
        # (no pause). Cache only skips recompute of deterministic quality.
        self.clock.mark_verified_feasible(t_abs)
        self.n_in_budget_feasible += 1
        if fp:
            self.unique_fps_in_budget.add(fp)
        self.unique_plan_uids_in_budget.add(plan_uid)

        pack, rows, from_cache = self._get_or_compute_pack(day, plan_uid)
        # Same monotonic clock as BudgetClock / OnlineReadyPlan (not a separate wall).
        quality_completed_at = float(self.clock.now())
        lb = float(pack["raw_layer_b"])
        band = pack.get("personal_band") or {}
        dev = band.get("total_band_deviation_g")

        if self.on_quality_ready is not None:
            try:
                self.on_quality_ready(
                    plan_uid,
                    pack,
                    rows,
                    eval_completed_at=t_abs,
                    quality_completed_at=quality_completed_at,
                    fingerprint_hash=fp or pack.get("fingerprint_hash"),
                    quality_from_cache=from_cache,
                )
            except Exception:
                # Ready-plan hook must not abort observation / search.
                pass

        t_log0 = time.perf_counter()
        prev = self.observed_candidates.get(plan_uid)
        if prev is not None:
            self.n_duplicate_plan_callbacks += 1
            prev["n_observe_events"] = int(prev.get("n_observe_events") or 1) + 1
            if from_cache:
                prev["n_cache_hits_for_plan"] = int(prev.get("n_cache_hits_for_plan") or 0) + 1
            # Keep earliest verified time; never forge a newer completion time.
            if float(t_rel) < float(prev.get("eval_completed_t_rel_s") or 1e18):
                prev["eval_completed_t_rel_s"] = t_rel
            # Refresh metrics only if this call recomputed a better raw LB
            # (should be identical under deterministic pack; keep for safety).
            if (not from_cache) and lb > float(prev.get("raw_layer_b") or -1e18):
                kept_t = prev.get("eval_completed_t_rel_s")
                prev.update(
                    {
                        "metrics": pack,
                        "selected_rows": rows,
                        "raw_layer_b": lb,
                        "total_band_deviation_g": dev,
                        "bld_min_weight_pass": pack.get("bld_min_weight_pass"),
                        "meal_grams": pack.get("meal_grams"),
                        "fingerprint_hash": fp or pack.get("fingerprint_hash"),
                        "eval_completed_t_rel_s": kept_t,
                    }
                )
        else:
            self.observed_candidates[plan_uid] = {
                "plan_uid": plan_uid,
                "fingerprint_hash": fp or pack.get("fingerprint_hash"),
                "eval_completed_t_rel_s": t_rel,
                "timing_status": "verified_in_budget",
                "timing_evidence": "eval_completed_at_le_deadline",
                "evaluator_feasible": True,
                "metrics": pack,
                "selected_rows": rows,
                "raw_layer_b": lb,
                "total_band_deviation_g": dev,
                "bld_min_weight_pass": pack.get("bld_min_weight_pass"),
                "meal_grams": pack.get("meal_grams"),
                "n_observe_events": 1,
                "n_cache_hits_for_plan": 1 if from_cache else 0,
                "first_quality_from_cache": from_cache,
            }
        self.t_log_write_s += time.perf_counter() - t_log0

        t_obs_upd0 = time.perf_counter()
        # Observer A: maximize raw Layer-B
        if self.layer_b_best is None or lb > self.layer_b_best + 1e-12:
            self.layer_b_best = lb
            self.layer_b_best_fp = fp
            self.layer_b_best_plan_uid = plan_uid
            t_c0 = time.perf_counter()
            self.layer_b_best_day = day.copy()
            self.t_df_copy_s += time.perf_counter() - t_c0
            self.layer_b_best_t_rel = t_rel
            self.layer_b_components = pack.get("layer_b_components")
            self.n_layer_b_updates += 1
            self.layer_b_events.append(
                {
                    "t_rel_s": t_rel,
                    "raw_layer_b": lb,
                    "layer_b_total": lb,
                    "fingerprint_hash": fp,
                    "plan_uid": plan_uid,
                    "kind": "layer_b_best",
                }
            )

        # Observer B: minimize band deviation
        if dev is not None:
            if self.band_best_dev is None or float(dev) < self.band_best_dev - 1e-12:
                self.band_best_dev = float(dev)
                self.band_best_fp = fp
                self.band_best_plan_uid = plan_uid
                t_c0 = time.perf_counter()
                self.band_best_day = day.copy()
                self.t_df_copy_s += time.perf_counter() - t_c0
                self.band_best_t_rel = t_rel
                self.band_best_detail = {
                    k: band.get(k)
                    for k in (
                        "total_under_g",
                        "total_over_g",
                        "total_norm_deviation",
                        "frac_nonforbidden_pass",
                        "n_nonforbidden_cells",
                        "n_nonforbidden_oob",
                        "n_forbidden_violations",
                        "status",
                    )
                }
                self.n_band_updates += 1
                self.band_events.append(
                    {
                        "t_rel_s": t_rel,
                        "total_band_deviation_g": float(dev),
                        "fingerprint_hash": fp,
                        "plan_uid": plan_uid,
                        "kind": "band_dev_best",
                    }
                )
        self.t_observer_update_s += time.perf_counter() - t_obs_upd0

        return {
            "accepted_in_budget": True,
            "plan_uid": plan_uid,
            "metrics": pack,
            "quality_from_cache": from_cache,
        }

    def candidate_list(self) -> list[dict]:
        return list(self.observed_candidates.values())

    def exclusive_timing_dict(self) -> dict[str, Any]:
        exclusive_sum = (
            self.t_plan_uid_s
            + self.t_cache_lookup_s
            + self.t_quality_pack_s
            + self.t_selected_rows_s
            + self.t_log_write_s
            + self.t_df_copy_s
            + self.t_observer_update_s
        )
        return {
            "note": (
                "Exclusive segment walls inside observe; they partition work and "
                "must NOT be nested-added on top of observe_overhead_s. "
                "observe_overhead_s is the inclusive call total."
            ),
            "observe_overhead_s_inclusive": self.observe_overhead_s,
            "exclusive_sum_s": exclusive_sum,
            "exclusive_unaccounted_s": max(0.0, self.observe_overhead_s - exclusive_sum),
            "t_plan_uid_s": self.t_plan_uid_s,
            "t_cache_lookup_s": self.t_cache_lookup_s,
            "t_quality_pack_s": self.t_quality_pack_s,
            "t_selected_rows_s": self.t_selected_rows_s,
            "t_log_write_s": self.t_log_write_s,
            "t_df_copy_s": self.t_df_copy_s,
            "t_observer_update_s": self.t_observer_update_s,
        }

    def cache_stats(self) -> dict[str, Any]:
        n = max(1, self.n_cache_hits + self.n_cache_misses)
        return {
            "observe_cache_version": OBSERVE_CACHE_VERSION,
            "n_cache_hits": self.n_cache_hits,
            "n_cache_misses": self.n_cache_misses,
            "cache_hit_rate": self.n_cache_hits / n,
            "n_cached_plans": len(self._quality_cache),
            "n_duplicate_plan_callbacks": self.n_duplicate_plan_callbacks,
            "approx_cache_entries": len(self._quality_cache),
            "context_bound": True,
            "identity": "plan_uid_food_meal_grams",
        }

    def summary(self) -> dict[str, Any]:
        lb_series = [e.get("raw_layer_b", e.get("layer_b_total")) for e in self.layer_b_events]
        lb_mono = all(
            lb_series[i] <= lb_series[i + 1] + 1e-9 for i in range(len(lb_series) - 1)
        )
        bd_series = [e["total_band_deviation_g"] for e in self.band_events]
        bd_mono = all(bd_series[i] >= bd_series[i + 1] - 1e-9 for i in range(len(bd_series) - 1))
        cands = self.candidate_list()
        return {
            "clock": self.clock.to_dict(),
            "n_in_budget_feasible_observations": self.n_in_budget_feasible,
            "n_unique_fingerprints_in_budget": len(self.unique_fps_in_budget),
            "n_unique_plan_uids_in_budget": len(self.unique_plan_uids_in_budget),
            "n_post_deadline_eval": self.n_post_deadline_eval,
            "observe_overhead_s": self.observe_overhead_s,
            "n_observe_calls": self.n_observe_calls,
            "observe_exclusive_timing": self.exclusive_timing_dict(),
            "observe_cache": self.cache_stats(),
            "observer_A_layer_b": {
                "best_raw_layer_b": self.layer_b_best,
                "best": self.layer_b_best,  # alias raw
                "fingerprint": self.layer_b_best_fp,
                "plan_uid": self.layer_b_best_plan_uid,
                "t_rel_s": self.layer_b_best_t_rel,
                "components": self.layer_b_components,
                "n_updates": self.n_layer_b_updates,
                "events": self.layer_b_events,
                "nondecreasing": lb_mono,
                "score_used": "raw_layer_b",
                "role": "post_selection_analysis_if_not_method_return",
            },
            "observer_B_personal_band": {
                "best_total_band_deviation_g": self.band_best_dev,
                "fingerprint": self.band_best_fp,
                "plan_uid": self.band_best_plan_uid,
                "t_rel_s": self.band_best_t_rel,
                "detail": self.band_best_detail,
                "n_updates": self.n_band_updates,
                "events": self.band_events,
                "nonincreasing_deviation": bd_mono,
                "role": "post_selection_analysis_if_not_method_return",
                "note": "Independent of observer A; best plans may differ",
            },
            "observed_in_budget_candidates": cands,
            "online_unique_fp_count": len(self.unique_fps_in_budget),
            "online_unique_plan_uid_count": len(self.unique_plan_uids_in_budget),
            "post_deadline_diagnostics_n": len(self.post_deadline_diagnostics),
            "log_scope": (
                "observed_validation_path_only_not_full_unobserved_population"
            ),
            "log_dedup": "unique_plan_uid_with_n_observe_events",
        }
