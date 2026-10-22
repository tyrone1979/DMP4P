"""
Optimizer-independent solution evaluator for Bronze search benchmarks.

Independence contract
---------------------
Every call re-aggregates nutrients/grams/meals from the provided selected rows.
It does **not** read V2 archive labels, fitness==1 flags, or cached feasibility.
Production archive helpers (e.g. ``check_condiment_oil_hard``) may be reused as
shared pure functions; inputs are always recomputed from the day DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from algo.diet_structure_priors import (
    check_bld_hard,
    check_condiment_oil_hard,
    day_structure_metrics,
    food_id_fingerprint,
    get_common_uncommon_sets,
    layer_b_score,
    load_ga_config,
)
from algo.meal_merge import keep_one_food_per_day, keep_one_food_per_eating_type
from dmp4p_runtime.constraints import VIOLATION_SCALES, ConstraintSpec

# Exact float compares for hard gates (no epsilon slack).
# Documented in constraint_catalog.yaml.


def _ensure_meal_cols(sol: pd.DataFrame) -> pd.DataFrame:
    out = sol.copy()
    if "meal" not in out.columns and "meal_id" in out.columns:
        from algo.diet_structure_priors import ID_TO_MEAL

        out["meal"] = out["meal_id"].map(ID_TO_MEAL)
    if "meal" not in out.columns and "eating_type" in out.columns:
        from algo.diet_structure_priors import meal_from_eating_type

        out["meal"] = out["eating_type"].map(meal_from_eating_type)
    if "food_group_code" not in out.columns and "food_id" in out.columns:
        from algo.diet_structure_priors import food_group_code

        out["food_group_code"] = out["food_id"].map(food_group_code)
    return out


def decode_selection(
    pool: pd.DataFrame,
    selected_indices: list[int] | np.ndarray | None = None,
    selected_mask: np.ndarray | None = None,
    *,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Materialize selected candidate rows; optionally collapse duplicate food_ids."""
    if selected_mask is not None:
        idx = np.flatnonzero(np.asarray(selected_mask, dtype=bool))
    elif selected_indices is not None:
        idx = np.asarray(selected_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(pool))]
    else:
        raise ValueError("provide selected_indices or selected_mask")
    if len(idx) == 0:
        return pool.iloc[0:0].copy()
    sol = pool.iloc[idx].copy().reset_index(drop=True)
    sol = _ensure_meal_cols(sol)
    if dedupe:
        sol = keep_one_food_per_day(keep_one_food_per_eating_type(sol))
    return sol.reset_index(drop=True)


@dataclass
class EvalResult:
    feasible: bool
    fingerprint: frozenset
    fingerprint_hash: str
    n_foods: int
    total_g: float
    nutrient_totals: dict[str, float]
    hard_violations: dict[str, float]
    normalized_violation: float
    soft_category_gram_deviation: float
    # Alias of legacy_selection_score (may include BLD+5). NOT raw Layer-B.
    layer_b_score: Optional[float]
    layer_b_detail: dict[str, Any] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)
    solution: Optional[pd.DataFrame] = None
    bld_ok: bool = True
    # Explicit score fields (score_fields_v1)
    raw_layer_b: Optional[float] = None
    legacy_selection_score: Optional[float] = None
    bld_min_weight_pass: bool = False

    def to_dict(self, include_solution: bool = False) -> dict[str, Any]:
        d = {
            "feasible": bool(self.feasible),
            "fingerprint_hash": self.fingerprint_hash,
            "fingerprint_food_ids": sorted(int(x) for x in self.fingerprint),
            "n_foods": int(self.n_foods),
            "total_g": float(self.total_g),
            "nutrient_totals": {k: float(v) for k, v in self.nutrient_totals.items()},
            "hard_violations": {k: float(v) for k, v in self.hard_violations.items()},
            "normalized_violation": float(self.normalized_violation),
            "soft_category_gram_deviation": float(self.soft_category_gram_deviation),
            "layer_b_score": None if self.layer_b_score is None else float(self.layer_b_score),
            "layer_b_score_meaning": "alias_of_legacy_selection_score_may_include_bld_plus_5",
            "raw_layer_b": None if self.raw_layer_b is None else float(self.raw_layer_b),
            "legacy_selection_score": (
                None
                if self.legacy_selection_score is None
                else float(self.legacy_selection_score)
            ),
            "bld_min_weight_pass": bool(self.bld_min_weight_pass),
            "layer_b_detail": dict(self.layer_b_detail),
            "failure_reasons": list(self.failure_reasons),
            "bld_ok": bool(self.bld_ok),
        }
        if include_solution and self.solution is not None:
            d["solution_records"] = self.solution.to_dict(orient="records")
        return d


class UnifiedEvaluator:
    """Re-aggregate selected rows and score hard/soft constraints from scratch."""

    def __init__(
        self,
        target_dict: dict,
        spec: ConstraintSpec,
        *,
        config: Optional[dict] = None,
        structure_prior: Optional[dict] = None,
        personal_category_structure: Optional[dict] = None,
        uncommon_ids: Optional[set[int]] = None,
        food_desc: Optional[dict[int, str]] = None,  # unused for hard gates; kept for Layer-B only
    ):
        self.target_dict = dict(target_dict)
        self.spec = spec
        self.cfg = config or load_ga_config()
        self.structure_prior = structure_prior or {}
        self.personal = personal_category_structure
        if uncommon_ids is None:
            _, uncommon = get_common_uncommon_sets(self.cfg)
            self.uncommon_ids = uncommon
        else:
            self.uncommon_ids = uncommon_ids
        self.food_desc = food_desc  # not used for breakfast hard gate anymore

    def evaluate_solution(self, sol: pd.DataFrame) -> EvalResult:
        failures: list[str] = []
        viol: dict[str, float] = {k: 0.0 for k in VIOLATION_SCALES}
        if sol is None or len(sol) == 0:
            viol["cardinality"] = float(self.spec.min_foods)
            return EvalResult(
                feasible=False,
                fingerprint=frozenset(),
                fingerprint_hash="",
                n_foods=0,
                total_g=0.0,
                nutrient_totals={},
                hard_violations=viol,
                normalized_violation=self._norm(viol),
                soft_category_gram_deviation=0.0,
                layer_b_score=None,
                failure_reasons=["empty_solution"],
                solution=sol,
                bld_ok=False,
            )

        sol = _ensure_meal_cols(sol)
        fids_raw = pd.to_numeric(sol["food_id"], errors="coerce").dropna().astype(int)
        n_dup = int(len(fids_raw) - fids_raw.nunique())
        if n_dup > 0:
            sol = keep_one_food_per_day(keep_one_food_per_eating_type(sol))
            viol["duplicate_food"] = float(n_dup)

        metrics = day_structure_metrics(sol)
        n_foods = int(metrics["n_unique_foods"])
        total_g = float(metrics["total_g"])
        fp = metrics["fingerprint"]
        fph = metrics["fingerprint_hash"]

        nut_totals: dict[str, float] = {}
        nut_pen = 0.0
        for nut, (lo, hi) in self.target_dict.items():
            if nut not in sol.columns:
                failures.append(f"missing_nutrient_col:{nut}")
                nut_pen += abs(float(lo)) + abs(float(hi))
                nut_totals[nut] = 0.0
                continue
            val = float(pd.to_numeric(sol[nut], errors="coerce").fillna(0).sum())
            nut_totals[nut] = val
            # Exact boundary: lo <= val <= hi
            if val < float(lo) or val > float(hi):
                nut_pen += max(0.0, float(lo) - val) + max(0.0, val - float(hi))
                failures.append(f"nutrient:{nut}")
        viol["nutrient"] = float(nut_pen)

        if n_foods < self.spec.min_foods:
            viol["cardinality"] = float(self.spec.min_foods - n_foods)
            failures.append("cardinality_lo")
        elif n_foods > self.spec.max_combination_size:
            viol["cardinality"] = float(n_foods - self.spec.max_combination_size)
            failures.append("cardinality_hi")

        if self.spec.hard_daily_weight and total_g > self.spec.max_daily_g:
            viol["daily_weight"] = float(total_g - self.spec.max_daily_g)
            failures.append("daily_weight")

        bld_ok, bld_pen = check_bld_hard(metrics, self.cfg)
        if self.spec.require_bld and not bld_ok:
            viol["bld"] = float(bld_pen)
            failures.append("bld")

        if self.spec.check_condiment:
            oil_ok, oil_fails = check_condiment_oil_hard(sol, self.cfg)
            if not oil_ok:
                viol["condiment"] = float(len(oil_fails))
                failures.extend(oil_fails or ["condiment"])

        # Breakfast placement: use frozen snapshot flags only
        if self.spec.check_breakfast_placement:
            if "bench_placement_allowed" in sol.columns:
                bad = (~sol["bench_placement_allowed"].astype(bool)).sum()
                if int(bad) > 0:
                    viol["breakfast_placement"] = float(bad)
                    failures.append("breakfast_placement")
            else:
                # Snapshot without flags → cannot claim production equivalence
                viol["breakfast_placement"] = 1.0
                failures.append("breakfast_placement_flags_missing")

        cat_dev = 0.0
        if self.personal is not None:
            try:
                from algo.structure_features import day_structure_feature_dict
                from algo.structure_category import score_personal_category_day

                feat = day_structure_feature_dict(sol)
                presence = score_personal_category_day(
                    feat, self.personal, presence_only=True
                )
                if self.spec.category_presence_hard and not presence.get("ok", True):
                    n_fail = len(presence.get("critical_fails") or [])
                    viol["category_presence"] = float(max(1, n_fail))
                    failures.append("category_presence")
                cat_dev = self._category_gram_deviation(feat, self.personal)
                if self.spec.category_grams_hard:
                    full = score_personal_category_day(
                        feat, self.personal, presence_only=False
                    )
                    if not full.get("ok", True):
                        viol["category_presence"] = max(
                            viol["category_presence"],
                            float(len(full.get("critical_fails") or []) or 1),
                        )
                        failures.append("category_grams_hard")
            except Exception as e:
                failures.append(f"category_eval_error:{type(e).__name__}")

        raw_lb = None
        legacy_sel = None
        lb_detail: dict[str, Any] = {}
        try:
            lb_detail = dict(
                layer_b_score(sol, self.structure_prior, self.uncommon_ids, self.cfg)
            )
            raw_lb = float(lb_detail.get("score", 0.0))
            legacy_sel = raw_lb + 5.0 if bld_ok else raw_lb
            # Keep detail["score"] as legacy for archive-compat readers; expose raw explicitly.
            lb_detail["raw_score"] = raw_lb
            lb_detail["score"] = legacy_sel
            lb_detail["legacy_includes_bld_plus_5"] = bool(bld_ok)
        except Exception:
            lb_detail = {}

        feasible = all(v <= 0.0 for v in viol.values())
        return EvalResult(
            feasible=bool(feasible),
            fingerprint=fp if isinstance(fp, frozenset) else food_id_fingerprint(fp),
            fingerprint_hash=str(fph),
            n_foods=n_foods,
            total_g=total_g,
            nutrient_totals=nut_totals,
            hard_violations=viol,
            normalized_violation=float(self._norm(viol)),
            soft_category_gram_deviation=float(cat_dev),
            # layer_b_score remains legacy alias (production-adjacent packing paths).
            layer_b_score=legacy_sel,
            layer_b_detail={
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in lb_detail.items()
            },
            failure_reasons=failures if not feasible else [],
            solution=sol,
            bld_ok=bool(bld_ok),
            raw_layer_b=raw_lb,
            legacy_selection_score=legacy_sel,
            bld_min_weight_pass=bool(bld_ok),
        )

    def evaluate_mask(self, pool: pd.DataFrame, mask: np.ndarray) -> EvalResult:
        return self.evaluate_solution(decode_selection(pool, selected_mask=mask))

    def evaluate_indices(self, pool: pd.DataFrame, indices: list[int] | np.ndarray) -> EvalResult:
        return self.evaluate_solution(decode_selection(pool, selected_indices=indices))

    @staticmethod
    def _norm(viol: dict[str, float]) -> float:
        total = 0.0
        for k, v in viol.items():
            scale = float(VIOLATION_SCALES.get(k, 1.0))
            total += float(v) / max(scale, 1e-9)
        return float(total)

    @staticmethod
    def _category_gram_deviation(feat: dict, personal: dict) -> float:
        bands = personal.get("bands") or {}
        upper_only = set(personal.get("upper_only_majors") or [])
        dev = 0.0
        for key, band in bands.items():
            v = float(feat.get(key, 0.0) or 0.0)
            lo = float(band.get("lo") or 0.0)
            hi = float(band.get("hi") if band.get("hi") is not None else 1e9)
            parts = key.split("_")
            try:
                mi = int(parts[2])
            except (IndexError, ValueError):
                mi = 0
            if hi <= 0:
                continue
            if v > hi:
                dev += v - hi
            elif lo > 0 and mi not in upper_only and v < lo:
                dev += lo - v
        return float(dev)


def evaluate_selection(
    pool: pd.DataFrame,
    selected_mask: np.ndarray,
    target_dict: dict,
    spec: ConstraintSpec,
    **kwargs,
) -> EvalResult:
    ev = UnifiedEvaluator(target_dict, spec, **kwargs)
    return ev.evaluate_mask(pool, selected_mask)


def pairwise_jaccard(fps: list[frozenset]) -> list[float]:
    out: list[float] = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            a, b = fps[i], fps[j]
            if not a and not b:
                out.append(1.0)
            else:
                inter = len(a & b)
                union = len(a | b)
                out.append(inter / union if union else 0.0)
    return out
