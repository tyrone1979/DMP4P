"""
Unified Layer-B / BLD score field definitions for Bronze quality benchmarks.

Field contract (do not overload names across paths):

- ``raw_layer_b``: ``layer_b_score()`` scalar **without** BLD+5 or iforest bonuses.
  Source of truth for benchmark return-strategy tie-breaks:
  ``compute_internal_quality(...).layer_b_total`` / ``.raw_layer_b``.

- ``legacy_selection_score``: historical selection scalar that may include BLD+5
  (and in production archive path, optional iforest). Exposed on
  ``EvalResult.legacy_selection_score``. For backward compatibility,
  ``EvalResult.layer_b_score`` is an **alias of legacy_selection_score**
  (not raw). Documented so the same name is not silently reinterpreted.

- ``bld_min_weight_pass``: independent boolean — breakfast/lunch/dinner meet
  ``config.min_meal_weight_g`` (production: 30/50/50). Not a hard gate when
  ``require_bld=False`` in the fair benchmark.

Historical impact: nutrient_ga / V2 finalize packing that maximized
``ev.layer_b_score`` used the legacy (+5) scale, while DualObservers used raw.
That can change **which plan is returned**, not only fingerprint agreement rates.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

import pandas as pd

from algo.diet_structure_priors import check_bld_hard, day_structure_metrics, load_ga_config
from dmp4p_runtime.quality_metrics import compute_internal_quality

# Frozen delivery thresholds (aligned with config/ga_multiobj_weights.yaml).
BLD_MIN_G = {"breakfast": 30.0, "lunch": 50.0, "dinner": 50.0}
SCORE_FIELD_VERSION = "score_fields_v1_raw_vs_legacy_bld"


def meal_grams_by_meal(day: pd.DataFrame) -> dict[str, float]:
    out = {m: 0.0 for m in ("breakfast", "lunch", "dinner", "snack")}
    if day is None or day.empty or "meal" not in day.columns:
        return out
    meal = day["meal"].astype(str).str.lower()
    grams = pd.to_numeric(day.get("grams", 0), errors="coerce").fillna(0.0)
    for m in out:
        out[m] = float(grams[meal == m].sum())
    return out


def bld_min_weight_pass_from_day(
    day: pd.DataFrame,
    *,
    config: Optional[dict] = None,
    thresholds: Optional[dict] = None,
) -> bool:
    """Independent BLD min-gram check (not gated by require_bld)."""
    thr = dict(thresholds or BLD_MIN_G)
    if config is not None:
        mins = (config.get("min_meal_weight_g") or {}) if isinstance(config, dict) else {}
        for m in ("breakfast", "lunch", "dinner"):
            if m in mins:
                thr[m] = float(mins[m])
    grams = meal_grams_by_meal(day)
    return all(grams[m] >= float(thr[m]) for m in ("breakfast", "lunch", "dinner"))


def bld_min_weight_pass_from_metrics(metrics: dict, config: Optional[dict] = None) -> bool:
    ok, _ = check_bld_hard(metrics, config)
    return bool(ok)


def plan_uid_from_day(day: pd.DataFrame) -> str:
    """
    Stable identity for a concrete plan (food + meal + grams).

    Food-id fingerprint alone is NOT unique when grams/meals differ.
    Always hash content triples — do not key solely on candidate_id (can be
    stale if grams are edited) or food-id set alone.
    """
    if day is None or day.empty:
        return hashlib.sha1(b"empty").hexdigest()[:16]
    parts = []
    for _, r in day.iterrows():
        parts.append(
            f"{int(float(r.get('food_id', 0)))}|"
            f"{str(r.get('meal', '')).strip().lower()}|"
            f"{float(r.get('grams', 0) or 0):.8g}"
        )
    payload = ";".join(sorted(parts))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def selected_rows_for_log(day: Optional[pd.DataFrame], limit: int = 40) -> Optional[list]:
    if day is None or day.empty:
        return None
    cols = [c for c in ("food_id", "meal", "grams", "candidate_id") if c in day.columns]
    if not cols:
        cols = list(day.columns)[:6]
    return day[cols].head(limit).to_dict(orient="records")


def unified_quality_pack(
    day: pd.DataFrame,
    *,
    structure_prior: Optional[dict] = None,
    uncommon_ids: Optional[set] = None,
    config: Optional[dict] = None,
    personal: Optional[dict] = None,
) -> dict[str, Any]:
    """Raw Layer-B + band + BLD pass + meal grams (benchmark reporting pack)."""
    cfg = config or load_ga_config()
    q = compute_internal_quality(
        day,
        structure_prior=structure_prior,
        uncommon_ids=uncommon_ids,
        config=cfg,
        personal=personal,
    )
    from dmp4p_runtime.personal_band_metrics import compute_personal_band_deviation

    band = compute_personal_band_deviation(day, personal)
    grams = meal_grams_by_meal(day)
    bld_pass = bld_min_weight_pass_from_day(day, config=cfg)
    metrics = day_structure_metrics(day)
    bld_from_cfg = bld_min_weight_pass_from_metrics(metrics, cfg)
    n_nf = band.get("n_nonforbidden_cells")
    n_oob = band.get("n_nonforbidden_oob")
    all_bands = None if not n_nf else int(n_oob or 0) == 0
    return {
        "score_field_version": SCORE_FIELD_VERSION,
        "raw_layer_b": q.get("layer_b_total"),
        "layer_b_total": q.get("layer_b_total"),  # alias of raw
        "layer_b_components": {
            k: q.get(k)
            for k in (
                "structure_fit",
                "positive_negative",
                "n_food_groups",
                "n_meals",
                "n_foods",
                "n_uncommon",
            )
        },
        "personal_band": band,
        "cell_pass_rate": band.get("frac_nonforbidden_pass"),
        "all_bands_simultaneous_pass": all_bands,
        "n_violating_nonforbidden_cells": n_oob,
        "bld_min_weight_pass": bool(bld_pass and bld_from_cfg),
        "meal_grams": grams,
        "plan_uid": plan_uid_from_day(day),
        "fingerprint_hash": q.get("fingerprint_hash"),
    }
