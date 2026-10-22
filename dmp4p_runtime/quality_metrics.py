"""
Internal quality metrics for Bronze quality experiments (E1+).

Layer-B and structure deviations are **mechanism / internal** metrics only.
HEI is explicitly unavailable until FPED scoring is implemented.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from algo.diet_structure_priors import (
    day_structure_metrics,
    layer_b_score,
    load_ga_config,
    structure_fit,
)


def compute_internal_quality(
    day: pd.DataFrame,
    *,
    structure_prior: Optional[dict] = None,
    uncommon_ids: Optional[set[int]] = None,
    config: Optional[dict] = None,
    personal: Optional[dict] = None,
) -> dict[str, Any]:
    cfg = config or load_ga_config()
    prior = structure_prior or {}
    lb = layer_b_score(day, prior, uncommon_ids, cfg)
    metrics = day_structure_metrics(day)
    s_fit = float(lb.get("structure_score", structure_fit(metrics, prior, cfg)))
    # Meal-share absolute deviations vs prior
    share_dev = {}
    for m in ("breakfast", "lunch", "dinner", "snack"):
        key = f"share_g_{m}"
        obs = float(metrics.get(key, 0.0))
        tgt = float(prior.get(key, 0.25)) if prior else 0.25
        share_dev[f"abs_dev_{key}"] = abs(obs - tgt)
    total_g = float(metrics.get("total_g", 0.0))
    p50 = float(prior.get("daily_weight_g_p50", prior.get("daily_weight_g_p75", total_g)) or total_g)
    raw = float(lb["score"])
    out = {
        "layer_b_total": raw,
        "raw_layer_b": raw,  # explicit alias — never includes BLD+5
        "structure_fit": s_fit,
        "positive_negative": float(lb["positive_negative"]),
        "n_food_groups": float(lb["n_unique_food_groups"]),
        "n_meals": float(lb["n_unique_meals"]),
        "n_foods": float(lb["n_unique_foods"]),
        "n_uncommon": float(lb["n_uncommon"]),
        "total_g": total_g,
        "abs_dev_total_g_vs_prior_p50": abs(total_g - p50),
        **share_dev,
        "fingerprint_hash": lb.get("fingerprint_hash"),
        "metric_class": "internal_mechanism_only",
        "score_note": "raw_layer_b_equals_layer_b_total_no_bld_plus_5",
    }
    # Optional personal category gram deviation (soft)
    if personal is not None:
        try:
            from algo.structure_category import score_personal_category_day
            from algo.structure_features import day_structure_feature_dict

            feat = day_structure_feature_dict(day)
            cat = score_personal_category_day(feat, personal, presence_only=False)
            out["category_ok_presence_or_grams"] = bool(cat.get("ok", False))
            out["category_penalty"] = float(cat.get("penalty", 0.0) or 0.0)
        except Exception as e:
            out["category_score_error"] = type(e).__name__
    return out


def hei_status() -> dict[str, Any]:
    """Independent HEI: not available (stub)."""
    from algo.hei_score import score_day_hei

    probe = score_day_hei(pd.DataFrame())
    return {
        "available": False,
        "score_day_hei_returns": probe,
        "version_intended": "HEI-2020",
        "missing": [
            "food_id → FPED day aggregation wired into score_day_hei",
            "NCI HEI-2015/2020 density macros",
            "pinned scorer version + test vectors",
        ],
        "note": "Do not invent a simplified HEI for this benchmark.",
    }
