"""
Personal meal×major band deviation (internal structure metric).

Distinct from Layer-B ``structure_fit`` (meal-share / day-weight priors).
Never claim clinical quality from these scores.

Frozen normalization (experiment-predefined):
  band_width_floor_g = 50.0
  denom_mc = max(hi_mc - lo_mc, band_width_floor_g)
  norm_dev_mc = (under_mc + over_mc) / denom_mc
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from algo.structure_category import UPPER_ONLY_MAJORS
from algo.structure_features import day_structure_feature_dict

# Frozen before experiments — do not retune to favor a method.
BAND_WIDTH_FLOOR_G = 50.0
PERSONAL_BAND_METRIC_VERSION = "personal_band_v1_floor50g"


def compute_personal_band_deviation(
    day: pd.DataFrame,
    personal: Optional[dict],
    *,
    band_width_floor_g: float = BAND_WIDTH_FLOOR_G,
) -> dict[str, Any]:
    """
    Re-aggregate G_mc from selected rows; compare to frozen personal bands.

    Returns missing markers when personal/bands absent (not silent perfection).
    """
    out: dict[str, Any] = {
        "metric_class": "internal_personal_band_mechanism_only",
        "metric_version": PERSONAL_BAND_METRIC_VERSION,
        "band_width_floor_g": float(band_width_floor_g),
        "distinct_from_layer_b_structure_fit": True,
    }
    if personal is None:
        out["status"] = "missing_personal"
        out["total_band_deviation_g"] = None
        return out
    bands = personal.get("bands") or {}
    if not bands:
        out["status"] = "missing_bands"
        out["total_band_deviation_g"] = None
        return out

    feat = day_structure_feature_dict(day)
    upper_only = set(personal.get("upper_only_majors") or UPPER_ONLY_MAJORS)

    total_under = 0.0
    total_over = 0.0
    n_nonforbidden = 0
    n_nonforbidden_pass = 0
    n_nonforbidden_oob = 0
    n_positive_lo = 0
    n_positive_lo_met = 0
    n_forbidden_cells = 0
    n_forbidden_violations = 0
    n_zero_width = 0
    n_missing_feat = 0
    norm_sum = 0.0
    cell_rows = []

    for fname, band in bands.items():
        parts = str(fname).split("_")
        # expected g_major_{mi}_{meal}
        try:
            mi = int(parts[2])
        except (IndexError, ValueError):
            mi = 0
        raw_lo = band.get("lo")
        raw_hi = band.get("hi")
        if raw_lo is None or raw_hi is None:
            n_missing_feat += 1
            cell_rows.append({"feature": fname, "status": "missing_band_bounds"})
            continue
        lo = float(raw_lo)
        hi = float(raw_hi)
        if hi < lo:
            # malformed
            n_missing_feat += 1
            cell_rows.append({"feature": fname, "status": "invalid_hi_lt_lo", "lo": lo, "hi": hi})
            continue

        g = float(feat.get(fname, 0.0) or 0.0)
        if fname not in feat:
            n_missing_feat += 1

        # Forbidden cell: hi <= 0
        if hi <= 0:
            n_forbidden_cells += 1
            viol = g > 1e-6
            if viol:
                n_forbidden_violations += 1
            cell_rows.append(
                {
                    "feature": fname,
                    "G_mc": g,
                    "lo": lo,
                    "hi": hi,
                    "forbidden": True,
                    "forbidden_violation": viol,
                }
            )
            continue

        n_nonforbidden += 1
        is_upper_only = mi in upper_only
        under = 0.0 if is_upper_only else max(lo - g, 0.0)
        over = max(g - hi, 0.0)
        width = hi - lo
        if width <= 1e-9:
            n_zero_width += 1
        denom = max(width, float(band_width_floor_g))
        norm = (under + over) / denom
        norm_sum += norm
        total_under += under
        total_over += over

        oob = (under > 0) or (over > 0)
        if oob:
            n_nonforbidden_oob += 1
        else:
            n_nonforbidden_pass += 1

        if (not is_upper_only) and lo > 0:
            n_positive_lo += 1
            if g + 1e-6 >= lo:
                n_positive_lo_met += 1

        cell_rows.append(
            {
                "feature": fname,
                "G_mc": g,
                "lo": lo,
                "hi": hi,
                "under_g": under,
                "over_g": over,
                "upper_only": is_upper_only,
                "denom_g": denom,
                "norm_dev": norm,
                "forbidden": False,
            }
        )

    total_dev = total_under + total_over
    out.update(
        {
            "status": "ok",
            "stratum": personal.get("stratum"),
            "total_under_g": total_under,
            "total_over_g": total_over,
            "total_band_deviation_g": total_dev,
            "total_norm_deviation": norm_sum,
            "norm_formula": (
                f"sum_mc (under_mc+over_mc)/max(hi_mc-lo_mc, {band_width_floor_g}); "
                "upper_only: under=0; forbidden hi<=0 excluded from soft totals"
            ),
            "n_nonforbidden_cells": n_nonforbidden,
            "n_nonforbidden_pass": n_nonforbidden_pass,
            "n_nonforbidden_oob": n_nonforbidden_oob,
            "frac_nonforbidden_pass": (
                None if n_nonforbidden == 0 else n_nonforbidden_pass / n_nonforbidden
            ),
            "n_positive_lo_cells": n_positive_lo,
            "n_positive_lo_met": n_positive_lo_met,
            "frac_positive_lo_met": (
                None if n_positive_lo == 0 else n_positive_lo_met / n_positive_lo
            ),
            "n_forbidden_cells": n_forbidden_cells,
            "n_forbidden_violations": n_forbidden_violations,
            "n_zero_width_nonforbidden": n_zero_width,
            "n_missing_or_invalid_band": n_missing_feat,
            # Keep per-cell optional for audits (can be large)
            "n_cells_scored": len(cell_rows),
        }
    )
    return out


def meal_share_and_day_weight_deviation(
    day: pd.DataFrame,
    structure_prior: Optional[dict],
) -> dict[str, Any]:
    """Meal-share and day-total deviations vs NHANES structure prior (not personal bands)."""
    from algo.diet_structure_priors import day_structure_metrics

    metrics = day_structure_metrics(day)
    prior = structure_prior or {}
    share_dev = {}
    for m in ("breakfast", "lunch", "dinner", "snack"):
        key = f"share_g_{m}"
        obs = float(metrics.get(key, 0.0))
        tgt = float(prior.get(key, 0.25)) if prior else None
        share_dev[f"abs_dev_{key}"] = None if tgt is None else abs(obs - tgt)
        share_dev[key] = obs
    total_g = float(metrics.get("total_g", 0.0))
    p50 = prior.get("daily_weight_g_p50")
    return {
        "metric_class": "internal_meal_share_day_weight",
        "distinct_from_personal_band": True,
        "total_g": total_g,
        "abs_dev_total_g_vs_prior_p50": (
            None if p50 is None else abs(total_g - float(p50))
        ),
        **share_dev,
        "missing_prior": not bool(prior),
    }
