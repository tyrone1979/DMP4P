"""
Category-gram structure prior: FNDDS major (1–9) × eating_type gram envelopes.

Learned from full adult days, stratified by sex×BMI×age only (no disease inputs).
GA Layer-A uses scaled [lo, hi] per (meal × major) as Hard envelope.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from algo.structure_features import (
    MAJOR_CODES,
    MAJOR_GRAM_FEATURES,
    MEAL_MAJOR_GRAM_FEATURES,
    MEAL_NAMES,
    stratum_key,
)
from algo.structure_learn import (
    _pick_envelope,
    _quantiles,
    expected_total_g,
)

ROOT = Path(__file__).resolve().parents[1]

# Category Hard axes = meal×major; keep day totals for g_scale reference.
CATEGORY_FEATURE_NAMES = list(MEAL_MAJOR_GRAM_FEATURES) + list(MAJOR_GRAM_FEATURES) + [
    "total_g",
    "total_kcal",
]

# Majors with lower-bound hard only when empirical p_lo > 0.
# Oils(8) and sweets/bev(9): upper-only always (any meal).
UPPER_ONLY_MAJORS = {8, 9}


def _band_feature_names() -> list[str]:
    """Axes that get quantile envelopes (meal×major + day major + mass refs)."""
    return list(CATEGORY_FEATURE_NAMES)


def fit_category_prior(
    feat_df: pd.DataFrame,
    *,
    q_lo: float = 0.05,
    q_hi: float = 0.95,
    min_stratum_n: int = 40,
    schema=None,
) -> dict[str, Any]:
    """
    Fit overall + stratum quantile bands on schema cell features (default meal×major).

    Person strata come from ``StructureSchema`` (default sex×BMI×age). Extra
    optional dims (race, cuisine, …) are ignored until marked active + columns exist.
    """
    from algo.structure_schema import get_structure_schema

    sch = get_structure_schema(schema)
    feat_df = feat_df.copy()
    names = list(sch.band_feature_names())
    # Keep legacy CATEGORY_FEATURE_NAMES filled for older callers / day aggregates
    for f in list(dict.fromkeys(list(CATEGORY_FEATURE_NAMES) + names)):
        if f not in feat_df.columns:
            feat_df[f] = 0.0

    X = feat_df[names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    overall = {f: _quantiles(X[f], q_lo, q_hi) for f in names}

    feat_df["_stratum"] = feat_df.apply(lambda r: sch.stratum_key_from_row(r), axis=1)
    strata: dict[str, Any] = {}
    for st, sub in feat_df.groupby("_stratum"):
        if len(sub) < min_stratum_n:
            continue
        Xs = sub[names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        strata[str(st)] = {f: _quantiles(Xs[f], q_lo, q_hi) for f in names}

    ref_total_g = float(overall.get("total_g", {}).get("p50") or 2500.0)
    ref_total_kcal = float(overall.get("total_kcal", {}).get("p50") or 2200.0)

    hard_cells = list(sch.hard_cell_features())
    groups_summary = []
    for f in hard_cells:
        q = overall.get(f) or {"n": 0, "p50": 0, "p_lo": 0, "p_hi": 0}
        parts = f.split("_")
        meal = parts[-1] if parts else ""
        try:
            code = int(parts[2]) if len(parts) >= 4 else parts[2]
        except (ValueError, IndexError):
            code = parts[2] if len(parts) > 2 else ""
        upper = isinstance(code, int) and code in UPPER_ONLY_MAJORS
        groups_summary.append(
            {
                "meal": meal,
                "axis_code": code,
                "major": code if isinstance(code, int) else None,
                "feature": f,
                "n": q["n"],
                "p50": round(q["p50"], 2),
                "lo": round(q["p_lo"], 2),
                "hi": round(q["p_hi"], 2),
                "upper_only": upper,
            }
        )

    return {
        "version": sch.version,
        "kind": "category_grams",
        "by_eating_type": True,
        "meals": list(sch.meals),
        "n_days": int(len(feat_df)),
        "q_lo": q_lo,
        "q_hi": q_hi,
        "feature_names": list(names),
        "meal_major_features": hard_cells,
        "major_codes": list(MAJOR_CODES),
        "upper_only_majors": sorted(UPPER_ONLY_MAJORS),
        "ref_total_g": ref_total_g,
        "ref_total_kcal": ref_total_kcal,
        "overall": overall,
        "strata": strata,
        "n_strata_kept": len(strata),
        "groups_summary": groups_summary,
        "scale_rule": "expected_total_g / ref_total_g",
        "schema": sch.to_meta(),
    }


def save_category_prior(model: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in model.items() if not k.startswith("_")}
    (out_dir / "prior.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(model.get("groups_summary") or []).to_csv(out_dir / "groups_summary.csv", index=False)

    # overall meal×major matrix for quick look
    rows = []
    for meal in MEAL_NAMES:
        row: dict[str, Any] = {"meal": meal}
        for i in MAJOR_CODES:
            f = f"g_major_{i}_{meal}"
            b = (model.get("overall") or {}).get(f) or {}
            row[f"m{i}_lo"] = b.get("p_lo")
            row[f"m{i}_p50"] = b.get("p50")
            row[f"m{i}_hi"] = b.get("p_hi")
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "overall_meal_major_bands.csv", index=False)

    # compact stratum × grain (major 5) by meal — common audit
    srows = []
    for st, bands in (model.get("strata") or {}).items():
        row = {"stratum": st, "n": int((bands.get("total_g") or {}).get("n") or 0)}
        for meal in MEAL_NAMES:
            f = f"g_major_5_{meal}"
            b = bands.get(f) or {}
            row[f"grain_{meal}_lo"] = b.get("p_lo")
            row[f"grain_{meal}_p50"] = b.get("p50")
            row[f"grain_{meal}_hi"] = b.get("p_hi")
        # day total grain for comparison
        bd = bands.get("g_major_5") or {}
        row["grain_day_lo"] = bd.get("p_lo")
        row["grain_day_p50"] = bd.get("p50")
        row["grain_day_hi"] = bd.get("p_hi")
        srows.append(row)
    if srows:
        pd.DataFrame(srows).to_csv(out_dir / "strata_grain_by_meal.csv", index=False)


def load_category_prior(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    return json.loads((out_dir / "prior.json").read_text(encoding="utf-8"))


def get_active_category_prior(
    config_path: str | Path = "",
    *,
    force: bool = False,
) -> Optional[dict[str, Any]]:
    """Load Full category-gram prior when ``active_structure`` / hard flag enables it."""
    cfg_path = Path(config_path) if config_path else ROOT / "config" / "structure_learn.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    active = str(cfg.get("active_structure") or "").strip().lower()
    hard = bool(cfg.get("category_grams_hard", False))
    if not force and active not in ("category_grams", "category", "full_category") and not hard:
        return None
    paths = cfg.get("paths") or {}
    rel = paths.get("full_category") or "outputs/structure_learn/full_category"
    out = Path(rel)
    if not out.is_absolute():
        out = ROOT / out
    if not (out / "prior.json").exists():
        return None
    model = load_category_prior(out)
    model["_active_path"] = str(out)
    model["_config"] = cfg
    return model


def _hard_axes(model: dict[str, Any]) -> list[str]:
    """Prefer meal×major axes; fall back to day majors for older priors."""
    axes = list(model.get("meal_major_features") or [])
    if axes:
        return axes
    # legacy day-only prior
    return [f"g_major_{i}" for i in MAJOR_CODES]


def resolve_personal_category_structure(
    model: dict[str, Any],
    *,
    sex=None,
    bmi: float | None = None,
    age=None,
    total_g_ref: float | None = None,
    calorie_lo: float | None = None,
    calorie_hi: float | None = None,
    g_scale_clip: tuple[float, float] = (0.55, 2.4),
) -> dict[str, Any]:
    """
    Select & freeze a person-specific category structure **before GA**.

    Uses demo only (sex×BMI×age) + optional calorie mid / total_g_ref for g_scale.
    Returns scaled meal×major [lo, hi] bands ready for Layer-A Hard.
    """
    shim = {
        "overall": model["overall"],
        "strata": model.get("strata") or {},
        "ref_total_g": model.get("ref_total_g"),
        "ref_total_kcal": model.get("ref_total_kcal"),
    }
    envelope, envelope_src = _pick_envelope(shim, sex=sex, bmi=bmi, age=age)
    ref_g = float(model.get("ref_total_g") or (model["overall"].get("total_g") or {}).get("p50") or 2500.0)
    if envelope_src != "overall":
        ref_g = float((envelope.get("total_g") or {}).get("p50") or ref_g)
    person_g = expected_total_g(
        shim,
        bmi=bmi,
        sex=sex,
        age=age,
        total_g_ref=total_g_ref,
        calorie_lo=calorie_lo,
        calorie_hi=calorie_hi,
    )
    scale = person_g / ref_g if ref_g > 0 else 1.0
    scale = float(np.clip(scale, float(g_scale_clip[0]), float(g_scale_clip[1])))
    upper_only = sorted(set(model.get("upper_only_majors") or UPPER_ONLY_MAJORS))

    bands: dict[str, dict[str, float]] = {}
    groups: dict[str, dict[str, dict[str, float]]] = {}
    for f in _hard_axes(model):
        band = envelope.get(f) or model["overall"].get(f)
        if not band:
            continue
        lo = float(band["p_lo"]) * scale
        hi = float(band["p_hi"]) * scale
        p50 = float(band.get("p50") or 0.0) * scale
        bands[f] = {
            "lo": lo,
            "hi": hi,
            "p50": p50,
            "raw_lo": float(band["p_lo"]),
            "raw_hi": float(band["p_hi"]),
            "raw_p50": float(band.get("p50") or 0.0),
        }
        parts = f.split("_")
        try:
            mi = str(int(parts[2]))
        except (IndexError, ValueError):
            continue
        meal = parts[3] if len(parts) >= 4 else "day"
        groups.setdefault(meal, {})[mi] = {"lo": lo, "hi": hi, "p50": p50}

    return {
        "kind": "personal_category_grams",
        "version": model.get("version"),
        "by_eating_type": bool(model.get("by_eating_type")),
        "stratum": envelope_src,
        "sex": sex,
        "bmi": bmi,
        "age": age,
        "g_scale": scale,
        "expected_total_g": person_g,
        "ref_total_g": ref_g,
        "calorie_lo": calorie_lo,
        "calorie_hi": calorie_hi,
        "upper_only_majors": upper_only,
        "bands": bands,
        "groups": groups,
        "meal_major_features": list(_hard_axes(model)),
        "source_path": model.get("_active_path"),
    }


def score_personal_category_presence(
    feat: dict[str, float],
    personal: dict[str, Any],
) -> dict[str, Any]:
    """
    Hard-check only food-type presence/absence (hi≤0 cells must be empty).

    Gram amounts outside [lo, hi] are allowed (Bronze); Silver may repair later.
    """
    fails: list[str] = []
    bands = personal.get("bands") or {}
    for f, band in bands.items():
        raw_hi = band.get("hi", 1e9)
        hi = float(1e9 if raw_hi is None else raw_hi)
        if hi > 0:
            continue
        v = float(feat.get(f, 0.0) or 0.0)
        if v > 1e-6:
            fails.append(f"{f}={v:.1f}>0 (forbidden type)")
    return {
        "ok": len(fails) == 0,
        "critical_fails": fails,
        "presence_only": True,
        "envelope_src": personal.get("stratum"),
        "g_scale": personal.get("g_scale"),
        "expected_total_g": personal.get("expected_total_g"),
        "ref_total_g": personal.get("ref_total_g"),
        "bands": bands,
        "by_eating_type": bool(personal.get("by_eating_type")),
        "personal": True,
    }


def score_personal_category_day(
    feat: dict[str, float],
    personal: dict[str, Any],
    *,
    presence_only: bool = False,
) -> dict[str, Any]:
    """Hard-check features against a pre-resolved personal category structure."""
    if presence_only:
        return score_personal_category_presence(feat, personal)
    upper_only = set(personal.get("upper_only_majors") or UPPER_ONLY_MAJORS)
    fails: list[str] = []
    bands = personal.get("bands") or {}
    for f, band in bands.items():
        parts = f.split("_")
        try:
            mi = int(parts[2])
        except (IndexError, ValueError):
            mi = 0
        v = float(feat.get(f, 0.0) or 0.0)
        lo = float(band["lo"])
        hi = float(band["hi"])
        if v > hi + 1e-6:
            fails.append(f"{f}={v:.1f}>{hi:.1f}")
        elif mi not in upper_only and lo > 0 and v < lo - 1e-6:
            fails.append(f"{f}={v:.1f}<{lo:.1f}")
    return {
        "ok": len(fails) == 0,
        "critical_fails": fails,
        "presence_only": False,
        "envelope_src": personal.get("stratum"),
        "g_scale": personal.get("g_scale"),
        "expected_total_g": personal.get("expected_total_g"),
        "ref_total_g": personal.get("ref_total_g"),
        "bands": bands,
        "by_eating_type": bool(personal.get("by_eating_type")),
        "personal": True,
    }


def format_personal_structure_brief(personal: dict[str, Any], *, majors: tuple[int, ...] = (2, 5, 7)) -> str:
    """One-line / multi-line summary for Bronze logs (meat/grain/veg by meal)."""
    lines = [
        f"personal structure: stratum={personal.get('stratum')} "
        f"g_scale={float(personal.get('g_scale') or 1):.3f} "
        f"expected_total_g={float(personal.get('expected_total_g') or 0):.0f} "
        f"bmi={personal.get('bmi')} sex={personal.get('sex')} age={personal.get('age')}"
    ]
    groups = personal.get("groups") or {}
    name = {2: "meat", 5: "grain", 7: "veg"}
    for meal in ("breakfast", "lunch", "dinner", "snack"):
        if meal not in groups:
            continue
        parts = []
        for mi in majors:
            g = groups[meal].get(str(mi))
            if not g:
                continue
            parts.append(f"{name.get(mi, mi)}={g['lo']:.0f}-{g['hi']:.0f}")
        if parts:
            lines.append(f"  {meal}: " + " ".join(parts))
    return "\n".join(lines)


def select_personal_category_for_bronze(
    participant_profile: dict[str, Any] | None,
    target_dict: dict[str, Any] | None = None,
    *,
    model: dict[str, Any] | None = None,
) -> Optional[dict[str, Any]]:
    """
    Bronze pre-GA hook: resolve category structure from basic demo features.
    Returns None if category prior unavailable.
    """
    prior = model if model is not None else get_active_category_prior()
    if prior is None:
        return None
    pp = dict(participant_profile or {})
    cal_lo = cal_hi = None
    if target_dict and "calorie" in target_dict:
        try:
            cal_lo = float(target_dict["calorie"][0])
            cal_hi = float(target_dict["calorie"][1])
        except (TypeError, ValueError, IndexError, KeyError):
            cal_lo = cal_hi = None
    return resolve_personal_category_structure(
        prior,
        sex=pp.get("sex", pp.get("gender")),
        bmi=pp.get("bmi"),
        age=pp.get("age"),
        total_g_ref=pp.get("total_g_ref"),
        calorie_lo=cal_lo,
        calorie_hi=cal_hi,
    )


def score_category_day(
    feat: dict[str, float],
    model: dict[str, Any],
    *,
    bmi: float | None = None,
    sex=None,
    age=None,
    total_g_ref: float | None = None,
    calorie_lo: float | None = None,
    calorie_hi: float | None = None,
    g_scale_clip: tuple[float, float] = (0.55, 2.4),
) -> dict[str, Any]:
    """
    Hard-check meal×major gram vector against BMI-stratum bands, scaled by expected day grams.
    Upper bound always; lower only when band p_lo > 0 and major not upper-only.
    """
    shim = {
        "overall": model["overall"],
        "strata": model.get("strata") or {},
        "ref_total_g": model.get("ref_total_g"),
        "ref_total_kcal": model.get("ref_total_kcal"),
    }
    envelope, envelope_src = _pick_envelope(shim, sex=sex, bmi=bmi, age=age)
    ref_g = float(model.get("ref_total_g") or (model["overall"].get("total_g") or {}).get("p50") or 2500.0)
    if envelope_src != "overall":
        ref_g = float((envelope.get("total_g") or {}).get("p50") or ref_g)
    person_g = expected_total_g(
        shim,
        bmi=bmi,
        sex=sex,
        age=age,
        total_g_ref=total_g_ref,
        calorie_lo=calorie_lo,
        calorie_hi=calorie_hi,
    )
    scale = person_g / ref_g if ref_g > 0 else 1.0
    scale = float(np.clip(scale, float(g_scale_clip[0]), float(g_scale_clip[1])))

    upper_only = set(model.get("upper_only_majors") or UPPER_ONLY_MAJORS)
    fails: list[str] = []
    bands_used: dict[str, dict[str, float]] = {}
    for f in _hard_axes(model):
        # parse major from feature name
        parts = f.split("_")
        # g_major_{i} or g_major_{i}_{meal}
        try:
            mi = int(parts[2])
        except (IndexError, ValueError):
            mi = 0
        v = float(feat.get(f, 0.0) or 0.0)
        band = envelope.get(f) or model["overall"].get(f)
        if not band:
            continue
        lo = float(band["p_lo"]) * scale
        hi = float(band["p_hi"]) * scale
        bands_used[f] = {"lo": lo, "hi": hi, "raw_lo": float(band["p_lo"]), "raw_hi": float(band["p_hi"])}
        if v > hi + 1e-6:
            fails.append(f"{f}={v:.1f}>{hi:.1f}")
        elif mi not in upper_only and lo > 0 and v < lo - 1e-6:
            fails.append(f"{f}={v:.1f}<{lo:.1f}")

    return {
        "ok": len(fails) == 0,
        "critical_fails": fails,
        "envelope_src": envelope_src,
        "g_scale": scale,
        "expected_total_g": person_g,
        "ref_total_g": ref_g,
        "bands": bands_used,
        "by_eating_type": bool(model.get("by_eating_type")),
        "personal": False,
    }
