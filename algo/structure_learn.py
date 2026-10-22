"""
Fit / apply learned day-structure priors (quantile envelope + IsolationForest).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from algo.structure_features import (
    FEATURE_NAMES,
    age_band_label,
    bmi_class_label,
    sex_label_stratum,
    stratum_key,
)

ROOT = Path(__file__).resolve().parents[1]

# Features that must stay inside univariate bands for Hard (critical axes).
CRITICAL_FEATURES = [
    "snack_kcal_over_max_meal",
    "max_slice_qty_meal",
    "max_fillet_qty_meal",
    "max_cereal_cup_day",
    "nonplate_g",
    "sugar_icing_g",
    "lunch_main_g",
    "dinner_main_g",
    "bread_g",
    "tortilla_g",
    "cereal_g",
    "max_mains_in_one_meal",
]


def _quantiles(s: pd.Series, lo: float, hi: float) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {"p_lo": 0.0, "p_hi": 0.0, "p50": 0.0, "n": 0}
    return {
        "p_lo": float(np.quantile(s, lo)),
        "p_hi": float(np.quantile(s, hi)),
        "p50": float(np.quantile(s, 0.50)),
        "p05": float(np.quantile(s, 0.05)),
        "p25": float(np.quantile(s, 0.25)),
        "p75": float(np.quantile(s, 0.75)),
        "p95": float(np.quantile(s, 0.95)),
        "mean": float(s.mean()),
        "n": int(len(s)),
    }


# When empirical p95 is 0 for a sparse stacking feature, use a sane floor so
# rare-but-real usage (e.g. 1 fillet) is not auto-rejected.
_SPARSE_UPPER_FLOOR = {
    "max_fillet_qty_meal": 2.0,
    "max_cereal_cup_day": 1.5,
    "max_slice_qty_meal": 3.0,
    "sugar_icing_g": 5.0,
    "nonplate_g": 0.0,  # keep strict: industrial non-plate should be ~0
    "plain_pasta_g": 200.0,
}


def fit_structure_prior(
    feat_df: pd.DataFrame,
    *,
    q_lo: float = 0.05,
    q_hi: float = 0.95,
    tau_pass: float = 0.95,
    iforest_estimators: int = 200,
    random_state: int = 42,
    min_stratum_n: int = 40,
) -> dict[str, Any]:
    """
    Fit overall (+ coarse stratum) quantile envelopes and a global IsolationForest.
    ``tau_pass`` = fraction of training days that should score as inliers.
    """
    X = feat_df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    overall = {f: _quantiles(X[f], q_lo, q_hi) for f in FEATURE_NAMES}
    for f, floor in _SPARSE_UPPER_FLOOR.items():
        if f not in overall:
            continue
        if overall[f]["p_hi"] < floor:
            overall[f]["p_hi"] = float(floor)
            overall[f]["p95"] = max(float(overall[f].get("p95", 0.0)), float(floor))

    # stratum envelopes (only if enough rows)
    feat_df = feat_df.copy()
    feat_df["_stratum"] = feat_df.apply(stratum_key, axis=1)
    strata: dict[str, Any] = {}
    for st, sub in feat_df.groupby("_stratum"):
        if len(sub) < min_stratum_n:
            continue
        Xs = sub[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        strata[str(st)] = {f: _quantiles(Xs[f], q_lo, q_hi) for f in FEATURE_NAMES}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    # contamination = 1 - tau so that ~tau are inliers in training
    contamination = float(np.clip(1.0 - tau_pass, 0.01, 0.5))
    clf = IsolationForest(
        n_estimators=iforest_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(Xs)
    raw_scores = clf.decision_function(Xs)  # higher = more normal
    # threshold: tau quantile of scores (keep top tau)
    thr = float(np.quantile(raw_scores, 1.0 - tau_pass))

    # summary table for humans
    summary_rows = []
    for f in FEATURE_NAMES:
        q = overall[f]
        summary_rows.append(
            {
                "feature": f,
                "n": q["n"],
                "mean": round(q["mean"], 4),
                "p05": round(q["p05"], 4),
                "p25": round(q["p25"], 4),
                "p50": round(q["p50"], 4),
                "p75": round(q["p75"], 4),
                "p95": round(q["p95"], 4),
                "band_lo": round(q["p_lo"], 4),
                "band_hi": round(q["p_hi"], 4),
                "critical": f in CRITICAL_FEATURES,
            }
        )

    ref_total_g = float(overall.get("total_g", {}).get("p50") or 2500.0)
    ref_total_kcal = float(overall.get("total_kcal", {}).get("p50") or 2200.0)
    return {
        "version": "1.1",
        "n_days": int(len(feat_df)),
        "q_lo": q_lo,
        "q_hi": q_hi,
        "tau_pass": tau_pass,
        "iforest_threshold": thr,
        "feature_names": list(FEATURE_NAMES),
        "critical_features": list(CRITICAL_FEATURES),
        "ref_total_g": ref_total_g,
        "ref_total_kcal": ref_total_kcal,
        "overall": overall,
        "strata": strata,
        "n_strata_kept": len(strata),
        "summary": summary_rows,
        "_scaler_mean": scaler.mean_.tolist(),
        "_scaler_scale": scaler.scale_.tolist(),
        "_iforest": clf,
    }


def save_structure_prior(model: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clf = model.get("_iforest")
    payload = {k: v for k, v in model.items() if not k.startswith("_")}
    (out_dir / "prior.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(model["summary"]).to_csv(out_dir / "feature_quantiles.csv", index=False)
    # sklearn dump
    import joblib

    joblib.dump(
        {
            "iforest": clf,
            "scaler_mean": model["_scaler_mean"],
            "scaler_scale": model["_scaler_scale"],
            "feature_names": model["feature_names"],
            "iforest_threshold": model["iforest_threshold"],
            "overall": model["overall"],
            "strata": model.get("strata") or {},
            "critical_features": model["critical_features"],
            "ref_total_g": model.get("ref_total_g"),
            "ref_total_kcal": model.get("ref_total_kcal"),
            "q_lo": model["q_lo"],
            "q_hi": model["q_hi"],
            "tau_pass": model["tau_pass"],
        },
        out_dir / "model.joblib",
    )


def load_structure_prior(out_dir: Path) -> dict[str, Any]:
    import joblib

    out_dir = Path(out_dir)
    blob = joblib.load(out_dir / "model.joblib")
    meta = json.loads((out_dir / "prior.json").read_text(encoding="utf-8"))
    meta["_iforest"] = blob["iforest"]
    meta["_scaler_mean"] = blob["scaler_mean"]
    meta["_scaler_scale"] = blob["scaler_scale"]
    return meta


def get_active_structure_prior(
    config_path: str | Path = "",
) -> Optional[dict[str, Any]]:
    """Load prior named by config/structure_learn.yaml ``active_corpus``."""
    cfg_path = Path(config_path) if config_path else ROOT / "config" / "structure_learn.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    active = str(cfg.get("active_corpus") or "full").strip().lower()
    paths = cfg.get("paths") or {}
    rel = paths.get(active) or f"outputs/structure_learn/{active}"
    out = Path(rel)
    if not out.is_absolute():
        out = ROOT / out
    if not (out / "model.joblib").exists():
        return None
    model = load_structure_prior(out)
    model["_active_corpus"] = active
    model["_config"] = cfg
    return model


def participant_stratum_key(
    *,
    sex=None,
    bmi=None,
    age=None,
    bmi_class: str | None = None,
) -> str:
    """Same scheme as ``structure_features.stratum_key`` for demo rows."""
    row = {
        "sex": sex,
        "gender": sex,
        "bmi": bmi,
        "bmi_class": bmi_class if bmi_class is not None else (bmi_class_label(bmi) if bmi is not None else "NA"),
        "age": age,
        "age_band": age_band_label(age) if age is not None else "NA",
    }
    return stratum_key(pd.Series(row))


def _pick_envelope(
    model: dict[str, Any],
    *,
    sex=None,
    bmi=None,
    age=None,
) -> tuple[dict[str, Any], str]:
    """Prefer BMI×sex×age stratum envelope; fall back to overall."""
    overall = model["overall"]
    strata = model.get("strata") or {}
    if not strata or bmi is None:
        return overall, "overall"
    key = participant_stratum_key(sex=sex, bmi=bmi, age=age)
    if key in strata:
        return strata[key], key
    # softer: same sex + BMI class, any age band (largest n)
    bc = bmi_class_label(bmi)
    sex_t = sex_label_stratum(sex)
    candidates = [
        k for k in strata if k.startswith(f"{sex_t}|") and f"|{bc}|" in k
    ]
    if not candidates:
        candidates = [k for k in strata if f"|{bc}|" in k]
    if candidates:
        def _n(k: str) -> int:
            return int((strata[k].get("total_g") or {}).get("n") or 0)

        key2 = max(candidates, key=_n)
        return strata[key2], key2
    return overall, "overall"


def expected_total_g(
    model: dict[str, Any],
    *,
    bmi=None,
    sex=None,
    age=None,
    total_g_ref: float | None = None,
    calorie_lo: float | None = None,
    calorie_hi: float | None = None,
) -> float:
    """
    Participant-expected day food mass (g).

    Priority:
      1) explicit ``total_g_ref`` (e.g. person's mean NHANES day grams)
      2) BMI×sex×age stratum p50(total_g), optionally re-scaled by
         calorie_mid / stratum p50(total_kcal) so high-EER days get more mass room
      3) overall corpus p50
    """
    if total_g_ref is not None and float(total_g_ref) > 0:
        return float(total_g_ref)
    env, _src = _pick_envelope(model, sex=sex, bmi=bmi, age=age)
    g = float((env.get("total_g") or {}).get("p50") or 0.0)
    kcal = float((env.get("total_kcal") or {}).get("p50") or 0.0)
    if g <= 0:
        g = float(model.get("ref_total_g") or (model["overall"].get("total_g") or {}).get("p50") or 2500.0)
        kcal = float(
            model.get("ref_total_kcal")
            or (model["overall"].get("total_kcal") or {}).get("p50")
            or 2200.0
        )
    cal_mid = None
    if calorie_lo is not None and calorie_hi is not None:
        cal_mid = 0.5 * (float(calorie_lo) + float(calorie_hi))
    elif calorie_lo is not None:
        cal_mid = float(calorie_lo)
    elif calorie_hi is not None:
        cal_mid = float(calorie_hi)
    if cal_mid is not None and cal_mid > 0 and kcal > 0:
        # same energy density as stratum median day → grams for this calorie demand
        return g * (cal_mid / kcal)
    return g


# Absolute mass / stacking axes: scale lo+hi with participant total_g / corpus (or stratum) ref.
_G_SCALED = {
    "bread_g",
    "tortilla_g",
    "cereal_g",
    "plain_pasta_g",
    "cracker_g",
    "chip_g",
    "nonplate_g",
    "sugar_icing_g",
    "lunch_main_g",
    "dinner_main_g",
    "main_grams_total",
    "total_g",
    "max_slice_qty_meal",
    "max_fillet_qty_meal",
    "max_cereal_cup_day",
}
# Ratios / counts / shares: do not mass-scale.
_NO_SCALE = {
    "snack_kcal_over_max_meal",
    "max_mains_in_one_meal",
    "has_bld",
    "n_mains",
    "n_unique_foods",
    "n_items",
    "n_food_groups",
    "share_g_breakfast",
    "share_g_lunch",
    "share_g_dinner",
    "share_g_snack",
    "share_kcal_breakfast",
    "share_kcal_lunch",
    "share_kcal_dinner",
    "share_kcal_snack",
    "total_kcal",
}


def score_day_features(
    feat: dict[str, float],
    model: dict[str, Any],
    *,
    require_iforest: bool = True,
    calorie_lo: float | None = None,
    calorie_hi: float | None = None,
    calorie_ref: float = 2500.0,  # legacy alias; prefer calorie_lo/hi mid
    bmi: float | None = None,
    sex=None,
    age=None,
    total_g_ref: float | None = None,
    g_scale_clip: tuple[float, float] = (0.55, 2.4),
) -> dict[str, Any]:
    """Return pass/fail + reasons for one feature dict.

    Bands are **not** one-size-fits-all:
      1) prefer BMI×sex×age stratum envelope when available;
      2) scale mass/stacking axes by ``expected_total_g / stratum_or_corpus_ref_g``
         (BMI day-grams × calorie-mid density, or explicit total_g_ref).
    """
    envelope, envelope_src = _pick_envelope(model, sex=sex, bmi=bmi, age=age)
    ref_g = float(
        model.get("ref_total_g")
        or (model["overall"].get("total_g") or {}).get("p50")
        or 2500.0
    )
    # Stratum-local reference when using a stratum envelope
    if envelope_src != "overall":
        ref_g = float((envelope.get("total_g") or {}).get("p50") or ref_g)
    person_g = expected_total_g(
        model,
        bmi=bmi,
        sex=sex,
        age=age,
        total_g_ref=total_g_ref,
        calorie_lo=calorie_lo,
        calorie_hi=calorie_hi,
    )
    g_scale = person_g / ref_g if ref_g > 0 else 1.0
    lo_c, hi_c = float(g_scale_clip[0]), float(g_scale_clip[1])
    scale = float(np.clip(g_scale, lo_c, hi_c))
    _ = calorie_ref  # kept for call-site compatibility

    critical = set(model.get("critical_features") or CRITICAL_FEATURES)
    band_hits = []
    critical_fails = []
    upper_only = {
        "snack_kcal_over_max_meal",
        "max_slice_qty_meal",
        "max_fillet_qty_meal",
        "max_cereal_cup_day",
        "nonplate_g",
        "sugar_icing_g",
        "bread_g",
        "cereal_g",
        "tortilla_g",
        "plain_pasta_g",
        "cracker_g",
        "chip_g",
        "max_mains_in_one_meal",
    }
    lower_only = {"lunch_main_g", "dinner_main_g", "main_grams_total", "has_bld"}

    for f in model["feature_names"]:
        v = float(feat.get(f, 0.0))
        band = envelope.get(f) or model["overall"][f]
        lo, hi = float(band["p_lo"]), float(band["p_hi"])
        if f in _G_SCALED and f not in _NO_SCALE and scale != 1.0:
            lo = lo * scale
            hi = hi * scale
        msg = None
        if f in lower_only:
            if lo > 0 and v < lo:
                msg = f"{f}={v:.3f}<{lo:.3f}"
        elif f in upper_only:
            if v > hi:
                msg = f"{f}={v:.3f}>{hi:.3f}"
        else:
            if v < lo or v > hi:
                msg = f"{f}={v:.3f} not in [{lo:.3f},{hi:.3f}]"
        if msg:
            band_hits.append(msg)
            if f in critical:
                critical_fails.append(msg)

    # iforest (global; not stratum-scaled)
    vec = np.array([[float(feat.get(f, 0.0)) for f in model["feature_names"]]], dtype=float)
    mean = np.array(model["_scaler_mean"], dtype=float)
    scaler_scale = np.array(model["_scaler_scale"], dtype=float)
    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    Xs = (vec - mean) / scaler_scale
    score = float(model["_iforest"].decision_function(Xs)[0])
    thr = float(model["iforest_threshold"])
    iforest_ok = score >= thr
    if require_iforest and not iforest_ok:
        critical_fails.append(f"iforest_score={score:.4f}<thr={thr:.4f}")

    ok = len(critical_fails) == 0
    return {
        "ok": ok,
        "iforest_score": score,
        "iforest_ok": iforest_ok,
        "n_band_violations": len(band_hits),
        "critical_fails": critical_fails,
        "band_violations": band_hits[:12],
        "envelope_src": envelope_src,
        "g_scale": scale,
        "expected_total_g": person_g,
        "ref_total_g": ref_g,
    }
