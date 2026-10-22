"""
Diet structure priors, meal mapping, and food-set fingerprints for GPU-GA v2.

Repetition rule (project): two days are duplicates if frozenset(food_id) is equal,
regardless of grams / portions.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "ga_multiobj_weights.yaml"

# NHANES eating_type → canonical meal (English + Spanish remap)
OCCASION_TO_MEAL = {
    1: "breakfast",
    2: "lunch",
    3: "dinner",
    4: "dinner",  # supper
    5: "breakfast",  # brunch
    6: "snack",
    7: "snack",  # drink
    8: "snack",  # infant feeding / other — treat as snack for adult plate hygiene
    9: "snack",  # extended consumption / other
    10: "breakfast",
    11: "lunch",
    12: "dinner",
    13: "snack",
    14: "dinner",
    15: "snack",
    16: "snack",
    17: "snack",
    18: "snack",
    19: "snack",
    91: "snack",  # other
    99: "snack",  # don't know / refused → snack bucket
}

MEAL_TO_ID = {"breakfast": 1, "lunch": 2, "dinner": 3, "snack": 4}
ID_TO_MEAL = {v: k for k, v in MEAL_TO_ID.items()}

SPANISH_TO_ENGLISH = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    10: 1, 11: 2, 12: 3, 13: 6, 14: 4, 15: 6, 16: 6, 17: 6, 18: 6, 19: 7,
}


def load_ga_config(path: Optional[Path] = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def meal_from_eating_type(eating_type) -> Optional[str]:
    if pd.isna(eating_type):
        return None
    try:
        code = int(float(eating_type))
    except (TypeError, ValueError):
        return None
    code = SPANISH_TO_ENGLISH.get(code, code)
    return OCCASION_TO_MEAL.get(code)


def meal_id_from_eating_type(eating_type) -> int:
    meal = meal_from_eating_type(eating_type)
    return MEAL_TO_ID.get(meal, 0) if meal else 0


def food_group_code(food_id) -> int:
    """FNDDS major group = first digit of 8-digit food code."""
    if pd.isna(food_id):
        return 0
    s = str(int(float(food_id))).zfill(8)
    try:
        return int(s[0])
    except ValueError:
        return 0


def food_id_fingerprint(food_ids: Iterable) -> frozenset:
    """Duplicate definition: same set of food kinds (ids), ignoring grams."""
    out = set()
    for fid in food_ids:
        if pd.isna(fid):
            continue
        out.add(int(float(fid)))
    return frozenset(out)


def fingerprint_hash(fp: frozenset) -> str:
    payload = ",".join(str(x) for x in sorted(fp))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def bmi_class(bmi: float) -> str:
    if pd.isna(bmi):
        return "unknown"
    if bmi < 18.5:
        return "underweight (<18.5)"
    if bmi < 25:
        return "normal (18.5–24.9)"
    if bmi < 30:
        return "overweight (25–29.9)"
    return "obesity (≥30)"


def age_band(age: float) -> str:
    if pd.isna(age):
        return "unknown"
    a = float(age)
    if a < 31:
        return "19–30"
    if a < 51:
        return "31–50"
    if a < 71:
        return "51–70"
    return "71+"


def sex_label(gender) -> str:
    try:
        g = str(int(float(gender)))
    except (TypeError, ValueError):
        return "unknown"
    return {"1": "Male", "2": "Female"}.get(g, "unknown")


@lru_cache(maxsize=1)
def _load_frequency_tables(config_path: str) -> tuple[pd.DataFrame, set[int], set[int]]:
    cfg = load_ga_config(Path(config_path) if config_path else None)
    freq_path = ROOT / cfg["paths"]["food_frequency"]
    if not freq_path.exists():
        return pd.DataFrame(), set(), set()
    freq = pd.read_csv(freq_path)
    freq["food_id"] = pd.to_numeric(freq["food_id"], errors="coerce").astype("Int64")
    common_min = float(cfg["pool"]["common_pct_days_min"])
    uncommon_max = float(cfg["pool"]["uncommon_pct_days_max"])
    common = set(freq.loc[freq["pct_days"] >= common_min, "food_id"].dropna().astype(int))
    uncommon = set(freq.loc[freq["pct_days"] < uncommon_max, "food_id"].dropna().astype(int))
    return freq, common, uncommon


def get_common_uncommon_sets(config: Optional[dict] = None) -> tuple[set[int], set[int]]:
    cfg = config or load_ga_config()
    _, common, uncommon = _load_frequency_tables(str(DEFAULT_CONFIG))
    return common, uncommon


_WEIGHT_KEYS = (
    "daily_weight_g_p5",
    "daily_weight_g_p25",
    "daily_weight_g_p50",
    "daily_weight_g_p75",
    "daily_weight_g_p95",
)
_SHARE_KEYS = (
    "share_g_breakfast",
    "share_g_lunch",
    "share_g_dinner",
    "share_g_snack",
)


def _row_to_prior(row: pd.Series) -> dict[str, float]:
    return {
        "share_g_breakfast": float(row.get("share_g_breakfast_mean", 0.20)),
        "share_g_lunch": float(row.get("share_g_lunch_mean", 0.22)),
        "share_g_dinner": float(row.get("share_g_dinner_mean", 0.36)),
        "share_g_snack": float(row.get("share_g_snack_mean", 0.19)),
        "daily_weight_g_p5": float(row.get("daily_weight_g_p5", 200)),
        "daily_weight_g_p25": float(row.get("daily_weight_g_p25", 700)),
        "daily_weight_g_p50": float(row.get("daily_weight_g_p50", 1700)),
        "daily_weight_g_p75": float(row.get("daily_weight_g_p75", 2500)),
        "daily_weight_g_p95": float(row.get("daily_weight_g_p95", 4000)),
    }


@lru_cache(maxsize=8)
def _read_structure_csv(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def demographics_from_user_row(row: Any) -> dict[str, Any]:
    """Extract sex label / BMI / age from a users.csv-like row."""
    if row is None:
        return {"sex": None, "bmi": None, "age": None}

    def _get(key):
        try:
            return row.get(key)
        except Exception:
            try:
                return row[key]
            except Exception:
                return None

    sex = sex_label(_get("gender") if _get("gender") is not None else _get("sex"))
    if sex == "unknown":
        sex = None
    bmi = None
    raw_bmi = _get("bmi")
    if raw_bmi is not None and pd.notna(raw_bmi):
        bmi = float(raw_bmi)
    else:
        w, h = _get("weight"), _get("height")
        if w is not None and h is not None and pd.notna(w) and pd.notna(h) and float(h) > 0:
            bmi = float(w) / (float(h) / 100.0) ** 2
    age = None
    raw_age = _get("age")
    if raw_age is not None and pd.notna(raw_age):
        age = float(raw_age)
    return {"sex": sex, "bmi": bmi, "age": age}


def load_structure_prior(
    sex: Optional[str] = None,
    bmi: Optional[float] = None,
    age: Optional[float] = None,
    config: Optional[dict] = None,
) -> dict[str, float]:
    """
    Meal-share + daily-weight priors stratified by sex / BMI / age.

    Meal shares: prefer sex×bmi → sex → overall.
    Weight band: intersect matched strata (min of uppers p75/p95; max of
    floors p5/p25) so the hard daily-weight cap reflects all available
    demographics. Falls back to overall when a stratum is missing.
    """
    cfg = config or load_ga_config()
    paths = cfg.get("paths") or {}

    default: dict[str, Any] = {
        "share_g_breakfast": 0.20,
        "share_g_lunch": 0.22,
        "share_g_dinner": 0.36,
        "share_g_snack": 0.19,
        "daily_weight_g_p5": 264.0,
        "daily_weight_g_p25": 740.0,
        "daily_weight_g_p50": 1707.0,
        "daily_weight_g_p75": 2505.0,
        "daily_weight_g_p95": 4056.0,
        "weight_stratum": "overall",
        "sex": sex or "",
        "bmi_class": "",
        "age_band": "",
    }

    overall_path = ROOT / paths.get("structure_overall", "")
    if overall_path.exists():
        ov = _read_structure_csv(str(overall_path))
        if len(ov):
            default.update(_row_to_prior(ov.iloc[0]))

    bc = bmi_class(bmi) if bmi is not None and not (isinstance(bmi, float) and np.isnan(bmi)) else None
    if bc == "unknown":
        bc = None
    ab = age_band(age) if age is not None and not (isinstance(age, float) and np.isnan(age)) else None
    if ab == "unknown":
        ab = None
    default["bmi_class"] = bc or ""
    default["age_band"] = ab or ""

    matched: list[tuple[str, dict[str, float]]] = []

    by_sex_path = ROOT / paths.get("structure_by_sex", "")
    if sex and by_sex_path.exists():
        sdf = _read_structure_csv(str(by_sex_path))
        hit = sdf[sdf["sex"] == sex] if "sex" in sdf.columns else sdf.iloc[0:0]
        if len(hit):
            matched.append((f"sex={sex}", _row_to_prior(hit.iloc[0])))

    by_bmi_path = ROOT / paths.get("structure_by_bmi", "")
    if bc and by_bmi_path.exists():
        bdf = _read_structure_csv(str(by_bmi_path))
        hit = bdf[bdf["bmi_class"] == bc] if "bmi_class" in bdf.columns else bdf.iloc[0:0]
        if len(hit):
            matched.append((f"bmi={bc}", _row_to_prior(hit.iloc[0])))

    by_age_path = ROOT / paths.get("structure_by_age", "")
    if ab and by_age_path.exists():
        adf = _read_structure_csv(str(by_age_path))
        hit = adf[adf["age_band"] == ab] if "age_band" in adf.columns else adf.iloc[0:0]
        if len(hit):
            matched.append((f"age={ab}", _row_to_prior(hit.iloc[0])))

    # sex×bmi interaction (optional table) — also used for meal-share preference
    sex_bmi_prior: Optional[dict[str, float]] = None
    by_sex_bmi_path = ROOT / paths.get(
        "structure_by_sex_bmi",
        "outputs/nhanes_food_only_operability/table_structure_by_sex_bmi.csv",
    )
    if sex and bc and by_sex_bmi_path.exists():
        sb = _read_structure_csv(str(by_sex_bmi_path))
        if {"sex", "bmi_class"}.issubset(sb.columns):
            hit = sb[(sb["sex"] == sex) & (sb["bmi_class"] == bc)]
            if len(hit):
                sex_bmi_prior = _row_to_prior(hit.iloc[0])
                matched.append((f"sex×bmi={sex}|{bc}", sex_bmi_prior))

    # Meal shares: most specific available
    if sex_bmi_prior is not None:
        for k in _SHARE_KEYS:
            default[k] = sex_bmi_prior[k]
    elif matched:
        # first match is sex if present
        for tag, prior in matched:
            if tag.startswith("sex="):
                for k in _SHARE_KEYS:
                    default[k] = prior[k]
                break

    # Weight band from matched strata only (overall is fallback when none match).
    # If sex×bmi exists, drop plain sex/bmi marginals to avoid double-counting;
    # always keep age when available.
    weight_sources = list(matched)
    if sex_bmi_prior is not None:
        weight_sources = [
            (tag, prior)
            for tag, prior in weight_sources
            if not (tag.startswith("sex=") or tag.startswith("bmi="))
        ]
    if not weight_sources:
        weight_sources = [("overall", {k: float(default[k]) for k in _WEIGHT_KEYS})]

    # Intersect: tighter upper (min p75/p95), tighter floor (max p5/p25); p50 = median
    for key in ("daily_weight_g_p75", "daily_weight_g_p95"):
        default[key] = float(min(prior[key] for _, prior in weight_sources))
    for key in ("daily_weight_g_p5", "daily_weight_g_p25"):
        default[key] = float(max(prior[key] for _, prior in weight_sources))
    p50s = sorted(prior["daily_weight_g_p50"] for _, prior in weight_sources)
    default["daily_weight_g_p50"] = float(p50s[len(p50s) // 2])
    default["weight_stratum"] = "+".join(tag for tag, _ in weight_sources)
    return default


def day_structure_metrics(selected_df: pd.DataFrame) -> dict[str, Any]:
    """Compute meal grams/shares and diversity stats for a selected day DataFrame."""
    df = selected_df.copy()
    if "meal" not in df.columns:
        df["meal"] = df["eating_type"].map(meal_from_eating_type)
    if "grams" not in df.columns:
        raise ValueError("selected_df must include grams")
    grams = pd.to_numeric(df["grams"], errors="coerce").fillna(0.0)
    meals = df["meal"].fillna("snack")
    meal_g = {m: float(grams[meals == m].sum()) for m in ("breakfast", "lunch", "dinner", "snack")}
    total = sum(meal_g.values())
    shares = {f"share_g_{m}": (meal_g[m] / total if total > 0 else 0.0) for m in meal_g}
    food_ids = pd.to_numeric(df["food_id"], errors="coerce").dropna().astype(int)
    groups = {food_group_code(x) for x in food_ids}
    groups.discard(0)
    present = {m for m, g in meal_g.items() if g > 0}
    fp = food_id_fingerprint(food_ids)
    return {
        "meal_g": meal_g,
        "total_g": total,
        **shares,
        "n_unique_foods": int(food_ids.nunique()),
        "n_unique_food_groups": len(groups),
        "n_unique_meals": len(present),
        "fingerprint": fp,
        "fingerprint_hash": fingerprint_hash(fp),
        "food_ids": list(fp),
    }


def structure_fit(metrics: dict, prior: dict, config: Optional[dict] = None) -> float:
    """Soft [0, 1] score: meal gram shares + optional daily weight band."""
    cfg = config or load_ga_config()
    tol = float(cfg["structure"]["share_tolerance"])
    parts = []
    for m in ("breakfast", "lunch", "dinner", "snack"):
        key = f"share_g_{m}"
        obs = float(metrics.get(key, 0.0))
        tgt = float(prior.get(key, 0.25))
        parts.append(max(0.0, 1.0 - abs(obs - tgt) / tol))
    share_score = float(np.mean(parts)) if parts else 0.0
    if not cfg["structure"].get("use_weight_band", True):
        return share_score
    w = float(metrics.get("total_g", 0.0))
    p5 = float(prior["daily_weight_g_p5"])
    p25 = float(prior["daily_weight_g_p25"])
    p75 = float(prior["daily_weight_g_p75"])
    p95 = float(prior["daily_weight_g_p95"])
    if p25 <= w <= p75:
        w_score = 1.0
    elif w < p25:
        w_score = max(0.0, (w - p5) / (p25 - p5 + 1e-9))
    else:
        w_score = max(0.0, (p95 - w) / (p95 - p75 + 1e-9))
    return 0.7 * share_score + 0.3 * w_score


def daily_weight_hard_max(prior: Optional[dict] = None, config: Optional[dict] = None) -> float:
    """Hard upper bound on day total grams (NHANES prior percentile or absolute)."""
    cfg = config or load_ga_config()
    st = cfg.get("structure") or {}
    if st.get("daily_weight_g_max") is not None:
        return float(st["daily_weight_g_max"])
    prior = prior or {}
    pct = str(st.get("daily_weight_g_max_percentile", "p75"))
    key = pct if pct.startswith("daily_weight_g_") else f"daily_weight_g_{pct}"
    if key in prior and prior[key] is not None:
        return float(prior[key])
    return float(st.get("daily_weight_g_max_fallback", 2505.0))


def layer_b_score(
    selected_df: pd.DataFrame,
    prior: dict,
    uncommon_ids: Optional[set[int]] = None,
    config: Optional[dict] = None,
) -> dict[str, float]:
    """Scalar Layer-B objectives for a feasible day."""
    cfg = config or load_ga_config()
    w = cfg["weights"]
    metrics = day_structure_metrics(selected_df)
    pos = float(pd.to_numeric(selected_df.get("positive", 0), errors="coerce").fillna(0).sum())
    neg = float(pd.to_numeric(selected_df.get("negative", 0), errors="coerce").fillna(0).sum())
    uncommon_ids = uncommon_ids if uncommon_ids is not None else get_common_uncommon_sets(cfg)[1]
    n_uncommon = sum(1 for fid in metrics["food_ids"] if fid in uncommon_ids)
    s_fit = structure_fit(metrics, prior, cfg)
    n_foods = int(metrics["n_unique_foods"])
    n_tgt = float(w.get("w_n_foods_target", 11))
    n_band = 2.0
    n_dev = max(0.0, abs(n_foods - n_tgt) - n_band)
    score = (
        w["w_positive_negative"] * (pos - neg)
        + w["w_food_groups"] * metrics["n_unique_food_groups"]
        + float(w.get("w_unique_foods", 0.0)) * metrics["n_unique_foods"]
        + w["w_meals"] * metrics["n_unique_meals"]
        + w["w_structure_fit"] * s_fit
        - w["w_uncommon_penalty"] * n_uncommon
        - float(w.get("w_n_foods_deviation", 0.0)) * n_dev
    )
    return {
        "score": float(score),
        "structure_score": float(s_fit),
        "diversity_score": float(
            metrics["n_unique_food_groups"] + metrics["n_unique_meals"] + 0.1 * metrics["n_unique_foods"]
        ),
        "positive_negative": float(pos - neg),
        "n_uncommon": float(n_uncommon),
        "fingerprint_hash": metrics["fingerprint_hash"],
        "fingerprint": metrics["fingerprint"],
        **{k: metrics[k] for k in (
            "total_g", "n_unique_food_groups", "n_unique_meals", "n_unique_foods",
            "share_g_breakfast", "share_g_lunch", "share_g_dinner", "share_g_snack",
        )},
    }


def enrich_food_dataframe(food_df: pd.DataFrame) -> pd.DataFrame:
    """Add meal_id, food_group_code, is_common columns."""
    df = food_df.copy()
    if "eating_type" in df.columns:
        df["eating_type"] = pd.to_numeric(df["eating_type"], errors="coerce")
        df["eating_type"] = df["eating_type"].map(lambda x: SPANISH_TO_ENGLISH.get(int(x), int(x)) if pd.notna(x) else x)
    df["meal"] = df["eating_type"].map(meal_from_eating_type)
    df["meal_id"] = df["eating_type"].map(meal_id_from_eating_type).fillna(0).astype(int)
    df["food_group_code"] = df["food_id"].map(food_group_code).astype(int)
    common, _ = get_common_uncommon_sets()
    fids = pd.to_numeric(df["food_id"], errors="coerce")
    df["is_common"] = fids.isin(common).astype(int)
    for col in ("positive", "negative", "grams"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


@lru_cache(maxsize=1)
def _food_desc_cache() -> dict[int, str]:
    for rel in ("data/nhanes/data/food_code.csv", "data/nhanes/food_code.csv"):
        p = ROOT / rel
        if p.exists():
            fc = pd.read_csv(p, usecols=["food_id", "food_desc"])
            return dict(zip(fc.food_id.astype(int), fc.food_desc.astype(str)))
    return {}


def target_needs_low_na_high_k(target_dict: dict) -> bool:
    """BP-like hard pair: finite sodium upper + potassium lower bound."""
    if "sodium" not in target_dict or "potassium" not in target_dict:
        return False
    na_lo, na_hi = target_dict["sodium"]
    k_lo, k_hi = target_dict["potassium"]
    try:
        return float(na_hi) < 1e8 and float(k_lo) > 0
    except (TypeError, ValueError):
        return False


def low_na_high_k_mask(
    df: pd.DataFrame,
    sodium_per_100g_max: float = 140.0,
    potassium_per_100g_min: float = 200.0,
    grams_min: float = 20.0,
    potassium_abs_min: float = 150.0,
) -> pd.Series:
    """Row mask by nutrient density + usable portion (skip dry powders / crumbs)."""
    if df.empty or "sodium" not in df.columns or "potassium" not in df.columns:
        return pd.Series(False, index=df.index)
    g = pd.to_numeric(df["grams"], errors="coerce")
    na_abs = pd.to_numeric(df["sodium"], errors="coerce").fillna(0.0)
    k_abs = pd.to_numeric(df["potassium"], errors="coerce").fillna(0.0)
    gg = g.replace(0, np.nan)
    na = na_abs / gg * 100.0
    k = k_abs / gg * 100.0
    return (
        (g >= float(grams_min))
        & (k_abs >= float(potassium_abs_min))
        & (na <= float(sodium_per_100g_max))
        & (k >= float(potassium_per_100g_min))
    )


def _positive_nutrient_coverage(
    df: pd.DataFrame,
    target_dict: dict,
    k: int = 40,
    *,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Ensure pool coverage for nutrients with lo>0: sample foods with amount > 0.

    No density ranking (avoids bran-cereal / cheat-food monopolies).
    """
    rng = rng or np.random.default_rng(0)
    parts = []
    grams = pd.to_numeric(df.get("grams"), errors="coerce").fillna(0.0)
    base = df[(grams >= 5) & (grams <= 500)].copy() if len(df) else df
    if base.empty:
        base = df
    for nut, bounds in target_dict.items():
        if nut not in base.columns:
            continue
        try:
            lo_f = float(bounds[0])
        except (TypeError, ValueError, IndexError):
            continue
        if lo_f <= 0:
            continue
        amt = pd.to_numeric(base[nut], errors="coerce").fillna(0.0)
        pos = base.loc[amt > 0]
        if pos.empty:
            continue
        take_n = min(int(k), len(pos))
        # Stratify by food_group when possible for diversity
        if "food_group_code" in pos.columns and take_n > 1:
            picked_idx: list = []
            groups = list(pos.groupby("food_group_code", sort=False))
            rng.shuffle(groups)
            per_g = max(1, take_n // max(1, len(groups)))
            for _, gdf in groups:
                n = min(len(gdf), per_g)
                if n <= 0:
                    continue
                chosen = gdf.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))
                picked_idx.extend(chosen.index.tolist())
                if len(picked_idx) >= take_n:
                    break
            if len(picked_idx) < take_n:
                rest = pos.drop(index=picked_idx, errors="ignore")
                need = take_n - len(picked_idx)
                if len(rest) and need > 0:
                    picked_idx.extend(
                        rest.sample(
                            n=min(need, len(rest)),
                            random_state=int(rng.integers(0, 2**31 - 1)),
                        ).index.tolist()
                    )
            parts.append(pos.loc[picked_idx[:take_n]])
        else:
            parts.append(
                pos.sample(n=take_n, random_state=int(rng.integers(0, 2**31 - 1)))
            )
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=False).drop_duplicates()


# Backward-compatible name (density boosters removed)
def _top_nutrient_boosters(df: pd.DataFrame, target_dict: dict, k: int = 40) -> pd.DataFrame:
    return _positive_nutrient_coverage(df, target_dict, k=k)


def greedy_pack_day(
    food_df: pd.DataFrame,
    target_dict: dict,
    n_max: int = 30,
    rng: Optional[np.random.Generator] = None,
    cand_sample: int = 6000,
) -> tuple[pd.DataFrame, dict[str, float], bool]:
    """
    Greedy day packer: maximize deficit help on protein/carb/vitD/folate/vitC
    while respecting hard uppers (sugar/sodium/sat_fat/chol/iron).
    Returns (selected_df, nutrient_sums, all_in_range).
    """
    rng = rng or np.random.default_rng(0)
    cols = [k for k in target_dict if k in food_df.columns]
    if not cols or food_df.empty:
        return food_df.iloc[0:0].copy(), {}, False
    need = {k: float(target_dict[k][0]) for k in cols}
    cap = {k: float(target_dict[k][1]) for k in cols}
    hard_upper = {"sugar", "sodium", "saturated_fat", "cholesterol", "iron"}
    heavy = {"protein", "carb", "vitamin_d", "folic_acid", "vitamin_c", "fiber"}
    P = food_df.reset_index(drop=True)
    vals = {k: pd.to_numeric(P[k], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) for k in cols}
    used = np.zeros(len(P), dtype=bool)
    sums = {k: 0.0 for k in cols}
    chosen: list[int] = []
    for _ in range(min(n_max, len(P))):
        deficit = {k: max(0.0, need[k] - sums[k]) for k in cols}
        if sum(deficit.values()) < 1e-6:
            break
        headroom = {k: max(0.0, cap[k] - sums[k]) for k in cols}
        cand = np.where(~used)[0]
        if len(cand) > cand_sample:
            cand = rng.choice(cand, cand_sample, replace=False)
        best_i, best_s = None, -1e99
        for i in cand:
            overflow = 0.0
            help_ = 0.0
            pressure = 0.0
            ok = True
            for k in cols:
                v = float(vals[k][i])
                if v > headroom[k] + 1e-9:
                    if k in hard_upper:
                        ok = False
                        break
                    overflow += v - headroom[k]
                if deficit[k] > 0:
                    w = 3.0 if k in heavy else 1.0
                    help_ += w * min(v, deficit[k])
                if k in hard_upper and headroom[k] < 1e9:
                    pressure += v / max(headroom[k], 1.0)
            if not ok:
                continue
            score = help_ - 2.0 * overflow - 0.3 * pressure
            if score > best_s:
                best_s, best_i = score, int(i)
        if best_i is None:
            break
        used[best_i] = True
        chosen.append(best_i)
        for k in cols:
            sums[k] += float(vals[k][best_i])
    sel = P.iloc[chosen].copy() if chosen else P.iloc[0:0].copy()
    ok = all(need[k] <= sums[k] <= cap[k] for k in cols)
    return sel, sums, ok


def _pool_rng_seed(target_dict: dict, pool_seed_offset: int = 0) -> int:
    """Stable seed so the same target rebuilds the same candidate pool."""
    import hashlib
    import json

    raw = json.dumps(
        {k: [float(v[0]), float(v[1])] for k, v in sorted(target_dict.items())},
        sort_keys=True,
    )
    h = hashlib.sha1(f"{raw}|{int(pool_seed_offset)}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def filter_candidate_pool(
    food_df: pd.DataFrame,
    target_dict: dict,
    user_history_food_ids: Optional[Iterable[int]] = None,
    config: Optional[dict] = None,
    pool_seed_offset: int = 0,
) -> pd.DataFrame:
    """
    Nutrient upper prefilter + moderate portions + balanced B/L/D/snack quotas
    + soft capability ranking + lo>0 nutrient coverage (amount>0 sample, no density top-k)
    + food-group stratification for diversity.

    Important: do NOT globally head() after concat — that previously wiped lunch/dinner
    when breakfast dominated the common-food set.
    """
    cfg = config or load_ga_config()
    need_enrich = not {"meal", "meal_id", "food_group_code", "is_common"}.issubset(food_df.columns)
    df = enrich_food_dataframe(food_df) if need_enrich else food_df

    already_plate = "meal_tags" in df.columns
    try:
        from algo.bronze_food_filter import load_bronze_exclusion_config

        excl_cfg = load_bronze_exclusion_config()
        div_cfg = dict(excl_cfg.get("diversity") or {})
        if not already_plate:
            from algo.bronze_food_filter import filter_bronze_food_df

            df, _ = filter_bronze_food_df(df, cfg=excl_cfg)
    except Exception:
        div_cfg = {}

    # Meal-tag safety (load_data already expands; drop any mismatched rows)
    if "meal_tags" in df.columns and "meal" in df.columns and len(df):
        from algo.food_meal_tags import MEALS as _MEALS

        mt = df["meal_tags"].fillna("").astype(str)
        meal = df["meal"].astype(str).str.lower()
        ok = mt.eq("") | mt.eq("nan")
        for meal_name in _MEALS:
            ok = ok | ((meal == meal_name) & mt.str.contains(rf"(?:^|\|){meal_name}(?:\||$)", regex=True))
        df = df.loc[ok].copy()

    # single-food cannot exceed nutrient upper
    for key, value in target_dict.items():
        if key not in df.columns:
            continue
        df = df[pd.to_numeric(df[key], errors="coerce").fillna(0) <= float(value[1])]
    if df.empty:
        return df

    gmin = float(cfg["pool"].get("grams_min", 5))
    gmax = float(cfg["pool"].get("grams_max", 500))
    grams = pd.to_numeric(df["grams"], errors="coerce").fillna(0.0)
    moderate = df[(grams >= gmin) & (grams <= gmax)]
    if len(moderate) >= 1000:
        df = moderate

    common, _ = get_common_uncommon_sets(cfg)
    hist = set(int(x) for x in (user_history_food_ids or []) if pd.notna(x))
    prefer = cfg["pool"].get("prefer_common", True)
    min_per_meal = int(cfg["pool"].get("min_per_meal", 200))
    max_pool = int(cfg["pool"].get("max_pool_size", 15000))
    meal_quota = {
        "breakfast": max(min_per_meal, max_pool // 4),
        "lunch": max(min_per_meal, max_pool // 4),
        "dinner": max(min_per_meal, max_pool // 4),
        "snack": max(min_per_meal // 2, max_pool // 8),
    }
    boost_k = int(cfg["pool"].get("nutrient_booster_k", 80))
    common_bonus = float(div_cfg.get("common_rank_bonus", 0.5))
    stratify = bool(div_cfg.get("stratify_by_food_group", True))
    min_per_fg = int(div_cfg.get("min_per_food_group", 12))
    explore_frac = float(div_cfg.get("random_explore_frac", 0.3))
    snack_nut_bonus = float(div_cfg.get("snack_nut_rank_bonus", 0.0))
    _desc_map = _food_desc_cache()
    from algo.bronze_food_filter import is_snack_preferred

    nak_cfg = dict((cfg.get("pool") or {}).get("low_na_high_k") or {})
    use_nak = bool(nak_cfg.get("enabled", True)) and target_needs_low_na_high_k(target_dict)
    if use_nak and "explore_frac" in nak_cfg:
        explore_frac = float(nak_cfg.get("explore_frac", explore_frac))
    pool_seed = _pool_rng_seed(target_dict, pool_seed_offset=pool_seed_offset)
    # mutable counter so nested _diverse_take gets distinct but reproducible states
    _sample_salt = [0]

    def _next_sample_state() -> int:
        _sample_salt[0] += 1
        return int(pool_seed + _sample_salt[0] * 1009)
    nak_mask = (
        low_na_high_k_mask(
            df,
            sodium_per_100g_max=float(nak_cfg.get("sodium_per_100g_max", 140)),
            potassium_per_100g_min=float(nak_cfg.get("potassium_per_100g_min", 200)),
            grams_min=float(nak_cfg.get("grams_min", 20)),
            potassium_abs_min=float(nak_cfg.get("potassium_abs_min", 150)),
        )
        if use_nak
        else pd.Series(False, index=df.index)
    )
    nak_bonus = float(nak_cfg.get("rank_bonus", 25.0)) if use_nak else 0.0
    nak_force = int(nak_cfg.get("per_meal_force", 40)) if use_nak else 0
    if use_nak:
        print(
            f"pool low-Na/high-K boost: candidates={int(nak_mask.sum())} "
            f"force/meal={nak_force} bonus={nak_bonus}",
            flush=True,
        )

    def _rank_meal(sub: pd.DataFrame, meal: str = "") -> pd.DataFrame:
        """Capability score: macros/micros density minus sodium/sugar pressure."""
        s = sub.copy()
        gg = pd.to_numeric(s["grams"], errors="coerce").replace(0, np.nan)
        dens = lambda col: pd.to_numeric(s.get(col, 0), errors="coerce").fillna(0.0) / gg * 100.0
        # Soft meal ranking: do NOT density-boost fiber (bran cereal monopoly).
        # Coverage of lo>0 nutrients is handled by _positive_nutrient_coverage.
        prio = (
            dens("protein").fillna(0.0)
            + 0.5 * dens("carb").fillna(0.0)
            + 5.0 * dens("vitamin_d").fillna(0.0)
            + 0.05 * dens("folic_acid").fillna(0.0)
            + 0.05 * dens("vitamin_c").fillna(0.0)
            - 0.01 * dens("sodium").fillna(0.0)
            - 0.5 * dens("sugar").fillna(0.0)
            - 0.2 * dens("saturated_fat").fillna(0.0)
            + common_bonus * s["is_common"].astype(float)
            + s["food_id"].isin(hist).astype(float)
        )
        if meal == "snack" and snack_nut_bonus and _desc_map:
            prio = prio + s["food_id"].map(
                lambda x: snack_nut_bonus
                if pd.notna(x) and is_snack_preferred(_desc_map.get(int(float(x)), ""))
                else 0.0
            )
        if use_nak and nak_bonus:
            # align mask to sub index
            prio = prio + nak_mask.reindex(s.index).fillna(False).astype(float) * nak_bonus
        s["_prio"] = prio
        return s.sort_values("_prio", ascending=False)

    def _diverse_take(ranked: pd.DataFrame, quota: int) -> pd.DataFrame:
        """Round-robin by food_group + random explore so pool is not 300 cheat foods."""
        if ranked.empty or quota <= 0:
            return ranked.iloc[0:0]
        if not stratify or "food_group_code" not in ranked.columns:
            n_cap = max(1, int(quota * (1.0 - explore_frac)))
            top = ranked.head(n_cap)
            rest = ranked.drop(top.index, errors="ignore")
            n_rand = min(len(rest), quota - len(top))
            if n_rand > 0:
                rnd = rest.sample(n=n_rand, random_state=_next_sample_state())
                return pd.concat([top, rnd]).drop_duplicates()
            return top

        # guarantee min_per_fg from each group present, then fill by global rank + explore
        picked_idx: list = []
        seen = set()
        for _, g in ranked.groupby("food_group_code", sort=False):
            take = g.head(min_per_fg)
            for i in take.index:
                if i not in seen:
                    picked_idx.append(i)
                    seen.add(i)
            if len(picked_idx) >= quota:
                break
        # fill remaining from global rank
        for i in ranked.index:
            if i in seen:
                continue
            picked_idx.append(i)
            seen.add(i)
            if len(picked_idx) >= int(quota * (1.0 - explore_frac)):
                break
        top = ranked.loc[picked_idx[:quota]]
        if len(top) >= quota:
            return top.head(quota)
        rest = ranked.drop(top.index, errors="ignore")
        n_rand = min(len(rest), quota - len(top))
        if n_rand > 0:
            rnd = rest.sample(n=n_rand, random_state=_next_sample_state())
            return pd.concat([top, rnd]).drop_duplicates().head(quota)
        return top

    kept_parts = []
    for meal, quota in meal_quota.items():
        sub = df[df["meal"] == meal]
        if sub.empty:
            continue
        ranked = _rank_meal(sub, meal=meal)
        force_part = ranked.iloc[0:0]
        if use_nak and nak_force > 0:
            force_part = ranked.loc[nak_mask.reindex(ranked.index).fillna(False)].head(nak_force)
        if prefer:
            preferred = ranked[
                ranked["food_id"].isin(common) | ranked["food_id"].isin(hist)
            ]
            n_pref = min(len(preferred), int(quota * 0.15)) if len(preferred) else 0
            pref_part = preferred.head(n_pref) if n_pref else ranked.iloc[0:0]
            others = ranked.drop(pref_part.index, errors="ignore")
            if len(force_part):
                others = others.drop(force_part.index, errors="ignore")
            n_other = max(0, quota - len(pref_part) - len(force_part))
            part = pd.concat(
                [force_part, _diverse_take(others, n_other), pref_part], ignore_index=False
            ).drop_duplicates()
        else:
            others = ranked.drop(force_part.index, errors="ignore") if len(force_part) else ranked
            part = pd.concat(
                [force_part, _diverse_take(others, max(0, quota - len(force_part)))],
                ignore_index=False,
            ).drop_duplicates()
        kept_parts.append(part)

    coverage = _positive_nutrient_coverage(
        df,
        target_dict,
        k=boost_k,
        rng=np.random.default_rng(pool_seed + 17),
    )
    if not coverage.empty:
        kept_parts.append(coverage)

    if not kept_parts:
        out = (
            df.sample(n=min(max_pool, len(df)), random_state=pool_seed) if len(df) else df
        )
    else:
        out = pd.concat(kept_parts, ignore_index=False).drop_duplicates()

    # Stratified trim (never drop a meal entirely; keep forced low-Na/high-K)
    if len(out) > max_pool:
        trimmed = []
        for meal, quota in meal_quota.items():
            sub = out[out["meal"] == meal]
            if sub.empty:
                continue
            ranked = _rank_meal(sub, meal=meal)
            force_part = ranked.iloc[0:0]
            if use_nak and nak_force > 0:
                force_part = ranked.loc[nak_mask.reindex(ranked.index).fillna(False)].head(nak_force)
            others = ranked.drop(force_part.index, errors="ignore") if len(force_part) else ranked
            trimmed.append(
                pd.concat(
                    [force_part, _diverse_take(others, max(0, quota - len(force_part)))],
                    ignore_index=False,
                ).drop_duplicates()
            )
        if not coverage.empty:
            trimmed.append(coverage)
        out = pd.concat(trimmed, ignore_index=False).drop_duplicates()
        if len(out) > max_pool:
            out = out.sample(n=max_pool, random_state=_next_sample_state())

    return out.reset_index(drop=True)


def check_bld_hard(metrics: dict, config: Optional[dict] = None) -> tuple[bool, float]:
    """Return (ok, penalty) for breakfast/lunch/dinner minimum grams."""
    cfg = config or load_ga_config()
    mins = cfg["min_meal_weight_g"]
    meal_g = metrics["meal_g"]
    penalty = 0.0
    for m in ("breakfast", "lunch", "dinner"):
        need = float(mins.get(m, 0))
        have = float(meal_g.get(m, 0.0))
        if have < need:
            penalty += need - have
    return penalty <= 0, penalty


def condiment_oil_meal_shares(
    day: pd.DataFrame,
    codes: Optional[list[int]] = None,
) -> dict[str, float]:
    """Per major meal: weight share from condiment/oil/beverage food groups."""
    codes = [int(x) for x in (codes or [8, 9])]
    grams = pd.to_numeric(day.get("grams", 0), errors="coerce").fillna(0.0)
    if "food_group_code" in day.columns:
        gcode = pd.to_numeric(day["food_group_code"], errors="coerce").fillna(0).astype(int)
    else:
        gcode = day["food_id"].map(food_group_code).fillna(0).astype(int)
    if "meal" in day.columns:
        meal = day["meal"].astype(str).str.lower()
    else:
        meal = day["meal_id"].map(ID_TO_MEAL).fillna("snack")
    shares: dict[str, float] = {}
    for m in ("breakfast", "lunch", "dinner"):
        mask = meal == m
        tot = float(grams[mask].sum())
        if tot <= 0:
            shares[m] = 0.0
            continue
        bad = float(grams[mask & gcode.isin(codes)].sum())
        shares[m] = bad / tot
    return shares


def check_condiment_oil_hard(
    day: pd.DataFrame,
    config: Optional[dict] = None,
) -> tuple[bool, list[str]]:
    """
    Hard operability: too many major meals dominated by oil/condiment/sweet-beverage
    weight (Silver-equivalent rule, now enforced in Bronze GA).
    """
    cfg = config or load_ga_config()
    hard = cfg.get("hard_operability") or {}
    if not hard:
        return True, []
    codes = [int(x) for x in hard.get("condiment_oil_food_group_codes", [8, 9])]
    ch = float(hard.get("condiment_oil_weight_share_hard", 0.85))
    bh = float(hard.get("beverage_weight_share_hard", 0.95))
    min_k = float(hard.get("condiment_min_meal_kcal", 200))
    need_n = int(hard.get("condiment_fail_n_meals", 2))

    shares = condiment_oil_meal_shares(day, codes)
    # meal kcal
    kcal = pd.to_numeric(day.get("calorie", 0), errors="coerce").fillna(0.0)
    if "meal" in day.columns:
        meal = day["meal"].astype(str).str.lower()
    else:
        meal = day["meal_id"].map(ID_TO_MEAL).fillna("snack")
    meal_kcal = {m: float(kcal[meal == m].sum()) for m in ("breakfast", "lunch", "dinner")}

    fails: list[str] = []
    bad_meals = []
    for m, sh in shares.items():
        if sh >= ch and meal_kcal.get(m, 0.0) >= min_k:
            bad_meals.append(f"{m}:{sh:.2f}/{meal_kcal.get(m, 0):.0f}kcal")
    if len(bad_meals) >= need_n:
        fails.append("condiment_dominated_meals=" + ",".join(bad_meals))
    for m, sh in shares.items():
        if sh >= bh and meal_kcal.get(m, 0.0) >= min_k:
            fails.append(f"beverage_meal:{m}={sh:.2f}")
            break
    return len(fails) == 0, fails
