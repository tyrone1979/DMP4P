"""
Day-level diet structure features for learning operable plate priors.

Used by structure_learn + (later) Bronze GA Layer-A/B.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from algo.diet_structure_priors import food_group_code, meal_from_eating_type
from algo.fndds_code_policy import is_main_cuisine_by_code, load_fndds_code_policy, match_prefix, unit_kind

# Industrial / non-plate families (subset of sequential_ban) — presence is a feature.
_NONPLATE_PREFIXES = (
    "95",
    "118",
    "929",
    "92191",
    "92291",
    "91745",
)
# Pure sugar / icing (FNDDS sweets)
_SUGAR_ICING_PREFIXES = ("91101000", "91305", "91301")
# Starch stack families
_BREAD_PREFIXES = ("51",)  # breads/rolls (includes French); tortillas separate
_TORTILLA_PREFIXES = ("52215",)
_CEREAL_PREFIXES = ("57",)
_PLAIN_PASTA_PREFIXES = ("561",)  # cooked macaroni/noodles without sauce plate codes
_CHIP_PREFIXES = ("712", "544")
_CRACKER_PREFIXES = ("54",)


# FNDDS one-digit majors 1–9 (dairy…sweets/bev) — category-gram structure prior.
MAJOR_CODES = tuple(range(1, 10))
MEAL_NAMES = ("breakfast", "lunch", "dinner", "snack")
MAJOR_GRAM_FEATURES = [f"g_major_{i}" for i in MAJOR_CODES]
# Per eating_type (meal) × major — primary Hard axes for category prior.
MEAL_MAJOR_GRAM_FEATURES = [f"g_major_{i}_{m}" for m in MEAL_NAMES for i in MAJOR_CODES]

FEATURE_NAMES = [
    "total_g",
    "total_kcal",
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
    "snack_kcal_over_max_meal",
    "n_mains",
    "main_grams_total",
    "lunch_main_g",
    "dinner_main_g",
    "max_mains_in_one_meal",
    "bread_g",
    "tortilla_g",
    "cereal_g",
    "plain_pasta_g",
    "cracker_g",
    "chip_g",
    "max_slice_qty_meal",
    "max_fillet_qty_meal",
    "max_cereal_cup_day",
    "nonplate_g",
    "sugar_icing_g",
    "has_bld",
] + MAJOR_GRAM_FEATURES
# Meal×major grams are category-prior features (not IsolationForest plate axes).
# Still emitted by day_structure_feature_dict for GA Hard scoring.


def _fid(x) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def _prefix_hit(fid: int, prefixes: tuple[str, ...]) -> bool:
    return match_prefix(fid, prefixes) is not None


def _ensure_meal(day: pd.DataFrame) -> pd.Series:
    if "meal" in day.columns and day["meal"].notna().any():
        return day["meal"].astype(str).str.lower()
    if "eating_type" in day.columns:
        return day["eating_type"].map(meal_from_eating_type).fillna("snack")
    return pd.Series(["snack"] * len(day), index=day.index)


def major_gram_dict(day: pd.DataFrame) -> dict[str, float]:
    """Sum grams by FNDDS one-digit major (1–9). Missing majors → 0."""
    out = {f"g_major_{i}": 0.0 for i in MAJOR_CODES}
    if day is None or len(day) == 0:
        return out
    work = day.copy()
    work["food_id"] = work["food_id"].map(_fid)
    work["grams"] = pd.to_numeric(work.get("grams"), errors="coerce").fillna(0.0)
    work["major"] = work["food_id"].map(food_group_code)
    for maj, g in work.groupby("major")["grams"].sum().items():
        mi = int(maj) if pd.notna(maj) else 0
        if mi in MAJOR_CODES:
            out[f"g_major_{mi}"] = float(g)
    return out


def meal_major_gram_dict(day: pd.DataFrame) -> dict[str, float]:
    """Sum grams by (meal × FNDDS major 1–9). Missing cells → 0."""
    out = {f: 0.0 for f in MEAL_MAJOR_GRAM_FEATURES}
    if day is None or len(day) == 0:
        return out
    work = day.copy()
    work["food_id"] = work["food_id"].map(_fid)
    work["grams"] = pd.to_numeric(work.get("grams"), errors="coerce").fillna(0.0)
    work["meal"] = _ensure_meal(work)
    work["major"] = work["food_id"].map(food_group_code)
    for (meal, maj), g in work.groupby(["meal", "major"])["grams"].sum().items():
        m = str(meal).lower()
        mi = int(maj) if pd.notna(maj) else 0
        if m in MEAL_NAMES and mi in MAJOR_CODES:
            out[f"g_major_{mi}_{m}"] = float(g)
    return out


def day_structure_feature_dict(day: pd.DataFrame) -> dict[str, float]:
    """Compute structure features for one day of food rows."""
    if day is None or len(day) == 0:
        empty = {k: 0.0 for k in FEATURE_NAMES}
        empty.update({f: 0.0 for f in MEAL_MAJOR_GRAM_FEATURES})
        return empty

    work = day.copy()
    work["food_id"] = work["food_id"].map(_fid)
    work["grams"] = pd.to_numeric(work.get("grams"), errors="coerce").fillna(0.0)
    if "calorie" in work.columns:
        work["calorie"] = pd.to_numeric(work["calorie"], errors="coerce").fillna(0.0)
    else:
        work["calorie"] = 0.0
    work["meal"] = _ensure_meal(work)
    work["display_qty"] = pd.to_numeric(work.get("display_qty"), errors="coerce")
    work["display_unit"] = work.get("display_unit", pd.Series([""] * len(work))).astype(str).str.lower()

    total_g = float(work["grams"].sum())
    total_kcal = float(work["calorie"].sum())
    eps = 1e-9

    meals = ["breakfast", "lunch", "dinner", "snack"]
    g_meal = {m: float(work.loc[work["meal"] == m, "grams"].sum()) for m in meals}
    k_meal = {m: float(work.loc[work["meal"] == m, "calorie"].sum()) for m in meals}
    max_meal_k = max(k_meal["breakfast"], k_meal["lunch"], k_meal["dinner"])
    if max_meal_k <= 0:
        # no B/L/D energy → treat snack ratio as 0 (avoid huge /eps artifacts)
        snack_over = 0.0
    else:
        snack_over = k_meal["snack"] / max_meal_k

    mains_mask = work["food_id"].map(is_main_cuisine_by_code)
    main_rows = work[mains_mask]
    n_mains = int(main_rows["food_id"].nunique()) if len(main_rows) else 0
    main_g = float(main_rows["grams"].sum()) if len(main_rows) else 0.0
    lunch_main_g = float(main_rows.loc[main_rows["meal"] == "lunch", "grams"].sum()) if len(main_rows) else 0.0
    dinner_main_g = float(main_rows.loc[main_rows["meal"] == "dinner", "grams"].sum()) if len(main_rows) else 0.0
    max_mains_one = 0
    if len(main_rows):
        max_mains_one = int(main_rows.groupby("meal")["food_id"].nunique().max())

    def grams_pref(prefixes: tuple[str, ...]) -> float:
        return float(work.loc[work["food_id"].map(lambda f: _prefix_hit(f, prefixes)), "grams"].sum())

    # max slice qty in any single meal
    max_slice = 0.0
    max_fillet = 0.0
    for m, sub in work.groupby("meal"):
        sl = sub.loc[sub["display_unit"].str.contains("slice", na=False), "display_qty"]
        if len(sl):
            max_slice = max(max_slice, float(sl.fillna(0).sum()))
        fl = sub.loc[sub["display_unit"].str.contains("fillet", na=False), "display_qty"]
        if len(fl):
            max_fillet = max(max_fillet, float(fl.fillna(0).sum()))
    # cereal cups day total
    cereal_mask = work["food_id"].map(lambda f: _prefix_hit(f, _CEREAL_PREFIXES))
    cereal_cups = work.loc[cereal_mask & work["display_unit"].str.contains("cup", na=False), "display_qty"]
    max_cereal_cup = float(cereal_cups.fillna(0).sum()) if len(cereal_cups) else 0.0
    # if no cup label, approximate cereal_g/30
    if max_cereal_cup <= 0 and cereal_mask.any():
        max_cereal_cup = float(work.loc[cereal_mask, "grams"].sum()) / 40.0

    nonplate = grams_pref(_NONPLATE_PREFIXES)
    sugar = grams_pref(_SUGAR_ICING_PREFIXES)

    groups = work["food_id"].map(food_group_code)
    major_g = major_gram_dict(work)
    meal_major_g = meal_major_gram_dict(work)

    out = {
        "total_g": total_g,
        "total_kcal": total_kcal,
        "n_unique_foods": float(work["food_id"].nunique()),
        "n_items": float(len(work)),
        "n_food_groups": float(groups.replace(0, np.nan).nunique(dropna=True)),
        "share_g_breakfast": g_meal["breakfast"] / (total_g + eps),
        "share_g_lunch": g_meal["lunch"] / (total_g + eps),
        "share_g_dinner": g_meal["dinner"] / (total_g + eps),
        "share_g_snack": g_meal["snack"] / (total_g + eps),
        "share_kcal_breakfast": k_meal["breakfast"] / (total_kcal + eps),
        "share_kcal_lunch": k_meal["lunch"] / (total_kcal + eps),
        "share_kcal_dinner": k_meal["dinner"] / (total_kcal + eps),
        "share_kcal_snack": k_meal["snack"] / (total_kcal + eps),
        "snack_kcal_over_max_meal": float(snack_over),
        "n_mains": float(n_mains),
        "main_grams_total": main_g,
        "lunch_main_g": lunch_main_g,
        "dinner_main_g": dinner_main_g,
        "max_mains_in_one_meal": float(max_mains_one),
        "bread_g": grams_pref(_BREAD_PREFIXES),
        "tortilla_g": grams_pref(_TORTILLA_PREFIXES),
        "cereal_g": grams_pref(_CEREAL_PREFIXES),
        "plain_pasta_g": grams_pref(_PLAIN_PASTA_PREFIXES),
        "cracker_g": grams_pref(_CRACKER_PREFIXES),
        "chip_g": grams_pref(_CHIP_PREFIXES),
        "max_slice_qty_meal": max_slice,
        "max_fillet_qty_meal": max_fillet,
        "max_cereal_cup_day": max_cereal_cup,
        "nonplate_g": nonplate,
        "sugar_icing_g": sugar,
        "has_bld": float(
            (g_meal["breakfast"] > 0) and (g_meal["lunch"] > 0) and (g_meal["dinner"] > 0)
        ),
    }
    out.update(major_g)
    out.update(meal_major_g)
    return out


def features_frame_from_days(
    items: pd.DataFrame,
    *,
    day_keys: list[str],
    id_col: str = "day_id",
) -> pd.DataFrame:
    """
    Group ``items`` by ``day_keys`` and emit one feature row per day.
    Adds ``id_col`` = join of day_keys.
    """
    rows = []
    for key, sub in items.groupby(day_keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        feat = day_structure_feature_dict(sub)
        feat[id_col] = "|".join(str(x) for x in key)
        for k, v in zip(day_keys, key):
            feat[f"key_{k}"] = v
        # demographics if present
        for c in ("sex", "gender", "age", "bmi", "age_band", "bmi_class"):
            if c in sub.columns:
                feat[c] = sub[c].iloc[0]
        rows.append(feat)
    return pd.DataFrame(rows)


def bmi_class_label(bmi) -> str:
    """Match nhanes day_structure.bmi_class vocabulary."""
    try:
        b = float(bmi)
    except (TypeError, ValueError):
        return "NA"
    if b < 18.5:
        return "underweight (<18.5)"
    if b < 25:
        return "normal (18.5–24.9)"
    if b < 30:
        return "overweight (25–29.9)"
    return "obesity (≥30)"


def age_band_label(age) -> str:
    """Match nhanes day_structure.age_band vocabulary."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "NA"
    if a < 31:
        return "19–30"
    if a < 51:
        return "31–50"
    if a < 71:
        return "51–70"
    return "71+"


def sex_label_stratum(sex) -> str:
    if sex is None or (isinstance(sex, float) and pd.isna(sex)):
        return "NA"
    try:
        si = int(float(sex))
        return "Male" if si == 1 else ("Female" if si == 2 else str(sex))
    except (TypeError, ValueError):
        s = str(sex).strip()
        sl = s.lower()
        if sl in ("m", "male", "1"):
            return "Male"
        if sl in ("f", "female", "2"):
            return "Female"
        if s in ("Male", "Female"):
            return s
        return s or "NA"


def stratum_key(row: pd.Series) -> str:
    """Person-stratum key from active ``StructureSchema`` (default sex|bmi|age)."""
    from algo.structure_schema import stratum_key_generic

    return stratum_key_generic(row)
