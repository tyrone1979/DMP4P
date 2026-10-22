"""Same-meal food hygiene: at most one row per food_id per eating occasion.

NOTE: ``merge_same_foods_in_meal`` (summing duplicate portions) is a **mistaken**
legacy behavior. Bronze must use ``keep_one_food_per_eating_type`` instead.
"""
from __future__ import annotations

import warnings

import pandas as pd

NUTRIENTS = [
    "calorie",
    "protein",
    "carb",
    "sugar",
    "fiber",
    "saturated_fat",
    "cholesterol",
    "folic_acid",
    "vitamin_b12",
    "vitamin_c",
    "vitamin_d",
    "calcium",
    "phosphorus",
    "potassium",
    "iron",
    "sodium",
]

# Stale after gram merge — callers should re-run operable units if needed.
_DROP_AFTER_MERGE = (
    "display_qty",
    "display_unit",
    "grams_snapped",
    "grams_per_unit",
    "portion",
)


def _group_keys(df: pd.DataFrame) -> tuple[list[str], bool]:
    from algo.diet_structure_priors import meal_from_eating_type

    if "meal" not in df.columns:
        if "eating_type" in df.columns:
            df["meal"] = df["eating_type"].map(meal_from_eating_type)
        else:
            df["meal"] = "snack"
    df["meal"] = df["meal"].astype(str).str.lower().fillna("snack")
    df["food_id"] = pd.to_numeric(df["food_id"], errors="coerce").astype("Int64")
    if "eating_type" in df.columns:
        df["eating_type"] = pd.to_numeric(df["eating_type"], errors="coerce")
        return ["eating_type", "food_id"], True
    return ["meal", "food_id"], False


def keep_one_food_per_eating_type(day: pd.DataFrame) -> pd.DataFrame:
    """
    Keep at most one row per (eating_type|meal, food_id).

    Does **not** sum grams. Prefers the row with largest ``grams`` (operable),
    then first. Drops legacy ``grams_snapped`` / ``_merged_n``.
    """
    if day is None or day.empty:
        return day
    df = day.copy()
    group_keys, _ = _group_keys(df)
    # largest operable grams first
    if "grams" in df.columns:
        df = df.assign(_g=pd.to_numeric(df["grams"], errors="coerce").fillna(0.0))
        df = df.sort_values("_g", ascending=False)
    out = df.drop_duplicates(subset=group_keys, keep="first")
    # Keep ``_merged_n`` so retrofit can undo mistaken sum-merges; callers drop it.
    out = out.drop(columns=[c for c in ("_g", "grams_snapped") if c in out.columns])
    return out.reset_index(drop=True)


def keep_one_food_per_day(day: pd.DataFrame) -> pd.DataFrame:
    """
    Keep at most one row per food_id in the whole day (across all meals).

    Prefers largest ``grams``. Does not sum portions across meals.
    """
    if day is None or day.empty:
        return day
    df = day.copy()
    df["food_id"] = pd.to_numeric(df["food_id"], errors="coerce").astype("Int64")
    if "grams" in df.columns:
        df = df.assign(_g=pd.to_numeric(df["grams"], errors="coerce").fillna(0.0))
        df = df.sort_values("_g", ascending=False)
    out = df.drop_duplicates(subset=["food_id"], keep="first")
    out = out.drop(columns=[c for c in ("_g", "grams_snapped") if c in out.columns])
    return out.reset_index(drop=True)


_MERGE_WARNED = False


def merge_same_foods_in_meal(day: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED / mistaken: summed duplicate food_id rows in one meal.

    Bronze must not call this. Use ``keep_one_food_per_eating_type``.
    Kept only for older Silver/Golden publish scripts.
    """
    global _MERGE_WARNED
    if not _MERGE_WARNED:
        warnings.warn(
            "merge_same_foods_in_meal is deprecated (mistaken sum-of-duplicates); "
            "use keep_one_food_per_eating_type",
            DeprecationWarning,
            stacklevel=2,
        )
        _MERGE_WARNED = True
    if day is None or day.empty:
        return day
    df = day.copy()
    from algo.diet_structure_priors import meal_from_eating_type

    group_keys, use_et = _group_keys(df)

    sum_cols = [c for c in (["grams", "grams_raw"] + NUTRIENTS) if c in df.columns]
    skip_meta = set(sum_cols) | {
        "meal",
        "eating_type",
        "food_id",
        "food_id_original",
        "replace_method",
        "golden_flag",
        "replace_distance",
        "food_name",
        "_merged_n",
        *_DROP_AFTER_MERGE,
    }
    meta_first = [c for c in df.columns if c not in skip_meta]

    parts: list[pd.DataFrame] = []
    for key, g in df.groupby(group_keys, sort=False, dropna=False):
        if use_et:
            et, fid = key
            meal = None
        else:
            meal, fid = key
            et = None
        if pd.isna(fid):
            parts.append(g.drop(columns=[c for c in _DROP_AFTER_MERGE if c in g.columns], errors="ignore"))
            continue
        row = g.iloc[0].copy()
        for c in _DROP_AFTER_MERGE:
            if c in row.index:
                row = row.drop(labels=[c])
        for c in sum_cols:
            row[c] = pd.to_numeric(g[c], errors="coerce").fillna(0).sum()
        for c in meta_first:
            if c in g.columns and c not in _DROP_AFTER_MERGE:
                row[c] = g.iloc[0][c]
        if "food_id_original" in g.columns:
            origs = []
            for x in g["food_id_original"]:
                try:
                    if pd.notna(x) and int(x) != int(fid):
                        origs.append(str(int(x)))
                except (TypeError, ValueError):
                    pass
            if origs:
                row["food_id_original"] = int(origs[0])
            else:
                row["food_id_original"] = int(fid)
        if "replace_method" in g.columns:
            methods = sorted({str(x) for x in g["replace_method"].fillna("") if str(x)})
            row["replace_method"] = "|".join(methods)
        if "golden_flag" in g.columns:
            flags = sorted({str(x) for x in g["golden_flag"].fillna("") if str(x)})
            row["golden_flag"] = "|".join(flags)
        row["food_id"] = int(fid)
        if use_et and pd.notna(et):
            row["eating_type"] = int(float(et))
            row["meal"] = meal_from_eating_type(et) or str(g.iloc[0].get("meal") or "snack")
        else:
            row["meal"] = meal if not use_et else str(g.iloc[0].get("meal") or "snack")
        row["_merged_n"] = int(len(g))
        parts.append(pd.DataFrame([row]))

    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
    drop_cols = [c for c in _DROP_AFTER_MERGE if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def merge_same_foods_by_day(items: pd.DataFrame) -> pd.DataFrame:
    """Apply merge_same_foods_in_meal for each (user_id, day) or fingerprint group."""
    if items is None or items.empty:
        return items
    df = items
    if {"user_id", "day"} <= set(df.columns):
        keys = ["user_id", "day"]
    elif "fingerprint_hash" in df.columns:
        keys = ["fingerprint_hash"]
    elif "week_day_index" in df.columns:
        keys = ["week_day_index"]
    else:
        return merge_same_foods_in_meal(df)

    parts = []
    for _, g in df.groupby(keys, sort=False):
        parts.append(merge_same_foods_in_meal(g))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
