"""
Multi-label meal tags (breakfast / lunch / dinner / snack) for food_ids.

Used by Bronze so each occasion draws only from foods tagged for that meal.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "config" / "food_meal_tags.yaml"
MEALS = ("breakfast", "lunch", "dinner", "snack")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


@lru_cache(maxsize=2)
def load_meal_tag_config(path: str = "") -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CFG
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _hit(desc: str, keywords: list[str]) -> bool:
    d = _norm(desc)
    for k in keywords:
        kk = _norm(k)
        if kk and kk in d:
            return True
    return False


def empirical_meal_tags(
    food_user: pd.DataFrame,
    cfg: Optional[dict] = None,
) -> dict[int, set[str]]:
    """Tag from NHANES occasion frequencies."""
    cfg = (cfg or load_meal_tag_config()).get("empirical") or {}
    if not cfg.get("enabled", True):
        return {}
    df = food_user.copy()
    if "meal" not in df.columns:
        from algo.diet_structure_priors import meal_from_eating_type

        df["meal"] = df["eating_type"].map(meal_from_eating_type)
    df["meal"] = df["meal"].astype(str).str.lower()
    df = df[df["meal"].isin(MEALS)]
    df["food_id"] = pd.to_numeric(df["food_id"], errors="coerce")
    df = df.dropna(subset=["food_id"])
    df["food_id"] = df["food_id"].astype(int)

    min_count = int(cfg.get("min_count", 3))
    min_share = float(cfg.get("min_share", 0.08))
    always_mode = bool(cfg.get("always_tag_mode", True))

    out: dict[int, set[str]] = {}
    for fid, g in df.groupby("food_id"):
        vc = g["meal"].value_counts()
        total = int(vc.sum())
        tags: set[str] = set()
        for meal, cnt in vc.items():
            if int(cnt) >= min_count and (int(cnt) / total) >= min_share:
                tags.add(str(meal))
        if always_mode and not tags and len(vc):
            tags.add(str(vc.index[0]))
        if tags:
            out[int(fid)] = tags
    return out


def apply_keyword_rules(
    tags: dict[int, set[str]],
    food_desc: dict[int, str],
    cfg: Optional[dict] = None,
) -> dict[int, set[str]]:
    cfg = cfg or load_meal_tag_config()
    rules = list(cfg.get("rules") or [])
    out = {fid: set(v) for fid, v in tags.items()}
    # ensure all described foods exist
    for fid in food_desc:
        out.setdefault(int(fid), set())

    locked: set[int] = set()
    # Pass 1: force_add / force_remove
    for rule in rules:
        if rule.get("force_only") is not None:
            continue
        kws = list(rule.get("keywords") or [])
        force_add = rule.get("force_add")
        force_remove = rule.get("force_remove")
        for fid, name in food_desc.items():
            if not _hit(str(name), kws):
                continue
            fid = int(fid)
            if fid in locked:
                continue
            cur = out.setdefault(fid, set())
            if force_add:
                cur |= set(str(x) for x in force_add)
            if force_remove:
                cur -= set(str(x) for x in force_remove)
            out[fid] = cur

    # Pass 2: force_only locks final tags (wins over force_add)
    for rule in rules:
        force_only = rule.get("force_only")
        if force_only is None:
            continue
        kws = list(rule.get("keywords") or [])
        for fid, name in food_desc.items():
            if not _hit(str(name), kws):
                continue
            fid = int(fid)
            out[fid] = set(str(x) for x in force_only)
            locked.add(fid)
    return out


def finalize_tags(
    tags: dict[int, set[str]],
    cfg: Optional[dict] = None,
) -> dict[int, set[str]]:
    cfg = cfg or load_meal_tag_config()
    fallback = set(str(x) for x in (cfg.get("fallback_tags") or list(MEALS)))
    out = {}
    for fid, t in tags.items():
        tt = {x for x in t if x in MEALS}
        out[int(fid)] = tt if tt else set(fallback)
    return out


def build_food_meal_tags(
    food_user: pd.DataFrame,
    food_desc: dict[int, str],
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """Return dataframe food_id, food_desc, tags (pipe-separated), plus boolean columns."""
    cfg = cfg or load_meal_tag_config()
    tags = empirical_meal_tags(food_user, cfg)
    tags = apply_keyword_rules(tags, food_desc, cfg)
    tags = finalize_tags(tags, cfg)

    rows = []
    for fid, name in food_desc.items():
        fid = int(fid)
        t = tags.get(fid) or set(cfg.get("fallback_tags") or MEALS)
        t = {x for x in t if x in MEALS}
        row = {
            "food_id": fid,
            "food_desc": name,
            "meal_tags": "|".join(m for m in MEALS if m in t),
            "n_tags": len(t),
        }
        for m in MEALS:
            row[f"tag_{m}"] = int(m in t)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("food_id").reset_index(drop=True)


def tags_dict_from_frame(df: pd.DataFrame) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for r in df.itertuples(index=False):
        tags = set(str(r.meal_tags).split("|")) if getattr(r, "meal_tags", None) else set()
        tags = {t for t in tags if t in MEALS}
        out[int(r.food_id)] = tags
    return out


def load_food_meal_tags(path: Optional[Path] = None) -> dict[int, set[str]]:
    cfg = load_meal_tag_config()
    p = Path(path) if path else ROOT / str((cfg.get("paths") or {}).get("out_csv", "data/nhanes/data/food_meal_tags.csv"))
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return tags_dict_from_frame(df)


def filter_rows_by_meal_tags(
    df: pd.DataFrame,
    tags: dict[int, set[str]],
    meal_col: str = "meal",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep rows whose meal is in the food's multi-label tag set."""
    if df.empty or not tags:
        return df.copy(), {"n_in": len(df), "n_out": len(df), "n_dropped_meal_tag": 0}

    out = df.copy()
    if meal_col not in out.columns:
        if "eating_type" in out.columns:
            from algo.diet_structure_priors import meal_from_eating_type

            out[meal_col] = out["eating_type"].map(meal_from_eating_type)
        elif "meal_id" in out.columns:
            from algo.diet_structure_priors import ID_TO_MEAL

            out[meal_col] = out["meal_id"].map(ID_TO_MEAL)

    n_in = len(out)
    fids = pd.to_numeric(out["food_id"], errors="coerce")
    meals = out[meal_col].astype(str).str.lower()

    def ok(fid, meal) -> bool:
        if pd.isna(fid) or meal not in MEALS:
            return False
        t = tags.get(int(fid))
        if not t:
            return True  # unknown food: keep
        return meal in t

    mask = [ok(f, m) for f, m in zip(fids, meals)]
    kept = out.loc[mask].copy()
    return kept.reset_index(drop=True), {
        "n_in": n_in,
        "n_out": len(kept),
        "n_dropped_meal_tag": n_in - len(kept),
    }


def expand_rows_to_tagged_meals(
    df: pd.DataFrame,
    tags: dict[int, set[str]],
    meal_col: str = "meal",
) -> pd.DataFrame:
    """
    For each distinct (food_id, grams) nutrient row, emit copies for every tagged meal.
    Enables multi-tag foods to appear in multiple occasion pools.
    """
    if df.empty or not tags:
        return df.copy()
    from algo.diet_structure_priors import MEAL_TO_ID

    eating = {"breakfast": 1, "lunch": 2, "dinner": 3, "snack": 6}
    src = df.drop_duplicates(subset=["food_id", "grams"], keep="first").copy()
    fid = pd.to_numeric(src["food_id"], errors="coerce")

    def _meals_for(x) -> list[str]:
        if pd.isna(x):
            return list(MEALS)
        t = tags.get(int(x))
        if not t:
            return list(MEALS)
        return [m for m in MEALS if m in t]

    src["_tag_meal"] = fid.map(_meals_for)
    out = src.explode("_tag_meal", ignore_index=True)
    out[meal_col] = out["_tag_meal"].astype(str)
    out["meal_id"] = out[meal_col].map(MEAL_TO_ID).fillna(0).astype(int)
    out["eating_type"] = out[meal_col].map(eating).fillna(6).astype(int)
    return out.drop(columns=["_tag_meal"])
