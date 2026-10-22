"""
Bronze food-pool filter: keep adult plate-like foods with FPED coverage.

Drops infant/human-milk/formula, gums/sweeteners, meal-replacements /
protein powders, pure oils, and other non-meal supplements before GA.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "config" / "bronze_food_exclusions.yaml"


@lru_cache(maxsize=2)
def load_bronze_exclusion_config(path: str = "") -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CFG
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def _allowed(desc_n: str, allow_kw: list[str]) -> bool:
    return any(a and a in desc_n for a in allow_kw)


@lru_cache(maxsize=8)
def _kw_regex(keywords: tuple[str, ...]) -> Optional[re.Pattern]:
    parts = [_norm(k) for k in keywords if k]
    parts = [re.escape(p) for p in parts if p]
    if not parts:
        return None
    return re.compile("|".join(parts))


def _desc_series(fids: pd.Series, food_desc: dict[int, str]) -> pd.Series:
    u = pd.to_numeric(fids, errors="coerce")
    uniq = u.dropna().astype(int).unique()
    mapped = {int(i): _norm(food_desc.get(int(i), "")) for i in uniq}
    return u.map(mapped).fillna("")


def is_breakfast_only(desc: str, cfg: Optional[dict] = None) -> bool:
    cfg = cfg or load_bronze_exclusion_config()
    d = _norm(desc)
    if not d:
        return False
    allow = [_norm(x) for x in (cfg.get("allow_keywords") or [])]
    if _allowed(d, allow):
        return False
    rx = _kw_regex(tuple(cfg.get("breakfast_only_keywords") or []))
    return bool(rx.search(d)) if rx else False


def is_breakfast_snack_only(desc: str, cfg: Optional[dict] = None) -> bool:
    cfg = cfg or load_bronze_exclusion_config()
    d = _norm(desc)
    if not d:
        return False
    allow = [_norm(x) for x in (cfg.get("allow_keywords") or [])]
    if _allowed(d, allow):
        return False
    if is_breakfast_only(desc, cfg):
        return False
    if is_snack_only(desc, cfg):
        return False
    rx = _kw_regex(tuple(cfg.get("breakfast_snack_only_keywords") or []))
    return bool(rx.search(d)) if rx else False


def is_snack_only(desc: str, cfg: Optional[dict] = None) -> bool:
    """Salty chips etc.: only allowed at snack (not B/L/D)."""
    cfg = cfg or load_bronze_exclusion_config()
    d = _norm(desc)
    if not d:
        return False
    allow = [_norm(x) for x in (cfg.get("allow_keywords") or [])]
    if _allowed(d, allow):
        return False
    rx = _kw_regex(tuple(cfg.get("snack_only_keywords") or []))
    return bool(rx.search(d)) if rx else False


def is_no_snack(desc: str, cfg: Optional[dict] = None) -> bool:
    """Loaf bread etc.: not allowed at snack."""
    cfg = cfg or load_bronze_exclusion_config()
    d = _norm(desc)
    if not d:
        return False
    allow = [_norm(x) for x in (cfg.get("allow_keywords") or [])]
    if _allowed(d, allow):
        return False
    rx = _kw_regex(tuple(cfg.get("no_snack_keywords") or []))
    return bool(rx.search(d)) if rx else False


def is_snack_preferred(desc: str, cfg: Optional[dict] = None) -> bool:
    cfg = cfg or load_bronze_exclusion_config()
    d = _norm(desc)
    rx = _kw_regex(tuple(cfg.get("snack_prefer_keywords") or []))
    return bool(rx.search(d)) if rx else False


def is_excluded_food_id(food_id: int, cfg: Optional[dict] = None) -> bool:
    """Hard drops by FNDDS code prefix (e.g. 117* = infant formula)."""
    cfg = cfg or load_bronze_exclusion_config()
    try:
        s = str(int(food_id)).zfill(8)
    except (TypeError, ValueError):
        return False
    for pref in cfg.get("exclude_food_id_prefixes") or []:
        p = str(pref or "").strip()
        if p and s.startswith(p):
            return True
    return False


def is_excluded_desc(desc: str, cfg: Optional[dict] = None) -> bool:
    cfg = cfg or load_bronze_exclusion_config()
    d = _norm(desc)
    if not d:
        return False
    allow = [_norm(x) for x in (cfg.get("allow_keywords") or [])]
    if _allowed(d, allow):
        return False
    rx = _kw_regex(tuple(cfg.get("exclude_keywords") or []))
    if rx and rx.search(d):
        return True
    for pat in cfg.get("exclude_regex") or []:
        if pat and re.search(pat, d, flags=re.I):
            return True
    return False


def is_sequential_ban_food_id(food_id: int) -> bool:
    """True if food_id participates in N+1 ban (exact major or family) — not 主菜 label."""
    from algo.fndds_code_policy import is_sequential_ban_food

    return is_sequential_ban_food(food_id)


def collect_sequential_ban_food_ids(
    day: pd.DataFrame,
    *,
    food_desc: Optional[dict[int, str]] = None,  # kept for call-site compat; unused
    pool_food_ids: Optional[Iterable[int]] = None,
    cfg: Optional[dict] = None,  # unused; policy loaded from fndds_food_code_policy.yaml
    include_mains: bool = True,
    expand_keyword_family: Optional[bool] = None,  # → expand_prefix_family (None=config)
) -> set[int]:
    """
    Food ids to remove before searching the next Bronze solution.
    Default: only food_ids on the day (no prefix-family wipe).
    """
    from algo.fndds_code_policy import collect_sequential_ban_food_ids as _collect

    return _collect(
        day,
        pool_food_ids=pool_food_ids,
        include_mains=include_mains,
        expand_prefix_family=expand_keyword_family,
    )


def is_excluded_food(food_id: int, desc: str = "", cfg: Optional[dict] = None) -> bool:
    cfg = cfg or load_bronze_exclusion_config()
    if is_excluded_food_id(food_id, cfg):
        return True
    return is_excluded_desc(desc, cfg)


# Canonical NHANES English eating_type codes for rewritten occasions
MEAL_TO_EATING_TYPE = {
    "breakfast": 1,
    "lunch": 2,
    "dinner": 3,
    "snack": 6,
}


def reassign_meal_occasions(
    df: pd.DataFrame,
    food_desc: Optional[dict[int, str]] = None,
    cfg: Optional[dict] = None,
    desc_col: str = "food_desc",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clarify eating type / meal for plate hygiene (reassign, do not drop):

    - snack_only (chips, pretzels, …) → snack
    - breakfast_only (oatmeal, …) → breakfast
    - breakfast_snack_only at lunch/dinner (RTE cereal, crackers, …) → breakfast
    - no_snack loaf bread at snack → breakfast
    """
    cfg = cfg or load_bronze_exclusion_config()
    if df is None or df.empty:
        return df, {"n_in": 0, "n_reassigned": 0}

    out = df.copy()
    if food_desc is None:
        if desc_col in out.columns:
            food_desc = {
                int(i): str(n)
                for i, n in zip(
                    pd.to_numeric(out["food_id"], errors="coerce").dropna().astype(int),
                    out[desc_col].astype(str),
                )
            }
        else:
            fc = pd.read_csv(ROOT / "data" / "nhanes" / "food_code.csv", usecols=["food_id", "food_desc"])
            food_desc = dict(zip(fc.food_id.astype(int), fc.food_desc.astype(str)))

    from algo.diet_structure_priors import MEAL_TO_ID, meal_from_eating_type

    if "meal" not in out.columns:
        if "eating_type" in out.columns:
            out["meal"] = out["eating_type"].map(meal_from_eating_type)
        else:
            out["meal"] = "snack"
    # Normalize: map() None / "none" / "nan" → snack (never leave literal "none")
    out["meal"] = out["meal"].map(
        lambda m: str(m).strip().lower()
        if m is not None and str(m).strip().lower() not in {"", "none", "nan", "nat"}
        else None
    )
    if "eating_type" in out.columns:
        inferred = out["eating_type"].map(meal_from_eating_type)
        out["meal"] = out["meal"].fillna(inferred)
    out["meal"] = out["meal"].fillna("snack").astype(str).str.lower()
    out["_meal_before"] = out["meal"]

    names = _desc_series(out["food_id"], food_desc)
    allow_rx = _kw_regex(tuple(_norm(x) for x in (cfg.get("allow_keywords") or [])))
    allowed = names.str.contains(allow_rx, na=False) if allow_rx else pd.Series(False, index=out.index)

    snack_only = _mask_kw(names, list(cfg.get("snack_only_keywords") or [])) & ~allowed
    breakfast_only = _mask_kw(names, list(cfg.get("breakfast_only_keywords") or [])) & ~allowed
    no_snack = _mask_kw(names, list(cfg.get("no_snack_keywords") or [])) & ~allowed
    bs_only = (
        _mask_kw(names, list(cfg.get("breakfast_snack_only_keywords") or []))
        & ~allowed
        & ~snack_only
        & ~breakfast_only
    )

    meals = out["meal"]
    new_meal = meals.copy()
    # Priority: snack_only > breakfast_only > no_snack@snack > cereal off B/S
    new_meal = new_meal.mask(snack_only & ~meals.eq("snack"), "snack")
    new_meal = new_meal.mask(breakfast_only & ~new_meal.eq("breakfast"), "breakfast")
    new_meal = new_meal.mask(no_snack & new_meal.eq("snack"), "breakfast")
    new_meal = new_meal.mask(
        bs_only & new_meal.isin(["lunch", "dinner"]), "breakfast"
    )

    changed = new_meal != out["_meal_before"]
    out["meal"] = new_meal
    if "meal_id" in out.columns:
        out.loc[changed, "meal_id"] = out.loc[changed, "meal"].map(
            lambda m: int(MEAL_TO_ID.get(m, 4))
        )
    if "eating_type" in out.columns:
        out.loc[changed, "eating_type"] = out.loc[changed, "meal"].map(
            lambda m: int(MEAL_TO_EATING_TYPE.get(m, 6))
        )

    reasons = {
        "to_snack_chips_etc": int((changed & snack_only & new_meal.eq("snack")).sum()),
        "to_breakfast_oats_etc": int((changed & breakfast_only & new_meal.eq("breakfast")).sum()),
        "bread_off_snack": int((changed & no_snack & new_meal.eq("breakfast")).sum()),
        "cereal_off_lunch_dinner": int(
            (changed & bs_only & new_meal.eq("breakfast")).sum()
        ),
    }
    audit = {
        "n_in": int(len(out)),
        "n_reassigned": int(changed.sum()),
        "by_reason": reasons,
        "version": cfg.get("version"),
    }
    out = out.drop(columns=["_meal_before"])
    return out, audit


@lru_cache(maxsize=1)
def _fped_ok_set() -> frozenset[int]:
    from algo.fped_coded import fped_ok_ids

    return fped_ok_ids()


def has_fped_coverage(food_id: int) -> bool:
    try:
        return int(food_id) in _fped_ok_set()
    except (TypeError, ValueError):
        return False


def _mask_kw(names: pd.Series, keywords: list[str]) -> pd.Series:
    rx = _kw_regex(tuple(keywords or []))
    if rx is None:
        return pd.Series(False, index=names.index)
    return names.str.contains(rx, na=False)


def filter_bronze_food_df(
    df: pd.DataFrame,
    food_desc: Optional[dict[int, str]] = None,
    cfg: Optional[dict] = None,
    desc_col: str = "food_desc",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Drop non-plate foods, non-FPED codes, and RTE cereal/cracker rows on lunch/dinner.
    """
    cfg = cfg or load_bronze_exclusion_config()
    out = df.copy()
    if out.empty:
        return out, {"n_in": 0, "n_out": 0}

    if food_desc is None:
        if desc_col in out.columns:
            food_desc = {
                int(i): str(n)
                for i, n in zip(
                    pd.to_numeric(out["food_id"], errors="coerce").dropna().astype(int),
                    out[desc_col].astype(str),
                )
            }
        else:
            fc = pd.read_csv(ROOT / "data" / "nhanes" / "food_code.csv", usecols=["food_id", "food_desc"])
            food_desc = dict(zip(fc.food_id.astype(int), fc.food_desc.astype(str)))

    if "meal" not in out.columns:
        if "eating_type" in out.columns:
            from algo.diet_structure_priors import meal_from_eating_type

            out["meal"] = out["eating_type"].map(meal_from_eating_type)
        elif "meal_id" in out.columns:
            from algo.diet_structure_priors import ID_TO_MEAL

            out["meal"] = out["meal_id"].map(ID_TO_MEAL)

    n_in = len(out)
    names = _desc_series(out["food_id"], food_desc)
    fids = pd.to_numeric(out["food_id"], errors="coerce")
    prefixes = [str(p).strip() for p in (cfg.get("exclude_food_id_prefixes") or []) if str(p).strip()]
    pref_excl = pd.Series(False, index=out.index)
    if prefixes:
        fid_str = fids.fillna(-1).astype(int).astype(str).str.zfill(8)
        for p in prefixes:
            pref_excl = pref_excl | fid_str.str.startswith(p)
    allow_rx = _kw_regex(tuple(_norm(x) for x in (cfg.get("allow_keywords") or [])))
    allowed = names.str.contains(allow_rx, na=False) if allow_rx else pd.Series(False, index=out.index)
    excl = (_mask_kw(names, list(cfg.get("exclude_keywords") or [])) & ~allowed) | pref_excl
    for pat in cfg.get("exclude_regex") or []:
        if pat:
            excl = excl | (names.str.contains(pat, case=False, regex=True, na=False) & ~allowed)
    n_kw = int(excl.sum())
    n_prefix = int(pref_excl.sum())
    keep = ~excl
    out = out.loc[keep].copy()
    names = names.loc[keep]

    n_meal = 0
    n_oats = 0
    if "meal" in out.columns and not out.empty:
        meals = out["meal"].astype(str).str.lower()
        oats = _mask_kw(names, list(cfg.get("breakfast_only_keywords") or [])) & ~allowed.loc[names.index]
        bad_oats = oats & ~meals.eq("breakfast")
        n_oats = int(bad_oats.sum())
        out = out.loc[~bad_oats].copy()
        names = names.loc[~bad_oats]
        meals = out["meal"].astype(str).str.lower()
        allowed_m = names.str.contains(allow_rx, na=False) if allow_rx else pd.Series(False, index=out.index)
        oats2 = _mask_kw(names, list(cfg.get("breakfast_only_keywords") or []))
        snack_only = _mask_kw(names, list(cfg.get("snack_only_keywords") or [])) & ~allowed_m
        bad_snack_slot = snack_only & ~meals.eq("snack")
        no_snack = _mask_kw(names, list(cfg.get("no_snack_keywords") or [])) & ~allowed_m
        bad_bread_snack = no_snack & meals.eq("snack")
        bs_only = _mask_kw(names, list(cfg.get("breakfast_snack_only_keywords") or [])) & ~oats2 & ~allowed_m & ~snack_only
        bad_occ = (bs_only & meals.isin(["lunch", "dinner"])) | bad_snack_slot | bad_bread_snack
        n_meal = int(bad_occ.sum())
        out = out.loc[~bad_occ].copy()

    n_fped = 0
    if bool(cfg.get("require_fped", True)) and not out.empty:
        ok_set = _fped_ok_set()
        fids = pd.to_numeric(out["food_id"], errors="coerce")
        ok_fped = fids.isin(ok_set)
        n_fped = int((~ok_fped).sum())
        out = out.loc[ok_fped].copy()

    audit = {
        "n_in": n_in,
        "n_out": len(out),
        "n_dropped_keyword": n_kw,
        "n_dropped_food_id_prefix": n_prefix,
        "n_dropped_oats_non_breakfast": n_oats,
        "n_dropped_lunch_dinner_cereal": n_meal,
        "n_dropped_no_fped": n_fped,
        "n_unique_food_in": int(pd.to_numeric(df["food_id"], errors="coerce").nunique()),
        "n_unique_food_out": int(pd.to_numeric(out["food_id"], errors="coerce").nunique())
        if len(out)
        else 0,
        "version": cfg.get("version"),
    }
    return out.reset_index(drop=True), audit
