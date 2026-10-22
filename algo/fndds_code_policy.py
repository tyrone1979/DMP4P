"""
FNDDS food_id prefix policy: type labels, main-cuisine, sequential-ban, unit kinds.

Config: config/fndds_food_code_policy.yaml
Match longest prefix first on the 8-digit zero-padded food_id.

Main dish (``main_cuisine``) and N+1 ban (``sequential_ban``) are independent.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "config" / "fndds_food_code_policy.yaml"


@lru_cache(maxsize=2)
def load_fndds_code_policy(path: str = "") -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CFG
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def food_id_str(food_id) -> str:
    try:
        return str(int(float(food_id))).zfill(8)
    except (TypeError, ValueError):
        return ""


def match_prefix(food_id, prefixes: Iterable[str]) -> Optional[str]:
    """Longest matching prefix among ``prefixes``, or None."""
    s = food_id_str(food_id)
    if not s:
        return None
    best = None
    for p in prefixes:
        pp = str(p or "").strip()
        if not pp:
            continue
        if s.startswith(pp) and (best is None or len(pp) > len(best)):
            best = pp
    return best


def food_type_label(food_id, cfg: Optional[dict] = None) -> str:
    cfg = cfg or load_fndds_code_policy()
    table = cfg.get("type_by_prefix") or {}
    hit = match_prefix(food_id, table.keys())
    return str(table[hit]) if hit else "other"


def unit_kind(food_id, cfg: Optional[dict] = None) -> str:
    """
    Preferred unit family for display-unit picking:
    liquid | dry_powder | fish | bread | cracker | cereal | snack_chip |
    cheese | nuts | produce | protein | grain | other
    """
    cfg = cfg or load_fndds_code_policy()
    table = cfg.get("unit_kind_by_prefix") or {}
    hit = match_prefix(food_id, table.keys())
    if hit:
        return str(table[hit])
    major = food_id_str(food_id)[:1]
    return str((cfg.get("unit_kind_by_major") or {}).get(major, "other"))


def _digits_set(raw) -> set[str]:
    return {str(d).strip() for d in (raw or []) if str(d).strip()}


def is_main_cuisine_by_code(food_id, cfg: Optional[dict] = None) -> bool:
    """
    Plate main dish (主菜):
      - majors in ``main_cuisine.first_digits`` (default 2* meat/fish, 3* eggs)
      - or longest match in ``main_cuisine.prefixes`` (meatless 418*, veg entrees, …)

    Broad 4* legumes/nuts and plain veg sides (752*) are not mains unless listed
    under prefixes (e.g. 418 meatless analogs).
    """
    cfg = cfg or load_fndds_code_policy()
    mc = cfg.get("main_cuisine") or {}
    digits = _digits_set(mc.get("first_digits"))
    if not digits:
        ban = cfg.get("sequential_ban") or {}
        digits = _digits_set(ban.get("main_cuisine_first_digits") or ["2", "3"])
        digits.discard("4")
    s = food_id_str(food_id)
    if s and s[0] in digits:
        return True
    prefs = [str(p).strip() for p in (mc.get("prefixes") or []) if str(p).strip()]
    return match_prefix(food_id, prefs) is not None


def is_exact_ban_major(food_id, cfg: Optional[dict] = None) -> bool:
    """True if food_id major digit is in sequential_ban.exact_ban_first_digits."""
    cfg = cfg or load_fndds_code_policy()
    ban = cfg.get("sequential_ban") or {}
    digits = _digits_set(ban.get("exact_ban_first_digits"))
    if not digits:
        # legacy: old main_cuisine_first_digits under sequential_ban
        digits = _digits_set(ban.get("main_cuisine_first_digits") or ["2", "3"])
    s = food_id_str(food_id)
    return bool(s) and s[0] in digits


def is_sequential_family_food(food_id, cfg: Optional[dict] = None) -> bool:
    """True if food_id is in a sequential family-ban prefix or unit_kind family."""
    cfg = cfg or load_fndds_code_policy()
    ban_cfg = cfg.get("sequential_ban") or {}
    fam = ban_cfg.get("family_prefixes") or []
    if match_prefix(food_id, fam) is not None:
        return True
    kinds = {str(k).strip() for k in (ban_cfg.get("family_unit_kinds") or []) if str(k).strip()}
    return bool(kinds) and unit_kind(food_id, cfg) in kinds


def is_sequential_ban_food(food_id, cfg: Optional[dict] = None) -> bool:
    """True if food would participate in N+1 ban (exact major or family)."""
    return is_exact_ban_major(food_id, cfg) or is_sequential_family_food(food_id, cfg)


def collect_sequential_ban_food_ids(
    day: pd.DataFrame,
    *,
    pool_food_ids: Optional[Iterable[int]] = None,
    cfg: Optional[dict] = None,
    include_mains: bool = True,
    expand_prefix_family: Optional[bool] = None,
    ban_all_day_food_ids: bool = True,
) -> set[int]:
    """
    After solution N: ban food_ids for N+1.

    Default (``expand_prefix_family`` false / config):
      - every ``food_id`` on the day (``ban_all_day_food_ids``), and/or
      - exact-ban-major ids on the day (2*/3* when ``include_mains``).

    Optional family wipe (``expand_prefix_family`` true):
      - also remove entire FNDDS prefix / unit_kind families from the pool.
    """
    if cfg is None:
        load_fndds_code_policy.cache_clear()
        cfg = load_fndds_code_policy()
    ban_cfg = cfg.get("sequential_ban") or {}
    if expand_prefix_family is None:
        expand_prefix_family = bool(ban_cfg.get("expand_prefix_family", False))
    fam_prefixes = [str(p).strip() for p in (ban_cfg.get("family_prefixes") or []) if str(p).strip()]
    exact_prefixes = [str(p).strip() for p in (ban_cfg.get("exact_prefixes") or []) if str(p).strip()]
    fam_kinds = {str(k).strip() for k in (ban_cfg.get("family_unit_kinds") or []) if str(k).strip()}

    out: set[int] = set()
    if day is None or day.empty or "food_id" not in day.columns:
        return out

    hit_families: set[str] = set()
    hit_kinds: set[str] = set()
    for fid in pd.to_numeric(day["food_id"], errors="coerce").dropna().astype(int):
        fid_i = int(fid)
        if ban_all_day_food_ids:
            out.add(fid_i)
        if include_mains and is_exact_ban_major(fid_i, cfg):
            out.add(fid_i)
        fam = match_prefix(fid_i, fam_prefixes)
        if fam:
            hit_families.add(fam)
            out.add(fid_i)
        ex = match_prefix(fid_i, exact_prefixes)
        if ex:
            out.add(fid_i)
        if fam_kinds:
            uk = unit_kind(fid_i, cfg)
            if uk in fam_kinds:
                hit_kinds.add(uk)
                out.add(fid_i)

    if expand_prefix_family and pool_food_ids is not None and (hit_families or hit_kinds):
        for fid in pool_food_ids:
            try:
                fid_i = int(fid)
            except (TypeError, ValueError):
                continue
            s = food_id_str(fid_i)
            if hit_families and any(s.startswith(p) for p in hit_families):
                out.add(fid_i)
                continue
            if hit_kinds and unit_kind(fid_i, cfg) in hit_kinds:
                out.add(fid_i)
    return out
