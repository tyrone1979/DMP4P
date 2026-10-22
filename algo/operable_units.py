"""
Food mass + FNDDS serving labels.

Bronze / daily sols: keep NHANES ``grams`` unchanged; attach FNDDS serve
(``measure_desc``, ``grams_per_unit``, ``serve_qty`` = grams / gpu). No
household snap that rewrites mass.

Legacy ``grams_to_display`` / ``apply_operable_units`` still snap for
actual-diet / weekly operable paths that opt in.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "config" / "golden_week.yaml"

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

_WHOLE_UNITS = frozenset({"piece", "slice", "unit", "fruit", "fillet"})


def load_week_config(path: Optional[Path] = None) -> dict:
    p = Path(path) if path else DEFAULT_CFG
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_display_unit_table(path: str = "") -> pd.DataFrame:
    cfg = load_week_config()
    rel = path or (cfg.get("paths") or {}).get("display_unit", "data/portion_units/food_display_unit.csv")
    p = Path(rel)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["food_id"] = pd.to_numeric(df["food_id"], errors="coerce").astype("Int64")
    return df.dropna(subset=["food_id"]).astype({"food_id": int}).drop_duplicates("food_id")


def _step_for_unit(unit: str, cfg: dict, *, whole_qty: bool = False) -> float:
    u = cfg.get("units") or {}
    unit = str(unit)
    if unit in ("mL", "fl_oz"):
        return float(u.get("step_ml", 10)) if unit == "mL" else float(u.get("step_fl_oz", 1))
    if unit == "L":
        return 0.1
    if unit in ("piece", "slice"):
        return 1.0 if whole_qty else float(u.get("step_piece", 1))
    if unit in ("unit", "fruit", "fillet", "tortilla", "serving"):
        return 1.0 if whole_qty else float(u.get("step_unit", 0.5))
    if unit == "cup":
        return float(u.get("step_cup", 0.5))
    return float(u.get("step_g", 5))


def _snap(qty: float, step: float) -> float:
    """Snap qty to step grid (may be 0). No minimum-one-step floor."""
    if step <= 0:
        return float(qty)
    return float(round(float(qty) / step) * step)


def _snap_array(qty: np.ndarray, step: np.ndarray) -> np.ndarray:
    """
    Vectorized ``_snap`` using Python3 banker's ``round`` per element.

    Portion quantities are small; a tight Python loop here is still far faster
    than calling ``grams_to_display`` (dict + DataFrame lookup) per row.
    """
    qty = np.asarray(qty, dtype=float)
    step = np.asarray(step, dtype=float)
    out = np.empty(qty.shape[0], dtype=float)
    for i in range(qty.shape[0]):
        s = step[i]
        q = qty[i]
        if s <= 0 or not np.isfinite(q):
            out[i] = q if np.isfinite(q) else 0.0
        else:
            out[i] = float(round(q / s) * s)
    return out


def _max_snap_rel_diff(cfg: dict) -> float:
    return float((cfg.get("units") or {}).get("max_snap_rel_diff", 0.5))


def _near_raw_grams(grams: float, cfg: dict) -> float:
    """Keep mass close to NHANES grams_raw (fine g grid; no one-step floor)."""
    grams = float(grams or 0.0)
    if grams <= 0:
        return 0.0
    step = float((cfg.get("units") or {}).get("step_g_near_raw", 1))
    g2 = _snap(grams, step)
    if g2 <= 0:
        # sub-step sprinkle: keep one decimal rather than wiping to 0
        return round(grams, 1)
    return float(g2)


def _snap_drifts_from_raw(grams_raw: float, grams_snapped: float, cfg: dict) -> bool:
    """True if snapped mass is 0 or relative error vs grams_raw is too large."""
    raw = float(grams_raw or 0.0)
    snapped = float(grams_snapped or 0.0)
    if raw <= 0:
        return False
    if snapped <= 0:
        return True
    return abs(snapped - raw) / raw > _max_snap_rel_diff(cfg)


def _near_raw_display(grams: float, cfg: dict, *, unit_source: str = "near_raw_g") -> dict[str, Any]:
    g2 = _near_raw_grams(grams, cfg)
    return {
        "display_qty": g2,
        "display_unit": "g",
        "grams_per_unit": 1.0,
        "grams_snapped": g2,
        "unit_source": unit_source,
        "measure_desc": "gram",
    }


def _finalize_vs_raw(info: dict[str, Any], grams: float, cfg: dict) -> dict[str, Any]:
    """Refuse household snaps that diverge far from NHANES grams_raw."""
    if _snap_drifts_from_raw(grams, float(info.get("grams_snapped", 0.0)), cfg):
        return _near_raw_display(grams, cfg)
    return info


def _max_qty_for_unit(unit: str, cfg: dict) -> Optional[float]:
    u = cfg.get("units") or {}
    unit = str(unit)
    key = {
        "cup": "max_qty_cup",
        "piece": "max_qty_piece",
        "slice": "max_qty_slice",
        "unit": "max_qty_unit",
        "fruit": "max_qty_unit",
        "fillet": "max_qty_unit",
    }.get(unit)
    if not key:
        return None
    v = u.get(key)
    return float(v) if v is not None else None


def grams_to_display(
    food_id: int,
    grams: float,
    display_table: Optional[pd.DataFrame] = None,
    cfg: Optional[dict] = None,
    *,
    whole_qty: bool = False,
    clamp_to_max_qty: bool = False,
) -> dict[str, Any]:
    """
    Canonical conversion: one food_id + grams → display qty/unit + snapped grams.

    All batch paths must match this function.
    When ``clamp_to_max_qty`` is True, qty above household max is clamped (keep unit)
    instead of falling back to raw grams.
    """
    cfg = cfg or load_week_config()
    table = display_table if display_table is not None else load_display_unit_table()
    grams = float(grams or 0.0)
    row = None
    if table is not None and len(table):
        hit = table[table["food_id"] == int(food_id)]
        if len(hit):
            row = hit.iloc[0]

    if row is None:
        step = _step_for_unit("g", cfg, whole_qty=whole_qty)
        g2 = _snap(grams, step)
        return _finalize_vs_raw(
            {
                "display_qty": g2,
                "display_unit": "g",
                "grams_per_unit": 1.0,
                "grams_snapped": g2,
                "unit_source": "fallback_g",
                "measure_desc": "gram",
            },
            grams,
            cfg,
        )

    unit = str(row.get("display_unit", "g"))
    gpu = float(row.get("grams_per_unit", 1.0) or 1.0)
    if gpu <= 0:
        gpu = 1.0
        unit = "g"

    min_piece_g = float((cfg.get("units") or {}).get("min_grams_per_piece", 2.0))
    if unit in ("piece", "slice", "unit") and gpu < min_piece_g:
        step_g = _step_for_unit("g", cfg, whole_qty=whole_qty)
        g2 = _snap(grams, step_g)
        return _finalize_vs_raw(
            {
                "display_qty": g2,
                "display_unit": "g",
                "grams_per_unit": 1.0,
                "grams_snapped": g2,
                "unit_source": "fallback_g_tiny_piece",
                "measure_desc": "gram",
            },
            grams,
            cfg,
        )

    raw_qty = grams / gpu
    step = _step_for_unit(unit, cfg, whole_qty=whole_qty)
    qty = _snap(raw_qty, step)
    max_q = _max_qty_for_unit(unit, cfg)
    if max_q is not None and qty > max_q:
        if clamp_to_max_qty:
            qty = float(max_q)
        else:
            step_g = _step_for_unit("g", cfg, whole_qty=whole_qty)
            g2 = _snap(grams, step_g)
            return _finalize_vs_raw(
                {
                    "display_qty": g2,
                    "display_unit": "g",
                    "grams_per_unit": 1.0,
                    "grams_snapped": g2,
                    "unit_source": "fallback_g_max_qty",
                    "measure_desc": "gram",
                },
                grams,
                cfg,
            )

    display_unit = unit
    display_qty = qty
    if unit == "mL" and qty >= 1000:
        display_unit = "L"
        display_qty = round(qty / 1000.0, 2)
        grams_snapped = float(display_qty) * 1000.0 * gpu
    else:
        grams_snapped = qty * gpu

    return _finalize_vs_raw(
        {
            "display_qty": float(display_qty),
            "display_unit": display_unit,
            "grams_per_unit": gpu,
            "grams_snapped": float(grams_snapped),
            "unit_source": str(row.get("unit_source", "fndds")),
            "measure_desc": str(row.get("measure_desc", "")),
        },
        grams,
        cfg,
    )


def scale_nutrients_row(row: pd.Series, grams_old: float, grams_new: float) -> pd.Series:
    out = row.copy()
    if grams_old <= 0:
        out["grams"] = grams_new
        return out
    factor = grams_new / grams_old
    for n in NUTRIENTS:
        if n in out.index and pd.notna(out[n]):
            out[n] = float(out[n]) * factor
    out["grams"] = grams_new
    return out


def _restore_to_grams_raw(df: pd.DataFrame) -> pd.DataFrame:
    """If grams were previously snapped, restore nutrients/grams to grams_raw basis."""
    out = df.copy()
    if "grams_raw" not in out.columns:
        out["grams_raw"] = pd.to_numeric(out.get("grams", 0), errors="coerce").fillna(0.0)
        return out
    raw = pd.to_numeric(out["grams_raw"], errors="coerce")
    cur = pd.to_numeric(out.get("grams", 0), errors="coerce")
    mask = raw.notna() & cur.notna() & (cur > 0) & (raw > 0) & ((raw - cur).abs() > 1e-6)
    if mask.any():
        restore = (raw / cur).where(mask, 1.0)
        for n in NUTRIENTS:
            if n in out.columns:
                out[n] = pd.to_numeric(out[n], errors="coerce").fillna(0.0) * restore
    out["grams"] = raw.fillna(cur).fillna(0.0)
    out["grams_raw"] = pd.to_numeric(out["grams"], errors="coerce").fillna(0.0)
    return out


def apply_operable_units(
    items: pd.DataFrame,
    display_table: Optional[pd.DataFrame] = None,
    cfg: Optional[dict] = None,
    scale_nutrients: bool = True,
    *,
    whole_qty: bool = False,
    prefer_grams_raw: bool = True,
    clamp_to_max_qty: bool = False,
) -> pd.DataFrame:
    """
    Batch conversion matching ``grams_to_display`` exactly.

    Vectorized lookups + Python-round snap (same as scalar ``_snap``).
    ``clamp_to_max_qty``: keep household unit and clamp qty (Bronze / sequential GA).
    """
    cfg = cfg or load_week_config()
    table = display_table if display_table is not None else load_display_unit_table()
    df = items.copy()
    df["food_id"] = pd.to_numeric(df["food_id"], errors="coerce").fillna(0).astype(int)
    # Preserve NHANES original if present; never overwrite with snapped grams.
    nhanes_raw = None
    if "grams_raw" in df.columns:
        nhanes_raw = pd.to_numeric(df["grams_raw"], errors="coerce").fillna(0.0)
    if prefer_grams_raw:
        df = _restore_to_grams_raw(df)
    else:
        # Snap from current ``grams`` (already operable); keep NHANES ``grams_raw``.
        g_cur = pd.to_numeric(df.get("grams", 0), errors="coerce").fillna(0.0)
        df["_snap_base"] = g_cur
        if nhanes_raw is None:
            df["grams_raw"] = g_cur

    for c in ("display_qty", "display_unit", "grams_snapped", "grams_per_unit", "unit_source", "measure_desc"):
        if c in df.columns:
            df = df.drop(columns=[c])

    n = len(df)
    if "_snap_base" in df.columns:
        g_raw = pd.to_numeric(df["_snap_base"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        df = df.drop(columns=["_snap_base"])
    else:
        g_raw = pd.to_numeric(df["grams_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if nhanes_raw is not None:
        df["grams_raw"] = nhanes_raw.reindex(df.index).fillna(0.0).to_numpy()
    fids = df["food_id"].to_numpy(dtype=int)
    step_g = float(_step_for_unit("g", cfg, whole_qty=whole_qty))
    min_piece_g = float((cfg.get("units") or {}).get("min_grams_per_piece", 2.0))

    # Align table columns to rows (NaN = missing food_id)
    if table is not None and len(table):
        meta = (
            table.drop_duplicates("food_id")
            .set_index("food_id")[["display_unit", "grams_per_unit", "unit_source", "measure_desc"]]
        )
        aligned = meta.reindex(fids)
        unit = aligned["display_unit"].astype(object).to_numpy()
        gpu = pd.to_numeric(aligned["grams_per_unit"], errors="coerce").to_numpy(dtype=float)
        src = aligned["unit_source"].astype(object).to_numpy()
        meas = aligned["measure_desc"].astype(object).to_numpy()
        miss = pd.isna(aligned["display_unit"]).to_numpy()
    else:
        unit = np.array([None] * n, dtype=object)
        gpu = np.full(n, np.nan)
        src = np.array([None] * n, dtype=object)
        meas = np.array([None] * n, dtype=object)
        miss = np.ones(n, dtype=bool)

    # Normalize: missing row → grams fallback; gpu<=0 → treat as unit=g, gpu=1 (keep table source)
    gpu_bad = (~miss) & (~np.isfinite(gpu) | (gpu <= 0))
    unit_s = np.empty(n, dtype=object)
    gpu_s = np.empty(n, dtype=float)
    for i in range(n):
        if miss[i]:
            unit_s[i] = "g"
            gpu_s[i] = 1.0
        elif gpu_bad[i]:
            unit_s[i] = "g"
            gpu_s[i] = 1.0
        else:
            unit_s[i] = str(unit[i])
            gpu_s[i] = float(gpu[i])

    tiny = np.isin(unit_s, ["piece", "slice", "unit"]) & (gpu_s < min_piece_g) & (~miss)

    # Steps per row
    step = np.full(n, step_g, dtype=float)
    for u_name in ("mL", "L", "cup", "piece", "slice", "unit", "fruit", "fillet", "tortilla"):
        step = np.where(unit_s == u_name, _step_for_unit(u_name, cfg, whole_qty=whole_qty), step)

    raw_qty = np.divide(g_raw, gpu_s, out=np.zeros(n, dtype=float), where=gpu_s > 0)
    qty = _snap_array(raw_qty, step)

    # Max qty overflow → grams
    max_q = np.full(n, np.inf, dtype=float)
    for u_name in ("cup", "piece", "slice", "unit", "fruit", "fillet"):
        mq = _max_qty_for_unit(u_name, cfg)
        if mq is not None:
            max_q = np.where(unit_s == u_name, float(mq), max_q)
    over = (~miss) & (~tiny) & (qty > max_q)
    if clamp_to_max_qty:
        qty = np.where(over, max_q, qty)
        clamped = over
        over = np.zeros(n, dtype=bool)
    else:
        clamped = np.zeros(n, dtype=bool)

    use_g = miss | tiny | over
    qty_g = _snap_array(g_raw, np.full(n, step_g, dtype=float))

    display_qty = np.where(use_g, qty_g, qty)
    display_unit = np.where(use_g, "g", unit_s)
    grams_per_unit = np.where(use_g, 1.0, gpu_s)
    grams_snapped = np.where(use_g, qty_g, qty * gpu_s)

    unit_source = np.empty(n, dtype=object)
    measure_desc = np.empty(n, dtype=object)
    for i in range(n):
        if miss[i]:
            unit_source[i] = "fallback_g"
            measure_desc[i] = "gram"
        elif tiny[i]:
            unit_source[i] = "fallback_g_tiny_piece"
            measure_desc[i] = "gram"
        elif over[i]:
            unit_source[i] = "fallback_g_max_qty"
            measure_desc[i] = "gram"
        else:
            s = src[i]
            m = meas[i]
            unit_source[i] = (
                "clamp_max_qty"
                if clamped[i]
                else (
                    "fndds"
                    if s is None or (isinstance(s, float) and np.isnan(s))
                    else str(s)
                )
            )
            measure_desc[i] = (
                ""
                if m is None or (isinstance(m, float) and np.isnan(m))
                else str(m)
            )

    # mL → L (mass follows rounded L display — same as grams_to_display)
    to_l = (~use_g) & (unit_s == "mL") & (qty >= 1000)
    if to_l.any():
        # Python round(..., 2) per element
        lit = np.empty(n, dtype=float)
        for i in np.flatnonzero(to_l):
            lit[i] = round(float(qty[i]) / 1000.0, 2)
        display_unit = np.where(to_l, "L", display_unit)
        display_qty = np.where(to_l, lit, display_qty)
        grams_snapped = np.where(to_l, lit * 1000.0 * gpu_s, grams_snapped)

    # No min-one-step floor; if mass drifts far from NHANES raw → keep near raw (g)
    rel = _max_snap_rel_diff(cfg)
    with np.errstate(divide="ignore", invalid="ignore"):
        drift = (g_raw > 0) & (
            (grams_snapped <= 0) | (np.abs(grams_snapped - g_raw) / g_raw > rel)
        )
    if drift.any():
        for i in np.flatnonzero(drift):
            g2 = _near_raw_grams(float(g_raw[i]), cfg)
            display_qty[i] = g2
            display_unit[i] = "g"
            grams_per_unit[i] = 1.0
            grams_snapped[i] = g2
            unit_source[i] = "near_raw_g"
            measure_desc[i] = "gram"

    df["display_qty"] = display_qty
    df["display_unit"] = display_unit
    df["grams_per_unit"] = grams_per_unit
    df["grams_snapped"] = grams_snapped
    df["unit_source"] = unit_source
    df["measure_desc"] = measure_desc

    if scale_nutrients:
        factor = np.where(g_raw > 0, grams_snapped / g_raw, 1.0)
        for nut in NUTRIENTS:
            if nut in df.columns:
                df[nut] = pd.to_numeric(df[nut], errors="coerce").fillna(0.0) * factor
        df["grams"] = grams_snapped
    return df


def attach_fndds_serves(
    items: pd.DataFrame,
    *,
    display_table: Optional[pd.DataFrame] = None,
    prefer_grams_raw: bool = True,
) -> pd.DataFrame:
    """
    Keep mass unchanged; attach FNDDS serving metadata.

    Writes:
      - measure_desc, grams_per_unit, unit_source, display_unit (from table)
      - serve_qty = grams / grams_per_unit (no rounding)
      - display_qty = serve_qty (compat alias; does **not** rewrite grams)
    """
    df = items.copy()
    if prefer_grams_raw:
        df = _restore_to_grams_raw(df)
    else:
        if "grams_raw" not in df.columns:
            df["grams_raw"] = pd.to_numeric(df.get("grams", 0), errors="coerce").fillna(0.0)

    grams = pd.to_numeric(df.get("grams", 0), errors="coerce").fillna(0.0)
    df["grams"] = grams
    df["grams_raw"] = pd.to_numeric(df["grams_raw"], errors="coerce").fillna(grams)

    table = display_table if display_table is not None else load_display_unit_table()
    if table is None or table.empty:
        df["display_unit"] = "g"
        df["grams_per_unit"] = 1.0
        df["measure_desc"] = "gram"
        df["unit_source"] = "grams_raw"
        df["serve_qty"] = grams
        df["display_qty"] = grams
        if "grams_snapped" in df.columns:
            df = df.drop(columns=["grams_snapped"])
        return df

    aligned = (
        df[["food_id"]]
        .merge(
            table.drop_duplicates("food_id").set_index("food_id")[
                ["display_unit", "grams_per_unit", "unit_source", "measure_desc"]
            ],
            left_on="food_id",
            right_index=True,
            how="left",
        )
    )
    unit = aligned["display_unit"].astype(object).where(aligned["display_unit"].notna(), "g")
    gpu = pd.to_numeric(aligned["grams_per_unit"], errors="coerce").fillna(1.0).clip(lower=1e-9)
    src = aligned["unit_source"].astype(object).where(aligned["unit_source"].notna(), "grams_raw")
    meas = aligned["measure_desc"].astype(object).where(aligned["measure_desc"].notna(), "gram")

    miss = aligned["display_unit"].isna()
    unit = unit.where(~miss, "g")
    gpu = gpu.where(~miss, 1.0)
    src = src.where(~miss, "grams_raw")
    meas = meas.where(~miss, "gram")

    serve = grams.to_numpy(dtype=float) / gpu.to_numpy(dtype=float)
    df["display_unit"] = unit.to_numpy()
    df["grams_per_unit"] = gpu.to_numpy(dtype=float)
    df["unit_source"] = src.to_numpy()
    df["measure_desc"] = meas.to_numpy()
    df["serve_qty"] = serve
    df["display_qty"] = serve  # informational; mass stays in ``grams``
    if "grams_snapped" in df.columns:
        df = df.drop(columns=["grams_snapped"])
    return df


def apply_daily_grams_only(items: pd.DataFrame) -> pd.DataFrame:
    """Daily menus: keep grams; attach FNDDS serves (no household snap)."""
    return attach_fndds_serves(items, prefer_grams_raw=True)


def canonicalize_operable_pool(
    food_user: pd.DataFrame,
    *,
    display_table: Optional[pd.DataFrame] = None,
    cfg: Optional[dict] = None,
    drop_over_max_qty: bool = False,
    resnap_from_raw: bool = True,
) -> pd.DataFrame:
    """
    Bronze pool: NHANES grams + FNDDS serve labels (no 化零为整 mass rewrite).

    Column contract:
      - ``grams_raw`` / ``grams``: same NHANES intake grams (GA uses ``grams``)
      - ``serve_qty`` / ``display_unit`` / ``measure_desc``: FNDDS serving info only
    """
    _ = cfg  # max-qty drop disabled for bronze; kept for call-site compat
    df = food_user.copy()
    disp = display_table if display_table is not None else load_display_unit_table()

    if "grams_raw" not in df.columns:
        df["grams_raw"] = pd.to_numeric(df.get("grams", 0), errors="coerce").fillna(0.0)

    # Always restore to NHANES mass; never snap for bronze pool.
    df = attach_fndds_serves(
        df,
        display_table=disp,
        prefer_grams_raw=bool(resnap_from_raw) or True,
    )

    if "meal" not in df.columns and "eating_type" in df.columns:
        from algo.diet_structure_priors import meal_from_eating_type

        df["meal"] = df["eating_type"].map(meal_from_eating_type)

    # optional legacy filter (off by default — serve_qty is not a snap target)
    if drop_over_max_qty and "serve_qty" in df.columns and "display_unit" in df.columns:
        cfg = cfg or load_week_config()
        qty = pd.to_numeric(df["serve_qty"], errors="coerce").fillna(0.0)
        unit = df["display_unit"].astype(str)
        keep = pd.Series(True, index=df.index)
        for u_name in ("cup", "piece", "slice", "unit", "fruit", "fillet", "fl_oz", "serving"):
            mq = _max_qty_for_unit(u_name, cfg)
            if mq is None:
                continue
            keep &= ~((unit == u_name) & (qty > float(mq) + 1e-9))
        df = df.loc[keep].copy()

    keys = [c for c in ("food_id", "meal", "grams") if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="first")
    return df.reset_index(drop=True)


def split_meal_pools(food_user: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Explicit breakfast / lunch / dinner / snack candidate pools."""
    df = food_user
    if "meal" not in df.columns and "eating_type" in df.columns:
        from algo.diet_structure_priors import meal_from_eating_type

        df = df.copy()
        df["meal"] = df["eating_type"].map(meal_from_eating_type)
    meal = df["meal"].astype(str).str.lower() if "meal" in df.columns else pd.Series("", index=df.index)
    out = {}
    for m in ("breakfast", "lunch", "dinner", "snack"):
        out[m] = df.loc[meal == m].copy()
    return out
