"""Build and load a frozen candidate-pool snapshot per experiment unit."""
from __future__ import annotations

import ast
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from algo.bronze_difficulty import classify_bronze_difficulty
from algo.diet_structure_priors import (
    filter_candidate_pool,
    load_ga_config,
    load_structure_prior,
)
from algo.structure_category import select_personal_category_for_bronze
from algo.structure_cell_pool import build_structure_cell_pool
from dmp4p_runtime.constraints import ConstraintSpec


def parse_target(t) -> dict:
    if isinstance(t, dict):
        return t
    return ast.literal_eval(str(t))


def demo_from_structure_key(structure_key: str) -> dict[str, Any]:
    """
    Representative demographics from stratum label
    e.g. ``Female|normal (18.5–24.9)|19–30``.
    """
    parts = [p.strip() for p in str(structure_key).split("|")]
    sex = parts[0] if parts else "unknown"
    bmi_lab = parts[1] if len(parts) > 1 else ""
    age_lab = parts[2] if len(parts) > 2 else ""
    bmi = 22.0
    low = bmi_lab.lower()
    if "underweight" in low:
        bmi = 17.0
    elif "normal" in low:
        bmi = 22.0
    elif "overweight" in low:
        bmi = 27.0
    elif "obesity" in low or "obese" in low:
        bmi = 33.0
    age = 25.0
    if "19" in age_lab and "30" in age_lab:
        age = 25.0
    elif "31" in age_lab:
        age = 40.0
    elif "51" in age_lab:
        age = 60.0
    elif "71" in age_lab:
        age = 75.0
    return {"sex": sex, "bmi": bmi, "age": age, "structure_key": structure_key}


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)


def hash_payload(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(stable_json(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


def pool_content_hash(pool: pd.DataFrame, nutrient_cols: list[str]) -> str:
    cols = ["food_id", "meal", "meal_id", "grams", "bench_placement_allowed"] + [
        c for c in nutrient_cols if c in pool.columns
    ]
    sub = pool.loc[:, [c for c in cols if c in pool.columns]].copy()
    for c in sub.columns:
        if c != "meal":
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    # row-wise stable fingerprint
    payload = sub.to_csv(index=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def build_pool_snapshot(
    food_df: pd.DataFrame,
    structure_key: str,
    target_dict: dict,
    *,
    config: Optional[dict] = None,
    user_history_food_ids=None,
    pool_seed: int = 0,
    cell_pool_seed: int = 0,
) -> dict[str, Any]:
    """
    Build one shared candidate pool for an experiment unit.

    Pipeline matches production: filter_candidate_pool → personal structure →
    build_structure_cell_pool (deterministic seeds).
    """
    cfg = config or load_ga_config()
    demo = demo_from_structure_key(structure_key)
    t0 = time.perf_counter()
    personal = select_personal_category_for_bronze(demo, target_dict)
    flat = filter_candidate_pool(
        food_df,
        target_dict,
        user_history_food_ids=user_history_food_ids,
        config=cfg,
        pool_seed_offset=int(pool_seed),
    )
    t_flat = time.perf_counter()
    quota = int(((cfg.get("structure") or {}).get("cell_ga") or {}).get("per_cell_quota", 400))
    ordered, ranges, meta = build_structure_cell_pool(
        flat,
        personal=personal,
        per_cell_quota=quota,
        random_state=int(cell_pool_seed),
    )
    from dmp4p_runtime.placement_flags import attach_placement_flags
    from dmp4p_runtime.candidate_ids import (
        assign_candidate_ids,
        pool_candidate_set_hash,
    )

    ordered, placement_meta = attach_placement_flags(ordered)
    nutrient_cols = list(target_dict.keys())
    ordered = assign_candidate_ids(ordered, nutrient_cols)
    cand_set_hash = pool_candidate_set_hash(ordered)
    t_cell = time.perf_counter()
    prior = load_structure_prior(
        sex=demo.get("sex"), bmi=demo.get("bmi"), age=demo.get("age"), config=cfg
    )
    spec = ConstraintSpec.from_ga_config(cfg, structure_prior=prior, require_bld=False)
    ph = pool_content_hash(ordered, nutrient_cols)
    cfg_hash = hash_payload(
        {k: cfg[k] for k in cfg if k not in ("paths",)},
        spec.to_dict(),
        pool_seed,
        cell_pool_seed,
        quota,
    )
    diff = classify_bronze_difficulty(target_dict, personal=personal)
    # Serialize personal lightly (bands only sizes)
    personal_meta = None
    if personal is not None:
        personal_meta = {
            "stratum": personal.get("stratum"),
            "g_scale": personal.get("g_scale"),
            "expected_total_g": personal.get("expected_total_g"),
            "n_bands": len(personal.get("bands") or {}),
            "forbidden_cells": list(meta.get("forbidden_cells") or []),
        }
    ranges_ser = {
        m: {str(k): [int(a), int(b)] for k, (a, b) in cells.items()}
        for m, cells in ranges.items()
    }
    return {
        "structure_key": structure_key,
        "target": target_dict,
        "demo": demo,
        "pool": ordered,
        "cell_ranges": ranges_ser,
        "forbidden_cells": [list(x) for x in (meta.get("forbidden_cells") or [])],
        "pool_hash": ph,
        "candidate_set_hash": cand_set_hash,
        "config_hash": cfg_hash,
        "n_pool": int(len(ordered)),
        "n_flat_pool": int(len(flat)),
        "per_cell_quota": quota,
        "pool_seed": int(pool_seed),
        "cell_pool_seed": int(cell_pool_seed),
        "constraint_spec": spec.to_dict(),
        "structure_prior": {k: prior[k] for k in prior if not str(k).startswith("_")},
        "personal_meta": personal_meta,
        "personal": personal,
        "difficulty": diff,
        "placement_meta": placement_meta,
        "require_bld": False,
        "timings": {
            "flat_pool_s": t_flat - t0,
            "cell_pool_s": t_cell - t_flat,
            "total_pool_build_s": t_cell - t0,
        },
    }


def save_snapshot(snap: dict[str, Any], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "pool.parquet"
    try:
        snap["pool"].to_parquet(pool_path, index=False)
    except Exception:
        pool_path = out_dir / "pool.pkl"
        snap["pool"].to_pickle(pool_path)
    meta = {k: v for k, v in snap.items() if k not in ("pool", "personal")}
    # personal bands can be large; store separately
    if snap.get("personal") is not None:
        pers_path = out_dir / "personal.json"
        # bands values are plain floats
        pers_path.write_text(stable_json(snap["personal"]), encoding="utf-8")
        meta["personal_path"] = pers_path.name
    meta["pool_file"] = pool_path.name
    (out_dir / "meta.json").write_text(stable_json(meta), encoding="utf-8")
    return out_dir


def load_snapshot(snap_dir: Path) -> dict[str, Any]:
    snap_dir = Path(snap_dir)
    meta = json.loads((snap_dir / "meta.json").read_text(encoding="utf-8"))
    pool_file = snap_dir / meta["pool_file"]
    if pool_file.suffix == ".parquet":
        pool = pd.read_parquet(pool_file)
    else:
        pool = pd.read_pickle(pool_file)
    personal = None
    if meta.get("personal_path"):
        personal = json.loads((snap_dir / meta["personal_path"]).read_text(encoding="utf-8"))
    # Backward compat: older snapshots without placement flags / candidate_id
    if "bench_placement_allowed" not in pool.columns:
        from dmp4p_runtime.placement_flags import attach_placement_flags

        pool, placement_meta = attach_placement_flags(pool)
        meta["placement_meta"] = placement_meta
        meta["placement_flags_attached_on_load"] = True
    if "candidate_id" not in pool.columns:
        from dmp4p_runtime.candidate_ids import (
            assign_candidate_ids,
            pool_candidate_set_hash,
        )

        nuts = list((meta.get("target") or {}).keys()) if isinstance(meta.get("target"), dict) else []
        if not nuts:
            nuts = [c for c in pool.columns if c in (
                "calorie", "protein", "carb", "sugar", "fiber", "saturated_fat",
                "cholesterol", "sodium", "potassium", "calcium", "iron",
            )]
        pool = assign_candidate_ids(pool, nuts)
        meta["candidate_set_hash"] = pool_candidate_set_hash(pool)
        meta["candidate_ids_attached_on_load"] = True
    meta["pool"] = pool
    meta["personal"] = personal
    # restore cell_ranges int keys
    ranges = {}
    for m, cells in (meta.get("cell_ranges") or {}).items():
        ranges[m] = {int(k): (int(a), int(b)) for k, (a, b) in cells.items()}
    meta["cell_ranges_restored"] = ranges
    meta["constraint_spec_obj"] = ConstraintSpec(**{
        k: v for k, v in (meta.get("constraint_spec") or {}).items() if k != "notes"
    })
    # restore notes
    notes = (meta.get("constraint_spec") or {}).get("notes")
    if notes:
        meta["constraint_spec_obj"].notes = list(notes)
    return meta
