#!/usr/bin/env python3
"""
Bronze meal-plan generation v2: GPU-GA with nutrient + B/L/D hard constraints,
structure/diversity Layer-B scoring, and per-user food_id-set anti-repetition.

Work unit = (structure stratum × nutrient target). Same target with different
sex×BMI×age strata runs separately. Supports per-unit checkpoint/resume and GPU sharding.

Usage:
  PYTHONPATH=. python find_bronze_multiobj.py --data data/nhanes/data -d 0 --all-targets \\
      --out outputs/ga_v2_plate_full --shard 0/2 --escalation-ladder --sequential-ban
  PYTHONPATH=. python find_bronze_multiobj.py --data data/nhanes/data -d 1 --all-targets \\
      --out outputs/ga_v2_plate_full --shard 1/2 --escalation-ladder --sequential-ban
"""
from __future__ import annotations

import argparse
import ast
import datetime
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from algo.diet_structure_priors import (  # noqa: E402
    enrich_food_dataframe,
    filter_candidate_pool,
    food_id_fingerprint,
    fingerprint_hash,
    load_ga_config,
    load_structure_prior,
    sex_label,
    target_needs_low_na_high_k,
)
from ga_gpu_v2 import FoodCombinerV2, assign_solutions_to_users  # noqa: E402
from utils.target_calculation import get_nutrition_target_range_by_situation  # noqa: E402


def load_unique_solutions_from_checkpoint(
    ckpt_dir: Path | None, target_hash: str
) -> list[pd.DataFrame]:
    """
    Load prior incomplete/ok sols for a target, one DataFrame per unique fingerprint.

    Checkpoint CSVs may repeat the same menu across users — collapse to exemplars
    ordered by seq_sol_index (then first seen).
    """
    if ckpt_dir is None:
        return []
    path = Path(ckpt_dir) / f"target_{target_hash}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return []
    if df.empty or "food_id" not in df.columns:
        return []

    df = df.copy()
    if "fingerprint_hash" not in df.columns:
        # Build fp from groups: seq_sol_index → day → whole file
        if "seq_sol_index" in df.columns:
            gcol = "seq_sol_index"
        elif "day" in df.columns:
            gcol = "day"
        else:
            gcol = None
        if gcol:
            fp_map = {
                k: fingerprint_hash(food_id_fingerprint(g["food_id"]))
                for k, g in df.groupby(gcol, sort=False)
            }
            df["fingerprint_hash"] = df[gcol].map(fp_map)
        else:
            df["fingerprint_hash"] = fingerprint_hash(food_id_fingerprint(df["food_id"]))

    if "seq_sol_index" in df.columns:
        order = (
            df.groupby("fingerprint_hash", sort=False)["seq_sol_index"]
            .min()
            .sort_values()
        )
        fp_order = [str(x) for x in order.index]
    else:
        fp_order = list(dict.fromkeys(df["fingerprint_hash"].astype(str)))

    sols: list[pd.DataFrame] = []
    seen: set[str] = set()
    for fp in fp_order:
        if fp in seen or fp in ("", "nan", "None"):
            continue
        g = df[df["fingerprint_hash"].astype(str) == fp]
        if g.empty:
            continue
        if "user_id" in g.columns:
            uid = g["user_id"].iloc[0]
            g = g[g["user_id"] == uid]
        seen.add(fp)
        sols.append(g.reset_index(drop=True).copy())
    return sols


def merge_solution_lists(
    existing: list[pd.DataFrame], new: list[pd.DataFrame] | None
) -> list[pd.DataFrame]:
    """Union by fingerprint_hash; existing first; re-number seq_sol_index."""
    out: list[pd.DataFrame] = []
    seen: set[str] = set()
    for sol in list(existing or []) + list(new or []):
        if sol is None or len(sol) == 0:
            continue
        s = sol.copy()
        if "fingerprint_hash" in s.columns and pd.notna(s["fingerprint_hash"].iloc[0]):
            h = str(s["fingerprint_hash"].iloc[0])
        else:
            h = fingerprint_hash(food_id_fingerprint(s["food_id"]))
            s["fingerprint_hash"] = h
        if h in seen:
            continue
        seen.add(h)
        s["seq_sol_index"] = len(out) + 1
        out.append(s)
    return out

# Target difficulty: nutrient base + optional structure bump (algo/bronze_difficulty)
from algo.bronze_difficulty import (  # noqa: E402
    classify_bronze_difficulty_label,
    classify_target_difficulty,
)


def find_food_from_target(food_df, target_dict, user_history=None, config=None, pool_seed_offset: int = 0):
    return filter_candidate_pool(
        food_df,
        target_dict,
        user_history_food_ids=user_history,
        config=config,
        pool_seed_offset=pool_seed_offset,
    )


def _parse_target(t):
    if isinstance(t, dict):
        return t
    return ast.literal_eval(str(t))


def _demo_from_row(r) -> dict:
    sex = sex_label(r.get("gender", r.get("sex", None)))
    bmi = None
    if pd.notna(r.get("bmi")):
        bmi = float(r["bmi"])
    elif "weight" in r and "height" in r and pd.notna(r["weight"]) and pd.notna(r["height"]):
        bmi = float(r["weight"]) / (float(r["height"]) / 100.0) ** 2
    age = float(r["age"]) if pd.notna(r.get("age")) else None
    return {"sex": sex, "bmi": bmi, "age": age}


def classify_target_difficulty_for_user(target_dict: dict, user_row=None) -> str:
    """Structure+target difficulty; nutrient-only if no demo row."""
    if user_row is None:
        return classify_target_difficulty(target_dict)
    demo = _demo_from_row(user_row)
    return classify_bronze_difficulty_label(
        target_dict, sex=demo["sex"], bmi=demo["bmi"], age=demo["age"]
    )


def _first_user_row(user: pd.DataFrame, uids: list):
    if not uids:
        return None
    u0 = user[user["user_id"] == uids[0]]
    return u0.iloc[0] if len(u0) else None


def _uids_for_target(target_dict, target_to_users: dict) -> list:
    t_str = str(target_dict)
    uids = target_to_users.get(t_str, [])
    if uids:
        return uids
    for k, v in target_to_users.items():
        try:
            if _parse_target(k) == target_dict:
                return v
        except Exception:
            continue
    return []


def difficulty_counts(
    target_dicts: list,
    *,
    user: pd.DataFrame | None = None,
    target_to_users: dict | None = None,
) -> dict[str, int]:
    from collections import Counter

    if user is None or target_to_users is None:
        return dict(Counter(classify_target_difficulty(td) for td in target_dicts))
    labels = []
    for td in target_dicts:
        row = _first_user_row(user, _uids_for_target(td, target_to_users))
        labels.append(classify_target_difficulty_for_user(td, row))
    return dict(Counter(labels))


def _target_hash(t_str: str) -> str:
    """Legacy nutrient-target-only hash (prefer ``_unit_hash``)."""
    return hashlib.sha1(str(t_str).encode("utf-8")).hexdigest()[:16]


def _unit_hash(structure_key: str, target_str: str) -> str:
    """Checkpoint identity for one (structure stratum × nutrient target) work unit."""
    raw = f"{structure_key}\n{target_str}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def user_stratum_key(row) -> str:
    """Person stratum: sex × BMI class × age band (StructureSchema default)."""
    from algo.structure_features import stratum_key

    demo = _demo_from_row(row)
    return stratum_key(
        pd.Series(
            {
                "sex": demo["sex"],
                "gender": demo["sex"],
                "bmi": demo["bmi"],
                "age": demo["age"],
            }
        )
    )


def build_structure_target_units(user: pd.DataFrame) -> list[dict]:
    """
    Unique work units: (structure_key, target_str) with adherent user_ids.

    Same nutrient target + different strata → separate units (separate GA runs).
    """
    from collections import OrderedDict

    groups: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for _, r in user.iterrows():
        if "target" not in r.index or pd.isna(r.get("target")):
            continue
        st = user_stratum_key(r)
        t_str = str(r["target"])
        uid = int(r["user_id"])
        key = (st, t_str)
        if key not in groups:
            groups[key] = []
        groups[key].append(uid)

    units: list[dict] = []
    for st, t_str in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        uids = list(dict.fromkeys(groups[(st, t_str)]))
        units.append(
            {
                "structure_key": st,
                "target_str": t_str,
                "target": _parse_target(t_str),
                "user_ids": uids,
            }
        )
    return units


def difficulty_counts_units(units: list[dict], user: pd.DataFrame) -> dict[str, int]:
    from collections import Counter

    labels = []
    for u in units:
        row = _first_user_row(user, u.get("user_ids") or [])
        labels.append(classify_target_difficulty_for_user(u["target"], row))
    return dict(Counter(labels))


def _with_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    return fh


def load_done_map(
    ckpt_dir: Path,
    min_solutions: int = 1,
    skip_not_found: bool = True,
) -> dict[str, dict]:
    """
    target_hash -> meta for targets that should be skipped on resume.
    - always skip ok with n_solutions >= min_solutions (complete ≥1 by default; GA aims ≤7)
    - skip not_found only if skip_not_found=True
    - incomplete is never skipped (will be retried)
    """
    path = ckpt_dir / "done_targets.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        h = str(r["target_hash"])
        status = str(r.get("status", ""))
        n = int(r.get("n_solutions", 0))
        if status == "ok" and n >= min_solutions:
            out[h] = r.to_dict()
        elif status == "not_found" and skip_not_found:
            out[h] = r.to_dict()
    return out


def list_failed_target_strings(ckpt_dir: Path, min_solutions: int = 1) -> list[str]:
    """Legacy: failed nutrient target strings (prefer ``list_failed_unit_hashes``)."""
    path = ckpt_dir / "done_targets.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    failed = []
    for _, r in df.iterrows():
        status = str(r.get("status", ""))
        n = int(r.get("n_solutions", 0))
        if status in ("incomplete", "not_found") or (status == "ok" and n < min_solutions):
            failed.append(str(r["target"]))
    seen = set()
    out = []
    for t in failed:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def list_failed_unit_hashes(ckpt_dir: Path, min_solutions: int = 1) -> set[str]:
    """incomplete + not_found (+ ok with too few sols) unit hashes for retry."""
    path = ckpt_dir / "done_targets.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    out: set[str] = set()
    for _, r in df.iterrows():
        status = str(r.get("status", ""))
        n = int(r.get("n_solutions", 0))
        if status in ("incomplete", "not_found") or (status == "ok" and n < min_solutions):
            out.add(str(r["target_hash"]))
    return out


def apply_hard_retry_config(cfg: dict) -> dict:
    """Stronger search for second-pass incomplete/not_found targets (legacy)."""
    cfg = {**cfg}
    cfg["archive_target"] = max(int(cfg.get("archive_target", 7)), 7)
    ad = dict(cfg.get("adaptive") or {})
    ad["enabled"] = True
    ad["no_feasible_escalate_gens"] = 60
    ad["stages"] = [
        {"name": "easy", "population_size": 1500, "generations": 100, "stop_at": 7},
        {"name": "medium", "population_size": 2000, "generations": 80, "stop_at": 7},
        {"name": "hard", "population_size": 4000, "generations": 150, "stop_at": 7},
    ]
    cfg["adaptive"] = ad
    pool = dict(cfg.get("pool") or {})
    pool["max_pool_size"] = max(int(pool.get("max_pool_size", 15000)), 20000)
    pool["nutrient_booster_k"] = max(int(pool.get("nutrient_booster_k", 120)), 200)
    cfg["pool"] = pool
    return cfg


def apply_escalation_pass(cfg: dict, pass_spec: dict) -> dict:
    """Force adaptive to a single stage matching one escalation pass."""
    cfg = {**cfg}
    p = int(pass_spec["population_size"])
    g = int(pass_spec["generations"])
    stop = int(pass_spec.get("stop_at", cfg.get("archive_target", 7)))
    name = str(pass_spec.get("name", f"p{p}_g{g}"))
    ad = dict(cfg.get("adaptive") or {})
    ad["enabled"] = True
    # single-stage → is_last always; full g used (no mid-stage early escalate)
    ad["no_feasible_escalate_gens"] = max(g, int(ad.get("no_feasible_escalate_gens", 60)))
    ad["stages"] = [
        {"name": name, "population_size": p, "generations": g, "stop_at": stop},
    ]
    cfg["adaptive"] = ad
    return cfg


def default_escalation_passes() -> list[dict]:
    return [
        {"name": "pass1_easy_p1000_g50", "difficulty": "easy", "population_size": 1000, "generations": 50, "stop_at": 7},
        {"name": "pass2_medium_p2000_g60_r1", "difficulty": "medium", "population_size": 2000, "generations": 60, "stop_at": 7},
        {"name": "pass2_medium_p2000_g60_r2", "difficulty": "medium", "population_size": 2000, "generations": 60, "stop_at": 7},
        {"name": "pass3_hard_p3000_g70_r1", "difficulty": "hard", "population_size": 3000, "generations": 70, "stop_at": 7},
        {"name": "pass3_hard_p3000_g70_r2", "difficulty": "hard", "population_size": 3000, "generations": 70, "stop_at": 7},
        {"name": "pass3_hard_p3000_g70_r3", "difficulty": "hard", "population_size": 3000, "generations": 70, "stop_at": 7},
        {"name": "pass4_extreme_p4000_g80_r1", "difficulty": "extreme", "population_size": 4000, "generations": 80, "stop_at": 7},
        {"name": "pass4_extreme_p4000_g80_r2", "difficulty": "extreme", "population_size": 4000, "generations": 80, "stop_at": 7},
        {"name": "pass4_extreme_p4000_g80_r3", "difficulty": "extreme", "population_size": 4000, "generations": 80, "stop_at": 7},
        {"name": "pass4_extreme_p4000_g80_r4", "difficulty": "extreme", "population_size": 4000, "generations": 80, "stop_at": 7},
        {"name": "pass_final_all_p5000_g100", "difficulty": "all", "population_size": 5000, "generations": 100, "stop_at": 7},
    ]


def append_stage_log(log_path: Path, rows: list[dict]) -> None:
    """Append per-stage (p,g,time,solutions) rows for SI / supplementary."""
    if not rows:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock = _with_lock(log_path.with_suffix(log_path.suffix + ".lock"))
    try:
        df_new = pd.DataFrame(rows)
        if log_path.exists():
            old = pd.read_csv(log_path)
            df = pd.concat([old, df_new], ignore_index=True)
        else:
            df = df_new
        _atomic_to_csv(df, log_path)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def count_checkpoint_status(ckpt_dir: Path) -> dict:
    path = ckpt_dir / "done_targets.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df["status"].value_counts().to_dict() if "status" in df.columns else {}


def append_done_row(ckpt_dir: Path, row: dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "done_targets.csv"
    lock = _with_lock(ckpt_dir / "done_targets.lock")
    try:
        df_new = pd.DataFrame([row])
        if path.exists():
            old = pd.read_csv(path)
            # replace same hash
            old = old[old["target_hash"].astype(str) != str(row["target_hash"])]
            df = pd.concat([old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(path, index=False)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV then fsync so a finished target is durable immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    with open(tmp, "rb") as fh:
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def save_target_checkpoint(
    ckpt_dir: Path,
    t_str: str,
    rows: pd.DataFrame,
    n_solutions: int,
    status: str,
    shard: str,
    live_csv=None,
    structure_key: str = "",
    unit_hash: str | None = None,
) -> Path:
    """Persist one structure×target unit as soon as it finishes."""
    h = unit_hash or (
        _unit_hash(structure_key, t_str) if structure_key else _target_hash(t_str)
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"target_{h}.csv"
    if rows is not None and len(rows):
        _atomic_to_csv(rows, path)
    append_done_row(
        ckpt_dir,
        {
            "target_hash": h,
            "structure_key": structure_key or "",
            "target": t_str,
            "n_solutions": int(n_solutions),
            "status": status,
            "path": str(path) if status in ("ok", "incomplete") and path.exists() else "",
            "shard": shard,
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    )
    if status == "ok" and live_csv is not None:
        merge_checkpoints(ckpt_dir, live_csv, min_solutions=1)
    print(
        f"  [disk] status={status} n_sol={n_solutions} structure={structure_key or '-'} "
        f"hash={h} -> {path.name}",
        flush=True,
    )
    return path


def merge_checkpoints(ckpt_dir: Path, out_file: Path, min_solutions: int = 1) -> pd.DataFrame:
    done = load_done_map(ckpt_dir, min_solutions=min_solutions, skip_not_found=False)
    frames = []
    for h, meta in done.items():
        if meta.get("status") != "ok":
            continue
        if int(meta.get("n_solutions", 0)) < min_solutions:
            continue
        p = Path(str(meta.get("path", "")))
        if not p.exists():
            p = ckpt_dir / f"target_{h}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    return df


def run_for_targets(
    food_df,
    user: pd.DataFrame,
    target_dicts: list | None = None,
    device: str = "cpu",
    population_size: int = 1000,
    generations: int = 50,
    crossover_rate: float = 0.2,
    mutate_rate: float = 0.9,
    config: dict | None = None,
    archive_target: int = 7,
    n_per_user: int = 7,
    ckpt_dir=None,
    shard_id: int = 0,
    shard_n: int = 1,
    min_solutions: int = 1,
    skip_not_found: bool = True,
    live_csv=None,
    stage_log_path=None,
    pass_name: str = "",
    difficulty_filter: str | None = None,
    units: list[dict] | None = None,
):
    """
    Run GA for structure×target work units.

    Prefer ``units`` from ``build_structure_target_units``. Legacy ``target_dicts``
    is expanded via the user table into units (structure from each adherent).
    """
    from algo.sequential_ban_plugin import (
        resolve_ban_plugin_name,
        sequential_search_enabled,
    )

    config = config or {}
    method = config.get("plan_generation_method", "BMOEA_gpu_ga_v2")
    if sequential_search_enabled(config):
        plugin_name = resolve_ban_plugin_name(config)
        method = config.get(
            "plan_generation_method",
            f"BMOEA_gpu_ga_v2_sequential_ban_{plugin_name}",
        )
    all_rows = []
    not_found = []
    shard_tag = f"{shard_id}/{shard_n}"

    user = user.copy()
    if units is None:
        if target_dicts is None:
            units = build_structure_target_units(user)
        else:
            # Restrict to requested targets, still split by structure.
            want = {str(td) for td in target_dicts}
            units = [
                u for u in build_structure_target_units(user) if u["target_str"] in want
            ]
            # Orphans (target with no user row): keep as structure="" units
            have = {u["target_str"] for u in units}
            for td in target_dicts:
                t_str = str(td)
                if t_str not in have:
                    units.append(
                        {
                            "structure_key": "",
                            "target_str": t_str,
                            "target": td if isinstance(td, dict) else _parse_target(td),
                            "user_ids": [],
                        }
                    )

    history_by_user = {}
    if "user_id" in food_df.columns:
        for uid, g in food_df.groupby("user_id"):
            history_by_user[int(uid)] = set(
                pd.to_numeric(g["food_id"], errors="coerce").dropna().astype(int)
            )

    pool_base = food_df.drop(columns=["user_id"], errors="ignore")
    done_map = (
        load_done_map(ckpt_dir, min_solutions=min_solutions, skip_not_found=skip_not_found)
        if ckpt_dir
        else {}
    )

    # shard by unit index; optionally keep one difficulty tier (structure+target)
    indexed = [(i, u) for i, u in enumerate(units) if (i % shard_n) == shard_id]
    if difficulty_filter:
        indexed = [
            (i, u)
            for i, u in indexed
            if classify_target_difficulty_for_user(
                u["target"], _first_user_row(user, u.get("user_ids") or [])
            )
            == difficulty_filter
        ]
        print(
            f"shard={shard_tag} difficulty={difficulty_filter} "
            f"assigned={len(indexed)}/{len(units)} units checkpoint_done={len(done_map)}",
            flush=True,
        )
    else:
        print(
            f"shard={shard_tag} assigned={len(indexed)}/{len(units)} units "
            f"checkpoint_done={len(done_map)}",
            flush=True,
        )

    for idx, unit in tqdm(indexed, desc=f"units[{shard_tag}]"):
        structure_key = str(unit.get("structure_key") or "")
        target_dict = unit["target"]
        t_str = str(unit.get("target_str") or target_dict)
        uids = list(unit.get("user_ids") or [])
        h = _unit_hash(structure_key, t_str)

        if h in done_map:
            meta = done_map[h]
            st = str(meta.get("status", ""))
            if st == "ok":
                p = Path(str(meta.get("path", ""))) or (ckpt_dir / f"target_{h}.csv")
                if p.exists():
                    all_rows.append(pd.read_csv(p))
                print(
                    f"[{idx}] resume skip ok structure={structure_key} hash={h}",
                    flush=True,
                )
                continue
            if st == "not_found" and skip_not_found:
                not_found.append(unit)
                print(
                    f"[{idx}] resume skip not_found structure={structure_key} hash={h}",
                    flush=True,
                )
                continue

        prior = load_structure_prior()
        hist = set()
        participant_profile: dict = {}
        personal_category_structure = None
        if uids:
            u0 = user[user["user_id"] == uids[0]]
            if len(u0):
                r = u0.iloc[0]
                demo = _demo_from_row(r)
                sex, bmi = demo["sex"], demo["bmi"]
                prior = load_structure_prior(sex=sex, bmi=bmi, age=demo["age"], config=config)
                participant_profile = {
                    "sex": sex,
                    "gender": r.get("gender", r.get("sex")),
                    "bmi": bmi,
                    "age": demo["age"],
                    "user_id": int(uids[0]),
                }
                try:
                    from algo.structure_category import (
                        format_personal_structure_brief,
                        select_personal_category_for_bronze,
                    )

                    personal_category_structure = select_personal_category_for_bronze(
                        participant_profile, target_dict
                    )
                    if personal_category_structure is not None:
                        print(
                            f"[{idx}] pre-GA structure={structure_key} "
                            + format_personal_structure_brief(personal_category_structure).replace(
                                "\n", " | "
                            ),
                            flush=True,
                        )
                except Exception as e:
                    print(f"[{idx}] personal category structure skip: {e}", flush=True)
                    personal_category_structure = None
            for uid in uids:
                hist |= history_by_user.get(int(uid), set())

        nak_cfg = dict((config.get("pool") or {}).get("low_na_high_k") or {})
        max_pool_tries = 1 + (
            int(nak_cfg.get("pool_retries", 0))
            if target_needs_low_na_high_k(target_dict)
            else 0
        )
        seed_sols = load_unique_solutions_from_checkpoint(ckpt_dir, h)
        if seed_sols:
            print(
                f"[{idx}] resume accumulate seed={len(seed_sols)} sols from checkpoint",
                flush=True,
            )
        solutions = None
        n_sol = 0
        stage_rows_acc = []
        for pool_try in range(max_pool_tries):
            found = find_food_from_target(
                pool_base,
                target_dict,
                user_history=hist,
                config=config,
                pool_seed_offset=pool_try,
            )
            if found.empty:
                print(f"[{idx}] empty pool for target (try={pool_try})", flush=True)
                continue
            meals = found["meal"].value_counts().to_dict() if "meal" in found.columns else {}
            print(f"[{idx}] pool={len(found)} meals={meals} try={pool_try}", flush=True)
            if found["meal"].isin(["lunch", "dinner"]).sum() == 0:
                print(f"[{idx}] pool missing lunch/dinner — try={pool_try}", flush=True)
                continue
            try:
                ad = config.get("adaptive") or {}
                stages = ad.get("stages") or []
                p0 = int(stages[0]["population_size"]) if stages else min(population_size, 1000)
                need_n = max(n_per_user, int(config.get("n_solutions_per_user", 7)))
                combiner = FoodCombinerV2(
                    found,
                    target_dict,
                    max_combination_size=int(config.get("max_combination_size", 30)),
                    population_size=p0,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutate_rate,
                    device=device,
                    debug=False,
                    structure_prior=prior,
                    config=config,
                    n_solutions=need_n,
                    archive_target=archive_target,
                    require_bld=False,
                    participant_profile=participant_profile,
                    personal_category_structure=personal_category_structure,
                )
                t_tgt0 = time.perf_counter()
                use_seq = sequential_search_enabled(config)
                if use_seq:
                    solutions = combiner.optimize_sequential_ban_mains(
                        target_idx=idx,
                        batch_index=shard_id,
                        crossover_rate=crossover_rate,
                        mutation_rate=mutate_rate,
                        generations_per_sol=generations
                        if not (ad.get("stages") or [])
                        else None,
                        population_size=population_size
                        if not (ad.get("stages") or [])
                        else None,
                        seed_solutions=seed_sols,
                    )
                elif ad.get("enabled", True):
                    solutions = combiner.optimize_adaptive(
                        target_idx=idx,
                        batch_index=shard_id,
                        crossover_rate=crossover_rate,
                        mutation_rate=mutate_rate,
                    )
                    # Non-sequential: still union with prior incomplete sols
                    solutions = merge_solution_lists(seed_sols, solutions)[:need_n]
                else:
                    solutions = combiner.optimize(
                        target_idx=idx, batch_index=shard_id, generations=generations
                    )
                    solutions = merge_solution_lists(seed_sols, solutions)[:need_n]
                tgt_elapsed = time.perf_counter() - t_tgt0
                for sl in getattr(combiner, "last_stage_log", None) or []:
                    stage_rows_acc.append(
                        {
                            "pass": pass_name or "",
                            "target_hash": h,
                            "structure_key": structure_key,
                            "target_idx": idx,
                            "shard": shard_tag,
                            "pool_try": pool_try,
                            "stage": sl.get("stage", ""),
                            "p": sl.get("p"),
                            "g": sl.get("g"),
                            "time_s": sl.get("time_s"),
                            "n_solutions": sl.get("n_solutions"),
                            "run_status": sl.get("run_status", ""),
                            "target_wall_s": round(tgt_elapsed, 3),
                            "logged_at": datetime.datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                if not getattr(combiner, "last_stage_log", None):
                    stage_rows_acc.append(
                        {
                            "pass": pass_name or "",
                            "target_hash": h,
                            "structure_key": structure_key,
                            "target_idx": idx,
                            "shard": shard_tag,
                            "pool_try": pool_try,
                            "stage": "legacy",
                            "p": p0,
                            "g": generations,
                            "time_s": round(tgt_elapsed, 3),
                            "n_solutions": len(solutions) if solutions else 0,
                            "run_status": "ok" if solutions else "empty",
                            "target_wall_s": round(tgt_elapsed, 3),
                            "logged_at": datetime.datetime.now().isoformat(timespec="seconds"),
                        }
                    )
            except Exception as e:
                print(f"[{idx}] optimize failed (try={pool_try}): {e}", flush=True)
                # Keep prior incomplete on failure — do not wipe checkpoint
                solutions = list(seed_sols) if seed_sols else None
                # CUDA device-side assert poisons the context — reset before next target
                if "cuda" in str(device).lower() or "CUDA" in str(e):
                    try:
                        import torch

                        torch.cuda.empty_cache()
                        if ":" in str(device):
                            torch.cuda.set_device(int(str(device).split(":")[-1]))
                        # Synchronize & clear error state when possible
                        torch.cuda.synchronize()
                    except Exception as ce:
                        print(f"[{idx}] cuda reset note: {ce}", flush=True)
                continue
            # Always accumulate vs seed (safety if optimizer returned only new)
            solutions = merge_solution_lists(seed_sols, solutions)
            n_sol = len(solutions) if solutions else 0
            if n_sol >= min_solutions:
                break
            if n_sol > 0:
                # partial — keep; further pool retries unlikely to be required for incomplete
                break
            if pool_try + 1 < max_pool_tries:
                print(
                    f"[{idx}] archive=0 — rebuild pool (offset={pool_try + 1}) and retry",
                    flush=True,
                )

        if stage_log_path and stage_rows_acc:
            append_stage_log(Path(stage_log_path), stage_rows_acc)

        # Final accumulate: never write fewer unique sols than checkpoint already had
        solutions = merge_solution_lists(seed_sols, solutions)
        n_sol = len(solutions) if solutions else 0
        if n_sol < min_solutions:
            print(
                f"[{idx}] insufficient solutions {n_sol}<{min_solutions} "
                f"(seed={len(seed_sols)}) — will not mark ok",
                flush=True,
            )
            if n_sol == 0:
                not_found.append(unit)
                if ckpt_dir and not seed_sols:
                    save_target_checkpoint(
                        ckpt_dir,
                        t_str,
                        pd.DataFrame(),
                        0,
                        "not_found",
                        shard_tag,
                        live_csv=live_csv,
                        structure_key=structure_key,
                        unit_hash=h,
                    )
                elif ckpt_dir and seed_sols:
                    print(f"[{idx}] keep prior incomplete seed={len(seed_sols)}", flush=True)
            else:
                assign_uids = uids if uids else [None]
                assigned = assign_solutions_to_users(solutions, assign_uids, n_per_user=n_per_user)
                frames = []
                for sol in assigned:
                    s = sol.copy()
                    s["years"] = 2025
                    s["target"] = t_str
                    s["structure_key"] = structure_key
                    s["plan_generation_method"] = method
                    frames.append(s)
                part = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                if ckpt_dir:
                    save_target_checkpoint(
                        ckpt_dir,
                        t_str,
                        part,
                        n_sol,
                        "incomplete",
                        shard_tag,
                        live_csv=live_csv,
                        structure_key=structure_key,
                        unit_hash=h,
                    )
            continue

        assign_uids = uids if uids else [None]
        assigned = assign_solutions_to_users(solutions, assign_uids, n_per_user=n_per_user)
        frames = []
        for sol in assigned:
            s = sol.copy()
            s["years"] = 2025
            s["target"] = t_str
            s["structure_key"] = structure_key
            s["plan_generation_method"] = method
            frames.append(s)
            all_rows.append(s)
        part = pd.concat(frames, ignore_index=True)
        if ckpt_dir:
            # Complete when n_sol >= min_solutions (default 1). Search stops at ≤7.
            save_target_checkpoint(
                ckpt_dir,
                t_str,
                part,
                n_sol,
                "ok",
                shard_tag,
                live_csv=live_csv,
                structure_key=structure_key,
                unit_hash=h,
            )

    if not all_rows:
        return pd.DataFrame(), not_found
    return pd.concat(all_rows, ignore_index=True), not_found


def _file_stamp(p: Path) -> tuple:
    if not p.exists():
        return (str(p), 0, 0)
    st = p.stat()
    return (str(p), int(st.st_mtime_ns), int(st.st_size))


def _pool_cache_key(data_path: Path) -> str:
    from algo.bronze_food_filter import load_bronze_exclusion_config
    from algo.food_meal_tags import load_meal_tag_config

    excl = load_bronze_exclusion_config()
    tag_cfg = load_meal_tag_config()
    tag_csv = ROOT / str((tag_cfg.get("paths") or {}).get("out_csv", "data/nhanes/data/food_meal_tags.csv"))
    parts = [
        "bronze_pool_v4_operable_meal_pools",
        str(excl.get("version", "")),
        str(tag_cfg.get("version", "")),
        _file_stamp(data_path / "food_user.csv"),
        _file_stamp(data_path / "food_nutrition.csv"),
        _file_stamp(data_path / "food_code.csv"),
        _file_stamp(ROOT / "config" / "bronze_food_exclusions.yaml"),
        _file_stamp(ROOT / "config" / "food_meal_tags.yaml"),
        _file_stamp(ROOT / "config" / "golden_week.yaml"),
        _file_stamp(ROOT / "data" / "portion_units" / "food_display_unit.csv"),
        _file_stamp(ROOT / "algo" / "operable_units.py"),
        _file_stamp(tag_csv),
    ]
    raw = repr(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _pool_cache_paths(data_path: Path, key: str) -> tuple[Path, Path]:
    cache_dir = data_path / "_cache"
    # Prefer parquet when engine available; else pickle
    try:
        import pyarrow  # noqa: F401

        return cache_dir / f"bronze_food_pool_{key}.parquet", cache_dir / f"bronze_food_pool_{key}.meta.json"
    except Exception:
        try:
            import fastparquet  # noqa: F401

            return cache_dir / f"bronze_food_pool_{key}.parquet", cache_dir / f"bronze_food_pool_{key}.meta.json"
        except Exception:
            return cache_dir / f"bronze_food_pool_{key}.pkl", cache_dir / f"bronze_food_pool_{key}.meta.json"


def _read_pool_cache(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def _write_pool_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)


def load_data(path: str, rebuild_pool_cache: bool = False):
    path = Path(path)
    user = pd.read_csv(path / "not_found_match_food_user.csv")
    user["target"] = user.apply(get_nutrition_target_range_by_situation, axis=1)

    cache_key = _pool_cache_key(path)
    cache_pq, cache_meta = _pool_cache_paths(path, cache_key)
    if not rebuild_pool_cache and cache_pq.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            if meta.get("key") == cache_key:
                t0 = datetime.datetime.now()
                food_user = _read_pool_cache(cache_pq)
                print(
                    f"bronze pool cache hit: {cache_pq.name} rows={len(food_user)} "
                    f"foods={food_user['food_id'].nunique()} "
                    f"({(datetime.datetime.now() - t0).total_seconds():.1f}s)",
                    flush=True,
                )
                not_found_target_file = path / "not_found_targets.csv"
                if not_found_target_file.exists():
                    unique_targets = pd.read_csv(not_found_target_file)["target"].values
                else:
                    unique_targets = user["target"].astype(str).unique()
                return food_user, user, unique_targets
        except Exception as e:
            print(f"bronze pool cache unusable ({e}); rebuilding…", flush=True)

    food = pd.read_csv(path / "food_code.csv")
    food_nutrition = pd.read_csv(path / "food_nutrition.csv")
    food_nutrition = pd.merge(
        food_nutrition, food[["food_id", "positive", "negative"]], on=["food_id"], how="left"
    )
    from algo.diet_structure_priors import SPANISH_TO_ENGLISH
    from algo.operable_food_user import (
        ensure_canonical_symlink,
        is_operable_frame,
        load_operable_food_user,
        resolve_operable_food_user_path,
    )
    from algo.operable_units import load_display_unit_table

    # Prefer one-shot operable food_user (same artifact as keep-mains / review ①)
    food_user = None
    try:
        ensure_canonical_symlink()
        op_path = resolve_operable_food_user_path()
        food_user = load_operable_food_user(op_path)
        print(
            f"operable food_user cache: {op_path.name} rows={len(food_user)} "
            f"pre_snapped={is_operable_frame(food_user)}",
            flush=True,
        )
    except FileNotFoundError:
        food_user = None

    if food_user is None or not is_operable_frame(food_user):
        food_user = pd.read_csv(path / "food_user.csv")
        food_user = pd.merge(food_user, food_nutrition, on=["food_id", "grams"], how="left")
        food_user["eating_type"] = pd.to_numeric(food_user["eating_type"], errors="coerce")
        food_user["eating_type"] = food_user["eating_type"].map(
            lambda x: SPANISH_TO_ENGLISH.get(int(x), int(x)) if pd.notna(x) else x
        )
        food_user = food_user[food_user["eating_type"] < 8]
        food_user = food_user.drop_duplicates()
        # Keep NHANES grams; FNDDS serves attached in canonicalize_operable_pool
        food_user["grams_raw"] = pd.to_numeric(food_user["grams"], errors="coerce").fillna(0.0)
        print(
            f"food_user NHANES grams (no snap): rows={len(food_user)}",
            flush=True,
        )
    else:
        food_user["eating_type"] = pd.to_numeric(food_user["eating_type"], errors="coerce")
        if food_user["eating_type"].notna().any():
            food_user["eating_type"] = food_user["eating_type"].map(
                lambda x: SPANISH_TO_ENGLISH.get(int(x), int(x)) if pd.notna(x) else x
            )
            food_user = food_user[food_user["eating_type"] < 8]
        # Attach positive/negative if missing
        if "positive" not in food_user.columns:
            food_user = pd.merge(
                food_user,
                food[["food_id", "positive", "negative"]],
                on="food_id",
                how="left",
            )
        food_user = food_user.drop_duplicates()

    # Bronze: NHANES grams + FNDDS serve labels (no 化零为整)
    from algo.operable_units import canonicalize_operable_pool, split_meal_pools

    n_op0 = len(food_user)
    food_user = canonicalize_operable_pool(
        food_user,
        display_table=load_display_unit_table(),
        drop_over_max_qty=False,
    )
    print(
        f"operable canonicalize: rows {n_op0}->{len(food_user)} "
        f"(grams=NHANES; FNDDS serve_qty attached; no mass snap)",
        flush=True,
    )

    food_user = enrich_food_dataframe(food_user)

    from algo.bronze_food_filter import filter_bronze_food_df
    from algo.food_meal_tags import (
        build_food_meal_tags,
        expand_rows_to_tagged_meals,
        load_meal_tag_config,
        tags_dict_from_frame,
    )

    desc = dict(zip(food.food_id.astype(int), food.food_desc.astype(str)))

    tag_cfg = load_meal_tag_config()
    tag_path = ROOT / str((tag_cfg.get("paths") or {}).get("out_csv", "data/nhanes/data/food_meal_tags.csv"))
    if tag_path.exists():
        tag_df = pd.read_csv(tag_path)
        tags = tags_dict_from_frame(tag_df)
        print(f"meal tags loaded: {tag_path} foods={len(tags)}", flush=True)
    else:
        print("meal tags CSV missing — building on the fly…", flush=True)
        tag_df = build_food_meal_tags(food_user, desc, tag_cfg)
        tag_path.parent.mkdir(parents=True, exist_ok=True)
        tag_df.to_csv(tag_path, index=False)
        tags = tags_dict_from_frame(tag_df)
        print(f"meal tags wrote: {tag_path} foods={len(tags)}", flush=True)

    n_before = len(food_user)
    u_before = int(food_user["food_id"].nunique())
    food_user, audit = filter_bronze_food_df(food_user, food_desc=desc)
    print(
        f"bronze food filter: rows {n_before}->{audit['n_out']} "
        f"unique_foods {u_before}->{audit['n_unique_food_out']} "
        f"(kw_drop={audit['n_dropped_keyword']} "
        f"oats_drop={audit.get('n_dropped_oats_non_breakfast', 0)} "
        f"ld_cereal_drop={audit.get('n_dropped_lunch_dinner_cereal', 0)} "
        f"fped_drop={audit['n_dropped_no_fped']}) "
        f"ver={audit.get('version')}",
        flush=True,
    )

    keep_ids = set(pd.to_numeric(food_user["food_id"], errors="coerce").dropna().astype(int))
    tags = {k: v for k, v in tags.items() if k in keep_ids}
    n_pre = len(food_user)
    food_user = expand_rows_to_tagged_meals(food_user, tags)
    food_user["meal_tags"] = pd.to_numeric(food_user["food_id"], errors="coerce").map(
        lambda x: "|".join(m for m in ("breakfast", "lunch", "dinner", "snack") if m in tags.get(int(x), set()))
        if pd.notna(x)
        else ""
    )
    # Meal-tag expand can reattach snack-only / breakfast-only foods onto wrong meals —
    # re-apply plate rules so B/L/D/snack pools stay clean.
    n_re = len(food_user)
    food_user, audit2 = filter_bronze_food_df(food_user, food_desc=desc)
    print(
        f"meal-slot re-filter after expand: rows {n_re}->{audit2['n_out']} "
        f"(slot_drop={audit2.get('n_dropped_lunch_dinner_cereal', 0)} "
        f"oats_drop={audit2.get('n_dropped_oats_non_breakfast', 0)})",
        flush=True,
    )
    # Re-attach FNDDS serves after meal expand (mass unchanged)
    food_user = canonicalize_operable_pool(
        food_user,
        display_table=load_display_unit_table(),
        drop_over_max_qty=False,
    )
    meal_pools = split_meal_pools(food_user)
    by_meal = {m: len(meal_pools[m]) for m in ("breakfast", "lunch", "dinner", "snack")}
    print(
        f"meal-tag expand: rows {n_pre}->{len(food_user)} "
        f"unique_foods={food_user['food_id'].nunique()} by_meal={by_meal}",
        flush=True,
    )
    print(
        "meal pools (explicit): "
        + ", ".join(
            f"{m}={len(meal_pools[m])}rows/"
            f"{meal_pools[m]['food_id'].nunique() if len(meal_pools[m]) else 0}foods"
            for m in ("breakfast", "lunch", "dinner", "snack")
        ),
        flush=True,
    )

    try:
        cache_pq.parent.mkdir(parents=True, exist_ok=True)
        _write_pool_cache(food_user, cache_pq)
        cache_meta.write_text(
            json.dumps(
                {
                    "key": cache_key,
                    "rows": int(len(food_user)),
                    "n_foods": int(food_user["food_id"].nunique()),
                    "created": datetime.datetime.now().isoformat(timespec="seconds"),
                    "format": cache_pq.suffix,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"bronze pool cache wrote: {cache_pq}", flush=True)
    except Exception as e:
        print(f"bronze pool cache write skipped: {e}", flush=True)

    not_found_target_file = path / "not_found_targets.csv"
    if not_found_target_file.exists():
        unique_targets_df = pd.read_csv(not_found_target_file)
        unique_targets = unique_targets_df["target"].values
    else:
        unique_targets = user["target"].astype(str).unique()
    return food_user, user, unique_targets


def main():
    parser = argparse.ArgumentParser(description="GPU-GA v2 multi-objective bronze generation")
    parser.add_argument("--data", type=str, default="data/nhanes/data")
    parser.add_argument("-d", type=int, default=0, help="GPU id")
    parser.add_argument("-p", type=int, default=1000, help="legacy single-stage population (if adaptive off)")
    parser.add_argument("-g", type=int, default=100, help="legacy single-stage generations (if adaptive off)")
    parser.add_argument("-c", type=float, default=0.2, help="crossover rate")
    parser.add_argument("-m", type=float, default=0.9, help="mutation rate")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument(
        "--limit-targets",
        type=int,
        default=0,
        help="debug: only first N structure×target units",
    )
    parser.add_argument("--limit-users", type=int, default=0, help="debug: subset users")
    parser.add_argument("--out", type=str, default="", help="output directory (default: data path)")
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help=(
            "ignore not_found_targets.csv; build structure×target work units "
            "from all users (full run)"
        ),
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="disable staged p/g; use -p/-g only",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default="0/1",
        help="GPU shard as i/n (e.g. 0/2 and 1/2 for dual-GPU)",
    )
    parser.add_argument(
        "--min-solutions",
        type=int,
        default=None,
        help=(
            "mark participant+target complete (status=ok) when n_solutions >= this "
            "(default: config min_solutions_complete=1; ≥1 after 50g counts). "
            "GA search stops at archive_target / n_solutions_per_user (default 7)."
        ),
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="only merge checkpoints into final CSV and exit",
    )
    parser.add_argument(
        "--retry-not-found",
        action="store_true",
        help="do not skip checkpointed not_found targets",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="second pass: only incomplete + not_found targets; stronger search; implies --retry-not-found",
    )
    parser.add_argument(
        "--escalation-ladder",
        action="store_true",
        help=(
            "run adaptive.escalation_passes by difficulty: "
            "easy→pass1(1000/50), medium→pass2(2000/60×2), hard→pass3(3000/70×3), "
            "extreme→pass4(4000/80×4), then final all remaining incomplete+not_found at 5000/100; "
            "each pass only runs its difficulty tier; logs p,g,time,n_sol to stage_run_log.csv; "
            "implies --retry-not-found"
        ),
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="",
        help="with --escalation-ladder: only run passes whose difficulty matches "
        "(easy|medium|hard|extreme|all)",
    )
    parser.add_argument(
        "--rebuild-pool-cache",
        action="store_true",
        help="rebuild filtered/expanded bronze food pool cache under data/_cache",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "find solutions one-by-one (sequential search). "
            "Pool ban after each sol is controlled by --ban-plugin (default: none)"
        ),
    )
    parser.add_argument(
        "--sequential-ban",
        action="store_true",
        help=(
            "shorthand: --sequential --ban-plugin fndds "
            "(N+1 ban day food_ids / FNDDS policy)"
        ),
    )
    parser.add_argument(
        "--ban-plugin",
        type=str,
        default="",
        choices=["", "none", "fndds"],
        help=(
            "N+1 pool-ban plugin for sequential search ablation: "
            "none (no shrink) | fndds (day ids / FNDDS policy). "
            "Implies --sequential when set."
        ),
    )
    parser.add_argument(
        "--users-csv",
        type=str,
        default="",
        help="restrict to user_id (+ optional target) rows from this CSV "
        "(e.g. outputs/ga_v2_plate_silver/not_found_or_lt7_users.csv)",
    )
    args = parser.parse_args()

    cfg = load_ga_config()
    if args.min_solutions is None:
        args.min_solutions = int(cfg.get("min_solutions_complete", 1))
    print(
        f"complete threshold min_solutions={args.min_solutions} "
        f"(GA archive_target={cfg.get('archive_target', 7)} "
        f"n_solutions_per_user={cfg.get('n_solutions_per_user', 7)})",
        flush=True,
    )
    if args.no_adaptive:
        cfg = {**cfg, "adaptive": {**(cfg.get("adaptive") or {}), "enabled": False}}
    if args.retry_failed and not args.escalation_ladder:
        cfg = apply_hard_retry_config(cfg)
        args.retry_not_found = True
        print("hard-retry config enabled for incomplete/not_found pass", flush=True)
    if args.escalation_ladder:
        args.retry_not_found = True
        print("escalation-ladder enabled (difficulty-stratified passes)", flush=True)
    ban_plugin = (args.ban_plugin or "").strip().lower()
    use_sequential = bool(args.sequential or args.sequential_ban or ban_plugin)
    if args.sequential_ban and not ban_plugin:
        ban_plugin = "fndds"
    if use_sequential and not ban_plugin:
        ban_plugin = str((cfg.get("sequential_ban") or {}).get("plugin") or "none")
    if use_sequential:
        cfg = {
            **cfg,
            "sequential_ban_mains": ban_plugin == "fndds",
            "sequential_ban": {
                **(cfg.get("sequential_ban") or {}),
                "enabled": True,
                "plugin": ban_plugin or "none",
            },
            "plan_generation_method": f"BMOEA_gpu_ga_v2_sequential_ban_{ban_plugin or 'none'}",
        }
        print(
            f"sequential search enabled ban_plugin={ban_plugin or 'none'}",
            flush=True,
        )
    device = "cpu" if args.cpu else f"cuda:{args.d}"
    out_path = Path(args.out) if args.out else Path(args.data)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stage_log_path = out_path / "stage_run_log.csv"

    shard_id, shard_n = 0, 1
    if "/" in args.shard:
        a, b = args.shard.split("/", 1)
        shard_id, shard_n = int(a), int(b)
        if shard_n < 1 or shard_id < 0 or shard_id >= shard_n:
            raise SystemExit(f"invalid --shard {args.shard}")

    out_file = out_path / "_daily_food_with_nutrition_target_bronze_v2.csv"
    if args.merge_only:
        df = merge_checkpoints(ckpt_dir, out_file, min_solutions=args.min_solutions)
        print(f"Merged rows={len(df)} -> {out_file}")
        return

    start = datetime.datetime.now()
    print(f"Start {start} device={device} shard={shard_id}/{shard_n}", flush=True)
    food_user, user, not_found_target = load_data(
        args.data, rebuild_pool_cache=args.rebuild_pool_cache
    )
    if args.users_csv:
        keep = pd.read_csv(args.users_csv, low_memory=False)
        if "user_id" not in keep.columns:
            raise SystemExit(f"--users-csv missing user_id: {args.users_csv}")
        keep_ids = set(pd.to_numeric(keep["user_id"], errors="coerce").dropna().astype(int))
        before_u = len(user)
        user = user[user["user_id"].isin(keep_ids)].copy()
        if "target" in keep.columns and keep["target"].notna().any():
            # Prefer targets from the CSV (raw-cohort personalized).
            # Keep demo cols for structure+target difficulty / category prior.
            demo_cols = [
                c
                for c in ("age", "gender", "sex", "bmi", "weight", "height")
                if c in keep.columns
            ]
            cols = ["user_id", "target"] + demo_cols
            user = keep[cols].drop_duplicates(subset=["user_id"]).copy()
            user["user_id"] = pd.to_numeric(user["user_id"], errors="coerce")
            user = user.dropna(subset=["user_id", "target"])
            user["user_id"] = user["user_id"].astype(int)
        not_found_target = user["target"].astype(str).unique()
        print(
            f"--users-csv {args.users_csv}: users {before_u}->{len(user)} "
            f"targets={len(not_found_target)}",
            flush=True,
        )
    if args.all_targets:
        not_found_target = user["target"].astype(str).unique()
        print(
            f"--all-targets: {len(not_found_target)} unique nutrient targets from users "
            f"(work units = structure×target, built next)",
            flush=True,
        )
    if args.limit_users > 0:
        keep = set(user["user_id"].unique()[: args.limit_users])
        user = user[user["user_id"].isin(keep)].copy()
        not_found_target = user["target"].astype(str).unique()

    want_targets = {str(t) for t in not_found_target}
    units = [
        u for u in build_structure_target_units(user) if u["target_str"] in want_targets
    ]
    n_targets = len(want_targets)
    n_strata = len({u["structure_key"] for u in units})
    print(
        f"users={len(user)} units={len(units)} (targets={n_targets}, strata={n_strata}) "
        f"food_rows={len(food_user)}",
        flush=True,
    )
    print(
        f"unit difficulty counts (structure+target): {difficulty_counts_units(units, user)}",
        flush=True,
    )
    print(
        f"target difficulty counts (nutrient-only): "
        f"{difficulty_counts([u['target'] for u in units])}",
        flush=True,
    )
    if args.retry_failed and not args.escalation_ladder:
        failed_hashes = list_failed_unit_hashes(ckpt_dir, min_solutions=args.min_solutions)
        before = len(units)
        units = [
            u
            for u in units
            if _unit_hash(u["structure_key"], u["target_str"]) in failed_hashes
        ]
        print(
            f"--retry-failed: {len(units)}/{before} units (incomplete+not_found)",
            flush=True,
        )
        if not units:
            print("nothing to retry; merging checkpoints only", flush=True)
            merge_checkpoints(ckpt_dir, out_file, min_solutions=args.min_solutions)
            return
    if args.limit_targets > 0:
        units = units[: args.limit_targets]

    # persist run meta
    (ckpt_dir / f"run_meta_shard{shard_id}.json").write_text(
        json.dumps(
            {
                "device": device,
                "shard": f"{shard_id}/{shard_n}",
                "min_solutions": args.min_solutions,
                "n_units": len(units),
                "n_targets": n_targets,
                "n_strata": n_strata,
                "escalation_ladder": bool(args.escalation_ladder),
                "started": start.isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )

    def _one_run(run_cfg: dict, pass_name: str = "", difficulty_filter: str | None = None):
        return run_for_targets(
            food_user,
            user,
            units=units,
            device=device,
            population_size=args.p,
            generations=args.g,
            crossover_rate=args.c,
            mutate_rate=args.m,
            config=run_cfg,
            archive_target=int(run_cfg.get("archive_target", 7)),
            n_per_user=int(run_cfg.get("n_solutions_per_user", 7)),
            ckpt_dir=ckpt_dir,
            shard_id=shard_id,
            shard_n=shard_n,
            min_solutions=args.min_solutions,
            skip_not_found=not args.retry_not_found,
            live_csv=out_file,
            stage_log_path=stage_log_path,
            pass_name=pass_name,
            difficulty_filter=difficulty_filter,
        )

    bronze_df, missing = pd.DataFrame(), []
    if args.escalation_ladder:
        passes = list((cfg.get("adaptive") or {}).get("escalation_passes") or []) or default_escalation_passes()
        if args.difficulty:
            want = str(args.difficulty).strip().lower()
            before_n = len(passes)
            passes = [
                p
                for p in passes
                if str(p.get("difficulty") or "").strip().lower() == want
            ]
            print(
                f"--difficulty={want}: {len(passes)}/{before_n} escalation passes",
                flush=True,
            )
            if not passes:
                raise SystemExit(f"no escalation passes for difficulty={want}")
        pi = 0
        while pi < len(passes):
            pass_spec = passes[pi]
            pname = str(pass_spec.get("name", f"pass{pi}"))
            p = int(pass_spec["population_size"])
            g = int(pass_spec["generations"])
            diff = pass_spec.get("difficulty")
            diff = str(diff) if diff else None
            if diff in ("all", "any", "*", "remaining"):
                diff = None  # retry every non-ok target (incomplete + not_found)
            before = count_checkpoint_status(ckpt_dir)
            print(
                f"=== escalation {pi + 1}/{len(passes)} {pname} "
                f"difficulty={diff or 'all'} p={p} g={g} "
                f"checkpoint_before={before} ===",
                flush=True,
            )
            pass_cfg = apply_escalation_pass(cfg, pass_spec)
            t_pass = time.perf_counter()
            bronze_df, missing = _one_run(pass_cfg, pass_name=pname, difficulty_filter=diff)
            after = count_checkpoint_status(ckpt_dir)
            print(
                f"=== end {pname} wall={time.perf_counter() - t_pass:.1f}s "
                f"checkpoint_after={after} stage_log={stage_log_path} ===",
                flush=True,
            )
            done_df_path = ckpt_dir / "done_targets.csv"
            if done_df_path.exists() and diff:
                ddf = pd.read_csv(done_df_path)
                ok_h = set(ddf.loc[ddf["status"] == "ok", "target_hash"].astype(str))
                shard_need = [
                    _unit_hash(u["structure_key"], u["target_str"])
                    for i, u in enumerate(units)
                    if (i % shard_n) == shard_id
                    and classify_target_difficulty_for_user(
                        u["target"], _first_user_row(user, u.get("user_ids") or [])
                    )
                    == diff
                ]
                if shard_need and all(h in ok_h for h in shard_need):
                    print(
                        f"shard {shard_id}/{shard_n}: all {diff} units ok — "
                        f"skip remaining rungs of this difficulty",
                        flush=True,
                    )
                    while (
                        pi + 1 < len(passes)
                        and str(passes[pi + 1].get("difficulty") or "") == diff
                    ):
                        pi += 1
                        print(
                            f"  skip {passes[pi].get('name')} (tier already complete)",
                            flush=True,
                        )
            elif done_df_path.exists():
                ddf = pd.read_csv(done_df_path)
                ok_h = set(ddf.loc[ddf["status"] == "ok", "target_hash"].astype(str))
                shard_need = [
                    _unit_hash(u["structure_key"], u["target_str"])
                    for i, u in enumerate(units)
                    if (i % shard_n) == shard_id
                ]
                if shard_need and all(h in ok_h for h in shard_need):
                    print(
                        f"shard {shard_id}/{shard_n}: all assigned units ok — stop ladder",
                        flush=True,
                    )
                    break
            pi += 1
    else:
        bronze_df, missing = _one_run(cfg, pass_name="")

    # merge all shards' checkpoints into final artifact
    merged = merge_checkpoints(ckpt_dir, out_file, min_solutions=args.min_solutions)
    if merged is not None and len(merged):
        print(
            f"Wrote {out_file} rows={len(merged)} users="
            f"{merged['user_id'].nunique() if 'user_id' in merged.columns else '?'}",
            flush=True,
        )
    elif bronze_df is not None and len(bronze_df):
        bronze_df.to_csv(out_file, index=False)
        print(f"Wrote {out_file} rows={len(bronze_df)}", flush=True)
    else:
        print("No solutions found", flush=True)

    if missing:
        miss_path = out_path / f"not_found_targets_v2_shard{shard_id}.csv"
        miss_rows = []
        for m in missing:
            if isinstance(m, dict) and "target_str" in m:
                miss_rows.append(
                    {
                        "structure_key": m.get("structure_key", ""),
                        "target": m.get("target_str", ""),
                    }
                )
            else:
                miss_rows.append({"structure_key": "", "target": str(m)})
        pd.DataFrame(miss_rows).to_csv(miss_path, index=False)
        print(f"missing units (this shard): {len(missing)} -> {miss_path}", flush=True)

    if stage_log_path.exists():
        print(f"stage log: {stage_log_path} rows={sum(1 for _ in open(stage_log_path)) - 1}", flush=True)

    print(f"Done in {datetime.datetime.now() - start}", flush=True)


if __name__ == "__main__":
    main()
