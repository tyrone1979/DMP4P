"""
Named V2 ablation flags for E2.

These are **benchmark-only** overlays applied when constructing FoodCombinerV2
config copies. They must not rewrite production YAML on disk.

Each flag documents which paths are disabled vs retained.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "v2_full": {
        "description": "Production-like: cell soft in fitness, Layer-B archive ranking",
        "disable_cell_soft": False,
        "disable_layer_b_rank": False,
        "uniform_archive": False,
        "disable_nutrient_guidance": False,
        "disable_freeze": False,
    },
    "v2_no_freeze": {
        "description": "Disable cell/elite freeze knobs when present in cell_ga cfg",
        "disable_cell_soft": False,
        "disable_layer_b_rank": False,
        "uniform_archive": False,
        "disable_nutrient_guidance": False,
        "disable_freeze": True,
    },
    "v2_no_structure_guidance": {
        "description": "Zero cell_soft influence; keep nutrient fitness + archive Layer-B; hi0 already in frozen pool",
        "disable_cell_soft": True,
        "disable_layer_b_rank": False,
        "uniform_archive": False,
        "disable_nutrient_guidance": False,
        "disable_freeze": False,
        "caveat": "Cannot prove hi0 cell-op benefit on pools that already removed hi0 rows",
    },
    "v2_no_nutrient_guidance": {
        "description": "Empty nutrient responsibility + no nutrient-directed mutate modes; structure soft/evolve retained",
        "disable_cell_soft": False,
        "disable_layer_b_rank": False,
        "uniform_archive": False,
        "disable_nutrient_guidance": True,
        "disable_freeze": False,
        "implementation_status": "wired_in_combiner",
    },
    "v2_uniform_archive": {
        "description": "Same hard gates + fingerprint dedupe; first-seen wins (no Layer-B compare)",
        "disable_cell_soft": False,
        "disable_layer_b_rank": True,
        "uniform_archive": True,
        "disable_nutrient_guidance": False,
        "disable_freeze": False,
    },
}


def apply_ablation_to_cfg(base_cfg: dict, variant: str) -> tuple[dict, dict]:
    """Return (cfg_copy, resolved_spec)."""
    if variant not in VARIANT_SPECS:
        raise KeyError(f"unknown ablation variant: {variant}")
    spec = dict(VARIANT_SPECS[variant])
    cfg = deepcopy(base_cfg)
    cfg["_benchmark_ablation"] = {"variant": variant, **spec}
    st = dict(cfg.get("structure") or {})
    cg = dict(st.get("cell_ga") or {})
    if spec.get("disable_freeze"):
        # Best-effort: clear freeze-related knobs if present
        for k in (
            "freeze_feasible_cells",
            "freeze_elites",
            "lock_feasible_cells",
            "freeze_true_cells",
            "protect_elites",
        ):
            if k in cg:
                cg[k] = False
        cg["benchmark_disable_freeze"] = True
    if spec.get("disable_cell_soft"):
        # Consumed by FoodCombinerV2.fitness_function (benchmark-only)
        cg["benchmark_zero_cell_soft"] = True
        cg["benchmark_ignore_structure_evolve"] = True
    st["cell_ga"] = cg
    cfg["structure"] = st
    if spec.get("uniform_archive") or spec.get("disable_layer_b_rank"):
        cfg["benchmark_uniform_archive"] = True
    if spec.get("disable_nutrient_guidance"):
        cfg["benchmark_disable_nutrient_guidance"] = True
    return cfg, spec


def variant_ids() -> list[str]:
    return list(VARIANT_SPECS.keys())
