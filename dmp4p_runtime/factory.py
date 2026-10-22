"""Generic algorithm construction, no paper launcher dependency."""
from dmp4p_runtime.algorithms.v2_wrapper import V2Production,V2QualityContinuation,V2QualityRecovery,V2Quality
from dmp4p_runtime.algorithms.nutrient_ga import NutrientGuidedGA
from dmp4p_runtime.algorithms.cpsat import CPSATSolver
from dmp4p_runtime.algorithms.structure_nutrient_ga import StructureNutrientGA
from dmp4p_runtime.algorithms.quality_ablation import QualityAblation,VARIANTS
from dmp4p_runtime.algorithms.cpsat_band import CPSATBand
SUPPORTED_METHODS=("v2_production","v2_quality_continuation","v2_quality","v2_quality_recovery","nutrient_ga","structure_nutrient_ga","cpsat","cpsat_band",*VARIANTS)

def make_e1a_algorithm(name,*,snap,evaluator,cfg,device,seed,v2_population_size=1000,ga_population_size=200,cpsat_workers=1):
    if cfg.get("evidence_population_size") is not None:
        v2_population_size=ga_population_size=int(cfg["evidence_population_size"])
    kwargs=dict(pool=snap["pool"],target_dict=snap["target"],evaluator=evaluator,config=cfg,cell_ranges=snap.get("cell_ranges_restored"),personal=snap.get("personal"),device=device,seed=int(seed))
    classes={"v2_production":V2Production,"v2_quality_continuation":V2QualityContinuation,"v2_quality_recovery":V2QualityRecovery,"v2_quality":V2Quality}
    if name in classes:algo=classes[name](population_size=v2_population_size,**kwargs)
    elif name in VARIANTS:algo=QualityAblation(variant=name,population_size=v2_population_size,**kwargs)
    elif name in ("nutrient_ga","structure_nutrient_ga"):
        algo=({"nutrient_ga":NutrientGuidedGA,"structure_nutrient_ga":StructureNutrientGA}[name])(population_size=ga_population_size,**kwargs)
    elif name in ("cpsat","cpsat_band"):
        algo=({"cpsat":CPSATSolver,"cpsat_band":CPSATBand}[name])(num_search_workers=cpsat_workers,**kwargs)
    else:raise ValueError("Unknown method: "+name)
    algo.name=name;algo._forbidden=snap.get("forbidden_cells") or []
    if cfg.get("evidence_observation_mode")=="top8_common" and not name.startswith("cpsat"):
        from dmp4p_runtime.observation_control import install_common_observation
        algo=install_common_observation(algo)
    return algo
