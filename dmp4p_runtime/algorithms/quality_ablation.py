"""Named grouped ablations. The frozen full V2.1 class is unchanged."""
from ga_gpu import GeneticAlgorithm
from dmp4p_runtime.v2_quality_recovery import FoodCombinerV2QualityRecovery
from dmp4p_runtime.algorithms.v2_wrapper import V2QualityRecovery

class NoQualitySelection(FoodCombinerV2QualityRecovery):
    def _install_selection(self):
        # Remove the complete selection bundle (bounded elites, BLD/band ranking,
        # cached verification feedback); retain variation and recovery.
        def choose(fitness):
            GeneticAlgorithm.select_next_generation(self.ga,fitness)
            self.quality_stats["selection_calls"] += 1
        self.ga.select_next_generation=choose

class NoStructureSwap(FoodCombinerV2QualityRecovery):
    def _quality_cell_swaps(self):
        return None

class NoRecovery(FoodCombinerV2QualityRecovery):
    def _nutrient_recovery_swaps(self):
        return None
    def _inject_structured_immigrants(self):
        return None

VARIANTS={"v21_no_quality_selection":NoQualitySelection,"v21_no_structure_swap":NoStructureSwap,"v21_no_recovery":NoRecovery}

class QualityAblation(V2QualityRecovery):
    def __init__(self,*args,variant,**kwargs):
        super().__init__(*args,**kwargs);self.variant=variant;self.name=variant
    def _build_combiner(self,n_solutions=1):
        combiner,device=super()._build_combiner(n_solutions)
        combiner.__class__=VARIANTS[self.variant]
        combiner._install_selection()
        return combiner,device
