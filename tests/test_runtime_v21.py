import itertools,inspect,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dmp4p_runtime.constraints import ConstraintSpec
from dmp4p_runtime.evaluator import UnifiedEvaluator
from dmp4p_runtime.algorithms.base import SearchBudget
from dmp4p_runtime.algorithms.structure_nutrient_ga import StructureNutrientGA
from dmp4p_runtime.algorithms.cpsat_band import CPSATBand
from dmp4p_runtime.algorithms.quality_ablation import NoQualitySelection,NoStructureSwap,NoRecovery
from dmp4p_runtime.v2_quality_search import select_quality_generation as copied_select

def fixture():
    pool=pd.DataFrame({"food_id":[51000001+i for i in range(6)],"grams":[30.,60.,50.,80.,50.,90.],"calorie":[30.,30.,50.,50.,50.,50.],"food_group_code":[5]*6,"meal":["breakfast"]*2+["lunch"]*2+["dinner"]*2,"meal_id":[1,1,2,2,3,3],"eating_type":[1,1,2,2,3,3],"food_desc":["synthetic test food"]*6,"positive":[1]*6,"negative":[0]*6,"is_common":[1]*6,"bench_placement_allowed":[True]*6})
    personal={"bands":{f"g_major_5_{m}":{"lo":0,"hi":v} for m,v in [("breakfast",40),("lunch",60),("dinner",60)]},"upper_only_majors":[8,9]}
    spec=ConstraintSpec(min_foods=3,max_combination_size=3,require_bld=False,hard_daily_weight=True,max_daily_g=300,category_presence_hard=True,category_grams_hard=False,check_breakfast_placement=False,check_condiment=False)
    target={"calorie":[120,150]}
    ev=UnifiedEvaluator(target,spec,config={},structure_prior={},personal_category_structure=personal)
    return pool,personal,target,ev

def test_structure_baseline_prefers_smaller_deviation_without_cell_variation():
    pool,personal,target,ev=fixture();a=StructureNutrientGA(pool,target,ev,personal=personal,config={},population_size=2)
    pop=torch.tensor([[0,1,0,1,0,1],[1,0,1,0,1,0]],dtype=torch.bool)
    fit,pen,_=a._fitness_batch(pop,torch.tensor(a._nut),torch.tensor(a._grams),torch.zeros(6,dtype=torch.bool))
    assert fit[1]>fit[0] and pen[0]==pen[1]==0
    assert not hasattr(a,"_quality_cell_swaps")

def test_cpsat_band_matches_small_exhaustive_optimum_and_reports_bounds():
    from ortools.sat.python import cp_model
    pool,personal,target,ev=fixture();a=CPSATBand(pool,target,ev,personal=personal,config={})
    model,x,meta=a._build_model(cp_model,active=np.arange(6),nogoods=[])
    solver=cp_model.CpSolver();assert solver.Solve(model)==cp_model.OPTIMAL
    selected=[i for i,v in x.items() if solver.Value(v)];assert selected==[0,2,4]
    assert solver.ObjectiveValue()==0
    assert ev.evaluate_indices(pool,selected).feasible
    result=a.solve(SearchBudget(wall_s=3,stop_at_first_feasible=False))
    assert result.status=="ok",result.failure_reason
    assert result.extra["integer_band_objective"]==0 and result.extra["integer_band_best_bound"]==0
    assert result.extra["raw_layer_b_optimized"] is False


def test_ablations_disable_only_named_paths():
    assert NoStructureSwap._quality_cell_swaps(object()) is None
    assert NoRecovery._nutrient_recovery_swaps(object()) is None
    assert NoRecovery._inject_structured_immigrants(object()) is None
    assert "selection_calls" in inspect.getsource(NoQualitySelection._install_selection)



def test_no_selection_keeps_recovery_schedule(monkeypatch):
    from types import SimpleNamespace
    from ga_gpu import GeneticAlgorithm
    seen=[]
    monkeypatch.setattr(GeneticAlgorithm,"select_next_generation",lambda ga,fit:seen.append(1))
    obj=NoQualitySelection.__new__(NoQualitySelection);obj.ga=SimpleNamespace();obj.quality_stats={"selection_calls":0}
    obj._install_selection()
    boundaries=[]
    for i in range(30):
        obj.ga.select_next_generation(torch.ones(2))
        if obj.quality_stats["selection_calls"]%10==0:boundaries.append(i+1)
    assert len(seen)==30 and boundaries==[10,20,30]

def test_cpsat_integer_model_feasible_subset_and_float_optimum():
    from ortools.sat.python import cp_model
    pool,personal,target,ev=fixture();a=CPSATBand(pool,target,ev,personal=personal,config={})
    float_cost=[];int_cost=[]
    for selected in itertools.combinations(range(6),3):
        verdict=ev.evaluate_indices(pool,list(selected))
        grams={m:pool.iloc[list(selected)].query("meal == @m").grams.sum() for m in ["breakfast","lunch","dinner"]}
        ok=verdict.feasible and verdict.bld_ok
        cost=sum(max(0,grams[m]-hi) for m,hi in [("breakfast",40),("lunch",60),("dinner",60)])
        if ok:float_cost.append(cost)
        model,x,_=a._build_model(cp_model,active=np.arange(6),nogoods=[])
        for i,var in x.items():model.Add(var==int(i in selected))
        solver=cp_model.CpSolver();status=solver.Solve(model)
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            assert ok
            int_cost.append(solver.ObjectiveValue()/10)
            assert abs(solver.ObjectiveValue()/10-cost)<=a.objective_meta["max_abs_objective_error_g"]
    assert len(int_cost)>1 and min(int_cost)==min(float_cost)==0
