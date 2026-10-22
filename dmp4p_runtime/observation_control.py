"""Optional shared top-8 generation observation for controlled supplements.
Internal archive/validation work remains part of each algorithm, within budget.
"""
import torch
from dmp4p_runtime.algorithms.structure_nutrient_ga import StructureNutrientGA

class SilentArchiveObserver:
    def __init__(self,real):self.real=real
    def observe_verified(self,*args,**kwargs):return None
    def __getattr__(self,name):return getattr(self.real,name)

class CommonTopK:
    def __init__(self,owner,pool):
        self.owner=owner;self.pool=pool;self.scorer=StructureNutrientGA(pool,owner.target_dict,owner.evaluator,personal=owner.personal,config=owner.config,population_size=8)
        self.tensors=None;self.calls=0;self.checked=0
    def observe(self,pop):
        clock=self.owner.budget_clock
        if not clock.in_budget():return
        if self.tensors is None:
            self.tensors=(torch.as_tensor(self.scorer._nut,device=pop.device),torch.as_tensor(self.scorer._grams,device=pop.device),torch.zeros(len(self.pool),dtype=torch.bool,device=pop.device))
        fitness,_,masks=self.scorer._fitness_batch(pop,*self.tensors)
        order=torch.argsort(fitness,descending=True)[:8].detach().cpu().tolist();self.calls+=1
        for i in order:
            if not clock.in_budget():break
            ev=self.owner.evaluator.evaluate_mask(self.pool,masks[i]);done=clock.now();self.checked+=1
            if ev.solution is not None:
                self.owner._evidence_real_observer.observe_verified(ev.solution,feasible=bool(ev.feasible),fingerprint_hash=ev.fingerprint_hash,eval_completed_at=done)

def install_common_observation(algo):
    original_solve=algo.solve
    def solve(budget,**kwargs):
        real=algo.dual_observers;algo._evidence_real_observer=real
        algo.dual_observers=SilentArchiveObserver(real)
        try:
            result=original_solve(budget,**kwargs)
            sample=getattr(algo,"_evidence_sampler",None)
            result.extra["observation_scope"]="shared_generation_top8_structural_order"
            result.extra["common_observe_generations"]=sample.calls if sample else 0
            result.extra["common_observe_evaluations"]=sample.checked if sample else 0
            return result
        finally:algo.dual_observers=real
    algo.solve=solve
    if hasattr(algo,"_build_combiner"):
        original_build=algo._build_combiner
        def build(n_solutions=1):
            combiner,device=original_build(n_solutions)
            sampler=CommonTopK(algo,combiner.food_df);algo._evidence_sampler=sampler
            fitness_function=combiner.fitness_function
            def fitness():
                result=fitness_function();sampler.observe(combiner.ga.population);return result
            combiner.fitness_function=fitness
            return combiner,device
        algo._build_combiner=build
    else:
        original_batch=algo._fitness_batch
        sampler=CommonTopK(algo,algo.pool);algo._evidence_sampler=sampler
        def fitness(pop,*args):
            result=original_batch(pop,*args);sampler.observe(pop);return result
        algo._fitness_batch=fitness
    return algo
