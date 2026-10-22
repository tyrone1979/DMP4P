"""V2-Quality recovery v2: independently verified parents and scheduled immigrants.

Separate named developer variant; no changes to original v2_quality behavior.
"""
from __future__ import annotations
import time
import random
import torch
from dmp4p_runtime.v2_quality_search import FoodCombinerV2Quality

class FoodCombinerV2QualityRecovery(FoodCombinerV2Quality):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.benchmark_cell_rng = random.Random(int(self.cfg.get("ga_seed", 0)) + 104729)
        self.quality_verified = {}
        self.quality_stats.update(verified_rejections=0,verified_acceptances=0,immigrant_batches=0,
                                  immigrant_individuals=0,nutrient_swap_attempts=0,nutrient_swap_accepted=0)
        self.quality_rejection_reasons = {}

    def _update_archive(self,valid_indices):
        evaluator = getattr(self,"quality_evaluator",None)
        for indices in valid_indices:
            deadline=getattr(self,"wall_deadline",None)
            if deadline is not None and time.perf_counter() >= deadline: break
            identity=tuple(sorted(int(x) for x in indices.detach().cpu().tolist() if x >= 0))
            if evaluator is not None and identity not in self.quality_verified:
                ev=evaluator.evaluate_indices(self.food_df,list(identity))
                self.quality_verified[identity]=(bool(ev.feasible),float(ev.normalized_violation),bool(ev.bld_ok))
                key="verified_acceptances" if ev.feasible else "verified_rejections"
                self.quality_stats[key]+=1
                for reason in ev.failure_reasons:
                    self.quality_rejection_reasons[reason]=self.quality_rejection_reasons.get(reason,0)+1
            super()._update_archive([indices])

    def _quality_parent_fitness(self,fitness):
        ranked=fitness.clone()
        if not self.quality_verified: return ranked
        pop=self.ga.population
        k=min(int(self.max_combination_size),pop.shape[1])
        values,indices=pop.float().topk(k,dim=1)
        indices=indices.masked_fill(values<0.5,-1).detach().cpu().tolist()
        original_scores=fitness.detach().cpu().tolist()
        updated_indices,updated_values=[],[]
        for i,row in enumerate(indices):
            identity=tuple(sorted(x for x in row if x>=0))
            verdict=self.quality_verified.get(identity)
            if verdict is None: continue
            ok,violation,bld_ok=verdict
            updated_indices.append(i)
            updated_values.append((2.0 if bld_ok else 0.9) if ok else min(original_scores[i],0.05/(1.0+violation)))
        if updated_indices:
            ranked[torch.tensor(updated_indices,device=pop.device)]=torch.tensor(updated_values,device=pop.device,dtype=ranked.dtype)
        return ranked

    def _structure_cell_crossover_mutate(self):
        super()._structure_cell_crossover_mutate()
        self._nutrient_recovery_swaps()
        # A fixed schedule, not tuned to whether this run beats its comparator.
        if self.quality_stats["selection_calls"] % 10 == 0:
            self._inject_structured_immigrants()

    def _inject_structured_immigrants(self):
        ga=self.ga
        original_pop,original_size=ga.population,ga.population_size
        count=min(max(1,original_size//10),original_size-int(ga.elite_size))
        if count<=0: return
        old_offset=self.cfg.get("pop_seed_offset",0)
        try:
            self.cfg["pop_seed_offset"]=int(old_offset)+100003*(self.quality_stats["immigrant_batches"]+1)
            ga.population_size=count
            ga.population=torch.zeros((count,original_pop.shape[1]),dtype=torch.bool,device=original_pop.device)
            self._seed_cell_population()
            fresh=ga.population.clone()
        finally:
            ga.population,ga.population_size=original_pop,original_size
            self.cfg["pop_seed_offset"]=old_offset
        ga.population[-count:]=fresh
        self.quality_stats["immigrant_batches"]+=1
        self.quality_stats["immigrant_individuals"]+=count

    def _nutrient_recovery_swaps(self):
        pop=self.ga.population
        start=int(self.ga.elite_size)
        if start>=len(pop): return
        rows=torch.randperm(len(pop)-start,device=pop.device)[:64]+start
        chosen=pop[rows].float()
        rows,chosen=rows[chosen.sum(1)>0],chosen[chosen.sum(1)>0]
        if not len(rows): return
        nut=self.features[:,self.META_DIM:]
        totals=chosen@nut
        scale=torch.maximum((self.target_max-self.target_min).abs(),self.target_max.abs()*0.1).clamp(min=1)
        def violation(values):
            return (((self.target_min-values).clamp(min=0)+(values-self.target_max).clamp(min=0))/scale).sum(1)
        before=violation(totals)
        active=before>1e-7
        rows,chosen,totals,before=rows[active],chosen[active],totals[active],before[active]
        if not len(rows): return
        old=torch.multinomial(chosen,1).flatten()
        grams,meals,majors=self.features[:,3],self.features[:,4],self.features[:,5]
        keys=(meals.long()-1)*9+majors.long()-1
        direction=((self.target_min-totals).clamp(min=0)-(totals-self.target_max).clamp(min=0))/scale.square()
        score=direction@nut.T
        allowed=(keys[None,:]==keys[old,None]) & ~pop[rows]
        score=score.masked_fill(~allowed,float('-inf'))
        k=min(5,pop.shape[1])
        candidates=score.topk(k,dim=1).indices
        new=candidates.gather(1,torch.randint(k,(len(rows),1),device=pop.device)).flatten()
        accept=allowed[torch.arange(len(rows),device=pop.device),new]
        after=totals+nut[new]-nut[old]
        accept &= violation(after)<before-1e-6
        occupied=pop[rows].clone()
        occupied[torch.arange(len(rows),device=pop.device),old]=False
        accept &= ~(occupied & (self.food_ids_tensor[None,:]==self.food_ids_tensor[new,None])).any(1)
        delta=grams[new]-grams[old]
        if self.hard_daily_weight: accept &= chosen@grams+delta<=self.max_daily_g
        accept &= (chosen.sum(1)>=self.min_foods)&(chosen.sum(1)<=self.max_combination_size)
        for meal,minimum in ((1,self.min_b),(2,self.min_l),(3,self.min_d)):
            accept &= chosen@(grams*(meals==meal))+delta*(meals[old]==meal)>=minimum
        self.quality_stats["nutrient_swap_attempts"]+=len(rows)
        hit=rows[accept]
        pop[hit,old[accept]]=False
        pop[hit,new[accept]]=True
        self.quality_stats["nutrient_swap_accepted"]+=int(accept.sum())
