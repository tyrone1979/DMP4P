"""Same generic variation as Nutrient-GA; add lexicographic BLD/band selection.
No cell variation, structured immigrants or V2 recovery operators.
"""
import torch
from dmp4p_runtime.algorithms.nutrient_ga import NutrientGuidedGA
from algo.structure_category import UPPER_ONLY_MAJORS

class StructureNutrientGA(NutrientGuidedGA):
    name = "structure_nutrient_ga"

    def _fitness_batch(self, pop_t, nut_t, grams_t, banned_t):
        fitness, penalty, masks = super()._fitness_batch(pop_t,nut_t,grams_t,banned_t)
        sel = torch.as_tensor(masks,device=pop_t.device).float()
        band = torch.zeros(len(sel),device=pop_t.device)
        meals = self.pool["meal"].astype(str).str.lower().to_numpy()
        majors = self.pool["food_group_code"].astype(int).to_numpy()
        upper_only = set((self.personal or {}).get("upper_only_majors") or UPPER_ONLY_MAJORS)
        for key, bounds in ((self.personal or {}).get("bands") or {}).items():
            _,_,major,meal=key.split("_",3); major=int(major)
            hi=float(bounds["hi"]); lo=0.0 if major in upper_only else float(bounds["lo"])
            if hi<=0: continue  # prohibited cells remain evaluator hard gates
            cell=torch.as_tensor((meals==meal)&(majors==major),device=pop_t.device)
            total=sel @ (grams_t*cell)
            band += (lo-total).clamp(min=0)+(total-hi).clamp(min=0)
        bld=torch.ones(len(sel),dtype=torch.bool,device=pop_t.device)
        for meal,minimum in (("breakfast",30.0),("lunch",50.0),("dinner",50.0)):
            mask=torch.as_tensor(meals==meal,device=pop_t.device)
            bld &= sel @ (grams_t*mask)>=minimum
        # Stable lex order: existing nutrient/cardinality/weight fitness, BLD, band.
        order=torch.argsort(band,stable=True)
        order=order[torch.argsort((~bld[order]).int(),stable=True)]
        order=order[torch.argsort(fitness[order],descending=True,stable=True)]
        ranked=torch.empty_like(fitness)
        ranked[order]=torch.arange(len(sel),0,-1,device=pop_t.device,dtype=fitness.dtype)/len(sel)
        self.last_structure_diagnostics={"bld":bld.detach().cpu().tolist(),"band":band.detach().cpu().tolist()}
        return ranked,penalty,masks
