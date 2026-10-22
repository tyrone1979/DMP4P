"""Experimental V2-Quality: bounded elites, feasible structure selection, cell swaps.

Search feasibility is a float32 proxy; final acceptance remains UnifiedEvaluator.
No change to production FoodCombinerV2 defaults. Raw Layer B is a return-time tie
breaker, not a breeding objective in this version.
"""
from __future__ import annotations
import torch
import time
from ga_gpu_v2 import FoodCombinerV2


def select_quality_generation(pop, fitness, bld, band, *, elite_fraction=0.05):
    """Bounded unique elites; 80% tournament exploitation, 20% exploration."""
    p = len(pop)
    feasible = fitness == 1
    # Stable lex order: nutrient proxy feasible, then BLD, then raw band grams.
    order = torch.argsort(band, stable=True)
    order = order[torch.argsort((~bld[order]).to(torch.int32), stable=True)]
    # Infeasible parents retain nutrient-fitness pressure; all feasible scores tie.
    order = order[torch.argsort(fitness[order], descending=True, stable=True)]
    # Full chromosome identity, not food-only fingerprints, preserves distinct portions.
    _, inverse = torch.unique(pop[order], dim=0, return_inverse=True)
    positions = torch.arange(p, device=pop.device)
    first = torch.full((p,), p, dtype=torch.long, device=pop.device)
    first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
    unique_order = order[positions == first[inverse]]
    n_elite = min(max(1, int(p * elite_fraction)), len(unique_order))
    elites = unique_order[:n_elite]
    # Tournament rank sampling avoids treating every feasible as an immortal elite.
    ranks = torch.empty(p, dtype=torch.long, device=pop.device)
    ranks[order] = torch.arange(p, device=pop.device)
    contenders = torch.randint(p, (p-n_elite, 3), device=pop.device)
    winners = contenders.gather(1, ranks[contenders].argmin(1)[:, None]).flatten()
    explore = torch.rand(len(winners), device=pop.device) < 0.20
    weights = torch.nan_to_num(fitness, nan=0.0, posinf=0.0).clamp(min=0) + 1e-6
    random_parents = torch.multinomial(weights, len(winners), replacement=True)
    winners = torch.where(explore, random_parents, winners)
    return pop[torch.cat((elites, winners))].clone(), n_elite


class FoodCombinerV2Quality(FoodCombinerV2):
    """Explicit experimental subclass, never used by production default."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.cell_ga_enabled or self._cell_band_lo is None:
            raise ValueError("v2_quality requires personal cell bands")
        self.quality_stats = {"selection_calls": 0, "swap_attempts": 0, "swap_accepted": 0, "archive_duplicate_skips": 0, "archive_unique_trials": 0}
        self._quality_seen_archive_rows = set()
        self._install_selection()

    def _update_archive(self, valid_indices):
        # Repeated exact chromosomes have deterministic archive validation/repair
        # within this immutable combiner pool. Avoid rescoring them every generation.
        for indices in valid_indices:
            deadline = getattr(self, "wall_deadline", None)
            if deadline is not None and time.perf_counter() >= float(deadline):
                break
            identity = tuple(sorted(int(x) for x in indices.detach().cpu().tolist() if x >= 0))
            if identity in self._quality_seen_archive_rows:
                self.quality_stats["archive_duplicate_skips"] += 1
                continue
            self._quality_seen_archive_rows.add(identity)
            self.quality_stats["archive_unique_trials"] += 1
            super()._update_archive([indices])

    def _install_selection(self):
        def choose(fitness):
            pop = self.ga.population
            sel = pop.float()
            grams = self.features[:, 3]
            meals = self.features[:, 4]
            from algo.structure_cell_ops import cell_grams_batch
            cell_g = cell_grams_batch(sel, grams, meals, self.features[:, 5])
            lo, hi = self._cell_band_lo, self._cell_band_hi
            under = (lo-cell_g).clamp(min=0)
            for major in self._cell_upper_only:
                under[:, :, int(major)-1] = 0
            deviation = torch.where(hi > 0, under + (cell_g-hi).clamp(min=0), 0).sum((1,2))
            bld = torch.ones(len(pop), dtype=torch.bool, device=pop.device)
            for meal, minimum in ((1, self.min_b), (2, self.min_l), (3, self.min_d)):
                bld &= sel @ (grams * (meals == meal)) >= minimum
            parent_fitness = getattr(self, "_quality_parent_fitness", lambda value: value)(fitness)
            self.ga.population, self.ga.elite_size = select_quality_generation(pop, parent_fitness, bld, deviation)
            self.quality_stats["selection_calls"] += 1
        self.ga.select_next_generation = choose

    def _rebuild_ga(self, *args, **kwargs):
        super()._rebuild_ga(*args, **kwargs)
        self._install_selection()

    def _structure_cell_crossover_mutate(self):
        super()._structure_cell_crossover_mutate()
        self._quality_cell_swaps()

    def _quality_cell_swaps(self):
        """Up to 64 offspring trials; only strict band improvements survive proxy gates."""
        pop = self.ga.population
        start = int(self.ga.elite_size)
        if start >= len(pop):
            return
        rows = torch.randperm(len(pop)-start, device=pop.device)[:64] + start
        chosen = pop[rows].float()
        nonempty = chosen.sum(1) > 0
        rows, chosen = rows[nonempty], chosen[nonempty]
        if not len(rows):
            return
        old = torch.multinomial(chosen, 1).flatten()
        grams, meals, majors = self.features[:,3], self.features[:,4], self.features[:,5]
        keys = (meals.long()-1)*9 + majors.long()-1
        new = old.clone()
        for key in torch.unique(keys[old]).tolist():
            mask = keys[old] == key
            candidates = torch.where(keys == key)[0]
            if len(candidates):
                new[mask] = candidates[torch.randint(len(candidates), (int(mask.sum()),), device=pop.device)]
        self.quality_stats["swap_attempts"] += len(rows)
        delta = grams[new] - grams[old]
        cell_current = (chosen * grams * (keys[None,:] == keys[old,None])).sum(1)
        mi, mj = meals[old].long()-1, majors[old].long()-1
        lo, hi = self._cell_band_lo[mi,mj], self._cell_band_hi[mi,mj]
        upper_only = torch.zeros_like(lo, dtype=torch.bool)
        for major in self._cell_upper_only:
            upper_only |= majors[old] == int(major)
        def dev(g):
            return torch.where(upper_only, 0, (lo-g).clamp(min=0)) + (g-hi).clamp(min=0)
        accept = (hi > 0) & (dev(cell_current+delta) < dev(cell_current)-1e-5)
        nut = self.features[:,self.META_DIM:]
        totals = chosen @ nut + nut[new] - nut[old]
        accept &= ((totals >= self.target_min) & (totals <= self.target_max)).all(1)
        accept &= ~pop[rows,new]
        occupied = pop[rows].clone()
        occupied[torch.arange(len(rows), device=pop.device), old] = False
        accept &= ~(occupied & (self.food_ids_tensor[None,:] == self.food_ids_tensor[new,None])).any(1)
        accept &= (chosen.sum(1) >= self.min_foods) & (chosen.sum(1) <= self.max_combination_size)
        if self.hard_daily_weight:
            accept &= chosen @ grams + delta <= self.max_daily_g
        for meal, minimum in ((1,self.min_b), (2,self.min_l), (3,self.min_d)):
            total = chosen @ (grams*(meals == meal)) + delta*(meals[old] == meal)
            accept &= total >= minimum
        hit = rows[accept]
        pop[hit,old[accept]] = False
        pop[hit,new[accept]] = True
        self.quality_stats["swap_accepted"] += int(accept.sum())
