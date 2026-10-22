"""
Nutrient-guided-GA: same init/select/crossover/repairs as Generic-GA;
only mutation differs (nutrient gap add/drop/replace).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from dmp4p_runtime.algorithms.base import AlgoResult, SearchBudget
from dmp4p_runtime.algorithms.ga_common import (
    apply_shared_repairs,
    init_population_in_range,
    population_diagnostics,
)
from dmp4p_runtime.algorithms.generic_ga import GenericGA
from dmp4p_runtime.rng_util import seed_all


class NutrientGuidedGA(GenericGA):
    name = "nutrient_ga"

    def _solve_impl(self, budget: SearchBudget, *, banned_food_ids: set[int]) -> AlgoResult:
        from dmp4p_runtime.budget_clock import BudgetClock

        seed_info = seed_all(self.seed, device=self.device)
        clock = getattr(self, "budget_clock", None) or BudgetClock(wall_s=float(budget.wall_s))
        self.budget_clock = clock
        t_init0 = clock.now()
        device = self.device
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        allowed_mask = self._allowed_mask(banned_food_ids)
        allowed_idx = np.flatnonzero(allowed_mask)
        mn = int(self.evaluator.spec.min_foods)
        mx = int(self.evaluator.spec.max_combination_size)
        if len(allowed_idx) < mn:
            return AlgoResult(
                algorithm=self.name,
                status="incomplete",
                failure_reason="insufficient_unbanned_rows",
                timings={"init_s": time.perf_counter() - t_init0},
            )

        p = self.population_size
        rng = np.random.default_rng(self.seed + 17)
        pop = init_population_in_range(
            p, self.n_rows, allowed_idx, min_foods=mn, max_foods=mx, rng=rng
        )
        t_rep0 = time.perf_counter()
        pop, repair0 = apply_shared_repairs(
            pop,
            food_ids=self._food_ids,
            allowed_mask=allowed_mask,
            min_foods=mn,
            max_foods=mx,
            rng=rng,
        )
        repair_s = time.perf_counter() - t_rep0
        diag0 = population_diagnostics(
            pop,
            food_ids=self._food_ids,
            nut=self._nut,
            lo=self._lo,
            hi=self._hi,
            grams=self._grams,
            max_daily_g=float(self.evaluator.spec.max_daily_g),
            min_foods=mn,
            max_foods=mx,
        )
        pop_t = torch.as_tensor(pop, device=device)
        nut_t = torch.as_tensor(self._nut, device=device)
        grams_t = torch.as_tensor(self._grams, device=device)
        banned_t = torch.as_tensor(~allowed_mask, device=device)
        dens = self._nut / np.maximum(self._grams.reshape(-1, 1), 1.0)
        t_init = clock.now() - t_init0

        t_search0 = clock.now()
        eval_count = 0
        best_ev = None
        best_norm = float("inf")
        found = None
        best_layer_b = None  # emit-time only; breeding unchanged
        time_nutrient_ok = None
        gen = 0
        sync = str(device).startswith("cuda")
        diag_log = [diag0]
        repair_time_total = repair_s
        dual = getattr(self, "dual_observers", None)
        post_deadline = []

        while True:
            if not clock.in_budget():
                break
            if sync:
                torch.cuda.synchronize()
            fitness, _pen, masks_cpu = self._fitness_batch(pop_t, nut_t, grams_t, banned_t)
            eval_count += int(p)
            order = torch.argsort(fitness, descending=True)
            for rank_i in range(min(8, p)):
                if not clock.in_budget():
                    break
                idx = int(order[rank_i].item())
                mask = masks_cpu[idx] & allowed_mask
                ev = self.evaluator.evaluate_mask(self.pool, mask)
                t_done = clock.now()
                found_at_rel = float(clock.rel(t_done) or 0.0)
                if not clock.in_budget(t_done):
                    packed = self._pack_eval(ev, found_at_s=found_at_rel, eval_count=eval_count)
                    packed["in_budget"] = False
                    post_deadline.append(packed)
                    if dual is not None and ev.solution is not None:
                        dual.observe_verified(
                            ev.solution,
                            feasible=bool(ev.feasible),
                            fingerprint_hash=ev.fingerprint_hash,
                            eval_completed_at=t_done,
                        )
                    break
                if ev.hard_violations.get("nutrient", 1) <= 0 and time_nutrient_ok is None:
                    time_nutrient_ok = found_at_rel
                if ev.normalized_violation < best_norm:
                    best_norm = ev.normalized_violation
                    best_ev = ev
                if ev.feasible:
                    packed = self._pack_eval(ev, found_at_s=found_at_rel, eval_count=eval_count)
                    packed["in_budget"] = True
                    if dual is not None and ev.solution is not None:
                        dual.observe_verified(
                            ev.solution,
                            feasible=True,
                            fingerprint_hash=ev.fingerprint_hash,
                            eval_completed_at=t_done,
                        )
                    lb = ev.layer_b_score
                    if found is None or (
                        lb is not None
                        and (best_layer_b is None or float(lb) > float(best_layer_b))
                    ):
                        found = packed
                        best_layer_b = float(lb) if lb is not None else best_layer_b
                    if budget.stop_at_first_feasible:
                        break
            if found and budget.stop_at_first_feasible:
                break
            if budget.max_evaluations and eval_count >= budget.max_evaluations:
                break
            if not clock.in_budget():
                break

            pop_t = self._next_generation_select_cx(pop_t, fitness)
            masks = pop_t.detach().cpu().numpy()
            masks = self._nutrient_mutate(masks, dens, allowed_mask, rng)
            t_r0 = clock.now()
            masks, _ = apply_shared_repairs(
                masks,
                food_ids=self._food_ids,
                allowed_mask=allowed_mask,
                min_foods=mn,
                max_foods=mx,
                rng=rng,
            )
            repair_time_total += clock.now() - t_r0
            pop_t = torch.as_tensor(masks, device=device)
            if gen < 5 or gen % 10 == 0:
                diag_log.append(
                    {
                        "gen": gen,
                        **population_diagnostics(
                            masks,
                            food_ids=self._food_ids,
                            nut=self._nut,
                            lo=self._lo,
                            hi=self._hi,
                            grams=self._grams,
                            max_daily_g=float(self.evaluator.spec.max_daily_g),
                            min_foods=mn,
                            max_foods=mx,
                        ),
                    }
                )
            gen += 1

        if sync:
            torch.cuda.synchronize()
        clock.mark_search_end()
        search_s = float(clock.last_search_end) - t_search0
        clock.mark_finalization_end()
        solutions = []
        status = "timeout"
        if found and found.get("feasible") and found.get("in_budget", True):
            solutions = [found]
            status = "ok"
        return AlgoResult(
            algorithm=self.name,
            status=status,
            failure_reason=None if status == "ok" else "no_feasible_in_budget",
            solutions=solutions,
            timings={
                "init_s": t_init,
                "search_s": search_s,
                "repair_s": repair_time_total,
                "decode_eval_s": 0.0,
                "init_search_decode_merged_note": (
                    "init+search+eval share one deadline from solve start"
                ),
                "budget_overflow_s": clock.budget_overflow_s(),
            },
            evaluation_count=eval_count,
            extra={
                "generations": gen,
                "seed_info": seed_info,
                "best_norm": best_norm,
                "device": device,
                "mutation": "nutrient_gap_add_drop_replace",
                "diagnostics_start": diag0,
                "diagnostics_log": diag_log[:20],
                "time_to_nutrient_ok": time_nutrient_ok,
                "budget_clock": clock.to_dict(),
                "n_post_deadline_diagnostic": len(post_deadline),
                "dual_observers": dual.summary() if dual is not None else None,
                "best_ev_norm_only_not_returned": best_ev is not None and status != "ok",
            },
        )

    def _next_generation_select_cx(self, pop_t, fitness):
        """Same selection + crossover as GenericGA; no bit-flip (nutrient mutate instead)."""
        p, n = pop_t.shape
        device = pop_t.device
        n_elite = max(2, p // 10)
        order = torch.argsort(fitness, descending=True)
        new = pop_t.clone()
        new[:n_elite] = pop_t[order[:n_elite]]
        for i in range(n_elite, p):
            a, b = torch.randint(0, p, (2,), device=device)
            winner = a if fitness[a] >= fitness[b] else b
            new[i] = pop_t[winner]
        n_pairs = (p - n_elite) // 2
        for j in range(n_pairs):
            if float(torch.rand(1, device=device)) > self.crossover_rate:
                continue
            i0 = n_elite + 2 * j
            i1 = i0 + 1
            pt = int(torch.randint(1, n, (1,), device=device))
            a, b = new[i0].clone(), new[i1].clone()
            new[i0, pt:], new[i1, pt:] = b[pt:], a[pt:]
        return new

    def _nutrient_mutate(self, masks, dens, allowed_mask, rng):
        p = len(masks)
        n_elite = max(2, p // 10)
        lo, hi = self._lo, self._hi
        for i in range(n_elite, p):
            if rng.random() > max(0.2, self.mutation_rate * 10):
                continue
            mask = masks[i].copy()
            mask &= allowed_mask
            totals = mask @ self._nut
            deficit = np.clip(lo - totals, 0, None)
            excess = np.clip(totals - hi, 0, None)
            if deficit.sum() > excess.sum() and deficit.sum() > 0:
                k = int(np.argmax(deficit))
                off = np.flatnonzero(~mask & allowed_mask)
                if len(off):
                    scores = dens[off, k]
                    pick = off[int(np.argmax(scores))]
                    if mask.sum() < self.evaluator.spec.max_combination_size:
                        mask[pick] = True
                    else:
                        on = np.flatnonzero(mask)
                        drop = on[int(rng.integers(0, len(on)))]
                        mask[drop] = False
                        mask[pick] = True
            elif excess.sum() > 0:
                k = int(np.argmax(excess))
                on = np.flatnonzero(mask)
                if len(on):
                    contrib = self._nut[on, k]
                    drop = on[int(np.argmax(contrib))]
                    mask[drop] = False
            else:
                on = np.flatnonzero(mask)
                off = np.flatnonzero(~mask & allowed_mask)
                if len(on) and len(off):
                    mask[on[int(rng.integers(0, len(on)))]] = False
                    mask[off[int(rng.integers(0, len(off)))]] = True
            masks[i] = mask
        return masks
