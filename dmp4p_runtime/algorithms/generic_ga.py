"""
Generic-GA baseline.

Binary candidate-row GA with shared cardinality / food_id repairs (no V2 guidance).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch

from dmp4p_runtime.algorithms.base import AlgoResult, BaseAlgorithm, SearchBudget
from dmp4p_runtime.algorithms.ga_common import (
    apply_shared_repairs,
    init_population_in_range,
    population_diagnostics,
)
from dmp4p_runtime.constraints import VIOLATION_SCALES
from dmp4p_runtime.rng_util import seed_all


class GenericGA(BaseAlgorithm):
    name = "generic_ga"

    def __init__(
        self,
        *args,
        population_size: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.population_size = int(population_size)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        nuts = []
        for c in self.nutrient_cols:
            if c in self.pool.columns:
                nuts.append(
                    pd.to_numeric(self.pool[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                )
            else:
                nuts.append(np.zeros(self.n_rows, dtype=np.float32))
        self._nut = np.stack(nuts, axis=1)
        self._grams = pd.to_numeric(self.pool["grams"], errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float32
        )
        self._food_ids = (
            pd.to_numeric(self.pool["food_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        )
        self._lo = np.array(
            [float(self.target_dict[c][0]) for c in self.nutrient_cols], dtype=np.float32
        )
        self._hi = np.array(
            [float(self.target_dict[c][1]) for c in self.nutrient_cols], dtype=np.float32
        )
        if "bench_placement_allowed" in self.pool.columns:
            self._placement_ok = self.pool["bench_placement_allowed"].astype(bool).to_numpy()
        else:
            self._placement_ok = np.ones(self.n_rows, dtype=bool)

    def solve(self, budget: SearchBudget, *, banned_food_ids: Optional[set[int]] = None) -> AlgoResult:
        try:
            return self._solve_impl(budget, banned_food_ids=banned_food_ids or set())
        except Exception as e:
            return AlgoResult(
                algorithm=self.name,
                status="error",
                failure_reason=f"exception:{type(e).__name__}:{e}",
            )

    def _allowed_mask(self, banned_food_ids: set[int]) -> np.ndarray:
        m = self._placement_ok.copy()
        if banned_food_ids:
            m &= ~np.isin(self._food_ids, list(banned_food_ids))
        return m

    def _solve_impl(self, budget: SearchBudget, *, banned_food_ids: set[int]) -> AlgoResult:
        seed_info = seed_all(self.seed, device=self.device)
        t_init0 = time.perf_counter()
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
        rng = np.random.default_rng(self.seed)
        pop = init_population_in_range(
            p, self.n_rows, allowed_idx, min_foods=mn, max_foods=mx, rng=rng
        )
        # Intentionally scramble a copy to measure pre-repair pathology after CX-like noise
        pop_noisy = pop.copy()
        for i in range(p):
            flips = rng.choice(self.n_rows, size=min(20, self.n_rows), replace=False)
            pop_noisy[i, flips] = ~pop_noisy[i, flips]
        diag_pre = population_diagnostics(
            pop_noisy,
            food_ids=self._food_ids,
            nut=self._nut,
            lo=self._lo,
            hi=self._hi,
            grams=self._grams,
            max_daily_g=float(self.evaluator.spec.max_daily_g),
            min_foods=mn,
            max_foods=mx,
        )
        t_rep0 = time.perf_counter()
        pop, repair_stats0 = apply_shared_repairs(
            pop,
            food_ids=self._food_ids,
            allowed_mask=allowed_mask,
            min_foods=mn,
            max_foods=mx,
            rng=rng,
        )
        # also show repair effect on noisy pop
        pop_fixed, _ = apply_shared_repairs(
            pop_noisy,
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
        diag_post_noise_repair = population_diagnostics(
            pop_fixed,
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
        t_init = time.perf_counter() - t_init0

        t_search0 = time.perf_counter()
        deadline = t_search0 + float(budget.wall_s)
        eval_count = 0
        best_ev = None
        best_norm = float("inf")
        found = None
        time_nutrient_ok = None
        gen = 0
        sync = str(device).startswith("cuda")
        diag_log = [diag0]
        repair_time_total = repair_s
        last_repair = repair_stats0

        while True:
            now = time.perf_counter()
            if now >= deadline:
                break
            if sync:
                torch.cuda.synchronize()
            fitness, _pen, masks_cpu = self._fitness_batch(pop_t, nut_t, grams_t, banned_t)
            eval_count += int(p)
            order = torch.argsort(fitness, descending=True)
            for rank_i in range(min(8, p)):
                idx = int(order[rank_i].item())
                mask = masks_cpu[idx] & allowed_mask
                ev = self.evaluator.evaluate_mask(self.pool, mask)
                found_at = time.perf_counter() - t_search0
                if found_at > float(budget.wall_s):
                    # late eval after deadline — do not count as in-budget success
                    break
                if ev.hard_violations.get("nutrient", 1) <= 0 and time_nutrient_ok is None:
                    time_nutrient_ok = found_at
                if ev.normalized_violation < best_norm:
                    best_norm = ev.normalized_violation
                    best_ev = ev
                if ev.feasible:
                    found = self._pack_eval(ev, found_at_s=found_at, eval_count=eval_count)
                    break
            if found and budget.stop_at_first_feasible:
                break
            if budget.max_evaluations and eval_count >= budget.max_evaluations:
                break
            if time.perf_counter() >= deadline:
                break

            pop_t = self._next_generation(pop_t, fitness, banned_t)
            # Shared repairs (CPU)
            t_r0 = time.perf_counter()
            masks = pop_t.detach().cpu().numpy()
            masks, last_repair = apply_shared_repairs(
                masks,
                food_ids=self._food_ids,
                allowed_mask=allowed_mask,
                min_foods=mn,
                max_foods=mx,
                rng=rng,
            )
            repair_time_total += time.perf_counter() - t_r0
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
        search_s = time.perf_counter() - t_search0
        overflow_s = max(0.0, search_s - float(budget.wall_s))

        solutions = []
        status = "timeout"
        # Only accept found if within budget
        if found and found.get("feasible") and float(found.get("found_at_s", 1e9)) <= float(budget.wall_s):
            solutions = [found]
            status = "ok"
        elif best_ev is not None:
            solutions = [self._pack_eval(best_ev, found_at_s=min(search_s, float(budget.wall_s)), eval_count=eval_count)]
            status = "timeout"

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
                "budget_overflow_s": overflow_s,
            },
            evaluation_count=eval_count,
            extra={
                "generations": gen,
                "seed_info": seed_info,
                "best_norm": best_norm,
                "device": device,
                "diagnostics_start": diag0,
                "diagnostics_pre_repair_noisy_demo_only": diag_pre,
                "diagnostics_post_repair_noisy_demo_only": diag_post_noise_repair,
                "diagnostics_log": diag_log[:20],
                "diagnostics_note": (
                    "diagnostics_log is actual population each gen; "
                    "*_noisy_demo_only is a synthetic pre/post-repair demo, not search state"
                ),
                "repair_stats_last": last_repair,
                "time_to_nutrient_ok": time_nutrient_ok,
                "time_to_first_feasible_unified_eval": (
                    found.get("found_at_s") if found and found.get("feasible") else None
                ),
                "candidate_set_hash": (
                    __import__(
                        "dmp4p_runtime.candidate_ids", fromlist=["pool_candidate_set_hash"]
                    ).pool_candidate_set_hash(self.pool)
                    if "candidate_id" in self.pool.columns
                    else None
                ),
                "sequential_ban_disabled": True,
            },
        )

    def _fitness_batch(self, pop_t, nut_t, grams_t, banned_t):
        if banned_t.any():
            pop_t = pop_t.clone()
            pop_t[:, banned_t] = False
        sel = pop_t.float()
        nut_sums = sel @ nut_t
        below = torch.clamp(torch.as_tensor(self._lo, device=pop_t.device) - nut_sums, min=0)
        above = torch.clamp(nut_sums - torch.as_tensor(self._hi, device=pop_t.device), min=0)
        nut_pen = (below + above).sum(dim=1) / VIOLATION_SCALES["nutrient"]
        n_sel = pop_t.sum(dim=1).float()
        mn = float(self.evaluator.spec.min_foods)
        mx = float(self.evaluator.spec.max_combination_size)
        card_pen = torch.where(
            n_sel < mn,
            (mn - n_sel) / VIOLATION_SCALES["cardinality"],
            torch.where(
                n_sel > mx,
                (n_sel - mx) / VIOLATION_SCALES["cardinality"],
                torch.zeros_like(n_sel),
            ),
        )
        total_g = (sel * grams_t).sum(dim=1)
        max_g = float(self.evaluator.spec.max_daily_g)
        wt_pen = torch.clamp(total_g - max_g, min=0.0) / VIOLATION_SCALES["daily_weight"]
        pen = nut_pen + card_pen + wt_pen
        fitness = 1.0 / (1.0 + pen)
        fitness = torch.where(pen <= 1e-9, torch.ones_like(fitness), fitness)
        return fitness, pen.detach(), pop_t.detach().cpu().numpy()

    def _next_generation(self, pop_t, fitness, banned_t):
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
        mut = torch.rand(p, n, device=device) < self.mutation_rate
        mut[:n_elite] = False
        new = torch.where(mut, ~new, new)
        if banned_t.any():
            new[:, banned_t] = False
        return new
