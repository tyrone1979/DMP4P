import torch


class GeneticAlgorithm:
    """遗传算法实现"""

    def __init__(self, population_size, chromosome_length, crossover_rate, mutation_rate,
                 device, debug=False):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = 0
        self.device = device
        self.population = None
        self.debug = debug
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.initialize_population()

    def initialize_population(self):
        p, n = self.population_size, self.chromosome_length
        k = torch.randint(10, 31, (p,), device=self.device)
        scores = torch.rand(p, n, device=self.device)
        _, idx = scores.topk(30, dim=1)
        mask = torch.arange(30, device=self.device).unsqueeze(0) < k.unsqueeze(1)
        pop = torch.zeros(p, n, dtype=torch.bool, device=self.device)
        pop.scatter_(1, idx, mask)
        self.population = pop

    def select_next_generation(self, fitness_scores):
        """使用选择操作来选择下一代个体"""
        valid_indices = torch.where(fitness_scores == 1)[0]
        invalid_indices = torch.where(fitness_scores != 1)[0]

        num_valid = len(valid_indices)
        num_valid = min(num_valid, self.population_size)
        if self.elite_size < num_valid:
            self.elite_size = num_valid
        new_population = self.population.clone()

        if num_valid > 0:
            new_population[:num_valid] = self.population[valid_indices[:num_valid]]

        if len(invalid_indices) > 0:
            invalid_fitness = fitness_scores[invalid_indices]
            if torch.any(invalid_fitness < 0):
                raise ValueError("negative values detected in invalid_fitness")
            if torch.any(torch.isnan(invalid_fitness)):
                raise ValueError("Nan values detected in invalid_fitness")
            if torch.any(torch.isinf(invalid_fitness)):
                raise ValueError("Inf fitness values detected in invalid_fitness")

            invalid_fitness_sum = float(invalid_fitness.sum().item())
            # All-zero / tiny mass → uniform (avoids CUDA multinomial cumdist==0 assert)
            if invalid_fitness_sum <= 0.0 or not bool(torch.any(invalid_fitness > 0).item()):
                invalid_probs = torch.ones(
                    len(invalid_indices), device=fitness_scores.device, dtype=fitness_scores.dtype
                )
                invalid_probs = invalid_probs / invalid_probs.sum()
            else:
                invalid_probs = invalid_fitness / invalid_fitness.sum()

            if torch.any(invalid_probs < 0) or torch.any(torch.isnan(invalid_probs)) or torch.any(
                    torch.isinf(invalid_probs)):
                raise ValueError("Invalid probabilities detected in invalid_probs")

            num_to_sample = min(self.population_size - num_valid, len(invalid_indices))
            selected_invalid = torch.multinomial(invalid_probs, num_to_sample, replacement=True)
            selected_invalid_indices = invalid_indices[selected_invalid]
            new_population[num_valid:] = self.population[selected_invalid_indices]

        self.population = new_population

    def crossover(self):
        """Vectorized single-point crossover (pairs i*2, i*2+1)."""
        p, n = self.population_size, self.chromosome_length
        pop = self.population
        n_pairs = p // 2
        start = min(int(self.elite_size), n_pairs)
        if start >= n_pairs:
            return
        do = torch.rand(n_pairs, device=self.device) < self.crossover_rate
        do[:start] = False
        pts = torch.randint(1, n, (n_pairs,), device=self.device)
        ar = torch.arange(n, device=self.device).unsqueeze(0)
        right = ar >= pts.unsqueeze(1)
        swap = do.unsqueeze(1) & right
        a = pop[0:n_pairs * 2:2]
        b = pop[1:n_pairs * 2:2]
        a2 = torch.where(swap, b, a)
        b2 = torch.where(swap, a, b)
        new = pop.clone()
        new[0:n_pairs * 2:2] = a2
        new[1:n_pairs * 2:2] = b2
        self.population = new

    def mutate(self):
        """One bit-flip per individual with mutation_rate (vectorized)."""
        p, n = self.population_size, self.chromosome_length
        hit = torch.rand(p, device=self.device) < self.mutation_rate
        pts = torch.randint(0, n, (p,), device=self.device)
        rows = torch.arange(p, device=self.device)[hit]
        cols = pts[hit]
        if len(rows):
            self.population[rows, cols] = ~self.population[rows, cols]
