import torch
from tqdm import tqdm
import time  # 导入时间模块
import os



# 设置随机种子以确保可重复性

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class FoodCombiner:
    """食物组合优化类"""

    def __init__(self, food_df, target_dict, max_combination_size=20, population_size=300, crossover_rate=0.8,mutation_rate=0.01,device='cuda', debug=False):
        # 创建数据集
        self.food_df = food_df
        # 根据 target_dict 动态确定特征列
        self.feature_columns = ['positive', 'negative', 'eating_type']
        self.nutrient_columns = list(target_dict.keys())
        self.feature_columns.extend(self.nutrient_columns)
        self.features = torch.tensor(self.food_df[self.feature_columns].values, dtype=torch.float32).to(device)
        if torch.any(torch.isnan(self.features)):
            raise ValueError("NaN values detected in self.features")
        self.target_dict = target_dict
        self.max_combination_size = max_combination_size
        self.device = device
        self.num_foods = len(self.food_df)
        if self.num_foods == 0:
            raise ValueError("Dataset is empty. No foods found for the given target.")
        self.debug = debug

        self.ga = GeneticAlgorithm(
            population_size=population_size,
            chromosome_length=self.num_foods,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            device=self.device,
            debug=debug
        )

        # 预计算目标范围张量
        self.target_min = torch.tensor([v[0] for v in target_dict.values()], device=device)
        self.target_max = torch.tensor([v[1] for v in target_dict.values()], device=device)

    def fitness_function(self):
        """定义适应度函数并返回满足条件的个体的索引"""
        valid_indices = []
        fitness_scores = torch.zeros(self.ga.population_size, device=self.device)

        # 获取所有染色体的非零索引
        nonzero_indices = [torch.nonzero(chromosome).squeeze() for chromosome in self.ga.population]

        # 筛选符合条件的染色体
        valid = []
        max_length = 0
        for i, indices in enumerate(nonzero_indices):
            if indices.dim() == 0 or len(indices) == 0:
                fitness_scores[i] = 0
                continue
            if len(indices) < 3 or len(indices) > self.max_combination_size:
                fitness_scores[i] = 0
                continue
            valid.append((i, indices))
            if len(indices) > max_length:
                max_length = len(indices)

        # 填充索引以使它们具有相同的长度
        if valid:
            padded_indices = []
            for _, indices in valid:
                padded = torch.cat(
                    [indices, torch.full((max_length - len(indices),), -1, dtype=torch.long, device=self.device)])
                padded_indices.append(padded)
            indices_tensor = torch.stack(padded_indices)
            selected_foods = self.features[indices_tensor]

            mask = indices_tensor != -1
            selected_foods[~mask] = 0.0
            # 检查 selected_foods 是否包含 NaN
            if torch.any(torch.isnan(selected_foods)):
                raise ValueError("NaN values detected in selected_foods")
            penalties = self.check_nutrient_range_batch(selected_foods)
            # 检查 penalties 是否包含 NaN
            if torch.any(torch.isnan(penalties)):
                raise ValueError("NaN values detected in penalties")
            for idx, (i, penalty) in enumerate(zip([v[0] for v in valid], penalties)):
                if penalty == 0:
                    fitness_scores[i] = 1
                    valid_indices.append(indices_tensor[idx])
                else:
                    fitness_scores[i] = 1 / (1 + penalty)

        return valid_indices, fitness_scores

    def check_nutrient_range_batch(self, selected_foods):
        # selected_foods 的形状为 (batch_size, max_length, num_nutrients + 3)
        nutrients = selected_foods[:, :, 3:]  # 提取营养素部分
        nutrient_sums = nutrients.sum(dim=1)  # 按批次计算总和，形状为 (batch_size, num_nutrients)

        below_min = self.target_min - nutrient_sums
        below_min = torch.clamp(below_min, min=0)

        above_max = nutrient_sums - self.target_max
        above_max = torch.clamp(above_max, min=0)

        penalties = below_min.sum(dim=1) + above_max.sum(dim=1)
        return penalties

    def optimize(self, target_idx, batch_index=0,generations=500):
        """执行优化过程"""
        valid_indices = []
        for _ in tqdm(range(generations), desc=f'Batch {batch_index} Processing {target_idx} optimize', position=0, leave=True):

            if self.debug:
                start_time = time.time()
                valid_indices, fitness_scores = self.fitness_function()
                fitness_time = time.time() - start_time
                print(f"适应度计算运行时间: {fitness_time:.4f} 秒")
                if len(valid_indices) >= 20: # 提前终止
                    print(f'find solutions:{len(valid_indices)}')
                    break

                start_time = time.time()
                self.ga.select_next_generation(fitness_scores)
                select_time = time.time() - start_time
                print(f"选择操作运行时间: {select_time:.4f} 秒")

                start_time = time.time()
                self.ga.crossover()
                crossover_time = time.time() - start_time
                print(f"交叉操作运行时间: {crossover_time:.4f} 秒")

                start_time = time.time()
                self.ga.mutate()
                mutate_time = time.time() - start_time
                print(f"变异操作运行时间: {mutate_time:.4f} 秒")
            else:
                valid_indices, fitness_scores = self.fitness_function()
                if len(valid_indices) >= 20: # 提前终止
                    break
                self.ga.select_next_generation(fitness_scores)
                self.ga.crossover()
                self.ga.mutate()

        solution_scores = []

        for selected_indices in valid_indices:
            # 过滤掉填充的 -1 值
            valid = selected_indices != -1
            selected_indices_np = selected_indices[valid].cpu().numpy()
            solution = self.food_df.iloc[selected_indices_np]
            positive_sum = solution["positive"].sum()
            negative_sum = solution["negative"].sum()
            # 计算 eating_type 的种类数
            eating_types = solution["eating_type"].tolist()
            eating_type_count = len(set(eating_types))
            # 计算评分
            score = positive_sum - negative_sum + eating_type_count
            solution_scores.append((solution, score))

        # 按评分从高到低排序并取前20个
        solution_scores.sort(key=lambda x: x[1], reverse=True)
        top_20_solutions = [s[0] for s in solution_scores[:20]]

        print(f'solution length: {len(top_20_solutions)}')
        return top_20_solutions


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
        torch.cuda.empty_cache()  # 清理缓存
        self.initialize_population()

    def initialize_population(self):
        """初始化种群，确保每个个体包含所有 eating_type"""
        population = []
        for _ in range(self.population_size):
            chromosome = torch.zeros(self.chromosome_length, dtype=torch.bool, device=self.device)
            other_indices = torch.randint(0, self.chromosome_length,
                                          (torch.randint(10, 31, (1,), device=self.device).item(),), device=self.device)
            chromosome[other_indices] = True
            population.append(chromosome)
        self.population = torch.stack(population)

    def select_next_generation(self, fitness_scores):
        """使用选择操作来选择下一代个体"""
        valid_indices = torch.where(fitness_scores == 1)[0]
        invalid_indices = torch.where(fitness_scores != 1)[0]

        num_valid = len(valid_indices)
        num_valid = min(num_valid, self.population_size)
        if self.elite_size<num_valid:
            self.elite_size=num_valid
        new_population = self.population.clone()

        if num_valid > 0:
            new_population[:num_valid] = self.population[valid_indices[:num_valid]]

        if len(invalid_indices) > 0:
            invalid_fitness = fitness_scores[invalid_indices]
            # 检查 invalid_fitness 是否包含非法值
            if torch.any(invalid_fitness < 0):
                raise ValueError("negative values detected in invalid_fitness")
            if torch.any(torch.isnan(invalid_fitness)):
                raise ValueError("Nan values detected in invalid_fitness")
            if torch.any(torch.isinf(invalid_fitness)):
                raise ValueError("Inf fitness values detected in invalid_fitness")

            invalid_fitness_sum = invalid_fitness.sum()

            # 防止除以零
            if invalid_fitness_sum == 0:
                invalid_fitness_sum = 1e-10

            invalid_probs = invalid_fitness / invalid_fitness_sum

            # 检查 invalid_probs 是否包含非法值
            if torch.any(invalid_probs < 0) or torch.any(torch.isnan(invalid_probs)) or torch.any(
                    torch.isinf(invalid_probs)):
                raise ValueError("Invalid probabilities detected in invalid_probs")

            # 确保采样数量不超过概率分布的有效范围
            num_to_sample = min(self.population_size - num_valid, len(invalid_indices))
            selected_invalid = torch.multinomial(invalid_probs, num_to_sample, replacement=True)
            selected_invalid_indices = invalid_indices[selected_invalid]
            new_population[num_valid:] = self.population[selected_invalid_indices]

        self.population = new_population

    def crossover(self):
        """执行单点交叉操作"""
        new_population = self.population.clone()
        for i in range(self.elite_size, self.population_size // 2):
            parent1_idx = i * 2
            parent2_idx = i * 2 + 1
            parent1 = new_population[parent1_idx]
            parent2 = new_population[parent2_idx]
            child1 = parent1.clone()
            child2 = parent2.clone()

            if torch.rand(1, device=self.device) < self.crossover_rate:
                crossover_point = torch.randint(1, self.chromosome_length, (1,), device=self.device).item()
                child1[crossover_point:] = parent2[crossover_point:]
                child2[crossover_point:] = parent1[crossover_point:]

            new_population[parent1_idx] = child1
            new_population[parent2_idx] = child2
        self.population = new_population

    def mutate(self):
        """执行变异操作"""
        for i in range(self.population_size):
            if torch.rand(1, device=self.device) < self.mutation_rate:
                mutation_point = torch.randint(0, self.chromosome_length, (1,), device=self.device).item()
                self.population[i, mutation_point] = ~self.population[i, mutation_point]
