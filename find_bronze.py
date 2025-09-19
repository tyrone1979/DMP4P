from ga_gpu import FoodCombiner
from tqdm import tqdm
import pandas as pd
from utils.target_calculation import get_nutrition_target_range_by_situation


def find_food_from_target(food_df, target_dict):
    found_df = food_df.copy()
    #found_df.fillna(0, inplace=True)
    for key, value in target_dict.items():
        found_df = found_df[found_df[key] <= (value[1])]
    return found_df


def find_food_from_targets(food_df: object, target_dicts: object, batch_index: object, population_size: object, generations: object,
                           crossover_rate : float,mutate_rate: float,device: int) -> object:
    """
    在多个目标字典上查找食物组合，带有进度条显示进度
    """
    results = []
    not_found = []

    # 使用 tqdm 创建进度条
    for idx, target_dict in enumerate(target_dicts):
        day = 3

        found_food_df = find_food_from_target(food_df, target_dict)
        if len(found_food_df) == 0:
            print('no dataset')
            continue
        combiner = FoodCombiner(found_food_df, target_dict, max_combination_size=30,
                                population_size=population_size, crossover_rate=crossover_rate,mutation_rate=mutate_rate,device='cuda:'+str(device), debug=False)
        best_solutions = combiner.optimize(idx, batch_index=batch_index, generations= generations)
        if len(best_solutions) > 0:
            for solution in best_solutions:
                # 创建一个独立的副本
                solution = solution.copy()
                solution['day'] = day
                solution['years'] = 2025
                solution['target'] = str(target_dict)
                day += 1
                results.append(solution)
        else:
            not_found.append(target_dict)
    if len(results) > 0:
        return pd.concat(results, ignore_index=True), not_found
    else:
        return results, not_found

import os

def process_and_save_results(food_user, user, parsed_targets, device=0,batch_size=1000, population_size=1000, generation=100,
                             crossover_rate=0.8,mutate_rate=0.01, out_path='data/nhanes/data'):
    all_results = pd.DataFrame()  # 用于存储所有结果
    # 分批次处理目标
    for i in tqdm(range(0, len(parsed_targets), batch_size), desc='batch processing...', position=0, leave=True):
        batch_targets = parsed_targets[i:i + batch_size]
        results, not_found = find_food_from_targets(food_user, batch_targets, i, population_size, generation,crossover_rate,mutate_rate,device)
        # 合并用户信息
        if results is not None and len(results)>0:
            results = pd.merge(results, user[['user_id', 'target']], on=['target'], how='left')
            # 将当前批次的结果添加到所有结果中
            all_results = pd.concat([all_results, results])
            # 保存当前批次的结果
            results.to_csv(f'{out_path}/_daily_food_with_nutrition_target_bronze_batch_{i // batch_size}.csv', index=False)

    # 保存所有结果
    bronze_file = f'{out_path}/_daily_food_with_nutrition_target_bronze.csv'
    if os.path.exists(bronze_file):
            existing_data = pd.read_csv(bronze_file)
    else:
            existing_data = pd.DataFrame()
            # 保存所有结果到目标文件
    if not all_results.empty:
        if not existing_data.empty:
                # 如果已有数据，将新结果追加到已有数据中
            all_results = pd.concat([existing_data, all_results])
            # 保存到目标文件
            all_results.to_csv(bronze_file, index=False)
            return all_results
    else:
        return existing_data


def load_data(path):
    user = pd.read_csv(f'{path}/not_found_match_food_user.csv')
    user['target'] = user.apply(get_nutrition_target_range_by_situation, axis=1)
    food = pd.read_csv(f'{path}/food_code.csv')
    food_nutrition = pd.read_csv(f'{path}/food_nutrition.csv')
    food_nutrition = pd.merge(food_nutrition, food[['food_id', 'positive', 'negative']], on=['food_id'], how='left')
    food_user = pd.read_csv(f'{path}/food_user.csv')
    food_user = pd.merge(food_user, food_nutrition, on=['food_id', 'grams'], how='left')
    # 创建一个映射字典
    spanish_to_english = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        10: 1,  # Desayuno -> Breakfast
        11: 2,  # Almuerzo -> Lunch
        12: 3,  # Comida -> Dinner
        13: 6,  # Merienda -> Snack
        14: 4,  # Cena -> Supper
        15: 6,  # Entre comida -> Between meals
        16: 6,  # Botana -> Snack
        17: 6,  # Bocadillo -> Snack
        18: 6,  # Tentempie -> Snack
        19: 7  # Bebida -> Drink
    }
    # 将 eating_type 列中的西班牙语编号映射到英语编号
    food_user['eating_type'] = food_user['eating_type'].map(spanish_to_english)
    food_user = food_user[food_user['eating_type'] < 8]
    food_user = food_user.drop_duplicates()
    food_user = food_user.drop(columns=['user_id'])

    not_found_target_file = f'{path}/not_found_targets.csv'
    # 判断文件是否存在
    if os.path.exists(not_found_target_file):
        print('Found not_found_targets file')
        unique_targets_df = pd.read_csv(not_found_target_file)
        unique_targets = unique_targets_df['target'].values
    else:
        print('Not found not_found_targets file')
        unique_targets = user["target"].unique()
    return food_user, user, unique_targets


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser(description='Process nutrition targets.')
    parser.add_argument('--data', type=str, help='Path to input data file ')
    parser.add_argument('-d', type=int, help='GPU number')
    parser.add_argument('-p', type=int, help='Population size')
    parser.add_argument('-g', type=int, help='generation')
    parser.add_argument('-c', type=float, help='crossover rate')
    parser.add_argument('-m', type=float, help='mutate rate')

    args = parser.parse_args()
    # Record the start time
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}\n")
    if not args.data:
        root_path= 'data/nhanes/data'
    else:
        root_path=args.data
    food_user, user, not_found_target = load_data(root_path)
    print(f'user count: {len(user)}, missing targets: {len(not_found_target)}')
    parsed_targets = [eval(d) for d in not_found_target]
    bronze_df=process_and_save_results(food_user, user, parsed_targets, device=args.d, population_size=args.p, generation=args.g,
                             crossover_rate=args.c,mutate_rate=args.m)
    # Record the end time
    end_time = datetime.datetime.now()
    print(f"End time: {end_time}\n")

    # Calculate the total time taken
    total_time = end_time - start_time
    print(f"completed! Total time taken: {total_time}")
    # post process
    not_found_target = user['target'].drop_duplicates()
    missing_targets = not_found_target[~not_found_target.isin(bronze_df['target'])]
    print(f'Bronze found {bronze_df["target"].nunique()} targets, not found {len(missing_targets)}')
    missing_targets_df = pd.DataFrame(missing_targets, columns=["target"])
    missing_targets_df.to_csv(f'{root_path}/not_found_targets.csv', index=False)
    print(f"updated {root_path}/not_found_targets.csv")
    print('completed!')