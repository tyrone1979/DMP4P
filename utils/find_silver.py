import torch
import pandas as pd
from tqdm import tqdm

def search_match_food_torch(target_users, foods):
    """
    find match food for target users
    Improve efficiency by using torch tensor
    :param target_users:
    :param foods:
    :return: found foods dataframe and not found user dataframe
    """
    # 将 Pandas DataFrame 转换为 PyTorch 张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 预处理用户数据
    user_ages = torch.from_numpy(target_users['age'].values).to(device).float()

    # 构建公共的营养目标维度（所有用户可能涉及的营养素）
    parsed_targets = [eval(d) for d in target_users['target']]

    all_nutrients = set()
    for user_target in parsed_targets:
        all_nutrients.update(user_target.keys())
    all_nutrients = sorted(list(all_nutrients))

    # 将所有用户的目标转换为张量，并填充缺失的营养目标
    user_targets_lower = []
    user_targets_upper = []
    for _, row in target_users.iterrows():
        user_target = eval(row['target'])
        lower = []
        upper = []
        for nutrient in all_nutrients:
            if nutrient in user_target:
                lower.append(user_target[nutrient][0])
                upper.append(user_target[nutrient][1])
            else:
                lower.append(0.0)  # 默认下限
                upper.append(float('inf'))  # 默认上限（表示无上限）
        user_targets_lower.append(lower)
        user_targets_upper.append(upper)

    user_targets_lower = torch.tensor(user_targets_lower, dtype=torch.float32).to(device)
    user_targets_upper = torch.tensor(user_targets_upper, dtype=torch.float32).to(device)

    # 预处理食物数据，确保与用户目标的营养维度一致
    food_nutrients_list = []
    for _, food_row in foods.iterrows():
        food_nutrients = []
        for nutrient in all_nutrients:
            if nutrient in food_row:
                food_nutrients.append(food_row[nutrient])
            else:
                food_nutrients.append(0.0)  # 如果食物数据中缺少某些营养素，填充为0
        food_nutrients_list.append(food_nutrients)

    food_nutrients = torch.tensor(food_nutrients_list, dtype=torch.float32).to(device)
    food_age_groups = torch.from_numpy(foods['age_group'].values).to(device).float()

    # 创建匹配结果张量
    found_mask = torch.zeros(target_users.shape[0], device=device, dtype=torch.bool)

    # 定义营养匹配条件检查张量化函数
    def check_nutrition_match_torch(user_target_lower, user_target_upper, food_nutrients):
        lower_match = food_nutrients >= user_target_lower.unsqueeze(0)
        upper_match = food_nutrients <= user_target_upper.unsqueeze(0)
        return torch.all(torch.logical_and(lower_match, upper_match), dim=1)

    # 将食物数据按年龄分组
    senior_food_idx = torch.where(food_age_groups == 7.0)[0]
    non_senior_food_idx = torch.where(food_age_groups != 7.0)[0]

    # 处理每个用户
    matched_food_indices = torch.full((target_users.shape[0],), -1, device=device, dtype=torch.long)
    with tqdm(total=target_users.shape[0], desc="Processing Users", unit="user") as pbar:
        for i in range(target_users.shape[0]):
            age = user_ages[i]
            lower = user_targets_lower[i]
            upper = user_targets_upper[i]

            matched_food_idx = -1  # 默认值，表示未找到匹配
            if age > 60:
                if senior_food_idx.numel() > 0:
                    matches = check_nutrition_match_torch(
                        lower,
                        upper,
                        food_nutrients[senior_food_idx]
                    )
                    if matches.any():
                        found_mask[i] = True
                        matched_food_idx = senior_food_idx[torch.where(matches)[0][0]]  # 获取第一个匹配的食物索引
            else:
                if non_senior_food_idx.numel() > 0:
                    matches = check_nutrition_match_torch(
                        lower,
                        upper,
                        food_nutrients[non_senior_food_idx]
                    )
                    if matches.any():
                        found_mask[i] = True
                        matched_food_idx = non_senior_food_idx[torch.where(matches)[0][0]]  # 获取第一个匹配的食物索引

            # 将匹配的食物索引存储到张量中
            matched_food_indices[i] = matched_food_idx

            pbar.update(1)  # 更新进度条

    # 收集结果
    found_idx = found_mask.nonzero().squeeze()
    not_found_idx = (~found_mask).nonzero().squeeze()

    found_df = target_users.iloc[found_idx.cpu().numpy()] if len(found_idx) > 0 else pd.DataFrame()
    not_found_df = target_users.iloc[not_found_idx.cpu().numpy()] if len(not_found_idx) > 0 else pd.DataFrame()

    # 将匹配的食物信息与用户信息合并
    if not found_df.empty:
        # 获取匹配的食物索引（过滤掉-1的索引，表示未找到匹配）
        valid_indices = found_idx[matched_food_indices[found_idx] != -1]
        valid_food_indices = matched_food_indices[found_idx][matched_food_indices[found_idx] != -1]

        # 获取匹配的食物数据
        matched_foods = foods.iloc[valid_food_indices.cpu().numpy()]

        # 将用户数据和匹配的食物数据合并
        found_df = pd.concat([found_df.reset_index(drop=True), matched_foods.reset_index(drop=True)], axis=1)

    return found_df, not_found_df


class BaseHandler:
    def __init__(self):
        # 初始化和运行管道
        data_path = '../food_user.csv'
        self.data = pd.read_csv(data_path)
        food_code = pd.read_csv('../food_code.csv')
        food_code=food_code.drop(columns=['years','food_desc_long'])
        # 将食物计数与食物代码表合并
        self.data = pd.merge(self.data, food_code, on='food_id', how='left')

    def getData(self):
        return self.data

    def print(self):
        print(self.data)


class FoodHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        # 统计每个用户吃每种食物的次数
        self.food_counts = self.data.groupby(['food_id','food_desc']).size().reset_index(name='count').sort_values(by='count',
                                                                                                       ascending=False)

    def get_eating_type_by_food_id(self, food_id):
        # 筛选与指定 food_id 匹配的所有记录
        matching_rows = self.data[self.data['food_id'] == food_id]
        matching_rows = matching_rows.drop_duplicates(subset='eating_type')
        matching_rows = matching_rows[matching_rows['eating_type'] < 20].sort_values(by='eating_type')
        # 提取 eating_type 列并转换为列表
        eating_types = matching_rows['eating_type'].tolist()
        return eating_types

    def top_k_favorite_food(self, k):
        # 获取最受欢迎的前 k 种食物
        # 确保 k 不超过食物种类的数量
        k = min(k, len(self.food_counts))
        # 获取前 k 种最受欢迎的食物
        top_k_food = self.food_counts.head(k)
        return top_k_food[['food_id', 'food_desc', 'count']]


class UserHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        # 统计每个用户吃每种食物的次数
        self.user_food_counts = self.data.groupby(['SEQN', 'food_id','food_desc']).size().reset_index(name='count')

    def get_favorite_food_by_user_id(self, user_id):
        # 筛选与指定 food_id 匹配的所有记录
        user_rows = self.user_food_counts[self.user_food_counts['SEQN'] == user_id]
        # 按 count 降序排序
        sorted_rows = user_rows.sort_values(by='count', ascending=False)
        # 返回排序后的结果
        return sorted_rows

