import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================
# 你原来的 特征定义 【完全恢复】
# ============================================
USER_BODY_FEATURES = ['age', 'gender', 'weight', 'height', 'level']
USER_DISEASE_FEATURES = [
    'under_weight', 'over_weight', 'blood_pressure', 'opioid_misuse', 'diabetes', 'anemia', 'blood_urea_nitrogen', 'osteoporosis'
]
USER_PREFERENCE_FEATURES = [
    'user_low_phosphorus', 'user_low_carb', 'user_low_calorie', 'user_high_calorie', 'user_low_sodium', 'user_high_potassium',  'user_low_saturated_fat', 'user_low_cholesterol', 'low_density_lipoprotein', 'user_low_protein', 'user_high_protein', 'user_low_sugar', 'user_high_fiber','user_high_vitamin_b12', 'user_high_folate_acid', 'user_high_iron', 'user_high_vitamin_c', 'user_high_calcium', 'user_high_vitamin_d'
]
MEAL_FEATURES = [
    'calorie', 'protein', 'carb', 'sugar', 'fiber',
    'saturated_fat', 'cholesterol', 'folic_acid', 'vitamin_b12',
    'vitamin_c', 'vitamin_d', 'calcium', 'phosphorus',
    'potassium', 'iron', 'sodium'
]
USER_FEATURES = USER_BODY_FEATURES + USER_DISEASE_FEATURES + USER_PREFERENCE_FEATURES

# ============================================
# 你原来的 预处理函数 【100% 原样恢复】
# ============================================
def preprocess_data(data):
    data['user_id'] = data['user_id'].astype(str)
    if 'gender' in data.columns and data['gender'].dtype == object:
        le = LabelEncoder()
        data['gender'] = le.fit_transform(data['gender'].fillna('unknown').astype(str))

    for col in USER_FEATURES:
        if col in data.columns and col != 'gender':
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col].median())

    for col in MEAL_FEATURES:
        if col in data.columns and col != 'gender':
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col].median())

    if 'weight' in data and 'height' in data:
        data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
        if 'bmi' not in USER_FEATURES:
            USER_FEATURES.append('bmi')

    data['meal_id'] = 'MEAL_' + data.groupby(MEAL_FEATURES).ngroup().astype(str)
    return data, MEAL_FEATURES


class UserKNN:
    def __init__(self, k=10):
        self.k = k
        self.scaler = StandardScaler()

    def fit(self, data, u_features):
        self.user_features = u_features
        self.user_vec = data.groupby('user_id')[self.user_features].first()
        self.user_vec = self.scaler.fit_transform(self.user_vec.values)
        self.user_ids = data.groupby('user_id').first().index.tolist()
        self.user_history = data.groupby('user_id')['meal_id'].apply(list).to_dict()
        self.knn = NearestNeighbors(n_neighbors=self.k, metric='cosine').fit(self.user_vec)

    def recommend(self, uid, topk=20):
        if uid not in self.user_ids:
            return []
        idx = self.user_ids.index(uid)
        _, neighbors = self.knn.kneighbors(self.user_vec[idx].reshape(1, -1))
        recs = []
        for n in neighbors[0]:
            n_uid = self.user_ids[n]
            recs.extend(self.user_history.get(n_uid, []))
        freq = {}
        for m in recs:
            freq[m] = freq.get(m, 0) + 1
        return sorted(freq.keys(), key=lambda x: -freq[x])[:topk]

class ItemKNN:
    def __init__(self, k=10):
        self.k = k
        self.scaler = StandardScaler()

    def fit(self, data, m_features):
        self.meal_features = m_features
        self.meal_vec = data.groupby('meal_id')[self.meal_features].mean()
        self.meal_vec_scaled = self.scaler.fit_transform(self.meal_vec.values)
        self.knn = NearestNeighbors(n_neighbors=self.k, metric='cosine').fit(self.meal_vec_scaled)
        self.user_history = data.groupby('user_id')['meal_id'].apply(list).to_dict()

    def recommend(self, uid, topk=20):
        history = self.user_history.get(uid, [])
        if not history:
            return []
        recs = []
        for mid in history:
            if mid not in self.meal_vec.index:
                continue
            v = self.meal_vec.loc[mid].values.reshape(1, -1)
            v = self.scaler.transform(v)
            _, neighbors = self.knn.kneighbors(v)
            recs.extend(self.meal_vec.index[neighbors[0]])
        freq = {}
        for m in recs:
            freq[m] = freq.get(m, 0) + 1
        return sorted(freq.keys(), key=lambda x: -freq[x])[:topk]


class ContentBased:
    def fit(self, data, m_features):
        self.scaler = StandardScaler()
        self.meal_vec = data.groupby('meal_id')[m_features].mean()
        self.meal_vec_scaled = self.scaler.fit_transform(self.meal_vec.values)
        self.meal_ids = self.meal_vec.index.tolist()
        self.user_history = data.groupby('user_id')['meal_id'].apply(list).to_dict()

    def recommend(self, uid, topk=20):
        history = self.user_history.get(uid, [])
        if not history:
            return []
        # 获取历史餐的索引
        idxs = [self.meal_ids.index(m) for m in history if m in self.meal_ids]
        if not idxs:
            return []
        profile = self.meal_vec_scaled[idxs].mean(axis=0).reshape(1, -1)
        sim = cosine_similarity(profile, self.meal_vec_scaled).flatten()
        ranked = sorted(zip(self.meal_ids, sim), key=lambda x: -x[1])
        return [m for m, _ in ranked[:topk]]


class Hybrid:
    def __init__(self, weights={'cb': 0.4, 'item': 0.4, 'user': 0.2}):
        """
        weights: 各模型的权重，默认ContentBased和ItemKNN各0.4，UserKNN 0.2
        """
        self.weights = weights
        self.cb = ContentBased()
        self.item = ItemKNN()
        self.user = UserKNN()

    def fit(self, data, u_features, m_features):
        """分别用对应的特征训练各子模型"""
        self.cb.fit(data, m_features)
        self.item.fit(data, m_features)
        self.user.fit(data, u_features)

    def recommend(self, uid, topk=20):
        """融合三个模型的推荐结果"""
        # 获取各模型推荐（每个模型多取一些，提高融合质量）
        cb_recs = self.cb.recommend(uid, topk * 2)
        item_recs = self.item.recommend(uid, topk * 2)
        user_recs = self.user.recommend(uid, topk * 2)

        # 加权打分
        scores = {}

        # ContentBased 打分
        for i, meal in enumerate(cb_recs):
            # 位置越靠前，分数越高
            score = self.weights['cb'] * (1 - i / (topk * 2))
            scores[meal] = scores.get(meal, 0) + score

        # ItemKNN 打分
        for i, meal in enumerate(item_recs):
            score = self.weights['item'] * (1 - i / (topk * 2))
            scores[meal] = scores.get(meal, 0) + score

        # UserKNN 打分
        for i, meal in enumerate(user_recs):
            score = self.weights['user'] * (1 - i / (topk * 2))
            scores[meal] = scores.get(meal, 0) + score

        # 按分数排序返回
        sorted_meals = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [meal for meal, _ in sorted_meals[:topk]]

# ============================================
# ✅ 你要求的：标准精确匹配评估（无相似度阈值）
# ============================================
def evaluate(rec_dict, ground_truth, k_list=[5, 10, 20]):
    res = {k: {'precision': 0, 'recall': 0, 'hit_rate': 0, 'ndcg': 0} for k in k_list}
    for k in k_list:
        precs, recalls, hit_rates, ndcgs = [], [], [], []
        for uid in rec_dict:
            rec_items = rec_dict[uid][:k]
            true_items = ground_truth[uid]
            hits = set(rec_items) & set(true_items)

            p = len(hits) / len(rec_items) if rec_items else 0
            r = len(hits) / len(true_items)
            hit = 1 if hits else 0

            dcg = 0.0
            for i, m in enumerate(rec_items):
                if m in true_items:
                    dcg += 1 / np.log2(i + 2)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0

            precs.append(p)
            recalls.append(r)
            hit_rates.append(hit)
            ndcgs.append(ndcg)

        res[k]['precision'] = np.mean(precs)
        res[k]['recall'] = np.mean(recalls)
        res[k]['hit_rate'] = np.mean(hit_rates)
        res[k]['ndcg'] = np.mean(ndcgs)
    return res

# ============================================
# 主流程
# ============================================
def main():
    from utils.data import root_path
    df = pd.read_csv(f'{root_path}/daily_meal_plan_positive.csv')
    df, meal_feats = preprocess_data(df)

    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 2].index.tolist()
    sample_users = np.random.choice(valid_users, 200, replace=False)

    models = {
        "UserKNN": UserKNN(),
        "ItemKNN": ItemKNN(),
        "ContentBased": ContentBased(),
        "Hybrid": Hybrid()  # 添加Hybrid模型
    }

    # 训练各模型
    models["UserKNN"].fit(df, USER_FEATURES)
    models["ItemKNN"].fit(df, MEAL_FEATURES)
    models["ContentBased"].fit(df, MEAL_FEATURES)
    models["Hybrid"].fit(df, USER_FEATURES, MEAL_FEATURES)  # ✅ 传入两个特征

    results = {}
    for name, model in models.items():
        print(f"\n===== {name} =====")
        rec_dict, gt_dict = {}, {}
        for uid in tqdm(sample_users, desc=name):
            meals = df[df.user_id == uid]['meal_id'].unique().tolist()
            test_meal = random.choice(meals)
            rec = model.recommend(uid, topk=20)
            rec_dict[uid] = rec
            gt_dict[uid] = [test_meal]
        results[name] = evaluate(rec_dict, gt_dict)

    # 输出结果
    print("\n" + "=" * 80)
    print(" STANDARD TOP-K RECOMMENDATION RESULTS ".center(80))
    print("=" * 80)

    for k in [5, 10, 20]:
        print(f"\n--- TOP-{k} ---")
        print(f"{'Model':<15} {'Precision':<10} {'Recall':<10} {'Hit Rate':<10} {'NDCG':<10}")
        print("-" * 65)
        for name in models:
            d = results[name][k]
            print(f"{name:<15} {d['precision']:<10.3f} {d['recall']:<10.3f} {d['hit_rate']:<10.3f} {d['ndcg']:<10.3f}")

if __name__ == "__main__":
    main()