import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, ttest_ind
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 疾病标签列表
# ============================================

DISEASE_COLS = [
    'under_weight',  # 体重不足
    'over_weight',  # 超重/肥胖
    'blood_pressure',  # 高血压
    'diabetes',  # 糖尿病
    'anemia',  # 贫血
    'osteoporosis',  # 骨质疏松
    'opioid_misuse',  # 阿片类药物滥用
    'blood_urea_nitrogen'  # 高血尿素氮（肾功能）
]

# 营养素列表
NUTRIENT_COLS = [
    'calorie', 'protein', 'carb', 'sugar', 'fiber',
    'saturated_fat', 'cholesterol', 'sodium', 'potassium',
    'phosphorus', 'iron', 'calcium', 'folic_acid',
    'vitamin_c', 'vitamin_d', 'vitamin_b12'
]

# 营养偏好标签
PREFERENCE_COLS = [
    'user_low_calorie', 'user_high_calorie', 'user_low_sugar',
    'user_high_fiber', 'user_low_sodium', 'user_high_potassium',
    'user_low_saturated_fat', 'user_low_cholesterol',
    'user_low_protein', 'user_high_protein'
]

print("=" * 70)
print("疾病与营养素关联性分析（原始 vs 优化膳食）")
print("=" * 70)


# ============================================
# 数据准备
# ============================================

def prepare_dataset(data, dataset_name):
    """准备分析数据"""
    # 只使用正例（如果有 label 列）
    if 'label' in data.columns:
        data = data[data['label'] == 1].copy()

    # 确保所有列存在
    for col in DISEASE_COLS:
        if col not in data.columns:
            data[col] = 0

    for col in NUTRIENT_COLS:
        if col not in data.columns:
            data[col] = 0

    print(f"\n{dataset_name} 数据规模: {len(data)} 条记录")

    return data


# ============================================
# 点二列相关系数分析
# ============================================

def point_biserial_correlation(data, dataset_name):
    """计算疾病与营养素之间的点二列相关系数"""

    print(f"\n{'=' * 60}")
    print(f"{dataset_name} - 点二列相关系数分析")
    print(f"{'=' * 60}")

    correlations = []

    for disease in DISEASE_COLS:
        for nutrient in NUTRIENT_COLS:
            x = data[disease].values
            y = data[nutrient].values

            try:
                corr, p_value = pointbiserialr(x, y)
                correlations.append({
                    'Disease': disease,
                    'Nutrient': nutrient,
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05
                })
            except:
                pass

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)

    print(f"\nTop 10 最强关联:")
    print(corr_df.head(10).to_string(index=False))

    return corr_df


# ============================================
# 患病/非患病群体营养素差异分析
# ============================================

def nutrient_difference_analysis(data, dataset_name):
    """分析患病与未患病群体的营养素差异"""

    print(f"\n{'=' * 60}")
    print(f"{dataset_name} - 患病/非患病群体营养素差异分析")
    print(f"{'=' * 60}")

    results = []

    for disease in DISEASE_COLS:
        diseased = data[data[disease] == 1]
        non_diseased = data[data[disease] == 0]

        if len(diseased) < 30 or len(non_diseased) < 30:
            continue

        for nutrient in NUTRIENT_COLS:
            t_stat, p_value = ttest_ind(
                diseased[nutrient].dropna(),
                non_diseased[nutrient].dropna()
            )

            mean_diff = diseased[nutrient].mean() - non_diseased[nutrient].mean()
            pooled_std = np.sqrt((diseased[nutrient].std() ** 2 + non_diseased[nutrient].std() ** 2) / 2)
            cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0

            results.append({
                'Disease': disease,
                'Nutrient': nutrient,
                'Diseased_Mean': diseased[nutrient].mean(),
                'Non_Diseased_Mean': non_diseased[nutrient].mean(),
                'Difference': mean_diff,
                'P_Value': p_value,
                'Cohen_d': cohen_d,
                'Significant': p_value < 0.05
            })

    results_df = pd.DataFrame(results)
    results_df = results_df[results_df['Significant'] == True]
    results_df = results_df.sort_values('Cohen_d', key=abs, ascending=False)

    print(f"\n显著差异的疾病-营养素对: {len(results_df)}")
    print("\nTop 10 最大差异 (效应量):")
    print(results_df.head(10).to_string(index=False))

    return results_df


# ============================================
# 汇总报告
# ============================================

def generate_comparison_report(original_corr, original_diff, positive_corr, positive_diff):
    """生成原始数据 vs 优化数据的对比报告"""

    print("\n" + "=" * 70)
    print("对比分析报告：原始膳食 vs 优化膳食")
    print("=" * 70)

    # 计算平均绝对相关系数
    original_avg_corr = original_corr['Correlation'].abs().mean()
    positive_avg_corr = positive_corr['Correlation'].abs().mean()

    print(f"\n📊 平均绝对相关系数:")
    print(f"   原始膳食: {original_avg_corr:.4f}")
    print(f"   优化膳食: {positive_avg_corr:.4f}")
    print(f"   提升幅度: {(positive_avg_corr - original_avg_corr) / original_avg_corr * 100:.1f}%")

    # 统计显著相关的数量
    original_sig = len(original_corr[original_corr['P_Value'] < 0.05])
    positive_sig = len(positive_corr[positive_corr['P_Value'] < 0.05])

    print(f"\n📊 显著相关对数量 (p<0.05):")
    print(f"   原始膳食: {original_sig}")
    print(f"   优化膳食: {positive_sig}")

    # 统计显著差异的疾病-营养素对
    original_diff_count = len(original_diff)
    positive_diff_count = len(positive_diff)

    print(f"\n📊 显著差异对数量 (疾病 vs 营养素):")
    print(f"   原始膳食: {original_diff_count}")
    print(f"   优化膳食: {positive_diff_count}")

    # 找出代表性差异
    print(f"\n📌 代表性发现:")

    # 骨质疏松 - 维生素D
    original_osteo = original_corr[(original_corr['Disease'] == 'osteoporosis') &
                                   (original_corr['Nutrient'] == 'vitamin_d')]
    positive_osteo = positive_corr[(positive_corr['Disease'] == 'osteoporosis') &
                                   (positive_corr['Nutrient'] == 'vitamin_d')]

    if len(original_osteo) > 0 and len(positive_osteo) > 0:
        print(f"\n   骨质疏松 ↔ 维生素D:")
        print(f"     原始: r={original_osteo['Correlation'].values[0]:.4f}")
        print(f"     优化: r={positive_osteo['Correlation'].values[0]:.4f}")

    # 高血压 - 钠
    original_hbp = original_corr[(original_corr['Disease'] == 'blood_pressure') &
                                 (original_corr['Nutrient'] == 'sodium')]
    positive_hbp = positive_corr[(positive_corr['Disease'] == 'blood_pressure') &
                                 (positive_corr['Nutrient'] == 'sodium')]

    if len(original_hbp) > 0 and len(positive_hbp) > 0:
        print(f"\n   高血压 ↔ 钠:")
        print(f"     原始: r={original_hbp['Correlation'].values[0]:.4f}")
        print(f"     优化: r={positive_hbp['Correlation'].values[0]:.4f}")

    # 糖尿病 - 碳水化合物
    original_diab = original_corr[(original_corr['Disease'] == 'diabetes') &
                                  (original_corr['Nutrient'] == 'carb')]
    positive_diab = positive_corr[(positive_corr['Disease'] == 'diabetes') &
                                  (positive_corr['Nutrient'] == 'carb')]

    if len(original_diab) > 0 and len(positive_diab) > 0:
        print(f"\n   糖尿病 ↔ 碳水化合物:")
        print(f"     原始: r={original_diab['Correlation'].values[0]:.4f}")
        print(f"     优化: r={positive_diab['Correlation'].values[0]:.4f}")


# ============================================
# 对比可视化
# ============================================

def plot_comparison_heatmaps(original_corr, positive_corr):
    """并排绘制两个热力图进行对比"""

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 原始数据热力图
    original_matrix = original_corr.pivot(index='Disease', columns='Nutrient', values='Correlation')
    sns.heatmap(original_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-0.8, vmax=0.8, ax=axes[0],
                cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Original NHANES Diet (Weak Association)', fontsize=14)
    axes[0].set_xlabel('Nutrients', fontsize=10)
    axes[0].set_ylabel('Diseases', fontsize=10)
    axes[0].tick_params(axis='x', rotation=45)

    # 优化数据热力图
    positive_matrix = positive_corr.pivot(index='Disease', columns='Nutrient', values='Correlation')
    sns.heatmap(positive_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-0.8, vmax=0.8, ax=axes[1],
                cbar_kws={'label': 'Correlation'})
    axes[1].set_title('DMP4P Optimized Diet (Strong Association)', fontsize=14)
    axes[1].set_xlabel('Nutrients', fontsize=10)
    axes[1].set_ylabel('Diseases', fontsize=10)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('disease_nutrient_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================
# 主函数
# ============================================

def main():
    """主函数"""

    from utils.data import root_path

    print("\n加载数据...")

    # 加载三个数据集
    original_samples = pd.read_csv(f'{root_path}daily_meal_plan_original.csv')
    positive_samples = pd.read_csv(f'{root_path}daily_meal_plan_positive.csv')

    print(f"原始数据大小: {len(original_samples)}")
    print(f"正样本大小: {len(positive_samples)}")

    # 准备分析数据
    original_data = prepare_dataset(original_samples, "原始NHANES膳食")
    positive_data = prepare_dataset(positive_samples, "DMP4P优化膳食")

    # ============================================
    # 分析原始数据（预期：弱关联）
    # ============================================
    print("\n" + "=" * 70)
    print("第一部分：原始NHANES膳食分析")
    print("=" * 70)

    original_corr = point_biserial_correlation(original_data, "原始NHANES膳食")
    original_diff = nutrient_difference_analysis(original_data, "原始NHANES膳食")

    # ============================================
    # 分析优化数据（预期：强关联）
    # ============================================
    print("\n" + "=" * 70)
    print("第二部分：DMP4P优化膳食分析")
    print("=" * 70)

    positive_corr = point_biserial_correlation(positive_data, "DMP4P优化膳食")
    positive_diff = nutrient_difference_analysis(positive_data, "DMP4P优化膳食")

    # ============================================
    # 对比分析
    # ============================================
    generate_comparison_report(original_corr, original_diff, positive_corr, positive_diff)

    # ============================================
    # 可视化对比
    # ============================================
    plot_comparison_heatmaps(original_corr, positive_corr)

    print("\n" + "=" * 70)
    print("✅ 分析完成")
    print("=" * 70)

    return original_corr, positive_corr, original_diff, positive_diff


if __name__ == "__main__":
    original_corr, positive_corr, original_diff, positive_diff = main()