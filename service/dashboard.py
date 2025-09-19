from db import *
from utils.target_calculation import TargetNutrition

from utils.constant import ALL_MEDICAL_TAGS, ALL_PREFERENCE_TAGS
import json, pathlib

STATS_FILE = pathlib.Path("stats.json")

def save_stats(stats: dict):
    """覆盖写最新快照"""
    STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

def load_stats():
    """读取最新快照；文件不存在返回空字典"""
    if not STATS_FILE.exists():
        return {}
    return json.loads(STATS_FILE.read_text(encoding="utf-8"))


def state_age_group():
    # 执行查询
    with engine.connect() as conn:
        age_distribution = conn.execute(sql_query_age_group).fetchall()
    result = []

    for row in age_distribution:
        result.append(f" {int(row.age_range)} - {int(row.age_range) + 9}, {row.user_count}")
    return result

def query_meal_plan_count_by_target(targets):
    sql_query_meal_plan = text(f"""
                               SELECT count(*) FROM meal_plan WHERE target = ANY(:targets)
               """)
    with engine.connect() as conn:
        return conn.execute(sql_query_meal_plan ,{'targets': targets}).scalar()

def meal_plan_tag_count(tag):
    sql_query_users_tag = text(f"""
                            SELECT * FROM users WHERE {tag}=1
            """)
    with engine.connect() as conn:
        users = conn.execute(sql_query_users_tag).fetchall()
        users = [user._asdict() for user in users]  # 将每一行转换为字典
        meal_plan_count = 0
        targets = set()
        for user in users:
            targetUser = TargetNutrition(user)
            target = targetUser.get_nutrition_target_range_by_situation()
            targets.add(str(target))
        meal_plan_count= query_meal_plan_count_by_target(list(targets))
        return meal_plan_count

def count_meal_plan():
    medical_counts = {}
    preference_counts = {}
    for tag in ALL_MEDICAL_TAGS:
        medical_counts[tag] = meal_plan_tag_count(tag)
    for tag in ALL_PREFERENCE_TAGS:
        preference_counts[tag] = meal_plan_tag_count(f'user_{tag}')
    return medical_counts, preference_counts

def user_tag_count(tag):
    sql_query_users_tag = text(f"""
            SELECT count(*) FROM users WHERE {tag}=1
            """)
    with engine.connect() as conn:
        return conn.execute(sql_query_users_tag).scalar()
def count_user():
    medical_counts = {}
    preference_counts = {}
    for tag in ALL_MEDICAL_TAGS:
        medical_counts[tag] = user_tag_count(tag)
    for tag in ALL_PREFERENCE_TAGS:
        preference_counts[tag] = user_tag_count(f'user_{tag}')
    return medical_counts, preference_counts


def statistics():
    with engine.connect() as conn:
        count = conn.execute(sql_users).scalar()
        meal_plans_count = conn.execute(sql_meal_plans).scalar()
        gold_meal_plans_count = conn.execute(sql_gold_meal_plans).scalar()
        silver_meal_plans_count = conn.execute(sql_silver_meal_plans).scalar()
        bronze_meal_plans_count = conn.execute(sql_bronze_meal_plans).scalar()
        #targets_count = conn.execute(sql_targets).scalar()
        male_count = conn.execute(sql_male).scalar()
        female_count = conn.execute(sql_female).scalar()
        #top_foods = conn.execute(sql_top_food).fetchall()
    meal_plan_medical_counts, meal_plan_preference_counts = count_meal_plan()
    user_medical_counts, user_preference_counts=count_user()
    stats = {
        'total_users': count,
        'age_group': state_age_group(),
        'meal_plans': meal_plans_count,
        'gold_meal_plans': gold_meal_plans_count,
        'silver_meal_plans': silver_meal_plans_count,
        'bronze_meal_plans': bronze_meal_plans_count,
        #'targets': targets_count,
        'male_users': male_count,
        'female_users': female_count,
        'male_users_ratio': int(male_count / count * 100),
        'female_users_ratio': int(female_count / count * 100),
        'user_medical': user_medical_counts,
        'user_preference': user_preference_counts,
        'meal_plan_medical': meal_plan_medical_counts,
        'meal_plan_preference': meal_plan_preference_counts,

    }
    return stats
