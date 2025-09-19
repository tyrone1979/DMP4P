from sqlalchemy import text
from db import engine,sql_query_meal_plan_w_target_count  # 直接导入全局 engine
from utils.target_calculation import TargetNutrition


def query_meal_plan_count(targe_range):
    conditions = []
    params = {}
    for nutrient, (min_val, max_val) in targe_range.items():
        conditions.append(f"{nutrient}_min >= :{nutrient}_min_val")
        conditions.append(f"{nutrient}_max <= :{nutrient}_max_val")
        params[f"{nutrient}_min_val"] = min_val
        params[f"{nutrient}_max_val"] = max_val

    sql_query_target_range = text(f"""
                    SELECT DISTINCT target
                    FROM   targets
                    WHERE  {' AND '.join(conditions)}
                """)
    with engine.connect() as conn:
        target_rows = conn.execute(sql_query_target_range, params).fetchall()

    if target_rows:
        # Extract unique target values
        unique_targets = list(set(row[0] for row in target_rows))
        # Step 2: For each unique target, search in daily_food_with_nutrition_target_{level} tables
        with engine.connect() as conn:
            row_count = conn.execute(sql_query_meal_plan_w_target_count, {'targets': tuple(unique_targets)}).scalar()
        return row_count
    # If no matches found in any level
    return 0


def query_users(medical_tags, preference_tags):
    query_condition = ''

    # 把选中的标签置 1
    for tag in medical_tags:
        query_condition += f' AND {tag}=1'

    for tag in preference_tags:
        query_condition += f' AND user_{tag}=1'

    sql_users_details_top_10 = text(f"""SELECT 
                        u.*
                    FROM users u 
                    WHERE 1=1
                    {query_condition}
                    LIMIT 20 ;
                     """)
    with engine.connect() as conn:
        users = conn.execute(sql_users_details_top_10).fetchall()
        users = [user._asdict() for user in users]  # 将每一行转换为字典
        for user_info in users:
            targetUser = TargetNutrition(user_info)
            user_info['target'] = targetUser.get_nutrition_target_range_by_situation()
            user_info['meal_plan_count'] = query_meal_plan_count(user_info['target'])
    return users

from db import sql_user_id
from service.mealplan import find_matching_daily_food,get_food_details
def query_target_and_meal_plan(userid):
    with engine.connect() as conn:
        row = conn.execute(sql_user_id, {'userid': userid}).first()
        user_info=row._asdict()
    targetUser = TargetNutrition(user_info)
    target_range = targetUser.get_nutrition_target_range_by_situation()
    target = targetUser.get_nutrition_target_by_situation()
    meal_plan = find_matching_daily_food(target_range)
    details=[]
    if meal_plan:
        details = get_food_details(meal_plan)
    return target_range,target,meal_plan,details