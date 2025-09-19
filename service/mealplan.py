from sqlalchemy import text
from db import engine,sql_query_detail_w_ids,sql_query_meal_plan_w_target  # 直接导入全局 engine
from utils.target_calculation import TargetNutrition

EATING_TYPE_MAP = {
    1: 'Breakfast',
    2: 'Lunch',
    3: 'Dinner',
    4: 'Supper',
    5: 'Brunch',
    6: 'Snack',
    7: 'Drink',
    8: 'Infant',
    9: 'Extended consumption'
}

def get_food_details(meal_plans):
    """
    :param daily_food_ids: list of daily_food_id
    :return: list of dictionaries containing food details
    """
    """
    根据 daily_food_ids 列表，实时去 food_user + food_code 里查详情
    返回 list[dict]，按 daily_food_id, eating_type 升序
    """
    details = []
    ids_list = [x['daily_food_id'] for x in meal_plans]
    with engine.connect() as conn:
        rows = conn.execute(sql_query_detail_w_ids, {'ids': ids_list}).fetchall()
        # 组装成 dict 并翻译 eating_type
        for r in rows:
            eating_type = EATING_TYPE_MAP.get(r.eating_type, 'Unknown')
            details.append({
                'user_id': r.user_id,
                'day': r.day,
                'years': r.years,
                'food_id': r.food_id,
                'grams': r.grams,
                'eating_type': eating_type,
                'food_desc': r.food_desc,
                'food_desc_long': r.food_desc_long,
                'daily_food_id': str(int(r.daily_food_id))
            })
    return details


def find_matching_daily_food(target):
    """
    先在 targets 表中查找符合 target 范围的 target 值，然后在对应的 daily_food_with_nutrition_target_{level} 表中查找 daily_food_id
    target : dict  {nutrient: [min, max], ...}
    return : list of daily_food_id, level
    """
    # Step 1: Find matching target in targets table
    conditions = []
    params = {}
    for nutrient, (min_val, max_val) in target.items():
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
            daily_rows = conn.execute(sql_query_meal_plan_w_target, {'targets': tuple(unique_targets)}).fetchall()
            # Store the results in the dictionary under the corresponding level key
            results = [
                {**row._asdict(), 'daily_food_id': row[0]}
                for row in daily_rows
            ]
        return results
    # If no matches found in any level
    return []

def get_all(user_info):
    ctx = {}
    targetUser = TargetNutrition(user_info)
    target_range = targetUser.get_nutrition_target_range_by_situation()
    target = targetUser.get_nutrition_target_by_situation()
    meal_plan = find_matching_daily_food(target_range)
    ctx.update(target=target, target_range=target_range, user_info=user_info, meal_plan=meal_plan)
    if not meal_plan:
        ctx.update(error='meal plan not found')
    else:
        details = get_food_details(meal_plan)
        ctx.update(details=details)
    return ctx