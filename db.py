# db.py
import os
from sqlalchemy import create_engine,text
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/nutri"
)
engine = create_engine(DB_URL, pool_pre_ping=True)

sql_users = text(f"""SELECT count(*)    
                        FROM   users
                    """)
sql_user_id = text(f"""SELECT DISTINCT * FROM users WHERE user_id= :userid
                        """)
sql_meal_plans = text(f"""SELECT count(*)    
                        FROM   meal_plan
                    """)
sql_gold_meal_plans = text(f"""SELECT count(*)    
                           FROM   meal_plan
                           where level='gold'
                       """)
sql_silver_meal_plans = text(f"""SELECT count(*)    
                           FROM   meal_plan
                           where level='silver'
                       """)
sql_bronze_meal_plans = text(f"""SELECT count(*)    
                           FROM   meal_plan
                           where level='bronze'
                       """)
sql_targets = text(f"""SELECT count(*)    
                        FROM  targets
                    """)
sql_male = text(f"""SELECT count(*)    
                        FROM   users
                    WHERE  gender=1
                    """)
sql_female = text(f"""SELECT count(*)    
                        FROM   users
                    WHERE  gender=2
                    """)
sql_medical_over_weight = text(f"""SELECT count(*)    
                            FROM   users
                        WHERE  over_weight=1
                        """)
sql_medical_under_weight = text(f"""SELECT count(*)    
                            FROM   users
                        WHERE  under_weight=1
                        """)
sql_medical_opioid_misuse = text(f"""SELECT count(*)    
                            FROM   users
                        WHERE  opioid_misuse=1
                        """)
sql_medical_low_density_lipoprotein = text(f"""SELECT count(*)    
                            FROM   users
                        WHERE  low_density_lipoprotein=1
                        """)
sql_medical_diabetes = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE diabetes=1
                            """)
sql_medical_blood_urea_nitrogen = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE blood_urea_nitrogen=1
                            """)
sql_medical_blood_pressure = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE blood_pressure=1
                            """)
sql_medical_anemia = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE anemia=1
                            """)
sql_medical_osteoporosis = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE osteoporosis=1
                            """)
sql_preference_low_calorie = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_low_calorie=1
                            """)
sql_preference_high_calorie = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_high_calorie=1
                            """)
sql_preference_low_carb = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_low_carb=1
                            """)
sql_preference_high_fiber = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_high_fiber=1
                            """)
sql_preference_low_protein = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_low_protein=1
                            """)
sql_preference_high_protein = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_high_protein=1
                            """)
sql_preference_low_saturated_fat = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_low_saturated_fat=1
                            """)
sql_preference_low_sugar = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_low_sugar=1
                            """)
sql_preference_low_sodium = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_low_sodium=1
                            """)
sql_preference_low_cholesterol = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_low_cholesterol=1
                            """)
sql_preference_low_phosphorus = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_low_phosphorus=1
                            """)
sql_preference_high_potassium = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_high_potassium=1
                            """)
sql_preference_high_folate_acid = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_high_folate_acid=1
                            """)
sql_preference_high_iron = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_high_iron=1
                            """)
sql_preference_high_vitamin_b12 = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_high_vitamin_b12=1
                            """)
sql_preference_high_calcium = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_high_calcium=1
                            """)
sql_preference_high_vitamin_c = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_high_vitamin_c=1
                            """)
sql_preference_high_vitamin_d = text(f"""SELECT count(*)    
                                FROM   users
                            WHERE user_high_vitamin_d=1
                            """)
sql_top_food = text(f"""SELECT fu.food_id food_id,
                             count(*)  count,
                             AVG(fu.grams) avg_grams,
                             fc.food_desc food_name,
                             fc.food_desc_long food_name_long
                             FROM   food_user fu
                             JOIN   food_code fc ON fu.food_id = fc.food_id
                            GROUP BY fu.food_id, fc.food_desc, fc.food_desc_long
                            ORDER  BY count desc
                            limit 10
                   """)
sql_query_detail_w_ids = text(f"""
                SELECT fu.user_id,
                       fu.day,
                       fu.years,
                       fu.food_id,
                       fu.grams,
                       fu.eating_type,
                       fu.daily_food_id,
                       fc.food_desc,
                       fc.food_desc_long
                FROM   food_user fu
                JOIN   food_code fc ON fu.food_id = fc.food_id
                WHERE  fu.daily_food_id = ANY(:ids)
                ORDER  BY fu.daily_food_id,
                          fu.eating_type,
                          fu.food_id
            """)

sql_query_meal_plan_w_target_count =text(f"""
                        SELECT COUNT(DISTINCT  d.daily_food_id)   
                        FROM   meal_plan d
                        WHERE  target IN :targets
                    """)

sql_query_meal_plan_w_target = text(f"""
                        SELECT DISTINCT  d.daily_food_id,
                                 d.calorie,
                                 d.protein,
                                 d.carb,
                                 d.sugar,
                                 d.fiber,
                                 d.saturated_fat,
                                 d.cholesterol,
                                 d.folic_acid,
                                 d.vitamin_b12,
                                 d.vitamin_c,
                                 d.vitamin_d,
                                 d.calcium,
                                 d.phosphorus,
                                 d.potassium,
                                 d.iron,
                                 d.sodium,
                                 d.level       
                        FROM   meal_plan d
                        WHERE  target IN :targets
                        LIMIT 20
                    """)
sql_query_age_group = text("""
    SELECT 
        FLOOR(age / 10) * 10 AS age_range,
        COUNT(*) AS user_count
    FROM 
        users
    GROUP BY 
        age_range
    ORDER BY 
        age_range;
""")

sql_food_code = text(f"""
                SELECT DISTINCT f_user.food_id, f_code.food_desc_long 
                FROM food_user f_user
                JOIN food_code f_code ON f_user.food_id =f_code.food_id
                ORDER BY food_id 
                LIMIT :batch_size OFFSET :offset
                """)
sql_update_food_desc_long_id =text(f"""
                UPDATE food_code SET food_desc_long = :translated_desc WHERE food_id =:food_id
                """)

sql_query_duplicate_meal_plan = text("""
WITH food_id_lists AS (
    SELECT
        mp.daily_food_id,
        STRING_AGG(f.food_id || '|' || f.grams::text, ',') AS food_id_list
    FROM meal_plan mp
    JOIN food_user f ON mp.daily_food_id = f.daily_food_id::numeric
    GROUP BY
        mp.daily_food_id
),
duplicates AS (
    SELECT
        food_id_list,
        ARRAY_AGG(daily_food_id) AS daily_food_ids
    FROM
        food_id_lists
    GROUP BY
        food_id_list
    HAVING
        COUNT(daily_food_id) > 1
)
SELECT
    food_id_list,
    daily_food_ids
FROM
    duplicates
ORDER BY
    food_id_list;
""")

sql_delete_duplicate_meal_plans=text(f"""WITH duplicate_meal_plans AS (
    SELECT
        daily_food_id,
        target,
        ROW_NUMBER() OVER (PARTITION BY daily_food_id, target ORDER BY daily_food_id) AS row_num
    FROM
        meal_plan
)
DELETE FROM meal_plan
WHERE (daily_food_id, target) IN (
    SELECT daily_food_id, target
    FROM duplicate_meal_plans
    WHERE row_num > 1
)""")

sql_query_target_w_target=text(f"""
                SELECT DISTINCT t.target
                FROM   targets t
                JOIN meal_plan mp ON mp.target=t.target
                WHERE mp.daily_food_id='21037202510'
""")

