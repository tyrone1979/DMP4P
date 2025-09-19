# Setup Guidance
This setup guidance contains two parts.
- Part 1: Data generation guidance
- Part 2: Rational database table Transform guidance

## Prepare environment
### operation system
- Ubuntu 24.04.6 LTS (GNU/Linux 5.4.0-205-generic x86_64) Tested

### software
- python >3.11.4
- libpq-dev
- postgresql


## Part 1. Data generation guidance
### preprocess
1. enter root path 
### Python package
- refer to [requirements.txt](requirements.txt)

### Create virtual environment
1. use virtualenv
```commandline
virtualenv meal
source meal/bin/activate
```

2. or use conda
```commandline
conda create --name meal python=3.11.4
conda activate meal
```

### run jupyterlab
```commandline
~/meal-rec-ai/jupyter-notebook
```

2. make users.csv data

```commandline
run 1_user_tagging.ipynb cells
```

3. make food_code,food_nutrition, food_user data
```commandline
run 2_food_tagging.ipynb cells
```

4. make pa(physical activity), supplement data
```commandline
run 3_Dietary_Supplement_and_Physical_Activity.ipynb cells
```

5. make daily nutrition data
```commandline
run 4_user_food_merge_statistics.ipynb cells
```
6. make daily_food_with_nutrition_target with gold,silver,bronze level
- Run find_bronze batch first time
```commandline
nohup python find_bronze.py --data='data/nhanes/data' -p=1000 -d=0 -g=300 -c=0.2 -m=0.9 &
```
- Run find_bronze batch next time
```commandline
nohup python find_bronze.py --data='data/nhanes/data' --targets='not_found_targets.csv' -p=2000 -d=0 -g=300 -c=0.2 -m=0.9 &
```
- --data: input data path, default: ../data/nhanes
- --targets: target data file, default: {input data path}/not_found_targets.csv
- -p: population size
- -d: GPU number single GPU support only
- -g: generations
- -c: crossover ratio
- -m: mutation ratio

## Part 2. Rational database table Transform guidance
### create database nutri
```commandline
sudo -u postgres psql
```
```sql
ALTER USER postgres WITH PASSWORD 'postgres';
CREATE DATABASE nutri WITH OWNER = postgres;
```

### create schema

```commandline
/meal# psql -U postgres -d nutri -f ./nutri_schema.sql
```

### import csv into nutri database
1. :~/meal# cd data/nhanes/data
2. run import commands
```commandline
sudo -u postgres psql -d nutri \
     -c "\copy food_code(food_id,food_desc,food_desc_long,years,positive,negative) FROM 'food_code.csv' CSV HEADER"
sudo -u postgres psql -d nutri \
     -c "\copy food_user(food_id,user_id,eating_type,grams,day,years,daily_food_id) FROM 'D:/projects/meal-rec-ai/data/nhanes/data/food_user.csv' CSV HEADER"
sudo -u postgres psql -d nutri \
     -c "\copy food_user(daily_food_id,food_id,user_id,eating_type,grams,day,years) FROM 'D:/projects/meal-rec-ai/data/nhanes/data/food_user_bronze.csv' CSV HEADER"
sudo -u postgres psql -d nutri \
     -c "\copy daily_food_with_nutrition_target_gold(user_id,age_group,gender,grams,calorie,protein,carb,sugar,fiber,saturated_fat,cholesterol,folic_acid,vitamin_b12,vitamin_c,vitamin_d,calcium,phosphorus,potassium,iron,sodium,age,user_low_phosphorus,user_low_carb,weight,height,under_weight,over_weight,user_low_calorie,user_high_calorie,user_low_sodium,user_high_potassium,blood_pressure,user_low_saturated_fat,user_low_cholesterol,low_density_lipoprotein,blood_urea_nitrogen,user_low_protein,user_high_protein,opioid_misuse,diabetes,user_low_sugar,user_high_fiber,anemia,user_high_vitamin_b12,user_high_folate_acid,user_high_iron,user_high_vitamin_c,user_high_calcium,user_high_vitamin_d,osteoporosis,level,b_calorie,b_carb,b_fiber,b_protein,b_saturated_fat,b_sugar,b_cholesterol,macro_health_score,b_sodium,b_phosphorus,b_potassium,b_iron,b_calcium,b_folic_acid,b_vitamin_c,b_vitamin_d,b_vitamin_b12,micro_health_score,match,total_positive_score,total_negative_score,daily_food_id,target) FROM 'daily_food_with_nutrition_target_gold.csv' CSV HEADER"    
sudo -u postgres psql -d nutri \
     -c "\copy daily_food_with_nutrition_target_silver(user_id,gender,age,user_low_phosphorus,user_low_carb,weight,height,under_weight,over_weight,user_low_calorie,user_high_calorie,user_low_sodium,user_high_potassium,blood_pressure,user_low_saturated_fat,user_low_cholesterol,low_density_lipoprotein,blood_urea_nitrogen,user_low_protein,user_high_protein,opioid_misuse,diabetes,user_low_sugar,user_high_fiber,anemia,user_high_vitamin_b12,user_high_folate_acid,user_high_iron,user_high_vitamin_c,user_high_calcium,user_high_vitamin_d,osteoporosis,level,target,age_group,grams,calorie,protein,carb,sugar,fiber,saturated_fat,cholesterol,folic_acid,vitamin_b12,vitamin_c,vitamin_d,calcium,phosphorus,potassium,iron,sodium,b_calorie,b_carb,b_fiber,b_protein,b_saturated_fat,b_sugar,b_cholesterol,b_sodium,b_phosphorus,b_potassium,b_iron,b_calcium,b_folic_acid,b_vitamin_c,b_vitamin_d,b_vitamin_b12,match,total_positive_score,total_negative_score,daily_food_id,macro_health_score,micro_health_score) FROM 'daily_food_with_nutrition_target_silver.csv' CSV HEADER"         
sudo -u postgres psql -d nutri \
     -c "\copy daily_food_with_nutrition_target_bronze(daily_food_id,target,user_id,grams,calorie,protein,carb,sugar,fiber,saturated_fat,cholesterol,folic_acid,vitamin_b12,vitamin_c,vitamin_d,calcium,phosphorus,potassium,iron,sodium,age_group,gender,age,user_low_phosphorus,user_low_carb,weight,height,under_weight,over_weight,user_low_calorie,user_high_calorie,user_low_sodium,blood_pressure,user_high_potassium,user_low_saturated_fat,user_low_cholesterol,low_density_lipoprotein,blood_urea_nitrogen,user_low_protein,user_high_protein,opioid_misuse,user_low_sugar,user_high_fiber,diabetes,user_high_folate_acid,user_high_iron,user_high_vitamin_b12,anemia,osteoporosis,user_high_calcium,user_high_vitamin_c,user_high_vitamin_d,level,match) FROM 'daily_food_with_nutrition_target_bronze.csv' CSV HEADER"     
```

### convert target column to targets table
```SQL
INSERT INTO targets (
    target,
    calorie_min, calorie_max,
    protein_min, protein_max,
    saturated_fat_min, saturated_fat_max,
    carb_min, carb_max,
    fiber_min, fiber_max,
    sugar_min, sugar_max,
    cholesterol_min, cholesterol_max,
    sodium_min, sodium_max,
    calcium_min, calcium_max,
    phosphorus_min, phosphorus_max,
    potassium_min, potassium_max,
    iron_min, iron_max,
    folic_acid_min, folic_acid_max,
    vitamin_c_min, vitamin_c_max,
    vitamin_d_min, vitamin_d_max,
    vitamin_b12_min, vitamin_b12_max
)
SELECT DISTINCT
    target AS target,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calorie'->>0, 'null'), '-1')::numeric AS calorie_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calorie'->>1, 'null'), '-1')::numeric AS calorie_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'protein'->>0, 'null'), '-1')::numeric AS protein_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'protein'->>1, 'null'), '-1')::numeric AS protein_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'saturated_fat'->>0, 'null'), '-1')::numeric AS saturated_fat_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'saturated_fat'->>1, 'null'), '-1')::numeric AS saturated_fat_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'carb'->>0, 'null'), '-1')::numeric AS carb_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'carb'->>1, 'null'), '-1')::numeric AS carb_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'fiber'->>0, 'null'), '-1')::numeric AS fiber_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'fiber'->>1, 'null'), '-1')::numeric AS fiber_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sugar'->>0, 'null'), '-1')::numeric AS sugar_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sugar'->>1, 'null'), '-1')::numeric AS sugar_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'cholesterol'->>0, 'null'), '-1')::numeric AS cholesterol_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'cholesterol'->>1, 'null'), '-1')::numeric AS cholesterol_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sodium'->>0, 'null'), '-1')::numeric AS sodium_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sodium'->>1, 'null'), '-1')::numeric AS sodium_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calcium'->>0, 'null'), '-1')::numeric AS calcium_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calcium'->>1, 'null'), '-1')::numeric AS calcium_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'phosphorus'->>0, 'null'), '-1')::numeric AS phosphorus_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'phosphorus'->>1, 'null'), '-1')::numeric AS phosphorus_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'potassium'->>0, 'null'), '-1')::numeric AS potassium_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'potassium'->>1, 'null'), '-1')::numeric AS potassium_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'iron'->>0, 'null'), '-1')::numeric AS iron_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'iron'->>1, 'null'), '-1')::numeric AS iron_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'folic_acid'->>0, 'null'), '-1')::numeric AS folic_acid_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'folic_acid'->>1, 'null'), '-1')::numeric AS folic_acid_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_c'->>0, 'null'), '-1')::numeric AS vitamin_c_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_c'->>1, 'null'), '-1')::numeric AS vitamin_c_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_d'->>0, 'null'), '-1')::numeric AS vitamin_d_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_d'->>1, 'null'), '-1')::numeric AS vitamin_d_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_b12'->>0, 'null'), '-1')::numeric AS vitamin_b12_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_b12'->>1, 'null'), '-1')::numeric AS vitamin_b12_max
FROM daily_food_with_nutrition_target_gold
WHERE target IS NOT NULL
ON CONFLICT (target) DO NOTHING;

INSERT INTO targets (
    target,
    calorie_min, calorie_max,
    protein_min, protein_max,
    saturated_fat_min, saturated_fat_max,
    carb_min, carb_max,
    fiber_min, fiber_max,
    sugar_min, sugar_max,
    cholesterol_min, cholesterol_max,
    sodium_min, sodium_max,
    calcium_min, calcium_max,
    phosphorus_min, phosphorus_max,
    potassium_min, potassium_max,
    iron_min, iron_max,
    folic_acid_min, folic_acid_max,
    vitamin_c_min, vitamin_c_max,
    vitamin_d_min, vitamin_d_max,
    vitamin_b12_min, vitamin_b12_max
)
SELECT DISTINCT
    target AS target,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calorie'->>0, 'null'), '-1')::numeric AS calorie_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calorie'->>1, 'null'), '-1')::numeric AS calorie_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'protein'->>0, 'null'), '-1')::numeric AS protein_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'protein'->>1, 'null'), '-1')::numeric AS protein_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'saturated_fat'->>0, 'null'), '-1')::numeric AS saturated_fat_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'saturated_fat'->>1, 'null'), '-1')::numeric AS saturated_fat_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'carb'->>0, 'null'), '-1')::numeric AS carb_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'carb'->>1, 'null'), '-1')::numeric AS carb_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'fiber'->>0, 'null'), '-1')::numeric AS fiber_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'fiber'->>1, 'null'), '-1')::numeric AS fiber_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sugar'->>0, 'null'), '-1')::numeric AS sugar_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sugar'->>1, 'null'), '-1')::numeric AS sugar_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'cholesterol'->>0, 'null'), '-1')::numeric AS cholesterol_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'cholesterol'->>1, 'null'), '-1')::numeric AS cholesterol_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sodium'->>0, 'null'), '-1')::numeric AS sodium_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sodium'->>1, 'null'), '-1')::numeric AS sodium_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calcium'->>0, 'null'), '-1')::numeric AS calcium_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calcium'->>1, 'null'), '-1')::numeric AS calcium_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'phosphorus'->>0, 'null'), '-1')::numeric AS phosphorus_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'phosphorus'->>1, 'null'), '-1')::numeric AS phosphorus_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'potassium'->>0, 'null'), '-1')::numeric AS potassium_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'potassium'->>1, 'null'), '-1')::numeric AS potassium_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'iron'->>0, 'null'), '-1')::numeric AS iron_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'iron'->>1, 'null'), '-1')::numeric AS iron_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'folic_acid'->>0, 'null'), '-1')::numeric AS folic_acid_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'folic_acid'->>1, 'null'), '-1')::numeric AS folic_acid_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_c'->>0, 'null'), '-1')::numeric AS vitamin_c_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_c'->>1, 'null'), '-1')::numeric AS vitamin_c_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_d'->>0, 'null'), '-1')::numeric AS vitamin_d_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_d'->>1, 'null'), '-1')::numeric AS vitamin_d_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_b12'->>0, 'null'), '-1')::numeric AS vitamin_b12_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_b12'->>1, 'null'), '-1')::numeric AS vitamin_b12_max
FROM daily_food_with_nutrition_target_silver
WHERE target IS NOT NULL
ON CONFLICT (target) DO NOTHING;

INSERT INTO targets (
    target,
    calorie_min, calorie_max,
    protein_min, protein_max,
    saturated_fat_min, saturated_fat_max,
    carb_min, carb_max,
    fiber_min, fiber_max,
    sugar_min, sugar_max,
    cholesterol_min, cholesterol_max,
    sodium_min, sodium_max,
    calcium_min, calcium_max,
    phosphorus_min, phosphorus_max,
    potassium_min, potassium_max,
    iron_min, iron_max,
    folic_acid_min, folic_acid_max,
    vitamin_c_min, vitamin_c_max,
    vitamin_d_min, vitamin_d_max,
    vitamin_b12_min, vitamin_b12_max
)
SELECT DISTINCT
    target AS target,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calorie'->>0, 'null'), '-1')::numeric AS calorie_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calorie'->>1, 'null'), '-1')::numeric AS calorie_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'protein'->>0, 'null'), '-1')::numeric AS protein_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'protein'->>1, 'null'), '-1')::numeric AS protein_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'saturated_fat'->>0, 'null'), '-1')::numeric AS saturated_fat_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'saturated_fat'->>1, 'null'), '-1')::numeric AS saturated_fat_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'carb'->>0, 'null'), '-1')::numeric AS carb_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'carb'->>1, 'null'), '-1')::numeric AS carb_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'fiber'->>0, 'null'), '-1')::numeric AS fiber_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'fiber'->>1, 'null'), '-1')::numeric AS fiber_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sugar'->>0, 'null'), '-1')::numeric AS sugar_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sugar'->>1, 'null'), '-1')::numeric AS sugar_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'cholesterol'->>0, 'null'), '-1')::numeric AS cholesterol_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'cholesterol'->>1, 'null'), '-1')::numeric AS cholesterol_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sodium'->>0, 'null'), '-1')::numeric AS sodium_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'sodium'->>1, 'null'), '-1')::numeric AS sodium_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calcium'->>0, 'null'), '-1')::numeric AS calcium_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'calcium'->>1, 'null'), '-1')::numeric AS calcium_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'phosphorus'->>0, 'null'), '-1')::numeric AS phosphorus_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'phosphorus'->>1, 'null'), '-1')::numeric AS phosphorus_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'potassium'->>0, 'null'), '-1')::numeric AS potassium_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'potassium'->>1, 'null'), '-1')::numeric AS potassium_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'iron'->>0, 'null'), '-1')::numeric AS iron_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'iron'->>1, 'null'), '-1')::numeric AS iron_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'folic_acid'->>0, 'null'), '-1')::numeric AS folic_acid_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'folic_acid'->>1, 'null'), '-1')::numeric AS folic_acid_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_c'->>0, 'null'), '-1')::numeric AS vitamin_c_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_c'->>1, 'null'), '-1')::numeric AS vitamin_c_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_d'->>0, 'null'), '-1')::numeric AS vitamin_d_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_d'->>1, 'null'), '-1')::numeric AS vitamin_d_max,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_b12'->>0, 'null'), '-1')::numeric AS vitamin_b12_min,
    COALESCE(NULLIF(replace(replace(target, '''', '"'), 'None', 'null')::jsonb->'vitamin_b12'->>1, 'null'), '-1')::numeric AS vitamin_b12_max
FROM daily_food_with_nutrition_target_bronze
WHERE target IS NOT NULL
ON CONFLICT (target) DO NOTHING;
```


### insert users table
```sql
INSERT INTO users (
    user_id,
    gender,
    age,
    weight,
    height,
    level,
    under_weight,
    over_weight,
    blood_pressure,
    low_density_lipoprotein,
    blood_urea_nitrogen,
    opioid_misuse,
    diabetes,
    anemia,
    osteoporosis,
    user_low_phosphorus,
    user_low_carb,
    user_low_calorie,
    user_high_calorie,
    user_low_sodium,
    user_high_potassium,
    user_low_saturated_fat,
    user_low_cholesterol,
    user_low_protein,
    user_high_protein,
    user_low_sugar,
    user_high_fiber,
    user_high_folate_acid,
    user_high_iron,
    user_high_vitamin_b12,
    user_high_calcium,
    user_high_vitamin_c,
    user_high_vitamin_d
)
SELECT DISTINCT ON (user_id)  
    user_id,
    gender,
    age,
    weight,
    height,
    level,
    under_weight,
    over_weight,
    blood_pressure,
    low_density_lipoprotein,
    blood_urea_nitrogen,
    opioid_misuse,
    diabetes,
    anemia,
    osteoporosis,
    user_low_phosphorus,
    user_low_carb,
    user_low_calorie,
    user_high_calorie,
    user_low_sodium,
    user_high_potassium,
    user_low_saturated_fat,
    user_low_cholesterol,
    user_low_protein,
    user_high_protein,
    user_low_sugar,
    user_high_fiber,
    user_high_folate_acid,
    user_high_iron,
    user_high_vitamin_b12,
    user_high_calcium,
    user_high_vitamin_c,
    user_high_vitamin_d
FROM daily_food_with_nutrition_target_silver
WHERE user_id IS NOT NULL;  
INSERT INTO users (
    user_id,
    gender,
    age,
    weight,
    height,
    level,
    under_weight,
    over_weight,
    blood_pressure,
    low_density_lipoprotein,
    blood_urea_nitrogen,
    opioid_misuse,
    diabetes,
    anemia,
    osteoporosis,
    user_low_phosphorus,
    user_low_carb,
    user_low_calorie,
    user_high_calorie,
    user_low_sodium,
    user_high_potassium,
    user_low_saturated_fat,
    user_low_cholesterol,
    user_low_protein,
    user_high_protein,
    user_low_sugar,
    user_high_fiber,
    user_high_folate_acid,
    user_high_iron,
    user_high_vitamin_b12,
    user_high_calcium,
    user_high_vitamin_c,
    user_high_vitamin_d
)
SELECT DISTINCT ON (user_id)  
    user_id,
    gender,
    age,
    weight,
    height,
    level,
    under_weight,
    over_weight,
    blood_pressure,
    low_density_lipoprotein,
    blood_urea_nitrogen,
    opioid_misuse,
    diabetes,
    anemia,
    osteoporosis,
    user_low_phosphorus,
    user_low_carb,
    user_low_calorie,
    user_high_calorie,
    user_low_sodium,
    user_high_potassium,
    user_low_saturated_fat,
    user_low_cholesterol,
    user_low_protein,
    user_high_protein,
    user_low_sugar,
    user_high_fiber,
    user_high_folate_acid,
    user_high_iron,
    user_high_vitamin_b12,
    user_high_calcium,
    user_high_vitamin_c,
    user_high_vitamin_d
FROM daily_food_with_nutrition_target_gold
WHERE user_id IS NOT NULL;  
INSERT INTO users (
    user_id,
    gender,
    age,
    weight,
    height,
    level,
    under_weight,
    over_weight,
    blood_pressure,
    low_density_lipoprotein,
    blood_urea_nitrogen,
    opioid_misuse,
    diabetes,
    anemia,
    osteoporosis,
    user_low_phosphorus,
    user_low_carb,
    user_low_calorie,
    user_high_calorie,
    user_low_sodium,
    user_high_potassium,
    user_low_saturated_fat,
    user_low_cholesterol,
    user_low_protein,
    user_high_protein,
    user_low_sugar,
    user_high_fiber,
    user_high_folate_acid,
    user_high_iron,
    user_high_vitamin_b12,
    user_high_calcium,
    user_high_vitamin_c,
    user_high_vitamin_d
)
SELECT DISTINCT ON (user_id)  
    user_id,
    gender,
    age,
    weight,
    height,
    level,
    under_weight,
    over_weight,
    blood_pressure,
    low_density_lipoprotein,
    blood_urea_nitrogen,
    opioid_misuse,
    diabetes,
    anemia,
    osteoporosis,
    user_low_phosphorus,
    user_low_carb,
    user_low_calorie,
    user_high_calorie,
    user_low_sodium,
    user_high_potassium,
    user_low_saturated_fat,
    user_low_cholesterol,
    user_low_protein,
    user_high_protein,
    user_low_sugar,
    user_high_fiber,
    user_high_folate_acid,
    user_high_iron,
    user_high_vitamin_b12,
    user_high_calcium,
    user_high_vitamin_c,
    user_high_vitamin_d
FROM daily_food_with_nutrition_target_bronze
WHERE user_id IS NOT NULL;  
```
### insert meal plan table
```sql
INSERT INTO meal_plan (
    daily_food_id,
    target,
    grams,
    calorie,
    protein,
    carb,
    sugar,
    fiber,
    saturated_fat,
    cholesterol,
    folic_acid,
    vitamin_b12,
    vitamin_c,
    vitamin_d,
    calcium,
    phosphorus,
    potassium,
    iron,
    sodium, 
    level
)
SELECT DISTINCT ON (daily_food_id)  
    daily_food_id,
    target,
    grams,
    calorie,
    protein,
    carb,
    sugar,
    fiber,
    saturated_fat,
    cholesterol,
    folic_acid,
    vitamin_b12,
    vitamin_c,
    vitamin_d,
    calcium,
    phosphorus,
    potassium,
    iron,
    sodium,
    'gold'
FROM daily_food_with_nutrition_target_gold
WHERE daily_food_id IS NOT NULL;  

INSERT INTO meal_plan (
    daily_food_id,
    target,
    grams,
    calorie,
    protein,
    carb,
    sugar,
    fiber,
    saturated_fat,
    cholesterol,
    folic_acid,
    vitamin_b12,
    vitamin_c,
    vitamin_d,
    calcium,
    phosphorus,
    potassium,
    iron,
    sodium, 
    level
)
SELECT DISTINCT ON (daily_food_id)  
    daily_food_id,
    target,
    grams,
    calorie,
    protein,
    carb,
    sugar,
    fiber,
    saturated_fat,
    cholesterol,
    folic_acid,
    vitamin_b12,
    vitamin_c,
    vitamin_d,
    calcium,
    phosphorus,
    potassium,
    iron,
    sodium,
    'silver'
FROM daily_food_with_nutrition_target_silver
WHERE daily_food_id IS NOT NULL;  

INSERT INTO meal_plan (
    daily_food_id,
    target,
    grams,
    calorie,
    protein,
    carb,
    sugar,
    fiber,
    saturated_fat,
    cholesterol,
    folic_acid,
    vitamin_b12,
    vitamin_c,
    vitamin_d,
    calcium,
    phosphorus,
    potassium,
    iron,
    sodium, 
    level
)
SELECT DISTINCT ON (daily_food_id)  -- 使用 DISTINCT ON 确保每个 daily_food_id 只插入一次
    daily_food_id,
    target,
    grams,
    calorie,
    protein,
    carb,
    sugar,
    fiber,
    saturated_fat,
    cholesterol,
    folic_acid,
    vitamin_b12,
    vitamin_c,
    vitamin_d,
    calcium,
    phosphorus,
    potassium,
    iron,
    sodium,
    'bronze'
FROM daily_food_with_nutrition_target_bronze
WHERE daily_food_id IS NOT NULL;  
```

### Create index
```sql
CREATE INDEX idx_food_user_id_grams ON food_user (food_id, grams);
CREATE INDEX idx_food_code_id ON food_code (food_id);
```

### Backup and restore DB
```commandline
su postgres
pg_dump -d nutri -f backup.sql
psql -d nutri -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
psql -d nutri -f /var/lib/postgresql/backup.sql #put backup.sql into /var/lib/postgresql
```

