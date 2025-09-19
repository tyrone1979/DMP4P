# DMP4P Src Guidance

There are two parts:
1. Data generation code
2. Web-based Recommendation code

## Data generation code

- Five python jupyter notebooks in ./preprocess/. Run one by one and get final dataset.
- ./utils contains utilities codes, eg. utils.target_calculation.py api for calculation.
```python
from utils.target_calculation import TargetNutrition
from service.mealplan import find_matching_daily_food
from utils.config import empty_user
user_info=empty_user()
targetUser = TargetNutrition(user_info)
target_range = targetUser.get_nutrition_target_range_by_situation()
print(f'nutrition requirement range is {target_range}')
#nutrition requirement range is {'calorie': [2100, 2700], 'protein': [0, 70], 'carb': [250, 400], 'saturated_fat': [0, 9999999.0], 'cholesterol': [0, 300]}
target = targetUser.get_nutrition_target_by_situation()
print(f'nutrition requirement is {target}')
#nutrition requirement is {'calorie': 2435.3239999999996, 'protein': 64, 'carb': 273.97394999999995, 'saturated_fat': 16, 'cholesterol': 300}
meal_plan = find_matching_daily_food(target_range)
print(f'find {len(meal_plan)} meal plans')
#find 20 meal plans
```
- find_bronze.py and ga_gpu.py for bronze level daily meal plan mining codes. ga_gpu is a genetic algorithm implemented by pytorch.

## Web-based Recommendation code
- service/, app.py, api.py, templates/ implement web system based on Flask framework
- web system provide restful api. access http://rec.nri.neusoft.edu.cn/apidocs/ for details.