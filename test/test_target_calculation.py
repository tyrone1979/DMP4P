import unittest
from utils.target_calculation import calc_EER,cal_MET,TargetNutrition
from service.mealplan import find_matching_daily_food
from utils.config import get_config,safe_eval,empty_user
config=get_config()
class MyTestCase(unittest.TestCase):
    def test_calc_EER(self):
        level=1
        gender="2"
        age=26
        height=180
        weight=90
        met=cal_MET(level, gender)
        calories=calc_EER(gender, age, height, weight, met)
        formula=config['MET_and_EER']['EER']
        test=safe_eval(formula,{'gender':gender,'age':age,'height':height,'weight':weight,'met':met })
        self.assertEqual(test, calories)  # add assertion here

    def test_cal_MET(self):
        level,gender=1,'1'
        met=cal_MET(level,gender)
        test = safe_eval(config['MET_and_EER']['MET'],{"level":level, "gender":gender})
        self.assertEqual(met, test)  # add assertion here
        level,gender=2,'1'
        met=cal_MET(level,gender)
        test = safe_eval(config['MET_and_EER']['MET'],{"level":level, "gender":gender})
        self.assertEqual(met, test)  # add assertion here
        level,gender=2,'2'
        met=cal_MET(level,gender)
        test = safe_eval(config['MET_and_EER']['MET'],{"level":level, "gender":gender})
        self.assertEqual(met, test)  # add assertion here
        level,gender=3,'2'
        met=cal_MET(level,gender)
        test = safe_eval(config['MET_and_EER']['MET'],{"level":level, "gender":gender})
        self.assertEqual(met, test)  # add assertion here
        level,gender=3,'1'
        met=cal_MET(level,gender)
        test = safe_eval(config['MET_and_EER']['MET'],{"level":level, "gender":gender})
        self.assertEqual(met, test)  # add assertion here
        level,gender=4,'2'
        met=cal_MET(level,gender)
        test = safe_eval(config['MET_and_EER']['MET'],{"level":level, "gender":gender})
        self.assertEqual(met, test)  # add assertion here
        level,gender=4,'1'
        met=cal_MET(level,gender)
        test = safe_eval(config['MET_and_EER']['MET'],{"level":level, "gender":gender})
        self.assertEqual(met, test)  # add assertion here

    def test_meal_plan(self):
        user_info=empty_user()
        targetUser = TargetNutrition(user_info)
        target_range = targetUser.get_nutrition_target_range_by_situation()
        print(f'nutrition requirement range is {target_range}')
        target = targetUser.get_nutrition_target_by_situation()
        print(f'nutrition requirement is {target}')
        meal_plan = find_matching_daily_food(target_range)
        print(f'find {len(meal_plan)} meal plans')

if __name__ == '__main__':
    unittest.main()
