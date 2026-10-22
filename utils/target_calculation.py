
import json
import math
import os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构造 config.json 的绝对路径
config_path = os.path.join(current_dir, 'config.json')

with open(config_path, 'r') as f:
    config = json.load(f)
    thresholds = config["thresholds"]
    food_categories = config["food_categories"]
    medical_cols = config["medical_cols"]
    nutrition_by_situation = config["nutrition_by_situation"]
    essentional_nutritions = config["essentional_nutritions"]
    cal_MET_config = config['cal_MET']
    calc_EER_config = config['calc_EER']
    INFINITY = config['infinity']
    nutrients = config["nutrients"]
    macro_nutrients = config["macro_nutrients"]
    micro_nutrients = config['micro_nutrients']

# 定义一个安全的解析函数
def safe_eval(formula, variables):
    try:
        # Allow max/min in config formulas (reviewer-aligned bounds).
        env = {"max": max, "min": min, **variables}
        return eval(formula, {"__builtins__": None}, env)
    except Exception as e:
        print(f"Error evaluating formula: {e}")
        return None


def _norm_gender(gender) -> str:
    """NHANES / CSV often store sex as float 1.0; formulas expect '1'|'2'."""
    g = str(int(float(gender)))
    if g not in ("1", "2"):
        raise ValueError(f"invalid gender: {gender!r}")
    return g


def _norm_level(level, default: int = 2) -> int:
    """
    PAL 1–4 only. Missing / 0 / 9 / garbage → low_active (2), matching
    tag_participants_health_raw + compute_participant_targets defaults.
    Never map unknown → level_4 MET (old cal_MET 'default' 1.45/1.48).
    """
    try:
        if level is None:
            return default
        lv = int(float(level))
    except (TypeError, ValueError):
        return default
    if lv not in (1, 2, 3, 4):
        return default
    return lv


def _flag_on(user_info, condition) -> bool:
    v = user_info.get(condition, 0)
    if v is None:
        return False
    try:
        return int(float(v)) == 1
    except (TypeError, ValueError):
        return bool(v)


def cal_MET(row):
    gender = _norm_gender(row["gender"])
    level = _norm_level(row.get("level", row.get("pal_level", 2)))
    if level == 1:
        return cal_MET_config["default_level_1"]
    sex = "male" if gender == "1" else "female"
    return cal_MET_config[sex][f"level_{level}"]


def calc_EER(gender, age, height, weight, met):
    gender = _norm_gender(gender)
    if gender == "1":  # Male
        return (
                calc_EER_config['male']['base'] +
                calc_EER_config['male']['age_factor'] * age +
                met * (
                        calc_EER_config['male']['weight_factor'] * weight +
                        calc_EER_config['male']['height_factor'] * height
                )
        )
    else:  # Female
        return (
                calc_EER_config['female']['base'] +
                calc_EER_config['female']['age_factor'] * age +
                met * (
                        calc_EER_config['female']['weight_factor'] * weight +
                        calc_EER_config['female']['height_factor'] * height
                )
        )


class TargetNutrition:
    def __init__(self, user_info):
        # Copy so float gender / bad level in CSV rows cannot poison MET lookup.
        self.user_info = dict(user_info)
        self.gender = _norm_gender(self.user_info["gender"])
        self.age = float(self.user_info["age"])
        self.height = float(self.user_info["height"])
        self.weight = float(self.user_info["weight"])
        level = _norm_level(self.user_info.get("level", self.user_info.get("pal_level", 2)))
        self.user_info["gender"] = self.gender
        self.user_info["level"] = level
        self.met = cal_MET(self.user_info)
        self.EER = calc_EER(self.gender, self.age, self.height, self.weight, self.met)


    def get_target_range(self, name):
        target_range_logic = config['target_range_logic']
        nutrient_logic = target_range_logic.get(name, {}).get('conditions', {})

        # Check for specific conditions that affect the target range
        for condition, formula in nutrient_logic.items():
            if condition == "default":
                continue
            if _flag_on(self.user_info, condition):
                # Evaluate the formula dynamically
                return safe_eval(formula, {
                    "age": self.age,
                    "gender": self.gender,
                    "EER": self.EER,
                    "calc_EER": calc_EER,
                    "height":self.height,
                    "weight": self.weight,
                    "thresholds": thresholds,
                    "INFINITY": INFINITY,
                    "math": math
                })

        # Default range if no specific condition is met
        default_formula = nutrient_logic.get('default')
        return safe_eval(default_formula, {
            "age": self.age,
            "EER": self.EER,
            "gender": self.gender,
            "calc_EER": calc_EER,
            "height": self.height,
            "weight": self.weight,
            "thresholds": thresholds,
            "INFINITY": INFINITY,
            "math": math
        })

    def get_target(self, name):
        target_logic = config['target_logic']
        nutrient_logic = target_logic.get(name, {}).get('conditions', {})

        # Check for specific conditions that affect the target range
        for condition, formula in nutrient_logic.items():
            if condition == "default":
                continue
            if _flag_on(self.user_info, condition):
                # Evaluate the formula dynamically
                return safe_eval(formula, {
                    "age": self.age,
                    "gender": self.gender,
                    "EER": self.EER,
                    "weight": self.weight,
                    "thresholds": thresholds,
                    "INFINITY": INFINITY,
                    "math": math
                })

        # Default range if no specific condition is met
        default_formula = nutrient_logic.get('default')
        return safe_eval(default_formula, {
            "age": self.age,
            "EER": self.EER,
            "gender": self.gender,
            "weight": self.weight,
            "thresholds": thresholds,
            "INFINITY": INFINITY,
            "math": math
        })

    def get_macro_nutrition_target_range(self):

        macro_nutrition_target_range = {}

        for nutrient in macro_nutrients:
            target_range = self.get_target_range(nutrient)
            macro_nutrition_target_range[nutrient] = target_range

        return macro_nutrition_target_range

    def get_micro_nutrition_target_range(self):

        micro_nutrition_target_range = {}

        for nutrient in micro_nutrients:
            micro_nutrition_target_range[nutrient] = self.get_target_range(nutrient)

        return micro_nutrition_target_range

    def get_nutrition_target_by_situation(self):
        """Point targets: macros + nutrients linked to the user's medical flags."""
        full_targets = {}
        for nutrient in nutrients:
            full_targets[nutrient] = self.get_target(nutrient)

        targets = {n: full_targets[n] for n in macro_nutrients if n in full_targets}
        for situation in medical_cols:
            if _flag_on(self.user_info, situation):
                for condition in nutrition_by_situation.get(situation) or []:
                    if condition in full_targets:
                        targets[condition] = full_targets[condition]
        return targets

    def get_nutrition_target_range_by_situation(self):
        """
        Bronze hard ranges: always macro nutrients, plus micros tied to medical
        situations (V1-style). Does NOT force the full 16-nutrient essential set.
        """
        full_targets = {
            **self.get_macro_nutrition_target_range(),
            **self.get_micro_nutrition_target_range(),
        }
        targets = {n: full_targets[n] for n in macro_nutrients if n in full_targets}
        for situation in medical_cols:
            if _flag_on(self.user_info, situation):
                for condition in nutrition_by_situation.get(situation) or []:
                    if condition in full_targets:
                        targets[condition] = full_targets[condition]
        return targets

def get_nutrition_target_range_by_situation(row):
    """
    Base on user's situation, calculate nutrition target range
    :param row: user situation
    :return: nutrition target range dict
    """
    target = TargetNutrition(row)
    targets = target.get_nutrition_target_range_by_situation()
    return str(targets)