import json
import os


def get_config():
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构造 config.json 的绝对路径
    config_path = os.path.join(current_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# 定义一个安全的解析函数
def safe_eval(formula, variables):
    try:
        return eval(formula, {"__builtins__": None}, variables)
    except Exception as e:
        print(f"Error evaluating formula: {e}")
        return None
def empty_user():
    config=get_config()
    medical_cols = config['medical_cols']
    user_info = {"age": 46, "gender": 1, "weight": 80, "height": 174, "level": 1}
    user_info.update({col: 0 for col in medical_cols})
    return user_info