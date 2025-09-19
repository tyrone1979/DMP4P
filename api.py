from flask import request, jsonify
from service.users import query_target_and_meal_plan, query_users
from flask import Flask
from flasgger import Swagger
from service.mealplan import get_all
from utils.target_calculation import get_config
from utils.log_config import setup_logging
from utils.constant import *

app = Flask(__name__)
Swagger(app)

# 设置日志配置
setup_logging()



@app.route('/user-meal-plan/<userid>', methods=['GET'])
def get_user_meal_plan(userid):
    """
    get meal plan by user id
    ---
    tags:
        - get meal plan by user id
    parameters:
        - name: userid
          in: path
          type: string
          required: true
          description: The user ID
    responses:
            target:
              type: string
              description: User nutrition target
            target_range:
              type: string
              description: User nutrition target range
            details:
              type: string
              description: Food items of the meal plan
            meal_plan:
              type: string
              description: Meal plan with nutrition values
            success:
              type: boolean
              description: Whether the meal plan was found
            error:
              type: string
              description: Error message if meal plan not found
    """
    ctx = {}
    target_range, target, meal_plan, details = query_target_and_meal_plan(userid)
    ctx.update(target=target, target_range=target_range, details=details, meal_plan=meal_plan)
    if not meal_plan:
        ctx.update(error='meal plan not found')
    else:
        ctx.update(success=True)
    return jsonify(ctx)


@app.route('/api_calculate', methods=['POST'])
def api_calculate():
    """
    Calculate nutrition target and meal plan based on user information
    ---
    tags:
        - Nutrition Calculation and Meal Plan Recommendation based on user information
    parameters:
        - in: body
          name: body
          description: User information
          required: true
          schema:
            id: user_info
            required:
              - age
              - gender
              - weight
              - height
              - level
            properties:
              age:
                type: integer
                description: User's age
              gender:
                type: string
                description: User's gender, 1 for male, 2 for female
              weight:
                type: number
                description: User's weight
              height:
                type: number
                description: User's height
              level:
                type: integer
                description: User's activity level , 1 - 4
              medical_info:
                type: array
                items:
                  type: string
                description: User's medical information ['over_weight', 'under_weight', 'opioid_misuse', 'low_density_lipoprotein', 'diabetes', 'blood_urea_nitrogen', 'blood_pressure', 'anemia', 'osteoporosis']
              user_preference:
                type: array
                items:
                  type: string
                description: User's preferences:['low_calorie', 'high_calorie', 'low_carb', 'high_fiber','low_protein', 'high_protein', 'low_saturated_fat','low_sugar', 'low_sodium', 'low_cholesterol', 'low_phosphorus','high_potassium', 'high_folate_acid', 'high_iron', 'high_vitamin_b12','high_calcium', 'high_vitamin_c', 'high_vitamin_d']
    responses:
      200:
        description: Nutrition target and meal plan
        schema:
          id: result
          properties:
            target:
              type: string
              description: Nutrition target
            target_range:
              type: string
              description: Nutrition target range
            user_info:
              type: object
              description: User information
            meal_plan:
              type: string
              description: Meal plan
            details:
              type: string
              description: Food details
            error:
              type: string
              description: Error message if meal plan not found
    """

    user_info = request.json
    # 初始化 user_info，所有标签默认为 0
    for tag in ALL_MEDICAL_TAGS:
        user_info.setdefault(tag, 0)
    for tag in ALL_PREFERENCE_TAGS:
        user_info.setdefault('user_' + tag, 0)
    for tag in user_info['medical_info']:
        user_info[tag] = 1
    for tag in user_info['user_preference']:
        user_info['user_' + tag] = 1
    ctx = get_all(user_info)
    return jsonify(ctx)


@app.route("/api_config")
def api_get_config():
    """
    Get all configurations including MET rate, EER calculation factors, healthy food categories, nutrition impact on medical, nutrition target calculation logics.
    ---
    tags:
        - Get all configurations
    responses:
      200:
        description: configurations
    """
    data = get_config()
    return jsonify(data)
