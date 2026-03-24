from flask import render_template, abort, send_from_directory, g
from api import *
import secrets
from service.dashboard import statistics, load_stats, save_stats
from flask_talisman import Talisman

# 配置 CSP 策略
csp = {
    'default-src': "'self'",
    'script-src': "'self' 'unsafe-inline' https://cdn.jsdelivr.net",
    'style-src': "'self' 'unsafe-inline' https://fonts.googleapis.com",
    'img-src': "'self' data:",
    'connect-src': "'self'",
    'font-src': "'self' https://fonts.gstatic.com",
    'object-src': "'none'",
    'media-src': "'self'",
    'frame-src': "'self'"
}

# 初始化 Talisman 并应用 CSP 策略
Talisman(app, content_security_policy=csp, force_https=False)

@app.before_request
def generate_request_nonce():
    g.nonce = secrets.token_hex(16)  # 生成一个随机的 16 字节的 hex 字符串

"""
@app.route('/static/js/<path:filename>')
def custom_static(filename):
    return send_from_directory(app.static_folder + '/js', filename, mimetype='application/javascript')
"""


@app.before_request
def block_options_method():
    if request.method == 'OPTIONS':
        abort(405)  # 方法不被允许



@app.after_request
def add_security_headers(response):

    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    #response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Permitted-Cross-Domain-Policies'] = 'none'
    response.headers['X-Download-Options'] = 'noopen'
    response.headers['Permissions-Policy'] = (
        "accelerometer=(), autoplay=(), "
        "camera=(), display-capture=(),  encrypted-media=(), "
        "fullscreen=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), midi=(), "
        "payment=(), picture-in-picture=(), "
        "publickey-credentials-get=(), screen-wake-lock=(), sync-xhr=(), "
        "usb=(), web-share=(), xr-spatial-tracking=()"
    )
    return response


@app.errorhandler(404)
def page_not_found(e):
    # 返回 404 错误页面
    return render_template('404.html', nonce=g.nonce), 404


@app.errorhandler(500)
def internal_server_error(e):
    # 返回 500 错误页面
    return render_template('500.html', nonce=g.nonce), 500


@app.errorhandler(Exception)
def handle_exception(e):
    # 返回通用错误页面
    app.logger.error(f"An error occurred: {e}")
    return render_template('error.html', error=str(e), nonce=g.nonce), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    ctx = {}
    user_base = {'age': 47, 'gender': '1', 'weight': 80, 'height': 174, 'level': 2}
    # 初始化 user_info，所有标签默认为 0
    user_info = {tag: 0 for tag in ALL_MEDICAL_TAGS}
    for tag in ALL_PREFERENCE_TAGS:
        user_info.setdefault('user_' + tag, 0)  # 如果已存在 medical，不再覆盖
    user_info.update(user_base)
    if request.method == 'GET':
        user_info['over_weight'] = 1
        user_info['blood_pressure'] = 1
        user_info['user_low_calorie'] = 1
        ctx.update(user_info=user_info)
        # 初始化为空
        ctx.update(details=[])
        ctx.update(target_range={})
        ctx.update(target={})
        ctx.update(error=None)
        ctx.update(meal_plan=[])
    elif request.method == 'POST':
        user_info['age'] = int(request.form['age'])
        user_info['gender'] = request.form['gender']
        user_info['weight'] = float(request.form['weight'])
        user_info['height'] = float(request.form['height'])
        user_info['level'] = int(request.form['activity_level'])
        medical_tags = request.form.getlist('medical_info')  # 列表
        # 把选中的标签置 1
        for tag in medical_tags:
            user_info[tag] = 1
        preference_tags = request.form.getlist('user_preference')  # 列表
        for tag in preference_tags:
            user_info['user_' + tag] = 1
        ctx = get_all(user_info)
        app.logger.info(f"Request from {request.remote_addr}: {request.method} {request.path} {user_info}")

    ctx.update(active_page='recommendation')
    ctx.update(nonce=g.nonce)
    return render_template('index.html', **ctx)


import csv
from io import StringIO


@app.route('/export-csv', methods=['POST'])
def export_csv():
    data = request.get_json()

    # Process data (this is a simplified example)
    output = StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(['Food ID', 'Food Name', 'Grams', 'Eating Type', 'Daily Food ID', 'level'])

    # Write data rows
    for daily_food_id in data:
        for item in data[daily_food_id]:
            writer.writerow([
                item.get('food_id', ''),
                item.get('food_desc', ''),
                item.get('grams', ''),
                item.get('eating_type', ''),
                item.get('daily_food_id', ''),
                item.get('level', '')
            ])

    # Create CSV response
    response = app.response_class(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-disposition": "attachment; filename=meal_plan.csv"}
    )
    return response


@app.route('/dashboard')
def dashboard():
    stats = load_stats()
    if stats == {}:
        stats = statistics()
        save_stats(stats)
    return render_template('dashboard.html', stats=stats, active_page='dashboard', nonce=g.nonce)


@app.route('/users', methods=['GET', 'POST'])
def users():
    medical_tags = request.form.getlist('medical_info')  # 列表
    preference_tags = request.form.getlist('user_preference')  # 列表
    users = query_users(medical_tags, preference_tags)
    return render_template('users.html', users=users, active_page='users', nonce=g.nonce)


@app.route("/config")
def config():
    data = get_config()
    return render_template("config.html", data=data, active_page='config')


# ---------------- 启动 ----------------
if __name__ == '__main__':
    app.run(debug=True, port=5001)
