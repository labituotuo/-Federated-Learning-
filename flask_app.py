# -*- coding: utf-8 -*-
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 修复Jinja2兼容性问题
try:
    from markupsafe import escape
except ImportError:
    from jinja2 import escape

try:
    from jinja2 import Markup
except ImportError:
    class Markup(str):
        def __html__(self):
            return self
    
    import jinja2
    jinja2.escape = escape
    jinja2.Markup = Markup

from flask import Flask, render_template, jsonify, request, Response, send_file, redirect, url_for, session
import torch
import pandas as pd
import numpy as np
import json
import threading
import time
from datetime import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MySQL数据库连接
import pymysql
from pymysql.cursors import DictCursor

from db_utils import get_db_connection, init_database
from user_metrics_utils import save_user_metrics as utils_save_user_metrics, get_user_metrics as utils_get_user_metrics, delete_user_metrics_record, get_user_id_by_username
from training_utils import save_training_result, get_all_training_results, get_training_result_by_id

app = Flask(__name__)
app.secret_key = 'fl_fed_learning_secret_key_2024'  # 用于session加密

# 初始化数据库
init_database()

# 导入 http_federated 模型结构
class HttpFederatedNet(torch.nn.Module):
    """HTTP联邦学习使用的模型结构（nn.Sequential）"""
    def __init__(self, input_dim=43):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# 全局变量
loaded_model = None
model_version = None
scaler = None
model_accuracy = None  # 模型准确率
model_saved_at = None  # 模型保存时间

def load_latest_model():
    """加载最新保存的联邦学习模型（优先使用http_federated模型）"""
    global loaded_model, model_version, scaler, model_accuracy, model_saved_at
    
    try:
        # 首先尝试从数据库加载准确率最高的模型
        model_path = None
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor(DictCursor)
                # 查询准确率最高的模型
                cursor.execute("""
                    SELECT 模型文件路径, 准确率 
                    FROM training_results 
                    WHERE 模型文件路径 IS NOT NULL AND 准确率 IS NOT NULL
                    ORDER BY 准确率 DESC 
                    LIMIT 1
                """)
                result = cursor.fetchone()
                cursor.close()
                
                if result:
                    model_path = result['模型文件路径']
                    model_accuracy = result['准确率']
                    print(f"正在加载准确率最高的模型: {os.path.basename(model_path)} (准确率: {model_accuracy:.2f}%)")
            except Exception as e:
                print(f"从数据库获取模型失败: {str(e)}")
            finally:
                conn.close()
        
        # 如果数据库中没有记录，回退到文件系统查找最新模型
        if not model_path:
            http_models_dir = os.path.join(os.path.dirname(__file__), 'http_federated', 'saved_models')
            if os.path.exists(http_models_dir):
                pth_files = [f for f in os.listdir(http_models_dir) if f.endswith('.pth')]
                if pth_files:
                    pth_files_with_time = [
                        (f, os.path.getmtime(os.path.join(http_models_dir, f)))
                        for f in pth_files
                    ]
                    pth_files_with_time.sort(key=lambda x: x[1], reverse=True)
                    latest_pth = pth_files_with_time[0][0]
                    model_path = os.path.join(http_models_dir, latest_pth)
                    print(f"正在加载最新模型（数据库无记录）: {latest_pth}")
        
        # 加载模型（无论从数据库还是文件系统获取）
        if model_path and os.path.exists(model_path):
            # 创建HTTP联邦学习模型实例
            model = HttpFederatedNet(input_dim=43).to(DEVICE)
            
            # 加载模型参数
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model.eval()
            
            # 初始化scaler（使用clients_data_3clients）
            from sklearn.preprocessing import StandardScaler
            clients_data_dir = os.path.join(os.path.dirname(__file__), 'clients_data_3clients')
            all_data = []
            for i in range(3):
                client_file = os.path.join(clients_data_dir, f'client_{i}.csv')
                if os.path.exists(client_file):
                    df = pd.read_csv(client_file)
                    X = df.iloc[:, :-1].values
                    all_data.append(X)
            
            if len(all_data) > 0:
                all_X = np.vstack(all_data)
                scaler = StandardScaler()
                scaler.fit(all_X)
                print(f"全局StandardScaler已创建")
            
            loaded_model = model
            model_version = os.path.basename(model_path)
            
            # 提取保存时间（从文件名）
            try:
                timestamp_str = model_version.replace('SCAFFOLD_3clients_', '').replace('FEDAVG_3clients_', '').replace('FEDPROX_3clients_', '').replace('.pth', '')
                model_saved_at = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
            except:
                model_saved_at = None
            
            model_accuracy = None  # HTTP模型没有直接存储准确率
            
            print(f"成功加载HTTP联邦学习模型: {model_version}")
            if model_saved_at:
                print(f"模型保存时间: {model_saved_at}")
            
            return True, f"成功加载HTTP联邦学习模型: {model_version}"
        
        return False, "没有找到已保存的模型"
        
    except Exception as e:
        import traceback
        print(f"加载模型失败: {str(e)}")
        print(traceback.format_exc())
        return False, f"加载模型失败: {str(e)}"

# 预测函数
def predict_diabetes(input_data):
    """使用加载的模型进行糖尿病预测"""
    global loaded_model, scaler
    
    if not loaded_model:
        # 尝试加载模型
        success, message = load_latest_model()
        if not success:
            return None, message
    
    try:
        # 准备输入数据
        # 确保输入数据的顺序正确（与训练时一致）
        feature_order = [
            '年龄', '性别', '种族', '社会经济地位', '教育水平', '体重指数',
            '吸烟状态', '饮酒量', '每周体育活动时间', '饮食质量', '睡眠质量',
            '糖尿病家族史', '妊娠期糖尿病', '多囊卵巢综合征', '既往糖尿病前期', '高血压',
            '收缩压', '舒张压', '空腹血糖', '糖化血红蛋白',
            '血清肌酐', '血尿素氮水平', '总胆固醇', '低密度脂蛋白胆固醇', '高密度脂蛋白胆固醇', '甘油三酯',
            '降压药物使用', '他汀类药物使用', '抗糖尿病药物使用',
            '尿频', '过度口渴', '不明原因体重下降', '疲劳程度', '视力模糊', '伤口愈合缓慢', '手脚刺痛',
            '生活质量评分', '重金属暴露', '职业化学物质暴露', '水质', '体检频率', '药物依从性', '健康素养'
        ]
        
        # 提取特征值
        features = []
        for feature in feature_order:
            if feature in input_data:
                features.append(input_data[feature])
            else:
                # 对于缺失的特征，使用默认值
                features.append(0)
        
        # 转换为numpy数组并标准化
        features_array = np.array([features], dtype=np.float32)
        if scaler:
            features_array = scaler.transform(features_array)
        
        # 转换为torch张量
        input_tensor = torch.tensor(features_array, dtype=torch.float32).to(DEVICE)
        
        # 模型预测
        with torch.no_grad():
            output = loaded_model(input_tensor)
            probability = output.item()
        
        return {
            'probability': probability,
            'model_version': model_version
        }, None
        
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None, f"预测失败: {str(e)}"

# 初始化时尝试加载模型
load_latest_model()

# 启动后台线程，定期检测新模型并自动重新加载
def auto_reload_model():
    """后台线程：每30秒检测一次是否有新模型"""
    global model_version
    import time
    
    while True:
        try:
            time.sleep(30)  # 每30秒检查一次
            
            # 检查http_federated/saved_models目录
            http_models_dir = os.path.join(os.path.dirname(__file__), 'http_federated', 'saved_models')
            if os.path.exists(http_models_dir):
                pth_files = [f for f in os.listdir(http_models_dir) if f.endswith('.pth')]
                if pth_files:
                    # 按文件修改时间排序
                    pth_files_with_time = [
                        (f, os.path.getmtime(os.path.join(http_models_dir, f)))
                        for f in pth_files
                    ]
                    pth_files_with_time.sort(key=lambda x: x[1], reverse=True)
                    latest_pth = pth_files_with_time[0][0]
                    
                    # 如果检测到新模型，自动重新加载
                    if latest_pth != model_version:
                        print(f'[自动检测] 发现新模型: {latest_pth}')
                        success, message = load_latest_model()
                        if success:
                            print(f'[自动加载] 成功加载新模型: {latest_pth}')
                        else:
                            print(f'[自动加载] 失败: {message}')
        except Exception as e:
            print(f'[自动检测] 错误: {str(e)}')

# 启动后台线程
reload_thread = threading.Thread(target=auto_reload_model, daemon=True)
reload_thread.start()
print('[后台线程] 模型自动检测已启动（每30秒检查一次）')

# Training state
is_training = False
current_num_clients = 5  # Default 5 clients
last_generated_clients = None  # 记录最后一次生成的客户端数量



# 读取数据集
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# 保存客户端数据
def save_clients_data(clients_data, output_dir='clients_data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 清理旧的客户端数据文件
    for filename in os.listdir(output_dir):
        if filename.startswith('client_') and (filename.endswith('.csv') or filename.endswith('.json')):
            os.remove(os.path.join(output_dir, filename))
    
    for client_id, data in clients_data.items():
        # 保存为CSV文件
        data.to_csv(os.path.join(output_dir, f'{client_id}.csv'), index=False)
        # 保存数据分布信息
        distribution = {
            'total_samples': len(data),
            'positive_samples': int(data['诊断结果'].sum()),
            'negative_samples': int(len(data) - data['诊断结果'].sum()),
            'positive_ratio': float(data['诊断结果'].mean())
        }
        with open(os.path.join(output_dir, f'{client_id}_distribution.json'), 'w', encoding='utf-8') as f:
            json.dump(distribution, f, ensure_ascii=False, indent=2)

# 实现标签和数量分布不均匀的结合版
def create_non_iid_label_size(df, num_clients=5, min_samples=100):
    """
    结合标签分布和数据量分布的非独立同分布场景
    num_clients: 客户端数量
    min_samples: 每个客户端的最小样本数
    """
    clients_data = {}
    
    # 打乱原始数据
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    
    # 随机分配数据量，确保每个客户端至少有min_samples个样本
    total_samples = len(df)
    # 计算每个客户端的最小样本数总和
    min_total = num_clients * min_samples
    
    # 检查是否有足够的样本
    if min_total > total_samples:
        raise ValueError(f"总样本数 {total_samples} 小于每个客户端最小样本数 {min_samples} 的总和 {min_total}")
    
    # 先为每个客户端分配最小样本数
    base_sizes = [min_samples] * num_clients
    # 计算剩余的样本数
    remaining_samples = total_samples - min_total
    
    # 随机分配剩余的样本数
    if remaining_samples > 0:
        # 生成随机数
        random_sizes = np.random.randint(1, remaining_samples + 1, num_clients)
        # 归一化并调整到剩余样本数
        extra_sizes = [int(s * remaining_samples / sum(random_sizes)) for s in random_sizes]
        extra_sizes[-1] += remaining_samples - sum(extra_sizes)  # 确保总和正确
        # 计算最终的样本数
        sizes = [base + extra for base, extra in zip(base_sizes, extra_sizes)]
    else:
        # 所有样本都分配为最小样本数
        sizes = base_sizes
    
    # 分配数据
    start = 0
    for i in range(num_clients):
        end = start + sizes[i]
        clients_data[f'client_{i}'] = shuffled_df.iloc[start:end].copy()
        start = end
    
    return clients_data

# 主页路由 - 默认跳转到登录页面
@app.route('/')
def index():
    # 直接重定向到登录页面
    return redirect('/login')

# HTTP联邦学习界面（仅限管理员）
@app.route('/http_fl')
def http_fl():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect('/login')
    return render_template('http_fl_new.html')

# 重定向旧路径到新路径
@app.route('/http_fl_new')
def http_fl_new_redirect():
    return redirect('/http_fl')

# 全局变量存储进程引用
http_server_process = None

def start_http_server():
    """启动http_federated/server.py服务器"""
    global http_server_process
    import subprocess
    import os
    import time
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    http_dir = os.path.join(project_root, 'http_federated')
    venv_python = os.path.join(project_root, '.venv', 'Scripts', 'python.exe')
    server_py = os.path.join(http_dir, 'server.py')
    
    # 如果进程已存在，先终止
    if http_server_process:
        try:
            http_server_process.terminate()
            http_server_process.wait(timeout=3)
        except:
            pass
        http_server_process = None
    
    # 启动server.py
    print('[HTTP服务器] 启动 http_federated/server.py ...')
    
    # 使用 start 命令在新窗口启动，这样服务器可以看到输出
    cmd = f'start "FL Server" /D "{http_dir}" "{venv_python}" server.py'
    
    http_server_process = subprocess.Popen(
        cmd,
        shell=True
    )
    
    # 等待服务器就绪
    time.sleep(6)
    print('[HTTP服务器] 服务器已启动（在新窗口中运行）')

@app.route('/api/start_fl', methods=['POST'])
def api_start_fl():
    """一键启动联邦学习：启动server和客户端"""
    import subprocess
    import os
    
    # 获取配置
    data = request.json
    algorithm = data.get('algorithm', 'fedavg')
    mu = data.get('mu', 0.01)
    privacy = data.get('privacy', 'none')
    
    print(f'[一键启动] 开始启动流程... (算法: {algorithm}, 隐私保护: {privacy})')
    
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        http_dir = os.path.join(project_root, 'http_federated')
        venv_python = os.path.join(project_root, '.venv', 'Scripts', 'python.exe')
        start_all_py = os.path.join(http_dir, 'start_all.py')
        
        # 检查文件是否存在
        if not os.path.exists(start_all_py):
            return jsonify({'success': False, 'message': f'start_all.py不存在: {start_all_py}'}), 500
        
        if not os.path.exists(venv_python):
            return jsonify({'success': False, 'message': f'Python虚拟环境不存在: {venv_python}'}), 500
        
        # 使用subprocess.Popen在新窗口启动start_all.py
        # start_all.py会自动启动server和3个client
        cmd = f'start "HTTP联邦学习" /D "{http_dir}" "{venv_python}" "{start_all_py}" --algorithm {algorithm} --mu {mu} --privacy {privacy}'
        
        print(f'[一键启动] 执行命令: {cmd}')
        subprocess.Popen(cmd, shell=True)
        
        print('[一键启动] 启动命令已发送，请查看新打开的窗口')
        
        return jsonify({
            'success': True, 
            'message': '联邦学习系统正在启动，请查看新打开的窗口'
        })
    except Exception as e:
        print(f'[一键启动] 失败: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'启动失败: {str(e)}'}), 500

# 登录页面路由
@app.route('/login')
def login_page():
    session.clear()  # 清除旧session
    return render_template('login.html')

# 管理员主页面
@app.route('/main')
def main_page():
    if 'user' not in session or session['user'] != 'admin':
        return redirect('/login')
    return render_template('index.html')

# 登录验证
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')
    
    print(f"[DEBUG] 登录请求: username={username}, role={role}")
    
    # 从数据库验证用户
    conn = get_db_connection()
    print(f"[DEBUG] 数据库连接: {conn}")
    
    if conn:
        try:
            with conn.cursor() as cursor:
                # 查询用户
                cursor.execute(
                    "SELECT * FROM users WHERE username = %s",
                    (username,)
                )
                user = cursor.fetchone()
                print(f"[DEBUG] 查询结果: {user}")
                
                # 使用SHA-256验证密码（数据库中存储的是哈希值）
                import hashlib
                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                print(f"[DEBUG] 输入密码哈希: {hashed_password}")
                
                if user:
                    print(f"[DEBUG] 数据库密码哈希: {user['password']}")
                    print(f"[DEBUG] 密码是否匹配: {user['password'] == hashed_password}")
                
                if user and user['password'] == hashed_password:
                    # 验证角色是否匹配
                    if user['role'] != role:
                        print(f"[DEBUG] 角色不匹配: 用户角色={user['role']}, 请求角色={role}")
                        return jsonify({'success': False, 'message': '角色选择错误'}), 401
                    
                    # 更新最后登录时间
                    cursor.execute(
                        "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                        (user['id'],)
                    )
                    conn.commit()
                    
                    # 设置session
                    session['user'] = user['username']
                    session['role'] = user['role']
                    
                    print(f"[DEBUG] 登录成功: {user['username']}")
                    return jsonify({'success': True, 'message': '登录成功', 'role': user['role']})
                else:
                    print(f"[DEBUG] 登录失败: 用户名或密码错误")
                    return jsonify({'success': False, 'message': '用户名或密码错误'}), 401
        except pymysql.MySQLError as e:
            print(f"[DEBUG] 数据库错误: {e}")
            return jsonify({'success': False, 'message': '登录验证失败'}), 500
        finally:
            conn.close()
    else:
        print(f"[DEBUG] 数据库连接失败，使用默认验证")
        # 数据库连接失败时使用默认验证
        import hashlib
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        # 默认admin密码哈希: SHA256('admin123')
        if username == 'admin' and hashed_password == '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9' and role == 'admin':
            session['user'] = 'admin'
            session['role'] = 'admin'
            print(f"[DEBUG] 默认验证成功: admin")
            return jsonify({'success': True, 'message': '登录成功', 'role': 'admin'})
        # 默认user密码哈希: SHA256('user123')
        elif username == 'user' and hashed_password == 'e606e38b0d8c19b24cf0ee3808183162ea7cd63ff7912dbb22b5e803286b4446' and role == 'user':
            session['user'] = 'user'
            session['role'] = 'user'
            return jsonify({'success': True, 'message': '登录成功', 'role': 'user'})
        else:
            return jsonify({'success': False, 'message': '用户名、密码或角色选择错误'}), 401

# 登出
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# 退出登录API（前端需要）
@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True, 'message': '登出成功'})

# 用户注册API
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')
    
    # 验证输入
    if not username or not password:
        return jsonify({'success': False, 'message': '用户名和密码不能为空'}), 400
    
    if len(username) < 2 or len(password) < 6:
        return jsonify({'success': False, 'message': '用户名至少2个字符，密码至少6个字符'}), 400
    
    if role not in ['admin', 'user']:
        return jsonify({'success': False, 'message': '角色选择错误'}), 400
    
    # 从数据库注册用户
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                # 检查用户名是否已存在
                cursor.execute(
                    "SELECT COUNT(*) as count FROM users WHERE username = %s",
                    (username,)
                )
                result = cursor.fetchone()
                
                if result['count'] > 0:
                    return jsonify({'success': False, 'message': '用户名已存在'}), 400
                
                # 插入新用户
                cursor.execute(
                    "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                    (username, password, role)
                )
                conn.commit()
                
                return jsonify({'success': True, 'message': '注册成功'})
        except pymysql.MySQLError as e:
            print(f"注册失败: {e}")
            return jsonify({'success': False, 'message': '注册失败'}), 500
        finally:
            conn.close()
    else:
        # 数据库连接失败时的处理
        return jsonify({'success': False, 'message': '数据库连接失败，无法注册'}), 500

# 预测页面路由（仅限普通用户）登录检查
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return jsonify({'success': False, 'message': '请先登录'}), 401
        return f(*args, **kwargs)
    return decorated_function

# 登录验证API
@app.route('/api/check_login')
def check_login():
    if 'user' in session:
        return jsonify({
            'logged_in': True,
            'username': session['user']
        })
    return jsonify({'logged_in': False})

# 认证状态API（前端需要）
@app.route('/api/auth/status')
def auth_status():
    if 'user' in session:
        return jsonify({
            'logged_in': True,
            'username': session['user'],
            'role': session.get('role', 'user')
        })
    return jsonify({'logged_in': False})

# 数据集信息API
@app.route('/dataset_info')
def dataset_info():
    try:
        df = pd.read_excel('diabetes_data_Chinese.xlsx')
        total_samples = len(df)
        positive_samples = int(df['诊断结果'].sum())
        negative_samples = total_samples - positive_samples
        positive_ratio = positive_samples / total_samples if total_samples > 0 else 0
        
        return jsonify({
            'success': True,
            'info': {
                'total_samples': total_samples,
                'positive_samples': positive_samples,
                'negative_samples': negative_samples,
                'positive_ratio': positive_ratio,
                'columns': list(df.columns)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# 获取上次生成的客户端数据分布
@app.route('/last_client_data')
def last_client_data():
    try:
        clients_info = []
        clients_dir = 'clients_data'
        
        # 检查是否存在客户端数据目录
        if os.path.exists(clients_dir):
            # 查找所有客户端CSV文件
            client_files = sorted([f for f in os.listdir(clients_dir) if f.endswith('.csv') and f.startswith('client_')])
            
            for client_file in client_files:
                client_id = client_file.replace('.csv', '')
                file_path = os.path.join(clients_dir, client_file)
                
                # 读取客户端数据
                df = pd.read_csv(file_path)
                total_samples = len(df)
                positive_samples = int(df['诊断结果'].sum())
                negative_samples = total_samples - positive_samples
                positive_ratio = positive_samples / total_samples if total_samples > 0 else 0
                
                clients_info.append({
                    'id': client_id,
                    'total_samples': total_samples,
                    'positive_samples': positive_samples,
                    'negative_samples': negative_samples,
                    'positive_ratio': positive_ratio
                })
        
        return jsonify({
            'success': True,
            'has_data': len(clients_info) > 0,
            'clients': clients_info,
            'num_clients': len(clients_info)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'has_data': False,
            'clients': []
        }), 500

# 生成客户端数据API
@app.route('/generate_data', methods=['POST'])
def generate_data():
    global last_generated_clients
    try:
        num_clients = int(request.form.get('num_clients', 3))
        last_generated_clients = num_clients
        
        # 读取原始数据
        df = load_data('diabetes_data_Chinese.xlsx')
        
        # 创建非IID数据分布
        clients_data = create_non_iid_label_size(df, num_clients=num_clients)
        
        # 保存客户端数据
        save_clients_data(clients_data)
        
        # 构建返回信息
        clients_info = []
        for client_id, data in clients_data.items():
            clients_info.append({
                'id': client_id,
                'total_samples': len(data),
                'positive_samples': int(data['诊断结果'].sum()),
                'negative_samples': int(len(data) - data['诊断结果'].sum()),
                'positive_ratio': float(data['诊断结果'].mean())
            })
        
        return jsonify({
            'success': True,
            'message': f'成功生成 {num_clients} 个客户端的数据',
            'clients_info': clients_info
        })
    except Exception as e:
        import traceback
        print(f"生成数据失败: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# 数据统计API
@app.route('/api/data_stats')
def get_data_stats():
    try:
        # 读取客户端数据
        client_id = request.args.get('client_id', 'client_0.csv')
        df = pd.read_csv(f'clients_data/{client_id}')
        
        # 基本统计
        total_samples = len(df)
        positive_samples = df['诊断结果'].sum()
        negative_samples = total_samples - positive_samples
        positive_ratio = positive_samples / total_samples if total_samples > 0 else 0
        
        # 年龄分布
        age_bins = [0, 30, 40, 50, 60, 70, 100]
        age_labels = ['<30', '30-40', '40-50', '50-60', '60-70', '>70']
        age_dist = pd.cut(df['年龄'], bins=age_bins, labels=age_labels).value_counts().sort_index().to_dict()
        
        # BMI分布
        bmi_bins = [0, 18.5, 25, 30, 100]
        bmi_labels = ['偏瘦', '正常', '超重', '肥胖']
        bmi_dist = pd.cut(df['体重指数'], bins=bmi_bins, labels=bmi_labels).value_counts().sort_index().to_dict()
        
        # 血糖分布
        glucose_bins = [0, 100, 126, 200, 500]
        glucose_labels = ['正常', '前期', '糖尿病', '危险']
        glucose_dist = pd.cut(df['空腹血糖'], bins=glucose_bins, labels=glucose_labels).value_counts().sort_index().to_dict()
        
        return jsonify({
            'success': True,
            'total_samples': total_samples,
            'positive_samples': int(positive_samples),
            'negative_samples': int(negative_samples),
            'positive_ratio': float(positive_ratio),
            'age_distribution': age_dist,
            'bmi_distribution': bmi_dist,
            'glucose_distribution': glucose_dist
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# 获取所有客户端列表
@app.route('/api/clients')
def get_clients():
    try:
        # 检查clients_data目录
        if not os.path.exists('clients_data'):
            return jsonify({
                'success': False,
                'message': '客户端数据不存在'
            }), 400
        
        clients = []
        for f in os.listdir('clients_data'):
            if f.endswith('.csv') and not f.endswith('_distribution.json'):
                client_id = f.replace('.csv', '')
                dist_file = f'clients_data/{client_id}_distribution.json'
                if os.path.exists(dist_file):
                    with open(dist_file, 'r', encoding='utf-8') as df:
                        dist = json.load(df)
                        clients.append({
                            'id': client_id,
                            'total_samples': dist['total_samples'],
                            'positive_samples': dist['positive_samples'],
                            'positive_ratio': dist['positive_ratio']
                        })
        
        return jsonify({
            'success': True,
            'clients': clients
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# 获取训练指标
@app.route('/api/train/metrics')
def get_metrics():
    return jsonify({
        'success': True,
        'metrics': None
    })

# 模型状态检查
@app.route('/api/model/status')
def model_status():
    global loaded_model, model_version, model_accuracy, model_saved_at
    
    status = {
        'model_loaded': loaded_model is not None,
        'model_version': model_version if model_version else '未加载',
        'message': '模型已加载' if loaded_model else '模型未加载'
    }
    
    if loaded_model:
        if model_accuracy is not None:
            status['accuracy'] = model_accuracy
        if model_saved_at:
            status['last_updated'] = model_saved_at
    
    return jsonify(status)

# 特征重要性分析API (使用SHAP)
@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    """重新加载最新模型（用于管理端特征分析）"""
    if 'user' not in session or session['user'] != 'admin':
        return jsonify({'success': False, 'message': '未登录或权限不足'}), 401
    
    success, message = load_latest_model()
    return jsonify({
        'success': success,
        'message': message,
        'model_version': model_version if success else None,
        'model_accuracy': model_accuracy if success else None,
        'model_saved_at': model_saved_at if success else None
    })

@app.route('/api/model/feature_importance')
def get_feature_importance():
    global loaded_model, scaler
    
    if loaded_model is None:
        return jsonify({
            'success': False,
            'message': '模型未加载，请先完成训练'
        }), 400
    
    try:
        import numpy as np
        import shap
        from sklearn.preprocessing import StandardScaler
        
        # 特征名称列表
        feature_names = [
            '年龄', '性别', '种族', '社会经济地位', '教育水平', '体重指数',
            '吸烟状态', '饮酒量', '每周体育活动时间', '饮食质量', '睡眠质量',
            '糖尿病家族史', '妊娠期糖尿病', '多囊卵巢综合征', '既往糖尿病前期', '高血压',
            '收缩压', '舒张压', '空腹血糖', '糖化血红蛋白',
            '血清肌酐', '血尿素氮水平', '总胆固醇', '低密度脂蛋白胆固醇', '高密度脂蛋白胆固醇', '甘油三酯',
            '降压药物使用', '他汀类药物使用', '抗糖尿病药物使用',
            '尿频', '过度口渴', '不明原因体重下降', '疲劳程度', '视力模糊', '伤口愈合缓慢', '手脚刺痛',
            '生活质量评分', '重金属暴露', '职业化学物质暴露', '水质', '体检频率', '药物依从性', '健康素养'
        ]
        
        # 检查是否为HTTP联邦学习模型（通过模型类型判断）
        is_http_model = isinstance(loaded_model, HttpFederatedNet)
        
        # 创建预测函数包装
        def predict_proba(x):
            """模型预测函数"""
            # HTTP模型已经使用了StandardScaler，输入数据应该先标准化
            if is_http_model and scaler is not None:
                x = scaler.transform(x)
            elif scaler is not None and not is_http_model:
                x = scaler.transform(x)
            
            input_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                output = loaded_model(input_tensor)
                return output.cpu().numpy()
        
        # 创建背景数据集（使用真实数据而非随机数据）
        clients_data_dir = os.path.join(os.path.dirname(__file__), 'clients_data_3clients' if is_http_model else 'clients_data')
        all_data = []
        for i in range(3):
            client_file = os.path.join(clients_data_dir, f'client_{i}.csv')
            if os.path.exists(client_file):
                df = pd.read_csv(client_file)
                X = df.iloc[:, :-1].values.astype(np.float32)
                all_data.append(X)
        
        if len(all_data) > 0:
            background_data = np.vstack(all_data)[:100]  # 使用前100个样本作为背景
        else:
            np.random.seed(42)
            background_data = np.random.randn(100, len(feature_names)).astype(np.float32)
        
        # 如果scaler不存在，创建一个
        if scaler is None and is_http_model:
            scaler_temp = StandardScaler()
            scaler_temp.fit(background_data)
            background_data = scaler_temp.transform(background_data)
        
        # 使用SHAP KernelExplainer
        explainer = shap.KernelExplainer(predict_proba, background_data)
        
        # 计算SHAP值（使用50个评估样本）
        eval_data = background_data[:50]
        
        shap_values = explainer.shap_values(eval_data)
        
        # 处理多维输出（取第一个类别-糖尿病阳性）
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # 计算每个特征的平均绝对SHAP值作为重要性
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # 构建特征重要性列表
        importance_list = []
        for i, (name, importance) in enumerate(zip(feature_names, mean_abs_shap)):
            importance_list.append({
                'feature': name,
                'importance': float(np.ravel(importance)[0]) if hasattr(importance, '__len__') else float(importance),
                'index': i
            })
        
        # 按重要性排序
        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        
        # 获取模型版本
        current_model_version = model_version if model_version else 'Unknown'
        
        return jsonify({
            'success': True,
            'features': importance_list,
            'model_version': current_model_version,
            'method': 'SHAP KernelExplainer',
            'model_type': 'HTTP_Federated' if is_http_model else 'Federated'
        })
        
    except Exception as e:
        import traceback
        print(f"SHAP特征分析错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'SHAP分析失败: {str(e)}'
        }), 500

# 获取最后一次生成的客户端数量
@app.route('/api/last_generated_clients')
def get_last_generated_clients():
    global last_generated_clients
    return jsonify({
        'num_clients': last_generated_clients if last_generated_clients else 3
    })



# 用户画像页面路由（仅限普通用户）
@app.route('/profile')
def profile_page():
    if 'user' not in session:
        return redirect('/login')
    if session.get('role') == 'admin':
        return redirect('/main')
    return render_template('profile.html')

# 测试预测页面路由
@app.route('/test_predict')
def test_predict_page():
    return render_template('test_predict.html')

# 预测页面路由（仅限普通用户）
@app.route('/predict')
def predict_page():
    if 'user' not in session:
        return redirect('/login')
    # 限制只有普通用户可以访问，管理员自动跳转至后台
    if session.get('role') == 'admin':
        return redirect('/main')
    return render_template('predict.html')

# AI健康管理报告页面（仅限普通用户）
@app.route('/report')
def report_page():
    if 'user' not in session:
        return redirect('/login')
    if session.get('role') == 'admin':
        return redirect('/main')
    return render_template('report.html')

# 预测API端点
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # 获取请求数据
        if not request.is_json:
            return jsonify({
                'success': False,
                'message': '请求需要JSON格式'
            }), 400
        
        input_data = request.json
        
        # 进行预测
        result, error = predict_diabetes(input_data)
        
        if error:
            return jsonify({
                'success': False,
                'message': error
            }), 400
        
        # 如果用户已登录，自动保存预测数据到用户画像
        print(f"[DEBUG] Session内容: {dict(session)}")
        if 'user' in session:
            print(f"[DEBUG] 用户已登录: {session['user']}")
            save_prediction_to_profile(input_data, result['probability'])
        else:
            print("[DEBUG] 用户未登录，跳过保存预测数据")
        
        return jsonify({
            'success': True,
            'probability': result['probability'],
            'model_version': result['model_version']
        })
        
    except Exception as e:
        print(f"预测API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'内部服务器错误: {str(e)}'
        }), 500

def save_prediction_to_profile(input_data, probability):
    """将预测数据保存到用户画像"""
    try:
        if 'user' not in session:
            print("[WARN] 用户未登录，跳过保存预测数据")
            return
        
        username = session['user']
        
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(DictCursor)
            
            # 先获取用户ID
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            if not user:
                print(f"[ERROR] 用户 {username} 不存在")
                cursor.close()
                conn.close()
                return
            
            user_id = user['id']
            
            # 准备所有特征字段
            feature_order = [
                '年龄', '性别', '种族', '社会经济地位', '教育水平', '体重指数',
                '吸烟状态', '饮酒量', '每周体育活动时间', '饮食质量', '睡眠质量',
                '糖尿病家族史', '妊娠期糖尿病', '多囊卵巢综合征', '既往糖尿病前期', '高血压',
                '收缩压', '舒张压', '空腹血糖', '糖化血红蛋白',
                '血清肌酐', '血尿素氮水平', '总胆固醇', '低密度脂蛋白胆固醇', '高密度脂蛋白胆固醇', '甘油三酯',
                '降压药物使用', '他汀类药物使用', '抗糖尿病药物使用',
                '尿频', '过度口渴', '不明原因体重下降', '疲劳程度', '视力模糊', '伤口愈合缓慢', '手脚刺痛',
                '生活质量评分', '重金属暴露', '职业化学物质暴露', '水质', '体检频率', '药物依从性', '健康素养'
            ]
            
            # 构建插入数据
            values = [user_id]
            
            for feature in feature_order:
                if feature in input_data:
                    values.append(input_data[feature])
                else:
                    values.append(None)
            
            # 添加风险概率（预测风险）
            values.append(probability)
            
            # 构建SQL语句
            columns = ', '.join(['user_id'] + feature_order + ['预测风险'])
            placeholders = ', '.join(['%s'] * len(values))
            
            sql = f"INSERT INTO user_metrics ({columns}) VALUES ({placeholders})"
            
            cursor.execute(sql, tuple(values))
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"[OK] 用户 {user_id} 的预测数据已保存到用户画像，风险概率: {probability}")
    except Exception as e:
        print(f"[ERROR] 保存预测数据到用户画像失败: {str(e)}")
        import traceback
        traceback.print_exc()

# DeepSeek AI 健康管理报告 API
@app.route('/api/health_report', methods=['POST'])
def generate_health_report():
    """使用 DeepSeek 模型生成个性化健康管理报告"""
    try:
        input_data = request.json
        
        # 导入 AI 报告生成模块
        from ai_report import generate_health_report as ai_report_func
        
        # 调用 AI 报告生成函数
        report, error = ai_report_func(input_data)
        
        if error:
            return jsonify({
                'success': False,
                'message': error
            }), 500
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        print(f"生成健康报告失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'AI报告生成失败: {str(e)}'
        }), 500

# 用户指标数据 API

@app.route('/api/user/metrics', methods=['POST'])
def save_user_metrics():
    """保存用户指标数据（支持所有44个特征）"""
    if 'user' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401

    try:
        data = request.json
        user_id, error = get_user_id_by_username(session['user'])
        if error:
            return jsonify({'success': False, 'message': error}), 404

        success, message = utils_save_user_metrics(user_id, data)
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message}), 500

    except Exception as e:
        print(f"保存用户指标失败: {str(e)}")
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'}), 500

@app.route('/api/user/metrics', methods=['GET'])
def get_user_metrics():
    """获取用户的所有历史指标数据"""
    if 'user' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401

    try:
        user_id, error = get_user_id_by_username(session['user'])
        if error:
            return jsonify({'success': False, 'message': error}), 404

        records, error = utils_get_user_metrics(user_id)
        if error:
            return jsonify({'success': False, 'message': error}), 500

        return jsonify({'success': True, 'data': records})

    except Exception as e:
        print(f"获取用户指标失败: {str(e)}")
        return jsonify({'success': False, 'message': f'获取失败: {str(e)}'}), 500

@app.route('/api/training/results', methods=['GET'])
def get_training_results():
    """获取所有训练结果历史记录（所有用户可访问）"""
    if 'user' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401

    try:
        records, error = get_all_training_results(50)
        if error:
            return jsonify({'success': False, 'message': error}), 500

        return jsonify({'success': True, 'data': records})

    except Exception as e:
        print(f"获取训练结果失败: {str(e)}")
        return jsonify({'success': False, 'message': f'获取失败: {str(e)}'}), 500

@app.route('/api/training/results/<int:result_id>', methods=['GET'])
def get_training_result_detail(result_id):
    """获取指定训练结果的详细信息"""
    if 'user' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'message': '需要管理员权限'}), 403

    try:
        record, error = get_training_result_by_id(result_id)
        if error:
            return jsonify({'success': False, 'message': error}), 500

        if not record:
            return jsonify({'success': False, 'message': '记录不存在'}), 404

        return jsonify({'success': True, 'data': record})

    except Exception as e:
        print(f"获取训练结果详情失败: {str(e)}")
        return jsonify({'success': False, 'message': f'获取失败: {str(e)}'}), 500

@app.route('/api/user/profile', methods=['GET'])
def get_user_profile():
    """获取用户最新的指标数据（用户画像）"""
    if 'user' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401
    
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': '数据库连接失败'}), 500
        
        cursor = conn.cursor(DictCursor)
        cursor.execute("SELECT id FROM users WHERE username = %s", (session['user'],))
        user = cursor.fetchone()
        if not user:
            return jsonify({'success': False, 'message': '用户不存在'}), 404
        
        user_id = user['id']
        
        cursor.execute('''
            SELECT * FROM user_metrics 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (user_id,))
        
        profile = cursor.fetchone()
        cursor.close()
        
        return jsonify({'success': True, 'profile': profile if profile else {}})
        
    except Exception as e:
        print(f"获取用户画像失败: {str(e)}")
        return jsonify({'success': False, 'message': f'获取失败: {str(e)}'}), 500

@app.route('/api/user/metrics/<int:record_id>', methods=['DELETE'])
def delete_user_metrics(record_id):
    """删除指定的指标记录"""
    if 'user' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401
    
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': '数据库连接失败'}), 500
        
        cursor = conn.cursor()
        
        # 检查记录是否属于当前用户
        cursor.execute('''
            SELECT user_id FROM user_metrics WHERE id = %s
        ''', (record_id,))
        
        record = cursor.fetchone()
        if not record:
            return jsonify({'success': False, 'message': '记录不存在'}), 404
        
        # 获取当前用户ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (session['user'],))
        user = cursor.fetchone()
        
        if record['user_id'] != user['id']:
            return jsonify({'success': False, 'message': '无权删除此记录'}), 403
        
        # 删除记录
        cursor.execute("DELETE FROM user_metrics WHERE id = %s", (record_id,))
        conn.commit()
        cursor.close()
        
        return jsonify({'success': True, 'message': '删除成功'})
        
    except Exception as e:
        print(f"删除用户指标失败: {str(e)}")
        return jsonify({'success': False, 'message': f'删除失败: {str(e)}'}), 500

# 用户健康分析 API
@app.route('/api/user/analysis', methods=['POST'])
def user_analysis():
    """分析用户两次预测之间的指标变化，生成专业健康分析报告"""
    if 'user' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401
    
    try:
        data = request.json
        
        if not data or 'changes' not in data:
            return jsonify({'success': False, 'message': '缺少数据'}), 400
        
        changes = data['changes']
        
        # 定义需要关注的指标及其健康范围
        health_ranges = {
            '体重指数': {'normal': (18.5, 24), 'unit': ''},
            '收缩压': {'normal': (90, 120), 'unit': 'mmHg'},
            '舒张压': {'normal': (60, 80), 'unit': 'mmHg'},
            '空腹血糖': {'normal': (3.9, 6.1), 'unit': 'mmol/L'},
            '糖化血红蛋白': {'normal': (4.0, 6.0), 'unit': '%'},
            '总胆固醇': {'normal': (0, 5.2), 'unit': 'mmol/L'},
            '甘油三酯': {'normal': (0, 1.7), 'unit': 'mmol/L'},
            '低密度脂蛋白胆固醇': {'normal': (0, 3.4), 'unit': 'mmol/L'},
            '高密度脂蛋白胆固醇': {'normal': (1.04, float('inf')), 'unit': 'mmol/L'}
        }
        
        # 分析正面变化
        positive_changes = []
        # 分析负面变化
        negative_changes = []
        # 分析超出正常范围的指标
        out_of_range = []
        
        for change in changes:
            label = change['label']
            newValue = change['newValue']
            diff = change['diff']
            percentage = change['percentage']
            
            # 判断是否超出正常范围
            if label in health_ranges:
                normal_range = health_ranges[label]
                if not (normal_range['normal'][0] <= newValue <= normal_range['normal'][1]):
                    out_of_range.append({
                        'label': label,
                        'value': newValue,
                        'range': normal_range,
                        'is_high': newValue > normal_range['normal'][1]
                    })
            
            # 判断变化的好坏
            bad_metrics = ['收缩压', '舒张压', '空腹血糖', '糖化血红蛋白', '总胆固醇', '甘油三酯', '低密度脂蛋白胆固醇', '血清肌酐', '疲劳程度']
            isBadMetric = label in bad_metrics
            isGoodChange = diff < 0 if isBadMetric else diff > 0
            
            if isGoodChange and abs(percentage) >= 5:
                positive_changes.append(f"{label} {'下降' if isBadMetric else '上升'} {abs(percentage):.1f}%，从 {change['oldValue']:.1f} 变为 {newValue:.1f}")
            elif not isGoodChange and abs(percentage) >= 5:
                negative_changes.append(f"{label} {'上升' if isBadMetric else '下降'} {abs(percentage):.1f}%，从 {change['oldValue']:.1f} 变为 {newValue:.1f}")
        
        # 生成分析报告
        analysis = {
            'overview': generate_overview(changes, out_of_range),
            'positiveChanges': positive_changes,
            'negativeChanges': negative_changes,
            'suggestions': generate_suggestions(changes, out_of_range)
        }
        
        return jsonify({'success': True, 'analysis': analysis})
        
    except Exception as e:
        print(f"用户分析失败: {str(e)}")
        return jsonify({'success': False, 'message': f'分析失败: {str(e)}'}), 500

def generate_overview(changes, out_of_range):
    """生成总体评价"""
    positive_count = sum(1 for c in changes if abs(c['percentage']) >= 5 and 
                        (c['diff'] < 0 if c['label'] in ['收缩压', '舒张压', '空腹血糖', '糖化血红蛋白', '总胆固醇', '甘油三酯', '低密度脂蛋白胆固醇', '血清肌酐', '疲劳程度'] else c['diff'] > 0))
    
    negative_count = sum(1 for c in changes if abs(c['percentage']) >= 5 and 
                        (c['diff'] > 0 if c['label'] in ['收缩压', '舒张压', '空腹血糖', '糖化血红蛋白', '总胆固醇', '甘油三酯', '低密度脂蛋白胆固醇', '血清肌酐', '疲劳程度'] else c['diff'] < 0))
    
    if positive_count > negative_count:
        if len(out_of_range) == 0:
            return "您的健康状况总体向好！多项指标出现积极变化，请继续保持健康的生活方式。"
        else:
            return f"您的健康状况总体向好，但有 {len(out_of_range)} 项指标超出正常范围，需要关注。"
    elif negative_count > positive_count:
        return f"检测到 {negative_count} 项指标出现明显下降趋势，建议关注并调整生活习惯。"
    else:
        if len(out_of_range) == 0:
            return "您的健康状况保持稳定，各项指标变化不大，请继续保持。"
        else:
            return f"您的健康状况基本稳定，但有 {len(out_of_range)} 项指标超出正常范围，建议咨询医生。"

def generate_suggestions(changes, out_of_range):
    """生成健康建议"""
    suggestions = []
    
    # 基于超出范围的指标生成建议
    for item in out_of_range:
        label = item['label']
        is_high = item['is_high']
        
        if label == '体重指数':
            suggestions.append(f"BMI{'' if is_high else '偏低'}，建议{'控制饮食，增加运动' if is_high else '增加营养摄入'}")
        elif label == '收缩压' or label == '舒张压':
            suggestions.append(f"血压偏{'高' if is_high else '低'}，建议{'低盐饮食，规律作息' if is_high else '适当补充水分，避免久坐'}")
        elif label == '空腹血糖':
            suggestions.append(f"空腹血糖偏{'高' if is_high else '低'}，建议{'控制碳水化合物摄入' if is_high else '规律饮食'}")
        elif label == '糖化血红蛋白':
            suggestions.append(f"糖化血红蛋白偏{'高' if is_high else '低'}，建议{'定期监测血糖' if is_high else '咨询医生'}")
        elif label == '总胆固醇':
            suggestions.append("总胆固醇偏高，建议减少高胆固醇食物摄入，增加膳食纤维")
        elif label == '甘油三酯':
            suggestions.append("甘油三酯偏高，建议减少糖分摄入，增加有氧运动")
        elif label == '低密度脂蛋白胆固醇':
            suggestions.append("LDL偏高，建议低脂饮食，遵医嘱服用降脂药物")
        elif label == '高密度脂蛋白胆固醇':
            suggestions.append("HDL偏低，建议增加不饱和脂肪酸摄入，如鱼类、坚果")
    
    # 基于变化趋势生成建议
    for change in changes:
        label = change['label']
        diff = change['diff']
        percentage = change['percentage']
        
        if label == '每周体育活动时间':
            if diff > 0:
                suggestions.append("运动时间增加，继续保持规律运动习惯")
            else:
                suggestions.append("运动时间减少，建议每周保持至少150分钟中等强度运动")
        elif label == '饮食质量':
            if diff < 0:
                suggestions.append("饮食质量下降，建议增加蔬菜、水果摄入，减少加工食品")
        elif label == '睡眠质量':
            if diff < 0:
                suggestions.append("睡眠质量下降，建议保持规律作息，改善睡眠环境")
        elif label == '疲劳程度':
            if diff > 0:
                suggestions.append("疲劳程度增加，建议合理安排工作休息，适当放松")
    
    # 去重并取前10条建议
    unique_suggestions = list(dict.fromkeys(suggestions))[:10]
    
    if len(unique_suggestions) == 0:
        unique_suggestions.append("继续保持健康的生活方式，定期进行健康检查")
    
    return unique_suggestions

if __name__ == '__main__':
    print("启动Flask服务器...")
    print("访问 http://localhost:5000 查看页面")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
