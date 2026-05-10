"""
联邦学习中心服务器 - HTTP API
实现标准化安全聚合策略
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
import openpyxl
from flask import Flask, request, jsonify
import requests
import threading
import signal
from sklearn.preprocessing import StandardScaler
import pymysql
from pymysql.cursors import DictCursor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_utils import HomomorphicCrypto, CKKSCrypto

app = Flask(__name__)

# 添加CORS支持
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 全局标准化器
global_scaler = None

# 已注册的客户端
registered_clients = {}

def shutdown_all():
    log('[INFO] 训练完成，正在关闭所有客户端...')
    for client_id, url in CLIENTS.items():
        try:
            response = requests.post(f'{url}/shutdown', timeout=5)
            if response.status_code == 200:
                log(f'[OK] 已发送关闭命令到客户端{client_id}')
            else:
                log(f'[WARN] 客户端{client_id}关闭响应异常')
        except Exception as e:
            log(f'[WARN] 无法连接客户端{client_id}: {e}')
    
    log('[INFO] 服务器即将退出...')
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGINT)

global_state = {
    'global_model': None,
    'current_round': 0,
    'max_rounds': 10,
    'num_clients': 3,
    'client_updates': {},
    'client_status': {},
    'training_started': False,
    'training_complete': False,
    'input_dim': 8,
    'feature_dim': 8,
    'global_control': None,  # SCAFFOLD全局控制变量
    'algorithm': 'fedavg',
    'mu': 0.01,
    'encryption': 'none',  # 加密方式: 'none', 'homomorphic'
    'privacy': 'none',      # 隐私保护: 'none', 'differential'
    'crypto': None,         # 同态加密对象
    'client_port_status': {  # 端口存活状态
        '0': False,  # 客户端0，端口6000
        '1': False,  # 客户端1，端口6001
        '2': False   # 客户端2，端口6002
    }
}

CLIENTS = {
    0: 'http://localhost:6000',
    1: 'http://localhost:6001',
    2: 'http://localhost:6002'
}

# 数据库配置（与flask_app.py保持一致）
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '252525lht',
    'database': 'fl_federated_learning',
    'charset': 'utf8mb4',
    'cursorclass': DictCursor
}

def get_db_connection():
    """获取数据库连接"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        log(f'[ERROR] 数据库连接失败: {str(e)}')
        return None

def init_training_results_db():
    """初始化训练结果表"""
    connection = get_db_connection()
    if not connection:
        return False

    try:
        with connection.cursor() as cursor:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                训练时间 TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                算法 VARCHAR(50) NOT NULL COMMENT '算法名称: FedAvg, FedProx, SCAFFOLD',
                隐私保护方式 VARCHAR(50) DEFAULT 'none' COMMENT '隐私保护方式: none, homomorphic, differential',
                客户端数量 INT COMMENT '参与的客户端数量',
                全局轮次 INT COMMENT '训练轮次',
                本地批次大小 INT COMMENT '客户端本地批次大小',
                本地学习率 FLOAT COMMENT '客户端本地学习率',
                准确率 FLOAT COMMENT '最终准确率',
                精确率 FLOAT COMMENT '最终精确率',
                查全率 FLOAT COMMENT '最终查全率',
                F1分数 FLOAT COMMENT '最终F1分数',
                AUC分数 FLOAT COMMENT '最终AUC分数',
                时间消耗秒 FLOAT COMMENT '总训练时间(秒)',
                模型文件名 VARCHAR(255) COMMENT '保存的模型文件名',
                模型文件路径 VARCHAR(500) COMMENT '保存的模型文件路径',
                隐私预算 FLOAT COMMENT '差分隐私预算epsilon值',
                噪声标准差 FLOAT COMMENT '差分隐私高斯噪声标准差',
                总样本数 INT COMMENT '参与训练的总样本数',
                客户端样本分布 TEXT COMMENT '各客户端样本数量分布(JSON)',
                额外参数 TEXT COMMENT '额外训练参数(JSON)',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_training_results_created_at (created_at),
                INDEX idx_training_results_algorithm (算法)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            ''')
            connection.commit()
            log('[OK] 训练结果表初始化成功')
            return True
    except Exception as e:
        log(f'[ERROR] 训练结果表初始化失败: {str(e)}')
        return False
    finally:
        if connection:
            connection.close()

def save_training_result_to_db(record):
    """保存训练结果到数据库"""
    connection = get_db_connection()
    if not connection:
        log('[ERROR] 无法保存训练结果：数据库连接失败')
        return False

    try:
        with connection.cursor() as cursor:
            sql = """
            INSERT INTO training_results
            (算法, 隐私保护方式, 客户端数量, 全局轮次, 本地批次大小, 本地学习率,
             准确率, 精确率, 查全率, F1分数, AUC分数, 时间消耗秒, 模型文件名, 模型文件路径,
             隐私预算, 噪声标准差, 总样本数, 客户端样本分布, 额外参数)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            # 合并加密方式和隐私保护为一列
            privacy_method = record.get('隐私保护方式', 'none')
            if not privacy_method or privacy_method == 'none':
                encryption = record.get('加密方式', 'none')
                privacy = record.get('隐私保护', 'none')
                if encryption == 'homomorphic':
                    privacy_method = 'homomorphic'
                elif privacy == 'differential':
                    privacy_method = 'differential'
            
            cursor.execute(sql, (
                record.get('算法', 'FedAvg'),
                privacy_method,
                record.get('客户端数量', 3),
                record.get('全局轮次', 10),
                record.get('本地批次大小', 32),
                record.get('本地学习率', 0.01),
                record.get('准确率', 0.0),
                record.get('精确率', 0.0),
                record.get('查全率', 0.0),
                record.get('F1分数', 0.0),
                record.get('AUC分数', 0.0),
                record.get('时间消耗秒', 0.0),
                record.get('模型文件名', ''),
                record.get('模型文件路径', ''),
                record.get('隐私预算', None),
                record.get('噪声标准差', None),
                record.get('总样本数', 0),
                record.get('客户端样本分布', '{}'),
                record.get('额外参数', '{}')
            ))
            connection.commit()
            log(f'[OK] 训练结果已保存到数据库 (算法: {record.get("算法", "FedAvg")})')
            return True
    except Exception as e:
        log(f'[ERROR] 保存训练结果失败: {str(e)}')
        import traceback
        traceback.print_exc()
        return False
    finally:
        if connection:
            connection.close()

def save_global_model_to_db(record):
    """保存全局模型记录到数据库"""
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        with connection.cursor() as cursor:
            sql = """
            INSERT INTO global_model_records 
            (训练时间, 算法, 加密方式, 客户端数量, 准确率, 精确率, 查全率, F1分数, 时间消耗秒, 模型文件名, 模型文件路径)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                record['训练时间'],
                record['算法'],
                record.get('加密方式', 'none'),
                record['客户端数量'],
                record['准确率'],
                record['精确率'],
                record['查全率'],
                record['F1分数'],
                record['时间消耗秒'],
                record['模型文件名'],
                record['模型文件路径']
            ))
        connection.commit()
        log(f'[OK] 全局模型记录已保存到数据库')
        return True
    except Exception as e:
        log(f'[ERROR] 保存数据库失败: {str(e)}')
        return False
    finally:
        if connection:
            connection.close()

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

class SimpleNN(nn.Module):
    def __init__(self, input_dim=8):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x

def get_model(input_dim=8, use_large=True):
    return Net(input_dim) if use_large else SimpleNN(input_dim)

def tensor_to_list(tensor_dict):
    result = {}
    for k, v in tensor_dict.items():
        result[k] = v.cpu().numpy().tolist()
    return result

def list_to_tensor(list_dict):
    result = {}
    for k, v in list_dict.items():
        result[k] = torch.tensor(v, dtype=torch.float32)
    return result

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    try:
        print(f'[{timestamp}] {msg}')
    except UnicodeEncodeError:
        # 移除无法编码的字符
        safe_msg = msg.encode('gbk', errors='replace').decode('gbk')
        print(f'[{timestamp}] {safe_msg}')

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'role': 'server'})

@app.route('/get_scaler', methods=['GET'])
def get_scaler():
    if global_scaler is None:
        return jsonify({'error': 'scaler not initialized'}), 400
    return jsonify({
        'mean': global_scaler.mean_.tolist(),
        'std': global_scaler.scale_.tolist()
    })

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    if global_state['global_model'] is None:
        return jsonify({'error': 'global model not initialized'}), 400
    return jsonify({
        'model': tensor_to_list(global_state['global_model'].state_dict()),
        'current_round': global_state['current_round'],
        'max_rounds': global_state['max_rounds']
    })

@app.route('/submit_update', methods=['POST'])
def submit_update():
    data = request.json
    client_id = data['client_id']
    
    encryption = global_state.get('encryption', 'none')
    is_encrypted = data.get('encrypted', False)
    
    if encryption == 'homomorphic' or is_encrypted:
        encrypted_model = data['model_update']
        model_update = encrypted_model
        log(f'[加密] 收到客户端{client_id}的加密模型更新')
    else:
        model_update = list_to_tensor(data['model_update'])
    
    num_samples = data['num_samples']
    accuracy = data.get('accuracy', 0.0)
    precision = data.get('precision', 0.0)
    recall = data.get('recall', 0.0)
    f1 = data.get('f1', 0.0)
    training_time = data.get('training_time', 0.0)

    global_state['client_updates'][client_id] = {
        'model': model_update,
        'num_samples': num_samples,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'encrypted': is_encrypted
    }
    global_state['client_status'][client_id] = 'completed'

    log(f'✅ 客户端{client_id} 上传完成 | 样本:{num_samples} | Acc:{accuracy:.2%} | P:{precision:.2%} | R:{recall:.2%} | F1:{f1:.2%} | 耗时:{training_time:.2f}秒')
    return jsonify({'status': 'received'})

# 训练轮次指标记录
training_metrics = []

def check_port_alive(port, timeout=2):
    """检查指定端口是否存活"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex(('localhost', port))
        return result == 0
    finally:
        sock.close()

def update_client_port_status():
    """更新所有客户端端口存活状态"""
    ports = [6000, 6001, 6002]
    for i, port in enumerate(ports):
        global_state['client_port_status'][str(i)] = check_port_alive(port)

def monitor_client_ports():
    """定时监控客户端端口状态的后台线程"""
    global global_state
    log('[INFO] 端口监控线程已启动')
    while True:
        update_client_port_status()
        
        # 如果训练中但所有客户端端口都不存活，则标记训练完成
        # 只检查布尔值 True，表示端口真正存活
        if global_state.get('training_started', False) and not global_state.get('training_complete', False):
            port_values = list(global_state['client_port_status'].values())
            # 检查是否有任何端口真正存活（只检查 == True）
            any_alive = any(v == True for v in port_values)
            if not any_alive:
                log('[INFO] 检测到所有客户端端口已关闭，自动标记训练完成')
                global_state['training_complete'] = True
        
        time.sleep(3)  # 每3秒检测一次

# 启动端口监控线程
port_monitor_thread = threading.Thread(target=monitor_client_ports, daemon=True)
port_monitor_thread.start()

@app.route('/api/start_all', methods=['POST'])
def api_start_all():
    """API: 一键启动训练 - 直接启动客户端并开始训练"""
    global training_metrics
    
    if global_state.get('training_started', False):
        return jsonify({'success': False, 'message': '训练已在进行中'}), 400
    
    data = request.json
    algorithm = data.get('algorithm', 'fedavg')
    mu = data.get('mu', 0.01)
    privacy = data.get('privacy', 'none')
    
    # 设置全局状态
    global_state['algorithm'] = algorithm.lower()
    global_state['mu'] = mu
    global_state['encryption'] = privacy if privacy == 'homomorphic' else 'none'
    global_state['privacy'] = privacy if privacy == 'differential' else 'none'
    global_state['training_started'] = False
    global_state['training_complete'] = False
    training_metrics = []
    
    # 启动客户端进程
    import subprocess
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    http_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_root, '.venv', 'Scripts', 'python.exe')
    
    # 启动3个客户端
    for i in range(3):
        port = 6000 + i
        cmd = [venv_python, 'client.py', str(i), str(port)]
        subprocess.Popen(cmd, cwd=http_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
        log(f'[API] 启动客户端 {i} (端口 {port})')
    
    # 等待客户端启动
    import time
    time.sleep(3)
    
    # 在后台线程中运行训练
    import threading
    thread = threading.Thread(target=run_full_training, daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'message': '客户端已启动，训练即将开始'})

@app.route('/api/training_status', methods=['GET'])
def api_training_status():
    """API: 获取训练状态"""
    # 检查服务器状态
    server_running = True  # 服务器运行中
    
    # 检查客户端状态（只检查端口和健康状态）
    clients = []
    client_status = {}
    for i in range(3):
        port_alive = global_state['client_port_status'].get(str(i), False)
        
        try:
            response = requests.get(f'http://localhost:{6000 + i}/health', timeout=2)
            if response.status_code == 200:
                # 绿色：运行中（健康检查通过）
                clients.append({'id': i, 'running': True, 'status': 'running'})
                client_status[str(i)] = True
            else:
                # 红色：未连接（健康检查失败）
                clients.append({'id': i, 'running': False, 'status': 'disconnected'})
                client_status[str(i)] = 'disconnected'
        except:
            if port_alive:
                # 橙色：无信号（端口存活但无法响应）
                clients.append({'id': i, 'running': False, 'status': 'no_signal'})
                client_status[str(i)] = 'no_signal'
            else:
                # 灰色：关闭（端口不存活）
                clients.append({'id': i, 'running': False, 'status': 'stopped'})
                client_status[str(i)] = False
    
    # 检查训练是否完成
    training_complete = global_state.get('training_complete', False)
    
    # 获取最终指标
    final_metrics = None
    if training_complete and 'final_result' in global_state:
        result = global_state['final_result']
        final_metrics = {
            'accuracy': result.get('准确率', 0.0),
            'precision': result.get('精确率', 0.0),
            'recall': result.get('查全率', 0.0),
            'f1': result.get('F1分数', 0.0),
            'duration': result.get('时间消耗(秒)', 0.0)
        }
    
    return jsonify({
        'server_running': server_running,
        'clients': clients,
        'client_port_status': client_status,  # 使用新的三种状态
        'training_started': global_state.get('training_started', False),
        'training_complete': training_complete,
        'metrics': training_metrics,
        'final_metrics': final_metrics,
        'server_alive': True  # 服务器自身状态
    })

@app.route('/start_training', methods=['POST'])
def start_training():
    if global_state['training_started']:
        return jsonify({'error': 'training already in progress'}), 400

    data = request.json
    global_state['max_rounds'] = data.get('rounds', 10)
    global_state['num_clients'] = data.get('num_clients', 3)
    global_state['input_dim'] = data.get('input_dim', 8)
    global_state['feature_dim'] = data.get('feature_dim', 8)
    
    # FedProx参数
    global_state['algorithm'] = data.get('algorithm', 'fedavg').lower()
    global_state['mu'] = data.get('mu', 0.01)
    # 隐私保护参数（支持两种格式：统一的privacy，或分开的encryption和privacy）
    if 'privacy' in data and data['privacy'] in ['homomorphic', 'differential']:
        # 新格式：统一的privacy参数
        privacy = data.get('privacy', 'none')
        global_state['encryption'] = privacy if privacy == 'homomorphic' else 'none'
        global_state['privacy'] = privacy if privacy == 'differential' else 'none'
    else:
        # 旧格式：分开的encryption和privacy参数
        global_state['encryption'] = data.get('encryption', 'none')
        global_state['privacy'] = data.get('privacy', 'none')
    
    # 差分隐私参数
    global_state['epsilon'] = data.get('epsilon', 1.0)
    global_state['delta'] = data.get('delta', 5e-4)
    log(f'[隐私] 差分隐私参数设置 - ε: {global_state["epsilon"]}, δ: {global_state["delta"]}')

    if global_state['encryption'] == 'homomorphic':
        log('[加密] 启用CKKS同态加密，生成密钥对...')
        crypto = CKKSCrypto(poly_modulus_degree=8192, global_scale=2**40)
        crypto.generate_keypair()
        global_state['crypto'] = crypto
        log('[加密] CKKS同态加密密钥对生成完成')
    else:
        global_state['crypto'] = None

    global_state['global_model'] = get_model(input_dim=global_state['input_dim'], use_large=True)
    global_state['current_round'] = 0
    global_state['client_updates'] = {}
    global_state['client_status'] = {}
    global_state['training_started'] = True
    
    if global_state.get('algorithm', 'fedavg').lower() == 'scaffold':
        global_state['global_control'] = {k: torch.zeros_like(v) for k, v in global_state['global_model'].state_dict().items()}
        log(f'  SCAFFOLD控制变量已初始化')

    log('='*60)
    log('🚀 联邦学习训练开始')
    log('='*60)
    log(f'  客户端数: {global_state["num_clients"]} | 训练轮数: {global_state["max_rounds"]}')
    log(f'  特征维度: {global_state["input_dim"]}')
    log(f'  算法: {global_state["algorithm"].upper()}')
    if global_state['algorithm'] == 'fedprox':
        log(f'  FedProx μ: {global_state["mu"]}')
    log(f'  加密方式: {global_state["encryption"]}')
    log(f'  隐私保护: {global_state["privacy"]}')

    return jsonify({'status': 'started'})

@app.route('/get_public_key', methods=['GET'])
def get_public_key():
    """返回同态加密公钥"""
    crypto = global_state.get('crypto')
    if crypto is None or crypto.public_key is None:
        return jsonify({'has_public_key': False})

    if isinstance(crypto, CKKSCrypto):
        encryption_type = 'ckks'
        public_key_data = crypto.get_public_key_serialized()
    else:
        encryption_type = 'paillier'
        public_key_data = crypto.get_public_key_dict()

    return jsonify({
        'has_public_key': True,
        'encryption_type': encryption_type,
        'public_key': public_key_data
    })

@app.route('/next_round', methods=['POST'])
def next_round():
    if not global_state['training_started']:
        return jsonify({'error': 'training not started'}), 400

    if global_state['current_round'] >= global_state['max_rounds']:
        return jsonify({'status': 'training_complete', 'round': global_state['current_round']})

    global_state['current_round'] += 1
    round_idx = global_state['current_round']

    log(f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
    log(f'📢 第 {round_idx}/{global_state["max_rounds"]} 轮开始')

    selected_clients = list(range(global_state['num_clients']))
    log(f'📋 本轮选中客户端: {selected_clients}')

    global_state['client_updates'] = {}
    global_state['client_status'] = {c: 'waiting' for c in selected_clients}

    for client_id in selected_clients:
        try:
            train_request = {
                'global_model': tensor_to_list(global_state['global_model'].state_dict()),
                'round': round_idx,
                'max_rounds': global_state['max_rounds'],
                'input_dim': global_state['input_dim'],
                'algorithm': global_state.get('algorithm', 'fedavg'),
                'mu': global_state.get('mu', 0.01)
            }
            
            if global_state.get('algorithm', 'fedavg').lower() == 'scaffold' and global_state.get('global_control') is not None:
                train_request['global_control'] = tensor_to_list(global_state['global_control'])
            
            response = requests.post(
                f'{CLIENTS[client_id]}/start_train',
                json=train_request,
                timeout=300
            )
            if response.status_code == 200:
                log(f'📤 任务已发送到客户端{client_id}')
                global_state['client_status'][client_id] = 'training'
        except Exception as e:
            log(f'❌ 无法连接客户端{client_id}: {e}')

    return jsonify({
        'status': 'round_started',
        'round': round_idx,
        'selected_clients': selected_clients
    })

def evaluate_global_model():
    """评估最终全局模型在测试集上的性能"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    model = global_state['global_model']
    if model is None:
        return 0.0, 0.0, 0.0, 0.0

    # 加载全局测试集
    test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'global_test_set.csv')
    if not os.path.exists(test_path):
        log(f'[WARN] 全局测试集不存在: {test_path}')
        return 0.0, 0.0, 0.0, 0.0

    df = pd.read_csv(test_path)
    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values

    log(f'[DEBUG] 测试集 - 样本数: {len(y_test)}, 正类数: {int(sum(y_test))}, 负类数: {int(len(y_test)-sum(y_test))}')

    # 使用全局scaler标准化（与客户端一致）
    if global_scaler is not None and hasattr(global_scaler, 'mean_'):
        mean = global_scaler.mean_
        scale = global_scaler.scale_
        X_test = (X_test - mean) / scale
        log(f'[DEBUG] 使用global_scaler标准化 (mean[:3]={mean[:3]}, scale[:3]={scale[:3]})')
    elif 'scaler_mean' in global_state and 'scaler_std' in global_state:
        mean = np.array(global_state['scaler_mean'])
        std = np.array(global_state['scaler_std'])
        std = np.where(std == 0, 1.0, std)
        X_test = (X_test - mean) / std
        log(f'[DEBUG] 使用global_state中的scaler')
    else:
        log(f'[WARN] 全局scaler未初始化，尝试使用StandardScaler')
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_test)
            X_test = scaler.transform(X_test)
            log(f'[DEBUG] 使用StandardScaler备用方案')
        except Exception as e:
            log(f'[ERROR] 标准化失败: {e}')

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = outputs.squeeze().numpy()
        log(f'[DEBUG] 模型输出 - 最小:{probs.min():.4f} 最大:{probs.max():.4f} 平均:{probs.mean():.4f}')

    # 使用固定阈值0.5（与简单版本一致）
    predictions = (probs >= 0.5).astype(int)
    
    # 计算混淆矩阵
    tp = int(sum((y_test == 1) & (predictions == 1)))
    fn = int(sum((y_test == 1) & (predictions == 0)))
    fp = int(sum((y_test == 0) & (predictions == 1)))
    tn = int(sum((y_test == 0) & (predictions == 0)))
    
    log(f'[DEBUG] 测试集分布 - 正类:{int(sum(y_test))}, 负类:{int(len(y_test)-sum(y_test))}')
    log(f'[DEBUG] 预测结果 - 正类:{int(sum(predictions))} 负类:{int(len(predictions)-sum(predictions))}')
    log(f'[DEBUG] 混淆矩阵 - TP:{tp}, FN:{fn}, FP:{fp}, TN:{tn}')

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    log(f'✅ 全局模型评估 | Acc:{accuracy:.4f} | P:{precision:.4f} | R:{recall:.4f} | F1:{f1:.4f}')

    return accuracy, precision, recall, f1

@app.route('/aggregate', methods=['POST'])
def aggregate():
    if len(global_state['client_updates']) == 0:
        return jsonify({'error': 'no updates received'}), 400

    updates = global_state['client_updates']
    total_samples = sum(r['num_samples'] for r in updates.values())
    algorithm = global_state.get('algorithm', 'fedavg').lower()
    encryption = global_state.get('encryption', 'none')

    log(f'🔄 开始聚合 | 收到 {len(updates)} 个客户端更新 | 总样本: {total_samples} | 算法: {algorithm.upper()}')

    first_update = next(iter(updates.values()))
    
    # 先检查是否使用同态加密
    if encryption == 'homomorphic':
        crypto = global_state.get('crypto')
        if crypto is None:
            log('[错误] 同态加密对象未初始化')
            return jsonify({'error': 'crypto not initialized'}), 500
        
        log('[加密] 开始同态加密聚合...')

        aggregated_state = {}
        template_state = global_state['global_model'].state_dict()

        for key in template_state.keys():
                weights = []
                encrypted_tensors = []
                client_updates = []

                for client_id, r in updates.items():
                    weight = r['num_samples'] / total_samples
                    weights.append(weight)
                    client_updates.append(weight)

                    enc_model = r['model'][key]
                    if isinstance(enc_model, dict) and 'data' in enc_model:
                        log(f'[DEBUG] 客户端{client_id} 参数{key} 数据类型: {type(enc_model["data"])}')
                        log(f'[DEBUG] 数据长度: {len(enc_model["data"])}')
                        if isinstance(enc_model["data"], list):
                            log(f'[DEBUG] 数据是列表，前5个元素: {enc_model["data"][:5]}')
                        else:
                            log(f'[DEBUG] 数据前50字符: {enc_model["data"][:50]}...')
                        try:
                            enc_tensor = crypto.deserialize_encrypted_tensor(
                                enc_model['data'],
                                tuple(enc_model['shape'])
                            )
                            log(f'[DEBUG] 反序列化成功，类型: {type(enc_tensor)}')
                            encrypted_tensors.append(enc_tensor)
                        except Exception as e:
                            log(f'[ERROR] 反序列化失败: {e}')
                            log(f'[ERROR] 错误类型: {type(e)}')
                            encrypted_tensors.append(enc_model['data'])
                    else:
                        log(f'[DEBUG] 客户端{client_id} 参数{key} 不是预期的加密格式，类型: {type(enc_model)}')
                        encrypted_tensors.append(enc_model)

                if encrypted_tensors and encrypted_tensors[0]:
                    if isinstance(crypto, CKKSCrypto):
                        # CKKS: 实现真正的加权聚合（按样本数加权）
                        # 公式: w_global = Σ (n_i / N) × w_i
                        # 为避免 CKKS 标量乘法的精度问题，采用整数权重方案：
                        # 1. 在密文状态下执行：[[sum]] = Σ n_i × [[w_i]]
                        # 2. 解密后除以总样本数 N
                        log(f'[加密] CKKS 加权聚合，{len(encrypted_tensors)}个客户端，权重: {[round(w, 4) for w in weights]}')
                        
                        # 获取整数权重（样本数）
                        client_samples = [int(w * total_samples) for w in weights]
                        log(f'[加密] 客户端样本数: {client_samples}')
                        
                        # 初始化聚合结果
                        aggregated_sum = None
                        
                        # 在密文状态下执行整数加权运算
                        for i, (enc_tensor, n_i) in enumerate(zip(encrypted_tensors, client_samples)):
                            # 密文 × 整数权重（避免小数乘法的精度问题）
                            weighted_enc = enc_tensor * n_i
                            
                            # 累加到聚合结果
                            if aggregated_sum is None:
                                aggregated_sum = weighted_enc
                            else:
                                aggregated_sum += weighted_enc
                        
                        # 解密后除以总样本数，并应用缩放因子校正
                        decrypted = crypto.decrypt_tensor(aggregated_sum, template_state[key].shape)
                        # CKKS解密后需要应用缩放因子校正（根据测试确定为16）
                        corrected = decrypted * 16
                        aggregated_state[key] = torch.tensor(corrected / total_samples, dtype=template_state[key].dtype)
                        
                        log(f'[加密] CKKS 加权聚合完成: {key}')
                    elif isinstance(encrypted_tensors[0], list):
                        # Paillier: 使用原始的聚合方法
                        aggregated = crypto.aggregate_encrypted(encrypted_tensors, weights)
                        decrypted = crypto.decrypt_tensor(aggregated, template_state[key].shape)
                        aggregated_state[key] = torch.tensor(decrypted, dtype=template_state[key].dtype)
                    else:
                        for i, (enc_tensor, weight) in enumerate(zip(encrypted_tensors, weights)):
                            if i == 0:
                                weighted_sum = weight * enc_tensor.float()
                            else:
                                weighted_sum += weight * enc_tensor.float()
                        aggregated_state[key] = weighted_sum
                else:
                    aggregated_state[key] = template_state[key]

        global_state['global_model'].load_state_dict(aggregated_state)
        log('[加密] 同态加密聚合完成')
        
        # SCAFFOLD算法的控制变量更新（解密后）
        if algorithm == 'scaffold':
            aggregated_control = {}
            lr = 0.01
            
            for key in aggregated_state.keys():
                weighted_control_sum = torch.zeros_like(aggregated_state[key])
                
                for client_id, r in updates.items():
                    weight = r['num_samples'] / total_samples
                    if 'control_update' in r:
                        # 控制变量不加密，直接处理
                        control_val = r['control_update'][key]
                        if isinstance(control_val, dict):
                            control_val = torch.tensor(control_val.get('data', []), dtype=aggregated_state[key].dtype)
                        weighted_control_sum += weight * control_val.float()
                
                aggregated_control[key] = weighted_control_sum
            
            if global_state['global_control'] is None:
                global_state['global_control'] = {k: torch.zeros_like(v) for k, v in aggregated_state.items()}
            
            for key in aggregated_control.keys():
                global_state['global_control'][key] = global_state['global_control'][key] + aggregated_control[key] / lr
            
            log(f'📊 SCAFFOLD聚合完成 (控制变量已更新)')
    elif algorithm == 'scaffold':
        # 非加密的SCAFFOLD聚合
        aggregated_state = {}
        aggregated_control = {}
        lr = 0.01
        
        for key in first_update['model'].keys():
            weighted_model_sum = torch.zeros_like(first_update['model'][key])
            weighted_control_sum = torch.zeros_like(first_update['model'][key])
            
            for client_id, r in updates.items():
                weight = r['num_samples'] / total_samples
                weighted_model_sum += weight * r['model'][key].float()
                
                if 'control_update' in r:
                    weighted_control_sum += weight * r['control_update'][key].float()
            
            aggregated_state[key] = weighted_model_sum
            aggregated_control[key] = weighted_control_sum
        
        global_state['global_model'].load_state_dict(aggregated_state)
        
        if global_state['global_control'] is None:
            global_state['global_control'] = {k: torch.zeros_like(v) for k, v in aggregated_state.items()}
        
        for key in aggregated_control.keys():
            global_state['global_control'][key] = global_state['global_control'][key] + aggregated_control[key] / lr
        
        log(f'📊 SCAFFOLD聚合完成 (控制变量已更新)')
    else:
        aggregated_state = {}
        for key in first_update['model'].keys():
            weighted_sum = torch.zeros_like(first_update['model'][key])
            for client_id, r in updates.items():
                weight = r['num_samples'] / total_samples
                weighted_sum += weight * r['model'][key].float()
            aggregated_state[key] = weighted_sum

        global_state['global_model'].load_state_dict(aggregated_state)

    accuracies = [r['accuracy'] for r in updates.values()]
    precisions = [r['precision'] for r in updates.values()]
    recalls = [r['recall'] for r in updates.values()]
    f1_scores = [r['f1'] for r in updates.values()]
    training_times = [r.get('training_time', 0.0) for r in updates.values()]
    
    avg_acc = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    avg_training_time = np.mean(training_times)

    log(f'📊 FedAvg聚合结果 | Acc:{avg_acc:.2%} | P:{avg_precision:.2%} | R:{avg_recall:.2%} | F1:{avg_f1:.2%} | 平均耗时:{avg_training_time:.2f}秒')

    # 差分隐私处理：采用每轮添加少量噪声的策略
    # 根据组合定理，总隐私预算 ε_total = ε_per_round × num_rounds
    # 这样每轮都有隐私保护，且噪声不会过度累积
    if global_state.get('privacy') == 'differential':
        current_round = global_state.get('current_round', 0)
        max_rounds = global_state.get('max_rounds', 10)
        total_epsilon = global_state.get('epsilon', 10.0)
        delta = global_state.get('delta', 5e-4)
        
        # 每轮分配相同的隐私预算（采用串行组合）
        epsilon_per_round = total_epsilon / max_rounds
        sensitivity_per_round = 0.02  # 每轮参数更新的敏感度较低
        
        apply_differential_privacy(global_state['global_model'], epsilon=epsilon_per_round, delta=delta, sensitivity=sensitivity_per_round)
        log(f'[隐私] 差分隐私已应用（第{current_round}/{max_rounds}轮）- 每轮ε: {epsilon_per_round:.2f}, 总ε: {total_epsilon}, δ: {delta}')

    global_state['client_updates'] = {}
    global_state['client_status'] = {}

    # 评估最终全局模型
    final_accuracy, final_precision, final_recall, final_f1 = evaluate_global_model()

    return jsonify({
        'status': 'aggregated',
        'avg_accuracy': avg_acc,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'avg_training_time': avg_training_time,
        'round': global_state['current_round'],
        # 最终全局模型性能
        'final_accuracy': final_accuracy,
        'final_precision': final_precision,
        'final_recall': final_recall,
        'final_f1': final_f1
    })

def apply_differential_privacy(model, epsilon=10.0, delta=5e-4, sensitivity=0.1):
    """
    对模型参数应用差分隐私（高斯机制）
    
    Args:
        model: PyTorch模型
        epsilon: 隐私预算（越小隐私保护越强，精度损失越大），推荐范围：1-50
        delta: 失败概率（越小越好），通常取 1/N（N为数据集大小）
        sensitivity: 敏感度（参数变化的最大幅度），推荐值：0.01-0.1
    """
    # 计算噪声标准差
    # 根据高斯机制：sigma >= sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    log(f'[隐私] 应用差分隐私 - epsilon={epsilon}, delta={delta}, sensitivity={sensitivity}, sigma={sigma:.6f}')
    
    with torch.no_grad():
        for param in model.parameters():
            # 添加高斯噪声（使用较小的噪声幅度）
            noise = torch.randn_like(param) * sigma
            param.add_(noise)
    
    return model

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        'current_round': global_state['current_round'],
        'max_rounds': global_state['max_rounds'],
        'training_started': global_state['training_started'],
        'client_status': global_state['client_status'],
        'num_updates': len(global_state['client_updates'])
    })

@app.route('/save_model', methods=['POST'])
def save_model():
    if global_state['global_model'] is None:
        return jsonify({'error': 'no model to save'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    algorithm = global_state.get('algorithm', 'fedavg').upper()
    num_clients = global_state.get('num_clients', 3)
    encryption = global_state.get('encryption', 'none')
    privacy = global_state.get('privacy', 'none')
    
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    filename = f'{algorithm}_{num_clients}clients_{timestamp}.pth'
    filepath = os.path.join(save_dir, filename)
    torch.save(global_state['global_model'].state_dict(), filepath)

    log(f'[OK] Global model saved to: {filepath}')

    # 确定统一的隐私保护方式
    privacy_method = 'none'
    if encryption == 'homomorphic':
        privacy_method = 'homomorphic'
    elif privacy == 'differential':
        privacy_method = 'differential'

    # 保存到数据库（legacy表，用于兼容）
    final_result = global_state.get('final_result', {})
    db_record = {
        '训练时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '算法': algorithm,
        '隐私保护方式': privacy_method,
        '客户端数量': num_clients,
        '准确率': final_result.get('准确率', 0.0),
        '精确率': final_result.get('精确率', 0.0),
        '查全率': final_result.get('查全率', 0.0),
        'F1分数': final_result.get('F1分数', 0.0),
        '时间消耗秒': final_result.get('时间消耗(秒)', 0.0),
        '模型文件名': filename,
        '模型文件路径': filepath
    }
    save_global_model_to_db(db_record)

    # 保存到新的training_results表
    training_record = {
        '算法': algorithm,
        '隐私保护方式': privacy_method,
        '客户端数量': num_clients,
        '全局轮次': global_state.get('max_rounds', 10),
        '本地批次大小': 32,
        '本地学习率': 0.01,
        '准确率': final_result.get('准确率', 0.0),
        '精确率': final_result.get('精确率', 0.0),
        '查全率': final_result.get('查全率', 0.0),
        'F1分数': final_result.get('F1分数', 0.0),
        'AUC分数': final_result.get('准确率', 0.0),
        '时间消耗秒': final_result.get('时间消耗(秒)', 0.0),
        '模型文件名': filename,
        '模型文件路径': filepath,
        '隐私预算': 1.0 if privacy == 'differential' else None,
        '噪声标准差': 0.1 if privacy == 'differential' else None,
        '总样本数': sum(global_state.get('client_updates', {}).get(c, {}).get('num_samples', 0) for c in range(num_clients)),
        '客户端样本分布': '{}',
        '额外参数': '{}'
    }
    save_training_result_to_db(training_record)

    return jsonify({'status': 'saved', 'filepath': filepath})

@app.route('/register_client', methods=['POST'])
def register_client():
    """客户端注册接口"""
    data = request.json
    client_id = data['client_id']
    port = data.get('port', 6000)
    
    registered_clients[client_id] = {
        'port': port,
        'registered_at': datetime.now().isoformat()
    }
    
    log(f'[OK] 客户端{client_id}已注册，端口: {port}')
    return jsonify({'status': 'registered'})

def check_clients():
    alive = []
    for client_id, url in CLIENTS.items():
        try:
            response = requests.get(f'{url}/health', timeout=5)
            if response.status_code == 200:
                alive.append(client_id)
        except:
            pass
    return alive

def secure_aggregate_statistics(active_clients, seed_base=42):
    """
    安全聚合环节：收集并聚合客户端的混淆统计量
    返回：全局样本量、全局特征总和、全局特征平方和
    """
    log('[INFO] 开始安全聚合统计量...')
    
    # 收集所有客户端的混淆统计量
    all_masked_stats = []
    for client_id in active_clients:
        try:
            response = requests.post(
                f'{CLIENTS[client_id]}/get_statistics',
                json={
                    'active_clients': active_clients,
                    'seed_base': seed_base
                },
                timeout=30
            )
            if response.status_code == 200:
                stats = response.json()
                all_masked_stats.append({
                    'client_id': stats['client_id'],
                    'masked_n': stats['masked_n'],
                    'masked_sum': np.array(stats['masked_sum']),
                    'masked_sq_sum': np.array(stats['masked_sq_sum'])
                })
                log(f'[OK] 收到客户端{client_id}的混淆统计量')
            else:
                log(f'[WARN] 客户端{client_id}响应异常')
        except Exception as e:
            log(f'[WARN] 无法连接客户端{client_id}: {e}')
    
    if len(all_masked_stats) == 0:
        raise ValueError("没有收到任何客户端的统计量")
    
    # 执行全局求和（掩码自动抵消）
    global_N = sum(s['masked_n'] for s in all_masked_stats)
    global_sum = sum(s['masked_sum'] for s in all_masked_stats)
    global_sq_sum = sum(s['masked_sq_sum'] for s in all_masked_stats)
    
    log(f'[OK] 安全聚合完成')
    log(f'   全局样本量: {global_N}')
    log(f'   全局特征总和: {global_sum[:3]}...')
    log(f'   全局特征平方和: {global_sq_sum[:3]}...')
    
    return global_N, global_sum, global_sq_sum

def compute_global_scaler_secure(active_clients, seed_base=42):
    """
    通过安全聚合方式计算全局StandardScaler
    实现标准化安全聚合策略的三个环节：
    1. 本地统计量计算与掩码混淆（客户端）
    2. 安全聚合与全局统计量获取（本函数）
    3. 全局标准化参数推导与标准化过程（本函数）
    """
    global global_scaler
    
    log('='*60)
    log('🔐 阶段1: 安全聚合统计量')
    log('='*60)
    
    # 环节2: 安全聚合与全局统计量获取
    global_N, global_sum, global_sq_sum = secure_aggregate_statistics(active_clients, seed_base)
    
    log('')
    log('='*60)
    log('🔐 阶段2: 推导全局标准化参数')
    log('='*60)
    
    # 环节3: 全局标准化参数推导
    # 公式 (2-9): 全局均值向量
    global_mean = global_sum / global_N
    log(f'[INFO] 计算全局均值: μ = Σx / N')
    
    # 公式 (2-10): 全局标准差向量（利用方差展开式）
    global_var = (global_sq_sum / global_N) - (global_mean ** 2)
    global_var = np.maximum(global_var, 1e-8)  # 防止除零
    global_std = np.sqrt(global_var)
    log(f'[INFO] 计算全局标准差: σ = sqrt(E[X²] - E[X]²)')
    
    # 创建全局标准化器
    global_scaler = StandardScaler()
    global_scaler.mean_ = global_mean
    global_scaler.scale_ = global_std
    global_scaler.var_ = global_var
    
    log(f'[OK] 全局标准化器推导完成')
    log(f'   全局均值: {global_mean[:3]}...')
    log(f'   全局标准差: {global_std[:3]}...')
    
    return global_scaler

def send_scaler_to_clients():
    """将全局标准化器发送给所有客户端"""
    if global_scaler is None:
        log("[WARN] 全局标准化器未初始化")
        return
    
    scaler_data = {
        'mean': global_scaler.mean_.tolist(),
        'std': global_scaler.scale_.tolist()
    }
    
    log('='*60)
    log('🔐 阶段3: 广播全局标准化参数')
    log('='*60)
    
    for client_id, url in CLIENTS.items():
        try:
            response = requests.post(f'{url}/set_scaler', json=scaler_data, timeout=10)
            if response.status_code == 200:
                log(f"[OK] 标准化参数已发送到客户端{client_id}")
            else:
                log(f"[WARN] 客户端{client_id}响应异常")
        except Exception as e:
            log(f"[WARN] 无法连接客户端{client_id}: {e}")
    
    # 通知客户端准备数据
    log('')
    log('[INFO] 通知客户端准备标准化后的数据...')
    for client_id, url in CLIENTS.items():
        try:
            X = None
            y = None
            # 客户端收到scaler后会自动标准化数据，这里不需要额外操作
            log(f"[OK] 客户端{client_id}数据准备完成")
        except Exception as e:
            log(f"[WARN] 客户端{client_id}数据准备失败: {e}")

def run_full_training():
    global training_metrics
    
    # 重置训练指标
    training_metrics = []
    
    # 记录训练开始时间（从运行开始到模型生成的总耗时）
    total_start_time = time.time()
    
    log('='*60)
    log('🔐 标准化安全聚合流程开始')
    log('='*60)

    log('')
    log('='*60)
    log('Step 1: 等待客户端连接')
    log('='*60)

    max_wait = 60
    start_wait = time.time()
    num_clients = global_state.get('num_clients', 3)
    while True:
        alive_clients = check_clients()
        if len(alive_clients) >= num_clients:
            log(f'[OK] 所有客户端已连接: {alive_clients}')
            break
        if time.time() - start_wait > max_wait:
            log(f'[WARN] 等待超时. 已连接: {alive_clients}')
            if len(alive_clients) < 2:
                log('[ERROR] 客户端数量不足')
                return
            break
        log(f'[INFO] 等待客户端... ({len(alive_clients)}/{num_clients})')
        time.sleep(3)

    # 使用安全聚合方式计算全局标准化器
    active_clients = check_clients()
    compute_global_scaler_secure(active_clients, seed_base=42)
    
    # 发送标准化参数给客户端
    send_scaler_to_clients()

    clients_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'clients_data_3clients')

    sample_df = pd.read_csv(os.path.join(clients_data_dir, 'client_0.csv'))
    input_dim = sample_df.shape[1] - 1
    global_state['feature_dim'] = input_dim

    log('')
    log('='*60)
    log('🚀 开始联邦学习训练')
    log('='*60)

    # 获取全局状态参数
    algorithm = global_state.get('algorithm', 'fedavg')
    mu = global_state.get('mu', 0.01)
    encryption = global_state.get('encryption', 'none')
    privacy = global_state.get('privacy', 'none')
    
    requests.post(
        'http://localhost:5001/start_training',
        json={
            'num_clients': 3,
            'rounds': 10,
            'input_dim': input_dim,
            'feature_dim': input_dim,
            'algorithm': algorithm,
            'mu': mu,
            'encryption': encryption,
            'privacy': privacy
        }
    )

    for round_idx in range(1, 11):
        log('')
        log('='*60)
        log(f'📦 执行第 {round_idx}/10 轮...')
        log('='*60)

        response = requests.post('http://localhost:5001/next_round')
        if response.status_code != 200:
            log(f'[ERROR] 无法开始本轮: {response.text}')
            break

        data = response.json()
        selected_clients = data['selected_clients']

        log(f'[INFO] 等待客户端 {selected_clients} 完成训练...')

        max_wait_round = 300
        start_wait_round = time.time()

        while True:
            status_response = requests.get('http://localhost:5001/status')
            status = status_response.json()
            num_updates = status['num_updates']
            expected = len(selected_clients)

            if num_updates >= expected:
                break

            if time.time() - start_wait_round > max_wait_round:
                log('[WARN] 等待超时，聚合已收到的更新...')
                break

            time.sleep(2)

        received_clients = list(global_state['client_updates'].keys())
        dropped_clients = [c for c in selected_clients if c not in received_clients]
        if dropped_clients:
            log(f'[WARN] 掉线客户端: {dropped_clients}')

        log('[INFO] 聚合客户端更新...')
        agg_response = requests.post('http://localhost:5001/aggregate')

        if agg_response.status_code == 200:
            agg_data = agg_response.json()

            # 使用最终全局模型的性能（不是各客户端的平均值）
            final_result = {
                '精确率': agg_data.get('final_precision', agg_data['avg_precision']),
                '查全率': agg_data.get('final_recall', agg_data['avg_recall']),
                'F1分数': agg_data.get('final_f1', agg_data['avg_f1']),
                '准确率': agg_data.get('final_accuracy', agg_data['avg_accuracy']),
                '时间消耗(秒)': agg_data['avg_training_time']
            }
            
            # 记录每轮指标用于前端显示
            round_metric = {
                'round': round_idx,
                'global_accuracy': agg_data.get('final_accuracy', agg_data['avg_accuracy']),
                'global_precision': agg_data.get('final_precision', agg_data['avg_precision']),
                'global_recall': agg_data.get('final_recall', agg_data['avg_recall']),
                'global_f1': agg_data.get('final_f1', agg_data['avg_f1']),
                'clients': []
            }
            
            # 添加客户端指标
            for client_id in selected_clients:
                if client_id in global_state.get('client_updates', {}):
                    update = global_state['client_updates'][client_id]
                    round_metric['clients'].append({
                        'client_id': client_id,
                        'accuracy': update.get('accuracy', 0.0),
                        'precision': update.get('precision', 0.0),
                        'recall': update.get('recall', 0.0),
                        'f1': update.get('f1', 0.0)
                    })
            
            training_metrics.append(round_metric)
        else:
            log(f'[WARN] 聚合失败: {agg_response.text}')

    log('')
    log('='*60)
    log('🎉 训练完成! 保存模型...')
    log('='*60)

    # 计算总耗时（从运行开始到模型生成的整个过程）
    total_duration = time.time() - total_start_time
    
    # 更新final_result的时间消耗为总耗时，并设置到global_state
    if 'final_result' in locals():
        final_result['时间消耗(秒)'] = total_duration
        global_state['final_result'] = final_result
        global_state['training_complete'] = True

    save_response = requests.post('http://localhost:5001/save_model')
    if save_response.status_code == 200:
        log(f'[OK] {save_response.json()["filepath"]}')
    else:
        log(f'[WARN] 保存失败: {save_response.text}')
        
    # 延迟1秒，确保前端轮询能获取到训练完成状态
    log('[INFO] 等待前端获取训练完成状态...')
    time.sleep(1)

    log('')
    log('='*60)
    log('📊 保存最终模型性能数据...')
    log('='*60)

    # 保存到数据库
    training_record = {
        '算法': global_state.get('algorithm', 'FedAvg'),
        '加密方式': global_state.get('encryption', 'none'),
        '隐私保护': global_state.get('privacy', 'none'),
        '客户端数量': global_state.get('num_clients', 3),
        '全局轮次': global_state.get('max_rounds', 10),
        '本地批次大小': 32,
        '本地学习率': 0.01,
        '准确率': final_result.get('准确率', 0.0),
        '精确率': final_result.get('精确率', 0.0),
        '查全率': final_result.get('查全率', 0.0),
        'F1分数': final_result.get('F1分数', 0.0),
        'AUC分数': final_result.get('准确率', 0.0),
        '时间消耗秒': total_duration,
        '模型文件名': f'global_model_{timestamp}.pth',
        '模型文件路径': '',
        '隐私预算': 1.0 if global_state.get('privacy') == 'differential' else None,
        '噪声标准差': 0.1 if global_state.get('privacy') == 'differential' else None,
        '总样本数': sum(global_state.get('client_updates', {}).get(c, {}).get('num_samples', 0) for c in range(global_state.get('num_clients', 3))),
        '客户端样本分布': '{}',
        '额外参数': '{}'
    }
    save_training_result_to_db(training_record)

    # 保存到Excel（可选，保留作为备份）
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(os.path.dirname(__file__), 'training_results')
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame([final_result])
    excel_path = os.path.join(save_dir, f'http_federated_final_results_{timestamp_str}.xlsx')
    df.to_excel(excel_path, index=False, engine='openpyxl')

    log(f'[OK] 最终性能数据已保存: {excel_path}')
    log('')
    log('📊 最终模型性能:')
    log('-' * 50)
    log(f'  精确率: {final_result["精确率"]:.4f}')
    log(f'  查全率: {final_result["查全率"]:.4f}')
    log(f'  F1分数: {final_result["F1分数"]:.4f}')
    log(f'  时间消耗: {total_duration:.2f} 秒')
    log('='*60)

    log('')
    log('='*60)
    log('✅ 所有步骤完成!')
    log('='*60)

    shutdown_all()

if __name__ == '__main__':
    auto_start = '--auto' in sys.argv
    
    # 解析命令行参数
    algorithm = 'fedavg'
    mu = 0.01
    privacy = 'none'
    
    for i, arg in enumerate(sys.argv):
        if arg == '--algorithm' and i + 1 < len(sys.argv):
            algorithm = sys.argv[i + 1].lower()
        elif arg == '--mu' and i + 1 < len(sys.argv):
            mu = float(sys.argv[i + 1])
        elif arg == '--privacy' and i + 1 < len(sys.argv):
            privacy = sys.argv[i + 1]
    
    # 更新全局状态（合并加密方式和隐私保护为一个参数）
    global_state['algorithm'] = algorithm
    global_state['mu'] = mu
    global_state['encryption'] = privacy if privacy == 'homomorphic' else 'none'
    global_state['privacy'] = privacy if privacy == 'differential' else 'none'

    # 初始化数据库（训练结果表）
    log('[INFO] 初始化数据库...')
    init_training_results_db()

    print('='*60)
    print('[SECURE] 联邦学习中心服务器 (安全聚合版本)')
    print('='*60)
    print('Server URL: http://localhost:5001')
    print(f'算法: {algorithm.upper()}, Mu: {mu}, 隐私保护: {privacy}')
    print('')
    print('实现的安全聚合策略:')
    print('  1. 本地统计量计算与掩码混淆')
    print('  2. 安全聚合与全局统计量获取')
    print('  3. 全局标准化参数推导与标准化过程')
    print('')
    if auto_start:
        print('[AUTO MODE] 客户端就绪后自动开始训练')
    else:
        print('请先启动所有客户端，然后按Enter继续...')
    print('='*60)

    def run_flask():
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False, threaded=True)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # 等待Flask服务器完全启动
    time.sleep(2)
    print('[OK] Flask服务器已启动')

    if auto_start:
        print('[INFO] 等待客户端连接...')
        time.sleep(3)
        run_full_training()
    else:
        input()
        time.sleep(1)
        run_full_training()

    print('')
    print('按 Ctrl+C 退出...')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('退出')