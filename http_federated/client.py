"""
联邦学习客户端 - HTTP API
实现标准化安全聚合策略
"""
import os
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import requests
import signal
import time
import threading
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_utils import HomomorphicCrypto, CKKSCrypto

if len(sys.argv) < 3:
    print('Usage: python client.py <client_id> <port>')
    sys.exit(1)

CLIENT_ID = int(sys.argv[1])
PORT = int(sys.argv[2])
SERVER_URL = 'http://localhost:5001'

app = Flask(__name__)

# 全局标准化器（由服务器下发）
global_scaler = None

# 本地原始数据（用于计算统计量）
local_raw_data = None

client_state = {
    'client_id': CLIENT_ID,
    'local_model': None,
    'train_data': None,
    'test_data': None,
    'input_dim': 8,
    'local_epochs': 5,
    'control_variable': None,  # SCAFFOLD控制变量
    'crypto': None,            # 同态加密对象
    'encryption': 'none'       # 加密方式
}

# SCAFFOLD学习率
SCAFFOLD_LR = 0.01

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
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [CLIENT-{CLIENT_ID}] {msg}')

def generate_pairwise_mask(client_id, other_id, feature_dim, seed_base=42):
    """
    生成客户端之间的成对掩码
    满足: mask(i,j) = -mask(j,i)
    """
    # 确保一致的种子生成
    if client_id < other_id:
        seed = seed_base + client_id * 1000 + other_id
    else:
        seed = seed_base + other_id * 1000 + client_id
    
    rng = np.random.RandomState(seed)
    mask = rng.randn(feature_dim).astype(np.float32)
    
    # 根据编号大小决定符号
    if client_id < other_id:
        return mask
    else:
        return -mask

def compute_local_statistics(X):
    """
    计算本地统计量
    返回: (样本量, 特征总和向量, 特征平方和向量)
    """
    n_i = X.shape[0]
    sum_i = X.sum(axis=0)
    sq_sum_i = (X ** 2).sum(axis=0)
    return n_i, sum_i, sq_sum_i

def mask_statistics(n_i, sum_i, sq_sum_i, active_clients, feature_dim, seed_base=42):
    """
    使用成对掩码混淆本地统计量
    对称掩码原理：
    - 客户端i和j生成掩码：mask(i,j) = -mask(j,i)
    - 客户端i发送: sum_i + Σ(mask(i,j)) for j<i - Σ(mask(i,j)) for j>i
    - 服务器汇总时，掩码自动两两抵消
    """
    masked_n = n_i
    masked_sum = sum_i.copy()
    masked_sq_sum = sq_sum_i.copy()

    for u in active_clients:
        if u != CLIENT_ID:
            # 生成与客户端u之间的掩码
            mask = generate_pairwise_mask(CLIENT_ID, u, feature_dim, seed_base)
            if u < CLIENT_ID:
                # u < i: 减去掩码（因为 generate_pairwise_mask 返回的是 -r_ui）
                masked_sum -= mask
                masked_sq_sum -= mask
            else:
                # u > i: 加上掩码（因为 generate_pairwise_mask 返回的是 r_ij）
                masked_sum += mask
                masked_sq_sum += mask

    return masked_n, masked_sum, masked_sq_sum

def load_raw_data():
    """加载原始数据（未标准化）"""
    global local_raw_data
    
    clients_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'clients_data_3clients'
    )

    csv_path = os.path.join(clients_data_dir, f'client_{CLIENT_ID}.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Data file not found: {csv_path}')

    log(f'[INFO] Loading raw data: {csv_path}')

    df = pd.read_csv(csv_path)
    X = df.drop('诊断结果', axis=1).values
    y = df['诊断结果'].values
    
    local_raw_data = {
        'X': X,
        'y': y,
        'feature_dim': X.shape[1]
    }
    
    log(f'[OK] Raw data loaded. Samples: {len(X)}, Features: {X.shape[1]}')
    return X, y

def prepare_train_test_data(X, y):
    """准备训练和测试数据（使用全局标准化器）"""
    global global_scaler
    
    # 使用全局标准化器进行标准化
    if global_scaler is not None:
        X_normalized = global_scaler.transform(X)
        log(f'[INFO] Data standardized using global scaler')
    else:
        X_normalized = X
        log(f'[WARN] No global scaler available, using raw data')

    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    num_samples = len(X)
    split_idx = int(0.8 * num_samples)

    X_train = X_tensor[:split_idx]
    y_train = y_tensor[:split_idx]
    X_test = X_tensor[split_idx:]
    y_test = y_tensor[split_idx:]

    client_state['input_dim'] = X.shape[1]
    log(f'[OK] Data prepared. Train: {len(X_train)}, Test: {len(X_test)}')
    return (X_train, y_train), (X_test, y_test)

@app.route('/get_statistics', methods=['POST'])
def get_statistics():
    """
    本地统计量计算与掩码混淆环节
    接收服务器的请求，计算并返回混淆后的本地统计量
    """
    global local_raw_data
    
    if local_raw_data is None:
        return jsonify({'error': 'data not loaded'}), 400
    
    data = request.json
    active_clients = data.get('active_clients', [0, 1, 2, 3])
    seed_base = data.get('seed_base', 42)
    
    X = local_raw_data['X']
    feature_dim = local_raw_data['feature_dim']
    
    # 步骤1: 计算本地统计量
    n_i, sum_i, sq_sum_i = compute_local_statistics(X)
    log(f'[INFO] Local statistics computed: n={n_i}')
    
    # 步骤2: 使用成对掩码混淆统计量
    masked_n, masked_sum, masked_sq_sum = mask_statistics(
        n_i, sum_i, sq_sum_i, active_clients, feature_dim, seed_base
    )
    log(f'[INFO] Statistics masked with pairwise masks')
    
    return jsonify({
        'client_id': CLIENT_ID,
        'masked_n': int(masked_n),
        'masked_sum': masked_sum.tolist(),
        'masked_sq_sum': masked_sq_sum.tolist()
    })

@app.route('/set_scaler', methods=['POST'])
def set_scaler():
    """接收服务器下发的全局标准化参数并准备数据"""
    global global_scaler
    data = request.json
    scaler_mean = np.array(data['mean'])
    scaler_std = np.array(data['std'])
    
    global_scaler = StandardScaler()
    global_scaler.mean_ = scaler_mean
    global_scaler.scale_ = scaler_std
    global_scaler.var_ = scaler_std ** 2
    
    log(f'[OK] Global scaler received: mean={scaler_mean[:3]}..., std={scaler_std[:3]}...')
    
    # 使用全局标准化器准备训练和测试数据
    if local_raw_data is not None:
        X, y = local_raw_data['X'], local_raw_data['y']
        client_state['train_data'], client_state['test_data'] = prepare_train_test_data(X, y)
    
    return jsonify({'status': 'scaler_set'})

def local_train(model, X_train, y_train, X_test, y_test, epochs=5, proximal_model=None, mu=0.01, global_control=None):
    """
    本地训练函数（支持FedAvg/FedProx/SCAFFOLD）
    :param model: 当前模型
    :param X_train, y_train: 训练数据
    :param X_test, y_test: 测试数据
    :param epochs: 训练轮数
    :param proximal_model: FedProx参考模型（全局模型参数）
    :param mu: FedProx近端系数（默认0.01）
    :param global_control: SCAFFOLD全局控制变量
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    start_time = time.time()

    # 如果启用FedProx，保存全局模型参数作为参考
    if proximal_model is not None:
        proximal_params = {k: v.detach().clone() for k, v in proximal_model.state_dict().items()}
        log(f'[INFO] FedProx enabled with mu={mu}')
    else:
        proximal_params = None

    # SCAFFOLD: 初始化本地控制变量
    if global_control is not None:
        if client_state['control_variable'] is None:
            client_state['control_variable'] = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        global_control_vars = {k: v.detach().clone() for k, v in global_control.items()}
        log(f'[INFO] SCAFFOLD enabled with lr={SCAFFOLD_LR}')

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)

        # FedProx近端项：L(θ) = L_data(θ) + (μ/2) * ||θ - θ^t||^2
        if proximal_params is not None:
            proximal_term = 0.0
            for k, v in model.named_parameters():
                if k in proximal_params:
                    proximal_term += torch.norm(v - proximal_params[k]) ** 2
            loss += (mu / 2) * proximal_term

        loss.backward()

        # SCAFFOLD: 计算梯度后更新控制变量
        if global_control is not None:
            with torch.no_grad():
                for k, v in model.named_parameters():
                    if k in global_control_vars and k in client_state['control_variable']:
                        g = v.grad
                        c = global_control_vars[k]
                        c_i = client_state['control_variable'][k]
                        client_state['control_variable'][k] = c_i - g + c

        optimizer.step()

    training_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs.squeeze() > 0.5).float()

        accuracy = (predicted == y_test).float().mean().item()

        y_true = y_test.numpy()
        y_pred = predicted.numpy()
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1, training_time

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'role': 'client', 
        'client_id': CLIENT_ID
    })

@app.route('/start_train', methods=['POST'])
def start_train():
    data = request.json
    global_model = list_to_tensor(data['global_model'])
    round_num = data['round']
    max_rounds = data['max_rounds']
    input_dim = data.get('input_dim', client_state['input_dim'])
    
    # FedProx参数
    algorithm = data.get('algorithm', 'fedavg')
    mu = data.get('mu', 0.01)

    log(f'[INFO] Received training task: Round {round_num}/{max_rounds}, Input Dim: {input_dim}, Algorithm: {algorithm}')

    model = get_model(input_dim=input_dim, use_large=True)
    model.load_state_dict(global_model)

    X_train, y_train = client_state['train_data']
    X_test, y_test = client_state['test_data']

    try:
        key_response = requests.get(f'{SERVER_URL}/get_public_key', timeout=5)
        if key_response.status_code == 200:
            key_data = key_response.json()
            if key_data.get('has_public_key'):
                encryption_type = key_data.get('encryption_type', 'paillier')
                log(f'[加密] 收到服务器公钥 (类型: {encryption_type})，开始初始化...')

                if encryption_type == 'ckks':
                    client_state['crypto'] = CKKSCrypto()
                    client_state['crypto'].load_public_key_from_serialized(key_data['public_key'])
                    log('[加密] CKKS同态加密已初始化')
                else:
                    client_state['crypto'] = HomomorphicCrypto()
                    client_state['crypto'].load_public_key_from_dict(key_data['public_key'])
                    log('[加密] Paillier同态加密已初始化')

                client_state['encryption'] = 'homomorphic'
            else:
                client_state['crypto'] = None
                client_state['encryption'] = 'none'
    except Exception as e:
        log(f'[加密] 获取公钥失败: {e}')
        client_state['crypto'] = None
        client_state['encryption'] = 'none'

    # FedProx：创建参考模型（全局模型副本）
    proximal_model = None
    if algorithm.lower() == 'fedprox':
        proximal_model = get_model(input_dim=input_dim, use_large=True)
        proximal_model.load_state_dict(global_model)

    log(f'[INFO] Starting local training ({client_state["local_epochs"]} epochs)...')
    
    # SCAFFOLD: 获取全局控制变量
    global_control = None
    if algorithm.lower() == 'scaffold':
        if 'global_control' in data:
            global_control = list_to_tensor(data['global_control'])
    
    accuracy, precision, recall, f1, training_time = local_train(
        model, X_train, y_train, X_test, y_test, 
        client_state['local_epochs'], 
        proximal_model=proximal_model,
        mu=mu,
        global_control=global_control
    )

    log(f'[OK] 训练完成 | Acc:{accuracy:.2%} | P:{precision:.2%} | R:{recall:.2%} | F1:{f1:.2%} | 耗时:{training_time:.2f}秒')

    log(f'[INFO] 上传更新到服务器...')
    
    # 构建上传数据
    model_state = model.state_dict()
    
    if client_state['encryption'] == 'homomorphic' and client_state['crypto'] is not None:
        log('[加密] 加密模型更新...')
        encrypted_model = {}
        for key, tensor in model_state.items():
            enc_tensor = client_state['crypto'].encrypt_tensor(tensor)
            encrypted_model[key] = {
                'data': client_state['crypto'].serialize_encrypted_tensor(enc_tensor),
                'shape': list(tensor.shape)
            }
        upload_data = {
            'client_id': CLIENT_ID,
            'model_update': encrypted_model,
            'num_samples': len(X_train),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': training_time,
            'encrypted': True
        }
    else:
        upload_data = {
            'client_id': CLIENT_ID,
            'model_update': tensor_to_list(model_state),
            'num_samples': len(X_train),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': training_time,
            'encrypted': False
        }
    
    # SCAFFOLD: 添加控制变量更新
    if algorithm.lower() == 'scaffold' and client_state['control_variable'] is not None:
        upload_data['control_update'] = tensor_to_list(client_state['control_variable'])
    
    try:
        response = requests.post(
            f'{SERVER_URL}/submit_update',
            json=upload_data,
            timeout=30
        )
        if response.status_code == 200:
            log(f'[OK] 更新已上传')
        else:
            log(f'[WARN] 上传失败: {response.text}')
    except Exception as e:
        log(f'[WARN] 上传错误: {e}')

    return jsonify({'status': 'completed', 'accuracy': accuracy, 'f1': f1, 'training_time': training_time})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    log('[INFO] 收到服务器的关闭命令')
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({'status': 'shutting_down'})

def init_secure_aggregation():
    """初始化安全聚合流程"""
    log('[INFO] 初始化安全聚合流程...')
    
    # 首先加载原始数据
    load_raw_data()
    
    # 向服务器注册并获取活跃客户端列表（带重试机制）
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(f'{SERVER_URL}/register_client', 
                                   json={'client_id': CLIENT_ID, 'port': PORT},
                                   timeout=10)
            if response.status_code == 200:
                log('[OK] 客户端注册成功')
                break
            else:
                log(f'[WARN] 注册响应异常: {response.text}')
        except Exception as e:
            log(f'[WARN] 注册失败 (尝试 {attempt + 1}/{max_retries}): {e}')
            if attempt < max_retries - 1:
                time.sleep(2)  # 等待2秒后重试
            else:
                log('[ERROR] 注册失败，但将继续运行')
    
    # 等待服务器发起统计量收集
    log('[INFO] 等待服务器收集统计量...')

if __name__ == '__main__':
    print('='*60)
    print(f'Federated Learning Client #{CLIENT_ID}')
    print('='*60)

    print(f'Client URL: http://localhost:{PORT}')
    print(f'Server URL: {SERVER_URL}')
    print('='*60)
    
    # 先启动Flask服务器，然后再注册到中心服务器
    def start_flask():
        app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
    
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    # 等待Flask服务器启动
    time.sleep(1)
    
    # Flask启动后再注册到中心服务器
    init_secure_aggregation()
    
    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f'\n[INFO] 客户端 {CLIENT_ID} 退出')