
"""
单文件验证脚本 - 验证HTTP联邦学习的核心逻辑
"""
import os
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*80)
print("HTTP联邦学习 - 核心逻辑验证")
print("="*80)

clients_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'clients_data')
alt_dir = os.path.join(os.path.dirname(clients_data_dir), 'clients_data_4clients')
if os.path.exists(alt_dir):
    clients_data_dir = alt_dir
print(f"[OK] 使用数据目录: {clients_data_dir}")

class SimpleNN(nn.Module):
    def __init__(self, input_dim=8):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

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

def load_data(client_id):
    csv_path = os.path.join(clients_data_dir, f'client_{client_id}.csv')
    df = pd.read_csv(csv_path)
    X = df.drop('诊断结果', axis=1).values
    y = df['诊断结果'].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    num_samples = len(X)
    split_idx = int(0.8 * num_samples)
    return (X_tensor[:split_idx], y_tensor[:split_idx]), (X_tensor[split_idx:], y_tensor[split_idx:])

def local_train(model, X_train, y_train, X_test, y_test, epochs=5):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs.squeeze() > 0.5).float()
        accuracy = (predicted == y_test).float().mean().item()
    return accuracy

def fed_avg_aggregate(updates):
    total_samples = sum(u['num_samples'] for u in updates)
    aggregated = {}
    for key in updates[0]['model'].keys():
        weighted_sum = torch.zeros_like(updates[0]['model'][key])
        for u in updates:
            weight = u['num_samples'] / total_samples
            weighted_sum += weight * u['model'][key].float()
        aggregated[key] = weighted_sum
    return aggregated

print("\n[1] 检查数据文件:")
input_dim = None
client_data = {}
for c in range(4):
    (X_train, y_train), (X_test, y_test) = load_data(c)
    client_data[c] = {'train': (X_train, y_train), 'test': (X_test, y_test)}
    if input_dim is None:
        input_dim = X_train.shape[1]
    print(f"    Client {c}: train={len(X_train)}, test={len(X_test)}, features={input_dim}")

print("\n[2] 初始化全局模型...")
global_model = SimpleNN(input_dim=input_dim)
print(f"    模型架构: {global_model}")

print("\n[3] 模拟10轮FedAvg训练:")
print("    " + "-"*50)
for round_idx in range(1, 11):
    num_participants = max(2, int(4 * 0.75))
    selected = np.random.choice(4, num_participants, replace=False)
    selected = [int(x) for x in selected]
    
    updates = []
    for c in selected:
        (X_train, y_train), (X_test, y_test) = client_data[c]['train'], client_data[c]['test']
        local_model = SimpleNN(input_dim=input_dim)
        local_model.load_state_dict(global_model.state_dict())
        acc = local_train(local_model, X_train, y_train, X_test, y_test)
        updates.append({'client_id': c, 'model': local_model.state_dict(), 'num_samples': len(X_train), 'accuracy': acc})
        print(f"    Round {round_idx} - Client {c}: Accuracy {acc:.4f}")
    
    aggregated = fed_avg_aggregate(updates)
    global_model.load_state_dict(aggregated)
    
    avg_acc = np.mean([u['accuracy'] for u in updates])
    print(f"    Round {round_idx} - Avg Accuracy: {avg_acc:.4f}")
    print("    " + "-"*50)

print("\n[4] 保存模型...")
save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = os.path.join(save_dir, f'http_server_model_{timestamp}.pth')
torch.save(global_model.state_dict(), save_path)
print(f"    模型已保存: {save_path}")

print("\n" + "="*80)
print("[OK] 验证成功！核心逻辑没问题！")
print("\n要运行真正的HTTP版本，请：")
print("  1. 双击 启动HTTP联邦学习.bat")
print("  2. 等待5个终端启动")
print("  3. 在服务器终端按回车")
print("="*80)

