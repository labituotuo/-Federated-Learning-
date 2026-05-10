import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 定义与http_federated相同的模型结构（使用nn.Sequential）
class Net(nn.Module):
    def __init__(self, input_dim=43):
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

# 特征名称列表（43个特征）
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

def load_model(model_path, input_dim=43):
    """加载训练好的HTTP联邦学习模型"""
    model = Net(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def load_test_data(data_dir='clients_data_3clients'):
    """加载测试数据并应用标准化"""
    from sklearn.preprocessing import StandardScaler
    
    # 首先创建全局scaler（与训练时一致）
    all_data = []
    for i in range(3):  # 3个客户端
        client_file = os.path.join(data_dir, f'client_{i}.csv')
        if os.path.exists(client_file):
            df = pd.read_csv(client_file)
            X = df.iloc[:, :-1].values
            all_data.append(X)
    
    if len(all_data) > 0:
        all_X = np.vstack(all_data)
        scaler = StandardScaler()
        scaler.fit(all_X)
        print(f"[OK] 全局StandardScaler已创建")
    else:
        raise FileNotFoundError("未找到任何客户端数据文件")
    
    # 使用client_0的数据作为测试集
    client_file = os.path.join(data_dir, 'client_0.csv')
    if os.path.exists(client_file):
        df = pd.read_csv(client_file)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values
        # 应用标准化
        X = scaler.transform(X).astype(np.float32)
        return X, y
    else:
        np.random.seed(42)
        X = np.random.randn(100, 43).astype(np.float32)
        return X, None

def generate_shap_bar_plot(model, X, feature_names, output_path):
    """生成SHAP特征重要性柱状图（前20个特征，灰度渐变）"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建预测函数
    def predict_proba(x):
        input_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            return output.numpy()
    
    # 使用SHAP KernelExplainer
    background_data = X[:50]
    explainer = shap.KernelExplainer(predict_proba, background_data)
    
    # 计算SHAP值
    eval_data = X[50:100]
    shap_values = explainer.shap_values(eval_data)
    
    # 处理多维输出
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # 计算每个特征的平均绝对SHAP值
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_abs_shap = np.ravel(mean_abs_shap)
    
    # 获取前20个重要特征
    top_indices = np.argsort(mean_abs_shap)[::-1][:20]
    top_features = [feature_names[int(i)] for i in top_indices]
    top_shap_values = mean_abs_shap[top_indices]
    
    # 创建水平柱状图（使用灰度渐变：深灰到浅灰）
    plt.figure(figsize=(14, 9))
    colors = [plt.cm.gray(0.3 + 0.6 * i / len(top_features)) for i in range(len(top_features))]
    bars = plt.barh(range(len(top_features)), top_shap_values, color=colors)
    
    # 设置标签和标题
    plt.yticks(range(len(top_features)), top_features, fontsize=12)
    plt.xlabel('平均绝对SHAP值', fontsize=14)
    plt.ylabel('特征', fontsize=14)
    plt.title('SCAFFOLD联邦学习模型 - SHAP特征重要性分析（前20位）', fontsize=18, fontweight='bold', pad=20)
    
    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.0005, bar.get_y() + bar.get_height()/2,
                 f'{width:.6f}', ha='left', va='center', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存为SVG格式
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] SHAP特征重要性图已保存到: {output_path}")
    
    # 打印详细排名
    print("\n📊 SHAP特征重要性排名（前20位）:")
    print("-" * 60)
    print(f"{'排名':<6} {'特征':<20} {'SHAP值':<15}")
    print("-" * 60)
    for i, idx in enumerate(top_indices):
        print(f"{i+1:<6} {feature_names[idx]:<20} {mean_abs_shap[idx]:<15.6f}")
    
    return top_features, top_shap_values

if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # 模型路径（指定的SCAFFOLD模型）
    model_path = 'http_federated/saved_models/SCAFFOLD_3clients_20260506_230418.pth'
    output_path = 'http_federated/saved_models/SCAFFOLD_3clients_20260506_230418_shap.svg'
    
    print("=" * 60)
    print("SCAFFOLD Federated Model SHAP Feature Analysis (Top 20)")
    print("=" * 60)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"\n[ERROR] 模型文件不存在: {model_path}")
        print("请确认文件路径是否正确")
        sys.exit(1)
    
    # 加载模型
    print("\n[INFO] 加载模型...")
    model = load_model(model_path)
    print("[OK] 模型加载成功")
    
    # 加载数据
    print("\n[INFO] 加载数据...")
    X, y = load_test_data()
    print(f"[OK] 数据加载成功 (样本数: {X.shape[0]}, 特征数: {X.shape[1]})")
    
    # 生成SHAP图
    print("\n[INFO] 计算SHAP值并生成图表...")
    generate_shap_bar_plot(model, X, feature_names, output_path)
    
    print("\n✅ 分析完成！")
