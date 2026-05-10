import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.model_selection import train_test_split
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def save_clients_data(clients_data, output_dir='clients_data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for client_id, data in clients_data.items():
        data.to_csv(os.path.join(output_dir, f'{client_id}.csv'), index=False)
        distribution = {
            'total_samples': len(data),
            'positive_samples': int(data['诊断结果'].sum()),
            'negative_samples': len(data) - int(data['诊断结果'].sum()),
            'positive_ratio': round(float(data['诊断结果'].mean()), 4)
        }
        with open(os.path.join(output_dir, f'{client_id}_distribution.json'), 'w', encoding='utf-8') as f:
            json.dump(distribution, f, ensure_ascii=False, indent=2)

def create_non_iid_label_size(df, num_clients=3, min_samples=100):
    clients_data = {}
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total_samples = len(df)
    min_total = num_clients * min_samples

    if min_total > total_samples:
        raise ValueError(f"总样本数{total_samples}不足，无法满足每个客户端最少{min_samples}条")

    base_sizes = [min_samples] * num_clients
    remaining_samples = total_samples - min_total

    if remaining_samples > 0:
        np.random.seed(42)
        random_sizes = np.random.randint(1, remaining_samples + 1, num_clients)
        extra_sizes = [int(s * remaining_samples / sum(random_sizes)) for s in random_sizes]
        extra_sizes[-1] += remaining_samples - sum(extra_sizes)
        sizes = [base + extra for base, extra in zip(base_sizes, extra_sizes)]
    else:
        sizes = base_sizes

    start = 0
    for i in range(num_clients):
        end = start + sizes[i]
        clients_data[f'client_{i}'] = shuffled_df.iloc[start:end].copy()
        start = end
    return clients_data

def main():
    df = load_data("diabetes_data_Chinese.xlsx")

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["诊断结果"]
    )
    df_test.to_csv("global_test_set.csv", index=False, encoding="utf-8")
    print("✅ 已生成独立全局测试集：global_test_set.csv")

    num_clients = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    clients_data = create_non_iid_label_size(df_train, num_clients=num_clients, min_samples=100)
    out_dir = f"clients_data_{num_clients}clients"
    save_clients_data(clients_data, out_dir)

    print(f"✅ 客户端Non-IID数据生成完成，目录：{out_dir}")
    for k,v in clients_data.items():
        print(f"  {k}：样本数{len(v)}，阳性占比{round(v['诊断结果'].mean(),4)}")

if __name__ == "__main__":
    main()