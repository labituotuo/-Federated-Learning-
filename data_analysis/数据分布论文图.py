import matplotlib.pyplot as plt
import numpy as np

# ========== 解决中文显示 ==========
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

# ========== 数据（与截图完全一致） ==========
clients = ['client_0', 'client_1', 'client_2']
positive = [288, 329, 135]    # 阳性样本
negative = [426, 480, 221]   # 阴性样本

# ========== 纯灰白配色 ==========
color_pos = "#f0f0f0"    # 浅灰白（阳性）
color_neg = "#b0b0b0"    # 中灰色（阴性）

# ========== 画图 ==========
plt.figure(figsize=(10, 6))
x = np.arange(len(clients))
width = 0.7

# 绘制堆叠柱状图
bars_pos = plt.bar(x, positive, width, color=color_pos, label="阳性样本")
bars_neg = plt.bar(x, negative, width, bottom=positive, color=color_neg, label="阴性样本")

# 给阳性样本柱子加数值标注
for bar in bars_pos:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height / 2,
        f'{int(height)}',
        ha='center', va='center', color='black', fontsize=12
    )

# 给阴性样本柱子加数值标注
for bar in bars_neg:
    height = bar.get_height()
    bottom = bar.get_y()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bottom + height / 2,
        f'{int(height)}',
        ha='center', va='center', color='white', fontsize=12
    )

# 标签与标题
plt.title("各客户端样本分布", fontsize=14, pad=15)
plt.xlabel("客户端", fontsize=10)
plt.ylabel("样本数量", fontsize=10)
plt.xticks(x, clients, fontsize=10)
plt.legend(loc="upper right", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存为 SVG 格式
plt.savefig("client_sample_distribution.svg", format="svg", bbox_inches="tight")
plt.show()