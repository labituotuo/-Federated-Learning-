import pandas as pd
import os

# 加载数据
df = pd.read_excel('diabetes_data_Chinese.xlsx')

# 连续数值特征列表（21个）
continuous_features = [
    '年龄', '体重指数', '饮酒量', '每周体育活动时间', '饮食质量', '睡眠质量',
    '收缩压', '舒张压', '空腹血糖', '糖化血红蛋白', '血清肌酐', '血尿素氮水平',
    '总胆固醇', '低密度脂蛋白胆固醇', '高密度脂蛋白胆固醇', '甘油三酯',
    '疲劳程度', '生活质量评分', '体检频率', '药物依从性', '健康素养'
]

# 提取连续特征数据
df_continuous = df[continuous_features]

# 按诊断结果分组
has_diabetes = df[df['诊断结果'] == 1][continuous_features]
no_diabetes = df[df['诊断结果'] == 0][continuous_features]

# 计算统计描述
stats_diabetes = has_diabetes.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].copy()
stats_no_diabetes = no_diabetes.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].copy()

# 重命名列
stats_diabetes.columns = ['均值', '标准差', '最小值', '25%分位数', '中位数', '75%分位数', '最大值']
stats_no_diabetes.columns = ['均值', '标准差', '最小值', '25%分位数', '中位数', '75%分位数', '最大值']

# 添加数据范围和变异系数
for stats in [stats_diabetes, stats_no_diabetes]:
    stats['取值范围'] = stats['最小值'].astype(str) + ' ~ ' + stats['最大值'].astype(str)
    stats['变异系数(CV)'] = (stats['标准差'] / stats['均值']).round(4)

# 保存为Excel，两个工作表
output_path = 'continuous_features_statistics_by_diagnosis.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    stats_diabetes.to_excel(writer, sheet_name='有糖尿病')
    stats_no_diabetes.to_excel(writer, sheet_name='无糖尿病')

print(f"✅ 已生成按糖尿病分类的连续特征统计描述表：{output_path}")
print("\n有糖尿病组 - 前5个特征：")
print(stats_diabetes.head())
print("\n无糖尿病组 - 前5个特征：")
print(stats_no_diabetes.head())
