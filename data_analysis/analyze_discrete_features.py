import pandas as pd
import numpy as np
from scipy import stats

# 加载数据
df = pd.read_excel('diabetes_data_Chinese.xlsx')

# 识别离散特征（唯一值数量 <= 10 且非连续数值）
continuous_features = [
    '年龄', '体重指数', '饮酒量', '每周体育活动时间', '饮食质量', '睡眠质量',
    '收缩压', '舒张压', '空腹血糖', '糖化血红蛋白', '血清肌酐', '血尿素氮水平',
    '总胆固醇', '低密度脂蛋白胆固醇', '高密度脂蛋白胆固醇', '甘油三酯',
    '疲劳程度', '生活质量评分', '体检频率', '药物依从性', '健康素养'
]

discrete_features = [col for col in df.columns if col not in continuous_features and col != '诊断结果']

print(f"识别到 {len(discrete_features)} 个离散特征: {discrete_features}\n")

# 按诊断结果分组
df_diabetes = df[df['诊断结果'] == 1]
df_no_diabetes = df[df['诊断结果'] == 0]

# 为每个离散特征生成频数分布和卡方检验
results = []

for feature in discrete_features:
    # 频数统计
    total_counts = df[feature].value_counts().sort_index()
    diabetes_counts = df_diabetes[feature].value_counts().sort_index()
    no_diabetes_counts = df_no_diabetes[feature].value_counts().sort_index()
    
    # 构建列联表
    contingency_table = pd.crosstab(df['诊断结果'], df[feature])
    
    # 卡方检验
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Cramer's V 效应量
    n = len(df)
    phi2 = chi2 / n
    k = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(phi2 / k) if k > 0 else 0
    
    # 计算占比
    diabetes_ratio = diabetes_counts / len(df_diabetes)
    no_diabetes_ratio = no_diabetes_counts / len(df_no_diabetes)
    
    # 众数
    mode_val = df[feature].mode()[0]
    mode_count = (df[feature] == mode_val).sum()
    
    results.append({
        '特征': feature,
        '唯一值数量': df[feature].nunique(),
        '众数': str(mode_val),
        '众数频数': mode_count,
        '卡方值': round(chi2, 4),
        'p值': f'{p_value:.2e}',
        '显著性': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else '不显著')),
        'Cramers_V': round(cramers_v, 4),
        '关联强度': '强' if cramers_v > 0.3 else ('中' if cramers_v > 0.1 else '弱'),
        '有糖尿病占比': {str(k): f'{v:.2%}' for k, v in diabetes_ratio.items()},
        '无糖尿病占比': {str(k): f'{v:.2%}' for k, v in no_diabetes_ratio.items()}
    })

# 输出结果摘要
print("="*80)
print("离散特征统计分析摘要")
print("="*80)
for r in results:
    print(f"\n【{r['特征']}】")
    print(f"  唯一值: {r['唯一值数量']} 个 | 众数: {r['众数']} (频数: {r['众数频数']})")
    print(f"  卡方检验: χ²={r['卡方值']}, p={r['p值']} {r['显著性']}")
    print(f"  关联强度: Cramer's V={r['Cramers_V']} ({r['关联强度']})")
    print(f"  有糖尿病组分布: {r['有糖尿病占比']}")
    print(f"  无糖尿病组分布: {r['无糖尿病占比']}")

# 保存为Excel
output_path = 'discrete_features_analysis.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 工作表1: 统计摘要
    summary_df = pd.DataFrame([{
        '特征': r['特征'],
        '唯一值数量': r['唯一值数量'],
        '众数': r['众数'],
        '众数频数': r['众数频数'],
        '卡方值': r['卡方值'],
        'p值': r['p值'],
        '显著性': r['显著性'],
        'Cramers_V': round(cramers_v, 4),
        '关联强度': r['关联强度']
    } for r in results])
    summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
    
    # 工作表2-N: 每个特征的详细频数分布
    for r in results:
        feature = r['特征']
        detail_data = []
        for val in sorted(df[feature].unique()):
            total = (df[feature] == val).sum()
            diabetes = (df_diabetes[feature] == val).sum()
            no_diabetes = (df_no_diabetes[feature] == val).sum()
            n_diabetes = len(df_diabetes)
            n_no_diabetes = len(df_no_diabetes)
            detail_data.append({
                '取值': val,
                '总频数': total,
                '总占比': f'{total/len(df):.2%}',
                '有糖尿病频数': diabetes,
                '有糖尿病占比': f'{diabetes/n_diabetes:.2%}',
                '无糖尿病频数': no_diabetes,
                '无糖尿病占比': f'{no_diabetes/n_no_diabetes:.2%}'
            })
        pd.DataFrame(detail_data).to_excel(writer, sheet_name=f'{feature[:20]}', index=False)

print(f"\n✅ 离散特征分析结果已保存至: {output_path}")
print("包含工作表: 统计摘要 + 各特征详细频数分布")
