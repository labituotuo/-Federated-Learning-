# -*- coding: utf-8 -*-
"""
AI 健康管理报告生成模块
基于 DeepSeek 大模型提供个性化糖尿病防控建议
"""
import os
import json
from openai import OpenAI


def generate_health_report(input_data):
    """
    使用 DeepSeek 模型生成个性化健康管理报告
    
    :param input_data: 用户健康指标数据（字典格式）
    :return: 生成的健康管理报告文本
    """
    try:
        # 构建用户健康档案
        user_profile = format_user_health_profile(input_data)
        
        # 构建系统提示词
        system_prompt = """你是智能糖尿病防控健康管理专家，擅长根据用户的各项健康指标，从饮食、运动、作息等多个生活领域提供精准、直截了当的建议。

请遵循以下原则：
1. 建议要简洁明了，分点列举，不要冗长
2. 针对用户的具体指标给出直戳要害的建议
3. 语言亲切但专业，避免医学晦涩术语
4. 如果指标异常，说明原因和改善方法
5. 给出可操作的、具体的改善方案
6. 每个领域2-3条核心建议即可

输出格式：
【饮食建议】
- xxx
- xxx

【运动建议】
- xxx
- xxx

【作息建议】
- xxx
- xxx

【用药及监测建议】
- xxx
- xxx

【其他生活方式建议】
- xxx
- xxx"""
        
        # 构建用户提示词
        user_prompt = f"""请根据以下用户健康数据，生成个性化糖尿病防控健康管理报告：

{user_profile}

请从饮食、运动、作息、用药监测及其他生活领域，给出简洁、针对性强的建议。"""
        
        # 获取 API Key
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            # 如果没有设置环境变量，使用默认值（生产环境请设置环境变量）
            api_key = 'your_deepseek_api_key_here'
        
        # 初始化 DeepSeek 客户端
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # 调用 API 生成报告
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # 提取生成的报告内容
        report = response.choices[0].message.content
        
        return report, None
        
    except Exception as e:
        error_msg = f"AI 报告生成失败: {str(e)}"
        print(error_msg)
        return None, error_msg


def format_user_health_profile(data):
    """
    格式化用户健康数据为可读文本
    
    :param data: 用户健康指标数据（字典格式）
    :return: 格式化的健康档案文本
    """
    profile_parts = []
    
    # 基本信息
    profile_parts.append(f"年龄：{data.get('年龄', '未知')}岁")
    profile_parts.append(f"性别：{'女性' if data.get('性别') == 1 else '男性'}")
    profile_parts.append(f"体重指数（BMI）：{data.get('体重指数', '未知')}")
    
    # 生活习惯
    smoking = "吸烟" if data.get('吸烟状态') == 1 else "不吸烟"
    alcohol = f"{data.get('饮酒量', 0)}单位/周"
    exercise = f"{data.get('每周体育活动时间', 0)}小时/周"
    diet_quality = f"饮食质量评分：{data.get('饮食质量', 5)}/10"
    sleep = f"睡眠质量评分：{data.get('睡眠质量', 7)}/10"
    
    profile_parts.append(f"吸烟状态：{smoking}")
    profile_parts.append(f"饮酒量：{alcohol}")
    profile_parts.append(f"每周运动时间：{exercise}")
    profile_parts.append(diet_quality)
    profile_parts.append(sleep)
    
    # 病史
    history = []
    if data.get('糖尿病家族史') == 1:
        history.append("有糖尿病家族史")
    if data.get('高血压') == 1:
        history.append("有高血压")
    if data.get('妊娠期糖尿病') == 1:
        history.append("有妊娠期糖尿病史")
    if data.get('多囊卵巢综合征') == 1:
        history.append("有多囊卵巢综合征")
    if data.get('既往糖尿病前期') == 1:
        history.append("有糖尿病前期史")
    
    if history:
        profile_parts.append("病史：" + "，".join(history))
    
    # 临床指标
    profile_parts.append(f"收缩压：{data.get('收缩压', '未知')}mmHg")
    profile_parts.append(f"舒张压：{data.get('舒张压', '未知')}mmHg")
    profile_parts.append(f"空腹血糖：{data.get('空腹血糖', '未知')}mg/dL")
    profile_parts.append(f"糖化血红蛋白：{data.get('糖化血红蛋白', '未知')}%")
    profile_parts.append(f"总胆固醇：{data.get('总胆固醇', '未知')}mg/dL")
    profile_parts.append(f"甘油三酯：{data.get('甘油三酯', '未知')}mg/dL")
    
    # 症状
    symptoms = []
    if data.get('尿频') == 1:
        symptoms.append("尿频")
    if data.get('过度口渴') == 1:
        symptoms.append("过度口渴")
    if data.get('不明原因体重下降') == 1:
        symptoms.append("不明原因体重下降")
    if data.get('疲劳程度', 0) > 5:
        symptoms.append(f"疲劳程度较高（{data.get('疲劳程度', 0)}/10）")
    if data.get('视力模糊') == 1:
        symptoms.append("视力模糊")
    if data.get('伤口愈合缓慢') == 1:
        symptoms.append("伤口愈合缓慢")
    if data.get('手脚刺痛') == 1:
        symptoms.append("手脚刺痛")
    
    if symptoms:
        profile_parts.append("主要症状：" + "、".join(symptoms))
    
    # 药物使用
    meds = []
    if data.get('降压药物使用') == 1:
        meds.append("降压药")
    if data.get('他汀类药物使用') == 1:
        meds.append("他汀类")
    if data.get('抗糖尿病药物使用') == 1:
        meds.append("降糖药")
    
    if meds:
        profile_parts.append("正在使用药物：" + "、".join(meds))
    
    # 其他指标
    profile_parts.append(f"生活质量评分：{data.get('生活质量评分', 49)}/100")
    
    return "\n".join(profile_parts)
