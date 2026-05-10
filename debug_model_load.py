"""调试模型加载逻辑"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask_app import load_model

# 调用模型加载函数
success, msg = load_model()
print(f"模型加载结果: {success}")
print(f"消息: {msg}")

# 检查全局变量
from flask_app import loaded_model, model_version, model_accuracy, model_saved_at
print(f"\n加载的模型版本: {model_version}")
print(f"模型准确率: {model_accuracy}")
print(f"模型保存时间: {model_saved_at}")