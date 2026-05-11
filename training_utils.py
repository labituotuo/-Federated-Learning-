# -*- coding: utf-8 -*-
"""训练结果工具模块"""
import json
from datetime import datetime
from db_utils import get_db_connection

def save_training_result(record):
    """保存训练结果到数据库"""
    conn = get_db_connection()
    if not conn:
        print('[ERROR] 无法保存训练结果：数据库连接失败')
        return False

    try:
        with conn.cursor() as cursor:
            # 合并加密方式和隐私保护为一列
            privacy_method = record.get('隐私保护方式', 'none')
            if not privacy_method or privacy_method == 'none':
                encryption_val = record.get('加密方式', 'none')
                privacy_val = record.get('隐私保护', 'none')
                if encryption_val == 'homomorphic':
                    privacy_method = 'homomorphic'
                elif privacy_val == 'differential':
                    privacy_method = 'differential'

            sql = """
            INSERT INTO training_results
            (算法, 隐私保护方式, 客户端数量, 全局轮次, 本地批次大小, 本地学习率,
             准确率, 精确率, 查全率, F1分数, AUC分数, 时间消耗秒, 模型文件名, 模型文件路径,
             隐私预算, 噪声标准差, 总样本数, 客户端样本分布, 额外参数)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
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
            conn.commit()
            print(f'[OK] 训练结果已保存到数据库 (算法: {record.get("算法", "FedAvg")})')
            return True
    except Exception as e:
        print(f'[ERROR] 保存训练结果失败: {str(e)}')
        return False
    finally:
        if conn:
            conn.close()

def save_federated_training_result(algorithm, encryption, privacy, num_clients, fed_rounds,
                                   batch_size, learning_rate, metrics, total_duration,
                                   model_filename='', model_path='', epsilon=None, delta=None,
                                   total_samples=0, client_distributions=None, extra_params=None):
    """保存联邦学习训练结果的便捷函数"""
    last_round = metrics.get('rounds', [{}])[-1] if metrics.get('rounds') else {}

    # 合并加密方式和隐私保护为一列
    privacy_method = 'none'
    if encryption == 'homomorphic':
        privacy_method = 'homomorphic'
    elif privacy == 'differential' or encryption == 'differential_privacy':
        privacy_method = 'differential'

    record = {
        '算法': algorithm,
        '隐私保护方式': privacy_method,
        '客户端数量': num_clients,
        '全局轮次': fed_rounds,
        '本地批次大小': batch_size,
        '本地学习率': learning_rate,
        '准确率': last_round.get('global_test_accuracy', 0.0),
        '精确率': last_round.get('global_test_precision', 0.0),
        '查全率': last_round.get('global_test_recall', 0.0),
        'F1分数': last_round.get('global_test_f1_score', 0.0),
        'AUC分数': last_round.get('global_test_accuracy', 0.0),
        '时间消耗秒': total_duration,
        '模型文件名': model_filename,
        '模型文件路径': model_path,
        '隐私预算': epsilon if privacy == 'differential' else None,
        '噪声标准差': None,
        '总样本数': total_samples,
        '客户端样本分布': json.dumps(client_distributions) if client_distributions else '{}',
        '额外参数': json.dumps(extra_params) if extra_params else '{}'
    }

    return save_training_result(record)

def get_all_training_results(limit=50):
    """获取所有训练结果"""
    conn = get_db_connection()
    if not conn:
        return None, '数据库连接失败'

    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM training_results
            ORDER BY created_at DESC
            LIMIT %s
        ''', (limit,))

        records = cursor.fetchall()
        cursor.close()
        return records, None

    except Exception as e:
        print(f"获取训练结果失败: {str(e)}")
        return None, f'获取失败: {str(e)}'

def get_training_result_by_id(result_id):
    """根据ID获取训练结果"""
    conn = get_db_connection()
    if not conn:
        return None, '数据库连接失败'

    try:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM training_results WHERE id = %s', (result_id,))
        record = cursor.fetchone()
        cursor.close()
        return record, None

    except Exception as e:
        print(f"获取训练结果详情失败: {str(e)}")
        return None, f'获取失败: {str(e)}'

def get_training_results_by_algorithm(algorithm):
    """根据算法类型获取训练结果"""
    conn = get_db_connection()
    if not conn:
        return None, '数据库连接失败'

    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM training_results
            WHERE 算法 = %s
            ORDER BY created_at DESC
            LIMIT 50
        ''', (algorithm,))

        records = cursor.fetchall()
        cursor.close()
        return records, None

    except Exception as e:
        print(f"获取训练结果失败: {str(e)}")
        return None, f'获取失败: {str(e)}'

def delete_training_result(result_id):
    """删除训练结果"""
    conn = get_db_connection()
    if not conn:
        return False, '数据库连接失败'

    try:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM training_results WHERE id = %s', (result_id,))
        conn.commit()
        cursor.close()
        return True, None

    except Exception as e:
        print(f"删除训练结果失败: {str(e)}")
        return False, f'删除失败: {str(e)}'