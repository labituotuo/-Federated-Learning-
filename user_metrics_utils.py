# -*- coding: utf-8 -*-
"""用户指标工具模块"""
from db_utils import get_db_connection

USER_METRICS_COLUMNS = [
    'user_id', '年龄', '性别', '种族', '社会经济地位', '教育水平',
    '体重指数', '吸烟状态', '饮酒量', '每周体育活动时间', '饮食质量', '睡眠质量',
    '糖尿病家族史', '妊娠期糖尿病', '多囊卵巢综合征', '既往糖尿病前期', '高血压',
    '收缩压', '舒张压', '空腹血糖', '糖化血红蛋白', '血清肌酐', '血尿素氮水平',
    '总胆固醇', '低密度脂蛋白胆固醇', '高密度脂蛋白胆固醇', '甘油三酯',
    '降压药物使用', '他汀类药物使用', '抗糖尿病药物使用',
    '尿频', '过度口渴', '不明原因体重下降', '疲劳程度', '视力模糊', '伤口愈合缓慢', '手脚刺痛',
    '生活质量评分', '重金属暴露', '职业化学物质暴露', '水质', '体检频率', '药物依从性', '健康素养',
    '预测风险', '备注'
]

def save_user_metrics(user_id, data):
    """保存用户指标数据"""
    conn = get_db_connection()
    if not conn:
        return False, '数据库连接失败'

    try:
        cursor = conn.cursor()
        placeholders = ', '.join(['%s'] * len(USER_METRICS_COLUMNS))
        columns = ', '.join(USER_METRICS_COLUMNS)

        sql = f'''
            INSERT INTO user_metrics ({columns})
            VALUES ({placeholders})
        '''

        values = [
            user_id,
            data.get('年龄'), data.get('性别'), data.get('种族'), data.get('社会经济地位'), data.get('教育水平'),
            data.get('体重指数'), data.get('吸烟状态'), data.get('饮酒量'), data.get('每周体育活动时间'),
            data.get('饮食质量'), data.get('睡眠质量'),
            data.get('糖尿病家族史'), data.get('妊娠期糖尿病'), data.get('多囊卵巢综合征'),
            data.get('既往糖尿病前期'), data.get('高血压'),
            data.get('收缩压'), data.get('舒张压'), data.get('空腹血糖'), data.get('糖化血红蛋白'),
            data.get('血清肌酐'), data.get('血尿素氮水平'),
            data.get('总胆固醇'), data.get('低密度脂蛋白胆固醇'), data.get('高密度脂蛋白胆固醇'),
            data.get('甘油三酯'),
            data.get('降压药物使用'), data.get('他汀类药物使用'), data.get('抗糖尿病药物使用'),
            data.get('尿频'), data.get('过度口渴'), data.get('不明原因体重下降'), data.get('疲劳程度'),
            data.get('视力模糊'), data.get('伤口愈合缓慢'), data.get('手脚刺痛'),
            data.get('生活质量评分'), data.get('重金属暴露'), data.get('职业化学物质暴露'),
            data.get('水质'), data.get('体检频率'), data.get('药物依从性'), data.get('健康素养'),
            data.get('预测风险'), data.get('备注')
        ]

        cursor.execute(sql, values)
        conn.commit()
        cursor.close()
        return True, '指标数据保存成功'

    except Exception as e:
        print(f"保存用户指标失败: {str(e)}")
        return False, f'保存失败: {str(e)}'

def get_user_metrics(user_id):
    """获取用户的所有历史指标数据"""
    conn = get_db_connection()
    if not conn:
        return None, '数据库连接失败'

    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM user_metrics
            WHERE user_id = %s
            ORDER BY created_at DESC
        ''', (user_id,))

        records = cursor.fetchall()
        cursor.close()
        return records, None

    except Exception as e:
        print(f"获取用户指标失败: {str(e)}")
        return None, f'获取失败: {str(e)}'

def get_latest_user_metrics(user_id):
    """获取用户最新的指标数据"""
    conn = get_db_connection()
    if not conn:
        return None, '数据库连接失败'

    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM user_metrics
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        ''', (user_id,))

        record = cursor.fetchone()
        cursor.close()
        return record, None

    except Exception as e:
        print(f"获取最新用户指标失败: {str(e)}")
        return None, f'获取失败: {str(e)}'

def delete_user_metrics_record(record_id):
    """删除用户指标记录"""
    conn = get_db_connection()
    if not conn:
        return False, '数据库连接失败'

    try:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM user_metrics WHERE id = %s', (record_id,))
        conn.commit()
        cursor.close()
        return True, None

    except Exception as e:
        print(f"删除用户指标失败: {str(e)}")
        return False, f'删除失败: {str(e)}'

def get_user_id_by_username(username):
    """根据用户名获取用户ID"""
    conn = get_db_connection()
    if not conn:
        return None, '数据库连接失败'

    try:
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            return user['id'], None
        return None, '用户不存在'

    except Exception as e:
        print(f"获取用户ID失败: {str(e)}")
        return None, f'获取失败: {str(e)}'

def compare_metrics(current_record, previous_record):
    """对比两次指标数据，返回变化较大的特征"""
    if not previous_record:
        return []

    changes = []
    numeric_columns = [
        '年龄', '体重指数', '饮酒量', '每周体育活动时间', '饮食质量', '睡眠质量',
        '收缩压', '舒张压', '空腹血糖', '糖化血红蛋白', '血清肌酐', '血尿素氮水平',
        '总胆固醇', '低密度脂蛋白胆固醇', '高密度脂蛋白胆固醇', '甘油三酯',
        '疲劳程度', '生活质量评分', '体检频率', '药物依从性', '健康素养'
    ]

    for col in numeric_columns:
        current_val = current_record.get(col)
        previous_val = previous_record.get(col)

        if current_val is not None and previous_val is not None and previous_val != 0:
            change_ratio = abs(current_val - previous_val) / abs(previous_val)
            if change_ratio > 0.05:
                changes.append({
                    'feature': col,
                    'current': current_val,
                    'previous': previous_val,
                    'change_ratio': round(change_ratio * 100, 2)
                })

    changes.sort(key=lambda x: x['change_ratio'], reverse=True)
    return changes[:5]