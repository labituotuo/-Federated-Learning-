# -*- coding: utf-8 -*-
"""数据库工具模块"""
import pymysql
from pymysql.cursors import DictCursor

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '252525lht',
    'database': 'fl_federated_learning',
    'charset': 'utf8mb4',
    'cursorclass': DictCursor
}

DB_CONNECTION = None

def get_db_connection():
    """获取数据库连接"""
    global DB_CONNECTION
    if DB_CONNECTION is None or not DB_CONNECTION.open:
        try:
            DB_CONNECTION = pymysql.connect(**DB_CONFIG)
        except pymysql.MySQLError as e:
            print(f"数据库连接失败: {e}")
            DB_CONNECTION = None
    return DB_CONNECTION

def reset_db_connection():
    """重置数据库连接"""
    global DB_CONNECTION
    if DB_CONNECTION and DB_CONNECTION.open:
        DB_CONNECTION.close()
    DB_CONNECTION = None

def init_database():
    """初始化数据库表"""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                ''')

                cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    训练时间 TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    算法 VARCHAR(50) NOT NULL COMMENT '算法名称: FedAvg, FedProx, SCAFFOLD',
                    隐私保护方式 VARCHAR(50) DEFAULT 'none' COMMENT '隐私保护方式: none, homomorphic, differential',
                    客户端数量 INT COMMENT '参与的客户端数量',
                    全局轮次 INT COMMENT '训练轮次',
                    本地批次大小 INT COMMENT '客户端本地批次大小',
                    本地学习率 FLOAT COMMENT '客户端本地学习率',
                    准确率 FLOAT COMMENT '最终准确率',
                    精确率 FLOAT COMMENT '最终精确率',
                    查全率 FLOAT COMMENT '最终查全率',
                    F1分数 FLOAT COMMENT '最终F1分数',
                    AUC分数 FLOAT COMMENT '最终AUC分数',
                    时间消耗秒 FLOAT COMMENT '总训练时间(秒)',
                    模型文件名 VARCHAR(255) COMMENT '保存的模型文件名',
                    模型文件路径 VARCHAR(500) COMMENT '保存的模型文件路径',
                    隐私预算 FLOAT COMMENT '差分隐私预算epsilon值',
                    噪声标准差 FLOAT COMMENT '差分隐私高斯噪声标准差',
                    总样本数 INT COMMENT '参与训练的总样本数',
                    客户端样本分布 TEXT COMMENT '各客户端样本数量分布(JSON)',
                    额外参数 TEXT COMMENT '额外训练参数(JSON)',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_training_results_created_at (created_at),
                    INDEX idx_training_results_algorithm (算法)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                ''')

                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_metrics (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    年龄 INT COMMENT '年龄范围: 20-90',
                    性别 TINYINT COMMENT '0-女, 1-男',
                    种族 TINYINT COMMENT '0-其他, 1-汉族, 2-少数民族, 3-外籍',
                    社会经济地位 TINYINT COMMENT '0-低, 1-中, 2-高',
                    教育水平 TINYINT COMMENT '0-小学及以下, 1-初中, 2-高中/中专, 3-大专及以上',
                    体重指数 FLOAT COMMENT 'BMI指数',
                    吸烟状态 TINYINT COMMENT '0-不吸烟, 1-吸烟',
                    饮酒量 FLOAT COMMENT '饮酒量',
                    每周体育活动时间 FLOAT COMMENT '每周运动时间(小时)',
                    饮食质量 FLOAT COMMENT '饮食质量评分',
                    睡眠质量 FLOAT COMMENT '睡眠质量评分',
                    糖尿病家族史 TINYINT COMMENT '0-无, 1-有',
                    妊娠期糖尿病 TINYINT COMMENT '0-无, 1-有',
                    多囊卵巢综合征 TINYINT COMMENT '0-无, 1-有',
                    既往糖尿病前期 TINYINT COMMENT '0-无, 1-有',
                    高血压 TINYINT COMMENT '0-无, 1-有',
                    收缩压 INT COMMENT '收缩压(mmHg)',
                    舒张压 INT COMMENT '舒张压(mmHg)',
                    空腹血糖 FLOAT COMMENT '空腹血糖(mmol/L)',
                    糖化血红蛋白 FLOAT COMMENT '糖化血红蛋白(%)',
                    血清肌酐 FLOAT COMMENT '血清肌酐',
                    血尿素氮水平 FLOAT COMMENT '血尿素氮水平',
                    总胆固醇 FLOAT COMMENT '总胆固醇(mmol/L)',
                    低密度脂蛋白胆固醇 FLOAT COMMENT '低密度脂蛋白胆固醇(mmol/L)',
                    高密度脂蛋白胆固醇 FLOAT COMMENT '高密度脂蛋白胆固醇(mmol/L)',
                    甘油三酯 FLOAT COMMENT '甘油三酯(mmol/L)',
                    降压药物使用 TINYINT COMMENT '0-未使用, 1-使用',
                    他汀类药物使用 TINYINT COMMENT '0-未使用, 1-使用',
                    抗糖尿病药物使用 TINYINT COMMENT '0-未使用, 1-使用',
                    尿频 TINYINT COMMENT '0-无, 1-有',
                    过度口渴 TINYINT COMMENT '0-无, 1-有',
                    不明原因体重下降 TINYINT COMMENT '0-无, 1-有',
                    疲劳程度 FLOAT COMMENT '疲劳程度评分',
                    视力模糊 TINYINT COMMENT '0-无, 1-有',
                    伤口愈合缓慢 TINYINT COMMENT '0-无, 1-有',
                    手脚刺痛 TINYINT COMMENT '0-无, 1-有',
                    生活质量评分 FLOAT COMMENT '生活质量评分',
                    重金属暴露 TINYINT COMMENT '0-无, 1-有',
                    职业化学物质暴露 TINYINT COMMENT '0-无, 1-有',
                    水质 TINYINT COMMENT '0-差, 1-好',
                    体检频率 FLOAT COMMENT '体检频率(次/年)',
                    药物依从性 FLOAT COMMENT '药物依从性评分',
                    健康素养 FLOAT COMMENT '健康素养评分',
                    预测风险 FLOAT COMMENT '糖尿病预测风险概率(0-1)',
                    备注 TEXT COMMENT '其他信息',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                ''')

                try:
                    cursor.execute('CREATE INDEX idx_user_metrics_user_id ON user_metrics(user_id)')
                except pymysql.MySQLError as e:
                    if "Duplicate key name" not in str(e):
                        print(f"创建索引警告: {e}")

                try:
                    cursor.execute('CREATE INDEX idx_user_metrics_created_at ON user_metrics(created_at)')
                except pymysql.MySQLError as e:
                    if "Duplicate key name" not in str(e):
                        print(f"创建索引警告: {e}")

                cursor.execute("SELECT COUNT(*) as count FROM users WHERE role = 'admin'")
                result = cursor.fetchone()

                if result['count'] == 0:
                    cursor.execute(
                        "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                        ('admin', 'admin123', 'admin')
                    )
                    cursor.execute(
                        "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                        ('user', 'user123', 'user')
                    )
                    conn.commit()
                    print("MySQL数据库初始化完成，默认用户已创建。")
                else:
                    print("MySQL数据库已就绪。")
        except pymysql.MySQLError as e:
            print(f"数据库初始化失败: {e}")
            conn.rollback()
        finally:
            conn.close()
    else:
        print("警告：无法连接到MySQL数据库。系统将使用内存模式（硬编码账号）运行。")
        print("提示：请检查 MySQL 是否启动，以及密码和数据库名是否正确。")