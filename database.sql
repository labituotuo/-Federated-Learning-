-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 用户健康指标数据表（用户画像）
CREATE TABLE IF NOT EXISTS user_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    
    -- 基本信息
    年龄 INT COMMENT '年龄范围: 20-90',
    性别 TINYINT COMMENT '0-女, 1-男',
    种族 TINYINT COMMENT '0-其他, 1-汉族, 2-少数民族, 3-外籍',
    社会经济地位 TINYINT COMMENT '0-低, 1-中, 2-高',
    教育水平 TINYINT COMMENT '0-小学及以下, 1-初中, 2-高中/中专, 3-大专及以上',
    
    -- 生活方式指标
    体重指数 FLOAT COMMENT 'BMI指数',
    吸烟状态 TINYINT COMMENT '0-不吸烟, 1-吸烟',
    饮酒量 FLOAT COMMENT '饮酒量',
    每周体育活动时间 FLOAT COMMENT '每周运动时间(小时)',
    饮食质量 FLOAT COMMENT '饮食质量评分',
    睡眠质量 FLOAT COMMENT '睡眠质量评分',
    
    -- 病史
    糖尿病家族史 TINYINT COMMENT '0-无, 1-有',
    妊娠期糖尿病 TINYINT COMMENT '0-无, 1-有',
    多囊卵巢综合征 TINYINT COMMENT '0-无, 1-有',
    既往糖尿病前期 TINYINT COMMENT '0-无, 1-有',
    高血压 TINYINT COMMENT '0-无, 1-有',
    
    -- 生理指标
    收缩压 INT COMMENT '收缩压(mmHg), 范围: 90-200',
    舒张压 INT COMMENT '舒张压(mmHg), 范围: 50-120',
    空腹血糖 FLOAT COMMENT '空腹血糖(mmol/L)',
    糖化血红蛋白 FLOAT COMMENT '糖化血红蛋白(%)',
    血清肌酐 FLOAT COMMENT '血清肌酐',
    血尿素氮水平 FLOAT COMMENT '血尿素氮水平',
    
    -- 血脂指标
    总胆固醇 FLOAT COMMENT '总胆固醇(mmol/L)',
    低密度脂蛋白胆固醇 FLOAT COMMENT '低密度脂蛋白胆固醇(mmol/L)',
    高密度脂蛋白胆固醇 FLOAT COMMENT '高密度脂蛋白胆固醇(mmol/L)',
    甘油三酯 FLOAT COMMENT '甘油三酯(mmol/L)',
    
    -- 药物使用
    降压药物使用 TINYINT COMMENT '0-未使用, 1-使用',
    他汀类药物使用 TINYINT COMMENT '0-未使用, 1-使用',
    抗糖尿病药物使用 TINYINT COMMENT '0-未使用, 1-使用',
    
    -- 症状指标
    尿频 TINYINT COMMENT '0-无, 1-有',
    过度口渴 TINYINT COMMENT '0-无, 1-有',
    不明原因体重下降 TINYINT COMMENT '0-无, 1-有',
    疲劳程度 FLOAT COMMENT '疲劳程度评分',
    视力模糊 TINYINT COMMENT '0-无, 1-有',
    伤口愈合缓慢 TINYINT COMMENT '0-无, 1-有',
    手脚刺痛 TINYINT COMMENT '0-无, 1-有',
    
    -- 其他指标
    生活质量评分 FLOAT COMMENT '生活质量评分',
    重金属暴露 TINYINT COMMENT '0-无, 1-有',
    职业化学物质暴露 TINYINT COMMENT '0-无, 1-有',
    水质 TINYINT COMMENT '0-差, 1-好',
    体检频率 FLOAT COMMENT '体检频率(次/年)',
    药物依从性 FLOAT COMMENT '药物依从性评分',
    健康素养 FLOAT COMMENT '健康素养评分',
    
    -- 预测风险
    预测风险 FLOAT COMMENT '糖尿病预测风险概率(0-1)',
    
    -- 备注和时间戳
    备注 TEXT COMMENT '其他信息',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_user_metrics_user_id ON user_metrics(user_id);
CREATE INDEX IF NOT EXISTS idx_user_metrics_created_at ON user_metrics(created_at);
