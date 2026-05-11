# -*- coding: utf-8 -*-
"""
数据库迁移脚本：将「加密方式」和「隐私保护」列合并为「隐私保护方式」列
"""
import sys
import pymysql

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '252525lht',
    'database': 'fl_federated_learning',
    'charset': 'utf8mb4'
}

def migrate_database():
    """执行数据库迁移"""
    print('[INFO] 尝试连接数据库...')
    
    try:
        connection = pymysql.connect(**DB_CONFIG)
        print('[OK] 数据库连接成功')
    except Exception as e:
        print(f'[ERROR] 数据库连接失败: {e}')
        return False

    try:
        with connection.cursor() as cursor:
            # 检查是否存在旧的加密方式列
            cursor.execute("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'fl_federated_learning' 
                AND TABLE_NAME = 'training_results' 
                AND COLUMN_NAME = '加密方式'
            """)
            has_old_columns = cursor.fetchone()[0] > 0

            if has_old_columns:
                print('[INFO] 发现旧的列结构，开始迁移...')
                
                # 1. 添加新列
                print('[STEP 1] 添加「隐私保护方式」列...')
                try:
                    cursor.execute("""
                        ALTER TABLE training_results 
                        ADD COLUMN 隐私保护方式 VARCHAR(50) DEFAULT 'none' 
                        COMMENT '隐私保护方式: none, homomorphic, differential'
                    """)
                    connection.commit()
                    print('[OK] 新列添加成功')
                except pymysql.err.OperationalError as e:
                    if e.args[0] == 1060:  # Column already exists
                        print('[INFO] 列已存在，跳过')
                    else:
                        raise

                # 2. 更新现有数据
                print('[STEP 2] 更新现有数据...')
                cursor.execute("""
                    UPDATE training_results 
                    SET 隐私保护方式 = CASE
                        WHEN 加密方式 = 'homomorphic' THEN 'homomorphic'
                        WHEN 隐私保护 = 'differential' THEN 'differential'
                        ELSE 'none'
                    END
                    WHERE 隐私保护方式 = 'none'
                """)
                connection.commit()
                print(f'[OK] 更新了 {cursor.rowcount} 条记录')

                # 3. 删除旧列（可选）
                print('[STEP 3] 删除旧的「加密方式」和「隐私保护」列...')
                for col_name in ['加密方式', '隐私保护']:
                    try:
                        cursor.execute(f"ALTER TABLE training_results DROP COLUMN {col_name}")
                        connection.commit()
                        print(f'[OK] 删除列 {col_name} 成功')
                    except pymysql.err.OperationalError as e:
                        if e.args[0] == 1091:  # Column doesn't exist
                            print(f'[INFO] 列 {col_name} 不存在，跳过')
                        else:
                            raise
                print('[OK] 旧列删除完成')

                print('[OK] 数据库迁移完成！')
            else:
                # 检查新列是否已经存在
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = 'fl_federated_learning' 
                    AND TABLE_NAME = 'training_results' 
                    AND COLUMN_NAME = '隐私保护方式'
                """)
                has_new_column = cursor.fetchone()[0] > 0
                
                if has_new_column:
                    print('[INFO] 数据库结构已是最新，无需迁移')
                else:
                    print('[INFO] 需要添加新列...')
                    cursor.execute("""
                        ALTER TABLE training_results 
                        ADD COLUMN IF NOT EXISTS 隐私保护方式 VARCHAR(50) DEFAULT 'none' 
                        COMMENT '隐私保护方式: none, homomorphic, differential'
                    """)
                    connection.commit()
                    print('[OK] 新列添加成功')

        return True
    except Exception as e:
        print(f'[ERROR] 迁移失败: {str(e)}')
        import traceback
        traceback.print_exc()
        return False
    finally:
        if connection:
            connection.close()

if __name__ == '__main__':
    print('='*60)
    print('数据库迁移脚本')
    print('='*60)
    print('此脚本将「加密方式」和「隐私保护」列合并为「隐私保护方式」列')
    print()
    
    if migrate_database():
        print()
        print('迁移成功！')
    else:
        print()
        print('迁移失败！')
        sys.exit(1)
