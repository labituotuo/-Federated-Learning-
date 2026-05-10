import pymysql

conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='252525lht',
    database='fl_federated_learning',
    charset='utf8mb4'
)

cursor = conn.cursor(pymysql.cursors.DictCursor)
cursor.execute('SELECT id, 准确率, 模型文件路径 FROM training_results ORDER BY 准确率 DESC LIMIT 5')
results = cursor.fetchall()

print('数据库中的模型（按准确率排序）:')
for r in results:
    model_name = r['模型文件路径'].split('/')[-1] if r['模型文件路径'] else 'None'
    print(f"ID:{r['id']} 准确率:{r['准确率']:.2f}% 模型:{model_name}")

conn.close()