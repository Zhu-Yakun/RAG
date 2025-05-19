import sqlite3
import os

# 显示Chroma数据库的内容
def view_chroma_database_content(db_path):
    # 检查文件路径是否存在
    if not os.path.exists(db_path):
        print(f"文件不存在: {db_path}")
        return

    # 连接到SQLite数据库
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            print("数据库中没有表。")
            return

        print("数据库中的表：")
        for table in tables:
            table_name = table[0]
            print(f"表名: {table_name}")
            # 获取表的列名
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            print(f"列名: {column_names}")
            # 获取表的内容
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 10;")  # 限制显示前10行
            rows = cursor.fetchall()
            print("内容:")
            for row in rows:
                print(row)

        # 关闭连接
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"无法打开数据库文件: {e}")

db_path = 'chroma.sqlite3'  # 替换为你的文件路径
view_chroma_database_content(db_path)