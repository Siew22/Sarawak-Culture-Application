import mysql.connector
import os
from dotenv import load_dotenv

# 载入 .env 文件
load_dotenv()

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
            ssl_disabled=True,  # ❌ 关闭 SSL（仅用于本地测试）
            pool_name="mypool",
            pool_size=5,
            connect_timeout=30
        )
        print("✅ 成功连接到数据库！")
        return connection
    except mysql.connector.Error as err:
        print(f"❌ 数据库连接失败: {err}")
        return None  # 避免应用崩溃
