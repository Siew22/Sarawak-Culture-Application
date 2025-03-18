import pandas as pd

# ✅ 确保 `header=0` 让 Pandas 识别第一行作为列名
df = pd.read_excel(r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\data\user_preferences.xlsx", header=0)

# ✅ 删除完全空的列和行，避免数据错位
df = df.dropna(how="all")  # 删除空行
df = df.dropna(axis=1, how="all")  # 删除空列

# ✅ 确保所有列名都是字符串格式
df.columns = [str(col).strip().lower() for col in df.columns]

# ✅ 打印结果检查
print(df.shape)  # 应该是 (1,3) 或者 (N,3)
print(df.head())  # 预览数据
