import pandas as pd

# 读取 Excel 文件
file_path = r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\data\user_preferences.xlsx"
df = pd.read_excel(file_path)

# 显示数据
print(df)
