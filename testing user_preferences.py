import pandas as pd
import io
import main

# 模拟 Excel 文件内容
excel_data = pd.DataFrame([
    ["Location", "Days", "Food (0-5)", "Attraction (0-3)", "Experience (0-2)"],
    ["Kuching", "3", "5", "2", "1"],
    ["Miri", "1", "1", "2", "2"],
    ["Perak", "4", "4", "2", "1"],
    ["Bintulu", "3", "5", "1", "0"],
    ["Kuala Lumpur", "7", "3", "2", "2"],
    ["Penang", "6", "4", "3", "2"],
    ["Sabah", "8", "5", "2", "2"]
])

# 保存到临时文件
excel_data.to_excel("test_preferences.xlsx", index=False, header=False)

# 读取文件并测试
with open("test_preferences.xlsx", "rb") as f:
    contents = f.read()
    preferences_data = main.get_user_preferences(io.BytesIO(contents))
    print("Parsed preferences:", preferences_data)