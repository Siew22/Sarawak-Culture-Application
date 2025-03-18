import pandas as pd
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_user_preferences(file_path=None):
    """读取用户 Excel 数据"""
    try:
        if file_path is None:
            # 使用相对路径，假设 user_preferences.xlsx 在 ai_travel_assistant/data 目录下
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\data\user_preferences.xlsx"
        
        logger.info(f"Loading data from: {file_path}")
        
        # 读取 Excel 文件，指定引擎为 openpyxl 以支持 .xlsx
        # 不指定 header，使用索引作为列名
        df = pd.read_excel(file_path, engine='openpyxl', header=None)
        
        # 添加调试信息
        logger.info(f"Excel 数据预览:\n{df.head()}")
        
        # 检查数据格式，如果是行式数据（列名在A列），转换为列式
        if df.shape[1] >= 2:  # 至少有两列（类别和数值）
            # 检查第一列是否包含我们期望的类别名称
            categories = df[0].astype(str).str.lower().tolist()
            required_categories = ['food', 'experiences', 'attractions']
            
            # 检查是否所有必需类别都存在
            missing_categories = [cat for cat in required_categories if cat not in categories]
            if missing_categories:
                logger.error(f"Excel 文件缺少必要的类别: {', '.join(missing_categories)}")
                raise ValueError(f"Excel 文件缺少必要的类别: {', '.join(missing_categories)}")
            
            # 创建新的 DataFrame，每个类别作为一列
            new_df = pd.DataFrame()
            for category in required_categories:
                # 找到类别所在的行
                category_rows = df[df[0].astype(str).str.lower() == category].index.tolist()
                if category_rows:
                    # 获取该类别的值（假设值在第二列）
                    new_df[category] = [df.iloc[category_rows[0], 1]]
                else:
                    raise ValueError(f"无法找到类别: {category}")
            
            df = new_df
            logger.info(f"转换后的数据:\n{df.head()}")
        else:
            # 假设数据已经是列式格式
            # 确保列名是字符串
            df.columns = [str(col) for col in df.columns]
            
            # 如果第一行包含列名，将其设为列标题
            if not df.empty:
                first_row = df.iloc[0]
                if any(str(val).lower() in ['food', 'experiences', 'attractions'] for val in first_row if pd.notna(val)):
                    # 使用第一行作为列名
                    df.columns = first_row
                    df = df.drop(0)
            
            # 检查必要的列是否存在
            required_columns = ['food', 'experiences', 'attractions']
            for col in required_columns:
                # 查找匹配的列（忽略大小写）
                matched_cols = [c for c in df.columns if str(c).lower() == col.lower()]
                if not matched_cols:
                    raise ValueError(f"Excel 文件缺少必要的列: {col}")
                
                # 重命名列为统一格式
                if matched_cols[0] != col:
                    df = df.rename(columns={matched_cols[0]: col})
        
        # 确保列是数值型
        for col in ['food', 'experiences', 'attractions']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # 转换为空值或数值
        
        logger.info("Data loaded successfully.")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except ValueError as e:
        logger.error(f"Invalid Excel format: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading file: {e}")
        return None

if __name__ == "__main__":
    df = load_user_preferences()
    if df is not None:
        print("成功读取数据:")
        print(df)  # 测试是否成功读取数据
    else:
        print("Failed to load data.")