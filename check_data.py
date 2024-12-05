import pandas as pd

# 读取数据文件
df = pd.read_csv('/home/lx/LSTMCoal/data/processed/processed_data.csv')

# 显示列名
print("列名：")
print(df.columns.tolist())

# 显示前几行数据
print("\n数据预览：")
print(df.head())