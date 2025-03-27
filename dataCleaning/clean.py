import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("bitcoin_prices_20250324.csv")
df['date'] = pd.to_datetime(df['date'])

# 检查缺失值
print("缺失值统计:\n", df.isnull().sum())

# 线性插值填充（适用于少量缺失）
df['price'] = df['price'].interpolate(method='linear')

# 检测异常值（价格波动超过3个标准差）
mean_price = df['price'].mean()
std_price = df['price'].std()
df['is_outlier'] = np.abs(df['price'] - mean_price) > 3 * std_price
print("异常值数量:", df['is_outlier'].sum())

# 剔除异常值（或保留但标记）
df_clean = df[~df['is_outlier']].copy()
print('-----------------------------')
# 计算对数收益率（更符合金融时间序列特性）
df_clean['return'] = np.log(df_clean['price']) - np.log(df_clean['price'].shift(1))

# 删除首行空值
df_clean.dropna(subset=['return'], inplace=True)

# 检查收益率分布
print(df_clean['return'].describe())