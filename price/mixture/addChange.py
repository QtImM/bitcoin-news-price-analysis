# 给指数加入一列 change_percent，表示当前价格与前一天价格变化的百分比（去掉百分号）
# 删除 change 列，并将 change_percent 列保留两位小数
import pandas as pd

# 读取数据
file_path = "price\\mixture\\cmc10_weighted_index.csv"
data = pd.read_csv(file_path)

# 删除 change 列
if 'change' in data.columns:
    data = data.drop(columns=['change'])

# 确保日期列为日期类型并按日期排序
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

# 添加 change_percent 列，计算当前价格与前一天价格变化的百分比
data['change_percent'] = data['price'].pct_change() * 100  # 计算百分比变化

# 将 change_percent 列保留两位小数
data['change_percent'] = data['change_percent'].round(2)

# 保存结果到新文件
output_path = "price\\mixture\\cmc10_weighted_index.csv"
data.to_csv(output_path, index=False)

print(f"已成功删除 change 列，并将 change_percent 列保留两位小数，结果保存至 {output_path}")