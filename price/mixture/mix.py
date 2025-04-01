#分别计算权重
import pandas as pd
import os

# 1. 定义权重配置（根据您提供的比例）
weights = {
    'BTC': 0.70,    # 70%
    'ETH': 0.10,    # 10%
    'XRP': 0.05,    # 5%
    'BNB': 0.04,    # 4%
    'SOL': 0.03,    # 3%
    'DOGE': 0.016,  # 1.6%
    'ADA': 0.016,   # 1.6%
    'TRX': 0.016,   # 1.6%
    'TON': 0.016,   # 1.6%
    'LINK': 0.016   # 1.6%
}

# 2. 读取所有CSV文件并预处理
def load_and_preprocess(file_path, coin_name):
    df = pd.read_csv(file_path)
    # 确保日期列统一格式
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    # 重命名价格列为币种名称
    df = df.rename(columns={'Price': coin_name})
    # 转换价格列为浮点数
    df[coin_name] = pd.to_numeric(df[coin_name], errors='coerce')  # 将价格列转换为浮点数
    return df[['Date', coin_name]]

# 3. 主处理函数
def process_files(input_folder, output_file):
    # 加载所有数据
    all_data = []
    for coin, weight in weights.items():
        file_path = os.path.join(input_folder, f"{coin}.csv")
        print(file_path)
        if os.path.exists(file_path):
            df = load_and_preprocess(file_path, coin)
            all_data.append(df)
        else:
            print(f"警告: {coin}.csv 文件未找到")

    # 检查是否有数据
    if not all_data:
        print("未找到任何数据文件，无法继续处理。")
        return

    # 合并所有数据
    merged_df = all_data[0]
    for df in all_data[1:]:
        merged_df = pd.merge(merged_df, df, on='Date', how='outer')
    
    # 按日期排序
    merged_df = merged_df.sort_values('Date')
    
    # 4. 计算加权价格
    for coin in weights.keys():
        if coin in merged_df.columns:
            merged_df[coin] = merged_df[coin] * weights[coin]
    
    # 计算综合价格（加权求和）
    merged_df['Price'] = (merged_df[weights.keys()].sum(axis=1) * 100).round(2)
    
    # 5. 保存结果
    result_df = merged_df[['date', 'price']]
    result_df.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}")
    
    # 显示前5行验证
    print("\n结果预览:")
    print(result_df.head())

# 6. 使用示例
if __name__ == "__main__":
    input_folder = "price/mixture/data"  # 存放10个CSV文件的文件夹
    output_file = "price/mixture/cmc10_weighted_index.csv"
    
    # 验证权重总和是否为100%
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.001:
        print(f"警告: 权重总和为{total_weight*100}%，不等于100%")
    
    process_files(input_folder, output_file)