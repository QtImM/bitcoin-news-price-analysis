import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.api import OLS
from statsmodels.tools import add_constant
import seaborn as sns
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保输出目录存在
os.makedirs("analysis", exist_ok=True)

def load_merged_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.dropna(subset=['price'])  
    return df

def generate_features(df):
    # 基础特征
    df['log_return'] = np.log(df['price']) - np.log(df['price'].shift(1))
    df['volatility_7d'] = df['log_return'].rolling(7).std() * np.sqrt(7)
    
    # 滞后特征
    for lag in [1, 3, 5]:
        df[f'sentiment_lag{lag}'] = df['sentiment_avg'].shift(lag)
        
    return df.dropna()

def advanced_garch_analysis(merged_data):
    """
    使用GARCH模型分析比特币收益率与新闻情感的关系
    """
    try:
        # 构建包含新闻情感的GARCH-M模型
        data = merged_data.copy()
        
        # 确保数据对齐
        data['log_return'] = np.log(data['price']).diff()
        data['sentiment_lag1'] = data['sentiment_avg'].shift(1)
        
        # 删除缺失值
        data = data.dropna(subset=['log_return', 'sentiment_lag1'])
        
        # 准备模型数据
        returns = data['log_return'] * 100
        X = data[['sentiment_lag1']]
        
        # 拟合GARCH模型
        model = arch_model(returns, vol='GARCH', p=1, q=1, mean='AR', lags=1, 
                         dist='skewt', x=X)
        res = model.fit(disp='off', update_freq=0)
        
        # 输出结果和绘图
        print(res.summary())
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 波动率图
        plt.subplot(2, 1, 1)
        plt.plot(data.date, res.conditional_volatility, 'r-', linewidth=1.5)
        plt.title('加密货币收益率条件波动率')
        plt.ylabel('波动率 (%)')
        plt.grid(True)
        
        # 散点图
        plt.subplot(2, 1, 2)
        plt.scatter(data.sentiment_lag1, returns, alpha=0.5)
        plt.xlabel('滞后一期情感得分')
        plt.ylabel('收益率 (%)')
        plt.title('滞后情感得分与收益率关系')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("analysis/garch_analysis.png")
        
        return res
        
    except Exception as e:
        print(f"GARCH模型拟合出错: {str(e)}")
        return None

def visualize_results(df):
    try:
        # 1. 价格趋势图
        plt.figure(figsize=(12, 6))
        plt.plot(df.date, df.price, 'b-')
        plt.title('价格走势')
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.grid(True)
        plt.savefig("analysis/price_trend.png")
        plt.close()
        
        # 2. 新闻数量与情感得分
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.bar(df.date, df.news_count, alpha=0.6)
        plt.title('每日新闻数量')
        plt.ylabel('新闻数量')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(df.date, df.sentiment_avg, 'g-')
        plt.title('平均情感得分')
        plt.xlabel('日期')
        plt.ylabel('情感得分')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("analysis/news_sentiment.png")
        plt.close()
        
        # 3. 情感与收益率的相关性热图
        plt.figure(figsize=(10, 8))
        corr_matrix = df[['price', 'news_count', 'sentiment_avg', 'log_return']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('变量相关性热图')
        plt.tight_layout()
        plt.savefig("analysis/correlation_heatmap.png")
        plt.close()
        
    except Exception as e:
        print(f"可视化生成出错: {str(e)}")

if __name__ == "__main__":
    try:
        # 设置输入文件路径
        INPUT_PATH = "analysis/merged_data.csv"
        
        # 读取数据
        print("正在读取数据...")
        df = load_merged_data(INPUT_PATH)
        df = generate_features(df)
        
        print("\n执行GARCH分析...")
        garch_res = advanced_garch_analysis(df)
        
        print("\n生成可视化结果...")
        visualize_results(df)
        
        print("分析完成！")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")