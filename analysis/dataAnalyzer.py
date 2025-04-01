import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.api import OLS
from statsmodels.tools import add_constant
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    
    try:
        model = arch_model(returns, vol='GARCH', p=1, q=1, mean='AR', lags=1, 
                         dist='skewt', x=X)
        res = model.fit(disp='off', update_freq=0)
        
        # 输出结果和绘图
        print(res.summary())
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(data.date, res.conditional_volatility, 'r-', linewidth=1.5)
        plt.title('比特币收益率条件波动率')
        plt.ylabel('波动率 (%)')
        plt.grid(True)
        
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
        print(f"GARCH模型拟合出错: {e}")
        return None
    """
    使用GARCH模型分析比特币收益率与新闻情感的关系
    """
    # 构建包含新闻情感的GARCH-M模型
    data = merged_data.copy().dropna(subset=['log_return', 'sentiment_avg'])
    returns = data['log_return'] * 100
    
    # 准备外生变量（滞后一期）
    exog_vars = data[['sentiment_avg']].shift(1).dropna()
    
    # 对齐数据
    aligned_data = pd.concat([returns, exog_vars], axis=1).dropna()
    y = aligned_data['log_return']
    X = aligned_data[['sentiment_avg']]
    
    try:
        model = arch_model(y, vol='GARCH', p=1, q=1, mean='AR', lags=1, 
                         dist='skewt', x=X)
        res = model.fit(disp='off', update_freq=0)
        
        # 输出结果和绘图
        print(res.summary())
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(data.date, res.conditional_volatility, 'r-', linewidth=1.5)
        plt.title('比特币收益率条件波动率')
        plt.ylabel('波动率 (%)')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.scatter(data.sentiment_avg, returns, alpha=0.5)
        plt.xlabel('情感得分')
        plt.ylabel('收益率 (%)')
        plt.title('情感得分与收益率关系')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("analysis/garch_analysis.png")
        
        return res
        
    except Exception as e:
        print(f"GARCH模型拟合出错: {e}")
        return None

def visualize_results(df):
    # 1. 价格趋势图
    plt.figure(figsize=(12, 6))
    plt.plot(df.date, df.price, 'b-')
    plt.title('价格走势')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.grid(True)
    plt.savefig("analysis/price_trend.png")
    
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
    
    # 3. 情感与收益率的相关性热图
    plt.figure(figsize=(10, 8))
    corr_matrix = df[['price', 'news_count', 'sentiment_avg', 'log_return']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('变量相关性热图')
    plt.tight_layout()
    plt.savefig("analysis/correlation_heatmap.png")

if __name__ == "__main__":
    INPUT_PATH = "analysis/merged_data.csv"
    
    df = load_merged_data(INPUT_PATH)
    df = generate_features(df)
    
    print("\n执行GARCH分析...")
    garch_res = advanced_garch_analysis(df)
    
    print("\n生成可视化结果...")
    visualize_results(df)
    
    print("分析完成！")