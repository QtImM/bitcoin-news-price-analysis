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
os.makedirs("analysis/results", exist_ok=True)

def load_merged_data(file_path):
    """加载并预处理数据"""
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.dropna(subset=['price'])
    return df

def generate_features(df):
    """生成特征"""
    # 基础特征
    df['log_return'] = np.log(df['price']) - np.log(df['price'].shift(1))
    df['volatility_7d'] = df['log_return'].rolling(7).std() * np.sqrt(7)
    
    # 滞后特征
    for lag in [1, 3, 5, 7]:
        df[f'sentiment_lag{lag}'] = df['sentiment_score'].shift(lag)
        df[f'return_lag{lag}'] = df['log_return'].shift(lag)
    
    # 技术指标 - 使用更短期的均线
    df['MA3'] = df['price'].rolling(window=3).mean()
    df['MA5'] = df['price'].rolling(window=5).mean()
    df['RSI'] = calculate_rsi(df['price'])
    
    return df.dropna()

def calculate_rsi(prices, periods=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_performance(df):
    """分析模型性能"""
    # 计算方向准确率
    df['direction'] = np.sign(df['log_return'])
    df['pred_direction'] = np.where(df['sentiment_score'] > 0, 1, -1)
    direction_accuracy = (df['direction'] == df['pred_direction']).mean() * 100
    
    # 计算相关系数
    correlation = df['sentiment_score'].corr(df['log_return'])
    
    # 计算MAE
    mae = np.abs(df['log_return'] - df['sentiment_score']).mean()
    
    return {
        '样本数量': len(df),
        'MAE': round(mae, 4),
        '相关系数': round(correlation, 4),
        '方向准确率': round(direction_accuracy, 2)
    }

def advanced_garch_analysis(df):
    """使用GARCH模型分析收益率与情感的关系"""
    try:
        # 准备数据
        returns = df['log_return'] * 100
        X = df[['sentiment_score']]
        
        # 拟合GARCH模型
        model = arch_model(returns, vol='GARCH', p=1, q=1, mean='AR', lags=1, 
                         dist='skewt', x=X)
        res = model.fit(disp='off', update_freq=0)
        
        # 输出结果
        print("\nGARCH模型分析结果:")
        print(res.summary())
        
        # 保存波动率图
        plt.figure(figsize=(12, 6))
        plt.plot(df.date, res.conditional_volatility, 'r-', linewidth=1.5)
        plt.title('条件波动率')
        plt.ylabel('波动率 (%)')
        plt.grid(True)
        plt.savefig("analysis/results/volatility.png")
        plt.close()
        
        return res
        
    except Exception as e:
        print(f"GARCH模型拟合出错: {str(e)}")
        return None

def visualize_results(df):
    """生成可视化结果"""
    try:
        # 1. 价格和情感走势对比
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        ax1.plot(df.date, df.price, 'b-', label='价格')
        ax1.set_title('价格走势')
        ax1.set_ylabel('价格')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(df.date, df.sentiment_score, 'g-', label='情感得分')
        ax2.fill_between(df.date, 0, df.sentiment_score, 
                        where=df.sentiment_score >= 0, 
                        color='green', alpha=0.3)
        ax2.fill_between(df.date, 0, df.sentiment_score, 
                        where=df.sentiment_score < 0, 
                        color='red', alpha=0.3)
        ax2.set_title('情感得分走势')
        ax2.set_ylabel('情感得分')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("analysis/results/price_sentiment_trend.png")
        plt.close()
        
        # 2. 相关性热图
        plt.figure(figsize=(10, 8))
        features = ['price', 'log_return', 'volatility_7d', 'sentiment_score', 
                   'sentiment_lag1', 'sentiment_lag3', 'sentiment_lag5']
        corr_matrix = df[features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('特征相关性热图')
        plt.tight_layout()
        plt.savefig("analysis/results/correlation_heatmap.png")
        plt.close()
        
        # 3. 收益率分布
        plt.figure(figsize=(12, 6))
        sns.histplot(df.log_return, kde=True)
        plt.title('收益率分布')
        plt.xlabel('对数收益率')
        plt.ylabel('频次')
        plt.savefig("analysis/results/return_distribution.png")
        plt.close()
        
    except Exception as e:
        print(f"可视化生成出错: {str(e)}")

if __name__ == "__main__":
    try:
        # 设置输入文件路径
        INPUT_PATH = "analysis/merged_DetailData.csv"
        
        print("正在读取数据...")
        df = load_merged_data(INPUT_PATH)
        df = generate_features(df)
        
        # 划分训练集和测试集
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        print("\n训练集分析结果:")
        train_metrics = analyze_performance(train_df)
        for metric, value in train_metrics.items():
            print(f"- {metric}: {value}")
            
        print("\n测试集分析结果:")
        test_metrics = analyze_performance(test_df)
        for metric, value in test_metrics.items():
            print(f"- {metric}: {value}")
        
        print("\n执行GARCH分析...")
        garch_res = advanced_garch_analysis(df)
        
        print("\n生成可视化结果...")
        visualize_results(df)
        
        print("\n分析完成！结果已保存到 analysis/results/ 目录")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")