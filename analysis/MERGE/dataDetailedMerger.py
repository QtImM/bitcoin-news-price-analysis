import pandas as pd
import numpy as np
from datetime import timedelta
import ast  # 用于解析字符串格式的字典

def load_and_clean_data(price_path, news_path):
    """
    加载并清理数据，适应新的数据格式
    """
    # 读取比特币价格数据
    try:
        price_df = pd.read_csv(price_path)
        price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
        price_df = price_df.sort_values('date').drop_duplicates('date')
    except Exception as e:
        print(f"读取价格数据时出错: {e}")
        return None, None

    # 读取新闻数据
    try:
        news_df = pd.read_csv(news_path)
        # 转换日期格式
        news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
        
        # 基本清理
        news_df = news_df.sort_values('date')
        news_df = news_df.drop_duplicates(['date', 'title'])
        
        # 确保必要的列存在
        required_columns = ['date', 'sentiment', 'title']
        if not all(col in news_df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in news_df.columns]
            print(f"新闻数据缺少必要的列: {missing_cols}")
            return None, None
        
        # 解析 sentiment 列
        news_df['sentiment'] = news_df['sentiment'].apply(ast.literal_eval)  # 将字符串转换为字典
        news_df['negative'] = news_df['sentiment'].apply(lambda x: 1 if x['class'] == 'negative' else 0)
        news_df['neutral'] = news_df['sentiment'].apply(lambda x: 1 if x['class'] == 'neutral' else 0)
        news_df['positive'] = news_df['sentiment'].apply(lambda x: 1 if x['class'] == 'positive' else 0)
        news_df['sentiment_score'] = news_df['sentiment'].apply(lambda x: x['polarity'])  # 提取极性分数
        
        # 清理 sentiment_score
        news_df['sentiment_score'] = pd.to_numeric(news_df['sentiment_score'], errors='coerce')
        news_df['sentiment_score'] = news_df['sentiment_score'].fillna(0)
        
    except Exception as e:
        print(f"读取新闻数据时出错: {e}")
        return None, None
    
    return price_df, news_df

def merge_datasets(price_df, news_df):
    """
    合并数据集，保留 date, price, change_percent, sentiment_score, title 和情感指标
    """
    if price_df is None or news_df is None:
        print("价格数据或新闻数据为空，无法合并")
        return None
    
    try:
        # 确保日期格式统一
        price_df['date'] = pd.to_datetime(price_df['date']).dt.normalize()
        news_df['date'] = pd.to_datetime(news_df['date']).dt.normalize()
        
        # 合并数据
        merged = pd.merge(news_df, price_df, on='date', how='left')
        
        # 填充缺失值
        merged['price'] = merged['price'].fillna(method='ffill')
        merged['change_percent'] = merged['change_percent'].fillna(0)  # 填充 change_percent
        
        # 只保留需要的列
        columns_to_keep = ['date', 'price', 'change_percent', 'sentiment_score', 'title', 'negative', 'neutral', 'positive','category']
        merged = merged[columns_to_keep]
        
        return merged
        
    except Exception as e:
        print(f"合并数据时出错: {e}")
        return None

if __name__ == "__main__":
    try:
        # 参数配置
        PRICE_PATH = "price/mixture/cmc10_weighted_index.csv"
        NEWS_PATH = "news\classified_data.csv"
        OUTPUT_PATH = "analysis/merged_DetailData.csv"
        
        # 执行流程
        print("开始加载数据...")
        price_df, news_df = load_and_clean_data(PRICE_PATH, NEWS_PATH)
        
        if price_df is not None and news_df is not None:
            print("开始合并数据...")
            merged_data = merge_datasets(price_df, news_df)
            
            if merged_data is not None:
                # 保存结果
                merged_data.to_csv(OUTPUT_PATH, index=False)
                print(f"数据已成功保存至: {OUTPUT_PATH}")
                print(f"数据形状: {merged_data.shape}")
                print("\n数据预览:")
                print(merged_data.head())
            else:
                print("数据合并失败")
        else:
            print("数据加载失败")
            
    except Exception as e:
        print(f"程序执行出错: {e}")