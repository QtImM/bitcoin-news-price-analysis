# data_merger.py
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
        required_columns = ['date', 'sentiment', 'source', 'text', 'title', 'url']
        if not all(col in news_df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in news_df.columns]
            print(f"新闻数据缺少必要的列: {missing_cols}")
            return None, None
        
        # 解析 sentiment 列
        news_df['sentiment'] = news_df['sentiment'].apply(ast.literal_eval)  # 将字符串转换为字典
        news_df['sentiment_score'] = news_df['sentiment'].apply(lambda x: x['polarity'])  # 提取极性分数
        news_df['event_type'] = news_df['sentiment'].apply(lambda x: x['class'])  # 提取事件类型
        
        # 清理 sentiment_score
        news_df['sentiment_score'] = pd.to_numeric(news_df['sentiment_score'], errors='coerce')
        news_df['sentiment_score'] = news_df['sentiment_score'].fillna(0)
        
    except Exception as e:
        print(f"读取新闻数据时出错: {e}")
        return None, None
    
    return price_df, news_df

def merge_datasets(price_df, news_df):
    """
    合并数据集，处理缺失数据,不包含event相关列
    """
    if price_df is None or news_df is None:
        print("价格数据或新闻数据为空，无法合并")
        return None
    
    try:
        # 确保日期格式统一
        price_df['date'] = pd.to_datetime(price_df['date']).dt.normalize()
        news_df['date'] = pd.to_datetime(news_df['date']).dt.normalize()
        
        # 创建完整的日期范围
        date_range = pd.date_range(
            start=min(price_df['date'].min(), news_df['date'].min()),
            end=max(price_df['date'].max(), news_df['date'].max())
        )
        
        # 填充价格数据
        price_filled = price_df.set_index('date').reindex(date_range).reset_index()
        price_filled = price_filled.rename(columns={'index': 'date'})
        price_filled['price'] = price_filled['price'].ffill().bfill()
        
        # 按日聚合新闻数据,只保留需要的列
        news_agg = news_df.groupby('date').agg({
            'title': 'count',
            'sentiment_score': 'mean'
        }).reset_index()
        
        news_agg.columns = ['date', 'news_count', 'sentiment_avg']
        
        # 合并数据
        merged = pd.merge(price_filled, news_agg, on='date', how='left')
        
        # 填充缺失值
        merged['news_count'] = merged['news_count'].fillna(0)
        merged['sentiment_avg'] = merged['sentiment_avg'].fillna(0)
        
        # 只保留需要的列
        columns_to_keep = ['date', 'price', 'news_count', 'sentiment_avg']
        merged = merged[columns_to_keep]
        
        return merged
        
    except Exception as e:
        print(f"合并数据时出错: {e}")
        return None
    """
    合并数据集，处理缺失数据
    """
    if price_df is None or news_df is None:
        print("价格数据或新闻数据为空，无法合并")
        return None
    
    try:
        # 确保日期格式统一
        price_df['date'] = pd.to_datetime(price_df['date']).dt.normalize()
        news_df['date'] = pd.to_datetime(news_df['date']).dt.normalize()
        
        # 创建完整的日期范围
        date_range = pd.date_range(
            start=min(price_df['date'].min(), news_df['date'].min()),
            end=max(price_df['date'].max(), news_df['date'].max())
        )
        
        # 填充价格数据
        price_filled = price_df.set_index('date').reindex(date_range).reset_index()
        price_filled = price_filled.rename(columns={'index': 'date'})
        price_filled['price'] = price_filled['price'].ffill().bfill()  # 前向和后向填充
        
        # 按日聚合新闻数据
        news_agg = news_df.groupby('date').agg({
            'title': 'count',
            'event_type': lambda x: list(x),
            'sentiment_score': 'mean',
            'source': lambda x: list(x)
        }).reset_index()
        
        news_agg.columns = ['date', 'news_count', 'event_types', 'sentiment_avg', 'sources']
        
        # 生成事件类型指标
        event_types = ['market', 'policy', 'technology', 'security', 'adoption']
        for event in event_types:
            news_agg[f'event_{event}'] = news_agg['event_types'].apply(
                lambda x: 1 if isinstance(x, list) and any(e.lower() == event for e in x) else 0
            )
        
        # 合并数据
        merged = pd.merge(price_filled, news_agg, on='date', how='left')
        
        # 填充缺失值
        merged['news_count'] = merged['news_count'].fillna(0)
        merged['sentiment_avg'] = merged['sentiment_avg'].fillna(0)
        
        # 填充事件类型列
        event_cols = [col for col in merged.columns if col.startswith('event_')]
        merged[event_cols] = merged[event_cols].fillna(0)
        
        # 删除不需要的列
        columns_to_drop = ['event_types', 'sources']
        merged = merged.drop(columns=[col for col in columns_to_drop if col in merged.columns])
        
        return merged
        
    except Exception as e:
        print(f"合并数据时出错: {e}")
        return None

if __name__ == "__main__":
    try:
        # 参数配置
        PRICE_PATH = "price\mixture\cmc10_weighted_index.csv"
        NEWS_PATH = "news\cryptonewsResearch.csv"
        OUTPUT_PATH = "analysis/merged_data.csv"
        
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
                print("\n数据统计:")
                print(merged_data.describe())
            else:
                print("数据合并失败")
        else:
            print("数据加载失败")
            
    except Exception as e:
        print(f"程序执行出错: {e}")