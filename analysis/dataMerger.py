# data_merger.py
import pandas as pd
import numpy as np
from datetime import timedelta

def load_and_clean_data(price_path, news_path):
    # 读取比特币价格数据
    price_df = pd.read_csv(price_path, parse_dates=['date'])
    price_df = price_df.sort_values('date').drop_duplicates('date')
    
    # 初始化news_df变量，确保它在任何情况下都有一个值
    news_df = None
    
    # 尝试读取新闻数据，如果出错则创建一个空DataFrame
    try:
        news_df = pd.read_csv(news_path, parse_dates=['date'])
        news_df = news_df.sort_values('date')
        
        # 新闻数据清洗
        news_df = news_df.drop_duplicates(['date', 'title'])
        news_df['event_type'] = news_df['event_type'].fillna('other')
        
        # 情感分数转换
        sentiment_map = {
            'positive': 1,
            'negative': -1,
            'neutral': 0
        }
        news_df['sentiment_score'] = news_df['sentiment_score'].map(sentiment_map)
    except Exception as e:
        print(f"读取新闻数据时出错: {e}")
        # 创建空的DataFrame，确保列名与预期一致
        news_df = pd.DataFrame(columns=['date', 'title', 'event_type', 'sentiment_score'])
    
    # 确保news_df已被定义
    if news_df is None:
        news_df = pd.DataFrame(columns=['date', 'title', 'event_type', 'sentiment_score'])
    
    return price_df, news_df

def merge_datasets(price_df, news_df):
    # 确保两个DataFrame的日期列都是datetime类型
    if news_df['date'].dtype != 'datetime64[ns]':
        # 尝试使用更复杂的格式或者使用errors='coerce'选项
        news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce', format='mixed')
    
    if price_df['date'].dtype != 'datetime64[ns]':
        price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce', format='mixed')
    
    # 处理可能的NaT值
    news_df = news_df.dropna(subset=['date'])
    price_df = price_df.dropna(subset=['date'])
    
    # 对日期进行规范化，只保留日期部分，去掉时间部分
    news_df['date'] = news_df['date'].dt.normalize()
    price_df['date'] = price_df['date'].dt.normalize()
    
    # 现在可以安全地比较日期
    date_range = pd.date_range(
        start=min(price_df['date'].min(), news_df['date'].min()),
        end=max(price_df['date'].max(), news_df['date'].max())
    )
    
    # 填充价格数据
    price_filled = price_df.set_index('date').reindex(date_range).reset_index()
    price_filled = price_filled.rename(columns={'index':'date'})
    price_filled['price'] = price_filled['price'].ffill()
    
    # 按日聚合新闻数据
    news_agg = news_df.groupby('date').agg({
        'title': 'count',
        'event_type': lambda x: list(x),
        'sentiment_score': 'mean'
    }).reset_index()
    news_agg.columns = ['date', 'news_count', 'event_types', 'sentiment_avg']
    
    # 生成事件类型哑变量
    all_events = set()
    for events in news_df['event_type'].dropna():
        if isinstance(events, str):
            all_events.add(events.lower())
    
    # 为每个事件类型创建列
    for event in all_events:
        news_agg[f'event_{event}'] = news_agg['event_types'].apply(
            lambda x: 1 if isinstance(x, list) and any(e.lower() == event for e in x) else 0
        )
    
    # 合并数据并填充
    merged = pd.merge(price_filled, news_agg, on='date', how='left')
    
    # 填充缺失值
    merged['news_count'] = merged['news_count'].fillna(0)
    merged['sentiment_avg'] = merged['sentiment_avg'].fillna(0)
    event_cols = [col for col in merged.columns if col.startswith('event_')]
    merged[event_cols] = merged[event_cols].fillna(0)
    
    # 删除event_types列
    if 'event_types' in merged.columns:
        merged = merged.drop('event_types', axis=1)
    
    return merged

if __name__ == "__main__":
    # 参数配置
    PRICE_PATH = "price/bitcoin_prices_20250324.csv"
    NEWS_PATH = "news/crypto_news_20250324.csv"
    OUTPUT_PATH = "analysis/merged_data.csv"
    
    # 执行流程
    price_df, news_df = load_and_clean_data(PRICE_PATH, NEWS_PATH)
    merged_data = merge_datasets(price_df, news_df)
    merged_data.to_csv(OUTPUT_PATH, index=False)
    print(f"数据已保存至: {OUTPUT_PATH}")