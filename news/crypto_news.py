import pandas as pd
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
import time

def get_cryptopanic_news(start_date, end_date, api_key):
    """
    获取 Cryptopanic 新闻数据
    """
    base_url = "https://cryptopanic.com/api/v1/posts/"
    news_data = []
    page = 1
    max_retries = 3
    
    try:
        while True:
            # 添加重试机制
            for retry in range(max_retries):
                try:
                    params = {
                        'auth_token': api_key,
                        'currencies': 'BTC',
                        'public': 'true',
                        'page': page,
                        'filter': 'important'  # 只获取重要新闻
                    }
                    
                    response = requests.get(base_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    break  # 如果成功就跳出重试循环
                except requests.exceptions.RequestException as e:
                    if retry == max_retries - 1:  # 最后一次重试
                        print(f"请求失败: {e}")
                        return pd.DataFrame(columns=['date', 'title', 'event_type', 'sentiment_score', 'source'])
                    time.sleep(2)  # 重试前等待
            
            # 检查是否有结果
            if not data.get('results'):
                print("没有更多数据")
                break
            
            # 处理每条新闻
            for article in data['results']:
                try:
                    date = datetime.strptime(article['published_at'][:10], '%Y-%m-%d')
                    
                    # 检查日期范围
                    if date < datetime.strptime(start_date, '%Y-%m-%d'):
                        print(f"已达到起始日期 {start_date}")
                        return pd.DataFrame(news_data)
                    
                    if date > datetime.strptime(end_date, '%Y-%m-%d'):
                        continue  # 跳过超出结束日期的新闻
                    
                    title = article['title']
                    
                    # 事件类型分类
                    event_type = classify_event_type(title.lower())
                    
                    # 获取更准确的情感分数
                    sentiment = get_sentiment_score(title)
                    
                    news_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'title': title,
                        'event_type': event_type,
                        'sentiment_score': sentiment,
                        'source': article.get('source', {}).get('title', 'Unknown')
                    })
                    
                except Exception as e:
                    print(f"处理新闻项时出错: {e}")
                    continue
            
            print(f"已处理第 {page} 页，当前获取 {len(news_data)} 条新闻")
            
            # 检查是否有下一页
            if not data.get('next'):
                print("已到达最后一页")
                break
                
            page += 1
            time.sleep(1.5)  # 增加延迟，避免触发API限制
        
        # 直接返回获取到的新闻数据
        return pd.DataFrame(news_data) if news_data else pd.DataFrame(columns=['date', 'title', 'event_type', 'sentiment_score', 'source'])
            
    except Exception as e:
        print(f"发生未预期的错误: {e}")
        return pd.DataFrame(columns=['date', 'title', 'event_type', 'sentiment_score', 'source'])

def classify_event_type(title):
    """
    基于标题关键词分类事件类型
    """
    keywords = {
        'policy': ['etf', 'sec', 'regulation', 'ban', 'law', 'legal', 'government', 'regulatory'],
        'technology': ['upgrade', 'protocol', 'network', 'fork', 'development', 'launch'],
        'market': ['price', 'market', 'trading', 'bull', 'bear', 'rally', 'crash', 'surge', 'plunge'],
        'security': ['hack', 'security', 'breach', 'stolen', 'scam', 'fraud'],
        'adoption': ['adopt', 'accept', 'partnership', 'integration', 'institutional']
    }
    
    for event_type, words in keywords.items():
        if any(word in title for word in words):
            return event_type
    return 'market'  # 默认分类为市场相关

def get_sentiment_score(title):
    """
    获取更准确的情感分数
    """
    # 基础分数
    base_score = TextBlob(title).sentiment.polarity
    
    # 金融市场特定词典
    positive_terms = ['bullish', 'surge', 'rally', 'gain', 'breakthrough', 'adopt', 'approve']
    negative_terms = ['bearish', 'crash', 'decline', 'hack', 'ban', 'risk', 'fraud']
    
    # 调整分数
    title_lower = title.lower()
    for term in positive_terms:
        if term in title_lower:
            base_score += 0.1
    for term in negative_terms:
        if term in title_lower:
            base_score -= 0.1
            
    # 确保分数在 -1 到 1 之间
    return max(min(base_score, 1.0), -1.0)

if __name__ == "__main__":
    try:
        # 设置API密钥
        api_key = "432c2ee9205fc550e26f5e874fd2c708f898e1ed"  # 替换为您的API密钥
        
        # 设置日期范围
        start_date = "2024-03-25"
        end_date = "2025-03-25"
        print(f"开始获取从 {start_date} 到 {end_date} 的新闻数据...")
        
        df = get_cryptopanic_news(start_date, end_date, api_key)
        
        if not df.empty:
            # 保存到CSV文件
            filename = f'news/crypto_news_{datetime.now().strftime("%Y%m%d")}.csv'
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            print(f"\n数据已保存到文件: {filename}")
            print(f"总共获取了 {len(df)} 条新闻")
            print("\n数据示例:")
            print(df.head())
            
            # 显示统计信息
            print("\n事件类型统计:")
            print(df['event_type'].value_counts())
            print("\n情感分布:")
            print(pd.cut(df['sentiment_score'], bins=5).value_counts())
        else:
            print("未能获取任何数据")
            
    except Exception as e:
        print(f"程序执行出错: {e}") 