#CoinDesk、Bitcoin Magzine、CrytoCoin News、Coin Telegraph、News Bitcoin、
# Bitcoin Subreddit、Crypto-Currency Sub Reddit 的 Cryto 货币新闻和讨论集合。
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import time
import csv
import os
from datetime import datetime, timedelta

# 定义起始和结束日期
start_date = datetime(2024, 3, 24)
end_date = datetime(2025, 3, 24)

# 创建结果列表
news_data = []

# 定义要爬取的网站列表
websites = [
    {"name": "CoinDesk", "url": "https://www.coindesk.com/"},
    {"name": "Bitcoin Magazine", "url": "https://bitcoinmagazine.com/"},
    {"name": "CryptoCoin News", "url": "https://www.ccn.com/"},
    {"name": "Coin Telegraph", "url": "https://cointelegraph.com/"},
    {"name": "News Bitcoin", "url": "https://news.bitcoin.com/"},
    {"name": "Bitcoin Subreddit", "url": "https://www.reddit.com/r/Bitcoin/"},
    {"name": "Crypto-Currency Subreddit", "url": "https://www.reddit.com/r/CryptoCurrency/"}
]

# 创建用户代理头，避免被网站屏蔽
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 为每个网站爬取数据
for website in websites:
    print(f"正在爬取 {website['name']} 的数据...")
    
    try:
        # 获取网页内容
        response = requests.get(website['url'], headers=headers)
        response.raise_for_status()
        
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 根据不同网站提取新闻
        # 注意：实际实现需要针对每个网站的HTML结构进行定制
        articles = []
        
        if "coindesk" in website['url']:
            articles = soup.find_all('article') or soup.find_all('div', class_='article')
        elif "reddit" in website['url']:
            articles = soup.find_all('div', class_='Post') or soup.find_all('div', class_='thing')
        else:
            # 通用选择器，可能需要针对特定网站调整
            articles = soup.find_all('article') or soup.find_all('div', class_='post')
        
        # 提取每篇文章的信息
        for article in articles[:10]:  # 限制每个网站最多10篇文章
            title_elem = article.find('h2') or article.find('h3') or article.find('h1')
            title = title_elem.text.strip() if title_elem else "无标题"
            
            # 尝试获取日期（实际实现需要针对每个网站调整）
            date_elem = article.find('time') or article.find('span', class_='date')
            date_str = date_elem.text.strip() if date_elem else datetime.now().strftime("%Y-%m-%d")
            
            # 尝试获取链接
            link_elem = article.find('a')
            link = link_elem.get('href') if link_elem else ""
            if link and not link.startswith('http'):
                link = website['url'].rstrip('/') + link
            
            # 添加到结果列表
            news_data.append({
                "网站": website['name'],
                "标题": title,
                "日期": date_str,
                "链接": link
            })
        
    except Exception as e:
        print(f"爬取 {website['name']} 时出错: {str(e)}")
    
    # 添加延迟，避免请求过于频繁
    time.sleep(2)

# 将数据保存到CSV文件
output_file = "crypto_news_data.csv"
try:
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["网站", "标题", "日期", "链接"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in news_data:
            writer.writerow(item)
    
    print(f"数据已成功保存到 {output_file}")
    print(f"共收集了 {len(news_data)} 条新闻")
except Exception as e:
    print(f"保存数据时出错: {str(e)}")
