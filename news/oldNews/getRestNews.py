import requests
from datetime import datetime

API_KEY = "432c2ee9205fc550e26f5e874fd2c708f898e1ed"  # 替换为 CryptoPanic 注册获取的密钥
BASE_URL = "https://cryptopanic.com/api/v1/posts/"
START_DATE = "2024-03-24"
END_DATE = "2024-12-04"

def fetch_cryptopanic_news():
    params = {
        "auth_token": API_KEY,
        "public": "true",  # 仅公开新闻
        "kind": "news",    # 筛选新闻类型
        "page": 1          # 分页参数
    }
    filtered_news = []
    
    while True:
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # 打印 API 响应内容以进行调试
            print(f"API 响应: {data}")  # 添加调试信息
            
            # 提取并过滤日期范围内的新闻
            for item in data.get("results", []):
                item_date = datetime.strptime(item["created_at"][:10], "%Y-%m-%d")
                start = datetime.strptime(START_DATE, "%Y-%m-%d")
                end = datetime.strptime(END_DATE, "%Y-%m-%d")
                
                if start <= item_date <= end:
                    filtered_news.append({
                        "title": item.get("title", "N/A"),
                        "date": item["created_at"],
                        "url": item.get("url", "N/A"),
                        "source": item.get("domain", "N/A")
                    })
                elif item_date < start:  # 超出时间范围则停止爬取
                    return filtered_news
            
            # 检测是否还有下一页
            if data.get("next") is None:
                break
            params["page"] += 1
            
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            break
    
    return filtered_news

# 调用函数并输出结果
news_data = fetch_cryptopanic_news()
if news_data:
    for idx, news in enumerate(news_data, 1):
        print(f"{idx}. {news['date']} | {news['title']}\n   链接: {news['url']}\n")
else:
    print("未找到符合条件的新闻。")