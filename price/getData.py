import pandas as pd
import requests
from datetime import datetime, timedelta

def get_bitcoin_price(days=365, vs_currency='usd'):
    """
    获取比特币历史价格数据
    
    参数:
    days (int): 获取多少天的数据，默认365天
    vs_currency (str): 计价货币,默认usd
    
    返回:
    pandas.DataFrame: 包含日期和价格的数据框
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': vs_currency,
        'days': str(days),
        'interval': 'daily'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        prices_data = response.json()['prices']
        
        df = pd.DataFrame(prices_data, columns=['timestamp', 'price'])
        # 转换时间戳为日期
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
        # 只保留日期和价格列，并按日期排序
        df = df[['date', 'price']].sort_values('date')
        # 去除重复的日期，保留每天的最后一个价格
        df = df.drop_duplicates(subset='date', keep='last')
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"发生错误: {e}")
        return pd.DataFrame()

# 使用示例
if __name__ == "__main__":
    try:
        # 获取一年的数据
        df = get_bitcoin_price(days=365)
        
        if not df.empty:
            # 保存到CSV文件
            filename = f'bitcoin_prices_{datetime.now().strftime("%Y%m%d")}.csv'
            df.to_csv(filename, index=False)
            
            print(f"\n数据已保存到文件: {filename}")
            print(f"总共获取了 {len(df)} 天的数据")
            print("\n前5天的数据示例:")
            print(df.head())
            print("\n最后5天的数据示例:")
            print(df.tail())
        else:
            print("未能获取数据")
    except Exception as e:
        print(f"发生错误: {e}") 