import pandas as pd
import requests
from dotenv import load_dotenv
import os

# 加载.env文件中的环境变量
load_dotenv()

def get_bitcoin_price(start_date, end_date=None, convert='USD'):
    """
    获取比特币历史价格数据
    
    参数:
    start_date (str): 开始日期,格式'YYYY-MM-DD'
    end_date (str): 结束日期,格式'YYYY-MM-DD',默认为None(获取到最新数据)
    convert (str): 转换的货币类型,默认USD
    
    返回:
    pandas.DataFrame: 包含日期和价格的数据框
    """
    # 获取API密钥
    api_key = os.getenv('COINMARKETCAP_API_KEY')
    if not api_key:
        raise ValueError("请在.env文件中设置COINMARKETCAP_API_KEY")

    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    headers = {
        'X-CMC_PRO_API_KEY': api_key,
        'Accept': 'application/json'
    }
    
    params = {
        'symbol': 'BTC',  # 使用符号而不是ID
        'convert': convert,
        'time_start': start_date
    }
    if end_date:
        params['time_end'] = end_date
        
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # 检查响应状态
        
        # 打印响应内容以检查实际的数据结构
        print("API响应:", response.json())
        
        data = response.json()
        quotes = data.get('data', {}).get('quotes', [])
        
        if not quotes:
            print("未找到数据")
            return pd.DataFrame()
            
        df = pd.DataFrame([{
            "date": x["time_open"][:10], 
            "price": x["quote"][convert]["close"]
        } for x in quotes])
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return pd.DataFrame()
    except KeyError as e:
        print(f"数据解析错误: {e}")
        print("完整响应:", response.text)  # 打印完整响应以便调试
        return pd.DataFrame()

# 使用示例
if __name__ == "__main__":
    try:
        df = get_bitcoin_price('2023-01-01')
        if not df.empty:
            print("\n比特币价格数据:")
            print(df.head())
        else:
            print("未能获取数据")
    except ValueError as e:
        print(e)