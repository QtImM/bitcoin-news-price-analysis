import requests
from datetime import datetime, timedelta
import pandas as pd
import time
from dotenv import load_dotenv
import os

# 加载.env文件
load_dotenv()

def get_cmc100_historical_data(api_key):
    url = "https://pro-api.coinmarketcap.com/v3/index/cmc100-historical"
    
    # 设置时间范围 - 获取过去一年的数据
    start_date = datetime(2021, 10, 12)
    end_date = datetime(2023, 12, 19)
    
    parameters = {
        'time_start': start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        'time_end': end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        'interval': 'daily',
        'count': '10'
    }
    
    headers = {
        'X-CMC_PRO_API_KEY': api_key,
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers, params=parameters)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'data' in data:
                # 创建DataFrame并只保留需要的列
                df = pd.DataFrame(data['data'])
                
                # 只保留date和price列
                df_simplified = pd.DataFrame({
                    'date': pd.to_datetime(df['timestamp']),
                    'price': df['value']
                })
                
                # 保存到CSV文件
                df_simplified.to_csv('cmc100_historical_data.csv', index=False)
                print("数据已成功保存到cmc100_historical_data.csv")
                
                return df_simplified
            else:
                print("响应中没有找到数据")
                return None
                
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

def get_cmc100_historical_data_paged(api_key):
    url = "https://pro-api.coinmarketcap.com/v3/index/cmc100-historical"
    
    # 创建空的DataFrame来存储所有数据
    all_data = pd.DataFrame(columns=['date', 'price'])
    
    # 设置固定的时间范围
    start_date = datetime(2021, 10, 12)
    end_date = datetime(2023, 12, 19)
    current_date = start_date
    
    while current_date <= end_date:
        parameters = {
            'time_start': current_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'time_end': current_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'interval': 'daily',
            'count': '1'  # 每次只获取一个数据点
        }
        
        headers = {
            'X-CMC_PRO_API_KEY': api_key,
            'Accept': 'application/json'
        }
        
        try:
            response = requests.get(url, headers=headers, params=parameters)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    # 获取价格数据
                    price_data = data['data'][0]
                    
                    # 创建新的数据行
                    new_row = pd.DataFrame({
                        'date': [current_date],
                        'price': [price_data.get('value', None)]
                    })
                    
                    # 添加到总数据中
                    all_data = pd.concat([all_data, new_row], ignore_index=True)
                    print(f"成功获取 {current_date.date()} 的数据，价格: {price_data.get('value', 'N/A')}")
                else:
                    print(f"未找到 {current_date.date()} 的数据")
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
            
            # 更新到下一天
            current_date += timedelta(days=1)
            
            # 添加延时以避免触发API限制
            time.sleep(1)
            
        except Exception as e:
            print(f"获取 {current_date.date()} 数据时发生错误: {str(e)}")
            current_date += timedelta(days=1)
            continue
    
    # 保存所有数据到CSV
    if not all_data.empty:
        # 确保数据按日期排序
        all_data = all_data.sort_values('date')
        
        # 保存数据
        filename = 'cmc100_historical_data_2024-2025.csv'
        all_data.to_csv(filename, index=False)
        print(f"\n数据已保存到 {filename}")
        print(f"总共获取了 {len(all_data)} 条数据")
        print("\n数据示例:")
        print(all_data.head())
    else:
        print("未获取到任何数据")
    
    return all_data

def main():
    # 从.env文件获取API密钥
    api_key = os.getenv('COINMARKETCAP_API_KEY')
    
    if not api_key:
        print("错误：未找到API密钥，请确保.env文件中设置了COINMARKETCAP_API_KEY")
        return
    
    print("开始获取coins数据...")
    print("时间范围: 2021-10-12 到 2023-12-19")
    
    # 获取数据
    df = get_cmc100_historical_data_paged(api_key)
    
    if df is not None and not df.empty:
        print("\n数据获取完成!")
        print("\n数据概览:")
        print(df.info())
        
        # 显示数据统计信息
        print("\n价格统计信息:")
        print(df['price'].describe())
        
        # 显示数据完整性信息
        total_days = (datetime(2023, 12, 19) - datetime(2021, 10, 12)).days + 1
        coverage = (len(df) / total_days) * 100
        print(f"\n数据完整性: {coverage:.2f}% ({len(df)}/{total_days} 天)")
    else:
        print("获取数据失败，请检查API密钥和网络连接")

if __name__ == "__main__":
    main()
