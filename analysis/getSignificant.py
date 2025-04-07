import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os
import matplotlib.dates as mdates  # 添加日期格式化模块

def load_data():
    """加载数据"""
    print("正在加载数据...")
    df = pd.read_csv('analysis/merged_DetailData.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def plot_price_trend(df):
    """绘制价格趋势图"""
    print("正在绘制价格趋势图...")
    plt.figure(figsize=(20, 10))
    
    # 设置日期格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 绘制价格趋势
    plt.plot(df['date'], df['price'], label='价格', color='blue', alpha=0.7, linewidth=2)
    
    # 标记重大事件
    for idx, row in df.iterrows():
        if abs(row['change_percent']) > 5:  # 涨跌幅超过5%的事件
            color = 'red' if row['change_percent'] < 0 else 'green'
            plt.scatter(row['date'], row['price'], 
                       color=color,
                       s=150, alpha=0.7,
                       edgecolors='black', linewidth=1)
            
            # 只显示日期和涨跌幅
            date_str = row['date'].strftime('%Y-%m-%d')
            plt.annotate(f"{date_str}\n{row['change_percent']:.1f}%", 
                        (row['date'], row['price']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        fontsize=8)
    # 将重大事件保存到CSV文件
    significant_events = []
    for idx, row in df.iterrows():
        if abs(row['change_percent']) > 5:
            significant_events.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'price': row['price'], 
                'change_percent': row['change_percent'],
                'sentiment_score': row['sentiment_score'],
                'title': row['title'],
                'negative': row['negative'],
                'neutral': row['neutral'], 
                'positive': row['positive'],
                'category': row['category']
            })
    
    # 创建DataFrame并保存
    events_df = pd.DataFrame(significant_events)
    events_df.to_csv('analysis/eventStudy/significant_events.csv', index=False, encoding='utf-8-sig')
    print(f"已将{len(significant_events)}个重大事件保存到CSV文件")
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    # 设置标题和标签
    plt.title('加密货币价格趋势与重大事件', fontsize=16, pad=20)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格', fontsize=12)
    
    # 设置网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('analysis/eventStudy/price_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
def main():
    # 创建输出目录
    os.makedirs('analysis/eventStudy', exist_ok=True)
    
    # 加载数据
    df = load_data()
    
    # 执行分析
    plot_price_trend(df)
    if __name__ == "__main__":
        main()