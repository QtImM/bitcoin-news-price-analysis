import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os
import matplotlib.dates as mdates  # 添加日期格式化模块

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    print("正在加载数据...")
    df = pd.read_csv('analysis\eventStudy\snews/News_filtered.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df



def analyze_price_changes(df):
    """分析涨跌幅分布"""
    print("正在分析涨跌幅分布...")
    
    # # 统计涨跌幅分布
    # plt.figure(figsize=(10, 6))
    # sns.histplot(df['change_percent'], bins=50, kde=True)
    # plt.title('价格涨跌幅分布')
    # plt.xlabel('涨跌幅 (%)')
    # plt.ylabel('频次')
    # plt.grid(True)
    # plt.savefig('analysis/eventStudy/price_change_distribution.png')
    # plt.close()
    
    # 统计极端涨跌幅事件
    extreme_events = df[abs(df['change_percent']) > 5].sort_values('change_percent')
    print("\n极端涨跌幅事件统计:")
    print(extreme_events[['date', 'change_percent']].to_string())
    
    return extreme_events

def analyze_sentiment(df):
    """分析情感分布和情感与价格的关联"""
    print("正在分析情感分布...")
    
    # 分析情感得分与涨跌幅的关系
    plt.figure(figsize=(12, 8))
    
    # 随机抽样减少数据点,但保持原始分布
    sampled_df = df.sample(n=min(1000, len(df)), random_state=42)
    
    # 创建散点图并添加趋势线
    # 使用抽样数据绘制散点图,但用全量数据拟合回归线
    sns.regplot(x='sentiment_score', y='change_percent', 
                data=sampled_df,  # 散点图用抽样数据
                scatter_kws={'alpha':0.5, 'color':'blue'},
                line_kws={'color': 'red'},
                fit_reg=False)  # 先不绘制回归线
                
    # 用全量数据拟合回归线
    sns.regplot(x='sentiment_score', y='change_percent',
                data=df,  # 回归线用全量数据
                scatter=False,  # 不显示散点
                line_kws={'color': 'red'})
    
    # 添加KDE密度等高线
    sns.kdeplot(data=df, x='sentiment_score', y='change_percent',
                levels=5, color='gray', alpha=0.3)
    
    plt.title('情感得分与涨跌幅关系分析', fontsize=14, pad=15)
    plt.xlabel('情感得分', fontsize=12)
    plt.ylabel('涨跌幅 (%)', fontsize=12)
    
    # 美化样式
    plt.grid(True, linestyle='--', alpha=0.7)
    sns.despine()  # 移除上边框和右边框
    
    # 保存图片
    plt.savefig('analysis/eventStudy/sentiment/sentiment_score_price_relation.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 计算情感得分与涨跌幅的相关系数
    correlation = df['sentiment_score'].corr(df['change_percent'])
    print(f"\n情感得分与涨跌幅的相关系数: {correlation:.3f}")
    
    # 比较正面和负面新闻的平均涨跌幅
    positive_news = df[df['positive'] == 1]['change_percent'].mean()
    negative_news = df[df['negative'] == 1]['change_percent'].mean()
    
    print("\n正面新闻和负面新闻的平均涨跌幅对比:")
    print(f"正面新闻平均涨跌幅: {positive_news:.2f}%")
    print(f"负面新闻平均涨跌幅: {negative_news:.2f}%")
    print(f"差异: {positive_news - negative_news:.2f}%")
    
    # # 分析正面、负面、中性情感的比例
    # sentiment_counts = pd.DataFrame({
    #     'negative': df['negative'].sum(),
    #     'neutral': df['neutral'].sum(),
    #     'positive': df['positive'].sum()
    # }, index=['count'])
    
    # plt.figure(figsize=(10, 6))
    # sentiment_counts.T.plot(kind='pie', autopct='%1.1f%%', subplots=True)
    # plt.title('情感分类分布')
    # plt.ylabel('')
    # plt.savefig('analysis/eventStudy/sentiment/sentiment_category_distribution.png')
    # plt.close()
    
    # print("\n情感分类统计:")
    # print(sentiment_counts)
    
    return 

def analyze_event_types(df):
    """分析事件类别和情感的组合分布"""
    print("正在分析事件类别和情感的组合分布...")
    
    # 筛选指定类别
    target_categories = ['market', 'technology', 'security', 'policy']
    df_filtered = df[df['category'].isin(target_categories)]
    
    # 计算每个类别下的情感分布
    category_sentiment = pd.DataFrame()
    for category in target_categories:
        category_data = df_filtered[df_filtered['category'] == category]
        sentiment_counts = {
            'positive': category_data['positive'].sum(),
            'neutral': category_data['neutral'].sum(), 
            'negative': category_data['negative'].sum()
        }
        category_sentiment[category] = pd.Series(sentiment_counts)
    
    # 绘制堆叠柱状图
    plt.figure(figsize=(12, 6))
    category_sentiment.plot(kind='bar', stacked=True)
    plt.title('事件类别与情感的组合分布')
    plt.xlabel('情感类型')
    plt.ylabel('频次')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(title='事件类别')
    plt.tight_layout()
    plt.savefig('analysis/eventStudy/category/category_sentiment_distribution.png')
    plt.close()
    
    # 计算每个类别和情感组合的平均涨跌幅
    category_sentiment_stats = []
    for category in target_categories:
        for sentiment in ['positive', 'neutral', 'negative']:
            mask = (df_filtered['category'] == category) & (df_filtered[sentiment] == 1)
            stats = df_filtered[mask]['change_percent'].agg(['mean', 'std', 'count']).to_dict()
            stats.update({'category': category, 'sentiment': sentiment})
            category_sentiment_stats.append(stats)
    
    stats_df = pd.DataFrame(category_sentiment_stats)
    print("\n事件类别与情感组合的涨跌幅统计:")
    print(stats_df)
    
    return stats_df

def main():
    # 创建输出目录
    os.makedirs('analysis/eventStudy', exist_ok=True)
    
    # 加载数据
    df = load_data()
    
    # 执行分析
   
    # extreme_events = analyze_price_changes(df)
    sentiment_stats = analyze_sentiment(df)
    # category_stats = analyze_event_types(df)
    
    # 保存分析结果
    # sentiment_stats.to_csv('analysis/eventStudy/sentiment/sentiment_stats.csv')
    # category_stats.to_csv('analysis/eventStudy/category/category_stats.csv')
    
    print("\n分析完成！结果已保存至eventStudy目录")

if __name__ == "__main__":
    main()
