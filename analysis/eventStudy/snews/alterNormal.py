import pandas as pd

# 读取原始数据
df = pd.read_csv('analysis\merged_DetailData.csv')

# 过滤数据
filtered_df = df[
    # 不保留中性情感(sentiment_score=0)的数据
    
    # 保留价格上涨且情感得分为正的数据 
    ((df['change_percent'] > 0) & (df['sentiment_score'] > -0.2 ))&(df['sentiment_score'] != 0) | 
    # 保留价格下跌且情感得分为负的数据
    ((df['change_percent'] < 0) & (df['sentiment_score'] < 0.2))&(df['sentiment_score'] != 0) 
]

# 保存到新文件
filtered_df.to_csv('analysis\eventStudy\snews/news_filtered.csv', index=False)

print(f'原始数据行数: {len(df)}')
print(f'过滤后数据行数: {len(filtered_df)}')
