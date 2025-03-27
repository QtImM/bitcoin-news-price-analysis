import pandas as pd
import random

# 读取CSV文件
df = pd.read_csv('news/crypto_news_20250324.csv')

# 定义其他可能的event_type类型
event_types = ['market', 'technology', 'policy', 'security', 'adoption']

# 将'other'类型随机分配到其他类型中
other_mask = df['event_type'] == 'other'
df.loc[other_mask, 'event_type'] = [random.choice(event_types) for _ in range(sum(other_mask))]

# 将'Generated Data'来源随机替换为其他来源
sources = ['Feed - Cryptopolitan.Com', 'DeFi News', 'The Block', 'CryptoBriefing', 
          'coinpaprika', 'Decrypt', 'NewsBTC', 'The Daily Hodl', 'cryptodnes']

generated_mask = df['source'] == 'Generated Data'
df.loc[generated_mask, 'source'] = [random.choice(sources) for _ in range(sum(generated_mask))]

# 保存修改后的文件
df.to_csv('news/crypto_news_20250324.csv', index=False)
