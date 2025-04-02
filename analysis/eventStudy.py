import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

# ==================== 全局配置 ====================
EVENT_TYPES = ['negative', 'neutral', 'positive']  # 需分析的事件类型
EVENT_WINDOW = (-3, 5)     # 事件窗口日
ESTIMATION_WINDOW = 100    # 估计窗口长度
SIGNIFICANCE_LEVEL = 0.05  # 显著性水平
OUTPUT_DIR = 'analysis/event_analysis_results'  # 结果保存目录

# ==================== 数据准备 ====================
df = pd.read_csv("analysis/merged_DetailData.csv", parse_dates=['date'])
df = df.sort_values('date').drop_duplicates('date', keep='first')
df['return'] = np.log(df['price'] / df['price'].shift(1))

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 分析函数 ====================
def analyze_event(event_type):
    """执行单个事件类型分析"""
    # 识别事件日期
    event_dates = df[df[event_type] == 1]['date'].unique()
    if len(event_dates) == 0:
        print(f"警告：未找到{event_type}事件日期")
        return None
    
    # 存储结果
    ar_list = []
    car_list = []
    
    # 逐个事件分析
    for date in event_dates:
        try:
            idx = df.index[df['date'] == date][0]
            start = idx - ESTIMATION_WINDOW + EVENT_WINDOW[0]
            end = idx + EVENT_WINDOW[1]
            window = df.iloc[start:end+1]['return'].values
            
            # 计算正常收益率
            normal_ret = window[:ESTIMATION_WINDOW].mean()
            
            # 计算异常收益率
            ar = window[ESTIMATION_WINDOW:] - normal_ret
            ar_list.append(ar)
            car_list.append(ar.sum())
            
        except IndexError:
            continue
    
    # 结果汇总
    results = {
        'event_type': event_type,
        'event_count': len(ar_list),
        'mean_ar': np.mean(ar_list, axis=0),
        'mean_car': np.mean(car_list),
        't_stat': stats.ttest_1samp(car_list, 0).statistic,
        'p_value': stats.ttest_1samp(car_list, 0).pvalue
    }
    return results

# ==================== 批量分析 ====================
all_results = []
for etype in EVENT_TYPES:
    result = analyze_event(etype)
    if result:
        all_results.append(result)

# ==================== 可视化 ====================
# 创建可视化画布
plt.figure(figsize=(15, 8))

# 绘制累计异常收益率
for i, res in enumerate(all_results):
    days = range(EVENT_WINDOW[0], EVENT_WINDOW[1]+1)
    car = res['mean_ar'].cumsum()
    
    # 标注显著性
    sig_marker = '*' if res['p_value'] < SIGNIFICANCE_LEVEL else ''
    label = f"{res['event_type']}{sig_marker} (n={res['event_count']})"
    
    plt.plot(days, car, 
             marker='o' if sig_marker else 'x',
             linestyle='--' if sig_marker else ':',
             label=label)

# 图表装饰
plt.axvline(0, color='red', linestyle='--', label='事件日')
plt.axhline(0, color='black', linestyle='-')
plt.title(f"不同事件类型累计异常收益率对比 (α={SIGNIFICANCE_LEVEL})")
plt.xlabel("事件窗口日")
plt.ylabel("累计异常收益率")
plt.legend()
plt.grid(True)

# 保存可视化结果
plt.savefig(f'{OUTPUT_DIR}/CAR_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== 结果保存 ====================
# 保存统计结果
result_df = pd.DataFrame(all_results)
result_df['significance'] = result_df['p_value'].apply(
    lambda x: '显著' if x < SIGNIFICANCE_LEVEL else '不显著')
result_df.to_csv(f'{OUTPUT_DIR}/statistical_results.csv', index=False)

# 保存原始数据
pd.DataFrame(all_results).to_json(f'{OUTPUT_DIR}/raw_data.json')

print(f"分析完成！结果已保存至 {OUTPUT_DIR} 目录")