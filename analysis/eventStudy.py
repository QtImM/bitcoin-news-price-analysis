import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# ==================== 全局配置 ====================
EVENT_TYPES = ['negative', 'neutral', 'positive']  # 需分析的事件类型
EVENT_WINDOW = (-3, 5)     # 事件窗口日
ESTIMATION_WINDOW = 10    # 估计窗口长度
SIGNIFICANCE_LEVEL = 0.05  # 显著性水平
OUTPUT_DIR = 'analysis/event_analysis_results'  # 结果保存目录

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 检查字体是否可用，如果不可用则使用其他备选字体
def check_chinese_font():
    all_fonts = set([f.name for f in mpl.font_manager.fontManager.ttflist])
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    
    for font in chinese_fonts:
        if font in all_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            print(f"使用字体: {font}")
            return True
    
    print("警告：未找到合适的中文字体，图表中的中文可能无法正确显示")
    return False

# ==================== 数据准备 ====================
df = pd.read_csv("analysis/merged_DetailData.csv", parse_dates=['date'])
df = df.sort_values('date').drop_duplicates('date', keep='first')
df['return'] = np.log(df['price'] / df['price'].shift(1))

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 检查中文字体
check_chinese_font()

def calculate_event_window_returns(event_date, df, event_window, estimation_window):
    """计算单个事件的窗口收益率"""
    try:
        # 找到事件日期的索引
        event_idx = df.index[df['date'] == event_date][0]
        
        # 计算窗口起止索引
        est_start = event_idx - estimation_window + event_window[0]
        event_end = event_idx + event_window[1]
        
        # 检查数据是否足够
        if est_start < 0 or event_end >= len(df):
            return None
        
        # 获取估计窗口和事件窗口的收益率
        estimation_returns = df.iloc[est_start:est_start + estimation_window]['return'].values
        event_returns = df.iloc[event_idx + event_window[0]:event_idx + event_window[1] + 1]['return'].values
        
        # 检查数据是否完整
        if len(estimation_returns) != estimation_window or len(event_returns) != event_window[1] - event_window[0] + 1:
            return None
            
        return estimation_returns, event_returns
        
    except (IndexError, KeyError):
        return None

def analyze_event(event_type):
    """执行单个事件类型分析"""
    print(f"\n分析 {event_type} 类型事件...")
    
    # 识别事件日期
    event_dates = df[df[event_type] == 1]['date'].unique()
    if len(event_dates) == 0:
        print(f"警告：未找到{event_type}事件日期")
        return None
    
    print(f"找到 {len(event_dates)} 个事件日期")
    
    # 存储结果
    valid_events = 0
    ar_matrix = []  # 用于存储所有事件的异常收益率
    cars = []       # 用于存储所有事件的累计异常收益率
    
    # 事件窗口的天数
    window_length = EVENT_WINDOW[1] - EVENT_WINDOW[0] + 1
    
    # 逐个事件分析
    for date in event_dates:
        result = calculate_event_window_returns(date, df, EVENT_WINDOW, ESTIMATION_WINDOW)
        if result is None:
            continue
            
        estimation_returns, event_returns = result
        
        # 计算正常收益率（使用估计窗口的平均值）
        normal_return = np.nanmean(estimation_returns)
        if np.isnan(normal_return):
            continue
            
        # 计算异常收益率
        ar = event_returns - normal_return
        if len(ar) == window_length:  # 确保长度一致
            ar_matrix.append(ar)
            cars.append(np.sum(ar))
            valid_events += 1
    
    if valid_events == 0:
        print(f"警告：{event_type}类型没有有效的事件数据")
        return None
    
    # 转换为numpy数组以便计算
    ar_matrix = np.array(ar_matrix)
    cars = np.array(cars)
    
    # 计算平均异常收益率和统计量
    mean_ar = np.nanmean(ar_matrix, axis=0)
    mean_car = np.nanmean(cars)
    car_std = np.nanstd(cars, ddof=1)
    t_stat = mean_car / (car_std / np.sqrt(valid_events)) if car_std > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), valid_events-1)) if car_std > 0 else 1
    
    print(f"有效事件数: {valid_events}")
    print(f"平均CAR: {mean_car:.4f}")
    print(f"t统计量: {t_stat:.4f}")
    print(f"p值: {p_value:.4f}")
    
    return {
        'event_type': event_type,
        'event_count': valid_events,
        'mean_ar': mean_ar,
        'mean_car': mean_car,
        't_stat': t_stat,
        'p_value': p_value,
        'car_std': car_std
    }

# ==================== 批量分析 ====================
print("开始事件研究分析...")
all_results = []
for etype in EVENT_TYPES:
    result = analyze_event(etype)
    if result:
        all_results.append(result)

if not all_results:
    print("错误：没有获得有效的分析结果")
    exit()

# ==================== 可视化 ====================
print("\n生成可视化结果...")
plt.figure(figsize=(15, 8))

# 绘制累计异常收益率
for res in all_results:
    days = range(EVENT_WINDOW[0], EVENT_WINDOW[1]+1)
    car = np.cumsum(res['mean_ar'])
    
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
print("\n保存分析结果...")
# 保存统计结果
result_df = pd.DataFrame([{
    '事件类型': r['event_type'],
    '事件数量': r['event_count'],
    '平均CAR': r['mean_car'],
    't统计量': r['t_stat'],
    'p值': r['p_value'],
    'CAR标准差': r['car_std'],
    '显著性': '显著' if r['p_value'] < SIGNIFICANCE_LEVEL else '不显著'
} for r in all_results])

result_df.to_csv(f'{OUTPUT_DIR}/statistical_results.csv', index=False, encoding='utf-8-sig')

# 保存每日异常收益率
ar_df = pd.DataFrame({
    '窗口日': range(EVENT_WINDOW[0], EVENT_WINDOW[1]+1)
})

for res in all_results:
    ar_df[f'{res["event_type"]}_AR'] = res['mean_ar']
    ar_df[f'{res["event_type"]}_CAR'] = np.cumsum(res['mean_ar'])

ar_df.to_csv(f'{OUTPUT_DIR}/daily_returns.csv', index=False, encoding='utf-8-sig')

print(f"\n分析完成！结果已保存至 {OUTPUT_DIR} 目录")
print("- CAR_comparison.png: 累计异常收益率对比图")
print("- statistical_results.csv: 统计检验结果")
print("- daily_returns.csv: 每日异常收益率数据")