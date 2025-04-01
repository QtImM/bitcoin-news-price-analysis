import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

# 读取数据
def load_data(file_path):
    """加载合并后的数据"""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    return df

# 定义事件分析参数
EVENT_WINDOW = (-3, 5)  # 事件窗口：事件前3天到事件后5天
ESTIMATION_WINDOW = (-30, -10)  # 估计窗口：事件前30天到事件前10天

def event_study(data, event_dates):
    """
    执行事件分析
    :param data: 包含价格和情感分数的数据
    :param event_dates: 事件日期列表
    :return: 分析结果 DataFrame
    """
    results = []

    for date in event_dates:
        try:
            idx = data.index.get_loc(date)
            if isinstance(idx, (np.ndarray, list)):  # 如果 idx 是数组或列表，取第一个匹配的索引
                idx = idx[0]
        except KeyError:
            continue

        # 事件窗口检查
        ew_start = idx + EVENT_WINDOW[0]
        ew_end = idx + EVENT_WINDOW[1]
        if ew_start < 0 or ew_end >= len(data):  # 确保 ew_start 和 ew_end 是整数
            continue

        # 估计期检查
        est_start = idx + ESTIMATION_WINDOW[0]
        est_end = idx + ESTIMATION_WINDOW[1]
        if est_start < 0 or est_end >= len(data):  # 确保 est_start 和 est_end 是整数
            continue

        # 提取数据
        event_window = data.iloc[ew_start:ew_end + 1]
        estimation_window = data.iloc[est_start:est_end + 1]

        # 有效性验证
        if len(event_window) != (EVENT_WINDOW[1] - EVENT_WINDOW[0] + 1):
            continue
        if len(estimation_window) < (ESTIMATION_WINDOW[1] - ESTIMATION_WINDOW[0] + 1):
            continue

        # 计算收益率
        normal_ret = estimation_window['return'].mean()
        ar = event_window['return'] - normal_ret

        # 存储结果
        results.append({
            'event_date': date,
            'ar': ar.values,
            'car': ar.sum(),
            'sentiment_score': data.loc[date, 'sentiment_score']
        })

    return pd.DataFrame(results)

# 主程序
if __name__ == "__main__":
    # 加载数据
    FILE_PATH = "analysis/merged_DetailData.csv"
    df = load_data(FILE_PATH)

    # 计算收益率
    df['return'] = np.log(df['price']) - np.log(df['price'].shift(1))

    # 筛选事件日期（情感分数不为0的日期）
    event_dates = df[df['sentiment_score'] != 0].index

    # 执行事件分析
    results = event_study(df, event_dates)

    # 结果矩阵
    ar_matrix = pd.DataFrame(
        [x for x in results.ar],
        columns=[f't{i + EVENT_WINDOW[0]}' for i in range(len(results.ar[0]))]
    )

    # 可视化分析
    plt.figure(figsize=(14, 8))

    # 异常收益率分布
    plt.subplot(2, 2, 1)
    ar_matrix.mean().plot(kind='bar', color='steelblue')
    plt.title('Average Abnormal Returns')
    plt.axhline(0, color='black', linestyle=':')

    # 累计收益率分布
    plt.subplot(2, 2, 2)
    ar_matrix.mean().cumsum().plot(color='darkorange')
    plt.title('Cumulative Abnormal Returns')

    # 按情感分组分析
    plt.subplot(2, 2, 3)
    for label, group in results.groupby(pd.cut(results.sentiment_score,
                                               bins=[-np.inf, -0.1, 0.1, np.inf],
                                               labels=['负面', '中性', '正面'])):
        group.ar.apply(pd.Series).mean().plot(label=label)
    plt.title('Abnormal Returns by Sentiment')
    plt.legend()

    # CAR分布
    plt.subplot(2, 2, 4)
    plt.hist(results.car, bins=30, color='green', alpha=0.7)
    plt.title('CAR Distribution')
    plt.axvline(results.car.mean(), color='red')

    plt.tight_layout()
    plt.show()

    # 统计输出
    print(f'''
综合分析结果（基于{len(results)}个有效事件）：
============================================
1. 整体市场反应：
   - 平均CAR: {results.car.mean():.4f} (p={ttest_1samp(results.car, 0)[1]:.3f})
   - 正向反应比例: {len(results[results.car > 0]) / len(results):.1%}

2. 情感分化效应：
   - 正面事件平均CAR: {results[results.sentiment_score > 0.1].car.mean():.4f}
   - 负面事件平均CAR: {results[results.sentiment_score < -0.1].car.mean():.4f}
   - 中性事件平均CAR: {results[(results.sentiment_score >= -0.1) & (results.sentiment_score <= 0.1)].car.mean():.4f}

3. 时效性分析：
   - 最大反应日: {ar_matrix.mean().abs().idxmax()}
   - 持续效应天数: {np.where(ar_matrix.mean().cumsum() == ar_matrix.mean().cumsum().max())[0][0] - 3}天
''')