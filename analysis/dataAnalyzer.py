# data_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.api import OLS
from statsmodels.tools import add_constant
import matplotlib as mpl
import statsmodels.api as sm
from datetime import timedelta
from statsmodels.stats.mediation import Mediation
import seaborn as sns
import scipy.stats as stats

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif'] # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_merged_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.dropna(subset=['price'])  # 删除无效行
    return df

def generate_features(df):
    # 基础特征
    df['log_return'] = np.log(df['price']) - np.log(df['price'].shift(1))
    df['volatility_7d'] = df['log_return'].rolling(7).std() * np.sqrt(7)
    
    # 滞后特征
    for lag in [1, 3, 5]:
        df[f'sentiment_lag{lag}'] = df['sentiment_avg'].shift(lag)
    
    # 事件持续性特征
    event_cols = [col for col in df.columns if 'event_' in col]
    for col in event_cols:
        # 确保事件列是数值类型
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: 1 if isinstance(x, list) and len(x) > 0 else 0)
        df[f'{col}_7dsum'] = df[col].rolling(7).sum()
    
    return df.dropna()

def event_impact_analysis(df):
    # 事件类型影响力排序
    event_cols = [col for col in df.columns if col.startswith('event_') and '_7dsum' not in col]
    print("可用的事件列:", event_cols)  
    
    impact_results = []
    
    for event in event_cols:
        try:
            # 使用简单的线性回归模型
            # 准备数据
            X = df[[event]]
            y = df['log_return'] * 100  # 转换为百分比
            
            # 添加常数项
            X = add_constant(X)
            
            # 拟合模型
            model = OLS(y, X).fit()
            
            # 获取事件系数
            coef = model.params[event]
            p_value = model.pvalues[event]
            
            impact_results.append({
                'event_type': event.replace('event_', ''),
                'coef': float(coef),
                'p_value': float(p_value)
            })
            
            print(f"事件 {event} 的系数: {coef:.4f}, p值: {p_value:.4f}")
            
        except Exception as e:
            print(f"处理事件 {event} 时出错: {e}")
            # 使用默认值
            impact_results.append({
                'event_type': event.replace('event_', ''),
                'coef': 0.0,
                'p_value': 1.0
            })
    
    return pd.DataFrame(impact_results)

def advanced_garch_analysis(merged_data):
    """
    使用GARCH-M模型分析比特币收益率与新闻特征的关系，并可视化结果
    
    Parameters:
    -----------
    merged_data : DataFrame
        包含价格、收益率和新闻特征的合并数据
        
    Returns:
    --------
    ARCHResult
        拟合的GARCH模型结果
    """
    # 构建包含新闻特征的GARCH-M模型
    data = merged_data.copy().dropna(subset=['log_return', 'event_policy', 'sentiment_avg'])
    returns = data['log_return'] * 100  # 百分比形式提升收敛性
    
    # 准备外生变量（滞后一期）
    exog_vars = data[['event_policy', 'sentiment_avg']].shift(1).dropna()
    
    # 对齐数据
    aligned_data = pd.concat([returns, exog_vars], axis=1).dropna()
    y = aligned_data['log_return']
    X = aligned_data[['event_policy', 'sentiment_avg']]
    
    # 拟合GARCH模型
    try:
        model = arch_model(
            y, 
            vol='GARCH', 
            p=1, 
            q=1, 
            mean='AR', 
            lags=1, 
            dist='skewt',
            x=X
        )
        res = model.fit(disp='off', update_freq=0)
        
        # 输出关键参数
        print(res.summary())
        
        # 提取条件波动率
        conditional_vol = res.conditional_volatility
        
        # 获取对应的日期
        vol_dates = data.loc[aligned_data.index, 'date']
        
        # 创建图表
        plt.figure(figsize=(12, 10))
        
        # 1. 条件波动率图
        plt.subplot(3, 1, 1)
        plt.plot(vol_dates, conditional_vol, 'r-', linewidth=1.5)
        plt.title('比特币收益率条件波动率 (GARCH模型估计)')
        plt.ylabel('波动率 (%)')
        plt.grid(True, alpha=0.3)
        
        # 2. 实际收益率与波动率对比
        plt.subplot(3, 1, 2)
        plt.plot(vol_dates, y, 'b-', alpha=0.5, label='实际收益率')
        plt.plot(vol_dates, conditional_vol, 'r-', alpha=0.7, label='条件波动率')
        plt.legend()
        plt.title('收益率与波动率对比')
        plt.ylabel('百分比 (%)')
        plt.grid(True, alpha=0.3)
        
        # 3. 新闻事件影响
        plt.subplot(3, 1, 3)
        
        # 累计事件发生次数（用于可视化）
        policy_events = data.loc[aligned_data.index, 'event_policy'].cumsum()
        high_sentiment = (data.loc[aligned_data.index, 'sentiment_avg'] > 0.5).cumsum()
        low_sentiment = (data.loc[aligned_data.index, 'sentiment_avg'] < -0.5).cumsum()
        
        plt.plot(vol_dates, policy_events, 'g-', label='政策事件累计次数')
        plt.plot(vol_dates, high_sentiment, 'r-', label='高情感累计次数')
        plt.plot(vol_dates, low_sentiment, 'b-', label='低情感累计次数')
        plt.title('新闻事件与情感累计发生次数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlabel('日期')
        
        plt.tight_layout()
        plt.savefig("analysis/garch_model_results.png")
        print("GARCH模型分析结果已保存为 analysis/garch_model_results.png")
        
        # 绘制参数重要性图表
        plt.figure(figsize=(10, 6))
        
        # 提取模型参数和p值
        params = res.params[-2:] # 仅提取外生变量参数
        p_values = res.pvalues[-2:]
        param_names = ['政策事件', '情感得分']
        
        # 创建条形图
        bar_colors = ['green' if p < 0.05 else 'gray' for p in p_values]
        bars = plt.bar(param_names, params, color=bar_colors)
        
        # 添加参数值标签
        for bar, param, pval in zip(bars, params, p_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{param:.4f}\n(p={pval:.4f})',
                    ha='center', va='bottom', rotation=0)
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('新闻特征对比特币收益率的影响 (GARCH模型系数)')
        plt.ylabel('系数大小')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("analysis/garch_parameters.png")
        print("GARCH模型参数图表已保存为 analysis/garch_parameters.png")
        
        # 波动率预测
        forecasts = res.forecast(horizon=10)
        fcast_variance = forecasts.variance.iloc[-1]
        fcast = np.sqrt(fcast_variance) * 100  # 使用numpy的sqrt函数
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), fcast, 'ro-')
        plt.title('未来10日波动率预测 (GARCH模型)')
        plt.xlabel('预测天数')
        plt.ylabel('预测波动率 (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("analysis/garch_volatility_forecast.png")
        print("GARCH波动率预测图表已保存为 analysis/garch_volatility_forecast.png")
        
        return res
        
    except Exception as e:
        print(f"GARCH模型拟合出错: {e}")
        # 尝试更简单的模型
        try:
            print("尝试拟合更简单的GARCH(1,1)模型...")
            simple_model = arch_model(y, vol='GARCH', p=1, q=1)
            simple_res = simple_model.fit(disp='off')
            
            # 保存简单模型结果
            plt.figure(figsize=(10, 6))
            plt.plot(vol_dates, simple_res.conditional_volatility, 'r-')
            plt.title('比特币收益率条件波动率 (简化GARCH模型)')
            plt.ylabel('波动率 (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("analysis/simple_garch_volatility.png")
            print("简化GARCH模型结果已保存为 analysis/simple_garch_volatility.png")
            
            return simple_res
        except:
            print("GARCH模型拟合失败")
            return None

# 在你的data_analyzer.py中添加的事件研究分析函数
def event_study_analysis(merged_data, event_type='policy', market_data=None):
    """
    对特定事件类型进行事件研究分析
    
    Parameters:
    -----------
    merged_data : DataFrame
        包含价格和事件数据的DataFrame
    event_type : str, default='policy'
        要分析的事件类型
    market_data : DataFrame, optional
        市场基准数据，如标普500指数
        
    Returns:
    --------
    DataFrame
        包含每个事件日期的累计异常收益率(CAR)
    """
    # 确保日期列是datetime类型
    if merged_data['date'].dtype != 'datetime64[ns]':
        merged_data['date'] = pd.to_datetime(merged_data['date'])
    
    # 计算日收益率，如果还没有计算
    if 'log_return' not in merged_data.columns:
        merged_data['log_return'] = np.log(merged_data['price']).diff()
    
    # 筛选重大事件日期
    event_col = f'event_{event_type}'
    if event_col not in merged_data.columns:
        print(f"错误：找不到事件列 '{event_col}'")
        return pd.DataFrame()
        
    event_dates = merged_data[merged_data[event_col] == 1]['date'].tolist()
    print(f"找到 {len(event_dates)} 个 {event_type} 类型的事件")
    
    if not event_dates:
        return pd.DataFrame()
    
    # 定义分析窗口
    results = []
    for date in event_dates:
        try:
            # 定义事件窗口
            window_start = date - pd.Timedelta(days=3)
            window_end = date + pd.Timedelta(days=5)
            
            # 提取窗口数据
            window_data = merged_data[
                (merged_data['date'] >= window_start) & 
                (merged_data['date'] <= window_end)
            ].copy()
            
            if len(window_data) < 3:  # 确保有足够数据进行分析
                continue
                
            # 基于情绪的异常收益率计算（不使用市场模型）
            # 使用事件前20天的平均收益率作为正常收益率
            estimation_start = date - pd.Timedelta(days=23)
            estimation_end = date - pd.Timedelta(days=3)
            
            estimation_data = merged_data[
                (merged_data['date'] >= estimation_start) & 
                (merged_data['date'] < window_start)
            ]
            
            if len(estimation_data) < 10:  # 确保有足够数据进行估计
                continue
                
            # 计算基于历史均值的异常收益率
            normal_return = estimation_data['log_return'].mean()
            window_data['ar'] = window_data['log_return'] - normal_return
            
            # 计算累计异常收益率(CAR)
            car = window_data['ar'].sum() * 100  # 转换为百分比
            
            # 记录结果
            results.append({
                'event_date': date,
                'car': car,
                'event_type': event_type
            })
            
            # print(f"事件日期: {date.strftime('%Y-%m-%d')}, CAR: {car:.2f}%")
            
        except Exception as e:
            print(f"处理事件日期 {date} 时出错: {e}")
    
    # 返回结果DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # 计算平均CAR
        avg_car = results_df['car'].mean()
        print(f"\n{event_type}类事件的平均累计异常收益率: {avg_car:.2f}%")
        
        # 绘制CAR分布图
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(results_df)), results_df['car'])
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=avg_car, color='g', linestyle='-', alpha=0.5)
        plt.title(f"{event_type}类事件的累计异常收益率(CAR)分布")
        plt.xlabel("事件编号")
        plt.ylabel("累计异常收益率(%)")
        plt.tight_layout()
        plt.savefig(f"analysis/event_study_{event_type}.png")
    
    return results_df

def visualize_results(df, impact_df):
    # 确保coef列是数值类型，替换None为0
    impact_df['coef'] = impact_df['coef'].fillna(0.0)
    impact_df['p_value'] = impact_df['p_value'].fillna(1.0)
    
    # 1. 价格与情感的热力图（按月聚合）
    plt.figure(figsize=(12, 8))
    
    # 添加月份列并按月聚合
    df['month'] = df['date'].dt.strftime('%Y-%m')
    monthly_data = df.groupby('month').agg({
        'price': 'mean',
        'sentiment_avg': 'mean',
        'log_return': lambda x: np.exp(x.mean()) - 1  # 月度收益率
    }).reset_index()
    
    # 创建收益率分类
    monthly_data['return_cat'] = pd.cut(
        monthly_data['log_return'], 
        bins=[-float('inf'), -0.1, -0.05, 0, 0.05, 0.1, float('inf')],
        labels=['大跌', '小跌', '持平', '小涨', '大涨', '暴涨']
    )
    
    # 创建情感分类
    monthly_data['sentiment_cat'] = pd.cut(
        monthly_data['sentiment_avg'],
        bins=[-1, -0.3, -0.1, 0.1, 0.3, 1],
        labels=['极负面', '负面', '中性', '正面', '极正面']
    )
    
    # 绘制热力图
    sentiment_counts = pd.crosstab(monthly_data['sentiment_cat'], monthly_data['return_cat'])
    plt.imshow(sentiment_counts, cmap='YlOrRd')
    plt.colorbar(label='频率')
    plt.xticks(np.arange(len(sentiment_counts.columns)), sentiment_counts.columns, rotation=45)
    plt.yticks(np.arange(len(sentiment_counts.index)), sentiment_counts.index)
    plt.title('情感与价格收益率关系热力图')
    plt.tight_layout()
    plt.savefig("analysis/sentiment_return_heatmap.png")
    
    # 2. 散点图：情感得分与日收益率的关系
    plt.figure(figsize=(10, 6))
    plt.scatter(df['sentiment_avg'], df['log_return']*100, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(df['sentiment_avg'], df['log_return']*100, 1)
    p = np.poly1d(z)
    plt.plot(df['sentiment_avg'], p(df['sentiment_avg']), "r--", alpha=0.8)
    
    plt.xlabel('情感得分')
    plt.ylabel('日收益率 (%)')
    plt.title('情感得分与比特币日收益率关系')
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis/sentiment_return_scatter.png")
    
    # 3. 情感与价格的滚动相关性
    plt.figure(figsize=(12, 6))
    rolling_corr = df['sentiment_avg'].rolling(30).corr(df['log_return'])
    plt.plot(df['date'], rolling_corr)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('情感与收益率的30日滚动相关性')
    plt.xlabel('日期')
    plt.ylabel('相关系数')
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis/sentiment_price_correlation.png")
    
    # 4. 情感前后的累计收益率变化
    # 计算情感极端值(前10%和后10%)
    high_sentiment = df['sentiment_avg'] > df['sentiment_avg'].quantile(0.9)
    low_sentiment = df['sentiment_avg'] < df['sentiment_avg'].quantile(0.1)
    
    # 创建情感事件窗口
    event_windows = pd.DataFrame()
    for i in range(-5, 6):  # 从事件前5天到事件后5天
        event_windows[f'high_t{i}'] = df.loc[high_sentiment, 'log_return'].shift(-i)
        event_windows[f'low_t{i}'] = df.loc[low_sentiment, 'log_return'].shift(-i)
    
    # 计算累计收益
    high_cum_return = event_windows.filter(like='high').mean().cumsum() * 100
    low_cum_return = event_windows.filter(like='low').mean().cumsum() * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(-5, 6), high_cum_return.values, 'g-', label='高情感后')
    plt.plot(range(-5, 6), low_cum_return.values, 'r-', label='低情感后')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('情感极值前后的累计收益率变化')
    plt.xlabel('事件窗口日(0=事件日)')
    plt.ylabel('累计收益率 (%)')
    plt.savefig("analysis/sentiment_event_returns.png")
    
    # 5. 事件影响力柱状图 (保留原有的事件影响分析图)
    plt.figure(figsize=(10, 6))
    impact_df = impact_df.sort_values('coef', ascending=False)
    plt.bar(impact_df['event_type'], impact_df['coef'])
    plt.xticks(rotation=45)
    plt.title("事件类型对收益率的影响")
    plt.ylabel("系数")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("analysis/event_impact.png")

def plot_event_window_returns(merged_data, event_type='policy', window_days=(-5, 5)):
    """
    绘制事件窗口内的平均异常收益率(AAR)和累计平均异常收益率(CAAR)时序图
    
    Parameters:
    -----------
    merged_data : DataFrame
        包含价格和事件数据的DataFrame
    event_type : str, default='policy'
        要分析的事件类型
    window_days : tuple, default=(-5, 5)
        事件窗口范围，如(-5, 5)表示事件前5天到事件后5天
    """
    # 确保日期列是datetime类型
    if merged_data['date'].dtype != 'datetime64[ns]':
        merged_data['date'] = pd.to_datetime(merged_data['date'])
    
    # 计算日收益率，如果还没有计算
    if 'log_return' not in merged_data.columns:
        merged_data['log_return'] = np.log(merged_data['price']).diff()
    
    # 筛选事件日期
    event_col = f'event_{event_type}'
    if event_col not in merged_data.columns:
        print(f"错误：找不到事件列 '{event_col}'")
        return
        
    event_dates = merged_data[merged_data[event_col] == 1]['date'].tolist()
    print(f"找到 {len(event_dates)} 个 {event_type} 类型的事件")
    
    if not event_dates:
        return
    
    # 创建存储所有事件异常收益率的DataFrame
    all_event_ars = pd.DataFrame()
    
    # 处理每个事件日期
    for date in event_dates:
        try:
            # 定义估计窗口和事件窗口
            estimation_start = date - pd.Timedelta(days=30)
            estimation_end = date - pd.Timedelta(days=abs(window_days[0])+1)
            
            window_start = date + pd.Timedelta(days=window_days[0])
            window_end = date + pd.Timedelta(days=window_days[1])
            
            # 提取估计窗口数据
            estimation_data = merged_data[
                (merged_data['date'] >= estimation_start) & 
                (merged_data['date'] <= estimation_end)
            ]
            
            if len(estimation_data) < 15:  # 确保有足够数据进行估计
                continue
                
            # 计算正常收益率（基于历史平均）
            normal_return = estimation_data['log_return'].mean()
            
            # 提取事件窗口数据
            window_data = merged_data[
                (merged_data['date'] >= window_start) & 
                (merged_data['date'] <= window_end)
            ].copy()
            
            if len(window_data) < abs(window_days[0]) + window_days[1] + 1:  # 检查事件窗口数据完整性
                continue
            
            # 计算异常收益率
            window_data['ar'] = window_data['log_return'] - normal_return
            
            # 计算相对于事件日的偏移天数
            window_data['event_day'] = (window_data['date'] - date).dt.days
            
            # 将此事件的异常收益率添加到总数据集
            all_event_ars = pd.concat([all_event_ars, window_data[['event_day', 'ar']]])
            
        except Exception as e:
            print(f"处理事件 {date} 时出错: {e}")
    
    if len(all_event_ars) == 0:
        print("没有足够的事件数据进行分析")
        return
    
    # 按事件日计算平均异常收益率(AAR)
    aar = all_event_ars.groupby('event_day')['ar'].mean() * 100  # 转换为百分比
    
    # 计算累计平均异常收益率(CAAR)
    caar = aar.cumsum()
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 绘制AAR
    ax1.bar(aar.index, aar.values, color='blue', alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='g', linestyle='--', linewidth=2)
    ax1.set_title(f'{event_type}类事件窗口内的平均异常收益率(AAR)')
    ax1.set_ylabel('平均异常收益率(%)')
    ax1.grid(True, alpha=0.3)
    
    # 为显著的AAR标注星号
    for day, value in aar.items():
        # 简单估计显著性，这里仅作示例
        if abs(value) > 1.5:  # 假设1.5%为显著性阈值
            ax1.annotate('*', xy=(day, value), 
                        xytext=(0, 5 if value > 0 else -15),
                        textcoords='offset points',
                        ha='center', va='center',
                        fontsize=14, color='red')
    
    # 绘制CAAR
    ax2.plot(caar.index, caar.values, 'r-', marker='o', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='g', linestyle='--', linewidth=2)
    ax2.fill_between(caar.index, 0, caar.values, alpha=0.2, color='red' if caar.iloc[-1] < 0 else 'green')
    ax2.set_title(f'{event_type}类事件窗口内的累计平均异常收益率(CAAR)')
    ax2.set_xlabel('相对事件日的偏移天数')
    ax2.set_ylabel('累计平均异常收益率(%)')
    ax2.grid(True, alpha=0.3)
    
    # 设置x轴刻度
    ax2.set_xticks(range(window_days[0], window_days[1] + 1))
    
    plt.tight_layout()
    plt.savefig(f"analysis/event_window_{event_type}.png")
    plt.close()
    
    print(f"{event_type}类事件的窗口分析图已保存")
    return aar, caar

# 新增中介效应检验 - 修复警告信息
import numpy as np
import pandas as pd
import scipy.stats as stats

def mediation_analysis(merged_data):
    """
    进行中介效应分析并生成可视化结果 - 改进布局
    
    Parameters:
    -----------
    merged_data : DataFrame
        包含自变量、中介变量和因变量的数据
        
    Returns:
    --------
    dict
        中介效应分析的结果字典
    """
    print("执行中介效应分析...")
    
    # 删除缺失值
    data = merged_data.dropna(subset=['event_policy', 'sentiment_avg', 'volatility_7d'])
    
    # 定义变量
    X = data['event_policy']     # 自变量
    M = data['sentiment_avg']    # 中介变量
    Y = data['volatility_7d']    # 因变量
    
    try:
        # 手动执行三阶段回归
        # 1. X -> M 路径 (a路径)
        X_with_const = sm.add_constant(X)
        model_m = sm.OLS(M, X_with_const).fit()
        a_coef = model_m.params.iloc[1]  # 使用.iloc代替[]
        a_pvalue = model_m.pvalues.iloc[1]  # 使用.iloc代替[]
        
        # 2. X + M -> Y 路径 (b和c'路径)
        XM_data = pd.DataFrame({'X': X, 'M': M})
        XM_with_const = sm.add_constant(XM_data)
        model_y = sm.OLS(Y, XM_with_const).fit()
        
        # 使用.iloc或基于列名访问
        b_coef = model_y.params.loc['M']  # M -> Y (b路径)
        b_pvalue = model_y.pvalues.loc['M']
        cprime_coef = model_y.params.loc['X']  # X -> Y 直接效应 (c'路径)
        cprime_pvalue = model_y.pvalues.loc['X']
        
        # 3. X -> Y 总效应 (c路径)
        model_c = sm.OLS(Y, X_with_const).fit()
        c_coef = model_c.params.iloc[1]  # 使用.iloc代替[]
        c_pvalue = model_c.pvalues.iloc[1]  # 使用.iloc代替[]
        
        # 计算间接效应
        indirect_effect = a_coef * b_coef
        direct_effect = cprime_coef
        total_effect = c_coef
        
        # 计算间接效应的标准误差 (Sobel检验)
        # 公式: SE_ab = sqrt(a^2*SE_b^2 + b^2*SE_a^2)
        se_a = model_m.bse.iloc[1]  # 使用.iloc代替[]
        se_b = model_y.bse.loc['M']  # 使用列名
        se_indirect = np.sqrt(a_coef**2 * se_b**2 + b_coef**2 * se_a**2)
        
        # 计算间接效应的Z统计量
        z_indirect = indirect_effect / se_indirect
        p_indirect = 2 * (1 - stats.norm.cdf(abs(z_indirect)))  # 双尾检验
        
        # 收集结果
        result = {
            'a_coeff': a_coef,
            'a_pvalue': a_pvalue,
            'b_coeff': b_coef,
            'b_pvalue': b_pvalue,
            'direct_effect': direct_effect,
            'direct_pvalue': cprime_pvalue,
            'indirect_effect': indirect_effect,
            'indirect_pvalue': p_indirect,
            'total_effect': total_effect,
            'total_pvalue': c_pvalue,
            'z_statistic': z_indirect
        }
        
        # 计算中介效应比例
        proportion_mediated = indirect_effect / total_effect if total_effect != 0 else 0
        result['proportion_mediated'] = proportion_mediated
        
        # 创建可视化结果 - 改进版
        plt.figure(figsize=(14, 10))  # 增加图形尺寸，提供更多空间
        
        # 1. 中介路径图 - 位置调整
        plt.subplot(2, 2, 1)
        # 创建简单的路径图
        plt.plot([0, 1], [0.3, 0.3], 'k-', linewidth=2)  # X->Y路径
        plt.plot([0, 0.5, 1], [0.3, 0.6, 0.3], 'b--', linewidth=2)  # X->M->Y路径
        
        # 添加节点 - 减小节点大小，防止拥挤
        plt.scatter([0, 0.5, 1], [0.3, 0.6, 0.3], s=400, c=['lightblue', 'lightgreen', 'lightblue'], 
                    edgecolors='black', zorder=5)
        
        # 改进图例位置和字体大小
        plt.text(0, 0.36, "政策事件\n(X)", ha='center', va='center', fontsize=10)
        plt.text(0.5, 0.66, "市场情感\n(M)", ha='center', va='center', fontsize=10)
        plt.text(1, 0.36, "波动率\n(Y)", ha='center', va='center', fontsize=10)
        
        # 改进系数位置，使用较小字体
        plt.text(0.25, 0.48, f"a={a_coef:.3f}\np={a_pvalue:.3f}", ha='center', va='center', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        plt.text(0.75, 0.48, f"b={b_coef:.3f}\np={b_pvalue:.3f}", ha='center', va='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        plt.text(0.5, 0.24, f"c'={direct_effect:.3f}\np={cprime_pvalue:.3f}", ha='center', va='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        
        plt.title("中介效应路径示意图", fontsize=12, pad=10)
        plt.axis('off')
        
        # 设置适当的边界
        plt.xlim(-0.2, 1.2)
        plt.ylim(0.1, 0.8)
        
        # 2. 直接效应、间接效应和总效应条形图
        plt.subplot(2, 2, 2)
        effects = [direct_effect, indirect_effect, total_effect]
        effect_names = ["直接效应\n{:.3f}".format(direct_effect), 
                       "间接效应\n{:.3f}".format(indirect_effect), 
                       "总效应\n{:.3f}".format(total_effect)]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = plt.bar(effect_names, effects, color=colors, alpha=0.7)
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title("中介效应分解", fontsize=12)
        plt.ylabel("效应大小")
        plt.grid(axis='y', alpha=0.3)
        
        # 3. 中介效应比例饼图 - 改进布局
        plt.subplot(2, 2, 3)
        
        if total_effect != 0:  # 避免总效应为0导致的除零错误
            # 确保有一定的间距，饼图不要太大
            plt.figure(plt.gcf().number)
            plt.subplot(2, 2, 3)
            
            # 创建更小的饼图，位于子图中央
            sizes = [abs(direct_effect/total_effect), abs(indirect_effect/total_effect)]
            labels = ['直接效应', '间接效应']  # 简化标签
            
            # 减小饼图大小，增加内部空白
            plt.pie(sizes, labels=labels, colors=['#3498db', '#e74c3c'],
                   autopct='%1.1f%%', startangle=90, explode=(0, 0.1), 
                   textprops={'fontsize': 10}, pctdistance=0.8, radius=0.8)
            
            # # 添加额外文本说明比例
            # plt.text(0.5, -1.1, f"直接效应: {abs(direct_effect/total_effect)*100:.1f}%\n"
            #                   f"间接效应: {abs(indirect_effect/total_effect)*100:.1f}%", 
            #        transform=plt.gca().transAxes, ha='center', fontsize=9)
            
            plt.title("中介效应比例", fontsize=12, pad=10)
        else:
            plt.text(0.5, 0.5, "总效应为零，无法计算比例", ha='center', va='center')
            plt.title("中介效应比例(总效应为零)", fontsize=12)
            plt.axis('off')
        
        # 4. 显著性检验结果
        plt.subplot(2, 2, 4)
        
        # 创建统计显著性表格
        cell_text = [
            ['总效应 (c)', f'{total_effect:.4f}', f'{c_pvalue:.4f}'],
            ['直接效应 (c\')', f'{direct_effect:.4f}', f'{cprime_pvalue:.4f}'],
            ['间接效应 (a*b)', f'{indirect_effect:.4f}', f'{p_indirect:.4f}'],
            ['a 路径', f'{a_coef:.4f}', f'{a_pvalue:.4f}'],
            ['b 路径', f'{b_coef:.4f}', f'{b_pvalue:.4f}']
        ]
        
        column_labels = ['路径', '系数', 'p值']
        
        # 添加颜色标记显著性
        cell_colors = []
        for row in cell_text:
            p_value = float(row[2])
            row_colors = ['#f9f9f9', '#f9f9f9', 
                          '#d5f5e3' if p_value < 0.05 else 
                          '#fcf3cf' if p_value < 0.1 else '#f9f9f9']
            cell_colors.append(row_colors)
        
        table = plt.table(cellText=cell_text, colLabels=column_labels, 
                         loc='center', cellLoc='center', cellColours=cell_colors)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # 添加Sobel检验结果
        plt.title(f"中介效应显著性检验 (Sobel Z = {z_indirect:.3f})", fontsize=12)
        plt.axis('off')
        
        # 优化间距 - 增加子图之间的间距
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, hspace=0.35, wspace=0.35)
        
        plt.suptitle("比特币政策事件、市场情感与波动率的中介效应分析", fontsize=14, y=0.98)
        
        plt.savefig("analysis/mediation_analysis.png", dpi=300, bbox_inches='tight')
        print("中介效应分析结果已保存为 analysis/mediation_analysis.png")
        
        # 额外创建中介效应摘要图
        plt.figure(figsize=(10, 6))
        
        # 设置文本框样式
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        # 创建摘要文本
        summary_text = (
            f"中介效应分析摘要:\n\n"
            f"1. 总效应 (c): {total_effect:.4f} (p={c_pvalue:.4f})\n"
            f"2. 直接效应 (c'): {direct_effect:.4f} (p={cprime_pvalue:.4f})\n"
            f"3. 间接效应 (a*b): {indirect_effect:.4f} (p={p_indirect:.4f})\n\n"
            f"4. X→M路径 (a): {a_coef:.4f} (p={a_pvalue:.4f})\n"
            f"5. M→Y路径 (b): {b_coef:.4f} (p={b_pvalue:.4f})\n\n"
            f"6. Sobel检验: Z={z_indirect:.3f}, p={p_indirect:.4f}\n\n"
        )
        
        if total_effect != 0:
            summary_text += f"7. 中介比例: {abs(proportion_mediated)*100:.2f}%\n\n"
        
        # 添加结论
        if abs(indirect_effect) > 0.01 and p_indirect < 0.05:
            summary_text += "结论: 存在显著的中介效应。"
            if direct_effect * indirect_effect > 0:
                summary_text += "\n这是部分中介效应。"
            elif abs(direct_effect) < 0.01 or cprime_pvalue > 0.05:
                summary_text += "\n这是完全中介效应。"
            else:
                summary_text += "\n直接效应和间接效应方向相反，表现为抑制效应。"
        else:
            summary_text += "结论: 未发现显著的中介效应。"
        
        # 将摘要文本放在图表中央
        plt.text(0.5, 0.5, summary_text, transform=plt.gca().transAxes,
                fontsize=14, verticalalignment='center', horizontalalignment='center',
                bbox=props)
        
        plt.axis('off')
        plt.title("中介效应分析摘要", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig("analysis/mediation_summary.png", dpi=300)
        print("中介效应摘要已保存为 analysis/mediation_summary.png")
        
        return result
        
    except Exception as e:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"中介效应分析失败: {str(e)}\n\n请检查数据完整性和变量间关系", 
                 ha='center', va='center', fontsize=14,
                 bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.6))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("analysis/mediation_error.png")
        print(f"中介效应分析出错: {str(e)}")
        return None

if __name__ == "__main__":
    # 参数配置
    INPUT_PATH = "analysis/merged_data.csv"
    
    # 执行流程
    df = load_merged_data(INPUT_PATH)
    df = generate_features(df)
    impact_df = event_impact_analysis(df)
    
    # 执行事件研究分析
    for event_type in ['policy', 'market', 'technology', 'adoption', 'security']:
        print(f"\n分析 {event_type} 类事件...")
        event_study_analysis(df, event_type=event_type)
        
        # 添加事件窗口分析
        print(f"\n生成 {event_type} 类事件窗口时序图...")
        plot_event_window_returns(df, event_type=event_type, window_days=(-5, 5))
    
    # 执行高级GARCH分析
    print("\n执行高级GARCH分析...")
    garch_res = advanced_garch_analysis(df)
    
    # 执行中介效应分析
    print("\n执行中介效应分析...")
    mediation_result = mediation_analysis(df)
    
    visualize_results(df, impact_df)
    impact_df.to_csv("analysis/event_impact.csv", index=False)
    print("分析完成！")