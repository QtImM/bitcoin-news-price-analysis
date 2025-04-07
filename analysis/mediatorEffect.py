import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import t

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def mediation_analysis(merged_data):
    """
    进行中介效应分析
    X: daily_news_count (每日新闻数量)
    M: avg_sentiment (每日平均情感得分)
    Y: change_percent (价格变化率)
    """
    print("执行中介效应分析...")
    
    # 准备数据
    data = merged_data.copy()
    
    # 数据预处理
    print("\n1. 数据预处理:")
    print(f"原始数据量: {len(data)}")
    
    # 按日期聚合数据
    daily_data = data.groupby('date').agg({
        'sentiment_score': ['count', 'mean'],  # count作为新闻数量，mean作为平均情感得分
        'change_percent': 'first',  # 使用每日第一个价格变化率
    }).reset_index()
    
    # 重命名列
    daily_data.columns = ['date', 'daily_news_count', 'avg_sentiment', 'change_percent']
    
    print("\n2. 每日数据统计:")
    print(f"天数: {len(daily_data)}")
    print("\n每日新闻数量统计:")
    print(daily_data['daily_news_count'].describe())
    print("\n每日平均情感得分统计:")
    print(daily_data['avg_sentiment'].describe())
    print("\n每日价格变化率统计:")
    print(daily_data['change_percent'].describe())
    
    # 删除缺失值
    daily_data = daily_data.dropna()
    print(f"\n删除缺失值后的天数: {len(daily_data)}")
    
    try:
        # 标准化变量
        for col in ['daily_news_count', 'avg_sentiment', 'change_percent']:
            daily_data[f'{col}_std'] = (daily_data[col] - daily_data[col].mean()) / daily_data[col].std()
        
        # 路径 a: X -> M
        model_m = ols('avg_sentiment_std ~ daily_news_count_std', data=daily_data).fit()
        a = model_m.params['daily_news_count_std']
        p_a = model_m.pvalues['daily_news_count_std']
        
        # 路径 b: M -> Y
        model_y = ols('change_percent_std ~ daily_news_count_std + avg_sentiment_std', data=daily_data).fit()
        b = model_y.params['avg_sentiment_std']
        p_b = model_y.pvalues['avg_sentiment_std']
        
        # 路径 c: X -> Y (总效应)
        model_c = ols('change_percent_std ~ daily_news_count_std', data=daily_data).fit()
        c = model_c.params['daily_news_count_std']
        p_c = model_c.pvalues['daily_news_count_std']
        
        # 直接效应 (c')
        direct_effect = model_y.params['daily_news_count_std']
        p_direct = model_y.pvalues['daily_news_count_std']
        
        # 间接效应 (a * b)
        indirect_effect = a * b
        
        # 总效应 (c)
        total_effect = c
        
        # 判断显著性
        def is_significant(p_value):
            return p_value < 0.05
        
        # 输出结果
        print("\n=== 中介效应分析结果 ===")
        print("\n1. 路径系数:")
        print(f"路径a (新闻数量 -> 情感得分): {a:.4f} (p={p_a:.4f}, {'显著' if is_significant(p_a) else '不显著'})")
        print(f"路径b (情感得分 -> 价格变化): {b:.4f} (p={p_b:.4f}, {'显著' if is_significant(p_b) else '不显著'})")
        print(f"路径c (总效应): {c:.4f} (p={p_c:.4f}, {'显著' if is_significant(p_c) else '不显著'})")
        
        print("\n2. 效应分解:")
        print(f"直接效应 (c'): {direct_effect:.4f} (p={p_direct:.4f}, {'显著' if is_significant(p_direct) else '不显著'})")
        print(f"间接效应 (a*b): {indirect_effect:.4f}")
        print(f"总效应 (c): {total_effect:.4f}")
        
        # 计算中介效应比例
        if total_effect != 0:
            mediation_ratio = (indirect_effect / total_effect) * 100
            print(f"\n3. 中介效应占比: {mediation_ratio:.2f}%")
        
        # 判断中介效应类型
        print("\n4. 中介效应类型:")
        if not is_significant(p_a) or not is_significant(p_b):
            print("无显著中介效应")
        else:
            if not is_significant(p_direct):
                print("完全中介效应")
            else:
                print("部分中介效应")
        
        # 可视化
        plt.figure(figsize=(15, 10))
        
        # 路径图
        plt.subplot(2, 2, 1)
        plt.title("中介效应路径图", fontsize=16)
        
        # 绘制节点和箭头
        plt.annotate('', xy=(0.3, 0.5), xytext=(0.1, 0.5),
                    arrowprops=dict(arrowstyle="->", color='red'))
        plt.annotate('', xy=(0.7, 0.5), xytext=(0.5, 0.5),
                    arrowprops=dict(arrowstyle="->", color='blue'))
        plt.annotate('', xy=(0.7, 0.3), xytext=(0.1, 0.3),
                    arrowprops=dict(arrowstyle="->", color='green'))
        
        # 添加标签
        plt.text(0.1, 0.3, '每日新闻数量\n(X)', ha='center', bbox=dict(facecolor='white'))
        plt.text(0.5, 0.5, '平均情感得分\n(M)', ha='center', bbox=dict(facecolor='white'))
        plt.text(0.7, 0.3, '价格变化\n(Y)', ha='center', bbox=dict(facecolor='white'))
        
        # 添加系数
        plt.text(0.2, 0.55, f'a={a:.3f}*', color='red' if is_significant(p_a) else 'black')
        plt.text(0.6, 0.55, f'b={b:.3f}*', color='blue' if is_significant(p_b) else 'black')
        plt.text(0.4, 0.25, f"c'={direct_effect:.3f}*", color='green' if is_significant(p_direct) else 'black')
        
        plt.axis('off')
        
        # 散点图
        plt.subplot(2, 2, 2)
        sns.regplot(x='daily_news_count', y='avg_sentiment', data=daily_data, 
                   scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title('每日新闻数量 -> 平均情感得分', fontsize=14)
        
        plt.subplot(2, 2, 3)
        sns.regplot(x='avg_sentiment', y='change_percent', data=daily_data,
                   scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
        plt.title('平均情感得分 -> 价格变化', fontsize=14)
        
        plt.subplot(2, 2, 4)
        sns.regplot(x='daily_news_count', y='change_percent', data=daily_data,
                   scatter_kws={'alpha':0.5}, line_kws={'color':'green'})
        plt.title('每日新闻数量 -> 价格变化', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('analysis/mediation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细结果到CSV
        results_df = pd.DataFrame({
            '路径': ['a (X->M)', 'b (M->Y)', "c' (直接效应)", 'c (总效应)', 'a*b (间接效应)'],
            '系数': [a, b, direct_effect, total_effect, indirect_effect],
            'p值': [p_a, p_b, p_direct, p_c, None],
            '显著性': [
                '显著' if is_significant(p_a) else '不显著',
                '显著' if is_significant(p_b) else '不显著',
                '显著' if is_significant(p_direct) else '不显著',
                '显著' if is_significant(p_c) else '不显著',
                '-'
            ]
        })
        results_df.to_csv('analysis/mediation_results.csv', index=False, encoding='utf-8-sig')
        
        return {
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'total_effect': total_effect,
            'path_a': {'coef': a, 'p_value': p_a},
            'path_b': {'coef': b, 'p_value': p_b},
            'path_c': {'coef': c, 'p_value': p_c},
            'model_m': model_m,
            'model_y': model_y,
            'model_c': model_c,
            'daily_data': daily_data
        }
        
    except Exception as e:
        print(f"\n分析过程中出错: {str(e)}")
        print("错误发生时的数据状态:")
        print(daily_data.info())
        return None

if __name__ == "__main__":
    try:
        # 读取数据
        df = pd.read_csv("analysis\merged_DetailData.csv", parse_dates=['date'])
        print(f"成功读取数据，共{len(df)}条记录")
        
        # 执行分析
        result = mediation_analysis(df)
        
        if result is not None:
            print("分析完成！结果已保存到 analysis/mediation_analysis.png")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")