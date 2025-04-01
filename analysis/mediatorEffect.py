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
    X: news_count (新闻数量)
    M: sentiment_avg (情感得分)
    Y: price_change (价格变化率)
    """
    print("执行中介效应分析...")
    
    # 准备数据
    data = merged_data.copy()
    data['price_change'] = data['price'].pct_change()  # 计算价格变化率
    data = data.dropna(subset=['news_count', 'sentiment_avg', 'price_change'])  # 删除缺失值
    
    try:
        # 标准化变量
        for col in ['news_count', 'sentiment_avg', 'price_change']:
            data[f'{col}_std'] = (data[col] - data[col].mean()) / data[col].std()
        
        # 路径 a: X -> M
        model_m = ols('sentiment_avg_std ~ news_count_std', data=data).fit()
        a = model_m.params['news_count_std']
        p_a = model_m.pvalues['news_count_std']
        
        # 路径 b: M -> Y
        model_y = ols('price_change_std ~ news_count_std + sentiment_avg_std', data=data).fit()
        b = model_y.params['sentiment_avg_std']
        p_b = model_y.pvalues['sentiment_avg_std']
        
        # 路径 c: X -> Y (总效应)
        model_c = ols('price_change_std ~ news_count_std', data=data).fit()
        c = model_c.params['news_count_std']
        p_c = model_c.pvalues['news_count_std']
        
        # 直接效应 (c')
        direct_effect = model_y.params['news_count_std']
        p_direct = model_y.pvalues['news_count_std']
        
        # 间接效应 (a * b)
        indirect_effect = a * b
        
        # 总效应 (c)
        total_effect = c
        
        # 判断显著性
        def is_significant(p_value):
            return p_value < 0.05
        
        # 输出结果
        print("中介效应分析结果:")
        print(f"直接效应 (c'): {direct_effect:.4f} (p={p_direct:.4f}, {'显著' if is_significant(p_direct) else '不显著'})")
        print(f"间接效应 (a*b): {indirect_effect:.4f} (p_a={p_a:.4f}, {'显著' if is_significant(p_a) else '不显著'}; "
              f"p_b={p_b:.4f}, {'显著' if is_significant(p_b) else '不显著'})")
        print(f"总效应 (c): {total_effect:.4f} (p={p_c:.4f}, {'显著' if is_significant(p_c) else '不显著'})")
        
        # 判断中介效应是否显著
        if is_significant(p_a) and is_significant(p_b):
            print("中介效应显著！")
        else:
            print("中介效应不显著。")
        
        # 可视化
        plt.figure(figsize=(12, 8))
        
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
        plt.text(0.1, 0.3, '新闻数量', ha='center', bbox=dict(facecolor='white'))
        plt.text(0.5, 0.5, '情感得分', ha='center', bbox=dict(facecolor='white'))
        plt.text(0.7, 0.3, '价格变化', ha='center', bbox=dict(facecolor='white'))
        
        # 添加系数
        plt.text(0.2, 0.55, f'a={a:.3f} ({p_a:.3f})', color='red' if is_significant(p_a) else 'black')
        plt.text(0.6, 0.55, f'b={b:.3f} ({p_b:.3f})', color='blue' if is_significant(p_b) else 'black')
        plt.text(0.4, 0.45, f"c'={direct_effect:.3f} ({p_direct:.3f})", color='green' if is_significant(p_direct) else 'black')
        
        plt.axis('off')
        
        # 散点图
        plt.subplot(2, 2, 2)
        sns.regplot(x='news_count_std', y='sentiment_avg_std', data=data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title('新闻数量 -> 情感得分', fontsize=14)
        
        plt.subplot(2, 2, 3)
        sns.regplot(x='sentiment_avg_std', y='price_change_std', data=data, scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
        plt.title('情感得分 -> 价格变化', fontsize=14)
        
        plt.subplot(2, 2, 4)
        sns.regplot(x='news_count_std', y='price_change_std', data=data, scatter_kws={'alpha':0.5}, line_kws={'color':'green'})
        plt.title('新闻数量 -> 价格变化', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('analysis/mediation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'total_effect': total_effect,
            'model_m': model_m,
            'model_y': model_y,
            'model_c': model_c
        }
        
    except Exception as e:
        print(f"分析出错: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        # 读取数据
        df = pd.read_csv("analysis/merged_data.csv", parse_dates=['date'])
        print(f"成功读取数据，共{len(df)}条记录")
        
        # 执行分析
        result = mediation_analysis(df)
        
        if result is not None:
            print("分析完成！结果已保存到 analysis/mediation_analysis.png")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")