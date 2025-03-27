# 中介效应分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.mediation import Mediation

def mediation_analysis(merged_data):
    """
    进行中介效应分析并生成可视化结果
    
    Parameters:
    -----------
    merged_data : DataFrame
        包含自变量、中介变量和因变量的数据
        
    Returns:
    --------
    MediationResults
        中介效应分析的结果对象
    """
    print("执行中介效应分析...")
    
    # 删除缺失值
    data = merged_data.dropna(subset=['event_policy', 'sentiment_avg', 'volatility_7d'])
    
    # 定义变量
    X = data['event_policy']     # 自变量
    M = data['sentiment_avg']    # 中介变量
    Y = data['volatility_7d']    # 因变量
    
    try:
        # 三阶段回归模型
        med_model = Mediation(
            Y,                   # 因变量
            X,                   # 自变量
            M,                   # 中介变量
            boot=True,           # 使用bootstrap进行推断
            seed=42,             # 随机数种子
            boot_iterations=5000 # 增加迭代次数提高稳定性
        )
        result = med_model.fit()
        
        # 提取关键结果
        total_effect = result.total_effect
        direct_effect = result.direct_effect
        indirect_effect = result.indirect_effect
        
        proportion_mediated = indirect_effect / total_effect if total_effect != 0 else 0
        
        # 创建可视化结果
        plt.figure(figsize=(12, 8))
        
        # 1. 中介路径图
        plt.subplot(2, 2, 1)
        # 创建简单的路径图
        plt.plot([0, 1], [0.3, 0.3], 'k-', linewidth=2)  # X->Y路径
        plt.plot([0, 0.5, 1], [0.3, 0.6, 0.3], 'b--', linewidth=2)  # X->M->Y路径
        
        # 添加节点
        plt.scatter([0, 0.5, 1], [0.3, 0.6, 0.3], s=800, c=['lightblue', 'lightgreen', 'lightblue'], 
                    edgecolors='black', zorder=5)
        
        # 添加标签
        plt.text(0, 0.3, "政策事件\n(X)", ha='center', va='center', fontsize=12)
        plt.text(0.5, 0.6, "市场情感\n(M)", ha='center', va='center', fontsize=12)
        plt.text(1, 0.3, "波动率\n(Y)", ha='center', va='center', fontsize=12)
        
        # 添加路径系数
        a_effect = result.a_coeff
        b_effect = result.b_coeff
        c_effect = result.c_coeff
        
        plt.text(0.25, 0.5, f"a={a_effect:.3f}", ha='center', va='center', fontsize=10)
        plt.text(0.75, 0.5, f"b={b_effect:.3f}", ha='center', va='center', fontsize=10)
        plt.text(0.5, 0.2, f"c'={direct_effect:.3f}", ha='center', va='center', fontsize=10)
        
        plt.title("中介效应路径示意图")
        plt.axis('off')
        
        # 2. 直接效应、间接效应和总效应条形图
        plt.subplot(2, 2, 2)
        effects = [direct_effect, indirect_effect, total_effect]
        effect_names = ["直接效应", "间接效应", "总效应"]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = plt.bar(effect_names, effects, color=colors, alpha=0.7)
        
        # 添加效应值标签
        for bar, effect in zip(bars, effects):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height > 0 else height - 0.08,
                    f'{effect:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title("中介效应分解")
        plt.ylabel("效应大小")
        plt.grid(axis='y', alpha=0.3)
        
        # 3. 中介效应比例饼图
        plt.subplot(2, 2, 3)
        if total_effect != 0:  # 避免总效应为0导致的除零错误
            sizes = [abs(direct_effect/total_effect), abs(indirect_effect/total_effect)]
            labels = [f'直接效应\n{abs(direct_effect/total_effect)*100:.1f}%', 
                     f'间接效应\n{abs(indirect_effect/total_effect)*100:.1f}%']
            plt.pie(sizes, labels=labels, colors=['#3498db', '#e74c3c'],
                   autopct='%1.1f%%', startangle=90, explode=(0, 0.1))
            plt.title("中介效应比例")
        else:
            plt.text(0.5, 0.5, "总效应为零，无法计算比例", ha='center', va='center')
            plt.title("中介效应比例(总效应为零)")
            plt.axis('off')
        
        # 4. 抽样分布图
        plt.subplot(2, 2, 4)
        
        # 如果有bootstrap结果，绘制间接效应的抽样分布
        if hasattr(result, 'indirect_conf_int'):
            try:
                # 提取bootstrap样本
                boot_samples = result.boot_ind_effects
                
                # 绘制直方图和核密度估计
                sns.histplot(boot_samples, kde=True, color='#e74c3c', alpha=0.6)
                
                # 添加置信区间
                low, high = result.indirect_conf_int
                plt.axvline(x=low, color='r', linestyle='--', alpha=0.7, label=f'95%置信区间下限: {low:.3f}')
                plt.axvline(x=high, color='r', linestyle='--', alpha=0.7, label=f'95%置信区间上限: {high:.3f}')
                plt.axvline(x=indirect_effect, color='k', linestyle='-', label=f'间接效应: {indirect_effect:.3f}')
                
                plt.title("间接效应Bootstrap抽样分布")
                plt.xlabel("间接效应值")
                plt.ylabel("频率")
                plt.legend(fontsize=8)
            except Exception as e:
                plt.text(0.5, 0.5, f"无法绘制抽样分布: {str(e)}", ha='center', va='center')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, "未进行Bootstrap抽样", ha='center', va='center')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("analysis/mediation_analysis.png", dpi=300)
        print("中介效应分析结果已保存为 analysis/mediation_analysis.png")
        
        # 额外创建中介效应摘要图
        plt.figure(figsize=(10, 6))
        
        # 设置文本框样式
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        # 创建摘要文本
        summary_text = (
            f"中介效应分析摘要:\n\n"
            f"1. 总效应 (c): {total_effect:.4f}\n"
            f"2. 直接效应 (c'): {direct_effect:.4f}\n"
            f"3. 间接效应 (a*b): {indirect_effect:.4f}\n\n"
            f"4. X→M路径 (a): {a_effect:.4f}\n"
            f"5. M→Y路径 (b): {b_effect:.4f}\n\n"
        )
        
        if total_effect != 0:
            summary_text += f"6. 中介比例: {abs(proportion_mediated)*100:.2f}%\n\n"
        
        # 添加结论
        if abs(indirect_effect) > 0.01 and result.indirect_pvalue < 0.05:
            summary_text += "结论: 存在显著的中介效应。"
            if direct_effect * indirect_effect > 0:
                summary_text += "\n这是部分中介效应。"
            elif abs(direct_effect) < 0.01 or result.direct_pvalue > 0.05:
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
