# 比特币新闻与价格分析项目

本项目分析比特币新闻事件与价格变动之间的关系，使用事件研究法和中介效应分析等方法研究不同类型新闻事件对比特币价格和波动率的影响。

## 项目结构

- `analysis/`: 包含分析脚本和结果
  - `dataAnalyzer.py`: 主要分析脚本
  - `dataMerger.py`: 数据合并处理脚本
  - `*.png`: 分析结果图表

## 主要功能

1. 事件研究分析 - 计算不同类型新闻事件的累积异常收益率(CAR)
2. 中介效应分析 - 研究政策事件通过市场情感影响价格波动的路径
3. GARCH模型分析 - 分析新闻特征对比特币收益率的影响
4. 事件窗口分析 - 研究事件前后的价格变动模式

## 安装与使用

1. 克隆仓库
   ```
   git clone https://github.com/QtImM/bitcoin-news-price-analysis.git
   ```

2. 安装依赖
   ```
   pip install -r requirements.txt
   ```

3. 运行分析
   ```
   python analysis/dataAnalyzer.py
   ```

## 分析结果示例

![中介效应分析](示例图片链接)

## 技术栈

- Python
- pandas
- matplotlib
- statsmodels
- arch (GARCH模型)

