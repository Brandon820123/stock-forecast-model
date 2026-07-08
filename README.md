# stock-forecast-model

一个完整可运行的股票预测模型项目，用于学习历史价格预测、下一交易日涨跌方向分类、Baseline 对照、模型评估和结果可视化。

本项目不是 TOPSIS 选股系统，而是一个独立的时间序列与机器学习预测流程示例。

## 项目目标

1. 使用历史股票数据预测未来收盘价。
2. 使用历史股票数据预测下一交易日涨跌方向。
3. 对比 Baseline、SARIMA 和机器学习分类模型的效果。
4. 输出评估指标和可视化结果。

默认股票为 `AAPL`，默认时间范围为 `2018-01-01` 到运行当天。

## 数据来源

项目使用 `yfinance` 下载历史 OHLCV 数据，并支持本地缓存：

- 如果 `data_cache` 中已有缓存，优先读取缓存，并在终端提示 `cache data`。
- 如果没有缓存，则尝试从 yfinance 下载真实数据，并在终端提示 `yfinance real data`。
- 如果 yfinance 下载失败且没有缓存，会生成 sample random-walk 数据，并在终端提示 `sample data`。

sample data 只能用于测试代码流程，不能用于真实投资分析。

## Baseline 是什么

项目包含两个 Baseline 模型：

1. `Naive Price Baseline`
   - 假设明天收盘价等于今天收盘价。
   - 用作价格预测任务的最基础对照。

2. `Majority Class Baseline`
   - 对涨跌方向预测，永远预测训练集中出现最多的类别。
   - 用作分类任务的最基础对照。

任何复杂模型都应至少和 Baseline 做比较，否则很难判断模型是否真正有用。

## SARIMA 是什么

SARIMA 是经典时间序列模型 ARIMA 的季节性扩展。本项目使用 `statsmodels` 的 `SARIMAX` 实现价格预测，默认参数为：

```python
order = (1, 1, 1)
seasonal_order = (0, 0, 0, 0)
```

默认预测 `Close` 收盘价，并按时间顺序切分训练集和测试集，不进行 shuffle。

输出指标：

- MAE
- RMSE
- MAPE

## 机器学习分类模型

分类目标为预测下一交易日涨跌方向：

```text
如果 next_close > current_close，label = 1
否则 label = 0
```

特征仅使用当前日及以前的数据，避免未来数据泄漏。特征包括：

- daily_return
- MA5
- MA10
- MA20
- volatility_5
- volatility_20
- momentum_5
- momentum_10
- RSI
- MACD
- volume_change

当前实现的分类模型：

- Logistic Regression
- Random Forest Classifier

输出指标：

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 如何运行

建议使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

macOS / Linux 激活虚拟环境可使用：

```bash
source .venv/bin/activate
```

## 输出结果说明

运行 `python main.py` 后，终端会打印：

1. 当前使用的数据来源：`yfinance real data`、`cache data` 或 `sample data`。
2. Baseline 价格预测结果。
3. SARIMA 价格预测结果。
4. Majority Class Baseline 分类结果。
5. Logistic Regression 分类结果。
6. Random Forest 分类结果。
7. 所有输出图片路径。

图片会保存到 `outputs` 文件夹：

- `price_prediction.png`：actual close vs predicted close
- `confusion_matrix.png`：Random Forest 混淆矩阵
- `feature_importance.png`：Random Forest 特征重要性
- `model_comparison.png`：Baseline / SARIMA / Random Forest 等模型对比

## 风险提示

本项目仅用于学习和研究时间序列建模、机器学习分类、Baseline 对照和量化数据处理流程，不构成任何投资建议。

股票市场噪声很大，受到宏观经济、公司基本面、市场情绪、流动性和突发事件等多重因素影响。预测模型不能保证稳定赚钱，也不应作为真实交易决策的唯一依据。

