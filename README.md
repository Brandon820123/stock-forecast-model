# stock-forecast-model

一个模块化的股票预测实验框架，用于学习和比较：

- 价格预测：Naive Price Baseline、SARIMA Close、SARIMA Log Return
- 涨跌方向分类：Majority Class Baseline、Logistic Regression、Random Forest
- 模型评估：MAE、RMSE、MAPE、Accuracy、Precision、Recall、F1-score、Confusion Matrix
- 可视化输出：价格预测、模型对比、混淆矩阵、特征重要性、预测分布

项目重点不是声称模型能赚钱，而是建立一个更严谨的实验流程：明确数据来源、避免未来数据泄漏、与 baseline 比较，并在复杂模型没有打败 baseline 时如实输出结论。

## 项目目标

1. 使用历史 OHLCV 数据做收盘价预测实验。
2. 使用历史特征预测 next-day direction 和 5-day future direction。
3. 比较复杂模型是否真正打败简单 baseline。
4. 区分真实 yfinance 数据、缓存数据和 sample random-walk 数据。
5. 输出完整、可复现的终端评估表格和图表。

默认股票为 `AAPL`，默认时间范围为 `2018-01-01` 到运行当天。

## 数据来源

数据加载逻辑在 `data_loader.py` 中：

1. 如果 `USE_CACHE=True` 且 `data_cache` 中已有缓存，优先使用缓存，并打印 `Using cached market data`。
2. 如果没有缓存，尝试使用 `yfinance` 下载真实市场数据，并打印 `Using real yfinance market data`。
3. 下载成功后保存到 `data_cache`。
4. 如果 `yfinance` 失败且没有可用缓存，才生成 sample random-walk data。
5. sample data 不会被保存成真实缓存，避免下次误当真实行情使用。

如果使用 sample data，终端会打印：

```text
WARNING: sample data was used. Results are only for testing workflow, not real investment analysis.
```

sample data 只能用于验证代码流程，不能代表真实投资表现。

## Baseline 的作用

本项目使用两个 baseline：

1. `Naive Price Baseline`
   - 假设明天收盘价等于今天收盘价。
   - 用于价格预测任务的最低参考线。

2. `Majority Class Baseline`
   - 永远预测训练集中最多的类别。
   - 用于涨跌方向分类任务的最低参考线。

任何复杂模型都应该先和 baseline 比较。若复杂模型无法稳定打败 baseline，说明它可能没有学到有效信号，或者当前数据和特征不足以支持该任务。

## SARIMA Close vs SARIMA Log Return

价格预测模块在 `sarima_model.py` 中，输出三种模型对比：

- `Naive Price Baseline`
- `SARIMA Close`
- `SARIMA Log Return`

区别：

- `SARIMA Close`：直接对 `Close` 收盘价建模并预测价格。
- `SARIMA Log Return`：先预测 log return，再还原成价格。

SARIMA 使用参数搜索：

```python
p in [0, 1, 2]
d in [0, 1]
q in [0, 1, 2]
```

参数通过训练集 AIC 最小值选择。测试阶段使用 rolling forecast：每次只预测下一天，然后把当天真实值加入状态，再预测下一天。SARIMA 和 Naive Price Baseline 使用完全相同的测试集。

如果 SARIMA 没有打败 Naive Baseline，终端会明确打印：

```text
SARIMA did not outperform the naive baseline on this dataset.
```

## 分类模型

分类模块在 `ml_model.py` 中，支持两个预测目标：

- `label_next_day`：如果 `Close(t+1) > Close(t)`，label = 1，否则为 0。
- `label_5d`：如果 `Close(t+5) > Close(t)`，label = 1，否则为 0。

模型包括：

- `Majority Class Baseline`
- `Logistic Regression`
- `Random Forest`

优化点：

- Logistic Regression 使用 `class_weight="balanced"`。
- Random Forest 使用 `class_weight="balanced"`。
- 分类模型使用 `predict_proba` 输出上涨概率。
- 默认阈值 `CLASSIFICATION_THRESHOLD = 0.55`：只有上涨概率大于 0.55 才预测上涨，否则预测下跌。
- 终端会输出每个模型的 `predicted up` 和 `predicted down` 数量。

## 特征工程

特征工程在 `features.py` 中。所有 rolling 特征只使用当前日及以前数据，未来价格只用于构造 label，不进入特征矩阵。

主要特征包括：

- `daily_return`
- `MA5`, `MA10`, `MA20`
- `volatility_5`, `volatility_20`
- `momentum_5`, `momentum_10`
- `RSI`
- `MACD`
- `volume_change`
- `return_lag_1`, `return_lag_2`, `return_lag_5`
- `rolling_mean_return_5`, `rolling_mean_return_20`
- `rolling_volatility_10`, `rolling_volatility_20`
- `price_vs_ma20`
- `volume_zscore_20`

训练前会删除包含 NaN 或无穷值的样本。

## 如何运行

建议使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

macOS / Linux 激活虚拟环境：

```bash
source .venv/bin/activate
```

## 终端输出

运行 `python main.py` 后，终端会打印：

1. 数据源类型
2. 股票代码
3. 数据时间范围
4. 样本数量
5. 特征数量
6. 价格预测模型对比表
7. SARIMA 参数搜索结果
8. 分类模型对比表
9. 每个分类模型预测上涨/下跌次数
10. Confusion Matrix
11. baseline 对比结论
12. 输出文件路径
13. 如果使用 sample data，会再次打印 warning

## 输出文件

图表保存到 `outputs` 文件夹：

- `price_prediction.png`：actual close、naive baseline、SARIMA Close、SARIMA Log Return。
- `model_comparison.png`：上半部分展示价格预测 MAE/RMSE/MAPE，下半部分展示分类 Accuracy/F1-score。
- `confusion_matrix_logistic.png`：Logistic Regression 混淆矩阵。
- `confusion_matrix_random_forest.png`：Random Forest 混淆矩阵。
- `feature_importance.png`：Random Forest 特征重要性。
- `prediction_distribution.png`：分类模型预测上涨/下跌数量对比。

## 配置项

主要配置在 `config.py`：

```python
TICKER = "AAPL"
START_DATE = "2018-01-01"
TEST_SIZE = 0.2

USE_CACHE = True
USE_LOCAL_CSV = True
LOCAL_CSV_PATH = "data_cache/AAPL.csv"
USE_SAMPLE_DATA_IF_DOWNLOAD_FAILS = False
DATA_CACHE_DIR = "data_cache"
OUTPUT_DIR = "outputs"

SARIMA_P_VALUES = [0, 1, 2]
SARIMA_D_VALUES = [0, 1]
SARIMA_Q_VALUES = [0, 1, 2]

CLASSIFICATION_THRESHOLD = 0.55
FUTURE_DIRECTION_DAYS = 5

RANDOM_STATE = 42
```

## 风险提示

本项目仅用于学习和研究，不构成投资建议。股票预测难度很高，模型结果不能直接用于真实交易。即使某次实验中模型打败 baseline，也不代表未来能稳定获利；真实市场会受到宏观经济、流动性、公司基本面、市场情绪和突发事件等多重因素影响。
