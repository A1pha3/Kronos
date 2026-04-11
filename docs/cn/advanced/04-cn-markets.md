# A 股市场预测实战 ⭐⭐⭐

> **目标读者**：想对 A 股个股进行 K 线预测的用户
> **前置要求**：[快速开始](../getting-started/02-quickstart.md) 已完成

### 学习目标

阅读本文后，你将能够：

- [ ] 使用 `prediction_cn_markets_day.py` 脚本对 A 股个股进行日 K 线预测
- [ ] 理解涨跌停限制的实现逻辑，并能针对不同板块调整 `limit_rate` 参数
- [ ] 根据预测场景选择合适的采样参数和预测步数

---

## 概述

本项目提供了一个完整的 A 股日 K 线预测脚本（`examples/prediction_cn_markets_day.py`），可以：

- 使用 [akshare](https://github.com/akfamily/akshare) 自动获取 A 股历史数据
- 使用 Kronos 模型预测未来 120 个交易日的走势
- 自动应用 A 股涨跌停限制（±10%）
- 保存预测结果（CSV）和可视化图表（PNG）

---

## 安装额外依赖

```bash
pip install akshare
```

---

## 使用方法

### 基本用法

```bash
# 预测上证指数（000001）
python examples/prediction_cn_markets_day.py --symbol 000001

# 预测比亚迪（002594）
python examples/prediction_cn_markets_day.py --symbol 002594
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | `000001` | A 股股票代码（如 `002594` 为比亚迪） |

---

## 脚本内部流程

### 1. 获取数据

通过 akshare 获取 A 股历史日 K 线数据：

```python
import akshare as ak

df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
```

脚本包含重试机制（最多 3 次），应对网络不稳定的情况。

### 2. 数据清洗

akshare 返回的数据需要清洗和重命名：

```python
# 列名重命名（中文 → 英文）
df.rename(columns={
    "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount"
}, inplace=True)

# 修复异常开盘价（停牌日记录为 0）
bad_open = (df["open"] == 0) | (df["open"].isna())
df.loc[bad_open, "open"] = df["close"].shift(1)

# 修复缺失成交额
if df["amount"].isna().all() or (df["amount"] == 0).all():
    df["amount"] = df["close"] * df["volume"]
```

### 3. 生成未来时间戳

预测未来交易日的日期：

```python
y_timestamp = pd.bdate_range(
    start=df["date"].iloc[-1] + pd.Timedelta(days=1),
    periods=PRED_LEN  # 120 个交易日
)
```

> **注意**：`pd.bdate_range` 仅跳过周末，**不跳过中国法定节假日**。这意味着预测时间戳中可能包含春节、国庆等假期的日期。对于日线级别的长期预测，这种偏差通常可以接受；如需精确跳过节假日，可以使用中国交易日历库（如 `chinese_calendar`）进行过滤。

### 4. 执行预测

```python
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=PRED_LEN,       # 120
    T=T,                      # 1.0
    top_p=TOP_P,              # 0.9
    sample_count=SAMPLE_COUNT # 1
)
```

### 5. 应用涨跌停限制

A 股有 ±10% 的涨跌停限制。脚本会逐根 K 线检查并裁剪预测结果：

```python
def apply_price_limits(pred_df, last_close, limit_rate=0.1):
    for i in range(len(pred_df)):
        limit_up = last_close * (1 + limit_rate)
        limit_down = last_close * (1 - limit_rate)

        for col in ["open", "high", "low", "close"]:
            value = pred_df.at[i, col]
            pred_df.at[i, col] = max(min(value, limit_up), limit_down)

        last_close = pred_df.at[i, "close"]  # 用修正后的收盘价作为下一根的基准
```

**为什么需要涨跌停限制？** 模型的预测不受市场规则约束，可能产生超出实际范围的预测。例如，如果模型预测某日收盘价上涨 15%，但 A 股主板涨跌停限制为 ±10%，则该预测在实际交易中不可能实现。应用涨跌停限制使预测结果更贴近实际。

### 6. 输出结果

```
outputs/
├── pred_000001_data.csv      # 合并的历史 + 预测数据
└── pred_000001_chart.png     # 可视化图表
```

---

## 脚本配置

可以在脚本中修改以下常量来调整预测行为：

```python
TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PRETRAINED = "NeoQuasar/Kronos-base"   # 使用 base 模型获得更好效果
DEVICE = "cpu"                                # "cuda:0" 使用 GPU
MAX_CONTEXT = 512                             # 最大上下文
LOOKBACK = 400                                # 历史窗口
PRED_LEN = 120                                # 预测步数
T = 1.0                                       # 温度
TOP_P = 0.9                                   # 核采样
SAMPLE_COUNT = 1                              # 采样次数
```

---

## 自定义修改建议

### 使用 GPU 加速

```python
DEVICE = "cuda:0"  # 如果有 NVIDIA GPU
```

### 增加采样次数以提高稳定性

```python
SAMPLE_COUNT = 5  # 5 次采样取平均
```

### 预测不同时间粒度

将 `period="daily"` 改为其他值：

```python
# 分钟线（需要 akshare 支持）
df = ak.stock_zh_a_hist(symbol=symbol, period="5", adjust="")

# 周线
df = ak.stock_zh_a_hist(symbol=symbol, period="weekly", adjust="")
```

### 调整涨跌停幅度

对于 ST 股票（涨跌停 ±5%）或科创板/创业板（±20%）：

```python
pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.05)  # ST 股
pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.20)  # 创业板
```

---

## 常见问题

### Q: akshare 获取数据失败？

**A**: akshare 的数据接口可能因源站变更而失效。建议：

1. 更新 akshare 到最新版本：`pip install akshare --upgrade`
2. 脚本已内置重试机制（3 次），耐心等待即可
3. 如果持续失败，可以从其他数据源下载 CSV 并手动加载

### Q: 预测结果看起来不太准确？

**A**: 金融市场预测本质上具有高度不确定性。Kronos 提供的是基于历史模式的概率性预测，不能保证准确性。以下因素会影响预测质量：

- 历史数据的长度和代表性（建议 `lookback >= 200`）
- 采样参数（`sample_count=5` 比单次采样更稳定）
- 市场状态（趋势市 vs 震荡市）

### Q: 预测 120 步是否太多？

**A**: 预测步数越长，不确定性越大。对于日线数据，建议将 `PRED_LEN` 设置为 20-60 个交易日。120 个交易日（约半年）的预测仅供参考。

---

## 动手练习

### 练习 1：预测一只 ST 股票并调整涨跌停参数

ST 股票的涨跌停限制为 ±5%，而非主板的 ±10%。尝试预测一只 ST 股票，并修改 `limit_rate` 参数：

```bash
# 预测一只 ST 股票（示例：ST *ST 博天，600686）
python examples/prediction_cn_markets_day.py --symbol 600686
```

在脚本中找到 `apply_price_limits` 的调用位置，将 `limit_rate` 从默认的 `0.1` 改为 `0.05`：

```python
# 修改前
pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

# 修改后
pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.05)
```

**验证方法**：对比修改前后生成的 `pred_600686_chart.png`，使用 `limit_rate=0.05` 时，预测曲线的单日最大涨跌幅不应超过 5%。可以检查 CSV 数据中相邻两根 K 线的收盘价变化率来确认。

---

## 自测清单

完成本指南后，检查你是否掌握了以下要点：

- [ ] 能解释数据清洗步骤中为什么需要修复开盘价为 0 的记录
- [ ] 知道 `pd.bdate_range` 生成的日期是否包含法定节假日（答案：可能包含）
- [ ] 理解 `apply_price_limits` 中为什么用修正后的收盘价作为下一根 K 线的基准
- [ ] 能针对 ST 股票和创业板分别设置正确的 `limit_rate` 值
- [ ] 知道如何通过 `SAMPLE_COUNT` 和 `PRED_LEN` 平衡预测的稳定性和实用性

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [批量预测指南](03-batch-prediction.md) | ⭐⭐⭐ | 同时预测多只股票 |
| [CSV 微调指南](02-finetune-csv.md) | ⭐⭐⭐ | 用 A 股数据微调模型 |

---
**文档元信息**
难度：⭐⭐⭐ | 类型：进阶指南 | 预计阅读时间：15 分钟
