# 使用场景与实战案例 ⭐⭐

> **目标读者**：想了解 Kronos 在实际场景中如何应用的各类用户
> **核心问题**：Kronos 能解决什么问题？在不同场景下如何配置和使用？

---

## 学习目标

阅读本文后，你将能够：

- [ ] 识别 Kronos 适用的预测场景及其局限性
- [ ] 根据不同市场和数据类型选择合适的模型配置
- [ ] 理解 Kronos 输出结果的正确使用方式（概率性预测，非确定性建议）

---

## 重要声明

Kronos 是一个基于历史模式的概率性预测模型，其输出仅供研究参考，**不构成任何投资建议**。金融市场的未来走势受多种不可预测因素影响，任何模型的预测都存在不确定性。

---

## 适用场景

### Kronos 擅长的场景

| 场景 | 说明 | 推荐配置 |
|------|------|----------|
| 趋势方向判断 | 判断未来 K 线的大致方向（涨/跌/横盘） | `T=1.0, sample_count=5` |
| 波动范围估计 | 预测未来价格波动的可能范围 | `T=1.2, sample_count=10` |
| 多场景模拟 | 生成多条可能的未来走势路径 | `T=1.5, sample_count=10` |
| 数据异常检测 | 对比预测与实际走势，发现异常偏离 | `T=0.5, sample_count=1` |
| 策略辅助研究 | 为量化策略提供价格走势参考 | 根据策略需求调整 |

### Kronos 不适用的场景

| 场景 | 原因 |
|------|------|
| 精确价格预测 | 模型输出是概率性的，无法精确预测具体价格 |
| 高频交易信号 | 推理延迟（秒级）不满足高频交易的毫秒级要求 |
| 突发事件预测 | 黑天鹅事件、政策突变等无法从历史数据中预测 |
| 单步确定性决策 | 模型的采样机制引入随机性，每次预测可能不同 |

---

## 场景 1：多市场 K 线趋势预测

Kronos 在 45+ 个全球交易所数据上进行了预训练，理论上支持任何金融市场的 K 线预测。

### 支持的市场类型

| 市场类型 | 数据要求 | 示例 |
|---------|---------|------|
| A 股 | 日线/分钟线 OHLCV | 上证指数、个股 |
| 美股 | 日线 OHLCV | SPY、AAPL |
| 加密货币 | 任意时间粒度 | BTC/USDT |
| 期货 | 日线/分钟线 | 原油、黄金 |
| 外汇 | 任意时间粒度 | EUR/USD |

### 不同市场的配置建议

```python
# A 股日线（波动适中）
pred_df = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=60,      # 约 3 个月交易日
    T=1.0,
    top_p=0.9,
    sample_count=5    # 多次采样提升稳定性
)

# 加密货币（波动较大）
pred_df = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=120,     # 加密货币 24/7 交易，可更长
    T=0.8,            # 稍低温度，减少极端预测
    top_p=0.85,
    sample_count=5
)

# 5 分钟线（短期预测）
pred_df = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=60,      # 5 小时
    T=1.0,
    top_p=0.9,
    sample_count=3
)
```

---

## 场景 2：多场景模拟

通过多次独立预测生成多种可能的未来走势，用于评估不确定性。

```python
import numpy as np
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 60

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 生成 10 条独立的预测路径
scenarios = []
for i in range(10):
    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=pred_len, T=1.2, top_p=0.95, sample_count=1
    )
    scenarios.append(pred_df['close'].values)

scenarios = np.array(scenarios)  # 形状: (10, pred_len)

# 计算统计量
mean_pred = scenarios.mean(axis=0)
std_pred = scenarios.std(axis=0)
p5 = np.percentile(scenarios, 5, axis=0)
p95 = np.percentile(scenarios, 95, axis=0)

print(f"最终预测步均值: {mean_pred[-1]:.2f}")
print(f"最终预测步 90% 置信区间: [{p5[-1]:.2f}, {p95[-1]:.2f}]")
```

**输出解读**：

- `mean_pred`：多条路径的平均值，反映模型的中心预期
- `p5` / `p95`：90% 置信区间的上下界，反映预测的不确定性范围
- 置信区间越宽，说明模型对未来走势越不确定

---

## 场景 3：波动率估计

通过多次预测的标准差估计未来波动率：

```python
# 使用不同随机种子生成多条预测
samples = []
for i in range(20):
    torch.manual_seed(i)
    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=pred_len, T=1.0, sample_count=1, verbose=False
    )
    # 计算每条路径的收益率
    returns = pred_df['close'].pct_change().dropna()
    samples.append(returns.values)

# 计算各时间步的波动率（标准差）
volatility = np.std(samples, axis=0)
print(f"平均预测波动率: {volatility.mean():.4f}")
```

---

## 场景 4：无成交量数据预测

Kronos 对没有成交量数据的场景同样适用。模型在预训练阶段已学习了只依赖价格信息的能力。

```python
# 只有 OHLC 四列的数据
x_df = df[['open', 'high', 'low', 'close']]

pred_df = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=60, T=1.0, sample_count=3
)

# 预测结果中 volume 和 amount 列为自动填充的 0
print(pred_df[['open', 'high', 'low', 'close']].head())
```

**适用数据源**：Yahoo Finance（部分市场无成交量）、自采数据、模拟数据等。

---

## 场景 5：自定义时间粒度

Kronos 不限制输入数据的时间粒度。以下展示不同粒度的使用方式：

```python
# 日线预测
df_daily = pd.read_csv("daily_data.csv")

# 15 分钟线预测
df_15min = pd.read_csv("15min_data.csv")

# 周线预测
df_weekly = pd.read_csv("weekly_data.csv")

# 所有粒度使用相同的 API 调用
pred_df = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=appropriate_len,  # 根据粒度调整
    T=1.0, sample_count=3
)
```

**推荐预测步数**（参考）：

| 时间粒度 | 推荐预测步数 | 对应时间跨度 | 说明 |
|---------|-------------|-------------|------|
| 1 分钟线 | 30-60 | 30-60 分钟 | 超短线分析 |
| 5 分钟线 | 60-120 | 5-10 小时 | 日内分析 |
| 15 分钟线 | 40-80 | 10-20 小时 | 日内趋势 |
| 1 小时线 | 24-72 | 1-3 天 | 短期趋势 |
| 日线 | 20-60 | 1-3 个月 | 中期趋势 |
| 周线 | 12-24 | 3-6 个月 | 长期趋势 |

---

## 场景 6：模型微调后的特定市场预测

当预训练模型在特定市场上表现不佳时，可以通过微调提升效果。

```python
# 步骤 1：微调模型（参考 CSV 微调指南）
# 步骤 2：使用微调后的模型
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("outputs/my_exp/tokenizer/best_model")
model = Kronos.from_pretrained("outputs/my_exp/basemodel/best_model")
predictor = KronosPredictor(model, tokenizer)

# 后续使用方式完全相同
pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=60)
```

**何时需要微调**：

| 情况 | 建议 |
|------|------|
| 预训练模型预测趋势大致正确 | 无需微调，可直接使用 |
| 数据特征与预训练数据差异较大（如加密货币衍生品） | 先微调分词器，再微调预测模型 |
| 只需适应新市场的短期规律 | 只微调预测模型，保留预训练分词器 |

---

## 结果使用建议

### 正确使用方式

1. **作为参考输入**：将 Kronos 的预测作为多个决策参考之一，而非唯一依据
2. **关注趋势方向**：相比精确价格，预测的涨跌方向更有参考价值
3. **使用置信区间**：通过多次采样生成概率区间，而非依赖单次预测
4. **结合其他指标**：将 Kronos 与传统技术指标、基本面分析结合使用

### 需要注意的局限

1. **预测不确定性随步数增加**：前 10-20 步的预测通常更可靠，越远的不确定性越大
2. **无法预测突发事件**：政策变化、黑天鹅事件等不在模型的能力范围内
3. **采样随机性**：每次运行结果可能不同，使用 `sample_count > 1` 可以获得更稳定的结果
4. **时间戳的精度影响**：`pd.bdate_range` 不跳过法定节假日，可能在假期期间产生"虚假"预测

---

## 自测清单

- [ ] 我能说出至少 3 种 Kronos 适合的使用场景
- [ ] 我知道 Kronos 不适合用于高频交易的原因
- [ ] 我能根据市场类型选择合适的预测参数
- [ ] 我能解释多次采样生成置信区间的方法
- [ ] 我理解 Kronos 预测结果"仅供研究参考"的含义

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [A 股市场预测实战](04-cn-markets.md) | ⭐⭐⭐ | 完整的 A 股预测示例 |
| [批量预测指南](03-batch-prediction.md) | ⭐⭐⭐ | 多市场并行预测 |
| [CSV 微调指南](02-finetune-csv.md) | ⭐⭐⭐ | 适配特定市场 |

---
**文档元信息**
难度：⭐⭐ | 类型：实战案例 | 预计阅读时间：20 分钟
