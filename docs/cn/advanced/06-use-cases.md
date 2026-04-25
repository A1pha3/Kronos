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

# 如果使用 Kronos-mini，需要搭配 Kronos-Tokenizer-2k 分词器：
# tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
# model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
# predictor = KronosPredictor(model, tokenizer, max_context=2048)  # mini 使用 2048 上下文
predictor = KronosPredictor(model, tokenizer, max_context=512)

df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])  # 必须转为 datetime 类型，否则 KronosPredictor 内部的 .dt 访问器会报错

lookback = 400
pred_len = 60

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']  # 此处 x_timestamp 已是 datetime 类型
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

### 如何解读多场景模拟结果

多场景模拟的价值不在于单条路径的精确性，而在于**整体分布特征**。以下是几个关键的解读维度：

1. **趋势一致性**：如果 10 条路径中有 7 条以上呈现相同的趋势方向（上涨/下跌），说明模型对该方向的置信度较高。如果涨跌各半，说明市场处于高不确定状态。

2. **置信区间宽度**：p5-p95 的宽度直接反映不确定性。宽度随时间步增长是正常的（远期更不确定），但如果初始几步的宽度就已经很大，说明模型对短期走势也缺乏信心。

3. **路径收敛性**：观察路径在哪些时间点开始"发散"。早期发散通常意味着近期历史信号弱；晚期发散则是正常的远期不确定性。

4. **极端路径分析**：关注 p5 和 p95 对应的极端路径，它们代表的不是"最可能的结果"，而是"合理的不确定性边界"。如果实际走势超出了这个边界，说明出现了模型未能捕捉的因素（如突发政策变化）。

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

> **关于 amount 列**：如果数据有 `volume` 但没有 `amount`，KronosPredictor 会自动推算 `amount = volume * mean(open, high, low, close)`（逐行算术均价）。如果你有真实的成交额数据，建议手动添加 `amount` 列以获得更准确的预测。详见 [常见问题](../references/faq.md)。

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

## 场景 7：预测结果质量评估

如何判断一次预测结果是否"合理"？以下提供一个系统化的评估框架：

### 基础检查

```python
import numpy as np

def evaluate_prediction(x_df, pred_df):
    """基础预测质量检查"""
    issues = []

    # 0. 输入保护：确保 x_df 有足够数据计算历史波动
    if len(x_df) < 2:
        issues.append("历史数据不足 2 行，无法计算波动对比")
        return False

    # 1. 连续性检查：相邻K线不应出现极端跳变
    for col in ['open', 'high', 'low', 'close']:
        changes = pred_df[col].pct_change().dropna()
        if len(changes) == 0:
            continue
        max_change = changes.abs().max()
        if max_change > 0.3:  # 单步变化超过30%
            issues.append(f"{col} 单步最大变化 {max_change:.1%}，可能异常")

    # 2. OHLC 逻辑检查：high >= max(open,close), low <= min(open,close)
    pred_df_check = pred_df.copy()
    violations = (
        (pred_df_check['high'] < pred_df_check[['open', 'close']].max(axis=1)).sum() +
        (pred_df_check['low'] > pred_df_check[['open', 'close']].min(axis=1)).sum()
    )
    if violations > 0:
        issues.append(f"有 {violations} 根K线违反 high >= max(open,close) 或 low <= min(open,close) 的逻辑")

    # 3. 价格范围合理性
    last_close = x_df['close'].iloc[-1]
    if last_close == 0:
        issues.append("最后一根K线收盘价为 0，无法计算相对波动")
    else:
        pred_range = (pred_df['close'].max() - pred_df['close'].min()) / abs(last_close)
        hist_range = (x_df['close'].max() - x_df['close'].min()) / abs(x_df['close'].iloc[-1])
        if hist_range > 0 and pred_range > hist_range * 3:
            issues.append(f"预测波动幅度 ({pred_range:.1%}) 远大于历史波动 ({hist_range:.1%})")

    if not issues:
        print("基础检查通过：预测结果在合理范围内。")
    else:
        print("发现以下问题：")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    return len(issues) == 0

# 使用示例
evaluate_prediction(x_df, pred_df)
```

### 进阶评估

更深入的质量评估需要对比预测结果与实际走势（需要有"未来"的真实数据）：

| 指标 | 计算方式 | 参考标准 |
|------|---------|---------|
| MAE（平均绝对误差） | `np.abs(pred - actual).mean()` | 相对价格越小越好 |
| 方向准确率 | `(np.sign(pred_change) == np.sign(actual_change)).mean()` | >50% 有参考价值 |
| 趋势捕获率 | 前 N 步的趋势方向与实际一致的比例 | 短期(1-20步) > 60% 为良好 |

> **重要提醒**：Kronos 的预测是概率性的，不应以"精确数值匹配"作为质量标准。更合理的评估维度是：趋势方向是否一致、波动范围是否合理、极端情况是否被覆盖。

---

## 场景 8：模型集成

Kronos 提供了多个不同规模的预训练模型（mini、small、base、large）。一个自然的思路是：能否组合多个模型的预测结果以获得更稳健的结论？

### 基本方法：多模型平均

```python
from model import Kronos, KronosTokenizer, KronosPredictor
import numpy as np

# 加载两个不同规模的模型
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

model_small = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor_small = KronosPredictor(model_small, tokenizer, max_context=512)

model_base = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor_base = KronosPredictor(model_base, tokenizer, max_context=512)

# 分别预测
pred_small = predictor_small.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=60, T=1.0, sample_count=3
)
pred_base = predictor_base.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=60, T=1.0, sample_count=3
)

# 取两个模型预测的平均值
ensemble_close = (pred_small['close'].values + pred_base['close'].values) / 2

# 检查两个模型的方向是否一致
small_direction = np.sign(pred_small['close'].iloc[-1] - pred_small['close'].iloc[0])
base_direction = np.sign(pred_base['close'].iloc[-1] - pred_base['close'].iloc[0])

if small_direction == base_direction:
    print(f"两个模型方向一致（{'看涨' if small_direction > 0 else '看跌'}），信号可信度更高")
else:
    print("两个模型方向分歧，建议谨慎对待此次预测")
```

### 何时有效

| 条件 | 说明 |
|------|------|
| 模型规模差异明显 | small 和 base 的参数量和训练容量不同，学到的模式侧重点有差异，集成时互补性更强 |
| 方向一致性高 | 如果多个模型独立预测出的趋势方向一致，说明该方向的信号在历史模式中较为稳健 |
| 数据与预训练分布接近 | 对于主流市场（A 股、美股日线等）的常见品种，不同模型的预训练知识都能覆盖 |

### 何时无效

| 情况 | 原因 |
|------|------|
| 同一模型多次采样 | 这是"多次采样"而非"模型集成"。同一个模型生成的多条路径高度相关，不提供独立视角 |
| 模型方向严重分歧 | 如果 small 看涨而 base 看跌，简单平均会抵消信号，不如分别分析各自的置信度 |
| 数据分布与预训练差距大 | 对于微调后的自定义模型与预训练模型做集成，两者学到的分布不同，平均结果可能 worse than either |
| mini 与其他模型集成 | mini 使用 2048 上下文和不同的分词器（Tokenizer-2k），其 token 粒度与其他模型不同，直接平均预测值在语义上不对齐 |

> **关键认知**：模型集成的前提是各个模型具有"独立的误差来源"。不同规模的 Kronos 模型共享相同的预训练数据，只是容量不同，因此集成的增益有限——不如增加单模型的采样次数来得经济。如果追求真正的多样性，建议结合 Kronos 预测与传统技术指标（如均线、MACD）进行多源信号集成。

### 集成效果评估

评估集成是否优于单模型时，建议关注以下指标：

| 指标 | 计算方式 | 说明 |
|------|---------|------|
| MSE | `((pred - actual) ** 2).mean()` | 数值精度——集成通常小幅降低 MSE |
| 方向准确率 | `(sign(pred_change) == sign(actual_change)).mean()` | 实用性核心——多模型方向一致时该指标提升明显 |
| 预测稳定性 | 多次运行的预测标准差 | 集成天然更平滑，波动更小 |

**何时集成有帮助**：当多个模型的方向判断一致时（如 small 和 base 均看涨），集成能提升方向准确率的置信度；在数据与预训练分布接近的主流市场（A 股、美股日线）上，集成对 MSE 的改善通常为边际提升（5%-15%）。

**何时集成帮助不大**：当模型间方向分歧严重时，简单平均会稀释信号，效果不如选择其中置信度更高的单一模型；在小众市场或微调后的自定义模型上，不同模型的预测分布可能不对齐，集成反而引入噪声。

---

## 结果使用建议

### 正确使用方式

1. **作为参考输入**：将 Kronos 的预测作为多个决策参考之一，而非唯一依据
2. **关注趋势方向**：相比精确价格，预测的涨跌方向更有参考价值
3. **使用置信区间**：通过多次采样生成概率区间，而非依赖单次预测
4. **结合其他指标**：将 Kronos 与传统技术指标、基本面分析结合使用
5. **多时间粒度交叉验证**：使用不同时间粒度的数据分别预测，交叉验证趋势信号

### 多时间粒度交叉验证

在实际交易决策中，不同时间粒度的 K 线反映了不同层级的市场信息。将 Kronos 应用于多个时间粒度并进行交叉验证，可以提高信号的可信度：

**核心思路**：用大粒度（如日线）判断大趋势方向，用小粒度（如分钟线）寻找具体入场时机。

```python
# 步骤 1：用日线判断大趋势
# 加载日线数据，预测未来 20 个交易日（约 1 个月）
pred_daily = predictor.predict(
    df=daily_x_df, x_timestamp=daily_x_ts, y_timestamp=daily_y_ts,
    pred_len=20, T=1.0, sample_count=5
)
daily_trend = "上涨" if pred_daily['close'].iloc[-1] > pred_daily['close'].iloc[0] else "下跌"

# 步骤 2：用 5 分钟线判断入场时机
# 加载同期的 5 分钟线数据，预测未来 60 根（约 5 小时）
pred_5min = predictor.predict(
    df=fivemin_x_df, x_timestamp=fivemin_x_ts, y_timestamp=fivemin_y_ts,
    pred_len=60, T=1.0, sample_count=3
)

# 步骤 3：交叉验证
if daily_trend == "上涨":
    # 大趋势看涨时，在分钟线中寻找短期回调后的买入时机
    # 例如：如果分钟线预测短期会回调后反弹，则回调时入场
    print("大趋势看涨，关注分钟线的短期回调买入机会")
else:
    # 大趋势看跌时，考虑观望或反向操作
    print("大趋势看跌，谨慎操作或关注做空机会")
```

**使用要点**：

| 维度 | 大粒度（日线/周线） | 小粒度（分钟线） |
|------|-------------------|-----------------|
| 作用 | 判断大趋势方向 | 寻找入场/出场时机 |
| 预测步数 | 较少（20-60 步） | 较多（60-120 步） |
| 采样次数 | 较多（5-10 次）以获得稳健方向 | 较少（1-3 次）以快速响应 |
| 决策权重 | 更高——逆势操作风险大 | 辅助——顺势时优化入场点 |

> **为什么这种方法有效？** 不同时间粒度的 K 线反映的是不同市场参与者的行为模式。日线反映机构和中长期资金的意图，分钟线反映短线交易者的博弈。当两个粒度的信号方向一致时，说明多个层级的市场参与者行为趋同，信号更可靠。当信号冲突时（如日线看涨但分钟线看跌），可能意味着短期调整或趋势转折，需要更加谨慎。

### 需要注意的局限

1. **预测不确定性随步数增加**：前 10-20 步的预测通常更可靠，越远的不确定性越大
2. **无法预测突发事件**：政策变化、黑天鹅事件等不在模型的能力范围内
3. **采样随机性**：每次运行结果可能不同，使用 `sample_count > 1` 可以获得更稳定的结果
4. **时间戳的精度影响**：`pd.bdate_range` 不跳过法定节假日，可能在假期期间产生"虚假"预测

### 预测结果与交易决策的关系

Kronos 的输出不应直接转化为交易信号。以下是一个推荐的决策框架：

```
Kronos 预测结果
      │
      ▼
┌─────────────────────┐
│ 1. 趋势方向确认      │  10 条路径中有几条看涨/看跌？
│    （多数路径一致？）  │  如果 7/10 路径看涨，趋势信号较强
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 2. 不确定性评估      │  置信区间（p5-p95）有多宽？
│    （区间宽度）        │  窄 = 高置信，宽 = 低置信
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 3. 多源交叉验证      │  是否与基本面、技术指标方向一致？
│    （结合其他信号）    │  多个信号一致 = 更可靠的决策基础
└─────────┬───────────┘
          │
          ▼
    交易决策（独立判断）
```

**关键原则**：Kronos 提供的是"如果历史模式延续，未来可能如何"的概率性参考。实际交易决策还需要考虑风险偏好、仓位管理、止损策略等综合因素。

### 如何设置合理的预期

在使用 Kronos 之前，建立正确的预期至关重要，可以避免对模型产生不切实际的期望：

1. **Kronos 不是"水晶球"**：模型学到的本质是历史 K 线模式的统计规律。当未来走势与历史模式相似时，预测有参考价值；当市场出现前所未有的情况时，预测可能完全失效。

2. **"方向比数值重要"**：不要期望模型精确预测"明天收盘价是 105.32 元"。合理的期望是：模型能在统计意义上给出未来一段时间内涨跌方向的倾向性判断，准确率略高于随机（如 55%-65%），而非确定性的方向预测。

3. **短期比长期可靠**：前 10-20 步的预测通常比 60-120 步的预测更有参考价值。可以将长程预测视为"趋势的大致走向参考"，而非具体的价格路径。

4. **单次预测无统计意义**：一次预测的结果可能是运气使然。需要在一个合理的时间窗口内（如 30 次以上独立预测）评估模型的总体表现，才能判断其是否对你关注的标的有参考价值。

5. **模型无法替代风险管理**：即使预测方向正确，幅度和时间点也可能偏离。止损策略、仓位控制、分散投资等风险管理原则不应因模型预测而被忽视。

---

## 动手练习

### 练习 1：多场景模拟——观察不同采样路径的分歧程度

使用同一组历史数据生成 20 条独立采样路径，计算收盘价在每一步的标准差，找出模型最不确定的时间步：

```python
import numpy as np
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback, pred_len = 400, 60
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 生成 20 条独立路径
paths = []
for _ in range(20):
    pred = predictor.predict(df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                            pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    paths.append(pred['close'].values)

paths_array = np.array(paths)  # 形状: (20, pred_len)
std_per_step = paths_array.std(axis=0)  # 每步标准差

# 找出分歧最大的 5 个时间步
top5_divergent = np.argsort(std_per_step)[-5:][::-1]
for step in top5_divergent:
    print(f"第 {step+1} 步: std={std_per_step[step]:.2f}, "
          f"90% CI=[{np.percentile(paths_array[:, step], 5):.2f}, {np.percentile(paths_array[:, step], 95):.2f}]")
```

**验证方法**：如果输出的标准差在后期时间步明显增大，说明模型在远期预测中不确定性增加，与预期一致。关注分歧最大的时间步——这些位置可能对应模型遇到陌生模式的区域。

### 练习 2：使用 evaluate_prediction 函数评估预测质量

将本文"预测质量评估"一节中的 `evaluate_prediction` 函数应用于一次真实预测，解读各项检查的含义：

```python
# 使用练习 1 中的 predictor 和数据
pred_df = predictor.predict(df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                           pred_len=pred_len, T=1.0, top_p=0.9, sample_count=5, verbose=True)

# 调用质量评估函数（代码见本文"预测质量评估"一节）
# 注意参数顺序：先是历史数据 x_df，再是预测结果 pred_df
passed = evaluate_prediction(x_df, pred_df)
print(f"检查结果: {'通过' if passed else '存在问题'}")
```

**验证方法**：如果输出"基础检查通过：预测结果在合理范围内"，说明连续性、OHLC 逻辑和价格范围三项检查均通过。如果输出"发现以下问题"，根据具体提示定位原因——单步变化超过 30% 可能是温度过高，OHLC 违反逻辑是正常现象（可用 `fix_ohlc_logic` 修复），波动远大于历史则需降低 `T`。

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

## 相关文档

- **前置**：[快速开始](../getting-started/02-quickstart.md) — 基础预测流程
- **前置**：[KronosPredictor 使用指南](../core-concepts/04-predictor.md) — 参数调节细节
- **实战**：[A 股市场预测实战](04-cn-markets.md) — A 股完整预测示例
- **进阶**：[批量预测指南](03-batch-prediction.md) — 多序列并行预测
- **进阶**：[CSV 微调指南](02-finetune-csv.md) — 适配特定市场数据
- **参考**：[模型选型指南](07-model-comparison.md) — 不同模型规格对比

