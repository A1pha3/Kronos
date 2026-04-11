# 数据准备指南 ⭐

> **目标读者**：已完成快速开始，准备使用自有数据的用户
> **预计时间**：10 分钟
> **前置要求**：[安装与环境配置](01-installation.md) 已完成（建议在虚拟环境中操作）

---

## 学习目标

完成本教程后，你将能够：

- [ ] 理解 Kronos 的数据格式要求
- [ ] 正确处理时间戳
- [ ] 处理常见的数据质量问题
- [ ] 独立从外部数据源（CSV、akshare）准备符合要求的模型输入
- [ ] 编写简单的数据质量验证脚本

---

## 数据格式要求

### 核心列（必填）

Kronos 要求输入数据至少包含以下四列，列名必须严格匹配：

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `open` | float | 开盘价 | `12.35` |
| `high` | float | 最高价 | `12.58` |
| `low` | float | 最低价 | `12.20` |
| `close` | float | 收盘价 | `12.45` |

### 扩展列（可选）

| 列名 | 类型 | 说明 | 缺失时的行为 |
|------|------|------|-------------|
| `volume` | float | 成交量 | 自动填充为 0 |
| `amount` | float | 成交额 | 自动通过 `volume × 均价` 推算，或填充 0 |

KronosPredictor 内部的处理逻辑（源码位于 `model/kronos.py` 的 `predict()` 方法中）：

```python
if self.vol_col not in df.columns:
    df[self.vol_col] = 0.0
    df[self.amt_vol] = 0.0
if self.amt_vol not in df.columns and self.vol_col in df.columns:
    df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)
```

### 时间戳

除了 OHLCV 数据本身，你还需要提供两个时间戳序列：

1. **`x_timestamp`**：历史数据对应的 timestamps（pandas DatetimeIndex 或 Series）
2. **`y_timestamp`**：要预测的未来时间点

时间戳用于生成时间特征（小时、星期几、月份等），帮助模型捕捉周期性规律。KronosPredictor 内部会从时间戳中提取 5 个时间特征：

```python
# model/kronos.py 中 calc_time_stamps() 函数
def calc_time_stamps(x_timestamp):
    time_df = pd.DataFrame()
    time_df['minute'] = x_timestamp.dt.minute    # 分钟 (0-59)
    time_df['hour'] = x_timestamp.dt.hour        # 小时 (0-23)
    time_df['weekday'] = x_timestamp.dt.weekday  # 星期 (0-6，0=周一)
    time_df['day'] = x_timestamp.dt.day          # 日 (1-31)
    time_df['month'] = x_timestamp.dt.month      # 月 (1-12)
    return time_df
```

**为什么需要时间戳？** 金融市场具有明显的时间周期性（如日内波动、周末效应、月初月末效应）。时间特征让模型能够学习这些规律。

---

## 数据准备流程

### 从 CSV 文件加载

```python
import pandas as pd

df = pd.read_csv("your_data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])
```

### 从 akshare 获取 A 股数据

```python
import akshare as ak

df = ak.stock_zh_a_hist(symbol="000001", period="daily", adjust="")
df.rename(columns={
    "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount"
}, inplace=True)
df["timestamps"] = pd.to_datetime(df["date"])
df = df.sort_values("timestamps").reset_index(drop=True)
```

### 准备模型输入

```python
lookback = 400   # 历史窗口长度
pred_len = 120   # 预测步数

# 历史数据（DataFrame 格式）
x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.iloc[-lookback:]['timestamps']

# 未来时间戳（需要你根据交易时间生成）
# 例如：下一个交易日开始
y_timestamp = pd.bdate_range(
    start=df['timestamps'].iloc[-1] + pd.Timedelta(days=1),
    periods=pred_len
)
```

---

## 数据规范化

你不需要手动做数据规范化。KronosPredictor 在 `predict()` 方法内部自动完成以下处理：

1. **实例级标准化**：对每条序列独立计算均值和标准差，进行 z-score 标准化
2. **裁剪**：将标准化后的值裁剪到 `[-clip, clip]` 范围（默认 `clip=5`）
3. **预测后反标准化**：将预测结果还原到原始尺度

```python
# model/kronos.py 中 predict() 方法内的标准化逻辑（简化）
x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
x = (x - x_mean) / (x_std + 1e-5)    # z-score 标准化
x = np.clip(x, -self.clip, self.clip) # 裁剪异常值

# ... 预测 ...

preds = preds * (x_std + 1e-5) + x_mean  # 反标准化
```

**为什么使用实例级标准化？** 不同股票的价格量级差异巨大（如贵州茅台 ~2000 元 vs ST 股票 ~1 元）。实例级标准化将每条序列独立归一化，使模型能泛化到任意价格范围。`clip=5` 的选择基于经验值，保留约 99.7% 的正常数据点（3 个标准差范围），同时抑制极端异常值的影响。

**为什么不用全局标准化？** 全局标准化需要预先计算整个市场的统计量，且不同股票的价格范围差异过大会导致某些股票被压缩到极小值。实例级标准化让每条序列都占据 [-clip, clip] 的完整范围，最大化利用模型的表达能力。

---

## 数据质量对预测的影响

不同的数据质量问题会对预测结果产生不同影响。了解这些影响有助于你在准备数据时抓住重点：

| 数据问题 | 对预测的影响 | 严重程度 |
|---------|-------------|---------|
| 包含 NaN | 程序直接报错，无法运行 | 致命（必须修复） |
| 开盘价为 0（停牌日） | 标准化后产生极端值，预测可能异常跳变 | 高（强烈建议修复） |
| 最高价 < 最低价 | 违反基本金融逻辑，模型行为不可预测 | 高（必须修复） |
| 数据未按时间排序 | 模型学习到错误的时间模式，预测完全失效 | 高（必须修复） |
| 缺少 volume/amount | 不影响价格预测，但成交相关特征缺失 | 低（自动补零处理） |
| lookback 不足 64 | 历史信息不够，预测可能退化为均值输出 | 中（建议 ≥ 200） |
| 数据时间跨度过短 | 模型无法捕捉长期周期性规律 | 中（建议 ≥ 数月数据） |

---

## 不同数据源的注意事项

| 数据源 | 列名特点 | 常见问题 | 处理建议 |
|--------|---------|---------|---------|
| CSV 文件（英文列名） | 通常已经是 `open/high/low/close` | 可能缺少 `timestamps` 列 | 确保有时间列并转为 datetime |
| akshare（A 股） | 中文列名（"开盘"、"收盘"等） | 停牌日开盘价为 0 | 重命名列 + 修复零值 |
| Yahoo Finance | 英文列名，可能有 Adj Close | 列名可能为 `Adj Close` 而非 `close` | 选择正确的价格列 |
| 自有数据库 | 取决于表结构 | 可能有额外列或缺失列 | 只提取需要的 6 列 |
| 加密货币交易所 | 通常为英文列名 | 24/7 交易，无"交易日"概念 | 时间戳连续，无需跳过周末 |

---

## 常见数据问题与处理

### 问题 1：数据中存在 NaN

KronosPredictor 会在预测前检查 NaN，如果发现会直接报错：

```python
if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
    raise ValueError("Input DataFrame contains NaN values in price or volume columns.")
```

**处理方法**：

```python
# 方法 1：前向填充（推荐）
df = df.ffill()

# 方法 2：用前一根 K 线的收盘价填充开盘价
df.loc[df['open'].isna(), 'open'] = df['close'].shift(1)
```

### 问题 2：开盘价为 0 或异常

某些数据源中停牌日可能记录为 0：

```python
# 用前一日收盘价替代异常开盘价
bad_open = (df['open'] == 0) | (df['open'].isna())
df.loc[bad_open, 'open'] = df['close'].shift(1)
```

### 问题 3：只有价格没有成交量

```python
# 只提供四列即可，KronosPredictor 会自动处理
x_df = df[['open', 'high', 'low', 'close']]
```

### 问题 4：历史窗口长度不足

`lookback` 建议设置为 **64-512** 之间。模型支持的最大上下文长度为 512（`max_context` 参数）。低于 64 的历史长度可能导致预测质量下降——模型需要足够的历史数据来捕捉价格波动模式，64 个时间点大约能覆盖一个完整的交易日（5 分钟数据）或约三个月的日线数据。

---

## 🧪 动手练习

### 练习 1：验证你的数据格式

用以下脚本检查你的数据是否符合 Kronos 输入要求：

```python
required_cols = ['open', 'high', 'low', 'close']
optional_cols = ['volume', 'amount']

# 检查必填列是否存在
missing = [c for c in required_cols if c not in df.columns]
assert not missing, f"缺少必填列：{missing}"

# 检查 NaN
assert not df[required_cols].isnull().any().any(), "必填列中存在 NaN"

# 检查时间戳类型
assert pd.api.types.is_datetime64_any_dtype(df['timestamps']), "timestamps 列不是 datetime 类型"

print("数据格式校验通过！")
```

**验证方法**：如果输出 `数据格式校验通过！`，说明你的数据格式正确。如果抛出 `AssertionError`，根据提示修复对应问题后再运行。

### 练习 2：对比有 NaN 和无 NaN 的数据处理

读取 CSV 后，手动插入一个 NaN 值，然后用 `df.ffill()` 处理，确认 NaN 被消除：

```python
df_test = df.copy()
df_test.loc[10, 'close'] = float('nan')
print("处理前 NaN 数量：", df_test['close'].isna().sum())

df_test = df_test.ffill()
print("处理后 NaN 数量：", df_test['close'].isna().sum())
```

**验证方法**：如果处理后输出 `NaN 数量：0`，说明前向填充生效。

---

## 你学到了什么

1. **数据格式**：至少需要 `open`、`high`、`low`、`close` 四列，`volume` 和 `amount` 可选
2. **时间戳**：必须提供历史和未来的时间戳，模型从中提取时间特征
3. **自动规范化**：KronosPredictor 内部完成标准化和反标准化，无需手动处理
4. **数据质量**：输入数据中不能有 NaN，需要在传入前处理完毕

---

## 常见问题

### Q: 如何验证数据质量？

**A**: 可以使用以下快速验证脚本，对 DataFrame 进行全面检查：

```python
def validate_kronos_data(df):
    """快速验证数据是否符合 Kronos 输入要求"""
    issues = []

    # 1. 检查必填列
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            issues.append(f"缺少必填列：{col}")

    # 2. 检查 NaN
    price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
    nan_counts = df[price_cols].isnull().sum()
    for col, cnt in nan_counts.items():
        if cnt > 0:
            issues.append(f"列 {col} 中有 {cnt} 个 NaN 值")

    # 3. 检查零值（可能是停牌数据）
    for col in price_cols:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            issues.append(f"列 {col} 中有 {zero_count} 个零值（可能是停牌数据）")

    # 4. 检查高低价逻辑
    if all(c in df.columns for c in ['high', 'low']):
        bad_hl = (df['high'] < df['low']).sum()
        if bad_hl > 0:
            issues.append(f"有 {bad_hl} 行的最高价低于最低价")

    if issues:
        print("发现以下数据问题：")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("数据质量检查通过，未发现问题。")

    return len(issues) == 0

# 使用示例
validate_kronos_data(df)
```

如果输出 `数据质量检查通过，未发现问题。`，说明你的数据已符合要求。

---

## ✅ 自测清单

- [ ] 我能解释 Kronos 对输入数据的必填列要求
- [ ] 我能独立从 CSV 或 akshare 准备出符合格式的 DataFrame
- [ ] 我能说出时间戳在预测中的作用（提取时间特征、捕捉周期性）
- [ ] 我能使用 `df.ffill()` 处理数据中的 NaN 值
- [ ] 我能解释为什么 KronosPredictor 使用实例级标准化而非全局标准化

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [项目总览与核心概念](../core-concepts/01-overview.md) | ⭐⭐ | 理解 Kronos 的设计架构 |
| [KronosPredictor 使用指南](../core-concepts/04-predictor.md) | ⭐⭐ | 深入了解预测参数与调优 |

---
**文档元信息**
难度：⭐ | 类型：入门教程 | 预计阅读时间：10 分钟
