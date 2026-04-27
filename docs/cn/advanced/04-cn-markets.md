# A 股市场预测实战 ⭐⭐⭐

> **目标读者**：想对 A 股个股进行 K 线预测的用户
> **前置要求**：[快速开始](../getting-started/02-quickstart.md) 已完成

### 学习目标

这篇展示 A 股日 K 线预测的完整流程：

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

### A 股市场的特殊性

在使用 Kronos 预测 A 股之前，了解以下市场特征有助于正确解读预测结果：

| 特征 | 说明 | 对预测的影响 |
|------|------|-------------|
| T+1 交易制度 | 当日买入的股票次日才能卖出 | 日内数据不受影响，但日线数据反映的是 T+1 约束下的交易行为 |
| 涨跌停限制 | 主板 ±10%，创业板/科创板 ±20% | 模型预测不受此限制，需要后处理裁剪 |
| 集合竞价 | 开盘和收盘前有集合竞价时段 | 可能产生跳空缺口，影响 OHLC 数据的连续性 |
| 停牌机制 | 个股可能因各种原因暂停交易 | 停牌日数据可能记录为开盘价 0、成交量 0，需要预处理 |
| 涨跌停封板 | 涨停/跌停时买卖单不平衡 | 可能导致成交量与价格变动的非线性关系 |
| 复权处理 | 除权除息导致价格跳跃（前复权/后复权/不复权） | 见下方详细说明 |

#### 复权说明

脚本使用 `adjust=""`（不复权）获取原始价格数据。这是因为 Kronos 模型学习的是原始价格模式，而复权操作会改变历史价格的真实数值。如果使用前复权数据，除权日之前的所有价格都会被调整，可能引入与实际交易价格不一致的模式。

| 复权方式 | akshare 参数 | 特点 | 是否推荐用于 Kronos |
|---------|-------------|------|-------------------|
| 不复权 | `adjust=""` | 原始交易价格，保留除权跳跃 | 推荐——与模型预训练数据一致 |
| 前复权 | `adjust="qfq"` | 以最新价格为基准回溯调整 | 不推荐——历史价格被人为修改 |
| 后复权 | `adjust="hfq"` | 以上市价格为基准向前调整 | 不推荐——价格数值偏离实际交易价 |

> **例外**：如果你对某只分红频繁的股票进行长期预测（如 5 年以上日线），不复权数据中除权日的价格跳跃可能干扰模型。此时可以尝试前复权数据，但需要意识到这是与预训练分布不同的输入。

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

# 转换日期并排序
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 数值列清洗：去除千分位逗号、替换缺失值标记、转为数值类型
numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace({"--": None, "": None})
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 修复异常开盘价（停牌日记录为 0 或 NaN）
bad_open = (df["open"] == 0) | (df["open"].isna())
df.loc[bad_open, "open"] = df["close"].shift(1)
df["open"].fillna(df["close"], inplace=True)  # 处理首行 shift 产生的 NaN

# 修复缺失成交额
if df["amount"].isna().all() or (df["amount"] == 0).all():
    df["amount"] = df["close"] * df["volume"]
```

#### 停牌数据处理说明

停牌日的数据可能表现为以下几种形式：

| 情况 | 数据表现 | 脚本处理方式 |
|------|---------|-------------|
| 完全停牌 | `open=0, close=0, volume=0` | `open` 被替换为前一日 `close`，但 `volume=0` 和 `close=0` 仍保留 |
| 部分数据缺失 | `open=0, close=正常值` | `open` 被替换为前一日 `close` |
| 数据源未记录停牌 | 该日无数据行 | 无需处理——日期序列中自然跳过 |

> **重要提示**：当前脚本仅修复了 `open=0` 的情况。如果停牌导致 `close=0` 且 `volume=0`，这些异常行可能影响模型预测质量。对于长时间停牌（如超过 5 个交易日）的个股，建议在数据预处理阶段完整移除停牌期间的所有行，或使用 `df = df[df["volume"] > 0]` 过滤掉无交易的日期。

#### 推荐预处理：完整数据清洗代码

以下代码片段整合了所有清洗步骤，可直接复制使用：

```python
import pandas as pd

def clean_a_share_data(df):
    """A 股日 K 线数据完整预处理（可直接复制使用）"""
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]

    # 1. 数值列清洗：去除千分位逗号、缺失值标记
    for col in numeric_cols:
        if df[col].dtype == object:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .replace({"--": None, "": None})
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. 移除停牌/无交易日（volume=0 的行）
    df = df[df["volume"] > 0].copy()

    # 3. 修复开盘价异常（open=0 或 NaN → 用前一日收盘价填充）
    bad_open = (df["open"] == 0) | (df["open"].isna())
    df.loc[bad_open, "open"] = df["close"].shift(1)
    df["open"].fillna(df["close"], inplace=True)

    # 4. 处理剩余 NaN（前向填充 + 回填）
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # 5. 修复缺失成交额
    if df["amount"].isna().all() or (df["amount"] == 0).all():
        df["amount"] = df["close"] * df["volume"]

    df.reset_index(drop=True, inplace=True)
    return df
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

如需跳过中国法定节假日，可使用 `chinese_calendar` 库：

```bash
pip install chinese_calendar
```

```python
import chinese_calendar
# 过滤掉节假日和周末
future_dates = pd.bdate_range(start=..., periods=pred_len*2)  # 多生成一些再过滤
future_dates = future_dates[~future_dates.map(lambda d: chinese_calendar.is_holiday(d.date()))][:pred_len]
```

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

A 股有 ±10% 的涨跌停限制。这一规则来自中国证监会的监管要求：主板股票当日涨跌幅不得超过前一日收盘价的 ±10%，涨停时只能买入无法卖出，跌停时只能卖出无法买入。

不同板块的涨跌停幅度不同：

| 板块 | 涨跌停幅度 | 适用股票 | limit_rate 设置 |
|------|-----------|---------|----------------|
| 主板 | ±10% | 大部分 A 股 | `0.1`（默认） |
| 创业板 | ±20% | 300xxx 开头 | `0.2` |
| 科创板 | ±20% | 688xxx 开头 | `0.2` |
| ST / *ST | ±5% | 特别处理股票 | `0.05` |
| 北交所 | ±30% | 8xxxxx 开头 | `0.3` |

脚本会逐根 K 线检查并裁剪预测结果：

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

> **为什么脚本默认使用 120 步？** 日线数据的每一步噪声比分钟线数据低（日 K 线聚合了一整天的交易信息，信噪比更高），因此在一定程度上支持更长的预测视野。然而，这并不意味着 120 步预测具有与 20 步预测同等的可靠性。用户应将超过 60 步的结果视为趋势性参考，而非精确预测。

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

### 练习 2：预测创业板股票并应用 ±20% 涨跌停限制

创业板（股票代码以 `300` 开头）的涨跌停幅度为 ±20%，与主板的 ±10% 不同。尝试预测一只创业板股票：

```bash
# 预测宁德时代（300750）
python examples/prediction_cn_markets_day.py --symbol 300750
```

在脚本中将 `limit_rate` 修改为 `0.2`：

```python
# 修改前
pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

# 修改后
pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.2)
```

**验证方法**：检查输出的 CSV 文件中，相邻两根预测 K 线的收盘价变化率不超过 20%。如果股票代码以 `688` 开头（科创板），涨跌停幅度同为 ±20%，处理方式相同。

> **进阶挑战**：修改脚本，使其根据股票代码自动判断板块并设置对应的 `limit_rate`（提示：根据 `symbol` 的前缀判断——`300`/`688` 开头为 0.2，`688` 开头为 0.2，`8` 开头为 0.3，其余为 0.1）。

---

## 自测清单

- [ ] 能解释数据清洗步骤中为什么需要修复开盘价为 0 的记录
- [ ] 知道 `pd.bdate_range` 生成的日期是否包含法定节假日（答案：可能包含）
- [ ] 理解 `apply_price_limits` 中为什么用修正后的收盘价作为下一根 K 线的基准
- [ ] 能针对 ST 股票和创业板分别设置正确的 `limit_rate` 值
- [ ] 知道如何通过 `SAMPLE_COUNT` 和 `PRED_LEN` 平衡预测的稳定性和实用性
- [ ] 理解为什么脚本使用不复权数据（`adjust=""`），以及何时可能需要使用复权数据

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [批量预测指南](03-batch-prediction.md) | ⭐⭐⭐ | 同时预测多只股票 |
| [CSV 微调指南](02-finetune-csv.md) | ⭐⭐⭐ | 用 A 股数据微调模型 |

## 相关文档

- **前置**：[快速开始](../getting-started/02-quickstart.md) — 基础预测流程
- **前置**：[KronosPredictor 使用指南](../core-concepts/04-predictor.md) — predict() 参数详解
- **并行**：[批量预测指南](03-batch-prediction.md) — 同时预测多只股票
- **微调**：[CSV 微调指南](02-finetune-csv.md) — 用 A 股数据微调提升效果
- **微调**：[Qlib 微调指南](01-finetune-qlib.md) — 使用 Qlib 平台微调

