# KronosPredictor 使用指南 ⭐⭐

> **目标读者**：想全面掌握 KronosPredictor 各项功能和参数的用户
> **核心问题**：如何高效使用 KronosPredictor？各参数如何影响预测结果？

---

## 学习目标

阅读本文档后，你将能够：

- [ ] 说明 `KronosPredictor` 的完整预测流水线（从原始数据到预测结果）
- [ ] 正确设置 `T`、`top_p`、`sample_count` 等采样参数并理解其对结果的影响
- [ ] 使用 `predict()` 和 `predict_batch()` 完成单序列与批量预测任务
- [ ] 解释自回归推理过程中的滑动窗口机制与 s1/s2 层级采样流程

---

## 概念定义

### 一句话定义

**KronosPredictor** 是 Kronos 的高层预测接口，将"数据预处理 → 分词编码 → 自回归推理 → 解码还原 → 后处理"的完整流程封装为 `predict()` 和 `predict_batch()` 两个方法。

### 为什么需要 KronosPredictor？

直接使用 KronosTokenizer + Kronos 进行预测需要手动处理大量细节：标准化、时间特征提取、令牌缓冲区管理、采样、反标准化等。KronosPredictor 将这些步骤封装起来，让用户只需关心输入数据和预测参数。

---

## 初始化

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

predictor = KronosPredictor(
    model,              # Kronos 预测模型（必填）
    tokenizer,          # KronosTokenizer 分词器（必填）
    device=None,        # 计算设备（可选，自动检测）
    max_context=512,    # 最大上下文窗口长度
    clip=5.0            # 标准化后的裁剪范围
)
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | Kronos | — | 已加载的预测模型 |
| `tokenizer` | KronosTokenizer | — | 已加载的分词器 |
| `device` | str 或 None | None | 计算设备。`None` 时按 CUDA → MPS → CPU 自动选择 |
| `max_context` | int | 512 | 模型处理的最大令牌序列长度。超过此长度的历史数据会被截断为最近 `max_context` 个 |
| `clip` | float | 5 | 标准化后的值裁剪范围。`clip=5` 表示标准化后的值限制在 [-5, 5] |

---

## predict() — 单序列预测

```python
pred_df = predictor.predict(
    df=x_df,                # 历史数据 DataFrame
    x_timestamp=x_timestamp,# 历史时间戳
    y_timestamp=y_timestamp,# 未来时间戳
    pred_len=120,           # 预测步数
    T=1.0,                  # 温度参数
    top_k=0,                # top-k 采样阈值
    top_p=0.9,              # 核采样阈值
    sample_count=1,         # 采样次数
    verbose=True            # 是否显示进度条
)
```

### 参数详解

#### df（输入数据）

类型：`pandas.DataFrame`，必须包含 `open`、`high`、`low`、`close` 列。`volume` 和 `amount` 列可选。

**自动处理逻辑**：

| 缺失列 | 处理方式 |
|--------|----------|
| `volume` | 填充为 0，`amount` 也填充为 0 |
| `amount`（有 `volume`） | 用 `volume × OHLC 均价` 推算 |
| 任何列含 NaN | 抛出 `ValueError` |

#### x_timestamp 和 y_timestamp

类型：`pandas.Series`（包含 datetime 值）。

- `x_timestamp` 的长度必须与 `df` 的行数一致
- `y_timestamp` 的长度必须等于 `pred_len`

#### T（温度）

控制采样分布的"尖锐程度"。

| 值 | 效果 | 适用场景 |
|----|------|----------|
| 0.1-0.3 | 几乎确定性，总是选概率最高的令牌 | 需要稳定预测结果时 |
| 1.0 | 标准采样，完全按照模型输出的概率分布 | 一般使用 |
| 1.5-2.0 | 更随机，探索更多可能性 | 生成多样化的预测场景 |

#### top_k 和 top_p（采样过滤）

| 参数 | 值 | 行为 |
|------|-----|------|
| `top_k=0` | 禁用 | 不过滤，考虑所有令牌 |
| `top_k=50` | 启用 | 只保留概率最高的 50 个令牌 |
| `top_p=1.0` | 禁用 | 不过滤 |
| `top_p=0.9` | 启用 | 保留累计概率达到 90% 的令牌 |

**推荐组合**：

- **确定性预测**：`top_k=1`（贪婪解码）或 `T=0.1`
- **常规预测**：`T=1.0, top_p=0.9`
- **多样化探索**：`T=1.5, top_p=0.95, sample_count=5`

#### sample_count（采样次数）

每次预测生成 `sample_count` 个独立样本，然后取平均值作为最终结果。

- `sample_count=1`：单次采样，速度快但随机性大
- `sample_count=5`：5 次采样取平均，更稳定但耗时约 5 倍

### 返回值

返回 `pandas.DataFrame`，包含 `open`、`high`、`low`、`close`、`volume`、`amount` 列，以 `y_timestamp` 为索引。

---

### 完整端到端示例

以下示例展示从数据加载到预测输出的完整流程（基于 `examples/prediction_example.py` 精简）：

```python
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

# 1. 加载模型与分词器
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 2. 准备数据
df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400   # 历史窗口长度
pred_len = 120   # 预测步数

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 3. 执行预测
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 4. 查看结果
print("预测结果前 5 行：")
print(pred_df.head())

# 5. 与真实值对比（可选）
true_df = df.loc[lookback:lookback+pred_len-1, ['open', 'high', 'low', 'close']]
mae = (pred_df[['open', 'high', 'low', 'close']] - true_df.values).abs().mean().mean()
print(f"OHLC 平均绝对误差: {mae:.4f}")
```

> **运行提示**：请确保已安装 `pandas`、`torch` 等依赖，且数据文件路径正确。若在 `examples/` 目录下运行，可将路径改为 `./data/XSHG_5min_600977.csv`。

---

## predict_batch() — 批量预测

对多条时间序列并行预测：

```python
pred_dfs = predictor.predict_batch(
    df_list=[df1, df2, df3],            # 多个 DataFrame 的列表
    x_timestamp_list=[ts1, ts2, ts3],    # 对应的历史时间戳列表
    y_timestamp_list=[yt1, yt2, yt3],    # 对应的未来时间戳列表
    pred_len=120,
    T=1.0,
    top_p=0.9,
    sample_count=1
)
# pred_dfs 是 DataFrame 列表，与输入一一对应
```

### 限制条件

批量预测要求所有序列具有**相同的长度**：

- 所有 `df_list` 中的 DataFrame 行数必须一致
- 所有 `y_timestamp_list` 中的长度必须等于 `pred_len`

如果序列长度不同，需要分别调用 `predict()`。

---

## 内部处理流程

`predict()` 方法内部依次执行以下步骤：

```
1. 输入验证
   └─ 检查必填列、NaN 值、补齐可选列

2. 时间特征提取
   └─ 从时间戳提取 minute, hour, weekday, day, month

3. 标准化
   └─ z-score: (x - mean) / (std + 1e-5)
   └─ 裁剪: clip(x, -clip, clip)

4. 分词编码
   └─ tokenizer.encode(x, half=True) → (s1_indices, s2_indices)

5. 自回归推理
   └─ 逐时间步: decode_s1 → 采样 s1 → decode_s2 → 采样 s2
   └─ 滑动窗口管理: 超过 max_context 时截断最早的历史

6. 解码还原
   └─ tokenizer.decode(predicted_indices, half=True) → OHLCV

7. 反标准化
   └─ x_original = x_normalized * (std + 1e-5) + mean

8. 返回 DataFrame
```

---

## 自回归推理细节

Kronos 的自回归推理过程（`auto_regressive_inference()` 函数，定义于 `model/kronos.py`）是预测的核心。它维护一个滑动窗口的令牌缓冲区：

```python
# 以下伪代码为简洁起见省略 batch 维度
# 完整实现中 batch_size = original_batch × sample_count

# ---- 初始化缓冲区 ----
# pre_buffer 形状:  [batch_size, max_context]   ← s1 令牌缓冲区
# post_buffer 形状: [batch_size, max_context]   ← s2 令牌缓冲区
pre_buffer = zeros(max_context)      # s1 令牌缓冲区
post_buffer = zeros(max_context)     # s2 令牌缓冲区

# 填入历史令牌（保留最近 max_context 个）
pre_buffer[:hist_len] = s1_history[-max_context:]    # 形状: [hist_len]
post_buffer[:hist_len] = s2_history[-max_context:]   # 形状: [hist_len]

# ---- 逐步生成 ----
for i in range(pred_len):
    current_seq_len = initial_seq_len + i
    window_len = min(current_seq_len, max_context)

    # 1. 预测 s1
    #    s1_logits 形状: [batch_size, window_len, vocab_s1]
    #    context 形状:   [batch_size, window_len, d_model]
    s1_logits, context = model.decode_s1(
        pre_buffer[:window_len],     # 形状: [window_len]
        post_buffer[:window_len],    # 形状: [window_len]
        current_timestamp            # 形状: [window_len, time_features]
    )
    s1_sample = sample(s1_logits[-1], T, top_k, top_p)   # 形状: 标量

    # 2. 基于 s1 预测 s2
    #    s2_logits 形状: [batch_size, window_len, vocab_s2]
    s2_logits = model.decode_s2(context, s1_sample)
    s2_sample = sample(s2_logits[-1], T, top_k, top_p)   # 形状: 标量

    # 3. 更新缓冲区（滑动窗口）
    if current_seq_len < max_context:
        # 尚未填满缓冲区，直接填入
        buffer[current_seq_len] = s1_sample
    else:
        # 缓冲区已满，左移一位，末尾填入新令牌
        buffer = roll(buffer, -1)
        buffer[-1] = s1_sample
```

当历史长度超过 `max_context` 时，最早的历史令牌被丢弃，只保留最近的 512 个令牌作为上下文。

### 多采样并行机制

`sample_count` 的实现方式是**将采样维度扩展到 batch 维度**，而非串行循环。源码中的关键操作（`kronos.py:394`）：

```python
# 将单条数据复制 sample_count 份，拼接到 batch 维度
x = x.unsqueeze(1).repeat(1, sample_count, 1, 1)
x = x.reshape(-1, x.size(1), x.size(2))
# 变换前: (batch=1, seq_len=400, features=6)
# 变换后: (batch=sample_count, seq_len=400, features=6)
```

这意味着 `sample_count=5` 时，5 条采样路径在 GPU 上**并行计算**，而非串行执行 5 次。推理结束后再 reshape 回 `(1, sample_count, pred_len, 6)` 并沿 `sample_count` 维度取平均。

**实际 batch 大小**：在 `predict_batch()` 中，实际 batch 大小为 `batch_num × sample_count`。例如 `batch_num=3, sample_count=5` 时实际 batch 为 15，需注意 GPU 显存是否足够。

---

## 常见误区

### 误区 1：sample_count 越大越好

**正确理解**：增加采样次数能降低随机性，但边际效益递减。通常 `sample_count=3-5` 已足够。更大的 `sample_count` 会线性增加推理时间。

### 误区 2：更长的历史数据一定更好

**正确理解**：模型最多处理 `max_context=512` 个令牌。超过 512 的历史数据会被截断。增加 `lookback` 超过 512 不会提供额外信息。

---

## 练习与实践

### 尝试不同 sample_count，观察预测结果的标准差变化

多次采样可以平滑随机性，但代价是推理时间。运行以下代码，对比 `sample_count=1` 和 `sample_count=10` 的差异：

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
pred_len = 120
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 对比不同 sample_count
for sc in [1, 3, 5, 10]:
    preds = []
    for _ in range(3):   # 每个设置重复 3 次以衡量稳定性
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=1.0, top_p=0.9, sample_count=sc, verbose=False
        )
        preds.append(pred_df['close'].values)

    preds_array = np.array(preds)                          # 形状: [3, pred_len]
    std_across_runs = preds_array.std(axis=0).mean()       # 3 次运行间的平均标准差
    print(f"sample_count={sc:>2d} | 3 次运行 close 标准差均值: {std_across_runs:.4f}")
```

**预期观察**：`sample_count` 越大，多次运行间的标准差越小（预测更稳定），但推理耗时近似线性增长。

---

## 自测清单

在继续阅读之前，确认你能回答以下问题：

- [ ] `predict()` 方法内部依次执行哪些步骤？（提示：8 步流水线）
- [ ] 如果只想要确定性最高的预测结果，应如何设置 `T` 和 `top_k`？
- [ ] 历史数据长度超过 `max_context=512` 时会发生什么？
- [ ] `sample_count=5` 与 `sample_count=1` 相比，推理时间大约差几倍？
- [ ] `predict_batch()` 对输入数据有什么限制条件？

---

## 知识关联

- **前置**：[快速开始](../getting-started/02-quickstart.md) ⭐ — 基础使用
- **相关**：[Kronos 模型详解](03-model.md) ⭐⭐ — 理解模型内部工作原理
- **进阶**：[批量预测指南](../advanced/03-batch-prediction.md) ⭐⭐⭐ — 批量预测实践

---
**文档元信息**
难度：⭐⭐ | 类型：核心概念 | 预计阅读时间：20 分钟
