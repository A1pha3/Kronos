# 批量预测指南 ⭐⭐⭐

> **目标读者**：想同时对多只股票/多个时间窗口进行预测的用户
> **前置要求**：已掌握 [KronosPredictor](../core-concepts/04-predictor.md) 的基本使用

### 学习目标

阅读本文后，你将能够：

- [ ] 使用 `predict_batch()` 对多条时间序列进行批量预测
- [ ] 理解批量预测的等长约束及其背后的张量堆叠机制
- [ ] 根据显存和速度需求合理设置 batch 数量和 `sample_count`

---

## 概述

`KronosPredictor.predict_batch()` 允许你同时对多条时间序列进行预测，利用 GPU 的并行计算能力显著提升吞吐量。

---

## 基本用法

```python
from model import Kronos, KronosTokenizer, KronosPredictor
import pandas as pd

# 加载模型
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 准备多条数据
df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

# 构建多条序列
dfs = []
xtsp = []
ytsp = []
for i in range(5):
    start = i * 400
    end = start + lookback

    dfs.append(df.loc[start:end-1, ['open', 'high', 'low', 'close', 'volume', 'amount']])
    xtsp.append(df.loc[start:end-1, 'timestamps'])
    ytsp.append(df.loc[end:end+pred_len-1, 'timestamps'])

# 批量预测
pred_dfs = predictor.predict_batch(
    df_list=dfs,
    x_timestamp_list=xtsp,
    y_timestamp_list=ytsp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# pred_dfs 是 DataFrame 列表，每个元素对应一条序列的预测结果
for i, pred_df in enumerate(pred_dfs):
    print(f"序列 {i}: 预测 {len(pred_df)} 步")
```

---

## 约束条件

批量预测有一个关键约束：**所有序列的历史长度必须相同**。

| 条件 | 要求 |
|------|------|
| 所有 `df` 的行数 | 必须一致 |
| 所有 `y_timestamp` 的长度 | 必须等于 `pred_len` |
| `df_list`、`x_timestamp_list`、`y_timestamp_list` 的长度 | 必须一致 |

**原因**：批量预测在内部将多条序列 stack 为一个三维张量 `(batch, seq_len, features)`，要求 `seq_len` 维度一致。

**为什么这个约束是必要的？** 如果尝试将不同长度的序列 stack，PyTorch 会抛出维度不匹配错误。例如，一条长度为 400 的序列和一条长度为 300 的序列无法拼接为形状统一的张量——这就像试图将一个 400 列的表格和一个 300 列的表格按行合并，列数不同会导致对齐失败。

### 如果序列长度不同

如果各序列的历史长度不同，需要分别调用 `predict()`：

```python
results = []
for df_i, x_ts_i, y_ts_i in zip(df_list, x_ts_list, y_ts_list):
    pred = predictor.predict(df=df_i, x_timestamp=x_ts_i, y_timestamp=y_ts_i, pred_len=pred_len)
    results.append(pred)
```

---

## 内部实现

`predict_batch()` 的核心处理流程：

```
1. 逐条验证输入
   └─ 检查列名、NaN、长度一致性

2. 逐条标准化
   └─ 每条序列独立计算 mean/std 并保存

3. Stack 为批量张量
   └─ x_batch: (B, seq_len, 6)
   └─ x_stamp_batch: (B, seq_len, 5)
   └─ y_stamp_batch: (B, pred_len, 5)

4. 调用 generate()
   └─ 内部自动处理 sample_count 的维度扩展

5. 逐条反标准化
   └─ 使用各自保存的 mean/std 还原

6. 返回 DataFrame 列表
```

### sample_count 与批量的交互

`predict_batch` 内部调用 `generate()`，而 `generate()` 会将 `sample_count` 扩展到 batch 维度：

```python
# auto_regressive_inference 中的维度扩展
x = x.unsqueeze(1).repeat(1, sample_count, 1, 1)
x = x.reshape(-1, x.size(1), x.size(2))  # (B*sample_count, seq_len, features)
```

因此，`predict_batch(df_list=5, sample_count=3)` 的实际 batch 大小为 15。需注意 GPU 显存是否足够。

---

## 性能建议

### 显存优化

| 策略 | 说明 |
|------|------|
| 减小 batch 数量 | 减少同时处理的序列数 |
| 减小 sample_count | 降低采样次数 |
| 减小 lookback | 缩短历史窗口（但不要低于 64） |
| 使用更小的模型 | `Kronos-small` 比 `Kronos-large` 占用更少显存 |

### 速度优化

```python
# 快速但不那么稳定
pred_dfs = predictor.predict_batch(
    df_list=dfs,
    x_timestamp_list=xtsp,
    y_timestamp_list=ytsp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,      # 单次采样
    verbose=False         # 关闭进度条
)

# 稳定但较慢
pred_dfs = predictor.predict_batch(
    df_list=dfs,
    x_timestamp_list=xtsp,
    y_timestamp_list=ytsp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=5,      # 多次采样取平均
    verbose=True
)
```

---

## 可视化批量预测结果

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(pred_dfs), 1, figsize=(12, 4 * len(pred_dfs)), sharex=True)

for i, (pred_df, ax) in enumerate(zip(pred_dfs, axes)):
    lookback_df = dfs[i]
    pred_df.index = range(len(lookback_df), len(lookback_df) + len(pred_df))

    ax.plot(range(len(lookback_df)), lookback_df['close'], label='Historical', color='blue')
    ax.plot(pred_df.index, pred_df['close'], label='Prediction', color='red')
    ax.set_title(f'Sequence {i+1}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
```

---

## 常见问题

### Q: 批量预测比循环调用 `predict()` 快多少？

**A**: 取决于 GPU 利用率。对于小 batch，加速比可能不明显（因为单次预测已充分利用 GPU）。对于 10+ 条序列，批量预测通常能带来 2-3 倍的速度提升。

### Q: 批量预测中 OOM（显存不足）怎么办？

**A**: 减小 batch 数量或 `sample_count`。也可以分批处理：

```python
batch_size = 3
all_preds = []
for i in range(0, len(dfs), batch_size):
    batch_preds = predictor.predict_batch(
        df_list=dfs[i:i+batch_size],
        x_timestamp_list=xtsp[i:i+batch_size],
        y_timestamp_list=ytsp[i:i+batch_size],
        pred_len=pred_len
    )
    all_preds.extend(batch_preds)
```

---

## 动手练习

### 练习 1：预测 5 条不同时间窗口的数据并绘制对比图

使用项目自带的测试数据，选取 5 个不同起始位置，批量预测后绘制对比图：

```python
from model import Kronos, KronosTokenizer, KronosPredictor
import pandas as pd
import matplotlib.pyplot as plt

# 加载模型和数据
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 200
pred_len = 60

# 构建不同起始位置的 5 条序列
dfs, xtsp, ytsp = [], [], []
for i in range(5):
    start = i * 250  # 每条间隔 250 行，确保有重叠但不完全相同
    end = start + lookback
    dfs.append(df.loc[start:end-1, ['open', 'high', 'low', 'close', 'volume', 'amount']])
    xtsp.append(df.loc[start:end-1, 'timestamps'])
    ytsp.append(df.loc[end:end+pred_len-1, 'timestamps'])

pred_dfs = predictor.predict_batch(
    df_list=dfs, x_timestamp_list=xtsp, y_timestamp_list=ytsp,
    pred_len=pred_len, sample_count=1, verbose=True
)

# 绘制对比图
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=False)
for i, (pred_df, ax) in enumerate(zip(pred_dfs, axes)):
    ax.plot(range(lookback), dfs[i]['close'].values, label='Historical', color='blue')
    ax.plot(range(lookback, lookback + pred_len), pred_df['close'].values, label='Prediction', color='red')
    ax.set_title(f'Window {i+1} (start={i*250})')
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig('batch_comparison.png', dpi=150)
plt.show()
```

**验证方法**：确认生成了包含 5 个子图的 `batch_comparison.png`，每个子图都包含蓝色历史段和红色预测段，且 5 条序列的起始位置各不相同。

---

## 自测清单

完成本指南后，检查你是否掌握了以下要点：

- [ ] 能解释为什么批量预测要求所有序列的历史长度一致
- [ ] 知道 `sample_count` 参数如何影响实际 batch 大小（batch 数 * sample_count）
- [ ] 遇到 OOM 时能写出分批处理的代码
- [ ] 能根据场景选择合适的速度/稳定性权衡策略

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [A 股市场预测实战](04-cn-markets.md) | ⭐⭐⭐ | 完整的 A 股预测示例 |
| [KronosPredictor 使用指南](../core-concepts/04-predictor.md) | ⭐⭐ | 详细参数说明 |

---
**文档元信息**
难度：⭐⭐⭐ | 类型：进阶指南 | 预计阅读时间：15 分钟
