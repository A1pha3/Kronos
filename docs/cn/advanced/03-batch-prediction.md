# 批量预测指南 ⭐⭐⭐

> **目标读者**：想同时对多只股票/多个时间窗口进行预测的用户
> **前置要求**：已掌握 [KronosPredictor](../core-concepts/04-predictor.md) 的基本使用

## 学习目标

这篇文档讲解批量预测的用法和约束：

- [ ] 使用 `predict_batch()` 对多条时间序列进行批量预测
- [ ] 解释批量预测的等长约束及其背后的张量堆叠机制
- [ ] 根据显存和速度需求合理设置 batch 数量和 `sample_count`
- [ ] 权衡 `predict()` 与 `predict_batch()` 的性能差异，选择适合当前场景的调用方式

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

**为什么这个约束是必要的？** 如果尝试将不同长度的序列 stack，PyTorch 会抛出维度不匹配错误。例如，一条长度为 400 的序列和一条长度为 300 的序列无法拼接为形状统一的张量——`seq_len` 维度不一致，`torch.stack()` 要求所有张量在每个维度上大小相同。

### 如果序列长度不同

如果各序列的历史长度不同，需要分别调用 `predict()`：

```python
results = []
for df_i, x_ts_i, y_ts_i in zip(df_list, x_ts_list, y_ts_list):
    pred = predictor.predict(df=df_i, x_timestamp=x_ts_i, y_timestamp=y_ts_i, pred_len=pred_len)
    results.append(pred)
```

> **为什么没有自动填充（padding）机制？** 当前实现不支持自动填充，因为不同长度的序列需要不同的填充掩码和独立的反标准化处理，增加了实现复杂度。如需处理不同长度的序列，推荐使用循环调用 `predict()` 的方式。

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
# auto_regressive_inference 中的维度扩展（源码 kronos.py:394）
# 注意：这是一条链式语句，Python 先求值右侧表达式再赋值给 x
# 因此 x.size(1) 和 x.size(2) 取的是 repeat 之前（原始）的维度
x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2))
# 结果: (B*sample_count, seq_len, features)
```

因此，`predict_batch` 传入 5 条序列且 `sample_count=3` 时，实际 batch 大小为 15。需注意 GPU 显存是否足够。

### 显存估算公式

批量推理的显存占用可以近似估算：

```
显存占用 ≈ 实际 batch × seq_len × d_model × n_layers × 4 字节 × 4（前向+反向+激活+中间）
```

以 `Kronos-small`（d_model=512, n_layers=8）为例：

| 配置 | 实际 batch | 预估显存 |
|------|-----------|---------|
| 1 条序列, sample=1 | 1 | ~1 GB |
| 5 条序列, sample=1 | 5 | ~2 GB |
| 5 条序列, sample=3 | 15 | ~4 GB |
| 10 条序列, sample=5 | 50 | ~8 GB+（可能 OOM） |

**安全策略**：当不确定显存是否足够时，使用分批处理将大批量拆分为小批量（见下方"分批处理"代码示例），逐步增加批量大小直到接近显存上限。

### 按 GPU 显存推荐批量大小

以下推荐基于 `Kronos-base` 模型、`lookback=400`、`pred_len=120` 的配置：

| GPU 显存 | 推荐 batch 数（sample_count=1） | 推荐 batch 数（sample_count=3） |
|---------|-------------------------------|-------------------------------|
| 4 GB | 1-2 | 1 |
| 8 GB | 3-5 | 1-2 |
| 16 GB | 8-15 | 3-5 |
| 32 GB | 20-30 | 8-10 |

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

**CPU 环境参考**：在没有 GPU 的机器上，批量预测同样可以工作，但速度较慢。在 M1 MacBook（CPU 模式）上，使用 `Kronos-small` 对 10 条序列（`lookback=400`、`pred_len=120`、`sample_count=1`）进行批量预测约需 3-5 分钟。实测值取决于数据量和 CPU 性能，建议先用小批量（2-3 条）测试耗时，再按比例估算全量预测时间。

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

## 常见误区

### 误区：batch_size 越大越好

增大批量可以提升 GPU 利用率，但存在上限。当 batch 过大时：

- **显存溢出（OOM）**：实际 batch = 序列数 x `sample_count`。例如 20 条序列 x `sample_count=3` = 实际 batch 60，可能直接超出 GPU 显存。
- **吞吐量不增反降**：当 batch 大小超过 GPU 流处理器数量的某个比例后，调度开销增加，单步推理耗时上升，整体吞吐反而下降。
- **没有统一的"最优 batch"**：最优值取决于模型规模、序列长度、`sample_count` 和 GPU 型号。建议从推荐值（见上文"按 GPU 显存推荐批量大小"表格）开始，逐步向上调整并观察显存占用和推理速度。

**实践建议**：先从 `batch_size=3`、`sample_count=1` 开始跑通流程，确认显存裕量后再逐步增大。

### 误区：批量预测和循环调用结果完全一致

批量预测（`predict_batch`）和循环调用 `predict()` 的数学流程相同——独立标准化、独立生成、独立反标准化——但由于 GPU 并行计算中浮点运算的累加顺序不同，两种方式在数值上可能存在**微小差异**（通常在 1e-6 量级）。

这类差异对预测结果的影响可以忽略不计，但如果你的下游逻辑对数值精度有严格要求（例如与某个 golden output 做逐位比对），需要注意这一点。

---

## 何时选择批量预测 vs 循环调用

根据以下条件选择合适的调用方式：

| 条件 | 批量预测 `predict_batch()` | 循环调用 `predict()` |
|------|---------------------------|---------------------|
| 序列数量 | >= 5 条，希望利用 GPU 并行 | < 5 条，或逐条处理即可 |
| 序列长度 | 所有序列长度一致 | 长度不一致，或需要每条独立处理 |
| GPU 显存 | 充足（>= 8 GB），可容纳整个批量 | 有限（<= 4 GB），逐条更安全 |
| 结果一致性 | 需要尽量一致的处理时序（同时标准化、同时生成） | 需要严格逐条隔离，互不影响 |
| 代码复杂度 | 需要预先对齐序列长度 | 无额外约束，直接循环即可 |

**决策流程**：

```
是否所有序列长度一致？
├─ 否 → 使用循环调用 predict()
└─ 是
    ├─ 序列数量 <= 3？
    │   └─ 使用循环调用 predict()（批量优势不明显）
    └─ 序列数量 > 3
        ├─ GPU 显存 >= 8 GB？
        │   └─ 使用 predict_batch()，参考推荐批量大小分批
        └─ GPU 显存 < 8 GB 或不确定
            └─ 先用小批量（2-3 条）测试，显存足够再增大
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

使用项目自带的测试数据，选取 5 个不同起始位置，批量预测后绘制对比图。

先按上文"基本用法"中的代码加载模型和数据，然后构建序列并预测：

```python
# 假设模型和数据已按"基本用法"一节加载完毕
lookback = 200
pred_len = 60

# 构建不同起始位置的 5 条序列
dfs, xtsp, ytsp = [], [], []
for i in range(5):
    start = i * 250
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

### 练习 2：实现分批处理避免 OOM

将练习 1 中的批量预测改写为分批处理版本——每次只处理 2 条序列，循环处理所有数据：

```python
batch_size = 2
all_preds = []
for i in range(0, len(dfs), batch_size):
    batch_preds = predictor.predict_batch(
        df_list=dfs[i:i+batch_size],
        x_timestamp_list=xtsp[i:i+batch_size],
        y_timestamp_list=ytsp[i:i+batch_size],
        pred_len=pred_len,
        sample_count=1,
        verbose=True
    )
    all_preds.extend(batch_preds)
```

在此基础上做以下修改并观察效果：
1. 将 `batch_size` 改为 1，比较总运行时间与 `batch_size=2` 的差异
2. 将 `sample_count` 改为 3，观察单批次的显存占用变化

**验证方法**：分批处理的结果数量应等于原始序列数（5 条）。对比 `batch_size=1` 和 `batch_size=2` 的结果应基本一致（允许采样带来的微小差异）。如果 GPU 显存允许，`sample_count=3` 的结果应比 `sample_count=1` 更稳定。

---

## 自测清单

- [ ] 能解释为什么批量预测要求所有序列的历史长度一致
- [ ] 知道 `sample_count` 参数如何影响实际 batch 大小（batch 数 * sample_count）
- [ ] 遇到 OOM 时能写出分批处理的代码
- [ ] 能根据场景选择合适的速度/稳定性权衡策略
- [ ] 了解批量预测与循环调用的结果可能存在微小数值差异
- [ ] 能根据序列数量、长度一致性和显存条件选择 `predict_batch()` 或循环调用 `predict()`

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [A 股市场预测实战](04-cn-markets.md) | ⭐⭐⭐ | 完整的 A 股预测示例 |
| [KronosPredictor 使用指南](../core-concepts/04-predictor.md) | ⭐⭐ | 详细参数说明 |

## 相关文档

- **前置**：[KronosPredictor 使用指南](../core-concepts/04-predictor.md) — 单条预测的基础用法
- **实战**：[A 股市场预测实战](04-cn-markets.md) — 批量预测 A 股多只个股
- **实战**：[使用场景与实战案例](06-use-cases.md) — 多场景模拟与置信区间
- **工具**：[Web UI 使用指南](05-webui-guide.md) — 浏览器界面进行预测
- **参考**：[模型选型指南](07-model-comparison.md) — 选择适合批量预测的模型规模

