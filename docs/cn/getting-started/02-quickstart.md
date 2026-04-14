# 快速开始：第一个预测 ⭐

> **目标读者**：已完成环境安装，想快速体验 Kronos 的用户
> **预计时间**：15 分钟
> **前置要求**：[安装与环境配置](01-installation.md) 已完成（建议在虚拟环境中操作）

---

## 学习目标

完成本教程后，你将能够：

- [ ] 使用预训练模型进行 K 线预测
- [ ] 理解预测流程的三个阶段：加载模型 → 准备数据 → 执行预测
- [ ] 看懂预测结果的含义
- [ ] 独立调整温度参数和采样次数来控制预测效果

---

## 核心概念速览

在动手之前，你需要知道一件事：**Kronos 的预测过程分为三个阶段**。

```
原始 OHLCV 数据 → KronosTokenizer（分词器）编码 → Kronos（模型）自回归推理 → 解码回 OHLCV 数据
```

- **KronosTokenizer**：将连续的 K 线数据压缩为离散的"令牌"序列
- **Kronos**：根据历史令牌序列，预测未来的令牌序列
- **KronosPredictor**：将上面两步封装为一个简单的 `predict()` 方法

日常使用中，你只需要跟 `KronosPredictor` 打交道。

---

## 步骤指南

### 步骤 1：加载模型和分词器（预计 2 分钟）

Kronos 的预训练模型托管在 HuggingFace Hub 上，首次运行会自动下载：

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# 加载分词器
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

# 加载预测模型（这里使用 small 版本，速度与效果的平衡点）
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 创建预测器（自动检测并使用最佳设备）
predictor = KronosPredictor(model, tokenizer, max_context=512)
```

**代码解释**：

- `from_pretrained()` 从 HuggingFace Hub 下载并加载预训练权重，返回已初始化的模型实例
- `KronosPredictor` 接收模型和分词器，封装了完整的预测流程。`max_context=512` 是模型支持的最大上下文窗口长度
- 如果不指定 `device` 参数，`KronosPredictor` 会按 CUDA → MPS → CPU 的顺序自动选择

> **注意**：如果使用 Kronos-mini，需要搭配专用分词器 `Kronos-Tokenizer-2k` 并将 `max_context` 设为 2048。详见 [模型对比与选型](../advanced/07-model-comparison.md)。

### 验证点 1

运行后应看到类似输出（首次运行会有下载进度条）：

```
Downloading: 100%|██████████| xxx/xxx [00:xx<00:00, xx kB/s]
```

### 步骤 2：准备数据（预计 5 分钟）

Kronos 接受包含 OHLCV 数据的 pandas DataFrame。你需要准备两份数据：

- **历史数据**（`x_df`）：模型基于这段历史进行预测
- **未来时间戳**（`y_timestamp`）：告诉模型要预测哪些时间点

```python
import pandas as pd

# 读取数据文件
df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# 设置参数
lookback = 400   # 使用 400 根历史 K 线
pred_len = 120   # 预测未来 120 根 K 线

# 提取历史数据（模型输入）
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']

# 提取未来时间戳（告诉模型预测的起始位置）
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
```

**数据格式要求**：

| 列名 | 类型 | 含义 | 是否必填 |
|------|------|------|----------|
| `open` | float | 开盘价 | 必填 |
| `high` | float | 最高价 | 必填 |
| `low` | float | 最低价 | 必填 |
| `close` | float | 收盘价 | 必填 |
| `volume` | float | 成交量 | 可选（缺失时自动填充 0） |
| `amount` | float | 成交额 | 可选（缺失时自动推算） |

> 如果你的数据只有 OHLC 四列，没有 `volume` 和 `amount`，KronosPredictor 会自动补零，不影响价格预测。这是因为模型在预训练阶段已经学习了处理缺失成交量数据的模式——标准化后的零值不会主导价格特征的建模。

### 步骤 3：执行预测（预计 5 分钟）

```python
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,      # 预测步数
    T=1.0,                  # 温度参数（1.0 = 标准采样）
    top_p=0.9,              # 核采样阈值
    sample_count=1,         # 采样次数（多次采样取平均可提升稳定性）
    verbose=True            # 显示生成进度条
)
```

**参数说明**：

- **`T`（温度）**：控制预测的随机性。`T=1.0` 是标准采样；降低温度（如 `0.5`）使预测更保守、更确定；升高温度使预测更多样化
- **`top_p`（核采样）**：保留累计概率达到 `top_p` 的候选令牌。`0.9` 表示从概率前 90% 的令牌中采样
- **`sample_count`**：多次采样后取平均值可以减少随机性。`sample_count=5` 表示生成 5 个预测并取均值

### 验证点 2

预测完成后，`pred_df` 是一个包含预测结果的 DataFrame：

```python
print(pred_df.head())
```

预期输出（具体数值取决于数据）：

<!-- 以下仅为示意，具体输出取决于你的数据和 y_timestamp 输入 -->

```
                     open    high     low   close    volume    amount
timestamps
2024-01-15 10:00:00  12.35  12.48  12.30  12.42  1234567  15234567
2024-01-15 10:05:00  12.42  12.55  12.38  12.50   987654  12123456
...
```

### 步骤 4：可视化结果（预计 2 分钟）

```python
import matplotlib.pyplot as plt

# 将预测结果与历史数据对齐
pred_df.index = df.index[lookback:lookback+pred_len]

# 绘制收盘价对比图
plt.figure(figsize=(10, 4))
plt.plot(df.loc[:lookback+pred_len-1, 'close'], label='Ground Truth', color='blue')
plt.plot(pred_df['close'], label='Prediction', color='red')
plt.xlabel('Time Step')
plt.ylabel('Close Price')
plt.legend()
plt.title('Kronos Prediction Result')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## 如何解读预测结果

预测完成后，`pred_df` 是一个标准的 DataFrame，包含 6 列 OHLCV 数据。以下是正确解读结果的几个关键点：

### 预测的是概率分布，不是精确价格

Kronos 的输出是**基于历史模式的概率性预测**。每次运行可能产生不同的结果（除非使用贪婪解码 `top_k=1`）。这不是缺陷——它反映了金融市场的不确定性本质。你应该关注**趋势方向和波动范围**，而非追求精确数值。

### 预测不确定性随步数递增

| 预测范围 | 可信度 | 说明 |
|---------|--------|------|
| 前 10-20 步 | 较高 | 短期趋势通常与历史惯性一致，模型在此范围内的预测最具参考价值 |
| 20-60 步 | 中等 | 趋势方向有参考价值，具体数值偏差增大，建议结合多次采样结果判断 |
| 60 步以上 | 较低 | 仅供参考，不确定性显著增大，适合用于极端场景分析而非具体决策 |

**为什么不确定性会随步数递增？** 这是因为自回归模型每一步的预测结果都成为下一步的输入，微小的偏差会逐步传播和放大。在离散令牌空间中，每一步的错误不会像连续值那样在数值上叠加，但预测路径仍会逐步偏离真实轨迹。

建议将 `pred_len` 设置在你真正需要的范围内，不要盲目追求长步数预测。

### 多次采样可以估计不确定性

将 `sample_count` 设为 10 并分别观察每条预测路径的分布，可以得到预测的"置信区间"——路径越分散，说明模型越不确定。这比单次预测能提供更丰富的信息：

```python
# 生成 10 条独立路径
paths = []
for _ in range(10):
    pred = predictor.predict(df=x_df, x_timestamp=x_timestamp,
                            y_timestamp=y_timestamp, pred_len=60,
                            T=1.0, top_p=0.9, sample_count=1, verbose=False)
    paths.append(pred['close'].values)

import numpy as np
paths = np.array(paths)  # 形状: (10, 60)
mean_path = paths.mean(axis=0)       # 平均预测
p5_path = np.percentile(paths, 5, axis=0)   # 90% 置信区间下界
p95_path = np.percentile(paths, 95, axis=0)  # 90% 置信区间上界
```

如果置信区间（p5 到 p95）很窄，说明模型对走势比较有信心；如果很宽，说明市场处于高不确定状态。

### 预测结果的边界条件

> **注意**：当 `lookback` 超过模型的 `max_context`（默认 512）时，最早的历史数据会被静默截断。确保 `lookback <= max_context` 以充分利用所有历史数据。

以下情况可能导致预测结果异常，需要特别注意：

| 情况 | 表现 | 原因 | 处理方法 |
|------|------|------|---------|
| 历史数据过短（<64 根） | 预测接近平坦直线 | 模型缺乏足够上下文 | 增加 `lookback` 到至少 200 |
| 数据含极端异常值 | 预测出现巨大跳变 | 标准化后异常值被 `clip` 截断，信息丢失 | 先处理异常值再传入 |
| 温度过低（T<0.1） | 预测为确定性重复模式 | 采样退化为贪婪选择 | 将 T 提高到 0.5 以上 |
| 预测步数过长（>200） | 后半段预测退化为均值 | 长程不确定性累积 | 关注前 60 步的结果 |

### 判断预测是否正常的快速检查

```python
# 检查预测结果是否在合理范围内
last_close = x_df['close'].iloc[-1]
pred_close = pred_df['close']

print(f"最后一根历史收盘价: {last_close:.2f}")
print(f"预测收盘价范围: [{pred_close.min():.2f}, {pred_close.max():.2f}]")
print(f"预测波动幅度: {(pred_close.max() - pred_close.min()) / last_close * 100:.1f}%")

# 如果预测波动幅度异常大（如超过 ±50%），可能需要调整参数
```

**正常输出特征**：预测曲线的波动幅度与历史数据相当，趋势方向与历史走势有逻辑关联，数值不会出现极端跳变。

---

## 完整代码

以下是可以直接运行的完整脚本：

```python
import pandas as pd
import matplotlib.pyplot as plt
from model import Kronos, KronosTokenizer, KronosPredictor

# === 1. 加载模型 ===
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# === 2. 准备数据 ===
df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# === 3. 执行预测 ===
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

# === 4. 查看结果 ===
print("预测结果前 5 行：")
print(pred_df.head())
```

---

## 🧪 动手练习

### 练习 1：调整温度参数，观察预测差异

在完整代码的基础上，分别设置 `T=0.3` 和 `T=1.5` 各运行一次预测，对比两次 `pred_df['close'].std()` 的值。

**验证方法**：如果 `T=1.5` 时的收盘价标准差明显大于 `T=0.3` 时的标准差，说明你做对了——高温带来更大的随机性。

### 练习 2：使用多次采样稳定预测

将 `sample_count` 从 1 改为 5，重新运行预测，观察预测曲线是否更加平滑。

**验证方法**：如果 `sample_count=5` 的预测结果中 `close` 列的标准差小于 `sample_count=1` 时的标准差，说明多次采样确实降低了随机波动。

---

## 你学到了什么

1. **三阶段流程**：Kronos 的预测经历"分词器编码 → 模型自回归生成 → 解码回原始数据"三个阶段，但 `KronosPredictor` 将其封装为一次 `predict()` 调用
2. **数据格式**：输入是包含 OHLCV 列的 DataFrame，输出也是同样格式的 DataFrame
3. **采样参数**：温度 `T` 和 `top_p` 控制预测的随机性与多样性
4. **采样次数**：增大 `sample_count` 可以通过多次采样取均值来提升预测稳定性

---

## 常见问题

### Q: 首次运行很慢？

**A**: 首次运行需要从 HuggingFace Hub 下载模型权重。`Kronos-small` 约 50 MB，`Kronos-base` 约 200 MB。下载完成后，后续运行会使用缓存。

### Q: 预测结果每次都不同？

**A**: 这是正常现象。Kronos 使用采样策略生成预测，具有随机性。设置 `sample_count=5` 或更高可以取多次采样的平均值，使结果更稳定。如果需要完全确定的结果，可以设置 `T=0.1` 和 `top_k=1`（贪婪解码）。

### Q: 只想预测价格，没有成交量数据怎么办？

**A**: 只需在 DataFrame 中提供 `open`、`high`、`low`、`close` 四列即可。KronosPredictor 会自动将 `volume` 和 `amount` 填充为 0。

### Q: 预测结果看起来像一条直线？

**A**: 可能的原因包括：（1）`lookback` 设置过小（建议不低于 64），模型没有足够的历史信息来捕捉波动规律；（2）温度 `T` 设置过低（如 0.01），导致模型几乎退化为确定性输出。尝试将 `lookback` 提高到 200 以上，并将 `T` 设为 0.8-1.2 之间。更多诊断方法参见 [常见问题与故障排查](../references/troubleshooting.md)。

### Q: 运行时提示 OOM（内存不足）怎么办？

**A**: OOM 通常由以下原因导致：（1）`lookback` 过大——尝试降低到 200 或更小；（2）`sample_count` 过大——先从 1 开始测试；（3）`pred_len` 过长——缩短预测步数。如果使用 GPU，可以在 `KronosPredictor` 中显式指定 `device="cpu"` 来避免显存不足。

### Q: 如何卸载或清理模型缓存？

**A**: Kronos 通过 HuggingFace Hub 加载模型，下载的权重文件会缓存在本地。默认缓存位置取决于操作系统：

| 操作系统 | 缓存路径 |
|----------|----------|
| macOS / Linux | `~/.cache/huggingface/hub/` |
| Windows | `C:\Users\<用户名>\.cache\huggingface\hub\` |

**查看缓存占用**：

```bash
# macOS / Linux
du -sh ~/.cache/huggingface/hub/
```

**清理特定模型**：

```bash
# 删除 Kronos 相关缓存（模型名以 NeoQuasar 开头）
rm -rf ~/.cache/huggingface/hub/models--NeoQuasar--Kronos-small
rm -rf ~/.cache/huggingface/hub/models--NeoQuasar--Kronos-Tokenizer-base
```

**清理全部 HuggingFace 缓存**：

```bash
# 谨慎操作：这会删除所有通过 HuggingFace Hub 下载的模型
rm -rf ~/.cache/huggingface/hub/
```

**自定义缓存位置**：如果磁盘空间紧张，可以通过环境变量将缓存指向其他路径：

```bash
export HF_HOME=/data/huggingface_cache    # 设置 HuggingFace 根目录
export HF_HUB_CACHE=/data/hf_hub_cache    # 单独设置 Hub 缓存目录
```

清理后首次运行 `from_pretrained()` 会重新下载模型权重。

---

## ✅ 自测清单

- [ ] 我能解释 Kronos 预测的三个阶段（分词 → 推理 → 解码）
- [ ] 我能独立完成从加载模型到获取预测结果的完整流程
- [ ] 我能说出温度参数 `T` 和 `top_p` 各自的作用
- [ ] 我能解释 `sample_count` 参数如何影响预测的稳定性
- [ ] 我能根据输出判断预测是否正常运行（检查 DataFrame 格式和数值合理性）

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [数据准备指南](03-data-preparation.md) | ⭐ | 深入了解数据格式与预处理 |
| [项目总览与核心概念](../core-concepts/01-overview.md) | ⭐⭐ | 理解两阶段框架的设计思想 |
| [KronosPredictor 使用指南](../core-concepts/04-predictor.md) | ⭐⭐ | 掌握参数调优与批量预测 |

---
**文档元信息**
难度：⭐ | 类型：入门教程 | 预计阅读时间：15 分钟 | 最后更新：2026-04-11
