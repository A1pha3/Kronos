# Kronos 模型详解 ⭐⭐

> **目标读者**：想理解 Kronos 自回归 Transformer 如何进行 K 线预测的用户和开发者
> **核心问题**：模型如何从历史令牌预测未来令牌？层级解码机制如何工作？

---

## 学习目标

阅读本文后，你应该能够：

- [ ] 说明 Kronos 模型的输入输出格式，以及 s1/s2 双头输出的含义
- [ ] 描述层级解码流程——s2 的预测为何条件依赖于 s1 的预测结果
- [ ] 区分 `forward()`、`decode_s1()` 和 `decode_s2()` 三个方法的适用场景
- [ ] 解释时间嵌入（TemporalEmbedding）在模型中的作用

---

## 概念定义

### 一句话定义

**Kronos** 是一个基于 Transformer 的自回归语言模型，接收历史 K 线令牌序列（s1, s2），逐时间步预测未来的 s1 和 s2 令牌。

### 为什么使用自回归模型？

K 线数据具有严格的时间顺序：每一根 K 线的走势都受到之前所有 K 线的影响。自回归模型天然契合这种特性——它按时间顺序逐个生成令牌，每一步的预测都基于之前所有已生成的结果。

与直接回归未来多步的模型相比，自回归模型的优势：

- **逐步细化**：每一步都可以基于最新上下文做出更准确的判断
- **灵活的采样策略**：可以通过温度、top-k、top-p 控制预测的多样性和保守性
- **天然的序列建模能力**：擅长捕捉长距离的时间依赖关系

---

## 模型架构

```
输入令牌 [s1_ids, s2_ids]
      │
      ▼
┌──────────────────────┐
│ HierarchicalEmbedding│  将 s1、s2 令牌嵌入并融合为统一表示
│ (emb_s1 + emb_s2 +  │  定义于 module.HierarchicalEmbedding
│  fusion_proj)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ + TemporalEmbedding   │  加入时间特征（分钟、小时、星期、日、月）
│                       │  定义于 module.TemporalEmbedding
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Token Dropout         │  令牌级 Dropout，防止过拟合
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Transformer Blocks × N          │
│ 定义于 self.transformer          │
│ ┌─────────┐  ┌────────────────┐ │
│ │ RMSNorm │→ │ Self-Attention │ │
│ │         │  │ (with RoPE)    │ │
│ └─────────┘  └────────────────┘ │
│ ┌─────────┐  ┌────────────────┐ │
│ │ RMSNorm │→ │ FeedForward    │ │
│ │         │  │ (SwiGLU)       │ │
│ └─────────┘  └────────────────┘ │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────┐
│ RMSNorm               │  最终归一化
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌──────────────────────┐
│ DualHead│  │ DependencyAwareLayer │
│ proj_s1 │  │ (交叉注意力)         │
│         │  │ s1 → s2 条件依赖    │
│         │  │ 定义于               │
│         │  │ module.Dependency-   │
│         │  │ AwareLayer           │
└────┬────┘  └──────────┬───────────┘
     │                  │
     ▼                  ▼
  s1_logits         s2_logits
```

### 关键参数

以下参数决定模型的规模与表达能力。预训练模型的参数配置存储在 HuggingFace Hub 上的 `config.json` 中，通过 `from_pretrained()` 加载时自动读取。

| 参数 | 含义 | 对模型的影响 |
|------|------|-------------|
| `n_layers` | Transformer 层数 | 层数越多，模型能捕捉越深层的历史模式，但推理时间近似线性增长 |
| `d_model` | 模型维度（隐藏状态维度） | 影响表达力的最关键参数。更大的 `d_model` 意味着更强的表示能力，但也需要更多训练数据 |
| `n_heads` | 注意力头数 | 多头并行捕捉不同子空间的模式。通常 `d_model / n_heads` 为每个头的维度 |
| `ff_dim` | 前馈网络维度 | SwiGLU 中间层维度，影响非线性变换的能力 |
| `s1_bits` | s1 比特数 | s1 词汇表大小为 2^s1_bits，控制粗粒度令牌的表达空间 |
| `s2_bits` | s2 比特数 | s2 词汇表大小为 2^s2_bits，控制细粒度令牌的表达空间 |
| `learn_te` | 时间嵌入是否可学习 | `True` 使用可训练的 Embedding，`False` 使用固定的正弦编码 |
| `token_dropout_p` | 令牌嵌入 Dropout 率 | 训练时随机将令牌嵌入置零的概率，防止过拟合 |

不同规模模型的参数配置有所不同。你可以通过以下方式查看实际值：

```python
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
print(model.config)
# 输出包含：n_layers, d_model, n_heads, ff_dim, s1_bits, s2_bits 等参数
```

各模型的详细参数对比见 [模型对比与选型指南](../advanced/07-model-comparison.md)。Kronos-mini 的参数规模最小（4.1M 总参数），Kronos-base 最大（102.3M 参数）。

> **注意**：`s1_bits` 和 `s2_bits` 在所有预训练模型中均为 10/10，保持一致。这是 BSQ 量化器将 20 维连续向量切分为两个 10 维子空间的配置。

---

## 前向推理：forward()

`forward()` 是模型的核心方法（定义于 `Kronos.forward()`），接收历史令牌和时间特征，输出 s1 和 s2 的预测 logits：

```python
s1_logits, s2_logits = model(s1_ids, s2_ids, stamp=timestamps, padding_mask=None)
```

### 输入参数

| 参数 | 形状 | 说明 |
|------|------|------|
| `s1_ids` | `(batch, seq_len)` | s1 令牌 ID，范围 [0, 2^s1_bits) |
| `s2_ids` | `(batch, seq_len)` | s2 令牌 ID，范围 [0, 2^s2_bits) |
| `stamp` | `(batch, seq_len, 5)` | 时间特征（分钟、小时、星期、日、月） |
| `padding_mask` | `(batch, seq_len)` 或 None | 填充掩码，1 表示填充位置 |

### 输出

| 输出 | 形状 | 说明 |
|------|------|------|
| `s1_logits` | `(batch, seq_len, 2^s1_bits)` | s1 令牌的未归一化概率分布 |
| `s2_logits` | `(batch, seq_len, 2^s2_bits)` | s2 令牌的未归一化概率分布 |

### 层级解码流程

Kronos 的一个独特设计是 **s2 的预测条件依赖于 s1 的预测结果**。以下流程摘自 `Kronos.forward()`：

```python
# 第 0 步：令牌嵌入 + 时间嵌入 + Dropout
x = self.embedding([s1_ids, s2_ids])
x = x + self.time_emb(stamp)
x = self.token_drop(x)  # 训练时随机将部分令牌嵌入置零

# 第 1 步：Transformer 编码
for layer in self.transformer:
    x = layer(x, key_padding_mask=padding_mask)
x = self.norm(x)

# 第 2 步：预测 s1
s1_logits = self.head(x)                              # DualHead 的 proj_s1

# 第 3 步：从 s1_logits 采样得到 s1_ids
s1_probs = F.softmax(s1_logits.detach(), dim=-1)
sample_s1_ids = torch.multinomial(s1_probs.view(-1, vocab_s1), 1)

# 第 3 步：用 s1 的嵌入作为条件，预测 s2
sibling_embed = self.embedding.emb_s1(sample_s1_ids)
x2 = self.dep_layer(x, sibling_embed)                 # DependencyAwareLayer
s2_logits = self.head.cond_forward(x2)                # DualHead 的 proj_s2
```

**为什么 s2 要依赖 s1？** s1 捕捉了 K 线的主要走势方向。s2 在 s1 的基础上提供精细修正。如果 s2 独立于 s1 预测，可能产生不一致的结果（如 s1 预测大涨但 s2 预测小跌）。通过交叉注意力机制，s2 的预测能够"看到" s1 的结果并做出协调一致的决策。

**训练时的 Teacher Forcing**：在训练阶段，`forward()` 支持通过 `use_teacher_forcing=True` 和 `s1_targets` 参数使用真实的 s1 标签（而非模型采样的 s1）作为 s2 解码的条件。这能加速训练收敛，但可能导致"暴露偏差"——推理时模型必须使用自己采样的 s1，与训练时使用真实 s1 的分布不一致。

```python
# Teacher Forcing 模式（训练时可选）
if use_teacher_forcing:
    sibling_embed = self.embedding.emb_s1(s1_targets)    # 使用真实 s1
else:
    sibling_embed = self.embedding.emb_s1(sample_s1_ids)  # 使用采样 s1（默认）
```

### 梯度截断与采样策略

源码中有一个容易被忽略的细节——`s1_logits.detach()`（`kronos.py:270`）：

```python
s1_probs = F.softmax(s1_logits.detach(), dim=-1)    # 注意 .detach()
sample_s1_ids = torch.multinomial(s1_probs.view(-1, vocab_s1), 1)
```

`.detach()` 的作用是**切断从 s2 预测到 s1 预测的梯度路径**。这意味着：

- s1 的预测头部（`proj_s1`）只通过 s1 的交叉熵损失接收梯度
- s2 的预测头部（`proj_s2`）只通过 s2 的交叉熵损失接收梯度
- s1 的采样结果作为 s2 的条件输入，但不参与 s1 头部的梯度计算

**为什么不把 s2 的梯度传导到 s1？** 如果允许梯度流通，s1 头部会同时收到来自 s1 损失和 s2 损失的信号。由于 s2 条件依赖于 s1 的采样结果，梯度会通过采样操作（`torch.multinomial`）反向传播——但 multinomial 本身是离散操作，梯度无法直接通过。即使使用 `detach()` 截断后用 `multinomial` 采样（而非使用 STE），模型仍然能通过两条独立的损失路径分别优化 s1 和 s2 的预测能力。

---

## 两步解码：decode_s1() + decode_s2()

在推理时，Kronos 使用分步解码以支持高效的 KV-Cache（虽然在当前实现中未显式使用 KV-Cache，但架构为此预留了接口）：

```python
# 第 1 步：解码 s1（定义于 Kronos.decode_s1()）
s1_logits, context = model.decode_s1(s1_ids, s2_ids, stamp=timestamps)
# s1_logits: 预测的 s1 logits
# context: Transformer 的上下文表示，供 decode_s2 使用

# 第 2 步：基于 s1 预测结果解码 s2（定义于 Kronos.decode_s2()）
sample_s1 = sample_from_logits(s1_logits[:, -1, :], ...)
s2_logits = model.decode_s2(context, sample_s1)
```

**为什么分两步？** 在自回归推理中，每个时间步只需要最后一个位置的预测。`decode_s1` 可以返回 Transformer 的上下文表示（`context`），`decode_s2` 复用这个上下文，避免重复计算 Transformer 的前向传播。

---

## 双头输出：DualHead

`DualHead` 模块（定义于 `module.DualHead`）包含两个独立的线性投影层：

```python
class DualHead(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model):
        self.vocab_s1 = 2 ** s1_bits   # s1 词汇表大小
        self.vocab_s2 = 2 ** s2_bits   # s2 词汇表大小
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)
```

- `forward(x)` → `proj_s1(x)` → s1 logits
- `cond_forward(x2)` → `proj_s2(x2)` → s2 logits

两个头独立预测不同粒度的令牌，共享 Transformer 主体提取的特征。

---

## 时间嵌入：TemporalEmbedding

Kronos 将时间信息编码为向量并注入模型（定义于 `module.TemporalEmbedding`）：

```python
# 从时间戳提取 5 个特征
stamp = [minute, hour, weekday, day, month]

# 每个特征独立嵌入，然后求和
time_emb = hour_embed(hour) + weekday_embed(weekday) + day_embed(day)
          + month_embed(month) + minute_embed(minute)
```

时间嵌入与令牌嵌入相加（`x = x + time_embedding`），让模型能够感知时间位置。

**`learn_te=True` 时**：使用可学习的 `nn.Embedding`，模型在训练中自动学习最优的时间编码。

**`learn_te=False` 时**：使用固定的正弦/余弦编码（类似原始 Transformer 的位置编码），不参与训练。

---

## 损失计算

训练时使用交叉熵损失，对 s1 和 s2 分别计算（定义于 `DualHead.compute_loss()`）：

```python
ce_s1 = CrossEntropy(s1_logits, s1_targets)
ce_s2 = CrossEntropy(s2_logits, s2_targets)
loss = (ce_s1 + ce_s2) / 2
```

---

## 动手练习

### 练习 1：对比 s1 和 s2 概率分布的熵

使用 `decode_s1()` 和 `decode_s2()` 分别获取 s1 和 s2 logits，对比 s1 概率分布的熵：

```python
import torch
import torch.nn.functional as F
from model.kronos import Kronos

# 创建模型（使用默认参数）
model = Kronos(
    s1_bits=10, s2_bits=10, n_layers=12,
    d_model=512, n_heads=8, ff_dim=1280,
    ffn_dropout_p=0.1, attn_dropout_p=0.1,
    resid_dropout_p=0.1, token_dropout_p=0.1,
    learn_te=True
)
model.eval()

# 模拟输入
batch_size, seq_len = 2, 32
s1_ids = torch.randint(0, 2**10, (batch_size, seq_len))
s2_ids = torch.randint(0, 2**10, (batch_size, seq_len))
stamp = torch.rand(batch_size, seq_len, 5)

# 使用分步解码
s1_logits, context = model.decode_s1(s1_ids, s2_ids, stamp=stamp)

# 从 s1 采样
s1_probs = F.softmax(s1_logits[:, -1, :], dim=-1)
sample_s1 = torch.multinomial(s1_probs, 1)  # (batch, 1)

# 扩展为完整序列长度用于 decode_s2
sample_s1_full = sample_s1.expand(batch_size, seq_len)
s2_logits = model.decode_s2(context, sample_s1_full)

# 计算熵
def compute_entropy(probs):
    """计算概率分布的熵（以 2 为底）"""
    log_probs = torch.log2(probs + 1e-10)
    return -(probs * log_probs).sum(dim=-1)

s1_entropy = compute_entropy(F.softmax(s1_logits[:, -1, :], dim=-1))
s2_entropy = compute_entropy(F.softmax(s2_logits[:, -1, :], dim=-1))

print(f"s1 平均熵: {s1_entropy.mean().item():.2f} bits (最大 {10.0})")
print(f"s2 平均熵: {s2_entropy.mean().item():.2f} bits (最大 {10.0})")
# s1 的熵通常较低（预测更确定），s2 的熵通常较高（精细修正更不确定）
```

---

## 自测清单

阅读完毕后，检验你是否理解了以下要点：

- [ ] 能否说明 `forward()` 中 s2 预测依赖 s1 预测的具体流程（采样 → 嵌入 → 交叉注意力）？
- [ ] 能否解释 `decode_s1()` 和 `decode_s2()` 相比直接调用 `forward()` 的优势？
- [ ] 能否描述 `HierarchicalEmbedding` 如何将两个离散令牌融合为一个连续向量？
- [ ] 能否说明 `DependencyAwareLayer` 中 query、key、value 分别来自哪里？
- [ ] 能否解释为什么 `s1_logits` 使用 detach 后再采样（避免梯度回传到 s2 分支）？

---

## 场景对比

| 场景 | 推荐方法 | 说明 |
|------|---------|------|
| 训练（完整序列） | `model.forward()` | 一次性获取整个序列的 s1 和 s2 logits |
| 自回归推理（逐时间步） | `model.decode_s1()` + `model.decode_s2()` | 分步解码，避免重复计算 Transformer 主体 |
| 批量推理 | `model.forward()` + KV-Cache | 架构预留接口，当前未显式使用 |
| Teacher Forcing 训练 | `model.forward(use_teacher_forcing=True, s1_targets=...)` | 使用真实 s1 标签代替采样结果 |

---

## 常见误区

### 误区 1：Kronos 是一个回归模型

**正确理解**：Kronos 是一个**分类模型**。它不直接回归 OHLCV 数值，而是在离散令牌空间中进行分类预测。从"预测哪个令牌"到"预测具体数值"的转换由分词器的解码器完成。

### 误区 2：模型越大效果越好

**正确理解**：模型规模与效果的关系取决于数据的复杂度和任务需求。对于分钟级数据或短期预测，small 模型可能已经足够。更大的模型在复杂场景（如长周期预测、多市场泛化）中优势更明显。关于各模型的详细对比与选型建议，参见[模型对比与选型](../advanced/07-model-comparison.md)。

### 误区 3：s1 和 s2 的预测完全独立

**正确理解**：s2 的预测条件依赖于 s1 的采样结果。在 `Kronos.forward()` 中，s2 通过 `DependencyAwareLayer` 的交叉注意力机制获取 s1 的嵌入信息，确保两者协调一致。

---

## 知识关联

- **前置**：[项目总览](01-overview.md) ⭐⭐ — 理解两阶段框架
- **相关**：[层级令牌体系](05-hierarchical-tokens.md) ⭐⭐ — s1/s2 的设计与 DependencyAwareLayer
- **进阶**：[Transformer 设计分析](../architecture/03-transformer-design.md) ⭐⭐⭐⭐ — RoPE、SwiGLU 等设计决策
- **相关**：[KronosPredictor 使用指南](04-predictor.md) ⭐⭐ — 模型推理的实际调用方式
- **参考**：[源码走读](../architecture/04-source-code-walkthrough.md) ⭐⭐⭐⭐ — Kronos 类的逐行解读

---
**文档元信息**
难度：⭐⭐ | 类型：核心概念 | 更新日期：2026-04-11 | 预计阅读时间：20 分钟
