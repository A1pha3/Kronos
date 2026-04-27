# 层级令牌体系 ⭐⭐

> **目标读者**：想理解 Kronos 独特的 s1/s2 层级令牌设计的用户和开发者
> **核心问题**：为什么需要两级令牌？它们如何协作？

---

## 学习目标

理解这些内容后，调整层级令牌配置会更有依据：

- [ ] 解释层级令牌体系将 20 比特码本拆分为 s1/s2 的动机与数学原理
- [ ] 说明 `HierarchicalEmbedding` 如何将 s1 和 s2 融合为统一向量表示
- [ ] 描述 `DependencyAwareLayer` 中交叉注意力机制如何实现"s2 依赖 s1"
- [ ] 根据 s1_bits/s2_bits 比例调整模型行为

---

## 概念定义

Kronos 的**层级令牌体系**将每根 K 线编码为一对令牌 `(s1, s2)`：s1 是粗粒度令牌（捕捉主要走势），s2 是细粒度令牌（在 s1 基础上提供精细修正），两者通过 DependencyAwareLayer 实现条件依赖。

单一令牌面临一个困境：码本太小则表达能力不足，码本太大则预测难度急剧增加（分类类别过多）。Kronos 的解决方案是**分层**——将 20 比特的码本拆分为两个 10 比特的子码本（默认配置 `s1_bits=10, s2_bits=10`）。模型先预测 s1（1,024 种选择），再在 s1 的约束下预测 s2（1,024 种选择），将 1,048,576 种组合的预测分解为两次 1,024 选 1 的决策。

从信息论角度看：如果一次预测 20 比特的完整码本，模型需要在 2^20 个类别上学习一个概率分布——数据有限时大多数类别的概率估计接近零。拆分为两个 10 比特预测后，每个子任务只需要在 1024 个类别上建立概率分布，每个类别获得更多训练样本，估计更稳定。代价是两次独立预测的组合无法精确表达子码本之间的相关性，但通过 `DependencyAwareLayer` 的条件依赖机制，这种相关性被部分恢复。

---

## 令牌结构

### 码本维度

BSQ 量化器将 20 维连续向量量化为 20 维二值向量 `{-1, +1}^20`（默认配置 `codebook_dim = s1_bits + s2_bits = 20`）。这 20 个比特被分为两部分：

```
┌─────────────────────── 20 bits ───────────────────────┐
│  s1_bits (例如 10)    │    s2_bits (例如 10)          │
│  粗粒度令牌           │    细粒度令牌                  │
│  2^10 = 1024 种       │    2^10 = 1024 种             │
└───────────────────────┴───────────────────────────────┘
```

### 索引计算

二值向量转换为整数索引的方式：

```python
# 二值向量 z_q ∈ {-1, +1}^20
# 转换为 0/1：b = (z_q + 1) / 2
# 索引 = Σ b_i × 2^(i)

# s1 索引：使用前 s1_bits 位
s1_index = sum(b[i] * 2^i for i in range(s1_bits))

# s2 索引：使用后 s2_bits 位
s2_index = sum(b[s1_bits + i] * 2^i for i in range(s2_bits))
```

---

## HierarchicalEmbedding（层级嵌入层）

Kronos 模型使用 `HierarchicalEmbedding` 将 s1 和 s2 令牌转换为统一的向量表示：

```python
class HierarchicalEmbedding(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model=256):
        vocab_s1 = 2 ** s1_bits     # s1 词汇表大小：1024
        vocab_s2 = 2 ** s2_bits     # s2 词汇表大小：1024
        self.emb_s1 = nn.Embedding(vocab_s1, d_model)   # s1 嵌入
        self.emb_s2 = nn.Embedding(vocab_s2, d_model)   # s2 嵌入
        self.fusion_proj = nn.Linear(d_model * 2, d_model)  # 融合投影
```

### 嵌入过程

```python
def forward(self, token_ids):
    # 1. 分别获取 s1 和 s2 的嵌入
    s1_emb = self.emb_s1(s1_ids) * sqrt(d_model)
    s2_emb = self.emb_s2(s2_ids) * sqrt(d_model)

    # 2. 拼接后线性融合
    return self.fusion_proj(concat([s1_emb, s2_emb], dim=-1))
```

**设计要点**：

- s1 和 s2 使用**独立的嵌入表**，不共享参数。这允许模型为不同粒度的令牌学习不同的表示空间
- 嵌入向量乘以 `sqrt(d_model)` 进行缩放，与 Transformer 的标准做法一致
- 拼接后通过线性层降维，将两个独立的表示融合为统一的向量

### 接受两种输入格式

`HierarchicalEmbedding` 支持两种输入方式：

```python
# 方式 1：分别提供 s1_ids 和 s2_ids（模型推理时使用）
embedding = layer([s1_ids, s2_ids])

# 方式 2：提供复合令牌 ID（内部自动拆分）
embedding = layer(composite_ids)  # composite_id = s1_id << s2_bits | s2_id
```

方式 2 中的 `split_token()` 方法将复合 ID 拆分为 s1 和 s2：

```python
s2_ids = composite_id & ((1 << s2_bits) - 1)   # 取低 s2_bits 位
s1_ids = composite_id >> s2_bits                 # 右移 s2_bits 位
```

---

## DependencyAwareLayer（依赖感知层）

这是实现"s2 依赖 s1"的关键模块：

```python
class DependencyAwareLayer(nn.Module):
    def __init__(self, d_model, n_heads=4):  # 参数已简化，实际构造器还包含 attn_dropout_p、resid_dropout 等
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads)  # 简化示意
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        # sibling_embed 是 s1 的嵌入向量
        attn_out = self.cross_attn(
            query=sibling_embed,      # s1 嵌入作为 query
            key=hidden_states,        # Transformer 上下文作为 key
            value=hidden_states       # Transformer 上下文作为 value
        )
        return self.norm(hidden_states + attn_out)  # 残差连接 + 归一化
```

### 工作原理

```
Transformer 输出 (hidden_states)
      │
      │    s1 采样结果 → emb_s1 → sibling_embed
      │                              │
      │                              ▼ (query)
      ├──────────→ Cross-Attention ──┤
      │             (key, value)     │
      │                              │
      │    ←─────────────────────────┘
      │         attn_output
      │
      ▼ (残差连接)
hidden_states + attn_output
      │
      ▼ (RMSNorm)
      │
      ▼
proj_s2 → s2_logits
```

**设计意图**：s1 的嵌入向量作为 query 去"查询" Transformer 的上下文，找到与 s1 预测结果最相关的信息，然后基于这些信息预测 s2。这确保了 s2 的预测与 s1 的预测保持一致。

### s1 预测错误如何传播到 s2

由于 s2 的预测通过 `DependencyAwareLayer` 条件依赖于 s1 的采样结果，s1 的错误会不可避免地影响 s2。以下是错误传播的具体机制：

1. **s1 采样错误**：当 s1 被采样为一个"错误"的令牌（与真实的 s1 不同）时，`embedding.emb_s1(sample_s1_ids)` 会产生一个与真实方向不一致的嵌入向量。
2. **交叉注意力的影响**：`DependencyAwareLayer` 将错误的 s1 嵌入作为 query 去检索 Transformer 的上下文。即使上下文本身是正确的，query 的偏差可能导致注意力权重聚焦在"不相关"的上下文位置上，使 s2 的预测基于了错误的信息子集。
3. **级联效应**：在自回归推理中，错误的 `(s1, s2)` 对会被填入缓冲区，成为后续时间步的输入。这意味着一个 s1 错误不仅影响当前步的 s2，还通过缓冲区影响所有后续时间步的预测。

**缓解机制**：Kronos 通过以下设计来限制错误传播的严重程度：
- **`detach()` 阻断梯度**：训练时 s1 头部只通过 s1 的交叉熵损失学习，不会因 s2 的错误而获得混乱的梯度信号
- **训练时可选 Teacher Forcing**：使用 `use_teacher_forcing=True` 时，s2 的条件输入是真实的 s1 而非采样的 s1，这有助于 Transformer 主体学习更鲁棒的特征
- **多采样取平均**：`sample_count > 1` 时，不同路径上的 s1 错误方向不同，取平均后误差部分抵消

**实际观察**：在训练良好的模型中，s1 的预测准确率通常高于 s2（因为 s1 的分类任务相对简单——1024 个类别覆盖了粗粒度方向）。s1 的 top-1 准确率如果较低，会显著拉低整体预测质量，因为所有后续的 s2 预测都建立在有缺陷的基础上。

---

## 推理时的层级解码

在实际推理中，层级解码按以下顺序进行：

```
时间步 t：

1. 将历史 [s1_1..s1_t, s2_1..s2_t] 送入 Transformer
2. 取最后一个位置的 Transformer 输出
3. 通过 proj_s1 得到 s1_logits
4. 采样得到 s1_{t+1}
5. 将 s1_{t+1} 嵌入后送入 DependencyAwareLayer
6. 通过 proj_s2 得到 s2_logits
7. 采样得到 s2_{t+1}
8. 将 s1_{t+1} 和 s2_{t+1} 加入历史缓冲区
9. 继续下一个时间步
```

**Teacher Forcing（训练时）**：

训练时可以选择使用真实的 s1 目标（而非采样的 s1）作为 s2 解码的条件：

```python
if use_teacher_forcing:
    sibling_embed = self.embedding.emb_s1(s1_targets)  # 使用真实 s1
else:
    sibling_embed = self.embedding.emb_s1(sample_s1_ids)  # 使用采样 s1
```

Teacher Forcing 在训练中加速收敛，但可能导致"暴露偏差"——推理时模型必须使用自己采样的 s1，与训练时使用真实 s1 的分布不一致。

---

## 调参建议

### s1_bits / s2_bits 比例对模型行为的影响

`HierarchicalEmbedding` 的 `s1_bits` 和 `s2_bits` 参数决定了粗细粒度令牌的信息容量分配。它们的总和固定为 20（BSQ 量化器的总比特数），因此调整比例本质上是**重新分配表达能力**：

| 配置 (s1_bits / s2_bits) | s1 词汇量 | s2 词汇量 | 行为特征 |
|--------------------------|-----------|-----------|----------|
| 10 / 10（默认） | 1,024 | 1,024 | 粗细均衡，适合大多数场景 |
| 12 / 8 | 4,096 | 256 | s1 表达力更强，模型倾向先做更精确的粗判断，s2 只做小幅修正 |
| 8 / 12 | 256 | 4,096 | s1 粗略定方向，s2 提供更多细节；s2 预测难度增大 |
| 14 / 6 | 16,384 | 64 | s1 几乎完成所有编码工作，s2 仅做微小调整 |

**调参原则**：

- **增大 s1_bits**：让模型在第一级就做出更精细的判断。适用于波动较大、需要精确方向判断的市场。代价是 s1 的分类空间变大，训练需要更多数据来覆盖。
- **增大 s2_bits**：让模型在第一级只做粗略分类，依赖第二级完成精细修正。适用于波动较小、方向明确但幅度多变的市场。代价是 s2 需要在更大空间中搜索。
- **保持均衡（默认 10/10）**：这是当前公开预训练模型使用的配置，文档和代码示例也都围绕这一配置展开

> **注意**：修改 `s1_bits` / `s2_bits` 后需要**重新定义模型结构并重新训练**。因为 `HierarchicalEmbedding` 的嵌入表维度（`vocab_s1 = 2^s1_bits`，`vocab_s2 = 2^s2_bits`）、`DualHead` 的分类头维度、`KronosTokenizer` 的两个后量化线性层（`post_quant_embed_pre` 和 `post_quant_embed`）都会随之改变。预训练模型均基于 10/10 配置，无法直接兼容不同的比特分配。

### 修改 s1_bits/s2_bits 对训练动态的影响

比特分配比例的变化不仅影响模型结构，还会从根本上改变训练过程中的优化动态：

**增大 s1_bits（如 12/8）**：
- s1 的分类空间从 1024 扩大到 4096，每个类别可获得的训练样本减少。在没有足够数据量的情况下，s1 头部的交叉熵损失可能难以充分收敛。
- `post_quant_embed_pre` 的输入维度从 10 变为 12，这个线性层需要从更少的比特中学习映射到 `d_model` 维度——但由于 s1 比特携带更多信息，粗粒度重建（`z_pre`）的质量应该提高。
- `DependencyAwareLayer` 的条件信号变得更丰富（s1 的 4096 种嵌入提供了更精细的条件），s2 只需要在 256 个类别中做小幅修正，s2 的预测难度降低。

**增大 s2_bits（如 8/12）**：
- s1 只在 256 个类别中做粗略方向判断，s1 的预测变得更容易收敛。
- 但 s2 需要在 4096 个类别中搜索，且 `DependencyAwareLayer` 的条件信号更粗略（s1 只有 256 种嵌入），s2 需要从更弱的条件线索中做出更精细的判断。这可能导致 s2 的交叉熵损失较高。
- `post_quant_embed` 的输入维度不变（始终是 `s1_bits + s2_bits = 20`），但解码路径的"重心"从 s1 转移到了 s2。

**对损失收敛的影响**：修改比特分配后，需要重新平衡分词器的两路重建损失。例如，当 s1_bits 增大时，`z_pre` 的重建质量应该更高（因为 s1 携带更多信息），因此 `recon_loss_pre` 的权重可以适当增大，以充分利用 s1 的信息容量。

---

## 对比：层级 vs 单层

| 维度 | 单层令牌 | 层级令牌 (s1 + s2) |
|------|----------|-------------------|
| 码本大小 | 2^20 = 1,048,576（单一词汇表） | s1 词汇量 2^10 = 1,024，s2 词汇量 2^10 = 1,024（两个独立词汇表，合计 2,048 个码字） |
| 预测难度 | 百万级分类 | 千级分类 × 2 |
| 精细度控制 | 无法区分粗细信息 | s1 粗粒度 + s2 精细 |
| 条件依赖 | 无 | s2 条件依赖 s1 |

### 从结构上如何理解层级令牌

这里优先只讲源码能证明的事实：

1. **分词器侧**：`BSQuantizer` 会输出总共 `s1_bits + s2_bits` 个二值位，`half=True` 时会把它拆成两组索引
2. **模型侧**：`DualHead` 分别负责 `s1` 与 `s2` 的分类输出
3. **生成顺序**：`decode_s1()` 先产出 `s1` logits，`decode_s2()` 再读取 `s1` 嵌入作为条件
4. **修改成本**：一旦你改动 `s1_bits` / `s2_bits`，嵌入层、分类头、分词器后量化层和预训练权重兼容性都会一起变化

这已经足以解释为什么它被称为“层级令牌体系”。至于它在理论上是否优于单层方案、优多少，需要论文或实验来支持，不能只靠代码结构直接下结论。

---

## 实际影响：对预测质量和推理的影响

### s1 错误 vs s2 错误的视觉表现

在实际预测中，s1 和 s2 的错误有不同的外在表现，了解这些差异有助于诊断模型行为：

| 错误类型 | 外在表现 | 原因 |
|---------|---------|------|
| s1 系统性偏差 | 预测趋势方向与实际相反（该涨时跌） | s1 令牌捕捉主走势，s1 编码错误导致方向判断偏离 |
| s1 随机噪声 | 多次采样间趋势方向不一致（有时涨有时跌） | s1 采样在几个高概率令牌间摇摆，说明方向信号弱 |
| s2 错误 | 方向大致正确但幅度偏差大 | s2 提供精细修正，s2 误差不影响方向但影响具体数值 |

### 推理延迟

层级令牌体系对推理速度有直接影响。在每一步自回归推理中：

1. 先执行一次完整的 Transformer 前向传播 + s1 头部计算
2. 采样 s1 后，再执行 `DependencyAwareLayer` 的交叉注意力 + s2 头部计算

s2 的计算（`DependencyAwareLayer` + `proj_s2`）额外增加了约 10-15% 的每步推理时间。这一开销是固定的，不受 `s1_bits` / `s2_bits` 比例影响，因为它们只改变嵌入表和分类头的大小，不改变交叉注意力的维度。

---

## 常见误区

### 误区 1：s1 和 s2 是独立预测的

**正确理解**：s2 的预测**依赖**于 s1 的预测结果。`DependencyAwareLayer` 通过交叉注意力让 s2 的预测"看到" s1 的采样结果，确保两者协调一致。

### 误区 2：s1 只预测方向，s2 只预测幅度

**正确理解**：s1 和 s2 都是 20 维 BSQ 量化结果的不同切分。s1 对应前 10 个比特（更接近编码器输出的"主要"维度），s2 对应后 10 个比特（"细节"维度）。它们各自编码了 K 线多个方面的信息，而非简单的方向/幅度分离。

---

## 练习与实践

### 编码一段数据，分别只用 s1 和只用 s2 解码，观察重建质量差异

以下代码演示如何分别利用 s1 和 s2 进行部分解码，直观感受两级令牌各自的信息量：

```python
import torch
import numpy as np
from model import Kronos, KronosTokenizer

# 加载模型
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

# 构造一段模拟 K 线数据（6 通道 × 20 个时间步）
torch.manual_seed(42)
x = torch.randn(1, 20, 6)  # 形状: [batch=1, seq_len=20, features=6]

# 编码为层级令牌
s1_indices, s2_indices = tokenizer.encode(x, half=True)
# s1_indices 形状: [1, 20]，s2_indices 形状: [1, 20]

# --- 仅用 s1 解码（s2 置零） ---
s2_zeros = torch.zeros_like(s1_indices)
decoded_s1_only = tokenizer.decode([s1_indices, s2_zeros], half=True)

# --- 仅用 s2 解码（s1 置零） ---
s1_zeros = torch.zeros_like(s2_indices)
decoded_s2_only = tokenizer.decode([s1_zeros, s2_indices], half=True)

# --- 完整解码（s1 + s2） ---
decoded_full = tokenizer.decode([s1_indices, s2_indices], half=True)

# 计算重建误差
def mse(a, b):
    return torch.mean((a - b) ** 2).item()

print(f"仅 s1 重建 MSE: {mse(x, decoded_s1_only):.4f}")
print(f"仅 s2 重建 MSE: {mse(x, decoded_s2_only):.4f}")
print(f"s1+s2 重建 MSE: {mse(x, decoded_full):.4f}")
```

**你应该关注什么**：

- 仅用单侧令牌解码时，重建结果会发生什么变化
- 完整解码与单侧解码在误差上的差别有多大
- 这种差别是否与你自己的数据分布一致

不要把某一种固定排序当成必然结论；更稳妥的做法是直接在你的样本上比较。

---

## 自测清单

- [ ] 层级令牌体系将 20 比特拆分为两次 10 比特预测，这对分类复杂度有何影响？
- [ ] `HierarchicalEmbedding` 中的 `fusion_proj` 层的作用是什么？
- [ ] `DependencyAwareLayer` 中 s1 嵌入作为交叉注意力的哪个角色（query / key / value）？
- [ ] Teacher Forcing 在训练中有什么好处？又可能导致什么问题？
- [ ] 将 s1_bits 从 10 增大到 14，s1 的词汇量变为多少？对训练有什么影响？

---

## 知识关联

- **前置**：[KronosTokenizer 详解](02-tokenizer.md) ⭐⭐ — 理解 BSQ 量化
- **相关**：[Kronos 模型详解](03-model.md) ⭐⭐ — 理解 DualHead 和前向推理
- **进阶**：[BSQ 量化算法原理](../architecture/02-bsq-algorithm.md) ⭐⭐⭐⭐ — 量化的数学推导
- **参考**：[系统架构分析](../architecture/01-system-architecture.md) ⭐⭐⭐⭐ — 理解层级令牌在整体数据流中的位置
