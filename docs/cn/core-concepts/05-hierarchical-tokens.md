# 层级令牌体系 ⭐⭐

> **目标读者**：想理解 Kronos 独特的 s1/s2 层级令牌设计的用户和开发者
> **核心问题**：为什么需要两级令牌？它们如何协作？

---

## 学习目标

阅读本文档后，你将能够：

- [ ] 解释层级令牌体系将 20 比特码本拆分为 s1/s2 的动机与数学原理
- [ ] 说明 `HierarchicalEmbedding` 如何将 s1 和 s2 融合为统一向量表示
- [ ] 描述 `DependencyAwareLayer` 中交叉注意力机制如何实现"s2 依赖 s1"
- [ ] 根据 s1_bits/s2_bits 比例调整模型行为

---

## 概念定义

### 一句话定义

Kronos 的**层级令牌体系**将每根 K 线编码为一对令牌 `(s1, s2)`：s1 是粗粒度令牌（捕捉主要走势），s2 是细粒度令牌（在 s1 基础上提供精细修正），两者通过 DependencyAwareLayer 实现条件依赖。

### 为什么需要两级令牌？

单一令牌面临一个困境：

- **令牌太少（码本小）**：表达能力不足，无法精确表示 K 线的变化
- **令牌太多（码本大）**：预测难度急剧增加（分类类别过多）

Kronos 的解决方案是**分层**：将 20 比特的码本拆分为两个 10 比特的子码本（默认配置 `s1_bits=10, s2_bits=10`，此值可通过模型配置调整）。模型先预测 s1（1,024 种选择），再在 s1 的约束下预测 s2（1,024 种选择），将 1,048,576 种组合的预测分解为两次 1,024 选 1 的决策。

**类比理解**：

> s1 就像回答"这根 K 线是大涨、小涨、横盘、小跌还是大跌？"——粗略的方向判断。s2 则是在确定方向后的精确幅度修正。就像先决定走哪个方向，再决定走多远。

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
    def __init__(self, d_model, n_heads=4):
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads)
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

---

## 对比：层级 vs 单层

| 维度 | 单层令牌 | 层级令牌 (s1 + s2) |
|------|----------|-------------------|
| 码本大小 | 2^20 = 1,048,576 | 2 × 2^10 = 2,048 |
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

在继续阅读之前，确认你能回答以下问题：

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

---
**文档元信息**
难度：⭐⭐ | 类型：核心概念 | 更新日期：2026-04-11 | 预计阅读时间：20 分钟
