# Transformer 设计分析 ⭐⭐⭐⭐

> **目标读者**：想深入理解 Kronos 中 Transformer 各组件设计决策的研究者
> **核心问题**：每个组件为什么这样设计？有哪些替代方案？

## 学习目标

完成本文后，你将能够：

- [ ] 解释 Kronos 选择 Pre-Norm + RMSNorm + RoPE + SwiGLU 的理由及各自替代方案
- [ ] 说明因果掩码和填充掩码在自注意力和交叉注意力中的不同行为
- [ ] 理解时间嵌入的分离编码+求和策略的设计权衡
- [ ] 评估"替换某个组件"对整体系统的影响

---

## 设计目标

Kronos 的 Transformer 需要处理金融时间序列的特殊性：

- **严格的时间顺序**：金融数据不能打乱，因果性至关重要
- **多尺度时间特征**：分钟、小时、星期等不同粒度的周期性
- **长序列依赖**：需要捕捉数百步之前的历史对当前的影响
- **资源效率**：推理需要在合理时间内完成（尤其是批量预测场景）

---

## TransformerBlock

Kronos 使用标准的 Pre-Norm Transformer 块：

```python
class TransformerBlock(nn.Module):
    def forward(self, x, key_padding_mask=None):
        residual = x
        x = self.norm1(x)                          # Pre-Norm
        attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + attn_out                     # 残差连接

        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x
```

### 为什么选择 Pre-Norm？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Pre-Norm**（Kronos 采用） | 训练更稳定，梯度流动顺畅，不需要 warmup | 理论上表达力略弱于 Post-Norm |
| Post-Norm | 原始 Transformer 方案，理论上表达力更强 | 训练不稳定，需要 learning rate warmup |

Pre-Norm 已成为现代 Transformer 的主流选择（GPT-2/3、LLaMA 等均采用）。

### 为什么使用 RMSNorm 而非 LayerNorm？

```python
class RMSNorm(nn.Module):
    def _norm(self, x):
        return x * rsqrt(mean(x², dim=-1) + eps)

    def forward(self, x):
        return _norm(x.float()).type_as(x) * weight
```

| 方案 | 计算 | 特点 |
|------|------|------|
| LayerNorm | 减去均值，除以标准差 | 需要计算均值和标准差 |
| **RMSNorm**（Kronos 采用） | 除以均方根 | 不减去均值，计算更快 |

RMSNorm 的设计假设输入已经接近零均值（经过残差连接后通常如此），省略均值计算在保持效果的同时提升了计算效率。这也是 LLaMA 等现代模型的选择。

---

## MultiHeadAttentionWithRoPE（旋转位置编码）

Kronos 使用带 RoPE（Rotary Positional Embedding）的多头因果注意力：

```python
class MultiHeadAttentionWithRoPE(nn.Module):
    def forward(self, x, key_padding_mask=None):
        q = q_proj(x).reshape(B, S, H, D).transpose(1, 2)
        k = k_proj(x).reshape(B, S, H, D).transpose(1, 2)
        v = v_proj(x).reshape(B, S, H, D).transpose(1, 2)

        q, k = self.rotary(q, k)  # 应用 RoPE

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=padding_mask,
            is_causal=True  # 因果掩码：未来位置不可见
        )
        return self.out_proj(output.transpose(1, 2).reshape(B, S, d_model))
```

### RoPE（旋转位置编码）

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        inv_freq = 1.0 / (10000 ** (arange(0, dim, 2) / dim))
        # 基础频率：10000^(-2k/d)，k = 0, 1, ..., d/2-1

    def forward(self, q, k):
        t = arange(seq_len)                    # 位置索引
        freqs = einsum('i,j->ij', t, inv_freq) # 频率矩阵
        emb = cat(freqs, freqs)                # 复制一份
        cos, sin = emb.cos(), emb.sin()

        return (q * cos) + (rotate_half(q) * sin), \
               (k * cos) + (rotate_half(k) * sin)
```

**RoPE 的核心思想**：通过旋转矩阵将位置信息编码到 q 和 k 中。位置 m 的 q 和位置 n 的 k 的点积结果只依赖于 q、k 本身和相对位置 m-n，与绝对位置无关。

**为什么选择 RoPE？**

| 方案 | 特点 |
|------|------|
| 正弦位置编码（原始 Transformer） | 固定编码，不可学习 |
| 可学习位置编码 | 需要固定最大长度 |
| **RoPE**（Kronos 采用） | 相对位置感知，外推性好，计算高效 |

RoPE 的优势在于它编码的是**相对位置**而非绝对位置，这对于金融时间序列特别重要——两个 K 线之间的时间间隔比它们的绝对位置更有意义。

### 因果掩码

```python
is_causal=True  # 在 scaled_dot_product_attention 中启用
```

因果掩码确保每个位置只能关注自身及之前的位置，维持时间序列的因果性。在推理时，这意味着模型只能基于已生成的令牌预测下一个令牌。

### 填充掩码

```python
if key_padding_mask is not None:
    attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
    attn_mask = attn_mask.expand(-1, n_heads, seq_len, -1)
```

填充掩码用于忽略批量中不同长度的序列的填充位置。掩码为 1 的位置表示填充，注意力计算时被排除。

---

## FeedForward（SwiGLU 前馈网络）

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.0):
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x):
        return dropout(w2(silu(w1(x)) * w3(x)))
```

这是 **SwiGLU**（Swish-Gated Linear Unit）激活函数的实现：

```
output = W2(SiLU(W1(x)) ⊙ W3(x))
```

### 为什么选择 SwiGLU？

| 方案 | 公式 | 特点 |
|------|------|------|
| ReLU | `max(0, x)` | 简单但有"神经元死亡"问题 |
| GELU | `x * Φ(x)` | GPT-2 使用，平滑的 ReLU 替代 |
| **SwiGLU**（Kronos 采用） | `SiLU(W1(x)) * W3(x)` | 门控机制，表达力更强 |

SwiGLU 通过门控机制（`W3(x)` 控制信息流）提供了比简单激活函数更强的表达能力。代价是需要三个权重矩阵（W1、W2、W3）而非两个。

**为什么不使用 bias？** 现代 Transformer 普遍省略线性层的偏置项（LLaMA、PaLM 等均如此）。在 Pre-Norm 架构中，RMSNorm 已经提供了偏移的能力，额外的 bias 参数几乎不提供收益，反而增加参数量和计算量。

---

## MultiHeadCrossAttentionWithRoPE（交叉注意力）

用于 `DependencyAwareLayer` 中，实现 s2 对 s1 的条件依赖：

```python
class MultiHeadCrossAttentionWithRoPE(nn.Module):
    def forward(self, query, key, value, key_padding_mask=None):
        q = q_proj(query)    # 来自 s1 嵌入
        k = k_proj(key)      # 来自 Transformer 上下文
        v = v_proj(value)    # 来自 Transformer 上下文

        q, k = self.rotary(q, k)

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=...,
            is_causal=self.training  # 训练时因果，推理时非因果
        )
        return out_proj(output)
```

### 与自注意力的区别

| 维度 | 自注意力 | 交叉注意力 |
|------|----------|-----------|
| query 来源 | x 自身 | s1 嵌入（外部输入） |
| key/value 来源 | x 自身 | Transformer 上下文 |
| 用途 | 序列内部信息交互 | 跨模块信息融合 |
| 因果性 | 始终因果 | 训练时因果，推理时非因果 |

**推理时为什么非因果？** 在推理阶段，DependencyAwareLayer 的 query 是当前步的 s1 采样结果，key/value 是完整的 Transformer 上下文。此时没有"未来信息泄漏"的问题（s1 已经确定），因此可以关注所有上下文位置。

---

## TemporalEmbedding（时间嵌入）

```python
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, learn_pe):
        minute_size = 60    # 0-59
        hour_size = 24      # 0-23
        weekday_size = 7    # 0-6
        day_size = 32       # 1-31
        month_size = 13     # 1-12

        Embed = FixedEmbedding if not learn_pe else nn.Embedding
        self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        return hour_embed(x[:, :, 1]) + weekday_embed(x[:, :, 2]) \
             + day_embed(x[:, :, 3]) + month_embed(x[:, :, 4]) \
             + minute_embed(x[:, :, 0])
```

### 设计要点

1. **五个独立的时间特征**：每个特征有独立的嵌入表，不共享参数
2. **相加融合**：五个嵌入向量直接求和，而非拼接或更复杂的融合方式
3. **可学习 vs 固定**：`learn_pe=True` 时使用可学习的嵌入，模型自动学习最优编码

### 为什么分别嵌入再求和？

**替代方案 1：拼接后线性变换** — 参数更多，但表达力不一定更强

**替代方案 2：直接用时间戳作为连续特征** — 无法捕捉周期性（如"周一"和"下周一"的关系）

**选择求和的理由**：每个时间特征独立编码后求和，既保留了各自的周期性语义（周一就是周一，不受小时数影响），又保持了维度不变（不需要额外的融合层）。

---

## 设计模式总结

| 组件 | 设计选择 | 理由 |
|------|----------|------|
| 归一化 | RMSNorm | 比 LayerNorm 更高效，效果相当 |
| 注意力 | RoPE + 因果掩码 | 相对位置感知，适合时间序列 |
| 前馈网络 | SwiGLU (无 bias) | 门控机制增强表达力，无 bias 减少参数 |
| 残差连接 | Pre-Norm | 训练稳定，梯度流动顺畅 |
| 时间编码 | 分离嵌入 + 求和 | 保留各时间维度的独立语义 |
| 交叉注意力 | s1 query, Transformer key/value | s2 条件依赖 s1 |

Kronos 的 Transformer 设计与 LLaMA 等现代大语言模型高度一致（RMSNorm + RoPE + SwiGLU + Pre-Norm），说明这些设计选择具有跨领域的通用性。

---

## 与 LLaMA 架构的逐项对照

Kronos 的 Transformer 设计大量借鉴了 LLaMA 的架构选择。以下逐项对照两者的一致与差异：

| 组件 | LLaMA | Kronos | 一致性 | 差异说明 |
|------|-------|--------|--------|---------|
| 归一化 | RMSNorm | RMSNorm | 完全一致 | — |
| 注意力 | GQA（分组查询注意力） | MHA（多头注意力） | 部分一致 | Kronos 使用标准多头注意力，未采用 GQA。原因是 Kronos 模型规模较小（d_model ≤ 832），标准 MHA 的计算开销可接受 |
| 位置编码 | RoPE | RoPE | 完全一致 | — |
| 前馈网络 | SwiGLU | SwiGLU | 完全一致 | — |
| 偏置项 | 无 bias | 无 bias | 完全一致 | — |
| 残差连接 | Pre-Norm | Pre-Norm | 完全一致 | — |
| 词汇表 | 文本 token（30K+） | K 线令牌（s1: 1024, s2: 1024） | 不同 | Kronos 的词汇表远小于 LLM，这是量化令牌设计的直接结果 |
| 序列长度 | 4096-8192+ | 512 | 不同 | 金融时间序列的有效信息窗口较短，512 已覆盖足够的历史模式 |
| 时间感知 | 无（文本无需时间特征） | TemporalEmbedding | Kronos 独有 | 金融数据的周期性（日内、周内、月内效应）需要显式时间编码 |
| 条件解码 | 无 | DependencyAwareLayer | Kronos 独有 | s2 令牌需要条件依赖于 s1 的采样结果，这是层级令牌体系的核心机制 |

**关键启示**：Kronos 在 Transformer 主体上复用了 LLM 领域验证过的最佳实践（RMSNorm + RoPE + SwiGLU + Pre-Norm），但在两个维度上做了金融领域特有的扩展——**时间感知**（TemporalEmbedding）和**条件解码**（DependencyAwareLayer）。这种"主干复用 + 领域适配"的设计思路值得在其他时间序列任务中借鉴。

---

## 组件替换影响分析

如果你打算替换某个 Transformer 组件，以下是关键注意事项：

| 替换操作 | 需要同步修改 | 风险等级 | 说明 |
|---------|-------------|---------|------|
| Pre-Norm → Post-Norm | 学习率调度（需增加 warmup）、训练稳定性 | 高 | 可能导致深层梯度爆炸，需重新调参 |
| RMSNorm → LayerNorm | 计算开销微增 | 低 | 功能等价，但需验证训练收敛性 |
| RoPE → 可学习位置编码 | 需固定最大序列长度 | 中 | 丧失相对位置感知和外推能力 |
| SwiGLU → GELU | 前馈网络参数量减少（3 个权重矩阵 → 2 个） | 低 | 表达力可能下降 |
| 无 bias → 有 bias | 增加参数量 | 低 | 在 Pre-Norm 架构中收益极小 |

---

## 🧪 动手练习

### 练习 1：观察 RoPE 对位置编码的影响

加载预训练模型，输入两段相同数据但不同时间偏移的时间戳，对比 Transformer 中间层输出的余弦相似度。如果 RoPE 正确编码了相对位置，相同相对距离的令牌对应具有较高的相似度。

```python
import torch
from model import Kronos, KronosTokenizer

model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
model.eval()

# 构造两段相同 OHLCV 但不同时间戳的输入
x = torch.randn(1, 10, 6)  # 随机 OHLCV
stamp1 = torch.arange(10).unsqueeze(0).unsqueeze(-1).expand(1, 10, 5).float()
stamp2 = (torch.arange(10) + 100).unsqueeze(0).unsqueeze(-1).expand(1, 10, 5).float()

with torch.no_grad():
    # 仅获取 embedding 层输出（跳过 Transformer）
    emb1 = model.embedding(model.tokenizer.encode(x, half=True)) + model.time_emb(stamp1)
    emb2 = model.embedding(model.tokenizer.encode(x, half=True)) + model.time_emb(stamp2)
    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1).mean()
    print(f"不同时间偏移下嵌入的余弦相似度: {cos_sim:.4f}")
    # 预期：相似度接近 1.0，因为 OHLCV 内容相同
```

**验证方法**：如果输出的余弦相似度接近 1.0，说明内容相同但时间偏移不同时，嵌入表示基本一致（时间信息叠加但不破坏内容语义）。

---

## ✅ 自测清单

- [ ] 我能说出 Kronos Transformer 与 LLaMA 在组件选择上的共同点
- [ ] 我能解释为什么交叉注意力在推理时使用非因果模式
- [ ] 我能说明 SwiGLU 相比 ReLU 的优势与代价
- [ ] 我能列出将 Pre-Norm 替换为 Post-Norm 时需要注意的事项
- [ ] 我能解释时间嵌入中"分离编码+求和"优于"拼接+线性变换"的理由

---

**文档元信息**
难度：⭐⭐⭐⭐ | 类型：专家设计 | 预计阅读时间：30 分钟
