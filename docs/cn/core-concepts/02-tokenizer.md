# KronosTokenizer 详解 ⭐⭐

> **目标读者**：想深入理解分词器工作原理的用户和开发者
> **核心问题**：KronosTokenizer 如何将连续的 OHLCV 数据转换为离散令牌？

---

## 学习目标

阅读本文后，你应该能够：

- [ ] 解释 KronosTokenizer 的编码器-解码器架构，说明解码器层是共享的而非独立两套
- [ ] 区分 `post_quant_embed_pre` 和 `post_quant_embed` 两个后量化线性层的不同用途
- [ ] 描述 BSQ 量化过程，并理解 s1/s2 半分割编码的含义
- [ ] 使用 `encode()` 和 `decode()` API 完成令牌的编解码操作

---

## 概念定义

### 一句话定义

**KronosTokenizer** 是一个编码器-解码器架构的神经网络，使用 Binary Spherical Quantization（BSQ，二值球面量化）将连续的 OHLCV 向量压缩为离散的二值令牌序列。

### 为什么需要分词器？

大语言模型（如 GPT）直接处理离散的文本令牌，但金融 K 线数据是连续的多维数值。直接在连续空间进行自回归预测存在以下问题：

- **误差累积**：每步预测的微小偏差会在多步预测中被放大
- **缺乏离散化**：无法利用成熟的令牌采样策略（温度、top-k、top-p）

KronosTokenizer 解决这个问题的方法是：**先将连续值"翻译"为离散令牌，再在离散空间进行预测**。

### 类比理解

> 就像图像压缩中的"有损压缩"：KronosTokenizer 将高精度的 OHLCV 数据"压缩"为有限的几种令牌（码本），虽然丢失了部分精度，但保留了最重要的信息特征。

---

## 架构组成

KronosTokenizer 由四个核心部分组成：

```
输入 OHLCV (B, T, 6)
      │
      ▼
┌─────────────┐
│ embed       │  线性映射：6 维 → d_model 维
│ (nn.Linear) │  定义于 KronosTokenizer.__init__()
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Encoder     │  N-1 层 Transformer 编码器（self.encoder）
│ (N-1 layers)│  提取深层特征
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ quant_embed │  线性映射：d_model → codebook_dim
│ (nn.Linear) │  定义于 KronosTokenizer.__init__()
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ BSQuantizer │  二值球面量化：连续 → 离散
│             │  输出：量化向量 + 损失 + 令牌索引
└─────┬───────┘
      │
      ├──────────────────────────────────────┐
      │ quantized_pre (s1_bits 维)           │ quantized (codebook_dim 维)
      ▼                                     ▼
┌───────────────────────┐    ┌───────────────────────┐
│ post_quant_embed_pre  │    │ post_quant_embed      │
│ nn.Linear(s1_bits,    │    │ nn.Linear(codebook_dim│
│          d_model)     │    │          , d_model)   │
│ 仅用 s1_bits 维       │    │ 用完整码本维度         │
└──────────┬────────────┘    └──────────┬────────────┘
           │ z_pre                       │ z
           ▼                             ▼
     ┌───────────────────────────────────────┐
     │   共享 Decoder（self.decoder）         │
     │   同一组 N-1 层 Transformer 解码器，    │
     │   被 z_pre 和 z 各遍历一次             │
     │   ⚠ 非独立的两套解码器                  │
     └───────────────┬───────────────────────┘
           │                    │
           ▼                    ▼
     ┌─────────────┐     ┌─────────────┐
     │ head        │     │ head        │
     │ (nn.Linear) │     │ (nn.Linear) │
     │ 共享的输出层  │     │ 共享的输出层  │
     └─────┬───────┘     └─────┬───────┘
           │                    │
           ▼                    ▼
   z_pre (粗粒度重建)      z (细粒度重建)
   输出 OHLCV (B, T, 6)   输出 OHLCV (B, T, 6)
```

### 解码器共享机制详解

源码中 `KronosTokenizer.forward()` 的解码器是 **共享的**，同一组 `self.decoder` 层被 `z_pre` 和 `z` 各遍历一次：

```python
# 摘自 model/kronos.py — KronosTokenizer.forward()
# Decoder layers (for pre part - s1 bits)
for layer in self.decoder:
    z_pre = layer(z_pre)
z_pre = self.head(z_pre)

# Decoder layers (for full codebook)
for layer in self.decoder:
    z = layer(z)
z = self.head(z)
```

这意味着解码器权重在粗粒度和细粒度重建之间共享，而非使用独立的两套解码器。这种设计的优势是减少模型参数量，并鼓励解码器学习对两种粒度都有用的通用重建能力。

### 两个后量化线性层

源码中定义了两个不同的后量化线性层（位于 `KronosTokenizer.__init__()`）：

| 层名 | 输入维度 | 输出维度 | 用途 |
|------|---------|---------|------|
| `post_quant_embed_pre` | `s1_bits`（10） | `d_model`（256） | 将仅包含 s1_bits 维的量化结果映射回模型维度 |
| `post_quant_embed` | `codebook_dim`（20） | `d_model`（256） | 将完整码本维度的量化结果映射回模型维度 |

### 关键参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `d_in` | 输入维度（OHLCV 特征数） | 6 |
| `d_model` | 模型内部维度 | 256 |
| `n_heads` | 注意力头数 | 4 |
| `ff_dim` | 前馈网络维度 | 512 |
| `n_enc_layers` | 编码器层数 | 4 |
| `n_dec_layers` | 解码器层数（共享） | 4 |
| `s1_bits` | 粗粒度令牌的比特数 | 10 |
| `s2_bits` | 细粒度令牌的比特数 | 10 |
| `codebook_dim` | 码本维度 = `s1_bits + s2_bits` | 20 |

---

## 核心 API

### encode() — 编码

将 OHLCV 数据编码为离散令牌索引（定义于 `KronosTokenizer.encode()`）：

```python
# 完整编码：返回完整码本的索引
z_indices = tokenizer.encode(x)
# z_indices 形状：(batch_size, seq_len)，每个值在 [0, 2^20) 范围

# 半分割编码：分别返回 s1 和 s2 的索引
z_indices = tokenizer.encode(x, half=True)
# z_indices 是列表：[s1_indices, s2_indices]
# s1_indices 形状：(batch_size, seq_len)，范围 [0, 2^10)
# s2_indices 形状：(batch_size, seq_len)，范围 [0, 2^10)
```

**`half=True` 的含义**：BSQ 量化器产生的码本维度为 `s1_bits + s2_bits`。`half=True` 将量化结果沿最后一个维度切分为前 `s1_bits` 维和后 `s2_bits` 维，分别转换为整数索引。这正是 Kronos 自回归模型使用的格式。

### decode() — 解码

将令牌索引还原为 OHLCV 数据（定义于 `KronosTokenizer.decode()`）：

```python
# 解码半分割的令牌
reconstructed = tokenizer.decode([s1_indices, s2_indices], half=True)
# reconstructed 形状：(batch_size, seq_len, 6)

# 解码完整码本的令牌
reconstructed = tokenizer.decode(full_indices)
```

> **注意**：`decode()` 方法内部只使用 `post_quant_embed`（完整码本维度），不经过 `post_quant_embed_pre`。这与 `forward()` 的行为不同。

### forward() — 前向传播（训练用）

训练时使用完整的前向传播，同时返回重建结果和量化损失（定义于 `KronosTokenizer.forward()`）：

```python
(z_pre, z), bsq_loss, quantized, z_indices = tokenizer(x)
# z_pre: 仅使用 s1 比特重建的结果（粗粒度）
# z: 使用全部码本重建的结果（细粒度）
# bsq_loss: 量化损失（commit loss + 熵正则化）
# quantized: 量化后的向量
# z_indices: 令牌索引
```

**为什么有两个重建结果？** `z_pre` 只用 `s1_bits` 维度的信息重建，用于验证粗粒度令牌是否足以捕捉主要信息。训练时两个重建结果都会计算 MSE 损失。

---

## 量化过程详解

KronosTokenizer 的核心是 **Binary Spherical Quantization（BSQ）**，它将连续向量量化为二值向量（每个维度取 +1 或 -1）。

### 量化步骤

```
连续向量 z ∈ R^20
    │
    ▼ L2 归一化
z_norm ∈ R^20
    │
    ▼ 逐维度二值化：z_i > 0 → +1, z_i ≤ 0 → -1
z_q ∈ {-1, +1}^20
    │
    ▼ 缩放：z_q / sqrt(20)
z_scaled ∈ R^20
    │
    ▼ 转换为整数索引
index = Σ(z_q_i + 1)/2 × 2^i
```

### 码本大小

- 完整码本：`2^(s1_bits + s2_bits) = 2^20 = 1,048,576` 种令牌
- s1 码本：`2^s1_bits = 2^10 = 1,024` 种粗粒度令牌
- s2 码本：`2^s2_bits = 2^10 = 1,024` 种细粒度令牌

通过分层设计，模型用 2 x 1,024 = 2,048 种令牌组合来近似表达 1,048,576 种完整令牌，大幅降低了预测难度。

---

## 训练损失

分词器训练时使用以下损失函数：

```python
# 重建损失
recon_loss_pre = MSE(z_pre, x)  # 粗粒度重建损失
recon_loss_all = MSE(z, x)      # 细粒度重建损失
recon_loss = recon_loss_pre + recon_loss_all

# 总损失
loss = (recon_loss + bsq_loss) / 2
```

其中 `bsq_loss` 包含两部分：

1. **Commit Loss**：衡量量化前后的距离，鼓励连续向量靠近量化点
2. **熵正则化**：鼓励码本被均匀使用，避免"死码"问题

---

## 使用场景

### 场景对比

| 场景 | 使用方式 | 调用方法 | 适用条件 |
|------|---------|---------|---------|
| 标准预测 | encode + 模型预测 + decode | `encode(x, half=True)` → 预测 → `decode(pred, half=True)` | 完整 OHLCV 数据 |
| 仅价格预测 | 自动补零 volume | `KronosPredictor.predict(df)` | DataFrame 缺少 volume 列 |
| 训练分词器 | forward 完整前向传播 | `tokenizer(x)` 返回 `(z_pre, z), loss, ...` | 需要两个重建结果和量化损失 |
| 直接解码 | 单步解码 | `tokenizer.decode(indices, half=True)` | 已有令牌索引 |

### 场景 1：标准预测

```python
# encode + decode 是最常见的组合
indices = tokenizer.encode(x, half=True)
# ... 模型预测新的 indices ...
reconstructed = tokenizer.decode(predicted_indices, half=True)
```

### 场景 2：仅价格预测（无成交量）

```python
# 即使只有 4 列数据，也能正常工作
# KronosPredictor 会自动补零 volume 和 amount
x_df = df[['open', 'high', 'low', 'close']]
```

---

## 动手练习

### 练习 1：编码随机数据并观察 s1/s2 索引分布

加载分词器，编码一段随机数据，观察 s1 和 s2 索引的分布：

```python
import torch
from model.kronos import KronosTokenizer

# 创建分词器（使用默认参数）
tokenizer = KronosTokenizer(
    d_in=6, d_model=256, n_heads=4, ff_dim=512,
    n_enc_layers=4, n_dec_layers=4,
    ffn_dropout_p=0.1, attn_dropout_p=0.1, resid_dropout_p=0.1,
    s1_bits=10, s2_bits=10,
    beta=0.25, gamma0=0.0, gamma=0.0, zeta=1e-8, group_size=1
)

# 生成随机 OHLCV 数据（batch=2, seq_len=16, features=6）
x = torch.randn(2, 16, 6)

# 半分割编码
s1_idx, s2_idx = tokenizer.encode(x, half=True)

print(f"s1 索引范围: [{s1_idx.min()}, {s1_idx.max()}]")
print(f"s2 索引范围: [{s2_idx.min()}, {s2_idx.max()}]")
print(f"s1 唯一值数量: {s1_idx.unique().numel()} / {2**10}")
print(f"s2 唯一值数量: {s2_idx.unique().numel()} / {2**10}")

# 验证：解码后形状是否正确
reconstructed = tokenizer.decode([s1_idx, s2_idx], half=True)
print(f"重建形状: {reconstructed.shape}")  # 应为 (2, 16, 6)
```

---

## 自测清单

阅读完毕后，检验你是否理解了以下要点：

- [ ] 能否解释为什么 `forward()` 返回两个重建结果 `z_pre` 和 `z`，而 `decode()` 只返回一个？
- [ ] 能否说明 `post_quant_embed_pre` 和 `post_quant_embed` 的输入维度差异及其原因？
- [ ] 能否描述共享解码器的设计——同一组 `self.decoder` 层如何被 `z_pre` 和 `z` 各遍历一次？
- [ ] 能否解释 `half=True` 在 `encode()` 和 `decode()` 中的作用？
- [ ] 能否说出 BSQ 量化的三个步骤（L2 归一化、二值化、缩放）？

---

## 常见误区

### 误区 1：分词器是确定性的

**正确理解**：分词器的 `encode()` 是确定性的（给定相同输入，总是产生相同的令牌）。不确定性来自模型预测阶段的采样过程。

### 误区 2：分词器的重建是完美的

**正确理解**：BSQ 量化是有损压缩，重建结果会丢失部分精度。但通过 s1 + s2 的层级设计，重建精度足以支撑有效的预测。

### 误区 3：解码器有两套独立的参数

**正确理解**：解码器层是共享的。`forward()` 中 `z_pre` 和 `z` 使用的是同一组 `self.decoder` 层，权重完全相同，只是各遍历一次。这是源码中明确的设计，并非独立的两套解码器。

---

## 知识关联

- **前置**：[项目总览](01-overview.md) ⭐⭐ — 理解两阶段框架
- **相关**：[层级令牌体系](05-hierarchical-tokens.md) ⭐⭐ — s1/s2 的设计原理
- **进阶**：[BSQ 量化算法原理](../architecture/02-bsq-algorithm.md) ⭐⭐⭐⭐ — 深入理解量化数学

---
**文档元信息**
难度：⭐⭐ | 类型：核心概念 | 预计阅读时间：15 分钟
