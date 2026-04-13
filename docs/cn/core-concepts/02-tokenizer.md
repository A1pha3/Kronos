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

KronosTokenizer 的核心是 **Binary Spherical Quantization（BSQ，二值球面量化）**，它将连续向量量化为二值向量（每个维度取 +1 或 -1）。

### 为什么选择 BSQ 而非传统 VQ？

| 维度 | 传统 VQ（如 VQ-VAE） | BSQ（Kronos 采用） |
|------|---------------------|-------------------|
| 码本表示 | 需要存储 K 个 D 维向量（显式码本） | 不需要显式码本——二值向量由符号判断直接生成 |
| 码本利用率 | 容易出现"死码"（部分码本条目从未被使用） | 天然均匀——每个维度的 ±1 使用概率接近 50% |
| 索引编码 | 需要查找最近邻，复杂度 O(K·D) | 直接用二进制位表示索引，复杂度 O(D) |
| 码本大小 | 受限于显存和搜索效率 | 2^D 种令牌，无需存储 |
| 训练稳定性 | 码本坍缩风险（所有向量映射到少数码字） | 熵正则化保证码本均匀使用 |

**直觉理解**：传统 VQ 像是在一个"图书馆"中找到最接近的书（需要搜索），而 BSQ 像是判断每个维度"偏向正还是偏向负"（只需要看符号）。后者虽然精度略低，但计算高效且不会遇到码本利用率问题。

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

### 码本容量分析

默认配置下，BSQ 量化器产生 `2^20 = 1,048,576` 种可能的令牌。但实际表达能力需要从以下几个角度理解：

- **理论最大值**：20 比特的二值空间有 1,048,576 个点。但 BSQ 的 L2 归一化将所有连续向量投影到单位球面上，因此并非所有 1,048,576 个码本条目都对应等概率的 K 线模式。
- **有效码本利用率**：BSQ 的熵正则化机制（`gamma0` 和 `gamma` 参数）鼓励每个比特的 ±1 使用概率接近 50%，从而推动码本被均匀使用。但实际训练中，与常见 K 线模式对应的码字会被更频繁地使用，而与罕见模式对应的码字使用频率较低。
- **层级近似的影响**：在层级令牌体系下，s1 和 s2 的组合数理论上是 `2^10 * 2^10 = 1,048,576`，但解码时 s1 和 s2 的二值向量是独立重建再拼接的（`indices_to_bits` 方法中分别处理 `x1` 和 `x2`），因此组合空间中的某些点可能永远不会被自然地访问到。

**实际含义**：对于大多数金融市场的 K 线模式（涨跌、十字星、锤子线等），1024 x 1024 的组合空间提供了足够的表达能力。但对于同时具有极端价格波动和极端成交量的罕见 K 线，量化精度可能不足以精确重建——这些极端值在 BSQ 量化后会被映射到最近的二值向量，信息损失较大。

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

### 为什么需要两路重建损失？

损失函数同时优化 `z_pre`（仅 s1 比特重建）和 `z`（完整码本重建）两路重建。这个设计有一个容易忽略的深层目的：

- **`z_pre` 损失的作用**：迫使 s1 比特**独立地**编码足够多的信息。如果没有这个辅助损失，s1 比特可能只编码一些次要信息，而把重要信息全部"推"给 s2 比特——因为最终解码时 s1 和 s2 是一起使用的。辅助损失确保了即使只看 s1，也能获得有意义的粗粒度重建。
- **`z` 损失的作用**：确保完整码本（s1 + s2）能够高质量地重建原始数据，这是分词器的最终目标。
- **两者之和**：在"粗粒度独立可用"和"完整重建高质量"之间取得平衡。

### 总损失为什么要除以 2？

`(recon_loss + bsq_loss) / 2` 中的除以 2 是为了平衡重建损失和量化损失的量级。`recon_loss` 是两路 MSE 之和（量级较大），而 `bsq_loss` 包含 commit loss 和熵正则化（量级较小）。如果不除以 2，重建损失可能主导梯度方向，导致量化约束被忽略。

### 调参失误的影响

| 参数 | 设置过大 | 设置过小 |
|------|---------|---------|
| `beta`（commit loss 权重） | 编码器输出被过度约束到量化点，重建质量下降 | 量化约束过弱，连续空间与离散空间的映射不够紧密 |
| `gamma0`（样本熵权重） | 每个样本被迫均匀使用所有码字，信息丢失 | 样本集中使用少量码字，码本利用率低 |
| `gamma`（码本熵权重） | 码本被强制均匀使用，但个别码字可能无意义 | 部分码字从未被使用（"死码"），表达能力浪费 |

如果微调时发现验证损失不收敛，可以尝试：降低 `beta`（放宽量化约束）或降低 `gamma0`/`gamma`（放松均匀性要求）。

### 训练异常的典型信号与诊断

| 异常信号 | 可能原因 | 诊断方法 |
|---------|---------|---------|
| `recon_loss` 下降但 `z_pre` 的 MSE 停滞 | s1 比特没有学到有用的信息，所有信息都被推给 s2 | 分别检查 `z_pre` 和 `z` 的重建 MSE；如果前者远大于后者，说明层级分工失败 |
| `bsq_loss` 持续为很大的正值 | commit loss 权重 `beta` 过大，或量化前后的 L2 距离没有缩小 | 检查编码器输出 `quant_embed(z)` 的范数是否与量化后的值域匹配 |
| 验证损失先降后升 | 过拟合——分词器在训练集上记忆了特定的 K 线模式 | 减小 `d_model` 或增大 `ffn_dropout_p`/`attn_dropout_p` |
| `z` 的重建 MSE 几乎不下降 | 码本容量不足以表达数据中的多样性 | 检查 `s1_bits + s2_bits` 是否足够；或者数据中是否存在过多的极端值 |
| 熵正则化损失发散 | `gamma0` 或 `gamma` 设置过大 | 降低熵正则化权重，观察 `avg_prob` 是否过于均匀（每个子组码字的概率接近 1/2） |

---

## 训练与推理的路径差异

上一节从损失函数的角度解释了 `forward()` 的双路重建，但一个容易被忽略的关键事实是：**推理时使用的 `encode()` 和 `decode()` 与训练时的 `forward()` 走的是完全不同的路径**。理解这种不对称性，对于正确使用分词器 API 和排查问题至关重要。

### encode()：跳过解码器的"快速通道"

```python
# 摘自 model/kronos.py — KronosTokenizer.encode()
def encode(self, x, half=False):
    z = self.embed(x)
    for layer in self.encoder:
        z = layer(z)
    z = self.quant_embed(z)
    bsq_loss, quantized, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
    return z_indices  # 只返回索引，完全跳过解码器
```

`encode()` 的数据流是：**embed -> encoder -> quant_embed -> BSQ -> 返回索引**。它从不经过 `post_quant_embed`、`decoder` 或 `head`。这是合理的——推理阶段的目标是提取离散令牌，不需要重建原始数据，因此跳过解码器避免了约一半的计算量。

此外，`collect_metrics=False` 意味着 BSQ 量化器在 `encode()` 中不计算样本熵和码本熵指标。这些指标仅在训练时用于熵正则化损失，推理时跳过可以节省不必要的计算开销。

### decode()：仅走完整码本路径

```python
# 摘自 model/kronos.py — KronosTokenizer.decode()
def decode(self, x, half=False):
    quantized = self.indices_to_bits(x, half)  # 索引转回二值向量
    z = self.post_quant_embed(quantized)        # 仅使用完整码本路径
    for layer in self.decoder:
        z = layer(z)
    z = self.head(z)
    return z
```

`decode()` 的数据流是：**indices_to_bits -> post_quant_embed -> decoder -> head**。它只经过 `post_quant_embed`（完整码本维度 `s1_bits + s2_bits` -> `d_model`），**从不经过 `post_quant_embed_pre`**（粗粒度路径）。

这意味着 `forward()` 中的 `z_pre` 粗粒度重建路径（`post_quant_embed_pre` -> decoder -> head）是一个**纯训练辅助结构**。它通过辅助损失鼓励 s1 比特独立编码有意义的粗粒度信息，但在推理时完全不会用到。

### 三条路径对比

| 方法 | 数据流路径 | 是否经过解码器 | 解码器遍历次数 | 用途 |
|------|-----------|---------------|---------------|------|
| `forward()` | embed -> encoder -> quant_embed -> BSQ -> **双路解码** | 是 | 2 次（z_pre 一次 + z 一次） | 训练：计算损失 |
| `encode()` | embed -> encoder -> quant_embed -> BSQ | 否 | 0 次 | 推理：提取令牌 |
| `decode()` | indices_to_bits -> post_quant_embed -> decoder -> head | 是 | 1 次 | 推理：还原数据 |

### 为什么推理不需要 z_pre 路径？

`z_pre` 路径的设计目的是在训练时提供一个**辅助监督信号**：仅用 s1 比特的量化结果重建原始数据，并计算 MSE 损失。这迫使 s1 比特独立编码足够多的粗粒度信息（主要价格走势），使得层级令牌体系（s1 捕捉主趋势 + s2 捕捉修正细节）真正有效。

但在推理时，`decode()` 拿到的是完整的令牌（s1 + s2），通过完整码本路径 `post_quant_embed` 映射后进入解码器，已经包含了所有信息。此时使用仅有 s1 比特的粗粒度路径毫无意义——它只会给出更低精度的重建，完全没有必要。

**记忆口诀**：训练时"双路同行"（z_pre 辅助损失 + z 主损失），推理时"各取所需"（encode 只提取令牌，decode 只走完整路径）。

### 噪声与极端值对 s1/s2 的影响

当输入数据包含极端值或噪声时，BSQ 量化器会有以下行为（可从 `module.py` 中 `quantize()` 方法的实现推导）：

**极端值**：KronosPredictor 在分词之前会执行 z-score 标准化和 clip（默认范围 [-5, 5]）。这意味着极端值在到达 BSQ 量化器之前已经被压缩。具体来说：
- 价格涨跌幅超过 5 个标准差的 K 线会被截断为 5 个标准差的等价表示
- 截断后，量化器将其映射到某个二值向量——这个二值向量可能与"真正的极端 K 线"不完全匹配，但仍然是一个有意义的近似

**高噪声数据**：当输入数据噪声较大时（如成交量数据剧烈波动），编码器输出的连续向量在 20 维空间中的位置会更加不稳定。但由于 BSQ 的二值化决策（`z_i > 0 -> +1, z_i <= 0 -> -1`）是一个硬阈值操作，小的噪声扰动通常不会改变量化结果——只有当噪声大到足以翻转某个维度的符号时，才会产生不同的令牌。这使得 BSQ 对小幅噪声具有天然的鲁棒性。

**s1 vs s2 的敏感度差异**：s1 对应编码器输出的前 10 个维度，s2 对应后 10 个维度。如果噪声主要影响某些特定的维度（如与成交量相关的维度），这些影响可能集中在 s1 或 s2 的某一部分，而非均匀分布。

### 编码与解码的计算成本差异

从源码可以明确计算三条路径的相对开销：

| 操作 | 计算量 | 说明 |
|------|--------|------|
| `encode()` | embed + N-1 层 encoder + quant_embed + BSQ | 不经过解码器，约占总计算量的 40-50% |
| `decode()` | indices_to_bits + post_quant_embed + N-1 层 decoder + head | 经过完整解码器，约占总计算量的 40-50% |
| `forward()` | encode 路径 + BSQ + 双路解码（decoder 遍历 2 次） | 完整路径，计算量最大 |

在标准预测流程中（`encode` 一次 + `decode` 一次），编码和解码的计算量大致相当。编码器和解码器都使用 N-1 层 Transformer（默认 N-1=4），它们的参数量和计算量基本对称。唯一的不对称在于解码器在 `forward()` 中被遍历两次（一次为 `z_pre`，一次为 `z`），但推理时只遍历一次。

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
- **参考**：[术语表](../references/glossary.md) — 概念速查
- **参考**：[源码走读](../architecture/04-source-code-walkthrough.md) ⭐⭐⭐⭐ — KronosTokenizer 源码逐行解读

---
**文档元信息**
难度：⭐⭐ | 类型：核心概念 | 更新日期：2026-04-11 | 预计阅读时间：15 分钟
