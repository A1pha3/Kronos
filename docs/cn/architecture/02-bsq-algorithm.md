# BSQ 量化算法原理 ⭐⭐⭐⭐

> **目标读者**：想深入理解 Binary Spherical Quantization 数学原理的研究者
> **核心问题**：BSQ 如何将连续向量量化为二值码？损失函数的设计意图是什么？

## 学习目标

完成本文后，你将能够：

- [ ] 手动计算一个 D=4 向量的 BSQ 量化过程（从连续值到索引）
- [ ] 解释直通估计器（STE）如何解决二值化的不可导问题
- [ ] 说明 commit loss 与双熵正则化的各自作用和调节方向
- [ ] 理解分组（group）机制如何平衡熵计算效率与精度

---

## 问题背景

### 向量量化的目标

给定一个连续向量 **z** ∈ R^D，向量量化的目标是将其映射到一个有限的码本 C = {c_1, c_2, ..., c_K} 中，找到最接近的码本向量。

传统方法（如 VQ-VAE）面临两个问题：

1. **码本利用率低**：大部分码本条目可能从未被使用（"死码"问题）
2. **高维灾难**：当 D 较大时，码本需要指数级增长才能保持表达能力

### BSQ 的解决思路

Binary Spherical Quantization（BSQ）将量化问题转化为**二值化问题**：将每个维度量化为 +1 或 -1，形成一个二值码。码本大小为 2^D，且**天然均匀分布**（每个码本条目的概率相等），从根本上避免了死码问题。

**论文出处**：[https://arxiv.org/pdf/2406.07548.pdf](https://arxiv.org/pdf/2406.07548.pdf)

---

## 量化过程

### Step 1：L2 归一化

```python
z = F.normalize(z, dim=-1)  # z / ||z||_2
```

将输入向量投影到单位超球面上。这确保所有向量在同一尺度上被量化。

### Step 2：二值化

```python
zhat = torch.where(z > 0, +1, -1)  # 逐维度二值化
```

对每个维度独立判断：大于 0 取 +1，否则取 -1。这相当于将超球面划分为 2^D 个"象限"，每个象限对应一个码本条目。

### Step 3：直通估计器（Straight-Through Estimator）

```python
zq = z + (zhat - z).detach()  # 前向用 zhat，反向用 z 的梯度
```

二值化操作不可导（阶跃函数）。直通估计器的技巧是：

- **前向传播**：使用量化后的 `zhat`
- **反向传播**：梯度直接"穿透"到连续的 `z`，仿佛量化操作不存在

数学上等价于：

```
∂zq/∂z = 1  （恒等映射）
```

### Step 4：缩放

```python
q_scale = 1.0 / sqrt(embed_dim)
zq = zq * q_scale
```

缩放因子使量化向量的 L2 范数为 1（因为 `||zq||_2 = sqrt(D)`，缩放后为 `sqrt(D) / sqrt(D) = 1`）。

### Step 5：索引计算

```python
# 二值向量 → 整数索引
b = (zq + 1) / 2               # {-1, +1} → {0, 1}
index = sum(b_i * 2^i for i in range(D))
```

将二值向量解释为一个 D 位二进制数，转换为对应的十进制索引。

### 数值示例

给定一个 D=4 的归一化后向量 `z = [0.3, -0.8, 0.1, -0.5]`：

```
Step 1: 二值化
  z[0]=0.3  > 0 → +1    z[1]=-0.8 < 0 → -1
  z[2]=0.1  > 0 → +1    z[3]=-0.5 < 0 → -1
  zhat = [+1, -1, +1, -1]

Step 2: 转换为 0/1
  b = (zhat + 1) / 2 = [1, 0, 1, 0]

Step 3: 计算索引
  index = 1×2⁰ + 0×2¹ + 1×2² + 0×2³ = 1 + 0 + 4 + 0 = 5
  （注意：源码 BinarySphericalQuantizer 使用大端序 basis=[8,4,2,1]，
   实际 index = 1×8 + 0×4 + 1×2 + 0×1 = 10。具体端序取决于实现）

Step 4: 缩放
  q_scale = 1/√4 = 0.5
  zq = [+1, -1, +1, -1] × 0.5 = [0.5, -0.5, 0.5, -0.5]
  ||zq||₂ = √(0.25×4) = 1.0 ✓
```

---

## 损失函数

BSQ 的损失由三部分组成：

```
total_loss = commit_loss + ζ × entropy_penalty / τ
```

### 1. Commit Loss（承诺损失）

```python
commit_loss = β × mean((zq.detach() - z)² 之和)
```

**含义**：衡量量化前后向量的距离。推动编码器输出的连续向量靠近其量化点。

**为什么 detach zq？** 我们希望梯度推动 z 向 zq 靠近，而不是推动 zq 向 z 靠近（zq 是固定的量化点）。

### 2. Entropy Penalty（熵正则化）

熵正则化包含两个部分，作用方向相反：

#### Per-Sample Entropy（样本熵）

```python
H_sample = -Σ p_i × log(p_i)  （对每个子组的概率分布计算熵）
```

**目标**：最大化样本熵（通过 `-γ₀ × H_sample` 实现）。这鼓励每个样本的量化结果更均匀地分布在码本中，避免所有样本都映射到少数几个码本条目。

#### Codebook Entropy（码本熵）

```python
H_codebook = -Σ (avg_prob_i) × log(avg_prob_i)
```

其中 `avg_prob` 是所有样本在每个子组上的平均概率。

**目标**：最大化码本熵（通过 `-γ × H_codebook` 实现）。这从宏观层面鼓励码本被均匀使用。

#### 软熵计算

在训练模式下，BSQ 使用**软熵**（soft entropy）来计算样本熵，避免离散化导致的梯度消失：

```python
# 将 z 按组大小切分
divided_z = rearrange(z, '... (g c) -> ... g c', c=group_size)

# 计算与子码本的距离
distance = -2 * einsum('... g c, d c -> ... g d', divided_z, group_codebook)
prob = (-distance * inv_temperature).softmax(dim=-1)

# 分析模式：直接用 sigmoid 计算
p = sigmoid(-4 * z / sqrt(D) * inv_temperature)
```

### 3. 超参数含义

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `β` (beta) | Commit loss 权重 | 0.05 |
| `γ₀` (gamma0) | 样本熵权重 | 1.0 |
| `γ` (gamma) | 码本熵权重 | 1.1 |
| `ζ` (zeta) | 总熵正则化权重 | 0.05 |
| `inv_temperature` | 软熵的温度逆数 | 1 |

### 参数敏感性分析

| 参数 | 增大时的效果 | 减小时的效果 | 风险 |
|------|-------------|-------------|------|
| `β` (beta) | 编码器输出更靠近量化点，但可能降低表达能力 | 量化对编码器约束放松，可能产生不稳定的编码 | 过大 → 重建质量下降 |
| `γ₀` (gamma0) | 样本更均匀地使用码本，但可能过度平滑 | 允许样本集中在少量码字 | 过大 → 信息丢失 |
| `γ` (gamma) | 码本使用更均匀（宏观） | 允许部分码字不被使用（死码） | 通常略大于 γ₀ |
| `ζ` (zeta) | 熵正则化对总损失的贡献增大 | 量化损失主导，可能忽略码本利用率 | 过大 → 训练不稳定 |
| `inv_temperature` | 软熵更"尖锐"（接近硬量化） | 软熵更平滑（梯度更好但更不精确） | 过大 → 梯度消失 |

---

## Group 机制

BSQ 引入了 group_size 参数，将 D 维向量分成 `D / group_size` 个子组：

```
D = 20, group_size = 4
→ 5 个子组，每个子组 4 位
→ 每个子组的码本大小：2^4 = 16
```

**为什么需要分组？** 当 D=20 时，完整的码本大小为 2^20 ≈ 100 万。直接在这个尺度上计算熵需要统计 100 万个条目的频率，计算量大且统计不可靠。分组后，只需要在每个 16 个条目的子码本上计算熵，然后求和近似。

```python
# 熵的分组近似
H_total ≈ Σ H_subgroup_i
```

### DifferentiableEntropyFunction

在非 soft_entropy 模式下，BSQ 使用自定义的 autograd 函数来计算码本熵：

```python
class DifferentiableEntropyFunction(Function):
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        # 计算每个码本条目的计数
        cnt = scatter_reduce(...)  # 统计每个索引出现次数
        prob = (cnt + eps) / (cnt + eps).sum()
        H = -(prob * log(prob)).sum()
        ctx.save_for_backward(zq, zi, prob)
        return H

    @staticmethod
    def backward(ctx, grad_output):
        # 梯度：推动码本使用更均匀
        grad_array = -grad_output * (log(prob) + 1) / numel / K
        ...
```

这个自定义 autograd 函数使得熵的计算可以直接反向传播，梯度引导模型更均匀地使用码本。

---

## BSQuantizer 封装

`BSQuantizer` 是 Kronos 对 `BinarySphericalQuantizer` 的封装，添加了 half 模式支持：

```python
class BSQuantizer(nn.Module):
    def __init__(self, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        self.codebook_dim = s1_bits + s2_bits
        self.bsq = BinarySphericalQuantizer(self.codebook_dim, ...)

    def forward(self, z, half=False, collect_metrics=True):
        z = F.normalize(z, dim=-1)          # L2 归一化
        quantized, bsq_loss, metrics = self.bsq(z)

        if half:
            q_pre = quantized[:, :, :s1_bits]    # 前 s1_bits 维
            q_post = quantized[:, :, s1_bits:]   # 后 s2_bits 维
            z_indices = [bits_to_indices(q_pre), bits_to_indices(q_post)]
        else:
            z_indices = bits_to_indices(quantized)

        return bsq_loss, quantized, z_indices
```

**`half=True`**：将量化结果切分为 s1 和 s2 两部分，分别转换为独立的索引。这是 Kronos 层级令牌体系的基础。

---

## 可复用的设计经验

1. **二值球面量化**是一种高效的向量量化方法，码本天然均匀、计算简单（只需符号判断）、索引编码直接（二进制表示）。

2. **直通估计器**是处理不可导操作的通用技巧，在 VQ-VAE、Gumbel-Softmax 等场景中广泛应用。

3. **分组熵近似**将高维码本的熵计算分解为低维子码本的熵之和，兼顾了计算效率和统计可靠性。

4. **双熵正则化**（样本熵 + 码本熵）从微观和宏观两个层面鼓励码本均匀使用，有效避免死码问题。

---

## 与相关量化方法的对比

BSQ 并非唯一的向量量化方法。以下对比有助于理解 Kronos 选择 BSQ 的原因：

| 维度 | VQ-VAE | Gumbel-Softmax | BSQ（Kronos） |
|------|--------|---------------|---------------|
| **码本结构** | K 个可学习 D 维向量 | K 个可学习 D 维向量 | 无显式码本，由二值向量隐式定义 |
| **量化操作** | 最近邻查找：argmin ‖z - e_k‖ | softmax 采样 | 符号判断：sign(z_i) → {+1, -1} |
| **码本大小** | 通常 512-8192 | 通常 512-8192 | 2^D（D=20 时约 100 万） |
| **梯度传递** | 直通估计器 + commitment loss | Gumbel 噪声使采样可微 | 直通估计器 + 熵正则化 |
| **死码问题** | 严重——需要 EMA 更新、重启等技巧 | 较轻——Gumbel 噪声天然鼓励探索 | 天然避免——二值空间中 ±1 概率趋于 50% |
| **计算复杂度** | O(K·D) 每次查找 | O(K·D) 每次采样 | O(D) 每次量化 |
| **存储开销** | 需存储 K×D 码本矩阵 | 需存储 K×D 码本矩阵 | 零额外存储 |

**为什么 BSQ 适合金融 K 线数据？** 金融 K 线的维度（6 维 OHLCV）相对较低，编码器将其映射到 20 维空间后，BSQ 的 2^20 ≈ 100 万种令牌已足够表达 K 线的变化模式。同时，金融数据中不存在 ImageNet 那样的"语义层次"（如"猫"vs"狗"），因此不需要可学习码本中的语义聚类能力。

---

## 🧪 动手练习

### 练习 1：手动计算量化过程

给定一个 D=8 的归一化向量 `z = [0.2, -0.6, 0.9, -0.1, 0.3, -0.7, 0.4, -0.5]`，手动执行 Step 2-5（二值化 → 缩放 → 索引计算），写出最终的整数索引。

**验证方法**：用以下代码核对答案：
```python
import torch
z = torch.tensor([[0.2, -0.6, 0.9, -0.1, 0.3, -0.7, 0.4, -0.5]])
zhat = torch.where(z > 0, 1, -1)
index = ((zhat + 1) / 2 * torch.tensor([1,2,4,8,16,32,64,128], dtype=torch.float)).sum(dim=-1).long()
print(f"索引: {index.item()}")
```

### 练习 2：观察码本利用率

加载预训练分词器，对一段真实数据编码，统计 s1 和 s2 索引的唯一值数量，评估码本利用率。

```python
import pandas as pd
import torch
from model import KronosTokenizer

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
tokenizer.eval()

df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])
features = df[['open','high','low','close','volume','amount']].values.astype('float32)
features = (features - features.mean(0)) / (features.std(0) + 1e-5)
x = torch.from_numpy(features).unsqueeze(0)

with torch.no_grad():
    indices = tokenizer.encode(x, half=True)
    s1_unique = len(indices[0].unique())
    s2_unique = len(indices[1].unique())
    print(f"s1 唯一令牌: {s1_unique}/{2**10} ({s1_unique/2**10*100:.1f}%)")
    print(f"s2 唯一令牌: {s2_unique}/{2**10} ({s2_unique/2**10*100:.1f}%)")
```

**验证方法**：如果两个利用率都高于 10%，说明码本被合理利用。如果低于 5%，可能说明数据分布过于集中或分词器需要微调。

---

## ✅ 自测清单

- [ ] 我能手动完成一个简单向量的 BSQ 量化过程
- [ ] 我能解释直通估计器（STE）为什么能在不可导操作中传递梯度
- [ ] 我能说明 commit loss 和熵正则化各自优化的目标
- [ ] 我能解释分组（group）机制如何降低熵计算复杂度
- [ ] 我能说出调整 β、γ₀、γ 参数对训练效果的影响方向

---

**文档元信息**
难度：⭐⭐⭐⭐ | 类型：专家设计 | 预计阅读时间：30 分钟
