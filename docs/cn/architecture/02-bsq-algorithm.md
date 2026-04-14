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

### BSQ 的几何直觉

理解 BSQ 的关键在于"超球面象限"的几何图像：

- L2 归一化将所有向量投影到**单位超球面**（D 维空间中所有长度为 1 的向量组成的曲面）
- 二值化将超球面划分为 **2^D 个"象限"**——就像 3D 空间被 x、y、z 三个坐标轴的正负分成 8 个卦限一样
- 每个象限的中心就是一个码本条目，所有码本条目在超球面上**均匀分布**
- 量化操作就是判断输入向量落入哪个象限——只需检查每个维度的正负号

这种几何结构解释了为什么 BSQ 天然避免了死码问题：象限是等大小划分的，不存在"没有向量落入"的象限。同时也解释了为什么 BSQ 不需要存储显式码本——码本条目由二值向量的所有可能组合隐式定义。

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

**为什么 STE 梯度不会爆炸？** STE 将量化操作的梯度设为恒等映射，这意味着来自后续层的梯度会原封不动地传回编码器。但实践中，BSQ 的训练稳定性由以下因素共同保证：

1. **L2 归一化**：量化前 `F.normalize(z, dim=-1)` 将向量投影到单位球面，隐式约束了 z 的范数，即使梯度恒等传递，编码器的输出值域也被限制在 [-1, 1] 范围内。
2. **commit loss**：`beta * mean((zq.detach() - z)^2)` 提供了额外的锚定力，推动连续的 z 向量化点 zq 靠近。这个损失对 z 的梯度是 `2 * beta * (z - zq)`，是一个有界的修正项。
3. **缩放因子 q_scale**：量化后乘以 `1/sqrt(D)` 使 zq 的范数为 1，后续层的梯度在通过这个缩放因子时被除以 sqrt(D)（D=20 时约为 4.47），自然衰减了梯度幅度。

因此，虽然 STE 本身不衰减梯度，但 L2 归一化 + commit loss + 缩放因子的组合有效地控制了梯度量级。

### Step 4：缩放

```python
q_scale = 1.0 / sqrt(embed_dim)
zq = zq * q_scale
```

缩放因子使量化向量的 L2 范数为 1（因为 `||zq||_2 = sqrt(D)`，缩放后为 `sqrt(D) / sqrt(D) = 1`）。

### Step 5：索引计算

源码使用**大端序**（most significant bit first）计算索引：

```python
# 源码 module.py:68 — 大端序 basis
basis = 2 ** torch.arange(embed_dim - 1, -1, -1)
# D=4 时：basis = [8, 4, 2, 1]

# 索引计算 module.py:169
b = (zhat + 1) / 2                           # {-1, +1} → {0, 1}
index = (b * basis).sum(dim=-1).to(torch.int64)  # 加权求和
```

**大端序的含义**：二值向量的第一个维度对应最高有效位。例如 `[1, 0, 1, 0]` 在大端序下表示二进制数 `1010`（十进制 10），而非 `0101`（十进制 5）。

### 数值示例

给定一个 D=4 的归一化后向量 `z = [0.3, -0.8, 0.1, -0.5]`：

```
Step 1: 二值化（逐维度判断正负号）
  z[0]=0.3  > 0 → +1    z[1]=-0.8 < 0 → -1
  z[2]=0.1  > 0 → +1    z[3]=-0.5 < 0 → -1
  zhat = [+1, -1, +1, -1]

Step 2: 转换为 0/1
  b = (zhat + 1) / 2 = [1, 0, 1, 0]

Step 3: 计算索引（大端序，basis = [8, 4, 2, 1]）
  index = 1×8 + 0×4 + 1×2 + 0×1 = 10
  等价于二进制 1010 = 十进制 10

Step 4: 缩放
  q_scale = 1/√4 = 0.5
  zq = [+1, -1, +1, -1] × 0.5 = [0.5, -0.5, 0.5, -0.5]
  ||zq||₂ = √(0.25×4) = 1.0 ✓
```

**端序为什么重要**：Kronos 的 BSQ 使用大端序意味着二值向量的**前几个维度的变化**对索引的影响更大。结合层级令牌设计——前 `s1_bits` 维对应 s1，后 `s2_bits` 维对应 s2——每个"半边"内部的索引计算也是大端序的。

---

## 损失函数

BSQ 的损失由三部分组成（源码 `module.py:111-126`）：

```python
# 源码中的实际计算
entropy_penalty = gamma0 * persample_entropy - gamma * cb_entropy
commit_loss = beta * mean(((zq.detach() - z) ** 2).sum(dim=-1))
total_loss = commit_loss + zeta * entropy_penalty / inv_temperature
```

### 1. Commit Loss（承诺损失）

```python
# module.py:119
commit_loss = beta * mean(((zq.detach() - z) ** 2).sum(dim=-1))
```

**含义**：衡量量化前后向量的欧氏距离（按向量维度求和后取均值）。推动编码器输出的连续向量靠近其量化点。

**为什么 detach zq？** 我们希望梯度推动 z 向 zq 靠近，而不是推动 zq 向 z 靠近（zq 是由 STE 产生的固定量化点）。注意源码中对每个向量求 `sum(dim=-1)` 再取 `mean`——这意味着 commit loss 对每个维度的偏差是等权求和的，与 BSQ 的"每个维度等价"假设一致。

### 2. Entropy Penalty（熵正则化）

熵正则化包含两个部分，作用方向相反：

#### Per-Sample Entropy（样本熵）

```python
H_sample = -Σ p_i × log(p_i)  （对每个子组的概率分布计算熵）
```

**目标**：最大化样本熵（通过 `-γ₀ × H_sample` 实现）。这鼓励每个样本的量化结果更均匀地分布在码本中，避免所有样本都映射到少数几个码本条目。

**analytical 模式下的计算**（源码 `module.py:141-146`）：

```python
p = sigmoid(-4 * z / sqrt(D) * inv_temperature)  # 每个维度取 +1 的概率
prob = stack([p, 1-p], dim=-1)                    # 每个维度 [p(+1), p(-1)]
per_sample_entropy = get_entropy(prob, dim=-1).sum(dim=-1).mean()
```

在 analytical 模式下，不依赖子码本查找，而是直接用 sigmoid 函数计算每个维度取 ±1 的概率。这更高效且数值更稳定——每个维度的"软熵"独立计算后求和。

**analytical 模式的数学推导**：为什么可以用 sigmoid 近似？

在 group 模式下，子码本有 2 个条目（每个维度的正/负），距离计算为 `distance = -2 * z * c`，其中 c ∈ {-1, +1}。softmax 概率为：

```
p(+1) = exp(2z / tau) / (exp(2z / tau) + exp(-2z / tau))
      = sigmoid(4z / tau)
```

其中 tau = sqrt(D) / inv_temperature。当 L2 归一化启用时，z 被归一化到单位球面，此时 `sigmoid(-4 * z / sqrt(D) * inv_temperature)` 正是上述公式的实现。analytical 模式直接使用这个闭式解，避免了显式构造子码本和计算 softmax 的开销，且由于 sigmoid 是单调函数，其数值稳定性优于在大码本上做 softmax（后者可能因指数溢出导致 NaN）。

#### Codebook Entropy（码本熵）

```python
# module.py:151-152
avg_prob = reduce(prob, '... g d -> g d', 'mean')  # 所有样本的平均概率分布
H_codebook = -Σ (avg_prob_i) × log(avg_prob_i)
```

**目标**：最大化码本熵（通过 `-γ × H_codebook` 实现）。这从宏观层面鼓励码本被均匀使用。注意 `avg_prob` 是对 batch 中所有样本的概率取平均后再计算熵，而非对每个样本的熵取平均。这种"先平均再计算"的方式能更好地反映码本的整体使用均匀度。

#### 软熵计算（group 模式）

当 `persample_entropy_compute='group'` 时（源码 `module.py:131-155`），BSQ 使用子码本距离来计算软熵：

```python
# 将 z 按组大小切分
divided_z = rearrange(z, '... (g c) -> ... g c', c=group_size)

# 计算与子码本的距离
distance = -2 * einsum('... g c, d c -> ... g d', divided_z, group_codebook)
prob = (-distance * inv_temperature).softmax(dim=-1)
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

### 边界情况：所有正则化参数为零

当 `beta=0, gamma0=0, gamma=0` 时（zeta 的值不再有意义），BSQ 的总损失变为 0。这意味着：

- **编码器没有约束**：编码器可以输出任意向量，量化操作仍然生效（二值化和 STE 仍在），但没有损失推动编码器输出"更好"的连续表示。
- **量化仍然工作**：BSQ 的量化操作（L2 归一化、二值化、STE）不依赖损失函数，前向传播仍会产生有效的离散令牌。但这些令牌的质量完全取决于编码器的初始化——没有任何力量推动它学习有意义的表示。
- **训练信号仅来自下游损失**：在 Kronos 的分词器训练中，重建损失（MSE）是主要训练信号。如果 BSQ 的正则化全为零，只有重建损失通过 STE 传回编码器。模型可能仍然可以训练，但码本利用率可能很低——大部分数据可能集中在少数几个码字上。
- **实际意义**：这种配置在实践中不推荐。beta 提供编码器与量化点的对齐，gamma0/gamma 确保码本被均匀使用。三者都为零时，模型失去了引导码本分布的能力，可能导致严重的"码本塌缩"（codebook collapse）。

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

**计算复杂度分析**：

| 模式 | 操作 | 复杂度 | 空间开销 |
|------|------|--------|---------|
| 无分组 | 统计 2^D 个码字频率 | O(N * 2^D) | O(2^D) |
| group_size=4 | 统计 (D/4) 组，每组 2^4=16 个码字 | O(N * D) | O(D/4 * 16) |
| group_size=9 | 统计 (D/9) 组，每组 2^9=512 个码字 | O(N * D * 512/9) | O(D/9 * 512) |
| analytical | 每维度一次 sigmoid | O(N * D) | O(N * D) |

以默认配置 D=20、group_size=4 为例：无分组需要处理 2^20 ≈ 100 万条目，分组后只需处理 5 * 16 = 80 条目，复杂度降低了约 13000 倍。analytical 模式更进一步，完全不需要显式码本查找。这种效率提升在每步训练中都会发生，对总训练时间的影响是实质性的。

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

**关于 codebook_dim 的说明**：在默认配置中，`s1_bits=10` 且 `s2_bits=10`，因此 `codebook_dim = s1_bits + s2_bits = 20`。但这并非硬编码值——`s1_bits` 和 `s2_bits` 是构造函数的可配置参数，可以根据任务需求调整（例如增大到 12+12=24 以获得更大的码本容量，或减小到 8+8=16 以降低预测难度）。修改这两个参数会直接影响 `HierarchicalEmbedding` 的词汇表大小和 `DualHead` 的输出维度，需要同步调整整个模型。

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
# 注意：此处使用小端序 [1,2,4,8,...]，与 bits_to_indices() 一致
# 而主文中的 codes_to_indexes() 使用大端序 [128,64,32,...,1]，两者计算的索引值不同
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
难度：⭐⭐⭐⭐ | 类型：专家设计 | 预计阅读时间：30 分钟 | 更新日期：2026-04-11
