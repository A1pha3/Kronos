# 源码走读 ⭐⭐⭐⭐

> **目标读者**：想逐行理解 Kronos 核心源码的开发者
> **核心问题**：每个文件、每个类、每个方法的设计意图是什么？
> **前置要求**：建议先完成[系统架构分析](01-system-architecture.md)和[Transformer 设计分析](03-transformer-design.md)，具备 PyTorch 基础知识（`nn.Module`、`autograd`、`F.scaled_dot_product_attention`）

## 学习目标

完成本文后，你将能够：

- [ ] 在源码中定位任意模块的实现并解释其设计意图
- [ ] 追踪 `auto_regressive_inference()` 中的滑动窗口缓冲区管理逻辑
- [ ] 解释直通估计器、共享解码器、DependencyAwareLayer 残差方向等容易忽略的实现细节
- [ ] 识别可复用的代码模式（Mixin、STE、滑动窗口推理、多采样并行）

---

## 源码结构

```
model/
├── __init__.py          # 模型注册与导出
├── kronos.py            # 三大核心类 + 推理函数
└── module.py            # 所有基础神经网络组件
```

本文按数据流顺序逐一解读每个文件的每个模块。

---

## module.py — 基础组件

### 不同可微熵函数 (DifferentiableEntropyFunction)

**文件位置**：`module.py` 第 10-32 行

```python
class DifferentiableEntropyFunction(Function):
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        zb = (zq + 1) / 2                                    # {-1,+1} → {0,1}
        zi = ((zb * basis).sum(-1)).to(torch.int64)           # 二进制 → 十进制索引
        cnt = scatter_reduce(zeros(2^K), 0, zi.flatten(), ones, 'sum')  # 统计每个码字出现次数
        prob = (cnt + eps) / (cnt + eps).sum()                # 频率 → 概率
        H = -(prob * log(prob)).sum()                          # 熵公式
        ctx.save_for_backward(zq, zi, prob)
        return H

    @staticmethod
    def backward(ctx, grad_output):
        zq, zi, prob = ctx.saved_tensors
        # 熵对 zq 的梯度：推动低频码字被更多使用
        grad_array = -grad_output * (log(prob) + 1) / numel / K
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        grad_input = reord_grad.unsqueeze(-1) * zq
        return grad_input, None, None, None, None
```

**设计意图**：这是一个自定义 PyTorch autograd 函数，实现码本熵及其梯度的精确计算。`forward` 计算码本熵，`backward` 提供梯度用于训练。梯度方向是：**增加低频码字的使用概率**，从而鼓励码本均匀利用。

---

### BinarySphericalQuantizer

**文件位置**：`module.py` 第 39-224 行

这是 BSQ 的核心实现。关键方法：

#### quantize()

```python
def quantize(self, z):
    zhat = torch.where(z > 0, +1, -1)  # 符号判断
    return z + (zhat - z).detach()       # 直通估计器
```

3 行代码，核心逻辑极其简洁：正维度取 +1，负维度取 -1，通过 detach 实现梯度穿透。

#### forward()

```python
def forward(self, z, collect_metrics=True):
    zq = self.quantize(z)
    zq = zq * q_scale                     # 缩放

    # 熵计算
    persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
    entropy_penalty = gamma0 * persample_entropy - gamma * cb_entropy

    # Commit loss
    commit_loss = beta * mean((zq.detach() - z)²)

    return zq, commit_loss + zeta * entropy_penalty / inv_temperature, metrics
```

**注意损失的组合方式**：`γ₀ × H_sample - γ × H_codebook`。两个熵项符号相反——γ₀ 项鼓励样本均匀分布（最大化 H_sample），γ 项鼓励码本均匀分布（最大化 H_codebook）。负号是因为梯度下降会最小化损失，而我们实际想最大化熵。

#### soft_entropy_loss()

```python
def soft_entropy_loss(self, z):
    divided_z = rearrange(z, '... (g c) -> ... g c', c=group_size)
    distance = -2 * einsum('... g c, d c -> ... g d', divided_z, group_codebook)
    prob = (-distance * inv_temperature).softmax(dim=-1)

    # 分析模式：直接用 sigmoid 计算
    if persample_entropy_compute == 'analytical':
        p = sigmoid(-4 * z / sqrt(D) * inv_temperature)
        prob = stack([p, 1-p], dim=-1)
        per_sample_entropy = get_entropy(prob, dim=-1).sum(-1).mean()

    # 码本熵：所有样本的平均概率分布的熵
    avg_prob = reduce(prob, '... g d -> g d', 'mean')
    codebook_entropy = get_entropy(avg_prob, dim=-1).sum()

    return per_sample_entropy, codebook_entropy, avg_prob
```

**analytical 模式**：直接用 sigmoid 函数计算每个维度取 ±1 的概率（无需显式枚举子码本），更高效且数值更稳定。

---

### BSQuantizer

**文件位置**：`module.py` 第 225-255 行

```python
class BSQuantizer(nn.Module):
    def forward(self, z, half=False, collect_metrics=True):
        z = F.normalize(z, dim=-1)                    # L2 归一化
        quantized, bsq_loss, metrics = self.bsq(z)
        if half:
            q_pre = quantized[:, :, :s1_bits]
            q_post = quantized[:, :, s1_bits:]
            z_indices = [bits_to_indices(q_pre), bits_to_indices(q_post)]
        else:
            z_indices = bits_to_indices(quantized)
        return bsq_loss, quantized, z_indices
```

**half 模式**的切分是 Kronos 层级令牌的关键连接点。`quantized` 的前 `s1_bits` 维对应 s1，后 `s2_bits` 维对应 s2。

---

### TransformerBlock

**文件位置**：`module.py` 第 465-483 行

```python
def forward(self, x, key_padding_mask=None):
    residual = x
    x = self.norm1(x)
    x = residual + self.self_attn(x, key_padding_mask=key_padding_mask)

    residual = x
    x = self.norm2(x)
    x = residual + self.ffn(x)
    return x
```

标准 Pre-Norm Transformer 块，无特殊逻辑。

---

### HierarchicalEmbedding

**文件位置**：`module.py` 第 400-443 行

```python
def forward(self, token_ids):
    if isinstance(token_ids, (tuple, list)):
        s1_ids, s2_ids = token_ids           # 已分离的 s1/s2
    else:
        s1_ids, s2_ids = self.split_token(token_ids, self.s2_bits)  # 复合 ID → s1/s2

    s1_emb = self.emb_s1(s1_ids) * sqrt(d_model)   # 独立嵌入 + 缩放
    s2_emb = self.emb_s2(s2_ids) * sqrt(d_model)
    return self.fusion_proj(cat([s1_emb, s2_emb], dim=-1))  # 拼接 → 线性融合
```

**split_token()**：

```python
def split_token(self, token_ids, s2_bits):
    t = token_ids.long()
    mask = (1 << s2_bits) - 1
    s2_ids = t & mask          # 低 s2_bits 位
    s1_ids = t >> s2_bits      # 高 s1_bits 位
    return s1_ids, s2_ids
```

位运算实现复合 ID 的拆分。`t & mask` 提取低 s2_bits 位，`t >> s2_bits` 提取高位。

---

### DependencyAwareLayer

**文件位置**：`module.py` 第 446-462 行

```python
def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
    attn_out = self.cross_attn(
        query=sibling_embed,       # s1 的嵌入
        key=hidden_states,         # Transformer 上下文
        value=hidden_states
    )
    return self.norm(hidden_states + attn_out)  # 残差 + RMSNorm
```

**注意残差连接的方向**：`hidden_states + attn_out`，不是 `sibling_embed + attn_out`。残差连接保留的是 Transformer 的上下文信息，交叉注意力的输出作为对上下文的修正。

**完整签名**：实际调用中 `DependencyAwareLayer.forward()` 还接收 `key_padding_mask` 参数（`kronos.py` 第 274 行），用于在填充位置屏蔽交叉注意力的 key/value。上面的伪代码省略了此参数。

---

### DualHead

**文件位置**：`module.py` 第 486-513 行

```python
def forward(self, x):
    return self.proj_s1(x)          # s1 logits

def cond_forward(self, x2):
    return self.proj_s2(x2)         # s2 logits

def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
    # 支持 padding_mask，忽略填充位置的损失
    ce_s1 = cross_entropy(s1_logits, s1_targets)
    ce_s2 = cross_entropy(s2_logits, s2_targets)
    return (ce_s1 + ce_s2) / 2, ce_s1, ce_s2
```

两个独立的线性层，分别映射到 s1 词汇表和 s2 词汇表。`compute_loss()` 支持填充掩码——只在非填充位置计算交叉熵。

---

### TemporalEmbedding

**文件位置**：`module.py` 第 536-562 行

```python
def forward(self, x):
    x = x.long()
    return (hour_embed(x[:,:,1]) + weekday_embed(x[:,:,2])
          + day_embed(x[:,:,3]) + month_embed(x[:,:,4])
          + minute_embed(x[:,:,0]))
```

5 个独立嵌入表求和。`FixedEmbedding` 使用固定的正弦/余弦编码（不参与训练），`nn.Embedding` 使用可学习的编码。

---

## kronos.py — 核心类与推理

### KronosTokenizer

**文件位置**：`kronos.py` 第 13-178 行

**构造函数**：创建嵌入层、编码器、量化器、解码器

```python
def __init__(self, d_in, d_model, ...):
    self.embed = nn.Linear(d_in, d_model)           # 6 → d_model
    self.encoder = ModuleList([TransformerBlock(...) for _ in range(n_enc_layers - 1)])
    self.quant_embed = nn.Linear(d_model, codebook_dim)  # d_model → s1+s2 bits
    self.post_quant_embed_pre = nn.Linear(s1_bits, d_model)
    self.post_quant_embed = nn.Linear(codebook_dim, d_model)
    self.tokenizer = BSQuantizer(s1_bits, s2_bits, ...)
    self.decoder = ModuleList([TransformerBlock(...) for _ in range(n_dec_layers - 1)])
    self.head = nn.Linear(d_model, d_in)            # d_model → 6
```

**注意编码器层数为 `n_enc_layers - 1`**：这是因为在 BSQ 论文的实现中，第一层被 `embed` 线性层替代。`n_enc_layers=4` 实际意味着 3 个 Transformer 层 + 1 个线性层。

**forward()**：

```python
def forward(self, x):
    z = self.embed(x)
    for layer in self.encoder: z = layer(z)
    z = self.quant_embed(z)
    bsq_loss, quantized, z_indices = self.tokenizer(z)

    # 双路解码
    z_pre = self.post_quant_embed_pre(quantized[:, :, :s1_bits])  # 仅 s1 部分
    z = self.post_quant_embed(quantized)                           # 完整码本

    for layer in self.decoder: z_pre = layer(z_pre)   # 粗粒度重建
    z_pre = self.head(z_pre)

    for layer in self.decoder: z = layer(z)            # 细粒度重建（共享解码器！）
    z = self.head(z)

    return (z_pre, z), bsq_loss, quantized, z_indices
```

**关键细节：解码器是共享的**。粗粒度和细粒度的重建使用**同一组**解码器层。这不是一个设计疏忽——共享参数减少了模型大小，且训练时两路重建的梯度共同更新解码器，效果与独立解码器相当。

---

### Kronos

**文件位置**：`kronos.py:198-328`

**构造函数** `kronos.py:198-224`：

```python
def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim, ...):
    self.s1_bits = s1_bits
    self.s2_bits = s2_bits
    self.embedding = HierarchicalEmbedding(s1_bits, s2_bits, d_model)
    self.time_emb = TemporalEmbedding(d_model, learn_te)
    self.token_drop = nn.Dropout(token_dropout_p)
    self.transformer = nn.ModuleList([
        TransformerBlock(d_model, n_heads, ff_dim, ...)
        for _ in range(n_layers)
    ])
    self.norm = RMSNorm(d_model)
    self.head = DualHead(s1_bits, s2_bits, d_model)
    self.dep_layer = DependencyAwareLayer(d_model)
```

**forward()** `kronos.py:239-276`：

```python
def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None,
            use_teacher_forcing=False, s1_targets=None):
    x = self.embedding([s1_ids, s2_ids])       # 层级令牌嵌入
    if stamp is not None: x = x + self.time_emb(stamp)  # 时间特征
    x = self.token_drop(x)                       # 令牌 Dropout

    for layer in self.transformer: x = layer(x, key_padding_mask=padding_mask)
    x = self.norm(x)                             # 最终 RMSNorm

    # s1 预测（kronos.py:263）
    s1_logits = self.head(x)                     # DualHead.forward() → proj_s1

    # s2 条件预测（kronos.py:265-275）
    if use_teacher_forcing:
        sibling_embed = self.embedding.emb_s1(s1_targets)    # 训练：用真实 s1
    else:
        s1_probs = softmax(s1_logits.detach(), dim=-1)
        sample_s1 = multinomial(s1_probs, 1)
        sibling_embed = self.embedding.emb_s1(sample_s1)      # 推理：用采样 s1

    x2 = self.dep_layer(x, sibling_embed)
    s2_logits = self.head.cond_forward(x2)

    return s1_logits, s2_logits
```

**token_drop**：在嵌入后应用 Dropout。在训练中随机将某些令牌的嵌入置零，迫使模型不依赖任何单个令牌，增强鲁棒性。

> **注意**：`decode_s1()` 和 `decode_s2()` 中也包含 `self.token_drop(x)` 调用。由于推理时这些方法是在 `model.eval()` + `torch.no_grad()` 下执行的，Dropout 层不会实际丢弃令牌（`eval()` 模式下 Dropout 是恒等操作）。但如果推理前忘记调用 `model.eval()`，`token_drop` 会导致预测结果不可复现。`auto_regressive_inference()` 使用 `torch.no_grad()` 但没有显式调用 `model.eval()`——`KronosPredictor.generate()` 也没有调用。这意味着如果模型之前处于训练模式，推理结果会是不确定的。正常使用场景中不会出现这个问题（`from_pretrained` 加载的模型默认是 eval 模式），但在自定义训练-推理交替的流程中需要留意。

---

### auto_regressive_inference()

**文件位置**：`kronos.py:389-469`

这是推理的核心函数，值得逐段解读。注意它的默认参数与 `KronosPredictor.predict()` 不同——`predict()` 使用 `top_p=0.9, sample_count=1`，而这里的默认值是 `top_p=0.99, sample_count=5`。由于 `predict()` 会将自己的参数值传递给 `generate()` 再到此处，**用户实际使用时由 `predict()` 的默认值控制**。

```python
def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp,
                               max_context, pred_len, clip=5, T=1.0,
                               top_k=0, top_p=0.99, sample_count=5):
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)

        # sample_count 扩展到 batch 维度（kronos.py:~420）
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1)
        x = x.reshape(-1, x.size(1), x.size(2))
        # 变换: (B, seq_len, feat) → (B, sample_count, seq_len, feat) → (B*sample_count, seq_len, feat)
        # 同样扩展 x_stamp 和 y_stamp

        # 编码历史数据
        x_token = tokenizer.encode(x, half=True)

        # 初始化滑动窗口缓冲区
        pre_buffer = zeros(batch_size, max_context)
        post_buffer = zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        pre_buffer[:, :buffer_len] = x_token[0][:, -buffer_len:]
        post_buffer[:, :buffer_len] = x_token[1][:, -buffer_len:]
```

**滑动窗口管理**：

```python
        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            # 提取当前窗口
            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]  # 固定大小窗口

            # 时间戳对齐
            context_start = max(0, current_seq_len - max_context)
            current_stamp = full_stamp[:, context_start:current_seq_len]

            # 预测 s1
            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]  # 只取最后一个位置
            sample_pre = sample_from_logits(s1_logits, T=T, top_k=top_k, top_p=top_p)

            # 预测 s2
            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, T=T, top_k=top_k, top_p=top_p)

            # 更新缓冲区
            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                # 滑动：左移一位，末尾填入新令牌
                pre_buffer.copy_(roll(pre_buffer, -1, dims=1))
                post_buffer.copy_(roll(post_buffer, -1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)
```

**缓冲区策略**：

- `current_seq_len < max_context`：历史数据未满窗口，直接在末尾追加
- `current_seq_len >= max_context`：窗口已满，左移一位丢弃最早的历史，末尾填入新令牌

`torch.roll` + 赋值末尾元素实现了高效的滑动窗口。

**缓冲区管理的边界情况**：

- **buffer_len = 0**：当 `initial_seq_len = 0` 时（传入空历史），`buffer_len = min(0, max_context) = 0`，缓冲区初始化为零。生成循环中 `current_seq_len = i`，第一步的 `window_len = min(0, max_context) = 0`，此时 `input_tokens` 的序列长度为 0——这会导致 Transformer 的注意力计算异常。实际上，源码中 `predict()` 要求输入 DataFrame 至少包含一行数据（否则后续的数据处理会出错），因此这个边界在正常使用中不会触发。
- **initial_seq_len > max_context**：当输入历史超过窗口大小时，`buffer_len = max_context`，且 `start_idx = initial_seq_len - max_context > 0`。初始化时只保留最后 `max_context` 个令牌，最早的历史被静默丢弃。这与推理过程中的滑动行为一致，但需要注意：如果用户传入 1000 步历史，前 488 步的信息完全不会被模型看到。
- **pred_len = 0**：当 `pred_len = 0` 时，`for i in range(0)` 循环不执行，`generated_pre` 和 `generated_post` 为空张量（形状 `(batch_size, 0)`）。解码阶段会将空张量与历史令牌拼接，最终返回的是历史窗口的重建结果而非预测结果。这在功能上是一个合法操作，但在实际使用中没有意义——用户不会调用预测来获取 0 步预测。源码中没有对 `pred_len` 的显式检查，传入 0 不会报错但结果无意义。

```python
        # 解码所有令牌（历史 + 生成的）
        full_pre = cat([x_token[0], generated_pre], dim=1)
        full_post = cat([x_token[1], generated_post], dim=1)

        z = tokenizer.decode([full_pre[:, context_start:], full_post[:, context_start:]], half=True)

        # reshape 为 (batch, sample_count, seq_len, features) 并取平均
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)

        return preds
```

---

### KronosPredictor

**文件位置**：`kronos.py:484-661`

**构造函数** `kronos.py:484-506`：

```python
def __init__(self, model, tokenizer, device=None, max_context=512, clip=5.0):
    self.model = model
    self.tokenizer = tokenizer
    self.max_context = max_context
    self.clip = clip
    # 设备自动检测：CUDA → MPS → CPU
    self.device = device or self._detect_device()
    self.price_cols = ['open', 'high', 'low', 'close']
    self.vol_col = 'volume'
    self.amt_vol = 'amount'
```

**generate() 方法** `kronos.py:508-517`：

```python
def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):
    x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
    x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
    y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

    preds = auto_regressive_inference(self.tokenizer, self.model, x_tensor, x_stamp_tensor,
                                       y_stamp_tensor, self.max_context, pred_len,
                                       self.clip, T, top_k, top_p, sample_count, verbose)
    preds = preds[:, -pred_len:, :]
    return preds
```

**generate() 作为 predict() 和 auto_regressive_inference() 之间的桥梁**，承担了三项职责：

1. **NumPy 到 Tensor 的转换**：`predict()` 处理的是 NumPy 数组，而 `auto_regressive_inference()` 需要 PyTorch Tensor。`generate()` 负责这个类型转换和设备转移。
2. **参数透传**：将 `KronosPredictor` 的实例属性（`self.max_context`, `self.clip`, `self.device`）和 `predict()` 的采样参数（T, top_k, top_p, sample_count）统一传递给 `auto_regressive_inference()`。
3. **结果截取**：`preds[:, -pred_len:, :]` 截取最后 `pred_len` 步。这一步看似冗余（`auto_regressive_inference()` 理论上只生成 `pred_len` 步），实际上是因为解码阶段会解码历史+生成的全部令牌，返回结果包含了历史重建部分。切片确保只返回预测部分。

这种三层结构（predict → generate → auto_regressive_inference）的分工是：`predict()` 处理数据和标准化，`generate()` 处理 Tensor 转换和设备管理，`auto_regressive_inference()` 处理核心推理逻辑。

**predict() 方法** `kronos.py:519-559`：

```python
def predict(self, df, x_timestamp, y_timestamp, pred_len,
            T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
    # 1. 输入验证（检查必填列、NaN）
    # 2. 补齐 volume/amount
    # 3. 时间特征提取（calc_time_stamps）
    # 4. 标准化 + 裁剪
    # 5. 升维: (N, 6) → (1, N, 6)
    # 6. 调用 generate()
    # 7. 降维 + 反标准化
    # 8. 返回 DataFrame
```

**predict_batch() 方法** `kronos.py:562-661`：

核心差异在于多条序列的 stack 操作和逐条反标准化：

```python
    x_batch = np.stack(x_list, axis=0)          # (B, N, 6)
    x_stamp_batch = np.stack(x_stamp_list, axis=0)
    y_stamp_batch = np.stack(y_stamp_list, axis=0)

    preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, ...)

    # 逐条反标准化——每条序列使用各自的 mean 和 std
    for i in range(num_series):
        preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
        pred_dfs.append(DataFrame(preds_i, columns=..., index=y_timestamp_list[i]))
```

**注意**：每条序列使用各自的 `mean` 和 `std` 进行反标准化，确保不同尺度的序列被正确还原。这在批量预测中尤其重要——如果错误地使用统一的统计量，高价股和低价股的预测结果会被严重扭曲。

---

## __init__.py — 模型注册

```python
from .kronos import KronosTokenizer, Kronos, KronosPredictor

model_dict = {
    'kronos_tokenizer': KronosTokenizer,
    'kronos': Kronos,
    'kronos_predictor': KronosPredictor
}

def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError
```

简洁的工厂模式。`get_model_class()` 按名称查找模型类，目前主要用于内部引用。外部用户通常直接 `from model import Kronos, KronosTokenizer, KronosPredictor`。

---

## 可复用的代码模式

1. **PyTorchModelHubMixin 混入模式**：通过多继承将 HuggingFace Hub 功能注入模型类，不修改模型代码本身

2. **直通估计器模式**：`z + (zhat - z).detach()` — 在前向使用量化值，在反向传递连续值的梯度

3. **滑动窗口推理**：使用固定大小的缓冲区 + `torch.roll` 管理超长序列的推理

4. **多采样并行**：将 `sample_count` 扩展到 batch 维度，一次推理同时生成多个样本

---

## 动手练习

### 练习 1：对照源码理解 BSQ 量化

打开 `model/module.py`，找到 `BinarySphericalQuantizer.quantize()` 方法（约第 82-88 行）。按照以下步骤验证你的理解：

1. 在方法内找到二值化操作 `torch.where(z > 0, +1, -1)`
2. 找到直通估计器 `z + (zhat - z).detach()`
3. 追踪 `quantize()` 在 `forward()` 中的调用位置
4. 追踪 `BinarySphericalQuantizer.forward()` 在 `BSQuantizer.forward()` 中的调用位置

**验证方法**：如果你能画出从 `BSQuantizer.forward()` → `BinarySphericalQuantizer.forward()` → `quantize()` 的完整调用链，并解释每一步的输入输出形状，说明你已理解量化链路。

### 练习 2：追踪滑动窗口缓冲区的生命周期

打开 `model/kronos.py`，找到 `auto_regressive_inference()` 函数。在生成循环（`for i in range(pred_len)`）中，手动模拟前 3 步迭代：

- 假设 `initial_seq_len=5, max_context=8, pred_len=3`
- 写出每一步 `pre_buffer` 和 `post_buffer` 的状态变化

**验证方法**：第 1 步后 `pre_buffer` 的第 5 个位置被填入新令牌；第 3 步后仍无需滚动（因为 5+3=8 ≤ max_context）。如果初始长度改为 8，则第 1 步就需要执行 `torch.roll`。

---

## 自测清单

- [ ] 我能在源码中定位 BSQ 量化的三行核心代码（二值化、直通、缩放）
- [ ] 我能解释 `KronosTokenizer.forward()` 为什么解码器被遍历两次（共享解码器）
- [ ] 我能说明 `auto_regressive_inference()` 中 `torch.roll` 的触发条件
- [ ] 我能解释 `DependencyAwareLayer` 中残差连接方向是 `hidden_states + attn_out` 而非 `sibling_embed + attn_out`
- [ ] 我能说出 `KronosPredictor.predict_batch()` 中每条序列独立反标准化的原因

---

## 常见实现陷阱

以下是修改 Kronos 源码时容易引入的微妙错误：

### 陷阱 1：共享解码器只遍历一次

`KronosTokenizer.forward()` 中，解码器被两个 for 循环分别遍历（s1 重建和完整重建）。如果误将两个循环合并为一个：

```python
# 错误：s_pre 和 z 共享同一次遍历
for layer in self.decoder:
    z_pre = layer(z_pre)
    z = layer(z)
```

这看似正确，但会改变语义。需要明确的是：**在 `forward()` 执行期间，两个循环之间权重不会被更新**——权重更新只发生在反向传播阶段。两个独立循环的真正区别在于**梯度累积**：反向传播时，第一个循环（s1 重建）和第二个循环（完整重建）各自对解码器权重计算的梯度会分别累积。如果合并为单次循环，虽然前向计算结果相同（因为权重在 forward 期间保持不变），但梯度的计算图结构会改变——两次解码器调用将共享同一次循环的中间状态，而非各自独立地经过完整的解码器前向路径。

更深层的原因是：s1 重建（z_pre）和完整重建（z）的输入来自不同的后量化嵌入，它们应该在**语义上独立的**前向路径中被解码器处理，以保持各自梯度的独立性。

### 陷阱 2：DependencyAwareLayer 的残差方向

`DependencyAwareLayer.forward()` 返回 `self.norm(hidden_states + attn_out)`。如果误将残差改为 `self.norm(sibling_embed + attn_out)`：

```python
# 错误：残差连接到 sibling_embed
return self.norm(sibling_embed + attn_out)
```

这会导致每步推理的输出只保留 s1 嵌入的信息，丢失了 Transformer 上下文。后续 `DualHead.proj_s2` 将基于 s1 嵌入而非 Transformer 表示来预测 s2，使整个 Transformer 编码失去意义。

### 陷阱 3：滑动窗口中忘记 contiguous()

`auto_regressive_inference()` 中，`current_stamp = full_stamp[:, context_start:context_end, :].contiguous()`。`.contiguous()` 确保切片后的张量在内存中是连续的。如果省略，某些 PyTorch 操作（尤其是传递给 SDPA 的注意力掩码）可能因步幅不连续而产生错误结果或性能下降。

### 陷阱 4：sample_count 扩展后的 reshape 顺序

多采样并行通过 `x.repeat(1, sample_count, 1, 1).reshape(-1, seq_len, feat)` 实现，其中 `seq_len = x.size(1)`（时间步数），`feat = x.size(2)`（特征数 6）。这个特定的 reshape 顺序确保同一序列的不同采样在 batch 维度上是连续的（[seq0_sample0, seq0_sample1, ..., seq1_sample0, seq1_sample1, ...]）。如果误用 `reshape(sample_count, -1, seq_len, feat)`，batch 维度的排列将被打乱，导致后续 `z.reshape(-1, sample_count, ...)` 还原时对应关系错误。

### 陷阱 5：bits_to_indices 的小端序 vs 大端序

`BSQuantizer.bits_to_indices()` 使用 `2 ** torch.arange(0, bits, 1)`，即 `[1, 2, 4, 8, ...]`（小端序）。而 `BinarySphericalQuantizer.codes_to_indexes()` 使用 `2 ** torch.arange(embed_dim - 1, -1, -1)`，即 `[..., 8, 4, 2, 1]`（大端序）。两者的索引值不同但各自一致。在修改或扩展时，必须注意使用哪个函数，不能混用两套位序。

### 陷阱 6：KronosPredictor 构造函数中的设备转移

`KronosPredictor.__init__()` 会自动执行 `self.tokenizer = self.tokenizer.to(self.device)` 和 `self.model = self.model.to(self.device)`。如果在构造 `KronosPredictor` 之前已经手动将模型移到 GPU，构造函数会再次调用 `.to()`——虽然结果是正确的（模型已在目标设备上），但会产生不必要的设备转移开销。更严重的是，如果之后修改了 `self.model` 的引用但没有同步更新 `KronosPredictor` 持有的引用，推理仍会使用旧设备上的模型。

### 陷阱 7：predict() 的 NaN 检查是严格的

`predict()` 和 `predict_batch()` 中的 NaN 检查会直接抛出 `ValueError`（`kronos.py` 第 534 行），不会尝试静默填充。这与 CSV 微调流水线的 `CustomKlineDataset`（使用前向填充）行为不同。如果你从 CSV 微调流水线切换到直接调用 `predict()`，需要确保输入数据中没有 NaN——否则需要在外部预处理阶段完成填充。

```python
# 如果数据可能包含 NaN，在调用 predict() 前预处理：
df = df.fillna(method='ffill').fillna(0)
```

---

## 相关文档

- [系统架构分析](01-system-architecture.md) — 从全局视角理解模块间的数据流与设计决策
- [Transformer 设计分析](03-transformer-design.md) — 深入理解各 Transformer 组件的设计原理与替代方案
- [开发扩展指南](../advanced/08-development-guide.md) — 基于源码理解进行二次开发与扩展的实践指南

