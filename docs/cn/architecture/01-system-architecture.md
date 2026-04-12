# 系统架构分析 ⭐⭐⭐⭐

> **目标读者**：想从全局视角理解 Kronos 系统设计的开发者与研究者
> **核心问题**：系统由哪些模块组成？数据如何在模块间流转？设计中有哪些关键决策？

## 学习目标

完成本文后，你将能够：

- [ ] 画出 Kronos 的三层架构（用户接口层、模型层、基础组件层）并说明各层职责
- [ ] 追踪一次 `predict()` 调用的完整数据流路径
- [ ] 解释实例级标准化、滑动窗口、层级令牌依赖等关键设计决策的理由与代价
- [ ] 定位"修改某个模块可能影响哪些其他模块"的依赖关系

---

## 设计目标与挑战

### 核心挑战

| 挑战 | 描述 | Kronos 的应对 |
|------|------|---------------|
| 连续数据的离散化 | OHLCV 是多维连续值，无法直接使用 NLP 的令牌建模 | BSQ 量化器将连续向量映射为二值码 |
| 预测精度与表达力的权衡 | 码本太小丢失信息，码本太大难以预测 | 层级令牌（s1 + s2）将大码本分解为两个小码本 |
| 跨市场泛化 | 不同市场、品种、时间粒度的数据特征差异大 | 实例级标准化 + 大规模预训练（45+ 交易所） |
| 长序列建模 | 金融时间序列可能很长，但计算资源有限 | 滑动窗口（max_context=512）+ RoPE 位置编码 |

---

## 模块总览

```
┌────────────────────────────────────────────────────────────────┐
│                         用户接口层                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  KronosPredictor                         │   │
│  │  predict() / predict_batch()                            │   │
│  │  ┌───────────┐ ┌──────────────┐ ┌──────────────────┐   │   │
│  │  │ 数据预处理 │ │ 推理流程编排  │ │ 后处理/反标准化  │   │   │
│  │  └───────────┘ └──────────────┘ └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
├──────────────────────────────┼──────────────────────────────────┤
│                         模型层                                  │
│                              │                                  │
│  ┌───────────────────────────┼───────────────────────────┐     │
│  │                 Kronos (Predictor)                      │     │
│  │  ┌───────────────────┐  │  ┌──────────────────────┐   │     │
│  │  │ HierarchicalEmbed │  │  │ DependencyAwareLayer │   │     │
│  │  └───────────────────┘  │  └──────────────────────┘   │     │
│  │  ┌───────────────────┐  │  ┌──────────────────────┐   │     │
│  │  │ TemporalEmbedding │  │  │ DualHead             │   │     │
│  │  └───────────────────┘  │  └──────────────────────┘   │     │
│  │  ┌────────────────────────────────────────────────┐   │     │
│  │  │ TransformerBlock × N                           │   │     │
│  │  │ ┌──────────┐  ┌──────────┐  ┌──────────────┐  │   │     │
│  │  │ │ RMSNorm  │  │ Attention│  │ FeedForward  │  │   │     │
│  │  │ │          │  │ (RoPE)   │  │ (SwiGLU)     │  │   │     │
│  │  │ └──────────┘  └──────────┘  └──────────────┘  │   │     │
│  │  └────────────────────────────────────────────────┘   │     │
│  └─────────────────────────────────────────────────────────┘     │
│                              │                                  │
│  ┌───────────────────────────┼───────────────────────────┐     │
│  │              KronosTokenizer (Tokenizer)                │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │     │
│  │  │ Embed    │→│ Encoder  │→│ BSQ      │→│ Decoder  │ │     │
│  │  │ (Linear) │ │ (Trans.) │ │Quantizer │ │ (Trans.) │ │     │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                         基础组件层                                │
│  ┌────────────┐ ┌────────┐ ┌─────────┐ ┌───────────┐            │
│  │ RMSNorm    │ │ RoPE   │ │SwiGLU   │ │ BSQ Core  │            │
│  │ (归一化)   │ │(位置)  │ │(前馈)   │ │ (量化核心) │            │
│  └────────────┘ └────────┘ └─────────┘ └───────────┘            │
└───────────────────────────────────────────────────────────────────┘
```

---

## 数据流分析

### 推理时的完整数据流

以下追踪一次 `predictor.predict()` 调用中数据的完整流转路径：

```
输入: DataFrame (N, 6) + x_timestamp + y_timestamp
│
├─ 1. 输入验证与补齐
│   └─ 检查必填列、补齐 volume/amount、检查 NaN
│
├─ 2. 时间特征提取
│   └─ calc_time_stamps(): timestamps → (minute, hour, weekday, day, month)
│   └─ 输出: x_stamp (N, 5), y_stamp (pred_len, 5)
│
├─ 3. 标准化
│   └─ z-score: x = (x - mean) / (std + 1e-5)
│   └─ clip: x = clip(x, -5, 5)
│   └─ 升维: x → (1, N, 6)
│
├─ 4. 分词编码 [KronosTokenizer.encode()]
│   ├─ embed: (1, N, 6) → (1, N, d_model)
│   ├─ encoder: N-1 层 Transformer
│   ├─ quant_embed: (1, N, d_model) → (1, N, 20)
│   ├─ BSQ: (1, N, 20) → z_q ∈ {-1, +1}^20
│   └─ bits_to_indices: 切分并转换为 s1_ids (1, N) + s2_ids (1, N)
│
├─ 5. 自回归推理 [auto_regressive_inference()]
│   │  维护滑动缓冲区: pre_buffer (1, 512) + post_buffer (1, 512)
│   │  注：Kronos-mini 使用 max_context=2048，其他型号默认 max_context=512
│   │
│   ├─ for i in range(pred_len):
│   │   ├─ model.decode_s1(buffer, stamp):
│   │   │   ├─ HierarchicalEmbedding → s1_emb + s2_emb → fusion
│   │   │   ├─ + TemporalEmbedding
│   │   │   ├─ TransformerBlocks × N
│   │   │   ├─ RMSNorm
│   │   │   └─ DualHead.proj_s1 → s1_logits (1, 1, 1024)
│   │   │   └─ 返回: s1_logits[:, -1, :], context
│   │   │
│   │   ├─ sample(s1_logits, T, top_k, top_p) → s1_id
│   │   │
│   │   ├─ model.decode_s2(context, s1_id):
│   │   │   ├─ emb_s1(s1_id) → sibling_embed
│   │   │   ├─ DependencyAwareLayer(context, sibling_embed)
│   │   │   │   └─ CrossAttention(query=sibling, key=context, value=context)
│   │   │   └─ DualHead.proj_s2 → s2_logits (1, 1, 1024)
│   │   │
│   │   ├─ sample(s2_logits, T, top_k, top_p) → s2_id
│   │   │
│   │   └─ 更新缓冲区 (滑动窗口)
│   │
│   └─ 输出: generated_pre (1, pred_len) + generated_post (1, pred_len)
│
├─ 6. 分词解码 [KronosTokenizer.decode()]
│   ├─ indices_to_bits: (s1_ids, s2_ids) → 二值向量 (1, pred_len, 20)
│   ├─ post_quant_embed: (1, pred_len, 20) → (1, pred_len, d_model)
│   ├─ decoder: N-1 层 Transformer
│   └─ head: (1, pred_len, d_model) → (1, pred_len, 6)  ← OHLCV
│
├─ 7. 多采样聚合
│   └─ reshape: (1, sample_count, pred_len, 6)
│   └─ mean: 沿 sample_count 维度取平均
│
├─ 8. 反标准化
│   └─ x_original = x_pred * (std + 1e-5) + mean
│
└─ 9. 输出: DataFrame (pred_len, 6)
```

---

## 关键设计决策

### 1. 实例级标准化 vs 全局标准化

**决策**：每条预测序列独立计算均值和标准差

**理由**：
- 不同股票的价格量级差异可达 1000 倍（贵州茅台 ~2000 vs ST 股 ~1）
- 全局标准化需要预计算全局统计量，不适用于实时推理
- 实例级标准化确保模型输入始终在 [-clip, clip] 范围内

**代价**：每次预测只基于自身历史进行标准化，不同时间窗口的同一股票可能有不同的标准化基准。

### 2. 滑动窗口 vs 完整历史

**决策**：使用固定大小的滑动窗口（max_context=512），丢弃超出窗口的最早历史

**理由**：
- Transformer 的注意力复杂度为 O(n²)，512 是计算效率与信息量的平衡点
- 金融市场的"近期效应"最强——越近的历史对预测越重要
- 固定窗口大小使推理时间可预测

**代价**：超过 512 步的长程依赖无法被模型捕捉。

### 3. s1 → s2 的顺序条件依赖

**决策**：s2 的预测条件依赖于 s1 的采样结果，通过交叉注意力实现

**理由**：
- 分解预测难度：两次 1,024 分类比一次 1,048,576 分类更容易
- s1 提供粗粒度约束，s2 在此约束下精细调整
- 交叉注意力允许 s2 动态关注与 s1 最相关的上下文信息

**替代方案（未采纳）**：独立预测 s1 和 s2，然后组合。缺点是可能产生不一致的预测。

### 从实现可以直接读出的影响

这一设计至少有三个可以直接从实现看出的结果：

1. **词表被拆成两个头**：`DualHead` 分别输出 `2^s1_bits` 和 `2^s2_bits` 个类别，而不是一次输出一个 `2^(s1_bits+s2_bits)` 的超大分类头
2. **第二阶段显式依赖第一阶段**：`decode_s2()` 接收 `s1` 的嵌入作为条件输入，说明 `s2` 不是独立生成的
3. **错误会沿链路传播**：因为 `s2` 依赖 `s1` 的采样结果，所以第一阶段采样出错时，第二阶段会在这个条件上继续生成

这里不进一步把这种结构推导成固定的“训练更稳定”或“信息量守恒”结论，因为仓库中没有给出相应实验或证明。

### 4. Pre-Norm vs Post-Norm

**决策**：使用 Pre-Norm（先归一化，再注意力/前馈）

```python
# Kronos 的 TransformerBlock（Pre-Norm）
x = x + self_attn(norm1(x))
x = x + ffn(norm2(x))
```

**从实现看**：每个 `TransformerBlock` 都是先做 `RMSNorm`，再进入注意力或前馈层，属于典型的 Pre-Norm 结构。文档把它理解为一种偏保守、常见的 Transformer 组织方式，但不把“训练更稳定”写成项目已经验证过的实验结论。

---

## 模型注册机制

所有模型类通过 `PyTorchModelHubMixin` 与 HuggingFace Hub 集成：

```python
class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    ...

class Kronos(nn.Module, PyTorchModelHubMixin):
    ...
```

这使得模型支持以下功能：

- `from_pretrained("NeoQuasar/Kronos-small")`：从 Hub 下载并加载
- `save_pretrained("./my_model")`：保存到本地（包含 config.json + model.safetensors）
- 自动处理模型配置的序列化与反序列化

模型工厂函数在 `model/__init__.py` 中注册：

```python
model_dict = {
    'kronos_tokenizer': KronosTokenizer,
    'kronos': Kronos,
    'kronos_predictor': KronosPredictor
}
```

---

## 微调流水线架构

```
finetune/  (Qlib 微调)
├── config.py              # Config 类，硬编码配置
├── dataset.py             # QlibDataset，从 pickle 加载 Qlib 数据
├── train_tokenizer.py     # 分词器 DDP 训练脚本
├── train_predictor.py     # 预测模型 DDP 训练脚本
├── qlib_data_preprocess.py# Qlib 数据预处理
├── qlib_test.py           # 测试与回测
└── utils/training_utils.py# 共享工具（DDP 设置、种子、格式化）

finetune_csv/  (CSV 微调)
├── config_loader.py       # YAML 配置加载器
├── finetune_tokenizer.py  # 分词器训练（支持单卡/分布式）
├── finetune_base_model.py # 预测模型训练 + CustomKlineDataset
└── train_sequential.py    # 顺序训练编排器（SequentialTrainer）
```

两种流水线的核心训练逻辑相同（相同的损失函数、优化器、学习率调度器），区别在于数据加载和配置管理方式。

---

## 模块修改影响矩阵

如果你打算修改某个模块，下表帮助你快速了解影响范围：

| 修改目标 | 直接影响的模块 | 需要同步检查 | 示例场景 |
|---------|---------------|-------------|---------|
| BSQ 量化器参数 | KronosTokenizer | Kronos（HierarchicalEmbedding 的词汇表大小）、所有微调流水线 | 调整 s1_bits/s2_bits |
| Transformer 层数/维度 | Kronos、KronosTokenizer | 推理内存占用、训练时间 | 增大 d_model 提升表达力 |
| 标准化方式 | KronosPredictor | predict_batch() 中的反标准化逻辑 | 改用 sliding-window 标准化 |
| 时间特征 | TemporalEmbedding | predict() 中 calc_time_stamps()、所有数据集类 | 增加"季度"特征 |
| 滑动窗口大小 | auto_regressive_inference | KronosPredictor 的 max_context 参数 | 增大到 1024 需更多内存 |

---

## 🧪 动手练习

### 练习 1：追踪一次预测的数据流

打开 `model/kronos.py`，找到 `KronosPredictor.predict()` 方法，按照本文"推理时的完整数据流"逐行对照源码，在每个关键步骤处标注张量形状。

**验证方法**：如果你能在每个步骤旁写出正确的张量形状注释（如 `(1, 400, 6)` → `(1, 400, d_model)`），说明你已理解数据流。

### 练习 2：设计一个模块修改方案

假设你需要将 `max_context` 从 512 增大到 1024，列出所有需要修改的文件和参数，并分析对内存和速度的影响。

---

## ✅ 自测清单

- [ ] 我能解释为什么 Kronos 使用"令牌化 + 自回归"而非直接回归
- [ ] 我能说出实例级标准化的优势和代价
- [ ] 我能解释滑动窗口在推理中如何管理超长历史
- [ ] 我能在源码中快速定位任意模块的实现位置
- [ ] 我能评估修改某个组件对系统其他部分的影响

---

## 可借鉴的实现观察

下面这些结论是基于当前仓库实现可以稳定读出的观察，适合作为二次开发时的检查清单：

1. **接口层与模型层解耦**：`KronosPredictor` 负责数据清洗、标准化、时间特征与推理编排，`Kronos` 和 `KronosTokenizer` 负责核心建模逻辑。这让你可以替换上层数据流程，而不必直接改模型主体。
2. **层级令牌会牵动多处定义**：`s1_bits` / `s2_bits` 不只是模型超参数，还会影响嵌入表、分类头、分词器量化维度与权重兼容性。改动它们时，不能只改一处。
3. **实例级标准化是接口契约的一部分**：当前预测入口在内部统一执行均值、标准差与裁剪逻辑。如果你改预处理策略，需要同步评估预测入口、批量入口和示例脚本的一致性。
4. **Hub 集成是模型类的基础能力**：`PyTorchModelHubMixin` 直接挂在模型类上，因此本地路径加载与 HuggingFace Hub 加载共享同一套接口。
5. **分词器训练态与推理态路径不同**：`forward()`、`encode()`、`decode()` 各自承担不同职责。做源码扩展时，不能只验证训练路径而忽略推理路径。

---

**文档元信息**
难度：⭐⭐⭐⭐ | 类型：专家设计 | 预计阅读时间：30 分钟 | 更新日期：2026-04-11
