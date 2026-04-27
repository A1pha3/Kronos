# 术语表

Kronos 项目中出现的中英文术语及其定义。遇到不熟悉的术语时回来查阅。

### 学习目标

熟悉这份术语表后：

- [ ] 能准确解释 BSQ、层级令牌、依赖感知层等 Kronos 核心概念
- [ ] 能区分容易混淆的术语（如 Kronos 分词器 vs NLP 分词器、Decoder 在不同语境下的含义）
- [ ] 能快速定位任意术语的详细文档出处

---

## 核心概念

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| 基础模型 | Foundation Model | 在大规模数据上预训练的通用模型，可适配多种下游任务 |
| K 线 / 蜡烛图 | Candlestick / K-line | 以矩形和线段表示一定时间周期内开盘价、最高价、最低价、收盘价的图表 |
| OHLCV | OHLCV | Open（开盘价）、High（最高价）、Low（最低价）、Close（收盘价）、Volume（成交量）的缩写 |
| 令牌 / 词元 | Token | 离散化的最小表示单元。Kronos 中每根 K 线被编码为一对令牌 (s1, s2) |
| 分词器 | Tokenizer | 将连续的 OHLCV 数据转换为离散令牌序列的模型组件。Kronos 的分词器（KronosTokenizer）使用 BSQ 量化器实现离散化，是两阶段框架的第一阶段 |
| 自回归 | Autoregressive | 一种生成模式：按时间顺序逐步预测，每一步基于之前所有已生成的结果。Kronos 使用自回归方式逐时间步预测未来的 s1 和 s2 令牌 |
| 两阶段框架 | Two-Stage Framework | Kronos 的架构设计：第一阶段将连续数据离散化，第二阶段在离散空间进行预测 |

---

## 模型组件

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| 编码器 | Encoder | KronosTokenizer 中将 OHLCV 映射为潜在表示的 Transformer 层 |
| 解码器 | Decoder | KronosTokenizer 中将潜在表示还原为 OHLCV 的 Transformer 层 |
| 量化器 | Quantizer | 将连续向量映射为离散码本的组件。Kronos 使用 BSQ 量化器 |
| 码本 | Codebook | 量化器中所有可能离散值的集合。BSQ 的码本大小为 2^D |
| 层级令牌 | Hierarchical Token | Kronos 将每根 K 线编码为两个层级的令牌：s1（粗粒度）和 s2（细粒度）。参见"二值球面量化"、"依赖感知层" |
| s1 令牌 | s1 Token / Pre Token | 粗粒度令牌，使用码本的前 s1_bits 位，捕捉主要走势。参见"s2 令牌"、"层级令牌" |
| s2 令牌 | s2 Token / Post Token | 细粒度令牌，使用码本的后 s2_bits 位，提供精细修正。参见"s1 令牌"、"依赖感知层" |
| 双头 | DualHead | Kronos 模型的输出层，分别预测 s1 和 s2 的 logits |
| 依赖感知层 | DependencyAwareLayer | 使用交叉注意力实现 s2 对 s1 条件依赖的模块。参见"交叉注意力"、"层级令牌" |
| 层级嵌入 | HierarchicalEmbedding | 将 s1 和 s2 令牌嵌入并融合为统一向量的模块 |
| 时间嵌入 | TemporalEmbedding | 将时间特征（分钟、小时等）编码为向量的模块 |
| 令牌嵌入 | Token Embedding | 将离散令牌 ID 映射为连续向量的查找表 |

---

## 算法术语

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| 二值球面量化 | Binary Spherical Quantization (BSQ) | 将连续向量量化为二值向量（±1）的方法，码本天然均匀分布 |
| 直通估计器 | Straight-Through Estimator (STE) | 处理不可导操作（如量化）的技巧：前向用离散值，反向传递连续值的梯度 |
| 承诺损失 | Commit Loss | 量化损失的一部分，衡量量化前后向量的距离，推动编码器输出靠近量化点。参见"熵正则化" |
| 熵正则化 | Entropy Regularization | 鼓励码本被均匀使用的正则化项，避免"死码"问题。参见"码本塌缩"、"承诺损失" |
| 样本熵 | Per-Sample Entropy | 单个样本在各子组上的熵，鼓励每个样本的量化结果均匀分布 |
| 码本熵 | Codebook Entropy | 所有样本的平均概率分布的熵，从宏观层面鼓励码本均匀使用 |
| 分组近似 | Group Approximation | 将高维码本的熵分解为低维子码本熵之和的近似方法 |
| 温度采样 | Temperature Sampling | 通过除以温度参数 T 控制采样分布尖锐程度的策略。参见"核采样"、"Top-k 采样"、"贪婪解码" |
| 核采样 | Nucleus Sampling / Top-p | 保留累计概率达到 p 的候选令牌的采样策略。参见"温度采样"、"Top-k 采样" |
| Top-k 采样 | Top-k Sampling | 只保留概率最高的 k 个候选令牌的采样策略。参见"核采样"、"温度采样" |
| 贪婪解码 | Greedy Decoding | 始终选择概率最高的令牌，等价于 T→0 或 top_k=1。参见"温度采样"、"Top-k 采样" |
| Teacher Forcing | Teacher Forcing | 训练时使用真实目标而非模型预测作为下一步输入的技术。参见"暴露偏差" |
| 暴露偏差 | Exposure Bias | Teacher Forcing 的副作用：训练时模型看到的是真实标签，推理时只能看到自身的预测结果，两者分布不一致可能导致误差累积。参见"Teacher Forcing" |
| 实例级标准化 | Instance-Level Normalization | 对每条预测序列独立计算均值和标准差的 z-score 标准化方法，是 KronosPredictor 的默认行为。详见"数据相关"分组 |
| z-score 标准化 | Z-Score Normalization | (x - mean) / std，将数据变换为零均值单位方差 |
| 码本塌缩 | Codebook Collapse | 量化器中大部分码字从未被使用的现象，模型只集中使用极少数码字，导致表达能力严重浪费。BSQ 通过熵正则化天然避免此问题。参见"熵正则化" |
| 大端序 / 小端序 | Big-Endian / Little-Endian | 索引计算中二值位到整数的映射方式。Kronos 中 `BinarySphericalQuantizer.codes_to_indexes()` 使用大端序（高位在前），`BSQuantizer.bits_to_indices()` 使用小端序（低位在前）。详见 [BSQ 量化算法原理](../architecture/02-bsq-algorithm.md) |
| 令牌 Dropout | Token Dropout | 训练时以 `token_dropout_p` 概率将令牌嵌入置零的正则化手段，防止模型过度依赖特定令牌位置 |
| 置信区间 | Confidence Interval | 通过多次采样预测结果计算的概率区间（如 p5-p95），用于量化预测的不确定性 |
| 蒙特卡洛采样 | Monte Carlo Sampling | 通过多次随机采样估计概率分布特征的方法。Kronos 中通过 `sample_count > 1` 实现多条独立预测路径 |

---

## Transformer 组件

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| 自注意力 | Self-Attention | 序列内部各位置之间计算注意力的机制。参见"交叉注意力" |
| 交叉注意力 | Cross-Attention | 两个不同序列之间计算注意力的机制（query 来自一个序列，key/value 来自另一个）。参见"自注意力"、"依赖感知层" |
| 多头注意力 | Multi-Head Attention | 使用多组独立的注意力头并行计算，捕捉不同子空间的信息。参见"自注意力"、"交叉注意力" |
| 旋转位置编码 | Rotary Positional Embedding (RoPE) | 通过旋转矩阵将相对位置信息编码到 q 和 k 中的位置编码方法 |
| 前馈网络 | Feed-Forward Network (FFN) | Transformer 块中的逐位置非线性变换层 |
| SwiGLU | Swish-Gated Linear Unit | 门控激活函数：SiLU(W1(x)) × W3(x)，比 ReLU 表达力更强 |
| RMSNorm | Root Mean Square Normalization | 不减去均值的归一化方法，计算效率高于 LayerNorm |
| Pre-Norm | Pre-Normalization | 在注意力/前馈操作之前进行归一化的 Transformer 变体 |
| 残差连接 | Residual Connection | 将层的输出与输入相加（x + f(x)），缓解梯度消失 |
| 因果掩码 | Causal Mask | 确保每个位置只能关注之前位置的注意力掩码 |
| 填充掩码 | Padding Mask | 在批量处理中标记和忽略填充位置的掩码 |
| half 模式 | Half Mode | BSQ 量化器 `half=True` 时的行为，将量化结果切分为前 s1_bits 维（s1 令牌）和后 s2_bits 维（s2 令牌），形成层级令牌体系 |
| 滑动窗口缓冲区 | Sliding Window Buffer | 自回归推理中管理历史令牌的固定大小缓冲区，通过 `torch.roll` 滚动更新，确保每次推理只使用最近 `max_context` 个令牌 |

---

## 训练相关

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| 微调 | Fine-tuning | 在预训练模型基础上，使用特定数据继续训练以适应新任务。参见"预训练"、"两阶段框架" |
| 预训练 | Pre-training | 在大规模通用数据上训练模型参数的初始阶段，使模型获得通用表示能力。微调在此基础之上进行 |
| 分布式数据并行 | Distributed Data Parallel (DDP) | 多 GPU 训练策略，每个 GPU 持有模型副本，处理不同数据子集 |
| 梯度累积 | Gradient Accumulation | 将多个小批量的梯度累积后再更新参数，模拟更大的批量大小 |
| 梯度裁剪 | Gradient Clipping | 将梯度范数限制在阈值内，防止梯度爆炸 |
| 学习率调度 | Learning Rate Schedule | 训练过程中动态调整学习率的策略 |
| OneCycleLR | OneCycleLR | 先升后降的学习率策略，先快速上升到峰值再缓慢下降 |
| 交叉熵损失 | Cross-Entropy Loss | 分类任务的标准损失函数，衡量预测分布与真实分布的差异。预测模型微调使用交叉熵衡量令牌预测准确率 |
| MSE 损失 | Mean Squared Error Loss | 回归任务的损失函数，衡量预测值与真实值的均方误差。分词器微调使用的重建损失即 MSE |
| Qlib | Qlib | 微软开源的量化投资平台，提供中国 A 股数据接口 |
| HuggingFace Hub | HuggingFace Hub | 模型和数据集的托管平台，Kronos 的预训练模型存储于此 |
| Safetensors | Safetensors | 一种安全的模型权重文件格式，避免了 pickle 的安全风险 |

---

## 数据相关

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| 滑动窗口 | Sliding Window | 在时间序列上按固定大小移动的窗口，用于采样训练数据和推理时的上下文管理。参见"最大上下文"、"滑动窗口缓冲区" |
| 回看窗口 | Lookback Window | 模型输入使用的历史数据长度。参见"预测窗口"、"最大上下文" |
| 预测窗口 | Prediction Window | 模型预测的未来数据长度。参见"回看窗口" |
| 最大上下文 | Max Context | 模型能处理的最大令牌序列长度。Kronos-small/base/large 默认 512，Kronos-mini 为 2048。超过此长度的历史令牌被丢弃 |
| 裁剪 | Clip | 将标准化后的值限制在 `[-clip, clip]` 范围内（默认 clip=5），抑制极端异常值的影响。KronosPredictor 在 z-score 标准化后自动执行裁剪 |
| amount 推算 | Amount Imputation | 当输入 DataFrame 有 `volume` 但无 `amount` 时，KronosPredictor 自动推算：`amount = volume × mean(open, high, low, close)`（逐行算术均价，非成交量加权均价） |
| 涨跌停 | Price Limit | A 股市场中单日价格涨跌幅不超过特定比例的限制。主板 ±10%，创业板/科创板 ±20%，ST 股 ±5%，北交所 ±30% |
| 复权 | Adjusted Price | 对历史价格进行分红、送股等因素的调整，使价格序列连续可比。Kronos 不要求复权数据 |
| 停牌 | Trading Suspension | 股票暂停交易的状态，数据中可能记录为开盘价为 0 或 NaN，需要在预测前处理 |

---

## API 与组件

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| KronosPredictor | KronosPredictor | Kronos 的高层预测接口（`kronos.py:482-662`），将数据预处理、分词编码、自回归推理、解码还原和后处理封装为 `predict()` 和 `predict_batch()` 两个方法 |
| predict() | predict() | 单序列预测方法，默认参数为 `T=1.0, top_k=0, top_p=0.9, sample_count=1` |
| predict_batch() | predict_batch() | 批量预测方法，要求所有序列具有相同的历史长度 |
| auto_regressive_inference | auto_regressive_inference | 自回归推理函数（`kronos.py:389-469`），维护滑动窗口缓冲区逐时间步生成令牌。默认参数为 `top_p=0.99, sample_count=5` |
| sample_from_logits | sample_from_logits | 令牌采样函数（`kronos.py:373-386`），支持温度、top-k、top-p 过滤 |
| generate() | generate() | KronosPredictor 内部方法，封装 tensor 设备转移和 `auto_regressive_inference()` 调用 |
| decode_s1 | decode_s1 | Kronos 模型的分步解码方法（`kronos.py:278-308`），返回 s1 logits 和 Transformer 上下文 |
| decode_s2 | decode_s2 | Kronos 模型的分步解码方法（`kronos.py:310-328`），基于 s1 采样结果和 Transformer 上下文预测 s2 |
| PyTorchModelHubMixin | PyTorchModelHubMixin | HuggingFace 提供的混入类，为模型添加 `from_pretrained()` 和 `save_pretrained()` 能力 |

---

## 模型标识

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| Kronos-mini | Kronos-mini | 最小规模模型（4.1M 参数），上下文长度 2048，使用专用分词器 `Kronos-Tokenizer-2k` |
| Kronos-small | Kronos-small | 小规模模型（24.7M 参数），上下文长度 512，使用 `Kronos-Tokenizer-base` 分词器 |
| Kronos-base | Kronos-base | 标准规模模型（102.3M 参数），上下文长度 512，使用 `Kronos-Tokenizer-base` 分词器 |
| Kronos-large | Kronos-large | 最大规模模型（499.2M 参数），上下文长度 512，使用 `Kronos-Tokenizer-base` 分词器。未开源 |
| Kronos-Tokenizer-2k | Kronos-Tokenizer-2k | Kronos-mini 专用分词器，支持 2048 上下文长度。与其他模型的分词器不通用 |
| Kronos-Tokenizer-base | Kronos-Tokenizer-base | Kronos-small/base/large 共用分词器，支持 512 上下文长度。可在三个模型间自由切换而无需更换 |

---

## 相关文档

术语的详细解释分散在各主题文档中，以下是快速索引：

| 术语 | 详细文档 |
|------|---------|
| BSQ、量化器、承诺损失、熵正则化 | [BSQ 量化算法原理](../architecture/02-bsq-algorithm.md) |
| s1/s2、层级令牌、DependencyAwareLayer | [层级令牌体系](../core-concepts/05-hierarchical-tokens.md) |
| KronosTokenizer、编码器、解码器 | [KronosTokenizer 详解](../core-concepts/02-tokenizer.md) |
| Transformer 组件、RoPE、SwiGLU、Pre-Norm | [Transformer 设计分析](../architecture/03-transformer-design.md) |
| 自回归推理、滑动窗口、采样策略 | [KronosPredictor 使用指南](../core-concepts/04-predictor.md) |
| 源码实现细节、行号引用 | [源码走读](../architecture/04-source-code-walkthrough.md) |
| 涨跌停、A 股数据处理 | [A 股市场预测实战](../advanced/04-cn-markets.md) |
| 数据预处理、amount 推算、裁剪、标准化流程 | [KronosPredictor 使用指南](../core-concepts/04-predictor.md) |
| 模型选型、参数量、上下文长度、分词器搭配 | [模型对比与选型指南](../advanced/07-model-comparison.md) |
| 微调、DDP、Qlib、训练超参数 | [Qlib 微调指南](../advanced/01-finetune-qlib.md) / [CSV 微调指南](../advanced/02-finetune-csv.md) |

## Kronos 特有术语

以下术语是 Kronos 项目中特有的，在其他项目中可能含义不同：

| 术语 | Kronos 中的含义 | 容易混淆的点 |
|------|----------------|-------------|
| Tokenizer（分词器） | KronosTokenizer，包含编码器-解码器的完整模型 | 不是 NLP 中的文本分词器（如 BPE），而是将连续 OHLCV 向量量化为离散令牌 |
| sibling_embed | `DependencyAwareLayer.forward()` 的参数名，指代 s1 令牌经过 `emb_s1` 嵌入后的向量，作为交叉注意力的 query 输入 | 与"兄弟令牌嵌入"同义——s1 和 s2 是同一根 K 线的两个层级令牌，s1 是 s2 的"兄弟"；该名称仅在源码接口层面使用，概念文档中通常直接称为"s1 嵌入" |
| Token（令牌） | BSQ 量化后的离散索引 | 不是文本单词，而是 K 线的离散化表示 |
| Decoder（解码器） | KronosTokenizer 的重建模块 | 不是语言模型的自回归解码器——它只做令牌到 OHLCV 的映射 |
| Context（上下文） | 输入的历史令牌序列 | 与 LLM 的上下文概念相同，但 Kronos 的上下文是 K 线令牌而非文本 |
| Vocabulary（词汇表） | s1 或 s2 的所有可能令牌集合 | 大小为 2^10 = 1024（默认预训练配置 s1_bits=10, s2_bits=10），远小于 LLM 的数万词汇量；这些值可在自定义模型中调整 |

---

## 动手练习

### 练习 1：术语速查与交叉引用

阅读以下描述，在术语表中找到对应的术语并写出中英文名称。完成后对照本文内容验证：

| # | 描述 | 你的答案 |
|---|------|---------|
| 1 | 将连续向量映射为 ±1 二值向量的量化方法 | |
| 2 | 确保 s2 依赖 s1 的交叉注意力模块 | |
| 3 | 训练时用真实标签而非模型预测作为输入的技术 | |
| 4 | 鼓励码本被均匀使用、避免死码的正则化项 | |
| 5 | 量化器中大部分码字从未被使用的现象 | |

**验证方法**：对照"算法术语"和"Transformer 组件"分组中的定义，确认你的答案与表中定义完全匹配。每个术语的"参见"字段指向的关联术语也应能一并找到。

### 练习 2：辨析容易混淆的术语

以下两组术语在 Kronos 语境中含义不同，请结合"Kronos 特有术语"表解释差异：

1. **Tokenizer（KronosTokenizer）** vs **NLP 分词器（如 BPE）**：两者处理的输入分别是什么？输出分别是什么？
2. **Decoder（KronosTokenizer 的解码器）** vs **语言模型的自回归解码器**：为什么说 Kronos 的 Decoder 不是自回归的？

**验证方法**：你的解释应覆盖"Kronos 特有术语"表中"容易混淆的点"列的内容。如果解释中遗漏了该列提到的关键区分，说明理解不够完整。

---

## 自测清单

- [ ] 能解释 BSQ 如何将连续向量量化为离散令牌
- [ ] 能区分 s1 令牌和 s2 令牌的角色（粗粒度 vs 细粒度）
- [ ] 理解承诺损失和熵正则化在分词器训练中的作用
- [ ] 知道 Kronos 的"分词器"与 NLP 中的分词器本质不同
- [ ] 能解释为什么 Decoder 在 Kronos 中不是自回归的
- [ ] 能说出至少 5 个术语在术语表中的"参见"关联术语

---
