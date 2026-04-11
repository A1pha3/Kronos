# 术语表

本文档汇总 Kronos 项目中出现的中英文术语及其定义。

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
| 层级令牌 | Hierarchical Token | Kronos 将每根 K 线编码为两个层级的令牌：s1（粗粒度）和 s2（细粒度） |
| s1 令牌 | s1 Token / Pre Token | 粗粒度令牌，使用码本的前 s1_bits 位，捕捉主要走势 |
| s2 令牌 | s2 Token / Post Token | 细粒度令牌，使用码本的后 s2_bits 位，提供精细修正 |
| 双头 | DualHead | Kronos 模型的输出层，分别预测 s1 和 s2 的 logits |
| 依赖感知层 | DependencyAwareLayer | 使用交叉注意力实现 s2 对 s1 条件依赖的模块 |
| 层级嵌入 | HierarchicalEmbedding | 将 s1 和 s2 令牌嵌入并融合为统一向量的模块 |
| 时间嵌入 | TemporalEmbedding | 将时间特征（分钟、小时等）编码为向量的模块 |
| 令牌嵌入 | Token Embedding | 将离散令牌 ID 映射为连续向量的查找表 |

---

## 算法术语

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| 二值球面量化 | Binary Spherical Quantization (BSQ) | 将连续向量量化为二值向量（±1）的方法，码本天然均匀分布 |
| 直通估计器 | Straight-Through Estimator (STE) | 处理不可导操作（如量化）的技巧：前向用离散值，反向传递连续值的梯度 |
| 承诺损失 | Commit Loss | 量化损失的一部分，衡量量化前后向量的距离，推动编码器输出靠近量化点 |
| 熵正则化 | Entropy Regularization | 鼓励码本被均匀使用的正则化项，避免"死码"问题 |
| 样本熵 | Per-Sample Entropy | 单个样本在各子组上的熵，鼓励每个样本的量化结果均匀分布 |
| 码本熵 | Codebook Entropy | 所有样本的平均概率分布的熵，从宏观层面鼓励码本均匀使用 |
| 分组近似 | Group Approximation | 将高维码本的熵分解为低维子码本熵之和的近似方法 |
| 温度采样 | Temperature Sampling | 通过除以温度参数 T 控制采样分布尖锐程度的策略 |
| 核采样 | Nucleus Sampling / Top-p | 保留累计概率达到 p 的候选令牌的采样策略 |
| Top-k 采样 | Top-k Sampling | 只保留概率最高的 k 个候选令牌的采样策略 |
| 贪婪解码 | Greedy Decoding | 始终选择概率最高的令牌，等价于 T→0 或 top_k=1 |
| Teacher Forcing | Teacher Forcing | 训练时使用真实目标而非模型预测作为下一步输入的技术 |
| 实例级标准化 | Instance-Level Normalization | 对每条序列独立计算均值和标准差的标准化方法 |
| z-score 标准化 | Z-Score Normalization | (x - mean) / std，将数据变换为零均值单位方差 |

---

## Transformer 组件

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| 自注意力 | Self-Attention | 序列内部各位置之间计算注意力的机制 |
| 交叉注意力 | Cross-Attention | 两个不同序列之间计算注意力的机制（query 来自一个序列，key/value 来自另一个） |
| 多头注意力 | Multi-Head Attention | 使用多组独立的注意力头并行计算，捕捉不同子空间的信息 |
| 旋转位置编码 | Rotary Positional Embedding (RoPE) | 通过旋转矩阵将相对位置信息编码到 q 和 k 中的位置编码方法 |
| 前馈网络 | Feed-Forward Network (FFN) | Transformer 块中的逐位置非线性变换层 |
| SwiGLU | Swish-Gated Linear Unit | 门控激活函数：SiLU(W1(x)) × W3(x)，比 ReLU 表达力更强 |
| RMSNorm | Root Mean Square Normalization | 不减去均值的归一化方法，计算效率高于 LayerNorm |
| Pre-Norm | Pre-Normalization | 在注意力/前馈操作之前进行归一化的 Transformer 变体 |
| 残差连接 | Residual Connection | 将层的输出与输入相加（x + f(x)），缓解梯度消失 |
| 因果掩码 | Causal Mask | 确保每个位置只能关注之前位置的注意力掩码 |
| 填充掩码 | Padding Mask | 在批量处理中标记和忽略填充位置的掩码 |

---

## 训练相关

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| 微调 | Fine-tuning | 在预训练模型基础上，使用特定数据继续训练以适应新任务 |
| 分布式数据并行 | Distributed Data Parallel (DDP) | 多 GPU 训练策略，每个 GPU 持有模型副本，处理不同数据子集 |
| 梯度累积 | Gradient Accumulation | 将多个小批量的梯度累积后再更新参数，模拟更大的批量大小 |
| 梯度裁剪 | Gradient Clipping | 将梯度范数限制在阈值内，防止梯度爆炸 |
| 学习率调度 | Learning Rate Schedule | 训练过程中动态调整学习率的策略 |
| OneCycleLR | OneCycleLR | 先升后降的学习率策略，先快速上升到峰值再缓慢下降 |
| 交叉熵损失 | Cross-Entropy Loss | 分类任务的标准损失函数，衡量预测分布与真实分布的差异 |
| MSE 损失 | Mean Squared Error Loss | 回归任务的损失函数，衡量预测值与真实值的均方误差 |
| Qlib | Qlib | 微软开源的量化投资平台，提供中国 A 股数据接口 |
| HuggingFace Hub | HuggingFace Hub | 模型和数据集的托管平台，Kronos 的预训练模型存储于此 |
| Safetensors | Safetensors | 一种安全的模型权重文件格式，避免了 pickle 的安全风险 |

---

## 数据相关

| 中文术语 | 英文原文 | 定义 |
|---------|----------|------|
| 滑动窗口 | Sliding Window | 在时间序列上按固定大小移动的窗口，用于采样训练数据 |
| 回看窗口 | Lookback Window | 模型输入使用的历史数据长度 |
| 预测窗口 | Prediction Window | 模型预测的未来数据长度 |
| 最大上下文 | Max Context | 模型能处理的最大令牌序列长度（Kronos 默认 512） |
| 涨跌停 | Price Limit | A 股市场中单日价格涨跌幅不超过 ±10% 的限制 |

---

**文档元信息**
类型：参考文档 | 词条数：55+
