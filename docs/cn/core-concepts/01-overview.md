# 项目总览与核心概念 ⭐⭐

> **目标读者**：已完成入门教程，想系统理解 Kronos 设计的用户
> **核心问题**：Kronos 是什么？为什么要这样设计？核心组件之间如何协作？

---

## 一句话定义

**Kronos** 是首个面向金融蜡烛图（K 线）数据的开源基础模型，通过"分词 + 自回归预测"的两阶段框架，将 K 线预测转化为语言建模问题。

---

## 为什么需要 Kronos？

### 传统方法的局限

传统的金融预测模型通常直接处理连续的数值数据，面临两个核心困难：

1. **连续空间建模困难**：OHLCV 数据是连续的多维时间序列，直接在连续空间进行自回归预测容易产生误差累积
2. **缺乏通用性**：针对特定市场或品种训练的模型难以迁移到其他市场

### Kronos 的解决思路

Kronos 借鉴了大语言模型（LLM）的成功范式：**先将连续数据离散化为"令牌"，再在离散令牌空间进行自回归预测**。

这种设计的优势：

- **离散空间更稳定**：避免了连续值回归中的误差累积问题
- **采样策略灵活**：可以应用温度采样、核采样等成熟的文本生成技术
- **泛化能力强**：在 45+ 个全球交易所数据上预训练，适用于不同市场和品种

---

## 两阶段框架

```
┌──────────────────────────────────────────────────────────────┐
│                    Kronos 两阶段框架                          │
│                                                              │
│  阶段 1：分词（Tokenization）                                │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐             │
│  │ OHLCV    │ →  │ Encoder    │ →  │ BSQ      │ → 离散令牌  │
│  │ 连续数据 │    │ Transformer│    │ 量化器   │   (s1, s2)  │
│  └──────────┘    └────────────┘    └──────────┘             │
│                                                              │
│  阶段 2：自回归预测（Autoregressive Prediction）             │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐             │
│  │ 历史令牌 │ →  │ Kronos     │ →  │ 采样策略 │ → 预测令牌  │
│  │ (s1, s2) │    │ Transformer│    │ top-k/p  │             │
│  └──────────┘    └────────────┘    └──────────┘             │
│                                       ↓                      │
│                    ┌────────────┐    ┌──────────┐            │
│                    │ OHLCV     │ ←  │ Decoder  │ ← 预测令牌 │
│                    │ 预测数据  │    │ Transformer│            │
│                    └────────────┘    └──────────┘            │
└──────────────────────────────────────────────────────────────┘
```

### 阶段 1：分词（Tokenization）

由 **KronosTokenizer** 完成。它将每根 K 线的 6 维 OHLCV 数据（open、high、low、close、volume、amount）编码为一对离散令牌 `(s1, s2)`：

- **s1（粗粒度令牌）**：捕捉 K 线的主要价格走势方向
- **s2（细粒度令牌）**：在 s1 的基础上提供更精细的修正信息

**类比理解**：s1 就像是"这笔交易大致是涨还是跌"，s2 则是"具体涨跌幅是多少"。

### 阶段 2：自回归预测

由 **Kronos** 模型完成。它接收历史令牌序列，逐个预测未来的 s1 和 s2 令牌：

1. 先预测 s1（粗粒度）
2. 基于已预测的 s1，再预测 s2（细粒度）
3. 重复此过程，直到生成指定数量的预测步

最后，分词器的解码器将预测令牌还原为 OHLCV 数值。

---

## 三大核心 API

Kronos 提供三个面向用户的类，全部通过 `from model import ...` 导入：

### 1. KronosTokenizer（分词器）

**职责**：OHLCV 连续数据 ↔ 离散令牌之间的双向转换

```python
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

# 编码：OHLCV → 令牌
z_indices = tokenizer.encode(x, half=True)  # 返回 (s1_indices, s2_indices)

# 解码：令牌 → OHLCV
reconstructed = tokenizer.decode(z_indices, half=True)
```

### 2. Kronos（预测模型）

**职责**：给定历史令牌序列，自回归预测未来令牌

```python
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 前向推理（返回 s1 和 s2 的预测 logits）
s1_logits, s2_logits = model(s1_ids, s2_ids, stamp=timestamps)
```

### 3. KronosPredictor（预测器）

**职责**：将分词器 + 模型封装为一步到位的预测接口

```python
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 一键预测
pred_df = predictor.predict(df=x_df, x_timestamp=..., y_timestamp=..., pred_len=120)
```

**推荐用法**：日常使用中直接使用 `KronosPredictor`，只有在微调或研究时才需要单独操作 `KronosTokenizer` 和 `Kronos`。

---

## 模型动物园

Kronos 提供不同规模的预训练模型，均托管在 HuggingFace Hub 的 `NeoQuasar` 组织下：

| 模型 | Hub 路径 | 特点 |
|------|----------|------|
| Kronos-Tokenizer-base | `NeoQuasar/Kronos-Tokenizer-base` | 分词器（所有预测模型共用同一个分词器） |
| Kronos-mini | `NeoQuasar/Kronos-mini` | 最小模型，推理最快 |
| Kronos-small | `NeoQuasar/Kronos-small` | 速度与效果的平衡点，推荐入门使用 |
| Kronos-base | `NeoQuasar/Kronos-base` | 标准模型 |
| Kronos-large | `NeoQuasar/Kronos-large` | 最大模型，效果最好但推理较慢 |

所有预测模型共用同一个分词器，因此更换预测模型不需要更换分词器。

---

## 支持的市场与时间粒度

Kronos 在来自 45+ 个全球交易所的数据上进行了预训练，理论上支持：

- **任何金融市场**：股票、期货、加密货币等
- **多种时间粒度**：分钟线、小时线、日线等
- **多空双向**：不预设方向偏见

Kronos 的输入是一个固定维度的 OHLCV 向量（6 维），与具体的市场和时间粒度无关。模型的泛化能力来自预训练阶段见过的多样化数据。

---

## 知识关联

- **前置**：[快速开始](../getting-started/02-quickstart.md) ⭐ — 已完成第一次预测
- **相关**：[KronosTokenizer 详解](02-tokenizer.md) ⭐⭐ — 深入理解分词器
- **相关**：[Kronos 模型详解](03-model.md) ⭐⭐ — 深入理解预测模型
- **进阶**：[层级令牌体系](05-hierarchical-tokens.md) ⭐⭐ — 理解 s1/s2 的设计

---
**文档元信息**
难度：⭐⭐ | 类型：核心概念 | 预计阅读时间：15 分钟
