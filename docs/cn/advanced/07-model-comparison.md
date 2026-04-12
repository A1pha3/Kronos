# 模型对比与选型指南 ⭐⭐

> **目标读者**：想了解不同规模模型的差异并选择最合适模型的用户
> **核心问题**：mini / small / base / large 四个模型有什么区别？我该选哪个？

---

## 学习目标

阅读本文后，你将能够：

- [ ] 说明四个模型在参数量、上下文长度、分词器搭配和开源状态上的差异
- [ ] 根据使用场景和资源条件选择最合适的模型
- [ ] 理解模型规模与预测效果之间的关系

---

## 模型概览

Kronos 提供四个规模的预测模型，大部分共用同一个分词器（`NeoQuasar/Kronos-Tokenizer-base`），**但 Kronos-mini 例外**——它使用专用的 `Kronos-Tokenizer-2k` 分词器，上下文长度为 2048 而非 512。模型均通过 `from_pretrained()` 从 HuggingFace Hub 加载。

| 模型 | Hub 路径 | 分词器 | 参数量 | 上下文长度 | 开源状态 |
|------|----------|--------|--------|-----------|---------|
| Kronos-mini | `NeoQuasar/Kronos-mini` | `Kronos-Tokenizer-2k` | 4.1M | 2048 | 已开源 |
| Kronos-small | `NeoQuasar/Kronos-small` | `Kronos-Tokenizer-base` | 24.7M | 512 | 已开源 |
| Kronos-base | `NeoQuasar/Kronos-base` | `Kronos-Tokenizer-base` | 102.3M | 512 | 已开源 |
| Kronos-large | `NeoQuasar/Kronos-large` | `Kronos-Tokenizer-base` | 499.2M | 512 | 未开源 |

> 上表仅保留仓库 `README.md` 直接给出的信息。文件大小、显存占用、单机推理耗时会受到硬件、框架版本和输入长度影响，因此不在这里写死为“标准答案”。

---

## 架构参数对比

所有模型使用相同的架构设计（Transformer + 层级令牌），但层数、维度和头数不同：

| 参数 | mini | small | base | large |
|------|------|-------|------|-------|
| 总参数量 | **4.1M** | **24.7M** | **102.3M** | **499.2M** |
| `s1_bits` / `s2_bits` | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 |
| **上下文长度** | **2048** | 512 | 512 | 512 |
| **专用分词器** | **Kronos-Tokenizer-2k** | Kronos-Tokenizer-base | Kronos-Tokenizer-base | Kronos-Tokenizer-base |

> **关于各模型的具体 `d_model`、`n_layers` 等内部参数**：这些值存储在 HuggingFace Hub 上各模型的 `config.json` 中。你可以通过加载模型后执行 `model.config` 查看。参数量从 4.1M 到 499.2M 跨越了约 120 倍——`d_model`（模型维度）和 `n_layers`（Transformer 层数）是参数量增长的主要来源。

**关键点**：

- 所有模型的 `s1_bits` 和 `s2_bits` 相同（均为 10/10），但 **Kronos-mini 使用专用的 `Kronos-Tokenizer-2k` 分词器**（支持 2048 上下文长度），而 small/base/large 共用 `Kronos-Tokenizer-base`（512 上下文长度）
- 这意味着你可以在 small/base/large 之间自由切换而无需更换分词器，但切换到/从 mini 切换时需要同时更换分词器
- `d_model` 是影响模型表达力的最关键参数——更大的 `d_model` 意味着更多的参数和更强的表达能力，但也需要更多的训练数据和计算资源
- `n_layers` 影响模型捕捉长距离依赖的能力。层数越多，模型能"记住"越远的历史模式，但推理时间也近似线性增长
- 推理时的显存占用主要由 `d_model`、`n_layers` 和 `seq_len` 决定，与 `s1_bits`/`s2_bits` 无关（因为令牌嵌入后已经转换为 `d_model` 维度）

> 可通过 `model.config` 查看实际参数值（加载模型后）。

---

## 如何理解“模型更大”？

源码和根目录 `README.md` 可以确认的是：模型参数量从 `mini` 到 `large` 逐级增加，而 `mini` 还拥有更长的上下文长度 `2048`。这通常意味着两个直接后果：

1. **资源需求更高**：更大的模型一般会占用更多内存或显存，加载和推理也更重
2. **表达空间更大**：参数更多，通常意味着模型有更强的表示能力，但这并不自动等价于“任何数据上都更准”

文档不直接给出固定的速度、显存、精度排名，因为这些结论必须结合你的机器、数据长度和评价指标实测。

---

## 选型建议

### 按场景选择

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 快速验证 Kronos 功能 | mini / small | 上手成本较低，文档和示例覆盖充分 |
| 历史窗口超过 512 | mini | `README.md` 明确给出 mini 的上下文长度为 2048 |
| 日常 K 线预测 | small / base | 已开源，且与主文档中的常见示例更贴近 |
| 需要后续微调或二次开发 | small / base | 开源、通用分词器一致，实验链路更顺手 |
| 资源受限 | mini / small | 参数量更小，通常更容易部署 |
| 想研究更大模型配置 | base / large | 先用 base 建立基线；large 目前在仓库说明中标为未开源 |

### 按资源条件选择

| 资源条件 | 推荐做法 |
|---------|---------|
| 不确定本机承载能力 | 先从 `mini` 或 `small` 开始 |
| 需要长上下文 | 优先评估 `mini` |
| 需要与 small/base 共用同一分词器 | 使用 `Kronos-Tokenizer-base` 搭配 `small` 或 `base` |
| 想进一步放大模型规模 | 先在 `base` 上完成验证，再考虑 `large` 的可用性与获取方式 |

### 选型决策流程

```
你的主要约束是什么？
├── 需要超过 512 的上下文
│   └── 选 `Kronos-mini`
├── 需要最稳妥的开源默认路线
│   └── 先用 `Kronos-small`
├── small 已经满足不了你的实验需求
│   └── 升到 `Kronos-base`
└── 想使用 `Kronos-large`
    └── 先确认获取方式；仓库说明中该模型未开源
```

---

## 切换模型

切换模型只需更换 `from_pretrained()` 的参数。**注意：Kronos-mini 需要搭配专用分词器**：

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# === 方式 1：使用 small/base（共用分词器） ===
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")   # 或 Kronos-base
predictor = KronosPredictor(model, tokenizer, max_context=512)

# === 方式 2：使用 mini（专用分词器） ===
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
predictor = KronosPredictor(model, tokenizer, max_context=2048)  # 注意：mini 支持 2048 上下文
```

> **重要**：混用分词器和预测模型会导致错误。例如，不能用 `Kronos-Tokenizer-base` 搭配 `Kronos-mini`，也不能用 `Kronos-Tokenizer-2k` 搭配 `Kronos-small`。

---

## 动手练习

### 练习 1：对比不同模型的预测差异

使用相同的输入数据，分别用 `Kronos-small` 和 `Kronos-base` 进行预测，对比预测结果的差异：

```python
import pandas as pd
import numpy as np
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 60
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

for model_name in ["NeoQuasar/Kronos-small", "NeoQuasar/Kronos-base"]:
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=pred_len, T=1.0, sample_count=3, verbose=False
    )

    print(f"\n{model_name}:")
    print(f"  收盘价范围: [{pred_df['close'].min():.2f}, {pred_df['close'].max():.2f}]")
    print(f"  收盘价标准差: {pred_df['close'].std():.4f}")
```

**验证方法**：重点比较你真正关心的指标，例如回测收益、方向判断、误差分布或稳定性，而不是只看单次运行的某一张图。

---

## 自测清单

- [ ] 我能说出四个模型在参数量上的大致排序（mini: 4.1M < small: 24.7M < base: 102.3M < large: 499.2M）
- [ ] 我知道 Kronos-mini 使用专用的 `Kronos-Tokenizer-2k` 分词器，其余模型共用 `Kronos-Tokenizer-base`
- [ ] 我知道 Kronos-mini 的上下文长度为 2048，而 small/base/large 为 512
- [ ] 我能根据上下文长度、分词器搭配和开源状态选择合适的模型
- [ ] 我知道切换模型需要注意分词器的匹配关系
- [ ] 我理解模型更大不一定在所有场景下都更好

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [使用场景与实战案例](06-use-cases.md) | ⭐⭐ | 不同场景的配置建议 |
| [批量预测指南](03-batch-prediction.md) | ⭐⭐⭐ | 多序列并行预测 |
| [CSV 微调指南](02-finetune-csv.md) | ⭐⭐⭐ | 微调提升特定市场效果 |

---
**文档元信息**
难度：⭐⭐ | 类型：选型指南 | 预计阅读时间：10 分钟
