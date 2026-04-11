# 模型对比与选型指南 ⭐⭐

> **目标读者**：想了解不同规模模型的差异并选择最合适模型的用户
> **核心问题**：mini / small / base / large 四个模型有什么区别？我该选哪个？

---

## 学习目标

阅读本文后，你将能够：

- [ ] 说明四个模型在参数量、推理速度和预测质量上的差异
- [ ] 根据使用场景和资源条件选择最合适的模型
- [ ] 理解模型规模与预测效果之间的关系

---

## 模型概览

所有模型共用同一个分词器（`NeoQuasar/Kronos-Tokenizer-base`），区别在于预测模型（`Kronos`）的规模。模型均通过 `from_pretrained()` 从 HuggingFace Hub 加载。

| 模型 | Hub 路径 | 参数量级 | 模型文件大小 |
|------|----------|---------|-------------|
| Kronos-mini | `NeoQuasar/Kronos-mini` | 最少 | ~25 MB |
| Kronos-small | `NeoQuasar/Kronos-small` | 较少 | ~50 MB |
| Kronos-base | `NeoQuasar/Kronos-base` | 中等 | ~200 MB |
| Kronos-large | `NeoQuasar/Kronos-large` | 最多 | ~500 MB |

> **注意**：以上文件大小为近似值。首次使用时模型会从 HuggingFace Hub 自动下载到本地缓存。

---

## 架构参数对比

所有模型使用相同的架构设计（Transformer + 层级令牌），但层数、维度和头数不同：

| 参数 | mini | small | base | large |
|------|------|-------|------|-------|
| `n_layers`（Transformer 层数） | 较少 | 12 | 12 | 更多 |
| `d_model`（模型维度） | 较小 | 512 | 832 | 更大 |
| `n_heads`（注意力头数） | 较少 | 8 | 16 | 更多 |
| `ff_dim`（前馈网络维度） | 较小 | 1280 | 2048 | 更大 |
| `s1_bits` / `s2_bits` | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 |

**关键点**：

- 所有模型的 `s1_bits` 和 `s2_bits` 相同（均为 10/10），因此共用同一个分词器。这意味着你可以随时切换预测模型而无需重新加载分词器
- `d_model` 是影响模型表达力的最关键参数——更大的 `d_model` 意味着更多的参数和更强的表达能力，但也需要更多的训练数据和计算资源
- `n_layers` 影响模型捕捉长距离依赖的能力。层数越多，模型能"记住"越远的历史模式，但推理时间也近似线性增长
- 推理时的显存占用主要由 `d_model`、`n_layers` 和 `seq_len` 决定，与 `s1_bits`/`s2_bits` 无关（因为令牌嵌入后已经转换为 `d_model` 维度）

> 可通过 `model.config` 查看实际参数值（加载模型后）。

---

## 性能对比

### 推理速度

推理速度受硬件、`lookback`、`pred_len`、`sample_count` 等参数影响。以下为参考值（CPU，`lookback=400, pred_len=120, sample_count=1`）：

| 模型 | 单步推理时间（近似） | 120 步总时间（近似） |
|------|---------------------|---------------------|
| Kronos-mini | ~20 ms | ~2-3 秒 |
| Kronos-small | ~50 ms | ~6-8 秒 |
| Kronos-base | ~150 ms | ~18-25 秒 |
| Kronos-large | ~400 ms | ~50-70 秒 |

> **注意**：GPU 上的推理速度可提升 2-10 倍，具体取决于 GPU 型号。

### 显存/内存占用

| 模型 | CPU 内存 | GPU 显存（单条） | GPU 显存（batch=5） |
|------|---------|-----------------|-------------------|
| Kronos-mini | ~1 GB | ~0.5 GB | ~1 GB |
| Kronos-small | ~2 GB | ~1 GB | ~2 GB |
| Kronos-base | ~4 GB | ~2 GB | ~4 GB |
| Kronos-large | ~8 GB | ~4 GB | ~8 GB |

### 预测质量

预测质量取决于多个因素，模型规模只是其中之一。一般趋势：

| 模型 | 短期预测（1-20 步） | 中期预测（20-60 步） | 长期预测（60+ 步） |
|------|---------------------|---------------------|-------------------|
| mini | 基本可用 | 方向大致正确 | 不确定性较大 |
| small | 良好 | 良好 | 中等 |
| base | 良好 | 优秀 | 良好 |
| large | 优秀 | 优秀 | 较好 |

> **注意**：预测质量受数据特征、市场状态、历史窗口长度等因素影响。模型更大不一定在所有场景下都更好——当数据特征较简单时，小模型可能已足够。

---

## 选型建议

### 按场景选择

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 快速验证 Kronos 功能 | mini / small | 下载快、推理快，足以验证流程 |
| 日常 K 线预测 | small / base | 效果与速度的良好平衡 |
| 追求最佳预测质量 | base / large | 资源充足时的最优选择 |
| 批量预测（多只股票） | small | 推理速度快，批量场景更实用 |
| 微调实验 | small | 参数少，训练快，迭代效率高 |
| 资源受限（CPU / 小显存） | mini / small | 内存占用小 |
| 研究与分析 | base / large | 更强的表达能力，适合深入分析 |

### 按资源条件选择

| 资源条件 | 推荐模型 |
|---------|---------|
| 仅 CPU，内存 ≤ 4 GB | mini |
| 仅 CPU，内存 ≥ 8 GB | small / base |
| GPU 显存 2-4 GB | small |
| GPU 显存 4-8 GB | base |
| GPU 显存 ≥ 8 GB | large |
| Apple Silicon Mac (8GB) | small / base |
| Apple Silicon Mac (16GB+) | base / large |

### 选型决策流程

```
你的 GPU 显存有多大？
├── < 4 GB 或仅 CPU
│   └── 使用 Kronos-small（推荐）或 mini
├── 4-8 GB
│   └── 使用 Kronos-base（推荐）或 small
└── ≥ 8 GB
    ├── 追求最佳效果 → Kronos-large
    └── 速度优先 → Kronos-base
```

---

## 切换模型

切换模型只需更换 `from_pretrained()` 的参数，分词器无需更换：

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# 分词器——所有模型共用
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

# 切换预测模型：只需修改这一行
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")   # 小模型
# model = Kronos.from_pretrained("NeoQuasar/Kronos-base")  # 标准模型
# model = Kronos.from_pretrained("NeoQuasar/Kronos-large") # 大模型

predictor = KronosPredictor(model, tokenizer, max_context=512)
```

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

**验证方法**：两个模型的预测结果通常在趋势方向上一致，但在具体数值和波动幅度上可能有差异。

---

## 自测清单

- [ ] 我能说出四个模型在参数量上的大致排序（mini < small < base < large）
- [ ] 我知道所有模型共用同一个分词器
- [ ] 我能根据 GPU 显存大小选择合适的模型
- [ ] 我知道切换模型只需要修改 `from_pretrained()` 的参数
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
