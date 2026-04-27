# 模型对比与选型指南

> **目标读者**：想了解不同规模模型的差异并选择最合适模型的用户
> **核心问题**：mini / small / base / large 四个模型有什么区别？我该选哪个？

---

## 学习目标

这篇对比四个 Kronos 模型的规格差异，并给出选型建议：

- [ ] 说明四个模型在参数量、上下文长度、分词器搭配和开源状态上的差异
- [ ] 根据使用场景、数据特征和硬件条件选择最合适的模型
- [ ] 正确执行模型切换，避免分词器不匹配等常见错误
- [ ] 理解模型规模与预测效果之间的非线性关系

---

## 模型概览

Kronos 提供四个规模的预测模型。**Kronos-mini 使用专用的 `Kronos-Tokenizer-2k` 分词器**（上下文 2048），其余三个模型共用 `Kronos-Tokenizer-base`（上下文 512）。模型均通过 `from_pretrained()` 从 HuggingFace Hub 加载。

| 模型 | Hub 路径 | 分词器 | 参数量 | 上下文长度 | 开源状态 |
|------|----------|--------|--------|-----------|---------|
| Kronos-mini | `NeoQuasar/Kronos-mini` | `Kronos-Tokenizer-2k` | 4.1M | **2048** | 已开源 |
| Kronos-small | `NeoQuasar/Kronos-small` | `Kronos-Tokenizer-base` | 24.7M | 512 | 已开源 |
| Kronos-base | `NeoQuasar/Kronos-base` | `Kronos-Tokenizer-base` | 102.3M | 512 | 已开源 |
| Kronos-large | `NeoQuasar/Kronos-large` | `Kronos-Tokenizer-base` | 499.2M | 512 | 未开源 |

> 参数量从 4.1M 到 499.2M 跨越约 120 倍。更大的模型通常有更强的表示能力，但这不自动等价于"任何数据上都更准"。文件大小、显存占用、推理速度受硬件、框架版本和输入长度影响，不在此表中写死为"标准答案"。

---

## 架构参数对比

所有模型使用相同的架构设计（Transformer + 层级令牌），但层数、维度和头数不同：

| 参数 | mini | small | base | large |
|------|------|-------|------|-------|
| 总参数量 | **4.1M** | **24.7M** | **102.3M** | **499.2M** |
| `s1_bits` / `s2_bits` | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 |
| **上下文长度** | **2048** | 512 | 512 | 512 |
| **专用分词器** | **Kronos-Tokenizer-2k** | Kronos-Tokenizer-base | Kronos-Tokenizer-base | Kronos-Tokenizer-base |
| `d_model` | 256 | 512 | 832 | — |
| `n_layers` | 4 | 8 | 12 | — |
| `n_heads` | 4 | 8 | 16 | — |
| `ff_dim` | 512 | 1024 | 2048 | — |
| `learn_te` | true | true | true | — |

> **数据来源**：mini、small、base 的参数已从各模型在 HuggingFace Hub 上公开的 `config.json` 验证确认。large 为私有仓库，参数不可公开获取。加载模型后可通过 `model.config` 查看完整参数：
>
> ```python
> from model import Kronos
> model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
> print(model.config)  # 查看 d_model, n_layers, n_heads, ff_dim, learn_te 等全部参数
> ```

**关键点**：

- 所有模型的 `s1_bits` 和 `s2_bits` 相同（均为 10/10），但 **Kronos-mini 使用专用的 `Kronos-Tokenizer-2k` 分词器**
- 你可以在 small/base/large 之间自由切换而无需更换分词器，但切换到/从 mini 切换时需要**同时更换分词器**
- `d_model`（模型维度）是影响模型表达力的最关键参数——更大的 `d_model` 意味着更强的表示能力，但也需要更多训练数据
- `n_layers` 影响模型捕捉长距离依赖的能力。层数越多，模型能"记住"越远的历史模式，但推理时间也近似线性增长
- 推理时的显存占用主要由 `d_model`、`n_layers` 和 `seq_len` 决定
- 从 small 到 base 的参数量增长约 4 倍（24.7M → 102.3M），主要来自 `d_model` 和 `n_layers` 的增大

### 参数量增长来源分析

模型的参数量主要由以下部分贡献：

| 组件 | 参数量公式 | 与 `d_model` 的关系 |
|------|-----------|---------------------|
| 令牌嵌入 | `2 × vocab × d_model` | 线性 |
| Transformer 层 | `≈ 4 × d_model² × n_layers`（自注意力 + 前馈） | **二次方** |
| 双头输出 | `(vocab_s1 + vocab_s2) × d_model` | 线性 |

因此，`d_model` 的增大对参数量的影响远大于 `n_layers`。从 small（d_model=512）到 base（d_model=832），`d_model` 增长约 63%，但 Transformer 层的参数量增长约 165%（832²/512² ≈ 2.65 倍）。

---

## Kronos-mini 的独特优势

Kronos-mini 是唯一使用 2048 上下文长度的模型。这意味着什么？

### 上下文长度对预测的影响

上下文长度决定了模型在预测时能"看到"多少历史数据：

| 上下文长度 | 5 分钟线 | 日线 | 周线 |
|-----------|---------|------|------|
| 512 | ~42 小时 | ~2 年 | ~10 年 |
| 2048 | ~7 天 | ~8 年 | ~39 年 |

如果你的任务需要模型参考更长的历史（例如捕捉长周期趋势、识别跨年度的价格模式），mini 的 2048 上下文是唯一的选择。

### mini 的权衡

mini 参数量只有 4.1M（small 的 1/6），这意味着：

- **优势**：推理速度最快，内存占用最低，适合资源受限的环境
- **代价**：表示能力有限，在复杂市场模式或高波动场景下可能不如更大的模型

### 什么时候 mini 是最佳选择

- 历史数据超过 512 个时间点，且你确信更长的上下文能提供额外信息
- 部署环境资源有限（边缘设备、移动端）
- 需要快速迭代实验，对精度要求不极致

---

## 选型决策树

```
你的主要约束是什么？
│
├── 历史数据超过 512 个时间点？
│   └── 选 Kronos-mini（上下文 2048，但参数量最小）
│
├── 想快速验证 Kronos 功能？
│   └── 选 Kronos-small（24.7M 参数，速度与效果的平衡点）
│
├── 需要最佳已开源效果？
│   └── 选 Kronos-base（102.3M 参数，已开源的最大模型）
│
├── 想微调并部署到生产？
│   ├── GPU 资源充足 → Kronos-base
│   └── GPU 资源有限 → Kronos-small
│
├── 资源极度受限（CPU only / 嵌入式）？
│   └── 选 Kronos-mini
│
└── 想使用 Kronos-large？
    └── 先确认获取方式；仓库说明中该模型未开源
```

---

## 按场景选择

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 快速验证 Kronos 功能 | mini / small | 上手成本低，文档和示例覆盖充分 |
| 历史窗口超过 512 | mini | 唯一支持 2048 上下文的模型 |
| 日常 K 线预测 | small / base | 已开源，且与主文档中的常见示例更贴近 |
| 需要后续微调 | small / base | 开源、通用分词器一致，实验链路更顺手 |
| 资源受限（CPU / 低配） | mini / small | 参数量更小，推理更快 |
| 追求最佳效果 | base / large | 先用 base 建立基线；large 目前未开源 |

---

## 按硬件条件选择

| 硬件条件 | 推荐做法 | 说明 |
|---------|---------|------|
| CPU only | mini / small | base 也可运行，但推理较慢 |
| GPU 4GB 显存 | small | base 可能 OOM，取决于 lookback 和 sample_count |
| GPU 8GB 显存 | small / base | base 通常可以运行，但需要注意批量大小 |
| GPU 16GB+ 显存 | base | 充足的资源，可尝试更大的 batch 和 sample_count |
| Apple Silicon Mac | small | MPS 后端加速，small 的推理体验最佳 |
| 不确定本机承载能力 | 先从 mini 或 small 开始 | 逐步增加规模，直到接近资源上限 |

### 显存估算参考

推理时的显存占用受多个因素影响，以下提供一个粗略估算（仅模型推理，不含系统开销）：

| 模型 | 模型权重占用 | lookback=400, pred_len=120, sample_count=1 | lookback=400, pred_len=120, sample_count=5 |
|------|-----------|------------------------------------------|------------------------------------------|
| mini (4.1M) | ~20 MB | ~100 MB | ~300 MB |
| small (24.7M) | ~100 MB | ~500 MB | ~1.5 GB |
| base (102.3M) | ~400 MB | ~1.5 GB | ~4 GB |

> 以上为 FP32 精度估算。实际占用因 PyTorch 内存管理、CUDA 内核等因素而异。建议首次运行时用 `torch.cuda.memory_allocated()` 监控实际占用。

---

## 切换模型

切换模型只需更换 `from_pretrained()` 的参数。**关键：Kronos-mini 需要搭配专用分词器**。

### 从 small 切换到 base（最常见）

分词器不变，只需更换模型：

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# small 和 base 共用同一个分词器
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

# 切换模型：只需改这一行
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")   # 原来是 Kronos-small
predictor = KronosPredictor(model, tokenizer, max_context=512)
```

### 切换到 mini（需要更换分词器）

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# mini 需要专用分词器
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
predictor = KronosPredictor(model, tokenizer, max_context=2048)  # 注意：2048
```

### 常见切换错误

| 错误 | 原因 | 解决方法 |
|------|------|---------|
| `RuntimeError: Error(s) in loading state_dict` | 分词器与模型不匹配 | 确保 mini 用 `Tokenizer-2k`，其他用 `Tokenizer-base` |
| 预测结果异常（如全零或全相同） | `max_context` 设置错误 | mini 应设为 2048，其他应设为 512 |
| `ValueError` 维度不匹配 | 模型权重文件损坏或版本不一致 | 重新下载模型，确保 `from_pretrained()` 成功 |

---

## 模型规模与预测效果的关系

### 不存在"越大越好"的线性关系

模型规模与预测效果的关系受以下因素调节：

1. **数据复杂度**：简单的低频数据（如周线），small 可能已经足够；高频分钟线数据中 base 可能表现更好
2. **市场规律性**：趋势明显的市场，小模型也能捕捉；震荡市中，更大的模型可能从噪声中提取更多信息
3. **上下文长度需求**：如果需要超过 512 的历史，mini 是唯一选择，即使参数量更小

### 如何评估哪个模型更适合你

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

# 对比 small 和 base
results = {}
for model_name in ["NeoQuasar/Kronos-small", "NeoQuasar/Kronos-base"]:
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    paths = []
    for seed in range(10):
        torch.manual_seed(seed)
        pred = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=1.0, sample_count=1, verbose=False
        )
        paths.append(pred['close'].values)

    paths = np.array(paths)
    results[model_name] = {
        'mean': paths.mean(axis=0),
        'std': paths.std(axis=0),
        'p5': np.percentile(paths, 5, axis=0),
        'p95': np.percentile(paths, 95, axis=0),
    }

    print(f"\n{model_name}:")
    print(f"  预测终值均值: {results[model_name]['mean'][-1]:.2f}")
    print(f"  90% 置信区间: [{results[model_name]['p5'][-1]:.2f}, {results[model_name]['p95'][-1]:.2f}]")
    print(f"  路径间标准差均值: {results[model_name]['std'].mean():.4f}")
```

**关注什么**：不要只看单次预测的数值精确度。更重要的指标是：
- 路径的**趋势一致性**（多条路径是否指向相同方向）
- **置信区间宽度**（模型对预测的确定性）
- 与真实值的**方向准确率**（涨跌方向是否正确）

---

## 动手练习

### 练习 1：切换模型并验证配置

使用项目自带的测试数据，分别加载 `Kronos-small` 和 `Kronos-base`，确认两者的 `model.config` 中 `d_model` 和 `n_layers` 的差异：

```python
from model import Kronos

for name in ["NeoQuasar/Kronos-small", "NeoQuasar/Kronos-base"]:
    model = Kronos.from_pretrained(name)
    cfg = model.config if hasattr(model, 'config') else model.model.config
    print(f"{name}: d_model={cfg.get('d_model')}, n_layers={cfg.get('n_layers')}, n_heads={cfg.get('n_heads')}")
```

**验证方法**：输出应与本文"架构参数对比"表中 small（d_model=512, n_layers=8, n_heads=8）和 base（d_model=832, n_layers=12, n_heads=16）的参数一致。

### 练习 2：对比两个模型的预测路径

使用测试数据，分别用 `Kronos-small` 和 `Kronos-base` 对同一时间段做预测（固定随机种子），对比路径间标准差：

```python
import numpy as np
import torch
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 200
pred_len = 30
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

for model_name in ["NeoQuasar/Kronos-small", "NeoQuasar/Kronos-base"]:
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    paths = []
    for seed in range(5):
        torch.manual_seed(seed)
        pred = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=1.0, sample_count=1, verbose=False
        )
        paths.append(pred['close'].values)

    paths = np.array(paths)
    print(f"\n{model_name}:")
    print(f"  预测终值均值: {paths[:, -1].mean():.2f}")
    print(f"  路径间标准差均值: {paths.std(axis=0).mean():.4f}")
```

**验证方法**：两个模型都应产生合理的预测值（与输入数据的量级相近）。base 模型的路径间标准差可能更小（更稳定），也可能更准确——但重要的是确认两个模型都能正常加载和预测，没有分词器不匹配等错误。

---

## 自测清单

- [ ] 能说出四个模型在参数量上的排序（mini: 4.1M < small: 24.7M < base: 102.3M < large: 499.2M）
- [ ] 知道 Kronos-mini 使用专用的 `Kronos-Tokenizer-2k` 分词器，其余模型共用 `Kronos-Tokenizer-base`
- [ ] 知道 Kronos-mini 的上下文长度为 2048，而 small/base/large 为 512
- [ ] 能根据上下文长度、分词器搭配和开源状态选择合适的模型
- [ ] 知道切换模型需要注意分词器的匹配关系
- [ ] 理解模型更大不一定在所有场景下都更好
- [ ] 能写出对比两个模型预测效果的代码

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [使用场景与实战案例](06-use-cases.md) | ⭐⭐ | 不同场景的配置建议 |
| [批量预测指南](03-batch-prediction.md) | ⭐⭐⭐ | 多序列并行预测 |
| [CSV 微调指南](02-finetune-csv.md) | ⭐⭐⭐ | 微调提升特定市场效果 |

## 相关文档

| 文档 | 说明 |
|------|------|
| [KronosPredictor 使用指南](../core-concepts/04-predictor.md) | 预测参数详解 |
| [使用场景与实战案例](06-use-cases.md) | 不同场景的参数配置建议 |
| [常见问题](../references/faq.md) | 模型选择相关常见问题 |

