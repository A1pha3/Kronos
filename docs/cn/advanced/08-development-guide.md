# 开发扩展指南 ⭐⭐⭐

> **目标读者**：希望深入 Kronos 代码进行二次开发或贡献代码的开发者
> **前置要求**：已完成 [系统架构分析](../architecture/01-system-architecture.md) 和 [源码走读](../architecture/04-source-code-walkthrough.md)

---

## 学习目标

阅读本文后，你将能够：

- [ ] 在 Kronos 的架构中定位需要修改的模块并预估影响范围
- [ ] 添加自定义的时间特征、数据集或采样策略
- [ ] 理解模型的扩展点和设计约束
- [ ] 运行回归测试验证修改的正确性

---

## 代码结构速览

```
Kronos/
├── model/
│   ├── __init__.py          # 模型注册（KronosTokenizer, Kronos, KronosPredictor）
│   ├── kronos.py            # 三大核心类 + auto_regressive_inference
│   └── module.py            # 基础组件（BSQ, Transformer, Embeddings 等）
├── finetune/                # Qlib 微调流水线
├── finetune_csv/            # CSV 微调流水线
├── webui/                   # Flask Web 界面
├── examples/                # 预测示例脚本
└── tests/
    └── test_kronos_regression.py  # 回归测试
```

修改时遵循的原则：

1. **优先修改 `module.py`**：基础组件的修改集中在此文件
2. **流水线逻辑在 `kronos.py`**：推理流程、数据处理逻辑在此文件
3. **保持向后兼容**：修改核心 API 时确保已有代码仍可运行
4. **运行回归测试**：每次修改后执行 `pytest tests/test_kronos_regression.py`

### 代码修改的安全边界

在修改 Kronos 代码之前，了解以下安全边界可以避免引入难以排查的问题：

| 修改类型 | 安全 | 需要谨慎 | 需要重新训练 |
|---------|------|---------|------------|
| 添加新的 API 端点 | ✅ | — | — |
| 修改采样策略（不影响模型权重） | ✅ | — | — |
| 添加新的数据预处理步骤 | — | ⚠️ 需验证对推理的影响 | — |
| 修改 `s1_bits` / `s2_bits` | — | — | ✅ 必须 |
| 修改 `d_model` / `n_layers` 等结构参数 | — | — | ✅ 必须 |
| 添加新的时间特征 | — | — | ✅ 必须（嵌入表不兼容） |
| 替换 RMSNorm 为 LayerNorm | — | ⚠️ 需验证训练收敛 | 取决于是否加载预训练权重 |
| 替换 RoPE 为其他位置编码 | — | ⚠️ 需验证长序列效果 | 取决于是否加载预训练权重 |

---

## 扩展点 1：添加自定义时间特征

### 当前实现

`TemporalEmbedding`（定义于 `module.TemporalEmbedding`）从时间戳中提取 5 个特征：`minute`、`hour`、`weekday`、`day`、`month`。特征提取逻辑位于 `calc_time_stamps()` 函数（定义于 `kronos.calc_time_stamps`）。

### 扩展步骤：添加"季度"特征

**步骤 1**：修改 `calc_time_stamps()` 添加新特征

```python
# model/kronos.py — calc_time_stamps()
def calc_time_stamps(x_timestamp):
    time_df = pd.DataFrame()
    time_df['minute'] = x_timestamp.dt.minute
    time_df['hour'] = x_timestamp.dt.hour
    time_df['weekday'] = x_timestamp.dt.weekday
    time_df['day'] = x_timestamp.dt.day
    time_df['month'] = x_timestamp.dt.month
    time_df['quarter'] = x_timestamp.dt.quarter    # 新增
    return time_df
```

**步骤 2**：修改 `TemporalEmbedding` 添加对应的嵌入表

```python
# model/module.py — TemporalEmbedding.__init__()
quarter_size = 5    # 季度：1-4
self.quarter_embed = Embed(quarter_size, d_model)

# model/module.py — TemporalEmbedding.forward()
quarter_x = self.quarter_embed(x[:, :, 5])    # 索引 5 对应新特征
return hour_x + weekday_x + day_x + month_x + minute_x + quarter_x
```

**步骤 3**：更新 `KronosPredictor` 中的时间特征列定义

```python
# model/kronos.py — KronosPredictor.__init__()
self.time_cols = ['minute', 'hour', 'weekday', 'day', 'month', 'quarter']
```

**影响范围**：

| 需要同步修改 | 文件 | 说明 |
|-------------|------|------|
| `calc_time_stamps` | `kronos.py` | 添加新列 |
| `TemporalEmbedding.__init__` | `module.py` | 添加嵌入表 |
| `TemporalEmbedding.forward` | `module.py` | 添加特征索引 |
| `KronosPredictor.time_cols` | `kronos.py` | 更新列名列表 |
| 微调流水线的数据集 | `finetune/dataset.py` 等 | 如果数据集有自定义时间特征处理 |
| 回归测试 | `tests/` | 可能需要更新测试 |

> **重要**：添加新的时间特征后，预训练模型的权重不再兼容。需要**重新训练**模型。

---

## 扩展点 2：自定义数据集

### CSV 微调流水线的数据集

`CustomKlineDataset`（`finetune_csv/finetune_base_model.py`）定义了 CSV 数据的加载和采样逻辑。

### 扩展步骤：实现自定义数据集

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class MyCustomDataset(Dataset):
    """自定义数据集示例：从数据库或 API 加载数据"""

    def __init__(self, data_source, lookback_window, predict_window,
                 clip=5.0, data_type="train", seed=100):
        self.lookback = lookback_window
        self.pred_window = predict_window
        self.clip = clip
        self.data_type = data_type
        self.seed = seed

        # 加载数据（替换为你自己的数据源）
        self.data = self._load_data(data_source)
        self.total_window = lookback_window + predict_window

    def _load_data(self, source):
        """从自定义数据源加载 OHLCV 数据"""
        # 示例：从 Parquet 文件加载
        import pandas as pd
        df = pd.read_parquet(source)
        df = df.sort_values('timestamps').reset_index(drop=True)
        return df

    def __len__(self):
        return max(0, len(self.data) - self.total_window)

    def __getitem__(self, idx):
        window = self.data.iloc[idx:idx + self.total_window]
        x = window[['open', 'high', 'low', 'close', 'volume', 'amount']].values.astype(np.float32)

        # 实例级标准化
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        return torch.from_numpy(x)
```

**关键要求**：

1. `__getitem__` 返回标准化后的张量，形状 `(lookback + predict_window, 6)`
2. 数据列顺序必须与预训练一致：`open, high, low, close, volume, amount`
3. 实例级标准化（每条样本独立）是跨市场泛化的关键

---

## 扩展点 3：修改采样策略

### 当前实现

`sample_from_logits()` 函数（定义于 `kronos.sample_from_logits`）支持温度采样、top-k 和 top-p 过滤。

### 扩展：实现 min-p 采样

```python
def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None,
                       min_p=None, sample_logits=True):
    """扩展采样函数：添加 min_p 支持"""
    logits = logits / temperature

    if min_p is not None and min_p > 0:
        # min-p：过滤掉概率低于最大概率 * min_p 的令牌
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        indices_to_remove = probs < (max_prob * min_p)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = probs.topk(k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x
```

**影响范围**：

| 需要同步修改 | 文件 |
|-------------|------|
| `sample_from_logits` | `kronos.py` |
| `auto_regressive_inference` 的参数签名 | `kronos.py` |
| `KronosPredictor.predict` 的参数签名 | `kronos.py` |
| `KronosPredictor.predict_batch` 的参数签名 | `kronos.py` |

---

## 扩展点 4：修改模型架构

### 常见修改场景

#### 增加 Transformer 层数

```python
# 修改构造函数中的 n_layers 参数
model = Kronos(
    s1_bits=10, s2_bits=10,
    n_layers=24,      # 从 12 增加到 24
    d_model=512,
    n_heads=8,
    ff_dim=1280,
    ...
)
```

**影响**：参数量近似翻倍，推理时间近似翻倍，显存占用增加。

**层数与性能的关系**：Transformer 层数决定模型能捕捉的"抽象层级"。每一层可以学习一种不同层次的模式——浅层捕捉局部波动特征（如单根 K 线的形态），深层捕捉全局趋势特征（如多根 K 线的周期性规律）。增加到 24 层意味着模型有更多层级来构建抽象表示，但也更容易过拟合（特别是在数据量有限时）。经验上，当训练数据量超过百万条 K 线时，更深层数的收益才开始明显体现。

```python
# 修改 s1_bits 和 s2_bits
tokenizer = KronosTokenizer(
    ..., s1_bits=12, s2_bits=8, ...    # 默认 10/10
)
model = Kronos(s1_bits=12, s2_bits=8, ...)
```

**影响**：s1 词汇量变为 4,096，s2 词汇量变为 256。需要**同时重新训练**分词器和预测模型。

#### 替换归一化层

将 RMSNorm 替换为 LayerNorm：

```python
# 在 module.py 中
class TransformerBlock(nn.Module):
    def __init__(self, ...):
        self.norm1 = nn.LayerNorm(d_model)    # 替换 RMSNorm
        self.norm2 = nn.LayerNorm(d_model)
```

**风险等级**：低。LayerNorm 和 RMSNorm 功能等价，但需验证训练收敛性。

---

## 回归测试

### 运行现有测试

```bash
pytest tests/test_kronos_regression.py -v
```

测试内容：

- **输出一致性测试**：对比模型预测结果与固定种子的预期输出
- **MSE 测试**：验证预测结果与预期值的均方误差在阈值内

测试参数化：

- 使用固定的模型版本（`MODEL_REVISION`、`TOKENIZER_REVISION`）确保可复现
- 在 CPU 上运行，避免 GPU 随机性

### 编写新测试

修改核心代码后，建议补充对应的单元测试：

```python
import torch
import pytest

def test_custom_sampling():
    """测试自定义采样策略"""
    from model.kronos import sample_from_logits

    logits = torch.randn(1, 1024)

    # 测试贪婪模式
    result = sample_from_logits(logits, sample_logits=False)
    assert result.shape == (1, 1)

    # 测试温度采样
    result = sample_from_logits(logits, temperature=0.5, sample_logits=True)
    assert result.shape == (1, 1)

def test_custom_dataset():
    """测试自定义数据集的输出形状"""
    # 使用项目自带的测试数据
    ds = CustomKlineDataset(
        data_path="./examples/data/XSHG_5min_600977.csv",
        lookback_window=128, predict_window=24
    )
    sample = ds[0]
    assert sample.shape == (152, 6)  # 128 + 24 = 152
```

---

## 贡献代码的建议

### 代码风格

- 遵循项目现有的命名规范（snake_case 函数/变量，PascalCase 类）
- 保持与现有代码一致的缩进和格式
- 新增公共方法需要添加 docstring

### 提交前检查清单

- [ ] 所有现有测试通过：`pytest tests/test_kronos_regression.py`
- [ ] 新功能有对应的测试
- [ ] 未破坏 `from model import Kronos, KronosTokenizer, KronosPredictor` 的导入路径
- [ ] 修改涉及参数变更时，确保默认值保持向后兼容
- [ ] 未引入新的硬编码依赖

---

## 自测清单

- [ ] 我能在源码中定位添加新时间特征需要修改的所有文件
- [ ] 我能实现一个自定义数据集并确保输出形状正确
- [ ] 我知道修改 `s1_bits` / `s2_bits` 后需要重新训练模型
- [ ] 我能运行回归测试并判断修改是否破坏了现有功能
- [ ] 我能评估一个修改的"影响范围"（参考架构文档的影响矩阵）

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [CSV 微调指南](02-finetune-csv.md) | ⭐⭐⭐ | 自定义数据集的微调流程 |
| [Qlib 微调指南](01-finetune-qlib.md) | ⭐⭐⭐ | A 股专用微调流水线 |
| [源码走读](../architecture/04-source-code-walkthrough.md) | ⭐⭐⭐⭐ | 深入理解实现细节 |

## 相关文档

| 文档 | 说明 |
|------|------|
| [系统架构分析](../architecture/01-system-architecture.md) | 理解模块间依赖与影响矩阵 |
| [源码走读](../architecture/04-source-code-walkthrough.md) | 逐模块理解核心代码 |
| [常见错误排查](../references/troubleshooting.md) | 调试技巧与错误排查方法 |
| [模型对比与选型](07-model-comparison.md) | 不同模型规模的参数差异 |

---
**文档元信息**
难度：⭐⭐⭐ | 类型：开发指南 | 预计阅读时间：25 分钟
更新日期：2026-04-11
