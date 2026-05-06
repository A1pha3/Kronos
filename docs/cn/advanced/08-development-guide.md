# 开发扩展指南 ⭐⭐⭐

> **目标读者**：希望深入 Kronos 代码进行二次开发或贡献代码的开发者
> **前置要求**：已完成 [系统架构分析](../architecture/01-system-architecture.md) 和 [源码走读](../architecture/04-source-code-walkthrough.md)

---

## 目录

- [学习目标](#学习目标)
- [代码结构速览](#代码结构速览)
- [扩展点 1：添加自定义时间特征](#扩展点-1添加自定义时间特征)
- [扩展点 2：自定义数据集](#扩展点-2自定义数据集)
- [扩展点 3：修改采样策略](#扩展点-3修改采样策略)
- [扩展点 4：修改模型架构](#扩展点-4修改模型架构)
- [常见开发错误](#常见开发错误)
- [常见误区](#常见误区)
- [回归测试](#回归测试)
- [贡献代码的建议](#贡献代码的建议)
- [动手练习](#动手练习)

---

## 学习目标

本文覆盖 Kronos 的主要扩展点和修改时的安全边界。完成学习后，你应能做到：

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

修改时遵循以下原则：

1. **基础组件改 `module.py`**：TransformerBlock、BSQuantizer、Embedding 等底层组件集中在此文件
2. **推理流程改 `kronos.py`**：预测接口、采样逻辑、自回归推理在此文件
3. **保持向后兼容**：修改核心 API 时确保已有调用方不受影响
4. **改完跑回归测试**：每次修改后执行 `pytest tests/test_kronos_regression.py`

### 代码修改的安全边界

下表列出各类修改的风险等级。修改前先查表，避免引入隐蔽的回归问题：

| 修改类型 | 风险 | 说明 |
|---------|------|------|
| 添加新的 API 端点 | 低 | 不影响已有调用方 |
| 修改采样策略（不影响模型权重） | 低 | 仅改变采样行为，不涉及权重 |
| 添加新的数据预处理步骤 | 中 | 需验证对推理结果的影响 |
| 修改 `s1_bits` / `s2_bits` | 高 | 词汇量变化，必须重新训练分词器和预测模型 |
| 修改 `d_model` / `n_layers` 等结构参数 | 高 | 张量形状变化，必须重新训练 |
| 添加新的时间特征 | 高 | 嵌入表不兼容，必须重新训练 |
| 替换 RMSNorm 为 LayerNorm | 高 | 从零训练可行；加载预训练权重时因结构不兼容（RMSNorm 仅含 `weight`，LayerNorm 含 `weight` + `bias`）会失败 |
| 替换 RoPE 为其他位置编码 | 高 | 需重新验证长序列效果；是否需重新训练取决于是否加载预训练权重 |

---

## 扩展点 1：添加自定义时间特征

### 当前实现

`TemporalEmbedding`（`module.py:536`）从时间戳中提取 5 个特征：`minute`、`hour`、`weekday`、`day`、`month`。特征提取逻辑位于 `calc_time_stamps()` 函数（`kronos.py:472`）。

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
quarter_size = 4    # 季度：1-4
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

        # 时间特征（训练管线需要同时提供数据和时间戳）
        if 'timestamps' in self.data.columns:
            ts = pd.to_datetime(window['timestamps'])
            x_stamp = np.stack([
                ts.dt.minute.values, ts.dt.hour.values,
                ts.dt.weekday.values, ts.dt.day.values,
                ts.dt.month.values
            ], axis=-1).astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(x_stamp)
        else:
            return torch.from_numpy(x)
```

**关键要求**：

1. `__getitem__` 应返回标准化后的数据张量和时间特征张量，形状分别为 `(lookback + predict_window, 6)` 和 `(lookback + predict_window, 5)`
2. 数据列顺序必须与预训练一致：`open, high, low, close, volume, amount`
3. 实例级标准化（每条样本独立）是跨市场泛化的关键
4. 时间特征提取方式应与 `calc_time_stamps()` 一致：`minute, hour, weekday, day, month`

---

## 扩展点 3：修改采样策略

### 当前实现

`sample_from_logits()` 函数（`kronos.py:373`）支持温度采样、top-k 和 top-p 过滤。

### 扩展：实现 min-p 采样（示例扩展，非现有代码）

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

| 需要同步修改 | 文件 | 说明 |
|-------------|------|------|
| `sample_from_logits` | `kronos.py:373` | 添加 `min_p` 参数和过滤逻辑 |
| `auto_regressive_inference` | `kronos.py:389` | 透传 `min_p` 参数 |
| `KronosPredictor.predict` | `kronos.py:519` | 暴露 `min_p` 参数给调用方 |
| `KronosPredictor.predict_batch` | `kronos.py:562` | 同上 |

> **注意**：`KronosPredictor.predict` 和 `predict_batch` 的现有签名中都包含 `top_k` 参数（默认值为 `0`，即不过滤）。如果添加 `min_p`，建议将参数放在 `top_p` 之后，保持与现有参数风格一致。

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

> **注意**：KronosTokenizer 的 encoder/decoder 层数存在"减一"行为——内部使用 `range(enc_layers - 1)` 构建 ModuleList，因此实际层数为参数值减 1。例如 `n_enc_layers=4` 只会创建 3 个 TransformerBlock。修改层数时请留意这一差异。

**层数与性能的关系**：Transformer 层数决定模型能捕捉的"抽象层级"——浅层捕捉局部波动特征（如单根 K 线形态），深层捕捉全局趋势特征（如多根 K 线的周期性规律）。增加到 24 层意味着模型有更多层级来构建抽象表示，但也更容易过拟合（尤其在数据量有限时）。经验上，训练数据量超过百万条 K 线时，更深层数的收益才开始明显体现。

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

**风险等级**：从零训练时为低风险（LayerNorm 和 RMSNorm 功能等价，但需验证训练收敛性）；加载预训练权重时为高风险——两者的权重结构不兼容（RMSNorm 仅含 `weight`，LayerNorm 同时含 `weight` 和 `bias`），`load_state_dict()` 会因键名/形状不匹配而失败。

---

## 常见开发错误

### 错误 1：修改模型结构后加载预训练权重失败

**症状**：`RuntimeError: Error(s) in loading state_dict for Kronos: ... size mismatch ...`

**原因**：修改了 `d_model`、`n_layers`、`s1_bits` 等结构参数后，预训练权重的张量形状与新模型不匹配。这是 Kronos 开发中最常见的错误之一。

**解决方法**：

```python
# 方法 1：从零开始训练（不加载预训练权重）
model = Kronos(...)
# 不要调用 from_pretrained()，直接使用随机初始化的权重

# 方法 2：仅加载兼容的部分权重
pretrained = Kronos.from_pretrained("NeoQuasar/Kronos-base")
model = Kronos(...)  # 新结构
model.load_state_dict(pretrained.state_dict(), strict=False)
# strict=False 会跳过形状不匹配的层，打印警告信息
```

### 错误 2：top_k 与 top_p 参数混淆

**症状**：采样结果异常——所有输出相同或完全随机。

**原因**：`sample_from_logits` 中 `top_k=0` 表示不过滤（保留所有令牌），而 `top_p=0.9` 表示核采样。两者是独立的过滤条件，不是"top-k 等于 0 就不采样"。

```python
# 正确用法
result = sample_from_logits(logits, top_k=50, top_p=0.9)   # top-k + top-p 联合过滤
result = sample_from_logits(logits, top_k=0, top_p=0.9)    # 仅 top-p（默认行为）
result = sample_from_logits(logits, top_k=50, top_p=1.0)   # 仅 top-k
```

### 错误 3：修改时间特征后忘记更新 predict 流程

**症状**：`IndexError: index 5 is out of bounds` 或预测结果维度不匹配。

**原因**：添加了新的时间特征（如"季度"），但 `KronosPredictor.time_cols` 仍为旧列表，导致 `calc_time_stamps()` 返回的列数与模型期望的不一致。需要同步更新三处：`calc_time_stamps`、`TemporalEmbedding.__init__` + `forward`、`KronosPredictor.time_cols`（参见扩展点 1）。

---

## 常见误区

以下是开发者在二次开发 Kronos 时容易产生的误解：

### 误区 1：修改模型结构后还能加载预训练权重

很多开发者认为只要保持类名和 `from_pretrained()` 调用不变，修改内部结构后仍可正常加载预训练权重。实际上，预训练权重按参数名和形状一一对应存储——任何结构参数的变化都会导致张量形状不匹配，`load_state_dict()` 默认（`strict=True`）会直接报错。

使用 `strict=False` 可以跳过不兼容的层，但被跳过的层将以随机初始化状态参与推理，输出质量会大幅下降。结构修改通常意味着必须重新训练。

### 误区 2：自定义数据集不需要实例级标准化

Kronos 的设计目标是跨市场、跨品种泛化。不同标的的价格量级差异巨大（A 股可能在 1-100 元，加密货币可能在数千至数万美元），如果使用全局标准化，模型将难以适应分布差异大的新数据。实例级标准化（对每条样本独立计算均值和标准差）是保证跨市场泛化的关键设计决策。自定义数据集时，务必在 `__getitem__` 中对每条样本单独执行标准化。

### 误区 3：回归测试覆盖了所有场景

现有的回归测试（`test_kronos_regression.py`）仅验证模型在固定输入下的输出一致性——确认预训练模型在特定版本下产生与预期相同的数值结果。它不验证新架构、新数据或新任务的正确性。如果你修改了模型结构或引入了新的训练流水线，回归测试通过只说明未破坏原有推理路径，但无法保证新功能的正确性。新架构或新功能需要编写专门的测试。

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

### 测试策略

在决定如何测试时，区分以下两种情况：

**何时添加新测试**：

- 新增了公共 API 方法或新的扩展点（如自定义采样策略、新的数据集类）
- 修改了影响推理结果的关键路径（如 `sample_from_logits` 的过滤逻辑、`BSQuantizer` 的量化逻辑）
- 引入了新的模型结构变体（需要验证新结构的前向传播不报错、输出形状正确）
- 添加了新的数据预处理或后处理步骤

**何时修改现有测试即可**：

- 仅修复了 bug 且修复后的预期输出发生变化（更新断言中的预期值）
- 调整了测试参数化（如增加新的 context length 组合）
- 重构了内部实现但外部行为不变（现有测试应仍然通过，无需额外测试）

**原则**：回归测试保证"改了不该改的东西会被发现"，新测试保证"新加的东西按预期工作"，两者互补。

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

- snake_case 函数/变量，PascalCase 类（与项目现有风格一致）
- 新增公共方法需要添加 docstring
- 缩进和格式与现有代码保持一致

### 提交前检查清单

- [ ] 所有现有测试通过：`pytest tests/test_kronos_regression.py`
- [ ] 新功能有对应的测试
- [ ] 未破坏 `from model import Kronos, KronosTokenizer, KronosPredictor` 的导入路径
- [ ] 修改涉及参数变更时，确保默认值保持向后兼容
- [ ] 未引入新的硬编码依赖
- [ ] 如果修改了模型结构参数，确认不会影响预训练权重加载（或已使用 `strict=False`）

> 修改的安全性评估可参考 [系统架构分析](../architecture/01-system-architecture.md) 中的"组件替换风险评估表"。

---

## 动手练习

### 练习 1：添加"季度"时间特征

在 `model/module.py` 中找到 `TemporalEmbedding`，按照"扩展点 1"的步骤，增加一个 `quarter_embed`（季度，取值 1-4）。修改后用以下代码验证：

```python
import torch
from model.module import TemporalEmbedding

# 创建带季度的时间嵌入（假设已修改）
te = TemporalEmbedding(d_model=256, learn_pe=True)
te.eval()

# 构造输入：6 列时间特征（原 5 列 + 季度）
stamp = torch.tensor([[[30, 14, 1, 15, 6, 2]]])  # minute=30, hour=14, weekday=1, day=15, month=6, quarter=2
with torch.no_grad():
    emb = te(stamp)
print(f"嵌入形状: {emb.shape}")  # 期望输出: (1, 1, 256)
```

验证清单（逐步确认）：

1. `calc_time_stamps()` 返回的 DataFrame 从 5 列变为 6 列，新增列名为 `quarter`
2. `TemporalEmbedding` 实例具有 `quarter_embed` 属性（可用 `hasattr(te, 'quarter_embed')` 检查）
3. 嵌入输出形状为 `(batch, seq_len, d_model)`，不含 NaN 或 Inf
4. 回归测试 `pytest tests/test_kronos_regression.py` 会失败（因为预训练权重不兼容），这是预期行为

**提示**：此修改需要同步更新三个位置 -- `model/kronos.py` 中的 `calc_time_stamps()` 函数、`model/module.py` 中的 `TemporalEmbedding`，以及 `model/kronos.py` 中 `KronosPredictor.time_cols` 列表。参考"扩展点 1"的影响范围表逐一修改。

### 练习 2：实现带数据增强的自定义数据集

基于"扩展点 2"的模板，创建一个在 `__getitem__` 中添加高斯噪声的数据增强 Dataset：

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AugmentedKlineDataset(Dataset):
    def __init__(self, csv_path, seq_len=512, noise_std=0.01):
        df = pd.read_csv(csv_path)
        self.data = df[['open', 'high', 'low', 'close', 'volume', 'amount']].values.astype('float32')
        self.seq_len = seq_len
        self.noise_std = noise_std
        self.valid_starts = list(range(0, len(self.data) - seq_len))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        x = self.data[start:start + self.seq_len].copy()
        # 数据增强：添加高斯噪声
        if self.noise_std > 0:
            x += np.random.normal(0, self.noise_std * x.std(axis=0), size=x.shape).astype('float32')
        return torch.from_numpy(x)
```

用以下代码验证：

```python
# 使用项目自带的示例数据
ds = AugmentedKlineDataset("examples/data/XSHG_5min_600977.csv", seq_len=128)

# 验证 1：输出形状
sample = ds[0]
assert sample.shape == (128, 6), f"形状错误: {sample.shape}"
print(f"输出形状: {sample.shape}")  # 期望: (128, 6)

# 验证 2：噪声效果（增强后标准差应略大于原始数据）
raw = ds.data[0:128]
print(f"原始 std: {raw.std():.4f}")
print(f"增强 std: {sample.std():.4f}")  # 期望略大于原始值

# 验证 3：多次调用结果不同（随机性）
assert not torch.equal(ds[0], ds[0]), "两次调用应产生不同结果"
```

**思考题**：为什么这个 Dataset 没有包含标准化步骤？如果在 `__getitem__` 中同时做标准化和加噪，应该先做哪个？

---

## 自测清单

- [ ] 能在源码中定位添加新时间特征需要修改的所有文件
- [ ] 能实现一个自定义数据集并确保输出形状正确
- [ ] 知道修改 `s1_bits` / `s2_bits` 后需要重新训练模型
- [ ] 能运行回归测试并判断修改是否破坏了现有功能
- [ ] 能评估一个修改的"影响范围"（参考架构文档的影响矩阵）

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

