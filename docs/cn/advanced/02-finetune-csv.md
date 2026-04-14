# CSV 微调指南 ⭐⭐⭐

> **目标读者**：想使用自有 CSV 数据微调 Kronos 的开发者
> **前置要求**：理解核心概念，熟悉 YAML 配置

### 学习目标

阅读本文后，你将能够：

- [ ] 创建 YAML 配置文件并使用 `train_sequential.py` 一键完成两阶段微调
- [ ] 理解 `CustomKlineDataset` 的数据加载逻辑及与 `SequentialTrainer` 的协作关系
- [ ] 根据数据特征决定是否需要微调分词器，并排查训练不收敛问题

---

## 概述

CSV 微调流水线是 Qlib 微调的轻量替代方案，不需要安装 Qlib，直接使用标准 CSV 文件作为输入。它支持单 GPU 和 DDP 分布式训练，并通过 YAML 配置文件管理所有参数。

### 为什么选择 YAML 配置？

Qlib 微调流水线使用 Python 类（`finetune/config.py` 中的 `Config` 类）管理配置——修改参数需要直接编辑 Python 代码。CSV 微调流水线改用 YAML 文件，带来以下优势：

| 维度 | Python 类配置 | YAML 配置 |
|------|-------------|----------|
| 修改方式 | 编辑 .py 文件，需要了解 Python 语法 | 编辑 .yaml 文件，纯文本格式 |
| 版本管理 | 配置散落在代码中，难以追踪变更 | 每个实验一个 .yaml 文件，Git 可追踪差异 |
| 可复现性 | 需要手动记录每次运行的参数 | 实验配置即文件，天然可复现 |
| 自动化 | 需要额外脚本管理参数 | 可用 `--config` 参数切换不同实验配置 |

如果你需要快速实验不同参数组合，只需复制一份 YAML 文件并修改参数值，无需改动任何 Python 代码。

---

## 与 Qlib 微调的区别

| 维度 | Qlib 微调 | CSV 微调 |
|------|----------|----------|
| 数据源 | Qlib 平台（中国 A 股） | 任意 CSV 文件 |
| 配置方式 | Python 类（`Config`） | YAML 文件 |
| 分布式训练 | 必须（`torchrun`） | 可选（支持单 GPU） |
| 训练策略 | 分别运行两个脚本 | 一键顺序执行（`train_sequential.py`） |
| 日志系统 | Comet ML | Python logging + 文件日志 |

---

## 数据准备

### CSV 格式要求

```csv
timestamps,open,high,low,close,volume,amount
2024-01-02 09:30:00,10.50,10.55,10.48,10.52,12345,129789.0
2024-01-02 09:35:00,10.52,10.58,10.50,10.56,15678,165164.68
...
```

**要求**：

- 第一列必须是 `timestamps`（`pd.to_datetime` 可解析的格式）
- 必须包含 `open`、`high`、`low`、`close` 四列
- `volume` 和 `amount` 列可选
- 数据按时间升序排列
- 不能有 NaN 值（如有，会自动前向填充）

> **最低数据量要求**：CSV 文件至少应包含 `lookback_window + predict_window` 行数据。例如，默认配置（512 + 48 = 560 行）是单条样本的最低要求。建议至少准备数万行数据以获得稳定的微调效果。

### 数据划分

数据按比例自动划分为训练集和验证集：

```yaml
data:
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0
```

---

## 配置文件

创建 `config.yaml` 配置文件：

```yaml
# 实验配置
experiment:
  name: "my_finetune"
  description: "Fine-tune Kronos on custom data"
  use_comet: false
  train_tokenizer: true        # 是否训练分词器
  train_basemodel: true        # 是否训练预测模型
  skip_existing: false         # 是否跳过已存在的模型
  pre_trained: true            # 是否使用预训练权重（tokenizer 和 predictor 均适用）

# 数据配置
data:
  data_path: "./data/my_kline.csv"
  lookback_window: 512         # 历史窗口
  predict_window: 48           # 预测窗口
  max_context: 512             # 最大上下文
  clip: 5.0                    # 裁剪范围
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0

# 训练配置
training:
  tokenizer_epochs: 30         # 分词器训练轮数
  basemodel_epochs: 30         # 预测模型训练轮数
  batch_size: 160
  log_interval: 50
  num_workers: 6
  seed: 100
  tokenizer_learning_rate: 2e-4
  predictor_learning_rate: 4e-5
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1
  accumulation_steps: 1

# 模型路径
model_paths:
  exp_name: "my_experiment"
  base_path: "./outputs"
  pretrained_tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
  pretrained_predictor: "NeoQuasar/Kronos-small"
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

# 设备配置
device:
  use_cuda: true
  device_id: 0

# 分布式配置
distributed:
  use_ddp: false
  backend: "nccl"
```

### 关键配置项说明

| 配置项 | 说明 |
|--------|------|
| `experiment.pre_trained` | 设为 `true` 使用预训练权重微调；设为 `false` 从随机初始化训练 |
| `experiment.train_tokenizer` | 设为 `false` 可跳过分词器训练，直接训练预测模型 |
| `experiment.skip_existing` | 设为 `true` 跳过已存在的模型文件，避免重复训练。注意：这不是断点续训，而是从头开始或完全跳过 |
| `model_paths.pretrained_tokenizer` | 可以是 HuggingFace Hub ID 或本地路径 |

---

## 启动训练

### 方式 1：顺序训练（推荐）

一键完成分词器 + 预测模型的顺序微调：

```bash
python finetune_csv/train_sequential.py --config config.yaml
```

可选参数：

```bash
--skip-tokenizer    # 跳过分词器训练
--skip-basemodel    # 跳过预测模型训练
--skip-existing     # 跳过已存在的模型
```

### 方式 2：分别训练

```bash
# 先训练分词器
python finetune_csv/finetune_tokenizer.py --config config.yaml

# 再训练预测模型
python finetune_csv/finetune_base_model.py --config config.yaml
```

### 分布式训练

```bash
torchrun --standalone --nproc_per_node=2 finetune_csv/train_sequential.py --config config.yaml
```

---

## 训练流程详解

### SequentialTrainer

`train_sequential.py` 中的 `SequentialTrainer` 类编排了完整的训练流程。它负责按顺序调用分词器训练和预测模型训练，并在各阶段创建对应的 `CustomKlineDataset` 实例：

```
1. 加载配置 → 创建输出目录
2. 如果 train_tokenizer=true：
   ├─ 加载预训练/随机初始化的分词器
   ├─ 创建 CustomKlineDataset(data_type="train"/"val")
   ├─ 训练分词器
   └─ 保存最佳模型
3. 如果 train_basemodel=true：
   ├─ 加载微调后的分词器
   ├─ 加载预训练/随机初始化的预测模型
   ├─ 创建 CustomKlineDataset(data_type="train"/"val")
   ├─ 训练预测模型
   └─ 保存最佳模型
```

> **关系说明**：`SequentialTrainer` 是编排器，负责流程控制和模型保存。`CustomKlineDataset`（定义在 `finetune_csv/finetune_base_model.py` 中）是数据层，负责从 CSV 文件中采样训练/验证样本。`SequentialTrainer` 在每个训练阶段都会创建新的 `CustomKlineDataset` 实例来加载数据。

### 日志系统

CSV 微调使用 Python `logging` 模块记录训练日志：

```
outputs/my_experiment/logs/
├── tokenizer_training_rank_0.log
└── basemodel_training_rank_0.log
```

日志文件支持轮转（每个文件最大 10 MB，保留 5 个备份）。

---

## CustomKlineDataset

`CustomKlineDataset` 定义在 `finetune_csv/finetune_base_model.py` 中，负责从 CSV 文件加载和采样数据：

```python
dataset = CustomKlineDataset(
    data_path="./data/my_kline.csv",
    data_type="train",          # "train" / "val" / "test"
    lookback_window=512,
    predict_window=48,
    clip=5.0,
    seed=100,
    train_ratio=0.9,
    val_ratio=0.1
)
```

### 数据采样策略

- **训练集**：使用确定性伪随机采样（基于 epoch seed），确保不同 epoch 的数据分布不同
- **验证集**：顺序采样，确保验证结果可复现

---

## 输出结构

```
outputs/my_experiment/
├── tokenizer/
│   ├── best_model/              # 分词器最佳模型
│   │   ├── config.json
│   │   └── model.safetensors
│   └── ...
├── basemodel/
│   ├── best_model/              # 预测模型最佳模型
│   │   ├── config.json
│   │   └── model.safetensors
│   └── ...
└── logs/
    ├── tokenizer_training_rank_0.log
    └── basemodel_training_rank_0.log
```

---

## 如何判断微调效果

### 验证损失的期望范围

- **分词器微调**：验证 MSE 通常在 0.01-0.1 之间。前几个 epoch 下降明显，后期趋于平稳。
- **预测模型微调**：验证交叉熵（CE）通常在 3.0-6.0 之间。与 Qlib 微调类似，CE 应随训练推进稳步下降。

### 如何与预训练模型对比

1. 微调前使用预训练模型对验证集做预测，记录基线指标
2. 微调完成后，用微调后的模型对同一验证集做预测
3. 对比维度：重建 MSE（分词器）、预测 CE（预测模型）、预测价格与真实价格的 MAE

### 过拟合的判断标准

- 训练损失持续下降，但验证损失不再改善甚至回升
- best_model 保存在非常早的 epoch（如第 3-5 轮），说明后续训练没有带来增益
- 微调后的模型在训练数据上表现很好，但在新数据上表现不如预训练模型

应对方法：减少 `tokenizer_epochs` / `basemodel_epochs`、增大 `adam_weight_decay`、减小学习率，或增加数据量。

---

## 常见问题

### Q: 训练数据量需要多少？

**A**: 取决于数据的时间跨度和市场特征。一般建议至少包含几千个数据点（例如，5 分钟线数据至少几个月）。数据太少可能导致过拟合。

### Q: 如何使用微调后的模型进行预测？

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("outputs/my_experiment/tokenizer/best_model")
model = Kronos.from_pretrained("outputs/my_experiment/basemodel/best_model")
predictor = KronosPredictor(model, tokenizer)
```

### Q: 分词器和预测模型都需要微调吗？

**A**: 不一定。如果数据特征与预训练数据相似（如常规股票 K 线），只微调预测模型通常足够。只有当数据特征差异较大（如加密货币、特殊指标）时，才需要微调分词器。

设置 `experiment.train_tokenizer: false` 即可跳过分词器训练。

### Q: 训练不收敛（损失不下降）怎么办？

**A**: 训练不收敛通常由以下原因导致，请逐一排查：

1. **数据量不足**：确保 CSV 文件至少包含几千行有效数据，数据太少会导致梯度估计不稳定
2. **学习率过大**：尝试将 `tokenizer_learning_rate` 降至 1e-4、`predictor_learning_rate` 降至 1e-5
3. **数据格式问题**：检查 CSV 中是否有大量 NaN（虽然会自动前向填充，但连续 NaN 过多会降低数据质量）、是否有异常值（如价格为 0 或负数）
4. **未使用预训练权重**：确认 `experiment.pre_trained: true`，从随机初始化训练需要更多数据和更长训练时间
5. **clip 值设置不当**：默认 `clip: 5.0`，如果数据波动极大或极小，可适当调整

---

## 动手练习

### 练习 1：使用测试数据运行一轮微调

项目自带测试数据文件 `examples/data/XSHG_5min_600977.csv`。创建一个最小化的 `config.yaml`，使用该数据完成一轮微调：

```yaml
experiment:
  name: "test_run"
  train_tokenizer: true
  train_basemodel: true
  pre_trained: true

data:
  data_path: "./examples/data/XSHG_5min_600977.csv"
  lookback_window: 128
  predict_window: 24
  clip: 5.0
  train_ratio: 0.9
  val_ratio: 0.1

training:
  tokenizer_epochs: 3
  basemodel_epochs: 3
  batch_size: 32
  log_interval: 10
  num_workers: 0

model_paths:
  exp_name: "test_experiment"
  base_path: "./outputs"
  pretrained_tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
  pretrained_predictor: "NeoQuasar/Kronos-small"

device:
  use_cuda: false

distributed:
  use_ddp: false
```

运行：

```bash
python finetune_csv/train_sequential.py --config config.yaml
```

**验证方法**：检查 `outputs/test_experiment/` 目录下是否生成了 `tokenizer/best_model/` 和 `basemodel/best_model/` 目录，以及对应的日志文件。

---

## 自测清单

完成本指南后，检查你是否掌握了以下要点：

- [ ] 能创建一份有效的 `config.yaml` 并运行 `train_sequential.py`
- [ ] 知道何时应将 `experiment.train_tokenizer` 设为 `false`
- [ ] 能解释 `CustomKlineDataset` 中训练集和验证集采样策略的区别
- [ ] 遇到训练不收敛时知道至少三种排查方向
- [ ] 能从输出目录结构中找到微调后的模型文件路径

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [Qlib 微调指南](01-finetune-qlib.md) | ⭐⭐⭐ | 对比两种微调方式 |
| [源码走读](../architecture/04-source-code-walkthrough.md) | ⭐⭐⭐⭐ | 深入理解训练代码 |

## 相关文档

- **并行**：[Qlib 微调指南](01-finetune-qlib.md) — A 股专用微调流水线（对比参考）
- **前置**：[KronosTokenizer 详解](../core-concepts/02-tokenizer.md) — 理解分词器训练目标
- **前置**：[KronosPredictor 使用指南](../core-concepts/04-predictor.md) — 理解预测模型结构
- **进阶**：[源码走读](../architecture/04-source-code-walkthrough.md) — 深入训练代码细节
- **实战**：[使用场景与实战案例](06-use-cases.md) — 微调后的应用示例

---
**文档元信息**
难度：⭐⭐⭐ | 类型：进阶指南 | 预计阅读时间：20 分钟
更新日期：2026-04-11
