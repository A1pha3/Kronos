# CSV 微调指南 ⭐⭐⭐

> **目标读者**：想使用自有 CSV 数据微调 Kronos 的开发者
> **前置要求**：理解核心概念，熟悉 YAML 配置

## 学习目标

这篇文档讲解如何用通用 CSV 数据微调 Kronos：

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

- 必须包含 `timestamps` 列（`pd.to_datetime` 可解析的格式）
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

> **`pre_trained` 的细粒度控制**：YAML 中的 `experiment.pre_trained` 是统一开关，同时控制分词器和预测模型是否加载预训练权重。如果需要独立控制（如分词器从零训练但预测模型使用预训练权重），可以在 YAML 中分别设置 `experiment.pre_trained_tokenizer: false` 和 `experiment.pre_trained_predictor: true`。当这两个细粒度标志存在时，它们会覆盖 `pre_trained` 的值。此行为由当前 `ConfigLoader` 实现（`finetune_csv/config_loader.py`）提供支持，如果你的代码版本较旧，请确认该文件中包含对 `pre_trained_tokenizer` / `pre_trained_predictor` 的解析逻辑。

### 配置文件的路径解析机制

`ConfigLoader`（`finetune_csv/config_loader.py`）在加载 YAML 后会自动解析动态路径：

```
model_paths:
  exp_name: "my_experiment"
  base_path: "./outputs"
  # 以下路径会自动生成（如果留空）：
  # base_save_path: "./outputs/my_experiment"
  # finetuned_tokenizer: "./outputs/my_experiment/tokenizer/best_model"
```

`_resolve_dynamic_paths()` 方法基于 `exp_name` 和 `base_path` 自动拼接输出目录。如果某个路径值包含 `{exp_name}` 占位符，会被替换为实际的实验名。这确保了预测模型微调阶段能自动找到分词器微调阶段的输出——无需手动指定 `finetuned_tokenizer` 路径。

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
    lookback_window=90,         # 代码默认值；YAML 配置中的值会覆盖此默认值
    predict_window=10,          # 代码默认值；YAML 配置中的值会覆盖此默认值
    clip=5.0,
    seed=100,
    train_ratio=0.9,
    val_ratio=0.1
)
```

> **注意**：`CustomKlineDataset` 的代码默认值为 `lookback_window=90, predict_window=10`，但实际训练时 `SequentialTrainer` 会从 YAML 配置文件中读取值并传入，覆盖代码默认值。上文中 YAML 示例配置的是 `512` / `48`。

### 数据采样策略

- **训练集**：使用确定性伪随机采样（基于 epoch seed），不同 epoch 通过 `(idx * 9973 + (epoch + 1) * 104729) % (max_start + 1)` 计算起始位置，确保每个 epoch 看到不同的样本顺序和位置
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

## 训练时间估算

以下为基于不同数据规模和硬件的粗略训练时间参考，实际时间受 CPU、磁盘 I/O、数据特征等因素影响：

| 数据规模 | 模型规格 | 硬件 | 分词器微调 | 预测模型微调 | 总计 |
|---------|---------|------|-----------|-------------|------|
| 1 万行（约 2 万样本） | Kronos-small | 单 GPU（如 RTX 3090） | 约 15-30 分钟 | 约 20-40 分钟 | 约 35-70 分钟 |
| 5 万行（约 10 万样本） | Kronos-small | 单 GPU（如 RTX 3090） | 约 1-2 小时 | 约 1.5-3 小时 | 约 2.5-5 小时 |
| 5 万行 | Kronos-base | 单 GPU（如 RTX 3090） | 约 2-3 小时 | 约 3-5 小时 | 约 5-8 小时 |
| 10 万行（约 20 万样本） | Kronos-small | 2 GPU（DDP） | 约 1-1.5 小时 | 约 1-2 小时 | 约 2-3.5 小时 |
| 10 万行 | Kronos-base | 2 GPU（DDP） | 约 2-3 小时 | 约 2.5-4 小时 | 约 4.5-7 小时 |

> **说明**：样本数按 `数据行数 / (lookback_window + predict_window)` 估算，实际可用样本取决于数据连续性。Kronos-large 在单 GPU 上可能遇到显存不足，建议使用 DDP 或减小 `batch_size`。

---

## 常见误区

### 误区 1：`pre_trained=false` 可以获得更好的定制化效果

**事实**：设置 `experiment.pre_trained: false` 意味着从随机初始化开始训练，而非在预训练权重基础上微调。这并非"更强的定制"，而是放弃模型已学到的通用 K 线表示能力。从零训练要达到同等效果，通常需要**数十倍甚至数百倍**的数据量和训练时间。

**正确做法**：绝大多数场景下应保持 `pre_trained: true`。只有在以下情况才考虑 `false`：数据特征与标准 K 线完全不同（如归一化方式特殊的衍生指标），且你有足够的数据（建议 10 万行以上）和计算资源进行充分训练。

### 误区 2：`batch_size` 越大训练效果越好

**事实**：增大 `batch_size` 可以提高 GPU 利用率和训练速度，但并非越大越好：

- **显存溢出（OOM）**：Kronos-small 在 `batch_size=160`、`lookback_window=512` 时约需 20-24 GB 显存。Kronos-base 或 Kronos-large 在相同配置下可能需要 40-80 GB 显存，超出单卡容量会直接报错
- **泛化性能下降**：过大的 batch_size 可能导致模型收敛到尖锐的局部最优，降低泛化能力。研究表明适度的 batch_size（如 32-128）往往比极大 batch_size 的泛化效果更好

**正确做法**：从 `batch_size: 32` 或 `batch_size: 64` 开始，确认训练正常后再逐步增大。如果遇到 OOM，优先减小 `batch_size` 而非 `lookback_window`，因为上下文长度对模型性能影响更大。如果显存充裕且需要等价于大 batch_size 的效果，可以通过 `accumulation_steps` 模拟更大的有效 batch_size。

### 误区 3：不需要验证集，直接用全部数据训练

**事实**：没有验证集就无法判断模型是否过拟合。验证集在 CSV 微调流程中有两个关键作用：

- **早停判断**：`SequentialTrainer` 保存 `best_model` 的依据是验证集上的最佳表现。没有验证集，模型只会在最后一个 epoch 保存，而这个 epoch 很可能已经严重过拟合
- **超参数调优**：调整学习率、epoch 数量等超参数时，必须依赖验证集指标做判断。仅看训练损失会给你"一切正常"的错误信心

**正确做法**：保持 `val_ratio` 至少为 0.1（即 10% 数据用于验证）。如果数据量很少（如不足 5000 行），可以考虑 `val_ratio: 0.15` 或使用 K 折交叉验证（需要自行实现）。不要为了多几个训练样本而牺牲验证集——失去过拟合检测能力的代价远大于少训练几百个样本的损失。

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

### 练习 2：分析训练日志判断收敛状态

完成练习 1 后，打开 `outputs/test_experiment/logs/` 目录中的日志文件，完成以下分析：

1. 在 `tokenizer_training_rank_0.log` 中，找到最后一个 epoch 的训练损失和验证损失，计算两者之比（训练损失/验证损失）。如果比值远大于 1，说明什么问题？
2. 在 `basemodel_training_rank_0.log` 中，观察验证 CE 的变化趋势——是持续下降、先降后升、还是基本不变？每种趋势分别意味着什么？

**验证方法**：对照本文"如何判断微调效果"和"过拟合的判断标准"两节的内容。训练损失远大于验证损失可能是因为训练损失包含了更多组件（如 bsq_loss）；验证 CE 先降后升是典型的过拟合信号，基本不变则说明模型可能没有充分学习。

---

## 自测清单

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

