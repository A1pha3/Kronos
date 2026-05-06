# Qlib 微调指南 ⭐⭐⭐

> **目标读者**：想使用中国 A 股数据微调 Kronos 的开发者
> **前置要求**：理解核心概念，熟悉 PyTorch 训练流程

## 学习目标

完成这篇文档后，你能独立完成 Qlib 微调的全流程：

- [ ] 配置并运行 Qlib 数据预处理与两阶段微调流水线
- [ ] 理解分词器微调与预测模型微调的损失函数设计及关键超参数含义
- [ ] 使用分布式训练（torchrun）完成多 GPU 微调并判断训练效果

---

## 概述

Kronos 提供基于 [Qlib](https://github.com/microsoft/qlib) 的微调流水线，专为**中国 A 股市场**设计。微调分为两个阶段：

1. **分词器微调**：让分词器更适应 A 股数据的特征分布
2. **预测模型微调**：让预测模型更适应 A 股的市场规律

两个阶段**必须按顺序执行**——预测模型微调依赖微调后的分词器。

### 为什么需要两阶段微调？

一个自然的疑问是：为什么不直接端到端训练，而要分成两个阶段？

**分词器微调和预测模型微调优化的是不同的目标**：

| 阶段 | 优化目标 | 损失函数 | 输出 |
|------|---------|---------|------|
| 分词器微调 | 提升量化和重建精度 | MSE（重建误差）+ BSQ 损失 | 更好的令牌表示 |
| 预测模型微调 | 提升令牌预测准确率 | 交叉熵（分类误差） | 更好的预测能力 |

如果跳过分词器微调直接微调预测模型，模型仍使用预训练分词器的令牌空间——这个空间可能没有充分适配 A 股数据的特征分布（如 A 股的涨跌停限制、T+1 交易制度等特殊规律）。

**什么时候可以跳过分词器微调？** 当目标市场的数据特征与预训练数据相近（如常规股票市场），只微调预测模型通常已足够。只有当数据特征差异较大时（如加密货币衍生品、特殊指标数据），分词器微调才显著提升效果。

---

## 环境准备

### 额外依赖

```bash
pip install qlib comet-ml
```

- **qlib**：微软开源的量化投资平台，提供 A 股数据接口
- **comet-ml**（可选）：实验跟踪与可视化工具

### 准备 Qlib 数据

Qlib 数据需要提前下载到本地。使用以下命令下载中国 A 股数据：

```bash
# 下载 Qlib 中国 A 股日频数据（约 1.5 GB）
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

下载完成后，在代码中初始化：

```python
import qlib
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
```

更多数据源和更新方式请参考 [Qlib 数据文档](https://qlib.readthedocs.io/en/latest/component/data.html)。

---

## 配置

所有配置集中在 `finetune/config.py` 的 `Config` 类中。

### 数据配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `qlib_data_path` | `~/.qlib/qlib_data/cn_data` | Qlib 数据目录 |
| `instrument` | `csi300` | 股票池：`csi300`、`csi800`、`csi1000` |
| `lookback_window` | 90 | 输入历史窗口长度 |
| `predict_window` | 10 | 预测未来窗口长度 |
| `feature_list` | `['open','high','low','close','vol','amt']` | 使用的特征列 |

> **命名差异说明**：Qlib 微调流水线（`finetune/config.py`）使用 `'vol'`/`'amt'` 作为成交量/成交额的列名，这是 Qlib 平台的命名惯例。而推理阶段的 `KronosPredictor`（`model/kronos.py`）使用 `'volume'`/`'amount'`。两者是独立的代码路径——Qlib 流水线从 pickle 文件加载数据时直接使用 Qlib 的列名，`KronosPredictor` 则面向标准 CSV 输入，不需要手动转换。

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 30 | 训练轮数 |
| `batch_size` | 50 | 每 GPU 批量大小 |
| `tokenizer_learning_rate` | 2e-4 | 分词器学习率 |
| `predictor_learning_rate` | 4e-5 | 预测模型学习率 |
| `accumulation_steps` | 1 | 梯度累积步数 |
| `clip` | 5.0 | 数据裁剪范围 |

> **数据量建议**：建议至少准备数千条 K 线数据用于微调。数据量不足时，验证损失会在前几个 epoch 后就开始回升（过拟合信号）。

### 模型路径配置

```python
# 必须修改为你的模型路径
config.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
config.pretrained_predictor_path = "NeoQuasar/Kronos-small"
```

---

## 分词器微调

### 启动训练

```bash
# 使用 torchrun 启动分布式训练
torchrun --standalone --nproc_per_node=2 finetune/train_tokenizer.py
```

> 注意：此脚本**必须**通过 `torchrun` 启动，不支持直接 `python train_tokenizer.py`。

### 训练流程

```
1. 加载预训练分词器
   └─ KronosTokenizer.from_pretrained(pretrained_tokenizer_path)

2. 创建 Qlib 数据集
   └─ QlibDataset 从 pickle 文件加载预处理数据
   └─ 滑动窗口采样：随机选择 (股票, 起始位置) 对

3. 训练循环
   └─ 前向：encode → BSQ → decode
   └─ 损失：(recon_loss_pre + recon_loss_all + bsq_loss) / 2
   └─ 优化器：AdamW + OneCycleLR（pct_start=0.03, div_factor=10）
   └─ 梯度裁剪：max_norm=2.0

4. 验证与保存
   └─ 每个 epoch 计算验证 MSE
   └─ 保存最佳模型到 save_path/checkpoints/best_model/
```

### 损失函数

```python
zs, bsq_loss, _, _ = model(batch_x)
z_pre, z = zs

recon_loss_pre = MSE(z_pre, batch_x)   # 粗粒度重建损失
recon_loss_all = MSE(z, batch_x)       # 细粒度重建损失
recon_loss = recon_loss_pre + recon_loss_all

loss = (recon_loss + bsq_loss) / 2
```

> **训练损失 vs 验证损失的差异**：训练时使用 `z_pre`（仅 s1 重建）和 `z`（完整重建）的 MSE 之和作为重建损失。但验证时，源码（`train_tokenizer.py` 第 180-182 行）只计算 `MSE(z, batch_x)`——即仅用细粒度重建的 MSE 作为验证指标。这意味着训练损失和验证损失不完全可比：验证损失通常低于训练中的总重建损失，因为它只衡量了其中一部分。如果你需要训练-验证损失严格可比，可以修改验证逻辑加入 `z_pre` 项。

---

## 预测模型微调

### 启动训练

```bash
torchrun --standalone --nproc_per_node=2 finetune/train_predictor.py
```

### 前置条件

预测模型微调**依赖微调后的分词器**。确保 `config.finetuned_tokenizer_path` 指向分词器微调阶段保存的最佳模型。

### 训练流程

```
1. 加载微调后的分词器（冻结，不参与训练）
   └─ tokenizer.eval()

2. 加载预训练预测模型
   └─ Kronos.from_pretrained(pretrained_predictor_path)

3. 训练循环
   └─ 在线编码：tokenizer.encode(batch_x, half=True) → (s1, s2)
   └─ 自回归预测：model(s1[:, :-1], s2[:, :-1], stamp[:, :-1])
   └─ 损失：CE(s1_logits, s1_targets) + CE(s2_logits, s2_targets)
   └─ 优化器：AdamW + OneCycleLR
   └─ 梯度裁剪：max_norm=3.0

4. 验证与保存
   └─ 每个 epoch 计算验证交叉熵
   └─ 保存最佳模型
```

### 关键细节

在预测模型微调中，分词器**不参与梯度计算**：

为什么冻结分词器？微调预测模型时，分词器的令牌分布需要保持稳定。如果分词器同时更新，令牌空间的语义会发生漂移，导致预测模型学到的令牌映射失效——类似于改变了字典中单词的含义后再用旧知识阅读。冻结分词器确保预测模型在一个固定的令牌空间中优化。

```python
with torch.no_grad():
    token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
```

输入格式采用**位移技巧**（shifted tokens）：

```python
# 输入：去掉最后一个令牌的序列
token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]

# 目标：去掉第一个令牌的序列（即下一个令牌）
token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
```

---

## 数据集：QlibDataset

`QlibDataset`（`finetune/dataset.py`）封装了 Qlib 数据的加载和采样逻辑：

### 数据格式

数据存储在 pickle 文件中，以股票代码为键，每个值为一个包含 `datetime` 索引和 OHLCV 列的 DataFrame。

### 滑动窗口采样

```python
window = lookback_window + predict_window + 1  # 总窗口大小

# 训练时随机采样
for epoch in epochs:
    train_dataset.set_epoch_seed(epoch * 10000 + rank)
    for batch in dataloader:
        # batch 包含随机位置的一个窗口数据
```

### 实例级标准化

每个样本独立计算均值和标准差：

```python
x = win_df[feature_list].values
x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
x = (x - x_mean) / (x_std + 1e-5)
x = np.clip(x, -clip, clip)
```

---

## 分布式训练

Qlib 微调流水线使用 PyTorch DDP（DistributedDataParallel）进行多 GPU 训练：

```bash
# 2 卡训练
torchrun --standalone --nproc_per_node=2 finetune/train_tokenizer.py

# 4 卡训练
torchrun --standalone --nproc_per_node=4 finetune/train_predictor.py
```

DDP 相关的实现细节：

- 使用 `DistributedSampler` 确保每个 GPU 看到不同的数据子集
- 验证损失通过 `dist.all_reduce` 在所有进程间聚合
- 模型保存只在 rank 0 进程执行
- 每轮结束后调用 `dist.barrier()` 同步所有进程

---

## 完整微调流程

```bash
# 1. 准备数据（首次需要）
python finetune/qlib_data_preprocess.py

# 2. 微调分词器
torchrun --standalone --nproc_per_node=2 finetune/train_tokenizer.py

# 3. 微调预测模型
torchrun --standalone --nproc_per_node=2 finetune/train_predictor.py

# 4. 验证微调效果
python finetune/qlib_test.py
```

---

## 如何判断微调效果

### 验证损失的期望范围

- **分词器微调**：验证 MSE 通常在 0.01-0.1 之间。初始值可能较高（0.5+），前 5-10 个 epoch 应快速下降，之后缓慢收敛。
- **预测模型微调**：验证交叉熵（CE）通常在 3.0-6.0 之间（取决于词表大小）。随着训练推进，CE 应稳步下降。

### 如何与预训练模型对比

建议在微调前先用预训练模型跑一遍 `qlib_test.py`，记录基线指标。微调完成后再次运行，对比以下维度：

- **重建误差**：分词器微调后，验证 MSE 是否低于预训练分词器
- **预测准确率**：预测模型微调后，CE 是否低于预训练模型
- **下游任务表现**：预测结果与真实价格的 MAE / 方向准确率是否提升

### 过拟合的判断标准

以下信号表明可能出现过拟合，需要调整策略：

- 训练损失持续下降，但验证损失在多个 epoch 内不再降低甚至回升
- 验证损失在早期 epoch 就达到最优，之后长期不改善
- 保存的 best_model 来自很早的 epoch（如前 5 轮），而训练总共跑了 30 轮

应对方法：减小学习率、增大 `accumulation_steps`、减少 `epochs`，或增加数据量。

### 微调效果的系统性诊断框架

当微调效果不理想时，按以下优先级逐一排查：

```
微调效果不理想？
│
├─ 1. 数据层检查
│   ├─ 数据量是否足够？（建议至少几千条 K 线）
│   ├─ 数据质量如何？（NaN、零值、异常值比例）
│   ├─ 训练/验证划分是否合理？（验证集太小会导致指标波动大）
│   └─ 标准化后数据分布是否合理？（大部分值在 [-5, 5] 内）
│
├─ 2. 训练配置检查
│   ├─ 学习率是否合适？（tokenizer: ~2e-4, predictor: ~4e-5）
│   ├─ batch_size 是否太小？（太小导致梯度估计不稳定）
│   ├─ 是否从预训练权重开始？（`pre_trained` 应为 True）
│   └─ 梯度裁剪是否生效？（max_norm=2.0 for tokenizer, 3.0 for predictor）
│
├─ 3. 流程层检查
│   ├─ 分词器微调是否在预测模型微调之前完成？
│   ├─ 预测模型微调是否加载了微调后的分词器（而非预训练分词器）？
│   └─ 分词器在预测模型微调中是否处于 eval 模式且冻结？
│
└─ 4. 评估方法检查
    ├─ 评估指标是否与训练目标一致？
    ├─ 是否在足够多的样本上评估？
    └─ 是否考虑了采样的随机性？（多次运行取平均）
```

---

## 常见误区

在动手微调之前，澄清几个容易踩坑的认知误区。

### 误区一：微调一定能提升效果

**事实**：微调并非万能药，效果取决于数据质量和目标任务与预训练数据的差距。

- 如果目标市场的数据特征与预训练数据相近（例如同为常规股票市场的日频数据），微调带来的增益可能非常有限，甚至因过拟合而劣于预训练模型。
- 微调需要足够的数据支撑。当数据量不足以覆盖模型的参数空间时，模型容易记住训练集中的噪声，导致泛化能力下降。
- **建议**：微调前先跑一遍预训练模型的基线指标。只有当基线明显低于预期、且有足够的高质量数据时，微调才有意义。

### 误区二：分词器和预测模型可以同时训练

**事实**：两阶段必须严格解耦，不能同时训练。

- 分词器决定了令牌空间的语义结构，预测模型在这个空间中学习令牌序列的规律。如果同时更新两者，令牌空间的语义会持续漂移，预测模型永远无法收敛到一个稳定的解。
- 这就好比一边改字典一边背单词——单词的含义不断变化，记忆自然无法建立。
- **正确做法**：先完成分词器微调并保存，再在冻结分词器的前提下微调预测模型。在预测模型微调代码中，分词器始终处于 `eval()` 模式并被 `torch.no_grad()` 包裹。

### 误区三：微调数据越多越好

**事实**：数据质量远比数量重要，盲目增加数据量可能适得其反。

- 低质量数据（包含大量缺失值、异常值、停牌日填充的零值）会引入噪声，让模型学到错误的规律。一千条干净数据的微调效果，往往好过一万条充满噪声的数据。
- 数据分布也很关键：如果训练数据集中在某个特定行情阶段（如持续牛市），模型可能只学会了单边行情的规律，面对震荡市或熊市时表现反而更差。
- **建议**：在增加数据量之前，先检查数据的覆盖性（是否包含上涨、下跌、震荡等不同行情）和干净程度（NaN 比例、零值比例）。宁可花时间清洗数据，也不要盲目堆量。

---

## 常见问题

### Q: 单 GPU 可以运行吗？

**A**: 当前 Qlib 微调脚本要求通过 `torchrun` 启动（检查 `WORLD_SIZE` 环境变量）。单 GPU 可以设置 `--nproc_per_node=1`：

```bash
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```

### Q: WORLD_SIZE 未设置的报错？

**A**: 如果直接运行 `python finetune/train_tokenizer.py`，会出现 `WORLD_SIZE` 环境变量未设置的报错。这是因为脚本使用 `torch.distributed` 初始化，必须通过 `torchrun` 启动。解决方法是始终使用 `torchrun` 命令运行训练脚本。

### Q: OOM（显存不足）错误？

**A**: 显存不足时可以尝试以下方案：

1. 减小 `batch_size`（默认 50，可降至 16 或 32）
2. 增大 `accumulation_steps` 以补偿 batch_size 缩小带来的梯度估计偏差
3. 使用更小的模型（如 `Kronos-small` 替代 `Kronos-base`）
4. 减小 `lookback_window`（默认 90，可降至 60）

### Q: Qlib 数据预处理（qlib_data_preprocess.py）报错？

**A**: 数据预处理阶段最常见的错误及解决方案：

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `FileNotFoundError: ~/.qlib/qlib_data/cn_data` | Qlib 数据未下载 | 先运行 `python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn` |
| `KeyError: 'vol'` / `KeyError: 'amt'` | Qlib 数据版本不匹配 | 检查 Qlib 版本是否兼容，确认数据包含 `$open/$close/$volume/$turn` 等字段 |
| pickle 文件为空或只有几百 KB | 股票池+时间范围内无有效数据 | 检查 `train_time_range` / `val_time_range` 是否与下载数据的时间范围重叠 |
| `RuntimeError: Can't open file` | 路径权限问题 | 确保对 `dataset_path` 有写入权限 |

> **空 pickle 文件的排查步骤**：这是最常见但最容易被忽略的问题。当 `qlib_data_preprocess.py` 运行完成但生成的 pickle 文件极小（几 KB 到几百 KB）时，说明没有股票通过了数据充分性过滤（`len(symbol_df) < lookback_window + predict_window + 1`）。按以下步骤排查：
>
> 1. **确认时间范围重叠**：打开 `config.py`，检查 `train_time_range` / `val_time_range` 的起止日期是否在你下载的 Qlib 数据覆盖范围内。Qlib 默认下载的 A 股日频数据通常覆盖 2007 年至最近更新日期。如果你的时间范围超出了数据范围，预处理脚本不会报错，但结果是空的。
> 2. **检查股票池与数据源的匹配**：确保 `instrument` 参数（如 `csi300`）与数据源对应。如果下载的是 `cn_data`，使用 `csi300`/`csi800`/`csi1000` 都是合理的。
> 3. **验证 Qlib 数据完整性**：在 Python 中运行以下代码，确认数据可以正常加载：
>
> ```python
> import qlib
> from qlib.data import D
> qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
> instruments = D.instruments("csi300")
> print(len(list(D.list_instruments(instruments))))  # 应输出约 300
> ```
>
> 如果输出为 0 或报错，说明 Qlib 数据本身有问题，需要重新下载。
>
> 4. **临时调大时间范围测试**：将 `train_time_range` 设为 `["2015-01-01", "2022-12-31"]` 等较宽的范围，再次运行预处理。如果此时文件变大了，说明原来的时间范围确实与数据不匹配。

### Q: 学习率调度器（OneCycleLR）的参数含义？

**A**: 两个训练脚本均使用 `OneCycleLR` 调度器，且参数完全一致（`pct_start=0.03, div_factor=10`）。具体位置：`train_tokenizer.py` 约第 104-111 行，`train_predictor.py` 约第 77-81 行。

- **pct_start=0.03**：前 3% 的训练步用于 warmup（从 `max_lr / 10` 线性上升到 `max_lr`），之后余弦退火到接近 0
- **div_factor=10**：初始学习率为 `max_lr / 10`，即分词器训练从 2e-5 开始，预测模型从 4e-6 开始

对于 30 epoch 的训练，warmup 阶段约占第 1 个 epoch 的前 3%（约几十个 batch），之后学习率快速下降。如果你发现训练初期损失剧烈震荡，可以增大 `div_factor`（如 25）来降低初始学习率。

### Q: 分词器微调和预测模型微调哪个更重要？

**A**: 取决于数据与预训练数据的差异程度。如果市场特征与预训练数据差异较大（如使用特殊指标），分词器微调更重要。如果只是适应新市场的规律，预测模型微调通常足够。

### Q: 如何使用 Comet ML 跟踪实验？

**A**: 在 `config.py` 中设置 `use_comet = True`，并填入你的 Comet API Key。训练过程会自动记录损失曲线、学习率等指标。

---

## 动手练习

### 练习 1：修改股票池观察数据集变化

修改 `finetune/config.py` 中 `Config` 类的 `instrument` 参数：

```python
config.instrument = "csi800"  # 原值为 "csi300"
```

然后重新运行数据预处理：

```bash
python finetune/qlib_data_preprocess.py
```

**验证方法**：观察预处理输出的 pickle 文件大小——csi800 包含约 800 只股票，数据量应明显大于 csi300 的 300 只。

### 练习 2：单 GPU 微调分词器

使用单 GPU 运行分词器微调，观察训练日志中的损失曲线：

```bash
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```

**验证方法**：检查 `save_path/checkpoints/best_model/` 目录下是否生成了模型文件，并确认验证 MSE 在训练过程中逐步下降。

### 练习 3：阅读源码理解损失计算

打开 `finetune/train_tokenizer.py`，找到损失计算相关的代码（约第 130-160 行），回答以下问题：

1. `recon_loss_pre` 和 `recon_loss_all` 分别使用了 `z_pre` 和 `z` 中的哪一个？它们分别对应粗粒度（仅 s1）和细粒度（s1+s2）的哪一级重建？
2. 验证阶段（约第 180-182 行）只计算了哪个重建损失？为什么训练损失和验证损失不可直接比较？

**验证方法**：对照本文"损失函数"一节中的代码片段确认你的理解。关键点是训练损失 = `(recon_loss_pre + recon_loss_all + bsq_loss) / 2`，而验证损失只用 `MSE(z, batch_x)`。

---

## 自测清单

- [ ] 能解释为什么分词器微调必须在预测模型微调之前完成
- [ ] 知道如何通过 `torchrun --nproc_per_node=N` 控制使用的 GPU 数量
- [ ] 能说出分词器微调和预测模型微调各自的损失函数组成
- [ ] 遇到 OOM 错误时知道至少两种应对策略
- [ ] 能根据验证损失曲线判断是否出现过拟合

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [CSV 微调指南](02-finetune-csv.md) | ⭐⭐⭐ | 通用数据的微调方法 |
| [源码走读](../architecture/04-source-code-walkthrough.md) | ⭐⭐⭐⭐ | 深入理解训练代码 |

## 相关文档

- **前置**：[KronosTokenizer 详解](../core-concepts/02-tokenizer.md) — 理解 BSQ 分词原理
- **前置**：[KronosPredictor 使用指南](../core-concepts/04-predictor.md) — 理解自回归推理流程
- **并行**：[CSV 微调指南](02-finetune-csv.md) — 通用数据的轻量微调方案
- **进阶**：[源码走读](../architecture/04-source-code-walkthrough.md) — 深入训练代码实现
- **实战**：[A 股市场预测实战](04-cn-markets.md) — 微调后用于 A 股预测

