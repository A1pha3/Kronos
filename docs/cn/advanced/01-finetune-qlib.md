# Qlib 微调指南 ⭐⭐⭐

> **目标读者**：想使用中国 A 股数据微调 Kronos 的开发者
> **前置要求**：理解核心概念，熟悉 PyTorch 训练流程

### 学习目标

阅读本文后，你将能够：

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

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 30 | 训练轮数 |
| `batch_size` | 50 | 每 GPU 批量大小 |
| `tokenizer_learning_rate` | 2e-4 | 分词器学习率 |
| `predictor_learning_rate` | 4e-5 | 预测模型学习率 |
| `accumulation_steps` | 1 | 梯度累积步数 |
| `clip` | 5.0 | 数据裁剪范围 |

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
   └─ 优化器：AdamW + OneCycleLR
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
torchrun --standalone --nproc_per_node=2 train_tokenizer.py

# 4 卡训练
torchrun --standalone --nproc_per_node=4 train_predictor.py
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

---

## 自测清单

完成本指南后，检查你是否掌握了以下要点：

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

---
**文档元信息**
难度：⭐⭐⭐ | 类型：进阶指南 | 预计阅读时间：25 分钟
更新日期：2026-04-11
