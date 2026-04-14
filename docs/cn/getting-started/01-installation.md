# 安装与环境配置 ⭐

> **目标读者**：第一次接触 Kronos 的新手
> **预计时间**：10 分钟
> **前置要求**：具备基本的命令行操作能力

---

## 学习目标

完成本教程后，你将能够：

- [ ] 在本地搭建 Kronos 运行环境
- [ ] 理解各依赖库的作用
- [ ] 验证环境是否配置成功

---

## Kronos 是什么

Kronos 是首个面向金融 K 线数据的开源基础模型，被 AAAI 2026 接收。它将 K 线预测转化为"语言建模"问题：先把 OHLCV 数据离散化为令牌，再用 Transformer 预测未来的令牌序列。Kronos 专为金融蜡烛图设计，在 45+ 个全球交易所数据上预训练，开箱即用。关于其架构原理的完整介绍，参见 [项目总览与核心概念](../core-concepts/01-overview.md)。

### 与传统方法的区别

| 维度 | 传统模型（LSTM、ARIMA） | Kronos |
|------|------------------------|--------|
| 建模方式 | 直接在连续数值空间做回归 | 先离散化为令牌，再做分类预测 |
| 多步预测 | 误差逐步累积 | 在离散令牌空间更稳定 |
| 泛化能力 | 通常需要针对单一序列训练 | 在 45+ 个全球交易所数据上预训练，开箱即用 |
| 生成策略 | 确定性输出 | 支持温度采样、核采样等可控随机策略 |
| 适用范围 | 需要为每个市场单独训练 | 统一模型适配股票、期货、加密货币等 |
| 时间感知 | 通常无显式时间编码 | 内置 TemporalEmbedding 捕捉周期性 |

> **分词器选择提示**：不同规模的 Kronos 模型搭配不同的分词器。`Kronos-mini` 使用专用分词器 `Kronos-Tokenizer-2k`，支持最长 2048 的上下文长度；其余模型（`Kronos-small`、`Kronos-base`、`Kronos-large`）均使用 `Kronos-Tokenizer-base`，上下文长度为 512。详见 [模型对比与选型](../advanced/07-model-comparison.md)。

---

## 环境要求

### 硬件要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 内存 | 8 GB | 16 GB 及以上 |
| GPU | 不需要（CPU 可运行） | NVIDIA GPU（显存 4 GB+）或 Apple Silicon Mac |
| 磁盘 | 2 GB（模型文件 + 依赖） | 5 GB |

Kronos 支持三种计算设备：

- **CPU**：所有平台均可，速度较慢但功能完整
- **CUDA GPU**：NVIDIA 显卡，推理速度最快
- **MPS**：Apple Silicon Mac（M1/M2/M3/M4 系列），利用 GPU 加速

KronosPredictor 会**自动检测**可用的计算设备，无需手动配置。

### 软件要求

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.10 | 需要支持类型注解等现代语法（代码库中大量使用函数参数的类型注解） |
| pip | 最新版 | 包管理器 |

---

## 安装步骤

### 虚拟环境准备

建议在安装前创建一个独立的虚拟环境，避免与系统或其他项目的包产生冲突：

```bash
# 方式一：使用 venv（Python 内置）
python -m venv kronos-env
source kronos-env/bin/activate   # Linux/macOS
# kronos-env\Scripts\activate    # Windows

# 方式二：使用 conda
conda create -n kronos-env python=3.10
conda activate kronos-env
```

### 步骤 1：克隆仓库

```bash
git clone https://github.com/NeoQuasar/Kronos.git
cd Kronos
```

### 步骤 2：安装核心依赖

Kronos 的核心依赖非常精简，只需要以下包：

```bash
pip install -r requirements.txt
```

核心依赖清单与作用说明：

| 包名 | 版本要求 | 作用 |
|------|----------|------|
| `torch` | ≥ 2.0.0 | 深度学习框架，提供张量计算与模型推理 |
| `numpy` | — | 数值计算，数据预处理与后处理 |
| `pandas` | 2.2.2 | 数据表格处理，输入输出数据管理 |
| `einops` | 0.8.1 | 张量维度操作，用于模型内部的 reshape 操作 |
| `huggingface_hub` | 0.33.1 | 从 HuggingFace Hub 下载预训练模型 |
| `safetensors` | 0.6.2 | 安全的模型权重文件格式 |
| `matplotlib` | 3.9.3 | 结果可视化（仅在示例脚本中使用） |
| `tqdm` | 4.67.1 | 进度条显示 |

### 步骤 3：（可选）安装 Web UI 依赖

如果你计划使用 Web 界面进行预测，还需安装额外的依赖：

```bash
pip install -r webui/requirements.txt
```

额外依赖包括 Flask（Web 框架）、Flask-CORS（跨域支持）和 Plotly（交互式图表）。

### 步骤 4：（可选）安装微调依赖

如果需要进行模型微调，根据你的数据来源安装对应的依赖：

**Qlib 微调**（中国 A 股数据）：
```bash
pip install qlib comet-ml
```

**A 股市场预测脚本**：
```bash
pip install akshare
```

---

## 验证安装

### 快速验证脚本

运行以下 Python 代码，验证所有核心依赖是否正确安装：

```python
import torch
import numpy as np
import pandas as pd
from huggingface_hub import PyTorchModelHubMixin

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"MPS 可用: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
print(f"NumPy 版本: {np.__version__}")
print(f"Pandas 版本: {pd.__version__}")
print("所有核心依赖安装成功！")
```

### 预期输出

如果你有 NVIDIA GPU 且安装了 CUDA 版本的 PyTorch：

```
PyTorch 版本: 2.x.x
CUDA 可用: True
MPS 可用: False
NumPy 版本: x.x.x
Pandas 版本: 2.2.2
所有核心依赖安装成功！
```

如果只有 CPU，`CUDA 可用` 会显示 `False`，这完全正常，Kronos 可以在 CPU 上运行所有功能。

**设备选择的实际影响**：使用 GPU 可以将推理速度提升 2-10 倍，但预测质量完全相同——Kronos 在 CPU 和 GPU 上产生一致的数学结果。对于初次体验，CPU 即可满足需求。

### 验证模型加载

```python
from model import Kronos, KronosTokenizer, KronosPredictor

print("Kronos 模块导入成功！")
print(f"可用类: KronosTokenizer, Kronos, KronosPredictor")
print("OK")
```

> 注意：首次导入时不会自动下载模型。模型会在你首次调用 `from_pretrained()` 时自动从 HuggingFace Hub 下载（如 `Kronos-small` 约 50 MB）。下载完成后权重会被缓存，后续加载无需重复下载。

---

## 常见问题

### Q: `pip install` 报错，提示 torch 安装失败？

**A**: PyTorch 需要根据你的操作系统和 CUDA 版本选择正确的安装包。访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合你系统的安装命令，先单独安装 PyTorch，再安装其他依赖。详见 [常见问题与故障排查](../references/troubleshooting.md)。

### Q: 模型下载速度慢或超时？

**A**: 预训练模型托管在 HuggingFace Hub 上。如果网络访问不稳定，可以：

1. 设置 `HF_ENDPOINT` 环境变量使用镜像站
2. 手动从 HuggingFace 下载模型文件到本地，然后使用本地路径加载

关于各模型的下载地址与分词器搭配，详见 [常见问题](../references/faq.md) 和 [模型对比与选型](../advanced/07-model-comparison.md)。

### Q: Mac 上能否使用 GPU 加速？

**A**: Apple Silicon Mac（M1/M2/M3/M4）支持 MPS 后端。KronosPredictor 会自动检测并使用 MPS 设备。Intel Mac 只能使用 CPU。更多设备与性能问题，详见 [常见问题与故障排查](../references/troubleshooting.md)。

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [快速开始：第一个预测](02-quickstart.md) | ⭐ | 10 分钟完成第一次 K 线预测 |
| [数据准备指南](03-data-preparation.md) | ⭐ | 了解数据格式要求 |

---
**文档元信息**
难度：⭐ | 类型：入门教程 | 预计阅读时间：10 分钟 | 最后更新：2026-04-11
