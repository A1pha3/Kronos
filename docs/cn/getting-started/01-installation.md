# 安装与环境配置 ⭐

> **目标读者**：第一次接触 Kronos 的新手
> **预计时间**：10 分钟
> **前置要求**：具备基本的命令行操作能力

> **三步安装摘要**：(1) `git clone https://github.com/shiyu-coder/Kronos.git && cd Kronos` (2) `pip install -r requirements.txt` (3) 运行 `python -c "from model import KronosPredictor"` 验证。以下展开详细步骤。

---

## 学习目标

完成本章后，你需要做到：

- [ ] 在本地环境中成功导入 `KronosPredictor`
- [ ] 知道 `torch`、`pandas`、`einops` 等核心依赖分别承担什么职责
- [ ] 通过验证脚本确认安装无误，并清楚自己的计算设备类型

---

## Kronos 是什么

Kronos 是首个面向金融 K 线数据的开源基础模型，被 AAAI 2026 接收。它把 K 线预测转化为一个"语言建模"问题：先将 OHLCV 数据离散化为令牌序列，再由 Transformer 自回归地预测未来令牌。模型在 45+ 个全球交易所数据上完成预训练，加载权重即可使用。

> **一句话理解**：输入历史行情，输出未来走势——和 GPT 生成文本的逻辑一样。

> **分词器搭配**：`Kronos-mini` 需搭配 `Kronos-Tokenizer-2k`（上下文 2048）；其余模型搭配 `Kronos-Tokenizer-base`（上下文 512）。详见 [模型对比与选型](../advanced/07-model-comparison.md)。

架构原理的完整介绍，参见 [项目总览与核心概念](../core-concepts/01-overview.md)。

---

## 环境要求

### 硬件要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 内存 | 8 GB | 16 GB 及以上 |
| GPU | 不需要（CPU 可运行） | NVIDIA GPU（显存 4 GB+）或 Apple Silicon Mac |
| 磁盘 | 2 GB（模型文件 + 依赖） | 5 GB |

Kronos 支持三种计算设备，`KronosPredictor` 会按优先级自动检测，无需手动配置：

- **CPU**——所有平台可用，功能完整但速度较慢
- **CUDA GPU**——NVIDIA 显卡，推理速度最快
- **MPS**——Apple Silicon Mac（M1/M2/M3/M4 系列），利用 GPU 加速

### 软件要求

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.10 | 部分依赖（如 `pandas==2.2.2`）对 Python 版本有要求；3.10 及以上可确保兼容性 |
| pip | 最新版 | 包管理器 |

---

## 安装步骤

### 虚拟环境准备

Python 的包管理器 pip 默认全局安装，不同项目对同一包的版本要求可能冲突。虚拟环境为每个项目创建独立的包目录，避免互相干扰。下面的步骤在新建虚拟环境中进行：

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
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
```

### 步骤 2：安装核心依赖

```bash
pip install -r requirements.txt
```

安装完成后，以下依赖会被一并安装（版本号来自 `requirements.txt` 中的锁定值，`torch` 为最低版本要求）：

> **关于 requirements.txt 的结构**：文件中 `pandas` 出现两次——首次无版本约束（兼容性声明），末尾 `pandas==2.2.2` 是实际生效的锁定版本。`numpy` 只出现一次且未锁定版本。pip 以最后一条匹配记录为准。

| 包名 | 版本 | 作用 |
|------|------|------|
| `torch` | ≥ 2.0.0 | 深度学习框架，提供张量计算与模型推理 |
| `numpy` | — | 数值计算，数据预处理与后处理 |
| `pandas` | 2.2.2 | 数据表格处理，输入输出数据管理 |
| `einops` | 0.8.1 | 张量维度操作，用于模型内部的 reshape 操作 |
| `huggingface_hub` | 0.33.1 | 从 HuggingFace Hub 下载预训练模型 |
| `safetensors` | 0.6.2 | 安全的模型权重文件格式 |
| `matplotlib` | 3.9.3 | 结果可视化（仅在示例脚本中使用） |
| `tqdm` | 4.67.1 | 进度条显示 |

### 步骤 3（可选）：安装 Web UI 依赖

Web 界面需要额外的 Flask、Flask-CORS 和 Plotly 依赖：

```bash
pip install -r webui/requirements.txt
```

### 步骤 4（可选）：安装微调依赖

微调依赖取决于你的数据来源。

**Qlib 微调**（中国 A 股数据）：
```bash
pip install qlib comet-ml
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
torchrun --standalone --nproc_per_node=1 finetune/train_predictor.py
```

**A 股市场预测脚本**：
```bash
pip install akshare
```

---

## 验证安装

以下两段脚本分别检查依赖和模型加载。

**检查核心依赖**：

```python
import torch
import numpy as np
import pandas as pd

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"MPS 可用: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
print(f"NumPy 版本: {np.__version__}")
print(f"Pandas 版本: {pd.__version__}")
```

有 NVIDIA GPU 且安装了 CUDA 版 PyTorch 时，`CUDA 可用` 显示 `True`；仅有 CPU 时显示 `False`，Kronos 的所有功能不受影响。GPU 可以将推理速度提升 2-10 倍，CPU 和 GPU 产生的预测结果在数学上完全一致。

**检查模型加载**：

```python
from model import Kronos, KronosTokenizer, KronosPredictor

print("Kronos 模块导入成功！")
print(f"可用类: KronosTokenizer, Kronos, KronosPredictor")
```

> 注意：导入模块时不会自动下载模型。权重在首次调用 `from_pretrained()` 时从 HuggingFace Hub 下载（`Kronos-small` 约 50 MB），下载后被缓存，后续加载无需重复下载。

---

## 自测检查

三项全过即可进入 [快速开始](02-quickstart.md)：

- [ ] `python -c "import torch; print(torch.__version__)"` 输出版本号
- [ ] `python -c "from model import KronosPredictor"` 无报错
- [ ] 设备类型（CPU / CUDA / MPS）与验证脚本输出一致

**排错参考**：`import torch` 报错说明 PyTorch 未安装或版本不匹配，参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 重新安装；`from model import ...` 报错通常是因为工作目录不在仓库根目录；设备检测异常详见 [故障排查](../references/troubleshooting.md)。

---

## 动手练习

### 练习 1：从零搭建虚拟环境

在新建的虚拟环境中完成全部安装步骤：

```bash
# 1. 创建并激活虚拟环境
python -m venv kronos-env
source kronos-env/bin/activate

# 2. 克隆仓库并安装依赖
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
pip install -r requirements.txt

# 3. 验证
python -c "import torch, numpy, pandas; print('OK')"
```

**验证方法**：在虚拟环境中运行 `python -c "from model import KronosPredictor"` 无报错即为成功。完成后用 `deactivate` 退出。

### 练习 2：测量模型加载耗时

用 `time` 模块测量 `from_pretrained()` 在不同设备上的加载时间：

```python
import time, torch

device = "cuda:0" if torch.cuda.is_available() else (
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
)

from model import Kronos, KronosTokenizer

start = time.time()
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
elapsed = time.time() - start
print(f"设备: {device}，加载耗时: {elapsed:.2f}s")
```

**验证方法**：首次运行包含模型下载时间，后续运行只反映磁盘 I/O。GPU 加载速度未必更快（瓶颈在磁盘），但推理阶段会有显著差异——可以在快速开始章节中对比实际感受。

---

## 常见问题

### Q: pip install 报错，提示 torch 安装失败？

PyTorch 需要根据操作系统和 CUDA 版本选择对应的安装包。到 [PyTorch 官网](https://pytorch.org/get-started/locally/) 找到适合你系统的安装命令，先单独安装 PyTorch，再装其余依赖。详见 [故障排查](../references/troubleshooting.md)。

### Q: 模型下载速度慢或超时？

预训练模型托管在 HuggingFace Hub 上。网络不稳定时可以：（1）设置 `HF_ENDPOINT` 环境变量使用镜像站；（2）手动下载模型文件到本地后用本地路径加载。下载地址与分词器搭配参见 [常见问题](../references/faq.md) 和 [模型对比与选型](../advanced/07-model-comparison.md)。

### Q: Mac 上能否使用 GPU 加速？

Apple Silicon Mac（M1/M2/M3/M4）支持 MPS 后端，`KronosPredictor` 会自动检测并使用。Intel Mac 只能用 CPU。更多设备与性能问题见 [故障排查](../references/troubleshooting.md)。

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [快速开始：第一个预测](02-quickstart.md) | ⭐ | 10 分钟完成第一次 K 线预测 |
| [数据准备指南](03-data-preparation.md) | ⭐ | 了解数据格式要求 |
