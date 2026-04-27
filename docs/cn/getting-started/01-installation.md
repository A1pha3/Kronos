# 安装与环境配置 ⭐

> **目标读者**：第一次接触 Kronos 的新手
> **预计时间**：10 分钟
> **前置要求**：具备基本的命令行操作能力

> **三步安装摘要**：① `git clone https://github.com/shiyu-coder/Kronos.git && cd Kronos` ② `pip install -r requirements.txt` ③ 运行 `python -c "from model import KronosPredictor"` 验证。以下展开详细步骤。

---

## 学习目标

装完之后你应该能做这三件事：

- [ ] 本地环境正常导入 Kronos 模块
- [ ] 说清楚 torch、pandas、einops 等核心依赖各自负责什么
- [ ] 运行验证脚本确认安装无误，判断自己的设备类型

---

## Kronos 是什么

Kronos 是首个面向金融 K 线数据的开源基础模型，被 AAAI 2026 接收。它将 K 线预测转化为"语言建模"问题：先把 OHLCV 数据离散化为令牌，再用 Transformer 预测未来的令牌序列。在 45+ 个全球交易所数据上预训练，开箱即用。

> **一句话理解**：用 Kronos 预测 K 线，就像用 GPT 生成文本一样——输入历史行情，输出未来走势。

> **分词器搭配**：`Kronos-mini` 使用 `Kronos-Tokenizer-2k`（上下文 2048）；其余模型使用 `Kronos-Tokenizer-base`（上下文 512）。详见 [模型对比与选型](../advanced/07-model-comparison.md)。

架构原理的完整介绍，参见 [项目总览与核心概念](../core-concepts/01-overview.md)。

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
| Python | ≥ 3.10 | 部分依赖（如 `pandas==2.2.2`）对 Python 版本有要求；3.10 及以上可确保兼容性 |
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
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
```

### 步骤 2：安装核心依赖

```bash
pip install -r requirements.txt
```

核心依赖清单与作用说明（版本号为 `requirements.txt` 中的锁定值，`torch` 为最低版本要求）：

> **关于 requirements.txt 的结构**：该文件中 `pandas` 出现两次——首次无版本约束（作为兼容性声明），末尾的 `pandas==2.2.2` 为实际生效的锁定版本。`numpy` 只出现一次且不锁定版本。安装时 pip 以最后一条匹配的记录为准。

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

## 自测检查

跑一遍下面的命令，三项全过就可以进入 [快速开始](02-quickstart.md) 了：

- [ ] `python -c "import torch; print(torch.__version__)"` 正常输出版本号
- [ ] `python -c "from model import KronosPredictor"` 无报错
- [ ] 计算设备类型（CPU / CUDA / MPS）与验证脚本输出一致

如果某一项没过：`import torch` 报错说明 PyTorch 未安装或版本不匹配，参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 重新安装；`from model import ...` 报错通常是因为当前目录不是仓库根目录；设备检测不符预期的话，详见 [常见错误排查](../references/troubleshooting.md)。

---

## 动手练习

### 练习 1：在虚拟环境中完成完整的环境搭建

从零开始，在一个新建的虚拟环境中完成全部安装步骤：

```bash
# 1. 创建并激活虚拟环境
python -m venv kronos-env
source kronos-env/bin/activate

# 2. 克隆仓库并安装依赖
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
pip install -r requirements.txt

# 3. 运行验证脚本（复制上文"快速验证脚本"中的代码）
python -c "import torch, numpy, pandas; print('OK')"
```

**验证方法**：在虚拟环境中运行 `python -c "from model import KronosPredictor"` 无报错，说明环境搭建成功。完成后可通过 `deactivate` 退出虚拟环境。

> **为什么要用虚拟环境？** Python 的包管理器 pip 是全局安装的——如果你同时维护两个项目，一个需要 `numpy>=2.0`，另一个依赖 `numpy<2.0`，直接安装会互相覆盖。虚拟环境为每个项目创建独立的包目录，从根本上杜绝这类冲突。

### 练习 2：对比 CPU 和 GPU 环境下的模型加载速度

如果你有 GPU（CUDA 或 MPS），可以用 `time` 模块测量不同设备上 `from_pretrained()` 的耗时差异：

```python
import time, torch

device = "cuda:0" if torch.cuda.is_available() else (
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
)

from model import Kronos, KronosTokenizer

# 测量模型加载 + 推理准备耗时
start = time.time()
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
elapsed = time.time() - start
print(f"设备: {device}，加载耗时: {elapsed:.2f}s")
```

**验证方法**：分别记录 CPU 和 GPU（如果可用）的加载时间。首次运行包含模型下载时间，后续运行只反映纯粹的加载耗时。GPU 加载通常不会更快（瓶颈在磁盘 I/O），但后续推理速度会有显著差异。

---

## 常见问题

### Q: pip install 报错，提示 torch 安装失败？

PyTorch 需要根据操作系统和 CUDA 版本选择对应的安装包。先到 [PyTorch 官网](https://pytorch.org/get-started/locally/) 拿到适合你系统的安装命令，单独安装 PyTorch 后再装其余依赖。详见 [常见问题与故障排查](../references/troubleshooting.md)。

### Q: 模型下载速度慢或超时？

预训练模型托管在 HuggingFace Hub 上，网络不稳定时可以：（1）设置 `HF_ENDPOINT` 环境变量使用镜像站；（2）手动下载模型文件到本地后用本地路径加载。各模型的下载地址与分词器搭配见 [常见问题](../references/faq.md) 和 [模型对比与选型](../advanced/07-model-comparison.md)。

### Q: Mac 上能否使用 GPU 加速？

Apple Silicon Mac（M1/M2/M3/M4）支持 MPS 后端，KronosPredictor 会自动检测并使用。Intel Mac 只能用 CPU。更多设备与性能问题见 [故障排查](../references/troubleshooting.md)。

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [快速开始：第一个预测](02-quickstart.md) | ⭐ | 10 分钟完成第一次 K 线预测 |
| [数据准备指南](03-data-preparation.md) | ⭐ | 了解数据格式要求 |
