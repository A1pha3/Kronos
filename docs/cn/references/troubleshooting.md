# 常见错误排查指南

> **目标读者**：在使用 Kronos 过程中遇到错误需要排查的用户
> **前置要求**：已完成[快速开始](../getting-started/02-quickstart.md)

---

## 目录

- [快速诊断：根据错误信息定位](#快速诊断根据错误信息定位)
- [内存类错误](#内存类错误)
- [数据类错误](#数据类错误)
- [张量维度错误](#张量维度错误)
- [模型加载错误](#模型加载错误)
- [训练类错误](#训练类错误)
- [预测结果类问题](#预测结果类问题)
- [MPS（Apple Silicon）相关问题](#mpsapple-silicon相关问题)
- [高级调试技巧](#高级调试技巧)

---

## 快速诊断：根据错误信息定位

看到报错后，先在下表中按关键词定位对应的排查步骤：

| 错误信息关键词 | 可能原因 | 跳转 |
|--------------|---------|------|
| `CUDA out of memory` | GPU 显存不足 | [OOM 错误](#runtimeerror-cuda-out-of-memory) |
| `MemoryError` | CPU 内存不足 | [CPU 内存不足](#memoryerrorcpu-内存不足) |
| `contains NaN` | 输入数据有缺失值 | [NaN 错误](#valueerror-input-dataframe-contains-nan-values) |
| `not found in DataFrame` | 缺少必填列 | [列缺失](#valueerror-price-columns-not-found-in-dataframe) |
| `shape mismatch` / `consistent historical lengths` | 批量预测长度不一致 | [维度错误](#runtimeerror-shape-mismatch) |
| `y_timestamp length should equal` | 时间戳长度不匹配 | [时间戳错误](#valueerror-y_timestamp-length-should-equal-pred_len) |
| `Model not found` / `OSError` | 模型下载失败 | [模型加载](#filenotfounderror--oserror-model-not-found) |
| `Error(s) in loading state_dict` | 权重与代码不匹配 | [权重加载](#runtimeerror-errors-in-loading-state_dict) |
| `NCCL error` | DDP 通信失败 | [NCCL 错误](#runtimeerror-nccl-error) |
| `WORLD_SIZE` 未设置 | 未使用 torchrun | [torchrun 错误](#world_size-环境变量未设置) |
| `MPS does not support` | Apple Silicon MPS 后端限制 | [MPS 错误](#mpsapple-silicon相关问题) |
| 训练损失不下降 | 超参数或数据问题 | [训练不收敛](#训练损失不下降) |
| 预测为直线 | 参数或数据问题 | [预测结果为一条直线](#预测结果为一条直线) |
| amount 列异常 | 有 volume 无 amount 时的自动推算 | [amount 异常](#valueerror-amount-列值异常有-volume-但-amount-被自动推算) |

---

## 按错误类型索引

如果已经知道问题类别，可通过下表直接跳转：

| 错误类型 | 常见场景 | 跳转 |
|---------|---------|------|
| OOM / 内存不足 | 预测、微调、批量推理 | [内存类错误](#内存类错误) |
| NaN 相关 | 数据准备、模型推理 | [数据类错误](#数据类错误) |
| 维度不匹配 | 自定义代码、批量预测 | [张量维度错误](#张量维度错误) |
| 模型加载失败 | 首次使用、网络问题 | [模型加载错误](#模型加载错误) |
| 训练不收敛 | 微调过程 | [训练类错误](#训练类错误) |
| 预测结果异常 | 参数设置、数据质量 | [预测结果类问题](#预测结果类问题) |
| MPS 不支持 | Apple Silicon Mac | [MPS 相关问题](#mpsapple-silicon相关问题) |

---

## 内存类错误

### RuntimeError: CUDA out of memory

**触发场景**：使用 GPU 推理或训练时，显存不足。

**错误信息示例**：

```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**排查步骤**：

1. **确认当前显存占用**：

```python
import torch
print(f"已分配显存: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
print(f"最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
```

2. **逐步降低资源使用**：

| 优化手段 | 操作 | 效果 | 代价 |
|---------|------|------|------|
| 减小历史窗口 | `lookback` 从 400 降至 200 | 显存占用近似线性减少 | 可用历史信息减少，预测可能欠佳 |
| 减少采样次数 | `sample_count` 从 5 降至 1 | 显存占用线性减少 | 预测稳定性下降 |
| 缩短预测步数 | `pred_len` 从 120 降至 60 | 总推理时间减少 | 预测范围缩短 |
| 使用更小模型 | `Kronos-small` 替代 `Kronos-base` | 显存占用显著减少 | 表达能力降低 |
| 分批处理 | 将大批量拆分为多个小批量 | 单批显存占用减少 | 总推理时间不变 |
| 使用 CPU | `device="cpu"` | 完全避免显存问题 | 推理速度下降 2-10 倍 |

3. **强制使用 CPU**：

```python
predictor = KronosPredictor(model, tokenizer, device="cpu")
```

### MemoryError（CPU 内存不足）

**排查步骤**：

- 减小 `lookback`（建议不低于 64）
- 使用 `Kronos-small` 或 `Kronos-mini`
- 避免同时持有多个 `predictor` 实例

---

## 数据类错误

### ValueError: Input DataFrame contains NaN values

**触发场景**：传入 `predict()` 的 DataFrame 中存在 NaN 值。

**错误信息**：

```
ValueError: Input DataFrame contains NaN values in price or volume columns.
```

**排查步骤**：

1. **定位 NaN 位置**：

```python
# 检查每列的 NaN 数量
nan_counts = df[['open', 'high', 'low', 'close', 'volume', 'amount']].isnull().sum()
print(nan_counts[nan_counts > 0])
```

2. **修复 NaN**：

```python
# 方法 1：前向填充（推荐）
df = df.ffill()

# 方法 2：删除包含 NaN 的行
df = df.dropna()

# 方法 3：用前一根 K 线的收盘价填充开盘价
df.loc[df['open'].isna(), 'open'] = df['close'].shift(1)
```

3. **验证修复结果**：

```python
assert not df[['open', 'high', 'low', 'close']].isnull().any().any(), "仍有 NaN"
```

### ValueError: Price columns not found in DataFrame

**触发场景**：DataFrame 中缺少必填列。

**排查步骤**：

```python
required_cols = ['open', 'high', 'low', 'close']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"缺少列：{missing}")
    # 检查列名是否大小写不同或使用了中文列名
    print(f"实际列名：{df.columns.tolist()}")
```

**常见原因**：

- CSV 列名为中文（如"开盘"而非 `open`）
- 列名大小写不匹配（如 `Open` 而非 `open`）
- 列名包含空格

**解决方法**：

```python
# 重命名列
df.rename(columns={
    "开盘": "open", "最高": "high",
    "最低": "low", "收盘": "close",
    "成交量": "volume", "成交额": "amount"
}, inplace=True)

# 或统一转小写
df.columns = df.columns.str.strip().str.lower()
```

### 预测结果中出现异常值（价格为 0 或极大值）

**排查步骤**：

1. 检查输入数据中是否有价格为 0 的记录（通常是停牌日）
2. 检查数据是否按时间升序排列
3. 检查是否有极端异常值（如价格为负数）

```python
# 检查异常值
for col in ['open', 'high', 'low', 'close']:
    zero_count = (df[col] == 0).sum()
    neg_count = (df[col] < 0).sum()
    if zero_count > 0 or neg_count > 0:
        print(f"{col}: {zero_count} 个零值, {neg_count} 个负值")
```

### amount 列自动推算的注意事项（有 volume 但无 amount）

**触发场景**：输入 DataFrame 有 `volume` 列但没有 `amount` 列时，KronosPredictor 自动推算 `amount = volume * mean(open, high, low, close)`（逐行算术均价，非成交量加权均价）。如果推算值与实际成交额差距大，可能影响预测质量。

**排查步骤**：

```python
# 检查推算 amount 与真实 amount 的偏差
if 'volume' in df.columns and 'amount' not in df.columns:
    estimated_amount = df['volume'] * df[['open', 'high', 'low', 'close']].mean(axis=1)
    print("amount 已自动推算（算术均价 x volume），非真实成交额")
    print("建议：如有真实 amount 数据，请手动添加到 DataFrame 中")
```

---

## 张量维度错误

### RuntimeError: shape mismatch

**触发场景**：批量预测时各序列长度不一致。

**错误信息**：

```
RuntimeError: shape mismatch: ... size ... vs ...
```

或

```
ValueError: Parallel prediction requires all series to have consistent historical lengths, got: [...]
```

**排查步骤**：

```python
# 检查各序列长度
for i, df in enumerate(df_list):
    print(f"序列 {i}: {len(df)} 行")
```

**解决方法**：

```python
# 方法 1：截断为相同长度（取最短的）
min_len = min(len(df) for df in df_list)
df_list = [df.iloc[:min_len] for df in df_list]

# 方法 2：分别调用 predict()
results = []
for df_i, x_ts_i, y_ts_i in zip(df_list, x_ts_list, y_ts_list):
    pred = predictor.predict(df=df_i, x_timestamp=x_ts_i, y_timestamp=y_ts_i, pred_len=pred_len)
    results.append(pred)
```

### ValueError: y_timestamp length should equal pred_len

**排查步骤**：

```python
# 确保 y_timestamp 的长度等于 pred_len
assert len(y_timestamp) == pred_len, f"y_timestamp 长度 {len(y_timestamp)} ≠ pred_len {pred_len}"
```

---

## 模型加载错误

### FileNotFoundError / OSError: Model not found

**触发场景**：从 HuggingFace Hub 下载模型失败。

**排查步骤**：

1. **检查网络连接**：确保能访问 `huggingface.co`

2. **使用镜像站**（中国大陆用户）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

3. **手动下载模型**：

```python
# 指定本地路径
tokenizer = KronosTokenizer.from_pretrained("/path/to/local/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("/path/to/local/Kronos-small")
```

4. **检查 HuggingFace 缓存**：

```python
from huggingface_hub import scan_cache_dir
cache = scan_cache_dir()
for repo in cache.repos:
    print(f"{repo.repo_id}: {repo.size_on_disk / 1024**2:.1f} MB")
```

### RuntimeError: Error(s) in loading state_dict

**触发场景**：模型权重与代码版本不匹配。

**排查步骤**：

```python
# 检查模型的 config
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
print(model.config)  # 查看实际配置参数
```

**常见原因**：

- 使用了错误版本的代码（模型结构已更改）
- 指定了错误的模型路径（如将分词器路径传给了 `Kronos.from_pretrained()`）

---

## 训练类错误

### 训练损失不下降

**排查清单**：

| 检查项 | 方法 | 期望 |
|--------|------|------|
| 数据质量 | 检查 NaN、零值、异常值 | 无异常 |
| 数据量 | 检查 CSV 行数 | 至少几千行 |
| 学习率 | 查看 config 中的 lr | tokenizer: ~2e-4, predictor: ~4e-5 |
| 预训练权重 | 检查 `pre_trained` 设置 | 应为 `true` |
| clip 值 | 查看标准化后数据范围 | 大部分值在 [-5, 5] 内 |

**快速诊断脚本**：

```python
import pandas as pd
import numpy as np

df = pd.read_csv("your_data.csv")
x = df[['open', 'high', 'low', 'close', 'volume', 'amount']].values.astype(np.float32)

# 检查数据分布
print(f"数据量: {x.shape}")
print(f"NaN 数量: {np.isnan(x).sum()}")
print(f"零值数量: {(x == 0).sum()}")
print(f"负值数量: {(x < 0).sum()}")

# 检查标准化后的分布
x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
x_norm = (x - x_mean) / (x_std + 1e-5)
x_clipped = np.clip(x_norm, -5, 5)  # KronosPredictor 内部会执行 clip
print(f"标准化后范围: [{x_norm.min():.2f}, {x_norm.max():.2f}]")
print(f"标准化后 >5 的比例: {(np.abs(x_norm) > 5).mean():.4f}")
print(f"clip 后范围: [{x_clipped.min():.2f}, {x_clipped.max():.2f}]")
```

> **注意**：KronosPredictor 的标准化流程是 `z-score -> clip(-5, 5)`。如果标准化后 `|x| > 5` 的比例超过 1%，说明数据中可能存在异常值需要先处理。

### DDP / torchrun 相关错误

#### RuntimeError: NCCL error

**排查步骤**：

1. 确认所有 GPU 可见：`nvidia-smi`
2. 减少进程数：`--nproc_per_node=1`
3. 检查 NCCL 后端：确保安装了 CUDA 版本的 PyTorch

#### WORLD_SIZE 环境变量未设置

**原因**：直接运行 `python train_tokenizer.py` 而非通过 `torchrun`。

**解决方法**：

```bash
# 必须使用 torchrun
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```

---

## 预测结果类问题

### 预测结果为一条直线

**可能原因与解决方法**（也可参考 [FAQ：预测结果每次不同](faq.md#q-预测结果每次都不同正常吗) 中的采样参数说明）：

| 原因 | 解决方法 | 为什么有效 |
|------|---------|-----------|
| `lookback` 过小（< 64） | 增大到至少 200 | 模型需要足够的历史数据来捕捉价格波动模式 |
| 温度 `T` 过低（如 0.01） | 增大到 0.8-1.2 | 极低温度使采样退化为确定性选择，失去多样性 |
| 数据本身波动极小 | 检查输入数据是否异常 | 如果历史数据几乎不变，模型会延续这个模式 |
| 模型未正确加载 | 确认 `from_pretrained()` 成功 | 模型权重未加载时，输出接近随机初始化 |

### 预测结果与真实值偏差很大

**排查步骤**：

1. **确认数据格式正确**：检查列名、数据类型、排序顺序
2. **检查时间戳**：确保时间戳与数据对应
3. **增加历史窗口**：`lookback` 建议不低于 200
4. **增加采样次数**：`sample_count=5` 可获得更稳定的结果
5. **调整温度**：`T=1.0` 是推荐起点
6. **确认模型处于 eval 模式**：`KronosPredictor` 内部的 `auto_regressive_inference` 使用 `torch.no_grad()` 禁用梯度计算，但**不会**自动调用 `model.eval()`。如果你的代码在 `predict()` 之前调用了 `model.train()`（例如在微调流程中），dropout 层会处于激活状态，导致预测结果不稳定。此时需要手动调用 `model.eval()` 恢复

> **注意**：金融预测本身具有高度不确定性。模型预测反映的是基于历史模式的概率分布，不是确定性预测。

### 预测结果包含负数价格

**可能原因**：

Kronos 在标准化空间进行预测，反标准化时可能产生负数（特别是历史波动极大或预测步数很长时）。

**解决方法**：

```python
# 后处理：将负数价格裁剪为 0
for col in ['open', 'high', 'low', 'close']:
    pred_df[col] = pred_df[col].clip(lower=0)

# 更好的方法：使用对数价格作为输入（需要自定义预处理）
```

### 预测的 OHLC 逻辑不一致（如 high < low）

**可能原因**：Kronos 将每个价格列独立预测，不保证 OHLC 之间的逻辑关系。

**后处理修复**：

推荐使用 pandas 向量化操作，速度远快于逐行迭代：

```python
def fix_ohlc_logic(pred_df):
    """修复预测结果中的 OHLC 逻辑不一致（向量化版本）"""
    pred_df['high'] = pred_df[['high', 'open', 'close']].max(axis=1)
    pred_df['low'] = pred_df[['low', 'open', 'close']].min(axis=1)
    return pred_df

pred_df = fix_ohlc_logic(pred_df)
```

> **为什么用向量化？** pandas 的 `max(axis=1)` / `min(axis=1)` 底层使用 NumPy C 扩展，比逐行迭代（`for i in range(len(pred_df))`）通常快一到两个数量级。

---

## MPS（Apple Silicon）相关问题

### RuntimeError: MPS does not support ...

**触发场景**：在 Apple Silicon Mac（M1/M2/M3/M4）上使用 MPS 后端时，部分 PyTorch 操作尚未被 MPS 支持。

**排查步骤**：

```python
# 检查 MPS 是否可用
import torch
print(f"MPS 可用: {torch.backends.mps.is_available()}")
print(f"MPS 已构建: {torch.backends.mps.is_built()}")
```

**常见解决方案**：

```python
# 方案 1：回退到 CPU
predictor = KronosPredictor(model, tokenizer, device="cpu")

# 方案 2：设置环境变量回退到 CPU
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

**注意**：MPS 后端在较新版本的 PyTorch（≥ 2.1）中支持更完善。如果频繁遇到 MPS 不支持的错误，建议升级 PyTorch 到最新版本。

### MPS 设备上的精度问题

MPS 后端在某些操作上可能产生与 CPU/CUDA 略有差异的结果（通常在 1e-6 量级）。这不影响预测质量，但如果需要严格一致性，使用 CPU。

---

## 高级调试技巧

### 使用 torch.autograd.detect_anomaly 定位 NaN 来源

当训练过程中出现 NaN 但不确定来自哪一步时，可以用异常检测定位：

```python
import torch

# 在训练脚本中启用异常检测
torch.autograd.set_detect_anomaly(True)

# 运行一次前向+反向传播，如果出现 NaN 会抛出 RuntimeError 并指出具体操作
```

启用后，PyTorch 会在产生 NaN 的具体算子处抛出异常，帮助精确定位问题源头。定位完成后记得关闭（设为 `False`），因为异常检测会使训练速度下降 30%-50%。

### 使用 torch.profiler 分析性能瓶颈

当推理速度不达预期时，可以用 profiler 分析耗时分布：

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=60, T=1.0, sample_count=1, verbose=False
    )

# 打印耗时最长的 10 个操作
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

### 逐步调试令牌化过程

当怀疑分词器的编解码有问题时，可以逐步检查中间结果：

```python
import torch
from model import KronosTokenizer

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
tokenizer.eval()

# 构造输入
x = torch.randn(1, 10, 6)

# Step 1: 检查嵌入
z = tokenizer.embed(x)
print(f"嵌入后范围: [{z.min():.2f}, {z.max():.2f}]")

# Step 2: 检查编码器输出
for i, layer in enumerate(tokenizer.encoder):
    z = layer(z)
    if torch.isnan(z).any():
        print(f"编码器第 {i} 层出现 NaN")
        break

# Step 3: 检查量化
z = tokenizer.quant_embed(z)
print(f"量化前范围: [{z.min():.2f}, {z.max():.2f}]")

# Step 4: 检查 BSQ 输出
s1_idx, s2_idx = tokenizer.encode(x, half=True)
print(f"s1 索引范围: [{s1_idx.min()}, {s1_idx.max()}]")
print(f"s2 索引范围: [{s2_idx.min()}, {s2_idx.max()}]")

# Step 5: 检查解码
decoded = tokenizer.decode([s1_idx, s2_idx], half=True)
print(f"解码结果范围: [{decoded.min():.2f}, {decoded.max():.2f}]")
if torch.isnan(decoded).any():
    print("解码结果包含 NaN")
```

---

## 自测清单

- [ ] 我知道遇到 OOM 时应该优先调整哪些参数
- [ ] 我能用脚本定位 DataFrame 中的 NaN 位置并修复
- [ ] 我知道批量预测要求所有序列长度一致的原因和解决方法
- [ ] 我知道如何排查模型下载失败的问题
- [ ] 我能说出训练不收敛的至少 3 个排查方向
- [ ] 我知道如何用 `detect_anomaly` 定位 NaN 来源
- [ ] 我知道分词器与模型不匹配时会出现什么错误
- [ ] 我知道 `KronosPredictor` 不会自动调用 `model.eval()`，在 `model.train()` 之后需手动恢复

---
