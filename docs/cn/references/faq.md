# 常见问题

按类别组织的 Kronos 常见问题解答。

---

## 目录

- [安装与环境](#安装与环境)
- [模型与预测](#模型与预测)
- [数据](#数据)
- [微调](#微调)
- [技术细节](#技术细节)
- [Web UI](#web-ui)

---

## 安装与环境

### Q: 支持 Windows 吗？

Kronos 是纯 Python 项目，理论上支持 Windows。但部分依赖（如 torch）在 Windows 上的安装可能需要额外步骤。推荐使用 Linux 或 macOS。

### Q: 需要多少 GPU 显存？

资源占用受以下因素影响：

| 因素 | 影响 |
|------|------|
| 模型规模 | mini ~100MB, small ~500MB, base ~1.5GB（推理时，基于模型文件大小估算） |
| `lookback` / `max_context` | 序列越长，显存占用越大 |
| `pred_len` | 预测步数越多，总推理时间越长 |
| `sample_count` | 每增加 1，显存占用近似翻倍 |
| `predict_batch()` 的 batch 数 | 实际 batch = batch 数 x sample_count |

没有 GPU 时，CPU 也可以运行；只是速度更慢。建议先用 `Kronos-mini` 或 `Kronos-small` 做一次小规模试跑，再逐步增加窗口和采样次数。

### Q: 如何更新到最新版本？

```bash
cd Kronos
git pull origin master
pip install -r requirements.txt --upgrade
```

### Q: 如何设置镜像加速模型下载？

在中国大陆访问 HuggingFace Hub 可能较慢。设置镜像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或者在 Python 代码中设置：

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

---

## 模型与预测

### Q: 预测结果每次都不同，正常吗？

正常。Kronos 使用采样策略生成预测，具有随机性。控制方法：

- **降低随机性**：减小温度（`T=0.3`）、使用 `top_k=1`（贪婪解码）
- **增加稳定性**：增大 `sample_count`，让多条采样路径做平均
- **完全确定性**：设置 `T=0.1, top_k=1, sample_count=1`（每次输出几乎相同）

### Q: 如何获得可复现的结果？

```python
import torch
torch.manual_seed(42)  # 在 predict() 调用前设置种子

pred_df = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=60, T=0.1, top_k=1, sample_count=1
)
```

注意：`top_k=1` 等价于贪婪解码（总是选概率最高的令牌），此时 T 和 top_p 不再有实际效果。

### Q: 应该选择哪个模型？

| 场景 | 推荐模型 |
|------|----------|
| 快速验证、资源极有限 | `Kronos-mini`（4.1M 参数，上下文 2048） |
| 日常使用、微调实验 | `Kronos-small`（24.7M 参数，推荐入门） |
| 效果优先、资源适中 | `Kronos-base`（102.3M 参数） |
| 历史数据超过 512 个时间点 | `Kronos-mini`（唯一支持 2048 上下文） |
| 追求最佳效果、资源充足 | `Kronos-large`（499.2M 参数，未开源） |

> **注意**：Kronos-mini 使用专用的 `Kronos-Tokenizer-2k` 分词器，其余模型共用 `Kronos-Tokenizer-base`。切换模型时请确保分词器匹配。详见 [模型对比与选型](../advanced/07-model-comparison.md)。

### Q: top_k 和 top_p 有什么区别？应该在什么场景下使用？

**top_k** 保留概率最高的 k 个令牌，适合需要确定性结果的场景（如 `top_k=1` 等价于贪婪解码）。**top_p**（核采样）保留累计概率达到 p 的令牌集合，适合需要多样性但排除极低概率选项的场景。推荐组合：日常使用 `top_k=0, top_p=0.9`（只用 top_p 过滤）；需要确定性时 `top_k=1`（贪婪解码）；探索性分析时 `top_k=0, top_p=0.95`。

### Q: Kronos-large 什么时候开源？

Kronos-large 目前未开源。仓库 README 中标注该模型仅用于学术研究。如需使用最大已开源模型，请选择 Kronos-base（102.3M 参数）。

### Q: 最大预测步数是多少？

接口层面对 `pred_len` 没有硬编码上限，但步数越长，不确定性通常越高。建议：

- **5 分钟线**：20-60 步（1-5 小时）
- **日线**：20-60 步（1-3 个月）
- **周线**：12-24 步（3-6 个月）

### Q: 只想预测价格，没有成交量怎么办？

只提供 `open`、`high`、`low`、`close` 四列即可。KronosPredictor 会自动将 `volume` 和 `amount` 填充为 0，不影响价格预测。若已有 `volume` 列但缺少 `amount` 列，`amount` 会被自动推算为 `volume × mean(open, high, low, close)`（即逐行对四个价格取算术均值作为均价，而非成交量加权均价）。

```python
x_df = df[['open', 'high', 'low', 'close']]
```

### Q: 历史数据需要多长？

模型支持的最大上下文长度为 512（Kronos-mini 为 2048）。超过此长度的历史会被截断。建议 `lookback` 设置在 200-512 之间，不少于 64。

### Q: predict() 和 auto_regressive_inference 的默认参数为什么不同？

`predict()` 的默认参数是 `T=1.0, top_k=0, top_p=0.9, sample_count=1`（面向用户，默认保守），而底层 `auto_regressive_inference()` 的默认参数是 `top_p=0.99, sample_count=5`（面向内部调用）。如果你直接调用 `auto_regressive_inference()`，请留意默认行为差异。

### Q: predict() 和 predict_batch() 有什么区别？

| 维度 | `predict()` | `predict_batch()` |
|------|-------------|-------------------|
| 输入 | 单条 DataFrame | DataFrame 列表 |
| 长度限制 | 无特殊限制 | 所有序列长度必须一致 |
| 速度 | 适合单条预测 | 多条并行，GPU 利用率更高 |
| 返回 | 单个 DataFrame | DataFrame 列表 |

长度不一致时，用循环分别调用 `predict()` 即可：

```python
results = [predictor.predict(df=d, x_timestamp=xt, y_timestamp=yt, pred_len=60) 
           for d, xt, yt in zip(df_list, xt_list, yt_list)]
```

### Q: 预测的 OHLC 逻辑不一致（如 high < low）怎么办？

Kronos 将每个价格列独立预测，不保证 OHLC 之间的逻辑关系。可以使用后处理修复：

```python
def fix_ohlc(df):
    for i in range(len(df)):
        o, h, l, c = df.at[i, 'open'], df.at[i, 'high'], df.at[i, 'low'], df.at[i, 'close']
        df.at[i, 'high'] = max(h, o, c)
        df.at[i, 'low'] = min(l, o, c)
    return df
```

详见 [错误排查指南](troubleshooting.md)。

---

## 数据

### Q: 支持哪些数据格式？

CSV 文件，包含 OHLCV 列和时间戳。列名必须严格匹配：`open`、`high`、`low`、`close`。`volume` 和 `amount` 可选。

### Q: 数据中有缺失值怎么办？

KronosPredictor **不接受**含 NaN 的输入。传入前需要先处理：

```python
# 前向填充（推荐）
df = df.ffill()

# 或删除缺失行
df = df.dropna()
```

### Q: 如何使用分钟线/小时线/日线？

Kronos 对时间粒度没有限制。只需确保：

1. 数据按时间升序排列
2. 时间戳格式正确（`pd.to_datetime` 可解析）
3. 历史窗口长度足够（建议 >= 64）

### Q: Kronos 能否用于非金融数据？

Kronos 的模型架构（分词器 + 自回归 Transformer）是为金融 K 线数据设计的。它的输入维度（6 维 OHLCV）、令牌化方式（BSQ 量化）、时间嵌入（TemporalEmbedding）都是针对 K 线特征量身定制的。

对于非金融时间序列（如天气、传感器、流量数据），Kronos 不适合直接使用。如果你需要处理通用时间序列，可以考虑 TimesFM、Chronos 等通用时序基础模型。

### Q: 如何在预测前验证数据质量？

```python
import pandas as pd
import numpy as np

df = pd.read_csv("your_data.csv")

# 检查必需列
required = ['open', 'high', 'low', 'close']
missing = [c for c in required if c not in df.columns]
assert not missing, f"缺少列: {missing}"

# 检查 NaN
assert not df[required].isnull().any().any(), "价格列存在 NaN"

# 检查非正价格（可能是停牌或异常数据）
for col in required:
    zeros = (df[col] <= 0).sum()
    if zeros > 0:
        print(f"警告: {col} 有 {zeros} 个非正值")

# 检查时间排序
if 'timestamp' in df.columns:
    assert df['timestamp'].is_monotonic_increasing, "时间戳未升序排列"
```

---

## 微调

### Q: 微调需要多少数据？

取决于数据的复杂度和微调目标。一般建议至少包含几千个数据点。数据越少越容易过拟合。可以用验证集的损失曲线判断数据量是否足够——如果验证损失在几个 epoch 后就开始回升，说明数据量可能不足。

### Q: 必须先微调分词器吗？

不一定。如果数据特征与预训练数据相似（常规股票 K 线），可以跳过分词器微调，只微调预测模型。

```yaml
# YAML 配置中跳过分词器训练
experiment:
  train_tokenizer: false
  train_basemodel: true
```

只有当数据特征差异较大（如加密货币衍生品、特殊指标数据）时，分词器微调才显著提升效果。

### Q: 微调数据的格式要求是什么？

CSV 微调流水线要求 CSV 文件包含 `open`、`high`、`low`、`close`、`volume`、`amount` 六列。Qlib 微调流水线由 Qlib 数据接口自动提供，无需手动准备 CSV。

常见的数据格式问题：

| 问题 | 检查方法 |
|------|---------|
| 列名不匹配 | 确认列名全部小写，无空格 |
| 时间未排序 | `df = df.sort_values('timestamp')` |
| 含 NaN | `df.isnull().sum()`，需在传入前填充或删除 |
| 停牌日价格为 0 | 需手动剔除或前向填充 |

### Q: 单 GPU 可以微调吗？

CSV 微调流水线支持单 GPU：

```bash
python finetune_csv/train_sequential.py --config config.yaml
```

Qlib 微调流水线需要 `torchrun`：

```bash
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```

### Q: 微调后如何使用新模型？

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("outputs/my_exp/tokenizer/best_model")
model = Kronos.from_pretrained("outputs/my_exp/basemodel/best_model")
predictor = KronosPredictor(model, tokenizer)
```

### Q: 微调后效果还不如预训练模型？

可能的原因和排查方向：

| 原因 | 排查方法 |
|------|---------|
| 过拟合（训练数据太少） | 检查验证损失是否在早期 epoch 就开始回升 |
| 学习率过大 | 尝试将 tokenizer_learning_rate 降至 1e-4，predictor 降至 1e-5 |
| 训练轮数过多 | 减少 epochs，使用 early stopping |
| 数据分布与预训练数据差异极大 | 先检查数据预处理是否正确（标准化后值在 [-5, 5] 范围内） |

---

## 技术细节

### Q: Kronos 和传统时间序列模型有什么区别？

| 维度 | 传统模型（LSTM、ARIMA） | Kronos |
|------|------------------------|--------|
| 建模方式 | 连续值回归 | 离散令牌分类 |
| 生成策略 | 确定性 | 可控采样（温度、top-p） |
| 泛化能力 | 通常围绕单任务建模 | 在 45+ 交易所数据上预训练，开箱即用 |
| 多步预测 | 误差逐步累积 | 在离散令牌空间更稳定 |
| 时间感知 | 通常无显式时间编码 | 内置 TemporalEmbedding 捕捉周期性 |

> **延伸阅读**：关于"连续值回归 vs 离散令牌分类"的差异，[项目总览](../core-concepts/01-overview.md)的"离散令牌预测为何比连续回归更稳定"一节提供了详细的失败模式对比表（误差累积、分布外漂移、多峰分布）。关于令牌空间的设计细节，参见 [层级令牌体系](../core-concepts/05-hierarchical-tokens.md) 和 [BSQ 算法解析](../architecture/02-bsq-algorithm.md)。

### Q: 为什么使用交叉熵损失而不是 MSE 损失来训练预测模型？

因为预测模型是一个**分类模型**，在离散令牌空间进行预测。默认预训练配置下（`s1_bits=10, s2_bits=10`），每个 s1/s2 令牌有 2^10 = 1,024 种选择，这是一个多分类问题，交叉熵是标准选择。`s1_bits` 和 `s2_bits` 是可配置参数，自定义模型可调整。MSE 用于分词器的重建损失（连续空间）。

### Q: DependencyAwareLayer 为什么使用交叉注意力而不是简单拼接？

简单拼接是一种静态融合——无论 s1 的具体取值如何，融合方式都相同。交叉注意力允许 s2 的预测**动态地**关注与 s1 最相关的上下文信息——不同的 s1 值会产生不同的注意力模式，实现更灵活的条件依赖。

详见 [Transformer 设计分析](../architecture/03-transformer-design.md)。

### Q: Kronos 的预测有没有置信度指标？

Kronos 没有直接输出"置信度分数"的 API。但你可以通过以下方式估计不确定性：

```python
# 生成多条路径，计算置信区间
paths = []
for _ in range(20):
    pred = predictor.predict(df=x_df, x_timestamp=xt, y_timestamp=yt,
                            pred_len=60, T=1.0, sample_count=1, verbose=False)
    paths.append(pred['close'].values)

import numpy as np
paths = np.array(paths)
p5, p95 = np.percentile(paths, 5, axis=0), np.percentile(paths, 95, axis=0)
print(f"90% 置信区间宽度: {(p95 - p5).mean():.2f}")
```

区间越宽，说明模型越不确定。

---

## Web UI

### Q: 如何启动 Web 界面？

```bash
pip install -r webui/requirements.txt
python webui/run.py
```

浏览器自动打开 `http://localhost:7070`。详见 [Web UI 使用指南](../advanced/05-webui-guide.md)。

### Q: Web UI 端口被占用怎么办？

修改 `webui/run.py` 中的端口号：

```python
app.run(host='0.0.0.0', port=8080)  # 改为 8080
```

> **注意**：`debug=True` 仅用于本地开发调试，不应在生产或局域网共享场景中使用。详见 [Web UI 使用指南](../advanced/05-webui-guide.md)。

---

## 自测清单

完成本 FAQ 阅读后，检查以下问题你是否能快速回答：

- [ ] Kronos 预测结果每次不同是正常的吗？如何获得确定性结果？
- [ ] 选择模型时，Kronos-mini 与其他模型在分词器和上下文长度上有什么区别？
- [ ] `predict()` 和 `predict_batch()` 的输入要求有何不同？
- [ ] 输入数据包含 NaN 会发生什么？正确的预处理方式是什么？
- [ ] 微调时是否必须重新训练分词器？什么情况下才需要？
- [ ] 如何通过多次采样估计预测的不确定性？

如果某个问题无法回答，请回到上方对应的章节。
