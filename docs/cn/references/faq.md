# 常见问题

按类别整理的 Kronos 常见问题解答。遇到问题时，先按关键词在下方的目录中定位对应章节。

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

Kronos 是纯 Python 项目，理论上可以在 Windows 上运行，但 PyTorch 的 CUDA 支持在 Windows 上配置较复杂。推荐使用 Linux 或 macOS，这两个平台开箱即用。

### Q: 需要多少 GPU 显存？

资源占用受以下因素影响：

| 因素 | 影响 |
|------|------|
| 模型规模 | mini ~100MB, small ~500MB, base ~1.5GB（推理时，基于模型文件大小估算） |
| `lookback` / `max_context` | 序列越长，显存占用越大 |
| `pred_len` | 预测步数越多，总推理时间越长 |
| `sample_count` | 每增加 1，显存占用近似翻倍 |
| `predict_batch()` 的 batch 数 | 实际 batch = batch 数 x sample_count |

没有 GPU 时，CPU 也可以运行，只是速度更慢。建议先用 `Kronos-mini` 或 `Kronos-small` 做一次小规模试跑，确认流程跑通后再逐步增大窗口和采样次数。

> **要点**：先小规模试跑，再逐步增大 `lookback`、`sample_count` 和模型规模。

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

> **要点**：国内用户设置 `HF_ENDPOINT=https://hf-mirror.com` 即可加速模型下载。

---

## 模型与预测

### Q: 预测结果每次都不同，正常吗？

正常。Kronos 使用采样策略生成预测，具有随机性。控制方法：

- **降低随机性**：减小温度（`T=0.3`）、使用 `top_k=1`（贪婪解码）
- **增加稳定性**：增大 `sample_count`，让多条采样路径做平均
- **完全确定性**：设置 `T=0.1, top_k=1, sample_count=1`（每次输出几乎相同）

> **要点**：采样导致随机性是正常行为；降低温度或使用 `top_k=1` 可获得确定性结果。

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

> **要点**：入门选 `Kronos-small`，资源有限选 `Kronos-mini`，效果优先选 `Kronos-base`；切换模型时务必同步更换分词器。

### Q: top_k 和 top_p 有什么区别？应该在什么场景下使用？

**top_k** 保留概率最高的 k 个令牌，适合需要确定性结果的场景（`top_k=1` 等价于贪婪解码）。**top_p**（核采样）保留累计概率达到 p 的令牌集合，适合需要多样性但排除极低概率选项的场景。推荐组合：

- 日常使用：`top_k=0, top_p=0.9`（仅 top-p 过滤，默认值）
- 需要确定性：`top_k=1`（贪婪解码）
- 探索性分析：`top_k=0, top_p=0.95`

### Q: Kronos-large 什么时候开源？

Kronos-large 目前未开源。仓库 README 中标注该模型仅用于学术研究。如需使用最大已开源模型，请选择 Kronos-base（102.3M 参数）。

### Q: 最大预测步数是多少？

接口层面对 `pred_len` 没有硬编码上限，但步数越长，不确定性通常越高。建议：

- **5 分钟线**：20-60 步（1-5 小时）
- **日线**：20-60 步（1-3 个月）
- **周线**：12-24 步（3-6 个月）

> **要点**：步数无硬上限，但建议 20-60 步；步数越长不确定性越高。

### Q: 只想预测价格，没有成交量怎么办？

只提供 `open`、`high`、`low`、`close` 四列即可。KronosPredictor 会自动将 `volume` 和 `amount` 填充为 0，不影响价格预测。若已有 `volume` 列但缺少 `amount` 列，`amount` 会被自动推算为 `volume × mean(open, high, low, close)`（即逐行对四个价格取算术均值作为均价，而非成交量加权均价）。

```python
x_df = df[['open', 'high', 'low', 'close']]
```

> **要点**：只需 OHLC 四列即可预测价格，`volume` 和 `amount` 会自动补零。详见 [数据准备指南](../getting-started/03-data-preparation.md)。

### Q: 历史数据需要多长？

模型支持的最大上下文长度为 512（Kronos-mini 为 2048）。超过此长度的历史会被截断。建议 `lookback` 设置在 200-512 之间，不少于 64。

> **要点**：`lookback` 建议 200-512，不少于 64；详见 [KronosPredictor 使用指南](../core-concepts/04-predictor.md)。

### Q: predict() 和 auto_regressive_inference 的默认参数为什么不同？

两者面向不同的使用场景：

| 参数 | `predict()` | `auto_regressive_inference()` |
|------|-------------|-------------------------------|
| `top_p` | 0.9（偏保守） | 0.99（更宽松） |
| `sample_count` | 1（单条路径） | 5（多条路径取平均） |

`predict()` 面向用户调用，默认参数偏向确定性输出。底层 `auto_regressive_inference()` 面向内部调用，默认参数偏向探索性采样。如果你直接调用 `auto_regressive_inference()`，注意默认行为差异。

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
def fix_ohlc_logic(pred_df):
    """修复预测结果中的 OHLC 逻辑不一致（向量化版本）"""
    pred_df['high'] = pred_df[['high', 'open', 'close']].max(axis=1)
    pred_df['low'] = pred_df[['low', 'open', 'close']].min(axis=1)
    return pred_df

pred_df = fix_ohlc_logic(pred_df)
```

> **要点**：OHLC 各列独立预测，后处理时用 `max/min` 修复即可。

详见 [错误排查指南](troubleshooting.md)。

---

## 数据

### Q: 支持哪些数据格式？

CSV 文件，包含 OHLCV 列和时间戳。列名必须严格匹配：`open`、`high`、`low`、`close`（必填），`volume` 和 `amount`（可选，缺失时自动填充为 0 或推算）。详见 [数据准备指南](../getting-started/03-data-preparation.md)。

### Q: 数据中有缺失值怎么办？

KronosPredictor **不接受**含 NaN 的输入。传入前需要先处理：

```python
# 前向填充（推荐）
df = df.ffill()

# 或删除缺失行
df = df.dropna()
```

> **要点**：含 NaN 会直接报错，传入前务必用 `df.ffill()` 或 `dropna()` 处理。

详见 [数据准备指南](../getting-started/03-data-preparation.md) 的"常见数据问题与处理"一节。

### Q: 如何使用分钟线/小时线/日线？

Kronos 对时间粒度没有限制。只需确保：

1. 数据按时间升序排列
2. 时间戳格式正确（`pd.to_datetime` 可解析）
3. 历史窗口长度足够（建议 >= 64）

### Q: Kronos 能否用于非金融数据？

不适合。Kronos 的输入维度（6 维 OHLCV）、令牌化方式（BSQ 量化）和时间嵌入都是针对 K 线特征定制的。对于天气、传感器、流量等通用时间序列，可考虑 TimesFM、Chronos 等通用时序基础模型。

### Q: 如何在预测前验证数据质量？

```python
import pandas as pd
import numpy as np

df = pd.read_csv("your_data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

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
if 'timestamps' in df.columns:
    assert df['timestamps'].is_monotonic_increasing, "时间戳未升序排列"
```

> **要点**：预测前务必检查缺列、NaN、非正价格和时间排序四项。更完整的质量检查脚本见 [数据准备指南](../getting-started/03-data-preparation.md) 的"动手练习"一节。

---

## 微调

### Q: 微调需要多少数据？

取决于数据的复杂度和微调目标。一般建议至少包含几千个数据点，数据越少越容易过拟合。可以用验证集的损失曲线判断数据量是否足够——验证损失在几个 epoch 后开始回升，说明数据量可能不足。

> **要点**：至少几千个数据点；验证损失提前回升说明数据量不足。详见 [CSV 微调指南](../advanced/02-finetune-csv.md)。

### Q: 必须先微调分词器吗？

不一定。如果数据特征与预训练数据相似（常规股票 K 线），可以跳过分词器微调，只微调预测模型。

```yaml
# YAML 配置中跳过分词器训练
experiment:
  train_tokenizer: false
  train_basemodel: true
```

只有当数据特征差异较大（如加密货币衍生品、特殊指标数据）时，分词器微调才显著提升效果。

> **要点**：常规股票数据可跳过分词器微调，直接训练预测模型；数据特征差异大时再考虑。详见 [Qlib 微调指南](../advanced/01-finetune-qlib.md) 和 [CSV 微调指南](../advanced/02-finetune-csv.md)。

### Q: 微调数据的格式要求是什么？

CSV 微调流水线要求 CSV 文件包含 `open`、`high`、`low`、`close`、`volume`、`amount` 六列。Qlib 微调流水线由 Qlib 数据接口自动提供，无需手动准备 CSV。详见 [数据准备指南](../getting-started/03-data-preparation.md) 和 [CSV 微调指南](../advanced/02-finetune-csv.md)。

常见的数据格式问题：

| 问题 | 检查方法 |
|------|---------|
| 列名不匹配 | 确认列名全部小写，无空格 |
| 时间未排序 | `df = df.sort_values('timestamps')` |
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

> **要点**：使用 `from_pretrained()` 指向本地 checkpoint 路径即可加载微调后的模型。详见 [KronosPredictor 使用指南](../core-concepts/04-predictor.md)。

### Q: 微调后效果还不如预训练模型？

可能的原因和排查方向：

| 原因 | 排查方法 |
|------|---------|
| 过拟合（训练数据太少） | 检查验证损失是否在早期 epoch 就开始回升 |
| 学习率过大 | 尝试将 tokenizer_learning_rate 降至 1e-4，predictor 降至 1e-5 |
| 训练轮数过多 | 减少 epochs，使用 early stopping |
| 数据分布与预训练数据差异极大 | 先检查数据预处理是否正确（标准化后值在 [-5, 5] 范围内） |

> **要点**：优先排查过拟合、学习率和数据量问题；验证损失曲线是最直接的诊断工具。

---

## 技术细节

### Q: Kronos 和传统时间序列模型有什么区别？

| 维度 | 传统模型（LSTM、ARIMA） | Kronos |
|------|------------------------|--------|
| 建模方式 | 连续值回归 | 离散令牌分类 |
| 生成策略 | 确定性 | 可控采样（温度、top-p） |
| 泛化能力 | 通常围绕单标的建模 | 在 45+ 交易所数据上预训练，开箱即用 |
| 多步预测 | 误差逐步累积 | 在离散令牌空间更稳定 |
| 时间感知 | 通常无显式时间编码 | 内置 TemporalEmbedding 捕捉周期性 |

> **延伸阅读**：关于"连续值回归 vs 离散令牌分类"的差异，[项目总览](../core-concepts/01-overview.md)的"离散令牌预测为何比连续回归更稳定"一节提供了详细的失败模式对比表（误差累积、分布外漂移、多峰分布）。关于令牌空间的设计细节，参见 [层级令牌体系](../core-concepts/05-hierarchical-tokens.md) 和 [BSQ 算法解析](../architecture/02-bsq-algorithm.md)。

### Q: 为什么使用交叉熵损失而不是 MSE 损失来训练预测模型？

因为预测模型是一个**分类模型**，在离散令牌空间进行预测。默认预训练配置下（`s1_bits=10, s2_bits=10`），每个 s1/s2 令牌有 2^10 = 1,024 种选择，这是一个多分类问题，交叉熵是标准选择。`s1_bits` 和 `s2_bits` 是可配置参数，自定义模型可调整。MSE 用于分词器的重建损失（连续空间）。

> **要点**：预测模型是分类任务（在 2^10 = 1024 个令牌中选一个），交叉熵是标准损失。详见 [层级令牌体系](../core-concepts/05-hierarchical-tokens.md) 和 [BSQ 算法解析](../architecture/02-bsq-algorithm.md)。

### Q: DependencyAwareLayer 为什么使用交叉注意力而不是简单拼接？

简单拼接是一种静态融合——无论 s1 的具体取值如何，融合方式都相同。交叉注意力允许 s2 的预测**动态地**关注与 s1 最相关的上下文信息——不同的 s1 值会产生不同的注意力模式，实现更灵活的条件依赖。

详见 [Transformer 设计分析](../architecture/03-transformer-design.md)。

### Q: Kronos 的预测有没有置信度指标？

Kronos 没有直接输出"置信度分数"的 API，但可以通过多次采样估计不确定性：

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

区间越宽，说明模型对该时间段的预测越不确定。

> **要点**：没有直接置信度 API，但用多次采样 + 百分位区间即可估计不确定性。

### Q: Kronos 的预测能不能用于实盘交易？

**不能直接用于实盘交易。** Kronos 是学术研究模型，旨在探索基于离散令牌的金融时序预测方法。将模型预测用于实际投资决策存在重大风险：

- 模型预测基于历史数据模式，不保证未来走势
- 突发事件（政策变动、黑天鹅等）无法被模型捕捉
- 预测结果具有随机性（采样策略），不同运行可能产生不同结论
- 模型未考虑交易成本、流动性、滑点等实盘因素

> **免责声明**：本项目仅供学术研究和教育目的使用。作者和贡献者不对因使用本模型而产生的任何投资损失承担责任。在做出任何投资决策前，请咨询专业的金融顾问。

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

- [ ] Kronos 预测结果每次不同是正常的吗？如何获得确定性结果？
- [ ] 选择模型时，Kronos-mini 与其他模型在分词器和上下文长度上有什么区别？
- [ ] `predict()` 和 `predict_batch()` 的输入要求有何不同？
- [ ] 输入数据包含 NaN 会发生什么？正确的预处理方式是什么？
- [ ] 微调时是否必须重新训练分词器？什么情况下才需要？
- [ ] 如何通过多次采样估计预测的不确定性？
- [ ] Kronos 的预测能否直接用于实盘交易？存在哪些风险？

无法回答的问题，回到上方对应章节重读。每个章节末尾的"要点"框是该节核心结论的速查摘要。
