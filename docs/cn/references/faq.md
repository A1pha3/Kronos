# 常见问题

按类别组织的 Kronos 常见问题解答。

---

## 安装与环境

### Q: 支持 Windows 吗？

Kronos 是纯 Python 项目，理论上支持 Windows。但部分依赖（如 torch）在 Windows 上的安装可能需要额外步骤。推荐使用 Linux 或 macOS。

### Q: 需要多少 GPU 显存？

资源占用受以下因素影响：

| 因素 | 影响 |
|------|------|
| 模型规模 | mini ~100MB, small ~500MB, base ~1.5GB（推理时） |
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

### Q: 最大预测步数是多少？

接口层面对 `pred_len` 没有硬编码上限，但步数越长，不确定性通常越高。建议：

- **5 分钟线**：20-60 步（1-5 小时）
- **日线**：20-60 步（1-3 个月）
- **周线**：12-24 步（3-6 个月）

### Q: 只想预测价格，没有成交量怎么办？

只提供 `open`、`high`、`low`、`close` 四列即可。KronosPredictor 会自动将 `volume` 和 `amount` 填充为 0，不影响价格预测。

```python
x_df = df[['open', 'high', 'low', 'close']]
```

### Q: 历史数据需要多长？

模型支持的最大上下文长度为 512（Kronos-mini 为 2048）。超过此长度的历史会被截断。建议 `lookback` 设置在 200-512 之间，不少于 64。

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

### Q: 为什么使用交叉熵损失而不是 MSE 损失来训练预测模型？

因为预测模型是一个**分类模型**，在离散令牌空间进行预测。默认配置下（`s1_bits=10, s2_bits=10`），每个 s1/s2 令牌有 2^10 = 1,024 种选择，这是一个多分类问题，交叉熵是标准选择。MSE 用于分词器的重建损失（连续空间）。

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
app.run(debug=True, host='0.0.0.0', port=8080)  # 改为 8080
```

---

**文档元信息**
类型：参考文档 | 问题数：30+ | 更新日期：2026-04-12
