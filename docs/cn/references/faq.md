# 常见问题

按类别组织的 Kronos 常见问题解答。

---

## 安装与环境

### Q: 支持 Windows 吗？

Kronos 是纯 Python 项目，理论上支持 Windows。但部分依赖（如 torch）在 Windows 上的安装可能需要额外步骤。推荐使用 Linux 或 macOS。

### Q: 需要多少 GPU 显存？

这没有一个脱离环境的固定答案。资源占用至少会受到以下因素影响：

- 选择的模型规模
- `lookback` / `max_context`
- `pred_len`
- `sample_count`
- 是否使用 `predict_batch()`

没有 GPU 时，CPU 也可以运行；只是速度通常更慢。建议先用 `Kronos-mini` 或 `Kronos-small` 做一次小规模试跑，再逐步增加窗口和采样次数。

### Q: 如何更新到最新版本？

```bash
cd Kronos
git pull origin master
pip install -r requirements.txt --upgrade
```

---

## 模型与预测

### Q: 预测结果每次都不同，正常吗？

正常。Kronos 使用采样策略生成预测，具有随机性。控制方法：

- **降低随机性**：减小温度（`T=0.3`）、使用 `top_k=1`（贪婪解码）
- **增加稳定性**：增大 `sample_count`，让多条采样路径做平均

### Q: 应该选择哪个模型？

| 场景 | 推荐模型 |
|------|----------|
| 快速验证、资源极有限 | `Kronos-mini`（4.1M 参数，上下文 2048） |
| 日常使用、微调实验 | `Kronos-small`（24.7M 参数，迭代快） |
| 效果优先、资源适中 | `Kronos-base`（102.3M 参数） |
| 追求最佳效果、资源充足 | `Kronos-large`（499.2M 参数，未开源） |

> **注意**：Kronos-mini 使用专用的 `Kronos-Tokenizer-2k` 分词器，其余模型共用 `Kronos-Tokenizer-base`。切换模型时请确保分词器匹配。

### Q: 最大预测步数是多少？

接口层面对 `pred_len` 没有写死上限，但步数越长，不确定性通常越高。更稳妥的做法是根据你的使用场景决定预测跨度，并用回测或留出集验证它是否仍有参考价值。

### Q: 只想预测价格，没有成交量怎么办？

只提供 `open`、`high`、`low`、`close` 四列即可。KronosPredictor 会自动将 `volume` 和 `amount` 填充为 0，不影响价格预测。

```python
x_df = df[['open', 'high', 'low', 'close']]
```

### Q: 历史数据需要多长？

至少要保证有足够的历史窗口供模型建立上下文。源码层面能直接确认的是：`Kronos-small/base/large` 常见上下文长度为 `512`，`Kronos-mini` 为 `2048`；超过这一长度的历史会被截断为最近的一段。至于“多少根最合适”，需要根据市场频率和任务目标自己验证。

### Q: predict() 和 predict_batch() 有什么区别？

| 维度 | `predict()` | `predict_batch()` |
|------|-------------|-------------------|
| 输入 | 单条 DataFrame | DataFrame 列表 |
| 长度限制 | 无特殊限制 | 所有序列长度必须一致 |
| 速度 | 适合单条预测 | 多条并行，GPU 利用率更高 |
| 返回 | 单个 DataFrame | DataFrame 列表 |

**为什么 `predict_batch()` 要求长度一致？** 因为内部使用 `np.stack()` 将多个序列堆叠为三维张量 `(batch, seq_len, features)`，这要求所有序列的 `seq_len` 维度相同。

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
3. 历史窗口长度足够（建议 ≥ 64）

---

## 微调

### Q: 微调需要多少数据？

取决于数据的复杂度和微调的目标。一般建议：

没有单一适用于所有任务的最小样本数。可以确定的是：数据越少，越容易过拟合；数据分布与预训练分布差异越大，越需要认真做验证集评估。

### Q: 必须先微调分词器吗？

不一定。如果数据特征与预训练数据相似（常规股票 K 线），可以跳过分词器微调，只微调预测模型。

在 YAML 配置中设置：
```yaml
experiment:
  train_tokenizer: false
  train_basemodel: true
```

### Q: 单 GPU 可以微调吗？

CSV 微调流水线支持单 GPU。直接运行：
```bash
python finetune_csv/train_sequential.py --config config.yaml
```

Qlib 微调流水线需要 `torchrun`：
```bash
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```

### Q: 微调后如何使用新模型？

```python
tokenizer = KronosTokenizer.from_pretrained("outputs/my_exp/tokenizer/best_model")
model = Kronos.from_pretrained("outputs/my_exp/basemodel/best_model")
predictor = KronosPredictor(model, tokenizer)
```

---

## 技术细节

### Q: Kronos 和传统时间序列模型（LSTM、ARIMA）有什么区别？

| 维度 | 传统模型 | Kronos |
|------|----------|--------|
| 建模方式 | 连续值回归 | 离散令牌分类 |
| 生成策略 | 确定性 | 可控采样（温度、top-p） |
| 泛化能力 | 通常围绕单任务建模 | 根目录 `README.md` 写明项目在 45+ 交易所数据上预训练 |
| 多步预测 | 常见做法是连续值外推 | Kronos 先预测离散令牌，再解码回连续值 |

### Q: 为什么使用交叉熵损失而不是 MSE 损失来训练预测模型？

因为预测模型是一个**分类模型**，在离散令牌空间进行预测。默认配置下（`s1_bits=10, s2_bits=10`），每个 s1/s2 令牌有 2^10 = 1,024 种选择，这是一个多分类问题，交叉熵是标准选择。MSE 用于分词器的重建损失（连续空间）。

### Q: DependencyAwareLayer 为什么使用交叉注意力而不是简单拼接？

简单拼接（将 s1 嵌入与 Transformer 输出拼接后线性映射）是一种静态融合。交叉注意力允许 s2 的预测**动态地**关注与 s1 最相关的上下文信息——不同的 s1 值会导致不同的注意力模式，实现更灵活的条件依赖。

具体来说，`DependencyAwareLayer` 的交叉注意力参数为：
- **query** = `sibling_embed`（s1 的令牌嵌入）
- **key/value** = `hidden_states`（Transformer 的上下文表示）

残差连接方向为 `hidden_states + attn_out`（保留 Transformer 上下文，用交叉注意力结果作为修正），然后经过 RMSNorm。训练时使用因果掩码（防止未来信息泄漏），推理时使用非因果模式（s1 已确定，无泄漏风险）。详见 [Transformer 设计分析](../architecture/03-transformer-design.md)。

---

## Web UI

### Q: 如何启动 Web 界面？

```bash
pip install -r webui/requirements.txt
python webui/run.py
```

浏览器自动打开 `http://localhost:7070`。

### Q: Web UI 端口被占用怎么办？

修改 `webui/run.py` 中的端口号：

```python
app.run(debug=True, host='0.0.0.0', port=8080)  # 改为 8080
```

---

**文档元信息**
类型：参考文档 | 问题数：25+
