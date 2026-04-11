# 常见问题

按类别组织的 Kronos 常见问题解答。

---

## 安装与环境

### Q: 支持 Windows 吗？

Kronos 是纯 Python 项目，理论上支持 Windows。但部分依赖（如 torch）在 Windows 上的安装可能需要额外步骤。推荐使用 Linux 或 macOS。

### Q: 需要多少 GPU 显存？

取决于模型大小和批量设置：

| 模型 | 单条预测 (CPU) | 单条预测 (GPU) | 批量预测 5 条 (GPU) |
|------|---------------|---------------|-------------------|
| Kronos-small | ~2 GB 内存 | ~1 GB 显存 | ~2 GB 显存 |
| Kronos-base | ~4 GB 内存 | ~2 GB 显存 | ~4 GB 显存 |
| Kronos-large | ~8 GB 内存 | ~4 GB 显存 | ~8 GB 显存 |

没有 GPU 时，CPU 可以运行所有功能，只是速度较慢。

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
- **增加稳定性**：增大 `sample_count`（推荐 3-5 次，默认为 1。超过 10 次的边际收益递减，且推理时间线性增长）

### Q: 应该选择哪个模型？

| 场景 | 推荐模型 |
|------|----------|
| 快速验证、资源有限 | `Kronos-small` |
| 日常使用 | `Kronos-base` |
| 追求最佳效果、资源充足 | `Kronos-large` |
| 微调实验 | `Kronos-small`（迭代快） |

### Q: 最大预测步数是多少？

技术上没有硬性限制。但预测步数越长，不确定性越大。推荐范围：

| 时间粒度 | 推荐预测步数 | 对应时间跨度 |
|---------|-------------|-------------|
| 5 分钟线 | 60-120 | 5-10 小时 |
| 日线 | 20-60 | 1-3 个月 |
| 周线 | 12-24 | 3-6 个月 |

### Q: 只想预测价格，没有成交量怎么办？

只提供 `open`、`high`、`low`、`close` 四列即可。KronosPredictor 会自动将 `volume` 和 `amount` 填充为 0，不影响价格预测。

```python
x_df = df[['open', 'high', 'low', 'close']]
```

### Q: 历史数据需要多长？

建议至少 64 根 K 线，推荐 200-512 根。低于 64 根可能导致预测质量下降。模型最大上下文为 512，超过 512 的历史会被截断为最近 512 根。

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
df = df.fillna(method='ffill')

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

- **分词器微调**：至少几千条 K 线（如 5 分钟线的 2-3 个月数据）
- **预测模型微调**：至少几万条 K 线（如日线的 1-2 年数据）

数据太少可能导致过拟合。

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
| 泛化能力 | 通常针对单一序列 | 预训练于 45+ 交易所数据 |
| 多步预测 | 误差累积 | 离散空间更稳定 |

### Q: 为什么使用交叉熵损失而不是 MSE 损失来训练预测模型？

因为预测模型是一个**分类模型**，在离散令牌空间进行预测。每个 s1/s2 令牌有 1,024 种选择，这是一个多分类问题，交叉熵是标准选择。MSE 用于分词器的重建损失（连续空间）。

### Q: DependencyAwareLayer 为什么使用交叉注意力而不是简单拼接？

简单拼接（将 s1 嵌入与 Transformer 输出拼接后线性映射）是一种静态融合。交叉注意力允许 s2 的预测**动态地**关注与 s1 最相关的上下文信息——不同的 s1 值会导致不同的注意力模式，实现更灵活的条件依赖。

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
