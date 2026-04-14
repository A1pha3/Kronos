# Web UI 使用指南 ⭐⭐

> **目标读者**：希望通过浏览器界面使用 Kronos 进行 K 线预测的用户
> **前置要求**：已完成[安装与环境配置](../getting-started/01-installation.md)

---

## 学习目标

阅读本文后，你将能够：

- [ ] 启动 Kronos Web 界面并通过浏览器进行 K 线预测
- [ ] 理解 Web UI 的数据加载、模型选择和预测参数配置流程
- [ ] 排查 Web UI 启动和使用中的常见问题

---

## 概述

Kronos 提供了一个基于 Flask 的 Web 界面，让你可以通过浏览器加载 K 线数据、选择模型、执行预测并查看交互式图表，无需编写代码。

Web UI 的核心功能：

- 从本地 CSV/Feather 文件加载 K 线数据
- 选择预训练模型（mini / small / base / large）
- 配置预测参数（历史窗口、预测步数、温度等）
- 生成 Plotly 交互式图表
- 将预测结果保存为 JSON 文件

---

## 环境准备

### 安装额外依赖

```bash
pip install -r webui/requirements.txt
```

额外依赖包含：

| 包名 | 版本 | 作用 |
|------|------|------|
| `flask` | 2.3.3 | Web 框架 |
| `flask-cors` | 4.0.0 | 跨域请求支持 |
| `plotly` | 5.17.0 | 交互式图表生成 |

### 准备数据文件

将 K 线数据文件（CSV 或 Feather 格式）放入数据目录。文件需要包含以下列：

- **必填**：`open`、`high`、`low`、`close`
- **可选**：`volume`、`amount`
- **时间戳**：`timestamps` 列（`pd.to_datetime` 可解析的格式）

---

## 安全注意事项

Web UI 基于 Flask 开发服务器，设计目标是**本地开发和演示**，不适合生产环境部署。

| 场景 | 是否合适 | 原因 |
|------|---------|------|
| 本地单机使用 | 合适 | 默认配置即为此场景设计 |
| 团队内部分享 | 需注意 | 绑定 `0.0.0.0` 后局域网可访问，无认证机制 |
| 公网部署 | 不合适 | 无 HTTPS、无认证、无速率限制 |

如果你需要将 Web UI 暴露到局域网，建议：

1. 确保运行在可信网络环境中
2. 通过防火墙限制访问 IP 范围
3. 使用反向代理（如 Nginx）添加 HTTPS 和认证

---

## 启动服务

```bash
python webui/run.py
```

启动后，浏览器会自动打开 `http://localhost:7070`。

### 启动过程

`run.py` 在启动时会执行以下检查：

1. **依赖检查**：检测 Flask、Plotly 等依赖是否已安装，缺失时提示安装命令
2. **模型可用性检查**：验证 `torch`、`huggingface_hub` 是否可用
3. **服务启动**：启动 Flask 服务器，默认端口 7070

### 自定义端口

如果 7070 端口被占用，可以修改 `webui/run.py` 中的端口：

```python
# 在 run.py 文件末尾找到如下行，修改 port 参数
app.run(host='0.0.0.0', port=8080)  # 改为 8080
```

> **注意**：`debug=True` 仅用于本地开发调试，不应在生产或局域网共享场景中使用。Flask 开发服务器在 debug 模式下会执行任意代码，存在安全风险。

---

## API 端点

Web UI 通过 REST API 与后端交互。以下列出所有可用端点：

### 1. 获取数据文件列表

```
GET /api/data-files
```

返回可用的数据文件列表（自动扫描 `data_dir` 目录下的 CSV 和 Feather 文件）。

**响应示例**：

```json
{
  "files": ["XSHG_5min_600977.csv", "000001_daily.csv"]
}
```

### 2. 加载数据文件

```
POST /api/load-data
```

**请求体**：

```json
{
  "filename": "XSHG_5min_600977.csv"
}
```

**响应示例**：

```json
{
  "success": true,
  "data": {
    "columns": ["open", "high", "low", "close", "volume", "amount"],
    "length": 1000,
    "start_date": "2024-01-02",
    "end_date": "2024-06-30"
  }
}
```

后端执行的数据验证（源码位于 `webui/app.py` 的 `load_data_file()` 函数）：

- 检查文件是否存在
- 检查必填列（`open`、`high`、`low`、`close`）是否存在
- 补齐可选列（`volume`、`amount` 缺失时填充 0）
- 检查 NaN 值

### 3. 加载模型

```
POST /api/load-model
```

**请求体**：

```json
{
  "model_name": "small"
}
```

可选的 `model_name` 值：`mini`、`small`、`base`、`large`。

> **注意**：`mini` 模型使用 `Kronos-Tokenizer-2k` 分词器，其他模型使用 `Kronos-Tokenizer-base`。Web UI 会根据所选模型自动加载对应的分词器。

### 4. 执行预测

```
POST /api/predict
```

**请求体**：

```json
{
  "filename": "XSHG_5min_600977.csv",
  "lookback": 400,
  "pred_len": 120,
  "temperature": 1.0,
  "top_p": 0.9,
  "sample_count": 1,
  "model_name": "small"
}
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `filename` | string | — | 数据文件名 |
| `lookback` | int | 400 | 历史窗口长度 |
| `pred_len` | int | 120 | 预测步数 |
| `temperature` | float | 1.0 | 温度参数 |
| `top_p` | float | 0.9 | 核采样阈值 |
| `sample_count` | int | 1 | 采样次数 |
| `model_name` | string | "small" | 模型名称 |

### 5. 获取可用模型

```
GET /api/available-models
```

返回所有可用的预训练模型列表。

### 6. 检查模型状态

```
GET /api/model-status
```

返回当前已加载的模型信息。

---

## 预测结果

### 图表展示

预测完成后，Web UI 使用 Plotly 生成交互式图表：

- **历史 K 线**：以蓝色显示
- **预测 K 线**：以红色显示
- 支持缩放、平移、悬停查看数值等交互操作

### 结果保存

预测结果自动保存为 JSON 文件：

```
outputs/
├── pred_result_<timestamp>.json    # 预测结果
```

JSON 文件包含完整的 OHLCV 预测数据和元信息（模型名称、参数配置等）。

---

## 内部工作流程

Web UI 后端的完整处理流程（源码位于 `webui/app.py`）：

```
1. 用户选择数据文件 → POST /api/load-data
   └─ 扫描数据目录、验证格式、返回数据概要

2. 用户选择模型 → POST /api/load-model
   └─ 从 HuggingFace Hub 下载（首次）并加载模型

3. 用户配置参数并预测 → POST /api/predict
   ├─ 加载并验证数据
   ├─ 构造 KronosPredictor
   ├─ 执行 predictor.predict()
   ├─ 生成 Plotly 图表
   └─ 保存结果 JSON

4. 返回图表和数据 → 浏览器渲染
```

### 并发访问与性能说明

Web UI 基于 Flask 开发服务器，按请求顺序处理——如果两个用户同时点击预测，第二个请求会排队等待第一个完成。对于本地单用户场景这不是问题，但在团队内部分享时需注意：

- 模型加载是全局状态——如果用户 A 加载了 `Kronos-small`，用户 B 再加载 `Kronos-base` 会覆盖前一个模型
- 预测是同步阻塞的——一次 120 步预测在 CPU 上可能需要数十秒，期间其他请求无法响应
- 大文件的加载（超过 10 万行 CSV）可能导致请求超时

---

## 常见问题

### Q: 启动时提示模块缺失？

**A**: 确保已安装 Web UI 的额外依赖：

```bash
pip install -r webui/requirements.txt
```

如果 `torch` 也缺失，先安装 PyTorch（参考[安装指南](../getting-started/01-installation.md)）。

### Q: 浏览器无法打开？

**A**: 检查以下几点：

1. 确认 Flask 服务已启动（终端应显示 `Running on http://...`）
2. 检查端口是否被占用（更换端口参考上面的"自定义端口"说明）
3. 手动在浏览器中访问 `http://localhost:7070`

### Q: 预测时报错 "No model loaded"？

**A**: 需要先通过界面加载模型，再执行预测。首次加载模型时需要从 HuggingFace Hub 下载权重，确保网络畅通。

### Q: 数据文件无法识别？

**A**: 确保：

1. 文件格式为 `.csv` 或 `.feather`
2. 文件位于数据目录下（默认为项目根目录的 `examples/data/`）
3. CSV 文件包含必填列：`open`、`high`、`low`、`close`
4. CSV 文件的 `timestamps` 列能被 `pd.to_datetime` 正确解析

### Q: 预测速度很慢？

**A**: 检查以下设置：

- 如果有 GPU，在 `webui/run.py` 或 `webui/app.py` 中设置 `device="cuda:0"`
- 使用较小的模型（如 `Kronos-small`）
- 减小 `sample_count`（设为 1 最快）
- 减小 `pred_len`（预测步数）

---

## 动手练习

### 练习 1：使用 Web UI 预测并查看结果

1. 启动 Web UI：`python webui/run.py`
2. 在浏览器中选择项目自带的测试数据 `XSHG_5min_600977.csv`
3. 选择 `Kronos-small` 模型
4. 设置 `lookback=200`、`pred_len=60`、`temperature=1.0`
5. 执行预测，观察生成的图表

**验证方法**：如果图表中出现蓝色历史段和红色预测段，且预测段紧接历史段末尾，说明预测成功。

### 练习 2：对比不同温度参数的预测效果

在同一数据集上，分别使用 `temperature=0.3` 和 `temperature=1.5` 执行两次预测，对比图表中预测曲线的波动程度。

**验证方法**：`temperature=1.5` 的预测曲线波动应明显大于 `temperature=0.3`。

---

## 自测清单

- [ ] 我能独立启动 Web UI 并在浏览器中访问
- [ ] 我能说明 Web UI 的 6 个 API 端点及其功能
- [ ] 我知道如何更换端口号和指定 GPU 设备
- [ ] 我能解释数据文件需要满足的格式要求
- [ ] 我知道预测结果保存在哪里以及包含什么内容

---

## 下一步

| 推荐内容 | 难度 | 说明 |
|---------|------|------|
| [KronosPredictor 使用指南](../core-concepts/04-predictor.md) | ⭐⭐ | 了解更多参数调节细节 |
| [批量预测指南](03-batch-prediction.md) | ⭐⭐⭐ | 多序列并行预测 |
| [A 股市场预测实战](04-cn-markets.md) | ⭐⭐⭐ | A 股完整预测示例 |

## 相关文档

- **前置**：[安装指南](../getting-started/01-installation.md) — 环境配置
- **前置**：[KronosPredictor 使用指南](../core-concepts/04-predictor.md) — 了解预测参数含义
- **并行**：[批量预测指南](03-batch-prediction.md) — 编程方式批量预测
- **实战**：[A 股市场预测实战](04-cn-markets.md) — A 股完整预测流程
- **参考**：[模型选型指南](07-model-comparison.md) — mini / small / base / large 对比

---
**文档元信息**
难度：⭐⭐ | 类型：进阶指南 | 预计阅读时间：15 分钟
更新日期：2026-04-11
