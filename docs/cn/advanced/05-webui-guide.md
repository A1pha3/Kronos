# Web UI 使用指南 ⭐⭐

> **目标读者**：希望通过浏览器界面使用 Kronos 进行 K 线预测的用户
> **前置要求**：已完成[安装与环境配置](../getting-started/01-installation.md)

---

## 学习目标

以下内容覆盖 Web UI 的安装、启动、API 端点和常见问题：

- [ ] 能启动 Kronos Web 界面并通过浏览器进行 K 线预测
- [ ] 理解 Web UI 的数据加载、模型选择和预测参数配置流程
- [ ] 能排查 Web UI 启动和使用中的常见问题

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

### 数据目录配置

Web UI 后端（`webui/app.py`）通过 `load_data_files()` 函数扫描数据目录。该目录的路径是**硬编码**的，指向项目根目录下的 `data/` 文件夹：

```python
# webui/app.py 中的关键代码
def load_data_files():
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # 项目根目录
        'data'  # 硬编码的 data/ 子目录
    )
```

也就是说，数据目录的实际路径为 `<项目根目录>/data/`。如果你希望使用自定义路径存放数据文件，需要修改 `app.py` 中 `load_data_files()` 函数内的 `data_dir` 变量。

> **为什么数据目录不放在 Web UI 目录内？** 因为 `data/` 目录同时被其他脚本（如 `examples/` 下的示例脚本）引用，放在项目根目录下可以实现数据共享，避免重复存放。

---

## 安全注意事项

Web UI 基于 Flask 开发服务器，设计目标是**本地开发和演示**，不适合生产环境部署。

| 场景 | 是否合适 | 原因 |
|------|---------|------|
| 本地单机使用 | 合适 | 默认配置即为此场景设计 |
| 团队内部分享 | 需注意 | 绑定 `0.0.0.0` 后局域网可访问，无认证机制 |
| 公网部署 | 不合适 | 无 HTTPS、无认证、无速率限制 |

#### debug 模式的安全风险

`run.py` 默认以 `debug=True` 启动 Flask（见源码 `app.run(debug=True, ...)`）。debug 模式启用了两个危险功能：

1. **交互式调试器（Werkzeug Debugger）**：当请求触发异常时，浏览器会展示一个交互式 Python 终端，攻击者可以借此执行任意 Python 代码——包括读取文件、删除数据、甚至接管整个系统。
2. **自动重载器**：监听源码文件变化并自动重启服务。在本地开发时很方便，但在网络可访问的场景下增加了攻击面。

> **为什么默认开启 debug？** Web UI 的定位是本地研究工具，debug 模式能在出错时直接在浏览器中查看 Python 堆栈信息，方便排查问题。如果你需要将服务暴露到局域网，请务必在 `run.py` 和 `app.py` 中将 `debug=True` 改为 `debug=False`。

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
[
  {"name": "XSHG_5min_600977.csv", "path": "/path/to/data/XSHG_5min_600977.csv", "size": "128.5 KB"},
  {"name": "000001_daily.csv", "path": "/path/to/data/000001_daily.csv", "size": "256.3 KB"}
]
```

### 2. 加载数据文件

```
POST /api/load-data
```

**请求体**：

```json
{
  "file_path": "/path/to/data/XSHG_5min_600977.csv"
}
```

**响应示例**：

```json
{
  "success": true,
  "data_info": {
    "rows": 1000,
    "columns": ["open", "high", "low", "close", "volume", "amount", "timestamps"],
    "start_date": "2024-01-02T00:00:00",
    "end_date": "2024-06-30T00:00:00",
    "price_range": {"min": 5.12, "max": 18.76},
    "prediction_columns": ["open", "high", "low", "close", "volume"],
    "timeframe": "5 minutes"
  },
  "message": "Successfully loaded data, total 1000 rows"
}
```

后端执行的数据验证（源码位于 `webui/app.py` 的 `load_data_file()` 函数）：

- 检查文件格式是否为 `.csv` 或 `.feather`
- 检查必填列（`open`、`high`、`low`、`close`）是否存在
- 通过 `pd.to_numeric(..., errors='coerce')` 先将 OHLC 列强制转为数值类型，再对 `volume`/`amount`（如存在）做同样转换，无法转换的值变为 `NaN`
- 调用 `df.dropna()` 移除所有包含 `NaN` 值的行（包括因数值转换失败产生的 NaN）

#### 时间戳列的自动识别

后端按以下优先级自动识别时间戳列：

1. `timestamps` 列（优先使用）
2. `timestamp` 列（重命名为 `timestamps`）
3. `date` 列（重命名为 `timestamps`）
4. 如果以上列均不存在，自动生成从 `2024-01-01` 起每小时一行的时间戳

> **提示**：如果你的数据文件使用 `date` 或 `datetime` 作为时间列名，Web UI 会自动处理，无需手动重命名。但建议统一使用 `timestamps` 列名以避免歧义。

#### 数据长度不足时的错误

预测时如果数据行数少于 `lookback` 参数（默认 400），后端会返回错误：

```json
{"error": "Insufficient data length, need at least 400 rows"}
```

解决方法：减小 `lookback` 参数值，或使用更长的数据文件。

### 3. 加载模型

```
POST /api/load-model
```

**请求体**：

```json
{
  "model_key": "kronos-small"
}
```

可选的 `model_key` 值（由 `app.py` 中 `AVAILABLE_MODELS` 字典定义）：

| `model_key` | 模型 ID | 分词器 | 上下文长度 | 参数量 |
|-------------|---------|--------|-----------|--------|
| `kronos-mini` | `NeoQuasar/Kronos-mini` | `Kronos-Tokenizer-2k` | 2048 | 4.1M |
| `kronos-small` | `NeoQuasar/Kronos-small` | `Kronos-Tokenizer-base` | 512 | 24.7M |
| `kronos-base` | `NeoQuasar/Kronos-base` | `Kronos-Tokenizer-base` | 512 | 102.3M |

> **注意**：Web UI 当前不支持 `Kronos-large`（未开源）。`mini` 模型使用专用的 `Kronos-Tokenizer-2k` 分词器，其余模型共用 `Kronos-Tokenizer-base`。Web UI 会根据所选模型自动加载对应的分词器和上下文长度配置。

#### 模型加载后的内存行为

Web UI 使用全局变量（`tokenizer`、`model`、`predictor`）存储已加载的模型实例，这意味着：

- **同模型重复请求不会重新下载**：HuggingFace Hub 的缓存机制会将模型权重缓存到本地磁盘（通常在 `~/.cache/huggingface/hub/`）。首次加载后，后续加载同一模型直接从缓存读取，无需再次下载。
- **切换模型会覆盖内存实例**：由于全局变量只有一个，加载新模型（如从 `small` 切换到 `base`）时，之前的模型实例会被 Python 垃圾回收，释放显存/内存。这保证了内存占用不会随切换操作累积增长。
- **重新加载同一模型也会重建实例**：即使请求加载的模型与当前内存中的相同，后端仍会重新执行 `from_pretrained()` 并创建新的 `KronosPredictor` 对象。这会带来短暂的加载延迟，但不会重新下载权重。

> **为什么采用这种设计？** Flask 开发服务器是单进程单线程的，全局状态在单用户本地使用场景下是最简单的实现方式。如需支持多模型并行服务，需要引入模型池或异步架构。

### 4. 执行预测

```
POST /api/predict
```

**请求体**：

```json
{
  "file_path": "/path/to/data/XSHG_5min_600977.csv",
  "lookback": 400,
  "pred_len": 120,
  "temperature": 1.0,
  "top_p": 0.9,
  "sample_count": 1
}
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file_path` | string | — | 数据文件的完整路径 |
| `lookback` | int | 400 | 历史窗口长度 |
| `pred_len` | int | 120 | 预测步数 |
| `temperature` | float | 1.0 | 温度参数 |
| `top_p` | float | 0.9 | 核采样阈值 |
| `sample_count` | int | 1 | 采样次数 |

### 5. 获取可用模型

```
GET /api/available-models
```

返回所有可用的预训练模型列表。

**响应示例**：

```json
{
  "models": {
    "kronos-mini": {
      "name": "Kronos-mini",
      "model_id": "NeoQuasar/Kronos-mini",
      "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
      "context_length": 2048,
      "params": "4.1M",
      "description": "Lightweight model, suitable for fast prediction"
    },
    "kronos-small": { "..." : "..." },
    "kronos-base": { "..." : "..." }
  },
  "model_available": true
}
```

### 6. 检查模型状态

```
GET /api/model-status
```

返回当前已加载的模型信息。

**响应示例（模型已加载）**：

```json
{
  "available": true,
  "loaded": true,
  "message": "Kronos model loaded and available",
  "current_model": {
    "name": "Kronos",
    "device": "cpu"
  }
}
```

**响应示例（模型未加载）**：

```json
{
  "available": true,
  "loaded": false,
  "message": "Kronos model available but not loaded"
}
```

**响应示例（模型库不可用）**：

```json
{
  "available": false,
  "loaded": false,
  "message": "Kronos model library not available, please install related dependencies"
}
```

---

## 预测结果

### 图表展示

预测完成后，Web UI 使用 Plotly 生成交互式图表：

- **历史 K 线**：以绿色（涨）和红色（跌）显示
- **预测 K 线**：以浅绿色（涨）和橙红色（跌）显示
- **实际 K 线**（如果存在对比数据）：以橙色（涨）和红色（跌）显示

#### Plotly 交互操作

图表由 Plotly.js 渲染，支持以下交互操作：

| 操作 | 鼠标动作 | 作用 |
|------|---------|------|
| **悬停查看数值** | 将鼠标移至任意 K 线上 | 显示该 K 线的开盘价、最高价、最低价、收盘价 |
| **框选缩放** | 按住鼠标左键拖拽选定区域 | 放大选定的时间范围，便于观察局部细节 |
| **平移** | 在缩放状态下拖拽图表 | 左右滑动查看不同时间段的数据 |
| **双击恢复** | 双击图表区域 | 恢复到初始的全局视图 |
| **保存图片** | 点击右上角工具栏的相机图标 | 将当前图表导出为 PNG 图片 |
| **切换数据系列** | 点击图例中的系列名称 | 显示/隐藏对应的数据系列（历史、预测、实际） |

> **实用技巧**：当预测步数较多（如 120 步）时，建议先框选放大到预测区域的起始位置，观察预测 K 线与历史 K 线的衔接是否平滑——这是判断预测质量的一个直观指标。

### 结果保存

预测结果自动保存为 JSON 文件：

```
webui/prediction_results/
├── prediction_<timestamp>.json    # 预测结果
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
2. 文件位于数据目录下（默认为项目根目录的 `data/`）
3. CSV 文件包含必填列：`open`、`high`、`low`、`close`
4. CSV 文件的 `timestamps` 列能被 `pd.to_datetime` 正确解析

### Q: 预测速度很慢？

**A**: 检查以下设置：

- 如果有 GPU，在加载模型时通过 API 请求体传入 `"device": "cuda:0"`（`POST /api/load-model` 的 `device` 参数）
- 使用较小的模型（如 `Kronos-mini` 或 `Kronos-small`）
- 减小 `sample_count`（设为 1 最快）
- 减小 `pred_len`（预测步数）

> **注意**：模型加载时的设备选择是通过 `/api/load-model` 请求体中的 `device` 字段控制的，不是修改源码中的常量。默认值为 `"cpu"`。

### Q: 预测结果中出现 "Kronos model not loaded" 错误？

**A**: 后端在启动时检测 `model` 模块的可用性。如果导入失败，会设置 `MODEL_AVAILABLE = False`，此时所有预测请求都会返回此错误。确保：

1. 项目根目录在 Python 搜索路径中（`run.py` 会自动处理）
2. `model/` 目录完整且 `torch`、`huggingface_hub` 已安装
3. 已通过 `POST /api/load-model` 成功加载模型（可先调用 `GET /api/model-status` 确认状态）

---

## 动手练习

### 练习 1：使用 Web UI 预测并查看结果

1. 启动 Web UI：`python webui/run.py`
2. 在浏览器中选择项目自带的测试数据 `XSHG_5min_600977.csv`
3. 选择 `Kronos-small` 模型
4. 设置 `lookback=200`、`pred_len=60`、`temperature=1.0`
5. 执行预测，观察生成的图表

**验证方法**：图表应显示历史 K 线段和预测 K 线段。预测段应紧接历史段末尾，无时间间隙。用 Plotly 的悬停功能检查预测 K 线的 OHLC 值是否在合理范围内（不出现负数或超过历史均价 10 倍的异常值）。

### 练习 2：对比不同温度参数的预测效果

在同一数据集上，分别使用 `temperature=0.3` 和 `temperature=1.5` 执行两次预测，对比图表中预测曲线的波动程度。

**验证方法**：`temperature=1.5` 的预测曲线波动应明显大于 `temperature=0.3`。如果两者差异不大，尝试增大对比（如 `T=0.1` vs `T=2.0`）。

### 练习 3：通过 API 端点获取预测结果

不使用浏览器界面，直接用 Python 调用 API 端点，验证 Web UI 的 API 层独立可用：

```python
import requests
import json

# 确认 Web UI 正在运行
resp = requests.get("http://localhost:7070/api/available-models")
print("可用模型:", resp.json())

# 获取数据文件列表
resp = requests.get("http://localhost:7070/api/data-files")
print("数据文件:", resp.json())

# 执行预测（需要先加载数据和模型）
# 步骤 1：加载数据文件
resp = requests.post("http://localhost:7070/api/load-data", json={"data_file": "XSHG_5min_600977.csv"})
print("数据加载:", resp.json())

# 步骤 2：加载模型
resp = requests.post("http://localhost:7070/api/load-model", json={"model_name": "Kronos-small"})
print("模型加载:", resp.json())

# 步骤 3：执行预测
resp = requests.post("http://localhost:7070/api/predict", json={
    "lookback": 200,
    "pred_len": 60,
    "temperature": 1.0
})
result = resp.json()
print("预测状态:", result.get("status", result.get("error", "unknown")))
```

**验证方法**：如果三个步骤依次返回成功响应，说明 API 层完整可用。如果返回 500 错误，检查数据文件路径是否在正确的 `data/` 目录下，以及模型是否加载成功（调用 `GET /api/model-status` 确认）。

---

## 自测清单

- [ ] 能独立启动 Web UI 并在浏览器中访问
- [ ] 能说明 Web UI 的 6 个 API 端点及其功能
- [ ] 知道如何更换端口号和指定 GPU 设备
- [ ] 能解释数据文件需要满足的格式要求
- [ ] 知道预测结果保存在哪里以及包含什么内容

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

