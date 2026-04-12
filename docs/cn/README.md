# Kronos 中文文档

> Kronos 是一个面向金融蜡烛图（K 线）数据的开源基础模型，被 AAAI 2026 接收。根据仓库根目录 [README](../../README.md)，项目采用”分词器 + 自回归模型”的两阶段框架，并提供多种预训练模型与微调脚本。

本文档体系覆盖从入门到源码分析的完整学习路径，共 24 篇文档，按难度从 1 星到 4 星递进。文档中的事实性描述以仓库中的 `README.md`、`model/`、`finetune/`、`finetune_csv/`、`webui/` 代码为准；凡是带有经验性建议的内容，均明确其适用边界，不把推测写成事实。所有行号引用基于源码实际位置，代码示例均可运行。

---

## 学习路径

### 用户路径：掌握 Kronos 的使用

从安装配置到独立完成各类预测任务，适合希望使用 Kronos 进行金融预测的用户。

| 文档 | 难度 | 内容 |
|------|------|------|
| [安装与环境配置](getting-started/01-installation.md) | ⭐ | Python 环境、依赖安装、设备检测、环境验证 |
| [快速开始：第一个预测](getting-started/02-quickstart.md) | ⭐ | 10 分钟完成第一次 K 线预测，理解完整预测流程 |
| [数据准备指南](getting-started/03-data-preparation.md) | ⭐ | 数据格式要求、列名规范、时间戳处理 |
| [项目总览与核心概念](core-concepts/01-overview.md) | ⭐⭐ | 两阶段框架、模型动物园、核心 API 全景 |
| [KronosTokenizer 详解](core-concepts/02-tokenizer.md) | ⭐⭐ | 分词器的编码/解码原理、参数配置、使用方法 |
| [Kronos 模型详解](core-concepts/03-model.md) | ⭐⭐ | 自回归 Transformer 的前向推理、双头输出、条件解码 |
| [KronosPredictor 使用指南](core-concepts/04-predictor.md) | ⭐⭐ | 高层预测接口、参数调节、批量预测 |
| [层级令牌体系](core-concepts/05-hierarchical-tokens.md) | ⭐⭐ | s1/s2 双层令牌的设计原理与工作机制 |
| [Qlib 微调指南](advanced/01-finetune-qlib.md) | ⭐⭐⭐ | 基于 Qlib 的中国 A 股微调流程（含 DDP 分布式训练） |
| [CSV 微调指南](advanced/02-finetune-csv.md) | ⭐⭐⭐ | 通用 CSV 数据微调流程、YAML 配置详解 |
| [批量预测指南](advanced/03-batch-prediction.md) | ⭐⭐⭐ | 多时间序列并行预测的实践方法 |
| [A 股市场预测实战](advanced/04-cn-markets.md) | ⭐⭐⭐ | 使用 akshare 获取 A 股数据并预测未来走势 |
| [Web UI 使用指南](advanced/05-webui-guide.md) | ⭐⭐ | 通过浏览器界面进行 K 线预测 |
| [使用场景与实战案例](advanced/06-use-cases.md) | ⭐⭐ | 多市场预测、多场景模拟、波动率估计 |
| [模型对比与选型](advanced/07-model-comparison.md) | ⭐⭐ | 四个模型的参数、上下文长度、分词器搭配与选型建议 |
| [开发扩展指南](advanced/08-development-guide.md) | ⭐⭐⭐ | 自定义时间特征、数据集、采样策略、架构修改 |
| [系统架构分析](architecture/01-system-architecture.md) | ⭐⭐⭐⭐ | 从数据流视角理解完整系统架构与模块协作 |
| [BSQ 量化算法原理](architecture/02-bsq-algorithm.md) | ⭐⭐⭐⭐ | Binary Spherical Quantization 的数学原理与实现 |
| [Transformer 设计分析](architecture/03-transformer-design.md) | ⭐⭐⭐⭐ | RoPE 位置编码、注意力机制、前馈网络的设计决策 |
| [源码走读](architecture/04-source-code-walkthrough.md) | ⭐⭐⭐⭐ | 逐模块阅读核心源码，理解每一行的设计意图 |

### 开发者路径：理解实现与贡献代码

适合希望深入理解 Kronos 内部实现或进行二次开发的开发者。建议先完成用户路径的 ⭐⭐ 级别文档。

| 步骤 | 文档 | 目标 |
|------|------|------|
| 1 | [系统架构分析](architecture/01-system-architecture.md) | 理解三层架构与模块协作 |
| 2 | [源码走读](architecture/04-source-code-walkthrough.md) | 逐模块阅读核心源码 |
| 3 | [BSQ 量化算法原理](architecture/02-bsq-algorithm.md) | 理解量化的数学基础 |
| 4 | [Transformer 设计分析](architecture/03-transformer-design.md) | 理解各组件的设计决策 |
| 5 | [开发扩展指南](advanced/08-development-guide.md) | 学习如何添加自定义功能 |

### 进阶路径：掌握算法原理与架构设计

适合希望理解"为什么这样设计"的研究者和架构师。建议先完成用户路径的 ⭐⭐⭐ 级别文档。

| 步骤 | 文档 | 目标 |
|------|------|------|
| 1 | [层级令牌体系](core-concepts/05-hierarchical-tokens.md) | 理解 s1/s2 的设计动机 |
| 2 | [BSQ 量化算法原理](architecture/02-bsq-algorithm.md) | 掌握量化的数学推导 |
| 3 | [Transformer 设计分析](architecture/03-transformer-design.md) | 评估组件替换的影响 |
| 4 | [系统架构分析](architecture/01-system-architecture.md) | 从全局视角理解设计权衡 |
| 5 | [源码走读](architecture/04-source-code-walkthrough.md) | 深入实现细节 |

---

## 快速入口

不知道从哪里开始？根据你的目标选择：

| 我想... | 推荐阅读 | 预计时间 |
|---------|---------|---------|
| 快速体验 Kronos 预测 | [安装](getting-started/01-installation.md) → [快速开始](getting-started/02-quickstart.md) | 25 分钟 |
| 用 Kronos 预测 A 股 | [快速开始](getting-started/02-quickstart.md) → [A 股实战](advanced/04-cn-markets.md) | 40 分钟 |
| 选择合适的模型 | [模型对比与选型](advanced/07-model-comparison.md) | 10 分钟 |
| 用自有数据微调 | [CSV 微调](advanced/02-finetune-csv.md) | 30 分钟 |
| 理解 Kronos 为什么这样设计 | [项目总览](core-concepts/01-overview.md) → [系统架构](architecture/01-system-architecture.md) | 45 分钟 |
| 通过浏览器预测（不写代码） | [Web UI 指南](advanced/05-webui-guide.md) | 15 分钟 |
| 解决报错 | [错误排查](references/troubleshooting.md) | 按需 |

---

## 参考资源

| 文档 | 说明 |
|------|------|
| [术语表](references/glossary.md) | 中英文术语对照与定义 |
| [常见问题](references/faq.md) | 按类别组织的常见问题解答 |
| [常见错误排查](references/troubleshooting.md) | OOM、NaN、维度错误等排查指南 |

---

## 阅读建议

- **纯用户**：按 ⭐ → ⭐⭐ → ⭐⭐⭐ 顺序阅读用户路径即可
- **想微调模型**：完成 ⭐⭐ 级别后，直接进入对应的微调指南
- **想理解原理**：完成 ⭐⭐ 级别后，阅读架构分析系列（⭐⭐⭐⭐）
- **想贡献代码**：完成全部用户路径后，从源码走读（⭐⭐⭐⭐）开始

---

## 文档原则

- **事实优先**：参数量、上下文长度、接口签名、默认值等内容以源码和根目录 `README.md` 为准
- **推断显式标注**：当文档需要解释设计动机时，会尽量写成”从实现可以看出”或”这是一种合理解释”，避免把未公开论证的内容写成结论
- **示例可落地**：示例代码优先复用仓库已有接口和路径，不额外发明仓库中不存在的能力
- **学习可递进**：重要文档尽量包含学习目标、阅读前提、常见误区和下一步阅读建议
- **源码可追溯**：行号引用标注具体文件和行号（如 `kronos.py:270`），方便对照源码理解
- **术语一致**：跨文档使用统一的中文译法，首次出现时标注英文原文

---

## 关键事实速查

以下是 Kronos 最常被查询的技术事实，供快速参考：

| 事实 | 值 |
|------|-----|
| 输入数据列 | `open`, `high`, `low`, `close`（必填）, `volume`, `amount`（可选） |
| 标准化方式 | 实例级 z-score + clip（默认 clip=5） |
| 最大上下文 | small/base/large: 512, mini: 2048 |
| 分词器搭配 | mini 使用 `Kronos-Tokenizer-2k`，其他使用 `Kronos-Tokenizer-base` |
| 令牌维度 | s1_bits=10, s2_bits=10（所有预训练模型一致） |
| predict() 默认参数 | T=1.0, top_k=0, top_p=0.9, sample_count=1 |
| 预训练市场 | 45+ 全球交易所 |
| 模型 Hub 路径 | `NeoQuasar/Kronos-{mini,small,base,large}` |
| 回归测试 | `pytest tests/test_kronos_regression.py` |
| Web UI 启动 | `python webui/run.py`（端口 7070） |
- **源码可追溯**：行号引用标注具体文件和行号（如 `kronos.py:270`），方便对照源码理解
- **术语一致**：跨文档使用统一的中文译法，首次出现时标注英文原文
