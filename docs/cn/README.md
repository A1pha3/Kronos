# Kronos 中文文档

> Kronos —— 首个面向金融蜡烛图（K 线）数据的开源基础模型，被 AAAI 2026 接收。

本文档体系覆盖从入门到专家的完整学习路径，所有内容均基于项目源码撰写，确保技术准确。

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
| [模型对比与选型](advanced/07-model-comparison.md) | ⭐⭐ | 四个模型的参数、速度、显存对比与选型建议 |
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
