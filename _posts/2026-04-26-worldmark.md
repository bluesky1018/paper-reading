---
layout: post
title: "WorldMark：交互式视频世界模型统一评测基准"
date: 2026-04-26
categories: [论文解读, 视频生成, 世界模型]
tags: [世界模型, 基准测试, 视频生成, 交互式生成, 评测]
---

> 📄 **论文**：WorldMark: A Unified Benchmark Suite for Interactive Video World Models
> 🔗 **arXiv**：[2604.21686](https://arxiv.org/abs/2604.21686)
> 🏢 **机构**：多机构合作（Yukang Feng、Kang He 等）

## 一句话总结
WorldMark 是首个针对交互式图像到视频（I2V）世界模型的标准化评测框架，通过统一动作接口、500 个测试案例和多维评估工具，实现了跨模型的公平比较。

## 背景与问题

Genie、YUME、HY-World、Matrix-Game 等交互式视频生成模型正在快速发展，但每个模型都在自己的私有场景和轨迹上进行评估，使得跨模型的公平比较几乎不可能实现。现有公开基准提供了轨迹误差、美学分数和基于 VLM 的判断等有用指标，但没有一个提供跨模型可比较所需的标准化测试条件——相同场景、相同动作序列和统一控制接口。

更深层的问题在于，不同世界模型采用完全异构的控制格式：有的使用文本描述（caption prompts），有的使用姿态参数（pose parameters），有的使用游戏手柄信号（gamepad signals），有的使用动作向量（action vectors）。这种接口异构性使得系统性比较研究面临根本性挑战。

## 核心方法

### WorldMark 框架架构

WorldMark 由五个核心组件构成：

1. **评估维度套件（Evaluation Dimension Suite）**：覆盖视觉质量（Visual Quality）、控制对齐（Control Alignment）和世界一致性（World Consistency）三大维度，共 8 个评估指标

2. **图像套件（Image Suite）**：500 个测试案例，包含第一人称和第三人称视角，涵盖自然、城市、室内等多样场景，以及逼真和风格化参考场景

3. **动作套件（Action Suite）**：15 条标准化动作序列，从基础平移旋转到组合轨迹，复杂度递增

4. **统一动作接口（Unified Action Interface）**：将共享的 WASD+左右偏转动作词汇表翻译为每个模型的原生控制格式

5. **模块化评估工作流（Evaluation Workflow）**：四阶段流水线将以上组件整合为完整的评测流程

![WorldMark Overview](https://arxiv.org/html/2604.21686v1/x1.png)
*图1：WorldMark 概览。统一动作接口将共享动作词汇翻译为各模型原生控制格式，实现公平跨模型比较*

### 图像套件

图像套件涵盖多样化的场景和风格，每个场景均在第一人称视角和生成的第三人称视角下呈现，确保评测的广度覆盖。

![Image Suite](https://arxiv.org/html/2604.21686v1/x2.png)
*图2：图像套件概览，涵盖多样场景和风格，第一/第三人称视角均有覆盖*

### 动作套件

15 条标准化动作序列设计了从基础到复杂的各类轨迹。

![Action Sequences](https://arxiv.org/html/2604.21686v1/x3.png)
*图3：15 条标准化动作序列，从基本的平移旋转到组合及循环轨迹*

### 上下文感知动作选择

利用 VLM 分析初始图像以识别物理约束，并为每个测试案例选择合理的动作序列。

![Context-aware Action Selection](https://arxiv.org/html/2604.21686v1/x4.png)
*图4：上下文感知动作选择——VLM 分析图像物理约束并筛选合适的动作序列*

## 实验结果

### 评测模型

WorldMark 对六个代表性的开源和商业交互式 I2V 世界模型进行了评测：
- 五个开源模型：YUME 1.5、Matrix-Game 2.0、HY-World 1.5、HY-GameCraft、Open-Oasis
- 一个商业模型：Google Genie 3

### 关键发现

| 发现 | 描述 |
|------|------|
| 视觉质量与世界一致性不相关 | 视频看起来美观并不意味着物理行为合理 |
| 控制对齐≠生成质量 | 精确的控制响应不代表总体生成质量高 |
| 第三人称生成仍是开放挑战 | 最差情况下旋转误差相比第一人称设置退化近一个数量级 |

### Spearman 相关性分析

通过相关性分析揭示了不同评估维度之间的关系，发现视觉质量指标与世界一致性指标之间存在显著的不相关性，这对模型开发有重要指导意义。

![Spearman Correlation](https://arxiv.org/html/2604.21686v1/figures/spearman_correlation.png)
*图：各评估维度之间的 Spearman 相关性热图*

## 总结

WorldMark 作为首个标准化的交互式 I2V 世界模型评测基准，通过三大支柱解决了该领域的评测碎片化问题：统一动作接口、标准化测试套件和多维评估工具。其开源设计允许仅通过添加一个动作映射适配器即可将新模型纳入评测体系，极大降低了参与门槛。

研究揭示的关键结论——视觉质量与世界一致性不相关、第三人称视角生成仍面临重大挑战——为下一代世界模型的研发指明了重点攻关方向。WorldMark 的发布将推动该领域建立统一评估标准，促进更系统化的研究进展。
