---
layout: post
title: "Agents-A1：用 35B 智能体达到万亿参数级性能——Horizon 扩展新范式"
date: 2026-07-01
categories: [论文解读, 大语言模型]
tags: [Agent, MoE, Scaling, Long-Horizon, Knowledge Distillation, Training]
---

> 📄 **论文**：Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter Performance with a 35B Agent
> 🔗 **arXiv**：[2606.30616](https://arxiv.org/abs/2606.30616)
> 🏢 **机构**：Shanghai Artificial Intelligence Laboratory (Agents-A1 Team)

## 一句话总结

Agents-A1 是一个 35B MoE 智能体模型，通过扩展智能体 Horizon（而非参数规模），在多项长程任务基准上达到甚至超越万亿参数级模型（如 Kimi-K2.6、DeepSeek-V4-pro）的性能。

## 背景与问题

当前提升长程智能体性能主要有两条路线：

1. **扩大参数规模**：万亿参数级模型（如 GPT-4、Kimi-K2）通过海量参数内化推理模式和领域知识，效果强但成本极高，难以复现
2. **扩展 Horizon（本文路线）**：不增加参数，而是让中间决策过程显式可训练——通过知识获取、动作执行、观测解析和验证反馈构成可训练的监督信号

然而，Horizon 扩展路线面临两大瓶颈：
- **知识基础设施不足**：支持长程轨迹训练需要将外部知识与动作、观测有机结合
- **异构能力统一困难**：不同领域（代码、科学、网页等）的专业知识需要统一到单一可部署模型中

## 核心方法

### 知识-动作基础设施（KAG Infrastructure）

论文构建了**长程知识-动作基础设施（Knowledge-Action Graph, KAG）**，将外部知识、动作、观测和验证器输出连接成完整的智能体轨迹：

- 平均轨迹长度：**45K tokens**（远超现有数据集）
- 覆盖领域：代码工程、科学研究、数学推理、信息检索等
- 支持实时知识注入和工具调用验证

![Agents-A1 基准性能](https://arxiv.org/html/2606.30616v1/x2.png)
*图1：Agents-A1 与万亿参数级模型的基准性能对比*

### 三阶段训练方案

**阶段一：全域监督微调（Full-domain SFT）**
- 对基础模型进行跨领域智能体行为对齐
- 建立广泛的工具使用和规划能力基础

**阶段二：领域级教师模型训练**
- 为每个专业领域（代码、科学、数学等）训练专门的教师模型
- 捕获各领域的深层专业知识和推理模式

**阶段三：多教师域路由在线蒸馏（Multi-teacher Domain-routed Distillation）**
- 提出 **显著词汇对齐（Salient Vocabulary Alignment）** 机制提升知识迁移效率
- 通过域路由将六个异构领域统一到一个可部署的学生模型中

![训练框架](https://arxiv.org/html/2606.30616v1/figures/train_framework.png)
*图2：Agents-A1 三阶段训练框架*

![KAG 基础设施](https://arxiv.org/html/2606.30616v1/figures/KAG_infra.png)
*图3：知识-动作图（KAG）基础设施架构*

## 实验结果

### 与万亿参数级模型对比

| 基准 | Kimi-K2.6 | DeepSeek-V4-pro | **Agents-A1 (35B)** |
|------|-----------|-----------------|---------------------|
| SEAL-0 | ~52 | ~50 | **56.4** |
| IFBench | ~76 | ~75 | **80.6** |
| HiPhO | ~43 | ~42 | **46.4** |
| FrontierScience-Olympiad | ~74 | ~76 | **79.0** |
| MolBench-Bind | ~52 | ~51 | **56.8** |
| SciCode | ~45 | ~46 | 44.3 |
| HLE | ~48 | ~49 | 47.6 |
| BrowseComp | ~73 | ~74 | **75.5** |

**关键发现**：
- 在 5/8 个长程任务基准上**超越**万亿参数级模型
- 参数量仅为对比模型的 1/30 左右
- 在科学研究和前沿推理任务上优势尤为明显

## 总结

Agents-A1 证明了"扩展 Horizon 而非参数"是实现长程智能体高性能的可行路线。通过精心构建的知识-动作基础设施和多教师蒸馏机制，35B MoE 模型在多项关键基准上达到甚至超越了参数量数十倍的模型。

这一工作为社区提供了一条实用的高性能智能体构建路径，不再需要依赖庞大的参数规模。局限性在于，知识-动作基础设施的构建本身需要大量工程投入，且对于极端复杂的推理任务（如 HLE），仍与顶级大参数模型存在差距。
