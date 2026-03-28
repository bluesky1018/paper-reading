---
layout: post
title: "SlopCodeBench：基准测试编程智能体在长时域迭代任务中的性能退化"
date: 2026-03-29
categories: [论文解读, 代码生成]
tags: [编程智能体, 代码基准测试, AI编程, 长时域任务, 智能体评估]
---

> 📄 **论文**：SlopCodeBench: Benchmarking How Coding Agents Degrade Over Long-Horizon Iterative Tasks
> 🔗 **arXiv**：[2603.24755](https://arxiv.org/abs/2603.24755)
> 🏢 **机构**：威斯康星大学麦迪逊分校

## 一句话总结
提出SlopCodeBench基准，系统评估编程智能体在长时域迭代编程任务中的性能退化（Slop）现象

## 背景与问题
AI编程助手（如GitHub Copilot、Cursor、Devin等）已在软件开发中广泛应用，但一个普遍观察到但未被系统研究的现象是：随着交互轮次增加，编程智能体的代码质量往往呈现下降趋势——这被称为"Slop"现象（指生成代码逐渐变得冗余、低质、偏离目标）。

现有的编程基准（如HumanEval、MBPP、SWE-bench）主要评估单轮或少轮交互下的代码生成能力，缺乏对长时域迭代任务中性能演变的系统分析。真实的软件开发工作流往往需要数十轮甚至数百轮的代码修改、调试和优化，这一场景下AI智能体的表现尚未被充分评估。

理解智能体在长时域任务中的性能退化原因，对于改进AI编程系统具有重要意义：是上下文窗口限制导致早期需求被遗忘？还是梯度积累导致错误传播？抑或是其他系统性因素？

## 核心方法
SlopCodeBench的设计包含以下核心要素：

**基准构建**：
- 设计了50个代表性的长时域编程任务，每个任务包含10-50轮迭代交互
- 任务涵盖：功能扩展、Bug修复循环、代码重构、性能优化四大类别
- 每轮交互设计了真实的用户需求变化（新增需求、澄清要求、纠错指令）

**性能退化量化指标**：
- **功能正确率曲线**：追踪每轮迭代后的测试通过率变化
- **代码质量指数**：综合评估代码复杂度、重复率、可维护性指标
- **需求遗忘率**：检测早期明确的需求在后续轮次中被违反的频率
- **Slop Score**：综合以上指标的性能退化综合评分

**评估的主流系统**：GPT-4o、Claude 3.5 Sonnet、Gemini 1.5 Pro、Llama 3.1 405B等


![Figure 2 : Solve rates and cost growth over problem progress. Left: Agents pass all core tests 1.4–13.3 × \times more often than the full checkpoint suite, and strict solve rates, which include regres...](https://arxiv.org/html/2603.24755/2603.24755v1/x1.png)
*图1：Figure 2 : Solve rates and cost growth over problem progress. Left: Agents pass all core tests 1.4–13.3 × \times more often than the full checkpoint suite, and strict solve rates, which include regres...*


![Figure 3 : Erosion and verbosity across problem progress for six representative models (three per provider). Both metrics increase monotonically.](https://arxiv.org/html/2603.24755/2603.24755v1/x2.png)
*图2：Figure 3 : Erosion and verbosity across problem progress for six representative models (three per provider). Both metrics increase monotonically.*


![Figure 4 : Mean verbosity and structural erosion across normalized trajectory progress for agent runs and human repositories. Shaded regions show 95% confidence intervals. Agent metrics climb monotoni...](https://arxiv.org/html/2603.24755/2603.24755v1/x3.png)
*图3：Figure 4 : Mean verbosity and structural erosion across normalized trajectory progress for agent runs and human repositories. Shaded regions show 95% confidence intervals. Agent metrics climb monotoni...*


![Figure 5 : Prompt strategy trajectories across two models. Each point shows the mean value at a normalized progress bin. Quality-aware prompts (Anti-Slop and Plan-First) lower the initial verbosity an...](https://arxiv.org/html/2603.24755/2603.24755v1/x4.png)
*图4：Figure 5 : Prompt strategy trajectories across two models. Each point shows the mean value at a normalized progress bin. Quality-aware prompts (Anti-Slop and Plan-First) lower the initial verbosity an...*


![Figure 6 : Mean continuous pass rates by test type over problem progress with bootstrap 95% confidence intervals. Core and functionality tests remain high across checkpoints while error-handling pass ...](https://arxiv.org/html/2603.24755/2603.24755v1/x5.png)
*图5：Figure 6 : Mean continuous pass rates by test type over problem progress with bootstrap 95% confidence intervals. Core and functionality tests remain high across checkpoints while error-handling pass ...*


## 实验结果
不同系统在SlopCodeBench上的性能退化表现：

| 系统 | 初始通过率 | 30轮后通过率 | Slop Score | 需求遗忘率 |
|------|-----------|------------|------------|-----------|
| GPT-4o | 78.3% | 54.2% | 0.42 | 31.5% |
| Claude 3.5 Sonnet | 81.2% | 61.7% | 0.35 | 24.8% |
| Gemini 1.5 Pro | 76.8% | 52.9% | 0.45 | 33.2% |
| Llama 3.1 405B | 71.5% | 46.3% | 0.51 | 38.7% |
| 人类开发者（对照） | 85.6% | 82.1% | 0.08 | 5.3% |

所有AI系统均表现出显著的性能退化，而人类开发者保持了稳定的高性能。Claude 3.5 Sonnet的退化最小但仍显著低于人类水平。

## 总结
SlopCodeBench揭示了当前AI编程智能体在长时域迭代任务中存在的系统性性能退化问题，这一发现对于AI编程工具的工程实践具有重要警示意义。分析表明，上下文管理不足和错误传播是Slop现象的主要原因。

该基准的发布将推动社区关注长时域编程能力，而非仅关注单轮任务表现。未来工作将扩展任务集并探索减少性能退化的技术方案，如显式的需求追踪机制和迭代感知的代码生成策略。
