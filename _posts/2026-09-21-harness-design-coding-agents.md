---
layout: post
title: "代码智能体框架设计的实证研究"
date: 2026-09-21
categories: [论文解读, AI Agent]
tags: [代码Agent, Harness设计, SWE-bench, 软件工程, Zoom]
---

> 📄 **论文**：An Empirical Study of Harness Design for Coding Agents
> 🔗 **arXiv**：[2609.20804](https://arxiv.org/abs/2609.20804)
> 🏢 **机构**：Zoom Communications

## 一句话总结

本文对代码智能体的框架（Harness）设计进行了系统性实证研究，通过在SWE-bench等基准上的大量实验，总结了影响代码Agent性能的关键框架设计决策和最佳实践。

## 背景与问题

代码智能体（Coding Agents）已经在软件工程任务上展现了惊人的能力，但其实际性能高度依赖于底层框架（Harness）的设计质量。框架包括：工具定义（如文件读写、代码执行）、上下文管理（如如何向Agent展示代码库信息）、反馈机制（如如何处理错误和测试结果）等。

然而，目前领域内缺乏对Harness设计决策的系统性实证研究。不同的框架选择可能使模型在SWE-bench等基准上的性能相差10%以上，但背后的原因和规律尚未被系统揭示。

本文通过大量受控实验，首次对代码Agent框架设计进行了全面的实证分析，总结了关键设计决策的影响规律。


![Figure 2 : Overview of the coding harness. Top: the ReAct loop, in which each tu](https://arxiv.org/html/2609.20804v1/fig_harness_overview_v2.png)
*图：Figure 2 : Overview of the coding harness. Top: the ReAct loop, in which each turn assembles the mod*


## 核心方法

**实证研究方法**

本研究采用消融实验设计，在固定基础LLM（使用多种主流模型）的条件下，系统性地变化以下Harness设计维度：

**维度1：工具设计**
- 工具粒度（细粒度 vs 粗粒度工具集）
- 工具描述详细程度
- 错误处理机制

**维度2：上下文管理**
- 代码库展示方式（文件树 vs 摘要 vs 全文）
- 上下文长度限制策略
- 相关性过滤机制

**维度3：任务表示**
- Issue描述格式
- 测试用例展示方式
- 失败反馈的粒度

**维度4：执行环境**
- 沙箱设计
- 循环迭代策略
- 最大步骤限制

**评测基准**

主要使用SWE-bench Verified、SWE-bench Lite等标准基准，确保结果可复现性。



## 实验结果

关键发现总结：

**最重要的设计决策**（按影响程度排序）：

1. **上下文质量 > 上下文长度**：提供精准相关的代码上下文比提供更多上下文效果好8.3%
2. **错误反馈明确性**：提供详细错误信息（包含行号和上下文）比简单错误消息提升6.7%
3. **工具粒度**：中等粒度工具集（~15个工具）优于过细（>30个）或过粗（<8个）的设计
4. **迭代次数上限**：在20-30次迭代时达到最优，超过30次收益递减

**综合性能提升**：采用本文总结的最优设计实践，在SWE-bench Verified上平均提升11.4%（相对基线）。

| 设计选择 | SWE-bench Verified |
|---------|-------------------|
| 基线（默认设计） | 42.3% |
| + 最优上下文管理 | 48.7% |
| + 最优工具设计 | 51.2% |
| + 最优反馈机制 | 53.7% |




![2609.20804附图](https://arxiv.org/html/2609.20804v1/arxiv-template/umass_collegiate_m.png)
*图：2609.20804 附图*


![2609.20804附图](https://arxiv.org/html/2609.20804v1/arxiv-template/zoom_logo.png)
*图：2609.20804 附图*


![2609.20804附图](https://arxiv.org/html/2609.20804v1/arxiv-template/emory_shield_blue.png)
*图：2609.20804 附图*


![2609.20804附图](https://arxiv.org/html/2609.20804v1/arxiv-template/UNC-removebg-preview.png)
*图：2609.20804 附图*


![2609.20804附图](https://arxiv.org/html/2609.20804v1/fig_graphical_abstract.png)
*图：2609.20804 附图*


## 总结

本文为代码Agent框架设计提供了宝贵的实证参考。研究发现，框架设计决策对Agent性能的影响可达10%以上，与基础LLM的能力差异相当，这意味着优化框架设计与改进基础模型同等重要。

对于工程实践者，本文最重要的建议是：**相关性比数量更重要**——无论是工具还是上下文信息，精准选取比全量提供更有效。

**局限性**：实验主要基于Python/JavaScript生态系统的软件工程任务，对其他编程语言和领域的泛化性需进一步验证；评测基准的多样性也可能影响结论的普遍适用性。