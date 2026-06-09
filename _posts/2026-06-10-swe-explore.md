---
layout: post
title: "SWE-Explore：代码智能体库探索能力基准测试"
date: 2026-06-10
categories: [论文解读, 代码智能体]
tags: [代码智能体, SWE-bench, 基准测试, 仓库探索, LLM]
---

> 📄 **论文**：SWE-Explore: Benchmarking How Coding Agents Explore Repositories
> 🔗 **arXiv**：[2606.07297](https://arxiv.org/abs/2606.07297)
> 🏢 **机构**：论文作者机构

## 一句话总结

提出SWE-Explore基准，专门评估代码智能体在代码库中的探索能力（仓库理解、上下文检索、代码定位），揭示了当前SOTA智能体在细粒度能力上的不足。

## 背景与问题

Repository-level coding benchmarks such as SWE-bench have driven a rapid surge in the capabilities of coding agents. Yet they usually treat coding tasks as a holistic, binary prediction problem (e.g., resolved or unresolved), neglecting fine-grained agent capabilities such as repository understanding, context retrieval, code localization, and bug diagnosis. In this paper, we introduce SWE-Explore , a benchmark that isolates the evaluation of repository exploration, a critical capability of codin


![Figure 1: Motivation of SWE-Explore. A holistic metric of resolution rate confla](https://arxiv.org/html/2606.07297/2606.07297v1/x1.png)
*图：Figure 1: Motivation of SWE-Explore. A holistic metric of resolution rate conflates exploration, loc*

Repository-level coding benchmarks, such as SWE-bench [ 11 ] , have driven a rapid surge in the capabilities of automated coding agents [ 6 , 33 , 7 ] . The ecosystem around these benchmarks has expanded quickly: new evaluation distributions now cover multilingual repositories, multimodal software issues, and harder, long-horizon professional tasks [ 30 , 32 , 29 , 2 ] . In parallel, scalable training-oriented resources like SWE-smith [ 30 ] , SWE-Gym [ 15 ] , and SWE-Dev [ 23 ] are actively fue

## 核心方法

We evaluate explorers from four families. Two baselines bound the dynamic range: Oracle returns directly, and Random returns uniformly sampled regions. Sparse retrievers are represented by BM25 [ 17 ] and TF–IDF [ 18 ] . As a lightweight dense retriever we use a RAG pipeline instantiated with Potion, a static word-embedding retriever distilled from a sentence transformer. Finally, the agentic explorers cover five general-purpose coding agents (Claude Code [ 1 ] , Codex, OpenHands [ 24 ] , Mini-SWE-Agent [ 28 ] , AweAgent [ 3 ] ) and four published academic localization agents (AutoCodeRover [ 


![Figure 2: Overview of SWE-Explore. From solution-verified trajectories, SWE-Expl](https://arxiv.org/html/2606.07297/2606.07297v1/x2.png)
*图：Figure 2: Overview of SWE-Explore. From solution-verified trajectories, SWE-Explore extracts read ac*


![Figure 1: Motivation of SWE-Explore. A holistic metric of resolution rate conflates exploration, loc](https://arxiv.org/html/2606.07297/2606.07297v1/x1.png)
*图1：Figure 1: Motivation of SWE-Explore. A holistic metric of resolution rate conflates exploration, loc*

![Figure 2: Overview of SWE-Explore. From solution-verified trajectories, SWE-Explore extracts read ac](https://arxiv.org/html/2606.07297/2606.07297v1/x2.png)
*图2：Figure 2: Overview of SWE-Explore. From solution-verified trajectories, SWE-Explore extracts read ac*

![Figure 3: Language distribution of the retained SWE-Explore instances across 10 different coding lan](https://arxiv.org/html/2606.07297/2606.07297v1/x3.png)
*图3：Figure 3: Language distribution of the retained SWE-Explore instances across 10 different coding lan*

![Figure 4: Example of a SWE-Explore instance. Left: an issue plus a repo snapshot with the highlighte](https://arxiv.org/html/2606.07297/2606.07297v1/x4.png)
*图4：Figure 4: Example of a SWE-Explore instance. Left: an issue plus a repo snapshot with the highlighte*

![Figure 5: Resolve rate as the visible context degrades from the Oracle’s full core set to either of ](https://arxiv.org/html/2606.07297/2606.07297v1/x5.png)
*图5：Figure 5: Resolve rate as the visible context degrades from the Oracle’s full core set to either of *


## 实验结果

We presented SWE-Explore , a benchmark for evaluating repository exploration independently from patch generation through ranked, line-level context selection. Using trajectory-derived supervision, SWE-Explore compares retrievers, search agents, and long-context selectors by the evidence they surface rather than only by final repair outcomes. Our experiments show that exploration metrics track downstream repair, that current agents are strong at finding relevant files but remain recall-limited at the line level, and that missing core evidence hurts more than moderate redundant context. We hope 


![Figure 3: Language distribution of the retained SWE-Explore instances across 10 ](https://arxiv.org/html/2606.07297/2606.07297v1/x3.png)
*图：Figure 3: Language distribution of the retained SWE-Explore instances across 10 different coding lan*


### 实验数据表格

| Benchmark                     | Exec. Based | Multi- Lingual | Line-Level GT | Trajectory- Grounded GT | Joint Expl. + Repair Eval | Ranked Region Eval |
| ----------------------------- | ----------- | -------------- | ------------- | ----------------------- | ------------------------- | ------------------ |
| Loc-Bench [ 5 ]               | ✗           | ✗              | ✗             | ✗                       | ✗                         | ✗                  |
| SWE-bench Verified [ 11 , 6 ] | ✓           | ✗              | ✗             | ✗                       | ✗                         | ✗                  |
| SWE-bench Multilingual [ 30 ] | ✓           | ✓              | ✗             | ✗                       | ✗                         | ✗                  |
| SWE-bench-Pro [ 7 ]           | ✓           | ✓              | ✗             | ✗                       | ✗                         | ✗                  |
| ContextBench [ 13 ]           | ✓           | ✓              | ✗             | ✗                       | ✓                         | ✗                  |
| SWE-ContextBench [ 37 ]       | ✓           | ✗              | ✗             | ✗                       | ✗                         | ✗                  |
| SWE-Explore (Ours)            | ✓           | ✓              | ✓             | ✓                       | ✓                         | ✓                  |

## 总结

SWE-Explore: Benchmarking How Coding Agents Explore Repositories 提出了一个新颖的研究框架，针对代码智能体领域的核心挑战提供了系统性解决方案。

**主要贡献：**
- 提出SWE-Explore基准，专门评估代码智能体在代码库中的探索能力（仓库理解、上下文检索、代码定位），揭示了当前SOTA智能体在细粒度能力上的不足。
- 通过系统实验验证了方法的有效性
- 为后续研究提供了重要的基准和参考

**局限性与展望：** 未来工作可进一步探索方法在更广泛场景下的应用，以及结合更多领域知识提升系统性能。