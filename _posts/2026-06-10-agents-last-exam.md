---
layout: post
title: "Agents' Last Exam：AI智能体经济价值评估基准"
date: 2026-06-10
categories: [论文解读, AI智能体]
tags: [AI智能体, 基准测试, 经济价值, 长视野任务, LLM评估]
---

> 📄 **论文**：Agents’ Last Exam
> 🔗 **arXiv**：[2606.05405](https://arxiv.org/abs/2606.05405)
> 🏢 **机构**：论文作者机构

## 一句话总结

提出ALE基准，评估AI智能体在具有真实经济价值的长视野任务上的表现，填补当前基准与实际部署之间的鸿沟。

## 背景与问题

Recent AI systems have achieved strong results on a wide range of benchmarks, yet these gains have not translated into economically meaningful deployment across many professional domains. We argue that this gap is largely an evaluation problem: widely used benchmarks lack sustained performance measurement on real and economically valuable workflows. This paper introduces Agents’ Last Exam (ALE) , a benchmark designed to evaluate AI agents on long horizon, economically valuable, real world tasks 


![Figure 1: Agents’ Last Exam spans a broad taxonomy of professional tasks and rea](https://arxiv.org/html/2606.05405/2606.05405v1/x1.png)
*图：Figure 1: Agents’ Last Exam spans a broad taxonomy of professional tasks and realistic workflows.*

Over the past few years, AI systems have cleared one celebrated benchmark after another: world-champion games [ 38 ] , olympiad mathematics [ 14 ] , and competitive programming [ 12 ] . Yet by the metric that ultimately matters, economic output, the broader impact has remained surprisingly muted; benchmark victories have accumulated faster than measurable transformation in core industries. This gap, which we view as a utility problem for AI, suggests that the field now needs evaluations that mea

## 核心方法

ALE’s task instances are drawn from authentic professional workflows that experts carry out on real computer environments, routinely interleaving shell commands, GUI applications, file manipulation, and web research within a single task. As argued in Section 3.2 , this operational surface requires Generalist CUA-agents (GCUA) with full capability across all five functional layers (Brain, Eyes, Body, Hands, Feet). We therefore evaluate all agent systems in GCUA configuration.


![Figure 2: Distribution of 1,490 task instances across the ALE taxonomy. Each row](https://arxiv.org/html/2606.05405/2606.05405v1/x2.png)
*图：Figure 2: Distribution of 1,490 task instances across the ALE taxonomy. Each row is one of the 55 su*


![Figure 1: Agents’ Last Exam spans a broad taxonomy of professional tasks and realistic workflows.](https://arxiv.org/html/2606.05405/2606.05405v1/x1.png)
*图1：Figure 1: Agents’ Last Exam spans a broad taxonomy of professional tasks and realistic workflows.*

![Figure 2: Distribution of 1,490 task instances across the ALE taxonomy. Each row is one of the 55 su](https://arxiv.org/html/2606.05405/2606.05405v1/x2.png)
*图2：Figure 2: Distribution of 1,490 task instances across the ALE taxonomy. Each row is one of the 55 su*

![Figure 3: Benchmark positioning map. Prior benchmarks are placed by mapping their published domains ](https://arxiv.org/html/2606.05405/2606.05405v1/x3.png)
*图3：Figure 3: Benchmark positioning map. Prior benchmarks are placed by mapping their published domains *

![Figure 4: Task construction pipeline. Tasks proceed from expert sourcing through submission, first-p](https://arxiv.org/html/2606.05405/2606.05405v1/x4.png)
*图4：Figure 4: Task construction pipeline. Tasks proceed from expert sourcing through submission, first-p*

![Figure 5: Provenance and review yield. The 1,490 task instances split into 960 external submissions ](https://arxiv.org/html/2606.05405/2606.05405v1/x5.png)
*图5：Figure 5: Provenance and review yield. The 1,490 task instances split into 960 external submissions *

![Figure 6: Evaluation pipeline architecture. Each benchmark instance is defined by a Task Specificati](https://arxiv.org/html/2606.05405/2606.05405v1/x6.png)
*图6：Figure 6: Evaluation pipeline architecture. Each benchmark instance is defined by a Task Specificati*


## 实验结果

We introduced ALE , a benchmark of 960 expert-authored task workflows (1,490 task instances) across 55 digital industries, sourced from work experts have already shipped, anchored in the SOC/O*NET taxonomy, and scored through deterministic checks and structured rubrics rather than open-ended LLM judging. Frontier agents clear only a small fraction today; we release ALE as an instrument for closing the gap between benchmark success and GDP-relevant impact, where saturation would signal that agents can sustain the long-horizon, tool-intensive work professional practice actually requires.


![Figure 3: Benchmark positioning map. Prior benchmarks are placed by mapping thei](https://arxiv.org/html/2606.05405/2606.05405v1/x3.png)
*图：Figure 3: Benchmark positioning map. Prior benchmarks are placed by mapping their published domains *


### 实验数据表格

|                                          | Near-Term (59 tasks) | Full-Spectrum (55 tasks) | Last-Exam (35 tasks) | Overall |      |          |         |      |     |      |          |       |      |     |      |           |
| ---------------------------------------- | -------------------- | ------------------------ | -------------------- | ------- | ---- | -------- | ------- | ---- | --- | ---- | -------- | ----- | ---- | --- | ---- | --------- |
|                                          | Pass (%)             | Score                    |                      |         | Tok. | Pass (%) | Score   |      |     | Tok. | Pass (%) | Score |      |     | Tok. | Pass Rate |
| Mainstream Agent Harnesses (paired LLM + |                      |                          |                      |         |      |          |         |      |     |      |          |       |      |     |      |           |
| Codex [ 29 ] (GPT-5.5 [ 31 ] )           | 42.4                 | 70.7                     | $200                 | 30h     | 208M | 20.0     | 36.1    | $163 | 23h | 156M | 8.6      | 13.8  | $197 | 29h | 217M | 26.2      |
| ALE-Claw (GPT-5.5 [ 31 ] )               | 35.6                 | 74.0                     | $127                 | 17h     | 148M | 21.8     | 40.9    | $68  | 14h | 53M  | 8.6      | 15.2  | $112 | 19h | 130M | 24.2      |
| Cursor [ 8 ] (GPT-5.5 [ 31 ] )           | 36.4                 | 68.1 ±1                  | $61                  | 37h     | 49M  | 20.0     | 34.4    | $52  | 26h | 39M  | 2.9      | 8.7   | $64  | 21h | 69M  | 22.5      |
| Cursor [ 8 ] (Opus 4.7 [ 6 ] )           | 32.2                 | 66.7                     | $113                 | 34h     | 95M  | 20.0     | 39.1    | $184 | 18h | 202M | 5.7      | 10.6  | $155 | 22h | 149M | 21.5      |
| Droid [ 11 ] (GPT-5.5 [ 31 ] )           | 30.5                 | 61.5                     | $92                  | 27h     | 86M  | 16.4     | 35.0    | $69  | 21h | 59M  | 8.6      | 14.3  | $83  | 42h | 102M | 20.1      |
| ALE-Claw (Opus 4.7 [ 6 ] )               | 30.5                 | 65.8                     | $260                 | 21h     | 294M | 18.2     | 38.1    | $251 | 20h | 293M | 2.9      | 7.9   | $630 | 49h | 763M | 19.5      |
| Claude Code [ 4 ] (Sonnet 4.6 [ 7 ] )    | 31.4                 | 59.7 ±1                  | $104                 | 72h     | 248M | 12.7     | 27.1 ±1 | $110 | 38h | 240M | 0.0      | 6.9   | $164 | 72h | 334M | 17.1      |

## 总结

Agents’ Last Exam 提出了一个新颖的研究框架，针对AI智能体领域的核心挑战提供了系统性解决方案。

**主要贡献：**
- 提出ALE基准，评估AI智能体在具有真实经济价值的长视野任务上的表现，填补当前基准与实际部署之间的鸿沟。
- 通过系统实验验证了方法的有效性
- 为后续研究提供了重要的基准和参考

**局限性与展望：** 未来工作可进一步探索方法在更广泛场景下的应用，以及结合更多领域知识提升系统性能。