---
layout: post
title: "ClawArena：动态信息环境下AI智能体的基准测试"
date: 2026-04-08
categories: [论文解读, AI Agent]
tags: [AI Agent, Benchmark, 动态环境, LLM, UNC Chapel Hill]
---

> 📄 **论文**：ClawArena: Benchmarking AI Agents in Evolving Information Environments
> 🔗 **arXiv**：[2604.04202](https://arxiv.org/abs/2604.04202)
> 🏢 **机构**：University of North Carolina at Chapel Hill
> 👥 **作者**：Haonian Ji, Kaiwen Xiong, Siwei Han, Peng Xia, Shi Qiu, Yiyang Zhou, Jiaqi Liu, Jinlong Li, Bingzhou Li, Zeyu Zheng, Cihang Xie, Huaxiu Yao

## 一句话总结
提出ClawArena基准，评估AI智能体在持续演化的信息环境中的决策与推理能力，模拟真实世界动态变化场景

## 背景与问题

AI agents deployed as persistent assistants must maintain correct beliefs as their information environment evolves. In practice, evidence is scattered across heterogeneous sources that often contradict one another, new information can invalidate earlier conclusions, and user preferences surface through corrections rather than explicit instructions. Existing benchmarks largely assume static, single-authority settings and do not evaluate whether agents can keep up with this complexity. We introduce ClawArena, a benchmark for evaluating AI agents in evolving information environments. Each scenario maintains a complete hidden ground truth while exposing the agent only to noisy, partial, and sometimes contradictory traces across multi-channel sessions, workspace files, and staged updates. Evaluation is organized around three coupled challenges: multi-source conflict reasoning, dynamic belief revision, and implicit personalization, whose interactions yield a 14-category question taxonomy. Two question formats, multi-choice (set-selection) and shell-based executable checks, test both reasoning and workspace grounding. The current release contains 64 scenarios across 8 professional domains, totaling 1{,}879 evaluation rounds and 365 dynamic updates. Experiments on five agent frameworks and five language models show that both model capability (15.4% range) and framework design (9.2%) substantially affect performance, that self-evolving skill frameworks can partially close model-capability gaps, and that belief revision difficulty is determined by update design strategy rather than the mere presence of updates. Code is available at this https URL.



## 核心方法

详见原文方法章节。


![Figure 1: Overview of ClawArena across 8 professional domains. Each scenario pre](https://arxiv.org/html/2604.04202/2604.04202v1/overview.png)
*图：Figure 1: Overview of ClawArena across 8 professional domains. Each scenario presents multi-channel session histories, workspace files, and evaluation*


![Figure 2: Dataset composition of ClawArena. The inner ring shows 8 professional ](https://arxiv.org/html/2604.04202/2604.04202v1/sunburst_v7_notext.png)
*图：Figure 2: Dataset composition of ClawArena. The inner ring shows 8 professional domains (64 scenarios, 1,879 rounds total); the outer ring breaks each*


![Figure 3: ClawArena construction pipeline. Real-world distributions and characte](https://arxiv.org/html/2604.04202/2604.04202v1/pipeline_v2.png)
*图：Figure 3: ClawArena construction pipeline. Real-world distributions and character profiles feed a three-stage bootstrap, producing 64 scenarios organi*


## 实验结果

详见原文实验章节。


![Figure 4: Per-option case study on two representative questions from ClawArena. ](https://arxiv.org/html/2604.04202/2604.04202v1/12.png)
*图：Figure 4: Per-option case study on two representative questions from ClawArena. Case 1 (MS-R): no configuration achieves a perfect score; the two high*


![Figure 5: Case 3 (MS+DU): Self-diagnostic accuracy varies sharply across configu](https://arxiv.org/html/2604.04202/2604.04202v1/34.png)
*图：Figure 5: Case 3 (MS+DU): Self-diagnostic accuracy varies sharply across configurations after an update reveals contamination-rate discrepancies. Case*


![Figure 6: Case 5 (exec_check): execution-verified bug fix where GPT-5.1 framewor](https://arxiv.org/html/2604.04202/2604.04202v1/56.png)
*图：Figure 6: Case 5 (exec_check): execution-verified bug fix where GPT-5.1 frameworks fail 39–47 of 49 tests despite claiming bugs are fixed, exposing a *


![Figure 7: Case 7 (MS+P): norm retroactivity bias after a code-style policy updat](https://arxiv.org/html/2604.04202/2604.04202v1/78.png)
*图：Figure 7: Case 7 (MS+P): norm retroactivity bias after a code-style policy update; no configuration achieves a perfect score, but Sonnet 4.6/claude-co*


## 总结

本文提出了 **ClawArena**，提出ClawArena基准，评估AI智能体在持续演化的信息环境中的决策与推理能力，模拟真实世界动态变化场景。该工作从理论和实践层面均有创新，为后续研究提供了重要参考。

**局限性与未来方向：** 如所有工作一样，该研究仍有一定局限性，后续可在更大规模数据集、更多样化场景下进行验证和拓展。
