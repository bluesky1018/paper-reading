---
layout: post
title: "LightThinker++：从推理压缩到内存管理的进化"
date: 2026-04-08
categories: [论文解读, 高效推理]
tags: [推理压缩, 内存管理, Chain-of-Thought, LLM, 思维链]
---

> 📄 **论文**：LightThinker++: From Reasoning Compression to Memory Management
> 🔗 **arXiv**：[2604.03679](https://arxiv.org/abs/2604.03679)
> 🏢 **机构**：见原文
> 👥 **作者**：Yuqi Zhu, Jintian Zhang, Zhenjie Wan, Yujie Luo, Shuofei Qiao, Zhengke Gui, Da Zheng, Lei Liang, Huajun Chen, Ningyu Zhang

## 一句话总结
提出LightThinker++，在原有推理压缩基础上引入动态内存管理机制，进一步提升大模型长链式思维推理的效率与质量

## 背景与问题

Large language models (LLMs) excel at complex reasoning, yet their efficiency is limited by the surging cognitive overhead of long thought traces. In this paper, we propose LightThinker, a method that enables LLMs to dynamically compress intermediate thoughts into compact semantic representations. However, static compression often struggles with complex reasoning where the irreversible loss of intermediate details can lead to logical bottlenecks. To address this, we evolve the framework into LightThinker++, introducing Explicit Adaptive Memory Management. This paradigm shifts to behavioral-level management by incorporating explicit memory primitives, supported by a specialized trajectory synthesis pipeline to train purposeful memory scheduling. Extensive experiments demonstrate the framework&#39;s versatility across three dimensions. (1) LightThinker reduces peak token usage by 70% and inference time by 26% with minimal accuracy loss. (2) In standard reasoning, LightThinker++ slashes peak token usage by 69.9% while yielding a +2.42% accuracy gain under the same context budget for maximum performance. (3) Most notably, in long-horizon agentic tasks, it maintains a stable footprint beyond 80 rounds (a 60%-70% reduction), achieving an average performance gain of 14.8% across different complex scenarios. Overall, our work provides a scalable direction for sustaining deep LLM reasoning over extended horizons with minimal overhead.



## 核心方法

详见原文方法章节。


![Figure 1: An illustration of the compressed reasoning paradigms. (a) A CoT examp](https://arxiv.org/html/2604.03679/2604.03679v1/x1.png)
*图：Figure 1: An illustration of the compressed reasoning paradigms. (a) A CoT example. Tokens highlighted in yellow represent critical reasoning tokens, *


![Figure 2: An overview of LightThinker, illustrated with a three-step reasoning e](https://arxiv.org/html/2604.03679/2604.03679v1/x2.png)
*图：Figure 2: An overview of LightThinker, illustrated with a three-step reasoning example. Fig. (a) shows the attention mask of Vanilla during both train*


![Figure 3: Relationship between context length and the number of generated tokens](https://arxiv.org/html/2604.03679/2604.03679v1/x3.png)
*图：Figure 3: Relationship between context length and the number of generated tokens across different methods. The Dependency metric corresponds to the ar*


![Figure 4: Overview of LightThinker++. a) Memory Action Space: Reasoning steps ar](https://arxiv.org/html/2604.03679/2604.03679v1/x4.png)
*图：Figure 4: Overview of LightThinker++. a) Memory Action Space: Reasoning steps are instantiated as dual-form entities ℐi=(Ri,Zi)\mathcal{I}_{i}=(R_{i},*


![Figure 5: Efficiency Analysis and Ablation Results. (a) shows the average number](https://arxiv.org/html/2604.03679/2604.03679v1/x5.png)
*图：Figure 5: Efficiency Analysis and Ablation Results. (a) shows the average number of generated tokens for each model on each dataset. (b) shows the dis*


![Figure 6: Case study. The figure shows a partial inference trace for one GSM8K e](https://arxiv.org/html/2604.03679/2604.03679v1/x6.png)
*图：Figure 6: Case study. The figure shows a partial inference trace for one GSM8K example. The full example is provided in App. C.1.6. Pink and light blu*


## 实验结果

详见原文实验章节。


![Figure 7: Efficiency Analysis and Ablation Results under the Throughput setting.](https://arxiv.org/html/2604.03679/2604.03679v1/x7.png)
*图：Figure 7: Efficiency Analysis and Ablation Results under the Throughput setting. (a) illustrates the average number of generated tokens retained in th*


![Figure 8: Case Study. The figure illustrates partial inference results of a case](https://arxiv.org/html/2604.03679/2604.03679v1/x8.png)
*图：Figure 8: Case Study. The figure illustrates partial inference results of a case of LThinker++ from GSM8K.*


![Figure 9: Thought segment length distribution. Kernel density estimates of per-t](https://arxiv.org/html/2604.03679/2604.03679v1/x9.png)
*图：Figure 9: Thought segment length distribution. Kernel density estimates of per-thought segment lengths (in characters) for LThinker_tho, LThinker_tho1*


![Figure 10: Overview of LightThinker++ for long-horizon agentic reasoning. LightT](https://arxiv.org/html/2604.03679/2604.03679v1/x10.png)
*图：Figure 10: Overview of LightThinker++ for long-horizon agentic reasoning. LightThinker++ follows the Thought–Action–Observation loop while explicitly *


![Figure 11: Quantitative Analysis of Context Management Efficiency. Fig.(a) illus](https://arxiv.org/html/2604.03679/2604.03679v1/x11.png)
*图：Figure 11: Quantitative Analysis of Context Management Efficiency. Fig.(a) illustrates the active context trajectories (CtC_{t}) across interaction ro*


![Figure 12: Quantitative Analysis of Context Management Efficiency. Fig.(a) illus](https://arxiv.org/html/2604.03679/2604.03679v1/x12.png)
*图：Figure 12: Quantitative Analysis of Context Management Efficiency. Fig.(a) illustrates the active context trajectories (CtC_{t}) across interaction ro*


## 总结

本文提出了 **LightThinker++**，提出LightThinker++，在原有推理压缩基础上引入动态内存管理机制，进一步提升大模型长链式思维推理的效率与质量。该工作从理论和实践层面均有创新，为后续研究提供了重要参考。

**局限性与未来方向：** 如所有工作一样，该研究仍有一定局限性，后续可在更大规模数据集、更多样化场景下进行验证和拓展。
