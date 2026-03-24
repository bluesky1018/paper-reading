---
layout: post
title: "OpenResearcher：全开放的长轨迹深度研究数据合成流水线"
date: 2026-03-25
categories: [论文解读, AI Agent]
tags: ["深度研究", "轨迹合成", "RAG", "AI Agent", "开源"]
---

> 📄 **论文**：OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis
> 🔗 **arXiv**：[2603.20278](https://arxiv.org/abs/2603.20278)
> 🏢 **机构**：Zhuofeng Li et al. (TIGER-Lab, 滑铁卢大学)

## 一句话总结

OpenResearcher提供完全开源的深度研究轨迹合成流水线，基于1500万文档离线语料生成超9.7万条研究轨迹。

## 背景与问题

1 Introduction Figure 1: Performance comparison on BrowseComp-Plus. Since the release of DeepSeek-R1 (Guo et al. , 2025 ) , there has been growing interest in collecting long-horizon reasoning trajectories from large reasoning models (LRMs) across diverse domains. Representative efforts include OpenThoughts (Guha et al. , 2025 ) , OpenMathReasoning (Moshkov et al. , 2025 ) , and OpenCodeReasoning (Ahmad et al. , 2025 ) . These trajectories are typically used to post-train smaller reasoning models via supervised fine-tuning (SFT). For instance, DeepSeek-R1-Distill (Guo et al. , 2025 ) achieves state-of-the-art performance solely via SFT over curated long-reasoning datasets. Recently, deep res



现有方法存在明显局限性：缺乏系统性的评测或方法框架来解决上述问题。本文的核心动机正是填补这一空白，提出更有效的解决方案。



## 核心方法

2 Preliminary Deep Research Workflow. Most deep research agents follow a ReAct-style paradigm (Yao et al. , 2022 ) . We formalize this interaction process as follows. Given a query q q , a system prompt s 0 s_{0} , and tool metadata (details in Appendix § C.4 ), the model interleaves reasoning and tool calls, receiving observations from the environment until termination. This process forms a trajectory ℋ T \mathcal{H}_{T} , which is a sequence of reasoning–action–observation triplets: ℋ T = { ( q , s 0 , 𝒯 m ​ e ​ t ​ a ) , ( r 1 , a 1 , o 1 ) , … , ( r i , a i , o i ) , … , ( r T , a T ) } , \mathcal{H}_{T}=\{(q,s_{0},\mathcal{T}_{meta}),(r_{1},a_{1},o_{1}),\dots,(r_{i},a_{i},o_{i}),\dots,(r_{T},a_{T})\}, (1) where r i r_{i} , a i a_{i} , and o i o_{i} denote the reasoning chain of though







## 实验结果

4 Experiments 4.1 Experimental Setup Training. To validate the effectiveness of the synthesized trajectories, we perform supervised fine-tuning (SFT) on a base model initialized from NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 (Blakeman et al. , 2025 ) . We curate the training data by applying rejection sampling: only trajectories that yield correct final answers are retained, resulting in around 55K trajectories. We adopt Megatron-LM (Shoeybi et al. , 2019 ) as the distributed training framework. All experiments follow a fixed and controlled configuration to ensure reproducibility. Training is conducted on 8 NVIDIA H100 GPUs for approximately 8 hours, with a learning rate of 5 × 10 − 5 5\times 10^{-5} without learning rate decay. To accommodate the long-horizon nature of our trajectories, se





## 总结

6 Conclusion OpenResearcher improves the reproducibility of long-horizon deep research trajectory synthesis by relocating the search-and-browse loop to a controllable offline environment. By replacing repeated live-web API calls with an offline setup, it reduces the cost of large-scale trajectory generation and lessens reliance on proprietary infrastructure. The explicit browser abstraction with search , open , and find further provides a simple interface for modeling realistic information-seeking behavior. Across both fixed-corpus evaluation and transfer to live-web benchmarks, trajectories s

本文工作的主要贡献包括：（1）OpenResearcher提供完全开源的深度研究轨迹合成流水线，基于1500万文档离线语料生成超9.7万条研究轨迹。；（2）通过充分的实验验证了方法的有效性。未来工作可在此基础上进一步探索更大规模、更多样化场景下的应用与扩展。

> 🔗 论文链接：[https://arxiv.org/abs/2603.20278](https://arxiv.org/abs/2603.20278)
