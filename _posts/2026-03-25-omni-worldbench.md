---
layout: post
title: "Omni-WorldBench：面向世界模型的综合交互评测基准"
date: 2026-03-25
categories: [论文解读, 评测基准]
tags: ["世界模型", "评测基准", "4D生成", "视频生成"]
---

> 📄 **论文**：Omni-WorldBench: Towards a Comprehensive Interaction-Centric Evaluation for World Models
> 🔗 **arXiv**：[2603.22212](https://arxiv.org/abs/2603.22212)
> 🏢 **机构**：Meiqi Wu et al. (中科院自动化所, 阿里巴巴)

## 一句话总结

提出Omni-WorldBench，首个专为世界模型4D交互响应能力设计的综合评测基准。

## 背景与问题

1 Introduction The world models aim to characterize the temporal evolution of environmental states under given interaction conditions, providing a foundation for counterfactual reasoning, planning, and decision-making [ 23 ] . Taking advantage of recent advances in video generation, this paradigm has increasingly adopted video synthesis as a core implementation pathway. By leveraging high-quality general-purpose video representations to model world dynamics, video-based world models have been widely applied to autonomous driving, embodied intelligence, and game agents, substantially accelerating progress in these domains. Unlike rapid progress in world model design, the development of dedica


![Figure 1: Overview of Omni-WorldBench. Left: Omni-WorldSuite defines three levels of interactions, each specified by an initial frame and a prompt. Ri](https://arxiv.org/html/2603.22212v1/x1.png)
*图1：Figure 1: Overview of Omni-WorldBench. Left: Omni-WorldSuite defines three levels of interactions, each specified by an initial frame and a prompt. Ri*


现有方法存在明显局限性：缺乏系统性的评测或方法框架来解决上述问题。本文的核心动机正是填补这一空白，提出更有效的解决方案。


![Figure 2: Omni-WorldSuite Construction Pipeline and Analysis. (a) Dataset-grounded prompt generation. Prompts are generated from open-source data usin](https://arxiv.org/html/2603.22212v1/x2.png)
*图2：Figure 2: Omni-WorldSuite Construction Pipeline and Analysis. (a) Dataset-grounded prompt generation. Prompts are generated from open-source data usin*


## 核心方法

2 Related Works 2.1 World Models Design World models characterize how environment states evolve over time under given interaction conditions, thereby providing effective support for tasks such as counterfactual simulation, planning, and decision-making [ 23 ] . Early world models primarily relied on multimodal large language models (MLLMs) [ 33 , 2 , 3 , 11 ] that represent world states through textual abstractions [ 66 , 53 ] . Recent advances in video generation [ 47 , 59 , 46 , 63 , 74 ] have driven a shift toward video-based world models, which offer a more expressive and grounded representation of complex environments and have emerged as a dominant paradigm in the field [ 14 , 76 , 72 ] . In this work, we focus on world models built upon video generation. Across different application 


![Figure 3: Omni-WorldSuite examples across three interaction levels. Left: Examples from the General Scene domain. Right: Examples from the Task-Orient](https://arxiv.org/html/2603.22212v1/x3.png)
*图3：Figure 3: Omni-WorldSuite examples across three interaction levels. Left: Examples from the General Scene domain. Right: Examples from the Task-Orient*



![Figure 4: Statistics of Omni-WorldSuite. (a) Overall Distributions; (b–g) Distributions of core principles; (h) prompt counts by interaction level; (i](https://arxiv.org/html/2603.22212v1/x4.png)
*图4：Figure 4: Statistics of Omni-WorldSuite. (a) Overall Distributions; (b–g) Distributions of core principles; (h) prompt counts by interaction level; (i*



![Table 1: Quantitative evaluation results of various models on the proposed benchmark. The metrics are grouped into Interaction Effect Fidelity, Genera](https://arxiv.org/html/2603.22212v1/x5.png)
*图5：Table 1: Quantitative evaluation results of various models on the proposed benchmark. The metrics are grouped into Interaction Effect Fidelity, Genera*


## 实验结果

4 Omni-Metric To facilitate an omni -directional assessment of world models, we introduce Omni-Metric , a framework designed to deliver a truly comprehensive evaluation experience. Omni-Metric delineates three pivotal dimensions: Generated Video Quality , which quantifies both static and dynamic visual fidelity; Camera-Object Controllability , which scrutinizes scene coherence and object controllability in the absence of external interventions; and Interaction Effect Fidelity , which evaluates adherence to physical laws, event interactions, and temporal sequence logic within realistic scenarios. Collectively, these dimensions establish a rigorous paradigm for benchmarking the perceptual quality, environmental stability, and causal reasoning capabilities inherent to advanced world models. 4


![Figure 5: Non-camera-controlled Interaction Comparison. Qualitative comparison of generated videos from different models under the same prompt and fir](https://arxiv.org/html/2603.22212v1/x6.png)
*图6：Figure 5: Non-camera-controlled Interaction Comparison. Qualitative comparison of generated videos from different models under the same prompt and fir*




## 总结

6 Conclusion Summary. In this work, we introduce Omni-WorldBench, the first benchmark dedicated to evaluating the interactive response capabilities of video world models. Unlike existing benchmarks that mainly focus on visual quality or motion realism, Omni-WorldBench emphasizes action-driven scene evolution, intermediate state transitions, and causal consistency under interactive prompts, providing a more comprehensive and holistic evaluation perspective. To support this goal, we establish a rigorous evaluation framework consisting of Omni-WorldSuite, a hierarchical prompt suite spanning dive

本文工作的主要贡献包括：（1）提出Omni-WorldBench，首个专为世界模型4D交互响应能力设计的综合评测基准。；（2）通过充分的实验验证了方法的有效性。未来工作可在此基础上进一步探索更大规模、更多样化场景下的应用与扩展。

> 🔗 论文链接：[https://arxiv.org/abs/2603.22212](https://arxiv.org/abs/2603.22212)
