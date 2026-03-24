---
layout: post
title: "LongCat-Flash-Prover：智能体工具集成强化学习助力形式化数学证明"
date: 2026-03-25
categories: [论文解读, 推理与数学]
tags: ["形式化推理", "Lean4", "强化学习", "MoE", "数学证明"]
---

> 📄 **论文**：LongCat-Flash-Prover: Advancing Native Formal Reasoning via Agentic Tool-Integrated Reinforcement Learning
> 🔗 **arXiv**：[2603.21065](https://arxiv.org/abs/2603.21065)
> 🏢 **机构**：Jianing Wang et al. (美团)

## 一句话总结

5600亿参数MoE模型LongCat-Flash-Prover，通过智能体工具集成强化学习实现原生形式化Lean4数学证明。

## 背景与问题

1 Introduction Recent advancements in large language models (LLMs) have shifted decisively toward enriching the reasoning capabilities, promoting the boundaries of artificial general intelligence (AGI) (OpenAI, 2024 ; Comanici et al., 2025 ; Guo et al., 2025a ; Yang et al., 2025 ) . While notable progress has been made in solving complex problems using natural language, current LLMs still struggle with formal theorem-proving tasks. These tasks necessitate the use of rigorous, verified formal languages (e.g., Lean4) to ensure reliable formal statements and proofs. Several previous efforts have been devoted to leveraging feedback from verification tools to train models in repairing Lean4 code 


![Figure 1: The performance comparison over proving tasks. The figures on the left and middle illustrate the performance with a limited 32 inference bud](https://arxiv.org/html/2603.21065v1/x1.png)
*图1：Figure 1: The performance comparison over proving tasks. The figures on the left and middle illustrate the performance with a limited 32 inference bud*


现有方法存在明显局限性：缺乏系统性的评测或方法框架来解决上述问题。本文的核心动机正是填补这一空白，提出更有效的解决方案。


![Figure 2: The overview of our hybrid-experts tool-integration synthesis pipeline. In this pipeline, we iteratively optimize three different experts (i](https://arxiv.org/html/2603.21065v1/x2.png)
*图2：Figure 2: The overview of our hybrid-experts tool-integration synthesis pipeline. In this pipeline, we iteratively optimize three different experts (i*


## 核心方法

2 Hybrid-Experts Iteration Framework Figure 2: The overview of our hybrid-experts tool-integration synthesis pipeline. In this pipeline, we iteratively optimize three different experts (i.e., auto-formalizer, lemma-style sketcher, and prover), and use these experts to synthesize trajectories based on pre-defined formal reasoning capabilities. Given one problem in natural language, we first transform it into a formal statement in Lean4 by interacting with the Lean4 compiler. Then, the formal statement will be used to generate a whole-proof and lemma-style sketch. If the whole-proof still fails to pass verification within a limited number of tool feedback rounds, the proof generated from the lemma-style sketch will be used instead for verification. In the rejection sampling phase, we will re


![Figure 3: The overview of the training pipeline of LongCat-Flash-Prover. We choose the LongCat Mid-train base model as the starting point, and then pe](https://arxiv.org/html/2603.21065v1/x3.png)
*图3：Figure 3: The overview of the training pipeline of LongCat-Flash-Prover. We choose the LongCat Mid-train base model as the starting point, and then pe*



![Table 1: Auto-formalization performance (Pass@8 metric, %) of different reasoning and specific auto-formalizer models across multiple benchmarks. Best](https://arxiv.org/html/2603.21065v1/x4.png)
*图4：Table 1: Auto-formalization performance (Pass@8 metric, %) of different reasoning and specific auto-formalizer models across multiple benchmarks. Best*



![Table 2: Theorem-proving performance (Pass@32 metric, %) of different reasoning and specific prover models across multiple benchmarks. Best in bold, s](https://arxiv.org/html/2603.21065v1/x5.png)
*图5：Table 2: Theorem-proving performance (Pass@32 metric, %) of different reasoning and specific prover models across multiple benchmarks. Best in bold, s*


## 实验结果

4 Experiments We conduct comprehensive evaluation of LongCat-Flash-Prover across both formal and informal reasoning tasks. In the formal domain, we specifically measure the model’s capabilities in auto-formalization and theorem proving. In the informal domain, we investigate whether training on formal reasoning tasks preserves or enhances the model’s performance on general reasoning tasks. In addition, we also conduct scaling behavior analysis to show the model training performance. Table 1: Auto-formalization performance (Pass@8 metric, %) of different reasoning and specific auto-formalizer models across multiple benchmarks. Best in bold , second in underlined . Auto-Formalization Combi- FormalMath- MathOlympiad- MiniF2F- ProofNet- Prover- Putnam- Bench Lite Bench Test Test Bench Bench (P


![Table 3: Theorem-proving performance (with different larger budgets, %) of different specific prover models across multiple benchmarks. Best in bold, ](https://arxiv.org/html/2603.21065v1/x6.png)
*图6：Table 3: Theorem-proving performance (with different larger budgets, %) of different specific prover models across multiple benchmarks. Best in bold, *



![Table 4: Performance (%) comparison across multiple general reasoning benchmarks. Best in bold. The result indicates that our LongCat-Flash-Prover can](https://arxiv.org/html/2603.21065v1/x7.png)
*图7：Table 4: Performance (%) comparison across multiple general reasoning benchmarks. Best in bold. The result indicates that our LongCat-Flash-Prover can*


## 总结

6 Conclusion In this work, we introduce LongCat-Flash-Prover, a 560 560 -billion-parameter Mixture-of-Experts (MoE) model that fuses the native formal reasoning with general reasoning capabilities. It achieves state-of-the-art performance among open-source models on multiple auto-formalization and theorem-proving tasks with verified tools. The core innovations underpinning LongCat-Flash-Prover are as follows: 1) an effective hybrid-experts iteration framework to better construct high-quality synthesized trajectories with multiple verified tools. 2) a Hierarchical Importance Sampling Policy Opt

本文工作的主要贡献包括：（1）5600亿参数MoE模型LongCat-Flash-Prover，通过智能体工具集成强化学习实现原生形式化Lean4数学证明。；（2）通过充分的实验验证了方法的有效性。未来工作可在此基础上进一步探索更大规模、更多样化场景下的应用与扩展。

> 🔗 论文链接：[https://arxiv.org/abs/2603.21065](https://arxiv.org/abs/2603.21065)
