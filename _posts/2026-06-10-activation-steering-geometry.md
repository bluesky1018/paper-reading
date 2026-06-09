---
layout: post
title: "激活引导的几何解析：角度-范数分解视角"
date: 2026-06-10
categories: [论文解读, LLM可解释性]
tags: [激活引导, LLM可解释性, 球面引导, 表征工程, 角度-范数分解]
---

> 📄 **论文**：A Geometric Account of Activation Steering through Angle–Norm Decomposition
> 🔗 **arXiv**：[2606.06735](https://arxiv.org/abs/2606.06735)
> 🏢 **机构**：论文作者机构

## 一句话总结

通过角度-范数分解的受控实验，重新审视线性激活引导的内在机制，发现隐藏状态的范数和方向均携带概念相关信息，为激活引导提供更精确的几何理论基础。

## 背景与问题

Linear activation steering has gained popularity as a simple and empirically effective way to control language model behavior. More recently, spherical steering paradigms have been proposed to address limitations of additive interventions, often motivated by the assumption that hidden-state norm does not carry concept-relevant information. In this work, we revisit this assumption through a controlled empirical study designed to disentangle the roles of angular and radial components. We show that


![Figure 1: Effect of norm scaling in SN. The left panel shows downstream task met](https://arxiv.org/html/2606.06735/2606.06735v1/x1.png)
*图：Figure 1: Effect of norm scaling in SN. The left panel shows downstream task metric change, and the *

Linear activation steering has become a widely used approach for controlling language model behavior through interventions on intermediate representations (Zou et al. , 2023 ; Turner et al. , 2023 ; Panickssery et al. , 2023 ) . Given a steering direction associated with a target concept, standard methods add this direction to hidden states with a scalar strength. These interventions are simple, training-free, and effective across behaviors such as truthfulness, sentiment, toxicity, and refusal 

## 核心方法

Models and steering layer. We evaluate all methods on seven transformer language models spanning 1B to 70B parameters: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-it, Llama-3.1-8B, Llama-3.2-1B-Instruct, Qwen2.5-3B-Instruct, and Llama-3.1-70B-Instruct. For each model, steering is applied to the residual-stream output at 75% depth. This gives steering layers 24, 21, 31, 24, 12, 27, and 60 respectively. We use a single forward hook at this layer, replacing each hidden state with a steered state at every token position during generation.


![Figure 2: Fraction of folds in which each value achieves the best perplexity or ](https://arxiv.org/html/2606.06735/2606.06735v1/x2.png)
*图：Figure 2: Fraction of folds in which each value achieves the best perplexity or task metric. At , ac*


![Figure 1: Effect of norm scaling in SN. The left panel shows downstream task metric change, and the ](https://arxiv.org/html/2606.06735/2606.06735v1/x1.png)
*图1：Figure 1: Effect of norm scaling in SN. The left panel shows downstream task metric change, and the *

![Figure 2: Fraction of folds in which each value achieves the best perplexity or task metric. At , ac](https://arxiv.org/html/2606.06735/2606.06735v1/x2.png)
*图2：Figure 2: Fraction of folds in which each value achieves the best perplexity or task metric. At , ac*

![Figure 3: T1: CV of hidden-state norms vs. layer for all 7 models, 10 corpora. Grey dotted = steerin](https://arxiv.org/html/2606.06735/2606.06735v1/x3.png)
*图3：Figure 3: T1: CV of hidden-state norms vs. layer for all 7 models, 10 corpora. Grey dotted = steerin*

![Figure 4: Linear probe accuracy versus layer for all four concept datasets. Each dataset contains th](https://arxiv.org/html/2606.06735/2606.06735v1/x11.png)
*图4：Figure 4: Linear probe accuracy versus layer for all four concept datasets. Each dataset contains th*

![Figure 5: Norm ratio for CAA-m at matched per-token target .](https://arxiv.org/html/2606.06735/2606.06735v1/x12.png)
*图5：Figure 5: Norm ratio for CAA-m at matched per-token target .*

![Figure 6: Downstream task metric, WikiText-103 perplexity, and MMLU accuracy under S and CAA-m at ma](https://arxiv.org/html/2606.06735/2606.06735v1/x13.png)
*图6：Figure 6: Downstream task metric, WikiText-103 perplexity, and MMLU accuracy under S and CAA-m at ma*


## 实验结果

Our study has several limitations. First, we apply steering at a single fixed layer, chosen at 75% depth for each model. Although this gives a controlled comparison across methods, the optimal angle–norm trade-off may vary across layers.


![Figure 3: T1: CV of hidden-state norms vs. layer for all 7 models, 10 corpora. G](https://arxiv.org/html/2606.06735/2606.06735v1/x3.png)
*图：Figure 3: T1: CV of hidden-state norms vs. layer for all 7 models, 10 corpora. Grey dotted = steerin*


### 实验数据表格

| Method | Norm preserved | Tokenwise |
| ------ | -------------- | --------- |
| CAA    |                |           |
| CAA-r  |                |           |
| CAA-m  |                |           |
| S      |                |           |
| AS     |                |           |
| SN     |                |           |

## 总结

A Geometric Account of Activation Steering through Angle–Norm Decomposition 提出了一个新颖的研究框架，针对LLM可解释性领域的核心挑战提供了系统性解决方案。

**主要贡献：**
- 通过角度-范数分解的受控实验，重新审视线性激活引导的内在机制，发现隐藏状态的范数和方向均携带概念相关信息，为激活引导提供更精确的几何理论基础。
- 通过系统实验验证了方法的有效性
- 为后续研究提供了重要的基准和参考

**局限性与展望：** 未来工作可进一步探索方法在更广泛场景下的应用，以及结合更多领域知识提升系统性能。