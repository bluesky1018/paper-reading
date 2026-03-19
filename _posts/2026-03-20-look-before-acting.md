---
layout: post
title: "行动前先看：增强视觉语言动作模型的视觉基础表征"
date: 2026-03-20
categories: [论文解读, 具身智能]
tags: [VLA模型, 机器人操控, 视觉语言模型, Mixture-of-Transformers, 视觉剪枝, 北京大学]
---

> 📄 **论文**：Look Before Acting: Enhancing Vision Foundation Representations for Vision-Language-Action Models
> 🔗 **arXiv**：[2603.15618](https://arxiv.org/abs/2603.15618)
> 🏢 **机构**：Peking University, Simplexity Robotics, The Chinese University of Hong Kong

## 一句话总结

DeepVision-VLA 通过系统性分析发现 VLA 模型在深层中对视觉 token 的敏感性逐渐下降，进而提出视觉语言混合 Transformer（VL-MoT）框架，将视觉专家的多级特征注入 VLA 主干深层，配合动作引导的视觉剪枝（AGVP），在仿真和真实世界任务上分别超越 SOTA 9.0% 和 7.5%。

## 背景与问题

视觉语言动作（VLA）模型已成为机器人操控的重要范式，准确的动作预测严重依赖于正确解释和整合以语言指令为条件的视觉观测。尽管近期研究尝试增强 VLA 模型的视觉能力，但大多数方法将 LLM 主干视为黑盒，对视觉信息如何被整合进动作生成过程缺乏理解。

### 关键发现：深层视觉敏感性退化

论文对多个不同动作生成范式的 VLA 模型进行了系统性分析，发现了一个关键现象：

**在动作生成过程中，深层对视觉 token 的敏感性逐渐降低。**

即在 VLA 模型的前几层，任务相关的视觉 token 获得了相对较高的注意力权重；但随着层数加深，对这些视觉 token 的注意力逐渐减弱，导致深层的视觉信息整合不足。这一现象是精细操控任务性能受限的重要根源。

![DeepVision-VLA概览](https://arxiv.org/html/2603.15618/x1.png)
*图1：(a) 在标准 VLA 模型中，对任务相关视觉 token 的注意力在深层中逐渐减弱；(b) DeepVision-VLA 引入 VL-MoT 框架，将视觉专家的多级特征注入 VLA 主干深层；(c) DeepVision-VLA 在多个真实世界操控任务中取得优越性能。*

## 核心方法

### DeepVision-VLA 框架

基于上述发现，论文提出了 **DeepVision-VLA**，构建于**视觉语言混合 Transformer（Vision-Language Mixture-of-Transformers, VL-MoT）**框架之上。

#### 视觉语言混合 Transformer（VL-MoT）

VL-MoT 的核心机制：
- **共享注意力**：视觉基础模型（Vision Foundation Model）与 VLA 主干之间建立共享注意力机制
- **多级视觉特征注入**：从视觉专家提取不同层级的视觉特征，注入到 VLA 主干的**深层**
- **目标**：通过补充深层的视觉信息，解决视觉敏感性退化问题

这使得 VLA 模型在生成精细操控动作时，能够持续访问高质量的视觉表征，而不仅仅依赖早层处理的视觉信息。

![VL-MoT框架详细图](https://arxiv.org/html/2603.15618/x2.png)
*图2：VL-MoT 框架的详细结构，展示了视觉专家特征如何通过共享注意力机制注入到 VLA 主干的深层。*

#### 动作引导的视觉剪枝（AGVP）

为了在增强视觉表征的同时控制计算开销，论文还提出了**动作引导的视觉剪枝（Action-Guided Visual Pruning, AGVP）**：
- **利用浅层注意力**：挖掘 VLA 主干浅层的注意力模式，识别与当前任务相关的视觉 token
- **剪除无关 token**：移除注意力低的无关视觉 token，保留任务关键的视觉线索
- **计算开销极低**：无需额外的复杂操作，直接利用已有的注意力权重进行剪枝

AGVP 实现了"精简但不损失"的效果：在减少视觉 token 数量的同时，通过保留最关键的视觉信息来强化操控线索。

![AGVP视觉剪枝示意](https://arxiv.org/html/2603.15618/x3.png)
*图3：动作引导的视觉剪枝（AGVP）流程示意，展示了如何利用浅层注意力权重识别并保留任务关键视觉 token。*

## 实验结果

DeepVision-VLA 在多个基准上与 SOTA 方法进行了系统对比：

| 场景 | 基线（SOTA） | DeepVision-VLA | 提升 |
|------|------------|----------------|------|
| 仿真任务 | 100% (基准) | **+9.0%** | 显著 |
| 真实世界任务 | 100% (基准) | **+7.5%** | 显著 |

**消融实验**验证了各组件的贡献：
- VL-MoT 单独带来主要性能提升
- AGVP 在保持性能的同时降低计算开销
- 多级特征注入优于单级特征注入

![真实世界操控结果](https://arxiv.org/html/2603.15618/x4.png)
*图4：DeepVision-VLA 在真实世界机器人操控任务中的性能表现，包括不同任务类型的成功率对比。*

## 总结

DeepVision-VLA 通过对 VLA 模型视觉信息流的系统性分析，发现并解决了深层视觉敏感性退化这一根本问题。VL-MoT 框架通过多级视觉特征注入持续增强深层的视觉表征，AGVP 则提供了高效的视觉 token 筛选机制，两者协同实现了精细操控能力的显著提升。

**局限性**：视觉专家的引入增加了模型参数量，对推理速度有一定影响；在场景中存在严重遮挡或极小目标时，视觉注意力的可靠性仍有待验证；当前方法主要在表格型操控任务上验证，泛化到更复杂的导航场景需要进一步研究。
