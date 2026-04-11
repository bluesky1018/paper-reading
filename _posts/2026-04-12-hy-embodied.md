---
layout: post
title: "HY-Embodied-0.5：面向真实世界智能体的具身基础模型"
date: 2026-04-12
categories: [论文解读, 具身智能]
tags: [Embodied AI, VLM, Robotics, Mixture-of-Transformers, RL, Tencent]
---

> 📄 **论文**：HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents
> 🔗 **arXiv**：[2604.07430](https://arxiv.org/abs/2604.07430)
> 🏢 **机构**：腾讯混元（Tencent Robotics X × HY Vision Team）

## 一句话总结

HY-Embodied-0.5 是腾讯混元推出的具身智能基础模型系列，通过 Mixture-of-Transformers 架构、迭代自进化后训练和大到小蒸馏，在22个基准测试上达到前沿水平，并在真实机器人控制中取得突破性成果。

## 背景与问题

将数字智能转化为物理世界行动力是当前 AI 的核心挑战之一。尽管视觉语言模型（VLM）在通用场景中取得了显著进展，但将其应用于具身智能体（embodied agents）仍面临两大根本性挑战：

**挑战一：精细视觉感知的不足**。具身任务要求模型精确理解三维空间关系、深度信息、物体位置等细粒度视觉细节。然而现有 VLM 在捕捉这些物理世界细节时仍存在明显缺陷——它们在静态网络图像上训练，缺乏对具身场景所需的空间感知能力。

**挑战二：具身推理能力的缺失**。主流 VLM 主要在静态网络规模数据集上训练，无法满足动态预测、物理交互和行动规划的需求。简单地说，"看图回答问题"与"制定机器人行动计划"是本质不同的任务。

HY-Embodied-0.5 的目标正是填补通用 VLM 与具身智能体需求之间的这道鸿沟。

## 核心方法

### 模型架构：Mixture-of-Transformers（MoT）

![HY-Embodied-0.5 MoT架构](https://arxiv.org/html/2604.07430v1/x2.png)
*图：HY-Embodied-0.5 的 Mixture-of-Transformers 架构。MoT 设计通过模态特定的 QKV 和 FFN 层解耦视觉与文本 token 的处理。*

HY-Embodied-0.5 的核心架构创新是 **Mixture-of-Transformers（MoT）**。传统 VLM 使用统一参数处理视觉和语言 token，导致视觉能力强化时语言能力退化。MoT 通过为视觉和文本 token 引入各自独立的 QKV 和 FFN 参数来解决这一问题：

- **模态自适应计算**：视觉 token 使用专用参数集，文本 token 使用原始 LLM 参数，两者并行处理
- **双向视觉注意力**：视觉数据不具备语言的单向性，因此对视觉 token 采用双向注意力机制
- **视觉辅助监督**：通过视觉下一代码预测任务（visual next-code prediction）为视觉分支提供更强的监督信号

**视觉潜在 Token（Visual Latent Tokens）** 是另一个关键设计：在每个视觉输入序列末尾附加可学习的潜在 token，并使用大型 ViT 的全局特征进行监督，进一步提升小模型的感知能力。

![注意力计算机制](https://arxiv.org/html/2604.07430v1/x3.png)
*图：模态自适应 MoT 的注意力计算。不同颜色展示多模态序列中视觉与文本 token 的不同注意力模式。*

### 数据策略：1亿+具身预训练样本

![预训练数据分布](https://arxiv.org/html/2604.07430v1/x4.png)
*图：预训练和中间训练阶段的数据分布，涵盖基础感知、空间感知、具身感知和推理规划。*

构建了超过 **1亿** 高质量预训练样本，覆盖：
- 基础感知（Basic Perception）
- 空间感知（Spatial Perception）
- 具身感知（Embodied Perception）
- 推理与规划（Reasoning and Planning）

### 训练策略：迭代自进化后训练

![训练流程](https://arxiv.org/html/2604.07430v1/x5.png)
*图：HY-Embodied-0.5 系列的训练流程，大规模预训练建立多模态表征基础，后训练阶段提升推理能力。*

后训练采用**迭代自进化范式**：
1. 少量冷启动数据（cold start data）建立初始能力
2. 迭代强化学习（iterative RL）
3. 拒绝采样监督微调（rejection sampling SFT）
4. **大到小在策略蒸馏**（large-to-small on-policy distillation）：将 32B 大模型的能力迁移到 2B 小模型

### 机器人强化学习

![具身强化学习奖励设计](https://arxiv.org/html/2604.07430v1/x6.png)
*图：具身强化学习的奖励设计，针对多样化具身任务系统性制定奖励函数。*

## 实验结果

在 22 个公开基准测试上进行综合评估，覆盖视觉感知、空间推理和具身理解三大类别：

| 模型 | 22基准平均分 | 相较参照模型提升 |
|------|------------|----------------|
| HY-Embodied-0.5-MoT-2B | 58.0% | — |
| Qwen3-VL-4B（通用VLM） | ~47.8% | -10.2% |
| RoboBrain2.5-4B（专用具身VLM） | ~49.4% | -8.6% |
| HY-Embodied-0.5-MoE-A32B | 67.0% | — |
| Gemini 3.0 Pro | 63.6% | -3.4% |

关键亮点：
- **MoT-2B** 在参照对比的22个基准中，16个达到同尺寸SOTA
- **MoE-A32B** 超越前沿模型 Gemini 3.0 Pro（67.0% vs 63.6%）
- 在下游真实机器人控制任务中取得显著成果

![通用理解基准对比](https://arxiv.org/html/2604.07430v1/x7.png)
*图：HY-Embodied-0.5 MoT-2B 与同尺寸通用 VLM 的比较，证明具身训练不以牺牲通用理解为代价。*

![视觉感知可视化](https://arxiv.org/html/2604.07430v1/x8.png)
*图：视觉感知任务的可视化结果，展示 HY-Embodied-0.5 MoT-2B 在细粒度感知上的卓越能力。*

## 总结

HY-Embodied-0.5 代表了具身 VLM 领域的重要进展。通过 MoT 架构解决了视觉-语言能力协同提升的矛盾，通过大规模具身数据和迭代后训练范式提升了实体任务推理能力，通过大到小蒸馏将强大能力下沉到边缘可部署的小模型。

主要局限在于：当前模型仍以视觉感知和规划为主，与实际机器人硬件的闭环控制集成仍需进一步工作；同时，百亿参数的大模型对计算资源要求较高。未来研究方向包括更强的时序理解、多模态交互能力，以及与更多类型机器人的集成验证。
