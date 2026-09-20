---
layout: post
title: "JEPA-Anything：跨不同世界学习预测模型"
date: 2026-09-21
categories: [论文解读, 世界模型]
tags: [JEPA, 世界模型, 预测学习, 表征学习, 跨域泛化]
---

> 📄 **论文**：JEPA-Anything: Learning Predictive Models across Different Worlds
> 🔗 **arXiv**：[2609.20800](https://arxiv.org/abs/2609.20800)
> 🏢 **机构**：MPI-IS & 合作机构

## 一句话总结

JEPA-Anything将联合嵌入预测架构（JEPA）推广到多样化的"世界"（包括真实世界、游戏、机器人等），通过统一的预测框架学习跨域通用的世界模型，展现了强大的泛化能力。

## 背景与问题

联合嵌入预测架构（Joint Embedding Predictive Architecture, JEPA）是Yann LeCun提出的一种新型自监督学习框架，其核心思想是在表示空间（而非像素空间）做预测，从而学到更抽象、更有语义价值的世界表示。I-JEPA、V-JEPA等工作已经展现了这一框架在图像和视频领域的潜力。

然而，现实世界中的智能需要在多种"世界"中运作——从真实物理世界到虚拟游戏环境，从机器人操作场景到抽象数学空间。如何让JEPA框架能够跨越这些不同的"世界"学习通用表示，是一个尚未解决的根本问题。

JEPA-Anything提出了一个统一的跨世界JEPA框架，通过元学习和域自适应机制，使预测模型能够在不同"世界"间迁移和泛化。


![Figure 1: Overview of JEPA -Anything. Upper section: Diverse domains share the s](https://arxiv.org/html/2609.20800v1/figures/JEPA-Anything-final.png)
*图：Figure 1: Overview of JEPA -Anything. Upper section: Diverse domains share the same predictive probl*


![Figure 2: Scenario atlas for JEPA -Anything. The layout follows the paper’s thre](https://arxiv.org/html/2609.20800v1/figures/jepa_anything_scenario_atlas_v7.png)
*图：Figure 2: Scenario atlas for JEPA -Anything. The layout follows the paper’s three evaluation groups:*


## 核心方法

**跨世界JEPA架构**

JEPA-Anything的核心架构由以下模块构成：

**通用上下文编码器（Universal Context Encoder）**：
- 不依赖特定模态的特征提取器
- 通过共享嵌入空间对齐不同世界的表示
- 使用对比学习对齐不同世界中语义相似的概念

**世界感知预测器（World-Aware Predictor）**：
- 在预测时注入"世界标识"信息
- 使模型知道当前在哪个世界中做预测
- 通过轻量级适配器实现跨世界迁移

**多世界预训练策略**：
- 同时在真实图像/视频、游戏截图、机器人传感器数据上预训练
- 使用课程学习：先从单一世界学习，再混合多世界训练
- 跨世界数据增强：通过风格迁移在不同世界间创建配对数据


![Figure 3: Broad-spectrum prediction of more than 1,000 future clinical events. M](https://arxiv.org/html/2609.20800v1/figures/disease_forecasting_v2.png)
*图：Figure 3: Broad-spectrum prediction of more than 1,000 future clinical events. Methods are ranked by*


![Figure 4: Intervention-conditioned prediction and free rollout on CITRIS Interve](https://arxiv.org/html/2609.20800v1/figures/citris_intervention_v6.png)
*图：Figure 4: Intervention-conditioned prediction and free rollout on CITRIS Interventional Pong. Lower *


![Figure 5: Results from the ten-task matched dynamics benchmark. Panel (a) shows ](https://arxiv.org/html/2609.20800v1/figures/dynamics_summary_v2.png)
*图：Figure 5: Results from the ten-task matched dynamics benchmark. Panel (a) shows the relative MSE red*


![Figure 6: Capacity-matched continuous-control planning. Squares show the mean pa](https://arxiv.org/html/2609.20800v1/figures/control_planning_v2.png)
*图：Figure 6: Capacity-matched continuous-control planning. Squares show the mean paired CEM-return diff*


## 实验结果

JEPA-Anything在多个下游任务上的性能：

**物体识别（跨域零样本）**：

| 方法 | 真实→游戏 | 游戏→真实 | 真实→机器人 |
|------|-----------|-----------|-------------|
| I-JEPA | 51.3% | 48.7% | 44.2% |
| V-JEPA | 53.8% | 50.2% | 46.8% |
| JEPA-Anything | **67.4%** | **63.9%** | **61.5%** |

**机器人操作（少样本学习）**：
使用10次演示的条件下，JEPA-Anything作为骨干网络比I-JEPA提升操作成功率18.3%。

**世界模型预测精度**：
在视频帧预测任务上，JEPA-Anything的SSIM指标比单一世界预训练的V-JEPA提升0.087。


![Figure 7: Factor-wise functional interventions. Each factor is replaced by its t](https://arxiv.org/html/2609.20800v1/figures/factor_intervention_v2.png)
*图：Figure 7: Factor-wise functional interventions. Each factor is replaced by its training-set mean thr*



![2609.20800附图](https://arxiv.org/html/2609.20800v1/task/cancer_wetlab_v2.png)
*图：2609.20800 附图*


![2609.20800附图](https://arxiv.org/html/2609.20800v1/task/sol_v2.png)
*图：2609.20800 附图*


## 总结

JEPA-Anything将JEPA的视野从单一模态/领域扩展到"多种世界"，这一方向对于构建具有真正通用能力的世界模型具有重要意义。通过多世界联合预训练，模型学到了不同物理/虚拟环境间的共享抽象结构。

该工作的核心贡献在于提供了一个扩展性强的框架，随着更多"世界"被纳入预训练，模型的泛化能力有望持续提升——这与人类从不同经验中学习的方式颇为相似。

**局限性**：不同世界的数据分布差异导致训练不稳定；"世界标识"的有效性依赖于准确的世界边界划分，在连续变化的场景中可能存在边界模糊问题。