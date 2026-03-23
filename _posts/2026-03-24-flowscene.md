---
layout: post
title: "FlowScene：基于多模态图整流流的风格一致室内场景生成"
date: 2026-03-24
categories: [论文解读, 3D生成]
tags: [3D场景生成, 整流流, 场景图, 室内设计, 风格一致性, 生成模型]
---

> 📄 **论文**：FlowScene: Style-Consistent Indoor Scene Generation with Multimodal Graph Rectified Flow
> 🔗 **arXiv**：[2603.19598](https://arxiv.org/abs/2603.19598)
> 🏢 **机构**：Peking University, Technical University of Munich, Beijing Jiaotong University, Beijing Digital Native Digital City Research Center, Theta Labs, Beijing Normal University

## 一句话总结
FlowScene 提出了一种基于多模态图整流流的三分支场景生成模型，能够同时生成场景布局、物体形状和物体纹理，在保持场景级风格一致性的同时实现对个体物体外观的精细控制。

## 背景与问题

场景生成具有广泛的工业应用，涵盖制造业和室内设计、VR/AR 内容创建、自动驾驶和机器人技术。这些场景要求高保真度和对几何及外观的精确控制，用户需要指定物体类别、语义及空间关系，以及所需的个体外观。

现有两类方法各有局限：
- **语言驱动检索方法**：从大型物体数据库中组合场景，但忽略了物体级控制，且常常无法保证场景级风格一致性
- **基于图的生成方法**：提供更高的可控性并通过显式建模关系来保证整体一致性，但现有方法难以生成高保真纹理结果，限制了实用性

![FlowScene Overview](https://arxiv.org/html/2603.19598v1/x1.png)
*图1：FlowScene 生成的室内场景示例，展示风格一致的布局、形状和纹理。*

## 核心方法

FlowScene 是一个以多模态图为条件的三分支场景生成模型，协同生成场景布局、物体形状和物体纹理。

**核心模块：多模态图整流流（Multimodal Graph Rectified Flow）**

整流流（Rectified Flow）学习时间依赖的速度场 $v_\theta(d, t)$，使 ODE $\dot{d}_t = v_\theta(d_t, t)$ 将数据分布传输到简单先验（如高斯分布）。

FlowScene 在此基础上设计了**多模态图整流流**模块：
- 在采样过程中**交换节点信息**，既满足个体条件又满足整体条件
- 通过图神经网络捕获物体间的语义和空间关系
- 对每个物体节点的布局、形状、纹理进行联合建模

**三分支设计：**
1. **布局分支**：生成场景中所有物体的位置、朝向和尺寸
2. **形状分支**：为每个物体生成 3D 网格/点云形状
3. **纹理分支**：为每个物体生成风格一致的纹理

![FlowScene Architecture](https://arxiv.org/html/2603.19598v1/x2.png)
*图2：FlowScene 整体架构，展示三分支设计和多模态图整流流模块。*

**多输入模式支持：**
- 基于文本描述从头生成场景
- 基于用户指定的物体和关系进行交互式生成
- 两种输入模式的混合使用

![FlowScene Method Details](https://arxiv.org/html/2603.19598v1/x3.png)
*图3：多模态图整流流的详细设计，展示节点信息交换机制。*

## 实验结果

在室内场景数据集上与多种基线方法进行对比，评估指标包括：场景级保真度（FID、FID_CLIP、KID）、物体级质量、风格一致性和人类偏好。

**场景级保真度（FID，越低越好）：**

| 方法 | FID ↓ | FID_CLIP ↓ | KID ↓ |
|------|-------|-----------|-------|
| 语言驱动检索方法 | 较高 | 较高 | 较高 |
| 图条件生成（基线） | 中等 | 中等 | 中等 |
| **FlowScene（ours）** | **最低** | **最低** | **最低** |

FlowScene 在真实感、可控性、风格一致性和人类偏好等维度上均优于竞争方法，同时相比基于扩散的流程显著提升了生成速度。

![FlowScene Results](https://arxiv.org/html/2603.19598v1/x4.png)
*图4：与基线方法的定量对比，FlowScene 在各项指标上均取得最佳性能。*

![FlowScene Qualitative](https://arxiv.org/html/2603.19598v1/x5.png)
*图5：生成场景的定性比较，FlowScene 生成的场景具有更高的视觉质量和风格一致性。*

![FlowScene Interactive](https://arxiv.org/html/2603.19598v1/x6.png)
*图6：交互式场景生成工作流，支持用户通过文字或交互选择来定制场景内容。*

## 总结

FlowScene 提出了一个优雅的解决方案来解决 3D 室内场景生成中的两大挑战：对个体物体的精细控制和跨物体的风格一致性。通过多模态图整流流模块，模型在采样过程中动态交换物体信息，实现了在满足局部（个体物体）约束的同时满足全局（场景整体）约束。

与基于扩散的方法相比，基于整流流的设计还带来了显著的效率提升。这一工作为面向工业应用的高质量 3D 场景生成提供了一个实用的技术路径。局限性方面，当前方法主要针对室内场景，对室外场景和更复杂的开放世界环境的泛化能力有待进一步研究。
