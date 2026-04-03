---
layout: post
title: "UniDriveVLA：统一理解、感知与行动规划的自动驾驶视觉-语言-动作模型"
date: 2026-04-04
categories: [论文解读, 自动驾驶]
tags: [自动驾驶, VLA模型, Mixture-of-Transformers, 空间感知, 端到端驾驶]
---

> 📄 **论文**：UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving
> 🔗 **arXiv**：[2604.02190](https://arxiv.org/abs/2604.02190)
> 🏢 **机构**：小米研究院（github.com/xiaomi-research/unidrivevla）

## 一句话总结

UniDriveVLA 通过 Mixture-of-Transformers 架构将理解、感知和行动三个专家模块解耦，彻底解决了自动驾驶VLA模型中空间感知与语义推理的表示冲突问题，在nuScenes和Bench2Drive两个基准上达到最先进水平。

## 背景与问题

Vision-Language-Action（VLA）模型正成为端到端自动驾驶的核心技术路线，但存在一个根本性矛盾：**空间感知能力**（精确的3D目标检测、地图构建）和**语义推理能力**（驾驶场景理解、自然语言交互）对特征表示的需求截然相反，难以在同一套共享参数中兼顾。

现有两类方案均存在明显缺陷：
- **2D VLM方案**：直接使用2D视觉语言模型，语义推理强但空间感知弱，3D感知精度低
- **3D增强方案**：额外引入3D表示，虽提升了空间感知，但会破坏VLM的语义推理能力

作者通过实验证明：在共享权重解码器中，LLM token与感知token的余弦相似度随层数加深趋向1，出现"特征崩塌"现象，两类任务的表示被强制对齐，反而两者都受损。

## 核心方法

**Mixture-of-Transformers（MoT）架构**

UniDriveVLA 的核心创新是将模型参数**解耦**为三个专业化专家模块，每个专家独立维护其表示空间：

1. **理解专家（Understanding Expert）**：因果掩码语义推理，处理驾驶场景理解和语言交互
2. **感知专家（Perception Expert）**：稀疏空间感知，专门处理3D目标检测、地图、运动预测等任务
3. **行动专家（Action Expert）**：基于Flow-Matching的轨迹生成

**掩码联合注意力（Masked Joint Attention）**

三个专家通过精心设计的注意力掩码协调：
- 理解token：仅因果掩码
- 感知token：可关注理解token（利用语义信息辅助感知）
- 行动token：聚合语义+空间双重上下文

![VLA范式对比](https://arxiv.org/html/2604.02190/x2.png)
*图1：三种VLA范式对比——2D VLA、3D增强型 vs UniDriveVLA（MoT解耦）*

![特征崩塌问题](https://arxiv.org/html/2604.02190/x3.png)
*图2：（左）共享权重解码器中余弦相似度随层数趋向1，出现特征崩塌；（右）MoT架构保持低相似度，维持任务专一性*

![UniDriveVLA架构](https://arxiv.org/html/2604.02190/x5.png)
*图3：UniDriveVLA整体架构——基于Qwen3-VL（SigLIP-2编码器+Qwen3 LM），三个MoT专家通过掩码联合注意力协调*

**三阶段渐进式训练**

1. 大规模多模态预训练（驾驶数据：通用数据 = 3:7）
2. 联合优化（LoRA + 0.5× VLM学习率）
3. 冻结VLM，单独精调感知/行动专家 + 运动目标

基础模型：Qwen3-VL（SigLIP-2视觉编码器 + Qwen3语言模型）。

![掩码联合注意力](https://arxiv.org/html/2604.02190/x6.png)
*图4：掩码联合注意力机制详解——感知token仅关注理解token，行动token聚合双路上下文*

## 实验结果

**Bench2Drive 闭环测试：**

| 方法 | 驾驶得分↑ | 成功率↑ | 效率↑ |
|------|----------|--------|------|
| DriveMOE | 74.22 | 48.64% | 175.96 |
| Orion | 77.74 | 54.62% | 151.48 |
| **UniDriveVLA** | **78.37** | 51.82% | **198.86** |

在合并（38.75%）和超车（80.00%）等多能力评估中达到最佳。

**nuScenes 规划（无ego状态）：**

| 方法 | 平均L2↓ |
|------|--------|
| SparseDrive | 0.55m |
| FSDrive | 0.53m |
| **UniDriveVLA-Large** | **0.51m** |

**nuScenes 感知：**
- 目标检测 mAP: 0.407，NDS: 0.460
- 地图 mAP: 0.535

**MoT vs 共享权重消融：**

| 架构 | 通用VQA | DriveBench | NDS | L2 |
|------|--------|------------|-----|-----|
| 共享权重 | 31.1% | 50.8% | 0.437 | 0.641 |
| **MoT** | **45.5%** | **54.9%** | **0.439** | **0.533** |

MoT在语义理解（VQA）和空间感知（NDS）上均优于共享权重方案。

## 总结

UniDriveVLA 通过 Mixture-of-Transformers 架构的专家解耦，从根本上消除了VLA模型中空间感知与语义推理的表示冲突，证明了"专业化分工"而非"强制融合"才是统一驾驶智能的正确路径。

局限性方面，三个专家模块的协同训练增加了优化复杂度；当前成功率（51.82%）在Bench2Drive上未超过Orion（54.62%），说明行动规划的鲁棒性仍有提升空间；此外，MoT架构的参数量和推理开销相比单一模型有所增加，工程部署需要额外优化。
