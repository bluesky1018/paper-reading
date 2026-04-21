---
layout: post
title: "OneVL：以答案级延迟实现超越显式CoT的潜在推理自动驾驶"
date: 2026-04-22
categories: [论文解读, 自动驾驶, 多模态]
tags: [视觉语言动作模型, 链式思维推理, 自动驾驶, 轨迹预测, 世界模型]
---

> 📄 **论文**：OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation
> 🔗 **arXiv**：[2604.18486](https://arxiv.org/abs/2604.18486)
> 🏢 **机构**：Xiaomi Embodied Intelligence（小米具身智能实验室）

## 一句话总结

OneVL 通过视觉世界模型解码器和语言辅助解码器对潜在推理 token 进行双重监督，首次让潜在 CoT 方法以答案级推理延迟全面超越显式 CoT 的轨迹预测精度。

## 背景与问题

在自动驾驶视觉语言动作模型（VLA）中，链式思维（Chain-of-Thought, CoT）推理已被证明能显著提升轨迹预测质量——通过逐步分析道路几何、障碍物运动和交通规则，模型能做出更合理的驾驶决策。

然而，显式 CoT 推理存在致命缺陷：**延迟过高**。生成自然语言推理链需要数千个 token 的自回归解码，严重不符合自动驾驶的实时性要求。以 AR CoT+Answer 为例，其延迟高达 6.58 秒（NAVSIM 基准），而安全驾驶通常要求 100ms 级别的响应时间。

为此，研究者提出了潜在 CoT 方法（如 COCONUT、CODI、SIM-CoT），将推理过程压缩到隐藏状态（latent space）中，避免生成冗长文本。但这些方法的表现普遍**低于显式 CoT**，甚至在某些基准上不如直接预测答案的基线模型。

**根本原因在哪里？** OneVL 的作者认为，纯语言潜在空间编码的是"符号抽象"，缺乏驾驶场景所需的空间关系和时序动态信息——道路几何、智能体运动、环境变化等关键信息在语言 latent 中严重欠表示。

## 核心方法

### 架构设计

OneVL 基于 **Qwen3-VL-4B-Instruct**（ViT + MLP 对齐器 + LLM），引入两类新型 latent token：

- **语言潜在 token**（𝒞t=2，实现为20个 token）：编码隐式语言推理
- **视觉潜在 token**（𝒞v=4，实现为35个 token）：编码空间/时序视觉推理

以及两个辅助解码器：

- **语言辅助解码器**：从 latent 隐藏状态重建 CoT 文本
- **视觉世界模型解码器**：预测未来 +0.5s 和 +1.0s 的场景帧（使用 IBQ tokenizer，131,072 词汇量）

### 联合训练目标

$$\mathcal{L} = \mathcal{L}_c + \lambda_l \cdot \mathcal{L}_l + \lambda_v \cdot \mathcal{L}_v$$

其中 λl=1.0，λv=0.1。这一设计迫使 latent token 同时捕获文本语义和视觉动态信息，而非仅优化最终轨迹误差。

### 四阶段训练流程

1. **初始化**：视觉解码器自监督预训练
2. **Stage 0**：主模型预热
3. **Stage 1**：辅助解码器预热（主模型冻结）
4. **Stage 2**：端到端联合微调

### 推理加速：Prefill 并行化

推理时，所有 latent token 置于 **prefill 阶段**，仅轨迹 token 自回归生成。辅助解码器在推理时完全丢弃。这使得 OneVL 的延迟与"仅输出答案"的基线几乎相同。

![OneVL精度-效率对比](https://arxiv.org/html/2604.18486/x1.png)
*图1：四个基准上 OneVL 的精度-效率对比。OneVL 以答案级延迟全面超越显式 CoT 方法。*

![三种CoT范式对比](https://arxiv.org/html/2604.18486/x2.png)
*图2：显式 CoT、隐式潜在 CoT 与 OneVL 双解码器方法的范式对比图。*

![OneVL完整架构](https://arxiv.org/html/2604.18486/x3.png)
*图3：OneVL 完整架构，展示 VLM 骨干、两类潜在 token 及双辅助解码器的协同工作方式。*

## 实验结果

### NAVSIM 基准（PDM-score ↑，延迟 ↓）

| 方法 | 模型规模 | PDM-score | 延迟(s) |
|------|---------|-----------|---------|
| AdaThinkDrive | 8B | 86.20 | — |
| LaST-VLA | 8B | 87.30 | — |
| AR Answer（基线） | 4B | 87.47 | 4.49 |
| AR CoT+Answer | 4B | 88.29 | 6.58 |
| COCONUT（潜在CoT） | 4B | 84.84 | 5.93 |
| CODI（潜在CoT） | 4B | 83.92 | 8.62 |
| **OneVL（本文）** | **4B** | **88.84** | **4.46** |

### ROADWork 基准（ADE/FDE ↓）

| 方法 | ADE | FDE | 延迟(s) |
|------|-----|-----|---------|
| AR Answer | 15.98 | 40.29 | 4.74 |
| AR CoT+Answer | 13.18 | 29.98 | 10.74 |
| COCONUT | 15.44 | 38.60 | 6.06 |
| **OneVL** | **12.49** | **28.80** | **4.71** |

### Impromptu 基准（ADE/FDE 单位：米）

| 方法 | ADE | FDE | 延迟(s) |
|------|-----|-----|---------|
| Impromptu VLA | 1.60 | 4.28 | 6.10 |
| AR CoT+Answer | 1.42 | 3.96 | 6.84 |
| **OneVL** | **1.34** | **3.70** | **4.02** |

### APR1 基准

| 方法 | 规模 | ADE | FDE | 延迟(s) |
|------|------|-----|-----|---------|
| Cosmos-Reason | 10B | 2.86 | 7.42 | — |
| AR CoT+Answer | 4B | 2.99 | 8.54 | 3.51 |
| **OneVL** | **4B** | **2.62** | **7.53** | **3.23** |

**OneVL 在4个自动驾驶基准上全部超越显式 CoT，且延迟与仅输出答案的基线相当，实现了精度与速度的双赢。**

## 总结

OneVL 的核心贡献在于揭示了潜在 CoT 在驾驶场景中表现不佳的深层原因——语言 latent 空间无法有效编码驾驶动态——并通过视觉世界模型解码器这一创新机制加以解决。让模型在学习推理的同时预测未来视觉帧，迫使其latent表征融合了空间感知和时序预测能力，这是本文最具启发性的设计思想。

值得注意的是，OneVL 的 4B 参数模型在大多数基准上已超越 8B 的竞争对手，体现了方法本身的高效性。

局限性方面，视觉世界模型解码器的训练需要额外的未来帧标注数据，这在某些场景下可能难以获取。此外，IBQ tokenizer 的131K词汇量带来了较高的显存开销，在边缘部署场景中需要进一步优化。

> 项目主页：[xiaomi-embodied-intelligence.github.io](https://xiaomi-embodied-intelligence.github.io)
