---
layout: post
title: "LocateAnything：基于并行框解码的快速高质量视觉语言定位"
date: 2026-05-28
categories: [论文解读, 视觉语言模型]
tags: [视觉定位, 目标检测, VLM, 并行解码, NVIDIA]
---

> 📄 **论文**：LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding
> 🔗 **arXiv**：[2605.27365](https://arxiv.org/abs/2605.27365)
> 🏢 **机构**：NVIDIA、清华大学

## 一句话总结

LocateAnything 提出了并行框解码（PBD）机制，将边界框作为原子单元一步解码，同时构建了包含 1.38 亿训练样本的大规模数据集，在显著提升吞吐量的同时取得了最优的视觉定位精度。

## 背景与问题

视觉语言模型（VLMs）在视觉定位和检测任务中，通常将 2D 边界框序列化为多个独立的 1D token 进行自回归生成。这种逐 token 解码方式存在两个根本缺陷：一是破坏了框坐标之间固有的几何耦合关系，导致结构性幻觉；二是严格的串行解码造成了推理瓶颈，限制了实际部署效率。

现有方法如 Qwen-VL、InternVL 等虽然将定位任务统一为文本生成范式，但本质上将四个坐标值（x1, y1, x2, y2）作为相互独立的 token 生成，忽视了它们作为一个整体边界框的几何一致性。这不仅影响定位精度，尤其在高 IoU 阈值下表现明显，还导致推理延迟较高，难以满足实时应用需求。

此外，高精度视觉定位任务缺乏足够规模和多样性的训练数据，是制约模型性能的另一重要瓶颈。

## 核心方法

### 并行框解码（Parallel Box Decoding, PBD）

LocateAnything 的核心创新是将边界框作为"原子单元"进行并行解码，而非逐 token 串行生成。框内所有元素在同一步骤中被预测，保留了空间坐标的几何一致性，同时解锁了显著的并行性。

![LocateAnything 框架概览](https://arxiv.org/html/2605.27365v1/x1.png)
*图：LocateAnything 支持多种定位任务的并行框解码，顶部展示多样化定位任务，底部对比不同解码方式的速度差异*

### 基于块的输出表示

模型将定位输出表示为固定长度的块序列，包含四种功能块类型：语义块、坐标块、置信度块和结束块。这种结构化表示既保证了框几何的完整性，又支持高效的并行生成。

![Token解码方法对比](https://arxiv.org/html/2605.27365v1/x2.png)
*图：NTP（逐token生成）vs 标准MTP vs 本文PBD方法的对比，展示不同解码范式的分布差异*

![模型架构与块表示](https://arxiv.org/html/2605.27365v1/x3.png)
*图：LocateAnything架构及基于块的输出表示，展示语义、坐标、置信度和结束四种块类型*

### 联合 NTP-MTP 训练

模型采用联合训练策略，将下一个 token 预测（NTP）与块级多 token 预测（MTP）对齐。通过精心设计的注意力掩码，共享上下文和 NTP 流使用因果注意力，MTP 块则遵循块间因果模式。

![注意力掩码设计](https://arxiv.org/html/2605.27365v1/x4.png)
*图：联合NTP-MTP训练的注意力掩码设计*

### 按需推理机制

框架支持混合推理模式，可动态平衡解码吞吐量和鲁棒性。当并行解码遇到格式不规则或空间歧义时，模型自动回退到 NTP 重新解码，确保预测的可靠性。

![NTP重解码机制](https://arxiv.org/html/2605.27365v1/x5.png)
*图：校正性NTP重解码流程——当并行解码失败时的回退机制*

### LocateAnything-Data 数据集

构建了一个超过 1.38 亿训练样本的大规模数据集，涵盖多种感知任务，显著提升数据多样性。

![数据集概览](https://arxiv.org/html/2605.27365v1/x6.png)
*图：LocateAnything-Data数据集概览，饼图展示任务分布，底部展示典型样本*

## 实验结果

### LVIS 和 COCO 目标检测

![LVIS和COCO结果](https://arxiv.org/html/2605.27365v1/x7.png)
*表1：在LVIS和COCO上的性能对比，LocateAnything在多个指标上达到最优*

### 密集目标检测

![Dense200和VisDrone结果](https://arxiv.org/html/2605.27365v1/x8.png)
*表2：在Dense200和VisDrone密集目标检测基准上的结果*

### GUI 定位任务

![GUI定位结果](https://arxiv.org/html/2605.27365v1/x9.png)
*表3：GUI定位任务性能对比*

### 指代表达理解（REC）

![REC结果](https://arxiv.org/html/2605.27365v1/x11.png)
*表5：指代表达理解基准评测结果*

### 消融研究与速度分析

![消融研究和速度分析](https://arxiv.org/html/2605.27365v1/x13.png)
*图7：框排序策略消融（左）和解码速度对比（右），PBD相比竞争方法实现最高2.5×加速*

**关键性能指标：**
- 相比竞争方法，解码吞吐量提升最高 **2.5×**
- 在 COCO、LVIS、RefCOCO 等多个基准上达到 SOTA
- 支持多任务统一：目标检测、视觉定位、GUI操作、OCR、文档版面分析

### 定性结果

![定性结果](https://arxiv.org/html/2605.27365v1/x14.png)
*图8：定性结果展示，不同颜色表示不同查询类别（属性、部件、空间关系等）*

![指代表达定性对比](https://arxiv.org/html/2605.27365v1/x25.png)
*图12：与Qwen3-VL和Rex-Omni的指代表达理解定性对比，LocateAnything展现出更优的组合定位能力*

## 总结

LocateAnything 从根本上重新审视了 VLM 中视觉定位的解码范式。通过将边界框提升为原子几何单元而非1D token流，实现了训练监督与空间坐标固有耦合特性的统一。

该方法的主要贡献包括：（1）PBD 机制在保持精度的同时显著提升推理速度；（2）大规模、高多样性的 LocateAnything-Data 数据集；（3）支持从视觉定位到 GUI 操作、OCR 的多任务统一框架。

局限性方面，当前方法主要关注2D边界框定位，向3D感知和更复杂场景的扩展仍需进一步探索。此外，尽管 PBD 提升了并行性，但对于极密集场景（数百个目标）的实时处理仍是挑战。
