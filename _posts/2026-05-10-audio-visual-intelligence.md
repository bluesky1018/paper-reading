---
layout: post
title: "大型基础模型中的音视频智能：综合综述"
date: 2026-05-10
categories: [论文解读, 多模态学习]
tags: [音视频, 多模态, 基础模型, 综述, 生成模型]
---

> 📄 **论文**：Audio-Visual Intelligence in Large Foundation Models: A Comprehensive Survey
> 🔗 **arXiv**：[2605.04045](https://arxiv.org/abs/2605.04045)
> 🏢 **机构**：National University of Singapore, University of Oxford, University of Toronto, UT Dallas, HKUST, Microsoft Research, University of Rochester

## 一句话总结

本文提供了大型基础模型视角下音视频智能（AVI）的首篇全面综述，建立了涵盖理解、生成、交互三大类任务的统一分类体系，系统梳理方法论基础、数据集和开放挑战。

## 背景与问题

音视频智能（Audio-Visual Intelligence, AVI）已成为人工智能的核心前沿领域，将听觉和视觉模态桥接起来，使机器能够在多模态真实世界中进行感知、生成和交互。在大型基础模型时代，音视频的联合建模变得日益关键——不仅用于理解，还用于跨动态时序信号的可控生成和推理。

近期的工业进展，如 Meta MovieGen 和 Google Veo-3，突显了统一音视频架构的重要性。然而，尽管进展迅速，现有文献仍然碎片化：
- 任务多样，分类体系不统一
- 评估实践异质化，难以系统比较
- 缺乏从大型基础模型视角的统一综合

本文旨在填补这一空白，提供第一篇全面梳理基础模型时代 AVI 的综述。

## 分类体系

本文建立了三层统一任务分类体系：

![AVI 统一分类体系](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2605.04045/fig_p1_4.png)
*图1：音视频智能的统一任务分类体系，涵盖理解、生成和交互三大类。*

### 第一类：理解世界——音视频感知
- 语音识别（Speech Recognition）
- 声音定位（Sound Localization）
- 音视频事件检测
- 音视频语义分割
- 情感识别等

### 第二类：创造世界——音视频生成
- 音频驱动的视频合成（Audio-Driven Video Synthesis）
- 视频到音频生成（Video-to-Audio）
- 语音驱动人脸动画
- 文本到视频+音频联合生成

### 第三类：与世界交互——统一感知与生成
- 对话系统（Dialogue）
- 具身交互（Embodied）
- 智能体接口（Agentic）

## 方法论基础

### 模态 Tokenization

![模态表示](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2605.04045/fig_p2_1.png)
*图2：音频和视觉模态的数据表示与 Tokenization 方法总览。*

**音频表示**：
- 波形（Waveform）
- 梅尔频谱（Mel Spectrogram）
- 离散音频 Token（如 EnCodec）

**视觉表示**：
- 图像 Patch
- 视频帧序列
- 视觉 Token（如 CLIP、DINO）

### 跨模态融合

![跨模态融合架构](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2605.04045/fig_p3_1.png)
*图3：各种跨模态融合架构，从早期融合到晚期融合，以及基于 Transformer 的注意力融合。*

### 生成方法

![生成方法对比](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2605.04045/fig_p4_1.png)
*图4：三类主要生成方法的对比：自回归生成、扩散模型、掩码自回归。*

**自回归生成**：统一 token 序列建模，支持音视频联合生成
**扩散模型**：通过去噪过程生成高质量音视频
**掩码自回归**：通过掩码 token 预测进行生成

### 大型语言模型中心方法

![LLM 中心架构](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2605.04045/fig_p5_1.png)
*图5：LLM 中心的音视频方法架构，包括编码器+LLM（感知）、LLM+生成器（生成）和统一模型。*

四种范式：
1. **Encoder + LLM**：用于多模态感知
2. **LLM + Generator**：用于多模态生成
3. **统一模型**：联合感知和生成
4. **智能体系统**：交互式感知和生成

## 关键任务回顾

### 音视频感知

![感知任务](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2605.04045/fig_p6_1.png)
*图6：音视频感知任务综述，包括声音定位、分割和事件检测。*

### 音视频生成

![生成任务](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2605.04045/fig_p7_1.png)
*图7：音视频生成任务综述，从文本到视频+音频的联合生成。*

### 统一感知与生成

![统一模型](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2605.04045/fig_p8_1.png)
*图8：统一音视频感知与生成模型，支持对话、具身和智能体交互。*

## 开放挑战

本文识别了以下关键开放挑战：

| 挑战 | 描述 |
|------|------|
| 时序同步 | 音频与视觉信号的精确时序对齐 |
| 空间推理 | 音源空间定位与视觉场景的联合推理 |
| 可控性 | 细粒度控制生成的音视频内容 |
| 安全性 | 深度伪造（Deepfake）检测与防御 |
| 评估标准化 | 跨任务的统一评估协议 |

![代表性数据集与基准](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2605.04045/fig_p9_1.png)
*图9：代表性音视频数据集和基准综览。*

![评估指标](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2605.04045/fig_p10_1.png)
*图10：不同 AVI 任务的评估指标汇总。*

## 总结

本综述通过建立统一分类体系和系统方法论总结，为快速发展的音视频智能领域提供了权威参考。关键贡献是：首次从大型基础模型视角全面梳理 AVI，整合了理解、生成和交互三大维度，并明确标注了开放研究挑战。

项目主页：https://github.com/JavisVerse/Awesome-AVI
