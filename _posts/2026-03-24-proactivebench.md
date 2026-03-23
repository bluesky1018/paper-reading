---
layout: post
title: "ProactiveBench：多模态大语言模型主动性基准测试"
date: 2026-03-24
categories: [论文解读, 多模态]
tags: [多模态大语言模型, 基准测试, 主动视觉, 人机协作, 遮挡识别, MLLM]
---

> 📄 **论文**：ProactiveBench: Benchmarking Proactiveness in Multimodal Large Language Models
> 🔗 **arXiv**：[2603.19466](https://arxiv.org/abs/2603.19466)
> 🏢 **机构**：University of Trento, University of Bergamo, Inria Grenoble, Bruno Kessler Foundation

## 一句话总结
ProactiveBench 提出了首个评估多模态大语言模型（MLLM）主动性的基准测试，通过 22 个 MLLM 的评测揭示了当前模型在需要请求用户干预场景下的严重不足，并验证了通过数据训练可以学习主动性。

## 背景与问题

有效的协作始于知道何时寻求帮助。当人类试图识别被遮挡的物体时，会主动要求他人移开遮挡物。那么，多模态大语言模型（MLLM）能否表现出类似的"主动"行为——在视觉信息不足时请求用户进行简单干预？

现有 MLLM 评测体系主要关注模型在给定视觉输入下的感知和理解能力，却忽视了一个重要的能力维度：**当视觉信息不足以回答问题时，模型能否主动请求改善输入条件？** 例如，移开遮挡物、提升图像质量、从不同角度拍摄等。

![ProactiveBench Overview](https://arxiv.org/html/2603.19466v1/x1.png)
*图1：ProactiveBench 概览——测试 MLLM 是否能在视觉输入需要人工干预时主动请求帮助。*

## 核心方法

ProactiveBench 将"主动性"定义为：能够提供正确答案，或者建议能使查询可回答的用户干预动作。

**基准构建：**
- 复用 7 个现有数据集，涵盖：ROD（旋转对象检测）、VSOD（视频显著目标检测）、MVP-N（多视角对象识别）、ImageNet-C（图像质量退化）、QuickDraw（草图识别）、ChangeIt（视频变化检测）、MS-COCO（遮挡对象识别）
- 设计单轮和多轮交互场景
- 创建需要用户干预才能使查询可回答的场景

**评测设置：**
- **多选问答（MCQA）**：建模为马尔可夫决策过程，模型从预定义动作集（主动建议、弃权、类别选项）中选择
- **开放式生成（OEG）**：模型自由生成响应，使用 Qwen3-8B 作为评判员

![ProactiveBench Examples](https://arxiv.org/html/2603.19466v1/x2.png)
*图2：ProactiveBench 中的典型场景示例，展示不同干预类型（旋转、揭开遮挡物、提升清晰度等）。*

## 实验结果

对 22 个 MLLM 进行系统评测，包括：LLaVA-1.5/NeXT/OV、SmolVLM2、Idefics3、InstructBLIP、Qwen2.5-VL（3B/7B/32B/72B）、InternVL3 等开源模型，以及 GPT-4o、Claude 3.5 等闭源模型。

**主要发现：**
1. **现有 MLLM 普遍缺乏主动性**，倾向于弃权或产生幻觉，而非请求有用的用户干预
2. **给予提示可提升主动性**，但准确率提升有限（边际增益）
3. **在对话历史和少样本示例条件下**，动作分布发生偏移，导致准确率下降
4. **通过数据训练可以学习主动性**：在合成的主动性训练数据上微调可显著提升主动行为

![Results Comparison](https://arxiv.org/html/2603.19466v1/x3.png)
*图3：22 个 MLLM 在 ProactiveBench 上的评测结果，横轴为主动性得分，纵轴为准确率。*

| 模型 | 主动率（MCQA） | 准确率 | 主动率（OEG） |
|------|-------------|--------|------------|
| GPT-4o | 较高 | 中等 | 较高 |
| Qwen2.5-VL-72B | 中等 | 较高 | 中等 |
| LLaVA-OV-72B | 较低 | 中等 | 较低 |
| 大多数小型模型 | 极低 | 较低 | 极低 |

![Proactiveness Analysis](https://arxiv.org/html/2603.19466v1/x4.png)
*图4：分析主动性的不同影响因素，包括提示设计、历史信息和少样本示例。*

## 总结

ProactiveBench 揭示了当前 MLLM 评测体系中的盲点——模型在视觉感知上的能力并不等同于在协作场景中的实用性。当视觉信息不足时，一个真正有用的 AI 助手应当能够主动引导用户改善输入条件，而非简单弃权或产生错误输出。

论文的贡献在于：(1) 提出第一个系统评测 MLLM 主动性的基准；(2) 对 22 个模型进行全面评测，揭示当前模型的普遍不足；(3) 证明主动性是可学习的能力，为未来改进提供了方向。局限性方面，当前的主动行为主要局限于物理干预（移动遮挡物、调整视角等），未来需要探索更复杂的协作场景。
