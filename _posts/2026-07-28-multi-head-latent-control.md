---
layout: post
title: "Multi-Head Latent Control：LLM Agent决策的统一控制接口"
date: 2026-07-28
categories: [论文解读, AI Agent]
tags: [LLM Agent, 模型路由, 推理控制, 隐状态, 工具使用, 决策优化]
---

> 📄 **论文**：Multi-Head Latent Control: A Unified Interface for LLM Agent Decision Making
> 🔗 **arXiv**：[2607.14277](https://arxiv.org/abs/2607.14277)
> 🏢 **机构**：阿尔伯塔大学 + 华为加拿大

## 一句话总结

Multi-Head Latent Control 是一个轻量级层，通过读取冻结 LLM 的隐状态轨迹来预测部署时控制信号，可在不修改模型参数的情况下将大模型使用量减少最高 90.7%，同时保持大模型的大部分性能。

## 背景与问题

大语言模型作为 Agent 部署时，可靠的智能体行为需要的不只是下一个 Token 预测。在推理时，Agent 需要决定：
- 是继续当前推理还是移交给更强的模型？
- 是否需要请求更多信息？
- 是否应该调用外部工具？
- 是否应该在当前设置下弃权？

现有方法的局限：
- **Prompt 级路由**：基于输入信号，无法利用模型自身的生成过程信息
- **外部编排**：成本高，随模型骨干演化难以维护
- **任务特定微调**：主要依赖输入侧信号，泛化性有限

核心问题：**这些控制决策能否直接从模型的隐状态生成过程中推断出来？**

## 核心方法

![Multi-Head Latent Control整体框架](https://arxiv.org/html/2607.14277v1/images/main_fig_v3.png)
*图1：Multi-Head Latent Control 作为冻结基础模型的内在控制接口。(a) 从冻结主模型的隐状态轨迹预测控制信号*

论文提出的 **Multi-Head Latent Control** 包含两个预测头：

### 能力预测头（Capability Head）

预测当前模型是否能解决该实例，或者是否应该移交给更强的协作模型。

训练数据：从同一冻结 LLM 骨干的潜在追踪中提取，**无需修改模型**。

![训练数据构成](https://arxiv.org/html/2607.14277v1/images/training_mix_donut.png)
*图3：能力头的 120K 训练混合数据构成——有意跨越视觉问答、推理、参数知识等多类任务*

### 分辨率预测头（Resolution Head）

预测合适的分辨率决策：
- **Clarification**：请求澄清更多信息
- **Tool Use**：调用外部工具
- **Abstention**：在当前设置下弃权
- **Direct Answering**：直接回答

两个头都只在来自同一冻结 LLM 骨干的潜在追踪上训练，支持**事后适配**（Post Hoc Adaptation），无需修改模型。

### 隐状态利用

设模型隐藏大小为 $d$，层 $\ell$ 的隐状态序列为 $H^{(\ell)} = [h_1^{(\ell)};...;h_N^{(\ell)}] \in \mathbb{R}^{N \times d}$，与生成输出 Token 对齐。控制头从这个隐状态轨迹读取信号。

## 实验结果

### 模型路由（成本-性能权衡）

![整体成本-性能权衡](https://arxiv.org/html/2607.14277v1/x1.png)
*图2：两个路由系统的整体成本-性能权衡：Qwen3.5-9B → Qwen3.5-27B-Thk（左）和 Qwen3-VL-4B-Thk → Qwen3-VL-32B-Thk（右）*

| 场景 | 大模型使用量减少 | 性能保留 |
|---|---|---|
| AndroidWorld（长程智能体任务） | **最高 90.7%** | 保留大部分 |
| 跨基准平均 | **27-53%** | 保留大部分 |
| 工具调用决策 | - | +158% 相对分数提升 |
| 减少遗漏必要工具调用 | - | 减少 **65.5%** |

### 与模型置信度对比

![能力头分数 vs 模型Token置信度](https://arxiv.org/html/2607.14277v1/Headvsconf.png)
*图4：能力头分数与 Qwen3.5-9B 模型 Token 置信度对比——能力头从隐状态中捕获了 Token 置信度无法捕获的信息*

关键发现：
- 能力头从隐状态轨迹中捕获了 Token 级置信度之外的丰富信息
- 即使是部分生成（前缀），也可以进行早期的能力预测
- 控制信号对不同骨干家族和基准套件均有效

## 总结

Multi-Head Latent Control 展示了隐状态轨迹可以作为轻量级控制接口的底座，为冻结基础模型提供实用的部署时控制能力。其核心优势在于无需修改模型就可以适配，且学到的控制信号在多模型协作、长程智能体执行、工具使用等场景下均能提升部署效率。

代码开源：https://github.com/Amirhosein-gh98/Multi-Head-Latent-Control
