---
layout: post
title: "RationalRewards：推理奖励在训练和测试时同步提升视觉生成质量"
date: 2026-04-17
categories: [论文解读, 视觉生成]
tags: [强化学习, 奖励模型, 视觉生成, 图像编辑, RLHF, 推理奖励]
---

> 📄 **论文**：RationalRewards: Reasoning Rewards Scale Visual Generation Both Training and Test Time
> 🔗 **arXiv**：[2604.11626](https://arxiv.org/abs/2604.11626)
> 🏢 **机构**：HKUST、University of Waterloo、Alibaba

## 一句话总结

RationalRewards 提出了一种通过结构化多维度批判（rationale）代替不透明标量输出的奖励模型，在仅使用 5% 训练数据的情况下超越 Gemini 2.5 Flash，并实现了无需参数更新的测试时优化。

## 背景与问题

视觉生成模型（如文生图、图像编辑）的 RL 训练高度依赖于奖励模型的质量。传统的标量奖励模型存在两大核心问题：

**第一，奖励黑箱化**：标量输出无法提供可解释的反馈，模型无法理解"哪里不好"。这导致 RL 训练过程中容易出现**奖励黑客攻击（reward hacking）**——奖励分数持续上升，但视觉质量却在下降。

**第二，测试时优化困难**：标量奖励无法提供可操作的改进方向，难以在不更新参数的情况下提升生成质量。

此前的方法（如 EditReward、MultiReward）均采用标量输出，且需要大量标注好的偏好数据。如何在数据高效的情况下训练出能够提供结构化批判的奖励模型，是本文要解决的核心挑战。

## 核心方法

### RationalRewards 框架

RationalRewards 的核心思想是：**奖励模型应先生成结构化批判（rationale），再给出分数**。批判从四个维度评估图像质量：
- 文本忠实度（Text Faithfulness）
- 图像忠实度（Image Faithfulness）
- 物理与视觉质量（Physical/Visual Quality）
- 文字渲染（Text Rendering）

这样的设计带来了两种优化路径：
1. **训练时**：结构化批判作为 RL 奖励信号，提供比标量更密集的监督
2. **测试时**：通过 **Generate-Critique-Refine (GCR)** 循环迭代改进生成结果，无需任何参数更新

![RationalRewards 框架概览](https://arxiv.org/html/2604.11626/x2.png)
*图：RationalRewards 双空间优化框架——左侧为训练时 RL，右侧为测试时 GCR 循环*

### PARROT：从偏好数据中挖掘批判

如何在没有昂贵批判标注的情况下训练 RationalRewards？论文提出了 **PARROT（Preference-Anchored Rationalization）**，一个变分框架，将批判视为从偏好数据中恢复的**隐变量**。

PARROT 包含三个阶段：

| 阶段 | 目标 | 方法 |
|------|------|------|
| 1. 批判生成 | 教师模型生成锚定到偏好标签的批判 | Qwen3-VL-32B 带标签提示生成 |
| 2. 一致性过滤 | 保留能够独立恢复偏好的批判 | 去除标签提示后验证一致性 |
| 3. 预见性蒸馏 | 训练学生模型无需标签即可生成批判 | 8B 模型 KL 散度最小化 |

约 **72% 的批判** 通过了一致性过滤器，最终用于训练 8B 参数的 RationalRewards 模型。

![PARROT 三阶段流程](https://arxiv.org/html/2604.11626/x4.png)
*图：PARROT 框架的三阶段批判挖掘流程*

![奖励黑客攻击对比](https://arxiv.org/html/2604.11626/x3.png)
*图：RationalRewards（绿色）显示平滑收敛曲线，而标量奖励模型（红色）出现奖励上升但质量下降的现象*

## 实验结果

### 偏好建模性能（Table 1）

| 模型 | MMRB2 T2I | EditReward Edit | GenAI T2I |
|------|-----------|-----------------|-----------|
| Qwen3-VL-8B（基础） | 59.4 | 61.7 | 50.1 |
| EditReward-7B | — | 67.2 | 65.7 |
| **RationalRewards (8B)** | **64.2** | **70.3** | **80.1** |
| Gemini 2.5 Flash | 63.1 | 66.5 | 73.0 |
| Gemini 2.5 Pro | 70.5 | 71.3 | 78.9 |

RationalRewards（8B）以 **10–20× 更少的训练数据** 超越了所有开源模型，并在 GenAI T2I 上超过了 Gemini 2.5 Flash（80.1 vs 73.0）。

### 文生图 RL 训练（UniGenBench++）

| 模型 | Overall |
|------|---------|
| FLUX.1-dev（基础） | 60.97 |
| +MultiReward | 60.12 |
| +Qwen3-VL-32B | 66.53 |
| +RationalRewards | **70.34** |
| Qwen-Image（基础） | 78.36 |
| +RationalRewards | **82.60** |

### 图像编辑（ImgEdit-Bench / GEdit-EN）

| 模型 | ImgEdit Overall | GEdit G_O |
|------|----------------|-----------|
| Flux.1 Kontext 基础 | 3.52 | 6.51 |
| +RL (EditReward) | 3.66 | 6.88 |
| +RL (RationalRewards) | 3.84 | 7.37 |
| **+PT (RationalRewards)** | **4.01** | **7.23** |

![GCR 循环示意](https://arxiv.org/html/2604.11626/x7.png)
*图：测试时 Generate-Critique-Refine (GCR) 循环，无需参数更新即可迭代提升生成质量*

### 关键发现：隐在能力假说

测试时提示调优（PT）以仅约 **0.4 秒的 VLM 推理开销** 匹配甚至超越需要 **384 GPU 小时** RL 训练的效果。这支持了"隐在能力假说"：生成器已经具备生成高质量输出的能力，只是次优的提示未能激活这些能力。

## 总结

RationalRewards 提供了一个优雅的解决方案：通过结构化批判而非标量奖励，同时解决了奖励黑客攻击和测试时优化两大难题。仅用 5% 的训练数据，在多个基准上达到或超越闭源大模型（Gemini 2.5 Flash）。

主要局限性在于：测试时的 GCR 循环仍需要 VLM 推理开销，在批量生成场景下可能成为瓶颈；此外，四维评估框架的设计是手工的，未来可探索自适应的评估维度。
