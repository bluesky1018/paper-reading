---
layout: post
title: "Astrolabe：面向蒸馏自回归视频模型的前向过程强化学习对齐框架"
date: 2026-03-24
categories: [论文解读, 视频生成]
tags: [视频生成, 强化学习, 自回归模型, 扩散模型, 人类偏好对齐, RLHF]
---

> 📄 **论文**：Astrolabe: Steering Forward-Process Reinforcement Learning for Distilled Autoregressive Video Models
> 🔗 **arXiv**：[2603.17051](https://arxiv.org/abs/2603.17051)
> 🏢 **机构**：HKUST, JD Explore Academy, HKU

## 一句话总结
Astrolabe 提出了一种高效的在线强化学习框架，专为蒸馏自回归视频生成模型设计，通过前向过程 RL 而非反向过程优化，无需重新蒸馏即可让模型更好地对齐人类视觉偏好。

## 背景与问题

近年来，扩散模型在视频生成领域取得了突破性进展，但其多步去噪过程导致推理延迟较高。为此，研究者提出了蒸馏自回归（Distilled AR）视频模型，通过分布匹配蒸馏（DMD）将双向视频扩散模型蒸馏为高效的自回归模型，利用 KV 缓存实现流式推理和实时生成。

然而，蒸馏过程仅保证学生模型模仿教师模型的分布，缺乏对人类偏好的优化。因此，生成输出中频繁出现伪影和不自然的运动动态，与人类视觉偏好不符。

将在线强化学习应用于对齐蒸馏流式视频模型面临独特挑战：已有的奖励引导蒸馏方法缺乏主动探索机制，而基于反向过程的 RL 需要沿采样轨迹进行对数概率估计，这使算法与特定求解器耦合，并需要存储中间轨迹状态，带来大量内存和计算开销，削弱了流式模型的效率优势。

## 核心方法

![Astrolabe 效果展示](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2603.17051/fig_p1_1.jpeg)
*图1：Astrolabe 在多种蒸馏流式视频模型上的效果，有效减少伪影并提升时序一致性。*

Astrolabe 包含三个核心创新：

**1. 前向过程强化学习（Forward-Process RL）**

基于 Negative-Aware Fine-Tuning 原则，通过直接对比推理端点处的正样本和负样本，建立隐式策略改进方向。该方法仅需干净的推理端点，无需反向过程展开（solver-specific unrolling）和完整轨迹存储，充分保留了流式架构的效率优势。

具体来说，前向过程 RL 损失定义为对比正负候选视频片段的分数差：
- 正样本（高奖励）：强化其生成概率
- 负样本（低奖励）：降低其生成概率
- 无需梯度通过整个推理轨迹反向传播

![Astrolabe 整体框架](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2603.17051/fig_p5_5.jpeg)
*图2：Astrolabe 整体框架。左：使用滚动 KV 缓存的分组流式采样；中：片段级前向过程 RL 优化；右：多奖励设计与不确定性感知选择性正则化，用于缓解奖励黑客问题。*

**2. 流式训练方案（Streaming Training Scheme）**

为将对齐扩展到长视频，提出流式长调优（Streaming Long Tuning）：
- 通过滚动 KV 缓存逐步生成视频序列
- 仅在本地片段窗口上应用 RL 更新，同时以先前上下文为条件，保持长程一致性
- 历史梯度分离（Detached Historical Gradients），实现峰值内存的常数级控制

**3. 多奖励目标与不确定性感知正则化**

为缓解奖励黑客（Reward Hacking）问题：
- **多奖励系统**：结合视觉质量（VQ）、运动质量（MQ）和文本对齐（TA）三类奖励
- **不确定性感知选择性正则化**：基于不确定性评分动态选择是否施加 KL 惩罚
- **动态参考更新**：周期性更新参考策略，保持正则化效果

![Astrolabe 方法细节](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2603.17051/fig_p5_20.jpeg)
*图3：Astrolabe 中流式训练机制与奖励设计的详细架构。*

## 实验结果

实验在三种蒸馏自回归基础模型上进行：Self-Forcing、Causal-Forcing 和 LongLive。训练使用 48 张 NVIDIA H200 GPU，采用 LoRA 微调（rank=256, α=256）。

**短视频单提示生成（VBench 协议，946 个标准提示）：**

| 方法 | 视觉质量 | 运动质量 | 文本对齐 | 综合得分 |
|------|---------|---------|---------|---------|
| Self-Forcing | 基线 | 基线 | 基线 | 基线 |
| Self-Forcing + Astrolabe | ↑ | ↑ | ↑ | ↑ |
| Causal-Forcing | 基线 | 基线 | 基线 | 基线 |
| Causal-Forcing + Astrolabe | ↑ | ↑ | ↑ | ↑ |
| LongLive | 基线 | 基线 | 基线 | 基线 |
| LongLive + Astrolabe | ↑ | ↑ | ↑ | ↑ |

Astrolabe 在所有 Self-Forcing 变体上均一致提升性能。在 LongLive 和 Causal-Forcing 上观察到的类似增益，进一步证明了框架在不同基础架构上的泛化性。

**人类偏好评估（MovieGenBench 100 个多样提示）：**
- HPSv3 和运动质量评分均优于基线模型
- **精确保持原始检查点的推理速度**（无需重新蒸馏）

![定性对比](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/2603.17051/fig_p9_1.jpeg)
*图4：短视频单提示设置下的定性比较。与基线相比，Astrolabe 生成的视频纹理更清晰，运动连贯性更好。*

## 总结

Astrolabe 成功解决了蒸馏自回归视频模型的人类偏好对齐问题，其核心贡献在于：(1) 提出无需反向过程展开的前向 RL 框架，保留流式架构效率；(2) 设计流式长调优方案，以常数峰值内存处理长视频；(3) 实现多奖励联合优化并通过不确定性感知正则化有效防止奖励黑客。

该工作的意义在于为视频生成模型的后训练对齐提供了一种实用的工业级解决方案，无需昂贵的重新蒸馏即可改善生成质量，且对多种基础架构均有效。局限性方面，当前依赖于 VideoAlign 和 HPSv3 等外部奖励模型，其质量直接影响对齐效果，且对长视频中跨场景一致性的保证仍有待深入研究。
