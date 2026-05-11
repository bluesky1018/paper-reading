---
layout: post
title: "列表式策略优化（LPO）：将基于组的RLVR统一为响应单纯形上的目标投影"
date: 2026-05-12
categories: [论文解读, 大语言模型]
tags: [强化学习, RLVR, 策略优化, LLM推理, GRPO]
---

> 📄 **论文**：Listwise Policy Optimization: Group-based RLVR as Target-Projection on the LLM Response Simplex
> 🔗 **arXiv**：[2605.06139](https://arxiv.org/abs/2605.06139)
> 🏢 **机构**：多所高校与工业研究机构联合

## 一句话总结

LPO 揭示了 GRPO、Dr.GRPO 等主流基于组的 RLVR 方法的共同几何结构——均是在响应单纯形上进行隐式目标投影，并在此基础上提出显式的列表式策略优化，在逻辑、数学、编程、多模态几何四类推理任务上一致性地提升训练性能。

## 背景与问题

带验证奖励的强化学习（RLVR）已成为大语言模型推理后训练的标准范式，而基于组的策略梯度方法（Group-based Policy Gradient）因其实现简单、无需价值网络而广受采用：从每个提示采样一组响应，通过组内相对优势信号更新策略。

然而，GRPO、Dr.GRPO、MaxRL 等方法在目标温度设计上各有不同，缺乏统一的理论解释：
- 为什么这些方法有效？其背后是否有共同的数学结构？
- 它们的隐式目标是什么？能否做得更好？
- 如何在保持优化稳定性的同时最大化响应多样性？

## 核心方法

### 揭示隐式几何结构

本文的核心理论发现：**基于组的策略梯度方法均隐式地在响应单纯形（Response Simplex）$\Delta^{K-1}$ 上定义了目标分布，并通过一阶近似向其投影。**

对于每个提示 $x$，策略在 $K$ 个采样响应上诱导出一个列表分布：

$$P_\theta = \text{softmax}(s_\theta), \quad s_{\theta,k} = \log\frac{\pi_\theta(y_k|x)}{\pi_b(y_k|x)}$$

现有方法（GRPO、Dr.GRPO、MaxRL）的差异仅在于其目标温度 $\tau$ 的选择，各自对应不同的隐式 Gibbs 目标分布。

![响应单纯形上的目标投影几何结构](https://arxiv.org/html/2605.06139v1/x1.png)
*图：响应单纯形上的目标投影几何结构，展示现有方法的隐式目标与LPO的显式目标*

### 列表式策略优化（LPO）

LPO 将每次迭代显式分解为两步：

$$\underbrace{w^* = \arg\max_{w \in \Delta^{K-1}} \hat{J}(w)}_{\text{(i) 目标：瞄准哪个分布}} \qquad \underbrace{\theta' = \arg\min_\theta D(w^* \| P_\theta)}_{\text{(ii) 投影：如何向目标优化}}$$

**第一步：目标诱导（Listwise Gibbs目标）**

最优目标分布（Theorem 1）：

$$w^*_k = \text{softmax}(\phi)_k, \quad \phi_k = \frac{R_k}{\tau} + s_{t,k}$$

这是以预更新策略 $P_t$ 为基准、向高奖励响应倾斜的再权重分布。温度 $\tau$ 控制锐度：$\tau \to 0$ 时趋向最高奖励响应，$\tau \to \infty$ 时退化为基准策略。

**第二步：投影（散度最小化）**

通过最小化 $w^*$ 与 $P_\theta$ 之间的散度进行精确投影，而非依赖一阶近似：
- **前向KL（$\text{LPO}_{\text{fwd}}$）**：$D_{\text{KL}}(w^* \| P_\theta)$，产生有界、零和、自校正的投影梯度
- **反向KL（$\text{LPO}_{\text{rev}}$）**：$D_{\text{KL}}(P_\theta \| w^*)$，具有不同的结构特性

理论保证（Theorem 2）：LPO 在列表目标上单调改进，且投影梯度满足有界性、零和性和自校正性。

![LPO方法示意图](https://arxiv.org/html/2605.06139v1/x2.png)
*图：LPO的两步显式目标-投影框架，与现有方法一阶近似的对比*

![各方法隐式目标分析](https://arxiv.org/html/2605.06139v1/x3.png)
*图：现有方法（GRPO、Dr.GRPO、MaxRL）隐式目标温度的分析，以及LPO与它们的关系*

## 实验结果

在四类推理任务上评估，使用 1.5B–14B 不同规模的 LLM 骨干：

### 逻辑推理（Countdown 游戏）

| 方法 | Countdown-34 Pass@1 | Countdown-4 Pass@1 |
|------|---------------------|---------------------|
| GRPO | 基线 | 基线 |
| $\text{LPO}_{\text{fwd}}$（对应GRPO温度） | **+↑** | **+↑** |
| $\text{LPO}_{\text{rev}}$（对应GRPO温度） | **+↑** | **+↑** |

相同温度条件下，LPO 在每个基线的配对实验中均优于 PG 方法。

### 数学推理（MATH → AIME/AMC/MATH500 等）

在 Qwen3-1.7B 和 Qwen3-8B 骨干上，LPO 相对于匹配目标的 GRPO 基线在 AIME24、AIME25、AMC23、MATH500、Minerva Math、OlympiadBench 上**一致性提升**训练性能，同时内在保持优化稳定性和响应多样性。

### 编程推理（PRIME 代码数据集）

在 Qwen3-1.7B 上，LPO 同样优于 GRPO 基线。

### 多模态几何推理（Geometry3k）

在 Qwen2.5-VL-3B 上，LPO 实现了更稳定的训练曲线，Pass@k 指标提升显著。

![训练性能曲线对比](https://arxiv.org/html/2605.06139v1/x4.png)
*图：LPO与基线方法在逻辑推理任务上的训练性能曲线，LPO在匹配温度条件下一致性地表现更优*

![数学推理实验结果](https://arxiv.org/html/2605.06139v1/x5.png)
*图：数学推理任务上的比较，LPO在多个数学基准上实现了一致性改进*

![多任务消融与泛化](https://arxiv.org/html/2605.06139v1/x6.png)
*图：跨不同模型家族和规模的泛化实验，验证LPO的广泛适用性*

## 总结

LPO 的价值不仅在于提出了一个更好的优化算法，更在于提供了一个**统一的理论框架**来理解现有基于组的 RLVR 方法。通过将隐式近似替换为显式的两步目标-投影过程，LPO 在不改变温度（即不改变目标）的条件下，单纯通过更精确的投影步骤实现了一致性性能提升——这直接证明了精确投影相对于一阶近似的优越性。

局限性方面，LPO 的计算开销略高于 GRPO（需要计算精确散度最小化），在极大规模模型下的效率影响有待评估；此外，散度类型（前向KL vs 反向KL）的选择对不同任务的影响规律还需进一步研究。
