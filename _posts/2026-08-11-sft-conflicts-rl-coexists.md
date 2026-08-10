---
layout: post
title: "SFT冲突，RL共存：大语言模型多任务学习的理论与实证分析"
date: 2026-08-11
categories: [论文解读, 大语言模型]
tags: [多任务学习, SFT, 强化学习, GRPO, 知识蒸馏, LLM]
---

> 📄 **论文**：SFT Conflicts, RL Coexists: A Theoretical and Empirical Analysis of Multi-Task Learning for LLMs
> 🔗 **arXiv**：[2608.03573](https://arxiv.org/abs/2608.03573)
> 🏢 **机构**：中国科学院自动化研究所 (CASIA)
> 💻 **代码**：[GaryStack/Parallel-RL](https://github.com/GaryStack/Parallel-RL)

## 一句话总结

多任务SFT训练会导致严重的任务冲突（平均性能下降23.1%），而多任务RL训练却能实现任务间的稳定共存（平均性能提升24.9%），二者差异源于参数更新的稀疏性与正交性。

## 背景与问题

大型语言模型在多任务推理上的提升是迈向AGI的关键目标。现有做法通常采用混合数据SFT（Supervised Fine-Tuning）或多阶段RL（Reinforcement Learning）来实现多任务能力增强。然而，这两种方式的效果差异巨大。

实验基于 DeepSeek-R1-Distill-Qwen-1.5B，在数学（Math）、科学（Science）、编程（Coding）和逻辑（Logic）四个任务上进行多任务训练。结果显示：多阶段SFT导致平均23.1%的性能下降，而多阶段RL则带来平均24.9%的性能提升。这一鲜明对比促使作者深入探究其背后的机理。

## 核心方法

### 参数层面实证分析

研究发现了两个关键的参数层面差异：

**观察1——更新幅度与稀疏性：**
- RL的平均L₂范数约为3×10⁻²，而SFT约为7.4，相差超过100倍
- RL中仅约20%的参数超过10⁻⁵幅值，而SFT高达93%

**观察2——跨任务正交性：**
- RL跨任务参数更新的余弦相似度约为10⁻⁵（近似正交）
- SFT的余弦相似度约为10⁻¹到1.0（高度相关甚至相反）

![论文核心图示](https://arxiv.org/html/2608.03573/x1.png)
*图1：SFT与RL在多任务学习中的行为差异*

![参数更新分析](https://arxiv.org/html/2608.03573/x2.png)
*图2：参数层面的稀疏性与正交性对比*

### 理论分析

论文从梯度公式角度给出理论解释：

- **SFT梯度（离线策略）**：g_SFT = E[∇_θ log π_θ(y|x)]，y ~ π_expert
- **RL梯度（在线策略）**：g_RL = E[A(x,y) ∇_θ log π_θ(y|x)]，y ~ π_θ

GRPO中使用标准化优势 Â_k = (r_k − μ_r)/σ_r，具有零和性质（Σ Â_k = 0）。这一特性消除了均值方向的干扰，将任务间干扰转化为残差内积，远小于SFT中的绝对内积。

**定理4.5——梯度干扰上界：**
- SFT（范数受限）：|I_SFT(i,j)| ≤ M_i · M_j
- RL（方差受限）：|I_RL(i,j)| ≤ V_i · V_j

由于V_i ≪ M_i，RL产生的任务间干扰在理论上更小。

### Parallel-RL 框架

基于RL跨任务更新近似正交的特性，论文提出 **Parallel-RL**：各任务独立训练后合并：

$$W_{final} = W_{base} + \mathcal{M}(\Delta W_1, \ldots, \Delta W_N)$$

提供四种合并变体：
1. **Naive Parallel-RL**：直接求和/平均
2. **TIES Parallel-RL**：稀疏化合并
3. **SVD Parallel-RL**：仅保留rank-1 SVD分量
4. **Adapted Parallel-RL**：合并后使用5%数据微调

![Parallel-RL框架](https://arxiv.org/html/2608.03573/x3.png)
*图3：Parallel-RL框架示意图*

## 实验结果

### 多任务训练策略对比（DeepSeek-R1-Distill-Qwen-1.5B）

| 策略 | Math | Science | Logic | Code | 平均变化 |
|------|------|---------|-------|------|---------|
| Base Model | 83.1 | 34.9 | 31.0 | 15.0 | — |
| Mixed Data SFT | 84.6 | 38.9 | 34.0 | 16.0 | +2.4% |
| Multi-Stage SFT | 78.2 | 31.1 | 9.0 | 14.3 | **−23.1%** |
| Mixed Data RL | 85.2 | 43.2 | 37.0 | 15.7 | +8.3% |
| Multi-Stage RL | 86.6 | 49.3 | 43.0 | 17.3 | **+24.9%** |

### Parallel-RL 主要结果（DeepSeek-R1-Distill-Qwen-1.5B）

| 方法 | MATH500 | AIME2025 | MMLU | GPQA | KK | LCB | ΔBase | 保留率 |
|------|---------|---------|------|------|-----|-----|-------|--------|
| Base Model | 82.0 | 26.9 | 34.9 | 32.3 | 31.0 | 15.0 | — | — |
| Single-Task RL | 87.4 | 32.7 | 51.8 | 39.9 | 44.0 | 21.6 | +9.3 | — |
| Multi-Stage SFT | 71.0 | 21.8 | 33.2 | 23.2 | 11.0 | 12.1 | −8.3 | — |
| Multi-Stage RL | 87.8 | 33.1 | 52.8 | 40.4 | 48.0 | 21.0 | +10.2 | — |
| Naive Parallel-RL | 86.4 | 32.3 | 48.0 | 37.4 | 39.0 | 18.4 | +6.6 | 94.2% |
| TIES Parallel-RL | 87.6 | 32.5 | 49.2 | 38.4 | 43.0 | 19.7 | +8.0 | 97.4% |
| **Adapted Parallel-RL** | **88.6** | **33.8** | 50.9 | **41.4** | **49.0** | **22.5** | **+10.7** | **103.2%** |

Adapted Parallel-RL 在大多数基准上超越了多阶段RL和单任务RL，且保留率达到 **103.2%**，意味着合并后性能超过了分别训练的单任务模型之和！

## 总结

本文揭示了SFT与RL在多任务学习中的本质差异：SFT梯度干扰是"范数受限的"，任务间存在严重冲突；而RL由于优势归一化和在线策略采样，产生稀疏、近似正交的参数更新，实现任务间共存。

基于此，Parallel-RL框架利用RL更新的正交性，实现各任务的独立训练与合并，不仅保持了多阶段RL的性能优势，还显著提升了训练效率与灵活性。该框架的核心贡献在于将理论分析直接转化为实践方法，Adapted Parallel-RL甚至在多个基准上超越了顺序多阶段RL训练，展示了参数合并方法在强化学习场景下的巨大潜力。

局限性在于，实验主要基于1.5B和7B规模的模型，对更大模型族的泛化性还需进一步验证；此外，当前仅在四类推理任务上进行了验证，对更多样化任务的适用性有待探索。
