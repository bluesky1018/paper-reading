---
layout: post
title: "亚二次方架构深析：从应用到原理——xLSTM、Mamba-2 与 Gated DeltaNet 全面对比"
date: 2026-06-12
categories: [论文解读, 模型架构]
tags: [xLSTM, Mamba, 线性注意力, 序列建模, 亚二次方, Transformer替代]
---

> 📄 **论文**：On Subquadratic Architectures: From Applications to Principles
> 🔗 **arXiv**：[2606.12364](https://arxiv.org/abs/2606.12364)
> 🏢 **机构**：约翰内斯·开普勒大学林茨（JKU Linz）

## 一句话总结

首次在代码预训练、Transformer 蒸馏、时序基础模型三类复杂任务上系统比较 xLSTM、Mamba-2 和 Gated DeltaNet，并通过统一理论框架与合成任务实验揭示：**xLSTM 的优势源于同时具备累积（Accumulation）和状态追踪（State Tracking）两种基本能力**。

## 背景与问题

Transformer 的二次方注意力计算复杂度推动了亚二次方替代架构的发展，其中 xLSTM、Mamba-2 和 Gated DeltaNet 已成为现代混合语言模型的核心候选组件（如 Samba、Nemotron Nano、Kimi Linear、Olmo Hybrid）。

然而，现有比较大多局限于标准英文网页预训练和常识推理基准，这些基准上各架构差异微小且难以区分。本文的核心问题是：**在具有复杂依赖关系的任务上，三种架构的性能差异是什么？其根本原因是什么？**

## 核心方法

### 统一框架

将三种架构统一表达为带门控机制的线性注意力形式：

**基础线性注意力（分块并行）：**
$$\mathbf{H}[n] = (\mathbf{Q}[n]\mathbf{K}[n]^\top \odot \mathbf{M})\mathbf{V}[n] + \mathbf{Q}[n]\mathbf{C}_{(n-1)C}$$

**xLSTM[1:0]（mLSTM）：**
- 使用指数输入门 $i_t = \exp(\mathbf{w}_i \mathbf{x}_t)$，行为类似时序 softmax
- 独立的输入门和遗忘门，具备最灵活的权重校正能力
- **支持累积（Accumulation）**

**xLSTM[0:1]（sLSTM）：**
- 使用递归权重 **R**，支持前一时刻隐状态的非线性更新
- **支持状态追踪（State Tracking）**

**Mamba-2：**
- 输入门和遗忘门耦合（tied gates），类似 GRU 结构
- $f_t = (1 - \sigma(\mathbf{w}_\Delta \mathbf{x}_t))^a$，输入门近似线性
- **既不能有效累积，也不能有效追踪状态**

**Gated DeltaNet：**
- 包含正交投影状态变换：$\mathbf{I} - \mathbf{k}_t \otimes \mathbf{k}_t / \|\mathbf{k}_t\|^2$，明确覆写旧值
- 擅长检索任务，但覆写机制导致**无法累积历史信息**
- 负特征值参数化变体可部分改善状态追踪

### 实验设置

评估三种设置下的复杂依赖任务表现：
1. **代码专注语言模型预训练**：400M 参数混合模型，在 Nemotron-CC-Code-v1 上预训练
2. **Transformer 蒸馏**：以 Qwen3-4B-Instruct 为教师进行知识蒸馏
3. **时序基础模型预训练**：1M-80M 参数规模，在 GIFT-Eval 上零样本评估

## 实验结果

### 代码预训练：HumanEval pass@k

![代码预训练结果](https://arxiv.org/html/2606.12364v1/x1.png)
*图1：400M参数混合模型在代码预训练后的 HumanEval pass@k 表现。xLSTM[7:1] 在所有设置中领先*

xLSTM[7:1] 在每个 pass@k 和每种训练配置下均领先：
- pass@64 提升：20B 代码 token +1.43，100B +0.90，混合语料 +1.81

### 蒸馏实验：代码生成 pass@1

| 模型 | HumanEval | HumanEval+ | MBPP | MBPP+ | **平均** |
|------|-----------|-----------|------|-------|---------|
| Qwen3-4B（教师） | 0.914 | 0.835 | 0.708 | 0.847 | 0.826 |
| **xLSTM[1:0]** | **0.831** | **0.764** | **0.689** | 0.788 | **0.768** |
| Gated DeltaNet | 0.802 | 0.739 | 0.677 | **0.802** | 0.755 |
| Gated DeltaNet[-1,1] | 0.813 | 0.745 | 0.671 | 0.796 | 0.756 |

![蒸馏结果](https://arxiv.org/html/2606.12364v1/x2.png)
*图2：代码蒸馏实验中各架构的 pass@k 完整曲线*

### 时序基础模型预训练：GIFT-Eval

![时序预训练结果](https://arxiv.org/html/2606.12364v1/x3.png)
*图3：GIFT-Eval 上五种参数规模（1M-80M）的 MASE 和 CRPS，xLSTM 在 1M-40M 参数范围内持续领先*

xLSTM[3:1] 在 1M~40M 参数下在 MASE 和 CRPS 双指标上均领先，差距在 80M 处收窄。

### 合成任务：累积 vs. 状态追踪

![合成任务长度泛化](https://arxiv.org/html/2606.12364v1/x4.png)
*图4：累积（多数投票计数）与状态追踪（奇偶性）任务上的长度泛化（训练长度128，评估到2048）*

模型在训练长度128上训练，泛化到512和2048的对比：

| 模型 | AnBn @2048 | Majority @2048 | Parity @2048 | S3 @2048 |
|------|-----------|---------------|-------------|---------|
| Mamba-2 | 0.241 | - | 0.352 | - |
| Gated DeltaNet | - | 0.268 | ≤0.5 | ≤0.5 |
| Gated DeltaNet[-1,1] | - | - | 0.472 | 0.667 |
| xLSTM[1:0] | **0.892** | **0.763** | 0.0 | 0.0 |
| **xLSTM[1:1]** | 较低 | 较低 | **1.000** | **1.000** |

**关键发现：** xLSTM[1:1] 是唯一能在计数和状态追踪两类任务上同时长度泛化的配置。

![完整合成实验结果](https://arxiv.org/html/2606.12364v1/x5.png)
*图5：六种合成任务（AnBn、AnBnCn、多数投票计数、奇偶性、模运算、对称群S3）的完整长度泛化结果*

## 总结

本文系统揭示了三种主流亚二次方架构的核心能力差异：

**理论洞察：**
- xLSTM[1:0]（mLSTM）的**指数输入门**提供类 softmax 的时序累积能力
- xLSTM[0:1]（sLSTM）的**循环权重**提供非线性有限状态追踪能力
- Mamba-2 的**门耦合**设计限制了两种能力
- Gated DeltaNet 的**明确覆写机制**改善检索但牺牲累积

**实践建议：**
- 代码和时序等复杂结构化任务需要**同时兼具累积和状态追踪**
- xLSTM 混合架构（如 xLSTM[7:1] 或 xLSTM[3:1]）是当前最佳选择
- 标准英文 web 预训练无法区分架构差异，需在复杂任务上才能暴露优劣

**局限性：**
- 评估主要集中在 400M 以下规模，大规模（10B+）行为有待验证
- 代码和时序之外的复杂任务（如科学推理）有待探索
