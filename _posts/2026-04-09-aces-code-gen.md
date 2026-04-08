---
layout: post
title: "ACES：谁来测试那些测试？代码生成中的留一法AUC一致性"
date: 2026-04-09
categories: [论文解读, 代码生成]
tags: [代码生成, LLM, 测试用例评估, AUC, 留一法, 代码选择]
---

> 📄 **论文**：ACES: Who Tests the Tests? Leave-One-Out AUC Consistency for Code Generation
> 🔗 **arXiv**：[2604.03922](https://arxiv.org/abs/2604.03922)
> 🏢 **机构**：南京大学人工智能学院 & 软件新技术国家重点实验室

## 一句话总结

本文提出 ACES 框架，通过留一法 AUC（LOO-AUC）一致性评分来估计 LLM 生成测试用例的质量权重，从理论上打破了"用测试评估代码、用代码评估测试"的循环依赖，显著提升了代码生成候选解的选择效果。

## 背景与问题

在大语言模型（LLM）的代码生成任务中，一个关键挑战是：从模型生成的众多候选代码中，如何选出正确的那一个？目前主流做法是同时让 LLM 生成测试用例，然后通过执行这些测试来对候选代码进行排序。然而，这里存在一个根本性的循环依赖：我们需要可靠的测试来判断代码是否正确，同时又需要可靠的代码来判断测试是否有效。

现有方法如 CodeT 以"共识集"大小来衡量测试质量，MBR-exec 和 SRank 则依赖超出二元 pass/fail 矩阵的成对输出比较，这些方法要么过于简单，要么依赖额外信息。更本质的问题在于，LLM 生成的测试用例质量参差不齐：有的测试能有效区分正确代码和错误代码（信息性测试），有的测试几乎没有区分力（无信息测试），甚至有的测试对正确代码和错误代码的通过率相近甚至反转（误导性测试）。如果对所有测试等权处理，误导性测试将直接损害选择质量。

本文的核心贡献是提出了一个基于信息论的框架，将"测试的区分能力"（discriminative power）形式化为可观测量，并证明留一法 AUC 一致性与测试的潜在区分力在期望意义下成正比。由此设计出两种实用算法 ACES-C（闭合形式加权）和 ACES-O（优化加权），无需任何额外监督信号，仅凭执行结果矩阵即可对测试用例进行质量评估和加权排序。

## 核心方法

### 问题建模

给定 $n$ 个候选代码和 $m$ 个测试用例，执行后得到二元 pass/fail 矩阵 $\mathbf{B} \in \{0,1\}^{n \times m}$，其中 $B_{ij} = 1$ 表示第 $i$ 个代码通过了第 $j$ 个测试。候选代码的最终得分由加权求和给出：

$$s_i(\mathbf{w}) = \sum_j w_j B_{ij}$$

目标是找到最优权重向量 $\mathbf{w}$，使得排名第一的代码尽可能正确。

### 测试的区分能力

对于每个测试 $t_j$，定义：
- $\alpha_j$：正确代码通过 $t_j$ 的概率（class-conditional pass rate for correct codes）
- $\beta_j$：错误代码通过 $t_j$ 的概率（class-conditional pass rate for incorrect codes）
- **区分力** $\delta_j = \alpha_j - \beta_j$

当 $\delta_j > 0$ 时为信息性测试，$\delta_j = 0$ 为无信息测试，$\delta_j < 0$ 为误导性测试。

![测试用例分类图](https://arxiv.org/html/2604.03922/x1.png)
*图1：按正确性（$\alpha_j$）和区分力（$\delta_j$）对测试用例进行分类的分类图*

### 理论基础：Pass@k 下界

**定理2（Pass@k 下界）**：在一定条件下，

$$\text{Pass@k} \geq 1 - \frac{n^-}{k} \exp\left(-\frac{R(\mathbf{w})}{2}\right)$$

其中 $n^-$ 是错误代码数，$R(\mathbf{w}) = M(\mathbf{w})^2 / \sum_j w_j^2$，$M(\mathbf{w}) = \sum_j w_j \delta_j$ 表示加权区分力之和。这个下界说明：**若能最大化 $R(\mathbf{w})$，则 Pass@1 性能也随之提升**。

### 核心定理：LOO-AUC 恒等式

**定理3（LOO-AUC 恒等式）**：

$$\mathbb{E}[\text{LOO-AUC}_j(\mathbf{w})] - \frac{1}{2} = c_j(\mathbf{w}) \cdot \delta_j$$

其中 $c_j(\mathbf{w}) > 0$ 是一个依赖当前权重的正比例系数。这个定理建立了**可观测量**（LOO-AUC 一致性）与**潜在量**（测试的区分力 $\delta_j$）之间的直接联系，从而打破了循环依赖。

LOO-AUC 的计算方式为：对于测试 $t_j$，将其从加权评分中去掉，计算剩余测试的得分排序与 $t_j$ 通过情况之间的 AUC 值。

### ACES-C：闭合形式加权

基于均匀初始权重 $\mathbf{w}^{\text{unif}}$，ACES-C 计算：

$$w_j^{\text{ACES-C}} = \max\left(0,\ \text{LOO-AUC}_j(\mathbf{w}^{\text{unif}}) - \frac{1}{2}\right) \cdot p_j(1 - p_j)$$

其中 $p_j$ 是测试 $t_j$ 的整体通过率，起到方差归一化的作用。ACES-C 的理论保证（定理6）：在假设平均区分力 $\bar{\delta} > 2\sqrt{\ln 2 / m}$ 成立时，$\mathbb{E}[q_j] \propto \delta_j$，即 ACES-C 权重在期望意义下能恢复真实区分力。

### ACES-O：优化加权

ACES-O 不依赖上述假设，而是通过迭代优化最大化目标函数：

$$J(\mathbf{w}) = \sum_j w_j \cdot \left(\text{LOO-AUC}_j(\mathbf{w}) - \frac{1}{2}\right)$$

为使目标可微，将二元 AUC 替换为 logistic 代理函数，并在概率单纯形约束下进行梯度上升优化。ACES-O 以 ACES-C 的结果作为初始化，通常只需少量迭代即可收敛。

### 说明性示例

![说明性示例](https://arxiv.org/html/2604.03922/x2.png)
*图2：8个候选代码与10个测试用例的 pass 矩阵示例（Easy 和 Hard 两种场景），直观展示 ACES 如何通过 LOO-AUC 权重识别有效测试*

## 实验结果

### 实验设置

- **评测基准**：HumanEval（164道题）、HumanEval+（164道题，更严格的测试集）、MBPP（427道题）
- **候选生成**：GPT-3.5-Turbo，每题约200个候选代码和500个测试用例（来自 Huang et al., 2024 的数据集）
- **评估指标**：Pass@1（%）

### 主要结果

| 方法 | HumanEval | HumanEval+ | MBPP |
|---|---|---|---|
| GPT-3.5-Turbo（直接推理） | 68.38 | 58.75 | 66.80 |
| GPT-4（直接推理） | 83.54 | 75.00 | - |
| DeepSeek-Coder-33B | 76.22 | 70.73 | 73.28 |
| WizardCoder-33B | 77.44 | 69.51 | 73.04 |
| CodeLlama-34B | 65.24 | 58.54 | 63.78 |
| Majority Voting | 80.49 | 69.51 | 68.62 |
| CodeT | 80.49 | 70.12 | 70.13 |
| MBR-exec | 82.93 | 72.56 | 71.49 |
| SC+Spec | 81.10 | 71.34 | 72.84 |
| MPSC | 79.27 | 70.73 | 71.96 |
| DS³ | 81.71 | 72.56 | **75.88** |
| **ACES-C** | 82.93 | 71.34 | 71.19 |
| **ACES-O** | **84.15** | **74.39** | 72.37 |
| ACES-C + DS³ | **85.37** | **77.44** | 76.11 |

**关键发现**：
- ACES-O 在所有三个基准上均超越了全部仅使用执行结果的方法
- 在更严格的 HumanEval+ 上，ACES-O（74.39%）甚至超越了使用额外信号的 DS³（72.56%）
- 相比 Majority Voting，ACES-O 的提升在 HumanEval+（+4.88%）上比 HumanEval（+3.66%）更大，说明 ACES 在测试质量更难保证时优势更明显
- ACES-C 与 DS³ 结合后，在 HumanEval 和 HumanEval+ 上分别达到 85.37% 和 77.44%，创下新的 state-of-the-art

### 理论假设验证

![假设4验证](https://arxiv.org/html/2604.03922/x3.png)
*图3：在 MBPP 数据集上验证假设4（平均区分力 $\bar{\delta} > 0$）的满足情况，绝大多数任务均满足该假设*

### 测试质量对 Pass@1 的影响分析

![区分力分箱分析](https://arxiv.org/html/2604.03922/x4.png)
*图4：按 $\delta_j$ 分箱后，各区间内测试用例对 Pass@1 和 AUC 的影响分析。误导性测试（$\delta_j < 0$）区间对 Majority Voting 造成最大伤害，而 ACES-O 对误导性测试的敏感度降低了 46%*

具体数据：在最有害的误导性测试区间，Majority Voting 的 Pass@1 下降 0.056，而 ACES-O 仅下降 0.030，降低了 46% 的敏感度。

### ACES-C 权重与真实区分力的对比

![测试质量检测](https://arxiv.org/html/2604.03922/x5.png)
*图5：ACES-C 权重与真实区分力 $\delta_j$ 的对比散点图，展示 ACES-C 能正确识别至少 94.8% 的信息性测试*

ACES-C 权重在三个基准上至少能正确识别 94.8% 的信息性测试（$\delta_j > 0$），验证了理论分析的正确性。

## 总结

本文从理论角度彻底分析了"用测试选择代码"任务中的核心难题，提出了 ACES 框架，其主要贡献包括：

1. **理论贡献**：证明了 LOO-AUC 一致性与测试区分力之间的恒等关系（定理3），以及加权区分力与 Pass@k 下界的关联（定理2），为测试加权提供了严格的理论依据，打破了循环依赖。

2. **算法贡献**：提出了两种实用算法——ACES-C（单步闭合形式，计算高效）和 ACES-O（迭代优化，无需均质假设），均仅需二元 pass 矩阵即可运行，无需额外信号。

3. **实验贡献**：在 HumanEval、HumanEval+、MBPP 三个基准上均取得了仅使用执行结果的方法中的最佳成绩，且与 DS³ 等使用额外信号的方法结合后达到新的 state-of-the-art。

**局限性**方面，ACES 依赖 LLM 同时生成足够数量的候选代码和测试用例，当候选数量或测试数量较少时，LOO-AUC 估计可能不够准确。此外，ACES-O 的优化过程虽然通常收敛较快，但在极端情况下可能存在局部最优的风险。未来工作可以探索将 ACES 与语义感知的测试生成方法结合，或将其扩展到更复杂的程序合成任务场景。
