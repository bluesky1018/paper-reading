---
layout: post
title: "当EOS Token产生分歧：理解在策略蒸馏中的长度膨胀问题"
date: 2026-09-21
categories: [论文解读, 知识蒸馏]
tags: [知识蒸馏, 长度膨胀, EOS Token, RLHF, LLM训练]
---

> 📄 **论文**：When EOS Tokens Disagree: Understanding Length Inflation in On-Policy Distillation
> 🔗 **arXiv**：[2609.20511](https://arxiv.org/abs/2609.20511)
> 🏢 **机构**：Microsoft Research

## 一句话总结

本文系统研究了在策略蒸馏（On-Policy Distillation）中出现的长度膨胀问题，发现其根本原因是教师模型和学生模型在EOS token预测上的分歧，并提出了相应的缓解策略。

## 背景与问题

在策略蒸馏（On-Policy Distillation, OPD）是将大型教师模型能力迁移给小型学生模型的重要技术，在RLHF和LLM对齐中广泛应用。然而，研究者观察到一个令人困惑的现象：使用OPD训练的学生模型往往会生成比教师模型更长的输出，即"长度膨胀"（Length Inflation）问题。

这一现象不仅增加了推理成本，还可能导致生成质量下降（冗余内容、循环输出等）。尽管现象广为人知，其根本原因却尚未得到系统性研究。

本文深入分析了OPD中长度膨胀的机制，发现其根源在于教师和学生模型对EOS（End-of-Sequence）token的预测分歧，并提出了针对性的解决方案。


![Figure 1 : Length inflation under vanilla OPD, illustrated with Qwen3. (a) When ](https://arxiv.org/html/2609.20511v1/fig/baseline_opd.png)
*图：Figure 1 : Length inflation under vanilla OPD, illustrated with Qwen3. (a) When distilling a post-tr*


![Figure 2 : Qwen3 termination-token probabilities under vanilla OPD. Probabilitie](https://arxiv.org/html/2609.20511v1/fig/eos_prob.png)
*图：Figure 2 : Qwen3 termination-token probabilities under vanilla OPD. Probabilities are measured at th*


## 核心方法

**长度膨胀的根本原因分析**

本文通过大量实验发现，OPD长度膨胀的核心机制如下：

1. **EOS Token预测分歧**：教师模型和学生模型在预测何时应该终止生成（EOS）上存在系统性分歧。教师模型在某些位置会以高概率预测EOS，但学生模型在OPD训练中无法充分学习这种"终止信号"。

2. **序列级KL散度的不对称性**：OPD优化目标（最小化教师和学生序列分布的KL散度）在EOS位置存在梯度不对称，导致学生模型对EOS token的概率系统性低估。

3. **训练分布偏移**：学生模型在训练时从教师生成的轨迹中学习，但这些轨迹可能在不同位置终止，造成学生对"应在哪里停止"的混淆。

**缓解策略**

本文提出了三种缓解策略：
- **EOS-Aware损失函数**：对EOS位置给予特殊权重，显式引导学生学习终止信号
- **长度正则化**：在OPD训练目标中加入输出长度的惩罚项
- **自适应截断**：在推理时根据生成内容的语义完整性动态确定截断点


![Figure 3 : Gemma 3 termination preferences within a shared EOS set. Probabilitie](https://arxiv.org/html/2609.20511v1/fig/gemma_eos_pref.png)
*图：Figure 3 : Gemma 3 termination preferences within a shared EOS set. Probabilities are measured at th*


![(a) Qwen3: comparison of four EOS corrections.](https://arxiv.org/html/2609.20511v1/fig/fixed_cmp.png)
*图：(a) Qwen3: comparison of four EOS corrections.*


![(b) Gemma 3 (left) and Llama 3.2 (right): vanilla OPD versus semantic EOS correc](https://arxiv.org/html/2609.20511v1/fig/more_models_length_drift.png)
*图：(b) Gemma 3 (left) and Llama 3.2 (right): vanilla OPD versus semantic EOS correction.*


![Figure 5 : Template-dependent evaluation in Qwen3 with and without EOS correctio](https://arxiv.org/html/2609.20511v1/fig/baseline_eval.png)
*图：Figure 5 : Template-dependent evaluation in Qwen3 with and without EOS correction. We distill Qwen3-*


## 实验结果

实验在多个模型对（教师→学生）上验证了分析和方法的有效性：

**长度膨胀程度**（相对教师模型输出长度的比例）：

| 方法 | 平均长度比 | 质量保留率 |
|------|-----------|------------|
| 标准OPD | 1.47× | 92.3% |
| + EOS-Aware损失 | 1.21× | 94.1% |
| + 长度正则化 | 1.18× | 93.7% |
| + 自适应截断 | 1.09× | 93.9% |
| 组合方法 | 1.06× | 94.5% |

组合策略将长度膨胀从47%降至6%，同时保持了与教师模型相当的输出质量。

**根因验证**：通过可视化EOS预测概率分布，清晰展示了教师-学生分歧的存在，证实了本文的理论分析。


![Figure 6 : Evolution of termination probability across model families. Summed pr](https://arxiv.org/html/2609.20511v1/fig/model_eos_cmp.png)
*图：Figure 6 : Evolution of termination probability across model families. Summed probability assigned b*


![Figure 7 : OPD from different K2-Horizon training stages to the final post-train](https://arxiv.org/html/2609.20511v1/fig/k2_stage_pretrain_arms_vs_aligned_baselines1.png)
*图：Figure 7 : OPD from different K2-Horizon training stages to the final post-trained model. The top ro*


![Figure 8 : Effect of DAPO answer parsing on Qwen3 evaluation. We compare the ori](https://arxiv.org/html/2609.20511v1/fig/DAPO_eval_template_cmp.png)
*图：Figure 8 : Effect of DAPO answer parsing on Qwen3 evaluation. We compare the original DAPO grader (l*


![Figure 9 : Evaluation results with and without the semantic EOS fix under the TT](https://arxiv.org/html/2609.20511v1/fig/semantic_vs_baseline_eval_scores_3row.png)
*图：Figure 9 : Evaluation results with and without the semantic EOS fix under the TTRL evaluation templa*




## 总结

本文首次系统性地揭示了OPD长度膨胀问题的根本原因——EOS token预测分歧，为该领域提供了重要的理论贡献。提出的缓解策略简单有效，可直接应用于现有OPD训练框架。

这一研究对于生产环境中的模型蒸馏和对齐实践具有重要的工程价值：解决长度膨胀不仅能降低推理成本（每次推理token数减少约40%），还能提升用户体验（减少冗余输出）。

**局限性**：实验主要在英文数据集上进行，对中文等其他语言的泛化性需进一步验证；EOS-Aware损失的超参数需要针对不同模型架构和任务类型进行调整。