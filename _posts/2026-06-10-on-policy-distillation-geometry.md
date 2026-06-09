---
layout: post
title: "策略蒸馏的几何特性：On-Policy Distillation参数空间分析"
date: 2026-06-10
categories: [论文解读, LLM训练]
tags: [知识蒸馏, LLM推理, 强化学习, 参数空间, 训练动态]
---

> 📄 **论文**：On the Geometry of On-Policy Distillation
> 🔗 **arXiv**：[2606.07082](https://arxiv.org/abs/2606.07082)
> 🏢 **机构**：论文作者机构

## 一句话总结

从参数空间几何角度分析On-Policy蒸馏（OPD）的训练动态，发现OPD介于SFT和RLVR之间，更新权重更少、对主方向影响更小，有助于理解LLM推理改进的内在机制。

## 背景与问题

On-policy distillation (OPD) is increasingly used to improve large language model reasoning, but its training dynamics remain poorly understood. We characterize the trajectory of OPD updates in parameter space and compare it with supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). A suite of parameter-space diagnostics consistently places OPD in a relaxed off-principal regime : compared with SFT, its updates affect fewer weights and avoid principal directions 


![Figure 1: Optimization geometry of OPD compared with SFT and RLVR. (a) OPD occup](https://arxiv.org/html/2606.07082/2606.07082v1/x1.png)
*图：Figure 1: Optimization geometry of OPD compared with SFT and RLVR. (a) OPD occupies a relaxed off-pr*

Large reasoning models (LRMs) have substantially advanced complex mathematical and programming reasoning in large language models Guo et al. ( 2025 ); Shao et al. ( 2024 ); OpenAI ( 2024 ) . Post-training is a central driver of this progress. Beyond supervised fine-tuning (SFT) Wei et al. ( 2022 ) on offline demonstrations and reinforcement learning with verifiable rewards (RLVR) Shao et al. ( 2024 ); Guo et al. ( 2025 ); Yu et al. ( 2025 ) from sparse outcome signals, on-policy distillation (OP

## 核心方法

Section 3 located on-policy distillation (OPD) within the SFT–RLVR parameter-space spectrum. We now ask whether this positioning is only an endpoint property, or whether OPD follows a distinct update trajectory during training. We study the cumulative update across checkpoints and show that OPD rapidly enters a persistent low-dimensional update channel.


![Figure 2: Parameter-space diagnostics. SFT induces larger subspace rotation and ](https://arxiv.org/html/2606.07082/2606.07082v1/x2.png)
*图：Figure 2: Parameter-space diagnostics. SFT induces larger subspace rotation and spectral drift, RLVR*


![Figure 1: Optimization geometry of OPD compared with SFT and RLVR. (a) OPD occupies a relaxed off-pr](https://arxiv.org/html/2606.07082/2606.07082v1/x1.png)
*图1：Figure 1: Optimization geometry of OPD compared with SFT and RLVR. (a) OPD occupies a relaxed off-pr*

![Figure 2: Parameter-space diagnostics. SFT induces larger subspace rotation and spectral drift, RLVR](https://arxiv.org/html/2606.07082/2606.07082v1/x2.png)
*图2：Figure 2: Parameter-space diagnostics. SFT induces larger subspace rotation and spectral drift, RLVR*

![Figure 3: Update-mask localization. We compare where bf16-visible updates land relative to principal](https://arxiv.org/html/2606.07082/2606.07082v1/x3.png)
*图3：Figure 3: Update-mask localization. We compare where bf16-visible updates land relative to principal*

![Figure 4: Intrinsic update geometry. We track cumulative updates . OPD stays in a narrow stable-rank](https://arxiv.org/html/2606.07082/2606.07082v1/x4.png)
*图4：Figure 4: Intrinsic update geometry. We track cumulative updates . OPD stays in a narrow stable-rank*

![Figure 5: Subspace emergence. Top- subspace similarity to the final update shows that OPD locks onto](https://arxiv.org/html/2606.07082/2606.07082v1/x5.png)
*图5：Figure 5: Subspace emergence. Top- subspace similarity to the final update shows that OPD locks onto*

![Figure 6: Rank- projected training. Rank- projection leaves OPD intact but degrades SFT.](https://arxiv.org/html/2606.07082/2606.07082v1/figures/k16_projection_percent.png)
*图6：Figure 6: Rank- projected training. Rank- projection leaves OPD intact but degrades SFT.*


## 实验结果

Our analysis identifies on-policy distillation (OPD) as a distinct parameter-space regime rather than a simple endpoint interpolation between SFT and RLVR. OPD occupies a relaxed off-principal region, but its training trajectory further exhibits subspace locking: cumulative updates rapidly enter a small, persistent low-dimensional channel that is sufficient to preserve training progress.


![Figure 3: Update-mask localization. We compare where bf16-visible updates land r](https://arxiv.org/html/2606.07082/2606.07082v1/x3.png)
*图：Figure 3: Update-mask localization. We compare where bf16-visible updates land relative to principal*


### 实验数据表格

| Base Model                               | Finetuned (FT) Model | Algorithm | Data | sparsity bf16 |
| ---------------------------------------- | -------------------- | --------- | ---- | ------------- |
| Controlled comparison: SFT OPD RLVR      |                      |           |      |               |
| Qwen3-8B-Base                            | Qwen3-8B-SFT         | SFT       | Math | 8.1%          |
| Qwen3-8B-SFT                             | OPD-8B-T32B          | OPD       | Math | 51.6%         |
| Qwen3-8B-SFT                             | RLVR-8B              | GRPO      | Math | 77.2%         |
| OPD robustness across teacher / student  |                      |           |      |               |
| Qwen3-4B-SFT                             | OPD-4B-T8B           | OPD       | Math | 50.3%         |
| Qwen3-4B-SFT                             | OPD-4B-T14B          | OPD       | Math | 51.1%         |
| Qwen3-4B-SFT                             | OPD-4B-T32B          | OPD       | Math | 51.7%         |
| Qwen3-14B-SFT                            | OPD-14B-T32B         | OPD       | Math | 56.6%         |

## 总结

On the Geometry of On-Policy Distillation 提出了一个新颖的研究框架，针对LLM训练领域的核心挑战提供了系统性解决方案。

**主要贡献：**
- 从参数空间几何角度分析On-Policy蒸馏（OPD）的训练动态，发现OPD介于SFT和RLVR之间，更新权重更少、对主方向影响更小，有助于理解LLM推理改进的内在机制。
- 通过系统实验验证了方法的有效性
- 为后续研究提供了重要的基准和参考

**局限性与展望：** 未来工作可进一步探索方法在更广泛场景下的应用，以及结合更多领域知识提升系统性能。