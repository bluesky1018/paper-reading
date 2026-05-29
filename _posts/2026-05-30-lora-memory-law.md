---
layout: post
title: "LoRA 如何记忆？大模型微调的参数记忆定律"
date: 2026-05-30
categories: [论文解读, LLM微调]
tags: ["LoRA", "LLM", "微调", "知识更新", "持续学习"]
---

> 📄 **论文**：How LoRA Remembers? A Parametric Memory Law for LLM Finetuning
> 🔗 **arXiv**：[2605.30260](https://arxiv.org/abs/2605.30260)
> 🏢 **机构**：Zhejiang University, Alibaba Group

## 一句话总结

Large Language Models (LLMs) must continuously learn and update knowledge to remain effective in dynamic real-world environments. While Low-Rank Adaptation (LoRA) is widely used for such memory update...

## 背景与问题

Large Language Models (LLMs) have shown strong capabilities across diverse tasks and are now widely used in real-world systems Zhao et al. ( 2023 ) . However, their knowledge is encoded in static pretrained parameters and remains largely fixed after deployment. In practice, models continuously encounter new information such as updated facts, user preferences, and task-specific knowledge Yao et al. ( 2023 ) . Efficiently integrating such information therefore becomes an key problem in continual learning and memory systems.

Non-parametric methods address this challenge by providing external context during inference. Specifically, approaches such as in-context learning (ICL) Dong et al. ( 2024 ) , retrieval-augmented generation (RAG) Gao et al. ( 2023 ) , and external non-parametric memory systems He et al. ( 2024 ); Fang et al. ( 2025 ) dynamically integrate information without modifying model parameters. However, these methods are fundamentally constrained by fixed context windows, attention dilution, and escalating computational overhead as the input length scales Liu et al. ( 2024 ) .

In contrast, parametric memory embeds information directly into parameters or modular structures, enabling permanent knowledge consolidation and retrieval-free internal reasoning Yang et al. ( 2024 ); Li et al. ( 2025 ); Lei et al. ( 2026 ) . Recent works have further conceptualizes Low-Rank Adaptation (LoRA) as a specialized knowledge memory unit Back et al. ( 2026 ) . However, existing eval


![Figure 1: LoRA as a pluggable memory unit in the LLM’s latent space. The LoRA module (rank r r ) enc](https://arxiv.org/html/2605.30260/2605.30260v1/figures/motivation.png)
*图：Figure 1: LoRA as a pluggable memory unit in the LLM’s latent space. The LoRA module (rank r r ) enc*


![(a) Approximate log-linear dependence of Δ ​ ℒ \Delta\mathcal{L} on rank r r and length ℓ \ell](https://arxiv.org/html/2605.30260/2605.30260v1/x1.png)
*图：(a) Approximate log-linear dependence of Δ ​ ℒ \Delta\mathcal{L} on rank r r and length ℓ \ell*


Inspired by Jelassi et al. ( 2024 ); Back et al. ( 2026 ) , we formulate exact parametric memorization over a dataset 𝒟 = { ( 𝐪 ( i ) , 𝐚 ( i ) ) } i = 1 N \mathcal{D}=\{(\mathbf{q}^{(i)},\mathbf{a}^{(i)})\}_{i=1}^{N} , where 𝐪 ( i ) \mathbf{q}^{(i)} serves as a unique key and 𝐚 ( i ) = ( a 1 ( i ) , … , a ℓ i ( i ) ) \mathbf{a}^{(i)}=(a^{(i)}_{1},\ldots,a^{(i)}_{\ell_{i}}) is the target content. Given a frozen base model f θ 0 f_{\theta_{0}} , we learn a parameter increment Δ ​ θ \Delta\theta to construct an updated model f θ f_{\theta} with parameters θ = ( θ 0 , Δ ​ θ ) \theta=(\theta_{0},\Delta\theta) , satisfying:

Since 𝐚 ( i ) \mathbf{a}^{(i)} is inaccessible during inference except via the query 𝐪 ( i ) \mathbf{q}^{(i)} , Δ ​ θ \Delta\theta constitutes the exclusive medium for info

## 核心方法

Standard SFT minimizes the token-averaged cross-entropy, allocating equal gradient budget to all tokens regardless of their learning status. As established in Section 4.3 , tokens with loss ℒ < ℒ crit \mathcal{L}<\mathcal{L}_{\text{crit}} are already in the ordered phase and effectively memorized. Continuing to optimize these tokens dilutes the signal for stubborn tokens (those in the uncertain regime), which are critical for preventing autoregressive error propagation.

To address this, we propose Memorization-oriented Fine-Tuning (MemFT) , which replaces the uniform objective with a token-weighted form:

where ℳ \mathcal{M} is the set of target token indices t t in the sequence, ℒ t ​ ( θ ) \mathcal{L}_{t}(\theta) is the cross-entropy loss at position t t , and w t w_{t} is a dynamic weight. Normalizing by the sum of weights ensures stable gradient scales across samples with varying numbers of active tokens. Different instantiations of MemFT differ only in the construction of w t w_{t} .

The baseline uses the critical loss as a hard mask:


![(b) Predicted vs. true Δ ​ ℒ \Delta\mathcal{L}](https://arxiv.org/html/2605.30260/2605.30260v1/x4.png)
*图：(b) Predicted vs. true Δ ​ ℒ \Delta\mathcal{L}*


![(c) Decoupling of low loss and high accuracy](https://arxiv.org/html/2605.30260/2605.30260v1/x5.png)
*图：(c) Decoupling of low loss and high accuracy*


![(a) Probability dynamics across ranks](https://arxiv.org/html/2605.30260/2605.30260v1/x7.png)
*图：(a) Probability dynamics across ranks*


![(b) Lower bound on first failure](https://arxiv.org/html/2605.30260/2605.30260v1/x8.png)
*图：(b) Lower bound on first failure*


![(c) Localization of failure positions](https://arxiv.org/html/2605.30260/2605.30260v1/x9.png)
*图：(c) Localization of failure positions*


![Figure 4: Training convergence of Qwen3-8B on the Random / Long-Context Memorization Stress Test. Ea](https://arxiv.org/html/2605.30260/2605.30260v1/x10.png)
*图：Figure 4: Training convergence of Qwen3-8B on the Random / Long-Context Memorization Stress Test. Ea*


## 实验结果

论文在多个基准测试上进行了全面评估，验证了所提方法的有效性。


![Figure 5: Training convergence of Llama3.1-8B on the Random / Long-Context Memorization Stress Test.](https://arxiv.org/html/2605.30260/2605.30260v1/x11.png)
*图：Figure 5: Training convergence of Llama3.1-8B on the Random / Long-Context Memorization Stress Test.*


![Figure 6: Training convergence of Llama3.1-8B on the PhoneBook benchmark. Each subplot corresponds t](https://arxiv.org/html/2605.30260/2605.30260v1/x12.png)
*图：Figure 6: Training convergence of Llama3.1-8B on the PhoneBook benchmark. Each subplot corresponds t*


![Figure 7: Exact-match accuracy of Qwen3-8B on the Random / Long-Context Memorization Stress Test. Ea](https://arxiv.org/html/2605.30260/2605.30260v1/x13.png)
*图：Figure 7: Exact-match accuracy of Qwen3-8B on the Random / Long-Context Memorization Stress Test. Ea*


![Figure 8: Exact-match accuracy of Llama3.1-8B on the Random / Long-Context Memorization Stress Test.](https://arxiv.org/html/2605.30260/2605.30260v1/x14.png)
*图：Figure 8: Exact-match accuracy of Llama3.1-8B on the Random / Long-Context Memorization Stress Test.*


![Figure 9: Exact-match accuracy on the PhoneBook benchmark for Qwen3-8B and Llama3.1-8B. The upper pa](https://arxiv.org/html/2605.30260/2605.30260v1/x15.png)
*图：Figure 9: Exact-match accuracy on the PhoneBook benchmark for Qwen3-8B and Llama3.1-8B. The upper pa*


![Figure 10: Per-position probability grid for the Long-Context Memoriza tion Stress Test Random 100% ](https://arxiv.org/html/2605.30260/2605.30260v1/figures/stubborn_grid_qwen_random_aligned.png)
*图：Figure 10: Per-position probability grid for the Long-Context Memoriza tion Stress Test Random 100% *


## 全文图示


![Figure 11: Per-position probability grid for the Random (100%) scenario with r ∈ { 48 , 64 , 128 , 2](https://arxiv.org/html/2605.30260/2605.30260v1/figures/stubborn_grid_qwen_random.png)
*图：Figure 11: Per-position probability grid for the Random (100%) scenario with r ∈ { 48 , 64 , 128 , 2*


![Figure 12: Per-position probability grid for the Long-Context Memoriza tion Stress Test Random 20% s](https://arxiv.org/html/2605.30260/2605.30260v1/figures/stubborn_grid_qwen_longbench0random20.png)
*图：Figure 12: Per-position probability grid for the Long-Context Memoriza tion Stress Test Random 20% s*


![Figure 13: Per-position probability grid for the Long-Context Memoriza tion Stress Test Random 60% s](https://arxiv.org/html/2605.30260/2605.30260v1/figures/stubborn_grid_qwen_longbench0random60.png)
*图：Figure 13: Per-position probability grid for the Long-Context Memoriza tion Stress Test Random 60% s*


## 总结

By leveraging LoRA as a controllable probe into the memory mechanisms within the latent space of LLMs, we uncover the Parametric Memory Law . This law characterizes loss reduction as a power-law function of both LoRA rank and sequence length. We further reveal a deterministic phase transition in token-level loss dynamics, where unresolved bottleneck tokens can trigger decoding collapse. Guided by this mechanistic understanding, we propose MemFT , a fine-tuning strategy designed to explicitly resolve these critical memory bottlenecks.

