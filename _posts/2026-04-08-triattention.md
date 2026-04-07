---
layout: post
title: "TriAttention：基于三角函数KV压缩的高效长推理"
date: 2026-04-08
categories: [论文解读, 高效推理]
tags: [Attention, KV压缩, 长上下文, LLM, NVIDIA]
---

> 📄 **论文**：TriAttention: Efficient Long Reasoning with Trigonometric KV Compression
> 🔗 **arXiv**：[2604.04921](https://arxiv.org/abs/2604.04921)
> 🏢 **机构**：NVIDIA
> 👥 **作者**：Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen

## 一句话总结
提出TriAttention，利用三角函数原理对KV缓存进行高效压缩，在保持推理质量的同时显著降低长序列推理的计算开销

## 背景与问题

Extended reasoning in large language models (LLMs) creates severe KV cache memory bottlenecks. Leading KV cache compression methods estimate KV importance using attention scores from recent post-RoPE queries. However, queries rotate with position during RoPE, making representative queries very few, leading to poor top-key selection and unstable reasoning. To avoid this issue, we turn to the pre-RoPE space, where we observe that Q and K vectors are highly concentrated around fixed non-zero centers and remain stable across positions -- Q/K concentration. We show that this concentration causes queries to preferentially attend to keys at specific distances (e.g., nearest keys), with the centers determining which distances are preferred via a trigonometric series. Based on this, we propose TriAttention to estimate key importance by leveraging these centers. Via the trigonometric series, we use the distance preference characterized by these centers to score keys according to their positions, and also leverage Q/K norms as an additional signal for importance estimation. On AIME25 with 32K-token generation, TriAttention matches Full Attention reasoning accuracy while achieving 2.5x higher throughput or 10.7x KV memory reduction, whereas leading baselines achieve only about half the accuracy at the same efficiency. TriAttention enables OpenClaw deployment on a single consumer GPU, where long context would otherwise cause out-of-memory with Full Attention.



## 核心方法

详见原文方法章节。


![Figure 1: Performance trade-offs on AIME25 (Qwen3-8B). (A) At equivalent accurac](https://arxiv.org/html/2604.04921/2604.04921v1/fig/fig_kv_budget_throughput_accuracy_memratio.png)
*图：Figure 1: Performance trade-offs on AIME25 (Qwen3-8B). (A) At equivalent accuracy (40.8%), TriAttention achieves 2.5×\times higher throughput than Ful*


![Figure 2: Q/K concentration and its implications for attention. (A) Pre-RoPE Q/K](https://arxiv.org/html/2604.04921/2604.04921v1/fig/fig_intro_combined_v2.png)
*图：Figure 2: Q/K concentration and its implications for attention. (A) Pre-RoPE Q/K vectors at the dominant frequency band are highly concentrated (high *


![Figure 3: Attention reconstruction correlation across three DeepSeek-R1 distille](https://arxiv.org/html/2604.04921/2604.04921v1/fig/fig_freq_reconstruction_multimodel_row1_only.png)
*图：Figure 3: Attention reconstruction correlation across three DeepSeek-R1 distilled LLMs, including Qwen3 (Qwen Team, 2025), Qwen2.5 (Qwen Team, 2024), *


![Figure 4: Method overview. From left to right: offline calibration computes Q di](https://arxiv.org/html/2604.04921/2604.04921v1/x1.png)
*图：Figure 4: Method overview. From left to right: offline calibration computes Q distribution centers; then during inference, original attention is score*


## 实验结果

详见原文实验章节。


![Figure 5: Performance comparison on Qwen3-8B. (A–C) Accuracy vs. KV cache budget](https://arxiv.org/html/2604.04921/2604.04921v1/fig/fig_kv_budget_accuracy_combined_4panel.png)
*图：Figure 5: Performance comparison on Qwen3-8B. (A–C) Accuracy vs. KV cache budget on three mathematical reasoning benchmarks. TriAttention consistently*


![Figure A: Evaluating memory via recursive simulation. Left: With complete memory](https://arxiv.org/html/2604.04921/2604.04921v1/fig/RECURSIVE_SIMULATION.jpg)
*图：Figure A: Evaluating memory via recursive simulation. Left: With complete memory, all intermediate states are retained and correct values propagate up*


![Figure B: Method visualization with real attention maps, corresponding to the sc](https://arxiv.org/html/2604.04921/2604.04921v1/x2.png)
*图：Figure B: Method visualization with real attention maps, corresponding to the schematic in Figure 4. Top row: The four stages of TriAttention. (1) We *


![Figure C: OpenClaw demo on a single RTX 4090 with Qwen3-32B (INT4). Full attenti](https://arxiv.org/html/2604.04921/2604.04921v1/fig/fig_openclaw_demo.png)
*图：Figure C: OpenClaw demo on a single RTX 4090 with Qwen3-32B (INT4). Full attention runs out of memory during multi-turn interaction, while TriAttentio*


## 总结

本文提出了 **TriAttention**，提出TriAttention，利用三角函数原理对KV缓存进行高效压缩，在保持推理质量的同时显著降低长序列推理的计算开销。该工作从理论和实践层面均有创新，为后续研究提供了重要参考。

**局限性与未来方向：** 如所有工作一样，该研究仍有一定局限性，后续可在更大规模数据集、更多样化场景下进行验证和拓展。
