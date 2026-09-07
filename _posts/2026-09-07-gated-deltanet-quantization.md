---
layout: post
title: "门控 DeltaNet 为何能经受 4-bit 量化：混合 27B LLM 循环半边的 NVFP4 W4A4"
date: 2026-09-07
categories: [论文解读, 模型量化]
tags: [量化, 混合LLM, 循环神经网络, DeltaNet, NVFP4, 推理加速]
---

> 📄 **论文**：Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM
> 🔗 **arXiv**：[2609.04098](https://arxiv.org/abs/2609.04098)
> 🏢 **机构**：Minima AI

## 一句话总结
通过对 Qwen3.8-27B 的门控 DeltaNet 层进行 NVFP4 W4A4 全量化，揭示了为何循环层比注意力层更易于量化，并提供了完整的量化配方和机理解释。

## 背景与问题
混合 LLM（Hybrid LLM）将 Softmax 注意力层与线性注意力层（如门控 DeltaNet，GDN）结合使用。Qwen3.8-27B 有 48 个 GDN 层和 16 个注意力层。

**现有量化实践的误区**：早期社区对 Qwen3.8-27B 进行 4-bit 量化时，通常将 GDN 块保持在 8-bit 或 16-bit 精度——尤其是其衰减门（decay gate）和写强度门（write-strength gate）——基于"循环状态中的误差会在长上下文中累积"的直觉。

**核心问题**：GDN 的门控机制在 4-bit 精度下真的不稳定吗？这一直觉是否正确？

## 核心方法
**Minima 量化方案**：对 Qwen3.8-27B 所有 496 个线性层应用 **NVFP4 W4A4 量化**，包括 GDN 层及其门控投影。

**四部分机理研究**，解释为何 GDN 实际上是量化友好的：

**机理 1：NVFP4 的 16 元素块缩放**
NVFP4 的分块缩放机制将残差流中的极端异常值局部化，均衡了不同层角色之间的激活误差分布。

**机理 2：门控投影最不敏感**
Softplus/指数和 Sigmoid 参数化将 ~11% 的 GEMM 误差压缩为 ~2% 的输出误差，这些非线性激活函数实际上起到了误差抑制的作用。

**机理 3：Delta 规则循环控制误差传播**
在注入噪声的实验中，32K token 内误差保持在平台水平，且状态冲击在数百步内被遗忘——因为每次写操作都沿当前 key 方向覆盖状态，循环并不会累积误差。

**机理 4：量化成本随上下文稀释**
每个 token 的量化成本随上下文增长而摊薄，而非复合增加。

**额外工程修复**：
- 修复了模块级校准的 NVFP4 checkpoint 被 fused GEMM 内核服务时的全局缩放不匹配问题
- 验证了校准 FP8 KV-cache 缩放的零性能损失方案

## 实验结果
在多个基准上，Minima 匹配 BF16 精度（5任务平均差距仅 -0.52）：

| 指标 | Minima (NVFP4 W4A4) | BF16 |
|------|---------------------|------|
| MMLU-Pro | ≈ BF16 | 基准 |
| GSM8K | ≈ BF16 | 基准 |
| AIME'25 | ≈ BF16 | 基准 |
| GPQA-Diamond | ≈ BF16 | 基准 |
| LiveCodeBench | ≈ BF16 | 基准 |
| RULER 检索 (64K) | ≈ BF16 | 基准 |
| 模型大小 | **17.5 GiB** | ~54 GiB |
| Prefill 速度 | **+14-19%** | 基准 |

checkpoint 已开源：https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4

## 总结
本文提供了"量化一切，随附 KV 缩放"的实用配方，并给出了"为何混合 LLM 的循环半边是量化简单半边"的机理解释。这一洞见挑战了业界对循环层量化脆弱性的普遍假设，对未来混合架构的高效部署具有直接指导意义。

局限性在于：研究聚焦于单一模型架构（Qwen3.8-27B），结论对其他循环架构的可推广性有待验证。
