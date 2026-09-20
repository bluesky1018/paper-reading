---
layout: post
title: "DeepSeek-V4.1-Flash：突破KV Cache压缩极限的万亿级多模态MoE模型"
date: 2026-09-21
categories: [论文解读, 大语言模型]
tags: [DeepSeek, KV Cache, MoE, Transformer, 长文本]
---

> 📄 **论文**：DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression
> 🔗 **arXiv**：[2609.19969](https://arxiv.org/abs/2609.19969)
> 🏢 **机构**：DeepSeek-AI

## 一句话总结

DeepSeek-V4.1-Flash通过创新的因果编解码器架构和CSA2注意力机制，将全局KV Cache压缩至仅890字节/token，相比前代减少约75%，同时性能全面超越DeepSeek-V4-Flash。

## 背景与问题

随着长程智能体（Long-horizon Agents）的广泛应用，模型的工作负载变得日益以"输入"为主导——每次推理需要处理大量历史上下文。这种趋势使得KV Cache的管理成为模型部署的核心挑战：全局KV Cache占据大量高带宽存储器（HBM），持久化KV Cache则需要消耗大量SSD和主机内存，而KV数据的传输也受带宽限制制约了服务吞吐量。

DeepSeek此前的工作（DeepSeek-V4-Flash）已经通过稀疏注意力机制显著降低了长序列处理的计算成本，但存储和通信瓶颈仍然突出。如何在不牺牲模型能力的前提下，将KV Cache的存储开销压缩到一个新量级，成为了核心研究问题。

DeepSeek-V4.1-Flash在这一背景下应运而生，它是一个具有552B骨干参数的多模态MoE模型，支持高达100万token的上下文长度。


![Figure 1: (a) Performance of DeepSeek-V4.1-Flash and its counterparts on agentic](https://arxiv.org/html/2609.19969v1/figures/teaser_a.png)
*图：Figure 1: (a) Performance of DeepSeek-V4.1-Flash and its counterparts on agentic benchmarks. (b) Glo*


![Figure 2: Single-token Decode FLOPs versus context length across generations of ](https://arxiv.org/html/2609.19969v1/figures/decode_flops_curves.png)
*图：Figure 2: Single-token Decode FLOPs versus context length across generations of DeepSeek models. We *


## 核心方法

**因果编解码器（Causal Encoder-Decoder, CED）架构**

CED是本文最核心的架构创新。传统Decoder-Only模型在prefill和decode阶段使用相同数量的参数，而CED架构将模型分为编码器和解码器两部分：
- **Prefill阶段**：仅激活8B参数（编码器），效率极高
- **Decode阶段**：激活16B参数（解码器），保证生成质量
- 解码器的全局KV Cache从编码器最后隐藏状态投影得到，无需在prefill阶段维护完整KV

**压缩稀疏注意力2（CSA2）**

在DeepSeek-V4的CSA基础上，CSA2引入了跨层KV Cache复用机制，多个相邻层共享同一组KV，大幅减少需要存储的KV Cache总量。同时结合FP4量化（4位浮点数）来进一步压缩每个token的存储需求。

**SWA有界回放（SWA Bounded Replay）**

这是针对持久化KV Cache的专项优化。通过精心设计的缓存替换策略，确保SSD上存储的持久化KV Cache保持在合理范围内，相比DeepSeek-V4-Flash减少约7/8的持久化存储需求。


![Figure 6: Bits-per-bytes (BPB) comparison of DeepSeek-V4-Flash-Base, DeepSeek-V4](https://arxiv.org/html/2609.19969v1/figures/pretrain_inhouse_ppl.png)
*图：Figure 6: Bits-per-bytes (BPB) comparison of DeepSeek-V4-Flash-Base, DeepSeek-V4-Pro-Base and DeepSe*


## 实验结果

DeepSeek-V4.1-Flash在多个关键指标上展现了突破性进展：

| 指标 | DeepSeek-V4-Flash | DeepSeek-V4.1-Flash | 改进幅度 |
|------|-------------------|---------------------|----------|
| 全局KV Cache（字节/token） | ~3560 | 890 | 降低75% |
| 持久化KV Cache | 基准 | 基准×1/8 | 降低87.5% |
| Prefill激活参数 | 16B | 8B | 减少50% |
| 支持上下文长度 | — | 100万token | — |

在性能评测方面，DeepSeek-V4.1-Flash在多个Agentic基准（包括代码生成、长文本理解、多模态推理等）上全面超越DeepSeek-V4-Flash，证明了激进的KV Cache压缩并不以牺牲模型能力为代价。

预训练数据规模达到45T token的多模态语料库，为模型提供了丰富的多模态知识基础。




![2609.19969附图](https://arxiv.org/html/2609.19969v1/figures/teaser_b.png)
*图：2609.19969 附图*


## 总结

DeepSeek-V4.1-Flash代表了大规模MoE语言模型在部署效率上的重大突破。通过CED架构、CSA2机制和FP4 KV缓存的组合创新，该模型将全局KV Cache压缩至每token仅890字节，相比第一代DeepSeek模型减少了437倍，为超长上下文智能体的经济化部署开辟了新路径。

该工作的意义不仅在于技术指标的提升，更在于它表明：通过合理的架构设计，可以同时实现存储压缩和性能提升，打破了传统认知中压缩必然损失精度的悖论。

**局限性**：CED架构的prefill-decode参数分离可能在某些需要深度prefill推理的任务上存在信息传递损失；FP4量化在极端精度敏感的任务上仍需谨慎评估；百万token上下文的实际服务延迟和成本尚待实际部署验证。