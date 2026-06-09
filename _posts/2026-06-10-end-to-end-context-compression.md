---
layout: post
title: "端到端上下文压缩的大规模应用研究"
date: 2026-06-10
categories: [论文解读, 长上下文LLM]
tags: [上下文压缩, KV缓存, 长上下文, LLM推理, 内存优化]
---

> 📄 **论文**：End-to-End Context Compression at Scale
> 🔗 **arXiv**：[2606.09659](https://arxiv.org/abs/2606.09659)
> 🏢 **机构**：论文作者机构

## 一句话总结

提出端到端上下文压缩方法，通过编码器-解码器架构将长上下文映射为紧凑表示，在不降低模型质量的同时大幅减少KV缓存内存占用。

## 背景与问题

Long-context language model inference is bottlenecked by memory, as the KV cache grows with context length. Recent techniques to compress the KV cache fall short: they either degrade model quality substantially or require considerable time and compute to compress a single long prompt. Furthermore, many methods require the input to fit within the target model’s context window, and are generally incompatible with modern production inference engines. Encoder-decoder compressors, which map a long to


![Figure 1 : Our Latent Context Language Models achieve high quality compression w](https://arxiv.org/html/2606.09659/2606.09659v1/x1.png)
*图：Figure 1 : Our Latent Context Language Models achieve high quality compression while being fast and *

Reasoning over long contexts is a crucial capability for state-of-the-art Large Language Models (LLMs), as it enables them to parse long documents, engage in multi-turn conversations, and perform long-horizon agentic tasks. However, in production systems, the input context, working horizon, and memory can grow to millions of tokens, making inference increasingly constrained by memory and latency due to the growth of the KV cache (Hooper et al., 2024 ) . Even when inputs fit within the model’s ma

## 核心方法

We now describe our multi-stage recipe and data for training. In contrast to prior context-compression work that trains specialized models on small-scale in-domain datasets (Tang et al., 2025 ; Feldman and Artzi, 2025 ; Pilchen et al., 2025 ; Cheng et al., 2024 ; Liao et al., 2025 ) , our goal is to preserve the strong performance of a powerful LLM across downstream tasks. To this end, we curate three types of data: continual pre-training data, Supervised Fine-Tuning (SFT) data, and auxiliary reconstruction data.


![Figure 2 : Examples of the three data types used to train LCLMs. We curate conti](https://arxiv.org/html/2606.09659/2606.09659v1/x2.png)
*图：Figure 2 : Examples of the three data types used to train LCLMs. We curate continual pre-training da*


![Figure 1 : Our Latent Context Language Models achieve high quality compression while being fast and ](https://arxiv.org/html/2606.09659/2606.09659v1/x1.png)
*图1：Figure 1 : Our Latent Context Language Models achieve high quality compression while being fast and *

![Figure 2 : Examples of the three data types used to train LCLMs. We curate continual pre-training da](https://arxiv.org/html/2606.09659/2606.09659v1/x2.png)
*图2：Figure 2 : Examples of the three data types used to train LCLMs. We curate continual pre-training da*

![Figure 3 : A from-scratch pre-training sweep identifies the best encoder-decoder compressor architec](https://arxiv.org/html/2606.09659/2606.09659v1/x3.png)
*图3：Figure 3 : A from-scratch pre-training sweep identifies the best encoder-decoder compressor architec*

![Figure 4 : Latent Context Language Models have lower TTFT and peak GPU memory as context length incr](https://arxiv.org/html/2606.09659/2606.09659v1/x4.png)
*图4：Figure 4 : Latent Context Language Models have lower TTFT and peak GPU memory as context length incr*

![Figure 5 : Latent Context Language Models establish a new Pareto frontier on long-context benchmarks](https://arxiv.org/html/2606.09659/2606.09659v1/x5.png)
*图5：Figure 5 : Latent Context Language Models establish a new Pareto frontier on long-context benchmarks*

![Figure 6 : LCLMs can use tools to retrieve compressed context and improve exact string-match accurac](https://arxiv.org/html/2606.09659/2606.09659v1/x6.png)
*图6：Figure 6 : LCLMs can use tools to retrieve compressed context and improve exact string-match accurac*


## 实验结果

We predominantly focus on long-context benchmarks: we evaluate on RULER (Hsieh et al., 2024 ) , LongBench (Bai et al., 2024 ) , and LongHealth (Adams et al., 2025 ) . For the long-context evaluation suite, we maintain instructions as uncompressed tokens, while we compress long-context segments. We report which parts of each benchmark are compressed and which are left as standard uncompressed tokens in Table ˜ 5 .


![Figure 3 : A from-scratch pre-training sweep identifies the best encoder-decoder](https://arxiv.org/html/2606.09659/2606.09659v1/x3.png)
*图：Figure 3 : A from-scratch pre-training sweep identifies the best encoder-decoder compressor architec*


### 实验数据表格

|                         | Stage 0                              | Stage 1          | Stage 2      | Stage 3 |
| ----------------------- | ------------------------------------ | ---------------- | ------------ | ------- |
|                         | Adapter Training                     | Encoder Training | LLM Training | SFT     |
| Training Config         |                                      |                  |              |         |
| Adapter Peak LR         |                                      |                  |              |         |
| Encoder Peak LR         | NA                                   |                  |              |         |
| LLM Peak LR             | NA                                   | NA               |              |         |
| LR scheduler            | 5% warmup steps with cosine decay to |                  |              |         |
| Optimizer               | AdamW ( , )                          |                  |              |         |
| Data                    |                                      |                  |              |         |
| LLM Batch size (tokens) | 4 Million                            |                  |              |         |

## 总结

End-to-End Context Compression at Scale 提出了一个新颖的研究框架，针对长上下文LLM领域的核心挑战提供了系统性解决方案。

**主要贡献：**
- 提出端到端上下文压缩方法，通过编码器-解码器架构将长上下文映射为紧凑表示，在不降低模型质量的同时大幅减少KV缓存内存占用。
- 通过系统实验验证了方法的有效性
- 为后续研究提供了重要的基准和参考

**局限性与展望：** 未来工作可进一步探索方法在更广泛场景下的应用，以及结合更多领域知识提升系统性能。