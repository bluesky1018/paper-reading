---
layout: post
title: "VideoDetective：融合外部查询与内在相关性的长视频线索检索框架"
date: 2026-03-25
categories: [论文解读, 视频理解]
tags: ["长视频理解", "多模态大模型", "视频问答", "视觉-时序亲和图"]
---

> 📄 **论文**：VideoDetective: Clue Hunting via Extrinsic Query and Intrinsic Relevance for Long Video Understanding
> 🔗 **arXiv**：[2603.22285](https://arxiv.org/abs/2603.22285)
> 🏢 **机构**：Ruoliu Yang et al. (南京大学)

## 一句话总结

VideoDetective通过构建视觉-时序亲和图，融合外部查询相关性和内在段间亲和性，有效定位长视频中的关键线索片段。

## 背景与问题

1 Introduction Long video understanding has become a central topic in the multimodal community, and a growing number of MLLMs tailored for long-video understanding (Chen et al., 2024a ; Zhang et al., 2024a ; Shen et al., 2025 ; Shu et al., 2025 ) have emerged. Despite this progress, processing massive information within limited context windows remains a critical challenge. As a result, many query-driven approaches focus on locating only the query-relevant clue segments, thereby substantially reducing the effective context length. However, reliably localizing such clues without exhaustively understanding the entire video is inherently difficult, especially for questions requiring complex reas



现有方法存在明显局限性：缺乏系统性的评测或方法框架来解决上述问题。本文的核心动机正是填补这一空白，提出更有效的解决方案。



## 核心方法

2 Related Work Multimodal Large Language Models. MLLMs (Hurst et al., 2024 ; Lin et al., 2024 ; Bai et al., 2025b ; Comanici et al., 2025 ) combine visual encoders (Radford et al., 2021 ; Zhai et al., 2023 ) with LLMs (Achiam et al., 2023 ; Liu et al., 2024a ; Yang et al., 2025 ) , achieving remarkable progress in vision-language tasks. However, most MLLMs struggle with long-form content due to attention complexity and limited context windows. While some recent models (Chen et al., 2024a ; Shen et al., 2025 ; Comanici et al., 2025 ) extend context window length to millions of tokens, the computational cost remains prohibitive for dense sampling. Long Video Understanding. Long video understanding remains challenging due to the long temporal horizon and limited context budgets. Recent advanc







## 实验结果

4 Experiments Figure 2 : Performance improvements across different backbones on VideoMME-long w/o subtitle. VideoDetective consistently enhances various vision-language models across different architectures and parameter scales, demonstrating its plug-and-play capability. 4.1 Experiments Setup Benchmarks. To comprehensively evaluate the overall performance of VideoDetective in long-video understanding, we conduct experiments on four representative benchmarks. Specifically, we evaluate on the long-video subset without subtitles (Long subset w/o subtitles) of VideoMME (Fu et al., 2025a ) and LVBench (Wang et al., 2025b ) without auxiliary transcripts, and complete evaluations on the validation split (Val split) of LongVideoBench (Wu et al., 2024 ) and the test split (Test split) of MLVU (Zho





## 总结

5 Conclusion We present VideoDetective , an inference framework that integrates both extrinsic query relevance and intrinsic video correlations. By modeling a long video as a visual–temporal affinity graph and performing a hypothesis–verification–refinement inference loop, we propagate query-relevance signals from sparse local observations to the entire video, thereby locating critical clues for long-video question answering. Extensive experiments on four challenging benchmarks demonstrate that our approach achieves competitive performance against strong MLLMs and consistently outperforms exis

本文工作的主要贡献包括：（1）VideoDetective通过构建视觉-时序亲和图，融合外部查询相关性和内在段间亲和性，有效定位长视频中的关键线索片段。；（2）通过充分的实验验证了方法的有效性。未来工作可在此基础上进一步探索更大规模、更多样化场景下的应用与扩展。

> 🔗 论文链接：[https://arxiv.org/abs/2603.22285](https://arxiv.org/abs/2603.22285)
