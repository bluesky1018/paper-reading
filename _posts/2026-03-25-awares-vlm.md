---
layout: post
title: "AwaRes：按需检索高分辨率图像块，让视觉语言模型更高效"
date: 2026-03-25
categories: [论文解读, 多模态模型]
tags: ["视觉语言模型", "高分辨率", "效率优化", "工具调用"]
---

> 📄 **论文**：Look Where It Matters: High-Resolution Crops Retrieval for Efficient VLMs
> 🔗 **arXiv**：[2603.16932](https://arxiv.org/abs/2603.16932)
> 🏢 **机构**：Nimrod Shabtay et al.

## 一句话总结

AwaRes框架让VLM在低分辨率全局视图下工作，仅对关键区域按需检索高分辨率图像块，显著降低计算开销。

## 背景与问题

1 Introduction Vision–language models (VLMs) increasingly rely on high-resolution visual inputs to solve detail-sensitive tasks such as document question answering, chart understanding, and understanding semantics and text in dense natural images. However, high resolution is expensive: the number of visual tokens grows rapidly with image resolution, making high-resolution inference a major bottleneck in practice. Existing approaches to reduce this cost largely fall into two camps. First, token pruning methods selectively discard visual tokens to reduce computation [ fastv , Pyramiddrop , VisionZip , SparseVLM , HoloV ] . While effective in principle, they often introduce irregular token patt


![Figure 1: AwaRes overview. Left: Given a low-resolution image, AwaRes uses tool-calling to request only the high-resolution crops needed to answer the](https://arxiv.org/html/2603.16932v1/x1.png)
*图1：Figure 1: AwaRes overview. Left: Given a low-resolution image, AwaRes uses tool-calling to request only the high-resolution crops needed to answer the*


现有方法存在明显局限性：缺乏系统性的评测或方法框架来解决上述问题。本文的核心动机正是填补这一空白，提出更有效的解决方案。


![Figure 2: Overview of the automatic supervision pipeline. Each sample is processed at two resolutions; an LLM judge determines resolution sufficiency ](https://arxiv.org/html/2603.16932v1/x2.png)
*图2：Figure 2: Overview of the automatic supervision pipeline. Each sample is processed at two resolutions; an LLM judge determines resolution sufficiency *


## 核心方法

2 Related Work Several strategies have emerged to prune, compress, or dynamically reduce the number of visual tokens in Vision Language Models. One line of research focuses on dynamic token pruning. Methods such as FastV [ fastv ] , HoloV [ HoloV ] , PyramidDrop [ Pyramiddrop ] , FitPrune [ fitprune ] , TopV [ TopV ] , SparseVILA [ SparseVILA ] , IVTP [ ivtp ] , LLaVolta [ llavolta ] , and SAINT [ saint ] discard uninformative tokens within the LLM layers based on attention scores or learned criteria. Alternatively, VisionZip [ VisionZip ] , FastVLM [ FastVLM ] , and SparseVLM [ SparseVLM ] prune tokens directly after the vision encoder. While effective, pruning-based approaches must commit to a fixed retention ratio before inference, applying the same token budget regardless of sample com


![Figure 3: Crop annotation example. Left: low-resolution input where text is illegible. Middle: oracle-predicted bounding box localizing the answer reg](https://arxiv.org/html/2603.16932v1/x3.png)
*图3：Figure 3: Crop annotation example. Left: low-resolution input where text is illegible. Middle: oracle-predicted bounding box localizing the answer reg*



![Table 1: Main results across vision-language benchmarks. We compare AwaRes against fixed-ratio efficient methods (VisionZIP, SparseVLM, Holo-V) and ad](https://arxiv.org/html/2603.16932v1/x4.png)
*图4：Table 1: Main results across vision-language benchmarks. We compare AwaRes against fixed-ratio efficient methods (VisionZIP, SparseVLM, Holo-V) and ad*



![Table 2: Agreement on resolution-selection labels. Confusion matrix comparing labels produced by LLaMA-3.3-70B against DeepSeek-V3.2 and ANLS. We obse](https://arxiv.org/html/2603.16932v1/x5.png)
*图5：Table 2: Agreement on resolution-selection labels. Confusion matrix comparing labels produced by LLaMA-3.3-70B against DeepSeek-V3.2 and ANLS. We obse*


## 实验结果

4 Experimental Results We evaluate AwaRes on six benchmarks spanning document understanding and general visual QA, and compare against both fixed-budget token-pruning methods and adaptive resolution-escalation baselines. We report (i) the dataset metric from lmms-eval [ lmmseval ] and (ii) an Retain Token Ratio (RTR), defined as the fraction of visual tokens processed relative to the full-resolution baseline. RTR directly reflects the model’s first-turn coupled-decision policy (answer directly vs. request crops), while accuracy reflects the quality of the full multi-turn interaction. We first describe our evaluation protocol 4.1 , evaluated datasets 4.2 and implementation details 4.3 . Then, we provide a detailed discussion of the main results 4.5 and conclude by extensive ablations 4.6 . 


![Figure 4: Performance vs. Wall Clock Time. AwaRes achieves sub-second average latency across all benchmarks by encoding resolution decisions in short ](https://arxiv.org/html/2603.16932v1/x6.png)
*图6：Figure 4: Performance vs. Wall Clock Time. AwaRes achieves sub-second average latency across all benchmarks by encoding resolution decisions in short *



![Figure 5: From Over-Using to Looking Where It Matters. The flow of crop selection decisions from Oracle GT (left), SFT-tuned model predictions (middle](https://arxiv.org/html/2603.16932v1/x7.png)
*图7：Figure 5: From Over-Using to Looking Where It Matters. The flow of crop selection decisions from Oracle GT (left), SFT-tuned model predictions (middle*


## 总结

6 Data Annotation 6.1 Data Curation Pipeline In this section, we provide in Figure 6 supplementary visual examples from our data curation pipeline. Additionally, in Figure 7 we illustrate failure cases where the automatic process did not produce satisfactory results. Figure 6 : Visual examples from our data curation pipeline. Each row shows the low-resolution input image, the oracle-detected bounding boxes (question region in blue, answer region in red), and the selected crop used for training. Top: A Chart example where the oracle correctly localizes the relevant bar and its label. Bottom: A 

本文工作的主要贡献包括：（1）AwaRes框架让VLM在低分辨率全局视图下工作，仅对关键区域按需检索高分辨率图像块，显著降低计算开销。；（2）通过充分的实验验证了方法的有效性。未来工作可在此基础上进一步探索更大规模、更多样化场景下的应用与扩展。

> 🔗 论文链接：[https://arxiv.org/abs/2603.16932](https://arxiv.org/abs/2603.16932)
