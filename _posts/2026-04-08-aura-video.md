---
layout: post
title: "AURA：基于视频流的全天候理解与实时辅助系统"
date: 2026-04-08
categories: [论文解读, 多模态AI]
tags: [视频理解, 实时推理, 多模态, 持续学习, Video LLM]
---

> 📄 **论文**：AURA: Always-On Understanding and Real-Time Assistance via Video Streams
> 🔗 **arXiv**：[2604.04184](https://arxiv.org/abs/2604.04184)
> 🏢 **机构**：多机构合作（见原文）
> 👥 **作者**：Xudong Lu, Yang Bo, Jinpeng Chen, Shuhan Li, Xintong Guo, Huankang Guan, Fang Liu, Dunyuan Xu, Peiwen Sun, Heyang Sun, Rui Liu, Hongsheng Li

## 一句话总结
提出AURA系统，实现对连续视频流的全天候理解与实时辅助，无需用户主动触发即可主动提供情境化帮助

## 背景与问题

Video Large Language Models (VideoLLMs) have achieved strong performance on many video understanding tasks, but most existing systems remain offline and are not well-suited for live video streams that require continuous observation and timely response. Recent streaming VideoLLMs have made progress, yet current approaches often rely on decoupled trigger-response pipelines or are limited to captioning-style narration, reducing their effectiveness for open-ended question answering and long-horizon interaction. We propose AURA (Always-On Understanding and Real-Time Assistance), an end-to-end streaming visual interaction framework that enables a unified VideoLLM to continuously process video streams and support both real-time question answering and proactive responses. AURA integrates context management, data construction, training objectives, and deployment optimization for stable long-horizon streaming interaction. It achieves state-of-the-art performance on streaming benchmarks and supports a real-time demo system with ASR and TTS running at 2 FPS on two 80G accelerators. We release the AURA model together with a real-time inference framework to facilitate future research.



## 核心方法

详见原文方法章节。


![Figure 1: Overview of our Interactive Video Stream Context Management mechanism.](https://arxiv.org/html/2604.04184/2604.04184v1/x3.png)
*图：Figure 1: Overview of our Interactive Video Stream Context Management mechanism. The framework uses a dual sliding-window strategy, where NN denotes t*


![Figure 2: The figure illustrates three types of streaming QA interactions. Real-](https://arxiv.org/html/2604.04184/2604.04184v1/x4.png)
*图：Figure 2: The figure illustrates three types of streaming QA interactions. Real-Time QA produces a single immediate response at the query time. Proact*


![Figure 3: Overview of the Coarse-to-Fine Streaming Data Engine in AURA. The pipe](https://arxiv.org/html/2604.04184/2604.04184v1/x5.png)
*图：Figure 3: Overview of the Coarse-to-Fine Streaming Data Engine in AURA. The pipeline comprises five stages: (1) Video Preparation, (2) QA Synthesis, (*


## 实验结果

We conduct all evaluations on computing nodes with the same specifications as those used for training, and use the official codebases for the corresponding benchmarks (lin2024streamingbench; niu2025ovo; wang2025omnimmi). We manage model context using our Interactive Video Stream Context Management mechanism. For other models, we report official results when complete results are publicly available (fu2025vispeak; wang2025omnimmi; xia2025streaming; yang2025streamagent); otherwise, we evaluate them by strictly following the code released in the official benchmark repositories (minicpmo45; bai2025qwen3). We also note that, although MiniCPM-o-4.5 supports a full-duplex multimodal live-streaming mode, we find that it often becomes silent in this setting and may produce irrelevant responses for v


![Figure 4: Overview of AURA’s end-to-end real-time inference system with video an](https://arxiv.org/html/2604.04184/2604.04184v1/x6.png)
*图：Figure 4: Overview of AURA’s end-to-end real-time inference system with video and speech input, multimodal inference, and speech output. The system is*


![Figure 5: Training data distribution: Left: QA type distribution; Right: Video d](https://arxiv.org/html/2604.04184/2604.04184v1/x7.png)
*图：Figure 5: Training data distribution: Left: QA type distribution; Right: Video domain distribution. It shows that the training set covers diverse ques*


![Figure 6: Inference performance of AURA. The figure compares (a) TTFT and (b) co](https://arxiv.org/html/2604.04184/2604.04184v1/x8.png)
*图：Figure 6: Inference performance of AURA. The figure compares (a) TTFT and (b) computed-token count across three settings: w/o sliding window, w/o pref*


## 总结

本文提出了 **AURA**，提出AURA系统，实现对连续视频流的全天候理解与实时辅助，无需用户主动触发即可主动提供情境化帮助。该工作从理论和实践层面均有创新，为后续研究提供了重要参考。

**局限性与未来方向：** 如所有工作一样，该研究仍有一定局限性，后续可在更大规模数据集、更多样化场景下进行验证和拓展。
