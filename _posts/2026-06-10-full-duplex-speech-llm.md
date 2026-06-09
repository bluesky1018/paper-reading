---
layout: post
title: "全双工语音模型中LLM文本能力的释放"
date: 2026-06-10
categories: [论文解读, 语音LLM]
tags: [全双工语音, LLM, 语音交互, 文本生成, 多模态]
---

> 📄 **论文**：Liberating LLM Capabilities in Full-Duplex Speech Models
> 🔗 **arXiv**：[2606.07547](https://arxiv.org/abs/2606.07547)
> 🏢 **机构**：论文作者机构

## 一句话总结

提出将文本输出能力引入全双工语音LLM，突破语音模型仅能生成口语回复的限制，实现代码生成、结构化分析等文本原生能力与实时语音交互的结合。

## 背景与问题

Speech-based large language models are typically constrained to spoken replies, which limits their user-facing outputs to what can be verbalized and suppresses text-native capabilities such as code generation, structured analysis, and multi-step reasoning in realtime interaction, for tasks that require persistent, structured, and inspectable intermediate outputs. Existing work improves spoken reasoning or full-duplex turn-taking, but still treats text as a hidden intermediate state or a subordin


![Figure 1: Listen-Write-Speak (LWS) tri-channel architecture. The interaction is ](https://arxiv.org/html/2606.07547/2606.07547v1/x1.png)
*图：Figure 1: Listen-Write-Speak (LWS) tri-channel architecture. The interaction is partitioned into tem*

Human communication is inherently multimodal, yet speech and writing do not share the burden equally. Decades of work in cognitive science, CSCW, and multimodal HCI suggest that different representational media support different functions: speech is especially effective for turn-taking, grounding, and pragmatic coordination, whereas written or visual representations are better suited to precise, structured, persistent, and spatially organized information (Clark and Brennan, 1991 ; Oviatt, 1999 ;

## 核心方法

Training LWS requires data with per-second cognitive annotations aligned to an audio timeline, a format that does not exist in public corpora. We therefore design a two-stage pipeline that starts from standard text QA pairs and converts them into the Unit-based interaction format used by the model.


![Figure 2: Per-channel training loss curves. (a) ls_cogn (listening-phase writing](https://arxiv.org/html/2606.07547/2606.07547v1/x2.png)
*图：Figure 2: Per-channel training loss curves. (a) ls_cogn (listening-phase writing), (b) speak (speaki*


![Figure 1: Listen-Write-Speak (LWS) tri-channel architecture. The interaction is partitioned into tem](https://arxiv.org/html/2606.07547/2606.07547v1/x1.png)
*图1：Figure 1: Listen-Write-Speak (LWS) tri-channel architecture. The interaction is partitioned into tem*

![Figure 2: Per-channel training loss curves. (a) ls_cogn (listening-phase writing), (b) speak (speaki](https://arxiv.org/html/2606.07547/2606.07547v1/x2.png)
*图2：Figure 2: Per-channel training loss curves. (a) ls_cogn (listening-phase writing), (b) speak (speaki*


## 实验结果

LWS combines full-duplex listening, visible writing, and real-time speaking within a standard autoregressive framework, but the present study remains subject to several limitations. (1) Reasoning depth remains constrained by real-time operation. Because visible writing and speech must be produced within each Unit, the system is optimized for responsiveness and is not yet well suited to longer-horizon reasoning, multi-step planning, or complex tool-mediated workflows, where stronger performance may require slower response generation or an explicit mechanism for deferring speech while deeper wri



### 实验数据表格

| Paradigm                | models                     | FD  | FT  | CL  | CS  |
| ----------------------- | -------------------------- | --- | --- | --- | --- |
| Think-Before-Speak      | Step-Audio 2, TVS          | ✗   | ✓   | ✗   | ✗   |
| Interleaved Think-Speak | STITCH, Mini-Omni-Reasoner | ✗   | ✓   | ✗   | ✓   |
| Dual-Brain Think-Speak  | MPS                        | ✗   | ✓   | ✗   | ✓   |
| Think-While-Listen      | SHANKS, Chron. Think.      | ✓ † | ✓   | ✓   | ✗   |
| Full-Duplex             | Moshi, LSLM, FlexDuo       | ✓   | ✗   | ✗   | ✗   |
| Parallel Text-Speech    | Qwen3-Omni, Kimi-Audio     | ✗   | ✓   | ✗   | ✗   |
| Multi-Channel Protocol  | OpenAI Harmony             | ✗   | ✓   | ✗   | ✗   |
| Listen-Write-Speak      | Ours                       | ✓   | ✓   | ✓   | ✓   |

## 总结

Liberating LLM Capabilities in Full-Duplex Speech Models 提出了一个新颖的研究框架，针对语音LLM领域的核心挑战提供了系统性解决方案。

**主要贡献：**
- 提出将文本输出能力引入全双工语音LLM，突破语音模型仅能生成口语回复的限制，实现代码生成、结构化分析等文本原生能力与实时语音交互的结合。
- 通过系统实验验证了方法的有效性
- 为后续研究提供了重要的基准和参考

**局限性与展望：** 未来工作可进一步探索方法在更广泛场景下的应用，以及结合更多领域知识提升系统性能。