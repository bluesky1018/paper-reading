---
layout: post
title: "LatentSkill：将文本技能转化为权重内潜在技能"
date: 2026-06-10
categories: [论文解读, LLM智能体]
tags: [LoRA, 智能体技能, 超网络, LLM, 参数高效微调]
---

> 📄 **论文**：LatentSkill : From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents
> 🔗 **arXiv**：[2606.06087](https://arxiv.org/abs/2606.06087)
> 🏢 **机构**：论文作者机构

## 一句话总结

提出LatentSkill框架，通过超网络将文本技能转化为即插即用的LoRA适配器，将技能知识存入权重空间而非上下文空间，大幅减少推理时的token消耗。

## 背景与问题

Agent systems increasingly use textual skills to encode reusable task procedures, but injecting these skills into the prompt at every step incurs substantial context overhead and exposes skill content as plaintext. We present LatentSkill , a framework that converts textual skills into plug-and-play LoRA adapters through a pretrained hypernetwork. LatentSkill stores skill knowledge in weight space rather than context space, removing per-step skill tokens while preserving modular loading, scaling,


![Figure 1: The key advantages of LatentSkill over in-context skill: (1) zero skil](https://arxiv.org/html/2606.06087/2606.06087v1/figures/motivation_1.png)
*图：Figure 1: The key advantages of LatentSkill over in-context skill: (1) zero skill tokens in prompt w*

LLM agents increasingly solve complex tasks by interleaving reasoning, action, and feedback from external environments (Yao et al. , 2023 ; Shinn et al. , 2023 ; Zhao et al. , 2024 ) . To handle specialized and long-horizon tasks, many systems further rely on external skills: reusable textual procedures that encode task strategies, tool-use patterns, and recovery heuristics (Wang et al. , 2023 ; Xia et al. , 2026 ; Wu et al. , 2026 ; Ouyang et al. , 2026 ; Pan et al. , 2026 ; Wang et al. , 2026 

## 核心方法

We evaluate LatentSkill on two agent benchmarks. ALFWorld (Shridhar et al. , 2021 ) is a text-based interactive environment aligned with the ALFRED embodied AI benchmark, comprising six categories of household tasks: Pick and Place (Pick), Look at Obj in Light (Look), Pick Clean then Place in Recep (Clean), Pick Heat then Place in Recep (Heat), Pick Cool then Place in Recep (Cool), and Pick Two Obj and Place (Pick2). Search-QA follows the evaluation protocol of Jin et al. ( 2025 ) and covers seven search-augmented QA datasets, including three single-hop benchmarks (NQ (Kwiatkowski et al. , 201


![Figure 2: Overview of LatentSkill . Left : textual skills are transformed into i](https://arxiv.org/html/2606.06087/2606.06087v1/figures/method.png)
*图：Figure 2: Overview of LatentSkill . Left : textual skills are transformed into in-weight latent skil*


![Figure 1: The key advantages of LatentSkill over in-context skill: (1) zero skill tokens in prompt w](https://arxiv.org/html/2606.06087/2606.06087v1/figures/motivation_1.png)
*图1：Figure 1: The key advantages of LatentSkill over in-context skill: (1) zero skill tokens in prompt w*

![Figure 2: Overview of LatentSkill . Left : textual skills are transformed into in-weight latent skil](https://arxiv.org/html/2606.06087/2606.06087v1/figures/method.png)
*图2：Figure 2: Overview of LatentSkill . Left : textual skills are transformed into in-weight latent skil*

![Figure 3: MDS visualization of LoRA weights. Left : in-domain ALFWorld and Search skills; Right : OO](https://arxiv.org/html/2606.06087/2606.06087v1/figures/lora_mds_heng.jpg)
*图3：Figure 3: MDS visualization of LoRA weights. Left : in-domain ALFWorld and Search skills; Right : OO*

![Figure 4: Scale-performance curves on ALFWorld under varying LoRA injection coefficient . Top : Pick](https://arxiv.org/html/2606.06087/2606.06087v1/figures/scale_analysis_clear.jpg)
*图4：Figure 4: Scale-performance curves on ALFWorld under varying LoRA injection coefficient . Top : Pick*

![Figure 5: Per-module discriminability (within-domain minus cross-domain cosine similarity gap) for t](https://arxiv.org/html/2606.06087/2606.06087v1/figures/submodule_discriminability.png)
*图5：Figure 5: Per-module discriminability (within-domain minus cross-domain cosine similarity gap) for t*


## 实验结果

LatentSkill converts textual agent skills into modular LoRA adapters through a pretrained hypernetwork, moving reusable procedural knowledge from context space into weight space. Across ALFWorld and Search-QA, this design improves over direct in-context skill prompting while substantially reducing the repeated prefill overhead introduced by skill text. Beyond efficiency, our analyses show that the generated skill LoRAs form a structured semantic geometry, can be controlled through the injection coefficient, and can be composed in parameter space when skill components are properly aligned. Thes


![Figure 3: MDS visualization of LoRA weights. Left : in-domain ALFWorld and Searc](https://arxiv.org/html/2606.06087/2606.06087v1/figures/lora_mds_heng.jpg)
*图：Figure 3: MDS visualization of LoRA weights. Left : in-domain ALFWorld and Search skills; Right : OO*


## 总结

LatentSkill : From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents 提出了一个新颖的研究框架，针对LLM智能体领域的核心挑战提供了系统性解决方案。

**主要贡献：**
- 提出LatentSkill框架，通过超网络将文本技能转化为即插即用的LoRA适配器，将技能知识存入权重空间而非上下文空间，大幅减少推理时的token消耗。
- 通过系统实验验证了方法的有效性
- 为后续研究提供了重要的基准和参考

**局限性与展望：** 未来工作可进一步探索方法在更广泛场景下的应用，以及结合更多领域知识提升系统性能。