---
layout: post
title: "模型何时应该改变想法？大语言模型中的上下文信念管理"
date: 2026-05-30
categories: [论文解读, LLM推理]
tags: ["LLM", "信念管理", "上下文学习", "长文本", "知识更新"]
---

> 📄 **论文**：When Should Models Change Their Minds? Contextual Belief Management in Large Language Models
> 🔗 **arXiv**：[2605.30219](https://arxiv.org/abs/2605.30219)
> 🏢 **机构**：

## 一句话总结

Long-horizon interactions require language models to manage accumulating information: when to update their state, when to preserve their state, and what to ignore. We study this challenge as Contextua...

## 背景与问题

Large language models (LLMs) are increasingly deployed in long-horizon interactions, where their behavior depends not only on parametric knowledge but also on context, memory, tools, and runtime protocols Yang et al. ( 2024 ); Zhou et al. ( 2023 ); Wu et al. ( 2024 ); Lee et al. ( 2026 ) . In such settings, models must manage beliefs as different types of information accumulate over time. Some information should revise the model’s current belief state, some should leave it unchanged, and some should be ignored altogether. Recent work on context learning, such as CL-Bench Dou et al. ( 2026 ) , studies whether models can absorb rules, knowledge, or procedures from context and translate them into effective behavior. However, absorbing contextual information is not enough: a model must also decide which information counts as formal evidence, when that evidence warrants belief revision, and when task-irrelevant context should be filtered out.

As Figure 1 shows, we study this problem as Contextual Belief Management (CBM): a model’s ability to maintain an evidence-aligned belief state throughout a multi-turn interaction. Rather than simulating open-ended dialogue, we operationalize CBM in a controlled closed-world setting. Specifically, we introduce BeliefTrack , a closed-world benchmark with two environments: Rule Discovery (RD) and Circuit Diagnosis (CD). Both environments define finite belief spaces and use symbolic verifiers, allowing exact turn-level comparison between predict


![Figure 1: Overview of Contextual Belief Management (CBM). CBM requires models to maintain a predicte](https://arxiv.org/html/2605.30219/2605.30219v1/x1.png)
*图：Figure 1: Overview of Contextual Belief Management (CBM). CBM requires models to maintain a predicte*


![Figure 2: Comparison between Contextual Belief Management and Theory of Mind .](https://arxiv.org/html/2605.30219/2605.30219v1/x2.png)
*图：Figure 2: Comparison between Contextual Belief Management and Theory of Mind .*


Knowledge Conflict. Deciding which information to trust is central to belief management in language models. Prior work shows that models struggle to resolve conflicts between parametric memory and context from passages, user claims, demonstrations (Longpre et al., 2021 ; Wang et al., 2024 ; Xu et al., 2024c ; Kortukov et al., 2024 ; Jin et al., 2024 ; Xie et al., 2024 ; Xu et al., 2024d ; Hagström et al., 2026 ) . Recent work further highlights belief dependencies in conflict resolution, where updating one fact can affect others (Yao et al., 2025 ; Xu et al., 2026 ) . By contrast, CBM does not introduce direct information conflicts, but tests whether models update beliefs only from formal evidence.

Multi-turn Reasoning Instability. LLMs often become unreliable in long interactions: they l

## 核心方法

We evaluate two methods for improving CBM: Belief-Tracking Prompt ( BT-Prompt ), a training-free prompt-based enhancement method, and RL with belief-state rewards , a verifier-guided reinforcement-learning method.

BT-Prompt is a parameter-free test-time baseline that encodes the CBM procedure in the system prompt. It instructs the model to maintain the current set of valid formal evidence, ignore non-evidential noise, re-evaluate all candidate hypotheses against the accumulated evidence, and revise the evidence set when explicit corrections invalidate earlier observations. This allows previously eliminated hypotheses to be restored when the evidence excluding them is removed. BT-Prompt is applied uniformly across both environments and all diagnostic trajectory types; full templates are provided in Appendix D.2 .


![Figure 3: BeliefTrack framework. Given a finite belief space, the model must output a predicted beli](https://arxiv.org/html/2605.30219/2605.30219v1/x3.png)
*图：Figure 3: BeliefTrack framework. Given a finite belief space, the model must output a predicted beli*


![Figure 4: Effects of temporal stress and task-irrelevant context on CBM. Left : FSR under increasing](https://arxiv.org/html/2605.30219/2605.30219v1/x4.png)
*图：Figure 4: Effects of temporal stress and task-irrelevant context on CBM. Left : FSR under increasing*


![Figure 5: Mechanistic probing and steering of CBM failures. (a) Probing compares Vanilla and RL by t](https://arxiv.org/html/2605.30219/2605.30219v1/x5.png)
*图：Figure 5: Mechanistic probing and steering of CBM failures. (a) Probing compares Vanilla and RL by t*


![Figure 6: RL training dynamics across checkpoints. We report FSR, FUR, and FIR on Rule Discovery and](https://arxiv.org/html/2605.30219/2605.30219v1/x6.png)
*图：Figure 6: RL training dynamics across checkpoints. We report FSR, FUR, and FIR on Rule Discovery and*


## 实验结果

Based on the diagnostic datasets defined in Section 3.3 , we use a strict k k -repeat evaluation protocol. For each diagnostic sample x x , the user-side multi-turn template is fixed, and we independently sample k k assistant-side trajectories. Let E m ( i ) ​ ( x ) ∈ { 0 , 1 } E_{m}^{(i)}(x)\in\{0,1\} indicate whether the i i -th trajectory exhibits the target failure mode m ∈ { stay , update , iso } m\in\{\mathrm{stay},\mathrm{update},\mathrm{iso}\} . Here, stay \mathrm{stay} , update \mathrm{update} , and iso \mathrm{iso} correspond to Failed Stay, Failed Update, and Failed Isolation, respectively. We define sample-level failure as

Thus, a sample fails if any repeated trajectory exhibits the target failure.

Failed Stay Rate (FSR). For x ∈ 𝒟 stay x\in\mathcal{D}^{\mathrm{stay}} , E stay ( i ) ​ ( x ) = 1 E_{\mathrm{stay}}^{(i)}(x)=1 if the i i -th trajectory makes a Failed Stay error on the evaluated post-lock turns.

Failed Update Rate (FUR). For x ∈ 𝒟 update x\in\mathcal{D}^{\mathrm{update}} , E update ( i ) ​ ( x ) = 1 E_{\mathrm{update}}^{(i)}(x)=1 if the i i -th trajectory makes a Failed Update error at the correction turn.


![Figure 7: Prompt Templates A](https://arxiv.org/html/2605.30219/2605.30219v1/x7.png)
*图：Figure 7: Prompt Templates A*


![Figure 8: Prompt Templates B](https://arxiv.org/html/2605.30219/2605.30219v1/x8.png)
*图：Figure 8: Prompt Templates B*


![Figure 9: Belief-State Drift and Backtracking Failure. As conversational depth increases or explicit](https://arxiv.org/html/2605.30219/2605.30219v1/x9.png)
*图：Figure 9: Belief-State Drift and Backtracking Failure. As conversational depth increases or explicit*


![Figure 10: Contextual Hijacking. When task-irrelevant context/noise is injected, models frequently a](https://arxiv.org/html/2605.30219/2605.30219v1/x10.png)
*图：Figure 10: Contextual Hijacking. When task-irrelevant context/noise is injected, models frequently a*


## 全文图示


![Figure 11: Latent-Output Gap. A frequent failure occurs when the Vanilla model ranks oracle-supporte](https://arxiv.org/html/2605.30219/2605.30219v1/x11.png)
*图：Figure 11: Latent-Output Gap. A frequent failure occurs when the Vanilla model ranks oracle-supporte*


## 总结

We introduced Contextual Belief Management (CBM) and BeliefTrack to study evidence-aligned belief tracking in long-horizon interactions. Current LLMs exhibit substantial CBM failures that prompting does not reliably fix, while verifier-guided reward learning improves belief management and generalizes across environments. Our probing and steering analyses further suggest that these failures are associated with modifiable representation-level patterns, making CBM both measurable and actionable.

