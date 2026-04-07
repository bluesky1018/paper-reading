---
layout: post
title: "LIBERO-Para：视觉-语言-动作模型的释义鲁棒性诊断基准"
date: 2026-04-08
categories: [论文解读, 机器人学习]
tags: [VLA, 视觉语言动作, 机器人, 鲁棒性, Benchmark]
---

> 📄 **论文**：LIBERO-Para: A Diagnostic Benchmark and Metrics for Paraphrase Robustness in VLA Models
> 🔗 **arXiv**：[2603.28301](https://arxiv.org/abs/2603.28301)
> 🏢 **机构**：Human-centered AI Laboratory (HAI-Lab)
> 👥 **作者**：Chanyoung Kim, Minwoo Kim, Minseok Kang, Hyunwoo Kim, Dahuin Jung

## 一句话总结
构建LIBERO-Para基准，专门用于诊断视觉-语言-动作（VLA）模型在语义等价但表述不同的指令下的鲁棒性，揭示了现有模型的语言理解瓶颈

## 背景与问题

Vision-Language-Action (VLA) models achieve strong performance in robotic manipulation by leveraging pre-trained vision-language backbones. However, in downstream robotic settings, they are typically fine-tuned with limited data, leading to overfitting to specific instruction formulations and leaving robustness to paraphrased instructions underexplored. To study this gap, we introduce LIBERO-Para, a controlled benchmark that independently varies action expressions and object references for fine-grained analysis of linguistic generalization. Across seven VLA configurations (0.6B-7.5B), we observe consistent performance degradation of 22-52 pp under paraphrasing. This degradation is primarily driven by object-level lexical variation: even simple synonym substitutions cause large drops, indicating reliance on surface-level matching rather than semantic grounding. Moreover, 80-96% of failures arise from planning-level trajectory divergence rather than execution errors, showing that paraphrasing disrupts task identification. Binary success rate treats all paraphrases equally, obscuring whether models perform consistently across difficulty levels or rely on easier cases. To address this, we propose PRIDE, a metric that quantifies paraphrase difficulty using semantic and syntactic factors. Our benchmark and corresponding code are available at: this https URL



## 核心方法

Across seven configurations spanning four architecture families (OpenVLA-OFT, π\pi0.5/Xiaomi-Robotics-0, X-VLA, VLA-Adapter), all models show substantial success rate drops under paraphrasing, ranging from 22.8 pp to 51.9 pp (Tab. 2). The 7.5B OpenVLA-OFT shows PRIDE scores comparable to the 0.9B X-VLA. All models exhibit PRIDE overestimation of 8.4–22.0% (Tab. 3). Overall, VLAs consistently experience significant performance degradation under paraphrased instructions, regardless of architecture or scale.


![Figure 1: Illustration of paraphrase robustness gap under data-scarce fine-tunin](https://arxiv.org/html/2603.28301/2603.28301v1/x1.png)
*图：Figure 1: Illustration of paraphrase robustness gap under data-scarce fine-tuning: VLA models can overfit to seen instruction phrasings during fine-tu*


![Figure 2: Overview of LIBERO-Para. Compared to LIBERO, LIBERO-Para evaluates par](https://arxiv.org/html/2603.28301/2603.28301v1/x2.png)
*图：Figure 2: Overview of LIBERO-Para. Compared to LIBERO, LIBERO-Para evaluates paraphrase robustness under data-scarce fine-tuning via a controlled two-*


![Figure 3: Examples of axis-specific paraphrases. Object variations modify target](https://arxiv.org/html/2603.28301/2603.28301v1/x3.png)
*图：Figure 3: Examples of axis-specific paraphrases. Object variations modify target object references (e.g., same-polarity substitution, addition), while*


![Figure 4: SKS_{K} (top) and STS_{T} (bottom) computation. SKS_{K} is based on se](https://arxiv.org/html/2603.28301/2603.28301v1/x4.png)
*图：Figure 4: SKS_{K} (top) and STS_{T} (bottom) computation. SKS_{K} is based on semantic matching between task-critical content words, while STS_{T} use*


![Figure 5: Average PRIDE score per Object × Action cell in LIBERO-Para (darker = ](https://arxiv.org/html/2603.28301/2603.28301v1/x5.png)
*图：Figure 5: Average PRIDE score per Object × Action cell in LIBERO-Para (darker = harder). Scores increase along both axes, with the most indirect actio*


![Figure 6: Model-average success rate per Object × Action cell. Object-paraphrase](https://arxiv.org/html/2603.28301/2603.28301v1/x6.png)
*图：Figure 6: Model-average success rate per Object × Action cell. Object-paraphrased rows drop sharply compared to object-preserved rows, reaching 30.4% *


## 实验结果

Each model is evaluated across 5 different random seeds (7, 8, 9, 10, 11) per task–paraphrase configuration. All reported success rates represent the mean over 5 seeds; standard deviations are not reported, as our analysis focuses on aggregate robustness trends across paraphrase types rather than per-configuration variance. We use the LIBERO simulation environment with its default evaluation settings (i.e., maximum episode length and success criteria) as defined in the original LIBERO benchmark.


![Figure 7: Success rate comparison between object-preserved (None, Addition) and ](https://arxiv.org/html/2603.28301/2603.28301v1/x7.png)
*图：Figure 7: Success rate comparison between object-preserved (None, Addition) and object-paraphrased (SP-contextual, SP-habitual) instructions. All mode*


![Figure 8: (Left) LIBERO scene for Task 2: Push the plate to the front of the sto](https://arxiv.org/html/2603.28301/2603.28301v1/x8.png)
*图：Figure 8: (Left) LIBERO scene for Task 2: Push the plate to the front of the stove. (Right) 3D end-effector trajectories under a paraphrased instructi*


![Figure 9: Overview of the LIBERO-Para dataset generation workflow. The process c](https://arxiv.org/html/2603.28301/2603.28301v1/x9.png)
*图：Figure 9: Overview of the LIBERO-Para dataset generation workflow. The process consists of four stages: (1) axis-wise paraphrase generation, (2) verif*


![Figure 10: Average Structural Distance (1−ST1-S_{T}) per Object × Action cell. T](https://arxiv.org/html/2603.28301/2603.28301v1/x10.png)
*图：Figure 10: Average Structural Distance (1−ST1-S_{T}) per Object × Action cell. This component reflects syntactic divergence only (SK weight = 0.0, ST *


![Figure 11: Average Keyword Distance (1 – SKS_{K}) per Object × Action cell. This](https://arxiv.org/html/2603.28301/2603.28301v1/x11.png)
*图：Figure 11: Average Keyword Distance (1 – SKS_{K}) per Object × Action cell. This component reflects lexical divergence only (SK weight = 1.0, ST weigh*


![Figure 12: LIBERO-Goal task instructions (left) and corresponding scene with can](https://arxiv.org/html/2603.28301/2603.28301v1/x12.png)
*图：Figure 12: LIBERO-Goal task instructions (left) and corresponding scene with canonical object names (right). Each object is referred to by a single un*


## 总结

本文提出了 **LIBERO-Para**，构建LIBERO-Para基准，专门用于诊断视觉-语言-动作（VLA）模型在语义等价但表述不同的指令下的鲁棒性，揭示了现有模型的语言理解瓶颈。该工作从理论和实践层面均有创新，为后续研究提供了重要参考。

**局限性与未来方向：** 如所有工作一样，该研究仍有一定局限性，后续可在更大规模数据集、更多样化场景下进行验证和拓展。
