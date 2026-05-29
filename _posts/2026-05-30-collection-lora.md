---
layout: post
title: "CollectionLoRA：通过多教师在线蒸馏在单一 LoRA 中集成 50 种效果"
date: 2026-05-30
categories: [论文解读, 图像编辑]
tags: ["LoRA", "图像编辑", "知识蒸馏", "扩散模型", "定制化"]
---

> 📄 **论文**：CollectionLoRA: Collecting 50 Effects in 1 LoRA via Multi-Teacher On-Policy Distillation
> 🔗 **arXiv**：[2605.25378](https://arxiv.org/abs/2605.25378)
> 🏢 **机构**：

## 一句话总结

Customized image editing aims to equip pre-trained diffusion models with specific visual effects using limited paired data, typically via Low-Rank Adaptation (LoRA). As the number of desired effects g...

## 背景与问题

Recently, diffusion models [ flux2024 , labs2025flux , flux-2-2025 , qwenimage , sd1 , esser2024scaling , peebles2023scalable ] have revolutionized the field of image editing, enabling unprecedented fine-grained control and high-quality content modification. For customized image editing [ mou2025dreamo , gal2022image , kumari2023multi , wu2025dcoardeepconceptinjection , Photodoodle , ye2023ip , zhang2023addingconditionalcontroltexttoimage , guo2025any2anytryon , zhang2025easycontrol , xie2023omnicontrol , huang2024incontextloradiffusiontransformers , liu2025llm4gen , she2025mosaic , liu2025tfcustom ] , the community typically trains specific effect LoRAs [ LoRA , huang2024incontextloradiffusiontransformers , OmniConsistency ] using limited paired data and cascades them with acceleration LoRA during inference to achieve rapid, few-step generation. However, scaling this paradigm in practice exposes three bottlenecks as illustrated in Fig. 2 (a): (i) Storage costs. Deploying all effect LoRAs imposes substantial storage overhead on individual devices. (ii) Routing latency and errors. Retrieving and dynamically loading specific LoRAs from the LoRA bank introduces inference latency and the risk of routing mismatches. (iii) LoRA conflicts. Linearly combining effect and acceleration LoRAs disrupts the original feature manifolds, inevitably causing concept bleeding and style degradation.

To fundamentally address deployment challenges, we aim to consolidate diverse visual effects into


![Figure 1 : We propose CollectionLoRA, a multi-teacher distillation framework capable of consolidatin](https://arxiv.org/html/2605.25378/2605.25378v1/x1.png)
*图：Figure 1 : We propose CollectionLoRA, a multi-teacher distillation framework capable of consolidatin*


![Figure 2 : Comparison between conventional multi-LoRA pipelines and the proposed CollectionLoRA. Con](https://arxiv.org/html/2605.25378/2605.25378v1/x2.png)
*图：Figure 2 : Comparison between conventional multi-LoRA pipelines and the proposed CollectionLoRA. Con*


Customized image generation has emerged as a pivotal task within the broader landscape of image synthesis, focusing on enabling pretrained diffusion models to understand specific concepts from limited data and re-render them in diverse contexts. Early optimization-based methods, such as Textual Inversion [ gal2022image ] and DreamBooth [ ruiz2023dreambooth ] , paved the way by learning specific tokens or fine-tuning the model for a single subject. Methods like ELITE [ wei2023elite ] , IP-Adapter [ ye2023ip ] , InstantID [ wang2024instantid ] , and MoMA [ song2024moma ] treat personalization as a vision-conditioned generation task by training specialized adapters. With the emergence of Diffusion Transformers (DiT) [ peebles2023scalable ] like FLUX [ flux2024 ] and SD3 [ esser2024scaling ] ,

## 核心方法

To integrate dozens of heterogeneous visual effects and few-step generation capabilities into a single LoRA, we propose the CollectionLoRA framework, which aims to address parameter interference and deployment overheads via multi-teacher distillation. In Sec. 4.1 , we first formally define the general paradigm of visual effect LoRA training and analyze the challenges of multi-LoRA deployment. In Sec. 4.2 , we detail the Probabilistic Dual-Stream Routing mechanism, which leverages general-domain data as structural regularization to enhance model generalization in few-shot effect learning. To ensure the isolation of distinct concepts within a shared parameter space, we describe the Asymmetric Orthogonal Prompting strategy in Sec. 4.3 . Finally, we present the Coarse-to-Fine Distillation Objective in Sec. 4.4 and the total training objective in Sec. 4.5 .

For standard personalized fine-tuning of diffusion models, given the pre-trained base model parameters θ b ​ a ​ s ​ e \theta_{base} and a limited set of paired data for a specific effect e ​ f ​ f ​ e ​ c ​ t i effect_{i} , Low-Rank Adaptation (LoRA) [ LoRA ] is typically employed to learn the effect-specific residual weights Δ ​ θ e ​ f ​ f ​ e ​ c ​ t i \Delta\theta_{effect}^{i} . The training is generally optimized via the Flow Matching loss [ lipman2023flowmatchinggenerativemodeling ] by regressing the target vector field:

where x 0 x_{0} represents the ground-truth target effect image, ϵ ∼ 𝒩 ​ ( 0 , I ) \epsilon\sim\mathcal{N}(0,I) is the sampled standard Gaussian noise, t ∈ [ 0 , 1 ] t\in[0,1] denotes the continuous time step, c c represents the conditioning input, comprising the editing instruction and the source reference image.


![Figure 3 : The overall framework of CollectionLoRA. (a) PDSR dynamically routes training batches int](https://arxiv.org/html/2605.25378/2605.25378v1/x3.png)
*图：Figure 3 : The overall framework of CollectionLoRA. (a) PDSR dynamically routes training batches int*


![Figure 4 : Effectiveness of C2F-DO. (a) Directly applying standard DMD to multi-teacher distillation](https://arxiv.org/html/2605.25378/2605.25378v1/x4.png)
*图：Figure 4 : Effectiveness of C2F-DO. (a) Directly applying standard DMD to multi-teacher distillation*


![Figure 5 : Evaluation of subject consistency metrics. While DINO often assigns high scores to failed](https://arxiv.org/html/2605.25378/2605.25378v1/x5.png)
*图：Figure 5 : Evaluation of subject consistency metrics. While DINO often assigns high scores to failed*


![Figure 6 : Qualitative comparison of CollectionLoRA against baseline methods. The visual results ind](https://arxiv.org/html/2605.25378/2605.25378v1/x6.png)
*图：Figure 6 : Qualitative comparison of CollectionLoRA against baseline methods. The visual results ind*


![Figure 7 : Zero-shot effect composition capability of CollectionLoRA. Given two independently learne](https://arxiv.org/html/2605.25378/2605.25378v1/x7.png)
*图：Figure 7 : Zero-shot effect composition capability of CollectionLoRA. Given two independently learne*


![Figure 8 : Qualitative ablation study. The progressive integration of our core components systematic](https://arxiv.org/html/2605.25378/2605.25378v1/x8.png)
*图：Figure 8 : Qualitative ablation study. The progressive integration of our core components systematic*


![Figure 9 : Qualitative ablation of training dynamics. Integrating TA-FM and TS significantly acceler](https://arxiv.org/html/2605.25378/2605.25378v1/x9.png)
*图：Figure 9 : Qualitative ablation of training dynamics. Integrating TA-FM and TS significantly acceler*


## 实验结果

Datasets. Our framework utilizes two datasets for training: an effect dataset comprising 50 specific effects (each with 20 animal/portrait image pairs), and a general dataset of 20K source images paired with MLLM-generated instructions, requiring no target images. For evaluation, we introduce EffectBench. Aligned with our training data, it comprises animal and portrait categories. We use Gemini-2.5 Pro and Qwen-Image [ qwenimage ] to generate 100 diverse test images per category, ensuring high variance in subject types, actions, scenes, and camera distances. This yields an evaluation protocol of 5,000 instructions per model.

Baseline Methods. We adopt Qwen-Image-Edit-2509 [ qwenimage ] as the base model and compare our approach against two standard paradigms: (1) Base Model + Effect LoRA, and (2) Base Model + Effect LoRA + Acceleration LoRA. For acceleration, we utilize the popular Qwen-Image-Edit-Lightning LoRA released by lightx2v [ lightx2v ] . To evaluate multi-concept injection capabilities, we also construct a strong baseline denoted as 50-in-1 (FM), which optimizes a unified LoRA on all aggregated training data using a standard flow matching objective.

Evaluation Metrics. We assess generation quality using several metrics: CLIP [ clip ] and DreamSim [ fu2023dreamsimlearningnewdimensions ] for style alignment, DINO [ zhang2022dinodetrimproveddenoising ] for subject consistency, and EditReward [ wu2026editrewardhumanalignedrewardmodel ] for instruction-following and ov


![(a) DreamSim Distance](https://arxiv.org/html/2605.25378/2605.25378v1/x10.png)
*图：(a) DreamSim Distance*


![(b) CLIP Score](https://arxiv.org/html/2605.25378/2605.25378v1/x11.png)
*图：(b) CLIP Score*


![Figure 13 : Comparison between (a) Backward Simulation and (b) our proposed Target Simulation. In th](https://arxiv.org/html/2605.25378/2605.25378v1/x12.png)
*图：Figure 13 : Comparison between (a) Backward Simulation and (b) our proposed Target Simulation. In th*


![Figure 14 : Comparison of simulation strategies. (a) Backward simulation leads to vanishing gradient](https://arxiv.org/html/2605.25378/2605.25378v1/x13.png)
*图：Figure 14 : Comparison of simulation strategies. (a) Backward simulation leads to vanishing gradient*


![Figure 16 : Detailed user study results. Evaluators were asked to choose the best result among four ](https://arxiv.org/html/2605.25378/2605.25378v1/x14.png)
*图：Figure 16 : Detailed user study results. Evaluators were asked to choose the best result among four *


![Figure 17 : Qualitative Evaluation.](https://arxiv.org/html/2605.25378/2605.25378v1/x15.png)
*图：Figure 17 : Qualitative Evaluation.*


![Figure 18 : Qualitative Evaluation.](https://arxiv.org/html/2605.25378/2605.25378v1/x16.png)
*图：Figure 18 : Qualitative Evaluation.*


## 全文图示


![Figure 19 : Qualitative Evaluation.](https://arxiv.org/html/2605.25378/2605.25378v1/x17.png)
*图：Figure 19 : Qualitative Evaluation.*


![Figure 20 : Qualitative Evaluation.](https://arxiv.org/html/2605.25378/2605.25378v1/x18.png)
*图：Figure 20 : Qualitative Evaluation.*


![Figure 21 : Qualitative Evaluation.](https://arxiv.org/html/2605.25378/2605.25378v1/x19.png)
*图：Figure 21 : Qualitative Evaluation.*


![Figure 22 : Qualitative Evaluation.](https://arxiv.org/html/2605.25378/2605.25378v1/x20.png)
*图：Figure 22 : Qualitative Evaluation.*


## 总结

We propose CollectionLoRA, a unified multi-teacher distillation framework that integrates diverse customized visual effects and few-step inference into a single module, eliminating the storage overhead and concept interference (e.g., semantic drift) inherent in traditional multi-LoRA deployments. To address training instability in few-shot multi-concept distillation, our framework features three components: Probabilistic Dual-Stream Routing (PDSR) for structural regularization, Asymmetric Orthogonal Prompting (AOP) for latent concept isolation, and a Coarse-to-Fine Distillation Objective (C2F-DO) to stabilize optimization and restore high-frequency details. Extensive experiments demonstrate that CollectionLoRA achieves superior concept fidelity, feature isolation, and high-quality generation surpassing single-task teachers, all while maintaining few-step generation capabilities.

