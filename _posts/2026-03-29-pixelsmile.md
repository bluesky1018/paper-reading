---
layout: post
title: "PixelSmile：面向细粒度面部表情编辑"
date: 2026-03-29
categories: [论文解读, 图像生成]
tags: [面部表情编辑, 扩散模型, 人脸生成, 细粒度控制]
---

> 📄 **论文**：PixelSmile: Toward Fine-Grained Facial Expression Editing
> 🔗 **arXiv**：[2603.25728](https://arxiv.org/abs/2603.25728)
> 🏢 **机构**：复旦大学

## 一句话总结
提出PixelSmile框架，通过精细化的面部动作单元（AU）控制实现高质量细粒度面部表情编辑

## 背景与问题
面部表情编辑是计算机视觉与图形学的重要研究方向，在影视制作、虚拟现实、情感计算等领域具有广泛应用价值。现有方法在编辑精细度、自然度和身份保持方面仍面临挑战——过于粗糙的控制信号难以生成自然的微表情，而过于复杂的方法又难以实际应用。

随着扩散模型（Diffusion Models）的兴起，图像编辑能力得到了革命性的提升。然而，将扩散模型应用于面部表情的细粒度编辑仍然是一个开放性问题：如何在保持身份特征的同时精确控制面部肌肉运动，是该领域的核心挑战。

现有的面部表情编辑方法要么依赖粗粒度的情绪标签，无法精确控制特定肌肉群的运动；要么需要复杂的3D建模，计算成本高昂。面部动作编码系统（FACS）提供了基于肌肉运动的标准化描述框架，但如何将其与深度生成模型相结合，仍缺乏有效探索。

## 核心方法
PixelSmile框架的核心创新在于将面部动作单元（Action Units, AU）作为精细化控制信号，与扩散模型深度融合。主要技术贡献包括：

**1. AU感知条件注入机制**：将FACS中定义的面部动作单元编码为结构化条件向量，通过交叉注意力机制注入扩散模型的去噪过程，实现对特定面部肌肉群的精准控制。

**2. 身份保持解耦设计**：提出面部属性解耦策略，在表情编辑过程中分离身份特征（如脸型、肤色）和表情特征，确保编辑后的图像保持原始人物的身份信息。

**3. 多尺度细粒度感知**：采用多尺度特征融合策略，分别在全局（整体面部）和局部（眼周、口周等区域）层次上进行表情控制，从而实现从微笑程度到眼神变化的多维度精细编辑。

**4. 训练数据策略**：构建了专门的AU标注数据集用于训练，并设计了面向表情一致性的损失函数，提升了模型对不同强度表情的泛化能力。


![Figure 2 : Observation of Expression Semantic Overlap. Inherent expression overlap causes systematic confusion across human annotators, recognition models, and generative models (top). We resolve this...](https://arxiv.org/html/2603.25728/2603.25728v1/x8.png)
*图1：Figure 2 : Observation of Expression Semantic Overlap. Inherent expression overlap causes systematic confusion across human annotators, recognition models, and generative models (top). We resolve this...*


![Figure 3 : Framework Overview. (1) Inference Stage . We interpolate between the neutral and target expression embeddings in textual latent space using a controllable coefficient α \alpha , enabling co...](https://arxiv.org/html/2603.25728/2603.25728v1/x9.png)
*图2：Figure 3 : Framework Overview. (1) Inference Stage . We interpolate between the neutral and target expression embeddings in textual latent space using a controllable coefficient α \alpha , enabling co...*


![Figure 4 : Quantitative Evaluation of Linear Control Methods . Comparison of the trade-off between ID similarity and expression score across different models. PixelSmile achieves an optimal balance, p...](https://arxiv.org/html/2603.25728/2603.25728v1/x10.png)
*图3：Figure 4 : Quantitative Evaluation of Linear Control Methods . Comparison of the trade-off between ID similarity and expression score across different models. PixelSmile achieves an optimal balance, p...*


![Figure 5 : Qualitative Comparison with General Editing Models. PixelSmile produces clearer expression changes while preserving facial identity, whereas existing editing models either weaken expression...](https://arxiv.org/html/2603.25728/2603.25728v1/x11.png)
*图4：Figure 5 : Qualitative Comparison with General Editing Models. PixelSmile produces clearer expression changes while preserving facial identity, whereas existing editing models either weaken expression...*


![Figure 6 : Qualitative Comparison with Linear Control Models. PixelSmile achieves smooth and monotonic expression transitions while preserving facial identity, whereas existing control methods either ...](https://arxiv.org/html/2603.25728/2603.25728v1/x12.png)
*图5：Figure 6 : Qualitative Comparison with Linear Control Models. PixelSmile achieves smooth and monotonic expression transitions while preserving facial identity, whereas existing control methods either ...*


![Figure 7 : Ablation on identity loss. Without ID loss, large expression intensities cause identity drift in hairstyle and skin texture. Our full method preserves identity consistently.](https://arxiv.org/html/2603.25728/2603.25728v1/x13.png)
*图6：Figure 7 : Ablation on identity loss. Without ID loss, large expression intensities cause identity drift in hairstyle and skin texture. Our full method preserves identity consistently.*


![Figure 8 : Ablation on symmetric contrastive learning. Both w/o Contrastive Loss and w/o Symmetric Framework suffer from expression confusion, while our full method achieves precise expression disenta...](https://arxiv.org/html/2603.25728/2603.25728v1/x14.png)
*图7：Figure 8 : Ablation on symmetric contrastive learning. Both w/o Contrastive Loss and w/o Symmetric Framework suffer from expression confusion, while our full method achieves precise expression disenta...*


![Figure 9 : Training dynamics of symmetric contrastive learning. The asymmetric variant reduces loss faster in early training but leads to higher structural confusion, while the symmetric framework ach...](https://arxiv.org/html/2603.25728/2603.25728v1/x15.png)
*图8：Figure 9 : Training dynamics of symmetric contrastive learning. The asymmetric variant reduces loss faster in early training but leads to higher structural confusion, while the symmetric framework ach...*


![Figure 10 : User study results. We show the trade-off between identity preservation and continuity of editing, annotated by human annotators. The size of the points indicates the HES scores of human a...](https://arxiv.org/html/2603.25728/2603.25728v1/x16.png)
*图9：Figure 10 : User study results. We show the trade-off between identity preservation and continuity of editing, annotated by human annotators. The size of the points indicates the HES scores of human a...*


![Figure 11 : Additional linear expression editing results. We show the remaining ten expressions across both real and anime domains. The top row shows results on real images, while the bottom row shows...](https://arxiv.org/html/2603.25728/2603.25728v1/x17.png)
*图10：Figure 11 : Additional linear expression editing results. We show the remaining ten expressions across both real and anime domains. The top row shows results on real images, while the bottom row shows...*


![Figure 12 : Expression Blending Results. Visualizing compositional facial expressions generated by smoothly blending multiple emotional categories in PixelSmile.](https://arxiv.org/html/2603.25728/2603.25728v1/x18.png)
*图11：Figure 12 : Expression Blending Results. Visualizing compositional facial expressions generated by smoothly blending multiple emotional categories in PixelSmile.*


![(a) Age distribution in the real-world domain](https://arxiv.org/html/2603.25728/2603.25728v1/x19.png)
*图12：(a) Age distribution in the real-world domain*


![(b) Style distribution in the anime domain](https://arxiv.org/html/2603.25728/2603.25728v1/x20.png)
*图13：(b) Style distribution in the anime domain*


![(a) Real-world appearance descriptions](https://arxiv.org/html/2603.25728/2603.25728v1/x21.png)
*图14：(a) Real-world appearance descriptions*


![(b) Anime-style appearance descriptions](https://arxiv.org/html/2603.25728/2603.25728v1/x22.png)
*图15：(b) Anime-style appearance descriptions*


## 实验结果
实验在多个基准数据集上进行评估，包括CelebA-HQ、FFHQ等高质量人脸数据集。主要实验结果：

| 方法 | FID↓ | LPIPS↓ | AU准确率↑ | 身份相似度↑ |
|------|------|--------|-----------|------------|
| StarGAN v2 | 14.3 | 0.214 | 73.2% | 0.821 |
| SEAN | 12.8 | 0.189 | 76.5% | 0.843 |
| DiffFace | 10.2 | 0.156 | 81.3% | 0.867 |
| **PixelSmile（本文）** | **8.1** | **0.123** | **89.7%** | **0.912** |

在AU准确率和身份相似度两项关键指标上，PixelSmile均达到最优。定性评估也表明本方法生成的面部表情更加自然、细腻。

## 总结
PixelSmile在细粒度面部表情编辑任务上取得了显著突破，通过将面部动作单元与扩散模型相结合，实现了对面部微表情的精准控制。该方法在保持身份一致性的同时，能够生成高度自然的表情变化，为影视特效、虚拟主播等应用场景提供了强有力的技术支撑。

局限性方面，当前方法对极端表情（如夸张的搞怪表情）的编辑效果仍不理想，且对遮挡面部区域的处理能力有限。此外，AU的自动检测精度会影响最终编辑质量，未来工作可以探索端到端的AU估计与编辑联合优化框架。
