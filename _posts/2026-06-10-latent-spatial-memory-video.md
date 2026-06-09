---
layout: post
title: "视频世界模型的潜在空间记忆机制"
date: 2026-06-10
categories: [论文解读, 视频生成]
tags: [视频世界模型, 3D空间记忆, 潜在空间, 视频生成, 场景一致性]
---

> 📄 **论文**：
> 🔗 **arXiv**：[2606.09828](https://arxiv.org/abs/2606.09828)
> 🏢 **机构**：论文作者机构

## 一句话总结

提出潜在空间记忆机制，在潜在特征空间（而非RGB空间）维护持久3D场景缓存，提升视频世界模型的空间一致性，同时降低计算开销。

## 背景与问题

Abstract. Video world models that maintain 3D spatial consistency across generated frames typically rely on explicit point cloud memory constructed in RGB space. This design is both computationally expensive, requiring repeated rendering and VAE encoding, and inherently lossy, as the round trip through pixel space discards rich features of the learned latent representation. In this paper, we introduce latent spatial memory for video world models, a persistent 3D cache that stores scene informati


![Figure 2: Latent spatial memory vs. RGB point cloud based memory for world model](https://arxiv.org/html/2606.09828/2606.09828v1/x3.png)
*图：Figure 2: Latent spatial memory vs. RGB point cloud based memory for world model. Top: prior systems*

Large-scale video diffusion models [ 36 , 43 , 31 , 56 , 2 ] have demonstrated remarkable ability to synthesize photorealistic sequences, motivating their use as world simulators that internalize visual dynamics and generate plausible future observations conditioned on camera trajectories or actions [ 3 , 37 , 42 , 1 , 5 ] . A central challenge in this paradigm is maintaining 3D spatial consistency: without explicit spatial memory, even powerful generators accumulate geometric drift, producing f

## 核心方法

Mirage maintains a persistent latent cache and generates videos by first initializing memory and then repeating a readout-update cycle over overlapping chunks, as illustrated in Figure 3 . Initialization: the initial frame is encoded by encoder and lifted into world space via depth-guided back-projection, seeding with one latent-attributed 3D point per latent cell (Section 4.2 ). Readout and denoising: For each chunk, is projected onto the target camera grids at latent resolution, producing target-view latent feature tensors. These tensors are injected into the diffusion backbone through a Con


![Figure 3: Overview of Mirage. Mirage initializes a 3D latent cache from by encod](https://arxiv.org/html/2606.09828/2606.09828v1/x4.png)
*图：Figure 3: Overview of Mirage. Mirage initializes a 3D latent cache from by encoding it into VAE late*


![Figure 2: Latent spatial memory vs. RGB point cloud based memory for world model. Top: prior systems](https://arxiv.org/html/2606.09828/2606.09828v1/x3.png)
*图1：Figure 2: Latent spatial memory vs. RGB point cloud based memory for world model. Top: prior systems*

![Figure 3: Overview of Mirage. Mirage initializes a 3D latent cache from by encoding it into VAE late](https://arxiv.org/html/2606.09828/2606.09828v1/x4.png)
*图2：Figure 3: Overview of Mirage. Mirage initializes a 3D latent cache from by encoding it into VAE late*

![Figure 4: Open-domain video comparison. Generations on out-of-domain prompts spanning outdoor and na](https://arxiv.org/html/2606.09828/2606.09828v1/x5.png)
*图3：Figure 4: Open-domain video comparison. Generations on out-of-domain prompts spanning outdoor and na*

![Figure 5: Efficiency scaling with rollout progress. Per-frame cache-read time ( left ) and peak cach](https://arxiv.org/html/2606.09828/2606.09828v1/x6.png)
*图4：Figure 5: Efficiency scaling with rollout progress. Per-frame cache-read time ( left ) and peak cach*

![Figure 6: Video comparison on RealEstate10K. Each block shows one RealEstate10K trajectory, with row](https://arxiv.org/html/2606.09828/2606.09828v1/x7.png)
*图5：Figure 6: Video comparison on RealEstate10K. Each block shows one RealEstate10K trajectory, with row*

![Figure 7: Closed-loop revisit comparison on RealEstate10K. In the closed-loop test, the camera traje](https://arxiv.org/html/2606.09828/2606.09828v1/x8.png)
*图6：Figure 7: Closed-loop revisit comparison on RealEstate10K. In the closed-loop test, the camera traje*


## 实验结果

This work was supported by computing resources from Microsoft.


![Figure 4: Open-domain video comparison. Generations on out-of-domain prompts spa](https://arxiv.org/html/2606.09828/2606.09828v1/x5.png)
*图：Figure 4: Open-domain video comparison. Generations on out-of-domain prompts spanning outdoor and na*


## 总结

 提出了一个新颖的研究框架，针对视频生成领域的核心挑战提供了系统性解决方案。

**主要贡献：**
- 提出潜在空间记忆机制，在潜在特征空间（而非RGB空间）维护持久3D场景缓存，提升视频世界模型的空间一致性，同时降低计算开销。
- 通过系统实验验证了方法的有效性
- 为后续研究提供了重要的基准和参考

**局限性与展望：** 未来工作可进一步探索方法在更广泛场景下的应用，以及结合更多领域知识提升系统性能。