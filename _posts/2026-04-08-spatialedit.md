---
layout: post
title: "SpatialEdit：细粒度图像空间编辑基准测试"
date: 2026-04-08
categories: [论文解读, 图像生成与编辑]
tags: [图像编辑, 空间理解, Benchmark, 扩散模型, Text-to-Image]
---

> 📄 **论文**：SpatialEdit: Benchmarking Fine-Grained Image Spatial Editing
> 🔗 **arXiv**：[2604.04911](https://arxiv.org/abs/2604.04911)
> 🏢 **机构**：JD Open Source（京东开源）
> 👥 **作者**：Yicheng Xiao, Wenhu Zhang, Lin Song, Yukang Chen, Wenbo Li, Nan Jiang, Tianhe Ren, Haokun Lin, Wei Huang, Haoyang Huang, Xiu Li, Nan Duan, Xiaojuan Qi

## 一句话总结
构建SpatialEdit基准，专注于评估图像生成模型在细粒度空间关系编辑任务上的能力，揭示当前模型的空间推理局限

## 背景与问题

Image spatial editing performs geometry-driven transformations, allowing precise control over object layout and camera viewpoints. Current models are insufficient for fine-grained spatial manipulations, motivating a dedicated assessment suite. Our contributions are listed: (i) We introduce SpatialEdit-Bench, a complete benchmark that evaluates spatial editing by jointly measuring perceptual plausibility and geometric fidelity via viewpoint reconstruction and framing analysis. (ii) To address the data bottleneck for scalable training, we construct SpatialEdit-500k, a synthetic dataset generated with a controllable Blender pipeline that renders objects across diverse backgrounds and systematic camera trajectories, providing precise ground-truth transformations for both object- and camera-centric operations. (iii) Building on this data, we develop SpatialEdit-16B, a baseline model for fine-grained spatial editing. Our method achieves competitive performance on general editing while substantially outperforming prior methods on spatial manipulation tasks. All resources will be made public at this https URL.



## 核心方法

详见原文方法章节。


![Figure 1: Illustration for image spatial editing. It comprises two components: (](https://arxiv.org/html/2604.04911/2604.04911v1/x1.png)
*图：Figure 1: Illustration for image spatial editing. It comprises two components: (1) camera-centric view manipulation, including pitch, yaw, and zoom tr*


![Figure 2: Statistics of SpatialEdit-500k. (a) Distribution of camera-level data ](https://arxiv.org/html/2604.04911/2604.04911v1/x2.png)
*图：Figure 2: Statistics of SpatialEdit-500k. (a) Distribution of camera-level data across seven sub-tasks in outdoor and intdoor scenes, where Y, P, and *


![Figure 3: SpatialEdit-500k data generation pipeline. We leverage Blender to synt](https://arxiv.org/html/2604.04911/2604.04911v1/x3.png)
*图：Figure 3: SpatialEdit-500k data generation pipeline. We leverage Blender to synthesize both objects and scenes, while preprocessing 3D assets using SA*


![Figure 4: Overview of SpatialEdit.](https://arxiv.org/html/2604.04911/2604.04911v1/x4.png)
*图：Figure 4: Overview of SpatialEdit.*


## 实验结果

详见原文实验章节。


![Figure 5: Comparison of camera view manipulation across various methods.](https://arxiv.org/html/2604.04911/2604.04911v1/x5.png)
*图：Figure 5: Comparison of camera view manipulation across various methods.*


![Figure 6: Comparison of object-level manipulation across various methods.](https://arxiv.org/html/2604.04911/2604.04911v1/x6.png)
*图：Figure 6: Comparison of object-level manipulation across various methods.*


![Figure 7: Serving as an enhancement tool for single-view reconstruction.](https://arxiv.org/html/2604.04911/2604.04911v1/x7.png)
*图：Figure 7: Serving as an enhancement tool for single-view reconstruction.*


![Figure 8: Comparison of object-level manipulation across various methods.](https://arxiv.org/html/2604.04911/2604.04911v1/x8.png)
*图：Figure 8: Comparison of object-level manipulation across various methods.*


![Figure 9: Comparison of camera-level manipulation across various methods.](https://arxiv.org/html/2604.04911/2604.04911v1/x9.png)
*图：Figure 9: Comparison of camera-level manipulation across various methods.*


## 总结

本文提出了 **SpatialEdit**，构建SpatialEdit基准，专注于评估图像生成模型在细粒度空间关系编辑任务上的能力，揭示当前模型的空间推理局限。该工作从理论和实践层面均有创新，为后续研究提供了重要参考。

**局限性与未来方向：** 如所有工作一样，该研究仍有一定局限性，后续可在更大规模数据集、更多样化场景下进行验证和拓展。
