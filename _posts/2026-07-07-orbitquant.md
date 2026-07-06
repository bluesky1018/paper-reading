---
layout: post
title: "OrbitQuant：面向图像与视频扩散Transformer的数据无关量化方法"
date: 2026-07-07
categories: [论文解读, 模型压缩]
tags: [量化, 扩散模型, DiT, 数据无关, FLUX, 视频生成]
---

> 📄 **论文**：OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers
> 🔗 **arXiv**：[2607.02461](https://arxiv.org/abs/2607.02461)
> 🏢 **机构**：Cantina Labs（Donghyun Lee、Jitesh Chavan 等8位作者）

## 一句话总结

OrbitQuant 通过随机置换块Hadamard（RPBH）旋转，将扩散Transformer的权重和激活映射到一个稳定的归一化分布空间，从而实现无需任何校准数据的训练后量化（PTQ），在FLUX.1等主流模型上达到最优量化效果。

## 背景与问题

扩散Transformer（DiTs）因其强大的生成能力已成为图像和视频生成的主流架构，但高昂的推理成本限制了实际部署。训练后量化（PTQ）是无需重训练即可压缩模型的有效手段，然而现有方法（如AdaTSQ、ViDiT-Q）存在一个严重缺陷：**需要为每个新检查点重新校准数据**——这意味着每次模型更新都需要收集代表性样本并重新运行校准流程，代价高昂。

此外，扩散模型的激活值在不同时间步、不同提示词和不同层之间分布差异极大，难以用单一统计量描述，使得现有校准方案十分脆弱。

## 核心方法

OrbitQuant 的核心思想是：**在旋转后的空间中量化，使分布变得稳定且可预测，从而彻底消除对校准数据的依赖。**

![OrbitQuant框架总览](https://arxiv.org/html/2607.02461v1/x2.png)
*图：OrbitQuant方法概览——RPBH旋转使激活分布归一化，单一Lloyd-Max码本适用所有时间步和层*

### RPBH旋转（随机置换块Hadamard）

$$\Pi_d = \text{blkdiag}(\mathbf{H}_h\mathbf{D}_1,\ldots,\mathbf{H}_h\mathbf{D}_{d/h})\cdot\mathbf{P}_\pi$$

- **块Hadamard矩阵** $\mathbf{H}_h$：在每个子块内做Hadamard变换，高效混合局部信息
- **Rademacher符号对角阵** $\mathbf{D}_i$：随机翻转符号，打散异常值
- **均匀随机置换** $\mathbf{P}_\pi$：将异常值分散到所有块，使每个坐标的边际分布收敛到 $\mathcal{N}(0,1/d)$

计算复杂度仅 $O(d\log h)$，而稠密Haar旋转需要 $O(d^2)$，RPBH快约**26倍**。

![旋转前后激活分布对比](https://arxiv.org/html/2607.02461v1/x3.png)
*图：RPBH旋转前（左）后（右）的激活分布——旋转后分布稳定收敛为高斯，消除了异常值*

### 数据无关量化流程

**离线阶段（一次性）**：
1. 对权重矩阵施加RPBH旋转：$\mathbf{W}' = \mathbf{W}\Pi_d^\top$
2. 分离方向与幅值，仅量化方向向量
3. 对固定分布 $f_d \approx \mathcal{N}(0,1/d)$ 运行Lloyd-Max算法，得到通用码本

**在线阶段（推理时）**：
1. 对激活施加前向RPBH旋转：$\mathbf{x}' = \Pi_d\mathbf{x}$
2. 使用离线码本量化旋转后的激活
3. 旋转在矩阵乘积中自动消除：$\mathbf{W}'\mathbf{x}' = \mathbf{W}\mathbf{x}$

由于分布稳定性有理论保证，**同一码本可服务所有时间步、所有提示词、所有层**，无需任何数据校准。

## 实验结果

### 图像生成（GenEval基准，FLUX.1系列）

| 方法 | FLUX.1-schnell W4A4 | FLUX.1-schnell W2A4 | Z-Image-Turbo W4A4 |
|------|--------------------|--------------------|-------------------|
| FP16基线 | 0.664 | — | 0.754 |
| AdaTSQ | 0.680 | — | 0.762 |
| QuaRot† | — | 0.001 | — |
| SVDQuant | 0.624 | — | — |
| **OrbitQuant** | **0.703** | **0.604** | **0.767** |

在极低比特W2A4下，现有方法（QuaRot、SmoothQuant、ViDiT-Q）的GenEval评分几乎为零，而OrbitQuant达到0.604，展现出对超低比特量化的卓越鲁棒性。

### 视频生成（VBench基准）

| 方法 | Wan 2.1-1.3B W4A6 | CogVideoX-2B W4A4 |
|------|------------------|------------------|
| 全精度 | 24.67% | 25.06% |
| SVDQuant | 23.26% | 22.89% |
| **OrbitQuant** | **24.35%** | **23.86%** |

### 效率对比（FLUX.1-dev，H100 GPU）

相比OrbitQuant，其他方法的额外延迟开销：SmoothQuant +9%，QuaRot +28%，ViDiT-Q +40%。

![定性效果对比](https://arxiv.org/html/2607.02461v1/x4.png)
*图：W4A4量化下的视觉质量对比——OrbitQuant生成质量明显优于竞品*

![延迟与内存对比](https://arxiv.org/html/2607.02461v1/x5.png)
*图：延迟与内存占用综合对比——OrbitQuant在速度和内存方面均具优势*

### 旋转矩阵消融（FLUX.1-schnell）

| 旋转类型 | W4A4 | W2A4 | 旋转耗时(s) |
|---------|------|------|----------|
| Haar（随机正交） | 0.696 | 0.591 | 11.65 |
| 全RHT | 0.691 | 0.587 | 0.452 |
| **RPBH（本文）** | **0.690** | **0.595** | 0.451 |

RPBH在低比特下性能最优，同时保持与高效RHT相当的计算速度。

## 总结

OrbitQuant 通过数学上严格的分布稳定性分析，将扩散Transformer量化问题转化为一个与数据无关的纯优化问题，这在量化方法设计上是一个重要的范式转变。其在W2A4超低比特场景下的压倒性优势（竞品近乎失效而本文方法仍保持0.604）尤为突出。

局限性方面：RPBH旋转在推理时仍有一定计算开销；AdaLN调制投影由于无法吸收旋转而需要单独处理；在极端低比特场景下，幅值量化策略可能成为下一个瓶颈。
