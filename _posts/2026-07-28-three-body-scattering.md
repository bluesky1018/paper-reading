---
layout: post
title: "用于生成建模的三体散射方法"
date: 2026-07-28
categories: [论文解读, 生成模型]
tags: [生成模型, 扩散模型, 一步生成, 能量距离, ImageNet, 西湖大学]
---

> 📄 **论文**：Three-Body Scattering for Generative Modeling
> 🔗 **arXiv**：[2607.18198](https://arxiv.org/abs/2607.18198)
> 🏢 **机构**：西湖大学、浙江大学、伦敦大学学院

## 一句话总结

TBSM（三体散射建模）将能量距离转化为恒定规模的逐样本交互，仅需一步前向推理即可在 ImageNet-256 上实现 FID=1.63 的生成质量，无需扩散过程或对抗训练。

## 背景与问题

现代生成模型通常依赖以下三类机制之一：
1. **对抗性判别器**（GAN）
2. **预设的噪声到数据路径**（扩散/流匹配）
3. **自回归因式分解**（AR 模型）

而直接的**一步分布匹配方法**通常需要从 minibatch 场估计监督信号，计算量大且方差高。

核心问题：**能否用一个恰当的分布能量函数为一步生成器提供恒定规模的逐样本随机交互，无需 Teacher 模型查询？**

## 核心方法

### 三体散射的核心思想

TBSM 将生成的样本视为"入射粒子"（Projectile）：
- **吸引力**：被一个真实样本（Real Source）吸引
- **排斥力**：被一个独立生成的样本（Generated Source）排斥

这三者构成"三体散射"系统。条件期望值等于 $\frac{1}{2}D_E^2(P_\theta, Q)$ 的 2-Wasserstein 梯度流速度。

### 数学形式化

对于一批 $B$ 个冻结目标事件，产生 $O(B)$ 个样本级别的损失，每个仅使用一个参考样本作为其条件，而非 Drifting Models 等方法使用的 minibatch 级全对场。

在线追踪这个条件期望可以降低场方差。

### 设计图谱

论文提供了一个关联扩散类监督、Drift-like 动态和 GAN-like 目标的设计图谱（$\rho$-$\lambda$ 参数空间）：

![生成设计图谱](https://arxiv.org/html/2607.18198v1/resources/figures/tbsm_jit_b_lam=1p0_rho=0p0_fid=10.31_is=133.02.jpg)
*图3a：设计图谱一角——$\rho=0, \lambda=1$（Drift-like，FID=10.31）*

![设计图谱最优点](https://arxiv.org/html/2607.18198v1/resources/figures/tbsm_jit_b_lam=0p9_rho=0p9_fid=0.99_is=328.0_fdr6=5.64.jpg)
*图3c：设计图谱最优点——$\rho=0.9, \lambda=0.9$（FID=0.99，IS=328.0）*

### 一步生成演示

![TBSM一步生成样本](https://arxiv.org/html/2607.18198v1/resources/figures/main_demo.jpg)
*图1：TBSM 训练的模型一步生成的样本。从左到右：MNIST、Fashion-MNIST、CIFAR-10（上到下）；像素空间 ImageNet-256 和潜在空间 ImageNet-256*

## 实验结果

在 ImageNet-256 上的一步生成结果（NFE=1）：

| 方法 | 架构 | FID ↓ |
|---|---|---|
| TBSM | PixelDiT-XL（像素空间） | **2.23** |
| TBSM | DiT-XL（潜在空间） | **1.63** |

![ImageNet-512生成效果](https://arxiv.org/html/2607.18198v1/resources/figures/tbsm_dit_xl_512_lam=0p9_rho=0p9_fid=1.92_is=252.1_fdr6=5.37.jpg)
*512×512 潜在空间生成效果，FID=1.92，IS=252.1*

关键特性：
- **无需多步推理**：一次前向传播即可生成高质量图像
- **无需 Teacher 模型**：不依赖预训练扩散模型提供监督
- **恒定规模交互**：每个样本的计算复杂度为 $O(B)$ 而非 $O(B^2)$

## 总结

TBSM 通过三体物理散射的类比，为一步生成模型提供了理论基础扎实的训练目标。在 ImageNet-256 上 FID=1.63 的结果展示了该方法在高维生成任务上的竞争力。

该方法同时提供了一个统一框架（设计图谱），将扩散类方法、漂移动态和 GAN 目标联系起来，有助于理解不同生成方法的内在关系。代码已开源：https://github.com/sp12138/TBSM。
