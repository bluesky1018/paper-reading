---
layout: post
title: "Flux-GS：蒙特卡洛能量聚合实现移动端高保真 3D 高斯泼溅"
date: 2026-07-01
categories: [论文解读, 三维视觉]
tags: [3D Gaussian Splatting, Mobile Rendering, Novel View Synthesis, Real-Time, Spherical Harmonics]
---

> 📄 **论文**：Monte Carlo Energy Aggregation for Mobile 3D Gaussian Splatting
> 🔗 **arXiv**：[2606.30017](https://arxiv.org/abs/2606.30017)
> 🏢 **机构**：Xiaobiao Du, YuAn Wang, Hao Li, Bosheng Wang, Xun Sun, Xin Yu（University of Adelaide）

## 一句话总结

Flux-GS 通过蒙特卡洛镜面能量聚合、属性条件化 SH 增强和多视角 Alpha 致密化策略，在显著减少移动端 Gaussian 数量和存储开销的同时保持了高视觉保真度。

## 背景与问题

3D Gaussian Splatting（3DGS）在新视角合成领域取得了突破性成功，但在移动平台部署时面临两大瓶颈：

1. **高阶球谐函数（SH）开销巨大**：3DGS 使用三阶 SH（48个系数/Gaussian）表示视角相关颜色，在移动端 GPU 上造成极大的推理和存储压力
2. **冗余 Gaussian 过多**：单视角梯度致密化策略倾向于对特定视角过拟合，产生大量冗余 Gaussian

现有的移动端优化方案（如 Mobile-GS）通过压缩 SH 阶数来减小模型体积，但往往牺牲了高频镜面光照等视觉细节。

## 核心方法

### 蒙特卡洛镜面能量聚合器（Monte Carlo Specular Energy Aggregator）

**核心思想**：对三阶辐射残差进行蒙特卡洛采样，将镜面能量聚合到紧凑的隐空间中：

1. 在推理时对高阶 SH 系数进行蒙特卡洛采样估计
2. 将采样的镜面能量投影到低阶隐表示中
3. 无需昂贵的预训练或知识蒸馏，直接在低阶频带中保留视觉显著的光照特征

这一设计的关键优势是在不改变推理架构的前提下，用统计估计替代完整的高阶计算。

![Flux-GS 方法对比](https://arxiv.org/html/2606.30017v1/x1.png)
*图1：Flux-GS 与 3DGS、Mobile-GS 的 FPS 和视觉质量对比*

### 属性条件化 SH 增强（Attribute-Conditioned SH Enhancement）

为弥补压缩过程中损失的高频细节：

- 基于 Gaussian 内在属性（位置、不透明度、颜色等）预测 Gaussian 感知偏移
- 在推理前将偏移叠加到一阶 SH 表示上，增强其高频表达能力
- **关键优势**：增强在推理前完成，不引入额外的推理计算开销

### 多视角 Alpha 致密化与剪枝（Multi-view Alpha-based Densification）

解决单视角致密化导致的过拟合和冗余问题：

- 利用多视角监督确保 Gaussian 结构的多视角一致性
- Alpha 值（不透明度）作为致密化判据，准确识别缺失细节区域
- 精确剪除低透明度的冗余 Gaussian，而非依赖简单的梯度阈值

![Gaussian 参数分布与 SH 保真度分析](https://arxiv.org/html/2606.30017v1/x2.png)
*图2：各 3DGS 变体的 Gaussian 参数分布和球谐保真度对比*

![Flux-GS 方法架构](https://arxiv.org/html/2606.30017v1/x3.png)
*图3：Flux-GS 完整方法架构，展示三个核心模块的协同工作*

## 实验结果

### 移动端性能对比（Snapdragon 8 Gen 3 GPU）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | 存储(MB)↓ |
|------|-------|-------|--------|------|----------|
| 3DGS | 27.82 | 0.843 | 0.187 | 3.2 | 847 |
| Mobile-GS | 27.14 | 0.829 | 0.213 | 31.7 | 203 |
| Scaffold-GS | 27.96 | 0.851 | 0.172 | 8.4 | 521 |
| **Flux-GS** | **27.71** | **0.841** | **0.191** | **38.4** | **149** |

### 复杂度分析

| 场景 | Gaussian 数量 (M) | 相比 3DGS 减少 |
|------|-----------------|--------------|
| 3DGS | 3.2M | - |
| Mobile-GS | 1.8M | 44% |
| **Flux-GS** | **1.2M** | **63%** |

**关键结论**：
- Flux-GS 将 Gaussian 数量减少 **61-63%**，存储减少约 **82%**
- 在 Snapdragon 8 Gen 3 上实现 **38.4 FPS** 的移动端实时渲染
- 视觉质量仅略低于完整 3DGS，PSNR 差距 < 0.2 dB
- 支持 **WebGL 跨平台渲染**，可在浏览器中无缝运行

![质量-速度权衡](https://arxiv.org/html/2606.30017v1/figs/k_vs_psnr.png)
*图4：不同采样数 K 对 PSNR 和速度的影响*

![复杂度对比](https://arxiv.org/html/2606.30017v1/figs/complexity_vs_camera_count.png)
*图5：随相机视角增加，各方法的计算复杂度对比*

## 总结

Flux-GS 通过三个相辅相成的技术创新（蒙特卡洛能量聚合、属性条件增强、多视角致密化），在移动端实现了 3DGS 渲染的速度-质量-存储三维最优权衡。38.4 FPS 的实时性能和 82% 的存储压缩率，使其成为移动 AR/VR 应用的理想选择。

局限性在于蒙特卡洛采样在极端光照条件下可能引入噪声，且对于高度镜面反射场景仍有提升空间。未来可探索自适应采样策略和更先进的隐空间表示。
