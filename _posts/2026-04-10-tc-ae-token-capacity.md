---
layout: post
title: "TC-AE：从Token空间视角解锁深度压缩自动编码器的生成潜力"
date: 2026-04-10
categories: [论文解读, 图像生成]
tags: [图像生成, 自动编码器, Token压缩, ViT, 扩散模型, 自监督学习]
---

> 📄 **论文**：TC-AE: Unlocking Token Capacity for Deep Compression Autoencoders
> 🔗 **arXiv**：[2604.07340](https://arxiv.org/abs/2604.07340)
> 🏢 **机构**：Inclusion AI (Ant Group), HKUST, ECNU, ZJU

## 一句话总结

TC-AE 从"Token 空间"视角重新审视深度压缩自动编码器的设计，通过分阶段 Token 压缩（STC）和联合自监督训练（iBOT），在仅64个Token和极低 FLOPs 下实现了 gFID 2.57 的顶尖生成质量，同时加速 DiT 收敛 **4.7×**。

## 背景与问题

潜在扩散模型（如 Stable Diffusion、FLUX）依赖 tokenizer（自动编码器）将图像压缩为低维潜变量。为追求更高的压缩率，现有深度压缩方法（如 DC-AE）通过增大潜变量通道数来弥补空间信息损失，但这会导致**潜变量表示崩溃**（Latent Representation Collapse）。

在 ViT-based tokenizer 中，图像先被分成 patch 得到 Token，再经过 Transformer 块压缩为潜变量。本文的核心发现是：

**问题根源不是通道数不够，而是 Token→潜变量的瓶颈压缩破坏了结构信息。**

实验证明：在固定潜变量容量下，增加 Token 数（减小 patch size）虽然改善重建质量，但**不改善甚至恶化生成质量**——patch size=8 时，Token 的语义质量（A₁=62.9）远高于 patch size=32，但经过瓶颈后潜变量几乎完全失去语义（A₂=5.33，结构损失高达 0.92）。

## 核心方法

### 分阶段Token压缩（STC）

**洞察**：将激进的 Token→潜变量压缩分两个阶段进行，让每步压缩更"温和"，从而保留更多结构信息。

Encoder 架构改变：
1. 小 patch size → M 个 ViT 块（高分辨率 Token）
2. **2层卷积中间模块（4× Token 压缩）**
3. N 个 ViT 块 → pixel-shuffle + MLP 最终瓶颈 → 潜变量 z

![TC-AE架构图](https://arxiv.org/html/2604.07340/x4.png)
*图：TC-AE 完整架构，展示分阶段压缩和 SSL 训练*

通过线性探测精度衡量结构损失：
- patch=8 无 STC：A₁=62.9 → A₂=5.33（结构损失 **0.92**）
- patch=8 有 STC：A₁=39.1 → A₂=12.1（结构损失 **0.69**）

gFID 从 25.36 → **16.39**（降低 35%）。

![潜变量结构崩溃可视化](https://arxiv.org/html/2604.07340/x3.png)
*图：潜变量结构崩溃可视化——小 patch size（p=8）尽管 Token 语义更强，但潜变量崩溃更严重*

### 联合自监督Token结构化（iBOT）

**洞察**：结合教师-学生蒸馏的 iBOT 框架，通过无需外部预训练模型的自监督目标，让 Token 空间具有更好的语义结构。

训练目标：
$$\mathcal{L}_{\text{TC-AE}} = \alpha \cdot \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{iBOT}}$$

其中 ℒ_iBOT 包含：
- **ℒ_MIM**：掩码图像建模（patch-token 预测）
- **ℒ_[CLS]**：全局语义对齐（class-token 蒸馏）

三种 SSL 方法比较（patch=16）：

| 方法 | gFID↓ | IS↑ |
|------|-------|-----|
| DINO | 20.38 | 62.12 |
| DINOv2 | 28.22 | 49.19 |
| **iBOT** | **17.22** | **69.00** |

## 实验结果

### STC + SSL 协同效果

| STC | SSL | rFID↓ | gFID↓ | IS↑ |
|-----|-----|-------|-------|-----|
| ✗ | ✗ | 1.33 | 44.72 | 33.62 |
| ✓ | ✗ | 0.75 | 32.92 | 43.58 |
| ✗ | ✓ | 0.90 | 25.36 | 52.38 |
| **✓** | **✓** | **0.90** | **16.39** | **71.33** |

![STC加速扩散收敛曲线](https://arxiv.org/html/2604.07340/x5.png)
*图：收敛曲线——STC 单独使 2.7×，完整 TC-AE 使 4.7× 更快的 DiT 收敛*

### 系统级对比（ImageNet 256×256）

| Tokenizer | Token数 | GFLOPs | gFID(无CFG)↓ | gFID(有CFG)↓ |
|-----------|--------|--------|------------|------------|
| DC-AE | 64 | 607 | 17.31 | — |
| ViTok S-B | 256 | — | — | 2.45 |
| MAETok | 256 | — | 2.21 | 1.73 |
| **TC-AE** | **64** | **164** | **7.16** | **2.57** |

TC-AE 仅需 **164 GFLOPs**（DC-AE 的 27%），生成质量大幅优于所有64-Token方案，与256-Token方案持平。

### Token 数扩展 vs 参数扩展

- 两种扩展方式都改善生成质量
- 相同 GFLOPs 预算下，**Token 数扩展收益更大**
- 两者互补，联合扩展效果最佳

![Token扩展与参数扩展协同](https://arxiv.org/html/2604.07340/x7.png)
*图：Token 数量扩展与参数扩展的协同效果对比*

## 总结

TC-AE 提出了一个简洁但深刻的观点：深度压缩 tokenizer 的生成质量瓶颈在于 Token→潜变量的信息损失，而非模型容量不足。分阶段压缩和 iBOT 自监督协同解决了这一问题，使得低 FLOPs 的 64-Token 方案能够达到与 256-Token 方案相当的生成质量。

值得注意的是，反直觉地，增加潜变量通道数（从128到256）实际上会**恶化**生成质量，进一步印证了问题根源在于表示结构的崩溃而非容量不足。未来可探索将 TC-AE 与更大规模扩散模型（如 Flux）结合，以及在视频生成领域的应用。
