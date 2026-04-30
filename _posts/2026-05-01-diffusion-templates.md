---
layout: post
title: "Diffusion Templates：统一的可控扩散插件框架"
date: 2026-05-01
categories: [论文解读, 图像生成]
tags: [扩散模型, 可控生成, 插件框架, KV-Cache, ControlNet, LoRA]
---

> 📄 **论文**：Diffusion Templates: A Unified Plugin Framework for Controllable Diffusion
> 🔗 **arXiv**：[2604.24351](https://arxiv.org/abs/2604.24351)
> 🏢 **机构**：ModelScope / DiffSynth-Studio 团队

## 一句话总结
Diffusion Templates 将可控扩散方法重新框架为可复用的"插件"，通过统一的 Template Cache 接口将异构控制能力（KV-Cache、LoRA 等）与基础扩散模型解耦，构建了包含结构控制、亮度/颜色调整、图像编辑、超分辨率等10+能力的多样化模型动物园。

## 背景与问题

扩散模型已成为视觉生成的主导基础，可控生成需求催生了大量技术（ControlNet、LoRA、IP-Adapter、InstructPix2Pix 等）。然而，这些方法通常作为孤立的、特定于骨干网络的系统开发，带来了严重的系统碎片化问题：

1. **训练基础设施不可复用**：不同控制方法需要不同的模型修改方式、参数化方案、预处理代码和优化目标
2. **部署不可组合**：每个方法暴露自己的运行时钩子和参数格式，集成新控制方法需要修改核心推理代码
3. **跨骨干迁移困难**：将一个能力从一个扩散骨干迁移到另一个需要重新训练

## 核心方法

![Diffusion Templates 框架设计](https://arxiv.org/html/2604.24351v1/x1.png)
*图：Diffusion Templates 框架的三大核心组件：Template Cache、Template Model 和 Template Pipeline*

### 三大核心组件

**Template Model（模板模型）**：
将任意特定任务输入映射到中间能力表征。每个 Template Model 是一个独立的可训练模块，封装特定的控制能力（结构控制、亮度调整、内容参考等）。

**Template Cache（模板缓存）**：
作为标准化能力注入接口，是连接 Template Model 和基础扩散运行时的桥梁。支持两种形式：
- **KV-Cache**：通过注意力机制传递控制信号，可沿序列维度拼接以融合多个控制
- **LoRA**：通过参数调整传递能力，可沿秩维度拼接融合

**Template Pipeline（模板流水线）**：
加载、合并和注入一个或多个 Template Cache 到基础扩散运行时，无需修改基础模型内部实现。

### 模型动物园（基于 FLUX.2-klein-base-4B）

框架已在多样化任务上验证，涵盖：

**低层视觉控制**：
| 任务 | 控制信号 | 架构 |
|------|---------|------|
| 结构控制 | 深度/边缘/分割/线稿 | KV-Cache + 轻量编码器 |
| 亮度调整 | 标量均值强度（归一化到[0,1]）| 位置编码 + FC 层 |
| 颜色调整 | RGB 三通道均值 | 与亮度相同 |
| 超分辨率 | 双线性上采样 → 高频细节恢复 | 与图像编辑相同 |
| 锐度增强 | Canny 边缘密度标量 | 与亮度相同 |

**高层语义控制**：
| 任务 | 控制信号 | 架构 |
|------|---------|------|
| 图像编辑 | 输入图像（转为 KV-Cache） | KV-Cache 结构控制同款 |
| 美学对齐 | GenAI-Arena 偏好排名 → LoRA | Image-to-LoRA |
| 内容参考 | SigLIP2 编码的参考图像 → LoRA | Image-to-LoRA |
| 局部修复 | 图像 + 掩码 | KV-Cache |
| 年龄控制 | 标量年龄值（10-90）| 与亮度相同 |

![结构控制效果](https://arxiv.org/html/2604.24351v1/assets/image_depth.jpg)
*图：基于深度图的结构控制效果，左为控制条件，右为生成结果*

![亮度调整效果](https://arxiv.org/html/2604.24351v1/assets/image_Brightness_dark.jpg)
*图：暗色调亮度控制示例*

![颜色调整效果](https://arxiv.org/html/2604.24351v1/assets/image_rgb_warm.jpg)
*图：暖色调颜色调整效果*

![图像编辑效果](https://arxiv.org/html/2604.24351v1/assets/image_Edit_hat.jpg)
*图：图像编辑能力展示（修改帽子细节）*

![超分辨率效果](https://arxiv.org/html/2604.24351v1/assets/image_Upscaler_1.png)
*图：超分辨率效果，从低分辨率图像恢复高频细节*

![美学对齐效果](https://arxiv.org/html/2604.24351v1/assets/image_Aesthetic_0.0.jpg)
*图：低美学分数控制示例*

![内容参考效果](https://arxiv.org/html/2604.24351v1/assets/image_ContentRef_1.jpg)
*图：内容参考控制，将参考图像风格迁移到新生成*

![年龄控制效果](https://arxiv.org/html/2604.24351v1/assets/image_age_20.jpg)
*图：年龄控制效果（约20岁）*

### Template 融合（可组合性）

多个 Template Model 可以在单一生成流水线中有效融合：
- **KV-Cache 融合**：沿序列维度拼接多个缓存
- **LoRA 融合**：沿秩维度拼接对应 LoRA 参数
- **混合融合**：不同格式的 Template（KV-Cache + LoRA）可在同一流水线中共存

![Template 融合示例](https://arxiv.org/html/2604.24351v1/x2.png)
*图：多个 Template 同时激活的组合控制效果*

## 实验结果

框架通过 10+ 个不同任务的案例研究验证其表达力和可扩展性。关键发现：

- **图像编辑速度提升**：通过 KV-Cache 传递编辑能力，实现了约 **2× 的经验加速**（相对于直接使用基础模型编辑），同时保持可比的编辑质量
- **美学对齐可泛化**：在 GenAI-Arena 训练的偏好 LoRA 能泛化到 Pick-a-Pic 数据集，表明跨数据集偏好迁移
- **跨任务架构复用**：亮度控制的轻量架构（位置编码 + FC 层）可直接复用于锐度增强和年龄控制，验证了架构的可复用性
- **异构控制可组合**：不同格式（KV-Cache、LoRA）的 Template 可在单一流水线中无冲突地组合使用

## 总结

Diffusion Templates 在系统设计层面定义了可控扩散的标准化接口，通过将控制能力封装为独立的可复用插件，显著降低了开发新可控生成方法的工程成本。KV-Cache 和 LoRA 作为两种通用能力载体的统一抽象，使框架能够覆盖从低层视觉属性（亮度、颜色）到高层语义控制（美学、身份、年龄）的广泛任务谱系。

局限性方面：局部修复时无法严格保证未掩码区域完全不变（仅软控制）；标量/向量控制信号对复杂视觉属性的描述能力有限；美学对齐依赖主观偏好标注，质量受数据集限制。所有代码、模型和数据集均将开源。
