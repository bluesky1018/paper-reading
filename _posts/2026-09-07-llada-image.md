---
layout: post
title: "LLaDA-Image：基于全开源训练方案构建强大图像生成器"
date: 2026-09-07
categories: [论文解读, 图像生成]
tags: [扩散模型, 图像生成, 多模态, DiT, 开源]
---

> 📄 **论文**：LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes
> 🔗 **arXiv**：[2609.03796](https://arxiv.org/abs/2609.03796)
> 🏢 **机构**：inclusionAI

## 一句话总结
LLaDA-Image 将 6B 参数扩散 Transformer 与冻结的视觉语言理解模块结合，在完全开放训练方案下实现了开源最优的图像生成性能。

## 背景与问题
高质量文生图模型（如 DALL-E 3、Midjourney、Stable Diffusion 3）的训练方案通常不公开，阻碍了学术界的进一步研究。现有开源图像生成器在以下方面仍有不足：

1. **指令遵循能力弱**：难以准确执行细粒度的文字编辑指令
2. **训练数据依赖配对数据**：从一开始就大量依赖配对图文数据，训练效率低
3. **缺乏统一的理解-生成框架**：理解模块与生成模块割裂，无法互相强化

LLaDA-Image 旨在解决以上问题，并提供**完整公开的训练方案**（模型权重、训练代码、数据配方），支持后续研究复现和改进。

## 核心方法
**架构设计**：LLaDA-Image 采用"双模块"统一框架：
- **生成模块**：从头训练的 6B 参数扩散 Transformer (DiT)，使用无参数 RMSNorm 和 Muon 优化器，提升训练稳定性和可扩展性
- **理解模块**：基于 LLaDA2.0-Mini 扩散语言模型的冻结视觉-语言理解模块

**训练策略（三阶段）**：
1. **图像纯训练（Image-only Pre-training）**：先仅用图像数据建立强视觉生成先验，无需配对文本
2. **中间训练（Mid-training）**：逐步引入图文配对数据，共 220M 样本（其中 98% 为真实图像）
3. **蒸馏加速**：将 LLaDA-Image 蒸馏为 LLaDA-Image-Turbo，实现 2-4 步快速推理

![Qwen Image Bench Overall](https://arxiv.org/html/2609.03796v1/2609.03796v1/qwen_image_bench_overall.png)
*图1：LLaDA-Image 在 Qwen-Image-Bench 上的整体得分对比*

![Demo 1](https://arxiv.org/html/2609.03796v1/2609.03796v1/figs/demo1.jpg)
*图2：LLaDA-Image 生成示例 1 - 高度逼真的图像生成效果*

![Demo 2](https://arxiv.org/html/2609.03796v1/2609.03796v1/figs/demo2.jpg)
*图3：LLaDA-Image 生成示例 2 - 细粒度编辑指令遵循*

## 实验结果
在 **Qwen-Image-Bench** 评测基准上，LLaDA-Image 取得开源最优成绩：

| 模型 | 英文分数 | 中文分数 |
|------|---------|---------|
| LLaDA-Image | **53.53** | **53.38** |
| 其他开源模型 | < 53.53 | < 53.38 |

- 在英文和中文两个赛道均创下**开源模型最优（SOTA）**成绩
- LLaDA-Image-Turbo 在保持竞争力的同时，将推理步数压缩至 2-4 步

## 总结
LLaDA-Image 的最大贡献不仅在于模型性能，更在于其**完全开放的训练方案**——模型权重、训练代码和详细数据配方全部公开发布，为开源社区的图像生成研究提供了强有力的基础。

"图像纯预训练 + 逐步引入配对数据"的训练策略在效率和效果上均有显著优势。局限性方面，对于特定的艺术风格或超高分辨率生成，仍与商业闭源模型存在差距；此外，使用 Muon 优化器的训练流程需要更多的超参数调优经验。
