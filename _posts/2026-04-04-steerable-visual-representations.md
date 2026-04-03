---
layout: post
title: "可操控视觉表示：通过自然语言将ViT注意力引导至目标概念"
date: 2026-04-04
categories: [论文解读, 计算机视觉]
tags: [视觉表示, Vision Transformer, 可操控性, 异常检测, 跨模态]
---

> 📄 **论文**：Steerable Visual Representations
> 🔗 **arXiv**：[2604.02327](https://arxiv.org/abs/2604.02327)
> 🏢 **机构**：Jona Ruthardt, Manu Gaur, Deva Ramanan, Makarand Tapaswi, Yuki M. Asano

## 一句话总结

SteerViT 通过在冻结ViT各层内部插入轻量级交叉注意力（早期融合），使自然语言能够精确引导视觉表示聚焦于目标概念，在目标检索、异常检测和个性化对象识别上大幅超越DINOv2基线。

## 背景与问题

预训练视觉Transformer（ViT）如DINOv2和MAE已成为通用视觉特征的标准来源，但它们存在一个根本性局限：**特征自发聚焦于视觉上最显著的元素，无法被引导至目标概念**。

例如，当图像中有一只猫（最显著）和一把椅子时，若用户想检索"椅子"，DINOv2的特征仍会主要编码猫的信息，导致检索失败。

现有的解决尝试均不令人满意：
- **CLIP等多模态编码器**：后期将文本与视觉特征融合（"晚期融合"），模型变得"以语言为中心"，失去了通用视觉特征的有效性
- **多模态LLM（MLLM）**：如InternVL3-1B在CORE基准上仅达约47%，远低于DINOv2（43.7%）的基础上再提升

关键问题：如何在不牺牲表示质量的前提下，让文本真正引导视觉特征？

## 核心方法

**SteerViT：早期融合交叉注意力**

核心创新：将文本信息通过轻量级**交叉注意力层**注入冻结ViT的内部块（而非拼接在输出端）。

架构细节：
- **冻结ViT骨干**：默认使用DINOv2 ViT-B/14
- **冻结文本编码器**：RoBERTa-Large
- **MLP适配器**：将文本特征投影到视觉空间
- **门控交叉注意力**：每隔一个Transformer块插入一层

门控公式（零初始化，确保训练初始与基础ViT完全一致）：
> Z_v^(ℓ+1) = Z_v^(ℓ) + tanh(α_ℓ) · Ẑ_v^(ℓ)

仅约**2100万参数**可训练（相比ViT主体参数量极小）。

**训练数据**

在162K图像/228万图文对上训练patch级参考分割任务，数据来源：RefCOCO、LVIS、Visual Genome、Mapillary Vistas。

![引导效果展示](https://arxiv.org/html/2604.02327/x1.png)
*图1：猫与椅子场景示例——SteerViT将注意力成功引导至文本描述的目标对象（椅子）*

![架构图](https://arxiv.org/html/2604.02327/x4.png)
*图4：SteerViT架构——早期融合交叉注意力层插入DINOv2各Transformer块*

![视觉编码分类](https://arxiv.org/html/2604.02327/x3.png)
*图3：视觉编码器分类分析——从纯视觉到多模态的Pareto权衡前沿*

## 实验结果

**主要基准对比：**

| 任务 | DINOv2 | SteerViT |
|------|--------|---------|
| CORE检索 (acc@1) | 43.7% | **96.0%** |
| MOSAIC PR-AUC | 14.3% | **50.2%** |
| PODS PR-AUC（描述性） | 29.6% | **58.1%** |
| GeneCIS R@1 | 9.6% | **25.4%** |
| MVTec PRO（异常检测） | — | **82.1** |

- 晚期融合（CLIP/SigLIP后处理）在CORE上仅提升 +0.02%（43.7→43.72%），几乎无效
- MLLM（InternVL3-1B）在CORE上达约47%，仍远低于SteerViT的96%

![CORE基准对比](https://arxiv.org/html/2604.02327/x6.png)
*图5：CORE基准上各方法对比——SteerViT以96%准确率远超所有基线*

![MOSAIC注意力图](https://arxiv.org/html/2604.02327/figures/mosaic_attention_maps.png)
*图6：MOSAIC场景下的注意力图——SteerViT精准聚焦到查询目标*

**门控缩放消融（Table 3）：**

| 配置 | CORE↑ | PODS↑ |
|------|-------|-------|
| 基础DINOv2 | 43.7 | 29.6 |
| 完整SteerViT | **96.0** | **58.1** |
| 仅晚期融合 | 93.3 | 36.6 |
| 无tanh门控 | 94.6 | 47.1 |
| 线性（无MLP） | 95.2 | 56.4 |

早期融合对精细任务（PODS差距：58.1 vs 36.6）尤其关键。

![UMAP嵌入拓扑](https://arxiv.org/html/2604.02327/x10.png)
*图9：UMAP可视化——文本引导使嵌入空间拓扑结构重组，不同语义描述的图像形成清晰聚类*

![异常检测效果](https://arxiv.org/html/2604.02327/figures/anomaly_detection_grid_carpet.png)
*图10：MVTec异常检测热力图——通过文本描述目标材质，实现零样本异常定位*

## 总结

SteerViT 首次实现了"高可操控性"与"高表示质量"的同时兼顾，填补了视觉表示领域的一个重要空白。早期融合设计使文本在特征提取过程的每一层都能参与引导，远比事后融合更加有效。

局限性方面，SteerViT 的引导效果依赖于文本描述的精确性——描述越具体，引导效果越好，但对于抽象或模糊的文本查询效果有限；此外，当前框架在单目标引导场景下性能卓越，多目标同时引导（"椅子和窗户"）的表现有待进一步研究。
