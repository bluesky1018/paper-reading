---
layout: post
title: "SING：用生成模型分析分类器的语义不变性"
date: 2026-03-18
categories: [论文解读, 计算机视觉]
tags: [可解释性, 零空间, CLIP, SVD, 分类器分析, 语义不变性]
---

> **论文**：Make it SING: Analyzing Semantic Invariants in Classifiers
> **arXiv**：[2603.14610](https://arxiv.org/abs/2603.14610)
> **机构**：作者 Harel Yadid、Meir Yossef Levi、Roy Betser、Guy Gilboa（Technion — Israel Institute of Technology）
> **发表日期**：2026年3月15日
> **领域**：计算机视觉与模式识别（cs.CV）、图像与视频处理（eess.IV）

## 一句话总结

本文提出 SING（Semantic Interpretation of the Null-space Geometry），一种通过对分类器头部进行奇异值分解（SVD）、将零空间方向映射至 CLIP 多模态语言-视觉空间的方法，从而将原本不可解释的分类器不变量转化为可量化的自然语言描述和可视化示例。

---

## 背景与问题

所有分类器，包括当前最先进的视觉模型，都天然存在"不变量"（invariants）——即输入空间中的某些变化不会改变模型的输出。这些不变量在线性层的几何结构中有迹可循，具体而言，它们存在于分类器权重矩阵的**零空间（null space）**中。零空间决定了哪些输入扰动不会改变 logit，从而导致等价输入集合的产生。

这些等价集合中有些是良性的，例如背景变化或光照差异；但若零空间泄漏了语义相关信息（如类别的外观特征），则会带来两方面问题：其一，模型可能无法充分利用类别语义进行判别；其二，攻击者可以在不改变分类结果的前提下，对图像施加显著的语义扰动，这构成潜在的安全风险。

现有方法的局限性在于：
- 基于数据协方差的特征空间分解（如 PCA on latent features）反映的是数据分布而非网络决策几何；
- 直接在权重诱导的零空间上操作的方法虽能识别不变方向的存在，却无法赋予其**语义含义**；
- 尚无方法能系统性地将分类器不变方向映射到多模态空间，从而获取文字描述和可视化示例。

本文的贡献正是填补这一空白：将零空间结构转化为**人类可读的语义解释**。

---

## 核心方法

SING 的整体流程如下图所示：

![方法总览](https://arxiv.org/html/2603.14610v1/Figs/method_2.png)
*图：SING 方法总览。主要包含三个模块：(1) 对分类器头部权重进行 SVD，分离主空间与零空间；(2) 训练线性 Translator 将分类器特征映射至 CLIP 图像空间；(3) 在零空间方向对特征施加扰动并生成等价图像，随后用 Translator 度量语义变化。*

### 3.1 问题设置

设分类器最后一层全连接层为 $W \in \mathbb{R}^{c \times m}$，将倒数第二层特征 $f \in \mathbb{R}^m$ 映射为 $c$ 维 logit 向量。SING 聚焦于该层的几何结构分析。

通过 SVD，可将 $W$ 分解为：

$$W = U \Sigma V^\top, \quad V = [V_p \; V_n]$$

其中 $V_p$ 对应非零奇异值（主空间方向），$V_n$ 为剩余列，张成零空间。对任意 $\nu \in \text{span}(V_n)$，有：

$$W(f + \nu) = Wf + W\nu = Wf$$

即零空间方向的扰动不改变 logit。对应的投影矩阵为：

$$\Pi_p = V_p V_p^\top, \quad \Pi_n = V_n V_n^\top$$

### 3.2 训练线性 Translator

为了赋予零空间方向以语义含义，SING 学习一个线性映射 $T_\Theta: \mathbb{R}^m \to \mathbb{R}^n$，将分类器特征 $f$ 映射到 CLIP 的图像特征空间 $z^{img} \in \mathbb{R}^n$。损失函数为均方误差加权重衰减：

$$\mathcal{L} = \|T_\Theta(f) - z^{img}\|_2^2 + \lambda \|\Theta\|_2^2$$

由于 Translator 是线性的，它天然支持加性特征分解：$T_\Theta(f + v) = T_\Theta(f) + T_\Theta(v)$，与 SING 的框架完美契合。

### 3.3 度量指标

**Attribute Score（AS）**：衡量零空间对类别语义的泄漏程度。设 $f$ 为原始特征，$\tilde{f} = f - \Pi_n f$ 为去除零空间分量后的等价特征，文本提示的 CLIP 嵌入为 $z^{text}$，则：

$$\text{AS}(f, \tilde{f} | z^{text}, T_\Theta) = \angle(T_\Theta(f), z^{text}) - \angle(T_\Theta(\tilde{f}), z^{text})$$

AS 为正表示去零空间后特征在 CLIP 空间中更靠近目标文本；AS 越大，说明该类别语义越多地泄漏至零空间。

**Image Score（IS）**：衡量不变空间中对类别无关语义变化的容忍度，反映模型不变空间的"丰富性"。理想模型应具有较低的 AS 和较高的 IS（IS/AS 比值越高越好）。

### 3.4 应用场景

SING 支持多级分析：

- **模型级比较**：对多个架构进行 AS/IS 统计对比，揭示哪些模型更容易发生语义泄漏；
- **类别级分析**：逐类计算 AS，找出最易泄漏语义的类别；
- **开放词汇概念分析**：用任意文本 prompt 度量 AS，探索特定概念与不变量的关联；
- **单图像调试**：在单张图上识别局部不变量；
- **零空间语义引导**：沿文本梯度方向在零空间内施加扰动，生成等价图像，展示可被操控的语义内容。

---

## 实验结果

### 4.1 数据集与模型

实验基于 ImageNet-1k（1000 类），使用五种预训练模型：
- **DinoViT**：ViT 架构，使用自监督 DINO 预训练
- **ResNet50**：经典 CNN 架构
- **ResNext101**：弱监督预训练
- **EfficientNetB4**：Noisy Student 训练
- **BiTResNetv2**：Big Transfer 预训练

每模型收集 10k 特征向量，训练专属 Translator。

### 4.2 模型级比较

![模型级比较散点图与柱状图](https://arxiv.org/html/2603.14610v1/Figs/to_paper/median_ratio_scatter_5models.png)
*图：五个模型的 Attribute Score（AS）与 Image Score（IS）联合分布（左）及 IS/AS 比值柱状图（右）。理想模型应在 AS 低、IS 高的区域，即图左上角。DinoViT 表现最佳，ResNext101 最差。*

![IS/AS比值柱状图](https://arxiv.org/html/2603.14610v1/Figs/to_paper/median_ratio_bar_1000cls_5models.png)
*图：五个模型在 1000 类上的 IS/AS 中位数比值（越高越好）。DinoViT 比值最高，ResNext101 最低，表明后者的零空间更易泄漏类别语义。*

**结论**：DinoViT 在 IS/AS 权衡上表现最佳，这与其在大规模多样化语料库上的基础模型预训练一致。ResNext101 的 AS 最高且方差大，表明其零空间存在明显的类别相关语义泄漏。

### 4.3 类别级分析（零空间小提琴图）

![ResNet50 类别AS分布](https://arxiv.org/html/2603.14610v1/Figs/violin/violin_null_semantic_change_resnet50_all.png)
*图：ResNet50 各类别 AS 的小提琴图分布。部分类别（如 Porcupine、Sports-Car）的 AS 幅值明显偏大，说明这些类别的语义信息大量泄漏至零空间。*

![DinoViT 类别AS分布](https://arxiv.org/html/2603.14610v1/Figs/violin/violin_null_semantic_change_dinovit_all.png)
*图：DinoViT 各类别 AS 的小提琴图分布。AS 幅值普遍较小（通常 |AS| < 1°），表明 DinoViT 几乎不存在类别语义向零空间的泄漏。*

DinoViT 跨类别行为稳定，AS 幅度极小；ResNet50 则表现出更大且更不均匀的 AS，提示其可能依赖虚假相关特征（spurious cues）。两模型的每类 AS 排名无显著相关性，说明该效应与模型架构相关，而非由数据集类别结构驱动。

### 4.4 开放词汇概念分析

![Arabian Camel 概念分析](https://arxiv.org/html/2603.14610v1/Figs/to_paper/string_graph_open_vocabulary_arabian_camel_dino.png)
*图：DinoViT 对 "Arabian Camel" 类进行开放词汇概念分析。蓝点为原始特征，红点为去除零空间后的等价特征，绿色箭头表示 AS（箭头越短，泄漏越少）。"Desert" 在 CLIP 空间中与该类别最相近，且 AS 普遍较小，表明该类别的语义不变性良好。*

![Jellyfish 概念分析](https://arxiv.org/html/2603.14610v1/Figs/to_paper/string_graph_open_vocabulary_jellyfish_dino.png)
*图：DinoViT 对 "Jellyfish" 类进行开放词汇概念分析。与 Arabian Camel 相比，Jellyfish 类别的 AS 幅值明显更大（绿色箭头更长），说明更多与该类别相关的概念泄漏至零空间。*

### 4.5 零空间语义引导（梯度方向分析）

![ResNet50 零空间语义扰动](https://arxiv.org/html/2603.14610v1/Figs/to_paper/resnet_perturbation.png)
*图：ResNet50 零空间语义引导示例。从原始图像出发，沿 CLIP 相似度梯度投影到零空间方向施加扰动，生成的等价图像（用 UnCLIP 可视化）呈现出朝向 Arabian Camel、Starfish、Pirate、Jellyfish、Jeep 等概念的显著语义偏移——而 logit 保持不变，揭示了潜在的安全风险。*

**表1：文本梯度零空间扰动实验**（以 IS=40° 为固定步长，从 Sports Car 向 "jellyfish" 方向扰动）：

| 模型 | \|AS\| 均值 ± 标准差（越低越好） |
|------|-------------------------------|
| ResNet50 | 12.04 ± 0.25 |
| EfficientNet | 12.38 ± 0.52 |
| BiTResNet | 9.19 ± 0.31 |
| **DinoViT** | **5.0 ± 0.59** |
| ResNext101 | 11.15 ± 0.53 |

DinoViT 对定向零空间操纵的抵抗力最强（AS 最低），ResNet50 和 EfficientNet 最容易被语义引导。

### 附录图示

![鲨鱼角度可视化](https://arxiv.org/html/2603.14610v1/Figs/shark_angles.png)
*图：单张鲨鱼图像的零空间角度可视化示例，展示局部不变量分析能力。*

![BiTResNet 类别AS分布](https://arxiv.org/html/2603.14610v1/Figs/violin/violin_null_semantic_change_bitresnet_all.png)
*图：BiTResNet 各类别 AS 小提琴图。*

![ResNext50 类别AS分布](https://arxiv.org/html/2603.14610v1/Figs/violin/violin_null_semantic_change_resnext50_all.png)
*图：ResNext50 各类别 AS 小提琴图。*

![EfficientNetB4 类别AS分布](https://arxiv.org/html/2603.14610v1/Figs/violin/violin_null_semantic_change_efficientnetb4noisystudent_all.png)
*图：EfficientNetB4（Noisy Student）各类别 AS 小提琴图。*

---

## 总结

SING 提供了一个简洁、通用且可解释的框架，将分类器零空间的抽象几何结构转化为人类可读的语义分析。其核心贡献在于：通过 SVD 分离主空间与零空间、训练线性 Translator 将特征映射至 CLIP 多模态空间，并设计 AS/IS 两个度量指标，实现了从模型级、类别级到单图像级的多粒度不变量分析。

实验表明，DinoViT 在所有测试模型中具有最佳的语义不变性（IS/AS 比值最高），而 ResNet50 和 ResNext101 则存在明显的类别语义泄漏。这一发现不仅有助于模型调试与比较，也揭示了一个安全隐患：攻击者可以在不改变分类结果的前提下，在零空间方向对图像施加语义显著的扰动。

未来工作的两个方向包括：(1) 在微调阶段引入定向数据增强，鼓励关键概念的 AS 趋近于零；(2) 通过投影正则化、秩调整或约束更新等线性代数手段，主动将有用语义从零空间迁移至主空间，同时保持 logit 不变。

---

Sources:
- [Make it SING: Analyzing Semantic Invariants in Classifiers (arXiv:2603.14610)](https://arxiv.org/abs/2603.14610)
