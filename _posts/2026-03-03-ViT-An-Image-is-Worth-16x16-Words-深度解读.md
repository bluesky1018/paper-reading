---
title: "ViT: An Image is Worth 16x16 Words — 一张图片值 16×16 个词"
date: 2026-03-03 16:20:00 +0800
categories: [Vision, Transformer, Foundation Model]
tags: [vit, vision-transformer, image-classification, google, self-attention, patch-embedding, transfer-learning]
math: true
---

## 基本信息

- **作者**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov 等
- **机构**: Google Research, Brain Team
- **发表**: ICLR 2021（arXiv 2020.10）
- **arXiv**: [2010.11929](https://arxiv.org/abs/2010.11929)
- **代码**: [github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)

## 一句话总结

**把图像切成 16×16 的 Patch，当作"单词"直接喂给标准 Transformer——不需要任何卷积**。在大规模数据预训练后，ViT 在 ImageNet 上达到 **88.55%** 的准确率，超越所有 CNN，且训练计算量更少。这篇论文彻底改变了计算机视觉的研究范式。

![ViT 模型总览：图像被切分为固定大小的 Patch，线性投影后加入位置嵌入，送入标准 Transformer 编码器，通过 [class] token 进行分类](/assets/img/vit/x1.png)
_Figure 1：Vision Transformer (ViT) 模型架构总览_

---

## 背景：为什么要把 Transformer 用在视觉上？

### NLP 的成功启示

到 2020 年，Transformer 已经统治了 NLP：BERT、GPT 系列证明了一个规律——**模型越大、数据越多，性能越好，而且看不到天花板**。但在计算机视觉领域，CNN（特别是 ResNet）仍然是绝对霸主。

之前也有人尝试把自注意力引入视觉：
- 在 CNN 基础上**添加**自注意力模块
- 用自注意力**替换**CNN 中的某些组件
- 像素级自注意力（计算量 $O(n^2)$，$n$ = 像素数，完全不实际）

但没人敢做最极端的事情：**完全抛弃卷积，直接用纯 Transformer 处理图像**。

### ViT 的大胆假设

ViT 的核心假设极其简单：

> **CNN 的归纳偏置（局部性、平移等变性）不是必需的。当数据足够多时，Transformer 可以从数据中学到这一切。**

---

## 架构设计：极简主义的胜利

ViT 的设计哲学是**尽可能少地修改标准 Transformer**，这样就能直接复用 NLP 社区已有的高效实现和训练基础设施。

### Patch Embedding：图像→序列

将图像 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ 切分为 $N$ 个大小为 $P \times P$ 的 Patch：

$$N = \frac{H \times W}{P^2}$$

例如 224×224 的图像，$P=16$ 时得到 $N = 196$ 个 Patch。

每个 Patch 展平后通过**可训练的线性投影**映射到 $D$ 维：

$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}};\, \mathbf{x}_p^1\mathbf{E};\, \mathbf{x}_p^2\mathbf{E};\, \cdots;\, \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{pos}$$

其中：
- $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$：Patch 嵌入矩阵
- $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$：可学习的 1D 位置嵌入
- $\mathbf{x}_{\text{class}}$：可学习的 [class] token（类似 BERT 的 [CLS]）

> **一个反直觉的发现**：使用简单的 1D 位置嵌入和 2D 位置嵌入效果几乎一样。模型可以自己学到 2D 空间关系。

### Transformer 编码器

完全标准的 Transformer 编码器，没有任何视觉特定的修改：

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$

$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

$$\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$$

特点：
- **Pre-Norm**：LayerNorm 在注意力/MLP **之前**（而非之后）
- **GELU 激活**
- 最终 [class] token 的输出作为图像表示

### 模型变体

ViT 的配置直接借鉴 BERT 的命名：

| 模型 | 层数 | 隐藏维度 | 头数 | 参数量 |
|------|------|---------|------|--------|
| ViT-Base | 12 | 768 | 12 | 86M |
| ViT-Large | 24 | 1024 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 16 | 632M |

命名规则 `ViT-L/16` 表示 Large 模型 + 16×16 Patch。Patch 越小序列越长，计算量越大。

### 混合架构（Hybrid）

作为替代方案，也可以用 CNN（如 ResNet）的中间特征图代替原始 Patch Embedding：

- 先用 ResNet 提取特征
- 将特征图的空间维度展平为序列
- 送入 Transformer

这种混合架构在小计算预算下略优于纯 ViT，但**随着模型变大，差距消失**。

---

## 归纳偏置：ViT vs CNN

这是论文最深刻的讨论之一。

### CNN 的内置先验

CNN 在每一层都强制施加三种归纳偏置：
1. **局部性（Locality）**：卷积核只看局部区域
2. **平移等变性（Translation Equivariance）**：同一个卷积核扫描整张图
3. **2D 邻域结构**：每一层都保持空间拓扑

### ViT 几乎没有视觉先验

ViT 中图像特有的归纳偏置**极其稀少**：
- **只在两个地方**体现 2D 结构：
  1. 初始的 Patch 切分
  2. 微调时位置嵌入的 2D 插值
- MLP 层是**局部+平移等变**的
- 自注意力层是**全局**的
- **所有空间关系必须从零学习**

> 这意味着 ViT 需要更多的数据才能学到 CNN "白送"的那些先验。但一旦数据够多，从数据中学到的表示比手工设计的先验**更强大、更灵活**。

---

## 训练策略

### 预训练

- **数据集**：ImageNet-1k（1.3M）、ImageNet-21k（14M）、JFT-300M（303M）
- **优化器**：Adam（$\beta_1=0.9$，$\beta_2=0.999$）
- **Batch size**：4096
- **权重衰减**：0.1（非常高，有利于迁移学习）
- **学习率**：线性 Warmup + 余弦衰减

### 微调

- 移除预训练分类头，添加零初始化的 $D \times K$ 线性层
- 使用 SGD + Momentum
- **高分辨率微调**：ViT-L/16 用 512，ViT-H/14 用 518
- 位置嵌入通过 **2D 插值**适应新分辨率

> 高分辨率微调是关键技巧：Patch 大小不变，序列变长，位置嵌入通过插值适配。这是 ViT 中**唯一手动注入 2D 结构信息**的地方。

---

## 核心实验结果

### 与 SOTA 的对比

![在不同预训练数据规模下的 ImageNet 迁移性能：ViT 在大数据集上超越 BiT ResNet](/assets/img/vit/x3.png)
_Figure 3：预训练数据规模 vs ImageNet 迁移准确率——数据越多，ViT 优势越明显_

| 模型 | 预训练数据 | ImageNet | ImageNet-ReaL | CIFAR-100 | VTAB (19 tasks) | 预训练 TPUv3-days |
|------|----------|----------|---------------|-----------|-----------------|-----------------|
| BiT-L (ResNet152x4) | JFT-300M | 87.54 | 90.54 | 93.51 | 76.3 | 9900 |
| Noisy Student (EfficientNet-L2) | JFT-300M | 88.4 | 90.55 | - | - | 12300 |
| **ViT-H/14** | **JFT-300M** | **88.55** | **90.72** | **94.55** | **77.63** | **2500** |
| ViT-L/16 | ImageNet-21k | 85.30 | 88.62 | 93.25 | 72.72 | 0.68k |

关键发现：
- **ViT-H/14 全面超越所有 CNN**，在 ImageNet、CIFAR-100、VTAB 上都是 SOTA
- **计算量仅为 BiT-L 的 1/4**（2500 vs 9900 TPUv3-days）
- 即使用较小的 ImageNet-21k 预训练，ViT-L/16 也很强，且只需 8 核 TPU 训练约 30 天

### VTAB 任务分解

![VTAB 性能分解：ViT 在 Natural 和 Structured 任务上超越 BiT，Specialized 任务上持平](/assets/img/vit/x2.png)
_Figure 2：VTAB 基准上的任务类别分解对比_

### 数据规模的关键性

这是论文**最重要的发现之一**：

![Few-shot 准确率 vs 预训练数据规模：ViT 在小数据上不如 ResNet，但大数据上反超](/assets/img/vit/imagenet_5shot.png)
_Figure 4：数据量对 ViT vs CNN 的影响——小数据时 CNN 的归纳偏置占优，大数据时 ViT 反超_

- **小数据**（ImageNet-1k）：ViT 不如同等大小的 ResNet，因为缺少归纳偏置导致过拟合
- **中数据**（ImageNet-21k）：两者持平
- **大数据**（JFT-300M）：ViT 全面超越，且**模型越大优势越明显**

> 这个结果优雅地验证了核心假设：**卷积的归纳偏置对小数据有用，但对大数据来说，从数据中直接学习更好**。

### 计算效率：性能 vs 算力

![性能 vs 预训练计算量：ViT 在相同算力下全面优于 ResNet](/assets/img/vit/x4.png)
_Figure 5：相同计算预算下，ViT 比 ResNet 性能更高；混合模型在小预算时略优，大预算时差距消失_

- ViT 在相同计算量下，性能比 ResNet 高 **2-4 倍效率**
- 混合模型在小预算下有微弱优势，但大模型时差距消失
- **ViT 的性能曲线看不到饱和**，暗示进一步扩大模型仍然有收益

---

## 模型可解释性：ViT 学到了什么？

论文对 ViT 的内部表示做了精彩的可视化分析。

### Patch Embedding 的滤波器

![ViT 学到的 Patch 嵌入滤波器的前 28 个主成分，类似 Gabor 滤波器](/assets/img/vit/x6.png)
_Figure 7 (左)：Patch Embedding 的主成分——类似于 CNN 第一层学到的 Gabor 滤波器_

线性投影层学到的滤波器类似于 CNN 第一层的边缘检测器和纹理检测器，说明模型自发学会了低级视觉特征提取。

### 位置嵌入的空间结构

![位置嵌入的余弦相似度矩阵，展示了清晰的行列结构和距离关系](/assets/img/vit/x7.png)
_Figure 7 (中)：位置嵌入的余弦相似度——模型自己学到了 2D 空间拓扑！_

这张图非常有说服力：
- **相邻 Patch 的位置嵌入更相似**
- 清晰的**行列结构**自发涌现
- 这解释了为什么 1D 位置嵌入就够用——模型自己学到了 2D 关系

### 注意力距离随深度变化

![注意力距离随网络深度的变化：浅层既有局部注意力也有全局注意力，深层主要是全局注意力](/assets/img/vit/x8.png)
_Figure 7 (右)：各层各头的平均注意力距离——浅层局部+全局并存，深层趋于全局_

- **浅层**：部分头关注局部（类似 CNN 的卷积核），部分头关注全局
- **深层**：几乎所有头都关注全局语义区域
- 这说明 ViT **同时具备局部和全局建模能力**，而不是只有全局

### 注意力可视化

![ViT 最后一层的注意力图：模型自动关注语义相关的图像区域](/assets/img/vit/x5.png)
_Figure 6：输出 token 对输入的注意力可视化——模型自动聚焦于分类相关的语义区域_

---

## 深度思考

### 1. 为什么这篇论文是里程碑？

ViT 的贡献远超一个新模型。它证明了一个**范式级别的结论**：

> 视觉领域不需要专门设计的架构。NLP 的 Transformer 可以直接迁移过来，只需要足够的数据。

这直接导致了后续几年的"Transformer 统一一切"浪潮：
- **检测**：DETR、DINO
- **分割**：SegFormer、Mask2Former
- **生成**：DiT（扩散 Transformer）
- **多模态**：CLIP、BLIP、LLaVA
- **视频**：VideoMAE、TimeSformer

### 2. 数据 vs 归纳偏置的辩证法

ViT 论文揭示了一个深刻的 trade-off：

| | 小数据 | 大数据 |
|---|--------|--------|
| **强归纳偏置（CNN）** | ✅ 泛化好 | ❌ 表达力受限 |
| **弱归纳偏置（ViT）** | ❌ 容易过拟合 | ✅ 从数据中学到更好的表示 |

这暗示了一个一般性原理：**当数据匮乏时，先验知识是宝贵的；当数据充足时，先验知识反而是枷锁**。

### 3. 计算效率的秘密

ViT 为什么用更少的算力就能达到更好的效果？

- Transformer 的**注意力机制天然适合并行化**，在 TPU/GPU 上效率极高
- CNN 的卷积操作虽然 FLOP 少，但**内存访问模式不够友好**
- ViT 可以直接利用 NLP 社区已经高度优化的 Transformer 实现

### 4. Patch 大小的设计权衡

$P = 16$ 这个选择是精心权衡的结果：
- **$P$ 太大**（如 32）：序列太短，信息损失
- **$P$ 太小**（如 4）：序列太长（3136 个 token），注意力的 $O(N^2)$ 成本爆炸
- **$P = 16$**：196 个 token，计算可控且保留足够信息

这也暗示了一个重要方向：如何在保持长序列的同时降低计算成本——后来的 Swin Transformer 用窗口注意力解决了这个问题。

### 5. 自监督学习的伏笔

论文末尾用 Masked Patch Prediction 做了初步的自监督实验（类似 BERT 的 MLM），ViT-B/16 达到 79.9% ImageNet 准确率。虽然比有监督低 4%，但这为后来的 **MAE（Masked Autoencoders）** 埋下了伏笔——MAE 后来证明 Masked Image Modeling 在 ViT 上极其有效。

### 6. 局限性

- **小数据表现差**：没有归纳偏置的代价，后来 DeiT 通过知识蒸馏和数据增强部分解决
- **分辨率限制**：位置嵌入是固定长度的，换分辨率需要插值
- **只做了分类**：检测、分割等密集预测任务未验证（后续工作补上了）
- **依赖 JFT-300M**：这是 Google 内部数据集，外部不可用

---

## 历史影响

ViT 发表后的连锁反应：

| 时间 | 后续工作 | 贡献 |
|------|---------|------|
| 2021.01 | **DeiT** | 证明 ViT 不需要海量数据，知识蒸馏 + 数据增强即可在 ImageNet-1k 上训练好 |
| 2021.03 | **Swin Transformer** | 引入窗口注意力 + 层次结构，解决 ViT 的多尺度问题 |
| 2021.03 | **CLIP** | ViT + 对比学习，开启视觉-语言预训练时代 |
| 2021.11 | **MAE** | Masked Autoencoder，证明自监督在 ViT 上极为有效 |
| 2022+ | **DiT, SAM, DINO v2** | ViT 成为视觉基础模型的默认架构 |

可以说，**ViT 之于计算机视觉，就像 Transformer 之于 NLP**——它开启了一个时代。

---

## 总结

ViT 用最简单的方式回答了一个大胆的问题：**纯 Transformer 能做视觉吗？** 答案是不仅能，而且在数据充足时比精心设计的 CNN 更好、更高效。

这篇论文的真正遗产不是 88.55% 的 ImageNet 准确率，而是一个深刻的认知转变：

> **当规模足够大时，简单的通用架构胜过复杂的专用设计。**

这个理念延续至今，是当前大模型时代的核心信条之一。

> **论文链接**: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
>
> **代码**: [github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)
