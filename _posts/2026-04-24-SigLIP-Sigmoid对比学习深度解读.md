---
title: "SigLIP — 用 Sigmoid 替代 Softmax,让 CLIP 训练更稳更快"
date: 2026-04-24 22:15:00 +0800
categories: [Pretraining, VLM]
tags: [siglip, contrastive-learning, sigmoid-loss, zhai-2023]
math: true
---

## 基本信息

- **作者**: Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer
- **机构**: Google DeepMind
- **发表**: ICCV 2023
- **arXiv**: [2303.15343](https://arxiv.org/abs/2303.15343)
- **全名**: *Sigmoid Loss for Language Image Pre-training*

## 一句话总结

Google 的 **SigLIP**——对 CLIP 的 InfoNCE softmax loss 做一个看似简单却深刻的改动:**把 batch 内 softmax 对比替换为独立的 sigmoid 二分类**。每对 (image, text) 独立判断"是否匹配",去掉 batch 维度的全局归一化。这一个改动解决了 CLIP 的**三个痛点**:(1) 训练小 batch 也能有效——CLIP 需 32K+ batch,SigLIP 在 4K 就 work;(2) 数值稳定性大幅提升;(3) 可以用 multi-GPU 独立计算 loss 再合并,不需要 all-gather。SigLIP 后来成为 **LLaVA-NeXT、PaliGemma、InternVL 等几乎所有现代 VLM 的默认视觉 encoder**,"SigLIP + MLP + LLM" 取代了 "CLIP + MLP + LLM" 成为 2024-2025 年 VLM 事实标准。

![SigLIP vs CLIP 的 loss 对比:CLIP 用 batch-wise softmax 对比,需要 all-gather 全局归一化;SigLIP 用 pair-wise sigmoid,每对独立判断,不需要全局 batch 信息。](/assets/img/siglip/x1.png)
_Figure 1:Sigmoid loss 的核心简化_

---

## 背景:CLIP 的训练痛点

### CLIP 的 InfoNCE softmax loss

CLIP 的对比 loss:

$$
L_{\text{image}\to\text{text}} = -\frac{1}{N}\sum_i \log\frac{\exp(S_{ii}/\tau)}{\sum_j \exp(S_{ij}/\tau)}
$$

每个 image 相对 **batch 内所有 text** 做归一化——这是 softmax 的本质。

### 痛点

1. **需要大 batch**:batch size < 16K 时性能明显降低,因为 negative samples 不够
2. **All-gather 通信昂贵**:multi-GPU 训练时所有 GPU 的 embedding 要互相共享
3. **数值不稳**:softmax 的 $\log\sum\exp$ 容易数值爆炸或消失
4. **GPU 显存爆炸**:batch 内 N×N similarity 矩阵占大量显存

Google 团队的问题:**能不能去掉 batch 内的全局归一化?**

---

## 核心机制:Sigmoid Loss

### 新的 loss 设计

SigLIP 把对比任务重新定义为**逐对二分类**:

对每对 $(I_i, T_j)$ 独立判断"是否匹配":

$$
L = -\frac{1}{N}\sum_{i,j} \log\frac{1}{1 + \exp(-z_{ij} \cdot (\tau \cdot \text{sim}(I_i, T_j) + b))}
$$

其中:
- $z_{ij} = +1$ 当 $i = j$(正样本),$z_{ij} = -1$ 否则(负样本)
- $\tau$ 是温度
- $b$ 是 **可学 bias**(关键!)

### 关键改动的含义

**每对独立**:$(I_i, T_j)$ 的 loss 只依赖这一对,不需要看其他 samples——完全独立。

**Bias $b$**:
- 正样本多,负样本少——类别不平衡
- $b$ 初始化为 $-10$ 左右(强偏向"不匹配"),随训练调整
- 这让 sigmoid 的 decision boundary 匹配实际正负比

### 训练效率的胜利

![SigLIP 在不同 batch size 下的性能:CLIP 在 batch 16K 以下性能大降,SigLIP 在 4K batch 已经 work,在 16K+ batch 上性能几乎饱和。SigLIP 让小 batch 训练变得可行。](/assets/img/siglip/x2.png)
_Figure 2:SigLIP 在小 batch 下的优势_

- **Batch 4K** 时 SigLIP 已工作良好,CLIP 还需 16K 才能匹配
- **Memory** 减半(不算 N×N 的 softmax 矩阵)
- **通信** 减少——不需要 all-gather,每 GPU 算本地 pairs 的 sigmoid 即可

---

## 实验结果

### 1. 零样本 ImageNet

| 方法 | 模型 | Batch | Zero-shot IN1k |
|------|------|-------|----------------|
| CLIP | ViT-B/16 | 32K | 68.3 |
| OpenCLIP | ViT-B/16 | 32K | 67.4 |
| **SigLIP** | **ViT-B/16** | **32K** | **73.4** |
| **SigLIP** | **ViT-L/16** | **32K** | **78.6** |
| **SigLIP** | **SoViT-400M/14** | **32K** | **82.8** |

SigLIP ViT-B/16 比 CLIP ViT-B/16 **高 5 分**——纯 loss function 改动。

### 2. 训练速度

同样的模型大小、同样的最终精度:

- SigLIP 比 CLIP 训练速度 **~2× 快**
- 或者说,同样训练时间 SigLIP 能达到 CLIP 达不到的精度

### 3. 下游任务

SigLIP 在各种下游任务(caption、VQA、retrieval)上都匹配或超过 CLIP。

---

## 变体:SigLIP 2

2024 年底 Google 发布 **SigLIP 2**:

- 更大训练数据
- 更好的 data filtering
- 在 SoViT-400M/14 上达到 ImageNet **83%+**

SigLIP 2 成为 2025 年最强 open-source vision encoder 之一。

---

## 历史影响

### 1. 现代 VLM 的默认视觉 encoder

自 2024 年起,几乎所有 SOTA 开源 VLM 默认用 **SigLIP(或 SigLIP 2)作为 vision encoder**:

- **PaliGemma** (Google 2024):SigLIP + Gemma
- **LLaVA-NeXT** (2024-01):用 SigLIP 改进版
- **InternVL 系列**:SigLIP + LLaMA
- **MiniCPM-V**:SigLIP
- **Qwen2-VL, Qwen2.5-VL**:SigLIP 改进版
- **DeepSeek-VL2**:SigLIP

**SigLIP 基本取代了 CLIP** 作为视觉 backbone。

### 2. 对比学习的 loss 再研究

SigLIP 的成功启发对 contrastive loss 的重新思考:

- **DINO 变体** 用类似思路
- **Masked Autoencoder** 对比版本
- **各种 multi-modal contrastive 方法**

### 3. 小团队能做 SOTA VLM

SigLIP 的小 batch 友好让**学术界和小公司**也能训练高质量视觉模型——不需要 Google 级别的算力。

### 4. Google 的视觉生态

Google 用 SigLIP 作为 PaliGemma、Gemini(推测)等多个产品的视觉组件。SigLIP 是 Google 开源视觉工作的主力。

---

## 局限

### 1. 只是 loss 改进,不改 encoder

SigLIP 的改进主要在 training efficiency,**encoder 本身仍是 ViT**。对需要精细视觉感知(OCR、spatial reasoning)的任务,SigLIP 的上限和 CLIP 相近。

### 2. Bias $b$ 的调参

$b$ 初始化敏感,不同 batch size / data 需要调。虽然论文给了默认值,但实际使用需要验证。

### 3. Downstream 任务的相对优势不大

在某些下游任务,SigLIP 只比 CLIP 好 1-3%。差距没有 zero-shot ImageNet 那么明显。

### 4. 对噪声敏感

SigLIP 的 pair-wise 判断对数据噪声更敏感(每对都算 loss)。数据清洗的要求更高。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **去掉 batch 维度的归一化是关键简化**:Softmax 的 batch-wise 归一化是 CLIP 的核心,但也是效率和稳定性的瓶颈。Sigmoid 的 pair-wise 独立判断是更优雅的方案
2. **Bias 调节是分类不平衡的标准手段**:SigLIP 用可学 bias 处理正负样本比 1:N 的严重不平衡——这是一个经典但常被忽略的技巧
3. **Loss function 改动带来的效率增益不输架构创新**:SigLIP 不改模型,只换 loss,性能和效率双提升。这提醒我们:训练目标的设计本身就是一个独立的研究方向
4. **SigLIP 取代 CLIP 成为 VLM 默认组件**:这个替换发生在 2024 年,是 VLM 领域最重要的"底层技术升级"之一。了解 SigLIP 就是了解现代 VLM 的基础设施
</callout>

---

## 延伸阅读

- [CLIP 深度解读]({% post_url 2026-04-24-CLIP-对比学习图文对齐深度解读 %}) —— SigLIP 的前身
- [LLaVA 深度解读]({% post_url 2026-04-24-LLaVA-视觉指令微调深度解读 %}) —— 使用 SigLIP 的下游应用
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193) —— 纯视觉对比学习
- [PaliGemma (Google 2024)](https://arxiv.org/abs/2407.07726) —— SigLIP 的旗舰应用
- [SigLIP 2 (Tschannen et al., 2025)](https://arxiv.org/abs/2502.14786) —— 改进版
