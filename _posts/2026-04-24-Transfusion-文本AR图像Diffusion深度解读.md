---
title: "Transfusion — 文本 AR + 图像 Diffusion 联合训练,打破 Omni 的路线二选一"
date: 2026-04-24 23:00:00 +0800
categories: [Pretraining, Omni, Multimodal]
tags: [transfusion, hybrid, diffusion, meta-2024]
math: true
---

## 基本信息

- **作者**: Chunting Zhou, Lili Yu, Arun Babu 等(Meta & Waymo)
- **机构**: Meta, Waymo
- **发表**: arXiv 2024-08
- **arXiv**: [2408.11039](https://arxiv.org/abs/2408.11039)

## 一句话总结

Meta 的 **Transfusion**——一个**打破"离散 vs 连续二选一"** 的 Omni 架构创新:**文本用 AR loss(next-token prediction),图像用 Diffusion loss,两者共享同一个 Transformer**。具体做法:文本 token 走 causal attention + 交叉熵 loss;图像 latent(连续)走 bidirectional attention + diffusion denoising loss;两种 loss 加权相加作为训练目标。相比 Chameleon 的"万物皆离散 token",Transfusion **在生成图像质量上显著更好**(因为 diffusion 本身就是图像生成的 SOTA 方法)而**保留文本的生成能力**。7B Transfusion 在图像生成 FID 上打败 Chameleon-34B,在文本理解上匹配 LLaMA 2。Transfusion 是 "Hybrid" Omni 路线的代表,被 Show-o、Qwen2.5-Omni 等后续工作继承。

![Transfusion 的训练目标:文本 token 上的位置计算 AR loss,图像 patches 上的位置计算 diffusion loss。同一 Transformer 处理,loss 加权。这是 "token-level 混合训练目标" 的巧思。](/assets/img/transfusion/x1.png)
_Figure 1:Transfusion 的混合训练目标_

---

## 背景:Chameleon 路线的痛点

Chameleon 把图像 token 化后做 next-token prediction——**统一但图像质量不够**。原因:

- VQ-VAE 的离散化损失不可逆
- Autoregressive raster scan 不是图像生成的最佳方式(比 diffusion 差)
- Codebook size 8K 的上限限制表达力

同时 Stable Diffusion 证明 **diffusion 是生成图像的最佳方法**,但 diffusion 不擅长文本生成(text 是离散的)。

**Transfusion 的关键问题:能不能让一个模型,对两种数据用两种不同的 training objective?**

---

## 核心机制

### 两种 loss 共存

对一个 sequence 包含文本和图像:

```
[Text token] [Text token] ... [IMG start] [image latents] [IMG end] [Text token] ...
```

- **Text positions**:causal attention,cross-entropy loss
- **Image positions**:bidirectional attention,diffusion denoising loss

训练目标:

$$
\mathcal{L} = \mathcal{L}_{\text{AR}} + \lambda \cdot \mathcal{L}_{\text{Diff}}
$$

同一 Transformer 的 forward 可以计算两种 loss——相加优化。

### 图像 latent 处理

图像不是离散 token,而是**连续的 VAE latent**:

- Image → VAE encoder → 连续 latent tensor $(h, w, c)$
- 展平为序列后加 noise
- Transformer 预测 noise(standard diffusion)

### Attention Mask 的巧思

![Transfusion 的 attention mask:文本位置只能看过去(causal);图像位置可以双向看所有 image tokens + 前面的文本(但之后的文本不行)。这个混合 mask 让模型既保持 AR 文本生成,又保持图像的双向处理。](/assets/img/transfusion/x2.png)
_Figure 2:混合 attention mask_

- 文本 token:只看前面(包括文本 + 更早的图像)
- 图像 token:双向看图像内部,并可以看过去的文本

这个 mask 让一个 Transformer 自然承担两种任务。

### 推理

- **生成文本**:标准 AR sampling
- **生成图像**:遇到 `[IMG]` 后,运行 diffusion 50-100 步
- **交错 output**:在 AR 流程中根据需要"切入" diffusion

![Transfusion 推理时的 "混合采样":文本用 AR sampling,图像用 diffusion 步骤。两种采样在同一输出流中自然交替。](/assets/img/transfusion/x3.png)
_Figure 3:Transfusion 的混合采样_

---

## 实验结果

### 图像生成

| 模型 | 参数 | FID (COCO) | GenEval |
|------|------|------------|---------|
| Chameleon-7B | 7B | 26.7 | 0.39 |
| Chameleon-34B | 34B | 19.3 | 0.47 |
| Stable Diffusion 3 | 2B | ~15 | 0.62 |
| **Transfusion-7B** | **7B** | **16.8** | **0.63** |

Transfusion-7B **FID 和 SD3 相当**,比同规模 Chameleon 好很多。

### 文本理解

对比纯 LLM baseline:

- **MMLU**:Transfusion-7B 60.1,LLaMA 2 7B 46.8(Transfusion 有更多训练 compute)
- **HumanEval**:Transfusion 25.0,LLaMA 2 13.5

文本能力持平或略强——**图像能力的加入没有伤害文本**。

### Scaling

Transfusion 对比 Chameleon 在相同 compute 下:

- 每单位 compute,Transfusion 的图像 FID 比 Chameleon 好 **~50%**
- 文本 loss 几乎相同

即**混合 loss 的 scaling 比纯 AR 好**——对图像部分 diffusion objective 更高效。

---

## 工程影响

### 1. Hybrid 路线的代表作

Transfusion 之后,"**AR + Diffusion 混合**"成为 Omni 研究的重要方向:

- **Show-o**(NUS 2024):类似思路,更小规模
- **Janus-Pro**(DeepSeek 2024):另一种混合方式
- **Qwen2.5-Omni**:Thinker-Talker 双轨

### 2. "不同数据不同 loss"范式

Transfusion 证明一个 Transformer 可以承担多种训练目标——只要 attention mask 和 loss 设计得当。这个思路可以推广到:

- 音频(可能用 diffusion 或 flow matching)
- 视频(按 patch 做 diffusion)
- 各种结构化数据

### 3. 验证 diffusion 对 LLM 训练栈友好

Diffusion 不只是图像工具,可以无缝集成到 Transformer 训练 pipeline——**这让大模型做图像生成的经济性提升**。

### 4. 推动 Omni 评测发展

统一模型需要 **综合评测**(文本 + 图像质量)。Transfusion 等工作推动了 Omni-bench 等新 benchmark。

---

## 局限

### 1. 推理成本高

图像部分需要 50-100 步 diffusion——对比 AR 生成 **慢 50-100×**。

### 2. 实现复杂

两种 attention mask、两种 loss、两种 sampling——工程实现比纯 AR 复杂得多。

### 3. 文本和图像 balance

$\lambda$ 的选择影响两边能力的 trade-off。太偏图像文本能力降,反之亦然。需 careful 调参。

### 4. 视频仍未完美支持

Transfusion 主要针对图像。视频需要时间维的 diffusion,更复杂。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **不同数据,不同 loss**:文本是离散的,用 AR loss;图像是连续的,用 diffusion loss。同一个 Transformer 可以承担两者——让每种数据用最适合的训练目标
2. **Attention mask 是 Hybrid 架构的关键**:文本 causal,图像 bidirectional,跨模态可以看但有约束。Mask 的精细设计决定了 Hybrid 能否 work
3. **Chameleon 和 Transfusion 代表 Omni 两条路线**:万物皆 token(离散统一)vs 按模态选 loss(混合)。目前 Transfusion 在图像质量上领先,但 Chameleon 的简洁也有优势
4. **Diffusion 可以无缝融入 Transformer**:过去 diffusion 和 LLM 是两个独立训练栈。Transfusion 证明它们可以合二为一——这是 Omni 工程基础设施的突破
</callout>

---

## 延伸阅读

- [Chameleon 深度解读]({% post_url 2026-04-24-Chameleon-早期融合全模态深度解读 %}) —— 纯离散路线对比
- [Stable Diffusion 3 / MMDiT (2024)](https://arxiv.org/abs/2403.03206) —— Diffusion 的 Omni 前驱
- [Show-o (NUS 2024)](https://arxiv.org/abs/2408.12528) —— 类似 Hybrid 路线
- [VAR (2024)](https://arxiv.org/abs/2404.02905) —— 新的离散路线
- [GPT-4o 官方介绍](https://openai.com/index/hello-gpt-4o/) —— 闭源 Omni 代表
