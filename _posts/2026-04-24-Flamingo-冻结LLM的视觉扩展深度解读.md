---
title: "Flamingo — 把视觉插进冻结的 LLM,开创 few-shot 多模态学习"
date: 2026-04-24 22:00:00 +0800
categories: [Pretraining, VLM]
tags: [flamingo, perceiver-resampler, few-shot, alayrac-2022]
math: true
---

## 基本信息

- **作者**: Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc 等
- **机构**: DeepMind
- **发表**: NeurIPS 2022
- **arXiv**: [2204.14198](https://arxiv.org/abs/2204.14198)

## 一句话总结

DeepMind 的 **Flamingo**——**第一个真正展示"few-shot 多模态学习"能力**的 VLM。核心设计:**冻结预训练 LLM(Chinchilla 70B),在 transformer 层之间插入 "gated cross-attention" 模块,让文本 token 可以选择性地 attend 到图像 token**——门控初始化为 0,保证初始状态下模型行为 = 原 LLM。配合 **Perceiver Resampler**(可变分辨率图像 → 64 固定 token)处理视觉,Flamingo 在 16 个 VQA / captioning 任务上**超过之前所有 fine-tuned 模型的 few-shot 成绩**。80B 总参(其中 10B 可学 = cross-attention + resampler)。Flamingo 是 VLM 方向的标志性工作,证明了"**冻结 LLM + 小 bridge**"的可行性,启发了 BLIP-2、LLaVA 等后续方案。

![Flamingo 的整体架构:视觉 encoder + Perceiver Resampler 压缩图像到 64 个 token,然后在冻结 LLM 的每几层之间插入 gated cross-attention,让文本 token "看到" 视觉信息。Gate 初始为 0 保证初始状态等同原 LLM。](/assets/img/flamingo/x1.png)
_Figure 1:Flamingo 的"冻结 LLM + 可学 cross-attention"架构_

---

## 背景:从"单图 VLM"到"多图 few-shot"

### 2022 年 VLM 的局限

- **CLIP**:只做 retrieval,不生成
- **VisualBERT / VinVL** 等:要 fine-tune per task
- **Zero-shot capability 有限**:给几个 image-text 示例不能帮助 task

**真正的 few-shot 多模态学习**(类比 GPT-3 的 in-context learning)还没实现。

### Flamingo 的目标

类比 GPT-3:给几个 (image, caption) 例子当 prompt,model 能做新 image 的同类任务——**without fine-tuning**。

如:
```
Image 1: [photo of cat] "a gray cat sitting on a chair"
Image 2: [photo of dog] "a brown dog running in the park"  
Image 3: [photo of bird] → ?
```

模型应该输出描述 bird 的类似句子。

---

## 核心机制

### 架构 1:Perceiver Resampler

![Perceiver Resampler:变尺寸的图像 feature(可能 1000+ token)通过 cross-attention 到 64 个 learnable queries,压缩到固定 64 token 供 LLM 使用。这是视觉侧的信息瓶颈。](/assets/img/flamingo/x2.png)
_Figure 2:Perceiver Resampler——固定 64 token 的视觉压缩_

不同分辨率 / 不同视频帧数的图像产出不同数量的 features——但 LLM 需要**固定数量的 token**。

**Perceiver Resampler**:

- **N 个 learnable query**(N=64)
- **Cross-attention**:queries 从 vision features 吸收信息
- 无论输入多少 feature,输出始终是 64 token

这与 BLIP-2 的 Q-Former 思想完全一致(Q-Former 可以看作 Resampler 的后继)。

### 架构 2:Gated Cross-Attention

![Gated Cross-Attention Layer:原 LLM 的 attention 不动,在其前/后插入一个新 cross-attn 层,attend 到视觉 token。Gate $\tanh(\alpha)$ 初始为 0,保证初始 = 原 LLM。训练中 gate 逐渐打开,视觉信息流入文本。](/assets/img/flamingo/x3.png)
_Figure 3:Gated Cross-Attention 的 0-init 设计_

关键设计 —— 在冻结 LLM 的每几层之间插入**一个可学 cross-attention layer**:

$$
h \leftarrow h + \tanh(\alpha) \cdot \text{CrossAttn}(h, \text{visual\_tokens})
$$

- $\alpha$ 是可学标量 **gate**,初始化为 **0**
- 初始时 $\tanh(\alpha) = 0$,所以 cross-attention 贡献为 0——模型行为 **完全等同于原 LLM**
- 训练中 $\alpha$ 逐渐非零,视觉信息开始流入
- 这个"**0 初始化的残差**"让训练稳定,不会"破坏"预训练 LLM 的能力

这是 Flamingo 最精妙的工程细节——**继承 LLM 全部能力的同时获得视觉能力**。

### 架构 3:冻结 LLM

Flamingo 的 **Chinchilla 70B LLM 完全冻结**。可学的只有:

- Perceiver Resampler(~100M)
- Gated Cross-attention layers(~10B)

总计 **~10B 可训练参数**。相比 Flamingo 80B 总参,训练的只有 **12.5%**。

### Interleaved image-text 训练数据

![训练数据是"交错图文"格式:text token1, text token2, <image 1>, text, <image 2>, text...。这模拟真实网页的图文混排,让 LLM 学会在文本流中处理图像。](/assets/img/flamingo/x1.png)
_Figure 4:Interleaved 图文数据格式_

关键:Flamingo **不只是训练 caption**,而是训练 **interleaved sequence**——图像和文本交替出现。这让模型能:

- 理解**多图上下文**
- 支持 **few-shot** image-text prompt
- 处理**视频**(把视频当成一系列图像)

数据来源:**M3W dataset**(DeepMind 自建)—— 1.85 亿 webpage,3 亿图像。

---

## 训练细节

- **Base LLM**:Chinchilla 70B(也训了 9B、3B、1.4B 小版本)
- **Vision encoder**:NFNet-F6(ResNet-like,CLIP 风格对比学习 pretrained)
- **Dataset**:M3W + ALIGN + LTIP
- **Compute**:几千 TPU-day

---

## 实验结果

### Few-shot 多模态的突破

在 16 个 vision-language benchmark 上:

| 任务 | Zero-shot | 4-shot | 32-shot | Fine-tune SOTA |
|------|-----------|--------|---------|---------------|
| COCO Captioning | 73 | 89 | 96 | 105 |
| VQAv2 | 56 | 63 | 67 | 74 |
| OKVQA | 45 | 50 | 52 | 57 |
| TextVQA | 35 | 40 | 47 | 48 |

**Few-shot Flamingo 常常超过 fine-tuned 专门模型**——第一次实现这点。

**32-shot VQAv2 67%** 对比当时 zero-shot 20-30% 是数倍提升。

### 视频理解

Flamingo 支持视频(每秒 1 帧)——处理多达 32 帧。

- **VATEX** (video captioning):**Flamingo 新 SOTA**
- **Kinetics-Action**:视觉动作识别也 few-shot 领先

### 对话能力

Flamingo 虽然训的是 caption + VQA 数据,但**涌现出对话能力**:

- 可以和用户讨论图像细节
- 多轮 QA 追问

这启发了后续 Visual Chat、LLaVA 等工作。

---

## 历史影响

### 1. 证明"冻结 LLM + 可学 bridge" 可行

Flamingo 之前,大家倾向于 full fine-tune。Flamingo 证明**冻结 LLM 不会损失多少能力**,且训练便宜——这启发了所有后续的 parameter-efficient VLM:

- BLIP-2(Q-Former)
- LLaVA(MLP projector)
- InternVL 的多阶段训练

### 2. Gated Cross-Attention 思想

"**0-init 的残差式新能力模块**"思想被广泛借鉴:

- **LoRA**(用 0-init 残差微调)
- 各种 **adapter** methods
- **Gemini**(推测)用类似的视觉扩展

### 3. Perceiver Resampler 的普及

"**可学 query + cross-attention 压缩**"成为 VLM 的标准组件:

- BLIP-2 Q-Former 是其演化
- Qwen-VL 用类似
- 各种 video VLM 用这个机制处理多帧

### 4. Interleaved 训练数据

Flamingo 之前,大家用 (image, caption) pair。Flamingo 的 interleaved format 被 **OpenFlamingo、IDEFICS、Idefics2** 等复现。

### 5. 没开源——但思想普及

**DeepMind 没开源 Flamingo**(可训代码、数据都没公开)。但开源社区复现(OpenFlamingo 2023, IDEFICS 2023)让思想广泛传播。

---

## 局限

### 1. Compute 需求大

80B 总参 + 10B 可训——小团队无法复现。OpenFlamingo 等复现版本规模小得多(9B、3B)。

### 2. 架构复杂

gated cross-attention + Perceiver Resampler 的工程复杂度高。**LLaVA 后来用 1 层 MLP 就达到类似效果**——简单胜出。

### 3. Fine-tune 才真正 SOTA

论文说 "few-shot 接近 fine-tune",但 **在 1-shot / 0-shot 上仍 gap 20+**。Fine-tune 仍是真正的上限。

### 4. 未开源

这让它的具体 recipe 无法被完整复现。相比 OpenFlamingo 的开源版本,原 Flamingo 的实际细节仍有 mystery。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Few-shot multimodal ICL 是可以实现的**:Flamingo 首次证明 GPT-3 的 few-shot learning 可以扩展到视觉——给几个例子,模型就能做新任务。这个能力是 VLM 能成为"通用视觉助手"的基础
2. **0-init 残差是继承能力的关键**:新能力模块用 $\tanh(\alpha)$ 门控,$\alpha=0$ 初始化——让训练初期模型行为完全等同原 LLM,然后逐渐"打开"新能力。这个思想被 LoRA、各种 adapter 继承
3. **冻结大 LLM + 训练小 bridge 是参数高效的正确方向**:Flamingo 可训参数只有 12.5%,但能力惊人。这让 VLM 训练成本降到可接受程度,开源 VLM 的爆发都是这个路线的延伸
4. **Interleaved 图文数据 > Caption pair**:模拟真实网页的图文混排训练,让模型学到"在文本流中处理图像"——这比 "(image, caption) pair" 更接近真实使用场景
</callout>

---

## 延伸阅读

- [CLIP 深度解读]({% post_url 2026-04-24-CLIP-对比学习图文对齐深度解读 %}) —— Flamingo 的视觉基础
- [BLIP-2 深度解读]({% post_url 2026-04-24-BLIP-2-Q-Former视觉语言深度解读 %}) —— 继承 Resampler 思想
- [LLaVA 深度解读]({% post_url 2026-04-24-LLaVA-视觉指令微调深度解读 %}) —— 极简路线对比
- [OpenFlamingo (Awadalla et al., 2023)](https://arxiv.org/abs/2308.01390) —— 开源复现
- [IDEFICS (HuggingFace, 2023)](https://huggingface.co/blog/idefics) —— Flamingo 思想的开源 VLM
