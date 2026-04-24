---
title: "BLIP-2 — 用 32 个可学习 query 桥接视觉和 LLM,Q-Former 范式奠基"
date: 2026-04-24 21:45:00 +0800
categories: [Pretraining, VLM]
tags: [blip-2, q-former, vlm, li-2023]
math: true
---

## 基本信息

- **作者**: Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi
- **机构**: Salesforce Research
- **发表**: ICML 2023
- **arXiv**: [2301.12597](https://arxiv.org/abs/2301.12597)

## 一句话总结

Salesforce 提出 **BLIP-2**——通过一个巧妙的**小型 Transformer bridge** 叫 **Q-Former**(Querying Transformer),**在冻结的 vision encoder 和冻结的 LLM 之间做参数高效对齐**。Q-Former 只有 **32 个可学习的 query token**,通过两阶段训练(表示学习 + 生成学习)成为视觉和语言的"信息瓶颈"——输出 32 个 token 给 LLM 当作"图像词"。关键优势:**只训 188M 的 Q-Former,不动 vision encoder(1B 级 ViT)和 LLM(11B OPT)**,大大降低训练成本。BLIP-2 是 2023 年最具影响力的 VLM 架构之一,确立了"Q-Former 范式",虽然后来被 LLaVA 的 MLP projector 在开源社区超越,但其**参数高效对齐思想**至今影响深远。

![BLIP-2 的架构:冻结的 vision encoder + 冻结的 LLM + 中间插入可学习的 Q-Former。32 个 learnable queries 通过 cross-attention 从图像吸收信息,然后作为 "visual token" 喂给 LLM。](/assets/img/blip2/x1.png)
_Figure 1:BLIP-2 的三组件架构_

---

## 背景:训练完整 VLM 太贵

### 2022-2023 年的 VLM 困境

- **Flamingo**(DeepMind):从头训练一个 80B 模型,成本几千万美元
- **CoCa**:也是 scratch 训,billion 级 compute

这些都不实用。开源社区的问题:**能不能利用已有的强 vision encoder 和强 LLM,只训一个小的 bridge?**

### 参数高效对齐的想法

- Vision encoder 已经很强(CLIP、EVA-ViT)
- LLM 已经很强(OPT、FlanT5、LLaMA)
- 只需要一个小 module 把两者对齐

BLIP-2 的贡献:**设计这个"小 module"的最佳形式**。

---

## 核心机制:Q-Former

### 什么是 Q-Former

Q-Former 是一个 **轻量级 Transformer**(类似 BERT-base 大小,188M 参数),关键特点:

- **32 个可学习的 query token**(像 slot 一样)
- 这些 query 通过 **cross-attention 从 vision encoder 吸收信息**
- 输出 32 个 token 作为 "visual prompt" 给 LLM

![Q-Former 的内部结构:32 个 learnable queries 和 image features 做 cross-attention,得到 32 个 compact 的视觉 token。这些 token 通过一个线性层投到 LLM 的 embedding 空间。](/assets/img/blip2/x2.png)
_Figure 2:Q-Former 内部架构_

### 为什么 32 个 query

设计考量:

- Vision encoder 输出 **257** 个 token(ViT-L/14 在 224×224 下)
- 直接喂给 LLM 太多——浪费 context
- 32 个 query 做"**信息压缩 bottleneck**"——逼 Q-Former 提炼关键视觉信息
- 类似 Perceiver Resampler 的思想(Flamingo)

---

## 两阶段训练

### Stage 1: Vision-Language Representation Learning

![Stage 1 用三个 loss 同时训 Q-Former:ITC(图像-文本对比)、ITM(图像-文本匹配)、ITG(图像为条件的文本生成)。三个 loss 共享 Q-Former 参数但用不同 attention mask。](/assets/img/blip2/x3.png)
_Figure 3:Stage 1 的三任务训练_

在这阶段,**不涉及 LLM**,只训 Q-Former:

- **ITC (Image-Text Contrastive)**:类似 CLIP,让 matching pair 相近
- **ITM (Image-Text Matching)**:二分类,判断是否匹配
- **ITG (Image-grounded Text Generation)**:给图生成 caption

三任务共享 Q-Former 但使用**不同 attention mask**(image-only attention、bidirectional、causal)。

数据:
- COCO, VG, CC3M, CC12M, SBU, LAION-400M
- 共 **~129M 图文对**

### Stage 2: Vision-to-Language Generative Learning

![Stage 2 把 Q-Former 接到冻结的 LLM 前。Q-Former 的输出作为 soft prompt,LLM 只需要做 captioning / VQA。Q-Former 的参数微调,LLM 完全冻结。](/assets/img/blip2/x4.png)
_Figure 4:Stage 2 接入 LLM_

现在加入 LLM:

- Q-Former 输出 32 个 token,经过 linear projection 进入 LLM embedding 空间
- LLM **完全冻结**,只训 Q-Former(188M 参数)
- 训练目标:给图像(和可选文本 prompt)生成正确的 caption / answer
- 数据:与 Stage 1 类似

**关键**:通过 Stage 1 的 Q-Former 预训练,Stage 2 收敛非常快——只需少量数据就能把 Q-Former 的输出和 LLM "对齐"。

---

## 实验结果

### 参数效率

| 模型 | 可训练参数 | 总参数 |
|------|------------|--------|
| Flamingo-80B | 10B | 80B |
| BLIP-2 (OPT-2.7B) | **188M** | 3.1B |
| BLIP-2 (OPT-6.7B) | **188M** | 7.8B |
| BLIP-2 (FlanT5-XXL 11B) | **188M** | 12.1B |

**可训练参数降低 50×**,同时在很多 benchmark 上和 Flamingo 持平。

### Zero-shot VQA

- **VQAv2**:BLIP-2 (FlanT5-XXL) **65.0%**,超过 Flamingo-80B 的 56.3%
- **OKVQA**:**45.9%**,超过 Flamingo 50.6%(近似)
- **GQA**:**44.7%**

### Captioning

- **COCO CIDEr**:BLIP-2 **145.8**,匹敌专门 captioning 模型

---

## 历史影响

### 1. 确立 "Q-Former 范式"

BLIP-2 之后,很多 VLM 采用 Q-Former 或类似 bottleneck:

- **InstructBLIP**(2023):BLIP-2 + instruction tuning
- **MiniGPT-4**(2023):类似架构
- **MiniGPT-v2**:Q-Former 变体

但这个范式后来被 LLaVA 的 MLP projector **在开源社区超越**——MLP projector 更简单,数据驱动效果更好。

### 2. 参数高效对齐的思想

"**只训一个小 bridge,不动 vision encoder 和 LLM**"的思想被广泛采用:

- LLaVA(projector)
- mPLUG-Owl
- InternVL 的训练策略
- 各种 LoRA-based VLM

### 3. 冻结 LLM 的证明

BLIP-2 证明:**完全冻结的 LLM,通过合适的 visual prompt 就能做 VQA、caption**。这启发了后续工作重新思考"是否真的需要 LLM fine-tune"。

### 4. Salesforce 的 VLM 血统

BLIP 系列(BLIP、BLIP-2、InstructBLIP、X-InstructBLIP)展示了 Salesforce 在 VLM 方向的持续投入。这些工作为 VLM 研究提供了重要方法论。

---

## 局限

### 1. 两阶段训练复杂

Stage 1 + Stage 2 的 pipeline 比 LLaVA 的 "pretrain projector + instruction tune" 复杂得多。训练时间和 debugging 成本更高。

### 2. 32 tokens 的信息瓶颈

32 个 token 对某些 fine-grained 视觉任务是太少的。**OCR、详细图表分析**上 BLIP-2 受限。

### 3. 指令遵循能力不足

BLIP-2 本身没有 instruction tuning——只能做 VQA 和 caption。**要做真正的 assistant 需要 InstructBLIP 等后续工作**。

### 4. 被 LLaVA 范式取代(某种程度)

在开源 VLM 主流,Q-Former 被 MLP projector 取代。原因:
- MLP projector 实现更简单
- 指令数据驱动效果更好
- Q-Former 的"信息瓶颈"在某些场景是 disadvantage

不过 **某些 VLM**(InternVL 一些变体)仍在用 Q-Former 思想。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **参数高效对齐是 VLM 的实用路线**:不从头训,用现成的强 encoder 和强 LLM,只训一个小 bridge——这是所有现代开源 VLM 的基本思路
2. **可学习 query 是强大的信息 bottleneck**:32 个 learnable queries 通过 cross-attention "提炼" 图像——这个设计在 Flamingo 的 Perceiver、BLIP-2 的 Q-Former、Fuyu 等都有体现
3. **两阶段训练的原则:先对齐表征,再学指令**:BLIP-2 的 Stage 1/2 分离,先把视觉-语言对齐好,再做下游任务。这个思想在 LLaVA 的两阶段训练、InternVL 的多阶段训练中延续
4. **简单 vs 复杂:数据驱动时代简单胜出**:Q-Former 是精心设计的复杂架构,LLaVA 用一个 MLP 就赢了。这再次证明:**数据时代,架构复杂度常常不是优势**
</callout>

---

## 延伸阅读

- [LLaVA 深度解读]({% post_url 2026-04-24-LLaVA-视觉指令微调深度解读 %}) —— 对立的极简路线
- [Flamingo 深度解读]({% post_url 2026-04-24-Flamingo-冻结LLM的视觉扩展深度解读 %}) —— 类似的 frozen LLM 思想
- [CLIP 深度解读]({% post_url 2026-04-24-CLIP-对比学习图文对齐深度解读 %}) —— BLIP-2 的视觉基础
- [InstructBLIP (Dai et al., 2023)](https://arxiv.org/abs/2305.06500) —— BLIP-2 的 instruction-tuning 版本
- [BLIP (Li et al., 2022)](https://arxiv.org/abs/2201.12086) —— BLIP-2 的前身
