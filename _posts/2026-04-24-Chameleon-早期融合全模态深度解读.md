---
title: "Chameleon — Meta 的统一 autoregressive 全模态模型,万物皆 token"
data: 2026-04-24 22:45:00 +0800
date: 2026-04-24 22:45:00 +0800
categories: [Pretraining, Omni, Multimodal]
tags: [chameleon, early-fusion, unified-model, meta-2024]
math: true
---

## 基本信息

- **作者**: Chameleon Team(Meta FAIR)
- **机构**: Meta FAIR
- **发表**: arXiv 2024-05
- **arXiv**: [2405.09818](https://arxiv.org/abs/2405.09818)

## 一句话总结

Meta 的 **Chameleon**——**第一个在 34B 规模上成功的"纯统一 autoregressive"全模态模型**。核心设计:**把图像也 tokenize**(VQ-VAE 产出离散 token),然后和文本 token **混在同一序列里**,用**一个 Transformer** 做 next-token prediction。这种 **early fusion** 架构彻底不做"视觉 encoder + LLM" 的拼接,而是让**同一模型原生同时理解和生成图像与文本**。在混合模态长文本任务上超过 Flamingo、IDEFICS 等 "adapter-based" VLM,且能**生成图像**(虽然质量不及专门的 Stable Diffusion)。Chameleon 是 GPT-4o 思路的开源先声,验证了"token 统一 + early fusion" 是一条可行的 Omni 路线,启发了后续 Transfusion、Emu3 等工作。

![Chameleon 的核心思路:图像用 VQ-VAE tokenize 成离散 token(如 1024 个 token 表示一张图),和文本 token 混在同一序列里,统一用 next-token prediction 训练。模型没有独立的视觉 encoder。](/assets/img/chameleon/x1.png)
_Figure 1:Chameleon 的统一 token 架构_

---

## 背景:VLM 的两条路线分歧

### 路线 A:Adapter / Late Fusion(主流)

- **CLIP + projector + LLM**(LLaVA)
- **Perceiver Resampler + LLM**(Flamingo)
- 视觉和语言有独立 encoder,拼接起来

优点:复用现成 LLM,训练便宜
缺点:**视觉只能"理解",不能"生成"**,两个分支难协同

### 路线 B:Token 统一(Chameleon 的选择)

- 把图像离散化为 token(像 Hugo Larochelle 的老 autoregressive image gen)
- 和文本 token 混用
- 一个 Transformer 处理全部

优点:**真正的全模态理解+生成**
缺点:训练困难、质量难保证——之前没成功过 34B 级别

**Meta 的 Chameleon 是第一个把 B 路线做到生产级质量的工作。**

---

## 核心机制

### 1. Image Tokenizer(VQ-VAE)

![Chameleon 的 image tokenizer:用 VQ-VAE 把 512×512 图像编码为 1024 个离散 token(codebook size 8192)。这些 token 和文本 token 共享一个词表。](/assets/img/chameleon/x2.png)
_Figure 2:Image Tokenizer + 统一词表_

- **VQ-VAE**:512×512 图 → **1024 个离散 token**(32×32 grid)
- **Codebook size**:8192
- Image token 和 text token **合并到一个词表**(text 32K + image 8K ≈ 40K)

模型看到的序列:

```
[BOS] text text [IMG] img_token × 1024 [/IMG] text text [EOS]
```

图像就像一段 "外星文字"——本质上和文本没有结构区别。

### 2. 架构

**没有独立的 vision encoder**——就是一个标准 Transformer decoder,只不过词表大了。

- **34B 参数**(也有 7B 版本)
- 标准 decoder-only(类似 LLaMA)
- RoPE + SwiGLU + RMSNorm
- **特殊 tokens**:`[IMG]` / `[/IMG]` 标记图像边界

### 3. 训练稳定性的挑战

Early fusion 的大挑战:**文本和图像 token 的分布差异大**,softmax 容易失衡。

Chameleon 用多个技巧:

- **QK-Norm**:防 attention logit 爆炸
- **Z-loss**:防 output logit 失控
- **Dropout after Norm**:某些位置加额外 dropout
- **Careful initialization**:小心的 layernorm scaling

论文详细讨论这些——因为 early fusion 失稳是出了名的难。

### 4. 训练数据

- **纯文本**:类似 LLaMA 2 数据
- **图文交错** (interleaved):网页数据,图像穿插在文本中
- **Image-caption pair**:COCO 等
- **Caption-image**(反向):促进 text-to-image 生成能力

总量:**4.4T 文本 tokens + 1.4B 图像**

### 5. 生成能力

![Chameleon 可以生成图像 + 文本的"交错" response:问题是文字,回答既可以有文字解释又可以在中间插入图片。这种"自然 interleave" 是 adapter-based VLM 做不到的。](/assets/img/chameleon/x3.png)
_Figure 3:Chameleon 的统一生成能力_

- 同一模型可以做:
  - **文本生成**(VQA, caption)
  - **图像生成**(text-to-image)
  - **图文交错生成**(描述混合图像和文本的长答案)

这是 **early fusion 最核心的优势**——adapter-based VLM 不能 directly 生成图像。

---

## 实验结果

### 1. 混合模态任务

Chameleon 专门测 **mixed-modal** 任务(输入输出都是图文混合):

- 创建图文混合长答案
- 从图生成类似风格的新图
- 多图长推理

在这些 benchmark 上 Chameleon 大幅超过 Flamingo / IDEFICS(它们根本不能做某些任务)。

### 2. 纯视觉理解

在纯 VQA / caption:

| 模型 | VQAv2 | COCO Cap |
|------|-------|----------|
| Flamingo-80B | 82.0 | 138 |
| IDEFICS-80B | 76.8 | 129 |
| **Chameleon-34B** | **83.0** | **140** |

**34B 击败 80B 的 adapter VLM**——token 统一路线的优势展现。

### 3. 图像生成

- COCO FID:~18(对比 SD-XL 的 ~10)
- **不如专门的图像生成模型**,但能在同一模型中产出

这是 early fusion 的 trade-off:多能性 vs 单任务质量。

---

## 工程影响

### 1. 证明 early fusion 的可行性

Chameleon 之前,early fusion 只在 < 1B 模型上验证过。**34B 规模成功**让业界认真对待这条路线。直接启发:

- **Transfusion**(Meta 2024-08)
- **Emu3**(BAAI 2024-10)
- **InternVL 3**(原生多模态)

### 2. GPT-4o 的开源先声

GPT-4o(2024-05)不久前发布,展示了"**原生全模态端到端**"。Chameleon 的 token 统一路线**几乎同时**出现——虽然架构可能不同,但思路一致。

### 3. 视觉 tokenization 的重新重视

VQ-VAE 在 DALL-E 时代就有,但 LLM 兴起后被 adapter 路线压制。Chameleon 让 **visual tokenization** 重新成为 Omni 研究的焦点。后续的 **VAR**(next-scale prediction)、**MAGVIT-v2** 等都是这条线。

### 4. 统一训练的稳定性 recipe

Chameleon 详细披露 early fusion 的 training recipe——**QK-Norm、Z-loss、careful init** 等。这个 recipe 被后续所有 unified multimodal 工作借鉴。

---

## 局限

### 1. 图像生成质量

不如 Stable Diffusion XL 等专用模型。分辨率也受限(512×512)。

### 2. 训练极难

Early fusion 的训练比 adapter 难得多——调参空间巨大、失稳风险高。Meta 都花了大功夫,更小团队更难。

### 3. 模态失衡

训练中文本和图像的 loss 分布差异大。需要 careful loss balancing——Chameleon 没给出很清晰的指导,留给后续工作解决。

### 4. Context 爆炸

图像 1024 token,一张高分辨率图 + 几轮对话就可能 8K+。Long-context 压力大。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Early fusion + token 统一是 Omni 的终极形态**:虽然目前质量不如 specialized 模型,但它的"一个模型 handle 一切"能力是 adapter 路线永远做不到的
2. **视觉 tokenization 重要性上升**:把图像压成 1024 个 token 是否合理?Codebook size 应该多大?这些问题的答案决定 Omni 模型的上限。后续 VAR、MAGVIT-v2 等正在深入
3. **早期融合需要专门的训练稳定性**:QK-Norm + Z-loss + careful init 是必备——这比 adapter VLM 训练难一个量级
4. **Chameleon 是 2024 年 Omni 开源的起点**:开启了 Transfusion、Emu3、Show-o、Qwen2.5-Omni 等一系列工作。理解它就理解了 2024 年 Omni 技术路线的起点
</callout>

---

## 延伸阅读

- [Transfusion 深度解读]({% post_url 2026-04-24-Transfusion-文本AR图像Diffusion深度解读 %}) —— Hybrid 路线
- [LLaVA 深度解读]({% post_url 2026-04-24-LLaVA-视觉指令微调深度解读 %}) —— Adapter 路线对比
- [VQ-VAE-2 (van den Oord et al., 2019)](https://arxiv.org/abs/1906.00446) —— Image tokenizer 的基础
- [VAR (Tian et al., 2024)](https://arxiv.org/abs/2404.02905) —— 新的 image autoregressive 范式
- [Emu3 (BAAI 2024)](https://arxiv.org/abs/2409.18869) —— Next-token prediction 统一三模态
