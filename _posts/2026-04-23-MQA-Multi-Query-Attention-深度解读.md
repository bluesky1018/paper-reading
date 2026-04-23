---
title: "MQA — Multi-Query Attention:一行公式拉开现代 LLM 推理优化的序幕"
date: 2026-04-23 23:20:00 +0800
categories: [Attention, Inference Optimization]
tags: [mqa, multi-query-attention, kv-cache, shazeer, decoding, inference]
math: true
---

## 基本信息

- **作者**: Noam Shazeer
- **机构**: Google
- **发表**: arXiv 2019-11(技术报告,5 页)
- **arXiv**: [1911.02150](https://arxiv.org/abs/1911.02150)(标题 *Fast Transformer Decoding: One Write-Head is All You Need*)

## 一句话总结

提出 **Multi-Query Attention (MQA)**——把 Multi-Head Attention 里**每头独立的 K 和 V 全部共享为一份**,只保留 Q 的多头独立。数学上极简,实现上极简,论文只有 5 页,连一张图都没有。但这一个改动**把 decode 阶段的 KV cache 带宽瓶颈几乎一次性解决**,在 decode 速度上带来 10× 以上的加速,为后来的 GQA、MLA、整个 LLM 推理栈铺平道路。

## 背景:Decode 阶段的隐藏瓶颈

Transformer 训练(包括 encoder 或 decoder 的 teacher-forcing)是**高度并行**的——整段序列 $N$ 个 token 一次性做 attention,用 matmul 打满 GPU。

但**自回归 decode** 完全不同:

- 每步只生成 1 个新 token
- 要做 attention 必须读全部历史 token 的 K、V
- KV cache 按 $(B, H, N, d_h)$ 存储,**每生成 1 个 token 就要把整个 KV 读一次**

Shazeer 的论文指出:**decode 阶段 GPU 算力大量空转,真正的瓶颈是 HBM → SRAM 搬 KV 的带宽**。

### 瓶颈数字

对 Transformer-big($H=16, d_h=64$,seq_len=1024):

- 每 decode 步需要加载约 2 MB 的 K、V
- A100 的 HBM 带宽 ~1.5 TB/s,理论上 decode 速度 ~ 750K 步/秒
- 实际测出只有 ~50K 步/秒——因为还要有计算时间、通信延迟等

**如果能把 KV 压小 $H$ 倍,decode 带宽瓶颈直接松 16 倍**。

---

## 核心机制:一个头写,多个头读

Shazeer 的观察是:**K 和 V 的多头并不是"关键"的——只有 Q 多头才让不同 head 关注不同内容**。

### MHA 原版

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)
$$

每头独立 $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_\text{model} \times d_h}$。KV cache 大小 = $2 H d_h N$。

### MQA 改动

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\; K W^K,\; V W^V)
$$

注意:

- $W_i^Q$ 仍然是**每头独立**(保留多头的选择性关注能力)
- $W^K, W^V$ **只有一份**,所有 head 共享

KV cache 大小 = $2 d_h N$——**压缩 $H$ 倍**。

### 为什么理论上可行?

对 Attention 输出做展开可以看到:

$$
\text{out}_i = \text{softmax}\!\left(\frac{(Q W_i^Q)(K W^K)^\top}{\sqrt{d_h}}\right) V W^V
$$

不同 head 在 Q 侧用不同投影,在 K、V 侧共享。这意味着每个 head 还是用自己的"视角"去查询,只是查的是同一份键值。Shazeer 的直觉是:**多头的表达力主要来自 Q 的多样投影,K 和 V 的多头更多是冗余。**

这个假设在实验上得到部分支持——MQA 的质量只略降于 MHA。

---

## 实验结果

论文在 WMT'14 英德翻译上对比:

| 模型 | BLEU (EN-DE) | Decoding Speed (µs / step) | 质量下降 |
|------|-------------|---------------------------|---------|
| MHA Transformer-big | 28.4 | 46 | — |
| **MQA Transformer-big** | 27.5 | **4.2** | -0.9 BLEU |

- **解码速度 ~11×**
- 质量下降 ~0.9 BLEU(可接受但非零)

这个 trade-off 当时看起来"**偏慢的单 head 速度换了一点质量**"。2019 年学界没怎么跟进——训练速度没变、翻译指标还下降、图都没有。

---

## 为什么 4 年后重焕生机

### 2023:大模型时代改变了经济学

2019 年 Transformer-big 的 decode 是毫秒级,不是瓶颈。2023 年情况彻底变了:

- 模型从 200M → 70B → 400B,KV cache 膨胀 1000 倍
- Chat 应用的**长上下文** + **高并发**使 KV cache 变成显存+带宽的主要消耗
- decode 速度直接决定 tokens/s → 用户体验 + 运营成本

此时 MQA 的"$H$ 倍 KV 压缩"突然从**可选优化**变成**必选项**。PaLM(2022)和 Falcon(2023)先用上,Llama 2 的 70B 则走了 MQA 的改进版——GQA。

### GQA:MQA 的补足

GQA(Ainslie 2023)发现 MQA 的 0.9 BLEU 质量下降在大模型下可能放大到更严重。提出"折中方案":不是全头共享,而是**分 $G$ 组**共享。$G=8$ 时质量几乎追平 MHA。

MQA 对应 $G=1$ 的极端。于是**MHA / GQA / MQA 形成一条可调轴**,工程师按质量-效率需求选点。

### MLA 和线性 attention:继续压 K/V

MLA(DeepSeek-V2,2024)把 K/V 进一步压到一个 **latent 向量**,不再是"多头共享"而是"低秩压缩"。
线性 attention / Mamba 则彻底不要 KV cache,改用固定大小的 state。

**所有这些工作都可以追溯到 Shazeer 2019 年那个一页纸的公式**。

---

## Shazeer 的这篇论文为什么是个时代性注脚

读这篇论文会有种奇妙的感觉:

- **5 页**,连一张图都没有
- 数学上只是把 $W_i^K, W_i^V$ 的下标 $i$ 去掉
- 2019 年没多少引用,很多人不知道有这篇

但 4 年后:

- LLaMA 2、Falcon、PaLM、Gemini 全系列使用 MQA 或 GQA
- "KV cache 是 decode 瓶颈"成为行业共识
- Shazeer 本人的命运也有趣——**后来的 Noam Shazeer 正是 Character.AI 创始人**,他的产品栈把 MQA/GQA 用到极致(见 [Character.AI Optimizing Inference 博客引介]({% post_url 2026-04-23-Character-AI-Optimizing-Inference-博客引介 %}))

这是一个很好的提醒:**影响力并不一定来自规模或新颖性,有时一个工程上简单而正确的洞察,就足以重塑一个行业**。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Decode 瓶颈在 KV 带宽,不在算力**——这条认识是 MQA 的起点,也是现代 LLM 推理栈的基本原理
2. **多头的表达力主要来自 Q**,K/V 的多头是冗余较大的——这个直觉启发了后续 GQA 的分组设计和 MLA 的 latent 压缩
3. **不起眼的简洁工作可能需要时间发酵**:MQA 写于 2019,要等 4 年才被大规模采用
4. **公式 + 几行代码的改动,比一张漂亮的架构图更能改变世界**:MQA 的实现差异只有几行 PyTorch,却是开启 GQA、MLA 等全条优化主线的起点
</callout>

---

## 延伸阅读

- [GQA 深度解读]({% post_url 2026-04-23-GQA-分组查询注意力深度解读 %}) —— MQA 的直接继承者,现代大模型事实标准
- [MLA / DeepSeek-V2 深度解读]({% post_url 2026-04-23-MLA-DeepSeek-V2-多头潜在注意力深度解读 %}) —— 把 MQA 的压缩思想推到极致
- [PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) —— KV cache 的系统层管理,与 MQA 互补
- [FlashAttention 深度解读]({% post_url 2026-04-23-FlashAttention-IO感知精确注意力深度解读 %}) —— 另一个解决 attention IO 瓶颈的关键工作
- [Character.AI Optimizing Inference 博客引介]({% post_url 2026-04-23-Character-AI-Optimizing-Inference-博客引介 %}) —— MQA 在产品级场景的综合应用
