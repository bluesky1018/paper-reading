---
title: "Transformer-XL — 用段级递归 + 相对位置编码突破长上下文"
date: 2026-04-23 22:55:00 +0800
categories: [Attention, Long Context]
tags: [transformer-xl, segment-recurrence, relative-position, long-context, dai-2019]
math: true
---

## 基本信息

- **作者**: Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov
- **机构**: CMU, Google Brain
- **发表**: ACL 2019
- **arXiv**: [1901.02860](https://arxiv.org/abs/1901.02860)
- **代码**: [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl)

## 一句话总结

提出 **Transformer-XL**——通过两个关键技巧突破原始 Transformer 的上下文限制:
1. **段级递归(Segment-level Recurrence)**:把前一个 segment 的隐状态**缓存下来**,下一 segment 的 self-attention 可以"回看"前面,有效上下文长度扩大数百倍
2. **相对位置编码(Relative Positional Encoding)**:取代绝对正弦编码,让 attention 分数只依赖**相对位置**,跨段时不会位置冲突

这两条技术把语言建模 SOTA 从"几百 token"一口气推到"**几千 token**",是 RoPE、ALiBi 等后续位置编码工作的直接前辈,也是 **第一个真正严肃对待"长上下文"** 的 Transformer 变体。

![原始 Transformer LM 的训练与评估行为:每个 segment 独立处理,跨 segment 完全切断——训练时固定长度的 attention 窗口,评估时每滑动一个 token 就要从头跑一次长 segment,计算浪费巨大。](/assets/img/transformer-xl/x1.png)
_Figure 1(a/b):原始 Transformer LM 的段独立处理——训练时跨段切断,评估时需要每滑一步重算整段_

---

## 背景:原始 Transformer 的两个长文本缺陷

### 缺陷 1:固定长度上下文,跨段无连接

vanilla Transformer LM (2018 年代做语言建模) 的做法:把长文档切成固定长度的 segment (如 512),每个 segment 独立训练,**跨段完全无信息流**。

- 第 1 段末尾的 token 无法看到第 2 段开头的 token
- 即使两个段语义紧密相连,模型也像**看一本书只看一章**,段与段独立
- **"上下文碎片化"** 导致长距离依赖无法建模

### 缺陷 2:评估的滑窗成本极高

推理时要预测新 token,就要把前面一段拉进来做完整 attention,每滑动一个 token 就重算一次。**每个 token 的评估成本是 O(L) 次完整 forward**,实际推理非常慢。

### 缺陷 3:绝对位置编码跨段冲突

原始 Transformer 的正弦位置编码是绝对位置,即 PE(0), PE(1), ..., PE(L-1)。如果硬把两段拼一起跑 attention,**第 2 段的位置 0 会和第 1 段的位置 0 冲突**,模型无法分辨它们。

---

## 核心技巧 1:段级递归(Segment-level Recurrence)

![Transformer-XL 的段级递归:训练和评估时,前一个 segment 的隐状态 h^{n-1} 被缓存(stop-gradient),下一个 segment 的 attention 可以同时 attend 到缓存和当前 segment。相当于每层都有"长期记忆"。](/assets/img/transformer-xl/x3.png)
_Figure 2(a/b):Transformer-XL 段级递归——隐状态跨段传递,评估时无需重算_

### 公式

设第 $\tau$ 个 segment 的第 $n$ 层隐状态为 $h_\tau^n$。在 $\tau+1$ 段:

$$
\tilde h_{\tau+1}^{n-1} = [\text{SG}(h_\tau^{n-1}) \circ h_{\tau+1}^{n-1}]
$$

$$
q_{\tau+1}^n = h_{\tau+1}^{n-1} W_q^\top,\quad k_{\tau+1}^n = \tilde h_{\tau+1}^{n-1} W_k^\top,\quad v_{\tau+1}^n = \tilde h_{\tau+1}^{n-1} W_v^\top
$$

$$
h_{\tau+1}^n = \text{TransformerLayer}(q, k, v)
$$

其中:
- $\text{SG}(\cdot)$ = stop-gradient(缓存的隐状态不反传,只作为 k/v 的额外上下文)
- $\circ$ = 按 seq 维拼接
- Query 只用当前段,Key/Value 同时用缓存 + 当前段

### 效果

- 每一层都能"看到"**更远的历史**(理论上可以回溯到 $O(N \times L)$,其中 $N$ 是层数、$L$ 是 segment 长度)
- **训练成本不变**(缓存是常数开销)
- **评估快非常多**:下一个 token 预测直接读缓存,不需要滑窗重算

---

## 核心技巧 2:相对位置编码

段级递归带来一个新问题:跨段的 k、v 和当前段的 q 在位置空间是不同 segment 的,**绝对位置编码会冲突**。

Transformer-XL 重写了 attention 的分数展开:

### vanilla Transformer attention 的绝对位置分数

$$
A_{i,j}^{\text{abs}} = \underbrace{E_{x_i}^\top W_q^\top W_k E_{x_j}}_{(a)} + \underbrace{E_{x_i}^\top W_q^\top W_k U_j}_{(b)} + \underbrace{U_i^\top W_q^\top W_k E_{x_j}}_{(c)} + \underbrace{U_i^\top W_q^\top W_k U_j}_{(d)}
$$

其中 $U_j$ 是位置 $j$ 的绝对位置编码。

### Transformer-XL 的相对位置分数

把 $U_j$ 换成相对位置 $R_{i-j}$,并且让位置相关的部分不依赖 $i$(因为 $i$ 的绝对位置在段内可以不同):

$$
A_{i,j}^{\text{rel}} = \underbrace{E_{x_i}^\top W_q^\top W_{k,E} E_{x_j}}_{(a)} + \underbrace{E_{x_i}^\top W_q^\top W_{k,R} R_{i-j}}_{(b)} + \underbrace{u^\top W_{k,E} E_{x_j}}_{(c)} + \underbrace{v^\top W_{k,R} R_{i-j}}_{(d)}
$$

关键变化:

- 拆出 $W_{k,E}$(内容 key)和 $W_{k,R}$(位置 key)
- 用两个可学习向量 $u, v$ 代替 $U_i^\top W_q$——因为在段内 $i$ 的绝对位置已经不重要
- $R_{i-j}$ 用**正弦函数的相对版本**,只依赖相对偏移

这套"相对位置 attention"后来启发了 **Shaw 2018、T5 bias、RoPE 等一系列相对位置方案**。

---

## 实验结果

### 语言建模 SOTA(2019 年)

| 任务 | 原最好 | Transformer-XL | 提升 |
|------|--------|---------------|------|
| WikiText-103 PPL | 20.5 | **18.3** | ⬇️ |
| enwik8 BPC | 1.03 | **0.99** | 首次破 1.0 |
| text8 BPC | 1.13 | **1.08** | ⬇️ |
| One Billion Word PPL | 23.7 | **21.8** | ⬇️ |
| Penn Treebank PPL | 55.3 | **54.5** | ⬇️ |

特别是 **enwik8 首次 BPC < 1.0**,标志语言建模进入新时代。

### 有效上下文长度

论文通过"相对有效上下文长度 (RECL)"度量:

- Vanilla Transformer LM(seg=128):RECL ≈ **128**
- Transformer-XL(seg=128, mem=128):RECL ≈ **900**
- Transformer-XL(seg=128, mem=512):RECL ≈ **3800**

**段级递归带来 ~30× 的有效上下文扩展**,没有增加训练 segment 长度(计算成本不变)。

### 推理速度

评估阶段,Transformer-XL 比 vanilla Transformer LM 快 **1874×**(不是笔误,是论文数字)。原因是 vanilla 需要滑窗重算,XL 直接读缓存。

---

## 为什么影响如此大

### 1. 打开了"长上下文"研究的大门

在 Transformer-XL 之前,几乎没有人认真研究 Transformer 的长上下文能力——大家默认"就 512 够了"。XL 证明:

- 上下文扩展到数千 token **可以大幅改善 LM 质量**
- 位置编码是关键瓶颈,**相对 > 绝对**

此后 ALiBi、RoPE、YaRN、LongRoPE 等**长上下文路线全部在这条路上**。

### 2. 相对位置编码的 "拆 attention" 思想

Transformer-XL 把 attention 分数拆成"内容-内容"、"内容-位置"、"位置-内容"、"位置-位置" 四项,然后按需设计每项。这个拆解后来被:

- **Shaw 2018** 简化使用
- **T5 bias** 用成 learnable scalar
- **RoFormer (RoPE)** 通过复数旋转彻底解决
- **ALiBi** 干脆把位置项拆成线性衰减 bias

所有这些都是在 Transformer-XL 拆的这张图上选不同"项"。

### 3. 段级递归思想的继承

- **Compressive Transformer** (2020) 把更远的历史压缩后再留缓存
- **Memorizing Transformer** (2022) 做 kNN 从历史中检索
- **Mamba / RWKV** (2023-24) 把"段级递归"彻底推到每个 token 级别
- **StreamingLLM** (2023) 只保留 attention sink + 最近 window 的 KV cache

所有"长上下文 + 缓存"思想都在 Transformer-XL 奠基的路径上。

### 4. 奠定了"cache 是长上下文关键"的观念

今天的 LLM 推理讨论 KV cache 占用、共享、量化,核心想法都是"**前一段的状态要经济地留给后面用**"。Transformer-XL 是第一次把这个思想做成产品级架构。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **长上下文 ≠ 把 seq_len 调大,而是让信息跨段传递**:Transformer-XL 用 cache + stop-gradient 实现了计算成本不变的"有效上下文扩展",这个思想后来演化成 PagedAttention、StreamingLLM 等
2. **绝对位置编码在长上下文天生不友好**:跨段冲突、不可外推,逼着研究者走向相对位置,这条路径最终开出了 RoPE 这朵花
3. **Attention 分数可以按"内容/位置"维度拆解**:Transformer-XL 的四项拆解是位置编码设计的范式,后续所有改进都是在这张图上做选择
4. **Evaluation 与 training 的计算不对称**:vanilla 训练 O(L),评估每 token O(L²) 重算;很多工程优化(KV cache、Flash Decoding)都是在解这个不对称
</callout>

---

## 延伸阅读

- [Attention Is All You Need 深度解读]({% post_url 2026-04-23-Attention-Is-All-You-Need-深度解读 %}) —— Transformer-XL 要改进的对象
- [Self-Attention with Relative Position Representations (Shaw et al., 2018)](https://arxiv.org/abs/1803.02155) —— 相对位置编码的前作
- [RoFormer / RoPE 深度解读]({% post_url 2026-04-23-RoFormer-RoPE-旋转位置编码深度解读 %}) —— Transformer-XL 相对位置的终极解
- [Compressive Transformers for Long-Range Sequence Modelling (Rae et al., 2020)](https://arxiv.org/abs/1911.05507) —— 段级递归的压缩升级版
- [Efficient Streaming Language Models with Attention Sinks (Xiao et al., 2023)](https://arxiv.org/abs/2309.17453) —— 继承并简化 Transformer-XL 的 cache 思想
