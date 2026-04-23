---
title: "Attention Is All You Need — Transformer 的起点与工程范式"
date: 2026-04-23 18:20:00 +0800
categories: [Attention, Transformer, Foundational]
tags: [transformer, attention, self-attention, multi-head, positional-encoding, vaswani, 2017]
math: true
---

## 基本信息

- **作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- **机构**: Google Brain / Google Research / University of Toronto
- **发表**: NeurIPS 2017
- **arXiv**: [1706.03762](https://arxiv.org/abs/1706.03762)
- **参考实现**: [TensorFlow Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) · [The Annotated Transformer (Harvard)](http://nlp.seas.harvard.edu/annotated-transformer/) · [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

## 一句话总结

提出 **Transformer**——完全基于 **Self-Attention** 与 **Multi-Head Attention** 的 encoder-decoder 架构,**抛弃 RNN 与卷积**,在 WMT'14 英德/英法翻译上刷新 SOTA 并把训练时间从天级压到小时级。奠定了此后所有大语言模型的底层范式,也启动了"系统硬件 + 注意力"这一持续至今的研究主线。

![Transformer 架构:左侧 encoder 6 层,右侧 decoder 6 层。每层由 multi-head self-attention + feed-forward 两个子层组成,配残差 + LayerNorm。Decoder 多出一个对 encoder 输出做 cross-attention 的子层。](/assets/img/attention-is-all-you-need/x1.png)
_Figure 1:Transformer 架构——完全基于注意力的 encoder-decoder_

---

## 核心问题:为什么抛弃 RNN?

2017 年前,序列建模几乎一统于 RNN/LSTM/GRU。但 RNN 有三个结构性问题:

1. **训练无法并行**:时间步 $t$ 必须等 $t-1$ 的隐状态,GPU 利用率极低
2. **长距离依赖衰减**:信息从 $t=0$ 传到 $t=100$ 经过 100 次门控,梯度几乎消失
3. **attention 只是 RNN 的配饰**:2014-2016 年 attention 已用于 seq2seq(Bahdanau/Luong),但都是"RNN 主干 + attention 上层",不是主角

Transformer 的回答非常激进:**直接去掉 RNN,仅留 attention**。

<callout emoji="bulb" background-color="light-blue" border-color="blue">
**核心洞察**:Self-Attention 的计算是**全序列位置两两相关**的一次矩阵乘,天然可并行;而且任意两个位置的路径长度都是 **1**,没有长距离衰减。代价是 $O(N^2)$ 复杂度——但只要 $N$ 不太大,GPU 完全吃得下。
</callout>

---

## 核心机制

### Scaled Dot-Product Attention

![Scaled Dot-Product Attention (左) 与 Multi-Head Attention (右) 的结构图](/assets/img/attention-is-all-you-need/x2.png)
_Figure 2:Scaled Dot-Product Attention(左)与 Multi-Head Attention(右)_

给定 Query $Q$、Key $K$、Value $V$,注意力的核心公式是:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

逐元素看:

| 步骤 | 形状 | 含义 |
|------|------|------|
| $QK^\top$ | $(N, N)$ | 每对 $(i,j)$ 的相似度分数 |
| $/\sqrt{d_k}$ | $(N, N)$ | 缩放,防止 $d_k$ 大时点积过大导致 softmax 饱和 |
| $\text{softmax}$ | $(N, N)$ | 按行归一化,得到注意力权重 |
| $\times V$ | $(N, d_v)$ | 加权求和 value 得到输出 |

**为什么 $\sqrt{d_k}$**?假设 $Q, K$ 元素独立、均值 0 方差 1,则 $QK^\top$ 的方差是 $d_k$。$d_k=64$ 时点积期望 ≈ ±8,喂给 softmax 几乎全部塌到单点。除以 $\sqrt{d_k}$ 把方差稳在 1,softmax 保持"可学习"。这条后来演化为 QK-norm、logit soft-cap 等训练稳定性技巧。

### Multi-Head Attention

**关键概念**:不要只算一次 attention,要**并行算 $h$ 次**,每次用独立的投影矩阵:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

参数形状(原论文默认 $d_\text{model}=512, h=8$):

- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_\text{model} \times d_k}$,$d_k = d_\text{model}/h = 64$
- $W^O \in \mathbb{R}^{h d_v \times d_\text{model}}$

**为什么分头**?单一 attention 只能学一种"关注模式",多头让模型在不同子空间同时学习不同关系类型——句法依赖、指代消解、长距离关联等(见后面的可视化)。

### Position-wise Feed-Forward

每个位置独立过一个 2 层 MLP:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

维度:$d_\text{model} \to 4 d_\text{model} \to d_\text{model}$(即 `512 → 2048 → 512`)。这一 "**膨胀-收缩**" 模式后来成了所有 Transformer 变体的标配;SwiGLU 等激活函数都是在此基础上的改进。

### Positional Encoding

没有 RNN/CNN,模型怎么知道 token 的顺序?加**正弦位置编码**:

$$
\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d_\text{model}})
$$

$$
\text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_\text{model}})
$$

设计动机:不同 $i$ 对应不同频率的三角函数,**$\text{PE}_{pos+k}$ 可以由 $\text{PE}_{pos}$ 线性表出**——理论上模型能学会"相对位置"而不是死记绝对位置。实际工作未必完美(后来 RoPE 从不同角度改进),但这个"注入位置信息 + 允许外推"的框架沿用至今。

---

## 架构全貌

Encoder 6 层,每层:

```
x → MultiHeadSelfAttention → +残差 → LayerNorm → FFN → +残差 → LayerNorm → 输出
```

Decoder 6 层,每层多一个 cross-attention 子层:

```
x → MaskedMultiHeadSelfAttention → +残差 → LayerNorm 
  → MultiHeadCrossAttention(Q=当前, K=V=encoder输出) → +残差 → LayerNorm 
  → FFN → +残差 → LayerNorm
```

**Masked self-attention**:decoder 生成第 $t$ 个 token 时不能"看到未来",靠一个下三角 mask 把 attention 矩阵的上三角填 $-\infty$。

---

## 复杂度对比

| 层类型 | 每层复杂度 | 顺序操作数 | 最大路径长度 |
|---------|-----------|-----------|--------------|
| Self-Attention | $O(N^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(N \cdot d^2)$ | $O(N)$ | $O(N)$ |
| Convolution | $O(k \cdot N \cdot d^2)$ | $O(1)$ | $O(\log_k N)$ |
| Self-Attention (restricted) | $O(r \cdot N \cdot d)$ | $O(1)$ | $O(N/r)$ |

**关键洞察**:self-attention 的"最大路径长度 $O(1)$"就是它能学长距离依赖的根源——任意两个位置之间只隔一次 attention。而"顺序操作数 $O(1)$"意味着完全可并行。唯一的代价是 $O(N^2)$——但只要序列不长,GPU 可以吞得下,而且随后 **FlashAttention** 告诉我们 $N^2$ 的 FLOPs 不是真瓶颈,HBM IO 才是。

---

## 实验结果

### WMT'14 翻译

| 模型 | BLEU (EN-DE) | BLEU (EN-FR) | 训练成本 (FLOPs) |
|------|--------------|--------------|-------------------|
| GNMT + RL | 24.6 | 39.92 | ~$10^{20}$ |
| ConvS2S | 25.16 | 40.46 | ~$10^{20}$ |
| **Transformer (base)** | **27.3** | 38.1 | ~$10^{18}$ |
| **Transformer (big)** | **28.4** | **41.8** | ~$2 \times 10^{19}$ |

Transformer **同时**把 BLEU 推到 SOTA,并把训练 FLOPs 降了 **一到两个数量级**。

### 模型大小与超参

- **base**: $d_\text{model}=512, d_\text{ff}=2048, h=8, N=6$,65M 参数,8× P100 训 12 小时
- **big** : $d_\text{model}=1024, d_\text{ff}=4096, h=16, N=6$,213M 参数,8× P100 训 3.5 天

对比 GNMT 需要 **96× K80 训 6 天**,成本优势一目了然。

---

## Attention 到底学到了什么?

论文的附录可视化是理解 attention 的最佳入门材料。

### 长距离依赖

![长距离依赖例子:encoder 第 5 层的多个 attention head 把动词 "making" 链接到它远处的修饰语 "more difficult",完成 "making...more difficult" 这个结构。不同颜色代表不同的 head。](/assets/img/attention-is-all-you-need/x3.png)
_Figure 3:encoder 第 5 层中,"making" 的多个头关注到远处的 "more difficult",形成长距离短语结构_

传统 RNN 里"making...more difficult"中间隔了十几个词,信息传递非常困难。Transformer 里多个头直接把这两个位置拉到一起,路径长度 = 1。

### 指代消解(Anaphora Resolution)

![第 5 层 head 5 的完整注意力图:代词 "its" 指向其先行词。](/assets/img/attention-is-all-you-need/x4.png)
_Figure 4 上:head 5 的完整注意力图,呈现典型的指代链接模式_

![同一层 head 5 和 head 6 对 "its" 一个词的孤立注意力:在两个头里都高度集中到前文的某个名词短语——典型的指代消解。](/assets/img/attention-is-all-you-need/x5.png)
_Figure 4 下:单看 "its" 的注意力,head 5 和 head 6 都精准锁到先行词_

这是 induction head / previous-token head 之前最早被发现的"可解释 attention 模式"之一,也是后来机械可解释性(mechanistic interpretability)研究的直接启发源。

### Head 专化

![第 5 层两个不同的 head 各自呈现和句法结构高度相关的行为。](/assets/img/attention-is-all-you-need/x6.png)
_Figure 5 上:一个 head 关注句法依赖关系_

![另一个 head 展现不同的结构化 attention 模式,两个头在同一层分工明显。](/assets/img/attention-is-all-you-need/x7.png)
_Figure 5 下:同层另一个 head,学到不同的结构任务_

两个头在同层**各干各的活**——这就是 Multi-Head 设计的直接证据。后来 Anthropic 的 *A Mathematical Framework for Transformer Circuits* 把这个观察系统化成 "circuits" 理论。

---

## 为什么影响如此之大

### 1. 工程范式的转折点

Transformer 前,深度学习主流是"小心翼翼设计 inductive bias"(CNN for 图像、RNN for 序列)。Transformer 证明了 **"只要参数量 + 数据 + 计算足够,通用架构能打败精心设计的专用架构"**——这直接启发了后来 ViT(Transformer 做 CV)、AlphaFold 2、RT-2 等跨域应用。

### 2. 启动了 Scaling Laws 时代

Transformer 的**结构极简 + 高度并行**特性让 scaling 变得简单:只需加层数、加 hidden 宽度、加数据。GPT 系列(2018-) 直接沿用 decoder-only 的 Transformer,把参数从 117M 扩到 175B、再到万亿规模,催生了今天的 LLM 革命。

### 3. 成为后续研究的共同底座

2017 后几乎所有 NLP/视觉/多模态模型都是 Transformer 变体:

- 位置编码:RoPE、ALiBi、YaRN → 推广到长上下文
- KV 头共享:MQA、GQA、MLA → 解决推理 KV cache
- 系统实现:FlashAttention、PagedAttention → 解决训练/推理 IO
- 替代注意力:Linformer/Performer/Mamba/RetNet → 降复杂度
- Hybrid:Jamba、Qwen3.5 → full + linear 混合

**每一篇"后续"论文的第一句,都在回答"相对于 vanilla Transformer,我改了什么"**。

### 4. 实验可复现性 & 开源文化

论文放出全部代码(Tensor2Tensor)、详细超参、训练步骤,后来被 Harvard NLP 的 *The Annotated Transformer* 转写为 500 行 PyTorch,成为无数研究者的入门教材。karpathy 的 nanoGPT 继承了同样的文化——**让最核心的架构能在 500 行内被理解**。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Attention 是一次可并行的矩阵乘**,这是它战胜 RNN 的全部秘密;任何现代 attention 变体都是在这个计算图上做文章
2. **Multi-Head 的意义不是提升单次 attention 能力,而是让模型在多个子空间并行学习不同关系**——head specialization 是经验事实,不只是理论
3. **$\sqrt{d_k}$ 不是玄学**:它是一个精确的方差修正,没它训练就炸;后来所有 QK-norm、logit soft-cap 都是同一个脉络
4. **通用架构 + 规模 = 专用架构 + 精巧设计的替代品**,这个工程哲学是 Transformer 改变 ML 的真正原因
</callout>

---

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/) —— 500 行 PyTorch 逐行对照
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) —— 最小可训练的 Transformer 实现
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) —— Transformer 时代的系统范式革命
- [A Mathematical Framework for Transformer Circuits (Anthropic, 2021)](https://transformer-circuits.pub/2021/framework/index.html) —— 把 attention 可视化推向科学化
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) —— Transformer 规模化的第一原理
