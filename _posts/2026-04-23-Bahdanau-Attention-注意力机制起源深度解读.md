---
title: "Bahdanau Attention — Attention 机制的起点(Neural Machine Translation by Jointly Learning to Align and Translate)"
date: 2026-04-23 22:40:00 +0800
categories: [Attention, Foundational, Seq2Seq]
tags: [bahdanau, attention, seq2seq, neural-machine-translation, encoder-decoder, 2014]
math: true
---

## 基本信息

- **作者**: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
- **机构**: Jacobs University Bremen, Université de Montréal
- **发表**: ICLR 2015(arXiv 2014-09)
- **arXiv**: [1409.0473](https://arxiv.org/abs/1409.0473)

## 一句话总结

2014 年首次在序列到序列(seq2seq)模型中提出**加性注意力机制**——让 decoder 在每一步生成时,**动态地给 encoder 每个位置的隐状态打分,按分加权求和**得到上下文向量。彻底解决了原 seq2seq 把整个源句压缩成**单一固定向量**导致长句翻译崩塌的问题。这是"attention"一词在现代深度学习里的**首次正式出现**,也是后来 Transformer 的直系祖先。

![Bahdanau attention 的图示:decoder 生成第 t 个词时,先对 encoder 每个位置 j 的 annotation h_j 打分 α_{tj},再用这些权重加权求和得到上下文向量 c_t,然后送入 decoder GRU 生成 y_t。](/assets/img/bahdanau-attention/x1.png)
_Figure 1:Bahdanau attention 的图示——"为每个目标词动态检索源句"_

---

## 背景:seq2seq 的单向量瓶颈

Cho 等人 2014 年提出的原始 seq2seq (RNN encoder-decoder):

- Encoder RNN 依次读入源句 $(x_1, ..., x_T)$,**只把最后一个隐状态** $h_T$ 作为"上下文"传给 decoder
- Decoder 基于这一个固定向量 $c = h_T$ 逐步生成目标句

这个"把整段源句压成一个向量"的设计在短句上还行,**长句翻译质量断崖式下跌**:30 词以上 BLEU 急剧下降。信息论角度的解释:不管输入多长,信息都被逼进固定维度的 $h_T$。长句超过容量,前面的信息就丢了。

## 核心机制:为每个目标词"检索"源句

Bahdanau 的关键改动:**不再用单一向量 $c$,而是让 $c_i$ 随目标位置 $i$ 变化**。

### 架构组成

- **Encoder**:双向 RNN,对每个源词位置 $j$ 产出 annotation $h_j = [\overrightarrow{h}_j;\overleftarrow{h}_j]$(双向隐状态拼接)
- **Decoder**:每步 $i$,先计算对齐分数 $e_{ij} = a(s_{i-1}, h_j)$,然后 softmax 得权重 $\alpha_{ij}$:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

- 最后得到随位置变化的上下文向量:

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$

- Decoder 用 $c_i$、上一步隐状态 $s_{i-1}$、上一步输出 $y_{i-1}$ 生成 $y_i$

### 关键:对齐分数函数 $a$

Bahdanau 选的是**加性注意力(additive attention)**,一个小 MLP:

$$
a(s_{i-1}, h_j) = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)
$$

后来 Luong(2015)提出更简单的"**dot-product attention**":

$$
a(s_{i-1}, h_j) = s_{i-1}^\top h_j
$$

再往后 *Attention Is All You Need* 把 dot-product 加上 $\sqrt{d_k}$ 缩放,就是今天的 scaled dot-product attention。

## 实验:长句翻译的转折

论文在 WMT'14 英法翻译上对比:

| 模型 | BLEU (all) | BLEU (no UNK) |
|------|-----------|---------------|
| RNN encdec-30(原 seq2seq,句长 ≤ 30)| 13.93 | 24.19 |
| RNN encdec-50 | 17.82 | 26.71 |
| **RNNsearch-30(Bahdanau)** | 21.50 | 31.44 |
| **RNNsearch-50** | **26.75** | **34.16** |

### 长度鲁棒性的可视化

![BLEU 分数随源句长度变化:原 seq2seq (encdec) 在句长超过 30 后 BLEU 直接崩塌;而加入 attention 的 RNNsearch 在 50+ 词长度仍保持稳定。这是 attention 机制的关键卖点。](/assets/img/bahdanau-attention/x2.png)
_Figure 2:BLEU 随句长的对比——attention 让长句翻译质量不再崩塌_

这张图是论文最具说服力的实验:**原 seq2seq 在长句处崩,带 attention 的版本稳定在高位**。信息瓶颈被打破。

## 最直观的收获:对齐可视化

attention 权重矩阵 $\alpha_{ij}$ 可以画成热力图,横轴为源词、纵轴为目标词。亮处表示 decoder 在生成该目标词时"看向了"哪些源词。

![四个样例 attention 对齐矩阵。大致沿对角线(反映语序对应),在语序变化处(如英法形容词-名词顺序不同)出现非对角走向,自动显示出"the European Economic Area" ↔ "la zone économique européenne" 这种跨词对齐。](/assets/img/bahdanau-attention/x3.png)
_Figure 3:Attention 对齐矩阵——自动学到了跨语言的词序重排_

这是人类第一次看到 **"神经网络在自动学习软对齐"** 的可视化证据。之前的统计机器翻译需要硬对齐(GIZA++ 等外部工具),attention 一次性内化了这件事。

![在含 UNK 的真实句子上继续观察 attention,可以看到模型对生僻词的 attention 有时会蔓延到相邻源词,即在"找不到"时做合理的猜测。](/assets/img/bahdanau-attention/x4.png)
_Figure 4:包含 UNK 的真实样例的 attention 模式_

![更长句子上的 attention 对齐,反映"长距离对齐"也可以由 attention 自然建模,不需要额外结构。](/assets/img/bahdanau-attention/x5.png)
_Figure 5:长句上的 attention 对齐_

![不同长度下 attention 呈现的对齐模式——是第一批"attention 可视化"研究的基础样本。](/assets/img/bahdanau-attention/x6.png)
_Figure 6:不同长度样本的 attention 模式汇总_

## 为什么影响巨大

### 1. 打开了 attention 这条研究线

在 Bahdanau 之前,"attention" 更多是计算神经科学用语,不是深度学习工具。**1409.0473 是让 attention 成为 ML 模块的起点**。此后 2014-2017 的 seq2seq 几乎全部加 attention,Luong 简化版、global/local attention、coverage attention 等全部是沿这条路。

### 2. 直接预示了 Transformer

从 Bahdanau 到 Transformer 的路径极清晰:

| 步骤 | 变化 |
|------|------|
| Bahdanau 2014 | decoder 对 encoder 加 attention |
| Luong 2015 | 用 dot-product 替代 additive,简化计算 |
| Graves 等 2016 | Self-attention 雏形(同一序列内部做 attention)|
| **Vaswani 2017** | 完全抛弃 RNN,全靠 self-attention + cross-attention,加 scaled dot-product,分多头 |

Transformer 里的 **cross-attention 子层几乎是 Bahdanau 的直接继承**。

### 3. 把"软对齐"从统计 MT 带进神经网络

统计 MT 时代,对齐(alignment)需要 GIZA++ 等专门工具硬算出来,然后喂给短语模型。Bahdanau 证明**软对齐可以和翻译任务端到端学习**。这个思路后来被推广到语音识别的 attention-based listen-attend-spell、图像描述、VQA 等几乎所有 seq2seq 类任务。

### 4. "动态上下文"的工程价值

这个思想更抽象的价值是:**一个下游任务可以在每个时间步从一个大的 key-value 库里按需检索**。这个"按需检索"的抽象后来被泛化成 attention 的通用接口,催生了 memory networks、外部记忆等一系列扩展。

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **单一固定向量的信息瓶颈**是 attention 最初要解决的问题;今天你看 MoE 的"细粒度专家"或 KV cache 的"分组共享",本质都在重复同一类"如何避免被单一瓶颈制约"的思考
2. **softmax 加权求和是一个可微的检索**:attention 让"从一组候选中按相关度取值"这件事变成梯度可传的子模块,这是它超越硬对齐的根本
3. **加性 vs 点积**:Bahdanau 原版是加性,今天的 Transformer 用点积——不是因为加性错了,而是点积在 GPU 上更高效;理解这段演化能帮你理解后续 Linformer/Performer 等变体都是在点积/加性这两条线上的新尝试
4. **Attention 矩阵可视化是可解释性的第一课**:Anthropic Circuits、induction head、attention sink 等 2020 年代的机械可解释性工作,方法论起源都在这篇 2014 年的论文的热力图里
</callout>

---

## 延伸阅读

- [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215) —— 原 seq2seq,Bahdanau 要改进的对象
- [Effective Approaches to Attention-based NMT (Luong et al., 2015)](https://arxiv.org/abs/1508.04025) —— 简化为 dot-product,为 Transformer 铺路
- [Attention Is All You Need 深度解读]({% post_url 2026-04-23-Attention-Is-All-You-Need-深度解读 %}) —— Bahdanau 的直系后代
- [A Mathematical Framework for Transformer Circuits (Anthropic, 2021)](https://transformer-circuits.pub/2021/framework/index.html) —— 把 attention 从"机制"上升到"电路"的科学化
