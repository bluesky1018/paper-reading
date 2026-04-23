---
title: "Linformer — 用低秩投影把 Self-Attention 压到线性复杂度"
date: 2026-04-23 23:35:00 +0800
categories: [Attention, Linear Attention]
tags: [linformer, low-rank, linear-attention, wang-2020, facebook]
math: true
---

## 基本信息

- **作者**: Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma
- **机构**: Facebook AI
- **发表**: arXiv 2020-06
- **arXiv**: [2006.04768](https://arxiv.org/abs/2006.04768)

## 一句话总结

提出 **Linformer**——通过实证证明 self-attention 矩阵**近似低秩**,然后用两个额外投影矩阵 $E, F$ 把 Key 和 Value 的**序列维**从 $N$ 压到一个固定的常数 $k$(如 $k=256$),从而让 attention 的时间与空间复杂度都从 $O(N^2)$ 降到 $O(N)$。是 2020 年"线性 attention"潮中最有实证色彩的方案之一。

![左两幅:预训练 Transformer 中 self-attention 矩阵的奇异值谱分析,显示绝大部分能量集中在少数奇异值上——attention 矩阵本质低秩。右侧:不同层/头的 attention 低秩分布。](/assets/img/linformer/x1.png)
_Figure 1:Attention 矩阵的奇异值谱——绝大部分能量集中在少数分量,支持"低秩假设"_

---

## 背景:Attention 的 $N^2$ 真的"需要"吗?

2020 年线性 attention 潮里涌现了许多方案——Reformer LSH、Sparse Transformer、Synthesizer、Performer 等。Linformer 走的是一条非常务实的路:

1. **先实证**:真实训好的 Transformer,它的 attention 矩阵有什么性质?
2. **再设计**:基于观察设计一个简单压缩方案

### 实证观察:Attention 矩阵近似低秩

作者对 RoBERTa 和 Wikitext 预训练的 Transformer,对每一层每一头的 attention 矩阵 $P \in \mathbb{R}^{N \times N}$ 做 SVD,发现:

- 前 128 个奇异值(约 $N = 512$ 的 1/4)就覆盖了约 90% 的能量
- 不同层、不同头基本都呈现这种"**顶部少数奇异值主导**"的模式

既然 $P$ 是低秩的,那么 attention 运算 $PV$ 就不需要完整的 $N \times N$ 矩阵——我们可以**先把 $V$(以及 $K$)投影到一个较小的维度 $k$**。

---

## 核心机制:对 Key/Value 的"序列维"做线性投影

![Linformer 架构(左)与推理时间对比(右):引入两个额外投影矩阵 E、F,先把 K 和 V 的序列维从 N 投影到 k (如 256),然后做 attention。推理时间随序列长度基本持平,实现真正的线性复杂度。](/assets/img/linformer/x2.png)
_Figure 2:Linformer 架构(左)与其在长序列下的推理耗时优势(右)_

### 数学形式

原始 attention:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

复杂度 $O(N^2 d)$。

Linformer 引入两个投影矩阵 $E_i \in \mathbb{R}^{k \times N}$ 和 $F_i \in \mathbb{R}^{k \times N}$(每头一组,可学),将 $K$ 和 $V$ 在**序列维**上做线性投影:

$$
\bar K = E K \in \mathbb{R}^{k \times d},\quad \bar V = F V \in \mathbb{R}^{k \times d}
$$

然后:

$$
\text{Attn}(Q, \bar K, \bar V) = \text{softmax}\!\left(\frac{Q \bar K^\top}{\sqrt{d}}\right) \bar V
$$

- $Q \bar K^\top \in \mathbb{R}^{N \times k}$ ← **不是 $N \times N$ 了**
- 复杂度 $O(N k d)$,当 $k$ 固定时变成 $O(N)$

### 与 MHA 的整合

每头独立拥有一对 $(E_i, F_i)$。论文也探索了**参数共享**,把不同层/不同头的 $E, F$ 共享,降低额外参数量。

### 代价

- **不是精确 attention**,是低秩近似
- $E, F$ 的尺寸 $k \times N$ 与**最大序列长度 $N$** 绑定,一旦换更长序列要重新投影或补齐
- 只适合 **encoder 自注意力**(需要一次看完整个序列),不适合 **causal decode**(因为 $E, F$ 会让未来信息"泄露"到过去)

---

## 实验结果

### 效率

![预训练 validation perplexity 曲线:Linformer 在 k=128 或 k=256 的不同设置下与原版 Transformer 收敛曲线几乎重合,但训练/推理时间远更低。](/assets/img/linformer/x3.png)
_Figure 3:预训练 perplexity 曲线——Linformer 收敛质量与 full attention 持平_

- $N = 512$:Linformer 比 Transformer 快 ~1.5×
- $N = 1024$:快 ~2.5×
- $N = 4096$:快 ~20×,显存占用降 4×+
- $N = 65536$:Transformer 直接 OOM,Linformer 照样跑

### 质量

在 MLM (masked language modeling) 预训练 + GLUE 下游任务上:

| 模型 | MRPC | SST-2 | QNLI | QQP | Avg |
|------|------|-------|------|-----|-----|
| RoBERTa-base | 87.5 | 94.6 | 92.5 | 91.6 | 91.6 |
| Linformer (k=128) | 86.3 | 94.3 | 91.3 | 91.2 | 90.8 |
| Linformer (k=256) | 87.3 | 94.6 | 91.9 | 91.4 | 91.3 |

**质量下降 < 1 分**,速度优势显著。

---

## 局限与反思

Linformer 虽然在 2020 年震动一时,但后续实战采用远不如理论声望:

### 1. 不适合 causal decode

$E, F$ 把序列维压到 $k$ 时,第 $i$ 个位置的压缩结果会包含**所有**位置信息。这对 encoder 或 BERT 型双向模型是 OK 的,但对自回归 decode(只能看过去)会**泄露未来**。这使 Linformer 在 LLM 时代几乎无用武之地。

### 2. 需要固定最大 $N$

$E, F$ 尺寸与训练时的 $N$ 绑定,推理长度一变就要重新投影。不如 **RoPE/ALiBi + Full attention** 的"训练 2K 外推 8K"来得灵活。

### 3. 低秩假设在大模型上变弱

后续工作发现,**模型规模越大,attention 矩阵的有效秩越高**——低秩近似的质量损失在 7B+ 模型上不再可忽略。Performer、Nyströmformer 等其他近似方案也面临同样挑战。

### 4. 未能落地 LLM 主线

从 2022 年开始,行业转向:

- **FlashAttention**:精确 attention + IO 优化,不损失质量
- **GQA / MLA**:在 full attention 内部优化 KV 压缩
- **Mamba / Gated DeltaNet**:完全抛弃 attention 的新递推

Linformer 作为"低秩近似" 这一路,在最需要它的长 context LLM 场景下,反而被更新的 exact 优化替代了。

---

## 为什么仍值得读

虽然不是主流方案,Linformer 贡献了几个值得内化的洞察:

### 1. 基于"测量 → 设计"的工程文化

**先观察真实模型的 attention 结构,再基于观察做架构改动**——这是极好的研究方法论。后续机械可解释性(Anthropic Circuits)、Zoology(retrieval 容量)、Chinchilla(scaling law)都在同一哲学上。

### 2. "序列维线性"的典型例子

它展示了**"attention 公式在序列维做线性投影"**这件事是**可学的**——不是只有 Q/K 点积 + softmax 这一种写法。这启发了后续 Performer、Nyströmformer 等变体。

### 3. 对比视角的锚点

今天的所有高效 attention 工作都会对标 Linformer——因为它是最早真正把 $N^2$ 压到线性的 paper 之一,是"长 context attention"研究的经典 baseline。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Attention 矩阵的低秩性质不是玄学**,是可以被 SVD 实证观察的;这个事实后续被多次独立验证
2. **线性 attention = "把序列维压缩"**:Linformer 把它写得最直接,后续 Performer 是"用核近似 softmax 代替投影"、Mamba 则是"用 recurrent state 代替投影"——都是同一问题的不同解法
3. **技术不是独立存在的,依附于场景**:Linformer 很美,但 causal decode 不支持让它与 LLM 时代擦肩而过;同样美的 RoPE 因为与 GQA/FlashAttention 兼容而大放异彩
4. **方法论比具体技术更持久**:Linformer 的"**实证观察驱动架构设计**"方法论可以迁移到任何新场景——先测,再设计
</callout>

---

## 延伸阅读

- [Performer (Choromanski et al., 2020)](https://arxiv.org/abs/2009.14794) —— 同期的另一条线性 attention 路线,用随机特征
- [Reformer (Kitaev et al., 2020)](https://arxiv.org/abs/2001.04451) —— 用 LSH 实现的亚二次 attention
- [Mamba 深度解读]({% post_url 2026-04-23-Mamba-选择性状态空间模型深度解读 %}) —— 线性复杂度的现代后继
- [FlashAttention 深度解读]({% post_url 2026-04-23-FlashAttention-IO感知精确注意力深度解读 %}) —— "保持精确 + 优化 IO" 的对立路线
- [Nyströmformer (Xiong et al., 2021)](https://arxiv.org/abs/2102.03902) —— Linformer 的后继,用 Nyström 方法近似
