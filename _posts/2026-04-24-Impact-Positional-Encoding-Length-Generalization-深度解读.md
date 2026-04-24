---
title: "The Impact of Positional Encoding on Length Generalization — 位置编码对外推能力的系统对比"
date: 2026-04-24 11:45:00 +0800
categories: [Attention, Positional Encoding, Length Generalization]
tags: [positional-encoding, length-generalization, nope, alibi, rope, kazemnejad-2023]
math: true
---

## 基本信息

- **作者**: Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy
- **机构**: Mila, McGill, IBM Research
- **发表**: NeurIPS 2023
- **arXiv**: [2305.19466](https://arxiv.org/abs/2305.19466)

## 一句话总结

一篇**反直觉**的实证论文:系统比较 Sinusoidal / Learned / ALiBi / Rotary / NoPE(**不加任何位置编码**)在 decoder-only 模型上的长度外推能力。**颠覆性发现:不加位置编码(NoPE)在长度外推上反而最好**——因为 causal mask 本身已经给了模型"相对位置"的信息。这挑战了"位置编码是必需的"这一共识,并引出后续对"Attention 如何隐式学习位置"的研究热潮。

![各种位置编码在合成算术任务上的长度外推表现:NoPE 意外地在大多数任务上胜出,ALiBi 次之,RoPE 在训练长度外衰退快。](/assets/img/pos-length-gen/x1.png)
_Figure 1:位置编码对比——NoPE 令人惊讶地最擅长外推_

---

## 背景:长度外推是 LLM 的核心短板

### 什么是"长度外推"

训练时序列最长 $L_{train}$,推理时长度 $L_{test} > L_{train}$。好的模型应该**在超长输入上不崩溃**。

### 几种典型位置编码

- **Sinusoidal (Vaswani 2017)**:固定正弦函数,加到 embedding 上
- **Learned**:每个位置一个可学 embedding
- **Rotary (RoPE, Su 2021)**:在 Q/K 上做复数旋转
- **ALiBi (Press 2021)**:在 attention score 上加线性衰减
- **NoPE (No Positional Encoding)**:**什么都不加**,只靠 causal mask

直觉上我们会预期:精心设计的位置编码(RoPE / ALiBi)外推比 NoPE 好。但这篇论文发现**事情不是这样**。

---

## 核心实验:合成 + 真实任务的系统对比

### Synthetic benchmark:算术和 reasoning

作者设计一组需要显式位置信息的任务:

- **Addition**:多位数加法
- **Parity**:计算序列中 1 的奇偶性
- **LEGO**:模拟记号系统
- **Scan**:输入 → 输出变换的函数合成

训练长度 $L_{train}$ 上 10-20,测试长度 20-100。

### 真实 benchmark

- **SCAN**:compositional generalization
- **CFQ**:composition reasoning

### 实测结果

![训练长度 20 的模型在测试长度 40、60、80 上的准确率。NoPE 曲线最高,RoPE 快速衰退,ALiBi 中等。Learned 彻底崩溃。](/assets/img/pos-length-gen/x2.png)
_Figure 2:主要结果——NoPE 在外推上综合最好_

| 位置编码 | $L_{train}$ 内 | 2× 外推 | 5× 外推 |
|---------|---------------|--------|--------|
| Sinusoidal | 95 | 30 | 10 |
| Learned | 100 | **0** | 0 |
| RoPE | 98 | 40 | 15 |
| ALiBi | 96 | 65 | 45 |
| **NoPE** | **97** | **85** | **70** |

**NoPE 在中长外推上领先 20-50 分**。这让人大跌眼镜——没有位置编码反而最好?

---

## 为什么 NoPE 能学到位置?

### 理论分析:Causal Mask 的隐式位置信号

![NoPE 能工作的理论解释:Causal mask 让位置 $i$ 只能看到前 $i+1$ 个 tokens。不同位置看到的"token 数量"不同——这本身就是一种位置信号。](/assets/img/pos-length-gen/x3.png)
_Figure 3:Causal mask 提供的隐式位置信息_

作者的理论分析:

- 位置 $i$ 的 attention 看到 $\{x_0, x_1, \ldots, x_i\}$——正好 $i+1$ 个 tokens
- 位置 $j > i$ 看到 $j+1$ 个 tokens
- 这个"能看到多少 tokens"的差异就是 **隐式的位置索引**

模型可以通过"我能 attend 到多少 token"间接推断自己的位置。不需要显式的位置编码。

### 可解释性证据

作者通过可视化发现,NoPE 模型的某些 attention head **学到了类似"计数"的行为**——在前 N 个位置 attend 到 anchors,通过 anchor 的数量算位置。

![特定 head 的 attention 模式:NoPE 训练出的模型自动形成"位置计数"attention pattern——某些 head 专门 attend 到每个开头 token,数的个数等效位置。](/assets/img/pos-length-gen/x4.png)
_Figure 4:NoPE 模型自发形成的"位置计数"head_

### NoPE 的外推优势

Learned / Sinusoidal / RoPE 都依赖特定的**位置表示**,超出训练范围这个表示就失效。NoPE 没有任何这样的"外部位置信号",完全靠**attention 的涌现行为**编码位置——这个行为是 length-invariant 的。

---

## ALiBi 的鲁棒性

![ALiBi 在各种外推场景下表现仅次于 NoPE,尤其在长距离任务上比 RoPE 好得多。但在需要精确位置匹配的任务(比如 arithmetic)上仍弱于 NoPE。](/assets/img/pos-length-gen/x5.png)
_Figure 5:ALiBi 的鲁棒性介于 RoPE 和 NoPE 之间_

ALiBi 作为"第二好"的位置编码,表现仅次于 NoPE。原因:

- ALiBi 只有"距离"信息,没有"绝对位置"
- 距离衰减函数 $-m|i-j|$ 在超训练长度的外推上仍然数学良定义
- 不像 RoPE 的频率在外推时失配

但 ALiBi 也不如 NoPE——因为 ALiBi 仍然**硬编码**了距离衰减的形状,不是完全灵活的。

---

## 为什么这篇论文重要

### 1. 挑战了"位置编码是必需的"这个假设

2023 年之前,"**Attention 没有位置概念,必须加位置编码**"是教科书级的共识。这篇论文实证证明这不完全对——**在 decoder-only + causal mask 的设置下,位置编码可能有害**。

### 2. 推动"Attention 隐式学习位置"的研究

论文引出一个深刻问题:**模型到底是怎么学会位置的?** 这启发了后续一系列可解释性研究:

- Olsson et al. 2022 的 Induction Heads
- Chan et al. 2024 的 in-context learning 机制

### 3. 对 LLM 架构设计的实用意义

某些新大模型(如 Gemma、部分 Qwen 变体)开始**只在部分层用 RoPE**,其他层 NoPE。这种混合设计可以兼得 RoPE 在训练长度内的精确性和 NoPE 的外推性。

### 4. 重新定位其他 PE 的价值

这篇论文让社区重新思考 RoPE / ALiBi 的使用场景:

- 训练长度内精确匹配任务:RoPE 仍然最好
- 需要强外推 + 合成任务:NoPE 或 ALiBi
- 长 context LLM(数千到百万):需要 RoPE + YaRN 等扩展,或 NoPE 风格

---

## 局限

### 1. 仅限 decoder-only + causal mask

论文的核心论证**完全依赖 causal mask 的隐式位置信号**。这在 BERT 等 encoder-only 模型上不成立——encoder 必须要位置编码。

### 2. 任务偏 synthetic

Main results 大量依赖合成算术任务。真实语言建模 / 代码的长度外推上,NoPE 是否仍然最优还需要进一步验证。

### 3. 模型规模限制

实验用的是 <500M 参数的小模型。更大规模(7B+)的 NoPE 表现如何,论文没覆盖。

### 4. 短文本性能有小下降

NoPE 在训练长度内的性能比 RoPE 略差(1-2%)。所以不是"全局最优",是外推 vs 精确的 trade-off。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Causal mask 本身就是一种位置信号**:位置 $i$ 看到的 token 数 $=i+1$——模型可以用这个差异隐式编码位置。这是 NoPE 能工作的根本原因
2. **显式位置编码越精密,外推越差**:Learned > Sinusoidal > RoPE > ALiBi > NoPE——**越 "hard-coded" 的位置信息,外推时越容易失配**
3. **Attention 的"涌现位置学习"值得关注**:模型不靠位置编码也能学会位置,说明 attention 机制本身有极强的结构发现能力。这是 in-context learning 的更深层机制之一
4. **每种 PE 有自己的最佳场景**:不存在"最好"的位置编码,只有"给定任务/长度的最优"。大模型设计需要按 use case 选
</callout>

---

## 延伸阅读

- [ALiBi 深度解读]({% post_url 2026-04-23-ALiBi-线性偏置注意力深度解读 %}) —— 论文中表现次好的位置方案
- [RoFormer / RoPE 深度解读]({% post_url 2026-04-23-RoFormer-RoPE-旋转位置编码深度解读 %}) —— 被论文挑战的主流方案
- [YaRN 深度解读]({% post_url 2026-04-24-YaRN-高效上下文扩展深度解读 %}) —— RoPE 的扩展路线(这篇的对立面)
- [Transformer Circuits Framework 深度解读]({% post_url 2026-04-24-Transformer-Circuits-数学框架深度解读 %}) —— Attention 涌现机制的可解释性研究
- [NoPE 后续工作(Gemma 2 position encoding analysis)](https://arxiv.org/abs/2403.08295) —— NoPE 思想在生产模型中的应用
