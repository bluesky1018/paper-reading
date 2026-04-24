---
title: "A Mathematical Framework for Transformer Circuits — 把 Transformer 拆成可分析的数学组件"
date: 2026-04-24 12:00:00 +0800
categories: [Interpretability, Attention, Circuit Analysis]
tags: [transformer-circuits, mechanistic-interpretability, induction-heads, anthropic-2021]
math: true
---

## 基本信息

- **作者**: Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, ... (Anthropic)
- **机构**: Anthropic
- **发表**: Transformer Circuits Thread, December 2021
- **链接**: [transformer-circuits.pub/2021/framework](https://transformer-circuits.pub/2021/framework/index.html)

## 一句话总结

Anthropic 推出的**机械可解释性(Mechanistic Interpretability)**里程碑论文——把 Transformer 的 attention 拆成一个**数学上优雅且可分析的框架**:**QK circuit**(决定 attend 哪里)和 **OV circuit**(决定 copy 什么)。用这个框架,作者发现了 Transformer 中最基础的算法原语——**Induction Head**——解释了为什么 Transformer 能做 in-context learning。这是机械可解释性领域的奠基之作,开启了"把神经网络当电路逐层逆向工程"的研究范式。

## 一张图理解核心思想

![Transformer 的每个 attention head 被分解为 QK circuit(决定 attention pattern)和 OV circuit(决定被 copy/写入的信息)。两个电路独立分析,组合理解 head 功能。](https://transformer-circuits.pub/2021/framework/images/qk_ov_circuit.png)
_图示(自绘):QK Circuit 和 OV Circuit 的电路视角_

---

## 背景:深度学习的可解释性困境

### 黑箱问题

深度神经网络被普遍视为"黑箱"——训练完成后模型内部发生什么,我们并不真正理解。对于 Transformer,尤其如此:

- 有数十亿参数
- 有多层 attention + FFN 交织
- 涌现行为(in-context learning, chain-of-thought)出现机制不明

在 2021 年之前,可解释性研究主要停留在**统计层面**:attention map 可视化、probing classifier、activation analysis 等。这些方法能告诉你"**what**"("模型关注了这个 token"),但不能告诉你"**why**"或"**how**"。

### Anthropic 的新范式:机械可解释性

Anthropic 团队(Chris Olah 等,从 OpenAI 离开创立)提出**机械可解释性**的口号:**把神经网络当作可逆向工程的电路**。

- 每个神经元 / head 都有明确的计算角色
- 通过数学推导 + 仔细观察,可以"读懂"这些角色
- 最终目标:**完全理解一个 Transformer 做了什么**

这篇 2021 年的 Framework 论文是第一次把这个理念落实到 Transformer 上。

---

## 核心数学:Residual Stream 视角

### Residual Stream

Transformer 的关键结构是**残差连接**:每一层把自己的输出**加到** residual stream 上,而不是覆盖。

$$
x_{l+1} = x_l + F_l(x_l)
$$

其中 $F_l$ 是第 $l$ 层(attention 或 FFN)的贡献。

作者把 $x$ 叫做 **residual stream**——每个 token 在所有层中保持的一条"主通道",每层 attention / FFN 通过**加**来写入信息,通过**读**做贡献。

这个视角的意义:

- Residual stream 是 Transformer 的**通信总线**
- 每一层可以"读" stream 上的现有信息、然后"写"新信息
- 不同层的 head / neuron 通过 stream 形成复杂的协作链路

### 从 Residual Stream 的角度看 Attention

标准 attention:

$$
\text{head}_h(x) = \text{softmax}\!\left(\frac{x W_Q^h (x W_K^h)^\top}{\sqrt{d}}\right) x W_V^h W_O^h
$$

重新整理:

$$
\text{head}_h(x) = \text{softmax}(\cdots) \cdot x \cdot (W_V^h W_O^h)
$$

注意 $W_V^h W_O^h$ 是一个 $d \times d$ 矩阵——作者称为 **OV circuit**。

同样地,$W_Q^h (W_K^h)^\top$ 是一个 $d \times d$ 矩阵——**QK circuit**。

---

## QK Circuit 与 OV Circuit 的分离

### QK Circuit(决定 attend 哪里)

$$
\text{attention pattern} = \text{softmax}\!\left(\frac{x_i \cdot W_{QK}^h \cdot x_j^\top}{\sqrt{d}}\right)
$$

其中 $W_{QK}^h = W_Q^h (W_K^h)^\top$。

这个 circuit 的输入是位置 $i$ 和 $j$ 的 residual stream 状态,输出是 **attend 强度**——"位置 $i$ 应该 attend 到 $j$ 的程度"。

### OV Circuit(决定 copy 什么)

$$
\text{output contribution} = \text{attention pattern}_{ij} \cdot (x_j \cdot W_{OV}^h)
$$

其中 $W_{OV}^h = W_V^h W_O^h$。

这个 circuit 的输入是被 attend 的 token $j$ 的 residual stream 状态,输出是**写入 residual stream 的内容**——"如果 $i$ attend 到 $j$,应该把 $j$ 的什么信息搬到 $i$ 的 stream 里"。

### 关键洞察:QK 和 OV 独立

QK circuit 和 OV circuit 是**独立的**线性变换,可以分开分析:

- $W_{QK}^h$ 是 "attention selectivity function"
- $W_{OV}^h$ 是 "information propagation function"

分开看,每个 head 的功能就变得清晰很多。

---

## 发现:Induction Head

论文最著名的发现。作者分析了 2 层 attention-only Transformer(无 FFN),发现第二层存在一种特殊的 head——**Induction Head**——实现了 "**pattern completion**" 功能:

### Induction Head 的行为

给定序列 `[A] [B] ... [A]`,Induction Head 让模型在最后一个 `[A]` 的位置**预测 `[B]`**。即"**看到 A,想起上次 A 后面跟着 B,所以现在也预测 B**"。

这就是 **in-context learning** 的最基本形式——**利用前文的 pattern 预测当前位置**。

### Induction Head 的内部电路

Induction Head 需要**两层 attention** 协作:

1. **第一层:Previous Token Head**
   - QK circuit 让每个 token attend 到它的前一个 token
   - OV circuit 把前一个 token 的信息写入当前 residual stream
   - 结果:每个位置的 stream 里现在有"我的前一个 token 是谁"的信息

2. **第二层:Induction Head**
   - QK circuit:query 是当前 token $X$,key 是每个前面位置的"前一个 token"信息
   - 因此会 attend 到"前面某个位置,它的前一个 token 也是 $X$"
   - OV circuit:把 attended 到的位置的**当前 token**(即"跟在上一个 $X$ 后面的 token")写到 stream
   - 结果:当前位置 stream 里有了"上次 $X$ 后面跟的是 $Y$"这个信息 → 下一个 token 预测 $Y$

这个两层协作的结构,是 Transformer **in-context learning 能力的最小原型**。

---

## 影响:机械可解释性革命

### 1. 开启了一个新研究领域

这篇论文发表后,机械可解释性(Mechanistic Interpretability,简称 MI)迅速成为 AI 安全研究的主要方向之一。Anthropic、DeepMind、OpenAI 的 Superalignment 团队、EleutherAI、Neel Nanda 的 TransformerLens 都在这条路上。

### 2. Induction Heads 的普遍性被验证

后续研究(Olsson et al. 2022)证明**所有从 GPT-2 到 Claude 规模的 Transformer 都有 Induction Heads**,而且它们的出现时间和 in-context learning 能力涌现时间**精确对应**——这让 ICL 这个神秘现象有了机械解释。

### 3. 数学框架被广泛使用

QK / OV circuit 的分解成为后续所有 Transformer 可解释性研究的标配工具。TransformerLens 库直接实现了这个分解视角,让社区能便捷地做 circuit 分析。

### 4. 引向更多电路发现

- **IOI Circuit**(Indirect Object Identification,Wang 2022):Transformer 怎么识别"Mary gave a book to John"里"John"是宾语
- **Entity Tracking Circuit**(Nanda 2024):追踪故事里的人物
- **Factual Recall Circuit**(Meng 2022):提取 "Paris is the capital of France" 这样的事实

每个 circuit 都是对 Transformer 内部算法的一块拼图。

### 5. 对 Safety 的意义

如果我们能**完全理解**一个大模型内部所有 circuits,就能:

- 知道模型为什么给出某个输出
- 发现模型是否有隐藏的有害行为
- 设计针对性的干预(activation steering,circuit breaking)

这是 AI Safety 的重要路径之一。

---

## 局限

### 1. 分析规模有限

这篇论文只分析了 2 层 attention-only 模型。扩展到 12+ 层、有 FFN 的模型,circuit 变得极其复杂,分析工作量巨大。

### 2. FFN 暂未涉及

论文的框架对 attention 很优雅,但 FFN 的分析更难——FFN 是非线性的,不能像 QK/OV 那样分解为矩阵。后续 Geva et al. 2020 把 FFN 理解为 key-value memories,补充了这块拼图。

### 3. 高层行为仍难完全解释

Induction Head 只是最简单的 in-context learning。大模型里的复杂 reasoning、few-shot learning 涉及数百 heads 的协作,目前的 MI 工具还不够。

### 4. 可复现性挑战

Circuit 的识别需要大量人工观察 + 精心设计的实验。把这个流程自动化、规模化仍是 open problem(Anthropic 在 2024 年的"Scaling Monosemanticity"工作上有重要进展)。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Residual Stream 是 Transformer 的通信总线**:每一层 attention / FFN 通过"读 + 写"这条主通道协作。理解 Transformer 先理解 residual stream
2. **QK / OV 分解让 attention 变得可分析**:Attention head 不再是"黑箱",而是两个可独立分析的线性映射——selectivity(QK)+ information flow(OV)
3. **Induction Head 是 in-context learning 的机制基础**:模型能从上下文学习,来自多层 attention head 的精心协作实现 pattern completion——不是"魔法"而是可验证的算法
4. **机械可解释性是 AI 安全的必要工具**:大模型能力强但行为难以完全预测。机械层面的理解是未来安全地使用大模型的关键基础设施
</callout>

---

## 延伸阅读

- [In-context Learning and Induction Heads (Olsson et al., 2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/) —— Induction heads 的后续系统研究
- [Zoology 深度解读]({% post_url 2026-04-24-Zoology-合成任务探测注意力深度解读 %}) —— 用合成任务探测 attention 容量
- [TransformerLens 资源引介]({% post_url 2026-04-23-TransformerLens-资源引介 %}) —— 实施 circuit 分析的开源工具
- [Scaling Monosemanticity (Anthropic, 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/) —— 对 FFN 和高层特征的机械理解
- [Anthropic Circuits Thread 主页](https://transformer-circuits.pub/) —— 机械可解释性的完整研究档案
