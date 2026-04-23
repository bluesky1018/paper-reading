---
title: "RoFormer / RoPE — 用"旋转"把相对位置注入 attention"
date: 2026-04-23 18:35:00 +0800
categories: [Attention, Positional Encoding]
tags: [rope, roformer, rotary-position-embedding, relative-position, long-context, su-et-al]
math: true
---

## 基本信息

- **作者**: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
- **机构**: Zhuiyi Technology (追一科技)
- **发表**: arXiv 2021 (Neurocomputing 2024)
- **arXiv**: [2104.09864](https://arxiv.org/abs/2104.09864)
- **苏神原博**: [让研究人员绞尽脑汁的 Transformer 位置编码](https://spaces.ac.cn/archives/8265)
- **代码**: [ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer)

## 一句话总结

提出 **RoPE (Rotary Position Embedding)**——把位置信息用**二维旋转矩阵**施加到 Query 与 Key 上,让 attention 内积天然依赖两个 token 的**相对位置 $m-n$**,而不是绝对位置。既无需额外参数、不扰动其他模块,又原生支持任意长序列外推,成为 **LLaMA、GPT-NeoX、Qwen、DeepSeek、ChatGLM 等几乎全部新一代大模型的事实标准位置编码**。

![RoPE 实现示意:把 query/key 切成 d/2 对二维子向量,每对按位置 m 旋转对应角度 mθ_i,然后再做点积,点积结果自动依赖相对偏移 m-n。](/assets/img/roformer/x1.png)
_Figure 1:RoPE 的几何实现——以二维旋转矩阵对 Q/K 施加位置信息_

---

## 背景:之前的位置编码问题在哪?

Transformer 本身对序列顺序**置换不变**,必须额外注入位置信息。之前有三类做法:

| 方案 | 代表 | 问题 |
|------|------|------|
| **绝对正弦/余弦** | 原始 Transformer | 把位置当特征加到 embedding,长度外推弱 |
| **可学习绝对位置** | BERT、GPT-2 | 超过训练长度就 OOV,无法外推 |
| **相对位置(加法/bias)** | Shaw 2018, T5 bias | 侵入 attention 计算,多头共享难处理 |

期望中的位置编码应该:

1. **仅依赖相对位置** $m-n$——这才是语言的本质
2. **不引入额外参数**或只引入极少
3. **长度可外推**——训练 2K 能外推到 16K
4. **与 attention 计算正交**——不干扰 query/key/value 的结构

RoPE 几乎一条不漏地满足了所有要求。

---

## 核心思想:二维旋转 = 点积里的相对位置

### 起点的一句话推导

我们想让 $\langle q_m, k_n \rangle$ 只依赖 $m - n$(相对位置),不依赖各自绝对位置。这需要一个函数 $f$ 满足:

$$
\langle f(q, m),\ f(k, n) \rangle = g(q, k, m-n)
$$

**在 2 维情况下,RoPE 把 $q,k$ 看作复数,乘以单位复数 $e^{im\theta}$**(即旋转角度 $m\theta$):

$$
f(q, m) = q \cdot e^{im\theta}
$$

那么点积变成:

$$
\langle f(q,m), f(k,n) \rangle = \text{Re}\!\left[q \cdot e^{im\theta} \cdot \overline{k \cdot e^{in\theta}}\right] = \text{Re}\!\left[q\bar k \cdot e^{i(m-n)\theta}\right]
$$

只依赖 $m-n$。**这就是 RoPE 的全部魔法**。

### 扩展到 d 维

把 $d$ 维向量切成 $d/2$ 对二维子空间,每一对用不同的 $\theta_i$:

$$
\theta_i = 10000^{-2(i-1)/d},\quad i \in \{1, \dots, d/2\}
$$

对第 $m$ 个位置的 query $q_m$,施加的变换是一个分块对角旋转矩阵:

$$
R^d_{\Theta, m} = \begin{pmatrix}
R_{m\theta_1} & & & \\
& R_{m\theta_2} & & \\
& & \ddots & \\
& & & R_{m\theta_{d/2}}
\end{pmatrix},\quad R_{m\theta_i} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
$$

实际实现用交错拆分 $(x_1,x_2,x_3,x_4,\dots) \to (x_1,x_2), (x_3,x_4), \dots$,每对套一个 $R_{m\theta_i}$。伪代码:

```python
# 预计算 cos/sin 表
freqs = 10000 ** (-2 * torch.arange(0, d, 2) / d)         # (d/2,)
t = torch.arange(seq_len)                                  # (N,)
angles = torch.outer(t, freqs)                             # (N, d/2)
cos, sin = angles.cos(), angles.sin()                      # (N, d/2)

def apply_rope(x):  # x: (B, H, N, d)
    x1, x2 = x[..., 0::2], x[..., 1::2]                    # 偶数维、奇数维
    rotated = torch.stack([x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)
```

只需 $O(Nd)$ 额外计算、**零可学习参数**、对前向和反向都是逐元素乘加。

---

## 关键性质:远距离衰减(Long-term Decay)

![RoPE 的远距离衰减特性:随着两个 token 距离 |m-n| 变大,上界函数 c(s) 平滑衰减。这与物理直觉一致——距离远的 token 自然应该相关性弱。](/assets/img/roformer/x2.png)
_Figure 2:RoPE 的远距离衰减——无需学习,几何性质天然保证_

作者证明了一个很优雅的事实:

$$
\left| \sum_{i=1}^{d/2} q_{n,i} \bar k_{m,i} e^{i(m-n)\theta_i} \right| \leq \left( \max_i |q_{n,i} \bar k_{m,i}| \right) \cdot \sum_{i=1}^{d/2} \left| S_{d/2}((m-n)\theta_i) \right|
$$

其中 $S_j$ 是部分和。关键是右边随 $|m-n|$ 增大而平滑衰减——**即使在最坏情况下,远距离 attention 也会自然减弱**。这是 RoPE 比可学习位置编码可靠的地方。

<callout emoji="bulb" background-color="light-blue" border-color="blue">
RoPE 的衰减**不是训练学出来的,是几何性质天然保证的**。这也解释了为什么 RoPE 可以长度外推——外推时没有"新的位置要学",旋转角度自然延伸出去就行,衰减性质依然成立。
</callout>

---

## 实验:BERT / PerFormer / Transformer 对比

![左:BERT vs RoFormer 训练 loss 对比,RoFormer 收敛更快。右:PerFormer(线性 attention)加 RoPE 后 loss 明显下降。](/assets/img/roformer/x3.png)
_Figure 3:RoPE 在 BERT 预训练与 PerFormer 线性注意力上均带来明显收敛增益_

### WMT'14 EN-DE 翻译

| 模型 | BLEU |
|------|------|
| Transformer base (原正弦位置编码) | 27.3 |
| **RoFormer base (RoPE)** | **27.5** |
| Transformer big | 28.4 |
| **RoFormer big (RoPE)** | **28.7** |

### GLUE 下游任务

RoFormer 在 MRPC、STS-B、QNLI 等 fine-tune 任务上与 BERT 持平或更好,特别是需要位置敏感性的任务。

### 中文预训练 + 长文档

在 CAIL2019-SCM(中文案件相似匹配)上,把输入长度从 512 扩到 1024,RoFormer 保持稳定,而 BERT 的可学习位置编码会在超长处崩坏。

---

## 为什么成为事实标准

### 1. 零侵入性

RoPE **不动 value、不动 FFN、不改 LayerNorm、不引入新可训练参数**。只在 Q/K 进入 attention 前做一次元素级旋转。这意味着:

- 既有代码几乎即插即用
- 对各种 attention 变体(MHA、MQA、GQA、Flash/Paged/Linear Attention)都兼容
- 不与其他改动冲突

### 2. 外推友好

相对可学习位置编码和 T5 相对 bias,RoPE **训练 2K 外推到 4K/8K** 默认就有一定效果。配合后续的 **NTK-aware scaling**、**YaRN** 等 base 调整技巧,可以把 RoPE 的外推能力推到 **100K+**。这是今天的 Llama 3 100K、Qwen 1M 上下文的技术基础之一。

![RoPE 的整体结构:在多头 attention 的 Q/K 上应用旋转,其他部分完全一致。这种设计让 RoPE 可以无缝嵌入任意 Transformer 变体。](/assets/img/roformer/x4.png)
_Figure 4:RoPE 嵌入 multi-head attention 的完整数据流_

### 3. 理论与工程都够优雅

- **理论**:复数旋转 + 远距离衰减的闭式证明,数学家眼中是漂亮的
- **工程**:分块 2×2 旋转矩阵,在 GPU 上就是两次逐元素乘 + 两次加
- **实现**:< 10 行 PyTorch 代码

一个算法能同时打动理论、工程、应用三群人,才能成为事实标准。RoPE 做到了。

### 4. 后续演化的主干

RoPE 之后,位置编码的改进几乎都在 RoPE 基础上:

| 技术 | 解决问题 |
|------|---------|
| **NTK-aware scaling** | 无需重训就能外推到 2-4× 训练长度 |
| **Position Interpolation** (Chen 2023) | 用已有位置线性插值扩展 |
| **YaRN** (Peng 2023) | 按频段组合内插/外推,效果更好 |
| **LongRoPE** (Ding 2024) | 非均匀频率搜索,扩到 2M+ |
| **3D RoPE / Multimodal RoPE** | 图像、视频、多模态扩展 |

这些工作的共同前提是:**RoPE 的 $\theta_i = 10000^{-2i/d}$ 这套频率谱,本质就是可调的**。调频率就能调外推能力。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **相对位置本身就在 attention 的点积里**——如果能把位置编码设计成"点积后剩下的只是相对位置",一切都变简单了。RoPE 就是把这件事从"愿望"变成"恒等式"
2. **几何视角看位置编码**:复数旋转 + 远距离衰减,不是玄学而是 $O(d)$ 内的闭式结果
3. **零侵入是护城河**:能无伤融入任何 attention 变体的模块,才能最终席卷整个领域
4. **频率谱 $\theta_i$ 是长度外推的旋钮**:调频率就等于调"模型在哪个尺度上用 attention"。YaRN、LongRoPE 等本质都是这件事
</callout>

---

## 延伸阅读

- [RoFormer 苏神原博(让研究人员绞尽脑汁的 Transformer 位置编码)](https://spaces.ac.cn/archives/8265)
- [Self-Attention with Relative Position Representations (Shaw et al., 2018)](https://arxiv.org/abs/1803.02155) —— 相对位置编码的早期尝试
- [YaRN: Efficient Context Window Extension (Peng et al., 2023)](https://arxiv.org/abs/2309.00071) —— RoPE 频率谱调节用于超长外推
- [Position Interpolation (Chen et al., 2023)](https://arxiv.org/abs/2306.15595) —— 线性插值法扩展 RoPE
- [LongRoPE: Extending LLM Context Window Beyond 2 Million (Ding et al., 2024)](https://arxiv.org/abs/2402.13753) —— 非均匀搜索推到 2M
- [The Impact of Positional Encoding on Length Generalization (Kazemnejad et al., 2023)](https://arxiv.org/abs/2305.19466) —— 对比各种位置编码的外推能力
