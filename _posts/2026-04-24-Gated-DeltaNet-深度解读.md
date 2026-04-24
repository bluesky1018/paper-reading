---
title: "Gated DeltaNet — 把 DeltaNet 加一个门,让 linear attention 学会'记忆管理'"
date: 2026-04-24 10:30:00 +0800
categories: [Attention, Linear Attention, State Space]
tags: [gated-deltanet, delta-rule, linear-attention, yang-2024]
math: true
---

## 基本信息

- **作者**: Songlin Yang, Jan Kautz, Ali Hatamizadeh
- **机构**: MIT, NVIDIA
- **发表**: arXiv 2024-12
- **arXiv**: [2412.06464](https://arxiv.org/abs/2412.06464)

## 一句话总结

提出 **Gated DeltaNet**——把 Schmidhuber 1992 年的 **Delta Rule (Fast Weight)** 重新搬回 linear attention 时代,并加入**输入自适应的 gating**。结果:在同参数下**兼具精确 recall(DeltaNet 的强项)和长距离 forget(Gating 的强项)**,在关联召回、复杂推理、code 等 benchmark 上超越 Mamba-2、GLA、RetNet 等所有 linear-attention 家族,且保持 $O(1)$ decode 成本。是 2024 年底 linear-attention 路线上的一个重要里程碑。

![Gated DeltaNet 的 layer 结构:delta 更新(精确 write)+ gating(soft forget)的组合。](/assets/img/gated-deltanet/x1.png)
_Figure 1:Gated DeltaNet layer——delta 写入 + 门控衰减_

---

## 背景:Linear Attention 的两大瓶颈

到 2024 年底,linear attention 已经有一个繁荣的家族:Linear Transformer、RetNet、Mamba、Mamba-2、Gated Linear Attention (GLA)、DeltaNet。但**没有一个能在全部任务上打败 Transformer**。

问题的症结有两个:

### 1. 精确 recall 弱

大多数 linear attention 的 state 更新是 $s_t = \gamma \cdot s_{t-1} + k_t v_t^\top$,新信息"加"到已有状态上。这让**重写(overwrite)成为难题**——要覆盖旧信息,只能靠 $\gamma$ 衰减它,但衰减是"广播式"的,影响所有信息。

### 2. 忘记能力弱

有些 linear attention(DeltaNet)具备写入精确性,但缺乏**主动遗忘**机制。固定的 state 容量被占满后,新信息就覆盖不了,历史信息也没法丢掉。

Gated DeltaNet 的目标:**同时拿下 DeltaNet 的精确写入 + Gating 的软遗忘**。

---

## 核心机制

### 1. Delta Rule:精确的 Fast Weight 更新

Schmidhuber 1992 年提出的 Fast Weight:state 是一个矩阵 $S \in \mathbb{R}^{d \times d}$,每步 delta 更新:

$$
S_t = S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^\top / \|k_t\|^2
$$

这个更新的作用是:**强制 $S_t k_t = v_t$**(精确 associate key $k_t$ 到 value $v_t$)。

直觉:如果把 $S$ 看成一个"键值字典",Delta Rule 就是"**键 $k_t$ 存在就覆盖,不存在就追加**"——一个精确的 key-value write 操作。

$\beta_t$ 是 learning rate,通常通过线性层从输入预测(input-dependent)。

### 2. Gating:软遗忘

纯 DeltaNet 没有忘记机制。Gated DeltaNet 引入一个**逐通道的衰减门** $\alpha_t \in (0, 1)^d$:

$$
S_t = \alpha_t \odot S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^\top / \|k_t\|^2
$$

![对比三种更新规则的矩阵形式:线性 attention($S_t = S_{t-1} + k v^\top$)、DeltaNet(精确 delta update)、Gated DeltaNet(delta + gating)。](/assets/img/gated-deltanet/x2.png)
_Figure 2:三种 linear-attention 更新规则对比_

- $\alpha_t$ 接近 1:保留大部分旧信息
- $\alpha_t$ 接近 0:几乎清空 state
- 逐通道:不同维度可以有不同"记忆时间尺度"

这个设计的意义:**既能"精确写入新 pair"(delta term),又能"软性丢弃不再需要的旧 pair"(gate term)**。

### 3. 输入自适应的 $\alpha_t, \beta_t$

$\alpha_t, \beta_t$ 不是预设的,而是**从输入 $x_t$ 通过线性层学习得到**:

$$
\alpha_t = \sigma(W_\alpha x_t),\quad \beta_t = \text{swish}(W_\beta x_t)
$$

这让模型能根据当前输入决定:**现在是应该记住(高 $\alpha$)还是遗忘(低 $\alpha$)?这个 key 要覆盖(高 $\beta$)还是不写(低 $\beta$)?**

---

## 高效硬件实现

和 Mamba-2 一样,Gated DeltaNet 的更新可以写成**矩阵形式 + 分块并行算法**:

1. 块内部:用矩阵运算一次算完所有 delta 更新
2. 块之间:用 recurrent 形式传递 state
3. 整体:类似 SSD 的"块并行 + 块间 scan"模式

结果:训练速度逼近 Mamba-2,推理保持 $O(1)$。

![Gated DeltaNet 的三种等价算法形式:Recurrent(推理)、Chunkwise(长序列训练)、Parallel(纯块内)。与 SSD 一脉相承。](/assets/img/gated-deltanet/x3.png)
_Figure 3:Gated DeltaNet 的算法三态_

---

## 实验结果

### 1. 关联召回任务:显著优于 Mamba-2

在 **MQAR (Multi-Query Associative Recall)** 这个专门测试"能记住多少 key-value pair"的任务上:

| 模型 (350M) | 16 pairs | 32 pairs | 64 pairs | 128 pairs |
|------------|----------|----------|----------|-----------|
| Mamba-2 | 99% | 87% | 62% | 34% |
| GLA | 98% | 85% | 58% | 30% |
| DeltaNet (vanilla) | 99% | 97% | 88% | 55% |
| **Gated DeltaNet** | **99.9%** | **99%** | **93%** | **72%** |

Gated DeltaNet 在 key 数量增加时的鲁棒性明显更好——**精确写入能力是与同类 linear attention 的核心差距**。

### 2. 综合 benchmark

在 MMLU / HellaSwag / ARC / PIQA 等综合 benchmark 上:

| 模型 (1.3B) | MMLU | HellaSwag | Ave |
|------------|------|-----------|-----|
| Transformer++ | 32.3 | 55.1 | 48.6 |
| Mamba-2 | 30.8 | 52.7 | 46.3 |
| GLA | 30.5 | 52.3 | 46.1 |
| **Gated DeltaNet** | **33.5** | **56.0** | **49.1** |

首次有 linear-attention 路线在同参数量下**全面超过 Transformer++**。

### 3. 长 context 任务

- **Phone-book (100K tokens)**:Gated DeltaNet 召回率 86%,Mamba-2 65%
- **Long-range arena**:Gated DeltaNet 平均 65,纯 Mamba 60

### 4. 效率:与 Mamba-2 同级

- **Decode TPS**:与 Mamba-2 相当(都是 $O(1)$ 每 token)
- **训练速度**:略慢于 Mamba-2(因为 Delta rule 的额外计算)
- **内存**:常数(不存 KV cache)

---

## 为什么是 linear attention 的重要进展

### 1. 恢复了 Fast Weight 的古老思想

Schmidhuber 1992 年的 Fast Weight 在深度学习热潮中被遗忘了 30 年。Gated DeltaNet 证明**这个古老思想在现代 linear attention 中仍然有效**——精确的 delta 写入 + 现代 gating = 强大的组合。

### 2. 统一了两条技术路线

- **SSM/Mamba 阵营**:靠 gating 实现 selective,但 state 容量有限且写入不精确
- **DeltaNet 阵营**:靠 fast weight 实现精确关联,但没有主动遗忘

Gated DeltaNet 结合二者,让 linear attention 的**理论容量和实际效果都大幅提升**。

### 3. 关联召回能力接近 Transformer

在需要 "记住 N 个 key-value pair 然后准确召回" 的任务上,Gated DeltaNet 是第一个**在大 $N$ 下接近 softmax attention 的 linear 方法**。这为 linear attention 替代 Transformer 提供了关键的一块拼图。

### 4. 为 Hybrid 架构提供更好的 SSM 候选

既然 Gated DeltaNet 比 Mamba-2 在精确 recall 上更强,那 Jamba 等 Hybrid 架构的"Mamba 层"就可以升级为"Gated DeltaNet 层",进一步减少 Attention 层的依赖。

---

## 局限

### 1. 仍然弱于 Transformer 在极限场景

在**超长 context 的精确匹配**(如 needle-in-haystack > 100K)上,Gated DeltaNet 仍不如 full attention。state 容量 $d \times d$ 是硬上限。

### 2. 实现复杂

Delta rule 的 chunkwise 并行算法比 Mamba-2 的 SSD 更复杂,kernel 优化门槛高。

### 3. 超参敏感

$\alpha_t, \beta_t$ 的预测网络初始化不当时容易不稳定。作者给出了一些小技巧,但不如 Mamba 稳健。

### 4. 理论地位未完全梳理

DeltaNet 和 Transformer attention 的数学关系(二者在什么条件下等价)仍在研究中。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **linear attention 的核心痛点是"精确 recall + 主动遗忘"**:之前的模型要么缺 A 要么缺 B,Gated DeltaNet 第一次把二者同时做好
2. **Fast Weight (Delta rule) 是被遗忘的经典**:Schmidhuber 1992 年的思想在现代 linear attention 中焕发生机,值得关注老论文
3. **每通道独立的 gating 是一种"多尺度记忆"**:让模型能根据特征类型决定不同衰减速度——这是从 Mamba 到 Gated DeltaNet 一贯的设计哲学
4. **"Transformer++ 全面超越"的门槛正在被突破**:Gated DeltaNet 是第一个在 MMLU 等综合 benchmark 上同参数打过 Transformer 的 linear attention——这让 2025 年 post-Transformer 架构的可能性大增
</callout>

---

## 延伸阅读

- [Mamba-2 深度解读]({% post_url 2026-04-24-Mamba-2-SSD深度解读 %}) —— Gated DeltaNet 的直接对标
- [Gated Linear Attention (Yang et al., 2024)](https://arxiv.org/abs/2312.06635) —— 前作,加 gate 但没加 delta
- [DeltaNet (Schlag et al., 2021)](https://arxiv.org/abs/2102.11174) —— 现代 DeltaNet 的重新发现
- [Fast Weights (Schmidhuber, 1992)](https://direct.mit.edu/neco/article/4/1/131/5696) —— 真正的起源
- [Performer 深度解读]({% post_url 2026-04-23-Performer-FAVOR随机特征核近似深度解读 %}) —— Linear attention 的数学基础
