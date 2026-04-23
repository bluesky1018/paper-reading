---
title: "Performer — 用随机特征 FAVOR+ 把 Softmax Attention 近似为线性复杂度"
date: 2026-04-23 23:15:00 +0800
categories: [Attention, Linear Attention, Kernel Methods]
tags: [performer, favor-plus, random-features, kernel-methods, choromanski-2020]
math: true
---

## 基本信息

- **作者**: Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller
- **机构**: Google, University of Cambridge, DeepMind, Alan Turing Institute
- **发表**: ICLR 2021
- **arXiv**: [2009.14794](https://arxiv.org/abs/2009.14794)

## 一句话总结

提出 **Performer** 与其核心算法 **FAVOR+ (Fast Attention Via positive Orthogonal Random features)**——用**正值随机特征**近似 softmax kernel,把原 attention 的 $O(N^2)$ 复杂度降到 $O(N)$,且**无偏、可证明误差界、兼容 causal decode**。是 2020 年线性 attention 潮中**数学最优雅**、理论最完备的一支,也是后来所有"基于核方法近似 attention"工作(如 Linear Transformer、RetNet、Gated Linear Attention)的数学基石。

![Linear Transformer 的核心结构:softmax(QK^T)V 的 O(N²) 计算被分解为 φ(Q)·(φ(K)^T·V) 的两步线性操作,通过结合律重新组合,时间/空间复杂度变为 O(N)。](/assets/img/performer/x1.png)
_Figure 1:线性 Transformer 的矩阵结合律技巧——把 O(N²) 换成 O(N)_

---

## 背景:核方法看 Attention

### Softmax attention 的核表示

给定 $Q, K \in \mathbb{R}^{N \times d}$,$V \in \mathbb{R}^{N \times d}$,定义:

$$
\text{Attn}(Q, K, V)_i = \frac{\sum_{j=1}^N \exp(q_i^\top k_j / \sqrt{d}) v_j}{\sum_{j=1}^N \exp(q_i^\top k_j / \sqrt{d})}
$$

把 $\exp(q_i^\top k_j)$ 看作一个**正定核**:$K(q, k) = \exp(q^\top k)$。这个核对应一个**无限维特征映射** $\phi$,即 $K(q, k) = \phi(q)^\top \phi(k)$。

### 核方法的结合律技巧

如果 $\phi$ 是**有限维**(比如 $\mathbb{R}^m$),可以这样改写:

$$
\text{Attn}(Q, K, V)_i = \frac{\sum_j \phi(q_i)^\top \phi(k_j) v_j}{\sum_j \phi(q_i)^\top \phi(k_j)} = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
$$

**关键**:分子分母都是"先算 $\sum_j \phi(k_j) (\cdot)$",再和 $\phi(q_i)$ 做内积。所以可以:

1. 先预计算 $\sum_j \phi(k_j) v_j^\top \in \mathbb{R}^{m \times d}$(常数大小,与 $N$ 无关)
2. 再对每个 query 做 $\phi(q_i)^\top (\cdot)$

**复杂度 $O(N m d)$,线性于 $N$!**

但挑战是:softmax 的 $\exp$ 核对应的特征映射 $\phi$ 是无限维的。必须**近似**。

---

## 核心创新:FAVOR+

![FAVOR+ 算法示意:softmax attention 矩阵被正值随机特征映射近似。每个位置 i 的 query/key 向量 $q_i, k_j$ 经过 $\phi$ 后,其点积 $\phi(q_i)^\top \phi(k_j)$ 无偏逼近 $\exp(q_i^\top k_j)$。](/assets/img/performer/x2.png)
_Figure 2:FAVOR+ 用随机特征把 softmax kernel 拆成可结合的内积_

### 随机特征近似

经典随机 Fourier 特征(Rahimi & Recht 2007)可用于近似 Gaussian 核:

$$
\phi_{\text{RFF}}(x) = \frac{1}{\sqrt{m}} [\cos(w_i^\top x), \sin(w_i^\top x)]_{i=1}^m
$$

但对 softmax 的 $\exp(q^\top k)$ 核,直接用 RFF 会得到**包含负数的近似**——而 $\exp > 0$,负数近似导致分母可能为 0 甚至负数,数值不稳定。

### Performer 的关键:正值特征(positive features)

FAVOR+ 设计了一族**非负随机特征**:

$$
\phi_+(x) = \frac{\exp(-\|x\|^2/2)}{\sqrt{m}} \cdot [\exp(w_i^\top x)]_{i=1}^m
$$

可以证明 $\mathbb{E}[\phi_+(q)^\top \phi_+(k)] = \exp(q^\top k)$,**无偏且永远非负**。

### 正交随机特征(Orthogonal features)

进一步,作者证明让 $w_i$ **两两正交**(而不是独立高斯)可以**显著减小近似方差**:

- 对任意固定误差,所需的 $m$ 减少
- 理论证明同时给出误差上界

### 综合算法:FAVOR+

- **F**ast **A**ttention **V**ia positive **O**rthogonal **R**andom features
- **+** 表示 positive(避免 negative variance issue)

**实用中 $m = 256$ 左右就能很好近似** $d = 64$ 的 attention。

---

## 架构整合

![Performer 整体模型视图:与原始 Transformer 结构相同,只是把 MultiHeadAttention 里的 softmax attention 替换成 FAVOR+ attention。其他所有组件(layer norm, FFN, residual)保持不变。](/assets/img/performer/x3.png)
_Figure 3:Performer 架构——"即插即用"替换 Transformer 的 attention 模块_

**Performer 不改模型架构**,只替换 attention 计算方式。这意味着:

- 现有 Transformer 代码几乎 drop-in 替换
- 预训练权重可以继续用(或轻微 fine-tune)
- 其他优化(残差、layer norm、FFN 等)兼容

---

## 因果 attention 的优雅实现

这是 Performer 相对 Linformer / Reformer 的一个关键优势——**天然支持 causal decode**。

因为 attention 被拆成 $\phi(Q)^\top \cdot (\sum_j \phi(k_j) v_j^\top)$,对 causal 版本只要把 $\sum_j$ 换成**前缀和**:

$$
S_i = \sum_{j \leq i} \phi(k_j) v_j^\top
$$

然后 $S_i = S_{i-1} + \phi(k_i) v_i^\top$——**recurrent 更新**。自回归生成时,每步只要更新 $S$ 和归一化项,$O(1)$ 时间、$O(md)$ 内存,**没有 KV cache 爆炸问题**。

这个"前缀和/scan"思想后来被 RetNet、Mamba、DeltaNet 等继承,演化出整个 linear-attention/SSM 家族。

---

## 实验结果

### 逼近精度

![Softmax attention 矩阵(左)与 FAVOR+ 近似(右)的对比。多数位置两者几乎相同,差异集中在极高值。](/assets/img/performer/x4.png)
_Figure 4:Softmax 原矩阵 vs FAVOR+ 近似——视觉上几乎不可区分_

随机特征维度 $m$ 增大,近似误差指数衰减。实用中 $m = 256$ 对 $d = 64$ 的 attention 误差已很小。

### 长序列任务

- **Protein modeling (TrEMBL)**:原 Transformer 最多训到 1K,Performer 能训 **8K**,PPL 更低
- **ImageNet 64×64 生成**:与 Sparse Transformer 持平,但代码简单得多

![Performer 在长序列蛋白质建模(左)与 ImageNet 64x64 生成(右)上的表现。长序列场景下优势尤其明显,原 Transformer 直接 OOM。](/assets/img/performer/x5.png)
_Figure 5:长序列蛋白质建模与图像生成——Performer 在原 Transformer 不可行的场景稳定工作_

### 速度与显存

- 训练速度:$N = 1024$ 快 2×,$N = 4096$ 快 6×,$N = 8192$ 快 12×
- 显存:$N = 8192$ 时 Transformer OOM,Performer ~ 10GB

---

## 局限与后续演化

### 局限 1:预训练迁移质量损失

用 FAVOR+ 替换训好的 Transformer 的 attention,在下游任务上会有 ~1-2 分的下降。主要原因是随机投影的噪声对极大 logit 敏感。这限制了 Performer 在大模型 fine-tune 场景的使用。

### 局限 2:随机特征方差

理论上近似是无偏的,但**单次采样的方差**不小。特别在深层网络中,方差会层层放大。需要足够大 $m$ 才稳。

### 局限 3:被 FlashAttention 超车

2022 年 FlashAttention 用**精确 attention + IO 优化**在速度上追平甚至超越了 Performer 的近似方案——而且质量不丢。这让 Performer 在 LLM 训练主线中淡出。

### 影响:RetNet、Mamba、Gated DeltaNet 的数学基础

Performer 的"**$\phi(Q)^\top \phi(K)$ 分解 + prefix-sum**"思想**直接启发了**:

- **Linear Transformer** (Katharopoulos 2020)—— 更简单的 $\phi = \text{elu+1}$
- **RetNet** (2023)—— 加上衰减 $\gamma^t$
- **Mamba** (2023)—— 把 $\phi(k)v$ 的 prefix-sum 替换为 SSM 递推
- **Gated Linear Attention** (2024)—— 再加一个逐 token 的门

今天所有 linear attention 的"**矩阵结合律 + 前缀和**"范式,可以追溯到 Performer 的 FAVOR+。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Softmax attention 本质上是一个核函数**:把它换成可分解的有限维核,就能用结合律把 $O(N^2)$ 降到 $O(N)$。这是线性 attention 的数学起点
2. **随机特征必须正值**:FAVOR+ 相对朴素 RFF 的最大技术突破。这个"positivity trick" 被后续工作普遍采用
3. **Prefix-sum 是 causal linear attention 的核心**:$S_i = S_{i-1} + \phi(k_i)v_i^\top$,让自回归 decode 天然线性且无 KV cache
4. **近似 vs 精确是动态天平**:硬件未优化时近似赢,硬件优化到位精确赢。Performer → FlashAttention 的兴衰就是例证
</callout>

---

## 延伸阅读

- [Linformer 深度解读]({% post_url 2026-04-23-Linformer-低秩自注意力深度解读 %}) —— 同期另一条线性 attention 路线
- [Reformer 深度解读]({% post_url 2026-04-23-Reformer-LSH注意力可逆网络深度解读 %}) —— 同期 LSH + 可逆网络路线
- [Transformers are RNNs (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236) —— Linear Transformer,用 $\phi = \text{elu+1}$ 的极简版本
- [Mamba 深度解读]({% post_url 2026-04-23-Mamba-选择性状态空间模型深度解读 %}) —— 前缀和思想的现代化身
- [FlashAttention 深度解读]({% post_url 2026-04-23-FlashAttention-IO感知精确注意力深度解读 %}) —— 精确路线的反超
