---
title: "Reformer — 用 LSH + 可逆网络把 Transformer 的内存需求降两个量级"
date: 2026-04-23 23:50:00 +0800
categories: [Attention, Memory Efficiency]
tags: [reformer, lsh, locality-sensitive-hashing, reversible-network, kitaev-2020]
math: true
---

## 基本信息

- **作者**: Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya
- **机构**: UC Berkeley, Google Research
- **发表**: ICLR 2020
- **arXiv**: [2001.04451](https://arxiv.org/abs/2001.04451)

## 一句话总结

提出 **Reformer**——把原始 Transformer 两个最耗内存的部分同时砍下来:
1. **LSH Attention**:用局部敏感哈希(Locality-Sensitive Hashing)让相似的 query/key 落在同一桶,只在桶内做 attention,把 $O(N^2)$ 变成 $O(N \log N)$
2. **Reversible Residual Layers**:可逆网络让中间激活可以从输出反推出来,**反向传播不需要存任何层的激活**,内存消耗不随层数增长

配合共享 QK 矩阵 + 分块 FFN,整体显存从 ~16 GB 级降到**可在单张 Tesla V100 16GB 上跑 64K 上下文**。

![LSH 的 angular 变体:把 Q 和 K 球面投影后做随机旋转,根据每个点的"符号轴"投影取 argmax 决定桶号。相近向量大概率落同桶,相远向量大概率分桶。](/assets/img/reformer/x1.png)
_Figure 1:Angular LSH——相似向量落同桶的几何诠释_

---

## 背景:Transformer 长序列的两座大山

2020 年的时候,训练 Transformer 遇到两个几乎独立的显存瓶颈:

### 第一座山:Attention 的 $O(N^2)$ 内存

前向计算 attention 矩阵 $N \times N$,反向传播还要保留它——$N = 64K$ 时光 attention 矩阵就 16 GB bf16。

### 第二座山:每一层激活都要保存

反向传播要计算梯度,每层的激活要保存。一个 12 层 Transformer,每层激活 $\sim 2N \cdot d$ bf16——**随层数线性增长**,很快爆显存。

**Reformer 同时拿下两座山**。

---

## 核心机制 1:LSH Attention

### 观察:Softmax 的稀疏性

$\text{softmax}(QK^\top / \sqrt{d_k})$ 的输出大部分情况下**主要由极少数最大分数决定**——其他分数指数衰减后贡献近似 0。

如果我们事先知道每个 query 的"topk 个 key",其他根本不用算。**LSH 就是快速近似 topk 的工具**。

### Locality-Sensitive Hashing

LSH 的性质:**相似的向量大概率 hash 到同桶,不相似的大概率不同桶**。Reformer 用 **angular LSH**:

1. 给定向量 $x$,做**球面投影**(归一化到单位球)
2. 用 $b/2$ 个随机旋转矩阵 $R$
3. 对 $Rx$ 按"**符号轴 argmax**"取桶号:$h(x) = \arg\max([Rx;\, -Rx])$
4. 使用 $b$ 个桶(由于正负轴都用,实际有 $b$ 个分区)

几何上:两个向量夹角越小,同桶概率越高(Figure 1)。

### 算法:Hash → Sort → Chunk → Attend

![LSH Attention 的工作流程:(a) 原始 attention 矩阵;(b) 按 hash 分桶重排;(c) 桶内排序后按块切分;(d) 仅在块内和相邻块做 attention。从 N² 的稠密矩阵变成块对角 + 相邻块的稀疏模式。](/assets/img/reformer/x2.png)
_Figure 2:LSH Attention 的 Hash → Sort → Chunk 流程_

具体步骤:

1. 对每个 Q/K 向量计算 hash $h(q)$,相同 hash 的归同桶
2. 按 hash 排序序列,把同桶 token 聚到相邻位置
3. 把排序后的序列切成固定大小的 chunk
4. 每个 token 只看同 chunk + 相邻 chunk 的 token(不跨桶 attend)
5. 为了减少碰撞误差,**多轮 hash** 取结果的并集

### 关键技巧:共享 Q 和 K

为了让 attention 是对称的(相似的 $q_i, k_j$ 一定在同桶,无论谁 hash),Reformer 强制 **$W_Q = W_K$**——即 $Q = K$。这看似限制,但实验证明在 LSH 场景下质量不降,反而因参数减半更稳定。

### 复杂度

- 理论上:$O(N \log N)$(排序 + 块内注意)
- 实践中($N = 64K$,bucket 数 $\sim 256$):相较 full attention **加速 10-100×**

---

## 核心机制 2:Reversible Transformer

### 背景:可逆残差网络(RevNet)

标准残差网络 $x_{l+1} = x_l + F(x_l)$,反向传播要存 $x_l$。
RevNet(Gomez 2017)把激活分成两半 $(X_1, X_2)$,前向定义:

$$
Y_1 = X_1 + F(X_2),\quad Y_2 = X_2 + G(Y_1)
$$

反向可以从 $(Y_1, Y_2)$ **精确恢复** $(X_1, X_2)$:

$$
X_2 = Y_2 - G(Y_1),\quad X_1 = Y_1 - F(X_2)
$$

**不需要保存任何中间激活**!所有层的激活都可以从最顶层反推出来。

### 应用到 Transformer

Reformer 把每个 transformer block 改成 reversible 形式:

- $F$ 承担 attention 子层
- $G$ 承担 FFN 子层

结果:内存随层数**恒定**,不再线性增长。12 层和 120 层的反向内存一样。

![共享 QK(左)和 可逆网络(右)对质量的影响:两个改动在 enwik8 和 imagenet64 上的训练曲线几乎与标准版本重合——即"内存优化"对质量几乎无副作用。](/assets/img/reformer/x3.png)
_Figure 3:共享 QK 与可逆网络——内存极省,质量不降_

### 代价

- **反向时要多一次前向**(重算激活)——计算量 ~1.3×,但对显存的节省换这点时间值得
- 实现比较精巧,PyTorch 不原生支持(作者写了专门的 autograd 扩展)

---

## 实验结果

### 质量与 hashing 鲁棒性

![LSH 轮数(1, 2, 4, 8)对 imagenet64 上生成质量的影响:轮数越多越接近 full attention。4 轮后收益已很小。](/assets/img/reformer/x4.png)
_Figure 4:多轮 LSH 投票——4 轮已逼近 full attention 的质量_

- 4 轮 LSH 后质量接近 full attention
- 更多轮数计算量翻倍,但收益递减

### 层数与速度

![左:LSH attention 在 enwik8 上随层数的 bpc 变化;右:评估速度随输入长度的对比。LSH attention 在 16K+ 长度明显领先,32K 时快近 100×。](/assets/img/reformer/x5.png)
_Figure 5:层数可伸缩性(左)与长序列加速(右)_

在 enwik8 和 imagenet64:

- 质量与 full attention 持平
- seq_len = 16K:Reformer 比 Transformer 快 ~16×
- seq_len = 64K:Transformer 直接 OOM,Reformer 照跑

---

## 为什么 Reformer 没有成为事实标准

Reformer 当时惊艳,但后来**实战采用有限**。原因:

### 1. LSH 的实现复杂

- 需要多轮 hash 取并集
- PyTorch 没有原生高效 LSH kernel
- 工程复现门槛高,社区很难复用

### 2. 对 decode 不够友好

和 Linformer 一样,LSH attention 在自回归生成时**很难增量更新**——新 token 的 hash 未知,桶结构要重算。这在 LLM 时代是致命短板。

### 3. 被 FlashAttention 在前向端超车

FlashAttention(2022)用 tiling + recomputation 在**保持精确 attention 的前提下**把前向速度和内存占用都降了。Reformer 的"近似换速度"不再必要。

### 4. 可逆网络的遗产仍在

尽管 LSH 那套在 LLM 时代淡出,**可逆网络思想仍被沿用**:

- **Gradient checkpointing** 是可逆的温和版本(只存部分激活,其余重算)
- **FlashAttention 的 backward 重算** 沿用同样哲学
- **Mamba 的 SSM scan** 反向也需要精巧的激活复用

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Softmax 是稀疏的**,不需要算完整 attention 矩阵——这个洞察启动了后续 Top-K attention、LSH attention、Block-Sparse attention 一整条路线
2. **可逆网络让内存与层数解耦**:这个思想在今天 gradient checkpointing、activation recomputation 里继续发挥作用
3. **"近似 attention" 的命运**:Reformer、Linformer、Performer 都不约而同在 LLM 时代被 FlashAttention 类精确方法超车——这提醒我们"近似"只在硬件约束下成立,一旦硬件优化到位,精确方法可能反超
4. **LSH 在向量检索领域仍很活跃**:Reformer 里的 angular LSH 直到今天在 ANN(近似最近邻)里仍是基础算法之一,只是不在 attention 层内部用了
</callout>

---

## 延伸阅读

- [Linformer 深度解读]({% post_url 2026-04-23-Linformer-低秩自注意力深度解读 %}) —— 另一条 2020 年的线性 attention 路线
- [Performer (Choromanski et al., 2020)](https://arxiv.org/abs/2009.14794) —— 用随机特征 FAVOR+ 近似 softmax
- [FlashAttention 深度解读]({% post_url 2026-04-23-FlashAttention-IO感知精确注意力深度解读 %}) —— 保持精确的对立方案
- [The Reversible Residual Network (Gomez et al., 2017)](https://arxiv.org/abs/1707.04585) —— 可逆网络原论文
- [Mamba 深度解读]({% post_url 2026-04-23-Mamba-选择性状态空间模型深度解读 %}) —— 后续的"真正线性"方案
