---
title: "FlashAttention: IO-Aware 精确注意力 — 把 Attention 从 memory-bound 里解救出来"
date: 2026-04-23 14:40:00 +0800
categories: [Attention, Systems, LLM Infrastructure]
tags: [flashattention, attention, io-aware, tiling, recomputation, memory-bound, tri-dao]
math: true
---

## 基本信息

- **作者**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- **机构**: Stanford University, University at Buffalo
- **发表**: NeurIPS 2022
- **arXiv**: [2205.14135](https://arxiv.org/abs/2205.14135)
- **代码**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

## 一句话总结

提出 **IO-Aware 的精确注意力算法 FlashAttention**，通过 **Tiling（分块计算）** 和 **Recomputation（反向时重算）** 两个技巧把注意力从 $O(N^2)$ 的 **HBM 显存读写** 压到 $O(N^2 d^2 / M)$（其中 $M$ 是 SRAM 大小），在 **FLOPs 完全没变**的情况下，GPT-2 训练**加速 3×**、长序列任务上**支持此前无法训练的 16K-64K 上下文**，并顺手推出 **块稀疏变体 Block-Sparse FlashAttention**。是现代 LLM 训练/推理栈的事实基础设施。

![FlashAttention 总览：左-分块策略避免物化 N×N 注意力矩阵；中-GPT-2 上相同 FLOPs 但墙钟时间大幅下降；右-长序列建模质量提升](/assets/img/flashattention/x1.png)
_Figure 1：FlashAttention 总览——左侧分块算法、中间墙钟时间对比、右侧长序列质量提升_

---

## 核心问题：注意力为什么慢？

### 被误解的瓶颈

在 FlashAttention 之前，几乎所有"efficient attention"的工作都在想办法**降低 FLOPs**：

- Sparse Attention：只算部分位置（O(N√N)）
- Linformer/Performer：用低秩或随机特征近似（O(N)）
- Reformer：用 LSH hash 把相似 query 放一起

但它们在实际 GPU 上**并没有变快多少**，有时甚至更慢。为什么？

> **核心洞察**：在现代 GPU 上，注意力**不是 compute-bound，而是 memory-bound**。
> FLOPs 不是瓶颈——**HBM（显存）读写才是**。

### GPU 存储金字塔

现代 GPU（以 A100 40GB 为例）的存储层级差距极大：

| 层级 | 容量 | 带宽 | 延迟 |
|------|------|------|------|
| **SRAM**（on-chip，每个 SM） | 192 KB | ~19 TB/s | 极低 |
| **HBM**（显存） | 40 GB | ~1.5 TB/s | 高（几百 cycle） |
| 主机内存 | 系统内存 | ~30 GB/s | 非常高 |

SRAM 带宽比 HBM 快约 **13×**，但容量只有 HBM 的百万分之一。**一次 HBM 往返就能抹掉所有 FLOPs 省下的时间。**

### 标准 attention 的 HBM 行为

标准实现（Algorithm 0，来自原论文）：

```
1. 从 HBM 读 Q, K → 写 S = QK^T 到 HBM           # 写一个 N×N 矩阵
2. 从 HBM 读 S → 写 P = softmax(S) 到 HBM        # 读+写 N×N
3. 从 HBM 读 P, V → 写 O = PV 到 HBM             # 再读 N×N
```

每一步都在 HBM 和 SRAM 之间搬运 $O(N^2)$ 数据，且 **$N \times N$ 矩阵要物化到 HBM**。当 $N = 8192$、fp16 时，仅这一个 $S$ 矩阵就占 **128 MB**（×batch×heads 之后轻松爆显存），而 softmax、mask、dropout 这些 pointwise 操作几乎全是 memory-bound——**算力大部分时间在等数据**。

![A100 上标准 attention vs FlashAttention 的 runtime 分解。左：前向+反向总耗时，HBM 访问是主导；中：HBM 访问次数降低带来加速；右：块稀疏变体进一步提速](/assets/img/flashattention/x2.png)
_Figure 2：A100 上 GPT-2 medium 的 runtime 分解——HBM 访问次数决定墙钟时间_

Figure 2 验证了核心论点：**减少 HBM 访问次数就减少了墙钟时间**，即使 FLOPs 不降。

---

## 核心算法：Tiling + Recomputation

FlashAttention 的目标是：**在不物化 $S=QK^T$ 到 HBM 的前提下，计算出最终输出 $O$**。

两个关键技巧：

1. **Tiling（分块计算）**：把 Q/K/V 切成小块，每块能装进 SRAM；在 SRAM 内完成 $QK^T$、softmax、乘 $V$ 的一条龙
2. **Recomputation（反向重算）**：前向不保存 $S$、$P$；反向时用前向保存的 softmax 归一化统计量**重算** $S$、$P$，省掉 $O(N^2)$ 的中间存储

### 难点：softmax 本质是全局的

Softmax 的定义是：

$$
\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

要归一化第 $i$ 行，必须先知道**整行的最大值**和**整行的指数和**。分块之后，每块只看到自己的一小段数据，怎么办？

### 解法：Online Softmax

逐块累积**运行中的最大值**和**归一化分母**，每见到新块就**"修正"前面已算出的部分结果**。

设两个块 $x^{(1)}, x^{(2)}$，记：

- $m(x) = \max(x)$（最大值）
- $\ell(x) = \sum_j e^{x_j - m(x)}$（归一化分母）
- $\text{softmax}(x) = e^{x - m(x)} / \ell(x)$

合并两个块时有：

$$
m = \max(m^{(1)}, m^{(2)}),\quad \ell = e^{m^{(1)} - m}\ell^{(1)} + e^{m^{(2)} - m}\ell^{(2)}
$$

这就是 **online softmax** 的更新规则——只需 $O(1)$ 的额外状态（当前最大值 $m$ 和当前分母 $\ell$），就能**增量**地算出整行的 softmax，完全不需要物化整行。

### Tiling 主循环（原论文 Algorithm 1 的直观版本）

```
输入: Q, K, V ∈ R^{N×d}  (在 HBM 中)
输出: O ∈ R^{N×d}        (写回 HBM)

Tc = ceil(N / Bc)      # K/V 的块数
Tr = ceil(N / Br)      # Q 的块数

在 HBM 中初始化 O = 0, ℓ = 0, m = -∞

for j = 1 ... Tc:                               # 外循环:K/V 块
    从 HBM 加载 Kj, Vj 到 SRAM
    for i = 1 ... Tr:                           # 内循环:Q 块
        从 HBM 加载 Qi, Oi, ℓi, mi 到 SRAM

        # 以下全部在 SRAM 内完成,无 HBM 写回中间结果
        Sij       = Qi @ Kj^T                   # (Br, Bc)
        m̃ij       = rowmax(Sij)                 # 当前块的行最大值
        P̃ij       = exp(Sij − m̃ij)
        ℓ̃ij       = rowsum(P̃ij)

        # online softmax 合并前面累积的 (mi, ℓi) 与当前块
        mi_new   = max(mi, m̃ij)
        ℓi_new   = exp(mi − mi_new)·ℓi + exp(m̃ij − mi_new)·ℓ̃ij

        # 修正先前写入的 Oi,并加入当前块的贡献
        Oi ← diag(ℓi_new)^{-1} · ( diag(ℓi)·exp(mi − mi_new)·Oi
                                 + exp(m̃ij − mi_new) · P̃ij @ Vj )

        把 Oi, ℓi_new, mi_new 写回 HBM
```

**关键点**：

- 两层循环里，每块 $Q_i, K_j, V_j$ 都能放进 SRAM
- $S_{ij}$（大小只有 $B_r \times B_c$）**永远不出 SRAM**
- 每个 $O_i$ 只在外循环每迭代一次时更新一次（**online softmax**负责修正）
- HBM 的总读写量从 $O(N^2)$ 降到 $O(N^2 d^2 / M)$，其中 $M$ 是 SRAM 大小

### Recomputation：反向传播不存 $S$、$P$

反向传播需要 $\frac{\partial L}{\partial Q}, \frac{\partial L}{\partial K}, \frac{\partial L}{\partial V}$，常规实现要用到前向算出的 $S$、$P$——这俩各是 $N \times N$ 的矩阵，存下来就爆显存。

FlashAttention 的做法：**前向只保存 $m$（行最大值）和 $\ell$（行分母），反向时从 $Q, K, V, m, \ell$ 重新计算 $S$、$P$**。

重算虽然多了 FLOPs，但**避免了 HBM 读写**，反而更快。额外内存从 $O(N^2)$ 降到 $O(N)$，训练长上下文成为可能。

---

## 复杂度对比

| 指标 | 标准 Attention | FlashAttention |
|------|----------------|----------------|
| FLOPs | $O(N^2 d)$ | $O(N^2 d)$（**相同**） |
| HBM 访问 | $\Theta(Nd + N^2)$ | $\Theta(N^2 d^2 / M)$ |
| 额外内存（反向） | $O(N^2)$ | $O(N)$ |
| 典型 $d=64, M=100KB$ 时 HBM 访问 | $N^2 \approx 10^8$ 量级 | 约降低 **$9\times$** |

<callout emoji="bulb" background-color="light-blue" border-color="blue">
FLOPs 一分没变,快就快在**少读少写**。这解释了为什么它能在学术上称为"exact attention",而不是"approximate"——它和标准 attention 数学上**完全等价**,只是数据搬运路径改了。
</callout>

---

## 实验结果

### 训练加速：GPT-2、BERT 端到端时间

论文在多种模型和序列长度下做了端到端训练时间对比：

- **BERT-large (seq 512)**：比 MLPerf 训练记录快 **15%**
- **GPT-2 small/medium (seq 1K)**：比 HuggingFace 原版快 **3×**，比 Megatron 快 **1.8×**
- **Long-range arena (seq 1K–4K)**：整体快 **2.4×**

### 内存 & runtime 随序列长度的 scaling

![A100 40GB 上不同 attention 实现的 runtime 与内存随序列长度的变化。FlashAttention 在内存和速度上都显著优于标准实现与常见 approximate 方案](/assets/img/flashattention/x3.png)
_Figure 3：不同 attention 实现的 runtime（左）与显存（右）随序列长度的 scaling_

关键观察：

- 标准 attention 在 seq ≥ 2K 时就开始 OOM
- FlashAttention 一路扩到 **16K** 仍然正常；**Block-Sparse FlashAttention** 更进一步支持 **64K**
- 许多 approximate 方法的**实际**内存消耗并不比标准 attention 低多少，因为它们引入了自己的开销

### 质量无损：perplexity 曲线完全重合

很多"高效 attention"以精度换速度。FlashAttention 的承诺是**数学等价 + 数值近似到 exp 的精度**，因此质量曲线应该和标准实现完全一致。

![GPT-2 small/medium 上 FlashAttention vs HuggingFace 标准实现的验证 perplexity 曲线。两者完全重合,证明 FlashAttention 是精确的、非近似的](/assets/img/flashattention/x4.png)
_Figure 4：验证 perplexity 曲线完全重合——FlashAttention 与标准实现数学等价_

这是 FlashAttention 能成为主流基础设施的关键：**调一行 kernel 就能加速训练，没有任何模型质量下降风险**。

### 更长的上下文带来更好的模型

得益于显存占用降到 $O(N)$，可以在同样硬件上训练**前所未有**的长上下文：

| 任务 | seq length | 结果 |
|------|-----------|------|
| **Path-X (LRA)** | 16K | **首次**有 Transformer 超过随机猜测（**61.4%**） |
| **Path-256** | 64K | **首次**有 Transformer 超过随机猜测（**63.1%**，Block-Sparse FlashAttention） |
| GPT-2 (长文档建模) | 4K | perplexity 降低 **0.7** |

这两个结果在当时意义重大——此前 Transformer 几乎被认为"学不会" Path-X 这种极长距离依赖任务。

---

## Block-Sparse FlashAttention（扩展）

既然 tiling 已经把计算组织成块，那么**整块跳过**就自然形成块稀疏 attention：

- 给定块稀疏 mask，跳过那些整块为 0 的 $(i, j)$ 对
- 复杂度变成 $O(N^2 d^2 s / M)$，其中 $s$ 是稀疏度
- 在 64K 序列上跑得通，Path-256 任务由此被攻克

<callout emoji="zap" background-color="pale-gray" border-color="gray">
这是 tiling 算法的"副产品":数据搬运路径清晰之后,稀疏化是顺其自然的事情。对比之下,传统 sparse attention 在标准实现下几乎无法高效落地,因为它们的内存访问模式对 GPU 极不友好。
</callout>

---

## 为什么影响如此之大

### 1. 它是 **system-first** 的研究

FlashAttention 之前的 efficient attention 大多是算法论文：发明一个新近似，证明理论复杂度低，然后套进 Transformer。**但它们往往忽略了真实硬件的 IO 模型**。FlashAttention 的起点是 roofline 分析，终点是工程 kernel——这种研究范式后来被证明是通往实际加速的唯一路径。

### 2. 它改写了后续所有 attention 变体的"出厂标配"

2022 年以后发表的 attention 变体几乎都要回答一个问题：**"你能不能跑 FlashAttention 路径"**？

- GQA / MQA / MLA 都是在 FlashAttention 之上做 KV 头共享，**反过来验证了 decode 瓶颈在 KV cache 带宽**
- SlidingWindow / Block-Sparse 在 FlashAttention 的 tile 上做 mask
- FlashAttention-2（2023）把 block 分配做了更好的 warp 映射
- FlashAttention-3（2024）利用 Hopper 的 WGMMA 和 TMA 做异步流水

### 3. 把 "long context" 从不可能变成日常

没有 FlashAttention 和它的后继者，今天的 32K/128K/1M 上下文模型无法存在——既训不动也跑不起。Claude 的 200K、Gemini 的 1M，底层都依赖这一系列 kernel。

### 4. 塑造了一种研究文化

"**减少数据搬运而不是减少 FLOPs**" 成为系统 ML 研究的默认范式。后续 PagedAttention、RingAttention、Ring-Flash、FlashDecoding、FlashInfer 都是同一思想路径。

---

## 读完 FlashAttention 你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **GPU 瓶颈分析要从 roofline 开始**:先量算术强度,再决定算法设计方向
2. **Tiling 的本质是"让数据住在 SRAM 里"**:凡是有"大中间张量"的算子都能套这个范式
3. **Online softmax 是可以分块累积的,这件事本身就值得记在脑子里**——后来很多长上下文算法的核心
4. **精确 vs 近似不是算法边界,是工程边界**:如果一个精确算法的 IO 设计得当,可能根本不需要近似
</callout>

---

## 延伸阅读

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)](https://arxiv.org/abs/2307.08691) —— warp/thread 级并行优化
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (Shah et al., 2024)](https://arxiv.org/abs/2407.08608) —— Hopper 架构 WGMMA/TMA 适配
- [Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867) —— FlashAttention 用到的 online softmax 原始工作
- [Efficient Memory Management for LLM Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) —— 把 IO-aware 思想带到推理服务
- [Making Deep Learning Go Brrrr from First Principles (Horace He)](https://horace.io/brrr_intro.html) —— roofline 分析的入门博客
