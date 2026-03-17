---
layout: post
title: "注意力残差：用深度方向注意力替代固定残差连接"
date: 2026-03-18
categories: [论文解读, 大语言模型]
tags: [残差连接, 注意力机制, LLM, PreNorm, 深度聚合, Kimi, MoE]
---

> 📄 **论文**：Attention Residuals
> 🔗 **arXiv**：[2603.15031](https://arxiv.org/abs/2603.15031)
> 🏢 **机构**：Moonshot AI（Kimi Team）
> 💻 **代码**：[https://github.com/MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)

## 一句话总结

本文提出 Attention Residuals（AttnRes），用 softmax 注意力替代传统固定权重的残差连接，让每一层能够根据输入内容自适应地从之前所有层的输出中选择性聚合信息，在 Kimi Linear 48B 模型上取得了全面的性能提升。

---

## 背景与问题

### 残差连接的局限性

残差连接（Residual Connections）配合 PreNorm 是现代大语言模型（LLM）的标准构建模块。传统残差连接将每一层的输出以固定单位权重进行累加：

$$h_l = h_{l-1} + f_l(h_{l-1})$$

展开这一递推关系，可以发现每一层的隐藏状态实际上是 embedding 和所有前序层输出的均匀加权求和。这种**均匀聚合**造成了以下几个根本性问题：

1. **无选择性访问**：不同类型的层（如注意力层 vs. MLP 层）接收的是相同的聚合状态，但实际上它们可能受益于不同的权重分配；
2. **不可逆信息损失**：通过聚合丢失的信息在更深的层中无法被选择性地恢复；
3. **隐藏状态增长失控（PreNorm Dilution）**：在 PreNorm 架构中，随着深度增加，隐藏状态的模值以 $O(\sqrt{L})$ 的速率增长，使得浅层的相对贡献被逐渐稀释。

### 与 RNN 的类比

作者发现深度方向的残差累加与时间维度的 RNN 循环存在形式上的对偶性：RNN 将历史信息压缩为单一状态，沿序列传播；残差连接同样将所有前序信息压缩为单一状态，沿深度传播。Transformer 用 self-attention 取代了 RNN 的时间循环，带来了序列建模的革命。本文受此启发，将同样的思路应用到深度维度上。

---

## 核心方法

### Full Attention Residuals（FullAttnRes）

AttnRes 将固定的残差累加替换为对前序所有层输出的 softmax 注意力：

$$h_l = \sum_{i=0}^{l-1} \alpha_{l,i} \cdot h_i$$

其中注意力权重计算方式如下。对每一层 $l$，定义一个可学习的伪查询向量 $q_l \in \mathbb{R}^d$，键由前序各层输出经 RMSNorm 归一化后得到：

$$\alpha_l = \text{softmax}\left(\left[q_l^\top \cdot \text{RMSNorm}(h_i)\right]_{i=0}^{l-1}\right)$$

RMSNorm 的作用是防止输出模值较大的层主导注意力权重。这一机制使得每一层可以**根据输入内容动态选择**来自更早层的哪些表示更相关。

![Attention Residuals 总览](https://arxiv.org/html/2603.15031/x1.png)
*图1：Attention Residuals 总览。(a) 标准残差：均匀加法累加；(b) Full AttnRes：每层通过学习的注意力权重选择性聚合所有前序层输出；(c) Block AttnRes：层被分组为若干 block，降低内存和通信开销。*

**计算开销**：在标准训练中，FullAttnRes 几乎不增加额外内存，因为所需的层输出在反向传播时本已保留。但在大规模分布式训练中，激活重计算和流水线并行被广泛采用，此时这些激活需要被显式保存并跨流水线阶段传输，导致内存和通信开销均为 $O(L)$。

### Block Attention Residuals（BlockAttnRes）

为应对大规模训练中的系统挑战，作者提出 BlockAttnRes：将 $L$ 层划分为 $N$ 个 block（每 block 含 $B$ 层），每个 block 内部通过标准残差求和形成一个 block 级表示，跨 block 时只对这些 block 级表示进行注意力计算。

具体地，设第 $n$ 个 block 的表示为：

$$\tilde{h}_n = \sum_{l \in \mathcal{B}_n} f_l(h_{l-1})$$

跨 block 的注意力值矩阵为：

$$V_l = [\tilde{h}_0, \tilde{h}_1, \ldots, \tilde{h}_{n-1}, p_l]$$

其中 $p_l$ 是当前 block 内的部分求和，$\tilde{h}_0$ 是 token embedding。这将内存和通信开销从 $O(L)$ 降至 $O(N)$（$N \ll L$）。

```python
def block_attn_res(blocks: list[Tensor], partial_block, proj_weight, norm):
    # Inter-block attention: attend over block reps + partial sum
    # blocks: N tensors of shape [B, T, d]
    V = torch.stack(blocks + [partial_block])  # [N+1, B, T, d]
    V = norm(V)
    logits = torch.einsum('d,nbtd->nbt', proj_weight.squeeze(), V)
    return torch.einsum('nbt,nbtd->btd', logits.softmax(-1), V)

def forward(self, blocks: list, partial_block, hidden_states):
    # apply block attn res before attn
    h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)
    # if reaches block boundary, start new block
    if self.layer_number % self.block_size == 0:
        blocks.append(partial_block)
        partial_block = None
    # self-attention layer
    attn_out = self.attn(self.attn_norm(h))
    partial_block = partial_block + attn_out if partial_block is not None else attn_out
    # apply block attn res before MLP
    h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
    # MLP layer
    mlp_out = self.mlp(self.mlp_norm(h))
    partial_block = partial_block + mlp_out
    return blocks, partial_block
```

*图2：BlockAttentionResiduals 的 PyTorch 风格伪代码。`block_attn_res` 使用可学习的伪查询计算 block 表示上的 softmax 注意力；`forward` 为单层传递，维护 `partial_block`（block 内部残差）和 `blocks`（跨 block 历史）。*

### 基础设施优化

**跨阶段缓存（Cross-stage Caching）**：在流水线并行中，朴素实现需要在每次阶段切换时传输所有累积的 block 表示，造成 $O(N \cdot \text{chunks})$ 的通信开销。通过在每个物理阶段本地缓存已接收的 block，只在切换时传输增量 block，将通信开销降至 $O(N)$，与标准残差相当，并可与计算完全重叠。

![跨阶段缓存流水线通信示意](https://arxiv.org/html/2603.15031/x2.png)
*图3：基于缓存的流水线通信示意（4 个物理 rank，每 rank 2 个虚拟阶段，阴影框表示 AttnRes block 结尾）。数字为 micro-batch 编号，每个 rank 缓存已接收的 block，阶段切换时只传输增量 block，而非完整历史。*

**两阶段推理策略（Two-phase Inference）**：

- **Phase 1（并行跨 block 注意力）**：将一个 block 内所有层的伪查询向量批量执行矩阵乘法，一次性计算所有层对历史 block 的注意力，并返回 softmax 统计量（max 和 log-sum-exp）；
- **Phase 2（顺序 block 内注意力 + 在线 softmax 合并）**：逐层计算 block 内部的注意力（基于演化的部分和），然后通过在线 softmax 与 Phase 1 的结果合并。

这将每层的内存访问从 $O(N)$ 降至摊销后的 $O(1)$，实测推理延迟开销小于 2%。

**表1：不同残差机制每 token 每层的内存访问开销对比**

| 方案 | 读 | 写 | 典型值（$d=4096$, $N=8$） |
|---|---|---|---|
| 标准残差 | $2d$ | $d$ | 低 |
| mHC（$m$ 流） | $2md + 2d$ | $md + d$ | 较高 |
| AttnRes Full Phase1（摊销） | $2Nd/B$ | — | 低 |
| AttnRes Full Phase2 | $2d$ | $d$ | 低 |
| AttnRes Block Phase1（摊销） | $2Nd/B$ | — | 极低 |
| AttnRes Block Phase2 | $2d$ | $d$ | 低 |

---

## 实验结果

### 模型规模与训练设置

**Scaling Law 实验**：扫描 5 种模型大小（194M 到 528M 激活参数），对每种大小训练三个变体：PreNorm 基线、FullAttnRes 和 BlockAttnRes（$N=8$ blocks）。所有变体共享相同超参数（有利于基线），确保对比公平。

**表2：Scaling Law 实验各规模验证损失对比**

| 激活参数 | Token 数 | Baseline | BlockAttnRes | FullAttnRes | mHC-lite |
|---|---|---|---|---|---|
| 194M | 38.7B | — | — | — | — |
| 241M | 45.4B | — | — | — | — |
| 296M | 62.1B | — | — | — | — |
| 436M | 87.9B | — | — | — | — |
| 528M | 119.0B | 1.714 | 1.692 | 更低 | — |

在 5.6 PFLOP/s-days 计算量下，BlockAttnRes 验证损失为 1.692，Baseline 为 1.714，等效于 **约 1.09× 的计算优势**。FullAttnRes 和 BlockAttnRes 的拟合幂律曲线斜率相似，但 AttnRes 在整个计算区间内持续实现更低的损失。

![Scaling Law 曲线](https://arxiv.org/html/2603.15031/x3.png)
*图4：Attention Residuals 的 Scaling Law 曲线。FullAttnRes 和 BlockAttnRes 在所有规模下均持续优于基线，BlockAttnRes 在最大规模时已接近 FullAttnRes 的效果。*

### Kimi Linear 48B 主实验

**架构设置**：Kimi Linear 48B 配置（27 Transformer blocks，54 层），8 out of 256 路由专家 + 1 共享专家，48B 总参数 / 3B 激活参数。BlockAttnRes 使用 6 层/block，共 9 个 block + token embedding = 10 个深度方向来源。

**训练方案**：与 KimiLinear 1.4T token 训练一致，WSD 预训练阶段（1T tokens）+ 中期训练阶段（400B 高质量 tokens），全局 batch size 8M tokens，Muon optimizer，4096 token 上下文窗口，后续扩展至 32K tokens。

### 训练动态分析

![训练动态对比](https://arxiv.org/html/2603.15031/x4.png)
*图5：Baseline 和 BlockAttnRes 的训练动态对比。(a) 训练过程中的验证损失；(b) 训练结束时各 Transformer block 的输出模值；(c) 各 Transformer block 的梯度模值。*

- **验证损失**：AttnRes 全程保持更低的验证损失，差距在学习率衰减阶段扩大，最终损失明显更低；
- **输出模值**：Baseline 受 PreNorm Dilution 影响，输出模值随深度单调增长；BlockAttnRes 将这种增长限制在每个 block 内，选择性聚合在 block 边界重置了累积，呈现有界的周期性模式；
- **梯度模值**：Baseline 的残差权重全为 1，无法调控梯度流，导致最浅层梯度过大；BlockAttnRes 的可学习 softmax 权重引入了源间的竞争机制，梯度分布更加均匀。

### 下游任务性能

**表3：AttnRes 与 Baseline 在各评测基准上的对比（预训练后）**

| 类别 | 任务 | Baseline | AttnRes |
|---|---|---|---|
| 通用理解 | MMLU | 73.5 | **74.6** (+1.1) |
| 通用理解 | MMLU-Pro | — | 提升 |
| 推理 | GPQA-Diamond | 36.9 | **44.4** (+7.5) |
| 推理 | BBH | 76.3 | 提升 |
| 推理 | ARC-Challenge | 64.6 | 提升 |
| 常识 | HellaSwag | 83.2 | 提升 |
| 知识 | TriviaQA | 69.9 | **71.8** (+1.9) |
| 数学 | GSM8K | 81.7 | 提升 |
| 数学 | MGSM | 64.9 | 提升 |
| 数学 | MATH | 53.5 | **57.1** (+3.6) |
| 数学 | CMath | 84.7 | 提升 |
| 代码 | HumanEval | 59.1 | **62.2** (+3.1) |
| 代码 | MBPP | 72.0 | 提升 |
| 中文 | CMMLU | 82.0 | 提升 |
| 中文 | C-Eval | 79.6 | 提升 |

AttnRes 在所有评测任务上均匹配或超过基线。提升最显著的是多步推理任务（GPQA-Diamond +7.5，Minerva Math +3.6）和代码生成（HumanEval +3.1），与假设"改善深度方向信息流有助于组合性任务"一致。

### 消融实验

**表4：关键组件消融（16 层模型）**

| 变体 | 验证损失 |
|---|---|
| Baseline（PreNorm） | 1.766 |
| DenseFormer（固定输入无关权重） | 1.767（无提升） |
| mHC（多流动态混合） | 1.747 |
| AttnRes Full | **1.737** |
| + 输入依赖查询 | **1.731**（但需额外投影层） |
| 输入无关混合标量 | 1.749 |
| softmax → sigmoid | 1.741 |
| 去除 RMSNorm | 1.743（Full）/ 1.750（Block） |
| 滑动窗口聚合 SWA（$k=4$） | 1.764 |
| BlockAttnRes（$B=6$）| **1.746** |
| 多头深度聚合（$H=4$） | 1.752（不如单头） |

关键发现：
- **DenseFormer 无提升**，说明输入相关的权重至关重要；
- **mHC** 通过多流 + 学习混合矩阵改善到 1.747，AttnRes 以更简单的单个查询向量达到更低的 1.737；
- **sigmoid** 不如 softmax，因为 softmax 的竞争归一化机制强制更尖锐的选择；
- **RMSNorm 不可少**，尤其对 BlockAttnRes，防止 block 级表示因累积更多层而模值过大；
- **多头不如单头**，说明最优的深度混合跨通道高度一致。

### Block Size 影响分析

![Block Size 对验证损失的影响](https://arxiv.org/html/2603.15031/x5.png)
*图6：block size $B$ 对验证损失的影响（16 层模型）。随着 $B$ 增大（粒度变粗），损失优雅地退化：$B$ 较小时均接近 1.746，$B$ 较大时向 Baseline 靠拢。实践中固定 $N=8$ blocks。*

### 最优架构分析

在固定计算（FLOPs）和参数量的条件下，对 25 种 depth-width-attention 配置进行扫描：

- Baseline 最优配置：$H=4, L=28$（$H/L$ 比值适中）；
- AttnRes 最优配置：$H=4, L=32$（更深更窄），说明 AttnRes 能更有效地利用额外深度；
- AttnRes 在全部 25 种配置下均优于 Baseline，差距约 0.002-0.003 loss。

### 学习到的注意力权重模式

![学习到的深度注意力权重分布](https://arxiv.org/html/2603.15031/x6.png)
*图8：16 头模型 Full AttnRes（上）和 Block AttnRes（下）的深度注意力权重分布（对 token 平均）。行为第 $l$ 个注意力/MLP 层，列为来源 block。对角线优势说明局部性仍是主要信息路径；来源 0（embedding）持续保持较大权重；偶尔出现的非对角线集中说明模型学到了跳跃连接。*

三个关键观察：
1. **保留局部性**：每层最强烈地关注其直接前驱，但选择性的非对角集中（如第 4 层关注早期来源，第 15-16 层在 block 设置下回望更远）表明学到了超越标准残差路径的跳跃连接；
2. **层级专业化**：Embedding 在整个网络中保持非微不足道的权重，尤其在预注意力层；预 MLP 层对近期表示有更尖锐的对角依赖，而预注意力层维持更宽的感受野；
3. **BlockAttnRes 保留结构**：对角优势、embedding 持续性和层级专业化都从 Full 到 Block 变体得到了保留。

---

## 方法对比与统一视角

通过**深度混合矩阵**（depth mixing matrix）$M$，其中 $M_{l,i}$ 是第 $l$ 层赋予第 $i$ 层输出的权重，可以统一表达各种残差变体：

| 方法 | 权重类型 | 来源 |
|---|---|---|
| 标准残差 | 固定（全1下三角矩阵） | 单状态 |
| ReZero / LayerScale | 静态标量/对角 | 单状态 |
| Highway | 动态（输入相关门控） | 单状态 |
| DeepNorm | 固定缩放 | 单状态 |
| SiameseNorm | 固定2流 | 2流 |
| HC / mHC | 动态（$m$ 流学习矩阵） | $m$ 流 |
| DenseFormer | 静态标量 | 跨层全访问 |
| **AttnRes Full（本文）** | **动态（softmax 注意力）** | **跨层全访问** |
| **AttnRes Block（本文）** | **动态（block 级 softmax）** | **跨 block 访问** |

标准残差对应深度方向的线性注意力（累加归一化），AttnRes 将其推广到深度方向的 softmax 注意力，完成了与序列维度上从 RNN 到 Transformer 相同的"线性到 softmax"的过渡。

![深度混合矩阵可视化](https://arxiv.org/html/2603.15031/x7.png)
*图9：四种残差变体的深度混合矩阵（$L=8$，BlockAttnRes 使用 block size $B=2$）。AttnRes 面板展示未归一化的分数；背景颜色将共享相同来源（Full）或来源 block（Block）的条目分组。*

---

## 总结

本文提出 Attention Residuals（AttnRes），通过将固定权重残差连接替换为深度方向的 softmax 注意力，解决了现代 LLM 中 PreNorm Dilution 导致的隐藏状态增长问题和层级贡献被稀释的问题。

核心贡献包括：
1. **AttnRes 方法**：每层通过单个可学习的伪查询向量，对所有前序层输出进行 softmax 注意力加权聚合，实现内容相关的深度方向选择性访问；
2. **BlockAttnRes 变体**：将层分组为 block，跨 block 只对 block 级表示做注意力，将内存和通信从 $O(L)$ 降至 $O(N)$（$N \ll L$）；
3. **基础设施优化**：跨阶段缓存消除流水线并行中的冗余通信，两阶段推理策略将推理延迟开销降至 2% 以下；
4. **全面验证**：Scaling Law 实验证实跨规模的持续提升，Kimi Linear 48B/1.4T tokens 的实验在所有评测任务上取得提升，训练动态分析揭示了 AttnRes 对 PreNorm Dilution 的缓解机制。

AttnRes 作为标准残差连接的即插即用替代方案，提供了一种低开销但效果显著的架构改进路径，为未来大规模 LLM 的深度信息流设计提供了新的思路。

---

*论文链接：[https://arxiv.org/abs/2603.15031](https://arxiv.org/abs/2603.15031)*
*代码仓库：[https://github.com/MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)*
