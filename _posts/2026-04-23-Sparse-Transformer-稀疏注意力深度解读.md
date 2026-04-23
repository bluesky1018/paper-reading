---
title: "Sparse Transformer — 用结构化稀疏把 Attention 从 O(N²) 压到 O(N√N)"
date: 2026-04-23 23:05:00 +0800
categories: [Attention, Sparse Attention]
tags: [sparse-transformer, strided-attention, fixed-attention, long-context, openai, child-2019]
math: true
---

## 基本信息

- **作者**: Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever
- **机构**: OpenAI
- **发表**: arXiv 2019-04
- **arXiv**: [1904.10509](https://arxiv.org/abs/1904.10509)

## 一句话总结

提出 **Sparse Transformer**——不再让每个 token 和所有其他 token 做 attention,而是用**结构化稀疏 mask(strided 或 fixed)** 限制每个 token 只看 $O(\sqrt{N})$ 个其他 token,总复杂度从 $O(N^2)$ 降到 $O(N\sqrt{N})$。配合一组 CUDA kernel 优化,**首次把 Transformer 成功应用到图像、音频、超长文本**等长序列模态上,并在 ImageNet 64 生成、Enwik8 语言建模上刷新 SOTA。是"长上下文 + 稀疏 attention" 主线的开山之作。

![OpenAI 用 Sparse Transformer 生成的 ImageNet 64×64 无条件样本(左)和古典音乐波形(右)。同一套架构处理图像、音频、文本三种模态,展示了稀疏 attention 在长序列上的威力。](/assets/img/sparse-transformer/x1.png)
_Figure 1:Sparse Transformer 在图像/音频上的无条件生成样本_

---

## 背景:Full Attention 的 $O(N^2)$ 为什么不可持续

原始 Transformer 的注意力是每个 token 与全部 $N$ 个 token 计算分数,复杂度 $O(N^2)$:

- 文本 seq_len = 1K:一次 $N^2$ 矩阵尚可,显存/算力都还行
- 图像 64×64 = 4096 个像素:$N^2$ 已经到 1600 万
- 音频 10 秒 × 16KHz = 160K:$N^2$ = 256 亿,彻底不可行

OpenAI 想做**一个通用架构处理图像、音频、文本**——文本勉强够 1K,图像音频直接爆。需要把 attention 变成**亚二次**才能上场。

### 为什么不直接用 $O(N)$ 近似?

2019 年的 Linformer、Performer 还没出现。当时的近似方法(如 Reformer LSH、低秩)要么效果差、要么工程复杂。**结构化稀疏**是最直接的思路:**硬性规定每个 token 只能看特定位置**。

但稀疏 attention 的挑战是:哪些位置该保留,既降复杂度又不丢长距离依赖?

---

## 洞察:真实模型里的 attention 本来就很稀疏

![128 层网络在 CIFAR-10 上学到的 attention 模式:不同层出现不同的稀疏结构——有些层是局部窗口、有些是跨行/跨列的 stride 模式、有些是长程点状。这启发了结构化稀疏 attention 的设计:"既然模型自己学出了稀疏模式,不如直接预设这种稀疏结构。"](/assets/img/sparse-transformer/learned.png)
_Figure 2:Full attention 训练后自发学到的稀疏模式——启发了结构化稀疏设计_

作者先训练了一个 128 层 full attention 模型,可视化发现 attention 权重**天然就呈现稀疏模式**:

- **Local 头**:只关注相邻几个位置
- **Strided 头**:等间隔关注(跨行/跨列)
- **Global 头**:偶尔看到远距离 token

既然模型自己会收敛到稀疏,**不如设计架构直接强制这种稀疏结构**。

---

## 核心机制:两种稀疏模式

Sparse Transformer 提出两个稀疏 attention 方案,都由**两个 head 组合**而成:

### Strided Attention(适合图像)

![三种 attention 对比:(a) Full attention(每个 token 看全部),(b) Strided Sparse(一个 head 看局部 l 个邻居,一个 head 跨 stride 看等间隔位置),(c) Fixed Sparse(一个 head 看局部,一个 head 看固定"关键"列)。](/assets/img/sparse-transformer/x2.png)
![Strided Sparse Transformer 的注意力示意](/assets/img/sparse-transformer/x3.png)
_Figure 3(a)(b):Full vs Strided Sparse 的注意力模式对比_

- **Head 1(row)**:只看前 $l = \sqrt{N}$ 个位置(局部窗口)
- **Head 2(column)**:以 stride = $\sqrt{N}$ 看等间隔位置(跨行)

对 64×64 的图像($N = 4096$,$\sqrt{N} = 64$),Head 1 看同一行的前 64 像素(局部),Head 2 看同一列的各行(跨行)——完美匹配图像的 2D 结构。

### Fixed Attention(适合文本)

![Fixed Sparse Transformer:row head 看局部,column head 只看每个 block 的最后几个 token(作为"信息汇聚点")。适合文本这种没有天然 2D 结构的数据。](/assets/img/sparse-transformer/x4.png)
_Figure 3(c):Fixed Sparse 的注意力模式——用"锚点"位置聚合长程信息_

- **Head 1**:看前 $l$ 个位置
- **Head 2**:只看每个 block 中的最后几个"关键位置"(它们汇聚了 block 的信息)

每个 token 最多关注 $O(\sqrt{N})$ 个位置。**两个 head 组合后,任意 token 对之间的路径长度最多 2 步**——不丢长距离依赖。

### 复杂度

$$
O(N \cdot \sqrt{N}) = O(N^{1.5})
$$

比 $O(N^2)$ 快一个数量级。以 $N = 16384$ 为例,$N^2 = 2.7 \times 10^8$,$N^{1.5} = 2 \times 10^6$——**快 100 倍以上**。

---

## 工程优化:让稀疏真正变快

算法上的稀疏**不等于** GPU 上快。稀疏矩阵在 CUDA 上存储和访问都低效。作者的工程贡献:

### Block-Sparse GPU Kernel

- 用**固定块大小**的稀疏(比如 32×32 块)取代任意稀疏
- 非 0 块用 dense matmul 计算,利用 tensor core
- 0 块跳过
- 这个思想后来演化成 FlashAttention 的 tiling + mask

### 训练稳定性

- Pre-activation 残差(而不是 Post-LN)
- 混合精度训练 + loss scaling
- Gradient checkpointing

![Sparse Transformer 残差块的结构图:灰色背景的张量是 checkpointed(为了省显存,反向时重算),其他张量(包括 attention 权重和 FFN 激活)不保存。这是"反向重算省显存"思想的早期实现。](/assets/img/sparse-transformer/x5.png)
_Figure 4:Sparse Transformer 残差块——早期版本的"checkpoint + recompute"_

**反向重算**这个思想在 3 年后被 FlashAttention 推到极致,Sparse Transformer 是它的早期探路。

---

## 实验结果

### Enwik8(压缩/语言建模)

| 模型 | BPC ↓ |
|------|-------|
| Transformer-XL | 0.99 |
| **Sparse Transformer** | **0.99** |

持平 SOTA,但 Sparse Transformer 用**更少 FLOPs** 达到。

### ImageNet 64×64 生成

![ImageNet 64×64 无条件生成样本,softmax 温度 1.0(未调低)。像素级建模的 Transformer 首次做到跨物体、跨背景的长距离一致性。](/assets/img/sparse-transformer/x6.png)
_Figure 5:ImageNet 64×64 无条件生成——像素级 Transformer 首次稳定生成_

| 模型 | bits/dim ↓ |
|------|-----------|
| PixelCNN++ | 3.92 |
| Image Transformer(full attention,需 multi-scale)| 3.77 |
| **Sparse Transformer(纯像素级)** | **3.44** |

第一次**单尺度纯像素 Transformer** 在图像生成上超越专用架构。

### 古典音乐(波形级)

- 输入:12K samples(~1 秒 @ 12KHz)
- Sparse Transformer 能生成**结构化、有乐感**的音乐片段
- 这比同期 RNN/CNN 方案在波形级建模上大幅领先

---

## 为什么影响巨大

### 1. "长上下文 attention" 研究的起点

Sparse Transformer 是第一个认真证明"**减少 attention 计算能换回长序列能力**"的工作。后续的:

- **Longformer** (2020) —— 滑窗 + 全局 attention
- **BigBird** (2020) —— 随机 + 局部 + 全局的稀疏组合
- **Routing Transformer** (2020) —— 学习稀疏
- **Sliding Window Attention** (Mistral 2023) —— 滑窗回归实用主义

全部沿这条线展开。

### 2. Block-Sparse Kernel 的范式

把稀疏固定到 32×32 或类似块大小,**用 dense matmul 计算非零块**——这个范式被:

- **FlashAttention-2** 的块稀疏扩展继承
- **Triton** 社区大量块稀疏算子
- **vLLM PagedAttention** 的块化 KV cache 管理

共享同一工程哲学。

### 3. 多模态统一架构的第一步

同一个架构**跨图像/文本/音频**,这个思路后来成为:

- **ViT** (2020) —— 图像 patch → Transformer
- **AudioLM / Whisper** —— 音频 → Transformer
- **Unified Multimodal Models** —— 文本 + 图像 + 音频统一 token 流

Sparse Transformer 的"长序列 Transformer 能做任意模态"论点是通用架构时代的先驱之一。

### 4. 重算省显存的早期实践

Figure 4 里显示的"checkpointed 张量 + 非 checkpointed 重算"思路,3 年后被 FlashAttention 推向极致(前向只保存 softmax 统计量、反向整体重算 S 和 P)。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Attention 的稀疏不是"压缩",而是"结构化归纳偏置"**:strided/fixed 模式是对数据结构(图像 2D、文本局部+锚点)的硬编码,不纯粹是为了省算力
2. **算法稀疏 + 工程块化 = 可用的稀疏 attention**:Python 里算得出不等于 GPU 里跑得快,block-sparse kernel 是落地关键
3. **每个 token 到任意 token 的最短路径决定长程能力**:Sparse Transformer 用"两个 head 组合"保证 ≤ 2 步,不丢长距离;后续 BigBird 的"random + local + global"也是同一思想
4. **长序列 Transformer 是跨模态统一架构的前提**:没有长序列能力,Transformer 就不能处理原始像素/波形,也就没有后来的 ViT、Whisper、AudioLM
</callout>

---

## 延伸阅读

- [Attention Is All You Need 深度解读]({% post_url 2026-04-23-Attention-Is-All-You-Need-深度解读 %}) —— Full attention 原版
- [Transformer-XL 深度解读]({% post_url 2026-04-23-Transformer-XL-段级递归注意力深度解读 %}) —— 同期的另一条长上下文路线
- [Longformer: The Long-Document Transformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150) —— 稀疏 attention 的直接继承者
- [BigBird: Transformers for Longer Sequences (Zaheer et al., 2020)](https://arxiv.org/abs/2007.14062) —— 理论 + 稀疏组合的集大成
- [FlashAttention 深度解读]({% post_url 2026-04-23-FlashAttention-IO感知精确注意力深度解读 %}) —— "块化 + 重算"思想的巅峰
