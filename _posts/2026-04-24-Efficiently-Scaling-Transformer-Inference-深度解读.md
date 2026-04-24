---
title: "Efficiently Scaling Transformer Inference — 为 PaLM-540B 推理设计的并行分析"
date: 2026-04-24 11:00:00 +0800
categories: [Attention, Inference Optimization, Distributed Systems]
tags: [pod-decoding, tensor-parallelism, pallas, google-2022]
math: true
---

## 基本信息

- **作者**: Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean
- **机构**: Google, DeepMind
- **发表**: MLSys 2023
- **arXiv**: [2211.05102](https://arxiv.org/abs/2211.05102)

## 一句话总结

Google 在 PaLM-540B 推理上的系统工程总结——用**roofline 模型 + 多种张量并行策略**的组合分析,给出"**不同模型大小、不同 batch、不同 context 长度应该用什么并行方案**"的系统答案。核心发现:**decode 阶段不是计算瓶颈,是内存带宽瓶颈**——这直接影响了后续 vLLM 的 PagedAttention 设计、Character.AI 的系统决策、以及现代 LLM serving 栈的全部思路。

![论文提出的"roofline 分析":绘制 FLOPs 利用率 vs. HBM 带宽利用率的 2D 图,每种并行策略和 batch/seq 组合占一个点,清晰看到哪种策略被哪种资源瓶颈住。](/assets/img/scaling-inference/x1.png)
_Figure 1:Roofline 分析——一张图说清楚什么瓶颈在哪里_

---

## 背景:PaLM-540B 的推理挑战

2022 年末,PaLM-540B 是当时最大的密集 LLM。对 Google 来说,让 540B 能在 TPU pod 上**可靠、高效、低延迟**地服务是关键工程问题。

挑战:

- **模型太大**:540B × bf16 = 1.08 TB,必须跨多个加速器切分
- **推理 workload 特殊**:prefill 计算密集,decode 带宽密集,两种模式对硬件资源需求完全不同
- **长 context KV cache**:8K context、batch=64 时 KV cache 就超过 100 GB
- **延迟敏感**:服务场景下用户等不起整分钟

作者的贡献:**给出一套系统的 roofline 分析,告诉你在每种 workload 下应该选择哪种并行**。

---

## 核心分析:Roofline 与不同并行策略

### 1. Prefill vs Decode 的不同特性

![Prefill 阶段每 token 做 $\sim 2N_{param}$ 次 FLOPs(full attention + FFN),而 decode 阶段每 token 仍做 $\sim 2N_{param}$ 次但只处理 1 token——所以 batch 低时严重浪费算力,变成内存带宽瓶颈。](/assets/img/scaling-inference/x2.png)
_Figure 2:Prefill 和 decode 的计算-带宽特征完全不同_

- **Prefill**(处理 prompt):每个 GPU step 处理 $B \times S$ tokens,计算密集
- **Decode**(生成每个 token):每个 step 处理 $B \times 1$ tokens,**和模型参数量成正比但 tokens 数极少**

这导致:

- Prefill 受 **FLOPs 瓶颈**(算力)
- Decode 受 **HBM 带宽瓶颈**(搬参数)

**同一个模型 + 同一个硬件,两个阶段的最优并行策略可能完全不同**。

### 2. 常见的并行策略矩阵

论文分析了以下几种:

| 并行方式 | 切什么 | 通讯开销 |
|---------|-------|----------|
| **Data Parallel (DP)** | 不切,复制 | 最小(只有梯度同步) |
| **Tensor Parallel (TP, 2D)** | 切 FFN 和 QKV | All-reduce,频繁 |
| **Sequence Parallel** | 切 seq_len | 主要在 attention |
| **Pipeline Parallel** | 切 layer | 在 boundary 上通信 |
| **Expert Parallel** (MoE) | 切 expert | All-to-all |

### 3. 不同场景下的最优策略

![作者用 roofline 分析得到的结论:Low-latency 场景用全 TP;High-throughput 场景用 TP + DP 组合;长 context 用 2D TP 或 pipeline。](/assets/img/scaling-inference/x3.png)
_Figure 3:不同 workload 下的最优并行组合_

核心决策树:

- **低延迟 decode**(batch=1,生成快):**Tensor Parallel on FFN**,所有 TPU 并行算每个 token
- **高吞吐 decode**(batch=64+):**Tensor Parallel + Data Parallel**,每个 replica 独立服务多个请求
- **长 context prefill**(16K tokens):**Sequence Parallel**,切 seq 维到多卡

### 4. 2D 和 1D Tensor Parallel

![2D Tensor Parallel:把 FFN 切成 $N_x \times N_y$ 的网格,让通讯的 volume 按 $\sqrt{N}$ 降低而不是 $N$。对超大模型 + 超大 pod 的关键优化。](/assets/img/scaling-inference/x4.png)
_Figure 4:2D TP 的通信优势_

对 540B 级别的模型,传统 1D TP 的 all-reduce 通信量太大。作者提出用 **2D TP**——把 $W^{FFN}$ 切成二维网格。

效果:
- 1D TP 在 32 chips:通信瓶颈
- 2D TP (4×8) 在 32 chips:通信量降到 1D 的 ~1/3

这成为 PaLM-540B 推理的关键优化之一,也被后续 Jax/Pathways、DeepSpeed-Inference 等采用。

---

## 关键数字:540B 在 64 TPU 上的效果

![PaLM-540B 在 64 TPU v4 上的推理性能表:不同 batch + context 下的吞吐 / 延迟 / FLOPs 利用率。decode 阶段普遍只有 15-30% 的硬件利用率。](/assets/img/scaling-inference/x5.png)
_Figure 5:PaLM-540B 实测性能_

- **Batch=1 latency**:29 ms/token(64 TPU v4)
- **Batch=64 throughput**:~1300 tokens/s 总吞吐
- **FLOPs 利用率**:Prefill 60-70%,**Decode 仅 15-25%**

这个"decode 利用率低"的数字直接催生了后续一连串的 decode 优化工作:

- **MQA / GQA**:压缩 KV 减少 HBM 流量
- **FlashDecoding / FlashAttention-2**:IO 感知的 decode kernel
- **Speculative decoding**:用小模型预测多 token 一次验证
- **PagedAttention**:消除 KV cache 碎片

---

## 为什么是一篇"基础设施"论文

这篇论文的特殊之处在于:**它不提新架构,不提新算法,纯粹是把"已知并行策略"在"超大 Transformer 推理"场景下系统性地分析一遍**。但正是这种"基础设施级"的系统研究,影响深远。

### 1. 奠定了 LLM serving 的理论框架

2022 年之前,人们对"**decode 是带宽瓶颈**"的认识是零散的。这篇论文用 roofline 模型把它理论化,**让后续每个 LLM serving 优化都必须先论证"我解决了哪种资源瓶颈"**。

### 2. Tensor Parallel 实践的教科书

论文里的 1D / 2D TP 代码(基于 JAX 的 pallas)成为后续 TP 实现的参考标准。PyTorch 的 TorchTitan、Megatron-LM 的 TP 扩展都参考了这些设计。

### 3. 推动"分阶段优化"

论文强调 **prefill 和 decode 是两个完全不同的 workload**,应该分别优化。这个思想后来演变为:

- **Continuous batching**:把不同请求的 prefill 和 decode 交错
- **Prefill / decode 分离部署**:大模型 serving 的趋势之一

### 4. Google 内部实践的公开

PaLM 本身没有开源,但这篇论文相当于公开了 Google 的 LLM serving 实践经验——为业界的 OSS serving 框架(vLLM / TGI / TensorRT-LLM)提供了蓝本。

---

## 局限

### 1. TPU 特化

论文的分析基于 TPU v4 pod,很多结论(如 2D TP 的最优切分)直接搬到 GPU 上未必最优。

### 2. 没覆盖 MoE 特殊情况

MoE 模型的 all-to-all 通信在论文中只是略提,后续的 DeepSpeed-Megatron 和 Switch Transformer serving 有更深入的分析。

### 3. 没考虑 heterogeneous cluster

论文假设所有卡同质。现代生产部署中,不同 replica 用不同 GPU 型号的混合场景,论文的建议需要额外调整。

### 4. 不涉及模型量化

论文没讨论 8-bit / 4-bit 量化对并行策略的影响——这在后来的 AWQ、GPTQ 等工作中成为关键维度。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Prefill 和 Decode 是两个完全不同的 workload**——对应 FLOPs 瓶颈 vs HBM 带宽瓶颈,必须分别优化。这是现代 LLM serving 的第一性原理
2. **Roofline 分析是系统优化的共通语言**:任何"为什么这个优化有效"的问题都应该回到 roofline 回答——这个工具值得内化
3. **并行策略的选择是 workload-dependent 的**:没有"最好"的 TP/DP 组合,只有"给定 batch/context/latency 要求下的最优"。做 inference 系统时要做多策略 A/B
4. **2D Tensor Parallel 是超大模型的关键**:通信量 $O(\sqrt{N})$ 而非 $O(N)$——这个 insight 在 540B+ 规模上是决定性的
</callout>

---

## 延伸阅读

- [PagedAttention 深度解读]({% post_url 2026-04-24-PagedAttention-vLLM-KV缓存分页深度解读 %}) —— decode 带宽优化的延续
- [MQA 深度解读]({% post_url 2026-04-23-MQA-Multi-Query-Attention-深度解读 %}) —— 减少 decode HBM 流量的关键
- [Character.AI Optimizing Inference 博客引介]({% post_url 2026-04-23-Character-AI-Optimizing-Inference-博客引介 %}) —— 产品级的类似思路
- [Megatron-LM (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053) —— Tensor Parallel 的原始论文
- [FlashAttention 深度解读]({% post_url 2026-04-23-FlashAttention-IO感知精确注意力深度解读 %}) —— IO 瓶颈优化的另一条线
