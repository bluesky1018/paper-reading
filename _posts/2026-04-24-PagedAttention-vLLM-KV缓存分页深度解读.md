---
title: "PagedAttention — 把操作系统的虚拟内存思想搬进 KV Cache,vLLM 的核心引擎"
date: 2026-04-24 09:15:00 +0800
categories: [Attention, Inference Optimization, Systems]
tags: [paged-attention, vllm, kv-cache, serving, kwon-2023]
math: true
---

## 基本信息

- **作者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **机构**: UC Berkeley, Stanford, UCSD
- **发表**: SOSP 2023
- **arXiv**: [2309.06180](https://arxiv.org/abs/2309.06180)

## 一句话总结

提出 **PagedAttention** 与 **vLLM**——把**虚拟内存分页(Virtual Memory Paging)**这个操作系统里 50 年的老思想搬到 LLM serving:把 KV cache 按固定大小的 **block(页)**管理,通过**块表**把逻辑上连续的 KV 映射到物理上任意的块,消除 80% 以上的内存碎片,让同样一张 GPU 能服务 **2-4 倍**的请求并发。vLLM 迅速成为 LLM 推理引擎的事实标准。

![传统 serving 系统的 KV cache 碎片化:每个请求预留 max_seq_len 大小,实际用量远小于此,导致内部碎片 + 外部碎片共占显存 60-80%。](/assets/img/paged-attention/x1.png)
_Figure 1:传统 KV cache 管理——大量显存被碎片化吞掉_

---

## 背景:LLM 推理的内存危机

### LLM 推理的特殊性

2023 年的 LLM serving 场景:

- 请求长度**极度参差**:有的 100 tokens,有的 2000 tokens,难以预估
- KV cache **随生成动态增长**:不能一次分配完
- 并发请求数决定 QPS:**每请求占多少显存**是核心瓶颈

### 传统 serving 的做法:给每个请求预分配最大长度

- 请求 A 最大 2048 tokens → 分配 2048 × 2 × H × d × N_layer 的连续显存
- 实际只用了 300 tokens → **1748 tokens 的空间被浪费**
- 多个请求同时运行时,这些碎片加起来占显存 60-80%

这直接导致**真实并发数远低于理论值**。A100-80GB 理论上能服务 ~30 个并发请求的 LLaMA-13B,实际只能 ~10 个。

---

## 核心机制:像 OS 一样分页管理 KV Cache

![PagedAttention 的核心概念:物理 KV 块(Physical Blocks)按固定大小(如 16 tokens)分配,每个序列用一个 block table 记录自己的逻辑块 → 物理块映射。逻辑上连续的序列在物理上可以分散存储。](/assets/img/paged-attention/x2.png)
_Figure 2:PagedAttention 的物理块 + block table 架构_

### 1. 从"连续分配"到"分页分配"

OS 的虚拟内存思想:

- **物理内存**按**固定大小的页(Page)**管理
- 进程看到的是**虚拟地址空间**,通过**页表**映射到物理页
- 每页可以在物理上任意位置,不需要连续

PagedAttention 完全照搬:

- **Physical block**:GPU 显存中固定大小(默认 16 tokens)的 KV 存储单元
- **Block table**:每个序列维护一个表,记录"序列第 $i$ 个逻辑块 → 物理块 $P_j$"
- 序列增长时,**只需要增加一个物理块**(而不是扩展原分配)

### 2. Attention 计算要怎么做

![修改后的 attention kernel:根据 block table 从分散的物理块读取 K/V,做 attention 计算。看似稍慢,实际因内存局部性提升反而更快。](/assets/img/paged-attention/x3.png)
_Figure 3:PagedAttention kernel 从 block table 读取分散的 KV_

原始 attention 是:

```
scores = Q @ K[0:N].T   # K 是连续 N × d
```

现在 K 散在多个物理块里,需要一个**定制的 CUDA kernel**:

1. Query 一次读进 SRAM
2. 遍历当前序列的 block table
3. 对每个物理块读进 SRAM,做局部 attention 累加
4. 最后归一化

这个 kernel 作者实现得相当高效,实测比原 naive 实现还快 20%(因为 block-level 的内存访问模式更规整,L1/L2 cache 命中率更高)。

### 3. 内存碎片几乎归零

![PagedAttention 下的 memory utilization:接近 96% 的 KV 显存被实际使用,相比传统方式的 20-38% 是数量级提升。](/assets/img/paged-attention/x4.png)
_Figure 4:显存利用率大幅提升,碎片消失_

分页后:

- **外部碎片**:消失。每个请求的分配粒度是 16 tokens,而不是 max_seq_len
- **内部碎片**:只剩每个请求末尾未填满的那一个 block(最多 15 tokens × 2bytes × H × d × L)——相对总量可忽略

整体 KV 显存利用率从 **~25% → ~96%**。同一张卡的并发数直接 **3-4×**。

---

## 加分项:共享前缀 KV

分页带来一个意外福利:**多个请求共享相同前缀的 KV 块**。

### 场景:Beam Search / Parallel Sampling

- 用户请求对同一个 prompt 生成 $n=4$ 个候选
- 传统方式:复制 4 份 prompt 的 KV cache,浪费 $4\times$ 显存
- PagedAttention:**4 个序列共享指向同一块物理 KV 的 block table 项**,直到分支点才 copy-on-write

![Copy-on-Write 机制:多个序列共享前缀块(灰色),只在需要修改时才复制(变绿)。Beam search 场景内存开销从 O(n·L) 降到 O(L + n·diff)。](/assets/img/paged-attention/x5.png)
_Figure 5:Copy-on-Write 让共享前缀的多请求只占一份 KV_

### 场景:System Prompt

所有请求共享一段 1000 tokens 的 system prompt:

- 传统方式:100 个并发 → 100 份 system prompt 的 KV cache
- PagedAttention:**1 份 system prompt KV + 100 个各自的生成部分**

这给 RAG、few-shot、chat 场景带来决定性的内存优势。

---

## 实验结果

在 LLaMA-7B / 13B 上对比 FasterTransformer、Orca:

- **吞吐量**:PagedAttention 比 Orca 高 **1.7 - 3.5×**
- **首 token 延迟**:降低 60%(因为能立即开始服务 vs. 等 OOM 释放)
- **长 prompt 场景(2K+)**:优势放大到 **4×+**

---

## vLLM 的架构:不只是 PagedAttention

vLLM 把 PagedAttention 作为核心,再加上一整套调度:

1. **Continuous batching**:请求动态加入和退出,消除 padding 浪费(Orca 引入,vLLM 完善)
2. **Preemption / Swap**:显存不够时把低优先级请求的 KV cache swap 到 CPU RAM 或者 recompute
3. **分布式执行**:多 GPU tensor parallelism 原生支持
4. **Prefix caching**:持久化 system prompt 的 KV,跨请求复用

这一套组合拳让 vLLM 成为 2023-2024 年最流行的 LLM serving 框架,被 OpenAI、Anthropic、Together 等厂商内部使用(或参考实现)。

---

## 为什么影响如此深远

### 1. 把系统思想引入 LLM 推理

在 PagedAttention 之前,LLM serving 更多是"用深度学习框架凑",很少有人从系统角度重新设计。vLLM 证明了**OS / DB 的经典优化可以直接迁移到 LLM 场景**——这打开了一扇门,后续 SGLang、RadixAttention、Prefix Caching 都是这个思路延伸。

### 2. 成为行业标准 API

现在很多模型发布时都**直接提供 vLLM 支持**。Huggingface 的 `text-generation-inference`、NVIDIA 的 TensorRT-LLM 也都加入了 PagedAttention 或类似机制。

### 3. 让长 prompt 和 RAG 实用化

PagedAttention 的 prefix sharing 让 RAG 和长 system prompt 的成本可接受。这对 2024 年 agent 应用的兴起是关键基础设施。

### 4. 彻底改变了 LLM 部署的经济学

"一张 A100 能服务多少用户"从 3-5 人变成 15-30 人,**单 token 成本下降 5-10×**。这直接推动了 API 定价下降、开源模型自部署成为可行选项。

---

## 局限

1. **只解决 KV 显存,不解决计算**:PagedAttention 不加速 prefill,也不改变 FLOPs
2. **block size 是 trade-off**:太小(4)会增加 block table 开销;太大(64+)退化为传统方式。实用中 16 是甜蜜点
3. **定制 kernel 的维护成本**:每当新的 attention 变体(MQA、GQA、MLA)出现,PagedAttention kernel 要跟着适配
4. **对超短请求优势不明显**:如果每个请求只生成 10 tokens,碎片本来就少,PagedAttention 的收益有限

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **OS 的分页思想可以直接救 LLM serving**——这是个极好的"跨领域类比"案例,值得每个工程师内化
2. **碎片是内存利用率的隐形杀手**:不仅 CPU 有,GPU 也有。PagedAttention 把 KV 碎片从 75% 压到 <5%
3. **Prefix sharing 是 LLM serving 的内在结构**:system prompt、beam search、few-shot 都可共享前缀——分页让共享变得自然
4. **"优化 IO 而非计算"有时比 "优化计算本身" 更关键**:FlashAttention(优化 SRAM IO)和 PagedAttention(优化 HBM IO)是两面旗帜,都不改算法但彻底改变性能
</callout>

---

## 延伸阅读

- [MQA 深度解读]({% post_url 2026-04-23-MQA-Multi-Query-Attention-深度解读 %}) —— KV cache 压缩的互补方案
- [GQA 深度解读]({% post_url 2026-04-23-GQA-分组查询注意力深度解读 %}) —— 与 PagedAttention 一同使用的主流组合
- [FlashAttention 深度解读]({% post_url 2026-04-23-FlashAttention-IO感知精确注意力深度解读 %}) —— 另一个 IO 感知的经典工作
- [vLLM 官方文档](https://docs.vllm.ai/) —— 实际使用 PagedAttention
- [SGLang (Zheng et al., 2023)](https://arxiv.org/abs/2312.07104) —— 更进一步的结构化 prompt serving
