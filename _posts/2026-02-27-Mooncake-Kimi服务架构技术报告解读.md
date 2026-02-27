---
title: "Mooncake: Kimi的KVCache-centric分离式LLM服务架构"
date: 2026-02-27 20:35:00 +0800
categories: [LLM Serving, Inference Optimization, Distributed Systems]
tags: [kimi, moonshot-ai, mooncake, kvcache, disaggregated-architecture, llm-serving, prefill-decoding]
math: true
---

## 基本信息

- **作者**: Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu
- **机构**: Moonshot AI (月之暗面), Tsinghua University
- **发表**: arXiv 2024
- **arXiv**: [2407.00079](https://arxiv.org/abs/2407.00079)
- **代码/Trace**: [github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- **关联产品**: [Kimi](https://kimi.moonshot.cn) - 月之暗面的LLM服务

## 一句话总结

提出了**KVCache-centric的分离式架构**，通过分离Prefill和Decoding集群、利用CPU/DRAM/SSD构建分布式KVCache缓存，实现了在长上下文场景下**吞吐量提升525%**、真实工作负载下**请求处理能力提升75%**的显著效果。

## 背景与动机

### LLM服务面临的挑战

随着LLM的广泛应用，推理服务面临多样化工作负载：

| 挑战 | 描述 |
|------|------|
| **输入输出长度差异大** | 从短prompt到百万级长上下文 |
| **到达模式不规则** | 流量波动大，存在明显高峰期 |
| **SLO多样化** | 不同场景对延迟要求不同 |
| **资源供给受限** | GPU/加速器供应紧张，难以弹性扩容 |

### Prefill vs Decoding 的本质差异

LLM推理包含两个计算特性截然不同的阶段：

| 特性 | Prefill阶段 | Decoding阶段 |
|------|-------------|--------------|
| **计算模式** | 并行处理所有输入token | 自回归，逐个生成token |
| **计算复杂度** | Attention: O(n²)，MLP: O(n) | 内存受限，计算轻量 |
| **瓶颈** | 计算密集型 | 内存带宽受限 |
| **关键指标** | TTFT (Time To First Token) | TBT (Time Between Tokens) |
| **Batch策略** | 一次性处理 | Continuous Batching |

**传统方案的问题**: Prefill和Decoding耦合在同一节点，导致资源利用率低、互相干扰。

### 核心洞察：KVCache是调度核心

提升吞吐量的两个途径：
1. **KVCache复用**: 减少重复计算
2. **增大Batch Size**: 提升MFU（Model FLOPs Utilization）

但两者都与延迟SLO存在冲突：
- 远程KVCache读取延长TTFT
- 大Batch Size增加TBT

## 核心贡献

### 1. 分离式架构设计

Mooncake采用三层分离架构：

```
┌─────────────────────────────────────────────────────────────┐
│                      Global Scheduler                        │
│                      (Conductor)                             │
└─────────────┬───────────────────────────────────────────────┘
              │
    ┌─────────┴─────────┐
    ↓                   ↓
┌──────────┐     ┌──────────────┐
│ Prefill  │     │   Decoding   │
│  Cluster │     │    Cluster   │
│  (GPU)   │     │    (GPU)     │
└────┬─────┘     └──────┬───────┘
     │                  │
     └────────┬─────────┘
              ↓
┌──────────────────────────────────────────────┐
│     Distributed KVCache Pool                 │
│  (CPU + DRAM + SSD, RDMA高速传输)              │
└──────────────────────────────────────────────┘
```

**关键设计**:
- **Prefill节点池**: 处理计算密集型的预填充
- **Decoding节点池**: 处理内存受限的自回归生成
- **分布式KVCache池**: 利用GPU集群中未充分利用的CPU/DRAM/SSD资源

### 2. KVCache-centric调度器 (Conductor)

**核心职责**:
1. 为每个请求选择Prefill和Decoding实例对
2. 最大化KVCache复用
3. 平衡负载，满足TTFT/TBT SLO
4. 管理KVCache的复制、迁移、换出

**调度流程**:

```
请求到达
    ↓
┌─────────────────────────────────────┐
│ 1. KVCache Reuse                    │
│    - 计算prefix cache block IDs     │
│    - 从远程CPU内存加载prefix KVCache│
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 2. Incremental Prefill              │
│    - 基于prefix cache增量计算        │
│    - Chunked Pipeline并行处理        │
│    - 新生成KVCache存入CPU内存        │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 3. KVCache Transfer                 │
│    - 层级别流式传输                  │
│    - RDMA高速传输                    │
│    - 与计算重叠隐藏延迟              │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 4. Decoding                         │
│    - 加载完整KVCache                │
│    - Continuous Batching生成         │
└─────────────────────────────────────┘
```

### 3. Chunked Pipeline Parallelism (CPP)

**长上下文处理难题**:
- 长文本Prefill计算量大，需多节点并行
- 传统Tensor Parallelism (TP)跨节点开销大
- Sequence Parallelism (SP)需要频繁跨节点通信

**Mooncake解决方案**: Chunked Pipeline Parallelism

```
长文本请求 (如100K tokens)
    ↓
分割为chunks (每chunk ≤ prefill_chunk, 通常>1000)
    ↓
┌─────┐   ┌─────┐   ┌─────┐
│Node1│ → │Node2│ → │Node3│  (Pipeline)
│Chunk1│   │Chunk2│   │Chunk3│
└─────┘   └─────┘   └─────┘
    ↓         ↓         ↓
  并行处理不同chunk
    ↓
  显著降低TTFT
```

**优势**:
- 仅pipeline边界需要跨节点通信
- 通信可与计算重叠
- 自然适配长短上下文，无需频繁弹性扩缩容
- 不抢占KVCache传输网络资源

### 4. Layer-wise Prefill

**VRAM优化策略**:

传统方式: 等整个Prefill完成后再传输KVCache → 占用VRAM时间长

Mooncake: **层级别异步传输**
```
Layer N Attention计算
    ↓
启动Layer N+1 KVCache异步加载
    ↓
Layer N 异步存储KVCache到CPU
    ↓
等待Layer N存储完成
```

**效果**: KVCache占用成本从 S×T 降低到 min(加载时间, 计算时间)

### 5. Cache-aware调度算法

**目标**: 平衡cache命中率和实例负载

```python
# 简化逻辑
for instance in prefill_pool:
    prefix_len = match_cache(request, instance)
    queue_time = estimate_queue_time(instance)
    transfer_time = estimate_transfer_time(instance, best_instance)
    
    # 方案1: 使用本地cache
    ttft_local = queue_time + prefill_time(input_len, prefix_len)
    
    # 方案2: 从最佳cache节点传输
    ttft_transfer = transfer_time + queue_time + 
                    prefill_time(input_len, best_prefix_len)
    
    # 选择TTFT最小的方案
```

**Hot-spot迁移**: 自动复制热点KVCache块到多个节点，避免传输拥塞

### 6. Overload-oriented调度

**现实挑战**: Kimi用户量快速增长，GPU资源有限，经常面临过载

**Early Rejection策略**:
- 在Prefill前预测Decoding负载
- 如果预测无法满足SLO，尽早拒绝请求
- 避免Prefill计算资源浪费

**负载波动问题**: 
- 简单Early Rejection导致Prefill和Decoding负载反相波动
- 原因: Decoding负载预测与实际执行存在时间差

**解决方案**: Prediction-based Early Rejection
- 预测未来短期负载
- 预测生成token长度
- 平滑负载波动

## 实验结果

### 1. 模拟场景吞吐量提升

| 场景 | 相比Baseline提升 |
|------|-----------------|
| 长上下文模拟场景 | **525%** |
| 混合工作负载 | 显著提升 |

### 2. 真实工作负载

- 请求处理能力提升: **75%**
- TTFT显著降低（相比随机调度和负载均衡调度）

### 3. Cache策略对比

基于开源的23,608条真实请求trace：

| Cache容量 | LRU命中率 | LFU命中率 |
|----------|----------|----------|
| 1,000 blocks | 30% | ~28% |
| 50,000 blocks | 50% | ~48% |

**热点现象**: 50%以上的cache块从未被访问，某些块被访问数万次

### 4. Layer-wise Prefill效果

KVCache存储延迟显著降低，尤其在长上下文场景下效果明显。

## 技术架构图解

### 图1: Mooncake架构总览

**整体架构三要素**:

1. **Prefill Cluster**: 
   - 处理输入token并行计算
   - 支持Chunked Pipeline Parallelism
   - 与Decoding集群分离部署

2. **Decoding Cluster**:
   - 处理自回归token生成
   - Continuous Batching
   - 专注TBT SLO

3. **Distributed KVCache Pool**:
   - 利用CPU/DRAM/SSD资源
   - RDMA高速传输
   - Global调度管理

**数据流**: 
Request → Tokenize → Conductor调度 → Prefill(复用KVCache) → 流式传输KVCache → Decoding → Response

### 图3: KVCache Pool结构

- **Paged存储**: KVCache以block为单位存储
- **Prefix Hash**: 基于token block计算hash，支持prefix复用
- **层级存储**: GPU VRAM → CPU DRAM → SSD
- **Eviction策略**: LRU/LFU/LengthAware

### 图4: 推理实例工作流

**Prefill实例**:
- Layer-wise的KVCache load/store
- 与计算并行执行
- 流式传输到Decoding节点

**Decoding实例**:
- 异步加载KVCache
- 与GPU decoding并行
- 避免GPU空闲

## 个人思考

### 为什么Mooncake能work

1. **架构解耦**: Prefill和Decoding分离，各自优化，互不干扰
2. **资源复用**: 利用闲置CPU/DRAM/SSD做KVCache缓存，零额外成本
3. **调度智能**: Cache-aware + Load-balancing双重优化
4. **工程极致**: Layer-wise、Chunked Pipeline等细节优化

### 关键设计哲学

| 设计 | 体现 |
|------|------|
| **分离** | Prefill/Decoding/KVCache三层分离 |
| **复用** | KVCache Prefix Cache最大化 |
| **异步** | 传输与计算重叠 |
| **预测** | 负载预测避免资源浪费 |

### 与DeepSeek-OCR的对比

| Mooncake | DeepSeek-OCR |
|----------|-------------|
| **领域** | LLM推理服务 | 视觉-语言模型 |
| **核心优化** | KVCache调度与复用 | 光学压缩 |
| **架构** | 分离式服务架构 | 编码器-解码器 |
| **目标** | 吞吐量+延迟SLO | 压缩比+准确率 |
| **共同点** | 都利用**分离架构**提升效率 | |

### 局限与未来方向

| 当前局限 | 可能改进 |
|---------|---------|
| Hot-spot阈值手动调整 | 自适应算法 |
| 生成长度预测 | 更准确的模型 |
| Batch API | 异步批量处理 |
| 多模态支持 | 扩展到VLM服务 |

### 开源贡献

Mooncake开源了真实工作负载trace：
- 23,608条请求
- 包含timestamp、input/output length、hash_ids
- 首个支持KVCache reuse分析的开源数据集

这对学术界和工业界研究LLM serving都有重要价值。

## 相关阅读

### Moonshot AI相关
- **[Kimi Chat](https://kimi.moonshot.cn)** - 月之暗面的对话产品
- **[Kimi K2.5](https://www.moonshot.cn)** - 最新大语言模型

### LLM Serving相关
- **[vLLM](https://github.com/vllm-project/vllm)** - PagedAttention开源实现
- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** - NVIDIA推理优化
- **[Text Generation Inference](https://github.com/huggingface/text-generation-inference)** - HuggingFace推理框架

### 分离式架构相关
- **[Splitwise](https://arxiv.org/abs/2311.18677)** - 分离Prefill和Decoding
- **[DistServe](https://arxiv.org/abs/2401.09670)** - 分离服务优化

### 长上下文相关
- **[Ring Attention](https://arxiv.org/abs/2310.01889)** - 长上下文并行
- **[LLaMA-3.1 405B](https://ai.meta.com/blog/meta-llama-3-1/)** - 128K上下文

## 引用

```bibtex
@article{qin2024mooncake,
  title={Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving},
  author={Qin, Ruoyu and Li, Zheming and He, Weiran and Zhang, Mingxing and Wu, Yongwei and Zheng, Weimin and Xu, Xinran},
  journal={arXiv preprint arXiv:2407.00079},
  year={2024}
}
```

---

*本文解读基于Mooncake论文（arXiv:2407.00079），如有侵权请联系删除。*
