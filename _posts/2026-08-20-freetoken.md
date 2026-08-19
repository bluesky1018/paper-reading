---
layout: post
title: "FreeToken：边缘端 MoE 推理系统，带宽自适应高效执行"
date: 2026-08-20
categories: [论文解读, 系统与效率]
tags: [MoE, Edge Computing, LLM Serving, Inference Optimization]
---

> 📄 **论文**：FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution
> 🔗 **arXiv**：[2608.16157](https://arxiv.org/abs/2608.16157)
> 🏢 **机构**：University of California, Berkeley

## 一句话总结

FreeToken 是一个面向边缘设备的 MoE（混合专家）模型推理系统，将个人机器视为统一弹性推理平台，通过带宽自适应执行策略，在 8 GB 笔记本 GPU 到单台工作站 GPU 上实现了从 35B 到 753B 大型 MoE 模型的实用级推理。

## 背景与问题

以 Kimi-K3、GLM-5.2、DeepSeek-V4-Flash 为代表的前沿开源模型正迅速缩小与顶级闭源系统的能力差距。然而，开源参数仅决定了谁能**获取**模型，而非谁能**运行**它。前沿开源模型仍然依赖稀缺的数据中心级 GPU 集群，成本极为高昂。

随着 Agentic 应用大幅提升推理需求，这一成本对个人用户和小团队尤为沉重。现有边缘推理引擎（如 llama.cpp、KVCache.AI）在 Agentic 工作负载下暴露出三大挑战：

1. **预填充成本高**：每次工具调用都会增长 time-to-first-token (TTFT)
2. **解码速度低**：远低于机器的理论内存带宽上限
3. **资源异构**：边缘硬件 CPU/GPU/内存组合千差万别，无法用固定策略适配

## 核心方法

### FreeToken 系统设计

FreeToken 将整个推理栈进行协同设计，核心理念是**持续地将计算和模型状态映射到实际可用的资源上**，而非固定的 offloading 策略。

系统设计围绕两个现实展开：
- **Agent 工作负载执行模式持续变化**（工具调用、上下文增长等）
- **边缘硬件暴露异构资源**，不同机器的 CPU/GPU/PCIe 带宽比例各异

**图2（FreeToken 概览）**描述了三个核心机制：

1. **预填充优化（Prefill）**：专家加载采用完整层粒度的双缓冲方式，在 GPU 计算第 l 层时通过 PCIe 流式传输第 l+1 层的专家权重；循环状态检查点和公共前缀复用（radix cache）消除了 Agentic 状态的冗余重计算

2. **解码优化（Decode）**：通过带宽感知的 CPU-GPU 协同执行，将 q* 个专家固定在 GPU 上，其余放在 CPU 执行，最优分配点 q* ≈ m × BP/BH（其中 BP 是 PCIe 带宽，BH 是主机侧 CPU 带宽）

3. **运行时内存管理**：图兼容的 LRU 专家缓存，无需代价高昂的设备同步

### 带宽自适应的 CPU-GPU 协同

关键洞察是：解码阶段每个 token 仅激活 k 个专家（远少于总专家数 E），但必须决定哪些专家放在 GPU（快速但容量小），哪些在 CPU 执行（慢但容量大）。

最优分割点满足：
$$q^* \approx m \cdot \frac{B_P}{B_H}$$

这一公式根据实际测量的 PCIe 带宽（BP）和 CPU 侧带宽（BH）自适应调整，充分利用了两侧的计算资源。

## 实验结果

### 测试硬件配置

| 系统 | GPU (VRAM) | PCIe | BP (GB/s) | CPU | DRAM | BH (GB/s) |
|------|------------|------|-----------|-----|------|-----------|
| RTX 5090 服务器 | 32 GB | 5.0×16 | 52.7 | 2× Xeon Gold 6459C | DDR5 180 GiB | 77.3 |
| RTX 4090 服务器 | 24 GB | 4.0×16 | 25.1 | 2× Xeon Platinum 8358P | DDR4 240 GiB | 63.2 |
| RTX 3090 服务器 | 24 GB | 4.0×16 | 25.3 | 2× Xeon Gold 6330 | DDR4 180 GiB | 56.7 |
| RTX 5090 桌面机 | 32 GB | 5.0×16 | 49.0 | Ryzen 9 9950X3D | DDR5 192 GiB | 53.8 |
| RTX 4060 笔记本 | 8 GB | 4.0×8 | - | - | - | 47.5 |

### 主要性能表现

- **笔记本（8 GB GPU）**：可实用地运行 Qwen3.6-35B-A3B 模型
- **游戏台式机**：可运行 284B 参数模型
- **单台工作站 GPU**：可运行 753B 参数的 GLM-5.2 模型
- 跨四种 Agentic 工作负载（AIME、OpenCode+SWE、Claude Code+SWE、邮件/日历）均实现交互级速度

在 RTX 5090 上，FreeToken 的预填充优化随 prompt 长度增加效果更显著；解码阶段专家缓存命中率随缓存大小（占总参数的百分比）快速提升。

## 总结

FreeToken 重新定义了边缘端 MoE 推理的可行边界，将原本需要数据中心的前沿大模型推向了普通消费者硬件。其核心创新在于：不再将边缘机器看作"小 GPU"，而是将其视为 CPU+GPU+内存的**统一弹性推理平台**，并根据实际可用带宽动态优化资源分配。

FreeToken 支持超过 20 个 MoE 模型，已开源。未来的挑战包括进一步优化量化精度与速度的权衡，以及在更多样化的 Agentic 工作负载下的鲁棒性优化。
