---
title: "DualPath — 突破 Agent LLM 推理的存储带宽瓶颈"
date: 2026-03-03 17:34:00 +0800
categories: [AI, 系统优化]
tags: [DeepSeek, 推理系统, KV-Cache, Agent, 存储带宽, PD分离, RDMA, 调度]
math: true
---

## 基本信息

- **论文标题**: DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference
- **作者**: Yongtong Wu, Shaoyuan Chen, Yinmin Zhong, Rilin Huang 等
- **机构**: 北京大学、清华大学、DeepSeek-AI
- **arXiv 链接**: [https://arxiv.org/abs/2602.21548](https://arxiv.org/abs/2602.21548)

## 一句话总结

DualPath 通过引入**双路径 KV-Cache 加载**机制——让空闲的 Decode 引擎也参与从存储读取 KV-Cache，再通过高速 RDMA 传输给 Prefill 引擎——突破了 Agent 场景下 Prefill 端存储带宽瓶颈，离线推理吞吐提升最高 **1.87×**，在线服务吞吐平均提升 **1.96×**。

## 背景与动机

### Agent 推理的范式转变

LLM 正从单轮对话演进为多轮 Agent 系统（编程助手、自主任务代理等），带来了根本性的工作负载变化：

- **多轮交互极长**：平均 157 轮，上下文可达百万 token
- **每轮追加极短**：平均仅 429 token
- **KV-Cache 命中率极高**：≥95%（典型值 98.7%）

这意味着 Agent 推理是**I/O 密集型**而非计算密集型——性能瓶颈从 GPU 计算转移到了 KV-Cache 的存储加载。

### 不对称的带宽利用

![图1：现有瓶颈与 DualPath 方案](/assets/img/dualpath/x1.png)
*图1：（左）现有架构中 Prefill 引擎存储 NIC 饱和而 Decode 引擎存储 NIC 空闲；（右）DualPath 通过双路径加载平衡带宽利用。*

在主流 PD 分离架构中：
- **Prefill 引擎**：需从远程存储加载大量 KV-Cache → 存储 NIC 饱和
- **Decode 引擎**：几乎不需要从存储读取 → 存储 NIC 空闲

这种不对称是整个系统的根本瓶颈。

### 硬件趋势加剧问题

![图3：硬件趋势与吞吐分析](/assets/img/dualpath/x3.png)
*图3：（左）从 Ampere 到 Blackwell，I/O 与计算比下降 14.4×——计算能力增长远超网络带宽；（右）不同请求批大小下的相对 token 吞吐。*

从 NVIDIA Ampere 到 Blackwell，GPU FLOPS 增长远超网络带宽和 HBM 容量，I/O-计算比下降 14.4×，使 Agent 推理的存储带宽瓶颈愈发严重。

## 架构/方法详解

### 核心思想：双路径 KV-Cache 加载

![图4：双路径加载示意](/assets/img/dualpath/x4.png)
*图4：(a) PE 读路径：KV-Cache 从存储直接加载到 Prefill 引擎；(b) DE 读路径：KV-Cache 先加载到 Decode 引擎，再通过 RDMA 转发到 Prefill 引擎。*

**PE 读路径**（传统路径）：
1. 从存储读取 KV-Cache 到 PE 缓冲区
2. 逐层传输到 PE HBM 进行计算
3. 计算完成后将完整 KV-Cache 传输给 DE

**DE 读路径**（创新路径）：
1. 从存储读取 KV-Cache 到 DE 缓冲区
2. PE 计算时，逐层通过 RDMA 将 KV-Cache 传输到 PE
3. 仅需传输缓存未命中 token 的 KV-Cache 给 DE

### 无瓶颈分析

关键问题：DE 读路径引入额外的计算网络流量，是否会造成新的瓶颈？

论文给出了严格的数学分析。对于 $g=8$ GPU/节点、$s=1$ 存储 NIC 的标准配置：

$$\frac{s}{g-s} \leq P/D \leq \min\left\{\frac{g-2s}{s}, \frac{g-s}{2s}, \frac{M/Bs-3}{2}\right\}$$

代入得无瓶颈范围为 $\frac{1}{7} \leq P/D \leq \frac{7}{2}$，覆盖绝大多数实际配置。

### CNIC 中心化流量管理

![图5：PE 引擎间调度示意](/assets/img/dualpath/x5.png)
*图5：引擎间 PE 调度示意——调度器根据磁盘读队列长度和未完成 token 数量选择最优 PE。*

**挑战**：KV-Cache 传输流量可能干扰延迟敏感的模型推理通信（如 EP AllToAll）。

**解决方案**：所有 GPU 进出数据必须通过配对的 Compute NIC（CNIC），利用 InfiniBand Virtual Lane 实现严格的流量隔离：
- **高优先级 VL**：模型推理通信，分配 ~99% 带宽
- **低优先级 VL**：KV-Cache 传输，利用空闲带宽

CNIC 辅助的 H2D/D2H 还有一个意外好处：提交单个 RDMA Write 仅需 ~1μs（vs. cudaMemcpyAsync 的 5-7μs），对大量小数据块传输效率更高。

### 自适应请求调度

![图6：引擎内调度](/assets/img/dualpath/x6.png)
*图6：引擎内调度——基于计算配额的批次选择和 GPU 时间线优化。*

**两级调度策略**：

**1. 引擎间调度（Inter-Engine）**：
- PE 调度：FIFO 顺序，优先分配给磁盘读队列短且未超载的引擎
- DE 调度：两阶段——先跨组平衡 token 总量，再组内平衡 HBM 使用
- 路径选择：选择读队列较短的一侧

**2. 引擎内调度（Intra-Engine）**：
- 使用 FIFO 打包，基于注意力层执行时间估计限制批大小
- 通过计算配额确保数据并行下各 GPU 工作负载均衡

## 关键实验结果

### 离线批量推理

![图7：不同模型和配置下的离线推理性能](/assets/img/dualpath/x7.png)
*图7：三个模型在不同 Agent 数量和最大上下文长度下的离线推理性能。DualPath 在所有配置下均显著优于 Basic 基线。*

| 模型 | 最大提升 | 关键观察 |
|------|---------|---------|
| DS 660B | **1.87×** | 接近 Oracle（零I/O）性能 |
| DS 27B | **1.78×** | 1P1D 配置下存储带宽仍有限 |
| Qwen 32B | **1.64×** | Dense 模型 KV-Cache 更大 |

### P/D 比例影响

![图8：不同 P/D 比例的性能](/assets/img/dualpath/x8.png)
*图8：DS 27B 在不同 P/D 比例下的性能对比。关键发现：DualPath 1P1D ≈ Basic 2P1D，因为两者可用存储带宽相同。*

DualPath 的 1P1D 性能等同于 Basic 的 2P1D——因为 DualPath 可以利用双倍的存储带宽。这直接验证了存储带宽是 Agent 场景的主导瓶颈。

### Append/Generation 长度影响

![图9：不同 append 和 generation 长度的影响](/assets/img/dualpath/x9.png)
*图9：（左）随 append 长度增加，瓶颈从 I/O 转向计算，Basic 逐渐接近 DualPath；（右）generation 长度趋势类似。*

### 在线服务

![图10：在线服务延迟指标](/assets/img/dualpath/x10.png)
*图10：不同 Agent 到达率下的 TTFT、TTST 和 TPOT。DualPath 在满足 SLO 条件下平均提升 1.96× 吞吐。*

- TTFT（首 token 延迟）：DualPath 显著降低
- TPOT（token 间延迟）：几乎无影响（流量隔离有效）
- 在线吞吐：平均提升 **1.96×**

## 深度思考

### 为什么重要？

1. **抓住了正确的瓶颈**：Agent 推理从计算密集变为 I/O 密集，传统优化方向（更快的 GPU kernel、更好的调度）不再是主要矛盾——存储带宽才是
2. **零额外硬件成本**：仅通过软件重新编排数据流，利用已有的空闲 Decode 端存储 NIC 带宽
3. **理论保证**：严格证明了在常见 P/D 比例下双路径不引入新瓶颈
4. **已在生产部署**：CNIC 中心化方案已广泛用于 DeepSeek 生产环境

### 局限性

1. **依赖 RDMA 和双网络隔离架构**：需要现代 AI 数据中心的标准配置（计算网络 + 存储网络隔离）
2. **未实现请求分割**：当前路径选择是全部或全部，论文承认将请求分割到两个路径读取可能更优
3. **DRAM 缓冲区引入额外延迟**：DE 缓冲区设计增加了一次 H2D 拷贝
4. **对短上下文场景收益有限**：当 KV-Cache 命中率低时（<90%），计算占主导，DualPath 优势减弱

### 深远影响

- **Agent 推理基础设施的范式转变**：随着 Agent 应用爆发，I/O 优化将超越计算优化成为推理系统的核心关注点
- **存储层次结构的重新设计**：HBM → DRAM → SSD 的多级 KV-Cache 存储将成为标配（结合 3FS）
- **与 Engram 的互补**：Engram 从架构层面减少需要计算的 KV-Cache，DualPath 从系统层面加速 KV-Cache 的加载——两者可以完美叠加
- **RL 训练的推理瓶颈**：Agent RL 的 rollout 阶段是典型的 Agent 推理场景，DualPath 直接加速了 RL 训练

## 总结

DualPath 优雅地解决了一个日益突出的系统问题：Agent LLM 推理中 Prefill 引擎存储带宽饱和而 Decode 引擎存储带宽空闲的不对称性。通过让 Decode 引擎也参与 KV-Cache 加载并通过 RDMA 转发，配合 CNIC 中心化流量隔离和自适应调度，DualPath 在不增加硬件的前提下将系统吞吐提升了近 2 倍。

这篇论文的价值不仅在于提出的解决方案，更在于清晰地指出了 Agent 时代推理系统的核心矛盾——**存储 I/O 而非计算成为了瓶颈**。这一洞察对未来的推理系统设计、数据中心架构规划、乃至模型架构设计（如何减少 KV-Cache 大小）都有深远影响。
