---
layout: post
title: "Embodied.cpp：面向异构机器人的具身AI模型可移植推理运行时"
date: 2026-07-07
categories: [论文解读, 具身智能]
tags: [具身AI, VLA, 机器人, 推理运行时, C++]
---

> 📄 **论文**：Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots
> 🔗 **arXiv**：[2607.02501](https://arxiv.org/abs/2607.02501)
> 🏢 **机构**：东南大学、南京大学、微软亚洲研究院、清华大学AIR等

## 一句话总结

Embodied.cpp 是一个专为具身AI设计的C++推理运行时框架，通过五层模块化架构统一支持VLA和WAM两大类模型，在真实机器人上实现了100%任务成功率，同时将内存占用减少71.8%。

## 背景与问题

具身AI模型（如VLA和WAM）的部署面临三大独特挑战：其一，现有推理框架（llama.cpp、vLLM等）是为请求-响应服务设计的，无法满足机器人实时控制的低延迟要求；其二，机器人硬件高度异构，从GPU工作站到ARM嵌入式芯片差异巨大；其三，具身模型形态多样，涵盖VLA（视觉-语言-动作）和WAM（世界动作模型）等多种架构。

现有解决方案或只支持单一模型类型，或缺乏机器人专用I/O支持，无法跨机器人平台迁移。

## 核心方法

Embodied.cpp 采用统一的**五层架构**，将所有具身AI模型的执行路径标准化：

![Embodied.cpp框架总览](https://arxiv.org/html/2607.02501v1/x5.png)
*图：Embodied.cpp项目总览——五层统一架构连接传感器输入与机器人控制输出*

| 层次 | 组件 | 功能 |
|------|------|------|
| 第一层 | 输入适配器（Input Adapters） | 处理相机、力觉/触觉、IMU等传感器数据 |
| 第二层 | 序列构建器（Sequence Builders） | 构建Transformer输入序列 |
| 第三层 | 骨干网络执行（Backbone Execution） | 统一的Transformer推理路径 |
| 第四层 | 头部插件（Head Plugins） | 可插拔的动作头和预测模块 |
| 第五层 | 部署适配器（Deployment Adapters） | 连接仿真器和真实机器人 |

### 三大设计原则

**1. 模块化多频率执行（Modular Multi-Rate Execution）**：视觉编码器、语言模型、动作头可以以不同频率独立运行，无需强制同步，允许高频低延迟的底层控制与低频高精度的视觉理解并行。

**2. 延迟优先融合推理（Latency-First Fused Inference）**：针对 batch=1 的单步推理场景优化，通过算子融合减少内存带宽瓶颈，而非追求高吞吐量。

**3. 可扩展算子与I/O支持（Extensible Operator and I/O）**：新模型只需实现标准接口即可接入，支持CUDA、Metal、CPU等多后端。

### 具身模型分类体系

论文系统梳理了具身AI模型的两大家族八种子类型：

![具身模型架构分类](https://arxiv.org/html/2607.02501v1/x4.png)
*图：具身AI模型架构分类树——VLA与WAM两大家族的层次化分类体系*

**VLA家族**：AR-Token VLA（OpenVLA）、VLM-Backboned VLA（pi0.5）、Hierarchical VLA（RT-H）、Asynchronous VLA（GR00T N1）

**WAM家族**：Predict-then-Act WAM（UniPi）、Unified AR-Modeling WAM（WorldVLA）、Shared-Backbone WAM（Cosmos Policy）、Latent-space WAM（LaWAM）

## 实验结果

### 真实机器人VLA部署

| 模型 | 骨干网络 | Action Chunk | 任务成功率 | 推理延迟(ms) | 显存(MiB) |
|------|---------|-------------|---------|------------|---------|
| HY-VLA | Hunyuan-VL | 20 | **100.0%** [83.9, 100.0] | 1340.3 | 6850 |
| pi0.5 | PaliGemma | 50 | **91.0%** [86, 94] | 266.6 | 6546 |

### WAM微基准测试（内存压缩）

| 运行时 | 量化精度 | 内存/块(MiB) | 延迟/块(ms) | 输出余弦相似度 |
|--------|---------|------------|----------|------------|
| Python原始 | BF16 | 312.2 | 3.236 | 1.000 |
| **Embodied.cpp** | **Q4_K** | **88.1** | **3.171** | **>0.9997** |

内存占用从312.2 MiB压缩至88.1 MiB，减少**71.8%**，输出质量几乎无损。

### 与现有系统全面对比

| 系统 | VLA支持 | WAM支持 | 模块化 | 边缘部署 | 异构硬件 | 机器人I/O |
|------|--------|--------|--------|--------|--------|---------|
| llama.cpp | ✗ | ✗ | ✗ | ✓ | △ | ✗ |
| ONNX Runtime | ○ | ○ | ○ | ✓ | ✓ | ✗ |
| vla.cpp | ✓ | ✗ | △ | ✓ | △ | ✓ |
| **Embodied.cpp** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

## 总结

Embodied.cpp 填补了具身AI研究与工程部署之间的重要空缺，通过五层统一抽象和C++实现，使得VLA/WAM模型可以真正在异构机器人上高效运行。71.8%的内存压缩和HY-VLA 100%的任务成功率充分证明了框架的实用价值。

主要局限性在于：目前只在有限的机器人型号（AgileX PiPER、WAM）上做了测试，泛化到其他机器人平台的难度未知；此外，高延迟（HY-VLA需要1340ms）在动态任务中可能成为瓶颈。
