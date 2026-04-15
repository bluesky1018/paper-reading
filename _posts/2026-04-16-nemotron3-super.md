---
layout: post
title: "Nemotron 3 Super 深度解读：NVIDIA 开源高效 MoE 混合 Mamba-Transformer 推理大模型"
date: 2026-04-16
categories: [论文解读, 大模型]
tags: [MoE, Mamba, Transformer, 推理模型, NVIDIA, 混合架构, LatentMoE, 投机解码, NVFP4, Agentic]
---

> **论文**：Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning
> **arXiv**：[2604.12374](https://arxiv.org/abs/2604.12374)
> **机构**：NVIDIA
> **日期**：2026-04-16

---

## 一句话总结

Nemotron 3 Super 是 NVIDIA 推出的一款 120B 总参数、仅 12B 激活参数的混合 Mamba-Attention MoE 开源大模型，通过创新的 LatentMoE 架构、原生多 Token 预测（MTP）投机解码以及首次在 NVFP4 精度下完成大规模预训练，在保持竞争力学术性能的同时，推理吞吐量相比同量级模型提升高达 7.5 倍。

---

## 背景与问题

随着大语言模型（LLM）走向实际部署，**推理效率**已成为与模型精度同等重要的核心指标。当前主流的密集型 Transformer 模型（如 GPT-4 系列、Llama 系列）在激活参数数量上的巨大需求，直接导致了推理成本高企、吞吐量受限的困境。

现有改进方案各有局限：

- **稀疏 MoE（Mixture-of-Experts）**：通过条件计算减少激活参数，但受限于专家路由的通信开销（all-to-all），在分布式推理中带宽压力巨大。
- **线性注意力 / Mamba 架构**：以线性时间复杂度处理长序列，但在需要精确全局注意力的任务上有所欠缺。
- **量化训练**：低精度训练可以降低存储和计算开销，但极低精度（如 FP4）对训练稳定性的影响尚不明朗。

Nemotron 3 Super 的目标是：**用最少的激活参数，实现最高的精度和推理效率**，并面向 Agentic 推理（工具调用、软件工程、长链条推理）场景进行专项优化。

---

## 核心方法

### 1. 整体架构：混合 Mamba-Attention MoE

Nemotron 3 Super 采用混合架构，将 Mamba-2 线性注意力层与稀疏 MoE 全连接层交错排列，并在关键位置插入标准全注意力层作为"锚点"。

![架构总览](https://arxiv.org/html/2604.12374/x2.png)
*图 1：Nemotron 3 Super 混合层结构示意图，展示 Mamba 块、MoE 块与注意力锚点的交错排列方式。*

主要架构参数如下：

| 参数 | 数值 |
|------|------|
| 总层数 | 88 |
| 模型维度 d | 4096 |
| 每层总专家数 | 512 |
| Top-k 激活专家数 | 22 |
| MoE Latent 维度 | 1024 |
| MTP 层数（共享权重） | 2 |
| 最大上下文长度 | 1M tokens |
| 总参数量 | 120B |
| 激活参数量 | 12B |

**Mamba-2 的作用**：以常数大小的 KV 状态处理长序列，无需随序列长度增长 KV 缓存，天然适合超长上下文推理。

**注意力锚点**：在 Mamba 主导的层间，战略性地插入全注意力层，确保模型在需要时仍能进行完整的 token 间交互，维持全局信息感知能力。

---

### 2. LatentMoE：高效稀疏专家架构

这是 Nemotron 3 Super 最核心的架构创新。传统 MoE 在进行 token 路由时，需要在隐层维度 d 上做 all-to-all 通信，带宽消耗与 d 成正比。LatentMoE 引入了一个小得多的**潜在维度 ℓ（Latent Dimension）**，在 token 被路由到专家之前，先将其投影到低维空间。

![标准 MoE 架构](https://arxiv.org/html/2604.12374/figures/standard_moe.png)
*图 2：标准 MoE 架构，token 在全维度 d 下进行路由和专家计算。*

![LatentMoE 架构](https://arxiv.org/html/2604.12374/figures/latent_moe.png)
*图 3：LatentMoE 架构，token 先投影至低维 ℓ，再进行路由，大幅减少通信开销。*

**LatentMoE 核心设计思路**：

设原始隐层维度为 d，潜在维度为 ℓ（ℓ < d），则：
- All-to-all 通信量减少因子：**d/ℓ**
- 专家总数从 N 扩展至 N' = N × (d/ℓ)
- Top-k 激活专家从 K 扩展至 K' = K × (d/ℓ)

在 Nemotron 3 Super 中，d = 4096，ℓ = 1024，即 d/ℓ = 4，这意味着：
- 通信带宽节省 4 倍
- 专家数量从 128 扩展到 512
- 激活专家从 6 扩展到 22

**四条设计原则**：
1. 缩减专家隐层维度以节省带宽；
2. 保持非线性激活预算（K × m，m 为专家 FFN 中间维度）；
3. 扩展专家数量 N 和 Top-k 数 K 以提升模型质量；
4. 在精度与效率之间寻求最优平衡点。

---

### 3. 多 Token 预测（MTP）：原生投机解码

传统大模型推理是逐 token 自回归生成，吞吐量瓶颈明显。投机解码（Speculative Decoding）通过让小"草稿模型"先生成多步候选 token，再由大模型验证，从而实现吞吐量的显著提升——但维护独立草稿模型会带来额外开销。

Nemotron 3 Super 原生集成了 **MTP（Multi-Token Prediction）层**，使模型本身即可充当草稿模型，无需外部辅助模型。

**关键设计**：
- 2 个共享权重的 MTP 头，分布在预测偏移量上；
- 共享权重提升了自回归草稿的鲁棒性；
- 草稿深度 D=3 在吞吐量-延迟 Pareto 前沿上取得最优效果。

![MTP 接受率](https://arxiv.org/html/2604.12374/x3.png)
*图 4：SPEED-Bench 上草稿长度为 7 时各草稿 Token 的接受率，Nemotron 3 Super 平均接受长度达 3.45。*

![MTP 吞吐量对比](https://arxiv.org/html/2604.12374/figures/mtp_trtllm_perf.png)
*图 5：NVFP4 checkpoint 下 MTP 关闭与草稿深度 1/3 的总吞吐量与用户吞吐量对比（B200 GPU）。*

**MTP 在推理中的表现**（SPEED-Bench 平均接受长度）：
| 模型 | 平均接受长度 |
|------|------------|
| Nemotron 3 Super | **3.45** |
| Qwen3-Next | 3.33 |
| DeepSeek-R1 | 2.70 |

---

### 4. NVFP4 精度预训练

Nemotron 3 Super 是**全球首个在 NVFP4 精度下完成大规模稳定预训练（至 25T tokens）的大模型**，这也是 Nemotron 3 家族的三大首创之一。

不同层类型采用不同精度策略：

| 层类型 | 精度 |
|--------|------|
| 大部分线性层 | NVFP4 |
| 网络末尾 15% | BF16 |
| QKV 与注意力投影 | BF16 |
| Mamba 输出投影 | MXFP8 |
| Latent 投影层 | BF16 |

**NVFP4 带来的挑战与解决方案**：

训练过程中发现，NVFP4 会产生约 3 倍于 BF16 的零值权重梯度，根源在于低范数专家通道中的梯度下溢。研究人员深入分析了通道幅值模式随训练进程的演变：

![通道幅值模式](https://arxiv.org/html/2604.12374/x4.png)
*图 6：专家层权重 FC1 和 FC2 矩阵中通道幅值分布，对比 0.5T tokens 和 23T tokens 时的变化。*

![零值梯度对比](https://arxiv.org/html/2604.12374/x5.png)
*图 7：BF16 训练与 NVFP4 训练的 Nemotron 3 Nano 中零值权重梯度元素占比对比。*

![零值梯度来源](https://arxiv.org/html/2604.12374/x6.png)
*图 8：500B 和 750B tokens 时路由专家层中零值梯度的来源分析。*

![精度切换效果](https://arxiv.org/html/2604.12374/x7.png)
*图 9：切换至 MXFP8 精度后下游任务精度变化，在 Annealing 前切换可改善 loss 但不提升下游准确率。*

---

### 5. 预训练数据：25 万亿 Token

预训练数据分为两个阶段：

- **Phase 1**（80%，约 20T tokens）：注重数据多样性，覆盖宽泛的知识领域；
- **Phase 2**（20%，约 5T tokens）：侧重高质量数据源，强化模型在关键基准上的表现。

![预训练数据分布 Phase 1](https://arxiv.org/html/2604.12374/x8.png)
*图 10：Phase 1 预训练数据混合比例分布饼图。*

![预训练数据分布 Phase 2](https://arxiv.org/html/2604.12374/x9.png)
*图 11：Phase 2 预训练数据混合比例分布饼图，高质量源占比更大。*

此次同步开源了多个全新合成数据集：

| 数据集名称 | 规模 | 内容 |
|-----------|------|------|
| Synthetic Code Concepts | ~1500 万样本 | Python 问题-解答对，涵盖 91 个编程概念 |
| Synthetic Unconditional Algorithmic | 2 亿 tokens | LeetCode 风格算法题 |
| Synthetic Economics | 若干 MCQ | 微观/宏观/计量经济学题目 |
| Synthetic Formal Logic | 若干样本 | 谓词/命题逻辑推理题 |
| Synthetic MCQ | ~350 万样本（约 16 亿 tokens） | MMLU 风格多选题 |

**Checkpoint 合并技巧**：在稳定学习率阶段采用滑动窗口权重平均，在 12 个基准的平均分上提升 2-4 分，且节省约 4T tokens 的计算（相当于总预训练 FLOPs 的 16%），无需专门的 Decay 训练阶段。

![Checkpoint 合并效果](https://arxiv.org/html/2604.12374/x10.png)
*图 12：预训练过程中 Checkpoint 合并前后在 12 个基准的平均准确率对比。*

---

### 6. 后训练流程

Nemotron 3 Super 的后训练管线精心设计，针对 Agentic 推理进行了全面强化。

![后训练流程](https://arxiv.org/html/2604.12374/x11.png)
*图 13：Nemotron 3 Super 完整后训练流程概览，包含 SFT 和多个 RL 阶段。*

#### SFT（监督微调）两阶段

**Stage 1**：Token 级别全局平均损失（256K 序列打包），专注于基础对话与指令跟随；

**Stage 2**：样本级别平均损失（512K 序列打包），防止超长输出主导梯度更新，维持对短输出样本的充分学习。

MTP 头在整个 SFT 过程中同步训练，缩放因子为 0.3。

**SFT 数据亮点**（与 Nemotron 3 Nano 相比大幅扩展）：

- **软件工程数据**：来自 SWE-Gym、R2E-Gym、SWE-rebench，轨迹来自 Qwen3-Coder-480B 蒸馏；

![Agentic CLI 数据集构建](https://arxiv.org/html/2604.12374/x12.png)
*图 14：Agentic CLI 数据集构建与训练流程，含任务来源、蒸馏模型与数据处理步骤。*

- **对话式工具调用**：279,116 条对话，覆盖 838 个领域（Nano 仅 15,588 条 / 5 个领域）；

![对话工具调用数据生成流程](https://arxiv.org/html/2604.12374/x13.png)
*图 15：对话式工具调用 SFT 数据生成流程概览。*

- **通用工具调用**：150 万条多样化工具调用轨迹；
- **长上下文**：基于 128K-512K 文档的 4-7 跳推理 QA；
- **金融推理**：36.6 万条 SEC 文件 QA；
- **CUDA 内核**：10 万条内核生成/修复/优化样本；
- **SQL**：96,500 条跨 3 种方言、60 个行业的 text-to-SQL 样本；
- **终端操作**：84,864 条使用 Terminus 2 框架的终端操作样本。

**推理模式**：新增"低强度（low-effort）"推理模式，通过 SFT 数据中 2% 来自 GPT-OSS-120B 低强度输出的样本引入，让模型在简单任务上避免过度推理，节省计算资源。

#### RL 强化学习多阶段

| 阶段 | 方法 | 说明 |
|------|------|------|
| RLVR | 21 个环境（数学、代码、STEM、安全、Agentic 等）同时训练，防止单任务回退 | 基于可验证奖励的强化学习 |
| SWE-RL | 独立阶段，专门针对长时程软件工程任务 | 由于 Rollout 长度需单独处理 |
| RLHF | 使用 Qwen3-Nemotron-235B-A22B-GenRM 作为奖励模型，提升指令跟随能力 | 人类反馈强化学习 |
| MTP Healing | 冻结主干权重，在 RLVR Rollouts 上以 NLL 损失重训 MTP 头 | 恢复 RL 后 MTP 头的准确性 |

---

## 实验结果

### 基座模型性能

| 基准 | Nemotron 3 Super | Ling-flash-Base-2.0 | GLM-4.5-Air-Base |
|------|-----------------|---------------------|-----------------|
| MMLU (5-shot) | **86.01** | 81.00 | 81.00 |
| MMLU-Pro | **75.65** | 62.10 | 58.20 |
| GPQA-Diamond | **60.00** | 36.00 | 23.20 |
| MATH | **84.84** | 63.80 | 50.36 |
| AIME 2024 (pass@32) | **53.33** | 30.00 | 20.00 |
| RULER 1M | **71.00** | — | — |

Nemotron 3 Super 基座模型在所有主要基准上全面领先同类开源模型，尤其在 GPQA-Diamond（科学推理）和 AIME 2024（数学竞赛）上差距显著。

### 推理吞吐量

在 B200 GPU 上，8K 输入 / 64K 输出场景：

| 对比模型 | 吞吐量提升 |
|---------|----------|
| vs. GPT-OSS-120B | 最高 **2.2×** |
| vs. Qwen3.5-122B | 最高 **7.5×** |

![吞吐量与精度对比](https://arxiv.org/html/2604.12374/x1.png)
*图 16：Nemotron 3 Super 与 GPT-OSS-120B、Qwen3.5-122B 的精度-吞吐量综合对比，展示 2.2× 和 7.5× 的吞吐量优势。*

### 量化版本

| Checkpoint | 精度 |
|-----------|------|
| NVFP4 量化后训练版 | NVFP4 |
| FP8 量化后训练版 | FP8 |
| BF16 后训练版 | BF16 |
| BF16 基座版 | BF16 |

AutoQuantize 算法在部署限制感知搜索下（线性层融合、MoE 层约束等）自动选择最优量化策略，Mamba 状态量化也单独进行了探索。

---

## 开源情况

NVIDIA 以 **CC BY 4.0 许可证**在 HuggingFace 上开放了以下资源：

- 基座模型 BF16 checkpoint
- 后训练版本（BF16、FP8、NVFP4）
- 专项预训练合成数据集
- 后训练数据集合集
- 训练方案（NVIDIA-NeMo/Nemotron GitHub 仓库）

---

## 总结

Nemotron 3 Super 代表了当前开源大模型在**效率与能力平衡**方面的一个重要里程碑，其核心贡献可以概括为以下几点：

1. **LatentMoE**：通过低维潜空间路由，在不损失模型容量的前提下大幅降低 MoE 通信开销，将吞吐量提升与专家扩展合二为一；

2. **原生 MTP 投机解码**：无需外部草稿模型，内置多 Token 预测头，实现 3.45 的平均草稿接受长度，显著提升实际部署吞吐量；

3. **NVFP4 大规模预训练**：首次在 FP4 精度下完成 25T tokens 规模的稳定预训练，为低精度训练工程化树立了新标杆；

4. **全栈 Agentic 优化**：从预训练数据（代码、逻辑、数学）到后训练多阶段 RL，全面针对工具调用、软件工程、长链条推理等 Agentic 场景进行定向强化；

5. **全面开源**：模型、数据、训练方案一并开放，有力推动社区研究与应用。

对于需要在有限计算资源下部署强力推理模型的应用场景，Nemotron 3 Super 是目前最值得关注的开源选择之一。其混合 Mamba-Attention 架构与 LatentMoE 的组合，也为未来大模型架构设计提供了新的探索方向。
