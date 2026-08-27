---
layout: post
title: "Next-Chunk 推理 RL 真的比 SFT 更好吗？重新审视无 CoT 数据下的训练策略"
date: 2026-08-28
categories: [论文解读, 大语言模型]
tags: [推理模型, 强化学习, SFT, CoT, 数学推理]
---

> 📄 **论文**：Is Next-Chunk Reasoning RL Really Better than SFT? Revisiting Training Strategies under no-CoT Data
> 🔗 **arXiv**：[2608.23256](https://arxiv.org/abs/2608.23256)
> 🏢 **机构**：中国科学技术大学、上海人工智能实验室

## 一句话总结

本文通过受控研究发现：简单的混合 SFT（同时训练无 CoT 和长 CoT 数据）在后 RLVR 性能上明显优于 Next-Chunk 推理 RL，同时计算量减少超过 60 倍，并揭示了"前 RLVR 准确率不是后 RLVR 潜力的可靠预测指标"。

## 背景与问题

**推理导向后训练**的核心扩展问题：如何利用**无 CoT 数据**（worked solutions、教材推导、研究论文等）？这类数据大量存在，但缺乏显式的思维链标注。

两种主要策略：
1. **直接 SFT**：在无 CoT 数据上做监督微调，但可能扭曲模型的现有推理格式
2. **Next-Chunk 推理 RL（NCR）**：模型生成隐式推理，通过预测下一段文本的能力来获得奖励

现有研究主要对比 NCR 与传统 SFT 基线，但未解答：**增益来自 RL 框架本身，还是更有效地暴露了无 CoT 数据？**

![Comparison Teaser](https://arxiv.org/html/2608.23256v1/teaser2_3.png)
*图：各训练策略的后 RLVR 性能对比——Mixed SFT 以极少算力显著超越 NCR 方法*

## 实验设计

### 模型与数据

- **基础模型**：Qwen3-30B-A3B-Base（选用 base checkpoint 以避免先验后训练的干扰）
- **长 CoT 数据**：从 AoPS 问题用 DeepSeek-V3.2 生成，152K 轨迹（≈1.95B tokens）
- **无 CoT 数据**：AoPS 原始简短题解（含推导但无显式推理链），421K 解答（≈0.53B tokens）
- **RLVR 数据**：DAPO-Math-17K

### 五种训练策略对比

| 策略 | 描述 |
|-----|------|
| **NTR（RPT）** | 次词粒度的 next-chunk 推理 RL |
| **NSR（RLPT）** | 次句粒度的 next-chunk 推理 RL |
| **Sequential SFT** | 先在无 CoT 数据上 SFT，再在长 CoT 上 SFT |
| **Mixed SFT** | 同时在无 CoT 和长 CoT 数据上 SFT |
| **Reasoning SFT** | 仅在长 CoT 数据上 SFT（基线） |

## 主要发现

### 发现 1：Mixed SFT 是更强的替代方案

| 方法 | 后 RLVR 平均准确率（域内） | 后 RLVR 平均准确率（域外） | 训练算力 |
|-----|------------------------|------------------------|---------|
| NCR 方法（NTR/NSR） | ~64.3 | - | 基准 |
| **Mixed SFT** | **67.4** | **更高** | **<1/60 的算力** |

Mixed SFT 在所有域内和域外基准上超越 next-chunk 推理 RL，同时 **计算量减少超过 60 倍**。

### 发现 2：数据混合方式至关重要

Mixed SFT 显著优于 Sequential SFT：
- **Sequential SFT** 的 Reasoning SFT 最终阶段会部分覆写无 CoT 数据学到的知识
- **Mixed SFT** 同时暴露两种数据类型，虽然前 RLVR 准确率更低（因输出格式冲突），但不会发生跨阶段遗忘

### 发现 3：前 RLVR 准确率不预测后 RLVR 潜力

| 方法 | 前 RLVR 准确率 | 后 RLVR 准确率（AIME25） | 改善幅度 |
|-----|--------------|----------------------|---------|
| NTR/NSR/Sequential SFT/Reasoning SFT | 45.9-48.8 | - | 3.3-10.3 |
| **Mixed SFT** | **27.5（最低！）** | **61.1（最高！）** | **+33.7** |

Mixed SFT 的前 RLVR 准确率比其他所有方法低约 20 个点，却达到最高的后 RLVR 准确率——提升幅度超过其他方法的三倍。

### 案例分析

![Mixed SFT Case](https://arxiv.org/html/2608.23256v1/case_mixed_sft.png)
*图：Mixed SFT 生成的推理轨迹案例——融合了无 CoT 数据的知识和长 CoT 的推理格式*

![NCR Case](https://arxiv.org/html/2608.23256v1/case_rpt.png)
*图：RPT（NTR）生成的推理轨迹案例——隐式推理的实际表现*

## 总结

本文的核心贡献是：通过严格控制实验，揭示了 next-chunk 推理 RL 的性能优势主要来自"更有效利用无 CoT 数据"的效果，而非 RL 框架本身——而这一效果可以通过更简单的 Mixed SFT 以极低代价实现。

**重要警示**：在评估无 CoT 训练策略时，必须在**完整后训练流水线**（包括 RLVR 阶段）的背景下进行评估，仅比较前 RLVR 性能会得出误导性结论。

局限性：实验基于 Qwen3-30B-A3B-Base，在其他架构和规模上的普适性需验证；此外，研究聚焦于数学推理，其他推理领域的结论可能有所不同。
