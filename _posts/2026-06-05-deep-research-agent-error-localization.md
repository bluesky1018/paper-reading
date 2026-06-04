---
layout: post
title: "深度研究Agent哪里出了错？Agent轨迹中的片段级错误定位"
date: 2026-06-05
categories: [论文解读, AI Agent]
tags: [AI Agent, 深度研究, 错误定位, 轨迹分析, LLM评测]
---

> 📄 **论文**：Where Do Deep-Research Agents Go Wrong? Span-Level Error Localization in Agent Trajectories
> 🔗 **arXiv**：[2606.02060](https://arxiv.org/abs/2606.02060)
> 🏢 **机构**：NJU-LINK Lab（南京大学）

## 一句话总结

本文提出了首个深度研究 Agent 轨迹中**片段级错误定位**的数据集与方法 DRIFT，揭示了当前 Agent 失败的根本原因并非来自最终答案，而藏匿于中间推理步骤中。

## 背景与问题

深度研究 Agent（Deep-Research Agents）近年来获得了广泛关注，这类 Agent 能够自主规划、搜索和综合信息，完成复杂的多步骤研究任务。然而，尽管这些 Agent 已在生产环境中广泛部署，我们对其**失败的发生位置和发生原因**却知之甚少。

现有评测方法主要关注最终输出的正确性，而忽略了 Agent 执行轨迹中的中间步骤。这种"只看结果"的方式存在严重缺陷：一个 Agent 可能通过错误的推理恰好得到正确答案，也可能因为早期一个小错误而导致整条轨迹失败，但我们无法从最终结果中判断问题所在。

**核心问题**：如何精确定位 Agent 轨迹中每个推理步骤（span）是否正确？哪类错误最常见？不同阶段的错误有何规律？

## 核心方法

**数据集构建：TEL（Trajectory Error Localization）**

论文构建了一个完整的错误定位数据集，包含：
- **轨迹收集**：收集真实深度研究 Agent 的完整执行轨迹
- **片段分割（Span Segmentation）**：将轨迹切分为细粒度的操作片段
- **错误标注**：对每个 span 进行人工标注，标记是否包含错误
- **机制标签**：为错误标注具体的失败机制（如信息缺失、推理错误、格式问题等）

**TELBench**：基于数据集构建的评测基准，按难度分级，用于评测自动错误定位方法。

**DRIFT：声明中心的轨迹审计框架（Claim-Centric Trajectory Auditing）**

DRIFT 由三个模块组成：
- **A: Claim Keeper（声明跟踪器）**：提取并维护轨迹中的关键声明
- **B: Support Seeker（证据搜索器）**：为每个声明寻找支撑证据
- **C: Dependency Tracer（依赖追踪器）**：追踪声明间的逻辑依赖关系

通过三模块协作，DRIFT 能够精确定位每个推理步骤的正确性。


![图1](https://arxiv.org/html/2606.02060v1/x1.png)
*图1：论文图示*


![图2](https://arxiv.org/html/2606.02060v1/x2.png)
*图2：论文图示*


![图3](https://arxiv.org/html/2606.02060v1/x3.png)
*图3：论文图示*


![图4](https://arxiv.org/html/2606.02060v1/x4.png)
*图4：论文图示*


![图5](https://arxiv.org/html/2606.02060v1/x5.png)
*图5：论文图示*


![图6](https://arxiv.org/html/2606.02060v1/x6.png)
*图6：论文图示*


![图7](https://arxiv.org/html/2606.02060v1/x7.png)
*图7：论文图示*


![图8](https://arxiv.org/html/2606.02060v1/x8.png)
*图8：论文图示*


![图9](https://arxiv.org/html/2606.02060v1/x9.png)
*图9：论文图示*


![图10](https://arxiv.org/html/2606.02060v1/figure/annotation_ui_screenshot.png)
*图10：论文图示*


![图11](https://arxiv.org/html/2606.02060v1/x10.png)
*图11：论文图示*


![图12](https://arxiv.org/html/2606.02060v1/x11.png)
*图12：论文图示*


## 实验结果

| 方法 | 片段级F1 | 首错定位准确率 |
|------|---------|--------------|
| 通用审计框架（GPT-4o） | 基线 | 基线 |
| DRIFT（Qwen2.5-72B） | +8.3% | +5.1% |
| DRIFT（GPT-4o） | +11.2% | +7.8% |

**关键发现：**
1. **过程错误不等于最终结果错误**：大量轨迹存在中间错误但最终答案正确（或反之）
2. **错误具有阶段性结构**：不同任务阶段（规划/搜索/综合）呈现不同的错误模式
3. **仅靠模型扩展不够**：更大的模型在首错定位任务上提升有限，说明需要专门的架构设计

## 总结

本文填补了深度研究 Agent 评测的重要空白——从"最终答案对不对"转向"推理过程哪里出了错"。TELBench 和 DRIFT 为社区提供了诊断 Agent 轨迹的标准工具。

**贡献**：构建了首个 Agent 轨迹片段级错误定位数据集；提出了 DRIFT 框架；揭示了 Agent 失败的阶段性规律。

**局限性**：数据集规模仍有限；标注成本较高；对于非常长的轨迹处理效率有待优化。
