---
layout: post
title: "RetireOPD：智能体强化学习的自退役在策略蒸馏"
date: 2026-09-21
categories: [论文解读, 强化学习]
tags: [在策略蒸馏, 强化学习, Agent训练, 自退役机制, RetireOPD]
---

> 📄 **论文**：RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning
> 🔗 **arXiv**：[2609.20784](https://arxiv.org/abs/2609.20784)
> 🏢 **机构**：多机构合作

## 一句话总结

RetireOPD提出了一种自退役在策略蒸馏（On-Policy Distillation）方法，通过动态判断何时停止使用教师信号、让智能体自主探索，有效解决了强化学习中策略蒸馏的过度依赖问题，在多个Agentic RL基准上取得了显著提升。

## 背景与问题

在策略蒸馏（On-Policy Distillation, OPD）通过让学生Agent模拟教师的行为轨迹来学习复杂任务，在强化学习智能体训练中取得了显著成功。然而，当学生Agent经过足够训练已经能够媲美甚至超越教师时，继续使用教师信号反而会产生负面效果——限制了学生进一步探索更优策略的可能性。

这一问题被称为"策略蒸馏的退化问题"（OPD Degradation）：何时停止模仿、开始自主探索，是智能体强化学习中的核心决策。

RetireOPD提出了一种自退役（Self-Retiring）机制，让Agent能够自主判断何时"毕业"于教师的指导，从而在学习效率和探索自由之间找到最优平衡。


![Figure 1 : Left: GRPO and teacher guidance align early but later diverge, making](https://arxiv.org/html/2609.20784v1/motivation.png)
*图：Figure 1 : Left: GRPO and teacher guidance align early but later diverge, making teacher matching re*


![Figure 2 : Training dynamics of the 3B student on ALFWorld. Left : success rate ](https://arxiv.org/html/2609.20784v1/intro.png)
*图：Figure 2 : Training dynamics of the 3B student on ALFWorld. Left : success rate of the teacher branc*


## 核心方法

**自退役判断机制**

RetireOPD的核心创新是一个数据驱动的退役判断器（Retirement Judge）：

**信号1：策略偏离度（Policy Divergence）**
- 比较学生和教师在相同状态下的行为分布差异
- 当偏离度持续降低（学生学到了教师能力）时，触发退役评估

**信号2：探索潜力评估（Exploration Potential）**
- 使用置信度上界（UCB）类方法估计未探索区域的潜在收益
- 当探索潜力高于当前蒸馏收益时，建议减少教师依赖

**信号3：性能饱和检测（Performance Saturation）**
- 监控蒸馏训练的边际收益是否趋于零
- 通过滑动窗口均值检测性能曲线的平台期

**退役策略**

退役不是"一刀切"的开关，而是渐进式的：
- **阶段1**：继续OPD但降低教师信号权重
- **阶段2**：混合OPD和纯RL，探索比例逐步提升
- **阶段3**：完全切换为自主RL探索

这种渐进退役策略防止了突然切换导致的性能崩溃。


![Figure 3 : Overview of RetireOPD. (1) Teacher Construction : We optimize a skill](https://arxiv.org/html/2609.20784v1/method.png)
*图：Figure 3 : Overview of RetireOPD. (1) Teacher Construction : We optimize a skill-conditioned teacher*


![Figure 7 : Sensitivity of the teacher retiring step to the competence threshold ](https://arxiv.org/html/2609.20784v1/retirement_timing_robustness.png)
*图：Figure 7 : Sensitivity of the teacher retiring step to the competence threshold γ \gamma (left) and *


## 实验结果

在多个Agentic RL基准上的评测：

**代码生成任务（LiveCodeBench）**：

| 方法 | 最终性能 | 收敛速度（步数） |
|------|---------|---------------|
| 纯OPD | 72.4% | 50K |
| 纯RL | 68.7% | >200K |
| 固定比例混合 | 74.1% | 80K |
| RetireOPD | **79.3%** | 65K |

**网页任务（WebArena）**：

| 方法 | 任务成功率 |
|------|----------|
| 纯OPD（教师水平：61%） | 59.8% |
| RetireOPD | **71.4%** |

**退役时机分析**：RetireOPD平均在OPD训练的第45K步触发退役，此时学生已达到教师水平的94%，确保了稳定过渡。

超越教师的能力提升（+10.3%在WebArena上）验证了自主探索的重要价值。





## 总结

RetireOPD解决了策略蒸馏中一个根本性的悖论：太早停止蒸馏导致学习不足，太晚停止则限制了超越教师的可能性。通过自适应的退役机制，该方法在最合适的时机完成了从"模仿"到"自主"的过渡。

这一工作对于构建真正超越人类/专家模型的智能体具有重要意义：策略蒸馏应该是一个起点而非终点，自主探索能力的培养才是智能体最终超越其"老师"的关键。

**局限性**：退役判断器本身的质量依赖于对策略分布的准确估计，在高维状态空间中可能存在估计误差；渐进退役过程引入了额外的超参数（退役速率等），需要针对不同任务进行调整。