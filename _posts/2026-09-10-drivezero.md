---
layout: post
title: "DriveZero：超越人类示范的端到端自动驾驶"
date: 2026-09-10
categories: [论文解读, 自动驾驶]
tags: [自动驾驶, 强化学习, 端到端, 视觉基础模型, 闭环训练]
---

> 📄 **论文**：DriveZero: End-to-End Driving Beyond Human Demonstrations
> 🔗 **arXiv**：[2609.06055](https://arxiv.org/abs/2609.06055)
> 🏢 **机构**：小米汽车 AD & Robotics L3 Team

## 一句话总结

DriveZero 将驾驶分解为感知模型（DriveVFM）和行为模型（DriveRL），分别在最适合的学习范式下预训练，再通过轨迹蒸馏融合为纯摄像头端到端规划器，实现了超越人类示范的驾驶性能。

## 背景与问题

当前主流端到端自动驾驶系统的核心局限在于：**依赖模仿人类驾驶日志**。这导致系统的行为天花板被人类驾驶数据的质量和覆盖范围所限制——遇到人类罕见或回避的危险场景时，系统无法有效处理。

此外，感知和行为的最优学习范式完全不同：
- **感知**需要理解世界，受益于海量多样的视觉数据
- **行为**需要与世界交互，必须依赖闭环反馈（开环模仿无法捕捉驾驶决策的因果链）

DriveZero 的洞察是：**不要把两者强行融入一个训练范式**，而是让它们各自在最适合的范式中充分发展，再组合起来。

## 核心方法

### 三阶段流程

![DriveZero 整体架构](https://arxiv.org/html/2609.06055v1/teaser.png)
*图：DriveZero 三阶段训练流程：① DriveRL 通过闭环 RL 从零训练特权教师策略；② DriveVFM 融合多个视觉基础模型预训练视觉主干；③ DriveZero 通过轨迹蒸馏将两者统一为纯摄像头端到端规划器。*

### 第一阶段：DriveRL——从零学习驾驶行为

DriveRL 是一个**闭环强化学习系统**，完全不使用行为克隆预训练：

- **输入**：特权结构化场景状态 + 导航目标点
- **输出**：连续动作（纵向加速度 jerk + 轮胎转向角速率）
- **仿真环境**：从真实 nuPlan 驾驶日志构建交互世界，背景车辆以日志回放、规则模型或学习策略混合控制
- **并行规模**：96 GPU 上同时运行最多 196,608 个世界
- **优化算法**：PPO，奖励包含安全事件、目标到达和驾驶质量

DriveRL 从随机初始化出发，仅用日志数据初始化场景和导航目标，训练约 21 小时后习得鲁棒驾驶行为。

**测试时计算缩放（DriveRL-TTS）**：利用训练好的 Critic 网络对 N 个采样轨迹进行价值评分，选择最优：

| 方法 | Val14↑ | Test14-hard↑ | Test14-random↑ | 平均↑ |
|------|--------|-------------|----------------|------|
| DriveRL | 95.16 | 89.97 | 94.50 | 93.01 |
| DriveRL-TTS (N=8) | 95.17 | 89.97 | 94.92 | 93.12 |
| DriveRL-TTS (N=32) | 95.49 | 90.95 | 95.83 | 93.40 |
| DriveRL-TTS (N=64) | 95.54 | 91.13 | 95.93 | **93.57** |

### 第二阶段：DriveVFM——视觉基础模型融合

DriveVFM 将多个冻结的视觉基础模型（DINOv3、SigLIP2、SAM、Depth Anything V2）蒸馏为一个统一主干，**仅从原始图像训练，无需任何任务标注**。

![视觉表征蒸馏与 PCA 分析](https://arxiv.org/html/2609.06055v1/pca.png)
*图：DriveVFM 的多模型融合蒸馏过程及学到的视觉表征 PCA 可视化，展示了对不同场景元素（车辆、行人、道路、障碍物）的清晰区分能力。*

### 第三阶段：DriveZero——多模态轨迹蒸馏

DriveZero 通过以下方式将 DriveRL 教师的行为迁移到纯摄像头学生：

![轨迹蒸馏框架](https://arxiv.org/html/2609.06055v1/distill.png)
*图：DriveZero 的多模态轨迹蒸馏框架，展示目标引导增强、轨迹提议评分和从 DriveRL 教师到摄像头学生的知识迁移流程。*

1. **多模态输入**：DriveVFM 提取的视觉特征 + 地图（可选）
2. **目标条件增强**：基于目标点生成多样化训练样本
3. **轨迹提议评分**：$\mathcal{L}_\text{DriveZero} = \lambda_\text{traj}\mathcal{L}_\text{traj} + \lambda_\text{score}\mathcal{L}_\text{score}$
4. 学生不直接接触特权信息，仅通过蒸馏轨迹学习

## 实验结果

在 NAVSIM v1/v2 和 HUGSIM 三个 Benchmark 上评测：

**NAVSIM v2（代表性结果，EPDMS 指标）**：

| 方法 | 使用人类数据 | EPDMS↑ |
|------|-----------|--------|
| PDM-Closed | ✓ | 56.6 |
| LTF | ✓ | 25.1 |
| GuideFlow | ✓ | 51.5 |
| SimScale | ✓ | 53.2 |
| DrivoR-Scale | ✓ | 54.6 |
| ZTRS | ✗ | 48.1 |
| GigaPixel | ✗ | 50.1 |
| **DriveRL** (特权) | ✗ | 55.6 |
| **DriveZero** | ✗ | 51.5 |
| **DriveZero-Scale** | ✗ | **57.1** |

DriveZero-Scale **不使用任何人类驾驶数据**，超越了所有依赖人类示范的方法（57.1 vs. 最佳 56.6）。

![DriveZero vs 人类驾驶对比](https://arxiv.org/html/2609.06055v1/drivezero_vs_human.png)
*图：DriveZero 与人类驾驶行为的直接对比，展示模型在超越人类示范方面的具体表现。*

![DriveZero vs SOTA 在 NAVSIM 上](https://arxiv.org/html/2609.06055v1/drivezero_vs_sota_navsim.png)
*图：DriveZero 与各 SOTA 方法在 NAVSIM benchmark 上的详细对比结果。*

![DriveZero vs SOTA 在 HUGSIM 上](https://arxiv.org/html/2609.06055v1/drivezero_vs_sota_hugsim_0.png)
*图：DriveZero 与 SOTA 方法在 HUGSIM benchmark 上的对比结果（第一部分）。*

## 总结

DriveZero 为自动驾驶提供了一个重要的范式转变：**从"模仿人类"到"超越人类"**。

**主要贡献**：
- 提出感知-行为解耦的预训练框架，各自使用最优学习范式
- DriveRL 用纯 RL 从零学习，无任何人类驾驶预训练
- DriveVFM 无标注融合多个视觉基础模型
- 首次实现不使用人类示范却超越人类示范的端到端驾驶系统

**局限性**：DriveZero 目前主要在仿真 Benchmark 上验证，真实道路部署还需应对更复杂的长尾场景；DriveRL 的闭环训练需要大规模 GPU 集群（96 GPU）；DriveVFM 仅使用摄像头，在极端天气和夜间场景的鲁棒性有待测试。
