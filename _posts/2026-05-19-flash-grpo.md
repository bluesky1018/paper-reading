---
layout: post
title: "Flash-GRPO：单步策略优化实现视频扩散模型的高效对齐训练"
date: 2026-05-19
categories: [论文解读, 视频生成]
tags: [视频扩散, GRPO, 强化学习对齐, 训练加速, 扩散模型]
---

> 📄 **论文**：Flash-GRPO: Efficient Alignment for Video Diffusion via One-Step Policy Optimization
> 🔗 **arXiv**：[2605.15980](https://arxiv.org/abs/2605.15980)
> 🏢 **机构**：多家机构合作

## 一句话总结

Flash-GRPO 通过等时分组和时序梯度修正两项关键技术，将视频扩散模型的 GRPO 对齐训练加速 6 倍，同时在低计算预算下超越完整轨迹训练的对齐质量。

## 背景与问题

GRPO（Group Relative Policy Optimization）近期成为视频扩散模型对齐训练的重要方法，但其计算成本极为高昂——训练一个 14B 参数的视频生成模型通常需要**数百 GPU 天**，这使得大多数研究团队难以承担。

现有的加速尝试（如滑动窗口子采样）虽然减少了时间步数，但带来了严重的**训练不稳定性**——奖励崩溃（reward collapse）和梯度震荡频发。问题的根源在于：

1. **时间步混淆方差**：同一 prompt 的不同 rollout 使用不同时间步，导致奖励估计方差过大
2. **跨时间步梯度不一致**：不同时间步上的梯度幅度差异悬殊，优化方向混乱

## 核心方法

### 技术一：等时分组（Iso-Temporal Grouping）

标准 GRPO 在计算每个 prompt 组的相对奖励时，同组内的 rollout 使用**不同时间步**进行评估，导致奖励差异既来自策略质量又来自时间步差异，产生方差混淆。

Flash-GRPO 要求同一 prompt 组内所有 rollout **共享相同时间步**（Iso-Temporal），彻底消除时间步混淆方差。组间通过分层采样保持时间步的多样性。

![Flash-GRPO框架示意图](https://arxiv.org/html/2605.15980/x2.png)
*图2：Flash-GRPO 框架——等时分组与时序梯度修正示意图*

### 技术二：时序梯度修正（Temporal Gradient Rectification）

通过理论推导，论文发现视频扩散中策略梯度包含一个**时间依赖的缩放因子**：

$$\lambda(t) = \frac{\sqrt{\Delta t}}{\sigma_t} + \frac{\sigma_t \sqrt{\Delta t}(1-t)}{2t}$$

不同时间步 $t$ 对应的 $\lambda(t)$ 差异可达**数量级**，导致梯度幅度极不平衡——某些时间步的梯度会"淹没"其他时间步的信号。

Flash-GRPO 在反向传播时除以对应的 $\lambda(t)$，将所有时间步的梯度归一化到统一量级，显著提升训练稳定性。

## 实验结果

### 主要性能对比（Wan 14B 模型，350 GPU小时预算）

| 方法 | GPU 小时 | Aesthetic↑ | Imaging↑ | Subject Consistency↑ |
|------|---------|-----------|---------|---------------------|
| Flow-GRPO-Fast1 | 350 | 65.92 | 65.96 | 98.46 |
| Flow-GRPO（完整） | **350** | 65.79 | 68.60 | 97.28 |
| **Flash-GRPO** | **350** | **66.43** | **68.28** | **98.70** |

Flash-GRPO 在相同计算预算下全面超越现有方法，且实现了相比完整轨迹训练 **6 倍**的训练加速。

### 奖励与质量指标

| 方法 | HPSv3 奖励↑ |
|------|-----------|
| Flow-GRPO | 5.1 |
| **Flash-GRPO** | **5.4** |

### 跨模型规模验证

实验在 **Wan 1.3B** 和 **Wan 14B** 两个规模上均验证了 Flash-GRPO 的有效性，表明方法具有良好的规模扩展性。

![性能总览对比](https://arxiv.org/html/2605.15980/x1.png)
*图1：Flash-GRPO 与现有方法的综合性能对比*

![定性对比](https://arxiv.org/html/2605.15980/x3.png)
*图3：Flash-GRPO 与基线方法在运动质量、美学和提示遵循上的定性对比*

![训练稳定性分析](https://arxiv.org/html/2605.15980/x8.png)
*图8：训练稳定性对比——Flash-GRPO 显著减少奖励崩溃现象*

## 总结

Flash-GRPO 通过两项精准的技术干预——消除时间步混淆方差和归一化跨时间步梯度——在不改变整体训练框架的前提下，同时解决了视频扩散 GRPO 训练的效率和稳定性问题。6 倍加速使得原本需要数百 GPU 天的训练成本下降至数十 GPU 天，大幅降低了视频生成模型对齐研究的门槛。

局限性在于当前实验主要基于 Wan 系列模型，不同视频扩散架构下的效果有待进一步验证。未来工作可探索将等时分组思想扩展至图像扩散对齐，以及与其他 RLHF 方法（如 DPO、PPO）的结合。
