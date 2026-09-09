---
layout: post
title: "Mask Forcing：双噪声掩码 Rollout 改善自回归视频扩散蒸馏"
date: 2026-09-10
categories: [论文解读, 视频生成]
tags: [视频生成, 扩散模型, 自回归, 知识蒸馏, 模式崩塌]
---

> 📄 **论文**：Mask Forcing: Improving Autoregressive Video Diffusion Distillation via Dual-Noise Masking Rollout
> 🔗 **arXiv**：[2609.09123](https://arxiv.org/abs/2609.09123)
> 🏢 **机构**：HKUST(GZ) / HKUST / LIGHTSPEED / UCSD / CUHK(SZ) / NUS

## 一句话总结

Mask Forcing 通过在自回归（AR）视频扩散蒸馏的 self-rollout 过程中沿空间和时间轴注入随机掩码干扰，有效缓解了反向 KL 目标（DMD）导致的模式崩塌问题，显著提升视频质量和真实感。

## 背景与问题

自回归（AR）视频扩散模型利用因果注意力机制逐块生成视频帧，天然支持实时推理和无限长视频生成，是当前视频生成的重要方向。然而，从预训练双向视频扩散教师蒸馏 AR 学生的主流方法——**Distribution Matching Distillation（DMD）**——存在严重问题：

**生成的视频过度饱和（over-saturation）+ 过度平滑（over-smoothing）**，视觉质量和真实感受限。

根本原因是 DMD 采用的**反向 KL（Reverse KL）目标的模式搜寻（mode-seeking）行为**：

$$D_{KL}(p_\text{fake} \| p_\text{real}) = \mathbb{E}\left[\log\frac{p_\text{fake}(x_t)}{p_\text{real}(x_t)}\right]$$

反向 KL 鼓励学生分布覆盖它自信的区域，而不是覆盖教师分布的所有模式。在 self-rollout 训练中，一旦学生的 rollout 轨迹集中在少数几个模式上，DMD 梯度就无法将学生推向其他模式——形成恶性循环，最终导致模式崩塌。

## 核心方法

### Mask Forcing：双噪声掩码 Rollout

![Mask Forcing 框架示意](https://arxiv.org/html/2609.09123v1/pipeline.png)
*图：Mask Forcing 的训练框架。(a) 标准 self-rollout DMD 训练因反向 KL 模式搜寻行为导致模式崩塌，产生过度饱和；(b) Mask Forcing 通过双噪声掩码 rollout 扰动学生轨迹，鼓励更广泛的教师模式覆盖，同时提供更干净的去噪上下文。*

**核心思想**：在 AR 视频扩散蒸馏的 self-rollout 过程中，沿**空间轴**和**时间轴**注入随机掩码，将更干净（低噪声）的信号混入噪声 rollout 输入中。

这一简单操作带来双重效益：

1. **扩大教师分布探索范围**：被干扰的 rollout 轨迹不再只访问学生已覆盖的少数模式，而是被"推向"更广泛的教师高密度区域，为 DMD 提供超出已覆盖模式的学习信号

2. **改善中间去噪预测**：更干净的 token 为其他噪声 token 提供去噪引导，类似掩码建模（Masked Modeling）的原理，减少错误积累

### 技术实现

视频 AR 生成：每个 chunk $x_i$ 的 flow-matching 加噪公式为：
$$x_i^t = (1-t)x_i^0 + t\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

在 self-rollout 的每个 chunk 中，Mask Forcing 随机选择部分 token，将其替换为具有更低噪声水平的采样（时间步在 $[t_\min, t]$ 范围内随机采样，其中 $t_\min < t$）：

- **空间掩码**：在 chunk 内帧之间随机掩码
- **时间掩码**：在 chunk 之间随机掩码

**关键超参数**：
- 掩码比例 $\alpha = 0.2$（20% token 被"净化"）
- 时间步窗口 $\Delta = 250$
- 最低时间步 $t_\min$ 对应 schedule 第 20 步

整个蒸馏训练仅需约 **1,500 步、14 小时（8 GPU）**。

## 实验结果

**基于 Wan2.1-T2V-1.3B（学生）和 Wan2.1-T2V-14B（教师）**，在 100-prompt 集（丰富运动场景）和 VBench 上评测：

**Chunk-wise AR 设置（3帧/chunk）**：

| 方法 | HPSv3↑ | Vision.↑ | Instruct.↑ | MQ↑ | Dynamic.↑ | Total↑ | Quality↑ | Semantic↑ |
|------|--------|----------|------------|-----|----------|-------|---------|---------|
| Self Forcing | 9.55 | 10.10 | 38.50 | 15.88 | 70 | 81.89 | 82.99 | 77.49 |
| **+ Ours** | **9.84 (+.29)** | **11.37 (+1.27)** | **45.03 (+6.53)** | **20.49 (+4.61)** | **82 (+12)** | **82.61 (+.72)** | **83.68 (+.69)** | **78.74 (+1.25)** |
| Causal Forcing | 9.37 | 10.36 | 40.41 | 17.73 | 76 | 82.67 | 83.58 | 78.98 |
| **+ Ours** | **10.17 (+.80)** | **11.58 (+1.22)** | **46.30 (+5.89)** | **21.54 (+3.81)** | 82 (+6) | **82.76 (+.09)** | **83.69 (+.11)** | **79.01 (+.03)** |
| LongLive | 9.11 | 10.77 | 42.48 | 21.20 | 76 | 82.02 | 82.87 | 78.66 |
| **+ Ours** | **10.14 (+1.03)** | 11.00 (+.23) | 42.65 (+.17) | **22.34 (+1.14)** | 69 (-7) | **82.75 (+.73)** | **83.71 (+.84)** | **78.91 (+.25)** |

**Frame-wise AR 设置（逐帧）**：

| 方法 | HPSv3↑ | Instruct.↑ | MQ↑ | Dynamic.↑ | Total↑ |
|------|--------|------------|-----|----------|-------|
| Self Forcing | 9.34 | 35.62 | 18.27 | 53 | 80.73 |
| **+ Ours** | 9.79 | 39.60 | 19.07 | **61 (+8)** | 81.49 |
| Causal Forcing | 9.67 | 37.56 | 20.25 | 28 | 80.64 |
| **+ Ours** | 9.96 | 39.60 | **23.69 (+3.44)** | **52 (+24)** | **82.28 (+1.64)** |
| LongLive | 9.19 | 38.50 | 12.14 | 25 | 80.97 |
| **+ Ours** | 9.46 | 42.78 | **19.32 (+7.18)** | **76 (+51)** | 81.47 |

Mask Forcing 作为**即插即用（plug-and-play）**改进，在所有三个基线（Self Forcing、Causal Forcing、LongLive）上均带来一致提升，尤其在指令遵循（Instruct.）、运动质量（MQ）和动态性（Dynamic.）上改善最为显著。

### 定性效果

![定性对比](https://arxiv.org/html/2609.09123v1/qualitative_1.png)
*图：Mask Forcing 与各基线的定性对比，展示更高视觉质量和真实感，过度饱和伪影明显减少，高频细节更丰富。*

![LongLive 长视频生成](https://arxiv.org/html/2609.09123v1/longlive_long.png)
*图：Mask Forcing 应用于 LongLive 的长视频生成结果，展示在时序连贯性和细节保持上的改进。*

![世界模型应用](https://arxiv.org/html/2609.09123v1/wm.png)
*图：Mask Forcing 在世界模型（World Model）应用场景中的生成效果展示。*

## 总结

Mask Forcing 提供了一个简洁而有效的视角来理解和解决 AR 视频扩散蒸馏中的模式崩塌问题：**不修改 DMD 目标函数本身，而是通过改变 self-rollout 轨迹的采样方式来改善探索多样性**。

**主要贡献**：
- 从模式搜寻行为角度分析 AR 视频蒸馏中过度饱和/平滑问题的根本原因
- 提出双噪声掩码 Rollout（空间+时间双轴掩码），仅需引入随机掩码超参数
- 即插即用：在 Self Forcing、Causal Forcing、LongLive 上均有效
- 训练成本极低（1.5K 步，14 小时，8 GPU）

**局限性**：LongLive 在 chunk-wise 设置的动态性（Dynamic.）指标有所下降（-7），说明掩码干扰在某些基线上可能影响时序连贯性；掩码比例等超参数需要针对不同基线进行调优；对更大规模教师模型的适用性有待验证。
