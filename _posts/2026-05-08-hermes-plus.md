---
layout: post
title: "HERMES++：统一3D场景理解与未来几何预测的驾驶世界模型"
date: 2026-05-08
categories: [论文解读, 自动驾驶]
tags: [自动驾驶, 世界模型, 3D感知, BEV, 多模态大模型, 场景生成]
---

> 📄 **论文**：HERMES++: Toward a Unified Driving World Model for 3D Scene Understanding and Generation
> 🔗 **arXiv**：[2604.28196](https://arxiv.org/abs/2604.28196)
> 🏢 **机构**：华中科技大学（HUST）、香港大学（HKU）、Mach Drive

## 一句话总结

HERMES++ 在单一框架内统一了驾驶场景的语义理解（VQA）与未来几何演化预测（3D点云生成），通过联合几何优化策略（显式点云约束+隐式潜在空间正则化）和动态记忆汇聚机制，在无需任何辅助监督（3D检测框/车道线）的情况下同时超越纯生成方法和纯理解方法的专项指标。

## 背景与问题

自动驾驶系统需要同时具备两种关键能力，但现有方法在这两者之间存在严重割裂：

**驾驶世界模型方向**（ViDAR, OccWorld, DriveX 等）：
- 专注未来场景的几何生成（2D视频或3D点云预测）
- **缺陷**：无语义理解能力，"often fail to articulate the semantic context"，无法回答直接查询

**VLM/LLM 驾驶方向**（OmniDrive, DriveLM 等）：
- 擅长当前场景的语义理解（VQA、场景描述、规划推理）
- **缺陷**："lacking the predictive capacity to anticipate how the scene geometry will evolve"，无法预测危险场景的几何演化

**核心 Gap**：理解当前场景"是什么"与预测未来场景"会变成什么"之间的断层，使得现有系统在安全关键场景下缺乏前瞻预判能力。

HERMES++ 提出的核心假设是：语义理解与几何预测之间存在**正向协同**——语义知识可以引导更合理的未来几何预测，而几何约束则为语言推理提供物理基础。

## 核心方法

### 整体架构

HERMES++ 的完整处理流程：

```
多视角图像 → BEV Tokenizer → BEV Tokens
                                  ↓
              [BEV tokens + 文本指令 + 世界查询] → LLM
                                  ↓                 ↓
                            文本回答          增强的世界查询
                                                     ↓
                                         Current-to-Future Link
                                                     ↓
                                         未来BEV特征 {B_{t+i}}
                                                     ↓
                                         BEV-to-Point Render
                                                     ↓
                                         未来点云 {P_{t+i}}
```

![HERMES++完整架构图](https://arxiv.org/html/2604.28196v1/x2.png)
*图：HERMES++的完整pipeline架构，展示BEV Tokenizer、LLM主干、Current-to-Future Link和BEV-to-Point Render的协同工作。*

### BEV Tokenizer 与 BEV-to-Point Render

**BEV Tokenizer（两阶段）**：
1. OpenCLIP ConvNeXt-L 提取多尺度透视特征
2. BEVFormerV2 将多视角特征聚合到 BEV 空间：

$$\mathbf{B}(x,y)=\sum_{i=1}^{N}\sum_{z\in\mathcal{H}}\text{DeformAttn}\left(\mathbf{Q}(x,y),\mathbf{F}_{i},\mathcal{P}_{i}(x,y,z)\right)$$

步幅卷积下采样×4（180×180 → 45×45 BEV 网格），线性投影为 LLM 可处理的 token 序列。

**BEV-to-Point Render**：基于 SDF 隐式场建模场景几何，通过体积渲染计算深度并转换回3D点云坐标，实现可微分的几何重建。

### LLM 知识迁移：世界查询机制

HERMES++ 的关键创新是通过**世界查询（World Queries）**实现 LLM 预训练知识向场景预测的迁移。

世界查询初始化（对压缩 BEV 特征做自适应最大池化）：

$$\mathbf{Q}^{w}=\phi\left(\text{Concat}_{i=1}^{\Delta t}(\mathbf{Q}\oplus\mathbf{e}_{t+i})\oplus\mathbf{FE}\right)$$

其中 $\mathbf{e}_{t+i}$ 为 ego-motion 嵌入，$\mathbf{FE}$ 为可学习帧嵌入。

LLM 处理时，世界查询通过因果注意力机制同时聚合 BEV 空间信息和文本上下文，实现 LLM 的物理世界知识注入。默认配置：n=4个查询/帧，预测未来Δt=3秒。

### Current-to-Future Link

这一模块负责将 LLM 输出的世界查询转化为未来 BEV 特征序列。

**Textual Injection**（文本驱动的预测引导）：
对 LLM 处理后的文本 tokens 做平均池化，提取文本语义嵌入 $\hat{\mathbf{T}}$，与世界查询联合作为交叉注意力的 Key/Value：

$$\mathbf{X}^{(l)}_{\text{cross}}=\mathbf{X}^{(l)}+\text{CrossAttn}(\text{LN}(\mathbf{X}^{(l)}),[\mathbf{Q}^{w}_{\epsilon,i};\hat{\mathbf{T}}])$$

这使得语言描述（如"前方有行人快速穿越"）能直接影响未来几何预测的走向。

**Ego Modulation（EM）**（自车运动感知）：

$$\text{EM}(\mathbf{x})=(\gamma+1)\odot\text{LN}(\mathbf{x})+\beta$$

MLP+Tanh 将 ego 运动参数编码为仿射变换的 γ 和 β，有效解耦相机运动与场景内在动态。

### 联合几何优化策略

这是 HERMES++ 超越所有对比方法的核心技术。

**显式几何约束**（点云级，L1损失）：

$$\mathcal{L}_{\text{render}}=\sum_{i=0}^{\Delta t}\lambda_i\frac{1}{N_i}\sum_{k=1}^{N_i}|d(\mathbf{r}_k)-\tilde{d}(\mathbf{r}_k)|$$

帧权重 $\lambda_i = 1 + 0.5\times i$，特别强调长期预测的准确性。

**隐式几何正则化**（潜在空间级）：

通过预训练（后冻结）的几何特征提取器将体积表示映射到几何语义空间，利用两种损失约束潜在表示的几何一致性：

- **Cosine 相似度损失**（体素级逐点一致性）：
$$\mathcal{L}_{\text{cos}}=1-\frac{1}{whz}\sum_{i,j,k}\frac{\hat{\mathbf{V}}_t(i,j,k)\cdot\mathbf{V}_t(i,j,k)}{\|\hat{\mathbf{V}}_t(i,j,k)\|_2\|\mathbf{V}_t(i,j,k)\|_2}$$

- **Gram 损失**（全局结构统计一致性，沿三个正交轴投影）：
$$\mathcal{L}_{\text{gram}}=\frac{1}{3}\sum_d\|\mathbf{G}^d_t-\hat{\mathbf{G}}^d_t\|_F^2$$

显式约束从点云几何出发，隐式约束从潜在流形出发，两者互补消除深度歧义和射线投影伪影。

总损失：$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{lang}}+10\mathcal{L}_{\text{render}}+\mathcal{L}_{\text{cos}}+\mathcal{L}_{\text{gram}}$

## 实验结果

### 主要性能对比（OmniDrive-nuScenes 验证集）

| 方法 | 类型 | LLM参数 | 辅助监督 | 0s↓ | 1s↓ | 2s↓ | 3s↓ | CIDEr↑ |
|------|------|---------|---------|-----|-----|-----|-----|--------|
| ViDAR | 仅生成 | - | - | - | 1.12 | 1.38 | 1.73 | - |
| DriveX | 仅生成 | - | - | - | 0.66 | 0.86 | 1.10 | - |
| Omni-L | 仅理解 | 7B | 3D框+车道线 | - | - | - | - | 0.732 |
| **Hermes（会议版）** | 统一 | 1.8B | **无** | 0.59 | 0.78 | 0.95 | 1.17 | 0.741 |
| **Hermes++ (1.8B)** | 统一 | 1.8B | **无** | 0.53 | 0.71 | 0.86 | 1.01 | 0.749 |
| **Hermes++ (3.8B)** | 统一 | 3.8B | **无** | 0.51 | 0.68 | 0.82 | **0.97** | **0.772** |

**关键发现**：
- Hermes++ 在**无任何辅助监督**的情况下，生成指标（3s CD）超越需要3D检测框+车道线监督的纯理解方法
- 相比会议版 Hermes，3秒 Chamfer Distance 降低 13.7%（1.17→1.01）
- 相比最强纯生成方法 DriveX，3s CD 降低 0.13

![定性结果](https://arxiv.org/html/2604.28196v1/x3.png)
*图：HERMES++的定性结果展示——左列为场景理解（VQA回答），右列为未来点云预测的可视化，两者在同一框架内协同完成。*

### 联合几何优化消融

| $\mathcal{L}_{\text{cos}}$ | $\mathcal{L}_{\text{gram}}$ | 3s CD↓ | CIDEr↑ |
|------|------|--------|--------|
| ✗ | ✗ | 1.637 | 0.722 |
| ✓ | ✗ | 1.441 | 0.717 |
| ✗ | ✓ | 1.544 | 0.717 |
| ✓ | ✓ | **1.436** | **0.720** |

两种隐式约束的组合带来最优结果，且相比无隐式约束（1.637）降低了12.3%。

### Current-to-Future Link 消融

| 模块 | 3s CD↓ | CIDEr↑ |
|------|--------|--------|
| 无 Link | 2.377 | 0.433 |
| 简单 Link | 1.542 | 0.718 |
| + Textual Injection | 1.506 | 0.717 |
| + Ego Modulation | 1.442 | 0.711 |
| + 更深网络（6层）| **1.436** | **0.720** |

Current-to-Future Link 本身对生成质量贡献最大（从2.377到1.542），Textual Injection 和 Ego Modulation 进一步精炼预测。

![特征可视化](https://arxiv.org/html/2604.28196v1/x5.png)
*图：仅使用显式约束（上）与使用联合几何优化（下）的内部体积特征对比，可见联合优化后特征分布更加清晰、几何结构更加精确。*

### NuScenes-QA 结果

| 方法 | 输入模态 | 准确率↑ |
|------|---------|--------|
| BEVDet+MCAN | 摄像头 | 57.9% |
| CenterPoint+MCAN | 激光雷达 | 59.5% |
| Omni-Q | 摄像头 | 59.2% |
| **Hermes++** | **摄像头** | **61.3%** |

## 总结

HERMES++ 证明了在单一框架内统一驾驶场景"理解"与"预测"的可行性，且两个任务之间存在正向协同——语义理解引导更合理的几何预测，几何约束为语言推理提供物理基础。无需任何辅助监督信号（如3D检测框、车道线标注），HERMES++ 在生成和理解两个维度上同时超越各自领域的专家方法。

**核心贡献**：
1. 首个统一3D场景理解（VQA）与未来几何预测（点云生成）的驾驶世界模型
2. 联合几何优化策略：显式点云约束+隐式潜在流形正则化的互补双重约束
3. World Queries 机制：实现 LLM 预训练知识从语言理解到物理几何预测的迁移

**局限性**：仅在 NuScenes 数据集上验证，实际开放道路场景的泛化性有待检验；BEV 表示对极端天气条件的鲁棒性需要进一步研究；纯摄像头方案在尺度估计精度上仍有局限。
