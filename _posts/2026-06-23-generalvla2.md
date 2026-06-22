---
layout: post
title: "GeneralVLA-2：几何感知重建与治理型记忆的机器人规划框架"
date: 2026-06-23
categories: [论文解读, 机器人学习]
tags: [视觉语言动作模型, 3D重建, 记忆管理, 机器人规划, VLA]
---

> 📄 **论文**：GeneralVLA-2: Geometry-Aware Reconstruction and Governed Memory for Robot Planning
> 🔗 **arXiv**：[2606.17480](https://arxiv.org/abs/2606.17480)
> 🏢 **机构**：北京大学（Boxin Shi、Hao Tang 团队）/ AIGeeksGroup

## 一句话总结

GeneralVLA-2 通过 GeoFuse-MV3D 几何融合重建和治理型 KnowledgeBank 记忆系统，分别解决了 VLA 系统中 3D 重建不准确和记忆管理质量差两大瓶颈，在机器人仿真和真实实验中均显著超越现有方法。

## 背景与问题

通用视觉-语言-动作（VLA）系统面临两个核心技术瓶颈：

**问题 1：3D 重建不准确**
单目 SAM3D 式重建在场景几何理解上存在位姿幻觉——模型基于单视角生成的 3D 结构往往在相机位姿和物体几何上产生明显错误，导致机器人的抓取和操作规划失效。

**问题 2：记忆管理质量差**
现有 KnowledgeBank 难以控制记忆冲突与置信度。当新经验与旧记忆矛盾时，系统缺乏有效的冲突检测和解决机制；记忆的可信度也缺乏量化评估，导致低质量记忆污染后续决策。

## 核心方法

### GeoFuse-MV3D：几何感知多视角重建

GeoFuse-MV3D 以 5 个校准视图作为输入，通过两路几何融合解决单视角幻觉问题：

![GeoFuse-MV3D 重建架构](https://arxiv.org/html/2606.17480/x2.png)
*图：GeoFuse-MV3D 重建分支架构，展示双路几何融合策略*

- **Source A（几何先验路）**：基于 MV-SAM3D 输出，集成 VGGT 外部几何先验，加轻量外观仿射校准；
- **Source B（轴向补偿路）**：无外部提供者，仅使用输入掩码和重建结果进行轴向偏差校正。

**掩码一致性得分**用于评估每个 3D 点在所有视图中的可见性支持度：
$$s(p) = \frac{1}{\max(|\mathcal{V}(p)|,1)}\sum_{i\in\mathcal{V}(p)}M_i(\pi_i(p))$$

低支持点通过有界收缩向中心点压缩，轴向校正进一步修正坐标偏差。

**最终融合（仅融合几何坐标，保留外观属性）**：
$$G_{\text{out}} = \{((1-\alpha)x_A^j + \alpha x_B^j,\ \theta_A^j)\}_{j=1}^N$$

### 治理型 KnowledgeBank：结构化记忆管理

每条记忆记录包含完整的元数据：$m = (q, c, y, z, \kappa, R, \mathcal{L}, v)$，分别对应查询、内容、类型、生命周期状态、置信度、质量分、冲突链接、验证元数据。

![治理型 KnowledgeBank 架构](https://arxiv.org/html/2606.17480/x3.png)
*图：治理型 KnowledgeBank 模块，展示记忆的存储、检索和验证机制*

**精准检索评分**综合考虑文本相似度、置信度、成功率奖励、时效性、使用频率，以及冲突和陈旧惩罚：
$$S(q_t, X_t, m) = r_{\text{text}} + \kappa_m + b_{\text{success}} + b_{\text{recency}} + b_{\text{usage}} - p_{\text{conflict}} - p_{\text{stale}}$$

**验证质量评分**通过多个评判标准对记忆的可靠性进行量化：
$$R_t = \frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}}\sum_{v\in\mathcal{V}} p_\theta(v|q_t, X_t, \tau_t, c)\cdot\phi(v)$$

## 实验结果

**GSO-30 重建评估（Table 3）：**

| 方法 | CD↓(×10⁻³) | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|:---:|
| MV-SAM3D 基线 | 45.8876 | 13.2421 | 0.8051 | 0.2795 |
| **GeoFuse-MV3D** | **44.8770** (-2.20%) | **13.5547** (+2.36%) | **0.8134** (+1.03%) | **0.2739** (-2.02%) |

**KnowledgeBank 基准评估（Table 5）：**

| 模型 | 方法 | Terminal-Bench SR↑ | SWE-Bench RR↑ |
|------|------|:---:|:---:|
| Qwen-3.5-Flash | ReasoningBank | 52.8±2.0 | 70.8±1.5 |
| Qwen-3.5-Flash | **KnowledgeBank** | **55.8±1.8** (+4.53%) | **73.4±1.2** (+3.73%) |
| Gemini-3.1-Pro | ReasoningBank | 73.0±1.0 | 82.2±1.6 |
| Gemini-3.1-Pro | **KnowledgeBank** | **75.7±1.3** | **85.3±1.2** |

**RLBench 仿真任务成功率（Table 1）：**

| 方法 | Put_block | Play_jenga | Open_jar | Close_box | Pickup_cup |
|------|:---:|:---:|:---:|:---:|:---:|
| VoxPoser | 70.70 | 0.00 | 0.00 | 0.00 | 26.70 |
| CAP | 84.00 | 0.00 | 0.00 | 0.00 | 14.67 |
| Hamster | 78.33 | 0.00 | 77.67 | 0.00 | 9.00 |
| **GeneralVLA-2** | **90.33** | **85.33** | **85.00** | **54.67** | **87.33** |

特别值得注意的是 Play_jenga（85.33% vs 基线 0%）和 Close_box（54.67% vs 基线 0%），这两个任务需要精确的 3D 几何理解。

**真实机器人实验（4 个任务）：**

| 方法 | Move_spray | Open_drawer | Open_jar | Sort_object |
|------|:---:|:---:|:---:|:---:|
| CAP (0-shot) | 6.67% | 0.00% | 36.67% | 70.00% |
| **GeneralVLA-2** | **63.33%** | **40.00%** | **53.33%** | **83.33%** |

![真实机器人任务执行演示](https://arxiv.org/html/2606.17480/x4.png)
*图：GeneralVLA-2 在真实机器人上执行四类操作任务的演示*

## 总结

GeneralVLA-2 系统性地解决了通用 VLA 的两大核心瓶颈：GeoFuse-MV3D 通过双路几何融合消除了单视角重建的位姿幻觉，治理型 KnowledgeBank 通过结构化元数据管理提升了记忆利用的可靠性。在仿真和真实机器人实验中均取得显著改进，尤其在需要精确几何理解的任务（Play_jenga、Close_box）上从 0% 提升到 50%+ 的成功率。

局限性方面，GeoFuse-MV3D 要求 5 个校准视图，在单目或视角受限的场景中需要额外适配。KnowledgeBank 的记忆质量评分依赖 LLM 验证，在低资源或实时部署场景中存在延迟问题。
