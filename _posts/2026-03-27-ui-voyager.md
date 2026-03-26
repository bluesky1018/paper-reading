---
layout: post
title: "UI-Voyager：通过失败经验自主进化的移动端GUI智能体"
date: 2026-03-27
categories: [论文解读, GUI Agent]
tags: [GUI Agent, Self-Evolving, Rejection Fine-Tuning, Self-Distillation, AndroidWorld, MLLM, Mobile Automation]
---

> **论文**：UI-Voyager: A Self-Evolving GUI Agent Learning via Failed Experience
> **arXiv**：[2603.24533](https://arxiv.org/abs/2603.24533)
> **机构**：Tencent Hunyuan
> **作者**：Zichuan Lin, Feiyu Liu, Yijun Yang, Jiafei Lyu, Yiming Gao, Yicheng Liu, Zhicong Lu, Yangbin Yu, Mingyu Yang, Junyou Li, Deheng Ye, Jie Jiang
> **代码**：[https://github.com/ui-voyager/UI-Voyager](https://github.com/ui-voyager/UI-Voyager)

## 一句话总结

UI-Voyager 提出了两阶段自进化框架（拒绝精调 RFT + 组相对自蒸馏 GRSD），使一个仅 4B 参数的移动端 GUI 智能体在 AndroidWorld 基准上取得 81.0% 的任务成功率，首次超越人类水平（80.0%），同时无需任何人工数据标注。

---

## 背景与问题

随着多模态大语言模型（MLLM）的快速发展，能够自主操作手机界面的 GUI 智能体受到越来越多的关注。然而，现有方法在两个核心问题上仍面临挑战：

**1. 失败轨迹的低效利用**

在长时程 GUI 任务中，智能体的探索结果大多以失败告终。传统的拒绝采样微调（Rejection Fine-Tuning）直接丢弃所有失败轨迹，只保留成功轨迹用于训练。这导致大量包含有价值信息的失败经验被浪费，训练效率极低。

**2. 稀疏奖励下的模糊信用分配**

强化学习方法（如 GRPO、PPO）通常依赖轨迹级别的稀疏奖励（任务完成=1，失败=0）。对于一条包含 30 步操作的轨迹，若因第 5 步的单次错误导致失败，那么前 29 步正确操作的"功劳"将被错误地归零。这种信用分配的模糊性严重阻碍了模型从失败中有效学习。

![性能对比图](https://arxiv.org/html/2603.24533v1/x1.png)
*图1：UI-Voyager（4B）在 AndroidWorld 上取得 81.0% 成功率，超越所有更大规模的基线模型和人类水平*

---

## 核心方法

UI-Voyager 提出了一套两阶段的自进化框架：

### 任务形式化

将 GUI 交互建模为部分可观测马尔可夫决策过程（POMDP），动作空间包含 11 种原子操作：

| 动作 | 描述 |
|------|------|
| `click(x,y)` | 点击坐标 |
| `long_press(x,y)` | 长按坐标 |
| `swipe(x,y,x',y')` | 滑动 |
| `open_app(app_name)` | 打开应用 |
| `input_text(text)` | 输入文字 |
| `keyboard_enter()` | 按下回车 |
| `navigate_back()` | 返回键 |
| `navigate_home()` | Home 键 |
| `wait()` | 等待 |
| `status(goal_status)` | 终止任务 |
| `answer(text)` | 返回最终答案 |

奖励由基于 Android Debug Bridge (adb) 的规则验证器自动判定。

![训练流程总览](https://arxiv.org/html/2603.24533v1/x2.png)
*图2：UI-Voyager 完整训练流程——第一阶段 RFT 迭代自进化，第二阶段 GRSD 从失败轨迹中提取步级监督信号*

---

### 第一阶段：拒绝精调（Rejection Fine-Tuning, RFT）

**核心思路：** 构建一个全自动的"数据-模型"共同进化循环。

**具体流程：**

1. **种子任务生成**：通过扰动模板参数（时间约束、数量、文件实体等）自动合成大量新任务，避免人工标注。
2. **轨迹采集**：让当前模型在真实 Android 环境中执行任务，自动收集交互轨迹。
3. **拒绝采样**：只保留通过验证器的成功轨迹，用于监督微调（SFT）。
4. **迭代更新**：用训练好的新模型生成下一轮轨迹（每轮使用全新任务防止过拟合）。

**实验结果：** 经过 4 轮 RFT 迭代，Pass@1 从初始的约 37% 稳步提升至 73.2%，同时 Pass@K 指标在所有迭代轮次中持续改善。

---

### 第二阶段：组相对自蒸馏（Group Relative Self-Distillation, GRSD）

这是 UI-Voyager 最核心的创新。其关键洞察在于：**对同一任务进行组采样时，成功轨迹与失败轨迹往往经历相同的屏幕状态，但在某个"分叉点"（Fork Point）选择了不同的动作。** 成功轨迹在分叉点的动作，正是对失败轨迹最精准的纠正信号。

![分叉点检测示意图](https://arxiv.org/html/2603.24533v1/x3.png)
*图3：分叉点检测原理——成功轨迹 τ⁺ 与失败轨迹 τ⁻ 在相同状态（蓝框）发生分叉，GRSD 提取成功动作作为失败步骤的教师信号*

#### 3.1 分叉点检测算法

**跨轨迹状态匹配：** 使用结构相似性指数（SSIM）判断两条轨迹的屏幕状态是否等价，预处理包括裁剪-缩放-灰度化，并用均值哈希（Mean-hash）做快速预过滤（相似度 < 0.80 的配对直接跳过）。

**对齐与分叉判断：**
- 若 τ⁺ 和 τ⁻ 在步骤 i 和 j 处状态相同，且下一状态也相同 → 轨迹在此处对齐（非分叉）
- 若状态相同但下一状态不同 → 发现分叉点，用 τ⁺ 在此处的动作作为 τ⁻ 在此处的教师

**单调性约束：** 后续失败步骤 j' > j 的匹配点必须满足 i ≥ i*(j)，确保时序一致性。

**完整算法伪代码：**
```
输入：成功轨迹 τ⁺，失败轨迹 τ⁻，阈值 θ
M ← ∅, i_min ← 0
for j = 0 to T⁻:
  if 转换对齐(i, j): i_min ← i+1; continue
  C(j) ← {i ≥ i_min | 状态等价(o_i⁺, o_j⁻) 且 状态分叉(i,j)}
  if C(j) = ∅: continue
  i*(j) ← 按 SSIM 最高、步骤最早排序选取
  M ← M ∪ {(j, i*(j))}; i_min ← i*(j)
返回 M
```

#### 3.2 步级自蒸馏训练

对每个分叉点 (j, i*(j))，构建训练样本：

```
x_j^train = [失败轨迹的上下文 prompt_j⁻ | 成功步骤的正确动作 response_{i*(j)}⁺]
```

训练目标为标准的自回归损失，仅对 response 部分的 token 计算梯度，**完全替代 GRPO/PPO** 作为第二阶段的唯一训练目标。

---

## 实验结果

### 主实验（AndroidWorld，116 个任务）

| 模型 | 参数量 | 成功率 |
|------|--------|--------|
| Qwen3-VL-2B | 2B | 36.4% |
| MAI-UI-2B | 2B | 49.1% |
| Ferret-UI Lite-3B | 3B | 28.0% |
| Qwen3-VL-4B（基础） | 4B | 45.3% |
| Step-GUI-4B | 4B | 63.9% |
| UI-Tars-7B | 7B | 33.0% |
| GUI-Owl-7B | 7B | 66.4% |
| Step-GUI-8B | 8B | 67.7% |
| MAI-UI-8B | 8B | 70.7% |
| GUI-Owl-1.5-8B-Thinking | 8B | 71.6% |
| UI-Venus-1.5-30B-A3B | 30B | 77.6% |
| MAI-UI-32B | 32B | 73.3% |
| UI-Tars-2 | 230B | 73.3% |
| MAI-UI-235B-A22B | 235B | 76.7% |
| Gemini-2.5-Pro | — | 69.7% |
| **人类水平** | — | **80.0%** |
| **UI-Voyager（本文）** | **4B** | **81.0%** |

UI-Voyager 使用仅 4B 参数的模型，超越了所有更大规模的开源和闭源基线，并首次超越人类水平。

### 消融分析

**RFT 迭代效果：**

![RFT迭代曲线](https://arxiv.org/html/2603.24533v1/figs/Qwen3-VL-4B-Instruct_pass_k_vs_k.png)
*图4（左）：4 轮 RFT 迭代中 Pass@K 的持续提升*

传统 RL 方法（GRPO/PPO）从基础模型出发，需要约 175 步才能达到 64.0%（相当于一轮 RFT 的效果），收敛极慢，验证了 RFT 作为"热启动"的必要性。

**GRSD vs. RL 方法对比：**

| 方法 | 起始点（RFT 后） | 最终成功率 |
|------|-----------------|------------|
| GRPO | 73.2% | ~76%（plateau） |
| PPO | 73.2% | ~76%（plateau） |
| **GRSD（本文）** | **73.2%** | **81.0%** |

![GRSD训练曲线](https://arxiv.org/html/2603.24533v1/figs/GRSD_iters.png)
*图8：GRSD 训练曲线稳定提升至 81%，而 GRPO/PPO 在约 76% 处出现平台期*

**分叉点检测案例——BrowserMaze（迷宫导航）：**

![BrowserMaze分叉点](https://arxiv.org/html/2603.24533v1/x5.png)
*图5：第 12 步，两条轨迹处于相同屏幕状态。失败轨迹选择"向右"（撞墙），成功轨迹选择"向下"。GRSD 检测到分叉点并将正确动作作为监督信号*

失败轨迹的错误推理：*"要到达右下角，我需要继续向右移动..."*
成功轨迹的正确推理：*"下一步应该向下移动以接近右下角..."*

**低成功率任务（10 个）的精细分析：**

![低成功率任务对比](https://arxiv.org/html/2603.24533v1/x9.png)
*图9：在成功样本极稀少的最难任务上，GRSD 的性能提升显著优于 PPO 和 GRPO*

---

## 方法局限性与未来方向

作者在讨论中坦诚地指出了当前方法的几个局限：

1. **SSIM 匹配的不精确性**：异步帧捕获可能导致时间对齐偏差，瞬态 UI 元素（光标闪烁、Toast 提示、时钟跳秒）会干扰状态等价判断。未来可结合 OCR、布局 token 或无障碍树（Accessibility Tree）提升匹配精度。

2. **预定义动作空间的局限**：当前的原子动作抽象了手势的细节（持续时间、轨迹、时序）。未来可探索分层动作建模——高层用于任务规划，低层处理精细手势。

3. **扩展到更多 GUI 领域**：目前方法主要在 AndroidWorld（移动端）验证，未来计划扩展至桌面端和网页端。

---

## 总结

UI-Voyager 的核心贡献在于：

1. **重新定义了"失败经验"的价值**：不再简单丢弃失败轨迹，而是通过组相对自蒸馏从中提取精准的步级监督信号。

2. **优雅解决了稀疏奖励下的信用分配难题**：分叉点检测精准定位"成功与失败的分岔口"，为错误步骤提供有针对性的纠正，而非对整条轨迹一概而论。

3. **极致的参数效率**：4B 模型超越 235B 闭源模型，且无需任何人工标注数据，完全依靠自动合成任务和真实环境反馈进行自进化。

这项工作为构建高效、自主进化的 GUI 智能体提供了一条清晰可行的技术路径，也为 RL 训练中的"失败经验利用"问题提供了新的解题思路。
