---
layout: post
title: "【论文精读】ICRL：无需SFT冷启动，纯强化学习让LLM学会使用工具"
date: 2026-03-13
categories: [AI, LLM, ReinforcementLearning]
tags: [强化学习, 工具调用, In-Context Learning, RLVR, 搜索引擎, arXiv]
---

> 📄 **论文精读 · arXiv 2603.08068**
>
> **In-Context Reinforcement Learning for Tool Use in Large Language Models**
>
> Yaoqi Ye, Yiran Zhao, Keyu Duan, Zeyu Zheng, Kenji Kawaguchi, Cihang Xie, Michael Qizhe Shieh
>
> National University of Singapore · 2026年3月
>
> GitHub：[applese233/ICRL](https://github.com/applese233/ICRL)
>
> 标签：强化学习 · 工具调用 · In-Context Learning · 搜索引擎增强

---

## ⚡ 核心发现（TL;DR）

- 现有方法让 LLM 学会使用工具的流程是 **SFT（有监督微调）→ RL（强化学习）**，需要大量标注数据做冷启动
- 本文提出 **ICRL（In-Context Reinforcement Learning）**，完全**跳过 SFT**，只用 RL，在 rollout 阶段用 few-shot 示例"教"模型如何调用工具
- 随着训练推进，**逐步减少 in-context 示例数量**（3-shot → 2-shot → 0-shot），最终模型在零样本下就能自主使用工具
- 在推理和工具调用基准上达到 **SOTA 性能**，且数据效率显著优于传统 SFT+RL 流程

---

## ABSTRACT · 摘要

虽然大语言模型（LLMs）展现出强大的推理能力，但其在复杂任务上的表现往往受限于内部知识的局限性。一种引人注目的解决方案是为模型配备外部工具——例如用于数学计算的 Python 解释器，或用于检索事实信息的搜索引擎。然而，让模型有效使用这些工具仍然是一项重大挑战。

*While large language models (LLMs) exhibit strong reasoning abilities, their performance on complex tasks is often constrained by the limitations of their internal knowledge. A compelling approach to overcome this challenge is to augment these models with external tools -- such as Python interpreters for mathematical computations or search engines for retrieving factual information.*

现有方法通常依赖**冷启动流水线**：先进行有监督微调（SFT），再进行强化学习（RL）。这些方法需要大量的 SFT 标注数据，而这类数据的标注或合成成本极高。

*Existing methods typically rely on cold-start pipelines that begin with supervised fine-tuning (SFT), followed by reinforcement learning (RL). These approaches often require substantial amounts of labeled data for SFT, which is expensive to annotate or synthesize.*

本文提出 **In-Context Reinforcement Learning（ICRL）**，这是一个纯 RL 框架，通过在 RL rollout 阶段利用 few-shot 提示，**彻底消除对 SFT 的需求**。具体来说，ICRL 在 rollout 提示中引入 in-context 示例，教会模型如何调用外部工具；随着训练推进，in-context 示例数量逐渐减少，最终达到零样本设置，模型学会独立调用工具。

*We propose In-Context Reinforcement Learning (ICRL), an RL-only framework that eliminates the need for SFT by leveraging few-shot prompting during the rollout stage of RL. Specifically, ICRL introduces in-context examples within the rollout prompts to teach the model how to invoke external tools. Furthermore, as training progresses, the number of in-context examples is gradually reduced, eventually reaching a zero-shot setting where the model learns to call tools independently.*

---

## 核心洞见：为什么需要 ICRL？

### 传统方案的痛点

```
传统方案：
标注工具调用示例 → SFT 微调 → RL 强化学习
       ↑
    成本高、数据少、冷启动难
```

**问题一：SFT 数据稀缺且昂贵**
工具调用的正确格式（什么时候调用、调用什么、如何解析返回值）很难自动合成，往往需要人工标注或强模型蒸馏，成本极高。

**问题二：冷启动问题**
在 RL 训练初期，模型对工具调用格式毫无概念，探索效率极低——大量 rollout 都因格式错误而浪费，RL 根本无法学到有效信号。

### ICRL 的解法

**用 in-context 示例替代 SFT 监督**

在 RL rollout 时，给模型提供几个工具调用的示例（few-shot），让模型知道"工具调用长什么样"——这样 rollout 一开始就能产生格式正确的轨迹，RL 信号马上变得丰富。

随着训练迭代，模型逐渐"内化"了工具调用的能力，这时减少 in-context 示例，最终到零样本——模型已经不需要示例了。

---

## SECTION 3 · ICRL 方法详解

### 框架概述：渐进式 In-Context 退火

![图4：ICRL 框架示意图——3-shot、2-shot、0-shot 三种 Rollout 模板的渐进演变](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/icrl/fig_4.png)

**图 4 · FIGURE 4**
ICRL 框架示意图，展示了三种 Rollout 模板的渐进演变过程。
- **3-Shot Rollout Template**（左）：提示中包含3个完整的工具调用示例（灰色=推理，蓝色=搜索查询，绿色=搜索结果，黄色=答案）
- **2-Shot Rollout Template**（中）：减少到2个示例，提示长度缩短
- **0-Shot Rollout Template**（右）：无任何示例，模型完全自主生成工具调用轨迹

*ICRL framework overview showing the progressive reduction of in-context examples from 3-shot to 0-shot during training. Each rollout template contains: Reasoning (gray), Search Query (blue), Observation/Search Result (green), and Answer (yellow).*

**核心流程：**

1. **初始阶段（多 shot）**：在 rollout 提示中包含 k 个（如3个）工具调用示例，模型参照示例格式生成响应，确保产生有效的工具调用轨迹
2. **训练推进**：RL 基于工具调用的结果（是否获得了有助于回答问题的信息）给出奖励，模型学习什么时候、怎么调用工具
3. **逐步退火**：随着训练步数增加，减少 in-context 示例数量（3→2→1→0），迫使模型将工具调用能力内化到参数中
4. **零样本收敛**：训练结束时，模型在无任何示例的情况下自主决定何时及如何调用工具

### 奖励设计

ICRL 使用基于工具调用结果的**验证奖励**：
- 对于搜索增强型问答：答案与标准答案匹配 → 正奖励
- 搜索调用格式正确但无助于答案 → 中性/轻微惩罚
- 无效工具调用（格式错误）→ 负奖励

这种设计的优势在于**奖励来源于任务本身**，无需额外的奖励模型。

---

## SECTION 4 · 实验结果

### 训练动态分析

以下三张图展示了不同训练设置（3-shot / 2-shot / 0-shot 起点）下的训练动态：

---

![图2：训练过程中的奖励变化曲线](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/icrl/fig_2.png)

**图 2 · FIGURE 2 — Reward 变化**
三种设置下的平均奖励曲线（均值）。
- **3-shot（红色）**：初始奖励高（得益于示例指导），但随着示例减少，奖励平稳后趋于稳定
- **2-shot（蓝色）**：初始稍低，但随训练推进奖励持续提升，最终与 3-shot 相当
- **0-shot（棕色）**：初始阶段奖励波动大（探索困难），但随训练深入显著提升，最终达到最高水平（训练后期甚至超过有示例的版本）

*Reward curves during training for 3-shot, 2-shot, and 0-shot settings. The 0-shot variant, while starting with higher variance, eventually achieves strong performance after sufficient training.*

---

![图1：训练过程中的响应长度变化](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/icrl/fig_1.png)

**图 1 · FIGURE 1 — Response Length 变化**
三种设置下的平均响应长度曲线。
- **3-shot（红色）**：响应长度稳定在约 300 tokens（受示例约束，格式更固定）
- **2-shot（蓝色）**：早期响应较长（约 400+ tokens），随后逐渐收敛
- **0-shot（棕色）**：早期响应较短（模型探索期），随训练逐步增长到 300-600 tokens

响应长度的增长说明模型在 0-shot 设置下**逐渐学会了生成更丰富的推理链和工具调用序列**，这是模型内化工具使用能力的证据。

*Response length curves showing how the model's generation behavior evolves during training. The 0-shot variant shows increasing response length, indicating the model is learning to generate more elaborate tool-use reasoning chains.*

---

![图3：训练过程中有效搜索次数变化](https://raw.githubusercontent.com/bluesky1018/paper-reading/main/assets/icrl/fig_3.png)

**图 3 · FIGURE 3 — Number of Valid Search 变化**
三种设置下每条响应中有效搜索调用次数的均值变化。
- **3-shot（红色）**：有效搜索次数稳定在约 1.1-1.2 次（受示例格式约束）
- **2-shot（蓝色）**：有效搜索次数显著增长，峰值达 2.0 次，训练后期稳定在约 1.5 次
- **0-shot（棕色）**：初期有效搜索次数低（约 0.75-1.0 次，格式不稳定），但随训练持续增长，后期达 1.5-2.5 次

**关键发现**：0-shot 设置下有效搜索次数的持续增长，直接证明了**模型通过 RL 训练自主学会了工具调用**——从几乎不会调用到越来越善于调用，完全无需 SFT 冷启动。

*Number of valid search calls per response. The 0-shot variant shows a clear learning trend, starting low and consistently increasing, proving that the model autonomously learns to invoke tools through RL alone.*

---

### 主要结果（基准测试）

实验在多个推理和工具调用基准上评估，包括：
- **数学推理**：AIME、MATH 等
- **事实问答**（搜索增强）：TriviaQA、PopQA 等
- **工具调用能力**：自定义工具调用基准

**ICRL 与基线对比：**

| 方法 | 工具调用能力 | SFT 数据需求 | 最终性能 |
|------|------------|-------------|---------|
| 零样本 RL（直接） | 几乎无法收敛 | 无 | 低 |
| SFT → RL（传统） | 强（有监督） | **大量标注数据** | 高 |
| **ICRL（本文）** | **自主学习** | **无需 SFT 数据** | **SOTA** |

ICRL 在无需任何 SFT 标注数据的情况下，在多个基准上达到甚至超越了传统 SFT+RL 流水线的性能。

---

## 结论

ICRL 证明了一个优雅的命题：**few-shot in-context 示例可以完全替代 SFT 冷启动**。通过在 RL rollout 时提供工具调用示例，再逐步退火到零样本，模型自主完成了从"被示例引导"到"内化能力"的转变。

*ICRL achieves state-of-the-art performance, demonstrating its effectiveness as a scalable, data-efficient alternative to traditional SFT-based pipelines.*

---

## ANALYSIS · 编者深度评析

### 🏆 最大贡献

**① 把 SFT 冷启动成本降为零**

工具调用的 SFT 数据是稀缺的——你需要大量示例来覆盖不同工具、不同场景的调用格式。ICRL 用 in-context 示例代替这些标注，意味着**任何工具都可以通过提供几个示例就立刻进行 RL 训练**，不再需要构建庞大的标注数据集。

**② 优雅的"渐进退火"设计**

从 k-shot 到 0-shot 的渐进式减少不是偶然的——这本质上是一种**课程学习（Curriculum Learning）**：先让模型在有辅助的情况下建立能力，再逐步撤掉辅助，迫使模型将能力内化到参数中。这个设计简洁而有效。

**③ 验证了 RL 能独立驱动工具使用能力的涌现**

"Number of Valid Search" 指标的持续增长（图3）是本文最有说服力的证据：不需要任何工具调用示范，仅靠任务奖励信号，模型自主发现了"调用搜索引擎有助于回答问题"这一策略。这是 RL 能力涌现的一个清晰案例。

### ⚠️ 不足之处

| 局限 | 说明 |
|------|------|
| **实验工具类型有限** | 论文主要聚焦于搜索引擎工具，Python 解释器等其他工具类型的实验相对有限，泛化性存疑。 |
| **退火策略的敏感性** | k-shot 的初始值选择、退火时机（何时从 k→k-1）对最终性能的影响未做充分消融。 |
| **训练稳定性问题** | 从图2可以看出，0-shot 设置下奖励曲线波动极大，训练稳定性明显弱于有示例的版本，实际部署时有风险。 |
| **与最强 SFT+RL 基线的差距** | 论文声称 SOTA，但在某些数据充足的场景下，经过充分 SFT 的模型结合 RL 是否真的比 ICRL 弱，需要更严格的对比。 |

### 💡 借鉴意义

**🎯 对工具调用系统开发者的启示**

如果你在为新工具训练 LLM，不再需要花大量精力构建 SFT 数据集。只需准备几个高质量的工具调用示例作为 in-context 示例，配上验证奖励函数，就能让模型自主学会使用新工具。

**🔧 对 RL 训练工程师的启示**

"渐进式 in-context 退火"是解决 RL 冷启动问题的通用方法论，不限于工具调用场景。在任何新任务上启动 RL 训练时，都可以考虑先用 in-context 示例建立初始分布，再逐步退火。

**⚡ 对 Agent 框架设计者的启示**

ICRL 暗示了一种新的 Agent 能力获取范式：**从"用示例教"到"用结果强化"**。前者是 SFT 的思路，后者是纯 RL 的思路，ICRL 将两者融合——先用示例初始化探索空间，再用 RL 精化策略。

**📐 对数据飞轮设计的启示**

如果把 ICRL 应用到个人助理场景：每当引入一个新工具（新 API、新能力），只需提供 3-5 个调用示例，让 Agent 在实际使用中通过 RL 内化该工具的使用方式，从而形成**无需标注的持续学习飞轮**。

### 📚 建议延伸阅读（5篇）

1. **必读·前置**：[ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)
   — Feng et al., 2025
   — ICRL 直接引用并对比的基线工具调用 RL 框架，理解 ICRL 需先了解 ReTool 的设计

2. **强烈推荐**：[DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948)
   — DeepSeek AI, 2025
   — RLVR 范式的代表性工作，ICRL 的奖励设计与 R1 一脉相承，是理解纯 RL 训练 LLM 的必读文献

3. **推荐**：[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
   — Yu et al., 2025
   — 大规模 RL 训练的工程优化，为 ICRL 的实际部署提供工程参考

4. **推荐**：[Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
   — Schick et al., Meta AI, 2023
   — 工具调用领域的奠基工作（自监督方法），与 ICRL 的对比揭示了从自监督到 RL 的范式演变

5. **延伸**：[Self-Play Fine-Tuning (SPIN)](https://arxiv.org/abs/2401.01335)
   — Chen et al., 2024
   — 用自生成数据替代标注数据的另一种思路，与 ICRL 的"无 SFT 数据"目标异曲同工

---

*原始论文：[arXiv 2603.08068](https://arxiv.org/abs/2603.08068) · GitHub：[applese233/ICRL](https://github.com/applese233/ICRL) · 翻译整理 by Claude · 2026-03-13*
