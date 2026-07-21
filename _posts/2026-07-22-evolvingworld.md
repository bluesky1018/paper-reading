---
layout: post
title: "EvolvingWorld：角色与世界协同演化的开放Schema框架"
date: 2026-07-22
categories: [论文解读, 角色扮演与世界模拟]
tags: [角色扮演, 大语言模型, 世界模型, 交互式叙事, 长程模拟, 多智能体]
---

> 📄 **论文**：EvolvingWorld: An Open-Schema Framework for Co-Evolving Role-Play Agents and World Model in Interactive Literary World
> 🔗 **arXiv**：[2607.17250](https://arxiv.org/abs/2607.17250)
> 🏢 **机构**：香港科技大学 / LIGHTSPEED / 华中科技大学

## 一句话总结

EvolvingWorld 提出了一个开放 Schema 的框架，将角色代理（Character Agent）与世界模型（World Model）耦合为协同演化的整体，通过7个可训练子任务实现长程文学世界中角色与世界状态的持续更新，在57本书籍构成的基准测试上显著降低了长程模拟中的性能退化问题。

## 背景与问题

大型语言模型（LLM）已能够流畅地进行角色扮演，模仿虚构角色并维持基于人格的对话。然而，模拟一个完整的文学世界带来了更难的**长程挑战**：随着故事展开，角色会修正信念、动机和关系，而地点、物体和背景条件也会随之改变。因此，目标不仅是生成下一句合理的话语，更要在多个场景中维持连贯的角色与世界状态。

现有的角色扮演系统存在三方面的不足：其一，大多数基于人格的代理仅依赖静态档案或短对话上下文；其二，多智能体环境通常使用手动指定的沙盒，难以扩展到多样化的文学世界；其三，已有的书籍交互系统只关注单场景角色扮演或部分长程更新，缺乏完整的演化机制。例如 BookWorld 虽然更新角色目标与全局事件，却缺乏完整档案演化、位置/实体级别的世界更新以及可训练的子任务监督。

作者认为，文学世界的模拟需要**开放 Schema 的协同演化**。"开放 Schema"意味着系统从每本书中推断出相关的角色与世界维度，而非将所有故事强行套入固定槽位：侦探的侦查习惯和维多利亚时代孤儿的社会地位所需的维度截然不同，世界维度也可能在学校规则、政治秩序或超自然体系之间切换。一旦这些维度确立，协同演化就会保持角色与世界状态的耦合：角色行动能重塑地点或社会秩序，而世界变化又能反过来改变角色的动机与档案。

## 核心方法

### 框架整体架构

EvolvingWorld 由两个耦合模块构成：**角色代理（Character Agent）** 与 **基于LLM的世界模型（World Model）**。给定一本书的快照，系统将向前模拟故事，两个模块协同维护持久的角色与世界状态。

![EvolvingWorld框架概览](https://arxiv.org/html/2607.17250v1/x1.png)
*图1：EvolvingWorld 模拟案例。从书籍中提取的快照出发，多个角色在角色状态与世界状态协同演化的过程中进行交互。*

![EvolvingWorld框架总览](https://arxiv.org/html/2607.17250v1/x2.png)
*图2：EvolvingWorld 框架总览，包含数据集构建（第1-2步）、模拟流水线（第3步）和评测方法（第4步）。*

### 角色代理（Character Agent）

角色代理使用**开放 Schema 档案**表示每个角色，因为来自不同书籍的角色差异很大。与先前的系统不同，EvolvingWorld 仅提供参考维度，允许 LLM 根据书籍的类型、背景和风格选择、合并或引入新字段。

角色代理支持以下关键特性：

- **多角色场景**：同时处理多个角色的交互，包括环境与角色群组作为特殊行动单元
- **持久档案演化**：将开放 Schema 档案中的每个维度视为可演化的角色状态组成部分
- **隐藏追踪器（Hidden Tracker）**：记录弱信号或新兴证据，防止过早触发档案更新；多次重复的信号跨场景积累后才触发变更

其中，**隐藏追踪器**是核心创新之一。不同维度的演化速度不同：情绪可能迅速改变，而性格特征通常需要积累的证据。隐藏追踪器将弱信号与主档案分开存储，待证据充分后再触发更新。

### 世界模型（World Model）

世界模型维护**全局世界状态**和**位置级物理状态**两个层级：

- **全局状态**：捕获世界层面的设置，如历史背景和社会制度，采用开放 Schema 设计以避免将全局状态压缩到固定维度
- **位置状态**：显式建模每个地点的物理状态，包括嵌套子位置（如房子中的房间）或独立原子位置

与依赖手动预定义单一世界的沙盒环境（如 Generative Agents）不同，EvolvingWorld 采用基于 LLM 的世界模型为所有位置构建详细的物理状态并在模拟过程中持续更新。

### 7个可训练任务

模拟流水线被分解为7个有序子任务：

| 任务编号 | 任务名称 | 执行模块 | 功能说明 |
|---------|---------|---------|---------|
| Task 1 | scene_cast | 世界模型 | 从角色集中选择参与本场景的角色子集 |
| Task 2 | location_scenario | 世界模型 | 生成场景计划，指定地点和场景描述 |
| Task 3 | motivation_update | 角色代理 | 为每个参与角色生成场景特定的动机 |
| Task 4 | next_character | 世界模型 | 选择下一个行动的角色 |
| Task 5 | interaction_gen | 角色代理 | 生成角色交互内容（包含思想、语言、行动） |
| Task 6 | world_update | 世界模型 | 根据交互内容更新全局和位置状态 |
| Task 7 | character_update | 角色代理 | 场景结束后更新每个参与角色的状态 |

### 数据集构建

作者从57本按时间顺序叙述的书籍中构建数据集，使用 Gemini-2.5-Pro 作为提取 LLM。数据集特点：

- **训练数据**：138,596 个监督训练样本
- **测试数据**：222 个快照（含域内和域外分布）
- **时间序列设计**：后续场景可作为超前参考，使角色和世界状态变化有文本证据支撑

![数据集词云](https://arxiv.org/html/2607.17250v1/x3.png)
*图4：开放Schema设计下角色档案（左）与全局世界状态（右）中状态维度的词云，字体大小与57本书籍语料库中的频率成比例。*

![书籍类型分布](https://arxiv.org/html/2607.17250v1/x4.png)
*图5：EvolvingWorld 57本书的两级类型分布，内环为5个粗粒度类型，外环为每个类型的主题子类别。*

![数据统计分布](https://arxiv.org/html/2607.17250v1/x5.png)
*图6：四个粒度级别的提取数据统计分布，每个面板显示带KDE曲线的直方图，虚线标注均值（红）和中位数（蓝）。*

### 评测框架

EvolvingWorld 引入了覆盖 **10个维度、20个指标** 的轨迹级 LLM-as-Judge 评测协议：

**CHARACTER 评分**（6个维度）：
- 角色一致性（Profile Fidelity, Speaking Style Fidelity, Motivation-Driven Behavior）
- 演化质量（Profile Update Fidelity, Profile Evolution Smoothness）
- 环境锚定（Environment Awareness, Environmental Utilization）
- 交互质量（Contextual Responsiveness, Narrative Progression）
- 动机生成（Motivation Quality）
- 指令遵从（Instruction Compliance）

**WORLD 评分**（4个维度）：
- 场景规划（Cast Selection Rationality, Location & Scenario Rationality, Scene Continuity & Coherence）
- 发言管理（Turn & Scene Orchestration）
- 世界状态维护（Global Update Sensitivity, Global State Accuracy, Location Update Sensitivity, Location State Accuracy）
- 指令遵从（Instruction Compliance）

## 实验结果

### 主要结果

在 Character Agent 评测上（表2），各规模模型的主要结果如下：

**闭源模型**：
- Claude-4.6-Opus 表现最优，平均得分 **94.97**
- Gemini-2.5-Pro 得分 85.20
- GPT-5.3-Chat 得分 85.36

**开源模型（EW训练）**：
- Qwen-32B (EW-ours) 平均得分 **57.06**，大幅超过同等规模的基线模型（Qwen2.5-32B-I：27.86）
- Qwen-14B (EW-ours) 得分 52.63
- Llama-8B (EW-ours) 得分 45.99

在 World Model 评测上（表3）：
- Claude-4.6-Opus 表现最优，平均得分 **77.76**
- DeepSeek-V3-0324 得分 57.58
- Qwen-32B (EW-ours) 得分 59.87

### 长程性能对比

![长程性能对比](https://arxiv.org/html/2607.17250v1/x9.png)
*图3：EvolvingWorld 与 BookWorld 在档案演化平滑度（PES）和场景连贯性（SCC）上的长程对比，阴影区域表示均值±std/4。*

EvolvingWorld 训练的模型与 BookWorld 相比，在两个关键长程指标上均表现更好，且随着场景数量增加性能退化更少，证明了协同演化机制在维持长程一致性方面的有效性。

### 与 BookWorld 的对比

| 模型 | CHARACTER Score (EW) | CHARACTER Score (BW) |
|-----|---------------------|---------------------|
| Qwen-7B | 45.53 | 明显更低 |
| Qwen-14B | 52.63 | 明显更低 |
| Llama-8B | 45.99 | 明显更低 |

EvolvingWorld 训练的模型在所有规模上均显著优于 BookWorld 方法，尤其在演化质量和世界状态维护维度上差距更为明显。

## 总结

EvolvingWorld 将交互式文学世界模拟从静态人格模仿重新定义为长程动态演化过程，通过开放 Schema 设计克服了现有系统对固定维度的依赖，使系统能够适应不同类型文学世界的多样性需求。角色代理与世界模型的耦合架构以及隐藏追踪器机制，使得不同速度的状态演化得以优雅处理，避免了过早更新或更新滞后的问题。

该框架提出的7个可训练子任务将复杂的端到端模拟分解为可独立学习的子问题，使小规模开源模型通过监督微调即可在 EvolvingWorld 基准上取得显著提升，弥补了与大型闭源模型之间的差距。从57本书中构建的多样化基准覆盖了悬疑、奇幻、历史、现实等多种类型，保证了评测的全面性与泛化性。

未来工作可以探索将 EvolvingWorld 框架扩展至更长的叙事时间线、与视频生成等下游应用的结合（论文附录J已展示初步结果），以及如何进一步提升域外分布上的泛化能力。该框架为构建更真实、更具文学深度的AI交互世界提供了重要的方法论基础。
