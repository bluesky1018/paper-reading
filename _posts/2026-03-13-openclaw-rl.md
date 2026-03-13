---
layout: post
title: "【论文精读】OpenClaw-RL：只需对话即可训练任意Agent"
date: 2026-03-13
categories: [AI, LLM, ReinforcementLearning]
tags: [强化学习, Agent, RLHF, 个人助理, 在线学习, arXiv]
---

> 📄 **论文精读 · arXiv 2603.10165**
>
> **OpenClaw-RL: Train Any Agent Simply by Talking**
>
> Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, Ling Yang · 2026年3月
>
> GitHub：[Gen-Verse/OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)
>
> 标签：强化学习 · 在线学习 · 个人Agent · 通用Agent · 下一状态信号

---

## ⚡ 核心发现（TL;DR）

- 每次 Agent 交互都会产生**下一状态信号**（用户回复、工具输出、GUI状态变化），但所有现有系统都在丢弃它——OpenClaw-RL 把它变成实时训练源
- 下一状态信号包含两类信息：**评估信号**（这次做得好不好）+ **指令信号**（应该怎么做）
- 提出**Hindsight-Guided On-Policy Distillation（OPD）**，将用户的文字纠正转化为 token 级别的梯度方向，比标量奖励信息密度高得多
- 同一套基础设施同时支持**个人 Agent 个性化**（用着用着就越来越懂你）和**通用 Agent 大规模 RL 训练**（Terminal/GUI/SWE/工具调用）

---

## ABSTRACT · 摘要

每次 Agent 交互后，都会产生一个下一状态信号——用户回复、工具执行结果、GUI 状态变化或测试结果。然而现有的 agentic RL 系统都将其仅仅作为下一次动作的上下文，而没有将其作为实时在线学习来源。

*Every agent interaction generates a next-state signal, namely the user reply, tool output, terminal or GUI state change that follows each action, yet no existing agentic RL system recovers it as a live, online learning source.*

我们提出 **OpenClaw-RL**，这一框架建立在一个简单的观察之上：下一状态信号是**通用的**，策略可以同时从所有信号中学习。个人对话、终端执行、GUI 交互、SWE 任务和工具调用轨迹并不是独立的训练问题——它们都是可以在同一个循环中训练同一策略的交互。

*We present OpenClaw-RL, a framework built on a simple observation: next-state signals are universal, and policy can learn from all of them simultaneously.*

下一状态信号编码两类信息：
- **评估信号（Evaluative signals）**：说明动作执行效果，通过 PRM Judge 提取为标量奖励
- **指令信号（Directive signals）**：说明动作应该如何改变，通过 Hindsight-Guided OPD 恢复

由于异步设计，模型服务实时请求、PRM 评判进行中的交互、训练器更新策略，三者同时进行，零协调开销。

---

## SECTION 1 · 引言：两大信号浪费

### 浪费一：评估信号

下一状态信号隐式地对前一个动作打分：用户重复提问意味着不满，测试通过意味着成功，错误跟踪意味着失败。这构成了一种**自然的过程奖励（Process Reward）**，无需额外标注流水线。

*The next-state signal implicitly scores the preceding action: a user re-query signals dissatisfaction, a passing test signals success, and an error trace signals failure.*

然而，现有系统要么忽略这个信号，要么只在离线、预收集的形式下利用它——依赖固定数据集或终端结果奖励。

### 浪费二：指令信号

超越打分，下一状态信号往往携带**指令性信息**：一个说"你应该先检查文件"的用户，不仅告诉模型回答错了，还指定了 token 级别的改进方向。同样，详细的 SWE 错误跟踪通常暗示了具体的纠正方向。

*Beyond scoring, next-state signals often carry directive information: a user who says "you should have checked the file first" specifies not only that the response was wrong, but also how it should change at the token level.*

当前 RLVR 方法使用标量奖励，因此**无法将这类信息转化为方向性策略梯度**——这是巨大的信息浪费。

![图1：OpenClaw-RL 系统架构概览](https://arxiv.org/html/2603.10165v1/x1.png)

**图 1 · FIGURE 1**
OpenClaw-RL 基础设施概览。交互流来自两类 Agent：**个人 Agent**（对话式，单用户，部署在个人设备上）和**通用 Agent**（Terminal、GUI、SWE、工具调用，部署在云服务上）。收集的样本流入基于异步 slime 框架构建的 RL 服务器，包含四个解耦组件：环境服务器、PRM/Judge（奖励计算）、Megatron（策略训练）、SGLang（策略服务）。

*OpenClaw-RL infrastructure overview. Interaction streams come from Personal Agents (conversational, single-user) and General Agents (terminal, GUI, SWE, tool-call). Four decoupled components: environment server, PRM/Judge, Megatron for training, SGLang for serving.*

---

## SECTION 2 · 问题建模

将每个交互流形式化为 MDP（S, A, T, r）：

| 组件 | 含义 |
|------|------|
| **状态 sₜ** | 到第 t 轮为止的完整对话或环境上下文 |
| **动作 aₜ** | Agent 的响应，由 πθ 生成的 token 序列 |
| **转移 T(sₜ₊₁\|sₜ, aₜ)** | 确定性的；sₜ₊₁ 是用户回复、执行结果或工具输出 |
| **奖励 r(aₜ, sₜ₊₁)** | 通过 PRM Judge 从下一状态信号推断 |

在标准 RLVR 中，结果 o 作为整个轨迹的奖励。但过程奖励 r(aₜ, sₜ₊₁) 包含丰富得多的信号——特别是当下一状态包含指令性信息时，在线策略蒸馏（on-policy distillation）能将其转化为 token 级别的教师监督。

---

## SECTION 3 · 系统架构

*OpenClaw-RL Infrastructure: Unified System for Personal and General Agents*

### 3.1 四组件异步流水线

核心架构原则是**完全解耦**：策略服务、环境托管、PRM 评判、策略训练作为四个完全独立的异步循环运行，相互之间没有阻塞依赖。

```
Policy Serving → Environment → Reward Judging → Policy Training
   (SGLang)      (Http/API)    (SGLang/API)      (Megatron)
```

模型服务下一个用户请求的同时，PRM 评判上一个响应，训练器应用梯度更新——三者互不等待。

### 3.2 支持的交互场景

**表 1：支持的 Agent 设置及其环境特性**

| 设置 | 环境 | 下一状态信号 | 时程 |
|------|------|-------------|------|
| **OpenClaw** | 个人设备 | 用户回复 / 工具调用结果 | 长 |
| **Terminal** | Shell 执行沙盒 | stdout/stderr, exit code | 长 |
| **GUI** | 屏幕状态 + 可访问性树 | 视觉状态差异, 任务进度 | 长 |
| **SWE** | 代码仓库 + 测试套件 | 测试结果, diff, lint 输出 | 长 |
| **Tool-call** | API/函数执行 | 返回值, 错误跟踪 | 中 |

### 3.3 会话感知环境服务器

每个 API 请求被分类为两种：
- **主线轮次（Main-line turn）**：Agent 的主要响应和工具执行结果 → 形成可训练样本
- **侧线轮次（Side turn）**：辅助查询、记忆组织、环境转换 → 转发但不产生训练数据

---

## SECTION 4 · 从下一状态信号学习

*Learning from Next-State Signals: Unified RL Across Interaction Types*

### 4.1 个人 Agent 的 Binary RL

**将评估性下一状态信号转化为标量过程奖励。**

PRM Judge 基于用户的下一条回复或工具调用结果评判响应质量：`PRM(aₜ, sₜ₊₁) → r ∈ {+1, −1, 0}`

运行 m 次独立查询并多数投票：`r_final = MajorityVote(r₁, …, rₘ)`

训练目标为标准 PPO 风格的截断代理，带非对称边界（ε=0.2, ε_high=0.28, β_KL=0.02）。

### 4.2 个人 Agent 的 Hindsight-Guided OPD

**将指令性下一状态信号转化为 token 级别的教师监督。**

Binary RL 将 sₜ₊₁ 的全部信息压缩为一个标量。但用户写"你应该在编辑前先检查文件"传达的远不止这些——不仅说明回答错了，还指出哪些 token 应该改变及如何改变。

![图2：OpenClaw 个性化优化演示](https://arxiv.org/html/2603.10165v1/x2.png)

**图 2 · FIGURE 2**
只需正常使用 OpenClaw 即可优化你的个人 Agent。这里展示了一个模拟结果。

*Optimize your OpenClaw simply by using it. We provide a simulation result here.*

**OPD 四步流程：**

**Step 1. 后见之明提示提取（Hindsight hint extraction）**

`Judge(aₜ, sₜ₊₁) → {score ∈ {+1, −1}, hint ∈ T*}`

关键设计：不直接使用原始 sₜ₊₁，而是让 Judge 模型将其提炼为简洁可操作的指令（1-3句），聚焦于"回答应该如何不同"。

**Step 2. 提示选择与质量过滤**

在字符数 > 10 的正面投票中，选取最长（最具信息量）的提示。如果不存在有效提示，直接丢弃该样本——OPD 以样本数量换取信号质量。

**Step 3. 增强教师上下文构建**

将提示附加到最后一条用户消息：`s_enhanced = sₜ ⊕ hint`

这就是"模型本应看到的"——如果用户预先提供了纠正的话。

**Step 4. Token 级别优势**

```
Aₜ = log π_teacher(aₜ | s_enhanced) − log πθ(aₜ | sₜ)
```

- Aₜ > 0：教师（知道提示）对此 token 赋予更高概率 → 学生应增大它
- Aₜ < 0：教师认为此 token 不合适 → 学生应降低它

与标量优势不同，这在单个响应内提供了**逐 token 的方向性指导**。

### 4.3 Binary RL 与 OPD 的互补性

**表 2：不同学习方法对比**

| 维度 | Binary RL | OPD | 组合 |
|------|-----------|-----|------|
| 信号类型 | 评估性（好/坏） | 方向性 | 评估 + 方向 |
| 优势形式 | 序列级标量 | Token 级方向 | 混合 |
| 覆盖密度 | 所有评分轮次 | 仅提示接受轮次 | 所有评分轮次 |

OPD 以样本数量换取信号质量（针对性高分辨率），Binary RL 以粗信号换取广覆盖——两者互补。

![图3：方法概述](https://arxiv.org/html/2603.10165v1/x3.png)

**图 3 · FIGURE 3**
方法概述。对于个人 Agent，同时支持 Binary RL（奖励优化）和在线策略蒸馏训练。实验发现两者组合能带来显著性能提升。对于通用 Agentic RL，在标准 RLVR 基础上额外提供逐步奖励和标准化方法。

*Method Overview. For personal agents, we support both binary-reward optimization and on-policy distillation training. For general agentic RL, we provide integrated step-wise rewards.*

### 4.4 通用 Agentic RL 的逐步奖励

在长时程 Agentic 任务中，仅有结果奖励只在最后一步提供梯度信号，绝大多数轮次无法被监督。PRM 为每个轮次分配奖励，提供**密集的逐步信用分配**。

结合方式：在每步 t 处，使用 `o + Σᵢ rᵢ/m` 作为奖励，将结果奖励和过程奖励直接相加。

---

## SECTION 5 · 实验结果

![图4：实验结果综合图](https://arxiv.org/html/2603.10165v1/assets/four_square_plots.png)

**图 4 · FIGURE 4**
四组实验场景的综合结果图，展示了 OpenClaw-RL 在 Personal Agent、Terminal、GUI、SWE 等不同设置下的训练曲线和性能提升。

*Comprehensive experimental results across four settings: personal agent personalization, terminal, GUI, and SWE agent RL training.*

### 5.1 个人 Agent 实验：两个模拟场景

**场景一：不想被发现用AI的学生**
- 学生用 OpenClaw 做作业，同时尽量不让别人看出是AI写的
- 策略：Qwen3-4B，学习率 1×10⁻⁵，每16条样本触发一次训练
- 结果：Binary RL + OPD 组合带来显著提升

**场景二：想要评语具体且友好的老师**
- 教师用 OpenClaw 批改作业，需要反馈既具体又友好
- 相同模型和训练设置

两个场景均验证了：**用着用着，Agent 真的越来越懂你。**

### 5.2 通用 Agent 实验设置

| 设置 | 模型 | 数据集 | 评估 |
|------|------|--------|------|
| Terminal | Qwen3-8B | SETA RL data | 平均 rollout 任务精度 |
| GUI | Qwen3VL-8B-Thinking | OSWorld-Verified | 训练集（排除 chrome 和 multi-apps） |
| SWE | Qwen3-32B | SWE-Bench-Verified | 平均 rollout 任务精度 |
| Tool-call | Qwen3-4B-SFT | DAPO RL data | AIME 2024 |

### 5.3 关键实验结论

> 🔑 **Binary RL + OPD 组合效果最佳**
>
> 在个人 Agent 设置中，单独使用 Binary RL 或 OPD 都有提升，但两者组合（加权损失）带来的提升显著超过各自单独使用，验证了评估信号与指令信号的互补性。

> 📊 **逐步奖励对长时程任务至关重要**
>
> 在 Terminal、GUI、SWE 等长时程设置中，加入 PRM 逐步奖励后性能持续优于仅使用结果奖励的基线，验证了密集信用分配的重要性。

---

## SECTION 7 · 结论

> 每次 Agent 交互都产生了一个下一状态信号，它编码了 Agent 的表现如何，以及通常它应该如何行动得不同。OpenClaw-RL 建立在一个洞察之上：这些信号是**流无关的（stream-agnostic）**，一个策略可以同时从所有信号中学习。
>
> Binary RL 将评估信号转化为标量过程奖励，OPD 将指令信号转化为 token 级别的优势监督。组合使用带来显著优化增益。结果是一个系统：模型同时对个人用户个性化，并在长时程 Agentic 任务上持续改进——完全从它已经进行的交互中训练。

---

## ANALYSIS · 编者深度评析

### 🏆 最大贡献

**① 把"用户每次说话"都变成训练信号**

这是核心洞见。现有所有系统都在把最宝贵的信号（用户的真实反馈）当作纯上下文丢弃。本文第一个系统性地将其变成实时训练源，且适用于所有交互类型（对话/终端/GUI/SWE/工具调用）。

**② OPD：超越标量奖励的 token 级方向**

Hindsight-Guided OPD 是本文最精彩的技术贡献。它解决了 RLHF 一直存在但从未被优雅解决的问题：用户的文字纠正本来包含 token 级别的方向性信息，但所有基于标量奖励的方法都把它压缩成了 +1/-1，信息损失巨大。OPD 通过"如果用户预先告诉我该怎么做，我会怎么回答"这个思维实验，优雅地提取出 token 级梯度。

**③ 真正的在线学习：零数据预收集**

现有所有 RL 基础设施都假设"先收集数据，再训练"。OpenClaw-RL 是第一个真正实现连续在线学习的系统——四组件异步架构使得服务、收集、评判、训练同时进行，互不阻塞。

### ⚠️ 不足之处

| 局限 | 说明 |
|------|------|
| **个人 Agent 实验为模拟** | 论文中个人 Agent 的实验是用 LLM 模拟用户，而非真实用户。真实世界中用户反馈的噪声和多样性可能带来更多挑战。 |
| **隐私风险** | 个人 Agent 的训练数据来自个人设备的真实对话，如何保证数据不被服务器滥用？虽然提到了"confidential API"，但细节不足。 |
| **OPD 依赖提示提取质量** | OPD 的效果高度依赖 Judge 模型能否准确提取"指令性提示"，如果用户反馈含糊或 Judge 提取失败，OPD 会直接丢弃样本。 |
| **通用 Agent 实验规模有限** | Terminal/GUI/SWE/Tool-call 各用一个模型、一个数据集，缺乏跨模型、跨任务的系统性对比。 |

### 💡 借鉴意义

**🎯 对个人助理产品设计的启示**

用户的每次纠正、重新提问、补充说明，都是黄金训练数据。如果你在做个人助理产品，考虑将用户反馈信号转化为模型改进信号，而不是仅仅用于产品分析。

**🔧 对 RLHF 研究者的启示**

OPD 提供了一个新思路：不需要收集"偏好对"（preferred/rejected），只需要用户的文字纠正 + 模型的自我对比，就能得到比标量奖励更丰富的训练信号。

**⚡ 对 Agent 工程师的启示**

异步四组件解耦架构是工程上的重大贡献：服务不停、数据不批处理、训练不阻塞。这是从"训练好再部署"到"边部署边训练"的基础设施范式转变。

### 📚 建议延伸阅读（5篇）

1. **必读·前置**：[OpenClaw（个人 Agent 系统本体）](https://github.com/Gen-Verse/OpenClaw-RL)
   — 理解 OpenClaw-RL 需要先了解其前身 OpenClaw 的设计

2. **强烈推荐**：[RLAnything: RL Training for General Agents](https://arxiv.org/abs/2602.05618)
   — Wang et al., 2026 · arXiv 2602.05618
   — 本文直接引用并在其基础上扩展的通用 Agent RL 框架，逐步奖励设计的直接来源

3. **推荐**：[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
   — Yu et al., 2025 · arXiv 2503.14476
   — 本文使用的工具调用训练数据集来源，理解大规模 RLVR 训练的重要参考

4. **推荐**：[Let's Verify Step by Step (PRM800K)](https://arxiv.org/abs/2305.20050)
   — Lightman et al., OpenAI, 2023
   — 过程奖励模型的奠基工作，本文中 PRM Judge 设计的理论基础

5. **延伸**：[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
   — Andrychowicz et al., 2017
   — OpenClaw-RL 的 OPD 在思想上与 HER（后见之明经验回放）一脉相承，值得追溯这一思想的起源

---

*原始论文：[arXiv 2603.10165](https://arxiv.org/abs/2603.10165) · GitHub：[Gen-Verse/OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) · 翻译整理 by Claude · 2026-03-13*
