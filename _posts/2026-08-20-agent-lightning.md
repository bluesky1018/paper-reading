---
layout: post
title: "Agent Lightning v1.0：驾驭 Agentic RL 的新框架"
date: 2026-08-20
categories: [论文解读, LLM Agent]
tags: [Agentic RL, Agent Harness, Reinforcement Learning, Coding Agent, Training Framework]
---

> 📄 **论文**：Agent Lightning v1.0: Towards Harnessed Agentic RL
> 🔗 **arXiv**：[2608.17528](https://arxiv.org/abs/2608.17528)
> 🏢 **机构**：多机构联合

## 一句话总结

本文提出"Harnessed Agentic RL"范式，Agent Lightning v1.0 通过约 3,500 行框架代码，解决了在部署时使用的 Agent Harness 直接参与模型训练所带来的重新分词、优势计算、损失归一化和训练调度等核心挑战。

## 背景与问题

现代 LLM Agent 并非以独立 LLM 方式运行——它们在 Agent Harness 内部运行，Harness 负责管理工具、上下文和控制流。著名的 Harness 包括 mini-SWE-agent、OpenHands、OpenCode、Claude Code、Codex（代码 Agent）以及 OpenClaw、Hermes（通用 Harness）等。

**传统 Agentic RL** 框架（如 verl、AReaL、slime）要求用户在训练框架内部重新实现 Agent 的交互循环，这导致：
- 与现有复杂 Harness 难以集成
- 训练时与实际部署时行为存在差距
- 无法利用 Harness 在工具执行、上下文管理上的成熟策略

**Harnessed Agentic RL** 是本文提出的新范式：让部署时的 Agent Harness 直接参与模型训练，训练系统通过服务边界观察和优化模型调用序列。

### 两种范式的本质差异

| 特性 | 传统 Agentic RL | Harnessed Agentic RL |
|------|----------------|---------------------|
| 状态 | 环境 | Harness + 环境 |
| 模型输入 | 连续 token 历史 | 每次调用的独立 prompt |
| Agent 类型 | 单一 ReAct Agent | 多 Agent、子 Agent 和切换 |
| 训练引擎 | 拥有交互循环 | 观察调用序列 |

## 核心方法

### 框架架构

![Agent Lightning 整体框架](https://arxiv.org/html/2608.17528v1/agl-lite-teaser-0814.png)
*图1：Agent Lightning v1.0 整体框架，通过 LLM 端点代理连接任意 Agent Harness 与 RL 训练*

![控制器设计](https://arxiv.org/html/2608.17528v1/agl-lite-controller.png)
*图2：训练控制器设计，管理 Rollout 收集、样本组装和训练更新*

### Harnessed Agentic RL 的核心挑战

**挑战1：重新分词（Retokenization）**

在 Harnessed Agentic RL 中，Harness 拥有上下文构建，因此 prompt 在不同调用间可能因 Harness 对上下文的重新格式化而发生变化，无法简单将调用序列拼接为线性训练样本。

Rollout 被正式定义为调用对序列：
$$\mathcal{C}(\rho) = \bigl((p_1, a_1), (p_2, a_2), \ldots, (p_{T_\rho}, a_{T_\rho})\bigr)$$

其中状态分解为 Harness 状态和环境状态：$s_t = (s_t^{\mathrm{harness}}, s_t^{\mathrm{env}})$

prompt token 化过程：
$$C_t^{\mathrm{msg}} = \operatorname{Context}_H(s_t^{\mathrm{harness}})$$
$$p_t^{\mathrm{tok}} = \operatorname{Tok}(\operatorname{Template}(C_t^{\mathrm{msg}}))$$

**挑战2：优势计算**

由于每次调用都是独立的 prompt，传统的时序差分（TD）方法不能直接应用，需要在调用级别进行优势估计。

**挑战3：损失归一化**

跨调用的长度差异悬殊，需要仔细设计归一化策略以避免短调用和长调用之间的梯度不平衡。

**挑战4：训练后端调度**

异步 Rollout 收集与 GPU 训练的高效调度是另一个关键工程挑战。

## 实验结果

实验在三个实际 Agent 训练场景中验证 Agent Lightning v1.0：

1. **搜索 Agent**：遵循 Search-R1 实验设置
2. **通用指令遵循 Agent**：遵循 LLM-in-Sandbox 设置
3. **代码 Agent**：基于 SWE-smith 构建训练数据，提供完整的可复现代码 Agent 训练示例

本文特别针对代码 Agent 提供了详细的训练流程描述，包括数据清洗、环境搭建和完整数据管道，以及奖励黑客（reward hacking）的检测与防范。

## 总结

Agent Lightning v1.0 为"Harnessed Agentic RL"这一新范式提供了系统性的理论框架和工程实现。通过约 3,500 行的框架代码，它支持任意 Agent Harness，使得在训练时充分利用部署时的 Harness 能力成为可能，显著缩小了训练与实际部署之间的差距。

近期多个主流框架（verl Uni-Agent、AReaL 2.0、slime v0.3.0、Polar）也已采用类似的代理架构，印证了这一方向的重要性。未来工作可进一步探索多 Agent 协作场景下的 Harnessed RL 以及更高效的调度算法。
