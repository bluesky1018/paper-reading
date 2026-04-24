---
title: "MetaGPT — 把软件公司的 SOP 编码进多 Agent 系统,让 LLM 协作写真实软件"
date: 2026-04-24 16:45:00 +0800
categories: [Agent, Multi-Agent, Software Engineering]
tags: [metagpt, multi-agent, sop, software-company, hong-2023]
math: true
---

## 基本信息

- **作者**: Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, et al.
- **机构**: DeepWisdom, KAUST, 清华等
- **发表**: ICLR 2024 (Oral)
- **arXiv**: [2308.00352](https://arxiv.org/abs/2308.00352)

## 一句话总结

提出 **MetaGPT**——**第一个把"软件公司的 SOP"完整编码进多 Agent 系统**的工作。不同 agent 扮演不同角色(PM、Architect、PM、Engineer、QA),按**软件开发流程**(需求分析 → 设计 → 编码 → 测试)协作,通过**结构化的中间产物**(PRD、设计文档、API 规范)而非自由对话传递信息。这一"SOP 约束"极大降低了多 agent 的幻觉累积,在 HumanEval 上从 GPT-4 的 67% 提到 **85.9%**,SoftwareDev(真实软件开发)完成率碾压 AutoGPT 等同期方案。MetaGPT 的思想"**把人类组织的流程编码给 Agent**"影响了 ChatDev、AutoGen 等后续多 Agent 框架。

![MetaGPT 的核心架构:一个任务输入后,依次流经 PM(写 PRD)、Architect(设计)、PM(拆任务)、Engineer(写代码)、QA(测试)五个角色。每个角色的输出是结构化文档,作为下一个角色的输入。](/assets/img/metagpt/x1.png)
_Figure 1:MetaGPT 的 SOP 流水线_

---

## 背景:多 Agent 系统的信息丢失问题

### AutoGPT 等早期多 Agent 的通病

2023 年初 AutoGPT / BabyAGI / CAMEL 兴起,但很快暴露一个严重问题:**多 agent 之间用自由对话传递信息,信息在传递中严重失真**。

典型场景:

1. Planner agent: "实现一个 todo app, 支持增删改查"
2. Coder agent 收到消息,开始写代码
3. 写到一半发现需求不清,问 Planner
4. Planner 脑子里的需求已经和 3 天前不一样,给了新答案
5. Coder 重写
6. ... 陷入死循环

核心问题:**Agent 之间没有"结构化契约"**——每次交互都是自由文本对话,没人记住全貌。

### 人类公司怎么解决这个?

看真实软件公司:PM 写**需求文档**(PRD),Architect 写**设计文档**,Engineer 写**代码 + API 规范**,QA 写**测试计划**。**每个角色的输出是结构化文档,被明确消费,不随意丢失**。

MetaGPT 的核心直觉:**把这套 SOP 照搬给 agent**。

---

## 核心机制

### 1. 角色分工

MetaGPT 定义 5 种角色:

- **Product Manager (PM)**:把用户需求转为 **PRD** 文档(Product Requirements Document)
- **Architect**:基于 PRD 输出**系统设计**(data structure, API, file structure)
- **Project Manager**:把设计拆解为**任务列表**
- **Engineer**:根据任务写代码
- **QA Engineer**:写测试、执行、反馈 bug

每个角色有自己的:

- **System prompt**(定义身份和职责)
- **Input schema**(只接受特定类型的上游输出)
- **Output schema**(必须产出特定格式)
- **Tools**(能用哪些工具)

### 2. 结构化消息共享(Shared Message Pool)

![MetaGPT 的通信架构:不是 peer-to-peer 对话,而是一个 shared message pool + subscription 机制。每个角色订阅特定类型的 message,产出也写回 pool,其他订阅者自动看到。](/assets/img/metagpt/x2.jpg)
_Figure 2:Shared Message Pool + Subscription 通信_

关键设计:

- **所有 agent 消息发布到共享池**
- **每个 agent 订阅特定类型的消息**(如 Engineer 订阅 "Task" 消息)
- **消息是结构化对象**(有 sender、type、content schema)

这替代了自由对话:

- PM 发布 "PRD" message → Architect 订阅并消费
- Architect 发布 "Design" message → PM 订阅
- ...

**信息在 pool 中持久化,不会在对话传递中丢失**。

### 3. 可执行反馈(Executable Feedback)

![Engineer 写完代码后,QA 执行并反馈具体 bug。Engineer 根据反馈修改。这个 loop 用"代码能否通过测试"作为客观信号,降低 LLM 幻觉。](/assets/img/metagpt/x3.jpg)
_Figure 3:Executable Feedback 闭环_

Engineer 的 output 经过 QA 执行,如果失败:

- 具体错误 message 回传 Engineer
- Engineer 修改后重新提交
- 循环直到通过

关键:**用"代码可执行 + 测试通过"作为客观信号**,不依赖 LLM 自己判断"对不对"。这与 SWE-Agent 的 linter、Voyager 的 env feedback 异曲同工。

### 4. 人类可读的结构化中间产物

每个文档都按模板产出:

- **PRD**:Goals / User Stories / Requirements Pool / Metrics
- **Design**:Data Structure / API Spec / File List / Data Flow
- **Task**:具体 class/function 实现列表
- **Test Plan**:用例、边界、覆盖

这让**人可以随时介入审阅**,也让 agent 间的信息明确不丢失。

---

## 实验结果

![HumanEval / MBPP 上 MetaGPT 对比 AutoGPT/AgentVerse/ChatDev/GPT-4:MetaGPT 在 HumanEval 达 85.9%,远超 GPT-4 的 67.0%。SoftwareDev 任务上完成率和代码质量也最高。](/assets/img/metagpt/x4.png)
_Figure 4:MetaGPT vs 同期多 Agent 方法_

### HumanEval / MBPP

| Method | HumanEval | MBPP |
|--------|-----------|------|
| GPT-4 (direct) | 67.0% | 66.4% |
| AutoGPT | 43.9% | - |
| AgentVerse | 73.3% | - |
| ChatDev | 73.1% | - |
| **MetaGPT** | **85.9%** | **87.7%** |

**领先 GPT-4 直接求解 18 分**——多 agent 正确使用时能大幅超越单 agent。

### SoftwareDev(真实项目)

70 个真实软件开发任务(todo app、贪吃蛇游戏、计算器等)。

- **成功运行率**:MetaGPT **100%**,AutoGPT 33%,AgentVerse 67%
- **代码质量 (CodeBLEU)**:MetaGPT 最高
- **执行轮数**:MetaGPT 最少(流水线式高效,其他方法反复返工)

---

## 工程影响

### 1. "SOP 约束 > 自由对话" 范式确立

MetaGPT 证明:**给 agent 强结构约束比让它们自由协作好**。这个思想影响了 ChatDev(waterfall + chat chain)、AutoGen(Group chat with manager)、CrewAI(Role-Goal-Task)等后续所有严肃多 Agent 框架。

### 2. 结构化消息成为标准

"Shared pool + subscription + typed message" 范式在 AutoGen、MCP、LangGraph 等都有类似实现。MetaGPT 的 message 设计是其中最早系统化的之一。

### 3. Executable feedback 思想延伸

Engineer ↔ QA 的测试 loop 后来被广泛采用:SWE-Agent、OpenHands、Devin 都有 "code → test → fix" 循环。

### 4. 工业化多 Agent 应用的基础

MetaGPT 开源后下载量数百万,成为"多 agent 写代码"场景的 go-to 框架。后续的 AI Scientist、agentic RAG 系统很多以 MetaGPT 为起点。

---

## 局限

### 1. SOP 僵化

MetaGPT 的 pipeline 是固定的 5 步——对于不需要这么多阶段的简单任务(比如改一个 bug)是 overkill。灵活性不如 AutoGen 的 group chat。

### 2. 每个 agent 仍是 GPT-4

成本高,因为每个角色都是一次 GPT-4 调用。70 个任务总 cost 数百美元。

### 3. 多 Agent vs 单 Agent 的争论

Cognition(Devin 团队)2024 发文《Don't Build Multi-Agents》认为:**多 agent 并不总是比单 agent 强,只有在 "可并行 + 信息瓶颈小" 时才值得**。对连续依赖强的任务,单 agent + 长 context + 好工具 reliably 更稳。MetaGPT 的 SOP 约束某种程度上就是对这个反对意见的回应。

### 4. 只适合有清晰 SOP 的领域

软件开发有成熟 SOP,MetaGPT 好用。开放任务(写小说、创意设计)没有清晰 SOP,强行模板化反而限制创意。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **把人类组织编码给 Agent 是强大的 prior**:软件公司的 SOP、客服流程、律所流程——这些人类千锤百炼的流程是多 Agent 系统的现成脚手架,用好它们比从零设计强得多
2. **结构化消息 + 共享池 > 自由对话**:信息在自由对话中必然失真,严格定义消息 schema 让多 Agent 系统可靠性质变
3. **Executable feedback 是幻觉的解药**:让代码跑起来、让测试通过、让 API 返回值是实的——这些"执行信号"是多 Agent 系统唯一可靠的客观 ground truth
4. **多 Agent 不是总好**:MetaGPT 证明多 Agent 对软件开发有用,但也启发了反思(Cognition 的 Don't Build Multi-Agents)。结构约束是关键区别——有结构的多 Agent >> 无结构多 Agent >> 单 Agent + 好工具 >> 无结构多 Agent
</callout>

---

## 延伸阅读

- [AutoGen (Wu et al., 2023)](https://arxiv.org/abs/2308.08155) —— 微软的可编程多 Agent 框架
- [ChatDev (Qian et al., 2023)](https://arxiv.org/abs/2307.07924) —— 另一条多 Agent 软件开发路线
- [CAMEL (Li et al., 2023)](https://arxiv.org/abs/2303.17760) —— Role-playing 多 Agent
- [Cognition《Don't Build Multi-Agents》](https://cognition.ai/blog/dont-build-multi-agents) —— 反对多 Agent 的立场
- [SWE-Agent 深度解读]({% post_url 2026-04-24-SWE-Agent-Agent-Computer-Interface深度解读 %}) —— 单 Agent + 好 harness 的代表
