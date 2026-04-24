---
title: "AutoGen — 用可编程对话把多 Agent 协作变成工业级框架"
date: 2026-04-24 17:00:00 +0800
categories: [Agent, Multi-Agent, Framework]
tags: [autogen, multi-agent, conversable-agent, wu-2023]
math: true
---

## 基本信息

- **作者**: Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, Chi Wang
- **机构**: Microsoft Research, Penn State, UW-Madison
- **发表**: ICLR 2024
- **arXiv**: [2308.08155](https://arxiv.org/abs/2308.08155)

## 一句话总结

提出 **AutoGen**——微软的**可编程多 Agent 对话框架**。核心抽象:**Conversable Agent**(可对话 Agent)+ **Conversation Programming**(把多 Agent 协作用"对话 + 函数"混合的方式编程)。每个 Agent 可以是 LLM、人类、工具执行器的任意组合,通过统一的 `initiate_chat` / `send` / `receive` 接口协作。这个"**协作即编程**"的抽象让复杂的 multi-agent workflow 可以用几十行代码搭起来,大幅降低多 Agent 应用的开发门槛。AutoGen 开源后迅速成为与 LangChain、MetaGPT 并列的 LLM app 三大框架之一,在 2025 年被重写为事件驱动的 AutoGen 0.4 + AG2 分支。

![AutoGen 的核心抽象:Conversable Agent。每个 Agent 有 LLM 后端、工具执行器、人类接口三个可选模块,通过统一的消息接口协作。这个抽象统一了"纯 LLM"、"工具执行"、"人机交互"三种 Agent 类型。](/assets/img/autogen/x1.png)
_Figure 1:AutoGen 的 Conversable Agent 抽象_

---

## 背景:2023 年多 Agent 框架的混乱

### 各种各样的多 Agent 框架

2023 年下半年,多 Agent 方向迅速涌现大量工作:

- **CAMEL**:role-playing 模板
- **AutoGPT/BabyAGI**:单 agent 无限 loop
- **MetaGPT**:SOP 流水线
- **ChatDev**:waterfall + chat chain
- **AgentVerse**:规模化 agent 评估

**每个框架都有自己的 API、自己的假设、自己的限制**。想把它们组合成一个系统几乎不可能。开发者需要一个**通用、可扩展**的抽象。

### AutoGen 的愿景

微软团队的想法:**不发明新方法论,而是提供一个足够通用的框架,让所有多 Agent 模式都能在它上面实现**。

就像 TensorFlow/PyTorch 不是一个具体模型,而是构建模型的基础设施——AutoGen 是构建多 Agent 系统的基础设施。

---

## 核心抽象

### 1. Conversable Agent(可对话 Agent)

AutoGen 统一所有 Agent 为"Conversable Agent"概念:

```python
class ConversableAgent:
    def generate_reply(messages):       # LLM 生成回复
    def execute_function(call):          # 工具执行
    def get_human_input():               # 人类输入
    def send(message, recipient):        # 发消息
    def receive(message, sender):        # 收消息
```

一个 Agent 实例可以:

- **纯 LLM**(默认)
- **LLM + tools**(加 `function_map`)
- **LLM + code execution**(加 code executor)
- **纯 human**(`human_input_mode=ALWAYS`)
- **混合**:LLM 先生成方案,人类审核,再执行代码

**所有类型共用同一 interface**,可以自由组合。

### 2. 两种主要的通信模式

![AutoGen 支持两种通信模式:两 Agent 对话(A 和 B 交替说话)和多 Agent 群聊(一个 manager 控制发言顺序)。两种模式都通过统一消息接口实现。](/assets/img/autogen/x2.png)
_Figure 2:两种通信模式_

- **Two-agent chat**:两个 Agent 交替对话(如 User ↔ Assistant)
- **Group chat**:多个 Agent,由一个 **GroupChatManager** 决定下一个发言者

Group chat manager 可以:

- **Round-robin**:按顺序发言
- **Manual**:人类指定
- **Auto**:LLM 决定下一位发言者

这覆盖了绝大多数多 Agent 场景。

### 3. Conversation Programming(对话即编程)

AutoGen 的关键创新:把复杂多 Agent workflow 用"**对话初始化 + 回调函数**"表达。

**例:一个数学家 + 代码执行器协作**

```python
math_expert = AssistantAgent("MathExpert", system_message="...")
code_executor = UserProxyAgent("Executor", code_execution_config=...)

code_executor.initiate_chat(
    math_expert,
    message="Find all positive integer solutions..."
)
```

运行时:
1. Executor 发题目给 MathExpert
2. MathExpert 生成 Python code
3. Executor 执行 code,反馈结果
4. MathExpert 根据结果调整,循环直到完成

**所有逻辑都隐藏在 `initiate_chat` 里**——开发者只需指定谁先发言、谁收、system message 是什么。

---

## 实验验证

### 1. 数学问题求解

MATH benchmark(高中竞赛数学):

| Method | 准确率 |
|--------|--------|
| GPT-4 single | 31.7% |
| CoT | 32.8% |
| Program-aided prompting | 36.6% |
| **AutoGen (MathExpert + Code Executor)** | **52.4%** |

LLM + 代码执行协作带来**显著提升**。

### 2. 编程任务

HumanEval:
- GPT-4 direct: 67%
- **AutoGen (Coder + Tester + Executor 三角)**: **86%**

与 MetaGPT 类似的 multi-agent 增益。

### 3. 其他应用场景

![AutoGen 被用于 Retrieval-Augmented Generation, Text-to-SQL, Online 决策等多个场景。每个场景都用不同的 agent 组合,展示框架的通用性。](/assets/img/autogen/x3.png)
_Figure 3:AutoGen 的多场景应用_

- **Retrieval-augmented generation**:Retriever agent + LLM agent
- **Text-to-SQL**:Schema agent + Writer agent + Executor agent
- **决策 benchmarks**(ALFWorld):Planner + Executor + Grounding

---

## 工程影响

### 1. 与 LangChain / MetaGPT 三足鼎立

2023 年底至 2024 年,AutoGen、LangChain、MetaGPT 成为 LLM app 开发的三大开源框架:

- **LangChain**:单 Agent + 链式调用为主
- **MetaGPT**:SOP 流水线
- **AutoGen**:通用多 Agent 对话

AutoGen 在"需要多 Agent 动态协作"的场景占主导。

### 2. Microsoft 内部使用

微软的 AI 产品(Power Platform、Copilot 等)内部大量使用 AutoGen 架构组织 Agent 工作流。

### 3. 启发 OpenAI Swarm、CrewAI

OpenAI 2024 年推出的 Swarm 框架、流行的 CrewAI 框架,核心思想都受 AutoGen 启发——都是"可组合 Agent + 统一消息接口"。

### 4. 演化为 AutoGen 0.4 / AG2

2025 年,团队分裂:官方 AutoGen 0.4 重写为**事件驱动架构**;社区 fork AG2 保留原 API。这反映 AutoGen 的影响力和社区活跃度。

---

## 局限

### 1. 通用 = 缺约束

AutoGen 的通用性让它能做任何事,**但也容易做得糟糕**。没有 MetaGPT 那种强 SOP,多 Agent 很容易陷入无效循环。需要开发者自己设计约束。

### 2. 调试困难

多 Agent 交互的调试比单 Agent 难一个数量级。看一段 10000 条消息的 group chat log 找 bug 非常痛苦。

### 3. Token 成本高

每个 agent 维护自己的对话历史,group chat 中信息经常被多个 agent 各自 consume 一遍——**token 消耗通常是单 agent 的 3-10×**。

### 4. 2024 年后对 "多 Agent 是否必要" 的争论

Cognition 的 《Don't Build Multi-Agents》直接挑战多 Agent 价值。2025 年部分开发者回到单 Agent + 强工具路线,AutoGen 的"通用多 Agent"定位受挑战。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **统一的 Agent 抽象是多 Agent 框架的核心**:Conversable Agent 把 LLM、人类、工具统一到一个接口——这是 AutoGen 能扩展到任意场景的关键
2. **"对话即编程"是一个有力的心智模型**:把多 Agent 协作写成"谁先发言、谁响应"的 dialogue,比写状态机 / DAG 更直观,也更贴近 LLM 的原生表达
3. **框架 vs 方法论的角色分工**:MetaGPT 提供方法论(SOP),AutoGen 提供框架(抽象)。二者互补——用 AutoGen 实现 MetaGPT 的 SOP 是很自然的
4. **通用性有代价**:AutoGen 让你能做任何事,但也让你能搞砸任何事。在多 Agent 设计上,"有约束 + 通用性"才是最佳平衡——这也是后续 LangGraph、CrewAI 等框架的设计方向
</callout>

---

## 延伸阅读

- [MetaGPT 深度解读]({% post_url 2026-04-24-MetaGPT-SOP多Agent软件公司深度解读 %}) —— SOP 约束路线
- [CAMEL (Li et al., 2023)](https://arxiv.org/abs/2303.17760) —— Role-playing 多 Agent 起点
- [Cognition《Don't Build Multi-Agents》](https://cognition.ai/blog/dont-build-multi-agents) —— 反对多 Agent 立场
- [AutoGen 官方仓库](https://github.com/microsoft/autogen) —— 框架实现
- [LangGraph](https://github.com/langchain-ai/langgraph) —— 后继的状态机式多 Agent 框架
