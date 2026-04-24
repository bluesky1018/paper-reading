---
title: "Reflexion — 用自然语言'反思'代替梯度更新,让 Agent 从失败中学习"
date: 2026-04-24 15:30:00 +0800
categories: [Agent, Reasoning, Reinforcement Learning]
tags: [reflexion, self-reflection, verbal-rl, shinn-2023]
math: true
---

## 基本信息

- **作者**: Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
- **机构**: Northeastern, MIT, Princeton
- **发表**: NeurIPS 2023
- **arXiv**: [2303.11366](https://arxiv.org/abs/2303.11366)

## 一句话总结

提出 **Reflexion** —— 用**自然语言反思**(verbal reinforcement learning)代替传统 RL 的参数更新,让 Agent 在失败后写一段"我哪里错了、下次应该怎么做"的反思文字,存入 episodic memory,下次尝试时作为 context 注入。这样**无需任何模型参数更新**,Agent 可以通过多次 trial-and-error 快速提升——在 HumanEval 上超过 GPT-4 直接 sample 11 分,在 ALFWorld 上从 ReAct 的 73% 升到 97%。Reflexion 把经典 RL 的"evaluator → actor"闭环搬到纯文本空间,是 2023 年 agent 领域最具原创性的工作之一。

![Reflexion 的核心闭环:Actor 执行任务,Evaluator 评估结果,Self-Reflection 根据评估生成反思文字,反思存入 episodic memory 供下一轮 Actor 使用。无需梯度,全是 text.](/assets/img/reflexion/x1.png)
_Figure 1:Reflexion 的 Actor-Evaluator-Self-Reflection 循环_

---

## 背景:ReAct 的错误不可恢复

ReAct(Yao 2022)让 Agent 能 think + act,但有个致命问题:**一旦走错方向,Agent 会一路错下去**。

例:解决一个编程 bug

1. Thought: "应该改 function A"
2. Action: 改 function A
3. Observation: 测试仍失败
4. Thought: "A 改错了?再改一下..."
5. ... 继续在 A 上反复修改

模型**没有机制回顾"是不是方向错了",跳出当前 trajectory**。

传统 RL 解决这种问题靠**参数更新**——试错后用 reward 调权重。但对 LLM 这代价太大:每次试错要跑梯度,且很容易把 base model 训坏。

Reflexion 的思路:**让反思用自然语言写出来,作为 context 注入下一次尝试**——既规避参数更新,又实现"学习"。

---

## 核心机制:三个组件

### 1. Actor(行动者)

就是 ReAct agent——输出 Thought/Action 与环境交互,产出一条 trajectory。

### 2. Evaluator(评估者)

评估这条 trajectory 是否成功。可以是:

- **环境返回**(编程任务的测试结果、游戏胜负)
- **规则检查**(某些约束是否满足)
- **LLM 打分**(让另一个 LLM 评估)

### 3. Self-Reflection(反思者)

![Self-Reflection 读入 (trajectory, evaluation),用 LLM 生成一段"失败原因 + 下次改进方向"的文字。这段文字成为 Actor 下次尝试的额外 context。](/assets/img/reflexion/x2.png)
_Figure 2:Self-Reflection 的文本生成过程_

关键创新。Reflection 是一个 LLM prompt:

```
Given your previous attempt:
[trajectory]

And the result: [FAILED/SUCCESS with reason]

Write a concise reflection on what went wrong
and what you should try differently next time:
```

输出例如:

> "I assumed function A was the bug but the test failure
>  was in the helper B that A calls. Next time I should
>  trace the failure backwards from the assertion."

这段反思存进 **episodic memory**(按 task 分组的短语料)。

### 4. 下次尝试注入反思

Actor 下次尝试同一 task 时,reflection 作为 system prompt 一部分:

```
[System: You tried this task before. Your reflections:
- <reflection 1>
- <reflection 2>]

Task: <original task>
Thought: ...
```

Agent 读到历史反思,**在新的 trajectory 里规避已知错误**。

---

## 与 RL 的类比

Reflexion 本质是一个**不需要梯度的 RL**:

| 传统 RL | Reflexion |
|---------|-----------|
| Actor = policy $\pi_\theta$ | Actor = LLM with prompt |
| Environment reward | Evaluator's text feedback |
| Gradient update to $\theta$ | Append reflection to memory |
| Value function critic | Self-Reflection LLM |
| Episode | Trial |

**权重不变,memory 变**——这让迭代极快(不用 GPU 训练,只做 LLM 推理)且稳定(不会 catastrophic forgetting)。

---

## 实验结果

### 1. 代码生成(HumanEval, MBPP)

![Reflexion 在 HumanEval 上从 80% → 91% pass@1,超越当时 GPT-4 的 baseline(80%)。MBPP、LeetcodeHardGym 类似提升。](/assets/img/reflexion/x3.png)
_Figure 3:代码生成任务——Reflexion 大幅超越 baseline_

| Method | HumanEval pass@1 |
|--------|------------------|
| GPT-4 (direct) | 80.1% |
| CoT | 75.9% |
| ReAct | 70.7% |
| **Reflexion (GPT-4)** | **91.0%** |

**11 分提升**——这是无参数更新的方法第一次大幅超越 GPT-4。

### 2. 决策任务(ALFWorld)

| Method | Success Rate |
|--------|-------------|
| BUTLER(RL agent) | 26% |
| ReAct | 73% |
| **Reflexion** | **97%** |

从 73% 到 97% —— 几乎完全解决。

### 3. 推理任务(Hotpot QA)

![HotpotQA 上各种方法对比:Reflexion 比 CoT+SC 高 8 分,体现反思对多跳推理的帮助。](/assets/img/reflexion/x4.png)
_Figure 4:HotpotQA 推理任务_

---

## 可扩展设计

### 反思的不同粒度

- **Trajectory-level**:对整条 trajectory 反思一段
- **Step-level**:对每一步反思(更细但更贵)
- **Task-level**:对整类任务反思共性教训

Reflexion 默认用 trajectory-level。但不同任务可以选不同粒度。

### Memory 的长短

- **Short-term memory**:当前任务的反思
- **Long-term memory**:跨任务积累的反思(如"编程任务中,测试失败要先 trace 调用链")

---

## 工程影响

### 1. 把 RL 概念搬进 prompt 工程

"Verbal RL"作为一种范式被广泛接受。后续 Self-Refine、SelfCheck、Reflexion++ 等都在这条路上迭代。

### 2. 启发 LATS(Language Agent Tree Search)

LATS(Zhou 2023)把 Reflexion 的反思 + ToT 的搜索 + ReAct 的行动组合起来,达到 scaffold 方向的一个集大成。

### 3. 推理模型训练的前兆

o1 和 R1 训练出的模型在推理中会**自然产生反思**("Wait, let me reconsider...")——这些行为与 Reflexion 中的 verbal 反思异曲同工,只是把能力从 prompt 层内化到模型权重里。

### 4. Coding Agent 的重要组件

现代 coding agent(SWE-Agent、OpenHands 等)大多包含 "reflect on failure" 步骤,基本都是 Reflexion 变体。

---

## 局限

### 1. 反思本身可能错

反思是 LLM 生成的——可能错判失败原因、指向错误改进方向。反思链路有 compound error 风险。

### 2. 反思可能指向 trivial 修改

有时 reflection 只输出"下次更仔细"这种空话,没有实际指导。需要精心 prompt 才能生成可操作的反思。

### 3. 需要 evaluator

不是所有任务都有明确的 evaluator。对开放式任务(写作、设计)没有"对错"信号,Reflexion 难以直接应用。

### 4. 长程反思会膨胀

每次尝试都加反思,memory 越来越长。需要 summarization 或 truncation。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Verbal reinforcement learning 是一个独立范式**:用自然语言代替梯度的"学习",没有参数更新但有跨 trial 的改进能力——这是一种全新的"学习"定义
2. **错误不可恢复是 agent 的核心问题**:ReAct 能执行但会一路错下去;Reflexion 让 agent 有了"从失败中翻身"的能力——这是 agent 向高可靠演化的关键一步
3. **反思是 test-time compute 的另一种形式**:除了多步推理、多次采样,**回顾 + 改进** 也是一种 inference-time scaling**
4. **Episodic memory 是 agent 学习的关键载体**:把经验沉淀到可检索的自然语言块,这与后来的 MemGPT / Claude Code auto memory 一脉相承
</callout>

---

## 延伸阅读

- [ReAct 深度解读]({% post_url 2026-04-24-ReAct-推理与行动交替深度解读 %}) —— Reflexion 的 Actor 基础
- [Self-Refine (Madaan et al., 2023)](https://arxiv.org/abs/2303.17651) —— 同期的另一条反思路线
- [LATS (Zhou et al., 2023)](https://arxiv.org/abs/2310.04406) —— Reflexion + ToT + MCTS 集成
- [Tree of Thoughts 深度解读]({% post_url 2026-04-24-Tree-of-Thoughts-思维树深度解读 %}) —— 互补的搜索方法
- [DeepSeek-R1 深度解读]({% post_url 2026-04-24-DeepSeek-R1-RL推理模型深度解读 %}) —— 反思内化到模型参数
