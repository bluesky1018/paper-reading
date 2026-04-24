---
title: "Chain-of-Thought — 让大模型'写出思考过程',一行 prompt 解锁推理能力"
date: 2026-04-24 15:15:00 +0800
categories: [Agent, Reasoning]
tags: [chain-of-thought, cot, reasoning, wei-2022]
math: true
---

## 基本信息

- **作者**: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou
- **机构**: Google Research, Brain Team
- **发表**: NeurIPS 2022
- **arXiv**: [2201.11903](https://arxiv.org/abs/2201.11903)

## 一句话总结

**Chain-of-Thought (CoT)** —— 一个看似微不足道的 prompt 技巧: 在 few-shot 示例里不仅给答案,还给**中间推理步骤**。就这一个改动,让 540B PaLM 在 GSM8K 数学题上从 17.9% 飙到 **58.1%**,超过当时微调过的 SOTA。更惊人的是一个"涌现"现象: **CoT 只在 >100B 参数时有效**,小模型用了反而更差——这是第一次清楚观察到 LLM 的"推理能力涌现"。CoT 直接开启了大模型的"推理时代",是 ReAct、ToT、o1、R1 等一切后续推理工作的起点。

![标准 prompt 直接输出答案;CoT prompt 在示例中展示"问题 → 推理步骤 → 答案"的链条,让模型模仿。结果:数学题正确率质变。](/assets/img/cot/x1.png)
_Figure 1:CoT 对比标准 prompting——简单改动带来质变_

---

## 背景:大模型的"推理瓶颈"

### 2021 年的困境

GPT-3 发布后,业界发现 LLM 在常识 QA、阅读理解上表现不错,但**一到需要多步推理**(算术、常识推理链、符号推理)就**崩溃**。

例如 GSM8K(小学数学应用题):GPT-3 175B 直接答题准确率不到 20%。当时主流观点:**"LLM 不具备真正的推理能力"**。

改进方向有两条:

1. **微调**:用带 rationale 的数据 fine-tune(成本高,需要人工标注)
2. **换架构**:专门设计 reasoning module(太复杂)

### Wei 等人的发现

作者尝试最简单的路: **不微调、不换架构,只改 prompt 里示例的格式**——看模型能不能模仿。

**结果震惊业界**。

---

## 核心机制:把"思考过程"放进示例

### Standard prompting vs CoT prompting

**Standard**:

```
Q: Roger has 5 balls. He buys 2 more cans of 3 balls each. How many?
A: 11

Q: The cafeteria had 23 apples. If they used 20 and bought 6 more, how many?
A: _____
```

模型必须**一步到位**从 question 跳到 answer。

**CoT**:

```
Q: Roger has 5 balls. He buys 2 more cans of 3 balls each. How many?
A: Roger started with 5 balls. 2 cans × 3 balls = 6 new balls. 5 + 6 = 11. Answer: 11.

Q: The cafeteria had 23 apples...
A: _____
```

模型看到示例包含**思考步骤**,会模仿这种格式——输出一串推理,最后给答案。

### 为什么有效:分解降低了单步难度

直接答一道数学题需要 LLM 在**一个 forward 内**完成所有运算。
CoT 让模型把计算**分解为多步**,每步只做简单运算,显著降低每步出错概率。

用计算科学的语言:**CoT 等效于把计算深度从固定的 $L$(网络层数)扩展到可变的 $K \cdot L$($K$ 是推理步数)**。推理步骤是一种 **test-time compute**——花更多算力换更好答案。

---

## 最震撼的发现:推理能力是"涌现"的

![CoT 的关键图:在小模型(<10B)上 CoT 反而让准确率下降!只有在 >62B 模型上 CoT 才大幅超越 standard prompting,体现出"推理能力涌现"的 scaling pattern。](/assets/img/cot/x2.png)
_Figure 2:CoT 的涌现性——只在大模型上生效_

关键观察:

- **小模型(< 10B)**:CoT 反而**更差**——因为模型没能力生成连贯的推理步骤,产出一堆 garbage,最后答案更乱
- **中等模型(~62B)**:CoT 接近 standard
- **大模型(> 100B)**:CoT **大幅超越** standard,部分任务提升 40+ 个百分点

这是**第一次大规模观察到 LLM 的能力涌现现象**——特定能力只在模型规模超过某个阈值后才出现。这个发现启发了后续大量 emergent abilities 研究(Wei 2022)和 scaling law 讨论。

---

## 覆盖任务:数学、常识、符号推理全面提升

![CoT 在三类推理任务(算术、常识、符号)上全面提升,特别是算术类(GSM8K +40 分)。](/assets/img/cot/x3.png)
_Figure 3:三类推理任务的提升_

| Task | Standard (PaLM 540B) | **CoT** |
|------|---------------------|---------|
| GSM8K(数学) | 17.9% | **56.5%** |
| SVAMP | 69.9% | **79.0%** |
| AQuA | 25.2% | **35.8%** |
| StrategyQA(常识) | 68.6% | **77.8%** |
| Coin Flip(符号) | 49.0% | **99.4%** |

**GSM8K 提升最显著**——40 分绝对值提升,因为数学最需要多步计算。

---

## 工程影响

### 1. 开启"推理时代"

2022 年 CoT 发布后,**每一篇推理相关工作几乎都引用 CoT**。Self-Consistency、Tree of Thoughts、ReAct、Reflexion、Quiet-STaR、o1、R1 全部追溯到 CoT。

### 2. 让 Prompt Engineering 成为一门学科

CoT 之前,大家觉得"prompt 就是一句话"。CoT 证明**prompt 的结构可以解锁模型隐藏能力**。这催生了整个 prompt engineering 社区:few-shot selection、prompt optimization、auto-CoT 等方向。

### 3. Zero-shot CoT(Kojima 2022)

Kojima et al. 发现:在问题后加一句 **"Let's think step by step."** 就能触发 CoT 行为——**连示例都不用**。这让 CoT 变得"免费"。

### 4. Test-Time Compute 作为 scaling 新轴

CoT 第一次让大家意识到:**推理时的计算量也是可以 scale 的**。这个思想在 Self-Consistency 中演化为"采样投票",在 ToT 中演化为"搜索",在 o1/R1 中演化为"RL 训练长 CoT"。

### 5. Distillation 源头

CoT 的推理轨迹可以作为**训练数据蒸馏给小模型**——STaR、Orca、很多 reasoning-model 蒸馏工作的数据就是 CoT 生成的。

---

## 局限

### 1. 只对大模型有效

< 10B 模型用 CoT 更差。这让小模型的应用受限——直到后来 reasoning model 蒸馏让小模型也能做 CoT。

### 2. Hallucination 可能更严重

CoT 生成更多 text 也意味着更多编造空间。"推理步骤看起来对,答案却错"的案例很多——因为模型按格式写了"推理",但其实是 bullshit。

### 3. 不适合非推理任务

对翻译、文本生成、摘要等任务,CoT 往往没有或反而有害。

### 4. 缺乏可验证性

CoT 的中间步骤是**自然语言**——没法机器验证。要验证需要 PRM(Process Reward Model,Lightman 2023)。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **推理是 prompt 格式能"解锁"的能力,不需要改模型**:这个发现本身就是 prompt engineering 的成立论据——格式即能力
2. **Test-time compute 是 scaling 新轴**:CoT 证明推理时多算一会儿能显著提质量,这启发了后续所有 reasoning-time scaling 工作
3. **涌现现象第一次被精确记录**:小模型用 CoT 变差、大模型用 CoT 暴涨——这种 "相变" pattern 让 scaling law 研究更加重要
4. **简单到难以置信的东西可能是范式**:CoT 只是改了 few-shot 示例的格式,却开启了整个大模型推理时代。技术价值 ≠ 复杂度
</callout>

---

## 延伸阅读

- [ReAct 深度解读]({% post_url 2026-04-24-ReAct-推理与行动交替深度解读 %}) —— CoT + Action 的融合
- [Self-Consistency (Wang et al., 2022)](https://arxiv.org/abs/2203.11171) —— 采样多条 CoT + 投票
- [Tree of Thoughts 深度解读]({% post_url 2026-04-24-Tree-of-Thoughts-思维树深度解读 %}) —— CoT 扩展到搜索
- [DeepSeek-R1 深度解读]({% post_url 2026-04-24-DeepSeek-R1-RL推理模型深度解读 %}) —— CoT 内化到 RL 训练
- [Zero-shot CoT (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916) —— "Let's think step by step"
