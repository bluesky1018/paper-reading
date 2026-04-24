---
title: "Generative Agents — 25 个 AI 居民的模拟小镇,Agent 记忆架构的开山之作"
date: 2026-04-24 18:15:00 +0800
categories: [Agent, Memory, Simulation]
tags: [generative-agents, smallville, memory-stream, reflection, park-2023]
math: true
---

## 基本信息

- **作者**: Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein
- **机构**: Stanford, Google Research
- **发表**: UIST 2023 (Best Paper)
- **arXiv**: [2304.03442](https://arxiv.org/abs/2304.03442)

## 一句话总结

提出 **Generative Agents**——在一个叫 **Smallville** 的 2D 小镇里放 **25 个 AI 居民**,让它们**自主生活**:每个 agent 有自己的人设、记忆、日常计划,能与其他 agent 对话、形成关系、协调活动。最震撼的实验:只给一个 agent 一个目标"办一个情人节派对",最终整个小镇居民通过口口相传、自发规划,竟然在当天聚集到正确地点——**完全涌现的群体行为**。技术上的核心贡献:一套**三件套 agent 记忆架构**——**Memory Stream**(时间序自然语言日志)、**Reflection**(阶段性反思总结)、**Planning**(分层日程规划)。这个架构奠定了"agent 认知"的研究范式,被后续 Voyager、Park 自己的后续工作、无数 chat agent 继承。

![Smallville 的俯视图:2D 小镇,25 个 AI 居民,每人有名字、人设、工作、家。论文里所有 agent 行为都是 GPT-3.5 驱动的自主决策,无任何脚本。](/assets/img/generative-agents/x2.png)
_Figure 1:Smallville——25 个 AI 居民的小镇_

---

## 背景:AI 能做出真正的"可信行为"吗?

### 游戏 NPC 的百年难题

游戏中的 NPC 一直"木讷":对话选项固定、行为脚本化、没有"记忆"、没有人设一致性。这让玩家永远觉得 NPC 是假的。

### LLM 为 NPC 带来希望

GPT-3/4 出现后,大家觉得 LLM 有"人格",能做对话。但一个更深的问题:**NPC 能不能有自己的生活?不只是响应玩家,而是在玩家不在场时自己吃饭、睡觉、工作、与其他 NPC 交流?**

Park et al. 的目标就是这个:**不是做一个对话 bot,是做一个能自主生活的 AI 居民**。

### 核心技术挑战

- **一致性**:agent 昨天的行为和今天要一致(记忆)
- **长期规划**:agent 要知道"今天要做什么、明天要做什么"
- **社会互动**:agent 要能和其他 agent 交互、形成关系
- **涌现行为**:希望整个小镇出现自发的社会现象

---

## 核心架构:Memory Stream + Reflection + Planning

![Generative Agent 的三层认知架构:底层是 memory stream(时间序自然语言日志),中层是 reflection(阶段性抽象总结),顶层是 planning(分层日程)。三者通过 LLM 连接。](/assets/img/generative-agents/x3.png)
_Figure 2:Generative Agents 的三层认知架构_

### 1. Memory Stream(记忆流)

每个 agent 有一个**自然语言日志**,按时间顺序记录所有经历:

```
[2023-02-13 09:15] Isabella woke up.
[2023-02-13 09:20] Isabella brushed her teeth.
[2023-02-13 09:30] Isabella had breakfast with Klaus at the cafe.
[2023-02-13 09:45] Isabella heard Klaus mention a Valentine's party.
...
```

每条记录是一个 **memory object**,包含:

- **时间戳**
- **自然语言描述**
- **重要性分数**(LLM 打分 1-10)
- **最近访问时间**

### 2. Retrieval:三因子排序

agent 做决策时要"回忆"相关记忆。检索评分 = **Recency + Importance + Relevance**:

- **Recency**:最近发生的事 recency 高(指数衰减)
- **Importance**:重要的事 importance 高(LLM 对每条打分)
- **Relevance**:与当前上下文语义相似的 relevance 高(embedding)

Top-k 记忆被注入 prompt 供决策用。**这个检索公式简单但深刻**——它对应人类"最近 + 重要 + 相关"的召回偏好。

### 3. Reflection(反思)

![Reflection 的 tree:底层是具体记忆(如"和 Klaus 喝咖啡"、"听到派对"),中层是 reflections(如"Isabella 关注 Valentine's 派对"),顶层是更高层的 reflections(如"Isabella 是社交型性格")。这种抽象层次让 agent 有"长期身份"。](/assets/img/generative-agents/x4.png)
_Figure 3:Reflection 的抽象层级_

当 agent 的 recent memory 的 importance 累加超过阈值,触发一次 reflection:

1. LLM 读最近 100 条 memory
2. 自问:**"What 3 high-level questions can I ask about these memories?"**
3. 对每个问题,LLM 基于具体记忆总结出 **reflection**(抽象结论)
4. Reflection 也作为 memory 存入 stream,但标记为 reflection type

例:

- Memory: "Isabella mentioned she wants to host a party"
- Memory: "Isabella smiled when talking about Valentine's Day"
- Memory: "Isabella ordered decorations"
- → **Reflection**: "Isabella is planning a Valentine's Day party"

Reflection 可以**递归**——对 reflection 再做 reflection,形成层次抽象。这让 agent 有"长期身份认识"。

### 4. Planning(规划)

每天早上,agent 做 **分层规划**:

1. **Daily plan**(粗粒度,LLM 生成):
   ```
   - 7am: wake up
   - 8am: breakfast
   - 9am-12pm: work at cafe
   - ...
   ```

2. **Hourly refinement**:每小时再细化当前小时的具体行动

3. **Minute-level action**:在当下情境中决定具体动作

当发生意外(如与 NPC 对话、收到消息),**重新规划**——可以改当天剩余的计划。

### 5. 对话生成

两 agent 相遇时,对话生成也调用 memory retrieval:

- 检索"与这个对话者的历史"
- 检索"当前话题的相关 memory"
- 组合成 prompt 让 LLM 生成回复

---

## 实验:涌现的社会行为

### 1. Valentine's Day 派对

著名实验:只给 Isabella **一个目标**——"想办一个情人节派对"。

观察结果:

- Isabella 自发邀请 Maria 和其他人
- Maria 回家告诉 Klaus
- Klaus 在 cafe 和其他人聊天时提起
- 最终 **12 个 agent 在派对当天自发聚集到正确地点**

**完全涌现的群体协调**——没有任何"指挥"。

### 2. 关系形成

在 2 天的 simulation 中观察到:

- agents 记住彼此的名字、职业、兴趣
- 形成友谊(互相访问更频繁)
- 甚至有了"情感联系"的描述(通过 reflection)

### 3. 人类评估

与 human 和 ablated 版本对比(去掉 reflection、去掉 planning 等),**full architecture 的 agent 行为最被认为"可信"**。人类评估员认为:

- Full Generative Agent > GPT-4 without memory > GPT-4 without planning > Human(Turing-test style)

是的,**一些人类评估员认为 AI 比人类更 believable**——这是因为 AI 的行为更一致而人类记忆有偏差。

---

## 工程影响

### 1. 定义了"Agent 认知架构"

Memory + Reflection + Planning 的三件套成为后续几乎所有严肃 agent 系统的起点:

- Voyager:Skill library + curriculum + verification(对应 memory/reflection/planning)
- MemGPT:main/recall/archival(memory 的分层细化)
- AutoGPT:task loop + reflection(简化版)
- Claude Code auto memory:feedback/project memory(对应 memory stream + reflection)

### 2. "Agent-as-character" 的可能

Park 的工作证明 LLM 可以扮演一个"连续的角色"——有自己的历史、性格、决策模式。这启发了 Character.AI 等产品路线,以及现在的 AI companion、AI NPC 市场。

### 3. Game NPC 的范式转变

NVIDIA、Inworld 等公司基于 Generative Agents 思想构建游戏 NPC 系统。将来的游戏 NPC 会**有真正的记忆和人格**——这个转变正在发生。

### 4. Simulation for AI 研究

用 AI agents 模拟人类社会行为成为一个新的研究方法——用于经济学、社会学、AI safety 等。Park 后续工作扩展到 1000 agents 的社会模拟。

---

## 局限

### 1. 成本高

25 个 agent × 2 天 simulation ≈ 数千美元的 GPT 调用。大规模 simulation 成本惊人。

### 2. Memory 线性扩展

Memory stream 只增不减——长期 simulation 下 memory 爆炸。需要 memory compression 或 archival 机制。

### 3. Reflection 的幻觉

Reflection 是 LLM 抽象出来的——可能抽象错(比如把随机事件总结为性格趋势)。这些错 reflection 又会影响后续行为,**错误放大**。

### 4. 无真正的 embodied 感知

agent 对世界的认识全部通过自然语言——没有视觉、声音、空间感知。真正的 embodied agent 需要更多。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **记忆 + 反思 + 规划是 agent 认知的三件套**:每一个都是独立的组件,每一个都有 LLM 来实现,三者组合出"可信行为"。这是 agent 设计的通用蓝图
2. **涌现可以从简单规则来**:只给 Isabella 一个"办派对"的目标,涌现出 12 个人自发聚集——复杂群体行为不需要复杂设计,简单的 agent 规则 + 多 agent 交互就够
3. **自然语言作为认知总线**:所有 memory、reflection、plan 都是自然语言字符串。这种"文本即数据"的设计让 agent 的所有行为都可解释、可调试、可 audit
4. **Generative Agents 是 "AI companion / NPC / persistent chatbot" 的祖先**:ChatGPT Memory、Claude 自动记忆、Character.AI 人设等等都有 Generative Agents 的 DNA
</callout>

---

## 延伸阅读

- [AI Agent 记忆系统全面解读(飞书文档)](https://feishu.cn/wiki/EUBmwWrPii1j5Skrwt0ccVqan7t) —— 记忆方向全景
- [MemGPT 深度解读]({% post_url 2026-04-24-MemGPT-操作系统式Agent记忆深度解读 %}) —— 另一种 agent memory
- [Voyager 深度解读]({% post_url 2026-04-24-Voyager-终身学习Agent深度解读 %}) —— 类似的三组件路线
- [Park et al. 1000 Agents Follow-up](https://arxiv.org/abs/2403.13193) —— 2024 年规模化版本
- [Smallville 演示视频](https://reverie.herokuapp.com/arXiv_Demo/) —— 官方 demo
