---
title: "Tree of Thoughts — 把 CoT 从线升级为树,给 LLM 一个可搜索的思维空间"
date: 2026-04-24 15:45:00 +0800
categories: [Agent, Reasoning, Search]
tags: [tree-of-thoughts, tot, search, yao-2023]
math: true
---

## 基本信息

- **作者**: Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan
- **机构**: Princeton, Google DeepMind
- **发表**: NeurIPS 2023
- **arXiv**: [2305.10601](https://arxiv.org/abs/2305.10601)

## 一句话总结

提出 **Tree of Thoughts (ToT)**——把 Chain-of-Thought 的"线性推理链"升级为"**可搜索的树**"。每一步 LLM 生成多个 thought 候选,并用自己评估每个 thought 的"有望程度",然后像 BFS/DFS 一样选择最有前景的分支继续。在 Game of 24(24 点)上从 CoT 的 4% 跳到 **74%**,证明**推理 = 搜索**不是比喻而是可工程化的设计。ToT 是 2023 年最具"推理范式创新"的工作之一,直接启发了 LATS、AlphaCode2、o1 的 tree search 思路。

![CoT 是一条线(从问题到答案单路径),Self-Consistency 是多条独立的线,ToT 是一棵树(多分支可回溯)。表达力递增。](/assets/img/tot/x1.png)
_Figure 1:从 CoT 到 ToT——推理拓扑的升级_

---

## 背景:CoT 的"线性"问题

### CoT 的根本局限

Chain-of-Thought 是一条**单向链**:

```
Step 1 → Step 2 → Step 3 → ... → Answer
```

一旦某一步选错,整条链就崩——**不能回退,不能探索其他分支,不能评估自己**。

Self-Consistency(Wang 2022)稍有改进:独立采样多条 CoT 链,对答案做 majority vote。但**链之间彼此独立**,仍无结构。

### 人类怎么解决难题?

人类解决数学题、规划、创意写作时,通常:

1. 生成**多个候选步骤**
2. **评估**每个候选("这一步靠谱吗?")
3. 选择最有前景的继续
4. 必要时**回退重试**

这就是**搜索** —— 人类思维天然是树状的。ToT 的目标:给 LLM 同样的机制。

---

## 核心机制:4 个组件

ToT 定义了 4 个可插拔的组件,组合成一个通用的"思维搜索"框架:

### 1. Thought decomposition(思维分解)

每个"thought"不是一个单词或一个答案,而是一个**有意义的中间步骤**——依任务而定:

- Game of 24:一步算术操作(如 "3 + 4 = 7, 剩余 [6, 8, 7]")
- Creative Writing:一段写作的 plan
- Crosswords:一行或一列的猜测

### 2. Thought generator(生成器)

在当前 state,让 LLM **生成 $k$ 个 thought 候选**。两种方式:

- **Sample**:独立多次采样同一 prompt
- **Propose**:一次 prompt 生成 $k$ 个候选

### 3. State evaluator(评估器)

![State evaluator 让 LLM 对每个候选 thought 打分:"有多大可能引向正确答案"。可以用 classification(sure/likely/impossible),也可以用 voting 让多个评估者投票。](/assets/img/tot/x2.png)
_Figure 2:ToT 的 State Evaluator——让 LLM 自评估_

对每个候选 state,用 LLM 评估"有多大可能通向答案":

- **Value**:直接打分(1-10)
- **Vote**:对多个 state 做 pairwise 对比投票

这是 ToT 的关键——**用 LLM 评估自己**,让搜索有方向。

### 4. Search algorithm(搜索算法)

用经典搜索算法:

- **BFS**:每一层保留 top-$b$ 个 states,逐层扩展
- **DFS**:深度优先搜索,有 backtracking

---

## 实验:三个代表任务

### Game of 24(24 点)

给 4 个数字,用 + − × ÷ 得到 24。

![Game of 24 上 ToT 的树展开:每个 node 是一次运算后的 state。LLM 评估每个 state 是否"可能到 24",BFS 扩展最有前景分支。最终找到 [4, 9, 10, 13] → (10-4)×(13-9) = 24。](/assets/img/tot/x3.png)
_Figure 3:Game of 24 的 ToT 展开_

| Method | Success Rate |
|--------|-------------|
| CoT (GPT-4) | 4.0% |
| CoT + SC (100 samples) | 9.0% |
| ToT (b=1, BFS) | 45.0% |
| **ToT (b=5, BFS)** | **74.0%** |

**CoT 4% → ToT 74%——绝对值提升 70 分**。这种量级在推理领域极罕见。

### Creative Writing(创意写作)

要求 LLM 按 4 个 random 结尾句写一段连贯文章。

ToT 先生成多个 writing plan,评估后选一个最优 plan,再基于这个 plan 生成文本。

- CoT 连贯性打分:6.19
- **ToT**:7.56

### Crosswords(填字游戏)

5x5 mini crossword。

| Method | Letter Acc | Word Acc | Game Acc |
|--------|-----------|----------|----------|
| CoT | 40.6% | 15.6% | 0.7% |
| **ToT** | **78.0%** | **60.0%** | **20.0%** |

---

## 与搜索算法的一般关系

![ToT 其实是一个"search over LLM states"的框架。BFS/DFS/MCTS 都可以作为 search policy。评估和生成都由 LLM 做——这是一种 LLM-as-world-model + LLM-as-policy 的组合。](/assets/img/tot/x4.png)
_Figure 4:ToT 作为 LLM 驱动的搜索框架_

ToT 本质是把**经典搜索算法搬到文本空间**,LLM 扮演两个关键角色:

- **Transition function**:给定 state 生成下个 state 候选
- **Heuristic**:评估 state 的价值

这让 AI 规划 50 年的 BFS/DFS/A*/MCTS 算法重新可用——只是"state"从符号变为自然语言,"operator"从专家规则变为 LLM prompt。

---

## 工程影响

### 1. "推理即搜索"的范式确立

ToT 之后,推理研究广泛引入搜索思路:LATS(Zhou 2023)用 MCTS,AlphaCode2(DeepMind 2023)用采样+过滤,o1 内部用某种形式的搜索。**"test-time search"**成为 reasoning model 的核心概念。

### 2. 启发 o1 和 R1 的长 CoT

o1 和 R1 的 "think longer" 本质上是让模型在**自己内部做隐式搜索**——生成多个想法、评估、回退。这些行为以前是 ToT 外部调度的,现在被 RL 训练到模型权重里。

### 3. Graph of Thoughts 等后续工作

Besta 2023 的 Graph of Thoughts 把树扩成 DAG,允许合并、引用。ReST-MCTS*、rStar 等在 ToT 上引入 MCTS、学习到的 value function 等。

### 4. 对小模型的限制

ToT 需要强 evaluator——小模型评估不准导致搜索方向错。这再次印证"推理能力随规模涌现"。

---

## 局限

### 1. 成本高

ToT 的 query 数是 CoT 的 10-100 倍。对简单任务是 overkill,对复杂任务才值得。

### 2. Evaluator 的 bias

LLM 评估不可靠——容易偏好某些格式、某些答案长度。这让搜索可能偏向错的方向。PRM(Process Reward Model)等工作的动机就来自这里。

### 3. Thought 分解依赖任务设计

每个新任务都要重新设计"thought 是什么"。ToT 不是一个开箱即用的通用算法。

### 4. 不处理"真的不知道答案"的情况

如果连最好的 thought 都通不向答案,ToT 会一直在错的空间里搜。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **推理拓扑升级**:CoT(线)→ Self-Consistency(多条线)→ ToT(树)→ GoT(图)——推理的表达力随拓扑复杂化而增强
2. **LLM 可以评估自己**:State Evaluator 让同一个 LLM 既做 generator 又做 critic——这是 ToT 实用化的关键,也是后续 PRM、self-play 等思想的前身
3. **经典搜索算法 + LLM 是强组合**:BFS/DFS/MCTS 这些 50 年的 AI 算法在 LLM 驱动下重新焕发生机。规划 + 学习的结合永远有价值
4. **Test-time search 是 scaling 新轴**:除了训练 compute 和 model size,**搜索宽度和深度**也是可调的旋钮。o1/R1 的长推理是 ToT 思想内化的结果
</callout>

---

## 延伸阅读

- [Chain-of-Thought 深度解读]({% post_url 2026-04-24-Chain-of-Thought-深度解读 %}) —— ToT 的线版本
- [ReAct 深度解读]({% post_url 2026-04-24-ReAct-推理与行动交替深度解读 %}) —— 推理 + 行动
- [Reflexion 深度解读]({% post_url 2026-04-24-Reflexion-反思型Agent深度解读 %}) —— 反思增强
- [Graph of Thoughts (Besta et al., 2023)](https://arxiv.org/abs/2308.09687) —— ToT 的图扩展
- [LATS (Zhou et al., 2023)](https://arxiv.org/abs/2310.04406) —— ToT + MCTS + Reflexion 集成
