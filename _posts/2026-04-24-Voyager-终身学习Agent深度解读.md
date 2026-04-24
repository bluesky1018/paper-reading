---
title: "Voyager — 在 Minecraft 里用 GPT-4 构建一个终身学习的 AI 探险家"
date: 2026-04-24 16:15:00 +0800
categories: [Agent, Lifelong Learning, Embodied]
tags: [voyager, minecraft, lifelong-learning, skill-library, wang-2023]
math: true
---

## 基本信息

- **作者**: Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar
- **机构**: NVIDIA, Caltech, UT Austin, Stanford, UW-Madison
- **发表**: TMLR 2024
- **arXiv**: [2305.16291](https://arxiv.org/abs/2305.16291)

## 一句话总结

提出 **Voyager**——**第一个在 Minecraft 里持续终身学习、自主扩展能力边界** 的 LLM Agent。核心设计:**自动课程(Automatic Curriculum)** 决定下一步学什么,**Skill Library** 存储已学会的可执行代码作为"程序记忆",**Iterative Prompting with Self-Verification** 让 GPT-4 写代码、运行、根据环境反馈改代码直到成功。三者形成终身学习闭环:没有梯度更新、没有人类标注,Voyager **自主解锁物品数量是基线的 3.3×,游戏中行进距离 2.3×**,甚至能把学到的技能 transfer 到新世界。是"lifelong agent"方向的开山之作,也是 embodied agent 的里程碑。

![Voyager 的三大组件:Automatic Curriculum 提出任务、Iterative Prompting 让 GPT-4 写代码并根据环境反馈迭代、Skill Library 把学会的程序存储检索。三者构成终身学习闭环。](/assets/img/voyager/x1.png)
_Figure 1:Voyager 的三组件架构_

---

## 背景:Minecraft 为什么是理想的 embodied benchmark

### 开放世界的特点

Minecraft 是一个**无尽、可交互、多任务**的世界:

- 没有单一目标,玩家自由探索
- 物品体系有丰富的**组合结构**(挖矿 → 冶炼 → 合成工具 → 探索更深)
- 不同环境有不同资源(沙漠、森林、海洋、下界)

这让 Minecraft 成为**终身学习**的完美测试床:能力是可以持续积累的、跨环境的、无上限的。

### 之前方法的局限

- **RL agent**(DreamerV3 等):在 Minecraft 上花 100M+ step 训练,只能学有限任务
- **Behavior Cloning**:需要大量人类演示数据
- **单次 LLM prompt**:GPT-4 能写 Minecraft 代码但不会"记住"并复用

Voyager 的目标:**让 GPT-4 自主学习,在不训练任何模型的前提下持续积累能力**。

---

## 核心机制

### 1. Automatic Curriculum(自动课程)

![Automatic Curriculum 让 GPT-4 根据当前状态(库存、周围环境、已学技能)自主提出"下一个合理学的任务",从简单的"收集木头"到复杂的"制作钻石镐"。](/assets/img/voyager/x2.png)
_Figure 2:Automatic Curriculum 示例_

传统 RL 需要人工设计 curriculum。Voyager 让 **GPT-4 看当前状态自己提任务**:

**Prompt:**
> 当前库存:[10 木头, 3 石头]
> 已学技能:[切木头, 制作工具台]
> 周围环境:[森林, 一条小溪]
>
> 下一个对你能力有提升的合理任务是?

GPT-4 提议:"制作一把木镐"→ 进入下一阶段。

**关键特性**:

- **Exploration-exploitation 平衡**:既不太难(避免一直失败),也不太容易(避免停滞)
- **基于当前能力**:已学技能影响下一任务选择
- **可以长程规划**:最终目标"获得钻石"会逐步分解

### 2. Skill Library(技能库)

这是 Voyager 的"长期记忆":**一个存储"已验证可执行代码"的向量库**。

- 每个技能是一个 JavaScript 函数(用 Mineflayer API 操作 Minecraft)
- 技能有自然语言描述(用作 embedding)
- 新任务时按语义相似度检索 top-k 相关技能,作为 context 帮助生成新技能

![Skill Library 的组织:每个 entry 是 (自然语言描述, JavaScript 代码) 对,向量化存储。新任务来时检索相关技能作为 ICL 示例。](/assets/img/voyager/x3.png)
_Figure 3:Skill Library_

这是**程序记忆**(procedural memory)——与 MemGPT 的对话记忆、Generative Agents 的事件记忆形成对比。

### 3. Iterative Prompting with Self-Verification

GPT-4 生成代码后不是一次性的,而是**闭环迭代**:

1. GPT-4 根据 task + retrieved skills 写代码
2. 在 Minecraft 里执行
3. 收集反馈:
   - **环境反馈**(error、状态变化)
   - **Self-verification**:另一个 GPT-4 instance 检查"任务真的完成了吗"
4. 如果没完成,把错误信息加到 prompt,GPT-4 改代码
5. 循环直到成功或达到步数上限

![Iterative Prompting 的闭环:写代码 → 执行 → 反馈 → 改代码。GPT-4 用自己的 self-verification 判断任务是否完成。](/assets/img/voyager/x4.png)
_Figure 4:Iterative Prompting 闭环_

这相当于**用 self-play RL**——但全在 prompt 空间完成,不改权重。

---

## 实验结果

### 物品解锁数量

![Voyager vs ReAct / Reflexion / AutoGPT 的物品解锁曲线:Voyager 在相同时间内解锁了 3.3× 的独特物品。](/assets/img/voyager/x5.png)
_Figure 5:物品解锁进度对比_

在 Minecraft tech tree 上:

| Method | 独特物品解锁 | 行进距离 |
|--------|------------|----------|
| ReAct | 28 | 1.2×baseline |
| Reflexion | 35 | 1.5× |
| AutoGPT | 22 | 0.8× |
| **Voyager** | **93** | **2.3×** |

**3.3× 物品数量**——基线(ReAct/Reflexion/AutoGPT)最多到石器时代,Voyager 已经解锁钻石时代。

### 技能迁移

Voyager 学到的技能可以 transfer 到新的 Minecraft 世界(不同 seed):

- 在原世界学到 100 个技能
- 放到新世界,**直接用这些技能可以快速探索**
- 相当于 "finetune" 了一个 agent,但实际上只是把 skill library 搬过去

这是**可组合、可迁移的智能体能力**的实证。

---

## 工程影响

### 1. 开启 "lifelong agent" 研究方向

Voyager 之前,LLM agent 研究多聚焦单任务或短 horizon。Voyager 证明 **agent 可以持续积累能力**——没有模型更新,纯靠外部存储 + prompt 调度。这启发了 AI-Scientist、Devin、长程 coding agent 等后续工作。

### 2. Skill Library 作为程序记忆

"**把能力编码为可执行代码并存入向量库**"——这个思想在后续 CodeAct、OS-Copilot、Anthropic 的 agent skills 中广泛采用。

### 3. Self-Verification 的重要性

Voyager 证明:**没有环境 ground truth 时,用另一个 LLM 验证也是可行的**。这启发了 Self-Check、Constitutional AI、Process Reward Model 等工作。

### 4. 启发课程学习的自动化

"让 LLM 自己设计课程"成为一个独立方向——VPT、Mind2Web 等后续工作都有类似思想。

---

## 局限

### 1. 依赖强 base model

Voyager 只在 GPT-4 上 work 良好。用弱模型(< GPT-4)各种子组件都会出问题。这再次印证"agent 能力随模型强度涌现"。

### 2. 单模态(文本 + 符号)

Voyager 只看 Minecraft 的文本状态(库存、坐标等),不看图像。对需要视觉感知的任务(如找特定地形)受限。后续 MineDojo、MP5 等把视觉加回来。

### 3. 限定游戏环境

Voyager 靠 Mineflayer API 提供抽象——真实世界没有这么规整的 API。转到真实机器人、桌面操作时需要大量工程。

### 4. Cost 高

数十万次 GPT-4 调用,单次实验花费可观。当时数千美元跑完一个完整 curriculum。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Agent 能力可以积累而不需要训练**:skill library + retrieval + prompt composition,就能实现"终身学习"——这是对"学习 = 更新权重"的重要挑战
2. **程序是一种最强的记忆形式**:Voyager 的 skill 是可执行代码,不是文本也不是向量——代码既可以查询又可以执行,这种"双向"记忆表达力最强
3. **自动课程是 LLM agent 的重要能力**:让模型自己决定学什么 → 自主进步的核心机制。这个思想在 AI-Scientist、Devin 的长程任务规划中反复出现
4. **Self-verification 作为"无环境 ground truth"的代替**:当外部信号缺失时,LLM 对自己的判断也是有用的信号——这是 RLAIF、自监督 RL 等路线的共通思想
</callout>

---

## 延伸阅读

- [ReAct 深度解读]({% post_url 2026-04-24-ReAct-推理与行动交替深度解读 %}) —— Voyager 的 base scaffold
- [Reflexion 深度解读]({% post_url 2026-04-24-Reflexion-反思型Agent深度解读 %}) —— 反思式自我改进
- [Generative Agents (Park et al., 2023)](https://arxiv.org/abs/2304.03442) —— 另一条 embodied agent 路线
- [MineDojo (Fan et al., 2022)](https://arxiv.org/abs/2206.08853) —— Voyager 的前身 Minecraft benchmark
- [AI Scientist (Sakana AI, 2024)](https://arxiv.org/abs/2408.06292) —— 类似思想在科研场景的应用
