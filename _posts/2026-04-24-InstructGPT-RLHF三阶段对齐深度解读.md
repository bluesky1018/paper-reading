---
title: "InstructGPT — 把 GPT-3 用 RLHF 对齐成'听话'的 AI,ChatGPT 的技术底座"
date: 2026-04-24 20:00:00 +0800
categories: [Pretraining, Alignment, RLHF]
tags: [instructgpt, rlhf, alignment, ouyang-2022]
math: true
---

## 基本信息

- **作者**: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida 等
- **机构**: OpenAI
- **发表**: NeurIPS 2022
- **arXiv**: [2203.02155](https://arxiv.org/abs/2203.02155)
- **全名**: *Training language models to follow instructions with human feedback*

## 一句话总结

OpenAI 的里程碑——**第一次系统化地把 RLHF (Reinforcement Learning from Human Feedback) 用到大语言模型**,把 GPT-3 从一个"会讲故事但不听话"的预训练模型,转变成一个"follow user instructions"的助手。核心流程是 **SFT → RM → PPO 三阶段**:(1) 用人工标注的 instruction-response 数据 **SFT**;(2) 用人工偏好对训练 **reward model**;(3) 用 PPO 根据 RM 信号 fine-tune SFT 模型。关键发现:**1.3B 参数的 InstructGPT 胜过 175B 的原始 GPT-3**——证明 alignment 比 scale 更影响用户感受。这篇论文是 **ChatGPT 的技术底座**(ChatGPT 基本就是 InstructGPT 的对话优化版),也定义了整个 LLM 时代的 post-training 范式。

![RLHF 三阶段:(1) 收集 demonstration 数据做 SFT;(2) 收集 comparison 数据训 reward model;(3) 用 PPO 在 RM 信号下 fine-tune SFT 模型。每阶段都需要 human 在 loop 中。](/assets/img/instructgpt/x1.png)
_Figure 1:RLHF 三阶段流程图_

---

## 背景:GPT-3 虽强但不"好用"

### GPT-3 的 alignment 问题

2020-2022 年期间,GPT-3 API 的用户反馈:

- "它不 follow 我的指令"——问它写邮件,它给段文学独白
- "它编造事实"——自信地给错误答案
- "它可能有害"——会产出偏见、危险内容
- "输出啰嗦或不准确"

核心问题:**GPT-3 的训练目标是"预测下一 token",不是"帮用户"**。这两个目标经常不一致。

### 已有的尝试

- **Prompt engineering**:改 prompt 让 GPT-3 更 helpful。但手动调耗时,且上限低
- **Fine-tune on demonstrations**:有效但只到一定水平,模型不懂"什么是更好的"
- **RL from human preferences**(Christiano 2017):在 summarization 等小任务验证过,但没用到大模型

InstructGPT 的目标:**把 RLHF scale 到 LLM,证明它能 systematic 解决 alignment 问题**。

---

## 核心机制:RLHF 三阶段

### Stage 1: SFT(Supervised Fine-tuning)

![SFT 数据示例:人工写的 instruction + response 对。比如 "Write a short poem about the moon" 配一首诗。用这些数据 supervised fine-tune 预训练 GPT-3。](/assets/img/instructgpt/x2.png)
_Figure 2:SFT 数据和流程_

**数据**:
- OpenAI API 用户实际 prompt(脱敏)
- 人工标注员写 high-quality response
- 共 **~13K prompts + responses**

**训练**:
- GPT-3 175B 为 base
- 标准 supervised LM objective
- 16 epochs

结果:**SFT 模型已经比 GPT-3 好用很多**——这个 baseline 就叫 SFT。但仍有很多不足。

### Stage 2: Reward Model(RM)

**数据**:
- 让 SFT 模型对每个 prompt 生成 K 个 completions(K=4-9)
- 人工标注员对这些 completions **两两比较**("A 比 B 好")
- 共 **~33K comparisons**

**训练**:
- 从 SFT 模型初始化,改最后一层为 scalar 输出
- Loss:Bradley-Terry 偏好模型:

$$
L(\theta) = -\log\sigma(r_\theta(x, y_w) - r_\theta(x, y_l))
$$

其中 $y_w$ 是 winner,$y_l$ 是 loser。

**关键**:训一个 **6B RM**(不用 175B),因为:
- 6B 够准
- PPO 阶段要频繁 query RM,小 RM 快得多

### Stage 3: PPO RL

![PPO 阶段:policy(正在优化的模型)生成 completion → RM 打分作为 reward → PPO 更新 policy。加入 KL penalty 防止偏离 SFT 太远。](/assets/img/instructgpt/x3.png)
_Figure 3:PPO RL 的闭环_

**数据**:
- API prompts(~31K)——无需 human annotation
- Policy 生成 completion
- RM 打分作为 reward

**Loss**:
$$
\mathcal{L} = \mathbb{E}[r_\theta(x, y) - \beta \log\frac{\pi^{RL}(y|x)}{\pi^{SFT}(y|x)}]
$$

- 第一项:最大化 RM 奖励
- 第二项:**KL penalty** 防止 policy 偏离 SFT 太远("reward hacking" 防御)

**变种 PPO-ptx**:额外加一项 "pretrain mixing loss",让模型不忘预训练知识:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{PPO} + \gamma \cdot \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}}[\log \pi^{RL}(x)]
$$

这让 InstructGPT 保留通用知识,不会因 RLHF 变蠢。

---

## 实验结果

### 1. 质量:1.3B InstructGPT > 175B GPT-3

![人类评估员偏好调查:1.3B InstructGPT(PPO)的 output 被人类评估员选择的频率(85%),远超 175B GPT-3 base(基准 50%),也超过 175B 的 SFT(74%)。即 135× 小的 InstructGPT 打败 GPT-3 base。](/assets/img/instructgpt/x4.png)
_Figure 4:人类偏好——InstructGPT 碾压 GPT-3_

关键结果:

| 模型 | 人类偏好 |
|------|---------|
| GPT-3 175B | 50%(基准) |
| GPT-3 175B + prompt | 56% |
| SFT 175B | 74% |
| **PPO 1.3B** | **85%** |
| **PPO 175B** | **88%** |

**1.3B InstructGPT 吊打 175B GPT-3**——alignment 贡献远大于 100× scale。

### 2. Helpfulness / Honesty / Harmlessness

- **Helpfulness**:InstructGPT 更 follow 指令
- **Honesty** (TruthfulQA):InstructGPT 减少幻觉约 50%
- **Harmlessness**:毒性输出减少 25%

但不完美——**有些 trade-off**(如对某些测试集 PPO-ptx 的 academic benchmark 略降)。

### 3. 分布外泛化

![训练时只用英文 instruction,测试时 InstructGPT 能 follow 非英文指令、code 任务等。这暗示 RLHF 不是在学具体任务,而是在学"follow instruction"这个 meta-capability。](/assets/img/instructgpt/x5.png)
_Figure 5:InstructGPT 的分布外泛化_

InstructGPT 能 follow 训练集中没有的 instruction 类型(代码、非英语等)——**RLHF 不是背具体答案,是学"服从"这个 meta-行为**。

---

## 历史影响

### 1. ChatGPT 的直接前身

2022-11-30 发布的 ChatGPT,技术上就是 **InstructGPT + 对话优化**:

- Base: GPT-3.5(比 InstructGPT 的 GPT-3 更好)
- SFT 数据扩展到多轮对话
- RLHF 流程基本相同

ChatGPT 引爆 AI 大众化,其技术底座就是这篇论文。

### 2. RLHF 成为所有 LLM 产品的标配

2022-2024 年,所有主流 LLM(Claude、Gemini、Llama 2/3、DeepSeek 等)都用 RLHF 或变体:

- **Claude**:Constitutional AI(RLAIF)
- **Gemini**:RLHF
- **LLaMA 2/3**:RLHF + iterative DPO

### 3. 开启 Post-training 时代

Pretraining 只是 LLM 的一半,**post-training 是另一半**。InstructGPT 让业界意识到:

- 一个好 LLM 产品 = 好 pretrain + 好 post-train
- Post-train 的 cost 虽低于 pretrain,但**决定用户感受**
- Human-in-the-loop 标注变成 AI 公司的核心能力

### 4. 催生 DPO / RLAIF 等简化方案

PPO 的复杂性促使简化研究:

- **DPO** (2023):直接 preference 优化,无需 RM 和 PPO
- **RLAIF / Constitutional AI** (2022):用 AI 替代 human 反馈
- **SimPO / ORPO**:更简洁的变体

这些都是对 InstructGPT 的回应和简化。

### 5. 人类标注员成为 AI 的"老师"

InstructGPT 用了 40 多个标注员,产出 ~46K 标注。这让"**标注员**"成为 AI 发展的关键环节——OpenAI、Anthropic、Scale AI 等公司投入巨资构建标注团队。

---

## 局限

### 1. RLHF 的 reward hacking

RM 不是完美的,policy 会找 RM 的 bug:

- 输出啰嗦(RM 可能偏爱长 response)
- 过度 disclaimer("As an AI,...")
- 避免直接回答风险问题

这些都是 "reward hacking" 的表现。

### 2. 标注成本高

~40 个标注员 + OpenAI 研究员 × 数月 = 不便宜。DPO 等简化方案就是想降低这个成本。

### 3. Alignment tax

某些 benchmark 上 InstructGPT 比 GPT-3 更差——**alignment 牺牲了一些能力**。这让"**aligned 模型 vs 原始模型谁更强**"成为持续争议。

### 4. 文化偏见

标注员主要来自北美。模型的 "偏好" 反映他们的文化 norm,不一定 global 适用。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Alignment > Scale**:1.3B InstructGPT 胜过 175B GPT-3——对 **用户感受**,对齐比参数更重要。这个发现改变了整个行业对 "什么是好模型" 的定义
2. **SFT → RM → PPO 三阶段是 LLM post-training 的原型**:之后几年的 DPO、GRPO、Iterative DPO 都是这个 pipeline 的变种简化
3. **RM 的信号可以 scale**:PPO 阶段不再需要 human,用 RM 代替。这让 RL 阶段的数据可以无限(虽然 RM 本身有限)——这是 RLHF 实用的关键
4. **KL penalty 是 RL 不崩的关键**:让 policy 不偏离 SFT 太远,平衡 "优化 reward" 和 "保留原能力"。这个思想延续到 GRPO、DPO(后者通过 closed-form 隐式实现相同效果)
</callout>

---

## 延伸阅读

- [GPT-3 深度解读]({% post_url 2026-04-24-GPT-3-语言模型是小样本学习者深度解读 %}) —— InstructGPT 的 base
- [DPO 深度解读]({% post_url 2026-04-24-DPO-直接偏好优化深度解读 %}) —— InstructGPT 的简化
- [Constitutional AI 深度解读]({% post_url 2026-04-24-Constitutional-AI-宪法式对齐深度解读 %}) —— RLAIF 路线
- [DeepSeekMath GRPO 深度解读]({% post_url 2026-04-24-DeepSeekMath-GRPO-深度解读 %}) —— 进一步简化的 PPO
- [LLaMA 3 深度解读]({% post_url 2026-04-24-LLaMA-3-405B开源大模型深度解读 %}) —— 现代 post-training 应用
