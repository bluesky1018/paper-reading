---
title: "DeepSeek-R1 — 用纯 RL 让模型自发涌现长思维链,开源复现 o1"
date: 2026-04-24 17:30:00 +0800
categories: [Agent, Reasoning, Reinforcement Learning]
tags: [deepseek-r1, grpo, reasoning-model, rl, deepseek-2025]
math: true
---

## 基本信息

- **作者**: DeepSeek-AI 团队
- **机构**: DeepSeek
- **发表**: arXiv 2025-01
- **arXiv**: [2501.12948](https://arxiv.org/abs/2501.12948)

## 一句话总结

**DeepSeek-R1** 是 2025 年 1 月震动整个 AI 行业的论文——首个**在开源模型上完整复现 OpenAI o1** 的工作,并给出所有技术细节。核心贡献三点:(1) **R1-Zero**:在 DeepSeek-V3-Base 上**纯用 GRPO RL(完全不做 SFT)训练**,模型自发涌现长思维链、自我反思、验证、回溯等行为;(2) **R1**:在 R1-Zero 基础上做 cold-start SFT + 多阶段 RL,进一步提升可读性;(3) **Distillation**:把 R1 的推理能力蒸馏到 1.5B-70B 的开源小模型,让 Qwen-7B 这种小模型在 AIME 上超过 GPT-4o。R1 发布后一周内引发全球复现浪潮,让"推理模型"从 OpenAI 专属变为开源社区公用技术。

![DeepSeek-R1 的训练 pipeline:R1-Zero 走纯 RL 路线,展示涌现能力;R1 多阶段训练(cold-start SFT → RL for reasoning → SFT on rejection sampling → RL for all scenarios),产出面向用户的最终模型。](/assets/img/deepseek-r1/x1.png)
_Figure 1:R1 的完整训练 pipeline_

---

## 背景:o1 的谜与开源复现之难

### 2024 年 9 月:o1 震动业界

OpenAI 发布 **o1**:在推理任务(AIME、Codeforces、PhD-level science)上大幅超越 GPT-4o。特点:

- 回答前先写一大段内部思考
- 测试时越"想"越久越准(**inference-time scaling**)
- 自发展现反思("wait, let me reconsider")、验证、回溯

但 OpenAI 几乎没公开任何训练细节——只提到"用了 RL"。业界疯狂猜测:

- 是 MCTS + PRM 吗?
- 还是 RLHF 变体?
- 数据怎么造的?

### 2024 年 9-12 月:复现尝试皆不理想

- 社区尝试 Self-Taught Reasoner (STaR) 路线 —— 效果差
- PRM + MCTS —— 复杂且不稳
- 各种 distillation —— 能力浅层,缺涌现

直到 R1 发布,业界才看到一个**work 的 recipe**。

---

## 核心机制一:R1-Zero 的纯 RL 范式

### 算法:GRPO + rule-based reward

DeepSeek 的做法惊人地简单:

- **Base model**:DeepSeek-V3-Base(671B MoE,未经任何 SFT)
- **RL 算法**:GRPO(见 [GRPO 深度解读]({% post_url 2026-04-24-DeepSeekMath-GRPO-深度解读 %}))
- **Reward**:规则化两部分
  - **Accuracy reward**:数学答案对 → +1,代码 AC → +1
  - **Format reward**:回答放在 `<think>...</think>` 和 `<answer>...</answer>` 里 → +0.1
- **Prompt**:简单模板,让模型先 think 再 answer

**完全不做 SFT**,直接开 RL。

### 惊人发现:涌现行为

![R1-Zero 在 AIME 2024 上的准确率随训练 step 平稳上升,从 ~15% 提到 ~70%,中间没有明显平台期。伴随准确率上升,每个 response 的长度也从 ~500 tokens 增加到 ~10000 tokens——模型自发学会了"想得更久"。](/assets/img/deepseek-r1/x2.png)
_Figure 2:R1-Zero 训练期间的 AIME 准确率与响应长度_

训练过程中,**模型自发出现以下行为**(没有任何人示范):

1. **响应变长**:从 500 tokens 逐渐增加到 10000+ tokens
2. **自我反思**:出现 "Wait, let me reconsider..."、"Hmm, that doesn't seem right"
3. **回溯**:会自己否定之前的思路、重开
4. **验证**:"Let me check this calculation..."
5. **多解法尝试**:对同一题尝试多种解法,交叉验证

![R1-Zero 的响应长度随训练 step 持续增加,从 500 tokens 长到 10000+ tokens。这是"test-time compute 自然扩张"的有力证据——模型自发学会"想得越久越准"。](/assets/img/deepseek-r1/x3.png)
_Figure 3:响应长度的"aha moment"涌现_

### "Aha Moment"

论文里最传奇的描述:训练中某个 step 突然观察到模型说出 **"Wait, wait. Wait. That's an aha moment I can flag here."**——仿佛它在解题中真的"意识到"了什么。

这是一个关键 insight:**涌现不仅是能力级的涌现,也是"元认知行为"的涌现**。模型没有被教过"反思",但在 RL 优化下自发学会了这种行为,因为它是提高 accuracy reward 的有效策略。

---

## 核心机制二:R1 的多阶段训练

R1-Zero 能力强但**可读性差**——响应混合中英文、格式混乱、有时难读。R1 是为用户可用而做的精修版本:

### 4 阶段训练

1. **Cold-start SFT**:收集几千条**高质量长 CoT 示例**(部分来自 R1-Zero 的输出,过滤后人工改写)SFT 一下 base model。让模型学一个"体面的"推理格式起点
2. **Reasoning-focused RL**:类似 R1-Zero 的纯 RL,但加入**语言一致性 reward**(鼓励单一语言)
3. **Rejection Sampling + SFT**:用阶段 2 的模型生成 800K 推理 + 200K 通用数据,只保留正确的做 SFT
4. **All-scenarios RL**:用 reward model 对所有场景(包括闲聊、写作)做 RL,补齐通用能力

### 性能

| Benchmark | DeepSeek-V3 | **R1** | OpenAI o1 |
|-----------|------------|--------|-----------|
| AIME 2024 | 39.2 | **79.8** | 79.2 |
| MATH-500 | 90.2 | **97.3** | 96.4 |
| GPQA Diamond | 59.1 | **71.5** | 75.7 |
| Codeforces rating | 1134 | **2029** | 2061 |
| SWE-bench Verified | 42.0 | **49.2** | 48.9 |

**R1 在数学、代码、推理上全面持平或略超 o1**。

---

## 核心机制三:蒸馏小模型

DeepSeek 把 R1 的推理能力**蒸馏到多种开源小模型**:

| Student | AIME 2024 | MATH-500 |
|---------|-----------|----------|
| Qwen 1.5B | 28.9 | 83.9 |
| Qwen 7B | 55.5 | 92.8 |
| Qwen 14B | **69.7** | **93.9** |
| Qwen 32B | **72.6** | **94.3** |
| Llama 70B | **70.0** | **94.5** |

**14B / 32B 级别小模型在 AIME 上超过 GPT-4o**——这是蒸馏效率的震惊展示。

蒸馏过程很直接:让 R1 生成大量推理数据,用这些数据 SFT 小模型。

---

## 工程影响

### 1. "纯 RL 涌现推理"范式确立

R1-Zero 证明:**不需要复杂 MCTS / PRM / MCTS rollout,不需要 SFT 启动,GRPO 纯 RL 就能涌现长 CoT 和反思**。这个结论对 reasoning model 研究范式影响巨大——简化了所有后续复现工作。

### 2. 开源赶上闭源的关键一步

2024 年 o1 让 OpenAI 领先开源 1 个 milestone。R1 一举把差距抹平(甚至在部分任务上领先)。这是开源 LLM 历史上最重要的事件之一。

### 3. 2025 年全球复现浪潮

R1 发布后几周内,复现工作爆炸:
- **Open-R1**(HuggingFace)
- **R1-V**(视觉版)
- **Open-Reasoner-Zero**
- **Simple-RL-Reasoning**
- 各家大模型公司(Qwen、Kimi、GLM、MiniMax)都发布 R1 风格的推理版本

### 4. 对可验证奖励的重视

R1 的成功依赖**可验证 reward**——数学答案、代码 AC。这让业界普遍意识到:**所谓"agentic RL"的核心是找到可验证的 reward signal**。SWE-Gym、Search-R1、ReTool 等工作都沿此思路。

### 5. Inference-time scaling 成为新 scaling 律

R1(继承 o1)证明:**test-time compute 有明显的 scaling curve**——越长越准。这让 scaling law 增加一个新维度:inference compute。对未来模型架构设计有深远影响。

---

## 局限

### 1. 只对可验证任务工作好

R1 最擅长数学、代码、逻辑——这些有明确 reward 的领域。对创意写作、开放对话等主观任务提升有限。

### 2. 响应长度 = 成本

R1 回答长度显著大于 V3(4-10×)。**推理成本相应上升**——对预算敏感场景不一定划算。

### 3. R1-Zero 可读性差

纯 RL 训练出的模型逻辑混乱、格式怪异、中英混杂——只能作为 R1 的前身,不能直接用户使用。

### 4. 对 base model 要求高

R1-Zero 在 V3-Base(671B)上 work;在 < 10B base 上直接纯 RL 几乎没效果。这暗示"涌现需要底座足够强"——这与 CoT 的涌现规律一脉相承。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **涌现不仅是能力的涌现,也是"元认知行为"的涌现**:反思、验证、回溯、多解法尝试——这些曾以为需要专门设计的能力,在 RL 优化下自发出现。这是对 RL 威力的重要证据
2. **简单 reward + 强 base + 好 RL 算法 = 奇迹**:R1-Zero 的 reward 函数只有 "答案对错 + 格式",却涌现出 o1 级别的推理。"让信号干净、让 base 强大"比"设计复杂 reward"更重要
3. **蒸馏是民主化推理能力的关键**:R1 14B 蒸馏模型能打 GPT-4o——说明推理能力可以低成本下沉到小模型。这对应用层是决定性利好
4. **Inference-time scaling 是新的 scaling 律**:过去 scale 参数、scale 数据,现在多了一个 scale test-time compute。这会深刻影响未来模型架构(更长 context、更高效 decode、更好的 tool use)
</callout>

---

## 延伸阅读

- [DeepSeekMath & GRPO 深度解读]({% post_url 2026-04-24-DeepSeekMath-GRPO-深度解读 %}) —— R1 的算法基础
- [DeepSeek-V3 FP8 训练深度解读]({% post_url 2026-04-24-DeepSeek-V3-FP8训练深度解读 %}) —— R1 的 base model
- [Chain-of-Thought 深度解读]({% post_url 2026-04-24-Chain-of-Thought-深度解读 %}) —— 推理的起点
- [Tree of Thoughts 深度解读]({% post_url 2026-04-24-Tree-of-Thoughts-思维树深度解读 %}) —— R1 内化的能力
- [OpenAI o1 官方博客](https://openai.com/index/learning-to-reason-with-llms/) —— R1 对标的闭源前驱
