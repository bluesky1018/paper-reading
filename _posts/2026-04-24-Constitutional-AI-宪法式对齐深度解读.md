---
title: "Constitutional AI — 用 AI 反馈替代人类反馈,让模型按'宪法'自我改进"
date: 2026-04-24 20:30:00 +0800
categories: [Pretraining, Alignment, Safety]
tags: [constitutional-ai, rlaif, anthropic, bai-2022]
math: true
---

## 基本信息

- **作者**: Yuntao Bai, Saurav Kadavath, Sandipan Kundu, ... (Anthropic)
- **机构**: Anthropic
- **发表**: arXiv 2022-12
- **arXiv**: [2212.08073](https://arxiv.org/abs/2212.08073)

## 一句话总结

Anthropic 提出 **Constitutional AI (CAI)** 与 **RLAIF (RL from AI Feedback)**——**用 AI 自己的反馈代替人类反馈**来做对齐。核心机制:写一份包含若干原则的"**宪法**"(如"不要帮助用户做危险的事"、"回答要诚实"),然后让模型根据宪法**自我批判**并**改写**有害 response,再用这些"改写对"训练 preference model 和 RL。这让对齐流程中**大部分 labeling 从人类转到 AI**,成本降低、速度提升、标注一致性更高。CAI 是 Claude 系列的对齐核心方法,也是 **RLAIF 范式**的奠基作,后续 Gemini、LLaMA 3 等都借鉴这个思路。

![Constitutional AI 的两阶段:(1) SL-CAI——用 AI 批判并改写有害 response,用改写数据做 SFT;(2) RL-CAI——让 AI 打 preference 标签(基于宪法),训 RM,再 PPO。每阶段都用"AI feedback"代替部分"human feedback"。](/assets/img/constitutional-ai/x1.png)
_Figure 1:Constitutional AI 的两阶段流程_

---

## 背景:人类反馈的 bottleneck

### RLHF 的人力瓶颈

InstructGPT 的成功证明 RLHF 有效,但:

- **标注成本高**:几十人全职标注几个月
- **速度慢**:一批数据要 2-4 周
- **一致性差**:不同标注员标准不同
- **扩展性有限**:想做更大规模标注只能招更多人

随着 LLM 能力提升,alignment 所需的**标注规模**也需扩大。人力不是答案。

### Anthropic 的假设

**强 LLM 自己就能判断 "这个 response 是否符合原则"**。那为什么还要人?

于是出现一个闭环想法:**让 AI 自己批判、自己改进**。

---

## 核心机制

### "宪法"(Constitution)是什么

Anthropic 写了一份 **~16 条原则** 的"宪法",例如:

- "Please choose the response that is most helpful, harmless, and honest."
- "Please choose the response that is less harmful, and avoid being insensitive, sexist, racist, or discriminatory."
- "Please choose the assistant response that sounds most similar to what a peaceful, ethical, and wise person like Martin Luther King Jr. would say."

这些原则是 **human 写的**——但**只需要写一次**(几百字),之后 AI 自己用。

### Stage 1: Supervised Learning CAI (SL-CAI)

![SL-CAI 流程:用 helpful-only 模型生成可能有害 response → 用"宪法"提示 AI 批判(critique)→ 让 AI 根据批判重写(revision)→ 用改写后的(prompt, response)对做 SFT。](/assets/img/constitutional-ai/x2.png)
_Figure 2:SL-CAI 的 critique + revision 循环_

具体步骤:

1. **生成 harmful response**:用一个没对齐的 helpful-only 模型对"red team prompts"(可能诱导有害 output 的 prompt)生成 response
2. **Critique**:用另一个 prompt 让模型根据宪法批判自己:
   ```
   Here's a response: [harmful]
   Critique this response based on: "Please choose the response
   that is most helpful, harmless, and honest"
   ```
3. **Revision**:让模型根据 critique 重写:
   ```
   Now rewrite the response to address the critique above.
   ```
4. **SFT**:用 `(prompt, revised_response)` 对做 SFT

**关键**:critique 和 revision 都由 AI 做,**没有人类介入**。宪法原则随机从 ~16 条中抽,让模型学到广义的"安全"而非特定规则。

### Stage 2: RL from AI Feedback (RLAIF)

![RL-CAI 流程:SL-CAI 模型为 red team prompts 生成多个 response → 用 AI(配宪法原则)对 pairs 打 preference → 训 RM → PPO。整个闭环从人类反馈变成 AI 反馈。](/assets/img/constitutional-ai/x3.png)
_Figure 3:RL-CAI——用 AI preference 替代 human preference_

具体:

1. SL-CAI 后的模型对 prompts 生成 **成对 responses** $(y_a, y_b)$
2. 用 AI 判断哪个 response 更符合宪法:
   ```
   Prompt: "..."
   Response A: "..."
   Response B: "..."
   Which response is more <constitutional principle>?
   ```
3. AI 的偏好答案作为 preference label
4. 训 RM,做 PPO

**等价于 InstructGPT 的 RLHF,但 preference 全部来自 AI**——"**RLAIF**"。

---

## 实验结果

### 1. Harmlessness vs Helpfulness 权衡

![在 harmlessness-helpfulness 二维图上:纯 RLHF(helpful-only)很 helpful 但偶尔 harmful;CAI 获得更好的帕累托——同 helpfulness 下更 harmless,同 harmless 下 helpful 不降。](/assets/img/constitutional-ai/x4.png)
_Figure 4:CAI vs RLHF 的帕累托前沿_

关键数字(human evaluation):

| 模型 | Helpfulness | Harmlessness |
|------|------------|--------------|
| Pretrained | -3.3 | -1.4 |
| Helpful-only RLHF | +1.8 | -0.4 |
| **CAI** | **+1.6** | **+1.3** |

CAI **牺牲 0.2 helpful 换 1.7 harmless**——极好的 trade-off。

### 2. AI preference 质量

- AI 对 harmless 维度的 preference 与人类 agreement **90%+**
- 对 helpful 维度 agreement 85%

这是 AI 替代 human 的关键——**agreement 足够高,CAI 才可行**。

### 3. Chain-of-Thought 提升

在 critique 步骤加入 CoT(让 AI "先想再判断"):

- Agreement 从 85% → 93%
- 最终模型的 harmlessness 更好

**"AI critic 需要 chain-of-thought"**——与人类一样。

---

## 历史影响

### 1. Claude 系列的核心方法

Claude 1/2/3/4 的对齐都基于 CAI + RLAIF。Anthropic 官方认为 CAI 是 Claude "helpful, harmless, honest" 的关键。

### 2. RLAIF 范式确立

CAI 之前,用 AI 做 reward 被认为不可靠。CAI 证明**在足够强的 base 下,AI feedback 可以替代人类 feedback**。这催生了:

- **Self-Rewarding LM**(Meta 2024):模型自评生成奖励
- **RLAIF for Summarization** (Google 2023)
- **PRM with LM judge**(OpenAI 2023-24)

### 3. 扩展对齐 scale

人类标注 100K 样本要几个月,AI 标注几小时。**对齐数据规模扩大了 10-100×**,让更细粒度的对齐成为可能。

### 4. "Constitution"的概念

"让 AI 按一个显式的 rule set 运行"思想影响深远:

- **Constitutional Classifier**(Anthropic 2025):用于防御 jailbreak
- **AgentGPT 等产品**:用类似宪法约束 agent
- **OpenAI Model Spec**(2024):定义模型应遵守的规则

### 5. 降低新模型 alignment 成本

新 model 训练后,**用旧 model 做 critic** 可以自动 bootstrap 对齐——不需要重新标数据。这是 alignment 工作的"复利"机制。

---

## 局限

### 1. 对 base model 能力要求高

CAI 依赖 AI 能准确判断"是否符合宪法"。**< 10B 的模型可能判断不准**,CAI 效果差。这限制了 CAI 在小模型上的直接应用。

### 2. 放大 base model 的 biases

如果 AI critic 本身有偏见(如对某些话题更保守),这些偏见会**在 CAI 过程中放大**——AI 批判 AI 会共振出极端立场。

### 3. "宪法"的价值观选择

宪法条目由 Anthropic 团队写——**隐含着他们的价值观**。这些价值观会被植入 Claude。对多元化的全球用户,是否合适是 open question。

### 4. Reward hacking 仍可能

AI 可能学会"满足宪法 critic"的某种 pattern,而非真的变得 harmless。这与 PPO 的 reward hacking 类似,只是对象从 RM 变成 AI critic。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **AI 可以监督 AI**:这是 CAI 最根本的观念突破。之前大家觉得 "AI 不够可信,一定要 human 在 loop",CAI 证明强 AI + 好 rule 可以替代大部分 human 标注
2. **"Constitution"是一个工程化的价值观注入机制**:把价值观写成显式条款而非训练数据——这让对齐的目标清晰、可审计、可迭代
3. **CAI + CoT 是 AI feedback 的关键**:让 AI 在评估前"想一想",agreement 大幅上升。这个模式在后续的 LM-as-judge、PRM 等工作中反复出现
4. **RLAIF 是 alignment 的 scalable 解**:成本降 10×,速度升 10×,覆盖范围扩大——这让"全方位对齐"成为经济可行。Claude 的差异化很大程度上来自这个方法论
</callout>

---

## 延伸阅读

- [InstructGPT 深度解读]({% post_url 2026-04-24-InstructGPT-RLHF三阶段对齐深度解读 %}) —— RLHF 原版
- [DPO 深度解读]({% post_url 2026-04-24-DPO-直接偏好优化深度解读 %}) —— 简化 PPO
- [RLAIF for Summarization (Lee et al., 2023)](https://arxiv.org/abs/2309.00267) —— RLAIF 在其他任务
- [Self-Rewarding LM (Yuan et al., 2024)](https://arxiv.org/abs/2401.10020) —— 模型自评更极端
- [Anthropic Responsible Scaling Policy](https://www.anthropic.com/news/anthropics-responsible-scaling-policy) —— CAI 的更广框架
