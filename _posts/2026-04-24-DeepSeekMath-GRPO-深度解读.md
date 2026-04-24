---
title: "DeepSeekMath & GRPO — 去掉 critic 网络的 PPO 变体,成为 R1 的算法基座"
date: 2026-04-24 17:15:00 +0800
categories: [Agent, Reasoning, Reinforcement Learning]
tags: [grpo, deepseekmath, rlhf, ppo, shao-2024]
math: true
---

## 基本信息

- **作者**: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
- **机构**: DeepSeek-AI, Tsinghua, Peking University
- **发表**: arXiv 2024-02
- **arXiv**: [2402.03300](https://arxiv.org/abs/2402.03300)

## 一句话总结

DeepSeek 发布的 **DeepSeekMath 7B**——数学能力接近 GPT-4 的开源模型。但这篇论文真正改变行业的是其中提出的 **GRPO (Group Relative Policy Optimization)**——一个**去掉 critic 网络**的 PPO 变体:对同一个 prompt **采样 $G$ 个 output**,用**组内相对得分**作为 advantage,完全无需单独训练 value function。这让 RL 训练数学/代码类"可验证奖励"任务变得**内存减半、实现极简**,且收敛稳定。GRPO 后来成为 **DeepSeek-R1 的核心算法**,开启了"纯 RL 涌现推理"的新范式,影响了 o1-style 全部开源复现工作。

![DeepSeekMath 7B 在 MATH benchmark 上达到 51.7%,逼近 GPT-4 的 52.9% 和 Gemini Ultra 的 53.2%。开源 7B 模型首次能对标前沿闭源模型的数学能力。](/assets/img/grpo/x1.png)
_Figure 1:DeepSeekMath 7B 的数学能力_

---

## 背景:PPO 在 LLM 上的痛点

### PPO 的标准形式

RLHF 的主流算法 PPO 需要**两个网络**:

- **Policy**:被训练的 LLM 本身
- **Value network (Critic)**:估计每个 state 的价值,用于计算 advantage

$$
A_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Value network 通常是另一个 LLM,大小与 policy 相当。这意味着:

- **显存开销翻倍**(两个 LLM 同时在 GPU)
- **训练不稳**:value 学不好会影响整个 RL

### DeepSeek 团队的观察

对于**数学题这种有明确正确答案**的任务:

- 最终答案对/错是**清晰 reward**
- 中间步骤的"价值"其实不是关键
- Value network 的存在更像是技术债

能不能**完全不要 value network**?

---

## 核心机制:Group Relative Policy Optimization

### 核心思想:用组内均值代替 value

对同一个 prompt,policy 采样 $G$ 个 completions(典型 $G = 8$ 或 16),得到 $G$ 个 reward $\{r_1, ..., r_G\}$。

GRPO 把每个 completion 的 advantage 定义为**相对于组内均值的标准化 reward**:

$$
A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}
$$

含义:**比同批次平均好的 completion 获得正 advantage,比平均差的获得负 advantage**。

这等效于把 "baseline" 从 value network 的输出换成了 **组内均值**——一个完全由当前 batch 计算出的统计量,无需额外模型。

### 优势

- **无需 critic**:显存开销砍半
- **实现简单**:几十行代码
- **baseline 自动更新**:每个 batch 重新算 mean/std,自然适应 policy 进步
- **对 variance 鲁棒**:std 归一化让不同任务难度下的信号 comparable

### 完整 loss

GRPO 的 loss 保留 PPO 的 clipped ratio + KL 正则:

$$
\mathcal{L} = -\mathbb{E}\!\left[\min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i)\right] + \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

其中 $\rho_i = \pi_\theta(o_i) / \pi_{\text{old}}(o_i)$ 是 importance ratio,$\pi_{\text{ref}}$ 是 SFT 模型。

### 与 RLOO / ReMax 的关系

GRPO 的思想和 **RLOO (REINFORCE Leave-One-Out)** 非常相似——两者都用组内对比做 baseline。主要区别:

- RLOO 用 leave-one-out 均值;GRPO 用全组均值 + std 归一化
- GRPO 保留 PPO 的 clipping 机制

---

## DeepSeekMath 的整体训练配方

![DeepSeekMath 的训练流程:预训练用 120B 数学语料 → SFT on MATH-Instruct → 用 GRPO 做 RL on MATH problems 验证奖励。每一阶段都带来显著提升。](/assets/img/grpo/x2.png)
_Figure 2:DeepSeekMath 三阶段训练_

### 1. 数学预训练(120B tokens)

- 从 Common Crawl 中用 fastText classifier 筛选数学相关网页
- 总量 **120B tokens** 数学语料
- 在 DeepSeek-Coder-Base 上继续预训练

这是 DeepSeekMath 质量的基础——高质量数学预训练数据。

### 2. SFT

使用 MATH instruction-tuning 数据(包括链式推理):

- MATH、GSM8K 等训练集
- 从 MetaMath 等扩展数据

### 3. RL with GRPO

- Reward function: 正确答案 → +1,错 → 0
- 8 个 completions per prompt
- GRPO 更新

### 关键数字

| Stage | MATH | GSM8K |
|-------|------|-------|
| Base | 35.7 | 64.2 |
| + SFT | 46.8 | 82.9 |
| + GRPO RL | **51.7** | **88.2** |

**RL 贡献约 5 分 MATH / 5 分 GSM8K 的提升**——这个数字级别的 RL 收益在当时非常罕见。

---

## GRPO vs PPO 的对比实验

![GRPO 和 PPO 训练曲线对比:GRPO 在 MATH 上收敛速度和最终性能都优于 PPO,且显存开销少 ~40%。](/assets/img/grpo/x3.png)
_Figure 3:GRPO vs PPO 训练曲线_

- **最终 MATH 准确率**:GRPO 51.7,PPO 50.3(+1.4)
- **显存**:GRPO 省 ~40%(无 value network)
- **训练速度**:GRPO 快 30-50%(每 step 少一次 value forward+backward)
- **稳定性**:GRPO 无 value estimation noise,训练曲线更平滑

这些优势让 GRPO 在后续的 R1 等大规模 RL 训练中成为首选。

---

## 工程影响:开启 R1 时代

### 1. DeepSeek-R1 的算法基座

2025 年 1 月 DeepSeek-R1 发布,明确说明使用 GRPO。**R1 的"纯 RL 涌现长 CoT + 反思"奇迹,算法层面的关键就是 GRPO**——稳定、低资源、可规模化。

### 2. 开源推理模型的事实标准

自 2024 年中起,几乎所有开源推理模型复现(Qwen-R1、Mistral-R1、Llama-R1 变体等)都用 GRPO 或其小改版本。PPO 在大模型 RL 上的主导地位被撼动。

### 3. 可验证奖励 + GRPO 的组合

GRPO 在"**环境 reward 清晰(数学答案、代码测试通过)**"的场景工作最好。这启发了一批"agentic RL"工作:

- Search-R1:搜索引擎结果作为 reward
- SWE-RL:SWE-Bench test pass 作为 reward
- ReTool:工具执行结果作为 reward

### 4. 简化 RL 工程栈

DeepSeek 团队开源了 GRPO 实现(在 verl、trl 等库中),让社区轻松复现大规模 RL。这个"**算法工程简化**"效应极大降低了 RL for LLM 的门槛。

---

## 局限

### 1. 需要高方差 reward 才有信号

如果一个 batch 内所有 completions 都全对或全错,GRPO 的 advantage 都是 0,无梯度。需要任务难度恰当。

### 2. 组大小是超参

$G = 4$ 太少,方差大;$G = 32$ 太多,计算成本高。需要调。

### 3. 不适合 preference-only reward

对于 reward 不是数值而是"偏好对"(如 A 好于 B),GRPO 不直接适用——需要先把 preference 转成 reward(例如用 RM)。

### 4. Credit assignment 粒度粗

GRPO 默认把 reward assign 给整个 completion。对长 CoT,中间步骤没有精细信号。DAPO、GSPO 等后续变体试图解决这一点。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Value network 不是 RL 必需品**:组内相对得分是一个简洁有效的 baseline,去掉 critic 省一半显存还能更稳——这是对 PPO 的重要简化
2. **可验证奖励改变 RL 经济学**:数学答案、代码测试、搜索正误等客观信号让 reward 变廉价且无歧义,这让大规模 RL 成为可行方案
3. **GRPO 是 R1 范式的隐形英雄**:R1 的震撼来自模型行为(自我反思、aha moment),但其工程基础是 GRPO 的稳定性和效率
4. **简化算法有时比复杂算法强**:PPO 是 AI 领域的经典,但 GRPO 证明"更简单"未必"更差"——有时恰好相反
</callout>

---

## 延伸阅读

- [DeepSeek-R1 深度解读]({% post_url 2026-04-24-DeepSeek-R1-RL推理模型深度解读 %}) —— GRPO 的最重要应用
- [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) —— 前身算法
- [RLOO (Ahmadian et al., 2024)](https://arxiv.org/abs/2402.14740) —— 同类思想的另一条线
- [DAPO (ByteDance, 2024)](https://arxiv.org/abs/2503.14476) —— GRPO 的改进
- [PRM800K / Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) —— Process Reward 的经典
