---
title: "Kaplan Scaling Laws — 第一次把大模型性能写成精确的幂律公式"
date: 2026-04-24 19:15:00 +0800
categories: [Pretraining, Scaling Law]
tags: [scaling-laws, kaplan, power-law, kaplan-2020]
math: true
---

## 基本信息

- **作者**: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
- **机构**: OpenAI, Johns Hopkins
- **发表**: arXiv 2020-01
- **arXiv**: [2001.08361](https://arxiv.org/abs/2001.08361)

## 一句话总结

OpenAI 的里程碑论文——**第一次系统化地把大模型性能写成精确的数学公式**。核心发现:对 Transformer 语言模型,**loss 与参数量 N、数据量 D、算力 C 之间严格遵循幂律(power law)关系**,且**这些关系跨越 7 个数量级都稳定**。具体:$L(N) \propto N^{-0.076}$,$L(D) \propto D^{-0.095}$,$L(C) \propto C^{-0.050}$。这意味着**可以在小模型上做实验,外推到大模型的性能**——scaling 从"经验"变成"可计算的工程问题"。这直接促成了 GPT-3 的诞生(OpenAI 依据这些公式"算出" 175B 是最佳规模),也是 2020 年代所有大模型训练决策的理论基础。

![Loss 与参数 N、数据 D、算力 C 的幂律关系:跨 7 个数量级的 compute,曲线都是直线(log-log 坐标)——这是"幂律规律"的强实证。](/assets/img/kaplan-scaling/x1.png)
_Figure 1:Loss 的三大幂律——N / D / C_

---

## 背景:scaling 以前是玄学

2019 年前后,训练大模型像是"**经验 + 运气**":

- "这个模型多大合适?" —— 凭感觉
- "训多久够了?" —— 跑 loss 曲线看
- "加倍 compute 带来多少 loss 下降?" —— 不知道

OpenAI 团队想:既然我们要训 GPT-3,**应该有科学方法指导这些决策**。

他们的假设:**loss(N, D, C) 有预测性的数学形式**。这个想法在物理学里很常见(如 scaling law of turbulence),在深度学习里第一次被系统化验证。

---

## 三条核心定律

### 1. Loss vs 参数量 N(数据充足)

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N},\quad \alpha_N \approx 0.076,\; N_c \approx 8.8 \times 10^{13}
$$

- $N$ 是非 embedding 参数量
- **参数加倍,loss 降 $2^{0.076} - 1 \approx 5.4\%$**

### 2. Loss vs 数据量 D(参数充足)

$$
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D},\quad \alpha_D \approx 0.095,\; D_c \approx 5.4 \times 10^{13}
$$

- **数据加倍,loss 降 $\approx 6.8\%$**

### 3. Loss vs 算力 C(最优配置)

$$
L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C},\quad \alpha_C \approx 0.050
$$

- **算力加倍,loss 降 $\approx 3.5\%$**

---

## 惊人的跨数量级规律

![Loss 随 compute 的变化在 log-log 坐标下是几乎完美的直线,跨越从小到大 7 个数量级的 compute。这种"跨 10^7 级的线性"在深度学习中极为罕见。](/assets/img/kaplan-scaling/x2.png)
_Figure 2:跨 7 个数量级的 loss-compute 关系_

关键观察:

- 从 $10^{-6}$ PF-days 到 $10^2$ PF-days,**同一条直线**
- 不同模型大小、不同数据规模、不同架构变体,都在这条线上
- 这意味着**可以在小规模实验外推到大规模**——巨大的工程价值

---

## 实用推论

### 1. Compute 的最优分配

给定 total compute $C$,作者推导出:

$$
N_{\text{opt}} \propto C^{0.73},\quad D_{\text{opt}} \propto C^{0.27}
$$

意思:**compute 加倍时,参数应该扩大 $2^{0.73} \approx 1.66\times$,数据只扩大 $2^{0.27} \approx 1.21\times$**。

这个"**大模型 + 少数据**"的结论直接影响了 GPT-3 的设计:175B 参数 + 300B tokens(比例 1:1.7)。

### 2. 架构相对不重要

作者比较 Transformer 的各种变体(不同层数、width、heads 等):

- **给定 N,架构细节对 loss 影响 <2%**
- **只要 N、D、C 规模对,架构细节可以忽略**

这是一个震撼结论——**架构优化往往抵不过 scale**。

### 3. Overfitting 的预测

![当数据 D 不够时 loss 会 plateau(过拟合);当 D 足够时 loss 按 N 的幂律下降。作者给出了一个"critical D"公式,指导什么时候需要更多数据。](/assets/img/kaplan-scaling/x3.png)
_Figure 3:Overfitting 区域的 scaling law_

---

## Chinchilla 对 Kaplan 的修正

2022 年 DeepMind 的 **Chinchilla** 工作挑战 Kaplan 的结论:

- Kaplan 说 $N_{\text{opt}} \propto C^{0.73}$,暗示"大模型比大数据更重要"
- Chinchilla 重做实验,发现 $N_{\text{opt}} \propto C^{0.5}$ 和 $D_{\text{opt}} \propto C^{0.5}$——**参数和数据应 1:1 等比扩展**
- 即 **N:D 最优比是 1:20**(1B 参数配 20B tokens)

Kaplan 的偏差来自实验设置(LR schedule、batch size 等)。Chinchilla 成为后续的标准。

但 Kaplan 的**方法论和大的 framework 完全正确**——都是幂律,都是可预测,只是指数系数不同。

---

## 历史影响

### 1. GPT-3 的理论基础

GPT-3(2020)的 175B 参数选择不是拍脑袋——是**按 Kaplan 公式"算"出来的**。这是第一个严格基于 scaling law 设计的模型。

### 2. 开创可预测的 ML 工程

以前 ML 是"跑起来看看",Kaplan 之后可以做:
- **小模型试验 + 大模型预测**(节省 compute)
- **训练预算规划**(我有 $1M compute,应该多大模型 + 多少数据?)
- **性能外推**(GPT-4 大致会是什么性能?)

### 3. 启发后续 scaling law 研究

- **Chinchilla**:修正比例
- **Hoffmann scaling for optical** / **Inverse scaling**:反例研究
- **Scaling laws for RL**(R1 等)
- **Scaling laws for multimodal**

scaling law 成为 ML 研究的一个独立子领域。

### 4. 影响大公司算力规划

Google / OpenAI / Meta 训练下一代模型前,都会做 Kaplan-style 小规模实验,拟合 scaling law,再决定主力训练的 N、D、C。**没有 scaling law,就没法理性地花几千万美元训一次模型**。

---

## 局限

### 1. 外推风险

实验范围是 $10^6 - 10^{10}$ 参数。外推到 $10^{12}$ 是假设——可能某个规模 scaling law 会 break。

### 2. 只看 loss

Kaplan 的 law 预测 cross-entropy loss,但**下游任务性能**(QA、reasoning 等)不一定完全按 loss 走。**Emergent abilities**(涌现能力)就是 loss 连续下降但 task performance 阶跃上升的现象。

### 3. 没考虑推理成本

Kaplan 的最优是"固定训练 compute 的最优",没考虑"模型训好后要部署"。Beyond Chinchilla-Optimal (2024) 修正了这点——推理成本考虑进来后,**应该 overtrain 小模型**。

### 4. 架构"无关"的假设有限

Kaplan 的结论基于 "vanilla Transformer"。**MoE、Mamba、Linear Attention** 等新架构有不同的 scaling law——架构还是重要的,只是在 vanilla Transformer 家族内不重要。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **ML 可以是物理**:loss 与 N/D/C 的幂律关系是 ML 第一次有了"基本定律"级别的预测规律。这个框架让 ML 研究可以像物理一样做外推实验
2. **Scale 是架构的十倍重要**:架构细节只影响 <2% loss,scale 影响数量级。这个观察驱动整个 2020-2025 的"暴力 scale"研究路线
3. **Scaling law 是可 transfer 的工程工具**:小模型实验 → 拟合 scaling law → 外推到大模型。这是大厂花几千万美元训模型前必做的工作
4. **定律会被修正,框架不会**:Kaplan 的具体指数被 Chinchilla 修正,但"幂律 + 可预测 + 可外推"的 framework 稳固至今
</callout>

---

## 延伸阅读

- [Chinchilla 深度解读]({% post_url 2026-04-24-Chinchilla-Compute最优训练深度解读 %}) —— Kaplan 的修正
- [GPT-3 深度解读]({% post_url 2026-04-24-GPT-3-语言模型是小样本学习者深度解读 %}) —— Kaplan 指导的第一个大规模产物
- [Beyond Chinchilla-Optimal (2024)](https://arxiv.org/abs/2401.00448) —— 考虑推理成本的修正
- [Emergent Abilities (Wei et al., 2022)](https://arxiv.org/abs/2206.07682) —— 对 Kaplan 连续性的反例
- [LLaMA 3 深度解读]({% post_url 2026-04-24-LLaMA-3-405B开源大模型深度解读 %}) —— 应用 scaling law 的现代例子
