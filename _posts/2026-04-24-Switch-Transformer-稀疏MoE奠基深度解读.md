---
title: "Switch Transformer — 每 token 只路由到 1 个专家,把 MoE 做到 1.6T 参数"
date: 2026-04-24 21:00:00 +0800
categories: [Pretraining, MoE]
tags: [switch-transformer, moe, sparse, fedus-2021]
math: true
---

## 基本信息

- **作者**: William Fedus, Barret Zoph, Noam Shazeer
- **机构**: Google Brain
- **发表**: JMLR 2022
- **arXiv**: [2101.03961](https://arxiv.org/abs/2101.03961)

## 一句话总结

Google 的 **Switch Transformer**——**把 Sparse MoE 推到 1.6T 参数的奠基之作**。核心简化:把之前 MoE(如 GShard)的 "top-2 routing" 改为 **"top-1 routing"**——每个 token **只路由到 1 个 expert**,计算量折半,通信简化。配合**capacity factor + load balancing loss + selective precision**等工程技巧,成功训出 1.6T 参数的 Switch-C 模型,在同等 compute 下**训练速度比 T5-XXL 快 7×**。Switch Transformer 是现代 Sparse MoE 的技术范式起点,直接影响了 GLaM、PaLM-2、Mixtral、DeepSeek-MoE 等几乎所有后续 MoE 工作。

![Switch 的核心简化:每层 FFN 有 N 个 expert,每 token 只路由到 1 个(top-1),而不是之前的 top-2。这让每 token 计算量减半,通信量也大减。简单但有效的工程决策。](/assets/img/switch-transformer/x1.png)
_Figure 1:Switch 的 top-1 routing_

---

## 背景:MoE 的复杂性

### MoE 的承诺与困境

2017 年 Shazeer 等人提出 **Sparsely-Gated MoE**,证明 MoE 可以在同 compute 下增加模型容量。但:

- **工程复杂**:每层 routing、通信、balance loss,代码量巨大
- **训练不稳**:easy collapse(router 偏向某 expert)
- **通信瓶颈**:多 expert 分布在多 GPU,token 要 all-to-all 传输

**GShard**(2020)把 MoE 推到 600B 参数,但复杂度很高。

### Switch 的简化哲学

Switch 作者(包括 Noam Shazeer)的想法:**能不能让 MoE 变得极简?**

三个简化:

1. **Top-1 routing**(而非 top-2)
2. **简单的负载均衡 loss**
3. **Capacity factor** 控制 token 溢出

---

## 核心机制

### 1. Top-1 Routing

传统 MoE:每 token 选 **top-k (k=2)** 个 expert,结果做加权平均。

Switch:每 token 只选 **1 个 expert**。

优势:
- **计算量减半**(k=1 vs k=2)
- **通信量减半**(每 token 只发一个 expert)
- **更简单的 routing logic**
- **更容易分析和 debug**

反对观点认为 k=2 提供冗余和多样性,Switch 证明 **k=1 够用**——只要其他设计到位。

### 2. Capacity Factor

![Switch 的 capacity factor 设计:每个 expert 有容量上限 $C = \text{capacity factor} \cdot (\text{tokens per batch}) / N$。超过容量的 token 会被"drop"(直接 skip 这层的 FFN,走 residual)。capacity factor 默认 1.25-2.0。](/assets/img/switch-transformer/x2.png)
_Figure 2:Capacity Factor 机制_

问题:即使 router 均匀,batch 内某些 expert 可能"挤"得多,超出该 expert 的容量。

**Capacity factor** $cf$:

$$
C = cf \cdot \frac{\text{tokens per batch}}{N}
$$

- $cf = 1.0$:严格平均,**容易 drop**
- $cf = 1.25$(推荐):5% 冗余
- $cf = 2.0$:大量冗余但计算贵

超容量的 token 被 **drop**(走 residual 跳过这层)——**容忍一点丢失换来简单和速度**。

### 3. Load Balancing Loss

为了让每个 expert 被均匀使用,加一个辅助 loss:

$$
L_{\text{aux}} = N \sum_{i=1}^N f_i \cdot P_i
$$

其中:
- $f_i$ 是分到 expert $i$ 的 token 比例
- $P_i$ 是 router 对 expert $i$ 的平均概率

这个 loss 鼓励 $f_i \cdot P_i$ 小,即每个 expert 使用均匀。加到主 loss 上,系数 ~0.01。

### 4. Selective Precision

![Switch 的 selective precision:router logits 用 bf16 可能不稳,但全用 fp32 又太慢。Switch 把 router 的小部分 (softmax) 留 fp32,其他 bf16——兼顾稳定和效率。](/assets/img/switch-transformer/x3.png)
_Figure 3:Selective Precision_

Router 的 softmax 对精度敏感(需要精确的概率梯度),但其他部分 bf16 没问题。

**选择性精度**:router 的 dispatch 和 combine 部分用 bf16,**softmax 和 loss 计算用 fp32**——大部分速度,关键部分精度。

这个 trick 让 MoE 训练稳定很多,后来被所有 MoE 工作继承。

---

## 实验结果

### 1. Scaling 曲线

![Switch 在相同 compute 下,不同 model size 的 validation loss:Switch-C(1.6T)和 T5-XXL(13B)compute 相近,但 Switch-C 的 loss 低得多。MoE 的"大容量 + 小激活"带来的效率完胜。](/assets/img/switch-transformer/x4.png)
_Figure 4:Switch vs T5 的 Scaling_

关键数字:

| 模型 | 参数 | 激活 | T5 XXL 同等 compute pre-train 时间 |
|------|------|------|--------------------------------------|
| T5-XXL | 13B | 13B | 100 天(基准) |
| Switch-Base (32 expert) | 395B | 7.4B | **14 天(7×快)** |
| Switch-Large | 1.1T | 26.6B | 13 天 |
| **Switch-C** | **1.6T** | **28B** | 13 天 |

**在相同 compute 下,Switch 达到 T5-XXL 质量只需 1/7 时间**。

### 2. Downstream 任务

在 SuperGLUE、TriviaQA 等下游:

- Switch-Base(395B)超过 T5-Base 多数任务 5-10 分
- Switch 的优势随 fine-tune 数据减少而放大(few-shot 优势大)

### 3. 训练不稳的问题

作者承认:**MoE 训练比 dense 更不稳**。

- Loss spike 时有发生
- 需要 careful hyperparam
- Selective precision 是关键稳定性技巧

---

## 历史影响

### 1. MoE 范式的奠基

Switch 的设计选择(top-1、capacity factor、load balance、selective precision)成为后续几乎所有 MoE 工作的**事实标准**:

- **GLaM** (Google 2021):64 expert,k=2,但很多其他 trick 沿用 Switch
- **Mixtral 8x7B** (2024):8 expert,k=2,基本是 Switch 的思路
- **DeepSeek-MoE** (2024):fine-grained 64 expert,核心机制同源
- **Grok-1**(xAI)、**Qwen-MoE**、**Arctic**:都基于 Switch 思想

### 2. 证明了"sparse = efficient"

Switch 是第一次系统化证明:**相同 compute 下,sparse model 可以胜过 dense**。这个发现让业界开始认真考虑 MoE 路线。

### 3. 推动 "trillion parameter" 时代

Switch 的 1.6T 参数是当时第一个公开的 trillion+ 模型。虽然不是 dense 1.6T(激活 28B),但**容量达到了前所未有的规模**——这激发了后续对超大模型的探索。

### 4. 启发 "sparse training" 研究

Switch 之后,sparse training 成为一个独立研究方向:

- Sparse attention
- Sparse activation in FFN
- Sparse routing
- Sparse checkpointing

### 5. Google 内部 MoE 的主力

Switch 的架构后来被用于 **GLaM**、**PaLM-2**(据信)等 Google 主力模型。虽然具体细节不公开,但大致路线相同。

---

## 局限

### 1. 训练不稳

MoE 训练**比 dense 显著更难**,Switch 承认这点。需要:
- 精心调 hyperparam
- Selective precision
- 较多 debugging 人力

这是开源 MoE 长期落后 Google 的主要原因——直到 Mistral 把这些都搞定。

### 2. 通信昂贵

Expert 分布在多 GPU,token 要 all-to-all 传输。对 inter-GPU 带宽要求高,在慢网络集群下 MoE 可能比 dense 慢。

### 3. Token dropping 带来质量损失

Capacity overflow 的 token 被 drop——理论上每个 token 应该有 "expert 贡献",但 MoE 妥协了部分质量。

### 4. Memory 仍是问题

即使 compute 只激活 1/N,**memory 要加载所有 N 个 expert**。1.6T 模型的内存需求是巨大的,不是所有团队都能跑。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Sparse computation 是 LLM 效率的重要方向**:同 compute 下更大容量意味着更好质量。这个思想贯穿 Switch、Mixtral、DeepSeek-V3 等各代 MoE
2. **Top-1 routing 够用**:直觉告诉我们 top-2 更稳,但 Switch 证明 top-1 + 好 load balancing + capacity factor 就够了。**简单即美德**
3. **Selective precision 是 MoE 稳训练的关键**:Router 的 softmax 用 fp32,其他 bf16——这种细致的精度管理是大模型训练成功的必需
4. **MoE 的突破需要整套工程,不是一个点**:Switch 的成功 = top-1 + capacity + load balance + selective precision + 许多 tuning——这提醒我们大模型研究中,单点突破不够,需要系统工程
</callout>

---

## 延伸阅读

- [Mixtral 8x7B 深度解读]({% post_url 2026-04-24-Mixtral-8x7B-开源MoE深度解读 %}) —— 开源的 Switch 继承者
- [DeepSeek-V3 FP8 训练深度解读]({% post_url 2026-04-24-DeepSeek-V3-FP8训练深度解读 %}) —— MoE 的当代顶峰
- [GShard (Google 2020)](https://arxiv.org/abs/2006.16668) —— Switch 的前身
- [GLaM (Google 2021)](https://arxiv.org/abs/2112.06905) —— Switch 思想的 k=2 改进
- [Shazeer et al. 2017 Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538) —— MoE 在 LLM 的起源
