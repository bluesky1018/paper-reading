---
title: "Jamba — 第一个生产级 Mamba + Transformer + MoE 混合大模型"
date: 2026-04-24 10:15:00 +0800
categories: [Attention, State Space Model, MoE, Hybrid Architecture]
tags: [jamba, mamba, transformer, moe, hybrid, ai21-2024]
math: true
---

## 基本信息

- **作者**: AI21 Labs 团队 (Lieber, Lenz, Bata, et al.)
- **机构**: AI21 Labs
- **发表**: arXiv 2024-03
- **arXiv**: [2403.19887](https://arxiv.org/abs/2403.19887)

## 一句话总结

**Jamba** 是第一个生产规模(**52B 总参 / 12B 激活**)的 Mamba + Transformer + MoE 三合一模型。核心设计思想:**每 8 层中有 1 层是 Transformer attention,其余是 Mamba;在其中的 FFN 位置嵌入 16 个专家的 MoE**。这个混合架构同时拿下了三个重要的性质:**256K 超长 context + 高 decode 吞吐 + 接近 70B 密集模型的质量**。开源权重 Apache 2.0,是首次让社区能上手玩 Hybrid SSM 模型。

![Jamba 的 block 结构:一个 "Jamba block" 由多个 Mamba 层 + 少数 Attention 层 + MoE FFN 交错组成。8:1 的 Mamba:Attention 比是核心设计。](/assets/img/jamba/x1.png)
_Figure 1:Jamba 的混合架构——多 Mamba + 少 Attention + MoE FFN_

---

## 背景:纯 SSM 和纯 Transformer 各有硬伤

到 2024 年初,社区对 SSM 和 Transformer 的理解已经相对成熟:

### 纯 Transformer 的痛点

- **KV cache 随 context 增长**:128K context 每个请求 10+ GB KV cache
- **Decode 成本高**:长 context + 高并发经济上不可行
- **但**:in-context learning、复杂推理、精确 recall 能力强

### 纯 Mamba / SSM 的痛点

- **Recurrent 状态容量有限**:对需要精确 recall 一长串 token 的任务表现弱
- **In-context learning 不稳定**:few-shot prompt 效果不如 Transformer
- **但**:decode 成本恒定、长 context 几乎零额外开销

AI21 的想法:**把二者混合**。用 Mamba 做大部分层获取效率,用少数 Attention 层保住"精确 recall + ICL"能力。

---

## 核心机制:Jamba block 的设计

![Jamba 每个 block 内部:Mamba-Attention-MoE 的交错组合。MoE 不是每层都有,而是每 2 层一个。](/assets/img/jamba/x3.png)
_Figure 2:Jamba block 内部结构——细粒度的组件交错_

### 1. 层结构:1:7 Attention:Mamba 比

Jamba 的核心超参:

- **Mamba layers per block**:7
- **Attention layers per block**:1(放在 block 开头附近)
- **Ratio**:1:7

作者做了大量 ablation 对比不同比例:

| A:M 比 | 模型质量 | Decode 速度 |
|--------|---------|-------------|
| 1:0(全 Attention) | Baseline | 最慢 |
| 1:3 | 持平 Attention | 1.8× |
| 1:7 | **持平 Attention** | **3.2×** ★ |
| 1:15 | 略有下降 | 4.8× |
| 0:1(全 Mamba) | 明显下降 | 5.5× |

**1:7 是甜蜜点**——既保住了 Attention 的关键能力,又获得了 Mamba 的大部分效率。

### 2. MoE 替代 FFN

![MoE 的配置:每 2 层 FFN 的其中一层换成 16 个专家的 MoE,每 token 激活 2 个专家。相对密集 FFN,MoE 增加总参但不增加激活参。](/assets/img/jamba/x4.png)
_Figure 3:Jamba 的 MoE 设置_

MoE 的设计:

- **每 2 层 FFN 替换一层为 MoE**(不是全部层)
- **16 个专家,top-2 路由**
- 总参数量 52B,单 token 激活参数约 12B

这让 Jamba 具备"**大模型的容量,小模型的计算**"——激活参数 12B 的推理 FLOPs,效果接近 70B 密集模型。

### 3. RoPE / ALiBi 的位置编码

Mamba 原生不需要位置编码,Attention 部分用 **RoPE**。由于 Attention 只占 1/8,位置编码的复杂度也被摊薄。

### 4. 无原生位置编码的 Mamba 层

Mamba 层不需要位置编码——**这反而让 Jamba 在长 context 上更稳定**。纯 Transformer 在超过训练长度时 RoPE 失配,Jamba 因为大部分层用 Mamba,对 context 长度的敏感度显著降低。

---

## 关键实验发现

### 1. 长 context:256K 原生可用

![Jamba 在 NeedleInHaystack 任务上:256K 上下文内召回率接近 100%,显著超过同规模 Llama-2、Mixtral。](/assets/img/jamba/x5.png)
_Figure 4:256K context 的 needle-in-haystack 测试_

关键数字:

- **训练 context**:256K(!)
- **KV cache 占用**:相同 seq len 下 Jamba 约为 Llama-70B 的 1/8(因为只有 1/8 层有 KV)
- **A100-80GB 单卡最大 batch size**:
  - Llama 13B @ 128K context:1
  - Jamba 12B 激活 @ 128K context:**16**

这是 Jamba 最具震撼力的数字——**单卡能支持 16× 的并发**。

### 2. 质量:接近 Mixtral 8x7B

在主流 benchmark 上:

| 模型 | MMLU | GSM8K | HumanEval | Ave |
|------|------|-------|-----------|-----|
| Llama-2 13B | 54.8 | 28.7 | 18.3 | 34 |
| Mixtral 8x7B (13B 激活) | 71.9 | 57.6 | 40.9 | 57 |
| **Jamba 12B 激活** | **67.4** | **59.9** | **29.3** | **52** |

Jamba 的表现介于 Mixtral 和 Llama-2 13B 之间,但**长 context 场景下优势明显**。

### 3. Attention 是"不可或缺"的少数

![Ablation 研究:完全去掉 Attention 层,模型在 needle-in-haystack 等需要精确 recall 的任务上明显掉分。保留少量 Attention 层即可恢复性能。](/assets/img/jamba/x2.png)
_Figure 5:Ablation——Attention 层的关键作用_

作者发现:

- **纯 Mamba 在 copy-paste、key-value recall 等任务上显著弱**
- 但**加 1 层 Attention/block 后即可接近 full attention**
- 关键原因:Attention 提供"精确读取任意历史位置"的能力,而 SSM 只有"加权汇总"

这个发现的意义:**Attention 的少数层承担"精确 recall",Mamba 的多数层承担"高带宽信息处理"**——两者分工明确。

---

## 为什么是架构史上的里程碑

### 1. 证明混合架构是实际可行的

在 Jamba 之前,"SSM + Transformer" 只是学术设想。Jamba 把它做到 52B 生产规模并开源,**证明了 Hybrid 路线工程上可行**。

### 2. 给长 context 经济学带来新解

纯 Transformer 的 128K context 经济上难以承担。Jamba 的混合架构让"**长 context 不再昂贵**"——这对 RAG、agent、长文档处理都有直接意义。

### 3. 推动 "Attention 是稀缺资源" 的认识

Jamba 之前,工程师倾向于把所有层都做成 attention。Jamba 让大家意识到:**attention 不需要每层都有,少数几层足够**。后续 Samba、Zamba 等更激进的混合架构都沿这条路走。

### 4. MoE + SSM 的协同

Jamba 展示了 MoE 和 SSM 混合的可能——两个独立的效率技术叠加起来,效率收益几乎相乘。52B 总参 + 12B 激活 + 1:7 attention 比,实际计算成本只有原始 Transformer 的 ~15%。

---

## 局限

### 1. 复杂度高,调优难

Mamba + Attention + MoE 三种结构交错,超参空间巨大:Attention 比例、MoE 专家数、每层选择哪个等等。Jamba 的具体配置是大量 ablation 的结果,直接迁移到新场景未必最优。

### 2. 质量上限仍差 SOTA 纯 Transformer

Jamba 的综合质量仍弱于同激活参数的 Llama-2 70B(dense)或 Mixtral 8x22B。**混合架构的"效率优势"是以一些"质量天花板降低"换来的**。

### 3. 对硬件 kernel 依赖高

Mamba 层和 Attention 层需要不同的 kernel,调度复杂。vLLM 等主流 inference 框架对 Jamba 的支持不如纯 Transformer 成熟。

### 4. 长 context 的有效 receptive field 仍有限

Jamba 能训练 256K,但在**超长精确 recall** 任务上仍有衰减——Mamba 层的 state capacity 仍是硬约束。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Attention 和 Mamba 是互补的,不是替代的**:Attention 擅长精确 recall,Mamba 擅长高效汇总——各用所长而非二选一
2. **1:7 的 Attention:Mamba 比是一个重要经验值**:这是工程师做 Hybrid 架构的起点,后续 Samba、Zamba 都在这附近微调
3. **MoE 可以无缝叠加在 Hybrid 架构上**:效率收益几乎相乘,成为超大模型的必备路线
4. **长 context 的经济性来自"结构级优化"**:不是靠 PagedAttention 这种"实现级"优化,而是靠"Attention 只用 1/8 层"这种"架构级"决策
</callout>

---

## 延伸阅读

- [Mamba 深度解读]({% post_url 2026-04-23-Mamba-选择性状态空间模型深度解读 %}) —— Jamba 的 SSM 基础
- [Mamba-2 深度解读]({% post_url 2026-04-24-Mamba-2-SSD深度解读 %}) —— SSM 的架构升级
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) —— 纯 Transformer MoE
- [Samba (Ren et al., 2024)](https://arxiv.org/abs/2406.07522) —— Jamba 之后的 Hybrid 架构
- [RWKV (Peng et al., 2023)](https://arxiv.org/abs/2305.13048) —— 另一条 RNN-Transformer 混合路线
