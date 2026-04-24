---
title: "Chinchilla — 修正 Kaplan:参数和数据应 1:1 等比例扩展,1B 模型配 20B tokens"
date: 2026-04-24 19:30:00 +0800
categories: [Pretraining, Scaling Law]
tags: [chinchilla, compute-optimal, scaling-law, hoffmann-2022]
math: true
---

## 基本信息

- **作者**: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch 等
- **机构**: DeepMind
- **发表**: arXiv 2022-03
- **arXiv**: [2203.15556](https://arxiv.org/abs/2203.15556)
- **全名**: *Training Compute-Optimal Large Language Models*

## 一句话总结

DeepMind 重做 Kaplan scaling law 实验,得出**颠覆性结论**:**之前所有大模型都训练不足**。Gopher 280B、GPT-3 175B、MT-NLG 530B 等模型**参数太大、数据太少**——按 compute 最优配置,它们应该小一半、数据多十倍。作者根据新的 scaling law 训练了一个 **70B 参数 + 1.4T tokens** 的 **Chinchilla** 模型,**在 compute 完全相同的前提下,全面超越 Gopher 280B**——MMLU 67.5% vs 60.0%。核心结论:**参数量 N 与数据量 D 应该 1:1 等比例 scale,最优比约 1:20**(1B 参数配 20B tokens)。这一发现直接影响了 LLaMA、Mistral 等所有后续大模型的训练配方,是继 Kaplan 之后 scaling 研究的第二个里程碑。

![Chinchilla 的核心结论:最优的 N vs C 和 D vs C 曲线。指数系数 0.5 意味着参数和数据应等比例扩展(之前 Kaplan 说 0.73 vs 0.27)。](/assets/img/chinchilla/x1.png)
_Figure 1:Chinchilla 修正的 Scaling Law 系数_

---

## 背景:大模型"越来越大" 真的对吗?

### Kaplan 之后的竞赛

GPT-3(175B)后,业界走上"越大越好"的路线:

- **Gopher** (DeepMind, 2021):280B 参数 + 300B tokens
- **MT-NLG** (Microsoft, 2022):530B + 270B tokens
- **PaLM** (Google, 2022):540B + 780B tokens

共同特点:**参数量巨大,但训练 token 数相对少**——这是因为大家跟随 Kaplan 的 $N \propto C^{0.73}$、$D \propto C^{0.27}$ 推荐。

### DeepMind 的质疑

Chinchilla 团队发现:**这个推荐可能错了**。原因:

- Kaplan 的 $L(D)$ 曲线是在 **固定 LR schedule** 下测的——没调大 batch/LR 配合大 dataset
- Kaplan 把 optimizer hyperparams 当作固定,实际在不同 scale 下最优 hyperparams 会变

DeepMind 重做实验,**让 hyperparams(特别是 cosine LR schedule 的 cycle length)跟随 D 变化**——结果完全不同。

---

## 三种方法,同一结论

### Approach 1:固定 model,扫 training token

对每种 model size(70M - 10B),在不同 training tokens 下测 loss。提取 **loss 随 D 的下降**。

### Approach 2:IsoFLOP 曲线

![IsoFLOP 曲线:固定 compute 预算 $C$,变化 model size $N$,观察 loss。最小 loss 对应的 $N^*$ 和 $D^* = C / (6N^*)$ 就是最优配置。不同 $C$ 下 $N^*$ 呈幂律。](/assets/img/chinchilla/x2.png)
_Figure 2:IsoFLOP 曲线——给定 compute 的最优 N_

对每个 compute 预算 $C$,扫不同 $N$(同时调整 $D$ 使 $C$ 固定),找 loss 最低的 $N^*$。

### Approach 3:参数化拟合

用公式 $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$ 拟合所有实验数据。

**三种方法独立地给出相同结论**:

$$
N_{\text{opt}} \propto C^{0.5},\quad D_{\text{opt}} \propto C^{0.5}
$$

即 **参数和数据应该 1:1 等比扩展**,比值 **N:D ≈ 1:20**。

---

## 对比:老规则 vs Chinchilla 规则

### 重新审视当时的大模型

![Chinchilla 的表:老模型按新规则都训练不足。Gopher 280B 按 Chinchilla 规则应该只需 63B 参数但要 1.4T tokens;MT-NLG 530B 类似。每个都"参数过大、数据不足"。](/assets/img/chinchilla/x3.png)
_Figure 3:按 Chinchilla 规则重新评估现有大模型_

| 模型 | 实际 N | 实际 D | Chinchilla 推荐 N | Chinchilla 推荐 D |
|------|--------|--------|------------------|------------------|
| GPT-3 | 175B | 300B | 63B | **1400B** |
| Gopher | 280B | 300B | 67B | **1400B** |
| MT-NLG | 530B | 270B | 48B | **2400B** |

**现有大模型都严重 under-trained**!

### Chinchilla 的实验:同 compute,更好结果

Chinchilla 团队训了一个 **70B 参数 + 1.4T tokens** 的模型(compute 与 Gopher 相当):

| 模型 | 参数 | Tokens | MMLU | BIG-bench |
|------|------|--------|------|-----------|
| Gopher | 280B | 300B | 60.0 | 52.1 |
| **Chinchilla** | **70B** | **1.4T** | **67.5** | **58.7** |

**小 4×,但全面更强**——这个对比极具震撼。

---

## 影响到 LLaMA 等后续工作

### LLaMA 2

Meta 的 LLaMA 2 开始遵循 Chinchilla 规则:
- 7B 配 2T tokens(比率 1:290, 其实 over-train)
- 70B 配 2T tokens(比率 1:29, 接近 Chinchilla 最优)

### 后来的"Beyond Chinchilla-Optimal"

2024 年有人发现:**Chinchilla 说的是"训练 compute 最优"**,没考虑**推理成本**。

如果模型要被大规模部署,小模型推理便宜得多。所以现实中应该:

- 大模型不 over-train(按 Chinchilla)
- 小模型 over-train(训久一点)

**LLaMA 3** 更激进:**8B 模型 overtrain 到 15T tokens**(Chinchilla 建议 200B)——推理时大量节省。

![LLaMA 3 的 8B 模型训练到 15T tokens,远超 Chinchilla 的 200B 推荐。这是 "overtrain 小模型" 策略的极致体现。](/assets/img/chinchilla/x4.png)
_Figure 4:LLaMA 3 对 Chinchilla 的后续修正_

---

## 历史影响

### 1. 修正整个业界的训练哲学

Chinchilla 之前:大模型 + 少数据
Chinchilla 之后:**合理大小 + 大量数据**

这个转变让开源社区能用**相对小的模型**达到 GPT-3 级别效果——LLaMA 7B、Mistral 7B 等都是这个思路。

### 2. "数据荒" 问题浮出水面

按 Chinchilla 规则,训 175B 模型需要 **3.5T tokens**——接近整个 Common Crawl 的高质量部分。

这让"**高质量数据耗尽**"(data wall)成为 2023-2024 年的热门话题——也是 Phi(合成数据)、FineWeb(更好的清洗)等工作的背景。

### 3. 开源 vs 闭源

Chinchilla 结论对开源有利:

- 小模型(7B-70B)在可接受的 compute 内可达 SOTA
- 开源团队的 compute 不如 OpenAI/Google,但通过更好的数据比例也能追
- 这直接催生 LLaMA → Mistral → Qwen → DeepSeek 等开源生态

### 4. 成为 scaling law 研究的主基准

"Chinchilla optimal" 成为大家谈论 compute-optimal 的默认名词。后续所有 scaling law 相关工作都以 Chinchilla 为 baseline。

---

## 局限

### 1. 只看 Pretrain 成本

Chinchilla 的"最优"只优化训练 compute。实际成本还包括:

- **推理 compute**:部署后每次请求的成本(推动 overtrain)
- **人力 + 基础设施**:训练 1.4T tokens 的数据清洗 / 存储成本

### 2. 架构局限

实验用的是 vanilla Transformer。对 **MoE、Mamba 等**新架构,scaling law 系数不同。

### 3. 英文数据假设

实验数据主要是英文 web。对多语言、代码、数学等特化数据,最优比例可能不同。

### 4. 下游任务的 scaling 未必线性

loss 的 scaling 是平滑的,但下游任务(MMLU、reasoning)可能有 **emergent abilities**——scaling 到某规模才出现,Chinchilla 公式预测不了这个。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **N 和 D 应该 1:1 等比扩展**:这个结论把大家从"参数崇拜"中拉出来,让"更多数据"成为另一个 first-class citizen。**1B 参数需要 20B tokens 才够**
2. **实验细节决定论文结论**:Kaplan 和 Chinchilla 的差异根源在 hyperparam(特别是 LR schedule)的处理——这提醒我们 scaling law 实验要极其仔细,否则结论会误导整个行业
3. **Chinchilla 后的世界:模型会越来越小,训练会越来越长**:LLaMA/Mistral/Qwen 的 7B-70B 配好几 T tokens,这是 Chinchilla 的直接遗产
4. **数据质量成为 2024 年后的主旋律**:按 Chinchilla 规则训模型需要大量高质量数据,这催生了合成数据、数据清洗、自动配比等整条研究线
</callout>

---

## 延伸阅读

- [Kaplan Scaling Laws 深度解读]({% post_url 2026-04-24-Kaplan-Scaling-Laws-深度解读 %}) —— Chinchilla 修正的对象
- [LLaMA 3 深度解读]({% post_url 2026-04-24-LLaMA-3-405B开源大模型深度解读 %}) —— Chinchilla 规则的当代应用
- [Beyond Chinchilla-Optimal (2024)](https://arxiv.org/abs/2401.00448) —— 引入推理成本的修正
- [Phi / Textbooks Are All You Need]({% post_url 2026-04-24-Phi-教科书式数据深度解读 %}) —— 数据质量的另一条路
- [DoReMi 深度解读]({% post_url 2026-04-24-DoReMi-自动数据配比深度解读 %}) —— 数据混合比例的后续
