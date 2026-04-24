---
title: "Small-scale proxies for large-scale Transformer training instabilities"
date: 2026-04-24 11:30:00 +0800
categories: [Training, Stability, Scaling]
tags: [training-stability, scaling, qk-norm, z-loss, wortsman-2023]
math: true
---

## 基本信息

- **作者**: Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, Simon Kornblith
- **机构**: Google DeepMind, Google Research
- **发表**: arXiv 2023-09
- **arXiv**: [2309.14322](https://arxiv.org/abs/2309.14322)

## 一句话总结

这是一篇罕见的**专门研究训练不稳定性的系统性论文**。作者发现,大模型训练中出现的**两类标志性不稳定现象**——(1) attention logit 在训练中期爆炸、(2) 输出 logit 的 softmax 与权重共享导致的震荡——**可以在小模型(~100M)上复现**。利用这个"小规模代理",他们系统测试了 QK-Norm、z-loss、学习率等干预措施,给出一套可直接迁移到大模型的配方。这篇论文让"训练稳定性"从"只能等大模型跑出问题再救火"变成了"小模型上可科学研究的对象"。

![两种标志性不稳定现象:左)attention logit 指数爆炸,导致 softmax 饱和;右)output logit 的 "divergence" 现象,loss 在中后期突然跳升。](/assets/img/training-instability/x1.png)
_Figure 1:大模型训练的两类标志性失败模式_

---

## 背景:大模型训练的黑箱稳定性问题

### 现实痛点

训练 10B+ 模型时,常遇到以下头痛现象:

- **Loss spike**:loss 突然从 2.5 跳到 5.0,然后可能恢复也可能 NaN
- **Attention logit 爆炸**:某些 head 的 $q^\top k$ 达到 $10^4$+,softmax 接近 one-hot
- **梯度爆炸/消失**:某些 layer 的梯度规模异常,训练停滞

这些问题在小模型(< 1B)上很少出现,常规 ablation 在 100M 上看不到——**等到 10B 上出现,已经烧掉几千 GPU-day,救火成本极高**。

### 这篇论文的 Key Insight

作者发现:**仔细设计的"代理实验"可以在小模型上复现这些失败模式**。具体方法:

1. 用**高学习率**推高训练压力
2. 训练**更多 step**(超过平常推荐)
3. 追踪一些**敏感指标**(QK logit 最大值、输出 logit 最大值等)

这样 100M 模型也会出现大模型的失败模式——**让快速、低成本 ablation 研究稳定性成为可能**。

---

## 核心发现一:Attention Logit 爆炸

![小模型上观察到的 attention logit 最大值随训练步数增长:不加 QK-Norm 时持续爆炸,加了后稳定。这个行为和 PaLM-540B 等大模型一致。](/assets/img/training-instability/x2.png)
_Figure 2:Attention logit 爆炸的小模型复现_

### 现象

训练中 $\max_{ij} |q_i^\top k_j|$ 持续增长,达到 $10^3 - 10^5$ 量级。Softmax 后一个 token 吸走全部概率,训练信号消失。

### 解法:QK-Norm

加入 QK-Norm(见[QK-Norm 深度解读]({% post_url 2026-04-24-QK-Norm-深度解读 %})):

$$
\hat{q} = \text{LayerNorm}(q),\quad \hat{k} = \text{LayerNorm}(k),\quad \text{attn} = \text{softmax}(\gamma \hat{q}^\top \hat{k})
$$

小模型 proxy 和大模型(实测到 9B)上都验证:**QK-Norm 完全消除这种不稳定**。

---

## 核心发现二:Output Logit Divergence

![Output logit 的 z(max logit)与 log Z(LogSumExp)在训练中期发散:某些 token 的 logit 增长到极大值,让 softmax 主要集中在一个位置。](/assets/img/training-instability/x3.png)
_Figure 3:Output logit 爆炸导致 softmax 退化_

### 现象

不只是 attention logit,**输出投影 $W_{out}^\top h$ 的 logit** 也会在训练中期出现数量级爆炸,特别是当:

- 输出权重和 embedding 共享(weight tying)
- 词表很大($|V| > 32000$)
- 训练后期

表现:一个特定 token 的 logit 远大于其他所有 token,softmax 变成 hard-max。

### 解法:Z-loss

作者提出 **Z-loss**:惩罚 $\log Z = \log \sum_i e^{l_i}$ 偏离 0 的 term:

$$
\mathcal{L}_{z} = \epsilon \cdot (\log Z)^2
$$

加到总 loss 上,$\epsilon \sim 10^{-4}$。这相当于"软约束" output logit 的总规模,防止 $\log Z$ 失控增长。

### 为什么是 $\log Z$ 而不是别的

$\log Z$ 直接反映 softmax 分母的量级。通过惩罚 $(\log Z)^2$ 让它不偏离 0 太远,等价于把所有 logit 的"平均量级"约束在合理范围。

实测:Z-loss 消除 output logit divergence,对模型质量影响可忽略。

---

## 发现三:超参对稳定性的影响

![论文系统测试各种超参与稳定性的关系:Adam $\epsilon$、weight decay、学习率 warmup 等都有显著影响。小模型 proxy 上的趋势与大模型一致。](/assets/img/training-instability/x4.png)
_Figure 4:超参扫描与稳定性—小模型代理的结论可迁移_

几个关键发现:

### Adam $\epsilon$ 要足够大

小的 Adam $\epsilon$(默认 $10^{-8}$)会让梯度很小时更新过猛,在大模型上放大到失稳。**$\epsilon = 10^{-6}$** 或 **$10^{-5}$** 更稳。

### Weight decay 很重要

弱的 weight decay(< $10^{-2}$)会让参数规模持续增长,attention logit 和 output logit 都更容易爆炸。**推荐 $10^{-1}$** 级别的 decay。

### Warmup 不能太短

LR warmup 太短(< 500 步)会在初期 push 模型过快,增加失稳概率。**建议 2000+ steps**。

### μP(Maximum Update Parametrization)

μP(Yang 2021)的初始化 / LR scaling 让不同宽度模型使用**相同的最优 LR**。作者发现 μP 的小模型与大模型在稳定性问题上的行为一致性更高——**做稳定性研究时 μP 是首选 parameterization**。

---

## 代理的可靠性验证

![在 100M proxy 上测出的干预效果,迁移到 9B 模型后依然成立。QK-Norm 和 Z-loss 在两种规模上都消除了对应的不稳定。](/assets/img/training-instability/x5.png)
_Figure 5:Proxy 结论迁移到大模型的验证_

作者用 9B-16B 模型验证:proxy 上测出的 "加 QK-Norm 解决 attention logit 爆炸" 和 "加 Z-loss 解决 output logit divergence" 在大模型上**完全复现**。这是该论文最重要的 scientific claim——**小模型代理不只是 anecdotal,是可迁移的**。

---

## 为什么这篇论文如此重要

### 1. 让稳定性从"工程问题"变为"科学问题"

之前稳定性问题都是"大模型训练中遇到再 debug"的工程 experience。这篇论文让它**可以在 100M 模型上系统研究**,节省 99% 的实验成本。这在方法论上是个突破。

### 2. 为 QK-Norm 和 Z-loss 提供权威 benchmark

QK-Norm 原论文 (2020) 规模较小,不足以说服大模型社区。这篇 2023 年的论文在 100M - 9B 范围系统验证,让 QK-Norm 成为**经过验证的大模型训练必备组件**。

### 3. Z-loss 的推广

Z-loss 在 PaLM 论文中有简要提及,但没有展开。这篇论文把 Z-loss 作为一阶稳定性工具系统化,让社区普遍采用。

### 4. 小模型代理的方法论

论文示范了如何**设计小模型代理**——高 LR + 长训练 + 敏感指标——来暴露大模型问题。这个方法论后来被 scaling law 研究、attention 变体 ablation 等领域广泛采用。

---

## 局限

### 1. 不是所有失败模式都能在小模型复现

有些更"细微"的失败模式(如 MoE 的路由坍塌、多模态训练的模态失衡)在小模型上仍难以复现。需要更大规模才能看到。

### 2. μP 不是 universal

μP 对稳定性测试很好,但不是所有模型都用 μP 初始化。非 μP 模型上结论迁移性略差。

### 3. Z-loss 的 $\epsilon$ 敏感

太小无效,太大损害性能。需要按模型规模和任务调节。

### 4. 数据分布因素忽略

稳定性可能也受数据分布的影响(比如代码 heavy 的数据 vs 自然语言)。论文没有深入这一维。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **稳定性问题在小模型上可以复现**:通过高 LR + 长训练 + 敏感指标,100M 模型就能暴露 9B 模型的 attention logit 爆炸——这让研究效率提升了一个数量级
2. **QK-Norm + Z-loss 是大模型训练的"稳定性双子星"**:分别解决 attention logit 和 output logit 两类爆炸,组合使用几乎消除常见的训练失稳
3. **超参的稳定性维度被严重低估**:Adam $\epsilon$、weight decay、warmup 长度都比大多数人以为的更关键。默认值常常不适合大模型
4. **稳定性研究的方法论更有长期价值**:比起"又发现一个训练 trick",这篇论文的贡献更在于"建立了一种研究训练稳定性的系统化方法",这种 meta-level 贡献可以指导未来十年的训练稳定性研究
</callout>

---

## 延伸阅读

- [QK-Norm 深度解读]({% post_url 2026-04-24-QK-Norm-深度解读 %}) —— 论文验证的关键干预措施
- [Maximum Update Parametrization (Yang, 2021)](https://arxiv.org/abs/2011.14522) —— μP 的原始论文
- [PaLM (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311) —— Z-loss 的最初引入
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) —— scaling 研究的经典
- [Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) —— compute-optimal 训练
