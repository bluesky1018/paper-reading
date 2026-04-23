---
title: "MLA / DeepSeek-V2 — 用"潜在压缩"把 KV cache 再砍一个数量级"
date: 2026-04-23 20:55:00 +0800
categories: [Attention, Inference Optimization, MoE]
tags: [mla, multi-head-latent-attention, deepseek-v2, kv-cache, low-rank, decoupled-rope, moe]
math: true
---

## 基本信息

- **作者**: DeepSeek-AI (130+ 人作者团)
- **机构**: DeepSeek-AI
- **发表**: arXiv 2024.05
- **arXiv**: [2405.04434](https://arxiv.org/abs/2405.04434)
- **代码/模型**: [deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)
- **延伸**: [DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 都沿用同一 MLA 架构

## 一句话总结

提出 **Multi-head Latent Attention (MLA)** —— 把每头独立的 K、V **联合低秩压缩**到一个小的 **latent 向量 $c^{KV}$**,只把 latent 存入 KV cache。推理时再现场从 latent 解出所有头的 K/V。
- **KV cache 比 GQA 再小 ~7×**,比 MHA 小 **~93%**
- **训练时数学等价于 MHA**,质量不打折
- 配合 **Decoupled RoPE** 解决"低秩 + 位置编码"冲突
- 叠加 **DeepSeekMoE** 的细粒度专家,DeepSeek-V2 以 **21B 激活参数**达到开源 SOTA,训练成本仅 DeepSeek 67B 的 **42.5%**

是 2024 年**推理经济学**的最大一步:让 128K-1M 上下文在一张 GPU 上跑起来,从此 MLA 成为 DeepSeek V2/V3/R1 的标志性架构。

![DeepSeek-V2 在 MMLU 上的激活参数-性能图,以 21B 激活匹敌同期 70B 级别开源模型。右图是训练成本 + 推理吞吐的对比:训练成本 -42.5%,推理吞吐 +5.76×,KV cache -93.3%。](/assets/img/mla-deepseek-v2/x1a.png)
_Figure 1(a):MMLU vs 激活参数——DeepSeek-V2 以 21B 激活参数达到 SOTA_

![训练成本、KV cache 大小、推理吞吐三项指标的对比,DeepSeek-V2 相对 DeepSeek 67B(Dense)分别实现 -42.5%、-93.3%、+5.76×。](/assets/img/mla-deepseek-v2/x1b.png)
_Figure 1(b):DeepSeek-V2 vs DeepSeek 67B——训练/推理经济学的全面提升_

---

## 背景:KV 头共享已到极限,还能怎么压?

回顾 GQA 那篇解读:通过让 Q 头分组共享 K/V,可以把 KV cache 压 8×(GQA-8)。但这已经是 **"多头共享"** 路径的天花板——再压就只能 MQA (所有头共享一份),质量会明显下降。

**DeepSeek 的思路彻底换了一条路**:不再是"多头共享 K/V",而是"**把 K/V 投影到一个低维 latent,每头从 latent 解出自己的 K/V**"。

---

## 核心机制

![MHA / GQA / MQA / MLA 四种架构对比。关键:MLA 把 K、V 联合压缩到一个 latent 向量 c^{KV},只缓存 latent;推理时每头用各自的权重从 c^{KV} 解出 K/V。相比 MHA 的每头独立,MLA 的 KV cache 仅随 latent 维度线性增长,与头数解耦。](/assets/img/mla-deepseek-v2/x3.png)
_Figure 3:MHA、GQA、MQA、MLA 四种注意力架构对比——MLA 的 latent 压缩方案_

### 低秩联合压缩

标准 MHA 的 K、V 计算:

$$
k_t^{(h)} = W_k^{(h)} h_t,\quad v_t^{(h)} = W_v^{(h)} h_t
$$

每头独立,KV cache = $2 \cdot H \cdot d_h \cdot N$。

**MLA 改动**:先把 $h_t$ 压到一个共享 latent:

$$
c_t^{KV} = W_{DKV}\, h_t \in \mathbb{R}^{d_c}
$$

其中 $d_c$ 远小于 $H \cdot d_h$(DeepSeek-V2 用 $d_c = 512$,对比 MHA 的 $128 \times 128 = 16384$ 小 32×)。

推理时再"**解压**"出每头的 K、V:

$$
k_t^{(h)} = W_{UK}^{(h)}\, c_t^{KV},\quad v_t^{(h)} = W_{UV}^{(h)}\, c_t^{KV}
$$

**KV cache 只需缓存 $c_t^{KV}$(一份 latent),而不是所有头的 K、V**:

$$
\text{KV cache size} = d_c \cdot N\quad \text{(MLA)}\quad \text{vs}\quad 2 H d_h N\quad \text{(MHA)}
$$

DeepSeek-V2 的配置下:**MLA 的 KV cache 是 MHA 的 1/16,是 GQA-8 的 1/2 左右**。

### 等价矩阵吸收

有人会问:训练时是不是 MHA 的模型,推理时怎么只缓存 latent?关键在于:

$$
\text{Attention}_{(h)} = \text{softmax}\!\left(\frac{q_t^{(h)} (W_{UK}^{(h)} c_t^{KV})^\top}{\sqrt{d_h}}\right) (W_{UV}^{(h)} c_t^{KV})
$$

$W_{UK}^{(h)}$ 可以**吸收到 $W_Q^{(h)}$ 里**:

$$
q_t^{(h)} (W_{UK}^{(h)} c_t^{KV})^\top = (q_t^{(h)} W_{UK}^{(h)})\, (c_t^{KV})^\top
$$

同样 $W_{UV}^{(h)}$ 可以吸收到输出投影 $W_O$。**推理时无需显式计算 K/V,只用 latent 本身做 attention**。**这就是 "Multi-head Latent" 的来源**。

---

## 关键难题:RoPE 与低秩压缩的冲突

RoPE(前一篇解读过)是把位置信息以**旋转**的方式施加到 $q, k$ 上:

$$
q_m^{\text{RoPE}} = R_m q_m,\quad k_n^{\text{RoPE}} = R_n k_n
$$

旋转矩阵 $R_m$ 和 $R_n$ **与位置相关**。如果 k 要从 latent $c^{KV}$ 解出,再做 RoPE 旋转,**$W_{UK}^{(h)}$ 就不能吸收到 $W_Q^{(h)}$ 里了**——因为 $R_n$ 在中间,破坏了可吸收的矩阵乘结构。

### Decoupled RoPE 的解法

把 K 分成**两部分**:

1. **不带 RoPE 的部分** $k_t^{C, (h)}$:走 latent 压缩路径,做 attention 的"内容"部分
2. **带 RoPE 的部分** $k_t^{R}$:**所有头共享**,单独用一个 MQA 风格的投影,做 attention 的"位置"部分

$$
k_t^{(h)} = [\underbrace{W_{UK}^{(h)} c_t^{KV}}_{\text{latent 解压,无 RoPE}};\ \underbrace{R_t\, W_{KR} h_t}_{\text{RoPE,全头共享}}]
$$

Q 也做同样拆分。最终 attention 在拼接向量上进行。

<callout emoji="bulb" background-color="light-blue" border-color="blue">
**直观理解**:内容信息走 "低秩 latent" 省显存;位置信息走 "全头共享 MQA" 相当于占一个小尾巴,不显著增大 cache。两者在 attention 里 **天然拼接**,互不干扰。
</callout>

**KV cache 新公式**:

$$
\text{size}_{MLA} = (d_c + d_h^R) \cdot N
$$

其中 $d_h^R$ 很小(如 64)。总体仍然远小于 MHA/GQA。

---

## 架构:MLA + DeepSeekMoE

![DeepSeek-V2 整体架构:每个 transformer 层 = MLA + DeepSeekMoE(细粒度专家:2 个 shared expert + 64 个 routed expert,每 token 激活 6 个 routed)。注意力端用 MLA 节省 KV cache,FFN 端用细粒度 MoE 提升质量-激活比。](/assets/img/mla-deepseek-v2/x2.png)
_Figure 2:DeepSeek-V2 架构——MLA + DeepSeekMoE 的组合拳_

两个维度一起压:

| 维度 | 技术 | 收益 |
|------|------|------|
| **注意力端** | MLA | KV cache -93%,推理吞吐 +5.76× |
| **FFN 端** | DeepSeekMoE(细粒度专家 + 共享专家) | 236B 总参 × 21B 激活,训练成本 -42.5% |

**细粒度 MoE** 的哲学:把传统 MoE 的大专家拆成**更多小专家**(64 个)+ 每 token 激活多个(6)+ 引入"共享专家" 2 个。这样每个 token 可以由**6+2 个专家联合刻画**,组合灵活性远高于"8 个大专家选 2 个"。

---

## 实验结果

### 推理效率

| 指标 | DeepSeek 67B (MHA) | DeepSeek-V2 (MLA+MoE) | 提升 |
|------|--------------------|-----------------------|------|
| 总参数 | 67B | 236B | 更大 |
| 激活参数 | 67B | 21B | **-69%** |
| KV cache (per token, 估计) | ~1.6 MB | ~110 KB | **-93.3%** |
| 最大吞吐(50k GPU·s) | 基准 | 5.76× | **+476%** |
| 训练成本 (每 T token) | 基准 | 42.5% | **-57.5%** |

### 质量评测

| Benchmark | DeepSeek 67B | DeepSeek-V2 | Llama-3 70B |
|-----------|--------------|-------------|-------------|
| MMLU | 71.3 | **78.5** | 79.5 |
| BBH | 68.7 | **78.9** | 81.0 |
| GSM8K | 63.4 | **79.2** | 83.0 |
| HumanEval | 42.7 | **48.8** | 50.0 |
| MT-Bench | 7.46 | **8.97** | 9.00 |

以 21B 激活匹敌 Llama-3 70B,**激活参数少 3×,KV cache 小 10×+**。

![HumanEval 与 LiveCodeBench 评测:DeepSeek-V2 Chat (RL) 在 LiveCodeBench 上超越许多更大规模的模型,体现训练配方优势。](/assets/img/mla-deepseek-v2/x5.png)
_Figure 5:代码能力评测——DeepSeek-V2 在新鲜代码基准上的稳定表现_

### 长上下文

- 支持 **128K** 原生上下文
- Needle-in-a-Haystack 在 128K 全长度保持 ~100% 准确率
- 内存占用和吞吐远优于同级 MHA/GQA 模型

---

## 为什么影响巨大

### 1. 推理经济学新标杆

GQA 已经把 KV cache 压了 8×,业界一度以为"已经到头"。MLA 在 GQA 基础上**再压一个数量级**,并且**质量完全不掉**。这让:

- 128K 上下文服务从"不可能"变成"经济"
- 同样的 GPU 能支持的**并发用户数**翻几倍
- **开源模型**第一次在推理经济学上超越闭源方案

### 2. 打破 "质量 / 效率" 的二元对立

MLA 的训练时等价于 MHA(甚至理论上表达力**更强**,因为 latent 本身是 learned compression),**不是以质量换效率**。这和 MQA/GQA 的"必然损失"哲学截然不同。

### 3. 被 DeepSeek 整个系列继承

- DeepSeek-V2 (2024.05):首发 MLA + 细粒度 MoE
- DeepSeek-V3 (2024.12):MLA + Multi-Token Prediction + FP8 训练
- DeepSeek-R1 (2025.01):在 V3 基础上 RL 推理,成本不变
- **整个系列的推理经济学优势**都基于 MLA

### 4. Decoupled RoPE 的通用意义

"把位置信息与内容信息在 attention 中**解耦**"的思想,被后续许多工作借鉴:

- 多模态模型:图像 RoPE 与文本 RoPE 解耦
- Cross-attention:Query 侧与 Key 侧不对称应用 RoPE
- Linear Attention + Position:给线性 attention 加位置的通用方案

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **"压多头"与"压 latent"是两条正交的路径**:GQA 是前者的终点,MLA 开辟了后者,两者可以独立优化
2. **矩阵吸收(absorption)是 MLA 免费效率的核心**:训练当 MHA 训,推理时把 $W_{UK}, W_{UV}$ 吸收到 Q 和 O,无需显式 K/V
3. **RoPE + 低秩压缩不兼容,解耦是唯一解**:把位置通路单独挂一个共享 MQA 尾巴,这个 pattern 可以推广到很多类似场景
4. **KV cache = 显存 + 带宽 + 并发 + 上下文长度,是 LLM 部署的四维绞杀**:MLA 一次性在所有四维上改善,这是它影响巨大的根本原因
</callout>

---

## 延伸阅读

- [DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437) —— MLA 2.0 版本 + MTP + FP8
- [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245) —— KV head 共享路线的代表
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) —— 与 MLA 互补的系统优化
- [DeepSeekMoE (Dai et al., 2024)](https://arxiv.org/abs/2401.06066) —— 细粒度专家 + 共享专家的 MoE 设计
- [RoFormer / RoPE (Su et al., 2021)](https://arxiv.org/abs/2104.09864) —— MLA 要 decouple 的位置编码
- [PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) —— MLA + PagedAttention 是当今开源推理服务的黄金组合
