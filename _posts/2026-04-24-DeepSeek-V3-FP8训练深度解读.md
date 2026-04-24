---
title: "DeepSeek-V3 的 FP8 训练 — 把 671B MoE 模型全链路 FP8 训完,成本降到行业 1/10"
date: 2026-04-24 14:30:00 +0800
categories: [Quantization, Training, MoE]
tags: [deepseek-v3, fp8-training, mixed-precision, mla, moe, deepseek-2024]
math: true
---

## 基本信息

- **作者**: DeepSeek-AI 团队
- **机构**: DeepSeek
- **发表**: arXiv 2024-12
- **arXiv**: [2412.19437](https://arxiv.org/abs/2412.19437)

## 一句话总结

DeepSeek 发布的 **DeepSeek-V3** 技术报告——一个 **671B 总参 / 37B 激活**的 MoE 模型。其中量化层面最具冲击力的贡献是:**全训练链路使用 FP8 精度**(此前最大规模的公开 FP8 训练)。通过一套精心设计的**分粒度量化 + 在线重标定 + 关键模块保留 BF16** 的工程组合,DeepSeek 实现了 FP8 训练**无精度损失**,训练总成本压到 **~$5.6M**(2048 张 H800,57 天)——比同规模的 GPT-4 / Claude 3 Opus 训练成本低了**一个数量级**。这是 FP8 训练从"实验室 demo"走向"万亿 token 生产级"的里程碑,也为"低精度训练 = 万亿模型基础设施"定下标杆。

![DeepSeek-V3 的 FP8 训练整体流程:GEMM 输入/输出 FP8,accumulate 用 FP32 寄存器提升;关键模块(embedding, output head, norm)保留 BF16 避免累积误差。](/assets/img/deepseek-v3-fp8/x1.png)
_Figure 1:DeepSeek-V3 的全链路 FP8 训练架构_

---

## 背景:FP8 训练的诱惑与难题

### 为什么要做 FP8 训练?

训练 LLM 的算力瓶颈在于 GEMM(矩阵乘法)。H100 / H800 的 **FP8 算力是 BF16 的 2×**:

- BF16: 989 TFLOPs
- FP8 (E4M3): **1979 TFLOPs**

如果能把整条训练链路变成 FP8,理论上**训练速度翻倍,能耗减半**。对万亿级 token 预训练,这意味着数百万美元的成本节约。

### 为什么一直没人大规模做?

FP8 训练难点在于:

1. **数值范围极窄**:E4M3 的最大值约 448,BF16 则能表示 $3 \times 10^{38}$——FP8 很容易溢出
2. **累积误差**:大矩阵乘法要累加几千个 FP8 乘积,误差会快速放大
3. **梯度的极端值**:训练初期梯度可能非常大或非常小,FP8 的动态范围不够
4. **loss 尖峰 / 散度**:FP8 训练时 loss spike 概率显著高于 BF16

之前 Nvidia 的 [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 实现了 FP8 训练框架,但最大规模只公开到 20B 左右。DeepSeek-V3 是第一个公开的 **670B 规模 FP8 预训练**。

---

## 核心机制一:分粒度量化

![DeepSeek-V3 的分粒度量化:Activation 用 per-token / per-128-channel 的 tile-wise scaling;weight 用 per-128 x 128 block 的 block-wise scaling。粒度细到足以控制 outlier,又不增加太多 scale 存储。](/assets/img/deepseek-v3-fp8/x2.png)
_Figure 2:Tile-wise / block-wise 分粒度量化_

### Activation:每 token × 每 128 channel

Activation 的量化粒度:

- 每行(每个 token)独立
- 沿着 channel 维度每 128 个一组

形式上:scale 矩阵大小为 $[N_{\text{tokens}}, \lceil d/128 \rceil]$——细粒度,能捕捉 outlier 的局部分布。

### Weight:每 128 × 128 block

Weight 的量化粒度为 **$128 \times 128$ 方块**,每块一个 FP32 scale。

这个粒度选择不是随机的:

- **H800 的 tensor core 以 128 tokens 为基本 block** 做 GEMM
- Block 与 hardware 对齐 → **量化不增加额外的 shuffle / pack overhead**

### 为什么粒度这么重要?

粗粒度(per-tensor)的 FP8 量化:outlier 一个通道把 scale 占完,其他通道归零。这和 LLM.int8() 发现的问题一样——但现在发生在**训练而非推理**。

Fine-grained(128 级)让 outlier 被"隔离"在一小块内,不污染其他块——这是 DeepSeek-V3 FP8 训练不崩的核心。

---

## 核心机制二:FP32 accumulation

### GEMM 的累积误差

FP8 矩阵乘法 $C = AB$ 中:

$$
C_{ij} = \sum_k A_{ik} B_{kj}
$$

累加几千个 FP8 乘积,每个都有 ~1% 的量化误差。直接累加的话,误差会线性累积到不可接受。

### 解法:寄存器 FP32 累加

![在 tensor core 内部,乘积(FP8 x FP8)结果用 FP32 寄存器累加,避免长累加链的精度损失。MMA 指令本身就支持这种"输入低精度、输出 FP32"模式。](/assets/img/deepseek-v3-fp8/x3.png)
_Figure 3:FP8 GEMM 内部的 FP32 累加_

H100/H800 的 tensor core 支持 "FP8 × FP8 → FP32" 模式:输入是 FP8,但内部乘积和累加用 FP32 寄存器。DeepSeek-V3 充分利用这一特性:

- 输入 activation, weight 都 FP8
- 中间 accumulation 全部 FP32
- 输出 BF16(再传给下一层)

这样**单次 GEMM 的累积误差接近 BF16 水平**——不是 FP8 原生误差。

### DeepSeek 的细节优化

作者还发现 H800 的 FP8 MMA 指令在 CUDA core 阶段有一个**精度退化 bug**:每 128 次累加后,FP32 精度会退化到约 14-bit 尾数。解决方案:**每 128 次累加后主动 flush 到另一个 FP32 buffer**——保持满 23-bit 精度。

这种硬件层面的工程细节是 DeepSeek-V3 训练稳定的关键之一。

---

## 核心机制三:关键模块保留 BF16

![并非所有模块都 FP8:embedding、MoE 门控、output head、所有 norm / softmax / activation function 都保留 BF16。只有 Linear GEMM 用 FP8。这些模块要么参数量小,要么对精度极敏感。](/assets/img/deepseek-v3-fp8/x4.png)
_Figure 4:保留 BF16 的模块划分_

DeepSeek-V3 不是"全模型 FP8"——而是对**数值敏感的模块**保留 BF16:

| 模块 | 精度 | 原因 |
|------|------|------|
| Embedding | BF16 | 参数量相对小,对 ID lookup 精度敏感 |
| Output head (LM head) | BF16 | 直接关系到 softmax 分布,FP8 会压低尾部概率 |
| LayerNorm / RMSNorm | BF16 | 数值范围大,FP8 易溢出 |
| Softmax / attention 内部 | BF16 | 浮点稳定性关键 |
| MoE gate | BF16 | 路由对精度极敏感 |
| **Linear GEMM** | **FP8** | **算力大头,FP8 收益最大** |

这个划分让 FP8 覆盖了 ~95% 的 FLOPs,但避开了最敏感的 5%——鱼与熊掌兼得。

---

## 核心机制四:在线 re-scale

训练中不同 step 的 activation 分布会变化。固定的 per-tensor scale 会在训练后期变得不适配。

DeepSeek-V3 的方案:**每个 step 在线重新计算 scale**。

- 前向之前:看当前 batch 的 activation 统计,算出 per-tile scale
- 反向之前:梯度再算一次 scale
- 所有 scale 在当前 step 结束后丢弃

这个"动态 scaling" 加 tile-wise 粒度,让 DeepSeek-V3 的 FP8 训练能适应整个 14.8T tokens 预训练过程中 activation 分布的演化。

---

## 实验结果

### 精度:FP8 vs BF16 loss 曲线完全重合

![用 1B / 16B MoE 小规模对比训练:FP8 和 BF16 的训练 loss 曲线几乎完全重合,差异 < 0.25%。下游 benchmark 准确率也一致。](/assets/img/deepseek-v3-fp8/x5.png)
_Figure 5:FP8 vs BF16 训练 loss——无显著差异_

在多个规模上验证:

| 模型 | FP8 loss | BF16 loss | Δ |
|------|----------|-----------|---|
| 1B dense | 2.342 | 2.341 | +0.001 |
| 16B MoE | 2.051 | 2.049 | +0.002 |
| 671B MoE (V3) | — | — | **无差异** |

FP8 训练的总精度损失 < **0.25%**——几乎不可感知。

### 成本:$5.6M 训完 671B / 14.8T tokens

公开数据:

- **训练卡**:2048 张 **H800**(美国出口管制版,算力被削)
- **时间**:**57 天**
- **总 GPU-hour**:~2.8M
- **按 $2/GPU-hour 算**:**$5.6M**

对比:

- GPT-4 (1.76T MoE 估计):**$60-100M**
- LLaMA 3 405B:**$30-60M**
- Claude 3 Opus(估计):**$100M+**

DeepSeek-V3 的成本降到行业标杆的 **1/10 - 1/20**——FP8 训练贡献了其中一大部分。

### 质量:对标 GPT-4o / Claude 3.5 Sonnet

DeepSeek-V3 在 MMLU、MATH、HumanEval 等 benchmark 上达到 GPT-4o 和 Claude 3.5 Sonnet 级别,部分任务(数学、代码)甚至超过。

**成本 1/10,质量 100%**——这就是为什么 DeepSeek-V3 在 2024 年末到 2025 年初震动整个行业。

---

## 工程影响

### 1. 证明 FP8 训练是万亿级基础设施

之前 FP8 训练是"学术想象"。DeepSeek-V3 首次在超大规模证明了可行性和稳定性,直接让 Meta、Google、xAI 等公司把 FP8 训练列入下一代预训练的标准工具。**Llama 4、Grok 3、Gemini 2.0 都采用了类似思路**。

### 2. 将硬件工程推向 FP8 主导

NVIDIA 的 Blackwell 架构(B100 / B200)把 FP8 算力进一步提升到 BF16 的 2.5-3×,**为 FP8 训练专门优化 tensor core**。DeepSeek-V3 等工作证明了这个方向的价值。

### 3. "分粒度 + 硬件对齐 + 选择性保留" 的量化训练范式

DeepSeek-V3 的量化方案不是 "暴力 FP8"——而是一整套精细组合。这个**方法论**被后续所有超大规模 FP8 / FP6 训练继承:

- 分粒度量化(tile-wise)
- Scale 与硬件 block 对齐
- 关键模块保留高精度
- 在线动态 re-scale
- Accumulation 用 FP32

### 4. 开源复盘改变行业竞争格局

DeepSeek-V3 **公开模型权重 + 详尽技术报告**,让社区能复现大部分工程细节。这打破了"前沿大模型训练是闭源秘籍"的格局,加速了全球开源 LLM 的进步。

---

## 局限

### 1. 对硬件依赖强

DeepSeek-V3 的 FP8 方案高度依赖 **H100/H800 的 tensor core 行为**。在 A100(无 FP8)或其他硬件上无法复现。

### 2. Activation 不是纯 FP8

Attention 内部(QKV 计算、softmax)仍然 BF16。真正的 FP8 FLOPs 占比虽高(~95%),但不是 100%。

### 3. 训练 recipe 精细

DeepSeek-V3 的稳定性依赖具体的 learning rate schedule、warmup、gradient clipping 等众多细节。直接"搬到"其他模型不一定立即工作。

### 4. FP4 还没能推广到此规模

DeepSeek-V3 用的是 FP8。FP4 训练目前仍在学术阶段——训练稳定性和精度差距更大。能否复制 DeepSeek-V3 的成功到 FP4 还未知。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **FP8 训练是"整套工程",不是"简单换 dtype"**:分粒度、硬件对齐、关键模块保留、在线 rescale、FP32 accumulation——五个组件缺一不可。这是大规模低精度训练的通用模板
2. **硬件与软件必须协同设计**:选 128-block 的粒度不是数学要求,是 H800 tensor core 的物理特性决定的。这种"硬件感知的算法设计"是大模型训练的新常态
3. **"选择性保留"比"全部降精度"更有效**:把 95% FLOPs 压到 FP8,5% 敏感模块保 BF16——取 80/20 最优位置,是工程智慧的体现
4. **开源 + 工程细节公开 = 加速全球进展**:DeepSeek-V3 的 report 让每一家 LLM 公司的训练能力提升一档。低精度训练的"公知化"是 2025 年前沿 LLM 训练成本急剧下降的关键原因
</callout>

---

## 延伸阅读

- [DeepSeek-V3 Hardware Insights 深度解读]({% post_url 2026-03-03-DeepSeek-V3-Hardware-Insights-硬件协同设计深度解读 %}) —— 同团队的硬件协同设计
- [MLA / DeepSeek-V2 深度解读]({% post_url 2026-04-23-MLA-DeepSeek-V2-多头潜在注意力深度解读 %}) —— DeepSeek 的 attention 创新
- [BitNet b1.58 深度解读]({% post_url 2026-04-24-BitNet-b1.58-三值LLM深度解读 %}) —— 更激进的低精度训练
- [FP8 Formats for DL (Micikevicius et al., 2022)](https://arxiv.org/abs/2209.05433) —— FP8 的基础论文
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) —— FP8 训练的主流框架
