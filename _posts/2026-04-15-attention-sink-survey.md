---
layout: post
title: "注意力沉没（Attention Sink）综述：Transformer 中的幽灵现象——利用、解释与缓解"
date: 2026-04-15
categories: [论文解读, Transformer, 大语言模型]
tags: [Attention Sink, LLM, 综述, ViT, MoE, 长上下文]
---

# 注意力沉没（Attention Sink）综述：Transformer 中的幽灵现象

> **论文信息**
> arXiv: [2604.10098](https://arxiv.org/abs/2604.10098)
> 通讯作者：Ngai Wong（香港大学）
> GitHub 论文列表：[Awesome-Attention-Sink](https://github.com/ZunhaiSu/Awesome-Attention-Sink)
> 覆盖文献：180+ 篇研究

---

## 1. 什么是注意力沉没？

在 Transformer 模型中存在一个普遍却令人困惑的现象：**无论输入内容是什么，注意力机制都会将大量权重集中在少数几个特定 token 上**——通常是序列的第一个 token（如 `[BOS]`），即便这些 token 与当前任务毫无语义关联。这种现象被称为**注意力沉没（Attention Sink，AS）**。

![标准 Transformer 与注意力沉没示意图](https://arxiv.org/html/2604.10098v1/x4.png)

*图4：标准 Transformer 结构与注意力沉没现象示意*

### 1.1 具体表现

以 LLaMA 系列模型为例：

- 在 **98%** 的注意力头中，序列第一个 token 接收到最高的注意力权重
- 这种现象在模型所有层中普遍存在，并非仅限于特定层
- 即便将第一个 token 替换为随机内容，注意力沉没仍然发生

![LLaMA-2-7B 各层注意力 logits 分布](https://arxiv.org/html/2604.10098v1/x5.png)

*图5：LLaMA-2-7B 各层注意力 logits 的分布，可见第一个 token 持续获得极高的 logit*

### 1.2 量化定义

研究者提出了注意力沉没强度的量化指标：

$$\tau = \frac{\text{累积注意力分数}}{\text{平均注意力分数}} \geq 1000$$

当某个 token 的累积注意力分数超过均值的 **1000倍**，即可认定其为注意力沉没 token。

---

## 2. 为何注意力沉没值得重视？

![论文发表量增长趋势](https://arxiv.org/html/2604.10098v1/x3.png)

*图6：注意力沉没相关研究的发表数量增长趋势：从2023年约10篇激增至2026年的180+篇*

注意力沉没并非无关痛痒的小 bug，它对模型性能有深远影响：

1. **KV 缓存污染**：在推理加速系统中，沉没 token 的 KV 缓存被无谓地保留，挤占有用信息的空间
2. **长上下文理解退化**：注意力被"吸走"后，真正重要的信息 token 可能被忽视
3. **模型内部结构的窗口**：AS 现象揭示了模型如何组织其内部表示——是解读模型机制的重要线索
4. **跨架构普遍存在**：不仅是 LLM，ViT、MoE、DiT、VLA 模型中均有发现，是 Transformer 家族的系统性问题

---

## 3. 三维分类体系

本综述构建了一个**三维度分类体系**，将180+篇文献系统组织为：利用（Utilization）、解释（Interpretation）与缓解（Mitigation）。

![综述结构概览](https://arxiv.org/html/2604.10098v1/x2.png)

*图7：综述的整体结构框架*

---

## 4. 第一维：基础利用（Fundamental Utilization）

这一维度研究如何**利用**注意力沉没现象来改善模型性能，而非直接消除它。

### 4.1 沉没 Token 保留（Sink Token Preservation）

**代表工作：StreamingLLM（2023）**

StreamingLLM 是最早系统利用 AS 的工作之一。其核心发现：在滑动窗口 KV 缓存中，**只要保留最初几个 token 的 KV 缓存**，模型就能保持稳定的性能，从而支持**无限长度推理**而不需要完整的历史 KV 缓存。

```
滑动窗口 KV 缓存：[沉没tokens] + [最近N个tokens]
                     ↑ 始终保留    ↑ 滑动更新
```

这一发现使得内存受限设备上的长文本推理成为可能。

### 4.2 注意力重分配（Attention Redistribution）

既然沉没 token 接收了"多余"的注意力，一个自然的想法是：**将这些注意力重新分配给真正有意义的 token**。

相关方法通过修改 attention score 的后处理步骤，将过度集中在沉没 token 上的注意力权重均匀或按语义相关性重分配到其他 token。

### 4.3 可学习前缀 Token（Learnable Prefix Tokens）

**代表工作：DINOv2 Register Tokens**

DINOv2 在 ViT 中引入了"注册 token（register tokens）"——一批可学习的前缀 token，专门承接注意力沉没。通过为 AS 提供"合法宣泄口"，保护真正有意义的视觉 token 不受 AS 干扰。

这是一种**主动预防**策略：与其让模型随机选择一个 token 成为沉没点，不如主动设置专用的沉没 token。

### 4.4 沉没 Token 再利用（Sink Token Repurposing）

**代表工作：RetoVLA**

RetoVLA 将沉没 token 从"废物"变为"资源"，把它们作为**寄存器（registers）**在机器人操控任务中存储跨时步的状态信息。

结果令人惊喜：在真实世界机器人操控任务中，性能提升 **+17.1%**，证明沉没 token 虽然在语义上"空洞"，却可以作为有效的状态存储介质。

| 利用策略 | 代表方法 | 核心思路 | 应用场景 |
|---------|---------|---------|---------|
| Sink Token 保留 | StreamingLLM | 始终保留沉没 token 的 KV 缓存 | 长上下文推理 |
| 注意力重分配 | 多种方法 | 将多余注意力导向有意义的 token | 提升理解质量 |
| 可学习前缀 Token | DINOv2 Registers | 设置专用沉没 token 保护语义 token | ViT 预训练 |
| Sink Token 再利用 | RetoVLA | 沉没 token 作为状态寄存器 | 机器人操控 (+17.1%) |

---

## 5. 第二维：机制解释（Mechanistic Interpretation）

这一维度探究**为什么**会出现注意力沉没。

### 5.1 Softmax 无操作理论（Softmax No-Op Theory）

最广泛接受的解释：**Softmax 需要一个"垃圾桶"来处理无关信息**。

当 attention 头认为当前查询不需要提取任何信息时，它仍然必须将注意力权重加总为1（Softmax 的约束）。解决方案就是将所有"多余"的注意力集中到一个固定的 token 上——于是第一个 token 成为了天然的"注意力垃圾桶"。

理论预测：
- 第一个 token 的 value 向量会趋向于零向量（因为它永远不会被"真正"读取）
- 实验证实了这一点：沉没 token 的 value 向量范数显著低于其他 token

### 5.2 异常值回路（Outlier Circuits）

部分研究从电路层面分析 AS，发现负责产生沉没现象的**特定注意力回路**。这些"异常值回路"形成了稳定的内部结构，解释了为何 AS 在微调后仍然持续存在。

### 5.3 隐式注意力偏置（Implicit Attention Bias）

训练数据中的统计规律可能导致模型学习到对初始 token 的隐式偏好。尤其是在自回归训练中，第一个 token 天然对所有后续 token 可见，使其成为"注意力集散地"的概率更高。

### 5.4 几何锚定（Geometric Anchoring）

从表示几何角度，沉没 token 在嵌入空间中充当**参考锚点**——其他 token 的注意力模式以沉没 token 为基准组织。这一理论预测 AS 与模型的泛化能力之间存在正相关关系。

| 理论 | 核心主张 | 关键预测 |
|------|---------|---------|
| Softmax 无操作 | AS 是 Softmax 约束的副产品 | 沉没 token value 向量趋近零 |
| 异常值回路 | 特定神经回路负责产生 AS | 可定位、可删除相关回路 |
| 隐式注意力偏置 | 训练数据统计导致位置偏好 | AS 随训练数据分布变化 |
| 几何锚定 | 沉没 token 是表示空间的参考点 | AS 与泛化能力正相关 |

---

## 6. 第三维：策略缓解（Strategic Mitigation）

这一维度研究如何**减弱或消除**注意力沉没的负面影响。

### 6.1 门控注意力（Gated Attention）

通过引入显式门控机制，允许模型动态调节是否激活沉没行为。门控信号由可学习参数控制，使模型可以根据上下文决定是否需要"使用"注意力沉没。

### 6.2 修改 Softmax（Modified Softmax）

**代表工作：SoFo / Sigmoid 注意力**

将标准 Softmax 替换为 Sigmoid 函数：

$$\text{Attention}(Q, K, V) = \sigma(QK^T / \sqrt{d}) \cdot V$$

Sigmoid 不要求权重加总为1，因此不存在"必须把注意力放在某处"的约束，从根本上消除了 AS 产生的数学基础。

### 6.3 可学习注意力偏置（Learnable Attention Bias）

在 attention logit 中添加可学习的位置偏置项，允许模型学习"抵消"对沉没 token 的过度关注。

### 6.4 预训练干预（Pre-training Interventions）

**代表工作：OSP（Outlier-Suppressed Pre-training）**

在预训练阶段主动设计训练策略，防止模型形成 AS 模式。OSP 通过修改训练目标，显式惩罚过于集中的注意力分布，从源头防止 AS 的形成。

| 缓解策略 | 代表方法 | 修改位置 | 优势 | 劣势 |
|---------|---------|---------|------|------|
| 门控注意力 | 多种 | 注意力计算 | 灵活自适应 | 增加参数 |
| 修改 Softmax | SoFo、Sigmoid 注意力 | 激活函数 | 从根本消除 AS | 需要重新训练 |
| 可学习注意力偏置 | 多种 | Logit 层 | 可微调 | 效果有限 |
| 预训练干预 | OSP | 训练过程 | 最彻底 | 代价最高 |

---

## 7. 跨架构分析：AS 无处不在

### 7.1 经典语言模型（BERT/RoBERTa）

在 BERT 等双向编码器中，`[CLS]` 和 `[SEP]` token 充当注意力沉没点。有趣的是，`[CLS]` 本身被设计为句子级表示，AS 与其功能定位存在天然重叠。

### 7.2 因果语言模型（GPT/LLaMA 系列）

![Decoder-only LLM 结构](https://arxiv.org/html/2604.10098v1/x6.png)

*图8：Decoder-only LLM 结构中的注意力沉没现象*

最典型的 AS 场景。LLaMA 系列中，98% 的注意力头将序列第一个 token 设为沉没点。这一发现已被大量后续工作证实。

### 7.3 混合专家模型（MoE）

![MoE LLM 架构](https://arxiv.org/html/2604.10098v1/x7.png)

*图9：MoE LLM 架构中的注意力沉没与"超级专家"现象*

MoE 模型中的 AS 发现了一个惊人的"超级专家（Super Expert）"现象：

**Qwen3-30B-A3B 的发现**：

![Qwen3 Super Expert 分析](https://arxiv.org/html/2604.10098v1/Figures/Section_2/Qwen3-30B-A3B-layer_3_sink_token_avg_logits_c4.png)

*图10：Qwen3-30B-A3B 中沉没 token 的专家激活 logits 分布*

![DeepSeek-V2-Lite 分析](https://arxiv.org/html/2604.10098v1/Figures/Section_2/DeepSeek-V2-Lite-layer_3_sink_token_avg_logits_c4.png)

*图11：DeepSeek-V2-Lite 中类似的超级专家激活模式*

在 Qwen3-30B-A3B 的 6,144 个专家中，仅有 **3个"超级专家"**专门处理沉没 token 的路由。移除这3个专家会导致**灾难性崩溃**——模型性能急剧下降，即便整体参数减少不足0.05%。

这一发现深刻揭示了 MoE 模型中存在不成比例的"关键少数"专家，对模型压缩和剪枝研究具有重大警示意义。

### 7.4 多模态大语言模型（MLLM）

![视觉注意力沉没](https://arxiv.org/html/2604.10098v1/x8.png)

*图12：多模态大语言模型中的视觉注意力沉没现象*

在处理图像的多模态 LLM 中，背景区域对应的视觉 token 往往成为注意力沉没点。这与直觉一致：背景像素通常携带较少语义信息，因此成为"注意力垃圾桶"的候选。

### 7.5 视觉 Transformer（ViT）

![ViT 中的异常值](https://arxiv.org/html/2604.10098v1/Figures/Section_2/outliers.png)

*图13：ViT 中背景 patch 上的注意力异常值分布*

在 ViT 中，对应图像**背景 patch** 的 token 容易成为沉没点。DINOv2 通过引入 register tokens（可学习前缀 token）成功缓解了这一问题，使视觉特征的空间一致性显著提升。

### 7.6 扩散 Transformer（DiT）

![DiT 中的注意力沉没](https://arxiv.org/html/2604.10098v1/Figures/Section_2/dit.png)

*图14：扩散 Transformer（DiT）中的注意力沉没现象*

在 Stable Diffusion 3 等 DiT 架构中同样发现了 AS，但其表现形式与 LLM 有所不同：AS 主要集中在特定的时间步嵌入 token 上。

### 7.7 视觉-语言-动作模型（VLA）

在机器人控制领域的 VLA 模型中，AS 出现在状态和动作序列的边界 token 上。RetoVLA 正是利用这一现象，将沉没 token 改造为动作状态寄存器，实现了 +17.1% 的操控性能提升。

---

## 8. 九大应用领域

本综述覆盖 AS 研究在以下9个应用领域的工作：

| 应用领域 | 核心问题 | 代表方法 |
|---------|---------|---------|
| 预训练 | 如何在训练时预防 AS | OSP、Sigmoid 注意力 |
| 微调 | 微调如何影响 AS | Adapter 对 AS 的影响研究 |
| 推理效率 | 利用/规避 AS 加速推理 | StreamingLLM、KV 缓存优化 |
| 可解释性 | AS 揭示了哪些模型机制 | 注意力回路分析 |
| 幻觉减少 | AS 与幻觉的关联 | 注意力重分配方法 |
| 安全性 | AS 是否影响安全对齐 | 安全 token 的 AS 分析 |
| 通用能力 | AS 如何影响推理能力 | Registers、门控注意力 |
| 长上下文 | 长序列中 AS 的动态变化 | StreamingLLM 及后续工作 |
| 多模态 | 跨模态 AS 的特殊性 | ViT Registers、RetoVLA |

---

## 9. 开放问题与未来方向

尽管已有180+篇研究，AS 领域仍存在重要的开放问题：

### 9.1 因果性问题
AS 到底是模型能力的**原因**还是**结果**？消除 AS 真的能提升性能，还是模型会找到其他"垃圾桶"？

### 9.2 跨架构一致性
不同架构（BERT vs GPT vs ViT）中 AS 的产生机制是否统一？不同架构需要不同的缓解策略吗？

### 9.3 训练动态
AS 是如何在训练过程中逐步形成的？是否存在特定的训练阶段或数据分布触发了 AS？

### 9.4 对齐影响
RLHF 等对齐训练如何改变 AS 的分布？对齐和安全性是否依赖于特定的 AS 模式？

### 9.5 MoE 超级专家
除了3个超级专家之外，MoE 模型中是否存在其他"关键少数"？这对 MoE 模型的理论分析有何启示？

---

## 10. 总结

注意力沉没是 Transformer 家族中一个普遍、持久且影响深远的现象。本综述系统梳理了：

- **现象定义**：98% 的 LLaMA 注意力头以第一个 token 为沉没点，量化阈值 τ = 1000
- **跨架构普遍性**：从 BERT 到 GPT，从 ViT 到 DiT，从 LLM 到 VLA，AS 无处不在
- **实用价值**：StreamingLLM 用 AS 实现无限长度推理；RetoVLA 用 AS 提升机器人操控 +17.1%
- **惊人发现**：MoE 超级专家——3个专家支撑整个 Qwen3-30B-A3B 的沉没机制
- **缓解路径**：从 Sigmoid 注意力（改激活函数）到 OSP（改预训练）的完整方案谱系

随着 LLM 不断向更长上下文、更大规模发展，对 AS 的深入理解将成为构建更高效、更可靠 Transformer 模型的关键。

---

*本文基于论文 [Attention Sink in Transformers: A Survey on Utilization, Interpretation, and Mitigation](https://arxiv.org/abs/2604.10098) 整理撰写。更多相关论文见 [Awesome-Attention-Sink](https://github.com/ZunhaiSu/Awesome-Attention-Sink)。*
