---
title: "SmoothQuant — 把 activation 的 outlier 难度'搬'到 weight 上"
date: 2026-04-24 13:15:00 +0800
categories: [Quantization, Inference Optimization]
tags: [smoothquant, w8a8, outlier-migration, xiao-2022]
math: true
---

## 基本信息

- **作者**: Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
- **机构**: MIT, NVIDIA
- **发表**: ICML 2023
- **arXiv**: [2211.10438](https://arxiv.org/abs/2211.10438)

## 一句话总结

提出 **SmoothQuant**——针对 LLM.int8() 发现的 activation outlier 问题,给出一种**纯数学等价**的解法:把 activation 的幅度"挪"到 weight 上。具体做法是对每个通道引入一个 scale $s_j$:activation 除以 $s_j$,weight 乘以 $s_j$,矩阵乘法结果不变,但 activation 变"平滑",weight 变"陡峭"一点——两者都可以 INT8,无需混合精度。这让 **W8A8 全量 INT8 推理**首次在 175B 模型上实用,速度 1.5-2× FP16,质量几乎不掉。

![SmoothQuant 的核心思想:activation 有 outlier 难量化,weight 分布平易量化。通过迁移因子 $s$ 把 activation 的"难度"转给 weight,两者都变成容易量化的形状。](/assets/img/smoothquant/x1.png)
_Figure 1:SmoothQuant 的"难度迁移"核心思想_

---

## 背景:LLM.int8() 的遗留问题

LLM.int8() 发现 activation outlier 并用**混合精度**解决——outlier 走 FP16,其他 INT8。但混合精度带来两个问题:

1. **硬件效率打折**:GPU 的 Tensor Core 对 INT8 × INT8 加速最好;FP16 + INT8 混合需要"分两次算再加",kernel 复杂度高
2. **没法走 W8A8 全 INT8 算力**:H100 等新硬件对**同时 INT8 权重和 INT8 激活**的加速比更高,LLM.int8() 用不上

SmoothQuant 的目标:**让 activation 和 weight 同时变成容易量化的形状,走纯 INT8 路径**。

---

## 核心机制:数学等价的 outlier 迁移

### 数学基础

矩阵乘法 $Y = X W$。引入一个**对角 scale 矩阵** $\text{diag}(s) \in \mathbb{R}^{d \times d}$:

$$
Y = X W = X \cdot \text{diag}(s)^{-1} \cdot \text{diag}(s) \cdot W = \hat X \cdot \hat W
$$

其中:

- $\hat X = X \cdot \text{diag}(s)^{-1}$:activation 第 $j$ 列除以 $s_j$
- $\hat W = \text{diag}(s) \cdot W$:weight 第 $j$ 行乘以 $s_j$

**数学上完全等价**——$Y$ 没变。但 $\hat X$ 和 $\hat W$ 的分布变了。

### 关键:选择 $s_j$ 让 outlier 平滑

![不同 $\alpha$ 值对 activation/weight 分布的影响:$\alpha = 0$ 完全不改(outlier 仍在 X),$\alpha = 1$ 全转移到 W(W 爆炸),$\alpha \approx 0.5$ 是平衡点。](/assets/img/smoothquant/x2.png)
_Figure 2:$\alpha$ 控制迁移比例——0.5 是甜蜜点_

$s_j$ 设计为:

$$
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}
$$

其中:

- $\max(|X_j|)$:第 $j$ 列 activation 的最大幅度
- $\max(|W_j|)$:第 $j$ 行 weight 的最大幅度
- $\alpha \in [0, 1]$:**迁移强度** hyperparameter

直觉:

- $\alpha = 0$:$s_j = 1/\max(|W_j|)$——weight 归一化,不改 activation
- $\alpha = 1$:$s_j = \max(|X_j|)$——activation 归一化,全部难度转给 weight
- $\alpha \approx 0.5$:**activation 和 weight 都变得相对平滑**——最佳

### 为什么 $\alpha = 0.5$ 是最佳

![OPT-175B 某一层 activation 的 per-channel max 分布:原始有 outlier(>40),SmoothQuant 后 outlier 消失(< 10),分布变平滑,INT8 可以精确表示。](/assets/img/smoothquant/x3.png)
_Figure 3:SmoothQuant 后 activation 的分布变化_

activation 的 max 分布极其不均(少数通道超大),weight 的 max 分布相对均匀。

- 完全保留 activation 的"剧烈分布"——量化困难
- 完全转移到 weight——weight 变剧烈,weight 也难量化了

$\alpha = 0.5$ 让**两者都变成中等分布**——都能用 INT8 精确表示。

### 离线计算,无需改变前向

关键工程特性:**$s_j$ 是静态的**——离线用少量校准数据统计一次,写入网络 weight 里。

- Inference 阶段**没有额外计算**
- $\hat W$ 被预乘后替换原 $W$
- $\hat X = X / s$ 融入前一层的 output rescaling(如 LayerNorm 的 $\gamma$)——一样没有额外 op

所以 SmoothQuant 是**零开销**的 transform——应用后直接走标准 W8A8 INT8 kernel。

---

## 完整量化流程

1. **校准**:用 ~128 个样本跑一遍,收集每层 activation 和 weight 的 per-channel max
2. **计算迁移因子**:$s_j = \max(|X_j|)^{0.5} / \max(|W_j|)^{0.5}$
3. **迁移**:$\hat W = \text{diag}(s) W$,$\hat X = X / s$(吸收进前层 norm)
4. **量化**:对 $\hat X$ 和 $\hat W$ 都做标准 INT8 量化(per-tensor 或 per-token)
5. **推理**:走 W8A8 INT8 GEMM kernel

---

## 实验结果

### 质量:接近 FP16

![SmoothQuant 在 OPT/BLOOM/GLM 系列上的 PPL 对比:与 FP16 baseline 几乎一致,远好于 W8A8 naive 量化(普遍崩溃)。](/assets/img/smoothquant/x4.png)
_Figure 4:各模型 PPL 对比_

关键数字(OPT-175B on WikiText-2):

| 方法 | PPL |
|------|-----|
| FP16 | 8.34 |
| W8A8 Naive | 18.5 |
| LLM.int8() | 8.35 |
| **SmoothQuant** | **8.38** |
| **SmoothQuant + per-token** | **8.36** |

质量与 LLM.int8() 持平,但走纯 INT8 路径,速度更快。

### 速度:1.5-2× FP16

![A100 上 OPT-175B 不同长度的推理速度:SmoothQuant 比 FP16 快 1.5-1.8×,显著超过 LLM.int8() 的 1.1×。](/assets/img/smoothquant/x5.png)
_Figure 5:SmoothQuant 的推理加速_

- OPT-175B:FP16 baseline → SmoothQuant 快 **1.51× 吞吐**
- 显存:FP16 的 **51%**(INT8 权重 + INT8 激活的 KV cache)

---

## 工程影响

### 1. W8A8 成为 INT8 推理的事实标准

SmoothQuant 让"weight INT8 + activation INT8"从"理论上可能但不实用"变成"一行脚本就能部署"。NVIDIA TensorRT-LLM、OpenVINO、vLLM 等推理引擎都支持 SmoothQuant workflow。

### 2. 启发"旋转消除 outlier"路线

SmoothQuant 的核心洞察是:**outlier 可以通过数学变换"挪走"**。这启发了 2024 年的一批工作:

- **Quarot** / **SpinQuant**:用正交旋转矩阵更彻底地消除 outlier
- **OmniQuant**:把 $\alpha$ 变成 learnable
- **DuQuant** / **QServe**:refined 版本

这整条路线的根在 SmoothQuant。

### 3. 简单到难以置信

SmoothQuant 的核心只是 $Y = X W = (X/s)(sW)$ 这个初中代数恒等式。但把它**系统化、找到最佳 $s$、证明能工作**——这是工程研究的价值。

### 4. 推动 W4A4 / 超低精度量化

SmoothQuant 在 W8A8 上成功后,类似思想被推广到 W4A4(INT4 权重 + INT4 激活)。QServe、Atom 等现代低精度推理引擎都借鉴了"迁移消除 outlier"思路。

---

## 局限

### 1. 只处理"通道维 outlier"

SmoothQuant 针对的是**特定通道幅度远大于其他通道**——这是 LLM 的主要 outlier pattern。但某些模型可能有"特定 token 位置的 outlier"或"跨 head 的 outlier",SmoothQuant 处理不了。

### 2. $\alpha$ 需要调参

$\alpha = 0.5$ 是对 OPT/BLOOM 系列的经验值。对其他架构(LLaMA、Mistral、DeepSeek)可能要重新搜索。后续 OmniQuant 解决了这个问题(让 $\alpha$ learnable)。

### 3. 需要校准数据

不像 LLM.int8() 可以零数据动态检测 outlier,SmoothQuant 需要 128 个样本左右的 calibration data——小但不是零。

### 4. 对超大 outlier 仍有残余

极个别 activation 值(如 > 100)经过 $\alpha = 0.5$ 迁移后仍然幅度大。后续工作(QServe 等)加入 clip 或混合精度再兜底。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Outlier 是可以"搬家"的**:数学等价变换 $Y = (X/s)(sW)$ 让 outlier 在 X 和 W 之间重新分配——正视 outlier 而不是回避它,这是 SmoothQuant 的哲学
2. **$\alpha = 0.5$ 的对称平衡点**:把 outlier 压力平均分给 activation 和 weight,两者都变成 INT8 能表示的形状。这种"把一个难问题对称化"的思路值得迁移到其他场景
3. **零推理开销的预变换**:SmoothQuant 的 scale 可以预融合到前层 norm,不改变推理图结构——这是它能广泛落地的关键工程优势
4. **SmoothQuant 开启的 outlier 处理路线至今活跃**:Quarot、SpinQuant、OmniQuant 等 2024 年工作都可追溯到"迁移 outlier"思路。理解 SmoothQuant 就理解了半部 LLM 量化史
</callout>

---

## 延伸阅读

- [LLM.int8() 深度解读]({% post_url 2026-04-24-LLM-int8-混合精度量化深度解读 %}) —— outlier 现象的首次发现
- [AWQ 深度解读]({% post_url 2026-04-24-AWQ-激活感知权重量化深度解读 %}) —— 另一条基于 activation 信息的路线
- [GPTQ 深度解读]({% post_url 2026-04-24-GPTQ-权重量化二阶误差补偿深度解读 %}) —— 纯权重量化代表
- [OmniQuant (Shao et al., 2023)](https://arxiv.org/abs/2308.13137) —— SmoothQuant 的 learnable 版本
- [Quarot (Ashkboos et al., 2024)](https://arxiv.org/abs/2404.00456) —— 用旋转更彻底消除 outlier
