---
layout: post
title: "SpectralSplats：基于频谱矩监督的鲁棒可微分追踪"
date: 2026-03-27
categories: [论文解读, 三维视觉]
tags: [3D Gaussian Splatting, 可微分渲染, 目标追踪, 频谱分析, 频率退火, 动态场景重建]
---

> **论文**：SpectralSplats: Robust Differentiable Tracking via Spectral Moment Supervision
> **arXiv**：[2603.24036](https://arxiv.org/abs/2603.24036)
> **机构**：Technion – Israel Institute of Technology & NVIDIA
> **作者**：Avigail Cohen Rimon, Amir Mann, Mirela Ben Chen, Or Litany

---

## 一句话总结

SpectralSplats 将 3D Gaussian Splatting 的追踪优化从空间域迁移至频率域，利用频谱矩监督彻底解决了"梯度消失"问题，使模型在渲染图与目标图像无空间重叠时依然能够收敛，在合成与真实视频数据集上全面超越基线方法。

---

## 背景与问题

### 3D Gaussian Splatting (3DGS) 中的可微分追踪

基于模型的视频追踪通过最小化光度误差估计运动参数 Θ：

$$L_{\text{photo}}(\Theta) = \frac{1}{2} \int \|I_{\text{rend}}(p;\Theta) - I_{\text{gt}}(p)\|_2^2 \, dp$$

这一框架在 3DGS 中表现优秀——3DGS 将场景表示为一组带位置、协方差、不透明度和球谐系数的三维高斯基元，可以端到端可微分渲染。

### 核心痛点：梯度消失问题

然而，标准光度损失存在一个根本性缺陷：**当渲染图像与目标图像没有空间重叠时，梯度严格为零**。

对光度损失梯度进行分解（论文 Eq. 2），可以识别出两项：

- **自项（Self-Term）**：对平移不变，对平行方向位移评估为零
- **目标监督项（Target Supervision）**：当渲染图与目标图空间不相交时完全消失

3DGS 的 tile-based 光栅化器将屏幕划分为 16×16 的区块，并使用 99% 置信区间裁剪基元，进一步强化了梯度归零的问题。

**现有方案的局限性：**
- LPIPS 深度特征：扩大了吸引域，但仍需要一定程度的空间重叠
- 类别专用先验（SMPL、3D Menagerie）：牺牲了方法的通用性

![梯度消失问题示意](https://arxiv.org/html/2603.24036v1/x1.png)
*图 1：零重叠初始化下的对比。左列：pixel-only 方法因梯度消失无法收敛；右列：SpectralSplats 成功完成追踪。*

---

## 核心方法

### 关键洞见：频域提供全局梯度

**正弦基函数是全局的**——空间位移对应频率域中的相位偏移，即使渲染图与目标图完全不重叠，频谱域中的梯度也不为零。这是整个方法的理论基石。

### 3.1 图像矩与频谱对偶性

矩匹配目标函数：

$$L_{\text{moment}}(\Theta) = \frac{1}{2}(M_{\text{rend}}(\Theta) - M_{\text{gt}})^2$$

其中 $M = \int I(p) F(p) \, dp$。

利用分部积分，梯度为：

$$\nabla_\Theta M_{\text{rend}} = \int I_{\text{rend}}(p;\Theta) \nabla_p F(p) \, dp$$

**频谱矩**（Eq. 6）定义为：

$$M(k_x, k_y; I) = \sum_p I(p) \cdot \exp(-j \, \omega_{k_x,k_y}^T p)$$

由 **Parseval 定理**，完整的频谱基等价于空间 L2 损失——因此有选择地控制频率至关重要。

### 3.2 频率退火调度（Frequency Annealing Schedule）

**相位包裹（Phase-wrapping）**是频域优化的核心挑战。当位移 $d_t$ 满足 $|\omega^T d_t| \geq \pi$ 时，会出现伪极小值（false minima），导致优化陷入错误的吸引域。

安全盆条件：$|\omega^T d_t| < \pi$

**余弦加权函数**（Eq. 7）：

$$w_k(t) = \frac{1 - \cos(\pi \cdot \text{clamp}(\alpha(t) - k, 0, 1))}{2}$$

其中 $\alpha(t)$ 从 0 线性增长到 K。

**保守策略**：暖机阶段保持 $\alpha(t)$ 不变，随后以**频率幅度线性**（次指数）扩展，而非按对数索引增长——始终保持在相位包裹阈值之下。

![1D 优化景观分析](https://arxiv.org/html/2603.24036v1/x2.png)
*图 2：1D 优化景观分析。蓝色：高频下的多极小值陷阱；橙色：低频下的全局吸引域；绿色：频率退火路径安全收敛。*

![暖机阶段对比](https://arxiv.org/html/2603.24036v1/x3.png)
*图 3：长暖机（左）vs 短暖机（右）对尾部细节恢复的影响，说明了保守退火策略的重要性。*

### 3.3 完整训练目标

**频谱损失**（Eq. 22）同时监督 RGB 与不透明度：

$$L_{\text{spectral}} = \sum_k w_k(t) \|M_k(I_{\text{rend}}) - M_k(I_{\text{gt}})\|_1 + \lambda_{\text{mask}} \cdot [\text{opacity term}]$$

**空间损失**（Eq. 23）：

$$L_{\text{pixel}} = \|I_{\text{rend}} - I_{\text{gt}}\|_2^2 + \|I_{\text{rend}} \odot O_{\text{rend}} - I_{\text{gt}} \odot O_{\text{gt}}\|_2^2 + \lambda_{\text{bce}} \text{BCE}(O_{\text{rend}}, O_{\text{gt}})$$

**总损失**：

$$L = \lambda_{\text{image}} L_{\text{image}} + \lambda_{\text{arap}} E_{\text{arap}}$$

训练策略：先用频谱阶段建立空间重叠，再过渡到空间损失进行精细优化。

---

## 实验结果

### 数据集与设置

**SC4D 数据集**：SC4D + Consistent4D 资产的合成 4D 动画，干净可控的测试环境。

**GART Dog 数据集**：真实单目视频（2022 National Dog Show + Adobe Stock），包含光照不一致和未知相机视角等挑战。

**变形参数化方案**：
1. **MLP 参数化**：TimeNet 预测时变控制点变形
2. **直接变形场**：直接优化位置偏移和旋转

### Table 1：SC4D 合成数据集结果（位移半径 = 0.5）

| 配置 | 方法 | LPIPS↓ | PSNR↑ | SSIM↑ | NV-LPIPS↓ | NV-PSNR↑ | NV-SSIM↑ |
|---|---|---|---|---|---|---|---|
| MLP + LPIPS | Pixel | 0.0852 | 23.61 | 0.9409 | 0.1153 | 18.32 | 0.9304 |
| MLP + LPIPS | **Ours** | **0.0489** | **27.15** | **0.9546** | **0.0948** | **19.20** | **0.9331** |
| MLP w/o LPIPS | Pixel | 0.1806 | 17.67 | 0.9108 | 0.2023 | 14.04 | 0.9107 |
| MLP w/o LPIPS | **Ours** | **0.0516** | **26.70** | **0.9507** | **0.1331** | **17.40** | **0.9159** |
| Direct + LPIPS | Pixel | 0.3133 | 11.66 | 0.8297 | 0.2443 | 12.38 | 0.8727 |
| Direct + LPIPS | **Ours** | **0.2000** | **15.46** | **0.8701** | **0.2501** | **12.79** | **0.8675** |
| Direct w/o LPIPS | Pixel | 0.2289 | 16.13 | 0.8562 | 0.2774 | 12.07 | 0.8491 |
| Direct w/o LPIPS | **Ours** | **0.1868** | **17.86** | **0.8789** | **0.2640** | **12.51** | **0.8598** |

MLP 参数化下，SpectralSplats 将 PSNR 从 23.61 提升至 27.15（+3.54 dB），LPIPS 降低 43%。

### Table 2：GART 真实视频数据集逐犬结果（位移半径 = 0.6）

| 犬种 | LPIPS Pixel | LPIPS Ours | PSNR Pixel | PSNR Ours | SSIM Pixel | SSIM Ours |
|---|---|---|---|---|---|---|
| Alaskan | 0.2875 | **0.2664** | 20.01 | **20.63** | 0.8793 | **0.8845** |
| Shiba | 0.2749 | **0.1788** | 20.82 | **25.36** | 0.9069 | **0.9344** |
| Hound | 0.3406 | **0.2514** | 16.28 | **19.45** | 0.8372 | **0.8769** |
| Corgi | 0.1164 | **0.1100** | 25.45 | **26.53** | 0.9497 | **0.9561** |
| French | 0.3038 | **0.2339** | 17.61 | **20.88** | 0.8888 | **0.9106** |
| English | 0.2367 | 0.2418 | 21.27 | **21.33** | 0.8939 | **0.8938** |
| Pitbull | 0.2505 | **0.2340** | 19.63 | **20.24** | 0.8851 | **0.8937** |
| **均值** | 0.2586 | **0.2166** | 20.15 | **22.06** | 0.8915 | **0.9071** |

在 7 只真实视频犬中，6 只获得改善，平均 PSNR 提升 +1.91 dB，LPIPS 降低 16%。

### 定量分析：位移半径的影响

![PSNR/SSIM/LPIPS vs 位移半径](https://arxiv.org/html/2603.24036v1/x4.png)
*图 4：不同位移半径下的性能对比（左：GART；右：SC4D）。随着位移增大，SpectralSplats 的优势愈发显著，在梯度消失最严重时表现出最大改善。*

### 定性对比

![SC4D 定性对比](https://arxiv.org/html/2603.24036v1/figures/SC4DQualitative.png)
*图 5：SC4D 合成数据集定性对比。像素监督方法出现模糊和结构错误，SpectralSplats 恢复出更清晰的细节。*

![GART 定性对比](https://arxiv.org/html/2603.24036v1/x5.png)
*图 6：GART 真实视频数据集定性对比。SpectralSplats 在毛发细节、四肢边界等困难区域均表现更优。*

### Table 3：空间损失消融研究（GART，位移=0.6，PSNR）

| 方法 | MSE | +Masked MSE | +BCE | All |
|---|---|---|---|---|
| MLP + Ours | 20.65 | 20.21 | 20.55 | **22.06** |
| MLP + Pixel | 16.45 | 16.14 | 20.00 | **20.15** |

BCE 损失的引入对两种方法都有显著提升，完整损失组合效果最佳。

### Table 4：对齐设置下的验证（位移=0.0）

即使初始化已对齐，SpectralSplats 仍持续匹配或超越像素监督方法——证明频谱监督在良好初始化情况下不会造成性能退化。

---

## 方法对比与技术分析

### 与 BARF 的异同

BARF 在位置编码上施加频谱退火用于相机注册，而 SpectralSplats：
- **直接作用于渲染输出**（而非位置编码）
- 提供了退火调度的**第一性原理推导**
- 专门针对追踪中的梯度消失问题设计

### 与 MomentsNeRF 的异同

MomentsNeRF 用矩约束进行少样本渲染，SpectralSplats 将其扩展为**追踪优化的主要监督信号**，并配以系统性的频率管理机制。

### 频率退火的数学保证

附录中的完整推导证明：
- 对于渲染图像为目标图像的位移副本 $I_{\text{rend}}(p) = I_{\text{gt}}(p-d)$，Fourier 位移定理给出 $M(\omega; I_{\text{rend}}) = M(\omega; I_{\text{gt}}) \cdot \exp(-j\omega^T d)$
- 频谱损失化简为 $E(d;\omega) = |M_{\text{gt}}|^2(1 - \cos(\omega^T d))$
- 梯度为 $\nabla_d E = |M_{\text{gt}}|^2 \sin(\omega^T d) \cdot \omega$
- 伪极小值出现在 $\omega^T d = 2n\pi$（$n \neq 0$）
- 安全盆：$|\omega^T d_t| < \pi$
- 在对数频率网格（$\|\omega_k\| \propto 2^k$）上，安全带宽以 $\gamma^{-t}$ 增长，需要频率索引 $k(t)$ **线性增长**

---

## 局限性与未来方向

**当前局限性：**
- 需要预初始化的规范资产（canonical asset），尚不支持从零开始的完全动态场景重建
- 极端遮挡或快速运动场景下的鲁棒性有待进一步验证

**未来方向：**
- 扩展至完整动态场景重建（无需预初始化）
- 探索替代矩类型（如 Zernike 矩、小波矩）
- 与基于扩散模型的先验相结合，进一步扩大追踪的稳健范围

---

## 总结

SpectralSplats 以优雅的数学框架解决了 3DGS 追踪中长期存在的梯度消失问题。核心贡献可以概括为三点：

1. **问题定性**：首次系统性地分析了 3DGS 光度损失中梯度消失的根本原因，将其分解为自项和目标监督项两个组成部分

2. **频谱矩监督**：将追踪优化迁移至频率域，利用正弦基函数的全局性提供非零梯度，从根本上解决了零重叠情况下的优化问题

3. **原理性退火调度**：通过严格的数学推导给出了频率退火的保守策略，在避免相位包裹的同时逐步引入高频细节监督

实验结果表明，在合成和真实视频数据集上，SpectralSplats 在所有评估指标（PSNR、SSIM、LPIPS）和所有参数化方案下均一致优于基线方法，尤其在大位移（梯度消失最严重）时优势最为显著。这一工作不仅解决了实际问题，更为基于渲染的追踪优化提供了新的理论视角。
