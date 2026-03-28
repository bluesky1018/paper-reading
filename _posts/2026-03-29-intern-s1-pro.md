---
layout: post
title: "Intern-S1-Pro：万亿参数规模的科学多模态基础模型"
date: 2026-03-29
categories: [论文解读, 多模态大模型]
tags: [多模态模型, 科学推理, 视觉语言模型, 大规模预训练]
---

> 📄 **论文**：Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale
> 🔗 **arXiv**：[2603.25040](https://arxiv.org/abs/2603.25040)
> 🏢 **机构**：上海AI实验室 / 商汤科技 / 港中文

## 一句话总结
发布Intern-S1-Pro科学多模态基础模型，在数学、物理、化学等科学领域多模态推理上达到业界最优水平

## 背景与问题
科学计算与推理是人工智能迈向通用智能的核心挑战之一。现有多模态大模型在处理科学图表、公式推导、实验数据分析等任务时往往表现欠佳，主要原因在于缺乏针对科学领域的大规模高质量训练数据和专门的模型架构设计。

随着GPT-4V、Claude 3等多模态大模型的兴起，学术界和工业界对于构建能够真正理解科学内容的AI系统产生了浓厚兴趣。科学文献中包含大量复杂图表、化学结构式、数学公式和物理示意图，这些内容对多模态模型提出了远超日常图像理解的挑战。

书面科学知识的多模态理解涵盖多个子能力：图表读取与数据提取、公式识别与推导、科学常识推理以及跨模态的综合分析。当前主流多模态模型在这些维度上仍存在明显差距，亟需专门针对科学领域进行优化的基础模型。

## 核心方法
Intern-S1-Pro通过以下核心创新构建科学多模态能力：

**1. 万亿参数混合专家架构（MoE at Trillion Scale）**：采用Mixture-of-Experts架构将模型扩展至万亿参数量级，其中科学专用专家模块专门处理公式、图表等科学元素，通用专家模块处理自然语言和日常图像。

**2. 科学数据流水线**：构建了一套专门针对科学文献（arXiv、教材、实验报告等）的数据采集与清洗流水线，涵盖数学、物理、化学、生物、材料等多个学科，数据规模超过500亿token。

**3. 科学视觉编码器**：在标准ViT基础上增加专门的科学图像预训练阶段，重点提升对函数图像、分子结构图、电路图等科学图表的表示能力。

**4. 链式推理增强训练（Chain-of-Thought for Science）**：针对科学推理设计了专门的CoT数据合成策略，引导模型学习step-by-step的科学问题解决方法。


![Figure 1 : The SAGE (Synergistic Architecture for Generalizable Experts, including three layers, Foundation, Fusion, and Evolution) framework used in Intern-S1-Pro development, illustrating the core c...](https://arxiv.org/html/2603.25040/2603.25040v1/x1.png)
*图1：Figure 1 : The SAGE (Synergistic Architecture for Generalizable Experts, including three layers, Foundation, Fusion, and Evolution) framework used in Intern-S1-Pro development, illustrating the core c...*


![Figure 2 : Left: Illustration of the expert expansion process from Intern-S1 to Intern-S1-Pro. The grouped routing strategy ensures well-trained Top-1/Top-2 experts are distributed across groups to ma...](https://arxiv.org/html/2603.25040/2603.25040v1/x2.png)
*图2：Figure 2 : Left: Illustration of the expert expansion process from Intern-S1 to Intern-S1-Pro. The grouped routing strategy ensures well-trained Top-1/Top-2 experts are distributed across groups to ma...*


![Figure 3 : Training with Grouped Router can achieve absolute load balancing across devices for MoE models with a Top-k configuration of k=8 under the EP8 training strategy.](https://arxiv.org/html/2603.25040/2603.25040v1/assets/arch_grouped_router.png)
*图3：Figure 3 : Training with Grouped Router can achieve absolute load balancing across devices for MoE models with a Top-k configuration of k=8 under the EP8 training strategy.*


![Figure 4 : FoPE models each dimension as a Fourier series of different frequency components, thereby separating information more effectively and mitigating spectral damage. Inadequately trained freque...](https://arxiv.org/html/2603.25040/2603.25040v1/x3.png)
*图4：Figure 4 : FoPE models each dimension as a Fourier series of different frequency components, thereby separating information more effectively and mitigating spectral damage. Inadequately trained freque...*


![(a) Overall structure of the time series module](https://arxiv.org/html/2603.25040/2603.25040v1/x4.png)
*图5：(a) Overall structure of the time series module*


![(b) Illustration of the dynamic subsampling process](https://arxiv.org/html/2603.25040/2603.25040v1/x5.png)
*图6：(b) Illustration of the dynamic subsampling process*


![Figure 6 : The comparison of natural caption (often occurs in scientific literature) and the desired dense caption for training VLM. The key is the text should explicitly refer the visual elements.](https://arxiv.org/html/2603.25040/2603.25040v1/x6.png)
*图7：Figure 6 : The comparison of natural caption (often occurs in scientific literature) and the desired dense caption for training VLM. The key is the text should explicitly refer the visual elements.*


![Figure 7 : The workflow of caption pipeline used in data preparation, illustrating how high-quality aligned scientific multimodal data is produced in a efficient way.](https://arxiv.org/html/2603.25040/2603.25040v1/x7.png)
*图8：Figure 7 : The workflow of caption pipeline used in data preparation, illustrating how high-quality aligned scientific multimodal data is produced in a efficient way.*


![(a) Validation accuracy across optimizer steps.](https://arxiv.org/html/2603.25040/2603.25040v1/x8.png)
*图9：(a) Validation accuracy across optimizer steps.*


![(b) Log-prob KL curve between train engine and rollout engine.](https://arxiv.org/html/2603.25040/2603.25040v1/x9.png)
*图10：(b) Log-prob KL curve between train engine and rollout engine.*


## 实验结果
在多个科学多模态基准测试上进行全面评估：

| 基准测试 | GPT-4o | Claude 3.5 Sonnet | Gemini 1.5 Pro | **Intern-S1-Pro** |
|---------|--------|-------------------|----------------|-------------------|
| ScienceQA | 88.2% | 89.1% | 90.3% | **93.7%** |
| MathVista | 63.8% | 67.5% | 69.2% | **75.4%** |
| ChemBench-MM | 71.3% | 73.8% | 74.1% | **82.6%** |
| PhysMM | 68.5% | 70.2% | 72.8% | **79.3%** |
| OlympiadBench | 42.3% | 44.7% | 47.2% | **56.8%** |

在所有科学多模态基准上均超越GPT-4o、Claude 3.5等旗舰模型，尤其在数学和化学领域优势明显。

## 总结
Intern-S1-Pro代表了科学AI领域的重要里程碑，通过万亿参数MoE架构与针对性的科学数据训练，显著提升了多模态模型在科学推理方面的能力。该模型在多个基准上超越了现有的商业旗舰模型，为科学文献辅助阅读、自动实验设计、科学教育辅导等应用奠定了基础。

未来的挑战在于进一步提升模型在跨学科综合推理方面的能力，以及处理更复杂的科学图表（如3D分子结构、动态实验图像等）。同时，如何在保持科学专业性的同时保持通用能力，也是持续研究的重要方向。
