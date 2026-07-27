---
layout: post
title: "DataPrep-Bench：评测LLM作为训练数据准备器的基准"
date: 2026-07-28
categories: [论文解读, 数据工程]
tags: [LLM, 数据准备, 基准测试, 数据质量评估, 微调]
---

> 📄 **论文**：DataPrep-Bench: Benchmarking LLMs as Training Data Preparators
> 🔗 **arXiv**：[2607.20465](https://arxiv.org/abs/2607.20465)
> 🏢 **机构**：北京大学（Peking University）

## 一句话总结

DataPrep-Bench 是首个统一评测 LLM 在训练数据准备（包括数据构建和数据质量评估）两大能力的基准，采用下游训练效果作为统一的评分标准。

## 背景与问题

大语言模型（LLM）的能力从根本上取决于训练数据的质量、多样性和规模。随着自然语料越来越难以满足需求，研究社区开始大量使用 LLM、Agent 和数据驱动工作流来自动化生成训练数据——这一范式被称为"LLM 驱动的数据准备"。

然而，目前对不同数据准备方法的比较大多是零散的：不同工作使用不同的原始数据源、不同的基础模型和不同的下游评测基准，导致很难判断哪种方法真正能生成高质量的训练数据，以及哪种质量评分方法能可靠地预测下游性能。

为此，论文提出 **DataPrep-Bench**，将 LLM 驱动的数据准备分解为两大能力：
1. **数据构建（Data Construction）**：将原始非可训练资源（领域书籍、技术手册、网页转储）转化为监督微调（SFT）数据；
2. **数据质量评估（Data Quality Evaluation）**：在实际训练前预测候选数据集对下游模型的训练价值。

## 核心方法

DataPrep-Bench 采用统一的下游基准分数作为核心评价标准，横跨六大领域（数学、科学、法律、金融、医疗、代码）和多个基础模型。

![DataPrep-Bench整体框架](https://arxiv.org/html/2607.20465v1/x4.png)
*图1：DataPrep-Bench 整体框架，统一评测数据构建与数据质量评估两大能力*

### 数据构建 Track

所有方法从**相同的原始领域资料**（书籍、技术文档等）出发，自动生成 QA 形式的 SFT 数据，再与 Dolly-15k 混合微调基础模型，最终用领域下游基准分数衡量构建质量：

$$\text{Score}_{\text{con}}(M, D_k) = \text{Perf}(f(\mathcal{X}_k^{(M)}), \mathcal{T}_k)$$

论文同时发布了 **Data-Construction-Skill**，一个技能引导的 Agent，在 Llama-3.1-8B Finance 上比 Dolly-only 基线提升接近 **20 个百分点**。

![数据构建Prompt设计](https://arxiv.org/html/2607.20465v1/x5.png)
*图2：指导 LLM 进行数据构建的 Prompt 设计*

![Agent数据构建Prompt](https://arxiv.org/html/2607.20465v1/x6.png)
*图3：Agent 驱动的数据构建 Prompt 设计*

### 数据质量评估 Track

各种评分函数在共享候选数据集池上打分，用 **Pearson 相关系数**衡量与下游性能的相关性：

$$\text{Score}_{\text{eval}}(S, D_k) = \text{Pearson}(S(\mathcal{P}_k), \text{Perf}(f(\cdot), \mathcal{T}_k))$$

论文还发布了 **DAS（Distributional Alignment Score）**，一种基于 MMD（最大均值差异）度量候选数据集与领域代理分布对齐程度的评估器。

## 实验结果

### 数据构建实验（Qwen2.5-7B）

| 数据生成方法 | Law Avg | Medical Avg | Science Avg |
|---|---|---|---|
| Dolly-15k only† | 74.5 | 36.3 | 27.9 |
| DataFlow | 77.2 | 34.3 | 20.1 |
| DataFlow-Skill | 74.7 | 33.9 | 23.4 |
| Claude Opus 4.6 | 75.6 | 29.6 | 23.1 |
| Gemini 3.0 Pro | 74.7 | 29.1 | 23.8 |
| GPT-5.2 | 75.0 | 33.3 | 22.4 |
| Agent（Qwen3.5-Plus） | 75.6 | 33.8 | 26.0 |

**关键发现**：增加更多合成领域数据并不一定有益，大多数生成器在医学和科学领域甚至会降低性能，说明"更多合成数据"并非万能解药，下游效果是必要的评估维度。

### 数据质量评估实验

**DAS** 在六个领域中的四个取得了最强的跨模型相关性，并且是**唯一**在数学、科学和医学三个领域同时达到 $r > 0.70$ 的评估指标，超越了现有的基于质量、多样性和启发式的评估器。

## 总结

DataPrep-Bench 填补了 LLM 驱动数据准备领域缺乏统一评测标准的空白，提供了基于下游训练效果的公平比较框架。其核心贡献包括：（1）提出统一的双轨道评测体系；（2）发布具有竞争力的 Data-Construction-Skill Agent；（3）提出在多领域均表现优秀的 DAS 质量评估指标。

局限性方面，当前基准仅涵盖六个领域，且数据构建 Track 主要针对 QA 格式的 SFT 数据，未来可扩展到更多任务格式和更广泛的领域。
