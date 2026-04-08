---
layout: post
title: "真实场景下的智能体技能效果评测：LLM技能使用基准测试"
date: 2026-04-09
categories: [论文解读, 智能体评测]
tags: [LLM Agent, Skill Retrieval, Benchmark, Agentic AI, RAG, Tool Use]
---

> 📄 **论文**：How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings
> 🔗 **arXiv**：[2604.04323](https://arxiv.org/abs/2604.04323)
> 🏢 **机构**：UC Santa Barbara, MIT CSAIL, MIT-IBM Watson AI Lab

## 一句话总结

在真实场景下，LLM智能体从大规模技能库中检索并使用技能的效果远不如理想化基准测试中展示的那么乐观，而基于查询的针对性技能精炼（Query-Specific Refinement）是目前最有效的改进手段。

## 背景与问题

大型语言模型（LLM）驱动的智能体系统越来越依赖"技能"（Skills）这一概念——即可重用的、领域特定的知识制品，帮助智能体解决复杂任务。已有工作（如SkillsBench）评测了技能对智能体的提升效果，但这些评测存在明显的理想化假设：技能是人工精心设计的、完全针对任务定制的，并且直接提供给模型，完全绕过了现实中技能发现与检索的挑战。

然而在真实部署场景中，智能体面临三重挑战：第一，**技能选择**——需要从海量技能库中识别出有用的技能；第二，**技能检索**——需要独立搜索大规模仓库；第三，**技能适配**——现有技能通常是通用型的，而非针对当前任务量身定制的。现有基准测试完全回避了这三个问题，导致其评测结论对实际系统的指导意义有限。

本文核心问题是：**当智能体必须从包含34,000多个真实世界技能的噪声库中自主检索相关技能时，技能是否仍然有帮助？** 作者系统性地构建了一个渐进式评测框架，从理想化条件逐步过渡到真实场景，量化每个环节对技能效益的影响，并探索技能精炼策略以弥补性能损失。

![SkillsBench任务示例](https://arxiv.org/html/2604.04323/x1.png)
*图1：SkillsBench中的flooding任务示例。经人工策划的技能几乎直接提供了完整解题步骤，这与真实场景中智能体需要自主搜索的情况相差甚远。*

## 核心方法

### 技能数据库构建

研究团队从 skillhub.club 和 skills.sh 收集了 **34,198 个真实世界技能**，涵盖 Web 开发、数据工程、DevOps、科学计算等多个领域。所有技能均通过 MIT/Apache 2.0 许可证过滤，并按文件内容去重，确保数据质量。

### 混合检索系统

技能搜索引擎采用双重表示策略对技能进行索引：
- **元数据表示**：技能名称 + 描述
- **全文内容表示**：完整的 SKILL.md 内容

检索方法综合运用两种技术：
- **稠密嵌入**：使用 Qwen3-Embedding-4B 模型
- **稀疏检索**：使用 SQLite FTS5 实现的 BM25

最终通过**互惠排名融合（Reciprocal Rank Fusion, RRF）**进行混合融合：

$$\text{score} = \sum \frac{w_s}{k + r_s}, \quad k = 60$$

下表对比了多种检索变体：

| 检索方法 | 描述 |
|---------|------|
| Direct (semantic) | 单次查询 → 返回 top-k 结果 |
| Agentic (keyword) | 智能体迭代使用 BM25 关键词检索 |
| Agentic (semantic) | 智能体迭代使用稠密语义嵌入 |
| Agentic hybrid w/o content | 元数据上的关键词 + 语义混合 |
| **Agentic hybrid w/ content** | **在完整 SKILL.md 内容上混合检索（最优）** |

### 渐进式评测框架

这是本文方法论的核心创新。研究设计了六个渐进式评测设置，从最理想化条件逐步过渡到真实场景：

| 设置 | 描述 |
|------|------|
| Curated + Forced Load | 精选技能直接注入，强制模型加载（最理想化） |
| Curated | 精选技能可用，但由智能体自主决定是否加载 |
| Curated + Distractors | 精选技能 + 大量无关干扰技能 |
| Retrieved (w/ curated) | 从真实库检索，精选技能包含在检索池中 |
| Retrieved (w/o curated) | 纯真实检索，精选技能不在检索池中（最真实场景） |
| No Skills | 无技能基线 |

![渐进式评测中Pass Rate的衰减趋势](https://arxiv.org/html/2604.04323/x2.png)
*图2：从理想化设置到真实场景，Pass Rate 呈现出一致性的渐进衰减趋势，揭示了真实部署中技能效益的脆弱性。*

### 实验模型与运行环境

评测使用了三个代表性模型，均在隔离的 Docker 容器中运行，每个任务重复3次：

| 模型 | 运行框架 |
|------|---------|
| Claude Opus 4.6 | Claude Code v2.1.19 |
| Kimi K2.5 | Terminus-2 |
| Qwen3.5-397B-A17B-FP8 | Qwen-Code v0.12.3 |

### 技能精炼策略

研究探索了两种技能精炼方案：

**查询无关精炼（Query-Agnostic Refinement）**：离线模式，无需任务知识。使用 Anthropic 的 skill-creator meta-skill 生成合成测试查询，通过 A/B 比较迭代改进每个技能，独立于具体任务进行。

**查询特定精炼（Query-Specific Refinement）**：智能体先读取任务，尝试使用检索到的技能解题，再反思技能的有用性，最后将多个来源的相关部分综合成精炼后的技能制品。

![查询特定精炼示例](https://arxiv.org/html/2604.04323/x4.png)
*图4：在tensor parallelism任务上的查询特定精炼示例。智能体将两个部分相关的技能来源综合成一个针对当前任务更有用的精炼技能。*

## 实验结果

### 检索性能（Recall@k，Claude Opus 4.6）

| 检索方法 | Recall@3 | Recall@5 | Recall@10 |
|---------|----------|----------|-----------|
| Direct (semantic) | 38.1% | 47.0% | 52.3% |
| Agentic (keyword) | 24.1% | 26.6% | 27.5% |
| Agentic (semantic) | 56.8% | 63.1% | 66.5% |
| Agentic hybrid w/o content | 57.7% | 63.5% | 66.7% |
| **Agentic hybrid w/ content** | **57.3%** | **65.5%** | **68.3%** |

智能体搜索在 Recall@3 上比直接检索高出 **+18.7 个百分点**；纯关键词检索在所有变体中表现最差。

### 渐进式评测结果（Pass Rate）

![各设置下的Pass Rate与技能加载率对比](https://arxiv.org/html/2604.04323/x3.png)
*图3：三个模型在所有评测设置下的 Pass Rate 和技能加载率（Skill Loading Rate）完整对比图。*

| 评测设置 | Claude Opus 4.6 | Kimi K2.5 | Qwen3.5 |
|---------|----------------|-----------|---------|
| Curated + Forced Load | 55.4% | 38.5% | — |
| Curated | 51.2% | 38.9% | 31.6% |
| Curated + Distractors | 43.5% | — | 33.7% |
| Retrieved (w/ curated) | 40.1% | 33.5% | 26.7% |
| Retrieved (w/o curated) | 38.4% | 19.8% | 19.7% |
| **No Skills（基线）** | **35.4%** | **21.8%** | **20.5%** |

关键发现：**Kimi 和 Qwen 在没有精选技能的情况下，Pass Rate 甚至低于无技能基线**，说明能力较弱的模型可能被低质量的检索技能所误导，产生负面效果。

### 技能加载行为分析

智能体是否真正加载可用技能是一个重要问题。即使在精选技能直接可用的情况下，Claude 只在 **49%** 的轨迹中加载了所有技能；加入干扰技能后，这一比例降至 **31%**。这表明智能体框架（harness）的设计对技能利用率有显著影响。

### 技能精炼效果（Claude Opus 4.6，SkillsBench）

| 设置 | Pass Rate | 技能加载率 |
|------|-----------|----------|
| Retrieved (w/ curated) | 40.1% | 44.4% |
| + Query-Specific Refinement | **48.2%** | **72.2%** |
| + Query-Agnostic Refinement | 42.0% | 32.9% |
| Retrieved (w/o curated) | 38.4% | 16.3% |
| + Query-Specific Refinement | 37.9% | 61.1% |
| + Query-Agnostic Refinement | 37.4% | 12.3% |

查询特定精炼在有精选技能的情况下带来了 **+8.1 个百分点**的显著提升，而在无精选技能时几乎没有改善，表明精炼的本质是对现有质量的**放大器**，而非凭空生成缺失知识的工具。

### Terminal-Bench 2.0 结果（Claude Opus 4.6）

| 设置 | Pass Rate |
|------|-----------|
| No Skills | 57.7% |
| Retrieved | 61.4% |
| Retrieved + Query-Specific | **65.5%** |

在更具挑战性的 Terminal-Bench 2.0 上，查询特定精炼同样带来了明显提升（+7.8pp over retrieval）。

### 覆盖率评分（LLM 评判，1-5分制）

| 评测场景 | Claude | Kimi | Qwen |
|---------|--------|------|------|
| SkillsBench w/ curated skills | 4.01 | 3.83 | 3.85 |
| SkillsBench w/o curated skills | 3.49 | 3.31 | 3.39 |
| Terminal-Bench 2.0 | 4.02 | 3.96 | 4.08 |

覆盖率 ≥3.83 的设置对查询特定精炼响应良好；覆盖率 ≤3.49 时则改善甚微，与 Pass Rate 结论完全吻合。

## 总结

本文揭示了一个重要的系统性问题：现有智能体技能评测基准过度理想化，其乐观结论在真实场景中无法复现。通过构建包含34,198个真实技能的大规模库，并设计渐进式评测框架，研究发现技能的收益随评测条件的真实化而持续衰减——在最真实场景中，较弱模型甚至因无关技能而出现性能退化。

在改进策略方面，**查询特定精炼**表现出最强的实际价值，在精选技能可用时能带来约8个百分点的稳定提升，并将技能加载率从44%大幅提升至72%。其原理是在执行时将多个部分相关的技能合成为针对当前任务的专用制品，本质上弥合了通用技能与特定任务需求之间的鸿沟。相比之下，查询无关的离线精炼效果不稳定，说明离线处理难以预判技能的实际使用场景。

本研究的局限性在于：评测规模受计算资源限制，部分模型组合的结果不完整；技能质量评估依赖LLM自动评判，可能存在偏差；此外，技能库的覆盖范围主要集中在软件工程领域，结论向其他领域的泛化能力仍需验证。未来工作方向包括更高效的在线技能精炼机制、智能体主动学习与技能积累，以及跨领域技能迁移能力的提升。
