---
layout: post
title: "SelfCI：互补自蒸馏实现 LLM 的上下文完整性对齐"
date: 2026-05-22
categories: [论文解读, 大语言模型]
tags: [隐私保护, 上下文完整性, LLM对齐, 自蒸馏, KAIST]
---

> 📄 **论文**：It Takes Two: Complementary Self-Distillation for Contextual Integrity in LLMs
> 🔗 **arXiv**：[2605.20258](https://arxiv.org/abs/2605.20258)
> 🏢 **机构**：KAIST AI

## 一句话总结

SelfCI 通过让 LLM 从自身生成的两种互补反馈中进行自蒸馏，实现了 LLM 对"上下文完整性"（Contextual Integrity，CI）原则的对齐，使模型能够判断信息分享是否符合情境规范。

## 背景与问题

上下文完整性（Contextual Integrity，CI）是信息隐私的核心原则：信息流动的恰当性不仅取决于信息本身，还取决于信息在何种情境下、以何种方式、在哪些主体之间流动。例如，将患者的医疗信息透露给其主治医生是恰当的，但透露给无关第三方则违反了 CI。

现有 LLM 在 CI 对齐方面存在明显不足：模型往往要么过于保守（拒绝回答合理请求），要么过于宽松（泄露不应透露的隐私信息）。如何让模型准确理解信息分享的情境规范是一项挑战。

![CI理想状态](https://arxiv.org/html/2605.20258v1/x1.png)
*图1：Figure 1: Conceptual illustration of the ideal CI state in Def.˜2.1. A CI-aligned assistant should remain sensitive to task-relevant allowed information while invariant to disallowed information.*

## 核心方法

![SelfCI框架](https://arxiv.org/html/2605.20258v1/x2.png)
*图2：Figure 2: SelfCI uses self-generated feedback to instantiate two teacher distributions from its own parameters, πallow\pi_{\textbf{{\color[rgb]{0.26953125,0.5,0.5234375}\definecolor[named]{pgfstrokecolor}{rgb}{0.26953125,0.5,0.5234375}allow}}} promot*

### 上下文完整性的形式化定义

研究团队首先给出了 CI 的形式化定义：
- **信息发送方**（Sender）、**接收方**（Recipient）、**主体**（Subject）
- **传输原则**（Transmission Principle）：定义信息流动的规范
- **违规检测**：判断特定信息流是否符合情境规范

### 互补自蒸馏框架

SelfCI 的核心创新是生成两种互补的教师分布：
1. **允许教师**（Permissive Teacher）：倾向于信息分享的角度，生成"应该分享"的反馈
2. **保守教师**（Conservative Teacher）：倾向于信息保护的角度，生成"不应分享"的反馈

通过从这两种互补视角进行自蒸馏，模型学会在极端之间找到符合 CI 原则的平衡点。

![实验结果表格](https://arxiv.org/html/2605.20258v1/x3.png)
*表1：Table 1: Results on instruction-tuned and reasoning models. We evaluate each method on the CI-RL test set (in-domain) and the PrivacyLens benchmark (out-of-domain). All metrics except Helpful are reported as percentages. Best results are bolded; seco*

![方法对比](https://arxiv.org/html/2605.20258v1/x4.png)
*图3：Figure 4: Violation rate on CIMemories under progressively accumulating tasks, measured with Qwen3-4B-Instruct.*

![消融研究](https://arxiv.org/html/2605.20258v1/x5.png)
*图4：Figure 5: Analysis of the ideal CI surrogate in Eq.˜1 using Qwen3-4B-Instruct. (Left) Utility scores of target distributions on the CI-RL test set. (Right) Per-epoch Utility and Integrity scores trained with Eq.˜1 or Eq.˜5.*

![案例分析](https://arxiv.org/html/2605.20258v1/x6.png)
*图5：Figure 6: (Left) Integrity-Utility balance on the CI-RL test set for Qwen3-4B-Instruct trained with different λ\lambda values in Eq.˜5. (Middle) Per-epoch Complete score of feedback-conditioned teachers on the CI-RL training set. (Right) Complete sco*

![分布分析](https://arxiv.org/html/2605.20258v1/x7.png)
*图6：Figure 7: (Left) Integrity and (Middle) Utility across training epochs for the utility-oriented teacher πallow\pi_{\textbf{{\color[rgb]{0.26953125,0.5,0.5234375}\definecolor[named]{pgfstrokecolor}{rgb}{0.26953125,0.5,0.5234375}allow}}}, the privacy-o*

## 实验结果

在 CI-RL 测试集上，SelfCI 与多种对比方法的性能比较：

| 方法 | CI 对齐分数 | 过度保守率 | 过度宽松率 |
|------|------------|----------|----------|
| 基础 LLM | 低 | 高/低（不稳定） | - |
| DPO/RLHF | 中 | - | - |
| **SelfCI** | **最高** | **平衡** | **平衡** |

SelfCI 在指令微调模型和推理模型上均取得了最佳效果，证明了互补蒸馏策略的广泛适用性。

## 总结

SelfCI 为 LLM 隐私对齐提供了一个新颖的自监督框架，无需人工标注偏好数据，利用模型自身生成互补信号进行对齐。这种"It Takes Two"的哲学——同时从两种极端视角获取信号并在中间取得平衡——对其他需要在多种价值观之间权衡的 LLM 对齐问题同样具有启发意义。

局限性：CI 的形式化定义可能无法覆盖所有文化和法律背景下的隐私规范；自生成的教师信号可能存在系统性偏差。
