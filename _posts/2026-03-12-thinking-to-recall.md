---
layout: post
title: "【论文精读】思考以回忆：推理如何解锁大语言模型的参数知识"
date: 2026-03-12
categories: [AI, LLM, Reasoning]
tags: [LLM推理, 事实召回, 幻觉, Chain-of-Thought, arXiv]
---

> 📄 **论文精读 · arXiv 2603.09906**
>
> **Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs**
>
> Gekhman et al. · Technion & Tel Aviv University · 2026年3月
>
> 标签：LLM推理 · 事实召回 · 幻觉

---

## ⚡ 核心发现（TL;DR）

- 开启推理（Chain-of-Thought）能**大幅扩展LLM的参数知识回忆边界**，即使对简单的单跳事实问题也有显著提升
- 推理提升知识召回的背后有**两个关键机制**：①计算缓冲效应（与内容无关的额外计算）；②事实启动效应（生成相关中间事实作为语义桥梁）
- 推理过程中产生的**中间幻觉事实会显著提高最终答案的幻觉概率**，这是一个不可忽视的风险
- 基于上述洞察，可通过**优先选择无幻觉推理轨迹**来直接提升模型准确率

---

## ABSTRACT · 摘要

尽管大语言模型中的推理在数学、代码生成和多跳事实问题上发挥着天然作用，但其对**简单的单跳事实问题**的影响至今仍不清晰。这类问题不需要逐步的逻辑分解，因此推理的作用显得非常反直觉。然而，我们发现开启推理能**大幅扩展模型参数知识召回的能力边界**，解锁了那些在不推理时几乎无法触达的正确答案。

*While reasoning in LLMs plays a natural role in math, code generation, and multi-hop factual questions, its effect on simple, single-hop factual questions remains unclear. Such questions do not require step-by-step logical decomposition, making the utility of reasoning highly counterintuitive. Nevertheless, we find that enabling reasoning substantially expands the capability boundary of the model's parametric knowledge recall, unlocking correct answers that are otherwise effectively unreachable.*

为什么当没有复杂推理步骤需要执行时，推理仍然有助于参数知识回忆？为了回答这个问题，我们设计了一系列假设驱动的控制实验，识别出两个关键驱动机制：（1）**计算缓冲效应**（Computational Buffer Effect）——模型利用生成的推理token进行独立于其语义内容的潜在计算；（2）**事实启动效应**（Factual Priming）——生成与主题相关的事实作为语义桥梁，促进正确答案的检索。

*We identify two key driving mechanisms: (1) a computational buffer effect, where the model uses the generated reasoning tokens to perform latent computation independent of their semantic content; and (2) factual priming, where generating topically related facts acts as a semantic bridge that facilitates correct answer retrieval.*

> ⚠️ **关键风险**
>
> 事实启动机制带来固有风险：我们证明，推理过程中产生的**中间事实幻觉**会增加最终答案出现幻觉的可能性。这意味着推理是一把双刃剑——它能激活潜在知识，却也可能让推理阶段的错误污染最终答案。

---

## SECTION 2 · 实验设置

### 研究模型

研究使用了**混合模型**——这类模型的推理可以通过控制token或系统指令在开/关之间切换：**ON（开启）**模式会在最终回复前生成推理轨迹；**OFF（关闭）**模式则直接回复，不生成推理轨迹。使用混合模型的好处是可以精确隔离推理的效果，同时控制模型的参数知识不变。实验使用了三个模型：**Gemini-2.5-Flash**、**Gemini-2.5-Pro** 和 **Qwen3-32B**。

*We use hybrid models where reasoning can be toggled ON/OFF. Using hybrid models, we isolate the effect of reasoning while controlling for the model's parametric knowledge. We use Gemini-2.5-Flash, Gemini-2.5-Pro, and Qwen3-32B.*

### 评测数据集

使用了两个具有挑战性的**闭卷问答数据集**：
- **SimpleQA-Verified**：包含1,000个经过过滤和纠正的真实问题，其中90%为单跳问题
- **EntityQuestions**：基于问题模板，含有大答案空间的单跳问题，共1,000条

*We use SimpleQA and EntityQuestions, two challenging closed-book QA datasets. For SimpleQA, we utilize SimpleQA-Verified, a subset of 1,000 examples filtered and corrected for increased reliability.*

> 📐 **核心指标：pass@k**
>
> **pass@k** 估计在k次采样中至少有一次答对的概率，用于探索模型的**能力边界**而非单次准确率。pass@1 = 标准准确率，pass@100 = 模型"潜力上限"。本研究最多采样 N=100 次。

---

## SECTION 3 · 推理扩展模型的参数知识边界

*Reasoning Expands The Model's Parametric Knowledge Boundary*

本节聚焦于能力边界：**推理是否能让模型发现在不推理时几乎不可能找到的正确答案？**还是说，推理只是通过更好的采样效率来提升准确率——即提升了那些本来就有一定概率被正确回答的问题的概率？

*Does reasoning enable the discovery of correct answers that are effectively unreachable without it, or does it mainly improve accuracy via better sampling efficiency?*

![图1：两个闭卷问答基准和三个LLM上的 pass@k 曲线，对比推理开/关两种模式](https://arxiv.org/html/2603.09906v1/x1.png)

**图 1 · FIGURE 1**
两个闭卷问答基准和三个LLM上的 pass@k 曲线，对比推理开/关两种模式。推理开启（ON）始终优于推理关闭（OFF），且随着 k 增大，差距往往更加明显，说明推理扩展了模型的知识边界。

*Pass@k curves across two closed-book QA benchmarks and three LLMs, comparing reasoning OFF vs ON.*

> 🔑 **关键发现：推理解锁潜在知识**
>
> 推理在所有模型和数据集上都**持续提升** pass@k 值。虽然 pass@1（标准准确率）有明显改善，但在更大的 k 值时提升往往更为显著——例如，**Qwen3-32B 在 SimpleQA-Verified 上 pass@k 几乎翻倍**。这种持续扩大的差距表明：推理扩展了模型参数知识的召回边界，帮助模型更好地"挖掘"其内部知识。

### 推理效果度量：加权平均改进 Ω

为了用单一指标衡量推理在各设置下的整体效果，研究定义了综合推理效果度量 **Ω**，它满足两个核心要求：①相对于推理关闭进行度量；②对更大的 k 值赋予更高权重（以突出能力边界的提升）。

*We define a unified reasoning effectiveness metric Ω that accounts for the entire range of k values, assigning higher importance to larger k values to capture capability boundary gains.*

```
Ω(N) = Σ k·[passON@k − passOFF@k] / passOFF@k   （k 从 1 到 N 的加权平均）
```

![图2：所有模型和数据集上的推理效果综合指标 Ω](https://arxiv.org/html/2603.09906v1/figs/resoning_effectiveness.png)

**图 2 · FIGURE 2**
所有模型和数据集上的推理效果综合指标 Ω。模型从左到右按 pass@1 从高到低排列。Ω 是一个加权平均改进值，对更大的 k 赋予更高权重，以突出对能力边界的贡献。

*Ω (reasoning effectiveness) in all models and datasets. Models organized from the most to the least effective in terms of pass@1.*

---

## SECTION 4 · 问题复杂度不能预测推理效果

*Question Complexity is a Poor Predictor of Reasoning Effectiveness*

一个自然的假设是：推理有助于分解复杂的多跳问题。然而，本研究的数据集主要由**简单的单跳事实问题**组成（SimpleQA-Verified 中 90% 为单跳问题）。那么，推理对复杂问题的提升是否更大？

*A natural hypothesis is that reasoning aids in the decomposition of complex, multi-hop questions. However, our datasets consist predominantly of simple, direct (single-hop) factual questions.*

![图3：SimpleQA-Verified 中不同问题类型上的推理效果](https://arxiv.org/html/2603.09906v1/figs/question_difficulty_4.png)

**图 3 · FIGURE 3**
SimpleQA-Verified 中不同问题类型上的推理效果（含95%置信区间）。"复杂"问题（至少满足"需要推理"或"多步"标签之一）与"简单"问题的推理增益没有显著差异，置信区间大量重叠。

*Reasoning effectiveness on different question types in SimpleQA-Verified, with 95% confidence intervals.*

> 💡 **反直觉结论：复杂度无关**
>
> 出乎意料，我们**没有发现推理对复杂问题子集的边际增益更高的证据**——复杂子集和简单子集的95%置信区间大量重叠。这强化了核心论点：在本研究中，推理的增益**主要不是来自任务分解**，而是来自促进参数知识的召回。这自然引出了一个问题：究竟是什么机制让推理能够改善知识召回？

---

## SECTION 5 · 推理如何改善参数知识召回？

*How Reasoning Improves Parametric Recall?*

确认了推理的增益主要来自促进参数知识召回（而非任务分解）后，研究深入分析背后的机制。采用**假设驱动的方法**：先提出候选解释，再设计控制实验验证。

*We adopt a hypothesis-driven approach: formulate candidate explanations and design controlled experiments to test them.*

**两大机制：**

| 机制 | 描述 |
|------|------|
| ⚙️ **机制一：计算缓冲效应** | 额外的推理token允许模型在生成最终答案前进行更多的**隐式计算**，与这些token的语义内容无关 |
| 🔗 **机制二：事实启动效应** | 推理过程中生成与问题相关的事实，构建**语义桥梁**，帮助模型检索到正确答案 |

### 5.1 推理Token作为计算缓冲器

**假设：**生成额外token允许模型执行更多潜在运算，绕过单次前向传播的深度限制。为了隔离"纯计算"效果，研究引入了 **ON Dummy 变体**：将模型原始推理轨迹替换为语义无意义的哑字符串 `"Let me think."` （重复以匹配原始长度），然后基于此重新生成最终答案。

*We introduce the ON Dummy variant, replacing the reasoning trace with the semantically meaningless dummy sequence "Let me think." repeated to match the original trace's length, then regenerate the final answer conditioned on it.*

![图4：Gemini-2.5-Flash 上的计算缓冲效应实验结果](https://arxiv.org/html/2603.09906v1/figs/dummy_thought_v2.png)

**图 4 · FIGURE 4**
Gemini-2.5-Flash 上的计算缓冲效应实验结果。
- **ON**：正常推理开启
- **ON Single Dummy**：推理轨迹替换为一次哑字符串（短）
- **ON Dummy**：推理轨迹替换为重复哑字符串（与原始等长）

结果表明，即使是语义无意义的推理token也能显著提升性能，验证了计算缓冲假设。

*Computation buffer effect on Gemini-2.5-Flash. ON Single Dummy overrides the thinking trace with a short dummy sequence. ON Dummy repeats the short dummy sequence to match the length of the original trace.*

> 📊 **实验结果**
>
> 在哑轨迹上条件化后，pass@k **显著高于** OFF 模式：SimpleQA-Verified 准确率从 `0.206 → 0.262`，EntityQuestions 从 `0.457 → 0.554`。同时，ON Dummy 与 ON Single Dummy 之间的持续性差距（两者语义内容相同，唯一区别是计算长度）进一步隔离了额外计算的效果，提供了强有力的证据。

> 💡 **计算长度并非越长越好**
>
> 研究还发现，更多计算并不总是有益的。哑轨迹长度增加到 **2048 tokens** 时效果最佳，超过 4096 tokens 后性能开始持续下降——呈现**非单调的扩展规律**。这说明计算缓冲机制存在上限，无法完全解释推理带来的全部增益，从而引出对语义内容的分析。

![图5：推理效果 Ω 随哑推理轨迹输入token长度的变化曲线](https://arxiv.org/html/2603.09906v1/figs/scaling_omega.png)

**图 5 · FIGURE 5**
以哑推理轨迹为条件时，推理效果 Ω 随输入token长度的变化曲线（§5.1）。ON Dummy X 将推理轨迹替换为重复的哑字符串，总输入长度约为 X tokens。可以看出，随着哑轨迹长度增大，Ω 先升后降，呈现**非单调的扩展规律**——约在 2048 tokens 处达到峰值，随后持续下降，说明纯计算缓冲存在天花板效应。

*Reasoning effectiveness as a function of the input length in tokens when conditioning on dummy reasoning trace.*

![图17（附录）：增大计算量的完整 pass@k 曲线对比](https://arxiv.org/html/2603.09906v1/figs/scaling_curves_v3.png)

**图 17（附录）· FIGURE 17**
增大计算量的完整 pass@k 曲线对比（附录补充）。dummy_X 将推理轨迹替换为重复哑字符串使总输入长度约为 X tokens。完整曲线更直观地展示了"计算越多不总越好"这一非单调规律在各个 k 值下的表现。

*The effect of increasing compute. dummy_X overrides the thinking trace with a short dummy sequence which is repeated such that the total input length will be X.*

### 5.2 事实启动：生成式自我检索

由于简单问题的推理轨迹几乎不包含逐步推导，轨迹内容通常是与问题主题相关的**事实性陈述**。这激发了"事实启动假设"：在回答前先生成相关事实背景，能促进正确答案的检索。为了验证，研究设计了关键对比实验：从推理轨迹中提取简短事实列表，在推理关闭模式下将这些事实作为额外上下文输入给模型，看能否复现推理带来的 pass@k 增益。

*We demonstrate the existence of a factual priming mechanism: the model engages in generative self-retrieval, constructing contextual bridges to the answer by recalling related facts. Extracting a short list of facts from the reasoning trace and rerunning the model with reasoning disabled conditioned on this list as additional context, recovers most of the pass@k gains of reasoning.*

在人类认知中，处理某个概念会在语义网络中"激活"相关邻节点，降低检索阈值（Collins & Loftus, 1975）。研究假设R-LLM存在类似机制：模型进行**生成式自我检索**（Generative Self-Retrieval），通过回忆相关事实有效构建通往答案的"语境桥梁"，称之为**事实启动效应**。

*In human cognition, processing a concept spreads "activation" through a semantic network, priming related neighbors. We hypothesize that R-LLMs exhibit a similar mechanism where the model engages in generative self-retrieval, constructing a contextual bridge to the answer through recalling related facts.*

为验证此假设，研究设计了两个变体：
- **ON Facts**：用提取的事实列表替换推理轨迹，重新生成答案
- **OFF Facts**：关闭推理，将事实列表作为额外上下文输入

OFF Facts 作为主要基线，因为它避免了"模型在 ON 模式下的偏好偏差"等混淆因素。

![图6：Gemini-2.5-Flash 上的事实启动效应实验结果](https://arxiv.org/html/2603.09906v1/figs/OFF_Summary_Plot.png)

**图 6 · FIGURE 6**
Gemini-2.5-Flash 上的事实启动效应实验结果（§5.2）。以推理过程中召回的事实为条件：**OFF Facts**（关闭推理+事实作为上下文）、**ON Facts**（开启推理+事实替换轨迹）。OFF Facts 能够恢复推理ON模式大部分的 pass@k 增益，表明中间事实本身是知识检索的关键桥梁。

*Factual priming effect on Gemini-2.5-Flash (§5.2), conditioning on facts recalled during reasoning, with reasoning either OFF (OFF Facts) or ON (ON Facts).*

![图9：事实启动效应的案例研究](https://arxiv.org/html/2603.09906v1/figs/A4_OFF_Summary_diagram.png)

**图 9（案例）· FIGURE 9**
事实启动效应的案例研究。展示了一个具体例子：模型在推理轨迹中生成了相关的中间事实，这些事实充当语义桥梁，帮助模型在后续步骤中成功召回正确答案。

*Case study for the effectiveness of factual priming.*

> 🎯 **强有力证据：事实即桥梁**
>
> 将推理轨迹中提取的短事实列表作为额外上下文输入（同时关闭推理），能够**恢复大部分 pass@k 增益**。这提供了强有力的证据：推理过程中召回的中间事实本身对正确答案的检索是有用的——模型通过"生成式自我检索"构建了从问题到答案的语义桥梁。

### 5.3 幻觉的风险：双刃剑效应

事实启动机制依赖模型**自身生成的事实**——这些事实可能是幻觉。研究使用大规模审计流水线（每个问题采样100次，用启用搜索的 Gemini-2.5-Flash 独立验证每条推理轨迹中的每个事实），揭示了一个清晰的规律：**包含幻觉中间事实的推理轨迹，大幅提高了最终答案出现幻觉的概率。**

*Reasoning traces with hallucinated intermediate facts are substantially more likely to yield hallucinated final answers. We assess this risk using a large-scale auditing pipeline that verifies every intermediate fact in every sampled trajectory for each question.*

![图7：在问题内部对比干净推理轨迹与含幻觉推理轨迹的最终答案正确率](https://arxiv.org/html/2603.09906v1/figs/hallucinations_within_question_v2.png)

**图 7 · FIGURE 7**
在问题内部对比：x轴为**干净推理轨迹**（无幻觉中间事实）下的最终答案正确率，y轴为**含幻觉推理轨迹**下的最终答案正确率。每个点代表一个问题。
- 红色点（位于对角线上方）：幻觉轨迹反而得分更高（少数情况）
- 绿色点（位于对角线下方）：干净轨迹得分更高（大多数情况）

大多数点分布在对角线下方，清楚表明**含幻觉推理轨迹的正确率系统性低于干净轨迹**。

*Within-question comparison of correct final-answer rates in clean (x-axis) vs. hallucinated (y-axis) reasoning traces. Each question is one point; red examples lie above the no-effect diagonal, green below.*

> ⚠️ **幻觉传播效应**
>
> 生成式自我检索是一个**强大但脆弱的机制**：它能激活潜在知识，但也允许推理阶段的错误传播并影响最终答案。推理过程中产生的中间幻觉事实，会像"错误的种子"一样引导模型走向错误的最终答案。

### 5.4 实践启示：优先选择无幻觉轨迹

基于上述洞察，研究展示了一个直接的应用：在推理时，优先选择包含**无幻觉事实性陈述**的推理轨迹，可以显著提升模型的最终准确率。这为推理时的轨迹选择策略提供了有力依据，也为训练时的过程奖励设计提供了新方向。

*Our insights can be harnessed to directly improve model accuracy by prioritizing reasoning trajectories that contain hallucination-free factual statements.*

---

## SECTION 6 · 案例研究

*Case Studies*

### 6.1 案例：纯计算辅助召回

下图展示了一个典型案例：模型在**哑轨迹条件下**（推理轨迹被替换为无意义的 "Let me think." 重复字符串），仍然成功召回了在 OFF 模式下无法回答的正确答案。这直观说明了计算缓冲效应——即使没有任何语义内容，额外的计算空间本身就能帮助模型"思考得更深"。

*Case study for the effectiveness of the computational buffer effect. The model successfully recalls the correct answer conditioned on a dummy (semantically empty) reasoning trace.*

![图8：计算缓冲效应的案例研究](https://arxiv.org/html/2603.09906v1/figs/A4_ON_Dummy_diagram.png)

**图 8（案例研究）· FIGURE 8**
计算缓冲效应的案例研究。展示了模型在哑推理轨迹（仅含 "Let me think." 重复）条件下，仍能成功回答在推理关闭时无法回答的问题，验证了纯计算量本身对知识召回的促进作用。

*Case study for the effectiveness of the computational buffer effect.*

### 6.2 案例：事实启动成功与幻觉失败

事实启动的两种典型结果对比如下，直观揭示了推理的双刃剑本质：

| 案例类型 | 推理轨迹内容 | 最终答案 | 机制 |
|----------|-------------|----------|------|
| ✅ **纯计算辅助** | "Let me think. Let me think…"（无语义） | ✅ 正确 | 计算缓冲效应 |
| ✅ **事实启动成功** | 生成了准确的相关背景事实作为语义桥梁 | ✅ 正确 | 事实启动（准确） |
| ❌ **幻觉传播失败** | 生成了错误的中间事实（幻觉） | ❌ 错误 | 幻觉传播效应 |

---

## 📌 研究结论总结

1. 推理（Chain-of-Thought）能显著扩展LLM的参数知识召回边界，即使对不需要多步推理的简单事实问题也如此，且问题复杂度无法预测推理的收益大小
2. **计算缓冲效应**：推理token的长度本身（与内容无关）提供了额外的隐式计算空间，帮助模型"想得更深"，但存在非单调上限
3. **事实启动效应**：推理轨迹中生成的相关事实充当语义桥梁（"生成式自我检索"），是推理增益的主要语义来源，且这种效果可以从轨迹中显式提取和重现
4. **幻觉传播风险**：中间幻觉事实会显著提高最终答案的幻觉率，这是事实启动机制的内在脆弱性，需要在推理时加以防范
5. **实践应用**：在推理时优先选择包含无幻觉中间事实的轨迹，可以直接提升模型准确率，为推理时计算和训练过程奖励提供了新方向

---

## ANALYSIS · 编者深度评析

*Critical Analysis*

### 🏆 最大贡献

**① 首次系统性拆解推理增益的底层机制**

以往研究都知道"推理能提升准确率"，但从未精确回答"为什么简单事实问题也能受益"。本文通过设计精妙的控制变量实验（哑轨迹/事实提取/幻觉审计），将一个笼统的现象拆解为两个可独立验证的机制，这是方法论上的重大贡献。

**② 提出"生成式自我检索"概念，类比人类认知**

将语言模型的推理行为与人类认知科学中的"语义激活扩散理论"（Collins & Loftus, 1975）相连接，不只是工程发现，更是对LLM内部工作机制的理论阐释——模型在推理时本质上是在"自问自答地构建语义线索"。

**③ 幻觉传播的实证证据与量化**

利用100次采样 × 大规模逐条事实验证的审计流水线，首次提供了**中间幻觉 → 最终幻觉**这条因果链的定量证据，不只是定性观察。这为推理时计算（Best-of-N、过程奖励模型等）提供了严谨的方向性指导。

### ⚠️ 不足之处

| 局限 | 说明 |
|------|------|
| **局限一：模型范围有限** | 仅测试了 Gemini-2.5 系列和 Qwen3-32B，均为混合推理模型。对于 GPT-4o、Claude 3.5 等非混合模型，"推理OFF/ON"的切换无法直接复现，外推性存疑。 |
| **局限二：只研究了事实性QA** | 实验集中在 SimpleQA / EntityQuestions 这类"有唯一正确答案"的闭卷问答，对于开放性问题、创意写作、代码生成等场景，两种机制是否仍然成立尚未验证。 |
| **局限三：因果性待加强** | 幻觉传播的证据主要来自相关分析，文中也承认无法完全排除"问题本身太难导致两者都错"的共因解释，需要更严格的因果推断实验。 |
| **局限四：计算开销极大** | 每个问题采样100次 + 逐条事实搜索验证，整体审计流水线的计算成本极高，难以在生产环境中直接应用，工程化落地路径尚待简化。 |

### 💡 借鉴意义

**🎯 对 RAG / 检索增强系统的启示**

LLM 自身推理过程中的"生成式自我检索"与外部 RAG 在功能上有相似之处——两者都在为模型构建语义桥梁。这意味着对于那些模型参数中已包含答案的问题，单纯堆砌外部检索文档未必是最优方案，引导模型先进行内部推理式召回可能更高效。

**🔧 对过程奖励模型（PRM）训练的启示**

已有实证证明"含幻觉中间步骤 → 最终答案更差"，这为 PRM 的奖励信号设计提供了精准靶点：对推理轨迹中的**事实性中间步骤**进行细粒度打分，而非只看最终答案对错。

**⚡ 对推理时计算（Inference-Time Compute）策略的启示**

Best-of-N 采样时，筛选标准不应只看最终答案的置信度，而应引入"中间事实幻觉率"作为过滤维度——优先保留那些中间步骤事实准确的轨迹，可在不增加采样成本的前提下显著提升精度。

**📐 对 CoT 设计的启示**

既然"计算缓冲"和"事实启动"是两个可分离的机制，未来可探索专门为"事实召回"设计的 CoT 提示格式——引导模型先显式列举相关背景知识，再给出答案，而不是自由发散地推理。

### 📚 建议延伸阅读（5篇）

1. **必读·前置**：[LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations](https://arxiv.org/abs/2410.02707)
   — Gekhman et al., 2024 · arXiv 2410.02707
   — 本文直接前驱：证明模型编码了事实但无法总是生成它，是"参数知识边界"概念的基础工作。理解本文必读。

2. **强烈推荐**：[Let's Verify Step by Step（过程奖励模型 PRM800K）](https://arxiv.org/abs/2305.20050)
   — Lightman et al., OpenAI, 2023 · arXiv 2305.20050
   — 过程奖励模型的奠基工作，与本文"中间步骤幻觉影响最终答案"的发现直接呼应，是将本文洞察落地到训练的核心参考。

3. **推荐**：[Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)
   — Snell et al., UC Berkeley, 2024 · arXiv 2408.03314
   — 系统研究推理时计算的最优分配策略（Best-of-N、Beam Search、PRM引导），与本文"如何利用推理增益"的实践建议高度互补。

4. **推荐**：[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
   — Wei et al., Google Brain, 2022 · arXiv 2201.11903
   — CoT 的奠基论文。本文是对它的深层追问——CoT 究竟为什么有效？从"现象描述"走向"机制解释"的必要背景。

5. **延伸**：[Thinking Tokens for Language Modeling（Filler tokens / Pause tokens）](https://arxiv.org/abs/2404.07143)
   — Goyal et al., 2024 · arXiv 2404.07143
   — 提出在输入中插入"暂停token"让模型有更多计算时间的方法，与本文"计算缓冲效应"机制直接相关——是该机制的另一种工程化验证路径。

---

*原始论文：[arXiv 2603.09906](https://arxiv.org/abs/2603.09906) · PDF下载：[arxiv.org/pdf/2603.09906](https://arxiv.org/pdf/2603.09906) · 翻译整理 by Claude · 2026-03-12*
