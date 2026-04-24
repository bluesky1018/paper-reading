---
title: "Toolformer — 让 LLM 自己教自己使用工具的自监督方法"
date: 2026-04-24 16:00:00 +0800
categories: [Agent, Tool Use]
tags: [toolformer, tool-learning, self-supervised, schick-2023]
math: true
---

## 基本信息

- **作者**: Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom
- **机构**: Meta AI Research
- **发表**: NeurIPS 2023
- **arXiv**: [2302.04761](https://arxiv.org/abs/2302.04761)

## 一句话总结

提出 **Toolformer**——首个让 LLM **自己给自己造训练数据学会用工具**的方法。关键思想:用 in-context learning 让 LLM 在海量文本中**自主提议**"这里可以调用 X API",然后**用一个客观指标**(API 调用前后 language modeling loss 是否下降)筛选出真正有用的调用,只保留这些样本作为训练数据。得到的 Toolformer(基于 6.7B GPT-J)能无缝调用 5 个 API(QA、Calculator、Translation、Wikipedia、Calendar),在下游任务上大幅超越基线甚至超过 GPT-3 175B。这是**工具学习**方向的开山之作,奠定了"让模型自己发现工具使用场景"这一范式。

![Toolformer 的 API 调用格式:在生成文本中插入 <API>Q("...")→result</API> 标记,模型用这些标记暂停生成、调用外部工具、消费返回值,再继续生成。](/assets/img/toolformer/x1.png)
_Figure 1:Toolformer 的 API 调用语法_

---

## 背景:LLM 不会用工具

### GPT-3 的硬伤

2022 年的大模型在不少任务上表现惊艳,但有些能力天然不行:

- **算术**:GPT-3 对大数乘法经常错(训练数据里这些"正确答案"密度低)
- **事实查询**:对小众实体、最新信息容易幻觉
- **翻译**:对低资源语言差
- **日期/时间**:没有"当前时间"概念

这些能力**不需要模型自己算**——外部工具(计算器、搜索引擎、翻译 API)早就存在。问题是:**LLM 不知道何时调用哪个工具**。

### 早期方案的局限

- **Few-shot prompting**(ReAct 式):只适用于有少量 agent 示例的场景,泛化差
- **Fine-tune with human data**:人工标注 API 调用位置太贵
- **RL with reward model**:训练成本高

Toolformer 的突破:**用 LLM 自己标注训练数据**,完全不需要人工。

---

## 核心机制:三步自监督数据构建

![Toolformer 数据构建的三个步骤:Sample candidates → Execute → Filter by loss。只保留那些"加了 API 调用后 LM loss 显著下降"的样本作为训练数据。](/assets/img/toolformer/x2.png)
_Figure 2:自监督数据构建 pipeline_

### Step 1: Sample API call candidates

对每个 API,设计一个"教 LLM 何时调用它"的 few-shot prompt。

对文本 $\mathbf{x} = [x_1, ..., x_n]$ 的每个位置 $i$,让 LLM 决定"在这里插入 API 调用会不会有帮助"。如果决定插入,生成具体的 API 调用。

对一段 10 万词的语料,LLM 可能生成数千个"候选 API 调用",大部分质量参差不齐。

### Step 2: Execute API calls

对每个候选调用,实际执行对应 API,得到返回值。

比如遇到 "The Paris 2024 Olympics took place from 26 July to...",LLM 提议调用 `Calendar()`,执行返回 "2024-07-26"。

### Step 3: Filter by self-supervised loss

**关键筛选步骤**。对每个候选,比较两个 loss:

- $L_i^-$:没有 API 调用,直接预测后续 token 的 loss
- $L_i^+$:API 调用结果作为 context,再预测后续 token 的 loss

如果 $L_i^+ < L_i^- - \tau$(阈值 $\tau$ 约 0.5),说明**这个调用真的降低了 language modeling loss**——保留。否则丢弃。

这是一个**纯自监督**信号:不需要人工标签,不需要答案正确性——**只看 API 是否让 LLM 更好地预测后续文本**。

### Step 4: Fine-tune

用通过筛选的样本(~100K 条)在原始 LM 目标上微调 GPT-J。LM 学到"在合适位置生成 API 调用 token + 使用返回值"的行为。

---

## API 调用的 inline 语法

Toolformer 定义了一种 inline API 语法:

```
The population of Canada is [QA("What is the population of Canada?")→38 million] about 38 million.
```

模型生成时:

1. 生成 `[QA("What is the population of Canada?")`
2. 系统拦截并执行 QA API,得到 "38 million"
3. 把 `→38 million]` 追加到 context
4. LM 继续生成后续文本

**不需要外部 orchestrator**,工具调用是 LM 自己输出的 token 的一部分。

---

## 实验结果

### 核心数字

![Toolformer (6.7B) vs GPT-J (6.7B, 无工具) 和 GPT-3 (175B) 在多个任务上的对比。Toolformer 在几乎所有任务上大幅领先 GPT-J,在大部分任务上甚至超过 GPT-3。](/assets/img/toolformer/x3.png)
_Figure 3:Toolformer 跨任务性能_

| Task | GPT-J | GPT-3 | **Toolformer** |
|------|-------|-------|---------------|
| LAMA (事实 QA) | 16.5 | 31.1 | **34.0** |
| ASDiv (数学) | 7.5 | 14.0 | **40.4** |
| MLQA (翻译) | 24.3 | 23.9 | **31.3** |
| TempLAMA (日期) | 13.7 | 0.3 | **16.3** |
| WikiSearch | 11.7 | 38.0 | **30.4** |

**6.7B 的 Toolformer 在数学、翻译、日期上超过 175B 的 GPT-3** —— 一个 26× 小的模型靠工具反超。

### 工具使用的模式

![Toolformer 会在合适的位置自主调用合适的工具:遇到算术调 Calculator,遇到人名调 QA,遇到日期调 Calendar。这种分工是从自监督数据中学来的,没有人工指定。](/assets/img/toolformer/x4.png)
_Figure 4:Toolformer 的工具使用分布_

- 算术任务:95% 调用 Calculator
- 翻译任务:80% 调用 Translation
- 事实 QA:60% 调用 QA,20% 调用 Wikipedia
- 无需工具的任务:<5% 调用任何工具(不 over-use)

---

## 工程影响

### 1. 开创"自监督工具学习"范式

Toolformer 之前,工具使用都靠人工标注 / RL。Toolformer 证明**纯 LM objective 就可以筛出好的工具调用样本**——打开了一个新范式,被后续 ToolLLM、Gorilla 等继承发扬。

### 2. 启发 OpenAI Function Calling

2023 年 6 月 OpenAI 推出 Function Calling 时,其 inline API 语法思想与 Toolformer 几乎完全一致。虽然 OpenAI 用的是不同实现(结构化 JSON 而非 inline token),但背后思想相通。

### 3. 证明小模型 + 工具 > 大模型 no tools

Toolformer 的 6.7B 超过 175B 的 GPT-3 是一个重要启示:**对特定任务,让模型调用合适工具比靠模型内部参数更有效**。这支持了后续"small model + powerful tools"的产品路线。

### 4. 局限启发下一代工作

Toolformer 的单工具调用、缺乏组合、训练数据局限等问题催生了 ToolLLM(万级真实 API)、Gorilla(retrieval + fine-tune)、ReAct(多步交互)等后续工作。

---

## 局限

### 1. 不支持多步组合

Toolformer 每次调用独立,不能"先查 A,再根据 A 调 B"。ReAct 解决这点。

### 2. API 种类固定

5 个 API 是预定义的,加新 API 需要重新构建数据和微调。ToolLLM 后来把这点扩展到万级。

### 3. 不支持复杂参数

API 参数相对简单(字符串、数字)。复杂嵌套参数 Toolformer 不擅长。Function Calling 的 JSON Schema 解决了这点。

### 4. 评估集偏简单

论文评估的主要是 QA / 算术 / 翻译这类简单任务。更复杂的 agent 任务(长 horizon、多工具组合)Toolformer 不擅长。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **LM loss 是一个出人意料的好教师**:降低 LM loss 作为"这个工具调用是否有用"的信号,完全自监督,不需要人工标签——这个 trick 优雅到惊艳
2. **小模型 + 工具 > 大模型无工具**:Toolformer 6.7B 超过 GPT-3 175B。对特定任务,工具比参数更有价值
3. **Inline API 语法是关键工程创新**:让工具调用成为 LM 生成的 token 序列一部分,而非外部调度——这个思路被 OpenAI Function Calling 基本继承
4. **自监督工具学习开创新范式**:不需要人工标注,不需要 RL,只需要原始文本 + 候选 API + 筛选——这套 pipeline 至今仍是工具学习工作的基础
</callout>

---

## 延伸阅读

- [ReAct 深度解读]({% post_url 2026-04-24-ReAct-推理与行动交替深度解读 %}) —— 多步工具使用
- [ToolLLM (Qin et al., 2023)](https://arxiv.org/abs/2307.16789) —— 扩展到万级真实 API
- [Gorilla (Patil et al., 2023)](https://arxiv.org/abs/2305.15334) —— Retrieval + Fine-tune 工具使用
- [OpenAI Function Calling 官方文档](https://platform.openai.com/docs/guides/function-calling) —— 工业化版本
- [MCP (Anthropic, 2024)](https://modelcontextprotocol.io/) —— 工具协议的终极形态
