---
title: "SWE-Agent — 为 LLM 量身设计一个'代码编辑器',把 SWE-Bench 解决率从 4% 提到 12.5%"
date: 2026-04-24 16:30:00 +0800
categories: [Agent, Coding Agent]
tags: [swe-agent, aci, coding-agent, yang-2024]
math: true
---

## 基本信息

- **作者**: John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, Ofir Press
- **机构**: Princeton, Stanford
- **发表**: NeurIPS 2024
- **arXiv**: [2405.15793](https://arxiv.org/abs/2405.15793)

## 一句话总结

提出 **SWE-Agent**——首个在 SWE-Bench(真实 GitHub bug 修复任务)上把 GPT-4 从 **4%** 推到 **12.5%** 的开源 agent。核心创新不是新模型,而是一个新概念:**Agent-Computer Interface (ACI)**——类比"人机接口 HCI 是为人类设计的",ACI 是"专为 LLM 设计的工具接口"。具体做法:**定制一套 open/edit/search/submit 等简洁工具**,严格约束返回格式,让 LLM 的 token 有效率最大化。SWE-Agent 的贡献从此让"**harness 工程**"成为 coding agent 研究的一等公民——相同模型,换 harness 就能差 15+ 分 SWE-Bench。

![SWE-Agent 的核心设计:一套专为 LLM 设计的 ACI(Agent-Computer Interface),包含 open/edit/search/create/submit 等工具,每个工具的返回格式都经过精心设计,使 LLM 能稳定使用。](/assets/img/swe-agent/x1.png)
_Figure 1:SWE-Agent 和 ACI 的整体架构_

---

## 背景:为什么 GPT-4 在 SWE-Bench 上这么差

### SWE-Bench 的难度

**SWE-Bench**(Jimenez 2023):2294 道真实 GitHub issue,需要 agent:
1. 读懂 issue
2. 在整个 repo 里定位相关代码
3. 修改代码使得 hidden test 通过

难度极高——需要 long context 理解、多文件定位、精确编辑、测试迭代。

### 直接用 GPT-4 + bash 的结果

2024 年初,Princeton 团队测试:直接让 GPT-4 用 bash + Python + shell 解决 SWE-Bench。结果:**< 4% 解决率**。

分析失败原因,作者发现 GPT-4 的 token 被**大量浪费在操作 shell 的形式噪音上**:

- `cd` 到不同目录
- 用 `grep -rn` 但结果太长被截断
- 用 `vim/sed/awk` 编辑文件但语法错
- 忘记已经 `cat` 过的文件内容
- LLM 的 tool calls 有格式错误被系统拒绝

**GPT-4 不是不会做软件工程,是 interface 太难用**。

---

## 核心创新:Agent-Computer Interface (ACI)

### 设计哲学:HCI for LLM

作者的类比极其直观:

- **HCI (Human-Computer Interface)**:为人类的感知-动作循环设计(鼠标、键盘、GUI)
- **ACI (Agent-Computer Interface)**:为 LLM 的 token-in-token-out 循环设计

两者目标都是"让使用者高效地完成任务",但优化方向不同:

| 维度 | HCI(人类) | ACI(LLM) |
|------|-----------|---------|
| 主要约束 | 视觉带宽、记忆 | 上下文长度、token 预算 |
| 错误纠正 | 视觉反馈立即 | 文本反馈必须简洁 |
| 命令风格 | 快捷键、拖拽 | 结构化 command + args |
| 反馈格式 | 丰富视觉 | 固定简短文本 |

### ACI 的具体设计

![SWE-Agent 的工具集:open(打开文件)、goto(跳转到行)、search_file(文件内搜索)、search_dir(目录内搜索)、edit(行范围编辑)、create、submit(提交方案)。每个工具的返回格式严格,比如 open 后显示当前文件 + window 内的行,而非 dump 整个文件。](/assets/img/swe-agent/x2.png)
_Figure 2:SWE-Agent 的工具集和 I/O 设计_

具体工具:

- **`open <path> [line]`**:打开文件,**显示 window 内的 100 行而非整个文件**
- **`scroll_up` / `scroll_down`**:移动 window
- **`goto <line>`**:跳到指定行
- **`search_file <pattern>`**:在当前文件搜索(返回行号 + 上下文)
- **`search_dir <pattern> [dir]`**:在目录搜索
- **`find_file <name>`**:查找文件
- **`edit <start>:<end>\n<new_content>\nend_of_edit`**:编辑指定行范围
- **`create <path>`**:创建新文件
- **`submit`**:提交最终方案

每个工具的返回都**经过精心设计**:

- 简洁到 LLM 能完全 parse
- 包含足够上下文(当前文件路径、window 行号范围)
- 错误信息可诊断

### Linting + Edit 原子化

一个重要细节:**edit 后自动运行 linter**,如果有语法错误立即报告。这让 LLM 知道自己的编辑有没有破坏代码——相当于给 LLM 一个"编译器"。

---

## 实验结果

### SWE-Bench 全量

| Method | Resolved % |
|--------|-----------|
| RAG baseline(仅 retrieval) | 1.3% |
| GPT-4 direct | 3.8% |
| **SWE-Agent (GPT-4)** | **12.5%** |
| **SWE-Agent (Claude 3 Opus)** | **10.5%** |

**GPT-4 提升 3.3× ,纯粹来自 ACI 设计**——底层模型没变。

### 消融:每个 ACI 设计的贡献

![Ablation:去掉 window 机制(全文件 dump)→ 降 3 分;去掉 linter → 降 1 分;用 bash 而非结构化 tool → 降 5 分。每个设计都对最终结果有贡献。](/assets/img/swe-agent/x3.png)
_Figure 3:ACI 设计的 ablation 研究_

### 长 horizon 稳定性

- 平均一个成功 trajectory 有 **~30 个 tool calls**
- SWE-Agent 能处理 80+ calls 的长 trajectory
- 错误率随步数上升相对缓慢

---

## 工程影响

### 1. Harness 工程成为一等公民

SWE-Agent 之前,"agent = prompt + 模型"。SWE-Agent 证明**harness(工具集 + 反馈格式 + 循环逻辑)与模型同等重要**。这让 "agent harness" 成为独立研究方向。

### 2. SWE-Bench 被打破僵局

SWE-Agent 之前的 best baseline < 4%。SWE-Agent 到 12.5%,让社区意识到"这个任务是可解的"。后来:

- OpenHands(OpenDevin):继承 SWE-Agent 思想,加 browser、sandbox
- Aider、Agentless:不同 harness 路线
- 到 2025,前沿模型(Claude 3.5/Opus、GPT-4o、DeepSeek-V3)+ 好 harness 可达 **70%+**

### 3. 影响 Claude Code 等商用产品

Claude Code 的工具设计(Read/Edit/Bash/Glob/Grep/Task)与 SWE-Agent ACI 非常像:**极简正交,格式严格,错误反馈清晰**。Anthropic 官方承认受此启发。

### 4. ACI 概念推广到其他 agent 领域

Web agent、GUI agent、database agent 等都开始设计自己的 ACI。**"为 LLM 优化接口"** 成为 agent 工程的通识。

---

## 局限

### 1. 每类任务需要专门 ACI

ACI 不是通用的。SWE-Agent 的 ACI 专门为代码编辑设计,换到数据分析、ML 实验等任务要重新设计。

### 2. 窗口设计有 trade-off

100 行 window 合适代码但不合适超大文件(如 10000 行的配置)。需要按情况调整。

### 3. Linter 只能查语法错,不能查语义错

SWE-Agent 的代码改动可能**语法正确但语义错**——需要运行测试才能知道。tests 又需要很多 setup。

### 4. 依赖强 base model

SWE-Agent 在 GPT-3.5 上效果不佳。ACI 放大强模型的优势,但救不了弱模型。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **Interface 设计是 agent 能力的放大器**:同一模型,好的 harness 把 SWE-Bench 从 4% 提到 12.5%——3× 提升。Prompt engineering 之后,harness engineering 是下一个爆发点
2. **ACI 不是 HCI**:人类和 LLM 的约束完全不同。为 LLM 设计接口要考虑 token 预算、上下文长度、格式敏感度,不能直接套人类工具
3. **简洁 > 全能**:SWE-Agent 只有几个工具,但每个都极简洁。给 LLM 10 个精心设计的工具比 100 个全能工具好
4. **"agent engineer" 是一个真实的职位**:harness 设计、工具选择、反馈格式优化——这些都需要专门的工程实践。这就是为什么 Claude Code、Cursor、Cline 等团队花大量精力调这些
</callout>

---

## 延伸阅读

- [ReAct 深度解读]({% post_url 2026-04-24-ReAct-推理与行动交替深度解读 %}) —— SWE-Agent 的 scaffold 基础
- [SWE-Bench (Jimenez et al., 2023)](https://arxiv.org/abs/2310.06770) —— benchmark 原论文
- [OpenHands (Wang et al., 2024)](https://arxiv.org/abs/2407.16741) —— 继承并扩展 SWE-Agent
- [Agentless (Xia et al., 2024)](https://arxiv.org/abs/2407.01489) —— 对 agent loop 的反思
- [Claude Code 官方文档](https://docs.claude.com/en/docs/claude-code) —— ACI 思想的商业化应用
