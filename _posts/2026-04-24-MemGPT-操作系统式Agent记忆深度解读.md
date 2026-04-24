---
title: "MemGPT — 把操作系统的虚拟内存思想搬给 LLM,让 Agent 有长期记忆"
date: 2026-04-24 17:45:00 +0800
categories: [Agent, Memory]
tags: [memgpt, agent-memory, virtual-memory, packer-2023]
math: true
---

## 基本信息

- **作者**: Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez
- **机构**: UC Berkeley
- **发表**: arXiv 2023-10
- **arXiv**: [2310.08560](https://arxiv.org/abs/2310.08560)

## 一句话总结

提出 **MemGPT**——把**操作系统的虚拟内存**思想搬给 LLM 的 Agent 记忆系统。LLM 上下文窗口像"主内存",外部存储(向量库 + recall DB)像"磁盘",LLM 自己通过 **function calls** 决定把什么 page-in、把什么 page-out,实现**超越 context window 限制**的长期记忆。不改模型,只改 prompt + tool set,让普通的 GPT-4 + MemGPT 在多 session 对话、大文档 QA 上显著超越原生 LLM。MemGPT 是"Agent Memory"方向的奠基作,开源仓库后来演化为 **Letta**(现在主流的 agent memory 框架)。

![MemGPT 的核心类比:LLM 的 context window 是"主内存"(容量小、速度快),外部存储是"磁盘"(容量大、速度慢)。LLM 通过 function call 做 page in/out,自主管理哪些信息在 context 里。](/assets/img/memgpt/x1.png)
_Figure 1:MemGPT 的 OS-inspired 记忆架构_

---

## 背景:LLM 的 Context Window 天花板

### Context 限制的实际痛点

即使 2023 年底已有 32K-128K context 的模型,实际应用中仍遇到:

- **多 session 对话**:用户和 AI 聊了 100 天,100 天的历史塞不进 context
- **大文档 QA**:一本 500 页的书,远超任何 context
- **Agent 长程任务**:一个复杂软件项目可能涉及几百个文件、上万次交互

### 之前的两条路

- **扩 context window**:RoPE 扩展、YaRN、无限 context——但都有衰减和成本问题
- **RAG**:检索相关片段——但检索粒度、相关性判断都是难题,而且 **LLM 没法主动决定"我需要什么"**

MemGPT 的思路:**让 LLM 像 OS 一样主动管理记忆**。

---

## 核心机制:操作系统的虚拟内存类比

### 三层记忆架构

| 层级 | OS 类比 | MemGPT 实现 | 容量 | 访问速度 |
|------|---------|-------------|------|----------|
| **Main Context** | RAM | LLM 的 context window | 8K-128K tokens | 无限快 |
| **Recall Storage** | 磁盘(近期) | 对话历史数据库 | 无限 | 中等 |
| **Archival Storage** | 磁盘(归档) | 向量数据库 | 无限 | 较慢 |

### Main Context 内部分区

Context window 本身被 MemGPT 进一步分为:

- **System Prompt**(固定 + 只读):定义 MemGPT 身份和工具用法
- **Working Memory**(只读 + 可更新):当前会话的核心事实
- **Recent Messages**(FIFO 队列):最近对话
- **Scratchpad**(可写):临时计算

### LLM 主动管理记忆的 function calls

关键创新:**LLM 通过 function call 主动调度记忆**。主要工具:

- `append_to_working_memory(content)`:把重要信息写入 working memory
- `evict_from_working_memory(id)`:移除不再重要的
- `search_archival(query)`:查归档
- `insert_archival(content)`:存入归档
- `search_recall(query)`:查近期对话
- `send_message(content)`:对用户回复

**这些工具的使用完全由 LLM 决定**——就像 OS 里进程通过 syscall 申请内存。

### 自我调度的闭环

![MemGPT 工作流:LLM 接收用户消息 → 判断是否需要查历史 → 决定是否把新信息写入 working memory → 生成回复。这个闭环由 LLM 自主控制。](/assets/img/memgpt/x2.png)
_Figure 2:MemGPT 的自主记忆调度_

例:

```
User: "记住我喜欢咖啡。"
MemGPT: [call append_to_working_memory("user likes coffee")]
MemGPT: "好的,我记住了。"

...(100 条对话后)...

User: "推荐一家适合我的咖啡店。"
MemGPT: [reads working_memory: "user likes coffee"]
MemGPT: "根据您之前提到喜欢咖啡..."
```

LLM **自己**决定什么时候存、什么时候查——这是 MemGPT 与传统 RAG 的根本区别。

---

## 实验结果

### 1. 多 session 对话(Multi-Session Chat)

MSC benchmark:跨 5 个 session 的对话,考察 agent 是否记住早期事实。

![MemGPT 在 MSC 上大幅超过 GPT-3.5 原生(无记忆)和 vanilla 扩展 context 方案。跨 session 一致性保持最好。](/assets/img/memgpt/x3.png)
_Figure 3:Multi-Session Chat 结果_

| Method | Session-5 Consistency |
|--------|----------------------|
| GPT-3.5 原生(无记忆) | 32% |
| GPT-3.5 + context extension | 48% |
| **GPT-3.5 + MemGPT** | **72%** |
| **GPT-4 + MemGPT** | **83%** |

### 2. 大文档 QA

![MemGPT 在长文档 QA 任务上对比 RAG:MemGPT 能主动多轮检索、更新理解,比单次 RAG 准确率高 20-30 分。](/assets/img/memgpt/x4.png)
_Figure 4:长文档 QA 上 MemGPT vs RAG_

关键观察:

- **RAG** 一次检索,依赖 query 质量
- **MemGPT** 可以多轮检索、精炼 query、整合多段信息——**像人读书一样"翻回去查"**

### 3. 工具使用成功率

![MemGPT 的 function call 正确率在不同 base model 上的表现。GPT-4 接近 100%,GPT-3.5 约 85%。小模型(<7B)使用 MemGPT 则基本失败。](/assets/img/memgpt/x5.png)
_Figure 5:function call 成功率_

再次验证 **agent 能力的模型规模依赖**:强模型才能正确使用 MemGPT 的复杂工具集。

---

## 工程影响

### 1. 开创"Agent Memory"方向

MemGPT 之前,LLM memory 主要是 RAG + 单次检索。MemGPT 首次提出**主动、分层、self-managed**的 memory 架构。这直接影响了后续 Mem0、HippoRAG、Letta、Claude Code auto memory 等众多系统。

### 2. Letta 框架的诞生

MemGPT 开源后演化为 **Letta**(原 MemGPT)——一个生产级 agent memory 框架,被各种 agent 应用使用。2024 年 Letta 获 Sequoia 投资,成为"agent memory"这个赛道的代表公司。

### 3. 启发 "OS for Agents" 思想

MemGPT 的 "LLM as OS" 类比启发了一系列更激进的思想:

- **AIOS**(Mei 2024):把 LLM 当作 OS 内核
- **SWE-Agent**:agent-computer interface
- **OpenDevin/OpenHands**:workspace 哲学

### 4. 双向影响 Context Window 研究

MemGPT 证明了**即使 context 足够大,主动 memory management 仍有价值**——因为 context 里"什么应该在"比"context 有多大"更关键。这影响了后续对 context engineering 的认识。

---

## 局限

### 1. 对 base model 要求高

MemGPT 的复杂工具集只有 GPT-4 级别模型能可靠使用。小模型用 MemGPT 经常失败。

### 2. 调用开销

每次决策是否调用 memory 工具本身就是一个 LLM call。复杂场景下 token 开销显著增加。

### 3. Memory 一致性问题

LLM 可能写入矛盾信息(如用户先说"喜欢咖啡",后改口"不喜欢咖啡")。MemGPT 没有显式的 memory 一致性管理。Mem0 等后续工作在这点上改进。

### 4. 没有遗忘机制

所有信息永久存储。对老旧、不相关信息没有自然淘汰机制。MemoryBank 的 Ebbinghaus 遗忘曲线是一个补充方向。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **LLM 可以像 OS 一样管理自己的记忆**:OS 的虚拟内存、分页、syscall 等概念可以直接映射到 agent memory 系统——这是一个极好的跨领域类比
2. **主动管理 > 被动检索**:RAG 是被动的(每次查询),MemGPT 是主动的(LLM 自己决定存什么、查什么)。主动性让 agent 的信息管理质量质变
3. **分层记忆是实用架构**:main context(快)+ recall(中)+ archival(慢)的三层划分,与 CPU L1/L2/L3 cache 的哲学一致,性价比最好
4. **Context engineering 比 context extension 更重要**:扩 context 到 1M 不等于解决问题;更关键的是"在有限 context 里放对东西"——MemGPT 从工程上践行了这个原则
</callout>

---

## 延伸阅读

- [AI Agent 记忆系统全面解读(飞书文档)](https://feishu.cn/wiki/EUBmwWrPii1j5Skrwt0ccVqan7t) —— 记忆方向全景
- [Mem0 深度解读]({% post_url 2026-04-24-Mem0-工业级长期记忆深度解读 %}) —— 接续 MemGPT 的工业化路线
- [Generative Agents (Park et al., 2023)](https://arxiv.org/abs/2304.03442) —— 另一种记忆架构
- [HippoRAG (Gutiérrez et al., 2024)](https://arxiv.org/abs/2405.14831) —— 图记忆路线
- [Letta 官方仓库](https://github.com/letta-ai/letta) —— MemGPT 的生产化演进
