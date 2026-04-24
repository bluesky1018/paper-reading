---
title: "UI-TARS — 端到端训练的原生 GUI Agent,让 LLM 直接操作屏幕"
date: 2026-04-24 18:30:00 +0800
categories: [Agent, Computer-Use, Multimodal]
tags: [ui-tars, gui-agent, computer-use, bytedance-2025]
math: true
---

## 基本信息

- **作者**: Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, ...(ByteDance)
- **机构**: ByteDance, Tsinghua
- **发表**: arXiv 2025-01
- **arXiv**: [2501.12326](https://arxiv.org/abs/2501.12326)

## 一句话总结

提出 **UI-TARS**——ByteDance 发布的**原生端到端 GUI Agent 模型**,直接输入屏幕截图、输出键鼠动作,完全抛弃了"prompt engineering + screenshot caption + action API"的传统 pipeline。模型从 2B 到 72B,在 **OSWorld 达到 24.6%**(GPT-4o baseline ~12%),**AndroidWorld 33%**(领先主流方案)。关键技术:(1) **五阶段训练**——perception / grounding / planning / reasoning / reflection 逐步构建能力;(2) **reflection tuning**——让模型学会出错时反思并恢复;(3) **system-2 思考**——支持长链推理再执行。UI-TARS 标志着 GUI Agent 进入"end-to-end 原生模型"时代,与 Anthropic Computer Use、OpenAI Operator 三足鼎立。

![UI-TARS 在 OSWorld(桌面)、AndroidWorld(手机)、ScreenSpot(GUI grounding)、Multimodal Mind2Web(网页)四大 benchmark 上全面领先主流方案。](/assets/img/uitars/x1.png)
_Figure 1:UI-TARS 跨基准性能_

---

## 背景:GUI Agent 的三代演化

### 2023:Prompt + Screenshot + OS API

早期方案(WebGPT、AutoGPT 桌面版):

- 截屏 → 用 OCR / caption 模型转文字
- 把文字 + 任务描述输入 GPT-4
- GPT-4 输出 action(点击、输入)
- 通过 accessibility API 执行

问题:**OCR/caption 丢失空间信息,GPT-4 不懂 GUI 布局,action 表达不精确**。

### 2024:Vision-Language Model + Grounding

CogAgent、SeeClick、ShowUI 等:

- 直接用 VLM 看截图
- 专门训练"GUI grounding"(指定一个按钮 → 输出坐标)
- 仍然 prompt-driven,一次一个 action

问题:**仍是 ReAct 式 prompt loop,长 horizon 任务易失败**。

### 2025:End-to-end 原生模型

UI-TARS、Claude Computer Use、OpenAI Operator:

- **模型本身就是 GUI Agent**——端到端训练,输入截图 + 任务,输出动作序列
- 没有显式的"截图 → 文字 → 决策 → 动作"几个阶段
- 支持复杂长 horizon 任务

UI-TARS 是这一代的开源代表。

---

## 核心机制

### 1. 五阶段训练 pipeline

![UI-TARS 的五阶段训练:Perception(感知 UI 元素)→ Grounding(元素定位)→ Planning(任务分解)→ Reasoning(长链推理)→ Reflection(错误反思)。每阶段都有专门的数据和训练目标。](/assets/img/uitars/x3.png)
_Figure 2:UI-TARS 的五阶段训练_

**Stage 1: Perception**
- 训练模型理解截图中有什么 UI 元素
- 数据:截图 + 密集元素标注(bounding box + 类别)

**Stage 2: Grounding**
- 给定描述(如"登录按钮"),输出精确坐标
- 数据:截图 + (description, coord) 对

**Stage 3: Planning**
- 给任务,输出分解成 steps
- 数据:task + step-by-step plan

**Stage 4: Reasoning**
- 让模型在每 step 之前写思考(long CoT)
- 数据:task + step + thought + action 轨迹

**Stage 5: Reflection**
- 让模型看到"上一步错了"时反思并尝试恢复
- 数据:带错误 + 恢复的 trajectory

每阶段用 SFT + RL,逐层构建能力。

### 2. 核心能力组件

![UI-TARS 的核心能力:感知(看懂屏幕)、reasoning(长链思考)、acting(精确操作)、memory(跨步骤状态)。这四项能力通过端到端训练融为一体。](/assets/img/uitars/x2.png)
_Figure 3:UI-TARS 的四大核心能力_

### 3. 原生动作空间

UI-TARS 的输出不是自然语言的"点一下登录按钮",而是**直接的键鼠动作**:

- `click(x, y)`
- `type(text)`
- `scroll(delta)`
- `key(combo)`
- `drag(x1, y1, x2, y2)`

**与鼠键事件一一对应**,不需要 accessibility API 转译。

### 4. Reflection Tuning(反思微调)

![UI-TARS 的反思数据:故意构造"走错一步后恢复"的轨迹,让模型学会 "遇到错误 → 观察 → 反思 → 纠正" 的能力,而不是盲目继续。](/assets/img/uitars/x4.png)
_Figure 4:Reflection Tuning 的数据构造_

关键技巧:**在训练数据中故意包含错误 + 反思恢复**。

构造方式:

- 让模型自己跑 trajectory,捕获失败案例
- 用更强模型(或人类)标注"错在哪",给出正确恢复步骤
- 把 "错误 trajectory + 反思 + 恢复" 整个序列作为新 SFT 数据

**效果**:模型遇到错误不再"blindly carry on",而是主动检测并恢复。这是从 DeepSeek-R1 继承的思想,应用到 GUI 场景。

---

## 实验结果

### OSWorld(桌面 Agent)

| Method | Success Rate |
|--------|-------------|
| GPT-4V + Prompt | 12.2% |
| Claude 3.5 Computer Use | 22.0% |
| **UI-TARS 72B** | **24.6%** |

### AndroidWorld

| Method | Success Rate |
|--------|-------------|
| SeeClick | 10.5% |
| GPT-4o + Mobile Agent | 28% |
| **UI-TARS 72B** | **33.1%** |

### ScreenSpot(GUI grounding)

| Method | Acc |
|--------|-----|
| GPT-4V | 30.1% |
| SeeClick | 75.6% |
| **UI-TARS 7B** | **89.5%** |

---

## 与 Claude Computer Use / Operator 的对比

| 维度 | Anthropic Computer Use | OpenAI Operator | **UI-TARS** |
|------|----------------------|-----------------|-------------|
| 模型 | Claude 3.5 Sonnet | GPT-4o 变体 | 2B/7B/72B 专用 |
| 训练 | 大概率有 computer-use RL | 同上 | 公开五阶段 recipe |
| API | 付费 API | 付费 API | **开源权重** |
| 长 horizon 能力 | 强 | 强 | 中等(但在提升) |
| 性价比 | 高 API 成本 | 同 | **自托管低成本** |

UI-TARS 的最大价值:**开源**。社区可以复现、微调、部署到自己环境。

---

## 工程影响

### 1. 开源 GUI Agent 的基准模型

UI-TARS 发布后,围绕它构建的 agent 产品快速涌现:

- 字节自己的"豆包桌面"
- 开源社区的 Browser-Use、Showrunner 等
- 企业内部 RPA 2.0(用 UI-TARS 代替传统 RPA)

### 2. "端到端训练 > prompt loop" 共识确立

社区广泛接受:**通用 LLM + prompt GUI workflow 永远追不上专门训练的 GUI 模型**。这推动 OpenAI、Anthropic 投入资源做专用模型。

### 3. 反思 tuning 思想的扩散

UI-TARS 的 reflection tuning 被其他 agent 工作借鉴——SWE-Gym 2.0、Search-R1 等都有"故意制造错误再教纠正"的数据构造思路。

### 4. GUI Agent 走向 RPA 替代

GUI Agent 开始真正进入企业 RPA 场景——**自动化办公软件操作、数据填表、跨系统集成**等。UI-TARS 类开源模型加速了这个转变。

---

## 局限

### 1. 仍未到"人类水平"

OSWorld 人类 72%,UI-TARS 24.6%——还有巨大差距。复杂任务(涉及多应用、文件操作、时序依赖)仍频繁失败。

### 2. 部署硬件要求

72B 模型需要多卡推理,对中小企业不友好。7B 版本能力明显弱。

### 3. 英文 GUI 为主

训练数据以英文 Windows/Ubuntu/Android 为主。中文 GUI、小众应用支持有限。

### 4. 没有跨应用长记忆

UI-TARS 专注单任务执行,没有"跨任务积累经验"的机制。这和传统 agent 的 memory 系统还没很好集成。

---

## 读完这篇你真正该带走的 4 件事

<callout emoji="white_check_mark" background-color="light-green" border-color="green">
1. **专用模型 > 通用模型 + prompt**:GUI Agent 领域,专门训练的模型把 prompt-based GPT-4o 碾压——这验证了"为特定任务专训"是比"搞一个 super-LLM"更务实的路线
2. **五阶段训练是通用 recipe**:Perception → Grounding → Planning → Reasoning → Reflection 的分层是 agent capability 的清晰拆解,可以迁移到其他 agent 任务
3. **Reflection Tuning 是 agent RL 的关键技巧**:不只是训练"怎么做对",也训练"做错了怎么办"——这种 "error-aware" 数据是高可靠 agent 的秘方
4. **开源 GUI Agent 的生态价值**:UI-TARS 开源后迅速催生一批工具链——开源 model 降低 agent 创业门槛,是推动整个行业的基础设施
</callout>

---

## 延伸阅读

- [SWE-Agent 深度解读]({% post_url 2026-04-24-SWE-Agent-Agent-Computer-Interface深度解读 %}) —— Coding Agent 的对应工作
- [CogAgent (清华, 2023)](https://arxiv.org/abs/2312.08914) —— GUI Agent 的早期代表
- [SeeClick (2024)](https://arxiv.org/abs/2401.10935) —— GUI Grounding 专门工作
- [DeepSeek-R1 深度解读]({% post_url 2026-04-24-DeepSeek-R1-RL推理模型深度解读 %}) —— Reflection 思想的源头
- [UI-TARS 官方仓库](https://github.com/bytedance/UI-TARS) —— 开源权重和代码
