---
title: "Transformer Circuits Thread — Anthropic 机械可解释性的持续更新圣经"
date: 2026-04-23 21:10:00 +0800
categories: [Resource Guide, Mechanistic Interpretability]
tags: [anthropic, circuits, interpretability, induction-head, superposition, features, sae]
---

## 基本信息

- **团队**: Anthropic Interpretability Team
- **类型**: 独立站点 + 可交互文章系列(持续更新,每篇独立可读)
- **主站**: [transformer-circuits.pub](https://transformer-circuits.pub/)

## 一句话总结

Anthropic 把"**机械可解释性 (mechanistic interpretability)**"的几乎每一步重大进展——从 induction head 的发现、到 superposition 假说、再到 Sparse Autoencoder 找特征、最后到 Claude 3 规模的 circuits 分析——用**数学 + 可交互图表 + 完整实验细节**呈现给公众。每篇文章都能独立读,但按时间线从头读完相当于把 mech interp 这个子领域的主线演进一次性体会到。

## 为什么它特殊

1. **不走传统会议论文套路**:发布为可交互的 HTML 长文,充分利用滑块、图表、悬停注释
2. **极高的数学透明度**:关键定义、引理、证明都公开;配套的 colab 笔记本可以复现核心分析
3. **每篇都有"we don't know yet"部分**:明确列出哪些结论是猜想、哪些反例尚未找到
4. **直接塑造了行业议程**:induction head、superposition、SAE、feature geometry 等术语大多由这个 thread 定义并推广

## 线索:几个关键里程碑

| 时间 | 标题 | 贡献 |
|------|------|------|
| 2021-12 | **A Mathematical Framework for Transformer Circuits** | 用数学把 attention 拆成可组合的 "OV / QK" circuits;定义 induction head |
| 2022-03 | **In-context Learning and Induction Heads** | 证实 induction head 与 ICL 能力的强相关,模型训练中的"相变"现象 |
| 2022-09 | **Toy Models of Superposition** | 提出 superposition 假说:一个神经元可能同时编码多个不相关特征 |
| 2023-10 | **Towards Monosemanticity(SAE)** | 用 Sparse Autoencoder 从 superposition 里"展开"出单义特征 |
| 2024-05 | **Scaling Monosemanticity** | 把 SAE 方法推到 Claude 3 Sonnet,发现 3000 万+ 可解释特征 |
| 2025 起 | **Circuits in Claude / Biology of LLMs** | 在生产级模型上做电路级分析 |

## 对哪些人最有价值

- **研究 attention 本身的人**:想理解 attention 到底学到了什么,这里是最系统的答案
- **设计 hybrid 架构**:想知道为什么几层 full attention 能补线性 attention 的 retrieval 能力,答案就在 induction head 理论里
- **做模型安全 / 对齐**:机械可解释性被视作"AI 对齐的测试集",这个 thread 是该方法论的发源
- **想理解 feature / circuit / neuron 的真正区别**:术语都是这里定义的,二手资料很容易搞错

## 阅读建议(给第一次来的人)

1. **先读** *A Mathematical Framework for Transformer Circuits* ——这是一切的起点,读完后续所有文章都变简单
2. **再读** *In-context Learning and Induction Heads* ——第一次看到 attention 里出现的"相变"曲线会很震撼
3. **然后按兴趣跳**:
   - 想搞模型内部表征 → *Toy Models of Superposition* + *Towards Monosemanticity*
   - 想做安全研究 → 最新的 *Biology of a Large Language Model* 系列
   - 想做工具 → 看 [TransformerLens 资源引介]({% post_url 2026-04-23-TransformerLens-资源引介 %})

## 注意事项

- 文章里很多交互图要 JavaScript 启用,移动端阅读体验不佳,建议桌面浏览器
- 有些早期文章的代码链接已迁移到新位置,以最新综述为主
- 数学符号频繁重用,跨文章时要留意上下文里的定义

## 延伸阅读

- [TransformerLens 资源引介]({% post_url 2026-04-23-TransformerLens-资源引介 %}) —— 能在家用 GPU 上复现 circuits 分析的工具
- [Attention Is All You Need 深度解读]({% post_url 2026-04-23-Attention-Is-All-You-Need-深度解读 %}) —— attention 原始机制,circuits 分析的对象
- [Anthropic 官方解释页](https://www.anthropic.com/research) —— 更高层次的研究概览
- [OpenAI Microscope 旧项目](https://microscope.openai.com/) —— 早期视觉模型的可解释性尝试,可对比阅读
