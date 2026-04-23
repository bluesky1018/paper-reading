---
title: "The Annotated Transformer — 哈佛 NLP 逐行讲透 Transformer 的经典教程"
date: 2026-04-23 21:12:00 +0800
categories: [Resource Guide, Education]
tags: [annotated-transformer, harvard-nlp, pytorch, tutorial, attention-is-all-you-need]
---

## 基本信息

- **原作者**: Alexander Rush(Harvard NLP,现 Cornell Tech)
- **2022 重写版作者**: Austin Huang 等
- **类型**: 交互式 Notebook 风格教程(HTML + colab)
- **原址**: [nlp.seas.harvard.edu/annotated-transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

## 一句话总结

把 *Attention Is All You Need* 论文**逐段拆开**,在每一段旁边贴上对应的 PyTorch 实现代码,形成一份"左边论文、右边代码"的对照手册。全篇可在 colab 里一键跑起来,整个 Transformer 从 embedding 到 beam search 都可以端到端训一遍。是**社区里被引用最多的 Transformer 入门材料**之一,也是 nanoGPT 等后继教学项目的精神前辈。

## 为什么经久不衰

- **论文 + 代码逐段对齐**:读到"Attention 公式"就能立刻看到 `scaled_dot_product`,不用来回切换
- **可完整训练**:在 colab 里可以训一个小尺度 seq2seq 翻译模型,实际看到 loss 下降、BLEU 上升
- **测试友好**:每个小模块都有 `test_*` 函数,可以逐步验证自己实现正确
- **PyTorch 原生**:没有任何 framework wrapper,教学价值高

## 它覆盖什么

按论文顺序全部实现:

| 模块 | 对应论文章节 |
|------|-------------|
| Scaled Dot-Product Attention | §3.2.1 |
| Multi-Head Attention | §3.2.2 |
| Position-wise FFN | §3.3 |
| Positional Encoding(sinusoidal)| §3.5 |
| Encoder / Decoder Stack | §3.1 |
| Label Smoothing + KL Loss | §5.4 |
| Adam + Warmup Schedule | §5.3 |
| Beam Search | §3.3(附录)|
| WMT EN-DE 训练 pipeline | §6.1 |

## 2022 年的重写

原版(2018)用的是早期 PyTorch 风格,用 `torchtext` 老 API、没有模块化。2022 年 Harvard NLP 做了全面重写:

- PyTorch 1.10+,torchtext 新 API
- 更清晰的模块拆分、类型注解
- 修复了若干数值小 bug
- 更新了 label smoothing 实现,与论文完全对齐

2022 版是目前的事实标准。

## 使用场景

- **第一次读 *Attention Is All You Need***:论文旁边打开它,边读边 run,能快速建立"公式 ↔ 代码"对应关系
- **教学/组内读书会**:两周课程容量正合适,学员能带走一份可运行代码
- **复习**:面试前快速扫一遍 attention、多头、positional encoding 的关键代码

## 什么时候不该用

- 想训**真正 LLM 级别**的模型:它只针对 seq2seq 翻译,不能直接套用到 GPT 风格 LM
- 想看**最现代的架构**:里面是 2017 原始 Transformer(有 encoder-decoder、正弦位置、无 RoPE、无 GQA),要看 GPT 风格看 [nanoGPT]({% post_url 2026-04-23-nanoGPT-资源引介 %})
- 想看**分布式 / 混合精度**:不涉及,在单 GPU 场景下教学

## 和 nanoGPT 的区别

| | Annotated Transformer | nanoGPT |
|---|-----------------------|---------|
| 模型类型 | 原始 encoder-decoder(翻译)| decoder-only GPT(LM)|
| 代码风格 | 学术风、模块化 | 工程风、极简 |
| 配套文档 | 长 notebook + 论文对照 | README + 一篇博客 + 视频 |
| 覆盖深度 | 单 GPU,含 beam search / label smoothing | 基础 + 少量现代 tricks |
| 适合阶段 | 第一次学 Transformer | 读完 Annotated 后想练手魔改 |

**建议先 Annotated,再 nanoGPT**。

## 延伸阅读

- [Attention Is All You Need 深度解读]({% post_url 2026-04-23-Attention-Is-All-You-Need-深度解读 %}) —— 配合 Annotated 最佳
- [nanoGPT 资源引介]({% post_url 2026-04-23-nanoGPT-资源引介 %}) —— 下一步学习
- [Harvard NLP 博客](http://nlp.seas.harvard.edu/) —— 团队其他资源
- [Transformer from Scratch (Peter Bloem)](https://peterbloem.nl/blog/transformers) —— 类似风格的另一份优秀教程
