---
title: "Attention Is All You Need"
date: 2025-02-15 08:00:00 +0800
categories: [NLP, Transformer]
tags: [attention, sequence-modeling, google, classic, deep-learning]
math: true
---

## 基本信息

- **作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- **机构**: Google Brain, Google Research
- **发表**: NIPS 2017 (NeurIPS 2017)
- **arXiv**: [1706.03762](https://arxiv.org/abs/1706.03762)
- **代码**: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

## 一句话总结

提出了 **Transformer** 架构，完全基于 **Self-Attention** 机制，摒弃了传统的 RNN 和 CNN，成为现代自然语言处理（NLP）和后续多模态学习的基石。

## 背景与动机

### 传统序列模型的局限

1. **RNN/LSTM/GRU**: 顺序计算，难以并行，长距离依赖建模困难
2. **CNN**: 局部连接，捕捉长距离依赖需要多层堆叠
3. **Encoder-Decoder + Attention**: 仍依赖 RNN 进行编码

### 核心问题

能否完全基于注意力机制，构建一个不使用循环和卷积的序列转导模型？

## 核心贡献

### 1. Self-Attention 机制

核心公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q$ (Query): 查询矩阵
- $K$ (Key): 键矩阵  
- $V$ (Value): 值矩阵
- $d_k$: Key 的维度，缩放因子防止 softmax 进入饱和区

### 2. Multi-Head Attention

并行计算多组注意力，捕捉不同子空间的特征：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

- 论文使用 $h=8$ 个头，$d_k = d_v = d_{model}/h = 64$

### 3. Position Encoding

由于不含循环和卷积，需要显式注入位置信息：

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$

$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

优点：
- 可以外推到比训练时更长的序列
- 相对位置可以表示为位置的线性函数

### 4. 完整的 Transformer 架构

```
Encoder (N=6 layers):
  - Multi-Head Self-Attention
  - Feed-Forward Network (FFN)
  - Residual Connection + LayerNorm

Decoder (N=6 layers):
  - Masked Multi-Head Self-Attention
  - Multi-Head Cross-Attention (Encoder-Decoder)
  - Feed-Forward Network
  - Residual Connection + LayerNorm
```

## 实验结果

### 机器翻译 (WMT 2014)

| 模型 | EN-DE BLEU | EN-FR BLEU | 训练时间 |
|------|-----------|-----------|---------|
| GNMT (Google) | 24.6 | - | 7 天 (96 TPU) |
| ConvS2S | 25.16 | 40.46 | - |
| **Transformer (base)** | **27.3** | **38.1** | **12 小时 (8 P100)** |
| **Transformer (big)** | **28.4** | **41.8** | **3.5 天 (8 P100)** |

### 关键发现

1. **训练速度**: 相比基于 RNN 的模型，训练时间大幅缩短
2. **泛化能力**: 在 English Constituency Parsing 上也取得 SOTA
3. **可扩展性**: 模型越大，效果越好，没有明显的性能饱和

## 消融实验

| 变体 | BLEU | 说明 |
|------|------|------|
| 基础模型 | 27.3 | 标准配置 |
| - positional encoding | 失效 | 位置信息至关重要 |
| - multi-head (single) | 25.8 | 多头机制提升明显 |
| - dot product attention | 25.6 | 缩放因子很重要 |
| - bigger model | 28.4 | 模型规模带来提升 |

## 个人思考

### 为什么 Transformer 能 work？

1. **并行计算**: 矩阵乘法可高度并行，适合 GPU/TPU 加速
2. **长距离依赖**: Self-attention 中任意位置距离都是 $O(1)$
3. **可解释性**: 注意力权重可以可视化，观察模型关注哪里

### 局限与改进方向

| 局限 | 后续改进 |
|------|---------|
| $O(n^2)$ 复杂度 | Linear Attention, Performer, Linformer |
| 缺乏归纳偏置 | 引入卷积偏置，如 Conformer |
| 位置编码固定 | 可学习位置编码，旋转位置编码 (RoPE) |
| 需要大量数据 | 预训练 + 微调范式 (BERT, GPT) |

### 后续跟进

- [ ] Linear Attention: 降低计算复杂度到 $O(n)$
- [ ] Flash Attention: IO-aware 的注意力优化
- [ ] Mamba/SSM: 探索非注意力架构
- [ ] Mixture of Experts: 稀疏激活的大模型

## 引用

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

## 相关阅读

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - 双向编码器
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - 解码器生成
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Transformer 进军 CV
