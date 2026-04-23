---
title: "Dao-AILab/flash-attention — FlashAttention 1/2/3 的官方 kernel 源"
date: 2026-04-23 21:08:00 +0800
categories: [Resource Guide, Attention, CUDA]
tags: [flash-attention, flash-attention-2, flash-attention-3, cuda, triton, tri-dao, official-repo]
---

## 基本信息

- **作者**: Tri Dao 等(Princeton / Together AI)
- **类型**: 开源代码仓(BSD 3-Clause)
- **仓库**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **论文**:
  - [FA-1 (2022)](https://arxiv.org/abs/2205.14135)
  - [FA-2 (2023)](https://arxiv.org/abs/2307.08691)
  - [FA-3 (2024)](https://arxiv.org/abs/2407.08608)

## 一句话总结

**FlashAttention 三代算法的官方 CUDA 实现**,是所有现代大模型训练/推理栈(HuggingFace、vLLM、TGI、Megatron、DeepSpeed 等)背后的**实际注意力 kernel**。装一行 `pip install flash-attn`,就能把 attention 从朴素 PyTorch 替换成 IO-aware 版本,速度 2-5×,显存占用下降一个数量级。

## 为什么这个仓库重要

FlashAttention 不是一个"可选优化",而是**现代 Transformer 训练的默认配置**:

- HuggingFace `transformers` 里 `attn_implementation="flash_attention_2"` 就是直接调它
- Megatron-LM 的 transformer engine 里内置 FA2 后端
- vLLM 的 PagedAttention 最终也转调 FA kernel
- 训 7B+ 级别模型时,**没装 flash-attention 等于自愿放弃 2-4× 的训练速度**

## 主要组件

| 目录 | 内容 |
|------|------|
| `csrc/flash_attn/` | FA-1/2 的 CUDA kernel(C++/CUDA) |
| `hopper/` | FA-3 的 Hopper (H100/H800) 专用 kernel,用 WGMMA + TMA |
| `flash_attn/` | Python 包装层,暴露 `flash_attn_func`、`flash_attn_varlen_func` 等 |
| `csrc/layer_norm/` · `csrc/rotary/` | RMSNorm / RoPE 等同样 IO-aware 的配套算子 |
| `tests/` | 数值精度测试(与 reference implementation 对齐) |

## 常见调用入口

```python
from flash_attn import flash_attn_func

# Q, K, V: (batch, seq, heads, head_dim),dtype=bf16 或 fp16
out = flash_attn_func(q, k, v, causal=True)
```

变长序列(多 batch 拼接,no padding)用 `flash_attn_varlen_func`。GQA/MQA 时 K/V 头数少于 Q,API 自动广播。

## 何时该用

- **训练任何中大规模 Transformer**:bf16 训练默认用 FA2 而不是 PyTorch SDPA,能省 ~30-50% 训练时间
- **推理长上下文模型**:大部分推理框架都基于它,不用自己显式调用
- **自研 attention 变体**:想写一个新的 attention 变体?不建议从头写 CUDA,先 fork 这个仓库的 kernel,改 mask / weight 规则比从零写高效 1000 倍
- **用 H100/H800/MI300**:必装,FA-3 的 Hopper kernel 是目前 attention 实现的天花板

## 何时不该用

- **注意力公式本身需要改(不是 mask,而是 softmax 公式、scale 方式等)**:自定义 forward 写起来不方便,这种场景用 Triton 或 PyTorch 原生更灵活
- **小模型(< 1B)CPU-only 开发**:FA 必须要 GPU,CPU 跑不了
- **非 sm_80+ 的老卡**:FA2 要 Ampere/Hopper,V100 只能用 FA1(性能有限),更老的卡直接不支持

## 安装与排坑

- `pip install flash-attn --no-build-isolation` 是最稳定的装法
- 本地编译要 15-30 分钟,建议加大 `MAX_JOBS` 或从 release 的预编译 wheel 安装
- 对 PyTorch 版本、CUDA 版本、CCCL 版本都敏感,预编译 wheel 对齐版本最省事
- 在 Docker 里训练最好用 `nvcr.io/nvidia/pytorch` 官方镜像,里面 FA 经常预装

## 延伸阅读

- [FlashAttention 深度解读]({% post_url 2026-04-23-FlashAttention-IO感知精确注意力深度解读 %}) —— FA-1 论文的中文深度解读
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) —— warp 级并行优化
- [FlashAttention-3 (Shah et al., 2024)](https://arxiv.org/abs/2407.08608) —— Hopper WGMMA / 异步 / FP8
- [PagedAttention (vLLM, 2023)](https://arxiv.org/abs/2309.06180) —— 把 FA 的 kernel 包进推理服务
- [Making Deep Learning Go Brrrr 博客引介]({% post_url 2026-04-23-Making-Deep-Learning-Go-Brrrr-博客引介 %}) —— FA 类工作的思维起点
