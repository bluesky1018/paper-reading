---
layout: post
title: "生成式原生音视频对齐"
date: 2026-05-30
categories: [论文解读, 多模态生成]
tags: ["音视频生成", "多模态", "对齐", "扩散模型", "联合生成"]
---

> 📄 **论文**：Native Audio-Visual Alignment for Generation
> 🔗 **arXiv**：[2605.30073](https://arxiv.org/abs/2605.30073)
> 🏢 **机构**：

## 一句话总结

Joint audio-video generation aims to synthesize temporally synchronized and semantically coherent visual-acoustic content. However, existing open-source methods mainly rely on either dual-tower design...

## 背景与问题

Audio-visual generation has made rapid progress in recent years. Compared with cascaded pipelines that synthesize one modality after another, joint audio-video generation models temporal and semantic correspondences within a unified generation process, thereby reducing error propagation and improving cross-modal coherence. Although commercial systems such as Seedance [ 19 ] , Kling [ 14 ] , and Veo [ 10 ] have demonstrated the potential of joint audio-video synthesis, their architectures and training recipes remain proprietary. Therefore, recent open-source efforts, including Ovi [ 16 ] , LTX [ 12 ] , and MoVA [ 20 ] , have become crucial for reproducible research in audio-visual generation.

Despite this progress, most open-source methods still adopt a dual-tower architecture, where audio and video are generated in separate streams, and cross-modal interaction is introduced through additional alignment modules. As illustrated in Fig. 1 (a), the paradigm conditions audio and video on textual context in separate feature spaces, and establishes audio-visual correspondence only through late-stage interaction. However, such posterior alignment weakens the joint evolution of audio and video during generation, making fine-grained synchronization and semantic consistency dependent on auxiliary cross-modal modules rather than a unified generative representation.

More recently, daVinci-MagiHuman [ 5 ] moves beyond dual-tower interaction by placing textual context, video, and audio to


![Figure 1: Comparison of different audio-visual generation paradigms. (a) Dual-Tower : Separate audio](https://arxiv.org/html/2605.30073/2605.30073v1/figs/teaser.png)
*图：Figure 1: Comparison of different audio-visual generation paradigms. (a) Dual-Tower : Separate audio*


![Figure 2: Overview of NAVA. NAVA adopts an Align-then-Fuse MMDiT architecture, which first establish](https://arxiv.org/html/2605.30073/2605.30073v1/figs/arch.png)
*图：Figure 2: Overview of NAVA. NAVA adopts an Align-then-Fuse MMDiT architecture, which first establish*


Video-to-audio generation synthesizes acoustic content conditioned on a given video, and often serves as a cascaded component for audio-visual content creation. Early methods explore multimodal representation learning and cross-modal conditioning, using Transformer architectures or visual-textual encoders to fuse video and text cues [ 1 ; 13 ] . Recent systems improve temporal precision and generation efficiency through high-frame-rate visual features, rectified flow matching, and large-scale audio-visual training [ 22 ; 27 ] . More recent works such as MMAudio [ 4 ] and Kling-Foley [ 26 ] adopt diffusion or MMDiT-style architectures and leverage large-scale video-audio corpora such as VGGSound [ 3 ] and WavCaps [ 17 ] . Although these approaches can generate plausible audio for existing v

## 核心方法

Let h a h_{a} , h v h_{v} , and c c denote audio tokens, video tokens, and context tokens, respectively. The context c c mainly contains textual conditions and can be augmented with control signals such as reference timbre embeddings. We use this notation to abstract how different audio-visual generation paradigms organize audio, video, and context interactions during denoising.

Existing dual-tower methods [ 16 ; 12 ; 20 ] maintain separate audio and video generation streams and condition each modality independently:

Audio-visual correspondence is then introduced through additional cross-modal interaction modules:

This posterior alignment paradigm allows each modality to evolve largely in its own feature space before cross-modal correspondence is explicitly established, making fine-grained synchronization dependent on late-stage interaction.


![Figure 3: Qualitative visualization of NAVA. We present various generated video frames, audio wavefo](https://arxiv.org/html/2605.30073/2605.30073v1/x2.png)
*图：Figure 3: Qualitative visualization of NAVA. We present various generated video frames, audio wavefo*


![Figure 4: Results of User study. Pairwise human preference comparisons between NAVA and representati](https://arxiv.org/html/2605.30073/2605.30073v1/figs/sub_comp.png)
*图：Figure 4: Results of User study. Pairwise human preference comparisons between NAVA and representati*


## 实验结果

NAVA has 6.3B parameters with 30 MMDiT blocks, where the first 10 blocks are Hierarchical Alignment Layers and the remaining 20 are Unified Fusion Layers . We initialize corresponding layers from Wan2.2-5B [ 23 ] , use Wan2.2-VAE for video latents with a 4 × 16 × 16 4\times 16\times 16 compression ratio, and use LTX2.3-VAE for multi-channel audio latents. The model is trained with AdamW at a learning rate of 5 × 10 − 5 5\times 10^{-5} on 128 NVIDIA H100 GPUs, with an effective batch size of 512 for 70K steps following the three-stage schedule in Sec. 2.4 . We apply random cross-modality attention masking and timbre-condition dropout with probabilities of 20 % 20\% each, and sample image conditions with probability 50 % 50\% .

## 全文图示

## 总结

We presented NAVA , a Native Audio-Visual Alignment framework for joint audio-video generation. NAVA decouples audio-visual synchronization from context conditioning by establishing audio-video correspondence in a dedicated alignment space and using context as external guidance. We instantiate this formulation with an Align-then-Fuse MMDiT architecture, which bridges modality-aware alignment and unified audio-video denoising, and introduce Timbre-in-Context Conditioning for segment-level reference-timbre control. Experiments on Verse-Bench and Seed-TTS demonstrate that NAVA achieves strong audio-visual synchronization, visual quality, semantic consistency, and timbre controllability. These results indicate that native audio-visual alignment with decoupled context conditioning is a promising direction for scalable and controllable audio-video generation.

Despite its strong overall performance, NAVA remains limited in generating certain long-tail and highly compositional audio events, s

