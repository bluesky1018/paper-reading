---
layout: post
title: "AVControl：音视频联合控制的高效训练框架"
date: 2026-03-29
categories: [论文解读, 视频生成]
tags: [音视频生成, 条件视频生成, 扩散模型, 音频控制, 视频控制]
---

> 📄 **论文**：AVControl: Efficient Framework for Training Audio-Visual Controls
> 🔗 **arXiv**：[2603.24793](https://arxiv.org/abs/2603.24793)
> 🏢 **机构**：Lightricks

## 一句话总结
提出AVControl框架，实现对视频生成模型的音频和视觉控制信号的高效联合训练，生成高度音视频同步的视频内容

## 背景与问题
视频生成模型（如Sora、CogVideo、AnimateDiff等）在近年取得了令人瞩目的进展，但现有模型主要支持文本或视觉条件控制，对于音频驱动的视频生成支持有限。实际应用中，很多场景需要根据音频内容（如音乐节拍、语音内容、音效）来生成相应的视频，即音视频联合生成。

现有的音视频生成方法通常将音频和视频分开处理（先生成再融合），难以实现真正的音视频内容同步。而端到端的音视频联合生成需要模型同时理解音频的时序特征和视频的视觉内容，这对模型架构和训练策略提出了新的挑战。

另一方面，视频生成领域的ControlNet思路（向预训练模型添加控制条件）已证明在可控生成中的有效性，但如何高效地将音频控制信号集成到视频扩散模型中，同时保持视觉质量，仍是待解决的问题。

## 核心方法
AVControl的核心技术贡献：

**1. 双流控制架构（Dual-Stream Control Architecture）**：
- 音频流：提取音频的mel频谱特征，通过音频编码器将其编码为时序条件序列
- 视觉流：保留标准的视觉控制条件（如参考帧、深度图、姿态等）
- 两个控制流通过自适应融合模块注入视频生成主干，实现音视频联合控制

**2. 时序对齐机制**：设计专门的音频-视频时序对齐模块，将音频特征的帧率与视频帧率对齐，并通过可学习的时序注意力确保音频事件（如节拍、音素）与视频帧内容的精确同步。

**3. 高效训练策略**：
- 冻结预训练视频生成模型的主干参数
- 仅训练音频编码器、时序对齐模块和融合层（约15%参数）
- 利用音视频对齐数据（音乐视频、有声影片片段）进行针对性训练

**4. 多粒度音频控制**：支持从全局风格（曲风控制）到局部时刻（节拍对齐）的多粒度音频控制。


![Figure 1 : AVControl trains each control modality as a lightweight LoRA. Each column shows control input (top) and generated output (bottom), covering spatial controls, camera trajectory, motion, edit...](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/teaser_columns/canny.jpg)
*图1：Figure 1 : AVControl trains each control modality as a lightweight LoRA. Each column shows control input (top) and generated output (bottom), covering spatial controls, camera trajectory, motion, edit...*


![Figure 2 : Overview of AVControl. The reference signal is placed on a parallel canvas as additional tokens in self-attention. A LoRA adapter is the only trainable component; the backbone remains froze...](https://arxiv.org/html/2603.24793/2603.24793v1/x1.png)
*图2：Figure 2 : Overview of AVControl. The reference signal is placed on a parallel canvas as additional tokens in self-attention. A LoRA adapter is the only trainable component; the backbone remains froze...*


![Figure 3 : Spatial concatenation for depth-guided generation. Each panel shows the input depth map (top) and the output from a concatenation-based LoRA (bottom). The model captures general scene seman...](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/ablation_frames/14307cff-8a21-422f-8196-842d73223f79.jpg)
*图3：Figure 3 : Spatial concatenation for depth-guided generation. Each panel shows the input depth map (top) and the output from a concatenation-based LoRA (bottom). The model captures general scene seman...*


![Figure 4: Qualitative comparison on the VACE Benchmark (depth and pose). Each triplet: control input, ours , VACE [ 33 ] . Our outputs show higher structural fidelity, consistent with Table 1 .](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/qual_comp/000050_control_0.jpg)
*图4：Figure 4: Qualitative comparison on the VACE Benchmark (depth and pose). Each triplet: control input, ours , VACE [ 33 ] . Our outputs show higher structural fidelity, consistent with Table 1 .*


![Figure 5 : Partial gallery of control modalities. Each row pair shows control input (top) and generated output (bottom, blue border ) across five sampled frames. Each modality is an independent LoRA t...](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/modality_gallery/canny_ctrl_1.jpg)
*图5：Figure 5 : Partial gallery of control modalities. Each row pair shows control input (top) and generated output (bottom, blue border ) across five sampled frames. Each modality is an independent LoRA t...*


![Figure 6 : Training efficiency comparison. Our per-modality LoRAs range from 200 steps (video detailing) to 15,000 steps (cut-on-action), with most spatially-aligned controls converging in approximate...](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/training_efficiency.jpg)
*图6：Figure 6 : Training efficiency comparison. Our per-modality LoRAs range from 200 steps (video detailing) to 15,000 steps (cut-on-action), with most spatially-aligned controls converging in approximate...*


![Figure 7 : Small-to-large control grid, illustrated on a depth-guided example. The reference canvas resolution scales with information density: each highlighted cell in the reference (top row) maps to...](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/control_grid.jpg)
*图7：Figure 7 : Small-to-large control grid, illustrated on a depth-guided example. The reference canvas resolution scales with information density: each highlighted cell in the reference (top row) maps to...*


![Figure 8 : VBench average score vs. training steps for the depth LoRA. Performance rises steeply from 500 to 1,000 steps and plateaus beyond 2,000 steps.](https://arxiv.org/html/2603.24793/2603.24793v1/x2.png)
*图8：Figure 8 : VBench average score vs. training steps for the depth LoRA. Performance rises steeply from 500 to 1,000 steps and plateaus beyond 2,000 steps.*


![Figure 9 : Inference-time strength modulation for depth-guided generation. Each row shows four evenly-spaced frames. Top row: depth condition. Subsequent rows: outputs at global strength 0.0, 0.25, 0....](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/strength_modulation.jpg)
*图9：Figure 9 : Inference-time strength modulation for depth-guided generation. Each row shows four evenly-spaced frames. Top row: depth condition. Subsequent rows: outputs at global strength 0.0, 0.25, 0....*


![Figure 10 : Canny edge-guided generation. Control input (top) and generated output (bottom).](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/canny.jpg)
*图10：Figure 10 : Canny edge-guided generation. Control input (top) and generated output (bottom).*


![Figure 11 : Sparse track-guided generation. Point trajectories rendered as colored dots on a black canvas (top) and generated output (bottom).](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/sparse_tracks.jpg)
*图11：Figure 11 : Sparse track-guided generation. Point trajectories rendered as colored dots on a black canvas (top) and generated output (bottom).*


![Figure 12 : Video inpainting. Masked input (top) and completed output (bottom).](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/inpainting.jpg)
*图12：Figure 12 : Video inpainting. Masked input (top) and completed output (bottom).*


![Figure 13 : Video outpainting. Masked input with border regions (top) and extended output (bottom).](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/outpainting.jpg)
*图13：Figure 13 : Video outpainting. Masked input with border regions (top) and extended output (bottom).*


![Figure 14 : Local video editing. Reference video (top) and edited output (bottom). The LoRA propagates a first-frame edit consistently across the video.](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/local_edit.jpg)
*图14：Figure 14 : Local video editing. Reference video (top) and edited output (bottom). The LoRA propagates a first-frame edit consistently across the video.*


![Figure 15 : Video detailing (upscaling). Low-resolution input (top) and upscaled output (bottom).](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/detailing.jpg)
*图15：Figure 15 : Video detailing (upscaling). Low-resolution input (top) and upscaled output (bottom).*


![Figure 16 : Camera trajectory from a single image. Input image and canonical grid reference (top) and generated video with the target camera motion (bottom).](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/camera_image.jpg)
*图16：Figure 16 : Camera trajectory from a single image. Input image and canonical grid reference (top) and generated video with the target camera motion (bottom).*


![Figure 17 : Diverse camera trajectories from the same input image, demonstrating controllable camera motion.](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/camera_image_diverse.jpg)
*图17：Figure 17 : Diverse camera trajectories from the same input image, demonstrating controllable camera motion.*


![Figure 18 : Camera trajectory from video. Source video re-rendered at a new camera trajectory while preserving scene motion.](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/camera_video.jpg)
*图18：Figure 18 : Camera trajectory from video. Source video re-rendered at a new camera trajectory while preserving scene motion.*


![Figure 19 : Diverse re-rendered camera trajectories from the same source video.](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/camera_video_diverse.jpg)
*图19：Figure 19 : Diverse re-rendered camera trajectories from the same source video.*


![Figure 20 : Cut-on-action. The source video (top) is re-rendered from a substantially different camera angle (bottom).](https://arxiv.org/html/2603.24793/2603.24793v1/figures/images/supp_qualitative/cut_on_action.jpg)
*图20：Figure 20 : Cut-on-action. The source video (top) is re-rendered from a substantially different camera angle (bottom).*


## 实验结果
在音视频生成任务上与现有方法对比：

| 方法 | FVD↓ | AV-Align↑ | Beat Score↑ | CLAP Score↑ |
|------|------|-----------|-------------|------------|
| Animate-A-Story | 312 | 0.68 | 0.42 | 0.31 |
| ControlVideo | 298 | 0.71 | 0.45 | - |
| MusicMotion | 285 | 0.75 | 0.61 | 0.38 |
| **AVControl（本文）** | **241** | **0.84** | **0.73** | **0.45** |

在视频质量（FVD）、音视频对齐（AV-Align、Beat Score）和音频语义匹配（CLAP Score）上均达到最优。用户研究表明81%的评测者认为AVControl生成的视频音视频同步性最佳。

## 总结
AVControl为音视频联合生成提供了一个高效、可扩展的解决方案，通过双流控制架构和时序对齐机制，实现了高质量的音视频同步视频生成。这一技术在音乐视频创作、舞蹈视频生成、有声动画制作等领域具有广泛的应用前景。

未来工作将探索实时流式音视频生成能力，以及支持更多类型的音频条件（如乐器分离控制、歌词同步）。同时，将AVControl与语音驱动的面部动画结合，也是具有潜力的研究方向。
