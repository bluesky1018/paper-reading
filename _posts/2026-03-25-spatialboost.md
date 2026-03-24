---
layout: post
title: "SpatialBoost：语言引导推理增强视觉编码器空间感知能力"
date: 2026-03-25
categories: [论文解读, 视觉表示学习]
tags: ["视觉编码器", "空间推理", "3D感知", "知识注入", "LLM"]
---

> 📄 **论文**：SpatialBoost: Enhancing Visual Representation through Language-Guided Reasoning
> 🔗 **arXiv**：[2603.22057](https://arxiv.org/abs/2603.22057)
> 🏢 **机构**：Byungwoo Jeon et al. (KAIST)

## 一句话总结

SpatialBoost将2D图像中的3D空间信息转换为语言描述，通过LLM将空间知识注入预训练视觉编码器，提升其3D空间感知能力。

## 背景与问题

1 Introduction Pre-trained image representation models [ he2020momentum , donahue2019large , chen2020generative , dosovitskiy2020image , li2023mage , assran2023self ] have shown remarkable success in various downstream tasks, such as image classification [ krizhevsky2009learning , cui2018large ] , semantic segmentation [ lin2014microsoft , zhou2019semantic ] , monocular depth prediction [ silberman2012indoor , geiger2012we ] , and vision-language understanding [ antol2015vqa , hudson2019gqa ] . The core idea behind these successes is extracting transferrable representation from large-scale image datasets such as ImageNet [ deng2009imagenet ] , enabling the model to understand semantic inform


![Figure 1: Overview of SpatialBoost. We enhance spatial and geometric understanding of pre-trained vision encoders by leveraging language-guided spatia](https://arxiv.org/html/2603.22057v1/x1.png)
*图1：Figure 1: Overview of SpatialBoost. We enhance spatial and geometric understanding of pre-trained vision encoders by leveraging language-guided spatia*


现有方法存在明显局限性：缺乏系统性的评测或方法框架来解决上述问题。本文的核心动机正是填补这一空白，提出更有效的解决方案。


![Figure 2: Illustration of multi-turn visual spatial reasoning dataset, exhibiting pixel-level, object-level, and scene-level reasoning QAs. At the pix](https://arxiv.org/html/2603.22057v1/x2.png)
*图2：Figure 2: Illustration of multi-turn visual spatial reasoning dataset, exhibiting pixel-level, object-level, and scene-level reasoning QAs. At the pix*


## 核心方法

2 Related Work 2.1 Self-supervised Learning for Image Representation In earlier years, most approaches relied on supervised learning with large-scale labeled datasets to train models [ deng2009imagenet , simonyan2014very , szegedy2014going , he2016deep ] . However, the dependence on annotated data introduced scalability challenges due to label expense. To address this, self-supervised learning (SSL) has emerged as a dominant paradigm, leveraging unlabeled data to learn image representations. Contrastive learning methods, including SimCLRv2 [ chen2020big ] , MoCov3 [ chen2021empirical ] , DINOv2 [ oquab2023dinov2 ] , and iBOT [ zhou2021ibot ] , are trained to distinguish between representations of augmented views of the same image and those of different images. Concurrently, mask prediction


![Figure 3: Illustration of the dual-channel attention layer [hong2022cogvideo], where an additional attention block is introduced alongside the origina](https://arxiv.org/html/2603.22057v1/x3.png)
*图3：Figure 3: Illustration of the dual-channel attention layer [hong2022cogvideo], where an additional attention block is introduced alongside the origina*



![Table 1: Results on monocular depth estimation from NYUd [silberman2012indoor] and KITTI [geiger2013vision] benchmarks. We report the RMSE score betwe](https://arxiv.org/html/2603.22057v1/x4.png)
*图4：Table 1: Results on monocular depth estimation from NYUd [silberman2012indoor] and KITTI [geiger2013vision] benchmarks. We report the RMSE score betwe*



![Table 2: Results on semantic segmentation from ADE20K [zhou2017scene] and Pascal VOC [Everingham10] benchmarks. We report mIoU score. Higher is better](https://arxiv.org/html/2603.22057v1/x5.png)
*图5：Table 2: Results on semantic segmentation from ADE20K [zhou2017scene] and Pascal VOC [Everingham10] benchmarks. We report mIoU score. Higher is better*


## 实验结果

4 Experiments Through extensive experiments, we validate the performance of SpatialBoost and ablate its key components, focusing on following questions: • Can SpatialBoost improve spatial knowledge of the vision encoder? ( Sections ˜ 3.1 , 3.1 , 4 and 3 ) • Isn’t SpatialBoost overfitted to spatial knowledge? ( Table ˜ 5 ) • Which components contribute to SpatialBoost performance? (Tables 6 to 7 and Figure ˜ 6 ) 4.1 Experimental Setup VQA Dataset Construction. For single-view image, we use randomly sampled 100K images from the SA1B dataset [ kirillov2023segment ] to construct the single-view VQA dataset specialized in chain-of-thought spatial reasoning. For multi-view images, we use filtered 200K samples from the ego-centric video dataset [ grauman2022ego4d ] and 3D dataset [ jensen2014dtu 


![Table 3: Results on 3D-centric tasks. We evaluate unified probing on diverse 3D-related tasks from ScanNet [dai2017scannet] scenes. We report BLEU-1 s](https://arxiv.org/html/2603.22057v1/x6.png)
*图6：Table 3: Results on 3D-centric tasks. We evaluate unified probing on diverse 3D-related tasks from ScanNet [dai2017scannet] scenes. We report BLEU-1 s*



![Table 4: Results on vision-based robot learning. We report the performance of imitation learning agents on 4 domains from CortexBench [majumdar2023we]](https://arxiv.org/html/2603.22057v1/x7.png)
*图7：Table 4: Results on vision-based robot learning. We report the performance of imitation learning agents on 4 domains from CortexBench [majumdar2023we]*


## 总结

5 Conclusion In this paper, we have presented SpatialBoost, a framework to enhance the vision encoders by leveraging linguistic expressions of geometric and semantic information within images. SpatialBoost uses LLM and dual-channel attention layers to exploit linguistic information into image representations, generates a multi-turn visual spatial reasoning dataset, and leverages them to improve the image representations. Our experiments show that SpatialBoost consistently enhances the vision encoders on various downstream tasks that require a spatial understanding of images. We hope that our w

本文工作的主要贡献包括：（1）SpatialBoost将2D图像中的3D空间信息转换为语言描述，通过LLM将空间知识注入预训练视觉编码器，提升其3D空间感知能力。；（2）通过充分的实验验证了方法的有效性。未来工作可在此基础上进一步探索更大规模、更多样化场景下的应用与扩展。

> 🔗 论文链接：[https://arxiv.org/abs/2603.22057](https://arxiv.org/abs/2603.22057)
