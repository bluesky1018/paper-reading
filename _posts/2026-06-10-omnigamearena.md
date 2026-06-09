---
layout: post
title: "OmniGameArena：基于UE5的VLM游戏智能体统一基准"
date: 2026-06-10
categories: [论文解读, VLM评估]
tags: [VLM, 游戏AI, 基准测试, Unreal Engine 5, 视觉语言模型]
---

> 📄 **论文**：OmniGameArena: A Unified UE5 Benchmark for VLM Game Agents with Improvement Dynamics
> 🔗 **arXiv**：[2606.09826](https://arxiv.org/abs/2606.09826)
> 🏢 **机构**：论文作者机构

## 一句话总结

基于虚幻引擎5构建12款游戏的统一实时基准OmniGameArena，解决VLM游戏智能体评估中单次评分、仅单人游戏、缺乏统一协议等问题。

## 背景与问题

Vision-language model (VLM) agents are increasingly deployed in interactive game environments. Yet game benchmarks for VLM agents typically report a single first-attempt score per (agent, game) pair, focus on single-agent Solo play, and lack unified protocols for evaluating heterogeneous agent classes (commercial VLMs, open-weight VLMs, and specialized game policies) on the same footing. We address these gaps with OmniGameArena, a real-time benchmark of twelve newly built Unreal Engine 5 games s


![Figure 1: OmniGameArena at a glance. Twelve newly built UE5 games span Solo (7),](https://arxiv.org/html/2606.09826/2606.09826v1/x1.png)
*图：Figure 1: OmniGameArena at a glance. Twelve newly built UE5 games span Solo (7), PvP (3), and Coop (*

Foundation models are increasingly evaluated by how they act, not only by what they answer, and games are a natural stress test for this shift (Wang et al., 2023 ; Tan et al., 2024 ; Paglieri et al., 2024 ) : an agent must read a changing visual scene, choose actions under time pressure, plan across delayed rewards, and adapt when the environment resists. Game benchmarks now span text-only worlds, 2D grid suites, and 3D open environments built on existing commercial titles, and have driven rapid

## 核心方法

OmniGameArena specifies the games; the harness specifies how an agent plays them. The harness has two layers: a per-episode loop (§ 4.1 ) that drives any agent during cold-start runs, and a reflective outer loop (§ 4.2 ) whose round-level scores form the Improvement Dynamics Curve studied in § 5.3 .


![Figure 2: Radar charts of the 12 OmniGameArena games across seven capability dim](https://arxiv.org/html/2606.09826/2606.09826v1/x2.png)
*图：Figure 2: Radar charts of the 12 OmniGameArena games across seven capability dimensions. The abbrevi*


![Figure 1: OmniGameArena at a glance. Twelve newly built UE5 games span Solo (7), PvP (3), and Coop (](https://arxiv.org/html/2606.09826/2606.09826v1/x1.png)
*图1：Figure 1: OmniGameArena at a glance. Twelve newly built UE5 games span Solo (7), PvP (3), and Coop (*

![Figure 2: Radar charts of the 12 OmniGameArena games across seven capability dimensions. The abbrevi](https://arxiv.org/html/2606.09826/2606.09826v1/x2.png)
*图2：Figure 2: Radar charts of the 12 OmniGameArena games across seven capability dimensions. The abbrevi*

![Figure 3: Overview of the Improvement Dynamics Curve (IDC) harness. The experience acquisition modul](https://arxiv.org/html/2606.09826/2606.09826v1/x3.png)
*图3：Figure 3: Overview of the Improvement Dynamics Curve (IDC) harness. The experience acquisition modul*

![Figure 4: PvP win rates of Player 1 (row) against Player 2 (column) per game over all pairings.](https://arxiv.org/html/2606.09826/2606.09826v1/x10.png)
*图4：Figure 4: PvP win rates of Player 1 (row) against Player 2 (column) per game over all pairings.*

![Figure 5: IDC curves: per-round mean episode score across 10 reflection rounds for four agents on La](https://arxiv.org/html/2606.09826/2606.09826v1/x19.png)
*图5：Figure 5: IDC curves: per-round mean episode score across 10 reflection rounds for four agents on La*

![Figure 6: PvP win rates of Player 1 (row) against Player 2 (column) on MidlineClash under latency co](https://arxiv.org/html/2606.09826/2606.09826v1/x22.png)
*图6：Figure 6: PvP win rates of Player 1 (row) against Player 2 (column) on MidlineClash under latency co*


## 实验结果

We introduced OmniGameArena, a benchmark of twelve newly built UE5 real-time games spanning Solo, PvP, and Coop, and the Improvement Dynamics Curve (IDC), an agentic-reflection harness that produces multi-round self-improvement trajectories. Beyond single-round leaderboard scores, the IDC exposes two additional observables for each (agent, game) pair: how the score evolves across reflection rounds, and how the learned skill behaves on held-out task variants.


![Figure 3: Overview of the Improvement Dynamics Curve (IDC) harness. The experien](https://arxiv.org/html/2606.09826/2606.09826v1/x3.png)
*图：Figure 3: Overview of the Improvement Dynamics Curve (IDC) harness. The experience acquisition modul*


### 实验数据表格

| Game                   | Description                              | Evaluation                               |
| ---------------------- | ---------------------------------------- | ---------------------------------------- |
| \rowcolor gray!15 Solo |                                          |                                          |
| ObstacleRun3D          | A 3D parkour game where the agent naviga | , where : agent pos., : start, : target  |
| ObstacleRun2D          | A 2D side-scrolling platformer where the | , where : agent pos., : start, : target  |
| LastStand              | A platform survival game where the agent | , where : time survived, : max duration  |
| MonsterShoot           | A survival shooting game to locate and e | , where : effective damage, : total enem |
| SceneEscape            | A scene-based puzzle game requiring the  | , where : completed tasks, : total tasks |
| CueChase               | A third-person exploration game to locat | , where : activated triggers, : total tr |
| SoloCraft              | A logistics game where the agent collect | , where : fulfilled value, : target valu |
| \rowcolor gray!15 PvP  |                                          |                                          |

## 总结

OmniGameArena: A Unified UE5 Benchmark for VLM Game Agents with Improvement Dynamics 提出了一个新颖的研究框架，针对VLM评估领域的核心挑战提供了系统性解决方案。

**主要贡献：**
- 基于虚幻引擎5构建12款游戏的统一实时基准OmniGameArena，解决VLM游戏智能体评估中单次评分、仅单人游戏、缺乏统一协议等问题。
- 通过系统实验验证了方法的有效性
- 为后续研究提供了重要的基准和参考

**局限性与展望：** 未来工作可进一步探索方法在更广泛场景下的应用，以及结合更多领域知识提升系统性能。