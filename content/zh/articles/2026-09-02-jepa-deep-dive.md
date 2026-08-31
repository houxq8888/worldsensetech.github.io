---
title: "JEPA 深度解读：从 I-JEPA 到 V-JEPA 2-AC，预测性表征如何走向世界模型"
slug: "2026-09-02-jepa-deep-dive"
date: 2026-09-02
draft: false
categories: ["世界模型", "论文解读"]
tags: ["JEPA", "I-JEPA", "V-JEPA", "V-JEPA 2", "V-JEPA 2-AC", "AMI Labs", "LeCun", "预测性表征学习", "自监督学习", "世界模型", "具身智能"]
description: "从 2022 年的理论蓝图到 I-JEPA、V-JEPA，再到 V-JEPA 2 的视频预测、动作条件预测与机器人规划演示，LeCun 的 JEPA 路线逐步从预测性表征学习走向世界模型研究。这篇文章逐篇拆解这条技术路线，并讨论它与 Dreamer/RSSM、生成式世界模型之间的真正区别。"
toc: true
related_articles:
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - world-model-intro
  - 2026-08-25-dreamer-explained
  - world-model-transformer
---

在[上一篇世界模型盘点](/zh/articles/2026-09-01-world-model-h2-review/)里，我花了大量篇幅讨论 Cosmos、Genie 3、Marble 这些项目，但有一个方向我只做了概述——JEPA。

这不是因为 JEPA 不重要。恰恰相反，**JEPA 可能是当前世界模型方向最有理论深度的一条路线**。它背后站着 Yann LeCun，有从 I-JEPA 到 V-JEPA 2-AC 的完整技术演进，有 AMI Labs 10.3 亿美元的融资验证，而且它代表了一种和以像素生成为主要训练目标的世界模型截然不同的技术思路。

这篇文章把 JEPA 系列从第一篇到最新一篇做一个完整的技术拆解。

## 一、JEPA 的核心思想：一句话版本

**不要预测像素，预测表征。**

这句话与以像素或视觉观测生成为主要训练目标的一类世界模型形成了鲜明对比。JEPA 并不是认为生成像素没有价值，而是认为：如果目标是学习可用于预测、理解和决策的抽象世界表征，那么要求模型重建所有观测细节未必是最优的学习目标。

生成式世界模型的逻辑是：给定历史观测，预测未来像素。JEPA 的逻辑是：给定历史观测，预测未来观测的**抽象表征**——在表征空间做预测，而不是在像素空间做预测。

为什么这个区别很重要？因为像素空间包含大量与任务无关的细节：光照变化、纹理细节、随机噪声。要求模型精确预测这些细节，不仅浪费模型容量，还会引入大量无用的梯度信号，拖慢学习速度。

LeCun 在 2022 年的[位置论文](https://openreview.net/forum?id=BZ5a1r-kVsf)里把这个问题说得很清楚：**对于学习高层语义和世界状态表示而言，要求模型精确预测所有像素并不是理想的学习目标。**

这不是说像素级预测没有价值——视频生成确实很酷。JEPA 的论点是：如果你的目标是学习对下游任务有用的世界表征，那么在表征空间做预测是更高效的学习策略。

但这个策略也留下了一个贯穿全文的核心问题：**JEPA 主动过滤掉的那些"不可预测的像素细节"，会不会恰好是机器人控制所需要的信息？** 这个问题在第九节的开放讨论中会回到。

## 二、I-JEPA：在图像上证明这条路走得通（2023）

### 论文信息

*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*，Meta AI，CVPR 2023。[论文](https://arxiv.org/abs/2301.08243)，[代码](https://github.com/facebookresearch/ijepa)。

### 架构

I-JEPA 的架构有三个核心组件：

**Context Encoder（上下文编码器）**：一个 Vision Transformer，接收可见区域的图像 patch，输出对应的表征。

**Target Encoder（目标编码器）**：架构和 Context Encoder 相同，但参数通过指数移动平均（EMA）更新——不直接参与梯度计算，而是缓慢跟踪 Context Encoder 的参数变化。它负责把被遮蔽区域编码成目标表征。

**Predictor（预测器）**：一个比编码器轻量得多的 Transformer（embedding dimension 384 vs 编码器的更大维度），接收 Context Encoder 的可见表征，预测被遮蔽区域在 Target Encoder 表征空间中的表示。

### 遮蔽策略

这是 I-JEPA 和 MAE（Masked Autoencoder）的关键区别。

MAE 在输入像素层面做遮蔽——随机遮掉 75% 的 patch，然后让模型重建这些像素。

I-JEPA 的遮蔽策略与 MAE 有相似的"遮挡—预测"结构，但关键区别不在于"在哪一层 mask"，而在于**预测目标是什么**。具体来说：I-JEPA 采用 multi-block 策略，选择 4 个目标区域（各占 15%-20%）和 1 个上下文区域（占 85%-100%），去掉重叠部分。Context Encoder 只接收上下文区域的 patch，Target Encoder 对目标区域的 patch 进行编码，Predictor 则根据上下文表征预测目标区域的表征。最终优化的是 representation-space prediction loss，而不是像素重建损失。

这意味着：模型不需要从像素级别重建被遮掉的内容，而是需要从可见区域的表征中**推断**被遮区域在抽象表征空间中应该是什么样子。

### 关键结果

I-JEPA 用 ViT-H/14 在 ImageNet 上的结果：

- **Linear probing**：79.3% top-1 准确率（vs MAE 的 77.2%）
- **448 分辨率**：81.1% top-1
- **1% low-shot**：73.3%（vs MAE 的 59.8%）——这个差距非常大
- **Full fine-tuning**：87.1%（300 epochs），而 MAE 的对应结果是 1600 epochs、87.8%

最后一点特别值得注意：在接近的最终性能下，I-JEPA 所需 fine-tuning epoch 数约少了 **5.3 倍**。这说明表征预测目标具有很强的训练效率潜力，但不能简单等同于端到端训练时间提升 5.3 倍——不同方法的 batch size、数据增强、GPU 利用率等都可能不同。

### 为什么重要

I-JEPA 的意义不在于它刷了什么 SOTA，而在于它用一个干净的实验证明了一件事：**不生成像素、只在表征空间做预测的自监督学习，不仅能 work，而且在学习视觉表征的任务中表现出很强的训练效率。**

这是 JEPA 路线的第一块基石。

## 三、V-JEPA：从图像到视频（2024）

### 论文信息

*V-JEPA: Latent Video Prediction for Visual Representation Learning*，Meta AI，ICLR 2024。作者：Quentin Garrido 等。

### 核心扩展

如果说 I-JEPA 证明了"在表征空间预测图像区域可行"，V-JEPA 要进一步回答的是：**能不能在表征空间预测视频中被遮蔽的时空区域，并由此学习动态相关的视觉表征？**

这比图像难很多。图像中被遮蔽的区域是空间上固定的，模型只需要理解空间结构。但视频中的时空预测涉及运动模式、时间依赖和场景演化，因此相比图像上的空间预测，需要建模更加复杂的时序结构。

V-JEPA 的做法是：把 I-JEPA 的框架扩展到时空维度。Context Encoder 接收历史视频帧的时空 patch，Predictor 需要在表征空间中预测被遮蔽时空区域的抽象表征。

### 技术要点

V-JEPA 保持了 JEPA 的核心设计哲学：

- Target Encoder 仍然用 EMA 更新
- 预测仍然在表征空间进行，不回到像素空间
- 遮蔽策略从空间遮蔽扩展到时空遮蔽——遮蔽部分时空区域

### 关键贡献

V-JEPA 的核心贡献是证明了 JEPA 框架可以自然地扩展到视频领域，而且学到的表征在动作识别、视频理解等下游任务上表现良好。在 frozen representation 评估下，V-JEPA 的 ViT-H/16 模型在 Kinetics-400、Something-Something-v2 和 ImageNet-1K 上分别取得 **81.9%、72.2% 和 77.9%**。

从路线演进来看，V-JEPA 更像是 JEPA 从图像表征学习走向视频动态建模的关键过渡：它证明了 latent video prediction 可以学习强大的时空表征。真正把这条路线进一步推进到"action-conditioned prediction + planning"的，则是 V-JEPA 2。

## 四、V-JEPA 2：从表征学习到世界模型（2025）

### 论文信息

*V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction, and Planning*，Meta AI，2025 年 6 月。[论文](https://arxiv.org/abs/2506.09985)。

### 这是最重要的一篇

如果说 I-JEPA 和 V-JEPA 是基础研究，V-JEPA 2 是 JEPA 路线第一次比较完整地展示 action-conditioned prediction 可以服务于真实机器人 planning。

### 架构升级

V-JEPA 2 的架构有重大升级：

**编码器**：Vision Transformer，提供多种规模——ViT-L（约 3 亿参数）、ViT-H（约 6 亿参数）、ViT-g（约 **10 亿参数**）。视频输入被切分为 2x16x16 的 tubelets（2 帧 × 16 × 16 像素）。位置编码使用 **3D-RoPE**（三维旋转位置编码），这是处理时空序列的关键设计。

**Action-conditioned 变体（V-JEPA 2-AC）**：这是 V-JEPA 2 最重要的新增。一个约 **3 亿参数的 predictor Transformer**，使用 **block-causal attention**，接收动作序列作为条件输入来预测未来表征。需要注意的是，**预测器的输入不只是 video representation 和 action——在机器人实验中还包括 end-effector state（机器人末端执行器状态）**。论文原文明确写的是"conditioned on past video frames, actions, and end-effector states"。因此 V-JEPA 2-AC 的实际预测形式更接近 z_{t+1} = f_θ(z_{≤t}, a_{≤t}, s_{≤t})，其中 s_t 是机器人末端状态，而不是一个简单的 Markov p(z_{t+1}|z_t, a_t)。从抽象层面看，它可以被理解为 action-conditioned latent transition model；但具体实现是一个带历史上下文、动作和末端状态的 autoregressive predictor。

这意味着：模型不仅能"看懂"视频，还能"想象"如果执行某个动作，世界会变成什么样。这是迈向世界模型能力的关键一步。

### 预训练

预训练规模很大：

- 使用 **VideoMix22M** 数据构建方案，规模约 **2200 万个视频/图像样本**；论文描述其数据来源覆盖超过百万小时级别的互联网视频
- **25.2 万次迭代**
- 采用渐进式分辨率策略（从低分辨率开始，逐步提高）
- 使用 mask-denoising feature prediction 目标

### 关键结果

**视频理解**：

V-JEPA 2 在多个视频理解 benchmark 上分别报告了结果：Something-Something-v2 上 77.3% top-1，Diving48 上 90.2%；在视频 QA 和物理世界理解任务上，经与 8B 语言模型对齐后，在 PerceptionTest（84.0）、TempCompass（76.9）等 benchmark 上也取得很强结果。超过 InternVideo2 和 DINOv2。

**机器人操作（这是最让我兴奋的部分）**：

V-JEPA 2-AC 在预训练的 V-JEPA 2 基础上，使用约 **62 小时机器人轨迹数据进行 action-conditioned post-training**，随后在未针对具体任务进行训练或环境适配的情况下测试机器人操作能力。这里的"zero-shot"应理解为**任务/环境层面的 zero-shot generalization**，而不是完全不使用机器人交互数据。

论文报告的具体任务结果如下：

| 任务 | V-JEPA 2-AC | Cosmos | Octo |
|---|---|---|---|
| Reach | 100% | 80% | — |
| Grasp cup | 60% | 0% | — |
| Grasp box | 20% | 20% | — |
| Pick-and-place cup | 80% | 0% | — |
| Pick-and-place box | 50% | 0% | 0% |

目标到达精度**小于 4 厘米**。在论文给出的同一 RTX 4090、CEM planning 设置下，Cosmos baseline 每个 action 的规划时间约 4 分钟，而 V-JEPA 2-AC 即使使用 **10× 更多候选 samples**（800 vs 80），规划时间也仅约 16 秒，相当于约 **15× 的时间优势**。

这个对比非常有说服力。V-JEPA 2-AC 在 reach task 上与 Cosmos 表现接近（100% vs 80%），但在 object interaction 任务上优势明显——Cosmos 在 cup/box 的 pick-and-place 上成功率均为 0%，而 V-JEPA 2-AC 分别达到 80% 和 50%。

### 为什么 V-JEPA 2 是 JEPA 路线的转折点

V-JEPA 2-AC 第一次展示了：

1. JEPA 框架可以扩展到十亿参数级别
2. Action-conditioned 预测可以服务于真实的机器人控制
3. 在表征空间做预测不仅理论上优雅，工程上也比像素级预测更高效
4. 在少量机器人数据 post-training 后，即可在多个操作任务上 zero-shot 泛化——不需要为每个具体任务重新训练

这正是 LeCun 在 2022 年论文里画的蓝图：**用预测性表征学习构建能理解物理世界的 AI 系统。** 三年后，V-JEPA 2-AC 把这个蓝图变成了可运行的系统。

### V-JEPA 2-AC 真正解决了什么？

值得把 V-JEPA 2-AC 的能力拆解成三层来看：

**第一层：representation。** 把视觉 observation 压缩成任务相关的 latent representation——过滤掉不可预测的像素细节，保留对下游有用的抽象信息。

**第二层：prediction。** 给定 action 序列，在表征空间中预测未来 latent state——不是预测像素，而是预测"世界的任务相关状态会怎样变化"。

**第三层：planning。** 对多个 candidate actions 做 latent rollout，选择预期回报最高的 action——这就是 CEM planning 在论文中做的事。

把这三层连起来看：

```
observation
    ↓
representation（编码当前状态）
    ↓
action-conditioned dynamics（预测 action 后果）
    ↓
latent rollout（想象多条未来轨迹）
    ↓
planning（选择最优 action）
    ↓
action
```

这其实和 Dreamer 的核心闭环已经非常接近。但起点不同：Dreamer 的 latent dynamics 从设计之初就是 model-based RL 的核心闭环；V-JEPA 2-AC 则是从大规模视觉表征模型出发，再通过机器人轨迹数据获得 action-conditioned prediction 能力。一条路是"从控制出发，学到好的表征"，另一条是"从表征出发，获得控制能力"。

### 旁支：V-JEPA 2.1（2026）

2026 年 3 月，Meta 又发布了 V-JEPA 2.1。与 V-JEPA 2-AC 的重点不同，V-JEPA 2.1 主要推进的是 dense visual representation：通过 dense predictive loss、deep self-supervision 和 multimodal tokenizers，使表征在空间和时间维度上更加细粒度和一致。论文同时报告了 Ego4D short-term object interaction anticipation（7.71 mAP）、EPIC-KITCHENS anticipation（40.8 R@5）、Something-Something-v2（77.7）、导航、深度估计以及真实机器人抓取上的进一步结果——机器人抓取相比 V-JEPA 2-AC 提升约 20 个百分点。

值得注意的是，V-JEPA 2.1 与 V-JEPA 2-AC 并不是简单的"下一版 action-conditioned world model"，而更像是 JEPA representation backbone 的继续演进——在表征质量和空间细粒度上做文章，而不是直接推进 action-conditioned dynamics。因此，本文仍以 V-JEPA 2-AC 作为"JEPA 走向 action-conditioned world model"的关键节点来讨论，不把 V-JEPA 2.1 混入这条控制路线。JEPA 路线仍在继续演进，V-JEPA 2.1 证明了这个框架的生命力。

## 五、AMI Labs：产业化观察（2026）

[AMI Labs](https://amilabs.xyz/)（Advanced Machine Intelligence Labs）于 2026 年 3 月正式公开亮相，宣布完成约 **10.3 亿美元**（约 8.9 亿欧元）的种子轮融资，pre-money 估值约 **35 亿美元**（据公司官方公告及 TechCrunch、Reuters 等报道）。这是近年来欧洲 AI 基础模型领域最大的种子轮，由 Cathay Innovation、Greycroft、Hiro Capital、HV Capital 等联合领投，NVIDIA、Temasek、Samsung 等机构及 Jeff Bezos、Eric Schmidt 等个人投资者参与。

核心团队包括 Yann LeCun（Executive Chairman）、Alex LeBrun（CEO）、Saining Xie（Chief Science Officer）、Michael Rabbat（VP of World Models）、Pascale Fung（Chief Research and Innovation Officer）。

AMI Labs 的官方技术方向是：构建能理解物理环境、保持长期信息、执行逻辑规划的世界模型——专注于从多模态传感器输入中学习抽象表征，过滤掉不可预测的细节，在概念空间中预测结果。特别值得注意的是，AMI Labs 明确强调 **action-conditioned world models**。

从公开人员背景和技术愿景来看，可以合理推测 JEPA 类 predictive representation learning 会是 AMI Labs 的重要研究方向之一；但这属于基于公开信息的推断，而不是 AMI 已经正式公布的架构结论。

## 六、JEPA vs Dreamer/RSSM：都在隐空间预测，区别在哪？

如果你读过我的 [RSSM 详解](/zh/articles/rssm-deep-dive/)，你会发现 JEPA 和 RSSM/Dreamer 有一个明显的相似点：**都在隐空间做预测，而不是直接预测像素。**

但它们不是同一个东西。

### 相似之处

两者都认识到像素空间不是好的预测目标。RSSM 用 encoder 把观测压缩到隐状态，然后用 dynamics model 在隐空间预测下一步；JEPA 用 encoder 把观测编码到表征空间，然后用 predictor 预测未来表征。

### 核心区别

**训练目标不同。** RSSM/Dreamer 的隐状态动力学模型最终要服务于 RL——它需要一个明确的 state → action → next state 结构来支持 planning 和 control。JEPA 的训练目标是自监督表征学习——它不显式建模 state 和 action 的分离，而是通过 masking 和 prediction 来学习有用的表征。

**Action 的角色不同。** 在 Dreamer 中，action 是 dynamics model 的显式输入——state transition 直接依赖于 action。在原始 I-JEPA 和 V-JEPA 中，没有 action 的概念。直到 V-JEPA 2 才引入了 action-conditioned 变体，但它的 action  conditioning 方式和 Dreamer 的 RSSM 也不同——V-JEPA 2 用的是 block-causal attention 把动作序列和视觉表征一起处理。

**应用场景不同。** Dreamer 的核心场景是 RL 和控制——想象未来、评估 action、选择最优策略。JEPA 的核心场景是表征学习和理解——学习对下游任务有用的世界表征。V-JEPA 2 的机器人实验展示了 JEPA 也能做控制，但方式和 Dreamer 不同。

### 一句话总结

Dreamer 是"用隐状态动力学模型做 planning"，JEPA 是"用预测性表征学习做理解"。两者都聪明，但解决的是不同的问题。

用更形式化的方式看：Dreamer/RSSM 学习的是 action-conditioned transition model p(z_{t+1} | z_t, a_t)，核心是隐状态动力学；JEPA 学习的是预测器 f_θ，在机器人实验中优化目标大致为 L = ||f_θ(z_{≤t}, a_{≤t}, s_{≤t}) − sg(z_{t+k}^{target})||，其中 sg 表示 stop-gradient，prediction target 是 target encoder 输出的表征而不是原始观测，s_t 为 end-effector state 等机器人本体状态。前者从设计之初就服务于 model-based RL 的 rollout 和 planning；后者的核心贡献在于"预测什么"而不是"怎么 rollout"。

还需要注意一个结构性的差异：Dreamer/RSSM 的 latent state 从设计上就是 s_t = (h_t, z_t)——deterministic recurrent state 加 stochastic latent state，共同构成 dynamics model 和 imagination 的状态变量。而 V-JEPA 的 representation 首先是为视觉理解和预测学习得到的高维 patch-level visual representation，它不是天然定义好的 Markov latent state。V-JEPA 2-AC 是在此基础上证明这些 representation 可以进一步承载 action-conditioned dynamics，而不是一开始就假设它们是完整的 Markov state。这也是为什么 V-JEPA 2-AC 的 predictor 需要在 feature map / token level 上做未来表征预测。

## 七、JEPA 在世界模型图谱中的位置

回到我在[盘点文章](/zh/articles/2026-09-01-world-model-h2-review/)里的四条技术路线：

- **Latent Dynamics**（Dreamer/RSSM）：state → action → next state
- **Generative Video**（Cosmos）：condition → future video frames
- **Interactive**（Genie 3）：state + action → interactive future
- **Spatial/3D**（Marble）：persistent spatial representation

JEPA 不完全属于其中任何一类。它最接近 Latent Dynamics，因为它也在隐空间做预测。但它不像 Dreamer 那样有显式的 state transition 结构，也不以 RL 和 planning 为主要目标。

如果非要归类，我会说 **JEPA 代表的是第五条路线：Predictive Representation Learning**（这里的"第五条路线"是本文为了分析方便提出的分类，而不是现有文献统一采用的 taxonomy）。它的核心贡献不是"怎么建模世界动态"，而是"用什么目标函数来学习世界表征"。不过更准确地说，这五个维度并不是严格互斥的类别——未来完全可以出现 JEPA representation + RSSM dynamics + language conditioning + 3D spatial memory 的组合。因此 JEPA 更像是一个与 world-model architecture **正交**的 representation/prediction paradigm，而不仅仅是第五种 world model。

更准确地说，JEPA 是一类 predictive representation learning architecture / objective family，而不仅仅是一种具体的 world-model architecture。从更高层的视角看，它体现了一种"先学习可预测、可决策的抽象状态，而不是重建全部观测细节"的建模思路。

### 什么时候 JEPA 才算世界模型？

这个问题值得单独拿出来讨论，因为它直接影响我们怎么理解整条技术路线。

**I-JEPA** 本身更适合被称为 predictive representation learning，而不是完整的 world model。它的目标是学习对下游任务有用的图像表征，没有时序动态建模，也没有 action 的概念。

**V-JEPA** 引入了 temporal prediction，但主要目标仍然是学习视频表征——它预测未来帧的表征，但不支持 action-conditioned 的规划。

**V-JEPA 2-AC** 则进一步加入了 action-conditioned prediction，并通过 model-based planning 在机器人上预测动作后果。如果采用"世界模型需要能够预测 action 后果并支持 planning"这一较宽松的定义，那么 V-JEPA 2-AC 已经具备了 world model 的关键能力；但它与 Dreamer/RSSM 那种显式 latent state transition model 仍然存在结构上的差异。

因此，如果要严格区分，我会把 I-JEPA / V-JEPA 称为"JEPA representation learning"，把 V-JEPA 2-AC 称为在宽松定义下的"JEPA-based latent world model"。这个区分很重要——它让我们更精确地理解 JEPA 路线中哪些部分是表征学习，哪些部分是世界建模。

### JEPA 路线技术总览

把上面几节的内容拉到一起看，JEPA 路线的演进并不是一个简单的线性链条"LeCun → JEPA → I-JEPA → V-JEPA → V-JEPA 2 → 机器人 → AMI Labs"。更准确的结构是一棵逐步分叉的树：

```
JEPA（2022 理论蓝图）
│
├── I-JEPA（2023）
│     └── 图像表征预测：证明"预测表征 ≠ 像素重建"可行
│
├── V-JEPA（2024）
│     └── 视频表征预测：从空间走向时空
│
└── V-JEPA 2（2025）
      ├── 视频理解（88.2 avg / 6 benchmarks）
      ├── latent video prediction
      ├── action-conditioned prediction（V-JEPA 2-AC）
      └── 机器人规划演示（zero-shot task evaluation）
              │
              └── 仍然开放：
                   ├── 长程 rollout 稳定性
                   ├── 表征充分性（state sufficiency）
                   ├── 鲁棒控制
                   └── 因果/反事实有效性

AMI Labs（2025 年底创办）
└── 产业化研究方向
      └── 与 JEPA 理念高度相关，但具体架构尚未公开
```

这个结构有两个值得注意的地方。第一，V-JEPA 2 并不是一个单一成果——它同时包含视频理解、latent prediction、action-conditioned prediction 和机器人规划四个层面的贡献，前三个是视频/表征层面的，只有第四个直接涉及控制。第二，AMI Labs 和 V-JEPA 2 之间是"技术理念连续"而非"架构继承"——把 AMI Labs 简单等同于"V-JEPA 2 的公司化版本"在技术上不严谨。

### 预测得好 ≠ 世界状态完整

JEPA 路线最值得讨论的问题之一，是"可预测表征"与"完整世界状态"之间到底是什么关系。

一个表征可以非常适合视频理解或动作识别，却未必保留规划所需的全部信息。理想的 world state 至少应该满足：从当前状态 z_t 和动作 a_t 出发，能够足够准确地预测未来相关状态，并支持稳定的多步 rollout 用于 planning。但 JEPA 的表征主动丢弃了"不可预测"的细节——问题在于，**什么信息被模型判定为"不可预测/无关"？** 如果精确几何、接触状态、摩擦、细小物体状态、affordance 被当作 nuisance information 丢掉，那么表征对分类很好，却可能对 control 不够充分。

这其实是上面"表征充分性"问题的更深一层：**representation 很好 ≠ state 很完整。** 一个对视频预测非常好的表征，不一定是一个对 downstream decision 足够充分的 state representation。如何保证在过滤不可预测细节的同时，不会丢掉未来决策真正需要的信息，是 predictive world model 的核心设计问题。

## 八、对从业者的意义

### 如果你在做机器人控制

V-JEPA 2-AC 的操作结果值得认真关注。在论文给出的同一 RTX 4090 + CEM planning 实验设置中，V-JEPA 2-AC 的规划耗时约 16 秒，而 Cosmos baseline 约 4 分钟；即使 V-JEPA 2-AC 使用 10× 更多 candidate trajectories（800 vs 80），仍保持约 15× 的 planning latency 优势。这个结果说明 latent-space planning **在该实验设置下**具有显著的计算优势，但不能直接外推为"所有 JEPA 系统都比生成式 world model 快 15×"——不同的 planning horizon、candidate 数量和 hardware 配置都会影响实际延迟。此外，**16 秒/action 本身距离实时机器人控制仍有明显距离**。如果你的系统可以容忍一定的规划延迟而非生成漂亮的视频，JEPA 路线可能更适合。

### 如果你在做自监督学习

I-JEPA 的训练效率表现（5.3× fewer epochs 达到与 MAE 接近的性能）已经足够引起注意。JEPA 框架提供了一个不依赖像素重建的自监督学习范式，这在计算成本越来越受关注的今天特别有价值。

### 如果你在做世界模型研究

JEPA 路线提供了一个重要的 alternative：不是所有世界模型都需要生成像素。如果你的下游任务需要的是理解而非生成，在表征空间做预测可能是更好的选择。

### 如果你在关注 AMI Labs

AMI Labs 的团队和融资规模说明了一件事：投资人对"world models + physical intelligence"这一大方向具有很强的兴趣。但也要清醒——**这笔融资不能直接解读为对 JEPA 具体技术路线有效性的验证**。AMI Labs 公开表述的是 world models、persistent memory、reasoning、planning 和 action-conditioned intelligence，而不是某个已经确定的 JEPA 产品架构。从论文到产品的距离仍然很远，AMI Labs 目前还没有公开的产品或 benchmark 结果超出 V-JEPA 2 的范围。

## 九、JEPA 路线的开放问题

最后说几个 JEPA 路线目前还没有很好回答的问题：

**表征充分性（representation sufficiency）。** 这可能是 JEPA 从 representation learning 走向真正 world model 时最关键的问题之一。JEPA 最大的潜在优势也是它最大的风险：模型主动过滤不可预测的像素细节。但对于机器人控制而言，一些看似低层的细节（精确几何、接触状态、摩擦、细小物体状态、affordance）可能恰恰决定动作是否成功。一个对视频分类非常好的 representation，不一定是一个对 control 足够充分的 state representation。因此真正需要回答的问题不是"representation 是否比 pixels 更好"，而是：**这个 representation 是否保留了 downstream planning 所需要的全部信息？**

**表征退化与目标设计。** JEPA 通过 target encoder、EMA 更新以及预测目标设计来避免表征退化，但"为什么这种机制能够稳定地学习到有信息量的表征"仍然是一个值得研究的问题。尤其当 predictor、target encoder 和 masking 策略进一步扩展到长时序和 action-conditioned prediction 后，表征是否保持充分且稳定，是比单纯避免 collapse 更重要的问题。

**长程预测的稳定性。** V-JEPA 2-AC 展示了 action-conditioned prediction 和 planning 能力。值得注意的是，作者已经通过 rollout loss 显式训练多步预测，以降低 autoregressive error accumulation——论文中 predictor 的输出会反馈回来做多步预测。但目前机器人实验所展示的规划 horizon 和任务复杂度仍不足以证明其具备长期、复杂任务上的稳定 world-model rollout 能力。相比之下，Dreamer 等 model-based RL 方法从一开始就把 multi-step latent rollout 和 policy/value learning 放在核心训练闭环中，因此两者在长程规划问题上的技术路径并不相同。

**和语言的对齐。** 当前 VLA（Vision-Language-Action）模型的核心能力之一是语言接地。JEPA 路线目前主要在视觉和动作空间工作，怎么和语言能力结合是一个重要的开放问题。

**可扩展性。** V-JEPA 2 已经扩展到十亿参数规模，但 JEPA 类模型是否存在稳定、可预测的 scaling law 仍缺乏充分证据。更关键的问题不是简单追求参数规模，而是随着模型规模、视频数据量和预测 horizon 增加，representation quality、action-conditioned prediction accuracy 和 downstream planning performance 是否持续提升。

---

JEPA 路线的核心洞察——在以视觉表征学习、预测和规划为目标的任务中，latent representation prediction 可以避免像素级生成的计算负担，并取得很强的下游性能——在 I-JEPA 到 V-JEPA 2-AC 的系列实验中得到了逐步验证。从学术假设到 AMI Labs 的工程化尝试，这条路线正在走向更广阔的应用场景。

但它不是世界模型的唯一答案。正如我在[盘点文章](/zh/articles/2026-09-01-world-model-h2-review/)的结论里说的，"world model"正在失去单一含义。不同的问题需要不同的工具。

JEPA 真正值得关注的地方，不是它提出了另一个"世界模型架构"，而是它重新定义了世界模型应该预测什么。像素生成路线通常通过预测未来观测来学习世界动态；JEPA 则把预测目标直接放在学习到的表征空间中。前者需要解释并生成大量观测细节，后者主动把预测重点放在表征中更可预测、对任务更有用的因素上。而 V-JEPA 2-AC 又往前走了一步："执行这个动作之后，世界的任务相关状态会怎样变化"。

但更深层的问题是：**一个用于预测和决策的内部状态，究竟需要包含多少观测信息？** 这才是 JEPA 路线真正挑战的核心问题——不是"pixels bad, latent good"，而是"what information is sufficient for prediction and control"。I-JEPA/V-JEPA 证明的是 predictive representation learning 的有效性；V-JEPA 2-AC 才开始验证 representation 是否能够承载 action-conditioned dynamics。但"representation learning 有效"与"representation 是控制充分状态"之间仍然存在一道尚未完全解决的鸿沟。这才是 JEPA 从 representation learning 走向 world modeling 的真正转折点，也是这条路线最值得持续关注的地方。

*下一篇，我打算聊具身智能方向创业的团队配置——不是商业计划书，而是一个工程师视角的务实分析。*
