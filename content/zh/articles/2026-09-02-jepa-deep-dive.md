---
title: "JEPA 深度解读：从 I-JEPA 到 V-JEPA 2，LeCun 的预测性表征学习路线"
slug: "2026-09-02-jepa-deep-dive"
date: 2026-09-02
draft: false
categories: ["世界模型", "论文解读"]
tags: ["JEPA", "I-JEPA", "V-JEPA", "V-JEPA 2", "AMI Labs", "LeCun", "预测性表征学习", "自监督学习", "世界模型", "具身智能"]
description: "从 2022 年的理论蓝图到 2023 年的 I-JEPA，从 2024 年的 V-JEPA 到 2025 年的 V-JEPA 2，再到 2026 年 AMI Labs 的 10.3 亿美元融资——LeCun 的 JEPA 路线用四年时间走完了从'预测像素不如预测表征'到'零样本机器人操作'的全过程。这篇逐篇拆解这条技术路线的核心思路和关键实验。"
toc: true
related_articles:
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - world-model-intro
  - 2026-08-25-dreamer-explained
  - world-model-transformer
---

在[上一篇世界模型盘点](/zh/articles/2026-09-01-world-model-h2-review/)里，我花了大量篇幅讨论 Cosmos、Genie 3、Marble 这些项目，但有一个方向我只做了概述——JEPA。

这不是因为 JEPA 不重要。恰恰相反，**JEPA 可能是当前世界模型方向最有理论深度的一条路线**。它背后站着 Yann LeCun，有从 I-JEPA 到 V-JEPA 2 的完整技术演进，有 AMI Labs 10.3 亿美元的融资验证，而且它代表了一种和当前主流的生成式世界模型完全不同的技术哲学。

这篇文章把 JEPA 系列从第一篇到最新一篇做一个完整的技术拆解。

## 一、JEPA 的核心思想：一句话版本

**不要预测像素，预测表征。**

这句话听起来简单，但它和当前绝大多数世界模型（包括 Cosmos、Genie 系列）的基本假设直接对立。

生成式世界模型的逻辑是：给定历史观测，预测未来像素。JEPA 的逻辑是：给定历史观测，预测未来观测的**抽象表征**——在表征空间做预测，而不是在像素空间做预测。

为什么这个区别很重要？因为像素空间包含大量与任务无关的细节：光照变化、纹理细节、随机噪声。要求模型精确预测这些细节，不仅浪费模型容量，还会引入大量无用的梯度信号，拖慢学习速度。

LeCun 在 2022 年的[位置论文](https://openreview.net/forum?id=BZ5a1r-kVsf)里把这个问题说得很清楚：**对于学习高层语义和世界状态表示而言，要求模型精确预测所有像素并不是理想的学习目标。**

这不是说像素级预测没有价值——视频生成确实很酷。JEPA 的论点是：如果你的目标是学习对下游任务有用的世界表征，那么在表征空间做预测是更高效的学习策略。

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

I-JEPA 在**表征层面**做遮蔽。具体来说：采用 multi-block 策略，选择 4 个目标区域（各占 15%-20%），1 个上下文区域（占 85%-100%），去掉重叠部分。关键区别在于，遮蔽发生在 Target Encoder 的输出端，而不是输入像素端。

这意味着：模型不需要从像素级别重建被遮掉的内容，而是需要从可见区域的表征中**推断**被遮区域在抽象表征空间中应该是什么样子。

### 关键结果

I-JEPA 用 ViT-H/14 在 ImageNet 上的结果：

- **Linear probing**：79.3% top-1 准确率（vs MAE 的 77.2%）
- **448 分辨率**：81.1% top-1
- **1% low-shot**：73.3%（vs MAE 的 59.8%）——这个差距非常大
- **Full fine-tuning**：87.1%，且使用的 epoch 数只有 MAE 的 1/5.3

最后一点特别值得注意：**I-JEPA 达到和 MAE 相当的性能，但训练效率高了一个数量级以上。** 这直接验证了 LeCun 的核心论点——在表征空间做预测比在像素空间做预测更高效。

### 为什么重要

I-JEPA 的意义不在于它刷了什么 SOTA，而在于它用一个干净的实验证明了一件事：**不生成像素、只在表征空间做预测的自监督学习，不仅能 work，而且比生成式方法更高效。**

这是 JEPA 路线的第一块基石。

## 三、V-JEPA：从图像到视频（2024）

### 论文信息

*V-JEPA: Latent Video Prediction for Visual Representation Learning*，Meta AI，ICLR 2024。作者：Quentin Garrido 等。

### 核心扩展

如果说 I-JEPA 证明了"在表征空间预测图像 patch 可行"，V-JEPA 要回答的问题是：**能不能在表征空间预测视频的未来帧？**

这比图像难很多。图像中被遮蔽的区域是空间上固定的，模型只需要理解空间结构。但视频中被预测的未来帧涉及时间动态——模型需要理解运动、因果关系和时间演化。

V-JEPA 的做法是：把 I-JEPA 的框架扩展到时空维度。Context Encoder 接收历史视频帧的时空 patch，Predictor 需要在表征空间中预测未来帧的抽象表征。

### 技术要点

V-JEPA 保持了 JEPA 的核心设计哲学：

- Target Encoder 仍然用 EMA 更新
- 预测仍然在表征空间进行，不回到像素空间
- 遮蔽策略从空间遮蔽扩展到时空遮蔽——遮蔽未来帧的部分区域

### 关键贡献

V-JEPA 的核心贡献是证明了 JEPA 框架可以自然地扩展到视频领域，而且学到的表征在动作识别、视频理解等下游任务上表现良好。

但坦率地说，V-JEPA 更像是一个"可行性验证"——它证明了这条路走得通，但还没有展现出改变游戏规则的能力。真正的突破在 V-JEPA 2。

## 四、V-JEPA 2：从表征学习到世界模型（2025）

### 论文信息

*V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction, and Planning*，Meta AI，2025 年 6 月。[论文](https://arxiv.org/abs/2506.09985)。

### 这是最重要的一篇

如果说 I-JEPA 和 V-JEPA 是基础研究，V-JEPA 2 是 JEPA 路线第一次真正展示"我能做世界模型的事"。

### 架构升级

V-JEPA 2 的架构有重大升级：

**编码器**：Vision Transformer，最大到 **10 亿参数**。视频输入被切分为 2x16x16 的 tubelets（2 帧 × 16 × 16 像素）。位置编码使用 **3D-RoPE**（三维旋转位置编码），这是处理时空序列的关键设计。

**Action-conditioned 变体**：这是 V-JEPA 2 最重要的新增。一个 **3 亿参数的 Transformer**，使用 **block-causal attention**，接收动作序列作为条件输入来预测未来表征。

这意味着：模型不仅能"看懂"视频，还能"想象"如果执行某个动作，世界会变成什么样。这就是世界模型的核心能力。

### 预训练

预训练规模很大：

- **2200 万样本**，超过 100 万小时的视频数据
- **25.2 万次迭代**
- 采用渐进式分辨率策略（从低分辨率开始，逐步提高）
- 使用 mask-denoising feature prediction 目标

### 关键结果

**视频理解**：

- 6 个 benchmark 平均 **88.2 分**
- 运动理解 **77.3%** top-1
- 动作预测 recall@5 **39.7**
- Video QA（PerceptionTest）**84.0**
- 超过 InternVideo2 和 DINOv2

**机器人操作（这是最让我兴奋的部分）**：

V-JEPa 2 在 62 小时的无标注机器人操作数据上做了 post-training，然后测试零样本（zero-shot）真实机器人操作能力：

- 目标到达精度：**小于 4 厘米**
- 杯子抓取放置：**80% 成功率**
- 每个动作规划时间：**16 秒**

作为对比：

- **Cosmos baseline**：计算一个动作需要 4 分钟，且物体操作任务失败
- **Octo baseline**：能到达但抓不住，盒子抓取 0% 成功率

这个对比非常有说服力。V-JEPA 2 不仅比生成式方法快了两个数量级（16 秒 vs 4 分钟），而且在操作任务上从零开始就能成功。

### 为什么 V-JEPA 2 是 JEPA 路线的转折点

V-JEPA 2 第一次展示了：

1. JEPA 框架可以扩展到十亿参数级别
2. Action-conditioned 预测可以服务于真实的机器人控制
3. 在表征空间做预测不仅理论上优雅，工程上也比像素级预测更高效
4. 零样本机器人操作——不需要为每个任务重新训练

这正是 LeCun 在 2022 年论文里画的蓝图：**用预测性表征学习构建能理解物理世界的 AI 系统。** 三年后，V-JEPA 2 把这个蓝图变成了可运行的系统。

## 五、AMI Labs：从论文到公司（2026）

### 基本信息

2026 年 3 月，Yann LeCun 在巴黎创办了 [AMI Labs](https://amilabs.xyz/)（Advanced Machine Intelligence Labs），完成约 **10.3 亿美元**（约 8.9 亿欧元）的种子轮融资。这是近年来欧洲 AI 基础模型领域最大的种子轮。

### 团队

AMI Labs 的团队阵容非常强：

- **Yann LeCun**：Executive Chairman（执行主席）
- **Saining Xie（谢赛宁）**：Chief Science Officer（首席科学官）——纽约大学教授，计算机视觉领域最有影响力的研究者之一，MAE 的作者
- **Alex LeBrun**：CEO
- **Michael Rabbat**：VP of World Models
- **Laurent Solly**：COO
- **Pascale Fung**：Chief Research and Innovation Officer

办公室分布在巴黎、纽约、蒙特利尔和新加坡。

### 投资人

种子轮由 Cathay Innovation、Greycroft、Hiro Capital 和 HV Capital 联合领投。个人投资者包括 **Jeff Bezos、NVIDIA、Eric Schmidt、Mark Cuban**。

### 技术方向

AMI Labs 的官方表述是：构建能理解物理环境、保持长期信息、执行逻辑规划的世界模型。他们认为真正的智能来源于物理环境而非文本，因此专注于从多模态传感器输入中学习抽象表征，过滤掉不可预测的细节，在概念空间中预测结果。

特别值得注意的是，AMI Labs 明确强调 **action-conditioned world models**——让自主智能体能预测动作的后果并规划后续步骤。

这和 V-JEPA 2 的技术方向高度一致。考虑到 Saining Xie（CSO）正是 JEPA 系列论文的核心作者之一，可以合理推断 AMI Labs 的技术路线就是 JEPA 的产业化延伸。

### 目标领域

工业过程控制、自动化、机器人、个人健康监测、医疗服务。这些都是需要高可控性和安全性的领域——恰好是生成式模型最难保证可靠性的领域。

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

## 七、JEPA 在世界模型图谱中的位置

回到我在[盘点文章](/zh/articles/2026-09-01-world-model-h2-review/)里的四条技术路线：

- **Latent Dynamics**（Dreamer/RSSM）：state → action → next state
- **Generative Video**（Cosmos）：condition → future video frames
- **Interactive**（Genie 3）：state + action → interactive future
- **Spatial/3D**（Marble）：persistent spatial representation

JEPA 不完全属于其中任何一类。它最接近 Latent Dynamics，因为它也在隐空间做预测。但它不像 Dreamer 那样有显式的 state transition 结构，也不以 RL 和 planning 为主要目标。

如果非要归类，我会说 **JEPA 代表的是第五条路线：Predictive Representation Learning**。它的核心贡献不是"怎么建模世界动态"，而是"用什么目标函数来学习世界表征"。

这也是为什么我在盘点文章里说 JEPA 不是"一个完整的 world-model 定义"——它更像是一种**学习哲学**：与其生成所有细节，不如预测有用的抽象。

## 八、对从业者的意义

### 如果你在做机器人控制

V-JEPA 2 的零样本操作结果值得认真关注。16 秒/action vs Cosmos 的 4 分钟/action，这个效率差距在实时控制场景下是决定性的。如果你的系统需要快速推理而不是生成漂亮的视频，JEPA 路线可能更适合。

### 如果你在做自监督学习

I-JEPA 的训练效率优势（比 MAE 快一个数量级）已经足够引起注意。JEPA 框架提供了一个不依赖像素重建的自监督学习范式，这在计算成本越来越受关注的今天特别有价值。

### 如果你在做世界模型研究

JEPA 路线提供了一个重要的 alternative：不是所有世界模型都需要生成像素。如果你的下游任务需要的是理解而非生成，在表征空间做预测可能是更好的选择。

### 如果你在关注 AMI Labs

AMI Labs 的团队和融资规模说明了一件事：投资人对 JEPA 路线的产业价值有很强的信心。但也要清醒——从论文到产品的距离仍然很远。AMI Labs 目前还没有公开的产品或 benchmark 结果超出 V-JEPA 2 的范围。

## 九、JEPA 路线的开放问题

最后说几个 JEPA 路线目前还没有很好回答的问题：

**表征坍塌（representation collapse）。** JEPA 用 EMA target encoder 来避免坍塌，但这不是理论上完美的解决方案。和 contrastive learning 一样，JEPA 需要仔细设计训练目标来确保表征不会退化为常数。

**长程预测的稳定性。** V-JEPA 2 展示了短程动作预测的能力，但对于需要长程规划的任务（比如机器人操作中的多步骤任务），JEPA 的预测会不会逐步发散？这个问题在 Dreamer 里通过 KL balancing 和 imagined rollout 来处理，JEPA 目前没有对等的机制。

**和语言的对齐。** 当前 VLA（Vision-Language-Action）模型的核心能力之一是语言接地。JEPA 路线目前主要在视觉和动作空间工作，怎么和语言能力结合是一个重要的开放问题。

**可扩展性的上限。** V-JEPA 2 到了 10 亿参数，但和 LLM 的千亿参数相比还有很大差距。JEPA 的 scaling law 是什么样的？能不能像 LLM 一样持续 scale？这些问题还没有答案。

---

JEPA 路线的核心洞察——**预测表征比预测像素更高效**——在理论和实验上都得到了验证。从 I-JEPA 到 V-JEPA 2 再到 AMI Labs，这条路线正在从学术假设走向工程实践。

但它不是世界模型的唯一答案。正如我在[盘点文章](/zh/articles/2026-09-01-world-model-h2-review/)的结论里说的，"world model"正在失去单一含义。JEPA 回答的是"怎么学习世界表征"这个问题，而不是"怎么做世界规划"或"怎么生成训练数据"。不同的问题需要不同的工具。

*下一篇，我打算聊具身智能方向创业的团队配置——不是商业计划书，而是一个工程师视角的务实分析。*
