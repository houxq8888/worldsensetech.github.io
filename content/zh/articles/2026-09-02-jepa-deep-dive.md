---
title: "JEPA 深度解读：从 I-JEPA 到 V-JEPA 2-AC，预测性表征如何走向世界模型"
slug: "2026-09-02-jepa-deep-dive"
date: 2026-09-02
draft: false
categories: ["世界模型", "论文解读"]
tags: ["JEPA", "I-JEPA", "V-JEPA", "V-JEPA 2", "V-JEPA 2-AC", "AMI Labs", "LeCun", "预测性表征学习", "自监督学习", "世界模型", "具身智能"]
description: "从 2022 年的理论蓝图到 2023 年的 I-JEPA，从 2024 年的 V-JEPA 到 2025 年的 V-JEPA 2 与 V-JEPA 2-AC，再到 2026 年 AMI Labs 的 10.3 亿美元融资——LeCun 的 JEPA 路线用四年时间走完了从'预测像素不如预测表征'到'action-conditioned 机器人操作'的全过程。这篇逐篇拆解这条技术路线的核心思路、关键实验和开放问题。"
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

如果说 I-JEPA 证明了"在表征空间预测图像 patch 可行"，V-JEPA 要回答的问题是：**能不能在表征空间预测视频的未来帧？**

这比图像难很多。图像中被遮蔽的区域是空间上固定的，模型只需要理解空间结构。但视频中被预测的未来帧涉及时间动态——模型需要理解运动、因果关系和时间演化。

V-JEPA 的做法是：把 I-JEPA 的框架扩展到时空维度。Context Encoder 接收历史视频帧的时空 patch，Predictor 需要在表征空间中预测未来帧的抽象表征。

### 技术要点

V-JEPA 保持了 JEPA 的核心设计哲学：

- Target Encoder 仍然用 EMA 更新
- 预测仍然在表征空间进行，不回到像素空间
- 遮蔽策略从空间遮蔽扩展到时空遮蔽——遮蔽未来帧的部分区域

### 关键贡献

V-JEPA 的核心贡献是证明了 JEPA 框架可以自然地扩展到视频领域，而且学到的表征在动作识别、视频理解等下游任务上表现良好。在 frozen representation 评估下，V-JEPA 取得了相当强的结果：Kinetics-400 上 82.1%，Something-Something-v2 上 71.2%，ImageNet 上 77.9%。

从路线演进来看，V-JEPA 更像是 JEPA 从图像表征学习走向视频动态建模的关键过渡：它证明了 latent video prediction 可以学习强大的时空表征。真正把这条路线进一步推进到"action-conditioned prediction + planning"的，则是 V-JEPA 2。

## 四、V-JEPA 2：从表征学习到世界模型（2025）

### 论文信息

*V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction, and Planning*，Meta AI，2025 年 6 月。[论文](https://arxiv.org/abs/2506.09985)。

### 这是最重要的一篇

如果说 I-JEPA 和 V-JEPA 是基础研究，V-JEPA 2 是 JEPA 路线第一次真正展示"我能做世界模型的事"。

### 架构升级

V-JEPA 2 的架构有重大升级：

**编码器**：Vision Transformer，提供多种规模——ViT-L（约 3 亿参数）、ViT-H（约 6 亿参数）、ViT-g（约 **10 亿参数**）。视频输入被切分为 2x16x16 的 tubelets（2 帧 × 16 × 16 像素）。位置编码使用 **3D-RoPE**（三维旋转位置编码），这是处理时空序列的关键设计。

**Action-conditioned 变体（V-JEPA 2-AC）**：这是 V-JEPA 2 最重要的新增。一个约 **3 亿参数的 predictor Transformer**，使用 **block-causal attention**，接收动作序列作为条件输入来预测未来表征。

这意味着：模型不仅能"看懂"视频，还能"想象"如果执行某个动作，世界会变成什么样。这就是世界模型的核心能力。

### 预训练

预训练规模很大：

- 使用 **VideoMix22M** 数据构建方案，规模约 **2200 万个视频/图像样本**；论文描述其数据来源覆盖超过百万小时级别的互联网视频
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

这正是 LeCun 在 2022 年论文里画的蓝图：**用预测性表征学习构建能理解物理世界的 AI 系统。** 三年后，V-JEPA 2 把这个蓝图变成了可运行的系统。

## 五、AMI Labs：从论文到公司（2026）

### 基本信息

2026 年 3 月，Yann LeCun 在巴黎创办了 [AMI Labs](https://amilabs.xyz/)（Advanced Machine Intelligence Labs），完成约 **10.3 亿美元**（约 8.9 亿欧元）的种子轮融资，pre-money 估值约 **35 亿美元**。这是近年来欧洲 AI 基础模型领域最大的种子轮。

### 团队

AMI Labs 的团队阵容非常强：

- **Yann LeCun**：Executive Chairman（执行主席）
- **Saining Xie（谢赛宁）**：Chief Science Officer（首席科学官）——纽约大学教授，长期从事视觉表征学习和自监督学习研究，参与了 MAE 等视觉基础模型方向的研究
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

AMI 的公开技术愿景与 JEPA 所强调的预测性表征、世界状态建模和 action-conditioned planning 高度一致；加上 Saining Xie 等视觉表征学习领域核心研究者的加入，使得 JEPA 很可能成为其重要技术基础之一。但截至目前公开资料仍不足以证明 AMI 的最终架构就是 V-JEPA 的直接产业化版本，因此这里更适合称为**技术延续关系**而非确定的架构继承关系。

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

### 什么时候 JEPA 才算世界模型？

这个问题值得单独拿出来讨论，因为它直接影响我们怎么理解整条技术路线。

**I-JEPA** 本身更适合被称为 predictive representation learning，而不是完整的 world model。它的目标是学习对下游任务有用的图像表征，没有时序动态建模，也没有 action 的概念。

**V-JEPA** 引入了 temporal prediction，但主要目标仍然是学习视频表征——它预测未来帧的表征，但不支持 action-conditioned 的规划。

**V-JEPA 2-AC** 则进一步加入了 action-conditioned prediction，并通过 model-based planning 在机器人上预测动作后果。到了这里，JEPA 才真正具备了我们通常意义上的 world model 核心结构：

**observation → latent state → action → predicted future state → planning。**

因此，如果要严格区分，我会把 I-JEPA / V-JEPA 称为"JEPA representation learning"，把 V-JEPA 2-AC 称为"JEPA-based latent world model"。这个区分很重要——它让我们更精确地理解 JEPA 路线中哪些部分是表征学习，哪些部分是世界建模。

## 八、对从业者的意义

### 如果你在做机器人控制

V-JEPA 2-AC 的操作结果值得认真关注。在论文设置下，16 秒/action vs Cosmos 的 4 分钟/action（约 15× 时间优势），这个效率差距在实时控制场景下是决定性的。如果你的系统需要快速推理而不是生成漂亮的视频，JEPA 路线可能更适合。

### 如果你在做自监督学习

I-JEPA 的训练效率表现（5.3× fewer epochs 达到与 MAE 接近的性能）已经足够引起注意。JEPA 框架提供了一个不依赖像素重建的自监督学习范式，这在计算成本越来越受关注的今天特别有价值。

### 如果你在做世界模型研究

JEPA 路线提供了一个重要的 alternative：不是所有世界模型都需要生成像素。如果你的下游任务需要的是理解而非生成，在表征空间做预测可能是更好的选择。

### 如果你在关注 AMI Labs

AMI Labs 的团队和融资规模说明了一件事：投资人对 JEPA 路线的产业价值有很强的信心。但也要清醒——从论文到产品的距离仍然很远。AMI Labs 目前还没有公开的产品或 benchmark 结果超出 V-JEPA 2 的范围。

## 九、JEPA 路线的开放问题

最后说几个 JEPA 路线目前还没有很好回答的问题：

**表征充分性（representation sufficiency）。** 这可能是 JEPA 从 representation learning 走向真正 world model 时最关键的问题之一。JEPA 最大的潜在优势也是它最大的风险：模型主动过滤不可预测的像素细节。但对于机器人控制而言，一些看似低层的细节（精确几何、接触状态、摩擦、细小物体状态、affordance）可能恰恰决定动作是否成功。一个对视频分类非常好的 representation，不一定是一个对 control 足够充分的 state representation。因此真正需要回答的问题不是"representation 是否比 pixels 更好"，而是：**这个 representation 是否保留了 downstream planning 所需要的全部信息？**

**表征坍塌（representation collapse）。** JEPA 用 EMA target encoder 来避免坍塌，但这不是理论上完美的解决方案。和 contrastive learning 一样，JEPA 需要仔细设计训练目标来确保表征不会退化为常数。

**长程预测的稳定性。** V-JEPA 2 展示了短程动作预测的能力，但对于需要长程规划的任务（比如机器人操作中的多步骤任务），JEPA 的预测会不会逐步发散？这个问题在 Dreamer 里通过 KL balancing 和 imagined rollout 来处理，JEPA 目前没有对等的机制。

**和语言的对齐。** 当前 VLA（Vision-Language-Action）模型的核心能力之一是语言接地。JEPA 路线目前主要在视觉和动作空间工作，怎么和语言能力结合是一个重要的开放问题。

**可扩展性的上限。** V-JEPA 2 到了 10 亿参数，但和 LLM 的千亿参数相比还有很大差距。JEPA 的 scaling law 是什么样的？能不能像 LLM 一样持续 scale？这些问题还没有答案。

---

JEPA 路线的核心洞察——在以视觉表征学习、预测和规划为目标的任务中，latent representation prediction 可以避免像素级生成的计算负担，并取得很强的下游性能——在 I-JEPA 到 V-JEPA 2-AC 的系列实验中得到了逐步验证。从学术假设到 AMI Labs 的工程化尝试，这条路线正在走向更广阔的应用场景。

但它不是世界模型的唯一答案。正如我在[盘点文章](/zh/articles/2026-09-01-world-model-h2-review/)的结论里说的，"world model"正在失去单一含义。不同的问题需要不同的工具。

JEPA 真正值得关注的地方，不是它提出了另一个"世界模型架构"，而是它重新定义了世界模型应该预测什么。像素生成路线问的是"未来看起来是什么"，JEPA 问的是"未来哪些状态变化值得被预测"，而 V-JEPA 2-AC 又往前走了一步："执行这个动作之后，世界的任务相关状态会怎样变化"。这才是 JEPA 从 representation learning 走向 world modeling 的真正转折点。

*下一篇，我打算聊具身智能方向创业的团队配置——不是商业计划书，而是一个工程师视角的务实分析。*
