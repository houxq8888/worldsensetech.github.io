---
title: "世界模型 2026 中期盘点：从 Cosmos、Genie 到 JEPA 的路线分化"
slug: "2026-09-01-world-model-h2-review"
date: 2026-09-01
draft: false
categories: ["世界模型"]
tags: ["世界模型", "2026盘点", "NVIDIA Cosmos", "Genie 3", "AMI Labs", "JEPA", "具身智能", "机器人AI", "论文推荐"]
description: "截至 2026 年 8 月底，世界模型方向正在经历一次深层分化。从 NVIDIA Cosmos 到 Google Genie 3，从 LeCun 的 AMI Labs 到 Fei-Fei Li 的 World Labs，不同技术路线开始走向不同的应用场景。本文尝试理清这些项目和论文之间的关系。"
toc: true
related_articles:
  - world-model-2026-trend
  - vla-vs-world-model
  - world-model-intro
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
  - world-model-transformer
---

2026 年过了大半，世界模型方向的节奏明显加快了。

上半年我写了不少基础性的文章——从 RSSM 的数学原理到 DreamerV3 的训练实战，从 VLA vs 世界模型的路线对比到 Sim-to-Real 的工程踩坑。有读者问我：现在想跟踪世界模型的最新进展，应该看什么？

今天这篇文章，我把截至 2026 年 8 月底值得关注的论文和项目做一个盘点。这里"值得关注"并不意味着这些项目都在下半年首次发布——有些（如 Cosmos、Genie）最早发布于 2025 年，但在 2026 年持续发展并产生了重要影响。我更关心的是：截至现在，它们仍然代表着接下来半年值得持续跟踪的技术路线。

这不是大而全的综述列表，而是我作为一个在这个方向工作的工程师，从"这些东西对我有什么用"的角度来筛选和点评。

但在进入具体项目之前，有一件事必须先说清楚。

## 一、先把"世界模型"分清楚

现在"世界模型"这个概念被用得太滥了——视频生成模型叫世界模型，游戏引擎叫世界模型，甚至一些简单的预测模型也自称世界模型。但上面这些项目，虽然都被称为"world model"，其实不是同一种东西。

我在阅读[《A Definition and Roadmap for World Models》（arxiv 2607.06401）](https://arxiv.org/html/2607.06401v1)这篇论文后，结合论文的分类框架和我自己的工程经验，把当前的世界模型分成四类：

**A. 隐状态动力学世界模型（Latent Dynamics World Model）**

代表：Dreamer / RSSM

核心逻辑：state → action → next state，在隐空间中学习环境动态。

目标：支持 planning、RL、control。这是我在博客里写得最多的类型。

**B. 生成式视频世界模型（Generative Video World Model）**

代表：NVIDIA Cosmos 等

核心逻辑：condition + history → future observations，生成未来视频帧。

目标偏向：data generation、simulation、prediction、perception。

**C. 交互式世界模型（Interactive World Model）**

代表：Google Genie 系列

核心逻辑：state + action → interactive future，根据动作生成可交互的未来。

关键能力：action-conditioned generation、temporal consistency、controllability。

**D. 空间/3D 世界模型（Spatial / 3D World Model）**

代表：World Labs Marble 等

重点：persistent scene、geometry、spatial consistency、navigability、3D representation。

有了这个分类，后面讨论具体项目时就不会把它们混为一谈。

2026 年"世界模型"最大的变化，并不是出现了一个统一的 World Model，而是**不同 world-model paradigms 开始分化**——各自走向不同的应用场景和评价标准。

下面按类别逐一来看。

## 二、生成式视频世界模型：NVIDIA Cosmos

### 不只是"视频生成"

NVIDIA 在 CES 2025 上首次发布了 [Cosmos 世界基础模型平台](https://www.nvidia.com/en-us/ai/cosmos/)，到 2026 年初下载量已突破 200 万。

这里需要做一个重要的区分：Cosmos 不是单纯的"视频生成模型"。它是一个面向 Physical AI 的开发平台，覆盖视频生成、世界状态理解、数据处理和合成数据生成等多种能力。把它简化成"视频生成"会低估它的技术野心。

为什么重要？因为我在[之前那篇合成数据的文章](/zh/articles/world-model-synthetic-data-for-vla/)里详细分析过，真实机器人数据的采集成本高、覆盖面窄，是世界模型落地的核心瓶颈。Cosmos 的思路是：用物理感知的生成模型来大规模产生合成训练数据，用于自动驾驶和机器人的感知与控制。

对从业者的意义：如果你在做机器人或自动驾驶，Cosmos 的开源模型和工具链值得认真看一看。如果其模型、数据工具和部署生态进一步成熟，它有潜力成为 Physical AI 领域的重要开源基础设施。

## 三、交互式世界模型：Google DeepMind Genie 3

### 实时生成 ≠ 可用于控制

Genie 系列是 Google DeepMind 在世界模型方向的重要布局。[Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) 在 2025 年 8 月发布，被定位为"第一个实时交互式通用世界模型"。

它能做什么？给它一张图片或一段文字描述，它能以 24fps 的帧率生成可交互的世界模拟——根据用户的动作输入，持续演化出视觉上连贯的场景。这不是预渲染的视频，也不是传统意义上可导出的、具有完整几何结构和物理引擎的 3D 世界。它更接近：从视觉/文本条件生成一个可交互、随动作变化而持续演化的世界模拟。

和前两代的区别：Genie 1 证明了从视频学习交互式环境的可行性，Genie 2 扩展到了大规模基础模型，Genie 3 则把实时性和通用性推到了新高度。

但这里有一个很重要的技术问题值得指出：**实时生成 ≠ 可用于控制**。

24fps 的帧率只是众多指标中的一个。对于机器人世界模型，更关键的还包括：temporal consistency（时间一致性）、action controllability（动作可控性）、long-horizon stability（长程稳定性）、spatial consistency（空间一致性）、object permanence（物体持久性）等等。帧率高并不意味着它已经适合作为机器人训练环境。

这个区分很关键，也是当前交互式世界模型面临的核心挑战之一。

## 四、空间/3D 世界模型：World Labs Marble

### 空间智能的商业化路径

李飞飞创办的 [World Labs](https://www.worldlabs.ai/) 在 2025 年 11 月发布了 Marble，能从各种媒体输入（图片、视频、文字）生成持久的、可下载的 3D 空间。

Marble 和 Cosmos 虽然都涉及世界建模，但产品目标和技术侧重点并不相同。Cosmos 更偏 Physical AI / simulation / synthetic data / robotics，而 Marble 更偏 3D world generation / spatial intelligence / world reconstruction / content creation。两者都在向"可生成、可交互、具有空间一致性的世界表示"推进，但方向不同。

这是世界模型走向商业化的一个重要信号——空间智能这条路线离内容创作和 AR/VR 应用更近。但值得注意的是，World Labs 在 2026 年也在向机器人仿真方向拓展，其 Marble 技术正在被用于生成机器人训练所需的 3D 场景数据。这意味着 Spatial/3D 世界模型和 Physical AI 之间的边界正在模糊。

## 五、预测性表征学习：AMI Labs 与 JEPA

### LeCun 的路线选择

Yann LeCun 在巴黎创办了 [AMI Labs](https://amigroup.ai/)（Advanced Machine Intelligence Labs），于 2026 年 3 月完成约 10.3 亿美元（约 8.9 亿欧元）融资，使其成为近年来欧洲 AI 基础模型领域最受关注的新公司之一。

AMI Labs 的技术路线基于 LeCun 多年来一直推崇的 [JEPA（Joint Embedding Predictive Architecture）](https://openreview.net/pdf?id=BZ5a1r-kVsf)。JEPA 的核心思想是：**predict representations rather than raw observations**——在抽象表征空间做预测，而不是在像素空间做预测。

这和 Dreamer 系列的思路有相通之处——我在 [RSSM 详解](/zh/articles/rssm-deep-dive/)里讲过，RSSM 也是在隐空间做动态预测，而不是直接预测下一帧图像。但两者并不是同一级别的东西：RSSM 是一个 latent dynamics model，目标是支持规划和控制；JEPA 是一个 predictive representation architecture，目标是学习对任务有用的世界状态表征。

JEPA 的核心论点不是"像素重建是错的"，更准确的说法是：**对于学习高层语义和世界状态表示而言，要求模型精确预测所有像素并不是理想的学习目标**——因为像素空间包含大量与任务无关的细节和随机性。

为什么值得关注？LeCun 是深度学习的奠基人之一，他对技术方向的判断有很强的参考价值。他选择 all-in 基于 JEPA 的世界模型路线，至少说明投资人和创始团队对这一技术路线未来几年的产业价值具有非常强的判断。当然，能不能做出东西，还要看执行。

## 六、重要综述论文：建立全局视野

如果你只想读几篇论文来建立世界模型的全局认知，我推荐以下几篇：

### 《A Definition and Roadmap for World Models》

这篇 [arxiv 论文（2607.06401）](https://arxiv.org/html/2607.06401v1) 做了一件很有价值的事：给世界模型一个正式的定义，并画出了一张技术路线图。上面的四类分类法就主要参考了这篇论文。

### 《World Model for Robot Learning: A Comprehensive Survey》

这篇[论文](https://huggingface.co/papers/2605.00080)专注于世界模型在机器人学习中的应用。如果你关心的是"世界模型怎么用在真实机器人上"，这篇更有针对性。它系统梳理了世界模型在感知、规划、控制三个环节的应用方式。

### 《A Comprehensive Survey on World Models for Embodied AI》

这篇[论文](https://arxiv.org/abs/2510.16732)从具身智能的角度综述世界模型，把世界模型放在了具身智能的大框架里来分析。[GitHub 上有配套的论文列表](https://github.com/Li-Zn-H/AwesomeWorldModels)可以作为延伸阅读。

## 七、一张表看清技术路线

把上面讨论的项目放在一起对比：

| 项目 | 核心范式 | 预测什么 | 怎么验证有用 | 当前成熟度 | 主要应用场景 |
|---|---|---|---|---|---|
| Cosmos | 生成式世界模型 | 未来视频帧 / 世界状态 | 合成数据对下游感知/控制的提升幅度 | 开源可用 | 自动驾驶 / 机器人训练数据 |
| Genie 3 | 交互式世界模型 | 动作条件下的未来视觉 | 交互一致性与长程稳定性 | 研究预览 | 仿真环境 / 原型验证 |
| Marble | 3D 世界生成 | 持久的空间几何表示 | 3D 重建精度与空间一致性 | 商业产品 | 空间智能 / 机器人仿真 |
| Dreamer | 隐状态动力学 | 隐空间中的下一步状态 | RL 任务得分与样本效率 | 学术成熟 | 机器人控制 / RL |
| JEPA 系列 | 预测性表征学习 | 抽象表征（非像素） | 下游任务表征质量 | 早期研究 | 表征学习 / 世界理解 |

这张表比大量形容词更有价值。当你看到一个新的"世界模型"时，先把它放进这个框架里，就能快速判断它和其他工作的关系。

## 八、世界模型最大的短板：评价体系

讲了这么多项目和技术，有一个问题始终绕不开：**我们到底怎么证明一个世界模型是"好"的？**

这是当前世界模型领域最薄弱的环节，也是我在阅读论文时最关注的维度。2025-2026 年的多篇综述已经把 benchmark、metrics、physical consistency、computational efficiency、long-horizon consistency 列为核心开放问题。

我认为世界模型的评价可以分成这样一个阶梯：

```
Generation quality（生成质量）
       ↓
Temporal consistency（时间一致性）
       ↓
Physical consistency（物理一致性）
       ↓
Action controllability（动作可控性）
       ↓
Counterfactual accuracy（反事实准确性）
       ↓
Long-horizon stability（长程稳定性）
       ↓
Downstream task improvement（下游任务收益）
```

越往下，越接近一个真正有用的世界模型。

目前大多数工作还停留在上面几层——视频生成模型在 generation quality 上表现惊艳，但能不能做到物理一致？能不能被动作精确控制？对下游任务到底有没有帮助？这些问题往往没有答案。

这也是为什么我在前面反复强调"实时生成 ≠ 可用于控制"、"不能因为一个系统有预测能力就称之为 VLA + world model"。**评价标准正在从"生成得像不像"转向"预测准不准、能不能被 action 控制、对下游任务到底有没有用"**——而大多数项目还停留在用上面几层的指标来证明自己。

真正值得关注的，是那些能走到阶梯底部、用下游任务收益来证明世界模型价值的工作。

## 九、关于 VLA + 世界模型的融合

我在 [VLA vs 世界模型](/zh/articles/vla-vs-world-model/)那篇文章里讨论过，VLA 和世界模型不是竞争关系，而是互补关系。2026 年确实出现了越来越多将两者结合的工作，但这里需要特别谨慎。

不能因为一个机器人系统具有预测或规划能力，就直接把它定义成"VLA + world model"。需要具体指出：哪个模块是 world model？是显式的 dynamics model，还是 policy 内部的 latent prediction？是 training-time simulation，还是 inference-time planning？

这种融合如果真的实现——比如世界模型的隐状态表征直接作为 VLA 的条件输入，VLA 的语言接地能力指导世界模型的想象方向——那会是一个非常强的架构。但目前大多数工作还在探索阶段，需要更具体的技术验证。

## 十、技术趋势：三个真正重要的变化

### 趋势一：Transformer 正在进入世界模型

我在[之前的文章](/zh/articles/world-model-transformer/)里详细讨论过这个话题。2026 年的趋势是：Transformer 在大规模序列建模和生成式世界模型中展现出了明显的 scaling 优势，因此正在逐渐进入传统 RSSM/GRU 世界模型过去占据的部分位置。

但需要注意的是，世界模型内部有多个不同模块——observation encoder、latent dynamics、action-conditioned transition、video generation、planner、policy、value model——Transformer 在这些模块里的角色完全不同。不能简单地说"世界模型正在变成 Transformer"。

### 趋势二：评价标准正在改变

2024-2025 年，世界模型的主要问题是"能不能 work"——能不能学会环境动态，能不能在想象中生成有用的数据。

2026 年，问题变成了"怎么用好"——怎么提高样本效率，怎么降低训练成本，怎么在真实机器人上稳定部署。

换句话说，**评价世界模型的标准正在从"生成得像不像"，逐渐转向"预测准不准、能不能被 action 控制、对下游任务到底有没有用"**。真正的分水岭已经从 generation quality 转向 controllability、predictive accuracy 和 downstream utility。

这也是为什么 Cosmos 这样的工业级平台会出现，为什么 DreamerV3 的训练工程实践（我在[这篇文章](/zh/articles/2026-08-28-dreamerv3-training-tips/)里详细写过）变得和算法本身一样重要。

### 趋势三：不同范式走向不同应用场景

正如第一节的分类所示，2026 年的世界模型不是一条统一的路线，而是多个范式各自找到了自己的应用场景：Latent Dynamics 走向 RL 和控制，Generative Video 走向合成数据，Interactive World 走向仿真环境，Spatial/3D 走向空间智能和内容创作。

这种分化是健康的。它说明世界模型不再是一个笼统的概念，而是正在形成具体的技术栈和产品形态。

## 十一、论文阅读建议

最后给一个实用的阅读框架。

**读任何 world model paper，先问 6 个问题：**

1. State representation 是什么？
2. Dynamics model 预测什么？
3. Action 是否进入 dynamics？
4. 预测 horizon 多长？
5. 如何验证预测真的有用？
6. 最终 downstream task 是什么？

尤其第 5、6 点非常关键。否则很容易出现：video prediction benchmark 很漂亮，但对机器人 control 没有帮助。这正是当前世界模型领域非常值得讨论的问题。

**如果你刚入门世界模型：**

先读上面提到的三篇综述中的任意一篇，建立全局认知。然后回头读我这个博客的基础系列——从[什么是世界模型](/zh/articles/world-model-intro/)到 [RSSM 详解](/zh/articles/rssm-deep-dive/)到 [Dreamer 解读](/zh/articles/2026-08-25-dreamer-explained/)，把核心概念搞清楚。

**如果你已经在做世界模型相关研究：**

重点看 Cosmos 的技术报告和 Genie 3 的论文，了解工业界在怎么做大规模世界模型。然后看 AMI Labs 的后续进展——JEPA 路线如果走通，可能会改变整个方向的技术范式。

**如果你关心工程落地：**

Cosmos 的开源工具链是第一优先级。然后看 World Model for Robot Learning 那篇综述里关于 Sim-to-Real 的章节。最后关注 NVIDIA 和 Google 在机器人部署方面的最新分享。

---

世界模型这个方向，2026 年最大的变化不是某个单一突破，而是**分化**——不同范式走向不同场景，评价标准从"能不能生成"转向"有没有用"。理解这个分化，比追任何一个具体项目都重要。

*下一篇文章，我打算聊聊具身智能方向创业需要什么样的团队配置——不是那种大而全的商业计划书，而是一个工程师视角的务实分析。敬请期待。*
