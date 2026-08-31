---
title: "世界模型 2026 下半年盘点：值得关注的论文和项目"
slug: "2026-09-01-world-model-h2-review"
date: 2026-09-01
draft: false
categories: ["世界模型"]
tags: ["世界模型", "2026盘点", "NVIDIA Cosmos", "Genie 3", "AMI Labs", "具身智能", "机器人AI", "论文推荐"]
description: "从 NVIDIA Cosmos 到 Google Genie 3，从 LeCun 的 AMI Labs 到 Fei-Fei Li 的 World Labs，盘点 2026 下半年世界模型方向最值得关注的论文、项目和趋势。"
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

今天这篇文章，我把 2026 下半年（主要是最近几个月）值得关注的论文和项目做一个盘点。不是大而全的综述列表，而是我作为一个在这个方向工作的工程师，从"这些东西对我有什么用"的角度来筛选和点评。

## 一、工业级世界基础模型：从论文到平台

### NVIDIA Cosmos：物理 AI 的数据引擎

NVIDIA 在今年 CES 上正式发布了 [Cosmos 世界基础模型平台](https://www.nvidia.com/en-us/ai/cosmos/)，到 2026 年 1 月下载量已经突破 200 万。

这不是一个学术 demo，而是一个工业级的平台。它的核心能力是：用物理感知的视频生成模型，为自动驾驶和机器人产生大规模合成训练数据。

为什么重要？因为我在[之前那篇合成数据的文章](/zh/articles/world-model-synthetic-data-for-vla/)里详细分析过，真实机器人数据的采集成本高、覆盖面窄，是世界模型落地的核心瓶颈。Cosmos 的思路是：既然世界模型已经学会了物理规律，那就让它"造数据"——生成各种场景、各种光照、各种物体交互的视频，用来训练下游的感知和控制模型。

对从业者的意义：如果你在做机器人或自动驾驶，Cosmos 的开源模型和工具链值得认真看一看。它可能成为物理 AI 领域的"Llama"——一个大家都能用的基础底座。

### Google DeepMind Genie 3：实时交互式世界模型

Genie 系列是 Google DeepMind 在世界模型方向的重要布局。[Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) 在 2025 年 8 月发布，被定位为"第一个实时交互式通用世界模型"。

它能做什么？给它一张图片或一段文字描述，它能以 24fps 的帧率渲染出可交互的 3D 空间。注意，这不是预渲染的视频，而是用"自学物理"实时生成的——没有硬编码的物理规则，模型自己学会了物体怎么运动、碰撞怎么发生。

和前两代的区别：Genie 1 证明了从视频学习交互式环境的可行性，Genie 2 扩展到了大规模基础模型，Genie 3 则把实时性和通用性推到了新高度。24fps 意味着它已经接近"可用"的门槛——虽然还达不到游戏引擎的精度，但作为训练环境和原型验证工具已经足够。

### World Labs Marble：空间智能的商业化尝试

李飞飞创办的 [World Labs](https://www.worldlabs.ai/) 在 2025 年 11 月发布了 Marble，能从各种媒体输入（图片、视频、文字）生成持久的、可下载的 3D 空间。

这是世界模型走向商业化的一个重要信号。Marble 的思路和 Cosmos 不同——Cosmos 面向的是数据生成和训练，Marble 面向的是内容创作和空间计算。但从技术底层看，它们都在解决同一个问题：让 AI 理解三维物理世界。

## 二、重磅新玩家：AMI Labs

### LeCun 的 5 亿欧元豪赌

2026 年 1 月，Yann LeCun 在巴黎创办了 [AMI Labs](https://amigroup.ai/)（Advanced Machine Intelligence Labs），拿到了 5 亿欧元的投资。这是世界模型方向迄今为止最大的一笔产业投资。

AMI Labs 的技术路线基于 LeCun 多年来一直推崇的 [JEPA（Joint Embedding Predictive Architecture）](https://openreview.net/pdf?id=BZ5a1r-kVsf)。简单来说，JEPA 的核心思想是：不要在像素空间做预测（那是生成模型的思路），要在抽象表征空间做预测。

这和 Dreamer 系列的思路有相通之处——我在 [RSSM 详解](/zh/articles/rssm-deep-dive/)里讲过，RSSM 也是在隐空间做动态预测，而不是直接预测下一帧图像。但 JEPA 更进一步：它认为连"重建像素"这个目标都是错的，模型应该只关注"对任务有用的抽象特征"。

为什么值得关注？因为 LeCun 不是一般的研究者。他是深度学习的奠基人之一，他对技术方向的判断有很强的参考价值。他选择 all-in 世界模型，而且拿到了 5 亿欧元，说明他认为这条路已经到了产业化的临界点。

当然，5 亿欧元能不能做出东西，还要看执行。LeCun 的学术能力毋庸置疑，但做公司和做研究是两回事。

## 三、重要综述论文：建立全局视野

如果你只想读一篇论文来建立世界模型的全局认知，我推荐以下几篇 2026 年的综述：

### 《A Definition and Roadmap for World Models》

这篇 [arxiv 论文（2607.06401）](https://arxiv.org/html/2607.06401v1) 做了一件很有价值的事：给世界模型一个正式的定义，并画出了一张技术路线图。

世界模型这个概念现在被用得太滥了——视频生成模型叫世界模型，游戏引擎叫世界模型，甚至一些简单的预测模型也自称世界模型。这篇论文试图厘清：什么才算世界模型，什么不算，以及从当前状态到真正的世界模型还缺什么。

### 《World Model for Robot Learning: A Comprehensive Survey》

这篇 [Hugging Face 上的论文](https://huggingface.co/papers/2605.00080) 专注于世界模型在机器人学习中的应用。如果你关心的是"世界模型怎么用在真实机器人上"，这篇比上面那篇更有针对性。

它系统梳理了世界模型在感知、规划、控制三个环节的应用方式，以及不同机器人形态（机械臂、移动机器人、人形机器人）对世界模型的不同需求。

### 《A Comprehensive Survey on World Models for Embodied AI》

这篇 [GitHub 上有人维护了配套论文列表](https://github.com/Li-Zn-H/AwesomeWorldModels)，从具身智能的角度综述世界模型。它的特点是把世界模型放在了具身智能的大框架里来分析，讨论了世界模型和 VLA、强化学习、仿真器之间的关系。

## 四、技术趋势：三个值得注意的方向

### 趋势一：世界模型 + Transformer 的规模化

我在[之前的文章](/zh/articles/world-model-transformer/)里详细讨论过这个话题。2026 年的趋势是：越来越多的世界模型开始采用 Transformer 架构，而不是传统的 RNN/GRU。

Cosmos 用了 Transformer-based 的扩散模型，Genie 3 底层也是 Transformer，甚至 Dreamer 系列的后续工作也在探索用 Transformer 替换 RSSM 中的 GRU 组件。

原因很简单：Transformer 的规模化能力远超 RNN。当世界模型要从"实验室级别的简单任务"走向"真实世界的复杂场景"，模型参数量和训练数据量都需要大幅提升，而 Transformer 是目前唯一被验证过能大规模扩展的架构。

### 趋势二：VLA 和世界模型的融合加速

我在 [VLA vs 世界模型](/zh/articles/vla-vs-world-model/)那篇文章里预测过，2026-2027 年会出现越来越多的混合架构。现在看来，这个趋势正在加速。

Google 的 Gemini Robotics On-Device 就是一个典型案例——它用 VLA 做感知和语言理解，用世界模型做规划和预测。NVIDIA 的 GR00T 也在走类似的路线。

这种融合不是简单的"两个模块拼在一起"，而是在架构层面深度整合。比如，世界模型的隐状态表征可以直接作为 VLA 的条件输入，VLA 的语言接地能力可以指导世界模型的想象方向。

### 趋势三：从"能不能用"到"怎么用好"

2024-2025 年，世界模型的主要问题是"能不能 work"——能不能学会环境动态，能不能在想象中生成有用的数据。

2026 年，问题变成了"怎么用好"——怎么提高样本效率，怎么降低训练成本，怎么在真实机器人上稳定部署。这也是为什么 Cosmos 这样的工业级平台会出现，为什么 DreamerV3 的训练工程实践（我在[这篇文章](/zh/articles/2026-08-28-dreamerv3-training-tips/)里详细写过）变得和算法本身一样重要。

## 五、我的阅读建议

最后给一个实用的阅读清单。根据你目前的背景和目标，我建议按以下优先级来读：

**如果你刚入门世界模型：**

先读上面提到的三篇综述中的任意一篇，建立全局认知。然后回头读我这个博客的基础系列——从[什么是世界模型](/zh/articles/world-model-intro/)到 [RSSM 详解](/zh/articles/rssm-deep-dive/)到 [Dreamer 解读](/zh/articles/2026-08-25-dreamer-explained/)，把核心概念搞清楚。

**如果你已经在做世界模型相关研究：**

重点看 Cosmos 的技术报告和 Genie 3 的论文，了解工业界在怎么做大规模世界模型。然后看 AMI Labs 的后续进展——LeCun 的 JEPA 路线如果走通，可能会改变整个方向的技术范式。

**如果你关心工程落地：**

Cosmos 的开源工具链是第一优先级。然后看 World Model for Robot Learning 那篇综述里关于 Sim-to-Real 的章节。最后关注 NVIDIA 和 Google 在机器人部署方面的最新分享。

世界模型这个方向，2026 年是从"学术探索"走向"产业落地"的关键一年。上面的论文和项目，每一个都代表了这种趋势的一个切面。不需要全部读完，但至少要了解全局，然后在你关心的那个切面上深入下去。

---

*下一篇文章，我打算聊聊具身智能方向创业需要什么样的团队配置——不是那种大而全的商业计划书，而是一个工程师视角的务实分析。敬请期待。*
