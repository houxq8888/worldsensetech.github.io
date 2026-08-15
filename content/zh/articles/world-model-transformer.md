---
title: "世界模型架构演进：RSSM、Transformer 与统一世界模型"
slug: "world-model-transformer"
aliases:
  - /articles/world-model-transformer.html
date: 2026-08-12
draft: false
categories: ["世界模型"]
tags: ["世界模型", "Transformer", "RSSM", "UniSim", "Dreamer", "架构演进"]
description: "世界模型架构演进：RSSM、Transformer 与统一世界模型 - WorldSense 技术笔记"
toc: true
---


前面几篇文章中，我们深入讲解了 DreamerV3 的 RSSM 架构和训练技巧。RSSM 是强化学习世界模型中的经典设计，但如果你关注最近的研究，会发现一个明显的趋势：世界模型正在走向 Transformer 化。
 

从 Google 的 UniSim 到 Wayve 的 GAIA-1，从 NVIDIA 的 Cosmos 到国内具身智能团队的方案，Transformer 正成为大规模世界模型的重要技术路线。
 

为什么会这样？RSSM 有什么问题？Transformer 带来了什么优势？世界模型的架构演进方向是什么？
 

这篇文章梳理这条技术路线。
 

在展开之前，先澄清一个概念。所谓统一世界模型（unified world model），并不是指一个模型精确复制整个物理世界，而是希望通过统一的表示空间同时理解视觉、语言、动作和环境动态——让同一个模型既能"看"懂场景，又能"想"清因果，还能"做"出决策。这是当前世界模型研究的一个前沿方向。
 
## RSSM 的设计逻辑与局限
 

先回顾一下 RSSM 为什么被设计出来。
 

RSSM（Recurrent State-Space Model）的核心思想是将隐状态分成两个轨道：确定性状态（deterministic state）和随机性状态（stochastic state）。确定性状态用 GRU 更新，捕捉规律性的动态；随机性状态通过离散分类分布采样——DreamerV3 使用 32 个离散 categorical 变量，每个变量包含 64 个类别，组合形成容量为 64^32 的随机隐空间，捕捉环境的不确定性。
 

这个设计有几个优点：
 

计算效率高。GRU 是循环网络，每一步只需要上一步的隐状态，不需要处理整个序列。训练和推理的内存开销较小。
 

隐空间结构清晰。确定性 + 随机性的双轨道设计，让模型同时学到"规律"和"不确定性"。这对物理世界的建模很重要——宏观物理规律通常具有较强确定性，而观测和环境扰动具有随机性。
 

想象训练方便。在隐空间中可以方便地"想象"未来轨迹：从当前状态出发，用 GRU 逐步推进，每一步采样随机状态，得到一系列隐状态，再解码成观测。
 

但 RSSM 也有几个明显的局限：
 

长程依赖问题。RSSM 通过 deterministic hidden state 维护历史信息，但由于状态更新仍是递归形式，超长时间跨度的信息需要压缩到固定维度的隐状态中，可能造成信息瓶颈。当任务需要记住很久以前的信息时（如延迟奖励、长程因果链），RSSM 的表现可能不够好。
 

并行化受限。循环网络在时间维度上是序列化的——每一步依赖上一步的输出。不过 RSSM 训练并非完全串行：observation encoder 可以并行处理，rollout imagination 也可以在 batch 维度并行。真正的瓶颈在于时间维度上的依赖关系，这使得训练速度受限于序列长度。
 

Scaling 体系差距。Transformer 更容易继承语言模型时代的大规模训练范式——大规模预训练经验、数据规模优势、工业基础设施成熟。而 RSSM 虽然也能扩大（如 DreamerV3 的 4096 维 deterministic state），但尚未形成同等规模的 scaling 体系。
 
## Transformer 世界模型的兴起
 

Transformer 在 LLM 中的成功，自然让人想到把它用到世界模型上。Transformer 用于世界模型的核心思路是：
 

将过去的观测和动作序列作为输入，用 Transformer 预测下一步的观测（或隐状态），本质上是一个序列到序列的预测问题。
 

这个思路有几个天然的优势：
 

长程依赖。Self-attention 可以直接建立任意时间位置之间的连接，因此缓解了循环结构中的长程信息传递瓶颈。对于需要长程记忆的任务，Transformer 理论上更强——当然，实际中仍受 context length 限制和 attention 计算成本的约束。
 

并行训练。Transformer 在训练阶段可以沿序列维度高度并行计算，相比循环网络减少了时间依赖带来的串行瓶颈。这意味着在 GPU 上训练更快，更容易利用大规模算力。
 

Scaling law。Transformer 在 LLM 中展示了良好的 scaling law。如果世界模型也用 Transformer，那么理论上可以借鉴 LLM 的 scaling 经验——增大模型、增加数据，性能持续提升。
 

和 LLM 的统一。如果世界模型也用 Transformer，那么它和 LLM 的架构是统一的。这意味着可以复用 LLM 的基础设施（训练框架、推理优化），甚至可以将语言理解和物理预测统一在一个模型中。
 

除了这些通用优势，世界模型还有几个独有的理由需要 Transformer：
 

空间维度。世界模型的输入通常是视频（time × height × width），Transformer 天然适合将图像切分为 patch token 进行建模。相比循环网络逐帧处理，Transformer 可以同时关注空间和时间两个维度的依赖关系。
 

多模态统一。世界模型的输入包括图像、语言指令、动作、本体感觉等多种模态。Transformer 的 token 化框架天然适合统一处理这些异构输入——每种模态编码为 token 序列后，用同一套 attention 机制处理。
 

长期规划。机器人任务往往需要跨时间的长期规划——当前动作可能影响几分钟后的目标达成。Transformer 的跨时间 attention 机制可以直接建立"现在动作"和"未来目标"之间的关联，这是循环网络难以高效实现的。
 

目前 Transformer 世界模型可以从两个维度来分类：
 

按预测空间：pixel space（直接在视觉空间生成，如早期视频预测模型）、latent space（在压缩表示空间中预测，Dreamer 系列属于这一类 latent dynamics model）、token space（将观测离散化为 token 后预测，如 Genie）。
 

按生成方式：autoregressive（自回归逐步生成，如 Genie、VideoPoet）、diffusion（条件扩散生成，如 DIAMOND、Cosmos）、masked prediction（掩码预测，类似 BERT 的 bidirectional 方式）。
 

这两个维度可以组合——例如 Cosmos 代表了 diffusion transformer 与 latent video modeling 相结合的一类路线。GAIA-1、Genie 等工作则属于 latent/token space + autoregressive 的组合。选择哪种组合，取决于任务对实时性、生成质量和计算效率的要求。
 
## 代表性工作
 

下面介绍几个 Transformer 世界模型的代表性工作。
 
### 1. UniSim（Google，2023-2024）
 

UniSim 是 Google 提出的统一仿真器（Universal Simulator）项目。它的核心思想是：利用大规模生成模型构建通用环境模拟器——基于视频生成模型和条件生成，输入"观测 + 动作 + 条件"，输出"下一步的观测"。条件可以是文本指令、目标图像、或者其他控制信号。
 

与 RSSM 这类显式 latent dynamics model 不同，UniSim 更偏向生成式环境模拟——它不显式建模隐状态转移，而是通过大规模视频生成来隐式地"学会"环境动态。UniSim 展示了利用大规模生成模型构建统一环境模拟器的可能性，不过它尚未达到"一个模型覆盖所有物理世界"的程度，更多是一个方向性的探索。
 
### 2. GAIA-1（Wayve，2023）
 

GAIA-1 是自动驾驶公司 Wayve 提出的生成式世界模型。它包含多模态 tokenizer、生成模型和世界表示三个核心模块，输入过去的视频帧和动作，预测未来的视频帧。
 

GAIA-1 的特别之处在于规模——公开资料显示为约 90 亿参数，在大量真实驾驶数据上训练。生成的视频质量相当高，能捕捉天气变化、光照变化、交通参与者行为等复杂动态。
 

GAIA-1 展示了生成式世界模型在真实驾驶数据上的扩展潜力。这是 LLM 的 scaling 思路在世界模型上的尝试。
 
### 3. Cosmos（NVIDIA，2025）
 

Cosmos 是 NVIDIA 发布的世界模型基础模型平台。Cosmos 系列采用 Transformer-based generative architecture，包括 diffusion transformer 和 autoregressive transformer 等路线，提供预训练的视频生成和物理仿真能力，支持在自定义数据上微调。Cosmos 的目标不是单纯预测视频，而是构建面向 Physical AI 的基础世界模型，为机器人和自动驾驶提供可学习的环境先验。
 

Cosmos 的推出说明：在工业界，世界模型正在从研究原型走向工程化平台。
 
### 4. DIAMOND（2024）
 

DIAMOND（DIffusion As a Model of eNvironment Dreams）将 diffusion model 引入世界模型。它把环境动态建模为一个条件扩散过程——给定过去的观测和动作，用扩散模型生成未来环境状态。DIAMOND 在 Atari 游戏上取得了很好的结果，展示了 diffusion-based 世界模型的潜力。
 
## RSSM vs Transformer：不是简单的替代
 

虽然 Transformer 有很多优势，但说它"替代"RSSM 可能过于简单。两者各有适用场景。下面从几个关键维度做对比：

| 维度 | RSSM（DreamerV3） | Transformer 世界模型 |
| --- | --- | --- |
| 序列建模 | GRU 循环，逐步更新隐状态 | Self-attention，全局并行计算 |
| 并行训练 | 时间维度序列化（encoder/rollout 可并行） | 沿序列维度高度并行 |
| 长程依赖 | 受限于固定隐状态容量 | 强（直接 attention） |
| 隐表示 | 结构化 latent state（deterministic + stochastic） | token / latent representation |
| 数据需求 | 小中规模数据友好 | 通常需要大规模预训练 |
| 代表工作 | DreamerV3 | UniSim, GAIA-1, Cosmos |

数据量。Transformer 通常需要大量数据才能发挥优势。如果数据有限（如只有几千条机器人操作数据），RSSM 的归纳偏置（循环 + 隐状态）可能更有效——它用更少的参数就能学到合理的动态。
 

实时性。RSSM 的循环结构在推理时是逐步推进的，每一步只需要上一步的隐状态。而 Transformer 推理通常需要维护历史 context，实际系统会通过 KV cache、窗口 attention 或 memory token 来降低成本。在对实时性要求很高的场景中，RSSM 可能更合适。
 

任务类型。如果任务需要长程记忆或全局推理（如导航、策略规划），Transformer 的 self-attention 更有优势。如果任务是短期的、局部的（如快速的操作反应），RSSM 可能就够用了。
 

计算资源。Transformer 的训练需要大量 GPU 资源。在相同计算预算下，RSSM 通常更容易部署到资源受限设备——这也是为什么 DreamerV3 在单张 GPU 上就能训练。
 

所以，更准确的说法是：Transformer 正在成为世界模型的重要方向，但 RSSM 仍有其价值。选择哪种架构，取决于具体的任务、数据和资源条件。
 
## 混合架构：两全其美？
 

既然 RSSM 和 Transformer 各有优劣，一个自然的问题是：能不能把两者结合起来？
 

已经有一些工作在探索这个方向：
 

Transformer + RSSM。用 Transformer 替代 RSSM 中的 GRU（确定性状态更新），保留随机性状态的离散分类分布设计。Dreamer 系列的核心一直是 CNN/MLP encoder + RSSM，但后续探索正在尝试用 Transformer 替代 GRU 做序列建模。这样既获得了 Transformer 的长程建模能力，又保留了 RSSM 的隐空间结构。
 

分层架构。底层用 RSSM 做短期的动态预测，高层用 Transformer 做长程的规划和推理。这种分层设计在 LLM 中也有类似的思路（如局部 attention + 全局 attention），在机器人控制中尤其有潜力——短期反应交给高效的循环网络，长期规划交给强大的 Transformer。
 

Token 化 + Transformer。将观测（如视频帧）离散化为 token，然后用 Transformer 在 token 序列上做预测。DeepMind 的 Genie（2024）就是这一思路的代表——它包含视频 tokenizer、latent action model、dynamics model 和 decoder 四个模块，其中 dynamics model 使用 Transformer 预测下一帧的 token 序列，实现了从单张图片生成可交互的 2D 环境。这种方式将世界模型和 LLM 的架构统一起来，是通往"通用世界模型"的一条可能路径。
 

VLM + World Model。未来世界模型可能不会独立存在，而是作为视觉语言行动模型（VLA）的内部预测模块。模型同时具备语言理解、场景理解、动态预测和动作规划能力——输入图像、语言指令和状态，通过 foundation model 预测未来状态，再由规划模块或策略模型输出动作。这是 Physical AI 的重要方向，也是世界模型从独立模块走向统一架构的关键一步。
 

混合架构是一个活跃的研究方向，目前还没有明确的最优方案。但从趋势看，Transformer 的元素正在越来越多地出现在世界模型中。
 
## 对从业者的启示
 

世界模型的 Transformer 化趋势，对从业者有几个启示：
 

学习 Transformer。如果你之前只了解 RSSM，现在是时候深入学习 Transformer 了。Self-attention、位置编码、KV cache 等概念，在世界模型中越来越重要。
 

关注 scaling。Transformer 世界模型的性能很大程度上取决于模型规模和数据量。关注 scaling law 在世界模型中的表现，理解"多大模型 + 多少数据 = 什么性能"。
 

不要放弃 RSSM。RSSM 在资源受限、数据有限的场景中仍有优势。DreamerV3 的训练技巧和经验，在 Transformer 世界模型中也有参考价值（如想象训练、KL 正则化等）。
 

关注统一架构。世界模型和 LLM 的边界正在模糊。未来可能出现"统一的 Physical AI 模型"——同时理解语言、视觉、动作。理解这个趋势，有助于把握长期的技术方向。
 
## 小结
 

世界模型正在进入多路线并存阶段：RSSM 代表的紧凑 latent dynamics、Transformer 代表的大规模序列建模、Diffusion 代表的生成式预测，以及 token-based foundation model 正在融合。Transformer 的长程建模、并行训练、scaling law 等优势，使其成为大规模世界模型的重要技术路线。UniSim、GAIA-1、Cosmos 等工作已经展示了这一方向的潜力。
 

但这不意味着 RSSM 过时了。在数据有限、资源受限、实时性要求高的场景中，RSSM 仍有其价值。混合架构可能是未来的方向——结合两者的优势，兼顾效率和性能。
 

对于从业者，关键是理解不同架构的设计逻辑和适用场景，而不是简单地追新。世界模型的核心问题——如何高效地学习和预测物理动态——是不变的，架构只是解决问题的工具。
