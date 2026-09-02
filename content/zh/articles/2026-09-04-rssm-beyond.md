---
title: "从 RSSM 到现代状态空间模型：世界模型的引擎如何演进"
slug: "2026-09-04-rssm-beyond"
date: 2026-09-04
draft: false
categories: ["世界模型", "论文解读"]
tags: ["RSSM", "状态空间模型", "TD-MPC", "Mamba", "DreamerV3", "世界模型", "隐状态动力学"]
description: "RSSM 是 Dreamer 系列世界模型的核心引擎，但状态空间建模的版图在过去几年发生了很大变化。从 S4 到 Mamba 到 TD-MPC2，本文把 RSSM 放在更大的 state-space modeling 演进中审视，讨论专用动力学模型与通用序列架构之间的张力，以及世界模型引擎的可能演进方向。"
toc: true
related_articles:
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
  - 2026-09-01-world-model-h2-review
  - 2026-09-02-jepa-deep-dive
  - 2026-08-24-rssm-recap
---

在[之前的 RSSM 系列](/zh/articles/rssm-deep-dive/)中，我详细拆解了 RSSM 的架构细节——确定性路径与随机性路径的双轨设计、categorical latent 的选择、KL balancing 的训练技巧、以及 imagine 阶段的 reset 策略。

但那组文章主要聚焦在 RSSM 本身。今天想换一个视角：**把 RSSM 放在更大的状态空间建模（state-space modeling）演进中看，它的位置是什么？在它之后，世界模型的"引擎"正在往哪个方向走？**

## RSSM 的核心设计选择

先快速回顾。RSSM（Recurrent State-Space Model）的核心是一个双轨隐状态结构：

```
              h_t (deterministic)     s_t (stochastic)
                   │                        │
              GRU update              categorical prior
                   │                        │
                   └────────┬───────────────┘
                            ↓
                    p(o_t | h_t, s_t)
                     observation model
```

**确定性路径 h_t** 用 GRU 捕获长程依赖，**随机性路径 s_t** 用 categorical distribution 建模环境的不确定性。两条路径合在一起为 observation model 提供条件。

这个设计有几个值得注意的特点：

**第一，它不是纯粹的 neural state-space model。** 经典的 state-space model 形式是 z_{t+1} = f(z_t, a_t), o_t = g(z_t)。RSSM 更接近一个 **partially observable** 的隐变量模型——h_t 承担了一部分 observable history encoding 的角色，s_t 才是"真正的" latent state。

**第二，categorical latent 是一个工程导向的选择。** 不同于连续高斯 latent，categorical latent 让 prior 和 posterior 的 KL 散度可以精确计算，避免了连续分布中常见的 KL 估计方差问题。这是一个实用主义的设计决策，不是理论最优解。

**第三，imagination 是核心创新。** RSSM 不仅用于拟合历史数据，还用于在隐空间中"想象"未来轨迹——从当前 posterior 出发，用 prior .rollout 出多条未来路径，然后在想象中学策略。这让 Dreamer 系列在 sample efficiency 上持续领先。

这些设计选择在 DreamerV1 → V2 → V3 的演进中被逐步验证。但它们是唯一的路线吗？

## 状态空间建模的另一条线：从 S4 到 Mamba

在 RSSM 发展的同一时期，NLP 和序列建模领域出现了一条平行的 state-space model 路线。

**S4（Structured State Space for Sequences，2022）** 引入了结构化参数化的连续时间 state-space model。它的核心形式是经典的线性 SSM：

```
h'(t) = A h(t) + B x(t)
y(t) = C h(t) + D x(t)
```

但关键创新在于对 A 矩阵的结构化约束——通过对角化（HiPPO 初始化）让模型能够捕获超长程依赖。S4 在长序列基准上展示了 Transformer 级别的性能，但计算效率更高（线性复杂度）。

**Mamba（2024）** 在 S4 的基础上引入了 **selectivity**——让 SSM 的参数能够根据输入动态变化。这打破了经典 SSM 的"时不变"限制，让模型能够选择性地记住或遗忘信息。Mamba 在语言建模上达到了接近 Transformer 的性能，同时保持了 SSM 的线性推理效率。

这两条线和 RSSM 的关系是什么？

**形式上相似，但目标不同。** RSSM 和 S4/Mamba 都使用"隐状态 + 状态转移"的框架，但 RSSM 的隐状态是**环境动力学的低维表示**（world model 的 latent state），而 S4/Mamba 的隐状态是**序列信息的压缩表示**（sequence model 的 hidden state）。

换句话说：

```
RSSM：    隐状态 ≈ 世界的压缩描述（物理状态、动力学）
S4/Mamba：隐状态 ≈ 序列的压缩描述（上下文、语义）
```

这个区别很关键。RSSM 的隐状态被设计来回答"世界现在是什么状态、接下来会怎么变"；S4/Mamba 的隐状态被设计来回答"这段序列的上下文是什么、下一个 token 应该是什么"。

## TD-MPC2：不用 RSSM 也能做世界模型

TD-MPC2（2024，arXiv:2310.16828）代表了一种完全不同的世界模型设计哲学。

TD-MPC2 的世界模型不使用 RSSM 的双轨结构，而是用一个更简洁的架构：

```
Encoder:     e_t = E(o_t)                    → 将观察编码为 latent
Dynamics:    z_{t+1} = f_θ(z_t, a_t)         → MLP ensemble 预测下一个 latent
Reward:      r_t = R(z_t, a_t)               → 预测奖励
Termination: d_t = D(z_t)                     → 预测终止
```

**没有确定性/随机性双轨，没有 categorical latent，没有 KL balancing。** 它用的是一个更直接的 approach：encoder 把观察映射到 latent，MLP ensemble 直接在 latent 空间做 dynamics prediction，然后用 MPC（Model Predictive Control）在想象中选择动作。

TD-MPC2 的关键创新不在世界模型架构本身，而在三个方面：

**第一，MLP ensemble 的不确定性估计。** 使用多个 MLP 的预测分歧来估计模型不确定性，然后在 MPC 中用这个不确定性来引导探索。

**第二，跨任务/跨 embodiment 的 scalability。** TD-MPC2 展示了在 139 个任务、多种机器人形态上的可扩展性，这是 Dreamer 系列没有系统展示的。

**第三，latent space 的 consistency。** 通过 consistency loss 确保 encoder 和 dynamics model 在 latent space 中保持一致，避免了 RSSM 中 KL balancing 的调参负担。

从 RSSM 到 TD-MPC2，一个明显的趋势是：**世界模型的架构正在变得更简洁，但训练目标和 scaling 策略变得更重要。**

## 三条路线的对比

把 RSSM、S4/Mamba、TD-MPC2 放在一起，可以看到三种不同的"世界模型引擎"设计哲学：

| 维度 | RSSM (Dreamer) | S4 / Mamba | TD-MPC2 |
|------|---------------|------------|---------|
| **核心形式** | 双轨隐状态（确定性 + 随机性） | 连续/离散线性 SSM | Encoder + MLP dynamics |
| **隐状态含义** | 环境动力学表示 | 序列上下文压缩 | 任务相关的 latent |
| **主要训练目标** | reconstruction + KL + imagination reward | next-token / sequence prediction | reconstruction + consistency + MPC |
| **推理方式** | imagination → actor-critic | 序列前向传播 | latent-space MPC |
| **优势** | sample efficiency, long-horizon imagination | 长序列效率, scalability | 架构简洁, cross-task scaling |
| **局限** | 调参复杂, categorical 精度有限 | 不直接建模动力学 | 依赖 MPC, 长程规划受限 |

这三种路线不是互相替代的关系。它们解决的是不同层面的问题：

- **RSSM** 解决的是"如何在隐空间中同时捕获确定性动态和随机性，并支持 imagination-based policy learning"
- **S4/Mamba** 解决的是"如何高效处理超长序列，同时保持对关键信息的选择性记忆"
- **TD-MPC2** 解决的是"如何用简洁的架构实现跨任务、跨形态的世界模型 scaling"

## 融合的方向：世界模型需要什么样的序列架构？

一个自然的问题是：这些路线会不会汇合？

从最近的工作来看，有几个融合的趋势值得关注。

### 趋势一：VLA 正在借用 SSM 架构

一些新的 VLA 工作开始探索用 Mamba 类架构替代 Transformer backbone 来处理视觉-语言-动作序列。动机很直接：Transformer 的自注意力机制是 O(n²) 复杂度，对于需要处理长 observation history 的机器人策略来说，计算成本很高。

但这里有一个概念上的张力：**Mamba 的隐状态是序列上下文压缩，不是环境动力学表示。** 用 Mamba 做 VLA backbone 可以让模型更高效地处理长序列，但它不会自动获得 RSSM 那种"在隐空间中模拟物理动态"的能力。

从建模角度看，这恰好连接到了 VLA 系列中讨论的问题：VLA 缺的不是序列处理能力，而是显式的 action-conditioned prediction interface。把 Mamba 放进 VLA 可以改善序列处理效率，但不会自动让 VLA 变成一个 world model。

### 趋势二：世界模型开始使用 Transformer 架构

反过来，一些世界模型工作开始用 Transformer 替代 RSSM 的 GRU + categorical latent 结构。比如 IRIS（Implicit Representation of Images with Self-supervised Transformers）用 Transformer 在离散 token 空间做世界建模。

这种方向的优势是：Transformer 的注意力机制天然支持"关注过去的关键时间步"，不需要 RSSM 那样通过 h_t 来压缩所有历史信息。劣势是：计算成本随序列长度二次增长，对实时控制不友好。

### 趋势三： latent dynamics + foundation model 的混合架构

一个更有意思的方向是：**用 foundation model 做语义理解和表示，用 latent dynamics model 做物理预测。**

```
Foundation Model (Transformer / Mamba)
      ↓
  semantic representation
      ↓
Latent Dynamics Model (RSSM-style / TD-MPC-style)
      ↓
  action-conditioned future prediction
      ↓
  planning / policy
```

这种混合架构试图结合两者的优势：foundation model 提供强大的语义泛化能力，latent dynamics model 提供物理预测能力。

## RSSM 的遗产

回到最初的问题：RSSM 在状态空间建模演进中的位置是什么？

我觉得可以从三个层面来总结 RSSM 的贡献。

**第一，它证明了 latent imagination 的实用性。** 在 Dreamer 之前，"在想象中学策略"更多是一个理论概念。Dreamer 系列通过 RSSM + imagination 的完整工程实现，证明了这个思路在真实机器人任务上是可行的、甚至是 sample-efficient 的。

**第二，它提供了一个参考架构。** RSSM 的双轨设计（确定性 + 随机性）不是一个偶然选择，而是反映了一个深层的设计原则：**世界模型需要同时捕获可预测的动态和不可预测的不确定性。** 这个原则不会过时，即使具体实现（GRU、categorical latent）可能被替代。

**第三，它划定了一个设计空间。** RSSM 证明了"在隐空间做动力学预测 + 在想象中训练策略"是一条可行的路线。后续的工作——无论是 TD-MPC2 的简洁化、还是 Transformer-based world model 的架构替换——都是在 RSSM 划定的这个设计空间内做探索。

从更宏观的视角看，RSSM 之于世界模型，可能类似于 LSTM 之于序列建模——它不是最终架构，但它证明了一个关键概念，并为后续工作提供了设计模板。

## 开放问题

有几个问题我觉得还没有清晰的答案。

**隐状态应该是什么？** RSSM 用 categorical latent，TD-MPC2 用连续 latent，S4/Mamba 用确定性隐状态。哪种表示最适合世界模型？可能没有唯一答案——不同任务、不同 embodiment 可能需要不同的隐状态结构。

**动力学模型应该是专用的还是通用的？** RSSM 是专用的动力学模型（为环境建模而设计），Mamba 是通用的序列模型。世界模型的引擎应该是哪一种？混合架构可能是答案，但最优的组合方式还不清楚。

**scaling 会改变架构选择吗？** 在小数据 regime 下，RSSM 的归纳偏置（双轨结构、categorical prior）可能很重要。但在大数据 regime 下，更简洁的架构（如 TD-MPC2）可能因为更容易 scaling 而胜出。这是一个值得系统研究的问题。

**想象力的边界在哪里？** RSSM 的 imagination 在 DreamerV3 中已经可以 rollout 很长的轨迹。但想象力的质量会随着 rollout 长度衰减——这是一个根本性限制，还是可以通过更好的架构解决？

---

*这篇是 RSSM 系列的补充视角。如果想看 RSSM 的具体架构细节，可以参考[之前的 RSSM 深度拆解](/zh/articles/rssm-deep-dive/)和[源码六部曲](/zh/articles/2026-08-19-rssm-code-walkthrough/)。*

*下一篇是 VLA 系列的中篇——π₀ 家族与动作接口的演进。*
