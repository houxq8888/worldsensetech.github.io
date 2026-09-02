---
title: '从 RSSM 到现代 Latent Dynamics：世界模型的"引擎"如何演进'
slug: "2026-09-04-rssm-beyond"
date: 2026-09-04
draft: false
categories: ["世界模型", "论文解读"]
tags: ["RSSM", "状态空间模型", "TD-MPC", "Mamba", "DreamerV3", "世界模型", "隐状态动力学"]
description: "RSSM 是 Dreamer 系列世界模型的核心引擎，但状态空间建模的版图在过去几年发生了很大变化。本文把 RSSM 放在更大的 state-space modeling 演进中审视——区分 latent dynamics 与 sequence backbone 两个层面，讨论 TD-MPC2 的 planning + value 融合路线，以及世界模型引擎的可能演进方向。"
toc: true
related_articles:
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
  - 2026-09-01-world-model-h2-review
  - 2026-09-02-jepa-deep-dive
  - 2026-08-24-rssm-recap
---

> **关于本文的讨论范围：** 本文涉及的"状态空间建模"包含两个相关但不同的层面：一类是用于环境动力学建模的 latent state-space models（如 RSSM、TD-MPC2 的 latent dynamics），另一类是用于高效序列处理的 SSM architectures（如 S4、Mamba）。二者形式上相似，但不能直接等同。

在[之前的 RSSM 系列](/zh/articles/rssm-deep-dive/)中，我详细拆解了 RSSM 的架构细节——确定性路径与随机性路径的双轨设计、categorical latent 的选择、KL balancing 的训练技巧、以及 imagine 阶段的 reset 策略。

但那组文章主要聚焦在 RSSM 本身。今天想换一个视角：**把 RSSM 放在更大的状态空间建模（state-space modeling）演进中看，它的位置是什么？在它之后，世界模型的"引擎"正在往哪个方向走？**

## RSSM 的核心设计选择

先快速回顾。RSSM（Recurrent State-Space Model）的核心是一个双轨隐状态结构：

```
              h_t (deterministic)     z_t (stochastic)
                   │                        │
              GRU update              categorical prior
                   │                        │
                   └────────┬───────────────┘
                            ↓
                    p(o_t | h_t, z_t)
                     observation model
```

**确定性路径 h_t** 由 h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) 递归更新，负责积累历史信息。**随机性路径 z_t** 通过 categorical distribution 的 prior/posterior 建模当前状态的不确定性。两者共同构成 RSSM 的 **belief state** s_t = (h_t, z_t)，为 observation model 提供条件。

这个设计有几个值得注意的特点：

**第一，RSSM 更准确地说是 belief-state model。** 经典的 state-space model 形式是 z_{t+1} = f(z_t, a_t), o_t = g(z_t)。RSSM 则更接近一个 **partially observable** 的隐变量模型。从 POMDP 角度看，RSSM 的 recurrent state (h_t, z_t) 可以理解为对历史观测与动作形成的 belief state 的参数化——它并不是直接恢复环境的"真实物理状态"，而是学习一个足以支持预测和控制的 latent belief。

**第二，categorical latent 是一个工程导向的选择。** categorical latent 的一个工程优势是，prior/posterior 都是显式离散分布，KL 可以直接解析计算；同时它提供了比单个连续 Gaussian latent 更灵活的离散随机表示。DreamerV2/V3 进一步结合 straight-through estimator 使用这种 latent。这是一个实用主义的设计决策，不是理论最优解。

**第三，Dreamer 的关键贡献之一，是把 learned latent dynamics 与 actor-critic learning 结合起来，使 policy/value 可以主要在 imagined latent trajectories 上训练。** model-based RL 中的 imagination / model rollout 概念早于 Dreamer 就存在。Dreamer 的创新在于：RSSM 不仅用于拟合历史数据，还在隐空间中"想象"未来轨迹——从当前 posterior 出发，用 prior rollout 出多条未来路径，然后在想象 trajectories 上训练 actor 和 critic。这让 Dreamer 系列在 sample efficiency 上持续领先。

这些设计选择在 DreamerV1 → V2 → V3 的演进中被逐步验证。但它们是唯一的路线吗？

## 序列建模的另一条线：从 S4 到 Mamba

在 RSSM 发展的同一时期，NLP 和序列建模领域出现了一条平行的 state-space model 路线。**需要强调的是：S4/Mamba 首先是 sequence models，不是 world models。** 它们解决的是高效序列处理问题，而非 action-conditioned environment dynamics。

**S4（Structured State Space for Sequences，2022）** 引入了结构化参数化的连续时间 state-space model。它的核心形式是经典的线性 SSM：

```
h'(t) = A h(t) + B x(t)
y(t) = C h(t) + D x(t)
```

S4 的关键不是简单"做了一个对角化"，而是对经典 SSM 的状态矩阵 A 做**结构化参数化**——结合 HiPPO 初始化、低秩修正（low-rank correction）和正规化/对角化参数化，并通过高效 Cauchy kernel 计算实现——使长程记忆既可表达又能高效计算。S4 在长序列基准上展示了 Transformer 级别的性能，但计算效率更高。

**Mamba（2024）** 在 S4 的基础上引入了 **selectivity**——让 SSM 的关键参数（B_t, C_t, Δ_t）成为输入相关的，从而获得 **selective state space**。这意味着 state transition / information retention 可以根据当前 token 内容变化，让模型能够选择性地记住或遗忘信息。Mamba 在语言建模上达到了接近 Transformer 的性能，同时在序列长度上具有线性 scaling，并通过 selective scan 实现高效硬件执行。

这两条线和 RSSM 的关系是什么？

**形式上相似，但目标不同。** RSSM 和 S4/Mamba 都使用"隐状态 + 状态转移"的框架，但 RSSM 的隐状态是**对环境历史的任务相关 latent belief**——足以支持 observation prediction、reward prediction 和 control；而 S4/Mamba 的隐状态是**对输入序列历史的上下文压缩**——服务于 sequence prediction。

换句话说：

```
RSSM：
  隐状态 ≈ 对环境历史的任务相关 latent belief
         → 足以支持 observation / reward / control prediction

S4/Mamba：
  隐状态 ≈ 对输入序列历史的上下文压缩
         → 服务于 sequence prediction
```

这个区别很关键。RSSM 的隐状态被设计来回答"世界现在是什么状态、接下来会怎么变"；S4/Mamba 的隐状态被设计来回答"这段序列的上下文是什么、下一个 token 应该是什么"。

## TD-MPC2：另一种 latent dynamics 路线

TD-MPC2（2024，arXiv:2310.16828）代表了一种不同于 RSSM 的世界模型设计哲学。

TD-MPC2 不使用 RSSM 的双轨结构，而是用一个更简洁的架构：

```
Encoder:     e_t = E(o_t)                    → 将观察编码为 latent
Dynamics:    z_{t+1} = f_θ(z_t, a_t)         → 预测下一个 latent
Reward:      r_t = R(z_t, a_t)               → 预测奖励
Q-function:  Q(z_t, a_t)                     → 长期价值估计（5 个 Q ensemble）
Policy:      π(a_t | z_t)                    → policy prior
```

**没有确定性/随机性双轨，没有 categorical latent，没有 KL balancing。** 它用的是一个更直接的 approach：encoder 把观察映射到 latent，在 latent 空间做 dynamics prediction，然后结合短视 MPC 与长期 Q-value estimation 来选择动作。

TD-MPC2 的核心不是复杂的 latent-state decomposition，而是把**简洁的 latent dynamics、task-conditioned representation、短视 MPC 与长期 Q-value estimation** 组合起来。它的三个关键创新：

**第一，latent-space MPC 与 Q-function ensemble 的结合。** TD-MPC2 在 latent dynamics 上进行短视 rollout，并使用 Q-function ensemble（默认 5 个 Q-functions，TD target 使用随机子采样 Q-function 的 minimum）提供长期价值估计，从而把短期 MPC planning 与长期 TD bootstrapping 结合起来。这是 TD-MPC2 真正漂亮的地方。

**第二，task-conditioned cross-task / cross-embodiment scaling。** TD-MPC2 展示了在 139 个任务、多种机器人形态上的可扩展性。这不是"一个 dynamics model 自动理解所有 embodiment"，而是通过 **task embeddings / task-conditioned components** 让同一个网络适应不同任务/形态——encoder、dynamics、reward、policy prior、Q 等组件都与 task embedding 联系起来。

**第三，latent representation 的稳定化。** TD-MPC2 使用 SimNorm 对 latent state 做归一化，并通过联合训练 encoder、dynamics、reward、policy prior 和 Q-functions，使 latent representation 服务于预测与控制，而不需要 RSSM 那样的 stochastic prior/posterior KL 约束。

从 RSSM 到 TD-MPC2，一个明显的趋势是：**TD-MPC2 展示了另一种路线：不依赖复杂的 stochastic recurrent state，而是在 compact latent space 中结合 dynamics prediction、short-horizon MPC 与 long-horizon value estimation。**

## 四层架构：不只是"三条路线"

把上面讨论的模型放在一起，会发现它们并不处于同一层级。更准确的理解是一个四层架构：

```
                    Sequence / State Modeling
                              │
              ┌───────────────┴───────────────┐
              ↓                               ↓
      Sequence Backbone                 Latent Dynamics
      S4 / Mamba / Transformer          RSSM / MLP / Transformer
              │                               │
              │                               ↓
              │                     Action-conditioned prediction
              │                               │
              └───────────────┬───────────────┘
                              ↓
                         World Model
                              │
                    ┌─────────┴─────────┐
                    ↓                   ↓
                 Imagination            MPC + Value
                 Dreamer             TD-MPC2
                    ↓                   ↓
                    └─────────┬─────────┘
                              ↓
                           Policy
```

这个分层很重要：

- **S4/Mamba** 是 sequence engine——解决如何高效维护 state/history
- **RSSM** 是 latent dynamics engine——解决如何在隐空间建模环境动态
- **TD-MPC2** 是 latent dynamics + value + MPC system——把短期规划与长期价值结合
- **Dreamer** 是 latent dynamics + imagination + actor-critic system——在想象中训练策略

它们之间的关系不是"三条平行路线"，而是**不同层级的组件可以组合**。例如，Mamba 可以作为 VLA 的 sequence backbone，但不会自动让 VLA 变成 world model——只有当它被训练成关于环境状态、动作和未来状态之间关系的可查询预测模型时，才真正成为 world model。

### 对比表

| 维度 | RSSM / Dreamer | S4 / Mamba | TD-MPC2 |
|------|---------------|------------|---------|
| **核心形式** | 双轨隐状态（确定性 + 随机性） | 连续/离散线性 SSM | Encoder + MLP dynamics + Q ensemble |
| **隐状态含义** | 对环境历史的任务相关 latent belief | 序列上下文压缩 | task-conditioned latent |
| **主要训练机制** | observation/reward/continuation prediction + KL regularization；actor-critic 在 imagined trajectories 上训练 | sequence modeling objective | latent dynamics/reward + TD/Q learning + MPC |
| **推理方式** | imagination → actor-critic | 序列前向传播 | latent-space MPC + Q-value bootstrapping |
| **优势** | sample efficiency, long-horizon imagination | 长序列效率, scalability | planning + value 融合, task-conditioned cross-task scaling |
| **局限** | 调参复杂, categorical 精度有限 | 不直接建模动力学 | 规划依赖有限 horizon 的 latent MPC，长期决策依赖 learned Q-function 的 bootstrapping |

## 融合的方向：世界模型需要什么样的序列架构？

一个自然的问题是：这些路线会不会汇合？

从最近的工作来看，有几个融合的趋势值得关注。

### 趋势一：VLA 正在借用 SSM 架构

一些新的 VLA 工作开始探索用 Mamba 类架构替代 Transformer backbone 来处理视觉-语言-动作序列。动机很直接：在标准全局 self-attention 下，Transformer 的计算/内存成本随序列长度呈二次增长；对于需要处理长 observation history 的机器人策略来说，计算成本很高。实际 world model 往往通过 token compression、局部 attention 或其他结构缓解这一问题。

但这里有一个概念上的张力：**Mamba 的隐状态是序列上下文压缩，不是环境动力学表示。** 用 Mamba 做 VLA backbone 可以让模型更高效地处理长序列，但它不会自动获得 RSSM 那种"在隐空间中模拟物理动态"的能力。

从建模角度看，这恰好连接到了 VLA 系列中讨论的问题：VLA 缺的不是序列处理能力，而是显式的 action-conditioned prediction interface。把 Mamba 放进 VLA 可以改善序列处理效率，但不会自动让 VLA 变成一个 world model。

### 趋势二：世界模型开始使用 Transformer 架构

反过来，一些世界模型工作开始用 Transformer 替代 RSSM 的 GRU + categorical latent 结构。比如 IRIS（*Transformers are Sample-Efficient World Models*）用离散图像 tokenizer + autoregressive Transformer 构建世界模型，把 image dynamics 变成 token sequence modeling。

这种方向的优势是：Transformer 的注意力机制天然支持"关注过去的关键时间步"，不需要 RSSM 那样通过 h_t 来压缩所有历史信息。在标准全局 self-attention 下，计算/内存成本随序列长度呈二次增长；但实际 world model 往往通过 token compression、局部 attention 或其他结构缓解这一问题。

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

**第一，它证明了 latent imagination 用于行为学习的实用性。** Dreamer 系列通过 RSSM + imagination 的完整工程实现，证明了 learned latent dynamics 与 actor-critic learning 的结合在真实机器人任务上是可行的、甚至是 sample-efficient 的。

**第二，它提供了一个参考架构。** RSSM 的双轨设计（确定性 + 随机性）不是一个偶然选择，而是反映了一个深层的设计原则：**世界模型需要同时捕获可预测的动态和不可预测的不确定性。** 这个原则不会过时，即使具体实现（GRU、categorical latent）可能被替代。

**第三，它代表并系统化了"latent dynamics + imagination-based control"这一条重要设计空间。** RSSM 证明了"在隐空间做动力学预测 + 在想象中训练策略"是一条可行的路线。后续工作既可以在这个空间内替换 dynamics engine（如 TD-MPC2 用 MLP dynamics 替代 RSSM），也可以完全采用不同的 predictive representation（如 JEPA 的 latent prediction、video generative models、diffusion world models）。

如果做一个带有比喻性的类比，RSSM 在 latent-dynamics world model 中的地位，有些类似 LSTM 在经典序列建模中的地位：它未必是终点，但证明了一种重要的结构可以规模化工作。

## 开放问题

有几个问题我觉得还没有清晰的答案。

**隐状态应该是什么？** RSSM 用 categorical latent，TD-MPC2 用连续 latent，S4/Mamba 用确定性隐状态。哪种表示最适合世界模型？可能没有唯一答案——不同任务、不同 embodiment 可能需要不同的隐状态结构。

**动力学模型应该是专用的还是通用的？** RSSM 是专用的 latent dynamics model（为环境建模而设计），Mamba 是通用的 sequence model。世界模型的 sequence engine 应该是哪一种？混合架构可能是答案，但最优的组合方式还不清楚。

**scaling 会改变架构选择吗？** 一个值得验证的假设是：在数据有限、观测复杂或环境高度随机的 regime 中，显式的 stochastic latent structure 可能提供有价值的 inductive bias；而在数据与任务规模扩大后，更简洁、统一的 latent dynamics architecture 是否更容易 scaling，则需要系统实验验证。

**想象力的边界在哪里？** RSSM 的 imagination 在 DreamerV3 中已经可以 rollout 很长的轨迹。但想象力的质量会随着 rollout 长度衰减——这是一个根本性限制，还是可以通过更好的架构解决？

---

*这篇是 RSSM 系列的补充视角。如果想看 RSSM 的具体架构细节，可以参考[之前的 RSSM 深度拆解](/zh/articles/rssm-deep-dive/)和[源码六部曲](/zh/articles/2026-08-19-rssm-code-walkthrough/)。*

*下一篇是 VLA 系列的中篇——π₀ 家族与动作接口的演进。*
