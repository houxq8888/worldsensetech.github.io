---
title: '从 RSSM 到现代 Latent Dynamics：世界模型的"引擎"如何演进'
slug: "2026-09-04-rssm-beyond"
date: 2026-09-04
draft: false
categories: ["世界模型", "论文解读"]
tags: ["RSSM", "状态空间模型", "TD-MPC", "Mamba", "DreamerV3", "世界模型", "隐状态动力学"]
description: "RSSM 是 Dreamer 系列世界模型的核心引擎，但状态空间建模的版图在过去几年发生了很大变化。本文把 RSSM 放在更大的 state-space modeling 演进中审视——区分 latent dynamics 与 sequence backbone 两个层面，讨论 TD-MPC2 的 decoder-free latent world model 路线和 planning + value 融合，以及世界模型引擎从'生成世界'到'提供预测接口'的设计趋势。"
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

**确定性路径 h_t** 由 h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) 递归更新，负责积累历史信息。**随机性路径 z_t** 通过 categorical distribution 的 prior/posterior 建模当前状态的不确定性。两者共同构成 RSSM 的 latent state s_t = (h_t, z_t)，为 observation model 提供条件。从 POMDP 视角看，这个 latent state 可以被理解为对历史观测与动作所形成 belief state 的一种参数化——它并不是直接恢复环境的"真实物理状态"，而是学习一个足以支持预测和控制的 latent belief representation。

这个设计有几个值得注意的特点：

**第一，RSSM 更准确地说是 belief-state model。** 经典的 state-space model 形式是 z_{t+1} = f(z_t, a_t), o_t = g(z_t)。RSSM 则更接近一个 **partially observable** 的隐变量模型。RSSM 的 recurrent state (h_t, z_t) 不是精确的 Bayesian belief state，而是其参数化近似。

**第二，categorical latent 是一个工程导向的选择。** categorical latent 的工程优势是，prior/posterior 都是显式离散分布，KL 可以直接解析计算；同时它提供了一种离散的随机表示，允许 prior/posterior 直接在 categorical distribution 上进行 KL 计算。DreamerV2 引入 categorical stochastic latent 并结合 straight-through estimator；DreamerV3 延续了这一表示方式。这是一个实用主义的设计决策，不是理论最优解。

**第三，Dreamer 的核心训练机制，是把 learned latent dynamics 与 actor-critic learning 结合起来，使 policy/value 可以主要在 imagined latent trajectories 上训练。** model-based RL 中的 imagination / model rollout 概念早于 Dreamer 就存在。Dreamer 的创新在于：RSSM 不仅用于拟合历史数据，还在隐空间中"想象"未来轨迹——从当前 posterior 出发，用 prior rollout 出多条未来路径，然后在想象 trajectories 上训练 actor 和 critic。这使 Dreamer 系列成为 latent imagination + actor-critic 路线中最具代表性的工作之一，并在多个视觉控制与强化学习 benchmark 上展示了很强的样本效率。

这些设计选择在 DreamerV1 → V2 → V3 的演进中被逐步验证。但它们是唯一的路线吗？

## 序列建模的另一条线：从 S4 到 Mamba

在 RSSM 发展的同一时期，NLP 和序列建模领域出现了一条平行的 state-space model 路线。**需要强调的是：S4/Mamba 首先是 sequence models，不是 world models。** 它们解决的是高效序列处理问题，而非 action-conditioned environment dynamics。

**S4（Structured State Spaces，2021/2022）** 引入了结构化参数化的连续时间 state-space model。它的核心形式是经典的线性 SSM：

```
h'(t) = A h(t) + B x(t)
y(t) = C h(t) + D x(t)
```

S4 的关键在于对 state matrix A 进行结构化参数化，并结合 HiPPO 初始化与低秩修正，使其能够稳定、高效地计算长程卷积。S4 在长序列基准上展示了 Transformer 级别的性能，但计算效率更高。

**Mamba（2023/2024）** 在 S4 的基础上引入了 **selectivity**——让 SSM 的关键参数（B_t, C_t, Δ_t）成为输入相关的，从而获得 **selective state space**。这意味着 state transition / information retention 可以根据当前 token 内容变化，让模型能够选择性地记住或遗忘信息。Mamba 在论文报告的语言建模实验中表现出与同规模 Transformer 竞争的性能，同时在序列长度上具有线性 scaling，并通过 selective scan 实现高效硬件执行。

### RSSM vs S4/Mamba：不是"表示世界 vs 表示上下文"

这两条线和 RSSM 的关系是什么？

**形式上相似，但目标和训练接口不同。** RSSM 和 S4/Mamba 都使用"隐状态 + 状态转移"的框架，但 RSSM 显式定义了 action-conditioned latent transition，以及与 observation/reward 相关的预测模型，因此它的 latent state 被训练成能够支持环境预测和控制的表示。S4/Mamba 则首先是一类通用 sequence architecture；它们的 hidden state 本身并不具有"上下文"或"世界状态"的固定语义，而是由具体训练目标决定。

因此，更准确的区别不是"RSSM 表示世界、Mamba 表示上下文"，而是：

```
RSSM：
  latent state + action-conditioned transition
  → environment prediction / imagination / control

S4/Mamba：
  recurrent/SSM hidden state
  → sequence processing
```

Mamba 的 hidden state 可以被用作高效的序列历史压缩状态，但它本身并不预设这一状态具有"上下文"或"世界状态"的语义。如果把 Mamba 训练成 action-conditioned latent dynamics，它同样可以成为 world-model engine；反之，RSSM 的 recurrent state 也可以被理解为一种特殊的 sequence state representation。**架构本身不决定语义，训练接口才决定。**

## TD-MPC2：另一种 latent dynamics 路线

TD-MPC2（2024，arXiv:2310.16828）是一个 model-based RL system，其核心 world model 采用 decoder-free latent dynamics。

TD-MPC2 不使用 RSSM 的双轨结构，而是用一个更简洁的架构：

```
Encoder:     e_t = E(o_t)                    → 将观察编码为 latent
Dynamics:    z_{t+1} = f_θ(z_t, a_t)         → 预测下一个 latent
Reward:      r_t = R(z_t, a_t)               → 预测奖励
Q-function:  Q(z_t, a_t)                     → 长期价值估计（5 个 Q ensemble）
Policy:      π(a_t | z_t)                    → policy prior
```

**没有确定性/随机性双轨，没有 categorical latent，没有 KL balancing。** 它用的是一个更直接的 approach：encoder 把观察映射到 latent，在 latent 空间做 dynamics prediction，然后结合短视 MPC 与长期 Q-value estimation 来选择动作。

### Decoder-free：从"生成世界"到"服务控制"

这里出现了一个非常重要的分野。

Dreamer 中通过 observation prediction 约束 latent representation 的路线是：

```
observation → latent → dynamics → predict observation
                                        ↓
                                   imagination
```

TD-MPC2 的路线是：

```
observation → encoder → z_t → latent dynamics → ẑ_{t+1}
                                │                    ↕
                          ┌──────┼──────        consistency
                          ↓      ↓      ↓         loss
                        reward   Q    policy    ↕
                                          encoder(o_{t+1})
```

**与 Dreamer 中通过 observation prediction 约束 latent representation 的路线不同，TD-MPC2 明确采用 decoder-free 的 implicit world model：它不要求 latent 能够重建像素，而是通过 latent consistency、reward prediction 和 value learning，让表示直接服务于控制目标。**

### 一个更深的视角：decision-sufficient vs observation-sufficient

这个区别值得进一步展开。

```
Generative world model (Dreamer)

z_t
 ↓
p(o_{t+1:t+H} | z_t, a_{t:t+H})
 ↓
future observations
 → imagination → actor-critic


Control-oriented world model (TD-MPC2)

z_t
 ↓
p(z_{t+1} | z_t, a_t)
 ↓
reward / value / terminal
 ↓
action selection (MPC)
```

前者要求 latent 保留足够的信息来生成未来观测；后者只要求 latent 保留对决策有用的信息。二者的 representation pressure 并不相同。TD-MPC2 优化的是 **decision-sufficient representation**，而不是 observation-sufficient representation——这解释了为什么它可以不做 decoder。

从本文采用的功能视角来看，这可以理解为一个从"生成世界"向"提供预测接口"的设计趋势。这是一种设计趋势，而不是领域共识——generative 和 control-oriented 两条路线目前明显并存。但这引出一个核心问题：**world model 不一定需要成为一个更强的"视频生成器"，更关键的问题是：它需要提供什么样的 action-conditioned predictive interface，才能以最低的计算与数据成本支持 planning、value estimation 或 policy learning。**

### TD-MPC2 的设计重点

TD-MPC2 的核心不是复杂的 latent-state decomposition，而是把**简洁的 latent dynamics、task-conditioned representation、短视 MPC 与长期 Q-value estimation** 组合起来。可以把 TD-MPC2 的设计重点概括为三个方面：

**第一，latent-space MPC 与 Q-function ensemble 的结合。** TD-MPC2 的显式 MPC planning horizon 很短（默认 3 steps），因此它并不是通过长 rollout 直接完成长期规划，而是让 learned Q-function 在规划边界处提供 long-term value bootstrap。**换言之，TD-MPC2 把"长期规划"从模型 rollout 问题部分转移成了 value estimation 问题。** 具体地，TD-MPC2 在 latent dynamics 上进行短视 rollout，并使用 Q-function ensemble（默认 5 个 Q-functions，TD target 使用随机子采样 Q-function 的 minimum）提供长期价值估计，从而把短期 MPC planning 与长期 TD bootstrapping 结合起来。

**第二，task-conditioned cross-task / cross-embodiment scaling。** TD-MPC2 在 104 个连续控制任务上进行评估，并进一步展示了一个 317M 参数的单一 agent 可以在 80 个任务上进行训练，覆盖不同任务、embodiment 和 action space。它并不是让一个 dynamics model 在没有条件信息的情况下"自动理解所有 embodiment"，而是通过 **task embeddings / task-conditioned components** 让同一套模型适应不同任务——encoder、dynamics、reward、policy prior、Q 等组件都与 task embedding 联系起来。

**第三，latent representation 的稳定化。** TD-MPC2 使用 SimNorm 对 latent state 做归一化，并通过联合训练 encoder、dynamics、reward、policy prior 和 Q-functions，使 latent representation 服务于预测与控制，而不需要 RSSM 那样的 stochastic prior/posterior KL 约束。

从 RSSM 到 TD-MPC2，一个明显的趋势是：**TD-MPC2 展示了另一种路线：不依赖复杂的 stochastic recurrent state，而是在 compact latent space 中结合 dynamics prediction、short-horizon MPC 与 long-horizon value estimation。**

## Architecture ≠ Function：从 Sequence Backbone 到 Control

把上面讨论的模型放在一起，会发现它们并不处于同一层级。更重要的是，**同一个 architecture 可以占据不同的功能角色**。与其用"四层架构"来描述，不如用一个二维 taxonomy：

**横轴：architecture**（Recurrent / SSM / Transformer / MLP）

**纵轴：functional role**（sequence backbone → latent dynamics → prediction interface → planning / policy）

```
Architecture \ Role    Seq Backbone    Latent Dynamics    Planning / Policy
GRU                    ✓ (RNN)         RSSM               ✓ (actor)
MLP                    —               TD-MPC2            —
Transformer            ✓ (GPT etc.)    IRIS / WM          ✓ (policy)
Mamba                  ✓ (SSM)         可以               ✓ (policy)
```

这张表表达了全文的核心观点：**Architecture ≠ function。** RSSM 用 GRU 做 latent dynamics，Mamba 也可以用做 latent dynamics——区别不在 architecture，而在训练接口和功能角色。

因此，"world model"更适合作为一种功能接口，而不是一种固定网络结构：它至少需要提供某种关于未来状态、观测、奖励或价值的可查询预测能力，并能够被用于决策、规划或策略学习。从这个角度看，RSSM、TD-MPC2、Transformer world model 甚至某些 JEPA-style predictive models 都可以属于 world-model family，但它们提供的 prediction interface 并不相同。

### 对比表

| 维度 | RSSM-based Dreamer | S4 / Mamba | TD-MPC2 |
|------|-------------------|------------|---------|
| **world-model role** | explicit latent dynamics | 本身不是 world model | implicit latent dynamics |
| **状态结构** | deterministic + stochastic latent | SSM hidden state | continuous latent + task conditioning |
| **是否 action-conditioned** | 是 | 默认不是 | 是 |
| **observation decoder** | Dreamer 中通常有 | 取决于具体任务 | 无（decoder-free） |
| **核心预测接口** | observation / reward + latent transition | 取决于训练 objective | latent transition + reward/value |
| **控制方式** | imagined actor-critic | 非控制算法本身 | latent MPC + Q bootstrap |
| **主要优势** | imagination / sample efficiency | long sequence efficiency | planning + value + multitask scaling |
| **主要限制** | stochastic latent / training complexity | 不天然提供 dynamics semantics | short MPC horizon + Q bootstrap |

## 融合的方向：世界模型需要什么样的序列架构？

一个自然的问题是：这些路线会不会汇合？

从最近的工作来看，有几个融合的趋势值得关注。

### 趋势一：VLA 正在借用 SSM 架构

VLA 领域已经出现直接采用 Mamba/SSM backbone 的工作。RoboMamba（NeurIPS 2024）是代表性案例之一——它将视觉编码与 Mamba 结合用于 vision-language-action reasoning，并在仿真和真实机器人实验中验证了其效率；近期工作也开始进一步探索将 selective SSM 用于 VLA 的 action expert。

动机很直接：标准 full self-attention 在训练时的 attention 计算与上下文长度呈 O(L²) 扩展；自回归推理虽然可以通过 KV cache 避免每一步重新计算整个 attention，但 KV cache 的内存仍随上下文长度增长。对需要持续处理长 observation history 的机器人策略而言，这会形成明显的计算和内存压力。

但从建模角度看，这恰好连接到了 VLA 系列中讨论的问题：VLA 缺的不是序列处理能力，而是显式的 action-conditioned prediction interface。把 Mamba 放进 VLA 可以改善序列处理效率，但不会自动让 VLA 变成一个 world model。

### 趋势二：世界模型开始使用 Transformer 架构

反过来，一些世界模型工作开始用 Transformer 替代 RSSM 的 GRU + categorical latent 结构。比如 IRIS（*Transformers are Sample-Efficient World Models*）用离散图像 tokenizer + autoregressive Transformer 构建世界模型。IRIS 并不是简单地把 RSSM 的 recurrent state 换成 Transformer，而是同时改变了 representation 和 dynamics：先把图像压缩成离散 token，再在 token 序列上用 autoregressive Transformer 建模——这是完全不同的 world-model design。IRIS 主要验证于 Atari 环境，因此这里更适合作为"Transformer world model"这一架构路线的代表，而不是直接的机器人 world-model benchmark。

这种方向的优势是：Transformer 的显式 attention 提供了跨时间位置的直接交互能力，因此模型可以根据当前 prediction target 动态利用不同历史位置的信息，而不必像固定维度 recurrent state 那样把历史压缩进单一递归状态。在标准全局 self-attention 下，训练时 attention 计算与上下文长度呈 O(L²) 扩展；但实际 world model 往往通过 token compression、局部 attention 或其他结构缓解这一问题。

### 趋势三： latent dynamics + foundation model 的混合架构

一个更有意思的方向是：**用 foundation model 做语义/视觉表示，再由专门的 latent dynamics 模块承担 action-conditioned future prediction。**

```
Foundation Model (Transformer / Mamba)
      ↓
  semantic / visual representation
      ↓
Latent Dynamics Model (RSSM-style / TD-MPC-style)
      ↓
  action-conditioned future prediction
      ↓
  planning / policy
```

这种混合架构试图让 foundation model 提供更强的语义/视觉表示，再由专门的 latent dynamics 模块承担 action-conditioned future prediction；但二者之间如何接口、哪些信息需要从 foundation model 保留到 dynamics latent，仍然是开放问题。

## RSSM 的遗产

回到最初的问题：RSSM 在状态空间建模演进中的位置是什么？

我觉得可以从三个层面来总结 RSSM 的贡献。

**第一，它证明了 latent imagination 用于行为学习的实用性。** Dreamer 系列通过 RSSM + imagination 的完整工程实现，证明了 learned latent dynamics 与 actor-critic learning 的结合可以在多种视觉控制与强化学习任务中取得很高的样本效率。

**第二，它提供了一个参考架构。** RSSM 的双轨设计体现了一个重要的建模原则：**对于部分可观测或未来具有多模态性的环境，显式表示不确定性往往有价值。** 这个设计维度不会过时，即使具体实现（GRU、categorical latent）可能被替代。

**第三，它代表并系统化了"latent dynamics + imagination-based control"这一条重要设计空间。** RSSM 证明了"在隐空间做动力学预测 + 在想象中训练策略"是一条可行的路线。后续工作既可以在这个空间内替换 dynamics engine（如 TD-MPC2 用 MLP dynamics 替代 RSSM），也可以完全采用不同的 predictive representation（如 JEPA 的 latent prediction、video generative models、diffusion world models）。

如果做一个带有比喻性的类比，RSSM 更像 latent-dynamics world model 中的一个 **canonical reference architecture**：它未必是最终形态，但它把"随机 latent + recurrent state + learned dynamics + imagination"这一整套范式具体化并证明了工程可行性。

## 开放问题

有几个问题我觉得还没有清晰的答案。

**隐状态应该是什么？** RSSM 用 categorical latent，TD-MPC2 用连续 latent，S4/Mamba 用确定性隐状态。哪种表示最适合世界模型？可能没有唯一答案——不同任务、不同 embodiment 可能需要不同的隐状态结构。

**动力学模型应该是专用的还是通用的？** RSSM 是专用的 latent dynamics model（为环境建模而设计），Mamba 是通用的 sequence model。世界模型的 sequence engine 应该是哪一种？混合架构可能是答案，但最优的组合方式还不清楚。

**scaling 会改变架构选择吗？** 一个值得验证的假设是：在数据有限、观测复杂或环境高度随机的 regime 中，显式的 stochastic latent structure 可能提供有价值的 inductive bias；而在数据与任务规模扩大后，更简洁、统一的 latent dynamics architecture 是否在大规模数据、多任务训练下具有更好的 scaling efficiency，则需要系统实验验证。

**想象力的边界在哪里？** DreamerV3 并不是通过一次无限延长的 latent rollout 来解决长期决策；其默认 imagination horizon 为 15 steps，长期回报主要通过 value bootstrap 传递。有趣的是，TD-MPC2 也采用了类似的策略——默认 MPC horizon 仅 3 steps，同样通过 Q-function bootstrap 处理超出显式 planning horizon 的长期影响。**两者都没有简单地通过无限延长 latent rollout 来解决长期决策，而是把远期预测部分交给 value function。** 因此，一个更准确的问题是：**有限 horizon 的 imagination/planning + value bootstrap 能够在多大程度上可靠地解决长时程任务？如果进一步增加 rollout horizon，模型误差又会以什么速度累积？**

---

*这篇是 RSSM 系列的补充视角。如果想看 RSSM 的具体架构细节，可以参考[之前的 RSSM 深度拆解](/zh/articles/rssm-deep-dive/)和[源码六部曲](/zh/articles/2026-08-19-rssm-code-walkthrough/)。*

*下一篇是 VLA 系列的中篇——π₀ 家族与动作接口的演进。*
