---
title: "读懂 Dreamer：世界模型是怎么学会'想象'的？"
slug: "2026-08-25-dreamer-explained"
date: 2026-08-25
draft: false
categories: ["世界模型"]
tags: ["DreamerV3", "世界模型", "RSSM", "强化学习", "imagination", "Dreamer系列"]
description: "从 RSSM 架构到 imagination 机制，完整拆解 Dreamer 如何在隐空间中构建世界模型、生成训练数据并优化策略，附源码级解读。"
toc: true
related_articles:
  - rssm-deep-dive
  - 2026-08-27-dreamer-actor-critic
  - 2026-08-28-dreamerv3-training-tips
  - world-model-intro
  - vla-vs-world-model
  - td-mpc-world-model-control
---

> **Dreamer 系列 · 第 1 篇**
>
> 本篇从架构层面讲清楚 Dreamer 的整体设计。如果你已经读过 [RSSM 代码解析系列](/zh/articles/2026-08-19-rssm-code-walkthrough/)，这篇可以帮你把零散的代码细节串成完整的架构理解。

Dreamer 最重要的思想，不是"训练一个会生成未来画面的模型"，而是**训练一个足够支持决策的 latent world model，然后让策略在这个内部世界里学习**。

这篇文章围绕这条主线，把 Dreamer 的设计逻辑从头到尾讲一遍。

## 一、Dreamer 到底在解决什么问题？

强化学习的核心循环是：智能体与环境交互 → 获得奖励 → 改进策略。这个循环的问题在于，每一步交互都需要真实的环境计算——对于机器人控制这类复杂任务，这意味着大量的仿真时间甚至真实机器人时间。

Dreamer 的思路是：与其在真实环境中反复试错，不如先**学一个世界模型**，然后在模型的"想象"中训练策略。

这听起来简单，但要做到需要解决几个关键问题：世界模型怎么表征环境状态？怎么从想象中学出有用的策略？怎么避免想象误差把策略带偏？

从 V1 到 V3，Dreamer 的演进可以粗略理解成三个方向：更好的 latent representation、更稳定的 imagination learning，以及更强的跨任务泛化能力。

## 二、Dreamer 为什么主要在 latent space 中预测？

在讲架构之前，先建立一个最重要的直觉。

Dreamer 并不是完全不预测 observation，而是**不需要把未来像素级视频作为策略学习的主要载体**。Dreamer 的关键是在 latent space 中学习对预测和决策有用的 dynamics。

假设机器人当前看到一张 1024×1024 的 RGB 图像。如果世界模型每一步都必须精确预测下一帧的每一个像素，大量计算都会浪费在与决策无关的视觉细节上。Dreamer 的做法是先把观测压缩成 latent state，再在 latent space 中预测未来。只要 latent representation 保留了与预测和决策相关的信息，就不必要求世界模型精确重建未来画面的每一个像素。

换句话说，**Dreamer 不是预测世界"长什么样"，而是预测决策需要知道的世界状态。**

这是理解后续所有设计的核心出发点。

## 三、全局架构图

Dreamer 的训练可以理解成两个循环：**Observe（观察）**负责从真实环境数据学习世界模型；**Imagine（想象）**负责在 latent space 中生成未来轨迹，并利用这些轨迹训练策略。

需要注意的是，imagination 中的 action 和真实环境中的 action 是两条不同的路径：imagination 中的 action 用来推进世界模型预测下一个 latent state，而真实环境中的 action 才真正改变环境。

```text
              ┌─────────────────────────────────────┐
              │          OBSERVE 阶段                 │
              │                                     │
              │   Real Environment                  │
              │        │                            │
              │   observation o_t                   │
              │        │                            │
              │        ▼                            │
              │    Encoder                          │
              │        │                            │
              │        ▼                            │
              │   ┌──────────────────────┐          │
              │   │        RSSM          │          │
              │   │                      │          │
              │   │  deterministic h     │          │
              │   │        +             │          │
              │   │  posterior z         │ ← obs    │
              │   └──────────┬───────────┘          │
              │              │                      │
              │         latent state                │
              └──────────────┼──────────────────────┘
                             │
              ┌──────────────┼──────────────────────┐
              │          IMAGINE 阶段                 │
              │              ▼                       │
              │   ┌──────────────────────┐           │
              │   │    IMAGINATION       │           │
              │   │                      │           │
              │   │    Prior rollout     │           │
              │   │         ↓            │           │
              │   │    latent states     │           │
              │   └──────────┬───────────┘           │
              │              │                       │
              │       ┌──────┴──────┐                │
              │       ▼             ▼                │
              │     Actor         Critic             │
              │       │                            │
              │       ▼                            │
              │  imagined action                   │
              │       │                            │
              │       ▼                            │
              │  Prior dynamics                    │
              │       │                            │
              │       ▼                            │
              │  next latent state                 │
              │  (继续 rollout)                     │
              └────────────────────────────────────┘

    同时，Actor 的策略也在真实环境中执行：

              Actor (当前策略)
                   │
                   ▼
                 action
                   │
                   ▼
            Real Environment
                   │
                   ▼
             observation
                   │
                   ▼
            RSSM / Posterior
            (更新世界模型)
```

在 Observe 阶段，Posterior 和 Prior 通过 KL loss 互相约束：

```text
Prior:      p(z_t | h_t)
                ↑
              KL loss（两侧 stop-gradient）
                ↓
Posterior:  q(z_t | h_t, o_t)
```

**Posterior 是训练世界模型时的"观测校正器"，Prior 是 imagination 时真正被 rollout 的模型。** 这个关系非常关键。

## 四、Observe：世界模型怎么学习？

Observe 阶段的目标是从真实环境数据中学习环境的 latent dynamics。

```text
观测 o_t → 编码器 → o_t^emb
                         ↓
              RSSM: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
                         ↓
              Posterior: q(z_t | h_t, o_t^emb)  ← 看得到观测
              Prior:     p(z_t | h_t)            ← 看不到观测
                         ↓
              feature_t = concat(h_t, z_t)
                        → deterministic state + stochastic state
                        → 具体维度由模型配置决定
```

这里有两个关键组件：

**Posterior 看得到观测。** 它利用真实观测来"纠正" latent state——相当于在说："给定我看到的画面和记住的历史，当前状态最可能是什么？"

**Prior 看不到观测。** 它只能依赖过去的 latent state 和 action——相当于在说："仅凭我的记忆，我预测下一步状态应该是什么？"

训练时通过 KL loss 约束 Prior 与 Posterior 的分布差异，并使用 KL balancing 控制两侧的学习力度，使模型既能学习可靠的 posterior representation，又能让 prior 在没有观测时复现这种 latent dynamics。这样在想象阶段（没有观测可用时），Prior 就能独立工作。

以 DreamerV3 的默认配置为例，deterministic state（8192维）和 stochastic state（32×64=2048维）拼接后形成 10240 维 feature。但这不是固定值——维度由模型配置决定。

关于 RSSM 的确定性转移、Block GRU、categorical latent 等代码细节，可以参考 [RSSM 代码解析系列](/zh/articles/2026-08-19-rssm-code-walkthrough/)。

## 五、Imagine：世界模型怎么"做梦"？

Imagine 阶段是 Dreamer 最核心的设计。

在 Dreamer 里，"想象"本质上就是让 RSSM 的 prior dynamics 在没有真实 observation 的情况下，根据当前 latent state 和 Actor 产生的 action，不断预测后续 latent states。

```text
从 replay buffer 中采样的真实观测序列对应的 posterior latent state 出发
         ↓
    只用 Prior 做 rollout（不看观测）
         ↓
    h_t, z_t ~ Prior → feature_t → Actor → action_t
         ↓
    feature_t → Critic → value_t
    feature_t → Reward predictor → reward_t
         ↓
    用 imagined (reward, value) 更新 Actor 和 Critic
```

Reward predictor 的作用是根据 imagined latent state 预测对应的 reward；这样 imagination 即使没有真实环境，也能获得训练 Actor 所需的 reward signal。

需要特别强调的是：**Actor 和 Critic 的参数更新主要在 imagined trajectories 上进行，但这些 imagination 的起点以及世界模型本身，都来自真实环境采集的数据。**

Dreamer 的完整循环是：真实环境收集数据 → 学习世界模型 → 从真实数据的 latent state 开始想象 → 训练 Actor/Critic → Actor 回到真实环境执行 → 产生新数据 → 继续训练世界模型。需要强调的是，Observe 和 Imagine 并不是一次性执行的两个独立训练阶段，而是在整个训练过程中反复交替进行的。

这和"纯模型自我训练"有本质区别。

## 六、为什么在"梦"里训练不会立刻跑偏？

一个自然的疑问是：在模型想象出来的轨迹上训练策略，不会越训越偏吗？

Dreamer 用几个机制来控制这个问题：

**有限 horizon + value bootstrap。** Dreamer 使用预先设定的有限 imagination horizon，避免让 latent model 无限向前 rollout；在 horizon 末端，通过 value estimate 对更远期的回报进行 bootstrap。因此，策略既能利用模型提供的短期 imagined futures，又不需要让世界模型承担无限长的预测任务。

**KL loss 约束 Prior 与 Posterior 的一致性。** 训练时通过 KL loss 约束 Prior 与 Posterior 的分布差异，并使用 KL balancing 控制两侧的学习力度。这样可以让 Prior 在真实数据覆盖的状态分布附近尽可能逼近 Posterior，为 imagination 提供更可靠的 latent dynamics；但它并不能消除长 horizon rollout 中的误差累积。

**持续重新数据采集。** Actor 在真实环境中执行当前策略并持续收集新数据，这些数据进入 replay buffer，用于进一步更新世界模型和策略。这样世界模型不会一直在自己的想象中"闭门造车"。

当然，想象训练的局限性也是存在的。对于需要极长时程规划的任务，或者环境动态非常复杂（如接触丰富的操控任务），世界模型的想象误差可能会显著影响策略质量。这也是当前 model-based RL 仍在积极探索的方向。

## 七、V1 → V2 → V3 到底改变了什么？

### DreamerV1：展示"想象训练"可行

DreamerV1 的核心贡献之一，是展示了一个重要思路：用学习到的 latent dynamics 生成 imagined trajectories，并在这些轨迹上训练 Actor-Critic，可以有效解决连续控制任务。

### DreamerV2：引入 categorical latent

V2 将 RSSM 的 stochastic state 改为 categorical latent，并使用 straight-through estimator，使离散 latent 可以参与端到端训练。多个 categorical variable 的组合为状态提供了紧凑的离散表征，这种离散 latent 表征是 V2 在 Atari 等视觉控制任务上取得提升的重要组成部分。

### DreamerV3：走向更强的通用性

V3 做了几个重要的工程改进，每个改进都针对一个具体问题：

| 改进 | 主要解决的问题 |
|------|--------------|
| **symlog** | 不同任务 reward/value 数值尺度差异 |
| **unimix** | categorical 分布过早变得过于尖锐 |
| **KL balancing** | prior / posterior 学习不平衡 |
| **free_nats** | KL 项过度影响 latent learning |
| **Block GRU + 大 deter** | 提升 deterministic memory capacity，同时控制计算成本 |

具体说明：

* **symlog 预测**：对 reward、value 等数值目标进行尺度压缩，使模型能够更稳定地处理不同任务中跨度很大的数值范围，从而减少对任务特定 reward/value scaling 的依赖。
* **unimix**：给 categorical 分布混入少量均匀分布；DreamerV3 的默认配置使用 `unimix=0.01`，用于避免 categorical distribution 过早变得过于尖锐。
* **KL balancing 与 free_nats**：KL balancing 用于控制 prior 与 posterior 两侧的梯度贡献；free-nats 则对 KL 项的有效优化压力进行限制，避免 KL 项在已经较小的区域继续主导 latent representation 的学习。具体到 DreamerV3 的实现，free-nats 的计算位置和 stop-gradient 方式值得单独展开，这也是 [RSSM 系列](/zh/articles/2026-08-22-rssm-kl-balancing/)中重点讨论的部分。
* **更大的 deterministic state + Block GRU**：DreamerV3 使用更大的 deterministic state（默认 `deter=8192`），并采用 Block GRU 结构来控制大 hidden state 带来的计算成本。

这些改进让 DreamerV3 在多种不同类型的任务上展现出了较强的通用性，也使它成为研究世界模型与 model-based RL 时非常有代表性的开源实现。

## 八、Dreamer 在世界模型中的位置

世界模型是一个很大的概念，Dreamer 是其中一条特定路线的代表。下面这张表把当前几种主要范式放在一起对比：

| 范式 | 代表 | 核心思路 |
|------|------|----------|
| **Latent dynamics + RL** | DreamerV1/V2/V3 | 在隐状态空间学习 dynamics，并通过 imagination 优化策略 |
| **Generative / interactive world model** | Genie 等 | 学习生成/预测环境未来状态 |
| **Video / diffusion world model** | DIAMOND 等 | 在视觉空间中生成未来观测 |
| **VLA policy** | RT-2、OpenVLA、π0 | 直接学习视觉/语言到动作的映射 |

需要说明的是，VLA 与前三者不是严格意义上的同一分类维度，这里的分类主要用于建立直觉，并不是严格互斥的 taxonomy。

Dreamer 路线的特点是：它不直接预测像素级的未来画面，而是在一个压缩的隐空间中预测未来状态。这通常可以显著降低预测和规划的表示空间成本，但代价是 latent representation 不一定保留所有像素级细节。

关于不同架构路线的对比，可以参考 [世界模型架构演进：RSSM、Transformer 与统一世界模型](/zh/articles/world-model-transformer/)。

## 九、一个直觉例子

为了把"世界模型 → imagination → policy"这条链路讲得更具体，可以想象一个简单的场景：

```text
机器人看到："前面有墙"

Posterior（看得到观测）：
  "我现在知道前面有墙"

Prior（看不到观测，只凭记忆）：
  "根据过去几步，我预测前面大概率还是墙"

Imagine（用 Prior rollout 未来）：

  向左行动：
    latent state → reward/value
    "未来回报更高"

  向右行动：
    latent state → reward/value
    "未来回报更低"

Actor：
  "那我选择向左"
```

注意这里不只是"Prior 自己知道哪边安全"——而是三者各司其职：

**Prior 负责"预测会发生什么"，Reward model 负责"会得到什么奖励"，Critic 负责"这个未来长期值多少"，Actor 负责"选择做什么"。**

```text
World Model   → What will happen?
Reward Model  → What reward will I get?
Critic        → How valuable is this future?
Actor         → What should I do?
```

这是 Dreamer 最核心的职责分工。

这就是 Dreamer 的核心循环：Posterior 从真实观测中得到可靠 latent → Prior 学会不看观测也能预测 latent → latent dynamics 支持 imagination → Reward/Critic 评价 imagined futures → Actor 根据评价学习 → 新策略回到真实环境收集数据 → 世界模型继续变好。

## 十、把之前的文章串起来

到这里，博客上的世界模型内容可以形成这样的阅读路径：

```text
世界模型入门 → RSSM 深度解析 → RSSM 代码系列（6篇）
                                       ↓
                              本篇：Dreamer 整体架构
                                       ↓
                        Dreamer 训练技巧 → GPU 选型
```

如果你刚接触世界模型，建议从 [什么是机器人世界模型？](/zh/articles/world-model-intro/) 开始。

如果你想看代码级别的 RSSM 拆解，[RSSM 代码解析系列](/zh/articles/2026-08-19-rssm-code-walkthrough/) 从 stochastic state 一路讲到 KL balancing 和 imagine reset。

如果你对训练过程中的实际问题感兴趣，[DreamerV3 训练技巧](/zh/articles/dreamerv3-training-tips/) 总结了从环境配置到超参调优的实战经验。

## 十一、总结：Dreamer 真正"梦"的是什么？

Dreamer 的"梦"并不是预测一段未来视频，而是让 agent 在一个学习出来的 latent world 中进行低成本的 counterfactual trial-and-error：

> "如果我现在做这个动作，接下来可能发生什么？这个未来值不值得？"

世界模型负责预测，Critic 负责评价，Actor 负责选择。真实环境则不断提供新的数据，纠正这个内部世界。

**所以 Dreamer 的核心不是"想象得多逼真"，而是"想象得是否足以支持正确决策"。**

下一篇我们会深入讨论 Dreamer 中 Actor-Critic 的设计——它们是怎么在想象空间中工作的，以及 symlog 变换为什么对价值学习如此重要。
