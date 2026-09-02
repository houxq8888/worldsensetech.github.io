---
title: "VLA 深度解读（下）：VLA 与世界模型、开放问题与三个判断"
slug: "2026-09-07-vla-world-models"
date: 2026-09-07
draft: false
categories: ["具身智能", "论文解读"]
tags: ["VLA", "World Model", "世界模型", "Planning", "具身智能", "Robot Foundation Model"]
description: "VLA 系列三篇的下篇。讨论 VLA 与世界模型的关系——区分被动预测、action-conditioned 和 subgoal generator 三类世界模型，并展开数据瓶颈、长时序、安全、缺失模态等开放问题，最后收束为三个判断。"
toc: true
related_articles:
  - 2026-09-03-vla-deep-dive
  - 2026-09-05-vla-pi-family
  - 2026-09-02-jepa-deep-dive
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
---

> **VLA 系列共三篇：** [上篇：RT-2 到 OpenVLA](/zh/articles/2026-09-03-vla-deep-dive/) · [中篇：π₀ 家族](/zh/articles/2026-09-05-vla-pi-family/) · 下篇（本文）

在[上篇](/zh/articles/2026-09-03-vla-deep-dive/)和[中篇](/zh/articles/2026-09-05-vla-pi-family/)中，我们走过了 VLA 从 RT-2 到 π₀.7 的完整技术演进。最后这篇收束全系列，把线索集中到一个最核心的问题：VLA 和世界模型到底是什么关系？

## 一、VLA 与世界模型：Policy Learning vs Predictive Modeling

这是我认为最值得深入讨论的部分。前面技术演进中已经多次触及这个问题——从 RT-2 的 model-free policy，到 π₀ 的 action chunk 不等于 planning，再到 π₀.7 的 visual subgoal 不等于 world model——现在把这条线索集中展开。

### VLA 缺的不是"预测能力"，而是显式、可查询的 action-conditioned prediction interface

一个常见的简化是：VLA 只能做动作，世界模型才能做预测。但这个说法不够精确。

VLA 当然可以做预测——一个足够大的自回归模型完全可以预测下一帧图像。真正的区别不在于"有没有预测能力"，而在于：**预测是不是模型的显式、可查询、action-conditioned interface？**

具体来说：

- **VLA 学的是 action distribution**：π(a_t | o_{≤t}, l)——只需要回答"现在该做什么动作"
- **世界模型学的是 future distribution**：一个典型 action-conditioned world model 可以表示为 p_θ(z_{t+1:t+H} | z_t, a_{t:t+H-1})，也可以是确定性的 latent transition function f_θ(z_t, a_t) → z_{t+1}——回答"如果执行这些动作，未来会变成什么样"

有了后者，才能自然地形成一种典型的 planning 形式：

```
候选动作 a⁽¹⁾ → 预测未来 ô⁽¹⁾ → 评估 J(a⁽¹⁾)
候选动作 a⁽²⁾ → 预测未来 ô⁽²⁾ → 评估 J(a⁽²⁾)
...
选择 J 最大的动作序列
```

**这是一种典型的 planning 形式：利用预测模型评估候选动作或轨迹的后果，再进行选择或优化。** Planning 还可以通过 trajectory optimization、MPC、gradient-based optimization、tree search、latent-space optimization、goal-conditioned planning 等多种方式实现，不一定必须显式生成离散的多个候选轨迹。**典型的 imitation-learning VLA 并没有把 action-conditioned future prediction 作为显式、可查询的模型接口。**

需要注意：典型的 imitation-learning VLA 并不显式学习可查询的 action-conditioned dynamics model——但 policy 本身可以隐式编码动态先验。这和"完全没有关于物理世界的内部表征"是两回事。

### 更准确的区分框架

| 维度 | VLA / Policy | Action-conditioned World Model |
|------|-------------|-------------------------------|
| **核心问题** | 现在应该做什么？ | 做了之后会发生什么？ |
| **学习目标** | π(a \| o, l) | p(z_future \| z, a) 或 f(z, a) → z' |
| **数据关系** | observation → action | observation + action → future |
| **输出** | 动作指令 | 预测的未来状态/latent |
| **典型用途** | execution | prediction / planning |
| **主要风险** | policy error / distribution shift | model bias / compounding prediction error |
| **是否需要搜索** | 不需要 | 可与搜索/MPC/optimization 结合 |
| **典型角色** | 偏 execution | 偏 prediction / planning |

简单来说：VLA 回答"我要做什么动作"；世界模型回答"执行这个动作后世界会变成什么样"。典型角色上，VLA 更偏 execution，world model 更偏 prediction/planning——但 VLA 也可以做 implicit planning、hierarchical policy、chain-of-thought，world model 也可以直接支持 policy learning。

### 被动世界模型 vs Action-conditioned 世界模型 vs Subgoal Generator

这里有一个容易混淆的概念需要区分。**本文所说的"用于机器人 planning 的 world model"，重点指具有可查询 action-conditioned prediction interface 的预测模型。** 广义的 world model 可以包括被动视频预测、latent dynamics、reward prediction、object-centric models、generative simulators、goal-conditioned models 等，但这里聚焦于与 planning 直接相关的类型。

**被动世界模型（passive world model）** 可以只用视频学习"世界如何变化"——从 o_t 预测 o_{t+1}，不需要动作标签。

**Action-conditioned 世界模型** 则需要 (o_t, a_t, o_{t+1}) 三元组数据，学习的是"**采取不同动作会导致什么结果**"。这里的 action 不一定非得是机器人低层 motor command——可以是 end-effector action、semantic action、甚至 high-level skill。"action-conditioned"真正要求的是：模型知道导致状态转移的 intervention / control variable 是什么。

**Subgoal generator（如 π₀.7 的 BAGEL-based world model）** 则是第三类：它不预测"执行某个动作后会发生什么"，而是根据任务条件和上下文生成"未来应该看到什么"的候选视觉子目标。

| 模型类型 | 条件 | 输出 | 是否直接回答"动作后果" |
|---------|------|------|------|
| Passive predictive | 当前状态 | future state | 否 |
| Action-conditioned WM | 当前状态 + action | future state | 是 |
| Subgoal generator | 当前状态 + task/context | visual goal | 否 |
| Policy | 当前状态 + task | action | — |

这张表可以作为理解 VLA 与 world model 关系的一个核心参照。

所以真正需要 action-labeled interaction data 的，是**用于 action-conditioned planning 的 world model**。这也正好解释了为什么 V-JEPA 2（被动视频预测）和 V-JEPA 2-AC（action-conditioned）需要在技术栈上分开——JEPA 本身是一类 predictive representation learning 方法；V-JEPA 2-AC 则是 V-JEPA 2 系列上的 **action-conditioned extension**，而不是简单意义上的下一代独立模型——它进一步把 action conditioning 引入预测过程，使其能够承担 action-conditioned world modeling 的角色。

### 统一模型的技术框架

真正的统一模型其实需要同时回答两个问题。一个概念化的联合建模形式可以写成：

p(a_{t:t+H}, z_{t+1:t+H} | z_t, l, g)

也就是同时学到：
- **我要做什么？**（policy）
- **这么做以后会发生什么？**（prediction）

这比单纯说"VLA + world model"更精确——它定义了一个同时具备 action distribution 和 future distribution 的联合模型。**联合建模 policy 与 future state 是 planning 的基础，但真正的 planning 仍需要目标函数、搜索、优化、MPC 或其他 action selection mechanism。** 联合分布本身不等于 planning。

### 两条路线正在靠近

需要纠正一个常见误解：世界模型路线并不是"没有语言"或"不能做 action"。V-JEPA 2 系列已经展示了从 web-scale 视频预训练到 action-conditioned latent prediction、再到机器人规划/控制的技术链条，包括 zero-shot robot deployment 和 image-goal planning。世界模型本身也可以通过语言对齐获得语义能力。

JEPA 路线的规划也不一定是"生成多条轨迹再选择"。它可以是 latent prediction → goal-conditioned planning，可以是 search、optimization 或 policy guidance。

所以更准确的图景是：**VLA 和世界模型正在从两个方向靠近同一个目标——一个同时具备 policy、prediction 和 planning 能力的机器人基础模型。** 未来系统更可能是 Actor + Predictor，而不是二选一。

```
                 Robot Foundation Model
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ↓              ↓              ↓
       Language       Perception      Robot Data
          │              │              │
          └──────────────┼──────────────┘
                         ↓
                 Shared Representation
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       Semantic       Policy        Prediction
       Subtask          │              │
          │             ↓              ↓
          │       Action Chunk    Future State
          │             │              │
          │             ↓              ↓
          │       Flow Matching    World Model
          │             │              │
          └─────────────┼──────────────┘
                        ↓
                  Physical Action
```

**未来的机器人 foundation model 很可能不是一个纯 VLA，也不是一个纯 world model，而是一个同时支持 semantic conditioning、policy execution 和 predictive modeling 的统一系统。**

### 它们互补吗？

我觉得答案是肯定的，但互补的方式比"把两个模块拼在一起"更微妙。

**典型的直接 policy VLA 缺什么？** 它不显式学习可查询的 action-conditioned dynamics model。当遇到训练中没见过的新情况时，它只能依赖预训练阶段学到的泛化能力——policy 可以隐式编码动态先验，但不能像世界模型那样显式模拟行动后果。

**世界模型缺什么？** 虽然世界模型正在获得语言和 action 能力，但在端到端策略学习的效率和语言接地的自然度上，目前仍不如 VLA 路线。

所以一个自然的想法是：**用世界模型做物理预测，用 VLA 做动作执行和语言理解。**

## 二、开放问题

**数据瓶颈——从"小时数"到"有效数据"。** 更有意义的问题可能不是"如何获得百万小时机器人数据"，而是：**机器人数据是否真的应该继续按"小时"计量？** 1 小时人类连续成功折 300 件衣服，和 1 小时机器人遇到 50 次失败、20 次恢复、10 种不同策略、5 种 embodiment——信息量完全不同。未来数据 scaling 的价值函数可能更像：

Data Value = f(diversity, failure, recovery, embodiment, task coverage)

而不只是 Data Value ∝ hours。这正好连接到 π₀.7 对 heterogeneous / suboptimal data 的探索。许多主流 VLA 数据集以成功 demonstration 为主，失败 / recovery 数据相对稀缺。**如何从"成功 demo 数据集"走向包含失败、恢复和策略变化的 experience dataset？** 是更关键的问题。π₀ 本身已经给出了一个直接的证据：只用高质量数据会让策略更 fluent，但容易缺少错误恢复能力；多样、较低质量的预训练数据则提供了 recovery / correction repertoire。下一阶段不是单纯 scale hours，而是 scale useful experience。

**长时序任务——错误传播而非步数。** 长时序真正的问题不是 H > 5 或 H > 50，而是错误累积的概率效应。在一个独立同分布、无恢复机制的极简近似下：

P(success over T) ≈ ∏_{t=1}^{T} p_t

如果每步成功率 p_t = p = 0.98，则 0.98^100 ≈ 13%。这当然不是机器人任务成功率的真实统计模型——真实任务中 decisions 不是独立的，recovery 可以改变后续概率，有些错误可恢复，有些关键动作比其他动作重要得多——但它很好地说明了 error accumulation 的指数性直觉：**long-horizon difficulty 的本质是 error accumulation，而不是简单的 sequence length。** 这也解释了为什么 hierarchical policy、recovery policy、replanning、world model 和 memory 都是应对长时序问题的自然方向。

**安全——三个层次。** VLA 的安全问题可以分成三个层次：

*Policy safety*：π(a|o) 会不会输出危险动作？

*Predictive safety*：p(o_future|o,a)——这个动作执行后会不会造成危险？

*Runtime safety*：即使模型错了，有没有独立 safety layer 拦截？

一种可能的 runtime safety architecture 是：

```
VLA
 ↓
candidate action
 ↓
world model / safety critic
 ↓
constraint checker
 ↓
robot
```

safety constraint 不能只作为语言层面的 alignment 问题——它是工程硬约束。

**缺失的模态。** 视觉和语言仍然是主流 VLA 的核心条件模态，proprioception 已经成为不少系统的标准输入（π₀ 和 π₀.7 均使用），而触觉、力/扭矩、听觉等模态在当前大规模 VLA 预训练体系中的覆盖仍明显不足。但对于精细操作（拧螺丝、插钥匙、折叠柔软物体），这些模态可能是关键信息源。

**VLA 是否需要世界模型？** 这个问题我觉得还没有确定的答案。π₀.7 引入视觉子目标作为 conditioning signal 确实提升了泛化能力，但这和"拥有一个显式世界模型"是两回事。真正的融合可能需要一个模型同时做到：语言接地、action-conditioned prediction、高频连续控制。**就本文讨论的公开代表性工作而言，还没有一个系统在大规模真实机器人任务上，以成熟且统一的方式同时解决语言接地、action-conditioned prediction 和高频连续控制。**

**离散 vs 连续的最终判断。** 从 RT-2 的离散 token 到 OpenVLA-OFT 的 continuous regression 再到 π₀ 的 flow matching，连续方法在控制精度和推理速度上展现了优势。但 π₀.5 和 π₀-FAST 表明，**离散和连续很可能承担不同的角色：离散负责"统一"——使动作可以和语言、视觉、语义子任务共享 sequence modeling 接口；连续负责"控制"——在最终执行阶段提供高频精细的连续动作输出。**

```
         Foundation-model pretraining
                    ↑
                    │
           discrete tokens
                    │
                    │
              shared LM space
                    │
                    ↓
       continuous action generation
                    │
                    ↓
           high-frequency control
```

这个判断比"continuous 最终取代 discrete"强得多，也更能解释为什么 π₀-FAST、π₀.5、π₀.7 看起来像是在"走回头路"，实际上是在探索不同的模型接口。

---

## 三、三个判断

最后把全文的论证收束为三个判断。前半部分是从已有论文归纳出的技术趋势，后半部分是基于这些趋势提出的未来架构判断。

**判断一：VLA 的进步不能用参数规模单轴解释，动作接口正在成为与 backbone scaling 并列的重要设计轴。** RT-2 → OpenVLA → OFT → π₀ 证明：backbone scaling + data scaling + action interface 共同决定性能。从 RT-2 的 55B 到 OpenVLA 的 7B 到 π₀ 的 3.3B，参数量在缩小；但从 256-bin 离散 token 到 parallel continuous regression 到 flow matching + 50-step action chunk，动作接口在不断进化。OFT 的结果表明，action interface、decoding strategy 和 temporal chunking 本身就是重要的系统设计轴，而不仅仅是 backbone scaling 的附属问题。

**判断二：通用机器人能力的关键瓶颈正在从 representation scaling 转向 effective data scaling、temporal abstraction 和 recovery。** π₀.5 的 97.6% 非目标域数据、π₀.7 对 suboptimal data 的利用、以及长时序任务中错误累积的结构性困难，这些工作共同提示——模型规模之外，effective data scaling、temporal abstraction 和 recovery 正逐渐成为独立的性能决定因素。

**判断三：真正的下一阶段可能不是"VLA 还是 World Model"，而是 policy、predictor 和 planner 的统一。** 未来机器人基础模型的技术地图可以画成：

```
                 Robot Foundation Models
                           │
          ┌────────────────┼────────────────┐
          │                │                │
       Backbone       Action Interface   Temporal Structure
          │                │                │
      VLM / VLA       discrete token      action
          ↓                ↓              chunk
      semantic        continuous            ↓
      grounding       regression       semantic subtask
          │                ↓                │
          │           flow matching         │
          │                │                │
          └────────────────┼────────────────┘
                           ↓
                    Generalist Policy
                           │
                ┌──────────┴──────────┐
                ↓                     ↓
             Action              Prediction
                │                     │
                │              future state /
                │              visual subgoal
                │                     │
                └──────────┬──────────┘
                           ↓
                   Planning / Recovery
                           │
                           ↓
                    Physical Robot
```

如果把这条技术主线压缩：

```
RT-2
│
├─ web knowledge → action token
│
↓
OpenVLA
│
├─ open multi-robot scaling
│
↓
OpenVLA-OFT
│
├─ action interface
│
↓
π₀
│
├─ continuous generative action
│
↓
π₀.5
│
├─ heterogeneous data
├─ semantic hierarchy
│
↓
π₀.7
│
├─ context-rich steering
├─ subgoal conditioning
├─ heterogeneous / suboptimal experience
│
↓
???
├─ policy
├─ prediction
└─ planning
```

如果要用一个方向性描述：

> Robot Foundation Model 的可能形态 = 在统一表征基础上，组合 Perception + Language + Policy + Prediction + Planning 能力

但必须马上补两句：**今天的公开系统通常只覆盖其中的一部分，π₀.5/π₀.7 等工作更像是在逐步扩大这个闭环，而不是已经完成统一。** 并且，这并不意味着每个 robot foundation model 都必须同时包含所有这些能力——更可能的未来形态，是一个能够在统一表征基础上灵活组合这几个能力的系统。

正如我在[世界模型盘点](/zh/articles/2026-09-01-world-model-h2-review/)里说的，"world model"正在失去单一含义。VLA 的加入让这个图景更复杂，也更有趣。

*下一篇，我打算深入聊 Sim-to-Real——从仿真到真实机器人的部署鸿沟到底有多宽，以及当前最好的迁移方法是什么。*

> **VLA 系列完结：** [上篇：RT-2 到 OpenVLA](/zh/articles/2026-09-03-vla-deep-dive/) · [中篇：π₀ 家族](/zh/articles/2026-09-05-vla-pi-family/) · [下篇：VLA 与世界模型](/zh/articles/2026-09-07-vla-world-models/)
