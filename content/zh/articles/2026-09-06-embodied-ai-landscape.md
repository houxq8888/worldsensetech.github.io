---
title: '2026 具身智能技术路线图：从 VLA 到世界模型，谁在解决什么问题？'
slug: "2026-09-06-embodied-ai-landscape"
date: 2026-09-06
draft: false
categories: ["具身智能", "行业观察"]
tags: ["具身智能", "人形机器人", "VLA", "世界模型", "Sim-to-Real", "Physical Intelligence", "Gemini Robotics", "GR00T", "Cosmos", "TD-MPC"]
description: "从 π₀、Gemini Robotics 到 GR00T、Cosmos、TD-MPC2：具身智能正在形成怎样的技术栈？本文从 Policy、Dynamics、Infrastructure 三个维度盘点主要玩家的技术路线与进展阶段，试图回答一个基本问题——这个领域谁在解决什么问题？"
toc: true
related_articles:
  - 2026-09-07-vla-world-models
  - 2026-09-03-vla-deep-dive
  - 2026-09-01-world-model-h2-review
  - 2026-09-02-jepa-deep-dive
  - 2026-09-04-rssm-beyond
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
---

过去半年，具身智能领域的节奏明显加快了。

Physical Intelligence 发布了 π₀.7，Google DeepMind 推出了 Gemini Robotics 1.5 与 Robotics-ER 1.5，NVIDIA 的 Physical AI 全栈从概念走向开源，Figure AI 的人形机器人在 BMW 工厂完成了超过 11 个月的部署测试。中国这边，宇树于 2026 年 8 月完成 IPO（估值约 90 亿美元），智元、银河通用等公司的融资规模持续扩大，人形机器人从实验室原型开始走向小批量生产。

这篇文章试图做一个基础性的盘点：**具身智能正在形成怎样的技术栈？主要玩家在解决什么问题？目前到了什么阶段？** "从 VLA 到世界模型"不是一条线性技术演进路线，而是观察具身智能技术栈的一条主线。不做预测，不做排名，只是把技术地图画清楚。

## 三个核心问题：Policy、Dynamics、Infrastructure

与其把具身智能的参与者分成"三个流派"，不如问三个不同的问题：**怎么做动作？怎么预测未来？怎么获得训练数据和环境？** 这三个问题对应着三个不同的技术层面，而一个完整的具身智能系统通常需要同时回答它们。

### Policy：怎么让机器人学会行动？

**核心思路：** 从感知（视觉、语言、本体感觉）直接映射到动作，或经过中间规划后输出动作。

**代表：** Physical Intelligence（π₀ 系列）、Google DeepMind（RT-2 → Gemini Robotics 1.5）、NVIDIA GR00T

Physical Intelligence 的 π₀ 系列是目前 VLA 路线最具代表性的工作。从 π₀ 的 flow matching 连续动作生成，到 π₀.5 的离散-连续混合 recipe，再到 π₀.7 的 context-rich steering + 视觉子目标——这条技术线在动作接口设计上持续迭代。

π₀.7 尤其值得关注，因为它恰好说明 **VLA 并没有简单地排斥世界模型，而是在逐渐吸收预测性组件**。π₀.7 的视觉子目标（visual subgoal）由一个 lightweight world model 生成——这意味着 π₀.7 的核心策略接口仍然不是一个显式的 action-conditioned rollout interface，但系统已经通过外部 predictive component 引入了对未来视觉状态的结构化预测。π₀.7 展示的 cross-embodiment zero-shot T-shirt folding 是一个标志性结果，但需要注意的是，这仍然是在特定评估协议下的结果，距离通用家庭操作还有相当距离。

Google DeepMind 的路线则从 RT-2 的视觉语言模型机器人化，逐渐发展到 Gemini Robotics，并进一步扩展到 Robotics-ER 1.5 及后续版本。其重要变化不只是"更大的 VLM"，而是逐渐形成由 VLA、embodied reasoning 和高层任务规划共同组成的分层系统。Gemini Robotics 1.5 已经不是简单的"VLM → robot policy"了——官方把 Robotics-ER 1.5 定义为高层 embodied reasoning / planning，而 Robotics 1.5 负责视觉到 motor commands 的映射。这条演化线的方向是：让语义理解、物理推理和动作生成在一个分层架构中各司其职。

NVIDIA 的 GR00T N1/N2 也属于这一层——它是开源的人形机器人基础模型，提供策略级别的 capability。

**这条路线的优势**是语义泛化能力强——VLA 从互联网预训练中继承的语义知识使其能处理新物体、新指令。**局限**在于：传统 VLA policy 的核心接口仍然是 observation / instruction → action，而不是显式输出可供 rollout 的 action-conditioned future state——但正如 π₀.7 和 Gemini Robotics 1.5 所展示的，近期系统正在逐渐加入预测、子目标和高层规划模块。

### Dynamics：怎么预测未来？

**核心思路：** 在隐空间中学习 action-conditioned dynamics，再通过 imagination、MPC 或 value estimation 等方式把预测模型用于控制。这里的 Dynamics 指承担"未来状态如何变化"这一功能的模型组件，而不是狭义的一种模型架构；完整的 world model 还可能包含 reward、termination、observation prediction 等接口。

**典型路线：** DreamerV3（RSSM + imagination）、TD-MPC2（latent dynamics + MPC）；扩展到 foundation-model 规模的路线则包括 NVIDIA Cosmos 等 world foundation models。

DreamerV3 通过 RSSM 在隐空间中学习动力学模型，然后在想象中 rollout 轨迹训练 actor-critic 策略。这条路线在 sample efficiency 上有明显优势——用少量真实交互数据就能学到复杂技能。

TD-MPC2 走了更简洁的路线：encoder + MLP ensemble dynamics + latent-space MPC + Q-function ensemble。不用 RSSM 的双轨结构，而是在 cross-task scaling 上展示潜力——在 104 个连续控制任务、4 个 domain 上进行评估，并展示了单一 317M 参数模型跨 80 个任务训练的 scaling 结果（关于 TD-MPC2 的架构细节，在[之前的文章](/zh/articles/2026-09-04-rssm-beyond/)中做过详细拆解）。

**这条路线的优势**是 sample efficiency，以及显式利用 action-conditioned dynamics 进行预测、规划或价值估计。需要注意的是，latent predictive model ≠ accurate physical simulator——隐空间预测模型的优势不在于"物理预测精度高"，而在于它提供了一个可用于 planning 和 value bootstrap 的预测接口。**局限**在于：世界模型的预测质量随 rollout 长度衰减，长程任务的可靠性仍然是瓶颈。对于以 latent dynamics + model-based control 为核心的传统世界模型路线（如 Dreamer、TD-MPC2），语言接地和开放世界语义泛化通常不是其主要优化目标，因此在这些能力上往往不如以 foundation model 为基础的 VLA。

NVIDIA Cosmos 也属于 Dynamics 这一层，但需要做一个区分：其中 Dreamer / TD-MPC2 更接近控制导向的 latent dynamics，而 Cosmos 属于面向物理世界建模、生成与策略模型训练的 foundation-model 路线，两者并非同一种 world model。Cosmos 能够用于物理世界预测、世界生成以及 action / policy model 的训练与生成，其定位已经远不止"合成数据生成器"，而是包含 world models、post-training、data processing、evaluation 在内的 Physical AI 开发栈。

### Infrastructure：怎么获得训练数据和环境？

**核心思路：** 提供仿真平台、数据工具链和基础模型，支撑上层 Policy 和 Dynamics 的训练与评估。

**代表：** NVIDIA（Isaac / Omniverse）、World Labs（Marble）、遥操作数据采集与 dataset curation 系统

Infrastructure 不只是仿真，也包括 robot data collection、teleoperation、dataset curation、synthetic data generation 和 evaluation。

NVIDIA 在这个层面构建的是一个 Physical AI 全栈：Isaac / Omniverse 提供仿真与数据生成基础设施，GR00T 提供机器人基础模型（Policy 层），Cosmos 则提供 world foundation models 与数据生成、预测和 post-training 能力（Dynamics 层）。NVIDIA 并不以自有量产机器人作为核心产品，而是希望通过全栈软硬件和开发工具服务机器人生态——这是一个杠杆效应很强的定位。

### 空间世界模型：World Labs

World Labs 值得单独讨论，因为它的技术定位与上述基础设施玩家有所不同。

World Labs 的 Marble 是一个 multimodal world model，能够从 text / image / video 等输入生成可探索、可编辑的 3D worlds。它目前更接近 **spatial intelligence / generative world modeling**，而不是完整的 robot policy stack。

```
World Labs Marble
  生成 / 重建 3D world
        ↓
  可成为 simulation / spatial environment asset
        ↓
  robotics / simulation downstream
```

Marble 有潜力成为 sim-to-real 链条中的一个重要环境生成环节，帮助提升仿真环境的多样性与真实性。但它本身不是一个机器人策略系统，而是为下游的 Policy 和 Dynamics 层提供空间智能基础设施。

## 人形机器人：硬件形态的赌注

2026 年最显眼的趋势之一是人形机器人的集体冲刺。

**Figure AI** 的 Figure 02 在 BMW Spartanburg 工厂进行了 11 个月部署，累计运行超过 1,250 小时，并参与了 30,000+ 辆 X3 的生产流程（以上数据来自 Figure 官方披露）。Figure 的技术路线结合了端到端学习（与 OpenAI 合作）和传统控制。在公开披露的人形机器人案例中，这是目前较接近长期真实部署的一类案例。

**Tesla Optimus** 持续迭代硬件设计，目标是工厂内部部署。Tesla 的优势在于垂直整合能力——自有工厂提供测试环境，自有芯片（Dojo/FSD）提供训练算力，自有 AI 团队提供算法。

**1X Technologies** 的 NEO 系列面向通用服务场景，技术路线偏向端到端学习。

**中国公司**在这个方向上的投入尤其密集。宇树（Unitree）从四足机器人切入人形，硬件迭代速度快，成本控制能力强，并已于 2026 年 8 月完成 IPO（定价对应约 90 亿美元估值）。智元（Agibot）和银河通用（Galaxy General）则在融资规模上持续扩大——银河通用的估值据报道已达 30 亿美元级别。

人形机器人是一个高风险高回报的赌注。优势是：人形可以适配为人类设计的环境（楼梯、门把手、工具）。风险是：人形的工程复杂度远高于专用形态机器人，而公开展示中仍有相当一部分 demo 依赖遥操作、预定义流程或受控策略，距离在开放环境中自主完成复杂任务仍有明显距离。

## 中国具身智能：快速追赶与差异化

中国具身智能赛道在过去半年呈现出几个特点。

**融资与上市加速。** 宇树完成 IPO，多家头部公司估值进入独角兽区间。资本正在从"投概念"转向"投落地"。

**硬件能力突出。** 中国在机器人硬件（电机、减速器、传感器）上的供应链优势正在转化为整机优势。宇树的成本控制能力在全球范围内具有竞争力。

**软件/算法层面仍有差距。** 从公开论文、公开模型和公开 benchmark 来看，中国具身智能公司在通用 VLA foundation model、跨 embodiment 数据规模和公开技术影响力方面，与 Physical Intelligence、Google DeepMind 等头部研究团队仍存在一定差距；但这一差距的真实大小很难仅凭公开材料准确量化。与此同时，开源生态、公开论文和人才流动正在降低部分技术门槛，但这些因素能否最终转化为基础模型能力上的同等竞争力，仍需要更多公开 benchmark 验证。

**应用场景差异化。** 部分中国公司更强调工业、仓储和商业服务等明确场景的落地，而美国头部公司则同时押注通用人形、foundation model 和跨场景泛化。两种策略各有取舍——场景聚焦更容易验证商业价值，但通用性的积累可能较慢。

## 几条值得关注的技术趋势

抛开具体公司，有几个技术趋势值得持续关注。

### 趋势一：VLA 与世界模型正在形成混合架构

VLA 和世界模型的关系已经不再是简单的"向中间靠拢"。更准确地说，行业正在逐渐形成一种可能的分层结构（注意：这不是行业已确定的 canonical architecture，实际系统的模块组合方式可能非常不同）：

```
         Semantic / Reasoning
                ↓
              VLA
                ↓
      Predictive / World Model
                ↓
        Planning / Value
                ↓
    High-frequency Control
                ↓
          Embodiment
                ↓
          Real World
```

在这个分层中，**VLA 提供 semantic prior / task understanding，world model 提供 predictive interface，低层 controller / policy 提供 high-frequency action execution。** 例如：π₀.7 的视觉子目标来自 lightweight world model；Gemini Robotics 1.5 把 embodied reasoning 与 VLA 分层；NVIDIA Cosmos 提供 world foundation model + synthetic data；GR00T 提供 foundation policy；TD-MPC2 提供 latent planning。

这说明行业竞争的核心问题可能不是"VLA vs World Model"，而是：**谁负责 semantic abstraction，谁负责 prediction，谁负责 control。** 截至目前，还没有一个系统同时具备成熟的语言接地、action-conditioned prediction 和高频连续控制——但多个系统正在通过组合不同模块来逼近这个目标。

### 趋势二：仿真成为主流研发流程的重要组成部分

几乎所有主要玩家都在大规模使用仿真进行训练或数据增强。NVIDIA Isaac Sim、MuJoCo、Isaac Lab 等仿真平台的使用已经成为主流机器人研发流程的重要组成部分。sim-to-real 的鸿沟仍然存在，尤其是精细操作、接触动力学和长尾环境。

### 趋势三：数据正在成为关键差异化因素

一个越来越明显的趋势是：单纯的模型架构差异正在变得不那么容易形成决定性优势，而数据规模、数据多样性和训练 recipe 的重要性正在上升。这不是说 architecture 不重要，而是 foundation model 时代的性能瓶颈正在更多转向 data / compute / embodiment coverage。谁能获得更多样、更高质量、覆盖更广 embodiment 与任务分布的机器人交互数据，谁就更有可能形成优势。这解释了为什么遥操作数据采集、合成数据生成、数据质量筛选等"数据工程"方向正在获得更多关注。

### 趋势四：从 demo 到部署的鸿沟

大多数公开的机器人能力展示仍然是"demo 级别"——在受控环境中完成特定任务。从 demo 到可靠部署之间有一个巨大的工程鸿沟——"能完成任务"和"能连续运行 8 小时"根本不是同一个问题。这个鸿沟涵盖 task success → robustness → failure recovery → long-horizon autonomy → fleet-level reliability → maintenance → economic viability 等多个层级。Figure AI 在 BMW 的 11 个月部署测试是目前公开信息中较接近长期真实部署的案例之一。

## 这张地图意味着什么？

如果把当前具身智能的技术地图做一个总结：

```
                         Embodied AI
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
     Policy                Dynamics           Infrastructure
        │                     │                     │
   π₀ / Gemini            RSSM / TD-MPC2       Isaac / Cosmos
   GR00T / ...             Video WMs            World Labs
        │                     │                     │
        └──────────────┬──────┴──────────────┬──────┘
                       │                     │
                  Data / Simulation / Evaluation     Embodiment
                       │                     │
                       └──────────┬──────────┘
                                  │
                            Robot Systems
                                  │
                         ┌────────┴────────┐
                         │                 │
                     Industrial         General
                      deployment        humanoid
```

真正竞争的不是三条孤立路线，而是 policy、dynamics、data、simulation、embodiment 最终怎么组合。

几个判断：

**第一，技术路线还没有收敛。** Policy、Dynamics、Infrastructure 三个层面各有未解决的问题。最终的系统可能是混合架构——VLA 提供语义、世界模型提供预测、低层控制提供执行——但现在还看不到清晰的收敛方向。

**第二，软件、数据和模型正在成为越来越重要的差异化来源。** 随着人形机器人硬件逐渐出现更多标准化组件和成熟供应链，差异化来源正在越来越多地扩展到软件、数据和模型。但这个转变还在早期——actuator、hand、force sensing、whole-body control 等硬件能力仍然高度影响机器人的实际表现，hardware + software co-design 可能反而越来越重要。

**第三，从 demo 到部署的鸿沟是当前最大的挑战。** 大多数公开结果仍然是"在特定条件下能工作"，而不是"在真实环境中可靠运行"。解决这个问题的关键可能不是更大的模型，而是更好的数据、更鲁棒的策略、更成熟的 sim-to-real 链条，以及 failure recovery、monitoring 和 fleet-level reliability 等系统工程能力。

**第四，中国在硬件供应链、制造和成本控制方面具有明显优势，在商业化探索上也非常积极。** 宇树的 IPO 和多家公司的融资进展表明硬件和商业化能力已经得到市场认可。但在大规模可靠部署和通用基础模型方面，公开证据仍不足以证明已经形成同等优势。开源生态、公开论文和人才流动正在降低部分技术门槛，但差距的真实大小仍需要更多 benchmark 数据来量化。

---

*这篇是一个时间切面的快照。具身智能领域变化很快，半年后这张地图可能需要重画。*

*下一篇是 VLA 系列的下篇——VLA 与世界模型的关系辨析、开放问题与三个判断。*
