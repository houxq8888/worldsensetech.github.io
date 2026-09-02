---
title: '具身智能的数据问题：当架构逐渐收敛，什么在决定性能？'
slug: "2026-09-08-data-and-training-recipes"
date: 2026-09-08
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "机器人数据", "训练 Recipe", "遥操作", "合成数据", "Sim-to-Real", "数据 Curation", "VLA", "世界模型", "Scaling Law"]
description: "随着基础模型架构逐渐收敛，数据分布、数据质量和 training recipe 正越来越成为决定机器人性能的重要变量。但机器人数据不是'越多越好'——机器人领域真正需要 scaling 的不只是 trajectory 数量，而是 interaction distribution。"
toc: true
related_articles:
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-07-vla-world-models
  - 2026-09-05-vla-pi-family
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
---

在[前面的行业地图](/zh/articles/2026-09-06-embodied-ai-landscape/)中，我提到过一个越来越明显的趋势：单纯的模型架构差异正在变得不那么容易形成决定性优势，而数据规模、数据多样性和训练 recipe 的重要性正在上升。

这篇文章想把这个问题展开来讲。核心论点是：**随着基础模型架构逐渐收敛，数据分布、数据质量和 training recipe 正越来越成为决定机器人性能的重要变量。** 但"数据更重要"不等于"数据越多越好"——机器人领域真正需要 scaling 的，不只是 trajectory 数量，而是 interaction distribution。

## 为什么机器人数据和互联网数据不是一回事

大语言模型可以利用互联网规模的文本和代码数据进行预训练，数据获取规模远高于机器人真实交互数据。视觉模型也类似，大规模图文数据集为 VLM 提供了基础。

但机器人数据有一个根本性的不同：**它不只是"观察"，而是"交互"。**

一段互联网文本只需要 text；一张互联网图片只需要 pixel。但一条机器人数据的核心不是"state"或"reward"这样的单一字段，而是**带有 action 和时间结构的 multimodal interaction trajectory**：

```
robot trajectory:
  (o_t, a_t, o_{t+1}, ...)

可选字段：
  language instruction
  proprioception
  reward
  success / failure
  termination
  environment metadata
  task / embodiment ID
```

这里的关键是：observation（RGB / RGB-D / proprioception / force-torque / joint state / end-effector pose）是机器人通常能直接获取的；而真正的 environment state 往往是不可直接观测的。同样，reward 也不是 demonstration 数据的必需字段——它只在需要训练 reward model 或 actor-critic 策略时才出现。

这个区别不是细节问题，而是根本性的数据结构差异。它决定了具身智能不能简单地复制 LLM 的"数据 scaling"路线。

## 数据来源的几条路线

目前具身智能的数据来源大致可以分成四类。

### 遥操作数据（Teleoperation Data）

最直接的来源是人操控机器人完成任务，记录观测-动作轨迹对。

**优势：** 数据质量高，直接展示"成功完成任务"的行为模式；天然包含人类的操作策略和常识。

**局限：** 采集速度慢、成本高；操作者的技能水平直接影响数据质量；覆盖的任务多样性和环境多样性受限于操作者的时间和想象力。

目前主流的遥操作系统包括 VR 手柄控制、空间鼠标（SpaceMouse）、以及基于视觉的 imitation 系统。多家机器人公司正在建设规模化的遥操作数据采集基础设施，但具体数据量和覆盖范围通常不公开。

### 自主采集数据：Online 与 Offline 的区分

让机器人在真实或仿真环境中采集交互数据。这里需要做一个重要的区分。

**Online interaction：** 策略当前在环境中产生动作 $a_t \sim \pi(\cdot|o_t)$，然后得到新的 trajectory。典型问题是 exploration efficiency、safety、reset cost、on-policy distribution。经典 RL 通常依赖这种方式——agent 在环境中反复试错。

**Offline data：** 已有 replay / demonstration 数据 $D=\{(o_t,a_t,o_{t+1},r_t)\}$，不继续和环境交互。

但在当前机器人 RL 实践中，纯粹的 online RL 并不是唯一范式。越来越常见的方式包括：offline RL、demonstration + RL、imitation pretraining + online RL、replay-based RL、以及 simulation RL + real fine-tuning。因此，机器人 RL 的数据来源实际上是多样的——online interaction、offline trajectories、demonstration data 和 simulation-generated data 都在被广泛使用。

### 仿真数据（Simulation Data）

在仿真环境中生成训练数据。

**优势：** 可以大规模并行、精确控制环境参数、自动标注；可以生成真实环境中难以获取的极端场景数据。

**局限：** sim-to-real gap 仍然存在——仿真中的物理动态（接触力学、摩擦、变形）与真实世界不完全一致。仿真数据的分布和真实数据的分布之间存在 mismatch，直接使用可能导致策略在真实环境中表现退化。

NVIDIA Isaac Sim、MuJoCo 等仿真平台正在被广泛用于生成训练数据。但仿真数据通常需要配合 domain randomization、system identification 或 real-world fine-tuning 来弥合 gap。

### 合成数据：世界模型作为经验生成器

一个越来越重要的方向是：**用训练好的世界模型来扩大 agent 的经验。**

但这里需要区分两种不同的机制：

**Model-based RL（如 Dreamer）：** 世界模型作为 **latent experience generator**，在隐空间中产生 imagined trajectories。actor/critic 在 latent imagination 中训练：$z_t \rightarrow a_t \rightarrow z_{t+1}$，并不需要生成 photorealistic RGB frame。

**Generative world model（如视频生成式世界模型、NVIDIA Cosmos）：** 进一步尝试生成接近真实观测的合成数据（synthetic observations / videos / trajectories），从而作为下游训练的数据来源。

两者都是"用模型扩大经验"，但数据形态完全不同。前者是在隐空间中的 planning 和 training interface；后者是更接近传统意义上的"合成数据生成"。

这里的"世界模型"包含两个相关但不同的概念：用于 latent planning / imagination 的 dynamics model，以及用于生成或预测视觉世界的 generative world model。

## 不同范式的数据接口

这是容易被忽略但非常重要的一个维度：**不同的技术路线需要的不是同一种数据。**

### VLA 的数据接口

最基本的 VLA 训练样本可以抽象为 $(o_t, l, a_{t:t+k})$，其中 $l$ 是语言指令，$a_{t:t+k}$ 是 action chunk。

但实际系统还可能包含：proprioception、历史观测窗口、task metadata、embodiment information。动作输出也不只是简单的 $(o,l) \rightarrow a$——现代 VLA 可能使用 action chunk、diffusion / flow action head、discrete action token 或 continuous action，以及 heterogeneous action representation。

这意味着 VLA 对数据的核心需求是：**高质量的 observation-action 配对，覆盖足够多样的任务和物体，同时需要适配不同 embodiment 的动作表示。**

### 世界模型的数据接口

世界模型的核心接口是：

```
输入：(o_{≤t}, a_{≤t})
输出：predicted future latent / observation
可选：reward / termination / task outcome
```

需要注意的是，**世界模型本身并不必须有 reward**。在 Dreamer 中，reward prediction 和 continuation prediction 是训练 actor-critic 所需的重要组成部分，但它们属于整体 agent architecture 的其他模块，而非世界模型本身的必需输出。世界模型的核心功能是学习 action-conditioned dynamics——预测在给定动作序列后，未来状态如何变化。

对于以 latent dynamics + model-based control 为核心的路线（如 Dreamer 的 RSSM、TD-MPC2），数据需要是时间上连贯的、action-annotated 的交互轨迹。

### RL 的数据接口

RL 的数据需求取决于具体范式：

- **On-policy**（如 PPO）：需要当前策略产生的数据，数据"新鲜度"很重要
- **Off-policy**（如 SAC）：可以复用历史数据，但需要足够的 diversity 避免 overfitting
- **Offline RL**：完全依赖预收集的 dataset，对数据分布覆盖度要求极高
- **Imitation + RL**：先用 demonstration 预训练，再用 online interaction fine-tune

从数据角度看，不同 RL 范式对 replay buffer 或 dataset 的质量和多样性有非常不同的要求。

### 数据接口不兼容的问题

一个实际中经常遇到的问题是：**不同 embodiment、不同传感器配置、不同动作空间的数据通常不能不经处理地直接用于同一个低层 policy。**

一个在 Franka 机械臂上采集的数据，由于动作空间维度、观测视角、动力学特性的差异，通常需要经过 action retargeting、action normalization、或 embodiment conditioning 才能用于其他机器人。

这就是为什么 cross-embodiment data 是一个重要的研究方向。但需要注意术语的精确性：**multi-task**（同一机器人完成多种任务）、**multi-embodiment**（训练数据来自多种机器人但分别处理）、和 **cross-embodiment**（模型能泛化到未见过的机器人）是三个不同层次的问题。

TD-MPC2 的 multi-task / multi-domain 能力主要通过 task embedding 实现——但 task conditioning ≠ embodiment conditioning。Embodiment 差异涉及 action space、observation space、morphology、dynamics、control frequency 等多个维度，不能简单地用一个 task embedding 来解决。π₀ 系列的 cross-embodiment 能力则更多来自大规模多样化数据的训练，而非某种特定的 conditioning 机制。

## 数据不是 dataset，而是 distribution

"数据量"是一个容易被量化的指标，但在具身智能中，**数据的有效规模不能简单用 trajectory 数量衡量。**

### 数据质量 > 数据数量

一个普遍观察是，高质量 demonstration 与大规模视觉语言预训练结合，可以显著提升机器人策略的泛化能力。但"质量"需要更精确的定义——demonstration 中的系统性次优行为或错误 action 会改变行为策略的目标分布；如果没有 filtering 或 weighting 机制，这些模式可能被模型学习。

现代 policy learning 中存在多种应对机制：augmentation、trajectory weighting、filtering、robust loss、advantage weighting、diffusion policy smoothing 等。但核心挑战依然存在：**机器人数据中的"噪声"不只是标注错误，还包括操作不流畅、次优策略、传感器噪声等系统性问题。**

### 数据多样性与课程学习：两个不同的维度

数据多样性和课程学习是两个正交的维度：

- **Diversity：** 我见过多少不同情况？——决定覆盖面
- **Curriculum：** 我以什么顺序见到这些情况？——决定优化路径

如果训练数据只覆盖一种杯子、一种光照、一种桌面，策略在遇到变化时就会失败——这是 diversity 不足。而课程学习（从简单到复杂）是一种训练策略，影响的是优化路径而非覆盖面本身。**Diversity 决定覆盖面，curriculum 决定优化路径。**

### 数据 Curation：从趋势到技术

"更多数据"不自动等于"更好性能"。数据 curation 可以拆成几个可操作的维度：

```
Curation = Quality + Diversity + Coverage + Deduplication + Relevance + Balance
```

对于机器人数据，每个维度都有特定的技术挑战：

- **Quality：** success rate、action smoothness（jerk）、collision-free、action consistency
- **Diversity：** scene diversity、object variety、lighting variation
- **Coverage：** task coverage、failure mode coverage、edge case coverage
- **Deduplication：** 相似轨迹去重，避免 overfitting
- **Relevance：** 数据是否与目标任务相关
- **Balance：** 不同任务、不同场景的数据比例

这些问题目前还没有标准化的解决方案，但正在成为独立的技术方向。

## Training Recipe：决定模型看到什么

Training recipe 不只是 hyperparameters，而是**决定什么数据、以什么权重、以什么顺序、通过什么 objective 被模型看到**的一整套 pipeline。

具体来说，一个完整的 robot training recipe 可能包括：

- **Sampling strategy：** 不同数据源的采样比例（互联网数据 vs 机器人数据 vs 合成数据）
- **Trajectory weighting：** 高质量轨迹的权重是否更高？
- **Action chunking：** 预测单步动作还是动作序列？chunk 长度如何？
- **Temporal horizon：** 训练时使用的上下文窗口长度
- **Loss weighting：** 不同损失项（action prediction、value、reward）的相对权重
- **Augmentation：** 视觉增强、动作增强、domain randomization
- **Observation / action normalization：** 不同 embodiment 的观测和动作如何归一化
- **Freezing / unfreezing schedule：** 预训练 backbone 何时冻结、何时解冻
- **Mixture-of-data sampling：** 多源数据的混合策略
- **Intervention data / failure data：** 是否包含人类干预数据或失败轨迹
- **Replay ratio：** offline 数据被重复使用的次数
- **Offline/online mixing：** 是否结合 offline pretraining 和 online fine-tuning
- **Fine-tuning schedule：** 学习率、batch size、训练轮数的调度

不同团队在这方面的选择可能非常不同，而这些选择往往对最终性能有显著影响——有时甚至超过模型架构的选择。这也是为什么 training recipe 很难通过一篇论文完整传递——它是一整套工程实践，而不是一组超参数。

## Sim-to-Real：四种不同的策略

仿真数据不能直接替代真实数据，但有多种策略来处理 sim-to-real 的问题。这四种策略的逻辑各不相同：

```
System identification
  real world → calibrate simulator
  目标：让仿真器更接近真实系统

Domain randomization
  simulator → enlarge training distribution
  目标：训练一个对一组可能的 domain 都鲁棒的 policy
  （不是在"弥合 distribution gap"，而是在扩大训练分布）

Real-world fine-tuning
  sim → real adaptation
  目标：用少量真实数据适配已在仿真中学到的策略

Domain adaptation
  sim ↔ real representation alignment
  目标：学习仿真和真实之间的共享表示
```

这四种策略通常不是互斥的，实际系统中往往组合使用。

## 机器人数据 Scaling：不只是"更多轨迹"

LLM 领域已经建立了相对清晰的 scaling law（更多数据 + 更大模型 + 更多计算 → 可预测的性能提升）。机器人领域是否也存在类似的 scaling law？

机器人数据规模至少有 5 个维度：

```
D = (D_trajectory, D_task, D_environment, D_embodiment, D_quality)
```

因此机器人领域的 scaling law 可能不是：

$$Performance = f(N)$$

而更像：

$$Performance = f(N_{steps}, N_{tasks}, N_{scenes}, N_{embodiments}, Q)$$

这意味着：**机器人领域真正需要 scaling 的，不只是 data volume，而是 interaction distribution。**

LLM 可以粗略问"我有多少 token？"；机器人更应该问"我覆盖了多少种任务、状态、环境、动作、失败模式和 embodiment？"

```
Robot Data Scaling ≠ More Trajectories

Data Scaling = Volume × Quality × Diversity × Coverage × Embodiment
```

这是一个值得验证的假设：**在 interaction distribution（而非纯 trajectory 数量）上 scaling，可能是机器人领域更有效的 scaling 方向。**

## 这意味着什么？

如果把前面几篇文章的线索串起来：

- [世界模型系列](/zh/articles/2026-09-01-world-model-h2-review/)建立了"预测接口"的概念
- [VLA 系列](/zh/articles/2026-09-03-vla-deep-dive/)分析了"语义 + 动作"的接口设计
- [RSSM 演进](/zh/articles/2026-09-04-rssm-beyond/)讨论了不同 latent dynamics 的数据需求
- [行业地图](/zh/articles/2026-09-06-embodied-ai-landscape/)指出数据正在成为关键差异化因素

这篇想说的是：**数据和 training recipe 可能正在成为具身智能最被低估的竞争壁垒。**

模型架构可以通过论文和开源代码传播；仿真平台正在被少数几个玩家标准化；但**高质量的机器人交互数据、有效的数据 curation 流程、和经过反复调试的 training recipe——这些很难通过一篇论文完整传递。**

而核心问题不是"谁有更多数据"，而是"谁覆盖了更广的 interaction distribution"。

---

*这篇是具身智能系列的延伸——从"谁在做什么"转向"什么在驱动性能"。下一篇可能会讨论 sim-to-real 的方法论细节。*
