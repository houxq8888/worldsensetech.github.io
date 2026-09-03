---
title: '具身智能的数据全景（上篇）：来源、接口、distribution 与 training recipe'
slug: "2026-09-08-data-and-training-recipes"
date: 2026-09-08
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "机器人数据", "训练 Recipe", "遥操作", "合成数据", "Sim-to-Real", "数据 Curation", "VLA", "世界模型"]
description: "随着 VLA、世界模型等基础范式逐渐出现较清晰的主流路线，数据分布、数据质量和 training recipe 正越来越成为决定机器人性能的重要变量。本篇（上篇）梳理机器人数据的来源与接口、为什么'数据不是 dataset 而是 distribution'、training recipe 如何决定模型真正看到的分布，以及 sim-to-real 的四类常见工具；下篇将在此基础上讨论机器人数据 scaling 的理论框架。"
toc: true
related_articles:
  - 2026-09-10-robot-data-scaling
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-07-vla-world-models
  - 2026-09-05-vla-pi-family
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
---

> 这是"具身智能的数据问题"两篇系列中的**上篇**。上篇聚焦数据全景——来源、接口、distribution、training recipe 与 sim-to-real；关于机器人数据 scaling 的理论框架（interaction coverage、marginal data value、data flywheel、sequential data allocation）放在[下篇：机器人数据 Scaling](/zh/articles/2026-09-10-robot-data-scaling/)。

在[前面的行业地图](/zh/articles/2026-09-06-embodied-ai-landscape/)中，我提到过一个越来越明显的趋势：单纯的模型架构差异正在变得不那么容易形成决定性优势，而数据规模、数据多样性和训练 recipe 的重要性正在上升。

这两篇文章想把这个问题展开来讲。但需要先说明：这不是一篇"机器人数据有哪些"的综述，而是想提出一个关于机器人 scaling 的分析框架——其中本篇（上篇）先把机器人数据的来源、接口、distribution 与 training recipe 这几层讲清楚，下篇再在这个基础上把 scaling 框架正式搭起来。整个系列的核心假设是：

$$Performance \neq f(\#trajectory)$$

$$Performance = f(interaction\ distribution,\ data\ quality,\ recipe)$$

也就是说，**随着 VLA、world model 等路线逐渐形成若干主流范式（尽管具体架构仍在快速演进——diffusion / flow / autoregressive action head、latent vs video world model、action representation 都还没定型），单纯依赖局部架构创新形成稳定性能优势的难度可能正在上升；真正决定性能的，越来越是模型看到的 interaction distribution、数据的 quality，以及把数据转换成参数的 training recipe。**（这里并不声称 architecture 不重要、或已经收敛，只是提出竞争优势的来源可能发生迁移。）但"数据更重要"不等于"数据越多越好"——**本系列的核心假设是：机器人领域真正值得 scaling 的，不只是 trajectory 数量，而是 interaction distribution 相对于 evaluation distribution 的有效覆盖。**

这里先给出 interaction distribution 的定义，它会贯穿全文：**本文所说的 interaction distribution，是训练数据中由 task、scene、embodiment 等条件共同决定的 trajectory 分布**，记作

$$p(\tau \mid task,\ scene,\ embodiment)$$

其中 trajectory $\tau=(o_{0:T},a_{0:T-1})$ 本身已经包含 observation、action 和 temporal dynamics，以及可能的 success/failure 信息；如果想写得更显式，也可以表示为 $p(o_{0:T},a_{0:T-1}\mid task,scene,embodiment)$。之所以用条件分布而不是把 task、state、action 都塞进一个联合分布，是因为 state 和 action 已经在 $\tau$ 里，而 failure mode 往往是对 trajectory 做后验分析得到的标签 $m=h(\tau)$，并不是采集时就存在的原始随机变量。这个定义也比单纯讲 diversity 更强，因为 diversity ≠ distribution coverage——一个 dataset 可以有很多 object，但都来自同一种 task distribution。

不过这里应当诚实地给定义补一句升级说明：**trajectory 分布并不真的只由 task、scene、embodiment 决定。** 它还依赖环境动力学（dynamics）、初始状态 / reset 分布、数据采集策略（behavior policy / operator policy / exploration / intervention mechanism），以及 sensor/actuator dynamics。换句话说，$p(\tau\mid task,scene,embodiment)$ 实际上已经把"谁在怎么行动"这些因素**边缘化（marginalize）掉了**。更严谨的写法是

$$p_D(\tau \mid c),\qquad c=(task,\ scene,\ embodiment)$$

其中下标 $D$ 提醒我们：这个分布隐含地依赖具体的采集 policy 与环境。之所以全文仍写成 $p(\tau\mid task,scene,embodiment)$ 的简写，是为了记号统一；但请读者记住——**下篇一旦讨论 coverage，我们尤其关心的恰恰就是这个被藏进 $D$ 里的 behavior distribution。**

还有一个技术读者会立刻想到的问题：既然我们真正在意的是"访问了哪些 state/action 区域"而不是"有几条 trajectory"，那更贴切的数学对象其实应该是 RL 里的 **policy-induced occupancy measure** $d^\pi(s,a)$——它衡量的正是"在 evaluation-relevant 的 state-action 区域里访问了多少"，和下篇 support/density 的区分高度一致。本文之所以仍用 trajectory distribution，是一个**有意为之的高层抽象**：为了把 VLA、world model、imitation 与 RL 的数据放在同一套记号下讨论，我们选择停留在 trajectory level；若把全文改写成 occupancy measure，文章会立刻从"具身智能数据分析"滑向"RL theory paper"。所以这不是忽略了 $d^\pi(s,a)$，而是在抽象层级上主动选择了更粗的那一层。

## 为什么机器人数据和互联网数据不是一回事

大语言模型可以利用互联网规模的文本和代码数据进行预训练，数据获取规模远高于机器人真实交互数据。视觉模型也类似，大规模图文数据集为 VLM 提供了基础。

但机器人数据有一个根本性的不同：**它不只是"观察"，而是"交互"。**

这里需要加一个限定，否则容易被反驳：机器人其实也大量使用 observation-only data——egocentric video、internet video、human activity video、纯 RGB observation、passive observation，甚至一些 VLA 数据 pipeline 会用到没有 robot action 的视觉/语言数据。所以更严谨的说法不是"机器人数据不能是 observation-only"，而是：**与互联网文本/图像相比，机器人控制数据的核心增量不是 observation 本身，而是 action-conditioned temporal interaction。** 真正想强调的是 $(o_t)$ 与 $(o_t, a_t, o_{t+1})$ 之间的那部分信息差。

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

这里的关键是：observation（RGB / RGB-D / proprioception / force-torque / joint state / end-effector pose）是机器人通常能直接获取的；而真正的 environment state 往往是不可直接观测的。用控制的语言说，就是 $o_t \neq s_t$：observation 是 partial observation，而 state 是用于描述环境 Markov dynamics 的 underlying（往往是 latent）state。这也正是机器人数据天然具有 partial observability 的原因，并且和世界模型的 transition $p(z_{t+1}\mid z_t,a_t)$ 在理论上直接衔接。同样，reward 也不是 demonstration 数据的必需字段——它只在需要训练 reward model 或 actor-critic 策略时才出现。

这个区别不是细节问题，而是根本性的数据结构差异。它决定了具身智能不能简单地复制 LLM 的"数据 scaling"路线。

## 数据来源的几条路线

目前具身智能的数据来源大致分成四类，各自的 cost/quality/coverage profile 并不相同。

### 遥操作数据（Teleoperation Data）

人操控机器人完成任务、记录观测-动作轨迹对，是最直接的来源。

**优势：** 相比纯自主 exploration，更容易获得任务相关、成功率较高、且具有明确行为意图的 trajectory；天然包含人类的操作策略和常识。**局限：** 采集速度慢、成本高；操作者技能直接影响质量；覆盖的任务与环境多样性受限于操作者的时间和想象力。

需要注意的是，teleop 数据并不自动等于高质量数据——hesitation、correction、多余动作、inconsistent behavior、operator bias、失败尝试与 recovery、不同操作者的技能差异都会混在里面（这也正是后文 curation 要处理的问题：human-generated ≠ high-quality）。主流系统包括 VR 手柄控制、SpaceMouse、以及基于视觉的 imitation 系统；多家机器人公司正在建设规模化遥操作基础设施，但具体数据量与覆盖范围通常不公开。

### 自主采集数据：Online 与 Offline 的区分

让机器人在真实或仿真环境中自行采集交互数据，需要区分两种模式：**Online interaction** 由当前策略 $a_t \sim \pi(\cdot|o_t)$ 与环境持续交互，关心 exploration efficiency、safety、reset cost、on-policy distribution；**Offline data** 使用已有的 replay / demonstration 集 $D=\{(o_t,a_t,o_{t+1},r_t)\}$，不再与环境交互。经典 RL 通常属于前者，但在当前机器人 RL 实践中，offline RL、demonstration + RL、imitation pretraining + online RL、replay-based RL、simulation RL + real fine-tuning 等混合范式已经非常普遍——机器人 RL 的数据来源实际上是多样的。

### 仿真数据（Simulation Data）

在仿真环境中生成训练数据。

**优势：** 大规模并行、精确控制环境参数、自动标注，并可生成真实环境中难以获取的极端场景。**局限：** sim-to-real gap 仍在——接触力学、摩擦、变形等物理动态与真实世界不完全一致，直接使用可能导致策略在真实环境表现退化。

NVIDIA Isaac Sim、MuJoCo（Todorov et al., IROS 2012；现已由 DeepMind 开源）、GPU 并行物理仿真如 Isaac Gym（Makoviychuk et al., 2021，arXiv:2108.10470）都在被广泛用于生成训练数据；仿真数据通常还需要配合 domain randomization（Tobin et al., IROS 2017，arXiv:1703.06907）、system identification 或 real-world fine-tuning 来弥合 gap。

### 合成数据：世界模型作为经验生成器

一个越来越重要的方向是**用训练好的世界模型扩大 agent 的经验**。这里需要区分两种机制：**Model-based RL**（如 Dreamer，Hafner et al., 2019，arXiv:1912.01603；DreamerV3，Hafner et al., 2023，arXiv:2301.04104）把世界模型当作 **latent experience generator**——actor/critic 在 $z_t \rightarrow a_t \rightarrow z_{t+1}$ 的隐空间想象中训练，并不需要生成 photorealistic RGB frame；**Generative world model**（如视频生成式世界模型、NVIDIA Cosmos）则进一步尝试生成接近真实观测的合成 observation / video / trajectory。Cosmos 把自己定位为面向 Physical AI 的**可微调 world foundation model 平台 / digital twin**，其生成视频被视为下游开发（含机器人）的**潜在数据来源**——不过这更多是平台的愿景与定位，论文本身并未就"生成数据能提升真实机器人策略训练"给出直接的实证结论（NVIDIA，2025，arXiv:2501.03575）。两者共同的关键点是：**世界模型生成的数据并不是"免费的真实数据扩张"。** 生成轨迹的分布 $\hat p(\tau)$ 一般不等于真实分布 $p(\tau)$，model error 还会随 rollout horizon 累积（compounding error），可写成

$$D_{\mathrm{real}} \rightarrow M \rightarrow \hat D_{\mathrm{synthetic}}$$

因此 synthetic trajectory 的有效性同时受 model bias、long-horizon compounding error 以及生成分布与真实 interaction distribution 之间 mismatch 的限制——它本质上仍是一个 distribution 问题，而不是"数据变多了"。

## 不同范式的数据接口

这是一个容易被忽略但非常重要的维度：**不同的技术路线需要的不是同一种数据。**

### VLA 的数据接口

最基本的 VLA 训练样本可抽象为 $(o_t, l, a_{t:t+k})$，$l$ 是语言指令、$a_{t:t+k}$ 是 action chunk。但要注意：**action chunk 只是一种常见的训练/推理接口，而不是 VLA 的定义**——VLA 的核心是 $(V, L) \rightarrow A$ 的映射，chunk 形式属于具体的 policy parameterization，一个不显式预测 chunk 的系统仍然可以是 VLA。实际系统还常包含 proprioception、历史观测窗口、task metadata、embodiment information；动作输出也不止 $(o,l) \rightarrow a$，而可能是 action chunk、diffusion / flow action head、discrete token 或 continuous action 等 heterogeneous representation。代表性工作有 RT-2（Brohan et al., CoRL 2023，arXiv:2307.15818）与 π₀（Black et al., Physical Intelligence, 2024，arXiv:2410.24164）。

由此，VLA 对数据的核心需求是：**高质量的 observation-action 配对，覆盖足够多样的任务和物体，并适配不同 embodiment 的动作表示。** 更深层的一点是 **action representation 本身就是接口设计的一部分**：$a_{t:t+k}$ 可能是 joint position/velocity、end-effector delta / absolute pose、gripper command、discretized token、continuous flow 甚至 latent action，因此 cross-embodiment 的真正难点不是"把不同机器人的数据倒进同一个 dataset"，而是找到一个足够通用的 observation/action representation，让不同 embodiment 的经验能在同一学习空间里共享。

### 世界模型的数据接口

世界模型的核心接口是：

```
输入：observation history + action history

核心输出：
  future latent state / transition distribution
  如 p(z_{t+1} | z_t, a_t)

可选：
  reconstructed observation (p(o_t | z_t))
  reward / termination / task outcome
```

要点是：**世界模型的核心是学习 action-conditioned dynamics，reward prediction 并非 dynamics model 逻辑上必需的输出。** 但也不宜反过来规定"reward 不属于 world model"——更准确的分层是 `dynamics model`（状态转移）、`reward model`（预测 reward）、`continuation / termination model`（预测 episode 是否继续），一些文献把这几块合起来统称 **world model**；所以在 Dreamer 这类 agent 里，reward 与 continuation 常和 dynamics model 一起构成完整的 world-model module。对以 latent dynamics + model-based control 为核心的路线（Dreamer 的 RSSM、TD-MPC2，Hansen et al., ICLR 2024，arXiv:2310.16828），数据需要是时间连贯、action-annotated 的交互轨迹。

### RL 的数据接口

RL 的数据需求取决于具体范式：

- **On-policy**（如 PPO）：需要当前策略产生的数据，数据"新鲜度"很重要
- **Off-policy**（如 SAC）：可复用历史 replay，数据复用能力更强；但 off-policy ≠ 自动更 sample efficient——效率仍取决于 replay distribution、exploration、critic quality、reward structure 与任务本身，buffer 的分布与覆盖度会影响泛化和稳定性
- **Offline RL**：完全依赖预收集 dataset，对分布覆盖度要求极高
- **Imitation + RL**：先用 demonstration 预训练，再用 online interaction fine-tune

不同 RL 范式对 replay buffer / dataset 的质量与覆盖有着很不一样的要求。

### 数据接口不兼容的问题

一个现实中的常见困难是：**不同 embodiment、传感器配置、动作空间的数据，通常不能不经处理就塞进同一个低层 policy。** 例如在 Franka 上采集的数据，因动作空间维度、观测视角、动力学差异，往往要经过 action retargeting、action normalization 或 embodiment conditioning 才能迁移到其他机器人——这正是 cross-embodiment data 成为重要方向的原因。

这里要区分三个层次：**multi-task**（同一机器人做多任务）、**multi-embodiment**（数据来自多种机器人但分别处理）、**cross-embodiment**（模型能泛化到未见过的机器人）。TD-MPC2 的 multi-task / multi-domain 能力主要靠 task embedding 实现，但 **task conditioning ≠ embodiment conditioning**——embodiment 差异涉及 action space、observation space、morphology、dynamics、control frequency 等多个维度，不是一个 task embedding 能解决的；π₀ 系列则展示了大规模多 embodiment 数据对跨机器人泛化的重要性（这是一个 empirical observation，而非对具体机制的 causal attribution），Open X-Embodiment（Open X-Embodiment Collaboration, 2023，arXiv:2310.08864）正是这类跨本体大规模数据集的代表。

## 数据不是 dataset，而是 distribution

"数据量"是一个容易被量化的指标，但在具身智能中，**数据的有效规模不能简单用 trajectory 数量衡量。**

### 数据量不是有效数据规模

一个普遍观察是，高质量 demonstration 与大规模视觉语言预训练结合，可以显著提升机器人策略的泛化能力。但"质量"需要更精确的定义——demonstration 中的系统性次优行为或错误 action 会改变行为策略的目标分布；如果没有 filtering 或 weighting 机制，这些模式可能被模型学习。

现代 policy learning 中存在多种应对机制：augmentation、trajectory weighting、filtering、robust loss、advantage weighting、diffusion policy smoothing 等。但核心挑战依然存在：**机器人数据中的"噪声"不只是标注错误，还包括操作不流畅、次优策略、传感器噪声等系统性问题。**

### 数据多样性与课程学习：两个不同的维度

数据多样性和课程学习是两个正交的维度：

- **Diversity：** 我见过多少不同情况？——决定覆盖面
- **Curriculum：** 我以什么顺序见到这些情况？——决定优化路径

如果训练数据只覆盖一种杯子、一种光照、一种桌面，策略在遇到变化时就会失败——这是 diversity 不足。而课程学习（从简单到复杂）是一种训练策略，影响的是优化路径而非覆盖面本身。**Diversity 决定覆盖面，curriculum 决定优化路径。**

这里还需要区分 diversity 和 coverage：**Diversity 描述样本之间有多不同，coverage 描述目标任务分布被覆盖了多少。** 例如：1000 个不同杯子的数据 → diversity 很高；但如果全部都是"桌面抓取杯子"这一个任务，task coverage 可能仍然很低。也正因为如此，diversity 更像 curation 时值得盯着的一个抓手，而不是一个能直接换来 scaling 收益的独立量——在下篇 $D_{\mathrm{effective}}$ 的分解里，我们不再把它列为与 Coverage 平级的独立乘子，它只有通过 coverage 才真正进入 scaling。

### 数据 Curation：从趋势到技术

"更多数据"不自动等于"更好性能"。数据 curation 可以拆成六个可操作的维度。这里刻意用集合而不是加号表示——因为这些维度并不是可以直接相加的同类量（Quality、Diversity、Coverage、Relevance、Balance 是数据的属性，而 Deduplication 更像一种操作）：

$$\mathrm{Curation} = \{\mathrm{Quality},\ \mathrm{Diversity},\ \mathrm{Coverage},\ \mathrm{Relevance},\ \mathrm{Balance},\ \mathrm{Deduplication}\}$$

对于机器人数据，每个维度都有特定的技术挑战：

- **Quality：** success rate、trajectory smoothness（velocity/acceleration/jerk 等运动学指标）、collision-free、action consistency
- **Diversity：** scene diversity、object variety、lighting variation
- **Coverage：** task coverage、failure mode coverage、edge case coverage
- **Deduplication：** 相似轨迹去重，避免 overfitting
- **Relevance：** 数据是否与目标任务相关
- **Balance：** 不同任务、不同场景的数据比例

这些问题目前还没有标准化的解决方案，但正在成为独立的技术方向。

这里需要特别强调一点：**Curation 并不意味着简单删除失败轨迹。** 对 imitation learning 而言，明显错误的 demonstration 可能需要过滤；但对于 world model、offline RL 或 recovery policy，失败和边界轨迹本身可能具有很高的信息价值。例如：抓取失败、物体滑落、碰撞、grasp recovery、occlusion、unexpected contact——这些可能是 policy robustness 最需要的数据。成功轨迹告诉模型"这样做可以成功"，而失败轨迹可能告诉模型"在这个状态下，这种 action 会导致什么后果"。真正需要优化的是数据对目标 objective 的 relevance，而不是简单最大化 success rate。

更进一步抽象，**failure data 的价值并不在于"失败"本身，而在于它提供了 action-conditioned negative outcome information（动作条件下的负向结果信息）。** 一条 $(s, a_{\mathrm{bad}}, s')$ 告诉模型"在这个 state 下，这个 action 之后观测到了什么后果"；而如果只有成功 demonstration $(s, a_{\mathrm{good}}, s')$，模型并不一定知道 $a_{\mathrm{bad}}$ 为什么不好。严格说 $(s, a_{\mathrm{bad}}, s')$ 只是一个 observed transition，而非一次 controlled intervention——只有在 state、action 与 confounder 都能被干预时，才谈得上严格的 intervention evidence；这里用"动作条件下的负向结果"而不是"负向干预"，正是为了不把因果结论建立在纯观察数据之上。

这里需要精确一点：严格意义上的 counterfactual 是"在相同 $s$ 下如果采取另一个 $a'$ 会发生什么"，而我们观测到的只有 $(s, a_{\mathrm{bad}}, s')$，并没有同时观测到配对的 $(s, a_{\mathrm{good}}, s'')$。因此 failure trajectory 本身更准确的定位是 **counterfactual-relevant information**，而不是严格意义上的 counterfactual data——**只有当它与成功轨迹或模型预测结合起来时，才进一步具备 counterfactual learning value。** 即便如此，这种动作条件下的负向结果信号仍然让 failure trajectory 对 world model 和 offline RL 具有独特价值——这把"失败数据有用"从一句经验描述，提升到了一个更清晰的 learning-theoretic intuition。

### Data Quality ≠ Data Utility

前面反复用到 quality、relevance、success、diversity、coverage 这些词，但它们其实可以被一个更基础的概念统一起来：**Data Utility（数据效用）**。

先把两个概念彻底分开：

- **Quality** 描述的是 trajectory 层面**可度量的性质（measurable properties / quality indicators）**——smoothness、consistency、collision-free、sensor quality、annotation correctness。它们大多在数据采集完成的那一刻就能被算出来，因此*看起来*像是数据的"内在属性"；但严格说，连 quality indicator 本身也带着 objective 依赖：对精密 manipulation 重要的 smoothness，对 aggressive locomotion 或 recovery maneuver 未必是好事（突然的 jerk 可能恰恰是合理行为）；collision-free 在 normal policy 里是质量信号，但在 collision-recovery 数据集里，collision 本身就是你想学的东西。
- **Utility** 描述的是这些 properties 在**特定 objective 和特定 evaluation distribution** 下折算出来的 *conditioned contribution*——IL utility、world-model utility、offline-RL utility、recovery utility，甚至某个具体 eval task 上的 utility。同一条 trajectory 在不同 objective、不同 evaluation distribution 下 utility 可以完全不同。

```
Quality (measurable)              Utility (conditioned)
  smoothness                        IL utility
  consistency                       WM utility
  collision-free                    RL utility
  sensor quality                    recovery utility
  annotation correctness            eval-specific utility
```

用公式表达就是：**数据效用并不是一个 absolute property，而是一个 objective- 与 evaluation-conditioned quantity。** 同一份数据 $D$，对不同训练目标 $\mathcal{L}$ 和不同 evaluation distribution $p_{\mathrm{eval}}$，效用并不相同：

$$U(D \mid \mathcal{L},\ p_{\mathrm{eval}})$$

$$U_{\mathrm{IL}}(D) \neq U_{\mathrm{WM}}(D) \neq U_{\mathrm{offlineRL}}(D)$$

这恰好解释了前面那个现象：一条 failure trajectory 对 imitation learning 可能是需要过滤的 noise，但对 world model 或 offline RL 却可能是 valuable signal——因为它在不同 objective 下的 utility 不同。所谓"高质量数据"，严格说应该是"对当前 objective、在当前 evaluation distribution 下高 utility 的数据"。

把 quality 重新理解成 utility，也顺带解决了 curation 里一个常见的误区：不存在一份"universally 好"的数据集，只存在"对某个 $(\mathcal{L}, p_{\mathrm{eval}})$ 好"的数据集。这也是为什么 data curation 必须和 training objective、evaluation distribution 一起定义，而不能脱离目标单独谈"数据质量"。

还值得补一句：严格来说，data utility 甚至还会随 **model class、当前训练状态和 compute budget** 改变——同一份数据对一个小模型可能没用、对一个大模型却非常有用；模型已经学会 A 之后再补 A 的数据价值很低，还没学会时则很有价值。因此本文的 $U$ 是一种**面向当前训练阶段的条件效用**，而不是数据的静态属性。（不过为了不让框架过度形式化，我们仍然只把它写成 $U(D \mid \mathcal{L}, p_{\mathrm{eval}})$，把 $M$、$C$ 这些依赖留在文字里。）

至于 $p_{\mathrm{eval}}$ 具体是什么、为什么它是这一版最关键的补丁，我们在[下篇](/zh/articles/2026-09-10-robot-data-scaling/)的机器人 scaling 框架里再正式引入——那里会说明：**没有 evaluation distribution 做参照系，"coverage"其实是一句没有主语的话。**

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

不同团队在这方面的选择可能非常不同，而这些选择往往对最终性能有显著影响——在许多团队的实践中，其影响甚至可能超过模型架构的选择。需要说明的是，这一判断目前更多来自工程经验与个案报告，尚缺乏跨任务的系统性定量对照实验；正因如此，它更适合作为一个值得检验的假设，而非已确立的结论。这也是为什么 training recipe 很难通过一篇论文完整传递——它是一整套工程实践，而不是一组超参数。

从更抽象的角度看，**training recipe 本质上是数据分布到模型参数之间的"转换函数"**：

$$D \rightarrow Training\ Recipe \rightarrow \theta$$

但更准确地说，recipe 并不是 distribution 之外的一个独立因素——**recipe 本身就会改变模型"真正看到的 distribution"。** 原始数据 $D$ 先经过 sampling $S_R(D)$，再经过 weighting $W_R(S_R(D))$，真正进入 optimization 的其实是被 recipe 变换后的 $D_R$：

$$D \xrightarrow{\ Recipe\ } D_R \xrightarrow{\ Optimization\ } \theta,\qquad D_R = W_R\big(S_R(D)\big)$$

用分布的语言写，就是把 recipe $R$ 看成一个作用在轨迹分布上的 transformation operator $T_R$：

$$p_{\mathrm{train}}(\tau) = T_R\big[\,p_{\mathrm{raw}}(\tau)\,\big]$$

也就是说，performance 依赖的是模型实际经历的 $p_{\mathrm{train}}(\tau)$，而不是仓库里静态的 $p_{\mathrm{raw}}(\tau)$；$p_{\mathrm{train}}$ 正是 recipe 对 raw distribution 做重采样、重加权、重排序之后的结果。相同 dataset $D$，换一个 sampling / weighting / objective / schedule（$R_1 \neq R_2$），就可能得到不同的 $p_{\mathrm{train}}$，进而 $\theta_1 \neq \theta_2$——这把"recipe 是壁垒"的观点从经验判断提升到了一个更清晰的技术框架。

不过要提醒：上面把 $p_{\mathrm{train}}$ 写成由 recipe 固定变换得到的静态分布，只对 **offline / fixed-dataset training** 成立。在 **online RL 或持续数据闭环系统**里，$p_{\mathrm{train},t} \rightarrow \theta_t \rightarrow \pi_t \rightarrow p_{\mathrm{collect},t+1}$ 本身是一个 **feedback loop**——policy 更新会反过来改变后续采集到的分布。本文为保持记号简洁，在非必要处仍省略时间下标 $t$，但读者在碰到 online / closed-loop 场景时应当把 $p_{\mathrm{train}}$ 理解为 $p_{\mathrm{train},t}$。

### Recipe 的两条作用路径

但如果只写到这里，读者会立刻发现一个 double-counting 的问题：既然 $p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$ 已经把 recipe 折叠进了 distribution，那下篇里再让 recipe 作为独立参数出现在 $Performance = g(D_{\mathrm{effective}},\ Capacity,\ Compute,\ Recipe)$ 中，不就把它算了两次？

但 **recipe 并不只是 distribution transformation，它同时对 optimization dynamics 起作用。** 为分析方便，可以把 recipe 的主要作用**粗略**分成两条并行路径（这条分界并不是一次干净的划分，后文会看到）：

```
              ┌── Path 1: Distribution Transformation ──►  p_train(τ) = T_R[p_raw(τ)]
              │
Raw Data D ───┤
              │
              └── Path 2: Optimization Dynamics ─────────►  learning-rate schedule
                                                            optimizer / momentum
                                                            loss weighting
                                                            freezing / unfreezing
                                                            curriculum / staging
                                                            gradient clipping
```

Path 1 决定了模型"看到什么"，Path 2 决定了模型"如何把看到的东西转成参数"。二者不能被互相还原：同一份 $p_{\mathrm{train}}$，配上不同的 lr schedule、optimizer、loss weighting 或 freezing 策略，仍然会得到显著不同的 $\theta$——这不是分布变了，而是优化过程本身变了。

需要提醒的是，这个"两条路径"只是**分析上的方便切分，并不是一次干净的 partition**。有些 recipe 选择同时横跨两条路径：最典型的就是 loss weighting——$L = \lambda_a L_{\text{action}} + \lambda_v L_{\text{value}}$ 既改变了不同数据/目标项的 effective weighting（Path-1 的味道），也直接改变了 optimization dynamics（Path-2）。同样地，image augmentation、action tokenization、temporal window / observation stacking、action chunk 构造、hindsight relabeling、reward / label 构造这类 representation 与 target 操作，也同时影响 effective distribution 和 optimization target。我们并不打算为此再立一条正式的"Path 3"——那只会让分类更碎；只是想说明：正因为这类操作横跨两侧，"两条路径"本身就是一次 coarse-graining，我们并不声称"任何 recipe 都能被唯一地分解成这两条路径"，只是想用它来说明：recipe 的作用至少有一部分是 $p_{\mathrm{train}}$ 无法吸收的。

因此，**下篇的 $g(D_{\mathrm{effective}},\ Capacity,\ Compute,\ Recipe)$ 中，$Recipe$ 保留的是 Path 2 那一段无法被 $p_{\mathrm{train}}$ 吸收的优化动力学**；Path 1 那一段已经体现在 $D_{\mathrm{effective}}$ 里。这个划分不是纯粹的记号问题——它直接决定后面"哪些手段算改数据、哪些手段算改训练"的判断，也是很多团队 recipe 差异真正难以复现的地方：论文能公开 Path 1（数据 mixture、weighting），但 Path 2 里那些"什么时候解冻 backbone、什么时候切 lr、什么时候换 loss weighting"的经验，往往不会写全。

### Data Mixture：数据怎么混？

一个容易被忽略但极其关键的问题是：**不同数据源怎么混合？**

```
以下比例仅用于说明 mixture 的概念，并非行业统计：

Internet / VLM data      ── 70%
Robot demonstrations     ── 20%
Synthetic / simulation   ── 10%
```

真正关键的可能不是"我们有多少机器人数据？"，而是：**机器人数据在整个 training mixture 中占多少？什么时候加入？以什么 loss 训练？** 这恰好是 training recipe 作为"转换函数"的核心体现——相同的数据，不同的 mixture 比例和调度策略，可能产生截然不同的模型能力。

## Sim-to-Real：四类常见工具

仿真数据不能直接替代真实数据，但有多种工具来处理 sim-to-real 的问题。需要说明的是，下面这四类并不是同层级、互斥的分类，而是作用在不同 abstraction level 上的工具：system identification 属于 model calibration，domain randomization 属于 training distribution manipulation，real-world fine-tuning 属于 optimization strategy，domain adaptation 属于 representation / distribution alignment。它们分别作用在 simulation fidelity、training distribution、representation 和 policy adaptation 的不同层面，因此可以组合使用。

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

这里需要区分两种不同类型的 sim-to-real 误差：**随机噪声**（如传感器噪声、微小物理参数波动）和 **systematic simulation bias**（如摩擦系数长期偏差、actuator delay、contact model 误差、deformable object dynamics 误差、camera latency、calibration error）。随机噪声可以通过 domain randomization 来增强鲁棒性；而 systematic bias 通常需要借助 system identification 来校准仿真器本身。更准确的说法是：**system identification 主要针对 systematic simulator mismatch（把仿真器本身"拉回"到真实系统附近），domain randomization 则主要通过扩大训练分布来提高策略对参数与环境变化的鲁棒性——当 randomization 的范围覆盖到真实系统时，它同样能缓解一部分 systematic mismatch，但方式不是校准，而是让策略学会"对一整族仿真器都工作"。** 两者并不互斥，也常常同时使用。

## 小结：为什么"数据全景"是 scaling 讨论的前提

到这里，上篇把机器人数据的几件事讲清楚了：它的**来源与接口**（teleoperation、simulation、autonomous exploration、synthetic generation，以及 observation ≠ state 这个根本结构）、为什么**数据不是 dataset 而是 distribution**（interaction distribution、quality ≠ utility、curation 的多个维度），以及 **training recipe 如何决定模型真正看到的分布**（$p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$、两条作用路径、data mixture），最后落到 **sim-to-real 的四类工具**上。

但这些都还停在"数据从哪里来、以什么形式进入训练"这一层。真正的 scaling 问题——**下一单位预算应该增加什么数据、它值不值**——需要一套更严格的框架：evaluation distribution 如何给 coverage 提供参照系、support / density / distribution similarity 三者如何分开、marginal data value 如何定义、data flywheel 与 sequential data allocation 如何把这套框架变成可操作的问题。这些放在[下篇：机器人数据 Scaling](/zh/articles/2026-09-10-robot-data-scaling/)。

## 参考文献

上篇涉及的主要工作如下（均可通过 arXiv ID 检索）：

- RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control — Brohan et al., CoRL 2023, arXiv:2307.15818
- π₀: A Vision-Language-Action Flow Model for General Robot Control — Black et al., Physical Intelligence, 2024, arXiv:2410.24164
- Open X-Embodiment: Robotic Learning Datasets and RT-X Models — Open X-Embodiment Collaboration, 2023, arXiv:2310.08864
- Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) — Hafner et al., 2019, arXiv:1912.01603
- Mastering Diverse Domains through World Models (DreamerV3) — Hafner et al., 2023, arXiv:2301.04104
- TD-MPC2: Scalable, Robust World Models for Continuous Control — Hansen et al., ICLR 2024, arXiv:2310.16828
- Cosmos World Foundation Model Platform for Physical AI — NVIDIA, 2025, arXiv:2501.03575
- MuJoCo: A physics engine for model-based control — Todorov, Erez & Tassa, IROS 2012, DOI:10.1109/IROS.2012.6386109
- Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning — Makoviychuk et al., 2021, arXiv:2108.10470
- Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World — Tobin et al., IROS 2017, arXiv:1703.06907

与上篇"data / distribution 才是关键"这一论点更直接相关的，是以下几类聚焦数据集本身（采集规模、多样性、质量筛选、仿真—真实混合）的实证研究：

- DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset — Khazatsky et al., 2024, arXiv:2403.12945（大规模、多场景真机操作数据集；它直接证明的是"数据规模与环境/任务多样性"，而非"diversity → scaling 收益"这一因果命题，后者仍是本系列的假设）
- SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning — Zhang et al., 2025, arXiv:2505.22626（自监督、可组合的数据清洗/质量筛选方法）
- Consistency Matters: Defining Demonstration Data Quality Metrics in Robot Learning from Demonstration — Sakr et al., 2024, arXiv:2412.14309（用一致性等质量指标衡量 demonstration，而非默认"人类演示 = 高质量"）
- Efficient Data Collection for Robotic Manipulation via Compositional Generalization — Gao et al., 2024, arXiv:2403.05110（通过对场景元素的组合式泛化降低数据采集成本）
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., 2025, arXiv:2503.24361（仿真与真实数据混合训练的系统性 recipe 研究）

（关于 scaling law 的经典工作——Kaplan et al., 2020, arXiv:2001.08361；Chinchilla / Hoffmann et al., 2022, arXiv:2203.15556；以及机器人模仿学习的 data scaling law——Lin et al., 2024, arXiv:2410.18647——随 scaling 框架一并放在[下篇](/zh/articles/2026-09-10-robot-data-scaling/)的参考文献里。）

需要说明的是，机器人学习目前尚不存在像 LLM 那样公认的单一 scaling law；本系列关于 effective data scale 的框架是一个 conceptual decomposition 与可检验假设，而非既成结论。上述数据侧工作提供的是分散的实证支持，尚不足以构成对该假设的完整定量验证。

---

*本篇是"具身智能的数据问题"两篇系列的上篇。下篇[《机器人数据 Scaling：从 interaction coverage 到 marginal data value》](/zh/articles/2026-09-10-robot-data-scaling/)将把这里的数据全景推进为一套可讨论的 scaling 框架。*
