---
title: '具身智能的数据问题：当主流范式逐渐清晰，什么在决定性能？'
slug: "2026-09-08-data-and-training-recipes"
date: 2026-09-08
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "机器人数据", "训练 Recipe", "遥操作", "合成数据", "Sim-to-Real", "数据 Curation", "VLA", "世界模型", "Scaling Law"]
description: "随着 VLA、世界模型等基础范式正在出现较清晰的主流路线，数据分布、数据质量和 training recipe 正越来越成为决定机器人性能的重要变量。但机器人数据不是'越多越好'——机器人领域真正需要 scaling 的不只是 trajectory 数量，而是相对于目标 evaluation distribution 的有效 interaction coverage。"
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

这篇文章想把这个问题展开来讲。但需要先说明：这不是一篇"机器人数据有哪些"的综述，而是想提出一个关于机器人 scaling 的分析框架。它的核心假设是：

$$Performance \neq f(\#trajectory)$$

$$Performance = f(interaction\ distribution,\ data\ quality,\ recipe)$$

也就是说，**随着 VLA、world model 等路线逐渐形成若干主流范式（尽管具体架构仍在快速演进——diffusion / flow / autoregressive action head、latent vs video world model、action representation 都还没定型），单纯依赖局部架构创新形成稳定性能优势的难度可能正在上升；真正决定性能的，越来越是模型看到的 interaction distribution、数据的 quality，以及把数据转换成参数的 training recipe。**（这里并不声称 architecture 不重要、或已经收敛，只是提出竞争优势的来源可能发生迁移。）但"数据更重要"不等于"数据越多越好"——**本文的核心假设是：机器人领域真正值得 scaling 的，不只是 trajectory 数量，而是 interaction distribution 相对于 evaluation distribution 的有效覆盖。**

这里先给出 interaction distribution 的定义，它会贯穿全文：**本文所说的 interaction distribution，是训练数据中由 task、scene、embodiment 等条件共同决定的 trajectory 分布**，记作

$$p(\tau \mid task,\ scene,\ embodiment)$$

其中 trajectory $\tau=(o_{0:T},a_{0:T-1})$ 本身已经包含 observation、action 和 temporal dynamics，以及可能的 success/failure 信息；如果想写得更显式，也可以表示为 $p(o_{0:T},a_{0:T-1}\mid task,scene,embodiment)$。之所以用条件分布而不是把 task、state、action 都塞进一个联合分布，是因为 state 和 action 已经在 $\tau$ 里，而 failure mode 往往是对 trajectory 做后验分析得到的标签 $m=h(\tau)$，并不是采集时就存在的原始随机变量。这个定义也比单纯讲 diversity 更强，因为 diversity ≠ distribution coverage——一个 dataset 可以有很多 object，但都来自同一种 task distribution。

不过这里应当诚实地给定义补一句升级说明：**trajectory 分布并不真的只由 task、scene、embodiment 决定。** 它还依赖环境动力学（dynamics）、初始状态 / reset 分布、数据采集策略（behavior policy / operator policy / exploration / intervention mechanism），以及 sensor/actuator dynamics。换句话说，$p(\tau\mid task,scene,embodiment)$ 实际上已经把"谁在怎么行动"这些因素**边缘化（marginalize）掉了**。更严谨的写法是

$$p_D(\tau \mid c),\qquad c=(task,\ scene,\ embodiment)$$

其中下标 $D$ 提醒我们：这个分布隐含地依赖具体的采集 policy 与环境。之所以全文仍写成 $p(\tau\mid task,scene,embodiment)$ 的简写，是为了记号统一；但请读者记住——**后文一旦讨论 coverage，我们尤其关心的恰恰就是这个被藏进 $D$ 里的 behavior distribution。**

还有一个技术读者会立刻想到的问题：既然我们真正在意的是"访问了哪些 state/action 区域"而不是"有几条 trajectory"，那更贴切的数学对象其实应该是 RL 里的 **policy-induced occupancy measure** $d^\pi(s,a)$——它衡量的正是"在 evaluation-relevant 的 state-action 区域里访问了多少"，和后文 support/density 的区分高度一致。本文之所以仍用 trajectory distribution，是一个**有意为之的高层抽象**：为了把 VLA、world model、imitation 与 RL 的数据放在同一套记号下讨论，我们选择停留在 trajectory level；若把全文改写成 occupancy measure，文章会立刻从"具身智能数据分析"滑向"RL theory paper"。所以这不是忽略了 $d^\pi(s,a)$，而是在抽象层级上主动选择了更粗的那一层。

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

这里的关键是：observation（RGB / RGB-D / proprioception / force-torque / joint state / end-effector pose）是机器人通常能直接获取的；而真正的 environment state 往往是不可直接观测的。用控制的语言说，就是 $o_t \neq s_t$：observation 是 partial observation，而 state 是用于描述环境 Markov dynamics 的 underlying（往往是 latent）state。这也正是机器人数据天然具有 partial observability 的原因，并且和后文世界模型的 transition $p(z_{t+1}\mid z_t,a_t)$ 在理论上直接衔接。同样，reward 也不是 demonstration 数据的必需字段——它只在需要训练 reward model 或 actor-critic 策略时才出现。

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

这里还需要区分 diversity 和 coverage：**Diversity 描述样本之间有多不同，coverage 描述目标任务分布被覆盖了多少。** 例如：1000 个不同杯子的数据 → diversity 很高；但如果全部都是"桌面抓取杯子"这一个任务，task coverage 可能仍然很低。也正因为如此，diversity 更像 curation 时值得盯着的一个抓手，而不是一个能直接换来 scaling 收益的独立量——在后文 $D_{\mathrm{effective}}$ 的分解里，我们不再把它列为与 Coverage 平级的独立乘子，它只有通过 coverage 才真正进入 scaling。

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

至于 $p_{\mathrm{eval}}$ 具体是什么、为什么它是这一版最关键的补丁，我们在下一节的机器人 scaling 框架里再正式引入——那里会说明：**没有 evaluation distribution 做参照系，"coverage"其实是一句没有主语的话。**

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

但如果只写到这里，读者会立刻发现一个 double-counting 的问题：既然 $p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$ 已经把 recipe 折叠进了 distribution，那下一节里再让 recipe 作为独立参数出现在 $Performance = g(D_{\mathrm{effective}},\ Capacity,\ Compute,\ Recipe)$ 中，不就把它算了两次？

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

因此，**下一节的 $g(D_{\mathrm{effective}},\ Capacity,\ Compute,\ Recipe)$ 中，$Recipe$ 保留的是 Path 2 那一段无法被 $p_{\mathrm{train}}$ 吸收的优化动力学**；Path 1 那一段已经体现在 $D_{\mathrm{effective}}$ 里。这个划分不是纯粹的记号问题——它直接决定后面"哪些手段算改数据、哪些手段算改训练"的判断，也是很多团队 recipe 差异真正难以复现的地方：论文能公开 Path 1（数据 mixture、weighting），但 Path 2 里那些"什么时候解冻 backbone、什么时候切 lr、什么时候换 loss weighting"的经验，往往不会写全。

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

## 机器人数据 Scaling：不只是"更多轨迹"

LLM 领域已经积累了较成熟的 scaling-law empirical framework（Kaplan et al., 2020，arXiv:2001.08361；Hoffmann et al., 2022，arXiv:2203.15556），描述在特定 compute-optimal / loss-scaling regime 下数据、参数与算力如何共同决定 loss——但这不是一条统一的"自然定律"，也不直接可以搬到机器人上。机器人领域是否也存在类似的 scaling law？

### Data Acquisition ≠ Data Scaling

在谈 scaling 之前，需要先区分两个经常被混为一谈的问题。

**Data acquisition** 回答"数据从哪里来"（teleoperation、simulation、autonomous exploration、synthetic generation 都是 *acquisition method*，本文前半部分讲的就是这一层）；**Data scaling** 回答"下一单位预算应该增加什么数据"（support expansion、density improvement、failure targeting、embodiment expansion 都是 *scaling strategy*）。一个是生成机制，一个是 allocation problem——把 acquisition 做得再强，也不自动回答 scaling。后半部分的重心因此从"数据从哪里来"转向"什么数据值得继续增加"，这也正是下面这套框架要回答的。

首先需要明确：**下面的公式不是严格的 scaling law，而是一个用于描述机器人数据有效规模的 conceptual decomposition。** 机器人数据规模至少可以分解为三个层面：

**Data volume：** $N_{\text{steps}}$（总交互步数）

**Distribution dimensions：** $task, scene, embodiment, \text{behavioral state}, action$（分布维度）

**Data quality：** $Q$（数据质量）

这里需要澄清 "state" 这一维：结合前文 $o_t \neq s_t$ 的讨论，我们说的并不是必须显式标注的 environment state（真实 $s$ 往往不可直接获得），而是 **behavioral-state coverage / state-space coverage**——即模型在训练过程中实际经历到的（往往是 latent 或 inferred 的）行为状态分布。这样它就不会和前面的 partial observability 讨论产生概念冲突。

### Distribution ≠ Coverage：引入 Evaluation Distribution

在展开 coverage 之前，必须先补一个此前一直被隐含使用的概念。前文我们把 interaction distribution 定义成

$$p(\tau \mid task,\ scene,\ embodiment)$$

这是一个**概率分布**。但在讨论 scaling 时，我们真正关心的其实是 distribution 的若干 *性质*——coverage、diversity、support、density——它们并不是同一件事：

$$\boxed{Distribution \neq Coverage}$$

更麻烦的是，"coverage"这个词单独出现时是**没有参照系的**。要说"训练数据覆盖得广不广"，就必须先回答"覆盖什么？"。这就把 evaluation distribution 逼出来了。

到目前为止我们只写了训练侧的分布：

$$p_{\mathrm{train}}(\tau)$$

但真正决定 performance 的，是它和 evaluation distribution 之间的关系：

$$p_{\mathrm{eval}}(\tau)$$

```text
training distribution
        ↓
 p_train(τ)

          ↕ mismatch / coverage

evaluation distribution
        ↓
 p_eval(τ)
```

在这里一次性交代两个限定，后文就不再重复。其一，本文把 $p_{\mathrm{eval}}$ 抽象成 **trajectory-level 分布**只是为了统一记号；在具体 benchmark 里，它往往更自然地定义在 task / scene / initial-state 等 **context** 上（记作 $p_{\mathrm{eval}}(c)$），再由 policy 在该 context 下诱导出 trajectory 分布。其二，把 $p_{\mathrm{eval}}$ 写成**固定**参考分布，主要是为了给 coverage 一个坐标系；但在 **closed-loop / online RL** 中，evaluation 时实际访问到的 state / trajectory 分布本身会被当前 policy 改变（即 $d^{\pi_\theta}$），严格说 $p_{\mathrm{eval}}$ 也可能是 **policy-dependent** 的。本文把这种 feedback effect 吸收进 evaluation distribution 里，而不展开成一套完整的 policy-induced distribution analysis。

一旦把 $p_{\mathrm{eval}}$ 显式写出来，很多原本模糊的直觉就会立刻清晰。考虑两个 dataset：

- **Dataset A：** 覆盖 100 个 task × 100 个 scene × 10 个 embodiment，但每个组合下只有很少的 trajectory。
- **Dataset B：** 只有 10 个 task × 10 个 scene × 1 个 embodiment，但每个组合下都有几十万条高质量 trajectory。

谁更好？答案是：**取决于 $p_{\mathrm{eval}}$、model capacity、任务对 precision 与 coverage 的相对需求，以及 optimization budget**。如果 evaluation 集中在少量高精度 manipulation 任务上，B 大概率更好；如果 evaluation 是开放世界多任务泛化，A 才有可能占优。所以更严谨的说法不是"coverage 决定 scaling"，而是：

> **Performance 取决于 $p_{\mathrm{train}}$ 覆盖 $p_{\mathrm{eval}}$ 相关区域的程度，以及在那些区域里的采样密度和数据质量。**

写成公式：

$$\Delta Performance \approx f\big(\Delta p_{\mathrm{train}},\ p_{\mathrm{eval}}\big)$$

这里还有一个值得点破的细节：**coverage 本身并不是一个天然的标量（scalar），也不是 training distribution 的绝对属性。** 设想 Dataset A 的 task coverage 很高但 scene coverage 很低，Dataset B 相反——到底"谁 coverage 更高"？在没有参照系时这个问题根本无法回答。所以更准确的写法是把它当成一个**关系量**：

$$\text{Coverage} = C\big(p_{\mathrm{train}},\ p_{\mathrm{eval}}\big),\qquad \text{而不是}\quad C(p_{\mathrm{train}})$$

也就是说，我们真正关心的从来不是某个 dataset 自带的"coverage score"，而是 $p_{\mathrm{train}}$ 相对于一个指定 $p_{\mathrm{eval}}$ 的覆盖程度。

这里还要立刻划清一条容易混的界线：**coverage、density、distribution similarity 是三件不同的事，不能被 "coverage" 一个词笼统盖住。** 上面的 $C(p_{\mathrm{train}},p_{\mathrm{eval}})$ 其实同时指向三个各自独立的量——**support coverage**（见没见过 evaluation-relevant 区域，一个偏 0/1 的问题）、**density**（见过的区域采了多少，是一个强度问题）、以及 **distribution similarity**（两个分布整体有多像，通常由某个距离度量给出）。三者会给出不同的排序：仍是上面的 Dataset A / B，用 support coverage 衡量时铺满整个 evaluation support 的 A 明显占优；用某个整体距离（如 $D_{KL}(p_{\mathrm{eval}}\,\|\,p_{\mathrm{train}})$，即 distribution similarity）衡量时，把 80% 区域高密度覆盖的 B 反而可能更低；而当 evaluation metric 对某个核心区域特别敏感时，高密度 B 又可能更好。所以本文的 support / density 分解**并不等价于"找一个单一的 distribution distance 把它最小化"**——它刻意把"见没见过""采了多少""像不像"当成三个独立问题来谈，也只有先分开，才谈得上后面把预算**分配（allocation）**到最该补的那一个上。

把这一步点明之后，下面这个 utility 定义也就不是凭空冒出来的记号，而是顺着"coverage 是关系量"这条线自然推出来的结果：这也直接把前文 utility 的定义收紧了一档——**数据效用不只是 objective-conditioned，还是 evaluation-conditioned**。

$$\boxed{U(D \mid \mathcal{L},\ p_{\mathrm{eval}})}$$

举个最简单的例子。假设 evaluation 是"厨房中不同光照条件下抓取杯子"：

- 新杯子 → 可能有价值（扩大 object support）
- 新厨房 → 有价值（扩大 scene support）
- 新光照 → 有价值（扩大 visual-condition support）
- 新机器人 embodiment → 未必有价值（ morphology 变了不代表 evaluation 变了）
- 新任务"叠衣服" → 几乎没价值（离开了 $p_{\mathrm{eval}}$ 的 support）

结论很直接：**"新数据"没有绝对价值，只有相对于 evaluation distribution 的价值。**

### Coverage 到底覆盖什么：不同维度负责不同泛化

"增大 distribution coverage"这句话如果不进一步拆解，其实还是太笼统。更准确的说法是：**不同的 distribution dimension 对不同的泛化问题负责，它们并不能混为一谈。**

可以分别写成条件分布的形式（注意：下面这些条件分布只是**描述 coverage 的分析视角**，并不是一条完整的 trajectory generative factorization——真实里 $s$ 依赖 history、$a$ 依赖 policy / observation / embodiment、embodiment 又会反过来影响 action space，且 $s$ 与 $a$ 在时间上互相塑造）：

$$p(task)\quad(\text{任务语义空间})$$

$$p(scene \mid task)\quad(\text{环境条件})$$

$$p(s \mid task, scene)\quad(\text{任务执行中访问到的 behavioral state})$$

$$p(a \mid s)\quad(\text{行为策略在给定 state 下实际采取的动作})$$

$$p(embodiment)\quad(\text{机器人形态与动作空间})$$

这里需要专门给 $p(a \mid s)$ 加一条限定，否则很容易被误读：在 imitation learning / offline dataset 中，我们观测到的其实是**行为策略分布** $p_D(a \mid s)$——它未必覆盖给定 state 下所有 *feasible* 的 action，可能只集中在 successful / slightly-suboptimal 那一小片区域，而 catastrophic 与 recovery action 都欠采样。所以严格说这一维度不是"action coverage"，而是 **behavior / intervention coverage**：它衡量的是我们在数据里见过多少种"实际被执行过的行为/干预"，而不是"物理上可执行的所有动作"。对 imitation learning 而言窄一点也许够用，但对 world model、offline RL、recovery policy 来说，behavior coverage 过窄会直接限制模型学到"另一种 action 会导致什么后果"。还要接上前文 $o_t \neq s_t$：真实机器人数据里 $s$ 往往**并不可直接观测**，因此 $p_D(a\mid s)$ 更准确地应理解为"以 latent / inferred behavioral state 为条件的行为分布"，实际估计时只能通过 observation history、proprioception 或 learned representation 去近似它。

$$\text{behavior / intervention coverage: } p_D(a \mid s)\ \text{而非}\ p_{\mathrm{feasible}}(a \mid s)$$

其实这一维度点破的，正是整篇文章一直在绕的那个概念——真正会伤害 performance 的 mismatch，本质上是 **distribution shift / covariate shift**（在 offline RL 里则表现为对欠覆盖 state-action 区域的 extrapolation）：policy 上线后会把自己带进训练时没见过的 state。所以关键从来不是"dataset 有多 diverse"，而是"**evaluation-relevant 的 state-action 区域有没有被 behavior distribution 覆盖到**"——这恰好是 $p_D(a\mid s)$ 想形式化的东西，也是只看 trajectory-level support 看不见的。

把它们对应到各自负责的泛化能力，就是：

```
Interaction Distribution
│
├── Task      → semantic generalization（任务语义泛化）
├── Scene     → visual / environment generalization（视觉与环境泛化）
├── State     → behavioral-state coverage（行为状态覆盖）
├── Behavior  → behavior / intervention coverage（行为与干预覆盖，即 p_D(a|s)）
└── Embodiment→ morphology / action-space transfer（形态与动作空间迁移）
```

而 **failure / recovery** 并不适合和 task、scene、embodiment 并列——它更像是 interaction distribution 内部一个具有特殊 learning value 的 **trajectory subset**：

```
Base Interaction Distribution
   │
   ├── successful trajectories
   ├── failure trajectories
   └── recovery trajectories
```

换句话说，failure 不是一个新的"分布维度"，而是在同一份 interaction distribution 上按后验标签 $m=h(\tau)$ 划出来的一个子集——它的价值在前文 Data Utility / action-conditioned negative outcome information 里已经讨论过，这里只是把它在 taxonomy 里放到正确的位置。

换句话说，"覆盖更多"必须问清楚"在哪个维度上覆盖更多、想换来哪种泛化"。增加 scene 多样性换来的是视觉/环境鲁棒性，增加 task 多样性换来的是语义泛化，扩大 behavior coverage 换来的是"在同一个 state 下见过更多种被执行过的动作"——把它们笼统地塞进一个 "diversity" 里，会让 scaling 的讨论停留在"多样性很重要"的经验层面。

因此机器人领域的 scaling law 可能不是：

$$Performance = f(N)$$

而更像：

$$D_{\mathrm{effective}} = f(N,\;Coverage,\;Q,\;Relevance)$$

$$Performance = g(D_{\mathrm{effective}},\;Capacity,\;Compute,\;Recipe)$$

这里刻意**不再把 Diversity 列为独立的一项**：样本之间的差异本身并不自动产生价值，只有当这种差异转化为 evaluation-relevant support 的扩张或密度的改善时，它才通过 $Coverage$ 生效；否则再多 diversity 也只是"多"，而不是"覆盖"。把 Diversity 塞进 $D_{\mathrm{effective}}$ 作为一个与 Coverage 平级的乘子，会诱导读者以为"越多样越好"，反而绕开了真正的问题——多样到**哪里**、多样到**够不够 evaluation 用**。所以下文的分解一律用 Coverage（并区分 support / density 两个侧面）来承担原本挂在 Diversity 上的语义。

这里刻意把 $Capacity$ 从 $D_{\mathrm{effective}}$ 中移出、只保留在 $Performance$ 里：否则 capacity 会同时经由 effective data scale 和 performance function 两条路径影响结果，让分解变得含混。effective data scale 描述的应当是"数据本身有多有效"，而容量、算力、recipe 描述的是"模型能把这些有效数据转化成多少性能"。

这意味着：**机器人领域真正需要 scaling 的，不只是 data volume，而是 effective data scale——即 interaction distribution 的有效覆盖。**

如果想更直观，可以把 $D_{\mathrm{effective}}$ 进一步写成一个概念性的乘积分解：

$$D_{\mathrm{effective}} \propto N_{\mathrm{eff}} \cdot \eta_{coverage} \cdot \eta_{quality} \cdot \eta_{relevance}$$

这里刻意用 $\propto$ 而不是 $=$：等号版本会暗示"每条数据最多只贡献 1 单位 information"，但现实并不如此——一条很长、很丰富的 trajectory 携带的信息可能远远超过一条短的。若真要严谨，用 information-theoretic 的写法 $I(\tau;\theta)$ 更自然；本文选择不走到那一步，只是为了保持 conceptual decomposition 的直观性。这些 $\eta$ 是**启发式的有效系数**，用来表达"有效样本量受到多个效率因子共同调制"这一直觉，而不是一个可以直接测量的公式。它把"100 万条 trajectory"这个问题，转换成"这 100 万条里到底有多少是新的、相关的、有效的 interaction information"——这其实更贴近全文真正想表达的东西。

值得注意的是，这里的 $N_{\mathrm{eff}}$ 本身也**不是 raw trajectory count**，而是一个经过相关性折算之后的 effective sample count，且

$$N_{\mathrm{eff}} \leq N$$

原因很具体：机器人数据有一个区别于静态 iid dataset 的特殊问题——**trajectory 内部存在强时间相关性，trajectory 之间又共享大量因素。** 一条"抓杯子"的 trajectory 有 200 个 timestep，并不等于 200 个独立样本；而 1000 条 trajectory 如果都来自同一个 operator、同一个厨房、同一个杯子、同一个 reset 分布、同一套策略，它们的 effective sample size 也可能远远小于 1000。随着重复采样增加，$N_{\mathrm{eff}}$ 的饱和速度会明显快于 raw count $N$——这恰恰解释了为什么"raw trajectory 数量"这个指标会越来越不可靠。一句话：

> **10 万个高度相关的 timestep，并不等于 10 万个独立的信息单位。**

而 $N_{\mathrm{eff}}$ 真正想点出的，是一个比"数据多不多"更重要的现象——**机器人数据在多个层次上同时发生有效样本折损：**

```text
100,000 timesteps → 10,000 trajectories → 1,000 unique scene-object
                  → 100 behavioral regions → 10 distinct failure modes
```

raw count 在每一层都会被折一次：时间相关性削弱 timestep-level independence，共享场景与操作者削弱 trajectory-level independence，重复任务与重复 failure mode 又进一步削弱 distribution-level novelty。换句话说，**机器人数据 scaling 里同时存在 sample redundancy 和 distribution redundancy**——这也是为什么"$N_{\mathrm{eff}}$"必须写成单独一个量，而不能直接用 $N$ 顶替。

（统计上确实有把时间相关性折算成有效样本量的经典直觉，形如 $N_{\mathrm{eff}} \approx N / (1 + 2\sum_k \rho_k)$；但本文刻意不在正文展开它，以免把讨论拖进"时间序列 ESS"的技术细节里。）

需要强调的是，**effective data scale 与 model capacity 并非独立**，但这种耦合应当体现在 $g(\cdot)$ 内部，而不是塞进 $D_{\mathrm{effective}}$：足够宽的 distribution coverage 只有在模型具有足够 capacity 时才能被充分利用。当模型容量较小时，盲目扩大覆盖范围可能收益有限甚至为负；而当容量足够时，同样宽覆盖的数据才能转化为更强的泛化能力（这里所谓"多样化数据"，也是就"覆盖更宽的数据"而言的——diversity 只有转化成 coverage 才起作用）。因此 $Performance$ 是由 $D_{\mathrm{effective}}$、$Capacity$、$Compute$ 和 $Recipe$ 共同决定的，而非任何单一变量的函数。

LLM 可以粗略问"我有多少 token？"；机器人更应该问"我覆盖了多少种任务、状态、环境、动作、失败模式和 embodiment？"

```
Robot Data Scaling ≠ More Trajectories

Effective Data Scale = f(Volume, Distribution Coverage, Quality)
```

这是一个值得验证的假设：**在 interaction distribution（而非纯 trajectory 数量）上 scaling，可能是机器人领域更有效的 scaling 方向。** 目前机器人学习还不存在像 LLM 那样公认的单一 scaling law，但已有针对数据规模的实证研究值得参考——例如关于模仿学习数据 scaling 的工作（Lin et al., 2024，*Data Scaling Laws in Imitation Learning for Robotic Manipulation*，arXiv:2410.18647）发现，策略泛化性能随**环境与物体（environments and objects）的数量**大致呈幂律关系，且环境/物体的多样性比单纯增加轨迹条数更关键——当每个环境/物体的演示数超过某个阈值后，继续堆演示带来的收益会迅速饱和。

为了让这个假设的定位更清晰，可以把文章的逻辑分层如下：

> **已知：** 数据量、数据质量、任务多样性都会影响机器人学习性能。
>
> **未知：** 在固定 compute 与 model capacity 下，哪种 distribution expansion 最有效？
>
> **本文假设：** effective interaction-distribution coverage 比 raw trajectory count 更能解释数据 scaling。

### Support scaling 与 Density scaling

如果只是笼统地说"重复数据收益递减、多样数据收益更高"，其实很容易被反例击穿。考虑一个高精度 manipulation 任务，比如把一个非常小的 connector 精确插入——此时大量高度相似的 trajectory 可能非常有价值，因为模型要学的不是 coverage，而是 precision、control stability、contact dynamics、sub-millimeter correction、action noise tolerance。在这种任务下，$10000$ 条高度相似但高质量的 trajectory，可能比 $1000$ 条非常 diverse 的 trajectory 更有用。

所以更准确的框架是把新数据的边际价值拆成两部分：

$$\Delta U(D) = \Delta U_{\text{support}} + \Delta U_{\text{density}}$$

对应机器人数据 scaling 的两种基本模式：

```
Support scaling（扩大分布支撑集）
  new task
  new object
  new scene
  new failure mode
  new embodiment（仅当它扩大了 evaluation-relevant 区域时才算）
  → 见到以前没见过的东西

Density scaling（提升已覆盖区域的采样密度）
  more trajectories
  more repetitions
  more demonstrations
  → 不只是"估计同一个分布"，还包括 variance reduction、robustness、optimization stability，以及 tiny contact / force 变化、actuator noise、timing、micro-corrections 与 failure boundary 的学习
```

**Support scaling** 回答的是"我是否见到了分布中新的区域"；**density scaling** 回答的是"我在已知区域里是否采样得足够充分"。需要强调，density scaling 并不只是"把已知行为的分布估得更准"这一件事：像把同一个 connector 反复插拔一万次，模型学到的其实是大量微小的 contact variation、force response、actuator noise、timing 与 micro-correction——这里面同时有 estimation、variance reduction、robustness 和 optimization stability 的成分。所以更稳妥的叫法是 $\Delta U_{\text{density}}$，而不是把它窄化成 $\Delta U_{\text{estimation}}$。二者都有价值，只是服务于不同的泛化目标——高精度、接触丰富的任务往往更需要 density scaling，而开放式、多场景的任务更需要 support scaling。

但这里必须给 support scaling 加一条前文 $p_{\mathrm{eval}}$ 的限定：**support expansion 本身并不是价值，只有落在 evaluation-relevant 区域内、且新增数据具有足够质量与可学习性的 support expansion，才有可能带来正的 marginal utility。** 换句话说，与 evaluation-relevant support 有交集只是**必要条件而非充分条件**——一条落在该区域内、但极度 noisy 的 trajectory，utility 完全可能接近 0 甚至为负。所以与其写成一个充要条件，不如把它写成一个受多因子共同调制的作用关系：

$$\Delta U_{\text{support}} = f\Big(\underbrace{\Delta \operatorname{Supp}_{\mathrm{eval}}}_{\text{是否扩大相关支撑}},\ \underbrace{Q}_{\text{质量}},\ \underbrace{R}_{\text{相关性}},\ \underbrace{\text{Learnability}}_{\text{可学习性}}\Big)$$

判断链应当是：

```text
新增数据
   ↓
是否扩大 training support？
   ↓
是否扩大 evaluation-relevant support？
   ↓
是否改善 performance？
```

在这条链上，**new embodiment 并不天然属于 positive support scaling**——只有当它的 morphology、action semantics、control frequency 恰好扩展了 $p_{\mathrm{eval}}$ 的相关区域（例如 evaluation 需要跨本体泛化），它才真正有价值；反之，一个和已有本体高度相似的新 embodiment，只是把 support 扩大到了 evaluation 不关心的方向。同理，"新任务"未必是好事：新任务如果完全落在 $p_{\mathrm{eval}}$ 之外（例如目标是厨房抓取，却新增了大量叠衣服数据），support 扩大了，utility 却几乎为零。

这也把前面的 utility 定义再次显式化：

$$\Delta U(D) = \Delta U\big(D \mid \mathcal{L},\ p_{\mathrm{eval}}\big)$$

Support vs density 的取舍，本质上是"当前 $p_{\mathrm{train}}$ 相对于 $p_{\mathrm{eval}}$ 是覆盖不足，还是密度不足"的判断。

这恰恰引出一个关键判断：**什么时候应该做 support scaling、什么时候应该做 density scaling，本身就是 training recipe 的核心问题**（呼应前文 $p_{\mathrm{train}}(\tau) = T_R[p_{\mathrm{raw}}(\tau)]$——recipe 决定了 raw 数据里哪些区域被放大、哪些被压缩）。

### 可验证预测

如果这个假设成立，那么在固定训练 compute 和模型规模下，可以做出以下可验证预测：

- 新增数据的 marginal value 取决于它是扩大了 evaluation-relevant support，还是在已覆盖的 support 内改善了 density——单纯"重复 vs 多样"不足以预测收益，必须结合任务对 precision 与 coverage 的相对需求；
- 增加能够扩大 $p_{\mathrm{eval}}$ 相关 support 的新 task / scene / embodiment，预期比简单重复已有 trajectory 具有更高的 marginal value（关键词是 *expand the evaluation-relevant support*，而不是"新 = 好"——如果新 embodiment 的 morphology、action semantics 与已有的高度相似，或者落在 $p_{\mathrm{eval}}$ 之外，其边际价值可能很低）；
- 针对 failure mode 的 targeted data 应该比随机增加数据更有效，前提是这些 failure 落在 $p_{\mathrm{eval}}$ 的相关区域内；
- data mixture 和 sampling recipe 的改变应该产生可重复的性能差异，且这种差异中可归因于 Path 1（distribution transformation）和 Path 2（optimization dynamics）的部分原则上可以分开消融；
- 任何关于 marginal value 的定量结论都必须绑定一个明确声明的 $p_{\mathrm{eval}}$——脱离 evaluation distribution 谈"数据是否有用"是不可证伪的；
- 在同等采集成本下，按估计 $MV$ 引导的 targeted data collection 应当优于随机 addition，且 $MV$ estimator 越准（uncertainty calibration 越好、failure statistics 越充分），优势越大。

这些预测原则上可以通过实验验证，而不是停留在"数据重要"的经验判断层面。

## 这意味着什么？

如果把前面几篇文章的线索串起来：

- [世界模型系列](/zh/articles/2026-09-01-world-model-h2-review/)建立了"预测接口"的概念
- [VLA 系列](/zh/articles/2026-09-03-vla-deep-dive/)分析了"语义 + 动作"的接口设计
- [RSSM 演进](/zh/articles/2026-09-04-rssm-beyond/)讨论了不同 latent dynamics 的数据需求
- [行业地图](/zh/articles/2026-09-06-embodied-ai-landscape/)指出数据正在成为关键差异化因素

如果要把这篇文章的核心收敛成"五根柱子"，它们是：

$$\boxed{Interaction\ Distribution:\ p(\tau \mid task,\ scene,\ embodiment)}$$

$$\boxed{Training\ vs.\ Evaluation:\ p_{\mathrm{train}}(\tau)\ \leftrightarrow\ p_{\mathrm{eval}}(\tau)}$$

$$\boxed{Support\ vs.\ Density\ Scaling}$$

$$\boxed{Data\ Utility:\ U(D \mid \mathcal{L},\ p_{\mathrm{eval}})}$$

$$\boxed{Recipe:\ p_{\mathrm{raw}}(\tau) \xrightarrow{T_R} p_{\mathrm{train}}(\tau)\ \text{+ Optimization Dynamics}}$$

这五者合起来想说的其实是一句话：**机器人 scaling 的基本单位，可能不是 trajectory，而是 interaction distribution 相对于目标 evaluation distribution 的有效覆盖。** 从"机器人数据很复杂"升级到"什么才算有效的机器人数据 scaling"，正是这篇文章试图迈出的那一步。

### Marginal Data Value：把整套框架收束成一个可操作的概念

前面所有的概念——interaction distribution、$p_{\mathrm{eval}}$、support/density、utility、recipe——最终都在回答同一个问题：**下一份数据值不值得采？** 这个问题值得一个正式的名字。但要先补一个 baseline：一份新增数据 $D'$ 的价值，永远是在"当前已经有什么数据 $D$"的前提下才有意义的，所以 $\Delta Performance$ 应当显式写成相对于 $D$ 的增量：

$$MV(D';\,D) \;=\; \frac{Performance(D \cup D') - Performance(D)}{Cost(D')}$$

对单条 trajectory 也一样：

$$MV(\tau;\,D) \;=\; \frac{Performance(D \cup \{\tau\}) - Performance(D)}{Cost(\tau)}$$

这个 "$D$" 上标看着只是记号，其实它把全文最核心的 distribution argument 编码进了定义里：**一份数据的价值，依赖于你已经拥有的数据。**

于是全文的核心论点可以收束成一句话：

> **Robot data scaling 的核心问题不是如何最大化 data volume，而是如何最大化 marginal data value。**

这句话比"effective interaction-distribution coverage"更容易被记住，也更贴近工程实践——因为 volume 是一个可以盲冲的量，而 $MV$ 强迫你回答"这一份数据相对于当前 $p_{\mathrm{train}}$ 和 $p_{\mathrm{eval}}$ 究竟补了什么、代价多少"。

而从 $MV(D';D)$ 这个带 baseline 的写法里，还能直接读出全文可能最核心的一个 insight：

$$MV(D';\,D_t) \;\neq\; MV(D';\,D_{t+1})$$

**Data value is state-dependent。** 同一条 trajectory，在数据稀缺的早期可能价值很高，等到 distribution 相关区域已经被填满之后，再采一份几乎同样的数据就可能完全没有价值。这也正是"数据集的质量不能被永久定义"的根本原因——所谓好数据，从来不是绝对的"好数据"，而是"**在当前 training state 与 evaluation gap 下具有高 marginal utility 的数据**"。

而既然 $MV$ 是 state-dependent 的，最优采集策略自然也不是一个 static 规则，而是一个随数据、模型与评估缺口不断变化的**策略**：

$$D'_t = \pi_{\mathrm{data}}(D_t,\ p_{\mathrm{eval}},\ \theta_t)$$

这才是 data flywheel 更深一层的含义——它不只是"收数据 → 训练 → 再收数据"，而是**在学习一个不断变化的数据采集策略**。于是可以留下全文最该被记住的一句 slogan：**机器人数据 scaling 不是一个静态 dataset construction 问题，而是一个 sequential data allocation 问题。** 到这里，Data Utility → Marginal Data Value → Data Flywheel 这三块才算真正闭合。

### 从 scaling 假设到 data flywheel

这篇想说的是：**数据和 training recipe 可能正在成为具身智能中最被低估的竞争优势。**

模型架构可以通过论文和开源代码传播；仿真平台正在被少数几个玩家标准化；但**高质量的机器人交互数据、有效的数据 curation 流程、和经过反复调试的 training recipe——这些很难通过一篇论文完整传递。**

不过需要更谨慎地区分"优势"与"壁垒"。单看某一项，它未必构成真正的护城河：数据可以被采购，teleoperation 基础设施可以被复制，training recipe 可能被逆向工程，foundation model 能力可以迁移，而 synthetic data 甚至可能反过来降低数据本身的壁垒。因此，把"拥有更多数据"直接等同于"拥有壁垒"并不严谨。

真正更难复制的，可能是把整条链路闭合起来形成的 **data flywheel**：

$$Data\ Collection \rightarrow Curation \rightarrow Evaluation \rightarrow Training \rightarrow Deployment$$

$$Deployment \rightarrow Failure \rightarrow Data \rightarrow Training \rightarrow Better\ Policy \rightarrow Deployment$$

也就是说，部署产生真实 failure，failure 回流为新的 targeted data，data 经 curation 后驱动更好的 policy，再进入下一轮部署。这种闭环一旦转起来，竞争者很难仅靠复制某个孤立环节来追赶——**壁垒来自飞轮的转动，而不是某一堆静态数据。**

但如果只停在这里，flywheel 还只是一个"工程战略"，和前面的 scaling theory 是脱节的。有了 $MV$ 之后，我们可以把它写成一个明确的定向采集规则：

$$D_{t+1} = D_t + D_{\text{targeted}}$$

$$D_{\text{targeted}} = \operatorname*{argmax}_{D'}\ MV(D';\,D_t) \;=\; \operatorname*{argmax}_{D'}\ \frac{Performance(D_t \cup D') - Performance(D_t)}{Cost(D')}$$

需要澄清一点：这里的 $\Delta Performance$ **并不假设是一个可以直接读到的 oracle**。工程上它通常通过 evaluation on a proxy distribution、failure statistics、model uncertainty estimation、offline RL 的 counterfactual proxy metric 等手段来估计。于是更准确地说，**真实系统优化的并不是 $MV$，而是它的一个估计量 $\widehat{MV} = MV + \epsilon$**——$\epsilon$ 来自有限评估集、带噪 failure label、uncertainty 估计、simulator bias、offline proxy bias 和 training variance。这带来一个很自然的升级：**targeted data collection 本质上是一个 active learning / decision-making under uncertainty 问题**——先估 $\widehat{MV}$、据此采集、拿到新数据后再得到更好的估计。把这一点显式写出来，公式才从"漂亮 slogan"变成一个真正的研究方向，而不是一个先知式的 argmax。

飞轮转动的意义，不在于 $N$ 变大，而在于每一轮都优先补上 $p(\tau \mid task, scene, embodiment)$ 中相对于 $p_{\mathrm{eval}}$ utility-per-cost 最高的那块缺口。换句话说，**最强的数据飞轮不是"不断收集数据"，而是"不断发现当前 distribution 相对于 evaluation 的缺口，并定向补数据"。**

### 全文的 capstone 流程

到这里其实可以把全文真正的主题说清楚：它表面上在讲"data scaling"，但本质上讲的是**有限数据预算下的 evaluation-aware distribution allocation**——不是被动地把 $p_{\mathrm{train}}$ "对齐"到某个固定的 $p_{\mathrm{eval}}$，而是主动地把有限的采集预算，按 $p_{\mathrm{eval}}$ 暴露出的缺口，一轮一轮地分配到 $p_{\mathrm{train}}$ 的 support 与 density 上，把每一单位预算花到 marginal data value 最高的地方。如果把整篇文章的分析框架压成一张图，它是这样闭合的——请注意 $p_{\mathrm{eval}}$ 位于顶端，是**整个系统的目标坐标系**：

```text
                              p_eval
                                ▲
                                │  gap
                                │
   Raw Interaction Data         │
          │                     │
          ▼                     │
 p_raw(τ | task, scene, embodiment)
          │                     │
          │  Training Recipe    │
          │  (Path 1: dist. transform; Path 2: optimization dynamics)
          ▼                     │
 p_train(τ | task, scene, embodiment)
          │                     │
   ┌──────┴──────┐              │
   ▼             ▼              │
 Support      Density           │
 Coverage     Estimation        │
   │             │              │
   └──────┬──────┘              │
          ▼                     │
   Data Utility / MV            │
  U(D | L, p_eval)              │
          │                     │
          ▼                     │
  Effective Data Scale          │
          │                     │
          ▼                     │
  Performance / Generalization  │
          │                     │
          ▼                     │
      Evaluation ───────────────┘
          │
          ▼
  Failure / Gap Analysis
          │
          ▼
  Targeted Data Collection
          │
          └──────────────→  p_raw' / p_train^new
```

写成公式，就是这一版最终想留下的闭环：

$$\boxed{p_{\mathrm{raw}} \xrightarrow{\ Recipe\ } p_{\mathrm{train}} \xrightarrow{\ Coverage\ /\ Density\ } U(D \mid \mathcal{L},\ p_{\mathrm{eval}}) \longrightarrow Performance}$$

$$\boxed{Performance \longrightarrow Evaluation\ Gap \longrightarrow Targeted\ Data \longrightarrow p_{\mathrm{train}}^{\,\mathrm{new}}}$$

到这一步，**data flywheel 就不再只是一个产业判断，而是前面 scaling hypothesis 的自然推论**：既然 performance 由 $p_{\mathrm{train}}$ 相对 $p_{\mathrm{eval}}$ 的覆盖 + 密度决定，那么最优的下一份数据，当然应当来自 evaluation 暴露出来的 gap。

因此这里的核心问题也不是"谁有更多数据"，而是"谁能让 $p_{\mathrm{train}}$ 沿着 $p_{\mathrm{eval}}$ 的方向持续、且有方向地扩张"，以及"谁能让 $MV$ 在每一轮部署后都得到更好的估计"。

## 参考文献

正文涉及的主要工作如下（均可通过 arXiv ID 检索）：

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
- Scaling Laws for Neural Language Models — Kaplan et al., 2020, arXiv:2001.08361
- Training Compute-Optimal Large Language Models (Chinchilla) — Hoffmann et al., 2022, arXiv:2203.15556
- Data Scaling Laws in Imitation Learning for Robotic Manipulation — Lin et al., 2024, arXiv:2410.18647

上述工作偏重模型与 scaling 框架。与本文"data / distribution 才是关键"这一论点更直接相关的，是以下几类聚焦数据集本身（采集规模、多样性、质量筛选、仿真—真实混合）的实证研究：

- DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset — Khazatsky et al., 2024, arXiv:2403.12945（大规模、多场景真机操作数据集；它直接证明的是"数据规模与环境/任务多样性"，而非"diversity → scaling 收益"这一因果命题，后者仍是本文的假设）
- SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning — Zhang et al., 2025, arXiv:2505.22626（自监督、可组合的数据清洗/质量筛选方法）
- Consistency Matters: Defining Demonstration Data Quality Metrics in Robot Learning from Demonstration — Sakr et al., 2024, arXiv:2412.14309（用一致性等质量指标衡量 demonstration，而非默认"人类演示 = 高质量"）
- Efficient Data Collection for Robotic Manipulation via Compositional Generalization — Gao et al., 2024, arXiv:2403.05110（通过对场景元素的组合式泛化降低数据采集成本）
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., 2025, arXiv:2503.24361（仿真与真实数据混合训练的系统性 recipe 研究）

需要说明的是，机器人学习目前尚不存在像 LLM 那样公认的单一 scaling law；本文关于 effective data scale 的框架是一个 conceptual decomposition 与可检验假设，而非既成结论。上述数据侧工作提供的是分散的实证支持，尚不足以构成对该假设的完整定量验证。

---

*这篇是具身智能系列的延伸——从"谁在做什么"转向"什么在驱动性能"。下一篇可能会讨论 sim-to-real 的方法论细节。*
