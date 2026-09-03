---
title: '具身智能的数据问题：当基础范式逐渐稳定，什么在决定性能？'
slug: "2026-09-08-data-and-training-recipes"
date: 2026-09-08
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "机器人数据", "训练 Recipe", "遥操作", "合成数据", "Sim-to-Real", "数据 Curation", "VLA", "世界模型", "Scaling Law"]
description: "随着 VLA、世界模型等基础范式逐渐形成相对稳定的技术路线，数据分布、数据质量和 training recipe 正越来越成为决定机器人性能的重要变量。但机器人数据不是'越多越好'——机器人领域真正需要 scaling 的不只是 trajectory 数量，而是 interaction distribution。"
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

也就是说，**随着 VLA、world model 等基础范式逐渐形成相对稳定的技术路线，单纯依靠模型架构差异形成性能优势正在变得更加困难；真正决定性能的，越来越是模型看到的 interaction distribution、数据的 quality，以及把数据转换成参数的 training recipe。** 但"数据更重要"不等于"数据越多越好"——机器人领域真正需要 scaling 的，不只是 trajectory 数量，而是 interaction distribution。

这里先给出 interaction distribution 的定义，它会贯穿全文：**本文所说的 interaction distribution，是训练数据中由 task、scene、embodiment 等条件共同决定的 trajectory 分布**，记作

$$p(\tau \mid task,\ scene,\ embodiment)$$

其中 trajectory $\tau=(o_{0:T},a_{0:T-1})$ 本身已经包含 observation、action 和 temporal dynamics，以及可能的 success/failure 信息；如果想写得更显式，也可以表示为 $p(o_{0:T},a_{0:T-1}\mid task,scene,embodiment)$。之所以用条件分布而不是把 task、state、action 都塞进一个联合分布，是因为 state 和 action 已经在 $\tau$ 里，而 failure mode 往往是对 trajectory 做后验分析得到的标签 $m=h(\tau)$，并不是采集时就存在的原始随机变量。这个定义也比单纯讲 diversity 更强，因为 diversity ≠ distribution coverage——一个 dataset 可以有很多 object，但都来自同一种 task distribution。

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

目前具身智能的数据来源大致可以分成四类。

### 遥操作数据（Teleoperation Data）

最直接的来源是人操控机器人完成任务，记录观测-动作轨迹对。

**优势：** 相比纯自主 exploration，更容易获得任务相关、成功率较高、且具有明确行为意图的 trajectory；天然包含人类的操作策略和常识。

但需要注意，teleop 数据并不自动等于高质量数据。它同样可能包含 hesitation、correction、多余动作、inconsistent behavior、operator bias、失败尝试与 recovery，以及不同技能水平操作者的差异——这也正是后文 curation 要处理的问题：human-generated ≠ high-quality。

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

NVIDIA Isaac Sim、MuJoCo（Todorov et al., IROS 2012；现已由 DeepMind 开源）等仿真平台正在被广泛用于生成训练数据；GPU 并行的物理仿真（如 Isaac Gym，Makoviychuk et al., 2021，arXiv:2108.10470）进一步降低了大规模采集成本。但仿真数据通常需要配合 domain randomization（Tobin et al., IROS 2017，arXiv:1703.06907）、system identification 或 real-world fine-tuning 来弥合 gap。

### 合成数据：世界模型作为经验生成器

一个越来越重要的方向是：**用训练好的世界模型来扩大 agent 的经验。**

但这里需要区分两种不同的机制：

**Model-based RL（如 Dreamer）：** 世界模型作为 **latent experience generator**，在隐空间中产生 imagined trajectories。actor/critic 在 latent imagination 中训练：$z_t \rightarrow a_t \rightarrow z_{t+1}$，并不需要生成 photorealistic RGB frame（Dreamer，Hafner et al., 2019，arXiv:1912.01603；DreamerV3，Hafner et al., 2023，arXiv:2301.04104）。

**Generative world model（如视频生成式世界模型、NVIDIA Cosmos）：** 进一步尝试生成接近真实观测的合成数据（synthetic observations / videos / trajectories），从而作为下游训练的数据来源（Cosmos World Foundation Model Platform for Physical AI，NVIDIA，2025，arXiv:2501.03575）。

两者都是"用模型扩大经验"，但数据形态完全不同。前者主要服务于 latent prediction、imagination 和 model-based control；后者是更接近传统意义上的"合成数据生成"。

这里的"世界模型"包含两个相关但不同的概念：用于 latent prediction / imagination / control 的 dynamics model，以及用于生成或预测视觉世界的 generative world model。

但这里有一个必须纳入全文 distribution 框架的关键点：**世界模型生成的数据并不是"免费的真实数据扩张"。** 生成轨迹的分布 $\hat p(\tau)$ 一般不等于真实分布 $p(\tau)$，而且 model error 会随着 rollout horizon 不断累积（compounding error）。可以把这条链路写成：

$$D_{\mathrm{real}} \rightarrow M \rightarrow \hat D_{\mathrm{synthetic}}$$

其中合成数据 $\hat D$ 继承了模型 $M$ 的 bias。因此 synthetic trajectory 的有效性受到 model bias、long-horizon compounding error，以及生成分布与真实 interaction distribution 之间 mismatch 的共同限制——它本质上仍然是一个 distribution 问题，而不只是"数据变多了"。

## 不同范式的数据接口

这是容易被忽略但非常重要的一个维度：**不同的技术路线需要的不是同一种数据。**

### VLA 的数据接口

最基本的 VLA 训练样本可以抽象为 $(o_t, l, a_{t:t+k})$，其中 $l$ 是语言指令，$a_{t:t+k}$ 是 action chunk。但需要强调：**action chunk 只是一种常见的训练/推理接口，而不是 VLA 的定义。** VLA 的核心其实是 $(V, L) \rightarrow A$ 的映射，而 $a_{t:t+k}$ 这种 chunk 形式属于具体的 policy parameterization——一个不显式预测 chunk 的系统，仍然可以是 VLA。

但实际系统还可能包含：proprioception、历史观测窗口、task metadata、embodiment information。动作输出也不只是简单的 $(o,l) \rightarrow a$——现代 VLA 可能使用 action chunk、diffusion / flow action head、discrete action token 或 continuous action，以及 heterogeneous action representation。将大规模视觉语言知识迁移到机器人控制的代表性工作包括 RT-2（Brohan et al., CoRL 2023，arXiv:2307.15818）与 π₀（Black et al., Physical Intelligence, 2024，arXiv:2410.24164）。

这意味着 VLA 对数据的核心需求是：**高质量的 observation-action 配对，覆盖足够多样的任务和物体，同时需要适配不同 embodiment 的动作表示。**

这里有一个更深层的问题：**action representation 本身就是数据接口设计的一部分。** $a_{t:t+k}$ 不只是"动作"——它可能是 joint position、joint velocity、end-effector delta pose、absolute pose、gripper command、discretized tokens、continuous flow、甚至 latent action。因此，cross-embodiment 的核心问题并不只是"把不同机器人的数据放进同一个 dataset"，而是寻找一个足够通用的 observation/action representation，使不同 embodiment 的经验能够在同一个学习空间中共享。

### 世界模型的数据接口

世界模型的核心接口是：

```
输入：observation history + action history

核心输出：
  future latent state / transition distribution
  如 p(z_{t+1} | z_t, a_t)

可选：
  reconstructed observation (p(o_t | z_t))
  reward
  termination
  task outcome
```

需要注意的是，**世界模型的核心是学习 action-conditioned dynamics；reward prediction 并非 dynamics model 在逻辑上的必需组成部分。** 但这里不宜反过来规定"reward 不属于 world model"。更准确的技术分层是：`dynamics model`（学习状态转移）、`reward model`（预测 reward）、`continuation / termination model`（预测 episode 是否继续）——在一些文献和系统定义里，这几个模块合起来就被称为 **world model**。因此在具体的 model-based RL agent（如 Dreamer）中，reward 和 continuation prediction 往往与 dynamics model 一起构成完整的 world-model module，只是它们并非 action-conditioned dynamics 在逻辑上必须的输出。

对于以 latent dynamics + model-based control 为核心的路线（如 Dreamer 的 RSSM、TD-MPC2，Hansen et al., ICLR 2024，arXiv:2310.16828），数据需要是时间上连贯的、action-annotated 的交互轨迹。

### RL 的数据接口

RL 的数据需求取决于具体范式：

- **On-policy**（如 PPO）：需要当前策略产生的数据，数据"新鲜度"很重要
- **Off-policy**（如 SAC）：可以复用历史 replay data，因此通常具有更高的数据复用能力；但 off-policy ≠ 自动更 sample efficient——最终的数据效率仍取决于 replay distribution、exploration、critic quality、reward structure 和任务本身，replay buffer 的分布和覆盖度会影响策略的泛化与稳定性
- **Offline RL**：完全依赖预收集的 dataset，对数据分布覆盖度要求极高
- **Imitation + RL**：先用 demonstration 预训练，再用 online interaction fine-tune

从数据角度看，不同 RL 范式对 replay buffer 或 dataset 的质量和多样性有非常不同的要求。

### 数据接口不兼容的问题

一个实际中经常遇到的问题是：**不同 embodiment、不同传感器配置、不同动作空间的数据通常不能不经处理地直接用于同一个低层 policy。**

一个在 Franka 机械臂上采集的数据，由于动作空间维度、观测视角、动力学特性的差异，通常需要经过 action retargeting、action normalization、或 embodiment conditioning 才能用于其他机器人。

这就是为什么 cross-embodiment data 是一个重要的研究方向。但需要注意术语的精确性：**multi-task**（同一机器人完成多种任务）、**multi-embodiment**（训练数据来自多种机器人但分别处理）、和 **cross-embodiment**（模型能泛化到未见过的机器人）是三个不同层次的问题。

TD-MPC2 的 multi-task / multi-domain 能力主要通过 task embedding 实现——但 task conditioning ≠ embodiment conditioning。Embodiment 差异涉及 action space、observation space、morphology、dynamics、control frequency 等多个维度，不能简单地用一个 task embedding 来解决。π₀ 系列则展示了大规模、多 embodiment 数据对于跨机器人泛化的重要性（这是一个 empirical observation，而非对具体机制的 causal attribution）——Open X-Embodiment（Open X-Embodiment Collaboration, 2023，arXiv:2310.08864）正是这种跨本体大规模数据集的代表。

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

这里还需要区分 diversity 和 coverage：**Diversity 描述样本之间有多不同，coverage 描述目标任务分布被覆盖了多少。** 例如：1000 个不同杯子的数据 → diversity 很高；但如果全部都是"桌面抓取杯子"这一个任务，task coverage 可能仍然很低。

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

更进一步抽象，**failure data 的价值并不在于"失败"本身，而在于它提供了 negative intervention information（负向干预证据）。** 一条 $(s, a_{\mathrm{bad}}, s')$ 告诉模型"在这个 state 下，这个 action 会产生什么后果"；而如果只有成功 demonstration $(s, a_{\mathrm{good}}, s')$，模型并不一定知道 $a_{\mathrm{bad}}$ 为什么不好。

这里需要精确一点：严格意义上的 counterfactual 是"在相同 $s$ 下如果采取另一个 $a'$ 会发生什么"，而我们观测到的只有 $(s, a_{\mathrm{bad}}, s')$，并没有同时观测到配对的 $(s, a_{\mathrm{good}}, s'')$。因此 failure trajectory 本身更准确的定位是 **counterfactual-relevant information**，而不是严格意义上的 counterfactual data——**只有当它与成功轨迹或模型预测结合起来时，才进一步具备 counterfactual learning value。** 即便如此，这种负向干预信号仍然让 failure trajectory 对 world model 和 offline RL 具有独特价值——这把"失败数据有用"从一句经验描述，提升到了一个更清晰的 learning-theoretic intuition。

### Data Quality ≠ Data Utility

前面反复用到 quality、relevance、success、diversity、coverage 这些词，但它们其实可以被一个更基础的概念统一起来：**Data Utility（数据效用）**。

核心观点是：**数据质量并不是一个 absolute property，而是一个 objective-conditioned utility。** 同一份数据 $D$，对不同训练目标 $\mathcal{L}$ 的效用并不相同：

$$U(D \mid \mathcal{L})$$

$$U_{\mathrm{IL}}(D) \neq U_{\mathrm{WM}}(D) \neq U_{\mathrm{offlineRL}}(D)$$

这恰好解释了前面那个现象：一条 failure trajectory 对 imitation learning 可能是需要过滤的 noise，但对 world model 或 offline RL 却可能是 valuable signal——因为它在不同 objective 下的 utility 不同。所谓"高质量数据"，严格说应该是"对当前 objective 高 utility 的数据"。

把 quality 重新理解成 utility，也顺带解决了 curation 里一个常见的误区：不存在一份" universally 好"的数据集，只存在"对某个 $\mathcal{L}$ 好"的数据集。这也是为什么 data curation 必须和 training objective 一起定义，而不能脱离目标单独谈"数据质量"。

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

这一步很关键，因为它把全文的两个核心概念——**data distribution** 和 **training recipe**——在数学上真正连了起来：performance 依赖的是模型实际经历的 $p_{\mathrm{train}}(\tau)$，而不是仓库里静态的 $p_{\mathrm{raw}}(\tau)$；而 $p_{\mathrm{train}}$ 正是 recipe 对 raw distribution 做重采样、重加权、重排序之后的结果。

相同 dataset $D$，换一个 sampling / weighting / objective / schedule（$R_1 \neq R_2$），就可能得到不同的 $p_{\mathrm{train}}$，进而 $\theta_1 \neq \theta_2$。这把"recipe 是壁垒"的观点从经验判断提升到了一个更清晰的技术框架。

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

这里需要区分两种不同类型的 sim-to-real 误差：**随机噪声**（如传感器噪声、微小物理参数波动）和 **systematic simulation bias**（如摩擦系数长期偏差、actuator delay、contact model 误差、deformable object dynamics 误差、camera latency、calibration error）。随机噪声可以通过 domain randomization 来增强鲁棒性；而 systematic bias 是策略可能系统性学错的东西，需要通过 system identification 来校准仿真器本身。换句话说：**domain randomization 解决的是 robustness，system identification 解决的是 simulator bias。**

## 机器人数据 Scaling：不只是"更多轨迹"

LLM 领域已经建立了相对清晰的 scaling law（更多数据 + 更大模型 + 更多计算 → 可预测的性能提升，如 Kaplan et al., 2020，arXiv:2001.08361；Hoffmann et al., 2022，arXiv:2203.15556）。机器人领域是否也存在类似的 scaling law？

### Data Acquisition ≠ Data Scaling

在谈 scaling 之前，需要先区分两个经常被混为一谈的问题。

**Data acquisition** 回答的是"我怎么获得更多数据？"——teleoperation、simulation、autonomous exploration、synthetic generation 都属于 *acquisition method*。本文前半部分讨论的"数据从哪里来"，本质上都是 acquisition 层面的问题。

**Data scaling** 回答的是"我应该增加什么数据，才能继续提高性能？"——support expansion、density improvement、failure targeting、embodiment expansion 属于 *scaling strategy*。

这两者完全不是一回事：把 acquisition method 做得再强，也不自动回答 scaling 的问题。而本文标题真正问的是"什么在决定性能"，因此后半部分的重心应当从"数据从哪里来"转向"什么数据值得继续增加"——这也是下面这套 scaling 框架想回答的。

首先需要明确：**下面的公式不是严格的 scaling law，而是一个用于描述机器人数据有效规模的 conceptual decomposition。** 机器人数据规模至少可以分解为三个层面：

**Data volume：** $N_{\text{steps}}$（总交互步数）

**Distribution dimensions：** $task, scene, embodiment, \text{behavioral state}, action$（分布维度）

**Data quality：** $Q$（数据质量）

这里需要澄清 "state" 这一维：结合前文 $o_t \neq s_t$ 的讨论，我们说的并不是必须显式标注的 environment state（真实 $s$ 往往不可直接获得），而是 **behavioral-state coverage / state-space coverage**——即模型在训练过程中实际经历到的（往往是 latent 或 inferred 的）行为状态分布。这样它就不会和前面的 partial observability 讨论产生概念冲突。

### Coverage 到底覆盖什么：不同维度负责不同泛化

"增大 distribution coverage"这句话如果不进一步拆解，其实还是太笼统。更准确的说法是：**不同的 distribution dimension 对不同的泛化问题负责，它们并不能混为一谈。**

可以分别写成条件分布的形式：

$$p(task)\quad(\text{任务语义空间})$$

$$p(scene \mid task)\quad(\text{环境条件})$$

$$p(s \mid task, scene)\quad(\text{任务执行中访问到的 behavioral state})$$

$$p(a \mid s)\quad(\text{策略实际采取的动作})$$

$$p(s, a \mid failure)\quad(\text{失败相关区域})$$

把它们对应到各自负责的泛化能力，就是：

```
Interaction Distribution
│
├── Task      → semantic generalization（任务语义泛化）
├── Scene     → visual / environment generalization（视觉与环境泛化）
├── State     → behavioral-state coverage（行为状态覆盖）
├── Action    → controllability / intervention coverage（可控性与干预覆盖）
├── Failure   → recovery / robustness（恢复与鲁棒性）
└── Embodiment→ morphology / action-space transfer（形态与动作空间迁移）
```

换句话说，"覆盖更多"必须问清楚"在哪个维度上覆盖更多、想换来哪种泛化"。增加 scene 多样性换来的是视觉/环境鲁棒性，而增加 task 多样性换来的是语义泛化——把它们笼统地塞进一个 "diversity" 里，会让 scaling 的讨论停留在"多样性很重要"的经验层面。

因此机器人领域的 scaling law 可能不是：

$$Performance = f(N)$$

而更像：

$$D_{\mathrm{effective}} = f(N,\;Coverage,\;Diversity,\;Q,\;Relevance)$$

$$Performance = g(D_{\mathrm{effective}},\;Capacity,\;Compute,\;Recipe)$$

这里刻意把 $Capacity$ 从 $D_{\mathrm{effective}}$ 中移出、只保留在 $Performance$ 里：否则 capacity 会同时经由 effective data scale 和 performance function 两条路径影响结果，让分解变得含混。effective data scale 描述的应当是"数据本身有多有效"，而容量、算力、recipe 描述的是"模型能把这些有效数据转化成多少性能"。

这意味着：**机器人领域真正需要 scaling 的，不只是 data volume，而是 effective data scale——即 interaction distribution 的有效覆盖。**

如果想更直观，可以把 $D_{\mathrm{effective}}$ 进一步写成一个概念性的乘积分解：

$$D_{\mathrm{effective}} = N \cdot \eta_{coverage} \cdot \eta_{quality} \cdot \eta_{relevance}$$

其中这些 $\eta$ 是**概念性的有效系数**（表示"有多少比例的数据真正有效"），而不是声称存在一个可精确计算的公式。它把"100 万条 trajectory"这个问题，转换成"这 100 万条里到底有多少是新的、相关的、有效的 interaction information"——这其实更贴近全文真正想表达的东西。

需要强调的是，**effective data scale 与 model capacity 并非独立**，但这种耦合应当体现在 $g(\cdot)$ 内部，而不是塞进 $D_{\mathrm{effective}}$：数据多样性只有在模型具有足够 capacity 时才能被充分利用。当模型容量较小时，盲目扩大 distribution diversity 可能收益有限甚至为负；而当容量足够时，同样的多样化数据才能转化为更强的泛化能力。因此 $Performance$ 是由 $D_{\mathrm{effective}}$、$Capacity$、$Compute$ 和 $Recipe$ 共同决定的，而非任何单一变量的函数。

LLM 可以粗略问"我有多少 token？"；机器人更应该问"我覆盖了多少种任务、状态、环境、动作、失败模式和 embodiment？"

```
Robot Data Scaling ≠ More Trajectories

Effective Data Scale = f(Volume, Distribution Coverage, Quality)
```

这是一个值得验证的假设：**在 interaction distribution（而非纯 trajectory 数量）上 scaling，可能是机器人领域更有效的 scaling 方向。** 目前机器人学习还不存在像 LLM 那样公认的单一 scaling law，但已有针对数据规模的实证研究值得参考——例如关于模仿学习数据 scaling 的工作（Lin et al., 2024，*Data Scaling Laws in Imitation Learning for Robotic Manipulation*，arXiv:2410.18647）显示，数据的有效性与其环境/任务多样性高度相关，而非单纯取决于轨迹条数。

为了让这个假设的定位更清晰，可以把文章的逻辑分层如下：

> **已知：** 数据量、数据质量、任务多样性都会影响机器人学习性能。
>
> **未知：** 在固定 compute 与 model capacity 下，哪种 distribution expansion 最有效？
>
> **本文假设：** effective interaction-distribution coverage 比 raw trajectory count 更能解释数据 scaling。

### Support scaling 与 Density scaling

如果只是笼统地说"重复数据收益递减、多样数据收益更高"，其实很容易被反例击穿。考虑一个高精度 manipulation 任务，比如把一个非常小的 connector 精确插入——此时大量高度相似的 trajectory 可能非常有价值，因为模型要学的不是 coverage，而是 precision、control stability、contact dynamics、sub-millimeter correction、action noise tolerance。在这种任务下，$10000$ 条高度相似但高质量的 trajectory，可能比 $1000$ 条非常 diverse 的 trajectory 更有用。

所以更准确的框架是把新数据的边际价值拆成两部分：

$$\Delta U(D) = \Delta U_{\text{support}} + \Delta U_{\text{estimation}}$$

对应机器人数据 scaling 的两种基本模式：

```
Support scaling（扩大分布支撑集）
  new task
  new object
  new scene
  new embodiment
  new failure mode
  → 见到以前没见过的东西

Density scaling（提升已覆盖区域的采样密度）
  more trajectories
  more repetitions
  more demonstrations
  → 在已经知道的区域，更好地估计已知行为
```

**Support scaling** 回答的是"我是否见到了分布中新的区域"；**density scaling** 回答的是"我在已知区域里是否采样得足够密、估计得足够准"。二者都有价值，只是服务于不同的泛化目标——高精度、接触丰富的任务往往更需要 density scaling，而开放式、多场景的任务更需要 support scaling。

这恰恰引出一个关键判断：**什么时候应该做 support scaling、什么时候应该做 density scaling，本身就是 training recipe 的核心问题**（呼应前文 $p_{\mathrm{train}}(\tau) = T_R[p_{\mathrm{raw}}(\tau)]$——recipe 决定了 raw 数据里哪些区域被放大、哪些被压缩）。

### 可验证预测

如果这个假设成立，那么在固定训练 compute 和模型规模下，可以做出以下可验证预测：

- 新增数据的 marginal value 取决于它是扩大了相关 support，还是在已覆盖的 support 内改进了 estimation——单纯"重复 vs 多样"不足以预测收益，必须结合任务对 precision 与 coverage 的相对需求；
- 增加能够扩大目标任务分布 support 的新 task / scene / embodiment，预期比简单重复已有 trajectory 具有更高的 marginal value（关键词是 *expand the support*，而不是"新 = 好"——如果新 embodiment 的 morphology、action semantics 与已有的高度相似，其边际价值可能很低）；
- 针对 failure mode 的 targeted data 应该比随机增加数据更有效；
- data mixture 和 sampling recipe 的改变应该产生可重复的性能差异。

这些预测原则上可以通过实验验证，而不是停留在"数据重要"的经验判断层面。

## 这意味着什么？

如果把前面几篇文章的线索串起来：

- [世界模型系列](/zh/articles/2026-09-01-world-model-h2-review/)建立了"预测接口"的概念
- [VLA 系列](/zh/articles/2026-09-03-vla-deep-dive/)分析了"语义 + 动作"的接口设计
- [RSSM 演进](/zh/articles/2026-09-04-rssm-beyond/)讨论了不同 latent dynamics 的数据需求
- [行业地图](/zh/articles/2026-09-06-embodied-ai-landscape/)指出数据正在成为关键差异化因素

如果要把这篇文章的核心收敛成"四根柱子"，它们是：

$$\boxed{Interaction\ Distribution:\ p(\tau \mid task, scene, embodiment)}$$

$$\boxed{Support\ vs.\ Density\ Scaling}$$

$$\boxed{Data\ Utility:\ U(D \mid \mathcal{L})}$$

$$\boxed{Recipe:\ p_{\mathrm{raw}}(\tau) \rightarrow p_{\mathrm{train}}(\tau)}$$

这四者合起来想说的其实是一句话：**机器人 scaling 的基本单位，可能不是 trajectory，而是 interaction distribution 的有效覆盖。** 从"机器人数据很复杂"升级到"什么才算有效的机器人数据 scaling"，正是这篇文章试图迈出的那一步。

这篇想说的是：**数据和 training recipe 可能正在成为具身智能中最被低估的竞争优势。**

模型架构可以通过论文和开源代码传播；仿真平台正在被少数几个玩家标准化；但**高质量的机器人交互数据、有效的数据 curation 流程、和经过反复调试的 training recipe——这些很难通过一篇论文完整传递。**

不过需要更谨慎地区分"优势"与"壁垒"。单看某一项，它未必构成真正的护城河：数据可以被采购，teleoperation 基础设施可以被复制，training recipe 可能被逆向工程，foundation model 能力可以迁移，而 synthetic data 甚至可能反过来降低数据本身的壁垒。因此，把"拥有更多数据"直接等同于"拥有壁垒"并不严谨。

真正更难复制的，可能是把整条链路闭合起来形成的 **data flywheel**：

$$Data\ Collection \rightarrow Curation \rightarrow Evaluation \rightarrow Training \rightarrow Deployment$$

$$Deployment \rightarrow Failure \rightarrow Data \rightarrow Training \rightarrow Better\ Policy \rightarrow Deployment$$

也就是说，部署产生真实 failure，failure 回流为新的 targeted data，data 经 curation 后驱动更好的 policy，再进入下一轮部署。这种闭环一旦转起来，竞争者很难仅靠复制某个孤立环节来追赶——**壁垒来自飞轮的转动，而不是某一堆静态数据。**

但如果只停在这里，flywheel 还只是一个"工程战略"，和前面的 scaling theory 是脱节的。从技术视角还可以再往前一步：真正有价值的并不是"deployment 产生更多数据"，而是"deployment **选择性地产生 distribution 中当前最缺的区域的数据**"。可以写成：

$$D_{t+1} = D_t + D_{\text{targeted}}$$

其中新增部分应当是单位成本收益最高的那块：

$$D_{\text{targeted}} = \operatorname*{argmax}_{D'}\ \frac{\Delta Performance}{Cost}$$

这一步把 **data flywheel** 和 **effective data scale** 真正接上了：飞轮转动的意义，不在于 $N$ 变大，而在于每一轮都优先补上 $p(\tau \mid task, scene, embodiment)$ 中 utility-per-cost 最高的缺口。换句话说，**最强的数据飞轮不是"不断收集数据"，而是"不断发现当前 distribution 的缺口，并定向补数据"。**

因此这里的核心问题也不是"谁有更多数据"，而是"谁覆盖了更广的 interaction distribution $p(\tau \mid task, scene, embodiment)$"，以及谁能让这个分布随着部署持续、且有方向地扩张。

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

- DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset — Khazatsky et al., 2024, arXiv:2403.12945（大规模、多场景真机操作数据集，强调环境与任务分布的多样性）
- SCIZOR: Self-Supervised and Composable Data Curation for Robotic Manipulation — Tian et al., 2025, arXiv:2505.22626（自监督、可组合的数据清洗/质量筛选方法）
- Consistency Matters: Revisiting Imitation Learning with Demonstration Quality Metrics — 2024, arXiv:2412.14309（用一致性等质量指标衡量 demonstration，而非默认"人类演示 = 高质量"）
- Efficient Data Collection for Robot Learning via Compositional Generalization — 2024, arXiv:2403.05110（通过组合式任务泛化降低数据采集成本）
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., 2025, arXiv:2503.24361（仿真与真实数据混合训练的系统性 recipe 研究）

需要说明的是，机器人学习目前尚不存在像 LLM 那样公认的单一 scaling law；本文关于 effective data scale 的框架是一个 conceptual decomposition 与可检验假设，而非既成结论。上述数据侧工作提供的是分散的实证支持，尚不足以构成对该假设的完整定量验证。

---

*这篇是具身智能系列的延伸——从"谁在做什么"转向"什么在驱动性能"。下一篇可能会讨论 sim-to-real 的方法论细节。*
