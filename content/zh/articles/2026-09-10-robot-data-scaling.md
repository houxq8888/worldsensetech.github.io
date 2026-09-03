---
title: '机器人数据 Scaling：从 interaction coverage 到 marginal data value'
slug: "2026-09-10-robot-data-scaling"
date: 2026-09-10
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "机器人数据", "Scaling Law", "数据分布", "Coverage", "Marginal Data Value", "Data Flywheel", "offline RL", "模仿学习"]
description: "机器人领域真正值得 scaling 的，不只是 trajectory 数量，而是 interaction distribution 相对于目标 evaluation distribution 的有效覆盖。本文给出一个从 interaction distribution、p_eval、support/density 分解，到 data utility、marginal data value，再到 data flywheel 与 sequential data allocation 的完整分析框架。"
toc: true
related_articles:
  - 2026-09-08-data-and-training-recipes
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
---

这是"具身智能的数据问题"两篇中的**下篇**。[上篇](/zh/articles/2026-09-08-data-and-training-recipes/)盘点了机器人数据的来源、不同范式的数据接口、"数据是 distribution 而非 dataset"的视角、training recipe 的两条作用路径，以及 sim-to-real 的四类工具。那一篇回答的是"数据从哪里来、以什么形态进入模型"；这一篇要回答的是全文真正的问题：**在有限的采集预算下，机器人数据到底应该怎么 scaling——下一单位预算，应该增加什么数据？**

为了让这一篇能独立阅读，先把上篇建立的几个记号收拢一下（详细的论证见上篇）：

> **Interaction distribution：** 训练数据中由 task、scene、embodiment 等条件共同决定的 trajectory 分布 $p(\tau \mid task,\ scene,\ embodiment)$，更严谨地写成 $p_D(\tau \mid c)$，其中下标 $D$ 提醒我们它隐含依赖具体的采集 policy 与环境。
>
> **Training vs Evaluation：** 数据先经过 recipe 变换成模型真正看到的 $p_{\mathrm{train}}(\tau) = T_R[p_{\mathrm{raw}}(\tau)]$；而 performance 取决于它和 evaluation distribution $p_{\mathrm{eval}}$ 的关系。
>
> **Quality ≠ Utility：** quality 是 trajectory 层面可度量的性质，utility 是这些性质在特定 objective 与特定 $p_{\mathrm{eval}}$ 下折算出的条件贡献，记作 $U(D \mid \mathcal{L},\ p_{\mathrm{eval}})$。
>
> **Recipe 的两条路径：** Path 1 改变分布（$p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$，已折进 $D_{\mathrm{effective}}$），Path 2 改变优化动力学（lr schedule、optimizer、loss weighting、freezing、curriculum 等，无法被 $p_{\mathrm{train}}$ 吸收）。

有了这套记号，下面就可以直接进入 scaling 本身。

## 机器人数据 Scaling：不只是"更多轨迹"

LLM 领域已经积累了较成熟的 scaling-law empirical framework（Kaplan et al., 2020，arXiv:2001.08361；Hoffmann et al., 2022，arXiv:2203.15556），描述在特定 compute-optimal / loss-scaling regime 下数据、参数与算力如何共同决定 loss——但这不是一条统一的"自然定律"，也不直接可以搬到机器人上。机器人领域是否也存在类似的 scaling law？

### Data Acquisition ≠ Data Scaling

在谈 scaling 之前，需要先区分两个经常被混为一谈的问题。

**Data acquisition** 回答"数据从哪里来"（teleoperation、simulation、autonomous exploration、synthetic generation 都是 *acquisition method*，上篇讲的就是这一层）；**Data scaling** 回答"下一单位预算应该增加什么数据"（support expansion、density improvement、failure targeting、embodiment expansion 都是 *scaling strategy*）。一个是生成机制，一个是 allocation problem——把 acquisition 做得再强，也不自动回答 scaling。本篇的重心因此从"数据从哪里来"转向"什么数据值得继续增加"，这也正是下面这套框架要回答的。

首先需要明确：**下面的公式不是严格的 scaling law，而是一个用于描述机器人数据有效规模的 conceptual decomposition。** 机器人数据规模至少可以分解为三个层面：

**Data volume：** $N_{\text{steps}}$（总交互步数）

**Distribution dimensions：** $task, scene, embodiment, \text{behavioral state}, action$（分布维度）

**Data quality：** $Q$（数据质量）

这里需要澄清 "state" 这一维：结合上篇 $o_t \neq s_t$ 的讨论，我们说的并不是必须显式标注的 environment state（真实 $s$ 往往不可直接获得），而是 **behavioral-state coverage / state-space coverage**——即模型在训练过程中实际经历到的（往往是 latent 或 inferred 的）行为状态分布。这样它就不会和上篇的 partial observability 讨论产生概念冲突。

### Distribution ≠ Coverage：引入 Evaluation Distribution

在展开 coverage 之前，必须先补一个此前一直被隐含使用的概念。上篇把 interaction distribution 定义成

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

把这一步点明之后，下面这个 utility 定义也就不是凭空冒出来的记号，而是顺着"coverage 是关系量"这条线自然推出来的结果：这也直接把上篇 utility 的定义收紧了一档——**数据效用不只是 objective-conditioned，还是 evaluation-conditioned**。

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

这里需要专门给 $p(a \mid s)$ 加一条限定，否则很容易被误读：在 imitation learning / offline dataset 中，我们观测到的其实是**行为策略分布** $p_D(a \mid s)$——它未必覆盖给定 state 下所有 *feasible* 的 action，可能只集中在 successful / slightly-suboptimal 那一小片区域，而 catastrophic 与 recovery action 都欠采样。所以严格说这一维度不是"action coverage"，而是 **behavior / intervention coverage**：它衡量的是我们在数据里见过多少种"实际被执行过的行为/干预"，而不是"物理上可执行的所有动作"。对 imitation learning 而言窄一点也许够用，但对 world model、offline RL、recovery policy 来说，behavior coverage 过窄会直接限制模型学到"另一种 action 会导致什么后果"。还要接上上篇 $o_t \neq s_t$：真实机器人数据里 $s$ 往往**并不可直接观测**，因此 $p_D(a\mid s)$ 更准确地应理解为"以 latent / inferred behavioral state 为条件的行为分布"，实际估计时只能通过 observation history、proprioception 或 learned representation 去近似它。

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

换句话说，failure 不是一个新的"分布维度"，而是在同一份 interaction distribution 上按后验标签 $m=h(\tau)$ 划出来的一个子集——它的价值在上篇 Data Utility / action-conditioned negative outcome information 里已经讨论过，这里只是把它在 taxonomy 里放到正确的位置。

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

这恰恰引出一个关键判断：**什么时候应该做 support scaling、什么时候应该做 density scaling，本身就是 training recipe 的核心问题**（呼应上篇 $p_{\mathrm{train}}(\tau) = T_R[p_{\mathrm{raw}}(\tau)]$——recipe 决定了 raw 数据里哪些区域被放大、哪些被压缩）。

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

如果把[前面的行业地图](/zh/articles/2026-09-06-embodied-ai-landscape/)、[世界模型系列](/zh/articles/2026-09-01-world-model-h2-review/)、[VLA 系列](/zh/articles/2026-09-03-vla-deep-dive/) 与 [RSSM 演进](/zh/articles/2026-09-04-rssm-beyond/) 的线索串起来，会发现它们分别建立了"预测接口""语义 + 动作接口""不同 latent dynamics 的数据需求"以及"数据正在成为关键差异化因素"这些概念。上下两篇合起来，真正想收束的是以下几块：

$$\boxed{Interaction\ Distribution:\ p(\tau \mid task,\ scene,\ embodiment)}$$

$$\boxed{Training\ vs.\ Evaluation:\ p_{\mathrm{train}}(\tau)\ \leftrightarrow\ p_{\mathrm{eval}}(\tau)}$$

$$\boxed{Support\ vs.\ Density\ Scaling}$$

$$\boxed{Data\ Utility:\ U(D \mid \mathcal{L},\ p_{\mathrm{eval}})}$$

$$\boxed{Recipe:\ p_{\mathrm{raw}}(\tau) \xrightarrow{T_R} p_{\mathrm{train}}(\tau)\ \text{+ Optimization Dynamics}}$$

这五者合起来想说的其实是一句话：**机器人 scaling 的基本单位，可能不是 trajectory，而是 interaction distribution 相对于目标 evaluation distribution 的有效覆盖。** 从"机器人数据很复杂"升级到"什么才算有效的机器人数据 scaling"，正是这两篇文章试图迈出的那一步。

### Marginal Data Value：把整套框架收束成一个可操作的概念

上下两篇里所有的概念——interaction distribution、$p_{\mathrm{eval}}$、support/density、utility、recipe——最终都在回答同一个问题：**下一份数据值不值得采？** 这个问题值得一个正式的名字。但要先补一个 baseline：一份新增数据 $D'$ 的价值，永远是在"当前已经有什么数据 $D$"的前提下才有意义的，所以 $\Delta Performance$ 应当显式写成相对于 $D$ 的增量：

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

到这里其实可以把全文真正的主题说清楚：它表面上在讲"data scaling"，但本质上讲的是**有限数据预算下的 evaluation-aware distribution allocation**——不是被动地把 $p_{\mathrm{train}}$ "对齐"到某个固定的 $p_{\mathrm{eval}}$，而是主动地把有限的采集预算，按 $p_{\mathrm{eval}}$ 暴露出的缺口，一轮一轮地分配到 $p_{\mathrm{train}}$ 的 support 与 density 上，把每一单位预算花到 marginal data value 最高的地方。如果把整个分析框架压成一张图，它是这样闭合的——请注意 $p_{\mathrm{eval}}$ 位于顶端，是**整个系统的目标坐标系**：

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
 Coverage     Scaling           │
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

- Scaling Laws for Neural Language Models — Kaplan et al., 2020, arXiv:2001.08361
- Training Compute-Optimal Large Language Models (Chinchilla) — Hoffmann et al., 2022, arXiv:2203.15556
- Data Scaling Laws in Imitation Learning for Robotic Manipulation — Lin et al., 2024, arXiv:2410.18647

与本文"data / distribution 才是关键"这一论点直接相关的，还有上篇末尾列出的那批聚焦数据集本身的实证工作（DROID、SCIZOR、Consistency、Compositional、Sim-and-Real Co-Training 等，见 [上篇参考文献](/zh/articles/2026-09-08-data-and-training-recipes/)）。

需要说明的是，机器人学习目前尚不存在像 LLM 那样公认的单一 scaling law；本文关于 effective data scale 的框架是一个 conceptual decomposition 与可检验假设，而非既成结论。上述数据侧工作提供的是分散的实证支持，尚不足以构成对该假设的完整定量验证。

---

*这篇是"具身智能的数据问题"两篇的下篇——上篇讲数据来源、接口与 training recipe，这一篇讲什么才算有效的数据 scaling。下一篇可能会讨论 sim-to-real 的方法论细节。*
