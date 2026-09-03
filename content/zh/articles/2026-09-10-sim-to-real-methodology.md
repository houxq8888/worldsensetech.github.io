---
title: '具身智能 Sim-to-Real 方法论深潜：把"从仿真到真实"当成一次误差预算分配'
slug: "2026-09-10-sim-to-real-methodology"
date: 2026-09-10
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "Sim-to-Real", "Domain Randomization", "System Identification", "可微仿真", "Residual Physics", "世界模型", "Domain Adaptation", "机器人数据"]
description: "sim-to-real 常被讲成某个单一迁移技巧，但它本质是一次带约束的误差预算分配。本文先把 reality gap 重述成一个 policy-conditioned 的多源 mismatch，再给出一个把 mismatch 分解、干预选择、真机预算分配与评估效用串起来的 conceptual 优化框架，逐一深潜 system identification、domain randomization、domain adaptation、real-world fine-tuning 四条 intervention axes 的机制与失效边界，并厘清可微仿真、residual physics、world model、sim-and-real co-training 这几条容易被混为一谈的路线。"
toc: true
related_articles:
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
  - 2026-08-25-dreamer-explained
---

> 这是[数据问题上篇](/zh/articles/2026-09-08-data-and-training-recipes/)与[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)的续篇。上篇我用一张图把 sim-to-real 粗分成四类工具，但那只是 taxonomy。这一篇想真正回答的问题，可以压缩成一句——

> **当你的仿真数据在若干 evaluation-relevant 方向上离真实世界差得远时，这些差距该用 system identification 校准、用 domain randomization 覆盖、用 domain adaptation 对齐，还是干脆花真机预算补上？**

这句话看着像工程直觉，但它其实是一个**带约束的分配问题**：给定一份有限的工程预算，你要在"仿真保真度、训练分布多样性、表示对齐、真机交互"这几条彼此独立的杠杆之间，把预算分配到最能压低最终性能落差的地方。本文想做的，就是把这句直觉从一个漂亮的 metaphor，尽量推成一个可以逐条讨论的 framework。而真实项目里最卡人的，往往也不是"不知道有这些方法"，而是"不知道这个方法在我这一类 gap 上到底管不管用、要花多少真机预算"。

## Reality Gap：不是一个标量，而是一个 policy-conditioned 的 mismatch

Sim-to-real 常被叙述成"训练一个 policy，让它从仿真迁移到真实"。更严谨的起点是**两个分布**：仿真器给出 $p_{\mathrm{sim}}(\tau)$，真实世界给出 $p_{\mathrm{real}}(\tau)$，二者一般不相等：

$$p_{\mathrm{sim}}(\tau) \;\neq\; p_{\mathrm{real}}(\tau)$$

但我们要关心的并不是这个分布差本身，而是它在某个任务上**表现出来的后果**——同一个 policy $\pi_\theta$ 在两边的性能之差：

$$\Delta J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)$$

这里必须先划清一层，否则后文容易滑坡：**distribution mismatch 不等于 performance gap**。$p_{\mathrm{sim}} \neq p_{\mathrm{real}}$ 并不自动意味着 $\Delta J$ 很大——因为不同 policy 对分布差的敏感度完全不同。一个只依赖粗粒度几何的 policy，可能在换了对摩擦系数建模后性能几乎不变；而一个依赖高频力反馈的精细装配 policy，同样的分布差就可能是致命的。

所以更准确的写法是：$\Delta J(\pi)$ 是一个**任务相关、policy 相关的可观测后果（observable consequence）**，而不是 reality gap 的完整定义。它至少同时依赖四样东西：

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ p_{\mathrm{sim}},\ p_{\mathrm{real}},\ \mathcal{E}\big)$$

其中 $\mathcal{E}$ 是 evaluation 的设定。**同一个 simulator，对 position control policy 可能 gap 很小，对 force-sensitive manipulation policy 可能 gap 巨大。** 这就是为什么"我们的仿真器很真"从来不是一个有意义的评价——reality gap 不是仿真器的内在属性，而是 $(\pi, p_{\mathrm{sim}}, p_{\mathrm{real}}, \mathcal{E})$ 这个四元组的属性。认清这一点，是后文所有"分配"讨论的前提：既然 gap 依赖 policy 与 evaluation，那么"补哪一层 gap"自然也只能相对于目标场景来回答。

### gap 到底在哪里：一个五层 mismatch 分解

既然 gap 是多源的，第一步就是把它拆开。工程上好用的粗粒度分解是五层（注意：这五层**性质不同、可参数化程度不同、补救成本更是天差地别**）：

```
Reality mismatch
├── Dynamics / contact        摩擦、接触、可形变体、柔顺结构
├── Observation / estimation  传感器物理、标定、噪声、遮挡、时延、状态估计
├── Actuation / timing        电机动力学、控制频率、执行器延迟、通信抖动
├── Initial-state / env.      reset 分布、场景布局、长尾、初始条件
└── Objective / constraint    reward 定义、安全约束、成功判据
```

这里要专门把**观测与状态估计**单独列成一层，而不是塞进"感知 gap"里。原因很具体：机器人真正执行的是

$$a_t = \pi(o_t), \qquad o_t = h(x_t) + \epsilon$$

而真实世界里的 camera 标定误差、depth bias、遮挡、proprioception drift、力传感器偏置、state estimator 的时延与不同步，**并不是简单的"画面看起来不一样"**——它们让 **policy 实际看到的 state 估计，与 simulator 里假设可用的 state 不一致**。在 manipulation 与 locomotion 里，这一类"状态估计 gap"往往比外观 gap 更伤 performance，因此值得独立成层：

```
Observation / Estimation
 ├─ sensor physics   成像/测距的物理过程
 ├─ calibration      内外参、手眼标定
 ├─ noise            曝光、量化、随机噪声
 ├─ occlusion        遮挡与部分可观测
 ├─ latency          传感与同步时延
 └─ state estimation 从 o 推断 s 的误差
```

还要立刻把**任务与初始状态**这一层再劈成两半，因为它们其实是两个问题：

- **Environment / initial-state shift：** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$——reset 分布、场景布局对不上；这是 simulation 里的 distribution 问题，属于 reality gap。
- **Objective / task shift：** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$——仿真里只要求 grasp success，真实里还要求 collision avoidance；仿真允许较大 penetration，真实里 hardware safety 不允许。

后者严格说**已经不是 reality gap，而是 objective mismatch**：仿真器把物理模拟得再准，如果你的 reward / 约束和真实目标不是一回事，那也不是"迁移失败"，而是"你评测的根本不是同一个任务"。把它和 physical gap 混进一个"sim-to-real 失败"里，会让讨论变成"这到底是 sim-to-real，还是 sim-to-task"。本文后面凡是谈 gap 压缩，都默认 objective 已经对齐；objective mismatch 需要单独靠 reward shaping / 约束建模解决，不在这四类工具的射程内。

## 把"误差预算分配"变成一个可写的框架

上一节把 gap 拆成了来源不同的层。现在可以给开头的直觉一个数学落脚点了。先声明：**下面的分解是 conceptual decomposition，不是严格定理**——各层误差会相互耦合、非线性放大，"加号"只是表达"总落差由若干来源共同贡献"这一直觉，而非声称它们严格可加独立。有了这个保留，我们可以写：

$$\Delta J \;\lesssim\; \underbrace{\Delta_{\mathrm{model}}}_{\text{动力学/接触建模误差}} \;+\; \underbrace{\Delta_{\mathrm{obs}}}_{\text{观测与状态估计}} \;+\; \underbrace{\Delta_{\mathrm{ctrl}}}_{\text{执行与时序}} \;+\; \underbrace{\Delta_{\mathrm{dist}}}_{\text{初始态/场景覆盖}} \;+\; \underbrace{\Delta_{\mathrm{opt}}}_{\text{残余优化与先验偏差}}$$

真正让"误差预算"这个说法成立的，是把**每一种干预（intervention）对应到它主要压低的那一项、以及它要花的预算**上。把工具选择写成一个带约束的优化问题：

$$\min_{m}\quad \mathbb{E}\big[\Delta J(m)\big] \qquad \text{s.t.}\quad C_{\mathrm{sim}}(m) + C_{\mathrm{real}}(m) + C_{\mathrm{eng}}(m) \;\le\; B$$

其中 $m$ 是从工具集里选出的一个（组合）策略，$C_{\mathrm{sim}}$ 是仿真器保真度/算力成本，$C_{\mathrm{real}}$ 是真机采集与磨损成本，$C_{\mathrm{eng}}$ 是工程与维护成本，$B$ 是总预算。这一下，四类工具不再是"谁更强"的互斥选项，而各自对应一个压缩项与一种成本：

| Intervention | 主要压缩项 | 主要成本 |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{sim}}$ + 少量 $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | 样本效率 / $C_{\mathrm{sim}}$ |
| Residual physics | $\Delta_{\mathrm{model}}$（残差部分） | 真机交互 $C_{\mathrm{real}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$ | 未标注真机数据 |
| Real-world fine-tuning | 残余 $\Delta J$（含 $\Delta_{\mathrm{opt}}$） | $C_{\mathrm{real}}$（磨损/安全） |
| World model | 改变 model source | $C_{\mathrm{real}}$ + $C_{\mathrm{sim}}$ |
| Sim-and-real co-training | 改变 $p_{\mathrm{train}}$（$\Delta_{\mathrm{dist}}$ 为主） | 混合数据成本 |

有了这张表，开头那句工程直觉就有了严格版本：**所谓"分配误差预算"，就是在约束 $B$ 下，选择一组干预，使得它们合力把主导项最大的那几层 $\Delta$ 压下去。** 一个摩擦主导的 peg insertion 与一个视觉主导的桌面整理，主导项不同，最优 $m$ 也不同——这正是"为什么别人管用的配方到我这就不管用"的结构性解释。

## 四条 intervention axes（而非四类互斥方法）

有了框架，接下来逐条看工具。但先修正一个分类上的坑：system identification、domain randomization、domain adaptation、real-world fine-tuning **并不是同一抽象层级上的四个并列类别**——SI 是 model calibration，DR 是 training distribution manipulation，DA 是 representation alignment，fine-tuning 是 optimization strategy。把它们并排成"四类方法"，会让人误以为要四选一。更准确的说法是：它们是**四条彼此正交的干预轴**，可以任意组合：

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

正因为正交，"组合"才天然成立——你可以在**同一条轴上换手段、在不同轴上同时发力**，而不必纠结"到底该用哪种方法"。下面按这四条轴展开。

### 一个更准的判别：parameterizable / identifiable / coverable

在展开前，先把最容易误导人的一句直觉换掉。很多人会记成"systematic 误差交给 SI，random 误差交给 DR"——作为口诀没错，但作为技术判断会被挑刺。因为 system identification 真正在做的，是估计

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; D\big(p_{\mathrm{sim}}(\tau \mid \phi),\; p_{\mathrm{real}}(\tau)\big)$$

它解决的其实是**可辨识、可参数化的 model mismatch**，而不是"凡是 systematic 都归它"（actuator gain、latency、friction、mass 本身都可以是随机过程，而非确定性的 systematic bias）。同理，DR 解决的是**能够被一个训练分布表示出来的 uncertainty / model variation**。所以更有用的划分，不是 systematic vs random，而是三连问：

> **这个 mismatch 能不能被一个可信的参数化模型表达（parameterizable）？能不能从有限真机数据里辨识出来（identifiable）？如果不能辨识，能不能通过扩大训练分布去覆盖（coverable）？**

按这三问，工具选择大致是：

| mismatch 的性质 | 更自然的工具 |
| --- | --- |
| 可参数化 + 可辨识 | System Identification |
| 可参数化但难辨识 / uncertainty 大 | Domain Randomization |
| 不可参数化、但有 residual structure | Residual learning |
| observation / appearance mismatch | Domain Adaptation |
| policy 在目标域仍有 systematic residual | Fine-tuning |

### Axis A — Model：system identification、可微仿真与 residual physics

这条轴处理 $\Delta_{\mathrm{model}}$，内部其实有三个**正交**的问题，值得一次分清，因为它们经常被打包成"可微仿真 = 更强的 SI"这种含糊说法：

$$f_{\mathrm{real}}(x,a) \;=\; \underbrace{f_{\mathrm{physics}}(x,a;\phi)}_{\text{可参数化的物理}} \;+\; \underbrace{r_\theta(x,a)}_{\text{残差}} \;+\; \epsilon$$

- **可微仿真回答"怎么优化模型"**——它提供 $\partial f/\partial\phi$ 这个 optimization interface；DiffTaichi（Hu et al., 1910.00935）、Interactive Differentiable Simulation（Heiden et al., ICRA 2021，1905.10706）让参数估计可以梯度化。
- **System identification回答"优化什么参数"**——即上面的 $\phi$。经典 SI 扫参数、拟合轨迹；可微仿真则把 $\phi$ 像权重一样反传更新；实际工作流还常常是 **real → identify → sim → train → real**，所以更准确的名字是 **real-to-sim-to-real**。
- **Residual physics回答"模型没解释掉的那部分由谁来解释"**——不去硬校准 $\phi$，而让网络学一个 $r_\theta$ 补差。它在 contact-rich、软体、浮力腿式这类**参数化模型函数形式本身就不对**的场景里特别有用，因为那种情况下 SI 校的是一个错误假设（残差例子见软体机器人 Michelis et al., 2402.01086；浮力腿式 Chae et al., 2303.09597）。

这里有一个非常关键、也最容易被"可微"二字掩盖的点：**可微性解决 optimization interface，不解决 model class correctness。** 换句话说——如果 simulator 的 contact model 根本没有表达某种真实现象，那么你对这个错误模型求再精确的梯度，也只能得到"错误模型之下的最优参数"。可微仿真让你把 $\phi$ 估得更准，但它不会替你把 $f_{\mathrm{physics}}$ 的函数形式写对；写不对的那部分，只能交给残差，或者干脆放弃"先建一个可信 sim"这个前提（见下一条轴之外的 world model）。这三者正交，把它们混成一个"高级可微仿真"叙事，恰恰会藏掉真正决定成败的 model class 问题。

### Axis B — Data distribution：domain randomization 及其家族

这条轴不追求逼近某个"最准"的 $p_{\mathrm{real}}$，而是让 policy 对一族参数 $\{\phi\}$ 都稳健：训练时随机化物理、视觉、初始状态与 delay，只要 real 落在这族支撑内，policy 就能顶住。Tobin（1703.06907）用纯视觉随机化把 sim 抓取的检测搬到真机；Peng（1710.06537）把随机化推进到 dynamics；OpenAI 的 in-hand manipulation（Akkaya et al., 1808.00177）几乎把 DR 推到极致——**不靠精确校准，靠"随机化范围足够宽"来吸收差异。**

需要修正一句常见但不严谨的表述。我初稿里写过"DR 是隐式 ensemble"，这个说法有启发性但会被挑刺：DR 训练的是**单个**共享 policy $\pi_\theta$，目标大致是

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

而通常意义的 ensemble 是 $\{\pi_1,\dots,\pi_K\}$ 多模型。所以更稳妥的说法是：**可以把 DR 理解成"对一族环境模型做 population-level 优化"，而不是针对单一 simulator 做优化。** 直觉上它有点像 ensemble training（policy 被迫在一族 dynamics 上同时表现良好），但严格说最终得到的是一个共享 policy，而非多个 policy。

"Adaptive / Automatic DR"也不是单一方法，而是一个家族，值得摊开以免读者以为只有一种：curriculum over randomization（随训练进程放宽范围）、adversarial domain randomization（采样最能击垮当前 policy 的参数）、automatic domain randomization（随 policy 表现自适应收缩）、posterior-based sampling（用辨识后验来采样）、performance-driven range adaptation。它们机制不同，但共同点是**避免"一开始就 over-randomize"**。

翻车边界：随机化范围过宽会诱发过度保守 / 平均化行为、样本效率下降；DR 只在它**真的覆盖了真实系统**时管用，real 落在族外就救不了；而面对不可参数化的 mismatch（复杂接触、软体、流体），根本没有对应的"随机化轴"可随——这类得回到 Axis A 的残差，或接受真机数据。

### Axis C — Representation：domain adaptation 与观测翻译

这条轴处理 $\Delta_{\mathrm{obs}}$，既不校准物理、也不随机化，而是在**表示层**把 sim 与 real 对齐：feature-level adapter、image translation（GAN / 扩散）、或 randomized-to-canonical 的翻译网络 RCAN（James et al., CVPR 2019，1812.07252）——它把随机化过的 sim 图"翻译"回一张近似 canonical 的干净图再喂给下游 policy，正好**把 Axis B 的 DR 和这条轴缝起来**，缓解过宽随机化的性能损失。它处理的是"物理其实差不多、但看起来完全不像"的那部分 gap。

边界：翻译网络可能把任务相关的语义一起抹掉（对齐得越好，某些精细信号反而越被平均掉）；无监督 DA 通常需要 real 侧**未标注**数据——但"未标注"不等于"不要钱"，真机采集本身依然贵，这笔账要记进 $C_{\mathrm{real}}$。

### Axis D — Optimization / adaptation：真机微调

这条轴处理的是"前三条轴都补完之后剩下的残余 $\Delta J$"——先在仿真里大规模 pre-training 学结构，再用真机数据接力。但这里必须把两个 regime 的成本结构分开，因为它们常被一句"做 RL 或 imitation 微调"糊在一起：

- **Offline / imitation：** $D_{\mathrm{real}} \to \theta$，主要成本是**数据采集**（一次性，可离线、可复用）。
- **Online RL：** $\pi_\theta \to a \to$ 真实 transition $\to \theta'$，主要成本是**交互 + 安全 + 硬件磨损 + 探索**（每一步都在消耗物理资源）。

这个区分直接决定了"值不值得微调"的判断：对真机学习而言，比较方法不能只看最终 success rate，还要看**达到某个目标性能所需的真机交互预算**。所以一个该进入上面 $C_{\mathrm{real}}$ 的务实指标是

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{或}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

也就是说，"每单位真机磨损换来的性能提升"。同样涨 5 个点，一个要 2 机时、一个要 40 机时，在物理上根本不是同一回事。至于翻车边界：如果 sim 学出的先验在真实里本来就是错的，微调可能被它带偏甚至越调越差（错误先验 / 灾难性遗忘）；而 fine-tune 到底比"从头真机采集"更省，取决于仿真 fidelity、任务，以及微调样本相对单价——目前没有统一经验法则。

## 两条松动"两个给定分布"假设的新路线

上面四条轴共享一个隐含前提：**$p_{\mathrm{sim}}$ 与 $p_{\mathrm{real}}$ 是两个给定的分布**，你要做的是校准、覆盖、对齐或接力。而近年有两条路线，恰恰在松动这个前提本身——它们不是"第五、第六种迁移技巧"，而是对整个问题的 reformulation，所以单独拿出来讲。

### World model：不是 sim-to-real 的一种技巧，而是另一种 model-based route

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)讨论过 world model 与 data utility 的关系。放进 sim-to-real 语境，要先纠正一个常见误读：**world model 并不天然属于 sim-to-real。** 经典 sim-to-real 与 world model 路线的 causal direction 是不同的：

```
经典 sim-to-real：  sim dynamics → train policy → deploy real
World model route： real interaction → learn dynamics → imagine → optimize policy
```

前者假设"先有一个可信 sim，再迁移"；后者干脆**放弃"先构建可信 simulator"这个前提**，直接从真实交互里学一个 latent dynamics $p_\theta(z_{t+1}\mid z_t,a_t)$，把 policy 的想象与规划都放进这个学到的模型里做（Dreamer 1912.01603、TD-MPC2 2310.16828）。所以更准确的定位是：当**人工 simulator 的 model bias 太大**、大到不值得先修 sim 时，world model 是**对 sim-to-real 问题本身的一种改写**，而不是它下面的一个 transfer technique。

DayDreamer（2206.14176）也常被误当成"sim 预训练 → real 微调"的例子，但它的关键贡献恰恰相反——**world model 直接从真实机器人交互中学习，并在 latent imagination 里做 policy improvement**，几乎不依赖人工 sim。它更适合作为"另一条替代 simulator 的路线"的样本，而不是 sim-to-real 迁移的样本。

诚实的边界："用真实数据学 dynamics" **不等于"抛弃 simulator"，更不意味着它天然优于仿真。** 它把"手工建模成本"换成了"真机采集成本 + 模型容量成本"；而在 contact-rich、long-tail、传感器噪声大的场景，学到的 model 常常在自己没见过的分布外区域给出**很自信、也很错的想象**，policy 会顺着错误规划走下去。所以它目前只是"手工 sim"与"直接真机 RL"之间的**又一个 trade-off**，不是终局。

### Sim-and-real co-training：把"迁移"改写成"混采混合"

Maddukuri et al.（2503.24361）提出的 Sim-and-Real Co-Training 是一个务实方向。**论文实际报告的结果**是：在同一批训练里把 sim 与 real 数据集混合采样，并在两个机器人平台、多任务上观测到平均性能提升。这一步没有做 sim→real 的单向迁移，而是用一个 recipe 直接决定两者比例与调度。

但下面这句是**本文的解读，不是论文证明的结论**：我倾向于把这一类方法进一步解释成一个 **data-mixture 问题**——一旦这样重述，DR 与真实数据就不再是替代关系，而是同一份 sampling distribution 上 $T_R[p_{\mathrm{raw}}]$ 的两个来源（呼应上篇"$p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$、recipe 是数据到参数的转换函数"）。这个 data-mixture view 能解释它为什么和上面四条轴正交：co-training 不改 sim 也不改 policy，只改 $p_{\mathrm{train}}$。

值得补一句的是，这条路线的**机理**也在被追问：2026 年已有后续工作（A Mechanistic Analysis of Sim-and-Real Co-Training，arXiv:2604.13645）尝试给出 co-training 为何有效的两个机制——structured representation alignment 与 importance reweighting。也就是说，"混合有效"正在从经验现象变成一个有候选解释的研究对象。

## 评估：你怎么知道自己把 gap 补好了？

一个常见但危险的做法是只在 sim benchmark 上报性能。这样的数字**衡量的是 policy 与你自己 simulator 之间的一致性，而不是与真实世界的一致性**。更可信的评估至少应做到：

- 报告 **zero-shot transfer**（不做任何真机微调）到真实系统的性能，以及 **few-shot / N-shot** 之后的曲线；
- 用一组 **held-out physical systems**（不同标定、不同相机、不同接触面）来测，而不是只有"那台部署机器人"；
- 明确声明 sim 与 real 的**任务、initial-state、evaluation distribution 是否一致**——否则比较根本不公平（这正是前面 objective mismatch 要先对齐的原因）；
- 做**失败归因**：是哪一层 $\Delta$ 主导（model / obs / ctrl / dist / opt）？不同层的补救成本天差地别，归因错了预算就花错了地方；
- 记 **$\eta_{\mathrm{real}}$**：对用了真机数据的路线，报"每单位真机磨损换来的性能"，否则"纯 sim"和"sim + fine-tune"根本没法公平比。

还有一个特别贴合本文论点的指标，值得单独强调。如果 sim 的定位是"真实世界的代理"，那么光看 $J_{\mathrm{sim}}$ 与 $J_{\mathrm{real}}$ 的**数值**接近与否还不够，还要问：**simulator 能不能正确预测"哪个 policy 更好"？** 考虑：

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

在 sim 上看起来 $A > B > C$，真机上却是 $B > C > A$。这时 simulator 不只是有 calibration error，而是**彻底失去了 model-selection utility**——你会用它挑出一个最差的政策。因此一个比"数值对齐"更本质的指标是排序相关性：

$$\rho_{\mathrm{rank}} \;=\; \mathrm{Spearman}\big(J_{\mathrm{sim}}(\pi_i),\ J_{\mathrm{real}}(\pi_i)\big)$$

因为 simulator 的价值**并不要求 $J_{\mathrm{sim}} \approx J_{\mathrm{real}}$，而至少要求它能正确排序候选 policy**。一个把所有分数都低估 20 分、但排序完全一致的 sim，仍然是个好工程工具；一个数值接近、排序却经常翻转的 sim，则可能比没有更危险。

## 组合与决策：从决策树到决策矩阵

把上面的框架收成一个"什么时候用哪个"的查询表。真实项目常常几个条件同时成立，所以更有用的不是单条 decision tree，而是一张 **gap × 可建模性 × 真机预算** 的决策矩阵（表中"真机预算"直接对应约束里的 $B$）：

| Gap | 可参数化 / 可辨识？ | Real data | 推荐 |
| --- | --- | ---: | --- |
| dynamics bias | 高 | 少 | SI |
| dynamics uncertainty | 中 | 少 | DR |
| dynamics residual | 低（但有结构） | 中 | Residual learning |
| visual appearance | 高 | 无 / 少 | DA / DR |
| actuator latency | 高 | 少 | SI + DR |
| unknown long-tail | 低 | 中 | real data / co-training |
| model unknown | 低 | 多 | world model（换 model source） |
| mixed | mixed | mixed | sim-and-real co-training 兜底 |

而现实中成功的系统，几乎靠的都是**四条轴的组合**而非单点技巧。一份很典型的 pipeline 是：

$$\text{SI}\ \rightarrow\ \text{DR}\ \rightarrow\ \text{DA}\ \rightarrow\ \text{Co-training / fine-tune}$$

每一步压的是**不同的 $\Delta$ 项**——SI 校一个"80% 对"的 sim，DR 在"说不清但可枚举"的方向上把族撑开，DA 处理视觉域差，最后用少量 real co-train / fine-tune 收尾残余。正因为它们作用在不同轴上，组合不是拼凑，而是"各补一块预算"。

## 这意味着什么？：把 sim-to-real 拉回数据与成本视角

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)的核心句是 evaluation-aware distribution allocation：有限预算下，把每一单位花到 marginal data value 最高的地方。把这条原则套回 sim-to-real，会得到一个自然推论——**仿真数据的 utility 从来不是 simulator 的内部属性，而是相对于真实 evaluation distribution 的属性：**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

这解释了一个很常见的挫败感：为什么"堆更多 sim 数据"经常看起来没用。但要注意别把这个说法讲绝对——增加 sim 数据有时确实会带来更广的 coverage、更多样的物体、更高的 rare-event 频率。更准确的表述是**有条件的**：

> **当主要瓶颈恰好是 simulator 与真实 evaluation distribution 之间的 support / fidelity mismatch 时，单纯增加同分布 simulation samples 的边际收益会快速下降——此时加 $N$ 主要提高的是采样密度，而不能自动创造缺失的 support 或修正 model bias。**

换句话说，它改善的是 density，而真正缺的是 support 与 $\Delta_{\mathrm{model}}$ 的保真度。于是与其问"我的 sim 有多好"，更值得问的是开头那句：**"我的 sim 在哪些 evaluation-relevant 区域上接近真实、在哪些区域上差得远？差得远的那些方向，该用 SI 校准、DR 覆盖、DA 对齐，还是干脆用真机数据补上？"**

把这条线走完，sim-to-real 就从"能不能迁移成功的开关"，变成了一个连贯的闭环：

$$\boxed{\text{Mismatch 分解} \rightarrow \text{intervention 选择} \rightarrow \text{真机预算分配} \rightarrow \text{评估效用}}$$

一句话收束：**sim-to-real 不是一个 transfer 技巧，而是一个"在 model fidelity、训练多样性、表示对齐、真机交互与工程成本之间做 constrained allocation"的问题。** 这和下篇"机器人数据 scaling 是一个 sequential data allocation 问题"其实是同一件事——只不过这一次，分配发生在仿真与真实之间。

---

## 参考文献

正文涉及的主要工作如下（均可通过 arXiv ID 检索）：

- Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World — Tobin et al., IROS 2017, arXiv:1703.06907
- Sim-to-Real Transfer of Robotic Control with Dynamics Randomization — Peng et al., ICRA 2018, arXiv:1710.06537
- Sim-to-Real: Learning Agile Locomotion For Quadruped Robots — Tan et al., RSS 2018, arXiv:1804.10332
- Learning Dexterous In-Hand Manipulation — Akkaya et al. (OpenAI), 2019, arXiv:1808.00177
- Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks — James et al., CVPR 2019, arXiv:1812.07252
- DiffTaichi: Differentiable Programming for Physical Simulation — Hu et al., ICLR 2020, arXiv:1910.00935
- Interactive Differentiable Simulation — Heiden et al., ICRA 2021, arXiv:1905.10706
- Residual Physics Learning and System Identification for Sim-to-real Transfer of Policies on Buoyancy Assisted Legged Robots — Chae et al., 2023, arXiv:2303.09597
- Sim-to-Real of Soft Robots with Learned Residual Physics — Michelis et al., 2024, arXiv:2402.01086
- Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) — Hafner et al., 2019, arXiv:1912.01603
- DayDreamer: World Models for Physical Robot Learning — Hafner et al., CoRL 2022, arXiv:2206.14176
- TD-MPC2: Scalable, Robust World Models for Continuous Control — Hansen et al., ICLR 2024, arXiv:2310.16828
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., 2025, arXiv:2503.24361
- A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies — 2026 follow-up, arXiv:2604.13645（作者与发表元数据以 arXiv 页面为准）
- MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning — Li et al., 2021, arXiv:2109.12674（其实验观察到：单纯增加异质仿真数据并不能消除与真实数据的 gap，加入 real cases 会改善真实测试表现——为"更多 synthetic data ≠ 自动解决 real-world generalization"提供了一个侧证）

需要说明的是，sim-to-real 目前尚不存在一份公认的"哪种方法更强"的跨任务定量对照——不同任务、不同硬件、不同 fidelity 上限下，结论可能完全颠倒。上述工作更多提供的是"这类 gap 用这个方法可行"的样本，而非可跨场景外推的排序。本文中关于四条 intervention axes 的正交分解、误差预算的 constrained-allocation 形式化、以及把 co-training 读作 data-mixture 的观点，都是 **conceptual framework 与作者解读，而非受控实验证明的结论**。

---

*本篇是"具身智能的数据问题"上下篇的续篇：上篇讲数据来源与接口、下篇讲数据 scaling 框架；这一篇把镜头拉到 sim-to-real，并刻意把它从"一堆迁移技巧"重述成一个带约束的误差预算分配问题——好让它可以接回下篇那条 sequential data allocation 的主线。*
