---
title: '具身智能 Sim-to-Real 方法论深潜：把"从仿真到真实"当成一次误差预算分配'
slug: "2026-09-10-sim-to-real-methodology"
date: 2026-09-10
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "Sim-to-Real", "Domain Randomization", "System Identification", "可微仿真", "Residual Physics", "世界模型", "Domain Adaptation", "机器人数据"]
description: "sim-to-real 常被讲成某个单一迁移技巧，但它本质是一个闭环的资源分配问题。本文把 reality gap 重述成 policy-conditioned 的多源 mismatch，用带权敏感度和边际效用（marginal utility）把“误差预算分配”写成一个可求解的框架，再逐一厘清 system identification、domain randomization、domain adaptation、real-world fine-tuning 四条正交 intervention axes 的机制与失效边界，并讨论 world model、residual physics、sim-and-real co-training 这几条容易被混为一谈的路线，最后回答一个常被回避的问题：什么时候最优解其实是“不做 sim-to-real”。"
toc: true
related_articles:
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
  - 2026-08-25-dreamer-explained
---

> 这是[数据问题上篇](/zh/articles/2026-09-08-data-and-training-recipes/)与[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)的续篇。上篇我用一张图把 sim-to-real 粗分成四类工具，但那只是 taxonomy。这一篇真正要回答的问题，可以压缩成一句——

> **当你的仿真数据在若干 evaluation-relevant 方向上离真实世界差得远时，下一单位的预算（工程时间、算力、还是机器人小时）应该花在哪条杠杆上：校准仿真器、扩大训练分布、对齐表示，还是干脆去采真机数据？**

这个问题乍看是工程直觉，其实是一个闭环的资源分配问题：给定几笔彼此不能互换的预算，你要不断问"现在把下一块钱花在哪，能换回最多的真实性能提升"。本文想做的，就是把这句直觉从一个漂亮的 metaphor，一路推到一个带边际效用的 framework。而真实项目里最卡人的，往往也不是"不知道有这些方法"，而是"不知道这个方法在我这类 gap 上到底管不管用、要花掉哪一种预算"。

## Reality Gap：不是一个标量，而是一个 policy-conditioned 的 mismatch

Sim-to-real 通常被叙述成"训练一个 policy，让它从仿真迁移到真实"。更严格的起点是**两个分布**：仿真器给出 $p_{\mathrm{sim}}(\tau)$，真实世界给出 $p_{\mathrm{real}}(\tau)$，二者一般不相等：

$$p_{\mathrm{sim}}(\tau) \;\neq\; p_{\mathrm{real}}(\tau)$$

但我们真正关心的不是这个分布差本身，而是它在某个任务上**表现出来的后果**——同一个 policy $\pi_\theta$ 在两边的性能之差：

$$\Delta J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)$$

这里要区分两件事：**distribution mismatch 不等于 performance gap**。$p_{\mathrm{sim}} \neq p_{\mathrm{real}}$ 并不自动意味着 $\Delta J$ 很大，因为不同 policy 对分布差的敏感度完全不同。一个只依赖粗粒度几何的 policy，换掉摩擦系数建模后性能可能几乎不变；一个依赖高频力反馈的精细装配 policy，同样的分布差就可能是致命的。

于是 $\Delta J(\pi)$ 是一个**任务相关、policy 相关的可观测后果（observable consequence）**，它至少同时依赖四样东西：

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ p_{\mathrm{sim}},\ p_{\mathrm{real}},\ \mathcal{E}\big)$$

其中 $\mathcal{E}$ 是 evaluation 的假设集合，包含 observation / action interface、initial-state 分布、horizon、reward 与 constraints 等。**同一个 simulator，对 position control policy 可能 gap 很小，对 force-sensitive manipulation policy 可能 gap 巨大。** 所以"我们的仿真器很真"从来不是一个有意义的评价——reality gap 不是仿真器的内在属性，而是这个四元组的属性。

顺带厘清一个容易混的点：$\mathcal{E}$ 与 $p_{\mathrm{sim}},p_{\mathrm{real}}$ 分工不同。$\mathcal{E}$ 是**规定评估协议**（在什么初始态、什么 horizon、什么 reward 下打分），而 $p_{\mathrm{sim}},p_{\mathrm{real}}$ 是**产生数据的机制**（动力学、观测、执行怎么演化）。观测模型 sim 与 real 不一致，是后者的差异；而"我们约定都在同一组 initial-state 上评"，是前者的设定——两者不该混为一谈。

### gap 到底在哪里：一个五层 mismatch 分解

第一步是把多源的 gap 拆开。工程上好用的粗粒度分解是五层，它们**性质不同、可参数化程度不同、补救成本更是天差地别**：

```
Reality mismatch
├── Dynamics / contact        摩擦、接触、可形变体、柔顺结构
├── Observation / estimation  传感器物理、标定、噪声、遮挡、时延、状态估计
├── Actuation / timing        电机动力学、控制频率、执行器延迟、通信抖动
├── Initial-state / env.      reset 分布、场景布局、长尾、初始条件
└── Objective / constraint    reward 定义、安全约束、成功判据
```

**观测与状态估计**值得单独成层，而不是塞进"感知 gap"里。原因很具体：机器人真正执行的是

$$a_t = \pi(o_t), \qquad o_t = h(x_t) + \epsilon$$

而真实世界里的 camera 标定误差、depth bias、遮挡、proprioception drift、力传感器偏置、state estimator 的时延与不同步，**并不是简单的"画面看起来不一样"**——它们让 **policy 实际看到的 state 估计，与 simulator 里假设可用的 state 不一致**。在 manipulation 与 locomotion 里，这一类"状态估计 gap"往往比外观 gap 更伤 performance：

```
Observation / Estimation
 ├─ sensor physics   成像/测距的物理过程
 ├─ calibration      内外参、手眼标定
 ├─ noise            曝光、量化、随机噪声
 ├─ occlusion        遮挡与部分可观测
 ├─ latency          传感与同步时延
 └─ state estimation 从 o 推断 s 的误差
```

**任务与初始状态**这一层则要劈成两半，因为它们是两个问题：

- **Environment / initial-state shift：** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$——reset 分布、场景布局对不上；这是 simulation 里的 distribution 问题，属于 reality gap。
- **Objective / task shift：** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$——仿真里只要求 grasp success，真实里还要求 collision avoidance；仿真允许较大 penetration，真实里 hardware safety 不允许。

后者严格说**已经不是 reality gap，而是 objective mismatch**：仿真器把物理模拟得再准，如果 reward / 约束和真实目标不是一回事，那不是"迁移失败"，而是"你评测的根本不是同一个任务"。本文后面谈 gap 压缩时，都默认 objective 已经对齐；objective mismatch 需要单独靠 reward shaping / 约束建模解决，不在这几条工具的射程内。

## 把"误差预算分配"写成一个可求解的框架

把 gap 拆成来源不同的层之后，就该给开头的直觉一个数学落点了。这里的写法是 **conceptual，不是严格定理**：这些误差项会强烈交互——例如仿真器假设精确的 proprioception，而真实有 latency，单看 latency 不致命、单看 dynamics mismatch 也不致命，但两者叠加可能直接让 controller 失稳。所以更稳妥的写法是先承认一个未知的耦合函数 $F$：

$$\Delta J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}},\ \Delta_{\mathrm{opt}}\big)$$

只有在工作点附近做工程归因时，才把它局部近似成一个加权和——而权重 $w_k$ 恰恰是**任务/policy 相关**的敏感度，这正呼应上一节"gap 是 policy-conditioned"的论点：

$$\Delta J \;\approx\; \sum_{k} w_k\, \Delta_k$$

有了这个近似，就能定义一个很有用的量：**第 $k$ 类 mismatch 对真实性能的敏感度**

$$S_k \;=\; \left|\frac{\partial J_{\mathrm{real}}}{\partial \Delta_k}\right|$$

它回答"这一类 gap 到底值不值得管"。而每一类 $\Delta_k$ 又能被某些干预以某种成本压下去，记干预 $k$ 压 gap 的效率为 $\partial \Delta_k / \partial C_k$。于是"先补哪一块"就有了一个干净优先级：

$$\text{priority}_k \;\propto\; S_k \cdot \frac{\partial \Delta_k}{\partial C_k}$$

用人话说就是：**优先处理"对任务最敏感、而且最便宜就能压下去"的那类 gap。**

### 真正的"分配"：把钱分给每条轴，而不是在方法里挑一个

到这里还只是"选方法"。要让"误差预算分配"这个说法名副其实，得让预算**连续地**分配到每条干预轴上。把总预算拆成一个向量 $b=(b_1,\dots,b_K)$，$b_k$ 是花在干预 $k$ 上的量——$b_{\mathrm{SI}}=2\text{h}$、$b_{\mathrm{DR}}=10^6$ 步 sim、$b_{\mathrm{real}}=4\text{h}$ 真机——而不是"用不用 SI"这种 0/1 选择。目标是最大化真实性能：

$$\max_{b}\quad J_{\mathrm{real}}\big(\pi_b\big)$$

关键是，机器人项目里的预算**根本不是同一种货币**。你可能 GPU 近乎无限、但真机机时极少；也可能有机器时间、却没有工程人力。所以正确的写法是**多预算约束**，而不是把它们折成一个标量 $B$：

$$\begin{aligned}
C_{\mathrm{real}}(b) &\le B_{\mathrm{real}}\\
C_{\mathrm{compute}}(b) &\le B_{\mathrm{compute}}\\
C_{\mathrm{eng}}(b) &\le B_{\mathrm{eng}}
\end{aligned}$$

一旦预算是分向量，**边际效用（marginal value）**就自然浮现：

$$MV_k \;=\; \frac{\partial J_{\mathrm{real}}}{\partial C_k}$$

这才是全文真正的核心句：**下一单位预算应该花在哪条干预轴上，取决于哪条轴当前的 marginal real-world utility 最高。** 而且 $MV_k$ 通常是递减的——这就解释了一个非常常见的现象：**"SI 先花 2 小时很值，继续花 20 小时就未必值。"** 因为最容易被辨识、影响最大的那几个参数早就在前 2 小时被校准掉了，剩下的边际收益迅速走低，此时同样这 18 小时挪去做 DR 或采真机，可能回报更高。

把每条干预对应到它主要压缩的项、以及它主要消耗哪一种预算，就得到这张表——注意成本这一列现在按**预算向量**拆开，因为把它们全塞进"sim 成本"会掩盖真相（system identification 的大头其实是真机激励实验 + 参数估计 + 仪器 + 仿真器工程 + 优化算力，而不是什么"simulator fidelity 成本"）：

| Intervention | 主要压缩项 | 主要预算 |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{eng}}$ + $C_{\mathrm{compute}}$ + 少量 $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | $C_{\mathrm{compute}}$（样本效率） |
| Residual physics | $\Delta_{\mathrm{model}}$（残差部分） | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$ | $C_{\mathrm{real}}$（未标注数据）+ $C_{\mathrm{compute}}$ |
| Real-world fine-tuning | 残余 $\Delta J$（含 $\Delta_{\mathrm{opt}}$） | $C_{\mathrm{real}}$（磨损 / 安全） |
| World model | 改变 model source | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Sim-and-real co-training | 改变 $p_{\mathrm{train}}$（$\Delta_{\mathrm{dist}}$ 为主） | 混合数据（$C_{\mathrm{real}}+C_{\mathrm{compute}}$） |

有了 $MV_k$，整篇文章就从"四种方法谁更好"变成一个闭环：**先定位哪一层 $\Delta_k$ 主导，再用 $S_k$ 判断它对当前任务有多敏感，然后沿边际效用最高的那条轴投下一份预算，最后在真实评估上量一量这份预算到底换回了多少 $J_{\mathrm{real}}$，据此决定下一份。** 这恰好接回下篇那句 evaluation-aware distribution allocation——只不过这里分配的对象，是仿真与真实之间的工程预算。

## 四条 intervention axes（而非四类互斥方法）

框架有了，再逐条看工具。先立一个结构：system identification、domain randomization、domain adaptation、real-world fine-tuning **不是同一抽象层级的并列类别**——SI 是 model calibration，DR 是 training distribution manipulation，DA 是 representation alignment，fine-tuning 是 optimization strategy。把它们并排成"四类方法"会误导人四选一。它们其实是**四条彼此正交的干预轴**，可以任意组合：

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

正因为正交，"组合"才天然成立——你可以在同一条轴上换手段、在不同轴上同时发力。

**选工具的标准，不是"systematic 交给 SI、random 交给 DR"。** 这个口诀当记忆法没错，但 SI 真正做的是估计

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; D\big(p_{\mathrm{sim}}(\tau \mid \phi),\; p_{\mathrm{real}}(\tau)\big)$$

它解决的是**可辨识、可参数化的 model mismatch**，而不是"凡是 systematic 都归它"（actuator gain、latency、friction、mass 本身都可能是随机过程，而非确定性的 systematic bias）。同理，DR 解决的是**能被一个训练分布表示出来的 uncertainty**。所以更有用的划分是三连问：

> **这个 mismatch 能不能被一个可信的参数化模型表达（parameterizable）？能不能从有限真机数据里辨识出来（identifiable）？如果不能辨识，能不能通过扩大训练分布去覆盖（coverable）？**

| mismatch 的性质 | 更自然的工具 |
| --- | --- |
| 可参数化 + 可辨识 | System Identification |
| 可参数化但难辨识 / uncertainty 大 | Domain Randomization |
| 不可参数化、但有 residual structure | Residual learning |
| observation / appearance mismatch | Domain Adaptation |
| policy 在目标域仍有 systematic residual | Fine-tuning |

### Axis A — Model：system identification、可微仿真与 residual physics

这条轴处理 $\Delta_{\mathrm{model}}$，内部有三个**正交**的问题，经常被"可微仿真 = 更强的 SI"这类含糊说法打包在一起：

$$f_{\mathrm{real}}(x,a) \;=\; \underbrace{f_{\mathrm{physics}}(x,a;\phi)}_{\text{可参数化的物理}} \;+\; \underbrace{r_\theta(x,a)}_{\text{残差}} \;+\; \epsilon$$

- **可微仿真回答"怎么优化模型"**——它提供 $\partial f/\partial\phi$ 这个 optimization interface；DiffTaichi（Hu et al., 1910.00935）、Interactive Differentiable Simulation（Heiden et al., ICRA 2021，1905.10706）让参数估计可以梯度化。
- **System identification回答"优化什么参数"**——即 $\phi$。经典 SI 扫参数、拟合轨迹；可微仿真把 $\phi$ 像权重一样反传更新。真实工作流还常常是 **real → identify → sim → train → real**，所以更准确的名字是 **real-to-sim-to-real**。
- **Residual physics回答"模型没解释掉的那部分由谁来解释"**——不去硬校准 $\phi$，而让网络学一个 $r_\theta$ 补差。

这里有一个决定成败、也最容易被"可微"二字掩盖的点：**可微性解决 optimization interface，不解决 model class correctness。** 如果 simulator 的 contact model 根本没表达某种真实现象，那么你对这个错误模型求再精确的梯度，也只能得到"错误模型之下的最优参数"。可微仿真让你把 $\phi$ 估得更准，却不会替你把 $f_{\mathrm{physics}}$ 的函数形式写对——写不对的部分只能交给残差，或者放弃"先建可信 sim"这个前提（见 world model）。三者正交，混成一个"高级可微仿真"叙事，恰恰会藏掉真正决定成败的 model class 问题。

SI 还有两个更细但很实在的坑。**其一，$p_{\mathrm{real}}(\tau)$ 在真实里几乎不可直接访问**，我们只有有限条真机轨迹 $\{\tau_i^{\mathrm{real}}\}_{i=1}^N$，所以上面那个 $\arg\min_\phi D(\cdot)$ 实际是在一个经验估计上跑的：$\hat\phi=\arg\min_\phi \sum_i \ell\big(\tau_i^{\mathrm{sim}}(\phi),\tau_i^{\mathrm{real}}\big)$，其中的 $p_{\mathrm{real}}$ 是由有限真机轨迹的经验分布近似的。**其二，参数存在 ≠ 参数可辨识**——identifiability 还依赖 excitation 与 sensor observability：质量、阻尼、刚度在某些激励条件下会产生几乎相同的可观测轨迹，无法被独立估计。把参数写进 simulator，绝不意味着它能从有限真机数据里被唯一估出来。

Residual physics 的适用边界也要收窄一句：它**不是"物理模型函数形式错了"就天然适用**。residual 的甜蜜点是"已有 physics model 能解释大部分结构、错误部分具有稳定且可学的结构"，即至少在目标分布上满足 $\|r_\theta\|\ll\|f_{\mathrm{physics}}\|$；如果 $f_{\mathrm{physics}}$ 完全错，$f_{\mathrm{physics}}+r_\theta$ 理论上也能拟合，但残差网络会退化成承担整个 dynamics、丢掉 physics 的 inductive bias——那还不如直接学一个 model。这条路线在软体（Michelis et al., 2402.01086）、浮力腿式（Chae et al., 2303.09597）这类"主干物理还算数、局部摩擦/接触/形变有稳定残差"的场景里最好用。

### Axis B — Data distribution：domain randomization 及其家族

这条轴不追求逼近某个"最准"的 $p_{\mathrm{real}}$，而是让 policy 对一族参数 $\{\phi\}$ 都稳健：训练时随机化物理、视觉、初始状态与 delay，只要真实落在这族里，policy 就能顶住。Tobin（1703.06907）用纯视觉随机化把 sim 抓取检测搬到真机；Peng（1710.06537）把随机化推进到 dynamics；OpenAI 的 in-hand manipulation（Akkaya et al., 1808.00177）几乎把 DR 推到极致——**不靠精确校准，靠"随机化范围足够宽"来吸收差异。**

一句常被写歪的直觉：DR 不是"隐式 ensemble"。它训练的是**单个**共享 policy $\pi_\theta$，目标大致是

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

而通常的 ensemble 是多个模型 $\{\pi_1,\dots,\pi_K\}$。更准确的说法是：**DR 是对一族环境模型做 population-level 优化，而不是针对单一 simulator 优化。** 直觉上它像 ensemble training（policy 被迫在一族 dynamics 上同时表现好），但结果是一个共享 policy，而非多个 policy 的集成。

DR 有效的条件也要写准：光"扩大范围"不够，真正要求的是**真实参数分布被 DR 的支撑覆盖、且在高质量区域上被训练到**——可以粗略记作 $p_{\mathrm{real}}(\phi) \ll p_{\mathrm{DR}}(\phi)$。$\phi_{\mathrm{real}}$ 落在 $\mathrm{support}(p_{\mathrm{DR}})$ 里、却正好落在极低概率的尾部，policy 在那里几乎没被训练过，照样会表现很差。所以过窄会漏、过散会虚，关键其实又回到分配那件事：**randomization 分布是否对齐 evaluation 分布与 objective**。过宽或与任务无关的 randomization 会降低样本效率、迫使 policy 在一堆互相冲突的 dynamics 上折中，从而显得过度保守；但在一些 robust / adversarial 设定下，适当扩大 uncertainty set 反而提升鲁棒性——所以"范围越宽越保守"并不是普遍规律，对齐与否才是。

"Adaptive / Automatic DR"也不是单一方法，而是一个家族，值得摊开：curriculum over randomization（随训练放宽范围）、adversarial domain randomization（采样最能击垮当前 policy 的参数）、automatic domain randomization（随表现自适应收缩范围）、posterior-based sampling（用辨识后验来采样）、performance-driven range adaptation。机制各异，共同点是**避免一开始就 over-randomize**。

### Axis C — Representation：domain adaptation 与观测翻译

这条轴处理 $\Delta_{\mathrm{obs}}$，既不校准物理、也不随机化，而是在**表示层**把 sim 与 real 对齐：feature-level adapter、image translation（GAN / 扩散）、或 randomized-to-canonical 的翻译网络 RCAN（James et al., CVPR 2019，1812.07252）——它把随机化过的 sim 图"翻译"回一张近似 canonical 的干净图再喂给下游 policy，正好**把 Axis B 的 DR 和这条轴缝起来**，缓解过宽随机化的性能损失。它处理的是"物理其实差不多、但看起来完全不像"的那部分 gap。

但有一条 DA 特有的失效边界值得单列：**对 policy learning，domain invariance 本身不是目标，task-relevant invariance 才是。** 只把两边特征对齐（$z_{\mathrm{sim}}\approx z_{\mathrm{real}}$）是不够的，理想状态是**在保持任务信息的同时缩小域差**——$I(z;y_{\mathrm{task}})$ 要高，同时 $D(z_{\mathrm{sim}},z_{\mathrm{real}})$ 要低。换句话说，对齐不是越强越好，而是只对齐那些与任务无关的变化。这跟上一节"过宽 DR 会抹掉任务信号"其实是同一件事，从表示层的角度看了一遍。

### Axis D — Optimization / adaptation：真机微调

这条轴处理"前三条轴补完之后剩下的残余 $\Delta J$"——先在仿真里大规模 pre-training 学结构，再用真机数据接力。但两个 regime 的成本结构完全不同，不能一句"做 RL 或 imitation 微调"糊在一起：

- **Offline / imitation：** $D_{\mathrm{real}} \to \theta$，主要成本是**数据采集**（一次性、可离线、可复用）。
- **Online RL：** $\pi_\theta \to a \to$ 真实 transition $\to \theta'$，主要成本是**交互 + 安全 + 硬件磨损 + 探索**（每一步都在消耗物理资源）。

这个区分决定了"值不值得微调"的判断：对真机学习，比较方法不能只看最终 success rate，还要看**达到目标性能所需的真机交互预算**。所以一个该进入 $C_{\mathrm{real}}$ 的务实指标是

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{或}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

同样涨 5 个点，一个要 2 机时、一个要 40 机时，在物理上根本不是同一回事。风险也不止"灾难性遗忘 / 错误先验"这一条：更常见的是**分布收窄**——真机 fine-tune 数据往往比 sim 分布窄得多（$D_{\mathrm{sim}} \to D_{\mathrm{real}}^{\mathrm{narrow}}$），微调后 policy 可能在目标部署切片上更好，鲁棒性却反而下降，等于**把 generalization 换成了 specialization**。如果 real 数据只覆盖一个狭窄 slice，fine-tune 会把一个 robust policy 拉回成一个 deployment-specific policy。

## 两条松动"两个给定分布"假设的新路线

上面四条轴共享一个隐含前提：**$p_{\mathrm{sim}}$ 与 $p_{\mathrm{real}}$ 是两个给定的分布**，你要做的是校准、覆盖、对齐或接力。而下面两条路线，恰恰在松动这个前提本身——它们不是"第五、第六种迁移技巧"，而是对整个问题的 reformulation。

### World model：不是取消 simulator，而是换掉 simulator 的来源

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)讨论过 world model 与 data utility 的关系。放进 sim-to-real 语境，先纠正一个定位上的误读：**world model 并不天然属于 sim-to-real。** 经典 sim-to-real 与 world model 路线的 causal direction 是不同的：

```
经典 sim-to-real：  sim dynamics → train policy → deploy real
World model route： real interaction → learn dynamics → imagine → optimize policy
```

需要说准的一点是：world model **并没有取消 simulator**，而是把 simulator 的角色从"手工指定的 physics model"换成"从交互数据里学出来的 predictive model"：

$$f_{\mathrm{hand\text{-}designed}} \;\longrightarrow\; f_{\mathrm{learned}}$$

真正被改变的是 **model source**。Dreamer（1912.01603）、TD-MPC2（2310.16828）体现了这条路。于是当**人工 simulator 的 model bias 太大、大到不值得先去修它**时，world model 提供的是**对 sim-to-real 问题本身的一种改写**，而不是它下面的一个 transfer technique。

DayDreamer（2206.14176）常被误当成"sim 预训练 → real 微调"的例子，但它的关键恰恰相反——world model 直接从真实机器人交互中学习、在 latent imagination 里做 policy improvement，几乎不依赖手工 sim。不过要说清楚：**不依赖手工 physics simulator，不等于 model-free。** world model 学习本身仍然吃满各种假设——representation、model architecture、action space、reward、exploration、真机数据质量；它只是把 inductive bias 从"显式 physics simulator"转移到了"learned world model"里。

诚实的边界："用真实数据学 dynamics" **不等于它天然优于仿真。** 它把"手工建模成本"换成了"真机采集成本 + 模型容量成本"；而在 contact-rich、long-tail、传感器噪声大的场景，学到的 model 常常在自己没见过的分布外区域给出**很自信、也很错的想象**，policy 会顺着错误规划走下去。所以它是"手工 sim"与"直接真机 RL"之间的**又一个 trade-off**，不是终局。

### Sim-and-real co-training：把"迁移"重述成 data mixture

Maddukuri et al.（2503.24361）提出的 Sim-and-Real Co-Training 是一个务实方向。**论文实际报告的结果**是：在同一批训练里把 sim 与 real 数据集混合采样，并在两个机器人平台、多任务上观测到平均性能提升；它不做 sim→real 单向迁移，而是用一个 recipe 决定两者比例与调度。

**本文的解读（非论文证明的结论）**是把它进一步理解成一个 **data-mixture 问题**：一旦这样重述，DR 与真实数据就不再是替代关系，而是同一份 sampling distribution 上 $T_R[p_{\mathrm{raw}}]$ 的两个来源（呼应上篇"$p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$、recipe 是数据到参数的转换函数"）。要写准的是，co-training **并不要求先去改造 simulator 本身，也不要求引入一个独立的 sim→real adapter**；它主要通过改变训练数据 mixture——$p_{\mathrm{train}}=\lambda\, p_{\mathrm{sim}}+(1-\lambda)\, p_{\mathrm{real}}$——来改变 policy 的优化分布（policy 参数 $\theta$ 自然仍会更新）。这也解释它为什么与上面四条轴正交：它不改 sim、不加工具，只动 mixture。

这条路线的**机理**目前也在被追问：2026 年已有后续工作（A Mechanistic Analysis of Sim-and-Real Co-Training，arXiv:2604.13645）尝试给出 co-training 为何有效的两个机制——structured representation alignment 与 importance reweighting。"混合有效"正在从经验现象变成一个有候选解释的研究对象。

## 评估：你怎么知道自己把 gap 补好了？

一个危险的做法是只在 sim benchmark 上报性能——那衡量的是 policy 与你自己 simulator 之间的一致性，而不是与真实世界的一致性。更可信的评估至少应做到：

- 报告 **zero-shot transfer**（不做任何真机微调）到真实系统的性能，以及 **few-shot / N-shot** 之后的曲线；
- 用一组 **held-out physical systems**（不同标定、不同相机、不同接触面）来测，而不是只有"那台部署机器人"；
- 明确声明 sim 与 real 的**任务、initial-state、evaluation distribution 是否一致**——否则比较根本不公平（这正是 objective mismatch 要先对齐的原因）；
- 做**失败归因**：是哪一层 $\Delta_k$ 主导？不同层的 $S_k$ 和补救成本天差地别，归因错了预算就花错了地方；
- 记 **$\eta_{\mathrm{real}}$**：对用了真机数据的路线，报"每单位真机磨损换来的性能"。

顺着"sim 是真实世界的代理"这句，还有一个比"数值对齐"更本质的问题：**simulator 能不能正确预测"哪个 policy 更好"？** 考虑：

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

在 sim 上看起来 $A > B > C$，真机上却是 $B > C > A$。这时 simulator 不只是有 calibration error，而是**失去了 model-selection utility**——你会用它挑出一个最差的政策。所以**当 simulator 被用于 policy / model selection 时**，一个比绝对数值误差更直接的指标是排序相关性：

$$\rho_{\mathrm{rank}} \;=\; \mathrm{Spearman}\big(J_{\mathrm{sim}}(\pi_i),\ J_{\mathrm{real}}(\pi_i)\big)$$

这是一个 **conditional metric**，只针对"用 sim 选 policy"这个用途——不代表它是评判 simulator 的唯一标准。sim 还有大量不必排序整个 policy family 的用途：representation pretraining、exploration、curriculum、safety filtering、controller initialization、rare-event generation，等等。但在"筛选 / 比较一整个候选 policy family"这个场景下，一个把所有分数低估 20 分却排序完全一致的 sim 仍是好工具，而一个数值接近、排序却经常翻转的 sim 可能比没有更危险。这也意味着：评价 simulator fidelity 时不该只盯着单个 policy，而应相对于**候选 policy family** 来评，即记作 $U_{\mathrm{sim}}=U(D_{\mathrm{sim}}\mid \Pi_{\mathrm{candidate}},\,p_{\mathrm{eval}}^{\mathrm{real}})$。

## 组合与决策，以及一个常被回避的问题

有了优先级，"什么时候用哪个"就不该是一条固定流水线，而是一张查询表。真实项目常常几个条件同时成立，所以更有用的是 **gap × 可建模性 × 真机预算** 的决策矩阵（"Real data"列直接对应上面的预算向量 $B$）：

| Gap | 可参数化 / 可辨识？ | Real data | 推荐 |
| --- | --- | ---: | --- |
| dynamics bias | 高 | 少 | SI |
| dynamics uncertainty | 中 | 少 | DR |
| dynamics residual | 低（但有结构） | 中 | Residual learning |
| visual appearance | 高 | 无 / 少 | DA / DR |
| actuator latency | 高 | 少 | SI + DR |
| unknown long-tail，可模拟 | 低 | 少 | targeted simulation / DR |
| unknown long-tail，sim 生成不可信 | 低 | 中 | real data |
| model unknown | 低 | 多 | world model |
| mixed | mixed | mixed | co-training 兜底 |

一份很常见的组合是 **SI → DR → DA → co-training / fine-tune**：SI 校一个"80% 对"的 sim，DR 在"说不清但可枚举"的方向上把族撑开，DA 处理视觉域差，最后用少量真机数据收尾。**但这里的箭头不是固定 workflow，只是一个示意组合**——实际顺序应当由当前主导 gap 与各干预的边际效用决定：如果真机数据本就充足，先做 SI 未必划算；如果视觉才是主导项，DA 就该提前；如果 SI 只有极少数据可用，先粗粗 DR、拿到能跑的 policy、再回头校准，常常更合理。

顺着这个逻辑，就能回答一个整篇几乎都在回避、但框架本身允许的反问：**什么时候最优解其实是"不做 sim-to-real"？**

- **真机数据已经便宜到 $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}$** 时——比如采集台架成熟、任务允许安全地反复跑——直接用真机数据训练可能比修仿真更省事。
- **仿真器 model class 本身就差（$\Delta_{\mathrm{model}}$ 主导且难以参数化）** 时——软体、流体、复杂接触——修 sim 的成本高到边际效用极低，不如走 world model 或真机数据学习。
- **部署分布非常固定** 时——根本不需要大规模 DR 去覆盖一整个族，少量 targeted real fine-tuning 往往更划算。

能大方承认"有时最优解是不做 sim-to-real"，恰恰是 allocation framing 应有的样子：**它不站"仿真"这个队，只站"下一单位预算换回最多真实性能"这个队。**

## 这意味着什么？：一个闭环，而不是一个开关

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)的核心句是 evaluation-aware distribution allocation：有限预算下，把每一单位花到 marginal data value 最高的地方。套回 sim-to-real，会得到一个自然推论——**仿真数据的 utility 从来不是 simulator 的内部属性，而是相对于真实 evaluation distribution 的属性：**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

这解释了一个常见挫败：为什么"堆更多 sim 数据"有时没用。但要讲准：增加 sim 数据确实也可能带来更广 coverage、更多样的物体、更高的 rare-event 频率。所以这句判断是**有条件的**——

> **当主要瓶颈恰好是 simulator 与真实 evaluation distribution 之间的 support / fidelity mismatch 时，单纯增加同分布 simulation samples 的边际收益会快速下降；此时加 $N$ 主要提高的是采样密度，而不能自动创造缺失的 support 或修正 model bias。**

它改善的是 density，而真正缺的是 support 与 $\Delta_{\mathrm{model}}$ 的保真度。于是与其问"我的 sim 有多好"，不如问开头那句：**"我的 sim 在哪些 evaluation-relevant 方向上接近真实、在哪些方向上差得远？差得远的那些，其敏感度 $S_k$ 有多高、用哪种预算压它最便宜？"**

把这条线走完，sim-to-real 就不再是"能不能迁移成功"的开关，而是这样一条闭环链路：

$$\boxed{\text{mismatch} \rightarrow \text{sensitivity} \rightarrow \text{intervention} \rightarrow \text{marginal utility} \rightarrow \text{budget allocation} \rightarrow \text{real evaluation}}$$

一句话收束：**sim-to-real 不是一个 transfer 技巧，而是在 model fidelity、训练多样性、表示对齐、真机交互与工程成本之间做 constrained allocation 的闭环问题。** 这和下篇"机器人数据 scaling 是一个 sequential data allocation 问题"其实是同一件事——只不过这一次，分配发生在仿真与真实之间。

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
- A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies — 2026, arXiv:2604.13645（作者与发表元数据以 arXiv 页面为准）

需要说明的是，sim-to-real 目前尚不存在一份公认的"哪种方法更强"的跨任务定量对照——不同任务、不同硬件、不同 fidelity 上限下，结论可能完全颠倒。上述工作更多提供的是"这类 gap 用这个方法可行"的样本，而非可跨场景外推的排序。本文中关于四条 intervention axes 的正交分解、误差预算的 constrained-allocation 形式化、敏感度 $S_k$ 与边际效用 $MV_k$ 的定义、以及把 co-training 读作 data-mixture 的观点，都是 **conceptual framework 与作者解读，而非受控实验证明的结论**。

---

*本篇是"具身智能的数据问题"上下篇的续篇：上篇讲数据来源与接口、下篇讲数据 scaling 框架；这一篇把镜头拉到 sim-to-real，把它从"一堆迁移技巧"重述成一个带边际效用的闭环分配问题——好让它能接回下篇那条 sequential data allocation 的主线。*
