---
title: '具身智能 Sim-to-Real 方法论深潜：把"从仿真到真实"当成一次误差预算分配'
slug: "2026-09-10-sim-to-real-methodology"
date: 2026-09-10
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "Sim-to-Real", "Domain Randomization", "System Identification", "可微仿真", "Residual Physics", "世界模型", "Domain Adaptation", "机器人数据"]
description: 'sim-to-real 常被讲成某个单一迁移技巧，但它本质是一个闭环的资源分配问题。本文把 reality gap 重述成 policy-conditioned 的多源 mismatch，用带权敏感度与经验边际效用把"误差预算分配"写成一个可估计、可迭代优化的决策框架，再逐一厘清 system identification、domain randomization、domain adaptation、real-world fine-tuning 四条相对独立的 intervention axes 的机制与失效边界，并讨论 world model、residual physics、sim-and-real co-training 这几条容易被混为一谈的路线，最后回答一个常被回避的问题：什么时候最优解其实是"不做 sim-to-real"。'
toc: true
related_articles:
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
  - 2026-08-25-dreamer-explained
---

> 接[数据问题上篇](/zh/articles/2026-09-08-data-and-training-recipes/)与[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)。上篇用一张图把 sim-to-real 粗分成四类工具，那只是 taxonomy。这一篇真正要回答的问题可以压缩成一句——

> **当你的仿真数据在若干 evaluation-relevant 方向上离真实世界差得远时，下一单位的预算（工程时间、算力、还是机器人小时）应该花在哪条杠杆上：校准仿真器、扩大训练分布、对齐表示，还是干脆去采真机数据？**

这个问题乍看是工程直觉，其实是一个闭环的资源分配问题：给定几笔不能互换的预算，你要不断问"下一块钱花在哪，能换回最多的真实性能"。本文想做的，就是把这句直觉从一个 metaphor 推到一个带边际效用的决策框架——真实项目里最卡人的往往不是"不知道有这些方法"，而是"不知道这个方法在我这类 gap 上管不管用、要花掉哪一种预算"。

先给标题里的"误差预算"降一档歧义：它**不是**给每个误差项预先分配一笔固定额度（$\Delta J=\sum_k \Delta_k$、逐项发钱），而是把预算花在**干预动作**上，通过 sequential allocation 逐步压低当前最有价值的那类 mismatch——误差是要压的对象，预算才是真正在分的东西。

## Reality Gap：不是一个标量，而是一个 policy-conditioned 的 mismatch

Sim-to-real 常被叙述成"训练一个 policy 从仿真迁移到真实"。更严格的起点是**两个分布**：同一条 policy $\pi$ 与环境交互，各自诱导轨迹分布 $p_{\mathrm{sim}}^{\pi}(\tau)$ 与 $p_{\mathrm{real}}^{\pi}(\tau)$，二者一般不等：

$$p_{\mathrm{sim}}^{\pi}(\tau) \;\neq\; p_{\mathrm{real}}^{\pi}(\tau)$$

轨迹分布本身是 **policy-induced** 的，随 $\pi$ 变而变，不是环境的固有属性。而我们真正关心的不是这个分布差本身，而是它在某个任务上**表现出来的后果**——同一个 policy $\pi$ 在两边的性能之差：

$$\boxed{\;\delta_J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)\;}$$

我把它叫 **transfer delta**，保留符号：$J$ 若是 success rate，真实反而更好时（比如 simulator 更保守、或 sim 里噪声比真实更狠）$\delta_J$ 会是正的，直觉上不该叫"gap"。所以另把它的幅度

$$G_J(\pi) \;=\; \big|\,\delta_J(\pi)\,\big|$$

单独叫 **performance gap**。下面谈敏感度时用的是 $G_J$ 这一语义，就不会和符号纠缠。

要注意 **distribution mismatch 不等于 performance gap**：$p_{\mathrm{sim}}^{\pi} \neq p_{\mathrm{real}}^{\pi}$ 并不自动意味着 $\delta_J$ 很大，因为不同 policy 对分布差的敏感度完全不同。只依赖粗粒度几何的 policy，换掉摩擦系数建模后性能可能几乎不变；依赖高频力反馈的精细装配 policy，同样的分布差就可能是致命的。

$\delta_J(\pi)$ 是**任务相关、policy 相关的可观测后果**，至少同时依赖四样东西：

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ p_{\mathrm{sim}},\ p_{\mathrm{real}},\ \mathcal{E}\big)$$

$\mathcal{E}$ 是 evaluation 的假设集合——规定评估协议（在什么 initial-state、horizon、reward、constraints 下打分），而 $p_{\mathrm{sim}},p_{\mathrm{real}}$ 是**产生数据的机制**；观测模型 sim 与 real 不一致属后者、"约定在同一组 initial-state 上评"属前者，两者不该混。所以"我们的仿真器很真"从来不是有意义的评价：同一个 simulator 对 position control policy 可能 gap 很小、对 force-sensitive manipulation policy 可能 gap 巨大。reality gap 是这个四元组的属性，不是仿真器本身的属性。

### gap 到底在哪里：reality mismatch 与 task-specification mismatch

第一步是把多源的 gap 拆开——它其实有**两大类来源**，不能全塞进"reality"这一个词下面：

```
Sim-to-real / task mismatch
├── Reality mismatch（物理层面）
│   ├── Dynamics / contact        摩擦、接触、可形变体、柔顺结构
│   ├── Observation / estimation  传感器物理、标定、噪声、遮挡、时延、状态估计
│   ├── Actuation / timing        电机动力学、控制频率、执行器延迟、通信抖动
│   └── Initial-state / env.      reset 分布、场景布局、长尾、初始条件
└── Task-specification mismatch
    └── Objective / constraint    reward 定义、安全约束、成功判据
```

两类来源不同，不要简单相加：reality mismatch 是"仿真和真实不是同一个世界"，task-specification mismatch 是"你优化的目标和部署的目标根本不是同一个任务"。

**观测与状态估计**值得单独成层，而不是塞进"感知 gap"里。机器人真正执行的是

$$a_t = \pi(o_t), \qquad o_t = h(x_t) + \epsilon$$

真实世界里的 camera 标定误差、depth bias、遮挡、proprioception drift、力传感器偏置、state estimator 时延，**不是简单的"画面看起来不一样"**——它们让 **policy 实际看到的 state 与 simulator 里假设可用的 state 不一致**。在 manipulation 与 locomotion 里，这类"状态估计 gap"往往比外观 gap 更伤 performance。

**初始状态**与**任务目标**则要分开看，因为它们分属两大类：

- **Initial-state / environment shift（属 reality mismatch）：** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$——reset 分布、场景布局对不上，这是仿真里的 distribution 问题。
- **Objective / task shift（属 task-specification mismatch）：** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$——仿真里只要求 grasp success，真实里还要求 collision avoidance；仿真允许较大 penetration，真实里 hardware safety 不允许。

后者严格说**已经不是 reality gap，而是 objective mismatch**：仿真器把物理模拟得再准，如果 reward / 约束和真实目标不是一回事，那不是"迁移失败"，而是"你评测的根本不是同一个任务"。下文谈 gap 压缩时默认 objective 已对齐；objective mismatch 需单独靠 reward shaping / 约束建模处理。

## 把"误差预算分配"写成一个可估计、可迭代优化的决策框架

拆完来源，就该给开头的直觉一个数学落点。这里的写法是 **conceptual，不是严格定理**：误差项强烈交互——仿真器假设 proprioception 精确、真实有 latency，单看都不致命、叠加却可能让 controller 失稳。所以更稳妥的写法是先承认一个未知的耦合函数 $F$：

$$\boxed{\;\delta_J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}}\big)\;}$$

**这一版把 $\Delta_{\mathrm{opt}}$（优化 / 学习误差）从 reality gap 里拿掉了。** 层级不同：对**同一个固定 policy**，仿真、观测、动力学都很准但 RL 没训好时，$\delta_J$ 其实很小（两边性能差不多）、policy 却很差——"policy 没学好" ≠ "sim-to-real gap 大"，把 $\Delta_{\mathrm{opt}}$ 塞进 $F$ 会把两件事重新搅在一起。正确的做法是把两件事在概念上分开（而不是强行相加）：

$$\underbrace{J_{\mathrm{real}}(\pi_{\mathrm{train}}) - J_{\mathrm{sim}}(\pi_{\mathrm{train}})}_{\text{transfer gap}}\qquad \underbrace{J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})}_{\text{learning / optimization gap}}$$

$$\text{deployment loss} \;=\; \text{transfer gap} \;+\; \text{learning / optimization gap}$$

前者回答"simulator 对已训出的 policy 有多大误导"，后者回答"policy 本身学得好不好"——这个区分后面还会让 fine-tuning 的定位干净很多。

只有在工作点附近做工程归因时，才把 $F$ 局部近似成一个加权和——而权重 $w_k$ 恰恰是**任务/policy 相关**的敏感度，这正呼应上一节"gap 是 policy-conditioned"的论点：

$$\delta_J \;\approx\; \sum_{k} w_k\, \Delta_k, \qquad S_k \;=\; \left|\frac{\partial J_{\mathrm{real}}}{\partial \Delta_k}\right|$$

$S_k$ 回答"这一类 gap 到底值不值得管"。而每一类 $\Delta_k$ 又能被某些干预以某种成本压下去，记效率为 $\partial \Delta_k / \partial C_k$，于是"先补哪一块"就有了一个干净优先级 $\text{priority}_k \propto S_k \cdot \partial\Delta_k / \partial C_k$——**优先处理"对任务最敏感、而且最便宜就能压下去"的那类 gap。**

但这里要给数学叙事降一档：$\Delta_k$ 并不是 dynamics error / latency / observation error 三个自由旋钮，$\Delta_{\mathrm{model}}$ 和 $\Delta_{\mathrm{ctrl}}$ 甚至可能不可辨识地互相补偿（sim 里 actuator gain 错了，policy 会靠不同 command distribution 补回来）。所以 $S_k$（以及下一节的 $MV$）**不应理解为能从 simulator 解析算出的物理量，而是通过 sensitivity experiments / ablation / 小规模真实评估估计的 decision statistics。** 承认这一点，框架反而更强：它不是"可解析求解的优化公式"，而是"用 sequential experiments 估边际收益的分配流程"。

### 真正的"分配"：把钱花在干预动作上，而不是在方法里挑一个

到这里还只是"选方法"。要让"预算分配"名副其实，得让预算**连续地**分到每条干预轴上：把总预算拆成向量 $b=(b_1,\dots,b_K)$，$b_k$ 是花在干预 $k$ 上的量——$b_{\mathrm{SI}}=2\text{h}$、$b_{\mathrm{DR}}=10^6$ 步 sim、$b_{\mathrm{real}}=4\text{h}$ 真机——而不是"用不用 SI"这种 0/1 选择。目标是最大化真实性能：

$$\max_{b}\quad J_{\mathrm{real}}\big(\pi_b\big)$$

关键是，机器人项目里的预算**不是同一种货币**：GPU 近乎无限但真机机时极少、有机器时间却没工程人力。所以正确写法是**多预算约束**，不折成标量 $B$：

$$\begin{aligned}
C_{\mathrm{real}}(b) &\le B_{\mathrm{real}}\\
C_{\mathrm{compute}}(b) &\le B_{\mathrm{compute}}\\
C_{\mathrm{eng}}(b) &\le B_{\mathrm{eng}}
\end{aligned}$$

预算是分向量之后，就该把决策变量从"gap"换成"干预动作"。工程师买不到"$\Delta_{\mathrm{model}}$ 的 2 个百分点"，能买到的是：做 30 分钟 SI、再跑 $10^6$ 步 sim、收 100 条真机轨迹、做一次 camera calibration、加一个 latency randomization、训一个 residual model。所以更自然的 formulation 是对干预 $m$ 定义边际效用——**干预不是直接改 $\Delta_k$，而是通过训练过程去改变 policy**：

$$\boxed{\;\pi_{b+m} \;=\; \operatorname{Train}\big(D_{\mathrm{sim}},\ D_{\mathrm{real}};\ m\big)\;}$$

于是"下一块钱花在哪"写成一个以干预为变量、需要在真实世界里逐步估计的量：

$$\boxed{\;m^{*} \;=\; \operatorname*{arg\,max}_{m}\; \frac{\mathbb{E}\big[\,J_{\mathrm{real}}(\pi_{b+m}) - J_{\mathrm{real}}(\pi_{b})\,\big]}{C(m)}\;}$$

这个 ratio 无法从 simulator 解析求得，只能用 pilot experiments / ablation / few-shot real evaluation **sequential 地估**。它天然是一个 **sequential empirical decision**（接近 bandit / active experimentation），也接回上一篇的 sequential data allocation。而 $MV$ 通常随投入递减，这就解释了**"SI 先花 2 小时很值、继续花 20 小时未必值"**：最易辨识、影响最大的几个参数早在前 2 小时被校准掉，剩下 18 小时挪去做 DR 或采真机往往回报更高。

把每条干预对应到主要压缩项、主要预算，就得到这张表——成本按**预算向量**拆开（SI 的大头其实是真机激励 + 参数估计 + 仪器 + 仿真器工程 + 优化算力，不是"simulator fidelity 成本"）：

| Intervention | 主要压缩项 | 主要预算 |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{eng}}$ + $C_{\mathrm{compute}}$ + 少量 $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | $C_{\mathrm{compute}}$（样本效率） |
| Residual physics | $\Delta_{\mathrm{model}}$（残差部分） | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$（appearance 子集） | $C_{\mathrm{real}}$（未标注数据）+ $C_{\mathrm{compute}}$ |
| Real-world fine-tuning | transfer residual + task-learning gap | $C_{\mathrm{real}}$（磨损 / 安全） |
| World model | 改变 model source | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Sim-and-real co-training | 改变 $p_{\mathrm{train}}$（$\Delta_{\mathrm{dist}}$ 为主） | 混合数据（$C_{\mathrm{real}}+C_{\mathrm{compute}}$） |

有了这套写法，全文就不是"四种方法谁更好"，而是一个闭环：定位主导的 $\Delta_k$、用 sensitivity 判断它有多重要、在 $MV$ 最高的那条干预上投一份预算、在真实评估上量回报、再决定下一份。这恰好接回下篇那句 evaluation-aware distribution allocation——只是这次分配的对象是仿真与真实之间的工程预算。

## 四条 intervention axes（更准确说，四个相对独立的干预维度）

框架有了，再逐条看工具。先立一个结构：SI、DR、DA、real-world fine-tuning **不是同一抽象层级的并列类别**——SI 是 model calibration，DR 是 training distribution manipulation，DA 是 representation alignment，fine-tuning 是 optimization strategy。并排成"四类方法"会误导人四选一，它们其实是**四个相对独立的干预维度**，可以组合：

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

但要诚实补一句：这里的"$\times$"表示**组合空间**，不是数学意义上的正交。DR 随机化物理、视觉、初始状态与 delay，同时触及 Model / Observation / Distribution；DA 可以发生在 input / feature / latent dynamics / policy / output——"DA = Representation 轴"也是 abstraction 而非严格定义。所以更准确的说法是"**相对独立、可以组合的维度**"，不是"彼此正交的轴"。

**选工具的标准，不是"systematic 交给 SI、random 交给 DR"。** 这个口诀当记忆法没错，但 SI 真正做的是估计

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; D\big(p_{\mathrm{sim}}(\tau \mid \phi),\; p_{\mathrm{real}}(\tau)\big)$$

它解决的是**可辨识、可参数化的 model mismatch**，不是"凡是 systematic 都归它"（actuator gain、latency、friction、mass 都可能是随机过程而非确定性 bias）。同理 DR 解决的是**能被一个训练分布表示出来的 uncertainty**。所以更有用的划分不是二元选择，而是一条从"点估计 → 后验 → 鲁棒随机化"的连续谱：

| mismatch 的性质 | 更自然的工具 |
| --- | --- |
| 可参数化 + 可辨识 | System Identification（point estimate $\hat\phi$） |
| 可参数化但只能给出不确定性 | Bayesian / posterior SI → posterior-guided DR |
| 可参数化但难辨识 / uncertainty 大 | Domain Randomization |
| 不可参数化、但有 residual structure | Residual learning |
| observation / appearance mismatch | Domain Adaptation |
| policy 在目标域仍有 systematic residual | Fine-tuning |

关键在于：**"不能精确辨识"和"完全不知道"不是一回事。** 拿到后验 $p(\phi \mid D_{\mathrm{real}})$ 之后，最自然的动作不是"既然不确定干脆 uniform DR"，而是 $\phi \sim p(\phi \mid D_{\mathrm{real}})$ 做 **posterior-guided randomization**——把 SI 与 DR 缝成一条连续谱，也更贴合全文的 allocation 主题。

### Axis A — Model：system identification、可微仿真与 residual physics

这条轴处理 $\Delta_{\mathrm{model}}$，内部有三个**不同层次**的问题，经常被"可微仿真 = 更强的 SI"这类含糊说法打包在一起：

$$x_{t+1} \;=\; \underbrace{f_{\mathrm{physics}}(x_t,a_t;\phi)}_{\text{可参数化的物理}} \;+\; \underbrace{r_\theta(x_t,a_t)}_{\text{残差}} \;+\; \epsilon_t$$

- **可微仿真回答"怎么优化模型"**——它提供 $\partial f/\partial\phi$ 这个 optimization interface；DiffTaichi（Hu et al., ICLR 2020，1910.00935）、Interactive Differentiable Simulation（Heiden et al., arXiv 2019，1905.10706）让参数估计可以梯度化。
- **System identification 回答"优化什么参数"**——即 $\phi$。经典 SI 扫参数、拟合轨迹；可微仿真把 $\phi$ 像权重一样反传更新。真实工作流还常常是 **real → identify → sim → train → real**，所以更准确的名字是 **real-to-sim-to-real**。
- **Residual physics 回答"模型没解释掉的那部分由谁来解释"**——不去硬校准 $\phi$，而让网络学一个 $r_\theta$ 补差。

这里 $r_\theta$ 只是**统一记号**：实际 residual 未必直接加在 $f$ 上，可以定义在状态转移、force、acceleration、contact impulse、deformation field 或其他 simulator latent 上——例如软体那篇工作学的就是施加到整个 simulated mesh 上的 residual force。

这里有一个决定成败、也最容易被"可微"二字掩盖的点：**可微性解决 optimization interface，不解决 model class correctness。** 如果 simulator 的 contact model 根本没表达某种真实现象，对这个错误模型求再精确的梯度，也只会得到"错误模型之下的最优参数"。可微仿真让你把 $\phi$ 估得更准，却不会替你写对 $f_{\mathrm{physics}}$ 的函数形式——写不对的部分只能交给残差，或者放弃"先建可信 sim"这个前提（见 world model）。

SI 还有两个更细但实在的坑。**其一，$p_{\mathrm{real}}(\tau)$ 在真实里几乎不可直接访问**，只有有限条真机轨迹 $\{\tau_i^{\mathrm{real}}\}_{i=1}^N$，所以上面那个 $\arg\min_\phi D(\cdot)$ 实际是在一个经验估计上跑：$\hat\phi=\arg\min_\phi \sum_i \ell\big(\tau_i^{\mathrm{sim}}(\phi),\tau_i^{\mathrm{real}}\big)$。**其二，参数存在 ≠ 参数可辨识**——identifiability 还依赖 excitation 与 sensor observability：质量、阻尼、刚度在某些激励下会产生几乎相同的可观测轨迹、无法被独立估出来。

Residual physics 的边界也要收窄：它**不是"物理模型函数形式错了"就天然适用**。常见甜蜜点是 residual 在目标分布上相对受限（$\|r_\theta\| \ll \|f_{\mathrm{physics}}\|$），但真正关键的不是残差必须"小"，而是 $f_{\mathrm{physics}}$ 是否仍提供**有用的结构性归纳偏置**——inductive bias、state representation、constraints、extrapolation prior。反之若 $f_{\mathrm{physics}}$ 完全错、残差必须独自承担整个 dynamics，那还不如直接学一个 model。它在软体（Gao et al., RA-L 2024，2402.01086）、浮力腿式（Sontakke et al., 2023，2303.09597）这类"主干物理还算数、局部摩擦/接触/形变有稳定残差"的场景里最好用。

### Axis B — Data distribution：domain randomization 及其家族

这条轴不追求逼近某个"最准"的 $p_{\mathrm{real}}$，而是让 policy 对一族参数 $\{\phi\}$ 都稳健：训练时随机化物理、视觉、初始状态与 delay，只要真实落在这族里，policy 就能顶住。Tobin（1703.06907）用纯视觉随机化把 sim 抓取检测搬到真机；Peng（1710.06537）把随机化推进到 dynamics；OpenAI 的 in-hand manipulation（Akkaya et al., 1808.00177）几乎把 DR 推到极致——**不靠精确校准，靠"随机化范围足够宽"来吸收差异。**

一句常被写歪的直觉：DR 不是"隐式 ensemble"。它训练的是**单个**共享 policy $\pi_\theta$，目标大致是

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

而通常的 ensemble 是多个模型 $\{\pi_1,\dots,\pi_K\}$。更准确的说法是：**DR 是对一族环境模型做 population-level 优化，而不是针对单一 simulator 优化**——直觉上像 ensemble training，但结果是一个共享 policy，不是多个 policy 的集成。

DR 有效的条件也要写准：光"扩大范围"不够，通常要求真实参数分布被 DR 的支撑覆盖、且在高质量区域上被训练到——可粗略记作 $p_{\mathrm{real}}(\phi) \ll p_{\mathrm{DR}}(\phi)$；$\phi_{\mathrm{real}}$ 落在 support 里却正好落在极低概率的尾部，policy 在那里几乎没训练过，照样会挂。但更要紧的是：**对 policy transfer，真正需要覆盖的不是参数空间的边际 support，而是 policy 在 evaluation 下实际访问到的 state-action / contact occupancy $d_{\mathrm{real}}^{\pi}(s,a)$**——参数覆盖 ≠ deployment 轨迹被覆盖，而且参数之间的 **joint correlation** 也很难被各维独立随机化替代。关键又回到分配：**randomization 分布是否对齐 evaluation 分布与 objective**。过宽或与任务无关的 randomization 会降低样本效率、迫使 policy 在冲突 dynamics 上折中；但在一些 robust / adversarial 设定下，适当扩大 uncertainty set 反而更稳——所以"越宽越保守"并非普遍规律，对齐与否才是。

"Adaptive / Automatic DR"也不是单一方法而是一个家族：curriculum over randomization、adversarial DR（采样最能击垮当前 policy 的参数）、automatic DR（根据训练表现或真实反馈自适应调整采样分布）、posterior-based sampling、performance-driven range adaptation——机制各异（有的扩大、有的缩小、有的专门找 hard domain），共同点是**避免一开始就 over-randomize**。

### Axis C — Representation：domain adaptation 与观测翻译

这条轴处理 $\Delta_{\mathrm{obs}}$，既不校准物理、也不随机化，而是在**表示层**把 sim 与 real 对齐：feature-level adapter、image translation（GAN / 扩散）、或 randomized-to-canonical 的翻译网络 RCAN（James et al., CVPR 2019，1812.07252）——它把随机化过的 sim 图"翻译"回一张近似 canonical 的干净图再喂给下游 policy，正好**把 Axis B 的 DR 和这条轴缝起来**。它处理的是"物理其实差不多、但看起来完全不像"的那部分 gap。

但要写准两条边界：**其一，DA 只是 observation mismatch 的一个子集**——尤其适合 appearance / representation shift，而 camera intrinsics/extrinsics、temporal sync、sensor bias、depth distortion、state estimation 更适合 calibration / SI / sensor modeling；不然读者会形成"observation gap → DA"的新错误口诀。**其二，对 policy learning，domain invariance 本身不是目标，task-relevant invariance 才是**——只把 $z_{\mathrm{sim}}\approx z_{\mathrm{real}}$ 对齐不够，理想是保持任务信息 $I(z;y_{\mathrm{task}})$ 高的同时压低 $D(z_{\mathrm{sim}},z_{\mathrm{real}})$，即只对齐与任务无关的变化。这跟上一节"过宽 DR 会抹掉任务信号"其实是同一件事，从表示层看了一遍。

### Axis D — Optimization / adaptation：真机微调

这条轴处理前三条轴补完之后剩下的残余——先在 sim 里大规模 pre-training 学结构，再用真机数据接力。承接前面那个区分，fine-tuning 同时压两样东西：**transfer residual** 与 **task-learning gap（policy 本身没学够）**，别只说成"补 reality gap"。而且两个 regime 的成本结构完全不同，不能一句"做 RL 或 imitation 微调"糊在一块：

- **Offline / imitation：** $D_{\mathrm{real}} \to \theta$，主要成本是**数据采集**（一次性、可离线、可复用）。
- **Online RL：** $\pi_\theta \to a \to$ 真实 transition $\to \theta'$，主要成本是**交互 + 安全 + 硬件磨损 + 探索**（每一步都在消耗物理资源）。

所以比较方法不能只看最终 success rate，还要看**达到目标性能所需的真机交互预算**。一个常被引用的粗略指标是

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{或}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

但它只是**粗略指标**：依赖 baseline（5%→10% 和 80%→85% 都是 +5%，意义完全不同），也不是真正的 marginal efficiency。真正该看 learning curve、达到目标所需 real samples、AULC、以及每 100 条轨迹的边际收益

$$MV_{\mathrm{real}} \;\approx\; \frac{J(N+\Delta N)-J(N)}{\Delta N}$$

——这才和全文 $MV$ 框架接上。风险也不止"灾难性遗忘 / 错误先验"：更常见的是**分布收窄**——真机 fine-tune 数据往往比 sim 窄得多（$D_{\mathrm{sim}} \to D_{\mathrm{real}}^{\mathrm{narrow}}$），微调后 policy 在目标切片上更好、鲁棒性却可能反而下降，等于**把 generalization 换成了 specialization**。

## 两条松动"两个给定分布"假设的新路线

上面四条轴共享一个隐含前提：**$p_{\mathrm{sim}}$ 与 $p_{\mathrm{real}}$ 是两个给定的分布**，你要做的是校准、覆盖、对齐或接力。而下面两条路线，恰恰在松动这个前提本身——它们不是"第五、第六种迁移技巧"，而是对整个问题的 reformulation。

### World model：不是取消 simulator，而是换掉 simulator 的来源

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)讨论过 world model 与 data utility 的关系。放进 sim-to-real 语境，先纠正一个定位误读：**world model 并不天然属于 sim-to-real**——两条路线的 causal direction 是不同的：

```
经典 sim-to-real：  sim dynamics → train policy → deploy real
World model route： real interaction → learn dynamics → imagine → optimize policy
```

需要说准：world model **并没有取消 simulator**，而是把 simulator 的角色从"手工指定的 physics model"换成"从交互数据里学出来的 predictive model"：

$$f_{\mathrm{hand\text{-}designed}} \;\longrightarrow\; f_{\mathrm{learned}}$$

真正被改变的是 **model source**。Dreamer（1912.01603）、TD-MPC2（2310.16828）体现了这条路。于是当**人工 simulator 的 model bias 大到不值得先去修它**时，world model 提供的是对 sim-to-real 问题本身的一种改写，而不是它下面的一个 transfer technique。

DayDreamer（2206.14176）常被误当成"sim 预训练 → real 微调"，但它的关键恰恰相反——world model 直接从真实交互中学习、在 latent imagination 里做 policy improvement，几乎不依赖手工 sim。要说清楚的是：**不依赖手工 physics simulator 不等于 model-free**，world model 学习仍吃满各种假设（representation、architecture、action space、reward、exploration、真机数据质量），只是把 inductive bias 从"显式 physics simulator"转移到"learned world model"里。

诚实的边界："用真实数据学 dynamics" **不等于天然优于仿真**。它把"手工建模成本"换成"真机采集 + 模型容量成本"；而在 contact-rich、long-tail、传感器噪声大的场景，学到的 model 常常在分布外给出**很自信、也很错的想象**，policy 会顺着错误规划走下去。所以它是"手工 sim"与"直接真机 RL"之间的**又一个 trade-off**，不是终局。

### Sim-and-real co-training：把"迁移"重述成 data mixture

Maddukuri et al.（RSS 2025，2503.24361）提出的 Sim-and-Real Co-Training 是一个务实方向。**论文实际报告的结果**是：在同一批训练里把 sim 与 real 数据集混合采样，在两个机器人平台、六个任务上观测到平均约 38%（精确 37.9%）的性能提升；它不做 sim→real 单向迁移，而是用一个 recipe 决定两者比例与调度。

**本文的解读（非论文证明的结论）**是把它进一步理解成一个 **data-mixture 问题**：一旦这样重述，DR 与真实数据不再是替代关系，而是同一份 sampling distribution 上 $T_R[p_{\mathrm{raw}}]$ 的两个来源（呼应上篇"$p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$"）。要写准的是：co-training 的**主要干预变量是 training mixture**——$p_{\mathrm{train}}=\lambda\, p_{\mathrm{sim}}+(1-\lambda)\, p_{\mathrm{real}}$——不是 simulator calibration，也不是 deployment-time adapter。但别简化成"只动数据分布这一根轴"：2026 年已有后续 mechanistic 分析（Lei et al., arXiv 2026，2604.13645）指出，mixture 的改变会诱发 **structured representation alignment 与 importance reweighting**。所以它是"以 mixture 为主要抓手、但效应跨多个维度"的一条路线，而非与前四条轴严格正交的第五根轴。

## 评估：你怎么知道自己把 gap 补好了？

一个危险的做法是只在 sim benchmark 上报性能——那衡量的是 policy 与你自己 simulator 之间的一致性，而不是与真实世界的一致性。更可信的评估至少应做到：

- 报告 **zero-shot transfer**（不做任何真机微调）到真实系统的性能，以及 **few-shot / N-shot** 之后的曲线；
- 用一组 **held-out physical systems**（不同标定、不同相机、不同接触面）来测，而不是只有"那台部署机器人"；
- 明确声明 sim 与 real 的**任务、initial-state、evaluation distribution 是否一致**——否则比较根本不公平（这正是 objective mismatch 要先对齐的原因）；
- 做**失败归因**：是哪一层 $\Delta_k$ 主导？不同层的敏感度和补救成本天差地别，归因错了预算就花错了地方。

顺着"sim 是真实世界的代理"这句，还有一个比"数值对齐"更本质的问题：**simulator 能不能正确预测"哪个 policy 更好"？** 考虑：

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

在 sim 上看起来 $A > B > C$，真机上却是 $B > C > A$。这时 simulator 不只是有 calibration error，而是**失去了 model-selection utility**——你会用它挑出一个最差的政策。所以**当 simulator 被用于 policy / model selection 时**，一个比绝对数值误差更直接的指标是排序相关性 $\rho_{\mathrm{rank}} = \mathrm{Spearman}\big(J_{\mathrm{sim}}(\pi_i),\ J_{\mathrm{real}}(\pi_i)\big)$。

但 Spearman 还不是最直接的。工程上真正怕的是"我相信 sim，结果 top-1 选错了"。所以还应加一个 policy-selection regret：

$$\pi_{\mathrm{sim}} = \operatorname*{arg\,max}_{\pi \in \Pi} J_{\mathrm{sim}}(\pi), \qquad R_{\mathrm{select}} = J_{\mathrm{real}}\big(\pi^{*}_{\mathrm{real}}\big) - J_{\mathrm{real}}\big(\pi_{\mathrm{sim}}\big)$$

比如 Spearman=0.95 却把 top-1 选错，对工程仍是灾难；反过来 Spearman=0.7 但 top-1 基本不出错，对"选一个能部署的 policy"反而够用。所以**当 simulator 用于 policy selection 时，rank correlation 与 top-1 / top-k selection regret 要一起看。** 这两者都是 **conditional metric**，只针对"用 sim 选 policy"这个用途：sim 还有大量不必排序整个 policy family 的用途（representation pretraining、exploration、curriculum、safety filtering、controller initialization、rare-event generation 等）。评价 simulator fidelity 时也不该只盯单个 policy，而应相对于**候选 policy family** 来评：$U_{\mathrm{sim}}=U(D_{\mathrm{sim}}\mid \Pi_{\mathrm{candidate}},\,p_{\mathrm{eval}}^{\mathrm{real}})$。

## 组合与决策，以及一个常被回避的问题

有了优先级，"什么时候用哪个"就不该是一条固定流水线，而是一张查询表。真实项目常常几个条件同时成立，所以更有用的是 **gap × 可建模性 × 真机预算** 的决策矩阵（"Real data"列直接对应上面的预算向量 $B$）：

| Gap | 可参数化 / 可辨识？ | Real data | 推荐 |
| --- | --- | ---: | --- |
| dynamics bias | 高 | 少 | SI |
| dynamics uncertainty | 中 | 少 | DR（或 posterior-guided DR） |
| dynamics residual | 低（但有结构） | 中 | Residual learning |
| visual appearance | 高 | 无 / 少 | DA / DR |
| actuator latency | 高 | 少 | SI + DR |
| unknown long-tail，可模拟 | 低 | 少 | targeted simulation / DR |
| unknown long-tail，sim 生成不可信 | 低 | 中 | real data |
| model class 不确定 | 低 | 多 | learned world model（若 real 稀缺则先 physics prior + residual / DR） |
| mixed | mixed | mixed | co-training candidate（需先验证正迁移条件：mixture ratio 与 cross-domain alignment） |

倒数第二行：光"model unknown"推不出 world model，真正的判据是 **model uncertainty × real-data budget**——模型类不确定**且**真实交互充足时 learned world model 才是合理候选；真实数据稀缺则先保 physics prior + residual / DR。最后一行同理："co-training 兜底"和全文 allocation 立场冲突——当 sim 质量差、real 很少、两边 action space / task semantics 不一致时，co-training 完全可能负迁移，须先验证正迁移条件。

常见组合是 **SI → DR → DA → co-training / fine-tune**：SI 校一个"80% 对"的 sim，DR 在"说不清但可枚举"的方向撑开族，DA 处理视觉域差，最后用少量真机数据收尾。**但箭头不是固定 workflow，只是示意组合**——实际顺序由当前主导 gap 与各干预的边际效用决定：真机充足时先做 SI 未必划算；视觉主导时 DA 该提前；SI 数据极少时先粗 DR、拿到能跑的 policy、再回头校准往往更合理。

顺着这个逻辑，就能回答一个整篇几乎都在回避、但框架本身允许的反问：**什么时候最优解其实是"不做 sim-to-real"？**

- **真机数据已经便宜到 $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}^{\mathrm{effective}}$ 时**——这里 $C_{\mathrm{real}}^{\mathrm{effective}}$ 是**有效真机成本**（不只是采集：还含安全、operator、reset、硬件磨损、失败恢复、deployment 多样性与可复现性）。只有把这些都算进去，"4 小时真机 vs 20 小时 sim"这种比较才不会在不同实验室里完全失真。
- **仿真器 model class 本身就差（$\Delta_{\mathrm{model}}$ 主导且难以参数化）** 时——软体、流体、复杂接触——修 sim 的成本高到边际效用极低，不如走 world model 或真机数据学习。
- **部署分布非常固定** 时——根本不需要大规模 DR 去覆盖一整个族，少量 targeted real fine-tuning 往往更划算。

能大方承认"有时最优解是不做 sim-to-real"，恰恰是 allocation framing 应有的样子：**它不站"仿真"这个队，只站"下一单位预算换回最多真实性能"这个队。**

## 这意味着什么？：一个闭环，而不是一个开关

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)的核心句是 evaluation-aware distribution allocation：有限预算下，把每一单位花到 marginal data value 最高的地方。套回 sim-to-real，会得到一个自然推论——**仿真数据的 utility 从来不是 simulator 的内部属性，而是相对于真实 evaluation distribution 的属性：**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

这解释了一个常见挫败：为什么"堆更多 sim 数据"有时没用。但要讲准：增加 sim 数据也可能带来更广 coverage、更多样的物体、更高 rare-event 频率，所以这句判断是**有条件的**——

> **当主要瓶颈恰好是 simulator 与真实 evaluation distribution 之间的 support / fidelity mismatch 时，单纯增加同分布 simulation samples 的边际收益会快速下降；此时加 $N$ 主要提高的是采样密度，而不能自动创造缺失的 support 或修正 model bias。**

它改善的是 density，真正缺的却是 support 与 $\Delta_{\mathrm{model}}$ 的保真度。与其问"我的 sim 有多好"，不如问开头那句：**"我的 sim 在哪些 evaluation-relevant 方向上接近真实、在哪些方向上差得远？差得远的那些，敏感度有多高、用哪种预算压它最便宜？"**

把这条线走完，sim-to-real 就不再是"能不能迁移成功"的开关，而是这样一条闭环链路：

$$\boxed{\text{mismatch} \rightarrow \text{sensitivity} \rightarrow \text{intervention} \rightarrow \text{marginal utility} \rightarrow \text{budget allocation} \rightarrow \text{real evaluation}}$$

但这条链不是"从 simulator 里解析跑一遍"就出答案，而是一个 **sequential empirical decision framework**：敏感度与边际收益都要靠小步实验在真实评估上估出来，一轮估完再决定下一份预算投到哪。一句话收束：**sim-to-real 不是一个 transfer 技巧，而是在 model fidelity、训练多样性、表示对齐、真机交互与工程成本之间做 constrained、可迭代估计的 allocation 问题。** 这和下篇"机器人数据 scaling 是一个 sequential data allocation 问题"是同一件事——只不过这一次，分配发生在仿真与真实之间。

---

## 参考文献

正文涉及的主要工作如下（均可通过 arXiv ID 检索）：

- Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World — Tobin et al., IROS 2017, arXiv:1703.06907
- Sim-to-Real Transfer of Robotic Control with Dynamics Randomization — Peng et al., ICRA 2018, arXiv:1710.06537
- Sim-to-Real: Learning Agile Locomotion For Quadruped Robots — Tan et al., RSS 2018, arXiv:1804.10332
- Learning Dexterous In-Hand Manipulation — Akkaya et al. (OpenAI), 2019, arXiv:1808.00177
- Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks — James et al., CVPR 2019, arXiv:1812.07252
- DiffTaichi: Differentiable Programming for Physical Simulation — Hu et al., ICLR 2020, arXiv:1910.00935
- Interactive Differentiable Simulation — Heiden et al., arXiv 2019, arXiv:1905.10706（同组另有 NeuralSim: Augmenting Differentiable Simulators with Neural Networks, ICRA 2021，是另一篇）
- Residual Physics Learning and System Identification for Sim-to-real Transfer of Policies on Buoyancy Assisted Legged Robots — Sontakke et al., 2023, arXiv:2303.09597
- Sim-to-Real of Soft Robots with Learned Residual Physics — Gao et al., IEEE RA-L 2024, pp. 8523–8530, arXiv:2402.01086
- Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) — Hafner et al., 2019, arXiv:1912.01603
- DayDreamer: World Models for Physical Robot Learning — Hafner et al., CoRL 2022, arXiv:2206.14176
- TD-MPC2: Scalable, Robust World Models for Continuous Control — Hansen et al., ICLR 2024, arXiv:2310.16828
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., RSS 2025, arXiv:2503.24361
- A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies — Lei et al. (Yu Lei, Minghuan Liu, Abhiram Maddukuri, Zhenyu Jiang, Yuke Zhu), arXiv 2026, arXiv:2604.13645

需要说明的是，sim-to-real 目前尚不存在一份公认的"哪种方法更强"的跨任务定量对照——不同任务、硬件与 fidelity 上限下结论可能完全颠倒；上述工作更多是"这类 gap 用这个方法可行"的样本，而非可跨场景外推的排序。本文中关于四个干预维度的分解、误差预算的 constrained-allocation 形式化、$S_k$ 与 $MV$ 的定义都是 **conceptual framework 与作者解读**：$S_k$、$MV$ 是需要通过 sensitivity experiments / ablation / 小规模真实评估估计的 decision statistics，**不是**能从 simulator 解析求出的量；把 co-training 读作 data-mixture、把 world model 读作 model-source replacement，同样不是受控实验证明的结论。

---

*本篇是"具身智能的数据问题"上下篇的续篇：上篇讲数据来源与接口、下篇讲数据 scaling 框架；这一篇把镜头拉到 sim-to-real，把它从"一堆迁移技巧"重述成一个带经验边际效用的闭环分配问题，好接回下篇 sequential data allocation 的主线。*
