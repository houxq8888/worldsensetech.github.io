---
title: '具身智能 Sim-to-Real 方法论深潜：把"从仿真到真实"当成一次误差预算分配'
slug: "2026-09-10-sim-to-real-methodology"
date: 2026-09-10
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "Sim-to-Real", "Domain Randomization", "System Identification", "可微仿真", "Residual Physics", "世界模型", "Domain Adaptation", "机器人数据"]
description: 'sim-to-real 不是单一迁移技巧、而是闭环的资源分配。本文把 reality gap 重述成 policy-conditioned 的多源 mismatch、以 intervention sensitivity 与经验边际效用把误差预算分配写成可估计、可迭代优化的决策框架、厘清 SI / DR / DA / real-world fine-tuning 四条 intervention axes 的机制与失效边界、并讨论 world model、residual physics、co-training 与"何时最优解是不做 sim-to-real"。'
toc: true
related_articles:
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
  - 2026-08-25-dreamer-explained
---

> 接[数据问题上篇](/zh/articles/2026-09-08-data-and-training-recipes/)与[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)。上篇把 sim-to-real 粗分四类工具、那只是 taxonomy，这一篇真正要回答的问题是——

> **当仿真数据在若干 evaluation-relevant 方向上离真实差得远时、下一单位预算（工程时间 / 算力 / 机器人小时）该花在哪条杠杆上：校准仿真器、扩大训练分布、对齐表示、还是采真机数据？**

乍看是工程直觉、其实是闭环资源分配：几笔不能互换的预算下、要不断问"下一块钱花在哪、换回最多真实性能"。项目里最卡人的不是"不知道有这些方法"、而是"这类 gap 管不管用、花哪种预算"。先给"误差预算"降一档歧义：它**不是**给每个误差项预分固定额度（$\Delta J=\sum_k \Delta_k$、逐项发钱）、而是花在**干预动作**上、通过 sequential allocation 逐步压低最有价值的 mismatch。

## Reality Gap：不是一个标量，而是一个 policy-conditioned 的 mismatch

Sim-to-real 常被叙述成"训练 policy 从仿真迁移到真实"。更严格的起点是**两个分布**：同一条 $\pi$ 与环境交互各自诱导 $p_{\mathrm{sim}}^{\pi}(\tau)$ 与 $p_{\mathrm{real}}^{\pi}(\tau)$、一般不等：

$$p_{\mathrm{sim}}^{\pi}(\tau) \;\neq\; p_{\mathrm{real}}^{\pi}(\tau)$$

轨迹分布本身是 **policy-induced**、随 $\pi$ 而变、不是环境固有属性。真正关心的不是分布差、而是它在任务上**表现的后果**——同一 $\pi$ 两边的性能差：

$$\boxed{\;\delta_J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)\;}$$

叫 **transfer delta**、保留符号：$J$ 若是 success rate、真实反而更好（sim 更保守、或噪声更狠）时 $\delta_J$ 会为正、直觉上不该叫 gap。故另把幅度

$$G_J(\pi) \;=\; \big|\,\delta_J(\pi)\,\big|$$

单独叫 **performance gap**——下文谈敏感度用这个语义、不与符号纠缠。

**distribution mismatch ≠ performance gap**：$p_{\mathrm{sim}}^{\pi} \neq p_{\mathrm{real}}^{\pi}$ 不自动意味着 $\delta_J$ 很大、不同 policy 对分布差的敏感度完全不同——只依赖粗粒度几何的 policy 换掉摩擦建模性能几乎不变；依赖高频力反馈的精细装配里同样分布差可能致命。

$\delta_J(\pi)$ 是**任务相关、policy 相关的可观测后果**。严格写要把 **mechanism 与 induced distribution 分开**：环境的 transition / observation / actuation kernel 记作机制 $M_{\mathrm{sim}}, M_{\mathrm{real}}$、给定 $\pi$ 下**诱导**出 $p_{\mathrm{sim}}^{\pi}(\tau),\ p_{\mathrm{real}}^{\pi}(\tau)$——概率分布**不是 mechanism**。gap 更干净的写法：

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ \mathcal{E};\ M_{\mathrm{sim}},\ M_{\mathrm{real}}\big)$$

（$p_{\mathrm{sim}}^{\pi}, p_{\mathrm{real}}^{\pi}$ 单列进四元组会与 $\pi$ 语义重复——它们是 mechanism 在 $\pi$ 下的派生量。）逻辑是 **mechanism → trajectory distribution → performance**：$\mathcal{E}$ 是 evaluation 假设集合（initial-state / horizon / reward / constraints）、"仿真器很真"从不是有意义的评价：同一 $M_{\mathrm{sim}}$ 对 position control policy 可能 gap 很小、对 force-sensitive manipulation policy 可能 gap 巨大。**reality gap 是这个四元组的属性、不是仿真器的属性**。

### gap 到底在哪里：reality mismatch 与 task-specification mismatch

第一步是把多源 gap 拆开——有**两大类来源**、不能全塞进"reality"一词下面：

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

两类来源不同、不要简单相加：reality mismatch 是"仿真与真实不是同一世界"、task-specification mismatch 是"优化目标与部署目标不是同一任务"。**观测与状态估计值得单独成层**——机器人真正执行 $a_t = \pi(o_t),\ o_t = h(x_t) + \epsilon$；camera 标定误差 / depth bias / 遮挡 / proprioception drift / 力传感器偏置 / state estimator 时延**不是"画面不一样"、而是让 policy 看到的 state 与 simulator 假设可用的 state 不一致**——manipulation / locomotion 里这类"状态估计 gap"往往比外观 gap 更伤 performance。

**Initial-state / environment mismatch** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$——reset 分布 / 场景布局对不上；**Objective / task shift** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$——仿真只要 grasp success、真实还要 collision avoidance。**initial-state 要小心**：若 sim 与 real **都能产生同样 $s_0$**、只是训练没覆盖（sim 生成红/蓝杯、部署全蓝杯、policy 只训红杯），这是**一般 train-test shift、不是 reality gap**；只有 sim-real reset / scene **实现本身**对不上（sim 恒 5 cm、real ±20 cm）才是 environment mismatch——**不应无条件归入 reality gap**。Objective shift 则**已是 objective mismatch 而非 reality gap**：物理再准、reward / 约束对不上就不是"迁移失败"、而是"评测的根本不是同一任务"；下文默认 objective 已对齐、objective mismatch 靠 reward shaping / 约束建模单独处理。

## 把"误差预算分配"写成一个可估计、可迭代优化的决策框架

拆完来源、给开头直觉一个数学落点。写法是 **conceptual、不是严格定理**：误差项强烈交互——sim 假设 proprioception 精确、真实有 latency、单看都不致命、叠加可能让 controller 失稳、故更稳妥的写法是先承认未知的耦合函数 $F$：

$$\boxed{\;\delta_J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}}\big)\;}$$

**这一版把 $\Delta_{\mathrm{opt}}$（优化 / 学习误差）从 reality gap 拿掉**：层级不同——同一固定 policy、仿真观测动力学都准但 RL 没训好、$\delta_J$ 很小、policy 却很差、"policy 没学好" ≠ "sim-to-real gap 大"；塞进 $F$ 会把两件事搅一起、应分开成**两个诊断量**：

$$\underbrace{J_{\mathrm{real}}(\pi_{\mathrm{train}}) - J_{\mathrm{sim}}(\pi_{\mathrm{train}})}_{\text{transfer delta } \delta_J}\qquad \underbrace{J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})}_{\text{real-domain learning gap}}$$

**这两个量不能无条件相加叫 deployment loss**：$\delta_J$ signed、两项 baseline 也不同、相加既非 deployment loss 也非统一 regret；它们是**不同层级的误差来源**、分别诊断、分别归因。

只在工作点附近做工程归因时、才把 $F$ 局部近似成加权和 $\delta_J \approx \sum_k w_k \Delta_k$——**这一层只是局部归因 heuristic、不是全文核心公式**：$w_k$ 是 surrogate decomposition 的 coefficient、与下式 $\hat S_k^{\mathrm{int}}$ 只在特定局部参数化下才近似对应、不必同时保留两套"敏感度"叙事。真正用来做 decision 的是每类 mismatch 挑一个 **intervention 变量** $\xi_k$ 后测得的 **intervention sensitivity**：

$$\hat S_k^{\mathrm{int}} \;\approx\; \frac{J_{\mathrm{real}}\big(\pi;\operatorname{do}(\xi_k+\delta)\big) - J_{\mathrm{real}}\big(\pi;\operatorname{do}(\xi_k)\big)}{\delta}$$

**关键澄清**：$\xi_k$ **不是"真实 gap 的天然坐标"、而是为 sensitivity experiment 人为定义的 intervention variable**——latency / friction / appearance 可直接扰动、camera calibration error / contact-model mismatch / state-estimation error 很难把 $\xi_k$ 在真实世界连续拨动；$\operatorname{do}(\cdot)$ 提醒读者"这是实验性介入、不是对固有量求导"。**$\hat S_k^{\mathrm{int}}$ 只是诊断阶段的辅助统计量**、**核心决策量是下节的 $MV(m\mid b,\pi)$**、叙事是 **diagnosis → intervention → empirical marginal utility → allocation**。再降一档：$\Delta_{\mathrm{model}}$ 与 $\Delta_{\mathrm{ctrl}}$ 甚至可能不可辨识地互相补偿（actuator gain 错、policy 靠 command distribution 补回）、两者 **都不是 simulator 解析可求的物理量、而是 sensitivity experiments / ablation / 小规模真实评估估出的 decision statistics**。

### 真正的"分配"：把钱花在干预动作上，而不是在方法里挑一个

到这里还只是"选方法"。要让预算分配名副其实、预算得**连续地**分到每条干预轴：总预算拆成向量 $b=(b_1,\dots,b_K)$、$b_k$ 是花在干预 $k$ 上的量——$b_{\mathrm{SI}}=2\text{h}$、$b_{\mathrm{DR}}=10^6$ 步 sim、$b_{\mathrm{real}}=4\text{h}$ 真机——而不是"用不用 SI"这种 0/1 选择。目标是最大化真实性能：

$$\max_{b}\quad J_{\mathrm{real}}\big(\pi_b\big)$$

机器人项目里的预算**不是同一种货币**：GPU 近乎无限但真机机时极少、有机器时间却没工程人力、故正确写法是**多预算约束**、不折成标量 $B$：

$$\begin{aligned}
C_{\mathrm{real}}(b) &\le B_{\mathrm{real}}\\
C_{\mathrm{compute}}(b) &\le B_{\mathrm{compute}}\\
C_{\mathrm{eng}}(b) &\le B_{\mathrm{eng}}
\end{aligned}$$

预算是分向量后、决策变量就该从"gap"换成"干预动作"：工程师买不到"$\Delta_{\mathrm{model}}$ 的 2 个百分点"、能买到 30 分钟 SI / $10^6$ 步 sim / 100 条真机轨迹 / 一次 camera calibration / 一个 residual model。对干预 $m$ 定义边际效用更自然——**干预不直接改 $\Delta_k$、而是通过训练过程改变 policy**：

$$\boxed{\;\pi_{b+m} \;=\; \operatorname{Train}\big(D_{\mathrm{sim}},\ D_{\mathrm{real}};\ m\big)\;}$$

于是"下一块钱花在哪"是以干预为变量、需要在真实世界里逐步估计的量。**本文的核心决策公式是 $MV$ 而不是 $w_k$ 或 $\hat S_k^{\mathrm{int}}$**：

$$\boxed{\;MV(m \mid b, \pi) \;=\; \frac{\mathbb{E}\big[\,J_{\mathrm{real}}(\pi_{b+m}) - J_{\mathrm{real}}(\pi_{b})\,\big]}{C(m)}, \qquad m^{*} = \operatorname*{arg\,max}_{m} MV(m \mid b, \pi)\;}$$

这个 ratio 无法从 simulator 解析求得、只能用 pilot / ablation / few-shot real evaluation **sequential 地估**。两条 caveat 一起写：**(i) 不确定性**——真机 $\Delta J$ 噪声极大、每种 intervention 往往只跑几次 pilot，例如 A = +3 ± 0.5、B = +5 ± 5，只看期望会选 B、但 B 可能只是噪声；allocation 还应看 CI / posterior / **lower confidence bound (LCB)**、否则高方差 intervention 会因一次偶然成功被错误优先。**(ii) 非线性成本**——$MV$ 是**局部决策统计量、真实性能与预算并不线性**：SI 一次 5 小时工程可让之后每次训练受益（fixed setup cost）、DR 每加 $10^6$ 步逐步饱和（diminishing returns）、fine-tune 数据少时几乎无 improvement、过门槛才起量（threshold effects）。

**不同干预的 $MV$ 也不是固定常数**：$MV_i = MV_i(b_{1:i-1},\ \pi_b,\ D_{\mathrm{real}})$——先 SI 缩窄 uncertainty set、DR 的 $MV$ 下降；先 DR 起点更 robust、fine-tune 的 $MV$ 上升；反过来先 fine-tune 再补 DR 有时更保守甚至冲突。**intervention 之间同时存在 complementarity、substitutability 与 occasional conflict**、故这不是"一次性 knapsack"、而是 **resource-constrained sequential experimentation / adaptive allocation**（接近 adaptive experimental design、但因没有严格 arm / stationary reward / regret 证明、**别写成 bandit algorithm**）。

还有一层更隐蔽的反馈：**intervention 不只压低 gap、还会改变 policy、进而改变 policy 对 gap 的敏感度本身**——$S_k^{\mathrm{int}} = S_k^{\mathrm{int}}(\pi)$、$\pi = \pi(m)$，故 $S_k^{\mathrm{new}} \neq S_k^{\mathrm{old}}$、闭环并非单向：

```
estimate mismatch → estimate sensitivity → intervention
       ↑                                          ↓
   re-estimate  ←  sensitivity changes  ←  policy changes
```

**这张 feedback loop 比任何新公式都更贴合本文 allocation thesis**：sim-to-real 不是一次解完的优化、是一轮做完重新估一轮的 sequential experiment。

把每条干预对应到主要压缩项与主要预算、成本按**预算向量**拆开（SI 大头是真机激励 + 参数估计 + 仪器 + 仿真器工程 + 优化算力）：

| Intervention | 主要压缩项 | 主要预算 |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{eng}}$ + $C_{\mathrm{compute}}$ + 少量 $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | $C_{\mathrm{compute}}$（样本效率） |
| Residual physics | $\Delta_{\mathrm{model}}$（残差部分） | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$（appearance 子集） | $C_{\mathrm{real}}$（未标注数据）+ $C_{\mathrm{compute}}$ |
| Real-world fine-tuning | adaptation mechanism（同时改 transfer delta 与 real-domain learning gap） | $C_{\mathrm{real}}$（磨损 / 安全） |
| World model | 改变 model source | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Sim-and-real co-training | 改变 $p_{\mathrm{train}}$（$\Delta_{\mathrm{dist}}$ 为主） | 混合数据（$C_{\mathrm{real}}+C_{\mathrm{compute}}$） |

有了这套写法、全文就不是"四种方法谁更好"、而是闭环：定位主导 $\Delta_k$、sensitivity 判断多重要、在 $MV$ 最高的干预上投一份预算、真实评估量回报、再决定下一份——接回下篇 evaluation-aware distribution allocation、只是这次分配的是仿真与真实间的工程预算。

## 四个 intervention lenses（更准确说，四个相对独立的分析维度）

框架有了、再逐条看工具。SI、DR、DA、real-world fine-tuning **不是同一抽象层级的并列类别**——SI 是 model calibration、DR 是 training distribution manipulation、DA 是 representation alignment、fine-tuning 是 optimization strategy，并排成"四类方法"会误导人四选一、其实是**四个相对独立的 intervention lens**、可组合（**本文的 analytical decomposition、非领域公认 ontology**）：

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

补一句："$\times$" 是**组合空间**、不是数学正交——DR 触及 Model / Observation / Distribution、DA 可发生在 input / feature / latent / policy / output、"DA = Representation 轴"只是本文的一层 abstraction。

**选工具的标准不是"systematic → SI、random → DR"**——口诀当记忆法没错、但 SI 真正做的是**在 identification objective 下拟合参数**：

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; \mathcal{L}_{\mathrm{ID}}\big(D_{\mathrm{real}},\ f_{\mathrm{sim}}(\cdot\,;\,\phi)\big)$$

$\mathcal{L}_{\mathrm{ID}}$ 可以是 trajectory prediction / one-step transition error / force-torque residual / state-estimation residual / likelihood 或 Bayesian posterior——**很多经典 SI 根本不做 trajectory distribution matching、只最小化预测误差**。它解决**可辨识、可参数化的 model mismatch**、不是"凡是 systematic 都归它"（actuator gain / latency / friction / mass 都可能是随机过程而非确定性 bias）；同理 DR 解决**能被训练分布表示出来的 uncertainty**。更有用的划分是"**点估计 → 后验 → 鲁棒随机化**"这条连续谱：

| mismatch 的性质 | 更自然的工具 |
| --- | --- |
| 可参数化 + 可辨识 | System Identification（point estimate $\hat\phi$） |
| 可参数化但只能给出不确定性 | Bayesian / posterior SI → posterior-guided DR |
| 可参数化但难辨识 / uncertainty 大 | Domain Randomization |
| 难以由低维物理参数充分表达、但有结构化 residual | Residual learning |
| observation / appearance mismatch | Domain Adaptation |
| policy 在目标域仍有 systematic residual | Fine-tuning |

关键：**"不能精确辨识"与"完全不知道"不是一回事**——拿到后验 $p(\phi \mid D_{\mathrm{real}})$ 后、最自然的动作不是"干脆 uniform DR"、而是 $\phi \sim p(\phi \mid D_{\mathrm{real}})$ 做 **posterior-guided randomization**、把 SI 与 DR 缝成连续谱。

### Axis A — Model：system identification、可微仿真与 residual physics

这条轴处理 $\Delta_{\mathrm{model}}$、内部三个**不同层次**常被"可微仿真 = 更强 SI"打包：

$$x_{t+1} \;=\; \underbrace{f_{\mathrm{physics}}(x_t,a_t;\phi)}_{\text{可参数化的物理}} \;+\; \underbrace{r_\theta(x_t,a_t)}_{\text{残差}} \;+\; \epsilon_t$$

- **可微仿真**回答"怎么优化模型"——提供 $\partial f/\partial\phi$ 这个 optimization interface；DiffTaichi（Hu et al., ICLR 2020，1910.00935）、Interactive Differentiable Simulation（Heiden et al., arXiv 2019，1905.10706）让参数估计可梯度化。
- **System identification** 回答"优化什么参数"——即 $\phi$。经典 SI 扫参数、拟合轨迹；可微仿真把 $\phi$ 像权重一样反传更新。真实工作流常是 **real → identify → sim → train → real**、更准确的名字是 **real-to-sim-to-real**。
- **Residual physics** 回答"模型没解释掉的部分由谁解释"——不硬校准 $\phi$、让网络学 $r_\theta$ 补差。

$r_\theta$ 只是**统一记号**：实际 residual 未必直接加在 $f$ 上、可定义在状态转移 / force / acceleration / contact impulse / deformation field 或其他 simulator latent 上。

有个决定成败、最容易被"可微"二字掩盖的点：**可微性解决 optimization interface、不解决 model class correctness**。若 simulator 的 contact model 根本没表达某种真实现象、再精确的梯度也只给你"错误模型下的最优参数"。可微仿真把 $\phi$ 估得更准、却不替你写对 $f_{\mathrm{physics}}$ 的函数形式——写不对的部分只能交给残差、或放弃"先建可信 sim"。**常被忽略的边界**：碰撞 / 摩擦 / 接触模式切换往往是 **nonsmooth / piecewise-smooth**——即使 $\partial f/\partial\phi$ 存在或工具能给、也不保证梯度稳定、长时域 rollout 不爆炸、contact mode 切换处梯度有意义、或优于 derivative-free optimization；实际系统常需 smoothing / relaxation / 专门 contact treatment。

SI 还有两个更细但实在的坑。**其一，$p_{\mathrm{real}}(\tau)$ 几乎不可直接访问**、只有有限条真机轨迹、$\arg\min_\phi$ 实际跑在经验估计 $\hat\phi=\arg\min_\phi \sum_i \ell(\tau_i^{\mathrm{sim}}(\phi),\tau_i^{\mathrm{real}})$ 上。**其二，参数存在 ≠ 可辨识**——identifiability 还依赖 excitation 与 sensor observability、质量 / 阻尼 / 刚度在某些激励下会产生几乎相同的可观测轨迹、无法独立估出。

Residual physics 的边界也要收窄：**不是"物理函数形式错了"就天然适用**。甜蜜点是 residual 在目标分布上相对受限（$\|r_\theta\| \ll \|f_{\mathrm{physics}}\|$）、但真正关键的不是残差要"小"、而是 $f_{\mathrm{physics}}$ 是否仍提供**有用的结构性归纳偏置**（inductive bias / state representation / constraints / extrapolation prior）；若 $f_{\mathrm{physics}}$ 完全错、残差独自承担整个 dynamics、不如直接学一个 model。软体（Gao et al., RA-L 2024，2402.01086）、浮力腿式（Sontakke et al., 2023，2303.09597）这类"主干物理还算数、局部摩擦 / 接触 / 形变有稳定残差"的场景最好用。**再加 caveat**：$r_\theta$ **并不天然等于"缺失物理"**——unrestricted additive residual 会吞下大量 model error（sensor bias / actuator error / timing / calibration / reward mismatch / policy-induced artifact）成 **error sponge**：训练分布内拟合好、一到 OOD 就失稳；故 residual 要配结构约束（低维 / 稀疏 / 力或加速度尺度 / 物理先验 / 只在特定 contact regime 生效）、否则"补差"就变成"补所有不该补的东西"。

### Axis B — Data distribution：domain randomization 及其家族

这条轴不追求逼近"最准"的 $p_{\mathrm{real}}$、而是让 policy 对一族参数 $\{\phi\}$ 都稳健：训练时随机化物理 / 视觉 / 初始状态 / delay、只要真实落在这族里就能顶住。Tobin（1703.06907）用纯视觉随机化把 sim 抓取检测搬到真机；Peng（1710.06537）把随机化推进到 dynamics；OpenAI in-hand manipulation（Akkaya et al., 1808.00177）几乎把 DR 推到极致——**不靠精确校准、靠"随机化范围足够宽"吸收差异**。

一句常被写歪的直觉：**DR 不是"隐式 ensemble"**——训练的是**单个**共享 policy $\pi_\theta$、目标大致是

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

通常 ensemble 是多个 $\{\pi_1,\dots,\pi_K\}$。更准确：**DR 是对一族环境模型做 population-level 优化**——直觉像 ensemble training、结果却是共享 policy、不是集成。上式是 **risk-neutral average-case DR 的 baseline abstraction**；robust / adversarial DR 还可换成 $\max_\theta \min_{\phi\in\Phi} J(\pi_\theta;\phi)$、CVaR $\max_\theta \operatorname{CVaR}_\alpha(J)$ 或其他风险敏感形式——接住下文"robust randomization"。DR 有效条件是**两件事叠在一起**。**Proxy**：真实参数分布要落在 DR 支撑内、被高密度训练到、粗略记作 $p_{\mathrm{real}}(\phi) \ll p_{\mathrm{DR}}(\phi)$——$\phi_{\mathrm{real}}$ 落在 support 里但正好在极低概率尾部照样会挂。**但 proxy 还有更根本的假设**：**隐含 real dynamics 可被同一 $\phi$-parameterization 表达**；若 simulator model class 不包含真实现象（无某种 contact mode、可形变性未编入 $\phi$、执行器动力学不存在这个 parameter、observation failure 不是简单 noise）、连 $\phi_{\mathrm{real}}$ 都无法定义、support **从根上不成立**——这时不是"DR 覆盖不够宽"、而是 model-class uncertainty。**主结论**：真正决定 transfer 的不是 parameter-space marginal support、而是 policy 在 evaluation 下实际访问的 **state-action / contact occupancy $d_{\mathrm{real}}^{\pi}(s,a)$** 与训练分布诱导的 occupancy 是否足够 overlap——**parameter coverage 是必要 proxy、非 deployment coverage 的充分条件**；friction / mass / latency 各自 range 都覆盖了、policy 仍可能进入 simulator 从未见的 contact mode。

再往下一层：**DR 不是选 scalar range、而是在设计 joint distribution**。$p(\phi_1,\phi_2) \neq p(\phi_1)p(\phi_2)$ 才是常态——payload ↑ 联动 actuator regime、temperature ↑ 联动 motor resistance / friction / battery。独立 uniform DR 只是方便的 baseline、不是"真实 uncertainty set"的自然表示。关键回到分配：**randomization 分布要对齐 evaluation 分布与 objective**——过宽或与任务无关会拉低样本效率、迫使 policy 在冲突 dynamics 上折中；但 robust / adversarial 设定下适当扩大 uncertainty set 反而更稳。**"越宽越保守"并非普遍规律、shape 与对齐才是。**

"Adaptive / Automatic DR" 也不是单一方法而是家族：curriculum over randomization / adversarial DR（采样最能击垮当前 policy 的参数）/ automatic DR（依训练表现或真实反馈自适应调整）/ posterior-based sampling / performance-driven range adaptation——机制各异（扩大 / 缩小 / 专门找 hard domain）、共同点是**避免一开始就 over-randomize**。

### Axis C — Observation / Representation：domain adaptation 与观测翻译

这条轴处理 $\Delta_{\mathrm{obs}}$、既不校准物理也不随机化、而在**观测/表示层**对齐 sim 与 real。先声明：**"Representation" 是本文 abstraction、不是 DA 标准定义**——DA 实际可发生在 input / feature / latent / output / policy / dynamics model 六层、"DA = Representation 轴"只是为了与 Model/Data/Optimization 三轴对齐做的命名。具体机制包括 feature-level adapter、image translation（GAN / 扩散）、randomized-to-canonical 的 RCAN（James et al., CVPR 2019，1812.07252）——**RCAN 更适合当"input-level canonicalization / sim-to-sim adaptation"的例子、不是 DA 通用代表**：把随机化过的 sim 图翻回近似 canonical 的干净图喂下游 policy、做 sim→sim 对齐、顺带把 Axis B 的 DR 与这条轴缝起来；处理"物理差不多、但看起来完全不像"的 gap。两条边界要写准：**其一，DA 只是 observation mismatch 子集**——camera intrinsics/extrinsics、temporal sync、sensor bias、depth distortion、state estimation 更适合 calibration / SI / sensor modeling、否则读者会形成"observation gap → DA"的新错误口诀。**其二，对 policy learning、domain invariance 本身不是目标、task-relevant invariance 才是**——只对齐 $z_{\mathrm{sim}}\approx z_{\mathrm{real}}$ 不够、理想是保持 $I(z;y_{\mathrm{task}})$ 高的同时压低 $D(z_{\mathrm{sim}},z_{\mathrm{real}})$、与上一节"过宽 DR 抹掉任务信号"同一件事。

### Axis D — Optimization / adaptation：真机微调

这条轴**不是一类 mismatch、而是 adaptation operator**：直接在目标域上继续优化 policy。**不是"最后一步"**——既可作前三条轴补完的收尾、也可作**早期诊断或快速 adaptation 手段**。承接前文区分、fine-tuning **可能同时改变 transfer delta 与 real-domain learning gap**、但两者仍分别诊断；两个 regime 成本结构完全不同：

- **Offline / imitation：** $D_{\mathrm{real}} \to \theta$、主要成本是**数据采集**（一次性、可离线、可复用）。
- **Online RL：** $\pi_\theta \to a \to$ 真实 transition $\to \theta'$、主要成本是**交互 + 安全 + 硬件磨损 + 探索**（每步都在消耗物理资源）。

所以比较方法不能只看最终 success rate、还要看**达到目标性能所需的真机交互预算**。常被引用的粗略指标：

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{或}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

但只是**粗略指标**：依赖 baseline（5%→10% 与 80%→85% 都算 +5%、意义完全不同）、也不是真正的 marginal efficiency。真正该看 learning curve / 达到目标所需 real samples / AULC / 每 100 条轨迹的边际收益

$$MV_{\mathrm{real}} \;\approx\; \frac{J(N+\Delta N)-J(N)}{\Delta N}$$

——这才与全文 $MV$ 框架接上。风险不止"灾难性遗忘 / 错误先验"：更常见是**分布收窄**——真机数据比 sim 窄得多（$D_{\mathrm{sim}} \to D_{\mathrm{real}}^{\mathrm{narrow}}$）、微调后 policy 在目标切片上更好、鲁棒性却可能反降、**generalization 换成了 specialization**。故 $MV_{\mathrm{real}}(N)$ **不保证始终为正**：前 100 条大涨、100–500 快速衰减、再往后可能过拟合甚至倒退——**fine-tuning 本身也可能进入负边际收益区间**、回到 allocation 主张："真机数据不是越多越好、是看当前 marginal value"。

## 两条松动"两个给定分布"假设的新路线

上面四条轴共享一个隐含前提：**$p_{\mathrm{sim}}$ 与 $p_{\mathrm{real}}$ 是两个给定分布**、你要做的是校准 / 覆盖 / 对齐 / 接力。下面两条路线恰在松动这个前提——不是"第五、第六种迁移技巧"、而是对整个问题的 reformulation。

### World model：不是取消 simulator，而是换掉 simulator 的来源

**本文 lens disclaimer**：在本文 allocation taxonomy 里、我把 world model 看成"model source replacement"的 reformulation——**这是本文的分析角度、不是 world model 的标准定义**。严格说 world model 可学自真实 / sim / sim+real / video / offline data / latent dynamics / observation model / reward model / multi-modal predictive model 等、外延比本节宽得多；本节只挑"相对 physics-sim 换掉了 model 来源"这个切面。

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)讨论过 world model 与 data utility 的关系。放进 sim-to-real 语境先纠正定位误读：**world model 并不天然属于 sim-to-real**——两条路线 causal direction 不同：

```
Physics-sim route：  hand-designed dynamics  → train / optimize → deploy real
Learned-model route：interaction data → learned dynamics → imagine → optimize
```

**interaction data 可来自 real / sim 或混合**——learned-model route ≠ real-only learning：sim pretrain、real adaptation、sim+real joint、physics + learned residual 都是它的实例化。

需要说准：world model **并未取消 simulator**、只是把它从"手工指定的 physics model"换成"从交互数据学出的 predictive model"——**改变的是 model source**：

$$f_{\mathrm{hand\text{-}designed}} \;\longrightarrow\; f_{\mathrm{learned}}$$

Dreamer（1912.01603）、TD-MPC2（2310.16828）体现这条路。当**人工 simulator 的 model bias 大到不值得先修**时、world model 提供的是对 sim-to-real 问题本身的改写、而不是它下面的一个 transfer technique。DayDreamer（2206.14176）常被误读成"sim 预训练 → real 微调"、更准表述：**它展示了 real-interaction-driven 的实验路线**——在真实机器人上直接学 world model、用 latent imagination 做 policy improvement、不依赖手工 physics simulator 预训练 policy。但这是 DayDreamer 的选择、不是 world-model family 的定义；**不依赖手工 simulator ≠ model-free**、world model 学习仍吃满各种假设（representation / architecture / action space / reward / exploration / 真机数据质量）、只是把 inductive bias 从"显式 physics"移到"learned world model"里。

诚实的边界："用真实数据学 dynamics" **不等于天然优于仿真**：它把"手工建模成本"换成"真机采集 + 模型容量成本"；contact-rich / long-tail / 传感器噪声大的场景里学到的 model 常在分布外给出**很自信、也很错的想象**——是"手工 sim"与"直接真机 RL"之间的**又一个 trade-off**、不是终局。

### Sim-and-real co-training：把"迁移"重述成 data mixture

Maddukuri et al.（RSS 2025，2503.24361）的 Sim-and-Real Co-Training 是个务实方向。**论文实际报告**：同一批训练把 sim 与 real 混合采样、**两平台、六视觉操作任务**上相对 baseline 观测到**平均约 37.9% 的 aggregate relative improvement**（**论文定义的 aggregate improvement metric**、随任务定义而异、不应读作 success rate 绝对百分点提升、也不是"50% → 87.9%"那种绝对差；per-task 数字请回原文核对）。它不做 sim→real 单向迁移、而是用一个 recipe 决定两者比例与调度。

**本文的解读（非论文证明）**：把它读成 **data-mixture 问题**——DR 与真实数据不再是替代关系、而是同一份 sampling distribution 上 $T_R[p_{\mathrm{raw}}]$ 的两个来源（呼应上篇"$p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$"）。co-training 的**主要干预变量是 training mixture** $p_{\mathrm{train}}=\lambda\, p_{\mathrm{sim}}+(1-\lambda)\, p_{\mathrm{real}}$、不是 simulator calibration 也不是 deployment-time adapter；**$\lambda$ 只是 sampling-level 简化**、真实 recipe 还通过 dataset size / batch construction / importance weighting / augmentation / loss weighting / curriculum 改变**有效**训练分布。别简化成"只动数据分布一根轴"：mechanistic 分析（Lei et al., arXiv 2026，2604.13645）指出 mixture 的改变会诱发 **structured representation alignment 与 importance reweighting**——"以 mixture 为主抓手、效应跨多维"、而非与前四条轴严格正交的第五根。

## 评估：你怎么知道自己把 gap 补好了？

危险的做法是只在 sim benchmark 上报性能——那衡量的是 policy 与自己 simulator 的一致性、不是与真实世界的一致性。更可信评估应至少：

- 报告 **zero-shot transfer**（不做真机微调）性能与 **few-shot / N-shot** 之后的曲线；
- 用一组 **held-out hardware / calibration / object / contact / environmental regimes**（unseen object / payload / surface / camera / battery level / temperature / reset regime……）来测、而不是只有"那台部署机器人"——多数实验室换不起 physical system、但完全能 hold out 这些 regime；
- 明确声明 sim 与 real 的**任务 / initial-state / evaluation distribution 是否一致**（objective mismatch 要先对齐）；
- 做**失败归因**：哪层 $\Delta_k$ 主导？不同层敏感度与补救成本天差地别、归因错了预算就花错地方；
- **不要只报均值**：至少 mean ± CI、跨多 seeds / resets；尽量 **paired evaluation**（相同 object / initial state / scene / seed 比 A、B）、把环境 noise 从比较里剥出去；
- **安全失败单独统计**：emergency stop 与没抓到物体混成同一 success rate 会掩盖真实部署成本、$J_{\mathrm{real}}$ 应并列 safety violation / e-stop / intervention count / hardware fault / recovery time。

顺着"sim 是真实世界的代理"这句、还有个比"数值对齐"更本质的问题：**simulator 能否正确预测"哪个 policy 更好"？**

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

在 sim 上看起来 $A > B > C$、真机却是 $B > C > A$。这时 simulator 不只是 calibration error、而是**失去了 model-selection utility**——你会用它挑出最差的政策；工程真正怕的是"我相信 sim、结果 top-1 选错"。故**当 simulator 被用于 policy / model selection** 时应同时看排序相关性 $\rho_{\mathrm{rank}} = \mathrm{Spearman}(J_{\mathrm{sim}}(\pi_i), J_{\mathrm{real}}(\pi_i))$ 与 selection regret：

$$\pi_{\mathrm{sim}} = \operatorname*{arg\,max}_{\pi \in \Pi} J_{\mathrm{sim}}(\pi), \qquad R_{\mathrm{select}} = J_{\mathrm{real}}\big(\pi^{*}_{\mathrm{real}}\big) - J_{\mathrm{real}}\big(\pi_{\mathrm{sim}}\big)$$

Spearman=0.95 却把 top-1 选错仍是灾难；反过来 Spearman=0.7 但 top-1 基本不出错、对"选一个能部署的 policy"够用。两者都是 **conditional metric**、只针对"用 sim 选 policy"这一用途。**由此提炼关键判断**：**simulator fidelity 是 task-of-use dependent、不是 absolute property**——sim 用作 policy selection 时 calibration error 未必第一优先级、decision-relevant ranking 与 regret 才是；换用途（预训练 / exploration / curriculum / safety filter）"哪些误差重要"整个变一遍。**另外**：$\pi^*_{\mathrm{real}}$ 通常不可获得、$R_{\mathrm{select}}$ 只能作为 **conceptual target metric**、实际用 best-observed 或 Pareto-best proxy。

至此、**本文另一个重要判断**（除 allocation framework 外）：**simulator utility 不是属性、而是三个不能互替的维度**——

| Simulator utility 维度 | 典型 metric |
| --- | --- |
| 数值预测准不准（absolute error / calibration） | MAE / RMSE $\mathbb{E}\big[|J_{\mathrm{sim}}(\pi) - J_{\mathrm{real}}(\pi)|\big]$、calibration curve、跨 operating regime 的误差分布；simulator 输出带 uncertainty 时可评估 prediction interval coverage |
| 排序准不准（ranking） | Spearman $\rho_{\mathrm{rank}}$、Kendall $\tau$ |
| 选出的 policy 好不好（decision quality） | $R_{\mathrm{select}} = J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$（$\pi^*_{\mathrm{real}}$ 通常不可知、实际用 best-observed proxy） |

一个 simulator 可校得很准却选错 policy（distribution narrow）、也可数值全错但排序稳、regret 小——三维不能互替。也正因此 $U_{\mathrm{sim}}$ 不该写成抽象标量、应展开成**按用途分类的 utility**：

$$U_{\mathrm{sim}} \;\in\; \big\{\ U_{\mathrm{pretrain}},\ U_{\mathrm{selection}},\ U_{\mathrm{exploration}},\ U_{\mathrm{curriculum}},\ U_{\mathrm{safety}}\ \big\}$$

评 fidelity 不能只盯单个 policy、而要相对**候选 policy family** 与**具体用途**：$U_{\mathrm{sim}}(\cdot \mid \Pi_{\mathrm{candidate}},\ p_{\mathrm{eval}}^{\mathrm{real}})$。**"我们的仿真器很真"从来不是有意义的评价**——它没回答**对哪一维、对哪个用途**。

## 组合与决策，以及一个常被回避的问题

有了优先级、"什么时候用哪个"就不该是固定流水线、而是查询表。真实项目常常几个条件同时成立、更有用的是 **gap × 可建模性 × 真机预算** 决策矩阵（"Real data" 列对应预算向量 $B$）：

| Gap | 可参数化 / 可辨识？ | Real data | 推荐 |
| --- | --- | ---: | --- |
| low-dimensional dynamics bias | 高 | 少 | SI |
| parameterizable dynamics uncertainty | 中 | 少 | posterior-guided DR / Bayesian SI → DR |
| dynamics residual | 低（但有结构） | 中 | Residual learning |
| visual appearance | 高 | 无 / 少 | DA / DR |
| actuator latency | 高 | 少 | SI + DR |
| unknown long-tail，可模拟 | 低 | 少 | targeted simulation / DR |
| unknown long-tail，sim 生成不可信 | 低 | 中 | real data |
| model class 不确定 | 低 | 多 | learned world model（若 real 稀缺则先 physics prior + residual / DR） |
| mixed | mixed | mixed | co-training candidate（需先验证正迁移条件：mixture ratio 与 cross-domain alignment） |

**前两行的限定词不能省**：若 uncertainty 来自 **model-class uncertainty**（simulator 函数形式本身表达不了真实现象）、SI 与 DR 都未必适用、得先落到 residual / world model / 真机数据那几行——把 "dynamics bias / uncertainty" 默认为"可参数化"是常见口诀化错误。

倒数第二行：光 "model unknown" 推不出 world model、判据是 **model uncertainty × real-data budget**——模型类不确定**且**真实交互充足时 learned world model 才合理；real 稀缺则先保 physics prior + residual / DR。最后一行同理："co-training 兜底"与 allocation 冲突——sim 质量差、real 很少、action space / task semantics 不一致时完全可能负迁移、须先验证正迁移条件。**"unknown long-tail" 两行合看**：真实数据最有价值的用法往往不是**大量覆盖**、而是**发现 simulator 没建模的 failure mode**、让 sim synthetically 放大——

$$\text{discover real tail} \rightarrow \text{identify structure} \rightarrow \text{synthetically amplify} \rightarrow \text{real validation}$$

也就是 **real 用来发现、sim 用来放大、real 再用来验证**、每单位预算都花在自己 $MV$ 最高的那一段。

常见组合 **SI → DR → DA → co-training / fine-tune**：SI 校一个"80% 对"的 sim、DR 在"说不清但可枚举"方向撑开族、DA 处理视觉域差、最后少量真机数据收尾。**箭头只是示意、不是固定 workflow**——实际顺序由当前主导 gap 与边际效用决定：真机充足时先 SI 未必划算；视觉主导时 DA 提前；SI 数据极少时先粗 DR 再回头校准往往更合理。

顺着这个逻辑、可以回答整篇几乎回避但框架本身允许的反问：**什么时候最优解其实是"不做 sim-to-real"？**
- **真机数据已便宜到 $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}^{\mathrm{effective}}$ 时**——$C_{\mathrm{real}}^{\mathrm{effective}}$ 是**有效真机成本**（不只采集、还含安全 / operator / reset / 磨损 / 失败恢复 / deployment 多样性与可复现性）、全算进去"4 小时真机 vs 20 小时 sim"才不会失真。**但不等式两侧不应读成一次性绝对小时数**：SI 那 2 小时可能让未来 100 小时 sim 都有价值、DR 收益也随轮数累计——**正确比较是"当前预算 horizon 内的 expected cumulative value / cost"**、不是"一次 intervention 的 raw hours"、与前文 fixed cost + diminishing returns 一致。
- **仿真器 model class 本身就差**（$\Delta_{\mathrm{model}}$ 主导且难参数化、如软体 / 流体 / 复杂接触）——修 sim 边际效用极低、不如走 world model 或真机数据学习。
- **部署分布非常固定**时——不需大规模 DR 覆盖整个族、少量 targeted real fine-tuning 往往更划算。

能大方承认"有时最优解是不做 sim-to-real"、恰恰是 allocation framing 应有的样子：**不站"仿真"、只站"下一单位预算换回最多真实性能"。**

## 这意味着什么？：一个闭环，而不是一个开关

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)核心句是 evaluation-aware distribution allocation：有限预算下把每一单位花到 marginal data value 最高的地方。套回 sim-to-real 得一个自然推论——**仿真数据的 utility 从来不是 simulator 的内部属性、而是相对于真实 evaluation distribution 的属性：**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

这解释了常见挫败："堆更多 sim 数据"有时没用。但要讲准：增加 sim 数据也可能带来更广 coverage / 更多物体 / 更高 rare-event 频率、所以这句**有条件**——**当主要瓶颈恰好是 simulator 与真实 evaluation distribution 之间的 support / fidelity mismatch**、加同分布 samples 的边际收益会快速下降；加 $N$ 改善的是 sampling density、**不能自动创造 evaluation-relevant coverage、也不能修正 model bias**。与其问"我的 sim 有多好"、不如问开头那句："我的 sim 在哪些 evaluation-relevant 方向上接近真实、哪些差得远？差得远的那些敏感度多高、用哪种预算压它最便宜？"

把这条线走完、sim-to-real 就不再是"能不能迁移成功"的开关、而是一条带反馈的闭环：

$$\boxed{\ \text{mismatch} \rightarrow \text{sensitivity} \rightarrow \text{intervention} \rightarrow \text{marginal utility} \rightarrow \text{budget allocation} \rightarrow \text{real evaluation} \rightarrow\ \circlearrowleft\ }$$

（最后一步会重新改变 sensitivity 与 mismatch 估计——见上文 feedback loop。）

这条链不是从 simulator 里解析跑一遍就出答案、而是一个 **resource-constrained adaptive sequential experimentation framework**：敏感度与边际收益都靠小步实验在真实评估上估出来、一轮估完再决定下一份预算投到哪。收束：**sim-to-real 不是 transfer 技巧、而是在 model fidelity / 训练多样性 / 表示对齐 / 真机交互与工程成本之间做 constrained、可迭代估计的 allocation 问题**——与下篇"数据 scaling 是 sequential data allocation"同一件事。

---

## 参考文献

正文涉及的主要工作（均按 arXiv ID 检索）：

- Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World — Tobin et al., IROS 2017, arXiv:1703.06907
- Sim-to-Real Transfer of Robotic Control with Dynamics Randomization — Peng et al., ICRA 2018, arXiv:1710.06537
- Sim-to-Real: Learning Agile Locomotion For Quadruped Robots — Tan et al., RSS 2018, arXiv:1804.10332
- Learning Dexterous In-Hand Manipulation — Akkaya et al. (OpenAI), 2019, arXiv:1808.00177
- Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks — James et al., CVPR 2019, arXiv:1812.07252
- DiffTaichi: Differentiable Programming for Physical Simulation — Hu et al., ICLR 2020, arXiv:1910.00935
- Interactive Differentiable Simulation — Heiden et al., arXiv 2019, arXiv:1905.10706（NeuralSim: Augmenting Differentiable Simulators with Neural Networks 是同组 ICRA 2021 的另一篇）
- Residual Physics Learning and System Identification for Sim-to-real Transfer of Policies on Buoyancy Assisted Legged Robots — Sontakke et al., 2023, arXiv:2303.09597
- Sim-to-Real of Soft Robots with Learned Residual Physics — Gao et al., IEEE RA-L 2024, pp. 8523–8530, arXiv:2402.01086
- Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) — Hafner et al., ICLR 2020, arXiv:1912.01603
- DayDreamer: World Models for Physical Robot Learning — Hafner et al., CoRL 2022, arXiv:2206.14176
- TD-MPC2: Scalable, Robust World Models for Continuous Control — Hansen et al., ICLR 2024, arXiv:2310.16828
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., RSS 2025, arXiv:2503.24361
- A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies — Lei et al. (Yu Lei, Minghuan Liu, Abhiram Maddukuri, Zhenyu Jiang, Yuke Zhu), arXiv preprint 2026, arXiv:2604.13645

sim-to-real 尚无公认的跨任务"哪种方法更强"定量对照——不同任务 / 硬件 / fidelity 上限下结论可能完全颠倒；上述工作更多是"这类 gap 用这个方法可行"的样本、而非可外推的排序。本文关于四个 intervention lens / 分析维度的分解、error-budget constrained-allocation 的形式化、$\hat S_k^{\mathrm{int}}$ 与 $MV$ 的定义都是 **conceptual framework 与作者解读**：$\hat S_k^{\mathrm{int}}$、$MV$ 是靠 sensitivity experiments / ablation / 小规模真实评估估出的 decision statistics、不是 simulator 解析可求的量；把 co-training 读作 data-mixture、把 world model 读作 model-source replacement、同样不是受控实验证明的结论。

---

*本篇是"具身智能的数据问题"上下篇续篇：上篇讲数据来源与接口、下篇讲数据 scaling 框架；本篇把镜头拉到 sim-to-real、把它从"一堆迁移技巧"重述成带经验边际效用的闭环分配问题、接回下篇 sequential data allocation 主线。*
