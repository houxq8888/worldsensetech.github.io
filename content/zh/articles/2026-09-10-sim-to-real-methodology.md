---
title: '具身智能 Sim-to-Real 方法论深潜：把"从仿真到真实"当成一次误差预算分配'
slug: "2026-09-10-sim-to-real-methodology"
date: 2026-09-10
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "Sim-to-Real", "Domain Randomization", "System Identification", "可微仿真", "Residual Physics", "世界模型", "Domain Adaptation", "机器人数据"]
description: 'sim-to-real 不是单一迁移技巧、而是闭环资源分配。本文把 reality gap 重述成 policy-conditioned 多源 mismatch、以 intervention sensitivity 与 cost-normalized marginal value 把误差预算写成可迭代的决策框架、厘清 SI / DR / DA / fine-tuning 四条 intervention lens 的机制与失效边界。'
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

本文不提出新的 sim-to-real algorithm、而是提出一个用于比较和组合既有 intervention 的 decision framework。全文按三个层级收敛：

- **Level 1 — Diagnosis**：$M_{\mathrm{sim}} \rightarrow p_{\mathrm{sim}}^\pi \rightarrow J_{\mathrm{sim}}$ vs. real——哪里不一样？哪里重要？
- **Level 2 — Intervention**：四个 lens（Model × Data × Representation × Optimization）——哪种干预可以改变这个 mismatch？
- **Level 3 — Allocation**：$MV(m \mid b, \pi)$——当前状态、预算与 uncertainty 下、下一块资源投哪里？

然后 $\text{real evaluation} \rightarrow \text{update belief} \rightarrow \text{next intervention}$、闭环。

四个核心贡献：**(1) 重新定义 mismatch**——reality gap 是 policy-conditioned、task-conditioned consequence、不是 simulator 固有标量；**(2) 区分 diagnosis 与 intervention**——mismatch descriptor / sensitivity 只是诊断、真正决策变量是 intervention；**(3) 把方法选择改写为 multi-resource sequential allocation**——SI / DR / DA / fine-tuning 是 intervention lenses、不是互斥方法；**(4) 把 simulator evaluation 从 fidelity 扩展到 downstream utility**——prediction / ranking / selection quality 是不同 utility、并作为 allocation framework 的 corollary。

乍看是工程直觉、其实是闭环资源分配：几笔不能互换的预算下、要不断问"下一块钱花在哪、换回最多真实性能"。项目里最卡人的不是"不知道有这些方法"、而是"这类 gap 管不管用、花哪种预算"。"误差预算"**不是**给每个误差项预分固定额度、而是花在**干预动作**上、通过 sequential allocation 逐步压低最有价值的 mismatch。

## Reality Gap：不是一个标量，而是一个 policy-conditioned 的 mismatch

Sim-to-real 常被叙述成"训练 policy 从仿真迁移到真实"。更严格的起点是**两个分布**：同一条 $\pi$ 与环境交互各自诱导 $p_{\mathrm{sim}}^{\pi}(\tau)$ 与 $p_{\mathrm{real}}^{\pi}(\tau)$、一般不等：

$$p_{\mathrm{sim}}^{\pi}(\tau) \;\neq\; p_{\mathrm{real}}^{\pi}(\tau)$$

轨迹分布本身是 **policy-induced**、随 $\pi$ 而变、不是环境固有属性。真正关心的不是分布差、而是它在任务上**表现的后果**——同一 $\pi$ 两边的性能差：

$$\boxed{\;\delta_J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)\;}$$

叫 **transfer delta**、保留符号：$J$ 若是 success rate、真实反而更好（sim 更保守、或噪声更狠）时 $\delta_J$ 会为正、直觉上不该叫 gap。故另把幅度

$$G_J(\pi) \;=\; \big|\,\delta_J(\pi)\,\big|$$

单独叫 **performance gap**——下文谈敏感度用这个语义、不与符号纠缠。**注意：$\delta_J$ 不等于 reality gap 本身**——它是 gap 在特定 policy + evaluation 下的 downstream consequence；reality gap 是更完整的四元组属性（见下文）。

**distribution mismatch ≠ performance gap**：$p_{\mathrm{sim}}^{\pi} \neq p_{\mathrm{real}}^{\pi}$ 不自动意味着 $\delta_J$ 很大、不同 policy 对分布差的敏感度完全不同——只依赖粗粒度几何的 policy 换掉摩擦建模性能几乎不变；依赖高频力反馈的精细装配里同样分布差可能致命。

更本质地说、真正影响 policy 的不是 marginal state distribution $p_{\mathrm{sim}}(s)$ vs $p_{\mathrm{real}}(s)$、而是 **policy-conditioned occupancy** $d_{\mathrm{sim}}^{\pi}(s,a)$ vs $d_{\mathrm{real}}^{\pi}(s,a)$——contact-rich manipulation 里甚至要加 contact mode 索引 $d^\pi(s,a,\text{contact mode})$。逻辑链因此是 **$\pi \rightarrow d^\pi \rightarrow \text{mismatch} \rightarrow J$**、而非仅仅 $\pi \rightarrow p^\pi(\tau)$。

$\delta_J(\pi)$ 是**任务相关、policy 相关的可观测后果**。严格写要把 **mechanism 与 induced distribution 分开**：环境的 transition / observation / actuation kernel 记作机制 $M_{\mathrm{sim}}, M_{\mathrm{real}}$、给定 $\pi$ 下**诱导**出 $p_{\mathrm{sim}}^{\pi}(\tau),\ p_{\mathrm{real}}^{\pi}(\tau)$。gap 更干净的写法：

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ \mathcal{E};\ M_{\mathrm{sim}},\ M_{\mathrm{real}}\big)$$

逻辑是 **mechanism → trajectory distribution → performance**：$\mathcal{E}$ 是 evaluation 假设集合（initial-state / horizon / reward / constraints）——同一 $M_{\mathrm{sim}}$ 对 position control policy 可能 gap 很小、对 force-sensitive manipulation policy 可能 gap 巨大。**reality gap 是这个四元组的属性、不是仿真器的属性**。

### gap 到底在哪里：reality mismatch 与 task-specification mismatch

第一步是把多源 gap 拆开——有**两大类来源**、不能全塞进"reality"一词下面：

```
Sim-to-real / task mismatch
├── Reality mismatch（物理层面）
│   ├── Dynamics / contact / stochasticity  摩擦、接触、可形变体、柔顺结构；motor stochasticity、friction variability、unmodeled disturbance、repeated-reset variability
│   ├── Observation / estimation  传感器物理、标定、噪声、遮挡、时延、状态估计
│   ├── Actuation / timing        电机动力学、控制频率、执行器延迟、通信抖动
│   └── Initial-state / env.      reset 分布、场景布局、长尾、初始条件
└── Task-specification mismatch
    └── Objective / constraint    reward 定义、安全约束、成功判据
```

两类来源不同、不要简单相加：reality mismatch 是"仿真与真实不是同一世界"、task-specification mismatch 是"优化目标与部署目标不是同一任务"。**观测与状态估计值得单独成层**——机器人真正执行 $a_t = \pi(o_t),\ o_t = h(x_t) + \epsilon$；camera 标定误差 / depth bias / 遮挡 / proprioception drift / 力传感器偏置 / state estimator 时延**不是"画面不一样"、而是让 policy 看到的 state 与 simulator 假设可用的 state 不一致**——manipulation / locomotion 里这类"状态估计 gap"往往比外观 gap 更伤 performance。

此外、**stochasticity mismatch**（motor 随机性、friction variability、sensor temporal correlation、communication jitter、unmodeled disturbance、repeated-reset variability）不等同于 parameter mismatch——它描述的是 dynamics 的**高阶统计量 / 随机过程结构**不同、而这恰恰是 DR 的 $p_{\mathrm{DR}}(\xi)$ 要覆盖的对象。

尤其要注意：timing mismatch（$\Delta t_{\mathrm{sim}} \neq \Delta t_{\mathrm{real}}$、action hold、sensor delay、policy inference latency、asynchronous observation）**可被闭环反馈动力学放大**、不是简单的 additive observation error——它能改变 closed-loop stability 本身。

**Initial-state / environment mismatch** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$——reset 分布 / 场景布局对不上；**Objective / task shift** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$——仿真只要 grasp success、真实还要 collision avoidance。**initial-state 归因要谨慎**：若 sim 与 real **都能产生同样 $s_0$**、只是训练没覆盖（sim 生成红/蓝杯、部署全蓝杯、policy 只训红杯），这是**一般 train-test shift、不是 reality gap**；只有 sim-real reset / scene **实现本身**对不上才是 environment mismatch。Objective shift 则**已是 objective mismatch 而非 reality gap**：物理再准、reward / 约束对不上就不是"迁移失败"、而是"评测的根本不是同一任务"；下文默认 objective 已对齐。

## 把"误差预算分配"写成一个可估计、可迭代优化的决策框架

拆完来源、给开头直觉一个数学落点。误差项强烈交互——sim 假设 proprioception 精确、真实有 latency、单看都不致命、叠加可能让 controller 失稳——更稳妥的写法是先承认未知的耦合函数 $F$：

$$\boxed{\;\delta_J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}}\big)\;}$$

**这一版把 $\Delta_{\mathrm{opt}}$（优化 / 学习误差）从 reality gap 拿掉**：层级不同——同一固定 policy、仿真观测动力学都准但 RL 没训好、$\delta_J$ 很小、policy 却很差——应分开成**两个诊断量**：

$$\underbrace{J_{\mathrm{real}}(\pi_{\mathrm{train}}) - J_{\mathrm{sim}}(\pi_{\mathrm{train}})}_{\text{transfer delta } \delta_J}\qquad \underbrace{J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})}_{\text{real-domain learning gap}}$$

**这两个量不能无条件相加叫 deployment loss**：$\delta_J$ signed、两项 baseline 也不同；它们是**不同层级的误差来源**、分别诊断、分别归因。

只在工作点附近做工程归因时、才把 $F$ 局部近似成加权和 $\delta_J \approx \sum_k w_k \Delta_k$——**这一层只是局部归因 heuristic、不是全文核心公式**。真正用来做 decision 的是每类 mismatch 挑一个 **intervention 变量** $\xi_k$ 后测得的 **intervention sensitivity**：

$$\hat S_k^{\mathrm{int}} \;\approx\; \frac{J_{\mathrm{real}}\big(\pi;\,\xi_k{+}\delta\big) \;-\; J_{\mathrm{real}}\big(\pi;\,\xi_k\big)}{\delta}$$

**关键澄清**：$\xi_k$ **不是"真实 gap 的天然坐标"、而是为 sensitivity experiment 人为定义的 intervention variable**——latency / friction / appearance 可直接扰动、camera calibration error / contact-model mismatch / state-estimation error 很难在真实世界连续拨动。这是**受控实验性扰动**、不是对固有量求导（此处刻意不用 Pearl-style $\operatorname{do}$-calculus 记号、避免暗示完整因果图假设）。**$\hat S_k^{\mathrm{int}}$ 只是诊断辅助**、**核心决策量是下节 $MV(m\mid b,\pi)$**、叙事为 diagnosis → intervention → empirical marginal utility → allocation；$\Delta_{\mathrm{model}}$ 与 $\Delta_{\mathrm{ctrl}}$ 可能不可辨识地互相补偿（actuator gain 错、policy 靠 command distribution 补回）、均为 sensitivity experiments / ablation 估出的 decision statistics。

### 真正的"分配"：把钱花在干预动作上，而不是在方法里挑一个

要让预算分配名副其实、预算得**连续地**分到每条干预轴：总预算拆成向量 $b=(b_1,\dots,b_K)$、$b_k$ 是花在干预 $k$ 上的量——$b_{\mathrm{SI}}=2\text{h}$、$b_{\mathrm{DR}}=10^6$ 步 sim、$b_{\mathrm{real}}=4\text{h}$ 真机——而不是"用不用 SI"这种 0/1 选择。目标是最大化真实性能：

$$\max_{b}\quad J_{\mathrm{real}}\big(\pi_b\big)$$

机器人项目里的预算**不是同一种货币**：GPU 近乎无限但真机机时极少、有机器时间却没工程人力、故正确写法是**多预算约束**、不折成标量 $B$：

$$\begin{aligned}
C_{\mathrm{real}}(b) &\le B_{\mathrm{real}}\\
C_{\mathrm{compute}}(b) &\le B_{\mathrm{compute}}\\
C_{\mathrm{eng}}(b) &\le B_{\mathrm{eng}}\\
C_{\mathrm{risk}}(b) &\le B_{\mathrm{risk}} \quad\text{（安全预算：e-stop 次数 / hardware fault 容忍 / operator intervention 上限）}
\end{aligned}$$

预算是分向量后、决策变量就该从"gap"换成"干预动作"：工程师买不到"$\Delta_{\mathrm{model}}$ 的 2 个百分点"、能买到 30 分钟 SI / $10^6$ 步 sim / 100 条真机轨迹 / 一次 camera calibration / 一个 residual model。对干预 $m$ 定义边际效用更自然——**干预不直接改 $\Delta_k$、而是通过训练过程改变 policy**：

$$\boxed{\;\pi_{b+m} \;=\; \operatorname{Train}\big(D_{\mathrm{sim}},\ D_{\mathrm{real}};\ m\big)\;}$$

于是"下一块钱花在哪"是以干预为变量、需要在真实世界里逐步估计的量。**本文的核心决策公式是 $MV$ 而不是 $w_k$ 或 $\hat S_k^{\mathrm{int}}$**（$MV$ 更准确的名字是 **cost-normalized marginal value（单位成本边际价值）**）：

$$\boxed{\;MV(m \mid b, \pi) \;=\; \frac{\mathbb{E}\big[\,J_{\mathrm{real}}(\pi_{b+m}) - J_{\mathrm{real}}(\pi_{b})\,\big]}{C(m)}, \qquad m^{*} = \operatorname*{arg\,max}_{m} MV(m \mid b, \pi)\;}$$

这个 ratio 无法从 simulator 解析求得、只能用 pilot / ablation / few-shot real evaluation **sequential 地估**。三条 caveat：**(i) 不确定性**——真机 $\Delta J$ 噪声极大、allocation 应看 CI / posterior / **lower confidence bound (LCB)**、否则高方差 intervention 会因一次偶然成功被错误优先。**(ii) 非线性成本**——$MV$ 是**局部决策统计量**：SI 一次工程可让后续训练受益（fixed cost）、DR 逐步饱和（diminishing returns）、fine-tune 有 threshold effects。**(iii) 非单调性**——**本文不假设 intervention 对真实性能单调改善**；over-randomization、过拟合式 fine-tuning、错误 residual、cross-domain negative transfer 都意味着 $MV(m\mid b,\pi)$ **可以为负**（$\Delta J_{\mathrm{real}} < 0$）——这正是 allocation 比 recipe 难、也比 recipe 有趣的地方。

**不同干预的 $MV$ 也不是固定常数**：$MV_i = MV_i(b_{1:i-1},\ \pi_b,\ D_{\mathrm{real}})$——先 SI 缩窄 uncertainty set、DR 的 $MV$ 下降；先 DR 起点更 robust、fine-tune 的 $MV$ 上升。**intervention 之间同时存在 complementarity、substitutability 与 occasional conflict**、故这是 **resource-constrained sequential experimentation / adaptive allocation**（接近 adaptive experimental design、但**别写成 bandit algorithm**——没有严格 arm / stationary reward / regret 证明）。

还有一层更隐蔽的反馈：**intervention 不只压低 gap、还会改变 policy、进而改变 policy 对 gap 的敏感度本身**——$S_k^{\mathrm{int}} = S_k^{\mathrm{int}}(\pi)$、$\pi = \pi(m)$、闭环并非单向：

```
estimate mismatch → estimate sensitivity → intervention
       ↑                                          ↓
   re-estimate  ←  sensitivity changes  ←  policy changes
```

**这张 feedback loop 比任何新公式都更贴合本文 allocation thesis**：sim-to-real 不是一次解完的优化、是一轮做完重新估一轮的 sequential experiment。

把每条干预对应到主要压缩项与主要预算：

| Intervention | 主要压缩项 | 主要预算 |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{eng}}$ + $C_{\mathrm{compute}}$ + 少量 $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | $C_{\mathrm{compute}}$（样本效率） |
| Residual physics | $\Delta_{\mathrm{model}}$（残差部分） | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$（appearance 子集） | $C_{\mathrm{real}}$（未标注数据）+ $C_{\mathrm{compute}}$ |
| Real-world fine-tuning | **不直接对应单一 mismatch；通过目标域 optimization 改变 policy**（可同时改 transfer delta 与 real-domain learning gap） | $C_{\mathrm{real}}$（磨损 / 安全） |
| World model | 改变 model source | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Sim-and-real co-training | 改变 $p_{\mathrm{train}}$（$\Delta_{\mathrm{dist}}$ 为主） | 混合数据（$C_{\mathrm{real}}+C_{\mathrm{compute}}$） |

有了这套写法、全文就不是"四种方法谁更好"、而是闭环：定位主导 $\Delta_k$、sensitivity 判断多重要、在 $MV$ 最高的干预上投一份预算、真实评估量回报、再决定下一份。

## 四个 intervention lenses（更准确说，四个相对独立的分析维度）

框架有了、再逐条看工具。SI、DR、DA、fine-tuning **不是同一抽象层级的并列类别**——SI 是 model calibration、DR 是 training distribution manipulation、DA 是 representation alignment、fine-tuning 是 optimization strategy——并排成"四类方法"会误导人四选一、其实是**四个相对独立的 intervention lens**、可组合（**本文的 analytical decomposition、非领域公认 ontology**）：

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

"$\times$" 是**组合空间**、不是数学正交——DR 触及 Model / Observation / Distribution、DA 可发生在 input / feature / latent / policy / output——"DA = Representation 轴"只是本文的一层 abstraction。

**选工具的标准不是"systematic → SI、random → DR"**——更有用的划分是"**点估计 → 后验 → 鲁棒随机化**"这条连续谱。SI 真正做的是**在 identification objective 下拟合参数**：

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; \mathcal{L}_{\mathrm{ID}}\big(D_{\mathrm{real}},\ f_{\mathrm{sim}}(\cdot\,;\,\phi)\big)$$

$\mathcal{L}_{\mathrm{ID}}$ 可以是 trajectory prediction / one-step transition error / force-torque residual / likelihood——**很多经典 SI 不做 trajectory distribution matching、只最小化预测误差**。它解决**可辨识、可参数化的 model mismatch**；同理 DR 解决**能被训练分布表示出来的 uncertainty**。

| mismatch 的性质 | 更自然的工具 |
| --- | --- |
| 可参数化 + 可辨识 | System Identification（point estimate $\hat\phi$） |
| 可参数化但只能给出不确定性 | Bayesian / posterior SI → posterior-guided DR |
| 可参数化但难辨识 / uncertainty 大 | Domain Randomization |
| 难以由低维物理参数充分表达、但有结构化 residual | Residual learning |
| observation / appearance mismatch | Domain Adaptation |
| policy 在目标域仍有 systematic residual | Fine-tuning |

关键：**"不能精确辨识"与"完全不知道"不是一回事**——拿到后验 $p(\phi \mid D_{\mathrm{real}})$ 后、最自然的动作是 $\phi \sim p(\phi \mid D_{\mathrm{real}})$ 做 **posterior-guided randomization**、把 SI 与 DR 缝成连续谱。

### Axis A — Model：system identification、可微仿真与 residual physics

这条轴处理 $\Delta_{\mathrm{model}}$、内部三个**不同层次**常被"可微仿真 = 更强 SI"打包：

$$x_{t+1} \;=\; \underbrace{f_{\mathrm{physics}}(x_t,a_t;\phi)}_{\text{可参数化的物理}} \;+\; \underbrace{r_\theta(x_t,a_t)}_{\text{残差}} \;+\; \epsilon_t$$

- **可微仿真**回答"怎么优化模型"——提供 $\partial f/\partial\phi$；DiffTaichi（Hu et al., ICLR 2020，1910.00935）、Interactive Differentiable Simulation（Heiden et al., arXiv 2019，1905.10706）让参数估计可梯度化。
- **System identification** 回答"优化什么参数"——真实工作流常是 **real → identify → sim → train → real**（real-to-sim-to-real）。
- **Residual physics** 回答"模型没解释掉的部分由谁解释"——让网络学 $r_\theta$ 补差。

$r_\theta$ 只是**统一记号**：实际 residual 可定义在状态转移 / force / acceleration / contact impulse / deformation field 或其他 latent 上。

最容易被"可微"二字掩盖的点：**可微性解决 optimization interface、不解决 model class correctness**。若 contact model 根本没表达某种真实现象、再精确的梯度也只给你"错误模型下的最优参数"。**常被忽略的边界**：碰撞 / 摩擦 / 接触模式切换往往是 **nonsmooth / piecewise-smooth**——即使 $\partial f/\partial\phi$ 存在、也不保证梯度稳定、contact mode 切换处梯度有意义、或优于 derivative-free optimization。

SI 还有两个细坑。**其一**、$p_{\mathrm{real}}(\tau)$ 几乎不可直接访问、只有有限条真机轨迹。**其二**、参数存在 ≠ 可辨识——identifiability 还依赖 excitation 与 sensor observability、质量 / 阻尼 / 刚度在某些激励下产生几乎相同的可观测轨迹、无法独立估出。

Residual physics 的边界要收窄：甜蜜点是 residual 在目标分布上相对受限、$f_{\mathrm{physics}}$ 仍提供**有用的结构性归纳偏置**（inductive bias / state representation / constraints / extrapolation prior）；若 $f_{\mathrm{physics}}$ 完全错、残差独自承担整个 dynamics、不如直接学一个 model。软体（Gao et al., RA-L 2024，2402.01086）、浮力腿式（Sontakke et al., 2023，2303.09597）这类"主干物理还算数、局部有稳定残差"的场景最好用。但 $r_\theta$ **并不天然等于"缺失物理"**——unrestricted additive residual 会吞下大量 model error（sensor bias / actuator error / timing / calibration / reward mismatch / policy-induced artifact）成 **error sponge**：训练分布内拟合好、一到 OOD 就失稳；故 residual 要配结构约束（低维 / 稀疏 / 力或加速度尺度 / 物理先验 / 只在特定 contact regime 生效）。

还有一点：$\phi$ 与 $r_\theta$ 之间存在 **confounding**——若残差足够灵活、它会把本应属于 $\phi$ 的效应也吸收掉、使得 $\hat\phi$ 不再有意义；identifiability 要求 $f_{\mathrm{physics}}$ 与 $r_\theta$ 的贡献能在数据上被区分（通常需正则化、scale separation 或 structural constraints）。

### Axis B — Data distribution：domain randomization 及其家族

这条轴不追求逼近"最准"的 $p_{\mathrm{real}}$、而是让 policy 对一族参数 $\{\phi\}$ 都稳健。Tobin（1703.06907）用纯视觉随机化把 sim 抓取检测搬到真机；Peng（1710.06537）推进到 dynamics；OpenAI in-hand manipulation（Akkaya et al., 1808.00177）几乎把 DR 推到极致——**不靠精确校准、靠"随机化范围足够宽"吸收差异**。

一句常被写歪的直觉：**DR 不是"隐式 ensemble"**——训练的是**单个**共享 policy $\pi_\theta$、目标是

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

更准确：**DR 是对一族环境模型做 population-level 优化**。上式是 **risk-neutral average-case DR 的 baseline abstraction**；robust / adversarial DR 还可换成 $\max_\theta \min_{\phi\in\Phi} J(\pi_\theta;\phi)$、CVaR 或其他风险敏感形式。DR 有效条件：**真实参数分布要落在 DR 支撑内**——更安全的表述是 $\mathrm{supp}(p_{\mathrm{real}}) \subseteq \mathrm{supp}(p_{\mathrm{DR}})$ 且 real-typical 区域获得足够 density。**但这还隐含一个更根本的前提**：real dynamics 可被同一 $\phi$-parameterization 表达；若 simulator model class 不包含真实现象、连 $\phi_{\mathrm{real}}$ 都无法定义、support **从根上不成立**——这时不是"DR 覆盖不够宽"、而是 model-class uncertainty。（**parameter uncertainty** 是"$\phi$ 落在哪"、可用后验或 DR 覆盖；**model-class uncertainty** 是"这个 $\phi$-parameterization 能不能表达 real dynamics"、不是加宽 range 能解决的。）**主结论**：真正决定 transfer 的不是 parameter-space marginal support、而是 policy 在 evaluation 下实际访问的 **state-action / contact occupancy $d_{\mathrm{real}}^{\pi}(s,a)$** 与训练分布诱导的 occupancy 是否足够 overlap——**parameter coverage 是必要 proxy、非 deployment coverage 的充分条件**。

再往下一层：**DR 不是选 scalar range、而是在设计 joint distribution**。$p(\phi_1,\phi_2) \neq p(\phi_1)p(\phi_2)$ 才是常态——payload ↑ 联动 actuator regime、temperature ↑ 联动 motor resistance / friction / battery。独立 uniform DR 只是方便的 baseline。关键回到分配：**randomization 分布要对齐 evaluation 分布与 objective**——过宽或无关会拉低样本效率；但 robust 设定下适当扩大 uncertainty set 反而更稳。**"越宽越保守"并非普遍规律、shape 与对齐才是。**

"Adaptive / Automatic DR" 是家族而非单一方法：curriculum / adversarial / automatic / posterior-based sampling / performance-driven range adaptation——共同点是**避免一开始就 over-randomize**。

### Axis C — Observation / Representation：domain adaptation 与观测翻译

这条轴处理 $\Delta_{\mathrm{obs}}$、在**观测/表示层**对齐 sim 与 real。**"Representation" 是本文 abstraction**——DA 实际可发生在 input / feature / latent / output / policy / dynamics model 六层。具体机制包括 feature-level adapter、image translation（GAN / 扩散）、RCAN（James et al., CVPR 2019，1812.07252）——**RCAN 更适合当"input-level canonicalization / sim-to-sim adaptation"的例子**：把随机化过的 sim 图翻回近似 canonical 的干净图喂下游 policy、顺带把 DR 与这条轴缝起来。两条边界：**其一、DA 只是 observation mismatch 子集**——camera intrinsics/extrinsics、temporal sync、sensor bias、depth distortion 更适合 calibration / SI / sensor modeling。**其二、task-relevant invariance 才是目标**——只对齐 $z_{\mathrm{sim}}\approx z_{\mathrm{real}}$ 不够、理想是保持 $I(z;y_{\mathrm{task}})$ 高的同时压低 $D(z_{\mathrm{sim}},z_{\mathrm{real}})$、与"过宽 DR 抹掉任务信号"同一件事。

### Axis D — Optimization / adaptation：真机微调

这条轴**不是一类 mismatch、而是 adaptation operator**：直接在目标域上继续优化 policy。既可作前三条轴补完的收尾、也可作**早期诊断或快速 adaptation 手段**。fine-tuning **可能同时改变 transfer delta 与 real-domain learning gap**、两者仍分别诊断；两个 regime 成本结构完全不同：

- **Offline / imitation：** $D_{\mathrm{real}} \to \theta$、主要成本是**数据采集**。
- **Online RL：** $\pi_\theta \to a \to$ 真实 transition $\to \theta'$、主要成本是**交互 + 安全 + 硬件磨损 + 探索**。

所以比较方法不能只看最终 success rate、还要看**达到目标性能所需的真机交互预算**。粗略指标：

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{或}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

但只是**粗略指标**：依赖 baseline、也不是真正的 marginal efficiency。真正该看 learning curve / AULC / 每 100 条轨迹的边际收益

$$MV_{\mathrm{real}} \;\approx\; \frac{J(N+\Delta N)-J(N)}{\Delta N}$$

——这才与全文 $MV$ 框架接上。风险不止"灾难性遗忘"：更常见是**分布收窄**——真机数据比 sim 窄得多、微调后 policy 在目标切片上更好、鲁棒性却可能反降、**generalization 换成了 specialization**。$MV_{\mathrm{real}}(N)$ **不保证始终为正**：前 100 条大涨、后续快速衰减、再往后可能过拟合甚至倒退——**fine-tuning 本身也可能进入负边际收益区间**。

## 两条松动"两个给定分布"假设的新路线

上面四条轴共享一个隐含前提：**$p_{\mathrm{sim}}$ 与 $p_{\mathrm{real}}$ 是两个给定分布**。下面两条路线恰在松动这个前提——不是"第五、第六种迁移技巧"、而是对整个问题的 reformulation。

### World model：不是取消 simulator，而是换掉 simulator 的来源

**本文 lens disclaimer**：在 allocation taxonomy 里、我把 world model 看成"model source replacement"的 reformulation——**这是本文的分析角度、不是 world model 的标准定义**。严格说 world model 外延比本节宽得多；本节只挑"相对 physics-sim 换掉了 model 来源"这个切面。

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)讨论过 world model 与 data utility 的关系。放进 sim-to-real 语境先纠正定位误读：**world model 并不天然属于 sim-to-real**——两条路线 causal direction 不同：

```
Physics-sim route：  hand-designed dynamics  → train / optimize → deploy real
Learned-model route：interaction data → learned dynamics → imagine → optimize
```

**interaction data 可来自 real / sim 或混合**——learned-model route ≠ real-only learning。

需要说准：world model **并未取消 simulator**、只是把它从"手工指定的 physics model"换成"从交互数据学出的 predictive model"——**改变的是 model source**：

$$f_{\mathrm{hand\text{-}designed}} \;\longrightarrow\; f_{\mathrm{learned}}$$

Dreamer（1912.01603）、TD-MPC2（2310.16828）体现这条路。当**人工 simulator 的 model bias 大到不值得先修**时、world model 提供的是对问题本身的改写。DayDreamer（2206.14176）常被误读成"sim 预训练 → real 微调"、更准表述：它展示了 **real-interaction-driven 的实验路线**——在真实机器人上直接学 world model、用 latent imagination 做 policy improvement。**不依赖手工 simulator ≠ model-free**、world model 学习仍吃满各种假设、只是把 inductive bias 从"显式 physics"移到"learned world model"里。

诚实的边界："用真实数据学 dynamics" **不等于天然优于仿真**——它把"手工建模成本"换成"真机采集 + 模型容量成本"；contact-rich / long-tail / 传感器噪声大的场景里学到的 model 常在分布外给出**很自信、也很错的想象**——是"手工 sim"与"直接真机 RL"之间的又一个 trade-off、不是终局。

### Sim-and-real co-training：把"迁移"重述成 data mixture

Maddukuri et al.（RSS 2025，2503.24361）的 Sim-and-Real Co-Training 是个务实方向。**论文实际报告**：同一批训练把 sim 与 real 混合采样、**两平台、六视觉操作任务**上相对 baseline 观测到**平均约 37.9% 的 aggregate relative improvement**（**论文定义的 aggregate metric**、不应读作 success rate 绝对百分点提升；per-task 数字请回原文核对）。它不做 sim→real 单向迁移、而是用一个 recipe 决定两者比例与调度。

**本文的解读（非论文证明）**：把它读成 **data-mixture 问题**——co-training 的**主要干预变量是 training mixture** $p_{\mathrm{train}}=\lambda\, p_{\mathrm{sim}}+(1-\lambda)\, p_{\mathrm{real}}$、不是 simulator calibration 也不是 deployment-time adapter；**$\lambda$ 只是 sampling-level 简化**、真实 recipe 还通过 dataset size / importance weighting / augmentation / curriculum 改变**有效**训练分布。Mechanistic 分析（Lei et al., arXiv 2026，2604.13645）指出 mixture 的改变会诱发 **structured representation alignment 与 importance reweighting**——"以 mixture 为主抓手、效应跨多维"、而非与前四条轴严格正交的第五根。

## 评估：你怎么知道自己把 gap 补好了？

危险的做法是只在 sim benchmark 上报性能。更可信评估应至少：

- 报告 **zero-shot transfer** 性能与 **few-shot / N-shot** 曲线；
- 用一组 **held-out hardware / calibration / object / contact / environmental regimes** 来测；
- 明确声明 sim 与 real 的**任务 / initial-state / evaluation distribution 是否一致**；
- 做**失败归因**：哪层 $\Delta_k$ 主导？归因错了预算就花错地方；
- **不要只报均值**：至少 mean ± CI、跨多 seeds / resets；尽量 **paired evaluation**；
- **安全失败单独统计**：$J_{\mathrm{real}}$ 应并列 safety violation / e-stop / intervention count / hardware fault / recovery time。

顺着"sim 是真实世界的代理"这句、还有个比"数值对齐"更本质的问题：**simulator 能否正确预测"哪个 policy 更好"？**

一个**概念性例子**（数值不代表实验结果）：

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

在 sim 上看起来 $A > B > C$、真机却是 $B > C > A$。这时 simulator **失去了 model-selection utility**——你会用它挑出最差的政策。故**当 simulator 被用于 policy / model selection** 时应同时看排序相关性 $\rho_{\mathrm{rank}} = \mathrm{Spearman}(J_{\mathrm{sim}}(\pi_i), J_{\mathrm{real}}(\pi_i))$ 与 selection regret：

$$\pi_{\mathrm{sim}} = \operatorname*{arg\,max}_{\pi \in \Pi} J_{\mathrm{sim}}(\pi), \qquad R_{\mathrm{select}} = J_{\mathrm{real}}\big(\pi^{*}_{\mathrm{real}}\big) - J_{\mathrm{real}}\big(\pi_{\mathrm{sim}}\big)$$

Spearman=0.95 却把 top-1 选错仍是灾难；反过来 Spearman=0.7 但 top-1 基本不出错、对"选一个能部署的 policy"够用。两者都是 **conditional metric**。**allocation framework 自然推出的结论**：simulator fidelity 是 task-of-use dependent、不是 absolute property——换用途（预训练 / exploration / curriculum / safety filter）"哪些误差重要"整个变一遍。$\pi^*_{\mathrm{real}}$ 通常不可获得、$R_{\mathrm{select}}$ 只能作为 **conceptual target metric**、实际用 best-observed 或 Pareto-best proxy。

不需要全排序时、更实用的指标是 **top-k recall**（真实 top policy 被 sim 选入 top-$k$ 候选的概率）、**regret@k**、或 "real best $\in$ sim top-$k$?" 二值判定——sim 只要能把好 policy 框进候选集就够、不必精确排序尾部。

至此、**allocation framework 的一个重要 corollary**：**simulator utility 不是单一属性、而是三个不能互替的维度**——

| Simulator utility 维度 | 典型 metric |
| --- | --- |
| 数值预测准不准（absolute error / calibration） | MAE / RMSE $\mathbb{E}\big[|J_{\mathrm{sim}}(\pi) - J_{\mathrm{real}}(\pi)|\big]$、calibration curve、prediction interval coverage |
| 排序准不准（ranking） | Spearman $\rho_{\mathrm{rank}}$、Kendall $\tau$、top-k recall、regret@k |
| 选出的 policy 好不好（decision quality） | $R_{\mathrm{select}} = J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$（实际用 best-observed proxy） |

一个 simulator 可校得很准却选错 policy（distribution narrow）、也可数值全错但排序稳、regret 小——三维不能互替。$U_{\mathrm{sim}}$ 不该写成抽象标量、应展开成**按用途分类的 utility**：

$$U_{\mathrm{sim}} \;\in\; \big\{\ U_{\mathrm{pretrain}},\ U_{\mathrm{selection}},\ U_{\mathrm{exploration}},\ U_{\mathrm{curriculum}},\ U_{\mathrm{safety}}\ \big\}$$

评 fidelity 不能只盯单个 policy、而要相对**候选 policy family** 与**具体用途**：$U_{\mathrm{sim}}(\cdot \mid \Pi_{\mathrm{candidate}},\ p_{\mathrm{eval}}^{\mathrm{real}})$。

## 组合与决策，以及一个常被回避的问题

有了优先级、真实项目更有用的是 **gap × 可建模性 × 真机预算** 决策矩阵：

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
| mixed | mixed | mixed | co-training candidate（需先验证正迁移条件） |

**前两行的限定词不能省**：若 uncertainty 来自 **model-class uncertainty**（simulator 函数形式本身表达不了真实现象）、SI 与 DR 都未必适用、得先落到 residual / world model / 真机数据那几行。倒数第二行：光 "model unknown" 推不出 world model、判据是 **model uncertainty × real-data budget**——模型类不确定**且**真实交互充足时 learned world model 才合理。最后一行："co-training 兜底"与 allocation 冲突——sim 质量差、real 很少、action space / task semantics 不一致时完全可能负迁移。

常见组合 **SI → DR → DA → co-training / fine-tune**：**箭头只是示意、不是固定 workflow**——实际顺序由当前主导 gap 与边际效用决定。真实数据最有价值的用法往往不是**大量覆盖**、而是**发现 simulator 没建模的 failure mode**、让 sim synthetically 放大——

$$\text{discover real tail} \rightarrow \text{identify structure} \rightarrow \text{synthetically amplify} \rightarrow \text{real validation}$$

也就是 **real 用来发现、sim 用来放大、real 再用来验证**。

顺着这个逻辑、可以回答整篇几乎回避但框架本身允许的反问：**什么时候最优解其实是"不做 sim-to-real"？**
- **真机数据已便宜到 $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}^{\mathrm{effective}}$ 时**——$C_{\mathrm{real}}^{\mathrm{effective}}$ 是**有效真机成本**（含安全 / operator / reset / 磨损 / 失败恢复 / deployment 多样性）。正确比较是"当前预算 horizon 内的 expected cumulative value / cost"、不是"一次 intervention 的 raw hours"。
- **仿真器 model class 本身就差**（$\Delta_{\mathrm{model}}$ 主导且难参数化、如软体 / 流体 / 复杂接触）——修 sim 边际效用极低、不如走 world model 或真机数据。
- **部署分布非常固定**时——不需大规模 DR、少量 targeted real fine-tuning 往往更划算。
- **simulator 只提供"廉价产生与真实数据高度相关的数据"、不提供独特 coverage / safety / exploration / counterfactual access 时**——$U_{\mathrm{sim}}^{\mathrm{downstream}} < C_{\mathrm{sim}}^{\mathrm{effective}}$：不是"sim 不好"、而是"它没有提供 unique utility、opportunity cost 超过收益"。

能大方承认"有时最优解是不做 sim-to-real"、恰恰是 allocation framing 应有的样子：**不站"仿真"、只站"下一单位预算换回最多真实性能"。**

## 这意味着什么？：一个闭环，而不是一个开关

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)核心句是 evaluation-aware distribution allocation。套回 sim-to-real——**仿真数据的 utility 从来不是 simulator 的内部属性、而是相对于真实 evaluation distribution 的属性：**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

这解释了常见挫败："堆更多 sim 数据"有时没用——**当主要瓶颈恰好是 simulator 与真实 evaluation distribution 之间的 support / fidelity mismatch**、加同分布 samples 的边际收益会快速下降；**不能自动创造 evaluation-relevant coverage、也不能修正 model bias**。与其问"我的 sim 有多好"、不如问："我的 sim 在哪些 evaluation-relevant 方向上接近真实、哪些差得远？差得远的那些敏感度多高、用哪种预算压它最便宜？"

把这条线走完、sim-to-real 就不再是"能不能迁移成功"的开关、而是一条带反馈的闭环：

$$\boxed{\ \text{mismatch} \rightarrow \text{sensitivity} \rightarrow \text{intervention} \rightarrow \text{marginal utility} \rightarrow \text{budget allocation} \rightarrow \text{real evaluation} \rightarrow\ \circlearrowleft\ }$$

（最后一步会重新改变 sensitivity 与 mismatch 估计——见上文 feedback loop。）

这条链是一个 **resource-constrained adaptive sequential experimentation framework**：敏感度与边际收益都靠小步实验在真实评估上估出来、一轮估完再决定下一份预算投到哪。收束：**sim-to-real 不是 transfer 技巧、而是在 model fidelity / 训练多样性 / 表示对齐 / 真机交互与工程成本之间做 constrained、可迭代估计的 allocation 问题**。

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

sim-to-real 尚无公认的跨任务"哪种方法更强"定量对照——不同任务 / 硬件 / fidelity 上限下结论可能完全颠倒；上述工作更多是"这类 gap 用这个方法可行"的样本、而非可外推的排序。本文关于四个 intervention lens / 分析维度的分解、simulator utility 的三维分解是 allocation 视角的自然推论、error-budget constrained-allocation 的形式化、$\hat S_k^{\mathrm{int}}$ 与 $MV$ 的定义都是 **conceptual framework 与作者解读**：$\hat S_k^{\mathrm{int}}$、$MV$ 是靠 sensitivity experiments / ablation / 小规模真实评估估出的 decision statistics、不是 simulator 解析可求的量；把 co-training 读作 data-mixture、把 world model 读作 model-source replacement、同样不是受控实验证明的结论。

---

*本篇是"具身智能的数据问题"上下篇续篇：上篇讲数据来源与接口、下篇讲数据 scaling 框架；本篇把镜头拉到 sim-to-real、把它从"一堆迁移技巧"重述成带经验边际效用的闭环分配问题、接回下篇 sequential data allocation 主线。*
