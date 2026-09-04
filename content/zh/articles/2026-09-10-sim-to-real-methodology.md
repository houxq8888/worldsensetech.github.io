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

> 接[数据问题上篇](/zh/articles/2026-09-08-data-and-training-recipes/)与[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)。上篇把 sim-to-real 粗分四类工具、那只是 taxonomy、这一篇真正要回答的是——

> **当仿真数据在若干 evaluation-relevant 方向上离真实差得远时、下一单位预算（工程时间 / 算力 / 机器人小时）该花在哪条杠杆上：校准 sim、扩大训练分布、对齐表示、还是采真机数据？**

本文不提出新 sim-to-real algorithm、而是给出一个比较与组合既有 intervention 的 decision framework。按三层收敛：

- **Level 1 — Diagnosis**：$M_{\mathrm{sim}} \rightarrow p_{\mathrm{sim}}^\pi \rightarrow J_{\mathrm{sim}}$ vs. real——哪里不一样、哪里重要？
- **Level 2 — Intervention**：四个 lens（Model × Data × Representation × Optimization）——哪种干预改变这个 mismatch？
- **Level 3 — Allocation**：在当前 state、预算、uncertainty 下、下一块资源投哪？（decision score 与 efficiency reading 在 §Allocation 一节 formalize）

主线是 **Diagnosis → Experiment → Intervention → Allocation → Re-evaluation** 闭环。**三个 framework contributions + 一个 downstream corollary**：**(1)** reality gap 是 policy-conditioned consequence、非 sim 固有标量；**(2)** descriptor / sensitivity 只是诊断、决策变量是 intervention；**(3)** SI / DR / DA / FT 是 multi-resource sequential allocation 下的 intervention lenses；**corollary**——sim evaluation 从 fidelity 扩到 downstream utility。约束是 $B_{\mathrm{real}}, B_{\mathrm{compute}}, B_{\mathrm{eng}}$、预算花在干预动作上。

## Reality Gap：不是一个标量，而是一个 policy-conditioned 的 mismatch

Sim-to-real 常被叙述成"训练 policy 从 sim 迁移到 real"。更严格的起点是**两个分布**：同一条 $\pi$ 与两边环境交互各自诱导 $p_{\mathrm{sim}}^{\pi}(\tau)$ 与 $p_{\mathrm{real}}^{\pi}(\tau)$、一般不等：

$$p_{\mathrm{sim}}^{\pi}(\tau) \;\neq\; p_{\mathrm{real}}^{\pi}(\tau)$$

**"同一条 $\pi$"有前提**——sim 与 real 必须**共享同一 policy interface**：observation schema（键 / shape / 单位 / 归一化）、action schema（连续 or 离散、力矩 / 速度 / 位置、clamping）、control freq / action hold / delay。interface 不一致、$\pi$ 非同一函数、$\delta_J$ 失去定义。

轨迹分布本身 **policy-induced**、随 $\pi$ 变、非环境固有属性。真正关心的不是分布差、而是它在任务上**表现的后果**——同一 $\pi$ 两边的性能差：

术语分三层：**(a) distribution mismatch** $D(p_{\mathrm{sim}}^\pi, p_{\mathrm{real}}^\pi)$；**(b) transfer delta**

$$\boxed{\;\delta_J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)\;}$$

signed、真实反而更好时为正；**(c) performance discrepancy** $G_J(\pi) = |\delta_J(\pi)|$ 只谈幅度、下文敏感度用 $G_J$。**$J$ 默认越大越好、若是 cost 符号反转结构不变**。$\delta_J$ **不等于 reality gap 本身**、是 gap 在特定 $\pi$ + evaluation 下的 downstream consequence。

**distribution mismatch ≠ performance gap**：$p_{\mathrm{sim}}^{\pi} \neq p_{\mathrm{real}}^{\pi}$ 不自动意味 $\delta_J$ 很大——依赖粗粒度几何的 policy 换掉摩擦建模几乎不变、依赖高频力反馈的精细装配里此差异可能致命。真正影响 policy 的非 marginal $p(s)$、而是 **policy-conditioned occupancy** $d_{\mathrm{sim}}^{\pi}(s,a)$ vs $d_{\mathrm{real}}^{\pi}(s,a)$（contact-rich 还要加 contact-mode 索引）、逻辑链 $\pi \rightarrow d^\pi \rightarrow \text{mismatch} \rightarrow J$。严格写要把 **mechanism 与 induced distribution 分开**：

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ \mathcal{E}_{\mathrm{shared}};\ M_{\mathrm{sim}},\ M_{\mathrm{real}}\big)$$

$\mathcal{E}_{\mathrm{shared}}$ 是 **共同 evaluation 协议**——initial-state / horizon / reward / constraints 在 sim 与 real 两侧**必须相同**（$\mathcal{E}_{\mathrm{sim}} = \mathcal{E}_{\mathrm{real}} = \mathcal{E}_{\mathrm{shared}}$）、否则 $\delta_J(\pi) = J_{\mathrm{real}}(\pi) - J_{\mathrm{sim}}(\pi)$ 就不是"同一 task specification 下的 transfer consequence"。若 $\mathcal{E}_{\mathrm{sim}} \neq \mathcal{E}_{\mathrm{real}}$、观测到的性能差里已经混入 task-specification mismatch、**本文不将其计入 operational reality gap**。同一 $M_{\mathrm{sim}}$ 对 position control 可能 gap 很小、对 force-sensitive manipulation 可能巨大——**reality gap 是四元组 $(\pi, \mathcal{E}_{\mathrm{shared}}, M_{\mathrm{sim}}, M_{\mathrm{real}})$ 下的 downstream discrepancy、非 sim 固有标量**（operational definition）。

### gap 到底在哪里：reality mismatch 与 task-specification mismatch

第一步是把多源 gap 拆开——**两大类来源**、不能全塞进"reality"：

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

两类来源不能混为一谈：reality mismatch 是"仿真与真实非同一世界"、task-spec mismatch 是"优化目标不对齐"。**一般项目中直接观测到的性能差可能同时混合两类；但本文定义的 operational $\delta_J$ 在 $\mathcal{E}_{\mathrm{shared}}$ 固定之后，只讨论 reality mismatch 引起的 downstream discrepancy**（$\mathcal{E}_{\mathrm{sim}} \neq \mathcal{E}_{\mathrm{real}}$ 的部分已按上一条不计入 reality gap）。**两者可独立调节**——完美 sim 若 reward 与 deployment 不一致仍有 task-spec gap、反之 sim 有 bias 仍有 reality gap。本文 focus reality mismatch。

## 把"误差预算分配"写成一个可估计、可迭代优化的决策框架

拆完来源、给直觉一个数学落点。误差项强烈交互（sim 假设 proprioception 精确 + 真实有 latency、单看都不致命、叠加可让 controller 失稳）——更稳写法是承认存在 schematic 依赖 $F$：

$$\boxed{\;\delta_J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}}\big)\;}$$

**$\Delta_k$ 是 mismatch descriptor**（scalar / vector / distribution）；$F$ 只是 schematic、由 sensitivity / ablation 探测局部响应、**不是待估 predictive model**。**四个 $\Delta_k$ 是 diagnostic buckets、非正交 latent variables**——actuator delay 可伪装成 obs、contact 可伪装成 dynamics。层级：reality discrepancies → buckets → observable evidence → intervention candidates。全文里 $\mathcal{D}_t$ 一律表示 **allocator 在 step $t$ 可获得的全部 evidence**（calibration / ID 测量、sim 诊断、real paired evaluation、failure traces、safety observations 等），**不特指训练集**；$D_{\mathrm{train}}$、$D_{\mathrm{eval}}$ 保留独立记号、避免与 belief state 混用。

**$\Delta_{\mathrm{opt}}$ 从 reality gap 拿掉**（层级不同：同固定 policy、sim 观测动力学都准但 RL 未训好、$\delta_J$ 小而 policy 差）——分成**两个诊断量**：

$$\underbrace{J_{\mathrm{real}}(\pi_{\mathrm{train}}) - J_{\mathrm{sim}}(\pi_{\mathrm{train}})}_{\text{transfer delta } \delta_J}\qquad \underbrace{J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})}_{\text{real-domain learning gap}}$$

**不能无条件相加叫 deployment loss**（signed、baseline / 层级不同）；$\pi^{*}_{\mathrm{real}}$ 不可得、右侧是 oracle-defined 量、实际用 $J_{\mathrm{real}}(\pi_{\mathrm{best\text{-}validated}})$ proxy（其中 $\pi_{\mathrm{best\text{-}validated}}$ 指在**独立 audit / held-out 评估切片上表现最好**的 policy、而非 noisy eval 里 argmax 的那个、避免 winner's curse）。工程归因 $F$ 局部近似 $\delta_J \approx \sum_k w_k \Delta_k$ 只是 heuristic；真正 decision 用的是每类 mismatch 挑一 **intervention 变量** $\xi_k$ 后测得的 **intervention sensitivity**：

$$\hat S_k^{\mathrm{int}} \;\approx\; \frac{J_{\mathrm{real}}(\pi;\xi_k{+}\delta) - J_{\mathrm{real}}(\pi;\xi_k)}{\delta}$$

$\hat S_k^{\mathrm{int}}$ 叫 **local intervention response statistic**、非真导数；$\xi_k$ 是 experiment 人为定义的变量、分三档：**direct perturbation**（拨动 latency / friction）、**proxy / surrogate**（借 sim 估 calibration error）、**diagnostic ablation**。跨 $\xi_k$ 单位不同不可直接比较、且 sensitivity 还具有 **policy-conditional dependence**——同一 $\xi_k$ 下 $S_k^{\mathrm{int}}(\pi_1) \neq S_k^{\mathrm{int}}(\pi_2)$、默认只在当前 baseline policy / protocol 内做局部比较、跨 policy 不承诺可比。因此 **sensitivity 是 candidate generation / prioritization 层的统计、不是最终 allocation primitive**；**performance / cost / continuation——即 $\Delta J(m\mid s_t)$、$\Delta C(m\mid s_t)$、$\mathrm{CVU}(m\mid s_t)$——是 allocation score 的三类核心 value inputs、safety 与 budget 则定义 feasibility（$\mathcal{M}_t^{\mathrm{safe}}$、$\mathcal{M}_t^{\mathrm{budget}}$），allocation 只能回到这四者**。层级：**descriptor → sensitivity / uncertainty → candidate generation → $(\Delta J,\;\Delta C,\;\mathrm{CVU})$ + feasibility → $Q_{\lambda_t}$ → allocation**。

**诊断 ≠ 归因**：单 perturb $\Delta_{\mathrm{friction}}$ 与 $\Delta_{\mathrm{latency}}$ 各自影响很小、组合却可 $\Delta J(\Delta_f,\Delta_l) \gg \Delta J(\Delta_f,0) + \Delta J(0,\Delta_l)$（synergy）。**Sensitivity experiments 只识别 locally influential intervention directions、不提供 additive causal attribution**；$\Delta_{\mathrm{model}}$ 与 $\Delta_{\mathrm{ctrl}}$ 也可能互相补偿、都是 ablation 估的 decision statistics、非严格分解。

### 真正的"分配"：把钱花在干预动作上，而不是在方法里挑一个

预算**连续地**分到每条干预轴：$b=(b_1,\dots,b_K)$、$b_k$ 花在干预 $k$ 上（$b_{\mathrm{SI}}=2\text{h}$、$b_{\mathrm{DR}}=10^6$ 步 sim、$b_{\mathrm{real}}=4\text{h}$ 真机）、非 0/1 选择。**部署 objective 不能只看均值**——mean 90% + catastrophic 1% 与 mean 88% + tail ≈ 0 是**不同种部署决策**。本文采用 **mean utility + tail/safety constraint**（而非把 tail 直接折进 scalar cost）、除非项目显式引入 CVaR / risk-penalized utility $\max \mathbb{E}[J] - \gamma\,\mathrm{TailRisk}(J)$：

$$\max_{b}\quad \mathbb{E}\big[J_{\mathrm{real}}(\pi_b)\big] \quad \text{s.t.}\quad \Pr\big[\text{unsafe} \mid \pi_b\big] \le \alpha$$

项目预算**非同一种货币**（GPU 近乎无限 / 真机机时稀缺 / 有机器时间却没工程人力）、正解是多预算 $C_{\mathrm{real}} \le B_{\mathrm{real}}$、$C_{\mathrm{compute}} \le B_{\mathrm{compute}}$、$C_{\mathrm{eng}} \le B_{\mathrm{eng}}$、不折成标量 $B$。**安全不进同一层 cost**——是 chance constraint（$\alpha$ 由 e-stop / hardware fault 上限决定）。**本文默认 deployment $J$ 已固定单一 utility 或已外部 scalarization**；若保留多目标、应上升到 **Pareto 或 lexicographic layer**。

预算是分向量后、决策变量从"gap"换成"干预动作"——能买到 30 min SI / $10^6$ 步 sim / 100 条真机轨迹；**干预不直接改 $\Delta_k$、通过更新 state 改变后续决策**：

$$\boxed{\;s_{t+1} \;=\; \mathcal{T}\big(s_t,\; m_t^*,\; Y_t\big),\quad Y_t \sim p\!\big(Y \mid s_t,\, m_t^*\big)\;}$$
$\mathcal{T}$ 同时更新 $\pi_t$、$\mathcal{D}_t$、$b_t$、$h_t$——最核心一条是 budget dynamics $b_{t+1} = b_t - \Delta C(m_t^* \mid s_t)$；$b_t$ 是 **remaining budget vector**（不是 cumulative expenditure）。policy-changing intervention 更新 $\pi_t$ 与预算（早期写作 $\pi_{b+m} = \operatorname{Train}(D_{\mathrm{sim}}, D_{\mathrm{real}};\, m)$ 是其 budget-indexed shorthand、state-based 版本一律用 $\pi_t$）、diagnostic experiment 主要更新 $\mathcal{D}_t \cup Y_m$、model-update intervention 同时更新 sim / surrogate state。三类 action 统一到同一个 sequential framework。

**$Q_{\lambda_t}$（decision score）与 $MV$（效率读数、不是 decision rule）**。conditional on $s_t$。**「DR 的 $MV$」问错了**、正解是「当前 $s_t$ 下加一单位 DR 的 expected value」。$m = (\text{role},\,\text{method},\,\text{protocol},\,\text{batch})$——SI / DR / DA / FT 只是 method 标签、真正的 candidate 由 role × method × protocol/batch 三元决定。**Cost 也 state-conditioned**：$\Delta C(m \mid s_t) = (\Delta C_{\mathrm{real}}, \Delta C_{\mathrm{compute}}, \Delta C_{\mathrm{eng}})$——同一 DR batch 在 GPU 满载时空闲时不同 cost、同一 real FT 在高温机器人上可行性也不同；$C_\lambda(m \mid s_t) = \lambda_t^\top \Delta C(m \mid s_t)$。

$$\boxed{\;MV(m \mid s_t;\lambda_t) \;=\; \frac{\mu_{\Delta J,t}(m)}{\lambda_t^\top \Delta C(m \mid s_t)},\qquad \mu_{\Delta J,t}(m) \;=\; \mathbb{E}\big[\Delta J(m) \mid s_t\big]\;}$$
其中 $\widehat{\Delta J}_t(m)$ 表示**本次 paired evaluation 实际观测到的 empirical gain**、$\mu_{\Delta J,t}(m) = \mathbb{E}[\Delta J(m)\mid s_t]$ 表示**当前 belief 下对未来 intervention gain 的期望**——两者不混用、allocation 里出现的一律是 $\mu_{\Delta J,t}$、Step 4/6 报告的是 $\widehat{\Delta J}_t$。

**$U_0(m\mid s_t) = \mu_{\Delta J,t}(m) - \lambda_t^\top \Delta C(m \mid s_t)$ 才是典型意义上的 Lagrangian-style performance net utility；本文真正的局部 decision score $Q_{\lambda_t}$ 是在 $U_0$ 之上再叠加 one-step continuation heuristic 的合成量、**严格地说已经不再属于标准 Lagrangian 形式**。且要与 global objective 一样把 continuation value uplift（后文记作 $\mathrm{CVU}$）纳入（否则出现"全局含 $\mathrm{CVU}$、局部只算 performance"的近似断点）：

$$\boxed{\begin{aligned}
&U_0(m \mid s_t) \;=\; \mu_{\Delta J,t}(m) \;-\; \lambda_t^\top \Delta C(m \mid s_t)\\[2pt]
&\widehat V_0(s) \;:=\; \max_{m' \in \mathcal{M}^{\mathrm{feasible}}(s)}\, U_0(m' \mid s) \quad\text{(performance-only continuation baseline)}\\[2pt]
&Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t) \;=\; U_0(m \mid s_t) \;+\; \beta\,\mathrm{CVU}(m \mid s_t)
\end{aligned}\;}$$

$U_0$ 是 **performance-only Lagrangian-style net utility**、同时充当 $\mathrm{CVU}$ 的 reference；$\mathrm{CVU}(\cdot)$ 只通过 continuation baseline $\widehat V_0(\cdot) := \max_{m'} U_0(m'\mid\cdot)$ 引用 $U_0$、不引用 $Q_{\lambda_t}^{\mathrm{perf+CVU}}$ 本身，**避免 $Q \leftrightarrow \mathrm{CVU}$ 自我递归**（真要走 self-consistent 需 fixed-point、本文不走）。performance-only $Q_{\lambda_t}^{\mathrm{perf}} = U_0$ 是 special case；$MV = \mu_{\Delta J,t} / \lambda_t^\top \Delta C(m \mid s_t)$ 是 efficiency readout、对 **diagnostic-only** action 与 **只更新 simulator / surrogate、不重新训练 policy 的 model-update action** 都天生 immediate $\Delta J = 0$、$MV \equiv 0$——这不是遗漏、而是 $MV$ 只度量 performance efficiency、后续价值由 $\mathrm{CVU}$ 承担（见 allocation caveats）。$Q_{\lambda_t}^{\mathrm{perf+CVU}}$ 正式叫作 **local decision score**（**不是** RL 意义上的 Bellman action value、也**不是**标准 Lagrangian——Lagrangian 只属于 $U_0$ 那一层）、是当前 state 下可估计的一步局部近似。

**$\mathcal{M}_t = \mathcal{M}(s_t)$ state-dependent**——直接缩 $\mathcal{M}_t$、不是让 $MV$ 变小；$\mathcal{M}_t^{\mathrm{feasible}} = \mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}}$。Safety 分成**两个不同 event**：**execution-level** $\mathcal{M}_t^{\mathrm{safe}} = \{m : \mathrm{UCB}_{1-\delta}[P_{\mathrm{exec}}(\text{unsafe} \mid s_t, m)] \le \alpha_{\mathrm{exec}}\}$ gate 每次 candidate 本身会不会把机器人推到危险；**deployment-level** $P_{\mathrm{deploy}}(\text{unsafe} \mid \pi_T, \mathcal{E}_{\mathrm{shared}}) \le \alpha_{\mathrm{deploy}}$ 约束最终 policy 在部署分布下的 outcome。两者可共用同一个 $\alpha$、但 event 不混。$\mathrm{UCB}$ 由 empirical frequency 模型、posterior predictive risk、simulation + uncertainty bound 或 conservative reachability estimate 等 estimation layer 给出、Clopper–Pearson 只是 binary execution outcome 下的一个特例。**执行-level safety gate 只对涉及真实执行风险的 candidate 有非平凡意义**：纯计算 / 离线 diagnosis（例如 simulation-only diagnosis、offline calibration 或 model-refresh-only action）不驱动真机硬件、其 $P_{\mathrm{exec}}$ 可退化为 $0$ 或一个 deterministic feasibility check、$\mathcal{M}_t^{\mathrm{safe}}$ 对该子集自动满足。**严格而言** $\mathcal{D}_t$ 是 raw evidence / history、真正进入决策的是其 sufficient 压缩 $q_t = q(\mathcal{D}_t)$（posterior 或 belief state），formal state 因此可读作 $s_t = (b_t,\pi_t,q_t,h_t)$；**为简化 exposition、本文仍以 $\mathcal{D}_t$ 直接指代 allocator 可访问的 evidence state、并默认它已被压缩到 $q_t$ 层级、不要求保留 raw history**。$m$ 涵盖 policy-changing intervention 与 diagnostic experiment（role ∈ {adaptation, diagnosis, model update}，**role 指 action 的 primary operational purpose、允许同一 action 产生其他 side effects**，例如一次 SI measurement 可能同时是 diagnosis + 隐式 model update）；同 type 不同 batch / recipe / protocol 视为不同 candidate。

$$m_t^* \;=\; \arg\max_{m \,\in\, \mathcal{M}_t^{\mathrm{feasible}}}\; Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t),\qquad s_{t+1} = \mathcal{T}(s_t,\, m_t^*,\, Y_t)$$
时间索引完全对齐 transition：在 $s_t$ 决策 $m_t^*$ → 观察 $Y_t$ → 进入 $s_{t+1}$。**注意 L4 不仅是给出一个 scalar score、argmax 本身就通过 $m_t^* = \mu_t^Q(s_t)$ 诱导出一个 state-conditioned 的 approximate allocation policy $\mu_t^Q : s_t \mapsto m_t$**；下一轮的 $m_{t+1}$ 由 $\mu_{t+1}^Q(s_{t+1})$ 决定、因此 L4 与 L1 的 global adaptive allocation policy 在语义上首尾闭合（见下文 L1 讨论）。

$MV$ **不作 decision rule**——极简 toy 展示分岔（$\lambda = (3, 0.1, 1)$ 是 toy resource weight 常数取值；$\Delta C = (\text{real-h},\text{compute},\text{eng-h})$；toy **取 $\beta = 0$**、只隔离 $MV$ 与 performance-only net utility $Q_{\lambda_t}^{\mathrm{perf}} = U_0$ 的差别）：

| Intervention | $\mu_{\Delta J}$ | real h | compute | eng h | $C_\lambda$ | $MV$ | $Q_{\lambda_t}^{\mathrm{perf}}$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 min SI | 1.5 | 0.2 | 0.5 | 0.5 | 1.15 | **1.30** | 0.35 |
| Big DR batch | 3.0 | 0.0 | 20.0 | 0.4 | 2.40 | 1.25 | **2.00** |
| Camera DA | 2.5 | 1.0 | 3.0 | 0.5 | 3.80 | 0.66 | −1.30 |
| Real FT | 5.0 | 2.0 | 1.0 | 1.0 | 7.10 | 0.70 | −2.10 |

$MV$ 把 SI 顶第一、$Q_{\lambda_t}^{\mathrm{perf}}$ 把 DR 顶第一——这里的 $Q_{\lambda_t}^{\mathrm{perf}} = U_0$ 是正式定义 $Q_{\lambda_t}^{\mathrm{perf+CVU}}$ 在 $\beta = 0$ 下的特例（**示例刻意关掉 continuation-value 通道、只用来看效率 vs 净值的分离、不是本文主张 information 无用**）。域条件：**当 $\lambda_t^\top \Delta C > 0$ 时** $MV < 0 \Leftrightarrow \mu_{\Delta J,t} < 0$、$Q_{\lambda_t}^{\mathrm{perf}} < 0 \Leftrightarrow \mu_{\Delta J,t} < C_\lambda$；DA / FT 在此 $\lambda_t$ 下被 economic stop 排除。

$m_t^*$ 只是 **one-step local rule**、非 global optimum；L1 定义全局优化问题时、**优化变量应是 allocation policy 序列 $\{\mu_t\}_{t=1}^T$（其中 $\mu_t : s_t \mapsto m_t$ 是 state-conditioned 决策规则）、而不是 open-loop action sequence $\{m_t\}$**——若用后者读起来就是"一开始就把 $m_1, \ldots, m_T$ 全钉死"、恰恰违反本文反复强调的 **adaptive sequential experimentation**（$Y_t \to s_{t+1} \to$ 重新选 $m_{t+1}$）；若进一步写成 $\{m_t^*\}$ 则会把 L4 的 heuristic argmax 与 L1 的 global decision variable 混起来、层级绕回。**L1 给出累计资源约束 $\sum_t \Delta C_r(\mu_t(s_t) \mid s_t) \le B_r$、是 global feasibility 层；$b_t$ 则是同一份预算的在线 state representation、递推 $b_{t+1} = b_t - \Delta C(\mu_t(s_t) \mid s_t)$ 供 local decision 使用**——两层同时写不是重复、而是分别服务"整体是否可行"与"当下还剩多少"两个问题。**本文把 $\Delta C(m \mid s_t)$ 视为执行 candidate 后可观察到的 realized incremental resource consumption；因此 L1 的预算约束按"每一条执行路径上的累计 realized 消耗 $\le B_r$"来解释（pathwise cumulative constraint）、由于 $s_t$ 本身是随机状态、这实质是一个随机累计约束。若成本本身具有显著 stochasticity（机器人维修、unexpected engineering effort 等）、可将 L1 约束改为 expected 或 chance-constrained resource consumption、本文正文按 realized cost 陈述。** Safety 分两层：**execution-level** 由 $\mathcal{M}_t^{\mathrm{safe}}$ 每一步 gate（$P_{\mathrm{exec}}$）、**deployment-level** 只作为 terminal chance constraint 出现（$P_{\mathrm{deploy}}$）。完整问题：**multi-resource sequential allocation with chance constraint over an adaptive allocation policy**：

$$\boxed{\;\max_{\{\mu_t\}_{t=1}^{T}}\ \mathbb{E}\big[J_{\mathrm{real}}(\pi_T)\big] \quad \text{s.t.}\quad \sum_{t} \Delta C_r\!\big(\mu_t(s_t) \mid s_t\big) \le B_r\ (r \in \{\mathrm{real},\mathrm{compute},\mathrm{eng}\}),\;\; P_{\mathrm{deploy}}(\text{unsafe} \mid \pi_T, \mathcal{E}_{\mathrm{shared}}) \le \alpha_{\mathrm{deploy}},\;\; m_t = \mu_t(s_t).\;}$$

$\lambda_r$ **在具有良好值函数与约束正则性的情形下、可解释为最优值函数对 $B_r$ 的边际价值**（ideal shadow price）、实际本文只需 resource-weight estimate $\lambda_t = \lambda(b_t, \mathcal{D}_t, \pi_t)$、随 allocation state 更新；下文一律简称 **resource weights $\lambda_t$**。由于 $\lambda_t$ 本身 state-dependent、$s_{t+1}$ 里的 continuation 使用**更新后的** $\lambda_{t+1} = \lambda(b_{t+1}, \mathcal{D}_{t+1}, \pi_{t+1})$——因此 $U_0(m' \mid s_{t+1}) = \mu_{\Delta J, t+1}(m') - \lambda_{t+1}^\top \Delta C(m' \mid s_{t+1})$ 里的 shadow price 是**下一步**的、不是当前 $\lambda_t$ 的沿用，$s_t \to \lambda_t \to m_t \to s_{t+1} \to \lambda_{t+1}$ 才形成完整闭环。五条 caveat：**(i)** $Q_{\lambda_t}$ 默认 posterior mean、风险敏感可换 LCB / CVaR-adjusted utility——禁"公式 mean、文字 LCB"。**(ii)** SI fixed cost、DR diminishing returns、FT threshold、negative transfer 可让 $MV < 0$。**(iii) $\mathrm{CVU}$ 采用 counterfactual 定义、避免与 $U_0$ 重复计费当前 ΔC。** $\mathrm{CVU}(m\mid s_t)$（**one-step, candidate-relative, performance-only continuation-value uplift heuristic**）定义为**执行 $m$ 所导致的下一状态**与**同一时间推进下不执行 $m$ 的 counterfactual 下一状态**之间、continuation baseline $\widehat V_0(\cdot)$ 的差：
$$\mathrm{CVU}(m\mid s_t) \;=\; \mathbb{E}_{Y \sim p(\cdot\mid s_t, m)}\!\big[\widehat V_0(s_{t+1}^{m, Y})\big] \;-\; \widehat V_0(s_{t+1}^{\varnothing}),\qquad \widehat V_0(s) := \max_{m' \in \mathcal{M}^{\mathrm{feasible}}(s)} U_0(m' \mid s).$$
其中 $s_{t+1}^{m,Y}$ 是执行 $m$ 并观察 $Y$ 后的 state（携带 $b_{t+1} = b_t - \Delta C(m\mid s_t)$、$\pi_{t+1}$、$\mathcal{D}_{t+1}$、$h_{t+1}$、$\lambda_{t+1}$），$s_{t+1}^{\varnothing}$ 是**相同时间推进 / background drift convention 下不执行 $m$ 的 counterfactual state**（预算 $\Delta C(m)$ 未被扣、其余 convention 与执行 $m$ 时可比）。这样当前 action 的即时资源消耗 $\Delta C$ **只在 $U_0$ 里被扣一次**、$\mathrm{CVU}$ 只度量"$m$ 相对 $\varnothing$ 为未来 decision space 带来的额外 value"、不再叠加同一份 cost——即 **$U_0$ 处理当前 action 的 gain-cost、$\mathrm{CVU}$ 处理它对未来的边际影响**、二者互补。$\mathrm{CVU}$ 依然不是 VoI 也不是 Bellman continuation value function：$s_{t+1}$ 除了 evidence 更新之外、还携带 policy 改变、hardware drift、candidate-set 变化，标准 continuation term 应是 $V_{t+1}(s_{t+1})$、这里用 $\widehat V_0$ 代替，因此是 heuristic 而非真 value function。**$\mathrm{CVU}$ 可以为负**（budget depletion、hardware degradation、policy transition、candidate elimination、adverse evidence 都能让 $\widehat V_0(s_{t+1}^{m,Y})$ 低于 $\widehat V_0(s_{t+1}^{\varnothing})$）；**本文不再讨论"信息是不是有负价值"这类 VoI 语境问题**——$\mathrm{CVU}$ 就是净 continuation uplift、符号直接由上式给出。**Terminal convention**：在终止步 $T$ 之后没有下一步决策、约定 $\widehat V_{T+1}(s) := 0$、因此 terminal step 的 $Q_T = U_0(m\mid s_T) + \beta \cdot 0 = U_0$——不是"CVU 自动退化"、而是显式 convention。$\beta$ 是 dimensionless 偏好权重、若 $\mathrm{CVU}$ 与 $U_0$ 同尺度可令 $\beta=1$。$V(\mathcal{D}) = -\Pr(\arg\max Q_{\lambda_t}$ flips$)$ 只是 decision-stability proxy。**(iv)** $\Delta J$ 非天然 causal effect——matched / paired evaluation、$\Delta C$ 含全部 incremental cost；$\widehat{\Delta J}_t(m)$（realized）与 $\mu_{\Delta J,t}(m) = \mathbb{E}[\Delta J(m)\mid s_t]$（belief）两层次分开、公式中一律用 $\mu_{\Delta J,t}$。**(v) Diagnostic-only action 与"只更新 simulator / surrogate、暂不重新训练当前 policy 的 model-refresh action"** 均满足 immediate $\pi_t^m = \pi_t^{\mathrm{control}}$、$\mu_{\Delta J,t} = 0$（**注**：若 model update 内含"更新 model 后立刻重训 policy"、$\Delta J \neq 0$、此时应按 adaptation 处理）；因此 diagnostic / model-refresh 的 immediate value 恰好等于 $-\lambda_t^\top \Delta C$（即"净机会成本"）、其信息收益完全通过 $\mathrm{CVU}$ 侧的 counterfactual continuation 差体现——**$MV$ 对二者不提供有效信息（且当 incremental cost 也为 0 时 $MV$ 未定义）**、**在本文一步近似下、这些非即时 performance effects（evidence / hypothesis posterior / candidate space / safety feasibility / simulator quality 的改善）统一通过 $\mathrm{CVU}$ 汇总、不再单独定义额外 reward channel**。数值只在固定 $p_{\mathrm{eval}}$ 下有意义。

**$MV_i = MV_i(s_t)$ state-dependent**。先 SI 可使 DR $MV$ 下降、先 DR 可使 FT $MV$ 上升——**方向取决于 interaction、不假设单调**。intervention 之间有 complementarity / substitutability / conflict（不写成 bandit）。反馈层：**intervention 改 policy、进而改 $S_k^{\mathrm{int}}(\pi)$**：

```
estimate mismatch → estimate sensitivity → intervention
       ↑                                          ↓
   re-estimate  ←  sensitivity changes  ←  policy changes
```

这张 feedback loop 比任何新公式更贴合 allocation thesis：sim-to-real 是一轮做完重估一轮的 sequential experiment。

把每条干预对应到主要压缩项与主要预算：

| Intervention | 主要压缩项 | 主要预算 |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{eng}}$ + $C_{\mathrm{compute}}$ + 少量 $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | $C_{\mathrm{compute}}$（样本效率） |
| Residual physics | $\Delta_{\mathrm{model}}$（残差部分） | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$（appearance 子集） | $C_{\mathrm{real}}$（未标注数据）+ $C_{\mathrm{compute}}$ |
| Real-world fine-tuning | **不直接对应单一 mismatch；通过目标域 optimization 改变 policy**（同时改 transfer delta 与 learning gap） | $C_{\mathrm{real}}$（磨损 / 安全） |
| World model | 改变 model source | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Sim-and-real co-training | 改变 $p_{\mathrm{train}}$（$\Delta_{\mathrm{dist}}$ 为主） | 混合数据（$C_{\mathrm{real}}+C_{\mathrm{compute}}$） |

有了这套写法、全文就非"四种方法谁更好"、而是闭环：定位主导 $\Delta_k$、sensitivity 判重要度、$\arg\max Q_{\lambda_t}$ 选下一步、真实评估回报、再定下一份。

## 四个 intervention lenses（可组合的分析维度）

SI / DR / DA / FT **非同一抽象层级**——SI 是 model calibration、DR 是 distribution manipulation、DA 是 representation alignment、FT 是 optimization strategy——并排成"四类方法"会误导四选一、其实是**四个可组合的 intervention lens**（本文 analytical decomposition、非领域公认 ontology）：

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

"$\times$" 是组合空间、非正交——DR 触及 Model / Observation / Distribution、DA 可发生在多层。

选工具标准是**"点估计 → 后验 → 鲁棒随机化"连续谱**。SI 可以做 point calibration、也可以进一步给出 posterior；下面先写 point estimate：

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; \mathcal{L}_{\mathrm{ID}}\big(D_{\mathrm{real}},\ f_{\mathrm{sim}}(\cdot\,;\,\phi)\big)$$

$\mathcal{L}_{\mathrm{ID}}$ 可取 trajectory prediction / one-step transition error / force-torque residual / likelihood——**经典 SI 的目标通常是参数估计或 transition / observation prediction error minimization、而不必显式做 trajectory-distribution matching**。SI 处理的是**可参数化的 model mismatch**：动力学残差、接触/摩擦系数、延迟、相机外参等——若 gap 落在 model class 之外（未建模 long tail、语义级视觉差），SI 就力不从心、需换 DR / DA / WM。

| mismatch 的性质 | 更自然的工具 |
| --- | --- |
| 可参数化 + 可辨识 | System Identification（point estimate $\hat\phi$） |
| 可参数化但只能给出不确定性 | Bayesian / posterior SI → posterior-guided DR |
| 可参数化但难辨识 / uncertainty 大 | Domain Randomization |
| 难以由低维物理参数充分表达、但有结构化 residual | Residual learning |
| observation / appearance mismatch | Domain Adaptation |
| policy 在目标域仍有 systematic residual | Fine-tuning |

关键：**"不能精确辨识" ≠ "完全不知道"**——拿到 $p(\phi\mid D_{\mathrm{real}})$、最自然动作 $\phi\sim p(\phi\mid D)$ 做 posterior-guided DR、**SI 与 DR 是连续谱两端**。

### Axis A — Model：system identification、可微仿真与 residual physics

这条轴处理 $\Delta_{\mathrm{model}}$、三层次常被混淆：

$$y_t \;=\; \underbrace{g_{\mathrm{physics}}(x_t,a_t;\phi)}_{\text{可参数化的物理}} \;+\; \underbrace{r_\theta\big(\psi(x_t,a_t)\big)}_{\text{残差}} \;+\; \epsilon_t$$

**这只是 representative parameterization**——$y_t$ 可为 $x_{t+1}$、contact impulse、acceleration、deformation field 或其他 observable、$\psi$ 是 residual 的 input view；additive state-transition form 是一种 parameterization assumption、部分动力学更自然的 residual 是加在 acceleration 或 latent dynamics 上、而非 observable 本身。

- **可微仿真**解决 optimization interface、不解决 model class correctness；
- **Residual physics** 保留 prior、有限修正；
- **Full-learned dynamics** 处理 physics 不适用的场景。

**可微性在 discontinuous contact / friction cone 切换处有 gradient vanishing 风险**（需 soft contact）。工程判据：physics 结构基本正确、参数或边界不准时、可微仿真性价比最高。

Residual physics 一个常见的适用区间是 $f_{\mathrm{physics}}$ 已提供**结构性归纳偏置**、residual 只在目标分布上有限修正的场景。风险：sim 有 residual 补偿后看似好、到 OOD 失效——**residual model 的 valid domain 需与 deployment condition 对齐**。可微仿真在 contact-rich 场景受 contact mode switches / complementarity constraints 带来的非光滑与梯度不稳定问题掣肘；在 physics 结构基本正确、残差相对局域的条件下，可微仿真通常更值得优先评估。

### Axis B — Data distribution：domain randomization 及其家族

这条轴让 policy 对一族参数 $\{\phi\}$ 都稳健、不追求逼近最准 $p_{\mathrm{real}}$。**Tobin et al.（1703.06907）是现代深度视觉 / 机器人 sim-to-real 文献中的经典代表性起点**（domain randomization 思想本身更早、此处指其在端到端视觉 policy transfer 里的代表性位置）。

**DR 非"隐式 ensemble"**——训练的是单个共享 $\pi_\theta$、目标是：

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

更准确：**DR 是对一族环境模型做 population-level 优化**、risk-neutral average-case baseline；worst-case 可写 $\max_\theta\min_\phi$。过度 DR 让 policy 过于保守、牺牲 performance。

**DR 非选 scalar range、而是设计 joint distribution**——**当真实参数本身存在显著 joint dependency 时**、$p(\phi_1,\phi_2)\neq p(\phi_1)p(\phi_2)$、independent sampling 会把有限 sampling budget 分配到大量低 deployment relevance 或物理不一致组合；若真实参数本就近似独立、independent DR 反而是合理近似。**correlated / adversarial curriculum** 是 dependency 存在时的对策。

### Axis C — Observation / Representation：domain adaptation 与观测翻译

处理 $\Delta_{\mathrm{obs}}$。DA 可发生在 input / feature / latent / policy / dynamics 多层。机制包括 feature-level adapters、latent alignment、RCAN (1812.07252)。**不把 DA 压成 image translation**。边界：camera intrinsics / temporal sync 更适合 calibration、非 DA。

### Axis D — Optimization / adaptation：真机微调

这条轴**是 adaptation operator**：直接在目标域继续优化。可作前三轴收尾、也可作早期诊断（少量 FT 暴露哪些 mismatch 最伤 deployment）。

- **Offline / imitation：** $D_{\mathrm{real}} \to \theta$、主要成本是**采集**。
- **Online RL：** $\pi_\theta \to a \to$ 真实 transition $\to \theta'$、主要成本是**交互 + 安全 + 磨损 + 探索**。

比较不能只看最终 success rate、还要看**达目标所需真机交互预算**。粗略指标：

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{或}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

但只是**粗略指标**：依赖 baseline、非真 marginal efficiency。真正该看 learning curve / AULC / 每 100 条轨迹的边际收益

$$MV_{\mathrm{real}} \;\approx\; \frac{J(N+\Delta N)-J(N)}{\Delta N}$$

——这才与全文 $MV$ 框架接上。风险不止灾难性遗忘、更常见是**分布收窄**——真机数据比 sim 窄得多、微调后目标切片更好但鲁棒性反降、**generalization 换 specialization**；$MV_{\mathrm{real}}(N)$ **不保证始终为正**、**FT 本身可进入负边际收益区间**。

## 两条松动 environment-generating-process 假设的新路线

上面四条轴共享一隐含前提：经典 framing 把 simulator / real environment 视为**两个给定的 environment-generating processes**（对应分布 $p_{\mathrm{sim}}$、$p_{\mathrm{real}}$）。下面两条路线恰在松动这个前提——非"第五第六种技巧"、是整个问题的 reformulation：**前四条 lens 改变 intervention、WM 与 co-training 改变的是 intervention 所作用的 underlying training substrate**、不塞回同一 taxonomy。

### World model：不是取消 simulator，而是换掉 simulator 的来源

**本文 lens**：本节把 world model 读作"model source replacement"的 reformulation、只挑"相对 physics-sim 换掉 model 来源与 inductive bias"这个切面——不是 world model 的标准定义、也不声称这是唯一读法。

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)讨论过 world model 与 data utility。放进 sim-to-real 语境先纠正定位误读：**world model 不天然属于 sim-to-real**——两条路线 causal direction 不同：

```
Physics-sim route：  hand-designed dynamics  → train / optimize → deploy real
Learned-model route：interaction data → learned dynamics → imagine → optimize
```

**interaction data 可来自 real / sim 或混合**——learned-model route ≠ real-only。

需要说准：WM **并未取消 sim**、仍在做 simulation / imagination、只是 predictive model 是学出来的、更精确表述是**改变 predictive model 的来源与 inductive bias**：

$$\text{model source} = \text{physics prior} + \text{learned dynamics} + \text{data}$$

三者可 hybrid、不必是 $f_{\mathrm{hand}} \rightarrow f_{\mathrm{learned}}$ 的二元替换。Dreamer（1912.01603）、TD-MPC2（2310.16828）体现这条路——**人工 sim 的 model bias 大到不值得先修**时、WM 提供的是问题本身的改写。DayDreamer（2206.14176）常被误读成"sim 预训练 → real 微调"、更准是展示 **real-interaction-driven 实验路线**。**不依赖手工 sim ≠ model-free**、WM 仍吃假设、只是把 inductive bias 从显式 physics 移到 learned model。

诚实边界：contact-rich / long-tail 场景学到的 model 常在 OOD 给出很自信也很错的想象。**WM net value = predictive utility − model uncertainty risk**——uncertainty 必须进入一个 risk-aware decision layer、按部署需求可选 **hard feasibility gate**（$\Pr(\text{model-induced unsafe}) \le \alpha$）或 **soft risk penalty**（$U_{\mathrm{WM}} = U_{\mathrm{prediction}} - \gamma R_{\mathrm{model}}$）；只有 safety-critical deployment 才更适合前者。

### Sim-and-real co-training：把"迁移"重述成 data mixture

Maddukuri et al.（RSS 2025, 2503.24361）的 Sim-and-Real Co-Training 是务实方向。**论文报告**：sim + real 混合采样、两平台六视觉操作任务、相对**real-only baseline** 观测到**约 37.9% aggregate relative improvement**（across 6 tasks / 2 embodiments）——是**跨任务归一化的 relative lift**、非绝对百分点；引用务必带 baseline 与 aggregation 定义。不做单向迁移、而是一个 recipe 决定比例与调度。

读成 **data-mixture**——$p_{\mathrm{train}}=\alpha_{\mathrm{mix}} p_{\mathrm{sim}}+(1-\alpha_{\mathrm{mix}}) p_{\mathrm{real}}$；$\alpha_{\mathrm{sampling}} \neq \alpha_{\mathrm{effective}}$。Mechanistic 分析（Lei et al., 2604.13645）指出在该 generative robot policy 设置中 mixture 诱发 structured representation alignment——**paper-specific、不外推为 universal**。

## 评估：你怎么知道自己把 gap 补好了？

本文 claim 挂在**三级证据层**上——$\boxed{\text{A: mechanism}\quad \text{B: policy-response}\quad \text{C: deployment}}$：A 是 friction ID / calibration / latency measurement、B 是 $\hat S_k^{\mathrm{int}}$ / ablation / 有限差分 attribution、C 是真机 $\Delta J$ / $Q_{\lambda_t}$ / $MV$ / sim ranking utility。**三层是证据层级、非固定执行顺序**——诊断可循环（真机 failure → 怀疑 latency → 回测 A）；**不能互相替代**——SI 拟合属 A、不等于 C deployment 改善。

危险的做法是只在 sim benchmark 报性能。可信评估至少：

- 报 **zero-shot** 与 **few-shot / N-shot** 曲线；
- 用一组 **held-out hardware / object / contact / environmental regimes**；
- 明确声明 sim 与 real evaluation distribution 是否一致；
- **不只报均值**：mean ± CI、多 seeds、paired evaluation；
- **安全失败单独统计**：$X\sim\mathrm{Bin}(n,p),\;X=0$ 只能给 $p$ 的 UCB（Clopper–Pearson）。

顺着"sim 是真实世界的代理"、还有个比"数值对齐"更本质的问题：**sim 能否正确预测"哪个 policy 更好"？**

一个**概念性例子**（数值不代表实验结果）：

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

在 sim 上 $A > B > C$、真机却是 $B > C > A$。这时 simulator **失去 model-selection utility**——你会用它挑出最差的 policy。故 **simulator 用于 policy / model selection** 时应同时看排序相关性 $\rho_{\mathrm{rank}} = \mathrm{Spearman}(J_{\mathrm{sim}}(\pi_i), J_{\mathrm{real}}(\pi_i))$ 与 selection regret：

$$\pi_{\mathrm{sim}} = \operatorname*{arg\,max}_{\pi \in \Pi} J_{\mathrm{sim}}(\pi), \qquad R_{\mathrm{select}} = J_{\mathrm{real}}\big(\pi^{*}_{\mathrm{real}}\big) - J_{\mathrm{real}}\big(\pi_{\mathrm{sim}}\big)$$

**在更大的 policy pool 上**、即使 $\rho_{\mathrm{rank}} = 0.95$、top-1 仍可能被选错、灾难不减；反过来 $\rho_{\mathrm{rank}} = 0.7$、若 top-1 基本不出错、对"选一个能部署的 policy"就够用（**注意**：此处 $\rho_{\mathrm{rank}}$ 的直觉例子是更大 policy 集合上的相关性、而非上文 $A/B/C$ 三个 policy 的统计量——$n = 3$ 时 Spearman 只能取有限离散值、$0.95$ 那样的连续数字不适用）。**sim fidelity 是 task-of-use dependent、不是 absolute property**——换用途（pretrain / exploration / curriculum / safety filter）"哪些误差重要"整个变一遍。$\pi^*_{\mathrm{real}}$ 不可得、$R_{\mathrm{select}}$ 与 real-domain learning gap 一样都是 oracle-defined 量、实际用 $J_{\mathrm{real}}(\pi_{\mathrm{best\text{-}validated}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$ 作 **validated-best observed proxy**（$\pi_{\mathrm{best\text{-}validated}}$ 在独立 audit / held-out 切片上评估最优、不是 noisy eval 里的 argmax、避免 winner's curse）。

**更重要的一层**：真实项目通常不要求 sim 精确排序所有 policy、只要求把值得上真机的候选压到可接受集合——**top-$k$ recall**、**regret@k** 应与 ranking 同级。**警惕 adaptive selection bias**：sim 若被 adaptive filter policy、用同一批被选候选反过来评 sim 造成 self-confirming 循环。**维护两个 pool**——$\Pi_{\mathrm{adapt}}$ 参与 training / selection、$\Pi_{\mathrm{audit}}$ 只做 held-out evaluation。**held-out set 并非无限次免疫的**：长期项目应保留 audit slice 或定期 refresh evaluation set、避免 adaptive experimentation 过拟合固定真机评测集。

至此、**allocation framework 的一个 corollary**：**sim utility 不是单一属性、是三个不能互替的维度；sim 内部自洽、低 prediction loss 或高 training reward 不能单独证明 downstream utility——必须由独立 real evidence 验证**——

| Simulator utility 维度 | 典型 metric |
| --- | --- |
| 数值预测准不准（absolute error / calibration） | MAE / RMSE $\mathbb{E}\big[|J_{\mathrm{sim}}(\pi) - J_{\mathrm{real}}(\pi)|\big]$、calibration curve、prediction interval coverage |
| 排序准不准（ranking） | Spearman $\rho_{\mathrm{rank}}$、Kendall $\tau$、top-k recall、regret@k |
| 选出的 policy 好不好（decision quality） | $R_{\mathrm{select}} = J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$（实际用 **best-validated proxy**、独立 audit 切片上的 argmax、非 noisy eval 上的 argmax） |

一个 sim 可校得很准却选错 policy、也可数值全错但排序稳、regret 小——三维不能互替，**三类 metric 不仅量纲不同、优化目标也不同、因此不存在一个自然的 universal scalar simulator score**。$U_{\mathrm{sim}}$ 不该写成抽象标量、应按用途**上标索引**：$U_{\mathrm{sim}}^{(u)}$、$u \in \{\text{pretrain},\ \text{selection},\ \text{exploration},\ \text{curriculum},\ \text{safety}\}$。评 fidelity 要相对**候选 policy family** 与用途：$U_{\mathrm{sim}}^{(u)}(\cdot \mid \Pi_{\mathrm{candidate}},\ p_{\mathrm{eval}}^{\mathrm{real}})$。**更关键的是**：simulator 的最终 utility 不只看 $U_{\mathrm{sim}}^{(u)}$ 本身、还看它在 downstream allocation 里引发的期望价值——一个数值预测不够准、但能稳定做 candidate screening 的 sim、其 downstream allocation utility 可能仍很高。

## 组合与决策，以及一个常被回避的问题

真实项目更有用的是 **gap × 可建模性 × 真机预算** 矩阵：

| Gap | 可参数化 / 可辨识？ | Real data | 推荐 |
| --- | --- | ---: | --- |
| low-dimensional dynamics bias | 高 | 少 | SI |
| parameterizable dynamics uncertainty | 中 | 少 | posterior-guided DR / Bayesian SI → DR |
| dynamics residual | 低（但有结构） | 中 | Residual learning |
| visual appearance | 高 | 无 / 少 | DA / DR |
| actuator latency | 高 | 少 | SI + DR |
| unobserved rare tail、可被 model family 表示 | 低 | 少 | targeted simulation / DR |
| unknown long-tail，sim 生成不可信 | 低 | 中 | real data |
| model class 不确定 | 低 | 多 | learned world model（若 real 稀缺则先 physics prior + residual / DR） |
| mixed | mixed | mixed | co-training candidate（需先验证正迁移条件） |

**限定词不能省**：model-class uncertainty 下 SI 与 DR 未必适用、得先落到 residual / WM / 真机。"co-training 兜底"与 allocation 冲突——sim 质量差时可能负迁移。

常见组合 **SI → DR → DA → co-training / fine-tune**：**箭头只是示意、非固定 workflow**、顺序由主导 gap 与边际效用决定。**当 sim 已有较强 coverage、主要未知来自 model misspecification**、real data 的高价值用途是**发现 sim 未建模的 failure mode** 让 sim 放大；**若 deployment distribution 已相当固定**、real data 也可能主要承担直接 adaptation / imitation、不必先走 discovery / amplify——

$$\text{discover real tail} \rightarrow \text{identify structure} \rightarrow \text{amplify} \rightarrow \text{real validation}$$

即 **real 发现、sim 放大、real 再验证**。**这条 chain 成立有硬前提**：发现的 failure mode 能被当前 model class / learned surrogate 以可信方式表示；否则 real discover 之后应直接转向 richer model / world model / 追加 real data、而不是强把不可表示的 tail 塞进 sim 放大（这正是 model-class uncertainty 那一段的具体化）。

**什么时候最优解其实是"不做 sim-to-real"？**
- **真机数据已便宜到 $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}^{\mathrm{effective}}$**（比较的是 horizon 内 cumulative value / cost）。
- **仿真器 model class 本身就差**（软体 / 流体 / 复杂接触）——不如 WM 或真机数据。
- **部署分布非常固定**——少量 targeted real FT 更划算。
- **sim 不提供 unique coverage / safety / exploration / counterfactual access**——$U_{\mathrm{sim}}^{\mathrm{downstream}} < C_{\mathrm{sim}}^{\mathrm{effective}}$。

**stopping rule 三类**：**(a) economic stop**——$\max_{m \in \mathcal{M}_t^{\mathrm{feasible}}} Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t) \le 0$（local one-step stop、非全局最优——已知强互补 portfolio 应作为 candidate 一并评估）；**(b) continuation-value stop**——**best remaining positive continuation uplift 已经接近零**：$\max_{m \in \mathcal{M}_t^{\mathrm{feasible}}} \mathrm{CVU}(m \mid s_t) \le \varepsilon$（$\varepsilon$ 为小正阈值；**因 $\mathrm{CVU}$ 可正可负、"expected CVU 接近 0" 表述会漏掉"当前最优 CVU 严重为负、必须立即停"的情形、$\max$ 而非 expectation 才是正确的停止判据**）；**(c) safety / feasibility stop**——剩余 candidate 全在 feasible 集外。任一触发即停。

## 一个最小可执行的 Sim-to-Real Allocation Protocol

框架不落到"明天项目组怎么跑"、就还是聪明的 framing。以下 6 步是**最小可执行版**、可跳过、但跳之前要说清对本项目 no-op 的原因。

**Step 1 — 固定 evaluation。** 锁死 task / initial-state 分布 / horizon / success metric / safety threshold / policy interface（obs + action schema + control freq）。**若 $\pi$ stochastic（$a_t \sim \pi_\theta(\cdot \mid o_t)$）、$J(\pi)$ 应理解成 evaluation protocol 下对 policy / reset / hardware randomness 的期望**、用 repeated runs / block evaluation 估计。**没这一步、后面 $\Delta J$ 没有共同基准**。

**Step 2 — 建 held-out real evaluation set。** 真机 eval 集与训练数据**必须分开**、覆盖 held-out hardware / calibration / object / 场景切片。用训练数据 evaluate、$\widehat{\Delta J}$ 一定 optimistic。**但 eval 结果可进入 allocator 的 belief update**：$\mathcal{D}_t$ = "allocator 在 step $t$ 可获得的全部 evidence"、包括 $D_{\mathrm{eval}}$ 反馈的 failure mode 与 uncertainty 变化；"不参与 training" 与 "参与 posterior update" 是两件事、不冲突。

**Step 3 — 列 mismatch hypotheses（可 falsify）。**

| Hypothesis | Evidence | Belief | 候选 intervention |
| --- | --- | ---: | --- |
| friction $\mu$ 偏低 | contact slip | med | SI + DR |
| actuator latency 未建模 | 高频振荡 | high | SI + timing |
| camera extrinsics 偏 | grasp offset | high | Calibration / DA |
| contact model 错 | 柔性物体 OOD 失败 | low | Residual / WM |

每条 hypothesis **必须能被具体实验否证**、写不出否证条件的先剔除。

**Step 4 — one-time initial calibration pilot**（Step 5 才进入 sequential adaptive allocation）。**对会直接改变当前 policy 的 action 估计 immediate effect distribution**（$\mu_{\Delta J,t}(m)$ 与其 spread、Bayesian 实现下即 posterior、频率派实现下即 CI）、**对 diagnosis / model-refresh action 主要估计 evidence quality / continuation uplift distribution**（其 immediate $\Delta J \equiv 0$、不存在 meaningful 的 immediate effect distribution 可估、experimental target 是 $\mathrm{CVU}$ 相关的 evidence 与后验改善量）；不预设固定样本数。**$\widehat{\Delta J}_t(m) = J_{\mathrm{real}}(\pi_t^{m}) - J_{\mathrm{real}}(\pi_t^{\mathrm{control}})$**——control 承担相同 training 步数、**相同 elapsed time（覆盖机器人温度 / 电量 / wear 等 background drift）**、相同 training seed（真机硬件扰动本身没有"seed"可对齐、只能靠 matched evaluation block 逼近），只关掉本 intervention；**diagnostic-only action 与"只更新 simulator / surrogate、暂不重新训练当前 policy 的 model-refresh action"的 $\widehat{\Delta J}_t \equiv 0$、其价值在本文一步近似中统一通过 $\mathrm{CVU}$ 汇总（unified continuation surrogate）、不再单独定义额外 reward channel**。**matched / paired / block 化评估**：同 training seed、并在可行时采用 matched evaluation blocks / hardware conditions、同 held-out slice；漂移系统记录 hardware state。**单 intervention matched control 识别的是 incremental effect relative to the current protocol、不识别高阶 interaction effect；组合 action（例如 SI+DR 或 SI+WM refresh）需作为独立 candidate 做 matched comparison**，否则 synergy / conflict 无法从数据里分离。

**Step 5 — sequential adaptive allocation**：$m_t^* = \arg\max_{m\in\mathcal{M}_t^{\mathrm{feasible}}(s_t)} Q_{\lambda_t}^{\mathrm{perf+CVU}}(m\mid s_t)$——$\lambda_t$ 是 resource-weight estimate、objective 与 local score 必须一致；cost 与预算同时 state-conditioned（$\Delta C(m\mid s_t)$、$b_{t+1} = b_t - \Delta C(m_t^*\mid s_t)$）。Execution-level safety 走 $\alpha_{\mathrm{exec}}$ gate、deployment-level safety 走 $\alpha_{\mathrm{deploy}}$ terminal 约束、都不进 cost。

**Step 6 — real evaluation → posterior update → 回到 Step 3。** 更新 $\mathcal{D}_t \rightarrow \mathcal{D}_{t+1}$、重估 $\lambda_t$、淘汰否证 hypothesis、新失败补入表。**最易跳过、最关键**——没 posterior update、流程退化为静态 checklist。

**定位**：最低落地版——小团队可合并 Step 3/4、大团队可加 portfolio opt。6 步都要写下来。

## 这意味着什么？：一个闭环，而不是一个开关

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)核心句是 evaluation-aware distribution allocation。套回 sim-to-real——**仿真数据 utility 不是 sim 内部属性、而是相对真实 evaluation distribution 的属性：**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

这解释了常见挫败："堆更多 sim 数据"有时没用——**当主要瓶颈恰好是 simulator 与真实 evaluation distribution 之间的 support / fidelity mismatch**、加同分布 samples 的边际收益会快速下降；**不能自动创造 evaluation-relevant coverage、也不能修正 model bias**。与其问"我的 sim 有多好"、不如问："我的 sim 在哪些 evaluation-relevant 方向上接近真实、哪些差得远？差得远的那些敏感度多高、用哪种预算压它最便宜？"

把这条线走完、sim-to-real 就不再是"能否迁移成功"的开关、而是带反馈的闭环：

$$\boxed{\ \text{diagnosis} \rightarrow \text{sensitivity / uncertainty} \rightarrow \text{intervention} \rightarrow \text{performance} + \text{information gains} \rightarrow \text{update }\mathcal{D}_t \rightarrow \text{re-allocate} \rightarrow\ \circlearrowleft\ }$$

配套的**闭环 spine 图**、比任何"四分类"表格都更贴合本文论点：

```text
                current state  s_t = (b_t, π_t, D_t, h_t)   ← b_t = remaining budget
                                  │
                                  ▼
                     mismatch diagnosis (D_t vs real)
                                  │
                                  ▼
                 sensitivity / uncertainty attribution
                                  │
                                  ▼
         candidate action  m = (role, lens, protocol/batch)
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         diagnosis         adaptation       model update
      (any lens:        (Model / Data /    (Model / Data /
       Model / Data…)    Representation /   Representation /
                         Optimization)      Optimization)
              └────────────────┼────────────────┘
                               ▼
                  real evaluation (paired, CI)
                               │
                    performance uplift  +  continuation uplift
                     (ΔJ_real)          (CVU: D_t, π_t, M_{t+1})
                               │
                               ▼
                update  s_{t+1} = T(s_t, m_t^*, Y_t),  Y_t ~ p(·|s_t, m_t^*)
                               │
                               ▼
                    stopping rule?  → deploy π_T
                               │
                               └────► next round (loop)
```

role × lens × protocol/batch **三元索引**共同定义 candidate——**它们是不同语义层的分类维度、不假设统计正交或物理独立**：role 描述"这次 action 在 sequential loop 中的 primary operational purpose"（adaptation / diagnosis / model update、允许同时产生其他 side effects）、lens 描述"通过什么机制干预"（Model / Data / Representation / Optimization、**"×" 是组合空间、本身也非正交**）、protocol/batch 描述"以什么规模与配方执行"。SI / DR / DA / FT 只是 method 标签、一个 method 可以落到不同 role（SI 既可以是 adaptation、也可以是纯诊断性 measurement、甚至可以顺带刷新 sim 参数 = primary role 是 diagnosis、side effect 是 model update），因此 method 不是 allocation 的 action space。真正的 decision unit 是 state-conditioned action $m \in \mathcal{M}_t^{\mathrm{feasible}}(s_t)$。**candidate set 本身也 state-dependent**：diagnostic experiment 关掉或打开后续 adaptation / model-update 的可行域、把 evidence 与 action set 的耦合直接暴露在读图上；最后一步改变 $s_{t+1}$ 里的 sensitivity 与 mismatch、驱动 feedback loop。

这条链是 **resource-constrained adaptive sequential experimentation framework**：敏感度与边际收益靠小步实验估出、一轮估完再定下一份预算。收成五层 spine：

$$\boxed{\begin{aligned}
&\textbf{L1}:\ \max_{\{\mu_t\}_{t=1}^{T}}\ \mathbb{E}[J_{\mathrm{real}}(\pi_T)]\quad\text{s.t.}\ \textstyle\sum_t \Delta C_r(\mu_t(s_t)\mid s_t)\le B_r,\ m_t=\mu_t(s_t),\ P_{\mathrm{deploy}}(\text{unsafe}\mid \pi_T, \mathcal{E}_{\mathrm{shared}})\le \alpha_{\mathrm{deploy}}\\
&\textbf{L2}:\ s_t = (b_t,\,\pi_t,\,\mathcal{D}_t,\,h_t),\quad b_t\ \text{= remaining},\quad b_{t+1} = b_t - \Delta C(m_t\mid s_t)\\
&\textbf{L3}:\ m_t \in \mathcal{M}_t^{\mathrm{feasible}}(s_t),\quad \mathcal{M}_t^{\mathrm{feasible}} = \mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}},\ \mathcal{M}_t^{\mathrm{safe}}\ \text{gates}\ P_{\mathrm{exec}}\\
&\textbf{L4}:\ m_t^* = \arg\max_{m\in\mathcal{M}_t^{\mathrm{feasible}}} Q_{\lambda_t}^{\mathrm{perf+CVU}}(m\mid s_t),\quad Q_{\lambda_t}^{\mathrm{perf+CVU}} = U_0(m\mid s_t) + \beta\,\mathrm{CVU}(m\mid s_t)\\
&\textbf{L5}:\ MV(m\mid s_t) = \mu_{\Delta J,t}(m)\;/\;\lambda_t^\top \Delta C(m\mid s_t),\quad \mu_{\Delta J,t}(m) = \mathbb{E}[\Delta J(m)\mid s_t]\\
&\textbf{Transition}:\ s_{t+1} = \mathcal{T}(s_t,\, m_t^*,\, Y_t),\quad Y_t \sim p(\cdot\mid s_t, m_t^*),\quad \lambda_{t+1} = \lambda(b_{t+1},\mathcal{D}_{t+1},\pi_{t+1})\\
&\textbf{Terminal}:\ \widehat V_{T+1}(s) := 0,\ \text{hence } Q_T = U_0(m\mid s_T)
\end{aligned}\;\longrightarrow\;\circlearrowleft}$$

层级：$\boxed{\text{global } \mathbb{E}[J_T] \supset \text{local } Q_{\lambda_t} \supset MV}$——**L1 定义的是优化问题本身（stochastic sequential allocation with chance constraint）、L4 只是它的一个 tractable action-selection approximation**（$m_t^* = \arg\max\,(U_0 + \beta\,\mathrm{CVU})$ 是 global sequential allocation 的 **one-step, performance-only continuation approximation**、不是 Bellman-style exact solution、$V_{t+1}(s_{t+1})$ 用 $\max_{m'} U_0(m' \mid s_{t+1})$ 替代）、$MV$ 是 efficiency statistic。**$\mathrm{CVU}$ 只用于中间决策 continuation look-ahead、不是 terminal deployment reward；到停止时刻、最终 objective 仍只评价 $J_{\mathrm{real}}(\pi_T)$ 与 $P_{\mathrm{deploy}}$ 约束**——因此 terminal step 的 $Q_{\lambda_t}$ 退化为 $U_0$。

收束：**sim-to-real 不是选一种 transfer technique、而是在当前 belief、不可互换预算与真实评估反馈下连续决定下一次 intervention**——这是全文理论 spine。**本文的贡献不是提出新的 optimization primitive、而是重新定义 sim-to-real 的 decision unit**——从「选一种方法」到「在当前 state 下选下一次 intervention」——并把 reality gap 与 sim utility 重述成 policy / evaluation-conditioned 的量。所有前文展开的符号最后收成一条主链、两条封口、以及一张 allocation primitive hierarchy：
$$\boxed{\begin{gathered}
\text{state } s_t \;\rightarrow\; \mathcal{M}_t^{\mathrm{feasible}}(s_t) \;\rightarrow\; \big(\mu_{\Delta J,t},\;\Delta C,\;\mathrm{CVU}\big) \;\rightarrow\; Q_{\lambda_t} \;\rightarrow\; m_t^* \;\rightarrow\; s_{t+1}\\[4pt]
\text{safety / budget define feasibility;}\\[-2pt]
MV\ \text{is only an efficiency diagnostic, not a decision rule.}
\end{gathered}}$$
**主链上不再显式画 $\mu_t$**——因为 L1 优化的是 ideal **global adaptive allocation policy** $\{\mu_t\}_{t=1}^T$、L4 通过 argmax 构造的是 computable **approximate allocation policy** $\mu_t^Q(s_t) := m_t^* = \arg\max_m Q_{\lambda_t}(m \mid s_t)$、两者分处 ideal / approximate 两层；若把 $\mu_t$ 与 $m_t^*$ 同时画进主链、就会被读成 "$\mu_t$ 先生成 $m_t$、然后 Q 再挑出 $m_t^*$" 的伪循环。整个 framework 的 allocation primitives 自上而下收成七层：
$$\boxed{\begin{array}{rcl}
\text{diagnostic layer} &:& \Delta_k,\ S_k^{\mathrm{int}},\ \text{uncertainty}\\[2pt]
\downarrow &&\\[2pt]
\text{candidate construction} &:& m = (\text{role},\,\text{lens},\,\text{protocol}/\text{batch})\\[2pt]
\downarrow &&\\[2pt]
\text{value estimation} &:& (\mu_{\Delta J,t},\ \Delta C,\ \mathrm{CVU})\\[2pt]
\downarrow &&\\[2pt]
\text{feasibility} &:& P_{\mathrm{exec}},\ \text{budget}\\[2pt]
\downarrow &&\\[2pt]
\text{local decision} &:& Q_{\lambda_t}\\[2pt]
\downarrow &&\\[2pt]
\text{action} &:& m_t^*\\[2pt]
\downarrow &&\\[2pt]
\text{state transition} &:& s_{t+1}
\end{array}}$$
读者只需要抓住 **五个主对象**（$s_t$、$\mathcal{M}_t^{\mathrm{feasible}}$、$Q_{\lambda_t}$、$m_t^*$、$s_{t+1}$）、**一条 feasibility 规则**（$\mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}}$ 与 $P_{\mathrm{exec}} / P_{\mathrm{deploy}}$ 双层）、以及 **$MV$ 只提供 efficiency 读数不充当 decision rule** 的分工，就把整套 formal framework 收在一张纸上；$(\mu_{\Delta J,t}, \Delta C, \mathrm{CVU})$ 是 $Q_{\lambda_t}$ 的三个 value inputs、$m_t$ 只是 generic action 记号、**均不再另列为一级对象**。

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

sim-to-real 尚无公认跨任务定量对照、不同任务 / 硬件 / fidelity 上限下结论可颠倒；上述工作更多是"这类 gap 用这方法可行"的样本、非可外推排序。本文四个 lens 分解、sim utility 三维切分、constrained-allocation 形式化都是 **conceptual framework 与作者解读**、非受控实验证明的结论。

---

*本篇是"具身智能的数据问题"上下篇续篇：上篇讲数据来源与接口、下篇讲数据 scaling 框架；本篇把镜头拉到 sim-to-real、把它从"一堆迁移技巧"重述成带经验边际效用的闭环分配问题、接回 sequential data allocation 主线。*
