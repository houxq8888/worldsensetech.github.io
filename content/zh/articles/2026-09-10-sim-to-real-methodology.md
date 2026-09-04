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

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ \mathcal{E};\ M_{\mathrm{sim}},\ M_{\mathrm{real}}\big)$$

$\mathcal{E}$ 是 evaluation 假设集合（initial-state / horizon / reward / constraints）；同一 $M_{\mathrm{sim}}$ 对 position control 可能 gap 很小、对 force-sensitive manipulation 可能巨大。**本文把 reality gap 操作性视为四元组 $(\pi,\mathcal{E},M_{\mathrm{sim}},M_{\mathrm{real}})$ 下的 downstream discrepancy、非 sim 固有标量**（operational definition）。

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

两类不能相加：reality mismatch 是"仿真与真实非同一世界"、task-spec mismatch 是"优化目标不对齐"；$\delta_J$ 混合两类。**两者可独立调节**——完美 sim 若 reward 与 deployment 不一致仍有 gap、反之 sim 有 bias 仍有 gap。本文 focus reality mismatch。

## 把"误差预算分配"写成一个可估计、可迭代优化的决策框架

拆完来源、给直觉一个数学落点。误差项强烈交互（sim 假设 proprioception 精确 + 真实有 latency、单看都不致命、叠加可让 controller 失稳）——更稳写法是承认存在 schematic 依赖 $F$：

$$\boxed{\;\delta_J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}}\big)\;}$$

**$\Delta_k$ 是 mismatch descriptor**（scalar / vector / distribution）；$F$ 只是 schematic、由 sensitivity / ablation 探测局部响应、**不是待估 predictive model**。**四个 $\Delta_k$ 是 diagnostic buckets、非正交 latent variables**——actuator delay 可伪装成 obs、contact 可伪装成 dynamics。层级：reality discrepancies → buckets → observable evidence → intervention candidates。

**$\Delta_{\mathrm{opt}}$ 从 reality gap 拿掉**（层级不同：同固定 policy、sim 观测动力学都准但 RL 未训好、$\delta_J$ 小而 policy 差）——分成**两个诊断量**：

$$\underbrace{J_{\mathrm{real}}(\pi_{\mathrm{train}}) - J_{\mathrm{sim}}(\pi_{\mathrm{train}})}_{\text{transfer delta } \delta_J}\qquad \underbrace{J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})}_{\text{real-domain learning gap}}$$

**不能无条件相加叫 deployment loss**（signed、baseline / 层级不同）；$\pi^{*}_{\mathrm{real}}$ 不可得、右侧是 oracle-defined 量、实际用 $J_{\mathrm{real}}(\pi_{\mathrm{best\text{-}observed}})$ proxy。工程归因 $F$ 局部近似 $\delta_J \approx \sum_k w_k \Delta_k$ 只是 heuristic；真正 decision 用的是每类 mismatch 挑一 **intervention 变量** $\xi_k$ 后测得的 **intervention sensitivity**：

$$\hat S_k^{\mathrm{int}} \;\approx\; \frac{J_{\mathrm{real}}(\pi;\xi_k{+}\delta) - J_{\mathrm{real}}(\pi;\xi_k)}{\delta}$$

$\hat S_k^{\mathrm{int}}$ 叫 **local intervention response statistic**、非真导数；$\xi_k$ 是 experiment 人为定义的变量、分三档：**direct perturbation**（拨动 latency / friction）、**proxy / surrogate**（借 sim 估 calibration error）、**diagnostic ablation**。跨 $\xi_k$ 单位不同不可直接比较、allocation 只能回到 $\Delta J$ 与 $\lambda^\top \Delta C$。

**诊断 ≠ 归因**：单 perturb $\Delta_{\mathrm{friction}}$ 与 $\Delta_{\mathrm{latency}}$ 各自影响很小、组合却可 $\Delta J(\Delta_f,\Delta_l) \gg \Delta J(\Delta_f,0) + \Delta J(0,\Delta_l)$（synergy）。**Sensitivity experiments 只识别 locally influential intervention directions、不提供 additive causal attribution**；$\Delta_{\mathrm{model}}$ 与 $\Delta_{\mathrm{ctrl}}$ 也可能互相补偿、都是 ablation 估的 decision statistics、非严格分解。

### 真正的"分配"：把钱花在干预动作上，而不是在方法里挑一个

预算**连续地**分到每条干预轴：$b=(b_1,\dots,b_K)$、$b_k$ 花在干预 $k$ 上（$b_{\mathrm{SI}}=2\text{h}$、$b_{\mathrm{DR}}=10^6$ 步 sim、$b_{\mathrm{real}}=4\text{h}$ 真机）、非 0/1 选择。**部署 objective 不能只写均值**——mean 90% + catastrophic 1% 与 mean 88% + tail ≈ 0 是**不同种部署决策**、写法是 mean-plus-tail + safety constraint：

$$\max_{b}\quad \mathbb{E}\big[J_{\mathrm{real}}(\pi_b)\big] \quad \text{s.t.}\quad \Pr\big[\text{unsafe} \mid \pi_b\big] \le \alpha$$

项目预算**非同一种货币**（GPU 近乎无限 / 真机机时稀缺 / 有机器时间却没工程人力）、正解是多预算 $C_{\mathrm{real}} \le B_{\mathrm{real}}$、$C_{\mathrm{compute}} \le B_{\mathrm{compute}}$、$C_{\mathrm{eng}} \le B_{\mathrm{eng}}$、不折成标量 $B$。**安全不进同一层 cost**——是 chance constraint（$\alpha$ 由 e-stop / hardware fault 上限决定）。**本文默认 deployment $J$ 已固定单一 utility 或已外部 scalarization**；若保留多目标、应上升到 **Pareto 或 lexicographic layer**。

预算是分向量后、决策变量从"gap"换成"干预动作"——能买到 30 min SI / $10^6$ 步 sim / 100 条真机轨迹；**干预不直接改 $\Delta_k$、通过更新 state 改变后续决策**：

$$\boxed{\;s_{t+1} \;=\; \mathcal{T}\big(s_t,\; m_t,\; Y_t\big),\quad Y_t \sim p\!\big(Y \mid \mathcal{D}_t,\, m_t\big)\;}$$
对于 policy-changing intervention、$\mathcal{T}$ 更新 $\pi_t$ 与预算、此时 $\pi_{b+m} = \operatorname{Train}(D_{\mathrm{sim}}, D_{\mathrm{real}};\, m)$ 是其**特例**；对于 diagnostic experiment、主要更新 $\mathcal{D}_t \cup Y_m$；对于 model-update intervention、同时更新 sim / surrogate state。三类 action 统一到同一个 sequential framework。

**$Q_\lambda$（decision score）与 $MV$（效率读数、不是 decision rule）**。conditional on $s_t$。**「DR 的 $MV$」问错了**、正解是「当前 $s_t$ 下加一单位 DR 的 expected value」。$m = (\text{type}, \Delta b_m)$、是可执行 batch。$\Delta C(m) = (\Delta C_{\mathrm{real}}, \Delta C_{\mathrm{compute}}, \Delta C_{\mathrm{eng}})$、$C_\lambda = \lambda^\top \Delta C(m)$。

$$\boxed{\;MV(m \mid s_t;\lambda) \;=\; \frac{\mathbb{E}\big[\,J_{\mathrm{real}}(\pi_{b'}) - J_{\mathrm{real}}(\pi_{b}) \;\big|\; \mathcal{D}_t\,\big]}{\lambda^\top \Delta C(m)}\;}$$

**真正的局部 decision score 是 Lagrangian net value**、且要与 global objective 一样把 VoI 纳入（否则出现"全局含 VoI、局部只算 performance"的近似断点）：

$$\boxed{\begin{aligned}
&U_0(m \mid \mathcal{D}_t) \;=\; \mathbb{E}\big[\Delta J(m) \mid \mathcal{D}_t\big] \;-\; \lambda^\top \Delta C(m)\\[2pt]
&Q_\lambda^{\mathrm{perf+info}}(m \mid s_t) \;=\; U_0(m \mid \mathcal{D}_t) \;+\; \beta\,\mathrm{VoI}(m \mid \mathcal{D}_t)
\end{aligned}\;}$$

$U_0$ 是 **performance-only reference utility**——$\mathrm{VoI}(\cdot)$ 只引用 $U_0$、不引用 $Q_\lambda^{\mathrm{perf+info}}$ 本身，**避免 $Q \leftrightarrow \mathrm{VoI}$ 自我递归**（真要走 self-consistent 需 fixed-point、本文不走）。performance-only $Q_\lambda^{\mathrm{perf}} = U_0$ 是 special case、$MV = \mathbb{E}[\Delta J] / \lambda^\top \Delta C$ 是效率读数。$Q_\lambda^{\mathrm{perf+info}}$ **是 current-state local score、非含未来 option value 的 Bellman-optimal action value**、可估计的一步局部近似。

**$\mathcal{M}_t = \mathcal{M}(s_t)$ state-dependent**——直接缩 $\mathcal{M}_t$、不是让 $MV$ 变小；$\mathcal{M}_t^{\mathrm{feasible}} = \mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}}$。$\mathcal{M}_t^{\mathrm{safe}} = \{m : \mathrm{UCB}_{1-\delta}[\Pr(\text{unsafe})] \le \alpha\}$——$\alpha$ = allowed failure probability、$\delta$ = statistical confidence tail。$m$ 涵盖 policy-changing intervention 与 diagnostic experiment（role ∈ {adaptation, diagnosis, model update}）；同 type 不同 batch / recipe / protocol 视为不同 candidate。

$$m_{t+1} \;=\; \arg\max_{m \,\in\, \mathcal{M}_t^{\mathrm{feasible}}}\; Q_\lambda^{\mathrm{perf+info}}(m \mid s_t)$$

$MV$ **不作 decision rule**——极简 toy 展示分岔（$\lambda = (3, 0.1, 1)$ 是 resource weight；$\Delta C = (\text{real-h},\text{compute},\text{eng-h})$）：

| Intervention | $\Delta J$ | real h | compute | eng h | $C_\lambda$ | $MV$ | $Q_\lambda$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 min SI | 1.5 | 0.2 | 0.5 | 0.5 | 1.15 | **1.30** | 0.35 |
| Big DR batch | 3.0 | 0.0 | 20.0 | 0.4 | 2.40 | 1.25 | **2.00** |
| Camera DA | 2.5 | 1.0 | 3.0 | 0.5 | 3.80 | 0.66 | −1.30 |
| Real FT | 5.0 | 2.0 | 1.0 | 1.0 | 7.10 | 0.70 | −2.10 |

$MV$ 把 SI 顶第一、$Q_\lambda$ 把 DR 顶第一。$MV < 0 \Leftrightarrow \mathbb{E}[\Delta J] < 0$、$Q_\lambda < 0 \Leftrightarrow \mathbb{E}[\Delta J] < C_\lambda$；DA / FT 在此 $\lambda$ 下被 economic stop 排除。

$m_{t+1}$ 只是 **one-step local rule**、非 global optimum。完整问题：**multi-resource sequential allocation with chance constraint**：

$$\max_{\{m_t\}_{t=1}^{T}}\ \mathbb{E}\big[J_{\mathrm{real}}(\pi_T)\big] \quad \text{s.t.}\quad \sum_{t} \Delta C_r(m_t) \le B_r\ (r \in \{\mathrm{real},\mathrm{compute},\mathrm{eng}\}),\;\; \Pr[\text{unsafe} \mid \pi_T] \le \alpha.$$

$\lambda_r$ **理想下可解释为最优值函数对 $B_r$ 的边际价值**、实际本文只需 resource-weight estimate $\lambda_t = \lambda(B_t, \mathcal{D}_t, \pi_t)$、随 allocation state 更新。四条 caveat：**(i)** $Q_\lambda$ 默认 posterior mean、风险敏感可换 LCB / CVaR-adjusted utility——禁"公式 mean、文字 LCB"。**(ii)** SI fixed cost、DR diminishing returns、FT threshold、negative transfer 可让 $MV < 0$。**(iii)** $\mathrm{VoI}(m\mid\mathcal{D})$ 语义：**执行 $m$ 所购买的信息使下一步最优 action 的 expected net value 增加多少**（不是"当前 intervention 自己的 future utility"）；$\mathrm{VoI}(m) = \mathbb{E}_Y[\max_{m'} U_0(m'\mid\mathcal{D},Y)] - \max_{m'} U_0(m'\mid\mathcal{D})$、reference utility 固定为 $U_0$。$\beta$ 是 dimensionless 偏好权重、若 VoI 与 $U_0$ 同尺度可令 $\beta=1$。$V(\mathcal{D}) = -\Pr(\arg\max Q_\lambda$ flips$)$ 只是 decision-stability proxy。**(iv)** $\Delta J$ 非天然 causal effect——matched / paired evaluation、$\Delta C$ 含全部 incremental cost；**diagnostic-only action 的 $\pi_t^m = \pi_t^{\mathrm{control}}$、$\Delta J = 0$、价值全通过 VoI 体现**。数值只在固定 $p_{\mathrm{eval}}$ 下有意义。

**$MV_i$ 是 state-dependent 的**。先 SI 可使 DR $MV$ 下降、先 DR 可使 FT $MV$ 上升——**方向取决于 interaction、不假设单调**。intervention 之间有 complementarity / substitutability / conflict（不写成 bandit）。反馈层：**intervention 改 policy、进而改 $S_k^{\mathrm{int}}(\pi)$**：

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

有了这套写法、全文就非"四种方法谁更好"、而是闭环：定位主导 $\Delta_k$、sensitivity 判重要度、$\arg\max Q_\lambda$ 选下一步、真实评估回报、再定下一份。

## 四个 intervention lenses（可组合的分析维度）

SI / DR / DA / FT **非同一抽象层级**——SI 是 model calibration、DR 是 distribution manipulation、DA 是 representation alignment、FT 是 optimization strategy——并排成"四类方法"会误导四选一、其实是**四个可组合的 intervention lens**（本文 analytical decomposition、非领域公认 ontology）：

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

"$\times$" 是组合空间、非正交——DR 触及 Model / Observation / Distribution、DA 可发生在多层。

选工具标准是**"点估计 → 后验 → 鲁棒随机化"连续谱**。SI 做的是给 $\phi$ calibration posterior：

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; \mathcal{L}_{\mathrm{ID}}\big(D_{\mathrm{real}},\ f_{\mathrm{sim}}(\cdot\,;\,\phi)\big)$$

$\mathcal{L}_{\mathrm{ID}}$ 可取 trajectory prediction / one-step transition error / force-torque residual / likelihood——**很多经典 SI 不做 trajectory distribution matching、只最小化预测误差**。SI 解

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

**这只是 representative parameterization**——$y_t$ 可为 $x_{t+1}$、contact impulse、acceleration、deformation field 或其他 observable、$\psi$ 是 residual 的 input view；additiv

- **可微仿真**解决 optimization interface、不解决 model class correctness；
- **Residual physics** 保留 prior、有限修正；
- **Full-learned dynamics** 处理 physics 不适用的场景。

**可微性在 discontinuous contact / friction cone 切换处有 gradient vanishing 风险**（需 soft contact）。工程判据：physics 结构基本正确、参数或边界不准时、可微仿真性价比最高。

Residual physics 甜蜜点是 $f_{\mathrm{physics}}$ 提供**结构性归纳偏置**、residual 只在目标分布上有限修正。风险：sim 有 residual 补偿后看似好、到 OOD 失效——**residual model 的 valid domain 需与 deployment condition 对齐**。可微仿真在 contact-rich 场景受 complementarity problem 非光滑瓶颈。

### Axis B — Data distribution：domain randomization 及其家族

这条轴让 policy 对一族参数 $\{\phi\}$ 都稳健、不追求逼近最准 $p_{\mathrm{real}}$。Tobin（1703.06991）是起点。

**DR 非"隐式 ensemble"**——训练的是单个共享 $\pi_\theta$、目标是：

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

更准确：**DR 是对一族环境模型做 population-level 优化**、risk-neutral average-case baseline；worst-case 可写 $\max_\theta\min_\phi$。过度 DR 让 policy 过于保守、牺牲 performance。

**DR 非选 scalar range、而是设计 joint distribution**——$p(\phi_1,\phi_2)\neq p(\phi_1)p(\phi_2)$ 时 independent sampling 产生大量物理不一致 combo、降低 effective coverage。**correlated / adversarial curriculum** 是对策。

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

## 两条松动"两个给定分布"假设的新路线

上面四条轴共享一隐含前提：**$p_{\mathrm{sim}}$ 与 $p_{\mathrm{real}}$ 是两个给定分布**。下面两条路线恰在松动这个前提——非"第五第六种技巧"、是整个问题的 reformulation：**前四条 lens 改变 intervention、WM 与 co-training 改变的是 intervention 所作用的 underlying training substrate**、不塞回同一 taxonomy。

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

诚实边界：contact-rich / long-tail 场景学到的 model 常在 OOD 给出很自信也很错的想象。**WM net value = predictive utility − model uncertainty risk**：uncertainty 必须做 feasibility gate。

### Sim-and-real co-training：把"迁移"重述成 data mixture

Maddukuri et al.（RSS 2025, 2503.24361）的 Sim-and-Real Co-Training 是务实方向。**论文报告**：sim + real 混合采样、两平台六视觉操作任务、相对**自带 baseline** 观测到**约 37.9% aggregate relative improvement**——是**跨任务归一化的 relative lift**、非绝对百分点；引用务必带 baseline 与 aggregation 定义。不做单向迁移、而是一个 recipe 决定比例与调度。

读成 **data-mixture**——$p_{\mathrm{train}}=\alpha_{\mathrm{mix}} p_{\mathrm{sim}}+(1-\alpha_{\mathrm{mix}}) p_{\mathrm{real}}$；$\alpha_{\mathrm{sampling}} \neq \alpha_{\mathrm{effective}}$。Mechanistic 分析（Lei et al., 2604.13645）指出在该 generative robot policy 设置中 mixture 诱发 structured representation alignment——**paper-specific、不外推为 universal**。

## 评估：你怎么知道自己把 gap 补好了？

本文 claim 挂在**三级证据层**上——$\boxed{\text{A: mechanism}\quad \text{B: policy-response}\quad \text{C: deployment}}$：A 是 friction ID / calibration / latency measurement、B 是 $\hat S_k^{\mathrm{int}}$ / ablation / 有限差分 attribution、C 是真机 $\Delta J$ / $Q_\lambda$ / $MV$ / sim ranking utility。**三层是证据层级、非固定执行顺序**——诊断可循环（真机 failure → 怀疑 latency → 回测 A）；**不能互相替代**——SI 拟合属 A、不等于 C deployment 改善。

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

Spearman=0.95 却把 top-1 选错仍是灾难；反过来 Spearman=0.7 但 top-1 基本不出错、对"选一个能部署的 policy"够用。**sim fidelity 是 task-of-use dependent、不是 absolute property**——换用途（pretrain / exploration / curriculum / safety filter）"哪些误差重要"整个变一遍。$\pi^*_{\mathrm{real}}$ 不可得、$R_{\mathrm{select}}$ 与 real-domain learning gap 一样都是 oracle-defined 量、实际用 $J_{\mathrm{real}}(\pi_{\mathrm{best\text{-}observed}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$ 或 Pareto-best proxy。

**更重要的一层**：真实项目通常不要求 sim 精确排序所有 policy、只要求把值得上真机的候选压到可接受集合——**top-$k$ recall**、**regret@k** 应与 ranking 同级。**警惕 adaptive selection bias**：sim 若被 adaptive filter policy、用同一批被选候选反过来评 sim 造成 self-confirming 循环。**维护两个 pool**——$\Pi_{\mathrm{adapt}}$ 参与 training / selection、$\Pi_{\mathrm{audit}}$ 只做 held-out evaluation。**held-out set 并非无限次免疫的**：长期项目应保留 audit slice 或定期 refresh evaluation set、避免 adaptive experimentation 过拟合固定真机评测集。

至此、**allocation framework 的一个 corollary**：**sim utility 不是单一属性、是三个不能互替的维度；sim 内部自洽、低 prediction loss 或高 training reward 不能单独证明 downstream utility——必须由独立 real evidence 验证**——

| Simulator utility 维度 | 典型 metric |
| --- | --- |
| 数值预测准不准（absolute error / calibration） | MAE / RMSE $\mathbb{E}\big[|J_{\mathrm{sim}}(\pi) - J_{\mathrm{real}}(\pi)|\big]$、calibration curve、prediction interval coverage |
| 排序准不准（ranking） | Spearman $\rho_{\mathrm{rank}}$、Kendall $\tau$、top-k recall、regret@k |
| 选出的 policy 好不好（decision quality） | $R_{\mathrm{select}} = J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$（实际用 best-observed proxy） |

一个 sim 可校得很准却选错 policy、也可数值全错但排序稳、regret 小——三维不能互替。$U_{\mathrm{sim}}$ 不该写成抽象标量、应按用途**上标索引**：$U_{\mathrm{sim}}^{(u)}$、$u \in \{\text{pretrain},\ \text{selection},\ \text{exploration},\ \text{curriculum},\ \text{safety}\}$。评 fidelity 要相对**候选 policy family** 与用途：$U_{\mathrm{sim}}^{(u)}(\cdot \mid \Pi_{\mathrm{candidate}},\ p_{\mathrm{eval}}^{\mathrm{real}})$。

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

**stopping rule 三类**：**(a) economic stop**——$\max_{m \in \mathcal{M}_t^{\mathrm{feasible}}} Q_\lambda^{\mathrm{perf+info}}(m \mid s_t) \le 0$（local one-step stop、非全局最优——已知强互补 portfolio 应作为 candidate 一并评估）；**(b) decision-information stop**——剩余 experiments 的 expected VoI 已接近 0（不是"argmax 翻转概率"——margin 增大也有价值）；**(c) safety / feasibility stop**——剩余 candidate 全在 feasible 集外。任一触发即停。

## 一个最小可执行的 Sim-to-Real Allocation Protocol

框架不落到"明天项目组怎么跑"、就还是聪明的 framing。以下 6 步是**最小可执行版**、可跳过、但跳之前要说清对本项目 no-op 的原因。

**Step 1 — 固定 evaluation。** 锁死 task / initial-state 分布 / horizon / success metric / safety threshold / policy interface（obs + action schema + control freq）。**若 $\pi$ stochastic（$a_t \sim \pi_\theta(\cdot \mid o_t)$）、$J(\pi)$ 应理解成 evaluation protocol 下对 policy / reset / hardware randomness 的期望**、用 repeated runs / block evaluation 估计。**没这一步、后面 $\Delta J$ 没有共同基准**。

**Step 2 — 建 held-out real evaluation set。** 真机 eval 集与训练数据**必须分开**、覆盖 held-out hardware / calibration / object / 场景切片。用训练数据 evaluate、$\Delta J$ 一定 optimistic。

**Step 3 — 列 mismatch hypotheses（可 falsify）。**

| Hypothesis | Evidence | Belief | 候选 intervention |
| --- | --- | ---: | --- |
| friction $\mu$ 偏低 | contact slip | med | SI + DR |
| actuator latency 未建模 | 高频振荡 | high | SI + timing |
| camera extrinsics 偏 | grasp offset | high | Calibration / DA |
| contact model 错 | 柔性物体 OOD 失败 | low | Residual / WM |

每条 hypothesis **必须能被具体实验否证**、写不出否证条件的先剔除。

**Step 4 — one-time initial calibration pilot**（Step 5 才进入 sequential adaptive allocation）。每类候选用最小可行样本估 $\mathbb{E}[\Delta J]$ 与 posterior、不预设固定样本数。**$\Delta J(m) = J_{\mathrm{real}}(\pi_t^{m}) - J_{\mathrm{real}}(\pi_t^{\mathrm{control}})$**——control 承担相同训练步数 / 时间 / seed、只关掉本 intervention；**diagnostic-only action 的 $\Delta J = 0$、价值全通过 VoI 体现**。**matched / paired / block 化评估**：同批 seed、同 held-out slice、一致 hardware condition；漂移系统记录 hardware state。

**Step 5 — sequential adaptive allocation**：$m_{t+1} = \arg\max_{m\in\mathcal{M}_t^{\mathrm{feasible}}} Q_\lambda^{\mathrm{perf+info}}$——$\lambda_r$ 是 resource-weight estimate、objective 与 local score 必须一致。Safety 走 $\alpha$ gate、不进 cost。

**Step 6 — real evaluation → posterior update → 回到 Step 3。** 更新 $\mathcal{D}_t \rightarrow \mathcal{D}_{t+1}$、重估 $\lambda$、淘汰否证 hypothesis、新失败补入表。**最易跳过、最关键**——没 posterior update、流程退化为静态 checklist。

**定位**：最低落地版——小团队可合并 Step 3/4、大团队可加 portfolio opt。6 步都要写下来。

## 这意味着什么？：一个闭环，而不是一个开关

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)核心句是 evaluation-aware distribution allocation。套回 sim-to-real——**仿真数据 utility 不是 sim 内部属性、而是相对真实 evaluation distribution 的属性：**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

这解释了常见挫败："堆更多 sim 数据"有时没用——**当主要瓶颈恰好是 simulator 与真实 evaluation distribution 之间的 support / fidelity mismatch**、加同分布 samples 的边际收益会快速下降；**不能自动创造 evaluation-relevant coverage、也不能修正 model bias**。与其问"我的 sim 有多好"、不如问："我的 sim 在哪些 evaluation-relevant 方向上接近真实、哪些差得远？差得远的那些敏感度多高、用哪种预算压它最便宜？"

把这条线走完、sim-to-real 就不再是"能否迁移成功"的开关、而是带反馈的闭环：

$$\boxed{\ \text{diagnosis} \rightarrow \text{sensitivity / uncertainty} \rightarrow \text{intervention} \rightarrow \text{performance} + \text{information gains} \rightarrow \text{update }\mathcal{D}_t \rightarrow \text{re-allocate} \rightarrow\ \circlearrowleft\ }$$

配套的**闭环 spine 图**、比任何"四分类"表格都更贴合本文论点：

```text
                current policy π + evidence D_t
                          │
                          ▼
             ┌─── mismatch diagnosis ───┐
             │ model / obs / ctrl / dist │
             └─────────────┬─────────────┘
                           ▼
              sensitivity / uncertainty set
                           │
                           ▼
                candidate interventions  m = (type, Δb)
              ┌────────────┼────────────┬────────────┐
              ▼            ▼            ▼            ▼
             SI            DR          DA / FT      World model
              │            │            │            │
              └────────────┴─────┬──────┴────────────┘
                                 ▼
                       real evaluation (paired, CI)
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             performance gain            information gain
             (ΔJ_real)                (uncertainty shrinks)
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                    update evidence D_{t+1} → reallocate
                                 │
                                 └──────↺
```

（最后一步改变 sensitivity 与 mismatch、见 feedback loop。）

这条链是 **resource-constrained adaptive sequential experimentation framework**：敏感度与边际收益靠小步实验估出、一轮估完再定下一份预算。收成五层 spine：

$$\boxed{\begin{aligned}
&\textbf{L1}:\ \max_{\{m_t\}}\ \mathbb{E}[J_{\mathrm{real}}(\pi_T)]\quad\text{s.t.}\ \textstyle\sum_t \Delta C_r(m_t)\le B_r,\ \Pr[\text{unsafe}]\le \alpha\\
&\textbf{L2}:\ s_t = (b_t,\,\pi_t,\,\mathcal{D}_t,\,h_t)\\
&\textbf{L3}:\ m_t \in \mathcal{M}_t(s_t),\quad \mathcal{M}_t^{\mathrm{feasible}} = \mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}}\\
&\textbf{L4}:\ Q_\lambda^{\mathrm{perf+info}} = U_0 + \beta\,\mathrm{VoI}\quad\text{(one-step look-ahead approximation)}\\
&\textbf{L5}:\ MV = \mathbb{E}[\Delta J\mid\mathcal{D}_t]\;/\;\lambda^\top \Delta C(m)\\
&\textbf{Transition}:\ s_{t+1} = \mathcal{T}(s_t,\, m_t,\, Y_t)
\end{aligned}\;\longrightarrow\;\circlearrowleft}$$

层级：$\boxed{\text{global } \mathbb{E}[J_T] \supset \text{local } Q_\lambda \supset MV}$——$Q_\lambda$ 是 global sequential allocation 的 **one-step look-ahead approximation**（非 terminal information reward）、$MV$ 是 efficiency statistic。

收束：**sim-to-real 不是选一种 transfer technique、而是在当前 belief、不可互换预算与真实评估反馈下连续决定下一次 intervention**——这是全文理论 spine。**本文的贡献不是提出新的 optimization primitive、而是重新定义 sim-to-real 的 decision unit**——从「选一种方法」到「在当前 state 下选下一次 intervention」——并把 reality gap 与 sim utility 重述成 policy / evaluation-conditioned 的量。

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
