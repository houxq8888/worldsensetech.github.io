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
- **Level 3 — Allocation**：$MV(m \mid b, \pi)$——当前状态、预算、uncertainty 下、下一块资源投哪？

主线是 **Diagnosis → Experiment → Intervention → Allocation → Re-evaluation** 闭环。四个核心贡献：**(1)** reality gap 是 policy-conditioned、task-conditioned consequence、非 sim 固有标量；**(2)** descriptor / sensitivity 只是诊断、决策变量是 intervention；**(3)** SI / DR / DA / FT 是 multi-resource sequential allocation 下的 intervention lenses、非互斥方法；**(4)** sim evaluation 从 fidelity 扩到 downstream utility（prediction / ranking / selection）、作为 allocation 的 corollary。几笔不能互换的预算下不断问"下一块钱花在哪"——**误差预算**不是给误差项预分额度、而是花在**干预动作**上。

## Reality Gap：不是一个标量，而是一个 policy-conditioned 的 mismatch

Sim-to-real 常被叙述成"训练 policy 从 sim 迁移到 real"。更严格的起点是**两个分布**：同一条 $\pi$ 与环境交互各自诱导 $p_{\mathrm{sim}}^{\pi}(\tau)$ 与 $p_{\mathrm{real}}^{\pi}(\tau)$、一般不等：

$$p_{\mathrm{sim}}^{\pi}(\tau) \;\neq\; p_{\mathrm{real}}^{\pi}(\tau)$$

**"同一条 $\pi$"有前提**——sim 与 real 必须**共享同一 policy interface**：observation schema（键 / shape / 单位 / 归一化）、action schema（连续 or 离散、力矩 / 速度 / 位置、clamping）、control freq 与 action hold、时序 / delay 假设。interface 不一致、$\pi$ 非同一函数、$\delta_J$ 失去定义。

轨迹分布本身 **policy-induced**、随 $\pi$ 变、不是环境固有属性。真正关心的不是分布差、而是它在任务上**表现的后果**——同一 $\pi$ 两边的性能差：

术语分三层：**(a) distribution mismatch** $D(p_{\mathrm{sim}}^\pi, p_{\mathrm{real}}^\pi)$；**(b) transfer delta**

$$\boxed{\;\delta_J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)\;}$$

signed、真实反而更好时为正；**(c) performance discrepancy** $G_J(\pi) = |\delta_J(\pi)|$ 只谈幅度、下文敏感度用 $G_J$ 语义。**$J$ 默认越大越好 utility；若 $J$ 是 cost、符号反转、结构不变。** $\delta_J$ **不等于 reality gap 本身**、是 gap 在特定 $\pi$ + evaluation 下的 downstream consequence。

**distribution mismatch ≠ performance gap**：$p_{\mathrm{sim}}^{\pi} \neq p_{\mathrm{real}}^{\pi}$ 不自动意味 $\delta_J$ 很大、不同 policy 敏感度不同——只依赖粗粒度几何的 policy 换掉摩擦建模几乎不变、依赖高频力反馈的精细装配里此差异可能致命。真正影响 policy 的非 marginal $p(s)$、而是 **policy-conditioned occupancy** $d_{\mathrm{sim}}^{\pi}(s,a)$ vs $d_{\mathrm{real}}^{\pi}(s,a)$（contact-rich manipulation 还要加 contact-mode 索引）、逻辑链 $\pi \rightarrow d^\pi \rightarrow \text{mismatch} \rightarrow J$。严格写要把 **mechanism 与 induced distribution 分开**——transition / observation / actuation kernel 记作机制 $M_{\mathrm{sim}}, M_{\mathrm{real}}$、给定 $\pi$ 诱导 $p^{\pi}(\tau)$：

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ \mathcal{E};\ M_{\mathrm{sim}},\ M_{\mathrm{real}}\big)$$

$\mathcal{E}$ 是 evaluation 假设集合（initial-state / horizon / reward / constraints）；同一 $M_{\mathrm{sim}}$ 对 position control 可能 gap 很小、对 force-sensitive manipulation 可能巨大。**本文把 reality gap 操作性地视为四元组 $(\pi,\mathcal{E},M_{\mathrm{sim}},M_{\mathrm{real}})$ 下的 downstream discrepancy、非 sim 固有标量**——operational definition、不宣称领域有统一 formal definition。

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

两类来源不同、不要简单相加：reality mismatch 是"仿真与真实非同一世界"、task-spec mismatch 是"优化目标与部署目标非同一任务"。**观测与状态估计值得单独成层**——机器人真正执行 $a_t = \pi(o_t),\ o_t = h(x_t) + \epsilon$；camera 标定误差 / depth bias / 遮挡 / proprioception drift / 力传感器偏置 / estimator 时延**让 policy 看到的 state 与 sim 假设 state 不一致**、manipulation / locomotion 里比外观 gap 更伤 performance。**Stochasticity mismatch**（motor 随机性 / friction variability / sensor temporal correlation / comm jitter / unmodeled disturbance / repeated-reset variability）是 dynamics 高阶统计量差异、正是 DR 覆盖对象。**Timing mismatch**（$\Delta t_{\mathrm{sim}} \neq \Delta t_{\mathrm{real}}$、action hold、sensor delay、inference latency、异步 observation）**可被闭环动力学放大**、非 additive error、可改变 closed-loop stability。**Initial-state / env mismatch** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$、只有 sim-real reset / scene 对不上才是 env mismatch；**Objective shift** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$ 已是 objective mismatch、下文默认已对齐。

## 把"误差预算分配"写成一个可估计、可迭代优化的决策框架

拆完来源、给直觉一个数学落点。误差项强烈交互（sim 假设 proprioception 精确 + 真实有 latency、单看都不致命、叠加可让 controller 失稳）——更稳写法是承认耦合 $F$：

$$\boxed{\;\delta_J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}}\big)\;}$$

**每个 $\Delta_k$ 是 mismatch descriptor、可 scalar / vector / distribution / set-valued**；$F$ 是 schematic、非待估 predictive model、只标记"存在某种未展开的依赖"、由 sensitivity experiments / ablation 探测局部响应。**四个 $\Delta_k$ 是便于归因的 diagnostic buckets、非世界本身的正交 latent variables**——actuator delay 可伪装成 obs、estimator lag 可伪装成 ctrl、contact stochasticity 可伪装成 dynamics、bucket 既不正交也不可唯一辨识。层级显式写为 **reality / task discrepancies → diagnostic buckets（$\Delta_k$ 在此层）→ observable evidence → intervention candidates**、$F(\cdot)$ 只是归因链上的一层 schematic、不是 latent factor model。

**$\Delta_{\mathrm{opt}}$ 从 reality gap 拿掉**（层级不同：同固定 policy、sim 观测动力学都准但 RL 未训好、$\delta_J$ 小而 policy 差）——分成**两个诊断量**：

$$\underbrace{J_{\mathrm{real}}(\pi_{\mathrm{train}}) - J_{\mathrm{sim}}(\pi_{\mathrm{train}})}_{\text{transfer delta } \delta_J}\qquad \underbrace{J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})}_{\text{real-domain learning gap}}$$

**不能无条件相加叫 deployment loss**（$\delta_J$ signed、baseline 不同、层级不同）。$\pi^{*}_{\mathrm{real}}$ 通常不可获得、右侧是 oracle-defined 诊断量、实际用 $J_{\mathrm{real}}(\pi_{\mathrm{best\text{-}observed}})$ proxy。工程归因时、$F$ 局部近似成 $\delta_J \approx \sum_k w_k \Delta_k$——只是局部 heuristic。真正用来 decision 的是每类 mismatch 挑一 **intervention 变量** $\xi_k$ 后测得的 **intervention sensitivity**：

$$\hat S_k^{\mathrm{int}} \;\approx\; \frac{J_{\mathrm{real}}(\pi;\xi_k{+}\delta) - J_{\mathrm{real}}(\pi;\xi_k)}{\delta}$$

$\hat S_k^{\mathrm{int}}$ 的严谨名字是 **local intervention response / sensitivity statistic**、非真导数 $\partial J / \partial \xi_k$；$\xi_k$ 也非"gap 天然坐标"、是 sensitivity experiment 人为定义的 intervention variable、很多不可直接控制——分三档：**direct perturbation**（真机连续拨动、latency / friction）、**proxy / surrogate**（借 sim / bench 估、calibration error）、**diagnostic ablation**（换模块 / model / 数据集、contact model / state estimator）。**受控实验性扰动、非固有量求导**（不用 Pearl-style $\operatorname{do}$-calculus）。$\hat S_k^{\mathrm{int}}$ **只是诊断辅助**、核心决策量是下节 $Q_\lambda$ / $MV$；跨 $\xi_k$ raw sensitivity 因单位不同（ms / 无量纲 / degree / kg）**不可直接比较**、allocation 只能回到 $\Delta J$ 与 $\lambda^\top \Delta C$。

**诊断 ≠ 归因**：单 perturb $\Delta_{\mathrm{friction}}$ 与 $\Delta_{\mathrm{latency}}$ 各自影响很小、组合却可能 $\Delta J(\Delta_f,\Delta_l) \gg \Delta J(\Delta_f,0) + \Delta J(0,\Delta_l)$（synergy）。**Sensitivity experiments 只识别 locally influential intervention directions、不提供 additive causal attribution**；$\Delta_{\mathrm{model}}$ 与 $\Delta_{\mathrm{ctrl}}$ 也可能互相补偿、都是 ablation 估的 decision statistics、非严格分解。

### 真正的"分配"：把钱花在干预动作上，而不是在方法里挑一个

预算**连续地**分到每条干预轴：$b=(b_1,\dots,b_K)$、$b_k$ 花在干预 $k$ 上（$b_{\mathrm{SI}}=2\text{h}$、$b_{\mathrm{DR}}=10^6$ 步 sim、$b_{\mathrm{real}}=4\text{h}$ 真机）、非 0/1 选择。**部署 objective 不能只写均值**——mean 90% + catastrophic 1% 与 mean 88% + tail ≈ 0 **非同种部署决策**、写法是 **mean-plus-tail / 带安全约束的期望**：

$$\max_{b}\quad \mathbb{E}\big[J_{\mathrm{real}}(\pi_b)\big] \quad \text{s.t.}\quad \Pr\big[\text{unsafe} \mid \pi_b\big] \le \alpha$$

项目预算**非同一种货币**（GPU 近乎无限 / 真机机时稀缺 / 有机器时间却没工程人力）、正确写法是**多预算约束**、不折成标量 $B$：$C_{\mathrm{real}} \le B_{\mathrm{real}}$、$C_{\mathrm{compute}} \le B_{\mathrm{compute}}$、$C_{\mathrm{eng}} \le B_{\mathrm{eng}}$。**安全不进同一层 cost / budget**——它是 **chance constraint**（$\alpha$ 由 e-stop 容忍度 / hardware fault 上限决定）、不是 $C_{\mathrm{risk}} \le B_{\mathrm{risk}}$ 那种可折价软预算。**本文默认 deployment $J$ 已被固定为单一 utility 或已完成外部 scalarization**；若保留多目标（success / energy / cycle time / wear / safety）、$\Delta J$ / $Q_\lambda$ / $MV$ 应上升到 **Pareto 或 lexicographic decision layer**、不硬折进未说明的 $J$。

预算是分向量后、决策变量就该从"gap"换成"干预动作"：工程师买不到"$\Delta_{\mathrm{model}}$ 的 2 个百分点"、能买到 30 min SI / $10^6$ 步 sim / 100 条真机轨迹 / calibration / residual model。对干预 $m$ 定义边际效用更自然——**干预不直接改 $\Delta_k$、通过训练改 policy**：

$$\boxed{\;\pi_{b+m} \;=\; \operatorname{Train}\big(D_{\mathrm{sim}},\ D_{\mathrm{real}};\ m\big)\;}$$

于是"下一块钱花在哪"是以干预为变量、需要在真实世界逐步估计的量。**本文 decision framework 围绕两个量分层展开：局部 decision score $Q_\lambda$ 与效率读数 $MV$**（$Q_\lambda$ 才是 $\arg\max$ 的对象、$MV$ = **cost-normalized marginal value / 单位影子成本的效率读数**、跨预算 / 跨 intervention 比较用、**不是 decision rule**）。二者都 conditional on allocation state $s_t = (b_t, \pi_t, \mathcal{D}_t, h_t)$——$h_t$ = hardware condition（battery / temperature / calibration / wear / mode）、显著非平稳时显式纳入、漂移小可吸进 $\mathcal{D}_t$。**「DR 的 $MV$ 是多少」这个问题本身就问错了**、正确问是「当前 $s_t$ 下再加一单位 DR 的 expected value」——让框架更贴近 **adaptive experimental design**、非普通 cost-benefit analysis。严格写 $m = (\text{type}, \Delta b_m)$、$b' = b + \Delta b_m$、期望 conditional on $\mathcal{D}$。**$m$ 是可执行 batch / 最小预算单元、非无限小的一块钱**（SI fixed cost、DR diminishing returns、FT threshold effects 让极小 $\Delta C$ 上 $\Delta J / \Delta C$ 失真）——**$MV$ 是 batch-level marginal efficiency、非 infinitesimal derivative**。成本是**向量** $\Delta C(m) = (\Delta C_{\mathrm{real}}, \Delta C_{\mathrm{compute}}, \Delta C_{\mathrm{eng}})$、归一化到 scalar 用**影子价格 $\lambda$**（每条 binding budget 的 marginal value、由 LP / KKT / 经验标定给出）：$C_\lambda(m) = \lambda^\top \Delta C(m)$。

$$\boxed{\;MV(m \mid s_t;\lambda) \;=\; \frac{\mathbb{E}\big[\,J_{\mathrm{real}}(\pi_{b'}) - J_{\mathrm{real}}(\pi_{b}) \;\big|\; \mathcal{D}_t\,\big]}{\lambda^\top \Delta C(m)}\;}$$

**真正的局部 decision score 是 Lagrangian net value**、不是这个 ratio：

$$\boxed{\;Q_\lambda(m \mid s_t) \;=\; \mathbb{E}\big[\,\Delta J(m) \;\big|\; \mathcal{D}_t\,\big] \;-\; \lambda^\top \Delta C(m)\;}$$

**Candidate set 也 state-dependent**：$\mathcal{M}_t = \mathcal{M}(s_t)$——SI 已做完、residual rollout 未过 validation gate、safety 不满足时 online FT 禁、real data 不够时某些 DA pipeline 跑不起来——**这些直接缩掉 $\mathcal{M}_t$、不是让 $MV$ 变小**；$\mathcal{M}_t^{\mathrm{feasible}} = \mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}}$、**safety 是 action feasibility gate、不是低权重 cost**。

$$m_{t+1} \;=\; \arg\max_{m \,\in\, \mathcal{M}_t^{\mathrm{feasible}}}\; Q_\lambda(m \mid s_t)$$

**$MV$ 保留为跨 intervention 的效率读数、不作 decision rule**。二者**不等价**——一个极简 toy（$\lambda = (3,\ 0.1,\ 1)$、$\Delta C = (\text{real-h},\ \text{compute},\ \text{eng-h})$）：

| Intervention | $\Delta J$ | real h | compute | eng h | $C_\lambda$ | $MV$ | $Q_\lambda$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 min SI | 1.5 | 0.2 | 0.5 | 0.5 | 1.15 | **1.30** | 0.35 |
| Big DR batch | 3.0 | 0.0 | 20.0 | 0.4 | 2.40 | 1.25 | **2.00** |
| Camera DA | 2.5 | 1.0 | 3.0 | 0.5 | 3.80 | 0.66 | −1.30 |
| Real FT | 5.0 | 2.0 | 1.0 | 1.0 | 7.10 | 0.70 | −2.10 |

$MV$ 排序把 SI 顶到第一、$Q_\lambda$ 排序把 DR 顶到第一——**ratio 看单位效率、Lagrangian 看扣除机会成本后的净增益**；DA / FT 在此 $\lambda$ 下 $Q_\lambda < 0$、被 **economic stop** 排除、不是"降权但仍候选"。

$m_{t+1}$ 只是 **one-step / local rule**、非 global optimum；不同 intervention **不在同一 intervention space**、由 $m$ 的 type 分量承担。完整问题写成 **带 chance constraint 的 multi-resource sequential allocation**：

$$\max_{\{m_t\}_{t=1}^{T}}\ \mathbb{E}\big[J_{\mathrm{real}}(\pi_T)\big] \quad \text{s.t.}\quad \sum_{t} \Delta C_r(m_t) \le B_r\ (r \in \{\mathrm{real},\mathrm{compute},\mathrm{eng}\}),\;\; \Pr[\text{unsafe} \mid \pi_T] \le \alpha.$$

$MV$ / $Q_\lambda$ 是 sequential problem 的 **局部决策统计量**；$\lambda_r$ 是当前 state 下资源 $r$ 的**估计影子价格**——越稀缺且越能约束 deployment objective 往往越高、**但非"utilization 高 = $\lambda$ 高"的同义关系**（shadow price = 最优目标对约束 RHS 的边际导数、不是剩余百分比）；只有 intervention 近似无 interaction、成本线性、无 fixed cost 时固定 $\lambda$ 的 greedy 才近似 global——本文不假设这些、$\lambda$ 是随 $t$ 缓慢更新的估计量。这个 ratio 无法从 sim 解析求得、只能用 pilot / ablation / few-shot real eval sequential 地估。五条 caveat：**(i) uncertainty**——真机 $\Delta J$ 噪声大、看 CI / posterior / **LCB**、否则高方差 intervention 因偶然成功被错优先。**(ii) 非线性成本**——SI fixed cost、DR diminishing returns、FT threshold effects。**(iii) 非单调 / 负 MV**——over-randomization、过拟合 FT、错误 residual、negative transfer 可让 $MV$ 为负。**(iv) info value 是完整 objective 的缺失项**——很多 pilot（如 20 min friction ID）即时 $\Delta J \approx 0$、但显著缩小后续 uncertainty set、贡献是 **learning what to do next**。完整 objective 含 performance + VoI：$\max (\mathbb{E}[J_{\mathrm{real}}(\pi_T)] + \beta V(\mathcal{D}_T))$；$V$ 的轻量 operationalization 是 **decision-relevant uncertainty** $V(\mathcal{D}) = -\Pr(\arg\max_m Q_\lambda \text{ 在下轮 evidence 后改变})$——**posterior 收窄本身不天然有价值、只有可能翻转 $Q_\lambda$ 排序或 deployment 选择时才算 info gain**。$MV_{\mathrm{perf}} = \mathbb{E}[\Delta J]/C_\lambda$、$MV_{\mathrm{info}} = \mathbb{E}[\Delta V]/C_\lambda$；**忽略 $MV_{\mathrm{info}}$ 系统性低估 pilot、退化成 exploit-only**。**(v) $\Delta J$ 非天然 causal effect**——observed $\Delta J$ 混 seed / optimizer trajectory / training duration / data ordering / 真机 temperature / battery / wear / reset / evaluator 变异 / intervention 本身；**pilot 应做 matched / paired evaluation**（同批 seed、同 held-out real slice、尽量一致的 hardware condition）、训练随机性显著时用多 seed 把 training variance 纳入 posterior。**$MV$ / $Q_\lambda$ 估的是 intervention protocol 下 incremental value、非无条件因果效应**；$\Delta C(m)$ **必须是全部 incremental cost**——顺带引入的 training compute / optimizer steps / robot wear / eng effort 一并计入、否则系统性偏向"训练更久"的 intervention。$MV$ / $Q_\lambda$ 数值**只在固定 $p_{\mathrm{eval}}$ 下有意义**、换 held-out hardware / object / contact regime、ranking 可能翻盘。

**$MV$ / $Q_\lambda$ 不是固定常数**：$MV_i = MV_i(b_{1:i-1}, \pi_b, D_{\mathrm{real}})$——先 SI 缩 uncertainty set 则 DR $MV$ 下降；先 DR 起点更 robust 则 FT $MV$ 上升。**intervention 之间同时有 complementarity、substitutability、occasional conflict**——是 **resource-constrained sequential experimentation / adaptive allocation**（接近 adaptive experimental design、**不写成 bandit algorithm**、无严格 arm / stationary reward / regret 证明）。还有一层反馈：**intervention 不只压低 gap、还改变 policy、进而改变 policy 对 gap 的敏感度** $S_k^{\mathrm{int}} = S_k^{\mathrm{int}}(\pi)$、$\pi = \pi(m)$：

```
estimate mismatch → estimate sensitivity → intervention
       ↑                                          ↓
   re-estimate  ←  sensitivity changes  ←  policy changes
```

这张 feedback loop 比任何新公式更贴合 allocation thesis：sim-to-real 不是一次解完、是一轮做完重估一轮的 sequential experiment。

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

## 四个 intervention lenses（更准确说，四个相对独立的分析维度）

框架有了、再逐条看工具。SI、DR、DA、FT **非同一抽象层级的并列类别**——SI 是 model calibration、DR 是 distribution manipulation、DA 是 representation alignment、FT 是 optimization strategy——并排成"四类方法"会误导人四选一、其实是**四个相对独立的 intervention lens**、可组合（本文 analytical decomposition、非领域公认 ontology）：

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

"$\times$" 是组合空间、非数学正交——DR 触及 Model / Observation / Distribution、DA 可发生在 input / feature / latent / policy / output——"DA = Representation 轴"只是本文的一层 abstraction。

**选工具标准非"systematic → SI、random → DR"**、更常用的是"**点估计 → 后验 → 鲁棒随机化**"这条连续谱。SI 真正做的是**在 identification objective 下拟合参数**：

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; \mathcal{L}_{\mathrm{ID}}\big(D_{\mathrm{real}},\ f_{\mathrm{sim}}(\cdot\,;\,\phi)\big)$$

$\mathcal{L}_{\mathrm{ID}}$ 可取 trajectory prediction / one-step transition error / force-torque residual / likelihood——**很多经典 SI 不做 trajectory distribution matching、只最小化预测误差**。SI 解决**可辨识、可参数化的 model mismatch**、DR 解决**能被训练分布表示的 uncertainty**。

| mismatch 的性质 | 更自然的工具 |
| --- | --- |
| 可参数化 + 可辨识 | System Identification（point estimate $\hat\phi$） |
| 可参数化但只能给出不确定性 | Bayesian / posterior SI → posterior-guided DR |
| 可参数化但难辨识 / uncertainty 大 | Domain Randomization |
| 难以由低维物理参数充分表达、但有结构化 residual | Residual learning |
| observation / appearance mismatch | Domain Adaptation |
| policy 在目标域仍有 systematic residual | Fine-tuning |

关键：**"不能精确辨识"与"完全不知道"不是一回事**——拿到后验 $p(\phi \mid D_{\mathrm{real}})$、最自然动作 $\phi \sim p(\phi \mid D_{\mathrm{real}})$ 做 **posterior-guided randomization**、把 SI 与 DR 缝成连续谱。

### Axis A — Model：system identification、可微仿真与 residual physics

这条轴处理 $\Delta_{\mathrm{model}}$、三**不同层次**常被"可微仿真 = 更强 SI"打包：

$$y_t \;=\; \underbrace{g_{\mathrm{physics}}(x_t,a_t;\phi)}_{\text{可参数化的物理}} \;+\; \underbrace{r_\theta\big(\psi(x_t,a_t)\big)}_{\text{残差}} \;+\; \epsilon_t$$

**这只是 representative parameterization**——$y_t$ 可为 $x_{t+1}$、contact impulse、acceleration、deformation field 或其他 observable、$\psi$ 是 residual 的 input view；additive state-transition 只是其中一种、soft robot 的 residual deformation field、contact impulse residual 与 state residual 非同一数学对象。

- **可微仿真**回答"怎么优化模型"——提供穿过 simulator parameters / states / controls 的 gradient path、可用作 identification / trajectory optimization 的 **optimization interface**（**不等于 SI 本身**）；DiffTaichi（Hu et al., ICLR 2020，1910.00935）、Interactive Differentiable Simulation（Heiden et al., arXiv 2019，1905.10706）是代表实现。
- **SI** 回答"优化什么参数"——真实工作流常是 **real → identify → sim → train → real**。
- **Residual physics** 回答"模型未解释掉的部分由谁解释"——让网络学 $r_\theta$ 补差；$r_\theta$ 只是统一记号、可定义在 state transition / force / acceleration / contact impulse / deformation field 或其他 latent 上。

**可微性解决 optimization interface、不解决 model class correctness**——contact model 若未表达某真实现象、再精确梯度也只给"错误模型下的最优参数"；碰撞 / 摩擦 / 接触模式切换往往 **nonsmooth / piecewise-smooth**、即使 $\partial f/\partial\phi$ 存在也不保证梯度稳定、切换处未必优于 derivative-free。SI 两个坑：$p_{\mathrm{real}}(\tau)$ 不可直接访问、只有有限条真机轨迹；参数存在 ≠ 可辨识、identifiability 依赖 excitation 与 sensor observability（质量 / 阻尼 / 刚度在某些激励下产生几乎相同可观测轨迹）。

Residual physics 甜蜜点是 $f_{\mathrm{physics}}$ 仍提供**结构性归纳偏置**、residual 只在目标分布上有限修正——软体（Gao et al., RA-L 2024，2402.01086）、浮力腿式（Sontakke et al., 2023，2303.09597）类"主干物理算数、局部稳定残差"场景最好用；若 $f_{\mathrm{physics}}$ 全错、残差独扛整个 dynamics、不如直接学一个 model。$r_\theta$ **并不天然等于"缺失物理"**——unrestricted additive residual 会吞下 sensor bias / actuator error / timing / calibration / reward mismatch 成 **error sponge**、训练分布内拟合好、OOD 失稳；需结构约束（低维 / 稀疏 / 力或加速度尺度 / 物理先验 / 只在特定 contact regime 生效）。$\phi$ 与 $r_\theta$ 之间还有 **confounding**——残差足够灵活会吸收本应属 $\phi$ 的效应、$\hat\phi$ 失去意义；identifiability 要求 $f_{\mathrm{physics}}$ 与 $r_\theta$ 贡献在数据上可区分（正则化、scale separation、structural constraints）。

### Axis B — Data distribution：domain randomization 及其家族

这条轴不追求逼近"最准"的 $p_{\mathrm{real}}$、而是让 policy 对一族参数 $\{\phi\}$ 都稳健。Tobin（1703.06907）用纯视觉随机化把 sim 抓取检测搬到真机；Peng（1710.06537）推进到 dynamics；OpenAI in-hand（Akkaya et al., 1808.00177）几乎把 DR 推到极致——**不靠精确校准、靠"范围足够宽"吸收差异**。

常被写歪的直觉：**DR 非"隐式 ensemble"**——训练的是**单个**共享 policy $\pi_\theta$、目标是

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

更准确：**DR 是对一族环境模型做 population-level 优化**、上式是 risk-neutral average-case DR 的 baseline abstraction；robust / adversarial DR 换成 $\max_\theta \min_{\phi\in\Phi} J$、CVaR 或其他风险敏感形式。DR 有效条件：$\mathrm{supp}(p_{\mathrm{real}}) \subseteq \mathrm{supp}(p_{\mathrm{DR}})$ 且 real-typical 区域有足够 density。**更根本前提**：real dynamics 可被同一 $\phi$-parameterization 表达；若 sim model class 不含真实现象、$\phi_{\mathrm{real}}$ 无法定义、support 从根上不成立——是 **model-class uncertainty**、加宽 range 不解决。**主结论**：parameter-space support 是有用的 design proxy、**deployment-relevant 的是它诱导的 policy-conditioned occupancy**——$p_{\mathrm{DR}}(\phi)$ 到 $d_{\mathrm{train}}^{\pi}$ / $d_{\mathrm{real}}^{\pi}$ 之间的 overlap 决定 downstream transfer；**parameter coverage 必要非充分**。

再一层：**DR 非选 scalar range、而是设计 joint distribution**——$p(\phi_1,\phi_2) \neq p(\phi_1)p(\phi_2)$ 才是常态（payload ↑ 联动 actuator regime、temperature ↑ 联动 motor resistance / friction / battery）、独立 uniform DR 只是 baseline。回到分配：**randomization 分布要对齐 evaluation 与 objective**、过宽或无关拉低样本效率；但 robust 设定下适当扩大 uncertainty set 反而更稳——**"越宽越保守"并非普遍规律、shape 与对齐才是**。"Adaptive / Automatic DR" 是家族而非单一方法（curriculum / adversarial / automatic / posterior-based sampling / performance-driven range adaptation）——共同点是**避免一开始 over-randomize**。

### Axis C — Observation / Representation：domain adaptation 与观测翻译

这条轴处理 $\Delta_{\mathrm{obs}}$、在**观测/表示层**对齐 sim 与 real。**"Representation" 是本文 abstraction**——DA 实际可发生在 input / feature / latent / output / policy / dynamics model 六层；具体机制包括 feature-level adapter、latent alignment、policy distillation、sim-to-sim canonicalization（image translation / GAN / 扩散只是 input-level 特例；典型如 RCAN, James et al., CVPR 2019, 1812.07252——把随机化 sim 图翻回近似 canonical 干净图喂下游 policy、顺带把 DR 与这条轴缝起来）。**别把 DA 简化成 image translation**。两条边界：**(a) DA 只是 observation mismatch 子集**——camera intrinsics/extrinsics、temporal sync、sensor bias、depth distortion 更适合 calibration / SI / sensor modeling；**(b) task-relevant invariance 才是目标**——只对齐 $z_{\mathrm{sim}}\approx z_{\mathrm{real}}$ 不够、理想是保持 $I(z;y_{\mathrm{task}})$ 高、压低 $D(z_{\mathrm{sim}},z_{\mathrm{real}})$、与过宽 DR 抹掉任务信号同源。

### Axis D — Optimization / adaptation：真机微调

这条轴**非一类 mismatch、而是 adaptation operator**：直接在目标域继续优化 policy。既可作前三条轴的收尾、也可作**早期诊断或快速 adaptation 手段**。FT **可能同时改变 transfer delta 与 real-domain learning gap**、分别诊断；两 regime 成本结构完全不同：

- **Offline / imitation：** $D_{\mathrm{real}} \to \theta$、主要成本是**采集**。
- **Online RL：** $\pi_\theta \to a \to$ 真实 transition $\to \theta'$、主要成本是**交互 + 安全 + 磨损 + 探索**。

比较不能只看最终 success rate、还要看**达目标所需的真机交互预算**。粗略指标：

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{或}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

但只是**粗略指标**：依赖 baseline、非真 marginal efficiency。真正该看 learning curve / AULC / 每 100 条轨迹的边际收益

$$MV_{\mathrm{real}} \;\approx\; \frac{J(N+\Delta N)-J(N)}{\Delta N}$$

——这才与全文 $MV$ 框架接上。风险不止灾难性遗忘、更常见是**分布收窄**——真机数据比 sim 窄得多、微调后 policy 目标切片更好、鲁棒性反降、**generalization 换 specialization**。$MV_{\mathrm{real}}(N)$ **不保证始终为正**：前 100 条大涨、后续快速衰减、再往后可能过拟合甚至倒退——**FT 本身可进入负边际收益区间**。

## 两条松动"两个给定分布"假设的新路线

上面四条轴共享一隐含前提：**$p_{\mathrm{sim}}$ 与 $p_{\mathrm{real}}$ 是两个给定分布**。下面两条路线恰在松动这个前提——非"第五第六种技巧"、是对整个问题的 reformulation：**前四条 lens 改变 intervention、WM 与 co-training 改变的是 intervention 所作用的 underlying training substrate**、抽象层级不同、不塞回同一 taxonomy。

### World model：不是取消 simulator，而是换掉 simulator 的来源

**本文 lens disclaimer**：在 allocation taxonomy 里、我把 world model 看成"model source replacement"的 reformulation——**这是本文的分析角度、非 world model 的标准定义**。严格说 world model 外延比本节宽得多、本节只挑"相对 physics-sim 换掉 model 来源与 inductive bias"这个切面。

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)讨论过 world model 与 data utility。放进 sim-to-real 语境先纠正定位误读：**world model 不天然属于 sim-to-real**——两条路线 causal direction 不同：

```
Physics-sim route：  hand-designed dynamics  → train / optimize → deploy real
Learned-model route：interaction data → learned dynamics → imagine → optimize
```

**interaction data 可来自 real / sim 或混合**——learned-model route ≠ real-only。

需要说准：world model **并未取消 sim**、仍在做 simulation / imagination、只是 predictive model 是学出来的；更精确表述是 **改变 predictive model 的来源与 inductive bias**：

$$\text{model source} = \text{physics prior} + \text{learned dynamics} + \text{data}$$

三者可 hybrid、不必是 $f_{\mathrm{hand}} \rightarrow f_{\mathrm{learned}}$ 的二元替换；读成 simulator replacement 会过度简化。Dreamer（1912.01603）、TD-MPC2（2310.16828）体现这条路——**人工 sim 的 model bias 大到不值得先修**时、WM 提供的是问题本身的改写。DayDreamer（2206.14176）常被误读成"sim 预训练 → real 微调"、更准是展示了 **real-interaction-driven 的实验路线**——真机上学 WM、用 latent imagination 做 policy improvement。**不依赖手工 sim ≠ model-free**、WM 仍吃假设、只是把 inductive bias 从显式 physics 移到 learned model。

诚实边界："用真实数据学 dynamics" **不等于天然优于仿真**——把"手工建模成本"换成"真机采集 + 模型容量成本"；contact-rich / long-tail / 噪声大场景里学到的 model 常在 OOD 给出**很自信、也很错的想象**——是"手工 sim"与"直接真机 RL"间的又一 trade-off、非终局。**WM net value = predictive utility − model uncertainty risk**——用于 counterfactual selection / rollout-based allocation 时、**model uncertainty 必须做 feasibility gate 或 decision penalty**、否则加大 sim budget 只是放大 model bias。

### Sim-and-real co-training：把"迁移"重述成 data mixture

Maddukuri et al.（RSS 2025，2503.24361）的 Sim-and-Real Co-Training 是务实方向。**论文实际报告**：sim + real 混合采样、两平台六视觉操作任务、相对**论文自带 baseline（train-on-real-only 与 train-on-sim-only 各自对照）**观测到**平均约 37.9% aggregate relative improvement**——按论文自定义 aggregate metric 跨任务归一化后的 **relative lift**、非 success rate 绝对百分点、不能与 per-task delta 直接比较；引用时务必带 baseline 与 aggregation 定义。不做 sim→real 单向迁移、而是用一个 recipe 决定两者比例与调度。

**本文解读（非论文证明）**：读成 **data-mixture 问题**——co-training 主要干预变量是 training mixture $p_{\mathrm{train}}=\alpha_{\mathrm{mix}} p_{\mathrm{sim}}+(1-\alpha_{\mathrm{mix}}) p_{\mathrm{real}}$（记 $\alpha_{\mathrm{mix}}$、避免与 shadow price $\lambda$ 冲突）、不是 sim calibration 也非 deployment-time adapter；**$\alpha_{\mathrm{sampling}} \neq \alpha_{\mathrm{effective}}$**——sample 重复、augmentation、importance / loss weighting、curriculum、batch composition 都改变 effective training contribution。Mechanistic 分析（Lei et al., arXiv 2026，2604.13645）指出、**在该工作的 generative robot policy 设置中**、mixture 改变诱发 structured representation alignment 与 importance reweighting——**paper-specific 机制、不外推为"任何 mixture 都 universal 产生这两种效应"**；但足以说明"以 mixture 为主抓手、效应跨多维"、不是第五根正交轴。

## 评估：你怎么知道自己把 gap 补好了？

危险的做法是只在 sim benchmark 报性能。可信评估至少：

- 报告 **zero-shot transfer** 与 **few-shot / N-shot** 曲线；
- 用一组 **held-out hardware / calibration / object / contact / environmental regimes** 来测；
- 明确声明 sim 与 real 的**任务 / initial-state / evaluation distribution 是否一致**；
- 做**失败归因**：哪层 $\Delta_k$ 主导？归因错了预算就花错地；
- **不要只报均值**：至少 mean ± CI、跨多 seeds / resets、尽量 **paired evaluation**；
- **安全失败单独统计**：$J_{\mathrm{real}}$ 应并列 safety violation / e-stop / intervention / hardware fault / recovery time；**对低频安全事件、不能以「20 次未见 failure」直接推论 probability 很低**——应用 binomial upper confidence bound 或 CVaR 等 **tail-risk measure**、不是普通 mean ± CI；**具体估法**：观测 $X \sim \mathrm{Binomial}(n, p)$、$X = 0$ 时只能给 $p$ 的 upper confidence bound（Clopper–Pearson 或 Bayesian Beta posterior 的 $1 - \delta$ quantile）、把 chance constraint $\Pr[\text{unsafe}] \le \alpha$ 变成**"upper bound ≤ $\alpha$"的 gate**、非 point-estimate 比较——这样 $\Pr[\text{unsafe}] \le \alpha$ 才不停留在符号层面。

顺着"sim 是真实世界的代理"、还有个比"数值对齐"更本质的问题：**sim 能否正确预测"哪个 policy 更好"？**

一个**概念性例子**（数值不代表实验结果）：

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

在 sim 上 $A > B > C$、真机却是 $B > C > A$。这时 simulator **失去 model-selection utility**——你会用它挑出最差的 policy。故 **simulator 用于 policy / model selection** 时应同时看排序相关性 $\rho_{\mathrm{rank}} = \mathrm{Spearman}(J_{\mathrm{sim}}(\pi_i), J_{\mathrm{real}}(\pi_i))$ 与 selection regret：

$$\pi_{\mathrm{sim}} = \operatorname*{arg\,max}_{\pi \in \Pi} J_{\mathrm{sim}}(\pi), \qquad R_{\mathrm{select}} = J_{\mathrm{real}}\big(\pi^{*}_{\mathrm{real}}\big) - J_{\mathrm{real}}\big(\pi_{\mathrm{sim}}\big)$$

Spearman=0.95 却把 top-1 选错仍是灾难；反过来 Spearman=0.7 但 top-1 基本不出错、对"选一个能部署的 policy"够用。两者都是 **conditional metric**。**allocation framework 的结论**：sim fidelity 是 task-of-use dependent、不是 absolute property——换用途（预训练 / exploration / curriculum / safety filter）"哪些误差重要"整个变一遍。$\pi^*_{\mathrm{real}}$ 通常不可获得、$R_{\mathrm{select}}$ 与 real-domain learning gap 一样都是 **oracle-defined diagnostic quantity**、实际用 $J_{\mathrm{real}}(\pi_{\mathrm{best\text{-}observed}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$ 或 Pareto-best proxy。

**更重要的一层**：真实项目通常不要求 sim 精确排序所有 policy、只要求**把值得上真机的候选压到可接受集合**——**sim → candidate filtering → small real evaluation**、与 allocation philosophy（real 发现、sim 放大、real 验证）同构。故 **top-$k$ candidate recall**（真实 top policy 是否进入 sim 选出的 top-$k$）、**regret@k**、或 "real best $\in$ sim top-$k$?" 应提升到与 ranking 同级、甚至更实操——sim 只需把好 policy 框进候选集、不必精确排序尾部。**同时警惕一层比 training/eval leakage 更隐蔽的 adaptive selection bias**：若 sim 被用来自适应筛选 policy（sim select → real eval → update → 再 sim select）、real evaluation set 应保留**独立的 held-out candidates / validation policies**、避免「先用 sim 选、再用同一批被选候选反过来评 sim 的 selection utility」造成 self-confirming 循环——**training / eval leakage ≠ adaptive selection bias**、后者系统性高估 sim ranking 可靠性。**比单次 held-out 集更稳的长期做法是维护两个 policy pool**——$\Pi_{\mathrm{adapt}}$ 参与 training / active selection / intervention、$\Pi_{\mathrm{audit}}$ 只做 held-out evaluation 与 sim ranking utility 的定期重估——**audit pool 不参与 adaptive selection**、sim utility 的估计才不被 self-confirming 污染。

至此、**allocation framework 的一个 corollary**：**sim utility 不是单一属性、是三个不能互替的维度**——

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
| unknown long-tail，可模拟 | 低 | 少 | targeted simulation / DR |
| unknown long-tail，sim 生成不可信 | 低 | 中 | real data |
| model class 不确定 | 低 | 多 | learned world model（若 real 稀缺则先 physics prior + residual / DR） |
| mixed | mixed | mixed | co-training candidate（需先验证正迁移条件） |

**前两行的限定词不能省**：若 uncertainty 来自 **model-class uncertainty**（函数形式表达不了真实现象）、SI 与 DR 未必适用、得先落到 residual / world model / 真机数据。倒数第二行：光 "model unknown" 推不出 world model、判据是 **model uncertainty × real-data budget**——模型类不确定**且**真实交互充足时 learned world model 才合理。最后一行："co-training 兜底"与 allocation 冲突——sim 质量差、real 稀缺、action space / task semantics 不一致时可能负迁移。

常见组合 **SI → DR → DA → co-training / fine-tune**：**箭头只是示意、非固定 workflow**、实际顺序由主导 gap 与边际效用决定。**在 sim 已有较强 coverage、主要未知来自 model misspecification 时**、real data 的高价值用途是**发现 sim 未建模的 failure mode**、让 sim 放大；**若 deployment distribution 已相当固定**、real data 也可能主要承担直接 adaptation / imitation 的角色、不必先走 discovery / amplify——

$$\text{discover real tail} \rightarrow \text{identify structure} \rightarrow \text{amplify} \rightarrow \text{real validation}$$

即 **real 发现、sim 放大、real 再验证**。**这条 chain 成立有硬前提**：发现的 failure mode 能被当前 model class / learned surrogate 以可信方式表示；否则 real discover 之后应直接转向 richer model / world model / 追加 real data、而不是强把不可表示的 tail 塞进 sim 放大（这正是 model-class uncertainty 那一段的具体化）。

顺着这个逻辑、可回答框架允许的反问：**什么时候最优解其实是"不做 sim-to-real"？**
- **真机数据已便宜到 $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}^{\mathrm{effective}}$ 时**——$C_{\mathrm{real}}^{\mathrm{effective}}$ 是**有效真机成本**（安全 / operator / reset / 磨损 / 失败恢复 / deployment 多样性）；比较的是 horizon 内 expected cumulative value / cost、不是一次 intervention 的 raw hours。
- **仿真器 model class 本身就差**（$\Delta_{\mathrm{model}}$ 主导且难参数化、如软体 / 流体 / 复杂接触）——修 sim 边际效用低、不如走 world model 或真机数据。
- **部署分布非常固定**——不需大规模 DR、少量 targeted real fine-tuning 往往更划算。
- **simulator 不提供独特 coverage / safety / exploration / counterfactual access 时**——$U_{\mathrm{sim}}^{\mathrm{downstream}} < C_{\mathrm{sim}}^{\mathrm{effective}}$：不是 sim 不好、而是没提供 unique utility、opportunity cost 超过收益。

能承认"有时最优是不做 sim-to-real"、正是 allocation framing 应有样子：**不站"仿真"、只站"下一单位预算换回最多真实性能"。** 顺此、**sequential allocation 需要显式 stopping rule、分三类原因触发**：**(a) economic stop**——$\max_{m \in \mathcal{M}_t^{\mathrm{feasible}}} Q_\lambda(m \mid s_t) \le 0$、剩余 intervention 扣除机会成本后不再值预算；**(b) decision-information stop**——新增 evidence 改变 $\arg\max_m Q_\lambda$ 或 deployment 决策的概率已近 0、继续 explore 只买 noise；**(c) safety / feasibility stop**——剩余 candidate 全部落在 $\mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}}$ 之外。三者任一触发即**停、不默认花完剩余预算**——这才能 loop 闭环。

## 一个最小可执行的 Sim-to-Real Allocation Protocol

框架不落到"明天项目组怎么跑"、就还是聪明的 framing。以下 6 步是**最小可执行版**、可跳过、但跳之前要说清对本项目 no-op 的原因。

**Step 1 — 固定 evaluation。** 锁死 task / initial-state 分布 / horizon / success metric / safety threshold / policy interface（obs + action schema + control freq）。**没这一步、后面 $\Delta J$ 就没有共同基准**。

**Step 2 — 建 held-out real evaluation set。** 真机 eval 集与训练数据**必须分开**、覆盖 held-out hardware / calibration / object / 场景切片。用训练数据 evaluate、$\Delta J$ 一定 optimistic。

**Step 3 — 列 mismatch hypotheses（可 falsify）。**

| Hypothesis | Evidence | Conf. | 候选 intervention |
| --- | --- | ---: | --- |
| friction $\mu$ 偏低 | contact slip | med | SI + DR |
| actuator latency 未建模 | 高频振荡 | high | SI + timing |
| camera extrinsics 偏 | grasp offset | high | Calibration / DA |
| contact model 错 | 柔性物体 OOD 失败 | low | Residual / WM |

每条 hypothesis **必须能被具体实验否证**、写不出否证条件的先剔除。

**Step 4 — 低成本 pilot 估 $\Delta J$ 与 uncertainty。** 每类候选 intervention 用最小可行样本估 $\mathbb{E}[\Delta J]$ 与 CI / posterior——**以估出 effect size、排除明显低价值 intervention 为目标、不预设固定样本数**（低方差任务只需少量轨迹、高方差需更多；固定「5 条」与 adaptive allocation 冲突）。**采用 matched / paired / block 化评估**：同批 policy seed、同 held-out real slice、尽可能一致的 hardware condition；**对存在明显漂移的系统**（轮胎磨损、电机温度、电池衰减、calibration drift）、$\mathcal{D}_t$ 之外还要记录 hardware condition / calibration state、避免把 hardware drift 误归因于 intervention。

**Step 5 — 在 feasible 集内按 $Q_\lambda$ 选下一份预算**。$m_{t+1} = \arg\max_{m \in \mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}}} Q_\lambda(m \mid s_t)$、$\lambda_r$ 是当前 $\mathcal{D}_t$ 下资源 $r$ 的**影子价格估计**（稀缺 + 强约束 deployment 才 $\lambda_r$ 高、非简单"剩余最少者最大"）、$MV$ 只作跨 intervention 效率读数；不问"哪个方法最先进"。Safety 走 Step 1 的 $\alpha$ chance constraint、**作 $\mathcal{M}_t^{\mathrm{safe}}$ gate、不进 $Q_\lambda$ 的 cost**。

**Step 6 — real evaluation → posterior update → 回到 Step 3。** 更新 $\mathcal{D}_t \rightarrow \mathcal{D}_{t+1}$、重估 $\lambda$、淘汰否证 hypothesis、新失败补入表。**最易跳过、最关键**——没 posterior update、流程退化为静态 checklist。

**定位**：allocation framework 的**最低落地版**、非唯一实现——小团队可合并 Step 3 / 4、大团队可在 Step 5 加 portfolio opt。**这 6 步都不能"心里知道却不写出来"**、写下来才能 review、防 allocation 退化成"用熟悉方法"。

## 这意味着什么？：一个闭环，而不是一个开关

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)核心句是 evaluation-aware distribution allocation。套回 sim-to-real——**仿真数据 utility 不是 sim 内部属性、而是相对于真实 evaluation distribution 的属性：**

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

（最后一步会重新改变 sensitivity 与 mismatch 估计——见上文 feedback loop。）

这条链是一个 **resource-constrained adaptive sequential experimentation framework**：敏感度与边际收益靠小步实验在真实评估上估出、一轮估完再定下一份预算投到哪。**收成五层结构**、全文所有派生都挂在这条 spine 上：

$$\boxed{\begin{aligned}
&\textbf{L1 目标}:\ \max\ \mathbb{E}[J_{\mathrm{real}}(\pi_T)]\ \text{s.t.}\ \textstyle\sum_t \Delta C_r(m_t)\le B_r,\ \Pr[\text{unsafe} \mid \pi_T]\le \alpha\\
&\textbf{L2 状态}:\ s_t = (b_t,\ \pi_t,\ \mathcal{D}_t,\ h_t)\\
&\textbf{L3 候选集}:\ m_t \in \mathcal{M}_t = \mathcal{M}(s_t),\quad \mathcal{M}_t^{\mathrm{feasible}} = \mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}}\\
&\textbf{L4 决策分}:\ Q_\lambda(m \mid s_t) = \mathbb{E}[\Delta J(m) \mid \mathcal{D}_t] - \lambda^\top \Delta C(m)\\
&\textbf{L5 效率读数}:\ MV(m \mid s_t) = \mathbb{E}[\Delta J(m) \mid \mathcal{D}_t] / \lambda^\top \Delta C(m)
\end{aligned}\;\longrightarrow\;\text{execute} \rightarrow \text{real eval} \rightarrow \mathcal{D}_{t+1} \rightarrow s_{t+1} \rightarrow \circlearrowleft}$$

$MV$ 与 $Q_\lambda$ **分工、不互换**——$MV$ 回答"每单位影子成本效率多少"、$Q_\lambda$ 回答"扣完机会成本值不值得做"；只有二者同时看、才既不被 ratio 误导、也不丢失跨预算可比性。收束：**sim-to-real 不是选择一种 transfer technique、而是在当前 belief、不可互换的多种预算和真实评估反馈下、连续决定下一次 intervention**——这是全文的理论 spine、其他都是它的具体派生。**本文的 methodological contribution 不是新 optimizer 或 $\Delta J / \Delta C$ 公式（这不是新数学）、而是改变 sim-to-real 的 decision unit**：从「选一种 transfer method」转为「在当前 belief 与多资源约束下选下一次 intervention」、并把 reality gap 与 sim utility 都重述成 policy / evaluation-conditioned 的量。

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

sim-to-real 尚无公认的跨任务"哪种方法更强"定量对照——不同任务 / 硬件 / fidelity 上限下结论可能颠倒；上述工作更多是"这类 gap 用这方法可行"的样本、非可外推排序。本文四个 lens 分解、simulator utility 三维切分、constrained-allocation 形式化、$\hat S_k^{\mathrm{int}}$ 与 $MV$ 定义都是 **conceptual framework 与作者解读**：这些是 sensitivity experiments / ablation / 小规模真实评估估出的 decision statistics、非 sim 解析可求；co-training 读作 data-mixture、world model 读作 model-source replacement、同样不是受控实验证明的结论。

---

*本篇是"具身智能的数据问题"上下篇续篇：上篇讲数据来源与接口、下篇讲数据 scaling 框架；本篇把镜头拉到 sim-to-real、把它从"一堆迁移技巧"重述成带经验边际效用的闭环分配问题、接回 sequential data allocation 主线。*
