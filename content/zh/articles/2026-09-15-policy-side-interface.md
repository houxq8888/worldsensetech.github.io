---
title: 'Contract 立起来之后：VLA、Diffusion Policy 与 π0 到底吃什么？'
slug: "2026-09-15-policy-side-interface"
date: 2026-09-15
draft: false
categories: ["具身智能", "策略学习"]
tags: ["具身智能", "策略学习", "VLA", "Diffusion Policy", "π0", "RT-2", "OpenVLA", "Action Tokenization", "Structured State Contract", "Consumer Contract", "Declared Quotient", "Query Family", "Schema Compatibility", "Semantic Preservation", "Decision-Relevant Preservation", "Decision Sufficiency", "Conditional Mutual Information", "Residual Contract Information", "Declared Coverage Loss", "Projection Residual Loss", "Decision Collapse Rate", "Safety Obligation", "Contract-Read Primitives", "Intervention Consistency", "Equivariance", "Order-Constrained Response", "Conditional Log-Likelihood Ratio", "Dependency-Aware Fusion", "Mass-Preserving Top-k", "Constraint Certification", "Contract-only Intervention", "World-consistent Counterfactual", "Action-Relevant Separation", "Contract Ablation Gap", "Fixed Policy vs Retrained Policy", "Compliance Evidence", "Contract Consumer", "Evaluation Metrics"]
description: '《多模态融合接口》那一篇把上游交付物立成了 Structured State Contract——本文问它的对偶：如果 estimator 真的按 contract 交付、policy 侧到底能不能吃到。核心 boxed 不等式是 **Structured estimator output ≠ structured policy input**、完整 pipeline $\mathcal C\to(\mathcal C_\pi,Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}\to\mathcal B_{\mathcal C}$。本文把 semantic preservation 与 decision sufficiency 严格分开、并再拆出**decision-relevant semantic preservation**（用 $\mathcal A^{*},\mathcal G^{*}$ 定义、只保护会导致下游 action / safety 差异的 distinctions）、并把"decision-relevant preservation ⇒ decision sufficiency"这条 implication 显式限定在声明过的 evaluation distribution 上。interface 对象是 declared query family $Q_{\mathcal C_\pi}$、并配一条**schema compatibility 规则**：$\mathrm{schema\_version}(\mathcal C)$ 与 $\mathrm{supported\_version}(\mathcal C_\pi)$ 不兼容时**必须 fail-closed 或走显式 adapter**。审计侧本文不再讲"三层 loss"、而是 **三个 semantic losses + 一个 safety obligation**：$L_{\mathrm{declared}}$ 从 entropy 差分降级为 **weighted undeclared-query coverage** $L_{\mathrm{declared}}=\sum_{q\in Q_{\mathcal C}^{\mathrm{req}}} w_q\,\mathbf 1[q\notin Q_{\mathcal C_\pi}]$、$L_{\mathrm{projection}}=I(Y_\pi;\hat S\mid Z_\pi,O,L)$ 明确是"projection 之后的 residual contract information"（可以 operationalize representation-stage loss、但不与 encoder 强行 1:1 绑定）、$L_{\mathrm{decision}}$ 从"supremum norm"降级为**action-relevant collapse rate** $\mathbb E_{\mathcal R}[\mathbf 1[D_{\mathcal A}(\pi_\theta(\cdot|\hat S),\pi_\theta(\cdot|\hat S^{\prime}))<\epsilon]]$、$D_{\mathcal A}$ 测的是两个 admissible / optimal action set 是否被 policy 支持、与 HPC / HSS 真正闭环、第四个槽位是**safety obligation** $\mathcal O_{\mathrm{safety}}$、不是 information loss。primitive 侧：`mode_select` 用 **mass-preserving top-$k$**（本文只保证 retained mass 与 discarded residual mass 可区分、严格 Bayesian update 需要残差 component 的 sufficient statistics 另行定义）；`age_gate` 三组结构、字段名统一为 payload $(\mu,\Sigma)$ + metadata $(\alpha,\ell,h,v,\iota)$（$\alpha$ = age、$\iota$ = availability）+ derived trust $q=\tau(\alpha,\ell,h,v,\iota)$、2×2 表把 availability 与 validity 语义正交化；provenance / dependency / negative evidence 三分——`dependency_aware_fusion` 走 logit-level $L^{\prime}_{ij}=L_{ij}+b(R_{ij})$、$b$ 可正可负可 learned、logit_bias_learned 是 transformer 上的一种"便捷实现候选"、不指定为 canonical default。训练侧 intervention 分三档、Type II 是 order-constrained response。评估侧是**四种 compliance evidence** $E_{\mathrm{semantic}}/E_{\mathrm{representation}}/E_{\mathrm{decision}}/E_{\mathrm{safety}}$；HPC 拆成**contract-only intervention** $T_k^{\mathrm{contract}}$（测 interface compliance）与**world-consistent counterfactual** $T_k^{\mathrm{world}}$（测 decision competence）、benchmark 不再混用；SDS 用 $\alpha$ 而不是 $a$ 做响应参数、避免与 action 变量撞名、oracle 从"唯一定义"降级为 response set $\mathcal R_{\mathcal C}(\alpha)$ 的一种 baseline；$\Delta J_{\mathrm{prov}}$ 拆成 $\Delta J_{\mathrm{where}} / \Delta J_{\mathrm{dep}} / \Delta J_{\mathrm{neg}}$ 三条、与 provenance / dependency / negative evidence 三个 primitive 一一对应；CAG 明确"同一份训好的 policy、不 retrain"、区分 $\mathrm{CAG}^{\mathrm{fixed}}$（本文的 compliance evidence）与 $\mathrm{CAG}^{\mathrm{retrained}}$（architecture comparison）。safety 侧本文最关键的加粗一句从 "invalid evidence cannot justify relaxing the constraint" 精化为 **"invalid evidence alone cannot justify relaxing the constraint"**、并引入三态 certification $\mathrm{certification}_j\in\{\text{safe},\text{unsafe},\text{unknown}\}$、对应 relax / tighten-or-stop / conservative fallback。全文收在升级后的 boxed thesis：**A policy is a contract consumer, not merely a function approximator**、四个 contract-consumer 必答问 + 三 loss 一 obligation + 完整 pipeline。VLA / Diffusion / Flow / ACT / SAC / PPO 由此降级为实现坐标、不是理论分类。'
toc: true
related_articles:
  - 2026-09-14-multimodal-fusion-interface
  - 2026-09-13-tactile-force-sensing
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-12-sim-to-real-evaluation-protocol
  - 2026-09-07-vla-world-models
  - 2026-09-05-vla-pi-family
  - 2026-09-03-vla-deep-dive
---

> 接 [拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口](/zh/articles/2026-09-14-multimodal-fusion-interface/)：那一篇把多模态融合从"融合时机"问题拉回到"融合之后交付什么"、并把这份交付立成一个契约对象——**Structured State Contract**——一个携带 hypothesis、provenance、observability、availability / validity / age、contact set 与 negative evidence 的结构化状态。收在那句 claim 上：**A good multimodal system must represent disagreement, not merely resolve it.** 这一篇问它的**对偶**：如果上游真的按 contract 交付、**policy 侧到底能不能吃到**。

短答：**structured estimator output 不意味着 structured policy input**。中间存在一层显式的 projection $\Pi_\pi$、它本身就是一个**可能破坏语义的接口**。VLA 用 tokenizer、Diffusion Policy 用 image-encoder + proprio concat、classical head 用手工 state vector——三种 projection 各自破坏 contract 的不同切片、并且**破坏发生得无声无息**：loss 曲线照样往下走、eval 分数照样往上涨、你从训练日志里看不出被压掉了什么。这不是模型规模的问题、**是接口的问题**——但**问题不在"哪个模型会 collapse"**、**问题在"没有一种架构对 contract-relevant semantics 作了显式 preservation guarantee"**。这一区分是本文与常见 "architecture critique" 类文章的分水岭。

本文不推销某个具体 backbone、不反对 end-to-end learning、也不主张"某族天生比另一族更好"。本文的核心 thesis 更 general 也更可验证：**任何 policy architecture 都需要回答——它的 input projection 是否保持上游 contract 中 decision-relevant 的语义？** 这个问题一旦被摆到台面上、"VLA 还是 Diffusion Policy"这种表层争论就自动降级成了一个具体设计决策、而不是立场站队。

## 0. 全文框架：$\mathcal C\to(\mathcal C_\pi,Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}$ 是一个可审计的 semantic pipeline

先把整篇的分析框架摆在开头、后面每一节都会回到这张图。这一节立起本文真正的**形式对象**——**三层 quotient 声明链** $\mathcal C\to\mathcal C_\pi\to{\sim_\pi}\to q_\pi$、**schema compatibility 规则**、**三档 preservation**（full / decision-relevant / decision sufficiency）、**conditional-MI 形式的 residual contract information** $L_{\mathcal C}^{\pi}$、以及**三个 semantic losses + 一个 safety obligation**（不再是"三层 loss"）。

### 0.1 论证主线

```
Structured State Contract  Ŝ_t                                （9/14 定义）
        │
        ▼
upstream contract  C                                          （完整语义、estimator 侧交付）
        │  declare + version check
        ▼
consumer contract  C_π  ⊆ C                                    （policy 声明"我只承担 C 的哪一部分"、附 supported_version）
        │  induce
        ▼
equivalence  ~_π  on  Ŝ                                        （由声明过的 query family Q_{C_π} 定义）
        │  quotient
        ▼
declared quotient  q_π :  Ŝ  ↦  [Ŝ]_{~_π}                     （允许丢什么的显式声明、不是神经网络）
        │
        ▼
neural encoding  e_π :  [Ŝ]_{~_π}  ↦  z_π                     （现有 policy 都在做、但没有 q_π 的语义账本）
        │
        ▼
policy head  π_θ :  (z_π, o, ℓ)  ↦  a                          （第三个可能被无声绕过的层）
        │
        ▼
safety layer  g_safety :  (a, evidence_j)  ↦  a_applied          （第四个槽位：不是 information loss、是 obligation）
        │
        ▼
三个 semantic losses + 一个 safety obligation 分别审计
   ├─ L_declared     q_π  声明覆盖不足（weighted undeclared-query loss）
   ├─ L_projection   Π_π  在侧信息给定的条件下还剩多少 contract 信息
   ├─ L_decision     π_θ  decision-relevant pair 被折叠的 rate
   └─ O_safety       g_safety  invalid / unknown evidence 是否触发保守反应
        │
        ▼
三条 contract-relevant invariants
   ├─ mode          多 hypothesis 的结构             ──▶  primitive: mode_select
   ├─ temporal      异质 staleness / health 的分布    ──▶  primitive: age_gate
   └─ source        provenance / dependency / 负证据  ──▶  primitives: provenance / dependency / negative evidence
        │
        ▼
训练（conditional representation-side probe + Type I equivariance / Type II order-constrained / Type III unconstrained）
部署（safety filter 读 constraint-relevant certification、invalid evidence ALONE cannot justify relaxing constraint）
评估（四种 compliance evidence：semantic / representation / decision / safety；CAG 是 decision 层聚合、需 oracle baseline 界定、且必须固定 retraining protocol）
```

### 0.2 三层 quotient 声明链：$\mathcal C\to\mathcal C_\pi\to{\sim_\pi}\to q_\pi$

上一版把 $\Pi_\pi$ 拆成 $e_\pi\circ q_\pi$ 两层——那一版接受 reviewer 的批评、**$q_\pi$ 的责任仍然过重、"quotient 到底谁定义"没有被回答**。v4 把 $q_\pi$ 前面再垫两层、变成一条**可审计的三层声明链**：

$$\boxed{\;\mathcal C\;\xrightarrow{\text{declare}}\;\mathcal C_\pi\;\xrightarrow{\text{induce}}\;{\sim_\pi}\;\xrightarrow{\text{quotient}}\;q_\pi\;}$$

其中：

- **$\mathcal C$ · Upstream contract**——estimator 侧交付的**完整**语义 schema（9/14 §6–§8 定义的那份）。
- **$\mathcal C_\pi$ · Consumer contract**——**policy 作为 consumer 明确声明它承担 $\mathcal C$ 的哪一部分**。它可以只是 $\mathcal C$ 的一个子集（"这个 policy 不消费 provenance 字段"）、也可以是 $\mathcal C$ 上的一个 coarse-graining（"hypothesis 结构折成 point estimate、但 age 保留成独立字段"）。**$\mathcal C_\pi$ 是接口规格里的一段、不是 encoder 权重里的隐式偏好**。
- **${\sim_\pi}$ · Induced equivalence**——由 $\mathcal C_\pi$ 上的一组**声明过的 contract queries / decision-relevant predicates** $Q_{\mathcal C_\pi}$ 所诱导：

$$\hat S \sim_\pi \hat S' \quad\Longleftrightarrow\quad Q_{\mathcal C_\pi}(\hat S) \;=\; Q_{\mathcal C_\pi}(\hat S').$$

$Q_{\mathcal C_\pi}$ 是本文真正的**接口对象**——它把"允许丢什么"从模糊的语义承诺变成一组可以逐条 review 的 queries。典型例子："$\hat S$ 里第 $c$ 通道的 age 是多少"、"这个 track 的 posterior-weighted top-3 hypothesis 是哪三个"、"contact set 是否包含 pad 面"。

- **$q_\pi$ · Quotient map**——真正的 $q_\pi : \hat S \mapsto [\hat S]_{\sim_\pi}$、把 $\hat S$ 折到 ${\sim_\pi}$ 定义的商空间里。

有了这条链、本文的核心主张终于可以写成一句 reviewer 无法追问"quotient 谁定义"的话：

> **Policy does not need to preserve the entire upstream contract $\mathcal C$. It must explicitly declare a consumer contract $\mathcal C_\pi$ together with a query family $Q_{\mathcal C_\pi}$ — and the resulting quotient $q_\pi$ is exactly the semantic loss the policy is allowed to take.**

对比上一版那句"要么保留整个商结构、要么声明 sufficient quotient"——这一版给出了**"quotient 从哪来"**的完整答案：它来自一份**写下来的** $\mathcal C_\pi$、加上一份**可枚举的** $Q_{\mathcal C_\pi}$。

**$\Pi_\pi$ 仍然等于 $e_\pi\circ q_\pi$、但 $q_\pi$ 不再是 primitive、它由 ${\sim_\pi}$ 唯一决定、而 ${\sim_\pi}$ 由 $\mathcal C_\pi$ 诱导**。工程上这意味着：**接口文档里必须能贴出一张 $Q_{\mathcal C_\pi}$ 清单**、否则 $q_\pi$ 就退化成 encoder 里的隐式行为——正是本文要攻击的对象。

### 0.2.3 Schema version / compatibility（v5 新增）

有了 $\mathcal C_\pi$ 与 $Q_{\mathcal C_\pi}$、必须再回答一个 software-interface 层面的问题：**Consumer contract 是针对哪个 schema version 的？** Estimator 升到 $\mathcal C$ 的 v2（新增 `observability` / `negative_evidence` / `sensor_health`）、如果 policy 的 $\mathcal C_\pi$ 还停在 v1、当前框架就会**默默丢掉新增字段**——这正是本文全文批评的 **undeclared semantic loss**。所以接口层再加一条小规则：

$$\boxed{\;\mathrm{schema\_version}(\mathcal C)\;\not\simeq\;\mathrm{supported\_version}(\mathcal C_\pi)\;\;\Longrightarrow\;\;\text{reject / explicit adapter required.}\;}$$

也就是：**schema mismatch must fail closed or pass through an explicitly declared adapter**——不兼容时要么**拒绝加载 policy**、要么必须走一份**写下来的 adapter**（把 v2 新增字段显式映射到 v1 已有 slot 或显式声明丢弃、并附一条 $\mathcal C_\pi^{v2} = \mathrm{adapt}(\mathcal C^{v2},\mathcal C_\pi^{v1})$ 规约）。这一条让 "contract" 这个词真正具有 software-interface 的味道、而不是只停留在 ML abstraction。

### 0.2.1 三档 preservation：full semantic / decision-relevant semantic / decision sufficiency

这一小节是本文的理论锚点、必须把**三个**容易混用的性质彻底分开——上一版只分了两个、reviewer 抓得对："full semantic preservation 作为接口 criterion 是 over-preserving 的"。

给定一组 policy 服务的 **contract-relevant decision variables** $Y_{\mathcal{C}}$（下游 controller / planner / safety filter / diagnostics 会读的量）、以及 §0.2 已经定义的 $\hat S$ 上的 contract equivalence $\sim_\pi$（由 $Q_{\mathcal C_\pi}$ 诱导）。本文区分三档性质。

**Property A · Full Semantic Preservation**——投影在 contract-equivalent class 之间不做不可逆折叠：

$$\hat S \not\sim_\pi \hat S' \quad \Longrightarrow \quad \Pi_\pi(\hat S) \not\equiv \Pi_\pi(\hat S').$$

这要求 $q_\pi$ 在 $\mathcal C_\pi$ 定义的商上 injective。**是一个很强的性质、但可能 over-preserving**。

**一个具体的反例（reviewer 提的、v4 接受）**：两个 hypothesis

$$H_1 = \text{"object at } x = 1.00\text{"},\qquad H_2 = \text{"object at } x = 1.01\text{"}.$$

从 estimator semantic contract 看、$H_1\not\sim H_2$。但如果下游 controller 的 action space 分辨率只有 5 cm、那么

$$\mathcal A^{*}(H_1) \;=\; \mathcal A^{*}(H_2).$$

此时 policy 把 $H_1, H_2$ 折叠起来完全合理——**强行要求 full semantic preservation 反而会给接口加上不必要的负担**。

**Property A′ · Decision-Relevant Semantic Preservation（本文真正的接口 criterion）**——只在会导致**下游 action 或 safety 后果**不同的 distinctions 上要求 injective。用 $\mathcal A^{*}$（optimal / admissible action set）与 $\mathcal G^{*}$（safety guardrail consequences）定义一个 decision-level equivalence：

$$\hat S \sim_{\pi,\mathcal D} \hat S' \quad\Longleftrightarrow\quad
\begin{cases}
\mathcal A^{*}(\hat S) \;=\; \mathcal A^{*}(\hat S')\\[1mm]
\mathcal G^{*}(\hat S) \;=\; \mathcal G^{*}(\hat S')
\end{cases}$$

decision-relevant preservation 写作：

$$\hat S \not\sim_{\pi,\mathcal D} \hat S' \quad\Longrightarrow\quad \Pi_\pi(\hat S) \not\equiv \Pi_\pi(\hat S').$$

因为 $\sim_{\pi,\mathcal D}$ 比 $\sim_\pi$ 更粗、**Property A 蕴含 Property A′、反之不成立**。这就是本文真正想要的接口要求。

**Property B · Decision Sufficiency (conditional MI form)**——用**条件互信息**而不是差分 MI 来衡量"给定 policy 已经拥有的 side information、projection 后 $Y_\pi$ 里还剩多少 contract 信息"。上一版写成 $L_{\mathrm{dec}} = I(\hat S;Y_{\mathcal C}) - I(\Pi_\pi(\hat S);Y_{\mathcal C})$、reviewer 抓到了致命问题：**policy 的完整输入是 $\pi(a\mid \hat S, o, \ell, \text{language})$、raw image $o$ 里往往已经带了 age / provenance 的 proxy**。差分形式下 $L_{\mathrm{dec}} > 0$ 只说明"$\Pi_\pi(\hat S)$ 单独看对 $Y_{\mathcal C}$ 不是充分统计"、**并不说明 policy 真的缺信息**（信息可能已经从 $o$ 补回来了）。

v4 把 contract information loss 重定义为**条件 MI**：

$$\boxed{\;L_{\mathcal C}^{\pi} \;=\; I\!\big(Y_\pi\,;\,\hat S \,\big|\, Z_\pi,\, O,\, L\big),\qquad Z_\pi = \Pi_\pi(\hat S, O, L).\;}$$

（若 projection 只作用在 contract 上、可以把 $Z_\pi$ 简化为 $\Pi_\pi(\hat S)$。）这个定义干净得多：

- $L_{\mathcal C}^{\pi} = 0$ **当且仅当** $Y_\pi \perp\!\!\!\perp \hat S \mid Z_\pi, O, L$——即 **$Z_\pi$ 对 $Y_\pi$ 是 sufficient given policy 已经拥有的其它输入**。
- 直接回答了"raw image 里面已经有 object identity / provenance proxy"这个 reviewer objection：**正因为我们用的是 conditional sufficiency、不是 unconditional MI、raw observation 已经补回来的那部分信息不会被误算成 loss**。

**三者的关系（v5 补上适用条件）**：

$$\text{Full semantic preservation}\;\Longrightarrow\;\text{Decision-relevant preservation}$$

是**无条件**成立的（因为 $\sim_{\pi,\mathcal D}$ 比 $\sim_\pi$ 更粗、pointwise injective 从前者直接推到后者）。但第二支箭头：

$$\boxed{\;\text{Decision-relevant preservation}\;\Longrightarrow\;\text{Decision sufficiency}\quad\text{only under the declared evaluation distribution.}\;}$$

**不再写成 unconditional theorem**。原因（reviewer 抓得很准）：preservation 是 pointwise distinction statement、sufficiency 是在某个 $P(\hat S, O, L)$ 下的 conditional independence、两者不同类。加上 evaluation distribution 假设之后可以这样说：**在 benchmark distribution 下、并假设 $Y_\pi$ 是充分的 decision descriptor、decision-relevant preservation 蕴含 decision sufficiency**。少了这个限定、就会被数学 reviewer 拆开问"preservation 是对所有 $\hat S$ 的 statement、你怎么推到 distribution-level conditional independence"。

反过来都不成立。**Decision sufficiency 是最弱的性质**、$L_{\mathcal C}^{\pi}=0$ 允许 policy 在 $Y_\pi$ 不区分的两个 $\hat S$ 上折叠；**decision-relevant preservation 居中**、只保护会引起 action / safety 差异的 distinctions；**full semantic preservation 最强**、是一个 diagnostic property、而不是接口 requirement。

于是本文的 killer 一句应该这么写：

> **Decision-relevant semantic preservation is the actual interface requirement; full semantic preservation is a stronger diagnostic property; decision sufficiency is a weaker consequence (under the declared evaluation distribution).**

**Policy 可以丢 information、但必须 either (a) preserve decision-relevant contract semantics, or (b) explicitly declare $\mathcal C_\pi$ + $Q_{\mathcal C_\pi}$、使得丢掉的 distinctions 被 $Q_{\mathcal C_\pi}$ 承认不在其关心的范围里**。未经声明地丢掉 decision-relevant contract semantics、才是接口违规。

### 0.2.2 三个 semantic losses + 一个 safety obligation（v5 重写）

v4 把整条 pipeline 拆到 $q_\pi$ / $e_\pi$ / $\pi_\theta$ 三层、每层配一个 loss——但 reviewer 抓得对：**这个"三层 loss"框架有两个 type mismatch、而且把 safety 错当成 loss**。v5 一次改三处：

**结构层面**：pipeline 已经是四层 $\mathcal C\to(\mathcal C_\pi, Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}$、其中**safety 不是 information loss、是一种 obligation**。所以本文最终框架不是"三层 loss"、而是 **"三个 semantic losses + 一个 safety obligation"**：

$$\boxed{\;\mathcal C\;\longrightarrow\;(\mathcal C_\pi, Q_{\mathcal C_\pi})\;\longrightarrow\;q_\pi\;\longrightarrow\;e_\pi\;\longrightarrow\;\pi_\theta\;\longrightarrow\;g_{\mathrm{safety}}\;\longrightarrow\;a.\;}$$

**对象层面**：v4 的 $L_{\mathrm{declared}}$ 与 $L_{\mathrm{decision}}$ 都是错的、v5 分别重写。

**(1) $L_{\mathrm{declared}}$ 从 entropy 差分降级为 weighted undeclared-query coverage**。v4 的 $H(Q_{\mathcal C}(Y_{\mathcal C})) - H(Q_{\mathcal C_\pi}(Y_{\mathcal C}))$ 有三个问题：$Q_{\mathcal C}(Y_{\mathcal C})$ 里的 random variable 从未定义、entropy 差分不一定非负（query 数量 / 编码 / 基数都会改变 entropy）、而且 $q_\pi$ 的核心语义是"声明哪些 distinctions 允许被丢"、天然是一个 **coverage / violation set**、不是一个 scalar entropy。v5 直接把它写成集合：

$$\mathfrak D_\pi \;=\; Q_{\mathcal C}^{\mathrm{req}}\setminus Q_{\mathcal C_\pi}\qquad\text{（required queries 里、consumer contract 没覆盖的部分）}$$

$$\boxed{\;L_{\mathrm{declared}} \;=\; \sum_{q\,\in\, Q_{\mathcal C}^{\mathrm{req}}} w_q\,\mathbf 1\!\big[q \notin Q_{\mathcal C_\pi}\big]\;}$$

$w_q$ 是 task-side 权重：frame semantics 高、age 中高、provenance task-dependent、跟当前 controller 完全无关的 diagnostic field 低。最粗的 cardinality 版本 $L_{\mathrm{declared}} = |\mathfrak D_\pi|$ 就是 $w_q \equiv 1$ 的特例。这样"审 $q_\pi$"就变成**对着 $Q_{\mathcal C}^{\mathrm{req}}$ 清单一条一条勾**、不再依赖 entropy 定义。

**(2) $L_{\mathrm{rep}}$ 改名 $L_{\mathrm{projection}}$、明确它是 residual contract information 而不是 encoder loss**。v4 里 $Z_\pi = \Pi_\pi(\hat S, O, L)$ 已经是"整条 projection 的输出"、不只是 encoder $e_\pi$ 的输出、所以把它叫 representation loss 是把数学对象与 pipeline layer 强行 1:1 绑定。v5 直接改名：

$$\boxed{\;L_{\mathrm{projection}} \;=\; I\!\big(Y_\pi\,;\,\hat S \,\big|\, Z_\pi,\, O,\, L\big)\;}$$

语义是 **"projection 之后的 residual contract information"**——可以 operationalize 为 representation-stage loss、但不宣称它就是 $e_\pi$ 那一段的损失。

**(3) $L_{\mathrm{decision}}$ 从"supremum norm"降级为 action-relevant collapse rate**。v4 的 $\sup_{\hat S \not\sim_{\pi,\mathcal D} \hat S'} \|\pi_\theta(\hat S) - \pi_\theta(\hat S')\|_{\text{action-distribution}}^{\!\perp}$ 是**类型错误**——norm 是距离、不是"pair 集合大小"。v5 把它写成 collapse rate：

$$\mathcal R_{\mathcal D} \;=\; \big\{(\hat S, \hat S') : \hat S \not\sim_{\pi,\mathcal D} \hat S'\big\}$$

$$\boxed{\;L_{\mathrm{decision}} \;=\; \mathbb E_{(\hat S,\hat S')\sim\mathcal R_{\mathcal D}}\!\Big[\mathbf 1\!\big(D_{\mathcal A}(\pi_\theta(\cdot\mid \hat S),\,\pi_\theta(\cdot\mid \hat S')) < \epsilon\big)\Big]\;}$$

其中 $D_{\mathcal A}$ **不是普通的 distribution distance**——它测的是"两份 action distribution 是否**支持不同的 admissible / optimal action set**"。这一改把 reviewer 的哲学 objection 一起接了：如果 $\mathcal A^*(\hat S_1) \neq \mathcal A^*(\hat S_2)$、但两个 state 的公共交集里存在一个 action 都 admissible、policy 输出那个共同 action 是合法的、$D_{\mathcal A}$ 应该识别这种情况、不当作 collapse。软阈值版本可以写成 $\mathbb E_{\mathcal R_{\mathcal D}}[\exp(-D_{\mathcal A}(\cdot))]$。$D_{\mathcal A}$ 与 §6.2 HPC / HSS 的语义闭环：**HPC 测 coverage、HSS 测 separation、$L_{\mathrm{decision}}$ 测两者的失败率**。

**(4) $\mathcal O_{\mathrm{safety}}$ 是 obligation、不是 loss**。safety 那一层不做"信息损失"的度量、它做的是**"当 evidence 不足或 unknown 时、filter 有没有采取保守反应"**的义务判定（详见 §5.3 三态 certification 与 §6.6 safety evidence）。**它是第四类量、不能与三个 loss 混称**。

| Slot | 语义 | Failure | Object |
|---|---|---|---|
| $q_\pi$ | 声明允许丢什么 | **Declared coverage loss**：$Q_{\mathcal C_\pi}$ 没覆盖 required query | $L_{\mathrm{declared}} = \sum w_q \mathbf 1[q\notin Q_{\mathcal C_\pi}]$ |
| $\Pi_\pi$ | projection 之后条件 residual | **Projection residual**：条件 MI > 0 | $L_{\mathrm{projection}} = I(Y_\pi;\hat S\mid Z_\pi,O,L)$ |
| $\pi_\theta$ | decision 是否用被保住的 distinctions | **Decision collapse rate**：action-relevant pair 折叠的比例 | $L_{\mathrm{decision}} = \mathbb E_{\mathcal R_{\mathcal D}}[\mathbf 1[D_{\mathcal A} < \epsilon]]$ |
| $g_{\mathrm{safety}}$ | evidence 不足时的义务反应 | **Safety obligation violation**：unknown / invalid 时没收紧 | $\mathcal O_{\mathrm{safety}}$ |

$\Pi_\pi$ injective **不蕴含** $L_{\mathrm{decision}} = 0$——三个 loss 分别可以独立爆。§6 的四种 compliance evidence 与 §6.7 的 skeleton table 都会直接引用这三个 loss + 一个 obligation。

### 0.3 三条本文的 boxed claim

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. 完整 pipeline $\mathcal C\to(\mathcal C_\pi,Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}$ 是**一个可审计的 semantic interface**、三个 loss 分别落在 $q_\pi$ / $\Pi_\pi$ / $\pi_\theta$ 上、第四个槽位是 safety obligation 而不是 loss。这就是本文定级 thesis 的雏形——**A policy is a contract consumer, not merely a function approximator**。

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state head、latent visuomotor policy、VLA / diffusion / flow policy 用的 conditioning 与 action generator 机制都不一样、但它们作为 contract consumer 都必须回答**同一组四个问题**——what may I discard ($q_\pi$)? what did I actually retain ($\Pi_\pi$, $L_{\mathrm{projection}}$)? how should decisions respond to contract interventions ($\pi_\theta$, $L_{\mathrm{decision}}$)? what happens when evidence becomes invalid or unknown ($g_{\mathrm{safety}}$, $\mathcal O_{\mathrm{safety}}$)? 攻击的对象是 interface contract、不是模型架构。

> **Claim 3 · Contract compliance should be tested by controlled intervention, not inferred from end-to-end success.** 端到端 success 衡量的是 policy 好不好用、而不是它有没有把 contract 语义读对。contract compliance 需要一组**受控测试**——invariance / equivariance / **order-constrained response** / task-conditional utility under contract interventions——**并且需要一条 oracle baseline 来界定每个指标的语义范围**、以及**明确的 retraining protocol**（§6.5 会区分 $\mathrm{CAG}^{\mathrm{fixed}}$ 与 $\mathrm{CAG}^{\mathrm{retrained}}$）。合规 argument 是**多证据合流**、不是单一 score。

## 1. 两种维度、而不是三个家族：conditioning representation / semantic interface × action head

**旧版把 "VLA / Diffusion Policy / engineered head" 当成三个互斥家族**、这个 taxonomy 有点粗糙——π0 就是 VLA + flow matching、既在 VLA 那一列也在 diffusion/flow 那一列。这一节改成**两个正交维度**、policy 家族的选择就变成 grid 上的一个坐标、而不是一个立场。

### 1.1 两个正交维度

**维度 1 · Conditioning representation / semantic interface**——policy 用什么形态读上游信息。**注意这一列不是纯 representation 分类、structured contract 那一档同时携带 representation + semantic schema + provenance + validity + semantics**、所以列名必须写成 "representation / semantic interface"、否则会被 reviewer 指为混层。四类典型：

- **engineered state**（人工写死的定长向量、$s_t \in \mathbb{R}^d$）
- **visual latent**（image encoder 之后的低维表示、$z_t = f_\phi(o_t)$）
- **multimodal token**（VLM 词表里的离散 token、image + language 共享空间）
- **structured contract**（显式 $\hat S_t$、字段可枚举、可被 controller / policy / safety filter 共同读；带 schema 与 semantic guarantee）

**维度 2 · Action head**——policy 怎么产生 action 分布。五类典型：

- **deterministic / Gaussian / mixture**（SAC 一脉、mean + variance / GMM）
- **autoregressive token**（离散词表、RT-2 / OpenVLA）
- **generative sequence decoder**（CVAE → chunked、ACT 一脉——**不是 diffusion 也不是 flow**）
- **diffusion**（多步 denoising、Diffusion Policy）
- **flow matching**（连续 action chunk 的 velocity regression、π0）

**上一版把 ACT 塞在 "generative action chunk" 那一栏容易被误读成"ACT 属于 diffusion 家族"、这一版把它单列出来、与 diffusion / flow 并列。**

### 1.2 Grid

| 代表 policy | Conditioning representation / semantic interface | Action head | 破坏 contract 的位置（本文 schema 下的分析） |
|---|---|---|---|
| Classical SAC/PPO head | engineered state | Gaussian / deterministic | 最小、字段人工写；但字段设计本身是艺术、跨任务差 |
| Visual-obs PPO / DrQ | visual latent | Gaussian | $z_t$ 一压、provenance / age / hypothesis 通常已经丢 |
| Diffusion Policy (Chi 2023) | visual + proprio latent | diffusion | 本文 schema 下：action-side multimodality OK、但 state-side 多 hypothesis 没有 slot |
| ACT / ALOHA (Zhao 2023) | visual + proprio latent | **generative sequence decoder**（CVAE → chunked） | 与 Diffusion Policy 类似的 conditioning、action generator 是另一族 |
| RT-2 | multimodal token (VLM) | autoregressive action tokens | frame / reference_point / convention 会被 tokenize 掉 |
| OpenVLA | multimodal token + proprio | autoregressive | 同 RT-2、proprio 侧多一条通道、但 contract 字段仍无处安放 |
| π0 (Black 2024) | multimodal token (VLM conditioning) | **flow matching**（连续 action chunk） | 本文 schema 下：action 侧连续、但 conditioning 侧仍走 VLM token、无 slot for hypothesis / provenance / age / negative evidence |

一句关键区分：**Continuous actions do not imply structured state semantics.** π0 的 action head 是 flow matching 出来的连续 chunk、听起来很"结构"——但它的 conditioning representation 是 VLM token。**注意这里的措辞**：**Under the Structured State Contract defined here, π0's conditioning interface does not expose an explicit slot for hypothesis, provenance, age, or negative-evidence semantics.** 这是**本文 schema 下的分析**、不是 π0 原论文自己声明的 limitation——π0 论文事实是"预训练 VLA + proprio token + noisy action chunk + flow matching"、contract-level 的批评来自本文的分析视角。同样的 caveat 适用于 Diffusion Policy 与 OpenVLA 那两行的"破坏 contract 位置"栏——**它们都是 paper fact + 本文 interface analysis 的混合、不是原论文承认的缺陷**。

一句 caveat：**这不是"哪个组合最好"的排序**。engineered state + Gaussian 依然是低维控制 baseline 之王、multimodal token + flow matching 依然是开放语义条件下唯一现实的路线——本文关心的是**每一种组合、它的 $\Pi_\pi = e_\pi \circ q_\pi$ 是否有一个写清楚的 $q_\pi$**。答案在多数现有工作里是"没有"——**不是某个家族天生不行、是这个 $q_\pi$ 层从来没有被当成接口设计过**。

## 2. "State" 在不同 policy 里意味着什么

这一节是**术语清障**。"state"这个词在具身智能文献里至少有五种互不相同的意思、policy 家族的选择往往就是"它默认哪一种"的选择：

**$\pi_{\mathrm{obs}}$：raw observation**——图像、点云、力 / 扭矩读数的原始流。绝大多数 imitation learning 训练管线的**输入形态**、但一般不是 policy 内部**实际使用的 state**（会被 encode 掉）。

**$z_t = f_\phi(o_{:t})$：encoded latent**——encoder 之后的低维表示。Diffusion Policy、VLA 的 image tokens、world model 的 RSSM state 都属于这一类。contract 到这里已经**经过一次 projection**、observability 与 provenance 通常已经丢了。

**$b_t$：belief / posterior**——POMDP 一脉的显式 belief state。contract 里"多 hypothesis + posterior weight"结构最贴合这个。但主流 VLA / Diffusion Policy 都不显式建模 belief——它被 encoder 隐式近似。

**$s_t = (b_t, \pi_t, q_t, h_t)$：9/10 Part 1 定义的 allocation state**——belief + policy + budget + hardware。这是**决策层**的 state、不是 policy 的输入 state。

**$\hat S_t$：structured state (contract)**——9/14 定义的那个契约对象。

**注意、这五样不是同一条流水线上的五个 stage**——它们是**五种可以互相替代的 abstraction**、不同 policy 家族从中挑一种（或几种并联）。上一版画成 $\pi_{\mathrm{obs}}\to z_t\to\hat S_t\to b_t\to s_t$ 一条链、reviewer 抓得对：**$\hat S\to b$ 并不是普遍成立的 posterior 关系**、而且实际架构常常**跳过**中间几步——纯 visuomotor 走 $o\to z\to\pi$、engineered head 直接走 $o\to\hat S\to\pi$、端到端 VLA 事实上停在 $o\to\text{token}\to\pi$。更合适的图是一棵**abstraction tree**：

```
                            ┌── visual latent   z_t         ──┐
                            │                                  │
Raw observation  π_obs  ────┼── multimodal token (VLM)         ├───►  policy π_θ
                            │                                  │
                            ├── structured state  Ŝ_t          │
                            │                                  │
                            └── belief / posterior  b_t        ──┘
```

一句话 caption：**These are alternative abstractions, not stages of a universal pipeline.**

回到 §0.2 的语言：policy 侧真正的问题、**不是"我能不能吃下更长的 token 序列"**、而是**它选择树上哪一根枝条作为 conditioning input、并把这个选择写成 $\mathcal C_\pi + Q_{\mathcal C_\pi}$**。engineered state 走 $\hat S_t$ 那根、visual latent 走 $z_t$ 那根、multimodal token 事实上走一条 $\pi_{\mathrm{obs}}$ 之后立即 tokenize 的第四根——**三种走法对应三种 $q_\pi$、不是一种"更聪明"、一种"更笨"**。

**"多模态融合"这个词在 policy 侧的常见误用**就是把某根枝条上的 cross-attention 当成"已经在做多模态状态估计"——它不是。真正的 state abstraction 要求 $\hat S_t$ 里的字段**跨传感器家族保持一致的语义、可被 controller / policy / world model / diagnostics 四种消费者共同读**——9/14 §6 已经把这个约定立起来了、本节要做的是**从 policy 一侧再问一遍：$q_\pi$ 是不是被显式声明过、还是被 $e_\pi$ 悄悄替代了**。

## 3. Interface mismatch 的四种失败模式

一旦 §1 的两维坐标落到具体 policy 上、contract 语义会以四种**具体失败模式**表现出来。这四条不是理论担忧、是**部署里真会翻车的东西**。四条都是**接口层的事实**、**不是某一族的性格缺陷**。

### 3.1 Failure 1：Multi-hypothesis 被无声折叠

contract 里同一物理量可能有多个 hypothesis（"这个 track_id 是不是同一个物体"、"这个接触是 pad 还是 edge"）带不同 posterior weight。**如果一个 policy 的 $\mathcal C_\pi$ 与 $Q_{\mathcal C_\pi}$ 里没有为 hypothesis 结构显式声明一个 readout slot、$e_\pi$ 就可能在 concat + MLP 的默认动力下把它折成 posterior mean**——问题不在"这个模型一定会 collapse"、问题在**"接口没有为多 hypothesis 结构提供 structural guarantee"**。许多常见实现（visual latent 路线与 multimodal token 路线都在内）**没有提供显式、typed 的 hypothesis readout contract、因此 hypothesis collapse is an unprotected failure mode**——没有 slot 不等于一定 collapse、例如 transformer 完全可以把 `H1, H2, H3 + weights` 编码进 latent；只是这件事**没有接口层担保**、能不能成完全取决于训练分布与归纳偏置。**这已经是本文需要的最强 claim、再往前一步就会被 reviewer 击中**。

需要精确化的一点：**diffusion / flow-matching 在 action 侧的多模态性、不自动意味着 state-side 多 hypothesis 被保留**。action 分布可以是 multimodal（denoising 出来多条 action 轨迹）、但 conditioning 里如果 $\hat S_t$ 的 hypothesis 结构已经被 $e_\pi$ 折叠、action-side multimodality 只是在**已经丢了上游区分**的输入上做输出侧采样。这两个 multimodality 不是一回事、不能互相担保。

一句改口径的话：**本文批评的不是 Diffusion Policy 会 collapse、是"没有架构显式承诺 hypothesis preservation"这件事**。

### 3.2 Failure 2：Semantic correctness 缺乏接口层保证

contract 里的 frame 字段（`orientation_frame: "tool_flange"`、`reference_point: "contact_center"`、`convention: "right-handed"`）进入 policy 时、若被 tokenizer 变成 token 序列——**注意、这里的问题不是"transformer 学不到 frame transform"**。理论上 transformer 完全可以通过 attention + MLP 学出 $\tau_{p_2} = \tau_{p_1} + (p_1 - p_2) \times f$ 这类硬约束。**真正的问题是、tokenization 本身不提供任何"这个变换被以正确方式解释"的接口层保证**。模型**可以**学到、也可能**没**学到、取决于训练分布是否覆盖 frame 变更的对照样本、取决于 architecture 是否归纳偏置友好、取决于有没有 auxiliary constraint 显式施加这一约束。

**但更重要的是：这个 failure mode 不是 tokenizer 的专属问题**。一个 engineered vector `[force_x, force_y, force_z, frame_id]` 完全可以被 MLP 学坏——`frame_id` 变成一个 embedding 之后、MLP 一样可能把它当常量、或者只在训练分布覆盖到的那几个 id 上学对。**真正的 abstraction 是 representation + semantic constraint、不是 tokenization 本身**。把 Failure 2 归罪于 tokenizer 会掩盖本文 thesis 的 architecture-independence 一面。

**一个具体形态**：真机部署里把 wrench 的 `reference_point` 从 `sensor_flange` 改成 `contact_center`——数据前处理管线换了、token 序列几乎一模一样、但 $\tau$ 数值按 transport theorem 变了一大块。VLA 在这两个"看起来同分布"的 dataset 上分别训练、**可能分别收敛到相似但都错**的 policy。**"都错"不是因为 transformer 学不了 frame transform、是因为训练数据里没有足够的 frame-contrast pair 让它学到、并且没有任何约束强制它学**。这类 bug 只在**部署后**、cross-dataset / cross-embodiment 复用时才暴露。

**改口径的关键句**：tokenization 不抹掉语义、它只是**把语义从"接口保证"降级成了"训练经验"**——**engineered vector 也会做同样的降级**。这是本文与"tokenizer 就是不行"这种直觉的分水岭。

### 3.3 Failure 3：Temporal alignment broken by concat

contract 里不同 channel 的 `age` 是显式的（例如 proprio 1 ms、F/T 5 ms、vision 100 ms、tactile 30 ms）。若 policy 只看到 concat 后的向量、$\Delta t$ 分布就丢了。表现为 policy 用 vision 的"100 ms 前的世界"与 tactile 的"30 ms 前的接触"作决策、以为它们同步；其实 9/14 §4 已经强调过 **timestamp sync ≠ causal sync**、更不等于 decision-time causal consistency。**concat 掩盖的是 $\Delta t$、不是 $\Delta t$ 的语义**。

### 3.4 Failure 4：Safety-blindness to validity vs staleness

contract 明确区分 availability（这个通道今天有没有数据）、validity（这个数据是否有效、例如 calibration 是否过期）、age（有多旧）。policy 若只看数值、就把"stale 但 valid"与"missing 但 valid"混成一类、把"calibration drift 后无效"与"传感器掉线"当成同一种降级。9/14 §8.5 的 degradation chain 已经拆开了——**masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption**——policy 若在 input 端不接这个 chain、训练时的 augmentation 与推理时的 guardrail 都会挂错地方。

四条 failure 都是**接口问题**——换 backbone 不解决、只有把 $q_\pi$ 层显式化才解决。这一点之所以重要、是因为它把"要不要换更大 VLA"这个决策与"policy 侧接口要不要重写"这个决策**解耦**。

## 4. Policy 需要的三族 contract-read primitives

顺着 §3 的四条 failure、policy 侧真正需要的是**三族原语**——不是更大的 transformer、不是更多的数据。三族里的第三族（source structure）本身又要再拆三层、见 §4.3。

### 4.1 `mode_select`（假设层的读出方式）

面对 contract 里的多 hypothesis posterior、policy 需要在**读出**时明确选一种：MAP、posterior sample、expected-mixture、或者 **mass-preserving top-$k$**。

$$\text{read}\!\big(\{(\mu_i, \Sigma_i, w_i)\}_{i=1}^{K}\big) \;=\; \left\{\begin{aligned}
&\mu_{\arg\max_i w_i} && \text{(MAP、丢弃低权 hypothesis)}\\
&\textstyle\sum_i w_i\, \mu_i && \text{(posterior mean、显式声明的折叠)}\\
&\mu_i + L_i \epsilon,\;\; i \sim w,\;\; \epsilon \sim \mathcal{N}(0, I) && \text{(posterior sample)}\\
&\big\{(\mu_i, \Sigma_i, \tilde w_i)\big\}_{i \in \mathrm{top}\text{-}k} \;\cup\; \{w_{\mathrm{other}}\} && \text{(mass-preserving top-}k\text{)}
\end{aligned}\right.$$

v4 把这个 primitive 叫 "calibration-aware top-$k$"、reviewer 抓得准：截断重归一化**只解决 probability mass conservation、不解决 calibration**。$\sum_i w_i = 1$ 不代表 posterior calibrated。v4 已经把它改叫 **mass-preserving top-$k$**、calibration 单独走 §6.2 的评估：

$$\tilde w_i \;=\; \frac{w_i}{\sum_{j \in \mathrm{top}\text{-}k} w_j}, \qquad w_{\mathrm{other}} \;=\; 1 - \sum_{i \in \mathrm{top}\text{-}k} w_i.$$

$\tilde w_i$ 是**在 top-$k$ 内部**重归一化的权重、$w_{\mathrm{other}}$ 是**剩下的残差**、两样都保留给下游。

**v5 补一句限定（reviewer 抓到 v4 表述过头）**：$w_{\mathrm{other}}$ 只是一个 **aggregate residual mass**——它没有告诉下游 residual hypothesis 的**位置、covariance、likelihood、component identity**。所以 mass-preserving top-$k$ 能保证的是 **probability mass accounting**（下游至少能区分 retained mass 与 discarded residual mass）、但**不能保证 Bayesian update correctness**——严格 posterior update 还需要为 residual component 单独定义 sufficient statistics。primitive 层的语义边界就写在这里、不再往前一步。

至于"这份 posterior 到底 calibrate 没 calibrate"、那是**评估属性**、不是 primitive 的一部分。本文的 calibration diagnostic 挂在 §6.2 representation-level 上、包含 **ECE$_{\text{top-}k}$ / NLL / Brier / calibration curve / coverage-credibility 五条**、任何一条都独立于 top-$k$ 截断本身。primitive 层与 evaluation 层完全拆开、reviewer 就不会再问 "What exactly makes your top-$k$ calibration-aware?"。

关键**不是"mean 不能用"**——mean 是一种完全正当的 readout、只要它是**显式声明**的折叠。真正的失败模式是"接口没有为 hypothesis 结构提供任何 readout slot、$e_\pi$ 只能靠 concat + MLP 隐式合并、结果把 mean 当成了默认"。这一区分很关键：**本文反对的是"无声明的默认折叠"、不是"折叠"本身**。

### 4.2 `age_gate`：observation payload + metadata quintuple + derived trust

**接口设计的常见 bug** 是把 staleness trust 直接乘进 measurement：$x_c^\pi = \tau_c(\alpha_c) \cdot \mu_c$。这**改变了 observation 的物理值**——10 N 的力、age 100 ms、被乘成 3 N 之后、policy 输入里"3 N"这个数字**看起来**就像"3 N 力"、而不是"10 N 力、但 trust 降低"。这直接违反了 contract 想保护的那个 distinction：**$(F = 3\,\mathrm{N},\, a = 0)$ 与 $(F = 10\,\mathrm{N},\, \alpha = 100\,\mathrm{ms})$ 是两个不同的语义事件**。

v4 已经把字段结构重组过、v5 再做一次 notation cleanup（reviewer 抓得好：全文里 $a$ 同时表示 action 与 age、§6.3 的 $R_\pi(a)$ 特别容易读错）。**v5 起、age 字段的符号统一为 $\alpha_c$、availability 字段的符号统一为 $\iota_c$**、$\alpha$ 与 $\iota$ 与 $a$（action）不再撞名。

**(i) Observation payload**——测量值与它的 covariance：

$$\text{payload}_c \;=\; (\mu_c,\;\Sigma_c).$$

**只有当 $\iota_c = 1$ 时、这个 payload 存在**。$\iota_c = 0$ 意味着 estimator 这一时刻根本没有可以交付的 $\mu_c$、$q_\pi$ 只能读到"no-data"这一 fact。

**(ii) Temporal / operational metadata（五元）**——描述 payload 是怎么被采集、什么时候、通过哪个传感器、以及时效性的：

$$m_c \;=\; \big(\underbrace{\alpha_c}_{\text{age}},\;\underbrace{\ell_c}_{\text{latency / causal status}},\;\underbrace{h_c}_{\text{sensor health}},\;\underbrace{v_c}_{\text{validity (calibration)}},\;\underbrace{\iota_c}_{\text{availability}}\big).$$

**(iii) Derived trust（一条、不是 primitive）**——从上面五元算出来：

$$q_c \;=\; \tau_c(\alpha_c,\;\ell_c,\;h_c,\;v_c,\;\iota_c,\;\ldots).$$

省略号允许 task-specific 输入（例如 controller mode、当前 dynamics regime）。$\tau_c$ 具体形态（learned / analytic / piecewise）留在接口外作为 policy-specific 设计决策。

于是完整的 policy input 结构是 **"six primitive metadata fields + one derived trust field"**——$(\mu_c,\Sigma_c,\alpha_c,\ell_c,h_c,v_c,\iota_c) + q_c$、数得过来、reviewer 不用再问"到底是七个还是八个"。

**availability 与 validity 的语义差别**——这两个字段 v3 没拉开、reviewer 抓得对。2×2 小表：

|  | $v_c = 1$（valid） | $v_c = 0$（invalid） |
|---|---|---|
| $\iota_c = 1$（有 payload） | 正常证据、policy 可以按 §4.1 消费 | 有数据、但**不能当 valid evidence 用**（例如 calibration 过期）——policy 应该走 §5.3 safety 侧 |
| $\iota_c = 0$（无 payload） | 语义上不可能（没数据、validity 位没意义） | 通道掉线 / masked、policy 读"no observation"这一 fact |

一句话：**availability $\iota_c$ 说的是"有没有 payload"、validity $v_c$ 说的是"这份 payload 算不算 valid evidence"——两者正交、不能塞进同一个位**。§5.2 的 degradation chain（masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption）本质上就是这五元 $m_c$ 之间的**不同 pattern**、这一版把 augmentation 与 slot 一一对上了。

**若确实需要在下游做 gating、gating 应该作用在 uncertainty 上、不作用在 measurement 上**。但**上一版把 $\tilde\Sigma_c = \Sigma_c / \tau_c(\alpha_c)$ 写成"stale ⇒ 有效 uncertainty 被放大"的一般原则、reviewer 也抓到了**——真实情况是、stale observation 的正确处理**不一定是简单的 inflation**、更一般的形式是**把 latent state 向前传播**：

$$p(x_t \mid y_{t-\Delta t}) \;=\; \int p(x_t \mid x_{t-\Delta t})\, p(x_{t-\Delta t} \mid y_{t-\Delta t})\, dx_{t-\Delta t}.$$

机器人静止时 vision age 200 ms、measurement 可能仍然很准；机器人高速运动时同样 200 ms、predictive uncertainty 可能巨大。**inflation 与 propagation 是两种不同的处理方式、不是同义词**。本文把 $\Sigma / \tau$ 明确定位成 **a simple conservative approximation**：

$$\Sigma_c^{\mathrm{eff}} \;=\; \mathrm{Propagate}\!\big(\Sigma_c,\; \Delta t,\; u_t,\; f_{\mathrm{dyn}}\big) \qquad\text{（一般形式）}$$

$$\Sigma_c^{\mathrm{eff}} \;=\; \Sigma_c \,/\, \tau_c(\alpha_c, \ell_c, h_c, v_c, \iota_c) \qquad\text{（一种 conservative approximation）}$$

正确的表述是：**staleness should modify the policy's uncertainty model; uncertainty inflation is one conservative implementation, while predictive state propagation is another**。这一版把 $\Sigma / \tau$ 从"canonical"降级为 "one implementation"、并把 `Propagate` 作为一般形式挂在旁边。$\iota_c$ 与 $v_c$ **并列挂在 metadata 里**、不能塞进 $\mu$、也不能塞进 $\tau$。

这一改动看着小、实际上把整个 age_gate 的语义从"打折读数"修正到了"读数 + 关于读数的元信息 + 派生的 trust"——是 §0.2 $L_{\mathcal C}^{\pi}$ 定义的一个具体投影：把 measurement 与关于 measurement 的**事实**混在一个数值里、就是 $L_{\mathrm{projection}}$ 的直接来源。

### 4.3 `provenance / dependency / negative evidence`：三件事拆开、语义层级不同

9/14 §6.1 里把 `contributing_mask`、`correlated_with`、`negative_evidence` 都挂在 `provenance` 下——从 estimator 侧看合理（三者都是"这个字段的来源结构"）、但从 **policy 侧的读法**看、三者的语义层级其实不同：

| primitive | semantic role | policy 侧怎么读 |
|---|---|---|
| **provenance** | 证据从 *where* 来 | `contributing_mask` 拼进 policy input 作为 conditioning（**provenance_harden**） |
| **dependency** | 证据之间 *how* statistically related | dependency-aware fusion / conditioning（**dependency_aware_fusion**） |
| **negative evidence** | *what expected evidence failed to appear* | observability-conditional likelihood ratio（**negative_evidence_read**） |

#### 4.3.1 `provenance_harden`

把 `contributing_mask` 直接拼进 policy input。engineered-state head、visual-latent head 都能做（多几维向量）。这一条相对简单、不展开。

#### 4.3.2 `dependency_aware_fusion`（原 `dependency_gate`）

**上一版把 `correlated_with` 处理成 attention bias、用来 "suppress double-counting"——这个直觉不是总成立的**。相关性 ≠ 冗余 ≠ double counting。举个反例：camera depth + tactile contact + F/T wrench 三者高度相关时、恰恰**因为它们给出一致证据**、才应该得到高 confidence；把这种相关性 suppress 掉是错的。**真正的问题是：correlation 有没有在 estimator / uncertainty model 那边被 accounting for**——若 $\Sigma_{12}$ 已经在 Kalman-style fusion 里被建模、$P(x \mid y_1, y_2)$ 就自动正确；**只有当条件相关结构没建模、才存在 double-counting**。

因此本文把这条 primitive 从 `dependency_gate` 升级成 **`dependency_aware_fusion`**：它的语义不是"suppress 相关的 token pair"、而是"让 policy 或 fusion 层知道证据之间的统计关系、并做出相应反应"。

**attention bias 是众多实现之一**、且**必须写在 logit-level、不写在 softmax 之后的权重上**——v3 写成 $\mathrm{Attn}'_{ij} = \mathrm{Attn}_{ij} - \beta\cdot\mathbb 1[\cdot]$、reviewer 立刻指出：softmax 之后的 attention weight 做减法可能变负、不再是合法概率分布。v4 起把式子挪到 logit 上：

$$L'_{ij} \;=\; L_{ij} \;+\; b(R_{ij}), \qquad A_{ij} \;=\; \operatorname{softmax}_{j}\!\big(L'_{ij}\big).$$

其中 $R_{ij}$ 是从 $Q_{\mathcal C_\pi}$ 里编译出的 field-pair relation 强度（可以是 $\Sigma_{12}$ 相关系数、也可以是 `correlated_with` 图上的 shortest-path distance）、$b(R)$ 是一个**可以是正的、可以是负的、也可以是 learned 的** bias function。

关键差别是：**$b$ 的符号与形态不再暗示 "correlated → suppress"**——它只是把 dependency relation 作为 operator-level 的输入喂给 fusion、由 $\pi_\theta$ 或下游 loss 决定该怎么反应。这才真正符合 primitive 的名字：**dependency-aware fusion**、而不是 dependency suppression。

其它实现包括：covariance-aware fusion（把 `correlated_with` 转成 $\Sigma_{12}$、走 Kalman / factor graph）、hierarchical mixture（把相关字段聚到同一 latent 下、避免被独立采样）、以及让网络自己从 `correlated_with` 学一个 bias 矩阵（MLP head 的 `learned` 路径）。

同时老实承认：**`correlated_with` 是 field-level 语义关系、attention bias 是 token-pair relation**、两者之间需要一层 $R_{\text{field}} \to R_{\text{token}}$ 的映射、**这是一个尚未解决的编译问题**、值得作为独立研究方向。本文只把 `dependency_aware_fusion` 这条 primitive 立起来、上面那条 logit-level bias 公式是"一种实现"、不声称它是 canonical。

#### 4.3.3 `negative_evidence_read`（likelihood-side evidence）

**上一版把 negative evidence 写成 $\Lambda^{-}(H)$、方向硬编码为"没检测到 ⇒ 支持 $\neg H$"——reviewer 抓得对**：如果假设 $H$ = "obstacle exists"、$E^-$ = "没有检测到 obstacle"、那么 $\Lambda^-(H) < 0$ 是自然的；但如果 $H$ = "scene is clear"、方向就反过来了。人为规定"negative evidence 一定是正 or 一定是负"是脆弱的。

**v4 起把它一般化成一个纯粹的 conditional log-likelihood ratio**、负证据只是这个 LLR 的一个特化输入：

$$\boxed{\;\Lambda(E;\,H,\,\mathcal O) \;=\; \log \frac{P(E \mid H,\,\mathcal O)}{P(E \mid \neg H,\,\mathcal O)}.\;}$$

其中 $E$ 是**任意一份可观察证据**（正证据 $E^+$ = "检测到了"、负证据 $E^-$ = "本应检测到但没有"）、$\mathcal O$ 编码 observability / sensor health / field of regard / calibration status。**"negative evidence" 就是把 $E = E^-$ 塞进这个 LLR、不再需要单独定义一个 primitive**。

文字解释写清一句：**sign of $\Lambda$ depends on the hypothesis being tested**——$H$ 定义的方向决定同一份 $E^-$ 是 $\Lambda > 0$ 还是 $\Lambda < 0$、这是贝叶斯证据理论的标准约定、不需要本文硬编码。

这才是"本应看到但没有"的正确刻画——**只有当 $\mathcal O$ 说该看到的时候、$E^-$ 才对 $H$ 有 likelihood 意义**。这也直接接上了 9/14 §6 里的 observability 字段：**negative evidence 是 likelihood-side evidence、不是 provenance**。

具体到 policy 侧：把 $\Lambda(E;H,\mathcal O)$（或 $\exp(\Lambda)$ 的 logit）作为一个 field 拼进 $[\hat S]_{\sim_\pi}$、让 policy 或 belief update 消费。这条 primitive 目前工程实现最薄、但它对 hypothesis-ranking 的影响往往最大。

三个 sub-primitive 的关系：**provenance 讲 `where evidence came from`、dependency 讲 `how evidence is statistically related`、negative evidence 讲 `what expected evidence failed to appear`——三者共同支撑 §0.2 里的 source-structure invariant**。

### 4.4 三族 primitives 与三条 invariants 的对应

回到 §0.1 的图：**mode / temporal / source** 三条 contract-relevant invariants 分别对应 **mode_select / age_gate / (provenance_harden + dependency_aware_fusion + negative_evidence_read)**。三条 invariant 缺一、§3 的四条 failure 至少复发两条。三族 primitive 也不是"接口设计题的完整答案"——observability / identifiability、frame convention、contact set 这三类 contract 字段还有各自更专门的读法（9/14 §7、§8.6 有对应讨论）、本文只处理**最容易在 $\Pi_\pi$ 层被无声破坏的这三条**。

## 5. Training-time 与 Deployment-time 的连锁后果

一旦 policy 输入端接了 §4 的 primitives、训练目标、augmentation、safety filter、evaluation 四处都要跟着改。**接口不是免费的**——但改动是**局部的、可控的**。

### 5.1 两类训练约束：representation-side probe + 三类 intervention consistency

一个自然的错误是：**为了"让 policy 用 contract"、要求 policy 输出 validity / hypothesis 的预测头**——这实际上是把"用 contract"偷换成了"复制 contract"、方向不对。**policy 完全可以只吃 contract、不吐 contract**——auxiliary prediction head 不是必要条件。

#### 类别 A · Representation-side probe（诊断）

给 policy 的中间表示 $z^{\pi}$ 挂几个 probe head、尝试从 $z^{\pi}$ 预测 contract 里的 `age`、`validity`、`observability`、hypothesis posterior。**这些 probe 不参与主 loss、只用来测**：probe 预测得好、说明 policy 内部保留了这些信息；probe 预测得差、说明 contract 在 $e_\pi$ 里已经被压掉了。$L_{\mathrm{probe}}$ 可以以很小的权重加到主 loss 上作为 regularization、但**它的诊断价值大于训练价值**。

$$\mathcal{L}_{\mathrm{total}} \;=\; \mathcal{L}_{\mathrm{action}} \;+\; \underbrace{\alpha\, \mathcal{L}_{\mathrm{probe}}}_{\text{weak regularization, mainly diagnostic}} \;+\; \underbrace{\sum_{\mathcal{C}} \gamma_{\mathcal{C}}\, \mathcal{L}^{\mathcal{C}}_{\mathrm{consistency}}}_{\text{Type I / II / III, see below}}$$

**但 probe 有一个上一版没解决的漏洞**——reviewer 抓得对：假设 $age$ 与 image embedding 高度相关（例如 "scene 越复杂、sensor 处理越慢、age 越大"）、那么 probe 从 $z_\pi$ 里可以很容易预测 age：

$$\mathrm{Acc}(\alpha\mid z_\pi) \;=\; 99\%.$$

**这并不证明 age 被 $e_\pi$ 保留了**——完全可能只是 image 里的 scene difficulty 泄漏。**正确的诊断应该用 conditional / nuisance-controlled probe**：

$$\text{Retention}_{\mathrm{cond}}(\alpha) \;=\; I(\alpha;\, z_\pi \mid o),$$

即在 raw observation $o$ 给定的条件下、$z_\pi$ 是否还**独立地**携带 age 信息；或者在 benchmark 里做"**same observation, different metadata intervention**"——固定 $o$、只改 $m_c$、看 $z_\pi$ 的响应。这与 §0.2.1 Property B 用 conditional MI 而不是差分 MI 的哲学一脉相承：**凡是"raw input 里已经有 proxy"的字段、probe 都必须 conditional**。

#### 类别 B · Intervention-consistency constraint（核心）

**上一版把这一类笼统写成 $D(\pi_\theta(\hat S), \pi_\theta(T_{\mathcal{C}}(\hat S)); \rho_{\mathcal{C}})$、reviewer 立刻问 $\rho_{\mathcal{C}}$ 从哪来**——四种 $T_{\mathcal{C}}$ 各自的"响应规律"不一样、有一类甚至根本不该预设"必须响应"。这一版按响应强度把 intervention 分成三档。

**Type I · Exact invariance / equivariance**（最干净的一档）。变换在 policy input 上有一个合法的 action-side 对应 $T^{\mathcal{C}}_\pi$、要求：

$$\pi_\theta\!\big(T^{\mathcal{C}}(\hat S),\, o,\, \ell\big) \;=\; T^{\mathcal{C}}_\pi\!\big(\pi_\theta(\hat S,\, o,\, \ell)\big).$$

典型：**frame transform**——把 `reference_point` 从 A 移到 B、$\tau$ 按 transport theorem 变换、**要求 action 侧做对应的坐标变换**（$\pi(T_g S) = T_g^A \pi(S)$）。类似：**permutation of equivalent hypotheses**（同 posterior weight 的 hypothesis 互换、action 分布必须等价）。这一档可以写成硬 loss、$\mathcal{L}_{\mathrm{consistency}}^{\mathrm{I}} = \|\pi_\theta(T\hat S) - T^\pi \pi_\theta(\hat S)\|^2$。

**Type II · Order-constrained response**（次强的一档、**上一版叫 monotone response、这一版把它数学化**）。"monotone" 不是随便就能用的词——只有当 $M(\cdot)$ 的值域上有偏序、并且 response functional 是**明确定义的标量或全序**时才能谈单调。上一版把 variance、action norm、fallback probability、covariance PSD 全塞进同一个 $\preceq$、reviewer 抓得对：**这几种 $\preceq$ 根本不是同一个 order**。

正确的提法是：先给出**contract intervention 的 severity partial order** $T_1 \preceq_{\mathcal C} T_2$（例如 "age 越大 = 越严重"、"observability 越低 = 越严重"、"validity 位为 false = 比 age 高更严重"——order 由 $\mathcal C_\pi$ 与 $Q_{\mathcal C_\pi}$ 定义、**不是由 policy 定义**）、再指定一个**response functional**

$$r:\mathcal P(\mathcal A) \;\longrightarrow\; \mathbb R$$

（可以是 $P(\text{fallback})$、$\mathbb E[\|a\|]$、$P(\text{stop})$、$\mathbb E[\mathrm{safe\_margin}]$ 之类、每个是 scalar、有全序 $\le$）、然后要求：

$$T_1 \preceq_{\mathcal C} T_2 \quad\Longrightarrow\quad r\!\big(\pi_\theta(T_1\hat S)\big) \;\le\; r\!\big(\pi_\theta(T_2\hat S)\big).$$

**这才是严格意义的 monotonicity**。举例：$r(\pi) = P_\pi(\text{fallback})$、$T_1$ = "age 从 5 ms 到 20 ms"、$T_2$ = "age 从 20 ms 到 200 ms"、则 $T_1 \preceq_{\mathcal C} T_2$ 且要求 $P_\pi(\text{fallback}\mid T_1) \le P_\pi(\text{fallback}\mid T_2)$。

但真实 policy 完全可能是**分段的**——

```
α < 50 ms      →   正常控制
50 ms ≤ α < 100 ms → fallback
α ≥ 100 ms     →   stop
```

这种 response **不 monotone、但是合法的 contract-specified response relation**。所以 Type II 的正确名字应该是 "**order-constrained response**"、**monotonicity 只是它的一个特例**。写成 loss 是：

$$\mathcal{L}_{\mathrm{consistency}}^{\mathrm{II}} \;=\; \sum_{T_1 \preceq_{\mathcal C} T_2} \max\!\big(0,\; r(\pi_\theta(T_1 \hat S)) - r(\pi_\theta(T_2 \hat S)) + \delta\big).$$

$\delta$ 是 margin、$r$ 与 $\preceq_{\mathcal C}$ 都必须在 $\mathcal C_\pi$ 里写死、不能事后凑。

**Type III · Unconstrained intervention**（最弱、也最重要的一档）。**不预设 policy 必须变化**——典型是 **provenance removal**：拿掉一个 contributing sensor、若另一个 sensor 完全冗余替代、**最优 action 可以完全不变**。这一档的正确提法是：**when the removed evidence was decision-relevant, does performance degrade?** 也就是把它交给 §6 的 CAG 面板去测、而不是训练时强加响应规律。写成 loss 就是：**不做任何 intervention consistency、只在 evaluation 阶段做 ablation**。这一档存在本身是对上一版的一个纠正——上一版把四种 $T_{\mathcal{C}}$ 一视同仁地塞进"要求响应"、是把 Type III 错当成了 Type II。

**这三类都不要求 policy 显式预测什么**；它们要求的是**响应函数符合 contract 语义档位、或者被允许符合"不变"**。这才是 §0.2 "$q_\pi$ 是显式声明的 quotient" 定义的**训练侧对应物**。相比"加几个 auxiliary 预测头"、这套三类约束更贴合本文 thesis、也更有研究味：**我们建议的不是让 policy 复制 contract、是让 policy 在 contract 变换下按对应的响应档位响应**。

### 5.2 Augmentation：degradation as causal operator

9/14 §8.5 强调 **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption**——六种降级各有不同的**因果起源**与不同的**下游读法**。上一版试图给每类降级标"对应哪个 slot"、reviewer 抓得对：**这种 one-to-one 映射过度简化**。举个具体的：

- **latency** 的 primary effect 是抬 $\alpha_c$、但如果没有做 compensation、secondary effect 是让 $\mu_c$ 变成"延迟时刻的真实值"——同一份 augmentation 同时动了两个 slot。
- **bias** 的 primary effect 是 shift $\mu_c$、但 secondary effect 通常还包括 inflate $\Sigma_c$（因为系统意识到 calibration 不可信）、甚至 flip $v_c$。
- **corruption** 可能同时改 $\mu_c, \Sigma_c, v_c, h_c$ 四样。
- **masking** 的 primary effect 是 $\iota_c = 0$、secondary effect 是让下游 $q_c$ 变成 undefined、$\Sigma_c^{\mathrm{eff}}$ 必须回落到 prior。

这一版把每条降级**改成一个 causal operator**：

$$T_d:\;(\mu_c,\,\Sigma_c,\,m_c)\;\longmapsto\;(\mu_c',\,\Sigma_c',\,m_c'),\qquad d \in \{\text{mask},\,\text{missing},\,\text{stale},\,\text{latency},\,\text{bias},\,\text{corruption}\}.$$

一句话原则：**each degradation has a primary semantic effect and potentially secondary effects on other fields**——aug pipeline 里必须**逐条列出** $T_d$ 具体动了哪些 slot、动了多少、而不是"aug 类别 → 单个 slot"。§4.2 的三组结构（payload / metadata / derived trust）在这张表上直接可用：**$T_d$ 作用在 $(\mu,\Sigma)$ 上是 payload-level corruption、作用在 $m_c$ 上是 metadata-level augmentation、$q_c$ 必须是从 modified $m_c$ 派生出来的、不能被 aug 端直接改写**（否则就是"aug 端在作弊、部署端读不到"）。

**上一版这里有个 reviewer 抓到的 shortcut 风险**："给每个 episode 打 degradation label、policy input 里显式携带"——如果 `degradation = stale_vision` 这个 label 直接作为 policy input、policy 会学成 $a = f(o, \text{label})$；但**真实部署里 label 本身未必可靠**、这个 shortcut 在训练里看起来无害、部署里就是灾难。这一版明确区分两样东西：

**Observed metadata** $m_c = (\alpha_c, \ell_c, h_c, v_c, \iota_c)$——由 estimator 或 sensor driver 直接得到、可以进 contract、可以进 policy input。**这是 §4.2 metadata quintuple 的来源**。

**Latent degradation class** $d_c \in \{\text{missing}, \text{stale}, \text{bias}, \text{corrupt}, \ldots\}$——augmentation 时你**知道**注入了哪一类 $T_{d}$、但部署时这个类是**latent 的**、只能由 contract estimator 从 $m_c$ 序列里推断、或只作为训练 annotation 用来加权 loss / 采样、**不能默认作为 ground-truth input 塞进 policy**。

具体做法：aug pipeline 采样 $d_c$、按 $d_c$ 通过 $T_{d_c}$ 生成 $(\mu_c', \Sigma_c', m_c')$、然后把 $m_c'$ 作为 policy input、$d_c$ 只作为 loss weighting 与 evaluation 分层的 key。**这不是 curriculum、是 conditioning on observed metadata**——差别在于 conditioning 让 policy 看见的是可靠的 $m_c'$、而不是不可靠的 $d_c$。§5.1 类别 B 的 Type II order-constrained response 与 degradation-conditioned aug 天然配对——**aug 端造 $T_{\mathcal{C}}$ 的 $m_c$ 变化、loss 端测 policy 对 $m_c$ 变化的响应是否符合 $\preceq_{\mathcal C}$ 与 $r$**。

### 5.3 Safety filter 与 contract 的接口：constraint certification（v5 三态化）

约束层（CBF / shield / runtime verifier）**必须读 $a_{\mathrm{proposed}}$**、否则它 filter 什么——这一点不用退让。但本文更精确的 claim 是：**safety filter 不应该把 policy 的 confidence 或 estimator 的通用 validity 位当成约束成立性的唯一证据；它应该直接访问 constraint-relevant evidence**。

**这里必须区分两种 validity**。Estimator 侧的通用 `validity` 位说的是"这条 measurement 从 calibration / sensor health 的角度看是否有效"——这是 field-level 的 general-purpose 声明。Safety 侧真正关心的是"**这条约束在这个时刻、这个 predicate 下是否 valid**"——这是 constraint-level 的语义。两者并不相同：`validity=true` 不意味着 constraint estimate 对**这条 safety predicate** 有效；例如"关节力矩读数 calibration OK" ≠ "当前接触估计足以支撑 collision constraint"。

本文建议 safety filter 读到的、是一个 constraint 特定的组合量：

$$v_j^{\mathrm{constraint}} \;=\; g\!\Big(\text{field validity},\; \text{observability},\; \text{hypothesis posterior},\; \text{age},\; \text{model coverage}_j\Big).$$

$g$ 是 constraint-specific 的组合规则（例如 CBF 那侧要求"距离估计在当前 hypothesis 下、observability 充分、且 dynamics model coverage 到当前状态区域"）、$v_j^{\mathrm{constraint}}$ 才是 safety filter 真正应该读的**constraint-relevant evidence**。9/14 §7 讲过 **track_id 是 hypothesis**——那么 safety filter 也不能只信 track_id 匹配、还要看 hypothesis posterior 是否稳定；两个 track 是否 merge / split、直接决定"这个障碍距离"的可信度。

**v5 补：三态 certification**。上一版把 $v_j^{\mathrm{constraint}}$ 写成 binary 位、reviewer 抓得对——**"证据不足"与"约束不成立"是两件不同的事**。举三个具体 case：invalid sensor 是 unknown、obstacle detected 是 unsafe、obstacle absent with high observability 是 safe。用 binary $v = 0/1$ 表达这三种情况会把它们全塞进同一个值、safety filter 拿不到必要的信息。所以本文引入三态：

$$\boxed{\;\mathrm{certification}_j \;\in\; \{\text{safe},\;\text{unsafe},\;\text{unknown}\}.\;}$$

对应的反应规则：**safe → 可以 relax**（例如把 minimum distance 收回正常）、**unsafe → tighten / stop**、**unknown → conservative fallback 或 tighten**（因为不能确定、所以按更谨慎的一侧处理）。这一改让 safety contract 更接近真正的 runtime safety semantics——$v_j^{\mathrm{constraint}} = 0$ 不再意味着 "constraint false"、它意味着 **"evidence is insufficient to certify the constraint predicate"**（对应 unknown 状态）。

**这一节最重要的一句话（v5 加了一个词）**：

> **$v_j^{\mathrm{constraint}}$ does NOT turn off the constraint when evidence is invalid or unknown.**
> **Invalid evidence ALONE cannot justify relaxing the constraint.**
> Equivalently: **Relaxation requires sufficient valid evidence; invalid evidence alone is never sufficient.**

上一版这里写得含糊、读起来像"validity 挂了、constraint 就可以 off"——**这在 safety 语义上是危险的**、也容易被 reviewer 用一个反例打穿（camera invalid 但 lidar valid 且已充分证明 obstacle 不存在、此时 constraint 完全可以 relax）。加上 "alone" 之后、语义变成："**invalid evidence 单靠自身不能成为 relax 的充分条件**；如果**另一路独立 valid evidence 已经充分证明 constraint 不成立**、那就是合法 relax"。这一改把 counterexample 挡在门外、同时不削弱本文的核心 safety claim。三种 reaction：

1. **fallback**：切到一个更保守的 controller 或 planner；
2. **conservative tightening**：**加大** constraint margin（例如把 minimum distance 从 20 cm 抬到 50 cm）、因为"证据不够 = 更谨慎"；
3. **stop**：完全 stop policy、等观测恢复。

**三种反应都不是"constraint off"**——除非另一路 valid evidence 已把 certification 显式推到 **safe**。安全语义上的核心事实是：**positive evidence（且充分）才能 justify relaxing a constraint；absence of valid evidence 永远不能单独做到这件事**。

这条也接上了 §4.3.3 的 negative evidence：**"本应看到但没有"** 会让 $\mathrm{certification}_j$ 变 unknown——例如雷达在这个角度什么都没扫到、$\Lambda(E^-; H_{\text{clear}}, \mathcal O)$ 变正（"clear" hypothesis 下"没看到东西"是自然结果、但反过来"有障碍" hypothesis 下"没看到东西"是不自然结果——**方向依赖 $H$ 的定义、见 §4.3.3**）、collision constraint 的 certification 应该保持 unknown、甚至**保守收紧**、不能因为 policy 的 belief 乐观就把 certification 推到 safe。

**这三样（constraint-specific certification、observability、negative evidence）不进 safety filter、filter 就会用 policy 的 belief 反推约束成立性**、这在低 observability 区域特别危险——policy 的 belief 之所以乐观、是因为它读不到 contract 里的 observability / validity / negative evidence、**filter 如果同样读不到、两者一起盲**。

### 5.4 与 9/10 Part 3 evaluation 的呼应

Sim utility 三维（prediction / ranking / decision）里、policy-side 的 evaluation 主要看 **decision** 这一维——但要加一个 **contract-preservation 维度**：把 contract 拆掉之后 policy 的表现下降多少、就是它对 contract 依赖度的**下界**。这一维度对应 §6 的 CAG 指标、且需要 oracle baseline 来界定解释边界、以及**明确的 retraining protocol**（§6.5）。

## 6. Evaluation：四种 compliance evidence、不是四个 metric

对应 §4 的 primitives 与 §5.1 的三类 intervention、本文提出**四种 compliance evidence**——**不是一个四层 metric panel、是四类各自回答不同问题的证据**。上一版把它们写成 metric hierarchy、reviewer 抓得对：probe / CAG / safety-pass **任何单一 score 都不构成 semantic compliance**、必须**多证据合流**。

$$\boxed{\;\text{Compliance Evidence} \;=\; \big\{E_{\mathrm{semantic}},\;E_{\mathrm{representation}},\;E_{\mathrm{decision}},\;E_{\mathrm{safety}}\big\}.\;}$$

四类各自回答一个不同的问题：

| 类型 | 回答什么 |
|---|---|
| **$E_{\mathrm{semantic}}$** | **Does the policy respond correctly to a known semantic transformation?** |
| **$E_{\mathrm{representation}}$** | **Is the contract information recoverable from the representation (given the side inputs)?** |
| **$E_{\mathrm{decision}}$** | **Does contract structure change task utility when it should?** |
| **$E_{\mathrm{safety}}$** | **Does degradation cause conservative / required guardrail behavior?** |

三条明确的 caveat：**probe ≠ semantic compliance、CAG ≠ semantic compliance、safety pass ≠ representation retention**。这四类证据**不能互相顶替**、也不能压成一个 scalar。以下按类展开。Temporal 与 source 两条 invariant 的专项测试分别落在 $E_{\mathrm{decision}}$ 的 **SDS** 与 **$\Delta J_{\mathrm{where}} / \Delta J_{\mathrm{dep}} / \Delta J_{\mathrm{neg}}$** 切片——它们不是新指标、是 decision 层的 specialized 观察。

### 6.1 Semantic evidence $E_{\mathrm{semantic}}$：Invariance / Equivariance Test

对应 §5.1 Type I。给定一组已知 $T^{\mathcal{C}}_\pi$ 的 contract 变换（frame / coordinate / hypothesis permutation）、测：

$$\mathrm{Equiv}(\mathcal{C}) \;=\; \mathbb{E}_{\hat S}\!\left[d\!\left(\pi_\theta(T^{\mathcal{C}}\hat S),\; T^{\mathcal{C}}_\pi\,\pi_\theta(\hat S)\right)\right].$$

$\mathrm{Equiv} \to 0$ 是硬要求、$\mathrm{Equiv} \gg 0$ 意味着 $e_\pi$ 学坏了、或者 $q_\pi$ 直接把这一层 quotient 丢了。这一类是最"干净"的一类、因为规则是 mathematically defined 的、不需要 oracle 也不需要 $J$ 的定义。§3.2 Failure 2 的 severity 可以直接由 $\mathrm{Equiv}(\text{frame})$ 量化。

### 6.2 Representation evidence $E_{\mathrm{representation}}$：Conditional Probes + HPC / HSS（v5 拆两种 counterfactual） + Calibration

**Conditional probe**（§5.1 类别 A 升级版）：Retention$_{\mathrm{cond}} = I(\text{field}; z_\pi \mid o)$——固定 raw observation $o$、测 $z_\pi$ 里还**独立**携带多少 contract 信息。这是**避免 image-proxy 泄漏**的必要形式。

**v4 的 HPC 用 $T_k^{\mathrm{hyp}}$、v5 把它拆成两种 intervention**——reviewer 抓得非常准：**"改变 hypothesis、但固定 raw observation"并不是天然合法的 counterfactual**。真实图像 $o$ 明明显示物体在左边、你把 contract 改成"$H_2$：物体在右边"、同时保持同一张图像——这时候你到底测的是什么？两种截然不同的东西被 v4 的记号混在了一起。v5 拆成：

**(A) Contract-only intervention**——$T_k^{\mathrm{contract}}:\hat S \mapsto \hat S_k$、**raw observation 不变**。它测的是：

> **policy 是否会响应 contract semantic change？**

这是本文真正想要的 **interface compliance**。但**不要**再把它叫"$H_k$ as true latent"、因为世界并没有变、变的只是 contract。它测的是"contract parser 对一份自相矛盾输入的响应能力"、不是"policy 在另一个真实世界的决策能力"。

**(B) World-consistent counterfactual**——同时改 $(o, \hat S)\mapsto(o_k, \hat S_k)$、其中 $o_k$ 与 $\hat S_k$ 都来自 simulator / renderer / privileged state、保证 observation 与 contract **一致**。它测的是：

> **policy 是否能在另一个真实世界 hypothesis 下正确行动？**

这是 **decision competence**。

两种 benchmark 明确分开、否则 reviewer 一句 **"Is your counterfactual intervention semantically consistent with the observation?"** 就能把整段拆开。本文下面公式默认走 (A) $T_k^{\mathrm{contract}}$ 测 interface compliance、并把 (B) $T_k^{\mathrm{world}}$ 留给下一篇 benchmark paper。

**Hypothesis Coverage (HPC)**：

$$\mathrm{HPC} \;=\; \frac{1}{K} \sum_{k=1}^{K} U\!\big(\pi_\theta(T_k^{\mathrm{contract}}(\hat S)),\;\mathcal{A}^{*}_k\big).$$

$U(\cdot, \mathcal A^*_k)$ 是 policy 输出与 counterfactual contract 下**理想 action set** $\mathcal A^*_k$ 的 utility、$\mathcal A^*_k$ 与 §6.5 的 oracle 一样在 benchmark 里由 privileged simulator state / oracle planner / offline expert rollouts 构造。这一改把 HPC 与全文 intervention philosophy 对齐——**HPC 测的是"如果 contract 被 counterfactually 换成 $H_k$、policy 会不会响应"、不再宣称"$H_k$ 就是 true latent"**。

**Hypothesis Separation Score (HSS)**——**必须只在 action-relevant hypothesis pairs 上平均**、reviewer 抓到了 gaming：如果 $\mathcal A^*(H_1) = \mathcal A^*(H_2)$、policy 输出不同不是优点、**是 noise**。定义 action-relevant pair set：

$$\mathcal R \;=\; \big\{(i, j): \mathcal A^{*}_i \not\equiv \mathcal A^{*}_j\big\}.$$

HSS 只在 $\mathcal R$ 上算：

$$\mathrm{HSS} \;=\; \frac{1}{|\mathcal R|} \sum_{(i, j) \in \mathcal R} D\!\big(\pi_\theta(\cdot \mid T_i^{\mathrm{contract}}\hat S),\;\pi_\theta(\cdot \mid T_j^{\mathrm{contract}}\hat S)\big).$$

**separation is useful only when distinctions are decision-relevant**——这一句必须写死、否则 HSS 会被 policy 用"每个 hypothesis 输出一个不同的随机 action"这种 gaming 拉满。

$D$ 的选择与 §0.2.2 的 $D_{\mathcal A}$ 用同一族——**测的是两个 action distribution 是否支持不同的 admissible / optimal action set**——HSS 与 $L_{\mathrm{decision}}$ 语义闭环。

HPC 与 HSS 一起才对应 decision-relevant semantic preservation（§0.2.1 Property A′）——**coverage** 保证"每个 action-relevant hypothesis 都被 support"、**separation** 保证"policy 保留了 action-relevant 的 hypothesis distinction"、**同时不惩罚那些不该区分的 pair**。这三条限制一起才让 representation evidence 有意义。

**Calibration diagnostic**——$\mathrm{ECE}_{\text{top-}k}$、NLL、Brier、calibration curve、coverage / credibility 五条并列（§4.1 已经把它们从 primitive 里剥出来、这一节是它们真正的家）。

### 6.3 Temporal slice：Staleness Response Compliance (SDS)（v5 记号 + baseline 双修）

**v4 版本 SDS 有两个问题（reviewer 都抓到了）**。第一、$R_\pi(a) = \pi_\theta(\cdot|\mathrm{do}(a_c = a), o)$ 里 $a$ 同时是 age 参数与全文里代表 action 的变量、notation collision。第二、$R^*(a)$ 作为"唯一 oracle response curve"过强——expert A 可能"age > 100 ms 就 fallback"、expert B 可能"通过 dynamics prediction 补偿后仍然正常控制"、两者都合理、强迫唯一 $R^*$ 会把合理的分段响应误判成 failure。

**v5 一次改两处**。记号侧：$\alpha$ 用作 age 值（§4.2 已经把字段名从 $a_c$ 改成 $\alpha_c$）、$a$ 从此只代表 action：

$$R_\pi(\alpha) \;=\; \pi_\theta\!\big(\cdot \,\big|\, \mathrm{do}(\alpha_c = \alpha),\, o\big).$$

baseline 侧：SDS 不再对着单一 $R^*$ 定义、而是对着 **contract-permitted response set** $\mathcal R_{\mathcal C}(\alpha)$ 定义——这个 set 由 §5.1 已经写进 $\mathcal C_\pi$ 的 $\preceq_{\mathcal C}$ 与 $r$ 声明、允许 piecewise、允许 flat、只要**不违反 relation 就是合法**：

$$\boxed{\;\mathrm{SDS} \;=\; D\!\big(R_\pi,\;\mathcal R_{\mathcal C}\big),\;}$$

$D$ 是 "distance from a curve to a set of legal curves"、可以是 sup-based violation measure $\sup_{\alpha_1 \preceq_{\mathcal C} \alpha_2} \max\!\big(0, r(R_\pi(\alpha_1)) - r(R_\pi(\alpha_2)) + \delta\big)$（等价于 §5.1 Type II hinge loss 在 evaluation 上的复用）、也可以是别的 curve-set divergence。**oracle policy curve $R^*$ 只是 $\mathcal R_{\mathcal C}$ 的一种 baseline、不是定义本身**。这样 SDS 就与 §5.1 Type II "order-constrained 不一定 monotone" 完全一致、不再强迫所有任务共用一条唯一正确的 staleness response。

响应属性 $R$ 根据任务定义、可以取 **variance / action norm / fallback probability / safety margin / stop probability**——**必须与 §5.1 Type II 里声明的 $r$ 是同一个**、否则训练与评估各说各话。平坦不代表差、只要与 $\mathcal R_{\mathcal C}$ 里某一条 legal curve 匹配即可。

导数形式仍保留、作为 response shape 的一种局部刻画：

$$\left.\frac{\partial\, \mathbb{E}\!\big[\pi_\theta(\cdot \mid \mathrm{do}(\alpha_c = \alpha),\, o)\big]}{\partial \alpha_c}\right|_{\alpha}\quad\text{与 } \mathcal R_{\mathcal C} \text{ 里 legal curves 的同阶导数分布比较}.$$

### 6.4 Source slice：$\Delta J_{\mathrm{where}}$、$\Delta J_{\mathrm{dep}}$、$\Delta J_{\mathrm{neg}}$（v5 三分）

**v4 把 $\Delta J_{\mathrm{prov}}$ 写成 "provenance + correlated_with" 一起 ablate、reviewer 抓得对——你花了一整节说 provenance ≠ dependency ≠ negative evidence、metric 又把它们揉在一起、那就回答不了"到底哪一条 primitive 起作用"**。v5 拆成三条、每条只 ablate 一个 primitive：

$$\Delta J_{\mathrm{where}} \;=\; J\!\big(\pi_\theta \mid \text{provenance}\big) \;-\; J\!\big(\pi_\theta \mid \text{provenance} = \varnothing\big),$$

$$\Delta J_{\mathrm{dep}} \;=\; J\!\big(\pi_\theta \mid \text{correlated\_with}\big) \;-\; J\!\big(\pi_\theta \mid \text{correlated\_with} = \varnothing\big),$$

$$\Delta J_{\mathrm{neg}} \;=\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda\big) \;-\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda = \varnothing\big).$$

三条各对应 §4.3 的三个 primitive：**$\Delta J_{\mathrm{where}}$ 测 `contributing_mask` 消费情况、$\Delta J_{\mathrm{dep}}$ 测 `correlated_with` 消费情况、$\Delta J_{\mathrm{neg}}$ 测 conditional LLR 消费情况**。$J$ 一律 higher-is-better。每条都可以独立爆零、互不顶替——这是"三个 primitive 语义独立"这一节 claim 在**评估层的对应兑现**。如果 reviewer 再问"性能改善是哪一个 primitive 起作用"、v5 可以一条条回答、v4 不能。

### 6.5 Decision evidence $E_{\mathrm{decision}}$：Contract Ablation Gap（CAG）+ Oracle baseline + Retraining protocol（v5 加限定）

**给定 contract 的一个特定 collapse 算子 $\mathrm{collapse}_X$**（把 $X$ 这一层 contract structure 无声明地折叠掉）、

$$\mathrm{CAG}_X \;=\; J_{\mathrm{full}} \;-\; J_{\mathrm{collapse}_X}, \qquad J_{\mathrm{full}} \equiv J(\pi_\theta \mid \hat S),\;\; J_{\mathrm{collapse}_X} \equiv J(\pi_\theta \mid \mathrm{collapse}_X(\hat S)),$$

四类 collapse 各对应一条 invariant：

- $\mathrm{collapse}_{\mathrm{hyp}}$：把 hypothesis set 折成单 Gaussian 或单点估计。
- $\mathrm{collapse}_{\mathrm{age}}$：把所有 channel 的 `age / health / latency / availability` 抹平。
- $\mathrm{collapse}_{\mathrm{prov}}$：drop `contributing_mask` 与 `correlated_with`、让 policy 无区别地 attend。
- $\mathrm{collapse}_{\mathrm{neg}}$：drop $\Lambda(E; H, \mathcal O)$。

**v5 必须写死的一句（reviewer 抓到 v4 隐藏问题）**：

> **The collapsed condition is evaluated with the same trained policy, without retraining, unless explicitly defined otherwise.**

因为**同一份训好的 policy 上做 collapse** 与**用 collapsed representation 重新训一份 policy**、是两个完全不同的问题：

- **不重新训练**（本文默认）：测的是 **trained policy 对 contract information 的 sensitivity**——这是 compliance evidence。
- **重新训练**：测的是 **这个 representation 本身是否足以支持任务**——这是 architecture / representation comparison。

两件事都成立、但不能混用同一个符号。v5 显式定义两个量：

$$\mathrm{CAG}_X^{\mathrm{fixed}} \;=\; J_{\mathrm{full}}^{\theta^*} - J_{\mathrm{collapse}_X}^{\theta^*},\qquad \mathrm{CAG}_X^{\mathrm{retrained}} \;=\; J_{\mathrm{full}}^{\theta^*} - J_{\mathrm{collapse}_X}^{\theta^*_X},$$

其中 $\theta^*$ 是原训练好的参数、$\theta^*_X$ 是用 collapsed representation **重新训练**出来的参数。**本文的 compliance evidence 只走 $\mathrm{CAG}^{\mathrm{fixed}}$；$\mathrm{CAG}^{\mathrm{retrained}}$ 是 architecture 论文的问题、不是本文的**。

**再加 oracle privileged-state baseline**：

$$J_{\mathrm{oracle}} \;=\; J(\pi^{*}_{\mathrm{oracle}} \mid s^{\mathrm{priv}}), \qquad \mathrm{Gap}_{\mathrm{oracle}} \;=\; J_{\mathrm{oracle}} - J_{\mathrm{full}}.$$

**v4 的四象限解释表其实略强了**——"CAG 高 + oracle 略高 → policy 真在读 contract"这句话数学上不支持。$\mathrm{CAG} > 0$ 只说明**去掉这个结构以后 task utility 下降**、不说明 policy 是通过正确 semantic mechanism 使用它。真实情况常常是训练数据里 `age ↔ task difficulty` 高度相关、policy 学到 $a = f(\alpha)$ 的 shortcut 而不是 $a = f(\text{actual sensor staleness semantics})$。这一版把表格降级成 **"每个观察能支持什么"**：

| 观察 | 能说明 |
|---|---|
| CAG 高 | contract structure 对当前任务有 utility（可能来自 semantic use、也可能来自 shortcut） |
| CAG ≈ 0 | 当前任务下该 contract structure 可能不必要 |
| CAG 高 + §5.1 Type I/II 通过 | **更有证据**表明 semantic use |
| CAG 高 + intervention fail | **很可能是 shortcut**——回到 §4.3 / §5.2 检查 training distribution |
| CAG ≈ 0 + oracle gap 高 | contract 里有信息、policy 没充分利用——回到 §0.2.2 的三个 loss 定位是哪一段掉了 |

也就是：

$$\boxed{\;\mathrm{CAG} \;\neq\; \text{contract understanding}.\;}$$

$$\boxed{\;\mathrm{CAG} \;=\; \text{task-conditional utility sensitivity}.\;}$$

**CAG 是 decision 层的聚合、必须与 $E_{\mathrm{semantic}}$（invariance / equivariance）+ $E_{\mathrm{representation}}$（conditional probe + HPC/HSS）+ §5.1 Type I/II controlled response 联合看**、才能构成 semantic use 的证据。这一版把 CAG 从"总指标"降回"decision 层聚合"、四种 compliance evidence 才完整。

### 6.6 Safety evidence $E_{\mathrm{safety}}$：Constraint certification intervention（v5 三态）

在部署 / 半仿真环境里主动把 §5.3 的 $\mathrm{certification}_j$ 打到 **unsafe** 或 **unknown**（例如注入 calibration drift、把 observability 关掉、把 $\Lambda(E^-; H, \mathcal O)$ 拉高）、看 safety filter 是否**在正确的时刻进入正确的 guardrail**、以及 guardrail 触发是否可归因到 certification 的哪一项。**关键判定不再是"filter 是否 stopped policy"、而是"filter 是否收紧了 constraint"**（§5.3 已经写死：invalid evidence alone cannot justify relaxing constraint）。测的三种正确反应是 fallback、conservative tightening、stop；测的**错误反应**是"在 evidence 不充分的条件下 relax"。观测到"unknown 或 invalid 单靠自身导致 relaxing"、$E_{\mathrm{safety}}$ 直接 fail。反过来、如果另一路独立 valid evidence（例如 lidar）已经把 certification 推到 **safe**、filter 允许 relax 是**正确反应**、v4 会把这种情形误判、v5 因为引入了三态所以不会。这一类是 §5.3 接口的直接对应、也是整篇 interface 主张真正**能不能落地**的测试。

四类合起来构成一个**多证据 compliance argument**：**semantic 测"响应规则对不对"、representation 测"信息还在不在（且独立于 raw observation）"、decision 测"用了没 / 有没有 shortcut"、safety 测"unknown / invalid 时收紧没"**。它们都**不能替代**任何端到端 success rate——它们衡量的是 policy 侧对 contract 的**读取度**、不是**表现力**。这一点与 §0.3 Claim 3 完全对齐：**contract compliance must be tested by controlled intervention、并且必须由四类证据合流支持**。

### 6.7 全文理论骨架表（v5 修订：三 loss + 一 obligation）

把 §0 到 §6 收在一张表上、reviewer 最希望看到的就是这个：

| Layer | Object | Failure | Evidence |
|---|---|---|---|
| Contract | $\mathcal C$ | schema ambiguity / version mismatch | schema audit + compatibility check |
| Declaration | $q_\pi$（由 $\mathcal C_\pi$ + $Q_{\mathcal C_\pi}$ 诱导） | undeclared semantic collapse | quotient audit（能贴出 $Q_{\mathcal C_\pi}$ 清单吗？） |
| Projection | $\Pi_\pi = e_\pi\circ q_\pi$ | residual contract information | conditional probe $I(\text{field};z_\pi\mid o)$ |
| Decision | $\pi_\theta$ | wrong use / shortcut / collapse rate | Type I equivariance + Type II order-constrained + Type III ablation + $L_{\mathrm{decision}}$ |
| Safety | $g_{\mathrm{safety}}$ | unsafe interpretation of invalid / unknown evidence | constraint certification intervention（是否**收紧**、而不是**放松**） |

**三个 semantic losses + 一个 safety obligation**：

$$\boxed{\begin{aligned}
L_{\mathrm{declared}} &: \;\textstyle\sum_{q\in Q_{\mathcal C}^{\mathrm{req}}} w_q\,\mathbf 1[q\notin Q_{\mathcal C_\pi}];\\[1mm]
L_{\mathrm{projection}} &= I(Y_\pi;\hat S\mid Z_\pi, O, L);\\[1mm]
L_{\mathrm{decision}} &= \mathbb E_{\mathcal R_{\mathcal D}}\!\big[\mathbf 1[D_{\mathcal A}(\pi_\theta(\cdot\mid\hat S),\pi_\theta(\cdot\mid\hat S'))<\epsilon]\big];\\[1mm]
\mathcal O_{\mathrm{safety}} &: \;\text{safe → relax allowed; unknown / invalid → tighten or stop.}
\end{aligned}}$$

三个 loss 分别落在 $q_\pi / \Pi_\pi / \pi_\theta$ 上、safety obligation 落在 $g_{\mathrm{safety}}$ 上、**四层各自独立审计**、合起来形成 compliance argument。这就是本文从"给 VLA 加 metadata"走到 "contract-aware policy design" 的具体形状。

## 7. 最小可执行接口草图

把 §4 三族 primitives 与 §6 四种 compliance evidence 合起来写成一个 Python 类骨架。**不是要给出一个具体 policy、是要给一个可读的接口约定**。

```python
class StructuredStateView:
    def __init__(self, contract: StructuredState,
                 consumer_contract: ConsumerContract,     # C_pi + supported_version
                 query_family: QueryFamily,               # Q_{C_pi}; induces ~_pi
                 required_queries: QueryFamily):          # Q_C^req, with weights w_q
        # schema compatibility gate: version mismatch must fail closed or run adapter
        self.compatibility = check_schema_compatibility(
            contract.schema_version,
            consumer_contract.supported_version,
            adapter=consumer_contract.adapter,   # may be None -> strict fail
        )
        self.contract = contract
        self.C_pi = consumer_contract
        self.Q = query_family
        self.Q_req = required_queries

    def declared_coverage_loss(self) -> float:
        # L_declared = sum_{q in Q_req} w_q * 1[q not in Q_C_pi]
        return sum(self.Q_req[q].weight for q in self.Q_req if q not in self.Q)

    def project(
        self,
        schema: PolicySchema,
        # --- mode_select (mass-preserving, not calibration) ---------
        mode: Literal["map", "posterior_sample", "topk"] = "topk",
        topk_residual: Literal["renormalize", "keep_residual"] = "renormalize",
        # --- staleness / uncertainty knobs --------------------------
        staleness: Literal["ignore", "parallel_field", "condition"] = "parallel_field",
        uncertainty: Literal["none", "conservative_inflation", "propagate"] = "conservative_inflation",
        # --- source-structure knobs ---------------------------------
        provenance: Literal["ignore", "harden"] = "harden",
        dependency: Literal["ignore", "logit_bias_learned",
                            "covariance_fusion", "hierarchical_mixture"] = "covariance_fusion",
        negative_evidence: Literal["ignore", "condition", "belief_update"] = "condition",
    ) -> PolicyInput:
        """
        Project StructuredState (upstream, 9/14) to PolicyInput (downstream, this piece).
        The (C_pi, Q_C_pi) pair literally defines ~_pi; q_pi is the induced quotient map.
        e_pi is the encoder of the specific backbone and must not silently drop anything
        q_pi declared preserved; pi_theta may further collapse distinctions at the
        action level (see §0.2.2: three semantic losses + one safety obligation).
        """
        slots = {}
        for field_name in schema.fields:
            h = self.contract[field_name]   # hypothesis set: [(mu_i, Sigma_i, w_i)]

            # --- mode_select: mass-preserving top-k (NOT calibration-aware) ---
            if mode == "map":
                mu, Sigma, w_payload, residual = h.most_likely().mu, h.most_likely().Sigma, None, None
            elif mode == "posterior_sample":
                mu, Sigma, w_payload, residual = h.sample().mu, h.sample().Sigma, None, None
            else:  # "topk"
                top = h.top_k(k=schema.k_per_field[field_name])
                residual = 1.0 - top.total_weight()
                if topk_residual == "renormalize":
                    w_payload = top.renormalize()          # sum w̃_i = 1 within top-k
                else:  # "keep_residual"
                    w_payload = top.weights                 # raw weights kept
                residual_declared = residual                # explicitly recorded, not silently dropped
                # NOTE: `residual_declared` is AGGREGATE residual mass, not residual sufficient
                # statistics. Strict posterior update over the residual component still requires
                # separate mu/Sigma/likelihood specifications (see §4.1 v5 caveat).
                # Calibration (ECE_top-k / NLL / Brier) is measured in §6.2, NOT here.

            # --- age_gate: three groups, never multiplicative on mu ---
            # (i) payload: mu, Sigma
            # (ii) metadata quintuple: (α, ℓ, h, v, ι)  — α = age, ι = availability
            # (iii) derived trust q = τ(α, ℓ, h, v, ι, ...)
            age          = h.age                       # α_c
            validity     = h.validity_ok               # v_c
            availability = h.available                 # ι_c  (payload exists?)
            health       = h.sensor_health             # h_c
            latency      = h.causal_status             # ℓ_c
            trust = clamp(
                tau_curve(age, latency, health, validity, availability,
                          schema.tau_config[field_name]),
                min=schema.trust_eps,                  # numerical guard: τ ≥ ε
            )                                          # q_c is meta, does NOT scale μ_c

            # uncertainty handling: propagation is the general form,
            # Σ/τ is only ONE conservative approximation.
            if uncertainty == "none":
                Sigma_eff = Sigma
            elif uncertainty == "propagate":
                Sigma_eff = propagate_uncertainty(
                    Sigma, dt=age, u=h.control_state, f=schema.dynamics_model
                )
            else:  # "conservative_inflation"
                Sigma_eff = inflate_uncertainty(Sigma, trust)   # e.g. Σ/clamp(τ, ε), guarded

            slots[field_name] = dict(
                mu=mu, Sigma=Sigma_eff,
                alpha=age, validity=validity, iota=availability,   # v5: α / ι names
                health=health, latency_status=latency,
                trust=trust, w_payload=w_payload,
                residual_declared=(residual if mode == "topk" else None),
            )

        # --- provenance / dependency / negative evidence, three reads ---
        prov_mask = self.contract.contributing_mask if provenance == "harden" else None
        if dependency == "logit_bias_learned":
            # b(R_ij) is learned and CAN be positive or negative.
            # Convenient implementation candidate on transformer backbones;
            # NOT the canonical default of dependency_aware_fusion (§4.3.2 v5).
            dep_payload = FieldRelationPayload(self.contract.correlated_with)
        elif dependency == "covariance_fusion":
            dep_payload = self.contract.correlated_with       # fed into Kalman / factor graph
        elif dependency == "hierarchical_mixture":
            dep_payload = self.contract.correlated_with       # latent grouping
        else:
            dep_payload = None
        lam = self.contract.conditional_llr if negative_evidence == "condition" else None
        # `lam` is Λ(E; H, O); negative evidence is just the special case E = E^-.

        return PolicyInput(slots=slots,
                           contributing_mask=prov_mask,
                           dependency_payload=dep_payload,
                           conditional_llr=lam,
                           schema=schema,
                           C_pi=self.C_pi, Q=self.Q,
                           compatibility=self.compatibility)


class SafetyFilterHead(nn.Module):
    """v5: safety is not a fourth information loss; it is a distinct obligation slot.
    Reads three-state certification, NOT a binary validity bit.
    """
    def forward(self, a_proposed, contract) -> Action:
        for j, constraint in enumerate(contract.constraints):
            cert = constraint.certification      # ∈ {safe, unsafe, unknown}
            if cert == "safe":
                continue                         # relaxation allowed by positive evidence
            if cert == "unsafe":
                return tighten_or_stop(a_proposed, constraint)
            if cert == "unknown":
                # invalid evidence ALONE cannot justify relaxing the constraint.
                return conservative_fallback(a_proposed, constraint)


class ContractAwarePolicy(nn.Module):
    def __init__(self, backbone, schema: PolicySchema, cfg: ContractReadConfig,
                 consumer_contract: ConsumerContract, query_family: QueryFamily,
                 required_queries: QueryFamily):
        self.backbone = backbone
        self.schema = schema
        self.cfg = cfg                 # cfg is a *materialization* of q_pi,
        self.C_pi = consumer_contract  # but the actual declared quotient lives in (C_pi, Q)
        self.Q = query_family
        self.Q_req = required_queries
        # three semantic loss sites + one safety obligation site:
        #   L_declared     — on (C_pi, Q, Q_req): coverage of required queries
        #   L_projection   — on Π_π output Z_π: I(Y_π; Ŝ | Z_π, O, L)
        #   L_decision     — on π_θ: E_{R_D}[ 1[D_A(π(.|S), π(.|S')) < ε] ]
        #   O_safety       — on g_safety: safe / unsafe / unknown response correctness

    def forward(self, state: StructuredState, obs, lang) -> ActionDistribution:
        x = StructuredStateView(state, self.C_pi, self.Q, self.Q_req).project(
            self.schema, **self.cfg.as_kwargs())
        return self.backbone(x, obs, lang)
```

三条 caveat 明确写死：

- **(i)** 这不是唯一读法、**`cfg` 是本文 §0.2 里的 $q_\pi$ 的一个 materialization、但真正的 declared quotient 是 $(\mathcal C_\pi, Q_{\mathcal C_\pi})$ 这一对**——同一个 contract、VLA 与 Diffusion Policy 的 `cfg` 就该不一样、engineered state head 与 visual latent head 的最优 `uncertainty` 也不同。接口文档里必须**贴出一张 $Q_{\mathcal C_\pi}$ 清单**、否则 $q_\pi$ 又退化成 encoder 里的隐式行为。**schema 兼容性也必须 fail-closed 或走显式 adapter**（§0.2.3）、不然 $\mathcal C$ 升到 v2 会静默丢字段、正落入本文批评的 undeclared semantic loss。
- **(ii)** 这个接口**只解决输入端**；§5.1 的三类 intervention 约束、§5.2 的 observed-metadata-only aug、§5.3 的三态 certification 直连——如果一处不改、$(\mathcal C_\pi, Q_{\mathcal C_\pi})$ 写得再漂亮也会被 $\pi_\theta$ 训练动力绕过去（$L_{\mathrm{decision}}$ 直接爆）。**三个 loss 加一个 safety obligation 缺一个都守不住接口**。
- **(iii)** `dependency="logit_bias_learned"` 在 transformer 上是一种**便捷实现候选、不是 canonical default**——因为 field-graph → token-graph 的编译问题（$R_{\text{field}} \to R_{\text{token}}$）本身还没解决（§4.3.2 v5 caveat）。MLP head 走 `covariance_fusion`、Kalman / factor-graph 融合走 `covariance_fusion`、grouped latent 走 `hierarchical_mixture`——**真正 recommended 的是 dependency_aware_fusion 这条 primitive、不是它的某种具体 realization**。

## 8. 三条收束 claim 与一个升级的 thesis（与 §0.3 对齐）

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. 完整 pipeline $\mathcal C\to(\mathcal C_\pi,Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}$ 是**一个可审计的 semantic interface**、三层 loss 分别落在 $q_\pi$ / $\Pi_\pi$ / $\pi_\theta$ 上、第四个槽位 $g_{\mathrm{safety}}$ 不是 loss、是 obligation。三个 loss 各自可能被无声破坏、对应 §0.2.2 的三种定义：$L_{\mathrm{declared}}$（$\sum w_q \mathbf 1[q\notin Q_{\mathcal C_\pi}]$、required queries 中未被覆盖的部分）、$L_{\mathrm{projection}} = I(Y_\pi;\hat S\mid Z_\pi, O, L)$（projection 之后的 conditional residual）、$L_{\mathrm{decision}} = \mathbb E_{\mathcal R_{\mathcal D}}[\mathbf 1[D_{\mathcal A}(\cdot\mid\hat S,\cdot\mid\hat S') < \epsilon]]$（action-relevant pair 的 collapse rate）。$\Pi_\pi$ injective **不蕴含** $L_{\mathrm{decision}} = 0$——这是本版最重要的补强。

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state head、latent visuomotor policy、autoregressive VLA、diffusion、flow policy 用的 conditioning 与 action generator 都不一样——但它们作为 contract consumer 都必须回答**同一组四个问题**（本文真正的 thesis、下面会 boxed）。攻击的对象是 interface contract、不是模型架构；π0 是 VLA + flow matching、Diffusion Policy 是 visual-latent + diffusion、ACT 是 visual-latent + generative sequence decoder——**用两个正交维度切、比用"三个家族"切更贴近事实、也更不容易被"某族天生好"的直觉误导**。

> **Claim 3 · Contract compliance should be tested by four types of evidence, not a single score.** 端到端 success 衡量 policy 好不好用、不衡量它有没有把 contract 语义读对。contract compliance 需要 §6 的**四种 compliance evidence**合流——$E_{\mathrm{semantic}}$（invariance / equivariance）+ $E_{\mathrm{representation}}$（**conditional probe + $T^{\mathrm{contract}}$ 版 HPC/HSS + calibration diagnostic**）+ $E_{\mathrm{decision}}$（**$\mathrm{CAG}^{\mathrm{fixed}}$ + oracle baseline + $\Delta J_{\mathrm{where}} / \Delta J_{\mathrm{dep}} / \Delta J_{\mathrm{neg}}$ 三条独立 ablation、且 CAG ≠ contract understanding**）+ $E_{\mathrm{safety}}$（**三态 constraint certification intervention、测的是 filter 有没有'收紧'、而不是有没有停 policy**）——并且要区分 §5.1 的三类响应档位（Type I exact equivariance / Type II order-constrained response / Type III unconstrained）；把 Type III 错当 Type II 强加响应、是上一版的一个具体错误、这一版把它留给 ablation 去测。

一句收束：**"融合"这个词、以后尽量不用**——它在时间轴上问的是"什么时候合并"、在语义轴上问的是"合并成什么"；9/14 与本文合起来把第二个问题拆成了**上游交付什么 + policy 作为 consumer 声明承担什么 + $q_\pi$ 允许丢什么 + $e_\pi$ 保住什么 + $\pi_\theta$ 用什么 + $g_{\mathrm{safety}}$ 承担什么义务**六段。剩下第一个问题（时机）已经被 §1 的两维对照回答得差不多——**时机是接口的结果、不是接口的决策变量**。

**最后一个可以往前走半步的命题**——也是这一版从 reviewer 那里学到、把整篇文章真正立起来的一句话：

$$\boxed{\;\textbf{Structured estimator output} \;\neq\; \textbf{structured policy input}.\;}$$

再进一步：

$$\boxed{\;\textbf{A policy is a contract consumer, not merely a function approximator.}\;}$$

而一个 contract consumer 至少要回答四个问题：

$$\boxed{\begin{array}{ll}
\text{1.} & \textbf{What may I discard?} \quad (q_\pi, \mathcal C_\pi, Q_{\mathcal C_\pi}, L_{\mathrm{declared}})\\[2mm]
\text{2.} & \textbf{What did I actually retain?} \quad (\Pi_\pi = e_\pi\circ q_\pi,\; L_{\mathrm{projection}})\\[2mm]
\text{3.} & \textbf{How should decisions respond to contract interventions?} \quad (\pi_\theta,\; L_{\mathrm{decision}})\\[2mm]
\text{4.} & \textbf{What happens when the evidence becomes invalid or unknown?} \quad (g_{\mathrm{safety}},\; \mathcal O_{\mathrm{safety}})
\end{array}}$$

这四问一旦立住、**VLA / Diffusion / Flow / ACT / SAC / PPO 都只是实现坐标、不再是理论分类**。

而 §6 提出的 intervention battery、也不再只是"下一篇可以做的 benchmark"、会自然变成本文理论框架的**实验性闭环**：

$$\boxed{\;\mathcal C\;\longrightarrow\;(\mathcal C_\pi, Q_{\mathcal C_\pi})\;\longrightarrow\;q_\pi\;\longrightarrow\;e_\pi\;\longrightarrow\;\pi_\theta\;\longrightarrow\;g_{\mathrm{safety}}\;\longrightarrow\;\mathcal B_{\mathcal C},\quad \mathcal B_{\mathcal C} = \{T_{\mathrm{frame}},\,T_{\mathrm{hyp}},\,T_{\alpha},\,T_{\mathrm{validity}},\,T_{\mathrm{prov}},\,T_{\mathrm{neg}}\}.\;}$$

下一篇直接立一个 **Contract-Preserving Policy Benchmark**——让 SAC / PPO / Diffusion Policy / ACT / OpenVLA / π0 全过同一套 $\mathcal B_{\mathcal C}$ intervention suite、每一个 loss 与 obligation 各自有对应的可测量。那时候这一系列文章就从"我认为 policy 应该读 contract"升格成"**给定同一份 Structured State Contract、如何系统地测不同 policy 是否 contract-compliant**"。这一句立住、这一系列就都值了。

## Sources

以下 arXiv ID 已联网核过；journal-only 引用不贴 arXiv。按支撑的 section 分组。

### A · VLA 家族（支撑 §1 grid、§3 Failure 1–2、§5.1 类别 B）

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)（**paper fact**：把 robot action 明确表达成 text token 与 VLM 联合 fine-tune · §3.2 Failure 2 的 tokenizer 侧典型形态、"contract flattening" 是本文分析、不是原论文的 limitation）
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)（**paper fact**：7B VLA、大规模机器人 demonstration 训练、强调 fine-tune 与 generalization；**本文分析**：其配置含多相机 / depth / proprioceptive state encoding、但"支持输入" ≠ "读到 contract 的哪一站"）
- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)（**paper fact**：预训练 VLM + proprio token + noisy action chunk + flow matching；**本文分析**：Under the Structured State Contract defined here, π0's conditioning interface does not expose an explicit slot for hypothesis / provenance / age / negative evidence——"Continuous actions do not imply structured state semantics" 是本文的分析、不是原论文的 self-limitation）
- Octo Model Team, *Octo: An Open-Source Generalist Robot Policy*, RSS 2024 · [arXiv:2405.12213](https://arxiv.org/abs/2405.12213)（transformer-based readout · §4.3.2 dependency_aware_fusion 中 logit_bias_learned 路径的一个参照）

### B · Diffusion / Generative-Sequence / Flow-Matching Policy（支撑 §1 grid、§5.1）

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 · [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)（**paper fact**：RGB stack + proprio concat + conditional denoising diffusion、强调 action-distribution multimodality；**本文分析**：action-side multimodality ≠ state-side hypothesis preservation——这是本文 schema 下的推论、不是原论文承认的 limitation）
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ACT / ALOHA）, RSS 2023 · [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)（**paper fact**：CVAE + transformer encoder-decoder、核心是 **action chunking over sequences**——本文把 ACT 归入 "generative sequence decoder"、与 diffusion / flow matching 并列、不属于 diffusion family）
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 · [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)（**paper fact**：establish the flow-matching objective as vector-field regression for generative modeling / CNF——**flow matching 本身不是 robot action chunking 论文**；"continuous robot action chunks" 是 π0 这类工作的具体应用、本文把 citation chain 拆成 "Lipman establishes objective / π0 applies it to action chunks"）

### C · 不确定性、校准与 belief-space 参照（支撑 §4.1、§5.1、§6.2 calibration diagnostic、§6.5 CAG）

- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)（现代网络过度自信、temperature scaling 起点 · §6.2 calibration diagnostic 的评估依据、不再挂在 primitive 上）
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels*（PlaNet / RSSM）, ICML 2019 · [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)（deterministic + stochastic latent · §4.2 predictive uncertainty propagation 的一个参照）
- Hafner et al., *Mastering Diverse Control Tasks through World Models*（DreamerV3）, Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)（离散 + 连续混合 latent、KL balancing · §4.1 posterior readout 与 §6.2 counterfactual-intervention HPC 的一条相邻路线）

### D · SAC / PPO 与 engineered-state head 的 baseline（支撑 §1 grid 首行）

- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, ICML 2018 · [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)（Gaussian NLL / max-entropy policy loss · §5.1 $\mathcal{L}_{\mathrm{action}}$ 的 engineered-state head 形态）

### E · 承接前文（本文与 9/13、9/14、Sim-to-Real P1/P3 的接口）

- 本博客《拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口》· `/zh/articles/2026-09-14-multimodal-fusion-interface/`（Structured State Contract 定义、Interface Property Benchmark、degradation chain · 本文 §0.2 $q_\pi$ 与 §3–§6 直接建立在其上）
- 本博客《只会看、不会摸：机器人为什么缺一双"手感"的手》· `/zh/articles/2026-09-13-tactile-force-sensing/`（力 / 触觉的四种控制范式、Closed-loop value · §1 grid 里的 action-head 谱系的历史来源）
- 本博客《Sim-to-Real 方法论（三）》· `/zh/articles/2026-09-12-sim-to-real-evaluation-protocol/`（三级证据层、decision utility 三维、allocation protocol · §5.4、§6 直接沿用其 evaluation spine）
- 本博客《Sim-to-Real 方法论（一）》· `/zh/articles/2026-09-10-sim-to-real-methodology/`（allocation state $s_t = (b_t, \pi_t, q_t, h_t)$、$\Delta_{\mathrm{queue}}$ vs $\Delta_{\mathrm{processing}}$ · §2、§4.2 定义直接接上）

---

> **相关阅读**
>
> - [拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口](/zh/articles/2026-09-14-multimodal-fusion-interface/)——本文的前作、把上游交付物立成 Structured State Contract
> - [只会看、不会摸：机器人为什么缺一双"手感"的手](/zh/articles/2026-09-13-tactile-force-sensing/)——§1 grid 里 action-head 谱系的历史脉络、四种力控范式
> - [具身智能 Sim-to-Real 方法论（三）](/zh/articles/2026-09-12-sim-to-real-evaluation-protocol/)——§5.4、§6 沿用其三级证据与 utility 三维
> - [VLA 与世界模型：两条路线的分岔与合流](/zh/articles/2026-09-07-vla-world-models/)——本文 §1 conditioning 维度的宏观背景、"世界模型不天然属于 sim-to-real" 的另一面
> - [VLA π 家族速览](/zh/articles/2026-09-05-vla-pi-family/)——π0、π0.5 与 flow-matching action head 的一个具体切面、§1 "continuous action ≠ structured state" 的现场证据
> - [什么是 VLA 模型？一篇讲清楚](/zh/articles/2026-09-03-vla-deep-dive/)——本文假设你已经读过
