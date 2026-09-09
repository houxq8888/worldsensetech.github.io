---
title: '政策侧接口（上）：Contract 立起来之后、VLA / Diffusion Policy / π0 到底吃什么？'
slug: "2026-09-15-policy-side-interface"
date: 2026-09-15
draft: false
categories: ["具身智能", "策略学习"]
tags: ["具身智能", "策略学习", "VLA", "Diffusion Policy", "π0", "RT-2", "OpenVLA", "Action Tokenization", "Structured State Contract", "Consumer Contract", "Consumer Contract Triple", "Declared Quotient", "Query Family", "Query Subsumption", "Schema Compatibility", "Semantic Preservation", "Decision-Relevant Preservation", "Decision Sufficiency", "Conditional Mutual Information", "Residual Contract Information", "Declared Coverage Loss", "Projection Residual Loss", "Decision Collapse Rate", "Safety Obligation", "Contract-Read Primitives", "Measurable Decoder", "Separately Auditable Failure Sites", "Contract Consumer"]
description: '本文是"政策侧接口"长文的**上半 · framework 篇**、下半（evaluation + Python 骨架 + 训练时连锁）在 9/16 [政策侧评估（下）](/zh/articles/2026-09-16-policy-side-evaluation/)。《多模态融合接口》那一篇把上游交付物立成了 Structured State Contract——本文问它的对偶：如果 estimator 真的按 contract 交付、policy 侧到底能不能吃到。核心 boxed 不等式是 **Structured estimator output ≠ structured policy input**、完整 pipeline $\mathcal C\to(\mathcal C_\pi,Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}\to\mathcal B_{\mathcal C}$。v6 把 Consumer Contract 拆成三段 $\mathcal C_\pi=(Q_\pi,\mathcal O_\pi,V_\pi)$——Queries / Obligations / Versions、真正长成 software-interface 的形状。§0.2.1 补两个悬空定义：**decision-relevant observable** $Y_{\mathcal C}^{\pi}=\{q(\hat S):q\in Q_{\mathcal C}^{\mathrm{req}},q\text{ influences consumer decision}\}$（由 Consumer Contract 锁定、不是 benchmark 里凭空的 latent）与 **representation equivalence** $Z_\pi(\hat S)\sim_Z Z_\pi(\hat S^{\prime})$ ⟺ 不存在 measurable decoder $h_\pi(Z_\pi,O,L)$ 在 $Y_{\mathcal C}^{\pi}$ 上把两者区分开——把 A / A′ 里的 $\equiv$ 从悬空符号变成 conditional-MI 可接线的对象。§0.2.2 里 $L_{\mathrm{declared}}$ 从字面 set membership 升级为 **query subsumption coverage** $L_{\mathrm{declared}}=\sum w_q\mathbf 1[\nexists q^{\prime}\in Q_{\mathcal C_\pi}:q^{\prime}\succeq q]$、"允许丢什么"从 checklist 变成 semantic capability lattice；$L_{\mathrm{projection}}=I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$ 明确用 $Y_{\mathcal C}^{\pi}$、$L_{\mathrm{decision}}$ 用 $D_{\mathcal A}$（action-equivalence-aware distance）、第四槽 $\mathcal O_{\mathrm{safety}}$ 是 obligation 不是 loss。三 loss 是 **three separately auditable failure sites**、**不是三个 statistically independent losses**、$q_\pi\to\Pi_\pi\to\pi_\theta$ 顺序耦合。§1 用**两个正交维度**（conditioning representation / semantic interface × action head）取代"三个家族"式分类、§1.2 给 grid。§2 把"State"在不同 policy 家族里的语义差异钉住。§3 列出**四种 interface-mismatch 失败模式**——multi-hypothesis 被无声折叠 / semantic correctness 缺接口层保证 / concat 打破 temporal alignment / safety 对 validity vs staleness 失明。§4 给**三族 contract-read primitives**——`mode_select` / `age_gate` / `provenance_harden + dependency_aware_fusion + negative_evidence_read`。§5 承上启下：本文上半立到 Claim 1（policy 是 contract consumer、不是仅仅 function approximator）与 Claim 2（architecture-agnostic 四问）、Claim 3（compliance 用 intervention 测、不用端到端 score 推）连同四类 compliance evidence、五层 evaluation hierarchy、Python skeleton v6 修复清单——全部交给下篇 9/16 展开。VLA / Diffusion / Flow / ACT / SAC / PPO 由此降级为实现坐标、不是理论分类。'
toc: true
related_articles:
  - 2026-09-16-policy-side-evaluation
  - 2026-09-14-multimodal-fusion-interface
  - 2026-09-13-tactile-force-sensing
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

- **$\mathcal C$ · Upstream contract**——estimator 侧交付的**完整**语义 schema（9/14 下篇 §2–下篇 §4 定义的那份）。
- **$\mathcal C_\pi$ · Consumer contract**——**policy 作为 consumer 明确声明它承担 $\mathcal C$ 的哪一部分**。它可以只是 $\mathcal C$ 的一个子集（"这个 policy 不消费 provenance 字段"）、也可以是 $\mathcal C$ 上的一个 coarse-graining（"hypothesis 结构折成 point estimate、但 age 保留成独立字段"）。**$\mathcal C_\pi$ 是接口规格里的一段、不是 encoder 权重里的隐式偏好**。
- **${\sim_\pi}$ · Induced equivalence**——由 $\mathcal C_\pi$ 上的一组**声明过的 contract queries / decision-relevant predicates** $Q_{\mathcal C_\pi}$ 所诱导：

$$\hat S \sim_\pi \hat S' \quad\Longleftrightarrow\quad Q_{\mathcal C_\pi}(\hat S) \;=\; Q_{\mathcal C_\pi}(\hat S').$$

$Q_{\mathcal C_\pi}$ 是本文真正的**接口对象**——它把"允许丢什么"从模糊的语义承诺变成一组可以逐条 review 的 queries。典型例子："$\hat S$ 里第 $c$ 通道的 age 是多少"、"这个 track 的 posterior-weighted top-3 hypothesis 是哪三个"、"contact set 是否包含 pad 面"。

- **$q_\pi$ · Quotient map**——真正的 $q_\pi : \hat S \mapsto [\hat S]_{\sim_\pi}$、把 $\hat S$ 折到 ${\sim_\pi}$ 定义的商空间里。

有了这条链、本文的核心主张终于可以写成一句 reviewer 无法追问"quotient 谁定义"的话：

> **Policy does not need to preserve the entire upstream contract $\mathcal C$. It must explicitly declare a consumer contract $\mathcal C_\pi$ together with a query family $Q_{\mathcal C_\pi}$ — and the resulting quotient $q_\pi$ is exactly the semantic loss the policy is allowed to take.**

对比上一版那句"要么保留整个商结构、要么声明 sufficient quotient"——这一版给出了**"quotient 从哪来"**的完整答案：它来自一份**写下来的** $\mathcal C_\pi$、加上一份**可枚举的** $Q_{\mathcal C_\pi}$。

**$\Pi_\pi$ 仍然等于 $e_\pi\circ q_\pi$、但 $q_\pi$ 不再是 primitive、它由 ${\sim_\pi}$ 唯一决定、而 ${\sim_\pi}$ 由 $\mathcal C_\pi$ 诱导**。工程上这意味着：**接口文档里必须能贴出一张 $Q_{\mathcal C_\pi}$ 清单**、否则 $q_\pi$ 就退化成 encoder 里的隐式行为——正是本文要攻击的对象。

**v6 补·Consumer Contract 的三段式分解**。上一版本 $\mathcal C_\pi$ 只写了"$\mathcal C$ 的一个子集 / coarse-graining"、reviewer 抓得对：**这只说了 policy 要读什么、没说 policy 对这些 semantics 承担什么义务、也没说 policy 认识哪个 schema version**。v6 把 $\mathcal C_\pi$ 明确拆成三段：

$$\boxed{\;\mathcal C_\pi \;=\; \big(Q_\pi,\;\mathcal O_\pi,\;V_\pi\big),\;}$$

其中

- $Q_\pi \equiv Q_{\mathcal C_\pi}$ · **Queries**——policy 声明"我要读哪些 contract fields / queries"（本节上面已经定义、也是 $q_\pi$ 的来源）。
- $\mathcal O_\pi$ · **Obligations**——policy 声明"读了这些 fields 之后我承诺怎样响应"。下篇 §1.1 的 Type I equivariance / Type II order-constrained response / Type III unconstrained、以及 下篇 §1.3 的三态 safety certification 义务、**都是 $\mathcal O_\pi$ 的具体条目**、而不是散落在文中的独立规则。§0.2.2 的三个 loss + 一个 obligation 也正是从 $\mathcal O_\pi$ 派生出来的可审计 failure modes。
- $V_\pi$ · **Versions**——policy 声明"我认识哪个 schema version"。§0.2.3 的 schema compatibility 规则就是 $V_\pi$ 的 operational form。

这一分解让 Consumer Contract 长得更像真正的 software interface、而不只是一个 ML abstraction：

```text
Consumer Contract  C_π
├── Q_π   Queries       What I read
├── O_π   Obligations   How I must respond
└── V_π   Versions      Which schema I understand
```

reviewer 后续追问"你说 policy 是 consumer、consumer 到底承诺了什么"、v6 有确切的答案：**它承诺 $Q_\pi$ 里的字段它都读、$\mathcal O_\pi$ 里的响应它都遵守、$V_\pi$ 里的 schema 它都兼容**。缺任何一段都不算一份完整的 consumer contract。

### 0.2.3 Schema version / compatibility（v5 新增 · 即 $V_\pi$ 段）

有了 $\mathcal C_\pi$ 与 $Q_{\mathcal C_\pi}$、必须再回答一个 software-interface 层面的问题：**Consumer contract 是针对哪个 schema version 的？** Estimator 升到 $\mathcal C$ 的 v2（新增 `observability` / `negative_evidence` / `sensor_health`）、如果 policy 的 $\mathcal C_\pi$ 还停在 v1、当前框架就会**默默丢掉新增字段**——这正是本文全文批评的 **undeclared semantic loss**。所以接口层再加一条小规则：

$$\boxed{\;\mathrm{schema\_version}(\mathcal C)\;\not\simeq\;\mathrm{supported\_version}(\mathcal C_\pi)\;\;\Longrightarrow\;\;\text{reject / explicit adapter required.}\;}$$

也就是：**schema mismatch must fail closed or pass through an explicitly declared adapter**——不兼容时要么**拒绝加载 policy**、要么必须走一份**写下来的 adapter**（把 v2 新增字段显式映射到 v1 已有 slot 或显式声明丢弃、并附一条 $\mathcal C_\pi^{v2} = \mathrm{adapt}(\mathcal C^{v2},\mathcal C_\pi^{v1})$ 规约）。这一条让 "contract" 这个词真正具有 software-interface 的味道、而不是只停留在 ML abstraction。

### 0.2.1 三档 preservation：full semantic / decision-relevant semantic / decision sufficiency

这一小节是本文的理论锚点、必须把**三个**容易混用的性质彻底分开。v6 对这一段做两处关键补定义（reviewer 抓得对）：第一、**$\equiv$ 在 A 与 A′ 里从未定义、是悬空符号**；第二、**$Y_\pi$ 到底是什么必须真正绑到 Consumer Contract、不能是一个凭空的 latent variable**。

**v6 补定义 · decision-relevant observable $Y_{\mathcal C}^{\pi}$**。上一版写 $Y_{\mathcal{C}}$ / $Y_\pi$、含义模糊。v6 把它锁定：

$$\boxed{\;Y_{\mathcal C}^{\pi} \;=\; \big\{q(\hat S) : q \in Q_{\mathcal C}^{\mathrm{req}},\;\text{$q$ influences consumer decision}\big\}.\;}$$

也就是：$Y_{\mathcal C}^{\pi}$ 是 required queries 里**真正影响 consumer decision 的那部分输出**——太宽（$Y_{\mathcal C}^{\pi}=\hat S$）会退化成"别丢任何 contract 信息"、太窄（$Y_{\mathcal C}^{\pi}=\text{current optimal action}$）会变成 decision sufficiency、不再是 contract semantics。$Y_{\mathcal C}^{\pi}$ 恰好是 Consumer Contract 思想应该吃掉的数学对象。

**v6 补定义 · representation equivalence $\sim_Z$**。A 与 A′ 里的 $\Pi_\pi(\hat S)\not\equiv\Pi_\pi(\hat S')$ 不能只是"数值不同"——否则任何 floating-point rounding 都算"preservation"。v6 定义：

$$Z_\pi(\hat S) \sim_Z Z_\pi(\hat S') \quad\Longleftrightarrow\quad \nexists\;\text{measurable decoder } h_\pi(Z_\pi,O,L)\;\text{s.t.}\; h_\pi \text{ distinguishes } \hat S \text{ from } \hat S' \text{ on } Y_{\mathcal C}^{\pi}.$$

用一句话讲：**如果下游 decision 在声明过的 consumer 下无法区分两份 representation、那它们就是 representation-equivalent**。这一条与 $L_{\mathcal C}^{\pi}=I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$ 天然接线——$L_{\mathcal C}^{\pi}=0$ **等价于** $Z_\pi(\hat S)\sim_Z Z_\pi(\hat S')$ 对所有 $Y_{\mathcal C}^{\pi}$-不等的 $(\hat S,\hat S')$ 都成立。

**Property A · Full Semantic Preservation**——投影在 contract-equivalent class 之间不做不可逆折叠：

$$\hat S \not\sim_\pi \hat S' \quad \Longrightarrow \quad Z_\pi(\hat S) \not\sim_Z Z_\pi(\hat S').$$

这要求 $q_\pi$ 在 $\mathcal C_\pi$ 定义的商上 injective、且 $e_\pi$ 不把不同 class 映到 $\sim_Z$-等价的 $Z_\pi$。**是一个很强的性质、但可能 over-preserving**。

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

$$\hat S \not\sim_{\pi,\mathcal D} \hat S' \quad\Longrightarrow\quad Z_\pi(\hat S) \not\sim_Z Z_\pi(\hat S').$$

因为 $\sim_{\pi,\mathcal D}$ 比 $\sim_\pi$ 更粗、**Property A 蕴含 Property A′、反之不成立**。这就是本文真正想要的接口要求。

**Property B · Decision Sufficiency (conditional MI form)**——用**条件互信息**衡量"给定 policy 已经拥有的 side information、projection 后 $Y_{\mathcal C}^{\pi}$ 里还剩多少 contract 信息"。上一版写成 $L_{\mathrm{dec}} = I(\hat S;Y_{\mathcal C}) - I(\Pi_\pi(\hat S);Y_{\mathcal C})$、reviewer 抓到了致命问题：**policy 的完整输入是 $\pi(a\mid \hat S, o, \ell, \text{language})$、raw image $o$ 里往往已经带了 age / provenance 的 proxy**。差分形式下 $L_{\mathrm{dec}} > 0$ 只说明"$\Pi_\pi(\hat S)$ 单独看对 $Y_{\mathcal C}$ 不是充分统计"、**并不说明 policy 真的缺信息**（信息可能已经从 $o$ 补回来了）。

v5 把 contract information loss 重定义为**条件 MI**、v6 把其中的 $Y_\pi$ 锁定为 $Y_{\mathcal C}^{\pi}$：

$$\boxed{\;L_{\mathcal C}^{\pi} \;=\; I\!\big(Y_{\mathcal C}^{\pi}\,;\,\hat S \,\big|\, Z_\pi,\, O,\, L\big),\qquad Z_\pi = \Pi_\pi(\hat S, O, L).\;}$$

（若 projection 只作用在 contract 上、可以把 $Z_\pi$ 简化为 $\Pi_\pi(\hat S)$。）这个定义干净得多：

- $L_{\mathcal C}^{\pi} = 0$ **当且仅当** $Y_{\mathcal C}^{\pi} \perp\!\!\!\perp \hat S \mid Z_\pi, O, L$——即 **$Z_\pi$ 对 $Y_{\mathcal C}^{\pi}$ 是 sufficient given policy 已经拥有的其它输入**。
- 直接回答了"raw image 里面已经有 object identity / provenance proxy"这个 reviewer objection：**正因为我们用的是 conditional sufficiency、不是 unconditional MI、raw observation 已经补回来的那部分信息不会被误算成 loss**。
- **v6 新增**：$Y_{\mathcal C}^{\pi}$ 的定义权在 Consumer Contract、不在 benchmark——这样"够不够"这个问题才有唯一答案、不会因 $Y$ 选得宽或窄而随意漂移。

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

**(1) $L_{\mathrm{declared}}$ 从 entropy 差分降级为 weighted undeclared-query coverage（v6 二次升级 · query subsumption）**。v4 的 $H(Q_{\mathcal C}(Y_{\mathcal C})) - H(Q_{\mathcal C_\pi}(Y_{\mathcal C}))$ 有三个问题：$Q_{\mathcal C}(Y_{\mathcal C})$ 里的 random variable 从未定义、entropy 差分不一定非负（query 数量 / 编码 / 基数都会改变 entropy）、而且 $q_\pi$ 的核心语义是"声明哪些 distinctions 允许被丢"、天然是一个 **coverage / violation set**、不是一个 scalar entropy。v5 直接把它写成集合 $\mathfrak D_\pi = Q_{\mathcal C}^{\mathrm{req}}\setminus Q_{\mathcal C_\pi}$——但 v6 reviewer 又抓一步：**"两条 query 相等"这件事本身就不是 set membership**。例：

- $q_1 = \text{"age"}$、$q_2 = \text{"whether age > 100ms"}$——$q_2$ 是 $q_1$ 的**函数**、$q_1$ 已经蕴含 $q_2$。
- $q_3 = \text{"hypothesis top-3"}$、$q_4 = \text{"MAP hypothesis"}$——$q_3 \succeq q_4$、但不等于。

所以 $q \notin Q_{\mathcal C_\pi}$ 不能只是字面集合成员。**v6 引入 query subsumption 序**：

$$q_1 \succeq q_2 \quad:\!\!\Longleftrightarrow\quad \text{$q_1$ 所保留的信息足以让 consumer 在任何 side information 下回答 $q_2$}$$

（formally：$\exists$ measurable $h$ 使 $h(q_1(\hat S)) = q_2(\hat S)$、或更弱的 conditional 版本 $H(q_2 \mid q_1) \le \varepsilon$。）于是 coverage 变成：

$$\boxed{\;L_{\mathrm{declared}} \;=\; \sum_{q\,\in\, Q_{\mathcal C}^{\mathrm{req}}} w_q\,\mathbf 1\!\Big[\nexists\, q' \in Q_{\mathcal C_\pi}:\;q' \succeq q\Big]\;}$$

这一步把 $L_{\mathrm{declared}}$ 从一个 checklist **升级成一个 semantic capability lattice**——$Q_{\mathcal C}^{\mathrm{req}}$ 与 $Q_{\mathcal C_\pi}$ 都在同一个 subsumption 偏序上、"允许丢什么"变成"consumer 声明的 capability 上界 $\{q' : q' \succeq q \text{ for some } q \in Q_{\mathcal C}^{\mathrm{req}}\}$ 是否覆盖 required 集合"。$w_q$ 是 task-side 权重：frame semantics 高、age 中高、provenance task-dependent、跟当前 controller 完全无关的 diagnostic field 低。最粗的 cardinality 版本 ($q' \succeq q \Leftrightarrow q' = q$) 就是 v5 的 $w_q \equiv 1$ 特例。这样"审 $q_\pi$"就变成**对着 $Q_{\mathcal C}^{\mathrm{req}}$ 沿 subsumption 逐条查上界覆盖**、不再依赖 entropy 定义、也不再被"字面 query 不等价"这种工程细节误伤。

**(2) $L_{\mathrm{rep}}$ 改名 $L_{\mathrm{projection}}$、明确它是 residual contract information 而不是 encoder loss**。v4 里 $Z_\pi = \Pi_\pi(\hat S, O, L)$ 已经是"整条 projection 的输出"、不只是 encoder $e_\pi$ 的输出、所以把它叫 representation loss 是把数学对象与 pipeline layer 强行 1:1 绑定。v5 直接改名、**v6 把其中的 $Y_\pi$ 锁定成 §0.2.1 已经定义的 $Y_{\mathcal C}^{\pi}$**：

$$\boxed{\;L_{\mathrm{projection}} \;=\; I\!\big(Y_{\mathcal C}^{\pi}\,;\,\hat S \,\big|\, Z_\pi,\, O,\, L\big),\qquad Y_{\mathcal C}^{\pi} = \{q(\hat S): q \in Q_{\mathcal C}^{\mathrm{req}},\;q \text{ influences consumer decision}\}.\;}$$

语义是 **"projection 之后的 residual contract information"**——可以 operationalize 为 representation-stage loss、但不宣称它就是 $e_\pi$ 那一段的损失。$Y_{\mathcal C}^{\pi}$ 由 Consumer Contract 定义、**不是 benchmark 里凭空的 latent**——这是本文 Consumer Contract 思想最应该吃掉的数学对象。

**(3) $L_{\mathrm{decision}}$ 从"supremum norm"降级为 action-relevant collapse rate**。v4 的 $\sup_{\hat S \not\sim_{\pi,\mathcal D} \hat S'} \|\pi_\theta(\hat S) - \pi_\theta(\hat S')\|_{\text{action-distribution}}^{\!\perp}$ 是**类型错误**——norm 是距离、不是"pair 集合大小"。v5 把它写成 collapse rate：

$$\mathcal R_{\mathcal D} \;=\; \big\{(\hat S, \hat S') : \hat S \not\sim_{\pi,\mathcal D} \hat S'\big\}$$

$$\boxed{\;L_{\mathrm{decision}} \;=\; \mathbb E_{(\hat S,\hat S')\sim\mathcal R_{\mathcal D}}\!\Big[\mathbf 1\!\big(D_{\mathcal A}(\pi_\theta(\cdot\mid \hat S),\,\pi_\theta(\cdot\mid \hat S')) < \epsilon\big)\Big]\;}$$

其中 $D_{\mathcal A}$ **不是普通的 distribution distance**——它测的是"两份 action distribution 是否**支持不同的 admissible / optimal action set**"。这一改把 reviewer 的哲学 objection 一起接了：如果 $\mathcal A^*(\hat S_1) \neq \mathcal A^*(\hat S_2)$、但两个 state 的公共交集里存在一个 action 都 admissible、policy 输出那个共同 action 是合法的、$D_{\mathcal A}$ 应该识别这种情况、不当作 collapse。软阈值版本可以写成 $\mathbb E_{\mathcal R_{\mathcal D}}[\exp(-D_{\mathcal A}(\cdot))]$。$D_{\mathcal A}$ 与 下篇 §2.2 HPC / HSS 的语义闭环：**HPC 测 coverage、HSS 测 separation、$L_{\mathrm{decision}}$ 测两者的失败率**。

**(4) $\mathcal O_{\mathrm{safety}}$ 是 obligation、不是 loss**。safety 那一层不做"信息损失"的度量、它做的是**"当 evidence 不足或 unknown 时、filter 有没有采取保守反应"**的义务判定（详见 下篇 §1.3 三态 certification 与 下篇 §2.6 safety evidence）。**它是第四类量、不能与三个 loss 混称**。

| Slot | 语义 | Failure | Object |
|---|---|---|---|
| $q_\pi$ | 声明允许丢什么 | **Declared coverage loss**：$Q_{\mathcal C_\pi}$ 沿 subsumption 没有覆盖 required query | $L_{\mathrm{declared}} = \sum w_q \mathbf 1[\nexists q' \in Q_{\mathcal C_\pi}: q' \succeq q]$ |
| $\Pi_\pi$ | projection 之后条件 residual | **Projection residual**：条件 MI > 0 | $L_{\mathrm{projection}} = I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$ |
| $\pi_\theta$ | decision 是否用被保住的 distinctions | **Decision collapse rate**：action-relevant pair 折叠的比例 | $L_{\mathrm{decision}} = \mathbb E_{\mathcal R_{\mathcal D}}[\mathbf 1[D_{\mathcal A} < \epsilon]]$ |
| $g_{\mathrm{safety}}$ | evidence 不足时的义务反应 | **Safety obligation violation**：unknown / invalid 时没收紧 | $\mathcal O_{\mathrm{safety}}$ |

$\Pi_\pi$ injective **不蕴含** $L_{\mathrm{decision}} = 0$——三个 loss 是**三个可分别审计的 failure sites**、**不是三个 statistical independent losses**。这一点 v6 reviewer 抓得很准：$L_{\mathrm{declared}}$、$L_{\mathrm{projection}}$、$L_{\mathrm{decision}}$ 在数学上是通过 $q_\pi \to \Pi_\pi \to \pi_\theta$ **顺序耦合**的——上游 declaration 变了、下游"允许的 quotient"也跟着变、三者不可能随机变量独立。**"independently" 全文改为 "separately auditable"**：它们各自定位到一个可打开审计的 pipeline 站点、而不是三者统计独立。下篇 §2 的四种 compliance evidence 与 下篇 §2.0 的 skeleton table 都会直接引用这三个 loss + 一个 obligation。

### 0.3 本文上半的两条 boxed claim（Claim 3 与下篇见 9/16）

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. 完整 pipeline $\mathcal C\to(\mathcal C_\pi,Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}$ 是**一个可审计的 semantic interface**、三个 loss 分别落在 $q_\pi$ / $\Pi_\pi$ / $\pi_\theta$ 上、第四个槽位是 safety obligation 而不是 loss。这就是本文定级 thesis 的雏形——**A policy is a contract consumer, not merely a function approximator**。

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state head、latent visuomotor policy、VLA / diffusion / flow policy 用的 conditioning 与 action generator 机制都不一样、但它们作为 contract consumer 都必须回答**同一组四个问题**——what may I discard ($q_\pi$)? what did I actually retain ($\Pi_\pi$, $L_{\mathrm{projection}}$)? how should decisions respond to contract interventions ($\pi_\theta$, $L_{\mathrm{decision}}$)? what happens when evidence becomes invalid or unknown ($g_{\mathrm{safety}}$, $\mathcal O_{\mathrm{safety}}$)? 攻击的对象是 interface contract、不是模型架构。



## 1. 两种维度、而不是三个家族：conditioning representation / semantic interface × action head

> **本文位置说明（v6 P1-18 加）**：§1–§3 是**接口理论的 implementation coordinates**、不是核心论证本身。§0 已经把 $\mathcal C \to (\mathcal C_\pi, Q_{\mathcal C_\pi}) \to q_\pi \to e_\pi \to \pi_\theta \to g_{\mathrm{safety}}$ 这条 pipeline、三档 preservation、三个 loss + 一个 obligation 讲清楚；§4 起进 primitives、下篇 §1 起进训练与部署、下篇 §2 起进 benchmark。§1–§3 只是给"现有 policy 到底落在 pipeline 哪一格"提供一个具体坐标、**读者若已经熟悉 VLA / Diffusion / flow / ACT / engineered-head 家族、可以直接跳到 §4**。这一节的存在不是为了 architecture ranking、是为了让 §0 定义的 pipeline 与实际系统对上号。

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

一句 caveat：**这不是"哪个组合最好"的排序**。本文关心的是**每一种组合、它的 $\Pi_\pi = e_\pi \circ q_\pi$ 是否有一个写清楚的 $q_\pi$**。答案在多数现有工作里是"没有"——**不是某个家族天生不行、是这个 $q_\pi$ 层从来没有被当成接口设计过**。

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

**"多模态融合"这个词在 policy 侧的常见误用**就是把某根枝条上的 cross-attention 当成"已经在做多模态状态估计"——它不是。真正的 state abstraction 要求 $\hat S_t$ 里的字段**跨传感器家族保持一致的语义、可被 controller / policy / world model / diagnostics 四种消费者共同读**——9/14 下篇 §2 已经把这个约定立起来了、本节要做的是**从 policy 一侧再问一遍：$q_\pi$ 是不是被显式声明过、还是被 $e_\pi$ 悄悄替代了**。

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

contract 明确区分 availability（这个通道今天有没有数据）、validity（这个数据是否有效、例如 calibration 是否过期）、age（有多旧）。policy 若只看数值、就把"stale 但 valid"与"missing 但 valid"混成一类、把"calibration drift 后无效"与"传感器掉线"当成同一种降级。9/14 下篇 §4.5 的 degradation chain 已经拆开了——**masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption**——policy 若在 input 端不接这个 chain、训练时的 augmentation 与推理时的 guardrail 都会挂错地方。

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

v4 把这个 primitive 叫 "calibration-aware top-$k$"、reviewer 抓得准：截断重归一化**只解决 probability mass conservation、不解决 calibration**。$\sum_i w_i = 1$ 不代表 posterior calibrated。v4 已经把它改叫 **mass-preserving top-$k$**、calibration 单独走 下篇 §2.2 的评估：

$$\tilde w_i \;=\; \frac{w_i}{\sum_{j \in \mathrm{top}\text{-}k} w_j}, \qquad w_{\mathrm{other}} \;=\; 1 - \sum_{i \in \mathrm{top}\text{-}k} w_i.$$

$\tilde w_i$ 是**在 top-$k$ 内部**重归一化的权重、$w_{\mathrm{other}}$ 是**剩下的残差**、两样都保留给下游。

**v5 补一句限定（reviewer 抓到 v4 表述过头）**：$w_{\mathrm{other}}$ 只是一个 **aggregate residual mass**——它没有告诉下游 residual hypothesis 的**位置、covariance、likelihood、component identity**。所以 mass-preserving top-$k$ 能保证的是 **probability mass accounting**（下游至少能区分 retained mass 与 discarded residual mass）、但**不能保证 Bayesian update correctness**——严格 posterior update 还需要为 residual component 单独定义 sufficient statistics。primitive 层的语义边界就写在这里、不再往前一步。

至于"这份 posterior 到底 calibrate 没 calibrate"、那是**评估属性**、不是 primitive 的一部分。本文的 calibration diagnostic 挂在 下篇 §2.2 representation-level 上、包含 **ECE$_{\text{top-}k}$ / NLL / Brier / calibration curve / coverage-credibility 五条**、任何一条都独立于 top-$k$ 截断本身。primitive 层与 evaluation 层完全拆开、reviewer 就不会再问 "What exactly makes your top-$k$ calibration-aware?"。

关键**不是"mean 不能用"**——mean 是一种完全正当的 readout、只要它是**显式声明**的折叠。真正的失败模式是"接口没有为 hypothesis 结构提供任何 readout slot、$e_\pi$ 只能靠 concat + MLP 隐式合并、结果把 mean 当成了默认"。这一区分很关键：**本文反对的是"无声明的默认折叠"、不是"折叠"本身**。

### 4.2 `age_gate`：observation payload + metadata quintuple + derived trust

**接口设计的常见 bug** 是把 staleness trust 直接乘进 measurement：$x_c^\pi = \tau_c(\alpha_c) \cdot \mu_c$。这**改变了 observation 的物理值**——10 N 的力、age 100 ms、被乘成 3 N 之后、policy 输入里"3 N"这个数字**看起来**就像"3 N 力"、而不是"10 N 力、但 trust 降低"。这直接违反了 contract 想保护的那个 distinction：**$(F = 3\,\mathrm{N},\, a = 0)$ 与 $(F = 10\,\mathrm{N},\, \alpha = 100\,\mathrm{ms})$ 是两个不同的语义事件**。

v4 已经把字段结构重组过、v5 再做一次 notation cleanup（reviewer 抓得好：全文里 $a$ 同时表示 action 与 age、下篇 §2.3 的 $R_\pi(a)$ 特别容易读错）。**v5 起、age 字段的符号统一为 $\alpha_c$、availability 字段的符号统一为 $\iota_c$**、$\alpha$ 与 $\iota$ 与 $a$（action）不再撞名。

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
| $\iota_c = 1$（有 payload） | 正常证据、policy 可以按 §4.1 消费 | 有数据、但**不能当 valid evidence 用**（例如 calibration 过期）——policy 应该走 下篇 §1.3 safety 侧 |
| $\iota_c = 0$（无 payload） | 语义上不可能（没数据、validity 位没意义） | 通道掉线 / masked、policy 读"no observation"这一 fact |

一句话：**availability $\iota_c$ 说的是"有没有 payload"、validity $v_c$ 说的是"这份 payload 算不算 valid evidence"——两者正交、不能塞进同一个位**。下篇 §1.2 的 degradation chain（masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption）本质上就是这五元 $m_c$ 之间的**不同 pattern**、这一版把 augmentation 与 slot 一一对上了。

**若确实需要在下游做 gating、gating 应该作用在 uncertainty 上、不作用在 measurement 上**。但**上一版把 $\tilde\Sigma_c = \Sigma_c / \tau_c(\alpha_c)$ 写成"stale ⇒ 有效 uncertainty 被放大"的一般原则、reviewer 也抓到了**——真实情况是、stale observation 的正确处理**不一定是简单的 inflation**、更一般的形式是**把 latent state 向前传播**：

$$p(x_t \mid y_{t-\Delta t}) \;=\; \int p(x_t \mid x_{t-\Delta t})\, p(x_{t-\Delta t} \mid y_{t-\Delta t})\, dx_{t-\Delta t}.$$

机器人静止时 vision age 200 ms、measurement 可能仍然很准；机器人高速运动时同样 200 ms、predictive uncertainty 可能巨大。**inflation 与 propagation 是两种不同的处理方式、不是同义词**。本文把 $\Sigma / \tau$ 明确定位成 **a simple conservative approximation**：

$$\Sigma_c^{\mathrm{eff}} \;=\; \mathrm{Propagate}\!\big(\Sigma_c,\; \Delta t,\; u_t,\; f_{\mathrm{dyn}}\big) \qquad\text{（一般形式）}$$

$$\Sigma_c^{\mathrm{eff}} \;=\; \Sigma_c \,/\, \tau_c(\alpha_c, \ell_c, h_c, v_c, \iota_c) \qquad\text{（一种 conservative approximation）}$$

正确的表述是：**staleness should modify the policy's uncertainty model; uncertainty inflation is one conservative implementation, while predictive state propagation is another**。这一版把 $\Sigma / \tau$ 从"canonical"降级为 "one implementation"、并把 `Propagate` 作为一般形式挂在旁边。$\iota_c$ 与 $v_c$ **并列挂在 metadata 里**、不能塞进 $\mu$、也不能塞进 $\tau$。

这一改动看着小、实际上把整个 age_gate 的语义从"打折读数"修正到了"读数 + 关于读数的元信息 + 派生的 trust"——是 §0.2 $L_{\mathcal C}^{\pi}$ 定义的一个具体投影：把 measurement 与关于 measurement 的**事实**混在一个数值里、就是 $L_{\mathrm{projection}}$ 的直接来源。

### 4.3 `provenance / dependency / negative evidence`：三件事拆开、语义层级不同

9/14 下篇 §2.1 里把 `contributing_mask`、`correlated_with`、`negative_evidence` 都挂在 `provenance` 下——从 estimator 侧看合理（三者都是"这个字段的来源结构"）、但从 **policy 侧的读法**看、三者的语义层级其实不同：

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

这才是"本应看到但没有"的正确刻画——**只有当 $\mathcal O$ 说该看到的时候、$E^-$ 才对 $H$ 有 likelihood 意义**。这也直接接上了 9/14 下篇 §2 里的 observability 字段：**negative evidence 是 likelihood-side evidence、不是 provenance**。

具体到 policy 侧：把 $\Lambda(E;H,\mathcal O)$（或 $\exp(\Lambda)$ 的 logit）作为一个 field 拼进 $[\hat S]_{\sim_\pi}$、让 policy 或 belief update 消费。这条 primitive 目前工程实现最薄、但它对 hypothesis-ranking 的影响往往最大。

三个 sub-primitive 的关系：**provenance 讲 `where evidence came from`、dependency 讲 `how evidence is statistically related`、negative evidence 讲 `what expected evidence failed to appear`——三者共同支撑 §0.2 里的 source-structure invariant**。

### 4.4 三族 primitives 与三条 invariants 的对应

回到 §0.1 的图：**mode / temporal / source** 三条 contract-relevant invariants 分别对应 **mode_select / age_gate / (provenance_harden + dependency_aware_fusion + negative_evidence_read)**。三条 invariant 缺一、§3 的四条 failure 至少复发两条。三族 primitive 也不是"接口设计题的完整答案"——observability / identifiability、frame convention、contact set 这三类 contract 字段还有各自更专门的读法（9/14 下篇 §3、下篇 §4.6 有对应讨论）、本文只处理**最容易在 $\Pi_\pi$ 层被无声破坏的这三条**。

## 5. 承上启下：本文立 framework、下一篇（9/16）立 evaluation

到这里本文（上半）已经完成它该做的一件事——**把 policy 侧接口立成一个可审计的语义对象**。回顾一下：

- §0 建立了 pipeline $\mathcal C\to(\mathcal C_\pi,Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}$、Consumer Contract 三段分解 $\mathcal C_\pi=(Q_\pi,\mathcal O_\pi,V_\pi)$、三档 preservation（full / decision-relevant / decision sufficiency）、三个 semantic losses + 一个 safety obligation、以及 §0.2.1 的 decision-relevant observable $Y_{\mathcal C}^{\pi}$ 与 representation equivalence $\sim_Z$ 定义。
- §1 用**两个正交维度**（conditioning representation / semantic interface × action head）取代"三个家族"式分类、给出 §1.2 grid。
- §2 把"State"在不同 policy 家族里的语义差异钉住——SAC / PPO 用 engineered state、Diffusion / ACT / Flow 用 visual latent、VLA 用 token sequence、π0 用 VLM + proprio token。
- §3 列出**四种 interface-mismatch 失败模式**——multi-hypothesis 被无声折叠 / semantic correctness 缺接口层保证 / concat 打破 temporal alignment / safety 对 validity vs staleness 失明。
- §4 给出**三族 contract-read primitives**——`mode_select`（假设层读出）、`age_gate`（observation payload + metadata quintuple + derived trust）、`provenance_harden / dependency_aware_fusion / negative_evidence_read`（三件事拆开、语义层级不同）。

两条 boxed claim 立在这里、作为本文上半的收束：

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. 完整 pipeline 是**一个可审计的 semantic interface**、三个 loss 分别落在 $q_\pi / \Pi_\pi / \pi_\theta$ 上、第四个槽位 $g_{\mathrm{safety}}$ 不是 loss 是 obligation——**四个 separately auditable failure sites**（v6 措辞、**不是四个 statistical independent losses**、四者通过 $q_\pi \to \Pi_\pi \to \pi_\theta$ 顺序耦合）。

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state head、latent visuomotor policy、autoregressive VLA、diffusion、flow policy 用的 conditioning 与 action generator 都不一样——但它们作为 contract consumer 都必须回答**同一组四个问题**：what may I discard（$q_\pi$；$\mathcal C_\pi$；$L_{\mathrm{declared}}$ via query subsumption）？what did I actually retain（$\Pi_\pi$；$L_{\mathrm{projection}} = I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$）？how should decisions respond to contract interventions（$\pi_\theta$；$L_{\mathrm{decision}}$ via $D_{\mathcal A}$；下篇 §2.2 拆 $E_{\mathrm{contract}}$ / HPC）？what happens when the evidence becomes invalid or unknown（$g_{\mathrm{safety}}=\bigcap_j g_j$；safe→relaxation permitted / unsafe→tighten or stop / unknown→conservative fallback）？攻击的对象是 interface contract、不是模型架构。

$$\boxed{\;\textbf{A policy is a contract consumer, not merely a function approximator.}\;}$$

### 5.1 Claim 3 与本文下半部分（9/16）

本文上半**只到"接口应该被读对"这一层**。要真正让这一 thesis 可检验、还需要三件事：

1. **如何测**——四类 compliance evidence（semantic / representation / decision / safety）+ 五层 evaluation hierarchy（Retention ≠ Sufficiency ≠ Behavioral use ≠ Utility ≠ Safety、四条不蕴含关系）+ 下篇 §2.2 HPC 拆 $E_{\mathrm{contract}}(T^{\mathrm{contract}})$ / HPC($T^{\mathrm{world}}$) + 下篇 §2.5 CAG matched null control。
2. **如何训**——训练时 representation-side probe + 三类 intervention consistency（Type I equivariance / Type II consumer-declared order-constrained response / Type III unconstrained）+ degradation as causal operator augmentation + 下篇 §1.3 safety filter 三态化 constraint certification。
3. **如何落地**——Python 最小可执行接口骨架、v6 修的三处真实 bug（`super().__init__()` 缺失、schema fail-open、top-$k$ API 混淆两个正交 knob）。

这三件事都在**下篇**——[《政策侧评估（下）：如何测、如何训、如何落地》](/zh/articles/2026-09-16-policy-side-evaluation/)。下篇的收束 thesis 是本文上半的对偶：

> **Claim 3 · Contract compliance should be tested by controlled intervention, not inferred from end-to-end success.** 端到端 success 衡量 policy 好不好用、不衡量它有没有把 contract 语义读对。合规 argument 需要**四类证据合流**、需要**一条 oracle baseline 界定每个指标的语义范围**、需要**明确的 retraining protocol**（区分 $\mathrm{CAG}^{\mathrm{fixed}}$ 与 $\mathrm{CAG}^{\mathrm{retrained}}$）。**Probe ≠ semantic compliance、CAG ≠ semantic compliance、safety pass ≠ representation retention**——四类证据不能互相顶替、也不能压成一个 scalar。

一篇文如果既讲 framework 又讲 protocol 又讲 Python skeleton、单文件会到 14000+ 字（本文与 9/14、9/13 同量级）、读者从概念走到落地要一路翻过 30+ 个小节。把 evaluation 与 implementation 拆到下篇、上下两半各自 6000-8000 字量级、读者按需入半：**理论读者读完上半就够、工程读者可以直接从下半 §0 的"与前篇分工"起步、只回上半补 §4 primitives**。

一句收束：**"融合"这个词、以后尽量不用**——它在时间轴上问的是"什么时候合并"、在语义轴上问的是"合并成什么"；9/14 与本文（上半）合起来把第二个问题拆成了**上游交付什么 + policy 作为 consumer 声明承担什么 + $q_\pi$ 允许丢什么 + $e_\pi$ 保住什么 + $\pi_\theta$ 用什么 + $g_{\mathrm{safety}}$ 承担什么义务**六段。剩下"如何测、如何训、如何落地"三段、交给 9/16 下篇。

## Sources

以下 arXiv ID 已联网核过；journal-only 引用不贴 arXiv。按支撑的 section 分组。**Uncertainty / calibration / belief-space 参照（Guo 2017 / PlaNet / DreamerV3）与 evaluation protocol 方法学参照在本文下篇（9/16）单独列出、本文不重复**。

### A · VLA 家族（支撑 §1 grid、§3 Failure 1–2、§5.1 类别 B 下篇展开）

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)（**paper fact**：把 robot action 明确表达成 text token 与 VLM 联合 fine-tune · §3.2 Failure 2 的 tokenizer 侧典型形态、"contract flattening" 是本文分析、不是原论文的 limitation）
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)（**paper fact**：7B VLA、大规模机器人 demonstration 训练；具体到"多相机 + depth + proprioceptive state encoding"是 §4 "Model Architecture & Training"、Table 2 与 §5.1 报告的输入配置、**不是 abstract-level claim**。**本文分析**：即便引到具体 implementation section、"支持哪些输入 modality" ≠ "读到 contract 的哪一站"、本文对 OpenVLA 的批评是 interface analysis、不是原论文 self-limitation）
- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)（**paper fact**：预训练 VLM + proprio token + noisy action chunk + flow matching；**本文分析**：Under the Structured State Contract defined here, π0's standard conditioning interface does not expose an explicit **first-class contract slot** for hypothesis / provenance / age / negative evidence——注意措辞是"standard conditioning interface 未暴露 first-class slot"、**不是说 π0 "丢失了 provenance"**、provenance 从没有被 π0 conditioning interface 承诺过。**"Continuous actions do not imply structured state semantics" 是本文的分析、不是原论文的 self-limitation**）

### B · Diffusion / Generative-Sequence / Flow-Matching Policy（支撑 §1 grid、§5.1 下篇展开）

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 · [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)（**paper fact**：RGB stack + proprio concat + conditional denoising diffusion、强调 action-distribution multimodality；**本文分析**：action-side multimodality ≠ state-side hypothesis preservation——这是本文 schema 下的推论、不是原论文承认的 limitation）
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ACT / ALOHA）, RSS 2023 · [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)（**paper fact**：CVAE + transformer encoder-decoder、核心是 **action chunking over sequences**——本文把 ACT 归入 "generative sequence decoder"、与 diffusion / flow matching 并列、不属于 diffusion family）
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 · [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)（**paper fact**：establish the flow-matching objective as vector-field regression for generative modeling / CNF——**flow matching 本身不是 robot action chunking 论文**；"continuous robot action chunks" 是 π0 这类工作的具体应用、本文把 citation chain 拆成 "Lipman establishes objective / π0 applies it to action chunks"）

### D · SAC / PPO 与 engineered-state head 的 baseline（支撑 §1 grid 首行）

- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, ICML 2018 · [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)（Gaussian NLL / max-entropy policy loss · §5.1 $\mathcal{L}_{\mathrm{action}}$ 的 engineered-state head 形态在下篇 §1.1 展开、本文只作 baseline 参照）

### E · 承接前文（本文与 9/13、9/14、以及本文下篇 9/16 的接口）

- 本博客《拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口》· `/zh/articles/2026-09-14-multimodal-fusion-interface/`（Structured State Contract 定义、Interface Property Benchmark、degradation chain · 本文 §0.2 $q_\pi$ 与 §3–§4 直接建立在其上）
- 本博客《只会看、不会摸：机器人为什么缺一双"手感"的手》· `/zh/articles/2026-09-13-tactile-force-sensing/`（力 / 触觉的四种控制范式、Closed-loop value · §1 grid 里的 action-head 谱系的历史来源）
- **本博客下篇**《政策侧评估（下）：如何测、如何训、如何落地》· `/zh/articles/2026-09-16-policy-side-evaluation/`（**本文的孪生下篇**、承接 §5.1 Claim 3、展开 §5 训练时连锁 / §6 四类 compliance evidence + 五层 hierarchy / §7 Python skeleton / §8 Claim 收束与下一步 benchmark）

---

> **相关阅读**
>
> - [政策侧评估（下）：如何测、如何训、如何落地](/zh/articles/2026-09-16-policy-side-evaluation/)——**本文下篇**、承接本文 framework、展开 evaluation 与 implementation
> - [拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口](/zh/articles/2026-09-14-multimodal-fusion-interface/)——本文的前作、把上游交付物立成 Structured State Contract
> - [只会看、不会摸：机器人为什么缺一双"手感"的手](/zh/articles/2026-09-13-tactile-force-sensing/)——§1 grid 里 action-head 谱系的历史脉络、四种力控范式
> - [VLA 与世界模型：两条路线的分岔与合流](/zh/articles/2026-09-07-vla-world-models/)——本文 §1 conditioning 维度的宏观背景、"世界模型不天然属于 sim-to-real" 的另一面
> - [VLA π 家族速览](/zh/articles/2026-09-05-vla-pi-family/)——π0、π0.5 与 flow-matching action head 的一个具体切面、§1 "continuous action ≠ structured state" 的现场证据
> - [什么是 VLA 模型？一篇讲清楚](/zh/articles/2026-09-03-vla-deep-dive/)——本文假设你已经读过
