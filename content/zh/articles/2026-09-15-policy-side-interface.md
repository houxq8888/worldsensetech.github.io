---
title: 'Contract 立起来之后：VLA、Diffusion Policy 与 π0 到底吃什么？'
slug: "2026-09-15-policy-side-interface"
date: 2026-09-15
draft: false
categories: ["具身智能", "策略学习"]
tags: ["具身智能", "策略学习", "VLA", "Diffusion Policy", "π0", "RT-2", "OpenVLA", "Action Tokenization", "Structured State Contract", "Contract-Preserving Projection", "Contract Information Loss", "Contract-Read Primitives", "Intervention Consistency", "Belief State", "Multimodal Hypothesis", "Provenance", "Dependency Graph", "Negative Evidence", "Uncertainty Calibration", "Safety Filter", "Contract Ablation Gap", "Hypothesis Preservation", "Staleness Response", "Evaluation Metrics"]
description: '《多模态融合接口》那一篇把上游交付物立成了 Structured State Contract——本文问它的对偶：如果 estimator 真的按 contract 交付、policy 侧到底能不能吃到。核心对象是一个显式定义的 policy projection $\Pi_\pi$、以及"是否 contract-preserving"这一可验证的接口性质：可以丢 information、但不能无声明地丢 contract semantics。本文不再按 "VLA / Diffusion / engineered head" 三个互斥家族来切、而是把 policy 拆成两个正交维度——conditioning representation × action head——并给出三族五式的 grid。interface 层给出三条 contract-read primitives——mode_select、age_gate 采用 measurement / uncertainty / age / validity / trust 五字段并列输入而非乘性衰减，provenance / dependency / negative_evidence 三者拆开——并把训练约束分成 representation-side probe 与 intervention-consistency 两类。评估侧提出以 **Contract Ablation Gap (CAG)** 作为总指标、HPS / SDS / PCE 作为分维度诊断、SDS 明确改写成 controlled intervention 下的 Staleness Response Curve。收在三条 claim 上：Contract semantics can be lost at the policy boundary；Contract preservation is not architecture-specific；Contract compliance should be tested by intervention, not inferred from end-to-end success。'
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

短答：**structured estimator output 不意味着 structured policy input**。中间存在一层显式的 projection $\Pi_\pi$、它本身就是一个**可能破坏语义的接口**。VLA 用 tokenizer、Diffusion Policy 用 image-encoder + proprio concat、classical head 用手工 state vector——三种 projection 各自破坏 contract 的不同切片、并且**破坏发生得无声无息**：loss 曲线照样往下走、eval 分数照样往上涨、你从训练日志里看不出被压掉了什么。这不是模型规模的问题、**是接口的问题**——但**问题不在"哪个模型会 collapse"**、**问题在"没有一种架构对 contract-relevant semantics 作了显式 preservation guarantee"**。这一区分很关键、它是本文与常见 "architecture critique" 类文章的分水岭。

本文不推销某个具体 backbone、不反对 end-to-end learning、也不主张"某族天生比另一族更好"。本文的核心 thesis 更 general 也更可验证：**任何 policy architecture 都需要回答——它的 input projection 是否保持上游 contract 中 decision-relevant 的语义？** 这个问题一旦被摆到台面上、"VLA 还是 Diffusion Policy"这种表层争论就自动降级成了一个具体设计决策、而不是立场站队。

## 0. 全文框架：policy projection $\Pi_\pi$ 是一个 semantic interface

先把整篇的分析框架摆在开头、后面每一节都会回到这张图。这一节还要立起本文真正的**形式对象**——$\Pi_\pi$ 与它的 contract-preserving 性质、以及一个可度量的**Contract Information Loss**。

### 0.1 论证主线

```
Structured State Contract  Ŝ_t                          （9/14 定义）
        │
        ▼
policy projection  Π_π(Ŝ_t, o_t, ℓ_t)                   （本文核心对象）
        │
        ▼
问题：哪些 decision-relevant semantics 被保留了？
        │
        ▼
三条 contract-relevant invariants
   ├─ mode          多 hypothesis 的结构             ──▶  primitive: mode_select
   ├─ temporal      异质 staleness 的分布             ──▶  primitive: age_gate
   └─ source        provenance / dependency / 负证据  ──▶  primitive: provenance / dependency / negative_evidence
        │
        ▼
训练（representation-side probe + intervention-consistency）
部署（safety filter 直接读 constraint-relevant contract fields）
评估（CAG 作为总指标 + HPS / SDS / PCE 作为分维度诊断）
```

### 0.2 $\Pi_\pi$ 的形式化：contract-preserving projection

本文把 $\Pi_\pi$ 明确定义成一个**语义接口**、而不是"模型的第一步计算"。给定一组 policy 服务的 **contract-relevant decision variables** $Y_{\mathcal{C}}$（这些是下游 controller / planner / safety filter / diagnostics 会读的量、比如"这个 track 是什么"、"这个接触力是多少牛"、"这个数据有多旧"、"这个通道还有效吗"）、在 $\hat S$ 上定义语义等价关系 $\sim_{\mathcal{C}}$：

$$\hat S \sim_{\mathcal{C}} \hat S' \quad \Longleftrightarrow \quad \forall\, Y_{\mathcal{C}}\text{-relevant query},\;\hat S \text{ 与 } \hat S' \text{ 给出同一份答案}.$$

**定义（Contract-preserving projection）**——$\Pi_\pi$ 是 contract-preserving 的、当且仅当它在语义不等价的 contract 之间**不做不可逆的坍缩**：

$$\hat S \not\sim_{\mathcal{C}} \hat S' \quad \Longrightarrow \quad \Pi_\pi(\hat S) \not\equiv \Pi_\pi(\hat S').$$

**定义（Contract Information Loss）**——用互信息给出一个可度量的降级：

$$L_{\mathcal{C}}(\Pi_\pi) \;=\; I(\hat S;\, Y_{\mathcal{C}}) \;-\; I\!\big(\Pi_\pi(\hat S);\, Y_{\mathcal{C}}\big).$$

由 data processing inequality、$L_{\mathcal{C}} \ge 0$；$\Pi_\pi$ 是 contract-preserving 意味着 $L_{\mathcal{C}} = 0$（或至少"任何 decision-relevant 区分都没被不可逆地折叠")。

这一层形式化很关键、因为它把常见的一句反驳——"当然 projection 会丢信息、任何神经网络都会丢信息"——**直接挡在门外**：**policy 可以丢 information、但不能无声明地丢 contract semantics。** 前者是物理事实、后者是接口违规。本文整篇要问的就是：**主流 policy 的 $\Pi_\pi$ 是不是把"无声明地丢 contract semantics"当成了默认行为。**

### 0.3 三条本文的 boxed claim

> **Claim 1（Contract semantics can be lost at the policy boundary）**——A structured estimator output does not imply a structured policy input. 投影 $\Pi_\pi$ 本身是一个**语义接口**、应该被当成一个语义接口来对待、而不是被当成一次"模型前向的第一步计算"。

> **Claim 2（Contract preservation is not architecture-specific）**——Engineered-state head、latent visuomotor policy、VLA / flow policy 各自丢的是 contract 里不同的语义、但**底层失败模式是同一个**：contract-relevant distinctions 在没有显式 preservation guarantee 的前提下被投影掉了。**攻击的对象是 interface contract、不是模型架构**。

> **Claim 3（Contract compliance should be tested by intervention, not inferred from end-to-end success）**——一个 policy 完全可以在忽略 disagreement、staleness、provenance 的同时仍然取得很高的 task success。contract compliance 必须用**受控干预指标**测、不能从端到端 success rate 反推。

## 1. 两种维度、而不是三个家族：conditioning representation × action head

**旧版把 "VLA / Diffusion Policy / engineered head" 当成三个互斥家族**、这个 taxonomy 有点粗糙——π0 就是 VLA + flow matching、既在 VLA 那一列也在 diffusion/flow 那一列。这一节改成**两个正交维度**、policy 家族的选择就变成 grid 上的一个坐标、而不是一个立场。

### 1.1 两个正交维度

**维度 1：Conditioning representation**——policy 用什么形态读上游信息。四类典型：

- **engineered state**（人工写死的定长向量、$s_t \in \mathbb{R}^d$）
- **visual latent**（image encoder 之后的低维表示、$z_t = f_\phi(o_t)$）
- **multimodal token**（VLM 词表里的离散 token、image + language 共享空间）
- **structured contract**（显式 $\hat S_t$、字段可枚举、可被 controller / policy / safety filter 共同读）

**维度 2：Action head**——policy 怎么产生 action 分布。五类典型：

- **deterministic regression**（一个 mean、不加噪声）
- **Gaussian / mixture stochastic head**（SAC 一脉、mean + variance / GMM）
- **autoregressive token**（离散词表、RT-2 / OpenVLA）
- **diffusion**（多步 denoising、Diffusion Policy）
- **flow matching**（连续 action chunk 的 velocity regression、π0）

### 1.2 Grid

| 代表 policy | Conditioning representation | Action head | 破坏 contract 的位置 |
|---|---|---|---|
| Classical SAC/PPO head | engineered state | Gaussian / deterministic | 最小、字段人工写；但字段设计本身是艺术、跨任务差 |
| Visual-obs PPO / DrQ | visual latent | Gaussian | $z_t$ 一压、provenance / age / hypothesis 通常已经丢 |
| Diffusion Policy (Chi 2023) | visual + proprio latent | diffusion | 破坏 provenance / age / hypothesis；action 侧多模态 OK、但**state 侧多 hypothesis 没有 slot** |
| ACT / ALOHA (Zhao 2023) | visual + proprio latent | CVAE → chunked | 与 Diffusion Policy 类似、state-side 隐式 |
| RT-2 | multimodal token (VLM) | autoregressive action tokens | frame / reference_point / convention 会被 tokenize 掉 |
| OpenVLA | multimodal token + proprio | autoregressive | 同 RT-2、proprio 侧多一条通道、但 contract 字段仍无处安放 |
| π0 (Black 2024) | multimodal token (VLM conditioning) | **flow matching**（连续 action chunk） | action 侧连续、但 state 侧仍走 VLM token、**structured contract 依然被 flatten** |

一句关键区分：**Continuous actions do not imply structured state semantics.** π0 的 action head 是 flow matching 出来的连续 chunk、听起来很"结构"——但它的 conditioning representation 是 VLM token、上游 contract 的 hypothesis / provenance / age 依然在这一层被压平。**action 侧的连续性、不挽救 state 侧的结构丢失**。这条区分是本文与常见"flow matching 更细腻、所以 contract 保留更好"这类直觉的分水岭。

一句 caveat：**这不是"哪个组合最好"的排序**。engineered state + Gaussian 依然是低维控制 baseline 之王、multimodal token + flow matching 依然是开放语义条件下唯一现实的路线——本文关心的是**每一种组合、它的 $\Pi_\pi$ 层是否对 contract-relevant semantics 作了显式 preservation guarantee**。答案在多数现有工作里是"没有"——**不是某个家族天生不行、是这个 layer 从来没有被当成接口设计过**。

## 2. "State" 在不同 policy 里意味着什么

这一节是**术语清障**。"state"这个词在具身智能文献里至少有五种互不相同的意思、policy 家族的选择往往就是"它默认哪一种"的选择：

**$\pi_{\mathrm{obs}}$：raw observation**——图像、点云、力 / 扭矩读数的原始流。绝大多数 imitation learning 训练管线的**输入形态**、但一般不是 policy 内部**实际使用的 state**（会被 encode 掉）。

**$z_t = f_\phi(o_{:t})$：encoded latent**——encoder 之后的低维表示。Diffusion Policy、VLA 的 image tokens、world model 的 RSSM state 都属于这一类。contract 到这里已经**经过一次 projection**、observability 与 provenance 通常已经丢了。

**$b_t$：belief / posterior**——POMDP 一脉的显式 belief state。contract 里"多 hypothesis + posterior weight"结构最贴合这个。但主流 VLA / Diffusion Policy 都不显式建模 belief——它被 encoder 隐式近似。

**$s_t = (b_t, \pi_t, q_t, h_t)$：9/10 Part 1 定义的 allocation state**——belief + policy + budget + hardware。这是**决策层**的 state、不是 policy 的输入 state。它要求 contract 里除了观测、还要能读到"当前哪条 policy、还剩多少 real-rollout 预算"——这些字段在 VLA 里目前完全没接口。

**$\hat S_t$：structured state (contract)**——9/14 定义的那个契约对象。

一条完整的链：

$$\pi_{\mathrm{obs}} \;\xrightarrow{\;\text{encoder}\;}\; z_t \;\xrightarrow{\;\text{abstraction}\;}\; \hat S_t \;\xrightarrow{\;\text{posterior}\;}\; b_t \;\xrightarrow{\;\text{allocation}\;}\; s_t$$

回到 §0.2 的语言：policy 侧真正的问题、**不是"我能不能吃下更长的 token 序列"**、而是"我在 $\pi_{\mathrm{obs}} \to z_t \to \hat S_t \to b_t \to s_t$ 这条链上、愿意承诺走到哪一站、以及是否对 $Y_{\mathcal{C}}$ 保留了 $L_{\mathcal{C}} = 0$ 或至少显式声明丢了什么"。engineered state 停在 $\hat S_t$、visual latent 停在 $z_t$、multimodal token 事实上停在 $\pi_{\mathrm{obs}}$ 之后的一层 tokenizer——**三种停法对应三种 $L_{\mathcal{C}}$、不是一种"更聪明"、一种"更笨"**。

**"多模态融合"这个词在 policy 侧的常见误用**就是把 $\pi_{\mathrm{obs}} \to z_t$ 的这一步 cross-attention、当成"已经在做多模态状态估计"——它不是。真正的 state abstraction 要求 $\hat S_t$ 里的字段**跨传感器家族保持一致的语义、可被 controller / policy / world model / diagnostics 四种消费者共同读**——9/14 §6 已经把这个约定立起来了、本节要做的是**从 policy 一侧再问一遍：$L_{\mathcal{C}}$ 是不是被无声地接受了**。

## 3. Interface mismatch 的四种失败模式

一旦 §1 的两维坐标落到具体 policy 上、contract 语义会以四种**具体失败模式**表现出来。这四条不是理论担忧、是**部署里真会翻车的东西**。四条都是**接口层的事实**、**不是某一族的性格缺陷**——这一点很重要、下面会看到。

### 3.1 Failure 1：Multi-hypothesis 被无声折叠

contract 里同一物理量可能有多个 hypothesis（"这个 track_id 是不是同一个物体"、"这个接触是 pad 还是 edge"）带不同 posterior weight。**如果一个 policy 的 $\Pi_\pi$ 显式对这些 hypothesis 做了 posterior mean / mean-pool、mode collapse 就是确定性的**——问题不在"这个模型会不会 collapse"、问题在**"这个模型有没有为多 hypothesis 结构提供一个显式的 preservation readout"**。绝大多数现有 policy 架构（visual latent 路线与 multimodal token 路线都在内）**没有提供这样的 readout slot**、这就使得 collapse 变成一个**接口默认行为**、而不是训练动力学问题。

需要精确化的一点：**diffusion / flow-matching 在 action 侧的多模态性、不自动意味着 state-side 多 hypothesis 被保留**。action 分布可以是 multimodal（denoising 出来多条 action 轨迹）、但 conditioning 里如果 $\hat S_t$ 的 hypothesis 结构已经被 encoder 折叠、action-side multimodality 只是在**已经丢了上游区分**的输入上做输出侧采样。这两个 multimodality 不是一回事、不能互相担保。

一句改口径的话：**本文批评的不是 Diffusion Policy 会 collapse、是"没有架构显式承诺 hypothesis preservation"这件事**。

### 3.2 Failure 2：Semantic correctness 缺乏接口层保证

contract 里的 frame 字段（`orientation_frame: "tool_flange"`、`reference_point: "contact_center"`、`convention: "right-handed"`）进入 policy 时、若被 tokenizer 变成 token 序列——**注意、这里的问题不是"transformer 学不到 frame transform"**。理论上 transformer 完全可以通过 attention + MLP 学出 $\tau_{p_2} = \tau_{p_1} + (p_1 - p_2) \times f$ 这类硬约束。**真正的问题是、tokenization 本身不提供任何"这个变换被以正确方式解释"的接口层保证**。模型**可以**学到、也可能**没**学到、取决于训练分布是否覆盖 frame 变更的对照样本、取决于 architecture 是否归纳偏置友好、取决于有没有 auxiliary constraint 显式施加这一约束。**在缺失这些条件时、semantic correctness 不是接口层的性质、只是训练数据的偶然产物**。

**一个具体形态**：真机部署里把 wrench 的 `reference_point` 从 `sensor_flange` 改成 `contact_center`——数据前处理管线换了、token 序列几乎一模一样（`"sensor_flange"` 与 `"contact_center"` 都作为 frame token 出现在相似语境里）、但 $\tau$ 数值按 transport theorem 变了一大块。VLA 在这两个"看起来同分布"的 dataset 上分别训练、**可能分别收敛到相似但都错**的 policy——**"都错"不是因为 transformer 学不了 frame transform、是因为训练数据里没有足够的 frame-contrast pair 让它学到、并且没有任何约束强制它学**。这类 bug 只在**部署后**、cross-dataset / cross-embodiment 复用时才暴露。

**改口径的关键句**：tokenization 不抹掉语义、它只是**把语义从"接口保证"降级成了"训练经验"**。这是本文与"tokenizer 就是不行"这种直觉的分水岭。

### 3.3 Failure 3：Temporal alignment broken by concat

contract 里不同 channel 的 `age` 是显式的（例如 proprio 1 ms、F/T 5 ms、vision 100 ms、tactile 30 ms）。若 policy 只看到 concat 后的向量、$\Delta t$ 分布就丢了。表现为 policy 用 vision 的"100 ms 前的世界"与 tactile 的"30 ms 前的接触"作决策、以为它们同步；其实 9/14 §4 已经强调过 **timestamp sync ≠ causal sync**、更不等于 decision-time causal consistency。**concat 掩盖的是 $\Delta t$、不是 $\Delta t$ 的语义**——即使你给每个通道都打上了时间戳、若 policy 没有把时间戳读进决策变量、时间戳就是装饰。

### 3.4 Failure 4：Safety-blindness to validity vs staleness

contract 明确区分 availability（这个通道今天有没有数据）、validity（这个数据是否有效、例如 calibration 是否过期）、age（有多旧）。policy 若只看数值、就把"stale 但 valid"与"missing 但 valid"混成一类、把"calibration drift 后无效"与"传感器掉线"当成同一种降级。9/14 §8.5 的 degradation chain 已经拆开了——**masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption**——policy 若在 input 端不接这个 chain、训练时的 augmentation 与推理时的 guardrail 都会挂错地方。

四条 failure 都是**接口问题**——换 backbone 不解决、只有把 $\Pi_\pi$ 层显式化才解决。这一点之所以重要、是因为它把"要不要换更大 VLA"这个决策与"policy 侧接口要不要重写"这个决策**解耦**。

## 4. Policy 需要的三族 contract-read primitives

顺着 §3 的四条 failure、policy 侧真正需要的是**三族原语**——不是更大的 transformer、不是更多的数据。三族里的第三族（source structure）本身又要再拆三层、见 §4.3。

### 4.1 `mode_select`（假设层的读出方式）

面对 contract 里的多 hypothesis posterior、policy 需要在**读出**时明确选一种：MAP（选最高权重 hypothesis）、sample（按 posterior 采样、鼓励 exploration）、expected-mixture（保留 mixture、下游 head 自己 attend）、或 ECE-preserving top-$k$（保留校准的 $k$ 个 hypothesis）。

$$\text{read}\!\big(\{(\mu_i, \Sigma_i, w_i)\}_{i=1}^{K}\big) \;=\; \left\{\begin{aligned}
&\mu_{\arg\max_i w_i} && \text{(MAP、丢弃低权 hypothesis)}\\
&\textstyle\sum_i w_i\, \mu_i && \text{(posterior mean、显式声明的折叠)}\\
&\mu_i + L_i \epsilon,\;\; i \sim w,\;\; \epsilon \sim \mathcal{N}(0, I) && \text{(posterior sample)}\\
&\big\{(\mu_i, \Sigma_i, w_i)\big\}_{i \in \mathrm{top}\text{-}k} && \text{(ECE-preserving top-}k\text{)}
\end{aligned}\right.$$

关键**不是"mean 不能用"**——mean 是一种完全正当的 readout、只要它是**显式声明**的折叠。真正的失败模式是"接口没有为 hypothesis 结构提供任何 readout slot、policy 只能靠 concat + MLP 隐式合并、结果把 mean 当成了默认"。这一区分很关键：**本文反对的是"无声明的默认折叠"、不是"折叠"本身**。ECE-preserving top-$k$ 之所以有价值、是因为它给了下游一个显式的、可 ablate 的 hypothesis structure、而不是因为 mean 天生错。

### 4.2 `age_gate`：measurement / uncertainty / age / validity / trust 五字段并列、不做乘性衰减

**接口设计的常见 bug** 是把 staleness trust 直接乘进 measurement：$x_c^\pi = \tau_c(a_c) \cdot \mu_c$。这**改变了 observation 的物理值**——10 N 的力、age 100 ms、被乘成 3 N 之后、policy 输入里"3 N"这个数字**看起来**就像"3 N 力"、而不是"10 N 力、但 trust 降低"。这直接违反了 contract 想保护的那个 distinction：**$(F = 3\,\mathrm{N},\, a = 0)$ 与 $(F = 10\,\mathrm{N},\, a = 100\,\mathrm{ms})$ 是两个不同的语义事件**。

正确的做法是把它们**并列**放进 policy input、不做乘法：

$$x_c^{\pi} \;=\; \big[\;\underbrace{\mu_c}_{\text{measurement}}\;,\;\underbrace{\Sigma_c}_{\text{uncertainty}}\;,\;\underbrace{a_c}_{\text{age}}\;,\;\underbrace{v_c}_{\text{validity}}\;,\;\underbrace{q_c}_{\text{trust} \,=\, \tau_c(a_c)}\;\big].$$

trust $q_c$ 是一个**元字段**、不缩放 $\mu_c$、只作为条件变量供 policy 使用。$\tau_c(a_c)$ 的具体形态（指数衰减 / sigmoid / step）不重要、重要的是它作为**独立通道**存在。若确实需要在下游做 gating、**gating 应该作用在 uncertainty 上、不作用在 measurement 上**：

$$\tilde\Sigma_c \;=\; \Sigma_c \,/\, \tau_c(a_c) \quad \text{（stale ⇒ 有效 uncertainty 被放大）}$$

这一直觉接近 Bayesian filter 里 process noise inflation for delayed update、而不是"把读数打折"。$\mathbb{1}[v_c]$（validity）与 availability 位仍**并列挂上**、不能塞进 $\mu$、也不能塞进 $\tau$：

$$\text{availability}_c,\;\;\mathbb{1}[v_c],\;\;a_c,\;\;\mu_c,\;\;\Sigma_c,\;\;q_c \quad \text{——六个 slot、每个各说各话.}$$

这一改动看着小、实际上把整个 age_gate 的语义从"打折读数"修正到了"读数 + 关于读数的元信息"——这是 §0.2 $L_{\mathcal{C}}$ 定义的一个具体投影：把 measurement 与关于 measurement 的**事实**混在一个数值里、就是 $L_{\mathcal{C}}$ 的直接来源。

### 4.3 `provenance / dependency / negative_evidence`：三件事拆开、不是一个大 bucket

9/14 §6.1 里把 `contributing_mask`、`correlated_with`、`negative_evidence` 都挂在 `provenance` 下——从 estimator 侧看合理（三者都是"这个字段的来源结构"）、但从 **policy 侧的读法**看、三者的语义层级其实不同：

| 字段 | 本质 | policy 侧怎么读 |
|---|---|---|
| `contributing_mask` | evidence source（哪些 sensor 贡献了） | 作为 conditioning 拼进 policy input（**provenance_harden**） |
| `correlated_with` | dependency structure（哪些字段结构性相关） | attention mask / bias、避免 double-counting（**dependency_gate**） |
| `negative_evidence` | hypothesis-conditioned absence of expected evidence、$P(\mathcal{E}^- \mid H)$ | 更接近 **belief update 的一等公民**、不属于 provenance bucket（**negative_evidence_read**） |

**`negative_evidence` 不是 provenance**。它是"哪些 sensor 本应看到 $E^-$、但没看到"、语义上更像 likelihood 的一项、与 belief update 关系深。把它塞进 provenance bucket 会让 §4.3 变成一个过大、边界模糊的 primitive。三个 sub-primitive 各自对应 contract 一层：

**`provenance_harden`**：把 `contributing_mask` 直接拼进 policy input。engineered-state head、visual-latent head 都能做（多几维向量）。

**`dependency_gate`**：把 `correlated_with` 转成 attention mask / bias、让 transformer 在 attend 时不重复计算相关证据。**一种可能的实现**（注意"一种可能"）是加性 attention bias：

$$\mathrm{Attn}'_{ij} \;=\; \mathrm{Attn}_{ij} \;-\; \beta \cdot \mathbb{1}\!\big[\text{fields}_i \text{ correlated\_with } \text{fields}_j\big].$$

但要老实承认：**`correlated_with` 是 field-level 语义关系、attention bias 是 token-pair relation**、两者之间需要一层 $R_{\text{field}} \to R_{\text{token}}$ 的映射。一个 field 常被拆成 value token / uncertainty token / age token / provenance token 多个 token、到底哪些 token pair 需要 suppress、**这不是一个已经解决的问题、是一个从 provenance graph 到 attention graph 的编译问题、值得作为独立研究方向**。本文只把这条 primitive 立起来、公式是"一种可能实现"、不声称它是 canonical。

**`negative_evidence_read`**：把"本应看到但没有"的观测作为 likelihood-side conditioning 直接进入 policy 或 belief 更新。具体形态可以是给 policy 一个 $P_{\mathrm{expected}}(\mathcal{E}^- \mid H_k)$ vs $P_{\mathrm{observed}}(\mathcal{E}^-)$ 的差值、让它 attend 到"哪些 hypothesis 因为缺证据正在变弱"。这条 primitive 目前工程实现最薄、但它对 hypothesis-ranking 的影响往往最大。

三个 sub-primitive 的关系：**provenance 讲"从哪来"、dependency 讲"不能重复算"、negative evidence 讲"没看见什么"——三者共同支撑 §0.2 里的 source-structure invariant**。

### 4.4 三族 primitives 与三条 invariants 的对应

回到 §0.1 的图：**mode / temporal / source** 三条 contract-relevant invariants 分别对应 **mode_select / age_gate / (provenance_harden + dependency_gate + negative_evidence_read)**。三条 invariant 缺一、§3 的四条 failure 至少复发两条。三族 primitive 也不是"接口设计题的完整答案"——observability / identifiability、frame convention、contact set 这三类 contract 字段还有各自更专门的读法（9/14 §7、§8.6 有对应讨论）、本文只处理**最容易在 $\Pi_\pi$ 层被无声破坏的这三条**。

## 5. Training-time 与 Deployment-time 的连锁后果

一旦 policy 输入端接了 §4 的 primitives、训练目标、augmentation、safety filter、evaluation 四处都要跟着改。**接口不是免费的**——但改动是**局部的、可控的**。

### 5.1 两类训练约束：representation-side probe 与 intervention-consistency

一个自然的错误是：**为了"让 policy 用 contract"、要求 policy 输出 validity / hypothesis 的预测头**——这实际上是把"用 contract"偷换成了"复制 contract"、方向不对。**policy 完全可以只吃 contract、不吐 contract**——auxiliary prediction head 不是必要条件。

本文建议的是两类更清晰的训练约束：

**类别 A · Representation-side probe**（不是硬约束、是诊断）。给 policy 的中间表示 $z^{\pi}$ 挂几个 probe head、尝试从 $z^{\pi}$ 预测 contract 里的 `age`、`validity`、`observability`、hypothesis posterior。**这些 probe 不参与主 loss、只用来测**：probe 预测得好、说明 policy 内部保留了这些信息；probe 预测得差、说明 contract 在 $\Pi_\pi$ 里已经被压掉了。$L_{\mathrm{probe}}$ 可以以很小的权重加到主 loss 上作为 regularization、但**它的诊断价值大于训练价值**。

$$\mathcal{L}_{\mathrm{total}} \;=\; \mathcal{L}_{\mathrm{action}} \;+\; \underbrace{\alpha\, \mathcal{L}_{\mathrm{probe}}}_{\text{weak regularization, mainly diagnostic}} \;+\; \underbrace{\sum_{\mathcal{C}} \gamma_{\mathcal{C}}\, \mathcal{L}^{\mathcal{C}}_{\mathrm{consistency}}}_{\text{class B, see below}}$$

**类别 B · Intervention-consistency constraint**（这才是核心）。给 contract 施加一个**已知的**变换 $T_{\mathcal{C}}$、要求 policy 输出满足对应的**响应规律**：

$$\mathcal{L}^{\mathcal{C}}_{\mathrm{consistency}} \;=\; D\!\Big(\pi_\theta\!\big(\hat S,\, o,\, \ell\big),\;\; \pi_\theta\!\big(T_{\mathcal{C}}(\hat S),\, o,\, \ell\big);\;\rho_{\mathcal{C}}\Big)$$

$D(\cdot, \cdot; \rho_{\mathcal{C}})$ 是关于变换 $\mathcal{C}$ 的一个特定 divergence 或 penalty $\rho_{\mathcal{C}}$——不是要求 policy 输出"相同"、是要求它按**已知规律**变化。四类典型 $T_{\mathcal{C}}$：

- **frame transform**：把 `reference_point` 从 A 移到 B、$\tau$ 按 transport theorem 变换——**要求 policy 的 action 侧对应地做等价的坐标变换**（equivariance、not invariance）。
- **age increase**：把某 channel 的 `age` 从 5 ms 抬到 200 ms、measurement 不变——**要求 policy 输出分布向 uncertainty inflation 的方向移动**（例如 variance 上升、或 action 更保守）、不能 mean 不动。
- **hypothesis reweighting**：保持 hypothesis set、只改 posterior weight——**要求 policy 输出分布按 weight shift 方向可预测地变化**、top-1 不必然翻转、但分布必须响应。
- **provenance removal**：把某 contributing sensor 从 `contributing_mask` 里去掉——**要求 policy 输出与该 sensor 相关的 confidence 下降**、不能一切照旧。

这四类**都不要求 policy 显式预测什么**、它们要求的是**响应函数符合 contract 语义**。这才是 §0.2 "$\Pi_\pi$ contract-preserving" 定义的**训练侧对应物**。相比"加几个 auxiliary 预测头"、这套约束更贴合本文 thesis、也更有研究味：**我们建议的不是让 policy 复制 contract、是让 policy 在 contract 变换下响应得对**。

### 5.2 Augmentation 必须按 degradation chain 分类生成

9/14 §8.5 强调 **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption**——六种降级各有不同的**因果起源**与不同的**下游读法**。训练时的数据增强如果只用一种（最常见的是 random masking）就会让 policy 把六种降级**全学成同一种**、部署时不能区分"这个通道今天坏了"与"这个通道数据延迟大"。aug 生成器必须**按 chain 分通道**、每类独立分布、并且**每类对应到 §4.2 的六个 slot**（measurement / uncertainty / age / validity / availability / trust）中哪一个被激活。

具体做法：给每个训练 episode 打一个 degradation label、policy input 里显式携带、aug pipeline 用同一个 label 采样。**这不是 curriculum、是 conditioning**——差别在于 curriculum 是"先学简单后学难"、conditioning 是"让 policy 知道现在处在哪种降级"。§5.1 类别 B 的 intervention consistency 与这里的 degradation-conditioned aug 天然配对——**aug 端造 $T_{\mathcal{C}}$ 的样本、loss 端测 policy 对 $T_{\mathcal{C}}$ 的响应是否符合 $\rho_{\mathcal{C}}$**。

### 5.3 Safety filter 与 contract 的接口

约束层（CBF / shield / runtime verifier）**必须读 $a_{\mathrm{proposed}}$**、否则它 filter 什么——这一点不用退让。但本文更精确的 claim 是：**safety filter 不应该把 policy 的 confidence 或 latent belief 当成约束成立性的唯一证据；它应该直接访问 constraint-relevant contract fields**。

具体地、safety filter 除了 $a_{\mathrm{proposed}}$、还应读到：`observability`（当前约束所依赖的量此刻**可不可观**）、`validity`（calibration 是否过期、过期的读数不能触发 constraint、也不能让 constraint 以为"一切正常"）、`negative_evidence`（本应看到但没有的观测、比如"雷达在这个角度什么都没扫到"、这一项接近 §4.3 `negative_evidence_read`）。9/14 §7 讲过 **track_id 是 hypothesis**——那么 safety filter 也不能只信 track_id 匹配、还要看 hypothesis posterior 是否稳定；两个 track 是否 merge / split、直接决定"这个障碍距离"的可信度。

**这三条不进 safety filter、filter 就会用 policy 的 belief 反推约束成立性**、这在低 observability 区域特别危险——policy 的 belief 之所以乐观、是因为它读不到 contract 里的 observability / validity / negative evidence、**filter 如果同样读不到、两者一起盲**。

### 5.4 与 9/10 Part 3 evaluation 的呼应

Sim utility 三维（prediction / ranking / decision）里、policy-side 的 evaluation 主要看 **decision** 这一维——但要加一个 **contract-preservation 维度**：把 contract 拆掉之后 policy 的表现下降多少、就是它对 contract 依赖度的**下界**。这一维度对应 §6 的 CAG 指标。

## 6. Evaluation：一个总指标 + 三个分维度诊断

对应 §4 的 primitives：一个**总指标 CAG** 直接回答"policy 到底有没有用 contract"、三个**分维度诊断 HPS / SDS / PCE** 分别对应 mode / temporal / source 三条 invariant。四个指标都是**干预型 evaluation**——这是本文 Claim 3 的直接对应物。

### 6.1 总指标：Contract Ablation Gap（CAG）

**定义**——给定 contract 的一个特定 collapse 算子 $\mathrm{collapse}_X$（把 $X$ 这一层 contract structure 无声明地折叠掉）、

$$\mathrm{CAG}_X \;=\; J\!\big(\pi_\theta \,\big|\, \hat S\big) \;-\; J\!\big(\pi_\theta \,\big|\, \mathrm{collapse}_X(\hat S)\big),$$

$J$ 是 higher-is-better 的 decision utility（比如任务成功率、或 $-\text{cost}$）。四类 collapse 各自对应一条 invariant：

- $\mathrm{collapse}_{\mathrm{hyp}}$：把 hypothesis set 折成单 Gaussian 或单点估计。
- $\mathrm{collapse}_{\mathrm{age}}$：把所有 channel 的 `age` 抹平为 0。
- $\mathrm{collapse}_{\mathrm{prov}}$：drop `contributing_mask` 与 `correlated_with`、让 attention 无区别地 attend。
- $\mathrm{collapse}_{\mathrm{neg}}$：drop `negative_evidence`。

**CAG 高 = policy 真的在用 contract；CAG ≈ 0 = policy 对 contract 结构漠不关心、即使 end-to-end success rate 很高也一样**。这一条**直接反驳了"我用端到端指标看这个 policy 表现好不好"的常见 evaluation 思路**——一个 policy 完全可以在高 success rate 的同时、把 contract 里所有 structure 都当摆设。CAG 是本文 Claim 3 "intervention-based evaluation" 的核心指标。

### 6.2 分维度诊断 1：Hypothesis Preservation Score (HPS)

同一时刻 contract 里有 $K$ 个 hypothesis、HPS 衡量 policy 输出在 $K$ 个 hypothesis 分别成立的世界里、能覆盖多少个"局部最优 action 簇"。

$$\mathrm{HPS} \;=\; \frac{1}{N} \sum_{n=1}^{N}\, \max_{k}\, \Pr\!\big[\pi_\theta(x_t^{\pi}) \in \mathcal{A}^{*}_k \,\big|\, H_k \text{ is true at } n\big]$$

**$\mathcal{A}^{*}_k$ 不要求在线可获得**——在 benchmark 里由 **privileged simulator state、oracle planner 或 offline expert rollouts** 构造、因此 HPS 是训练 / 评测阶段指标、**不是 deployment-time observable**。这一点必须明说、否则 HPS 会被 reviewer 直接指为不可操作。MAP 读出与 posterior-sample 读出的 HPS 差距、就是 §3.1 Failure 1 的严重度量化。

### 6.3 分维度诊断 2：Staleness Response Curve (SDS)

**旧版把 SDS 定义成一个 KL、$D_{\mathrm{KL}}(\pi(\cdot \mid a^{\mathrm{fresh}}) \| \pi(\cdot \mid a^{\mathrm{stale}}))$——这个定义不完整**。SDS ≈ 0 不能证明 policy 没读 age（可能它读了、然后正确判断这个 action 不需要考虑 staleness）、SDS ≫ 0 也不能证明变化来自 age（stale 图像可能同时改变了 visual content、变化归因不了）。

正确的 SDS 应该是**受控干预下的响应曲线**：固定 observation content、只干预 $a_c$、测 policy 的**响应规律**是否匹配 §5.1 类别 B 里定义的 $\rho_{\mathcal{C}}$（期望 variance 单调升、mean 保持稳定）：

$$\mathrm{SDS}_c(a_1, a_2) \;=\; D\!\Big(\pi_\theta\!\big(\cdot \,\big|\, \mathrm{do}(a_c = a_1),\, o\big),\;\; \pi_\theta\!\big(\cdot \,\big|\, \mathrm{do}(a_c = a_2),\, o\big)\Big)$$

以及导数形式：

$$\left.\frac{\partial\, \mathbb{E}\!\big[\pi_\theta(\cdot \mid \mathrm{do}(a_c = a),\, o)\big]}{\partial a_c}\right|_{a}\quad\text{与 oracle policy 的同阶导数比较}.$$

**Staleness Response Curve** 是这一族量的整体——不是一个标量。曲线的**形状**（variance 单调升 vs 单调降 vs 平坦）、**方向**（mean 是否保持不变）与**尺度**（对多少 ms 起响应）三个性质合起来、才是 policy 对 `age` 读取度的完整画像。平坦不代表一定差、要对着 oracle 的期望响应曲线比。

### 6.4 分维度诊断 3：Provenance-Conditioning Effect (PCE)

把 `correlated_with` 从 policy 输入里去掉、看它在 double-counting-sensitive 场景（比如 proprio + F/T 融合 $\hat F_{\mathrm{ext}}$）上的表现下降多少：

$$\Delta J_{\mathrm{prov}} \;=\; J\!\big(\pi_\theta \mid \text{provenance}\big) \;-\; J\!\big(\pi_\theta \mid \text{provenance} = \varnothing\big),$$

其中 $J$ 一律取 higher-is-better 的 decision utility（成功率、$-\text{cost}$、$-\text{ECE}$ 之类）。写成 $\Delta J_{\mathrm{prov}}$ 而不是"PCE"是为了避免给一个 delta 量起"metric"的名字引起歧义。$\Delta J_{\mathrm{prov}} > 0$ = policy 真在用 provenance；$\Delta J_{\mathrm{prov}} \approx 0$ = 它只是把 provenance 当装饰；$\Delta J_{\mathrm{prov}} < 0$ = 加了反而更差、通常意味着 conditioning 与 backbone 的归纳偏置冲突、要单独 debug。$\Delta J_{\mathrm{prov}}$ 也告诉你 §4.3 里 attention bias 强度 $\beta$ 是否调对了。

四个指标合起来构成一个**干预型 evaluation 面板**：**CAG 是"contract 有没有被用"的总回答、HPS / SDS / PCE 分别是 mode / temporal / source 三条 invariant 的归因**。它们都**不能替代**任何端到端 success rate——它们衡量的是 policy 侧对 contract 的**读取度**、不是**表现力**。这一点与 §0.3 Claim 3 完全对齐：**contract compliance must be tested by intervention**。

## 7. 最小可执行接口草图

把 §4 三族 primitives 与 §6 四指标合起来写成一个 Python 类骨架。**不是要给出一个具体 policy、是要给一个可读的接口约定**。

```python
class StructuredStateView:
    def __init__(self, contract: StructuredState):
        self.contract = contract

    def project(
        self,
        schema: PolicySchema,
        mode: Literal["map", "sample", "ece_preserving"] = "ece_preserving",
        staleness: Literal["ignore", "parallel_field", "condition"] = "parallel_field",
        provenance: Literal["ignore", "harden", "attention_bias"] = "harden",
        dependency: Literal["ignore", "attention_bias", "learned"] = "attention_bias",
        negative_evidence: Literal["ignore", "condition", "belief_update"] = "condition",
    ) -> PolicyInput:
        """
        Project StructuredState (upstream, 9/14) to PolicyInput (downstream, this piece).
        五个 knob 都是 policy-specific——同一个 contract、VLA 与 diffusion policy 的
        knob 值就该不一样；staleness 默认走 parallel_field、绝不乘性打折 measurement。
        """
        slots = {}
        for field_name in schema.fields:
            h = self.contract[field_name]   # hypothesis set: [(mu_i, Sigma_i, w_i)]

            # --- mode_select ------------------------------------------
            if mode == "map":
                mu, Sigma, w_meta = h.most_likely().mu, h.most_likely().Sigma, None
            elif mode == "sample":
                mu, Sigma, w_meta = h.sample().mu, h.sample().Sigma, None
            else:  # "ece_preserving"
                mu, Sigma, w_meta = h.top_k_with_weights(k=schema.k_per_field[field_name])

            # --- age_gate: parallel fields, never multiplicative on mu -
            age = h.age
            validity = h.validity_ok
            availability = h.available
            trust = tau_curve(age, schema.tau_bar[field_name])   # q_c, a meta field
            # optional: uncertainty inflation for downstream gating
            Sigma_eff = Sigma / trust if schema.use_uncertainty_gate else Sigma

            slots[field_name] = dict(
                mu=mu, Sigma=Sigma_eff, age=age,
                validity=validity, availability=availability,
                trust=trust, w_meta=w_meta,
            )

        # --- provenance / dependency / negative_evidence, three reads ---
        prov_mask = self.contract.contributing_mask if provenance == "harden" else None
        attn_bias = (compile_field_graph_to_token_graph(self.contract.correlated_with)
                     if dependency == "attention_bias" else None)
        neg_ev = self.contract.negative_evidence if negative_evidence == "condition" else None

        return PolicyInput(slots=slots,
                           contributing_mask=prov_mask,
                           attention_bias=attn_bias,
                           negative_evidence=neg_ev,
                           schema=schema)


class ContractAwarePolicy(nn.Module):
    def __init__(self, backbone, schema: PolicySchema, cfg: ContractReadConfig):
        self.backbone = backbone
        self.schema = schema
        self.cfg = cfg

    def forward(self, state: StructuredState, obs, lang) -> ActionDistribution:
        x = StructuredStateView(state).project(self.schema, **self.cfg.as_kwargs())
        return self.backbone(x, obs, lang)
```

三条 caveat 明确写死：

- **(i)** 这不是唯一读法、五个 knob 都是 policy-specific——同一个 contract、VLA 与 Diffusion Policy 的 knob 值就该不一样、engineered state head 与 visual latent head 的最优 `staleness` 也不同。
- **(ii)** 这个接口**只解决输入端**；§5.1 的两类训练约束、§5.2 的 degradation-conditioned aug、§5.3 的 safety-filter 直连、如果一处不改、$\Pi_\pi$ 层写得再漂亮也会被训练动力绕过去（loss 会自己找到最省事的"把 contract 压扁"路径）。
- **(iii)** `dependency="attention_bias"` 只在 transformer-family backbone 上有实现路径、且**field-graph → token-graph 编译**（`compile_field_graph_to_token_graph`）本文只给了函数名、没给 canonical 实现——这是一个真实的开放问题；MLP head 走 `learned` 那条、让网络自己从 `correlated_with` 学一个偏置矩阵。

## 8. 三条收束 claim（与 §0.3 对齐）

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. 投影 $\Pi_\pi$ 本身是一个**语义接口**、应该被当成语义接口来对待、不是"模型前向的第一步计算"。

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state head、latent visuomotor policy、VLA / flow policy 各自丢的是 contract 里不同的语义、**但底层失败模式是同一个**——contract-relevant distinctions 在没有显式 preservation guarantee 的前提下被投影掉了。攻击的对象是 interface contract、不是模型架构；π0 是 VLA + flow matching、Diffusion Policy 是 visual-latent + diffusion——用两个正交维度切、比用"三个家族"切更贴近事实、也更不容易被"某族天生好"的直觉误导。

> **Claim 3 · Contract compliance should be tested by intervention, not inferred from end-to-end success.** 一个 policy 完全可以在忽略 disagreement、staleness、provenance 的同时、仍然取得很高的 task success。contract compliance 必须用**受控干预指标**测——CAG 作为总回答、HPS / SDS / PCE 作为分维度归因、§5.1 类别 B 的 intervention-consistency loss 作为训练侧对应物——不能从端到端 success rate 反推。

一句收束：**"融合"这个词、以后尽量不用**——它在时间轴上问的是"什么时候合并"、在语义轴上问的是"合并成什么"；9/14 与本文合起来把第二个问题拆成了**上游交付什么 + 下游读出什么**两半。剩下第一个问题（时机）已经被 §1 的两维对照回答得差不多——**时机是接口的结果、不是接口的决策变量**。这一句立住、这一系列的文章就都值了。

## Sources

以下 arXiv ID 已联网核过；journal-only 引用不贴 arXiv。按支撑的 section 分组。

### A · VLA 家族（支撑 §1 grid、§3 Failure 1–2、§5.1 类别 B）

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)（把 action 表达成 text token 与 VLM 联合 fine-tune · §3 Failure 2 semantic correctness 缺乏接口层保证的典型形态）
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)（开源 VLA 基线 · 公开配置含多相机 / depth / proprioceptive state encoding；"支持输入" ≠ "读到 contract 的哪一站"）
- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)（VLM backbone + proprio token + noisy action chunk + flow matching · **§1.2 "Continuous actions do not imply structured state semantics"** 的直接依据）
- Octo Model Team, *Octo: An Open-Source Generalist Robot Policy*, RSS 2024 · [arXiv:2405.12213](https://arxiv.org/abs/2405.12213)（transformer-based readout · §4.3 dependency_gate attention_bias 路径的一个参照）

### B · Diffusion / Flow-Matching Policy（支撑 §1 grid、§5.1）

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 · [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)（RGB stack + proprio concat + denoising · §3.1 "action-side multimodality 不担保 state-side hypothesis preservation" 的现场证据）
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ACT / ALOHA）, RSS 2023 · [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)（CVAE + transformer encoder-decoder、chunked action · 家族 B 与 C 的一个中间形态）
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 · [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)（continuous action chunk 的 velocity regression · §5.1 $\mathcal{L}_{\mathrm{action}}$ 的一种具体形式）

### C · 不确定性、校准与 belief-space 参照（支撑 §4.1、§5.1、§6.1 CAG）

- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)（现代网络过度自信、temperature scaling 起点 · §5.1 $\mathcal{L}_{\mathrm{probe}}$ 校准维度动机）
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels*（PlaNet / RSSM）, ICML 2019 · [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)（deterministic + stochastic latent、多 hypothesis 的一种工程实现 · §4.1 read 的一个参照）
- Hafner et al., *Mastering Diverse Control Tasks through World Models*（DreamerV3）, Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)（离散 + 连续混合 latent、KL balancing · §4.1 ECE-preserving top-$k$ 的一条相邻路线）

### D · SAC / PPO 与 engineered-state head 的 baseline（支撑 §1 grid 首行）

- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, ICML 2018 · [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)（Gaussian NLL / max-entropy policy loss · §5.1 $\mathcal{L}_{\mathrm{action}}$ 的 engineered-state head 形态）

### E · 承接前文（本文与 9/13、9/14、Sim-to-Real P1/P3 的接口）

- 本博客《拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口》· `/zh/articles/2026-09-14-multimodal-fusion-interface/`（Structured State Contract 定义、Interface Property Benchmark、degradation chain · 本文 §0.2 $L_{\mathcal{C}}$ 与 §3–§6 直接建立在其上）
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
