---
title: 'Contract 立起来之后：VLA、Diffusion Policy 与 π0 到底吃什么？'
slug: "2026-09-15-policy-side-interface"
date: 2026-09-15
draft: false
categories: ["具身智能", "策略学习"]
tags: ["具身智能", "策略学习", "VLA", "Diffusion Policy", "π0", "RT-2", "OpenVLA", "Action Tokenization", "Structured State Contract", "Declared Quotient", "Semantic Preservation", "Decision Sufficiency", "Contract Information Loss", "Contract-Read Primitives", "Intervention Consistency", "Equivariance", "Monotone Response", "Likelihood-Side Evidence", "Dependency-Aware Fusion", "Calibration-Aware Top-k", "Constraint-Relevant Validity", "Contract Ablation Gap", "Hypothesis Coverage", "Hypothesis Separation", "Staleness Response Compliance", "Evaluation Metrics"]
description: '《多模态融合接口》那一篇把上游交付物立成了 Structured State Contract——本文问它的对偶：如果 estimator 真的按 contract 交付、policy 侧到底能不能吃到。核心对象是一个显式分解的 policy projection $\Pi_\pi = e_\pi \circ q_\pi$——先声明允许丢什么（$q_\pi$、declared quotient）、再让神经网络去做实际编码（$e_\pi$）；本文的 thesis 是把 semantic preservation（quotient 上的 injectivity）与 decision sufficiency（$L_{\mathrm{dec}}(\Pi_\pi)$）严格分开——两个不等价、前者强于后者。本文不再按 "VLA / Diffusion / engineered head" 三个互斥家族来切、而是把 policy 拆成两个正交维度——conditioning representation / semantic interface × action head——并给出七行 grid。interface 层给出三条 contract-read primitives：mode_select 用 calibration-aware top-$k$ 读出并处理 residual mass；age_gate 用 measurement / uncertainty / age / validity / health / latency / trust 七个 slot 并列、$\tau_c$ 不再只是 age 的函数、$\Sigma/\tau$ 只作为 conservative approximation 的一种、与 predictive state propagation 并列；provenance / dependency / negative evidence 三分、`dependency_gate` 升级为 **dependency-aware fusion**（相关性 ≠ 冗余 ≠ double counting）、negative evidence 定义成 observability-conditional likelihood ratio $\Lambda^-(H)$。训练侧的 intervention-consistency 分成三类：**Type I exact equivariance / Type II monotone response / Type III unconstrained**、不再预设"变化即合规"。评估侧改成**四层面板**——Semantic / Representation / Decision / Safety——并以 J_full / J_collapse / J_oracle 三条 baseline 界定 CAG 的解释边界：**CAG 衡量 task-conditional contract utility、而不是 semantic understanding**。收在三条 claim 上：$\Pi_\pi$ 决定哪些 contract distinctions 仍对下游可用；所有 policy 都必须回答同一个 interface 问题；contract compliance 必须通过 invariance / equivariance / monotone response / task-conditional utility 四类受控测试、不能从端到端 success rate 反推。'
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

## 0. 全文框架：policy projection $\Pi_\pi = e_\pi \circ q_\pi$ 是一个 semantic interface

先把整篇的分析框架摆在开头、后面每一节都会回到这张图。这一节还要立起本文真正的**形式对象**——$\Pi_\pi$ 的**两层分解**、**semantic preservation** 与 **decision sufficiency** 的**区分**、以及一个可度量的 **Contract Information Loss**。

### 0.1 论证主线

```
Structured State Contract  Ŝ_t                             （9/14 定义）
        │
        ▼
declared quotient  q_π :  Ŝ_t ↦ [Ŝ_t]_{C_π}                （本文提出、允许丢什么的显式声明）
        │
        ▼
neural encoding  e_π :  [Ŝ]_{C_π} ↦ z_π                    （现有 policy 都在做、但没有 q_π 的语义账本）
        │
        ▼
policy projection  Π_π  =  e_π ∘ q_π
        │
        ▼
问题：哪些 decision-relevant semantics 被保留了？
        │
        ▼
三条 contract-relevant invariants
   ├─ mode          多 hypothesis 的结构             ──▶  primitive: mode_select
   ├─ temporal      异质 staleness / health 的分布    ──▶  primitive: age_gate
   └─ source        provenance / dependency / 负证据  ──▶  primitives: provenance / dependency / negative evidence
        │
        ▼
训练（representation-side probe + Type I equivariance / Type II monotone / Type III unconstrained）
部署（safety filter 读 constraint-relevant evidence、非 estimator 通用 validity 位）
评估（四层面板：Semantic / Representation / Decision / Safety、CAG 只是其中 decision 层的聚合、且需 oracle baseline 界定）
```

### 0.2 $\Pi_\pi = e_\pi \circ q_\pi$：先声明允许丢什么、再谈编码

本文把 $\Pi_\pi$ 明确定义成一个**两层可分解的语义接口**、而不是"模型的第一步计算"。**这一层分解是本文相对上一版最重要的改动**：上一版把 $\Pi_\pi$ 当单个 projection 处理、reviewer 会立刻指出"$L_{\mathcal{C}} = 0$ 与 injectivity 不等价"、这一版通过显式引入 **declared quotient** $q_\pi$ 把这两个性质分派到不同的层。

**Layer q · Declared quotient**——$q_\pi : \hat S \mapsto [\hat S]_{\mathcal{C}_\pi}$、把上游 contract 折到某个**显式声明过的等价类空间**上。$q_\pi$ 不是神经网络、是接口规格里的一段——**它回答"这个 policy 声明自己允许丢哪些 contract distinctions"**。它可以声明"我保留 hypothesis 结构"、也可以声明"我把 hypothesis 折成 point estimate、但把 age 保留成独立字段"、**关键是这个折法必须写在接口上、而不是藏在 encoder 里**。

**Layer e · Neural encoding**——$e_\pi : [\hat S]_{\mathcal{C}_\pi} \mapsto z_\pi$、把商空间里的等价类送进 encoder / tokenizer / concat、变成 policy 实际吃下的向量或 token 序列。绝大多数现有工作只有 $e_\pi$、没有显式的 $q_\pi$——这就是本文要攻击的对象。

$$\Pi_\pi \;=\; e_\pi \circ q_\pi.$$

**真正要检查的是 $q_\pi$**——它有没有把**不允许丢的 contract distinctions** 合并。这一层分解让整篇文章从"给 VLA 加 metadata"升级成一个更一般的主张：**contract-aware representation design**。

### 0.2.1 Semantic preservation ≠ Decision sufficiency

这一小节是本文的理论锚点、必须把两个容易混用的性质彻底分开。

给定一组 policy 服务的 **contract-relevant decision variables** $Y_{\mathcal{C}}$（这些是下游 controller / planner / safety filter / diagnostics 会读的量、比如"这个 track 是什么"、"这个接触力是多少牛"、"这个数据有多旧"、"这个通道还有效吗"）、在 $\hat S$ 上定义 contract 语义等价关系 $\sim_{\mathcal{C}}$：

$$\hat S \sim_{\mathcal{C}} \hat S' \quad \Longleftrightarrow \quad \forall\, Y_{\mathcal{C}}\text{-relevant query},\;\hat S \text{ 与 } \hat S' \text{ 给出同一份答案}.$$

**性质 A · Semantic preservation**——投影在 contract 语义不等价之间不做不可逆坍缩：

$$\hat S \not\sim_{\mathcal{C}} \hat S' \quad \Longrightarrow \quad \Pi_\pi(\hat S) \not\equiv \Pi_\pi(\hat S').$$

这实际上是要求 $\Pi_\pi$（或更准确地说 $q_\pi$）**在 contract-semantic quotient 上是 injective**。这是一个**强性质**。

**性质 B · Decision sufficiency**——用互信息衡量"当前定义的决策变量所需信息是否被保留"：

$$L_{\mathrm{dec}}(\Pi_\pi) \;=\; I(\hat S;\, Y_{\mathcal{C}}) \;-\; I\!\big(\Pi_\pi(\hat S);\, Y_{\mathcal{C}}\big).$$

由 data processing inequality，$L_{\mathrm{dec}} \ge 0$；$L_{\mathrm{dec}} = 0$ 意味着 $Y_{\mathcal{C}} \perp\!\!\!\perp \hat S \mid \Pi_\pi(\hat S)$、即 $\Pi_\pi(\hat S)$ 对 $Y_{\mathcal{C}}$ 是 **sufficient representation**。

**关键**：性质 A 与性质 B 不是同一个东西。$L_{\mathrm{dec}} = 0$ 只保证充分性、**并不蕴含** injectivity——完全可以在保持 $Y_{\mathcal{C}}$ 充分的前提下、把两个语义不等价的 $\hat S, \hat S'$ 折叠到同一个 $\Pi_\pi$ 输出上（只要 $Y_{\mathcal{C}}$ 本身不区分它们）。**Semantic preservation is strictly stronger than decision sufficiency**。

于是本文的要求可以写成一句很干净的话：

> **Policy 可以丢 information、但必须 either (a) preserve contract semantics, or (b) explicitly declare a sufficient quotient of the contract.**

也就是：**要么保留整个商结构、要么把"丢了哪一层"作为接口的一部分写出来、并且证明这个 quotient 对它声明的 $Y_{\mathcal{C}}$ 是 sufficient 的**。这一区分也解释了为什么"projection 一定会丢信息"不是对本文的反驳：信息损失本身不是接口违规；**未经声明地丢掉 decision-relevant contract semantics、才是接口违规**。

$e_\pi$ 那一层丢不丢得干净、$q_\pi$ 那一层允不允许丢、reviewer 一眼能审——这就是本文相对上一版的最大升级。

### 0.3 三条本文的 boxed claim

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. The policy projection $\Pi_\pi$ is **a semantic interface**: it determines which contract distinctions remain available to downstream decision making——不是"模型前向的第一步计算"。

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state head、latent visuomotor policy、VLA / diffusion / flow policy 用的 conditioning 与 action generator 机制都不一样、但它们都必须回答**同一个 interface 问题**——哪些 contract distinctions 被显式保留、哪些被压缩、哪些被丢弃？攻击的对象是 interface contract、不是模型架构。

> **Claim 3 · Contract compliance should be tested by controlled intervention, not inferred from end-to-end success.** 端到端 success 衡量的是 policy 好不好用、而不是它有没有把 contract 语义读对。contract compliance 需要一组**受控测试**——invariance / equivariance / monotone response / task-conditional utility under contract interventions——**并且需要一条 oracle baseline 来界定每个指标的语义范围**。

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

一条完整的链：

$$\pi_{\mathrm{obs}} \;\xrightarrow{\;\text{encoder}\;}\; z_t \;\xrightarrow{\;\text{abstraction}\;}\; \hat S_t \;\xrightarrow{\;\text{posterior}\;}\; b_t \;\xrightarrow{\;\text{allocation}\;}\; s_t$$

回到 §0.2 的语言：policy 侧真正的问题、**不是"我能不能吃下更长的 token 序列"**、而是"我在 $\pi_{\mathrm{obs}} \to z_t \to \hat S_t \to b_t \to s_t$ 这条链上、愿意承诺走到哪一站、以及是否**把允许丢哪一层写成 $q_\pi$**、使得 $e_\pi$ 之后至少对 $Y_{\mathcal{C}}$ 是 sufficient 的"。engineered state 停在 $\hat S_t$、visual latent 停在 $z_t$、multimodal token 事实上停在 $\pi_{\mathrm{obs}}$ 之后的一层 tokenizer——**三种停法对应三种 $q_\pi$、不是一种"更聪明"、一种"更笨"**。

**"多模态融合"这个词在 policy 侧的常见误用**就是把 $\pi_{\mathrm{obs}} \to z_t$ 的这一步 cross-attention、当成"已经在做多模态状态估计"——它不是。真正的 state abstraction 要求 $\hat S_t$ 里的字段**跨传感器家族保持一致的语义、可被 controller / policy / world model / diagnostics 四种消费者共同读**——9/14 §6 已经把这个约定立起来了、本节要做的是**从 policy 一侧再问一遍：$q_\pi$ 是不是被显式声明过、还是被 $e_\pi$ 悄悄替代了**。

## 3. Interface mismatch 的四种失败模式

一旦 §1 的两维坐标落到具体 policy 上、contract 语义会以四种**具体失败模式**表现出来。这四条不是理论担忧、是**部署里真会翻车的东西**。四条都是**接口层的事实**、**不是某一族的性格缺陷**。

### 3.1 Failure 1：Multi-hypothesis 被无声折叠

contract 里同一物理量可能有多个 hypothesis（"这个 track_id 是不是同一个物体"、"这个接触是 pad 还是 edge"）带不同 posterior weight。**如果一个 policy 的 $q_\pi$ 没有声明 hypothesis 层的 quotient、$e_\pi$ 就会在 concat + MLP 的默认动力下把它折成 posterior mean**——问题不在"这个模型会不会 collapse"、问题在**"接口有没有为多 hypothesis 结构提供一个显式的 preservation readout slot"**。绝大多数现有 policy 架构（visual latent 路线与 multimodal token 路线都在内）**没有提供这样的 slot**、这就使得 collapse 变成一个**接口默认行为**、而不是训练动力学问题。

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

面对 contract 里的多 hypothesis posterior、policy 需要在**读出**时明确选一种：MAP、posterior sample、expected-mixture、或者 **calibration-aware top-$k$**。

$$\text{read}\!\big(\{(\mu_i, \Sigma_i, w_i)\}_{i=1}^{K}\big) \;=\; \left\{\begin{aligned}
&\mu_{\arg\max_i w_i} && \text{(MAP、丢弃低权 hypothesis)}\\
&\textstyle\sum_i w_i\, \mu_i && \text{(posterior mean、显式声明的折叠)}\\
&\mu_i + L_i \epsilon,\;\; i \sim w,\;\; \epsilon \sim \mathcal{N}(0, I) && \text{(posterior sample)}\\
&\big\{(\mu_i, \Sigma_i, \tilde w_i)\big\}_{i \in \mathrm{top}\text{-}k} \;\cup\; \{w_{\mathrm{other}}\} && \text{(calibration-aware top-}k\text{)}
\end{aligned}\right.$$

**"ECE-preserving top-$k$" 上一版的写法被 reviewer 抓了个正着**——top-$k$ 截断本身并不 preserve ECE、因为 $\sum_{i \in \text{top-}k} w_i < 1$、residual mass 被无声扔掉。这一版改叫 **calibration-aware top-$k$ readout**、并显式处理 residual：

$$\tilde w_i \;=\; \frac{w_i}{\sum_{j \in \mathrm{top}\text{-}k} w_j}, \qquad w_{\mathrm{other}} \;=\; 1 - \sum_{i \in \mathrm{top}\text{-}k} w_i.$$

$\tilde w_i$ 是**在 top-$k$ 内部**重归一化的权重、$w_{\mathrm{other}}$ 是**剩下的残差**、两样都保留给下游——只有这样才能保证下游若真的要用 hypothesis weight 做 Bayesian update 时、mass 是守恒的。**"calibration-aware" 是一个 design criterion、不是一个 guarantee**——这一句必须写清楚、否则又是一个未定义的漂亮名字。

关键**不是"mean 不能用"**——mean 是一种完全正当的 readout、只要它是**显式声明**的折叠。真正的失败模式是"接口没有为 hypothesis 结构提供任何 readout slot、$e_\pi$ 只能靠 concat + MLP 隐式合并、结果把 mean 当成了默认"。这一区分很关键：**本文反对的是"无声明的默认折叠"、不是"折叠"本身**。

### 4.2 `age_gate`：measurement / uncertainty / age / validity / health / latency / trust 七字段并列

**接口设计的常见 bug** 是把 staleness trust 直接乘进 measurement：$x_c^\pi = \tau_c(a_c) \cdot \mu_c$。这**改变了 observation 的物理值**——10 N 的力、age 100 ms、被乘成 3 N 之后、policy 输入里"3 N"这个数字**看起来**就像"3 N 力"、而不是"10 N 力、但 trust 降低"。这直接违反了 contract 想保护的那个 distinction：**$(F = 3\,\mathrm{N},\, a = 0)$ 与 $(F = 10\,\mathrm{N},\, a = 100\,\mathrm{ms})$ 是两个不同的语义事件**。

正确的做法是把它们**并列**放进 policy input、不做乘法：

$$x_c^{\pi} \;=\; \big[\;\underbrace{\mu_c}_{\text{measurement}}\;,\;\underbrace{\Sigma_c}_{\text{covariance}}\;,\;\underbrace{a_c}_{\text{age}}\;,\;\underbrace{v_c}_{\text{validity}}\;,\;\underbrace{h_c}_{\text{sensor health}}\;,\;\underbrace{\ell_c}_{\text{latency / causal}}\;,\;\underbrace{q_c}_{\text{trust}}\;\big].$$

**上一版只写了 $q_c = \tau_c(a_c)$、reviewer 立刻指出这与 §3.3 "timestamp sync ≠ causal sync" 内部矛盾**——trust 显然不只取决于 age、还取决于传感器健康、latency、calibration status、task context。这一版改成：

$$q_c \;=\; \tau_c(a_c,\; v_c,\; h_c,\; \ell_c).$$

age 只是四个输入之一、$\tau_c$ 的具体形态（learned / analytic / piecewise）作为 policy-specific 设计决策留在接口外。

**若确实需要在下游做 gating、gating 应该作用在 uncertainty 上、不作用在 measurement 上**。但**上一版把 $\tilde\Sigma_c = \Sigma_c / \tau_c(a_c)$ 写成"stale ⇒ 有效 uncertainty 被放大"的一般原则、reviewer 也抓到了**——真实情况是、stale observation 的正确处理**不一定是简单的 inflation**、更一般的形式是**把 latent state 向前传播**：

$$p(x_t \mid y_{t-\Delta t}) \;=\; \int p(x_t \mid x_{t-\Delta t})\, p(x_{t-\Delta t} \mid y_{t-\Delta t})\, dx_{t-\Delta t}.$$

机器人静止时 vision age 200 ms、measurement 可能仍然很准；机器人高速运动时同样 200 ms、predictive uncertainty 可能巨大。**inflation 与 propagation 是两种不同的处理方式、不是同义词**。本文把 $\Sigma / \tau$ 明确定位成 **a simple conservative approximation**：

$$\Sigma_c^{\mathrm{eff}} \;=\; \mathrm{Propagate}\!\big(\Sigma_c,\; \Delta t,\; u_t,\; f_{\mathrm{dyn}}\big) \qquad\text{（一般形式）}$$

$$\Sigma_c^{\mathrm{eff}} \;=\; \Sigma_c \,/\, \tau_c(a_c, v_c, h_c, \ell_c) \qquad\text{（一种 conservative approximation）}$$

正确的表述是：**staleness should modify the policy's uncertainty model; uncertainty inflation is one conservative implementation, while predictive state propagation is another**。这一版把 $\Sigma / \tau$ 从"canonical"降级为 "one implementation"、并把 `Propagate` 作为一般形式挂在旁边。availability 与 $\mathbb{1}[v_c]$ 仍**并列挂上**、不能塞进 $\mu$、也不能塞进 $\tau$。

这一改动看着小、实际上把整个 age_gate 的语义从"打折读数"修正到了"读数 + 关于读数的元信息"——是 §0.2 $L_{\mathrm{dec}}$ 定义的一个具体投影：把 measurement 与关于 measurement 的**事实**混在一个数值里、就是 $L_{\mathrm{dec}}$ 的直接来源。

### 4.3 `provenance / dependency / negative evidence`：三件事拆开、语义层级不同

9/14 §6.1 里把 `contributing_mask`、`correlated_with`、`negative_evidence` 都挂在 `provenance` 下——从 estimator 侧看合理（三者都是"这个字段的来源结构"）、但从 **policy 侧的读法**看、三者的语义层级其实不同：

| primitive | semantic role | policy 侧怎么读 |
|---|---|---|
| **provenance** | *where* evidence came from | `contributing_mask` 拼进 policy input 作为 conditioning（**provenance_harden**） |
| **dependency** | *how* evidence is statistically related | dependency-aware fusion / conditioning（**dependency_aware_fusion**） |
| **negative evidence** | *what expected evidence failed to appear* | observability-conditional likelihood ratio（**negative_evidence_read**） |

#### 4.3.1 `provenance_harden`

把 `contributing_mask` 直接拼进 policy input。engineered-state head、visual-latent head 都能做（多几维向量）。这一条相对简单、不展开。

#### 4.3.2 `dependency_aware_fusion`（原 `dependency_gate`）

**上一版把 `correlated_with` 处理成 attention bias、用来 "suppress double-counting"——这个直觉不是总成立的**。相关性 ≠ 冗余 ≠ double counting。举个反例：camera depth + tactile contact + F/T wrench 三者高度相关时、恰恰**因为它们给出一致证据**、才应该得到高 confidence；把这种相关性 suppress 掉是错的。**真正的问题是：correlation 有没有在 estimator / uncertainty model 那边被 accounting for**——若 $\Sigma_{12}$ 已经在 Kalman-style fusion 里被建模、$P(x \mid y_1, y_2)$ 就自动正确；**只有当条件相关结构没建模、才存在 double-counting**。

因此本文把这条 primitive 从 `dependency_gate` 升级成 **`dependency_aware_fusion`**：它的语义不是"suppress 相关的 token pair"、而是"让 policy 或 fusion 层知道证据之间的统计关系、并做出相应反应"。**attention bias 只是众多实现之一**（且只适用于 transformer-family backbone）：

$$\mathrm{Attn}'_{ij} \;=\; \mathrm{Attn}_{ij} \;-\; \beta \cdot \mathbb{1}\!\big[\text{fields}_i \text{ correlated\_with } \text{fields}_j\big].$$

其它实现包括：covariance-aware fusion（把 `correlated_with` 转成 $\Sigma_{12}$、走 Kalman / factor graph）、hierarchical mixture（把相关字段聚到同一 latent 下、避免被独立采样）、以及让网络自己从 `correlated_with` 学一个 bias 矩阵（MLP head 的 `learned` 路径）。

同时老实承认：**`correlated_with` 是 field-level 语义关系、attention bias 是 token-pair relation**、两者之间需要一层 $R_{\text{field}} \to R_{\text{token}}$ 的映射、**这是一个尚未解决的编译问题**、值得作为独立研究方向。本文只把 `dependency_aware_fusion` 这条 primitive 立起来、上面那条 attention bias 公式是"一种实现"、不声称它是 canonical。

#### 4.3.3 `negative_evidence_read`（likelihood-side evidence）

**上一版把 negative_evidence 简单写成 $P(\mathcal{E}^- \mid H)$、reviewer 抓得对——这个定义不够严**。$E^-$ = "没观察到" 并不自动意味着 $P(E^- \mid H)$ 低——**occlusion / limited FOV / sensor saturation / low SNR / calibration failure / timing mismatch** 都可能造成"没看到"、这时 $E^-$ 与 $H$ 无关、是 sensor state 的问题。

正确的定义必须把 observability 与 sensor condition 挂进去、写成**observability-conditional likelihood ratio**：

$$\Lambda^-(H) \;=\; \log \frac{P\!\left(\mathcal{E}^- \mid H,\, \mathcal{O}\right)}{P\!\left(\mathcal{E}^- \mid \neg H,\, \mathcal{O}\right)},$$

其中 $\mathcal{O}$ 编码了 observability / sensor health / field of regard / calibration status。这才是"本应看到但没有"的正确刻画——**只有当 $\mathcal{O}$ 说该看到的时候、$E^-$ 才对 $H$ 有 likelihood 意义**。这也直接接上了 9/14 §6 里的 observability 字段：**negative evidence 是 likelihood-side evidence、不是 provenance**。

具体到 policy 侧：把 $\Lambda^-(H)$（或 $\exp(\Lambda^-)$ 的 logit）作为一个 field 拼进 $[\hat S]_{\mathcal{C}_\pi}$、让 policy 或 belief update 消费。这条 primitive 目前工程实现最薄、但它对 hypothesis-ranking 的影响往往最大。

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

#### 类别 B · Intervention-consistency constraint（核心）

**上一版把这一类笼统写成 $D(\pi_\theta(\hat S), \pi_\theta(T_{\mathcal{C}}(\hat S)); \rho_{\mathcal{C}})$、reviewer 立刻问 $\rho_{\mathcal{C}}$ 从哪来**——四种 $T_{\mathcal{C}}$ 各自的"响应规律"不一样、有一类甚至根本不该预设"必须响应"。这一版按响应强度把 intervention 分成三档。

**Type I · Exact invariance / equivariance**（最干净的一档）。变换在 policy input 上有一个合法的 action-side 对应 $T^{\mathcal{C}}_\pi$、要求：

$$\pi_\theta\!\big(T^{\mathcal{C}}(\hat S),\, o,\, \ell\big) \;=\; T^{\mathcal{C}}_\pi\!\big(\pi_\theta(\hat S,\, o,\, \ell)\big).$$

典型：**frame transform**——把 `reference_point` 从 A 移到 B、$\tau$ 按 transport theorem 变换、**要求 action 侧做对应的坐标变换**（$\pi(T_g S) = T_g^A \pi(S)$）。类似：**permutation of equivalent hypotheses**（同 posterior weight 的 hypothesis 互换、action 分布必须等价）。这一档可以写成硬 loss、$\mathcal{L}_{\mathrm{consistency}}^{\mathrm{I}} = \|\pi_\theta(T\hat S) - T^\pi \pi_\theta(\hat S)\|^2$。

**Type II · Monotone response**（次强的一档）。指定一个 scalar 或 partial-order 泛函 $M(\cdot)$（例如 action distribution 的 variance、conservative action norm、stop probability）、只要求：

$$M\!\big(\pi_\theta(T^{\mathcal{C}}\hat S)\big) \;\preceq\; M\!\big(\pi_\theta(\hat S)\big) \quad \text{（或反向的 } \succeq \text{）}.$$

典型：**age increase**（measurement 不变、age 从 5 ms 抬到 200 ms）——**要求 policy 输出 variance 单调上升**（或 action 更保守、或 fallback 概率单调上升）；**hypothesis reweighting**——只改 posterior weight、**要求对应 hypothesis 的 action 质量随 weight 单调响应**（top-1 不必翻转、但分布必须按 weight shift 方向变）。这一档写成 hinge / soft-ranking penalty、不指定完整 output。

**Type III · Unconstrained intervention**（最弱、也最重要的一档）。**不预设 policy 必须变化**——典型是 **provenance removal**：拿掉一个 contributing sensor、若另一个 sensor 完全冗余替代、**最优 action 可以完全不变**。这一档的正确提法是：**when the removed evidence was decision-relevant, does performance degrade?** 也就是把它交给 §6 的 CAG 面板去测、而不是训练时强加响应规律。写成 loss 就是：**不做任何 intervention consistency、只在 evaluation 阶段做 ablation**。这一档存在本身是对上一版的一个纠正——上一版把四种 $T_{\mathcal{C}}$ 一视同仁地塞进"要求响应"、是把 Type III 错当成了 Type II。

**这三类都不要求 policy 显式预测什么**；它们要求的是**响应函数符合 contract 语义、或者被允许符合"不变"**。这才是 §0.2 "$q_\pi$ 是显式声明的 quotient" 定义的**训练侧对应物**。相比"加几个 auxiliary 预测头"、这套三类约束更贴合本文 thesis、也更有研究味：**我们建议的不是让 policy 复制 contract、是让 policy 在 contract 变换下按对应的响应档位响应**。

### 5.2 Augmentation：observed metadata vs latent degradation class

9/14 §8.5 强调 **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption**——六种降级各有不同的**因果起源**与不同的**下游读法**。训练时的数据增强如果只用一种（最常见的是 random masking）就会让 policy 把六种降级**全学成同一种**、部署时不能区分"这个通道今天坏了"与"这个通道数据延迟大"。aug 生成器必须**按 chain 分通道**、每类独立分布、并且**每类对应到 §4.2 的七个 slot 中哪一个被激活**。

**上一版这里有个 reviewer 抓到的 shortcut 风险**："给每个 episode 打 degradation label、policy input 里显式携带"——如果 `degradation = stale_vision` 这个 label 直接作为 policy input、policy 会学成 $a = f(o, \text{label})$；但**真实部署里 label 本身未必可靠**、这个 shortcut 在训练里看起来无害、部署里就是灾难。这一版明确区分两样东西：

**Observed metadata** $m_c = (\text{age}, \text{validity}, \text{availability}, h_c, \ell_c)$——由 estimator 或 sensor driver 直接得到、可以进 contract、可以进 policy input。**这是 §4.2 七个 slot 的来源**。

**Latent degradation class** $d_c \in \{\text{missing}, \text{stale}, \text{bias}, \text{corrupt}, \ldots\}$——augmentation 时你**知道**注入了哪一类、但部署时这个类是**latent 的**、只能由 contract estimator 从 $m_c$ 序列里推断、或只作为训练 annotation 用来加权 loss / 采样、**不能默认作为 ground-truth input 塞进 policy**。

具体做法：aug pipeline 采样 $d_c$、按 $d_c$ 生成 $m_c$（例如 $d_c = \text{stale}$ ⇒ 抬 $a_c$、把 $\mu_c$ 保持在延迟时刻的真实值、$v_c = \text{true}$）、然后把 $m_c$ 作为 policy input、$d_c$ 只作为 loss weighting 与 evaluation 分层的 key。**这不是 curriculum、是 conditioning on observed metadata**——差别在于 conditioning 让 policy 看见的是可靠的 $m_c$、而不是不可靠的 $d_c$。§5.1 类别 B 的 Type II monotone response 与 degradation-conditioned aug 天然配对——**aug 端造 $T_{\mathcal{C}}$ 的 $m_c$ 变化、loss 端测 policy 对 $m_c$ 变化的响应是否符合 $\rho_{\mathcal{C}}$**。

### 5.3 Safety filter 与 contract 的接口：constraint-relevant validity

约束层（CBF / shield / runtime verifier）**必须读 $a_{\mathrm{proposed}}$**、否则它 filter 什么——这一点不用退让。但本文更精确的 claim 是：**safety filter 不应该把 policy 的 confidence 或 estimator 的通用 validity 位当成约束成立性的唯一证据；它应该直接访问 constraint-relevant evidence**。

**这里必须区分两种 validity**。Estimator 侧的通用 `validity` 位说的是"这条 measurement 从 calibration / sensor health 的角度看是否有效"——这是 field-level 的 general-purpose 声明。Safety 侧真正关心的是"**这条约束在这个时刻、这个 predicate 下是否 valid**"——这是 constraint-level 的语义。两者并不相同：`validity=true` 不意味着 constraint estimate 对**这条 safety predicate** 有效；例如"关节力矩读数 calibration OK" ≠ "当前接触估计足以支撑 collision constraint"。

本文建议 safety filter 读到的、是一个 constraint 特定的组合量：

$$v_j^{\mathrm{constraint}} \;=\; g\!\Big(\text{field validity},\; \text{observability},\; \text{hypothesis posterior},\; \text{age},\; \text{model coverage}_j\Big).$$

$g$ 是 constraint-specific 的组合规则（例如 CBF 那侧要求"距离估计在当前 hypothesis 下、observability 充分、且 dynamics model coverage 到当前状态区域"）、$v_j^{\mathrm{constraint}}$ 才是 safety filter 真正应该读的**constraint-relevant evidence**。9/14 §7 讲过 **track_id 是 hypothesis**——那么 safety filter 也不能只信 track_id 匹配、还要看 hypothesis posterior 是否稳定；两个 track 是否 merge / split、直接决定"这个障碍距离"的可信度。

这条也接上了 §4.3.3 的 negative evidence：**"本应看到但没有"** 会让 $v_j^{\mathrm{constraint}}$ 变 false——例如雷达在这个角度什么都没扫到、$\Lambda^-(H_{\text{clear}})$ 变正、collision constraint 的 $v^{\mathrm{constraint}}$ 应该保持 active、不能因为 policy 的 belief 乐观就 deactivate。

**这三样（constraint-specific validity、observability、negative evidence）不进 safety filter、filter 就会用 policy 的 belief 反推约束成立性**、这在低 observability 区域特别危险——policy 的 belief 之所以乐观、是因为它读不到 contract 里的 observability / validity / negative evidence、**filter 如果同样读不到、两者一起盲**。

### 5.4 与 9/10 Part 3 evaluation 的呼应

Sim utility 三维（prediction / ranking / decision）里、policy-side 的 evaluation 主要看 **decision** 这一维——但要加一个 **contract-preservation 维度**：把 contract 拆掉之后 policy 的表现下降多少、就是它对 contract 依赖度的**下界**。这一维度对应 §6 的 CAG 指标、且需要 oracle baseline 来界定解释边界。

## 6. Evaluation：一个四层面板

对应 §4 的 primitives 与 §5.1 的三类 intervention、本文提出**四层评估面板**——不是一堆并列 metric、而是各自回答不同问题。**CAG 只是 decision 层的聚合、它不衡量 semantic understanding；contract compliance 要靠四层一起看**。

| Level | 测试类型 | 回答什么 |
|---|---|---|
| **Semantic** | invariance / equivariance test (§5.1 Type I) | 语义变换后是否按规则响应 |
| **Representation** | probes (§5.1 A) + hypothesis coverage & separation | contract 信息是否还在 $z_\pi$ 里、hypothesis 是否被 preserve & distinguished |
| **Decision** | CAG + task-conditional utility (§5.1 Type III) | contract structure 是否真的改善决策、增益多少 |
| **Safety** | constraint intervention (§5.3 $v_j^{\mathrm{constraint}}$) | contract 降级时是否进入正确的 guardrail |

以下按层展开。Temporal 与 source 两条 invariant 的专项测试分别落在 Decision 层的 **SDS** 与 **$\Delta J_{\mathrm{prov}}$**——它们不是新指标、是 CAG 面板的两个 specialized 切片。

### 6.1 Semantic level：Invariance / Equivariance Test

对应 §5.1 Type I。给定一组已知 $T^{\mathcal{C}}_\pi$ 的 contract 变换（frame / coordinate / hypothesis permutation）、测：

$$\mathrm{Equiv}(\mathcal{C}) \;=\; \mathbb{E}_{\hat S}\!\left[d\!\left(\pi_\theta(T^{\mathcal{C}}\hat S),\; T^{\mathcal{C}}_\pi\,\pi_\theta(\hat S)\right)\right].$$

$\mathrm{Equiv} \to 0$ 是硬要求、$\mathrm{Equiv} \gg 0$ 意味着 $e_\pi$ 学坏了、或者 $q_\pi$ 直接把这一层 quotient 丢了。这一层是最"干净"的一层、因为规则是 mathematically defined 的、不需要 oracle 也不需要 $J$ 的定义。§3.2 Failure 2 的 severity 可以直接由 $\mathrm{Equiv}(\text{frame})$ 量化。

### 6.2 Representation level：Probes + HPC + HSS

**Probe 分数**（§5.1 类别 A）：从 $z_\pi$ 反向预测 contract 字段、报告 accuracy / calibration。这是诊断、不是核心指标。

**上一版的 HPS 用 $\max_k$ 有个明显漏洞**——若 $K = 8$、policy 只对 $H_1$ 反应正确、$\max_k$ 依然拉满、但 hypothesis structure 实际上完全没有 preserve。这一版把它拆成两个方向相反的指标：

**Hypothesis Coverage (HPC)**——衡量**每个 hypothesis 是否都被 support**：

$$\mathrm{HPC} \;=\; \frac{1}{K} \sum_{k=1}^{K} \Pr\!\big[\pi_\theta(x_t^\pi) \in \mathcal{A}^{*}_k \,\big|\, H_k \text{ is true}\big].$$

**Hypothesis Separation Score (HSS)**——衡量 policy 输出在不同 hypothesis 条件下**是否可区分**：

$$\mathrm{HSS} \;=\; \frac{1}{K(K-1)} \sum_{i \neq j} D\!\big(\pi_\theta(\cdot \mid H_i),\; \pi_\theta(\cdot \mid H_j)\big).$$

HPC 与 HSS 一起才对应 semantic-preservation thesis——**coverage** 保证"都能处理"、**separation** 保证"policy 保留了 hypothesis distinction"。$\mathcal{A}^{*}_k$ 与 §6.3 的 oracle 一样、**在 benchmark 里由 privileged simulator state / oracle planner / offline expert rollouts 构造、因此 HPC / HSS 是训练评测阶段指标、不是 deployment-time observable**——这一点上一版已经加了、保留。

### 6.3 Temporal slice：Staleness Response Compliance (SDS)

**上一版 SDS 定义成 $D(\pi(\cdot|\mathrm{do}(a=a_1), o), \pi(\cdot|\mathrm{do}(a=a_2), o))$ 并写"期望 variance 单调升"、这一版有两处收紧**。第一，SDS 不是标量、是一族量、reviewer 抓的"variance 单调升不是 universal law"完全对——真实 policy 完全可能是**分段的**：age < 50 ms 走视觉、age > 100 ms 切 proprio-only fallback、age > 200 ms 停止；response curve 是 piecewise discontinuous 的、不能预设单调。第二，与 §6.5 CAG 一样、SDS 需要 oracle baseline 来界定"变化对不对"。

本文把 SDS 重新定义为**相对 oracle 的响应保真度**：

$$R_\pi(a) \;=\; \pi_\theta\!\big(\cdot \,\big|\, \mathrm{do}(a_c = a),\, o\big),$$

$$\mathrm{SDS} \;=\; D\!\big(R_\pi(a),\; R^{*}(a)\big),$$

其中 $R^*(a)$ 是**oracle policy 在同一干预下的响应曲线**、$D$ 是 curve-level divergence（例如在 $a$ 上积分的 KL 或 sliced Wasserstein）。响应属性 $R$ 根据任务定义、可以取 **variance / action norm / fallback probability / safety margin / stop probability**——不是硬编码成 variance 单调升、而是**要求 policy 与 oracle 在同一属性上响应形状一致**。平坦不代表差、要对着 oracle 的期望响应曲线比。

导数形式仍保留、作为 response shape 的一种局部刻画：

$$\left.\frac{\partial\, \mathbb{E}\!\big[\pi_\theta(\cdot \mid \mathrm{do}(a_c = a),\, o)\big]}{\partial a_c}\right|_{a}\quad\text{与 } R^* \text{ 的同阶导数比较}.$$

### 6.4 Source slice：$\Delta J_{\mathrm{prov}}$ 与 $\Delta J_{\mathrm{neg}}$

**上一版的 $\Delta J_{\mathrm{prov}}$ 保留、但解释要跟着 §4.3 的调整同步改**——dependency-aware fusion 不是"suppress double-counting"、所以 $\Delta J_{\mathrm{prov}}$ 的语义也不是"policy 有没有 suppress 相关对"、而是"**policy 有没有把 `correlated_with` 当合法证据结构消费**"。$J$ 一律 higher-is-better：

$$\Delta J_{\mathrm{prov}} \;=\; J\!\big(\pi_\theta \mid \text{provenance + correlated\_with}\big) \;-\; J\!\big(\pi_\theta \mid \text{both} = \varnothing\big).$$

$\Delta J_{\mathrm{prov}} > 0$ = policy 真在用 source structure；$\Delta J_{\mathrm{prov}} \approx 0$ = 它只是把 source 当装饰；$\Delta J_{\mathrm{prov}} < 0$ = 加了反而更差、通常意味着 conditioning 与 backbone 归纳偏置冲突、要单独 debug。

**再补一条 $\Delta J_{\mathrm{neg}}$**——把 §4.3.3 的 $\Lambda^-(H)$ 从输入里去掉、看 hypothesis ranking accuracy 掉多少：

$$\Delta J_{\mathrm{neg}} \;=\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda^-\big) \;-\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda^- = \varnothing\big).$$

这一条对 negative_evidence primitive 是直接的、也回应了 §5.3 的 safety 侧接口。

### 6.5 Decision level：Contract Ablation Gap（CAG）+ Oracle baseline

**给定 contract 的一个特定 collapse 算子 $\mathrm{collapse}_X$**（把 $X$ 这一层 contract structure 无声明地折叠掉）、

$$\mathrm{CAG}_X \;=\; J_{\mathrm{full}} \;-\; J_{\mathrm{collapse}_X}, \qquad J_{\mathrm{full}} \equiv J(\pi_\theta \mid \hat S),\;\; J_{\mathrm{collapse}_X} \equiv J(\pi_\theta \mid \mathrm{collapse}_X(\hat S)),$$

四类 collapse 各对应一条 invariant：

- $\mathrm{collapse}_{\mathrm{hyp}}$：把 hypothesis set 折成单 Gaussian 或单点估计。
- $\mathrm{collapse}_{\mathrm{age}}$：把所有 channel 的 `age / health / latency` 抹平。
- $\mathrm{collapse}_{\mathrm{prov}}$：drop `contributing_mask` 与 `correlated_with`、让 policy 无区别地 attend。
- $\mathrm{collapse}_{\mathrm{neg}}$：drop $\Lambda^-(H)$。

**但 reviewer 抓得对——单看 $\mathrm{CAG}_X$ 无法区分"policy 没用 contract"与"contract 对当前任务不够 informative"、也无法排除 policy 把某字段当 shortcut**。这一版加入 **oracle privileged-state baseline**：

$$J_{\mathrm{oracle}} \;=\; J(\pi^{*}_{\mathrm{oracle}} \mid s^{\mathrm{priv}}), \qquad \mathrm{Gap}_{\mathrm{oracle}} \;=\; J_{\mathrm{oracle}} - J_{\mathrm{full}}.$$

于是四种诊断组合的解读是：

| CAG 与 oracle 的关系 | 解读 |
|---|---|
| CAG 高、oracle 略高于 full | policy 真的在读 contract、contract 接近充分 |
| CAG 高、oracle 明显高于 full | policy 在读、但 contract 本身漏了一些 decision-relevant 信息（回去看 9/14 schema 是否完整） |
| CAG ≈ 0、oracle ≈ full | contract 结构对当前任务没用、CAG ≈ 0 正常 |
| CAG ≈ 0、oracle 明显高于 full | **真正的红旗**——contract 里有信息、policy 没用 |

**这一版最要写清的一句是**：**CAG measures task-conditional utility of contract structure, not semantic understanding by itself**。高 CAG 有可能是 policy 把某个字段当 shortcut（例如 age 与任务难度在训练分布里强相关）、collapse 掉 age 后 performance 掉、但 semantic understanding 并没有发生。真正的 compliance 需要 **CAG + §5.1 Type I/II controlled response + §6.1 invariance/equivariance** 三件一起看。这一版把 CAG 从"总指标"降回"decision 层聚合"、四层面板才完整。

### 6.6 Safety level：Constraint intervention

在部署 / 半仿真环境里主动把 §5.3 的 $v_j^{\mathrm{constraint}}$ 打下来（例如注入 calibration drift、把 observability 关掉、把 $\Lambda^-(H)$ 拉高）、看 safety filter 是否**在正确的时刻进入正确的 guardrail**、以及 guardrail 触发是否可归因到 $v_j^{\mathrm{constraint}}$ 的哪一项。这一层是 §5.3 接口的直接对应、也是整篇 interface 主张真正**能不能落地**的测试。

四层合起来构成一个**干预型 evaluation 面板**：**Semantic 测"响应规则对不对"、Representation 测"信息还在不在"、Decision 测"用了没 / 有没有 shortcut"、Safety 测"降级时接住没"**。它们都**不能替代**任何端到端 success rate——它们衡量的是 policy 侧对 contract 的**读取度**、不是**表现力**。这一点与 §0.3 Claim 3 完全对齐：**contract compliance must be tested by controlled intervention**。

## 7. 最小可执行接口草图

把 §4 三族 primitives 与 §6 四层面板合起来写成一个 Python 类骨架。**不是要给出一个具体 policy、是要给一个可读的接口约定**。

```python
class StructuredStateView:
    def __init__(self, contract: StructuredState):
        self.contract = contract

    def project(
        self,
        schema: PolicySchema,
        # --- readout knobs ---------------------------------------
        mode: Literal["map", "posterior_sample", "topk"] = "topk",
        topk_calibration: Literal["renormalize", "keep_residual"] = "renormalize",
        # --- staleness / uncertainty knobs -----------------------
        staleness: Literal["ignore", "parallel_field", "condition"] = "parallel_field",
        uncertainty_inflation: Literal["none", "conservative", "propagate"] = "conservative",
        # --- source-structure knobs ------------------------------
        provenance: Literal["ignore", "harden"] = "harden",
        dependency: Literal["ignore", "attention_bias", "covariance_fusion", "learned"] = "covariance_fusion",
        negative_evidence: Literal["ignore", "condition", "belief_update"] = "condition",
    ) -> PolicyInput:
        """
        Project StructuredState (upstream, 9/14) to PolicyInput (downstream, this piece).
        `cfg` literally encodes q_pi — the declared quotient.
        e_pi is the encoder of the specific backbone and must never silently drop
        anything that q_pi said was preserved.
        """
        slots = {}
        for field_name in schema.fields:
            h = self.contract[field_name]   # hypothesis set: [(mu_i, Sigma_i, w_i)]

            # --- mode_select -------------------------------------
            if mode == "map":
                mu, Sigma, w_payload, residual = h.most_likely().mu, h.most_likely().Sigma, None, None
            elif mode == "posterior_sample":
                mu, Sigma, w_payload, residual = h.sample().mu, h.sample().Sigma, None, None
            else:  # "topk"
                top = h.top_k(k=schema.k_per_field[field_name])
                residual = 1.0 - top.total_weight()
                if topk_calibration == "renormalize":
                    w_payload = top.renormalize()          # sum w̃_i = 1 within top-k
                    residual_declared = residual           # explicitly recorded, not silently dropped
                else:  # "keep_residual"
                    w_payload = top.weights                 # raw weights kept
                    residual_declared = residual

            # --- age_gate: seven parallel slots, never multiplicative on mu ---
            age = h.age
            validity = h.validity_ok
            availability = h.available
            health = h.sensor_health                        # h_c
            latency_status = h.causal_status                # ℓ_c
            trust = clamp(
                tau_curve(age, validity, health, latency_status,
                          schema.tau_config[field_name]),
                min=schema.trust_eps,                       # numerical guard: τ ≥ ε
            )                                                # q_c is meta field, does NOT scale μ_c

            # uncertainty handling: propagation is the general form,
            # Σ/τ is only ONE conservative approximation.
            if uncertainty_inflation == "none":
                Sigma_eff = Sigma
            elif uncertainty_inflation == "propagate":
                Sigma_eff = propagate_uncertainty(
                    Sigma, dt=age, u=h.control_state, f=schema.dynamics_model
                )
            else:  # "conservative"
                Sigma_eff = inflate_uncertainty(Sigma, trust)   # e.g. Σ/clamp(τ, ε), guarded

            slots[field_name] = dict(
                mu=mu, Sigma=Sigma_eff, age=age,
                validity=validity, availability=availability,
                health=health, latency_status=latency_status,
                trust=trust, w_payload=w_payload,
                residual_declared=(residual if mode == "topk" else None),
            )

        # --- provenance / dependency / negative evidence, three reads ---
        prov_mask = self.contract.contributing_mask if provenance == "harden" else None
        if dependency == "attention_bias":
            dep_payload = compile_field_graph_to_token_graph(self.contract.correlated_with)
        elif dependency == "covariance_fusion":
            dep_payload = self.contract.correlated_with       # fed into Kalman / factor graph
        else:
            dep_payload = None
        lam_neg = self.contract.likelihood_ratio_minus if negative_evidence == "condition" else None

        return PolicyInput(slots=slots,
                           contributing_mask=prov_mask,
                           dependency_payload=dep_payload,
                           lambda_minus=lam_neg,
                           schema=schema)


class ContractAwarePolicy(nn.Module):
    def __init__(self, backbone, schema: PolicySchema, cfg: ContractReadConfig):
        self.backbone = backbone
        self.schema = schema
        self.cfg = cfg   # cfg literally encodes q_pi — the declared quotient

    def forward(self, state: StructuredState, obs, lang) -> ActionDistribution:
        x = StructuredStateView(state).project(self.schema, **self.cfg.as_kwargs())
        return self.backbone(x, obs, lang)
```

三条 caveat 明确写死：

- **(i)** 这不是唯一读法、**`cfg` 就是本文 §0.2 里的 declared quotient $q_\pi$**——同一个 contract、VLA 与 Diffusion Policy 的 `cfg` 就该不一样、engineered state head 与 visual latent head 的最优 `uncertainty_inflation` 也不同。接口规格里必须写着 $q_\pi$、不允许它藏在 encoder 权重里。
- **(ii)** 这个接口**只解决输入端**；§5.1 的三类 intervention 约束、§5.2 的 observed-metadata-only aug、§5.3 的 $v_j^{\mathrm{constraint}}$ 直连——如果一处不改、$q_\pi$ 写得再漂亮也会被训练动力绕过去（loss 会自己找到最省事的"把 contract 压扁"路径）。
- **(iii)** `dependency="attention_bias"` 只在 transformer-family backbone 上有实现路径、且**field-graph → token-graph 编译**（`compile_field_graph_to_token_graph`）本文只给了函数名、没给 canonical 实现；MLP head 走 `learned`、Kalman / factor-graph 融合走 `covariance_fusion`——**attention_bias 只是 dependency_aware_fusion 的众多实现之一、不是 primitive 本身**。

## 8. 三条收束 claim（与 §0.3 对齐）

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. The policy projection $\Pi_\pi = e_\pi \circ q_\pi$ 是一个语义接口、它决定哪些 contract distinctions 仍对下游决策可用。$q_\pi$ 是**允许丢什么的显式声明**、$e_\pi$ 是实际的 neural encoding——两者不分离、就没有可审计的接口。

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state head、latent visuomotor policy、autoregressive VLA、diffusion、flow policy 用的 conditioning 与 action generator 都不一样——但它们都必须回答**同一个 interface 问题**：哪些 contract distinctions 被显式保留、哪些被压缩、哪些被丢弃？攻击的对象是 interface contract、不是模型架构；π0 是 VLA + flow matching、Diffusion Policy 是 visual-latent + diffusion、ACT 是 visual-latent + generative sequence decoder——**用两个正交维度切、比用"三个家族"切更贴近事实、也更不容易被"某族天生好"的直觉误导**。

> **Claim 3 · Contract compliance should be tested by controlled intervention, not inferred from end-to-end success.** 端到端 success 衡量 policy 好不好用、不衡量它有没有把 contract 语义读对。contract compliance 需要 §6 的四层面板一起看——**Semantic（invariance / equivariance）+ Representation（HPC / HSS + probes）+ Decision（CAG 加 oracle baseline）+ Safety（constraint intervention）**——并且需要区分 §5.1 的三类响应档位（Type I exact equivariance / Type II monotone / Type III unconstrained）；把 Type III 错当 Type II 强加响应、是上一版的一个具体错误、这一版把它留给 ablation 去测。

一句收束：**"融合"这个词、以后尽量不用**——它在时间轴上问的是"什么时候合并"、在语义轴上问的是"合并成什么"；9/14 与本文合起来把第二个问题拆成了**上游交付什么 + 下游声明允许丢什么 + 编码保留什么**三半。剩下第一个问题（时机）已经被 §1 的两维对照回答得差不多——**时机是接口的结果、不是接口的决策变量**。

**最后一个可以往前走半步的命题**：本文真正立起来的东西、可以浓缩成一条 pipeline：

$$\boxed{\text{Contract} \;\longrightarrow\; \text{Declared Quotient } q_\pi \;\longrightarrow\; \text{Policy Representation } e_\pi \;\longrightarrow\; \text{Intervention Tests}}$$

也就是说：**policy 不只是 function approximator、它是一个 contract consumer**。这个 framing 允许下一篇直接立一个 **Contract-Preserving Policy Benchmark**——构造 $\mathcal{B}_{\mathcal{C}} = \{T_{\mathrm{frame}}, T_{\mathrm{hyp}}, T_{\mathrm{age}}, T_{\mathrm{validity}}, T_{\mathrm{prov}}, T_{\mathrm{neg}}\}$、让 SAC / PPO / Diffusion Policy / ACT / OpenVLA / π0 全过同一套 intervention suite——那时候这一系列文章就从"我认为 policy 应该读 contract"升格成"**给定同一份 Structured State Contract、如何系统地测不同 policy 是否 contract-compliant**"。这一句立住、这一系列就都值了。

## Sources

以下 arXiv ID 已联网核过；journal-only 引用不贴 arXiv。按支撑的 section 分组。

### A · VLA 家族（支撑 §1 grid、§3 Failure 1–2、§5.1 类别 B）

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)（**paper fact**：把 robot action 明确表达成 text token 与 VLM 联合 fine-tune · §3.2 Failure 2 的 tokenizer 侧典型形态、"contract flattening" 是本文分析、不是原论文的 limitation）
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)（**paper fact**：7B VLA、大规模机器人 demonstration 训练、强调 fine-tune 与 generalization；**本文分析**：其配置含多相机 / depth / proprioceptive state encoding、但"支持输入" ≠ "读到 contract 的哪一站"）
- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)（**paper fact**：预训练 VLM + proprio token + noisy action chunk + flow matching；**本文分析**：Under the Structured State Contract defined here, π0's conditioning interface does not expose an explicit slot for hypothesis / provenance / age / negative evidence——"Continuous actions do not imply structured state semantics" 是本文的分析、不是原论文的 self-limitation）
- Octo Model Team, *Octo: An Open-Source Generalist Robot Policy*, RSS 2024 · [arXiv:2405.12213](https://arxiv.org/abs/2405.12213)（transformer-based readout · §4.3.2 dependency_aware_fusion 中 attention_bias 路径的一个参照）

### B · Diffusion / Generative-Sequence / Flow-Matching Policy（支撑 §1 grid、§5.1）

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 · [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)（**paper fact**：RGB stack + proprio concat + conditional denoising diffusion、强调 action-distribution multimodality；**本文分析**：action-side multimodality ≠ state-side hypothesis preservation——这是本文 schema 下的推论、不是原论文承认的 limitation）
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ACT / ALOHA）, RSS 2023 · [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)（**paper fact**：CVAE + transformer encoder-decoder、核心是 **action chunking over sequences**——本文把 ACT 归入 "generative sequence decoder"、与 diffusion / flow matching 并列、不属于 diffusion family）
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 · [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)（**paper fact**：establish the flow-matching objective as vector-field regression for generative modeling / CNF——**flow matching 本身不是 robot action chunking 论文**；"continuous robot action chunks" 是 π0 这类工作的具体应用、本文把 citation chain 拆成 "Lipman establishes objective / π0 applies it to action chunks"）

### C · 不确定性、校准与 belief-space 参照（支撑 §4.1、§5.1、§6.5 CAG）

- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)（现代网络过度自信、temperature scaling 起点 · §4.1 calibration-aware top-$k$ 的 design-criterion 动机）
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels*（PlaNet / RSSM）, ICML 2019 · [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)（deterministic + stochastic latent · §4.2 predictive uncertainty propagation 的一个参照）
- Hafner et al., *Mastering Diverse Control Tasks through World Models*（DreamerV3）, Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)（离散 + 连续混合 latent、KL balancing · §4.1 posterior readout 的一条相邻路线）

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
