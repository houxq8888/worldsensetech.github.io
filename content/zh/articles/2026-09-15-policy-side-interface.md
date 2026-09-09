---
title: 'Contract 立起来之后：VLA、Diffusion Policy 与 π0 到底吃什么？'
slug: "2026-09-15-policy-side-interface"
date: 2026-09-15
draft: false
categories: ["具身智能", "策略学习"]
tags: ["具身智能", "策略学习", "VLA", "Diffusion Policy", "π0", "RT-2", "OpenVLA", "Action Tokenization", "Structured State Contract", "Contract-read Primitives", "Belief State", "Multimodal Hypothesis", "Provenance", "Uncertainty Calibration", "Safety Filter", "Hypothesis Preservation", "Staleness Discrimination", "Evaluation Metrics"]
description: '《多模态融合接口》那一篇把上游交付物立成了 Structured State Contract——本文问它的对偶：如果 estimator 真的按 contract 交付、policy 侧到底能不能吃到。答案不能乐观——VLA / Diffusion Policy / engineered-state head 三个主流家族各自在输入边界上悄悄压扁 contract 的不同切片，破坏发生得无声无息、并且不会因为 backbone 换大而自愈。本文提出 policy-side interface 的三条 contract-read primitives——mode_select 处理多 hypothesis 的读出、age_gate 处理异质时滞的信任衰减、provenance_condition 处理相关证据的双重计费——并把 loss、augmentation、safety filter 三处训练-部署连锁一并接上。评估侧给出 HPS / SDS / PCE 三个接口层指标。收在三条 claim 上：A good policy must read disagreement, not merely react to it；action space is not just kinematic, it is semantic；contract compliance is measurable at the policy boundary, not only at the estimator boundary。'
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

短答：**大多数主流 policy 架构会在输入边界上悄悄把 contract 压扁**。VLA 用一层 tokenizer、Diffusion Policy 用 image encoder + proprio concat、classical head 用手工 state vector——三种 projection 各自破坏 contract 的不同切片、并且**破坏发生得无声无息**：loss 曲线照样往下走、eval 分数照样往上涨、你从训练日志里看不出被压掉了什么。这不是模型规模的问题、**是接口的问题**。

这一篇聊的是"上游交付物到 policy 输入端之间的这一段路"。不打算推荐某个具体 backbone、也不打算反对 end-to-end learning。本文真正反对的是——**把 policy 侧接口的选择、当成一个"参数量与训练数据量"的问题**。文章会先把 contract 从 policy 一侧看一遍、然后按三个主流家族各自的 projection loss 拆开、再给出 policy 端真正需要显式写下来的**三个 contract-read primitives**（mode_select / age_gate / provenance_condition）、以及它们对训练目标、augmentation、safety filter、evaluation 的连锁改动。最后收在三条 claim 上。

## 0. 全文框架：contract 从下游看是什么样

先把整篇的分析框架摆在开头、后面每一节都会回到这张图。

```
              上游（9/14 已定义）                        下游（本文）
    ┌────────────────────────────────┐         ┌───────────────────────────┐
    │  Structured State Contract     │         │  Policy 家族              │
    │  ─────────────────────────────  │         │  ─────────────────────    │
    │  hypothesis + posterior w       │  ──Π──▶ │  A. engineered-state head │
    │  availability / validity / age  │  ──Π──▶ │  B. diffusion / flow      │
    │  observability / frame          │  ──Π──▶ │  C. VLA                   │
    │  contact set / negative evidence│         │                           │
    │  provenance / correlated_with   │         │  三种 Π 各自破坏哪一片    │
    └────────────────────────────────┘         └───────────────────────────┘
                    │                                          ▲
                    │   contract-read primitives               │
                    │   ───────────────────────                │
                    └──▶ mode_select ─ age_gate ───────────────┘
                              └──▶ provenance_condition ───────┘
```

三条本文的 boxed claim：

> **Claim 1（Contract can die at the input boundary）**——Structured State Contract 的语义可以在 policy input 的一层 projection 里被静默摧毁；"模型多大、数据多少"救不回来、只有显式的 contract-read primitives 才救得回来。

> **Claim 2（Three policy families, three projection losses）**——VLA / Diffusion Policy / engineered-state head 各自破坏 contract 的不同切片；三种家族之间真正的差别、不是参数量与模态覆盖度、而是**对 state schema 承诺了什么**。

> **Claim 3（Policy needs primitives, not bigger backbones）**——policy 侧真正需要的、是在 input 边界上把 mode_select、age_gate、provenance_condition 三个 primitives 显式写下来；不是把 backbone 从 3B 换到 7B。

一句术语约定：以下"policy 输入"一律写 $x_t^{\pi}$、它是 contract $\hat S_t$ 与观测 $o_t$、语言 goal $\ell_t$ 的一个投影 $x_t^{\pi} = \Pi_{\pi}(\hat S_t, o_t, \ell_t)$。**Contract-preserving 与否、关键全在 $\Pi_{\pi}$ 保留了 $\hat S_t$ 的哪些切片**。

## 1. 三种 policy 家族、各自吃什么

**家族 A：Engineered-State Policy Head**——MLP / GRU / 小型 transformer、读手工 state vector。它的祖先是 classical RL（PPO / SAC 在 MuJoCo / Isaac Lab 上的那一脉）、今天依然是 sim-to-real 与 industrial robot learning 的默认架构。输入约定：一个定长 $s_t \in \mathbb{R}^d$、字段是人工写死的关节角、末端位姿、力 / 扭矩读数、可选 contact flag。它对 contract **最友好**——因为字段本来就是显式的、你想往里加 `availability` / `validity` / `age` / `observability` 完全做得出来。代价是表达力天花板低、跨任务复用差、字段设计本身是一种艺术。

**家族 B：Diffusion Policy 与 Flow-Matching Policy**——把 policy 变成一个 denoising / flow-matching 的 generative model、读 image latent + proprio + 语言 goal（有时不读语言）。Diffusion Policy（Chi et al. 2023）读 $k$ 帧 RGB stack + 关节 proprio、输出一个 action chunk；ACT（Zhao et al. 2023）与后续 ALOHA 一脉走 Transformer encoder-decoder + CVAE 或 flow matching。它对**连续多模态 action distribution** 特别强、这是它的核心卖点——但 contract 处理是**隐式**的：state 被 image encoder 压进 latent、proprio 被 concat 进 noise-conditioning、**没有专门的 slot 承载 hypothesis、provenance、age**。contract 一旦进入它的输入边界、就被拉平成一条向量。

**家族 C：VLA (Vision-Language-Action)**——RT-2 / OpenVLA / π0 / Octo 一脉。读 image tokens + language tokens、输出 action tokens。RT-2 与 OpenVLA 是 autoregressive over discretized action tokens（把 action 变成一个 256-way 或类似规模词表里的 token）；π0 是 flow-matching over continuous action chunks、用 VLM backbone 作 conditioning。VLA 对 contract 的破坏最彻底：**language conditioning 与 image tokens 一起塞进 transformer、action 侧要么是离散化后的词表 token、要么是 flow-matching 出来的连续 action 段**。上游 contract 里的 `wrench.frame.reference_point` 之类的结构化字段、要么在 tokenizer 之前就得被压成自然语言（"low grip force, contact on pad"）、要么直接丢弃。

三种家族并排看：

| 家族 | State 表示 | Action 表示 | Contract 破坏面 | 表达力上限 | 主要成本 |
|---|---|---|---|---|---|
| A · Engineered-State Head | 显式 $s_t \in \mathbb{R}^d$ | 连续 mean + variance（或 SAC Q-head） | 最小、字段人工写 | 低 | 手工设计 state、跨任务差 |
| B · Diffusion / Flow | image latent + proprio concat | 连续 action chunk（denoised / flow） | 破坏 provenance / age / hypothesis | 中-高 | 推理多步、训练不稳 |
| C · VLA | 离散 image / language token | 离散 token 或 flow-matching action | 破坏 frame / reference_point / observability | 高（跨任务复用） | 训练贵、tokenize 不可逆 |

"破坏面"这一列是关键——**它决定 contract 里哪一部分字段在 policy 里还有意义**。9/14 §6.1 里定义的那些字段（`availability`、`validity`、`age`、`provenance.negative_evidence`、`wrench.count_uncertainty`、`contact.geometry.observability`）在家族 A 里可以原样搬、家族 B 里只能挑几个 concat、家族 C 里绝大部分要么写进 prompt、要么放弃。

一句 caveat：**这不是"哪个家族更好"的排序**。家族 A 在低维控制任务上仍然是 baseline 之王、家族 C 在开放语义条件下无可替代——问题不在于选谁、**问题在于选完之后、policy 有没有一条明确的接口约定、告诉你 contract 会被读到哪一步**。

## 2. "State" 在不同 policy 里意味着什么

这一节是本文的**术语清障**。"state"这个词在具身智能文献里至少有五种互不相同的意思、policy 家族的选择往往就是"它默认哪一种"的选择：

**$\pi_{\mathrm{obs}}$：raw observation**——图像、点云、力 / 扭矩读数的原始流。绝大多数 imitation learning 训练管线的**输入形态**、但一般不是 policy 内部**实际使用的 state**（会被 encode 掉）。

**$z_t = f_\phi(o_{:t})$：encoded latent**——encoder 之后的低维表示。Diffusion Policy、VLA 的 image tokens、world model 的 RSSM state 都属于这一类。contract 到这里已经**经过一次 projection**、observability 与 provenance 通常已经丢了。

**$b_t$：belief / posterior**——POMDP 一脉的显式 belief state。contract 里"多 hypothesis + posterior weight"结构最贴合这个。但主流 VLA / Diffusion Policy 都不显式建模 belief——它被 encoder 隐式近似。

**$s_t = (b_t, \pi_t, q_t, h_t)$：9/10 Part 1 定义的 allocation state**——belief + policy + budget + hardware。这是**决策层**的 state、不是 policy 的输入 state。它要求 contract 里除了观测、还要能读到"当前哪条 policy、还剩多少 real-rollout 预算"——这些字段在 VLA 里目前完全没接口。

**$\hat S_t$：structured state (contract)**——9/14 定义的那个契约对象。

一条完整的链是这样的：

$$\pi_{\mathrm{obs}} \;\xrightarrow{\;\text{encoder}\;}\; z_t \;\xrightarrow{\;\text{abstraction}\;}\; \hat S_t \;\xrightarrow{\;\text{posterior}\;}\; b_t \;\xrightarrow{\;\text{allocation}\;}\; s_t$$

Policy 侧真正的问题、**不是"我能不能吃下更长的 token 序列"**、而是"我在 $\pi_{\mathrm{obs}} \to z_t \to \hat S_t \to b_t \to s_t$ 这条链上、愿意承诺走到哪一步"。家族 A 停在 $\hat S_t$（显式结构化 state、字段人写）、家族 B 停在 $z_t$（encoder latent）、家族 C 事实上停在 $\pi_{\mathrm{obs}}$ 之后的一层 tokenizer（image tokens 只是压缩后的观测、没做任何 state abstraction）。

**"多模态融合"这个词在 policy 侧的常见误用**就是把 $\pi_{\mathrm{obs}} \to z_t$ 的这一步 cross-attention、当成"已经在做多模态状态估计"——它不是。真正的 state abstraction 要求 $\hat S_t$ 里的字段**跨传感器家族保持一致的语义、可被 controller / policy / world model / diagnostics 四种消费者共同读**——9/14 §6 已经把这个约定立起来了、本节要做的是**从 policy 一侧再问一遍：三种家族愿意读到链上的哪一站**。

## 3. Interface mismatch 的四种失败模式

一旦 §1 的三种 projection 落到具体 policy 里、contract 的语义会以四种**具体的失败模式**表现出来。这四条不是理论担忧、是**部署里真会翻车的东西**。

**Failure 1：Mode Collapse via Averaging**。contract 里同一物理量可能有多个 hypothesis（比如"这个 track_id 是不是同一个物体"、"这个 contact 是 pad 还是 edge"）带不同 posterior weight；policy 若在 latent 上以均值方式合并、就把 multimodal posterior 拍平成了 unimodal。表现为：policy 在"两种可能的世界"上给出一个"平均"的动作、两边都不是最优、但 loss 曲线看起来很好——**因为训练分布里"平均动作"往往就是数据里最常见的动作**。这不是参数量问题、**是 $\Pi_\pi$ 的形态问题**。只要 projection 是 L2 / mean、mode collapse 就是默认结局。

**Failure 2：Semantic Drift via Tokenization**。contract 里的 frame 字段（比如 `orientation_frame: "tool_flange"`、`reference_point: "contact_center"`、`convention: "right-handed"`）如果被 tokenizer 变成 token 序列、语义会漂移——模型可以学到 "contact_center" 这个 token 的**分布**、但学不到"reference_point 变更意味着 $\tau$ 必须做 $\tau_{p_2} = \tau_{p_1} + (p_1 - p_2) \times f$ 修正"这种硬约束。结果是 policy 在 frame 换掉的边缘 case 上错得莫名其妙：**token 分布可能几乎不变、但 wrench 值必须重新计算**。Attention 学到的相似度、捕获不了这类物理级 convention 变更。**一个具体形态**：真机部署里把 wrench 的 `reference_point` 从 `sensor_flange` 改成 `contact_center`——数据前处理管线换了、token 序列几乎一模一样（`"sensor_flange"` 与 `"contact_center"` 都作为 frame token 出现在相似语境里），但 $\tau$ 数值按 transport theorem 变了一大块。VLA 在这两个"看起来同分布"的 dataset 上分别训练、会**分别收敛到相似但都错**的 policy——因为它学到的是 frame token 的语言学共现、不是 frame 变更对物理量的**函数式**修正。这类 bug 只在**部署后**、cross-dataset 复用时才暴露。

**Failure 3：Temporal Alignment Broken by Concat**。contract 里不同 channel 的 `age` 是显式的（例如 proprio 1 ms、F/T 5 ms、vision 100 ms、tactile 30 ms）。若 policy 只看到 concat 后的向量、$\Delta t$ 分布就丢了。表现为 policy 用 vision 的"100 ms 前的世界"与 tactile 的"30 ms 前的接触"作决策、以为它们同步；其实 9/14 §4 已经强调过 **timestamp sync ≠ causal sync**、更不等于 decision-time causal consistency。**concat 掩盖的是 $\Delta t$、不是 $\Delta t$ 的语义**——即使你给每个通道都打上了时间戳、若 policy 没有把时间戳读进决策变量、时间戳就是装饰。

**Failure 4：Safety-Blindness to Validity vs Staleness**。contract 明确区分 availability（这个通道今天有没有数据）、validity（这个数据是否有效、例如 calibration 是否过期）、age（有多旧）。policy 若只看数值、就把"stale 但 valid"与"missing 但 valid"混成一类、把"calibration drift 后无效"与"传感器掉线"当成同一种降级。这在 9/14 §8.5 的 degradation chain 里已经拆开了——**masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption**——policy 若在 input 端不接这个 chain、训练时的 augmentation 与推理时的 guardrail 都会挂错地方。

四条 failure 都是**接口问题**、不是模型问题——换 backbone 不解决、只有 $\Pi_\pi$ 层显式化才解决。这一点非常重要、因为它把"要不要换更大 VLA"这个决策与"policy 侧接口要不要重写"这个决策**解耦**了。

## 4. Policy 需要的三种 contract-read primitives

顺着 §3 的四条 failure、policy 侧真正需要的是**三个原语**——不是更大的 transformer、不是更多的数据。

### 4.1 `mode_select`（假设层的读出方式）

面对 contract 里的多 hypothesis posterior、policy 需要在**读出**时明确选一种：MAP（选最高权重 hypothesis）、sample（按 posterior 采样、鼓励 exploration）、expected-mixture（保留 mixture、下游 head 自己 attend）、或 ECE-preserving top-$k$（保留校准的 $k$ 个 hypothesis）。MAP 与 expected-mixture 的差别就是 Failure 1 会不会发生的差别。

$$\text{read}\!\big(\{(\mu_i, \Sigma_i, w_i)\}_{i=1}^{K}\big) \;=\; \left\{\begin{aligned}
&\mu_{\arg\max_i w_i} && \text{(MAP、丢弃低权 hypothesis)}\\
&\textstyle\sum_i w_i\, \mu_i && \text{(mean collapse、默认危险区)}\\
&\mu_i + L_i \epsilon,\;\; i \sim w,\;\; \epsilon \sim \mathcal{N}(0, I) && \text{(posterior sample)}\\
&\big\{(\mu_i, \Sigma_i, w_i)\big\}_{i \in \mathrm{top}\text{-}k} && \text{(ECE-preserving top-}k\text{)}
\end{aligned}\right.$$

关键：**"mean collapse" 是多数 policy 的默认行为、因为它对应最简单的 concat + MLP 处理**。contract-preserving 要求你在 $\Pi_\pi$ 里显式写"这里不许 mean"、否则你精心维护的 multimodal posterior 会在一层 mean-pooling 里彻底蒸发。家族 B、家族 C 尤其危险——transformer 的 mean-pooling 或 class-token readout 天然是 collapse 器。

### 4.2 `age_gate`（时滞的信任衰减）

channel $c$ 的 `age` $a_c$ 进入 policy 时、应该显式衰减它对当前决策的贡献。最简形式是乘性 trust $\tau_c = \exp(-a_c / \bar a_c)$（$\bar a_c$ 是这个 channel 的时间常数）、更严格的是把 $a_c$ 作为条件变量喂给 policy 让它自己学衰减。前者是**接口层的物理约束**、后者是**训练目标**。9/10 Part 1 已经区分过 $\Delta_{\mathrm{queue}}$ 与 $\Delta_{\mathrm{processing}}$——$a_c$ 就是 $\Delta_{\mathrm{queue}}$ 的显式化。

$$x_c^{\pi} \;=\; \underbrace{\tau_c(a_c)}_{\text{staleness trust decay}} \;\cdot\; \mu_c \;\cdot\; \mathbb{1}\!\big[\text{validity}_c = \mathrm{OK}\big] \;\cdot\; \mathbb{1}\!\big[\text{availability}_c = \mathrm{OK}\big]$$

三个 $\mathbb{1}$ 相乘、正是把 9/14 §6.1 里 `availability` / `validity` / `age` 三条正交轴**原样搬进 policy 输入端**。少了任何一个、都会踩到 Failure 4。$\tau_c$ 的具体形式（指数衰减 / sigmoid / step）不重要、**重要的是它显式存在**——否则 channel 的时滞只活在数据集 metadata 里、policy 看不见。

### 4.3 `provenance_condition`（把来源当 conditioning、不当噪声）

contract 里 `provenance.contributing_mask`（哪些 sensor 贡献了这个字段）、`negative_evidence`（哪些 sensor 期望看到什么、但没看到）、`correlated_with`（哪些字段之间有已知相关性、不能双重计费）。这三条在 policy 输入端应该**作为 conditioning 变量**进入——不是作为噪声模型、也不是作为 dropout。

两种做法：

- **Hard conditioning**：把 `contributing_mask` 直接拼进 policy input。家族 A / B 都可以（多几维向量而已）。
- **Attention biasing**：把 `correlated_with` 转成 attention mask / bias、让 transformer 在 attend 时不重复计算相关证据。家族 C（VLA）更适合、因为 transformer 本来就有 attention 接口。

$$\mathrm{Attn}'_{ij} \;=\; \mathrm{Attn}_{ij} \;-\; \beta \cdot \mathbb{1}\!\big[\text{fields}_i \text{ correlated\_with } \text{fields}_j\big]$$

这一项减的不是"注意力权重"、是"重复计费"——正好呼应 9/14 §6.2 的 caveat (ii)：**proprio-derived $\hat F_{\mathrm{ext}}$ 与 F/T-measured wrench 是同源相关的、朴素融合会 double-count**。同理、如果 policy 同时看到 proprio 与 F/T 又看到 $\hat F_{\mathrm{ext}}$、三路 concat 之后 attention 会**以为这是三份独立证据**、事实上只有两份。**一个具体形态**：$\hat F_{\mathrm{ext}} = \arg\min_F \|\tau_{\mathrm{res}} - J^{\top} F\|_W^2 + \lambda R(F)$ 是从关节力矩残差解出来的**外部接触力估计**、$\tau_{\mathrm{res}}$ 里已经用了 proprio、F/T 又用了一次关节信息——三者 $correlated\_with$ 关系是**结构性的**、不是经验相关。若不做 §4.3 的 attention biasing、policy 会把这**同一份信息数三遍**、在低摩擦 / 高 contact 场景里显著高估 confidence、直接触发 §5.3 safety filter 的漏检。**这条 correlated_with 不是学出来的、是从 contract 里读出来的**——这就是 provenance 作为 conditioning、不作为噪声的核心含义。

### 4.4 三个 primitive 之间的关系

**`mode_select` 处理空间上的多 hypothesis**、**`age_gate` 处理时间上的异质 $\Delta t$**、**`provenance_condition` 处理因果上的相关性**。三者缺一、§3 的四条 failure 至少复发两条。三者的组合也**不是"接口设计题的完整答案"**——observability / identifiability、frame convention、contact set 这三类 contract 字段还有各自更专门的读法（9/14 §7、§8.6 有对应讨论）、本文只处理**最容易被 policy 静默破坏的这三条**。

## 5. Training-time 与 Deployment-time 的连锁后果

一旦 policy 输入端接了三个 primitives、训练目标、augmentation、safety filter、evaluation 四处都要跟着改。**接口不是免费的**——但改动是**局部的、可控的**。

### 5.1 Loss 挂哪里

VLA 的 autoregressive CE、Diffusion Policy 的 score matching、Flow Matching 的 velocity regression、classical head 的 Gaussian NLL / SAC Q-target——**这些 loss 都只覆盖 action 生成**、不覆盖 contract 侧的字段。contract-preserving 训练需要加**auxiliary loss**：让 policy 的中间表示预测 `age`、`validity`、`observability`、hypothesis posterior——**不是为了让 policy 更"懂"contract、是为了让它不能通过把 contract 压扁来 shortcut 到 loss 最小**。

$$\mathcal{L}_{\mathrm{total}} \;=\; \mathcal{L}_{\mathrm{action}} \;+\; \alpha_1\, \mathcal{L}_{\mathrm{calibration}} \;+\; \alpha_2\, \mathcal{L}_{\mathrm{hypothesis}} \;+\; \alpha_3\, \mathcal{L}_{\mathrm{validity\text{-}pred}}$$

三条 auxiliary 各自的动机：$\mathcal{L}_{\mathrm{calibration}}$ 逼 policy 的 action-distribution 与自己宣称的 uncertainty 一致；$\mathcal{L}_{\mathrm{hypothesis}}$ 逼它 top-$k$ 覆盖数据里的真实 hypothesis（例如 contact mode、object identity）；$\mathcal{L}_{\mathrm{validity\text{-}pred}}$ 逼它读 $\mathbb{1}[\text{validity}_c]$ 而不是靠数值分布"猜"。α 权重是 tuning 决策、**但结构是接口决策**——不能因为 α 难调就把 auxiliary 全去掉。

### 5.2 Augmentation 必须按 degradation chain 分类生成

9/14 §8.5 强调 **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption**——六种降级各有不同的**因果起源**与不同的**下游读法**。训练时的数据增强如果只用一种（最常见的是 random masking）就会让 policy 把六种降级**全学成同一种**、部署时不能区分"这个通道今天坏了"与"这个通道数据延迟大"。aug 生成器必须**按 chain 分通道**、每类独立分布、并且**每类对应到 §4.2 的 $\mathbb{1}$ 与 $\tau_c$ 的具体形态**。

具体做法：给每个训练 episode 打一个 degradation label、policy input 里显式携带、aug pipeline 用同一个 label 采样。**这不是 curriculum、是 conditioning**——差别在于 curriculum 是"先学简单后学难"、conditioning 是"让 policy 知道现在处在哪种降级"。

### 5.3 Safety filter 与 contract 的接口

约束层（CBF / shield / runtime verifier）应该读到 contract 的 `observability`、不是 policy 的 $\mu$。原因：约束的成立域取决于**可观性**、不取决于 policy 相信什么。9/14 §7 讲过 **track_id 是 hypothesis**——那么 safety filter 也不能只信 track_id 匹配、还要看 hypothesis posterior 是否稳定；两个 track 是否 merge / split、直接决定"这个障碍距离"的可信度。

具体地、safety filter 应该读到：`observability`（当前约束所依赖的量此刻**可不可观**）、`validity`（calibration 是否过期、过期的读数不能触发 constraint）、`negative_evidence`（本应看到但没有的观测、比如"雷达在这个角度什么都没扫到"）。**这三条不进 safety filter、filter 就会用 stale 或 invalid 数据下判断**——比 policy 更危险。

### 5.4 与 9/10 Part 3 evaluation 的呼应

Sim utility 三维（prediction / ranking / decision）里、policy-side 的 evaluation 主要看 **decision** 这一维——但要加一个 **contract-preservation 维度**：把 contract 拆掉（比如把所有 hypothesis 合成均值、把 `age` 抹平）之后 policy 的表现下降多少、就是它对 contract 依赖度的**下界**。这一维度对应 §6 的三个指标。

## 6. Evaluation：policy-side 的三个接口层指标

对应 §4 的三个 primitives：

**Metric 1（Hypothesis Preservation Score, HPS）**——同一时刻 contract 里有 $K$ 个 hypothesis、policy 输出在 $K$ 个 hypothesis 分别成立的世界里、能覆盖多少个"局部最优 action 簇"。

$$\mathrm{HPS} \;=\; \frac{1}{N} \sum_{n=1}^{N}\, \max_{k}\, \Pr\!\big[\pi_\theta(x_t^{\pi}) \in \mathcal{A}^{*}_k \,\big|\, H_k \text{ is true at } n\big]$$

$\mathcal{A}^{*}_k$ 是 hypothesis $H_k$ 下的最优 action 集合。MAP 读出与 posterior-sample 读出的 HPS 差距、就是 Failure 1 的严重度量化。

**Metric 2（Staleness Discrimination Score, SDS）**——构造两批测试集：A 组所有 channel 都 fresh、B 组故意让某几个 channel staleness 显著。若 policy 的输出分布对 A、B 几乎相同、SDS ≈ 0——说明它根本没在读 `age`、Failure 3 + 4 直接坐实。SDS 用 KL 散度量化：

$$\mathrm{SDS}_c \;=\; D_{\mathrm{KL}}\!\Big(\pi_\theta(\cdot \mid o, a_c^{\mathrm{fresh}}) \;\Big\|\; \pi_\theta(\cdot \mid o, a_c^{\mathrm{stale}})\Big)$$

$\mathrm{SDS}_c$ 接近 0 意味着"channel $c$ 的 staleness 对 policy 输出没有影响"——这是接口层最直接的 ablation。

**Metric 3（Provenance-Conditioning Effect, PCE）**——把 `correlated_with` 从 policy 输入里去掉、看它在 double-counting-sensitive 场景（比如 proprio + F/T 融合 $\hat F_{\mathrm{ext}}$）上的表现下降多少。

$$\mathrm{PCE} \;=\; J\big(\pi_\theta \mid \text{provenance}\big) \;-\; J\big(\pi_\theta \mid \text{provenance} = \varnothing\big)$$

PCE 高 = policy 真的在用 provenance、PCE 低 = 它只是把 provenance 当装饰。PCE 也告诉你 $\beta$（§4.3 里的 attention bias 强度）是否调对了。

三个指标的共同点：**都不是 accuracy、都是接口层的 ablation**——直接呼应 9/10 Part 3 §5 里"评估要沿三条正交轴拆"的做法。它们也**不能替代**任何端到端 success rate——它们衡量的是 policy 侧对 contract 的**读取度**、不是**表现力**。

## 7. 最小可执行接口草图

把 §4 的三个 primitives、写成一个 Python 类的骨架。**不是要给出一个具体 policy、是要给一个可读的接口约定**。

```python
class StructuredStateView:
    def __init__(self, contract: StructuredState):
        self.contract = contract

    def project(
        self,
        schema: PolicySchema,
        mode: Literal["map", "sample", "ece_preserving"] = "ece_preserving",
        staleness: Literal["ignore", "decay", "condition"] = "decay",
        provenance: Literal["ignore", "hard", "attention_bias"] = "attention_bias",
    ) -> PolicyInput:
        """
        Project StructuredState (upstream, 9/14) to PolicyInput (downstream, this piece).
        `mode / staleness / provenance` 是 policy-specific knob、不是全局默认。
        同一个 contract、VLA 与 diffusion policy 的 knob 值就该不一样。
        """
        out = {}
        for field_name in schema.fields:
            h = self.contract[field_name]                 # hypothesis set: [(mu_i, Sigma_i, w_i)]
            # --- Primitive 1: mode_select ------------------------------
            if mode == "map":
                v = h.most_likely().mu
            elif mode == "sample":
                v = h.sample().mu
            elif mode == "ece_preserving":
                v = h.top_k_with_weights(k=schema.k_per_field[field_name])
            # --- Primitive 2: age_gate ---------------------------------
            if staleness == "decay":
                v = v * exp(-h.age / schema.tau_bar[field_name])
            elif staleness == "condition":
                v = concat(v, onehot_bucket(h.age, schema.age_buckets))
            # validity / availability 的指示位、无论 staleness 选什么都必须挂上
            v = v * h.validity_ok * h.available
            # --- Primitive 3: provenance_condition ---------------------
            if provenance == "hard":
                v = concat(v, h.contributing_mask, h.negative_evidence_summary)
            # attention_bias 分支不在这里、由 backbone 在 attention 层读
            # self.contract.correlated_with 并注入 bias mask
            out[field_name] = v
        attn_bias = self.contract.correlated_with if provenance == "attention_bias" else None
        return PolicyInput(features=out, attention_bias=attn_bias,
                           schema=schema, provenance=attn_bias is not None)


class ContractAwarePolicy(nn.Module):
    def __init__(self, backbone, schema: PolicySchema, cfg: ContractReadConfig):
        self.backbone = backbone
        self.schema = schema
        self.cfg = cfg

    def forward(self, state: StructuredState, obs, lang) -> ActionDistribution:
        x = StructuredStateView(state).project(
            self.schema,
            mode=self.cfg.mode,
            staleness=self.cfg.staleness,
            provenance=self.cfg.provenance,
        )
        return self.backbone(x, obs, lang)
```

三条 caveat 明确写死：

- **(i)** 这不是唯一读法、`mode / staleness / provenance` 三个 knob 都是 policy-specific——同一个 contract、VLA 与 Diffusion Policy 的 knob 值就该不一样、家族 A 与家族 B 的最优 `staleness` 也不同。
- **(ii)** 这个接口**只解决输入端**；§5 的 loss、augmentation、safety filter 三处连锁如果不改、$\Pi_\pi$ 层写得再漂亮也会被训练动力绕过去（loss 会自己找到最省事的"把 contract 压扁"路径）。
- **(iii)** `provenance="attention_bias"` 只在 transformer-family backbone 上有实现路径；对 MLP head 走 `hard` 那条。

## 8. 三条收束 claim

> **Claim A**：**A good policy must read disagreement, not merely react to it.** —— 9/14 说好的多模态系统要**表征 disagreement**；本文补一句：好的 policy 要在**读出**层面**尊重这个表征**、不能通过 projection 悄悄把 disagreement 平均掉。read 与 react 的区别就是 §4.1 里 mean collapse 与 ECE-preserving top-$k$ 的区别。

> **Claim B**：**Action space is not just kinematic — it is semantic.** —— action 层的 frame / reference_point / convention 是物理量、不是元数据；tokenizer 抹不掉它的语义、只能把它变成一个"分布相似但值不对"的静默 bug。这是 §3 Failure 2 的收束表述。

> **Claim C**：**Contract compliance is measurable at the policy boundary, not only at the estimator boundary.** —— 9/14 §8 的 Interface Property Benchmark 是 estimator 侧；本文 §6 的 HPS / SDS / PCE 是 policy 侧。两侧合起来、contract 才是一个**可验证的对象**、不是一句**接口倡议**。

一句收尾：**"融合"这个词、以后尽量不用**——它在时间轴上问的是"什么时候合并"、在语义轴上问的是"合并成什么"；9/14 与本文合起来把第二个问题拆成了**上游交付什么 + 下游读出什么**两半。剩下第一个问题（时机）已经被 §1 的三家族对照回答得差不多——**时机是接口的结果、不是接口的决策变量**。这一点如果能立住、这一系列的文章就都值了。

## Sources

以下 arXiv ID 已联网核过；journal-only 引用不贴 arXiv。按支撑的 section 分组。

### A · VLA 家族（支撑 §1 家族 C、§3 Failure 1–2、§5.1 auxiliary loss）

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)（把 action 表达成 text token 与 VLM 联合 fine-tune · §3 Failure 2 semantic drift 的典型形态）
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)（开源 VLA 基线 · 公开配置含多相机 / depth / proprioceptive state encoding；"支持输入" ≠ "读到 contract 的哪一站"）
- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)（VLM backbone + proprio token + noisy action chunk + flow matching · §4.1 mode_select 的一个具体形态）
- Octo Model Team, *Octo: An Open-Source Generalist Robot Policy*, RSS 2024 · [arXiv:2405.12213](https://arxiv.org/abs/2405.12213)（transformer-based readout · §4.3 attention_bias 路径的一个参照）

### B · Diffusion / Flow-Matching Policy（支撑 §1 家族 B、§5.1）

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 · [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)（RGB stack + proprio concat + denoising · §3 Failure 1 mode collapse 与 §3 Failure 3 temporal-alignment-broken 的现场证据）
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ACT / ALOHA）, RSS 2023 · [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)（CVAE + transformer encoder-decoder、chunked action · 家族 B 与 C 的一个中间形态）
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 · [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)（continuous action chunk 的 velocity regression · §5.1 $\mathcal{L}_{\mathrm{action}}$ 的一种具体形式）

### C · 不确定性、校准与 belief-space 参照（支撑 §4.1、§5.1、§6）

- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)（现代网络过度自信、temperature scaling 起点 · §5.1 $\mathcal{L}_{\mathrm{calibration}}$ 的动机）
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels*（PlaNet / RSSM）, ICML 2019 · [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)（deterministic + stochastic latent、多 hypothesis 的一种工程实现 · §4.1 read 的一个参照）
- Hafner et al., *Mastering Diverse Control Tasks through World Models*（DreamerV3）, Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)（离散 + 连续混合 latent、KL balancing · §4.1 ECE-preserving top-$k$ 的一条相邻路线）

### D · SAC / PPO 与家族 A 的 baseline（支撑 §1 家族 A）

- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, ICML 2018 · [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)（Gaussian NLL / max-entropy policy loss · §5.1 $\mathcal{L}_{\mathrm{action}}$ 的家族 A 形态）

### E · 承接前文（本文与 9/13、9/14、9/10 Part 3 的接口）

- 本博客《拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口》· `/zh/articles/2026-09-14-multimodal-fusion-interface/`（Structured State Contract 定义、Interface Property Benchmark、degradation chain · 本文 §3–§6 直接建立在其上）
- 本博客《只会看、不会摸：机器人为什么缺一双"手感"的手》· `/zh/articles/2026-09-13-tactile-force-sensing/`（力 / 触觉的四种控制范式、Closed-loop value · 家族 A 与家族 B 之间的历史分水岭）
- 本博客《Sim-to-Real 方法论（三）》· `/zh/articles/2026-09-12-sim-to-real-evaluation-protocol/`（三级证据层、decision utility 三维、allocation protocol · §5.4、§6 直接沿用其 evaluation spine）
- 本博客《Sim-to-Real 方法论（一）》· `/zh/articles/2026-09-10-sim-to-real-methodology/`（allocation state $s_t = (b_t, \pi_t, q_t, h_t)$、$\Delta_{\mathrm{queue}}$ vs $\Delta_{\mathrm{processing}}$ · §2、§4.2 定义直接接上）

---

> **相关阅读**
>
> - [拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口](/zh/articles/2026-09-14-multimodal-fusion-interface/)——本文的前作、把上游交付物立成 Structured State Contract
> - [只会看、不会摸：机器人为什么缺一双"手感"的手](/zh/articles/2026-09-13-tactile-force-sensing/)——§1 家族 A/B 分水岭的历史脉络、四种力控范式
> - [具身智能 Sim-to-Real 方法论（三）](/zh/articles/2026-09-12-sim-to-real-evaluation-protocol/)——§5.4、§6 沿用其三级证据与 utility 三维
> - [VLA 与世界模型：两条路线的分岔与合流](/zh/articles/2026-09-07-vla-world-models/)——本文 §1 家族 C 的宏观背景、"世界模型不天然属于 sim-to-real" 的另一面
> - [VLA π 家族速览](/zh/articles/2026-09-05-vla-pi-family/)——π0、π0.5 与 flow-matching action head 的一个具体切面
> - [什么是 VLA 模型？一篇讲清楚](/zh/articles/2026-09-03-vla-deep-dive/)——家族 C 的入门版、本文假设你已经读过
