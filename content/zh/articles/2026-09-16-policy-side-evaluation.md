---
title: '政策侧评估（下）：如何测、如何训、如何落地——四类 compliance evidence 与最小可执行接口'
slug: "2026-09-16-policy-side-evaluation"
date: 2026-09-16
draft: false
categories: ["Embodied AI", "Policy Learning"]
tags: ["Embodied AI", "Policy Learning", "Structured State Contract", "Consumer Contract", "Contract Consumer", "Compliance Evidence", "Five-Layer Evaluation", "Separately Auditable Failure Sites", "Interface Compliance Metric", "Matched Null Control", "Consumer-Declared Order", "Multi-Constraint Intersection", "Contract-Read Primitives", "Intervention Consistency", "Equivariance", "Order-Constrained Response", "Conditional Log-Likelihood Ratio", "Dependency-Aware Fusion", "Mass-Preserving Top-k", "Constraint Certification", "Contract-only Intervention", "World-consistent Counterfactual", "Action-Relevant Separation", "Contract Ablation Gap", "Fixed Policy vs Retrained Policy", "Evaluation Metrics"]
description: '本文承接 9/15《契约立起来之后：VLA、Diffusion Policy、π0 到底在消费什么》，把 policy 侧 contract consumer 的**评估协议、训练约束与最小可执行接口**独立成篇。前篇负责 framework 与失败模式、本文负责"如何测 / 如何训 / 如何落地"。核心 boxed thesis：**Contract compliance should be tested by controlled intervention, not inferred from end-to-end success**——端到端 success 衡量 policy 好不好用、不衡量它有没有把 contract 语义读对。本文提出**四种 compliance evidence**（$E_{\mathrm{semantic}}$ / $E_{\mathrm{representation}}$ / $E_{\mathrm{decision}}$ / $E_{\mathrm{safety}}$、不是四个 metric、不能压成一个 scalar）、并在 §1 开头用**理论骨架表 + 五层 evaluation hierarchy**（Retention ≠ Sufficiency ≠ Behavioral use ≠ Utility ≠ Safety、四条不蕴含关系）当全文评估地图。§2 训练时连锁改动：representation-side probe、三类 intervention consistency（Type I equivariance / Type II consumer-declared order-constrained response / Type III unconstrained）、degradation as causal operator 的 augmentation、§2.3 safety filter 的三态化 constraint certification（safe→relaxation permitted / unsafe→tighten or stop / unknown→conservative fallback、$g_{\mathrm{safety}}=\bigcap_j g_j$）。§3 展开四类 evidence——§3.2 v6 把 HPC 二次拆成 $E_{\mathrm{contract}}(T^{\mathrm{contract}})$（interface compliance、checked against $\mathcal R_{\mathcal C}$、no oracle required）与 HPC($T^{\mathrm{world}}$)（decision competence、$\mathcal A^*_k$ 由 simulator 定）、HSS 用 $D_{\mathcal A}$ 加**competence gate**、四列并报防 gaming；§3.3 SDS 给操作化 $V_{\mathrm{order}}$ 与 $V_{\mathrm{trans}}$ violation estimator；§3.5 CAG 加 **matched null control** $\Delta J_{\mathrm{null}}$、只有当 $\Delta J_{\mathrm{contract}}\gg\Delta J_{\mathrm{null}}$ 时才算 contract-use。§4 Python 最小可执行接口骨架 v6 修复 `super().__init__()`、fail-closed `raise SchemaCompatibilityError`、top-$k$ 拆 `topk_weight_mode` / `residual_mode` 两个正交 knob、加 `benchmark_rng` 走 common random numbers。§5 收束：Claim 3 升级为本文 thesis、"policy 只是函数近似器"的时代结束、下一步是 Contract-Preserving Policy Benchmark。'
toc: true
related_articles:
  - 2026-09-15-policy-side-interface
  - 2026-09-14-multimodal-fusion-interface
  - 2026-09-13-tactile-force-sensing
  - 2026-09-12-sim-to-real-evaluation-protocol
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-08-data-and-training-recipes
---

> 本文是**政策侧接口**这篇长文的**下半部分**。前篇 [《契约立起来之后：VLA、Diffusion Policy、π0 到底在消费什么》](/zh/articles/2026-09-15-policy-side-interface/) 建立 framework——**Consumer Contract 三段分解** $\mathcal C_\pi=(Q_\pi,\mathcal O_\pi,V_\pi)$、三档 preservation、三个 semantic losses + 一个 safety obligation、四种 interface-mismatch 失败模式、三族 contract-read primitives。本文回答它的对偶问题：**给定这些框架、policy 侧要"如何测、如何训、如何落地"**。

> 一句话承接：**Contract compliance should be tested by controlled intervention, not inferred from end-to-end success**——端到端 success 衡量 policy 好不好用、不衡量它有没有把 contract 语义读对。合规 argument 需要**四类证据合流**（semantic / representation / decision / safety）、需要**一条 oracle baseline 界定每个指标的语义范围**、需要**明确的 retraining protocol**。这一句在前篇 §0.3 是 Claim 3、在本文是主 thesis。

本文不重复前篇的完整 framework 推导。所有 $\mathcal C_\pi / q_\pi / \Pi_\pi / \pi_\theta / g_{\mathrm{safety}}$、三档 preservation、$Y_{\mathcal C}^{\pi}$、$L_{\mathrm{declared}} / L_{\mathrm{projection}} / L_{\mathrm{decision}}$ 与 $\mathcal O_{\mathrm{safety}}$ 的定义都在前篇 §0；本文只保留评估与实现所需的最小回顾，并在 §0.2 用一张 boxed pipeline 图钉住引用点。

## 0. 本文框架：四类 compliance evidence、五层 evaluation hierarchy、一个最小可执行接口

### 0.1 与前篇的分工

前篇（9/15）建立的是 policy 侧接口的**语义层**——它回答"policy 到底在消费什么、现有接口为什么不行"。本文（9/16）建立的是 policy 侧接口的**度量层与实现层**——它回答三个具体问题：

1. **如何测**：给定一个 policy checkpoint，你用什么协议判断它是否 contract-compliant？（§2 四类 compliance evidence + §2.0 理论骨架表 + 五层 evaluation hierarchy）
2. **如何训**：为了让 policy 真的读 contract，训练目标、augmentation、safety filter 各要改什么？（§1 训练时连锁后果）
3. **如何落地**：一个最小可执行的 contract-aware policy 骨架长什么样、接口层的 bug 一般藏在哪？（§3 Python skeleton + fail-closed + top-$k$ API 拆分 + `benchmark_rng`）

三块合起来构成**"policy 是 contract consumer"这一 thesis 的实验闭环**——前篇立 claim、本文兑现 claim。

### 0.2 复用前篇 pipeline，作为本文的评估坐标系

$$\boxed{\;\mathcal C\;\longrightarrow\;(\mathcal C_\pi,\,Q_{\mathcal C_\pi})\;\longrightarrow\;q_\pi\;\longrightarrow\;e_\pi\;\longrightarrow\;\pi_\theta\;\longrightarrow\;g_{\mathrm{safety}}\;\longrightarrow\;\mathcal B_{\mathcal C}\;}$$

其中 $\mathcal C_\pi=(Q_\pi,\mathcal O_\pi,V_\pi)$ 是 Consumer Contract 三段分解（Queries / Obligations / Versions）、$\mathcal B_{\mathcal C}$ 是 contract 侧干预 battery $\{T_{\mathrm{frame}},T_{\mathrm{hyp}},T_{\alpha},T_{\mathrm{validity}},T_{\mathrm{prov}},T_{\mathrm{neg}}\}$。前篇 §0.2 给出 $q_\pi / \Pi_\pi / \pi_\theta$ 上三个 semantic loss 与 $g_{\mathrm{safety}}$ 上一个 safety obligation 的完整定义；本文**只**用最小回顾：

- $L_{\mathrm{declared}} = \sum_{q\in Q_{\mathcal C}^{\mathrm{req}}} w_q\,\mathbf 1[\nexists q'\in Q_{\mathcal C_\pi}:q'\succeq q]$（**query subsumption coverage**、不是字面集合成员、详见前篇 §0.2.2）
- $L_{\mathrm{projection}} = I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$（$Y_{\mathcal C}^{\pi}$ 由 Consumer Contract 定义、不是 benchmark 里凭空的 latent、详见前篇 §0.2.1）
- $L_{\mathrm{decision}} = \mathbb E_{\mathcal R_{\mathcal D}}[\mathbf 1[D_{\mathcal A}(\pi_\theta(\cdot\mid\hat S),\pi_\theta(\cdot\mid\hat S'))<\epsilon]]$（action-relevant pair 的 collapse rate、$D_{\mathcal A}$ 与 §2.2 HSS 同族）
- $\mathcal O_{\mathrm{safety}}$：safe → relaxation **permitted**（not required）；unsafe → tighten or stop；unknown → conservative fallback；$g_{\mathrm{safety}}=\bigcap_j g_j$。

**三个 loss + 一个 obligation 是四个 separately auditable failure sites、不是四个 statistically independent losses**——它们通过 $q_\pi\to\Pi_\pi\to\pi_\theta$ 顺序耦合。这一点前篇 §0.2.2 已完整讨论、本文所有 evidence 章节都遵循这个措辞。

### 0.3 本文的三条 boxed claim

> **Claim A · Compliance is a multi-evidence argument, not a score.** Contract compliance 需要 §2 的**四种 compliance evidence**合流——$E_{\mathrm{semantic}}$（invariance / equivariance）+ $E_{\mathrm{representation}}$（conditional probe + $E_{\mathrm{contract}}(T^{\mathrm{contract}})$ + HPC($T^{\mathrm{world}}$) + HSS + calibration）+ $E_{\mathrm{decision}}$（CAG$^{\mathrm{fixed}}$ + oracle baseline + matched null + 三条 independent source ablation）+ $E_{\mathrm{safety}}$（三态 constraint certification intervention）——**probe ≠ semantic compliance、CAG ≠ semantic compliance、safety pass ≠ representation retention**。四类证据不能互相顶替、也不能压成一个 scalar。

> **Claim B · Evaluation layers do not imply each other.** §2.0 的五层 hierarchy **Retention ≠ Sufficiency ≠ Behavioral use ≠ Utility ≠ Safety**：Retention 通过不蕴含 Sufficiency（信息在可能也不足以回答 query）、Sufficiency 通过不蕴含 Behavioral use（够信息 policy 可能不用）、Behavioral use 通过不蕴含 Utility（响应对了 utility 仍可能差、因为 contract 未必是当前瓶颈）、Utility 与 Safety 完全正交。**这四条"不蕴含"关系**是 §2.2 拆 $E_{\mathrm{contract}}$ / HPC、§2.5 拆 CAG fixed / retrained、§2.6 拆 constraint certification intervention 的根本动机。

> **Claim C · The interface is where the bugs hide, not the architecture.** §3 的 Python 骨架 v6 集中修了三个真实在 contract-aware 接口层反复出现的 bug：`super().__init__()` 缺失导致 `nn.Module` 子参数没被注册、fail-open 的 schema 兼容检查让 version mismatch 静默降级、top-$k$ API 把两种正交决策（weight mode / residual mode）揉成一个字符串。这三处都不是模型架构问题、都是**接口层问题**——与本文主 thesis 完全一致：**policy 是 contract consumer、不是仅仅是 function approximator**。

### 0.4 阅读路径

- 只关心**怎么评**：读 §2（骨架表 + 五层 hierarchy + 四类 evidence）、§2.5 CAG matched null 是防"OOD sensitivity 冒充 contract-use"的关键。
- 只关心**怎么训**：读 §1（三类 intervention consistency + degradation-as-operator augmentation + 三态 safety filter）。
- 只关心**怎么落地**：读 §3（Python skeleton + fail-closed + top-$k$ API 拆分 + `benchmark_rng` CRN）。
- 想看**全文如何闭环**：读 §0.2 一张 boxed pipeline 图 + §2.0 两张表（loss/obligation 落站点 + evidence 落五层）+ §4 收束。

## 1. 训练时连锁改动：训练约束、augmentation、safety filter 三处接口

一旦 policy 输入端接了 前篇 §4 的 primitives、训练目标、augmentation、safety filter、evaluation 四处都要跟着改。**接口不是免费的**——但改动是**局部的、可控的**。

### 1.1 两类训练约束：representation-side probe + 三类 intervention consistency

一个自然的错误是：**为了"让 policy 用 contract"、要求 policy 输出 validity / hypothesis 的预测头**——这实际上是把"用 contract"偷换成了"复制 contract"、方向不对。**policy 完全可以只吃 contract、不吐 contract**——auxiliary prediction head 不是必要条件。

#### 类别 A · Representation-side probe（诊断）

给 policy 的中间表示 $z^{\pi}$ 挂几个 probe head、尝试从 $z^{\pi}$ 预测 contract 里的 `age`、`validity`、`observability`、hypothesis posterior。**这些 probe 不参与主 loss、只用来测**：probe 预测得好、说明 policy 内部保留了这些信息；probe 预测得差、说明 contract 在 $e_\pi$ 里已经被压掉了。$L_{\mathrm{probe}}$ 可以以很小的权重加到主 loss 上作为 regularization、但**它的诊断价值大于训练价值**。

$$\mathcal{L}_{\mathrm{total}} \;=\; \mathcal{L}_{\mathrm{action}} \;+\; \underbrace{\alpha\, \mathcal{L}_{\mathrm{probe}}}_{\text{weak regularization, mainly diagnostic}} \;+\; \underbrace{\sum_{\mathcal{C}} \gamma_{\mathcal{C}}\, \mathcal{L}^{\mathcal{C}}_{\mathrm{consistency}}}_{\text{Type I / II / III, see below}}$$

**但 probe 有一个上一版没解决的漏洞**——reviewer 抓得对：假设 $age$ 与 image embedding 高度相关（例如 "scene 越复杂、sensor 处理越慢、age 越大"）、那么 probe 从 $z_\pi$ 里可以很容易预测 age：

$$\mathrm{Acc}(\alpha\mid z_\pi) \;=\; 99\%.$$

**这并不证明 age 被 $e_\pi$ 保留了**——完全可能只是 image 里的 scene difficulty 泄漏。**正确的诊断应该用 conditional / nuisance-controlled probe**：

$$\text{Retention}_{\mathrm{cond}}(\alpha) \;=\; I(\alpha;\, z_\pi \mid o),$$

即在 raw observation $o$ 给定的条件下、$z_\pi$ 是否还**独立地**携带 age 信息；或者在 benchmark 里做"**same observation, different metadata intervention**"——固定 $o$、只改 $m_c$、看 $z_\pi$ 的响应。这与 前篇 §0.2.1 Property B 用 conditional MI 而不是差分 MI 的哲学一脉相承：**凡是"raw input 里已经有 proxy"的字段、probe 都必须 conditional**。

#### 类别 B · Intervention-consistency constraint（核心）

**上一版把这一类笼统写成 $D(\pi_\theta(\hat S), \pi_\theta(T_{\mathcal{C}}(\hat S)); \rho_{\mathcal{C}})$、reviewer 立刻问 $\rho_{\mathcal{C}}$ 从哪来**——四种 $T_{\mathcal{C}}$ 各自的"响应规律"不一样、有一类甚至根本不该预设"必须响应"。这一版按响应强度把 intervention 分成三档。

**Type I · Exact invariance / equivariance**（最干净的一档）。变换在 policy input 上有一个合法的 action-side 对应 $T^{\mathcal{C}}_\pi$、要求：

$$\pi_\theta\!\big(T^{\mathcal{C}}(\hat S),\, o,\, \ell\big) \;=\; T^{\mathcal{C}}_\pi\!\big(\pi_\theta(\hat S,\, o,\, \ell)\big).$$

典型：**frame transform**——把 `reference_point` 从 A 移到 B、$\tau$ 按 transport theorem 变换、**要求 action 侧做对应的坐标变换**（$\pi(T_g S) = T_g^A \pi(S)$）。类似：**permutation of equivalent hypotheses**（同 posterior weight 的 hypothesis 互换、action 分布必须等价）。这一档可以写成硬 loss、$\mathcal{L}_{\mathrm{consistency}}^{\mathrm{I}} = \|\pi_\theta(T\hat S) - T^\pi \pi_\theta(\hat S)\|^2$。

**Type II · Order-constrained response**（次强的一档、**上一版叫 monotone response、这一版把它数学化**）。"monotone" 不是随便就能用的词——只有当 $M(\cdot)$ 的值域上有偏序、并且 response functional 是**明确定义的标量或全序**时才能谈单调。上一版把 variance、action norm、fallback probability、covariance PSD 全塞进同一个 $\preceq$、reviewer 抓得对：**这几种 $\preceq$ 根本不是同一个 order**。

正确的提法是：先给出**contract-declared intervention order** $T_1 \preceq_{\mathcal C_\pi}^{\mathrm{declared}} T_2$（例如 "age 越大 = 越严重"、"observability 越低 = 越严重"、"validity 位为 false = 比 age 高更严重"——**关键：这个 order 由 $\mathcal C_\pi$ 与 $Q_{\mathcal C_\pi}$ 声明、不是"客观物理 degradation order"**。reviewer 举了一个非常锋利的反例：机器人静止时、vision age 从 10 ms 到 100 ms **对 action 完全无影响**、"age 越大 = severity 越高"在这种任务下就不是天然成立的偏序——它必须由 consumer contract 显式声明、而不是从物理量猜出来。这也是 v6 把符号从 $\preceq_{\mathcal C}$ 改成 $\preceq_{\mathcal C_\pi}^{\mathrm{declared}}$ 的原因——上标 $\mathrm{declared}$、下标 $\mathcal C_\pi$ 都在提醒读者：order 属于 consumer 声明、不属于世界本身）。order 定了之后、再指定一个**response functional**

$$r:\mathcal P(\mathcal A) \;\longrightarrow\; \mathbb R$$

（可以是 $P(\text{fallback})$、$\mathbb E[\|a\|]$、$P(\text{stop})$、$\mathbb E[\mathrm{safe\_margin}]$ 之类、每个是 scalar、有全序 $\le$）、然后要求：

$$T_1 \preceq_{\mathcal C_\pi}^{\mathrm{declared}} T_2 \quad\Longrightarrow\quad r\!\big(\pi_\theta(T_1\hat S)\big) \;\le\; r\!\big(\pi_\theta(T_2\hat S)\big).$$

**这才是严格意义的 monotonicity**。举例：$r(\pi) = P_\pi(\text{fallback})$、$T_1$ = "age 从 5 ms 到 20 ms"、$T_2$ = "age 从 20 ms 到 200 ms"、则 $T_1 \preceq_{\mathcal C_\pi}^{\mathrm{declared}} T_2$ 且要求 $P_\pi(\text{fallback}\mid T_1) \le P_\pi(\text{fallback}\mid T_2)$。

但真实 policy 完全可能是**分段的**——

```
α < 50 ms      →   正常控制
50 ms ≤ α < 100 ms → fallback
α ≥ 100 ms     →   stop
```

这种 response **不 monotone、但是合法的 contract-specified response relation**。所以 Type II 的正确名字应该是 "**order-constrained response**"、**monotonicity 只是它的一个特例**。写成 loss 是：

$$\mathcal{L}_{\mathrm{consistency}}^{\mathrm{II}} \;=\; \sum_{T_1 \preceq_{\mathcal C_\pi}^{\mathrm{declared}} T_2} \max\!\big(0,\; r(\pi_\theta(T_1 \hat S)) - r(\pi_\theta(T_2 \hat S)) + \delta\big).$$

$\delta$ 是 margin、$r$ 与 $\preceq_{\mathcal C_\pi}^{\mathrm{declared}}$ 都必须在 $\mathcal C_\pi$ 里写死、不能事后凑。

**Type III · Unconstrained intervention**（最弱、也最重要的一档）。**不预设 policy 必须变化**——典型是 **provenance removal**：拿掉一个 contributing sensor、若另一个 sensor 完全冗余替代、**最优 action 可以完全不变**。这一档的正确提法是：**when the removed evidence was decision-relevant, does performance degrade?** 也就是把它交给 §2 的 CAG 面板去测、而不是训练时强加响应规律。写成 loss 就是：**不做任何 intervention consistency、只在 evaluation 阶段做 ablation**。这一档存在本身是对上一版的一个纠正——上一版把四种 $T_{\mathcal{C}}$ 一视同仁地塞进"要求响应"、是把 Type III 错当成了 Type II。

**这三类都不要求 policy 显式预测什么**；它们要求的是**响应函数符合 contract 语义档位、或者被允许符合"不变"**。这才是 前篇 §0.2 "$q_\pi$ 是显式声明的 quotient" 定义的**训练侧对应物**。相比"加几个 auxiliary 预测头"、这套三类约束更贴合本文 thesis、也更有研究味：**我们建议的不是让 policy 复制 contract、是让 policy 在 contract 变换下按对应的响应档位响应**。

### 1.2 Augmentation：degradation as causal operator

9/14 §4.5 强调 **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption**——六种降级各有不同的**因果起源**与不同的**下游读法**。上一版试图给每类降级标"对应哪个 slot"、reviewer 抓得对：**这种 one-to-one 映射过度简化**。举个具体的：

- **latency** 的 primary effect 是抬 $\alpha_c$、但如果没有做 compensation、secondary effect 是让 $\mu_c$ 变成"延迟时刻的真实值"——同一份 augmentation 同时动了两个 slot。
- **bias** 的 primary effect 是 shift $\mu_c$、但 secondary effect 通常还包括 inflate $\Sigma_c$（因为系统意识到 calibration 不可信）、甚至 flip $v_c$。
- **corruption** 可能同时改 $\mu_c, \Sigma_c, v_c, h_c$ 四样。
- **masking** 的 primary effect 是 $\iota_c = 0$、secondary effect 是让下游 $q_c$ 变成 undefined、$\Sigma_c^{\mathrm{eff}}$ 必须回落到 prior。

这一版把每条降级**改成一个 causal operator**：

$$T_d:\;(\mu_c,\,\Sigma_c,\,m_c)\;\longmapsto\;(\mu_c',\,\Sigma_c',\,m_c'),\qquad d \in \{\text{mask},\,\text{missing},\,\text{stale},\,\text{latency},\,\text{bias},\,\text{corruption}\}.$$

一句话原则：**each degradation has a primary semantic effect and potentially secondary effects on other fields**——aug pipeline 里必须**逐条列出** $T_d$ 具体动了哪些 slot、动了多少、而不是"aug 类别 → 单个 slot"。前篇 §4.2 的三组结构（payload / metadata / derived trust）在这张表上直接可用：**$T_d$ 作用在 $(\mu,\Sigma)$ 上是 payload-level corruption、作用在 $m_c$ 上是 metadata-level augmentation、$q_c$ 必须是从 modified $m_c$ 派生出来的、不能被 aug 端直接改写**（否则就是"aug 端在作弊、部署端读不到"）。

**上一版这里有个 reviewer 抓到的 shortcut 风险**："给每个 episode 打 degradation label、policy input 里显式携带"——如果 `degradation = stale_vision` 这个 label 直接作为 policy input、policy 会学成 $a = f(o, \text{label})$；但**真实部署里 label 本身未必可靠**、这个 shortcut 在训练里看起来无害、部署里就是灾难。这一版明确区分两样东西：

**Observed metadata** $m_c = (\alpha_c, \ell_c, h_c, v_c, \iota_c)$——由 estimator 或 sensor driver 直接得到、可以进 contract、可以进 policy input。**这是 前篇 §4.2 metadata quintuple 的来源**。

**Latent degradation class** $d_c \in \{\text{missing}, \text{stale}, \text{bias}, \text{corrupt}, \ldots\}$——augmentation 时你**知道**注入了哪一类 $T_{d}$、但部署时这个类是**latent 的**、只能由 contract estimator 从 $m_c$ 序列里推断、或只作为训练 annotation 用来加权 loss / 采样、**不能默认作为 ground-truth input 塞进 policy**。

具体做法：aug pipeline 采样 $d_c$、按 $d_c$ 通过 $T_{d_c}$ 生成 $(\mu_c', \Sigma_c', m_c')$、然后把 $m_c'$ 作为 policy input、$d_c$ 只作为 loss weighting 与 evaluation 分层的 key。**这不是 curriculum、是 conditioning on observed metadata**——差别在于 conditioning 让 policy 看见的是可靠的 $m_c'$、而不是不可靠的 $d_c$。§1.1 类别 B 的 Type II order-constrained response 与 degradation-conditioned aug 天然配对——**aug 端造 $T_{\mathcal{C}}$ 的 $m_c$ 变化、loss 端测 policy 对 $m_c$ 变化的响应是否符合 $\preceq_{\mathcal C_\pi}^{\mathrm{declared}}$ 与 $r$**。

### 1.3 Safety filter 与 contract 的接口：constraint certification（v5 三态化）

约束层（CBF / shield / runtime verifier）**必须读 $a_{\mathrm{proposed}}$**、否则它 filter 什么——这一点不用退让。但本文更精确的 claim 是：**safety filter 不应该把 policy 的 confidence 或 estimator 的通用 validity 位当成约束成立性的唯一证据；它应该直接访问 constraint-relevant evidence**。

**这里必须区分两种 validity**。Estimator 侧的通用 `validity` 位说的是"这条 measurement 从 calibration / sensor health 的角度看是否有效"——这是 field-level 的 general-purpose 声明。Safety 侧真正关心的是"**这条约束在这个时刻、这个 predicate 下是否 valid**"——这是 constraint-level 的语义。两者并不相同：`validity=true` 不意味着 constraint estimate 对**这条 safety predicate** 有效；例如"关节力矩读数 calibration OK" ≠ "当前接触估计足以支撑 collision constraint"。

本文建议 safety filter 读到的、是一个 constraint 特定的组合量：

$$v_j^{\mathrm{constraint}} \;=\; g\!\Big(\text{field validity},\; \text{observability},\; \text{hypothesis posterior},\; \text{age},\; \text{model coverage}_j\Big).$$

$g$ 是 constraint-specific 的组合规则（例如 CBF 那侧要求"距离估计在当前 hypothesis 下、observability 充分、且 dynamics model coverage 到当前状态区域"）、$v_j^{\mathrm{constraint}}$ 才是 safety filter 真正应该读的**constraint-relevant evidence**。9/14 §3 讲过 **track_id 是 hypothesis**——那么 safety filter 也不能只信 track_id 匹配、还要看 hypothesis posterior 是否稳定；两个 track 是否 merge / split、直接决定"这个障碍距离"的可信度。

**v5 补：三态 certification**。上一版把 $v_j^{\mathrm{constraint}}$ 写成 binary 位、reviewer 抓得对——**"证据不足"与"约束不成立"是两件不同的事**。举三个具体 case：invalid sensor 是 unknown、obstacle detected 是 unsafe、obstacle absent with high observability 是 safe。用 binary $v = 0/1$ 表达这三种情况会把它们全塞进同一个值、safety filter 拿不到必要的信息。所以本文引入三态：

$$\boxed{\;\mathrm{certification}_j \;\in\; \{\text{safe},\;\text{unsafe},\;\text{unknown}\}.\;}$$

对应的反应规则：**safe → relaxation is permitted**（注意 **permitted 不是 required**——见下文）、**unsafe → tighten / stop**、**unknown → conservative fallback 或 tighten**（因为不能确定、所以按更谨慎的一侧处理）。这一改让 safety contract 更接近真正的 runtime safety semantics——$v_j^{\mathrm{constraint}} = 0$ 不再意味着 "constraint false"、它意味着 **"evidence is insufficient to certify the constraint predicate"**（对应 unknown 状态）。

**v6 补·safe ≠ 必须 relax**。上一版把 "safe → 可以 relax" 写成 "safe → relax"、reviewer 举了一个非常简单的反例：collision constraint = safe、joint torque constraint = unknown——即使 collision constraint 已经 safe、也**不能因此放松整个 safety envelope**、因为 joint torque 那一路的 constraint 还在 unknown 状态。正确的语义是：**safe 只表示"这一条 constraint 允许按正常 margin 执行"、不表示"这一条 constraint 允许被拆开"**。因此本文把三态反应规则精化为：

1. **safe → allow_normal_margin**（不放松 constraint、只是把这一条 constraint 的 margin 保持默认、继续与其它 constraint 求交）；
2. **unsafe → tighten_or_stop**；
3. **unknown → conservative_fallback**。

**多条 constraint 应该组合、而不是第一条 unknown 就 return**。safety filter 真正的形式是 $g_{\mathrm{safety}} = \bigcap_j g_j$——每条 constraint 独立给出它允许的 action subset、最终允许的 action 是**所有 constraint 的交集**。工程写法对应：

$$a_{\mathrm{applied}} \;=\; \Big(\bigcap_{j:\,\mathrm{cert}_j = \text{safe}} g_j^{\mathrm{normal}}(a_{\mathrm{proposed}})\Big) \;\cap\; \Big(\bigcap_{j:\,\mathrm{cert}_j = \text{unsafe}} g_j^{\mathrm{tighten}}(a_{\mathrm{proposed}})\Big) \;\cap\; \Big(\bigcap_{j:\,\mathrm{cert}_j = \text{unknown}} g_j^{\mathrm{fallback}}(a_{\mathrm{proposed}})\Big).$$

任何一条 constraint 单独把 action 空间"松"回去都不合法——只有**所有 constraint 都允许**的那部分 action 才能被 apply。这条性质也决定了 §3 里 `SafetyFilterHead.forward` 的正确写法（详见 v6 版代码：`continue` 只跳过本条 constraint 的收紧、绝不 `return` 提前结束循环）。

**这一节最重要的一句话（v5 加了一个词）**：

> **$v_j^{\mathrm{constraint}}$ does NOT turn off the constraint when evidence is invalid or unknown.**
> **Invalid evidence ALONE cannot justify relaxing the constraint.**
> Equivalently: **Relaxation requires sufficient valid evidence; invalid evidence alone is never sufficient.**

上一版这里写得含糊、读起来像"validity 挂了、constraint 就可以 off"——**这在 safety 语义上是危险的**、也容易被 reviewer 用一个反例打穿（camera invalid 但 lidar valid 且已充分证明 obstacle 不存在、此时 constraint 完全可以 relax）。加上 "alone" 之后、语义变成："**invalid evidence 单靠自身不能成为 relax 的充分条件**；如果**另一路独立 valid evidence 已经充分证明 constraint 不成立**、那就是合法 relax"。这一改把 counterexample 挡在门外、同时不削弱本文的核心 safety claim。三种 reaction：

1. **fallback**：切到一个更保守的 controller 或 planner；
2. **conservative tightening**：**加大** constraint margin（例如把 minimum distance 从 20 cm 抬到 50 cm）、因为"证据不够 = 更谨慎"；
3. **stop**：完全 stop policy、等观测恢复。

**三种反应都不是"constraint off"**——除非另一路 valid evidence 已把 certification 显式推到 **safe**。安全语义上的核心事实是：**positive evidence（且充分）才能 justify relaxing a constraint；absence of valid evidence 永远不能单独做到这件事**。

这条也接上了 前篇 §4.3.3 的 negative evidence：**"本应看到但没有"** 会让 $\mathrm{certification}_j$ 变 unknown——例如雷达在这个角度什么都没扫到、$\Lambda(E^-; H_{\text{clear}}, \mathcal O)$ 变正（"clear" hypothesis 下"没看到东西"是自然结果、但反过来"有障碍" hypothesis 下"没看到东西"是不自然结果——**方向依赖 $H$ 的定义、见 前篇 §4.3.3**）、collision constraint 的 certification 应该保持 unknown、甚至**保守收紧**、不能因为 policy 的 belief 乐观就把 certification 推到 safe。

**这三样（constraint-specific certification、observability、negative evidence）不进 safety filter、filter 就会用 policy 的 belief 反推约束成立性**、这在低 observability 区域特别危险——policy 的 belief 之所以乐观、是因为它读不到 contract 里的 observability / validity / negative evidence、**filter 如果同样读不到、两者一起盲**。

### 1.4 与 9/10 Part 3 evaluation 的呼应

Sim utility 三维（prediction / ranking / decision）里、policy-side 的 evaluation 主要看 **decision** 这一维——但要加一个 **contract-preservation 维度**：把 contract 拆掉之后 policy 的表现下降多少、就是它对 contract 依赖度的**下界**。这一维度对应 §2 的 CAG 指标、且需要 oracle baseline 来界定解释边界、以及**明确的 retraining protocol**（§2.5）。

## 2. Evaluation：四类 compliance evidence、五层 evaluation hierarchy、不是四个 metric

对应 前篇 §4 的 primitives 与 §1.1 的三类 intervention、本文提出**四种 compliance evidence**——**不是一个四层 metric panel、是四类各自回答不同问题的证据**。上一版把它们写成 metric hierarchy、reviewer 抓得对：probe / CAG / safety-pass **任何单一 score 都不构成 semantic compliance**、必须**多证据合流**。

$$\boxed{\;\text{Compliance Evidence} \;=\; \big\{E_{\mathrm{semantic}},\;E_{\mathrm{representation}},\;E_{\mathrm{decision}},\;E_{\mathrm{safety}}\big\}.\;}$$

四类各自回答一个不同的问题：

| 类型 | 回答什么 |
|---|---|
| **$E_{\mathrm{semantic}}$** | **Does the policy respond correctly to a known semantic transformation?** |
| **$E_{\mathrm{representation}}$** | **Is the contract information recoverable from the representation (given the side inputs)?** |
| **$E_{\mathrm{decision}}$** | **Does contract structure change task utility when it should?** |
| **$E_{\mathrm{safety}}$** | **Does degradation cause conservative / required guardrail behavior?** |

三条明确的 caveat：**probe ≠ semantic compliance、CAG ≠ semantic compliance、safety pass ≠ representation retention**。这四类证据**不能互相顶替**、也不能压成一个 scalar。以下按类展开。Temporal 与 source 两条 invariant 的专项测试分别落在 $E_{\mathrm{decision}}$ 的 **SDS** 与 **$\Delta J_{\mathrm{where}} / \Delta J_{\mathrm{dep}} / \Delta J_{\mathrm{neg}}$** 切片——它们不是新指标、是 decision 层的 specialized 观察。

### 2.0 全文理论骨架表（v6 修订：三 loss + 一 obligation + 五层 evaluation hierarchy）

把 前篇 §0 到 §2 收在一张表上、reviewer 最希望看到的就是这个：

| Layer | Object | Failure | Evidence |
|---|---|---|---|
| Contract | $\mathcal C$ | schema ambiguity / version mismatch | schema audit + compatibility check |
| Declaration | $q_\pi$（由 $\mathcal C_\pi = (Q_\pi, \mathcal O_\pi, V_\pi)$ 诱导） | undeclared semantic collapse | quotient audit（能贴出 $Q_{\mathcal C_\pi}$ 清单吗？subsumption 覆盖吗？） |
| Projection | $\Pi_\pi = e_\pi\circ q_\pi$ | residual contract information | conditional probe $I(\text{field};z_\pi\mid o)$ |
| Decision | $\pi_\theta$ | wrong use / shortcut / collapse rate | Type I equivariance + Type II order-constrained + Type III ablation + $L_{\mathrm{decision}}$ |
| Safety | $g_{\mathrm{safety}}$ | unsafe interpretation of invalid / unknown evidence | constraint certification intervention（是否**收紧**、而不是**放松**） |

**三个 semantic losses + 一个 safety obligation**（v6 版、公式与 前篇 §0.2.2 严格对齐）：

$$\boxed{\begin{aligned}
L_{\mathrm{declared}} &: \;\textstyle\sum_{q\in Q_{\mathcal C}^{\mathrm{req}}} w_q\,\mathbf 1\!\big[\nexists\, q' \in Q_{\mathcal C_\pi}:\, q' \succeq q\big];\\[1mm]
L_{\mathrm{projection}} &= I\!\big(Y_{\mathcal C}^{\pi};\hat S \mid Z_\pi, O, L\big);\\[1mm]
L_{\mathrm{decision}} &= \mathbb E_{\mathcal R_{\mathcal D}}\!\big[\mathbf 1[D_{\mathcal A}(\pi_\theta(\cdot\mid\hat S),\pi_\theta(\cdot\mid\hat S'))<\epsilon]\big];\\[1mm]
\mathcal O_{\mathrm{safety}} &: \;\text{safe} \Rightarrow \text{relaxation permitted (constraint policy still applies)};\\
&\quad \text{unsafe} \Rightarrow \text{tighten or stop};\quad \text{unknown} \Rightarrow \text{conservative fallback}.
\end{aligned}}$$

三个 loss 分别落在 $q_\pi / \Pi_\pi / \pi_\theta$ 上、safety obligation 落在 $g_{\mathrm{safety}}$ 上、**四个可分别审计的 pipeline 站点**（**four separately auditable failure sites**）、合起来形成 compliance argument。**注意 v6 措辞**——本文**不宣称**这四个槽位是"statistically independent"的（reviewer 抓得对：$L_{\mathrm{declared}}$ 变了会改 $q_\pi$、改 $q_\pi$ 会改 $\Pi_\pi$ 的允许 quotient、进而改 $L_{\mathrm{decision}}$ 的评估集、四者是**顺序耦合**的、只是各自定位到一个可打开审计的站点）。这就是本文从"给 VLA 加 metadata"走到 "contract-aware policy design" 的具体形状。

**v6 再补·五层 evaluation hierarchy（reviewer 提的最值得加的一张表）**。本文一直想区分"信息还在不在 / 够不够 / 有没有用 / 用了值不值 / 不确定时怎么反应"——v6 之前散在 §2.1–§2.6、没有一张表把它们钉在一起。这张表是全文 evaluation 层的**类型系统**：

| 层 | 问题 | 主要指标 / primitive | 与其它层的分离点 |
|---|---|---|---|
| **Retention** | contract field 在 $z_\pi$ 里还在吗？ | conditional probe $I(\text{field}; z_\pi \mid o)$ | 只测"在不在"、不测"够不够" |
| **Sufficiency** | $z_\pi$ 加 side information 是否足以回答 $Y_{\mathcal C}^{\pi}$？ | $L_{\mathrm{projection}} = I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$ | conditional MI = 0、不宣称 representation-level injective |
| **Behavioral use** | policy 对 contract intervention 响应是否合法？ | Type I equivariance / Type II order-constrained / §2.2 $E_{\mathrm{contract}}(T_k)$ vs $\mathcal R_{\mathcal C}$ | 与"响应对不对"绑定、不测最终 utility |
| **Utility** | 用了 contract 信息、decision 真的改善了吗？ | $\mathrm{CAG}^{\mathrm{fixed}}$ + §2.2 HPC（$T^{\mathrm{world}}$ 版）+ §2.4 $\Delta J_{\mathrm{where/dep/neg}}$ | 必须配 matched null 与 oracle baseline、避免只是 OOD sensitivity |
| **Safety** | 不确定 / 无效 evidence 时有没有保守反应？ | §2.6 constraint certification intervention + §1.3 三态 certification + $\mathcal O_{\mathrm{safety}}$ | 独立于"policy 用没用 contract"、测的是 filter 有没有兜住 |

这五层是**层层递进、但不是相互蕴含**的关系——Retention 通过不蕴含 Sufficiency（信息在可能也不足以回答 query）、Sufficiency 通过不蕴含 Behavioral use（够信息 policy 可能不用）、Behavioral use 通过不蕴含 Utility（响应对了 utility 可能仍然差、因为 contract 未必是当前瓶颈）、Utility 与 Safety 完全正交（safety 侧兜底与 utility 侧表现是两件事）。**这四条"不蕴含"关系**是 §2.2 拆 $E_{\mathrm{contract}}$ / HPC、§2.5 拆 CAG fixed / retrained、§2.6 拆 constraint certification intervention 的根本动机。**Retention ≠ Sufficiency ≠ Use ≠ Utility**——这一句 v6 之前只是隐含、v6 明确写出来。

合起来，§2.7 的两张表共同承担本文 evaluation 部分的**理论骨架**：上一张表把 loss/obligation 落到 pipeline 的四个可审计站点、下一张表把 evidence 落到五个层层不相互蕴含的评估层。这两张表是本文"从概念文章走到 benchmark protocol 文章"最直接的接口。

### 2.1 Semantic evidence $E_{\mathrm{semantic}}$：Invariance / Equivariance Test

对应 §1.1 Type I。给定一组已知 $T^{\mathcal{C}}_\pi$ 的 contract 变换（frame / coordinate / hypothesis permutation）、测：

$$\mathrm{Equiv}(\mathcal{C}) \;=\; \mathbb{E}_{\hat S}\!\left[d\!\left(\pi_\theta(T^{\mathcal{C}}\hat S),\; T^{\mathcal{C}}_\pi\,\pi_\theta(\hat S)\right)\right].$$

$\mathrm{Equiv} \to 0$ 是硬要求、$\mathrm{Equiv} \gg 0$ 意味着 $e_\pi$ 学坏了、或者 $q_\pi$ 直接把这一层 quotient 丢了。这一类是最"干净"的一类、因为规则是 mathematically defined 的、不需要 oracle 也不需要 $J$ 的定义。前篇 §3.2 Failure 2 的 severity 可以直接由 $\mathrm{Equiv}(\text{frame})$ 量化。

### 2.2 Representation evidence $E_{\mathrm{representation}}$：Conditional Probe + Interface-Compliance $E_{\mathrm{contract}}$ + Decision-Competence HPC + HSS（v6 二次拆分） + Calibration

**Conditional probe**（§1.1 类别 A 升级版）：Retention$_{\mathrm{cond}} = I(\text{field}; z_\pi \mid o)$——固定 raw observation $o$、测 $z_\pi$ 里还**独立**携带多少 contract 信息。这是**避免 image-proxy 泄漏**的必要形式。

**v5 已经把 $T_k^{\mathrm{hyp}}$ 拆成 $T_k^{\mathrm{contract}}$（raw obs 不变）与 $T_k^{\mathrm{world}}$（obs 与 contract 联合 re-render）两档、但 HPC 公式本身仍然是错的**——v6 reviewer 一句戳穿：

> **If the raw observation is held fixed and the contract is intentionally made inconsistent with it, what defines the ground-truth optimal action set $\mathcal{A}^{*}_k$?**

这个问题 v5 答不上来。$T_k^{\mathrm{contract}}$ 只改了 contract、世界并没有变——那么"world-consistent optimal action set"仍然是原世界的 $\mathcal{A}^*(O, W)$、不该因为你把 contract 换成 $H_k$ 就自动出现一个新的 $\mathcal{A}_k^*$。**v5 的 HPC 公式实际上偷偷假设了"改变 contract = 改变 world"**、恰好把 v5 花大篇幅拆开的两件事又揉在一起。v6 二次拆分——**HPC 不再同时承担两件事、拆成两个 metric**：

**(A) Contract-only intervention → Interface compliance $E_{\mathrm{contract}}$**——$T_k^{\mathrm{contract}}:\hat S \mapsto \hat S_k$、raw observation 固定不变、**不引入任何 $\mathcal{A}^*_k$ oracle**。它测的是：policy 在 contract 声明变化后、是否落在 §1.1 已经写进 $\mathcal{C}_\pi$ 的**声明响应关系** $\mathcal{R}_{\mathcal{C}}(T_k)$ 之内——与 §1.1 Type I equivariance / Type II order-constrained response 完全闭环：

$$\boxed{\;E_{\mathrm{contract}}(T_k) \;=\; D_{\mathcal{A}}\!\big(\pi_\theta(T_k^{\mathrm{contract}}(\hat S)),\;\mathcal{R}_{\mathcal{C}}(T_k)\big),\qquad E_{\mathrm{contract}}^{\mathrm{overall}} = \tfrac{1}{K}\sum_k \mathbf 1\!\big[E_{\mathrm{contract}}(T_k) > \epsilon_{\mathrm{resp}}\big].\;}$$

$D_{\mathcal{A}}$ 与 前篇 §0.2.2 的 $L_{\mathrm{decision}}$、§2.2 HSS 用同一族（action-equivalence-aware distance、测两个 action 分布是否**支持不同的 admissible / optimal action set**）。$\mathcal{R}_{\mathcal{C}}(T_k)$ 里允许 piecewise、允许 flat、允许 conditional、只要不违反 §1.1 声明的 $\preceq_{\mathcal{C}_\pi}^{\mathrm{declared}}$ 与 $r$ 就合法——**reviewer 追问"如果 contract 与 observation 自相矛盾、ground-truth action 从哪来"、这一版的回答是"不需要 ground-truth action、只需要 contract 自己声明过的响应集合"**。这才是 interface compliance 该有的样子。

**(B) World-consistent counterfactual → Decision competence HPC**——同时改 $(O, \hat S) \mapsto (O_k, \hat S_k)$、$O_k$ 与 $\hat S_k$ 都来自 simulator / renderer / privileged state、保证 observation 与 contract 一致。此时 $\mathcal{A}^*_k$ 有天然的定义——**它就是世界 $k$ 里的 optimal / admissible action set**、由 simulator 的 ground-truth state 与 reward 决定：

$$\boxed{\;\mathrm{HPC} \;=\; \frac{1}{K} \sum_{k=1}^{K} U\!\big(\pi_\theta(T_k^{\mathrm{world}}(\hat S, O)),\;\mathcal{A}^{*}_k\big),\qquad U(\pi, \mathcal{A}^*_k) = \Pr_{a \sim \pi}\!\big[a \in \mathcal{A}^{*}_k\big].\;}$$

HPC **只在 $T_k^{\mathrm{world}}$ 下有意义**——它测的是"policy 在另一个真实世界 hypothesis 下的 decision competence"、而不是"policy 会不会响应 contract 变化"。两件事由两个不同的 metric 承担、benchmark 报告时**必须并列 $E_{\mathrm{contract}}$ 与 HPC**、不能合并成一个总分。这一拆分正好与本文全篇"contract 变化 ≠ world 变化"的哲学一致。

**Hypothesis Separation Score (HSS)**——**必须只在 action-relevant hypothesis pairs 上平均**、reviewer 抓到了 gaming：如果 $\mathcal A^*(H_1) = \mathcal A^*(H_2)$、policy 输出不同不是优点、**是 noise**。定义 action-relevant pair set（走 前篇 §0.2.1 已经定义的 $\sim_{\pi,\mathcal D}$、不是"raw representation 不同"）：

$$\mathcal R \;=\; \big\{(i, j): \hat S_i \not\sim_{\pi,\mathcal D} \hat S_j\big\} \;\cap\; \big\{(i, j): T_i, T_j \text{ 都属于 } \{T^{\mathrm{contract}}, T^{\mathrm{world}}\} \text{ 中的一类}\big\}.$$

**第二个交集很关键**——$T^{\mathrm{contract}}$ 与 $T^{\mathrm{world}}$ 语义不同、不能混在同一个 HSS 里平均。HSS 只在 $\mathcal R$ 上算、**并且 $D$ 一律用 $D_{\mathcal A}$、不用普通 distribution distance**（KL、TV、Wasserstein 都可能被"每个 hypothesis 输出一个不同随机分布"这种 gaming 拉满）：

$$\mathrm{HSS}^{c} \;=\; \frac{1}{|\mathcal R^{c}|} \sum_{(i, j) \in \mathcal R^{c}} D_{\mathcal{A}}\!\big(\pi_\theta(\cdot \mid T_i^{c}\hat S),\;\pi_\theta(\cdot \mid T_j^{c}\hat S)\big),\qquad c \in \{\mathrm{contract},\mathrm{world}\}.$$

**separation is useful only when distinctions are decision-relevant**——这一句必须写死。**并且光靠 $D_{\mathcal A}$ 还挡不住"完全随机 policy 不同 hypothesis 下随机分布也不同"这种 gaming**、所以 v6 引入 **competence gate**：**HSS 只在 $E_{\mathrm{contract}}$ 已经通过（$E_{\mathrm{contract}}^{\mathrm{overall}} < \tau_E$）的前提下才作为 positive evidence**、否则高 HSS 只说明 policy 在乱动。**不要把两个 metric 硬乘成一个总分**（例如 $E_{\mathrm{contract}} \cdot \mathrm{HSS}$）——reviewer 更接受**并列报告**：

| 指标 | 问题 | 抗 gaming 手段 |
|---|---|---|
| $E_{\mathrm{contract}}^{\mathrm{overall}}$ | policy 是否落在声明过的响应集合内 | 用 $D_{\mathcal A}$、$D$ 与 $\mathcal R_{\mathcal C}$ 都对齐 §1.1 |
| $\mathrm{HPC}$ | world-consistent 下 policy 是否支持正确的 action set | 只在 $T^{\mathrm{world}}$ 上算、$\mathcal A^*_k$ 由 simulator 定 |
| $\mathrm{HSS}^{c}$ | policy 是否区分 action-relevant pairs | $D_{\mathcal A}$ + competence gate（$E_{\mathrm{contract}}$ 已 pass） |
| Entropy / diversity（对照） | 是不是只是随机化 | 与 HSS 并列报告、随机 policy 的 entropy 会异常高、HSS 会不涨 |

HPC / HSS / $E_{\mathrm{contract}}$ / entropy **四列并排**、reviewer 一眼能看出"高 HSS + 高 entropy"就是 gaming、"高 HSS + 低 entropy + $E_{\mathrm{contract}}$ 通过"才是真的 separation。

**四者合力才对应 decision-relevant semantic preservation（前篇 §0.2.1 Property A′）**——$E_{\mathrm{contract}}$ 保证"contract 变化时 policy 响应合法"、**HPC** 保证"world-consistent 下每个 action-relevant hypothesis 都被 support"、**HSS** 保证"policy 保留了 action-relevant 的 hypothesis distinction"、**entropy 对照**保证"HSS 不是随机化 gaming"、**同时不惩罚那些不该区分的 pair**。四条限制一起才让 representation evidence 有意义。

**Calibration diagnostic**——$\mathrm{ECE}_{\text{top-}k}$、NLL、Brier、calibration curve、coverage / credibility 五条并列（前篇 §4.1 已经把它们从 primitive 里剥出来、这一节是它们真正的家）。

### 2.3 Temporal slice：Staleness Response Compliance (SDS)（v5 记号 + baseline 双修）

**v4 版本 SDS 有两个问题（reviewer 都抓到了）**。第一、$R_\pi(a) = \pi_\theta(\cdot|\mathrm{do}(a_c = a), o)$ 里 $a$ 同时是 age 参数与全文里代表 action 的变量、notation collision。第二、$R^*(a)$ 作为"唯一 oracle response curve"过强——expert A 可能"age > 100 ms 就 fallback"、expert B 可能"通过 dynamics prediction 补偿后仍然正常控制"、两者都合理、强迫唯一 $R^*$ 会把合理的分段响应误判成 failure。

**v5 一次改两处**。记号侧：$\alpha$ 用作 age 值（前篇 §4.2 已经把字段名从 $a_c$ 改成 $\alpha_c$）、$a$ 从此只代表 action：

$$R_\pi(\alpha) \;=\; \pi_\theta\!\big(\cdot \,\big|\, \mathrm{do}(\alpha_c = \alpha),\, o\big).$$

baseline 侧：SDS 不再对着单一 $R^*$ 定义、而是对着 **contract-permitted response set** $\mathcal R_{\mathcal C}(\alpha)$ 定义——这个 set 由 §1.1 已经写进 $\mathcal C_\pi$ 的 $\preceq_{\mathcal C_\pi}^{\mathrm{declared}}$ 与 $r$ 声明、允许 piecewise、允许 flat、只要**不违反 relation 就是合法**：

$$\boxed{\;\mathrm{SDS} \;=\; D\!\big(R_\pi,\;\mathcal R_{\mathcal C}\big),\;}$$

$D$ 是 "distance from a curve to a set of legal curves"、可以是 sup-based violation measure $\sup_{\alpha_1 \preceq_{\mathcal C_\pi}^{\mathrm{declared}} \alpha_2} \max\!\big(0, r(R_\pi(\alpha_1)) - r(R_\pi(\alpha_2)) + \delta\big)$（等价于 §1.1 Type II hinge loss 在 evaluation 上的复用）、也可以是别的 curve-set divergence。**oracle policy curve $R^*$ 只是 $\mathcal R_{\mathcal C}$ 的一种 baseline、不是定义本身**。这样 SDS 就与 §1.1 Type II "order-constrained 不一定 monotone" 完全一致、不再强迫所有任务共用一条唯一正确的 staleness response。

**v6 补·SDS 的操作化定义（$V_{\mathrm{order}}$ violation estimator）**——上面 $D(R_\pi, \mathcal R_{\mathcal C})$ 是漂亮但抽象的记号、reviewer 一句 **"How do you compute distance to a set of legal curves?"** 就能把 benchmark 版打穿（$\mathcal R_{\mathcal C}$ 一般是无穷集）。v6 明确 SDS 的**推荐实现是 violation functional**、不是 curve-to-set distance：

$$\boxed{\;\mathrm{SDS} \;=\; V_{\mathrm{order}} \;=\; \mathbb E_{(\alpha_i, \alpha_j)\,:\,\alpha_i \preceq_{\mathcal C_\pi}^{\mathrm{declared}} \alpha_j}\!\Big[\max\!\big(0,\; r\!\big(R_\pi(\alpha_i)\big) - r\!\big(R_\pi(\alpha_j)\big) + \delta\big)\Big].\;}$$

这个式子是**有限可测**的：给定一组 grid $\{\alpha_1, \alpha_2, \ldots\}$、遍历所有 declared order 兼容的 pair、跑 policy 读出 $r(R_\pi(\alpha))$、算 hinge、平均即可。$\delta$ 与 §1.1 Type II hinge loss 里的 $\delta$ **取同一个值、由任务级原则选取**（例如 $\delta = \sigma_r / 2$、$\sigma_r$ 是 $r$ 在 benchmark noise 下的经验标准差；或者 $\delta$ 与 §2.2 $E_{\mathrm{contract}}$ 的 $\epsilon_{\mathrm{resp}}$ 通过同一族 action-distance 校准）——不能只是叫 "margin"、不能事后凑。

如果 §1.1 Type II 声明的是 **piecewise policy**（例如 $\alpha<50$ → normal、$50\le\alpha<100$ → fallback、$\alpha\ge 100$ → stop），那么 $V_{\mathrm{order}}$ 直接测**状态 transition violation**就够了：

$$V_{\mathrm{trans}} = \mathbb E\!\big[\mathbf 1\big[\text{observed state}\big(R_\pi(\alpha)\big) \neq \text{declared state at }\alpha\big]\big].$$

正文里保留 $D(R_\pi, \mathcal R_{\mathcal C})$ 是**理论定义**、实验实现一律走 $V_{\mathrm{order}}$ 或 $V_{\mathrm{trans}}$、这样 benchmark paper 版可以直接把 estimator 抄过去。

响应属性 $R$ 根据任务定义、可以取 **variance / action norm / fallback probability / safety margin / stop probability**——**必须与 §1.1 Type II 里声明的 $r$ 是同一个**、否则训练与评估各说各话。平坦不代表差、只要与 $\mathcal R_{\mathcal C}$ 里某一条 legal curve 匹配即可。

导数形式仍保留、作为 response shape 的一种局部刻画：

$$\left.\frac{\partial\, \mathbb{E}\!\big[\pi_\theta(\cdot \mid \mathrm{do}(\alpha_c = \alpha),\, o)\big]}{\partial \alpha_c}\right|_{\alpha}\quad\text{与 } \mathcal R_{\mathcal C} \text{ 里 legal curves 的同阶导数分布比较}.$$

### 2.4 Source slice：$\Delta J_{\mathrm{where}}$、$\Delta J_{\mathrm{dep}}$、$\Delta J_{\mathrm{neg}}$（v5 三分）

**v4 把 $\Delta J_{\mathrm{prov}}$ 写成 "provenance + correlated_with" 一起 ablate、reviewer 抓得对——你花了一整节说 provenance ≠ dependency ≠ negative evidence、metric 又把它们揉在一起、那就回答不了"到底哪一条 primitive 起作用"**。v5 拆成三条、每条只 ablate 一个 primitive：

$$\Delta J_{\mathrm{where}} \;=\; J\!\big(\pi_\theta \mid \text{provenance}\big) \;-\; J\!\big(\pi_\theta \mid \text{provenance} = \varnothing\big),$$

$$\Delta J_{\mathrm{dep}} \;=\; J\!\big(\pi_\theta \mid \text{correlated\_with}\big) \;-\; J\!\big(\pi_\theta \mid \text{correlated\_with} = \varnothing\big),$$

$$\Delta J_{\mathrm{neg}} \;=\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda\big) \;-\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda = \varnothing\big).$$

三条各对应 前篇 §4.3 的三个 primitive：**$\Delta J_{\mathrm{where}}$ 测 `contributing_mask` 消费情况、$\Delta J_{\mathrm{dep}}$ 测 `correlated_with` 消费情况、$\Delta J_{\mathrm{neg}}$ 测 conditional LLR 消费情况**。$J$ 一律 higher-is-better。每条都可以独立爆零、互不顶替——这是"三个 primitive 语义独立"这一节 claim 在**评估层的对应兑现**。如果 reviewer 再问"性能改善是哪一个 primitive 起作用"、v5 可以一条条回答、v4 不能。

### 2.5 Decision evidence $E_{\mathrm{decision}}$：Contract Ablation Gap（CAG）+ Oracle baseline + Retraining protocol（v5 加限定）

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

**v6 补·matched null control**（reviewer 抓得非常狠的一步）——上面的 $\mathrm{CAG}^{\mathrm{fixed}}$ 有一个**极容易被误解读**的地方：$\mathrm{collapse}_X$ 之后 policy 拿到的输入很可能**已经不在训练分布里**。比如训练时 hypothesis payload 长这样：

```text
hypothesis = [(μ1, Σ1, w1), (μ2, Σ2, w2), ...]
```

collapse 后变成：

```text
collapse_hyp → 单个 Gaussian (μ̄, Σ̄)
```

那么 policy 遇到的**未必是**"contract information 被拿掉"、也可能只是**"输入格式突然变成模型没见过的东西"**——于是 $\mathrm{CAG} > 0$ 可能只是 **OOD sensitivity**、跟"contract 语义有没有被用"没关系。这一版必须给 CAG 加一个 **matched null control**：

**Format-preserving null intervention** $S \mapsto \tilde S$ 满足四条：

- shape 与 collapsed version 相同（比如同样折成"一个 Gaussian"）；
- marginal distribution 与 $S$ 相同（比如 $\tilde\mu$ 从原 mixture 里按权重随机取一个 component）；
- **decision-relevant information 不变**（比如保留 top-1 hypothesis 的 mode identity）；
- **irrelevant semantics 改变**（比如把 hypothesis identity 的 index permutation、把 component 的 provenance tag 打个随机重排、保留 shape 与 marginal）。

跑同一份 $\theta^*$、得到 $\Delta J_{\mathrm{null}} = J_{\mathrm{full}}^{\theta^*} - J_{\mathrm{null}}^{\theta^*}$。**benchmark 报告 CAG 的时候必须并排 $\Delta J_{\mathrm{null}}$**：

$$\boxed{\;\text{只有当 } \Delta J_{\mathrm{contract}} \gg \Delta J_{\mathrm{null}} \text{ 时、CAG 才能作为 contract-use 的证据；否则它只是 OOD sensitivity}.\;}$$

这一步把"policy 只是讨厌输入格式变化"这个 alternative hypothesis 从 CAG 里剥出去。一个具体的对照实验设计（v6 reviewer 建议 §2.5 的 CAG shortcut detection 应该配的）：**打乱 `age` 与 `task difficulty` 的相关性重训一份对照 policy**、然后观察 §1.1 Type I/II pass/fail 与 CAG 之间的关系是否改变——如果 correlation-shuffled 之后 CAG 归零、那原来的 CAG 就是 shortcut；如果 CAG 仍在、才是真的在读 contract semantic。

**再加 oracle privileged-state baseline**：

$$J_{\mathrm{oracle}} \;=\; J(\pi^{*}_{\mathrm{oracle}} \mid s^{\mathrm{priv}}), \qquad \mathrm{Gap}_{\mathrm{oracle}} \;=\; J_{\mathrm{oracle}} - J_{\mathrm{full}}.$$

**v4 的四象限解释表其实略强了**——"CAG 高 + oracle 略高 → policy 真在读 contract"这句话数学上不支持。$\mathrm{CAG} > 0$ 只说明**去掉这个结构以后 task utility 下降**、不说明 policy 是通过正确 semantic mechanism 使用它。真实情况常常是训练数据里 `age ↔ task difficulty` 高度相关、policy 学到 $a = f(\alpha)$ 的 shortcut 而不是 $a = f(\text{actual sensor staleness semantics})$。这一版把表格降级成 **"每个观察能支持什么"**：

| 观察 | 能说明 |
|---|---|
| CAG 高 | contract structure 对当前任务有 utility（可能来自 semantic use、也可能来自 shortcut） |
| CAG ≈ 0 | 当前任务下该 contract structure 可能不必要 |
| CAG 高 + §1.1 Type I/II 通过 | **更有证据**表明 semantic use |
| CAG 高 + intervention fail | **很可能是 shortcut**——回到 前篇 §4.3 / §1.2 检查 training distribution |
| CAG ≈ 0 + oracle gap 高 | contract 里有信息、policy 没充分利用——回到 前篇 §0.2.2 的三个 loss 定位是哪一段掉了 |

也就是：

$$\boxed{\;\mathrm{CAG} \;\neq\; \text{contract understanding}.\;}$$

$$\boxed{\;\mathrm{CAG} \;=\; \text{task-conditional utility sensitivity}.\;}$$

**CAG 是 decision 层的聚合、必须与 $E_{\mathrm{semantic}}$（invariance / equivariance）+ $E_{\mathrm{representation}}$（conditional probe + HPC/HSS）+ §1.1 Type I/II controlled response 联合看**、才能构成 semantic use 的证据。这一版把 CAG 从"总指标"降回"decision 层聚合"、四种 compliance evidence 才完整。

### 2.6 Safety evidence $E_{\mathrm{safety}}$：Constraint certification intervention（v5 三态）

在部署 / 半仿真环境里主动把 §1.3 的 $\mathrm{certification}_j$ 打到 **unsafe** 或 **unknown**（例如注入 calibration drift、把 observability 关掉、把 $\Lambda(E^-; H, \mathcal O)$ 拉高）、看 safety filter 是否**在正确的时刻进入正确的 guardrail**、以及 guardrail 触发是否可归因到 certification 的哪一项。**关键判定不再是"filter 是否 stopped policy"、而是"filter 是否收紧了 constraint"**（§1.3 已经写死：invalid evidence alone cannot justify relaxing constraint）。测的三种正确反应是 fallback、conservative tightening、stop；测的**错误反应**是"在 evidence 不充分的条件下 relax"。观测到"unknown 或 invalid 单靠自身导致 relaxing"、$E_{\mathrm{safety}}$ 直接 fail。反过来、如果另一路独立 valid evidence（例如 lidar）已经把 certification 推到 **safe**、filter 允许 relax 是**正确反应**、v4 会把这种情形误判、v5 因为引入了三态所以不会。这一类是 §1.3 接口的直接对应、也是整篇 interface 主张真正**能不能落地**的测试。

四类合起来构成一个**多证据 compliance argument**：**semantic 测"响应规则对不对"、representation 测"信息还在不在（且独立于 raw observation）"、decision 测"用了没 / 有没有 shortcut"、safety 测"unknown / invalid 时收紧没"**。它们都**不能替代**任何端到端 success rate——它们衡量的是 policy 侧对 contract 的**读取度**、不是**表现力**。这一点与 前篇 §0.3 Claim 3 完全对齐：**contract compliance must be tested by controlled intervention、并且必须由四类证据合流支持**。

## 3. 最小可执行接口草图（Python skeleton · v6 修复版）

把 前篇 §4 三族 primitives 与 §2 四种 compliance evidence 合起来写成一个 Python 类骨架。**不是要给出一个具体 policy、是要给一个可读的接口约定**。

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
        # v6 P1-10: fail-closed is actually enforced, not just recorded.
        if not self.compatibility.accepted:
            raise SchemaCompatibilityError(
                f"schema_version={contract.schema_version} incompatible with "
                f"supported_version={consumer_contract.supported_version}; "
                f"either upgrade C_pi or provide an explicit adapter."
            )
        self.contract = contract
        self.C_pi = consumer_contract
        self.Q = query_family
        self.Q_req = required_queries

    def declared_coverage_loss(self) -> float:
        # v6 P1-4: L_declared via query subsumption, not literal set membership.
        # A required q is covered if ∃ q' ∈ Q_C_pi such that q' ≽ q.
        def covered(q):
            return any(self.Q[q2].subsumes(q) for q2 in self.Q)
        return sum(self.Q_req[q].weight for q in self.Q_req if not covered(q))

    def project(
        self,
        schema: PolicySchema,
        # --- mode_select (mass-preserving, not calibration) ---------
        mode: Literal["map", "posterior_sample", "topk"] = "topk",
        # v6 P1-11: two ORTHOGONAL knobs (previously conflated in `topk_residual`)
        topk_weight_mode: Literal["conditional", "raw"] = "conditional",
        residual_mode: Literal["drop", "mass_only", "sufficient_stats"] = "mass_only",
        # --- staleness / uncertainty knobs --------------------------
        staleness: Literal["ignore", "parallel_field", "condition"] = "parallel_field",
        uncertainty: Literal["none", "conservative_inflation", "propagate"] = "conservative_inflation",
        # --- source-structure knobs ---------------------------------
        provenance: Literal["ignore", "harden"] = "harden",
        dependency: Literal["ignore", "logit_bias_learned",
                            "covariance_fusion", "hierarchical_mixture"] = "covariance_fusion",
        negative_evidence: Literal["ignore", "condition", "belief_update"] = "condition",
        # --- v6 P1-12: benchmark-side RNG control -------------------
        benchmark_rng: Optional[np.random.Generator] = None,
    ) -> PolicyInput:
        """
        Project StructuredState (upstream, 9/14) to PolicyInput (downstream, this piece).
        The (C_pi, Q_C_pi) pair literally defines ~_pi; q_pi is the induced quotient map.
        e_pi is the encoder of the specific backbone and must not silently drop anything
        q_pi declared preserved; pi_theta may further collapse distinctions at the
        action level (see 前篇 §0.2.2: three semantic losses + one safety obligation).

        Benchmark note (v6 P1-12): when `mode='posterior_sample'` is used inside an
        intervention benchmark (T_i vs T_j), caller MUST pass a shared `benchmark_rng`
        so both arms consume the same random numbers. Otherwise the observed distance
        D(π(T_i S), π(T_j S)) mixes intervention effect with sampling noise and
        contaminates HSS / SDS / E_contract.
        """
        slots = {}
        for field_name in schema.fields:
            h = self.contract[field_name]   # hypothesis set: [(mu_i, Sigma_i, w_i)]

            # --- mode_select: mass-preserving top-k (NOT calibration-aware) ---
            if mode == "map":
                mu, Sigma, w_payload, residual = h.most_likely().mu, h.most_likely().Sigma, None, None
            elif mode == "posterior_sample":
                rng = benchmark_rng or h.default_rng   # v6: shared RNG in benchmark mode
                sample = h.sample(rng=rng)
                mu, Sigma, w_payload, residual = sample.mu, sample.Sigma, None, None
            else:  # "topk"
                top = h.top_k(k=schema.k_per_field[field_name])
                residual_mass = 1.0 - top.total_weight()
                # v6 P1-11: weight handling and residual handling are ORTHOGONAL.
                if topk_weight_mode == "conditional":
                    w_payload = top.renormalize()          # sum w̃_i = 1 within top-k
                else:  # "raw"
                    w_payload = top.weights                # raw weights kept
                # residual_mode expresses what we DO with the discarded mass:
                if residual_mode == "drop":
                    residual_declared = None               # explicitly dropped, not silently
                elif residual_mode == "mass_only":
                    residual_declared = residual_mass      # aggregate residual mass only
                else:  # "sufficient_stats"
                    residual_declared = h.residual_sufficient_stats()  # full μ/Σ/likelihood
                # NOTE: mass_only ≠ sufficient_stats. Strict Bayesian update over the residual
                # component requires separate mu/Sigma/likelihood specifications (see 前篇 §4.1).
                # Calibration (ECE_top-k / NLL / Brier) is measured in §2.2, NOT here.

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
                residual_declared=(residual_declared if mode == "topk" else None),
            )

        # --- provenance / dependency / negative evidence, three reads ---
        prov_mask = self.contract.contributing_mask if provenance == "harden" else None
        if dependency == "logit_bias_learned":
            # b(R_ij) is learned and CAN be positive or negative.
            # Convenient implementation candidate on transformer backbones;
            # NOT the canonical default of dependency_aware_fusion (前篇 §4.3.2 v5).
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
    """v6: safety is not a fourth information loss; it is a distinct obligation slot.
    Reads three-state certification, NOT a binary validity bit.
    Multi-constraint combination is INTERSECTION, not short-circuit on first unsafe/unknown.
    """
    def __init__(self):
        super().__init__()

    def forward(self, a_proposed, contract) -> Action:
        # g_safety = ⋂_j g_j — every constraint contributes its own allowed subset,
        # we compose them; we never `return` early on the first tightening.
        a_applied = a_proposed
        for j, constraint in enumerate(contract.constraints):
            cert = constraint.certification      # ∈ {safe, unsafe, unknown}
            if cert == "safe":
                # safe → relaxation is PERMITTED, not required: constraint policy
                # still applies at its normal margin. Do NOT `continue` past it.
                a_applied = allow_normal_margin(a_applied, constraint)
            elif cert == "unsafe":
                a_applied = tighten_or_stop(a_applied, constraint)
            elif cert == "unknown":
                # invalid evidence ALONE cannot justify relaxing the constraint.
                a_applied = conservative_fallback(a_applied, constraint)
        return a_applied


class ContractAwarePolicy(nn.Module):
    def __init__(self, backbone, schema: PolicySchema, cfg: ContractReadConfig,
                 consumer_contract: ConsumerContract, query_family: QueryFamily,
                 required_queries: QueryFamily):
        super().__init__()                 # v6 P1-10: nn.Module init (was a real bug)
        self.backbone = backbone
        self.schema = schema
        self.cfg = cfg                 # cfg is a *materialization* of q_pi,
        self.C_pi = consumer_contract  # but the actual declared quotient lives in (C_pi, Q)
        self.Q = query_family
        self.Q_req = required_queries
        # three semantic loss sites + one safety obligation site:
        #   L_declared     — on (C_pi, Q, Q_req): coverage via query subsumption
        #   L_projection   — on Π_π output Z_π: I(Y_C^π; Ŝ | Z_π, O, L)
        #   L_decision     — on π_θ: E_{R_D}[ 1[D_A(π(.|S), π(.|S')) < ε] ]
        #   O_safety       — on g_safety: safe→allow_normal_margin /
        #                                     unsafe→tighten_or_stop /
        #                                     unknown→conservative_fallback

    def forward(self, state: StructuredState, obs, lang) -> ActionDistribution:
        x = StructuredStateView(state, self.C_pi, self.Q, self.Q_req).project(
            self.schema, **self.cfg.as_kwargs())
        return self.backbone(x, obs, lang)
```

三条 caveat 明确写死：

- **(i)** 这不是唯一读法、**`cfg` 是本文 前篇 §0.2 里的 $q_\pi$ 的一个 materialization、但真正的 declared quotient 是 $(\mathcal C_\pi, Q_{\mathcal C_\pi})$ 这一对**——同一个 contract、VLA 与 Diffusion Policy 的 `cfg` 就该不一样、engineered state head 与 visual latent head 的最优 `uncertainty` 也不同。接口文档里必须**贴出一张 $Q_{\mathcal C_\pi}$ 清单**、否则 $q_\pi$ 又退化成 encoder 里的隐式行为。**schema 兼容性也必须 fail-closed 或走显式 adapter**（前篇 §0.2.3）、不然 $\mathcal C$ 升到 v2 会静默丢字段、正落入本文批评的 undeclared semantic loss。
- **(ii)** 这个接口**只解决输入端**；§1.1 的三类 intervention 约束、§1.2 的 observed-metadata-only aug、§1.3 的三态 certification 直连——如果一处不改、$(\mathcal C_\pi, Q_{\mathcal C_\pi})$ 写得再漂亮也会被 $\pi_\theta$ 训练动力绕过去（$L_{\mathrm{decision}}$ 直接爆）。**三个 loss 加一个 safety obligation 缺一个都守不住接口**。
- **(iii)** `dependency="logit_bias_learned"` 在 transformer 上是一种**便捷实现候选、不是 canonical default**——因为 field-graph → token-graph 的编译问题（$R_{\text{field}} \to R_{\text{token}}$）本身还没解决（前篇 §4.3.2 v5 caveat）。MLP head 走 `covariance_fusion`、Kalman / factor-graph 融合走 `covariance_fusion`、grouped latent 走 `hierarchical_mixture`——**真正 recommended 的是 dependency_aware_fusion 这条 primitive、不是它的某种具体 realization**。

## 4. 收束：四条 compliance evidence、五层 evaluation hierarchy、一个可落地的接口

前篇（9/15）已经立住"policy 是 contract consumer"这一 thesis 的语义层——Consumer Contract 三段分解、三档 preservation、三个 semantic losses + 一个 safety obligation、四种 interface-mismatch 失败模式、三族 contract-read primitives。本文（9/16）承担的是它的**度量层与实现层**——四类 compliance evidence、五层 evaluation hierarchy、最小可执行接口骨架。两半合起来才把"policy 侧接口"这一整块从**概念**推到**协议**。

本文的三条收束 claim 与 §0.3 对齐：

> **Claim A · Compliance is a multi-evidence argument, not a score.** §2 四类 evidence——semantic / representation / decision / safety——各自回答不同问题、**不能互相顶替**、**不能压成一个 scalar**。$E_{\mathrm{semantic}}$ 测"响应规则对不对"、$E_{\mathrm{representation}}$ 测"信息还在不在（且独立于 raw observation）"、$E_{\mathrm{decision}}$ 测"用了没 / 有没有 shortcut"、$E_{\mathrm{safety}}$ 测"unknown / invalid 时收紧没"。这四类都不能替代端到端 success rate——它们衡量的是 policy 侧对 contract 的**读取度**、不是**表现力**。

> **Claim B · Evaluation layers do not imply each other.** §2.0 五层 hierarchy **Retention ≠ Sufficiency ≠ Behavioral use ≠ Utility ≠ Safety** 的四条不蕴含关系是 §2.2 拆 $E_{\mathrm{contract}}(T^{\mathrm{contract}})$ / HPC($T^{\mathrm{world}}$)、§2.5 拆 CAG$^{\mathrm{fixed}}$ / CAG$^{\mathrm{retrained}}$、§2.6 拆 constraint certification intervention 的根本动机。任何把某一层 score 顶到另一层的做法都是 shortcut。

> **Claim C · The interface is where the bugs hide, not the architecture.** §3 Python 骨架 v6 修的三处 bug——`super().__init__()` 缺失、fail-open schema 兼容检查、top-$k$ API 把两个正交决策（`topk_weight_mode` / `residual_mode`）揉成一个字符串——都不是模型架构问题、都是**接口层问题**。这类 bug 会静默地把 contract 语义压平、loss 曲线不响、eval score 不响、只能靠 §2 的四类 evidence 合流才能测出来。

### 4.1 一张收束图

$$\boxed{\;\mathcal C\;\longrightarrow\;(\mathcal C_\pi=(Q_\pi,\mathcal O_\pi,V_\pi),\;Q_{\mathcal C_\pi})\;\longrightarrow\;q_\pi\;\longrightarrow\;\Pi_\pi\;\longrightarrow\;\pi_\theta\;\longrightarrow\;g_{\mathrm{safety}}=\textstyle\bigcap_j g_j\;\longrightarrow\;\mathcal B_{\mathcal C}\;}$$

$$\boxed{\;\text{Compliance Evidence}=\big\{E_{\mathrm{semantic}},\;E_{\mathrm{representation}},\;E_{\mathrm{decision}},\;E_{\mathrm{safety}}\big\}\;\longrightarrow\;\text{Five-Layer Evaluation Hierarchy}\;}$$

前篇的 boxed thesis——**A policy is a contract consumer, not merely a function approximator**——在本文的兑现是：一个 contract consumer 不仅要能回答前篇四问（what may I discard / what did I retain / how should decisions respond / what happens under invalid-or-unknown evidence），还要能被**四类证据分别审计**、**五层 hierarchy 层层不蕴含**地评估、**Python 骨架里的每一个接口 bug** 都能被 §2 的某个 evidence 定位到。**接口对了、模型选择才是次要问题**；接口不对、任何 backbone 都会静默把 contract 语义压平。

### 4.2 下一步：Contract-Preserving Policy Benchmark

本文与 9/15 前篇合起来把这一系列推到"下一篇文章就能立 benchmark"的位置：

$$\textbf{Contract-Preserving Policy Benchmark}:\;\text{给定同一份 Structured State Contract、如何系统地测 SAC / PPO / Diffusion Policy / ACT / OpenVLA / }\pi_0\text{ 是否 contract-compliant.}$$

每一个 loss 与 obligation 各自对应 §2 的一类可测量、每一个 pipeline 站点各自对应 §2.0 表格的一行、每一个 §3 的接口 bug 各自对应 §2 的一个 counterexample 测试。benchmark 侧还需要明确的**三件事**：$\tau_E$ competence gate 阈值的跨任务设定原则、§2.3 hinge loss $\delta$ 与 §2.5 CAG matched null 的 $\delta$ 的一致任务级选取、§3.5 shortcut detection 的具体对照实验（打乱 observation age 与 task difficulty 之间的相关性）。这三件事本文不做、留给 benchmark 篇。

一句收束：**VLA / Diffusion / Flow / ACT / SAC / PPO 都只是实现坐标、不再是理论分类**——真正决定一个 policy 能不能上生产的是它的接口对不对、能不能被 §2 的四类 evidence 合流审计通过、能不能在 §2.0 五层 hierarchy 里逐层给出可测量的响应。前篇把 interface 立起来、本文把 protocol 与 skeleton 立起来；下一篇文章把 benchmark 立起来。

## Sources

以下 arXiv ID 已联网核过；journal-only 引用不贴 arXiv。按支撑的 section 分组。前篇（9/15）的 Sources A · VLA / B · Diffusion / D · SAC-PPO 在本文不重复列举——本文只列 evaluation 与 implementation 侧新引入的参考文献。

### C · 不确定性、校准与 belief-space 参照（支撑 §2.2 calibration diagnostic、§2.5 CAG oracle baseline）

- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)（现代网络过度自信、temperature scaling 起点 · §2.2 calibration diagnostic 的评估依据、不挂在 primitive 上）
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels*（PlaNet / RSSM）, ICML 2019 · [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)（deterministic + stochastic latent · §2.2 conditional probe 与 §2.5 CAG oracle baseline 的一条相邻参照路线）
- Hafner et al., *Mastering Diverse Control Tasks through World Models*（DreamerV3）, Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)（离散 + 连续混合 latent、KL balancing · §2.2 HPC($T^{\mathrm{world}}$) counterfactual-intervention 的一条相邻路线；注意：本文对 DreamerV3 的引用是 **evaluation reference**、不是说 DreamerV3 已经实现 contract-aware 接口）

### E · 承接前文与政策侧接口前篇（本文的 framework 输入与 evaluation spine 输入）

- 本博客《契约立起来之后：VLA、Diffusion Policy、π0 到底在消费什么》· `/zh/articles/2026-09-15-policy-side-interface/`（**本文前篇**、建立 §0.2 复用 pipeline 的全部 framework：Consumer Contract 三段分解、三档 preservation、三个 semantic losses + 一个 safety obligation、四种 interface-mismatch 失败模式、三族 contract-read primitives；本文 §1–§3 直接在前篇 framework 上做 evaluation 与 implementation）
- 本博客《拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口》· `/zh/articles/2026-09-14-multimodal-fusion-interface/`（Structured State Contract 定义、Interface Property Benchmark、degradation chain · 前篇 §0.2 $q_\pi$ 与本文 §1 augmentation、§2 evidence 建立在其上）
- 本博客《只会看、不会摸：机器人为什么缺一双"手感"的手》· `/zh/articles/2026-09-13-tactile-force-sensing/`（力 / 触觉的四种控制范式、Closed-loop value · §2.4 $\Delta J_{\mathrm{where/dep/neg}}$ source 切片的物理动机、§3 safety filter 三态化的一个具体现场）
- 本博客《具身智能 Sim-to-Real 方法论（三）》· `/zh/articles/2026-09-12-sim-to-real-evaluation-protocol/`（三级证据层、decision utility 三维、allocation protocol · **本文 §2 evaluation spine 直接沿用**）
- 本博客《具身智能 Sim-to-Real 方法论（一）》· `/zh/articles/2026-09-10-sim-to-real-methodology/`（allocation state $s_t=(b_t,\pi_t,q_t,h_t)$、$\Delta_{\mathrm{queue}}$ vs $\Delta_{\mathrm{processing}}$ · 前篇 §2/§4.2 定义直接接上、本文 §2.3 SDS 的 temporal slice 与 §1.2 Type II order-constrained response 沿用其 staleness 语言）

### F · Evaluation protocol 参照（本文 §1.1 / §2 / §3 用到的评估方法学参照）

- Osband et al., *What Type of Code is Relevance in Randomized Control Trials?*, 2018 · [arXiv:1802.08670](https://arxiv.org/abs/1802.08670)（**evaluation methodology** 参照、本文 §2.5 matched null control 与 §2.2 competence gate 借用其"null intervention"思想、**不宣称本文与 RCT 在因果结构上等价**）
- Adebayo et al., *Diagnostic Explanations for Deep Learning Models*, ICML 2018 · [arXiv:1806.07537](https://arxiv.org/abs/1806.07537)（**probing-based diagnostic** 参照、本文 §2.2 conditional probe 与其同族、但本文的 probe 是**在 $o$ 条件上的 residual MI**、不是原始 probing classifier accuracy）
- D'Amour et al., *Underspecification Presents Challenges for Credibility in Modern Machine Learning*, JMLR 2021 · [arXiv:2011.03395](https://arxiv.org/abs/2011.03395)（**evaluation underspecification** 参照、本文 §2.0 五层 hierarchy 与四条不蕴含关系是对"单一 score 隐含假设"这一 failure mode 的直接回应）

> **注意**：本文对 π0 / OpenVLA / Diffusion Policy 家族的引用是**前篇的分析性引用**（interface critique）；本文不重复这些引用、只引 evaluation 与 implementation 侧方法学参照。前篇的 A / B / D 三块 Sources 在本文一律不重复列举。

---

> **相关阅读**
>
> - [契约立起来之后：VLA、Diffusion Policy、π0 到底在消费什么](/zh/articles/2026-09-15-policy-side-interface/)——**本文前篇**、policy 侧接口的 framework 篇
> - [拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口](/zh/articles/2026-09-14-multimodal-fusion-interface/)——Structured State Contract 上游定义、本文与 9/15 共同的上游交付物
> - [只会看、不会摸：机器人为什么缺一双"手感"的手](/zh/articles/2026-09-13-tactile-force-sensing/)——力 / 触觉侧、§2.4 source slice 与 §3 safety filter 三态化的物理动机
> - [具身智能 Sim-to-Real 方法论（三）](/zh/articles/2026-09-12-sim-to-real-evaluation-protocol/)——本文 §2 evaluation spine 的直接来源
> - [具身智能 Sim-to-Real 方法论（一）](/zh/articles/2026-09-10-sim-to-real-methodology/)——staleness / queue vs processing 的语言来源
> - [VLA 与世界模型：两条路线的分岔与合流](/zh/articles/2026-09-07-vla-world-models/)——§2.2 HPC($T^{\mathrm{world}}$) 与 world-model-side oracle 的宏观背景
