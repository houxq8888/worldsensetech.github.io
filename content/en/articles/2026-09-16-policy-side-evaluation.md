---
title: 'Policy-Side Evaluation (Part 2): How to Test, How to Train, How to Land — Four Compliance Evidence Types and a Minimal Executable Interface'
slug: "2026-09-16-policy-side-evaluation"
date: 2026-09-16
draft: false
categories: ["Embodied AI", "Policy Learning"]
tags: ["Embodied AI", "Policy Learning", "Structured State Contract", "Consumer Contract", "Contract Consumer", "Compliance Evidence", "Five-Layer Evaluation", "Separately Auditable Failure Sites", "Interface Compliance Metric", "Matched Null Control", "Consumer-Declared Order", "Multi-Constraint Intersection", "Contract-Read Primitives", "Intervention Consistency", "Equivariance", "Order-Constrained Response", "Conditional Log-Likelihood Ratio", "Dependency-Aware Fusion", "Mass-Preserving Top-k", "Constraint Certification", "Contract-only Intervention", "World-consistent Counterfactual", "Action-Relevant Separation", "Contract Ablation Gap", "Fixed Policy vs Retrained Policy", "Evaluation Metrics"]
description: 'This piece is the **lower half** of the policy-side interface discussion. The upper half (9/15 [After the Contract Stands](/en/articles/2026-09-15-policy-side-interface/)) built the framework — Consumer Contract triple $\mathcal C_\pi=(Q_\pi,\mathcal O_\pi,V_\pi)$, three tiers of preservation, three semantic losses + one safety obligation, four interface-mismatch failure modes, three families of contract-read primitives. This piece answers the dual question: **given the framework, how to test, how to train, and how to land it**. Core boxed thesis: **Contract compliance should be tested by controlled intervention, not inferred from end-to-end success** — end-to-end success measures whether a policy is useful, not whether it reads contract semantics correctly. Compliance needs four evidence types (**$E_{\mathrm{semantic}}$ / $E_{\mathrm{representation}}$ / $E_{\mathrm{decision}}$ / $E_{\mathrm{safety}}$**, not four metrics, cannot be compressed into a single scalar) plus a **five-layer evaluation hierarchy** (**Retention ≠ Sufficiency ≠ Behavioral use ≠ Utility ≠ Safety**, four non-implications) that §2.0 opens with as the roadmap of this piece. §1 covers training-side knock-ons — representation-side probe, three tiers of intervention consistency (Type I equivariance / Type II consumer-declared order / Type III unconstrained), augmentation via degradation as a causal operator, and §1.3 three-state safety-filter constraint certification with **safe → relaxation permitted (not required)** / **unsafe → tighten or stop** / **unknown → conservative fallback**, composed via $g_{\mathrm{safety}}=\bigcap_j g_j$. §2 expands the four evidence types — §2.2 v6 splits HPC a second time into $E_{\mathrm{contract}}(T^{\mathrm{contract}})$ (interface compliance, checked against $\mathcal R_{\mathcal C}$, no $\mathcal A^*_k$ oracle) and HPC($T^{\mathrm{world}}$) (decision competence, $\mathcal A^*_k$ pinned by simulator); HSS uses $D_{\mathcal A}$ and passes a **competence gate**, reported in parallel with entropy to prevent gaming; §2.3 SDS gets an **operational violation estimator** $V_{\mathrm{order}}=\mathbb E[\max(0,r(R_\pi(\alpha_i))-r(R_\pi(\alpha_j))+\delta)]$ with $V_{\mathrm{trans}}$ for piecewise cases; §2.5 CAG gains a **matched null control** — only when $\Delta J_{\mathrm{contract}}\gg\Delta J_{\mathrm{null}}$ does the ablation count as contract-use. §3 is the v6 Python skeleton, fixing `super().__init__()`, fail-closed `raise SchemaCompatibilityError`, splitting top-$k$ into two orthogonal knobs `topk_weight_mode` and `residual_mode`, and adding `benchmark_rng` for common random numbers. §4 closes on three lower-half claims (multi-evidence argument / five-layer non-implications / the interface is where the bugs hide) and hands the next step to a Contract-Preserving Policy Benchmark.'
toc: true
related_articles:
  - 2026-09-15-policy-side-interface
  - 2026-09-14-multimodal-fusion-interface
  - 2026-09-13-tactile-force-sensing
  - 2026-09-12-sim-to-real-evaluation-protocol
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-08-data-and-training-recipes
---

> This piece is the **lower half** of the policy-side interface discussion. The upper half [After the Contract Stands: What Do VLA, Diffusion Policy and π0 Actually Consume?](/en/articles/2026-09-15-policy-side-interface/) built the framework — **Consumer Contract triple** $\mathcal C_\pi=(Q_\pi,\mathcal O_\pi,V_\pi)$, three tiers of preservation, three semantic losses + one safety obligation, four interface-mismatch failure modes, three families of contract-read primitives. This piece answers its dual: **given that framework, how to test, how to train, and how to land it**.

> In one sentence: **Contract compliance should be tested by controlled intervention, not inferred from end-to-end success** — end-to-end success measures whether a policy is useful, not whether it reads contract semantics correctly. Compliance needs **four evidence types converging** (semantic / representation / decision / safety), **an oracle baseline pinning the semantic range of each metric**, and **an explicit retraining protocol**. That sentence is Claim 3 in the upper half §0.3; it is the main thesis here.

This piece does not repeat the full framework derivation from the upper half. All $\mathcal C_\pi / q_\pi / \Pi_\pi / \pi_\theta / g_{\mathrm{safety}}$, three tiers of preservation, $Y_{\mathcal C}^{\pi}$, $L_{\mathrm{declared}} / L_{\mathrm{projection}} / L_{\mathrm{decision}}$ and $\mathcal O_{\mathrm{safety}}$ are defined in the upper half §0; this piece keeps only what the evaluation and implementation sections need, and pins the reference with a boxed pipeline figure at §0.2.

## 0. Framing of this piece: four compliance evidence types, a five-layer evaluation hierarchy, one minimal executable interface

### 0.1 How this piece splits from the upper half

The upper half (9/15) built the **semantic layer** of the policy-side interface — it answered "what does a policy actually consume, and why do today's interfaces fail". This piece (9/16) builds the **measurement layer and the implementation layer** — it answers three concrete questions:

1. **How to test** — given a policy checkpoint, what protocol tells you whether it is contract-compliant? (§2 four compliance evidence types + §2.0 skeleton table and five-layer evaluation hierarchy)
2. **How to train** — what has to change in the training objective, the augmentation, and the safety filter in order for the policy to actually read contract? (§1 training-side knock-on effects)
3. **How to land** — what does a minimal executable contract-aware policy skeleton look like, and where do the interface-layer bugs hide? (§3 Python skeleton + fail-closed + top-$k$ API split + `benchmark_rng`)

Together the three answer the experimental closure of the upper-half thesis — the upper half states the claim, this piece delivers the evidence.

### 0.2 Reusing the upper-half pipeline as the coordinate system of this piece

$$\boxed{\;\mathcal C\;\longrightarrow\;(\mathcal C_\pi,\,Q_{\mathcal C_\pi})\;\longrightarrow\;q_\pi\;\longrightarrow\;e_\pi\;\longrightarrow\;\pi_\theta\;\longrightarrow\;g_{\mathrm{safety}}\;\longrightarrow\;\mathcal B_{\mathcal C}\;}$$

Here $\mathcal C_\pi=(Q_\pi,\mathcal O_\pi,V_\pi)$ is the Consumer Contract triple (Queries / Obligations / Versions) and $\mathcal B_{\mathcal C}$ is the contract-side intervention battery $\{T_{\mathrm{frame}},T_{\mathrm{hyp}},T_{\alpha},T_{\mathrm{validity}},T_{\mathrm{prov}},T_{\mathrm{neg}}\}$. Upper-half §0.2 defines the three semantic losses and the safety obligation. This piece uses only a minimum recap:

- $L_{\mathrm{declared}} = \sum_{q\in Q_{\mathcal C}^{\mathrm{req}}} w_q\,\mathbf 1[\nexists q'\in Q_{\mathcal C_\pi}:q'\succeq q]$ (**query subsumption coverage**, not literal set membership; see upper half §0.2.2).
- $L_{\mathrm{projection}} = I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$ ($Y_{\mathcal C}^{\pi}$ is defined by the Consumer Contract, not an arbitrary latent in a benchmark; see upper half §0.2.1).
- $L_{\mathrm{decision}} = \mathbb E_{\mathcal R_{\mathcal D}}[\mathbf 1[D_{\mathcal A}(\pi_\theta(\cdot\mid\hat S),\pi_\theta(\cdot\mid\hat S'))<\epsilon]]$ (collapse rate over action-relevant pairs; $D_{\mathcal A}$ is the same family as §2.2 HSS).
- $\mathcal O_{\mathrm{safety}}$: safe → relaxation **permitted** (not required); unsafe → tighten or stop; unknown → conservative fallback; composed via $g_{\mathrm{safety}}=\bigcap_j g_j$.

**Three losses + one obligation are four separately auditable failure sites, not four statistically independent losses** — they are sequentially coupled via $q_\pi\to\Pi_\pi\to\pi_\theta$. Upper half §0.2.2 discusses this at length; every evidence section in this piece follows that wording.

### 0.3 Three boxed claims of this piece

> **Claim A · Compliance is a multi-evidence argument, not a score.** Contract compliance needs §2's **four compliance evidence types** to converge — $E_{\mathrm{semantic}}$ (invariance / equivariance) + $E_{\mathrm{representation}}$ (conditional probe + $E_{\mathrm{contract}}(T^{\mathrm{contract}})$ + HPC($T^{\mathrm{world}}$) + HSS + calibration diagnostic) + $E_{\mathrm{decision}}$ (CAG$^{\mathrm{fixed}}$ + oracle baseline + matched null + three independent source ablations) + $E_{\mathrm{safety}}$ (three-state constraint certification intervention). **Probe ≠ semantic compliance, CAG ≠ semantic compliance, safety pass ≠ representation retention.** The four types cannot substitute for each other and cannot be compressed into a single scalar.

> **Claim B · Evaluation layers do not imply each other.** §2.0's five-layer hierarchy — **Retention ≠ Sufficiency ≠ Behavioral use ≠ Utility ≠ Safety** — carries four explicit non-implications: Retention passing does not imply Sufficiency (a field can be present in $z_\pi$ yet still not answer $Y_{\mathcal C}^{\pi}$), Sufficiency passing does not imply Behavioral use (enough information yet the policy still ignores it), Behavioral use passing does not imply Utility (correct response but contract was not the bottleneck), Utility and Safety are orthogonal (safety-side fallback is a different thing from utility-side improvement). These four non-implications are the direct motivation for §2.2 splitting $E_{\mathrm{contract}}$ / HPC, §2.5 splitting CAG$^{\mathrm{fixed}}$ / CAG$^{\mathrm{retrained}}$, and §2.6 isolating constraint certification intervention.

> **Claim C · The interface is where the bugs hide, not the architecture.** The §3 Python skeleton fixes three bugs that recur in real contract-aware interfaces — a missing `super().__init__()` that silently drops submodule registration, a fail-open schema-compatibility check that quietly degrades on version mismatch, and a top-$k$ API that conflates two orthogonal decisions (`topk_weight_mode` and `residual_mode`) into one string. None are architecture issues; all are interface issues. This is the thesis again — **a policy is a contract consumer, not merely a function approximator**.

### 0.4 Reading path

- Only care about **how to test**: read §2 (§2.0 skeleton + five-layer hierarchy + four evidence types). §2.5 CAG matched null is the key defense against "OOD sensitivity masquerading as contract-use".
- Only care about **how to train**: read §1 (three intervention-consistency tiers + degradation-as-operator augmentation + three-state safety filter).
- Only care about **how to land**: read §3 (Python skeleton + fail-closed + top-$k$ API split + `benchmark_rng` CRN).
- Want the **full closure**: read §0.2's boxed pipeline figure + §2.0's two tables (loss / obligation onto pipeline sites; evidence onto five layers) + §4 closing.

## 1. Training-time knock-ons: constraints, augmentation, safety filter

Once the upper-half §4 primitives are on the input side, four downstream things must be adjusted — training objective, augmentation, safety filter, evaluation. **Interfaces are not free** — but the changes are **local and bounded**.

> **Bridge between §1 and upper-half §4 (v6 explicit)**: upper-half §4's three families of primitives are the **read-side** contract invariances — "contract semantics is not silently broken at $\Pi_\pi$". This §1's three knock-ons are the **train-side** counterparts — "the policy's response to a contract transformation satisfies a $\mathcal{C}_\pi$-declared relation from the same family". **Contract invariance and augmentation invariance are duals of the same property**, translated twice — once on the reader side, once on the trainer side — not two independent engineering efforts.

| Upper-half §4 primitive | Read-side contract invariance | §1 train-side counterpart |
|---|---|---|
| `mode_select` (§4.1) | hypothesis permutation invariance / equivariance | §1.1 Class B **Type I** (exact invariance) + §2.5 $\mathrm{collapse}_{\mathrm{hyp}}$ (this is an evaluation-side collapse operator, **not one of §1.2's six $T_d$ aug operators**; if a benchmark wants hypothesis collapse on the training side too, it must add an explicit $T_{\mathrm{hypcollapse}}$ entry to §1.2) |
| `age_gate` (§4.2) | order-constrained staleness response (consumer-declared order) | §1.1 Class B **Type II** (order-constrained) + §1.2 $T^{\mathrm{stale}} / T^{\mathrm{latency}}$ aug |
| `provenance_harden` / `dependency_aware_fusion` / `negative_evidence_read` (§4.3) | no response direction is prescribed; only "if removed evidence was decision-relevant, utility should drop" + three-state safety | §1.1 Class B **Type III** (unconstrained — **does not enter the training loss**; it is realized purely on the evaluation side via three §2.4 $\Delta J$ slices) + §1.2 $T^{\mathrm{missing}} / T^{\mathrm{bias}}$ + §1.3 three-state certification |

How to read the table: §1 is not "adding constraints to training" — it is "translating upper-half §4's read-side contracts into loss / augmentation / filter implementations"; every row is the same property written three ways. **The Type I / II / III-to-§4.1 / §4.2 / §4.3 correspondence is the structural anchor between §1 and §2 in this piece**: Type I and Type II are genuine training-side constraints (they enter the loss), while **Type III is a training-side non-constraint that only shows up on the evaluation side in §2.4** (the earlier version conflated Type III with a training constraint; §1.1's own wording is "no intervention consistency; only ablation at evaluation time"). This distinction ensures §1.1's three tiers are not arbitrarily chosen but are forced by upper-half §4's primitives, and ensures §2's evaluation does not mistake an unconstrained training-side element for a constraint training was supposed to satisfy.

### 1.1 Two classes of training constraint: representation-side probe + three tiers of intervention consistency

A natural mistake is: **to make the policy "use the contract", require it to output validity / hypothesis predictions** — this actually sneaks "use the contract" into "copy the contract", which is the wrong direction. **A policy is entirely allowed to only eat the contract and never spit it back out**; auxiliary prediction heads are not a necessary condition.

#### Class A · Representation-side probe (diagnostic)

Hang a few probe heads off the policy's intermediate representation $z^{\pi}$ and try to predict the contract's `age`, `validity`, `observability` and hypothesis posterior from $z^{\pi}$. **These probes do not enter the main loss; they are only used for measurement**: good probe scores mean the policy preserves this information internally, poor scores mean the contract has already been flattened inside $e_\pi$. $L_{\mathrm{probe}}$ may be added to the main loss with a small weight as regularization, but **its diagnostic value exceeds its training value**.

$$\mathcal{L}_{\mathrm{total}} \;=\; \mathcal{L}_{\mathrm{action}} \;+\; \underbrace{\alpha\, \mathcal{L}_{\mathrm{probe}}}_{\text{weak regularization, mainly diagnostic}} \;+\; \underbrace{\sum_{\mathcal{C}} \gamma_{\mathcal{C}}\, \mathcal{L}^{\mathcal{C}}_{\mathrm{consistency}}}_{\text{Type I / II / III, see below}}$$

**But there is a hole the previous version did not plug** — the reviewer was right on this. Suppose $age$ is highly correlated with image embedding (e.g. "more complex scene ⇒ slower processing ⇒ larger age"). Then a probe from $z_\pi$ predicts age with ease:

$$\mathrm{Acc}(\alpha\mid z_\pi) \;=\; 99\%.$$

**This does not prove that age is retained by $e_\pi$** — it may just be a leak of scene difficulty from the image. **The correct diagnostic is a conditional / nuisance-controlled probe**:

$$\text{Retention}_{\mathrm{cond}}(\alpha) \;=\; I(\alpha;\, z_\pi \mid o),$$

i.e. given the raw observation $o$, does $z_\pi$ still **independently** carry age information? Alternatively, the benchmark can run "**same observation, different metadata intervention**" — hold $o$ fixed, vary $m_c$, and observe $z_\pi$'s response. This is philosophically aligned with upper-half §0.2.1 Property B using conditional MI rather than an MI difference: **any field for which the raw input already contains a proxy must be probed conditionally**.

#### Class B · Intervention-consistency constraint (the real core)

**The previous draft wrote this class generically as $D(\pi_\theta(\hat S), \pi_\theta(T_{\mathcal{C}}(\hat S)); \rho_{\mathcal{C}})$, and a reviewer immediately asked "where does $\rho_{\mathcal{C}}$ come from?"** — the four $T_{\mathcal{C}}$ do not all have the same "response pattern", and one of them should not even assume a response is required. This version splits intervention consistency into three tiers by response strength.

**Type I · Exact invariance / equivariance** (the cleanest tier). The transformation has a legal counterpart $T^{\mathcal{C}}_\pi$ on the action side, and we require:

$$\pi_\theta\!\big(T^{\mathcal{C}}(\hat S),\, o,\, \ell\big) \;=\; T^{\mathcal{C}}_\pi\!\big(\pi_\theta(\hat S,\, o,\, \ell)\big).$$

Typical: **frame transform** — move `reference_point` from A to B, $\tau$ transforms per the transport theorem, **the action side must undergo the corresponding coordinate transformation** ($\pi(T_g S) = T_g^A \pi(S)$). Also: **permutation of equivalent hypotheses** (hypotheses with the same posterior weight may be permuted; the action distribution must be invariant). This tier can be written as a hard loss, $\mathcal{L}_{\mathrm{consistency}}^{\mathrm{I}} = \|\pi_\theta(T\hat S) - T^\pi \pi_\theta(\hat S)\|^2$.

**Type II · Order-constrained response** (the middle tier — **previously called monotone response, this version makes it mathematical**). "Monotone" is not a word that can be used casually — one needs a partial order on the range of $M(\cdot)$ and the response functional to be a **well-defined scalar or totally-ordered value**. The previous version crammed variance, action norm, fallback probability, covariance PSD into the same $\preceq$, and the reviewer was right: **those $\preceq$ relations are not the same order at all**.

The correct framing: first define a **severity partial order on contract interventions** $T_1 \preceq_{\mathcal C_\pi}^{\mathrm{declared}} T_2$ (e.g. "older age = more severe", "lower observability = more severe", "validity=false = more severe than high age" — the order is defined by $\mathcal C_\pi$ and $Q_{\mathcal C_\pi}$, **not by the policy**), then specify a **response functional**

$$r:\mathcal P(\mathcal A) \;\longrightarrow\; \mathbb R$$

(it can be $P(\text{fallback})$, $\mathbb E[\|a\|]$, $P(\text{stop})$, $\mathbb E[\mathrm{safe\_margin}]$ and so on — each is a scalar with the total order $\le$), and require:

$$T_1 \preceq_{\mathcal C_\pi}^{\mathrm{declared}} T_2 \quad\Longrightarrow\quad r\!\big(\pi_\theta(T_1\hat S)\big) \;\le\; r\!\big(\pi_\theta(T_2\hat S)\big).$$

**This is what monotonicity actually means**. Example: let $r(\pi) = P_\pi(\text{fallback})$, $T_1$ = "age from 5 ms to 20 ms", $T_2$ = "age from 20 ms to 200 ms"; then $T_1 \preceq_{\mathcal C_\pi}^{\mathrm{declared}} T_2$ and we require $P_\pi(\text{fallback}\mid T_1) \le P_\pi(\text{fallback}\mid T_2)$.

But a real policy may well be **piecewise** —

```
α < 50 ms            →   nominal control
50 ms ≤ α < 100 ms   →   fallback
α ≥ 100 ms           →   stop
```

This kind of response **is not monotone, but it is a legitimate contract-specified response relation**. So Type II's proper name is "**order-constrained response**", and **monotonicity is only one special case**. The loss is:

$$\mathcal{L}_{\mathrm{consistency}}^{\mathrm{II}} \;=\; \sum_{T_1 \preceq_{\mathcal C_\pi}^{\mathrm{declared}} T_2} \max\!\big(0,\; r(\pi_\theta(T_1 \hat S)) - r(\pi_\theta(T_2 \hat S)) + \delta\big).$$

$\delta$ is a margin; both $r$ and $\preceq_{\mathcal C_\pi}^{\mathrm{declared}}$ must be hard-coded in $\mathcal C_\pi$, not fit after the fact.

**Type III · Unconstrained intervention** (the weakest and most important tier). **Do not assume the policy must change.** Typical example: **provenance removal** — dropping a contributing sensor whose information is fully redundant may leave the optimal action unchanged, and that is fine. The correct question here is: **when the removed evidence was decision-relevant, does performance degrade?** This is handed to §2's CAG panel rather than being imposed as a training-time response. Written as a loss: **no intervention consistency at all; only ablation at evaluation**. This tier's very existence is a correction to the previous draft, which treated all four $T_{\mathcal{C}}$ as "must respond" and thereby mistook a Type III intervention for a Type II.

**None of these three tiers require the policy to predict anything explicitly**; they require **the response function to conform to contract semantics, or to be permitted to remain unchanged**. This is the **training-side counterpart** of upper-half §0.2's "$q_\pi$ is a declared quotient". Compared to "adding a few auxiliary prediction heads", this three-tier scheme fits the thesis better and is more research-flavored: **what we propose is not for the policy to copy the contract, but for the policy to respond — or legitimately not respond — according to the tier of intervention**.

### 1.2 Augmentation: degradation as a causal operator

9/14 §4.5 emphasized **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption** — six degradations with different **causal origins** and different **downstream readouts**. The previous version tried to map each degradation to "which slot it corresponds to", and the reviewer was right: **this one-to-one mapping over-simplifies**. Concrete examples:

- **latency** has primary effect on $\alpha_c$; without compensation, secondary effect on $\mu_c$ (a delayed-time value) — one augmentation moves two slots at once.
- **bias** primary effect shifts $\mu_c$; secondary effects often inflate $\Sigma_c$ (the system knows calibration is untrustworthy) and even flip $v_c$.
- **corruption** may move $\mu_c, \Sigma_c, v_c, h_c$ simultaneously.
- **masking** primary effect sets $\iota_c = 0$; secondary effect renders $q_c$ undefined and forces $\Sigma_c^{\mathrm{eff}}$ to fall back to prior.

This version **rewrites each degradation as a causal operator**:

$$T_d:\;(\mu_c,\,\Sigma_c,\,m_c)\;\longmapsto\;(\mu_c',\,\Sigma_c',\,m_c'),\qquad d \in \{\text{mask},\,\text{missing},\,\text{stale},\,\text{latency},\,\text{bias},\,\text{corruption}\}.$$

One-line principle: **each degradation has a primary semantic effect and potentially secondary effects on other fields** — the aug pipeline must **enumerate explicitly** which slots $T_d$ touches and by how much, not "class → single slot". upper-half §4.2's three-group structure (payload / metadata / derived trust) is directly usable here: **$T_d$ acting on $(\mu,\Sigma)$ is payload-level corruption, acting on $m_c$ is metadata-level augmentation, and $q_c$ must be derived from the modified $m_c$, never rewritten by the aug side** (otherwise "aug cheats and deployment does not see").

**The previous draft had a shortcut risk that reviewers caught**: "label each episode with a degradation class, carry it explicitly on the policy input" — if `degradation = stale_vision` is fed directly as a policy input, the policy learns $a = f(o, \text{label})$; but **in real deployment the label itself is not reliable**, and the shortcut is harmless in training and disastrous at deployment. This version explicitly separates two things:

**Observed metadata** $m_c = (\alpha_c, \ell_c, h_c, v_c, \iota_c)$ — obtained directly by the estimator or sensor driver, may enter the contract and may enter the policy input. **This is exactly upper-half §4.2's metadata quintuple**.

**Latent degradation class** $d_c \in \{\text{missing}, \text{stale}, \text{bias}, \text{corrupt}, \ldots\}$ — augmentation knows which class was injected, but at deployment this class is **latent**: it can only be inferred by the contract estimator from the $m_c$ stream, or used as a training annotation for loss weighting and sampling, **never as a ground-truth input to the policy by default**.

Concretely: the aug pipeline samples $d_c$, applies $T_{d_c}$ to generate $(\mu_c', \Sigma_c', m_c')$, then feeds only $m_c'$ into the policy and uses $d_c$ solely for loss weighting and per-class evaluation slicing. **This is not curriculum, it is conditioning on observed metadata** — the difference is that conditioning lets the policy see reliable $m_c'$, not reliable $d_c$. §1.1 Class B Type II order-constrained response pairs naturally with degradation-conditioned aug — **aug generates the $m_c$ shift induced by $T_{\mathcal{C}}$, loss measures whether the policy's response to $m_c$ matches $\preceq_{\mathcal C_\pi}^{\mathrm{declared}}$ and $r$**.

### 1.3 Safety filter and its interface to the contract: constraint certification (three-state in v5)

The constraint layer (CBF / shield / runtime verifier) **must read $a_{\mathrm{proposed}}$**, otherwise what is it filtering — that point is not up for negotiation. But the more precise claim of this piece is: **the safety filter should not treat the policy's confidence or the estimator's general-purpose validity bit as the sole evidence that a constraint holds; it should directly access constraint-relevant evidence**.

**Two kinds of validity must be distinguished here.** The estimator-side general `validity` bit says "this measurement is valid from a calibration / sensor-health perspective" — that is a field-level, general-purpose declaration. What safety actually cares about is **"is this constraint valid at this moment, under this predicate"** — that is constraint-level semantics. The two differ: `validity = true` does not imply that the constraint estimate is valid for **this particular safety predicate**; e.g. "joint-torque calibration OK" ≠ "current contact estimate supports the collision constraint".

This piece proposes that the safety filter read a **constraint-specific composite**:

$$v_j^{\mathrm{constraint}} \;=\; g\!\Big(\text{field validity},\; \text{observability},\; \text{hypothesis posterior},\; \text{age},\; \text{model coverage}_j\Big).$$

$g$ is a constraint-specific composition rule (e.g. a CBF side may require "distance estimate is stable under the current hypothesis, observability is sufficient, and dynamics model coverage reaches the current state region"). $v_j^{\mathrm{constraint}}$ is what the safety filter should actually read as **constraint-relevant evidence**. 9/14 §3 established that **track_id is a hypothesis** — the safety filter must not simply trust track_id matching, it must also check whether the hypothesis posterior is stable; whether two tracks merge or split directly determines the credibility of "how far is that obstacle".

**v5 addition · three-state certification.** The previous draft wrote $v_j^{\mathrm{constraint}}$ as a binary bit, and the reviewer was right — **"insufficient evidence" and "constraint does not hold" are two different things**. Three concrete cases: an invalid sensor is *unknown*, "obstacle detected" is *unsafe*, "obstacle absent with high observability" is *safe*. Encoding all three with $v \in \{0, 1\}$ collapses them into the same value and the safety filter loses the information it actually needs. This piece therefore introduces a three-state:

$$\boxed{\;\mathrm{certification}_j \;\in\; \{\text{safe},\;\text{unsafe},\;\text{unknown}\}.\;}$$

The reaction rules follow directly: **safe → relaxation is permitted** (note carefully: **permitted, not required** — see below), **unsafe → tighten or stop**, **unknown → conservative fallback or tighten** (since we cannot certify, we default to the more cautious side). This upgrade brings the safety contract closer to real runtime-safety semantics — $v_j^{\mathrm{constraint}} = 0$ no longer means "constraint false"; it means **"evidence is insufficient to certify the constraint predicate"** (which corresponds to the unknown state).

**v6 refinement · safe ≠ "must relax".** The previous version wrote "safe → relax", and the reviewer produced a simple counterexample: **collision constraint = safe, joint torque constraint = unknown** — even though the collision constraint is already safe, that does **not** license relaxing the whole safety envelope, because the joint-torque branch is still in the unknown state. The correct semantics is: **safe means "this constraint can run at its normal margin"; it does not mean "this constraint can be pulled apart"**. The reaction rules are therefore refined into three named actions:

1. **safe → allow_normal_margin** — keep this constraint's margin at its default value; do **not** loosen the constraint; continue intersecting with all other constraints;
2. **unsafe → tighten_or_stop**;
3. **unknown → conservative_fallback**.

**Multiple constraints must be combined, not short-circuited on the first unknown.** The safety filter's real form is $g_{\mathrm{safety}} = \bigcap_j g_j$ — each constraint independently contributes the subset of actions it permits, and the applied action is **the intersection over all constraints**. Written out:

$$a_{\mathrm{applied}} \;=\; \Big(\bigcap_{j:\,\mathrm{cert}_j = \text{safe}} g_j^{\mathrm{normal}}(a_{\mathrm{proposed}})\Big) \;\cap\; \Big(\bigcap_{j:\,\mathrm{cert}_j = \text{unsafe}} g_j^{\mathrm{tighten}}(a_{\mathrm{proposed}})\Big) \;\cap\; \Big(\bigcap_{j:\,\mathrm{cert}_j = \text{unknown}} g_j^{\mathrm{fallback}}(a_{\mathrm{proposed}})\Big).$$

No single constraint loosening the action set is legal on its own — **only what every constraint jointly permits may actually be applied**. This property also fixes the correct shape of `SafetyFilterHead.forward` in §3 (see v6 code: `continue` only skips tightening for *this one* constraint, it never `return`s and never terminates the loop early).

**The single most important sentence of this section (v5 adds one word)**:

> **$v_j^{\mathrm{constraint}}$ does NOT turn off the constraint when evidence is invalid or unknown.**
> **Invalid evidence ALONE cannot justify relaxing the constraint.**
> Equivalently: **Relaxation requires sufficient valid evidence; invalid evidence alone is never sufficient.**

The previous version's phrasing read as "if validity falls, the constraint may switch off" — **that is dangerous safety semantics** and easy for a reviewer to pierce with a single counterexample (camera invalid but lidar valid and already sufficiently establishes obstacle absence; in that case the constraint can be relaxed legitimately). Adding "alone" repairs the semantics: **"invalid evidence by itself cannot serve as a sufficient condition for relaxation; if another independent valid evidence stream has sufficiently established that the constraint does not need to hold, that is a legal relax"**. This change keeps the counterexample out while not weakening the piece's core safety claim. Three legitimate reactions:

1. **fallback** — switch to a more conservative controller or planner;
2. **conservative tightening** — **increase** the constraint margin (e.g. raise minimum distance from 20 cm to 50 cm), because "less evidence ⇒ more caution";
3. **stop** — halt the policy and wait for observation to recover.

**None of the three is "constraint off"** — unless an independent valid evidence stream has explicitly pushed certification to **safe**. The core safety fact is: **positive (and sufficient) evidence can justify relaxing a constraint; absence of valid evidence alone never can**. This sentence lifts the piece from an ML-interface discussion to safety semantics, and it is the real definition of §2's safety-level compliance evidence — what is being tested is not "does the policy stop making mistakes when validity=0", but "**does the safety filter tighten — rather than relax — when certification is unknown**".

This also wires in upper-half §4.3.3's negative evidence: **"what should have been seen but was not" pushes $\mathrm{certification}_j$ toward unknown** — for example, the radar swept this angle and saw nothing, $\Lambda(E^-; H_{\text{clear}}, \mathcal O)$ turns positive (under the "clear" hypothesis "nothing seen" is natural; under the "obstacle present" hypothesis "nothing seen" is unnatural — **the direction depends on how $H$ is defined, see upper-half §4.3.3**), and the collision constraint's certification should stay in **unknown** or even be **conservatively tightened**, never pushed to safe just because the policy's belief is optimistic.

**If these three (constraint-specific certification, observability, negative evidence) do not enter the safety filter, the filter will infer constraint validity from the policy's belief** — which is especially dangerous in low-observability regions. The policy's belief is optimistic precisely because it cannot see the contract's observability / validity / negative evidence, and **if the filter also cannot see them, the two go blind together**.

### 1.4 Echo with 9/10 Part 3 evaluation

Among the three sim-utility dimensions (prediction / ranking / decision), policy-side evaluation is mainly about **decision** — but with an added **contract-preservation dimension**: how much the policy degrades when the contract is torn apart is a **lower bound** on how much it depends on the contract. This dimension maps to the CAG metric of §2, and needs an oracle baseline to bound its interpretation **plus an explicit retraining protocol (§2.5)**.

## 2. Evaluation: four compliance evidence types, five-layer hierarchy, not four metrics

Aligned with the upper-half §4 primitives and the three tiers of §1.1, this piece proposes **four types of compliance evidence** — **not a four-level metric panel, but four evidence categories each answering a different question**. The previous version wrote them as a metric hierarchy, and the reviewer was right: **no single score among probe / CAG / safety-pass constitutes semantic compliance**; compliance must be a **multi-evidence conjunction**.

$$\boxed{\;\text{Compliance Evidence} \;=\; \big\{E_{\mathrm{semantic}},\;E_{\mathrm{representation}},\;E_{\mathrm{decision}},\;E_{\mathrm{safety}}\big\}.\;}$$

Each type answers a distinct question:

| Type | Question answered |
|---|---|
| **$E_{\mathrm{semantic}}$** | **Does the policy respond correctly to a known semantic transformation?** |
| **$E_{\mathrm{representation}}$** | **Is the contract information recoverable from the representation (given the side inputs)?** |
| **$E_{\mathrm{decision}}$** | **Does contract structure change task utility when it should?** |
| **$E_{\mathrm{safety}}$** | **Does degradation cause conservative / required guardrail behavior?** |

Three explicit caveats: **probe ≠ semantic compliance, CAG ≠ semantic compliance, safety pass ≠ representation retention**. These four evidence types **cannot substitute for each other and cannot be collapsed into a single scalar**. Each is unfolded below. The temporal and source invariants have specialized tests inside $E_{\mathrm{decision}}$: **SDS** for temporal, and **$\Delta J_{\mathrm{where}} / \Delta J_{\mathrm{dep}} / \Delta J_{\mathrm{neg}}$** for source — they are not new metrics but focused slices of the decision-layer panel.

### 2.0 The theoretical skeleton table (v6 revision: three losses + one obligation + five-layer evaluation hierarchy)

Compress upper-half §0 through §2 into one table — this is what a reviewer most wants to see:

| Layer | Object | Failure | Evidence | Evidence defined in § | Loss / Obligation |
|---|---|---|---|---|---|
| Contract | $\mathcal C$ | schema ambiguity / version mismatch | schema audit + compatibility check | upper-half §0.2.3 + this piece §3 (`SchemaCompatibilityError` fail-closed) | —— (precondition, no loss) |
| Declaration | $q_\pi$ (induced by $\mathcal C_\pi = (Q_\pi, \mathcal O_\pi, V_\pi)$) | undeclared semantic collapse | quotient audit — can you display a $Q_{\mathcal C_\pi}$ list? does the declared set upper-cover required queries along subsumption? | upper-half §0.2.2 (query subsumption definition) + §0.2.3 (version compatibility) + the $L_{\mathrm{declared}}$ boxed formula below | $L_{\mathrm{declared}} = \sum w_q \mathbf 1[\nexists q' \succeq q]$ (§2.0 boxed) |
| Projection | $\Pi_\pi = e_\pi\circ q_\pi$ | residual contract information | conditional probe $I(\text{field};z_\pi\mid o)$ | §2.2 (first half) | $L_{\mathrm{projection}} = I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$ |
| Decision | $\pi_\theta$ | wrong use / shortcut / collapse rate | Type I equivariance + Type II order-constrained + Type III ablation + $L_{\mathrm{decision}}$ + §2.2 $E_{\mathrm{contract}}$ / HPC / HSS | §2.1 (Type I) · §2.2 ($E_{\mathrm{contract}}$ / HPC / HSS) · §2.3 (SDS temporal) · §2.4 (source three-slice) · §2.5 (CAG utility) | $L_{\mathrm{decision}} = \mathbb E[\mathbf 1[D_{\mathcal A}(\cdot)<\epsilon]]$ |
| Safety | $g_{\mathrm{safety}}$ | unsafe interpretation of invalid / unknown evidence | constraint certification intervention (does it **tighten**, not **relax**) | §2.6 · training-time §1.3 | $\mathcal O_{\mathrm{safety}}$: safe → relaxation permitted / unsafe → tighten or stop / unknown → conservative fallback |

**Three semantic losses + one safety obligation** (v6 form, aligned with upper-half §0.2.2):

$$\boxed{\begin{aligned}
L_{\mathrm{declared}} &: \;\textstyle\sum_{q\in Q_{\mathcal C}^{\mathrm{req}}} w_q\,\mathbf 1\!\big[\nexists\, q' \in Q_{\mathcal C_\pi}:\, q' \succeq q\big];\\[1mm]
L_{\mathrm{projection}} &= I\!\big(Y_{\mathcal C}^{\pi};\hat S \mid Z_\pi, O, L\big);\\[1mm]
L_{\mathrm{decision}} &= \mathbb E_{\mathcal R_{\mathcal D}}\!\big[\mathbf 1[D_{\mathcal A}(\pi_\theta(\cdot\mid\hat S),\pi_\theta(\cdot\mid\hat S'))<\epsilon]\big];\\[1mm]
\mathcal O_{\mathrm{safety}} &: \;\text{safe} \Rightarrow \text{relaxation permitted (constraint policy still applies);}\\
&\quad \text{unsafe} \Rightarrow \text{tighten or stop};\quad \text{unknown} \Rightarrow \text{conservative fallback}.
\end{aligned}}$$

The three losses sit on $q_\pi / \Pi_\pi / \pi_\theta$ respectively, and the safety obligation sits on $g_{\mathrm{safety}}$ — **four separately auditable failure sites** that together form a compliance argument. **Note v6's wording**: the paper does **not** claim these four slots are "statistically independent" (the reviewer was right: changing $L_{\mathrm{declared}}$ changes $q_\pi$, changing $q_\pi$ changes the admissible quotient at $\Pi_\pi$, which in turn changes the evaluation set for $L_{\mathrm{decision}}$ — the four are **sequentially coupled**; each only localizes to an openable audit site). This is the concrete shape of the piece's journey from "add metadata to a VLA" to "contract-aware policy design".

**v6 second table · Five-layer evaluation hierarchy** — the single most valuable table the reviewer suggested adding. This piece has been implicitly trying to distinguish "is the information still there / is it enough / is it used / is using it worth it / how does the filter react when evidence is uncertain"; before v6 these were scattered across §2.1–§2.6 without a unifying frame. This table is the piece's evaluation-layer **type system**:

| Layer | Question | Primary metric / primitive | What separates it from the others | Defined in § | Training-time §1.x |
|---|---|---|---|---|---|
| **Retention** | Is the contract field still in $z_\pi$? | Conditional probe $I(\text{field}; z_\pi \mid o)$ | Only "is it there", not "is it enough" | §2.2 (first half) | §1.1 Class A (representation-side probe) |
| **Sufficiency** | Given $z_\pi$ and side info, can we answer $Y_{\mathcal C}^{\pi}$? | $L_{\mathrm{projection}} = I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$ | Conditional MI = 0; does not claim representation-level injectivity | §2.2 (first half) + §2.0 boxed | §1.1 Class A |
| **Behavioral use** | Does the policy respond legally to contract interventions? | Type I equivariance / Type II order-constrained / §2.2 $E_{\mathrm{contract}}(T_k)$ vs $\mathcal R_{\mathcal C}$ | Bound to "is the response legal", not to final utility | §2.1 · §2.2 · §2.3 · §2.4 | §1.1 Class B (Type I / II enter the loss; Type III is evaluation-only via §2.4) |
| **Utility** | Using contract information — did the decision actually improve? | $\mathrm{CAG}^{\mathrm{fixed}}$ + §2.2 HPC ($T^{\mathrm{world}}$ form) + §2.4 $\Delta J_{\mathrm{where/dep/neg}}$ | Must be paired with matched null + oracle baseline, otherwise it is only OOD sensitivity | §2.5 · §2.2 HPC · §2.4 | §1.2 augmentation (degradation as causal operator) |
| **Safety** | Does the filter take a conservative reaction under unknown / invalid evidence? | §2.6 constraint certification intervention + §1.3 three-state certification + $\mathcal O_{\mathrm{safety}}$ | Independent of "did the policy use the contract" — measures whether the filter backs it up | §2.6 | §1.3 safety filter three-state |

These five layers are **stepwise progressive but not mutually implying**. Retention passing does not imply Sufficiency (the field can be present yet not enough to answer a query); Sufficiency does not imply behavioral use (enough information does not mean the policy reads it); behavioral use does not imply utility (the response can be legal yet utility still low, because contract may not be the current bottleneck); and utility is fully orthogonal to safety (the filter's backstop is a different fact from the policy's task performance). **These four non-implications are exactly the motivation** for §2.2's $E_{\mathrm{contract}}$ / HPC split, §2.5's CAG fixed / retrained split, and §2.6's separate constraint-certification intervention. **Retention ≠ Sufficiency ≠ Use ≠ Utility ≠ Safety** — this line was implicit before v6 and is now explicit.

> **How to use these two tables**: reviewers or readers entering this piece's evaluation chapter land here first. **Table 1** answers "if this pipeline site is broken, which § holds the evidence, and which loss / obligation does it realize". **Table 2** answers "if this evaluation layer passes or fails, which § carries the concrete formula and protocol, and which training-time change is the counterpart". Each of §2.1–§2.6 opens with a **"Section anchor"** line that points back to specific rows / layers in these two tables. Tables + anchors together form a **bidirectional jump index** for the evaluation chapter — from the table to a specific section, from the section back to its exact layer and loss / obligation.

Together, §2.0's two tables carry the theoretical skeleton of the piece's evaluation section: the first maps losses / obligation to four auditable pipeline sites; the second maps evidence to five non-implicational layers. These two tables are the piece's most direct interface for moving from a conceptual article to a benchmark-protocol article.

### 2.1 Semantic evidence $E_{\mathrm{semantic}}$: Invariance / Equivariance Test

> **Section anchor**: §2.0 Table 1 Layer = **Decision** (Type I slice) · Table 2 Layer = **Behavioral use** (response correctness, not utility) · training-time counterpart **§1.1 Class B Type I** (equivariance constraint).

Corresponding to §1.1 Type I. Given a family of contract transformations with known $T^{\mathcal{C}}_\pi$ (frame / coordinate / hypothesis permutation), measure:

$$\mathrm{Equiv}(\mathcal{C}) \;=\; \mathbb{E}_{\hat S}\!\left[d\!\left(\pi_\theta(T^{\mathcal{C}}\hat S),\; T^{\mathcal{C}}_\pi\,\pi_\theta(\hat S)\right)\right].$$

$\mathrm{Equiv} \to 0$ is a hard requirement; large $\mathrm{Equiv}$ means either $e_\pi$ learned it wrong, or $q_\pi$ dropped the quotient entirely. This is the cleanest type, because the rule is mathematically defined — no oracle and no $J$ definition needed. upper-half §3.2 Failure 2's severity can be quantified directly by $\mathrm{Equiv}(\text{frame})$.

### 2.2 Representation evidence $E_{\mathrm{representation}}$: Conditional Probe + Interface-Compliance $E_{\mathrm{contract}}$ + Decision-Competence HPC + HSS (v6 second split) + Calibration

> **Section anchor**: §2.0 Table 1 Layer = **Projection** (conditional probe / $L_{\mathrm{projection}}$) + **Decision** ($E_{\mathrm{contract}}$ / HPC / HSS — three sub-metrics) · Table 2 Layers = **Retention + Sufficiency + Behavioral use + Utility**, spanning four layers simultaneously (this is the only section doing so) · training-time counterparts **§1.1 Class A + Class B**.

**Conditional probe** (upgraded §1.1 Class A): Retention$_{\mathrm{cond}} = I(\text{field}; z_\pi \mid o)$ — hold raw observation $o$ fixed and measure how much contract information $z_\pi$ **independently** carries. This is the necessary form to avoid image-proxy leakage.

**v5 already split $T_k^{\mathrm{hyp}}$ into $T_k^{\mathrm{contract}}$ (raw observation fixed) and $T_k^{\mathrm{world}}$ (observation and contract re-rendered jointly), but the HPC formula itself was still wrong** — the v6 reviewer nailed it in one sentence:

> **If the raw observation is held fixed and the contract is intentionally made inconsistent with it, what defines the ground-truth optimal action set $\mathcal{A}^{*}_k$?**

v5 could not answer this. Under $T_k^{\mathrm{contract}}$ only the contract changed — the world did not. So the "world-consistent optimal action set" is still $\mathcal{A}^*(O, W)$ of the original world; a new $\mathcal{A}_k^*$ does not magically appear because you rewrote the contract to $H_k$. **v5's HPC formula tacitly assumed "changing the contract = changing the world", precisely re-fusing the two things v5 spent a whole paragraph separating.** v6 splits a second time — **HPC no longer carries both roles; it becomes two metrics**:

**(A) Contract-only intervention → Interface compliance $E_{\mathrm{contract}}$** — $T_k^{\mathrm{contract}}:\hat S \mapsto \hat S_k$ with the raw observation held fixed, and **no $\mathcal{A}^*_k$ oracle is introduced at all**. What it measures is: after the contract's declared change, does the policy land inside the **declared response set** $\mathcal{R}_{\mathcal{C}}(T_k)$ already written into $\mathcal{C}_\pi$ in §1.1 — closing the semantic loop with §1.1's Type I equivariance and Type II order-constrained response:

$$\boxed{\;E_{\mathrm{contract}}(T_k) \;=\; D_{\mathcal{A}}\!\big(\pi_\theta(T_k^{\mathrm{contract}}(\hat S)),\;\mathcal{R}_{\mathcal{C}}(T_k)\big),\qquad E_{\mathrm{contract}}^{\mathrm{overall}} = \tfrac{1}{K}\sum_k \mathbf 1\!\big[E_{\mathrm{contract}}(T_k) > \epsilon_{\mathrm{resp}}\big].\;}$$

$D_{\mathcal{A}}$ is the same family used by upper-half §0.2.2's $L_{\mathrm{decision}}$ and §2.2's HSS (action-equivalence-aware distance; it measures whether two action distributions **support different admissible / optimal action sets**). $\mathcal{R}_{\mathcal{C}}(T_k)$ may be piecewise, flat, or conditional — as long as it does not violate the $\preceq_{\mathcal{C}_\pi}^{\mathrm{declared}}$ and $r$ declared in §1.1 it is legal. **When the reviewer asks "if the contract and the observation contradict each other, where does the ground-truth action come from?", v6's answer is: we do not need a ground-truth action; we only need the response set the contract itself declared.** That is what interface compliance should actually look like.

**(B) World-consistent counterfactual → Decision competence HPC** — jointly change $(O, \hat S) \mapsto (O_k, \hat S_k)$, where both $O_k$ and $\hat S_k$ come from a simulator / renderer / privileged state, guaranteeing observation and contract remain consistent. Now $\mathcal{A}^*_k$ has a natural meaning — **it is the optimal / admissible action set of world $k$**, determined by the simulator's ground-truth state and reward:

$$\boxed{\;\mathrm{HPC} \;=\; \frac{1}{K} \sum_{k=1}^{K} U\!\big(\pi_\theta(T_k^{\mathrm{world}}(\hat S, O)),\;\mathcal{A}^{*}_k\big),\qquad U(\pi, \mathcal{A}^*_k) = \Pr_{a \sim \pi}\!\big[a \in \mathcal{A}^{*}_k\big].\;}$$

HPC **only makes sense under $T_k^{\mathrm{world}}$** — it measures "policy's decision competence under a different real-world hypothesis", not "policy's response to a contract change". The two roles are now carried by two different metrics; when reporting, a benchmark **must present $E_{\mathrm{contract}}$ and HPC side by side, never merged into a single composite**. This split is exactly aligned with the piece-wide philosophy "contract change ≠ world change". **Note**: this section's HPC $\mathcal{A}^*_k$ (**world-$k$ action set**) and §2.5's oracle baseline $\pi^*_{\mathrm{oracle}}$ (**a policy reading the privileged state**) are two orthogonal oracle objects and should not be merged — see the closing note of §2.5 for the full disambiguation.

**Hypothesis Separation Score (HSS)** — **must be averaged only over action-relevant hypothesis pairs**; the reviewer caught gaming: if $\mathcal A^*(H_1) = \mathcal A^*(H_2)$, different outputs are not a virtue, **they are noise**. Define the action-relevant pair set (using upper-half §0.2.1's $\sim_{\pi,\mathcal D}$, not "raw representation differs"):

$$\mathcal R \;=\; \big\{(i, j): \hat S_i \not\sim_{\pi,\mathcal D} \hat S_j\big\} \;\cap\; \big\{(i, j): T_i, T_j \text{ belong to the same branch of } \{T^{\mathrm{contract}}, T^{\mathrm{world}}\}\big\}.$$

**The second intersection is critical** — $T^{\mathrm{contract}}$ and $T^{\mathrm{world}}$ have different semantics; they must not be averaged inside one HSS. HSS is computed only on $\mathcal R$, **and $D$ is uniformly $D_{\mathcal{A}}$, not an ordinary distribution distance** (KL, TV, Wasserstein can all be maxed out by a policy that emits a different random distribution per hypothesis):

$$\mathrm{HSS}^{c} \;=\; \frac{1}{|\mathcal R^{c}|} \sum_{(i, j) \in \mathcal R^{c}} D_{\mathcal{A}}\!\big(\pi_\theta(\cdot \mid T_i^{c}\hat S),\;\pi_\theta(\cdot \mid T_j^{c}\hat S)\big),\qquad c \in \{\mathrm{contract},\mathrm{world}\}.$$

**Separation is useful only when distinctions are decision-relevant** — this sentence must be locked down. **And even $D_{\mathcal A}$ alone cannot fully block the "fully-random policy produces different random distributions under different hypotheses" game**, so v6 introduces a **competence gate**: HSS only counts as positive evidence **when $E_{\mathrm{contract}}$ has already passed** ($E_{\mathrm{contract}}^{\mathrm{overall}} < \tau_E$); otherwise high HSS only means the policy is thrashing. **Do not force the two metrics into a single product** (e.g. $E_{\mathrm{contract}} \cdot \mathrm{HSS}$) — the reviewer prefers **parallel reporting**:

| Metric | Question | Anti-gaming |
|---|---|---|
| $E_{\mathrm{contract}}^{\mathrm{overall}}$ | Does the policy stay inside the declared response set? | $D_{\mathcal A}$; both $D$ and $\mathcal R_{\mathcal C}$ aligned to §1.1 |
| $\mathrm{HPC}$ | Under world-consistency, does the policy support the correct action set? | Only computed on $T^{\mathrm{world}}$; $\mathcal A^*_k$ fixed by simulator |
| $\mathrm{HSS}^{c}$ | Does the policy separate action-relevant pairs? | $D_{\mathcal A}$ + competence gate ($E_{\mathrm{contract}}$ already passed) |
| Entropy / diversity (control) | Is it just randomization? | Reported alongside HSS; a random policy has abnormally high entropy but HSS does not rise |

HPC / HSS / $E_{\mathrm{contract}}$ / entropy **four columns in parallel** — the reviewer can immediately tell that "high HSS + high entropy" is gaming, while "high HSS + low entropy + $E_{\mathrm{contract}}$ passed" is real separation.

**All four together correspond to decision-relevant semantic preservation (upper-half §0.2.1 Property A′)** — $E_{\mathrm{contract}}$ guarantees "when the contract changes, the policy's response is legal"; **HPC** guarantees "under world-consistency, every action-relevant hypothesis is supported"; **HSS** guarantees "the policy preserves action-relevant hypothesis distinctions"; **the entropy control** guarantees "HSS is not randomization gaming"; **and none of them penalize pairs that should not be distinguished**. Only the four constraints jointly give representation evidence meaning.

**Calibration diagnostic** — $\mathrm{ECE}_{\text{top-}k}$, NLL, Brier, calibration curve, coverage / credibility listed in parallel (upper-half §4.1 has already stripped these from the primitive; this is their real home).

### 2.3 Temporal slice: Staleness Response Compliance (SDS) (v5 notation + baseline fix)

> **Section anchor**: §2.0 Table 1 Layer = **Decision** (temporal sub-slice) · Table 2 Layer = **Behavioral use** (Type II order-constrained response; §1.1 Type II evaluation-side counterpart) · training-time counterparts **§1.1 Type II + §1.2 augmentation staleness**.

**v4's SDS had two problems (the reviewer caught both).** First, $R_\pi(a) = \pi_\theta(\cdot|\mathrm{do}(a_c = a), o)$ used $a$ simultaneously as the age parameter and as the paper's action variable — a direct notation collision. Second, treating $R^*(a)$ as the **unique oracle response curve** is too strong: expert A might fallback whenever age > 100 ms, expert B might compensate through dynamics prediction and stay nominal, both are legitimate, and forcing a unique $R^*$ would misclassify a reasonable piecewise response as a failure.

**v5 fixes both in one shot.** Notation: use $\alpha$ as the age value (upper-half §4.2 has already renamed the field from $a_c$ to $\alpha_c$), and reserve $a$ exclusively for action:

$$R_\pi(\alpha) \;=\; \pi_\theta\!\big(\cdot \,\big|\, \mathrm{do}(\alpha_c = \alpha),\, o\big).$$

Baseline: SDS is no longer defined against a single $R^*$, but against a **contract-permitted response set** $\mathcal R_{\mathcal C}(\alpha)$ — this set is declared in $\mathcal C_\pi$ via §1.1's $\preceq_{\mathcal C_\pi}^{\mathrm{declared}}$ and $r$; piecewise is allowed, flat is allowed, **anything that does not violate the declared relation is legal**:

$$\boxed{\;\mathrm{SDS} \;=\; D\!\big(R_\pi,\;\mathcal R_{\mathcal C}\big),\;}$$

where $D$ is a "distance from a curve to a set of legal curves". One concrete form is the sup-based violation measure $\sup_{\alpha_1 \preceq_{\mathcal C_\pi}^{\mathrm{declared}} \alpha_2} \max\!\big(0, r(R_\pi(\alpha_1)) - r(R_\pi(\alpha_2)) + \delta\big)$ (the §1.1 Type II hinge loss reused at evaluation), but other curve-set divergences are also valid. **The oracle curve $R^*$ is only one baseline inside $\mathcal R_{\mathcal C}$, not the definition itself.** This makes SDS fully consistent with §1.1 Type II "order-constrained does not have to mean monotone" — no longer forcing every task to share one unique correct staleness response.

**v6 addition · Operational estimator for SDS ($V_{\mathrm{order}}$)** — the notation $D(R_\pi, \mathcal R_{\mathcal C})$ above is clean but abstract; a single reviewer question **"How do you compute distance to a set of legal curves?"** is enough to break a benchmark version ($\mathcal R_{\mathcal C}$ is typically an infinite set). v6 explicitly states that the recommended SDS implementation is a **violation functional**, not a curve-to-set distance:

$$\boxed{\;\mathrm{SDS} \;=\; V_{\mathrm{order}} \;=\; \mathbb E_{(\alpha_i, \alpha_j)\,:\,\alpha_i \preceq_{\mathcal C_\pi}^{\mathrm{declared}} \alpha_j}\!\Big[\max\!\big(0,\; r\!\big(R_\pi(\alpha_i)\big) - r\!\big(R_\pi(\alpha_j)\big) + \delta\big)\Big].\;}$$

This is **finitely measurable**: given a grid $\{\alpha_1, \alpha_2, \ldots\}$, enumerate every pair compatible with the declared order, run the policy to read out $r(R_\pi(\alpha))$, compute the hinge, average. $\delta$ **takes the same value as §1.1's Type II hinge loss and is chosen by a task-level principle** (for example $\delta = \sigma_r / 2$ with $\sigma_r$ the empirical standard deviation of $r$ under benchmark noise; or $\delta$ calibrated through the same action-distance family as §2.2's $\epsilon_{\mathrm{resp}}$) — it cannot just be called "margin", and it cannot be picked after the fact.

When §1.1 Type II declares a **piecewise policy** (e.g. $\alpha<50$ → normal, $50\le\alpha<100$ → fallback, $\alpha\ge 100$ → stop), $V_{\mathrm{order}}$ can be replaced by a direct **state-transition violation**:

$$V_{\mathrm{trans}} = \mathbb E\!\big[\mathbf 1\big[\text{observed state}\big(R_\pi(\alpha)\big) \neq \text{declared state at }\alpha\big]\big].$$

The body keeps $D(R_\pi, \mathcal R_{\mathcal C})$ as the **theoretical definition**; experiment implementations uniformly go through $V_{\mathrm{order}}$ or $V_{\mathrm{trans}}$, so a benchmark paper can copy the estimator directly.

The response property $R$ is task-defined and may take **variance / action norm / fallback probability / safety margin / stop probability** — **it must be the same $r$ declared in §1.1 Type II**, otherwise training and evaluation talk past each other. Flat is not automatically bad, as long as it matches some legal curve in $\mathcal R_{\mathcal C}$.

The derivative form is retained as one local characterization of response shape:

$$\left.\frac{\partial\, \mathbb{E}\!\big[\pi_\theta(\cdot \mid \mathrm{do}(\alpha_c = \alpha),\, o)\big]}{\partial \alpha_c}\right|_{\alpha}\quad\text{compared to the same-order derivative distribution of legal curves in } \mathcal R_{\mathcal C}.$$

### 2.4 Source slice: $\Delta J_{\mathrm{where}}$, $\Delta J_{\mathrm{dep}}$, $\Delta J_{\mathrm{neg}}$ (v5 three-way split)

> **Section anchor**: §2.0 Table 1 Layer = **Decision** (source sub-slice) · Table 2 Layers = **Behavioral use + Utility** (the three $\Delta J$ slices give response legitimacy and utility impact simultaneously) · training-time counterpart **§1.1 Type III primarily** (provenance / dependency / negative evidence all take the "no prescribed response" Type III stance on the training side; this section is where they get split into three separate $\Delta J$ evaluations); **Type II is additionally invoked** when provenance or negative evidence correlates with a $\preceq^{\mathrm{declared}}_{\mathcal C_\pi}$ (§1.1's declared order), for example "lower sensor health ⇒ lower contributing weight".

**v4 lumped provenance and `correlated_with` into a single $\Delta J_{\mathrm{prov}}$ — the reviewer was right: after devoting a whole section to arguing provenance ≠ dependency ≠ negative evidence, the metric folds them back together and can no longer answer "which primitive actually helped".** v5 splits the source panel into three, each ablating exactly one upper-half §4.3 primitive:

$$\Delta J_{\mathrm{where}} \;=\; J\!\big(\pi_\theta \mid \text{provenance}\big) \;-\; J\!\big(\pi_\theta \mid \text{provenance} = \varnothing\big),$$

$$\Delta J_{\mathrm{dep}} \;=\; J\!\big(\pi_\theta \mid \text{correlated\_with}\big) \;-\; J\!\big(\pi_\theta \mid \text{correlated\_with} = \varnothing\big),$$

$$\Delta J_{\mathrm{neg}} \;=\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda\big) \;-\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda = \varnothing\big).$$

The three correspond one-to-one to upper-half §4.3's primitives: **$\Delta J_{\mathrm{where}}$ measures `contributing_mask` consumption, $\Delta J_{\mathrm{dep}}$ measures `correlated_with` consumption, $\Delta J_{\mathrm{neg}}$ measures conditional-LLR consumption**. $J$ is always higher-is-better. Each can independently hit zero — none of the three substitutes for another. This is the evaluation-layer payoff of upper-half §4.3's claim that "the three primitives are semantically independent". If the reviewer asks "which primitive drove the improvement", v5 answers line by line; v4 could not.

### 2.5 Decision evidence $E_{\mathrm{decision}}$: Contract Ablation Gap (CAG) + Oracle baseline + Retraining protocol (v5 qualifier)

> **Section anchor**: §2.0 Table 1 Layer = **Decision** (CAG utility sub-slice) · Table 2 Layer = **Utility** (paired with matched null control to prevent OOD sensitivity from masquerading as contract-use) · training-time counterparts **§1.2 augmentation + retraining protocol** (distinguishing CAG$^{\mathrm{fixed}}$ from CAG$^{\mathrm{retrained}}$).

**Given a specific collapse operator $\mathrm{collapse}_X$ on the contract** (which silently folds layer $X$ of contract structure):

$$\mathrm{CAG}_X \;=\; J_{\mathrm{full}} \;-\; J_{\mathrm{collapse}_X}, \qquad J_{\mathrm{full}} \equiv J(\pi_\theta \mid \hat S),\;\; J_{\mathrm{collapse}_X} \equiv J(\pi_\theta \mid \mathrm{collapse}_X(\hat S)),$$

with four collapses each corresponding to one invariant:

- $\mathrm{collapse}_{\mathrm{hyp}}$: fold the hypothesis set into a single Gaussian or a point estimate.
- $\mathrm{collapse}_{\mathrm{age}}$: flatten every channel's $\alpha$ / `health` / `latency` / $\iota$.
- $\mathrm{collapse}_{\mathrm{prov}}$: drop `contributing_mask` and `correlated_with`, let policy attend indiscriminately.
- $\mathrm{collapse}_{\mathrm{neg}}$: drop $\Lambda(E; H, \mathcal O)$.

**The clause v5 has to lock down (reviewer caught v4's hidden problem):**

> **The collapsed condition is evaluated with the same trained policy, without retraining, unless explicitly defined otherwise.**

Because **collapsing on a trained policy** and **retraining a policy on a collapsed representation** are two different questions:

- **No retraining (this piece's default)**: measures **the trained policy's sensitivity to contract information** — this is compliance evidence.
- **Retrained**: measures **whether this representation itself can support the task** — this is architecture / representation comparison.

Both are legitimate but cannot share one symbol. v5 defines two variants explicitly:

$$\mathrm{CAG}_X^{\mathrm{fixed}} \;=\; J_{\mathrm{full}}^{\theta^*} - J_{\mathrm{collapse}_X}^{\theta^*},\qquad \mathrm{CAG}_X^{\mathrm{retrained}} \;=\; J_{\mathrm{full}}^{\theta^*} - J_{\mathrm{collapse}_X}^{\theta^*_X},$$

where $\theta^*$ is the original trained parameters and $\theta^*_X$ is a fresh parameter set **retrained under collapsed representation $X$**. **This piece's compliance evidence uses only $\mathrm{CAG}^{\mathrm{fixed}}$; $\mathrm{CAG}^{\mathrm{retrained}}$ is an architecture-paper question, not this piece's.**

**v6 addition · Matched null control** — this is the step the reviewer pressed on hardest. The $\mathrm{CAG}^{\mathrm{fixed}}$ above has a **very easy misreading**: after $\mathrm{collapse}_X$, the input the policy receives is likely **no longer in the training distribution**. For example, during training the hypothesis payload looks like

```text
hypothesis = [(μ1, Σ1, w1), (μ2, Σ2, w2), ...]
```

After collapse it becomes

```text
collapse_hyp → a single Gaussian (μ̄, Σ̄)
```

So what the policy actually encounters may **not** be "the contract information has been removed"; it may just be **"the input format suddenly became something the model has never seen"** — then $\mathrm{CAG} > 0$ can be pure **OOD sensitivity**, with nothing to do with whether contract semantics is being used. This version therefore adds a **matched null control** to CAG:

**Format-preserving null intervention** $S \mapsto \tilde S$ satisfies four properties:

- same **shape** as the collapsed version (also folded into "a single Gaussian");
- same **marginal distribution** as $S$ (e.g. $\tilde\mu$ sampled from one component of the original mixture according to weights);
- **decision-relevant information preserved** (e.g. retain the top-1 hypothesis's mode identity);
- **irrelevant semantics changed** (e.g. apply a random permutation to the hypothesis index, shuffle the component's provenance tag, keeping shape and marginals intact).

Run the same $\theta^*$ and obtain $\Delta J_{\mathrm{null}} = J_{\mathrm{full}}^{\theta^*} - J_{\mathrm{null}}^{\theta^*}$. **A benchmark reporting CAG must show $\Delta J_{\mathrm{null}}$ alongside it**:

$$\boxed{\;\text{CAG counts as evidence of contract-use only when } \Delta J_{\mathrm{contract}} \gg \Delta J_{\mathrm{null}};\;\text{otherwise it is OOD sensitivity}.\;}$$

This step peels the "policy just dislikes format change" alternative hypothesis out of CAG. One concrete control experiment the reviewer suggested for §2.5 shortcut detection: **retrain a control policy on a version of the training set where the correlation between `age` and `task difficulty` is broken** (shuffle `age` relative to difficulty labels), then observe whether §1.1 Type I/II pass and the CAG relationship changes — if correlation-shuffling drives CAG to zero, the original CAG was a shortcut; if CAG survives the shuffle, the policy is genuinely reading contract semantics.

**Reviewers are also right that $\mathrm{CAG}_X$ alone cannot distinguish "policy not using contract" from "contract not informative enough for this task", nor rule out a shortcut.** This version also adds an **oracle privileged-state baseline**:

$$J_{\mathrm{oracle}} \;=\; J(\pi^{*}_{\mathrm{oracle}} \mid s^{\mathrm{priv}}), \qquad \mathrm{Gap}_{\mathrm{oracle}} \;=\; J_{\mathrm{oracle}} - J_{\mathrm{full}}.$$

**The previous version's four-quadrant interpretation was actually too strong** — the cell "CAG high, oracle slightly above full ⇒ policy really reads contract" is not mathematically supported. $\mathrm{CAG} > 0$ only says **removing this structure drops task utility**; it does not say the policy uses it through a correct semantic mechanism. The real situation is often that training data has $\alpha \leftrightarrow \text{task difficulty}$ heavily correlated, and the policy learns a shortcut $a = f(\alpha)$ rather than $a = f(\text{actual sensor staleness semantics})$. This version downgrades the table to **"what each observation supports"**:

| Observation | What it supports |
|---|---|
| CAG high | Contract structure is useful for the task (may come from semantic use, or from a shortcut) |
| CAG ≈ 0 | Contract structure may not be necessary for this task |
| CAG high + §1.1 Type I/II pass | **More evidence** for semantic use |
| CAG high + intervention fail | **Likely a shortcut** — go back and check training distribution in upper-half §4.3 / §1.2 |
| CAG ≈ 0 + oracle gap high | Contract has information the policy is not using — locate which layer dropped it via upper-half §0.2.2's three-loss decomposition ($L_{\mathrm{declared}} / L_{\mathrm{projection}} / L_{\mathrm{decision}}$) |

That is:

$$\boxed{\;\mathrm{CAG} \;\neq\; \text{contract understanding}.\;}$$

$$\boxed{\;\mathrm{CAG} \;=\; \text{task-conditional utility sensitivity}.\;}$$

**CAG is a decision-layer aggregate; it must be viewed jointly with $E_{\mathrm{semantic}}$ (invariance / equivariance) + $E_{\mathrm{representation}}$ (conditional probe + HPC/HSS) + §1.1 Type I/II controlled response** — only the conjunction is evidence of semantic use. This version demotes CAG from "the aggregate" back to "decision-layer aggregate", so the four compliance evidence types stay complete.

**§2.5 closer · Note: this section's oracle and §2.2 HPC's $\mathcal{A}^*_k$ are two distinct oracle objects** (the reviewer asked for one explicit sentence so that a benchmark report does not merge the two "oracles" into a single total).

| | §2.2 HPC's $\mathcal{A}^*_k$ | This section's oracle $\pi^*_{\mathrm{oracle}}$ |
|---|---|---|
| **Object type** | **action set** — the optimal / admissible action set in world $k$ | **policy** — a policy allowed to read the privileged state $s^{\mathrm{priv}}$ directly |
| **Defined by** | Simulator world $k$'s ground-truth state + reward | A planner / expert run on the simulator's privileged state, producing $\pi^*_{\mathrm{oracle}}$ |
| **What it measures** | $U(\pi_\theta, \mathcal{A}^*_k) = \Pr_{a \sim \pi_\theta}[a \in \mathcal{A}^*_k]$ — the probability that policy outputs fall inside world $k$'s optimal action set under a **counterfactual world** | $J_{\mathrm{oracle}} = J(\pi^*_{\mathrm{oracle}} \mid s^{\mathrm{priv}})$ — the **task-utility upper bound on the same eval distribution** when all state is visible |
| **Which evaluation layer (Table 2 in §2.0)** | **Utility** (response legality belongs to $E_{\mathrm{contract}}$; see the Behavioral-use row of §2.0 Table 2) | Utility-layer upper bound ($\mathrm{Gap}_{\mathrm{oracle}} = J_{\mathrm{oracle}} - J_{\mathrm{full}}$ scales "how much contract information the policy is not using") |
| **Requires counterfactual rollout?** | **Yes** — joint re-render of $(O_k, \hat S_k)$, i.e. $T_k^{\mathrm{world}}$ | **No** — original eval distribution with privileged input |
| **Can the two be merged into one number?** | **No** | **No** |

One-line summary: **HPC's $\mathcal{A}^*_k$ is a set; the oracle baseline's $\pi^*_{\mathrm{oracle}}$ is a policy; they are two orthogonal oracles** — one attacks "policy just dislikes format change", the other attacks "policy learned an age↔difficulty shortcut" — and both are needed to peel those alternative hypotheses off CAG. **Benchmark reports must present them side-by-side, not combined into a single score**: collapsing them would simultaneously lose HPC's counterfactual-world capability signal and the oracle gap's current-eval upper-bound signal, which in §2.0 Table 2 live on different evaluation layers and answer different questions.

### 2.6 Safety evidence $E_{\mathrm{safety}}$: Constraint certification intervention (v5 three-state)

> **Section anchor**: §2.0 Table 1 Layer = **Safety** (constraint certification intervention) · Table 2 Layer = **Safety** (independent of utility) · training-time counterpart **§1.3 three-state filter** (safe → relaxation permitted / unsafe → tighten or stop / unknown → conservative fallback).

Actively drive §1.3's $\mathrm{certification}_j$ to **unsafe** or **unknown** in deployment / semi-simulation (inject calibration drift, cut observability, push $\Lambda(E^-; H, \mathcal O)$ high), and check whether the safety filter **enters the correct guardrail at the correct moment**, and whether guardrail activation is **attributable to which component of certification**. **The key criterion is no longer "did the filter stop the policy", but "did the filter tighten the constraint"** (§1.3 has locked this down: invalid evidence alone cannot justify relaxing the constraint). The three correct responses are fallback, conservative tightening, stop; the **wrong response is relaxing on the strength of invalid / unknown evidence alone**. One observed relaxation caused by unknown / invalid evidence by itself ⇒ $E_{\mathrm{safety}}$ fails outright. Conversely, if an independent valid evidence stream (e.g. lidar) has already pushed certification to **safe**, letting the filter relax is a **correct response** — v4 would have misflagged this case; v5, with three states, does not. This type is the direct instantiation of §1.3's interface, and the real test of whether the piece's interface proposal is **actually deployable**.

The four types together form a **multi-evidence compliance argument**: **semantic measures "does it respond by rule", representation measures "is the information still there (and independent of raw observation)", decision measures "did it use it / is there a shortcut", safety measures "does the guardrail tighten when things degrade"**. None of them replaces an end-to-end success rate — they measure the policy-side **read-completeness** of the contract, not the policy's **expressive power**. This aligns fully with upper-half §0.3 Claim 3: **contract compliance must be tested by controlled intervention and supported by a conjunction of four evidence types**.

## 3. A minimal executable interface sketch (Python skeleton, v6 revised)

Combine the upper-half §4 primitives and §2 four evidence types into a Python class skeleton. **Not a concrete policy; a readable interface contract**.

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
        action level (see upper-half §0.2.2: three semantic losses + one safety obligation).

        Benchmark note (v6 P1-12): when `mode='posterior_sample'` is used inside an
        intervention benchmark (T_i vs T_j), the caller MUST pass a shared `benchmark_rng`
        so both arms consume common random numbers. Otherwise the observed distance
        D(π(T_i S), π(T_j S)) mixes intervention effect with sampling noise and
        contaminates HSS / SDS / E_contract.
        """
        slots = {}
        for field_name in schema.fields:
            h = self.contract[field_name]   # hypothesis set: [(mu_i, Sigma_i, w_i)]

            # --- mode_select: mass-preserving top-k (NOT calibration-aware) ---
            if mode == "map":
                mu, Sigma, w_payload, residual_declared = h.most_likely().mu, h.most_likely().Sigma, None, None
            elif mode == "posterior_sample":
                rng = benchmark_rng or h.default_rng   # v6: shared RNG in benchmark mode
                sample = h.sample(rng=rng)
                mu, Sigma, w_payload, residual_declared = sample.mu, sample.Sigma, None, None
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
                # component requires separate mu/Sigma/likelihood specifications (see upper-half §4.1).
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
            # NOT the canonical default of dependency_aware_fusion (upper-half §4.3.2 v5).
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
                # still applies at its normal margin. Do NOT skip this constraint.
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

Three caveats written in stone:

- **(i)** This is not the only way to read; **`cfg` is a materialization of upper-half §0.2's $q_\pi$, but the actual declared quotient is the pair $(\mathcal C_\pi, Q_{\mathcal C_\pi})$** — the same contract should be projected with different `cfg` by a VLA and a Diffusion Policy, and the optimal `uncertainty` differs between engineered-state heads and visual-latent heads. **The interface spec must display a $Q_{\mathcal C_\pi}$ list**, otherwise $q_\pi$ degenerates back into encoder behavior. **Schema compatibility must also fail-closed or run an explicit adapter (upper-half §0.2.3)** — the v6 code enforces this via `raise SchemaCompatibilityError`, not just by recording a compatibility object; otherwise when $\mathcal C$ upgrades to v2, new fields get silently dropped and the piece falls straight into the undeclared semantic loss it criticizes.
- **(ii)** This interface **only addresses the input side**; §1.1's three-tier intervention constraints, §1.2's observed-metadata-only aug, §1.3's three-state certification direct wiring — if any one is left unchanged, $(\mathcal C_\pi, Q_{\mathcal C_\pi})$ may be written beautifully and still be routed around by $\pi_\theta$'s training dynamics ($L_{\mathrm{decision}}$ blows up directly). **Three losses plus one safety obligation — miss any one and the interface does not hold**. Recall v6's wording: the three losses sit at **separately auditable sites** (sequentially coupled through $q_\pi \to \Pi_\pi \to \pi_\theta$), not statistically independent losses.
- **(iii)** `dependency="logit_bias_learned"` is a **convenient implementation candidate on transformer-family backbones, not the canonical default** — the field-graph → token-graph compilation problem ($R_{\text{field}} \to R_{\text{token}}$) itself is unsolved (upper-half §4.3.2 v5 caveat). MLP heads use `covariance_fusion`, Kalman / factor-graph fusion uses `covariance_fusion`, grouped latent uses `hierarchical_mixture` — **what is truly recommended is the `dependency_aware_fusion` primitive, not any one of its realizations**.

### 3.1 Head ↔ §2 evidence mapping (benchmark-side pseudo-stub)

The skeleton above defines what each head's knobs look like on the **reader** side; it does not yet say "which output of which knob a benchmark should feed to which §2 evidence call". This subsection wires that line up — **every knob maps to a specific §2.0–§2.6 evidence site**, so no field is left dangling.

| §3 head / method output or knob | Consumed by which §2 evidence | Concrete consumption (mapped to §2.0 Table 1 site / Table 2 layer) |
|---|---|---|
| `StructuredStateView.__init__`'s `compatibility` | §2.0 Table 1 **Contract** row · fail-closed `SchemaCompatibilityError` | schema audit: `assert compat.accepted` or run an explicit adapter; otherwise the whole pipeline is unauditable |
| `StructuredStateView.declared_coverage_loss()` | §2.0 Table 1 **Declaration** row · $L_{\mathrm{declared}}$ via query subsumption | directly reports $L_{\mathrm{declared}}$; benchmark side just lists non-zero $q$ weighted by $w_q$ |
| `project`'s `topk_weight_mode` / `residual_mode` | §2.2 upper-half conditional probe (**Retention**) + §2.5 $\mathrm{CAG}^{\mathrm{fixed}}_{\mathrm{hyp}}$ | Fix $o$, swap the two knobs, check whether $z_\pi$ still independently carries field information (Retention layer); apply $\mathrm{collapse}_{\mathrm{hyp}}$ to the same trained $\theta^*$, measure $\mathrm{CAG}^{\mathrm{fixed}}_{\mathrm{hyp}}$ side by side with §2.5 matched-null $\Delta J_{\mathrm{null}}$ (requiring $\Delta J_{\mathrm{contract}} \gg \Delta J_{\mathrm{null}}$) |
| `project`'s `staleness` / `uncertainty` | §2.3 **SDS** (temporal slice) + §2.2 conditional probe | sample $\alpha$ along $\preceq^{\mathrm{declared}}_{\mathcal C_\pi}$, measure $V_{\mathrm{order}}$ and $V_{\mathrm{trans}}$, and check $I(\alpha; z_\pi \mid o)$ retention |
| `project`'s `provenance` / `dependency` / `negative_evidence` | §2.4 $\Delta J_{\mathrm{where}} / \Delta J_{\mathrm{dep}} / \Delta J_{\mathrm{neg}}$ + §2.5 $\mathrm{CAG}^{\mathrm{fixed}}_X$ | toggle each of the three knobs from `ignore` to `harden / covariance_fusion / condition`; evaluator computes utility delta and runs matched null control to rule out OOD sensitivity |
| Whole `PolicyInput` output by `project`, transformed by frame / coordinate / hypothesis permutation $T^{\mathcal C}$ | §2.1 **Type I equivariance** | measure $D_{\mathcal A}(\pi_\theta(T\hat S),\, T^{\mathcal C}_\pi \pi_\theta(\hat S))$; $\mathrm{Equiv}\to 0$ is a hard requirement. This row is about the **structural** equivariance of the whole `PolicyInput`, not about any single slot |
| `PolicyInput.slots[...].trust` | §2.3 **SDS** (Type II realization) | trust $\tau(\alpha,\ell,h,v,\iota)$ varies along $\preceq^{\mathrm{declared}}_{\mathcal C_\pi}$; check whether $r(\pi_\theta(\cdot))$ satisfies §1.1's declared order-constrained response |
| `PolicyInput.slots[...].w_payload` | §2.2 **HPC / HSS** (counterfactual-hypothesis side) | when `topk_weight_mode` swaps or hypotheses are permuted, measure $U(\pi_\theta, \mathcal A^*_k)$ and $\mathrm{HSS}^c$ (under §2.2 competence gate) |
| `PolicyInput.slots[...].residual_declared` | §2.2 upper-half conditional probe (**Retention**) | as `residual_mode` climbs `drop → mass_only → sufficient_stats`, check whether $I(\text{field}; z_\pi \mid o)$ rises |
| `SafetyFilterHead.forward` three-state branch | §2.6 **Constraint certification intervention** | benchmark actively drives `constraint.certification` to `unsafe` / `unknown` (inject calibration drift, cut observability, raise $\Lambda(E^-; H, \mathcal O)$); assert `a_applied` **tightens** (not relaxes) and that the loop composes constraints via **intersection**, not first-return |
| `ContractAwarePolicy.forward`'s final `ActionDistribution` | §2.2 (A) $E_{\mathrm{contract}}(T^{\mathrm{contract}})$ + §2.2 (B) HPC($T^{\mathrm{world}}$) + §2.2 HSS$^c$ + §2.5 CAG + §2.5 oracle gap | consumed by the five-layer hierarchy (Retention / Sufficiency / Behavioral use / Utility / Safety); benchmark report keeps them **side by side, not merged** (see §2.5 closer) |

**Benchmark-side pseudo-stub** (**not a full implementation — just a paper-nail tying §3 heads to §2 evidence calls**):

```python
def run_contract_benchmark(
    policy, dataset, C_pi, Q_C_pi, Q_req,
    T_contract_set, T_world_set, collapse_set, alpha_grid, oracle_pi,
    tau_E, delta, eps,
):
    R = {}  # evidence report keyed by §2.0 Table 1 sites and Table 2 layers

    # ---- Contract site (§2.0 Table 1, row 1) --------------------------------
    R["schema_compat"] = audit_schema_compatibility(dataset, C_pi)  # fail-closed

    # ---- Declaration site (L_declared, §2.0 Table 1, row 2) -----------------
    R["L_declared"] = StructuredStateView(
        dataset.sample_contract(), C_pi, Q_C_pi, Q_req
    ).declared_coverage_loss()

    # ---- Projection site: Retention + Sufficiency (§2.2 upper half) ---------
    R["retention_cond"] = conditional_probe_I(policy, field="alpha", given="obs")
    R["L_projection"]   = conditional_MI(policy, Y_C_pi, S_hat,
                                         Z_pi=..., O=..., L=...)

    # ---- Decision site (§2.1 – §2.5) ----------------------------------------
    R["Equiv_TypeI"]  = [D_A(policy(T(S)), T_a(policy(S)))
                         for T in T_contract_set]                        # §2.1
    R["E_contract"]   = [D_A(policy(T_c(S)), R_C(T_c))
                         for T_c in T_contract_set]                      # §2.2 (A)
    R["HPC"]          = [Pr_a_in_A_star(policy, T_w, k)
                         for k, T_w in enumerate(T_world_set)]           # §2.2 (B)
    R["HSS_c"]        = hypothesis_separation(policy, T_c=T_c, D=D_A,
                         competence_gate=(R["E_contract"] < tau_E))      # §2.2
    R["SDS_V_order"]  = staleness_order_violation(policy, alpha_grid, r, delta)
    R["SDS_V_trans"]  = trans_section_distance(policy, alpha_grid, r)    # §2.3
    R["dJ_where_dep_neg"] = source_slice_ablations(policy, dataset)      # §2.4
    R["CAG_fixed"]    = {X: J(policy, S) - J(policy, collapse_X(S))
                         for X in collapse_set}                           # §2.5
    R["dJ_null"]      = J(policy, S) - J(policy, format_preserving_null(S))
    R["Gap_oracle"]   = J(oracle_pi, s_priv) - J(policy, S)               # §2.5 upper bound

    # ---- Safety site (§2.6) --------------------------------------------------
    R["O_safety_cert"] = constraint_certification_intervention(
        policy.safety_head,
        drive_to={"unsafe", "unknown"},   # benchmark actively injects
        expect="tighten_or_fallback",     # observe tightening, not relaxation
    )
    return R
```

**How to read the stub**: every comment aligns with one site in §2.0 Table 1 or one layer in §2.0 Table 2 — the benchmark report is exactly this dict, pretty-printed under Contract / Declaration / Projection / Decision / Safety, each site listing its §2.x metrics. **Note the `HSS_c` field's `competence_gate=(R["E_contract"] < tau_E)`** — the gaming scenario the reviewer flagged ("a random policy can also inflate HSS") is blocked precisely by this gate; the benchmark report must display `E_contract` and `HSS` **side by side**, not merged into one composite. **Similarly**, `Gap_oracle` (this section) and `HPC` (§2.2) are two orthogonal oracles; see the §2.5 closing note — they must not be merged.

## 4. Closing: four compliance evidence types, a five-layer evaluation hierarchy, one executable interface

The upper half (9/15) already stood the policy-side interface as an auditable semantic object — Consumer Contract triple, three tiers of preservation, three semantic losses + one safety obligation, four interface-mismatch failure modes, three families of contract-read primitives. This piece (9/16) delivers the **measurement layer and the implementation layer** — four compliance evidence types, a five-layer evaluation hierarchy, and a minimal executable interface skeleton. Both halves together push "policy-side interface" from **concept** to **protocol**.

Three closing claims of this piece, aligned with §0.3:

> **Claim A · Compliance is a multi-evidence argument, not a score.** §2's four evidence types — semantic / representation / decision / safety — answer different questions, **cannot substitute for each other**, and **cannot be compressed into a single scalar**. $E_{\mathrm{semantic}}$ tests "are the response rules correct?", $E_{\mathrm{representation}}$ tests "is the information still there (and independent of the raw observation)?", $E_{\mathrm{decision}}$ tests "is it actually used, or is there a shortcut?", $E_{\mathrm{safety}}$ tests "when evidence becomes unknown or invalid, does behavior tighten?". None of them can replace end-to-end success rate — they measure the **reading fidelity** of the contract on the policy side, not its **expressiveness**.

> **Claim B · Evaluation layers do not imply each other.** §2.0's five-layer hierarchy **Retention ≠ Sufficiency ≠ Behavioral use ≠ Utility ≠ Safety** with its four non-implications is the fundamental motivation for §2.2 splitting $E_{\mathrm{contract}}(T^{\mathrm{contract}})$ / HPC($T^{\mathrm{world}}$), §2.5 splitting CAG$^{\mathrm{fixed}}$ / CAG$^{\mathrm{retrained}}$, and §2.6 isolating the constraint certification intervention. Any attempt to substitute one layer's score for another layer's conclusion is a shortcut.

> **Claim C · The interface is where the bugs hide, not the architecture.** The three §3 Python bugs v6 fixes — missing `super().__init__()`, fail-open schema-compatibility check, and top-$k$ API conflating two orthogonal knobs into one string — are none of them architecture issues; they are all interface issues. This class of bug silently flattens contract semantics; the loss curve stays quiet, the eval score stays quiet, and only §2's four-evidence convergence can surface them.

### 4.1 One closing figure

$$\boxed{\;\mathcal C\;\longrightarrow\;(\mathcal C_\pi=(Q_\pi,\mathcal O_\pi,V_\pi),\;Q_{\mathcal C_\pi})\;\longrightarrow\;q_\pi\;\longrightarrow\;\Pi_\pi\;\longrightarrow\;\pi_\theta\;\longrightarrow\;g_{\mathrm{safety}}=\textstyle\bigcap_j g_j\;\longrightarrow\;\mathcal B_{\mathcal C}\;}$$

$$\boxed{\;\text{Compliance Evidence}=\big\{E_{\mathrm{semantic}},\;E_{\mathrm{representation}},\;E_{\mathrm{decision}},\;E_{\mathrm{safety}}\big\}\;\longrightarrow\;\text{Five-Layer Evaluation Hierarchy}\;}$$

The upper-half boxed thesis — **A policy is a contract consumer, not merely a function approximator** — is delivered here as follows: a contract consumer not only has to answer the upper-half four questions (what may I discard / what did I retain / how should decisions respond / what happens under invalid-or-unknown evidence), it also has to be **auditable by four evidence types**, **evaluable layer by layer with non-implications across the five-layer hierarchy**, and **able to localize every interface bug in §3 to a specific §2 evidence signal**. **Once the interface is right, backbone choice becomes secondary**; when the interface is wrong, every architecture silently flattens contract semantics. The upper half stands the interface; this piece stands the protocol and the skeleton; the next piece stands the benchmark.

### 4.2 Next step: Contract-Preserving Policy Benchmark

Together with 9/15 this piece brings the series to a position where the next article can be a benchmark:

$$\textbf{Contract-Preserving Policy Benchmark}:\;\text{given the same Structured State Contract, how to systematically test whether SAC / PPO / Diffusion Policy / ACT / OpenVLA / }\pi_0\text{ is contract-compliant.}$$

Each loss and each obligation maps to a specific evidence class in §2, each pipeline site maps to a row in the §2.0 table, each §3 interface bug maps to a counterexample in §2. Three items still need to be pinned before the benchmark can run:

1. A **cross-task setting principle** for the $\tau_E$ competence gate in §2.2 — currently a symbol without a rule.
2. A **consistent per-task selection** for the hinge-loss margin $\delta$ in §2.3 and the CAG matched-null margin $\delta$ in §2.5 — the two need to line up.
3. A **concrete shortcut-detection experiment** for §2.5 — deliberately breaking the correlation between observation age and task difficulty so that $\Delta J_{\mathrm{null}}$ can be tuned to expose a "response to staleness" that is really a "response to difficulty".

The benchmark article is where this piece ends and the next begins. In one sentence: **VLA / Diffusion / Flow / ACT / SAC / PPO are only implementation coordinates, no longer theoretical classifications** — what actually decides whether a policy can go to production is whether its interface is correct, whether it can be audited by §2's four-evidence convergence, and whether it produces measurable responses across the five-layer hierarchy of §2.0. The upper half stood the interface; this piece stands the protocol and the skeleton; the next article will stand the benchmark.

## Sources

arXiv IDs below have been verified online; journal-only references are not accompanied by an arXiv link. Grouped by the section they support. **The upper-half Sources A (VLA) / B (Diffusion) / D (SAC / PPO) are not re-listed here — this piece only adds the evaluation and implementation references.**

### C · Uncertainty, calibration, and belief-space references (support §2.2 calibration diagnostic, §2.5 CAG oracle baseline)

- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) (modern networks are over-confident; temperature scaling baseline; the evaluation basis of the §2.2 calibration diagnostic, not a primitive claim)
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels* (PlaNet / RSSM), ICML 2019 · [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) (deterministic + stochastic latent; one adjacent reference for §2.2 conditional probe and §2.5 CAG oracle baseline)
- Hafner et al., *Mastering Diverse Control Tasks through World Models* (DreamerV3), Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) (discrete + continuous mixed latent, KL balancing; one adjacent route for §2.2 HPC($T^{\mathrm{world}}$) counterfactual-intervention. Note: this piece cites DreamerV3 as an **evaluation reference**, not as an implementation of a contract-aware interface)

### E · Series continuations and the upper half (framework input and evaluation spine input for this piece)

- This blog, *After the Contract Stands: What Do VLA, Diffusion Policy and π0 Actually Consume?* · `/en/articles/2026-09-15-policy-side-interface/` (**upper half of this piece**; establishes the framework reused in §0.2 — Consumer Contract triple, three tiers of preservation, three semantic losses + one safety obligation, four interface-mismatch failure modes, three families of contract-read primitives. §1–§3 of this piece directly evaluate and implement on top of that framework)
- This blog, *Splicing is not seeing: what robot multimodal fusion is missing is an interface, not a model* · `/en/articles/2026-09-14-multimodal-fusion-interface/` (Structured State Contract definition, Interface Property Benchmark, degradation chain; upper-half §0.2 $q_\pi$ and this piece's §1 augmentation and §2 evidence build on it)
- This blog, *Only seeing, never touching: why robots lack a hand with feel* · `/en/articles/2026-09-13-tactile-force-sensing/` (four force/tactile control paradigms, closed-loop value; physical motivation for §2.4 source slice and §1.3 three-state safety filter)
- This blog, *Embodied AI Sim-to-Real Methodology (III): evaluation protocol* · `/en/articles/2026-09-12-sim-to-real-evaluation-protocol/` (three-tier evidence, decision-utility dimensions, allocation protocol; **the §2 evaluation spine of this piece is inherited directly from here**)
- This blog, *Embodied AI Sim-to-Real Methodology (I): allocation state* · `/en/articles/2026-09-10-sim-to-real-methodology/` (allocation state $s_t=(b_t,\pi_t,q_t,h_t)$, $\Delta_{\mathrm{queue}}$ vs $\Delta_{\mathrm{processing}}$; upper-half §2/§4.2 define on top of it; §2.3 temporal slice and §1.2 Type II order-constrained response inherit its staleness vocabulary)

### F · Evaluation-protocol methodology references (methods used by §1.1 / §2 / §3)

- Osband et al., *What Type of Code is Relevance in Randomized Control Trials?*, 2018 · [arXiv:1802.08670](https://arxiv.org/abs/1802.08670) (**evaluation methodology** reference; §2.5 matched null control and §2.2 competence gate borrow its null-intervention idea; **this piece does not claim causal equivalence with RCT**)
- Adebayo et al., *Diagnostic Explanations for Deep Learning Models*, ICML 2018 · [arXiv:1806.07537](https://arxiv.org/abs/1806.07537) (**probing-based diagnostic** reference; §2.2 conditional probe belongs to the same family, but the probe in this piece is a **residual MI conditional on $o$**, not raw probing classifier accuracy)
- D'Amour et al., *Underspecification Presents Challenges for Credibility in Modern Machine Learning*, JMLR 2021 · [arXiv:2011.03395](https://arxiv.org/abs/2011.03395) (**evaluation underspecification** reference; §2.0 five-layer hierarchy with its four non-implications is a direct response to the "single score hides assumptions" failure mode)

> **Note**: citations to π0 / OpenVLA / Diffusion Policy families in this piece are **the upper half's analytical citations** (interface critique). This piece does not repeat them; only evaluation- and implementation-side methodology is added here. Upper-half Sources A / B / D are **not re-listed** in this piece.

---

> **Related reading**
>
> - [After the Contract Stands: What Do VLA, Diffusion Policy and π0 Actually Consume?](/en/articles/2026-09-15-policy-side-interface/) — **upper half of this piece**; policy-side interface framework
> - [Splicing is not seeing: what robot multimodal fusion is missing is an interface, not a model](/en/articles/2026-09-14-multimodal-fusion-interface/) — Structured State Contract upstream definition, shared source for 9/15 and this piece
> - [Only seeing, never touching: why robots lack a hand with feel](/en/articles/2026-09-13-tactile-force-sensing/) — tactile / force side; physical motivation for §2.4 source slice and §3 safety filter three-state
> - [Embodied AI Sim-to-Real Methodology (III): evaluation protocol](/en/articles/2026-09-12-sim-to-real-evaluation-protocol/) — the direct source of this piece's §2 evaluation spine
> - [Embodied AI Sim-to-Real Methodology (I): allocation state](/en/articles/2026-09-10-sim-to-real-methodology/) — staleness / queue vs processing vocabulary
> - [VLA and World Models: two diverging routes and their intersection](/en/articles/2026-09-07-vla-world-models/) — macro backdrop for §2.2 HPC($T^{\mathrm{world}}$) and the world-model-side oracle
