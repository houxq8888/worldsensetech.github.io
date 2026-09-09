---
title: 'After the Contract Stands: What Do VLA, Diffusion Policy and π0 Actually Consume?'
slug: "2026-09-15-policy-side-interface"
date: 2026-09-15
draft: false
categories: ["Embodied AI", "Policy Learning"]
tags: ["Embodied AI", "Policy Learning", "VLA", "Diffusion Policy", "π0", "RT-2", "OpenVLA", "Action Tokenization", "Structured State Contract", "Declared Quotient", "Semantic Preservation", "Decision Sufficiency", "Contract Information Loss", "Contract-Read Primitives", "Intervention Consistency", "Equivariance", "Monotone Response", "Likelihood-Side Evidence", "Dependency-Aware Fusion", "Calibration-Aware Top-k", "Constraint-Relevant Validity", "Contract Ablation Gap", "Hypothesis Coverage", "Hypothesis Separation", "Staleness Response Compliance", "Evaluation Metrics"]
description: 'The multimodal-fusion piece stood up the upstream deliverable as a Structured State Contract. This piece asks the dual question: if the estimator really delivers per contract, can the policy side actually consume it. The core object is an explicitly factorized policy projection $\Pi_\pi = e_\pi \circ q_\pi$: first declare which contract distinctions may be dropped ($q_\pi$, the declared quotient), then let the neural encoder perform the actual encoding ($e_\pi$). The thesis is a strict separation between **semantic preservation** (injectivity on the contract quotient) and **decision sufficiency** ($L_{\mathrm{dec}}(\Pi_\pi)$); the two are not equivalent, and the former is strictly stronger. The piece no longer slices along three mutually-exclusive families (VLA / Diffusion / engineered head); instead it decomposes any policy along two orthogonal dimensions — conditioning representation / semantic interface × action head — and gives a seven-row grid. At the interface layer three families of contract-read primitives are proposed: `mode_select` uses a calibration-aware top-$k$ readout with explicit residual mass; `age_gate` places measurement / uncertainty / age / validity / health / latency / trust as seven parallel slots, $\tau_c$ no longer depends on age alone, and $\Sigma/\tau$ is one conservative approximation alongside predictive state propagation; the source structure splits into provenance / dependency / negative evidence, with `dependency_gate` upgraded to **dependency-aware fusion** (correlation ≠ redundancy ≠ double-counting) and negative evidence formalized as an observability-conditional likelihood ratio $\Lambda^-(H)$. Training-side intervention-consistency is split into three tiers: **Type I exact equivariance / Type II monotone response / Type III unconstrained** — change is not assumed to equal compliance. On the evaluation side the four metrics become a **four-level panel** — Semantic / Representation / Decision / Safety — with J_full / J_collapse / J_oracle as three baselines that bound the interpretation of CAG: **CAG measures task-conditional utility of contract structure, not semantic understanding by itself.** The piece closes on three claims: $\Pi_\pi$ determines which contract distinctions remain available downstream; every policy must answer the same interface question; contract compliance must be tested by invariance / equivariance / monotone response / task-conditional utility, not inferred from end-to-end success.'
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

> Picks up from [Splicing is not seeing: what robot multimodal fusion is missing is an interface, not a model](/en/articles/2026-09-14-multimodal-fusion-interface/): that piece reframed multimodal fusion from a "when to fuse" question to a "what is delivered after fusion" question, and stood the deliverable up as a contract object — the **Structured State Contract** — a structured state carrying hypothesis, provenance, observability, availability / validity / age, contact set and negative evidence. It closed on the claim **A good multimodal system must represent disagreement, not merely resolve it.** This piece asks its **dual**: if the upstream really delivers per contract, **can the policy side actually consume it**.

Short answer: **a structured estimator output does not imply a structured policy input**. Between them sits an explicit projection $\Pi_\pi$ which is itself **a semantic interface that can silently destroy semantics**. VLA uses a tokenizer, Diffusion Policy uses image-encoder + proprio concat, classical heads use a hand-designed state vector — the three projections each destroy a different slice of the contract, and **the destruction is silent**: the loss curve still goes down, eval scores still go up, and you cannot see from the training log what has been flattened away. This is not a model-size problem, **it is an interface problem** — but the problem is not "which architecture collapses", **the problem is that no architecture carries an explicit preservation guarantee for contract-relevant semantics**. This distinction is what separates this piece from the usual "architecture critique" genre.

This piece does not push a specific backbone, does not oppose end-to-end learning, and does not claim that one family is inherently better than another. Its thesis is more general and more verifiable: **every policy architecture has to answer — does its input projection preserve the decision-relevant semantics of the upstream contract?** Once that question is on the table, "VLA versus Diffusion Policy" demotes itself from a position to a design decision.

## 0. Framing: the policy projection $\Pi_\pi = e_\pi \circ q_\pi$ is a semantic interface

Set up the whole analysis framework up front; every later section returns to this figure. This section also stands up the piece's real **formal object** — the **two-layer factorization** of $\Pi_\pi$, the **separation** between semantic preservation and decision sufficiency, and a measurable **Contract Information Loss**.

### 0.1 Argumentation chain

```
Structured State Contract  Ŝ_t                             (defined in 9/14)
        │
        ▼
declared quotient  q_π :  Ŝ_t ↦ [Ŝ_t]_{C_π}                (this piece's proposal — explicit declaration of what may be dropped)
        │
        ▼
neural encoding  e_π :  [Ŝ]_{C_π} ↦ z_π                    (all existing policies do this; almost none declare q_π)
        │
        ▼
policy projection  Π_π  =  e_π ∘ q_π
        │
        ▼
question: which decision-relevant semantics are preserved?
        │
        ▼
three contract-relevant invariants
   ├─ mode          multi-hypothesis structure            ──▶  primitive: mode_select
   ├─ temporal      heterogeneous staleness / health      ──▶  primitive: age_gate
   └─ source        provenance / dependency / negative ev. ──▶  primitives: provenance / dependency / negative evidence
        │
        ▼
training (representation-side probe + Type I equivariance / Type II monotone / Type III unconstrained)
deployment (safety filter reads constraint-relevant evidence, not estimator's general-purpose validity bit)
evaluation (four-level panel: Semantic / Representation / Decision / Safety; CAG is only the Decision-layer aggregate and needs an oracle baseline)
```

### 0.2 $\Pi_\pi = e_\pi \circ q_\pi$: declare what may be dropped, then encode

This piece defines $\Pi_\pi$ explicitly as a **two-layer factorizable semantic interface**, not as "the first computation of the model". **This factorization is the most important change relative to the previous version**: the previous draft treated $\Pi_\pi$ as a single projection and a reviewer immediately pointed out that "$L_{\mathcal{C}} = 0$ is not equivalent to injectivity"; this version assigns those two properties to different layers by explicitly introducing the **declared quotient** $q_\pi$.

**Layer q · Declared quotient** — $q_\pi : \hat S \mapsto [\hat S]_{\mathcal{C}_\pi}$ folds the upstream contract into a **declared equivalence-class space**. $q_\pi$ is not a neural network; it is a paragraph in the interface specification — **it answers "which contract distinctions does this policy declare it is allowed to drop"**. It can declare "I preserve hypothesis structure"; it can also declare "I collapse hypotheses to a point estimate but keep age as a separate field"; **the point is that the collapse rule must be written on the interface, not hidden inside the encoder**.

**Layer e · Neural encoding** — $e_\pi : [\hat S]_{\mathcal{C}_\pi} \mapsto z_\pi$ takes the equivalence classes and passes them through encoder / tokenizer / concat into whatever the policy actually consumes. Most existing work has only $e_\pi$ and no explicit $q_\pi$ — that is what this piece attacks.

$$\Pi_\pi \;=\; e_\pi \circ q_\pi.$$

**What must actually be checked is $q_\pi$** — did it merge any of the distinctions that were not allowed to be merged. This factorization upgrades the whole piece from "add metadata to a VLA" to a more general proposal: **contract-aware representation design**.

### 0.2.1 Semantic preservation ≠ Decision sufficiency

This subsection is the theoretical anchor of the piece; the two commonly conflated properties must be pulled apart.

Fix a set of **contract-relevant decision variables** $Y_{\mathcal{C}}$ that the policy serves — the quantities downstream controller / planner / safety filter / diagnostics will read, e.g. "which object is this track", "how many newtons is this contact force", "how old is this measurement", "is this channel still valid". Define a semantic-equivalence relation on $\hat S$:

$$\hat S \sim_{\mathcal{C}} \hat S' \quad \Longleftrightarrow \quad \forall\, Y_{\mathcal{C}}\text{-relevant query},\;\hat S \text{ and } \hat S' \text{ give the same answer}.$$

**Property A · Semantic preservation** — the projection does not irreversibly collapse semantically distinct contracts:

$$\hat S \not\sim_{\mathcal{C}} \hat S' \quad \Longrightarrow \quad \Pi_\pi(\hat S) \not\equiv \Pi_\pi(\hat S').$$

This effectively requires $\Pi_\pi$ (or, more precisely, $q_\pi$) to be **injective on the contract-semantic quotient**. This is a **strong property**.

**Property B · Decision sufficiency** — mutual information measures whether the information needed for currently-defined decision variables is preserved:

$$L_{\mathrm{dec}}(\Pi_\pi) \;=\; I(\hat S;\, Y_{\mathcal{C}}) \;-\; I\!\big(\Pi_\pi(\hat S);\, Y_{\mathcal{C}}\big).$$

By the data processing inequality, $L_{\mathrm{dec}} \ge 0$; $L_{\mathrm{dec}} = 0$ means $Y_{\mathcal{C}} \perp\!\!\!\perp \hat S \mid \Pi_\pi(\hat S)$ — i.e. $\Pi_\pi(\hat S)$ is a **sufficient representation** for $Y_{\mathcal{C}}$.

**The key point**: Property A and Property B are not the same thing. $L_{\mathrm{dec}} = 0$ only guarantees sufficiency; **it does not imply injectivity** — you can absolutely fold semantically distinct $\hat S, \hat S'$ to the same $\Pi_\pi$ output while still being sufficient for $Y_{\mathcal{C}}$ (as long as $Y_{\mathcal{C}}$ itself does not distinguish them). **Semantic preservation is strictly stronger than decision sufficiency.**

The requirement can then be written very cleanly:

> **A policy may drop information, but must either (a) preserve contract semantics, or (b) explicitly declare a sufficient quotient of the contract.**

That is: **either keep the whole quotient structure, or write "which layer was dropped" into the interface and prove that the declared quotient is sufficient for the declared $Y_{\mathcal{C}}$**. This also explains why "any projection must lose information" is not a rebuttal to this piece: information loss is not itself an interface violation; **silently dropping decision-relevant contract semantics — without declaring — is the violation**.

Whether $e_\pi$ encodes cleanly is one question, whether $q_\pi$ allows the drop is another — a reviewer can now audit them separately. That separation is the largest upgrade relative to the previous draft.

### 0.3 Three boxed claims of this piece

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. The policy projection $\Pi_\pi$ is **a semantic interface**: it determines which contract distinctions remain available to downstream decision making — not "the first forward computation of a model".

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state heads, latent visuomotor policies, and VLA / diffusion / flow policies use different conditioning and action-generation mechanisms — but they all face the **same interface question**: which contract distinctions are intentionally preserved, which are compressed, and which are discarded? The target of criticism is the interface contract, not the model architecture.

> **Claim 3 · Contract compliance should be tested by controlled intervention, not inferred from end-to-end success.** End-to-end success measures whether the policy works, not whether it interpreted contract semantics correctly. Contract compliance requires a battery of **controlled tests** — invariance / equivariance / monotone response / task-conditional utility under contract interventions — **and needs an oracle baseline to bound the interpretation of each metric**.

## 1. Two dimensions instead of three families: conditioning representation / semantic interface × action head

**An earlier draft sliced "VLA / Diffusion Policy / engineered head" as three mutually-exclusive families**; that taxonomy is too coarse — π0 is VLA + flow matching and lands in both columns. This section switches to **two orthogonal dimensions**; a policy family's choice becomes a coordinate on a grid rather than a stance.

### 1.1 Two orthogonal dimensions

**Dimension 1 · Conditioning representation / semantic interface** — the shape in which the policy reads upstream information. **Note this column is not a pure representation taxonomy: structured contract carries representation + semantic schema + provenance + validity + semantics simultaneously**, hence the column must be labeled "representation / semantic interface" — otherwise reviewers will point out that the layers are mixed. Four typical values:

- **engineered state** (a fixed-length vector hand-written per task, $s_t \in \mathbb{R}^d$)
- **visual latent** (encoder output, $z_t = f_\phi(o_t)$)
- **multimodal token** (discrete tokens in a VLM vocabulary; image and language share the space)
- **structured contract** (an explicit $\hat S_t$; fields are enumerable and jointly readable by controller / policy / safety filter; carries schema and semantic guarantees)

**Dimension 2 · Action head** — how the policy produces an action distribution. Five typical values:

- **deterministic / Gaussian / mixture** (the SAC line, mean + variance / GMM)
- **autoregressive token** (a discrete vocabulary, RT-2 / OpenVLA)
- **generative sequence decoder** (CVAE → chunked, the ACT line — **neither diffusion nor flow**)
- **diffusion** (multi-step denoising, Diffusion Policy)
- **flow matching** (velocity regression over continuous action chunks, π0)

**The previous draft's placement of ACT under a combined "generative action chunk" bucket could be mis-read as "ACT belongs to the diffusion family"; this version gives ACT its own row alongside diffusion and flow matching.**

### 1.2 Grid

| Representative policy | Conditioning representation / semantic interface | Action head | Where the contract is destroyed (analysis under this piece's schema) |
|---|---|---|---|
| Classical SAC/PPO head | engineered state | Gaussian / deterministic | Minimal; fields hand-written; but state design is an art and cross-task transfer is poor |
| Visual-obs PPO / DrQ | visual latent | Gaussian | Once $z_t$ is squeezed, provenance / age / hypothesis are usually already gone |
| Diffusion Policy (Chi 2023) | visual + proprio latent | diffusion | Under this schema: action-side multimodality is fine, but state-side multi-hypothesis has no slot |
| ACT / ALOHA (Zhao 2023) | visual + proprio latent | **generative sequence decoder** (CVAE → chunked) | Conditioning similar to Diffusion Policy; the action generator is a different family |
| RT-2 | multimodal token (VLM) | autoregressive action tokens | Frame / reference_point / convention are tokenized away |
| OpenVLA | multimodal token + proprio | autoregressive | Same as RT-2, proprio adds one channel, but the contract fields still have nowhere to go |
| π0 (Black 2024) | multimodal token (VLM conditioning) | **flow matching** (continuous action chunk) | Under this schema: action is continuous, but conditioning is still VLM tokens; no explicit slot for hypothesis / provenance / age / negative evidence |

One key distinction: **Continuous actions do not imply structured state semantics.** π0's action head is a flow-matched continuous chunk and sounds "structured", but its conditioning representation is a VLM token stream. **The wording matters**: **Under the Structured State Contract defined here, π0's conditioning interface does not expose an explicit slot for hypothesis, provenance, age, or negative-evidence semantics.** This is an **analysis under this piece's schema**, not a limitation the π0 paper claims about itself. π0's paper facts are pretrained VLA + proprio token + noisy action chunk + flow matching; the contract-level critique comes from this piece's analytical lens. The same caveat applies to the "Where the contract is destroyed" column for Diffusion Policy and OpenVLA — **those cells are a mix of paper facts and this piece's interface analysis, not admitted limitations of the original papers**.

One caveat: **this is not a ranking of "which combination is best"**. Engineered state + Gaussian is still the low-dimensional control baseline champion; multimodal token + flow matching is still the only realistic path for open-semantic settings. What this piece cares about is **for every combination, is there a written-out $q_\pi$ in $\Pi_\pi = e_\pi \circ q_\pi$**. In most existing work the answer is "no" — **not because a family is inherently bad, but because this $q_\pi$ layer has never been treated as an interface design problem**.

## 2. What "State" means across policies

This section is **terminology cleanup**. "State" in the embodied-AI literature means at least five mutually-distinct things, and a policy family's choice is often implicitly "which of these five it commits to".

**$\pi_{\mathrm{obs}}$ — raw observation**: images, point clouds, force/torque streams. The **input form** of most imitation-learning pipelines, but typically not the state the policy actually uses internally (it gets encoded away).

**$z_t = f_\phi(o_{:t})$ — encoded latent**: the low-dimensional representation post-encoder. Diffusion Policy's and VLA's image tokens, and RSSM states in world models, all fall here. The contract has already **undergone one projection** at this stage; observability and provenance are usually already lost.

**$b_t$ — belief / posterior**: the explicit belief state of the POMDP line. The "multi-hypothesis + posterior weight" structure in the contract maps most cleanly here. But mainstream VLA / Diffusion Policy do not model belief explicitly — it is only implicitly approximated by the encoder.

**$s_t = (b_t, \pi_t, q_t, h_t)$ — the allocation state of 9/10 Part 1**: belief + policy + budget + hardware. This is the state at the **decision layer**, not the state at the policy's input.

**$\hat S_t$ — structured state (contract)**: the contract object defined in 9/14.

A full chain:

$$\pi_{\mathrm{obs}} \;\xrightarrow{\;\text{encoder}\;}\; z_t \;\xrightarrow{\;\text{abstraction}\;}\; \hat S_t \;\xrightarrow{\;\text{posterior}\;}\; b_t \;\xrightarrow{\;\text{allocation}\;}\; s_t$$

Back to the language of §0.2: the real policy-side question is **not** "can I consume longer token sequences", it is **"how far along $\pi_{\mathrm{obs}} \to z_t \to \hat S_t \to b_t \to s_t$ am I willing to commit, and did I write the permitted drop into $q_\pi$ so that $e_\pi$ is at least sufficient for $Y_{\mathcal{C}}$"**. Engineered state stops at $\hat S_t$, visual latent stops at $z_t$, multimodal token effectively stops at a tokenizer one layer past $\pi_{\mathrm{obs}}$ — **three stopping points correspond to three values of $q_\pi$, not to "one smart, one dumb"**.

**The common misuse of the phrase "multimodal fusion" on the policy side** is treating the cross-attention in $\pi_{\mathrm{obs}} \to z_t$ as if it were "already doing multimodal state estimation". It is not. Real state abstraction requires the fields inside $\hat S_t$ to **carry consistent semantics across sensor families and be jointly readable by four consumers — controller / policy / world model / diagnostics** — 9/14 §6 has already established this convention; what this section adds is **asking from the policy side once more: was $q_\pi$ declared explicitly, or was it silently replaced by $e_\pi$?**.

## 3. Four interface-mismatch failure modes

Once the §1 coordinates are instantiated in a concrete policy, contract semantics show up as **four concrete failure modes**. None of the four are theoretical worries; they are **what actually breaks in deployment**. All four are **facts about the interface**, not personality defects of any single family.

### 3.1 Failure 1: Multi-hypothesis silently collapsed

Inside the contract a single physical quantity may have multiple hypotheses ("does this track_id refer to the same object", "is this contact on the pad or the edge") with different posterior weights. **If a policy's $q_\pi$ does not declare a hypothesis-layer quotient, $e_\pi$ will — under the default concat + MLP dynamics — fold them into a posterior mean.** The issue is not "does this model collapse"; the issue is **"does the interface provide an explicit hypothesis-preserving readout slot"**. Most existing policy architectures — the visual-latent and multimodal-token lines included — **do not provide such a slot**, which makes collapse an **interface default** rather than a training-dynamics accident.

An important refinement: **action-side multimodality in diffusion / flow-matching does not automatically mean state-side multi-hypothesis is preserved**. The action distribution can be multimodal (denoising yields multiple action trajectories), but if the hypothesis structure of $\hat S_t$ has already been folded by $e_\pi$, action-side multimodality is just sampling on an input that has already lost upstream distinctions. The two kinds of multimodality are not the same thing and cannot guarantee each other.

A precise rephrasing: **this piece does not criticize Diffusion Policy for collapsing; it criticizes the fact that no architecture explicitly commits to hypothesis preservation**.

### 3.2 Failure 2: No interface-level guarantee of semantic correctness

Frame fields inside the contract (`orientation_frame: "tool_flange"`, `reference_point: "contact_center"`, `convention: "right-handed"`) enter the policy and, if tokenized into a token sequence — **note, the issue here is not that "transformers cannot learn frame transforms"**. In principle a transformer can learn a hard constraint like $\tau_{p_2} = \tau_{p_1} + (p_1 - p_2) \times f$ through attention + MLP. **The real problem is that tokenization provides no interface-level guarantee that this transformation is interpreted correctly**. The model **can** learn it and **can also** fail to learn it, depending on whether the training distribution covers frame-contrast pairs, whether the architecture has a friendly inductive bias, and whether an auxiliary constraint explicitly imposes it.

**But more importantly: this failure mode is not the tokenizer's exclusive problem.** An engineered vector `[force_x, force_y, force_z, frame_id]` can absolutely be learned wrong by an MLP — once `frame_id` becomes an embedding, the MLP may treat it as a constant, or learn it correctly only for the specific ids covered in training. **The real abstraction is representation + semantic constraint, not tokenization per se.** Blaming Failure 2 on the tokenizer would mask the architecture-independence of this piece's thesis.

**A concrete shape**: in a real-robot deployment we change the wrench's `reference_point` from `sensor_flange` to `contact_center` — the preprocessing pipeline changes, the token sequence barely changes, yet $\tau$ shifts substantially per the transport theorem. A VLA trained on the two "look-alike" datasets separately **may converge separately to similar-but-wrong policies** — **"both wrong" is not because transformers cannot learn frame transforms, it is because the training data has too few frame-contrast pairs and no constraint forces it to learn**. This class of bug only surfaces **post-deployment**, on cross-dataset / cross-embodiment reuse.

**The key rephrasing**: tokenization does not erase semantics — it **demotes semantics from an interface guarantee to a training-time experience** — and **engineered vectors do the same demotion**. This is where this piece parts ways with the intuition "tokenizers just don't work".

### 3.3 Failure 3: Temporal alignment broken by concat

The contract makes per-channel `age` explicit (e.g. proprio 1 ms, F/T 5 ms, vision 100 ms, tactile 30 ms). If the policy only sees a concatenated vector, the $\Delta t$ distribution is lost. Symptom: the policy decides using vision's "world 100 ms ago" and tactile's "contact 30 ms ago" as if they were synchronous. 9/14 §4 already stressed **timestamp sync ≠ causal sync**, and even less so decision-time causal consistency. **Concat hides $\Delta t$; it does not hide the semantics of $\Delta t$**.

### 3.4 Failure 4: Safety-blindness to validity vs staleness

The contract explicitly separates availability (does the channel have data today), validity (is the data valid, e.g. is calibration current), and age (how old). A policy that reads only the numbers conflates "stale but valid" with "missing but valid", and treats "invalid after calibration drift" the same as "sensor disconnected". 9/14 §8.5 already broke the degradation chain apart — **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption** — if the policy does not ingest this chain at the input, both training-time augmentation and inference-time guardrail will latch onto the wrong place.

All four failures are **interface problems**; swapping backbones does not solve them; only making the $q_\pi$ layer explicit does. This matters because it **decouples** the decision "should we switch to a bigger VLA" from the decision "should we rewrite the policy-side interface".

## 4. Three families of contract-read primitives

Following the four failures in §3, the policy side needs **three families of primitives** — not bigger transformers, not more data. The third family (source structure) itself splits into three, see §4.3.

### 4.1 `mode_select` (the readout of the hypothesis layer)

Faced with a multi-hypothesis posterior inside the contract, the policy must pick one of several **explicit readouts**: MAP, posterior sample, expected-mixture, or **calibration-aware top-$k$**.

$$\text{read}\!\big(\{(\mu_i, \Sigma_i, w_i)\}_{i=1}^{K}\big) \;=\; \left\{\begin{aligned}
&\mu_{\arg\max_i w_i} && \text{(MAP, drop low-weight hypotheses)}\\
&\textstyle\sum_i w_i\, \mu_i && \text{(posterior mean, an explicitly declared collapse)}\\
&\mu_i + L_i \epsilon,\;\; i \sim w,\;\; \epsilon \sim \mathcal{N}(0, I) && \text{(posterior sample)}\\
&\big\{(\mu_i, \Sigma_i, \tilde w_i)\big\}_{i \in \mathrm{top}\text{-}k} \;\cup\; \{w_{\mathrm{other}}\} && \text{(calibration-aware top-}k\text{)}
\end{aligned}\right.$$

**"ECE-preserving top-$k$" from the previous version was rightly called out by reviewers** — top-$k$ truncation by itself does not preserve ECE, because $\sum_{i \in \text{top-}k} w_i < 1$ and residual mass is silently discarded. This version renames it to **calibration-aware top-$k$ readout** and handles the residual explicitly:

$$\tilde w_i \;=\; \frac{w_i}{\sum_{j \in \mathrm{top}\text{-}k} w_j}, \qquad w_{\mathrm{other}} \;=\; 1 - \sum_{i \in \mathrm{top}\text{-}k} w_i.$$

$\tilde w_i$ is the **within-top-$k$ renormalized** weight; $w_{\mathrm{other}}$ is the **residual mass**; both are passed downstream — only this way can a subsequent Bayesian update actually be mass-conserving. **"Calibration-aware" is a design criterion, not a guarantee** — this has to be said explicitly, otherwise it is just another pretty undefined name.

The point is **not "mean is forbidden"** — mean is a perfectly legitimate readout, provided the collapse is **explicitly declared**. The real failure mode is "the interface provides no readout slot for hypothesis structure, $e_\pi$ can only implicitly merge via concat + MLP, and mean becomes the default". This distinction is critical: **this piece is against undeclared default collapse, not against collapse per se**.

### 4.2 `age_gate`: parallel measurement / uncertainty / age / validity / health / latency / trust — seven slots

**A common interface-design bug** is to multiply staleness trust directly into the measurement: $x_c^\pi = \tau_c(a_c) \cdot \mu_c$. This **changes the physical value of the observation** — 10 N read at 100 ms age gets multiplied into "3 N", and the "3 N" at the policy input **looks** like "a 3 N force", not "a 10 N force whose trust has decayed". This directly violates the very distinction the contract wants to preserve: **$(F = 3\,\mathrm{N},\, a = 0)$ and $(F = 10\,\mathrm{N},\, a = 100\,\mathrm{ms})$ are two different semantic events**.

The right thing is to put them in **parallel** on the policy input and not multiply:

$$x_c^{\pi} \;=\; \big[\;\underbrace{\mu_c}_{\text{measurement}}\;,\;\underbrace{\Sigma_c}_{\text{covariance}}\;,\;\underbrace{a_c}_{\text{age}}\;,\;\underbrace{v_c}_{\text{validity}}\;,\;\underbrace{h_c}_{\text{sensor health}}\;,\;\underbrace{\ell_c}_{\text{latency / causal}}\;,\;\underbrace{q_c}_{\text{trust}}\;\big].$$

**The previous version wrote only $q_c = \tau_c(a_c)$ and a reviewer immediately noted the internal contradiction with §3.3's "timestamp sync ≠ causal sync"** — trust clearly depends not only on age but also on sensor health, latency, calibration status, and task context. This version writes:

$$q_c \;=\; \tau_c(a_c,\; v_c,\; h_c,\; \ell_c).$$

Age is one of four inputs; the concrete form of $\tau_c$ (learned / analytic / piecewise) is a policy-specific design decision left outside the interface.

**If a downstream gate really is required, it should gate on uncertainty, not on measurement.** But **the previous version wrote $\tilde\Sigma_c = \Sigma_c / \tau_c(a_c)$ as a general principle "stale ⇒ effective uncertainty is inflated" — also rightly called out by reviewers.** The correct treatment of a stale observation is not necessarily simple inflation; the more general form is **propagating the latent state forward**:

$$p(x_t \mid y_{t-\Delta t}) \;=\; \int p(x_t \mid x_{t-\Delta t})\, p(x_{t-\Delta t} \mid y_{t-\Delta t})\, dx_{t-\Delta t}.$$

When the robot is stationary, vision age 200 ms may leave the measurement still very accurate; when the robot is moving fast, the same 200 ms can imply huge predictive uncertainty. **Inflation and propagation are two different treatments, not synonyms.** This piece positions $\Sigma / \tau$ explicitly as **a simple conservative approximation**:

$$\Sigma_c^{\mathrm{eff}} \;=\; \mathrm{Propagate}\!\big(\Sigma_c,\; \Delta t,\; u_t,\; f_{\mathrm{dyn}}\big) \qquad\text{(general form)}$$

$$\Sigma_c^{\mathrm{eff}} \;=\; \Sigma_c \,/\, \tau_c(a_c, v_c, h_c, \ell_c) \qquad\text{(one conservative approximation)}$$

The right phrasing is: **staleness should modify the policy's uncertainty model; uncertainty inflation is one conservative implementation, while predictive state propagation is another.** This version demotes $\Sigma / \tau$ from "canonical" to "one implementation" and hangs `Propagate` next to it as the general form. Availability and $\mathbb{1}[v_c]$ still **hang on as parallel slots** — they must not be folded into $\mu$, nor into $\tau$.

This looks like a small change but it semantically repairs `age_gate` from "discounting a measurement" to "measurement + facts about the measurement" — a direct instantiation of the §0.2 $L_{\mathrm{dec}}$ definition: mixing a measurement with **facts about** the measurement into a single scalar is a direct source of $L_{\mathrm{dec}}$.

### 4.3 `provenance / dependency / negative evidence`: three things, not one bucket

9/14 §6.1 groups `contributing_mask`, `correlated_with` and `negative_evidence` under `provenance` — from the estimator side that makes sense (all three are "source structure of this field"), but from the **policy-side readout** they sit at different semantic levels:

| primitive | semantic role | How the policy reads it |
|---|---|---|
| **provenance** | `where` evidence came from | `contributing_mask` concatenated into policy input as conditioning (**provenance_harden**) |
| **dependency** | `how` evidence is statistically related | dependency-aware fusion / conditioning (**dependency_aware_fusion**) |
| **negative evidence** | `what expected evidence failed to appear` | observability-conditional likelihood ratio (**negative_evidence_read**) |

#### 4.3.1 `provenance_harden`

Concatenate `contributing_mask` directly into the policy input. Engineered-state heads and visual-latent heads both support this (a few more dimensions of vector). Nothing further needed.

#### 4.3.2 `dependency_aware_fusion` (was `dependency_gate`)

**The previous draft turned `correlated_with` into an attention bias to "suppress double-counting" — the intuition is not always valid.** Correlation ≠ redundancy ≠ double-counting. Counter-example: camera depth, tactile contact and F/T wrench may all be highly correlated, and **precisely because they agree**, confidence should go up, not down; suppressing that correlation is wrong. **The real question is whether the correlation is already accounted for by the estimator / uncertainty model.** If $\Sigma_{12}$ is already modeled in a Kalman-style fusion, then $P(x \mid y_1, y_2)$ is automatically correct; **double-counting exists only when conditional correlation structure is not modeled.**

Therefore this piece upgrades the primitive from `dependency_gate` to **`dependency_aware_fusion`**: its semantics is not "suppress correlated token pairs" but "let the policy or fusion layer know the statistical relation between evidence and react accordingly". **Attention bias is only one implementation** (and only for transformer-family backbones):

$$\mathrm{Attn}'_{ij} \;=\; \mathrm{Attn}_{ij} \;-\; \beta \cdot \mathbb{1}\!\big[\text{fields}_i \text{ correlated\_with } \text{fields}_j\big].$$

Other implementations include: covariance-aware fusion (turn `correlated_with` into $\Sigma_{12}$, feed Kalman / factor graph), hierarchical mixture (aggregate correlated fields under a shared latent so they cannot be sampled independently), and letting the network learn a bias matrix from `correlated_with` (the `learned` path for MLP heads).

It must also be honestly acknowledged that **`correlated_with` is a field-level semantic relation, whereas attention bias is a token-pair relation** — the two require a $R_{\text{field}} \to R_{\text{token}}$ mapping in between. A single field is often decomposed into value / uncertainty / age / provenance tokens, and which token pairs should be suppressed is **not a solved problem — it is a compilation problem from a provenance graph to an attention graph and deserves to be treated as an independent research direction**. This piece only stands `dependency_aware_fusion` up as a primitive; the attention-bias formula is "one implementation", not canonical.

#### 4.3.3 `negative_evidence_read` (likelihood-side evidence)

**The previous draft defined negative_evidence simply as $P(\mathcal{E}^- \mid H)$ — reviewers rightly noted this is not strict enough.** $E^-$ = "not observed" does not automatically imply $P(E^- \mid H)$ is low — **occlusion, limited FOV, sensor saturation, low SNR, calibration failure, timing mismatch** can all produce "not seen", in which case $E^-$ is uninformative about $H$ and reflects only sensor state.

The correct definition must condition on observability and sensor state, written as an **observability-conditional likelihood ratio**:

$$\Lambda^-(H) \;=\; \log \frac{P\!\left(\mathcal{E}^- \mid H,\, \mathcal{O}\right)}{P\!\left(\mathcal{E}^- \mid \neg H,\, \mathcal{O}\right)},$$

where $\mathcal{O}$ encodes observability / sensor health / field of regard / calibration status. This is the correct characterization of "what should have been seen but was not" — **only when $\mathcal{O}$ says something should have been seen does $E^-$ carry likelihood-side meaning about $H$**. This also plugs directly into the observability field defined in 9/14 §6: **negative evidence is likelihood-side evidence, not provenance.**

On the policy side: expose $\Lambda^-(H)$ (or the logit of $\exp(\Lambda^-)$) as a field inside $[\hat S]_{\mathcal{C}_\pi}$ so the policy or belief update can consume it. This primitive is the thinnest on engineering maturity but often the strongest in effect on hypothesis ranking.

The three sub-primitives together — **provenance covers `where evidence came from`, dependency covers `how evidence is statistically related`, negative evidence covers `what expected evidence failed to appear` — support the source-structure invariant of §0.2**.

### 4.4 Mapping primitives to invariants

Back to the §0.1 figure: **mode / temporal / source** correspond to **mode_select / age_gate / (provenance_harden + dependency_aware_fusion + negative_evidence_read)**. Miss any one invariant, at least two of the four §3 failures recur. Three families are also **not a complete interface design** — observability / identifiability, frame convention, and contact set each have their own more specialized readouts (9/14 §7, §8.6); this piece only handles **the three most easily destroyed silently at the $\Pi_\pi$ layer**.

## 5. Training-time and deployment-time knock-on effects

Once the §4 primitives are on the input side, four downstream things must be adjusted — training objective, augmentation, safety filter, evaluation. **Interfaces are not free** — but the changes are **local and bounded**.

### 5.1 Two classes of training constraint: representation-side probe + three tiers of intervention consistency

A natural mistake is: **to make the policy "use the contract", require it to output validity / hypothesis predictions** — this actually sneaks "use the contract" into "copy the contract", which is the wrong direction. **A policy is entirely allowed to only eat the contract and never spit it back out**; auxiliary prediction heads are not a necessary condition.

#### Class A · Representation-side probe (diagnostic)

Hang a few probe heads off the policy's intermediate representation $z^{\pi}$ and try to predict the contract's `age`, `validity`, `observability` and hypothesis posterior from $z^{\pi}$. **These probes do not enter the main loss; they are only used for measurement**: good probe scores mean the policy preserves this information internally, poor scores mean the contract has already been flattened inside $e_\pi$. $L_{\mathrm{probe}}$ may be added to the main loss with a small weight as regularization, but **its diagnostic value exceeds its training value**.

$$\mathcal{L}_{\mathrm{total}} \;=\; \mathcal{L}_{\mathrm{action}} \;+\; \underbrace{\alpha\, \mathcal{L}_{\mathrm{probe}}}_{\text{weak regularization, mainly diagnostic}} \;+\; \underbrace{\sum_{\mathcal{C}} \gamma_{\mathcal{C}}\, \mathcal{L}^{\mathcal{C}}_{\mathrm{consistency}}}_{\text{Type I / II / III, see below}}$$

#### Class B · Intervention-consistency constraint (the real core)

**The previous draft wrote this class generically as $D(\pi_\theta(\hat S), \pi_\theta(T_{\mathcal{C}}(\hat S)); \rho_{\mathcal{C}})$, and a reviewer immediately asked "where does $\rho_{\mathcal{C}}$ come from?"** — the four $T_{\mathcal{C}}$ do not all have the same "response pattern", and one of them should not even assume a response is required. This version splits intervention consistency into three tiers by response strength.

**Type I · Exact invariance / equivariance** (the cleanest tier). The transformation has a legal counterpart $T^{\mathcal{C}}_\pi$ on the action side, and we require:

$$\pi_\theta\!\big(T^{\mathcal{C}}(\hat S),\, o,\, \ell\big) \;=\; T^{\mathcal{C}}_\pi\!\big(\pi_\theta(\hat S,\, o,\, \ell)\big).$$

Typical: **frame transform** — move `reference_point` from A to B, $\tau$ transforms per the transport theorem, **the action side must undergo the corresponding coordinate transformation** ($\pi(T_g S) = T_g^A \pi(S)$). Also: **permutation of equivalent hypotheses** (hypotheses with the same posterior weight may be permuted; the action distribution must be invariant). This tier can be written as a hard loss, $\mathcal{L}_{\mathrm{consistency}}^{\mathrm{I}} = \|\pi_\theta(T\hat S) - T^\pi \pi_\theta(\hat S)\|^2$.

**Type II · Monotone response** (the middle tier). Specify a scalar or partial-order functional $M(\cdot)$ (variance of the action distribution, conservative action norm, stop probability, etc.), and require only:

$$M\!\big(\pi_\theta(T^{\mathcal{C}}\hat S)\big) \;\preceq\; M\!\big(\pi_\theta(\hat S)\big) \quad \text{(or the reverse } \succeq \text{)}.$$

Typical: **age increase** (measurement held constant, age raised from 5 ms to 200 ms) — **variance monotonically rises**, or actions become more conservative, or fallback probability monotonic. **Hypothesis reweighting** — hold hypothesis set constant, change only posterior weight; the action mass of the corresponding hypothesis must respond monotonically along the weight direction (top-1 need not flip, but the distribution must respond). This tier is written as a hinge / soft-ranking penalty and does not fix the full output.

**Type III · Unconstrained intervention** (the weakest and most important tier). **Do not assume the policy must change.** Typical example: **provenance removal** — dropping a contributing sensor whose information is fully redundant may leave the optimal action unchanged, and that is fine. The correct question here is: **when the removed evidence was decision-relevant, does performance degrade?** This is handed to §6's CAG panel rather than being imposed as a training-time response. Written as a loss: **no intervention consistency at all; only ablation at evaluation**. This tier's very existence is a correction to the previous draft, which treated all four $T_{\mathcal{C}}$ as "must respond" and thereby mistook a Type III intervention for a Type II.

**None of these three tiers require the policy to predict anything explicitly**; they require **the response function to conform to contract semantics, or to be permitted to remain unchanged**. This is the **training-side counterpart** of §0.2's "$q_\pi$ is a declared quotient". Compared to "adding a few auxiliary prediction heads", this three-tier scheme fits the thesis better and is more research-flavored: **what we propose is not for the policy to copy the contract, but for the policy to respond — or legitimately not respond — according to the tier of intervention**.

### 5.2 Augmentation: observed metadata vs latent degradation class

9/14 §8.5 emphasized **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption** — six degradations with different **causal origins** and different **downstream readouts**. If training-time augmentation uses only one (most often random masking), the policy learns them all as the same thing and cannot distinguish "this channel is broken today" from "this channel is high-latency" at deployment. The aug generator must **separate per chain**, sample independent distributions per class, and **each class must correspond to one of §4.2's seven slots** as the one being activated.

**The previous draft had a shortcut risk that reviewers caught**: "label each episode with a degradation class, carry it explicitly on the policy input" — if `degradation = stale_vision` is fed directly as a policy input, the policy learns $a = f(o, \text{label})$; but **in real deployment the label itself is not reliable**, and the shortcut is harmless in training and disastrous at deployment. This version explicitly separates two things:

**Observed metadata** $m_c = (\text{age}, \text{validity}, \text{availability}, h_c, \ell_c)$ — obtained directly by the estimator or sensor driver, may enter the contract and may enter the policy input. **These are exactly the seven slots of §4.2.**

**Latent degradation class** $d_c \in \{\text{missing}, \text{stale}, \text{bias}, \text{corrupt}, \ldots\}$ — augmentation knows which class was injected, but at deployment this class is **latent**: it can only be inferred by the contract estimator from the $m_c$ stream, or used as a training annotation for loss weighting and sampling, **never as a ground-truth input to the policy by default**.

Concretely: the aug pipeline samples $d_c$, generates $m_c$ conditioned on $d_c$ (e.g. $d_c = \text{stale}$ ⇒ raise $a_c$, keep $\mu_c$ at the delayed-time value, set $v_c = \text{true}$), then feeds only $m_c$ into the policy and uses $d_c$ solely for loss weighting and per-class evaluation slicing. **This is not curriculum, it is conditioning on observed metadata** — the difference is that conditioning lets the policy see reliable $m_c$, not reliable $d_c$. §5.1 Class B Type II monotone response pairs naturally with degradation-conditioned aug — **aug generates the $m_c$ shift induced by $T_{\mathcal{C}}$, loss measures whether the policy's response to $m_c$ matches $\rho_{\mathcal{C}}$**.

### 5.3 Safety filter and its interface to the contract: constraint-relevant validity

The constraint layer (CBF / shield / runtime verifier) **must read $a_{\mathrm{proposed}}$**, otherwise what is it filtering — that point is not up for negotiation. But the more precise claim of this piece is: **the safety filter should not treat the policy's confidence or the estimator's general-purpose validity bit as the sole evidence that a constraint holds; it should directly access constraint-relevant evidence**.

**Two kinds of validity must be distinguished here.** The estimator-side general `validity` bit says "this measurement is valid from a calibration / sensor-health perspective" — that is a field-level, general-purpose declaration. What safety actually cares about is **"is this constraint valid at this moment, under this predicate"** — that is constraint-level semantics. The two differ: `validity = true` does not imply that the constraint estimate is valid for **this particular safety predicate**; e.g. "joint-torque calibration OK" ≠ "current contact estimate supports the collision constraint".

This piece proposes that the safety filter read a **constraint-specific composite**:

$$v_j^{\mathrm{constraint}} \;=\; g\!\Big(\text{field validity},\; \text{observability},\; \text{hypothesis posterior},\; \text{age},\; \text{model coverage}_j\Big).$$

$g$ is a constraint-specific composition rule (e.g. a CBF side may require "distance estimate is stable under the current hypothesis, observability is sufficient, and dynamics model coverage reaches the current state region"). $v_j^{\mathrm{constraint}}$ is what the safety filter should actually read as **constraint-relevant evidence**. 9/14 §7 established that **track_id is a hypothesis** — the safety filter must not simply trust track_id matching, it must also check whether the hypothesis posterior is stable; whether two tracks merge or split directly determines the credibility of "how far is that obstacle".

This also wires in §4.3.3's negative evidence: **"what should have been seen but was not" makes $v_j^{\mathrm{constraint}}$ false** — for example the radar swept this angle and saw nothing, $\Lambda^-(H_{\text{clear}})$ turns positive, and the collision constraint's $v^{\mathrm{constraint}}$ must stay active rather than be deactivated by the policy's optimistic belief.

**If these three (constraint-specific validity, observability, negative evidence) do not enter the safety filter, the filter will infer constraint validity from the policy's belief** — which is especially dangerous in low-observability regions. The policy's belief is optimistic precisely because it cannot see the contract's observability / validity / negative evidence, and **if the filter also cannot see them, the two go blind together**.

### 5.4 Echo with 9/10 Part 3 evaluation

Among the three sim-utility dimensions (prediction / ranking / decision), policy-side evaluation is mainly about **decision** — but with an added **contract-preservation dimension**: how much the policy degrades when the contract is torn apart is a **lower bound** on how much it depends on the contract. This dimension maps to the CAG metric of §6, and needs an oracle baseline to bound its interpretation.

## 6. Evaluation: a four-level panel

Aligned with the §4 primitives and the three tiers of §5.1, this piece proposes a **four-level evaluation panel** — not a pile of parallel metrics, but four levels each answering a distinct question. **CAG is only the Decision-level aggregate and does not measure semantic understanding; contract compliance requires the four levels together.**

| Level | Test | Question answered |
|---|---|---|
| **Semantic** | invariance / equivariance test (§5.1 Type I) | Does the policy respond according to the transformation rule? |
| **Representation** | probes (§5.1 A) + hypothesis coverage & separation | Is contract information still present in $z_\pi$? Are hypotheses preserved and distinguished? |
| **Decision** | CAG + task-conditional utility (§5.1 Type III) | Does contract structure actually improve decisions, and by how much? |
| **Safety** | constraint intervention (§5.3 $v_j^{\mathrm{constraint}}$) | When contract degrades, does the correct guardrail engage? |

Below we unfold each level. The temporal and source slices are specializations of the Decision level (**SDS** and **$\Delta J_{\mathrm{prov}}$**) — not new metrics but focused views of the CAG panel.

### 6.1 Semantic level: Invariance / Equivariance Test

Corresponding to §5.1 Type I. Given a family of contract transformations with known $T^{\mathcal{C}}_\pi$ (frame / coordinate / hypothesis permutation), measure:

$$\mathrm{Equiv}(\mathcal{C}) \;=\; \mathbb{E}_{\hat S}\!\left[d\!\left(\pi_\theta(T^{\mathcal{C}}\hat S),\; T^{\mathcal{C}}_\pi\,\pi_\theta(\hat S)\right)\right].$$

$\mathrm{Equiv} \to 0$ is a hard requirement; large $\mathrm{Equiv}$ means either $e_\pi$ learned it wrong, or $q_\pi$ dropped the quotient entirely. This level is the cleanest, because the rule is mathematically defined — no oracle and no $J$ definition needed. §3.2 Failure 2's severity can be quantified directly by $\mathrm{Equiv}(\text{frame})$.

### 6.2 Representation level: Probes + HPC + HSS

**Probe scores** (§5.1 Class A): reverse-predict contract fields from $z_\pi$ and report accuracy / calibration. Diagnostic, not core.

**The previous version's HPS used $\max_k$, and reviewers rightly caught the loophole** — with $K = 8$, if the policy only serves $H_1$ correctly and fails $H_2$ through $H_8$, $\max_k$ still scores full, but hypothesis structure is not preserved at all. This version splits HPS into two metrics pointing in different directions:

**Hypothesis Coverage (HPC)** — how many hypotheses are supported:

$$\mathrm{HPC} \;=\; \frac{1}{K} \sum_{k=1}^{K} \Pr\!\big[\pi_\theta(x_t^\pi) \in \mathcal{A}^{*}_k \,\big|\, H_k \text{ is true}\big].$$

**Hypothesis Separation Score (HSS)** — whether the policy's outputs are distinguishable across hypotheses:

$$\mathrm{HSS} \;=\; \frac{1}{K(K-1)} \sum_{i \neq j} D\!\big(\pi_\theta(\cdot \mid H_i),\; \pi_\theta(\cdot \mid H_j)\big).$$

HPC and HSS together correspond to the semantic-preservation thesis — **coverage** ensures "every hypothesis is handled"; **separation** ensures "the policy preserves hypothesis distinctions". Like the oracle in §6.3, $\mathcal{A}^{*}_k$ is **not required to be online-available** — it is constructed in the benchmark from **privileged simulator state, oracle planners, or offline expert rollouts**, hence HPC / HSS are **training / evaluation-time metrics, not deployment-time observables**. This caveat carried over from the previous version and remains.

### 6.3 Temporal slice: Staleness Response Compliance (SDS)

**The previous SDS definition $D(\pi(\cdot|\mathrm{do}(a=a_1), o), \pi(\cdot|\mathrm{do}(a=a_2), o))$ with "expected variance monotonically rising" needs two tightenings.** First, SDS is not a scalar; the reviewer is right that "variance monotonically rising" is not a universal law — a real policy may be **piecewise**: age < 50 ms uses vision, age > 100 ms switches to proprio-only fallback, age > 200 ms stops; the response curve is piecewise discontinuous and monotonicity cannot be assumed. Second, like §6.5's CAG, SDS needs an oracle baseline to bound what counts as "correctly responding".

This piece redefines SDS as **response fidelity against oracle**:

$$R_\pi(a) \;=\; \pi_\theta\!\big(\cdot \,\big|\, \mathrm{do}(a_c = a),\, o\big),$$

$$\mathrm{SDS} \;=\; D\!\big(R_\pi(a),\; R^{*}(a)\big),$$

where $R^*(a)$ is **the oracle policy's response curve under the same intervention**, and $D$ is a curve-level divergence (e.g. KL integrated over $a$, or sliced Wasserstein). The response property $R$ is task-defined and may take **variance / action norm / fallback probability / safety margin / stop probability** — not hard-coded as "variance monotonically rises" but "policy and oracle agree on the shape of the response in this attribute". Flat is not automatically bad; compare against the oracle's expected response curve.

The derivative form is retained as one local characterization of response shape:

$$\left.\frac{\partial\, \mathbb{E}\!\big[\pi_\theta(\cdot \mid \mathrm{do}(a_c = a),\, o)\big]}{\partial a_c}\right|_{a}\quad\text{compared to the same-order derivative of } R^*.$$

### 6.4 Source slice: $\Delta J_{\mathrm{prov}}$ and $\Delta J_{\mathrm{neg}}$

**The previous $\Delta J_{\mathrm{prov}}$ stays, but its interpretation must be updated alongside §4.3.** Dependency-aware fusion is not "suppress double-counting", so $\Delta J_{\mathrm{prov}}$ does not measure "did the policy suppress correlated pairs" — it measures **"did the policy consume `correlated_with` as legitimate evidence structure"**. $J$ is always higher-is-better:

$$\Delta J_{\mathrm{prov}} \;=\; J\!\big(\pi_\theta \mid \text{provenance + correlated\_with}\big) \;-\; J\!\big(\pi_\theta \mid \text{both} = \varnothing\big).$$

$\Delta J_{\mathrm{prov}} > 0$ = the policy really uses source structure; $\Delta J_{\mathrm{prov}} \approx 0$ = it treats source as decoration; $\Delta J_{\mathrm{prov}} < 0$ = conditioning actively hurts, usually indicating that conditioning conflicts with the backbone's inductive bias and deserves separate debugging.

**Add a companion $\Delta J_{\mathrm{neg}}$** — remove $\Lambda^-(H)$ from §4.3.3 and measure the drop in hypothesis-ranking accuracy:

$$\Delta J_{\mathrm{neg}} \;=\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda^-\big) \;-\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda^- = \varnothing\big).$$

This directly probes the negative_evidence primitive and answers the safety-side interface in §5.3.

### 6.5 Decision level: Contract Ablation Gap (CAG) + Oracle baseline

**Given a specific collapse operator $\mathrm{collapse}_X$ on the contract** (which silently folds layer $X$ of contract structure):

$$\mathrm{CAG}_X \;=\; J_{\mathrm{full}} \;-\; J_{\mathrm{collapse}_X}, \qquad J_{\mathrm{full}} \equiv J(\pi_\theta \mid \hat S),\;\; J_{\mathrm{collapse}_X} \equiv J(\pi_\theta \mid \mathrm{collapse}_X(\hat S)),$$

with four collapses each corresponding to one invariant:

- $\mathrm{collapse}_{\mathrm{hyp}}$: fold the hypothesis set into a single Gaussian or a point estimate.
- $\mathrm{collapse}_{\mathrm{age}}$: flatten every channel's `age / health / latency`.
- $\mathrm{collapse}_{\mathrm{prov}}$: drop `contributing_mask` and `correlated_with`, let policy attend indiscriminately.
- $\mathrm{collapse}_{\mathrm{neg}}$: drop $\Lambda^-(H)$.

**But reviewers are right — $\mathrm{CAG}_X$ alone cannot distinguish "policy not using contract" from "contract not informative enough for this task", nor rule out the policy treating some field as a shortcut.** This version adds an **oracle privileged-state baseline**:

$$J_{\mathrm{oracle}} \;=\; J(\pi^{*}_{\mathrm{oracle}} \mid s^{\mathrm{priv}}), \qquad \mathrm{Gap}_{\mathrm{oracle}} \;=\; J_{\mathrm{oracle}} - J_{\mathrm{full}}.$$

The interpretation matrix becomes:

| CAG vs oracle | Reading |
|---|---|
| CAG high, oracle slightly above full | Policy really reads contract; contract is nearly sufficient |
| CAG high, oracle clearly above full | Policy reads, but the contract itself is missing decision-relevant information (go back and check whether the 9/14 schema is complete) |
| CAG ≈ 0, oracle ≈ full | Contract structure is not useful for this task; CAG ≈ 0 is expected |
| CAG ≈ 0, oracle clearly above full | **The real red flag** — contract carries information, policy is not using it |

**The one sentence this version must write clearly**: **CAG measures task-conditional utility of contract structure, not semantic understanding by itself.** High CAG may reflect a shortcut (e.g. age strongly correlates with task difficulty in the training distribution; collapsing age hurts performance without any semantic understanding). Real compliance requires **CAG + §5.1 Type I / II controlled response + §6.1 invariance / equivariance** all three together. This version demotes CAG from "the aggregate" back to "Decision-level aggregate", so that the four-level panel is complete.

### 6.6 Safety level: Constraint intervention

Actively suppress §5.3's $v_j^{\mathrm{constraint}}$ in deployment / semi-simulation (e.g. inject calibration drift, cut observability, raise $\Lambda^-(H)$), and check whether the safety filter **enters the correct guardrail at the correct moment**, and whether guardrail activation is **attributable to which input of $v_j^{\mathrm{constraint}}$**. This level is the direct instantiation of §5.3's interface, and the real test of whether the piece's interface proposal is **actually deployable**.

The four levels together form an **intervention-based evaluation panel**: **Semantic measures "does it respond by rule", Representation measures "is the information still there", Decision measures "did it use it / is there a shortcut", Safety measures "does the guardrail engage when things degrade"**. None of them replaces an end-to-end success rate — they measure the policy-side **read-completeness** of the contract, not the policy's **expressive power**. This is exactly aligned with §0.3 Claim 3: **contract compliance must be tested by controlled intervention**.

## 7. A minimal executable interface sketch

Combine the §4 primitives and §6 four levels into a Python class skeleton. **Not a concrete policy; a readable interface contract**.

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

Three caveats written in stone:

- **(i)** This is not the only way to read; **`cfg` is exactly the declared quotient $q_\pi$ from §0.2** — the same contract should be projected with different `cfg` by a VLA and a Diffusion Policy, and the optimal `uncertainty_inflation` differs between engineered-state heads and visual-latent heads. The interface specification must write $q_\pi$ down; it may not hide inside encoder weights.
- **(ii)** This interface **only addresses the input side**. The three tiers of §5.1 intervention constraints, observed-metadata-only aug in §5.2, and the $v_j^{\mathrm{constraint}}$ direct wiring in §5.3 — if any one is left unchanged, no matter how beautifully $q_\pi$ is declared, training dynamics will route around it (the loss will find the cheapest "flatten the contract" path on its own).
- **(iii)** `dependency="attention_bias"` only has an implementation path on transformer-family backbones, and **field-graph → token-graph compilation** (`compile_field_graph_to_token_graph`) is a function name in this piece, not a canonical implementation; MLP heads use `learned`, Kalman / factor-graph fusion uses `covariance_fusion` — **attention_bias is one of several implementations of dependency_aware_fusion, not the primitive itself**.

## 8. Three closing claims (aligned with §0.3)

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. The policy projection $\Pi_\pi = e_\pi \circ q_\pi$ is a semantic interface: it determines which contract distinctions remain available downstream. $q_\pi$ is the explicit declaration of what may be dropped, $e_\pi$ is the actual neural encoding — without separating them, there is no auditable interface.

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state heads, latent visuomotor policies, autoregressive VLAs, diffusion policies and flow policies all use different conditioning and action generators — but they must all answer **the same interface question**: which contract distinctions are intentionally preserved, which are compressed, and which are discarded? The target of criticism is the interface contract, not the model architecture; π0 is VLA + flow matching, Diffusion Policy is visual-latent + diffusion, ACT is visual-latent + generative sequence decoder — slicing along two orthogonal dimensions is closer to reality than three families, and is less easily misled by the intuition "some family is inherently better".

> **Claim 3 · Contract compliance should be tested by controlled intervention, not inferred from end-to-end success.** End-to-end success measures whether the policy works, not whether it interpreted contract semantics correctly. Contract compliance requires the four-level panel of §6 together — **Semantic (invariance / equivariance) + Representation (HPC / HSS + probes) + Decision (CAG with oracle baseline) + Safety (constraint intervention)** — and requires distinguishing the three tiers of §5.1 (Type I exact equivariance / Type II monotone / Type III unconstrained). Treating Type III as Type II and forcing a response was a specific error of the previous draft; this version hands it to ablation instead.

A closing line: **try to stop using the word "fusion"** — on the time axis it asks "when to merge", on the semantic axis it asks "merge into what"; together, 9/14 and this piece split the second question into **what upstream delivers + what downstream declares may be dropped + what encoding actually preserves**. The first question (when) is largely answered by §1's two-dimensional grid — **timing is a consequence of the interface, not a decision variable of the interface**.

**One step further, as a research proposition**: what this piece really stands up can be compressed into a single pipeline:

$$\boxed{\text{Contract} \;\longrightarrow\; \text{Declared Quotient } q_\pi \;\longrightarrow\; \text{Policy Representation } e_\pi \;\longrightarrow\; \text{Intervention Tests}}$$

That is: **a policy is not merely a function approximator; it is a contract consumer**. This framing allows the next piece to stand up a **Contract-Preserving Policy Benchmark** directly — construct $\mathcal{B}_{\mathcal{C}} = \{T_{\mathrm{frame}}, T_{\mathrm{hyp}}, T_{\mathrm{age}}, T_{\mathrm{validity}}, T_{\mathrm{prov}}, T_{\mathrm{neg}}\}$ and run SAC, PPO, Diffusion Policy, ACT, OpenVLA and π0 through the same intervention suite — at which point this series upgrades from "I think policy should read contract" to **"given the same Structured State Contract, how do we systematically test whether different policies are contract-compliant"**. Stand that line up, and the series has been worth it.

## Sources

All arXiv IDs below have been verified online; journal-only citations do not carry an arXiv link. Grouped by the sections they support.

### A · VLA family (supports §1 grid, §3 Failures 1–2, §5.1 Class B)

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) (**paper fact**: robot actions explicitly expressed as text tokens, jointly fine-tuned with a VLM. §3.2 Failure 2's tokenizer-side shape; "contract flattening" is this piece's analysis, not an admitted limitation of the paper)
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) (**paper fact**: 7B VLA trained on large-scale robot demonstrations, emphasizing fine-tuning and generalization; **this piece's analysis**: configurations include multi-camera / depth / proprioceptive state encoding, but "supports input" ≠ "how far along the contract chain the policy reads")
- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) (**paper fact**: pretrained VLM + proprio token + noisy action chunk + flow matching; **this piece's analysis**: Under the Structured State Contract defined here, π0's conditioning interface does not expose an explicit slot for hypothesis / provenance / age / negative evidence — "Continuous actions do not imply structured state semantics" is this piece's analysis, not the paper's self-declared limitation)
- Octo Model Team, *Octo: An Open-Source Generalist Robot Policy*, RSS 2024 · [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) (transformer-based readout · a reference for the attention_bias path within §4.3.2's dependency_aware_fusion)

### B · Diffusion / Generative-Sequence / Flow-Matching Policy (supports §1 grid, §5.1)

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 · [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) (**paper fact**: RGB stack + proprio concat + conditional denoising diffusion, emphasizing action-distribution multimodality; **this piece's analysis**: action-side multimodality ≠ state-side hypothesis preservation — an inference under this schema, not an admitted limitation)
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT / ALOHA), RSS 2023 · [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) (**paper fact**: CVAE + transformer encoder-decoder, core is **action chunking over sequences** — this piece places ACT under "generative sequence decoder", alongside diffusion and flow matching but not inside the diffusion family)
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 · [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) (**paper fact**: establishes the flow-matching objective as vector-field regression for generative modeling / CNF — **flow matching itself is not a robot action chunking paper**; "continuous robot action chunks" is a downstream application introduced by π0-class work; the citation chain is split here as "Lipman establishes objective / π0 applies it to action chunks")

### C · Uncertainty, calibration, and belief-space references (support §4.1, §5.1, §6.5 CAG)

- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) (modern networks over-confident; temperature scaling origin · motivation for §4.1 calibration-aware top-$k$ as a design criterion)
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels* (PlaNet / RSSM), ICML 2019 · [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) (deterministic + stochastic latent · a reference for §4.2 predictive uncertainty propagation)
- Hafner et al., *Mastering Diverse Control Tasks through World Models* (DreamerV3), Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) (discrete + continuous mixed latent, KL balancing · an adjacent path for §4.1 posterior readout)

### D · SAC / PPO and engineered-state baseline (supports the first row of §1 grid)

- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, ICML 2018 · [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) (Gaussian NLL / max-entropy policy loss · the engineered-state-head shape of §5.1 $\mathcal{L}_{\mathrm{action}}$)

### E · Continuations of the series (how this piece plugs into 9/13, 9/14, Sim-to-Real P1/P3)

- This blog, *Splicing is not seeing: what robot multimodal fusion is missing is an interface, not a model* · `/en/articles/2026-09-14-multimodal-fusion-interface/` (Structured State Contract, Interface Property Benchmark, degradation chain · the §0.2 $q_\pi$ and §3–§6 build directly on it)
- This blog, *Robots can see but cannot feel: why manipulation lacks a hand* · `/en/articles/2026-09-13-tactile-force-sensing/` (four control paradigms of force/tactile, Closed-loop value · the historical source of §1 grid's action-head spectrum)
- This blog, *Sim-to-Real Methodology (III)* · `/en/articles/2026-09-12-sim-to-real-evaluation-protocol/` (three-level evidence, three-dimensional decision utility, allocation protocol · §5.4, §6 inherit its evaluation spine)
- This blog, *Sim-to-Real Methodology (I)* · `/en/articles/2026-09-10-sim-to-real-methodology/` (allocation state $s_t = (b_t, \pi_t, q_t, h_t)$, $\Delta_{\mathrm{queue}}$ vs $\Delta_{\mathrm{processing}}$ · §2, §4.2 definitions plug in directly)

---

> **Further reading**
>
> - [Splicing is not seeing: what robot multimodal fusion is missing is an interface, not a model](/en/articles/2026-09-14-multimodal-fusion-interface/) — the prequel; stands up Structured State Contract
> - [Robots can see but cannot feel: why manipulation lacks a hand](/en/articles/2026-09-13-tactile-force-sensing/) — the historical lineage of §1's action-head spectrum, four force-control paradigms
> - [Embodied AI Sim-to-Real Methodology (III)](/en/articles/2026-09-12-sim-to-real-evaluation-protocol/) — §5.4, §6 reuse its three-level evidence and utility dimensions
> - [VLA and World Models: Two Roads Diverging and Converging](/en/articles/2026-09-07-vla-world-models/) — macro backdrop for §1's conditioning dimension, the other face of "world models do not naturally belong to sim-to-real"
> - [VLA π-family at a Glance](/en/articles/2026-09-05-vla-pi-family/) — a concrete cut through π0, π0.5 and the flow-matching action head, field evidence for §1's "continuous action ≠ structured state"
> - [What is a VLA model? A single explainer](/en/articles/2026-09-03-vla-deep-dive/) — this piece assumes you have already read it
