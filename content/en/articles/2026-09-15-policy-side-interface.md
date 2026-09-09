---
title: 'After the Contract Stands: What Do VLA, Diffusion Policy and π0 Actually Consume?'
slug: "2026-09-15-policy-side-interface"
date: 2026-09-15
draft: false
categories: ["Embodied AI", "Policy Learning"]
tags: ["Embodied AI", "Policy Learning", "VLA", "Diffusion Policy", "π0", "RT-2", "OpenVLA", "Action Tokenization", "Structured State Contract", "Consumer Contract", "Declared Quotient", "Query Family", "Semantic Preservation", "Decision-Relevant Preservation", "Decision Sufficiency", "Conditional Mutual Information", "Contract Information Loss", "Declared Loss", "Representation Loss", "Decision-Use Loss", "Contract-Read Primitives", "Intervention Consistency", "Equivariance", "Order-Constrained Response", "Likelihood-Side Evidence", "Conditional Log-Likelihood Ratio", "Dependency-Aware Fusion", "Mass-Preserving Top-k", "Constraint-Relevant Validity", "Counterfactual Hypothesis Intervention", "Action-Relevant Separation", "Contract Ablation Gap", "Compliance Evidence", "Contract Consumer", "Evaluation Metrics"]
description: 'The multimodal-fusion piece stood up the upstream deliverable as a Structured State Contract. This piece asks the dual question: if the estimator really delivers per contract, can the policy side actually consume it. The core object is an explicitly layered pipeline $\mathcal C\to\mathcal C_\pi\xrightarrow{\text{induce}}\sim_\pi\xrightarrow{\text{quotient}}q_\pi\circ e_\pi\circ\pi_\theta$: first declare which slice of the contract the policy consumes as a contract consumer ($\mathcal C_\pi$), which induces a query family $Q_{\mathcal C_\pi}$ and an equivalence $\sim_\pi$, on top of which the declared quotient $q_\pi$ folds the state, $e_\pi$ encodes, and $\pi_\theta$ decides. The piece separates **semantic preservation** from **decision sufficiency** and adds a third tier — **decision-relevant semantic preservation** (defined via $\mathcal A^{*},\mathcal G^{*}$, protecting only those distinctions that change downstream action or safety consequences). Contract information loss is written as **conditional MI** $L_{\mathcal C}^{\pi}=I(Y_\pi;\hat S\mid Z_\pi,O,L)$, which directly answers the objection that "raw image already contains an age / provenance proxy". On the pipeline side three losses map to three failure sites: $q_\pi$ (declared loss), $e_\pi$ (representation loss), $\pi_\theta$ (decision-use loss) — $e_\pi$ injectivity does not imply $\pi_\theta$ uses the distinctions it preserved. Interface primitives: `mode_select` uses a **mass-preserving top-$k$** (no longer called calibration-aware; calibration is evaluated separately via ECE / NLL / Brier / coverage); `age_gate` restructures into observation payload $(\mu,\Sigma)$ + a five-tuple of temporal / operational metadata $(a,\ell,h,v,\alpha)$ + one derived trust $q=\tau(a,\ell,h,v,\alpha)$ with an availability × validity 2×2 table; the source structure splits into provenance / dependency / negative evidence — `dependency_aware_fusion` now writes at the logit level $L^{\prime}_{ij}=L_{ij}+b(R_{ij})$ (with $b$ positive, negative or learned), and negative evidence generalizes to a conditional LLR $\Lambda(E;H,\mathcal O)$ with $E=E^-$ as its special case. Training-side interventions keep three tiers but **Type II is upgraded from monotone to order-constrained response** ($T_1\preceq_{\mathcal C}T_2\Rightarrow r(\pi(T_1S))\le r(\pi(T_2S))$; monotonicity is one special case). Evaluation becomes **four types of compliance evidence** $E_{\mathrm{semantic}}/E_{\mathrm{representation}}/E_{\mathrm{decision}}/E_{\mathrm{safety}}$; HPC is constructed via counterfactual $T_k^{\mathrm{hyp}}$; HSS is averaged only over action-relevant pairs $\mathcal R=\{(i,j):\mathcal A^{*}_i\not\equiv\mathcal A^{*}_j\}$; CAG is explicitly boxed as $\boxed{\mathrm{CAG}=\text{task-conditional utility sensitivity}\neq\text{contract understanding}}$ with a five-row observation-support table. On the safety side, one line is bolded: **invalid evidence cannot justify relaxing the constraint.** The piece closes on the upgraded boxed thesis: **A policy is a contract consumer, not merely a function approximator** — four contract-consumer questions (what may I discard / what did I retain / how should decisions respond / what happens when evidence becomes invalid) plus a full pipeline $\mathcal C\to\mathcal C_\pi\to q_\pi\to e_\pi\to\pi_\theta\to\mathcal B_{\mathcal C}$. VLA / Diffusion / Flow / ACT / SAC / PPO demote from theoretical classifications to implementation coordinates.'
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

## 0. Framing: $\mathcal C\to\mathcal C_\pi\to q_\pi\to e_\pi\to\pi_\theta$ is an auditable semantic pipeline

Set up the whole analysis framework up front; every later section returns to this figure. This section stands up the piece's real **formal objects** — the **three-layer quotient-declaration chain** $\mathcal C\to\mathcal C_\pi\to{\sim_\pi}\to q_\pi$, the **separation** of semantic preservation from decision sufficiency with a third tier of decision-relevant preservation added, the **conditional-MI form** of contract information loss $L_{\mathcal C}^{\pi}$, and the **three-loss decomposition** $L_{\mathrm{declared}} / L_{\mathrm{rep}} / L_{\mathrm{decision}}$.

### 0.1 Argumentation chain

```
Structured State Contract  Ŝ_t                                (defined in 9/14)
        │
        ▼
upstream contract  C                                          (full semantic schema, delivered by the estimator)
        │  declare
        ▼
consumer contract  C_π  ⊆ C                                    (policy declares "which slice of C I consume")
        │  induce
        ▼
equivalence  ~_π  on  Ŝ                                        (induced by a declared query family Q_{C_π})
        │  quotient
        ▼
declared quotient  q_π :  Ŝ  ↦  [Ŝ]_{~_π}                     (explicit statement of what may be dropped; not a network)
        │
        ▼
neural encoding  e_π :  [Ŝ]_{~_π}  ↦  z_π                     (all existing policies do this; almost none declare q_π)
        │
        ▼
policy head  π_θ :  (z_π, o, ℓ)  ↦  a                          (the third layer that can silently be routed around)
        │
        ▼
three losses, each audited separately
   ├─ L_declared     what q_π  intentionally discards
   ├─ L_rep          what e_π  actually retains
   └─ L_decision     whether π_θ uses the preserved distinctions
        │
        ▼
three contract-relevant invariants
   ├─ mode          multi-hypothesis structure            ──▶  primitive: mode_select
   ├─ temporal      heterogeneous staleness / health      ──▶  primitive: age_gate
   └─ source        provenance / dependency / negative ev. ──▶  primitives: provenance / dependency / negative evidence
        │
        ▼
training (representation-side conditional probe + Type I equivariance / Type II order-constrained / Type III unconstrained)
deployment (safety filter reads constraint-relevant evidence; invalid evidence cannot justify relaxing constraint)
evaluation (four compliance evidence types: semantic / representation / decision / safety; CAG is only the decision-layer aggregate and needs an oracle baseline)
```

### 0.2 Three-layer quotient-declaration chain: $\mathcal C\to\mathcal C_\pi\to{\sim_\pi}\to q_\pi$

The previous draft split $\Pi_\pi$ into two layers, $e_\pi \circ q_\pi$. This version accepts the reviewer's push-back — **$q_\pi$ was still carrying too much weight and the question "who defines the quotient" was not answered**. This version places two more layers in front of $q_\pi$, turning the interface into an **auditable three-layer declaration chain**:

$$\boxed{\;\mathcal C\;\xrightarrow{\text{declare}}\;\mathcal C_\pi\;\xrightarrow{\text{induce}}\;{\sim_\pi}\;\xrightarrow{\text{quotient}}\;q_\pi\;}$$

where:

- **$\mathcal C$ · Upstream contract** — the **complete** semantic schema delivered by the estimator (defined in 9/14 §6–§8).
- **$\mathcal C_\pi$ · Consumer contract** — **the policy, as a consumer, explicitly declares which slice of $\mathcal C$ it takes on**. $\mathcal C_\pi$ may be a subset of $\mathcal C$ ("this policy does not consume provenance"), or a coarse-graining of $\mathcal C$ ("hypothesis structure is folded to a point estimate but age is preserved as its own field"). **$\mathcal C_\pi$ is a paragraph in the interface spec, not an implicit preference inside encoder weights**.
- **${\sim_\pi}$ · Induced equivalence** — induced by a **declared set of contract queries / decision-relevant predicates** $Q_{\mathcal C_\pi}$ on $\mathcal C_\pi$:

$$\hat S \sim_\pi \hat S' \quad\Longleftrightarrow\quad Q_{\mathcal C_\pi}(\hat S) \;=\; Q_{\mathcal C_\pi}(\hat S').$$

$Q_{\mathcal C_\pi}$ is the real **interface object** of this piece — it turns "what may be dropped" from a vague semantic promise into an enumerable list of queries that can be reviewed one by one. Typical examples: "what is the age of channel $c$ in $\hat S$", "which are the top-3 posterior-weighted hypotheses for this track", "does the contact set include the pad face".

- **$q_\pi$ · Quotient map** — the actual $q_\pi : \hat S \mapsto [\hat S]_{\sim_\pi}$, folding $\hat S$ into the quotient space defined by ${\sim_\pi}$.

With this chain in place, the piece's core claim can finally be phrased in a way a reviewer cannot follow up with "who defines the quotient":

> **Policy does not need to preserve the entire upstream contract $\mathcal C$. It must explicitly declare a consumer contract $\mathcal C_\pi$ together with a query family $Q_{\mathcal C_\pi}$ — and the resulting quotient $q_\pi$ is exactly the semantic loss the policy is allowed to take.**

Compared to the previous version's "either preserve the whole quotient, or explicitly declare a sufficient quotient", this version delivers a complete answer to **where the quotient comes from**: it comes from a **written** $\mathcal C_\pi$ plus an **enumerable** $Q_{\mathcal C_\pi}$.

**$\Pi_\pi$ is still $e_\pi \circ q_\pi$, but $q_\pi$ is no longer a primitive — it is uniquely determined by ${\sim_\pi}$, which is in turn induced by $\mathcal C_\pi$.** Engineering consequence: **the interface spec must be able to display a list of $Q_{\mathcal C_\pi}$ queries**, otherwise $q_\pi$ degenerates back into encoder behavior — exactly what this piece is attacking.

### 0.2.1 Three tiers of preservation: full semantic / decision-relevant semantic / decision sufficiency

This subsection is the theoretical anchor; the **three** commonly conflated properties must be pulled apart — the previous version separated only two, and the reviewer was right that "full semantic preservation as an interface criterion is over-preserving".

Fix a set of **contract-relevant decision variables** $Y_{\mathcal{C}}$ served by the policy (quantities downstream controller / planner / safety filter / diagnostics will read), together with the contract equivalence $\sim_\pi$ on $\hat S$ already defined in §0.2 (induced by $Q_{\mathcal C_\pi}$). Three properties are distinguished.

**Property A · Full Semantic Preservation** — the projection does not irreversibly collapse semantically distinct contracts:

$$\hat S \not\sim_\pi \hat S' \quad \Longrightarrow \quad \Pi_\pi(\hat S) \not\equiv \Pi_\pi(\hat S').$$

This requires $q_\pi$ to be injective on the quotient defined by $\mathcal C_\pi$. It is **a strong property, but possibly over-preserving**.

**A concrete counter-example (raised by reviewer, accepted by this version)**: two hypotheses

$$H_1 = \text{"object at } x = 1.00\text{"},\qquad H_2 = \text{"object at } x = 1.01\text{"}.$$

From the estimator's semantic contract, $H_1 \not\sim H_2$. But if the downstream controller's action-space resolution is only 5 cm,

$$\mathcal A^{*}(H_1) \;=\; \mathcal A^{*}(H_2).$$

Collapsing $H_1$ and $H_2$ at the policy interface is then perfectly reasonable — **demanding full semantic preservation would place an unnecessary burden on the interface**.

**Property A′ · Decision-Relevant Semantic Preservation (this piece's actual interface criterion)** — require injectivity only on distinctions that would change downstream **action or safety consequences**. Define a decision-level equivalence via $\mathcal A^{*}$ (optimal / admissible action set) and $\mathcal G^{*}$ (safety guardrail consequences):

$$\hat S \sim_{\pi,\mathcal D} \hat S' \quad\Longleftrightarrow\quad
\begin{cases}
\mathcal A^{*}(\hat S) \;=\; \mathcal A^{*}(\hat S')\\[1mm]
\mathcal G^{*}(\hat S) \;=\; \mathcal G^{*}(\hat S')
\end{cases}$$

Decision-relevant preservation is then:

$$\hat S \not\sim_{\pi,\mathcal D} \hat S' \quad\Longrightarrow\quad \Pi_\pi(\hat S) \not\equiv \Pi_\pi(\hat S').$$

Because $\sim_{\pi,\mathcal D}$ is coarser than $\sim_\pi$, **Property A implies Property A′, and not vice versa**. This is the interface requirement this piece actually wants.

**Property B · Decision Sufficiency (conditional-MI form)** — measure "given the side information the policy already has, how much contract information about $Y_\pi$ survives projection" using **conditional MI**, not raw MI difference. The previous version wrote $L_{\mathrm{dec}} = I(\hat S; Y_{\mathcal C}) - I(\Pi_\pi(\hat S); Y_{\mathcal C})$, and the reviewer caught the fatal problem: **the policy's full input is $\pi(a\mid \hat S, o, \ell, \text{language})$, and the raw image $o$ often already carries age / provenance proxies**. Under the difference form, $L_{\mathrm{dec}} > 0$ only says "$\Pi_\pi(\hat S)$ in isolation is not a sufficient statistic for $Y_{\mathcal C}$" — **it does not say the policy actually lacks information** (the information may already be recovered from $o$).

This version redefines contract information loss as **conditional MI**:

$$\boxed{\;L_{\mathcal C}^{\pi} \;=\; I\!\big(Y_\pi\,;\,\hat S \,\big|\, Z_\pi,\, O,\, L\big),\qquad Z_\pi = \Pi_\pi(\hat S, O, L).\;}$$

(When the projection acts only on the contract, $Z_\pi$ can be simplified to $\Pi_\pi(\hat S)$.) This definition is much cleaner:

- $L_{\mathcal C}^{\pi} = 0$ **if and only if** $Y_\pi \perp\!\!\!\perp \hat S \mid Z_\pi, O, L$ — i.e. **$Z_\pi$ is sufficient for $Y_\pi$ given the other inputs the policy already has**.
- This directly answers the reviewer objection "raw image already contains object identity / provenance proxies": **because we use conditional sufficiency rather than unconditional MI, information already recovered from the raw observation is not miscounted as loss**. This sentence deserves to be in the body.

**Relations among the three**:

$$\text{Full semantic preservation}\;\Longrightarrow\;\text{Decision-relevant preservation}\;\Longrightarrow\;\text{Decision sufficiency}.$$

None of the implications reverse. **Decision sufficiency is the weakest property** — $L_{\mathcal C}^{\pi}=0$ permits collapsing two $\hat S$ values that $Y_\pi$ does not distinguish; **decision-relevant preservation is in the middle** — it protects only distinctions that would change action or safety; **full semantic preservation is the strongest** — a diagnostic property, not an interface requirement.

The killer line then becomes:

> **Decision-relevant semantic preservation is the actual interface requirement; full semantic preservation is a stronger diagnostic property; decision sufficiency is a weaker consequence.**

**A policy may drop information, but must either (a) preserve decision-relevant contract semantics, or (b) explicitly declare $\mathcal C_\pi$ + $Q_{\mathcal C_\pi}$ such that dropped distinctions are acknowledged by $Q_{\mathcal C_\pi}$ as outside its scope of concern**. Silently dropping decision-relevant contract semantics — without declaring — is the interface violation.

### 0.2.2 Three-loss decomposition: $L_{\mathrm{declared}} / L_{\mathrm{rep}} / L_{\mathrm{decision}}$

The previous version split $\Pi_\pi$ only into $q_\pi$ and $e_\pi$, and left an engineering principle "$e_\pi$ must never silently drop anything that $q_\pi$ said was preserved" — the reviewer was right to push on this: **even if $e_\pi$ is injective at the representation level, $\pi_\theta$ may still collapse those distinctions back at the final decision**. The $q_\pi$ / $e_\pi$ boundary was drawn too absolutely. This version extends the pipeline to four stages and attaches one loss to each:

$$\boxed{\;\mathcal C\;\longrightarrow\;q_\pi\;\longrightarrow\;e_\pi\;\longrightarrow\;\pi_\theta\;\longrightarrow\;a.\;}$$

| Layer | Semantic role | Failure | Loss |
|---|---|---|---|
| $q_\pi$ | declares what may be dropped | **Declared loss** — $\mathcal C_\pi / Q_{\mathcal C_\pi}$ simply does not say | $L_{\mathrm{declared}}$ |
| $e_\pi$ | actually encodes what is preserved | **Representation loss** — declared to be preserved but training dynamics quietly drop it | $L_{\mathrm{rep}}$ |
| $\pi_\theta$ | decides using preserved distinctions | **Decision-use loss** — the encoder has it, the head learns a shortcut instead | $L_{\mathrm{decision}}$ |

The three definitions:

$$L_{\mathrm{declared}} \;=\; H\!\big(Q_{\mathcal C}(Y_{\mathcal C})\big) \;-\; H\!\big(Q_{\mathcal C_\pi}(Y_{\mathcal C})\big)\quad\text{(how much upstream decision-relevant entropy $\mathcal C_\pi$ intentionally discards)}$$

$$L_{\mathrm{rep}} \;=\; I\!\big(Y_\pi\,;\,\hat S \,\big|\, Z_\pi,\, O,\, L\big) \;=\; L_{\mathcal C}^{\pi}\quad\text{(post-$e_\pi$ conditional residual)}$$

$$L_{\mathrm{decision}} \;=\; \sup_{\hat S \not\sim_{\pi,\mathcal D} \hat S'}\; \big\|\pi_\theta(\hat S) - \pi_\theta(\hat S')\big\|_{\mathrm{action\text{-}distribution}}^{\!\!\perp}\quad\text{(measure of pairs collapsed to the same action distribution)}$$

$e_\pi$ injective **does not imply** $L_{\mathrm{decision}} = 0$ — this is exactly the "three layers on the policy side can all silently break the contract" point the piece keeps making. §6's four compliance evidence types and §6.7's skeleton table both draw on these three quantities directly.

### 0.3 Three boxed claims of this piece

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. The full policy pipeline $\mathcal C\to q_\pi\to e_\pi\to\pi_\theta$ is **an auditable semantic interface**, with each of the three layers capable of silently breaking the contract. This is the embryo of the piece's upgraded thesis — **A policy is a contract consumer, not merely a function approximator**.

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state heads, latent visuomotor policies, and VLA / diffusion / flow policies use different conditioning and action-generation mechanisms — but as contract consumers they must all answer **the same four questions**: what may I discard ($q_\pi$)? what did I actually retain ($e_\pi$)? how should decisions respond to contract interventions ($\pi_\theta$)? what happens when evidence becomes invalid (safety)? The target of criticism is the interface contract, not the model architecture.

> **Claim 3 · Contract compliance should be tested by controlled intervention, not inferred from end-to-end success.** End-to-end success measures whether the policy works, not whether it interpreted contract semantics correctly. Contract compliance requires a battery of **controlled tests** — invariance / equivariance / **order-constrained response** / task-conditional utility under contract interventions — **and needs an oracle baseline to bound the interpretation of each metric**. The compliance argument is a **multi-evidence conjunction**, not a single score.

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

**Note: these five are not five stages of one pipeline**. They are **five mutually-substitutable abstractions**, and different policy families pick one (or wire a few in parallel). The previous version drew them as a chain $\pi_{\mathrm{obs}}\to z_t\to\hat S_t\to b_t\to s_t$, and the reviewer was right that **$\hat S\to b$ is not a universal posterior relation**, and that real architectures often **skip** middle stages — pure visuomotor runs $o\to z\to\pi$, engineered heads run $o\to\hat S\to\pi$, end-to-end VLA effectively runs $o\to\text{token}\to\pi$. A better diagram is an **abstraction tree**:

```
                            ┌── visual latent   z_t         ──┐
                            │                                  │
Raw observation  π_obs  ────┼── multimodal token (VLM)         ├───►  policy π_θ
                            │                                  │
                            ├── structured state  Ŝ_t          │
                            │                                  │
                            └── belief / posterior  b_t        ──┘
```

Caption: **These are alternative abstractions, not stages of a universal pipeline.**

Back to the language of §0.2: the real policy-side question is **not** "can I consume longer token sequences", it is **which branch of this tree a policy chooses as its conditioning input, and whether it writes that choice into $\mathcal C_\pi + Q_{\mathcal C_\pi}$**. Engineered state takes the $\hat S_t$ branch, visual latent takes the $z_t$ branch, multimodal token effectively takes a fourth branch that tokenizes immediately after $\pi_{\mathrm{obs}}$ — **three choices correspond to three values of $q_\pi$, not to "one smart, one dumb"**.

**The common misuse of the phrase "multimodal fusion" on the policy side** is treating a cross-attention on one branch as if it were "already doing multimodal state estimation". It is not. Real state abstraction requires the fields inside $\hat S_t$ to **carry consistent semantics across sensor families and be jointly readable by four consumers — controller / policy / world model / diagnostics** — 9/14 §6 has already established this convention; what this section adds is **asking from the policy side once more: was $q_\pi$ declared explicitly, or was it silently replaced by $e_\pi$?**.

## 3. Four interface-mismatch failure modes

Once the §1 coordinates are instantiated in a concrete policy, contract semantics show up as **four concrete failure modes**. None of the four are theoretical worries; they are **what actually breaks in deployment**. All four are **facts about the interface**, not personality defects of any single family.

### 3.1 Failure 1: Multi-hypothesis silently collapsed

Inside the contract a single physical quantity may have multiple hypotheses ("does this track_id refer to the same object", "is this contact on the pad or the edge") with different posterior weights. **If a policy's $\mathcal C_\pi$ and $Q_{\mathcal C_\pi}$ do not declare an explicit hypothesis-readout slot, $e_\pi$ may — under the default concat + MLP dynamics — fold them into a posterior mean.** The issue is not "this model will definitely collapse"; the issue is **"the interface provides no structural guarantee for multi-hypothesis structure"**. Many common implementations — the visual-latent and multimodal-token lines included — **do not provide an explicit, typed hypothesis readout contract, therefore hypothesis collapse is an unprotected failure mode**. The absence of a slot does not imply that collapse will happen: a transformer can perfectly well encode `H1, H2, H3 + weights` into its latent; the point is that **the interface provides no guarantee**, and success depends entirely on training distribution and inductive bias. **This is already the strongest claim this piece needs; going further invites a reviewer hit**.

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

Faced with a multi-hypothesis posterior inside the contract, the policy must pick one of several **explicit readouts**: MAP, posterior sample, expected-mixture, or **mass-preserving top-$k$**.

$$\text{read}\!\big(\{(\mu_i, \Sigma_i, w_i)\}_{i=1}^{K}\big) \;=\; \left\{\begin{aligned}
&\mu_{\arg\max_i w_i} && \text{(MAP, drop low-weight hypotheses)}\\
&\textstyle\sum_i w_i\, \mu_i && \text{(posterior mean, an explicitly declared collapse)}\\
&\mu_i + L_i \epsilon,\;\; i \sim w,\;\; \epsilon \sim \mathcal{N}(0, I) && \text{(posterior sample)}\\
&\big\{(\mu_i, \Sigma_i, \tilde w_i)\big\}_{i \in \mathrm{top}\text{-}k} \;\cup\; \{w_{\mathrm{other}}\} && \text{(mass-preserving top-}k\text{)}
\end{aligned}\right.$$

The previous version called this primitive "calibration-aware top-$k$", and the reviewer was right — renormalizing after truncation **only solves probability-mass conservation, not calibration**. $\sum_i w_i = 1$ does not imply posterior calibrated. This version renames the primitive to **mass-preserving top-$k$**, and evaluation of calibration is moved to §6:

$$\tilde w_i \;=\; \frac{w_i}{\sum_{j \in \mathrm{top}\text{-}k} w_j}, \qquad w_{\mathrm{other}} \;=\; 1 - \sum_{i \in \mathrm{top}\text{-}k} w_i.$$

$\tilde w_i$ is the **within-top-$k$ renormalized** weight; $w_{\mathrm{other}}$ is the **residual mass**; both are passed downstream — only this way can a subsequent Bayesian update actually be mass-conserving. **This is exactly what a primitive should do: semantically honest, with dropped mass explicitly carried on the output**.

Whether the posterior is actually calibrated is an **evaluation property**, not part of the primitive. The piece's calibration diagnostics live in §6's representation-level evidence and comprise **five separate measures: ECE$_{\text{top-}k}$, NLL, Brier, calibration curve, coverage–credibility**, each independent of the top-$k$ truncation itself. Primitive layer and evaluation layer are fully separated, and a reviewer can no longer ask "What exactly makes your top-$k$ calibration-aware?".

The point is **not "mean is forbidden"** — mean is a perfectly legitimate readout, provided the collapse is **explicitly declared**. The real failure mode is "the interface provides no readout slot for hypothesis structure, $e_\pi$ can only implicitly merge via concat + MLP, and mean becomes the default". This distinction is critical: **this piece is against undeclared default collapse, not against collapse per se**.

### 4.2 `age_gate`: observation payload + metadata quintuple + derived trust

**A common interface-design bug** is to multiply staleness trust directly into the measurement: $x_c^\pi = \tau_c(a_c) \cdot \mu_c$. This **changes the physical value of the observation** — 10 N read at 100 ms age gets multiplied into "3 N", and the "3 N" at the policy input **looks** like "a 3 N force", not "a 10 N force whose trust has decayed". This directly violates the very distinction the contract wants to preserve: **$(F = 3\,\mathrm{N},\, a = 0)$ and $(F = 10\,\mathrm{N},\, a = 100\,\mathrm{ms})$ are two different semantic events**.

The previous version wrote the interface as "seven parallel slots", and the reviewer was right — **the heading says seven, but the formula has eight**; $\alpha$ (availability) was added later and never included in the primary count. This version **re-groups the fields** so the counting inconsistency disappears.

**(i) Observation payload** — the measurement and its covariance:

$$\text{payload}_c \;=\; (\mu_c,\;\Sigma_c).$$

**This payload exists only when $\alpha_c = 1$**. $\alpha_c = 0$ means the estimator has nothing to deliver at this instant; $q_\pi$ can only read "no-data" as a fact.

**(ii) Temporal / operational metadata (five fields)** — describing how, when, and by which sensor the payload was acquired, and how current it is:

$$m_c \;=\; \big(\underbrace{a_c}_{\text{age}},\;\underbrace{\ell_c}_{\text{latency / causal status}},\;\underbrace{h_c}_{\text{sensor health}},\;\underbrace{v_c}_{\text{validity (calibration)}},\;\underbrace{\alpha_c}_{\text{availability}}\big).$$

**(iii) Derived trust (one field, not a primitive)** — computed from the five-tuple:

$$q_c \;=\; \tau_c(a_c,\;\ell_c,\;h_c,\;v_c,\;\alpha_c,\;\ldots).$$

The ellipsis admits task-specific inputs (controller mode, current dynamics regime, etc.). The concrete form of $\tau_c$ (learned / analytic / piecewise) is a policy-specific design decision left outside the interface.

The complete policy-input structure is thus **"six primitive metadata fields + one derived trust field"** — $(\mu_c,\Sigma_c,a_c,\ell_c,h_c,v_c,\alpha_c) + q_c$, countable on fingers; the reviewer no longer needs to ask "seven or eight".

**Availability vs validity — semantic difference** — the previous version did not pull these apart, and the reviewer was right. A 2×2 table:

|  | $v_c = 1$ (valid) | $v_c = 0$ (invalid) |
|---|---|---|
| $\alpha_c = 1$ (payload present) | Normal evidence; policy consumes via §4.1 | Data exists but **must not be treated as valid evidence** (e.g. expired calibration); policy should be handled through §5.3 safety side |
| $\alpha_c = 0$ (no payload) | Semantically impossible (no data ⇒ validity bit meaningless) | Channel down / masked; policy reads the fact "no observation" |

One line: **availability says "is there a payload", validity says "does this payload count as valid evidence" — orthogonal, cannot be collapsed into one bit**. §5.2's degradation chain (masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption) is essentially **different patterns** among the five metadata fields, and this version aligns augmentation with slots one-to-one.

**If a downstream gate really is required, it should gate on uncertainty, not on measurement.** But **the previous version wrote $\tilde\Sigma_c = \Sigma_c / \tau_c(a_c)$ as a general principle "stale ⇒ effective uncertainty is inflated" — also rightly called out by reviewers**. The correct treatment of a stale observation is not necessarily simple inflation; the more general form is **propagating the latent state forward**:

$$p(x_t \mid y_{t-\Delta t}) \;=\; \int p(x_t \mid x_{t-\Delta t})\, p(x_{t-\Delta t} \mid y_{t-\Delta t})\, dx_{t-\Delta t}.$$

When the robot is stationary, vision age 200 ms may leave the measurement still very accurate; when the robot is moving fast, the same 200 ms can imply huge predictive uncertainty. **Inflation and propagation are two different treatments, not synonyms.** This piece positions $\Sigma / \tau$ explicitly as **a simple conservative approximation**:

$$\Sigma_c^{\mathrm{eff}} \;=\; \mathrm{Propagate}\!\big(\Sigma_c,\; \Delta t,\; u_t,\; f_{\mathrm{dyn}}\big) \qquad\text{(general form)}$$

$$\Sigma_c^{\mathrm{eff}} \;=\; \Sigma_c \,/\, \tau_c(a_c, \ell_c, h_c, v_c, \alpha_c) \qquad\text{(one conservative approximation)}$$

The right phrasing is: **staleness should modify the policy's uncertainty model; uncertainty inflation is one conservative implementation, while predictive state propagation is another.** This version demotes $\Sigma / \tau$ from "canonical" to "one implementation" and hangs `Propagate` next to it as the general form. $\alpha_c$ and $v_c$ still **hang on as parallel metadata slots** — they must not be folded into $\mu$, nor into $\tau$.

This looks like a small change but it semantically repairs `age_gate` from "discounting a measurement" to "measurement + facts about the measurement + a derived trust" — a concrete projection of the §0.2 $L_{\mathcal C}^{\pi}$ definition: mixing a measurement with **facts about** the measurement into a single scalar is a direct source of $L_{\mathrm{rep}}$.

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

Therefore this piece upgrades the primitive from `dependency_gate` to **`dependency_aware_fusion`**: its semantics is not "suppress correlated token pairs" but "let the policy or fusion layer know the statistical relation between evidence and react accordingly".

**Attention bias is only one implementation** and **must be written at the logit level, not on post-softmax weights** — the previous version wrote $\mathrm{Attn}'_{ij} = \mathrm{Attn}_{ij} - \beta\cdot\mathbb 1[\cdot]$, and the reviewer immediately noted that subtracting from post-softmax weights can drive them negative and break the probability distribution. This version moves the formula to the logits:

$$L'_{ij} \;=\; L_{ij} \;+\; b(R_{ij}), \qquad A_{ij} \;=\; \operatorname{softmax}_{j}\!\big(L'_{ij}\big).$$

Here $R_{ij}$ is a field-pair relation strength compiled from $Q_{\mathcal C_\pi}$ (it can be the $\Sigma_{12}$ correlation coefficient, or a shortest-path distance on the `correlated_with` graph), and $b(R)$ is a bias function that **can be positive, negative, or learned**.

The key difference is that **the sign and shape of $b$ no longer imply "correlated → suppress"** — it just feeds the dependency relation as an operator-level input into fusion, and $\pi_\theta$ or the downstream loss decides how to react. That is what the primitive's name actually means: **dependency-aware fusion**, not dependency suppression.

Other implementations include: covariance-aware fusion (turn `correlated_with` into $\Sigma_{12}$, feed Kalman / factor graph), hierarchical mixture (aggregate correlated fields under a shared latent so they cannot be sampled independently), and letting the network learn a bias matrix from `correlated_with` (the `learned` path for MLP heads).

It must also be honestly acknowledged that **`correlated_with` is a field-level semantic relation, whereas attention bias is a token-pair relation** — the two require a $R_{\text{field}} \to R_{\text{token}}$ mapping in between. **This is an unsolved compilation problem and deserves to be treated as an independent research direction.** This piece only stands `dependency_aware_fusion` up as a primitive; the logit-level bias formula is "one implementation", not canonical.

#### 4.3.3 `negative_evidence_read` (likelihood-side evidence)

**The previous version wrote negative evidence as $\Lambda^{-}(H)$, with the sign hard-coded to "not-detected ⇒ supports $\neg H$" — the reviewer was right**. If $H$ = "obstacle exists" and $E^-$ = "no obstacle detected", then $\Lambda^-(H) < 0$ is natural; but if $H$ = "scene is clear", the direction flips. Artificially mandating "negative evidence must be positive" or "must be negative" is fragile.

**This version generalizes the object to a pure conditional log-likelihood ratio**; negative evidence is just one specialization of the input:

$$\boxed{\;\Lambda(E;\,H,\,\mathcal O) \;=\; \log \frac{P(E \mid H,\,\mathcal O)}{P(E \mid \neg H,\,\mathcal O)}.\;}$$

Here $E$ is **any observable evidence** ($E^+$ = "observed", $E^-$ = "should have been observed but was not"), and $\mathcal O$ encodes observability / sensor health / field of regard / calibration status. **"Negative evidence" simply means feeding $E = E^-$ into this LLR — no separate primitive is needed**.

The verbal caveat is worth spelling out: **sign of $\Lambda$ depends on the hypothesis being tested** — how $H$ is defined decides whether the same $E^-$ produces $\Lambda > 0$ or $\Lambda < 0$; this is standard Bayesian evidence theory and does not need to be hard-coded.

This is the correct characterization of "what should have been seen but was not" — **only when $\mathcal O$ says something should have been seen does $E^-$ carry likelihood-side meaning about $H$**. This also plugs directly into the observability field defined in 9/14 §6: **negative evidence is likelihood-side evidence, not provenance**.

On the policy side: expose $\Lambda(E;H,\mathcal O)$ (or the logit of $\exp(\Lambda)$) as a field inside $[\hat S]_{\sim_\pi}$ so the policy or belief update can consume it. This primitive is the thinnest on engineering maturity but often the strongest in effect on hypothesis ranking.

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

**But there is a hole the previous version did not plug** — the reviewer was right on this. Suppose $age$ is highly correlated with image embedding (e.g. "more complex scene ⇒ slower processing ⇒ larger age"). Then a probe from $z_\pi$ predicts age with ease:

$$\mathrm{Acc}(age\mid z_\pi) \;=\; 99\%.$$

**This does not prove that age is retained by $e_\pi$** — it may just be a leak of scene difficulty from the image. **The correct diagnostic is a conditional / nuisance-controlled probe**:

$$\text{Retention}_{\mathrm{cond}}(age) \;=\; I(age;\, z_\pi \mid o),$$

i.e. given the raw observation $o$, does $z_\pi$ still **independently** carry age information? Alternatively, the benchmark can run "**same observation, different metadata intervention**" — hold $o$ fixed, vary $m_c$, and observe $z_\pi$'s response. This is philosophically aligned with §0.2.1 Property B using conditional MI rather than an MI difference: **any field for which the raw input already contains a proxy must be probed conditionally**.

#### Class B · Intervention-consistency constraint (the real core)

**The previous draft wrote this class generically as $D(\pi_\theta(\hat S), \pi_\theta(T_{\mathcal{C}}(\hat S)); \rho_{\mathcal{C}})$, and a reviewer immediately asked "where does $\rho_{\mathcal{C}}$ come from?"** — the four $T_{\mathcal{C}}$ do not all have the same "response pattern", and one of them should not even assume a response is required. This version splits intervention consistency into three tiers by response strength.

**Type I · Exact invariance / equivariance** (the cleanest tier). The transformation has a legal counterpart $T^{\mathcal{C}}_\pi$ on the action side, and we require:

$$\pi_\theta\!\big(T^{\mathcal{C}}(\hat S),\, o,\, \ell\big) \;=\; T^{\mathcal{C}}_\pi\!\big(\pi_\theta(\hat S,\, o,\, \ell)\big).$$

Typical: **frame transform** — move `reference_point` from A to B, $\tau$ transforms per the transport theorem, **the action side must undergo the corresponding coordinate transformation** ($\pi(T_g S) = T_g^A \pi(S)$). Also: **permutation of equivalent hypotheses** (hypotheses with the same posterior weight may be permuted; the action distribution must be invariant). This tier can be written as a hard loss, $\mathcal{L}_{\mathrm{consistency}}^{\mathrm{I}} = \|\pi_\theta(T\hat S) - T^\pi \pi_\theta(\hat S)\|^2$.

**Type II · Order-constrained response** (the middle tier — **previously called monotone response, this version makes it mathematical**). "Monotone" is not a word that can be used casually — one needs a partial order on the range of $M(\cdot)$ and the response functional to be a **well-defined scalar or totally-ordered value**. The previous version crammed variance, action norm, fallback probability, covariance PSD into the same $\preceq$, and the reviewer was right: **those $\preceq$ relations are not the same order at all**.

The correct framing: first define a **severity partial order on contract interventions** $T_1 \preceq_{\mathcal C} T_2$ (e.g. "older age = more severe", "lower observability = more severe", "validity=false = more severe than high age" — the order is defined by $\mathcal C_\pi$ and $Q_{\mathcal C_\pi}$, **not by the policy**), then specify a **response functional**

$$r:\mathcal P(\mathcal A) \;\longrightarrow\; \mathbb R$$

(it can be $P(\text{fallback})$, $\mathbb E[\|a\|]$, $P(\text{stop})$, $\mathbb E[\mathrm{safe\_margin}]$ and so on — each is a scalar with the total order $\le$), and require:

$$T_1 \preceq_{\mathcal C} T_2 \quad\Longrightarrow\quad r\!\big(\pi_\theta(T_1\hat S)\big) \;\le\; r\!\big(\pi_\theta(T_2\hat S)\big).$$

**This is what monotonicity actually means**. Example: let $r(\pi) = P_\pi(\text{fallback})$, $T_1$ = "age from 5 ms to 20 ms", $T_2$ = "age from 20 ms to 200 ms"; then $T_1 \preceq_{\mathcal C} T_2$ and we require $P_\pi(\text{fallback}\mid T_1) \le P_\pi(\text{fallback}\mid T_2)$.

But a real policy may well be **piecewise** —

```
age < 50 ms          →   nominal control
50 ms ≤ age < 100 ms →   fallback
age ≥ 100 ms         →   stop
```

This kind of response **is not monotone, but it is a legitimate contract-specified response relation**. So Type II's proper name is "**order-constrained response**", and **monotonicity is only one special case**. The loss is:

$$\mathcal{L}_{\mathrm{consistency}}^{\mathrm{II}} \;=\; \sum_{T_1 \preceq_{\mathcal C} T_2} \max\!\big(0,\; r(\pi_\theta(T_1 \hat S)) - r(\pi_\theta(T_2 \hat S)) + \delta\big).$$

$\delta$ is a margin; both $r$ and $\preceq_{\mathcal C}$ must be hard-coded in $\mathcal C_\pi$, not fit after the fact.

**Type III · Unconstrained intervention** (the weakest and most important tier). **Do not assume the policy must change.** Typical example: **provenance removal** — dropping a contributing sensor whose information is fully redundant may leave the optimal action unchanged, and that is fine. The correct question here is: **when the removed evidence was decision-relevant, does performance degrade?** This is handed to §6's CAG panel rather than being imposed as a training-time response. Written as a loss: **no intervention consistency at all; only ablation at evaluation**. This tier's very existence is a correction to the previous draft, which treated all four $T_{\mathcal{C}}$ as "must respond" and thereby mistook a Type III intervention for a Type II.

**None of these three tiers require the policy to predict anything explicitly**; they require **the response function to conform to contract semantics, or to be permitted to remain unchanged**. This is the **training-side counterpart** of §0.2's "$q_\pi$ is a declared quotient". Compared to "adding a few auxiliary prediction heads", this three-tier scheme fits the thesis better and is more research-flavored: **what we propose is not for the policy to copy the contract, but for the policy to respond — or legitimately not respond — according to the tier of intervention**.

### 5.2 Augmentation: degradation as a causal operator

9/14 §8.5 emphasized **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption** — six degradations with different **causal origins** and different **downstream readouts**. The previous version tried to map each degradation to "which slot it corresponds to", and the reviewer was right: **this one-to-one mapping over-simplifies**. Concrete examples:

- **latency** has primary effect on $a_c$; without compensation, secondary effect on $\mu_c$ (a delayed-time value) — one augmentation moves two slots at once.
- **bias** primary effect shifts $\mu_c$; secondary effects often inflate $\Sigma_c$ (the system knows calibration is untrustworthy) and even flip $v_c$.
- **corruption** may move $\mu_c, \Sigma_c, v_c, h_c$ simultaneously.
- **masking** primary effect sets $\alpha_c = 0$; secondary effect renders $q_c$ undefined and forces $\Sigma_c^{\mathrm{eff}}$ to fall back to prior.

This version **rewrites each degradation as a causal operator**:

$$T_d:\;(\mu_c,\,\Sigma_c,\,m_c)\;\longmapsto\;(\mu_c',\,\Sigma_c',\,m_c'),\qquad d \in \{\text{mask},\,\text{missing},\,\text{stale},\,\text{latency},\,\text{bias},\,\text{corruption}\}.$$

One-line principle: **each degradation has a primary semantic effect and potentially secondary effects on other fields** — the aug pipeline must **enumerate explicitly** which slots $T_d$ touches and by how much, not "class → single slot". §4.2's three-group structure (payload / metadata / derived trust) is directly usable here: **$T_d$ acting on $(\mu,\Sigma)$ is payload-level corruption, acting on $m_c$ is metadata-level augmentation, and $q_c$ must be derived from the modified $m_c$, never rewritten by the aug side** (otherwise "aug cheats and deployment does not see").

**The previous draft had a shortcut risk that reviewers caught**: "label each episode with a degradation class, carry it explicitly on the policy input" — if `degradation = stale_vision` is fed directly as a policy input, the policy learns $a = f(o, \text{label})$; but **in real deployment the label itself is not reliable**, and the shortcut is harmless in training and disastrous at deployment. This version explicitly separates two things:

**Observed metadata** $m_c = (a_c, \ell_c, h_c, v_c, \alpha_c)$ — obtained directly by the estimator or sensor driver, may enter the contract and may enter the policy input. **This is exactly §4.2's metadata quintuple**.

**Latent degradation class** $d_c \in \{\text{missing}, \text{stale}, \text{bias}, \text{corrupt}, \ldots\}$ — augmentation knows which class was injected, but at deployment this class is **latent**: it can only be inferred by the contract estimator from the $m_c$ stream, or used as a training annotation for loss weighting and sampling, **never as a ground-truth input to the policy by default**.

Concretely: the aug pipeline samples $d_c$, applies $T_{d_c}$ to generate $(\mu_c', \Sigma_c', m_c')$, then feeds only $m_c'$ into the policy and uses $d_c$ solely for loss weighting and per-class evaluation slicing. **This is not curriculum, it is conditioning on observed metadata** — the difference is that conditioning lets the policy see reliable $m_c'$, not reliable $d_c$. §5.1 Class B Type II order-constrained response pairs naturally with degradation-conditioned aug — **aug generates the $m_c$ shift induced by $T_{\mathcal{C}}$, loss measures whether the policy's response to $m_c$ matches $\preceq_{\mathcal C}$ and $r$**.

### 5.3 Safety filter and its interface to the contract: constraint-relevant validity

The constraint layer (CBF / shield / runtime verifier) **must read $a_{\mathrm{proposed}}$**, otherwise what is it filtering — that point is not up for negotiation. But the more precise claim of this piece is: **the safety filter should not treat the policy's confidence or the estimator's general-purpose validity bit as the sole evidence that a constraint holds; it should directly access constraint-relevant evidence**.

**Two kinds of validity must be distinguished here.** The estimator-side general `validity` bit says "this measurement is valid from a calibration / sensor-health perspective" — that is a field-level, general-purpose declaration. What safety actually cares about is **"is this constraint valid at this moment, under this predicate"** — that is constraint-level semantics. The two differ: `validity = true` does not imply that the constraint estimate is valid for **this particular safety predicate**; e.g. "joint-torque calibration OK" ≠ "current contact estimate supports the collision constraint".

This piece proposes that the safety filter read a **constraint-specific composite**:

$$v_j^{\mathrm{constraint}} \;=\; g\!\Big(\text{field validity},\; \text{observability},\; \text{hypothesis posterior},\; \text{age},\; \text{model coverage}_j\Big).$$

$g$ is a constraint-specific composition rule (e.g. a CBF side may require "distance estimate is stable under the current hypothesis, observability is sufficient, and dynamics model coverage reaches the current state region"). $v_j^{\mathrm{constraint}}$ is what the safety filter should actually read as **constraint-relevant evidence**. 9/14 §7 established that **track_id is a hypothesis** — the safety filter must not simply trust track_id matching, it must also check whether the hypothesis posterior is stable; whether two tracks merge or split directly determines the credibility of "how far is that obstacle".

**The single most important sentence of this section (raised by the reviewer, accepted and bolded here)**:

> **$v_j^{\mathrm{constraint}} = 0$ does NOT mean "constraint disabled."**
> **Invalid evidence cannot justify relaxing the constraint.**

The previous version's phrasing read as "if validity falls, constraint may switch off" — **that is dangerous safety semantics**. $v_j^{\mathrm{constraint}} = 0$ actually says: **the current evidence is not sufficient to positively support that this constraint holds** — this can trigger one of three legitimate responses, chosen by $g$ and the task:

1. **fallback** — switch to a more conservative controller or planner;
2. **conservative tightening** — **increase** the constraint margin (e.g. raise minimum distance from 20 cm to 50 cm), because "less evidence ⇒ more caution";
3. **stop** — halt the policy and wait for observation to recover.

**None of the three is "constraint off"**. The core safety fact is: **positive evidence can justify relaxing a constraint; absence of valid evidence never can**. This sentence lifts the piece from an ML-interface discussion to safety semantics, and it is the real definition of §6's safety-level compliance evidence — what is being tested is not "does the policy stop making mistakes when validity=0", but "**does the safety filter tighten — rather than relax — when validity=0**".

This also wires in §4.3.3's negative evidence: **"what should have been seen but was not" makes $v_j^{\mathrm{constraint}}$ false** — for example the radar swept this angle and saw nothing, $\Lambda(E^-; H_{\text{clear}}, \mathcal O)$ turns positive (under the "clear" hypothesis "nothing seen" is natural; under the "obstacle present" hypothesis "nothing seen" is unnatural — **the direction depends on how $H$ is defined, see §4.3.3**), and the collision constraint's $v^{\mathrm{constraint}}$ should stay active or even **tighten**, not deactivate because the policy's belief is optimistic.

**If these three (constraint-specific validity, observability, negative evidence) do not enter the safety filter, the filter will infer constraint validity from the policy's belief** — which is especially dangerous in low-observability regions. The policy's belief is optimistic precisely because it cannot see the contract's observability / validity / negative evidence, and **if the filter also cannot see them, the two go blind together**.

### 5.4 Echo with 9/10 Part 3 evaluation

Among the three sim-utility dimensions (prediction / ranking / decision), policy-side evaluation is mainly about **decision** — but with an added **contract-preservation dimension**: how much the policy degrades when the contract is torn apart is a **lower bound** on how much it depends on the contract. This dimension maps to the CAG metric of §6, and needs an oracle baseline to bound its interpretation.

## 6. Evaluation: four types of compliance evidence, not four metrics

Aligned with the §4 primitives and the three tiers of §5.1, this piece proposes **four types of compliance evidence** — **not a four-level metric panel, but four evidence categories each answering a different question**. The previous version wrote them as a metric hierarchy, and the reviewer was right: **no single score among probe / CAG / safety-pass constitutes semantic compliance**; compliance must be a **multi-evidence conjunction**.

$$\boxed{\;\text{Compliance Evidence} \;=\; \big\{E_{\mathrm{semantic}},\;E_{\mathrm{representation}},\;E_{\mathrm{decision}},\;E_{\mathrm{safety}}\big\}.\;}$$

Each type answers a distinct question:

| Type | Question answered |
|---|---|
| **$E_{\mathrm{semantic}}$** | **Does the policy respond correctly to a known semantic transformation?** |
| **$E_{\mathrm{representation}}$** | **Is the contract information recoverable from the representation (given the side inputs)?** |
| **$E_{\mathrm{decision}}$** | **Does contract structure change task utility when it should?** |
| **$E_{\mathrm{safety}}$** | **Does degradation cause conservative / required guardrail behavior?** |

Three explicit caveats: **probe ≠ semantic compliance, CAG ≠ semantic compliance, safety pass ≠ representation retention**. These four evidence types **cannot substitute for each other and cannot be collapsed into a single scalar**. Each is unfolded below. The temporal and source slices are specializations of $E_{\mathrm{decision}}$ (**SDS** and **$\Delta J_{\mathrm{prov}}$**) — not new metrics but focused views of the decision-layer panel.

### 6.1 Semantic evidence $E_{\mathrm{semantic}}$: Invariance / Equivariance Test

Corresponding to §5.1 Type I. Given a family of contract transformations with known $T^{\mathcal{C}}_\pi$ (frame / coordinate / hypothesis permutation), measure:

$$\mathrm{Equiv}(\mathcal{C}) \;=\; \mathbb{E}_{\hat S}\!\left[d\!\left(\pi_\theta(T^{\mathcal{C}}\hat S),\; T^{\mathcal{C}}_\pi\,\pi_\theta(\hat S)\right)\right].$$

$\mathrm{Equiv} \to 0$ is a hard requirement; large $\mathrm{Equiv}$ means either $e_\pi$ learned it wrong, or $q_\pi$ dropped the quotient entirely. This is the cleanest type, because the rule is mathematically defined — no oracle and no $J$ definition needed. §3.2 Failure 2's severity can be quantified directly by $\mathrm{Equiv}(\text{frame})$.

### 6.2 Representation evidence $E_{\mathrm{representation}}$: Conditional Probes + Counterfactual HPC + Action-Relevant HSS + Calibration

**Conditional probe** (upgraded §5.1 Class A): Retention$_{\mathrm{cond}} = I(\text{field}; z_\pi \mid o)$ — hold raw observation $o$ fixed and measure how much contract information $z_\pi$ **independently** carries. This is the necessary form to avoid image-proxy leakage.

**The previous version's HPS used $\max_k$ and had an obvious loophole** — with $K = 8$, if the policy only serves $H_1$ correctly and fails $H_2$ through $H_8$, $\max_k$ still scores full, but hypothesis structure is not preserved at all. This version splits HPS into two metrics pointing in opposite directions, and **defines both via counterfactual intervention**.

**Hypothesis Coverage (HPC)** — the reviewer was right that treating "$H_k$ is input as truth" as ground truth is inappropriate; the real world has only one state. This version makes it explicit: **each $H_k$ in HPC is constructed via a counterfactual contract intervention $T_k^{\mathrm{hyp}}$** — "put $H_k$ as the true latent, hold nuisance observations fixed as appropriate":

$$\mathrm{HPC} \;=\; \frac{1}{K} \sum_{k=1}^{K} U\!\big(\pi_\theta(T_k^{\mathrm{hyp}}(\hat S)),\;\mathcal{A}^{*}_k\big).$$

$U(\cdot, \mathcal A^*_k)$ is the utility of the policy's output versus the counterfactual contract's **ideal action set** $\mathcal A^*_k$; $\mathcal A^*_k$ is constructed in the benchmark from **privileged simulator state, oracle planners, or offline expert rollouts**, just like §6.5's oracle. This change aligns HPC fully with the piece's intervention philosophy — **HPC tests "if we counterfactually make $H_k$ true, does the policy do the right thing", not "given an $H_k$-labelled input, does the policy read it correctly"**.

**Hypothesis Separation Score (HSS)** — **must be averaged only over action-relevant hypothesis pairs** — the reviewer caught gaming: if $\mathcal A^*(H_1) = \mathcal A^*(H_2)$, different outputs are not a virtue, **they are noise**. Define the action-relevant pair set:

$$\mathcal R \;=\; \big\{(i, j): \mathcal A^{*}_i \not\equiv \mathcal A^{*}_j\big\}.$$

HSS averages only over $\mathcal R$:

$$\mathrm{HSS} \;=\; \frac{1}{|\mathcal R|} \sum_{(i, j) \in \mathcal R} D\!\big(\pi_\theta(\cdot \mid T_i^{\mathrm{hyp}}\hat S),\;\pi_\theta(\cdot \mid T_j^{\mathrm{hyp}}\hat S)\big).$$

**separation is useful only when distinctions are decision-relevant** — this sentence must be locked down, or HSS can be maxed out by a policy that outputs a different random action per hypothesis.

HPC and HSS together correspond to decision-relevant semantic preservation (§0.2.1 Property A′) — **coverage** ensures every action-relevant hypothesis is supported; **separation** ensures the policy preserves action-relevant hypothesis distinctions; **and neither penalizes pairs that should not be distinguished**. These three constraints together make representation evidence meaningful.

**Calibration diagnostic** — $\mathrm{ECE}_{\text{top-}k}$, NLL, Brier, calibration curve, coverage–credibility listed in parallel (§4.1 has already stripped these from the primitive; this is their real home).

### 6.3 Temporal slice: Staleness Response Compliance (SDS)

**The previous SDS definition $D(\pi(\cdot|\mathrm{do}(a=a_1), o), \pi(\cdot|\mathrm{do}(a=a_2), o))$ with "expected variance monotonically rising" needs two tightenings.** First, SDS is not a scalar but a family — the reviewer is right that "variance monotonically rising" is not a universal law. Real policies can be **piecewise** (§5.1 Type II order-constrained response has formalized this): age < 50 ms uses vision, age > 100 ms switches to proprio-only fallback, age > 200 ms stops; the response curve is piecewise discontinuous, **and it is legitimate as long as it agrees with the relation declared by $\preceq_{\mathcal C}$ and $r$**. Second, like §6.5's CAG, SDS needs an oracle baseline to bound what counts as "correctly responding".

This piece redefines SDS as **response fidelity against oracle**:

$$R_\pi(a) \;=\; \pi_\theta\!\big(\cdot \,\big|\, \mathrm{do}(a_c = a),\, o\big),$$

$$\mathrm{SDS} \;=\; D\!\big(R_\pi(a),\; R^{*}(a)\big),$$

where $R^*(a)$ is **the oracle policy's response curve under the same intervention**, and $D$ is a curve-level divergence (e.g. KL integrated over $a$, or sliced Wasserstein). The response property $R$ is task-defined and may take **variance / action norm / fallback probability / safety margin / stop probability** — **it must be the same $r$ declared in §5.1 Type II**, otherwise training and evaluation talk past each other. Flat is not automatically bad; compare against the oracle's expected response curve.

The derivative form is retained as one local characterization of response shape:

$$\left.\frac{\partial\, \mathbb{E}\!\big[\pi_\theta(\cdot \mid \mathrm{do}(a_c = a),\, o)\big]}{\partial a_c}\right|_{a}\quad\text{compared to the same-order derivative of } R^*.$$

### 6.4 Source slice: $\Delta J_{\mathrm{prov}}$ and $\Delta J_{\mathrm{neg}}$

**The previous $\Delta J_{\mathrm{prov}}$ stays, but its interpretation must be updated alongside §4.3.** Dependency-aware fusion is not "suppress double-counting", so $\Delta J_{\mathrm{prov}}$ does not measure "did the policy suppress correlated pairs" — it measures **"did the policy consume `correlated_with` as legitimate evidence structure"**. $J$ is always higher-is-better:

$$\Delta J_{\mathrm{prov}} \;=\; J\!\big(\pi_\theta \mid \text{provenance + correlated\_with}\big) \;-\; J\!\big(\pi_\theta \mid \text{both} = \varnothing\big).$$

$\Delta J_{\mathrm{prov}} > 0$ = the policy really uses source structure; $\Delta J_{\mathrm{prov}} \approx 0$ = it treats source as decoration; $\Delta J_{\mathrm{prov}} < 0$ = conditioning actively hurts, usually indicating that conditioning conflicts with the backbone's inductive bias and deserves separate debugging.

**Add a companion $\Delta J_{\mathrm{neg}}$** — remove $\Lambda(E; H, \mathcal O)$ from §4.3.3 and measure the drop in hypothesis-ranking accuracy:

$$\Delta J_{\mathrm{neg}} \;=\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda\big) \;-\; J_{\mathrm{rank}}\!\big(\pi_\theta \mid \Lambda = \varnothing\big).$$

This directly probes the negative_evidence primitive and answers the safety-side interface in §5.3.

### 6.5 Decision evidence $E_{\mathrm{decision}}$: Contract Ablation Gap (CAG) + Oracle baseline

**Given a specific collapse operator $\mathrm{collapse}_X$ on the contract** (which silently folds layer $X$ of contract structure):

$$\mathrm{CAG}_X \;=\; J_{\mathrm{full}} \;-\; J_{\mathrm{collapse}_X}, \qquad J_{\mathrm{full}} \equiv J(\pi_\theta \mid \hat S),\;\; J_{\mathrm{collapse}_X} \equiv J(\pi_\theta \mid \mathrm{collapse}_X(\hat S)),$$

with four collapses each corresponding to one invariant:

- $\mathrm{collapse}_{\mathrm{hyp}}$: fold the hypothesis set into a single Gaussian or a point estimate.
- $\mathrm{collapse}_{\mathrm{age}}$: flatten every channel's `age / health / latency / availability`.
- $\mathrm{collapse}_{\mathrm{prov}}$: drop `contributing_mask` and `correlated_with`, let policy attend indiscriminately.
- $\mathrm{collapse}_{\mathrm{neg}}$: drop $\Lambda(E; H, \mathcal O)$.

**But reviewers are right — $\mathrm{CAG}_X$ alone cannot distinguish "policy not using contract" from "contract not informative enough for this task", nor rule out the policy treating some field as a shortcut.** This version adds an **oracle privileged-state baseline**:

$$J_{\mathrm{oracle}} \;=\; J(\pi^{*}_{\mathrm{oracle}} \mid s^{\mathrm{priv}}), \qquad \mathrm{Gap}_{\mathrm{oracle}} \;=\; J_{\mathrm{oracle}} - J_{\mathrm{full}}.$$

**The previous version's four-quadrant interpretation was actually too strong** — the cell "CAG high, oracle slightly above full ⇒ policy really reads contract" is not mathematically supported. $\mathrm{CAG} > 0$ only says **removing this structure drops task utility**; it does not say the policy uses it through a correct semantic mechanism. The real situation is often that training data has `age ↔ task difficulty` heavily correlated, and the policy learns a shortcut $a = f(age)$ rather than $a = f(\text{actual sensor staleness semantics})$. This version downgrades the table to **"what each observation supports"**:

| Observation | What it supports |
|---|---|
| CAG high | Contract structure is useful for the task (may come from semantic use, or from a shortcut) |
| CAG ≈ 0 | Contract structure may not be necessary for this task |
| CAG high + §5.1 Type I/II pass | **More evidence** for semantic use |
| CAG high + intervention fail | **Likely a shortcut** — go back and check training distribution in §4.3 / §5.2 |
| CAG ≈ 0 + oracle gap high | Contract has information the policy is not using — locate which layer dropped it via $q_\pi / e_\pi / \pi_\theta$ three-loss decomposition |

That is:

$$\boxed{\;\mathrm{CAG} \;\neq\; \text{contract understanding}.\;}$$

$$\boxed{\;\mathrm{CAG} \;=\; \text{task-conditional utility sensitivity}.\;}$$

**CAG is a decision-layer aggregate; it must be viewed jointly with $E_{\mathrm{semantic}}$ (invariance / equivariance) + $E_{\mathrm{representation}}$ (conditional probe + HPC/HSS) + §5.1 Type I/II controlled response** — only the conjunction is evidence of semantic use. This version demotes CAG from "the aggregate" back to "decision-layer aggregate", so the four compliance evidence types stay complete.

### 6.6 Safety evidence $E_{\mathrm{safety}}$: Constraint intervention

Actively suppress §5.3's $v_j^{\mathrm{constraint}}$ in deployment / semi-simulation (inject calibration drift, cut observability, raise $\Lambda(E^-; H, \mathcal O)$), and check whether the safety filter **enters the correct guardrail at the correct moment**, and whether guardrail activation is **attributable to which input of $v_j^{\mathrm{constraint}}$**. **The key criterion is no longer "did the filter stop the policy", but "did the filter tighten the constraint"** (§5.3 has locked this down: invalid evidence cannot justify relaxing constraint). The three correct responses are fallback, conservative tightening, stop; the **wrong response is relaxing** — one observed relaxation ⇒ $E_{\mathrm{safety}}$ fails outright. This type is the direct instantiation of §5.3's interface, and the real test of whether the piece's interface proposal is **actually deployable**.

The four types together form a **multi-evidence compliance argument**: **semantic measures "does it respond by rule", representation measures "is the information still there (and independent of raw observation)", decision measures "did it use it / is there a shortcut", safety measures "does the guardrail tighten when things degrade"**. None of them replaces an end-to-end success rate — they measure the policy-side **read-completeness** of the contract, not the policy's **expressive power**. This aligns fully with §0.3 Claim 3: **contract compliance must be tested by controlled intervention and supported by a conjunction of four evidence types**.

### 6.7 The theoretical skeleton table of this piece

Compress §0 through §6 into one table — this is what a reviewer most wants to see:

| Layer | Object | Failure | Evidence |
|---|---|---|---|
| Contract | $\mathcal C$ | schema ambiguity | schema audit |
| Declaration | $q_\pi$ (induced by $\mathcal C_\pi$ + $Q_{\mathcal C_\pi}$) | undeclared semantic collapse | quotient audit (can you display a $Q_{\mathcal C_\pi}$ list?) |
| Representation | $e_\pi$ | information loss | conditional probe $I(\text{field};z_\pi\mid o)$ |
| Decision | $\pi_\theta$ | wrong use / shortcut | Type I equivariance + Type II order-constrained + Type III ablation |
| Safety | $g_j$ | unsafe interpretation of invalid evidence | constraint intervention (does it **tighten**, not **relax**) |

**Four core quantities**:

$$\boxed{\begin{aligned}
L_{\mathrm{declared}} &: \;\text{what } q_\pi \text{ intentionally discards};\\
L_{\mathrm{rep}} &= I(Y_\pi;\hat S\mid z_\pi, O, L);\\
E_{\mathrm{int}} &: \;\text{intervention-consistency error (Type I / II)};\\
\Delta J &: \;\text{task-conditional utility gap (CAG, } \Delta J_{\mathrm{prov}}, \Delta J_{\mathrm{neg}}).
\end{aligned}}$$

The four quantities sit on four layers respectively, **each can be audited independently, or combined into a compliance argument**. This is the concrete shape of the piece's journey from "add metadata to a VLA" to "contract-aware policy design".

## 7. A minimal executable interface sketch

Combine the §4 primitives and §6 four evidence types into a Python class skeleton. **Not a concrete policy; a readable interface contract**.

```python
class StructuredStateView:
    def __init__(self, contract: StructuredState,
                 consumer_contract: ConsumerContract,     # C_pi
                 query_family: QueryFamily):              # Q_{C_pi}; together they induce ~_pi
        self.contract = contract
        self.C_pi = consumer_contract
        self.Q = query_family

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
                            "covariance_fusion", "hierarchical_mixture"] = "logit_bias_learned",
        negative_evidence: Literal["ignore", "condition", "belief_update"] = "condition",
    ) -> PolicyInput:
        """
        Project StructuredState (upstream, 9/14) to PolicyInput (downstream, this piece).
        The (C_pi, Q_C_pi) pair literally defines ~_pi; q_pi is the induced quotient map.
        e_pi is the encoder of the specific backbone and must not silently drop anything
        q_pi declared preserved; pi_theta may further collapse distinctions at the
        action level (see three-loss decomposition in §0.2.2).
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
                # calibration (ECE_top-k / NLL / Brier) is measured in §6.2, NOT here.

            # --- age_gate: three groups, never multiplicative on mu ---
            # (i) payload: mu, Sigma
            # (ii) metadata quintuple: (a, ℓ, h, v, α)
            # (iii) derived trust q = τ(a, ℓ, h, v, α, ...)
            age          = h.age
            validity     = h.validity_ok          # v_c
            availability = h.available            # α_c (payload exists?)
            health       = h.sensor_health        # h_c
            latency      = h.causal_status        # ℓ_c
            trust = clamp(
                tau_curve(age, latency, health, validity, availability,
                          schema.tau_config[field_name]),
                min=schema.trust_eps,             # numerical guard: τ ≥ ε
            )                                      # q_c is meta, does NOT scale μ_c

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
                age=age, validity=validity, availability=availability,
                health=health, latency_status=latency,
                trust=trust, w_payload=w_payload,
                residual_declared=(residual if mode == "topk" else None),
            )

        # --- provenance / dependency / negative evidence, three reads ---
        prov_mask = self.contract.contributing_mask if provenance == "harden" else None
        if dependency == "logit_bias_learned":
            # b(R_ij) is learned and CAN be positive or negative.
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
                           C_pi=self.C_pi, Q=self.Q)


class ContractAwarePolicy(nn.Module):
    def __init__(self, backbone, schema: PolicySchema, cfg: ContractReadConfig,
                 consumer_contract: ConsumerContract, query_family: QueryFamily):
        self.backbone = backbone
        self.schema = schema
        self.cfg = cfg                 # cfg is a *materialization* of q_pi,
        self.C_pi = consumer_contract  # but the actual declared quotient lives in (C_pi, Q)
        self.Q = query_family
        # three loss sites:
        #   L_declared      — measured on (C_pi, Q): did we explicitly declare the loss?
        #   L_rep           — measured on e_pi output z_pi: I(Y; S | z_pi, o, l)
        #   L_decision      — measured on pi_theta: does action distribution still
        #                     distinguish pairs that ~_{pi,D} says should differ?

    def forward(self, state: StructuredState, obs, lang) -> ActionDistribution:
        x = StructuredStateView(state, self.C_pi, self.Q).project(
            self.schema, **self.cfg.as_kwargs())
        return self.backbone(x, obs, lang)
```

Three caveats written in stone:

- **(i)** This is not the only way to read; **`cfg` is a materialization of §0.2's $q_\pi$, but the actual declared quotient is the pair $(\mathcal C_\pi, Q_{\mathcal C_\pi})$** — the same contract should be projected with different `cfg` by a VLA and a Diffusion Policy, and the optimal `uncertainty` differs between engineered-state heads and visual-latent heads. **The interface spec must display a $Q_{\mathcal C_\pi}$ list**, otherwise $q_\pi$ degenerates back into encoder behavior.
- **(ii)** This interface **only addresses the input side**; §5.1's three-tier intervention constraints, §5.2's observed-metadata-only aug, §5.3's $v_j^{\mathrm{constraint}}$ direct wiring — if any one is left unchanged, $(\mathcal C_\pi, Q_{\mathcal C_\pi})$ may be written beautifully and still be routed around by $\pi_\theta$'s training dynamics ($L_{\mathrm{decision}}$ blows up directly). **Three loss sites, none can be missing**.
- **(iii)** `dependency="logit_bias_learned"` is the most-recommended implementation for transformer-family backbones — **$b(R_{ij})$ lives at the logit level, can be positive or negative, and is learned end-to-end by $\pi_\theta$; it no longer implies correlated → suppress**. MLP heads use `covariance_fusion`, Kalman / factor-graph fusion use `covariance_fusion`, grouped latent uses `hierarchical_mixture` — **logit_bias_learned is only one of several implementations of dependency_aware_fusion, not the primitive itself**.

## 8. Three closing claims and one upgraded thesis (aligned with §0.3)

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. The full policy pipeline $\mathcal C\to\mathcal C_\pi\to q_\pi\to e_\pi\to\pi_\theta\to a$ is **an auditable semantic interface**, with each of the three layers capable of silently breaking the contract, corresponding to §0.2.2's three losses: $L_{\mathrm{declared}}$ (what $q_\pi$ intentionally discards), $L_{\mathrm{rep}} = I(Y_\pi;\hat S\mid z_\pi, O, L)$ (post-$e_\pi$ conditional residual), and $L_{\mathrm{decision}}$ (whether $\pi_\theta$ collapses decision-relevant distinctions at the action level). **$e_\pi$ injective does not imply $L_{\mathrm{decision}} = 0$** — this is the most important strengthening in this version.

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state heads, latent visuomotor policies, autoregressive VLAs, diffusion and flow policies all use different conditioning and action generators — but as contract consumers they must answer **the same four questions** (this piece's real thesis, boxed below). The target of criticism is the interface contract, not the model architecture; π0 is VLA + flow matching, Diffusion Policy is visual-latent + diffusion, ACT is visual-latent + generative sequence decoder — **slicing along two orthogonal dimensions is closer to reality than three families, and less easily misled by the intuition "some family is inherently better"**.

> **Claim 3 · Contract compliance should be tested by four types of evidence, not a single score.** End-to-end success measures whether the policy works, not whether it interpreted contract semantics correctly. Contract compliance requires §6's **four compliance evidence types** together — $E_{\mathrm{semantic}}$ (invariance / equivariance) + $E_{\mathrm{representation}}$ (**conditional probe + counterfactual-intervention HPC + action-relevant-pair HSS + calibration diagnostic**) + $E_{\mathrm{decision}}$ (**CAG + oracle baseline, with CAG ≠ contract understanding, only task-conditional utility sensitivity**) + $E_{\mathrm{safety}}$ (**constraint intervention, testing whether the filter tightens, not whether it halts the policy**) — and requires distinguishing §5.1's three response tiers (Type I exact equivariance / **Type II order-constrained response** / Type III unconstrained). Treating Type III as Type II and forcing a response was a specific error of the previous draft; this version hands it to ablation.

A closing line: **try to stop using the word "fusion"** — on the time axis it asks "when to merge", on the semantic axis it asks "merge into what"; together, 9/14 and this piece split the second question into **what upstream delivers + what the policy declares it consumes as a contract consumer + what $q_\pi$ allows to be dropped + what $e_\pi$ preserves + what $\pi_\theta$ uses**. The first question (when) is largely answered by §1's two-dimensional grid — **timing is a consequence of the interface, not a decision variable of the interface**.

**One step further, the proposition that this version finally stands up** — the reviewer's framing, now accepted as this piece's own thesis:

$$\boxed{\;\textbf{A policy is a contract consumer, not merely a function approximator.}\;}$$

And a contract consumer must answer at least four questions:

$$\boxed{\begin{array}{ll}
\text{1.} & \textbf{What may I discard?} \quad (q_\pi, \mathcal C_\pi, Q_{\mathcal C_\pi})\\[2mm]
\text{2.} & \textbf{What did I actually retain?} \quad (e_\pi, L_{\mathrm{rep}})\\[2mm]
\text{3.} & \textbf{How should decisions respond to contract interventions?} \quad (\pi_\theta, L_{\mathrm{decision}})\\[2mm]
\text{4.} & \textbf{What happens when the evidence becomes invalid?} \quad (\text{safety}, v_j^{\mathrm{constraint}})
\end{array}}$$

Once these four questions stand, **VLA / Diffusion / Flow / ACT / SAC / PPO are all just implementation coordinates, no longer theoretical classifications**.

And §6's intervention battery, no longer just "a benchmark for the next piece", becomes the **experimental closure** of this piece's theoretical frame:

$$\boxed{\;\mathcal C\;\longrightarrow\;\mathcal C_\pi\;\longrightarrow\;q_\pi\;\longrightarrow\;e_\pi\;\longrightarrow\;\pi_\theta\;\longrightarrow\;\mathcal B_{\mathcal C},\quad \mathcal B_{\mathcal C} = \{T_{\mathrm{frame}},\,T_{\mathrm{hyp}},\,T_{\mathrm{age}},\,T_{\mathrm{validity}},\,T_{\mathrm{prov}},\,T_{\mathrm{neg}}\}.\;}$$

The next piece can directly stand up a **Contract-Preserving Policy Benchmark** — run SAC / PPO / Diffusion Policy / ACT / OpenVLA / π0 through the same $\mathcal B_{\mathcal C}$ intervention suite, with each loss segment matched to its own measurable. At that point this series upgrades from "I think policy should read contract" to **"given the same Structured State Contract, how do we systematically test whether different policies are contract-compliant"**. Stand that line up, and the series has been worth it.

## Sources

All arXiv IDs below have been verified online; journal-only citations do not carry an arXiv link. Grouped by the sections they support.

### A · VLA family (supports §1 grid, §3 Failures 1–2, §5.1 Class B)

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) (**paper fact**: robot actions explicitly expressed as text tokens, jointly fine-tuned with a VLM. §3.2 Failure 2's tokenizer-side shape; "contract flattening" is this piece's analysis, not an admitted limitation of the paper)
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) (**paper fact**: 7B VLA trained on large-scale robot demonstrations, emphasizing fine-tuning and generalization; **this piece's analysis**: configurations include multi-camera / depth / proprioceptive state encoding, but "supports input" ≠ "how far along the contract chain the policy reads")
- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) (**paper fact**: pretrained VLM + proprio token + noisy action chunk + flow matching; **this piece's analysis**: Under the Structured State Contract defined here, π0's conditioning interface does not expose an explicit slot for hypothesis / provenance / age / negative evidence — "Continuous actions do not imply structured state semantics" is this piece's analysis, not the paper's self-declared limitation)
- Octo Model Team, *Octo: An Open-Source Generalist Robot Policy*, RSS 2024 · [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) (transformer-based readout · a reference for the logit_bias_learned path within §4.3.2's dependency_aware_fusion)

### B · Diffusion / Generative-Sequence / Flow-Matching Policy (supports §1 grid, §5.1)

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 · [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) (**paper fact**: RGB stack + proprio concat + conditional denoising diffusion, emphasizing action-distribution multimodality; **this piece's analysis**: action-side multimodality ≠ state-side hypothesis preservation — an inference under this schema, not an admitted limitation)
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT / ALOHA), RSS 2023 · [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) (**paper fact**: CVAE + transformer encoder-decoder, core is **action chunking over sequences** — this piece places ACT under "generative sequence decoder", alongside diffusion and flow matching but not inside the diffusion family)
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 · [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) (**paper fact**: establishes the flow-matching objective as vector-field regression for generative modeling / CNF — **flow matching itself is not a robot action chunking paper**; "continuous robot action chunks" is a downstream application introduced by π0-class work; the citation chain is split here as "Lipman establishes objective / π0 applies it to action chunks")

### C · Uncertainty, calibration, and belief-space references (support §4.1, §5.1, §6.2 calibration diagnostic, §6.5 CAG)

- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) (modern networks over-confident; temperature scaling origin · the evaluation basis of §6.2's calibration diagnostic; no longer hangs on the primitive)
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels* (PlaNet / RSSM), ICML 2019 · [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) (deterministic + stochastic latent · a reference for §4.2 predictive uncertainty propagation)
- Hafner et al., *Mastering Diverse Control Tasks through World Models* (DreamerV3), Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) (discrete + continuous mixed latent, KL balancing · an adjacent path for §4.1 posterior readout and §6.2 counterfactual-intervention HPC)

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
