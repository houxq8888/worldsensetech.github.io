---
title: 'After the Contract Stands: What Do VLA, Diffusion Policy and π0 Actually Consume?'
slug: "2026-09-15-policy-side-interface"
date: 2026-09-15
draft: false
categories: ["Embodied AI", "Policy Learning"]
tags: ["Embodied AI", "Policy Learning", "VLA", "Diffusion Policy", "π0", "RT-2", "OpenVLA", "Action Tokenization", "Structured State Contract", "Contract-Preserving Projection", "Contract Information Loss", "Contract-Read Primitives", "Intervention Consistency", "Belief State", "Multimodal Hypothesis", "Provenance", "Dependency Graph", "Negative Evidence", "Uncertainty Calibration", "Safety Filter", "Contract Ablation Gap", "Hypothesis Preservation", "Staleness Response", "Evaluation Metrics"]
description: 'The multimodal-fusion piece stood up the upstream deliverable as a Structured State Contract. This piece asks the dual question: if the estimator really delivers per contract, can the policy side actually consume it. The core object is an explicitly defined policy projection $\Pi_\pi$ and a verifiable interface property, contract-preserving: information may be dropped, but contract semantics may not be dropped silently. This piece no longer slices along three mutually-exclusive families (VLA / Diffusion / engineered head); instead it decomposes any policy along two orthogonal dimensions—conditioning representation × action head—and gives a seven-row grid. At the interface layer three families of contract-read primitives are proposed: `mode_select`, `age_gate` (parallel fields of measurement / uncertainty / age / validity / trust rather than a multiplicative decay), and a three-way split of provenance / dependency / negative evidence. Training constraints split into representation-side probes and intervention-consistency losses; on the evaluation side, `Contract Ablation Gap (CAG)` is proposed as the top-line aggregate metric, with HPS / SDS / PCE as per-dimension diagnostics and SDS rewritten as a Staleness Response Curve under controlled intervention. The piece lands on three claims: contract semantics can be lost at the policy boundary; contract preservation is not architecture-specific; contract compliance should be tested by intervention, not inferred from end-to-end success.'
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

## 0. Framing: the policy projection $\Pi_\pi$ is a semantic interface

Set up the whole analysis framework up front; every later section returns to this figure. This section also stands up the piece's real **formal object** — $\Pi_\pi$, its contract-preserving property, and a measurable **Contract Information Loss**.

### 0.1 Argumentation chain

```
Structured State Contract  Ŝ_t                          (defined in 9/14)
        │
        ▼
policy projection  Π_π(Ŝ_t, o_t, ℓ_t)                   (the core object of this piece)
        │
        ▼
question: which decision-relevant semantics are preserved?
        │
        ▼
three contract-relevant invariants
   ├─ mode          multi-hypothesis structure            ──▶  primitive: mode_select
   ├─ temporal      heterogeneous staleness distribution   ──▶  primitive: age_gate
   └─ source        provenance / dependency / negative ev. ──▶  primitives: provenance / dependency / negative_evidence
        │
        ▼
training (representation-side probe + intervention-consistency)
deployment (safety filter directly reads constraint-relevant contract fields)
evaluation (CAG as aggregate + HPS / SDS / PCE as per-dimension diagnostics)
```

### 0.2 Formalizing $\Pi_\pi$: contract-preserving projection

This piece defines $\Pi_\pi$ explicitly as **a semantic interface**, not as "the first computation of the model". Fix a set of **contract-relevant decision variables** $Y_{\mathcal{C}}$ that the policy serves — the quantities downstream controller / planner / safety filter / diagnostics will read, e.g. "which object is this track", "how many newtons is this contact force", "how old is this measurement", "is this channel still valid". Define a semantic-equivalence relation on $\hat S$:

$$\hat S \sim_{\mathcal{C}} \hat S' \quad \Longleftrightarrow \quad \forall\, Y_{\mathcal{C}}\text{-relevant query},\;\hat S \text{ and } \hat S' \text{ give the same answer}.$$

**Definition (Contract-preserving projection)** — $\Pi_\pi$ is contract-preserving iff it does not irreversibly collapse semantically distinct contracts onto the same input:

$$\hat S \not\sim_{\mathcal{C}} \hat S' \quad \Longrightarrow \quad \Pi_\pi(\hat S) \not\equiv \Pi_\pi(\hat S').$$

**Definition (Contract Information Loss)** — a measurable degradation via mutual information:

$$L_{\mathcal{C}}(\Pi_\pi) \;=\; I(\hat S;\, Y_{\mathcal{C}}) \;-\; I\!\big(\Pi_\pi(\hat S);\, Y_{\mathcal{C}}\big).$$

By the data processing inequality $L_{\mathcal{C}} \ge 0$; $\Pi_\pi$ being contract-preserving corresponds to $L_{\mathcal{C}} = 0$ (or, more practically, "no decision-relevant distinction is irreversibly folded away").

This formalization matters because it **shuts the door on a very common rebuttal** — "of course a projection loses information, every neural network loses information". The correct statement is: **a policy may drop information; it may not silently drop contract semantics.** The former is a physical fact; the latter is an interface violation. The whole piece asks one question: **do mainstream policies' $\Pi_\pi$ treat "silently dropping contract semantics" as the default?**

### 0.3 Three boxed claims of this piece

> **Claim 1 (Contract semantics can be lost at the policy boundary)** — A structured estimator output does not imply a structured policy input. The projection $\Pi_\pi$ is itself **a semantic interface** and should be treated as one, not as "the first forward computation of a model".

> **Claim 2 (Contract preservation is not architecture-specific)** — Engineered-state heads, latent visuomotor policies, and VLA / flow policies lose different slices of the contract, but **the underlying failure mode is the same**: contract-relevant distinctions are projected away without an explicit preservation guarantee. The target of criticism is the interface contract, not the model architecture.

> **Claim 3 (Contract compliance should be tested by intervention, not inferred from end-to-end success)** — A policy can achieve very high task success while ignoring disagreement, staleness and provenance. Contract compliance must be measured with **controlled intervention metrics**, not back-derived from an end-to-end success rate.

## 1. Two dimensions instead of three families: conditioning representation × action head

**An earlier draft sliced "VLA / Diffusion Policy / engineered head" as three mutually-exclusive families**; that taxonomy is too coarse — π0 is VLA + flow matching and lands in both columns. This section switches to **two orthogonal dimensions**; a policy family's choice becomes a coordinate on a grid rather than a stance.

### 1.1 Two orthogonal dimensions

**Dimension 1 · Conditioning representation** — the shape in which the policy reads upstream information. Four typical values:

- **engineered state** (a fixed-length vector hand-written per task, $s_t \in \mathbb{R}^d$)
- **visual latent** (encoder output, $z_t = f_\phi(o_t)$)
- **multimodal token** (discrete tokens in a VLM vocabulary; image and language share the space)
- **structured contract** (an explicit $\hat S_t$; fields are enumerable and jointly readable by controller / policy / safety filter)

**Dimension 2 · Action head** — how the policy produces an action distribution. Five typical values:

- **deterministic regression** (a mean, no noise)
- **Gaussian / mixture stochastic head** (the SAC line, mean + variance / GMM)
- **autoregressive token** (a discrete vocabulary, RT-2 / OpenVLA)
- **diffusion** (multi-step denoising, Diffusion Policy)
- **flow matching** (velocity regression over continuous action chunks, π0)

### 1.2 Grid

| Representative policy | Conditioning representation | Action head | Where the contract is destroyed |
|---|---|---|---|
| Classical SAC/PPO head | engineered state | Gaussian / deterministic | Minimal; fields hand-written; but state design is an art and cross-task transfer is poor |
| Visual-obs PPO / DrQ | visual latent | Gaussian | Once $z_t$ is squeezed, provenance / age / hypothesis are usually already gone |
| Diffusion Policy (Chi 2023) | visual + proprio latent | diffusion | Destroys provenance / age / hypothesis; action-side multimodality is fine, but **state-side multi-hypothesis has no slot** |
| ACT / ALOHA (Zhao 2023) | visual + proprio latent | CVAE → chunked | Similar to Diffusion Policy; state-side remains implicit |
| RT-2 | multimodal token (VLM) | autoregressive action tokens | Frame / reference_point / convention are tokenized away |
| OpenVLA | multimodal token + proprio | autoregressive | Same as RT-2, proprio adds one channel, but the contract fields still have nowhere to go |
| π0 (Black 2024) | multimodal token (VLM conditioning) | **flow matching** (continuous action chunk) | Action is continuous, but conditioning still goes through VLM tokens — **the structured contract is still flattened** |

One key distinction: **Continuous actions do not imply structured state semantics.** π0's action head is a flow-matched continuous chunk and sounds "structured", but its conditioning representation is a VLM token stream, and the upstream contract's hypothesis / provenance / age are flattened at that stage. **Continuity on the action side does not rescue structural loss on the state side.** This distinction separates this piece from the intuition "flow matching is finer-grained, so contract preservation must be better".

One caveat: **this is not a ranking of "which combination is best"**. Engineered state + Gaussian is still the low-dimensional control baseline champion; multimodal token + flow matching is still the only realistic path for open-semantic settings. What this piece cares about is **for every combination, does its $\Pi_\pi$ layer carry an explicit preservation guarantee for contract-relevant semantics**. In most existing work the answer is "no" — **not because a family is inherently bad, but because this layer has never been treated as an interface design problem**.

## 2. What "State" means across policies

This section is **terminology cleanup**. "State" in the embodied-AI literature means at least five mutually-distinct things, and a policy family's choice is often implicitly "which of these five it commits to".

**$\pi_{\mathrm{obs}}$ — raw observation**: images, point clouds, force/torque streams. The **input form** of most imitation-learning pipelines, but typically not the state the policy actually uses internally (it gets encoded away).

**$z_t = f_\phi(o_{:t})$ — encoded latent**: the low-dimensional representation post-encoder. Diffusion Policy's and VLA's image tokens, and RSSM states in world models, all fall here. The contract has already **undergone one projection** at this stage; observability and provenance are usually already lost.

**$b_t$ — belief / posterior**: the explicit belief state of the POMDP line. The "multi-hypothesis + posterior weight" structure in the contract maps most cleanly here. But mainstream VLA / Diffusion Policy do not model belief explicitly — it is only implicitly approximated by the encoder.

**$s_t = (b_t, \pi_t, q_t, h_t)$ — the allocation state of 9/10 Part 1**: belief + policy + budget + hardware. This is the state at the **decision layer**, not the state at the policy's input. It requires that the contract, in addition to observations, expose "which policy is currently active and how much real-rollout budget remains" — fields for which VLAs currently have no interface at all.

**$\hat S_t$ — structured state (contract)**: the contract object defined in 9/14.

A full chain:

$$\pi_{\mathrm{obs}} \;\xrightarrow{\;\text{encoder}\;}\; z_t \;\xrightarrow{\;\text{abstraction}\;}\; \hat S_t \;\xrightarrow{\;\text{posterior}\;}\; b_t \;\xrightarrow{\;\text{allocation}\;}\; s_t$$

Back to the language of §0.2: the real policy-side question is **not** "can I consume longer token sequences", it is **"how far along $\pi_{\mathrm{obs}} \to z_t \to \hat S_t \to b_t \to s_t$ am I willing to commit, and do I preserve $L_{\mathcal{C}} = 0$ for $Y_{\mathcal{C}}$ — or at least declare what I am dropping"**. Engineered state stops at $\hat S_t$, visual latent stops at $z_t$, multimodal token effectively stops at a tokenizer one layer past $\pi_{\mathrm{obs}}$ — **three stopping points correspond to three values of $L_{\mathcal{C}}$, not to "one smart, one dumb"**.

**The common misuse of the phrase "multimodal fusion" on the policy side** is treating the cross-attention in $\pi_{\mathrm{obs}} \to z_t$ as if it were "already doing multimodal state estimation". It is not. Real state abstraction requires the fields inside $\hat S_t$ to **carry consistent semantics across sensor families and be jointly readable by four consumers — controller / policy / world model / diagnostics** — 9/14 §6 has already established this convention; what this section adds is **asking from the policy side once more: has $L_{\mathcal{C}}$ been silently accepted?**.

## 3. Four interface-mismatch failure modes

Once the §1 coordinates are instantiated in a concrete policy, contract semantics show up as **four concrete failure modes**. None of the four are theoretical worries; they are **what actually breaks in deployment**. All four are **facts about the interface**, not personality defects of any single family — as we will see below.

### 3.1 Failure 1: Multi-hypothesis silently collapsed

Inside the contract a single physical quantity may have multiple hypotheses ("does this track_id refer to the same object", "is this contact on the pad or the edge") with different posterior weights. **If a policy's $\Pi_\pi$ performs an explicit posterior mean / mean-pool on these hypotheses, mode collapse is deterministic**. The issue is not "does this model collapse"; the issue is **"does this model provide an explicit hypothesis-preserving readout"**. Most existing policy architectures — the visual-latent and multimodal-token lines included — **do not provide such a readout slot**, which makes collapse an **interface default** rather than a training-dynamics accident.

An important refinement: **action-side multimodality in diffusion / flow-matching does not automatically mean state-side multi-hypothesis is preserved**. The action distribution can be multimodal (denoising yields multiple action trajectories), but if the hypothesis structure of $\hat S_t$ has already been folded by the encoder, action-side multimodality is just sampling on an input that has already lost upstream distinctions. **The two kinds of multimodality are not the same thing and cannot guarantee each other**.

A precise rephrasing: **this piece does not criticize Diffusion Policy for collapsing; it criticizes the fact that no architecture explicitly commits to hypothesis preservation**.

### 3.2 Failure 2: No interface-level guarantee of semantic correctness

Frame fields inside the contract (`orientation_frame: "tool_flange"`, `reference_point: "contact_center"`, `convention: "right-handed"`) enter the policy and, if tokenized into a token sequence — **note, the issue here is not that "transformers cannot learn frame transforms"**. In principle a transformer can learn a hard constraint like $\tau_{p_2} = \tau_{p_1} + (p_1 - p_2) \times f$ through attention + MLP. **The real problem is that tokenization provides no interface-level guarantee that this transformation is interpreted correctly**. The model **can** learn it and **can also** fail to learn it, depending on whether the training distribution covers frame-contrast pairs, whether the architecture has a friendly inductive bias, and whether an auxiliary constraint explicitly imposes it. **Absent these, semantic correctness is not an interface property, only an accident of training data**.

**A concrete shape**: in a real-robot deployment we change the wrench's `reference_point` from `sensor_flange` to `contact_center` — the preprocessing pipeline changes, the token sequence barely changes (`"sensor_flange"` and `"contact_center"` both show up as frame tokens in similar contexts), yet $\tau$ shifts substantially per the transport theorem. A VLA trained on the two "look-alike" datasets separately **may converge separately to similar-but-wrong policies** — **"both wrong" is not because transformers cannot learn frame transforms, it is because the training data has too few frame-contrast pairs and no constraint forces it to learn**. This class of bug only surfaces **post-deployment**, on cross-dataset / cross-embodiment reuse.

**The key rephrasing**: tokenization does not erase semantics — it **demotes semantics from an interface guarantee to a training-time experience**. This is where this piece parts ways with the intuition "tokenizers just don't work".

### 3.3 Failure 3: Temporal alignment broken by concat

The contract makes per-channel `age` explicit (e.g. proprio 1 ms, F/T 5 ms, vision 100 ms, tactile 30 ms). If the policy only sees a concatenated vector, the $\Delta t$ distribution is lost. Symptom: the policy decides using vision's "world 100 ms ago" and tactile's "contact 30 ms ago" as if they were synchronous. 9/14 §4 already stressed **timestamp sync ≠ causal sync**, and even less so decision-time causal consistency. **Concat hides $\Delta t$; it does not hide the semantics of $\Delta t$** — even if you stamp every channel, if the policy does not read those stamps into its decision variables, the stamps are decorations.

### 3.4 Failure 4: Safety-blindness to validity vs staleness

The contract explicitly separates availability (does the channel have data today), validity (is the data valid, e.g. is calibration current), and age (how old). A policy that reads only the numbers conflates "stale but valid" with "missing but valid", and treats "invalid after calibration drift" the same as "sensor disconnected". 9/14 §8.5 already broke the degradation chain apart — **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption** — if the policy does not ingest this chain at the input, both training-time augmentation and inference-time guardrail will latch onto the wrong place.

All four failures are **interface problems**; swapping backbones does not solve them; only making the $\Pi_\pi$ layer explicit does. This matters because it **decouples** the decision "should we switch to a bigger VLA" from the decision "should we rewrite the policy-side interface".

## 4. Three families of contract-read primitives

Following the four failures in §3, the policy side needs **three families of primitives** — not bigger transformers, not more data. The third family (source structure) itself splits into three, see §4.3.

### 4.1 `mode_select` (the readout of the hypothesis layer)

Faced with a multi-hypothesis posterior inside the contract, the policy must pick one of several **explicit readouts**: MAP (argmax of posterior weight), sample (draw from the posterior, encouraging exploration), expected-mixture (keep the mixture, let downstream heads attend), or ECE-preserving top-$k$ (retain $k$ calibrated hypotheses).

$$\text{read}\!\big(\{(\mu_i, \Sigma_i, w_i)\}_{i=1}^{K}\big) \;=\; \left\{\begin{aligned}
&\mu_{\arg\max_i w_i} && \text{(MAP, drop low-weight hypotheses)}\\
&\textstyle\sum_i w_i\, \mu_i && \text{(posterior mean, an explicitly declared collapse)}\\
&\mu_i + L_i \epsilon,\;\; i \sim w,\;\; \epsilon \sim \mathcal{N}(0, I) && \text{(posterior sample)}\\
&\big\{(\mu_i, \Sigma_i, w_i)\big\}_{i \in \mathrm{top}\text{-}k} && \text{(ECE-preserving top-}k\text{)}
\end{aligned}\right.$$

The point is **not "mean is forbidden"** — mean is a perfectly legitimate readout, provided the collapse is **explicitly declared**. The real failure mode is "the interface provides no readout slot for hypothesis structure, the policy can only implicitly merge via concat + MLP, and mean becomes the default". This distinction is critical: **this piece is against undeclared default collapse, not against collapse per se**. ECE-preserving top-$k$ is valuable because it hands downstream an explicitly ablatable hypothesis structure, not because mean is inherently wrong.

### 4.2 `age_gate`: parallel measurement / uncertainty / age / validity / trust — not multiplicative decay

**A common interface-design bug** is to multiply staleness trust directly into the measurement: $x_c^\pi = \tau_c(a_c) \cdot \mu_c$. This **changes the physical value of the observation** — 10 N read at 100 ms age gets multiplied into "3 N", and the "3 N" at the policy input **looks** like "a 3 N force", not "a 10 N force whose trust has decayed". This directly violates the very distinction the contract wants to preserve: **$(F = 3\,\mathrm{N},\, a = 0)$ and $(F = 10\,\mathrm{N},\, a = 100\,\mathrm{ms})$ are two different semantic events**.

The right thing is to put them in **parallel** on the policy input and not multiply:

$$x_c^{\pi} \;=\; \big[\;\underbrace{\mu_c}_{\text{measurement}}\;,\;\underbrace{\Sigma_c}_{\text{uncertainty}}\;,\;\underbrace{a_c}_{\text{age}}\;,\;\underbrace{v_c}_{\text{validity}}\;,\;\underbrace{q_c}_{\text{trust} \,=\, \tau_c(a_c)}\;\big].$$

Trust $q_c$ is a **meta field**; it does not scale $\mu_c$, it is a conditioning variable the policy may choose to consume. The concrete shape of $\tau_c(a_c)$ (exponential / sigmoid / step) is not important; what matters is that it exists as a **separate channel**. If a downstream gate really is required, **it should gate on uncertainty, not on measurement**:

$$\tilde\Sigma_c \;=\; \Sigma_c \,/\, \tau_c(a_c) \quad \text{(stale ⇒ effective uncertainty is inflated)}$$

This intuition is closer to process-noise inflation for delayed updates in a Bayesian filter than to "discounting the reading". Availability and $\mathbb{1}[v_c]$ (validity) still **hang on as parallel slots** — they must not be folded into $\mu$, nor into $\tau$:

$$\text{availability}_c,\;\;\mathbb{1}[v_c],\;\;a_c,\;\;\mu_c,\;\;\Sigma_c,\;\;q_c \quad \text{—— six slots, each says its own thing.}$$

This looks like a small change but it semantically repairs `age_gate` from "discounting a measurement" to "measurement + facts about the measurement" — a direct instantiation of the §0.2 $L_{\mathcal{C}}$ definition: mixing a measurement with **facts about** the measurement into a single scalar is a direct source of $L_{\mathcal{C}}$.

### 4.3 `provenance / dependency / negative_evidence`: three things, not one bucket

9/14 §6.1 groups `contributing_mask`, `correlated_with` and `negative_evidence` under `provenance` — from the estimator side that makes sense (all three are "source structure of this field"), but from the **policy-side readout** they sit at different semantic levels:

| Field | Essence | How policy reads it |
|---|---|---|
| `contributing_mask` | evidence source (which sensors contributed) | concatenated into policy input as conditioning (**provenance_harden**) |
| `correlated_with` | dependency structure (which fields are structurally correlated) | attention mask / bias, to avoid double-counting (**dependency_gate**) |
| `negative_evidence` | hypothesis-conditioned absence of expected evidence, $P(\mathcal{E}^- \mid H)$ | closer to a **first-class citizen of belief update**, not a provenance-bucket item (**negative_evidence_read**) |

**`negative_evidence` is not provenance**. It is "which sensors should have seen $E^-$ but did not" — semantically it is a likelihood term, more closely related to belief update than to source. Filing it under provenance turns §4.3 into an oversized bucket with fuzzy boundaries. The three sub-primitives each address a distinct layer:

**`provenance_harden`**: concatenate `contributing_mask` directly into the policy input. Engineered-state heads and visual-latent heads both support this (a few more dimensions of vector).

**`dependency_gate`**: turn `correlated_with` into an attention mask / bias so the transformer does not double-attend to correlated evidence. **One possible implementation** (note: one possible) is an additive attention bias:

$$\mathrm{Attn}'_{ij} \;=\; \mathrm{Attn}_{ij} \;-\; \beta \cdot \mathbb{1}\!\big[\text{fields}_i \text{ correlated\_with } \text{fields}_j\big].$$

But it must be honestly acknowledged that **`correlated_with` is a field-level semantic relation, whereas attention bias is a token-pair relation** — the two require an $R_{\text{field}} \to R_{\text{token}}$ mapping in between. A single field is often decomposed into value token / uncertainty token / age token / provenance token, and **which token pairs need suppression is not a solved problem — it is a compilation problem from a provenance graph to an attention graph and deserves to be treated as an independent research direction**. This piece only stands the primitive up; the formula above is "one possible realization", not canonical.

**`negative_evidence_read`**: expose "what should have been seen but was not" as a likelihood-side conditioning that flows into either the policy or the belief update. Concretely, hand the policy a $P_{\mathrm{expected}}(\mathcal{E}^- \mid H_k)$ vs $P_{\mathrm{observed}}(\mathcal{E}^-)$ gap so it can attend to "which hypotheses are weakening due to missing evidence". This primitive is the thinnest on engineering maturity but often the strongest in effect on hypothesis ranking.

The three sub-primitives together — **provenance says "where it came from", dependency says "cannot count twice", negative evidence says "what was not seen"** — support the source-structure invariant of §0.2.

### 4.4 Mapping primitives to invariants

Back to the §0.1 figure: **mode / temporal / source** correspond to **mode_select / age_gate / (provenance_harden + dependency_gate + negative_evidence_read)**. Miss any one invariant, at least two of the four §3 failures recur. Three families are also **not a complete interface design** — observability / identifiability, frame convention, and contact set each have their own more specialized readouts (9/14 §7, §8.6); this piece only handles **the three most easily destroyed silently at the $\Pi_\pi$ layer**.

## 5. Training-time and deployment-time knock-on effects

Once the §4 primitives are on the input side, four downstream things must be adjusted — training objective, augmentation, safety filter, evaluation. **Interfaces are not free** — but the changes are **local and bounded**.

### 5.1 Two classes of training constraint: representation-side probe and intervention-consistency

A natural mistake is: **to make the policy "use the contract", require it to output validity / hypothesis predictions** — this actually sneaks "use the contract" into "copy the contract", which is the wrong direction. **A policy is entirely allowed to only eat the contract and never spit it back out**; auxiliary prediction heads are not a necessary condition.

This piece proposes two cleaner classes of training constraint instead.

**Class A · Representation-side probe** (not a hard constraint, a diagnostic). Hang a few probe heads off the policy's intermediate representation $z^{\pi}$ and try to predict the contract's `age`, `validity`, `observability` and hypothesis posterior from $z^{\pi}$. **These probes do not enter the main loss; they are only used for measurement**: good probe scores mean the policy preserves this information internally, poor scores mean the contract has already been flattened inside $\Pi_\pi$. $L_{\mathrm{probe}}$ may be added to the main loss with a small weight as regularization, but **its diagnostic value exceeds its training value**.

$$\mathcal{L}_{\mathrm{total}} \;=\; \mathcal{L}_{\mathrm{action}} \;+\; \underbrace{\alpha\, \mathcal{L}_{\mathrm{probe}}}_{\text{weak regularization, mainly diagnostic}} \;+\; \underbrace{\sum_{\mathcal{C}} \gamma_{\mathcal{C}}\, \mathcal{L}^{\mathcal{C}}_{\mathrm{consistency}}}_{\text{class B, see below}}$$

**Class B · Intervention-consistency constraint** (this is the real core). Apply a **known** transformation $T_{\mathcal{C}}$ to the contract and require the policy output to respond with the **expected pattern**:

$$\mathcal{L}^{\mathcal{C}}_{\mathrm{consistency}} \;=\; D\!\Big(\pi_\theta\!\big(\hat S,\, o,\, \ell\big),\;\; \pi_\theta\!\big(T_{\mathcal{C}}(\hat S),\, o,\, \ell\big);\;\rho_{\mathcal{C}}\Big)$$

$D(\cdot, \cdot; \rho_{\mathcal{C}})$ is a specific divergence or penalty $\rho_{\mathcal{C}}$ tailored to the transformation $\mathcal{C}$ — it does not require the policy to output the "same thing", it requires it to vary **in a known pattern**. Four typical $T_{\mathcal{C}}$:

- **Frame transform**: move `reference_point` from A to B; $\tau$ transforms per the transport theorem. **The action side must correspondingly undergo the equivalent coordinate transformation** (equivariance, not invariance).
- **Age increase**: raise one channel's `age` from 5 ms to 200 ms, leave measurement unchanged. **The output distribution must shift in the direction of uncertainty inflation** (e.g. variance rises, or actions become more conservative); mean should not move.
- **Hypothesis reweighting**: hold the hypothesis set constant, change only the posterior weights. **The output distribution must respond predictably along the weight-shift direction**; top-1 need not flip, but the distribution must respond.
- **Provenance removal**: drop a contributing sensor from `contributing_mask`. **The output confidence associated with that sensor must fall**; it cannot stay as before.

**None of these four require the policy to predict anything explicitly**; they require **the response function to conform to contract semantics**. This is the **training-side counterpart** of §0.2's "$\Pi_\pi$ is contract-preserving". Compared to "adding a few auxiliary prediction heads", this class of constraint fits the piece's thesis better and is more research-flavored: **what we propose is not for the policy to copy the contract, but for the policy to respond correctly under contract transformations**.

### 5.2 Augmentation must be generated along the degradation chain

9/14 §8.5 emphasized **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption** — six degradations with different **causal origins** and different **downstream readouts**. If training-time augmentation uses only one (most often random masking), the policy learns them all as the same thing and cannot distinguish "this channel is broken today" from "this channel is high-latency" at deployment. The aug generator must **separate per chain**, sample independent distributions per class, and **each class must correspond to one of §4.2's six slots** (measurement / uncertainty / age / validity / availability / trust) as the one being activated.

Concretely: label every training episode with a degradation class, carry that label explicitly on the policy input, and let the aug pipeline sample conditioned on the same label. **This is not curriculum, it is conditioning** — curriculum is "learn easy first, hard later", conditioning is "let the policy know which degradation it is currently under". §5.1 Class B intervention consistency and degradation-conditioned augmentation pair naturally — **aug generates the $T_{\mathcal{C}}$ samples, loss measures whether the policy's response to $T_{\mathcal{C}}$ matches $\rho_{\mathcal{C}}$**.

### 5.3 Safety filter and its interface to the contract

The constraint layer (CBF / shield / runtime verifier) **must read $a_{\mathrm{proposed}}$**, otherwise what is it filtering — that point is not up for negotiation. But the more precise claim of this piece is: **the safety filter should not treat the policy's confidence or latent belief as the sole evidence that a constraint holds; it should directly access constraint-relevant contract fields**.

Concretely, in addition to $a_{\mathrm{proposed}}$, the safety filter should also read: `observability` (is the quantity that this constraint depends on observable right now), `validity` (is calibration current — expired readings must not fire constraints, nor should they let a constraint assume "all is fine"), `negative_evidence` (observations that should be present but are not, e.g. "the radar swept this angle and saw nothing" — this maps to §4.3 `negative_evidence_read`). 9/14 §7 established that **track_id is a hypothesis** — the safety filter must not simply trust track_id matching, it must also check whether the hypothesis posterior is stable; whether two tracks merge or split directly determines the credibility of "how far is that obstacle".

**If these three do not enter the safety filter, the filter will infer constraint validity from the policy's belief** — which is especially dangerous in low-observability regions. The policy's belief is optimistic precisely because it cannot see the contract's observability / validity / negative evidence, and **if the filter also cannot see them, the two go blind together**.

### 5.4 Echo with 9/10 Part 3 evaluation

Among the three sim-utility dimensions (prediction / ranking / decision), policy-side evaluation is mainly about **decision** — but with an added **contract-preservation dimension**: how much the policy degrades when the contract is torn apart is a **lower bound** on how much it depends on the contract. This dimension maps to the CAG metric of §6.

## 6. Evaluation: one aggregate + three per-dimension diagnostics

Aligned with the §4 primitives: an **aggregate metric CAG** directly answers "does the policy actually use the contract", and three **per-dimension diagnostics HPS / SDS / PCE** correspond to the mode / temporal / source invariants respectively. All four are **intervention-based evaluation** — the direct instantiation of Claim 3.

### 6.1 Aggregate: Contract Ablation Gap (CAG)

**Definition** — given a specific collapse operator $\mathrm{collapse}_X$ on the contract (which silently folds layer $X$ of contract structure):

$$\mathrm{CAG}_X \;=\; J\!\big(\pi_\theta \,\big|\, \hat S\big) \;-\; J\!\big(\pi_\theta \,\big|\, \mathrm{collapse}_X(\hat S)\big),$$

where $J$ is a higher-is-better decision utility (task success rate, or $-\text{cost}$). Four collapses each correspond to one invariant:

- $\mathrm{collapse}_{\mathrm{hyp}}$: fold the hypothesis set into a single Gaussian or a point estimate.
- $\mathrm{collapse}_{\mathrm{age}}$: flatten every channel's `age` to 0.
- $\mathrm{collapse}_{\mathrm{prov}}$: drop `contributing_mask` and `correlated_with`, let attention attend indiscriminately.
- $\mathrm{collapse}_{\mathrm{neg}}$: drop `negative_evidence`.

**High CAG = the policy really uses the contract; CAG ≈ 0 = the policy is indifferent to contract structure, no matter how high its end-to-end success rate**. This **directly rebuts the common evaluation reflex of "just look at end-to-end success"**. A policy can score high task success while treating every piece of contract structure as decoration. CAG is the core metric of Claim 3 "intervention-based evaluation".

### 6.2 Per-dimension diagnostic 1: Hypothesis Preservation Score (HPS)

At a given instant the contract contains $K$ hypotheses; HPS measures how many "locally-optimal action clusters" the policy output covers across the worlds in which each $H_k$ is true.

$$\mathrm{HPS} \;=\; \frac{1}{N} \sum_{n=1}^{N}\, \max_{k}\, \Pr\!\big[\pi_\theta(x_t^{\pi}) \in \mathcal{A}^{*}_k \,\big|\, H_k \text{ is true at } n\big]$$

**$\mathcal{A}^{*}_k$ is not required to be online available** — in benchmarks it is constructed from **privileged simulator state, oracle planners, or offline expert rollouts**; therefore HPS is a training / evaluation metric, **not a deployment-time observable**. This has to be said explicitly, otherwise HPS is immediately flagged as non-operational. The HPS gap between MAP readout and posterior-sample readout quantifies the severity of §3.1 Failure 1.

### 6.3 Per-dimension diagnostic 2: Staleness Response Curve (SDS)

**An earlier draft defined SDS as a single KL, $D_{\mathrm{KL}}(\pi(\cdot \mid a^{\mathrm{fresh}}) \| \pi(\cdot \mid a^{\mathrm{stale}}))$ — this definition is incomplete**. SDS ≈ 0 does not prove the policy is not reading age (it may be reading age and correctly concluding staleness is irrelevant to this action); SDS ≫ 0 does not prove the change came from age (a stale image may simultaneously alter visual content, so attribution fails).

The correct SDS is a **response curve under controlled intervention**: fix observation content, intervene only on $a_c$, and measure whether the **pattern of response** matches $\rho_{\mathcal{C}}$ from §5.1 Class B (variance monotonically rising, mean held stable):

$$\mathrm{SDS}_c(a_1, a_2) \;=\; D\!\Big(\pi_\theta\!\big(\cdot \,\big|\, \mathrm{do}(a_c = a_1),\, o\big),\;\; \pi_\theta\!\big(\cdot \,\big|\, \mathrm{do}(a_c = a_2),\, o\big)\Big)$$

and in derivative form:

$$\left.\frac{\partial\, \mathbb{E}\!\big[\pi_\theta(\cdot \mid \mathrm{do}(a_c = a),\, o)\big]}{\partial a_c}\right|_{a}\quad\text{compared to the same-order derivative of an oracle policy.}$$

**Staleness Response Curve** is the whole family of these quantities, not a single scalar. The three properties of the curve — its **shape** (variance monotonically rising vs falling vs flat), its **direction** (is mean held constant), and its **scale** (at how many ms it starts responding) — together form the full portrait of how much the policy is reading `age`. Flat is not automatically bad; you must compare against the oracle's expected response curve.

### 6.4 Per-dimension diagnostic 3: Provenance-Conditioning Effect (PCE)

Remove `correlated_with` from the policy input and measure how much performance drops on double-counting-sensitive scenarios (e.g. proprio + F/T fused $\hat F_{\mathrm{ext}}$):

$$\Delta J_{\mathrm{prov}} \;=\; J\!\big(\pi_\theta \mid \text{provenance}\big) \;-\; J\!\big(\pi_\theta \mid \text{provenance} = \varnothing\big),$$

with $J$ always higher-is-better decision utility (success rate, $-\text{cost}$, $-\text{ECE}$, etc.). Written as $\Delta J_{\mathrm{prov}}$ rather than "PCE" to avoid the ambiguity of naming a delta as a metric. $\Delta J_{\mathrm{prov}} > 0$ = the policy really uses provenance; $\Delta J_{\mathrm{prov}} \approx 0$ = it treats provenance as decoration; $\Delta J_{\mathrm{prov}} < 0$ = conditioning actively hurts, usually indicating that conditioning conflicts with the backbone's inductive bias and deserves separate debugging. $\Delta J_{\mathrm{prov}}$ also tells you whether the §4.3 attention-bias strength $\beta$ is tuned correctly.

The four together form an **intervention-based evaluation panel**: **CAG is the aggregate "does the policy use the contract", and HPS / SDS / PCE are per-dimension attribution to the mode / temporal / source invariants respectively**. None of them replaces an end-to-end success rate — they measure the policy-side **read-completeness** of the contract, not the policy's **expressive power**. This is exactly aligned with §0.3 Claim 3: **contract compliance must be tested by intervention**.

## 7. A minimal executable interface sketch

Combine the §4 primitives and §6 metrics into a Python class skeleton. **Not a concrete policy; a readable interface contract**.

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
        All five knobs are policy-specific — the same contract should be projected
        differently by a VLA and by a diffusion policy; the default for staleness
        is parallel_field and never a multiplicative discount on measurement.
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

Three caveats written in stone:

- **(i)** This is not the only way to read. All five knobs are policy-specific — the same contract should be projected with different values by a VLA and a Diffusion Policy, and the optimal `staleness` setting differs between engineered-state heads and visual-latent heads.
- **(ii)** This interface **only addresses the input side**. The two classes of training constraint in §5.1, degradation-conditioned augmentation in §5.2, and the safety-filter direct wiring in §5.3 — if any one of these is left unchanged, the $\Pi_\pi$ layer will be routed around by training dynamics no matter how beautifully it is written (the loss will find the cheapest "flatten the contract" path on its own).
- **(iii)** `dependency="attention_bias"` only has an implementation path on transformer-family backbones, and **field-graph → token-graph compilation** (`compile_field_graph_to_token_graph`) is a function name in this piece, not a canonical implementation — it is a real open problem; MLP heads use the `learned` path and let the network learn a bias matrix from `correlated_with`.

## 8. Three closing claims (aligned with §0.3)

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. The projection $\Pi_\pi$ is itself **a semantic interface** and should be treated as one, not as "the first forward computation of a model".

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state heads, latent visuomotor policies, and VLA / flow policies lose different slices of the contract, but **the underlying failure mode is the same** — contract-relevant distinctions are projected away without an explicit preservation guarantee. The target of criticism is the interface contract, not the model architecture; π0 is VLA + flow matching, Diffusion Policy is visual-latent + diffusion — slicing along two orthogonal dimensions is closer to reality than three families, and is less easily misled by the intuition "some family is inherently better".

> **Claim 3 · Contract compliance should be tested by intervention, not inferred from end-to-end success.** A policy can achieve very high task success while ignoring disagreement, staleness, and provenance. Contract compliance must be measured with **controlled intervention metrics** — CAG as the aggregate answer, HPS / SDS / PCE as per-dimension attribution, and §5.1 Class B intervention-consistency loss as the training-side counterpart — not back-derived from an end-to-end success rate.

A final line: **try to stop using the word "fusion"** — on the time axis it asks "when to merge", on the semantic axis it asks "merge into what"; together, 9/14 and this piece split the second question into **what upstream delivers + what downstream reads**. The first question (when) is largely answered by §1's two-dimensional grid — **timing is a consequence of the interface, not a decision variable of the interface**. Stand that line up, and this series has been worth it.

## Sources

All arXiv IDs below have been verified online; journal-only citations do not carry an arXiv link. Grouped by the sections they support.

### A · VLA family (supports §1 grid, §3 Failures 1–2, §5.1 Class B)

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) (actions expressed as text tokens jointly fine-tuned with a VLM · a canonical shape for §3 Failure 2 "no interface-level guarantee of semantic correctness")
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) (open-source VLA baseline; supported inputs include multi-camera / depth / proprioceptive state encoding; "supports input" ≠ "how far along the contract chain the policy reads")
- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) (VLM backbone + proprio token + noisy action chunk + flow matching · the direct source of §1.2's **"Continuous actions do not imply structured state semantics"**)
- Octo Model Team, *Octo: An Open-Source Generalist Robot Policy*, RSS 2024 · [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) (transformer-based readout · a reference point for the §4.3 dependency_gate attention_bias path)

### B · Diffusion / Flow-Matching Policy (supports §1 grid, §5.1)

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 · [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) (RGB stack + proprio concat + denoising · field evidence for §3.1 "action-side multimodality does not guarantee state-side hypothesis preservation")
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT / ALOHA), RSS 2023 · [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) (CVAE + transformer encoder-decoder, chunked action · a mid-point between families B and C)
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 · [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) (velocity regression over continuous action chunks · one concrete form of §5.1 $\mathcal{L}_{\mathrm{action}}$)

### C · Uncertainty, calibration, and belief-space references (support §4.1, §5.1, §6.1 CAG)

- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) (modern networks are over-confident; temperature scaling origin · motivation for the calibration dimension of §5.1 $\mathcal{L}_{\mathrm{probe}}$)
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels* (PlaNet / RSSM), ICML 2019 · [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) (deterministic + stochastic latent, an engineering instantiation of multi-hypothesis · one reference for §4.1 readout)
- Hafner et al., *Mastering Diverse Control Tasks through World Models* (DreamerV3), Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) (discrete + continuous mixed latent, KL balancing · an adjacent path for §4.1 ECE-preserving top-$k$)

### D · SAC / PPO and engineered-state baseline (supports the first row of §1 grid)

- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, ICML 2018 · [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) (Gaussian NLL / max-entropy policy loss · the engineered-state-head shape of §5.1 $\mathcal{L}_{\mathrm{action}}$)

### E · Continuations of the series (how this piece plugs into 9/13, 9/14, Sim-to-Real P1/P3)

- This blog, *Splicing is not seeing: what robot multimodal fusion is missing is an interface, not a model* · `/en/articles/2026-09-14-multimodal-fusion-interface/` (Structured State Contract, Interface Property Benchmark, degradation chain · the §0.2 $L_{\mathcal{C}}$ and §3–§6 build directly on it)
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
