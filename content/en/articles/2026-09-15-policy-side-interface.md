---
title: 'Policy-Side Interface (Part 1): After the Contract Stands, What Do VLA / Diffusion Policy / π0 Actually Consume?'
slug: "2026-09-15-policy-side-interface"
date: 2026-09-15
draft: false
categories: ["Embodied AI", "Policy Learning"]
tags: ["Embodied AI", "Policy Learning", "VLA", "Diffusion Policy", "π0", "RT-2", "OpenVLA", "Action Tokenization", "Structured State Contract", "Consumer Contract", "Consumer Contract Triple", "Declared Quotient", "Query Family", "Query Subsumption", "Schema Compatibility", "Semantic Preservation", "Decision-Relevant Preservation", "Decision Sufficiency", "Conditional Mutual Information", "Residual Contract Information", "Declared Coverage Loss", "Projection Residual Loss", "Decision Collapse Rate", "Safety Obligation", "Contract-Read Primitives", "Measurable Decoder", "Separately Auditable Failure Sites", "Contract Consumer"]
description: 'This piece is the **upper half / framework** of the policy-side interface discussion. The lower half (evaluation protocol + training-time knock-ons + Python skeleton) is delivered in 9/16 [Policy-Side Evaluation (Part 2)](/en/articles/2026-09-16-policy-side-evaluation/). The multimodal-fusion piece stood up the upstream deliverable as a Structured State Contract. This piece asks the dual question: if the estimator really delivers per contract, can the policy side actually consume it. Core boxed inequality: **Structured estimator output ≠ structured policy input**, over the full pipeline $\mathcal C\to(\mathcal C_\pi,Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}\to\mathcal B_{\mathcal C}$. v6 turns the Consumer Contract into a real software interface by splitting it into a triple $\mathcal C_\pi=(Q_\pi,\mathcal O_\pi,V_\pi)$ — Queries / Obligations / Versions. §0.2.1 closes two dangling objects that v5 left hanging: the **decision-relevant observable** $Y_{\mathcal C}^{\pi}=\{q(\hat S):q\in Q_{\mathcal C}^{\mathrm{req}},q\text{ influences consumer decision}\}$ (pinned by the Consumer Contract, not an arbitrary latent in the benchmark) and the **representation equivalence** $Z_\pi(\hat S)\sim_Z Z_\pi(\hat S^{\prime})$ iff no measurable decoder $h_\pi(Z_\pi,O,L)$ distinguishes the two on $Y_{\mathcal C}^{\pi}$ — turning the dangling $\equiv$ in Properties A / A-prime into a conditional-MI-wireable object. In §0.2.2, $L_{\mathrm{declared}}$ is upgraded from literal set membership to **query subsumption coverage** $L_{\mathrm{declared}}=\sum w_q\mathbf 1[\nexists q^{\prime}\in Q_{\mathcal C_\pi}:q^{\prime}\succeq q]$ — "what may be dropped" becomes a semantic capability lattice; $L_{\mathrm{projection}}=I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$ uses $Y_{\mathcal C}^{\pi}$, $L_{\mathrm{decision}}$ uses $D_{\mathcal A}$ (action-equivalence-aware distance), and the fourth slot $\mathcal O_{\mathrm{safety}}$ is an obligation, not a loss. The three losses are **three separately auditable failure sites**, **not three statistically independent losses** — sequentially coupled through $q_\pi\to\Pi_\pi\to\pi_\theta$. §1 replaces the three-families classification with **two orthogonal dimensions** — conditioning representation / semantic interface × action head — and gives a §1.2 grid. §2 pins what "State" means across policy families. §3 lists **four interface-mismatch failure modes** — multi-hypothesis silently collapsed, no interface-level guarantee of semantic correctness, temporal alignment broken by concat, safety-blindness to validity vs staleness. §4 defines **three families of contract-read primitives** — `mode_select`, `age_gate`, and the three-way split of `provenance / dependency / negative evidence`. §5 bridges to the lower half: this upper half stops at Claim 1 (policy is a contract consumer, not merely a function approximator) and Claim 2 (architecture-agnostic four questions); Claim 3 (compliance is measured by controlled intervention, four evidence types, five-layer hierarchy) plus the training-side knock-ons, evaluation protocol, and Python skeleton — all delivered in 9/16. VLA / Diffusion / Flow / ACT / SAC / PPO demote from theoretical classifications to implementation coordinates.'
toc: true
related_articles:
  - 2026-09-16-policy-side-evaluation
  - 2026-09-14-multimodal-fusion-interface
  - 2026-09-13-tactile-force-sensing
  - 2026-09-07-vla-world-models
  - 2026-09-05-vla-pi-family
  - 2026-09-03-vla-deep-dive
---


> Picks up from [Splicing is not seeing: what robot multimodal fusion is missing is an interface, not a model](/en/articles/2026-09-14-multimodal-fusion-interface/): that piece reframed multimodal fusion from a "when to fuse" question to a "what is delivered after fusion" question, and stood the deliverable up as a contract object — the **Structured State Contract** — a structured state carrying hypothesis, provenance, observability, availability / validity / age, contact set and negative evidence. It closed on the claim **A good multimodal system must represent disagreement, not merely resolve it.** This piece asks its **dual**: if the upstream really delivers per contract, **can the policy side actually consume it**.

Short answer: **a structured estimator output does not imply a structured policy input**. Between them sits an explicit projection $\Pi_\pi$ which is itself **a semantic interface that can silently destroy semantics**. VLA uses a tokenizer, Diffusion Policy uses image-encoder + proprio concat, classical heads use a hand-designed state vector — the three projections each destroy a different slice of the contract, and **the destruction is silent**: the loss curve still goes down, eval scores still go up, and you cannot see from the training log what has been flattened away. This is not a model-size problem, **it is an interface problem** — but the problem is not "which architecture collapses", **the problem is that no architecture carries an explicit preservation guarantee for contract-relevant semantics**. This distinction is what separates this piece from the usual "architecture critique" genre.

This piece does not push a specific backbone, does not oppose end-to-end learning, and does not claim that one family is inherently better than another. Its thesis is more general and more verifiable: **every policy architecture has to answer — does its input projection preserve the decision-relevant semantics of the upstream contract?** Once that question is on the table, "VLA versus Diffusion Policy" demotes itself from a position to a design decision.

## 0. Framing: $\mathcal C\to(\mathcal C_\pi,Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}$ is an auditable semantic pipeline

Set up the whole analysis framework up front; every later section returns to this figure. This section stands up the piece's real **formal objects** — the **three-layer quotient-declaration chain** $\mathcal C\to\mathcal C_\pi\to{\sim_\pi}\to q_\pi$, a **schema-compatibility rule**, **three tiers of preservation** (full / decision-relevant / decision sufficiency), the **conditional-MI residual contract information** $L_{\mathcal C}^{\pi}$, and the **three semantic losses + one safety obligation** framing (this piece no longer speaks of "three losses").

### 0.1 Argumentation chain

```
Structured State Contract  Ŝ_t                                (defined in 9/14)
        │
        ▼
upstream contract  C                                          (full semantic schema, delivered by the estimator)
        │  declare + version check
        ▼
consumer contract  C_π  ⊆ C                                    (policy declares "which slice of C I consume"; carries supported_version)
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
safety layer  g_safety :  (a, evidence_j)  ↦  a_applied         (the fourth slot: not an information loss, an obligation)
        │
        ▼
three semantic losses + one safety obligation, each audited separately
   ├─ L_declared     q_π     declaration under-coverage (weighted undeclared-query loss)
   ├─ L_projection   Π_π     residual contract info given the side inputs
   ├─ L_decision     π_θ     rate at which decision-relevant pairs collapse
   └─ O_safety       g_safety  does invalid / unknown evidence trigger a conservative reaction
        │
        ▼
three contract-relevant invariants
   ├─ mode          multi-hypothesis structure            ──▶  primitive: mode_select
   ├─ temporal      heterogeneous staleness / health      ──▶  primitive: age_gate
   └─ source        provenance / dependency / negative ev. ──▶  primitives: provenance / dependency / negative evidence
        │
        ▼
training (representation-side conditional probe + Type I equivariance / Type II order-constrained / Type III unconstrained)
deployment (safety filter reads constraint-relevant certification; invalid evidence ALONE cannot justify relaxing constraint)
evaluation (four compliance evidence types: semantic / representation / decision / safety; CAG is only the decision-layer aggregate, needs an oracle baseline and a fixed retraining protocol)
```

### 0.2 Three-layer quotient-declaration chain: $\mathcal C\to\mathcal C_\pi\to{\sim_\pi}\to q_\pi$

The previous draft split $\Pi_\pi$ into two layers, $e_\pi \circ q_\pi$. This version accepts the reviewer's push-back — **$q_\pi$ was still carrying too much weight and the question "who defines the quotient" was not answered**. This version places two more layers in front of $q_\pi$, turning the interface into an **auditable three-layer declaration chain**:

$$\boxed{\;\mathcal C\;\xrightarrow{\text{declare}}\;\mathcal C_\pi\;\xrightarrow{\text{induce}}\;{\sim_\pi}\;\xrightarrow{\text{quotient}}\;q_\pi\;}$$

where:

- **$\mathcal C$ · Upstream contract** — the **complete** semantic schema delivered by the estimator (defined in 9/14 lower half §2–lower half §4).
- **$\mathcal C_\pi$ · Consumer contract** — **the policy, as a consumer, explicitly declares which slice of $\mathcal C$ it takes on**. $\mathcal C_\pi$ may be a subset of $\mathcal C$ ("this policy does not consume provenance"), or a coarse-graining of $\mathcal C$ ("hypothesis structure is folded to a point estimate but age is preserved as its own field"). **$\mathcal C_\pi$ is a paragraph in the interface spec, not an implicit preference inside encoder weights**.
- **${\sim_\pi}$ · Induced equivalence** — induced by a **declared set of contract queries / decision-relevant predicates** $Q_{\mathcal C_\pi}$ on $\mathcal C_\pi$:

$$\hat S \sim_\pi \hat S' \quad\Longleftrightarrow\quad Q_{\mathcal C_\pi}(\hat S) \;=\; Q_{\mathcal C_\pi}(\hat S').$$

$Q_{\mathcal C_\pi}$ is the real **interface object** of this piece — it turns "what may be dropped" from a vague semantic promise into an enumerable list of queries that can be reviewed one by one. Typical examples: "what is the age of channel $c$ in $\hat S$", "which are the top-3 posterior-weighted hypotheses for this track", "does the contact set include the pad face".

- **$q_\pi$ · Quotient map** — the actual $q_\pi : \hat S \mapsto [\hat S]_{\sim_\pi}$, folding $\hat S$ into the quotient space defined by ${\sim_\pi}$.

With this chain in place, the piece's core claim can finally be phrased in a way a reviewer cannot follow up with "who defines the quotient":

> **Policy does not need to preserve the entire upstream contract $\mathcal C$. It must explicitly declare a consumer contract $\mathcal C_\pi$ together with a query family $Q_{\mathcal C_\pi}$ — and the resulting quotient $q_\pi$ is exactly the semantic loss the policy is allowed to take.**

Compared to the previous version's "either preserve the whole quotient, or explicitly declare a sufficient quotient", this version delivers a complete answer to **where the quotient comes from**: it comes from a **written** $\mathcal C_\pi$ plus an **enumerable** $Q_{\mathcal C_\pi}$.

**$\Pi_\pi$ is still $e_\pi \circ q_\pi$, but $q_\pi$ is no longer a primitive — it is uniquely determined by ${\sim_\pi}$, which is in turn induced by $\mathcal C_\pi$.** Engineering consequence: **the interface spec must be able to display a list of $Q_{\mathcal C_\pi}$ queries**, otherwise $q_\pi$ degenerates back into encoder behavior — exactly what this piece is attacking.

**v6 addition · Three-part decomposition of the Consumer Contract.** The previous version only described $\mathcal C_\pi$ as "a subset / coarse-graining of $\mathcal C$" — the reviewer was right: **that only says what the policy reads; it says nothing about what the policy commits to in response, nor which schema version the policy understands**. v6 makes $\mathcal C_\pi$ an explicit triple:

$$\boxed{\;\mathcal C_\pi \;=\; \big(Q_\pi,\;\mathcal O_\pi,\;V_\pi\big),\;}$$

where

- $Q_\pi \equiv Q_{\mathcal C_\pi}$ · **Queries** — "which contract fields / queries I read" (already defined above; also the source of $q_\pi$).
- $\mathcal O_\pi$ · **Obligations** — "having read those fields, how I commit to respond". lower half §1.1's Type I equivariance, Type II order-constrained response, Type III unconstrained clause, and lower half §1.3's three-state safety certification obligation **are all concrete entries in $\mathcal O_\pi$** — not separate rules scattered through the paper. §0.2.2's three-losses-plus-one-obligation framing is derived directly from $\mathcal O_\pi$ as its auditable failure modes.
- $V_\pi$ · **Versions** — "which schema version I understand". §0.2.3's schema-compatibility rule is $V_\pi$'s operational form.

This decomposition makes the Consumer Contract look like a real software interface rather than only an ML abstraction:

```text
Consumer Contract  C_π
├── Q_π   Queries       What I read
├── O_π   Obligations   How I must respond
└── V_π   Versions      Which schema I understand
```

When the reviewer follows up with "you said the policy is a consumer — what does a consumer actually commit to?", v6 has a precise answer: **it commits to reading every field in $Q_\pi$, honoring every response rule in $\mathcal O_\pi$, and remaining compatible with every schema in $V_\pi$**. Missing any one of the three means it is not a complete consumer contract.

### 0.2.3 Schema version / compatibility (v5 · the $V_\pi$ slice of the triple)

With $\mathcal C_\pi$ and $Q_{\mathcal C_\pi}$ in hand, one more software-interface question must be answered: **which schema version does the consumer contract target?** Suppose the estimator upgrades to a v2 of $\mathcal C$ that adds `observability` / `negative_evidence` / `sensor_health`, while the policy's $\mathcal C_\pi$ is still stuck at v1. Under the current framework the new fields would be **silently dropped** — which is precisely the **undeclared semantic loss** this whole piece attacks. So the interface layer gains one more small rule:

$$\boxed{\;\mathrm{schema\_version}(\mathcal C)\;\not\simeq\;\mathrm{supported\_version}(\mathcal C_\pi)\;\;\Longrightarrow\;\;\text{reject / explicit adapter required.}\;}$$

That is, **schema mismatch must fail closed or pass through an explicitly declared adapter** — on incompatibility, either **refuse to load the policy**, or route through a **written** adapter (explicitly mapping v2's new fields onto existing v1 slots, or explicitly declaring them dropped, together with a rule $\mathcal C_\pi^{v2} = \mathrm{adapt}(\mathcal C^{v2},\mathcal C_\pi^{v1})$). This single clause is what gives the word "contract" a genuine software-interface feel rather than leaving it as a mere ML abstraction.

### 0.2.1 Three tiers of preservation: full semantic / decision-relevant semantic / decision sufficiency

This subsection is the theoretical anchor; the **three** commonly conflated properties must be pulled apart — the previous version separated only two, and the reviewer was right that "full semantic preservation as an interface criterion is over-preserving". v6 makes two additional closure steps that were still hanging: **the $\equiv$ in Properties A and A′ has never actually been defined**, and **$Y_\pi$ must be pinned to the Consumer Contract rather than being an arbitrary latent variable**.

**v6 definition · decision-relevant observable $Y_{\mathcal C}^{\pi}$**. The previous version wrote $Y_{\mathcal C}$ / $Y_\pi$ with drifting semantics. v6 locks it down:

$$\boxed{\;Y_{\mathcal C}^{\pi} \;=\; \big\{q(\hat S) : q \in Q_{\mathcal C}^{\mathrm{req}},\;\text{$q$ influences consumer decision}\big\}.\;}$$

In words: $Y_{\mathcal C}^{\pi}$ is the subset of required-query outputs that **actually influences the consumer's decision**. Too wide ($Y_{\mathcal C}^{\pi}=\hat S$) collapses back to "don't drop any contract information"; too narrow ($Y_{\mathcal C}^{\pi}=\text{current optimal action}$) drifts into decision sufficiency and is no longer contract semantics. $Y_{\mathcal C}^{\pi}$ is exactly the mathematical object the Consumer Contract idea should eat.

**v6 definition · representation equivalence $\sim_Z$**. The $\Pi_\pi(\hat S)\not\equiv\Pi_\pi(\hat S')$ in A and A′ cannot just mean "numerically different" — otherwise every floating-point rounding counts as preservation. v6 defines:

$$Z_\pi(\hat S) \sim_Z Z_\pi(\hat S') \quad\Longleftrightarrow\quad \nexists\;\text{measurable decoder } h_\pi(Z_\pi,O,L)\;\text{s.t.}\; h_\pi \text{ distinguishes } \hat S \text{ from } \hat S' \text{ on } Y_{\mathcal C}^{\pi}.$$

**If no downstream decision can, under the declared consumer, distinguish two representations, they are representation-equivalent.** This ties in directly with $L_{\mathcal C}^{\pi}=I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$: $L_{\mathcal C}^{\pi}=0$ is equivalent to $Z_\pi(\hat S)\sim_Z Z_\pi(\hat S')$ for all $(\hat S,\hat S')$ that differ on $Y_{\mathcal C}^{\pi}$.

Fix a set of **contract-relevant decision variables** served by the policy (quantities downstream controller / planner / safety filter / diagnostics will read), captured by the boxed $Y_{\mathcal C}^{\pi}$ above, together with the contract equivalence $\sim_\pi$ on $\hat S$ already defined in §0.2 (induced by $Q_{\mathcal C_\pi}$). Three properties are distinguished.

**Property A · Full Semantic Preservation** — the projection does not irreversibly collapse semantically distinct contracts:

$$\hat S \not\sim_\pi \hat S' \quad \Longrightarrow \quad Z_\pi(\hat S) \not\sim_Z Z_\pi(\hat S').$$

This requires $q_\pi$ to be injective on the quotient defined by $\mathcal C_\pi$ and $e_\pi$ to not map distinct classes to $\sim_Z$-equivalent $Z_\pi$. It is **a strong property, but possibly over-preserving**.

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

$$\hat S \not\sim_{\pi,\mathcal D} \hat S' \quad\Longrightarrow\quad Z_\pi(\hat S) \not\sim_Z Z_\pi(\hat S').$$

Because $\sim_{\pi,\mathcal D}$ is coarser than $\sim_\pi$, **Property A implies Property A′, and not vice versa**. This is the interface requirement this piece actually wants.

**Property B · Decision Sufficiency (conditional-MI form)** — measure "given the side information the policy already has, how much contract information about $Y_{\mathcal C}^{\pi}$ survives projection" using **conditional MI**, not raw MI difference. The previous version wrote $L_{\mathrm{dec}} = I(\hat S; Y_{\mathcal C}) - I(\Pi_\pi(\hat S); Y_{\mathcal C})$, and the reviewer caught the fatal problem: **the policy's full input is $\pi(a\mid \hat S, o, \ell, \text{language})$, and the raw image $o$ often already carries age / provenance proxies**. Under the difference form, $L_{\mathrm{dec}} > 0$ only says "$\Pi_\pi(\hat S)$ in isolation is not a sufficient statistic for $Y_{\mathcal C}$" — **it does not say the policy actually lacks information** (the information may already be recovered from $o$).

v5 redefined contract information loss as **conditional MI**; v6 pins the observable down to $Y_{\mathcal C}^{\pi}$:

$$\boxed{\;L_{\mathcal C}^{\pi} \;=\; I\!\big(Y_{\mathcal C}^{\pi}\,;\,\hat S \,\big|\, Z_\pi,\, O,\, L\big),\qquad Z_\pi = \Pi_\pi(\hat S, O, L).\;}$$

(When the projection acts only on the contract, $Z_\pi$ can be simplified to $\Pi_\pi(\hat S)$.) This definition is much cleaner:

- $L_{\mathcal C}^{\pi} = 0$ **if and only if** $Y_{\mathcal C}^{\pi} \perp\!\!\!\perp \hat S \mid Z_\pi, O, L$ — i.e. **$Z_\pi$ is sufficient for $Y_{\mathcal C}^{\pi}$ given the other inputs the policy already has**.
- This directly answers the reviewer objection "raw image already contains object identity / provenance proxies": **because we use conditional sufficiency rather than unconditional MI, information already recovered from the raw observation is not miscounted as loss**.
- **v6 addition**: $Y_{\mathcal C}^{\pi}$ is defined by the Consumer Contract, not by the benchmark — this is what makes "enough" have a unique answer that does not drift with how wide or narrow $Y$ is picked.

**Relations among the three (v5 makes the second arrow conditional)**:

$$\text{Full semantic preservation}\;\Longrightarrow\;\text{Decision-relevant preservation}$$

is **unconditional** — because $\sim_{\pi,\mathcal D}$ is coarser than $\sim_\pi$, pointwise injectivity from the former directly gives the latter. But the second arrow:

$$\boxed{\;\text{Decision-relevant preservation}\;\Longrightarrow\;\text{Decision sufficiency}\quad\text{only under the declared evaluation distribution.}\;}$$

is **no longer written as an unconditional theorem**. The reason (caught precisely by the reviewer): preservation is a pointwise statement about distinctions, sufficiency is a conditional independence under some $P(\hat S, O, L)$ — the two are not the same type of object. Once the evaluation-distribution assumption is added, the claim survives as: **under the benchmark distribution, and assuming $Y_\pi$ is a sufficient decision descriptor, decision-relevant preservation implies decision sufficiency**. Without that qualifier a math reviewer can immediately ask "preservation is a statement about all $\hat S$ — how do you push it to a distribution-level conditional independence?"

None of the implications reverse. **Decision sufficiency is the weakest property** — $L_{\mathcal C}^{\pi}=0$ permits collapsing two $\hat S$ values that $Y_\pi$ does not distinguish; **decision-relevant preservation is in the middle** — it protects only distinctions that would change action or safety; **full semantic preservation is the strongest** — a diagnostic property, not an interface requirement.

The killer line then becomes:

> **Decision-relevant semantic preservation is the actual interface requirement; full semantic preservation is a stronger diagnostic property; decision sufficiency is a weaker consequence (under the declared evaluation distribution).**

**A policy may drop information, but must either (a) preserve decision-relevant contract semantics, or (b) explicitly declare $\mathcal C_\pi$ + $Q_{\mathcal C_\pi}$ such that dropped distinctions are acknowledged by $Q_{\mathcal C_\pi}$ as outside its scope of concern**. Silently dropping decision-relevant contract semantics — without declaring — is the interface violation.

### 0.2.2 Three semantic losses + one safety obligation (v5 rewrite · v6 subsumption + separately-auditable)

The v4 draft split the whole pipeline into three layers — $q_\pi$ / $e_\pi$ / $\pi_\theta$ — and attached one loss to each, but the reviewer was right: **this "three-loss" framing has two type mismatches, and it mislabels safety as a loss**. v5 fixes three things at once; v6 then tightens (1) from a checklist into a subsumption order and (4) into a sequentially-coupled audit statement.

**Structure**: the pipeline is already four stages $\mathcal C\to(\mathcal C_\pi, Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}$, and **safety is not an information loss but an obligation**. So the final framing is not "three losses" but **"three semantic losses + one safety obligation"**:

$$\boxed{\;\mathcal C\;\longrightarrow\;(\mathcal C_\pi, Q_{\mathcal C_\pi})\;\longrightarrow\;q_\pi\;\longrightarrow\;e_\pi\;\longrightarrow\;\pi_\theta\;\longrightarrow\;g_{\mathrm{safety}}\;\longrightarrow\;a.\;}$$

**Objects**: the v4 $L_{\mathrm{declared}}$ and $L_{\mathrm{decision}}$ are both wrong; v5 rewrites each; v6 upgrades (1) once more.

**(1) $L_{\mathrm{declared}}$ — from entropy difference, to weighted coverage, to query subsumption (v6 second upgrade).** The v4 form $H(Q_{\mathcal C}(Y_{\mathcal C})) - H(Q_{\mathcal C_\pi}(Y_{\mathcal C}))$ has three problems: the random variable inside $Q_{\mathcal C}(Y_{\mathcal C})$ is never defined; an entropy difference is not guaranteed non-negative (query count / encoding / cardinality all move entropy); and the core semantics of $q_\pi$ is "declare which distinctions may be dropped", which is naturally a **coverage / violation set**, not a scalar entropy. v5 wrote it as a set $\mathfrak D_\pi = Q_{\mathcal C}^{\mathrm{req}}\setminus Q_{\mathcal C_\pi}$ — but the v6 reviewer pushed one more step: **"two queries being equal" is not a set-membership relation to begin with**. Two examples:

- $q_1 = \text{"age"}$, $q_2 = \text{"whether age > 100ms"}$ — $q_2$ is a **function of** $q_1$; $q_1$ already subsumes $q_2$.
- $q_3 = \text{"top-3 hypotheses"}$, $q_4 = \text{"MAP hypothesis"}$ — $q_3 \succeq q_4$, but $q_3 \neq q_4$.

So $q \notin Q_{\mathcal C_\pi}$ cannot stay a literal membership check. **v6 introduces a query subsumption order**:

$$q_1 \succeq q_2 \quad:\!\!\Longleftrightarrow\quad \text{the information preserved by } q_1 \text{ is enough for a consumer, under any side information, to answer } q_2$$

(formally: there exists a measurable $h$ such that $h(q_1(\hat S)) = q_2(\hat S)$; or the weaker conditional version $H(q_2 \mid q_1) \le \varepsilon$). Coverage then becomes:

$$\boxed{\;L_{\mathrm{declared}} \;=\; \sum_{q\,\in\, Q_{\mathcal C}^{\mathrm{req}}} w_q\,\mathbf 1\!\Big[\nexists\, q' \in Q_{\mathcal C_\pi}:\;q' \succeq q\Big]\;}$$

This upgrades $L_{\mathrm{declared}}$ from a checklist into a **semantic capability lattice** — $Q_{\mathcal C}^{\mathrm{req}}$ and $Q_{\mathcal C_\pi}$ both sit on the same subsumption partial order, and "what may be dropped" becomes "does the upper closure $\{q' : q' \succeq q \text{ for some } q \in Q_{\mathcal C}^{\mathrm{req}}\}$ declared by the consumer cover the required set". $w_q$ is a task-side weight: frame semantics high, age medium-high, provenance task-dependent, a diagnostic field wholly irrelevant to the current controller low. The crudest cardinality form ($q' \succeq q \Leftrightarrow q' = q$) is exactly the $w_q \equiv 1$ special case of v5. Auditing $q_\pi$ now means **walking the subsumption order one required query at a time and checking upper-cover**, no longer leaning on an entropy definition and no longer tripped up by "the literal query strings differ".

**(2) $L_{\mathrm{rep}}$ renamed $L_{\mathrm{projection}}$, made explicit as residual contract information rather than an encoder loss; v6 pins the observable to $Y_{\mathcal C}^{\pi}$.** In v4, $Z_\pi = \Pi_\pi(\hat S, O, L)$ is already the output of the **whole projection**, not just the encoder $e_\pi$, so calling it a representation loss forces a one-to-one identity between a math object and a pipeline layer. v5 renames it; v6 locks the $Y$ inside to the Consumer-Contract observable defined in §0.2.1:

$$\boxed{\;L_{\mathrm{projection}} \;=\; I\!\big(Y_{\mathcal C}^{\pi}\,;\,\hat S \,\big|\, Z_\pi,\, O,\, L\big),\qquad Y_{\mathcal C}^{\pi} = \{q(\hat S): q \in Q_{\mathcal C}^{\mathrm{req}},\;q \text{ influences consumer decision}\}.\;}$$

Its meaning is **residual contract information after the projection** — it can be operationalized as a representation-stage loss, but does not claim to be the loss of the $e_\pi$ segment alone. $Y_{\mathcal C}^{\pi}$ is defined by the Consumer Contract, **not an arbitrary latent in the benchmark** — this is precisely the mathematical object the Consumer Contract idea should eat.

**(3) $L_{\mathrm{decision}}$ — from supremum norm to action-relevant collapse rate.** The v4 form $\sup_{\hat S \not\sim_{\pi,\mathcal D} \hat S'} \big\|\pi_\theta(\hat S) - \pi_\theta(\hat S')\big\|_{\text{action-distribution}}^{\!\perp}$ is a **type error** — a norm is a distance, not a "size of a pair set". v5 writes it as a collapse rate:

$$\mathcal R_{\mathcal D} \;=\; \big\{(\hat S, \hat S') : \hat S \not\sim_{\pi,\mathcal D} \hat S'\big\}$$

$$\boxed{\;L_{\mathrm{decision}} \;=\; \mathbb E_{(\hat S,\hat S')\sim\mathcal R_{\mathcal D}}\!\Big[\mathbf 1\!\big(D_{\mathcal A}(\pi_\theta(\cdot\mid \hat S),\,\pi_\theta(\cdot\mid \hat S')) < \epsilon\big)\Big]\;}$$

where $D_{\mathcal A}$ is **not an ordinary distribution distance** — it measures whether the two action distributions **support different admissible / optimal action sets**. This also absorbs the reviewer's philosophical objection: if $\mathcal A^*(\hat S_1) \neq \mathcal A^*(\hat S_2)$ but some action in their common intersection is admissible for both, a policy emitting that shared action is legitimate and $D_{\mathcal A}$ must not count it as collapse. A soft-threshold version is $\mathbb E_{\mathcal R_{\mathcal D}}[\exp(-D_{\mathcal A}(\cdot))]$. $D_{\mathcal A}$ closes the semantic loop with lower half §2.2's HPC / HSS / $E_{\mathrm{contract}}$: **HPC measures world-consistent coverage, HSS measures separation, $E_{\mathrm{contract}}$ measures interface compliance, and $L_{\mathrm{decision}}$ measures their failure rate**.

**(4) $\mathcal O_{\mathrm{safety}}$ is an obligation, not a loss.** The safety layer does not measure "information loss"; it judges the **obligation of whether the filter takes a conservative reaction when evidence is insufficient or unknown** (see lower half §1.3's three-state certification and lower half §2.6's safety evidence). **It is a fourth kind of object and must not be lumped together with the three losses.**

| Slot | Semantic | Failure | Object |
|---|---|---|---|
| $q_\pi$ | declares what may be dropped | **Declared coverage loss** — $Q_{\mathcal C_\pi}$ fails to upper-cover a required query along $\succeq$ | $L_{\mathrm{declared}} = \sum w_q \mathbf 1[\nexists q' \in Q_{\mathcal C_\pi}: q' \succeq q]$ |
| $\Pi_\pi$ | post-projection conditional residual | **Projection residual** — conditional MI > 0 | $L_{\mathrm{projection}} = I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$ |
| $\pi_\theta$ | does the decision use preserved distinctions | **Decision collapse rate** — fraction of action-relevant pairs collapsed | $L_{\mathrm{decision}} = \mathbb E_{\mathcal R_{\mathcal D}}[\mathbf 1[D_{\mathcal A} < \epsilon]]$ |
| $g_{\mathrm{safety}}$ | obligation response when evidence is insufficient | **Safety obligation violation** — fails to tighten on unknown / invalid | $\mathcal O_{\mathrm{safety}}$ |

$\Pi_\pi$ injective **does not imply** $L_{\mathrm{decision}} = 0$ — but v6 states this precisely: the three losses are **three separately auditable failure sites**, not three statistically independent losses. The reviewer's point is exactly right: $L_{\mathrm{declared}}$, $L_{\mathrm{projection}}$, $L_{\mathrm{decision}}$ are **sequentially coupled** through $q_\pi \to \Pi_\pi \to \pi_\theta$ — once the upstream declaration changes, the admissible quotient downstream changes with it; they cannot be random-variable independent. **"Independently" is replaced throughout the paper by "separately auditable"**: each loss localizes to an openable pipeline site, not to an independence claim. lower half §2's four compliance evidence types and lower half §2.0's skeleton table both draw directly on these three losses plus the one obligation.

### 0.3 Two boxed claims of this piece (Claim 3 and the lower half live in 9/16)

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. The full pipeline $\mathcal C\to(\mathcal C_\pi,Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}$ is **an auditable semantic interface**, with the three semantic losses sitting on $q_\pi$ / $\Pi_\pi$ / $\pi_\theta$ as **three separately auditable failure sites (sequentially coupled through the pipeline), not statistically independent losses**, and the fourth slot being a safety obligation, not a loss. This is the embryo of the piece's upgraded thesis — **A policy is a contract consumer, not merely a function approximator**.

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state heads, latent visuomotor policies, and VLA / diffusion / flow policies use different conditioning and action-generation mechanisms — but as contract consumers they must all answer **the same four questions**: what may I discard ($q_\pi$)? what did I actually retain ($\Pi_\pi$, $L_{\mathrm{projection}}$)? how should decisions respond to contract interventions ($\pi_\theta$, $L_{\mathrm{decision}}$)? what happens when evidence becomes invalid or unknown ($g_{\mathrm{safety}}$, $\mathcal O_{\mathrm{safety}}$)? The target of criticism is the interface contract, not the model architecture.



## 1. Two dimensions instead of three families: conditioning representation / semantic interface × action head

> **Position of this section (v6 addition · navigation signpost).** §1–§3 are **implementation coordinates for the interface theory already set up in §0**, not the core argument itself. §0 has already defined the pipeline $\mathcal C\to(\mathcal C_\pi, Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}$, the three tiers of preservation, and the three-losses-plus-one-obligation framing. §4 onward introduces the primitives, lower half §1 introduces training and deployment consequences, lower half §2 introduces benchmarking. §1–§3 only provide a concrete grid for "which cell of the pipeline each existing policy actually sits in" — **readers already familiar with the VLA / Diffusion / flow / ACT / engineered-head families can jump straight to §4**. This section exists to align §0's pipeline with real systems, not to rank architectures.

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

One caveat: **this is not a ranking of "which combination is best"**. What this piece cares about is **for every combination, is there a written-out $q_\pi$ in $\Pi_\pi = e_\pi \circ q_\pi$**. In most existing work the answer is "no" — **not because a family is inherently bad, but because this $q_\pi$ layer has never been treated as an interface design problem**.

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

**The common misuse of the phrase "multimodal fusion" on the policy side** is treating a cross-attention on one branch as if it were "already doing multimodal state estimation". It is not. Real state abstraction requires the fields inside $\hat S_t$ to **carry consistent semantics across sensor families and be jointly readable by four consumers — controller / policy / world model / diagnostics** — 9/14 lower half §2 has already established this convention; what this section adds is **asking from the policy side once more: was $q_\pi$ declared explicitly, or was it silently replaced by $e_\pi$?**.

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

The contract explicitly separates availability (does the channel have data today), validity (is the data valid, e.g. is calibration current), and age (how old). A policy that reads only the numbers conflates "stale but valid" with "missing but valid", and treats "invalid after calibration drift" the same as "sensor disconnected". 9/14 lower half §4.5 already broke the degradation chain apart — **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption** — if the policy does not ingest this chain at the input, both training-time augmentation and inference-time guardrail will latch onto the wrong place.

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

The previous version called this primitive "calibration-aware top-$k$", and the reviewer was right — renormalizing after truncation **only solves probability-mass conservation, not calibration**. $\sum_i w_i = 1$ does not imply posterior calibrated. This version renames the primitive to **mass-preserving top-$k$**, and evaluation of calibration is moved to lower half §2:

$$\tilde w_i \;=\; \frac{w_i}{\sum_{j \in \mathrm{top}\text{-}k} w_j}, \qquad w_{\mathrm{other}} \;=\; 1 - \sum_{i \in \mathrm{top}\text{-}k} w_i.$$

$\tilde w_i$ is the **within-top-$k$ renormalized** weight; $w_{\mathrm{other}}$ is the **residual mass**; both are passed downstream.

**v5 adds a qualifier (the reviewer caught the v4 over-claim).** $w_{\mathrm{other}}$ is only an **aggregate residual mass** — it does not tell the downstream consumer the residual hypotheses' **location, covariance, likelihood, or component identity**. So mass-preserving top-$k$ guarantees **probability-mass accounting** (the consumer can at least distinguish retained mass from discarded residual mass), but it does **not** guarantee **Bayesian-update correctness** — a strict posterior update over the residual component still requires a separate specification of its sufficient statistics. The semantic boundary of the primitive is drawn right here, and the piece does not step past it.

Whether the posterior is actually calibrated is an **evaluation property**, not part of the primitive. The piece's calibration diagnostics live in lower half §2's representation-level evidence and comprise **five separate measures: ECE$_{\text{top-}k}$, NLL, Brier, calibration curve, coverage–credibility**, each independent of the top-$k$ truncation itself. Primitive layer and evaluation layer are fully separated, and a reviewer can no longer ask "What exactly makes your top-$k$ calibration-aware?".

The point is **not "mean is forbidden"** — mean is a perfectly legitimate readout, provided the collapse is **explicitly declared**. The real failure mode is "the interface provides no readout slot for hypothesis structure, $e_\pi$ can only implicitly merge via concat + MLP, and mean becomes the default". This distinction is critical: **this piece is against undeclared default collapse, not against collapse per se**.

### 4.2 `age_gate`: observation payload + metadata quintuple + derived trust

**A common interface-design bug** is to multiply staleness trust directly into the measurement: $x_c^\pi = \tau_c(\alpha_c) \cdot \mu_c$. This **changes the physical value of the observation** — 10 N read at 100 ms age gets multiplied into "3 N", and the "3 N" at the policy input **looks** like "a 3 N force", not "a 10 N force whose trust has decayed". This directly violates the very distinction the contract wants to preserve: **$(F = 3\,\mathrm{N},\, \alpha = 0)$ and $(F = 10\,\mathrm{N},\, \alpha = 100\,\mathrm{ms})$ are two different semantic events**.

The previous version (v4) already re-grouped the fields so that the "seven vs eight" counting inconsistency disappears. v5 does one more **notation cleanup** — the reviewer was right that throughout the piece $a$ denotes both the action and the age, and lower half §2.3's $R_\pi(a)$ is especially easy to misread. **From v5 onward, the age field is uniformly $\alpha_c$ and the availability field is uniformly $\iota_c$; $\alpha$ and $\iota$ no longer collide with $a$ (the robot action).**

**(i) Observation payload** — the measurement and its covariance:

$$\text{payload}_c \;=\; (\mu_c,\;\Sigma_c).$$

**This payload exists only when $\iota_c = 1$**. $\iota_c = 0$ means the estimator has nothing to deliver at this instant; $q_\pi$ can only read "no-data" as a fact.

**(ii) Temporal / operational metadata (five fields)** — describing how, when, and by which sensor the payload was acquired, and how current it is:

$$m_c \;=\; \big(\underbrace{\alpha_c}_{\text{age}},\;\underbrace{\ell_c}_{\text{latency / causal status}},\;\underbrace{h_c}_{\text{sensor health}},\;\underbrace{v_c}_{\text{validity (calibration)}},\;\underbrace{\iota_c}_{\text{availability}}\big).$$

**(iii) Derived trust (one field, not a primitive)** — computed from the five-tuple:

$$q_c \;=\; \tau_c(\alpha_c,\;\ell_c,\;h_c,\;v_c,\;\iota_c,\;\ldots).$$

The ellipsis admits task-specific inputs (controller mode, current dynamics regime, etc.). The concrete form of $\tau_c$ (learned / analytic / piecewise) is a policy-specific design decision left outside the interface.

The complete policy-input structure is thus **"six primitive metadata fields + one derived trust field"** — $(\mu_c,\Sigma_c,\alpha_c,\ell_c,h_c,v_c,\iota_c) + q_c$, countable on fingers; the reviewer no longer needs to ask "seven or eight".

**Availability vs validity — semantic difference** — the previous version did not pull these apart, and the reviewer was right. A 2×2 table:

|  | $v_c = 1$ (valid) | $v_c = 0$ (invalid) |
|---|---|---|
| $\iota_c = 1$ (payload present) | Normal evidence; policy consumes via §4.1 | Data exists but **must not be treated as valid evidence** (e.g. expired calibration); policy should be handled through lower half §1.3 safety side |
| $\iota_c = 0$ (no payload) | Semantically impossible (no data ⇒ validity bit meaningless) | Channel down / masked; policy reads the fact "no observation" |

One line: **availability says "is there a payload", validity says "does this payload count as valid evidence" — orthogonal, cannot be collapsed into one bit**. lower half §1.2's degradation chain (masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption) is essentially **different patterns** among the five metadata fields, and this version aligns augmentation with slots one-to-one.

**If a downstream gate really is required, it should gate on uncertainty, not on measurement.** But **the previous version wrote $\tilde\Sigma_c = \Sigma_c / \tau_c(\alpha_c)$ as a general principle "stale ⇒ effective uncertainty is inflated" — also rightly called out by reviewers**. The correct treatment of a stale observation is not necessarily simple inflation; the more general form is **propagating the latent state forward**:

$$p(x_t \mid y_{t-\Delta t}) \;=\; \int p(x_t \mid x_{t-\Delta t})\, p(x_{t-\Delta t} \mid y_{t-\Delta t})\, dx_{t-\Delta t}.$$

When the robot is stationary, vision age 200 ms may leave the measurement still very accurate; when the robot is moving fast, the same 200 ms can imply huge predictive uncertainty. **Inflation and propagation are two different treatments, not synonyms.** This piece positions $\Sigma / \tau$ explicitly as **a simple conservative approximation**:

$$\Sigma_c^{\mathrm{eff}} \;=\; \mathrm{Propagate}\!\big(\Sigma_c,\; \Delta t,\; u_t,\; f_{\mathrm{dyn}}\big) \qquad\text{(general form)}$$

$$\Sigma_c^{\mathrm{eff}} \;=\; \Sigma_c \,/\, \tau_c(\alpha_c, \ell_c, h_c, v_c, \iota_c) \qquad\text{(one conservative approximation)}$$

The right phrasing is: **staleness should modify the policy's uncertainty model; uncertainty inflation is one conservative implementation, while predictive state propagation is another.** This version demotes $\Sigma / \tau$ from "canonical" to "one implementation" and hangs `Propagate` next to it as the general form. $\iota_c$ and $v_c$ still **hang on as parallel metadata slots** — they must not be folded into $\mu$, nor into $\tau$.

This looks like a small change but it semantically repairs `age_gate` from "discounting a measurement" to "measurement + facts about the measurement + a derived trust" — a concrete projection of the §0.2 $L_{\mathcal C}^{\pi}$ definition: mixing a measurement with **facts about** the measurement into a single scalar is a direct source of $L_{\mathrm{projection}}$.

### 4.3 `provenance / dependency / negative evidence`: three things, not one bucket

9/14 lower half §2.1 groups `contributing_mask`, `correlated_with` and `negative_evidence` under `provenance` — from the estimator side that makes sense (all three are "source structure of this field"), but from the **policy-side readout** they sit at different semantic levels:

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

This is the correct characterization of "what should have been seen but was not" — **only when $\mathcal O$ says something should have been seen does $E^-$ carry likelihood-side meaning about $H$**. This also plugs directly into the observability field defined in 9/14 lower half §2: **negative evidence is likelihood-side evidence, not provenance**.

On the policy side: expose $\Lambda(E;H,\mathcal O)$ (or the logit of $\exp(\Lambda)$) as a field inside $[\hat S]_{\sim_\pi}$ so the policy or belief update can consume it. This primitive is the thinnest on engineering maturity but often the strongest in effect on hypothesis ranking.

The three sub-primitives together — **provenance covers `where evidence came from`, dependency covers `how evidence is statistically related`, negative evidence covers `what expected evidence failed to appear` — support the source-structure invariant of §0.2**.

### 4.4 Mapping primitives to invariants

Back to the §0.1 figure: **mode / temporal / source** correspond to **mode_select / age_gate / (provenance_harden + dependency_aware_fusion + negative_evidence_read)**. Miss any one invariant, at least two of the four §3 failures recur. Three families are also **not a complete interface design** — observability / identifiability, frame convention, and contact set each have their own more specialized readouts (9/14 lower half §3, lower half §4.6); this piece only handles **the three most easily destroyed silently at the $\Pi_\pi$ layer**.

## 5. Bridge to the lower half: this piece ends at framework, evaluation starts in 9/16

By this point the upper half has done exactly one job — **stand the policy-side interface as an auditable semantic object**. A recap:

- §0 built the pipeline $\mathcal C\to(\mathcal C_\pi,Q_{\mathcal C_\pi})\to q_\pi\to e_\pi\to\pi_\theta\to g_{\mathrm{safety}}$, the Consumer Contract triple $\mathcal C_\pi=(Q_\pi,\mathcal O_\pi,V_\pi)$, three tiers of preservation (full / decision-relevant / decision sufficiency), three semantic losses + one safety obligation, and the §0.2.1 decision-relevant observable $Y_{\mathcal C}^{\pi}$ together with representation equivalence $\sim_Z$.
- §1 replaced the "three families" classification with **two orthogonal dimensions** — conditioning representation / semantic interface × action head — and gave the §1.2 grid.
- §2 pinned what "State" means across policy families — engineered state for SAC / PPO, visual latent for Diffusion / ACT / Flow, token sequence for VLA, VLM + proprio token for π0.
- §3 listed **four interface-mismatch failure modes** — multi-hypothesis silently collapsed, no interface-level guarantee of semantic correctness, temporal alignment broken by concat, safety-blindness to validity vs staleness.
- §4 defined **three families of contract-read primitives** — `mode_select` (hypothesis-layer readout), `age_gate` (observation payload + metadata quintuple + derived trust), `provenance_harden / dependency_aware_fusion / negative_evidence_read` (three things split, different semantic layers).

Two boxed claims stand here as the closing of the upper half:

> **Claim 1 · Contract semantics can be lost at the policy boundary.** A structured estimator output does not imply a structured policy input. The full pipeline is **one auditable semantic interface**, with the three losses landing on $q_\pi / \Pi_\pi / \pi_\theta$ and the fourth slot $g_{\mathrm{safety}}$ — **not a loss but an obligation**. **Four separately auditable failure sites** (v6 wording, **not four statistically independent losses**; sequentially coupled via $q_\pi \to \Pi_\pi \to \pi_\theta$).

> **Claim 2 · Contract preservation is not architecture-specific.** Engineered-state heads, latent visuomotor policies, autoregressive VLAs, diffusion policies, and flow policies all use different conditioning and action generators — but as contract consumers they must answer **the same four questions**: what may I discard ($q_\pi$; $\mathcal C_\pi$; $L_{\mathrm{declared}}$ via query subsumption)? what did I actually retain ($\Pi_\pi$; $L_{\mathrm{projection}} = I(Y_{\mathcal C}^{\pi};\hat S\mid Z_\pi,O,L)$)? how should decisions respond to contract interventions ($\pi_\theta$; $L_{\mathrm{decision}}$ via $D_{\mathcal A}$; §2.2 of the lower half splits $E_{\mathrm{contract}}$ / HPC)? what happens when the evidence becomes invalid or unknown ($g_{\mathrm{safety}}=\bigcap_j g_j$; safe → relaxation permitted / unsafe → tighten or stop / unknown → conservative fallback)? The target of critique is the interface contract, not the model architecture.

$$\boxed{\;\textbf{A policy is a contract consumer, not merely a function approximator.}\;}$$

### 5.1 Claim 3 and what the lower half carries

The upper half stops at "the interface should be read correctly". To turn this thesis into something testable, three more layers of work are needed:

1. **How to test** — four compliance evidence types (semantic / representation / decision / safety) + five-layer evaluation hierarchy (Retention ≠ Sufficiency ≠ Behavioral use ≠ Utility ≠ Safety, four non-implications) + §2.2 splitting HPC into $E_{\mathrm{contract}}(T^{\mathrm{contract}})$ / HPC($T^{\mathrm{world}}$) + §2.5 CAG matched null control.
2. **How to train** — training-side representation probe + three tiers of intervention consistency (Type I equivariance / Type II consumer-declared order / Type III unconstrained) + augmentation via degradation-as-causal-operator + §1.3 three-state constraint certification.
3. **How to land** — Python minimal executable interface skeleton; the three v6 bug fixes (`super().__init__()`, fail-open schema check, top-$k$ API conflating two orthogonal knobs).

These three all live in the **lower half** — [Policy-Side Evaluation (Part 2): How to Test, How to Train, How to Land](/en/articles/2026-09-16-policy-side-evaluation/). The lower half's closing thesis is the dual of the upper half:

> **Claim 3 · Contract compliance should be tested by controlled intervention, not inferred from end-to-end success.** End-to-end success measures whether a policy is useful, not whether it reads contract semantics correctly. Compliance needs **four evidence types converging**, an **oracle baseline** pinning the semantic range of each metric, and an **explicit retraining protocol** (distinguishing $\mathrm{CAG}^{\mathrm{fixed}}$ from $\mathrm{CAG}^{\mathrm{retrained}}$). **Probe ≠ semantic compliance, CAG ≠ semantic compliance, safety pass ≠ representation retention** — the four evidence types cannot substitute for each other and cannot be compressed into a single scalar.

If a single article were to carry framework, protocol and Python skeleton at the same time, it would balloon to 20K+ words in English (matching the size of 9/14 and 9/13 combined), and readers would have to cross 30+ subsections from concept to landing. Splitting the two halves lets each one sit at 8-10K words — **theory readers stop at the upper half; engineering readers start at lower-half §0.1 and jump back to upper-half §4 only when they need primitives**.

One closing note: the term "fusion" should be used sparingly from here on. On the time axis it asks "when to merge"; on the semantic axis it asks "merge into what". The 9/14 piece and this upper half together decompose the second question into six segments — **what upstream delivers + what policy declares as a consumer + what $q_\pi$ may discard + what $e_\pi$ preserves + what $\pi_\theta$ uses + what $g_{\mathrm{safety}}$ is obligated to**. The remaining three segments — **how to test, how to train, how to land** — belong to the lower half (9/16).

## Sources

arXiv IDs below have been verified online; journal-only references are not accompanied by an arXiv link. Grouped by the section they support. **Uncertainty / calibration / belief-space references (Guo 2017 / PlaNet / DreamerV3) and evaluation-protocol methodology are listed separately in the lower half (9/16) and are not repeated here.**

### A · VLA family (supports §1 grid, §3 Failures 1–2, §5.1 Class B elaborated in the lower half)

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) (**paper fact**: robot action is expressed explicitly as text tokens and jointly fine-tuned with the VLM · typical form of §3.2 Failure 2 on the tokenizer side; "contract flattening" is this piece's analysis, not the original paper's limitation)
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) (**paper fact**: 7B VLA trained on large-scale robot demonstrations; the "multi-camera + depth + proprioceptive state encoding" configuration is what §4 *Model Architecture & Training*, Table 2 and §5.1 of the original paper report — **not an abstract-level claim**. **This piece's analysis**: even at that implementation-section level, "which input modalities are supported" ≠ "which contract site is actually read"; the OpenVLA critique here is an interface analysis, not the paper's self-limitation)
- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) (**paper fact**: pretrained VLM + proprio token + noisy action chunk + flow matching; **this piece's analysis**: under the Structured State Contract defined here, π0's standard conditioning interface does not expose an explicit **first-class contract slot** for hypothesis / provenance / age / negative evidence — note the wording is "standard conditioning interface does not expose a first-class slot", **not** "π0 lost provenance". Provenance was never promised by π0's conditioning interface. "Continuous actions do not imply structured state semantics" is this piece's analysis, not the original paper's self-limitation)

### B · Diffusion / Generative-Sequence / Flow-Matching Policy (supports §1 grid, §5.1 elaborated in the lower half)

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 · [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) (**paper fact**: RGB stack + proprio concat + conditional denoising diffusion, emphasizing action-distribution multimodality; **this piece's analysis**: action-side multimodality ≠ state-side hypothesis preservation — an inference under this piece's schema, not a limitation the original paper admits)
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT / ALOHA), RSS 2023 · [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) (**paper fact**: CVAE + transformer encoder-decoder, core mechanism is **action chunking over sequences** — this piece places ACT in the "generative sequence decoder" category alongside diffusion / flow matching, not in the diffusion family)
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 · [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) (**paper fact**: establishes the flow-matching objective as vector-field regression for generative modeling / CNF — **flow matching itself is not a robot action chunking paper**; "continuous robot action chunks" is a specific application in work such as π0; this piece splits the citation chain accordingly — Lipman establishes the objective, π0 applies it to action chunks)

### D · SAC / PPO and engineered-state head baseline (supports the first row of §1 grid)

- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, ICML 2018 · [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) (Gaussian NLL / max-entropy policy loss; the engineered-state head form of §5.1 $\mathcal{L}_{\mathrm{action}}$ is elaborated in the lower half §1.1, referenced here as baseline only)

### E · Series continuations (how this piece plugs into 9/13, 9/14, and the lower half 9/16)

- This blog, *Splicing is not seeing: what robot multimodal fusion is missing is an interface, not a model* · `/en/articles/2026-09-14-multimodal-fusion-interface/` (Structured State Contract definition, Interface Property Benchmark, degradation chain; §0.2 $q_\pi$ and §3–§4 of this piece build directly on top)
- This blog, *Only seeing, never touching: why robots lack a hand with feel* · `/en/articles/2026-09-13-tactile-force-sensing/` (four force / tactile control paradigms, closed-loop value; historical source of the action-head spectrum in §1 grid)
- **Lower half of this piece** · *Policy-Side Evaluation (Part 2): How to Test, How to Train, How to Land* · `/en/articles/2026-09-16-policy-side-evaluation/` (this piece's twin lower half; picks up §5.1 Claim 3 and expands training-time knock-ons / four compliance evidence types + five-layer evaluation hierarchy / Python skeleton / closing claims and next-step benchmark)

---

> **Related reading**
>
> - [Policy-Side Evaluation (Part 2): How to Test, How to Train, How to Land](/en/articles/2026-09-16-policy-side-evaluation/) — **lower half of this piece**; picks up the framework and expands evaluation and implementation
> - [Splicing is not seeing: what robot multimodal fusion is missing is an interface, not a model](/en/articles/2026-09-14-multimodal-fusion-interface/) — upstream piece; stands the deliverable as a Structured State Contract
> - [Only seeing, never touching: why robots lack a hand with feel](/en/articles/2026-09-13-tactile-force-sensing/) — historical lineage of the action-head spectrum in §1 grid; four force-control paradigms
> - [VLA and World Models: two diverging routes and their intersection](/en/articles/2026-09-07-vla-world-models/) — macro backdrop of the conditioning dimension in §1; the other face of "world models do not inherently belong to sim-to-real"
> - [A quick tour of the VLA π family](/en/articles/2026-09-05-vla-pi-family/) — a concrete cut at π0, π0.5 and flow-matching action heads; live evidence for §1 "continuous action ≠ structured state"
> - [What is a VLA model? A single-piece walkthrough](/en/articles/2026-09-03-vla-deep-dive/) — assumed read by this piece
