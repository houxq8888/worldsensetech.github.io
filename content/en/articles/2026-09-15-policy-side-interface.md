---
title: 'After the Contract: What VLA, Diffusion Policy and π0 Actually Consume'
slug: "2026-09-15-policy-side-interface"
date: 2026-09-15
draft: false
categories: ["Embodied AI", "Policy Learning"]
tags: ["Embodied AI", "Policy Learning", "VLA", "Diffusion Policy", "π0", "RT-2", "OpenVLA", "Action Tokenization", "Structured State Contract", "Contract-read Primitives", "Belief State", "Multimodal Hypothesis", "Provenance", "Uncertainty Calibration", "Safety Filter", "Hypothesis Preservation", "Staleness Discrimination", "Evaluation Metrics"]
description: 'The companion to the multimodal-fusion interface piece. That article stood the upstream deliverable up as a Structured State Contract; this one asks its dual: if estimators really ship by the contract, can the policy side actually consume it. The answer is not reassuring—VLA, Diffusion Policy and engineered-state heads each silently flatten a different slice of the contract at their input boundary, the damage is invisible in the loss curve and does not heal by scaling the backbone. This piece proposes three contract-read primitives on the policy side—mode_select for hypothesis-set readout, age_gate for heteroscedastic staleness trust decay, provenance_condition for correlated-evidence double-counting—and connects the follow-through changes at the loss, augmentation, and safety-filter layers. Three interface-level metrics are given for evaluation: HPS, SDS, PCE. Closes on three claims: a good policy must read disagreement, not merely react to it; action space is not just kinematic, it is semantic; contract compliance is measurable at the policy boundary, not only at the estimator boundary.'
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

> Continues [Piling Modalities Together Is Not Understanding: What Robot Multimodal Fusion Lacks Is an Interface, Not a Model](/en/articles/2026-09-14-multimodal-fusion-interface/). That piece dragged multimodal fusion away from "which fusion timing" and back to "what is delivered after fusion", and stood that deliverable up as a contract object—the **Structured State Contract**—a structured state carrying hypothesis, provenance, observability, availability / validity / age, contact set and negative evidence. It closed on the claim: **A good multimodal system must represent disagreement, not merely resolve it.** This piece asks the **dual**: if the upstream really ships by the contract, **can the policy side actually consume it**.

Short answer: **most mainstream policy architectures silently flatten the contract at their input boundary**. VLA routes through a tokenizer, Diffusion Policy routes through image-encoder + proprio concat, engineered-state heads route through a hand-written state vector—three projections, three different slices of the contract broken. Worse, **the breakage is invisible**: the loss curve keeps going down, the eval score keeps going up, and there is nothing in the training log telling you what got squeezed out. This is not a model-size problem, **it is an interface problem**.

This is a piece about the segment between upstream deliverable and downstream policy input. It is not going to recommend a specific backbone, and it is not going to argue against end-to-end learning. What it actually argues against is—**treating the choice of policy-side interface as a question of "parameter count and dataset size"**. The article first looks at the contract from the policy side, breaks down the projection loss for each of the three mainstream families, then gives the three **contract-read primitives** the policy side genuinely needs to write down (mode_select / age_gate / provenance_condition), and traces the follow-through changes at training loss, data augmentation, safety filtering and evaluation. It closes on three claims.

## 0. Frame: what the contract looks like from downstream

The frame first, every later section returns to it.

```
              Upstream (defined in 9/14)              Downstream (this piece)
    ┌────────────────────────────────┐         ┌───────────────────────────┐
    │  Structured State Contract     │         │  Policy family            │
    │  ─────────────────────────────  │         │  ─────────────────────    │
    │  hypothesis + posterior w       │  ──Π──▶ │  A. engineered-state head │
    │  availability / validity / age  │  ──Π──▶ │  B. diffusion / flow      │
    │  observability / frame          │  ──Π──▶ │  C. VLA                   │
    │  contact set / negative evidence│         │                           │
    │  provenance / correlated_with   │         │  Which slice each Π kills │
    └────────────────────────────────┘         └───────────────────────────┘
                    │                                          ▲
                    │   contract-read primitives               │
                    │   ───────────────────────                │
                    └──▶ mode_select ─ age_gate ───────────────┘
                              └──▶ provenance_condition ───────┘
```

Three boxed claims the article stands on:

> **Claim 1 (Contract can die at the input boundary)**—the semantics of a Structured State Contract can be silently destroyed by a single projection layer at the policy input. Nothing about "how big the model is" or "how much data you have" can save it—only explicit contract-read primitives can.

> **Claim 2 (Three policy families, three projection losses)**—VLA / Diffusion Policy / engineered-state head each break a different slice of the contract. The real difference between the three is not parameter count or modality coverage; it is **what state schema each of them commits to**.

> **Claim 3 (Policy needs primitives, not bigger backbones)**—what the policy side genuinely needs is to write down mode_select, age_gate and provenance_condition explicitly at the input boundary; not to move from a 3B backbone to a 7B one.

A notation convention: below, "policy input" is written $x_t^{\pi}$ and is a projection of the contract $\hat S_t$, observations $o_t$ and language goal $\ell_t$, namely $x_t^{\pi} = \Pi_{\pi}(\hat S_t, o_t, \ell_t)$. **Whether the projection is contract-preserving depends entirely on which slices of $\hat S_t$ the operator $\Pi_{\pi}$ retains.**

## 1. Three policy families, and what each eats

**Family A: Engineered-State Policy Head**—MLP / GRU / small transformer, hand-designed state vector. Its lineage is classical RL (the PPO / SAC line on MuJoCo / Isaac Lab), and it is still the default architecture in sim-to-real and industrial robot learning today. Input convention: a fixed-length $s_t \in \mathbb{R}^d$ whose fields are hand-picked joint angles, end-effector pose, wrench readings, optional contact flags. It is the **most contract-friendly** of the three—because fields are already explicit, adding `availability` / `validity` / `age` / `observability` is just bookkeeping. The cost is a low expressivity ceiling, poor cross-task transfer, and hand-crafting the state schema is its own art.

**Family B: Diffusion Policy and Flow-Matching Policy**—policy as a denoising / flow-matching generative model, consuming image latents + proprio + language goal (sometimes without language). Chi et al. 2023 (Diffusion Policy) reads $k$ RGB frames + joint proprio and outputs an action chunk; Zhao et al. 2023 (ACT / ALOHA) uses a transformer encoder-decoder with a CVAE head or, in later variants, flow matching. Its killer feature is a genuinely multimodal **continuous action distribution**—but its handling of the contract is **implicit**: state is squashed into an image-encoder latent, proprio is concatenated into the noise-conditioning, **there is no dedicated slot for hypothesis, provenance or age**. Once the contract crosses the input boundary, it is flattened into a single vector.

**Family C: VLA (Vision-Language-Action)**—RT-2 / OpenVLA / π0 / Octo. Reads image tokens + language tokens, outputs action tokens. RT-2 and OpenVLA are autoregressive over discretized action tokens (actions become tokens from a 256-way vocabulary or similar); π0 does flow matching over continuous action chunks with a VLM backbone for conditioning. VLA breaks the contract the hardest: **language conditioning and image tokens are shoved through the same transformer, and the action side is either a vocabulary of discretized tokens or a flow-matched continuous chunk**. Structured fields like `wrench.frame.reference_point` from upstream must be either paraphrased into natural language ("low grip force, contact on pad") before the tokenizer, or dropped entirely.

The three side by side:

| Family | State representation | Action representation | Contract breakage surface | Expressivity ceiling | Main cost |
|---|---|---|---|---|---|
| A · Engineered-State Head | Explicit $s_t \in \mathbb{R}^d$ | Continuous mean + variance (or SAC Q-head) | Smallest, fields are hand-written | Low | Manual state design, poor transfer |
| B · Diffusion / Flow | image latent + proprio concat | Continuous action chunk (denoised / flow) | Breaks provenance / age / hypothesis | Medium–High | Multi-step inference, unstable training |
| C · VLA | Discretized image / language token | Discrete token or flow-matched action | Breaks frame / reference_point / observability | High (cross-task reuse) | Expensive to train, tokenization is irreversible |

The "breakage surface" column is what matters—**it decides which fields of the contract still mean anything inside the policy**. The fields defined in §6.1 of the multimodal-fusion piece—`availability`, `validity`, `age`, `provenance.negative_evidence`, `wrench.count_uncertainty`, `contact.geometry.observability`—move through Family A basically intact, only partially through Family B via concat, and in Family C most of them either have to be written into the prompt or dropped.

A caveat before we move on: **this is not a ranking of which family is best**. Family A still wins the low-dimensional control baseline; Family C is irreplaceable for open-semantic-conditioned tasks. The question is not which one to pick; **the question is whether, after picking, the policy has a clear interface convention telling you how far down the contract it reads**.

## 2. "State" means different things to different policies

This section clears terminology. "State" carries at least five distinct meanings in the embodied-AI literature, and the choice of policy family is often the choice of which one is assumed by default:

**$\pi_{\mathrm{obs}}$: raw observation**—images, point clouds, force / torque readings as raw streams. This is the **input form** of most imitation-learning pipelines, but usually not the state the policy **actually uses internally** (it will be encoded).

**$z_t = f_\phi(o_{:t})$: encoded latent**—the low-dimensional representation after an encoder. Diffusion Policy, VLA image tokens and world-model RSSM states all belong here. By this stage the contract has already been **projected once**, and observability / provenance are typically already gone.

**$b_t$: belief / posterior**—the explicit belief state in the POMDP line. Structurally the closest match to the contract's "multi-hypothesis + posterior weight" object. Mainstream VLA / Diffusion Policy do not model belief explicitly—it is approximated implicitly by the encoder.

**$s_t = (b_t, \pi_t, q_t, h_t)$: the allocation state defined in Part 1 of the Sim-to-Real trilogy**—belief + policy + budget + hardware. This is the **decision-layer** state, not the policy's input state. It requires the contract to expose "which policy is running, how much real-rollout budget is left"—fields with no interface in VLA today.

**$\hat S_t$: structured state (contract)**—the contract object defined in the multimodal-fusion piece.

The full chain looks like this:

$$\pi_{\mathrm{obs}} \;\xrightarrow{\;\text{encoder}\;}\; z_t \;\xrightarrow{\;\text{abstraction}\;}\; \hat S_t \;\xrightarrow{\;\text{posterior}\;}\; b_t \;\xrightarrow{\;\text{allocation}\;}\; s_t$$

The real question on the policy side is **not "can I eat a longer token sequence"**, but **"how far along the chain $\pi_{\mathrm{obs}} \to z_t \to \hat S_t \to b_t \to s_t$ am I willing to commit"**. Family A stops at $\hat S_t$ (explicit structured state, hand-written fields); Family B stops at $z_t$ (encoder latent); Family C effectively stops at the tokenizer one step past $\pi_{\mathrm{obs}}$—image tokens are compressed observations without any state abstraction.

**A common misuse of the term "multimodal fusion" on the policy side** is treating the cross-attention step $\pi_{\mathrm{obs}} \to z_t$ as if it already **were** multimodal state estimation. It is not. Real state abstraction requires fields of $\hat S_t$ to hold consistent semantics **across sensor families**, and to be jointly readable by four consumers—controller / policy / world model / diagnostics. The multimodal-fusion piece already established that convention in its §6; the job here is to **ask again from the policy side: which station on this chain is each family willing to read to**.

## 3. Four interface-mismatch failure modes

Once the three projections of §1 land on concrete policies, contract semantics manifest as **four concrete failure modes**. These are not theoretical worries—**they are deployment-level breakages you will hit**.

**Failure 1: Mode Collapse via Averaging**. In the contract, the same physical quantity may carry multiple hypotheses ("is this track_id the same object", "is this contact a pad or an edge") with distinct posterior weights. If the policy merges them at the latent by averaging, the multimodal posterior is flattened into a unimodal one. Symptom: the policy emits an "average" action across "two possible worlds", neither of which is optimal—yet the loss curve looks fine, **because the "average action" is often the most common action in the training distribution**. This is not a parameter-count problem, **it is a shape-of-$\Pi_\pi$ problem**. As long as the projection is L2 / mean, mode collapse is the default outcome.

**Failure 2: Semantic Drift via Tokenization**. Contract fields like frame—`orientation_frame: "tool_flange"`, `reference_point: "contact_center"`, `convention: "right-handed"`—drift when a tokenizer turns them into a token sequence: the model can learn the **distribution** of the "contact_center" token but cannot learn the hard constraint that "changing reference_point means $\tau$ must be corrected via $\tau_{p_2} = \tau_{p_1} + (p_1 - p_2) \times f$". The consequence is that the policy fails mysteriously on edge cases where the frame flips: **the token distribution may barely change, but the wrench value must be recomputed**. The similarity an attention layer learns cannot capture physics-level convention changes. **A concrete shape**: on real deployment, switch a wrench's `reference_point` from `sensor_flange` to `contact_center`—the preprocessing pipeline changes, the token sequence looks almost identical (both "sensor_flange" and "contact_center" appear as frame tokens in similar contexts), but the value of $\tau$ changes substantially via the transport theorem. A VLA trained separately on these two "look-alike" datasets will converge to **similar but both-wrong** policies, because what it learned is the linguistic co-occurrence of frame tokens, not the functional correction that a frame change imposes on the physical quantity. This class of bug only surfaces **after deployment**, during cross-dataset reuse.

**Failure 3: Temporal Alignment Broken by Concat**. In the contract, each channel's `age` is explicit (say proprio 1 ms, F/T 5 ms, vision 100 ms, tactile 30 ms). If the policy only sees a concat vector, the $\Delta t$ distribution is lost. Symptom: the policy decides using vision's "world from 100 ms ago" alongside tactile's "contact from 30 ms ago" as if they were synchronous. The multimodal-fusion piece already argued that **timestamp sync ≠ causal sync**—and neither is decision-time causal consistency. **What concat hides is not $\Delta t$, it is the semantics of $\Delta t$**—even if you timestamp every channel, if the policy never reads the timestamp into a decision variable, the timestamp is decoration.

**Failure 4: Safety-Blindness to Validity vs Staleness**. The contract explicitly separates availability (does the channel have data right now), validity (is the data still valid—say, calibration not expired), and age (how stale it is). If the policy only looks at numbers, it merges "stale but valid" with "missing but valid", and treats "invalid after calibration drift" as the same degradation as "sensor disconnected". The multimodal-fusion piece's §8.5 degradation chain already pulled these apart—**masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption**—and if the policy does not read the chain at its input, both training-time augmentation and inference-time guardrail will attach at the wrong place.

All four are **interface problems**, not model problems—swapping backbones does not fix them, only making $\Pi_\pi$ explicit does. This distinction matters because it **decouples** two decisions that are often conflated: "should I move to a bigger VLA" versus "should I rewrite the policy-side interface".

## 4. Three contract-read primitives the policy needs

Following §3, what the policy side genuinely needs is **three primitives**—not a bigger transformer, not more data.

### 4.1 `mode_select` (how a hypothesis set is read out)

Given a multi-hypothesis posterior in the contract, the policy must pick one of four explicit readout strategies: MAP (take the highest-weight hypothesis), sample (draw from the posterior, encouraging exploration), expected-mixture (retain the mixture, let downstream heads attend), or ECE-preserving top-$k$ (keep $k$ calibrated hypotheses). The difference between MAP and expected-mixture **is** the difference between whether Failure 1 happens or does not.

$$\text{read}\!\big(\{(\mu_i, \Sigma_i, w_i)\}_{i=1}^{K}\big) \;=\; \left\{\begin{aligned}
&\mu_{\arg\max_i w_i} && \text{(MAP, drop low-weight hypotheses)}\\
&\textstyle\sum_i w_i\, \mu_i && \text{(mean collapse, default danger zone)}\\
&\mu_i + L_i \epsilon,\;\; i \sim w,\;\; \epsilon \sim \mathcal{N}(0, I) && \text{(posterior sample)}\\
&\big\{(\mu_i, \Sigma_i, w_i)\big\}_{i \in \mathrm{top}\text{-}k} && \text{(ECE-preserving top-}k\text{)}
\end{aligned}\right.$$

The key point: **"mean collapse" is the default behavior of most policies, because it corresponds to the simplest concat + MLP treatment**. A contract-preserving policy must **write "no mean-pooling here" explicitly into $\Pi_\pi$**, otherwise your carefully maintained multimodal posterior evaporates in a single mean-pool. Families B and C are especially prone—transformer mean-pooling or class-token readout is a natural collapse operator.

### 4.2 `age_gate` (staleness trust decay)

When a channel $c$ with `age` $a_c$ enters the policy, its contribution to the current decision must decay explicitly. The simplest form is multiplicative trust $\tau_c = \exp(-a_c / \bar a_c)$, where $\bar a_c$ is the channel's time constant; a stricter form feeds $a_c$ as a conditioning variable and lets the policy learn its own decay. The former is **an interface-level physical constraint**; the latter is **a training objective**. The Sim-to-Real Part 1 already separated $\Delta_{\mathrm{queue}}$ from $\Delta_{\mathrm{processing}}$—$a_c$ is the explicit form of $\Delta_{\mathrm{queue}}$.

$$x_c^{\pi} \;=\; \underbrace{\tau_c(a_c)}_{\text{staleness trust decay}} \;\cdot\; \mu_c \;\cdot\; \mathbb{1}\!\big[\text{validity}_c = \mathrm{OK}\big] \;\cdot\; \mathbb{1}\!\big[\text{availability}_c = \mathrm{OK}\big]$$

Multiplying those three $\mathbb{1}$s is exactly the operation of **moving the three orthogonal axes `availability` / `validity` / `age` defined in the multimodal-fusion piece's §6.1 straight into the policy input**. Drop any one of them and you hit Failure 4. The specific shape of $\tau_c$ (exponential decay / sigmoid / step) is not what matters—**what matters is that it exists explicitly**, otherwise channel staleness lives only in dataset metadata where the policy cannot see it.

### 4.3 `provenance_condition` (source as conditioning, not as noise)

The contract carries `provenance.contributing_mask` (which sensors contributed to this field), `negative_evidence` (which sensors expected something and did not see it), `correlated_with` (which fields have known correlations and must not be double-counted). All three must enter the policy as **conditioning variables**—not as a noise model, not as dropout.

Two implementation paths:

- **Hard conditioning**: concatenate `contributing_mask` directly into the policy input. Families A and B can both do this (a few extra vector dimensions).
- **Attention biasing**: convert `correlated_with` into an attention mask / bias, so a transformer does not double-count correlated evidence when attending. Family C (VLA) fits this naturally, since transformers already expose an attention interface.

$$\mathrm{Attn}'_{ij} \;=\; \mathrm{Attn}_{ij} \;-\; \beta \cdot \mathbb{1}\!\big[\text{fields}_i \text{ correlated\_with } \text{fields}_j\big]$$

What this subtracts is not "attention weight", it is **double-counting**—mirroring exactly the caveat (ii) in §6.2 of the multimodal-fusion piece: **proprio-derived $\hat F_{\mathrm{ext}}$ and F/T-measured wrench are structurally correlated, and naive fusion double-counts them**. By the same token, if the policy sees proprio and F/T and $\hat F_{\mathrm{ext}}$ all at once, after a three-way concat the attention layer will treat these as **three independent pieces of evidence**, when in fact only two exist. **A concrete shape**: $\hat F_{\mathrm{ext}} = \arg\min_F \|\tau_{\mathrm{res}} - J^{\top} F\|_W^2 + \lambda R(F)$ is an estimate of the external contact force solved from joint torque residuals; $\tau_{\mathrm{res}}$ has already used proprio, and F/T re-uses the same joint information—so the $correlated\_with$ relation among the three is **structural, not empirical**. Skip the attention biasing in §4.3 and the policy will count **the same information three times**, substantially over-estimating confidence in low-friction / high-contact scenarios, and directly causing the safety-filter misses described in §5.3. **This `correlated_with` is not learned—it is read out of the contract.** That is the whole point of treating provenance as conditioning rather than as noise.

### 4.4 The relationship among the three primitives

**`mode_select` handles multi-hypothesis in space**, **`age_gate` handles heteroscedastic $\Delta t$ in time**, **`provenance_condition` handles correlation in causality**. Drop any one of the three and at least two of §3's failures will recur. The three together are **not the complete answer to interface design** either—observability / identifiability, frame convention, and contact set each have their own more specialized readouts (the multimodal-fusion piece's §7 and §8.6 have discussions); this piece only handles the three most likely to be **silently broken by the policy**.

## 5. Training-time and Deployment-time consequences

Once the policy input side adopts the three primitives, training objective, augmentation, safety filtering and evaluation all have to follow. **Interfaces are not free**—but the changes are **local and manageable**.

### 5.1 Where loss attaches

VLA's autoregressive CE, Diffusion Policy's score matching, Flow Matching's velocity regression, classical heads' Gaussian NLL / SAC Q-target—**all of these cover only action generation**, not the contract-side fields. Contract-preserving training requires an **auxiliary loss**: make the policy's intermediate representation predict `age`, `validity`, `observability`, and hypothesis posteriors—not so the policy "understands the contract better", but so **it cannot shortcut to minimum loss by flattening the contract**.

$$\mathcal{L}_{\mathrm{total}} \;=\; \mathcal{L}_{\mathrm{action}} \;+\; \alpha_1\, \mathcal{L}_{\mathrm{calibration}} \;+\; \alpha_2\, \mathcal{L}_{\mathrm{hypothesis}} \;+\; \alpha_3\, \mathcal{L}_{\mathrm{validity\text{-}pred}}$$

Motivations for each auxiliary: $\mathcal{L}_{\mathrm{calibration}}$ forces the policy's action distribution to agree with its own claimed uncertainty; $\mathcal{L}_{\mathrm{hypothesis}}$ forces its top-$k$ to cover the true hypotheses in data (contact mode, object identity, etc.); $\mathcal{L}_{\mathrm{validity\text{-}pred}}$ forces it to read $\mathbb{1}[\text{validity}_c]$ instead of "guessing" from numerical distributions. The α weights are a tuning decision; **but the structure is an interface decision**—you cannot delete all auxiliaries just because α is hard to set.

### 5.2 Augmentation must be classified along the degradation chain

The multimodal-fusion piece's §8.5 insisted on **masking ≠ missing ≠ staleness ≠ latency ≠ bias ≠ corruption**—six degradations, each with a distinct **causal origin** and a distinct **downstream read**. If training-time data augmentation uses only one of them (the most common being random masking), the policy learns all six as one, and at deployment cannot distinguish "this channel is broken today" from "this channel is heavily delayed". The aug generator must **channel per class of the chain**, each class with its own distribution, and each mapped to the specific form of $\mathbb{1}$ and $\tau_c$ from §4.2.

Concretely: attach a degradation label to every training episode, carry it explicitly into the policy input, and let the aug pipeline sample conditioned on the same label. **This is not curriculum, this is conditioning**—curriculum is "learn easy first, then hard"; conditioning is "let the policy know which degradation it is currently under".

### 5.3 Safety filter and contract interface

Constraint layers (CBF / shield / runtime verifier) should read the contract's `observability`, not the policy's $\mu$. Reason: the domain on which a constraint holds depends on **observability**, not on what the policy believes. The multimodal-fusion piece's §7 already established that **track_id is a hypothesis**—so a safety filter cannot trust track_id matching alone; it must also see whether the hypothesis posterior is stable. Whether two tracks merge or split directly determines the trustworthiness of "obstacle distance".

Concretely, the safety filter should read: `observability` (is the quantity this constraint depends on observable right now), `validity` (has calibration expired—expired readings must not trigger a constraint), `negative_evidence` (an observation that should have been seen but was not—"the LiDAR swept nothing in this sector"). **If those three do not enter the safety filter, the filter will make decisions on stale or invalid data**—more dangerous than the policy itself.

### 5.4 Echo with the Sim-to-Real Part 3 evaluation

Among the three sim-utility dimensions (prediction / ranking / decision), policy-side evaluation primarily looks at **decision**—but a fourth dimension, **contract preservation**, must be added: how much performance drops when the contract is destructured (say, all hypotheses merged to mean, all `age` flattened) is a **lower bound** on how much the policy actually depends on the contract. That dimension is what §6's three metrics measure.

## 6. Evaluation: three interface-level metrics on the policy side

Corresponding one-to-one with §4's three primitives:

**Metric 1 (Hypothesis Preservation Score, HPS)**—at any given moment, the contract holds $K$ hypotheses. HPS measures how many "locally optimal action clusters" the policy's output covers across the $K$ worlds in which each hypothesis is true.

$$\mathrm{HPS} \;=\; \frac{1}{N} \sum_{n=1}^{N}\, \max_{k}\, \Pr\!\big[\pi_\theta(x_t^{\pi}) \in \mathcal{A}^{*}_k \,\big|\, H_k \text{ is true at } n\big]$$

$\mathcal{A}^{*}_k$ is the set of optimal actions under hypothesis $H_k$. The HPS gap between MAP readout and posterior-sample readout quantifies how severe Failure 1 is.

**Metric 2 (Staleness Discrimination Score, SDS)**—construct two test sets: group A with every channel fresh, group B with a few channels deliberately made visibly stale. If the policy's output distribution is nearly identical on A and B, SDS ≈ 0—meaning it is not reading `age` at all, and Failures 3 + 4 are confirmed. SDS is quantified via KL divergence:

$$\mathrm{SDS}_c \;=\; D_{\mathrm{KL}}\!\Big(\pi_\theta(\cdot \mid o, a_c^{\mathrm{fresh}}) \;\Big\|\; \pi_\theta(\cdot \mid o, a_c^{\mathrm{stale}})\Big)$$

A $\mathrm{SDS}_c$ near zero means "channel $c$'s staleness has no effect on the policy's output"—the most direct ablation at the interface layer.

**Metric 3 (Provenance-Conditioning Effect, PCE)**—remove `correlated_with` from the policy input and measure how much performance drops in double-counting-sensitive scenarios (say, proprio + F/T fused $\hat F_{\mathrm{ext}}$).

$$\mathrm{PCE} \;=\; J\big(\pi_\theta \mid \text{provenance}\big) \;-\; J\big(\pi_\theta \mid \text{provenance} = \varnothing\big)$$

High PCE means the policy is genuinely using provenance; low PCE means it is treating provenance as decoration. PCE also tells you whether $\beta$ (the attention bias strength in §4.3) is tuned correctly.

What the three metrics share: **none of them is accuracy; all three are interface-layer ablations**—echoing the Sim-to-Real Part 3 §5 approach of "evaluate along three orthogonal axes". They also **cannot replace** any end-to-end success rate—they measure the **policy side's degree of reading** the contract, not the **policy side's performance ceiling**.

## 7. A minimum executable interface sketch

Turn §4's three primitives into a Python class skeleton. **Not to propose a specific policy, but to state a readable interface convention**.

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
        `mode / staleness / provenance` are policy-specific knobs, not global defaults.
        Same contract, different knob values across VLA and diffusion policy—that is fine.
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
            # validity / availability indicators, always attached
            v = v * h.validity_ok * h.available
            # --- Primitive 3: provenance_condition ---------------------
            if provenance == "hard":
                v = concat(v, h.contributing_mask, h.negative_evidence_summary)
            # attention_bias branch handled downstream: the backbone reads
            # self.contract.correlated_with and injects a bias mask
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

Three caveats written explicitly:

- **(i)** This is not the only readout. The three knobs (`mode / staleness / provenance`) are policy-specific—same contract, but VLA and Diffusion Policy should be running different knob values; Family A and Family B also disagree on optimal `staleness`.
- **(ii)** This interface **only handles the input side**. If §5's loss / augmentation / safety-filter follow-through is not implemented, no matter how cleanly $\Pi_\pi$ is written, training dynamics will bypass it—loss will find the laziest path to "flatten the contract".
- **(iii)** `provenance="attention_bias"` has an implementation path only on transformer-family backbones; MLP heads take the `hard` route.

## 8. Three closing claims

> **Claim A**: **A good policy must read disagreement, not merely react to it.** The multimodal-fusion piece said a good multimodal system must **represent disagreement**; this piece adds that a good policy must **respect that representation at the readout level**, and cannot silently average disagreement away through a projection. The difference between "read" and "react" is precisely the difference between mean collapse and ECE-preserving top-$k$ in §4.1.

> **Claim B**: **Action space is not just kinematic — it is semantic.** Frame / reference_point / convention at the action layer are physical quantities, not metadata; tokenization cannot erase their semantics, it can only turn them into a silent bug with "similar distribution, wrong value". This is the closing formulation of §3's Failure 2.

> **Claim C**: **Contract compliance is measurable at the policy boundary, not only at the estimator boundary.** The multimodal-fusion piece's §8 Interface Property Benchmark is the estimator side; §6's HPS / SDS / PCE here are the policy side. Only together do the two turn the contract into a **verifiable object** rather than an **interface proposal**.

A closing line: **stop using the word "fusion"**—it asks about the timing axis ("when to merge") and about the semantics axis ("into what to merge"); the multimodal-fusion piece and this piece together decompose the second question into **what upstream ships + what downstream reads**. The first question (timing) is already largely answered by the §1 comparison of three families—**timing is a consequence of interface, not a decision variable of interface**. If this point sticks, both articles earn their keep.

## Sources

arXiv IDs below have been web-verified; journal-only references do not carry arXiv. Grouped by the section they support.

### A · VLA family (supports §1 Family C, §3 Failures 1–2, §5.1 auxiliary loss)

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) (actions expressed as text tokens, co-fine-tuned with a VLM · canonical form of §3 Failure 2, semantic drift)
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) (open-source VLA baseline; multi-camera / depth / proprioceptive state encoding in the public config, but "supports as input" ≠ "reads as far as the contract")
- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) (VLM backbone + proprio token + noisy action chunk + flow matching · a concrete realization of §4.1 mode_select)
- Octo Model Team, *Octo: An Open-Source Generalist Robot Policy*, RSS 2024 · [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) (transformer-based readout · a reference point for the §4.3 attention_bias path)

### B · Diffusion / Flow-Matching Policy (supports §1 Family B, §5.1)

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 · [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) (RGB stack + proprio concat + denoising · field evidence for §3 Failure 1 mode collapse and §3 Failure 3 temporal-alignment-broken)
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT / ALOHA), RSS 2023 · [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) (CVAE + transformer encoder-decoder, chunked action · an intermediate form between Families B and C)
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 · [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) (velocity regression for continuous action chunks · a concrete form of §5.1 $\mathcal{L}_{\mathrm{action}}$)

### C · Uncertainty, Calibration and Belief-space References (supports §4.1, §5.1, §6)

- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) (modern networks over-confident, temperature scaling origin · motivation for §5.1 $\mathcal{L}_{\mathrm{calibration}}$)
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels* (PlaNet / RSSM), ICML 2019 · [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) (deterministic + stochastic latent, an engineering realization of multi-hypothesis · a reference for §4.1 read)
- Hafner et al., *Mastering Diverse Control Tasks through World Models* (DreamerV3), Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) (discrete + continuous mixed latent, KL balancing · an adjacent route to §4.1's ECE-preserving top-$k$)

### D · SAC / PPO Baseline for Family A (supports §1 Family A)

- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, ICML 2018 · [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) (Gaussian NLL / max-entropy policy loss · §5.1 $\mathcal{L}_{\mathrm{action}}$ in Family A form)

### E · Predecessors (this piece's interface with 9/13, 9/14, and Sim-to-Real Part 3)

- This blog, *Piling Modalities Together Is Not Understanding: What Robot Multimodal Fusion Lacks Is an Interface, Not a Model* · `/en/articles/2026-09-14-multimodal-fusion-interface/` (Structured State Contract definition, Interface Property Benchmark, degradation chain · §3–§6 of this piece build directly on it)
- This blog, *The Hand Robots Don't Have: Tactile and Force Sensing in Embodied AI* · `/en/articles/2026-09-13-tactile-force-sensing/` (four parallel force-control paradigms, closed-loop value · the historical watershed between Family A and Family B)
- This blog, *Sim-to-Real Methodology (III)* · `/en/articles/2026-09-12-sim-to-real-evaluation-protocol/` (three-tier evidence stack, decision utility, allocation protocol · §5.4 and §6 reuse its evaluation spine)
- This blog, *Sim-to-Real Methodology (I)* · `/en/articles/2026-09-10-sim-to-real-methodology/` (allocation state $s_t = (b_t, \pi_t, q_t, h_t)$, $\Delta_{\mathrm{queue}}$ vs $\Delta_{\mathrm{processing}}$ · §2 and §4.2 attach directly to those definitions)

---

> **Related reading**
>
> - [Piling Modalities Together Is Not Understanding: What Robot Multimodal Fusion Lacks Is an Interface, Not a Model](/en/articles/2026-09-14-multimodal-fusion-interface/) — this piece's predecessor, which stood the upstream deliverable up as a Structured State Contract
> - [The Hand Robots Don't Have: Tactile and Force Sensing in Embodied AI](/en/articles/2026-09-13-tactile-force-sensing/) — the historical watershed between §1's Family A and Family B, and the four parallel force-control paradigms
> - [Sim-to-Real Methodology (III)](/en/articles/2026-09-12-sim-to-real-evaluation-protocol/) — §5.4 and §6 reuse its three-tier evidence stack and utility dimensions
> - [VLA and World Models: Where Two Roads Diverge and Converge](/en/articles/2026-09-07-vla-world-models/) — macro backdrop for §1 Family C, and the other side of "world models do not naturally belong to sim-to-real"
> - [A Quick Tour of the VLA π Family](/en/articles/2026-09-05-vla-pi-family/) — a concrete cut at π0, π0.5 and flow-matching action heads
> - [What Is a VLA Model? A Complete Primer](/en/articles/2026-09-03-vla-deep-dive/) — the entry-level version of Family C; this piece assumes you have already read it
