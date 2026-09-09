---
title: 'Embodied AI Sim-to-Real Methodology (II): Four Intervention Lenses and Two Reformulation Routes'
slug: "2026-09-11-sim-to-real-intervention-lenses"
date: 2026-09-11
draft: false
categories: ["Embodied AI", "Training Methods"]
tags: ["Embodied AI", "Sim-to-Real", "System Identification", "Domain Randomization", "Differentiable Simulation", "Residual Physics", "Domain Adaptation", "Real-world Fine-tuning", "World Model", "Co-training"]
description: 'Trilogy - Methods. Read SI / DR / DA / FT as four composable intervention lenses (Model x Data x Representation x Optimization), not four exclusive choices; add World Model and Sim-and-Real Co-training as two routes that loosen the environment-generating-process assumption. Each lens: what mismatch it handles, mechanism, failure boundary. Theory spine in Part 1, evaluation and protocol in Part 3.'
toc: true
related_articles:
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-12-sim-to-real-evaluation-protocol
  - 2026-09-13-tactile-force-sensing
  - 2026-09-14-multimodal-fusion-interface
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
---


> **Recap** (from Part 1): This is the second piece of the Sim-to-Real Methodology trilogy. The core framework is established in [Part 1](/en/articles/2026-09-10-sim-to-real-methodology/) — reality gap is recast as a downstream discrepancy under the quadruple $(\pi,\; \mathcal{E}_{\mathrm{shared}},\; M_{\mathrm{sim}},\; M_{\mathrm{real}})$, the error budget becomes a state-conditioned sequential allocation, and the decision unit is the intervention action $m_t^* = \arg\max\, Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t)$. Notation: $s_t = (b_t, \pi_t, q_t, h_t)$, $\mathcal{M}_t^{\mathrm{feasible}}$ (feasible action set), $MV$ (efficiency readout, not decision rule), $\mathrm{CVU}$ (one-step continuation uplift). If you have not read Part 1, start there.

## Four intervention lenses (composable analytical dimensions)

SI, DR, DA, and FT **are not peers at the same abstraction level** — SI is model calibration, DR is distribution manipulation, DA is representation alignment, FT is optimization strategy. Together they form **four composable intervention lenses** (analytical decomposition, not domain-recognized ontology):

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

"$\times$" here is a **combinatorial space**, not mathematical orthogonality — DR touches Model / Observation / Distribution, DA can happen at input / feature / latent / policy / output, and "DA = the Representation axis" is only one abstraction layer of this article.

**The tool criterion is not "systematic → SI, random → DR"** — the useful partition is the continuum "**point estimate → posterior → robust randomization**". SI can do **point calibration** or, further, produce a **posterior**; we start with the point-estimate form:

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; \mathcal{L}_{\mathrm{ID}}\big(D_{\mathrm{real}},\ f_{\mathrm{sim}}(\cdot\,;\,\phi)\big)$$

$\mathcal{L}_{\mathrm{ID}}$ may be trajectory prediction / one-step transition error / force-torque residual / likelihood — **classical SI objectives are typically parameter estimation or transition / observation prediction-error minimization, without an explicit trajectory-distribution-matching term**. SI handles **parameterizable model mismatch**: dynamics residual, contact / friction coefficients, latency, camera extrinsics — when the gap lies outside the model class (unobserved long tail, semantic-level visual difference), SI runs out of leverage and one must switch to DR / DA / WM.

| Nature of the mismatch | More natural tool |
| --- | --- |
| Parameterizable + identifiable | System Identification (point estimate $\hat\phi$) |
| Parameterizable but only uncertainty available | Bayesian / posterior SI → posterior-guided DR |
| Parameterizable but hard to identify / high uncertainty | Domain Randomization |
| Difficult to express with low-dimensional physical parameters, but has a structured residual | Residual learning |
| Observation / appearance mismatch | Domain Adaptation |
| Policy still has systematic residual on the target domain | Fine-tuning |

Key point: **"not precisely identifiable" $\neq$ "no knowledge at all"** — with a posterior $p(\phi \mid D_{\mathrm{real}})$, the natural move is $\phi \sim p(\phi \mid D_{\mathrm{real}})$ for **posterior-guided randomization**, stitching SI and DR into a continuous spectrum.

### Axis A — Model: system identification, differentiable simulation, and residual physics

This axis handles $\Delta_{\mathrm{model}}$ and contains three **distinct levels** usually bundled into "differentiable simulation is stronger SI":

$$y_t \;=\; \underbrace{g_{\mathrm{physics}}(x_t,a_t;\phi)}_{\text{parameterizable physics}} \;+\; \underbrace{r_\theta\big(\psi(x_t,a_t)\big)}_{\text{residual}} \;+\; \epsilon_t$$

**Only a representative parameterization.** $y_t$ may be the next state $x_{t+1}$, a contact impulse, an acceleration, a deformation field, or another observable; $\psi$ is the residual's input view. The additive state-transition form is one instance; soft-robot residual deformation field, contact-impulse residual, and state residual are not the same mathematical object.

- **Differentiable simulation** answers "how to optimize the model" — a gradient path through simulator parameters / states / controls, an **optimization interface** for identification and trajectory optimization (**not itself system identification**). DiffTaichi (Hu et al., ICLR 2020, 1910.00935) and Interactive Differentiable Simulation (Heiden et al., arXiv 2019, 1905.10706) are representative implementations.
- **System identification** answers "which parameter to optimize" — real workflow often **real → identify → sim → train → real**, more accurately **real-to-sim-to-real**.
- **Residual physics** answers "who explains what the model missed" — instead of forcing calibration of $\phi$, let a network learn $r_\theta$ to fill the gap.

$r_\theta$ is only **unified notation**: the actual residual may be defined on state transition, force, acceleration, contact impulse, deformation field, or other latents.

A make-or-break point hidden behind "differentiable": **differentiability solves the optimization interface, not model-class correctness**. If the contact model simply cannot express a real phenomenon, gradients only give "the optimum under a wrong model." **Commonly ignored**: contact-mode switches and complementarity constraints in collision / friction produce **nonsmooth or piecewise-smooth dynamics** — even when $\partial f/\partial\phi$ exists, no guarantee the gradient is stable (it may be discontinuous, high-variance, or simply ill-defined — not necessarily vanishing), that the gradient at mode transitions is meaningful, or that it beats derivative-free optimization. Soft-contact modeling and smooth relaxation are common engineering workarounds.

SI has two further pitfalls. **First, $p_{\mathrm{real}}(\tau)$ is essentially never directly accessible** — only a finite set of real trajectories. **Second, parameters existing $\neq$ identifiable** — identifiability also depends on excitation and sensor observability: mass / damping / stiffness can produce nearly identical observable trajectories under some excitations and cannot be estimated independently.

Residual physics needs a narrowed boundary: a **common applicability range** is where $f_{\mathrm{physics}}$ still provides a **useful structural inductive bias** and the residual makes only a bounded correction on the target distribution — soft robots (Gao et al., RA-L 2024, 2402.01086) and buoyancy-assisted legged robots (Sontakke et al., 2023, 2303.09597) are exactly "trunk physics counts, local residual stable." If $f_{\mathrm{physics}}$ is fully wrong and the residual has to carry the whole dynamics, better learn a model outright. $r_\theta$ **is not naturally "the missing physics"** — an unrestricted additive residual absorbs sensor bias / actuator error / timing / calibration / reward mismatch into an **error sponge** that fits the training distribution and falls apart OOD; so it needs structural constraints (low-dim / sparse / force or acceleration scale / physical priors / active only in specific contact regimes). Under those structural conditions, differentiable simulation is usually the first thing worth evaluating.

Also **confounding** between $\phi$ and $r_\theta$ — a flexible enough residual absorbs effects belonging to $\phi$, making $\hat\phi$ meaningless; identifiability requires $f_{\mathrm{physics}}$ and $r_\theta$ to be distinguishable in data (regularization / scale separation / structural constraints).

### Axis B — Data distribution: domain randomization and its family

This axis does not chase some "most accurate" $p_{\mathrm{real}}$; it makes the policy robust to a family $\{\phi\}$. **Tobin et al. (1703.06907) is the classic representative starting point of modern deep-vision / robotics sim-to-real literature** (the idea of domain randomization predates it; the point is its representative position in end-to-end visual policy transfer). Peng et al. (1710.06537) pushed randomization into dynamics; OpenAI in-hand manipulation (Akkaya et al., 1808.00177) nearly took DR to its extreme — **absorbing difference not through precise calibration but through "a randomization range wide enough."**

A commonly mis-written intuition: **DR is not an "implicit ensemble"** — what is trained is a **single** shared policy $\pi_\theta$, with the objective

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

More precisely: **DR is a population-level optimization over a family of environment models**. The formula above is a **baseline abstraction of risk-neutral average-case DR**; robust / adversarial DR instead uses $\max_\theta \min_{\phi \in \Phi} J(\pi_\theta; \phi)$, CVaR, or other risk-sensitive forms. A common engineering heuristic is **support inclusion**, $\mathrm{supp}(p_{\mathrm{real}}) \subseteq \mathrm{supp}(p_{\mathrm{DR}})$ with adequate density in the real-typical region — read it as a **conservative / sufficient coverage proxy**, not as a necessary condition for transfer success: a policy can remain robust outside its training support. **But this carries a deeper premise**: real dynamics must be expressible by the same $\phi$-parameterization; if the sim's model class cannot express the phenomenon, $\phi_{\mathrm{real}}$ is undefined and the support statement **collapses at the root** — the problem becomes **model-class** uncertainty, not "DR not wide enough." (Parameter uncertainty = "where in $\phi$-space reality sits," addressable by posterior or DR; model-class uncertainty = "can this $\phi$-parameterization express real dynamics," not solvable by widening ranges.) **Main conclusion**: parameter-space support is a design proxy, **but the deployment-relevant object is the policy-conditioned occupancy it induces** — from $p_{\mathrm{DR}}(\phi)$ through $d_{\mathrm{train}}^{\pi}(s,a,\text{mode})$ and $d_{\mathrm{real}}^{\pi}(s,a,\text{mode})$, their overlap governs transfer. Parameter-space differences are **not automatically sufficient for deployment failure**.

One layer down: **DR is not choosing scalar ranges, it is designing a joint distribution** — **when the true parameters have significant joint dependency** (payload co-varies with actuator regime, temperature with motor resistance / friction / battery), independent sampling wastes finite budget on low-deployment-relevance or physically inconsistent combinations; independent uniform DR remains a reasonable approximation when the true parameter distribution is close to independent. $p(\phi_1,\phi_2) \neq p(\phi_1)p(\phi_2)$ is common but not universal. The question returns to allocation: **the randomization distribution must align with evaluation and objective**; overly wide or task-irrelevant randomization hurts sample efficiency, but in robust settings a larger uncertainty set can help. **"Wider = more conservative" is not a universal rule; shape and alignment matter.**

"Adaptive / Automatic DR" is a family rather than a single method: curriculum / adversarial / automatic / posterior-based sampling / performance-driven range adaptation — the common thread is **avoiding over-randomization from the start**.

### Axis C — Observation / Representation: domain adaptation and observation translation

This axis handles $\Delta_{\mathrm{obs}}$, aligning sim and real **at the observation / representation layer**. **"Representation" is this article's abstraction** — DA acts on input / feature / latent / output / policy / dynamics model. Mechanisms include feature-level adapters, latent alignment, policy distillation, sim-to-sim canonicalization (image translation / GAN / diffusion is an input-level special case; RCAN, James et al., CVPR 2019, 1812.07252, translates randomized sim images toward a canonical clean image before the policy, stitching DR to this axis). **Do not flatten DA into "DA = image translation."** Two boundaries: **DA only covers part of $\Delta_{\mathrm{obs}}$** — camera intrinsics / extrinsics, temporal sync, sensor bias, depth distortion are better handled by calibration / SI / sensor modeling. **Task-relevant invariance is the goal** — aligning $z_{\mathrm{sim}} \approx z_{\mathrm{real}}$ is not enough; keep $I(z; y_{\mathrm{task}})$ high while pushing $D(z_{\mathrm{sim}}, z_{\mathrm{real}})$ low, same statement as "overly wide DR washes out the task signal."

### Axis D — Optimization / adaptation: real-world fine-tuning

This axis **is not a mismatch class — it is an adaptation operator**: keep optimizing the policy on the target domain. It can be both the closing relay after the first three axes and an **early diagnostic or fast-adaptation tool**. FT **may change transfer delta and real-domain learning gap simultaneously**, but the two still diagnose separately; two regimes with very different cost structures:

- **Offline / imitation:** $D_{\mathrm{real}} \to \theta$, main cost is **data collection**.
- **Online RL:** $\pi_\theta \to a \to$ real transition $\to \theta'$, main cost is **interaction + safety + hardware wear + exploration**.

So comparing methods cannot look only at final success rate; it must also consider **the real-robot interaction budget required to reach target performance**. A rough indicator:

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{or}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

but only a **rough indicator**: baseline-dependent, not true marginal efficiency. Look instead at learning curve / AULC / marginal gain per 100 trajectories,

$$MV_{\mathrm{real}} \;\approx\; \frac{J(N+\Delta N)-J(N)}{\Delta N}$$

— the only form that connects with the article-wide $MV$. Risks go beyond catastrophic forgetting: more common is **distribution narrowing** — real FT data is much narrower than sim, so the post-FT policy is better on the target slice but robustness can drop, **generalization traded for specialization**. $MV_{\mathrm{real}}(N)$ is **not guaranteed positive**: the first 100 may buy a big jump, later returns decay quickly, and beyond that you may overfit or regress — **FT itself can enter a negative marginal-return region**.

## Two new routes that loosen the environment-generating-process assumption

The four axes above share an implicit premise: the classical framing treats simulator and real environment as **two given environment-generating processes** (with distributions $p_{\mathrm{sim}}$, $p_{\mathrm{real}}$). The two routes below loosen this premise itself — not "the fifth and sixth tricks" but a reformulation: **the first four change the intervention; world model and co-training change the underlying training substrate on which interventions operate** — a different abstraction level, not foldable back into the same taxonomy.

### World model: not cancelling the simulator, but replacing the simulator's source

**This section's lens**: we read WM as a "model source replacement" reformulation — only the slice "relative to physics sim, the model source and inductive bias have been replaced." This is not a standard definition of world models and not the only reading.

[Data Scaling for Robots](/en/articles/2026-09-09-robot-data-scaling/) already covered WM and data utility. Placed into sim-to-real, first correct a misreading: **WM does not naturally belong to sim-to-real** — the two routes have different causal directions:

```
Physics-sim route：  hand-designed dynamics  → train / optimize → deploy real
Learned-model route：interaction data → learned dynamics → imagine → optimize
```

**Interaction data can come from real, sim, or a mixture** — the learned-model route ≠ real-only learning.

Precisely: WM **does not cancel the simulator** — still simulating / imagining, only the predictive model is now learned. Better phrasing: **changing the source and inductive bias of the predictive model**:

$$\text{model source} \;=\; \text{physics prior} \;+\; \text{learned dynamics} \;+\; \text{data}$$

**The three can be hybrid** — reading WM as "simulator replacement" (a binary swap $f_{\mathrm{hand}} \rightarrow f_{\mathrm{learned}}$) oversimplifies.

Dreamer (1912.01603) and TD-MPC2 (2310.16828) embody this route. When **the model bias of a hand-crafted simulator is too large to be worth fixing first**, the world model offers a rewrite of the problem itself. DayDreamer (2206.14176) is often misread as "sim pretraining → real fine-tuning"; the more accurate statement is: **it demonstrates a real-interaction-driven experimental route** — learning a world model directly on a real robot and doing policy improvement via latent imagination. **Not depending on a handcrafted simulator ≠ model-free** — world-model learning still eats its full share of assumptions; it merely moves the inductive bias from "explicit physics" into the "learned world model."

Honest boundary: "learning dynamics from real data" **does not mean naturally better than simulation** — it swaps "hand-modeling cost" for "real collection + model capacity cost"; in contact-rich / long-tail / sensor-noisy settings, learned models often give **very confident and very wrong imagination** OOD — another trade-off between handcrafted sim and direct real-world RL, not the endgame. Once uncertainty enters the allocation core, the WM ↔ uncertainty interface must be explicit: **WM net value = predictive utility − model-uncertainty risk**; uncertainty must enter a **risk-aware decision layer**, implemented either as a hard feasibility gate ($\Pr(\text{model-induced unsafe}) \le \alpha$) or as a soft risk penalty ($U_{\mathrm{WM}} = U_{\mathrm{prediction}} - \gamma R_{\mathrm{model}}$), depending on deployment requirements. Only safety-critical deployments should default to the hard-gate form; otherwise a larger simulation budget just amplifies model bias.

### Sim-and-real co-training: reframing "transfer" as data mixture

Maddukuri et al. (RSS 2025, 2503.24361) proposed Sim-and-Real Co-Training as a pragmatic direction. **What the paper actually reports**: mixing sim and real within one training run yields an **average aggregate relative improvement of roughly 37.9% over the real-only baseline** across **two platforms and six visual manipulation tasks (across 6 tasks / 2 embodiments)** — a **relative lift under a paper-defined aggregate metric**, **not an absolute success-rate gain**, and not directly comparable to per-task deltas. When quoting 37.9%, always state the baseline (real-only) and aggregation definition; check per-task numbers against the original paper. It is not one-way sim→real transfer but a single recipe setting the ratio and schedule between the two.

**This article's reading (not the paper's proof)**: push one step further into a **data-mixture problem** — co-training's **primary intervention variable is the mixture** $p_{\mathrm{train}}=\alpha_{\mathrm{mix}}\, p_{\mathrm{sim}}+(1-\alpha_{\mathrm{mix}})\, p_{\mathrm{real}}$ ($\alpha_{\mathrm{mix}}$ avoids clashing with $\lambda$), not sim calibration and not a deployment-time adapter; **$\alpha_{\mathrm{mix}}$ is only a sampling-level simplification, $\alpha_{\mathrm{sampling}} \neq \alpha_{\mathrm{effective}}$** — sample repetition, augmentation, importance / loss weighting, curriculum, batch composition all change effective contribution; mixture weight $\neq$ dataset proportion. Lei et al. (arXiv 2026, 2604.13645) show that, **within the generative-robot-policy setting that paper studies**, changing the mixture induces **structured representation alignment and importance reweighting** — a **paper-specific explanation, not a universal claim**. Enough to establish "mixture as the primary lever, spanning multiple dimensions," not a fifth axis strictly orthogonal to the previous four.
---

> **Previous (Part 1)**: [Sim-to-Real Methodology - Theory](/en/articles/2026-09-10-sim-to-real-methodology/) -- reality gap quadruple and L1-L5 spine.
>
> **Next (Part 3)**: [Evaluation, Decision Matrix, and Protocol](/en/articles/2026-09-12-sim-to-real-evaluation-protocol/) -- evidence layers, composition, 6-step protocol.

*This is the Methods piece of the three-part Sim-to-Real Methodology series. Theory framework in Part 1, evaluation and protocol in Part 3.*
