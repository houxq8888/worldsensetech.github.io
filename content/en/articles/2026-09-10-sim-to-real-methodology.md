---
title: 'A Deep Dive into Sim-to-Real Methodology for Embodied AI: Treating "From Simulation to Reality" as an Error-Budget Allocation'
slug: "2026-09-10-sim-to-real-methodology"
date: 2026-09-10
draft: false
categories: ["Embodied AI", "Training Methods"]
tags: ["Embodied AI", "Sim-to-Real", "Domain Randomization", "System Identification", "Differentiable Simulation", "Residual Physics", "World Model", "Domain Adaptation", "Robot Data"]
description: 'Sim-to-real is not a single transfer trick but a closed-loop resource allocation. This article reframes the reality gap as a policy-conditioned, multi-source mismatch, turns error-budget allocation into an estimable, iteratively optimizable decision framework through intervention sensitivity and cost-normalized marginal value, and works through the mechanisms and failure boundaries of the four intervention lenses - SI, DR, DA, and fine-tuning.'
toc: true
related_articles:
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
  - 2026-08-25-dreamer-explained
---

> This piece follows [Data Sources and Interfaces](/en/articles/2026-09-08-data-and-training-recipes/) and [Data Scaling for Robots](/en/articles/2026-09-09-robot-data-scaling/). The first article split sim-to-real roughly into four tool families, but that was only a taxonomy; the question this article actually wants to answer is—

> **When simulation data falls short of reality along several evaluation-relevant directions, which lever should the next unit of budget (engineering time, compute, or robot-hours) go to: calibrating the simulator, widening the training distribution, aligning representations, or collecting real-robot data?**

This article does not propose a new sim-to-real algorithm; it proposes a decision framework for comparing and composing existing interventions. The whole piece converges along three levels:

- **Level 1 — Diagnosis**: $M_{\mathrm{sim}} \rightarrow p_{\mathrm{sim}}^\pi \rightarrow J_{\mathrm{sim}}$ vs. real — where are they different? Where does it matter?
- **Level 2 — Intervention**: four lenses (Model × Data × Representation × Optimization) — which intervention can change a given mismatch?
- **Level 3 — Allocation**: $MV(m \mid b, \pi)$ — given the current state, budget, and uncertainty, where does the next unit of resource go?

Then $\text{experiment} \rightarrow \text{real evaluation} \rightarrow \text{update belief }\mathcal{D}_t \rightarrow \text{next intervention}$, and the loop closes. The article's real spine is therefore **Diagnosis → Experiment → Intervention → Allocation → Re-evaluation** — a shape that captures the original framing better than the four-lens taxonomy on its own.

Four core contributions: **(1) redefining the mismatch** — the reality gap is a policy-conditioned, task-conditioned consequence, not an intrinsic scalar of the simulator; **(2) separating diagnosis from intervention** — mismatch descriptors / sensitivity are only diagnosis, while the real decision variable is the intervention; **(3) rewriting method selection as multi-resource sequential allocation** — SI / DR / DA / fine-tuning are intervention lenses, not mutually exclusive methods; **(4) extending simulator evaluation from fidelity to downstream utility** — prediction / ranking / selection quality are distinct utilities, presented as a corollary of the allocation framework.

At first glance this is engineering intuition; in reality it is closed-loop resource allocation: under several non-exchangeable budgets, you keep asking "where does the next dollar go, and where does it buy back the most real-world performance." What stalls teams is usually not "not knowing these methods exist" but "does this method do anything for this gap, and which budget will it eat." By "error budget" I do **not** mean a fixed pre-assigned quota per error term; it is spending on **intervention actions**, using sequential allocation to progressively push down whichever mismatch is currently most valuable.

## Reality Gap: not a scalar, but a policy-conditioned mismatch

Sim-to-real is usually narrated as "train a policy in simulation and transfer it to reality." A more rigorous starting point is **two distributions**: the same $\pi$ interacting with each environment induces $p_{\mathrm{sim}}^{\pi}(\tau)$ and $p_{\mathrm{real}}^{\pi}(\tau)$, which are generally not equal:

$$p_{\mathrm{sim}}^{\pi}(\tau) \;\neq\; p_{\mathrm{real}}^{\pi}(\tau)$$

The trajectory distribution is itself **policy-induced** — it changes with $\pi$; it is not an intrinsic property of the environment. What we actually care about is not the distributional difference but its **manifest consequence on the task** — the performance difference of the same $\pi$ in the two worlds:

Terminology is split into three strict levels to keep later ontology clean: **(a) trajectory / distribution mismatch** $D(p_{\mathrm{sim}}^\pi,\ p_{\mathrm{real}}^\pi)$, a process-level distributional difference; **(b) transfer delta**,

$$\boxed{\;\delta_J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)\;}$$

which is signed — if reality is in fact better (the simulator is more conservative, or its noise bites harder), $\delta_J$ comes out positive and intuitively should not be called a gap; **(c) performance discrepancy**,

$$G_J(\pi) \;=\; \big|\,\delta_J(\pi)\,\big|$$

which is the absolute magnitude — sensitivity is discussed under this semantics below, so we never tangle with the sign. **$J$ is by default a "higher-is-better" utility; if $J$ is a cost / minimization objective, the sign convention flips while the structure stays identical.** Also note: **$\delta_J$ is not the reality gap itself** — it is a downstream consequence of the gap under a specific $\pi$ + evaluation; the reality gap is closer to a full four-tuple property (see the next paragraph).

**Distribution mismatch $\neq$ performance gap**: $p_{\mathrm{sim}}^{\pi} \neq p_{\mathrm{real}}^{\pi}$ does not automatically imply a large $\delta_J$; different policies have entirely different sensitivities to the same distributional gap. A policy that relies only on coarse geometry barely changes performance when you swap the friction model; in precision assembly that leans on high-frequency force feedback, the same distributional difference can be fatal.

More fundamentally, what actually matters for a policy is not the marginal state distributions $p_{\mathrm{sim}}(s)$ vs $p_{\mathrm{real}}(s)$ but **policy-conditioned occupancy** $d_{\mathrm{sim}}^{\pi}(s,a)$ vs $d_{\mathrm{real}}^{\pi}(s,a)$ — in contact-rich manipulation even $d^\pi(s,a,\text{contact mode})$. The causal chain is $\pi \rightarrow d^\pi \rightarrow \text{mismatch} \rightarrow J$, not merely $\pi \rightarrow p^\pi(\tau)$.

$\delta_J(\pi)$ is a **task-relevant, policy-relevant observable consequence**. Written rigorously, one must **separate the mechanism from the induced distribution**: denote the environment's transition / observation / actuation kernels as mechanisms $M_{\mathrm{sim}}, M_{\mathrm{real}}$; under a given $\pi$ they **induce** $p_{\mathrm{sim}}^{\pi}(\tau),\ p_{\mathrm{real}}^{\pi}(\tau)$. A cleaner writing of the gap is:

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ \mathcal{E};\ M_{\mathrm{sim}},\ M_{\mathrm{real}}\big)$$

The logic is **mechanism → trajectory distribution → performance**: $\mathcal{E}$ is the set of evaluation assumptions (initial-state / horizon / reward / constraints) — the same $M_{\mathrm{sim}}$ can yield a small gap for a position-control policy and a huge gap for a force-sensitive manipulation policy. **This article operationally treats the reality gap as the downstream discrepancy under the four-tuple $(\pi,\mathcal{E},M_{\mathrm{sim}},M_{\mathrm{real}})$, not as a single intrinsic scalar of the simulator** — this is an **operational definition / framing**, not a claim that the community has a unified formal definition; every later occurrence of "reality gap" below uses this notion.

### Where exactly the gap sits: reality mismatch and task-specification mismatch

The first move is to unpack the multi-source gap — there are **two big families of causes**, and they cannot all be shoved under the single word "reality":

```
Sim-to-real / task mismatch
├── Reality mismatch (physical layer)
│   ├── Dynamics / contact / stochasticity  friction, contact, deformables, compliant structures; motor stochasticity, friction variability, unmodeled disturbance, repeated-reset variability
│   ├── Observation / estimation  sensor physics, calibration, noise, occlusion, latency, state estimation
│   ├── Actuation / timing        motor dynamics, control rate, actuator lag, comms jitter
│   └── Initial-state / env.      reset distribution, scene layout, long tail, initial conditions
└── Task-specification mismatch
    └── Objective / constraint    reward definition, safety constraints, success criterion
```

The two families have different sources and should not be simply added: reality mismatch is "simulation and reality are not the same world"; task-specification mismatch is "the objective you optimize and the objective you deploy are not the same task." **Observation and state estimation deserve their own layer** — the robot actually executes $a_t = \pi(o_t),\ o_t = h(x_t) + \epsilon$; camera calibration error, depth bias, occlusion, proprioception drift, force-sensor bias, and state-estimator latency are **not "the picture looks different" — they make the state the policy actually sees inconsistent with the state the simulator assumes is available**. In manipulation and locomotion, this "state-estimation gap" often hurts performance more than the appearance gap.

In addition, **stochasticity mismatch** (motor stochasticity, friction variability, sensor temporal correlation, communication jitter, unmodeled disturbance, repeated-reset variability) is not the same as parameter mismatch — it concerns differences in the **higher-order statistics / stochastic-process structure** of the dynamics, which is precisely what DR's $p_{\mathrm{DR}}(\xi)$ is meant to cover.

Note in particular that timing mismatch ($\Delta t_{\mathrm{sim}} \neq \Delta t_{\mathrm{real}}$, action hold, sensor delay, policy inference latency, asynchronous observation) **can be amplified by closed-loop feedback dynamics** — it is not a simple additive observation error, since it can alter closed-loop stability itself.

**Initial-state / environment mismatch** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$; **objective / task shift** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$. Attribution deserves care: if sim and real **can both produce the same $s_0$** and training simply missed it, that is **ordinary train-test shift, not a reality gap**; only when the sim-real reset / scene **implementation itself** differs is it environment mismatch. Objective shift is already objective mismatch: no matter how accurate the physics, if reward / constraints do not line up you are not looking at "transfer failure" but at "you never evaluated the same task in the first place." Below we assume the objective is aligned.

## Writing "error-budget allocation" as an estimable, iteratively optimizable decision framework

With the sources unpacked, the opening intuition needs a mathematical landing. The error terms interact strongly — the simulator assumes perfect proprioception, reality has latency; neither alone is fatal, stacked they can destabilize a controller — so a safer move is to first admit an unknown coupling function $F$:

$$\boxed{\;\delta_J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}}\big)\;}$$

**Each $\Delta_k$ here is a mismatch descriptor — potentially scalar, vector, distribution, or set-valued.** The stochasticity / occupancy / model-class uncertainty discussed later do not fit a single scalar "error magnitude," so this equation is schematic and does not presuppose a common scalar metric.

**This version takes $\Delta_{\mathrm{opt}}$ (optimization / learning error) out of the reality gap**: they live at different levels — for the same fixed policy, if the simulated dynamics and observations are both accurate but RL never converged, $\delta_J$ is small while the policy is bad; they should be split into **two diagnostic quantities**:

$$\underbrace{J_{\mathrm{real}}(\pi_{\mathrm{train}}) - J_{\mathrm{sim}}(\pi_{\mathrm{train}})}_{\text{transfer delta } \delta_J}\qquad \underbrace{J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})}_{\text{real-domain learning gap}}$$

**These two quantities cannot be unconditionally summed and called "deployment loss"**: $\delta_J$ is signed, the two terms have different baselines; they are **error sources at different levels**, to be diagnosed and attributed separately. **$\pi^{*}_{\mathrm{real}}$ is typically unavailable** — the right-hand term is an **oracle-defined diagnostic quantity**; in practice, use $J_{\mathrm{real}}(\pi_{\mathrm{best\text{-}observed}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})$ as a proxy, and the selection-regret term below uses the same treatment for terminological consistency.

Only when doing engineering attribution near an operating point do we locally approximate $F$ as a weighted sum $\delta_J \approx \sum_k w_k \Delta_k$ — **this layer is a local attribution heuristic, not the article's core formula**. What actually drives decisions is the **intervention sensitivity** measured after picking an **intervention variable** $\xi_k$ for each mismatch class:

$$\hat S_k^{\mathrm{int}} \;\approx\; \frac{J_{\mathrm{real}}\big(\pi;\,\xi_k{+}\delta\big) \;-\; J_{\mathrm{real}}\big(\pi;\,\xi_k\big)}{\delta}$$

**Key clarification**: $\xi_k$ is **not a natural coordinate of the true gap — it is an intervention variable defined artificially for a sensitivity experiment**, and many $\xi_k$ in the real world are simply **not directly controllable**. So sensitivity itself should be tiered into three kinds: **direct perturbation sensitivity** ($\xi_k$ is continuously tunable on the real robot — latency / friction / appearance), **proxy / surrogate sensitivity** (estimated through the simulator or a controlled bench — camera calibration error), and **diagnostic ablation** (module / model / dataset swaps producing finite-difference attribution — contact model, state estimator). This is a **controlled experimental perturbation, not a derivative of an intrinsic quantity** (deliberately avoiding Pearl-style do-calculus notation so as not to imply a full causal-graph assumption). **$\hat S_k^{\mathrm{int}}$ is only a diagnostic aid**; **the core decision quantity is $MV(m\mid b,\pi,\mathcal{D})$ in the next section**.

**Diagnosis $\neq$ attribution (important).** Sensitivity is not attribution. Perturbing $\Delta_{\mathrm{friction}}$ alone may have a tiny effect, perturbing $\Delta_{\mathrm{latency}}$ alone may also be small, yet together they can produce $\Delta J(\Delta_f,\Delta_l) \gg \Delta J(\Delta_f,0) + \Delta J(0,\Delta_l)$ — an interaction / synergy effect. **Sensitivity experiments identify locally influential intervention directions; they do not by themselves provide an additive causal attribution of the observed deployment gap.** $\Delta_{\mathrm{model}}$ and $\Delta_{\mathrm{ctrl}}$ can even compensate each other unidentifiably (the actuator gain is wrong, and the policy quietly offsets it through its command distribution); both remain decision statistics estimated via sensitivity experiments / ablation, not strict decompositions.

### The real "allocation": spend on intervention actions, not pick one method off a shelf

For budget allocation to be literal, the budget must be split **continuously** across the intervention axes: decompose the total budget into a vector $b=(b_1,\dots,b_K)$, where $b_k$ is the amount spent on intervention $k$ — $b_{\mathrm{SI}}=2\text{h}$, $b_{\mathrm{DR}}=10^6$ sim steps, $b_{\mathrm{real}}=4\text{h}$ real robot — not a 0/1 choice like "use SI or not." The objective is to maximize real-world performance:

$$\max_{b}\quad J_{\mathrm{real}}\big(\pi_b\big)$$

In robotics projects the budget is **not one currency**: GPU may be near unlimited while real robot-hours are scarce; you may have machine time but no engineering headcount — so the correct writing is **multi-budget constraints**, not a collapse into a scalar $B$:

$$\begin{aligned}
C_{\mathrm{real}}(b) &\le B_{\mathrm{real}}\\
C_{\mathrm{compute}}(b) &\le B_{\mathrm{compute}}\\
C_{\mathrm{eng}}(b) &\le B_{\mathrm{eng}}\\
C_{\mathrm{risk}}(b) &\le B_{\mathrm{risk}} \quad\text{(safety budget: e-stop count / hardware-fault tolerance / operator-intervention ceiling)}
\end{aligned}$$

Once the budget is a vector, the decision variable should shift from "gap" to "intervention": an engineer cannot buy "two percentage points of $\Delta_{\mathrm{model}}$"; what they can buy is 30 minutes of SI, $10^6$ sim steps, 100 real trajectories, a camera calibration, a residual model. It is more natural to define marginal utility on an intervention $m$ — **an intervention does not directly change $\Delta_k$; it changes the policy through the training process**:

$$\boxed{\;\pi_{b+m} \;=\; \operatorname{Train}\big(D_{\mathrm{sim}},\ D_{\mathrm{real}};\ m\big)\;}$$

So "where does the next dollar go" becomes a quantity defined on interventions that must be estimated step by step in the real world. **The core decision formula of this article is $MV$, not $w_k$ or $\hat S_k^{\mathrm{int}}$** ($MV$ is more accurately named **cost-normalized marginal value**). Written rigorously, an intervention is a pair $m = (\text{type},\ \Delta b_m)$ where $\Delta b_m$ is the resource increment consumed by that intervention, and executing it yields $b' = b + \Delta b_m$; the expectation is **explicitly conditional on the current evidence $\mathcal{D}$** (existing real evaluations, pilot results, and current calibration state):

$$\boxed{\;MV(m \mid b,\pi,\mathcal{D}) \;=\; \frac{\mathbb{E}\big[\,J_{\mathrm{real}}(\pi_{b'}) - J_{\mathrm{real}}(\pi_{b}) \;\big|\; \mathcal{D}\,\big]}{C(m)}\;}$$

The real novelty of this framing is not "which method is better" but **given current evidence, what is the expected value of the next unit of resource**. Correspondingly, $m^{*} = \arg\max_m MV(m \mid b,\pi,\mathcal{D})$ is only a **one-step / local allocation rule**, not a global optimum; and different interventions (30 min SI, $10^6$ DR steps, 100 real trajectories) do **not live in the same intervention space** — the "type" component of $m$ is precisely what encodes this. The full problem should be written as a **multi-resource sequential allocation**:

$$\max_{\{m_t\}_{t=1}^{T}}\ \mathbb{E}\big[J_{\mathrm{real}}(\pi_T)\big] \quad \text{s.t.}\quad \sum_{t} C_r(m_t) \le B_r,\;\; r \in \{\mathrm{real},\mathrm{compute},\mathrm{eng},\mathrm{risk}\}.$$

$MV$ is a **local decision statistic** for this sequential problem; only under the additional assumptions of negligible interaction between interventions, linear cost, and no fixed cost does greedy selection approximate global optimality — none of which is assumed in this article. This ratio cannot be computed analytically from the simulator; it can only be estimated **sequentially** via pilot experiments / ablation / few-shot real evaluation. Four caveats: **(i) uncertainty** — real-world $\Delta J$ is extremely noisy, so allocation should also look at CI / posterior / **lower confidence bound (LCB)**, otherwise a high-variance intervention gets wrongly prioritized by a single lucky run. **(ii) non-linear cost** — a one-off SI engineering pass can benefit every subsequent training run (fixed cost); DR saturates gradually (diminishing returns); fine-tuning has threshold effects. **(iii) non-monotonicity / negative MV** — **this article does not assume interventions monotonically improve real performance**; over-randomization, overfitting-style fine-tuning, an incorrect residual, and cross-domain negative transfer all mean $MV$ **can be negative** ($\Delta J_{\mathrm{real}} < 0$). **(iv) information value** — many pilot interventions (e.g. a 20-minute friction identification) have $\Delta J \approx 0$ immediately but sharply shrink the uncertainty set for downstream allocation; their real contribution is **learning what to do next**, not an instant lift in the policy. This can be formalized as $MV_{\mathrm{perf}} = \mathbb{E}[\Delta J]/C(m)$ and $MV_{\mathrm{info}} = \mathbb{E}[V(\mathcal{D}_{t+1}) - V(\mathcal{D}_t)]/C(m)$, where $V$ is a value-of-information functional over the evidence set. Total allocation score can be loosely read as $MV_{\mathrm{perf}} + \lambda\, MV_{\mathrm{info}}$ — but **$MV_{\mathrm{info}}$ stays at the narrative layer and does not enter the core boxed formula**, to avoid framework inflation.

**$MV$ across different interventions is also not a fixed constant**: $MV_i = MV_i(b_{1:i-1},\ \pi_b,\ D_{\mathrm{real}})$ — doing SI first narrows the uncertainty set and DR's $MV$ falls; doing DR first yields a more robust starting point and fine-tuning's $MV$ rises. **Interventions exhibit complementarity, substitutability, and occasional conflict simultaneously**, so this is **resource-constrained sequential experimentation / adaptive allocation** (close to adaptive experimental design, but **do not write it as a bandit algorithm** — there are no strict arms / stationary reward / regret guarantees).

There is one more subtle layer of feedback: **an intervention does not only push the gap down — it changes the policy, and thereby changes the policy's own sensitivity to the gap** — $S_k^{\mathrm{int}} = S_k^{\mathrm{int}}(\pi)$, $\pi = \pi(m)$, so the loop is not one-directional:

```
estimate mismatch → estimate sensitivity → intervention
       ↑                                          ↓
   re-estimate  ←  sensitivity changes  ←  policy changes
```

**This feedback loop fits the allocation thesis of this article better than any new equation**: sim-to-real is not an optimization you solve once; it is a sequential experiment where each round ends by re-estimating for the next.

Mapping each intervention to its primary compressed term and primary budget:

| Intervention | Primary term compressed | Primary budget |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{eng}}$ + $C_{\mathrm{compute}}$ + a little $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | $C_{\mathrm{compute}}$ (sample efficiency) |
| Residual physics | $\Delta_{\mathrm{model}}$ (residual part) | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$ (appearance subset) | $C_{\mathrm{real}}$ (unlabeled data) + $C_{\mathrm{compute}}$ |
| Real-world fine-tuning | **not directly corresponding to a single mismatch; changes the policy through target-domain optimization** (can simultaneously alter the transfer delta and the real-domain learning gap) | $C_{\mathrm{real}}$ (wear / safety) |
| World model | change model source | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Sim-and-real co-training | change $p_{\mathrm{train}}$ (mostly $\Delta_{\mathrm{dist}}$) | mixed data ($C_{\mathrm{real}}+C_{\mathrm{compute}}$) |

With this writing, the whole article is not "which of the four methods is better" but a loop: locate the dominant $\Delta_k$, judge how much it matters via sensitivity, invest one unit of budget in the intervention with the highest $MV$, measure the return in real evaluation, then decide the next unit.

## Four intervention lenses (more precisely, four relatively independent analytical dimensions)

With the framework in place, look at the tools axis by axis. SI, DR, DA, and real-world fine-tuning **are not peer categories at the same level of abstraction** — SI is model calibration, DR is training distribution manipulation, DA is representation alignment, fine-tuning is an optimization strategy; lining them up as "four method families" misleads people into four-way picking, while in reality they are **four relatively independent intervention lenses** that compose (**this article's analytical decomposition, not a community-accepted ontology**):

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

"$\times$" here is a **combinatorial space**, not mathematical orthogonality — DR touches Model / Observation / Distribution, DA can happen at input / feature / latent / policy / output, and "DA = the Representation axis" is only one abstraction layer of this article.

**The criterion for choosing a tool is not "systematic goes to SI, random goes to DR"** — the more useful partition is the continuum "**point estimate → posterior → robust randomization**". What SI actually does is **fit parameters under an identification objective**:

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; \mathcal{L}_{\mathrm{ID}}\big(D_{\mathrm{real}},\ f_{\mathrm{sim}}(\cdot\,;\,\phi)\big)$$

$\mathcal{L}_{\mathrm{ID}}$ can be trajectory prediction / one-step transition error / force-torque residual / likelihood — **many classical SI methods never do trajectory distribution matching at all; they just minimize prediction error**. SI addresses **identifiable, parameterizable model mismatch**; symmetrically, DR addresses **uncertainty that can be expressed by the training distribution**.

| Nature of the mismatch | More natural tool |
| --- | --- |
| Parameterizable + identifiable | System Identification (point estimate $\hat\phi$) |
| Parameterizable but only uncertainty available | Bayesian / posterior SI → posterior-guided DR |
| Parameterizable but hard to identify / high uncertainty | Domain Randomization |
| Difficult to express with low-dimensional physical parameters, but has a structured residual | Residual learning |
| Observation / appearance mismatch | Domain Adaptation |
| Policy still has systematic residual on the target domain | Fine-tuning |

The key point: **"not precisely identifiable" and "no knowledge at all" are not the same thing** — once you have a posterior $p(\phi \mid D_{\mathrm{real}})$, the most natural move is $\phi \sim p(\phi \mid D_{\mathrm{real}})$ for **posterior-guided randomization**, stitching SI and DR into a continuous spectrum.

### Axis A — Model: system identification, differentiable simulation, and residual physics

This axis handles $\Delta_{\mathrm{model}}$, and internally contains three **distinct levels** that are usually bundled into "differentiable simulation is stronger SI":

$$y_t \;=\; \underbrace{g_{\mathrm{physics}}(x_t,a_t;\phi)}_{\text{parameterizable physics}} \;+\; \underbrace{r_\theta\big(\psi(x_t,a_t)\big)}_{\text{residual}} \;+\; \epsilon_t$$

**This is only a representative parameterization.** $y_t$ may be the next state $x_{t+1}$, a contact impulse, an acceleration, a deformation field, or some other observable, and $\psi$ is the residual's input view. The additive state-transition form is just one instance; a soft robot's residual deformation field, a contact-impulse residual, and a state residual are not the same mathematical object.

- **Differentiable simulation answers "how to optimize the model"** — it provides a gradient path through simulator parameters, states, and controls, and can serve as an **optimization interface** for identification, trajectory optimization, and similar problems (**it is not itself system identification**). DiffTaichi (Hu et al., ICLR 2020, 1910.00935) and Interactive Differentiable Simulation (Heiden et al., arXiv 2019, 1905.10706) are representative implementations of this gradient path.
- **System identification answers "which parameter to optimize"** — the real workflow is often **real → identify → sim → train → real**, so a more accurate name is **real-to-sim-to-real**.
- **Residual physics answers "who explains the part the model did not explain"** — instead of forcing a calibration of $\phi$, let a network learn $r_\theta$ to fill the gap.

$r_\theta$ is only a **unified notation**: the actual residual can be defined on state transition, force, acceleration, contact impulse, deformation field, or other latents.

There is a make-or-break point most easily hidden behind the word "differentiable": **differentiability solves the optimization interface, not model class correctness**. If the simulator's contact model simply does not express a real phenomenon, then however precise the gradients, they only give you "the optimum under a wrong model." **A commonly ignored boundary**: collision / friction / contact-mode switching are typically **nonsmooth or piecewise-smooth** — even if $\partial f/\partial\phi$ exists, there is no guarantee that the gradient is stable, that the gradient at contact-mode transitions is meaningful, or that it beats derivative-free optimization.

SI has two further, finer but real pitfalls. **First, $p_{\mathrm{real}}(\tau)$ is essentially never directly accessible** — you only have a finite set of real trajectories. **Second, parameters existing $\neq$ parameters identifiable** — identifiability also depends on excitation and sensor observability: mass / damping / stiffness can produce nearly identical observable trajectories under some excitations and cannot be estimated independently.

Residual physics also needs a narrowed boundary: the sweet spot is where $f_{\mathrm{physics}}$ still provides a **useful structural inductive bias** and the residual only makes a bounded correction on the target distribution — soft robots (Gao et al., RA-L 2024, 2402.01086) and buoyancy-assisted legged robots (Sontakke et al., 2023, 2303.09597) are exactly the "trunk physics counts, local residual is stable" regime. If $f_{\mathrm{physics}}$ is completely wrong and the residual has to carry the entire dynamics alone, you are better off just learning a model. $r_\theta$ **is not naturally equal to "the missing physics"** — an unrestricted additive residual absorbs sensor bias / actuator error / timing / calibration / reward mismatch into an **error sponge**: it fits the training distribution and falls apart out of distribution; so the residual needs structural constraints (low-dimensional / sparse / force or acceleration scale / physical priors / active only in specific contact regimes).

There is also **confounding** between $\phi$ and $r_\theta$ — if the residual is flexible enough, it absorbs effects that should belong to $\phi$, making $\hat\phi$ meaningless; identifiability requires that the contributions of $f_{\mathrm{physics}}$ and $r_\theta$ be distinguishable in data (typically needing regularization, scale separation, or structural constraints).

### Axis B — Data distribution: domain randomization and its family

This axis does not chase some "most accurate" $p_{\mathrm{real}}$; it makes the policy robust to a family of parameters $\{\phi\}$. Tobin (1703.06907) used pure visual randomization to bring sim grasp detection to the real robot; Peng (1710.06537) pushed randomization into dynamics; OpenAI's in-hand manipulation (Akkaya et al., 1808.00177) nearly took DR to its extreme — **absorbing difference not through precise calibration but through "a randomization range wide enough."**

A commonly mis-written intuition: **DR is not an "implicit ensemble"** — what is trained is a **single** shared policy $\pi_\theta$, with the objective

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

More precisely: **DR is a population-level optimization over a family of environment models**. The formula above is a **baseline abstraction of risk-neutral average-case DR**; robust / adversarial DR can instead use $\max_\theta \min_{\phi\in\Phi} J(\pi_\theta;\phi)$, CVaR, or other risk-sensitive forms. The condition for DR to work is that **the real parameter distribution lands inside DR's support** — a safer statement is $\mathrm{supp}(p_{\mathrm{real}}) \subseteq \mathrm{supp}(p_{\mathrm{DR}})$ together with adequate density in the real-typical region. **But this still carries a more fundamental premise**: real dynamics must be expressible by the same $\phi$-parameterization; if the simulator's model class does not contain the real phenomenon, then $\phi_{\mathrm{real}}$ cannot even be defined and the support statement **collapses at the root** — at that point the problem is model-class uncertainty, not "DR was not wide enough." (**Parameter uncertainty** is "where in $\phi$-space does reality sit," addressable by the posterior or by DR; **model-class uncertainty** is "can this $\phi$-parameterization express real dynamics at all," which is not solvable by widening ranges.) **Main conclusion**: parameter-space support is a useful design proxy, **but the deployment-relevant object is the policy-conditioned occupancy it induces** — from $p_{\mathrm{DR}}(\phi)$ through to the training-induced $d_{\mathrm{train}}^{\pi}(s,a,\text{mode})$ and the real $d_{\mathrm{real}}^{\pi}(s,a,\text{mode})$, it is the overlap between these occupancies that actually governs downstream transfer. Different parameters can induce highly overlapping policy-relevant trajectories, so parameter-space differences are **not necessarily a sufficient condition for deployment failure**. **Parameter coverage is a necessary proxy, not a sufficient condition for deployment coverage**.

One layer further down: **DR is not about choosing scalar ranges, it is about designing a joint distribution** — $p(\phi_1,\phi_2) \neq p(\phi_1)p(\phi_2)$ is the norm (payload increase co-varies with actuator regime, temperature co-varies with motor resistance / friction / battery), and independent uniform DR is only a convenient baseline. The question returns to allocation: **the randomization distribution must align with evaluation and objective**; overly wide or task-irrelevant randomization lowers sample efficiency, but in robust settings appropriately enlarging the uncertainty set can help. **"Wider is more conservative" is not a universal rule; shape and alignment are what matter.**

"Adaptive / Automatic DR" is a family rather than a single method: curriculum / adversarial / automatic / posterior-based sampling / performance-driven range adaptation — the common thread is **avoiding over-randomization from the start**.

### Axis C — Observation / Representation: domain adaptation and observation translation

This axis handles $\Delta_{\mathrm{obs}}$, aligning sim and real **at the observation / representation layer**. **"Representation" is this article's abstraction** — DA in practice can act on input / feature / latent / output / policy / dynamics model, six layers. Concrete mechanisms include feature-level adapters, latent alignment, policy distillation, and simulation-to-simulation canonicalization, among others (image translation / GAN / diffusion is only an input-level special case; the representative example RCAN, James et al., CVPR 2019, 1812.07252, translates randomized sim images back toward a canonical clean image before feeding the downstream policy, incidentally stitching DR to this axis). **Do not flatten DA into "DA = image translation."** Two boundaries: **First, DA is only a subset of observation mismatch** — camera intrinsics/extrinsics, temporal sync, sensor bias, and depth distortion are better handled by calibration / SI / sensor modeling. **Second, task-relevant invariance is the goal** — aligning $z_{\mathrm{sim}}\approx z_{\mathrm{real}}$ alone is not enough; ideally you keep $I(z;y_{\mathrm{task}})$ high while pushing $D(z_{\mathrm{sim}},z_{\mathrm{real}})$ low, which is the same statement as "overly wide DR washes out the task signal."

### Axis D — Optimization / adaptation: real-world fine-tuning

This axis itself **is not a class of mismatch — it is an adaptation operator**: continue optimizing the policy directly on the target domain. It can serve both as the closing relay after the first three axes and as an **early diagnostic or fast-adaptation tool**. Fine-tuning **may change transfer delta and real-domain learning gap simultaneously**, but the two should still be diagnosed separately; the two regimes have completely different cost structures:

- **Offline / imitation:** $D_{\mathrm{real}} \to \theta$, main cost is **data collection**.
- **Online RL:** $\pi_\theta \to a \to$ real transition $\to \theta'$, main cost is **interaction + safety + hardware wear + exploration**.

So comparing methods cannot look only at final success rate; it must also consider **the real-robot interaction budget required to reach target performance**. A rough indicator:

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{or}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

but this is only a **rough indicator**: it depends on the baseline and is not true marginal efficiency. What you should actually look at is the learning curve / AULC / marginal gain per 100 trajectories,

$$MV_{\mathrm{real}} \;\approx\; \frac{J(N+\Delta N)-J(N)}{\Delta N}$$

— only this connects with the article-wide $MV$ framework. Risks are not limited to "catastrophic forgetting": more common is **distribution narrowing** — real fine-tuning data is much narrower than sim, so after fine-tuning the policy is better on the target slice but its robustness can actually drop, **generalization traded for specialization**. $MV_{\mathrm{real}}(N)$ is **not guaranteed to stay positive**: the first 100 trajectories may buy a big jump, later returns decay quickly, and beyond that you may overfit or even regress — **fine-tuning itself can enter a negative marginal-return region**.

## Two new routes that loosen the "two given distributions" assumption

The four axes above share an implicit premise: **$p_{\mathrm{sim}}$ and $p_{\mathrm{real}}$ are two given distributions**. The two routes below loosen this premise itself — they are not "the fifth and sixth transfer tricks" but a reformulation of the whole problem.

### World model: not cancelling the simulator, but replacing the simulator's source

**This article's lens disclaimer**: inside the allocation taxonomy, I read the world model as a reformulation of "model source replacement" — **this is this article's analytical angle, not a standard definition of world models**. Strictly speaking, a world model's extension is much wider than this section; this section only takes the slice "relative to physics sim, the model source has been replaced."

The [Data Scaling for Robots](/en/articles/2026-09-09-robot-data-scaling/) article already discussed world models and data utility. Placed into the sim-to-real context, first correct a positioning misreading: **the world model does not naturally belong to sim-to-real** — the two routes have different causal directions:

```
Physics-sim route：  hand-designed dynamics  → train / optimize → deploy real
Learned-model route：interaction data → learned dynamics → imagine → optimize
```

**Interaction data can come from real, sim, or a mixture** — the learned-model route ≠ real-only learning.

To say it precisely: the world model **does not cancel the simulator** — it swaps the simulator from "a hand-specified physics model" to "a predictive model learned from interaction data"; what changes is the **model source**:

$$f_{\mathrm{hand\text{-}designed}} \;\longrightarrow\; f_{\mathrm{learned}}$$

Dreamer (1912.01603) and TD-MPC2 (2310.16828) embody this route. When **the model bias of a hand-crafted simulator is too large to be worth fixing first**, the world model offers a rewrite of the problem itself. DayDreamer (2206.14176) is often misread as "sim pretraining → real fine-tuning"; the more accurate statement is: **it demonstrates a real-interaction-driven experimental route** — learning a world model directly on a real robot and doing policy improvement via latent imagination. **Not depending on a handcrafted simulator ≠ model-free** — world-model learning still eats its full share of assumptions; it merely moves the inductive bias from "explicit physics" into the "learned world model."

The honest boundary: "learning dynamics from real data" **does not mean it is naturally better than simulation** — it swaps "hand-modeling cost" for "real collection + model capacity cost"; in contact-rich, long-tail, sensor-noisy settings, learned models often give **very confident and very wrong imagination** out of distribution — this is **yet another trade-off** between "handcrafted sim" and "direct real-world RL," not the endgame.

### Sim-and-real co-training: reframing "transfer" as data mixture

Maddukuri et al. (RSS 2025, 2503.24361) proposed Sim-and-Real Co-Training as a pragmatic direction. **What the paper actually reports**: mixing sim and real within one training run yields an **average aggregate relative improvement of roughly 37.9%** relative to the paper's **own baselines (train-on-real-only and train-on-sim-only, each as its own reference)** across **two platforms and six visual manipulation tasks**. This is a **relative lift under a paper-defined aggregate metric normalized across tasks** — **not an absolute percentage-point gain in success rate**, and not directly comparable to per-task success-rate deltas. When quoting 37.9%, always state the baseline and the aggregation definition; check the per-task numbers against the original paper. It does not do one-way sim→real transfer but is a single recipe that decides the ratio and schedule between the two.

**This article's reading (not something the paper proves)** is to push it one step further into a **data-mixture problem**: co-training's **primary intervention variable is the training mixture** $p_{\mathrm{train}}=\lambda\, p_{\mathrm{sim}}+(1-\lambda)\, p_{\mathrm{real}}$, not simulator calibration and not a deployment-time adapter; **$\lambda$ itself is only a sampling-level simplification** — real recipes also change the **effective** training distribution through dataset size / importance weighting / augmentation / curriculum. Follow-up mechanistic analysis (Lei et al., arXiv 2026, 2604.13645) shows that changing the mixture also induces **structured representation alignment and importance reweighting** — "mixture as the primary lever, with effects spanning multiple dimensions," rather than a fifth axis strictly orthogonal to the previous four.

## Evaluation: how do you know you actually closed the gap?

A dangerous practice is reporting performance only on sim benchmarks. A more credible evaluation should at least:

- report **zero-shot transfer** performance together with curves after **few-shot / N-shot** adaptation;
- test on a set of **held-out hardware / calibration / object / contact / environmental regimes**;
- explicitly declare whether the **task / initial-state / evaluation distributions** match between sim and real;
- do **failure attribution**: which layer of $\Delta_k$ dominates? get attribution wrong and the budget goes to the wrong place;
- **do not report means alone**: at least mean ± CI across multiple seeds / resets; prefer **paired evaluation**;
- **report safety failures separately**: $J_{\mathrm{real}}$ should be reported alongside safety violation / e-stop / intervention count / hardware fault / recovery time.

Following the line "the simulator is a proxy for the real world," there is a question more fundamental than numerical alignment: **can the simulator correctly predict which policy is better?**

A **conceptual example** (numbers do not represent experimental results):

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

In sim it looks like $A > B > C$; on the real robot it is $B > C > A$. Here the simulator has **lost model-selection utility** — you would use it to pick out the worst policy. So **when the simulator is used for policy / model selection**, look at rank correlation $\rho_{\mathrm{rank}} = \mathrm{Spearman}(J_{\mathrm{sim}}(\pi_i), J_{\mathrm{real}}(\pi_i))$ together with selection regret:

$$\pi_{\mathrm{sim}} = \operatorname*{arg\,max}_{\pi \in \Pi} J_{\mathrm{sim}}(\pi), \qquad R_{\mathrm{select}} = J_{\mathrm{real}}\big(\pi^{*}_{\mathrm{real}}\big) - J_{\mathrm{real}}\big(\pi_{\mathrm{sim}}\big)$$

Spearman = 0.95 but a wrong top-1 is still a disaster; conversely, Spearman = 0.7 but a top-1 that rarely misses is enough for "pick one deployable policy." Both are **conditional metrics**. **A conclusion the allocation framework naturally yields**: **simulator fidelity is task-of-use dependent, not an absolute property** — change the use (pretraining / exploration / curriculum / safety filter) and "which errors matter" changes entirely. $\pi^*_{\mathrm{real}}$ is typically unavailable, so $R_{\mathrm{select}}$ — like the real-domain learning gap earlier — is an **oracle-defined diagnostic quantity**; in practice use $J_{\mathrm{real}}(\pi_{\mathrm{best\text{-}observed}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$ or a Pareto-best proxy.

When full ranking is unnecessary, more practical metrics are **top-k recall** (the probability that the true-best policy appears in the sim-selected top-$k$), **regret@k**, or the binary question "real best $\in$ sim top-$k$?" — the simulator only needs to include good policies in its candidate set, not precisely rank the tail.

At this point, **an important corollary of the allocation framework**: **simulator utility is not a single property but three non-substitutable dimensions** —

| Simulator utility dimension | Typical metric |
| --- | --- |
| Numerical prediction accuracy (absolute error / calibration) | MAE / RMSE $\mathbb{E}\big[|J_{\mathrm{sim}}(\pi) - J_{\mathrm{real}}(\pi)|\big]$, calibration curve, prediction interval coverage |
| Ranking accuracy | Spearman $\rho_{\mathrm{rank}}$, Kendall $\tau$, top-k recall, regret@k |
| Quality of the selected policy (decision quality) | $R_{\mathrm{select}} = J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$ (in practice use a best-observed proxy) |

A simulator can be calibrated very accurately and still pick the wrong policy (narrow distribution); another can be numerically wrong across the board yet rank stably with small regret — the three dimensions cannot substitute for each other. $U_{\mathrm{sim}}$ should not be written as an abstract scalar; it should be expanded into **utility classified by use**:

$$U_{\mathrm{sim}} \;\in\; \big\{\ U_{\mathrm{pretrain}},\ U_{\mathrm{selection}},\ U_{\mathrm{exploration}},\ U_{\mathrm{curriculum}},\ U_{\mathrm{safety}}\ \big\}$$

Evaluating fidelity cannot stare at a single policy; it must be relative to the **candidate policy family** and the **concrete use**: $U_{\mathrm{sim}}(\cdot \mid \Pi_{\mathrm{candidate}},\ p_{\mathrm{eval}}^{\mathrm{real}})$.

## Composition, decision, and a question usually dodged

With priorities in hand, a more useful shape for real projects is a **gap × modellability × real-budget** decision matrix:

| Gap | Parameterizable / identifiable? | Real data | Recommendation |
| --- | --- | ---: | --- |
| low-dimensional dynamics bias | high | scarce | SI |
| parameterizable dynamics uncertainty | medium | scarce | posterior-guided DR / Bayesian SI → DR |
| dynamics residual | low (but structured) | medium | Residual learning |
| visual appearance | high | none / scarce | DA / DR |
| actuator latency | high | scarce | SI + DR |
| unknown long-tail, simulatable | low | scarce | targeted simulation / DR |
| unknown long-tail, sim untrustworthy | low | medium | real data |
| model class uncertain | low | abundant | learned world model (if real is scarce, prefer physics prior + residual / DR) |
| mixed | mixed | mixed | co-training candidate (verify positive-transfer conditions first) |

**The qualifiers in the first two rows cannot be dropped**: if the uncertainty comes from **model-class uncertainty** (the simulator's functional form itself cannot express the real phenomenon), neither SI nor DR may apply, and you have to fall to the residual / world model / real-data rows first. The second-to-last row: "model unknown" alone does not imply a world model; the criterion is **model uncertainty × real-data budget** — a learned world model is reasonable only when the model class is uncertain **and** real interaction is abundant. The last row is the same: "co-training as a safety net" clashes with the allocation thesis — when sim quality is bad, real data is scarce, and the two sides disagree on action space / task semantics, negative transfer is entirely possible.

A common combo is **SI → DR → DA → co-training / fine-tune**: **the arrows are only a schematic, not a fixed workflow** — the real order is determined by the currently dominant gap and by marginal utility. The most valuable use of real data is often not **broad coverage** but **discovering failure modes the simulator has not modeled**, and then having sim amplify them synthetically —

$$\text{discover real tail} \rightarrow \text{identify structure} \rightarrow \text{synthetically amplify} \rightarrow \text{real validation}$$

That is, **real is used to discover, sim is used to amplify, and real is used again to validate**.

Following this logic, we can answer the counter-question the whole article has almost dodged but that the framework itself allows: **when is the optimal move to not do sim-to-real at all?**
- **When real data is already so cheap that $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}^{\mathrm{effective}}$** — here $C_{\mathrm{real}}^{\mathrm{effective}}$ is the **effective real-robot cost** (safety / operator / reset / wear / failure recovery / deployment diversity). The correct comparison is "expected cumulative value / cost within the current budget horizon," not the "raw hours of one intervention."
- **When the simulator's model class itself is bad** ($\Delta_{\mathrm{model}}$ dominates and is hard to parameterize — soft bodies / fluids / complex contact) — fixing the sim has such low marginal utility that a world model or real-data learning is often cheaper.
- **When the deployment distribution is very fixed** — no need for large-scale DR; targeted real fine-tuning is usually more cost-effective.
- **When the simulator offers no unique coverage / safety / exploration / counterfactual access** — $U_{\mathrm{sim}}^{\mathrm{downstream}} < C_{\mathrm{sim}}^{\mathrm{effective}}$: not that sim is "bad," but that it provides no **unique utility** and its opportunity cost exceeds its benefit.

Being willing to admit "sometimes the optimal move is not doing sim-to-real" is exactly what the allocation framing should look like: **it does not take the "simulation" team; it takes the "next unit of budget buys the most real-world performance" team.**

## What this means: a loop, not a switch

The core sentence of [Data Scaling for Robots](/en/articles/2026-09-09-robot-data-scaling/) is evaluation-aware distribution allocation. Applied back to sim-to-real — **the utility of simulation data is never an internal property of the simulator; it is a property relative to the real evaluation distribution:**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

This explains a common frustration: "more sim data" sometimes does not help — **when the dominant bottleneck happens to be a support / fidelity mismatch between the simulator and the real evaluation distribution, the marginal return of adding same-distribution samples drops quickly; adding data cannot automatically create evaluation-relevant coverage or correct model bias**. Instead of asking "how good is my sim," ask: "in which evaluation-relevant directions is my sim close to reality and in which does it fall short? For the ones it falls short on, how sensitive are they, and which budget pushes each one most cheaply?"

Walk that line through and sim-to-real stops being a "did the transfer succeed" switch and becomes a loop with feedback:

$$\boxed{\ \text{diagnosis} \rightarrow \text{sensitivity / uncertainty} \rightarrow \text{intervention} \rightarrow \text{performance} + \text{information gains} \rightarrow \text{update }\mathcal{D}_t \rightarrow \text{re-allocate} \rightarrow\ \circlearrowleft\ }$$

The corresponding **closed-loop spine**, which fits the article's thesis better than any four-way table:

```text
              current policy π + evidence D_t
                          │
                          ▼
             ┌─── mismatch diagnosis ───┐
             │ model / obs / ctrl / dist │
             └─────────────┬─────────────┘
                           ▼
              sensitivity / uncertainty set
                           │
                           ▼
                candidate interventions  m = (type, Δb)
              ┌────────────┼────────────┬────────────┐
              ▼            ▼            ▼            ▼
             SI            DR          DA / FT      World model
              │            │            │            │
              └────────────┴─────┬──────┴────────────┘
                                 ▼
                       real evaluation (paired, CI)
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             performance gain            information gain
             (ΔJ_real)                (uncertainty shrinks)
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                    update evidence D_{t+1} → reallocate
                                 │
                                 └──────↺
```

(The last step re-enters the loop by changing the sensitivity and mismatch estimates — see the feedback loop above.)

This chain is a **resource-constrained adaptive sequential experimentation framework**: sensitivity and marginal return are both estimated via small-step experiments on real evaluation, and each round ends by deciding where the next slice of budget goes. Closing: **sim-to-real is not the choice of a transfer technique; it is the continuous decision, under the current belief, under several non-substitutable budgets and real evaluation feedback, of what the next intervention should be** — this is the theoretical spine of the article, and everything else is a corollary of it.

---

## References

The main works referenced in the text (all searchable via arXiv ID):

- Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World — Tobin et al., IROS 2017, arXiv:1703.06907
- Sim-to-Real Transfer of Robotic Control with Dynamics Randomization — Peng et al., ICRA 2018, arXiv:1710.06537
- Sim-to-Real: Learning Agile Locomotion For Quadruped Robots — Tan et al., RSS 2018, arXiv:1804.10332
- Learning Dexterous In-Hand Manipulation — Akkaya et al. (OpenAI), 2019, arXiv:1808.00177
- Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks — James et al., CVPR 2019, arXiv:1812.07252
- DiffTaichi: Differentiable Programming for Physical Simulation — Hu et al., ICLR 2020, arXiv:1910.00935
- Interactive Differentiable Simulation — Heiden et al., arXiv 2019, arXiv:1905.10706 (NeuralSim: Augmenting Differentiable Simulators with Neural Networks is another paper by the same group, ICRA 2021)
- Residual Physics Learning and System Identification for Sim-to-real Transfer of Policies on Buoyancy Assisted Legged Robots — Sontakke et al., 2023, arXiv:2303.09597
- Sim-to-Real of Soft Robots with Learned Residual Physics — Gao et al., IEEE RA-L 2024, pp. 8523–8530, arXiv:2402.01086
- Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) — Hafner et al., ICLR 2020, arXiv:1912.01603
- DayDreamer: World Models for Physical Robot Learning — Hafner et al., CoRL 2022, arXiv:2206.14176
- TD-MPC2: Scalable, Robust World Models for Continuous Control — Hansen et al., ICLR 2024, arXiv:2310.16828
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., RSS 2025, arXiv:2503.24361
- A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies — Lei et al. (Yu Lei, Minghuan Liu, Abhiram Maddukuri, Zhenyu Jiang, Yuke Zhu), arXiv preprint 2026, arXiv:2604.13645

There is not yet a widely accepted cross-task quantitative comparison in sim-to-real saying "this method is stronger" — across different tasks / hardware / fidelity ceilings, conclusions can flip entirely; the works above are more like "this method is workable for this kind of gap" samples than an extrapolatable ranking. The decomposition into four intervention lenses, the three-dimensional cut of simulator utility as a corollary of the allocation view, the error-budget constrained-allocation formalization, and the definitions of $\hat S_k^{\mathrm{int}}$ and $MV$ are all **conceptual framework and the author's reading**: these are decision statistics estimated via sensitivity experiments / ablation / small-scale real evaluation, not analytically computable from the simulator; reading co-training as data mixture and the world model as model-source replacement is likewise not a conclusion proven by controlled experiments.

---

*This article continues the two-part "data problem for embodied AI" series: the first part covered data sources and interfaces, the second covered the data scaling framework; here the camera pans to sim-to-real, reframing it from "a pile of transfer tricks" into a closed-loop allocation problem with empirical marginal utility — reattaching to the next article's sequential data allocation thread.*
