---
title: 'A Deep Dive into Sim-to-Real Methodology for Embodied AI: Treating "From Simulation to Reality" as an Error-Budget Allocation'
slug: "2026-09-10-sim-to-real-methodology"
date: 2026-09-10
draft: false
categories: ["Embodied AI", "Training Methods"]
tags: ["Embodied AI", "Sim-to-Real", "Domain Randomization", "System Identification", "Differentiable Simulation", "Residual Physics", "World Model", "Domain Adaptation", "Robot Data"]
description: 'Sim-to-real is not a single transfer trick but a closed-loop resource-allocation problem. This article reframes the reality gap as a policy-conditioned, multi-source mismatch, turns error-budget allocation into an estimable, iteratively optimizable decision framework through intervention sensitivity and empirical marginal utility, works through the mechanisms and failure boundaries of the four intervention lenses - system identification, domain randomization, domain adaptation, and real-world fine-tuning - and discusses world models, residual physics, co-training, and when the optimal answer is not to do sim-to-real at all.'
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

At first glance this is engineering intuition; in reality it is closed-loop resource allocation: under several budgets that cannot be exchanged for one another, you keep asking "where does the next dollar go, and where does it buy back the most real-world performance." What stalls teams in real projects is usually not "not knowing these methods exist" but "does this method do anything for this kind of gap, and which budget will it eat." First, one round of disambiguation for "error budget": it is **not** a pre-assigned fixed quota for each error term ($\Delta J=\sum_k \Delta_k$, paying out line by item); it is spending on **intervention actions**, using sequential allocation to progressively push down whichever mismatch is currently most valuable.

## Reality Gap: not a scalar, but a policy-conditioned mismatch

Sim-to-real is usually narrated as "train a policy in simulation and transfer it to reality." A more rigorous starting point is **two distributions**: the same $\pi$ interacting with each environment induces $p_{\mathrm{sim}}^{\pi}(\tau)$ and $p_{\mathrm{real}}^{\pi}(\tau)$, which are generally not equal:

$$p_{\mathrm{sim}}^{\pi}(\tau) \;\neq\; p_{\mathrm{real}}^{\pi}(\tau)$$

The trajectory distribution is itself **policy-induced** — it changes with $\pi$; it is not an intrinsic property of the environment. What we actually care about is not the distributional difference but its **manifest consequence on the task** — the performance difference of the same $\pi$ in the two worlds:

$$\boxed{\;\delta_J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)\;}$$

I call this the **transfer delta**, sign included: if $J$ is success rate and reality is in fact better (the simulator is more conservative, or its noise bites harder), $\delta_J$ comes out positive and, intuitively, it should not be called a gap. So I separately reserve the magnitude

$$G_J(\pi) \;=\; \big|\,\delta_J(\pi)\,\big|$$

and call it the **performance gap** — that is the semantics used for sensitivity below, so we never tangle with the sign.

**Distribution mismatch $\neq$ performance gap**: $p_{\mathrm{sim}}^{\pi} \neq p_{\mathrm{real}}^{\pi}$ does not automatically imply a large $\delta_J$; different policies have entirely different sensitivities to the same distributional gap. A policy that relies only on coarse geometry barely changes performance when you swap the friction model; in precision assembly that leans on high-frequency force feedback, the same distributional difference can be fatal.

$\delta_J(\pi)$ is a **task-relevant, policy-relevant observable consequence**. Written rigorously, one must **separate the mechanism from the induced distribution**: denote the environment's transition / observation / actuation kernels as mechanisms $M_{\mathrm{sim}}, M_{\mathrm{real}}$; under a given $\pi$ they **induce** $p_{\mathrm{sim}}^{\pi}(\tau),\ p_{\mathrm{real}}^{\pi}(\tau)$ — the probability distributions are **not the mechanism**. A cleaner writing of the gap is:

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ \mathcal{E};\ M_{\mathrm{sim}},\ M_{\mathrm{real}}\big)$$

(listing $p_{\mathrm{sim}}^{\pi}, p_{\mathrm{real}}^{\pi}$ separately inside the tuple would duplicate the semantics of $\pi$ — they are derived quantities of the mechanisms under $\pi$). The logic is **mechanism → trajectory distribution → performance**: $\mathcal{E}$ is the set of evaluation assumptions (initial-state / horizon / reward / constraints). "The simulator is very realistic" is never a meaningful verdict: the same $M_{\mathrm{sim}}$ can yield a small gap for a position-control policy and a huge gap for a force-sensitive manipulation policy. **The reality gap is a property of this tuple, not a property of the simulator.**

### Where exactly the gap sits: reality mismatch and task-specification mismatch

The first move is to unpack the multi-source gap — there are **two big families of causes**, and they cannot all be shoved under the single word "reality":

```
Sim-to-real / task mismatch
├── Reality mismatch (physical layer)
│   ├── Dynamics / contact        friction, contact, deformables, compliant structures
│   ├── Observation / estimation  sensor physics, calibration, noise, occlusion, latency, state estimation
│   ├── Actuation / timing        motor dynamics, control rate, actuator lag, comms jitter
│   └── Initial-state / env.      reset distribution, scene layout, long tail, initial conditions
└── Task-specification mismatch
    └── Objective / constraint    reward definition, safety constraints, success criterion
```

The two families have different sources and should not be simply added: reality mismatch is "simulation and reality are not the same world"; task-specification mismatch is "the objective you optimize and the objective you deploy are not the same task." **Observation and state estimation deserve their own layer** — the robot actually executes $a_t = \pi(o_t),\ o_t = h(x_t) + \epsilon$; camera calibration error, depth bias, occlusion, proprioception drift, force-sensor bias, and state-estimator latency are **not "the picture looks different" — they make the state the policy actually sees inconsistent with the state the simulator assumes is available**. In manipulation and locomotion, this "state-estimation gap" often hurts performance more than the appearance gap.

**Initial-state / environment mismatch** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$ — the reset distribution and scene layout do not match; **objective / task shift** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$ — simulation only requires grasp success, reality also requires collision avoidance. **Initial state deserves care**: if sim and real can **both produce the same $s_0$ distribution** and training simply did not cover it (the simulator can generate red and blue cups, deployment is all blue cups, the policy was trained only on red cups), that is **ordinary train-test distribution shift, not a reality gap**; only when the sim-real reset / scene **implementation itself** differs (the simulator resets always at 5 cm, reality at ±20 cm) is it environment mismatch — it **should not be unconditionally assigned to the reality gap**. Objective shift, meanwhile, is **already objective mismatch rather than a reality gap**: no matter how accurate the physics, if the reward / constraints do not line up you are not looking at "transfer failure" but at "you never evaluated the same task in the first place." Below we assume the objective is already aligned; objective mismatch is handled separately via reward shaping / constraint modeling.

## Writing "error-budget allocation" as an estimable, iteratively optimizable decision framework

With the sources unpacked, the opening intuition needs a mathematical landing. The writing below is **conceptual, not a strict theorem**: the error terms interact strongly — the simulator assumes perfect proprioception, reality has latency; neither alone is fatal, stacked they can destabilize a controller — so a safer move is to first admit an unknown coupling function $F$:

$$\boxed{\;\delta_J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}}\big)\;}$$

**This version takes $\Delta_{\mathrm{opt}}$ (optimization / learning error) out of reality gap**: they live at different levels — for the same fixed policy, if the simulated dynamics and observations are both accurate but RL never converged, $\delta_J$ is small while the policy is bad; "the policy was not trained well" ≠ "the sim-to-real gap is large." Stuffing it into $F$ blurs two things back together; they should be split into **two diagnostic quantities**:

$$\underbrace{J_{\mathrm{real}}(\pi_{\mathrm{train}}) - J_{\mathrm{sim}}(\pi_{\mathrm{train}})}_{\text{transfer delta } \delta_J}\qquad \underbrace{J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})}_{\text{real-domain learning gap}}$$

**These two quantities cannot be unconditionally summed and called "deployment loss"**: $\delta_J$ is signed, the two terms have different baselines, and their sum is neither a deployment loss nor a unified regret; they are **error sources at different levels**, to be diagnosed and attributed separately.

Only when doing engineering attribution near an operating point do we locally approximate $F$ as a weighted sum $\delta_J \approx \sum_k w_k \Delta_k$ — **this layer is a local attribution heuristic, not the article's core formula**: $w_k$ is a coefficient of the surrogate decomposition, which corresponds to $\hat S_k^{\mathrm{int}}$ below only under a particular local parameterization; there is no need to keep two competing "sensitivity" narratives at once. What actually drives decisions is the **intervention sensitivity** measured after picking an **intervention variable** $\xi_k$ for each mismatch class:

$$\hat S_k^{\mathrm{int}} \;\approx\; \frac{J_{\mathrm{real}}\big(\pi;\operatorname{do}(\xi_k+\delta)\big) - J_{\mathrm{real}}\big(\pi;\operatorname{do}(\xi_k)\big)}{\delta}$$

**Key clarification**: $\xi_k$ is **not a natural coordinate of the true gap — it is an intervention variable defined artificially for a sensitivity experiment** — latency / friction / appearance can be perturbed directly, but for camera calibration error, contact-model mismatch, or state-estimation error it is very hard to turn $\xi_k$ continuously in the real world; the $\operatorname{do}(\cdot)$ is there to remind the reader that this is an experimental intervention, not a derivative of an intrinsic quantity. **$\hat S_k^{\mathrm{int}}$ is only an auxiliary statistic for the diagnostic stage**; **the core decision quantity is $MV(m\mid b,\pi)$ in the next section**; the narrative is **diagnosis → intervention → empirical marginal utility → allocation**. One further downgrade: $\Delta_{\mathrm{model}}$ and $\Delta_{\mathrm{ctrl}}$ can even compensate each other unidentifiably (the actuator gain is wrong, and the policy quietly offsets it through its command distribution), so both **are not physical quantities analytically computable from the simulator — they are decision statistics estimated via sensitivity experiments / ablation / small-scale real evaluation**.

### The real "allocation": spend on intervention actions, not pick one method off a shelf

Up to here this is still "choosing a method." For budget allocation to be literal, the budget must be split **continuously** across the intervention axes: decompose the total budget into a vector $b=(b_1,\dots,b_K)$, where $b_k$ is the amount spent on intervention $k$ — $b_{\mathrm{SI}}=2\text{h}$, $b_{\mathrm{DR}}=10^6$ sim steps, $b_{\mathrm{real}}=4\text{h}$ real robot — not a 0/1 choice like "use SI or not." The objective is to maximize real-world performance:

$$\max_{b}\quad J_{\mathrm{real}}\big(\pi_b\big)$$

In robotics projects the budget is **not one currency**: GPU may be near unlimited while real robot-hours are scarce; you may have machine time but no engineering headcount — so the correct writing is **multi-budget constraints**, not a collapse into a scalar $B$:

$$\begin{aligned}
C_{\mathrm{real}}(b) &\le B_{\mathrm{real}}\\
C_{\mathrm{compute}}(b) &\le B_{\mathrm{compute}}\\
C_{\mathrm{eng}}(b) &\le B_{\mathrm{eng}}
\end{aligned}$$

Once the budget is a vector, the decision variable should shift from "gap" to "intervention": an engineer cannot buy "two percentage points of $\Delta_{\mathrm{model}}$"; what they can buy is 30 minutes of SI, $10^6$ sim steps, 100 real trajectories, a camera calibration, a residual model. It is more natural to define marginal utility on an intervention $m$ — **an intervention does not directly change $\Delta_k$; it changes the policy through the training process**:

$$\boxed{\;\pi_{b+m} \;=\; \operatorname{Train}\big(D_{\mathrm{sim}},\ D_{\mathrm{real}};\ m\big)\;}$$

So "where does the next dollar go" becomes a quantity defined on interventions that must be estimated step by step in the real world. **The core decision formula of this article is $MV$, not $w_k$ or $\hat S_k^{\mathrm{int}}$**:

$$\boxed{\;MV(m \mid b, \pi) \;=\; \frac{\mathbb{E}\big[\,J_{\mathrm{real}}(\pi_{b+m}) - J_{\mathrm{real}}(\pi_{b})\,\big]}{C(m)}, \qquad m^{*} = \operatorname*{arg\,max}_{m} MV(m \mid b, \pi)\;}$$

This ratio cannot be computed analytically from the simulator; it can only be estimated **sequentially** via pilot experiments / ablation / few-shot real evaluation. Two caveats belong together: **(i) uncertainty** — real-world $\Delta J$ is extremely noisy and each intervention often gets only a few pilots; for example A = +3 ± 0.5 and B = +5 ± 5: looking only at expectations you choose B, but B may be pure noise; allocation should also look at CI / posterior / **lower confidence bound (LCB)**, otherwise a high-variance intervention gets wrongly prioritized by a single lucky run. **(ii) non-linear cost** — $MV$ is a **local decision statistic; real performance is not linear in budget**: five hours of SI engineering can benefit every subsequent training run (fixed setup cost); each additional $10^6$ steps of DR saturates gradually (diminishing returns); fine-tuning barely improves on little data and only takes off past a threshold (threshold effects).

**$MV$ across different interventions is also not a fixed constant**: $MV_i = MV_i(b_{1:i-1},\ \pi_b,\ D_{\mathrm{real}})$ — doing SI first narrows the uncertainty set and DR's $MV$ falls; doing DR first yields a more robust starting point and fine-tuning's $MV$ rises; conversely, fine-tuning first and adding DR afterwards can be more conservative or even conflicting. **Interventions exhibit complementarity, substitutability, and occasional conflict simultaneously**, so this is not a "one-shot knapsack" but **resource-constrained sequential experimentation / adaptive allocation** (close to adaptive experimental design, but with no strict arms / stationary reward / regret guarantees — **do not write it as a bandit algorithm**).

There is one more subtle layer of feedback: **an intervention does not only push the gap down — it changes the policy, and thereby changes the policy's own sensitivity to the gap** — $S_k^{\mathrm{int}} = S_k^{\mathrm{int}}(\pi)$, $\pi = \pi(m)$, so $S_k^{\mathrm{new}} \neq S_k^{\mathrm{old}}$; the loop is not one-directional:

```
estimate mismatch → estimate sensitivity → intervention
       ↑                                          ↓
   re-estimate  ←  sensitivity changes  ←  policy changes
```

**This feedback loop fits the allocation thesis of this article better than any new equation**: sim-to-real is not an optimization you solve once; it is a sequential experiment where each round ends by re-estimating for the next.

Mapping each intervention to its primary compressed term and primary budget, with costs broken down along the **budget vector** (SI's bulk is actually real excitation + parameter estimation + instrumentation + simulator engineering + optimization compute):

| Intervention | Primary term compressed | Primary budget |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{eng}}$ + $C_{\mathrm{compute}}$ + a little $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | $C_{\mathrm{compute}}$ (sample efficiency) |
| Residual physics | $\Delta_{\mathrm{model}}$ (residual part) | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$ (appearance subset) | $C_{\mathrm{real}}$ (unlabeled data) + $C_{\mathrm{compute}}$ |
| Real-world fine-tuning | adaptation mechanism (changes transfer delta and real-domain learning gap simultaneously) | $C_{\mathrm{real}}$ (wear / safety) |
| World model | change model source | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Sim-and-real co-training | change $p_{\mathrm{train}}$ (mostly $\Delta_{\mathrm{dist}}$) | mixed data ($C_{\mathrm{real}}+C_{\mathrm{compute}}$) |

With this writing, the whole article is not "which of the four methods is better" but a loop: locate the dominant $\Delta_k$, judge how much it matters via sensitivity, invest one unit of budget in the intervention with the highest $MV$, measure the return in real evaluation, then decide the next unit — reconnecting to the next article's evaluation-aware distribution allocation, except that this time what is being allocated is engineering budget between simulation and reality.

## Four intervention lenses (more precisely, four relatively independent analytical dimensions)

With the framework in place, look at the tools axis by axis. SI, DR, DA, and real-world fine-tuning **are not peer categories at the same level of abstraction** — SI is model calibration, DR is training distribution manipulation, DA is representation alignment, fine-tuning is an optimization strategy; lining them up as "four method families" misleads people into four-way picking, while in reality they are **four relatively independent intervention lenses** that compose (**this article's analytical decomposition, not a community-accepted ontology**):

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

One more note: "$\times$" here is a **combinatorial space**, not mathematical orthogonality — DR touches Model / Observation / Distribution, DA can happen at input / feature / latent / policy / output, and "DA = the Representation axis" is only one abstraction layer of this article.

**The criterion for choosing a tool is not "systematic goes to SI, random goes to DR."** That aphorism is fine as a memory aid, but what SI actually does is **fit parameters under an identification objective**:

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; \mathcal{L}_{\mathrm{ID}}\big(D_{\mathrm{real}},\ f_{\mathrm{sim}}(\cdot\,;\,\phi)\big)$$

$\mathcal{L}_{\mathrm{ID}}$ can be trajectory prediction / one-step transition error / force-torque residual / state-estimation residual / likelihood or Bayesian posterior — **many classical SI methods never do trajectory distribution matching at all; they just minimize prediction error**. SI addresses **identifiable, parameterizable model mismatch**, not "everything systematic belongs to it" (actuator gain / latency / friction / mass can all be stochastic processes rather than deterministic biases); symmetrically, DR addresses **uncertainty that can be expressed by the training distribution**. The more useful partition is the continuum "**point estimate → posterior → robust randomization**":

| Nature of the mismatch | More natural tool |
| --- | --- |
| Parameterizable + identifiable | System Identification (point estimate $\hat\phi$) |
| Parameterizable but only uncertainty available | Bayesian / posterior SI → posterior-guided DR |
| Parameterizable but hard to identify / high uncertainty | Domain Randomization |
| Difficult to express with low-dimensional physical parameters, but has a structured residual | Residual learning |
| Observation / appearance mismatch | Domain Adaptation |
| Policy still has systematic residual on the target domain | Fine-tuning |

The key point: **"not precisely identifiable" and "no knowledge at all" are not the same thing** — once you have a posterior $p(\phi \mid D_{\mathrm{real}})$, the most natural move is not "then just do uniform DR" but $\phi \sim p(\phi \mid D_{\mathrm{real}})$ for **posterior-guided randomization**, stitching SI and DR into a continuous spectrum.

### Axis A — Model: system identification, differentiable simulation, and residual physics

This axis handles $\Delta_{\mathrm{model}}$, and internally contains three **distinct levels** that are usually bundled into "differentiable simulation is stronger SI":

$$x_{t+1} \;=\; \underbrace{f_{\mathrm{physics}}(x_t,a_t;\phi)}_{\text{parameterizable physics}} \;+\; \underbrace{r_\theta(x_t,a_t)}_{\text{residual}} \;+\; \epsilon_t$$

- **Differentiable simulation answers "how to optimize the model"** — it supplies the optimization interface $\partial f/\partial\phi$; DiffTaichi (Hu et al., ICLR 2020, 1910.00935) and Interactive Differentiable Simulation (Heiden et al., arXiv 2019, 1905.10706) turn parameter estimation into a gradient-based procedure.
- **System identification answers "which parameter to optimize"** — namely $\phi$. Classical SI scans parameters and fits trajectories; differentiable simulation updates $\phi$ via backprop like weights. The real workflow is often **real → identify → sim → train → real**, so a more accurate name is **real-to-sim-to-real**.
- **Residual physics answers "who explains the part the model did not explain"** — instead of forcing a calibration of $\phi$, let a network learn $r_\theta$ to fill the gap.

$r_\theta$ is only a **unified notation**: the actual residual need not sit additively on $f$ — it can be defined on state transition, force, acceleration, contact impulse, deformation field, or other simulator latents.

There is a make-or-break point most easily hidden behind the word "differentiable": **differentiability solves the optimization interface, not model class correctness**. If the simulator's contact model simply does not express a real phenomenon, then however precise the gradients, they only give you "the optimum under a wrong model." Differentiable simulation estimates $\phi$ better; it does not write the functional form of $f_{\mathrm{physics}}$ correctly for you — the parts it cannot get right go to the residual, or you abandon the premise "first build a trustworthy sim." **A commonly ignored boundary**: collision / friction / contact-mode switching are typically **nonsmooth or piecewise-smooth** — even if $\partial f/\partial\phi$ exists or a tool can provide it, there is no guarantee that the gradient is stable, that long-horizon rollouts do not explode, that the gradient at contact-mode transitions is meaningful, or that it beats derivative-free optimization; real systems often need smoothing / relaxation / specialized contact treatment.

SI has two further, finer but real pitfalls. **First, $p_{\mathrm{real}}(\tau)$ is essentially never directly accessible** — you only have a finite set of real trajectories, and the $\arg\min_\phi$ actually runs on the empirical estimate $\hat\phi=\arg\min_\phi \sum_i \ell(\tau_i^{\mathrm{sim}}(\phi),\tau_i^{\mathrm{real}})$. **Second, parameters existing $\neq$ parameters identifiable** — identifiability also depends on excitation and sensor observability: mass / damping / stiffness can produce nearly identical observable trajectories under some excitations and cannot be estimated independently.

Residual physics also needs a narrowed boundary: it is **not automatically applicable whenever "the physics function form is wrong."** The sweet spot is a residual that is relatively bounded on the target distribution ($\|r_\theta\| \ll \|f_{\mathrm{physics}}\|$), but what really matters is not that the residual be "small" — it is whether $f_{\mathrm{physics}}$ still provides a **useful structural inductive bias** (inductive bias / state representation / constraints / extrapolation prior); if $f_{\mathrm{physics}}$ is completely wrong and the residual has to carry the entire dynamics alone, you are better off just learning a model. It works best in settings like soft robots (Gao et al., RA-L 2024, 2402.01086) and buoyancy-assisted legged robots (Sontakke et al., 2023, 2303.09597), where the "trunk physics still counts and the local friction / contact / deformation has a stable residual." **One more caveat**: $r_\theta$ **is not naturally equal to "the missing physics"** — an unrestricted additive residual absorbs a large share of model error (sensor bias / actuator error / timing / calibration / reward mismatch / policy-induced artifacts) into an **error sponge**: it fits the training distribution and falls apart out of distribution; so the residual needs structural constraints (low-dimensional / sparse / force or acceleration scale / physical priors / active only in specific contact regimes), otherwise "filling the gap" turns into "filling in everything that should not be filled."

### Axis B — Data distribution: domain randomization and its family

This axis does not chase some "most accurate" $p_{\mathrm{real}}$; it makes the policy robust to a family of parameters $\{\phi\}$: randomize physics / vision / initial state / delay during training, and as long as reality lands inside that family, the policy holds up. Tobin (1703.06907) used pure visual randomization to bring sim grasp detection to the real robot; Peng (1710.06537) pushed randomization into dynamics; OpenAI's in-hand manipulation (Akkaya et al., 1808.00177) nearly took DR to its extreme — **absorbing difference not through precise calibration but through "a randomization range wide enough."**

A commonly mis-written intuition: **DR is not an "implicit ensemble"** — what is trained is a **single** shared policy $\pi_\theta$, with roughly the objective

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

whereas a classic ensemble is a set $\{\pi_1,\dots,\pi_K\}$. More precisely: **DR is a population-level optimization over a family of environment models** — it feels like ensemble training, but the result is one shared policy, not an aggregation. The formula above is a **baseline abstraction of risk-neutral average-case DR**; robust / adversarial DR can instead use $\max_\theta \min_{\phi\in\Phi} J(\pi_\theta;\phi)$ or CVaR $\max_\theta \operatorname{CVaR}_\alpha(J)$ or other risk-sensitive forms — which is what "robust randomization" below connects to. The condition for DR to actually work is **two things stacked together**. **Proxy**: the real parameter distribution has to land inside DR's support and be densely trained there, roughly $p_{\mathrm{real}}(\phi) \ll p_{\mathrm{DR}}(\phi)$ — if $\phi_{\mathrm{real}}$ is in the support but exactly in an extremely low-probability tail, the policy will still fail. **But the proxy carries an even more fundamental assumption**: it **implies that real dynamics can be expressed in the same $\phi$-parameterization**; if the simulator's model class does not contain the phenomenon at all (a contact mode is missing, deformability is not parameterized into $\phi$, the actuator dynamics has no such parameter, observation failure is not simple noise), then $\phi_{\mathrm{real}}$ cannot even be defined and the support statement **collapses at the root** — at that point the problem is model-class uncertainty, not "DR was not wide enough." **Main conclusion**: what actually determines transfer is not the parameter-space marginal support but whether the **state-action / contact occupancy $d_{\mathrm{real}}^{\pi}(s,a)$ that the policy actually visits under evaluation** sufficiently overlaps the occupancy induced by the training distribution — **parameter coverage is a necessary proxy, not a sufficient condition for deployment coverage**; friction / mass / latency can each be within range while the policy still enters a contact mode the simulator has never seen.

One layer further down: **DR is not about choosing scalar ranges, it is about designing a joint distribution**. $p(\phi_1,\phi_2) \neq p(\phi_1)p(\phi_2)$ is the norm — payload increase co-varies with actuator regime, temperature increase co-varies with motor resistance / friction / battery. Independent uniform DR is only a convenient baseline, not the natural representation of the real uncertainty set. The question returns to allocation: **the randomization distribution must align with the evaluation distribution and the objective** — overly wide or task-irrelevant randomization lowers sample efficiency and forces the policy to compromise across conflicting dynamics; but in robust / adversarial settings, appropriately enlarging the uncertainty set can actually help. **"Wider is more conservative" is not a universal rule; shape and alignment are what matter.**

"Adaptive / Automatic DR" is also not one method but a family: curriculum over randomization / adversarial DR (sampling the parameters that break the current policy) / automatic DR (adapting the range from training performance or real feedback) / posterior-based sampling / performance-driven range adaptation — mechanisms differ (some widen, some narrow, some specifically hunt hard domains), but the common thread is **avoiding over-randomization from the start**.

### Axis C — Observation / Representation: domain adaptation and observation translation

This axis handles $\Delta_{\mathrm{obs}}$: it neither calibrates physics nor randomizes, but aligns sim and real **at the observation / representation layer**. First, a declaration: **"Representation" is this article's abstraction, not a standard definition of DA** — DA in practice can act on input / feature / latent / output / policy / dynamics model, six layers, and "DA = the Representation axis" is just a naming choice to keep this axis alongside Model / Data / Optimization. Concrete mechanisms include feature-level adapters, image translation (GAN / diffusion), and the randomized-to-canonical translation network RCAN (James et al., CVPR 2019, 1812.07252) — **RCAN is better read as an example of input-level canonicalization / sim-to-sim adaptation, not a general DA representative**: it translates randomized sim images back toward a canonical clean image before feeding the downstream policy, doing sim→sim alignment and incidentally stitching Axis B's DR to this axis; it handles the part of the gap where "physics is actually close, but things just look nothing alike." Two boundaries need to be written precisely: **First, DA is only a subset of observation mismatch** — camera intrinsics/extrinsics, temporal sync, sensor bias, depth distortion, and state estimation are better handled by calibration / SI / sensor modeling; otherwise readers form a new wrong aphorism, "observation gap → DA." **Second, for policy learning, domain invariance itself is not the goal — task-relevant invariance is** — aligning $z_{\mathrm{sim}}\approx z_{\mathrm{real}}$ alone is not enough; ideally you keep $I(z;y_{\mathrm{task}})$ high while pushing $D(z_{\mathrm{sim}},z_{\mathrm{real}})$ low, which is the same statement as "overly wide DR washes out the task signal" seen from the representation layer.

### Axis D — Optimization / adaptation: real-world fine-tuning

This axis itself **is not a class of mismatch — it is an adaptation operator**: continue optimizing the policy directly on the target domain. Consequently it is **not "the last step"** — it can serve both as the closing relay after the first three axes and as an **early diagnostic or fast-adaptation tool**. Picking up the earlier distinction, fine-tuning **may change transfer delta and real-domain learning gap simultaneously**, but the two should still be diagnosed separately; the two regimes have completely different cost structures:

- **Offline / imitation:** $D_{\mathrm{real}} \to \theta$, main cost is **data collection** (one-shot, offline, reusable).
- **Online RL:** $\pi_\theta \to a \to$ real transition $\to \theta'$, main cost is **interaction + safety + hardware wear + exploration** (every step consumes physical resources).

So comparing methods cannot look only at final success rate; it must also consider **the real-robot interaction budget required to reach target performance**. A commonly cited rough indicator:

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{or}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

but it is only a **rough indicator**: it depends on the baseline (5%→10% and 80%→85% are both +5%, with entirely different meanings), and it is not true marginal efficiency. What you should actually look at is the learning curve / real samples to reach target / AULC / marginal gain per 100 trajectories,

$$MV_{\mathrm{real}} \;\approx\; \frac{J(N+\Delta N)-J(N)}{\Delta N}$$

— only this connects with the article-wide $MV$ framework. Risks are also not limited to "catastrophic forgetting / wrong priors": more common is **distribution narrowing** — real fine-tuning data is much narrower than sim ($D_{\mathrm{sim}} \to D_{\mathrm{real}}^{\mathrm{narrow}}$), so after fine-tuning the policy is better on the target slice but its robustness can actually drop — **generalization traded for specialization**. Hence $MV_{\mathrm{real}}(N)$ is **not guaranteed to stay positive**: the first 100 trajectories may buy a big jump, 100–500 decay quickly, and beyond that you may overfit or even regress — **fine-tuning itself can enter a negative marginal-return region**, which is exactly the allocation thesis: real data is not better the more you have; what matters is the current marginal value.

## Two new routes that loosen the "two given distributions" assumption

The four axes above share an implicit premise: **$p_{\mathrm{sim}}$ and $p_{\mathrm{real}}$ are two given distributions**, and your job is to calibrate, cover, align, or relay. The two routes below loosen this premise itself — they are not "the fifth and sixth transfer tricks" but a reformulation of the whole problem.

### World model: not cancelling the simulator, but replacing the simulator's source

**This article's lens disclaimer**: inside the allocation taxonomy of this article, I read the world model as a reformulation of "model source replacement" — **this is this article's analytical angle, not a standard definition of world models**. Strictly speaking, a world model can be learned from real data / sim / sim+real / video / offline data / latent dynamics / observation model / reward model / multi-modal predictive model and more; its extension is much wider than this section — this section only takes the slice "relative to physics sim, the model source has been replaced."

The [Data Scaling for Robots](/en/articles/2026-09-09-robot-data-scaling/) article already discussed world models and data utility. Placed into the sim-to-real context, first correct a positioning misreading: **the world model does not naturally belong to sim-to-real** — the two routes have different causal directions:

```
Physics-sim route：  hand-designed dynamics  → train / optimize → deploy real
Learned-model route：interaction data → learned dynamics → imagine → optimize
```

**Interaction data can come from real, sim, or a mixture** — the learned-model route ≠ real-only learning: sim pretrain, real adaptation, sim+real joint, and physics + learned residual are all its instantiations.

To say it precisely: the world model **does not cancel the simulator** — it swaps the simulator from "a hand-specified physics model" to "a predictive model learned from interaction data"; what changes is the **model source**:

$$f_{\mathrm{hand\text{-}designed}} \;\longrightarrow\; f_{\mathrm{learned}}$$

Dreamer (1912.01603) and TD-MPC2 (2310.16828) embody this route. When **the model bias of a hand-crafted simulator is too large to be worth fixing first**, the world model offers a rewrite of the sim-to-real problem itself, not a transfer technique under it. DayDreamer (2206.14176) is often misread as "sim pretraining → real fine-tuning"; the more accurate statement is: **it demonstrates a real-interaction-driven experimental route** — learning a world model directly on a real robot and doing policy improvement via latent imagination, without relying on a handcrafted physics simulator to pretrain the policy. But that is DayDreamer's choice, not the definition of the world-model family; and **not depending on a handcrafted simulator ≠ model-free** — world-model learning still eats its full share of assumptions (representation / architecture / action space / reward / exploration / real-data quality); it merely moves the inductive bias from "explicit physics" into the "learned world model."

The honest boundary: "learning dynamics from real data" **does not mean it is naturally better than simulation**: it swaps "hand-modeling cost" for "real collection + model capacity cost"; in contact-rich, long-tail, sensor-noisy settings, learned models often give **very confident and very wrong imagination** out of distribution — this is **yet another trade-off** between "handcrafted sim" and "direct real-world RL," not the endgame.

### Sim-and-real co-training: reframing "transfer" as data mixture

Maddukuri et al. (RSS 2025, 2503.24361) proposed Sim-and-Real Co-Training as a pragmatic direction. **What the paper actually reports**: mixing sim and real within one training run yields an **average aggregate relative improvement of roughly 37.9%** over baselines on **two platforms and six visual manipulation tasks** — note this is a **paper-defined aggregate improvement metric** that varies with task definition; it is not a percentage-point gain in success rate, and it is not an absolute difference like "50% → 87.9%"; check the per-task numbers against the original paper. It does not do one-way sim→real transfer but is a single recipe that decides the ratio and schedule between the two.

**This article's reading (not something the paper proves)** is to push it one step further into a **data-mixture problem**: once framed that way, DR and real data are no longer substitutes but two sources of $T_R[p_{\mathrm{raw}}]$ on the same sampling distribution (echoing the previous article's "$p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$"). Co-training's **primary intervention variable is the training mixture** $p_{\mathrm{train}}=\lambda\, p_{\mathrm{sim}}+(1-\lambda)\, p_{\mathrm{real}}$, not simulator calibration and not a deployment-time adapter; **$\lambda$ itself is only a sampling-level simplification** — real recipes also change the **effective** training distribution through dataset size / batch construction / importance weighting / augmentation / loss weighting / curriculum. So do not reduce it to "moving only the data-distribution axis": follow-up mechanistic analysis (Lei et al., arXiv 2026, 2604.13645) shows that changing the mixture also induces **structured representation alignment and importance reweighting** — "mixture as the primary lever, with effects spanning multiple dimensions," rather than a fifth axis strictly orthogonal to the previous four.

## Evaluation: how do you know you actually closed the gap?

A dangerous practice is reporting performance only on sim benchmarks — that measures the consistency between the policy and its own simulator, not between the policy and the real world. A more credible evaluation should at least:

- report **zero-shot transfer** (no real fine-tuning) performance together with curves after **few-shot / N-shot** adaptation;
- test on a set of **held-out hardware / calibration / object / contact / environmental regimes** (unseen object / payload / surface / camera / battery level / temperature / reset regime...) rather than only "that one deployed robot" — most labs cannot afford to swap physical systems, but they can absolutely hold out these regimes;
- explicitly declare whether the **task / initial-state / evaluation distributions** match between sim and real (objective mismatch must be aligned first);
- do **failure attribution**: which layer of $\Delta_k$ dominates? Sensitivity and remediation cost differ wildly across layers; get attribution wrong and the budget goes to the wrong place;
- **do not report means alone**: at least mean ± CI across multiple seeds / resets; prefer **paired evaluation** (compare A and B on the same object / initial state / scene / seed), stripping environment noise out of the comparison;
- **report safety failures separately**: mixing emergency stops and grasp failures into one success rate hides the true deployment cost — $J_{\mathrm{real}}$ should be reported alongside safety violation / e-stop / intervention count / hardware fault / recovery time.

Following the line "the simulator is a proxy for the real world," there is a question more fundamental than numerical alignment: **can the simulator correctly predict which policy is better?**

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

In sim it looks like $A > B > C$; on the real robot it is $B > C > A$. Here the simulator is not just carrying calibration error — it has **lost model-selection utility** — you would use it to pick out the worst policy. What engineering truly dreads is "I trusted sim, and its top-1 was wrong." So **when the simulator is used for policy / model selection**, look at rank correlation $\rho_{\mathrm{rank}} = \mathrm{Spearman}(J_{\mathrm{sim}}(\pi_i), J_{\mathrm{real}}(\pi_i))$ together with selection regret:

$$\pi_{\mathrm{sim}} = \operatorname*{arg\,max}_{\pi \in \Pi} J_{\mathrm{sim}}(\pi), \qquad R_{\mathrm{select}} = J_{\mathrm{real}}\big(\pi^{*}_{\mathrm{real}}\big) - J_{\mathrm{real}}\big(\pi_{\mathrm{sim}}\big)$$

Spearman = 0.95 but a wrong top-1 is still a disaster; conversely, Spearman = 0.7 but a top-1 that rarely misses is enough for "pick one deployable policy." Both are **conditional metrics**, relevant only to the "use sim to pick a policy" job. **From this we distill the key judgment**: **simulator fidelity is task-of-use dependent, not an absolute property** — when sim is used for policy selection, calibration error may not be the first priority; decision-relevant ranking and regret are; change the use (pretraining / exploration / curriculum / safety filter) and "which errors matter" changes entirely. **Also**: $\pi^*_{\mathrm{real}}$ is typically unavailable, so $R_{\mathrm{select}}$ can only serve as a **conceptual target metric** — in practice use a best-observed or Pareto-best proxy within the candidate set.

At this point, **another important judgment of this article** (besides the allocation framework): **simulator utility is not a property but three non-substitutable dimensions** —

| Simulator utility dimension | Typical metric |
| --- | --- |
| Numerical prediction accuracy (absolute error / calibration) | MAE / RMSE $\mathbb{E}\big[|J_{\mathrm{sim}}(\pi) - J_{\mathrm{real}}(\pi)|\big]$, calibration curve, error distribution across operating regimes; when the simulator outputs uncertainty, prediction interval coverage can be assessed |
| Ranking accuracy | Spearman $\rho_{\mathrm{rank}}$, Kendall $\tau$ |
| Quality of the selected policy (decision quality) | $R_{\mathrm{select}} = J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$ ($\pi^*_{\mathrm{real}}$ is typically unknowable; in practice use a best-observed proxy) |

A simulator can be calibrated very accurately and still pick the wrong policy (narrow distribution); another can be numerically wrong across the board yet rank stably with small regret — the three dimensions cannot substitute for each other. Precisely for this reason $U_{\mathrm{sim}}$ should not be written as an abstract scalar; it should be expanded into **utility classified by use**:

$$U_{\mathrm{sim}} \;\in\; \big\{\ U_{\mathrm{pretrain}},\ U_{\mathrm{selection}},\ U_{\mathrm{exploration}},\ U_{\mathrm{curriculum}},\ U_{\mathrm{safety}}\ \big\}$$

Evaluating fidelity cannot stare at a single policy; it must be relative to the **candidate policy family** and the **concrete use**: $U_{\mathrm{sim}}(\cdot \mid \Pi_{\mathrm{candidate}},\ p_{\mathrm{eval}}^{\mathrm{real}})$. **"Our simulator is very realistic" is never a meaningful verdict** — it answers neither **which dimension, nor which use**.

## Composition, decision, and a question usually dodged

With priorities in hand, "when to use which" should not be a fixed pipeline but a lookup table. Real projects often satisfy several conditions at once, so a more useful shape is a **gap × modellability × real-budget** decision matrix (the "Real data" column corresponds to the budget vector $B$):

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
| mixed | mixed | mixed | co-training candidate (verify positive-transfer conditions first: mixture ratio and cross-domain alignment) |

**The qualifiers in the first two rows cannot be dropped**: if the uncertainty comes from **model-class uncertainty** (the simulator's functional form itself cannot express the real phenomenon), neither SI nor DR may apply, and you have to fall to the residual / world model / real-data rows first — defaulting "dynamics bias / uncertainty" to "parameterizable" is a common aphorism-shaped error.

The second-to-last row: "model unknown" alone does not imply a world model; the criterion is **model uncertainty × real-data budget** — a learned world model is reasonable only when the model class is uncertain **and** real interaction is abundant; when real data is scarce, keeping physics prior + residual / DR is often safer. The last row is the same: "co-training as a safety net" clashes with the allocation thesis — when sim quality is bad, real data is scarce, and the two sides disagree on action space / task semantics, negative transfer is entirely possible, so positive-transfer conditions must be verified first. **Read the two "unknown long-tail" rows together**: often the highest-value use of real data is not **broad coverage** but **discovering failure modes the simulator has not modeled**, and then having sim amplify them synthetically —

$$\text{discover real tail} \rightarrow \text{identify structure} \rightarrow \text{synthetically amplify} \rightarrow \text{real validation}$$

That is, **real is used to discover, sim is used to amplify, and real is used again to validate** — every unit of budget spent on the segment where its own $MV$ is highest.

A common combo is **SI → DR → DA → co-training / fine-tune**: SI calibrates an "80% right" sim, DR widens the family along "hard to name but enumerable" directions, DA handles the visual domain gap, and a bit of real data closes it out. **The arrows are only a schematic of composition, not a fixed workflow** — the real order is determined by the currently dominant gap and by marginal utility: when real data is abundant, doing SI first may not be worth it; when vision dominates, DA should move earlier; when SI has very little data, doing a coarse DR first and coming back to calibrate is often more reasonable.

Following this logic, we can answer the counter-question the whole article has almost dodged but that the framework itself allows: **when is the optimal move to not do sim-to-real at all?**
- **When real data is already so cheap that $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}^{\mathrm{effective}}$** — here $C_{\mathrm{real}}^{\mathrm{effective}}$ is the **effective real-robot cost** (not just collection: safety / operator / reset / wear / failure recovery / deployment diversity and reproducibility are all in there); only with everything counted does a "4 real hours vs 20 sim hours" comparison stop being distorted. **But the two sides of the inequality should not be read as one-off absolute hours**: those 2 hours of SI may make a future 100 hours of simulation valuable, and DR's returns also accumulate across rounds — **the correct comparison is expected cumulative value / cost within the current budget horizon**, not the "raw hours of one intervention," consistent with the fixed setup cost + diminishing returns discussed above.
- **When the simulator's model class itself is bad** ($\Delta_{\mathrm{model}}$ dominates and is hard to parameterize — soft bodies / fluids / complex contact) — fixing the sim has such low marginal utility that going to a world model or learning on real data is often cheaper.
- **When the deployment distribution is very fixed** — you simply do not need large-scale DR to cover a whole family; a bit of targeted real fine-tuning is usually more cost-effective.

Being willing to admit "sometimes the optimal move is not doing sim-to-real" is exactly what the allocation framing should look like: **it does not take the "simulation" team; it takes the "next unit of budget buys the most real-world performance" team.**

## What this means: a loop, not a switch

The core sentence of [Data Scaling for Robots](/en/articles/2026-09-09-robot-data-scaling/) is evaluation-aware distribution allocation: under a limited budget, put each unit where the marginal data value is highest. Applied back to sim-to-real, a natural corollary falls out — **the utility of simulation data is never an internal property of the simulator; it is a property relative to the real evaluation distribution:**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

This explains a common frustration: "more sim data" sometimes does not help. But to say it precisely: more sim data can also bring wider coverage / more objects / higher rare-event frequency, so this claim is **conditional** — **when the dominant bottleneck happens to be a support / fidelity mismatch between the simulator and the real evaluation distribution, the marginal return of adding same-distribution samples drops quickly; adding $N$ improves sampling density but cannot automatically create evaluation-relevant coverage or correct model bias**. Instead of asking "how good is my sim," ask the opening question: "in which evaluation-relevant directions is my sim close to reality and in which does it fall short? For the ones it falls short on, how sensitive are they, and which budget pushes each one most cheaply?"

Walk that line through and sim-to-real stops being a "did the transfer succeed" switch and becomes a loop with feedback:

$$\boxed{\ \text{mismatch} \rightarrow \text{sensitivity} \rightarrow \text{intervention} \rightarrow \text{marginal utility} \rightarrow \text{budget allocation} \rightarrow \text{real evaluation} \rightarrow\ \circlearrowleft\ }$$

(The last step re-enters the loop by changing the sensitivity and mismatch estimates — see the feedback loop above.)

This chain is not something you analytically run through the simulator once to get an answer; it is a **resource-constrained adaptive sequential experimentation framework**: sensitivity and marginal return are both estimated via small-step experiments on real evaluation, and each round ends by deciding where the next slice of budget goes. The closing sentence: **sim-to-real is not a transfer trick; it is a constrained, iteratively estimable allocation problem across model fidelity, training diversity, representation alignment, real interaction, and engineering cost** — the same thing as the next article's "data scaling is sequential data allocation."

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

There is not yet a widely accepted cross-task quantitative comparison in sim-to-real saying "this method is stronger" — across different tasks / hardware / fidelity ceilings, conclusions can flip entirely; the works above are more like "this method is workable for this kind of gap" samples than a ranking that extrapolates. The decomposition into four intervention lenses / analytical dimensions, the error-budget constrained-allocation formalization, and the definitions of $\hat S_k^{\mathrm{int}}$ and $MV$ are all **conceptual framework and the author's reading**: $\hat S_k^{\mathrm{int}}$ and $MV$ are decision statistics estimated via sensitivity experiments / ablation / small-scale real evaluation, not quantities analytically computable from the simulator; reading co-training as data mixture and the world model as model-source replacement is likewise not a conclusion proven by controlled experiments.

---

*This article continues the two-part "data problem for embodied AI" series: the first part covered data sources and interfaces, the second covered the data scaling framework; here the camera pans to sim-to-real, reframing it from "a pile of transfer tricks" into a closed-loop allocation problem with empirical marginal utility — reattaching to the next article's sequential data allocation thread.*
