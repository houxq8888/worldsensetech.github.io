---
title: 'Sim-to-Real Methodology for Embodied AI: Treating "From Simulation to Reality" as an Error-Budget Allocation'
slug: "2026-09-10-sim-to-real-methodology"
date: 2026-09-10
draft: false
categories: ["Embodied AI", "Training Methods"]
tags: ["Embodied AI", "Sim-to-Real", "Domain Randomization", "System Identification", "Differentiable Simulation", "Residual Physics", "World Model", "Domain Adaptation", "Robot Data"]
description: 'Sim-to-real is usually presented as a single transfer trick, but it is really a closed-loop resource-allocation problem. This post reframes the reality gap as a policy-conditioned, multi-source mismatch, turns "error-budget allocation" into an estimable, iteratively refineable decision framework via weighted sensitivities and empirical marginal utility, walks through the mechanisms and failure boundaries of four relatively independent intervention axes (system identification, domain randomization, domain adaptation, real-world fine-tuning), disentangles world models, residual physics, and sim-and-real co-training, and finally answers a question usually dodged: when is the optimal move to not do sim-to-real at all?'
toc: true
related_articles:
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
  - 2026-08-25-dreamer-explained
---

> This piece follows [Data Sources and Interfaces](/en/articles/2026-09-08-data-and-training-recipes/) and [Data Scaling for Robots](/en/articles/2026-09-09-robot-data-scaling/). The first article used a single diagram to split sim-to-real into four tool families, but that was only a taxonomy. The question this article actually wants to answer can be compressed into one sentence—

> **When your simulation data falls short of the real world along several evaluation-relevant directions, which lever should the next unit of budget (engineering time, compute, or robot-hours) go to: calibrating the simulator, widening the training distribution, aligning representations, or just going out and collecting real data?**

That question looks like engineering intuition, but it is really a closed-loop resource-allocation problem: given several budgets that are not interchangeable, you keep asking "where does the next dollar buy the most real-world performance." What this article wants to do is push that intuition from a metaphor into a decision framework with marginal utility. In a real project, what stalls teams is usually not "not knowing these methods exist," but "not knowing whether this method works for my kind of gap, and which budget it will eat."

First, disambiguate the "error budget" in the title: it does **not** mean pre-assigning each error term a fixed quota ($\Delta J=\sum_k \Delta_k$, paying each line item separately). It means spending the budget on **intervention actions**, using sequential allocation to progressively push down whichever mismatch is currently most valuable. Errors are the thing we compress; budget is the thing we allocate.

## Reality Gap: not a scalar, but a policy-conditioned mismatch

Sim-to-real is usually narrated as "train a policy in simulation and transfer it to reality." A more rigorous starting point is **two distributions**: a fixed policy $\pi$ interacting with each environment induces trajectory distributions $p_{\mathrm{sim}}^{\pi}(\tau)$ and $p_{\mathrm{real}}^{\pi}(\tau)$, which are generally not equal:

$$p_{\mathrm{sim}}^{\pi}(\tau) \;\neq\; p_{\mathrm{real}}^{\pi}(\tau)$$

The trajectory distribution is **policy-induced** — it changes when $\pi$ changes; it is not an intrinsic property of the environment. What we actually care about is not the distributional gap itself but its **consequence on a task**: the performance gap of the same $\pi$ in the two worlds,

$$\boxed{\;\delta_J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)\;}$$

I call this the **transfer delta**, keeping the sign: if $J$ is success rate and reality is in fact better (e.g., the simulator is more conservative, or noise inside sim is harsher than in reality), $\delta_J$ is positive and it is not intuitive to call that a "gap." So I reserve the magnitude

$$G_J(\pi) \;=\; \big|\,\delta_J(\pi)\,\big|$$

as **performance gap**. When we talk sensitivity below, $G_J$ is the semantics we use, so we do not get tangled with sign.

Note **distribution mismatch is not the same as performance gap**: $p_{\mathrm{sim}}^{\pi} \neq p_{\mathrm{real}}^{\pi}$ does not automatically mean a large $\delta_J$, because different policies have very different sensitivities to the distributional gap. A policy that only relies on coarse geometry may barely change performance when you swap the friction model; a fine assembly policy that leans on high-frequency force feedback can be broken by the same distributional gap.

$\delta_J(\pi)$ is a **task-relevant, policy-relevant observable consequence**, and it depends on at least four things simultaneously:

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ p_{\mathrm{sim}},\ p_{\mathrm{real}},\ \mathcal{E}\big)$$

$\mathcal{E}$ is the set of evaluation assumptions — it fixes the protocol (which initial states, horizon, reward, constraints you score under) — whereas $p_{\mathrm{sim}},p_{\mathrm{real}}$ are the **mechanisms that generate data**; sim-vs-real inconsistency in the observation model belongs to the latter, while "we agreed to evaluate on the same set of initial states" belongs to the former, and the two should not be conflated. So "our simulator is very accurate" is never a meaningful claim on its own: the same simulator may induce a small gap for a position-control policy and a huge gap for a force-sensitive manipulation policy. Reality gap is a property of the four-tuple, not of the simulator alone.

### Where exactly the gap sits: reality mismatch vs. task-specification mismatch

The first move is to unpack the multi-source gap — it actually has **two big families of causes**, and they cannot all be shoved under the word "reality":

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

The two families have different sources and should not be simply added: reality mismatch is "simulation and reality are not the same world," while task-specification mismatch is "the objective you optimize and the objective you deploy are not the same task at all."

**Observation and state estimation** deserve their own layer rather than being folded into "perception gap." The robot actually executes

$$a_t = \pi(o_t), \qquad o_t = h(x_t) + \epsilon$$

and in reality camera calibration error, depth bias, occlusion, proprioception drift, force-sensor bias, and state-estimator latency are **not just "the picture looks different"** — they cause **the state the policy actually sees to be inconsistent with the state the simulator assumed to be available**. In manipulation and locomotion, this "state-estimation gap" often hurts performance more than the appearance gap.

**Initial state** and **task objective** must also be split, since they belong to different families:

- **Initial-state / environment shift (belongs to reality mismatch):** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$ — reset distribution and scene layout do not match; this is a distribution problem inside the simulation.
- **Objective / task shift (belongs to task-specification mismatch):** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$ — sim only requires grasp success, reality also requires collision avoidance; sim tolerates larger penetration, but hardware safety does not.

Strictly speaking, the latter **is already not a reality gap but an objective mismatch**: no matter how accurately the simulator models physics, if the reward / constraints do not match the real objective, you are not looking at "transfer failure" but at "you never evaluated the same task in the first place." Below, when discussing gap compression, we assume objective is already aligned; objective mismatch has to be handled separately via reward shaping / constraint modeling.

## Writing "error-budget allocation" as an estimable, iteratively refineable decision framework

Once the sources are unpacked, the intuition from the opening needs a mathematical landing. What follows is **conceptual, not a strict theorem**: error terms interact strongly — the simulator assumes perfect proprioception, reality has latency; neither alone is fatal, together they can destabilize a controller. So a more careful move is to first admit an unknown coupling function $F$:

$$\boxed{\;\delta_J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}}\big)\;}$$

**This version removes $\Delta_{\mathrm{opt}}$ (optimization / learning error) from reality gap.** Different levels: for **the same fixed policy**, if simulation, observation, and dynamics are all accurate but RL has not converged, $\delta_J$ is actually small (both sides score about the same) while the policy is bad. "Policy is not well trained" ≠ "sim-to-real gap is large" — stuffing $\Delta_{\mathrm{opt}}$ into $F$ blurs the two back together. The right move is to keep them conceptually separate (rather than forced-additive):

$$\underbrace{J_{\mathrm{real}}(\pi_{\mathrm{train}}) - J_{\mathrm{sim}}(\pi_{\mathrm{train}})}_{\text{transfer gap}}\qquad \underbrace{J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})}_{\text{learning / optimization gap}}$$

$$\text{deployment loss} \;=\; \text{transfer gap} \;+\; \text{learning / optimization gap}$$

The first answers "how much does the simulator mislead the already-trained policy"; the second answers "how well did the policy itself learn." This distinction also cleans up the positioning of fine-tuning later on.

Only for near-operating-point engineering attribution do we locally approximate $F$ as a weighted sum — and the weights $w_k$ are precisely the **task / policy dependent** sensitivities, which echoes the previous section's claim that "gap is policy-conditioned":

$$\delta_J \;\approx\; \sum_{k} w_k\, \Delta_k, \qquad S_k \;=\; \left|\frac{\partial J_{\mathrm{real}}}{\partial \Delta_k}\right|$$

$S_k$ answers "is this class of gap worth caring about." Each $\Delta_k$ can be pushed down by some intervention at some cost; write the efficiency as $\partial \Delta_k / \partial C_k$, giving a clean priority $\text{priority}_k \propto S_k \cdot \partial\Delta_k / \partial C_k$ — **fix first the class of gap that is both most task-relevant and cheapest to shrink.**

But here we need to downgrade the mathematical narrative a notch: $\Delta_k$ are not three free knobs (dynamics error / latency / observation error); $\Delta_{\mathrm{model}}$ and $\Delta_{\mathrm{ctrl}}$ can even compensate each other unidentifiably (if the actuator gain is wrong in sim, the policy can quietly re-tune it via a different command distribution). So $S_k$ (and next section's $MV$) **should not be read as quantities analytically computable from the simulator; they are decision statistics estimated via sensitivity experiments / ablation / small-scale real evaluation.** Admitting this makes the framework stronger, not weaker: it is not "an analytically solvable optimization formula" but "an allocation procedure whose marginal returns are estimated by sequential experiments."

### Real "allocation": spend on intervention actions, not pick one from a shelf of methods

So far this is still "picking a method." To make "budget allocation" literal, we must let budget flow **continuously** onto each intervention axis: split the total budget into a vector $b=(b_1,\dots,b_K)$, where $b_k$ is the amount spent on intervention $k$ — $b_{\mathrm{SI}}=2\text{h}$, $b_{\mathrm{DR}}=10^6$ sim steps, $b_{\mathrm{real}}=4\text{h}$ real robot — not a 0/1 choice like "SI or not." The objective is to maximize real-world performance:

$$\max_{b}\quad J_{\mathrm{real}}\big(\pi_b\big)$$

The key point is that budget in a robotics project is **not one currency**: GPU may be near unlimited but real robot-hours are scarce; you may have machine time but no engineering headcount. So the correct writing is a **multi-budget constraint**, not collapsing everything into a scalar $B$:

$$\begin{aligned}
C_{\mathrm{real}}(b) &\le B_{\mathrm{real}}\\
C_{\mathrm{compute}}(b) &\le B_{\mathrm{compute}}\\
C_{\mathrm{eng}}(b) &\le B_{\mathrm{eng}}
\end{aligned}$$

Once the budget is a vector, the decision variable should shift from "gap" to "intervention." An engineer cannot buy "two percentage points of $\Delta_{\mathrm{model}}$"; what they can buy is: 30 minutes of SI, another $10^6$ sim steps, 100 real trajectories, a camera calibration, a latency-randomization pass, a residual-model training run. So the more natural formulation is to define marginal utility on interventions $m$ — **an intervention does not directly touch $\Delta_k$; it changes the policy through the training process**:

$$\boxed{\;\pi_{b+m} \;=\; \operatorname{Train}\big(D_{\mathrm{sim}},\ D_{\mathrm{real}};\ m\big)\;}$$

Then "where does the next dollar go" becomes a quantity defined on interventions that must be estimated sequentially in the real world:

$$\boxed{\;m^{*} \;=\; \operatorname*{arg\,max}_{m}\; \frac{\mathbb{E}\big[\,J_{\mathrm{real}}(\pi_{b+m}) - J_{\mathrm{real}}(\pi_{b})\,\big]}{C(m)}\;}$$

This ratio is not analytically solvable from the simulator; it must be **sequentially estimated** via pilot experiments / ablation / few-shot real evaluation. It is naturally a **sequential empirical decision** (close to bandit / active experimentation), and it reconnects with the previous article's sequential data allocation. Since $MV$ typically diminishes with spend, we also get the intuition for why **the first 2 hours of SI feel worth it but another 20 hours may not**: the parameters that are easiest to identify and matter most are already calibrated in the first 2 hours, and the remaining 18 are often better spent on DR or real data collection.

Mapping each intervention to its primary compressed term and primary budget gives us this table — cost is broken along the **budget vector** (SI's bulk is actually real excitation + parameter estimation + instrumentation + simulator engineering + optimization compute, not "simulator fidelity cost"):

| Intervention | Primary term compressed | Primary budget |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{eng}}$ + $C_{\mathrm{compute}}$ + a little $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | $C_{\mathrm{compute}}$ (sample efficiency) |
| Residual physics | $\Delta_{\mathrm{model}}$ (residual part) | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$ (appearance subset) | $C_{\mathrm{real}}$ (unlabeled) + $C_{\mathrm{compute}}$ |
| Real-world fine-tuning | transfer residual + task-learning gap | $C_{\mathrm{real}}$ (wear / safety) |
| World model | change model source | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Sim-and-real co-training | change $p_{\mathrm{train}}$ (mostly $\Delta_{\mathrm{dist}}$) | mixed data ($C_{\mathrm{real}}+C_{\mathrm{compute}}$) |

With this writing, the whole article is not "which of four methods wins" but a loop: locate the dominant $\Delta_k$, use sensitivity to judge how much it matters, spend a slice of budget on the intervention with the highest estimated $MV$, measure how much $J_{\mathrm{real}}$ bought back, then decide the next slice. That closes the loop with the previous article's evaluation-aware distribution allocation — only this time the object being allocated is engineering budget across simulation and reality.

## Four intervention axes (more precisely, four relatively independent intervention dimensions)

With the framework, look at the tools. First, set up a structure: SI, DR, DA, and real-world fine-tuning **are not peer categories at the same level of abstraction** — SI is model calibration, DR is training distribution manipulation, DA is representation alignment, fine-tuning is an optimization strategy. Presenting them as four parallel methods misleads people into picking one; they are actually **four relatively independent intervention dimensions** that compose:

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

But honestly add: the "$\times$" here denotes a **combinatorial space**, not orthogonality in the mathematical sense. DR randomizes physics, vision, initial state, and delay, simultaneously touching Model / Observation / Distribution; DA can act on input / feature / latent dynamics / policy / output — "DA = representation axis" is also an abstraction, not a strict definition. The honest phrasing is "**relatively independent, composable dimensions**," not "mutually orthogonal axes."

**The tool-selection rule is not "systematic goes to SI, random goes to DR."** That aphorism is fine as memory, but SI actually estimates

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; D\big(p_{\mathrm{sim}}(\tau \mid \phi),\; p_{\mathrm{real}}(\tau)\big)$$

which addresses **identifiable, parameterizable model mismatch**, not "everything systematic belongs to it" (actuator gain, latency, friction, and mass may be stochastic processes rather than deterministic biases). Symmetrically, DR addresses **uncertainty that a training distribution can express**. So the more useful partition is not a binary but a continuous spectrum "point estimate → posterior → robust randomization":

| Nature of the mismatch | More natural tool |
| --- | --- |
| Parameterizable + identifiable | System Identification (point estimate $\hat\phi$) |
| Parameterizable but only a posterior is available | Bayesian / posterior SI → posterior-guided DR |
| Parameterizable but hard to identify / high uncertainty | Domain Randomization |
| Not parameterizable but has residual structure | Residual learning |
| Observation / appearance mismatch | Domain Adaptation |
| Policy still has systematic residual on target domain | Fine-tuning |

The key is: **"not precisely identifiable" and "no knowledge at all" are not the same thing.** Once you have a posterior $p(\phi \mid D_{\mathrm{real}})$, the natural move is not "since it is uncertain, do uniform DR" but $\phi \sim p(\phi \mid D_{\mathrm{real}})$ for **posterior-guided randomization** — this stitches SI and DR into a continuous spectrum, and fits the article's allocation theme.

### Axis A — Model: system identification, differentiable simulation, residual physics

This axis handles $\Delta_{\mathrm{model}}$ and internally contains three **distinct levels** of question, usually conflated under "differentiable simulation is stronger SI":

$$x_{t+1} \;=\; \underbrace{f_{\mathrm{physics}}(x_t,a_t;\phi)}_{\text{parameterizable physics}} \;+\; \underbrace{r_\theta(x_t,a_t)}_{\text{residual}} \;+\; \epsilon_t$$

- **Differentiable simulation answers "how to optimize the model"** — it supplies the optimization interface $\partial f/\partial\phi$; DiffTaichi (Hu et al., ICLR 2020, 1910.00935) and Interactive Differentiable Simulation (Heiden et al., arXiv 2019, 1905.10706) turn parameter estimation into a gradient-based procedure.
- **System identification answers "which parameter to optimize"** — namely $\phi$. Classical SI scans parameters and fits trajectories; differentiable simulation updates $\phi$ via backprop like weights. The real workflow is often **real → identify → sim → train → real**, so a more accurate name is **real-to-sim-to-real**.
- **Residual physics answers "who explains the part the model did not explain"** — instead of forcing a calibration of $\phi$, let a network learn $r_\theta$ to fill the gap.

Here $r_\theta$ is just a **unified notation**: the actual residual need not sit directly on $f$; it can be defined on state transition, force, acceleration, contact impulse, deformation field, or other simulator latents — the soft-robot paper, for example, learns a residual force applied to the whole simulated mesh.

There is a make-or-break point that is easily hidden behind the word "differentiable": **differentiability solves the optimization interface, not model class correctness.** If the simulator's contact model simply does not express a real phenomenon, then differentiating through that wrong model as precisely as you like still gives "the optimum under a wrong model." Differentiable simulation lets you estimate $\phi$ better; it does not write $f_{\mathrm{physics}}$ correctly for you — the parts it cannot get right are handed to the residual, or you abandon the premise "first build a trustworthy sim" (see world model).

SI also has two smaller but practical pitfalls. **First, $p_{\mathrm{real}}(\tau)$ is essentially not directly accessible** — you only have a finite set of real trajectories $\{\tau_i^{\mathrm{real}}\}_{i=1}^N$, so the $\arg\min_\phi D(\cdot)$ above is actually run on an empirical estimate: $\hat\phi=\arg\min_\phi \sum_i \ell\big(\tau_i^{\mathrm{sim}}(\phi),\tau_i^{\mathrm{real}}\big)$. **Second, parameters existing ≠ parameters identifiable** — identifiability depends on excitation and sensor observability: mass, damping, and stiffness can produce nearly identical observable trajectories under some excitations and cannot be estimated separately.

Residual physics also needs a narrower boundary: it is **not automatically applicable whenever the physics function form is wrong.** A common sweet spot is residual being relatively bounded on the target distribution ($\|r_\theta\| \ll \|f_{\mathrm{physics}}\|$), but the real question is not whether the residual must be "small" — it is whether $f_{\mathrm{physics}}$ still provides a **useful structural inductive bias** — inductive bias, state representation, constraints, extrapolation prior. Conversely, if $f_{\mathrm{physics}}$ is completely wrong and the residual has to carry the entire dynamics alone, you are better off just learning a model. It works best in settings like soft robots (Gao et al., RA-L 2024, 2402.01086) and buoyancy-assisted legged robots (Sontakke et al., 2023, 2303.09597) where the "trunk physics still counts, and the local friction / contact / deformation has a stable residual."

### Axis B — Data distribution: domain randomization and its family

This axis does not chase some "most accurate" $p_{\mathrm{real}}$; instead it makes the policy robust to a family of parameters $\{\phi\}$: randomize physics, vision, initial state, and delay during training, and as long as reality falls inside that family, the policy holds up. Tobin (1703.06907) used pure visual randomization to bring sim grasp detection to the real table; Peng (1710.06537) pushed randomization into dynamics; OpenAI's in-hand manipulation (Akkaya et al., 1808.00177) nearly took DR to its extreme — **absorb difference not by precise calibration but by "wide enough randomization range."**

A commonly mis-written intuition: DR is not an "implicit ensemble." It trains **a single** shared policy $\pi_\theta$, with roughly the objective

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

whereas a classic ensemble is a set of models $\{\pi_1,\dots,\pi_K\}$. A more accurate phrasing is: **DR is a population-level optimization over a family of environment models, not an optimization against a single simulator** — intuitively it resembles ensemble training, but the outcome is one shared policy, not an aggregation of many.

The condition for DR to work also has to be written carefully: "widening the range" is not enough; you generally need the real parameter distribution to be supported by DR and trained in high-density regions — approximately $p_{\mathrm{real}}(\phi) \ll p_{\mathrm{DR}}(\phi)$; if $\phi_{\mathrm{real}}$ technically falls inside the support but lands in an extremely low-probability tail, the policy was essentially never trained there and will still fail. What matters more, however, is: **for policy transfer, what really has to be covered is not the marginal support of parameter space but the state-action / contact occupancy $d_{\mathrm{real}}^{\pi}(s,a)$ the policy actually visits under evaluation** — parameter coverage ≠ deployment trajectory coverage, and **joint correlations** across parameters are hard to reproduce by independent per-dimension randomization. So the question returns to allocation: **is the randomization distribution aligned with the evaluation distribution and the objective.** Overly wide or task-irrelevant randomization lowers sample efficiency and forces the policy to compromise across conflicting dynamics; but in some robust / adversarial settings, widening the uncertainty set actually helps — so "wider is more conservative" is not a universal rule; alignment is what matters.

"Adaptive / Automatic DR" is also not one method but a family: curriculum over randomization, adversarial DR (sample parameters that break the current policy), automatic DR (adapt the sampling distribution based on training performance or real feedback), posterior-based sampling, performance-driven range adaptation — mechanisms vary (some widen, some narrow, some specifically hunt hard domains), but the common thread is **avoiding over-randomization from the start**.

### Axis C — Representation: domain adaptation and observation translation

This axis handles $\Delta_{\mathrm{obs}}$: it neither calibrates physics nor randomizes, but aligns sim and real **at the representation level** — feature-level adapters, image translation (GAN / diffusion), or the randomized-to-canonical translation network RCAN (James et al., CVPR 2019, 1812.07252) which "translates" randomized sim images back to something like a canonical clean image before feeding a downstream policy, and neatly **stitches Axis B's DR to this axis**. It handles that part of the gap where "physics is actually pretty close, but things just look nothing alike."

But two boundaries need to be written precisely: **First, DA is only a subset of observation mismatch** — especially good for appearance / representation shift, while camera intrinsics/extrinsics, temporal sync, sensor bias, depth distortion, and state estimation are better handled by calibration / SI / sensor modeling; otherwise readers will form a new wrong aphorism, "observation gap → DA." **Second, for policy learning, domain invariance is not the goal — task-relevant invariance is.** Aligning $z_{\mathrm{sim}}\approx z_{\mathrm{real}}$ alone is not enough; ideally you keep $I(z;y_{\mathrm{task}})$ high while pushing $D(z_{\mathrm{sim}},z_{\mathrm{real}})$ low, i.e., align only those variations irrelevant to the task. This is really the same statement as "overly wide DR washes out the task signal," seen from the representation layer.

### Axis D — Optimization / adaptation: real-world fine-tuning

This axis handles what is left after the first three — large-scale pre-training in sim to learn structure, then real data as the relay. Picking up the earlier distinction, fine-tuning compresses two things at once: **transfer residual** and **task-learning gap (policy itself not trained enough)** — do not describe it just as "closing reality gap." And the two regimes have completely different cost structures; do not fudge them together as "RL or imitation fine-tuning":

- **Offline / imitation:** $D_{\mathrm{real}} \to \theta$, main cost is **data collection** (one-shot, offline, reusable).
- **Online RL:** $\pi_\theta \to a \to$ real transition $\to \theta'$, main cost is **interaction + safety + hardware wear + exploration** (every step consumes physical resources).

So comparing methods cannot just look at final success rate; it must also consider **the real interaction budget needed to hit target performance**. A commonly cited rough indicator is

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{or}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

but this is only a **rough indicator**: it depends on baseline (5%→10% and 80%→85% are both +5%, with completely different meanings), and it is not real marginal efficiency. What you should actually look at is the learning curve, real samples to reach target, AULC, and marginal gain per 100 trajectories,

$$MV_{\mathrm{real}} \;\approx\; \frac{J(N+\Delta N)-J(N)}{\Delta N}$$

— only that connects with the article-wide $MV$ framework. Risks are also not limited to "catastrophic forgetting / wrong priors": more common is **distribution narrowing** — real fine-tune data is often much narrower than sim ($D_{\mathrm{sim}} \to D_{\mathrm{real}}^{\mathrm{narrow}}$), so after fine-tuning the policy looks better on the target slice but its robustness can drop, which is **trading generalization for specialization**.

## Two new routes that loosen the "two given distributions" assumption

The four axes above share an implicit premise: **$p_{\mathrm{sim}}$ and $p_{\mathrm{real}}$ are two given distributions**, and your job is to calibrate, cover, align, or relay. The two routes below loosen this premise itself — they are not "the fifth and sixth transfer tricks" but a reformulation of the whole problem.

### World model: not cancelling the simulator, but replacing the simulator's source

The [Data Scaling for Robots](/en/articles/2026-09-09-robot-data-scaling/) article already discussed world models and data utility. Placed into sim-to-real, first correct a positioning misreading: **world model does not naturally belong to sim-to-real** — the two routes have different causal directions:

```
Classic sim-to-real:  sim dynamics → train policy → deploy real
World model route:    real interaction → learn dynamics → imagine → optimize policy
```

To say it precisely: world model **does not cancel the simulator** — it swaps the simulator's role from "hand-specified physics model" to "predictive model learned from interaction data":

$$f_{\mathrm{hand\text{-}designed}} \;\longrightarrow\; f_{\mathrm{learned}}$$

What really changes is the **model source**. Dreamer (1912.01603) and TD-MPC2 (2310.16828) embody this route. When **the model bias of a hand-crafted simulator is too large to be worth fixing first**, world model offers a rewrite of the sim-to-real problem itself, not a transfer technique under it.

DayDreamer (2206.14176) is often misread as "sim pretraining → real fine-tune," but its point is actually the opposite — the world model learns directly from real interaction and does policy improvement inside latent imagination, barely relying on hand-crafted sim. But be careful: **not relying on an explicit physics simulator does not mean model-free**; world-model learning still eats its share of assumptions (representation, architecture, action space, reward, exploration, real data quality), it merely shifts inductive bias from "explicit physics simulator" to "learned world model."

The honest boundary: "learning dynamics from real data" **does not mean it is naturally better than simulation**. It swaps "modeling cost" for "real collection + model capacity cost"; in contact-rich, long-tail, sensor-noisy settings, learned models often produce **confidently wrong imagination** out of distribution, and the policy will happily plan along that error. So this is **another trade-off** between "handcrafted sim" and "direct real-world RL," not the end state.

### Sim-and-real co-training: reframing "transfer" as data mixture

Maddukuri et al. (RSS 2025, 2503.24361) proposed Sim-and-Real Co-Training as a pragmatic direction. **What the paper actually reports**: mixing sim and real datasets inside one training run yields about 38% (precisely 37.9%) average performance improvement across two robot platforms and six tasks; it is not one-way sim→real transfer but a recipe that fixes the ratio and schedule.

**This article's reading (not the paper's conclusion)** is to push it one step further into a **data-mixture problem**: once reframed that way, DR and real data are no longer substitutes but two sources of $T_R[p_{\mathrm{raw}}]$ on the same sampling distribution (echoing the previous article's "$p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$"). To write it precisely: co-training's **primary intervention variable is the training mixture** — $p_{\mathrm{train}}=\lambda\, p_{\mathrm{sim}}+(1-\lambda)\, p_{\mathrm{real}}$ — not simulator calibration and not an explicit deployment-time adapter. But do not reduce it to "only the data-distribution axis": follow-up mechanistic analyses in 2026 (Lei et al., arXiv 2026, 2604.13645) show that changing the mixture induces **structured representation alignment and importance reweighting**. So this is a route where "mixture is the primary lever, but the effects cut across multiple dimensions," not a fifth axis strictly orthogonal to the previous four.

## Evaluation: how do you know you actually closed the gap?

A dangerous practice is reporting performance only on sim benchmarks — that measures consistency between your policy and your simulator, not between your policy and the real world. A more credible evaluation should at least do the following:

- Report **zero-shot transfer** (no real fine-tuning) to the real system, plus curves after **few-shot / N-shot** adaptation;
- Test on a set of **held-out physical systems** (different calibration, cameras, contact surfaces), not just "that one deployed robot";
- Explicitly declare whether **task, initial state, and evaluation distribution** match between sim and real — otherwise comparisons are simply not fair (which is why objective mismatch must be aligned first);
- Do **failure attribution**: which layer of $\Delta_k$ dominates? Sensitivity and remediation cost differ wildly across layers; get attribution wrong and the budget goes to the wrong place.

Following "the simulator is a proxy for the real world," there is a question more fundamental than numerical alignment: **can the simulator correctly predict which policy is better?** Consider:

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

In sim it looks like $A > B > C$; on the real robot it flips to $B > C > A$. Here the simulator is not just carrying calibration error — it has **lost model-selection utility** — you would use it to pick out the worst policy. So **when the simulator is used for policy / model selection**, a more direct metric than absolute numerical error is rank correlation $\rho_{\mathrm{rank}} = \mathrm{Spearman}\big(J_{\mathrm{sim}}(\pi_i),\ J_{\mathrm{real}}(\pi_i)\big)$.

But Spearman is still not the most direct. What engineering really dreads is "I trusted sim, and it picked the wrong top-1." So also add a policy-selection regret:

$$\pi_{\mathrm{sim}} = \operatorname*{arg\,max}_{\pi \in \Pi} J_{\mathrm{sim}}(\pi), \qquad R_{\mathrm{select}} = J_{\mathrm{real}}\big(\pi^{*}_{\mathrm{real}}\big) - J_{\mathrm{real}}\big(\pi_{\mathrm{sim}}\big)$$

For example, Spearman = 0.95 but the top-1 pick is wrong is still an engineering disaster; conversely, Spearman = 0.7 but top-1 rarely misses is enough for "pick one deployable policy." So **when the simulator is used for policy selection, look at rank correlation together with top-1 / top-k selection regret.** Both are **conditional metrics**, only for "using sim to pick a policy"; sim has many other uses that do not require ranking the whole policy family (representation pretraining, exploration, curriculum, safety filtering, controller initialization, rare-event generation, and so on). Judging simulator fidelity also should not focus on a single policy but on a **candidate policy family**, writing $U_{\mathrm{sim}}=U(D_{\mathrm{sim}}\mid \Pi_{\mathrm{candidate}},\,p_{\mathrm{eval}}^{\mathrm{real}})$.

## Composition, decision, and a question usually dodged

With a priority order, "when to use which" should not be a fixed pipeline but a lookup table. Real projects often satisfy multiple conditions at once, so a more useful shape is a **gap × modellability × real-budget** decision matrix (the "Real data" column directly corresponds to the budget vector $B$ above):

| Gap | Parameterizable / identifiable? | Real data | Recommendation |
| --- | --- | ---: | --- |
| dynamics bias | high | scarce | SI |
| dynamics uncertainty | medium | scarce | DR (or posterior-guided DR) |
| dynamics residual | low (but structured) | medium | Residual learning |
| visual appearance | high | none / scarce | DA / DR |
| actuator latency | high | scarce | SI + DR |
| unknown long-tail, simulatable | low | scarce | targeted simulation / DR |
| unknown long-tail, sim untrustworthy | low | medium | real data |
| model class uncertain | low | abundant | learned world model (if real is scarce, prefer physics prior + residual / DR) |
| mixed | mixed | mixed | co-training candidate (verify positive transfer first: mixture ratio and cross-domain alignment) |

The second-to-last row: "model unknown" alone does not imply world model. The real criterion is **model uncertainty × real-data budget** — learned world model is a reasonable candidate only when the model class is uncertain **and** real interaction is abundant; when real data is scarce, keeping physics prior + residual / DR is often safer. The last row is the same: "co-training as a safety net" clashes with the article's allocation stance — when sim quality is bad, real data is scarce, and the two sides disagree on action space / task semantics, co-training can produce negative transfer; positive-transfer conditions must be verified first.

A common combo is **SI → DR → DA → co-training / fine-tune**: SI calibrates a "80% right" sim, DR widens the family along "hard to name but enumerable" directions, DA handles visual domain shift, and a bit of real data closes it out. **But the arrows are not a fixed workflow — just a schematic.** The real order is determined by the currently dominant gap and by each intervention's marginal utility: when real data is abundant, doing SI first may not be worth it; when vision dominates, DA should move earlier; when SI has very little data, doing a coarse DR first to get a running policy and coming back to calibrate is often better.

Following this logic lets us finally answer the counter-question the whole article has been tiptoeing around but the framework explicitly allows: **when is the optimal move to not do sim-to-real at all?**

- **When real data is already so cheap that $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}^{\mathrm{effective}}$** — here $C_{\mathrm{real}}^{\mathrm{effective}}$ is the **effective real-robot cost**, not just collection: safety, operator, reset, hardware wear, failure recovery, deployment diversity, and reproducibility are all in there. Only when those are added does a "4 real hours vs 20 sim hours" comparison stop being distorted across labs.
- **When the simulator's model class itself is bad** ($\Delta_{\mathrm{model}}$ dominates and is hard to parameterize) — soft bodies, fluids, complex contact — fixing sim has such low marginal utility that going to a world model or learning directly on real data is often cheaper.
- **When the deployment distribution is very fixed** — you simply do not need large-scale DR to cover a whole family; a bit of targeted real fine-tuning is usually more cost-effective.

Being willing to admit "sometimes the optimal move is not doing sim-to-real" is exactly what the allocation framing should look like: **it does not take the "simulation" team; it takes the "next unit of budget buys the most real-world performance" team.**

## What this means: a loop, not a switch

The core sentence of [Data Scaling for Robots](/en/articles/2026-09-09-robot-data-scaling/) is evaluation-aware distribution allocation: under a limited budget, put each unit where the marginal data value is highest. Applied back to sim-to-real, a natural corollary falls out — **the utility of simulation data is never an internal property of the simulator; it is a property relative to the real evaluation distribution:**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

This explains a common frustration: why "adding more sim data" sometimes does not help. But to say it precisely: more sim data can also bring wider coverage, more diverse objects, higher rare-event frequency; so the claim is **conditional** —

> **When the dominant bottleneck happens to be a support / fidelity mismatch between the simulator and the real evaluation distribution, the marginal value of adding same-distribution simulation samples drops quickly; adding $N$ at that point mostly raises sampling density, and cannot automatically create missing support or correct model bias.**

It improves density, while what is missing is support and $\Delta_{\mathrm{model}}$ fidelity. So instead of "how good is my sim," ask the opening question: **"In which evaluation-relevant directions is my sim close to reality and in which does it fall short? For the ones it falls short on, how sensitive are they, and which budget pushes each most cheaply?"**

Walk that line through and sim-to-real stops being a "did the transfer succeed" switch and becomes this loop:

$$\boxed{\text{mismatch} \rightarrow \text{sensitivity} \rightarrow \text{intervention} \rightarrow \text{marginal utility} \rightarrow \text{budget allocation} \rightarrow \text{real evaluation}}$$

But this chain is not something you analytically run through the simulator to get an answer — it is a **sequential empirical decision framework**: sensitivity and marginal return are estimated via small experiments on real evaluation, and each round decides where the next slice of budget goes. One-sentence wrap-up: **sim-to-real is not a transfer trick; it is a constrained, iteratively estimable allocation problem across model fidelity, training diversity, representation alignment, real interaction, and engineering cost.** That is the same thing as the previous article's "robot data scaling is a sequential data allocation problem" — only this time, the allocation happens between simulation and reality.

---

## References

The following are the main works referenced in the text (all searchable via arXiv ID):

- Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World — Tobin et al., IROS 2017, arXiv:1703.06907
- Sim-to-Real Transfer of Robotic Control with Dynamics Randomization — Peng et al., ICRA 2018, arXiv:1710.06537
- Sim-to-Real: Learning Agile Locomotion For Quadruped Robots — Tan et al., RSS 2018, arXiv:1804.10332
- Learning Dexterous In-Hand Manipulation — Akkaya et al. (OpenAI), 2019, arXiv:1808.00177
- Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks — James et al., CVPR 2019, arXiv:1812.07252
- DiffTaichi: Differentiable Programming for Physical Simulation — Hu et al., ICLR 2020, arXiv:1910.00935
- Interactive Differentiable Simulation — Heiden et al., arXiv 2019, arXiv:1905.10706 (the same group also has NeuralSim: Augmenting Differentiable Simulators with Neural Networks, ICRA 2021, which is a separate paper)
- Residual Physics Learning and System Identification for Sim-to-real Transfer of Policies on Buoyancy Assisted Legged Robots — Sontakke et al., 2023, arXiv:2303.09597
- Sim-to-Real of Soft Robots with Learned Residual Physics — Gao et al., IEEE RA-L 2024, pp. 8523–8530, arXiv:2402.01086
- Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) — Hafner et al., 2019, arXiv:1912.01603
- DayDreamer: World Models for Physical Robot Learning — Hafner et al., CoRL 2022, arXiv:2206.14176
- TD-MPC2: Scalable, Robust World Models for Continuous Control — Hansen et al., ICLR 2024, arXiv:2310.16828
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., RSS 2025, arXiv:2503.24361
- A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies — Lei et al. (Yu Lei, Minghuan Liu, Abhiram Maddukuri, Zhenyu Jiang, Yuke Zhu), arXiv 2026, arXiv:2604.13645

A caveat: there is not yet a widely accepted cross-task quantitative comparison saying "this method is stronger" in sim-to-real — across different tasks, hardware, and fidelity ceilings, conclusions can flip entirely; the works above are more like "this method is workable for this kind of gap" samples than a ranking that transfers across settings. The decomposition into four intervention dimensions, the constrained-allocation formalization of the error budget, and the definitions of $S_k$ and $MV$ are **conceptual framework and author's reading**: $S_k$ and $MV$ are decision statistics to be estimated via sensitivity experiments / ablation / small-scale real evaluation, **not** quantities analytically obtainable from the simulator; reading co-training as data-mixture and world model as model-source replacement is likewise an interpretation, not a theorem proven by controlled experiments.

---

*This article continues the two-part "data problem for embodied AI" series: the first part covered data sources and interfaces, the second covered the data scaling framework; here the camera pans to sim-to-real, reframing it from "a pile of transfer tricks" into a closed-loop allocation problem with empirical marginal utility — so it can reattach to the previous article's sequential data allocation thread.*
