---
title: 'Sim-to-Real Methodology for Embodied AI: Treating "From Simulation to Reality" as an Error-Budget Allocation'
slug: "2026-09-10-sim-to-real-methodology"
date: 2026-09-10
draft: false
categories: ["Embodied AI", "Training Methods"]
tags: ["Embodied AI", "Sim-to-Real", "Domain Randomization", "System Identification", "Differentiable Simulation", "Residual Physics", "World Model", "Domain Adaptation", "Robot Data"]
description: "Sim-to-real is usually presented as a single transfer trick, but it is really a closed-loop resource-allocation problem. This post reframes the reality gap as a policy-conditioned, multi-source mismatch, uses weighted sensitivities and marginal utility to turn 'error-budget allocation' into a solvable framework, walks through the four orthogonal intervention axes (system identification, domain randomization, domain adaptation, real-world fine-tuning) and their failure boundaries, disentangles world models, residual physics, and sim-and-real co-training, and finally answers a question that is usually dodged: when is the optimal move to not do sim-to-real at all?"
toc: true
related_articles:
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
  - 2026-08-25-dreamer-explained
---

> This is the sequel to [Part 1 of the data problem](/en/articles/2026-09-08-data-and-training-recipes/) and [Part 2 on data scaling](/en/articles/2026-09-09-robot-data-scaling/). In Part 1 I sketched sim-to-real as four families of tools in a single diagram, but that was only a taxonomy. The question this post actually wants to answer can be compressed into one sentence —

> **When your simulation data falls far short of the real world along several evaluation-relevant directions, where should the next unit of budget — engineering time, compute, or robot-hours — be spent: calibrating the simulator, widening the training distribution, aligning representations, or simply collecting real data?**

This reads like engineering intuition, but it is really a closed-loop resource-allocation problem: given several budgets that are not interconvertible, you keep asking "where does the next dollar buy the most real-world performance gain?" What this post tries to do is push that intuition from a nice metaphor all the way to a framework with marginal utility. In real projects what actually blocks people is rarely "I didn't know these methods existed," but "I don't know whether this method does anything for *my kind* of gap, or which kind of budget it will consume."

## The Reality Gap: Not a Scalar, but a Policy-Conditioned Mismatch

Sim-to-real is usually narrated as "train a policy, then transfer it from simulation to reality." A more rigorous starting point is **two distributions**: the simulator induces $p_{\mathrm{sim}}(\tau)$, the real world induces $p_{\mathrm{real}}(\tau)$, and in general

$$p_{\mathrm{sim}}(\tau) \;\neq\; p_{\mathrm{real}}(\tau)$$

But what we actually care about is not this distribution difference per se — it is its **consequence** on a given task: the performance gap of the same policy $\pi_\theta$ across the two:

$$\Delta J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)$$

Two things must be kept apart here: **distribution mismatch is not performance gap.** $p_{\mathrm{sim}} \neq p_{\mathrm{real}}$ does not automatically imply a large $\Delta J$, because different policies have wildly different sensitivity to the distributional difference. A policy relying only on coarse geometry may be almost unchanged if you re-model the friction coefficient; a fine-assembly policy that depends on high-frequency force feedback can be killed by the very same difference.

So $\Delta J(\pi)$ is a **task- and policy-dependent observable consequence** that depends on at least four things:

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ p_{\mathrm{sim}},\ p_{\mathrm{real}},\ \mathcal{E}\big)$$

where $\mathcal{E}$ is the set of evaluation assumptions: observation / action interface, initial-state distribution, horizon, reward and constraints. **The same simulator can have a small gap for a position-control policy and a huge one for a force-sensitive manipulation policy.** So "our simulator is very accurate" is never a meaningful claim on its own — the reality gap is not an intrinsic property of the simulator but of this four-tuple.

One easily-conflated point to clear up: $\mathcal{E}$ and $p_{\mathrm{sim}},p_{\mathrm{real}}$ have different roles. $\mathcal{E}$ **specifies the evaluation protocol** (under what initial states, horizon, and reward you score), whereas $p_{\mathrm{sim}},p_{\mathrm{real}}$ are the **data-generating mechanisms** (how dynamics, observation, and actuation evolve). The sim-vs-real difference in the observation model belongs to the latter; the convention "we evaluate both from the same set of initial states" belongs to the former — they should not be mixed.

### Where exactly is the gap: a five-layer mismatch decomposition

The first step is to split the multi-source gap. A coarse but engineering-useful decomposition has five layers, which **differ in nature, in how parameterizable they are, and — crucially — in how much it costs to remediate each**:

```
Reality mismatch
├── Dynamics / contact        friction, contact, deformables, compliance
├── Observation / estimation  sensor physics, calibration, noise, occlusion, latency, state estimation
├── Actuation / timing        motor dynamics, control frequency, actuator delay, comms jitter
├── Initial-state / env.      reset distribution, scene layout, long-tail, initial conditions
└── Objective / constraint    reward definition, safety constraints, success criteria
```

**Observation / state-estimation** deserves to stand on its own rather than be buried in "perception gap." The reason is concrete: what a robot actually executes is

$$a_t = \pi(o_t), \qquad o_t = h(x_t) + \epsilon$$

and in the real world, camera calibration error, depth bias, occlusion, proprioception drift, force-sensor bias, state-estimator latency and imperfect synchronization are **not simply "the image looks different"** — they make the **state estimate the policy actually sees inconsistent with the state the simulator assumes is available**. In manipulation and locomotion this "state-estimation gap" often hurts performance more than the appearance gap:

```
Observation / Estimation
 ├─ sensor physics     the imaging / ranging physical process
 ├─ calibration        intrinsic/extrinsic, hand-eye calibration
 ├─ noise              exposure, quantization, random noise
 ├─ occlusion          occlusion and partial observability
 ├─ latency            sensing and synchronization delay
 └─ state estimation   error in inferring s from o
```

The **task / initial-state** layer has to be split in two, because they are really two problems:

- **Environment / initial-state shift:** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$ — the reset distribution or scene layout does not match; this is a distribution problem inside simulation and belongs to the reality gap.
- **Objective / task shift:** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$ — simulation only requires grasp success, while reality also requires collision avoidance; simulation tolerates large penetration, while real hardware safety does not.

The latter is, strictly speaking, **no longer a reality gap but an objective mismatch**: no matter how faithfully the simulator models physics, if the reward / constraints are not the same object as the real target, it is not "transfer failed" but "you were never evaluating the same task." Everywhere below, when I talk about closing gaps I assume the objective is already aligned; objective mismatch requires separate handling via reward shaping / constraint modeling and is out of range for these tools.

## Making "Error-Budget Allocation" Something You Can Solve

Having split the gap into sources of different kinds, we can give the opening intuition a mathematical footing. The treatment here is **conceptual, not a strict theorem**: these error terms interact strongly — for example, the simulator assumes exact proprioception while reality has latency; neither latency alone nor dynamics mismatch alone is fatal, but the two combined can directly destabilize the controller. So the safer move is to first admit an unknown coupling function $F$:

$$\Delta J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}},\ \Delta_{\mathrm{opt}}\big)$$

Only near a working point, for engineering attribution, do we locally approximate it as a weighted sum — and the weights $w_k$ are precisely **task/policy-dependent** sensitivities, which echoes the previous section's point that the gap is policy-conditioned:

$$\Delta J \;\approx\; \sum_{k} w_k\, \Delta_k$$

With this approximation we can define a useful quantity: **the sensitivity of real performance to the $k$-th mismatch**

$$S_k \;=\; \left|\frac{\partial J_{\mathrm{real}}}{\partial \Delta_k}\right|$$

It answers "is this kind of gap worth caring about at all." Each $\Delta_k$ can in turn be pushed down by some intervention at some cost; write the efficiency of intervention $k$ as $\partial \Delta_k / \partial C_k$. Then "which block to fix first" gets a clean priority:

$$\text{priority}_k \;\propto\; S_k \cdot \frac{\partial \Delta_k}{\partial C_k}$$

In plain language: **prioritize the gap that is both most sensitive for the task and cheapest to push down.**

### Real "allocation": distribute the budget across axes, not pick one method

So far this is still "choosing a method." For "error-budget allocation" to earn its name, the budget must be distributed **continuously** across the intervention axes. Split the total budget into a vector $b=(b_1,\dots,b_K)$, where $b_k$ is the amount spent on intervention $k$ — $b_{\mathrm{SI}}=2\text{h}$, $b_{\mathrm{DR}}=10^6$ sim steps, $b_{\mathrm{real}}=4\text{h}$ real — rather than a 0/1 "use SI or not" choice. The objective is to maximize real performance:

$$\max_{b}\quad J_{\mathrm{real}}\big(\pi_b\big)$$

The key is that budgets in a robotics project are **not the same currency**. You may have nearly unlimited GPU but very little robot time; or you may have machine time but almost no engineering manpower. So the correct form is a **multi-budget constraint**, not a scalar $B$ with everything folded into it:

$$\begin{aligned}
C_{\mathrm{real}}(b) &\le B_{\mathrm{real}}\\
C_{\mathrm{compute}}(b) &\le B_{\mathrm{compute}}\\
C_{\mathrm{eng}}(b) &\le B_{\mathrm{eng}}
\end{aligned}$$

Once the budget is a vector, **marginal value** emerges naturally:

$$MV_k \;=\; \frac{\partial J_{\mathrm{real}}}{\partial C_k}$$

This is the real core sentence of the whole post: **the next unit of budget should go to whichever intervention axis currently has the highest marginal real-world utility.** And $MV_k$ usually diminishes — which explains a very common phenomenon: **"spending 2 hours on SI is worth it; spending another 20 is not necessarily."** The few parameters that are easiest to identify and matter most were already calibrated in the first 2 hours, so the marginal return falls off fast; those same 18 hours spent on DR or real collection may pay back more.

Mapping each intervention to the term it primarily buys down, and to which budget it primarily consumes, gives this table — note the cost column is now split along the **budget vector**, because lumping everything into "sim cost" hides the truth (most of system identification's cost is real excitation experiments + parameter estimation + instrumentation + simulator engineering + optimization compute, not some "simulator fidelity cost"):

| Intervention | Primary term compressed | Primary budget |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{eng}}$ + $C_{\mathrm{compute}}$ + some $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | $C_{\mathrm{compute}}$ (sample efficiency) |
| Residual physics | $\Delta_{\mathrm{model}}$ (residual part) | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$ | $C_{\mathrm{real}}$ (unlabeled data) + $C_{\mathrm{compute}}$ |
| Real-world fine-tuning | residual $\Delta J$ (incl. $\Delta_{\mathrm{opt}}$) | $C_{\mathrm{real}}$ (wear / safety) |
| World model | changes the model source | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Sim-and-real co-training | changes $p_{\mathrm{train}}$ (mostly $\Delta_{\mathrm{dist}}$) | mixed data ($C_{\mathrm{real}}+C_{\mathrm{compute}}$) |

With $MV_k$, the whole post stops being "which of four methods is best" and becomes a loop: **first localize which layer's $\Delta_k$ dominates, then use $S_k$ to judge how sensitive it is for the current task, then invest the next unit of budget along the axis with the highest marginal utility, and finally measure on real evaluation how much $J_{\mathrm{real}}$ that budget actually bought, to decide the next one.** This connects straight back to Part 2's evaluation-aware distribution allocation — only here the thing being allocated is engineering budget between simulation and reality.

## Four Intervention Axes (Not Four Mutually Exclusive Methods)

With the framework in place, let's examine the tools one by one. First a structural point: system identification, domain randomization, domain adaptation, and real-world fine-tuning are **not categories at the same level of abstraction** — SI is model calibration, DR is training-distribution manipulation, DA is representation alignment, fine-tuning is an optimization strategy. Lining them up as "four methods" wrongly implies you must pick one. They are in fact **four mutually orthogonal intervention axes** that can be freely combined:

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

Precisely because they are orthogonal, "combination" is the natural thing — you can swap means within one axis and push on several axes at once.

**The test for choosing a tool is not "systematic error goes to SI, random error goes to DR."** That mnemonic is fine, but what SI actually does is estimate

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; D\big(p_{\mathrm{sim}}(\tau \mid \phi),\; p_{\mathrm{real}}(\tau)\big)$$

what it addresses is **identifiable, parameterizable model mismatch** — not "anything systematic belongs to it" (actuator gain, latency, friction, and mass can themselves be stochastic processes rather than deterministic systematic bias). Likewise, what DR addresses is **uncertainty that can be represented by a training distribution**. So the more useful partition is a triple test:

> **Can this mismatch be expressed by a trustworthy parameterized model (parameterizable)? Can it be identified from a limited amount of real-robot data (identifiable)? If it cannot be identified, can it be covered by widening the training distribution (coverable)?**

| Nature of the mismatch | More natural tool |
| --- | --- |
| Parameterizable + identifiable | System Identification |
| Parameterizable but hard to identify / high uncertainty | Domain Randomization |
| Not parameterizable but with residual structure | Residual learning |
| Observation / appearance mismatch | Domain Adaptation |
| Policy still has systematic residual in the target domain | Fine-tuning |

### Axis A — Model: system identification, differentiable simulation, and residual physics

This axis handles $\Delta_{\mathrm{model}}$, and internally there are three **orthogonal** questions, routinely collapsed into vague phrases like "differentiable simulation = more powerful SI":

$$f_{\mathrm{real}}(x,a) \;=\; \underbrace{f_{\mathrm{physics}}(x,a;\phi)}_{\text{parameterizable physics}} \;+\; \underbrace{r_\theta(x,a)}_{\text{residual}} \;+\; \epsilon$$

- **Differentiable simulation answers "how do we optimize the model"** — it provides the optimization interface $\partial f/\partial\phi$; DiffTaichi (Hu et al., 1910.00935) and Interactive Differentiable Simulation (Heiden et al., ICRA 2021, 1905.10706) make parameter estimation gradient-based.
- **System identification answers "which parameters do we optimize"** — the $\phi$ above. Classical SI sweeps parameters and fits trajectories; differentiable simulation instead updates $\phi$ by backpropagation like network weights. The actual workflow is moreover often **real → identify → sim → train → real**, so a more accurate name is **real-to-sim-to-real**.
- **Residual physics answers "who explains the part the model didn't explain"** — instead of hard-calibrating $\phi$, let a network learn a residual $r_\theta$ to absorb the difference.

There is a decisive point here, precisely the one most easily masked by the word "differentiable": **differentiability solves the optimization interface, not the model-class correctness.** If the simulator's contact model does not represent some real phenomenon at all, then however precisely you take gradients with respect to that wrong model, you can only obtain "the optimal parameters *under the wrong model*." Differentiable simulation lets you estimate $\phi$ more accurately, but will not write the functional form of $f_{\mathrm{physics}}$ correctly for you; the part it can't write, you hand to a residual, or you abandon the premise of "build a trustworthy sim first" (see world model). The three are orthogonal, and bundling them into a single "advanced differentiable simulation" narrative is exactly what hides the model-class problem that actually decides success or failure.

SI has two further, subtler but very real traps. **First, $p_{\mathrm{real}}(\tau)$ is almost never directly accessible in reality** — we only have finitely many real trajectories $\{\tau_i^{\mathrm{real}}\}_{i=1}^N$, so that $\arg\min_\phi D(\cdot)$ above actually runs on an empirical estimate: $\hat\phi=\arg\min_\phi \sum_i \ell\big(\tau_i^{\mathrm{sim}}(\phi),\tau_i^{\mathrm{real}}\big)$, where $p_{\mathrm{real}}$ is approximated by the empirical distribution of the finite real trajectories. **Second, the parameters existing $\neq$ the parameters being identifiable** — identifiability also depends on excitation and sensor observability: mass, damping, and stiffness can produce nearly identical observable trajectories under certain excitation regimes and cannot be estimated independently. Being able to write a parameter into the simulator never means it can be uniquely estimated from a finite amount of real-robot data.

Residual physics also deserves its boundary tightened: it is **not automatically applicable "because the functional form of the physics model is wrong."** Its sweet spot is "an existing physics model explains most of the structure, and the remaining error has a stable, learnable structure," i.e. at least on the target distribution $\|r_\theta\|\ll\|f_{\mathrm{physics}}\|$; if $f_{\mathrm{physics}}$ is entirely wrong, then $f_{\mathrm{physics}}+r_\theta$ may still fit in principle, but the residual network degrades into carrying the whole dynamics and loses the physics inductive bias — at which point you might as well learn a model directly. This route is most useful in settings like soft robots (Michelis et al., 2402.01086) and buoyancy-based legged robots (Chae et al., 2303.09597), where "the backbone physics still counts, but the local friction / contact / deformation has a stable residual."

### Axis B — Data distribution: domain randomization and its family

This axis does not chase some "most accurate" $p_{\mathrm{real}}$; instead it makes the policy robust to a family of parameters $\{\phi\}$: randomize physics, vision, initial state, and delay during training, so that as long as the real system lands inside the family's support, the policy holds up. Tobin (1703.06907) used pure visual randomization to move sim-trained grasp detection onto a real robot; Peng (1710.06537) pushed randomization into dynamics; OpenAI's in-hand manipulation (Akkaya et al., 1808.00177) is almost a maximalist demonstration of DR — **relying not on precise calibration but on "the randomization range being wide enough" to absorb most of the difference.**

A commonly mis-stated intuition: DR is not an "implicit ensemble." It trains a **single** shared policy $\pi_\theta$, with an objective roughly

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

whereas an ensemble in the usual sense is multiple models $\{\pi_1,\dots,\pi_K\}$. The accurate statement is: **DR is population-level optimization over a family of environment models, not optimization against a single simulator.** Intuitively it resembles ensemble training (the policy is forced to do well across a whole family of dynamics), but the result is one shared policy, not an ensemble of policies.

The condition for DR to work also needs to be stated precisely: merely "widening the range" is not enough — what is really required is that **the real parameter distribution be covered by the support of DR and actually trained in the high-density region**, loosely written $p_{\mathrm{real}}(\phi) \ll p_{\mathrm{DR}}(\phi)$. If $\phi_{\mathrm{real}}$ lies inside $\mathrm{support}(p_{\mathrm{DR}})$ but happens to fall in an extremely low-probability tail where the policy was almost never trained, it will still perform badly. So too narrow misses and too diffuse dilutes; the crux again returns to allocation — **whether the randomization distribution is aligned with the evaluation distribution and objective.** Overly wide or task-irrelevant randomization lowers sample efficiency and forces the policy to compromise across many conflicting dynamics, making it look overly conservative; but under some robust / adversarial settings, widening the uncertainty set can instead improve robustness — so "wider range means more conservative" is not a universal rule; alignment is what matters.

"Adaptive / Automatic DR" is not a single method either but a family worth spreading out: curriculum over randomization (widening the range as training progresses), adversarial domain randomization (sampling the parameters that most break the current policy), automatic domain randomization (adapting and contracting the range with performance), posterior-based sampling (sampling from an identified posterior), and performance-driven range adaptation. Their mechanisms differ; they share one aim — **avoid over-randomizing at the start.**

### Axis C — Representation: domain adaptation and observation translation

This axis handles $\Delta_{\mathrm{obs}}$; it neither calibrates physics nor randomizes, but aligns sim and real at the **representation level**: a feature-level adapter, image translation (GAN / diffusion), or a randomized-to-canonical translation network such as RCAN (James et al., CVPR 2019, 1812.07252), which "translates" randomized sim images back into an approximately canonical clean image before feeding the downstream policy — neatly **stitching DR from Axis B onto this axis** to offset the performance loss of overly wide randomization. It handles the portion of the gap where "the physics is actually close, but it just looks nothing alike."

But one failure mode is specific to DA and worth calling out: **for policy learning, domain invariance is not the goal; task-relevant invariance is.** Merely aligning the two features ($z_{\mathrm{sim}}\approx z_{\mathrm{real}}$) is not enough — the ideal is **to shrink the domain difference while preserving task information**: keep $I(z;y_{\mathrm{task}})$ high while making $D(z_{\mathrm{sim}},z_{\mathrm{real}})$ low. In other words, alignment is not "the stronger the better," but "align only the task-irrelevant variation." This is the same thing as the previous section's "overly wide DR erases task signal," seen from the representation side.

### Axis D — Optimization / adaptation: real-world fine-tuning

This axis handles "the residual $\Delta J$ left after the first three axes have done their work" — pre-train at scale in simulation to learn structure, then hand off to real-robot data. But two regimes have completely different cost structures, and a single phrase "fine-tune with RL or imitation" blurs them:

- **Offline / imitation:** $D_{\mathrm{real}} \to \theta$; the main cost is **data collection** (one-off, offline, reusable).
- **Online RL:** $\pi_\theta \to a \to$ a real transition $\to \theta'$; the main cost is **interaction + safety + hardware wear + exploration** (every step consumes physical resources).

This distinction decides "is fine-tuning worth it": for real-robot learning, comparing methods cannot stop at the final success rate — you must also look at **how much real-robot interaction budget it takes to reach a target performance**. So a pragmatic metric that should enter the $C_{\mathrm{real}}$ above is

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{or}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

The same +5 points, if one route needs 2 robot-hours and another 40, are simply not the same thing physically. The risks are not limited to "catastrophic forgetting / false prior"; more common is **distribution narrowing** — real fine-tuning data is usually much narrower than the sim distribution ($D_{\mathrm{sim}} \to D_{\mathrm{real}}^{\mathrm{narrow}}$), so after fine-tuning the policy may be better on the target deployment slice yet *less* robust overall — trading **generalization for specialization**. If the real data only covers one narrow slice, fine-tuning pulls a robust policy back into a deployment-specific one.

## Two Newer Routes That Loosen the "Two Given Distributions" Assumption

The four axes above share an implicit premise: **$p_{\mathrm{sim}}$ and $p_{\mathrm{real}}$ are two given distributions**, and what you do is calibrate, cover, align, or relay. The two routes below loosen that premise itself — they are not "a fifth or sixth transfer trick" but a reformulation of the whole problem.

### World model: not removing the simulator, but replacing its source

[Part 2 on data scaling](/en/articles/2026-09-09-robot-data-scaling/) discussed the relation between world models and data utility. Placed in the sim-to-real context, a positioning misreading has to be corrected: **a world model does not inherently belong to sim-to-real.** The causal directions of classical sim-to-real and the world-model route differ:

```
Classical sim-to-real:   sim dynamics → train policy → deploy real
World-model route:       real interaction → learn dynamics → imagine → optimize policy
```

A point to state precisely: a world model **does not remove the simulator** — it replaces the simulator's role from "a hand-specified physics model" to "a predictive model learned from interaction data":

$$f_{\mathrm{hand\text{-}designed}} \;\longrightarrow\; f_{\mathrm{learned}}$$

What genuinely changes is the **model source**. Dreamer (1912.01603) and TD-MPC2 (2310.16828) embody this route. So when the **artificial simulator's model bias is too large — too large to be worth fixing the sim first** — the world model offers **a rewrite of the sim-to-real problem itself**, not a transfer technique under it.

DayDreamer (2206.14176) is often mis-cited as a "sim pre-train → real fine-tune" example, but its key contribution is precisely the opposite: a world model learned directly from real-robot interaction, with policy improvement carried out in latent imagination, relying almost entirely on no hand-built sim. Still, let's be clear: **not depending on a hand-built physics simulator does not mean model-free.** World-model learning is itself full of assumptions — representation, model architecture, action space, reward, exploration, real-data quality; it merely moves the inductive bias from an explicit physics simulator into a learned world model.

An honest boundary: "learn dynamics from real data" **does not mean it is inherently better than simulation.** It trades "hand-modeling cost" for "real-collection cost + model-capacity cost"; and in contact-rich, long-tail, high-sensor-noise settings, the learned model very often produces **confident but wrong imagination** in out-of-distribution regions it never saw, and the policy will confidently plan into that error. So it is just **yet another trade-off** between "hand-built sim" and "direct real-robot RL," not an endgame.

### Sim-and-real co-training: rewriting "transfer" as a data mixture

The Sim-and-Real Co-Training of Maddukuri et al. (2503.24361) is a pragmatic route. **What the paper actually reports** is: mixing sim and real datasets within the same training run, and observing average performance gains across two robot platforms and multiple tasks; it does no one-way sim→real transfer, instead letting a single recipe decide the ratio and schedule between the two sources.

**This post's interpretation (not a conclusion proven by the paper)** is to read it further as a **data-mixture problem**: once reframed that way, DR and real data are no longer substitutes but two sources feeding the same $T_R[p_{\mathrm{raw}}]$ sampling distribution (echoing Part 1's "$p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$ — the recipe is the transformation from raw data to the distribution the model actually sees"). To state it precisely: co-training **does not require modifying the simulator itself, nor introducing a separate sim→real adapter**; it mainly changes the policy's optimization distribution by changing the training-data mixture — $p_{\mathrm{train}}=\lambda\, p_{\mathrm{sim}}+(1-\lambda)\, p_{\mathrm{real}}$ (of course the policy parameters $\theta$ still update). This is also why it is orthogonal to the four axes above: it changes neither the sim nor the tools, only the mixture.

The **mechanism** of this route is now itself under scrutiny: a 2026 follow-up (A Mechanistic Analysis of Sim-and-Real Co-Training, arXiv:2604.13645) proposes two mechanisms for why co-training works — structured representation alignment and importance reweighting. "Mixing works" is turning from an empirical phenomenon into a research object with candidate explanations.

## Evaluation: How Do You Know You've Closed the Gap?

A dangerous practice is reporting performance only on a sim benchmark — that measures the consistency between the policy and *your own* simulator, not between it and the real world. A more credible evaluation should at least do the following:

- report **zero-shot transfer** (no real fine-tuning at all) performance on the real system, plus the curve after **few-shot / N-shot**;
- test against a set of **held-out physical systems** (different calibration, different cameras, different contact surfaces), not just "that one deployment robot";
- explicitly state whether sim and real share the same **task, initial-state, and evaluation distribution** — otherwise the comparison is simply unfair (this is exactly why the objective mismatch must be aligned first);
- do **failure attribution**: which layer's $\Delta_k$ dominates? Their sensitivities $S_k$ and remediation costs differ by orders of magnitude, and a wrong attribution sends the budget to the wrong place;
- record **$\eta_{\mathrm{real}}$**: for routes that use real data, report "performance per unit of real-robot wear."

Following "sim is a proxy for the real world," there is a question more fundamental than numerical agreement: **can the simulator correctly predict which policy is better?** Consider:

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

On sim it looks like $A > B > C$; on the real robot it is $B > C > A$. Here the simulator does not merely have a calibration error — it has **lost model-selection utility**: you would use it to pick out the worst policy. So **when the simulator is used for policy / model selection**, a metric more direct than absolute numerical error is rank correlation:

$$\rho_{\mathrm{rank}} \;=\; \mathrm{Spearman}\big(J_{\mathrm{sim}}(\pi_i),\ J_{\mathrm{real}}(\pi_i)\big)$$

This is a **conditional metric**, specific to the "use sim to pick policies" use case — it is not the sole standard for judging a simulator. Sim has many uses that do not require ranking a whole policy family: representation pretraining, exploration, curriculum, safety filtering, controller initialization, rare-event generation, and so on. But for "screening / comparing an entire candidate policy family," a sim that underestimates every score by 20 points yet ranks them perfectly consistently is still a good tool, whereas a sim whose numbers are close but whose rankings flip often may be more dangerous than having none. This also means: when assessing simulator fidelity you should not fixate on a single policy but evaluate against the **candidate policy family**, i.e. write it as $U_{\mathrm{sim}}=U(D_{\mathrm{sim}}\mid \Pi_{\mathrm{candidate}},\,p_{\mathrm{eval}}^{\mathrm{real}})$.

## Combination, Decision — and a Question Usually Dodged

With a priority in hand, "which one, when" should not be a fixed pipeline but a lookup table. Real projects usually have several conditions holding at once, so a **gap × parameterizability × real-data-budget** matrix is more useful (the "Real data" column maps directly onto the budget vector $B$):

| Gap | Parameterizable / identifiable? | Real data | Recommended |
| --- | --- | ---: | --- |
| dynamics bias | high | little | SI |
| dynamics uncertainty | medium | little | DR |
| dynamics residual | low (but structured) | some | Residual learning |
| visual appearance | high | none / little | DA / DR |
| actuator latency | high | little | SI + DR |
| unknown long-tail, simulatable | low | little | targeted simulation / DR |
| unknown long-tail, sim untrustworthy | low | some | real data |
| model unknown | low | a lot | world model |
| mixed | mixed | mixed | sim-and-real co-training as a backstop |

A very common combination is **SI → DR → DA → co-training / fine-tune**: SI calibrates a sim that is "80% right," DR widens the family along directions that are "hard to pin down but enumerable," DA handles the visual domain difference, and a little real data closes the residual. **But the arrows here are not a fixed workflow, only an illustrative combination** — the actual order should be set by the current dominant gap and the marginal value of each intervention: if real data is already plentiful, doing SI first may not pay; if vision dominates, DA should come earlier; if only very little data is available for SI, doing a rough DR first to get a runnable policy and then returning to calibration is often more sensible.

Following this logic, we can answer a question the whole post has otherwise dodged but which the framework itself permits: **when is the optimal move to not do sim-to-real at all?**

- **When real data is already cheap enough that $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}$** — e.g. a mature collection rig, a task that safely allows repeated runs — training directly on real data may be less trouble than fixing the simulation.
- **When the simulator's model class itself is poor ($\Delta_{\mathrm{model}}$ dominant and hard to parameterize)** — soft bodies, fluids, complex contact — the cost of fixing the sim is so high that its marginal utility is tiny, and it is better to go to a world model or real-data learning.
- **When the deployment distribution is very fixed** — you simply do not need large-scale DR to cover a whole family — a little targeted real fine-tuning is often more cost-effective.

Being willing to admit "sometimes the optimal move is to not do sim-to-real" is exactly what an allocation framing should look like: **it does not take the side of "simulation"; it only takes the side of "the next unit of budget that buys the most real-world performance."**

## What Does This Mean? A Loop, Not a Switch

The core sentence of [Part 2 on data scaling](/en/articles/2026-09-09-robot-data-scaling/) is evaluation-aware distribution allocation: under a limited budget, spend every unit wherever marginal data value is highest. Applying that principle back to sim-to-real yields a natural corollary — **the utility of simulation data is never an intrinsic property of the simulator, but a property relative to the real evaluation distribution:**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

This explains a common frustration: why "piling on more sim data" is sometimes useless. But state it precisely — more sim data can indeed bring wider coverage, more diverse objects, and higher rare-event frequency. So the claim is **conditional**:

> **When the dominant bottleneck happens to be the support / fidelity mismatch between the simulator and the real evaluation distribution, the marginal value of adding more same-distribution simulation samples falls off quickly; at that point increasing $N$ mainly raises sampling density within the existing simulation distribution, and cannot automatically create missing support or correct model bias.**

It improves density, while what is really missing is support and the fidelity of $\Delta_{\mathrm{model}}$. So rather than asking "how good is my sim," ask the opening question: **"Along which evaluation-relevant directions is my sim actually close to reality, and along which is it far off? For the far-off directions, how high is the sensitivity $S_k$, and which budget pushes them down most cheaply?"**

Walking that line to its end, sim-to-real stops being a "does transfer succeed or not" switch and becomes a closed loop:

$$\boxed{\text{mismatch} \rightarrow \text{sensitivity} \rightarrow \text{intervention} \rightarrow \text{marginal utility} \rightarrow \text{budget allocation} \rightarrow \text{real evaluation}}$$

To close in one sentence: **sim-to-real is not a transfer technique but a closed-loop problem of constrained allocation among model fidelity, training diversity, representation alignment, real interaction, and engineering cost.** This is the same thing as Part 2's claim that "robot data scaling is a sequential data allocation problem" — only this time the allocation happens between simulation and reality.

---

## References

The main works referenced in the text are listed below (all searchable via arXiv ID):

- Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World — Tobin et al., IROS 2017, arXiv:1703.06907
- Sim-to-Real Transfer of Robotic Control with Dynamics Randomization — Peng et al., ICRA 2018, arXiv:1710.06537
- Sim-to-Real: Learning Agile Locomotion For Quadruped Robots — Tan et al., RSS 2018, arXiv:1804.10332
- Learning Dexterous In-Hand Manipulation — Akkaya et al. (OpenAI), 2019, arXiv:1808.00177
- Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks — James et al., CVPR 2019, arXiv:1812.07252
- DiffTaichi: Differentiable Programming for Physical Simulation — Hu et al., ICLR 2020, arXiv:1910.00935
- Interactive Differentiable Simulation — Heiden et al., ICRA 2021, arXiv:1905.10706
- Residual Physics Learning and System Identification for Sim-to-real Transfer of Policies on Buoyancy Assisted Legged Robots — Chae et al., 2023, arXiv:2303.09597
- Sim-to-Real of Soft Robots with Learned Residual Physics — Michelis et al., 2024, arXiv:2402.01086
- Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) — Hafner et al., 2019, arXiv:1912.01603
- DayDreamer: World Models for Physical Robot Learning — Hafner et al., CoRL 2022, arXiv:2206.14176
- TD-MPC2: Scalable, Robust World Models for Continuous Control — Hansen et al., ICLR 2024, arXiv:2310.16828
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., 2025, arXiv:2503.24361
- A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies — 2026, arXiv:2604.13645 (author and publication metadata per the arXiv page)

It should be noted that sim-to-real currently has no accepted "which method is stronger" cross-task quantitative comparison — across different tasks, hardware, and fidelity ceilings, conclusions may reverse entirely. The works above mainly provide samples of the form "for this kind of gap, this method is viable," rather than an ordering that extrapolates across scenarios. In this post, the orthogonal decomposition into four intervention axes, the constrained-allocation formalization of the error budget, the definitions of sensitivity $S_k$ and marginal utility $MV_k$, and the reading of co-training as a data-mixture view are all **a conceptual framework and the author's interpretation, not conclusions proven by controlled experiments.**

---

*This post is the sequel to the two-part "Data Problem" series: Part 1 covered data sources and interfaces, Part 2 covered the data-scaling framework; this one pulls the camera onto sim-to-real, reframing it from "a pile of transfer tricks" into a closed-loop allocation problem with marginal utility — so that it reconnects to Part 2's main thread of sequential data allocation.*
