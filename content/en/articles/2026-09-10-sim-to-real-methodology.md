---
title: 'Sim-to-Real Methodology for Embodied AI: Treating "From Simulation to Reality" as an Error-Budget Allocation'
slug: "2026-09-10-sim-to-real-methodology"
date: 2026-09-10
draft: false
categories: ["Embodied AI", "Training Methods"]
tags: ["Embodied AI", "Sim-to-Real", "Domain Randomization", "System Identification", "Differentiable Simulation", "Residual Physics", "World Model", "Domain Adaptation", "Robot Data"]
description: "Sim-to-real is usually presented as a single transfer trick, but it is essentially a constrained error-budget allocation. This post first reframes the reality gap as a policy-conditioned, multi-source mismatch, then sketches a conceptual framework that chains mismatch decomposition, intervention selection, real-data budget allocation, and evaluation utility; it dives into the mechanisms and failure boundaries of system identification, domain randomization, domain adaptation, and real-world fine-tuning, and disentangles differentiable simulation, residual physics, world models, and sim-and-real co-training."
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

> **When your simulation data falls far short of the real world along several evaluation-relevant directions, which of those gaps should you close by system-identifying, which by domain-randomizing, which by domain-adapting, and which should you simply pay real-robot budget to collect?**

That reads like engineering intuition, but it is actually a **constrained allocation problem**: given a limited engineering budget, you distribute it across several mutually independent levers — simulator fidelity, training-distribution diversity, representation alignment, and real-robot interaction — spending each unit wherever it buys down the most of the final performance gap. What this post tries to do is push that intuition from a nice metaphor toward a framework you can examine line by line. In real projects what actually blocks people is rarely "I didn't know these methods existed," but "I don't know whether this method does anything for *my kind* of gap, or how much real-robot budget it will cost."

## The Reality Gap: Not a Scalar, but a Policy-Conditioned Mismatch

Sim-to-real is usually narrated as "train a policy, then transfer it from simulation to reality." A more rigorous starting point is **two distributions**: the simulator induces $p_{\mathrm{sim}}(\tau)$, the real world induces $p_{\mathrm{real}}(\tau)$, and in general

$$p_{\mathrm{sim}}(\tau) \;\neq\; p_{\mathrm{real}}(\tau)$$

But what we actually care about is not this distribution difference per se — it is its **consequence** on a given task: the performance gap of the same policy $\pi_\theta$ across the two:

$$\Delta J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)$$

One distinction has to be drawn right here, or everything downstream will slide: **distribution mismatch is not performance gap.** $p_{\mathrm{sim}} \neq p_{\mathrm{real}}$ does not automatically imply a large $\Delta J$ — because different policies have wildly different sensitivity to the distributional difference. A policy that relies only on coarse geometry may be almost unchanged if you re-model the friction coefficient; whereas for a fine-assembly policy that depends on high-frequency force feedback, the very same distributional difference can be fatal.

So the more accurate statement is: $\Delta J(\pi)$ is a **task- and policy-dependent observable consequence**, not the full definition of the reality gap. It depends on at least four things:

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ p_{\mathrm{sim}},\ p_{\mathrm{real}},\ \mathcal{E}\big)$$

where $\mathcal{E}$ is the evaluation setup. **The same simulator can have a small gap for a position-control policy and a huge one for a force-sensitive manipulation policy.** This is why "our simulator is very accurate" is never a meaningful claim on its own — the reality gap is not an intrinsic property of the simulator but a property of the tuple $(\pi, p_{\mathrm{sim}}, p_{\mathrm{real}}, \mathcal{E})$. Recognizing this is the precondition for every "allocation" discussion below: if the gap depends on the policy and the evaluation, then "which layer of the gap to close" can only be answered relative to a target scenario.

### Where exactly is the gap: a five-layer mismatch decomposition

Since the gap is multi-source, the first step is to split it. A coarse but engineering-useful decomposition has five layers (note: these five layers **differ in nature, in how parameterizable they are, and — crucially — in how much it costs to remediate each**):

```
Reality mismatch
├── Dynamics / contact        friction, contact, deformables, compliance
├── Observation / estimation  sensor physics, calibration, noise, occlusion, latency, state estimation
├── Actuation / timing        motor dynamics, control frequency, actuator delay, comms jitter
├── Initial-state / env.      reset distribution, scene layout, long-tail, initial conditions
└── Objective / constraint    reward definition, safety constraints, success criteria
```

Here the **observation / state-estimation** layer deserves to stand on its own rather than be buried in "perception gap." The reason is concrete: what a robot actually executes is

$$a_t = \pi(o_t), \qquad o_t = h(x_t) + \epsilon$$

and in the real world, camera calibration error, depth bias, occlusion, proprioception drift, force-sensor bias, state-estimator latency and imperfect synchronization are **not simply "the image looks different"** — they make the **state estimate the policy actually sees inconsistent with the state the simulator assumes is available**. In manipulation and locomotion this "state-estimation gap" often hurts performance more than the appearance gap, so it merits its own layer:

```
Observation / Estimation
 ├─ sensor physics     the imaging / ranging physical process
 ├─ calibration        intrinsic/extrinsic, hand-eye calibration
 ├─ noise              exposure, quantization, random noise
 ├─ occlusion          occlusion and partial observability
 ├─ latency            sensing and synchronization delay
 └─ state estimation   error in inferring s from o
```

The **task / initial-state** layer also has to be split in two, because they are really two problems:

- **Environment / initial-state shift:** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$ — the reset distribution or scene layout does not match; this is a distribution problem inside simulation and belongs to the reality gap.
- **Objective / task shift:** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$ — simulation only requires grasp success, while reality also requires collision avoidance; simulation tolerates large penetration, while real hardware safety does not.

The latter is, strictly speaking, **no longer a reality gap but an objective mismatch**: no matter how faithfully the simulator models physics, if your reward / constraints are not the same object as the real target, then it is not "transfer failed," but "you were never evaluating the same task." Folding it into "sim-to-real failure" turns the discussion into "is this sim-to-real, or sim-to-task?" Everywhere below, when I talk about closing gaps I assume the objective is already aligned; objective mismatch requires separate handling via reward shaping / constraint modeling and is out of range for the four tool families.

## Making "Error-Budget Allocation" Something You Can Actually Write Down

The previous section split the gap into sources of different kinds. Now we can give the opening intuition a mathematical footing. First a disclaimer: **the decomposition below is a conceptual decomposition, not a strict theorem** — the layers couple and amplify each other non-linearly, so the "plus signs" only express the intuition that the total shortfall is jointly contributed by several sources, not a claim that they are strictly additive and independent. With that caveat, we can write:

$$\Delta J \;\lesssim\; \underbrace{\Delta_{\mathrm{model}}}_{\text{dynamics / contact modeling}} + \underbrace{\Delta_{\mathrm{obs}}}_{\text{observation \& estimation}} + \underbrace{\Delta_{\mathrm{ctrl}}}_{\text{actuation \& timing}} + \underbrace{\Delta_{\mathrm{dist}}}_{\text{initial-state / scene coverage}} + \underbrace{\Delta_{\mathrm{opt}}}_{\text{residual optimization \& prior bias}}$$

What actually makes the phrase "error budget" legitimate is mapping **each intervention to the term it primarily buys down, and to the cost it incurs**. Write the tool-selection step as a constrained optimization problem:

$$\min_{m}\quad \mathbb{E}\big[\Delta J(m)\big] \qquad \text{s.t.}\quad C_{\mathrm{sim}}(m) + C_{\mathrm{real}}(m) + C_{\mathrm{eng}}(m) \;\le\; B$$

Here $m$ is a (combined) choice from the tool set, $C_{\mathrm{sim}}$ is simulator-fidelity / compute cost, $C_{\mathrm{real}}$ is real-robot collection and wear cost, $C_{\mathrm{eng}}$ is engineering and maintenance cost, and $B$ is the total budget. With this, the four families of tools stop being mutually exclusive options competing over "which is stronger," and each maps to a compression term and a cost type:

| Intervention | Primary term compressed | Primary cost |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{sim}}$ + some $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | sample efficiency / $C_{\mathrm{sim}}$ |
| Residual physics | $\Delta_{\mathrm{model}}$ (residual part) | real interaction $C_{\mathrm{real}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$ | unlabeled real data |
| Real-world fine-tuning | residual $\Delta J$ (incl. $\Delta_{\mathrm{opt}}$) | $C_{\mathrm{real}}$ (wear / safety) |
| World model | changes the model source | $C_{\mathrm{real}}$ + $C_{\mathrm{sim}}$ |
| Sim-and-real co-training | changes $p_{\mathrm{train}}$ (mostly $\Delta_{\mathrm{dist}}$) | mixed-data cost |

With this table, the opening engineering intuition gets its rigorous version: **what "allocating an error budget" means is choosing, under constraint $B$, a set of interventions whose combined effect pushes down the terms whose dominant $\Delta$ is largest.** A friction-dominated peg insertion and a vision-dominated tabletop tidy-up have different dominant terms, hence different optimal $m$ — which is exactly the structural explanation for "why the recipe that worked for someone else doesn't work for me."

## Four Intervention Axes (Not Four Mutually Exclusive Methods)

With the framework in place, let's examine the tools one by one. But first, correct a classification trap: system identification, domain randomization, domain adaptation, and real-world fine-tuning are **not four categories sitting at the same level of abstraction** — SI is model calibration, DR is training-distribution manipulation, DA is representation alignment, fine-tuning is an optimization strategy. Lining them up as "four methods" wrongly implies you must pick one. The more accurate statement: they are **four mutually orthogonal intervention axes** that can be freely combined:

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

Precisely because they are orthogonal, "combination" is the natural thing — you can **swap means within one axis and push on several axes at once**, instead of agonizing over "which single method to use." Below I develop the four axes.

### A more accurate test: parameterizable / identifiable / coverable

Before expanding, let's replace the one intuition most likely to mislead. Many people memorize it as "systematic error goes to SI, random error goes to DR" — fine as a mnemonic, but it invites objections as a technical claim. What system identification actually does is estimate

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; D\big(p_{\mathrm{sim}}(\tau \mid \phi),\; p_{\mathrm{real}}(\tau)\big)$$

what it really addresses is **identifiable, parameterizable model mismatch** — not "anything systematic belongs to it" (actuator gain, latency, friction, and mass can themselves be stochastic processes rather than deterministic systematic bias). Likewise, what DR addresses is **uncertainty / model variation that can be represented by a training distribution**. So the more useful partition is not systematic vs random but a triple test:

> **Can this mismatch be expressed by a trustworthy parameterized model (parameterizable)? Can it be identified from a limited amount of real-robot data (identifiable)? If it cannot be identified, can it be covered by widening the training distribution (coverable)?**

Along these three questions, tool selection roughly becomes:

| Nature of the mismatch | More natural tool |
| --- | --- |
| Parameterizable + identifiable | System Identification |
| Parameterizable but hard to identify / high uncertainty | Domain Randomization |
| Not parameterizable but with residual structure | Residual learning |
| Observation / appearance mismatch | Domain Adaptation |
| Policy still has systematic residual in the target domain | Fine-tuning |

### Axis A — Model: system identification, differentiable simulation, and residual physics

This axis handles $\Delta_{\mathrm{model}}$, and internally there are actually three **orthogonal** questions worth separating once, because they're routinely collapsed into vague phrases like "differentiable simulation = more powerful SI":

$$f_{\mathrm{real}}(x,a) \;=\; \underbrace{f_{\mathrm{physics}}(x,a;\phi)}_{\text{parameterizable physics}} \;+\; \underbrace{r_\theta(x,a)}_{\text{residual}} \;+\; \epsilon$$

- **Differentiable simulation answers "how do we optimize the model"** — it provides the optimization interface $\partial f/\partial\phi$; DiffTaichi (Hu et al., 1910.00935) and Interactive Differentiable Simulation (Heiden et al., ICRA 2021, 1905.10706) make parameter estimation differentiable and thus gradient-based.
- **System identification answers "which parameters do we optimize"** — the $\phi$ above. Classical SI sweeps parameters and fits trajectories; differentiable simulation instead updates $\phi$ by backpropagation like network weights. The actual workflow is moreover often **real → identify → sim → train → real**, so a more accurate name is **real-to-sim-to-real**.
- **Residual physics answers "who explains the part the model didn't explain"** — instead of hard-calibrating $\phi$, let a network learn a residual $r_\theta$ to absorb the difference. It is especially useful in contact-rich, soft-body, and buoyancy-based legged settings — cases where the **functional form of the parameterized model itself may just be wrong**, in which case SI is calibrating a false premise (residual examples: soft robots, Michelis et al., 2402.01086; buoyancy-based legged robots, Chae et al., 2303.09597).

There is a point here that is critical and precisely the one most easily masked by the word "differentiable": **differentiability solves the optimization interface, not the model-class correctness.** Put differently — if the simulator's contact model does not represent some real phenomenon at all, then however precisely you take gradients with respect to that wrong model, you can only obtain "the optimal parameters *under the wrong model*." Differentiable simulation lets you estimate $\phi$ more accurately, but it will not write the functional form of $f_{\mathrm{physics}}$ correctly for you; the part it can't write, you either hand to a residual, or you abandon the very premise of "build a trustworthy sim first" (see the world-model route below). These three are orthogonal, and bundling them into a single "advanced differentiable simulation" narrative is exactly what hides the model-class problem that actually decides success or failure.

### Axis B — Data distribution: domain randomization and its family

This axis does not chase some "most accurate" $p_{\mathrm{real}}$; instead it makes the policy robust to a family of parameters $\{\phi\}$: randomize physics, vision, initial state, and delay during training, so that as long as the real system lands inside the family's support, the policy holds up. Tobin (1703.06907) used pure visual randomization to move sim-trained grasp detection onto a real robot; Peng (1710.06537) pushed randomization into dynamics; OpenAI's in-hand manipulation (Akkaya et al., 1808.00177) is almost a maximalist demonstration of DR — **relying not on precise calibration but on "the randomization range being wide enough" to absorb most of the difference.**

One common but imprecise phrasing has to be corrected. My earlier draft called DR "an implicit ensemble" — suggestive, but it will be objected to. DR trains a **single** shared policy $\pi_\theta$, with an objective roughly

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

whereas an ensemble in the usual sense is multiple models $\{\pi_1,\dots,\pi_K\}$. So the safer phrasing is: **it is best understood as population-level optimization over a family of environment models, not optimization against a single simulator.** Intuitively it resembles ensemble training (the policy is forced to do well across a whole family of dynamics), but strictly the result is still one shared policy, not an ensemble of policies.

"Adaptive / Automatic DR" is also not a single method but a family, and it's worth spreading out so readers don't think there's only one: curriculum over randomization (widening the range as training progresses), adversarial domain randomization (sampling the parameters that most break the current policy), automatic domain randomization (adapting and contracting the range with policy performance), posterior-based sampling (sampling from an identified posterior), and performance-driven range adaptation. Their mechanisms differ, but they share one aim: **avoid over-randomizing at the start.**

Failure boundaries: too wide a randomization range induces overly conservative / averaged behavior and drops sample efficiency sharply; DR only works if it **actually covers the real system**, and cannot save you when reality lies outside the family; and for non-parameterizable mismatches (complex contact, soft bodies, fluids) there is simply no "randomization axis" to randomize along — those must fall back to the residual on Axis A, or accept real-robot data.

### Axis C — Representation: domain adaptation and observation translation

This axis handles $\Delta_{\mathrm{obs}}$; it neither calibrates physics nor randomizes, but aligns sim and real at the **representation level**: a feature-level adapter, image translation (GAN / diffusion), or a randomized-to-canonical translation network such as RCAN (James et al., CVPR 2019, 1812.07252), which "translates" randomized sim images back into an approximately canonical clean image before feeding the downstream policy — neatly **stitching DR from Axis B onto this axis** to offset the performance loss of overly wide randomization. It handles the portion of the gap where "the physics is actually close, but it just looks nothing alike."

Boundaries: a translation network may erase task-relevant semantics along with the domain difference (the better the alignment, the more some fine signals get averaged away); unsupervised DA usually needs **unlabeled** data on the real side — but "unlabeled" is not "free"; real-robot collection is still expensive, and that bill belongs in $C_{\mathrm{real}}$.

### Axis D — Optimization / adaptation: real-world fine-tuning

This axis handles "the residual $\Delta J$ left after the first three axes have done their work" — pre-train at scale in simulation to learn structure, then hand off to real-robot data. But two regimes have to be separated here, because a single phrase like "fine-tune with RL or imitation" blurs them, and their cost structures are completely different:

- **Offline / imitation:** $D_{\mathrm{real}} \to \theta$; the main cost is **data collection** (one-off, offline, reusable).
- **Online RL:** $\pi_\theta \to a \to$ a real transition $\to \theta'$; the main cost is **interaction + safety + hardware wear + exploration** (every step consumes physical resources).

This distinction directly decides "is fine-tuning worth it": for real-robot learning, comparing methods cannot stop at the final success rate — you must also look at **how much real-robot interaction budget it takes to reach a target performance**. So a pragmatic metric that should enter the $C_{\mathrm{real}}$ above is

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{or}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

that is, "performance gain per unit of real-robot wear." The same +5 points, if one route needs 2 robot-hours and another 40, are simply not the same thing physically. As for failure boundaries: if the prior learned in sim is itself wrong in reality, fine-tuning can be dragged astray by it — or get worse the more you tune (false prior / catastrophic forgetting); and whether fine-tuning really saves more than "collecting from scratch on the robot" depends on simulator fidelity, the task, and the relative unit price of fine-tuning samples — there is no universal rule of thumb yet.

## Two Newer Routes That Loosen the "Two Given Distributions" Assumption

The four axes above share an implicit premise: **$p_{\mathrm{sim}}$ and $p_{\mathrm{real}}$ are two given distributions**, and what you do is calibrate, cover, align, or relay. Two more recent routes loosen that premise itself — they are not "a fifth or sixth transfer trick" but a reformulation of the whole problem, which is why I treat them separately.

### World model: not a sim-to-real technique, but an alternative model-based route

[Part 2 on data scaling](/en/articles/2026-09-09-robot-data-scaling/) discussed the relation between world models and data utility. Placed in the sim-to-real context, a common misreading has to be corrected first: **a world model does not inherently belong to sim-to-real.** The causal directions of classical sim-to-real and the world-model route differ:

```
Classical sim-to-real:   sim dynamics → train policy → deploy real
World-model route:       real interaction → learn dynamics → imagine → optimize policy
```

The former assumes "there is a trustworthy sim first, then transfer"; the latter **abandons the very premise of "build a trustworthy simulator first,"** instead learning a latent dynamics $p_\theta(z_{t+1}\mid z_t,a_t)$ directly from real interaction and running the policy's imagination and planning inside that learned model (Dreamer 1912.01603, TD-MPC2 2310.16828). So the more accurate positioning is: when the **artificial simulator's model bias is too large** — too large to be worth fixing the sim first — the world model is **a rewrite of the sim-to-real problem itself**, not a transfer technique under it.

DayDreamer (2206.14176) is often mis-cited as a "sim pre-train → real fine-tune" example, but its key contribution is precisely the opposite: **a world model learned directly from real-robot interaction, with policy improvement carried out in latent imagination**, relying almost entirely on no artificial sim. It is better used as a sample of "another route that replaces the simulator" than as a sample of sim-to-real transfer.

An honest boundary: "learn dynamics from real data" **is not the same as "abandon the simulator," and still less does it imply being inherently better than simulation.** It trades "hand-modeling cost" for "real-collection cost + model-capacity cost"; and in contact-rich, long-tail, high-sensor-noise settings, the learned model very often produces **confident but wrong imagination** in out-of-distribution regions it never saw, and the policy will confidently plan its way into that error. So for now this is just **yet another trade-off** between "hand-built sim" and "direct real-robot RL," not an endgame.

### Sim-and-real co-training: rewriting "transfer" as "mixed sampling"

The Sim-and-Real Co-Training of Maddukuri et al. (2503.24361) is a pragmatic route. **What the paper actually reports** is: mixing sim and real datasets within the same training run, and observing average performance gains across two robot platforms and multiple tasks. This step does no one-way sim→real transfer; a single recipe directly decides the ratio and schedule between the two sources.

But the following is **this post's interpretation, not a conclusion proven by the paper**: I tend to read this class of methods further as a **data-mixture problem** — once reframed that way, DR and real data are no longer substitutes but two sources feeding the same $T_R[p_{\mathrm{raw}}]$ sampling distribution (echoing Part 1's "$p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$ — the recipe is the transformation from raw data to the distribution the model actually sees"). This data-mixture view explains why it is orthogonal to the four axes above: co-training changes neither the sim nor the policy, only $p_{\mathrm{train}}$.

It is worth adding that the **mechanism** of this route is itself now under scrutiny: a 2026 follow-up (A Mechanistic Analysis of Sim-and-Real Co-Training, arXiv:2604.13645) proposes two mechanisms for why co-training works — structured representation alignment and importance reweighting. In other words, "mixing works" is turning from an empirical phenomenon into a research object with candidate explanations.

## Evaluation: How Do You Know You've Closed the Gap?

A common but dangerous practice is reporting performance only on a sim benchmark. Such a number **measures the consistency between the policy and *your own* simulator, not the consistency between it and the real world.** A more credible evaluation should at least do the following:

- report **zero-shot transfer** (no real fine-tuning at all) performance on the real system, plus the curve after **few-shot / N-shot**;
- test against a set of **held-out physical systems** (different calibration, different cameras, different contact surfaces), not just "that one deployment robot";
- explicitly state whether sim and real share the same **task, initial-state, and evaluation distribution** — otherwise the comparison is simply unfair (this is exactly why the objective mismatch above must be aligned first);
- do **failure attribution**: which layer's $\Delta$ dominates (model / obs / ctrl / dist / opt)? Remediation cost differs by orders of magnitude across layers, and a wrong attribution sends the budget to the wrong place;
- record **$\eta_{\mathrm{real}}$**: for routes that use real data, report "performance per unit of real-robot wear," or else "pure sim" vs "sim + fine-tune" simply cannot be compared fairly.

There is one more metric especially aligned with this post's thesis and worth highlighting on its own. If sim's role is to be a "proxy for the real world," then matching the *numerical* values of $J_{\mathrm{sim}}$ and $J_{\mathrm{real}}$ is not enough — you must also ask: **can the simulator correctly predict which policy is better?** Consider:

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

On sim it looks like $A > B > C$; on the real robot it is $B > C > A$. Here the simulator does not merely have a calibration error — it has **entirely lost model-selection utility**: you would use it to pick out the worst policy. So a metric more fundamental than "numerical agreement" is rank correlation:

$$\rho_{\mathrm{rank}} \;=\; \mathrm{Spearman}\big(J_{\mathrm{sim}}(\pi_i),\ J_{\mathrm{real}}(\pi_i)\big)$$

because a simulator's value **does not require $J_{\mathrm{sim}} \approx J_{\mathrm{real}}$, but at least requires it to correctly rank candidate policies.** A sim that underestimates every score by 20 points but ranks them perfectly consistently is still a good engineering tool; a sim whose numbers are close but whose rankings flip often may be more dangerous than having none.

## Combination and Decision: From a Decision Tree to a Decision Matrix

Collapsing the framework into a "which one, when" lookup. Real projects usually have several conditions holding at once, so a **gap × parameterizability × real-data-budget** matrix is more useful than a one-dimensional decision tree (the "Real data" column maps directly onto the constraint $B$):

| Gap | Parameterizable / identifiable? | Real data | Recommended |
| --- | --- | ---: | --- |
| dynamics bias | high | little | SI |
| dynamics uncertainty | medium | little | DR |
| dynamics residual | low (but structured) | some | Residual learning |
| visual appearance | high | none / little | DA / DR |
| actuator latency | high | little | SI + DR |
| unknown long-tail | low | some | real data / co-training |
| model unknown | low | a lot | world model (change model source) |
| mixed | mixed | mixed | sim-and-real co-training as a backstop |

Successful systems in practice almost always rely on a **combination of the four axes**, not a single trick. A very typical pipeline is:

$$\text{SI}\ \rightarrow\ \text{DR}\ \rightarrow\ \text{DA}\ \rightarrow\ \text{Co-training / fine-tune}$$

Each step buys down a **different $\Delta$ term** — SI calibrates a sim that is "80% right," DR widens the family along directions that are "hard to pin down but enumerable," DA handles the visual domain difference, and finally a little real co-train / fine-tune closes the residual. Precisely because they act on different axes, combining them is not patchwork but "each paying down a different slice of the budget."

## What Does This Mean? Pulling Sim-to-Real Back to the Data-and-Cost View

The core sentence of [Part 2 on data scaling](/en/articles/2026-09-09-robot-data-scaling/) is evaluation-aware distribution allocation: under a limited budget, spend every unit wherever marginal data value is highest. Applying that principle back to sim-to-real yields a natural corollary — **the utility of simulation data is never an intrinsic property of the simulator, but a property relative to the real evaluation distribution:**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

This explains a very common frustration: why "piling on more sim data" so often looks pointless. But be careful not to state it in absolute terms — more sim data sometimes does bring wider coverage, more diverse objects, and higher rare-event frequency. The more accurate claim is **conditional**:

> **When the dominant bottleneck happens to be the support / fidelity mismatch between the simulator and the real evaluation distribution, the marginal value of adding more same-distribution simulation samples falls off quickly — at that point increasing $N$ mainly raises sampling density within the existing simulation distribution, and cannot automatically create missing support or correct model bias.**

Put another way, it improves density, while what is really missing is support and the fidelity of $\Delta_{\mathrm{model}}$. So rather than asking "how good is my sim," the better question is the one from the top: **"Along which evaluation-relevant regions is my sim actually close to reality, and along which is it far off? For the far-off directions, should I calibrate with SI, cover with DR, align with DA, or just pay real-robot budget to collect data?"**

Walking that line to its end, sim-to-real becomes, from a "does transfer succeed or not" switch, a coherent loop:

$$\boxed{\text{Mismatch decomposition} \rightarrow \text{intervention selection} \rightarrow \text{real-data budget allocation} \rightarrow \text{evaluation utility}}$$

To close in one sentence: **sim-to-real is not a transfer technique but a problem of doing constrained allocation among model fidelity, training diversity, representation alignment, real interaction, and engineering cost.** This is the same thing as Part 2's claim that "robot data scaling is a sequential data allocation problem" — only this time the allocation happens between simulation and reality.

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
- A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies — 2026 follow-up, arXiv:2604.13645 (author and publication metadata per the arXiv page)
- MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning — Li et al., 2021, arXiv:2109.12674 (its experiments observe that simply adding heterogeneous simulation data does not close the gap to real data, while adding real cases improves real-test performance — a side datum for "more synthetic data ≠ automatically solving real-world generalization")

It should be noted that sim-to-real currently has no accepted "which method is stronger" cross-task quantitative comparison — across different tasks, hardware, and fidelity ceilings, conclusions may reverse entirely. The works above mainly provide samples of the form "for this kind of gap, this method is viable," rather than an ordering that extrapolates across scenarios. In this post, the orthogonal decomposition into four intervention axes, the constrained-allocation formalization of the error budget, and the reading of co-training as a data-mixture view are all **a conceptual framework and the author's interpretation, not conclusions proven by controlled experiments.**

---

*This post is the sequel to the two-part "Data Problem" series: Part 1 covered data sources and interfaces, Part 2 covered the data-scaling framework; this one pulls the camera onto sim-to-real, deliberately reframing it from "a pile of transfer tricks" into a constrained error-budget allocation problem — so that it can be reconnected to Part 2's main thread of sequential data allocation.*
