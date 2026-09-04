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

> This piece follows [Data Sources and Interfaces](/en/articles/2026-09-08-data-and-training-recipes/) and [Data Scaling for Robots](/en/articles/2026-09-09-robot-data-scaling/). The first split sim-to-real into four tool families — only a taxonomy. The question actually worth asking:

> **When sim data falls short of reality along several evaluation-relevant directions, which lever should the next unit of budget (engineering time, compute, or robot-hours) go to: calibrating the sim, widening the training distribution, aligning representations, or collecting real-robot data?**

This article does not propose a new algorithm; it proposes a decision framework for comparing and composing existing interventions. The piece converges along three levels: **Diagnosis** (where different, where it matters) · **Intervention** (Model × Data × Representation × Optimization) · **Allocation** (given state, budget, uncertainty — where the next unit of resource goes). **Three framework contributions + one downstream corollary**: (1) reality gap is policy-conditioned, not an intrinsic scalar; (2) descriptor / sensitivity is diagnosis, decision variable is intervention; (3) SI / DR / DA / FT are intervention lenses under sequential allocation; **corollary** — sim evaluation widens from fidelity to downstream utility. Constraints: $B_{\mathrm{real}}, B_{\mathrm{compute}}, B_{\mathrm{eng}}$; budget spent on intervention actions.

The spine is **Diagnosis → Experiment → Intervention → Allocation → Re-evaluation**. **"$\times$" means composable space, not mathematical orthogonality** — DR touches Model / Observation / Distribution, DA can occur at multiple layers.

In practice this is closed-loop resource allocation under several non-exchangeable budgets: not "does this method exist" but "does it help this gap, and which budget does it eat." **"Error budget" is a metaphor; the formal object is sequential resource allocation under model uncertainty** — constraints $B_{\mathrm{real}}, B_{\mathrm{compute}}, B_{\mathrm{eng}}$, **not** $\sum_k \Delta_k \le B_{\mathrm{error}}$; the budget is spent on **intervention actions** that progressively push down whichever mismatch currently pays best, not a fixed quota per error term.

## Reality Gap: not a scalar, but a policy-conditioned mismatch

Sim-to-real is usually narrated as "train in sim, transfer to reality." A more rigorous starting point is **two distributions**: the same $\pi$ induces $p_{\mathrm{sim}}^{\pi}(\tau)$ and $p_{\mathrm{real}}^{\pi}(\tau)$, generally not equal:

$$p_{\mathrm{sim}}^{\pi}(\tau) \;\neq\; p_{\mathrm{real}}^{\pi}(\tau)$$

**"The same $\pi$" carries a precondition** — sim and real must **share one policy interface**: obs schema (keys / shape / units / normalization), action schema (continuous vs discrete, torque / velocity / position, clamping), control frequency and action-hold semantics, timing / delay assumptions. Different interfaces → $\pi$ is not the same function, $\delta_J$ loses its definition. This assumption is not restated later.

The trajectory distribution is **policy-induced** — it changes with $\pi$, not an intrinsic property of the environment. What matters is not the distributional difference but its **manifest consequence on the task** — performance of the same $\pi$ in the two worlds:

Terminology is split three ways to keep the ontology clean: **(a) trajectory / distribution mismatch** $D(p_{\mathrm{sim}}^\pi,\ p_{\mathrm{real}}^\pi)$ (process-level); **(b) transfer delta**,

$$\boxed{\;\delta_J(\pi) \;=\; J_{\mathrm{real}}(\pi) \;-\; J_{\mathrm{sim}}(\pi)\;}$$

signed — if reality is actually better (sim more conservative, its noise bites harder), $\delta_J$ comes out positive and should not be called a gap; **(c) performance discrepancy**,

$$G_J(\pi) \;=\; \big|\,\delta_J(\pi)\,\big|$$

the absolute magnitude — sensitivity below uses this semantics, so we never tangle with the sign. **$J$ defaults to a "higher-is-better" utility; for a cost / minimization objective the sign flips, structure stays identical.** **$\delta_J$ is not the reality gap itself** — it is a downstream consequence under a specific $\pi$ + evaluation; the gap is closer to a four-tuple property (see next paragraph).

**Distribution mismatch $\neq$ performance gap**: $p_{\mathrm{sim}}^{\pi} \neq p_{\mathrm{real}}^{\pi}$ does not automatically imply a large $\delta_J$ — different policies have very different sensitivities to the same distributional gap. A coarse-geometry policy barely changes when you swap friction models; precision assembly leaning on high-frequency force feedback can be fatal under the same difference.

More fundamentally, what matters for a policy is not marginal $p_{\mathrm{sim}}(s)$ vs $p_{\mathrm{real}}(s)$ but **policy-conditioned occupancy** $d_{\mathrm{sim}}^{\pi}(s,a)$ vs $d_{\mathrm{real}}^{\pi}(s,a)$ — in contact-rich manipulation even $d^\pi(s,a,\text{contact mode})$. Causal chain: $\pi \rightarrow d^\pi \rightarrow \text{mismatch} \rightarrow J$, not merely $\pi \rightarrow p^\pi(\tau)$.

$\delta_J(\pi)$ is a **task- and policy-relevant observable consequence**. Rigorously, one must **separate mechanism from induced distribution**: let $M_{\mathrm{sim}}, M_{\mathrm{real}}$ be the transition / observation / actuation kernels; under $\pi$ they **induce** $p_{\mathrm{sim}}^{\pi}(\tau),\ p_{\mathrm{real}}^{\pi}(\tau)$. Cleaner writing:

$$\text{Reality gap} \;=\; \mathrm{Gap}\big(\pi,\ \mathcal{E}_{\mathrm{shared}};\ M_{\mathrm{sim}},\ M_{\mathrm{real}}\big)$$

Logic is **mechanism → trajectory distribution → performance**. $\mathcal{E}_{\mathrm{shared}}$ is the **shared evaluation protocol** — initial-state / horizon / reward / constraints must be **the same on both sides** ($\mathcal{E}_{\mathrm{sim}} = \mathcal{E}_{\mathrm{real}} = \mathcal{E}_{\mathrm{shared}}$); otherwise $\delta_J(\pi) = J_{\mathrm{real}}(\pi) - J_{\mathrm{sim}}(\pi)$ is not the "transfer consequence under one task specification." When $\mathcal{E}_{\mathrm{sim}} \neq \mathcal{E}_{\mathrm{real}}$, the observed performance difference already contains **task-specification mismatch** — this article does **not count that portion as operational reality gap**. Same $M_{\mathrm{sim}}$ can yield a small gap for a position-control policy and a huge gap for force-sensitive manipulation. **This article operationally treats the reality gap as the downstream discrepancy under the four-tuple $(\pi, \mathcal{E}_{\mathrm{shared}}, M_{\mathrm{sim}}, M_{\mathrm{real}})$, not an intrinsic simulator scalar** — an operational framing, not a claim of community-wide definition; every later occurrence of "reality gap" uses this notion.

### Where exactly the gap sits: reality mismatch and task-specification mismatch

First move: unpack the multi-source gap — **two big families of causes**, not all shovable under "reality":

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

The two families have different sources and cannot be summed: reality mismatch is "sim and real are not the same world," task-specification mismatch is "the objective you optimize and the objective you deploy are not the same task." **In ordinary projects the raw performance difference observed end-to-end may mix both. However, the operational $\delta_J$ used in this article already fixes $\mathcal{E}_{\mathrm{sim}} = \mathcal{E}_{\mathrm{real}} = \mathcal{E}_{\mathrm{shared}}$ and therefore only discusses the reality-mismatch portion of the downstream discrepancy** (the task-spec portion has been excluded by the previous definition). **Observation and state estimation deserve their own layer** — the robot executes $a_t = \pi(o_t),\ o_t = h(x_t) + \epsilon$; camera calibration error, depth bias, occlusion, proprioception drift, force-sensor bias, state-estimator latency are **not "the picture looks different" — they make the state the policy actually sees inconsistent with the state the simulator assumes is available**. In manipulation and locomotion this "state-estimation gap" often hurts more than the appearance gap.

**Stochasticity mismatch** (motor stochasticity, friction variability, sensor temporal correlation, communication jitter, unmodeled disturbance, repeated-reset variability) is not parameter mismatch — it concerns differences in the **higher-order statistics / stochastic-process structure** of dynamics, precisely what DR's $p_{\mathrm{DR}}(\xi)$ covers.

Timing mismatch ($\Delta t_{\mathrm{sim}} \neq \Delta t_{\mathrm{real}}$, action hold, sensor delay, policy inference latency, asynchronous observation) **can be amplified by closed-loop feedback** — not additive observation error, it can alter closed-loop stability itself.

**Initial-state / environment mismatch** $p_{\mathrm{train}}(s_0) \neq p_{\mathrm{eval}}(s_0)$; **objective / task shift** $R_{\mathrm{train}} \neq R_{\mathrm{eval}}$. Attribution deserves care: if sim and real **can both produce the same $s_0$** and training simply missed it, that is **ordinary train-test shift, not reality gap**; only when the sim-real reset / scene **implementation** differs is it environment mismatch. Objective shift is already objective mismatch: no matter how accurate the physics, if reward / constraints do not line up you are looking at "you never evaluated the same task," not "transfer failure." Below, objective is assumed aligned.

## Writing "error-budget allocation" as an estimable, iteratively optimizable decision framework

Sources unpacked, the intuition needs a mathematical landing. Error terms interact strongly — sim assumes perfect proprioception, reality has latency; neither alone is fatal, stacked they can destabilize a controller — so a safer move is to first admit an unknown coupling $F$:

$$\boxed{\;\delta_J \;=\; F\big(\Delta_{\mathrm{model}},\ \Delta_{\mathrm{obs}},\ \Delta_{\mathrm{ctrl}},\ \Delta_{\mathrm{dist}}\big)\;}$$

**Each $\Delta_k$ is a mismatch descriptor — scalar / vector / distribution / set-valued**; stochasticity / occupancy / model-class uncertainty do not fit one scalar "error magnitude," so the equation is schematic and presupposes no common metric. **$F$ is not an estimable predictive model** — it marks "some un-unfolded dependency"; the actual work is probing local response via sensitivity experiments and ablation, not fitting $F$. **The four $\Delta_k$ are diagnostic buckets, not four orthogonal latent variables** — actuator delay can masquerade as obs, estimator lag as ctrl, contact stochasticity as dynamics; the buckets are neither orthogonal nor uniquely identifiable. The hierarchy is **reality / task discrepancies → diagnostic buckets ($\Delta_k$ lives here) → observable evidence → intervention candidates**; $F(\cdot)$ is one schematic link, **not a latent factor model**. Throughout the article, $\mathcal{D}_t$ denotes **all evidence available to the allocator at step $t$** — calibration / ID measurements, simulation diagnostics, real paired evaluations, failure traces, safety observations, etc.; $D_{\mathrm{train}}$ and $D_{\mathrm{eval}}$ are kept as separate notation and are **not conflated with the belief state**.

**$\Delta_{\mathrm{opt}}$ (optimization / learning error) is out of the reality gap** — different levels: same fixed policy, sim dynamics and observations both accurate but RL never converged, $\delta_J$ small and the policy bad; split into **two diagnostic quantities**:

$$\underbrace{J_{\mathrm{real}}(\pi_{\mathrm{train}}) - J_{\mathrm{sim}}(\pi_{\mathrm{train}})}_{\text{transfer delta } \delta_J}\qquad \underbrace{J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})}_{\text{real-domain learning gap}}$$

**Cannot be unconditionally summed as "deployment loss"**: $\delta_J$ is signed, the two have different baselines, and they are error sources at different levels — diagnose separately. **$\pi^{*}_{\mathrm{real}}$ is typically unavailable** — the right term is **oracle-defined**; use $J_{\mathrm{real}}(\pi_{\mathrm{best\text{-}validated}}) - J_{\mathrm{real}}(\pi_{\mathrm{train}})$ as a proxy, where $\pi_{\mathrm{best\text{-}validated}}$ denotes the policy that performs best on an **independent audit / held-out evaluation slice** rather than the argmax over a single noisy evaluation (avoiding winner's curse); the same treatment applies to the selection-regret term below.

Only when doing engineering attribution near an operating point do we locally approximate $F$ as a weighted sum $\delta_J \approx \sum_k w_k \Delta_k$ — **this layer is a local attribution heuristic, not the article's core formula**. What actually drives decisions is the **intervention sensitivity** measured after picking an **intervention variable** $\xi_k$ for each mismatch class:

$$\hat S_k^{\mathrm{int}} \;\approx\; \frac{J_{\mathrm{real}}\big(\pi;\,\xi_k{+}\delta\big) \;-\; J_{\mathrm{real}}\big(\pi;\,\xi_k\big)}{\delta}$$

$\hat S_k^{\mathrm{int}}$ is a **local intervention response / sensitivity statistic**, not a true $\partial J / \partial \xi_k$; $S$ is kept only for intuition. $\xi_k$ is **not a natural gap coordinate but an intervention variable defined for a sensitivity experiment**, and many $\xi_k$ are **not directly controllable**. Sensitivity tiers three ways: **direct perturbation** (real-robot latency / friction / appearance), **proxy / surrogate** (via sim or bench, e.g. camera calibration error), **diagnostic ablation** (module / model / dataset swap). **Controlled experimental perturbation, not a derivative of an intrinsic quantity.** Because units across $\xi_k$ are incomparable — and because sensitivity additionally carries a **policy-conditional dependence** ($S_k^{\mathrm{int}}(\pi_1) \neq S_k^{\mathrm{int}}(\pi_2)$ under the same $\xi_k$) — the statistic is meant for **local comparison inside the current baseline policy / protocol**, and cross-policy comparability is not claimed. Therefore **sensitivity is a candidate-generation / prioritization statistic, not a final allocation input (value input)**; **performance / cost / continuation — $\Delta J(m\mid s_t)$, $\Delta C(m\mid s_t)$, $\mathrm{CVU}(m\mid s_t)$ — are the three core value inputs of the allocation score, while safety and budget define feasibility ($\mathcal{M}_t^{\mathrm{safe}}$, $\mathcal{M}_t^{\mathrm{budget}}$); allocation must return to these four things.** The hierarchy is **descriptor → sensitivity / uncertainty → candidate generation → $(\Delta J,\;\Delta C,\;\mathrm{CVU})$ + feasibility → $Q_{\lambda_t}$ → allocation**.

**Diagnosis $\neq$ attribution.** Perturbing $\Delta_{\mathrm{friction}}$ alone may be tiny, $\Delta_{\mathrm{latency}}$ alone may also be small, yet together $\Delta J(\Delta_f,\Delta_l) \gg \Delta J(\Delta_f,0) + \Delta J(0,\Delta_l)$ — synergy. **Sensitivity experiments identify locally influential intervention directions; they do not give an additive causal attribution of the deployment gap.** $\Delta_{\mathrm{model}}$ and $\Delta_{\mathrm{ctrl}}$ can even compensate unidentifiably (actuator gain wrong, policy offsets via its command distribution); both remain decision statistics from sensitivity / ablation, not strict decompositions.

### The real "allocation": spend on intervention actions, not pick one method off a shelf

For budget allocation to be literal, budget is split **continuously** across intervention axes: $b = (b_1, \dots, b_K)$, $b_k$ the amount on intervention $k$ — $b_{\mathrm{SI}} = 2\text{h}$, $b_{\mathrm{DR}} = 10^6$ sim steps, $b_{\mathrm{real}} = 4\text{h}$ real — not 0/1 like "use SI or not." **The deployment objective cannot be mean only**: mean success 90% + catastrophic 1% vs mean 88% + tail ≈ 0 is often **not the same deployment decision** on a real robot. This article uses **mean utility + tail / safety constraint** (rather than folding tail directly into a scalar cost); unless the project explicitly introduces CVaR / risk-penalized utility $\max \mathbb{E}[J] - \gamma\,\mathrm{TailRisk}(J)$, the formal shape stays "mean objective, safety-constrained":

$$\max_{b}\quad \mathbb{E}\big[J_{\mathrm{real}}(\pi_b)\big] \quad \text{s.t.}\quad \Pr\big[\text{unsafe} \mid \pi_b\big] \le \alpha$$

Robotics budget is **not one currency**: GPU may be near unlimited while real robot-hours are scarce, or you may have machine time but no engineering headcount — the correct writing is **multi-budget constraints**, not a scalar $B$:

$$\begin{aligned}
C_{\mathrm{real}}(b) &\le B_{\mathrm{real}}\\
C_{\mathrm{compute}}(b) &\le B_{\mathrm{compute}}\\
C_{\mathrm{eng}}(b) &\le B_{\mathrm{eng}}
\end{aligned}$$

**Safety does not belong in the same cost / budget layer** — it is a **chance constraint** $\Pr[\text{unsafe} \mid \pi_b] \le \alpha$ ($\alpha$ set by e-stop tolerance / hardware-fault ceiling), not a discountable soft budget like $C_{\mathrm{risk}}(b) \le B_{\mathrm{risk}}$. Risk and compute have different semantics; mixing them invites "spend more risk to buy more compute." **To keep notation tight, this article assumes the deployment $J$ is fixed to a single utility or externally scalarized**; if the project keeps multiple objectives (success / energy / cycle time / wear / safety), $\Delta J$, $Q_{\lambda_t}$, $MV$ should lift to a **Pareto or lexicographic decision layer**, not silently collapse into an unspecified $J$.

Once budget is a vector, the decision variable shifts from "gap" to "intervention": an engineer cannot buy "two percentage points of $\Delta_{\mathrm{model}}$"; what they can buy is 30 min SI, $10^6$ sim steps, 100 real trajectories, a camera calibration, a residual model. Marginal utility on an intervention $m$ is more natural — **an intervention does not directly change $\Delta_k$; it changes the policy through training**:

$$\boxed{\;\pi_{b+m} \;=\; \operatorname{Train}\big(D_{\mathrm{sim}},\ D_{\mathrm{real}};\ m\big)\;}$$

"Where does the next dollar go" is a quantity on interventions, estimated step by step. **The framework centers on $Q_{\lambda_t}$ ($\arg\max$ object) and $MV$ (efficiency reading, not the decision rule)**, both conditioning on $s_t = (b_t, \pi_t, \mathcal{D}_t, h_t)$. **"What is DR's $MV$?" is the wrong question**; the right one is "given current $s_t$, what is one more unit of DR worth?" $m = (\text{role},\,\text{lens},\,\text{protocol}/\text{batch})$ — three **semantic indices** jointly label a candidate (not statistical orthogonality): role = primary operational purpose (adaptation / diagnosis / model update, side effects allowed), lens = mechanism of intervention (Model / Data / Representation / Optimization, itself combinatorial not orthogonal), protocol/batch = scale and recipe. SI / DR / DA / FT are method labels that map into specific (role, lens) cells, not the action space itself. **Cost is also state-conditioned**: $\Delta C(m \mid s_t) = (\Delta C_{\mathrm{real}}, \Delta C_{\mathrm{compute}}, \Delta C_{\mathrm{eng}})$ — the same DR batch costs differently under a busy vs idle GPU; the same real FT differs in cost / feasibility on an overheated robot. $C_\lambda(m \mid s_t) = \lambda_t^\top \Delta C(m \mid s_t)$.

$$\boxed{\;MV(m \mid s_t;\lambda_t) \;=\; \frac{\mu_{\Delta J,t}(m)}{\lambda_t^\top \Delta C(m \mid s_t)},\qquad \mu_{\Delta J,t}(m) \;=\; \mathbb{E}\big[\Delta J(m) \mid s_t\big]\;}$$
Here $\widehat{\Delta J}_t(m)$ denotes the **empirical gain observed in a particular paired evaluation**, while $\mu_{\Delta J,t}(m) = \mathbb{E}[\Delta J(m)\mid s_t]$ is the **belief-weighted expectation of future intervention gain**. The two are never conflated: allocation formulas always use $\mu_{\Delta J,t}$, Step 4 / Step 6 report $\widehat{\Delta J}_t$. Note $MV$ uses $\pi_t$-based notation (rather than the earlier budget-indexed shorthand $\pi_b$); budget is now just a component of $s_t$.

But **$U_0(m\mid s_t) = \mu_{\Delta J,t}(m) - \lambda_t^\top \Delta C(m \mid s_t)$ is the Lagrangian-style performance net utility; the article's actual local decision score $Q_{\lambda_t}$ is not a standard Lagrangian — it augments $U_0$ with a one-step continuation heuristic and, to remain consistent with the global objective, must fold continuation-value uplift (denoted $\mathrm{CVU}$ below) into the local score** (otherwise we would get an approximation break: "global includes CVU, local is performance-only"):

$$\boxed{\begin{aligned}
&U_0(m \mid s_t) \;=\; \mu_{\Delta J,t}(m) \;-\; \lambda_t^\top \Delta C(m \mid s_t)\\[2pt]
&G_0(s) \;:=\; \max_{m' \in \mathcal{M}^{\mathrm{feasible}}(s)}\, U_0(m' \mid s) \quad\text{(one-step performance-only continuation surrogate)}\\[2pt]
&Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t) \;=\; U_0(m \mid s_t) \;+\; \beta\,\mathrm{CVU}(m \mid s_t)\\[2pt]
&s_{t+1} \;=\; \mathcal{T}(s_t,\; m_t^*,\; Y_t),\quad Y_t \sim p(Y\mid s_t,\,m_t^*)
\end{aligned}\;}$$
$\mathcal{T}$ updates $\pi_t$, $\mathcal{D}_t$, $b_t$, and $h_t$ jointly — the most concrete piece is the budget dynamics $b_{t+1} = b_t - \Delta C(m_t^*\mid s_t)$, where $b_t$ denotes the **remaining budget vector** (not cumulative expenditure) and serves as the **online state representation of L1's cumulative constraint**; the two coexist without redundancy — L1 answers "is the whole plan feasible," $b_t$ answers "how much is left for the next local decision." For policy-changing interventions, the earlier budget-indexed shorthand $\pi_{b+m} = \operatorname{Train}(D_{\mathrm{sim}}, D_{\mathrm{real}};\,m)$ becomes a special case of the state-based form; for diagnostic experiments, $\mathcal{T}$ mainly updates $\mathcal{D}_t \cup Y_m$; for model-update interventions, it also refreshes the simulator / surrogate state. All three roles fit one sequential framework — the same intervention (e.g. "30 min SI") induces different $Y_t$ under different $\pi_t$ or hardware condition, which is exactly why the posterior predictive must condition on $s_t$, not only on $\mathcal{D}_t$.

$U_0$ is a **performance-only Lagrangian-style net utility**, and it also serves as the reference for $\mathrm{CVU}$; $\mathrm{CVU}(\cdot)$ only references $U_0$ through the one-step continuation surrogate $G_0(\cdot) := \max_{m'} U_0(m'\mid\cdot)$, not $Q_{\lambda_t}^{\mathrm{perf+CVU}}$ itself, avoiding $Q \leftrightarrow \mathrm{CVU}$ self-reference (a truly self-consistent definition would need a fixed-point; we do not go there). Performance-only $Q_{\lambda_t}^{\mathrm{perf}} = U_0$ is a special case. $MV = \mu_{\Delta J,t} / \lambda_t^\top \Delta C(m \mid s_t)$ is an **efficiency readout**; for **diagnostic-only** actions and for **model-refresh-only actions (which refresh the simulator / surrogate but do not retrain the current policy)**, immediate $\Delta J = 0$ and $MV$ is uninformative (undefined when incremental cost is also zero) — not a formula-derived $MV \equiv 0$, but a scope mismatch: $MV$ tracks performance efficiency, so the value of pure information or model-state improvement must be priced by $\mathrm{CVU}$, not $MV$. **Within the one-step approximation, all non-immediate effects — evidence update, hypothesis posterior, candidate-space opening / closing, safety feasibility, simulator-quality improvement — are absorbed into a single $\mathrm{CVU}$ surrogate; no additional reward channel is defined.** $Q_{\lambda_t}^{\mathrm{perf+CVU}}$ is formally called a **local decision score inducing a greedy one-step allocation policy** (not an RL-style Bellman action value, and **not a standard Lagrangian either** — Lagrangian structure lives only at the $U_0$ layer); it is an estimable one-step local approximation under the current state, and its argmax is precisely L4's computable approximate policy $\mu_t^{Q}(s_t) = \arg\max_m Q_{\lambda_t}(m \mid s_t)$. $\beta$ is a **dimensionless project-level preference weight**; if $\mathrm{CVU}$ and $U_0$ share the same utility scale, set $\beta = 1$.

**$\mathcal{M}_t = \mathcal{M}(s_t)$ state-dependent** — SI done / residual not through gate / safety blocks FT → shrink $\mathcal{M}_t$ directly, not lower $MV$; $\mathcal{M}_t^{\mathrm{feasible}} = \mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}}$. Safety is **two distinct events**: **execution-level** $\mathcal{M}_t^{\mathrm{safe}} = \{m : \mathrm{UCB}_{1-\delta}[P_{\mathrm{exec}}(\text{unsafe} \mid s_t, m)] \le \alpha_{\mathrm{exec}}\}$ gates whether running candidate $m$ itself can push the robot into an unsafe state; **deployment-level** $P_{\mathrm{deploy}}(\text{unsafe} \mid \pi_T, \mathcal{E}_{\mathrm{shared}}) \le \alpha_{\mathrm{deploy}}$ bounds the terminal policy's outcome under the deployment distribution. Both may share the same numerical $\alpha$, but the events do not mix. The $\mathrm{UCB}$ at the candidate gate may come from an empirical frequency model, a posterior predictive risk model, simulation plus an uncertainty bound, or a conservative reachability estimate — **Clopper–Pearson is one instance for binary execution outcomes, not the only one**. **The execution-level safety gate is non-trivial only for candidates that carry real execution risk.** For pure-computation or offline diagnostic candidates — simulation-only diagnosis, offline calibration, or model-refresh-only actions that do not touch the physical robot — $P_{\mathrm{exec}}$ degenerates to $0$ or to a deterministic feasibility check, and $\mathcal{M}_t^{\mathrm{safe}}$ is automatically satisfied on that subset. **Notation locked: $\mathcal{D}_t$ denotes raw evidence / history; what actually enters the decision is its sufficient compression — the belief state $q_t = q(\mathcal{D}_t)$ — so the formal Markov decision state is written $s_t = (b_t,\,\pi_t,\,q_t,\,h_t)$ throughout. Wherever $\mathcal{D}_t$ appears below it is shorthand for the evidence that generates $q_t$ and may be read as its belief compression $q_t$; raw history need not be retained.** $m$ covers policy-changing interventions and diagnostic experiments (role ∈ {adaptation, diagnosis, model update}; **role denotes the action's primary operational purpose and permits side effects — e.g. a single SI measurement can be a diagnosis and an implicit model update at the same time**); same type at different batch / recipe / protocol counts as distinct candidates. Decision rule:

$$m_t^* \;=\; \arg\max_{m \,\in\, \mathcal{M}_t^{\mathrm{feasible}}}\; Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t),\qquad s_{t+1} = \mathcal{T}(s_t,\, m_t^*,\, Y_t)$$
Time indices align exactly with the transition: at $s_t$ choose $m_t^*$, observe $Y_t$, enter $s_{t+1}$. **Note that L4 does more than produce a scalar score: the argmax itself induces a state-conditioned *approximate allocation policy* $\mu_t^Q : s_t \mapsto m_t$, with $m_t^* = \mu_t^Q(s_t)$; the next round's $m_{t+1}$ is then decided by $\mu_{t+1}^Q(s_{t+1})$. This is what makes L4 semantically compatible with L1's global adaptive allocation policy (see L1 below) rather than an open-loop plan.**

**$MV$ stays as an efficiency reading across interventions, not the decision rule.** The two **answer different questions** — $MV \to$ efficiency, $Q_{\lambda_t} \to$ decision — **not two competing rankings**; a toy makes the divergence concrete (**$\lambda = (3,\ 0.1,\ 1)$ is a hand-calibrated resource weight, not a dual-derived shadow price**; $\Delta C = (\text{real-h},\ \text{compute},\ \text{eng-h})$; **toy takes $\beta = 0$** to isolate $MV$ from performance-only net utility $Q_{\lambda_t}^{\mathrm{perf}} = U_0$):

| Intervention | $\mu_{\Delta J,t}$ | real h | compute | eng h | $C_\lambda$ | $MV$ | $Q_{\lambda_t}^{\mathrm{perf}}$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 min SI | 1.5 | 0.2 | 0.5 | 0.5 | 1.15 | **1.30** | 0.35 |
| Big DR batch | 3.0 | 0.0 | 20.0 | 0.4 | 2.40 | 1.25 | **2.00** |
| Camera DA | 2.5 | 1.0 | 3.0 | 0.5 | 3.80 | 0.66 | −1.30 |
| Real FT | 5.0 | 2.0 | 1.0 | 1.0 | 7.10 | 0.70 | −2.10 |

$MV$ ranks SI first; $Q_{\lambda_t}^{\mathrm{perf}}$ ranks DR first — ratio reads efficiency, Lagrangian reads net value after opportunity cost. **The $Q$ column here is $Q_{\lambda_t}^{\mathrm{perf}} = U_0$ (the $\beta = 0$ special case of the formal $Q_{\lambda_t}^{\mathrm{perf+CVU}}$)**; the example deliberately turns off the continuation-value channel to isolate efficiency from net value, not to claim information is useless. Domain condition: **when $\lambda_t^\top \Delta C > 0$**, $MV < 0 \Leftrightarrow \mu_{\Delta J,t} < 0$; $Q_{\lambda_t}^{\mathrm{perf}} < 0 \Leftrightarrow \mu_{\Delta J,t} < C_\lambda$. DA / FT dropped by the local net-value stop (economic stop) under this $\lambda_t$.

The novelty is not "which method is better" but **given current evidence, the expected value of the next intervention**. $m_t^*$ is a one-step / local rule, not a global optimum; when we write down L1 as a global optimization, **the decision variable must be an allocation policy sequence $\{\mu_t\}_{t=1}^T$ (with $\mu_t : s_t \mapsto m_t$ a state-conditioned decision rule), not an open-loop action sequence $\{m_t\}$** — the latter would be read as "fix $m_1, \ldots, m_T$ up front," which is precisely what this article's *adaptive sequential experimentation* thesis ($Y_t \to s_{t+1} \to$ re-choose $m_{t+1}$) rejects; using $\{m_t^*\}$ instead would further conflate L4's heuristic argmax with L1's global decision variable and collapse the hierarchy. **Each $\mu_t$ must be a history-measurable / adapted policy: it depends only on the state $s_t = (b_t,\pi_t,q_t,h_t)$ available at step $t$, so the sequence satisfies non-anticipativity — future realizations $Y_{t'}\ (t' > t)$ cannot inform the current action selection, and $\mu_t$ is not an oracle that already knows $Y_{t+1},Y_{t+2},\ldots$** **L1 states the cumulative resource constraint $\sum_t \Delta C_r(\mu_t(s_t) \mid s_t) \le B_r$ at the global-feasibility level, while $b_t$ is the same budget's online state representation, evolving via $b_{t+1} = b_t - \Delta C(\mu_t(s_t) \mid s_t)$ for local decision use. Writing both is not double-counting — L1 answers "is the whole plan feasible," $b_t$ answers "how much is left for the next local choice," and only the second is a real decision input at step $t$.** **This article treats $\Delta C(m \mid s_t)$ as the *realized incremental resource consumption* observed after executing candidate $m$; consequently, L1's budget constraint is read *pathwise* — "on every realized execution path, cumulative $\Delta C_r \le B_r$." Because $s_t$ is itself a stochastic state, this is a stochastic cumulative constraint, read strictly in the **almost-sure sense**, i.e. $\Pr\!\big(\sum_t \Delta C_r(\mu_t(s_t)\mid s_t)\le B_r,\ \forall r\big)=1$. If cost has substantial stochasticity (robot repair, unexpected engineering effort, etc.), L1 may be reformulated as an expected or chance-constrained resource budget; the main text keeps the pathwise realized-cost reading for simplicity.** Formally, the whole object can be read as a belief-state adaptive allocation problem (sequential decision-making on $q_t$); the article does not pursue its exact Bellman solution, nor treat it as a standard POMDP — we take this positioning and go no further.** Safety splits across two levels: **execution-level** is gated every step through $\mathcal{M}_t^{\mathrm{safe}}$ (event $P_{\mathrm{exec}}$); **deployment-level** appears only as a terminal chance constraint (event $P_{\mathrm{deploy}}$). Full problem: **multi-resource sequential allocation with chance constraint over an adaptive allocation policy**.

$$\boxed{\;\max_{\{\mu_t\}_{t=1}^{T}}\ \mathbb{E}\big[J_{\mathrm{real}}(\pi_T)\big] \quad \text{s.t.}\quad \sum_{t} \Delta C_r\!\big(\mu_t(s_t) \mid s_t\big) \le B_r\ (r \in \{\mathrm{real},\mathrm{compute},\mathrm{eng}\}),\;\; P_{\mathrm{deploy}}(\text{unsafe} \mid \pi_T, \mathcal{E}_{\mathrm{shared}}) \le \alpha_{\mathrm{deploy}},\;\; m_t = \mu_t(s_t).\;}$$

$MV$ / $Q_{\lambda_t}$ are **local decision statistics**; $\lambda_r$ **under suitable regularity of the value function and constraint qualification can be interpreted as the marginal value of resource $B_r$ at the optimum** (the ideal shadow price), but this article only needs a resource-weight estimate consistent with that meaning: $\lambda_t = \lambda(s_t)$ (expandable in an implementation as $\lambda(s_t) = \lambda(b_t, q_t, \pi_t, h_t)$, so $h_t$ is no longer artificially dropped) — updates with allocation state, one pivotal experiment can shift it sharply; hereafter we abbreviate uniformly as **resource weights $\lambda_t$**. Since $\lambda_t$ is itself state-dependent, the continuation at $s_{t+1}$ uses the **updated** $\lambda_{t+1} = \lambda(s_{t+1})$; hence $U_0(m' \mid s_{t+1}) = \mu_{\Delta J, t+1}(m') - \lambda_{t+1}^\top \Delta C(m' \mid s_{t+1})$ uses the *next-step* shadow price, not a hold-over of $\lambda_t$ — only with this update does $s_t \to \lambda_t \to m_t \to s_{t+1} \to \lambda_{t+1}$ close. Fixed-$\lambda$ greedy approximates global optimality only under negligible interaction, linear cost, and no fixed cost. Five caveats: **(i) uncertainty and the decision functional** — main text uses posterior mean; risk-sensitive projects may swap LCB / CVaR-adjusted utility — do not write formula with mean and prose with LCB. **(ii) non-linear cost** — SI fixed, DR diminishing, FT threshold, negative transfer can all drive $MV < 0$. **(iii) $\mathrm{CVU}$ uses a counterfactual definition to avoid double-counting $\Delta C$ with $U_0$** — we deliberately write $\mathrm{CVU}(m\mid s_t)$ (**signed, one-step, candidate-relative, performance-only continuation uplift heuristic**) rather than $\mathrm{VoI}$ or a Bellman continuation term. The reason for the counterfactual form is accounting hygiene: $U_0$ already subtracts the current action's $\lambda_t^\top \Delta C(m\mid s_t)$, and $b_{t+1}$ inside $s_{t+1}$ already reflects the same budget depletion, so a naive "$\max U_0(m'\mid s_{t+1}) - \max U_0(m'\mid s_t)$" would penalize the same $\Delta C$ twice (once directly in $U_0$, once indirectly through the shrunken future feasible set). **The clean definition therefore compares two counterfactual continuations**:
$$\mathrm{CVU}(m\mid s_t) \;=\; \mathbb{E}_{Y \sim p(\cdot\mid s_t, m)}\!\big[G_0(s_{t+1}^{m, Y})\big] \;-\; G_0(s_{t+1}^{\varnothing}),\qquad G_0(s) := \max_{m' \in \mathcal{M}^{\mathrm{feasible}}(s)} U_0(m' \mid s).$$
Here $s_{t+1}^{m,Y}$ is the state after executing $m$ and observing $Y$ (carrying $b_{t+1} = b_t - \Delta C(m\mid s_t)$, $\pi_{t+1}$, $q_{t+1}$, $h_{t+1}$, $\lambda_{t+1}$); $s_{t+1}^{\varnothing}$ is the **counterfactual next state under the same time / background-drift convention when $m$ is not executed** (it preserves the current policy $\pi_{t+1}^{\varnothing} = \pi_t$ and the un-deducted action-specific budget ($\Delta C(m)$ is not subtracted from $b_t$), advancing only the same background time / drift process as executing $m$; all other conventions are matched). Under this definition, **$U_0$ handles the current action's gain-cost, $\mathrm{CVU}$ handles its marginal effect on the future decision state / opportunity set (policy, belief, budget, hardware, candidate set) — the same $\Delta C$ is only counted once**. $\mathrm{CVU}$ is still not VoI or a Bellman continuation value function: $s_{t+1}$ carries evidence update, policy change, hardware drift, and candidate-set change; the standard continuation term should be $V_{t+1}(s_{t+1})$, but we substitute $G_0$ — hence it is a heuristic, not a true value function. **$\mathrm{CVU}$ may be negative** — budget depletion, hardware degradation, policy transition, candidate elimination, or adverse evidence can all make $G_0(s_{t+1}^{m,Y})$ fall below $G_0(s_{t+1}^{\varnothing})$; **this article no longer argues about whether "information has negative value"** — $\mathrm{CVU}$ is a net continuation uplift and its sign follows directly from the counterfactual difference above. **Terminal convention**: at step $T$ no next decision exists; by convention $V^{\mathrm{cont}}_{T+1}(s) := 0$, so terminal $Q_T = U_0(m\mid s_T) + \beta \cdot 0 = U_0$ — this is an explicit convention, not a "CVU automatically degenerates" claim. $\beta$ is a **dimensionless project-level preference weight**; if $\mathrm{CVU}$ and $U_0$ share the same utility scale, set $\beta = 1$. $V(\mathcal{D}) = -\Pr(\arg\max Q_{\lambda_t}$ flips$)$ is only a decision-stability proxy; posterior narrowing $\neq$ decision value. **(iv) $\Delta J$ is not a natural causal effect** — matched / paired evaluation; $\Delta C$ includes all incremental cost; $\widehat{\Delta J}_t(m)$ (realized) and $\mu_{\Delta J,t}(m) = \mathbb{E}[\Delta J(m)\mid s_t]$ (belief expectation) are two distinct levels — formulas use $\mu_{\Delta J,t}$, evaluation reports $\widehat{\Delta J}_t$. **(v) Diagnostic-only actions and model-refresh-only actions (which refresh the simulator / surrogate but do not retrain the current policy)** both satisfy immediate $\pi_t^m = \pi_t^{\mathrm{control}}$, $\mu_{\Delta J,t} = 0$ — **note that if the model update itself includes "retrain the policy with the new model," $\Delta J \neq 0$ and the action should be treated as adaptation**; their immediate value is therefore exactly $-\lambda_t^\top \Delta C$ (a net opportunity cost), and the informational benefit is realized entirely through the $\mathrm{CVU}$ counterfactual-continuation term. **$MV$ is uninformative for these (undefined when incremental cost is also zero)**; **within the one-step approximation, all non-immediate effects — evidence update, hypothesis posterior, candidate-space opening / closing, safety feasibility, simulator-quality improvement — are absorbed into a single $\mathrm{CVU}$ surrogate; no additional reward channel is defined**. Values only meaningful under fixed $p_{\mathrm{eval}}$; comparing $MV(m \mid s_t)$ across time additionally requires the utility ($J$) scale and the resource-weight calibration (the interpretation of $\lambda_t$) to stay consistent — otherwise $MV$ values at different times no longer sit on the same economic scale and cannot be joined into a single trend line.

**$MV_i = MV_i(s_t)$ is state-dependent.** SI first can lower DR's $MV$; DR first can raise FT's — direction depends on interaction, no fixed monotonic law. **Interventions exhibit complementarity, substitutability, and occasional conflict** (not a bandit). Feedback: interventions change the policy, which changes $S_k^{\mathrm{int}} = S_k^{\mathrm{int}}(\pi)$:

One more layer of feedback: **an intervention does not only push the gap down — it changes the policy, and thereby the policy's own sensitivity to the gap** — $S_k^{\mathrm{int}} = S_k^{\mathrm{int}}(\pi)$, $\pi = \pi(m)$, loop not one-directional:

```
estimate mismatch → estimate sensitivity → intervention
       ↑                                          ↓
   re-estimate  ←  sensitivity changes  ←  policy changes
```

**This feedback loop fits the allocation thesis better than any new equation**: sim-to-real is not solved once; it is a sequential experiment where each round re-estimates for the next.

Each intervention mapped to its primary compressed term and budget:

| Intervention | Primary term compressed | Primary budget |
| --- | --- | --- |
| System Identification | $\Delta_{\mathrm{model}}$ | $C_{\mathrm{eng}}$ + $C_{\mathrm{compute}}$ + a little $C_{\mathrm{real}}$ |
| Domain Randomization | $\Delta_{\mathrm{model}} + \Delta_{\mathrm{dist}}$ | $C_{\mathrm{compute}}$ (sample efficiency) |
| Residual physics | $\Delta_{\mathrm{model}}$ (residual part) | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Domain Adaptation | $\Delta_{\mathrm{obs}}$ (appearance subset) | $C_{\mathrm{real}}$ (unlabeled) + $C_{\mathrm{compute}}$ |
| Real-world fine-tuning | **no single mismatch; changes the policy through target-domain optimization** (alters transfer delta and learning gap simultaneously) | $C_{\mathrm{real}}$ (wear / safety) |
| World model | change model source | $C_{\mathrm{real}}$ + $C_{\mathrm{compute}}$ |
| Sim-and-real co-training | change $p_{\mathrm{train}}$ (mostly $\Delta_{\mathrm{dist}}$) | mixed ($C_{\mathrm{real}}+C_{\mathrm{compute}}$) |

With this framing, the article is not "which of the four methods is better" but a loop: locate the dominant $\Delta_k$, judge importance via sensitivity, take $m_t^* = \arg\max Q_{\lambda_t}$ over the feasible set, measure return in real evaluation, decide the next unit.

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

## Evaluation: how do you know you actually closed the gap?

**All claims rest on a three-tier evidence stack** — $\boxed{\text{A: mechanism}\quad \text{B: policy-response}\quad \text{C: deployment}}$ — A: friction ID / calibration / latency; B: $\hat S_k^{\mathrm{int}}$ / ablation / finite-difference; C: real $\Delta J$ / $Q_{\lambda_t}$ / $MV$ / sim ranking. **Tiers are evidence levels, not a fixed execution order** — diagnosis cycles between them (deployment failure → suspect latency → back to A); but **cannot substitute** — SI fitting well is A, not C improvement.

Reporting performance only on sim benchmarks is dangerous. A credible evaluation should at least:

Reporting performance only on sim benchmarks is dangerous. A credible evaluation should at least:
- report **zero-shot transfer** alongside curves after **few-shot / N-shot** adaptation;
- test on **held-out hardware / calibration / object / contact / environmental regimes**;
- declare whether **task / initial-state / evaluation distributions** match between sim and real;
- do **failure attribution**: which $\Delta_k$ dominates? wrong attribution sends the budget to the wrong place;
- **not just means**: at least mean ± CI across seeds / resets; prefer **paired evaluation**;
- **report safety failures separately**: $J_{\mathrm{real}}$ alongside violation / e-stop / intervention count / hardware fault / recovery time; **for low-frequency events, "zero failures in 20 runs" cannot conclude failure probability is low** — use binomial UCB or CVaR-style **tail-risk measure**, not mean ± CI. **Concrete estimator**: with $X \sim \mathrm{Binomial}(n, p)$ and $X = 0$, only an upper confidence bound on $p$ (Clopper–Pearson or Bayesian Beta posterior $1-\delta$ quantile) turns $\Pr[\text{unsafe}] \le \alpha$ into a **gate on the upper bound $\le \alpha$**, not a point-estimate comparison — this keeps the chance constraint from staying purely symbolic.

Following "the simulator is a proxy for reality," a more fundamental question than numerical alignment: **can the sim correctly predict which policy is better?**

A **conceptual example** (numbers do not represent experimental results):

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

In sim it looks like $A > B > C$; on the real robot it is $B > C > A$. Here the simulator has **lost model-selection utility** — you would use it to pick out the worst policy. So **when the simulator is used for policy / model selection**, look at rank correlation $\rho_{\mathrm{rank}} = \mathrm{Spearman}(J_{\mathrm{sim}}(\pi_i), J_{\mathrm{real}}(\pi_i))$ together with selection regret:

$$\pi_{\mathrm{sim}} = \operatorname*{arg\,max}_{\pi \in \Pi} J_{\mathrm{sim}}(\pi), \qquad R_{\mathrm{select}} = J_{\mathrm{real}}\big(\pi^{*}_{\mathrm{real}}\big) - J_{\mathrm{real}}\big(\pi_{\mathrm{sim}}\big)$$

**On a larger policy pool**, even $\rho_{\mathrm{rank}} = 0.95$ can still miss the true top-1 — the disaster is unchanged; conversely $\rho_{\mathrm{rank}} = 0.7$ can be enough to "pick one deployable policy" as long as top-1 is rarely wrong. (**Note**: this Spearman intuition refers to a large policy pool; on the $A/B/C$ three-policy toy above, Spearman $\rho$ can only take a discrete set of values and $0.95$ is not applicable — the continuous-number example belongs to the general case, not that toy.) Both are **conditional metrics**. **The allocation framework naturally yields**: **simulator fidelity is task-of-use dependent, not absolute** — change the use (pretraining / exploration / curriculum / safety filter) and "which errors matter" changes entirely. $\pi^*_{\mathrm{real}}$ is typically unavailable, so $R_{\mathrm{select}}$ — like the earlier learning gap — is **oracle-defined**; in practice use $J_{\mathrm{real}}(\pi_{\mathrm{best\text{-}validated}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$ as a **validated-best observed proxy** (best on an independent audit / held-out slice, not argmax over a single noisy evaluation, avoiding winner's curse).

**More importantly**, real projects rarely need the sim to precisely rank every policy — only to narrow candidates to an acceptable set. **top-$k$ recall** and **regret@k** should be peers of ranking. **Beware adaptive selection bias**: if sim adaptively filters policies (sim select → real eval → update → re-select), using the same selected candidates to evaluate sim creates self-confirming loops. **Maintain two pools**: $\Pi_{\mathrm{adapt}}$ for training/selection, $\Pi_{\mathrm{audit}}$ for held-out evaluation. **Held-out evaluation sets are not infinitely immune** — long-running projects should reserve an audit slice or periodically refresh the evaluation set to avoid adaptive experimentation overfitting a fixed real benchmark.

At this point, **a corollary of the allocation framework**: **simulator utility is not a single property but three non-substitutable dimensions — and it must be validated by independent real evidence; internal consistency, low prediction loss, or high training reward cannot alone prove downstream utility** —

| Simulator utility dimension | Typical metric |
| --- | --- |
| Numerical prediction accuracy (absolute error / calibration) | MAE / RMSE $\mathbb{E}\big[|J_{\mathrm{sim}}(\pi) - J_{\mathrm{real}}(\pi)|\big]$, calibration curve, prediction interval coverage |
| Ranking accuracy | Spearman $\rho_{\mathrm{rank}}$, Kendall $\tau$, top-k recall, regret@k |
| Quality of the selected policy (decision quality) | $R_{\mathrm{select}} = J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$ (in practice use a **best-validated proxy** — argmax on an independent audit slice, not on a single noisy eval) |

A simulator can be very well calibrated and still pick the wrong policy (narrow distribution); another can be numerically wrong across the board yet rank stably with small regret — the three dimensions cannot substitute, **and the three metric families differ not only in scale but in the loss they optimize, so there is no natural universal scalar simulator score**. $U_{\mathrm{sim}}$ should not be an abstract scalar; index it **by use as a superscript**: $U_{\mathrm{sim}}^{(u)}$, $u \in \{\text{pretrain},\ \text{selection},\ \text{exploration},\ \text{curriculum},\ \text{safety}\}$. Evaluating fidelity is not staring at a single policy; it must be relative to the **candidate family** and the **concrete use**: $U_{\mathrm{sim}}^{(u)}(\cdot \mid \Pi_{\mathrm{candidate}},\ p_{\mathrm{eval}}^{\mathrm{real}})$.

## Composition, decision, and a question usually dodged

With priorities in hand, a more useful shape for real projects is a **gap × modellability × real-budget** decision matrix:

| Gap | Parameterizable / identifiable? | Real data | Natural candidates (final choice still set by state-conditioned $\Delta J,\Delta C,\mathrm{CVU}$) |
| --- | --- | ---: | --- |
| low-dimensional dynamics bias | high | scarce | SI |
| parameterizable dynamics uncertainty | medium | scarce | posterior-guided DR / Bayesian SI → DR |
| dynamics residual | low (but structured) | medium | Residual learning |
| visual appearance | high | none / scarce | DA / DR (candidates) |
| actuator latency | high | scarce | SI + DR |
| unobserved rare tail, representable by current model family | low | scarce | targeted simulation / DR |
| unknown long-tail, sim untrustworthy | low | medium | real data |
| model class uncertain | low | abundant | learned world model (if real is scarce, prefer physics prior + residual / DR) |
| mixed | mixed | mixed | co-training candidate (verify positive-transfer conditions first) |

**The first-two-row qualifiers cannot be dropped**: if uncertainty is **model-class uncertainty** (the sim's functional form cannot express the real phenomenon), neither SI nor DR may apply — fall to residual / WM / real-data rows first. Second-to-last: "model unknown" alone does not imply WM; criterion is **model uncertainty × real-data budget** — learned WM is reasonable only when the model class is uncertain **and** real interaction is abundant. Last row: "co-training as a safety net" clashes with the allocation thesis — when sim quality is bad, real data scarce, and the two disagree on action space / task semantics, negative transfer is entirely possible.

A common combo is **SI → DR → DA → co-training / FT**: **arrows are schematic, not a fixed workflow** — real order is set by the currently dominant gap and marginal utility. **When sim has strong coverage and the dominant unknown is model misspecification**, the most valuable use of real data is not broad coverage but **discovering failure modes the sim has not modeled**, then having sim amplify them; **when the deployment distribution is already fairly fixed**, real data mainly serves direct adaptation / imitation, no need to walk discovery / amplify first —

$$\text{discover real tail} \rightarrow \text{identify structure} \rightarrow \text{synthetically amplify} \rightarrow \text{real validation}$$

**Real discovers, sim amplifies, real re-validates.** Hard precondition: the discovered failure modes must be representable in the current model class / learned surrogate with acceptable fidelity; otherwise after discovery turn directly to a richer model / WM / more real data, rather than force an unrepresentable tail through sim amplification (this is the concrete form of the model-class-uncertainty point above).

This lets us answer the counter-question the article has almost dodged but the framework itself allows: **when is the optimal move to not do sim-to-real at all?**
- **Real data already so cheap that $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}^{\mathrm{effective}}$** — $C_{\mathrm{real}}^{\mathrm{effective}}$ = **effective real-robot cost** (safety / operator / reset / wear / failure recovery / deployment diversity). Compare "expected cumulative value / cost within the current budget horizon," not "raw hours of one intervention."
- **Simulator's model class itself is bad** ($\Delta_{\mathrm{model}}$ dominates, hard to parameterize — soft bodies / fluids / complex contact) — fixing sim has such low marginal utility that WM or real-data learning is often cheaper.
- **Deployment distribution is very fixed** — no need for large-scale DR; targeted real FT is usually more cost-effective.
- **Simulator offers no unique coverage / safety / exploration / counterfactual access** — $U_{\mathrm{sim}}^{\mathrm{downstream}} < C_{\mathrm{sim}}^{\mathrm{effective}}$: not that sim is "bad," but no **unique utility**, opportunity cost exceeds benefit.

Admitting "sometimes the optimal move is not doing sim-to-real" is exactly what the allocation framing looks like: **it does not take the "simulation" team; it takes the "next unit of budget buys the most real-world performance" team.** Sequential allocation needs an explicit stopping rule with three triggers: **(a) local net-value stop (economic stop)** — $\max_{m \in \mathcal{M}_t^{\mathrm{feasible}}} Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t) \le 0$; local one-step stop, not global optimal stopping — if a complementary portfolio is known a priori, evaluate as a portfolio candidate. **(b) continuation-value stop** — **the best remaining positive continuation uplift is already near zero**: $\max_{m \in \mathcal{M}_t^{\mathrm{feasible}}} \mathrm{CVU}(m \mid s_t) \le \varepsilon$ for a small positive threshold $\varepsilon$ (note: because $\mathrm{CVU}$ may be negative, an "expected $\mathrm{CVU} \approx 0$" formulation silently misses the case "current best CVU is strongly negative, must stop immediately" — the $\max$ operator, not the expectation, is the correct stopping test). **(c) safety / feasibility stop** — remaining candidates all outside feasible set. Any one triggers stop, not "spend down by default."

## A minimum executable Sim-to-Real allocation protocol

A framework that never lands on "how the project runs tomorrow" is only clever framing. Six steps below are the **minimum executable version** — any one can be skipped, but only with an explicit reason it is a no-op here.

**Step 1 — Freeze the evaluation.** Lock down task / initial-state distribution / horizon / success metric / safety threshold / policy interface (obs + action schema + control frequency). **If $\pi$ is stochastic ($a_t \sim \pi_\theta(\cdot \mid o_t)$), $J(\pi)$ is the expectation over policy / reset / hardware randomness under the evaluation protocol**, estimated via repeated / block runs. Without this, every downstream $\Delta J$ uses a different ruler.

**Step 2 — Build a held-out real evaluation set.** Real evaluation data must be **strictly disjoint from real training data** and cover held-out hardware / calibration / objects / scene slices. Evaluating interventions on training data makes $\widehat{\Delta J}$ systematically optimistic. **But eval results can still feed the allocator's belief update**: $\mathcal{D}_t$ is "all evidence available to the allocator at step $t$," which includes $D_{\mathrm{eval}}$-derived failure modes and uncertainty shifts. "Not participating in training" and "participating in posterior update" are two separate claims — not contradictory.

**Step 3 — Enumerate mismatch hypotheses as a falsifiable table.**

| Hypothesis | Evidence | Belief | Candidate intervention |
| --- | --- | ---: | --- |
| friction $\mu$ too low | contact slip | med | SI + DR |
| actuator latency unmodeled | high-frequency oscillation | high | SI + timing re-ID |
| camera extrinsics off | systematic grasp offset | high | Calibration / DA-input-level |
| contact model wrong | soft-object OOD failure | low | Residual / world model |

Every hypothesis must be **falsifiable by a concrete experiment**; drop any that cannot specify what would refute it.

**Step 4 — one-time initial calibration pilot** (Step 5 is where sequential adaptive allocation begins, avoiding the pilot-selection circularity). **For actions that will directly change the current policy, estimate an immediate effect distribution** for $\mu_{\Delta J,t}(m)$ (Bayesian implementations realize it as a posterior; frequentist implementations report a CI); **for diagnosis and model-refresh-only actions, estimate evidence quality / continuation uplift distribution instead** — their immediate $\Delta J \equiv 0$ means there is no meaningful "immediate effect distribution" to estimate, and the experimental target is the CVU-side evidence and posterior improvement. Screening a diagnostic by performance gain is a category error. No preset sample count. **$\widehat{\Delta J}_t(m) = J_{\mathrm{real}}(\pi_t^{m}) - J_{\mathrm{real}}(\pi_t^{\mathrm{control}})$** — control bears the same extra training steps, **same elapsed time (so robot temperature, battery, wear, and other background drift are matched)**, and the same **training seed**; note that real hardware itself has no "seed" to share, so the physical side is aligned through matched evaluation blocks / hardware conditions, not through seed equality. Only this intervention is toggled. $\widehat{\Delta J}_t$ is incremental deployment utility, not absolute post-intervention performance; **for diagnostic-only actions and for model-refresh-only actions (update simulator / surrogate without retraining the current policy), $\widehat{\Delta J}_t \equiv 0$; within the article's one-step approximation their value is aggregated through $\mathrm{CVU}$ as a unified continuation surrogate — no separate reward channel is defined**. Matched / paired / block evaluation: same training seed, and where feasible matched evaluation blocks / hardware conditions on the same held-out slice; record drift (tire, motor, battery). **A single-intervention matched control identifies the incremental effect relative to the current protocol; it does not identify higher-order interaction effects. Combined actions (e.g. SI + DR, or SI + WM refresh) must be evaluated as independent candidates through their own matched comparison**; otherwise synergy / conflict cannot be recovered from data.

**Step 5 — sequential adaptive allocation**: $m_t^* = \arg\max_{m \in \mathcal{M}_t^{\mathrm{feasible}}(s_t)} Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t)$ — $\lambda_t$ is a resource-weight estimate; objective and local score must agree. Cost and budget are simultaneously state-conditioned ($\Delta C(m \mid s_t)$, $b_{t+1} = b_t - \Delta C(m_t^*\mid s_t)$). Execution-level safety flows through $\alpha_{\mathrm{exec}}$ gate on every step; deployment-level safety is a terminal $\alpha_{\mathrm{deploy}}$ chance constraint on $\pi_T$; neither enters the cost term.

**Step 6 — Real evaluation → belief update → back to Step 3.** Update $\mathcal{D}_t \rightarrow \mathcal{D}_{t+1}$ (Bayesian implementation realizes this as a posterior; other implementations use moment / CI updates), re-estimate $\lambda_t$, retire falsified hypotheses, add newly observed failure modes, run the next round. **The most-skipped, most-important step** — without belief update the pipeline degrades to a static checklist.

**Positioning.** This is the **minimum landing version** of the allocation framework, not the only implementation. Small teams can merge Step 3 and Step 4; larger teams can add a portfolio-optimization layer on top of Step 5. But none of the six steps may stay implicit — writing them down makes review possible, and review is what stops allocation from quietly degrading into "using whichever method the team already knows."

## What this means: a loop, not a switch

The core sentence of [Data Scaling for Robots](/en/articles/2026-09-09-robot-data-scaling/) is evaluation-aware distribution allocation. Applied to sim-to-real — **simulation data's utility is never an internal property of the simulator; it is a property relative to the real evaluation distribution:**

$$U\big(D_{\mathrm{sim}} \mid \mathcal{L},\ p_{\mathrm{eval}}^{\mathrm{real}}\big)$$

This explains a common frustration: "more sim data" sometimes does not help — **when the dominant bottleneck is a support / fidelity mismatch between sim and the real evaluation distribution, marginal return of adding same-distribution samples drops quickly; adding data cannot create evaluation-relevant coverage or correct model bias**. Instead of "how good is my sim," ask: "in which evaluation-relevant directions is my sim close to reality and in which does it fall short? For the ones it falls short on, how sensitive are they, and which budget pushes each most cheaply?"

Walk that line through and sim-to-real stops being a "did the transfer succeed" switch and becomes a loop:

$$\boxed{\ \text{diagnosis} \rightarrow \text{sensitivity / uncertainty} \rightarrow \text{intervention} \rightarrow \text{performance} + \text{information gains} \rightarrow \text{update }\mathcal{D}_t \rightarrow \text{re-allocate} \rightarrow\ \circlearrowleft\ }$$

The corresponding **closed-loop spine**, which fits the article's thesis better than any four-way table:

```text
              current state  s_t = (b_t, π_t, q_t, h_t)   ← b_t = remaining budget
                                  │
                                  ▼
                     mismatch diagnosis (D_t vs real)
                                  │
                                  ▼
                 sensitivity / uncertainty attribution
                                  │
                                  ▼
         candidate action  m = (role, lens, protocol/batch)
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         diagnosis         adaptation       model update
      (any lens:         (Model / Data /  (Model / Data /
       Model / Data…)     Represent. /     Represent. /
                          Optimization)    Optimization)
              └────────────────┼────────────────┘
                               ▼
                  real evaluation (paired, CI)
                               │
                    performance uplift  +  continuation uplift
                     (ΔJ_real)          (CVU: q_t, π_t, M_{t+1})
                               │
                               ▼
                update  s_{t+1} = T(s_t, m_t^*, Y_t),  Y_t ~ p(·|s_t, m_t^*)
                               │
                               ▼
                    stopping rule?  → deploy π_T
                               │
                               └────► next round (loop)
```

Role × lens × protocol/batch are **three semantic indices** that jointly label a candidate — they are categorization dimensions at different levels and do **not** assume statistical orthogonality or physical independence. Role describes the action's **primary operational purpose** inside the sequential loop (adaptation / diagnosis / model update, **side effects permitted** — a single SI measurement may primarily be a diagnosis with an implicit model refresh on the side), lens describes the mechanism through which it intervenes (Model / Data / Representation / Optimization, **itself combinatorial rather than orthogonal**), protocol/batch fixes scale and recipe. **SI / DR / DA / FT are method labels that map into specific (role, lens) cells, not the action space of allocation** — the same method can appear in different roles (SI as adaptation vs SI as pure measurement). The true decision unit is a state-conditioned action $m \in \mathcal{M}_t^{\mathrm{feasible}}(s_t)$; the candidate set itself is state-dependent: a diagnostic experiment can open or close downstream adaptation / model-update feasibility, and the last step changes the sensitivity and mismatch inside $s_{t+1}$, driving the feedback loop.

This chain is a **resource-constrained adaptive sequential experimentation framework**: sensitivity and marginal return are both estimated via small-step experiments on real evaluation, each round ending by deciding the next budget slice. The framework collapses into a **five-layer spine** that everything else hangs from:

$$\boxed{\begin{aligned}
&\textbf{L1}:\ \max_{\{\mu_t\}_{t=1}^{T}}\ \mathbb{E}[J_{\mathrm{real}}(\pi_T)]\quad\text{s.t.}\ \textstyle\sum_t \Delta C_r(\mu_t(s_t)\mid s_t)\le B_r,\ m_t=\mu_t(s_t),\ P_{\mathrm{deploy}}(\text{unsafe}\mid \pi_T, \mathcal{E}_{\mathrm{shared}})\le \alpha_{\mathrm{deploy}}\\
&\textbf{L2}:\ s_t = (b_t,\,\pi_t,\,q_t,\,h_t),\quad q_t = q(\mathcal{D}_t)\ \text{(belief state)},\quad b_t\ \text{= remaining},\quad b_{t+1} = b_t - \Delta C(m_t\mid s_t)\\
&\textbf{L3}:\ m_t \in \mathcal{M}_t^{\mathrm{feasible}}(s_t),\quad \mathcal{M}_t^{\mathrm{feasible}} = \mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}},\ \mathcal{M}_t^{\mathrm{safe}}\ \text{gates}\ P_{\mathrm{exec}}\\
&\textbf{L4}:\ m_t^* = \arg\max_{m\in\mathcal{M}_t^{\mathrm{feasible}}} Q_{\lambda_t}^{\mathrm{perf+CVU}}(m\mid s_t),\quad Q_{\lambda_t}^{\mathrm{perf+CVU}} = U_0(m\mid s_t) + \beta\,\mathrm{CVU}(m\mid s_t)\\
&\textbf{L5}:\ MV(m\mid s_t) = \mu_{\Delta J,t}(m)\;/\;\lambda_t^\top \Delta C(m\mid s_t),\quad \mu_{\Delta J,t}(m) = \mathbb{E}[\Delta J(m)\mid s_t]\\
&\textbf{Transition}:\ s_{t+1} = \mathcal{T}(s_t,\, m_t^*,\, Y_t),\quad Y_t \sim p(\cdot\mid s_t, m_t^*),\quad \lambda_{t+1} = \lambda(s_{t+1})\\
&\textbf{Terminal}:\ V^{\mathrm{cont}}_{T+1}(s) := 0,\ \text{hence } Q_T = U_0(m\mid s_T)
\end{aligned}\;\longrightarrow\;\circlearrowleft}$$

Hierarchy: $\boxed{\text{global } \mathbb{E}[J_T] \supset \text{local } Q_{\lambda_t} \supset MV}$ — **L1 defines the optimization problem itself (stochastic sequential allocation with chance constraint); L4 defines one tractable action-selection approximation to it** ($m_t^* = \arg\max\,(U_0 + \beta\,\mathrm{CVU})$ is a **one-step, performance-only continuation approximation** of the global sequential allocation, not a Bellman-style exact solution — $V_{t+1}(s_{t+1})$ is replaced by $\max_{m'} U_0(m'\mid s_{t+1})$); $MV$ is an efficiency statistic. **$\mathrm{CVU}$ is used only for intermediate-decision continuation look-ahead, not as a terminal deployment reward: at stopping, the final objective evaluates only $J_{\mathrm{real}}(\pi_T)$ under $P_{\mathrm{deploy}}$**, so the terminal $Q_{\lambda_t}$ collapses to $U_0$.

$MV$ and $Q_{\lambda_t}$ **split duties, not interchangeable** — $MV$ answers "how efficient per unit resource cost," $Q_{\lambda_t}$ answers "worth doing after opportunity cost"; reading both prevents ratio-driven misprioritization while keeping cross-budget comparability. Closing: **sim-to-real is not the choice of a transfer technique; it is the continuous decision, under current belief, non-substitutable budgets, and real-evaluation feedback, of what the next intervention should be** — the article's spine. **The contribution is not a new optimization primitive; it is a redefinition of sim-to-real's decision unit** — from "pick a transfer method" to "pick the next intervention under current state and multi-resource constraints" — while reframing reality gap and simulator utility as policy- and evaluation-conditioned quantities. All symbols introduced above compress into one main chain, two closing fences, and one allocation stack:
$$\boxed{\begin{gathered}
\text{state } s_t \;\rightarrow\; \mathcal{M}_t^{\mathrm{feasible}}(s_t) \;\rightarrow\; \big(\mu_{\Delta J,t},\;\Delta C,\;\mathrm{CVU}\big) \;\rightarrow\; Q_{\lambda_t} \;\rightarrow\; m_t^* \;\rightarrow\; s_{t+1}\\[4pt]
\text{safety / budget define feasibility;}\\[-2pt]
MV\ \text{is only an efficiency diagnostic, not a decision rule.}
\end{gathered}}$$
**$\mu_t$ is intentionally not drawn on the main chain** — L1 optimizes the ideal **global adaptive allocation policy** $\{\mu_t\}_{t=1}^T$, while L4's argmax constructs a computable **approximate allocation policy** $\mu_t^Q(s_t) := m_t^* = \arg\max_m Q_{\lambda_t}(m \mid s_t)$; the two live at different layers (ideal vs. approximate), and drawing both on the chain would falsely read as "μ_t generates m_t, then Q picks m_t*," a spurious circularity. The article's allocation stack, top-down, forms seven layers:
$$\boxed{\begin{array}{rcl}
\text{diagnostic layer} &:& \Delta_k,\ S_k^{\mathrm{int}},\ \text{uncertainty}\\[2pt]
\downarrow &&\\[2pt]
\text{candidate construction} &:& m = (\text{role},\,\text{lens},\,\text{protocol}/\text{batch})\\[2pt]
\downarrow &&\\[2pt]
\text{value estimation} &:& (\mu_{\Delta J,t},\ \Delta C,\ \mathrm{CVU})\\[2pt]
\downarrow &&\\[2pt]
\text{feasibility} &:& P_{\mathrm{exec}},\ \text{budget}\\[2pt]
\downarrow &&\\[2pt]
\text{local decision} &:& Q_{\lambda_t}\\[2pt]
\downarrow &&\\[2pt]
\text{action} &:& m_t^*\\[2pt]
\downarrow &&\\[2pt]
\text{state transition} &:& s_{t+1}
\end{array}}$$
Readers only need to hold **five main objects** ($s_t$, $\mathcal{M}_t^{\mathrm{feasible}}$, $Q_{\lambda_t}$, $m_t^*$, $s_{t+1}$), **one feasibility rule** ($\mathcal{M}_t^{\mathrm{safe}} \cap \mathcal{M}_t^{\mathrm{budget}}$, with $P_{\mathrm{exec}} / P_{\mathrm{deploy}}$ as its two layers), and **the $MV$-as-efficiency-reading division of labor** — the whole formal framework closes on one page. $(\mu_{\Delta J,t}, \Delta C, \mathrm{CVU})$ are the three value inputs to $Q_{\lambda_t}$, and $m_t$ is only a generic action label — **neither is elevated to a first-class object**.

---

## References

The main works referenced in the text (all searchable via arXiv ID):

- Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World — Tobin et al., IROS 2017, arXiv:1703.06907
- Sim-to-Real Transfer of Robotic Control with Dynamics Randomization — Peng et al., ICRA 2018, arXiv:1710.06537
- Sim-to-Real: Learning Agile Locomotion For Quadruped Robots — Tan et al., RSS 2018, arXiv:1804.10332
- Learning Dexterous In-Hand Manipulation — Akkaya et al. (OpenAI), 2019, arXiv:1808.00177
- Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via RCAN — James et al., CVPR 2019, arXiv:1812.07252
- DiffTaichi: Differentiable Programming for Physical Simulation — Hu et al., ICLR 2020, arXiv:1910.00935
- Interactive Differentiable Simulation — Heiden et al., ICRA 2021, arXiv:1905.10706
- Residual Physics + SI for Sim-to-real on Buoyancy-Assisted Legged Robots — Sontakke et al., 2023, arXiv:2303.09597
- Sim-to-Real of Soft Robots with Learned Residual Physics — Gao et al., IEEE RA-L 2024, arXiv:2402.01086
- Dream to Control: Learning Behaviors by Latent Imagination — Hafner et al., ICLR 2020, arXiv:1912.01603
- DayDreamer: World Models for Physical Robot Learning — Hafner et al., CoRL 2022, arXiv:2206.14176
- TD-MPC2: Scalable, Robust World Models for Continuous Control — Hansen et al., ICLR 2024, arXiv:2310.16828
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., RSS 2025, arXiv:2503.24361
- A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies — Lei et al., arXiv 2026, arXiv:2604.13645

There is not yet a widely accepted cross-task quantitative comparison in sim-to-real saying "this method is stronger" — across tasks / hardware / fidelity ceilings conclusions can flip; the works above are more like "this method is workable for this kind of gap" samples than an extrapolatable ranking. The four-lens decomposition, the three-dimensional simulator-utility cut, the error-budget constrained-allocation formalization, and the definitions of $\hat S_k^{\mathrm{int}}$ and $MV$ are all **conceptual framework and the author's reading**: decision statistics estimated via sensitivity experiments / ablation / small-scale real evaluation, not analytically computable from the simulator; reading co-training as data mixture and the world model as model-source replacement is likewise not proven by controlled experiments.

---

*This piece continues the two-part "data problem for embodied AI" series: the first covered data sources and interfaces, the second covered the data-scaling framework; here the camera pans to sim-to-real, reframing it from "a pile of transfer tricks" into a closed-loop allocation problem with empirical marginal utility.*
