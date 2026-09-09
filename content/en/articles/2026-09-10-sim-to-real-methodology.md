---
title: 'Embodied AI Sim-to-Real Methodology (I): Treating Sim-to-Real as an Error-Budget Allocation'
slug: "2026-09-10-sim-to-real-methodology"
date: 2026-09-10
draft: false
categories: ["Embodied AI", "Training Methods"]
tags: ["Embodied AI", "Sim-to-Real", "Reality Gap", "Error-Budget Allocation", "Sequential Allocation", "Policy-conditioned Mismatch", "Domain Randomization", "System Identification", "World Model", "Domain Adaptation"]
description: 'Trilogy - Theory. Sim-to-real is not a single transfer trick but a closed-loop resource allocation. This piece recasts reality gap as a policy-conditioned multi-source mismatch, formalizes error-budget allocation as an estimable iterative decision framework (L1-L5 spine), and defines the formal allocation structure for "where to invest the next unit of budget." Method genealogy in Part 2, evaluation and protocol in Part 3.'
toc: true
related_articles:
  - 2026-09-11-sim-to-real-intervention-lenses
  - 2026-09-12-sim-to-real-evaluation-protocol
  - 2026-09-13-tactile-force-sensing
  - 2026-09-14-multimodal-fusion-interface
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
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
---

> **Next (Part 2)**: [Sim-to-Real Methodology (II): Four Intervention Lenses and Two Reformulation Routes](/en/articles/2026-09-11-sim-to-real-intervention-lenses/) -- SI / DR / DA / FT / World Model / Co-training unpacked.
>
> **Part 3**: [Evaluation, Decision Matrix, and Protocol](/en/articles/2026-09-12-sim-to-real-evaluation-protocol/) -- three evidence levels, sim utility tripartition, 6-step executable protocol.

*This is the Theory piece of the three-part Sim-to-Real Methodology series. Method genealogy is in Part 2, evaluation and protocol in Part 3. The trilogy continues the Data Problem articles (upper and lower).*
