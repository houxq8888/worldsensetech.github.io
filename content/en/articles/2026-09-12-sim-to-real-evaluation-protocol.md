---
title: 'Embodied AI Sim-to-Real Methodology (III): Evaluation, Decision Matrix, and a Minimum Executable Protocol'
slug: "2026-09-12-sim-to-real-evaluation-protocol"
date: 2026-09-12
draft: false
categories: ["Embodied AI", "Training Methods"]
tags: ["Embodied AI", "Sim-to-Real", "Evaluation", "Sim Utility", "Ranking Correlation", "Selection Regret", "Allocation Protocol", "Stopping Rule", "Composition"]
description: 'Trilogy - Evaluation and Deployment. Sim fidelity has three non-interchangeable dimensions (prediction accuracy, ranking quality, decision quality); this piece provides the gap x modelability x real-budget decision matrix, three stopping rules, and a 6-step minimum executable Sim-to-Real Allocation Protocol. Theory spine in Part 1, method genealogy in Part 2.'
toc: true
related_articles:
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-11-sim-to-real-intervention-lenses
  - 2026-09-13-tactile-force-sensing
  - 2026-09-14-multimodal-fusion-interface
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
---


> **Recap** (from Part 1): This is the third piece of the Sim-to-Real Methodology trilogy. The core framework is established in [Part 1](/en/articles/2026-09-10-sim-to-real-methodology/) — reality gap is recast as a downstream discrepancy under the quadruple $(\pi,\; \mathcal{E}_{\mathrm{shared}},\; M_{\mathrm{sim}},\; M_{\mathrm{real}})$, the error budget becomes a state-conditioned sequential allocation, and the decision unit is the intervention action $m_t^* = \arg\max\, Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t)$. Notation: $s_t = (b_t, \pi_t, q_t, h_t)$, $\mathcal{M}_t^{\mathrm{feasible}}$ (feasible action set), $MV$ (efficiency readout, not decision rule), $\mathrm{CVU}$ (one-step continuation uplift). If you have not read Part 1, start there.

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
---

> **Previous (Part 2)**: [Four Intervention Lenses and Two Reformulation Routes](/en/articles/2026-09-11-sim-to-real-intervention-lenses/) -- SI / DR / DA / FT / WM / Co-training in detail.
>
> **Part 1**: [Sim-to-Real Methodology - Theory](/en/articles/2026-09-10-sim-to-real-methodology/) -- reality gap and allocation framework.

*This is the Evaluation & Protocol piece of the three-part Sim-to-Real Methodology series. Theory in Part 1, methods in Part 2.*
