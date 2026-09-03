---
title: "Robot Data Scaling: From Interaction Coverage to Marginal Data Value"
slug: "2026-09-09-robot-data-scaling"
date: 2026-09-09
draft: false
categories: ["Embodied Intelligence", "Training Methods"]
tags: ["Embodied Intelligence", "Robot Data", "Scaling Law", "Data Distribution", "Coverage", "Marginal Data Value", "Data Flywheel", "Offline RL", "Imitation Learning"]
description: "What robotics truly deserves to scale is not just trajectory count, but the effective coverage of the interaction distribution relative to a target evaluation distribution. This post gives a complete analytical framework — from interaction distribution, p_eval, and the support/density decomposition, through data utility and marginal data value, to the data flywheel and sequential data allocation."
toc: true
related_articles:
  - 2026-09-08-data-and-training-recipes
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
---

This is **Part 2** of the two-post series on "the data problem in embodied AI." [Part 1](/en/articles/2026-09-08-data-and-training-recipes/) took inventory of where robot data comes from, the data interfaces of different paradigms, the "data is a distribution, not a dataset" lens, the two paths along which training recipes act, and the four common sim-to-real tools. That post answered "where does data come from, and in what form does it enter the model"; this post answers the question the whole series is really about: **given a limited collection budget, how should robot data actually be scaled — what data should the next unit of budget add?**

To let this post stand on its own, let me first gather a few notations established in Part 1 (the detailed arguments live in Part 1):

> **Interaction distribution:** the trajectory distribution jointly determined by conditions such as task, scene, and embodiment in the training data, $p(\tau \mid task,\ scene,\ embodiment)$, written more rigorously as $p_D(\tau \mid c)$, where the subscript $D$ reminds us that it implicitly depends on the concrete collection policy and environment.
>
> **Training vs Evaluation:** data is first transformed by the recipe into the distribution the model actually sees, $p_{\mathrm{train}}(\tau) = T_R[p_{\mathrm{raw}}(\tau)]$; performance then depends on its relationship to the evaluation distribution $p_{\mathrm{eval}}$.
>
> **Quality ≠ Utility:** quality is a measurable property at the trajectory level; utility is the conditioned contribution those properties cash out to under a specific objective and a specific $p_{\mathrm{eval}}$, written $U(D \mid \mathcal{L},\ p_{\mathrm{eval}})$.
>
> **Recipe's two paths:** Path 1 changes the distribution ($p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$, already folded into $D_{\mathrm{effective}}$); Path 2 changes the optimization dynamics (lr schedule, optimizer, loss weighting, freezing, curriculum, and so on, which cannot be absorbed by $p_{\mathrm{train}}$).

With this notation in hand, we can go straight into scaling itself.

## Robot Data Scaling: Not Just "More Trajectories"

The LLM domain has accumulated a fairly mature scaling-law empirical framework (Kaplan et al., 2020, arXiv:2001.08361; Hoffmann et al., 2022, arXiv:2203.15556), describing how data, parameters, and compute jointly determine loss under specific compute-optimal / loss-scaling regimes — but this is not a unified "natural law," nor does it transfer directly to robotics. Does a comparable scaling law exist for robotics?

### Data Acquisition ≠ Data Scaling

Before talking about scaling, we need to separate two questions that are often conflated.

**Data acquisition** answers "where does data come from?" (teleoperation, simulation, autonomous exploration, and synthetic generation are all *acquisition methods* — what Part 1 discussed), while **Data scaling** answers "what data should I add next, per unit of budget?" (support expansion, density improvement, failure targeting, and embodiment expansion are *scaling strategies*). One is a generation mechanism, the other an allocation problem — no matter how strong your acquisition methods are, they do not automatically answer the scaling question. This post therefore shifts its center of gravity from "where data comes from" to "what data is worth continuing to add," which is exactly what the framework below tries to answer.

First, an important clarification: **the formulas below are not strict scaling laws, but a conceptual decomposition for describing robot data's effective scale.** Robot data scale can be decomposed into at least three layers:

**Data volume:** $N_{\text{steps}}$ (total interaction steps)

**Distribution dimensions:** $task, scene, embodiment, \text{behavioral state}, action$ (distribution dimensions)

**Data quality:** $Q$ (data quality)

The "state" dimension needs clarification here: in line with the Part 1 $o_t \neq s_t$ discussion, we do not mean explicitly annotated environment state (the true $s$ is often not directly obtainable), but rather **behavioral-state coverage / state-space coverage** — the (often latent or inferred) behavioral-state distribution the model actually visits during training. Framed this way, it does not conflict conceptually with the Part 1 partial-observability discussion.

### Distribution ≠ Coverage: Introducing an Evaluation Distribution

Before we start talking about coverage, we have to patch in a concept that has been implicit all along. In Part 1 we defined the interaction distribution as

$$p(\tau \mid task,\ scene,\ embodiment)$$

This is a **probability distribution**. But when we discuss scaling, what we actually care about are several *properties* of that distribution — coverage, diversity, support, density — and they are not the same thing:

$$\boxed{Distribution \neq Coverage}$$

Worse, the word "coverage" on its own **has no reference frame**. To say "the training data covers a lot" is empty until we answer "covers what?" — and that immediately forces us to introduce an evaluation distribution.

So far we have only written down the training-side distribution:

$$p_{\mathrm{train}}(\tau)$$

But what actually determines performance is its relationship to the evaluation distribution:

$$p_{\mathrm{eval}}(\tau)$$

```text
training distribution
        ↓
 p_train(τ)

          ↕ mismatch / coverage

evaluation distribution
        ↓
 p_eval(τ)
```

Two limits are worth stating once here, so they need not be repeated later. First, abstracting $p_{\mathrm{eval}}$ as a **trajectory-level** distribution is only for unified notation; in a concrete benchmark it is often more naturally defined over **contexts** $c=(task, scene, initial\ state, \ldots)$ — written $p_{\mathrm{eval}}(c)$ — and only *induces* a trajectory distribution once the policy acts within each context. Second, writing $p_{\mathrm{eval}}$ as a **fixed** reference distribution is mainly to give coverage a coordinate system; in **closed-loop / online RL**, the state / trajectory distribution actually visited at evaluation time is itself shaped by the current policy (i.e. $d^{\pi_\theta}$), so strictly speaking $p_{\mathrm{eval}}$ may also be **policy-dependent**. This article absorbs that feedback effect into the evaluation distribution rather than developing a full policy-induced distribution analysis.

Once $p_{\mathrm{eval}}$ is written down explicitly, many previously fuzzy intuitions become sharp. Consider two datasets:

- **Dataset A:** covers 100 tasks × 100 scenes × 10 embodiments, but with only a handful of trajectories in each cell.
- **Dataset B:** only 10 tasks × 10 scenes × 1 embodiment, but with hundreds of thousands of high-quality trajectories in each cell.

Which is better? The answer is: **it depends on $p_{\mathrm{eval}}$, model capacity, the task's relative need for precision vs coverage, and the optimization budget.** If evaluation concentrates on a small set of high-precision manipulation tasks, B likely wins; if evaluation is open-world multi-task generalization, only A has a chance. So the rigorous claim is not "coverage determines scaling" but rather:

> **Performance depends on how well $p_{\mathrm{train}}$ covers the relevant regions of $p_{\mathrm{eval}}$, together with sufficient sampling density and data quality in those regions.**

Written as a formula:

$$\Delta Performance \approx f\big(\Delta p_{\mathrm{train}},\ p_{\mathrm{eval}}\big)$$

There is one more detail worth spelling out: **coverage itself is not a natural scalar, nor an absolute property of the training distribution.** Suppose Dataset A has high task coverage but low scene coverage, while Dataset B is the reverse — which one has "higher coverage"? Without a reference frame, the question simply cannot be answered. So the more accurate way to write it is as a **relation**:

$$\text{Coverage} = C\big(p_{\mathrm{train}},\ p_{\mathrm{eval}}\big),\qquad \text{rather than}\quad C(p_{\mathrm{train}})$$

That is, what we really care about has never been some "coverage score" a dataset carries on its own, but the degree to which $p_{\mathrm{train}}$ covers a *specified* $p_{\mathrm{eval}}$. Making this step explicit is also what turns the utility definition just below from a notation that seems to appear out of nowhere into a result derived naturally along the line "coverage is a relation."

One line should be drawn immediately: **coverage, density, and distribution similarity are three different things, and cannot be swept together under the single word "coverage."** The $C(p_{\mathrm{train}},p_{\mathrm{eval}})$ above actually points at three independent quantities — **support coverage** (have we seen the evaluation-relevant region yet, a roughly 0/1 question), **density** (how much have we sampled within it, an intensity question), and **distribution similarity** (how similar the two distributions are overall, typically given by some distance metric). They rank Dataset A / B differently: measured by support coverage, A — which fills the whole evaluation support — clearly wins; measured by an overall distance such as $D_{KL}(p_{\mathrm{eval}}\,\|\,p_{\mathrm{train}})$, i.e. distribution similarity, B — which densely covers 80% of the region — may actually score lower; and if the evaluation metric is especially sensitive to one core region, high-density B may again be better. So this article's support / density decomposition is **not equivalent to "finding a single distribution distance and minimizing it"** — it deliberately treats "have we seen it," "how much did we sample," and "do the two look alike" as three separate questions, and only after separating them can the budget later be **allocated** to whichever one most needs filling.

This also tightens the Part 1 utility definition by one notch: **data utility is not only objective-conditioned — it is evaluation-conditioned.**

$$\boxed{U(D \mid \mathcal{L},\ p_{\mathrm{eval}})}$$

Here is a very simple example. Suppose the evaluation is "grasp a cup under varying lighting conditions in a kitchen":

- A new cup → potentially valuable (expands object support)
- A new kitchen → valuable (expands scene support)
- A new lighting condition → valuable (expands visual-condition support)
- A new robot embodiment → not necessarily valuable (a morphology change does not by itself change the evaluation)
- A new task like "fold laundry" → nearly worthless (falls outside the support of $p_{\mathrm{eval}}$)

The conclusion is direct: **"new data" has no absolute value — only value relative to an evaluation distribution.** This is the crucial patch that rescues the whole framework from naive "more data is better" intuitions.

### What Does Coverage Actually Cover: Different Dimensions Serve Different Generalization

The phrase "increase distribution coverage" is still too vague if not further decomposed. A more accurate statement is: **different distribution dimensions are responsible for different generalization problems, and they cannot be lumped together.**

They can be written respectively as conditional distributions (note: these conditional distributions are only a **lens for analyzing coverage**, not a complete trajectory generative factorization — in reality $s$ depends on history, $a$ depends on the policy / observation / embodiment, embodiment in turn shapes the action space, and $s$ and $a$ co-determine each other over time):

$$p(task)\quad(\text{task semantic space})$$

$$p(scene \mid task)\quad(\text{environment conditions})$$

$$p(s \mid task, scene)\quad(\text{behavioral states visited during task execution})$$

$$p(a \mid s)\quad(\text{actions the behavior policy actually takes at a given state})$$

$$p(embodiment)\quad(\text{robot morphology and action space})$$

The formula $p(a \mid s)$ needs an explicit caveat here, otherwise it is very easy to misread. In imitation-learning / offline datasets, what we actually observe is the **behavior-policy distribution** $p_D(a \mid s)$ — it does not necessarily cover all *feasible* actions at a given state; it may concentrate on the successful / slightly-suboptimal sliver, while catastrophic and recovery actions are severely undersampled. So strictly speaking this dimension is not "action coverage" but **behavior / intervention coverage**: it measures how many kinds of *actually executed behaviors / interventions* we have seen in the data, not "all physically executable actions." A narrow distribution may be tolerable for imitation learning, but for world models, offline RL, and recovery policies, narrow behavior coverage directly limits what the model can learn about "what would happen if a different action were taken."

One more caveat closes the loop with the Part 1 $o_t \neq s_t$: in real robot data, $s$ is often **not directly observable** at all, so $p_D(a \mid s)$ is more accurately read as a behavior distribution conditioned on a *latent / inferred* behavioral state. In practice it can only be approximated through observation history, proprioception, or a learned representation.

$$\text{behavior / intervention coverage: } p_D(a \mid s)\ \text{not}\ p_{\mathrm{feasible}}(a \mid s)$$

What this dimension really pins down is the concept the whole article keeps circling back to: **the mismatch that actually hurts performance is, at bottom, distribution shift / covariate shift** — the policy, once deployed, steers itself into states its training distribution never covered. In offline RL this takes a sharper form: if the evaluation policy visits $(s,a)$ regions the behavior dataset barely covers, the value estimate is forced to extrapolate, and support mismatch becomes the dominant failure mode. So the operative question is never "is the dataset diverse," but "is the *evaluation-relevant state-action support* adequately covered by the behavior distribution" — which connects straight back to $p_D(a \mid s)$ above.

Mapping them to the generalization abilities they are responsible for:

```
Interaction Distribution
│
├── Task      → semantic generalization
├── Scene     → visual / environment generalization
├── State     → behavioral-state coverage
├── Behavior  → behavior / intervention coverage (i.e., p_D(a|s))
└── Embodiment→ morphology / action-space transfer
```

**Failure / recovery**, on the other hand, should not be listed as a parallel axis alongside task, scene, and embodiment — it is better modeled as a **trajectory subset** inside the base interaction distribution that carries special learning value:

```
Base Interaction Distribution
   │
   ├── successful trajectories
   ├── failure trajectories
   └── recovery trajectories
```

In other words, failure is not a new "distribution dimension"; it is a subset carved out of the same interaction distribution by a posterior label $m=h(\tau)$. Its value was already discussed under Part 1's Data Utility / action-conditioned negative outcome information; here we are simply putting it in the right slot of the taxonomy.

In other words, "covering more" must always ask "covering more along which dimension, and to gain which kind of generalization." Increasing scene diversity buys visual/environmental robustness, increasing task diversity buys semantic generalization, and widening behavior coverage buys "having seen more distinct actions actually executed at the same state" — stuffing all of these vaguely into a single "diversity" keeps the scaling discussion at the empirical level of "diversity matters."

Therefore, robotics' scaling law may not be:

$$Performance = f(N)$$

but rather:

$$D_{\mathrm{effective}} = f(N,\;Coverage,\;Q,\;Relevance)$$

$$Performance = g(D_{\mathrm{effective}},\;Capacity,\;Compute,\;Recipe)$$

We deliberately **no longer list Diversity as a separate term**: difference between samples does not create value on its own — it only matters once that difference turns into an expansion of evaluation-relevant support or an improvement in density, at which point it acts *through* $Coverage$. Otherwise more diversity is just "more," not "coverage." Treating Diversity as a multiplier on par with Coverage inside $D_{\mathrm{effective}}$ would tempt readers into a "more diverse is always better" reading and dodge the real question — diverse *where*, and *enough for the evaluation or not*. So the decomposition below uses Coverage throughout (split into its support and density facets) to carry the meaning previously hung on Diversity.

Here $Capacity$ is deliberately moved out of $D_{\mathrm{effective}}$ and kept only in $Performance$: otherwise capacity would influence the outcome through two paths at once — via effective data scale and via the performance function — making the decomposition muddy. Effective data scale should describe "how effective the data itself is," while capacity, compute, and recipe describe "how much of that effective data the model can convert into performance."

This means: **what robotics truly needs to scale is not just data volume, but effective data scale — i.e., effective coverage of the interaction distribution.**

For a more intuitive view, $D_{\mathrm{effective}}$ can be further written as a conceptual product decomposition:

$$D_{\mathrm{effective}} \propto N_{\mathrm{eff}} \cdot \eta_{coverage} \cdot \eta_{quality} \cdot \eta_{relevance}$$

We deliberately write $\propto$ rather than $=$ here: an equality form would imply "each trajectory contributes at most 1 unit of information," which is not realistic — a long, rich trajectory can carry far more information than a short one. If we wanted to be fully rigorous, an information-theoretic formulation like $I(\tau;\theta)$ would be more natural; this article deliberately stops short of that step in order to keep the conceptual decomposition intuitive. The $\eta$ factors are **heuristic effectiveness coefficients**, used to express the intuition that "effective sample size is jointly modulated by several efficiency factors," rather than a directly measurable formula. This turns the question "1 million trajectories" into "of these 1 million, how many are actually new, relevant, effective interaction information" — which is closer to what this article really wants to express.

Note that $N_{\mathrm{eff}}$ here is itself **not a raw trajectory count**, but an effective sample count after adjusting for correlation, and

$$N_{\mathrm{eff}} \leq N$$

The reason is concrete: robot data has a special problem that distinguishes it from a static iid dataset — **there is strong temporal correlation within a trajectory, and trajectories share many factors with one another.** A 200-timestep "grasp the cup" trajectory is not 200 independent samples; and 1000 trajectories that all come from the same operator, the same kitchen, the same cup, the same reset distribution, and the same strategy may have an effective sample size far below 1000. As repeated sampling grows, $N_{\mathrm{eff}}$ saturates noticeably faster than the raw count $N$ — which is exactly why "raw trajectory count" becomes an increasingly unreliable metric. In one sentence:

> **100,000 highly correlated timesteps do not equal 100,000 independent units of information.**

And the attrition is not single-layer but multi-layer: raw counts lose value at every level of abstraction.

```text
100,000 timesteps
      ↓
 10,000 trajectories
      ↓
  1,000 unique scene–object configurations
      ↓
    100 meaningful behavioral regions
      ↓
     10 truly distinct failure modes
```

So robot data scaling contends with **both sample redundancy and distribution redundancy** at once: temporal correlation erodes timestep-level independence, shared scenes and operators erode trajectory-level independence, and repeated tasks and repeated failure modes erode distribution-level novelty. This is precisely why $N_{\mathrm{eff}}$ has to stand as its own quantity and cannot simply be replaced by $N$.

(Statistically, there is a classic intuition for folding temporal correlation into an effective sample size, of the form $N_{\mathrm{eff}} \approx N / (1 + 2\sum_k \rho_k)$; this article deliberately keeps it out of the main text so as not to drag the discussion into the technical details of "time-series ESS.")

It should be emphasized that **effective data scale and model capacity are not independent**, but this coupling belongs inside $g(\cdot)$ rather than being stuffed into $D_{\mathrm{effective}}$: a sufficiently wide distribution coverage can only be fully exploited when the model has enough capacity. When model capacity is small, blindly enlarging the covered region may yield limited or even negative returns; only when capacity is sufficient can the same wide-coverage data be converted into stronger generalization (and by "diverse data" here we really mean "data covering a wider region" — diversity counts only once it turns into coverage). So $Performance$ is jointly determined by $D_{\mathrm{effective}}$, $Capacity$, $Compute$, and $Recipe$, rather than being a function of any single variable.

LLMs can roughly ask "how many tokens do I have?"; robotics should rather ask "how many tasks, states, environments, actions, failure modes, and embodiments have I covered?"

```
Robot Data Scaling ≠ More Trajectories

Effective Data Scale = f(Volume, Distribution Coverage, Quality)
```

This is a hypothesis worth testing: **scaling on interaction distribution (rather than pure trajectory count) may be the more effective scaling direction for robotics.** Robotics does not yet have a single universally-accepted scaling law like LLMs, but there is empirical work on data scale worth referencing — for example, the study on data scaling in imitation learning (Lin et al., 2024, *Data Scaling Laws in Imitation Learning for Robotic Manipulation*, arXiv:2410.18647) finds that policy generalization performance follows an approximate power law in the **number of environments and objects**, and that environment/object diversity matters more than merely adding trajectory count — once the number of demonstrations per environment/object exceeds a certain threshold, the returns from piling on further demonstrations saturate rapidly.

To make the positioning of this hypothesis clearer, the article's logic can be layered as follows:

> **Known:** data volume, data quality, and task diversity all affect robot learning performance.
>
> **Unknown:** under fixed compute and model capacity, which kind of distribution expansion is most effective?
>
> **This article's hypothesis:** effective interaction-distribution coverage explains data scaling better than raw trajectory count.

### Support Scaling vs Density Scaling

Simply saying "duplicate data has diminishing returns while diverse data yields more" can easily be shot down by counterexamples. Consider a high-precision manipulation task, such as precisely inserting a very small connector — here a large number of highly similar trajectories can be very valuable. But what they teach is *not* merely "estimate the same distribution density a bit more precisely." Plugging and unplugging the same connector ten thousand times is also how the model comes to learn tiny contact variations, force response, actuator noise, timing, micro-corrections, and where the failure boundary actually lies — alongside variance reduction and optimization stability. For such a task, $10000$ highly similar but high-quality trajectories may be more useful than $1000$ highly diverse ones.

So a more accurate framework splits the marginal value of new data into two parts:

$$\Delta U(D) = \Delta U_{\text{support}} + \Delta U_{\text{density}}$$

corresponding to the two basic modes of robot data scaling:

```
Support scaling (expand the support of the distribution)
  new task
  new object
  new scene
  new failure mode
  new embodiment (only counts when it expands an evaluation-relevant region)
  → seeing things never seen before

Density scaling (raise sampling density in already-covered regions)
  more trajectories
  more repetitions
  more demonstrations
  → in regions already known, sharpen estimation, reduce variance, and learn contact / force / noise / timing structure
```

**Support scaling** answers "have I seen new regions of the distribution?"; **density scaling** answers "within regions I already know, have I sampled densely enough?" Density scaling is often doing several things at once — improving local distribution estimation, but also reducing variance, hardening robustness, and stabilizing optimization — so it should not be collapsed into "estimation" alone. Both are valuable, but they serve different generalization goals — high-precision, contact-rich tasks often need density scaling more, while open-ended, multi-scene tasks need support scaling more.

But we must add a $p_{\mathrm{eval}}$-relative qualifier to support scaling here: **support expansion is not valuable in itself; only support expansion that both falls inside the evaluation-relevant region and is of sufficient quality and learnability can plausibly yield positive marginal utility.** In other words, intersecting the evaluation-relevant support is a **necessary but not sufficient condition** — a single trajectory that lands inside that region but is extremely noisy may have utility near zero, or even negative. So rather than writing a biconditional, it is more accurate to express it as a relation modulated jointly by several factors:

$$\Delta U_{\text{support}} = f\Big(\underbrace{\Delta \operatorname{Supp}_{\mathrm{eval}}}_{\text{expands relevant support?}},\ \underbrace{Q}_{\text{quality}},\ \underbrace{R}_{\text{relevance}},\ \underbrace{\text{Learnability}}_{\text{learnable?}}\Big)$$

The decision chain should be:

```text
new data
   ↓
does it expand training support?
   ↓
does it expand evaluation-relevant support?
   ↓
does it improve performance?
```

Along this chain, **a new embodiment does not automatically count as positive support scaling** — it is only genuinely valuable when its morphology, action semantics, and control frequency happen to expand a $p_{\mathrm{eval}}$-relevant region (for example, when the evaluation requires cross-embodiment generalization). Otherwise, a new embodiment that closely resembles existing ones only enlarges the support in directions the evaluation does not care about. The same goes for "new tasks": a new task that lies entirely outside $p_{\mathrm{eval}}$ (e.g., adding a large pile of laundry-folding data when the target is kitchen grasping) expands the support but yields near-zero utility.

This makes the utility definition from earlier explicit again:

$$\Delta U(D) = \Delta U\big(D \mid \mathcal{L},\ p_{\mathrm{eval}}\big)$$

The support-vs-density tradeoff is essentially the judgment of whether the current $p_{\mathrm{train}}$ is *coverage-deficient* or *density-deficient* relative to $p_{\mathrm{eval}}$.

This leads directly to a key judgment: **deciding when to support-scale and when to density-scale is itself a core training-recipe question** (echoing Part 1's $p_{\mathrm{train}}(\tau) = T_R[p_{\mathrm{raw}}(\tau)]$ — the recipe determines which regions of the raw data are amplified and which are compressed).

### Testable Predictions

If this hypothesis holds, then under fixed training compute and model scale, the following testable predictions can be made:

- The marginal value of newly added data depends on whether it expands the *evaluation-relevant* support or improves estimation within already-covered support — a simple "duplicate vs diverse" framing is not enough to predict returns; it must be combined with the task's relative need for precision vs coverage;
- Adding new tasks / scenes / embodiments that expand the support of $p_{\mathrm{eval}}$-relevant regions is expected to have higher marginal value than simply repeating existing trajectories (the keyword is *expand the evaluation-relevant support*, not "new = good" — if a new embodiment's morphology and action semantics closely resemble existing ones, or if it falls entirely outside $p_{\mathrm{eval}}$, its marginal value may be very low);
- Targeted data addressing failure modes should be more effective than randomly adding data, provided those failures fall inside the relevant region of $p_{\mathrm{eval}}$;
- Changes to data mixture and sampling recipes should produce reproducible performance differences, and the portion attributable to Path 1 (distribution transformation) vs Path 2 (optimization dynamics) should in principle be separately ablatable;
- Any quantitative claim about marginal value must be bound to an explicitly declared $p_{\mathrm{eval}}$ — discussing "whether data is useful" without an evaluation distribution is unfalsifiable;
- At equal collection cost, targeted data collection guided by an estimated $MV$ should outperform random addition, and the advantage should grow as the $MV$ estimator becomes more accurate (better uncertainty calibration, richer failure statistics).

These predictions are in principle experimentally verifiable, rather than remaining at the level of "data matters" as empirical observation.

## What Does This Mean?

If we connect the threads from [the industry landscape](/en/articles/2026-09-06-embodied-ai-landscape/), [the world model series](/en/articles/2026-09-01-world-model-h2-review/), [the VLA series](/en/articles/2026-09-03-vla-deep-dive/), and [RSSM evolution](/en/articles/2026-09-04-rssm-beyond/), we find that each established one of the concepts — a "prediction interface," a "semantics + action" interface, the data requirements of different latent dynamics, and the fact that data is becoming a key differentiator. Taken together, the two posts are really trying to condense into the following five pillars:

$$\boxed{Interaction\ Distribution:\ p(\tau \mid task,\ scene,\ embodiment)}$$

$$\boxed{Training\ vs.\ Evaluation:\ p_{\mathrm{train}}(\tau)\ \leftrightarrow\ p_{\mathrm{eval}}(\tau)}$$

$$\boxed{Support\ vs.\ Density\ Scaling}$$

$$\boxed{Data\ Utility:\ U(D \mid \mathcal{L},\ p_{\mathrm{eval}})}$$

$$\boxed{Recipe:\ p_{\mathrm{raw}}(\tau) \xrightarrow{T_R} p_{\mathrm{train}}(\tau)\ \text{+ Optimization Dynamics}}$$

Taken together, these five are really trying to say one thing: **the basic unit of robot scaling may not be the trajectory, but the effective coverage of the interaction distribution relative to a target evaluation distribution.** Moving from "robot data is complicated" to "what actually counts as effective robot data scaling" is precisely the step this two-post series is trying to take.

### Marginal Data Value: Condensing the Whole Framework into an Actionable Concept

Every concept across the two posts — interaction distribution, $p_{\mathrm{eval}}$, support/density, utility, recipe — is ultimately answering the same question: **is the next batch of data worth collecting?** This question deserves a formal name. But first, a baseline must be added: the value of a new batch $D'$ only ever makes sense *given what data you already have*, $D$ — so $\Delta Performance$ should be written explicitly as an increment relative to $D$:

$$MV(D';\,D) \;=\; \frac{Performance(D \cup D') - Performance(D)}{Cost(D')}$$

For a single trajectory we can also write:

$$MV(\tau;\,D) \;=\; \frac{Performance(D \cup \{\tau\}) - Performance(D)}{Cost(\tau)}$$

The "$D$" argument looks like a notational detail, but it actually encodes the article's most core distributional argument right into the definition: **the value of a batch of data depends on the data you already have.**

With this in hand, the whole series' thesis condenses into a single sentence:

> **The core question of robot data scaling is not how to maximize data volume, but how to maximize marginal data value.**

This sentence is easier to remember than "effective interaction-distribution coverage," and closer to engineering practice — because volume is a quantity one can push blindly, while $MV$ forces you to answer "relative to the current $p_{\mathrm{train}}$ and $p_{\mathrm{eval}}$, what exactly does this batch fill in, and at what cost?"

And from the baseline-carrying $MV(D';D)$ notation, one can read off what may be the article's single most central insight:

$$MV(D';\,D_t) \;\neq\; MV(D';\,D_{t+1})$$

**Data value is state-dependent.** The same trajectory may be very valuable in the early, data-scarce phase, yet nearly worthless once the relevant regions of the distribution have already been filled in. This is precisely the root reason why "a dataset's quality cannot be permanently defined" — good data was never absolutely "good data," but "**data with high marginal utility under the current training state and evaluation gap.**" It is here that Data Utility → Marginal Data Value → Data Flywheel finally close into a loop.

The natural consequence is that the optimal collection policy cannot be a *static* one. If $MV(D';D_t) \neq MV(D';D_{t+1})$, then which data to collect next must itself depend on the current dataset, the evaluation target, and the model's current parameters:

$$D'_t = \pi_{\mathrm{data}}(D_t,\ p_{\mathrm{eval}},\ \theta_t)$$

The flywheel is therefore not simply "collect data → train → collect again," but *learning a data-collection policy that keeps changing as the state changes.* Put as a single line: **robot data scaling is not a static dataset construction problem; it is a sequential data allocation problem.**

### From Scaling Hypothesis to Data Flywheel

What this post wants to say is: **data and training recipes may be becoming embodied AI's most underestimated competitive advantage.**

Model architectures can be disseminated through papers and open-source code; simulation platforms are being standardized by a few players; but **high-quality robot interaction data, effective data curation processes, and repeatedly refined training recipes — these are difficult to fully transmit through a single paper.**

That said, we should more carefully distinguish "advantage" from "moat." Taken alone, any single item may not constitute a real moat: data can be purchased, teleoperation infrastructure can be replicated, training recipes can potentially be reverse-engineered, foundation model capabilities can transfer, and synthetic data may even lower the data barrier itself. So equating "having more data" directly with "having a moat" is not rigorous.

What is genuinely harder to replicate may be closing the whole chain into a **data flywheel**:

$$Data\ Collection \rightarrow Curation \rightarrow Evaluation \rightarrow Training \rightarrow Deployment$$

$$Deployment \rightarrow Failure \rightarrow Data \rightarrow Training \rightarrow Better\ Policy \rightarrow Deployment$$

That is, deployment produces real failures, failures flow back as new targeted data, data drives better policies after curation, which then enter the next round of deployment. Once this loop starts turning, competitors can hardly catch up by merely copying one isolated link — **the moat comes from the flywheel turning, not from any static pile of data.**

But if we stop here, the flywheel is still just an "engineering strategy," disconnected from the scaling theory above. With $MV$ in hand, we can rewrite it as an explicit targeted-collection rule:

$$D_{t+1} = D_t + D_{\text{targeted}}$$

$$D_{\text{targeted}} = \operatorname*{argmax}_{D'}\ MV(D';\,D_t) \;=\; \operatorname*{argmax}_{D'}\ \frac{Performance(D_t \cup D') - Performance(D_t)}{Cost(D')}$$

One clarification is needed: the $\Delta Performance$ here is **not assumed to be a directly readable oracle**. In practice it is typically estimated through evaluation on a proxy distribution, failure statistics, model uncertainty estimation, or offline-RL counterfactual proxy metrics. Making this explicit is what turns the formula from a pretty slogan into an actual research direction — **targeted data collection is fundamentally an estimation + optimization problem, not an oracle-style argmax.**

Pushing one level deeper: what a real system optimizes is never $MV$ itself but an estimate $\widehat{MV} = MV + \epsilon$, where $\epsilon$ comes from the finite evaluation set, noisy failure labels, imperfect uncertainty estimates, simulator bias, offline-proxy bias, and training variance. This quietly reframes the whole loop — **choosing which data to collect next is itself an active-learning / decision-making-under-uncertainty problem**: a better $\widehat{MV}$ yields better-targeted data, which in turn yields a better $\widehat{MV}$. The flywheel is thus not just accumulating data, but progressively sharpening its own estimate of where data is worth collecting.

The flywheel thereby ties directly back to **effective data scale**: the point of the loop turning is not that $N$ grows, but that each round prioritizes filling the gap in $p(\tau \mid task, scene, embodiment)$ that has the highest utility-per-cost relative to $p_{\mathrm{eval}}$. In other words, **the strongest data flywheel is not "keep collecting data," but "keep discovering where the current distribution falls short of the evaluation, and refill in a targeted way."**

### The Capstone Flow of the Whole Article

At this point we can name the series' real subject: on the surface it talks about "data scaling," but what it is fundamentally about is **evaluation-aware distribution allocation under a limited data budget** — not passively "aligning" $p_{\mathrm{train}}$ to some fixed $p_{\mathrm{eval}}$, but actively allocating a limited collection budget, round by round, to the support and density gaps in $p_{\mathrm{train}}$ that $p_{\mathrm{eval}}$ exposes, spending each unit of budget wherever marginal data value is highest. Compressing the entire analytical framework into a single diagram, it closes as follows — note that $p_{\mathrm{eval}}$ sits at the top, as **the target coordinate system of the whole loop**:

```text
                              p_eval
                                ▲
                                │  gap
                                │
   Raw Interaction Data         │
          │                     │
          ▼                     │
 p_raw(τ | task, scene, embodiment)
          │                     │
          │  Training Recipe    │
          │  (Path 1: dist. transform; Path 2: optimization dynamics)
          ▼                     │
 p_train(τ | task, scene, embodiment)
          │                     │
   ┌──────┴──────┐              │
   ▼             ▼              │
 Support      Density           │
 Coverage     Scaling           │
   │             │              │
   └──────┬──────┘              │
          ▼                     │
   Data Utility / MV            │
  U(D | L, p_eval)              │
          │                     │
          ▼                     │
  Effective Data Scale          │
          │                     │
          ▼                     │
  Performance / Generalization  │
          │                     │
          ▼                     │
      Evaluation ───────────────┘
          │
          ▼
  Failure / Gap Analysis
          │
          ▼
  Targeted Data Collection
          │
          └──────────────→  p_raw' / p_train^new
```

Written as formulas, the closed loop this version ultimately wants to leave behind is:

$$\boxed{p_{\mathrm{raw}} \xrightarrow{\ Recipe\ } p_{\mathrm{train}} \xrightarrow{\ Coverage\ /\ Density\ } U(D \mid \mathcal{L},\ p_{\mathrm{eval}}) \longrightarrow Performance}$$

$$\boxed{Performance \longrightarrow Evaluation\ Gap \longrightarrow Targeted\ Data \longrightarrow p_{\mathrm{train}}^{\,\mathrm{new}}}$$

At this point, **the data flywheel is no longer just an industry-level judgment — it becomes a natural corollary of the scaling hypothesis above**: if performance is determined by how well $p_{\mathrm{train}}$ covers $p_{\mathrm{eval}}$ (together with density in those regions), then of course the best next batch of data should come from the gap that evaluation just exposed.

And the core question here is therefore not "who has more data," but "who can keep $p_{\mathrm{train}}$ expanding continuously and directionally along $p_{\mathrm{eval}}$" — and "who can keep $MV$ better estimated after every round of deployment."

## References

The main works referenced in the text are listed below (all searchable via arXiv ID):

- Scaling Laws for Neural Language Models — Kaplan et al., 2020, arXiv:2001.08361
- Training Compute-Optimal Large Language Models (Chinchilla) — Hoffmann et al., 2022, arXiv:2203.15556
- Data Scaling Laws in Imitation Learning for Robotic Manipulation — Lin et al., 2024, arXiv:2410.18647

Directly relevant to this post's "data / distribution is what matters" thesis is also the set of dataset-focused empirical works listed at the end of [Part 1's references](/en/articles/2026-09-08-data-and-training-recipes/) (DROID, SCIZOR, Consistency, Compositional, Sim-and-Real Co-Training, and so on).

Note that robotics does not yet have a single universally-accepted scaling law comparable to that of LLMs; the effective-data-scale framework in this post is a conceptual decomposition and a testable hypothesis, not an established conclusion. The data-side works above provide scattered empirical support, not yet a full quantitative validation of that hypothesis.

---

*This is Part 2 of the two-post series on "the data problem in embodied AI" — Part 1 covers data sources, interfaces, and training recipes; this post covers what actually counts as effective data scaling. The next article may discuss sim-to-real methodology in detail.*
