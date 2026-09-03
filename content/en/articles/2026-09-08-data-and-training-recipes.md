---
title: "The Data Problem in Embodied AI: As Mainstream Paradigms Come Into Focus, What Determines Performance?"
slug: "2026-09-08-data-and-training-recipes"
date: 2026-09-08
draft: false
categories: ["Embodied Intelligence", "Training Methods"]
tags: ["Embodied Intelligence", "Robot Data", "Training Recipe", "Teleoperation", "Synthetic Data", "Sim-to-Real", "Data Curation", "VLA", "World Model", "Scaling Law"]
description: "As foundational paradigms like VLA and world models are converging toward clearer mainstream approaches, data distribution, data quality, and training recipes are increasingly becoming important variables determining robot performance. But robot data is not simply 'the more the better' — what robotics truly needs to scale is not just trajectory count, but effective interaction coverage relative to a target evaluation distribution."
toc: true
related_articles:
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-07-vla-world-models
  - 2026-09-05-vla-pi-family
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
---

In [the previous industry landscape article](/en/articles/2026-09-06-embodied-ai-landscape/), I mentioned an increasingly obvious trend: pure model architecture differences are becoming less likely to form decisive advantages, while the importance of data scale, data diversity, and training recipes is rising.

This article wants to expand on that problem. But first, a clarification: this is not a survey of "what kinds of robot data exist," but an attempt to propose an analytical framework for robot scaling. Its core hypothesis is:

$$Performance \neq f(\#trajectory)$$

$$Performance = f(interaction\ distribution,\ data\ quality,\ recipe)$$

That is: **as foundational paradigms like VLA and world models converge toward a few mainstream routes (although concrete architectures are still evolving rapidly — diffusion / flow / autoregressive action heads, latent vs video world models, and action representations are all still unsettled), the difficulty of gaining a stable performance edge from local architecture innovation alone may be rising. What increasingly determines performance is the interaction distribution the model sees, the quality of the data, and the training recipe that converts data into parameters.** (This does not claim architecture is unimportant, or that it has already converged — only that the source of competitive advantage may be shifting.) But "data matters more" doesn't mean "more data is better" — **the core hypothesis of this article is: what robotics truly deserves to scale is not just trajectory count, but the effective coverage of the interaction distribution relative to an evaluation distribution.**

Let me define interaction distribution upfront, since it runs through the entire article: **the interaction distribution referred to here is the trajectory distribution jointly determined by conditions such as task, scene, and embodiment in the training data,** written as

$$p(\tau \mid task,\ scene,\ embodiment)$$

where the trajectory $\tau=(o_{0:T},a_{0:T-1})$ already contains observations, actions, and temporal dynamics, as well as possible success/failure information; written more explicitly it can also be expressed as $p(o_{0:T},a_{0:T-1}\mid task,scene,embodiment)$. The reason for using a conditional distribution rather than stuffing task, state, and action all into one joint distribution is that state and action are already inside $\tau$, while failure mode is often a label obtained by posterior analysis of a trajectory, $m=h(\tau)$, rather than a raw random variable that exists at collection time. This definition is also stronger than simply talking about diversity, because diversity ≠ distribution coverage — a dataset can contain many objects yet all come from the same task distribution.

One honest upgrade to the definition is worth stating here: **the trajectory distribution is not really determined only by task, scene, and embodiment.** It also depends on environment dynamics, the initial-state / reset distribution, the data-collection strategy (behavior policy / operator policy / exploration / intervention mechanism), and sensor/actuator dynamics. In other words, $p(\tau\mid task,scene,embodiment)$ has already **marginalized out** the "who is acting, and how" factors. A more rigorous notation would be

$$p_D(\tau \mid c),\qquad c=(task,\ scene,\ embodiment)$$

where the subscript $D$ reminds us that this distribution implicitly depends on the concrete collection policy and environment. The reason the article still writes the shorthand $p(\tau\mid task,scene,embodiment)$ is simply notational consistency; but the reader should keep in mind that — **once we start discussing coverage later on, what we care about most is precisely this behavior distribution hidden inside $D$.**

There is another question a technical reader will think of immediately: since what we really care about is "which state/action regions are visited" rather than "how many trajectories there are," the more apt mathematical object would arguably be the RL **policy-induced occupancy measure** $d^\pi(s,a)$ — it measures exactly "how much of the evaluation-relevant state-action region has been visited," and it aligns very well with the support/density distinction discussed later. This article still uses the trajectory distribution as a **deliberately chosen higher-level abstraction**: to discuss VLA, world model, imitation, and RL data under one notation, we choose to stay at the trajectory level; rewriting the whole piece in terms of occupancy measures would immediately turn it from "embodied AI data analysis" into an "RL theory paper." So this is not ignorance of $d^\pi(s,a)$ — it is an intentional choice of the coarser abstraction layer.

## Why Robot Data Is Not Like Internet Data

Large language models can leverage internet-scale text and code data for pre-training, with data acquisition scale far exceeding robot real-world interaction data. Vision models are similar — large-scale image-text datasets provide the foundation for VLMs.

But robot data has a fundamental difference: **it's not just "observation," it's "interaction."**

A qualifier is needed here, or this can easily be rebutted: robots actually make heavy use of observation-only data — egocentric video, internet video, human activity video, pure RGB observations, passive observation, and even some VLA data pipelines use visual/language data with no robot action. So the more rigorous statement is not "robot data can't be observation-only," but rather: **compared to internet text/images, the core increment of robot control data is not observation itself, but action-conditioned temporal interaction.** What we really want to emphasize is the information gap between $(o_t)$ and $(o_t, a_t, o_{t+1})$.

A piece of internet text only needs text; a web image only needs pixels. But the core of robot data is not a single field like "state" or "reward" — it is a **multimodal interaction trajectory with action and temporal structure:**

```
robot trajectory:
  (o_t, a_t, o_{t+1}, ...)

Optional fields:
  language instruction
  proprioception
  reward
  success / failure
  termination
  environment metadata
  task / embodiment ID
```

The key point here is: observations (RGB / RGB-D / proprioception / force-torque / joint state / end-effector pose) are what robots can typically directly obtain; while true environment state is often not directly observable. In control terms, $o_t \neq s_t$: an observation is a partial observation, whereas the state is the underlying (often latent) state used to describe the environment's Markov dynamics. This is precisely why robot data is naturally partially observable, and it connects directly, in theory, to the world model transition $p(z_{t+1}\mid z_t,a_t)$ discussed later. Similarly, reward is not a required field for demonstration data — it only appears when training reward models or actor-critic policies.

This difference is not a detail — it's a fundamental data structure difference. It determines that embodied AI cannot simply replicate LLM's "data scaling" approach.

## Several Data Source Approaches

Data sources for embodied AI fall into roughly four categories, each with a different cost / quality / coverage profile.

### Teleoperation Data

The most direct source: humans control robots to complete tasks, recording observation-action trajectory pairs.

**Advantage:** compared to pure autonomous exploration, it more easily yields task-relevant, higher-success-rate trajectories with clear behavioral intent, and naturally carries human manipulation strategies and common sense. **Limitation:** slow and costly collection; operator skill directly shapes data quality; task and environmental diversity are bounded by the operator's time and imagination.

Teleop data does not automatically equal high-quality data, though — it can carry hesitation, correction, redundant motion, inconsistent behavior, operator bias, failed attempts and recovery, and differences across skill levels (exactly what curation must handle later: human-generated ≠ high-quality). Mainstream systems include VR-controller control, SpaceMouse, and vision-based imitation; several robotics companies are building scaled teleop infrastructure, though specific volumes and coverage are usually not public.

### Autonomous Data: Online vs Offline Distinction

Letting robots collect interaction data themselves, in real or simulated environments, splits into two modes: **online interaction**, where the current policy acts in the environment ($a_t \sim \pi(\cdot|o_t)$) and the concerns are exploration efficiency, safety, reset cost, and on-policy distribution; and **offline data**, a fixed replay / demonstration set $D=\{(o_t,a_t,o_{t+1},r_t)\}$ with no further interaction. Classic RL is usually the former, but current robot RL practice widely mixes paradigms — offline RL, demonstration + RL, imitation pretraining + online RL, replay-based RL, and simulation RL + real fine-tuning — so robot RL data sources are genuinely diverse.

### Simulation Data

Generating training data in simulated environments.

**Advantage:** massive parallelism, precisely controlled parameters, automatic annotation, and extreme scenarios that are hard to obtain in the real world. **Limitation:** the sim-to-real gap remains — contact mechanics, friction, and deformation in simulation do not perfectly match reality, so direct use can degrade real-world policy performance.

NVIDIA Isaac Sim, MuJoCo (Todorov et al., IROS 2012; now open-sourced by DeepMind), and GPU-parallel simulation such as Isaac Gym (Makoviychuk et al., 2021, arXiv:2108.10470) are widely used to generate training data; simulation data typically still needs domain randomization (Tobin et al., IROS 2017, arXiv:1703.06907), system identification, or real-world fine-tuning to bridge the gap.

### Synthetic Data: World Models as Experience Generators

An increasingly important direction is **using trained world models to expand agent experience**, via two distinct mechanisms. **Model-based RL** (e.g., Dreamer, Hafner et al., 2019, arXiv:1912.01603; DreamerV3, Hafner et al., 2023, arXiv:2301.04104) uses the world model as a **latent experience generator**, training actor/critic in latent imagination ($z_t \rightarrow a_t \rightarrow z_{t+1}$) without generating photorealistic RGB frames. **Generative world models** (video-generative models, NVIDIA Cosmos) instead try to produce synthetic observations / videos / trajectories close to real ones; Cosmos positions itself as a **fine-tunable world foundation model platform / digital-twin** layer for Physical AI whose generated video is a **potential data source** for downstream work — though this is more the platform's vision and positioning, and the paper itself offers no direct empirical conclusion that "generated data improves real-robot policy training" (NVIDIA, 2025, arXiv:2501.03575). The shared key point is that **model-generated data is not "free real-data expansion":** the generated distribution $\hat p(\tau)$ generally differs from $p(\tau)$, and model error compounds over the rollout horizon:

$$D_{\mathrm{real}} \rightarrow M \rightarrow \hat D_{\mathrm{synthetic}}$$

so synthetic trajectories inherit the bias of model $M$ and are jointly limited by model bias, long-horizon compounding error, and the mismatch between generated and real interaction distributions — a distribution problem, not merely "more data."

## Data Interfaces for Different Paradigms

This is an easily overlooked but very important dimension: **different technical approaches don't need the same kind of data.**

### VLA's Data Interface

The most basic VLA training sample can be abstracted as $(o_t, l, a_{t:t+k})$, where $l$ is the language instruction and $a_{t:t+k}$ is an action chunk. But note: **an action chunk is only a common training/inference interface, not the definition of VLA** — the core of VLA is the $(V, L) \rightarrow A$ mapping, while the chunk form is a concrete policy parameterization, so a system that does not explicitly predict chunks can still be a VLA. Real systems also often add proprioception, historical observation windows, task metadata, and embodiment information, and the action output is not simply $(o,l) \rightarrow a$ but may be action chunks, diffusion / flow action heads, discrete tokens, or continuous action (heterogeneous representations). Representative works transferring large-scale vision-language knowledge to robotic control include RT-2 (Brohan et al., CoRL 2023, arXiv:2307.15818) and π₀ (Black et al., Physical Intelligence, 2024, arXiv:2410.24164).

So VLA's core data requirement is **high-quality observation-action pairs, covering sufficiently diverse tasks and objects, while adapting to different embodiment action representations.** A deeper point is that **action representation is itself part of interface design:** $a_{t:t+k}$ may be joint position/velocity, end-effector delta / absolute pose, gripper command, discretized tokens, continuous flow, or even latent action — so the real challenge of cross-embodiment is not "dumping different robots' data into one dataset" but finding a general enough observation/action representation for different embodiments' experience to be shared in one learning space.

### World Model's Data Interface

The world model's core interface is:

```
Input: observation history + action history

Core output:
  future latent state / transition distribution
  e.g., p(z_{t+1} | z_t, a_t)

Optional:
  reconstructed observation (p(o_t | z_t))
  reward / termination / task outcome
```

The key point is that **the core of a world model is learning action-conditioned dynamics; reward prediction is not a logically required output of the dynamics model.** But we should not go the other way and dictate that "reward doesn't belong to the world model" — a cleaner layering is a `dynamics model` (state transitions), a `reward model` (reward), and a `continuation / termination model` (whether the episode continues), with some literature calling these modules together the **world model**; so in an agent like Dreamer, reward and continuation often form, together with the dynamics model, the full world-model module. For routes centered on latent dynamics + model-based control (Dreamer's RSSM, TD-MPC2, Hansen et al., ICLR 2024, arXiv:2310.16828), data needs to be temporally coherent, action-annotated interaction trajectories.

### RL's Data Interface

RL's data requirements depend on the specific paradigm:

- **On-policy** (e.g., PPO): needs data produced by the current policy; data "freshness" matters
- **Off-policy** (e.g., SAC): can reuse historical replay, giving stronger data-reuse capability; but off-policy ≠ automatically more sample-efficient — efficiency still depends on replay distribution, exploration, critic quality, reward structure, and the task, and the buffer's distribution and coverage affect generalization and stability
- **Offline RL**: relies entirely on pre-collected datasets; extremely high requirements for distribution coverage
- **Imitation + RL**: first pre-train with demonstrations, then fine-tune with online interaction

Different RL paradigms impose very different requirements on replay-buffer / dataset quality and coverage.

### Data Interface Incompatibility

A common practical difficulty: **data from different embodiments, sensor configurations, and action spaces usually cannot be fed to the same low-level policy without processing.** Data collected on a Franka arm, differing in action-space dimension, viewpoint, and dynamics, typically needs action retargeting, normalization, or embodiment conditioning before transfer — which is why cross-embodiment data is an important direction.

Here three levels must be distinguished: **multi-task** (one robot, many tasks), **multi-embodiment** (data from many robots, handled separately), and **cross-embodiment** (the model generalizes to unseen robots). TD-MPC2's multi-task / multi-domain capability rests mainly on task embeddings, but **task conditioning ≠ embodiment conditioning** — embodiment differences span action space, observation space, morphology, dynamics, control frequency and more, and cannot be collapsed into one task embedding. The π₀ series demonstrates the importance of large-scale multi-embodiment data for cross-robot generalization (an empirical observation, not a causal attribution to a specific mechanism), and Open X-Embodiment (Open X-Embodiment Collaboration, 2023, arXiv:2310.08864) is a representative large-scale cross-embodiment dataset.

## Data Is Not a Dataset, It's a Distribution

"Data volume" is an easily quantified metric, but in embodied AI, **data's effective scale cannot be simply measured by trajectory count.**

### Data Volume Is Not Effective Data Scale

A common observation is that high-quality demonstrations combined with large-scale vision-language pre-training can significantly improve robot policy generalization. But "quality" needs a more precise definition — systematic suboptimal behavior or erroneous actions in demonstrations will change the behavior policy's target distribution; without filtering or weighting mechanisms, these patterns may be learned by the model.

Modern policy learning contains multiple response mechanisms: augmentation, trajectory weighting, filtering, robust loss, advantage weighting, diffusion policy smoothing, etc. But the core challenge remains: **"noise" in robot data is not just annotation errors — it also includes systematic problems like non-smooth manipulation, suboptimal strategies, and sensor noise.**

### Data Diversity and Curriculum Learning: Two Orthogonal Dimensions

Data diversity and curriculum learning are two orthogonal dimensions:

- **Diversity:** How many different situations have I seen? — determines coverage
- **Curriculum:** In what order do I see these situations? — determines optimization path

If training data only covers one type of cup, one lighting condition, one table surface, the policy will fail when encountering variation — this is insufficient diversity. Curriculum learning (simple to complex) is a training strategy that affects the optimization path, not coverage itself. **Diversity determines coverage; curriculum determines optimization path.**

Here we also need to distinguish diversity from coverage: **Diversity describes how different samples are from each other; coverage describes how much of the target task distribution has been covered.** For example: data of 1000 different cups → high diversity; but if all are "desktop cup grasping" tasks, task coverage may still be very low. For this reason diversity is better seen as a handle worth watching during curation than as an independent quantity that buys scaling returns on its own — in the later $D_{\mathrm{effective}}$ decomposition we no longer list it as a multiplier alongside Coverage; it enters scaling only by translating into coverage.

### Data Curation: From Trend to Technical

"More data" doesn't automatically equal "better performance." Data curation can be broken into six operational dimensions. These are deliberately written as a set rather than with plus signs — because they are not commensurable quantities that can simply be added (Quality, Diversity, Coverage, Relevance, and Balance are attributes of data, while Deduplication is more of an operation):

$$\mathrm{Curation} = \{\mathrm{Quality},\ \mathrm{Diversity},\ \mathrm{Coverage},\ \mathrm{Relevance},\ \mathrm{Balance},\ \mathrm{Deduplication}\}$$

For robot data, each dimension has specific technical challenges:

- **Quality:** success rate, trajectory smoothness (velocity/acceleration/jerk and other kinematic metrics), collision-free, action consistency
- **Diversity:** scene diversity, object variety, lighting variation
- **Coverage:** task coverage, failure mode coverage, edge case coverage
- **Deduplication:** deduplicate similar trajectories to avoid overfitting
- **Relevance:** whether data is relevant to the target task
- **Balance:** data ratios across different tasks and scenes

These problems currently have no standardized solutions, but are becoming an independent technical direction.

An important point needs emphasis here: **Curation does not mean simply deleting failure trajectories.** For imitation learning, clearly erroneous demonstrations may need filtering; but for world models, offline RL, or recovery policies, failure and boundary trajectories themselves may have very high informational value. For example: grasping failures, object slippage, collisions, grasp recovery, occlusion, unexpected contact — these may be the data most needed for policy robustness. Successful trajectories tell the model "doing this leads to success," while failure trajectories may tell the model "in this state, this action leads to what consequences." What truly needs to be optimized is data relevance to the target objective, not simply maximizing success rate.

Abstracting further, **the value of failure data lies not in "failure" itself, but in the action-conditioned negative outcome information it provides.** A single $(s, a_{\mathrm{bad}}, s')$ tells the model "in this state, this observation followed this action"; whereas if there are only successful demonstrations $(s, a_{\mathrm{good}}, s')$, the model does not necessarily know why $a_{\mathrm{bad}}$ is bad. Strictly speaking, $(s, a_{\mathrm{bad}}, s')$ is only an *observed transition*, not a *controlled intervention* — a rigorous notion of intervention evidence only applies when the state, action, and confounders can all be intervened upon. The reason we say "action-conditioned negative outcome" rather than "negative intervention" here is precisely to avoid grounding a causal conclusion on purely observational data.

To be precise: a strict counterfactual asks "what would happen at the same $s$ if a different $a'$ were taken," but all we observe is $(s, a_{\mathrm{bad}}, s')$ — we do not simultaneously observe the paired $(s, a_{\mathrm{good}}, s'')$. So a failure trajectory is more accurately **counterfactual-relevant information** rather than strict counterfactual data — **it acquires genuine counterfactual learning value only when combined with success trajectories or model predictions.** Even so, this action-conditioned negative outcome signal still gives failure trajectories unique value for world models and offline RL — elevating "failure data is useful" from an empirical remark to a clearer learning-theoretic intuition.

### Data Quality ≠ Data Utility

The preceding text repeatedly used words like quality, relevance, success, diversity, and coverage, but they can actually be unified under a more fundamental concept: **Data Utility.**

Let's first fully separate the two concepts:

- **Quality** describes a trajectory's *measurable properties / quality indicators* — smoothness, consistency, collision-free, sensor quality, annotation correctness. Most of these can be computed the moment the data is collected, so they *look* like intrinsic attributes; but strictly speaking even the quality indicators carry objective dependence: smoothness that matters for precision manipulation is not necessarily good for aggressive locomotion or recovery maneuvers (a sudden jerk may be exactly the right behavior), and collision-free is a quality signal for a normal policy but the collision itself is what you want to learn in a collision-recovery dataset.
- **Utility** describes the *conditioned contribution* those properties are converted into under **a specific training objective and a specific evaluation distribution** — IL utility, world-model utility, offline-RL utility, recovery utility, or even utility on one particular eval task. The same trajectory can have wildly different utility under different objectives and different evaluation distributions.

```
Quality (measurable)              Utility (conditioned)
  smoothness                        IL utility
  consistency                       WM utility
  collision-free                    RL utility
  sensor quality                    recovery utility
  annotation correctness            eval-specific utility
```

Written as a formula: **data utility is not an absolute property, but an objective- and evaluation-conditioned quantity.** The same dataset $D$ has different utility for different training objectives $\mathcal{L}$ and different evaluation distributions $p_{\mathrm{eval}}$:

$$U(D \mid \mathcal{L},\ p_{\mathrm{eval}})$$

$$U_{\mathrm{IL}}(D) \neq U_{\mathrm{WM}}(D) \neq U_{\mathrm{offlineRL}}(D)$$

This neatly explains the earlier phenomenon: a failure trajectory may be noise to be filtered for imitation learning, yet valuable signal for a world model or offline RL — because its utility differs under different objectives. Strictly speaking, "high-quality data" should really be "data with high utility under the current objective and the current evaluation distribution."

Recasting quality as utility also resolves a common misconception in curation: there is no such thing as a "universally good" dataset, only a dataset that is "good for some $(\mathcal{L}, p_{\mathrm{eval}})$." This is why data curation must be defined together with the training objective and evaluation distribution, rather than discussing "data quality" in isolation from any goal.

It is also worth adding: strictly speaking, data utility even varies with the **model class, the current training state, and the compute budget** — the same data may be useless for a small model yet very useful for a large one; once a model has already learned A, adding more A data has low value, whereas before it has learned A the same data is highly valuable. So the $U$ in this article is a **stage-conditioned utility**, not a static property of the data. (To keep the framework from becoming over-formalized, however, we still write it simply as $U(D \mid \mathcal{L}, p_{\mathrm{eval}})$ and leave the $M$ and $C$ dependencies in prose.)

As for what $p_{\mathrm{eval}}$ actually is, and why it is the single most important patch in this version — we formally introduce it in the next section on robot scaling, where the point becomes clear: **without an evaluation distribution as a reference frame, "coverage" is really a sentence with no subject.**

## Training Recipes: Determining What the Model Sees

Training recipes are not just hyperparameters — they are **the entire pipeline determining what data, with what weight, in what order, through what objective, gets seen by the model.**

Specifically, a complete robot training recipe may include:

- **Sampling strategy:** sampling ratios across data sources (internet data vs robot data vs synthetic data)
- **Trajectory weighting:** should higher-quality trajectories receive more weight?
- **Action chunking:** predict single-step actions or action sequences? What chunk length?
- **Temporal horizon:** context window length used during training
- **Loss weighting:** relative weights of different loss terms (action prediction, value, reward)
- **Augmentation:** visual augmentation, action augmentation, domain randomization
- **Observation / action normalization:** how to normalize observations and actions across different embodiments
- **Freezing / unfreezing schedule:** when to freeze and unfreeze pre-trained backbones
- **Mixture-of-data sampling:** mixing strategies for multi-source data
- **Intervention data / failure data:** whether to include human intervention data or failure trajectories
- **Replay ratio:** how many times offline data is reused
- **Offline/online mixing:** whether to combine offline pretraining and online fine-tuning
- **Fine-tuning schedule:** learning rate, batch size, training epoch scheduling

Different teams' choices in these areas can vary significantly, and these choices often have significant impact on final performance — in many teams' practice, this impact may even exceed the choice of model architecture. It should be noted that this judgment currently rests more on engineering experience and case reports than on systematic, cross-task controlled quantitative experiments; for this reason, it is better treated as a hypothesis worth testing rather than an established conclusion. This is also why training recipes are difficult to fully transmit through a single paper — they are an entire engineering practice, not a set of hyperparameters.

From a more abstract perspective, **a training recipe is essentially the "transfer function" from data distribution to model parameters:**

$$D \rightarrow Training\ Recipe \rightarrow \theta$$

But more accurately, the recipe is not a factor independent of the distribution — **the recipe itself changes the distribution the model actually "sees."** Raw data $D$ first goes through sampling $S_R(D)$, then through weighting $W_R(S_R(D))$; what really enters optimization is the recipe-transformed $D_R$:

$$D \xrightarrow{\ Recipe\ } D_R \xrightarrow{\ Optimization\ } \theta,\qquad D_R = W_R\big(S_R(D)\big)$$

In distributional language, we can view the recipe $R$ as a transformation operator $T_R$ acting on the trajectory distribution:

$$p_{\mathrm{train}}(\tau) = T_R\big[\,p_{\mathrm{raw}}(\tau)\,\big]$$

In other words, performance depends on the $p_{\mathrm{train}}(\tau)$ the model actually experiences, not the static $p_{\mathrm{raw}}(\tau)$ sitting in storage; $p_{\mathrm{train}}$ is precisely the result of the recipe resampling, reweighting, and reordering the raw distribution. Given the same dataset $D$, changing the sampling / weighting / objective / schedule ($R_1 \neq R_2$) can yield different $p_{\mathrm{train}}$ and hence $\theta_1 \neq \theta_2$ — which elevates the "recipe as moat" argument from empirical observation to a clearer technical framework.

One caveat, though: writing $p_{\mathrm{train}}$ as a distribution fixed by a recipe only really holds for **offline / fixed-dataset training**. In **online RL or continuous closed-loop data systems**, $p_{\mathrm{train},t} \rightarrow \theta_t \rightarrow \pi_t \rightarrow p_{\mathrm{collect},t+1}$ is itself a **feedback loop** — the policy update changes the distribution of subsequently collected data. To keep the notation clean this article omits the time subscript $t$ where it is not essential, but in online / closed-loop settings the reader should read $p_{\mathrm{train}}$ as $p_{\mathrm{train},t}$.

### Recipe's Two Paths of Influence

But if we stop here, an attentive reader will immediately notice a double-counting problem: if $p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$ already folds the recipe into the distribution, then letting Recipe reappear as an independent argument in $Performance = g(D_{\mathrm{effective}},\ Capacity,\ Compute,\ Recipe)$ in the next section would seem to count it twice.

Here we have to concede: **a recipe is not only a distribution transformation — it simultaneously acts on optimization dynamics.** For analytical convenience, we can **roughly** split the recipe's main effects into two parallel paths (as we will see below, this boundary is not a clean partition):

```
              ┌── Path 1: Distribution Transformation ──►  p_train(τ) = T_R[p_raw(τ)]
              │
Raw Data D ───┤
              │
              └── Path 2: Optimization Dynamics ─────────►  learning-rate schedule
                                                            optimizer / momentum
                                                            loss weighting
                                                            freezing / unfreezing
                                                            curriculum / staging
                                                            gradient clipping
```

Path 1 determines what the model *sees*; Path 2 determines how the model *turns what it sees into parameters*. The two cannot be reduced to each other: given the same $p_{\mathrm{train}}$, different lr schedules, optimizers, loss weightings, or freezing strategies still yield significantly different $\theta$ — this is not the distribution changing, it is the optimization process itself changing.

A caveat is in order: this "two paths" framing is only an **analytical convenience, not a clean partition.** Some recipe choices straddle both paths at once — the most typical being loss weighting: $L = \lambda_a L_{\text{action}} + \lambda_v L_{\text{value}}$ both changes the effective weighting of different data/objective terms (a Path-1 flavor) and directly changes the optimization dynamics (Path 2). Likewise, operations such as image augmentation, action tokenization, temporal-window / observation stacking, action-chunk construction, hindsight relabeling, and reward / label construction affect both the effective distribution and the optimization target. We deliberately do **not** promote these into a formal "Path 3" — that would only fragment the taxonomy further; the point is that precisely because such operations span both sides, the "two paths" picture is itself a coarse-graining. So we do not claim that "any recipe can be uniquely decomposed into these two paths"; we only use the picture to make the point that part of the recipe's effect cannot be absorbed by $p_{\mathrm{train}}$ at all.

Therefore, in the next section's $g(D_{\mathrm{effective}},\ Capacity,\ Compute,\ Recipe)$, the $Recipe$ argument specifically retains the **Path 2 component that cannot be absorbed by $p_{\mathrm{train}}$** — the optimization dynamics — while Path 1 has already been folded into $D_{\mathrm{effective}}$. This split is not a notational nicety: it directly determines which interventions count as "changing the data" vs "changing the training," and it is exactly where many teams' recipe differences become hard to reproduce. Papers can publish Path 1 (data mixture, weighting), but Path 2 knowledge — when to unfreeze the backbone, when to switch the lr, when to change loss weighting — is often left out.

### Data Mixture: How to Mix Data?

An easily overlooked but extremely critical question is: **how are different data sources mixed?**

```
The following ratios are illustrative only, meant to convey the concept of a mixture — they are not industry statistics:

Internet / VLM data        ── 70%
Robot demonstrations       ── 20%
Synthetic / simulation     ── 10%
```

What truly matters may not be "how much robot data do we have?" but rather: **what proportion does robot data occupy in the overall training mixture? When is it introduced? With what loss is it trained?** This is precisely the core manifestation of training recipe as "transfer function" — the same data, with different mixture ratios and scheduling strategies, can produce markedly different model capabilities.

## Sim-to-Real: Four Common Tools

Simulation data cannot directly replace real data, but there are multiple tools for handling the sim-to-real problem. It should be noted that the following four are not a same-level, mutually-exclusive taxonomy, but tools acting at different abstraction levels: system identification is model calibration, domain randomization is training distribution manipulation, real-world fine-tuning is an optimization strategy, and domain adaptation is representation / distribution alignment. They operate respectively on simulation fidelity, training distribution, representation, and policy adaptation — and can therefore be combined.

```
System identification
  real world → calibrate simulator
  Goal: make the simulator closer to the real system

Domain randomization
  simulator → enlarge training distribution
  Goal: train a policy robust to a set of possible domains
  (not "bridging distribution gap," but enlarging training distribution)

Real-world fine-tuning
  sim → real adaptation
  Goal: adapt policies learned in simulation with a small amount of real data

Domain adaptation
  sim ↔ real representation alignment
  Goal: learn shared representations between simulation and reality
```

These four strategies are typically not mutually exclusive — practical systems often combine them.

Here we need to distinguish two different types of sim-to-real error: **random noise** (e.g., sensor noise, small physical parameter fluctuations) and **systematic simulation bias** (e.g., friction coefficient long-term bias, actuator delay, contact model errors, deformable object dynamics errors, camera latency, calibration error). Random noise can be addressed through domain randomization to enhance robustness, while systematic bias typically requires system identification to calibrate the simulator itself. A more accurate statement is: **system identification primarily targets systematic simulator mismatch (pulling the simulator itself back toward the real system), while domain randomization primarily broadens the training distribution so that the policy becomes robust to parameter and environment variation — when the randomization range covers the real system, it can also mitigate part of the systematic mismatch, not by calibrating the simulator, but by making the policy learn to work over an entire family of simulators.** The two are not mutually exclusive, and in real systems are often combined.

## Robot Data Scaling: Not Just "More Trajectories"

The LLM domain has accumulated a fairly mature scaling-law empirical framework (Kaplan et al., 2020, arXiv:2001.08361; Hoffmann et al., 2022, arXiv:2203.15556), describing how data, parameters, and compute jointly determine loss under specific compute-optimal / loss-scaling regimes — but this is not a unified "natural law," nor does it transfer directly to robotics. Does a comparable scaling law exist for robotics?

### Data Acquisition ≠ Data Scaling

Before talking about scaling, we need to separate two questions that are often conflated.

**Data acquisition** answers "where does data come from?" (teleoperation, simulation, autonomous exploration, and synthetic generation are all *acquisition methods* — what the first half of this article discussed), while **Data scaling** answers "what data should I add next, per unit of budget?" (support expansion, density improvement, failure targeting, and embodiment expansion are *scaling strategies*). One is a generation mechanism, the other an allocation problem — no matter how strong your acquisition methods are, they do not automatically answer the scaling question. The second half therefore shifts its center of gravity from "where data comes from" to "what data is worth continuing to add," which is exactly what the framework below tries to answer.

First, an important clarification: **the formulas below are not strict scaling laws, but a conceptual decomposition for describing robot data's effective scale.** Robot data scale can be decomposed into at least three layers:

**Data volume:** $N_{\text{steps}}$ (total interaction steps)

**Distribution dimensions:** $task, scene, embodiment, \text{behavioral state}, action$ (distribution dimensions)

**Data quality:** $Q$ (data quality)

The "state" dimension needs clarification here: in line with the earlier $o_t \neq s_t$ discussion, we do not mean explicitly annotated environment state (the true $s$ is often not directly obtainable), but rather **behavioral-state coverage / state-space coverage** — the (often latent or inferred) behavioral-state distribution the model actually visits during training. Framed this way, it does not conflict conceptually with the earlier partial-observability discussion.

### Distribution ≠ Coverage: Introducing an Evaluation Distribution

Before we start talking about coverage, we have to patch in a concept that has been implicit all along. Earlier we defined the interaction distribution as

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

This also tightens the earlier utility definition by one notch: **data utility is not only objective-conditioned — it is evaluation-conditioned.**

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

One more caveat closes the loop with the earlier $o_t \neq s_t$: in real robot data, $s$ is often **not directly observable** at all, so $p_D(a \mid s)$ is more accurately read as a behavior distribution conditioned on a *latent / inferred* behavioral state. In practice it can only be approximated through observation history, proprioception, or a learned representation.

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

In other words, failure is not a new "distribution dimension"; it is a subset carved out of the same interaction distribution by a posterior label $m=h(\tau)$. Its value has already been discussed under Data Utility / action-conditioned negative outcome information; here we are simply putting it in the right slot of the taxonomy.

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

This leads directly to a key judgment: **deciding when to support-scale and when to density-scale is itself a core training-recipe question** (echoing the earlier $p_{\mathrm{train}}(\tau) = T_R[p_{\mathrm{raw}}(\tau)]$ — the recipe determines which regions of the raw data are amplified and which are compressed).

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

If we connect the threads from previous articles:

- [The world model series](/en/articles/2026-09-01-world-model-h2-review/) established the "prediction interface" concept
- [The VLA series](/en/articles/2026-09-03-vla-deep-dive/) analyzed "semantics + action" interface design
- [RSSM evolution](/en/articles/2026-09-04-rssm-beyond/) discussed different latent dynamics' data requirements
- [The industry landscape](/en/articles/2026-09-06-embodied-ai-landscape/) pointed out data is becoming a key differentiator

If we condense this article's core into "five pillars," they are:

$$\boxed{Interaction\ Distribution:\ p(\tau \mid task,\ scene,\ embodiment)}$$

$$\boxed{Training\ vs.\ Evaluation:\ p_{\mathrm{train}}(\tau)\ \leftrightarrow\ p_{\mathrm{eval}}(\tau)}$$

$$\boxed{Support\ vs.\ Density\ Scaling}$$

$$\boxed{Data\ Utility:\ U(D \mid \mathcal{L},\ p_{\mathrm{eval}})}$$

$$\boxed{Recipe:\ p_{\mathrm{raw}}(\tau) \xrightarrow{T_R} p_{\mathrm{train}}(\tau)\ \text{+ Optimization Dynamics}}$$

Taken together, these five are really trying to say one thing: **the basic unit of robot scaling may not be the trajectory, but the effective coverage of the interaction distribution relative to a target evaluation distribution.** Moving from "robot data is complicated" to "what actually counts as effective robot data scaling" is precisely the step this article tries to take.

### Marginal Data Value: Condensing the Whole Framework into an Actionable Concept

Every concept above — interaction distribution, $p_{\mathrm{eval}}$, support/density, utility, recipe — is ultimately answering the same question: **is the next batch of data worth collecting?** This question deserves a formal name. But first, a baseline must be added: the value of a new batch $D'$ only ever makes sense *given what data you already have*, $D$ — so $\Delta Performance$ should be written explicitly as an increment relative to $D$:

$$MV(D';\,D) \;=\; \frac{Performance(D \cup D') - Performance(D)}{Cost(D')}$$

For a single trajectory we can also write:

$$MV(\tau;\,D) \;=\; \frac{Performance(D \cup \{\tau\}) - Performance(D)}{Cost(\tau)}$$

The "$D$" argument looks like a notational detail, but it actually encodes the article's most core distributional argument right into the definition: **the value of a batch of data depends on the data you already have.**

With this in hand, the whole article's thesis condenses into a single sentence:

> **The core question of robot data scaling is not how to maximize data volume, but how to maximize marginal data value.**

This sentence is easier to remember than "effective interaction-distribution coverage," and closer to engineering practice — because volume is a quantity one can push blindly, while $MV$ forces you to answer "relative to the current $p_{\mathrm{train}}$ and $p_{\mathrm{eval}}$, what exactly does this batch fill in, and at what cost?"

And from the baseline-carrying $MV(D';D)$ notation, one can read off what may be the article's single most central insight:

$$MV(D';\,D_t) \;\neq\; MV(D';\,D_{t+1})$$

**Data value is state-dependent.** The same trajectory may be very valuable in the early, data-scarce phase, yet nearly worthless once the relevant regions of the distribution have already been filled in. This is precisely the root reason why "a dataset's quality cannot be permanently defined" — good data was never absolutely "good data," but "**data with high marginal utility under the current training state and evaluation gap.**" It is here that Data Utility → Marginal Data Value → Data Flywheel finally close into a loop.

The natural consequence is that the optimal collection policy cannot be a *static* one. If $MV(D';D_t) \neq MV(D';D_{t+1})$, then which data to collect next must itself depend on the current dataset, the evaluation target, and the model's current parameters:

$$D'_t = \pi_{\mathrm{data}}(D_t,\ p_{\mathrm{eval}},\ \theta_t)$$

The flywheel is therefore not simply "collect data → train → collect again," but *learning a data-collection policy that keeps changing as the state changes.* Put as a single line: **robot data scaling is not a static dataset construction problem; it is a sequential data allocation problem.**

### From Scaling Hypothesis to Data Flywheel

What this article wants to say is: **data and training recipes may be becoming embodied AI's most underestimated competitive advantage.**

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

At this point we can name the article's real subject: on the surface it talks about "data scaling," but what it is fundamentally about is **evaluation-aware distribution allocation under a limited data budget** — not passively "aligning" $p_{\mathrm{train}}$ to some fixed $p_{\mathrm{eval}}$, but actively allocating a limited collection budget, round by round, to the support and density gaps in $p_{\mathrm{train}}$ that $p_{\mathrm{eval}}$ exposes, spending each unit of budget wherever marginal data value is highest. Compressing the entire analytical framework into a single diagram, it closes as follows — note that $p_{\mathrm{eval}}$ sits at the top, as **the target coordinate system of the whole loop**:

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

- RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control — Brohan et al., CoRL 2023, arXiv:2307.15818
- π₀: A Vision-Language-Action Flow Model for General Robot Control — Black et al., Physical Intelligence, 2024, arXiv:2410.24164
- Open X-Embodiment: Robotic Learning Datasets and RT-X Models — Open X-Embodiment Collaboration, 2023, arXiv:2310.08864
- Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) — Hafner et al., 2019, arXiv:1912.01603
- Mastering Diverse Domains through World Models (DreamerV3) — Hafner et al., 2023, arXiv:2301.04104
- TD-MPC2: Scalable, Robust World Models for Continuous Control — Hansen et al., ICLR 2024, arXiv:2310.16828
- Cosmos World Foundation Model Platform for Physical AI — NVIDIA, 2025, arXiv:2501.03575
- MuJoCo: A physics engine for model-based control — Todorov, Erez & Tassa, IROS 2012, DOI:10.1109/IROS.2012.6386109
- Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning — Makoviychuk et al., 2021, arXiv:2108.10470
- Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World — Tobin et al., IROS 2017, arXiv:1703.06907
- Scaling Laws for Neural Language Models — Kaplan et al., 2020, arXiv:2001.08361
- Training Compute-Optimal Large Language Models (Chinchilla) — Hoffmann et al., 2022, arXiv:2203.15556
- Data Scaling Laws in Imitation Learning for Robotic Manipulation — Lin et al., 2024, arXiv:2410.18647

The works above lean toward models and scaling frameworks. More directly relevant to this article's thesis that "data / distribution is what matters" are the following classes of empirical studies focused on the dataset itself (collection scale, diversity, quality filtering, simulation–real mixing):

- DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset — Khazatsky et al., 2024, arXiv:2403.12945 (a large-scale, multi-scene real-robot manipulation dataset; what it directly demonstrates is data scale and environment/task diversity, not the causal claim "diversity → scaling benefit," which remains this article's hypothesis)
- SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning — Zhang et al., 2025, arXiv:2505.22626 (self-supervised, composable data cleaning / quality filtering)
- Consistency Matters: Defining Demonstration Data Quality Metrics in Robot Learning from Demonstration — Sakr et al., 2024, arXiv:2412.14309 (measuring demonstrations with quality metrics such as consistency, rather than assuming "human demonstration = high quality")
- Efficient Data Collection for Robotic Manipulation via Compositional Generalization — Gao et al., 2024, arXiv:2403.05110 (reducing data collection cost by compositionally generalizing over scene elements)
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., 2025, arXiv:2503.24361 (a systematic study of recipes for mixing simulation and real data)

Note that robotics does not yet have a single universally-accepted scaling law comparable to that of LLMs; the effective-data-scale framework in this article is a conceptual decomposition and a testable hypothesis, not an established conclusion. The data-side works above provide scattered empirical support, not yet a full quantitative validation of that hypothesis.

---

*This article extends the embodied AI series — from "who is doing what" to "what is driving performance." The next article may discuss sim-to-real methodology in detail.*
