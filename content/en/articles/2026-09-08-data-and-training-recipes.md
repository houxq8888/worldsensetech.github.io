---
title: "The Data Landscape of Embodied AI (Part 1): Sources, Interfaces, Distribution, and Training Recipes"
slug: "2026-09-08-data-and-training-recipes"
date: 2026-09-08
draft: false
categories: ["Embodied Intelligence", "Training Methods"]
tags: ["Embodied Intelligence", "Robot Data", "Training Recipe", "Teleoperation", "Synthetic Data", "Sim-to-Real", "Data Curation", "VLA", "World Model"]
description: "As foundational paradigms like VLA and world models converge toward clearer mainstream routes, data distribution, data quality, and training recipes are increasingly becoming important variables determining robot performance. This post (Part 1) walks through where robot data comes from and how it enters training — sources and interfaces, why 'data is a distribution, not a dataset', how training recipes shape what the model actually sees, and the four common sim-to-real tools; Part 2 then builds the robot-data scaling framework on top of this landscape."
toc: true
related_articles:
  - 2026-09-10-robot-data-scaling
  - 2026-09-06-embodied-ai-landscape
  - 2026-09-07-vla-world-models
  - 2026-09-05-vla-pi-family
  - 2026-09-04-rssm-beyond
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
---

> This is **Part 1** of a two-post series on "the data problem in embodied AI." Part 1 focuses on the data landscape — sources, interfaces, distribution, training recipes, and sim-to-real; the theoretical scaling framework (interaction coverage, marginal data value, data flywheel, sequential data allocation) lives in [Part 2: Robot Data Scaling](/en/articles/2026-09-10-robot-data-scaling/).

In [the previous industry landscape article](/en/articles/2026-09-06-embodied-ai-landscape/), I mentioned an increasingly obvious trend: pure model architecture differences are becoming less likely to form decisive advantages, while the importance of data scale, data diversity, and training recipes is rising.

This two-post series wants to expand on that problem. But first, a clarification: this is not a survey of "what kinds of robot data exist," but an attempt to propose an analytical framework for robot scaling — where this post (Part 1) first lays out the layers of robot data (sources, interfaces, distribution, training recipes), and Part 2 then formally builds the scaling framework on top of them. The core hypothesis of the whole series is:

$$Performance \neq f(\#trajectory)$$

$$Performance = f(interaction\ distribution,\ data\ quality,\ recipe)$$

That is: **as foundational paradigms like VLA and world models converge toward a few mainstream routes (although concrete architectures are still evolving rapidly — diffusion / flow / autoregressive action heads, latent vs video world models, and action representations are all still unsettled), the difficulty of gaining a stable performance edge from local architecture innovation alone may be rising. What increasingly determines performance is the interaction distribution the model sees, the quality of the data, and the training recipe that converts data into parameters.** (This does not claim architecture is unimportant, or that it has already converged — only that the source of competitive advantage may be shifting.) But "data matters more" doesn't mean "more data is better" — **the core hypothesis of this series is: what robotics truly deserves to scale is not just trajectory count, but the effective coverage of the interaction distribution relative to an evaluation distribution.**

Let me define interaction distribution upfront, since it runs through the entire article: **the interaction distribution referred to here is the trajectory distribution jointly determined by conditions such as task, scene, and embodiment in the training data,** written as

$$p(\tau \mid task,\ scene,\ embodiment)$$

where the trajectory $\tau=(o_{0:T},a_{0:T-1})$ already contains observations, actions, and temporal dynamics, as well as possible success/failure information; written more explicitly it can also be expressed as $p(o_{0:T},a_{0:T-1}\mid task,scene,embodiment)$. The reason for using a conditional distribution rather than stuffing task, state, and action all into one joint distribution is that state and action are already inside $\tau$, while failure mode is often a label obtained by posterior analysis of a trajectory, $m=h(\tau)$, rather than a raw random variable that exists at collection time. This definition is also stronger than simply talking about diversity, because diversity ≠ distribution coverage — a dataset can contain many objects yet all come from the same task distribution.

One honest upgrade to the definition is worth stating here: **the trajectory distribution is not really determined only by task, scene, and embodiment.** It also depends on environment dynamics, the initial-state / reset distribution, the data-collection strategy (behavior policy / operator policy / exploration / intervention mechanism), and sensor/actuator dynamics. In other words, $p(\tau\mid task,scene,embodiment)$ has already **marginalized out** the "who is acting, and how" factors. A more rigorous notation would be

$$p_D(\tau \mid c),\qquad c=(task,\ scene,\ embodiment)$$

where the subscript $D$ reminds us that this distribution implicitly depends on the concrete collection policy and environment. The reason the article still writes the shorthand $p(\tau\mid task,scene,embodiment)$ is simply notational consistency; but the reader should keep in mind that — **once we start discussing coverage in Part 2, what we care about most is precisely this behavior distribution hidden inside $D$.**

There is another question a technical reader will think of immediately: since what we really care about is "which state/action regions are visited" rather than "how many trajectories there are," the more apt mathematical object would arguably be the RL **policy-induced occupancy measure** $d^\pi(s,a)$ — it measures exactly "how much of the evaluation-relevant state-action region has been visited," and it aligns very well with the support/density distinction discussed in Part 2. This article still uses the trajectory distribution as a **deliberately chosen higher-level abstraction**: to discuss VLA, world model, imitation, and RL data under one notation, we choose to stay at the trajectory level; rewriting the whole piece in terms of occupancy measures would immediately turn it from "embodied AI data analysis" into an "RL theory paper." So this is not ignorance of $d^\pi(s,a)$ — it is an intentional choice of the coarser abstraction layer.

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

The key point here is: observations (RGB / RGB-D / proprioception / force-torque / joint state / end-effector pose) are what robots can typically directly obtain; while true environment state is often not directly observable. In control terms, $o_t \neq s_t$: an observation is a partial observation, whereas the state is the underlying (often latent) state used to describe the environment's Markov dynamics. This is precisely why robot data is naturally partially observable, and it connects directly, in theory, to the world model transition $p(z_{t+1}\mid z_t,a_t)$. Similarly, reward is not a required field for demonstration data — it only appears when training reward models or actor-critic policies.

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

Here we also need to distinguish diversity from coverage: **Diversity describes how different samples are from each other; coverage describes how much of the target task distribution has been covered.** For example: data of 1000 different cups → high diversity; but if all are "desktop cup grasping" tasks, task coverage may still be very low. For this reason diversity is better seen as a handle worth watching during curation than as an independent quantity that buys scaling returns on its own — in the Part 2 $D_{\mathrm{effective}}$ decomposition we no longer list it as a multiplier alongside Coverage; it enters scaling only by translating into coverage.

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

As for what $p_{\mathrm{eval}}$ actually is, and why it is the single most important patch in this version — we formally introduce it in [Part 2](/en/articles/2026-09-10-robot-data-scaling/) on robot scaling, where the point becomes clear: **without an evaluation distribution as a reference frame, "coverage" is really a sentence with no subject.**

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

But if we stop here, an attentive reader will immediately notice a double-counting problem: if $p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$ already folds the recipe into the distribution, then letting Recipe reappear as an independent argument in Part 2's $Performance = g(D_{\mathrm{effective}},\ Capacity,\ Compute,\ Recipe)$ would seem to count it twice.

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

Therefore, in Part 2's $g(D_{\mathrm{effective}},\ Capacity,\ Compute,\ Recipe)$, the $Recipe$ argument specifically retains the **Path 2 component that cannot be absorbed by $p_{\mathrm{train}}$** — the optimization dynamics — while Path 1 has already been folded into $D_{\mathrm{effective}}$. This split is not a notational nicety: it directly determines which interventions count as "changing the data" vs "changing the training," and it is exactly where many teams' recipe differences become hard to reproduce. Papers can publish Path 1 (data mixture, weighting), but Path 2 knowledge — when to unfreeze the backbone, when to switch the lr, when to change loss weighting — is often left out.

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

## Recap: Why the Data Landscape Is a Prerequisite for the Scaling Discussion

By this point, Part 1 has laid out several things about robot data: its **sources and interfaces** (teleoperation, simulation, autonomous exploration, synthetic generation, and the fundamental `observation ≠ state` structure), why **data is a distribution rather than a dataset** (interaction distribution, quality ≠ utility, the multiple dimensions of curation), and how **training recipes determine the distribution the model actually sees** ($p_{\mathrm{train}} = T_R[p_{\mathrm{raw}}]$, the two paths of influence, data mixture) — finally landing on the **four common sim-to-real tools**.

But all of this still sits at the layer of "where data comes from, and in what form it enters training." The real scaling question — **what data should the next unit of budget add, and is it worth it** — needs a stricter framework: how an evaluation distribution gives coverage a reference frame, how support / density / distribution similarity come apart into three distinct things, how marginal data value is defined, and how the data flywheel and sequential data allocation turn this framework into an actionable problem. That is what [Part 2: Robot Data Scaling](/en/articles/2026-09-10-robot-data-scaling/) takes up.

## References

The main works referenced in Part 1 are listed below (all searchable via arXiv ID):

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

More directly relevant to Part 1's thesis that "data / distribution is what matters" are the following classes of empirical studies focused on the dataset itself (collection scale, diversity, quality filtering, simulation–real mixing):

- DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset — Khazatsky et al., 2024, arXiv:2403.12945 (a large-scale, multi-scene real-robot manipulation dataset; what it directly demonstrates is data scale and environment/task diversity, not the causal claim "diversity → scaling benefit," which remains this series' hypothesis)
- SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning — Zhang et al., 2025, arXiv:2505.22626 (self-supervised, composable data cleaning / quality filtering)
- Consistency Matters: Defining Demonstration Data Quality Metrics in Robot Learning from Demonstration — Sakr et al., 2024, arXiv:2412.14309 (measuring demonstrations with quality metrics such as consistency, rather than assuming "human demonstration = high quality")
- Efficient Data Collection for Robotic Manipulation via Compositional Generalization — Gao et al., 2024, arXiv:2403.05110 (reducing data collection cost by compositionally generalizing over scene elements)
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., 2025, arXiv:2503.24361 (a systematic study of recipes for mixing simulation and real data)

(The classic scaling-law works — Kaplan et al., 2020, arXiv:2001.08361; Chinchilla / Hoffmann et al., 2022, arXiv:2203.15556; and the data scaling law for robotic imitation learning, Lin et al., 2024, arXiv:2410.18647 — are listed together with the scaling framework in [Part 2](/en/articles/2026-09-10-robot-data-scaling/)'s references.)

Note that robotics does not yet have a single universally-accepted scaling law comparable to that of LLMs; the effective-data-scale framework across this series is a conceptual decomposition and a testable hypothesis, not an established conclusion. The data-side works above provide scattered empirical support, not yet a full quantitative validation of that hypothesis.

---

*This is Part 1 of the two-post series on "the data problem in embodied AI." Part 2, [Robot Data Scaling: From Interaction Coverage to Marginal Data Value](/en/articles/2026-09-10-robot-data-scaling/), advances this data landscape into a discussable scaling framework.*
