---
title: "The Data Problem in Embodied AI: As Foundational Paradigms Stabilize, What Determines Performance?"
slug: "2026-09-08-data-and-training-recipes"
date: 2026-09-08
draft: false
categories: ["Embodied Intelligence", "Training Methods"]
tags: ["Embodied Intelligence", "Robot Data", "Training Recipe", "Teleoperation", "Synthetic Data", "Sim-to-Real", "Data Curation", "VLA", "World Model", "Scaling Law"]
description: "As foundational paradigms like VLA and world models gradually form relatively stable technical approaches, data distribution, data quality, and training recipes are increasingly becoming important variables determining robot performance. But robot data is not simply 'the more the better' — what robotics truly needs to scale is not just trajectory count, but interaction distribution."
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

That is: **as foundational paradigms like VLA and world models gradually form relatively stable technical approaches, it is becoming increasingly difficult to gain performance advantages solely through model architecture differences. What increasingly determines performance is the interaction distribution the model sees, the quality of the data, and the training recipe that converts data into parameters.** But "data matters more" doesn't mean "more data is better" — what robotics truly needs to scale is not just trajectory count, but interaction distribution.

Let me define interaction distribution upfront, since it runs through the entire article: **the interaction distribution referred to here is the trajectory distribution jointly determined by conditions such as task, scene, and embodiment in the training data,** written as

$$p(\tau \mid task,\ scene,\ embodiment)$$

where the trajectory $\tau=(o_{0:T},a_{0:T-1})$ already contains observations, actions, and temporal dynamics, as well as possible success/failure information; written more explicitly it can also be expressed as $p(o_{0:T},a_{0:T-1}\mid task,scene,embodiment)$. The reason for using a conditional distribution rather than stuffing task, state, and action all into one joint distribution is that state and action are already inside $\tau$, while failure mode is often a label obtained by posterior analysis of a trajectory, $m=h(\tau)$, rather than a raw random variable that exists at collection time. This definition is also stronger than simply talking about diversity, because diversity ≠ distribution coverage — a dataset can contain many objects yet all come from the same task distribution.

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

Currently, data sources for embodied AI can be roughly divided into four categories.

### Teleoperation Data

The most direct source is having humans control robots to complete tasks, recording observation-action trajectory pairs.

**Advantage:** Compared to pure autonomous exploration, it is easier to obtain trajectories that are task-relevant, have higher success rates, and carry clear behavioral intent; it naturally contains human manipulation strategies and common sense.

Note, however, that teleop data does not automatically equal high-quality data. It can likewise contain hesitation, correction, redundant motion, inconsistent behavior, operator bias, failed attempts and recovery, and differences across operators of varying skill levels — which is exactly the problem curation must handle later: human-generated ≠ high-quality.

**Limitation:** Slow collection speed, high cost; operator skill level directly affects data quality; task diversity and environmental diversity are limited by the operator's time and imagination.

Current mainstream teleoperation systems include VR controller-based control, SpaceMouse, and vision-based imitation systems. Multiple robotics companies are building scaled teleoperation data collection infrastructure, though specific data volumes and coverage are typically not public.

### Autonomous Data: Online vs Offline Distinction

Having robots collect interaction data in real or simulated environments. Here an important distinction needs to be made.

**Online interaction:** The policy currently produces actions in the environment $a_t \sim \pi(\cdot|o_t)$, then obtains new trajectories. Typical problems are exploration efficiency, safety, reset cost, on-policy distribution. Classic RL typically relies on this approach — agents repeatedly trial-and-error in environments.

**Offline data:** Existing replay / demonstration data $D=\{(o_t,a_t,o_{t+1},r_t)\}$, without continuing to interact with the environment.

But in current robot RL practice, pure online RL is not the only paradigm. Increasingly common approaches include: offline RL, demonstration + RL, imitation pretraining + online RL, replay-based RL, and simulation RL + real fine-tuning. Therefore, robot RL data sources are actually diverse — online interaction, offline trajectories, demonstration data, and simulation-generated data are all widely used.

### Simulation Data

Generating training data in simulated environments.

**Advantage:** Can be massively parallelized, precisely controlled environment parameters, automatic annotation; can generate extreme scenario data difficult to obtain in real environments.

**Limitation:** Sim-to-real gap still exists — physical dynamics in simulation (contact mechanics, friction, deformation) don't perfectly match the real world. The distribution of simulation data and real data have mismatches, and direct use can cause policy performance degradation in real environments.

NVIDIA Isaac Sim, MuJoCo (Todorov et al., IROS 2012; now open-sourced by DeepMind), and other simulation platforms are being widely used to generate training data; GPU-parallel physics simulation (e.g., Isaac Gym, Makoviychuk et al., 2021, arXiv:2108.10470) further lowers the cost of large-scale collection. But simulation data typically needs to be combined with domain randomization (Tobin et al., IROS 2017, arXiv:1703.06907), system identification, or real-world fine-tuning to bridge the gap.

### Synthetic Data: World Models as Experience Generators

An increasingly important direction is: **using trained world models to expand agent experience.**

But here two different mechanisms need to be distinguished:

**Model-based RL (e.g., Dreamer):** The world model serves as a **latent experience generator**, producing imagined trajectories in latent space. Actor/critic trains in latent imagination: $z_t \rightarrow a_t \rightarrow z_{t+1}$, without needing to generate photorealistic RGB frames (Dreamer, Hafner et al., 2019, arXiv:1912.01603; DreamerV3, Hafner et al., 2023, arXiv:2301.04104).

**Generative world models (e.g., video-generative world models, NVIDIA Cosmos):** Further attempt to generate synthetic data (synthetic observations / videos / trajectories) close to real observations, serving as data sources for downstream training (Cosmos World Foundation Model Platform for Physical AI, NVIDIA, 2025, arXiv:2501.03575).

Both are "using models to expand experience," but the data forms are completely different. The former primarily serves latent prediction, imagination, and model-based control; the latter is closer to the traditional sense of "synthetic data generation."

The "world model" here includes two related but different concepts: dynamics models for latent prediction / imagination / control, and generative world models for generating or predicting visual worlds.

But there is a point that must be brought into the article's distribution framework: **data generated by a world model is not "free real-data expansion."** The distribution of generated trajectories $\hat p(\tau)$ generally does not equal the real distribution $p(\tau)$, and model error accumulates over the rollout horizon (compounding error). This chain can be written as:

$$D_{\mathrm{real}} \rightarrow M \rightarrow \hat D_{\mathrm{synthetic}}$$

where the synthetic data $\hat D$ inherits the bias of the model $M$. Therefore the effectiveness of synthetic trajectories is jointly limited by model bias, long-horizon compounding error, and the mismatch between the generated distribution and the real interaction distribution — it remains fundamentally a distribution problem, not merely "more data."

## Data Interfaces for Different Paradigms

This is an easily overlooked but very important dimension: **different technical approaches don't need the same kind of data.**

### VLA's Data Interface

The most basic VLA training sample can be abstracted as $(o_t, l, a_{t:t+k})$, where $l$ is the language instruction and $a_{t:t+k}$ is an action chunk. But it should be emphasized: **an action chunk is only a common training/inference interface, not the definition of VLA.** The core of VLA is really the $(V, L) \rightarrow A$ mapping, while the $a_{t:t+k}$ chunk form is a concrete policy parameterization — a system that does not explicitly predict chunks can still be a VLA.

But actual systems may also include: proprioception, historical observation windows, task metadata, embodiment information. Action output is also not simply $(o,l) \rightarrow a$ — modern VLAs may use action chunks, diffusion / flow action heads, discrete action tokens or continuous action, and heterogeneous action representations. Representative works transferring large-scale vision-language knowledge to robotic control include RT-2 (Brohan et al., CoRL 2023, arXiv:2307.15818) and π₀ (Black et al., Physical Intelligence, 2024, arXiv:2410.24164).

This means VLA's core data requirement is: **high-quality observation-action pairs, covering sufficiently diverse tasks and objects, while needing to adapt to different embodiment action representations.**

There is a deeper problem here: **action representation itself is part of data interface design.** $a_{t:t+k}$ is not just "action" — it could be joint position, joint velocity, end-effector delta pose, absolute pose, gripper command, discretized tokens, continuous flow, or even latent action. Therefore, the core problem of cross-embodiment is not simply "putting data from different robots into the same dataset," but finding a sufficiently general observation/action representation that enables experiences from different embodiments to be shared in the same learning space.

### World Model's Data Interface

The world model's core interface is:

```
Input: observation history + action history

Core output:
  future latent state / transition distribution
  e.g., p(z_{t+1} | z_t, a_t)

Optional:
  reconstructed observation (p(o_t | z_t))
  reward
  termination
  task outcome
```

It's important to note that **the core of a world model is learning action-conditioned dynamics; reward prediction is not a logically required component of the dynamics model.** But we should not go the other way and dictate that "reward doesn't belong to the world model." A more accurate technical layering is: a `dynamics model` (learning state transitions), a `reward model` (predicting reward), and a `continuation / termination model` (predicting whether the episode continues) — and in some literature and system definitions, these modules together are called the **world model**. So in a concrete model-based RL agent (such as Dreamer), reward and continuation prediction often form, together with the dynamics model, the complete world-model module; they are simply not logically required outputs of action-conditioned dynamics itself.

For approaches centered on latent dynamics + model-based control (such as Dreamer's RSSM, TD-MPC2, Hansen et al., ICLR 2024, arXiv:2310.16828), data needs to be temporally coherent, action-annotated interaction trajectories.

### RL's Data Interface

RL's data requirements depend on the specific paradigm:

- **On-policy** (e.g., PPO): needs data produced by the current policy; data "freshness" matters
- **Off-policy** (e.g., SAC): can reuse historical replay data, thus usually having higher data-reuse capability; but off-policy ≠ automatically more sample-efficient — final sample efficiency still depends on replay distribution, exploration, critic quality, reward structure, and the task itself, and the replay buffer's distribution and coverage affect policy generalization and stability
- **Offline RL**: relies entirely on pre-collected datasets; extremely high requirements for data distribution coverage
- **Imitation + RL**: first pre-train with demonstrations, then fine-tune with online interaction

From a data perspective, different RL paradigms have very different requirements for replay buffer or dataset quality and diversity.

### Data Interface Incompatibility

A commonly encountered practical problem is: **data from different embodiments, different sensor configurations, different action spaces typically cannot be directly used for the same low-level policy without processing.**

Data collected on a Franka arm, due to differences in action space dimensions, observation viewpoints, and dynamic characteristics, typically needs action retargeting, action normalization, or embodiment conditioning before it can be used for other robots.

This is why cross-embodiment data is an important research direction. But terminology precision is needed: **multi-task** (same robot completing multiple tasks), **multi-embodiment** (training data from multiple robots but handled separately), and **cross-embodiment** (model generalizes to unseen robots) are three different levels of problem.

TD-MPC2's multi-task / multi-domain capability is primarily achieved through task embeddings — but task conditioning ≠ embodiment conditioning. Embodiment differences involve action space, observation space, morphology, dynamics, control frequency, and multiple other dimensions, which cannot be simply solved with a task embedding. The π₀ series demonstrates the importance of large-scale, multi-embodiment data for cross-robot generalization (this is an empirical observation, not a causal attribution about the specific mechanism) — Open X-Embodiment (Open X-Embodiment Collaboration, 2023, arXiv:2310.08864) is a representative example of such large-scale cross-embodiment datasets.

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

Here we also need to distinguish diversity from coverage: **Diversity describes how different samples are from each other; coverage describes how much of the target task distribution has been covered.** For example: data of 1000 different cups → high diversity; but if all are "desktop cup grasping" tasks, task coverage may still be very low.

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

Abstracting further, **the value of failure data lies not in "failure" itself, but in the negative / counterfactual information it provides.** A single $(s, a_{\mathrm{bad}}, s')$ tells the model "in this state, this action produces what consequence"; whereas if there are only successful demonstrations $(s, a_{\mathrm{good}}, s')$, the model does not necessarily know why $a_{\mathrm{bad}}$ is bad. It is precisely this counterfactual signal that gives failure trajectories unique value for world models and offline RL — elevating "failure data is useful" from an empirical remark to a clearer learning-theoretic intuition.

### Data Quality ≠ Data Utility

The preceding text repeatedly used words like quality, relevance, success, diversity, and coverage, but they can actually be unified under a more fundamental concept: **Data Utility.**

The core point is: **data quality is not an absolute property, but an objective-conditioned utility.** The same dataset $D$ has different utility for different training objectives $\mathcal{L}$:

$$U(D \mid \mathcal{L})$$

$$U_{\mathrm{IL}}(D) \neq U_{\mathrm{WM}}(D) \neq U_{\mathrm{offlineRL}}(D)$$

This neatly explains the earlier phenomenon: a failure trajectory may be noise to be filtered for imitation learning, yet valuable signal for a world model or offline RL — because its utility differs under different objectives. Strictly speaking, "high-quality data" should really be "data with high utility for the current objective."

Recasting quality as utility also resolves a common misconception in curation: there is no such thing as a "universally good" dataset, only a dataset that is "good for some $\mathcal{L}$." This is why data curation must be defined together with the training objective, rather than discussing "data quality" in isolation from any goal.

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

Given the same dataset $D$, changing the sampling / weighting / objective / schedule ($R_1 \neq R_2$) can yield $\theta_1 \neq \theta_2$. This elevates the "recipe as moat" argument from empirical observation to a clearer technical framework.

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

Here we need to distinguish two different types of sim-to-real error: **random noise** (e.g., sensor noise, small physical parameter fluctuations) and **systematic simulation bias** (e.g., friction coefficient long-term bias, actuator delay, contact model errors, deformable object dynamics errors, camera latency, calibration error). Random noise can be addressed through domain randomization to enhance robustness; while systematic bias is what policies may systematically learn incorrectly, requiring system identification to calibrate the simulator itself. In other words: **domain randomization addresses robustness, system identification addresses simulator bias.**

## Robot Data Scaling: Not Just "More Trajectories"

The LLM domain has established relatively clear scaling laws (more data + larger models + more compute → predictable performance improvement, e.g., Kaplan et al., 2020, arXiv:2001.08361; Hoffmann et al., 2022, arXiv:2203.15556). Does a similar scaling law exist for robotics?

First, an important clarification: **the formulas below are not strict scaling laws, but a conceptual decomposition for describing robot data's effective scale.** Robot data scale can be decomposed into at least three layers:

**Data volume:** $N_{\text{steps}}$ (total interaction steps)

**Distribution dimensions:** $task, scene, embodiment, state, action$ (distribution dimensions)

**Data quality:** $Q$ (data quality)

Therefore, robotics' scaling law may not be:

$$Performance = f(N)$$

but rather:

$$D_{\mathrm{effective}} = f(N,\;Coverage,\;Diversity,\;Q,\;Capacity)$$

$$Performance = g(D_{\mathrm{effective}},\;Model\ Capacity,\;Compute)$$

This means: **what robotics truly needs to scale is not just data volume, but effective data scale — i.e., effective coverage of the interaction distribution.**

It should be emphasized that **effective data scale and model capacity are not independent**: data diversity can only be fully exploited when the model has enough capacity. When model capacity is small, blindly enlarging distribution diversity may yield limited or even negative returns; only when capacity is sufficient can the same diverse data be converted into stronger generalization. So $Performance$ is better understood as jointly determined by $D_{\mathrm{effective}}$, $Model\ Capacity$, and $Compute$, rather than as a function of any single variable.

LLMs can roughly ask "how many tokens do I have?"; robotics should rather ask "how many tasks, states, environments, actions, failure modes, and embodiments have I covered?"

```
Robot Data Scaling ≠ More Trajectories

Effective Data Scale = f(Volume, Distribution Coverage, Quality)
```

This is a hypothesis worth testing: **scaling on interaction distribution (rather than pure trajectory count) may be the more effective scaling direction for robotics.** Robotics does not yet have a single universally-accepted scaling law like LLMs, but there is empirical work on data scale worth referencing — for example, studies on data scaling in imitation learning (Lin et al., 2024, *Data Scaling Laws in Imitation Learning for Robotic Manipulation*, arXiv:2410.18647) show that data effectiveness correlates strongly with environmental/task diversity rather than trajectory count alone.

To make the positioning of this hypothesis clearer, the article's logic can be layered as follows:

> **Known:** data volume, data quality, and task diversity all affect robot learning performance.
>
> **Unknown:** under fixed compute and model capacity, which kind of distribution expansion is most effective?
>
> **This article's hypothesis:** effective interaction-distribution coverage explains data scaling better than raw trajectory count.

### Testable Predictions

If this hypothesis holds, then under fixed training compute and model scale, the following testable predictions can be made:

- The returns from adding repeated trajectories should diminish rapidly;
- Adding new tasks / scenes / embodiments that expand the support of the target task distribution is expected to have higher marginal value than simply repeating existing trajectories (the keyword is *expand the support*, not "new = good" — if a new embodiment's morphology and action semantics closely resemble existing ones, its marginal value may be very low);
- Targeted data addressing failure modes should be more effective than randomly adding data;
- Changes to data mixture and sampling recipes should produce reproducible performance differences.

These predictions are in principle experimentally verifiable, rather than remaining at the level of "data matters" as empirical observation.

## What Does This Mean?

If we connect the threads from previous articles:

- [The world model series](/en/articles/2026-09-01-world-model-h2-review/) established the "prediction interface" concept
- [The VLA series](/en/articles/2026-09-03-vla-deep-dive/) analyzed "semantics + action" interface design
- [RSSM evolution](/en/articles/2026-09-04-rssm-beyond/) discussed different latent dynamics' data requirements
- [The industry landscape](/en/articles/2026-09-06-embodied-ai-landscape/) pointed out data is becoming a key differentiator

What this article wants to say is: **data and training recipes may be becoming embodied AI's most underestimated competitive advantage.**

Model architectures can be disseminated through papers and open-source code; simulation platforms are being standardized by a few players; but **high-quality robot interaction data, effective data curation processes, and repeatedly refined training recipes — these are difficult to fully transmit through a single paper.**

That said, we should more carefully distinguish "advantage" from "moat." Taken alone, any single item may not constitute a real moat: data can be purchased, teleoperation infrastructure can be replicated, training recipes can potentially be reverse-engineered, foundation model capabilities can transfer, and synthetic data may even lower the data barrier itself. So equating "having more data" directly with "having a moat" is not rigorous.

What is genuinely harder to replicate may be closing the whole chain into a **data flywheel**:

$$Data\ Collection \rightarrow Curation \rightarrow Evaluation \rightarrow Training \rightarrow Deployment$$

$$Deployment \rightarrow Failure \rightarrow Data \rightarrow Training \rightarrow Better\ Policy \rightarrow Deployment$$

That is, deployment produces real failures, failures flow back as new targeted data, data drives better policies after curation, which then enter the next round of deployment. Once this loop starts turning, competitors can hardly catch up by merely copying one isolated link — **the moat comes from the flywheel turning, not from any static pile of data.**

And the core question here is not "who has more data," but "who covers a broader interaction distribution $p(\tau \mid task, scene, embodiment)$," and who can keep expanding that distribution through deployment.

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

- DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset — Khazatsky et al., 2024, arXiv:2403.12945 (large-scale, multi-scene real-robot manipulation dataset emphasizing diversity of environment and task distributions)
- SCIZOR: Self-Supervised and Composable Data Curation for Robotic Manipulation — Tian et al., 2025, arXiv:2505.22626 (self-supervised, composable data cleaning / quality filtering)
- Consistency Matters: Revisiting Imitation Learning with Demonstration Quality Metrics — 2024, arXiv:2412.14309 (measuring demonstrations with quality metrics such as consistency, rather than assuming "human demonstration = high quality")
- Efficient Data Collection for Robot Learning via Compositional Generalization — 2024, arXiv:2403.05110 (reducing data collection cost via compositional task generalization)
- Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation — Maddukuri et al., 2025, arXiv:2503.24361 (a systematic study of recipes for mixing simulation and real data)

Note that robotics does not yet have a single universally-accepted scaling law comparable to that of LLMs; the effective-data-scale framework in this article is a conceptual decomposition and a testable hypothesis, not an established conclusion. The data-side works above provide scattered empirical support, not yet a full quantitative validation of that hypothesis.

---

*This article extends the embodied AI series — from "who is doing what" to "what is driving performance." The next article may discuss sim-to-real methodology in detail.*
