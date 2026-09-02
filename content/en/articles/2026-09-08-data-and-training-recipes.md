---
title: "The Data Problem in Embodied AI: As Architectures Converge, What Determines Performance?"
slug: "2026-09-08-data-and-training-recipes"
date: 2026-09-08
draft: false
categories: ["Embodied Intelligence", "Training Methods"]
tags: ["Embodied Intelligence", "Robot Data", "Training Recipe", "Teleoperation", "Synthetic Data", "Sim-to-Real", "Data Curation", "VLA", "World Model", "Scaling Law"]
description: "As foundation model architectures gradually converge, data distribution, data quality, and training recipes are increasingly becoming important variables determining robot performance. But robot data is not simply 'the more the better' — what robotics truly needs to scale is not just trajectory count, but interaction distribution."
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

This article wants to expand on that problem. The core thesis is: **as foundation model architectures gradually converge, data distribution, data quality, and training recipes are increasingly becoming important variables determining robot performance.** But "data matters more" doesn't mean "more data is better" — what robotics truly needs to scale is not just trajectory count, but interaction distribution.

## Why Robot Data Is Not Like Internet Data

Large language models can leverage internet-scale text and code data for pre-training, with data acquisition scale far exceeding robot real-world interaction data. Vision models are similar — large-scale image-text datasets provide the foundation for VLMs.

But robot data has a fundamental difference: **it's not just "observation," it's "interaction."**

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

The key point here is: observations (RGB / RGB-D / proprioception / force-torque / joint state / end-effector pose) are what robots can typically directly obtain; while true environment state is often not directly observable. Similarly, reward is not a required field for demonstration data — it only appears when training reward models or actor-critic policies.

This difference is not a detail — it's a fundamental data structure difference. It determines that embodied AI cannot simply replicate LLM's "data scaling" approach.

## Several Data Source Approaches

Currently, data sources for embodied AI can be roughly divided into four categories.

### Teleoperation Data

The most direct source is having humans control robots to complete tasks, recording observation-action trajectory pairs.

**Advantage:** High data quality, directly demonstrating "successful task completion" behavior patterns; naturally contains human manipulation strategies and common sense.

**Limitation:** Slow collection speed, high cost; operator skill level directly affects data quality; task diversity and environmental diversity are limited by the operator's time and imagination.

Current mainstream teleoperation systems include VR controller-based control, SpaceMouse, and vision-based imitation systems. Multiple robotics companies are building scaled teleoperation data collection infrastructure, though specific data volumes and coverage are typically not public.

### Autonomous Data: Online vs Offline Distinction

Having robots collect interaction data in real or simulated environments. Here an important distinction needs to be made.

**Online interaction:** The policy currently produces actions in the environment $a_t \sim \pi(\cdot|o_t)$, then obtains new trajectories. Typical problems are exploration efficiency, safety, reset cost, on-policy distribution. Classic RL typically relies on this approach — agents repeatedly trial-and-error in environments.

**Offline data:** Existing replay / demonstration data $D=\{(o_t,a_t,o_{t+1},r_t)\}$, without continuing to interact with the environment.

But in current robot RL practice, pure online RL is not the only paradigm. Increasingly common approaches include: offline RL, demonstration + RL, imitation pretraining + online RL, replay-based RL, and simulation RL + real fine-tuning. Therefore, robot RL data sources are actually diverse — online interaction, offline trajectories, demonstration data, and simulation-generated data are all widely used.

### Simulation Data

Generating training data in simulated environments.

**Advantage:** Can be大规模 parallelized, precisely controlled environment parameters, automatic annotation; can generate extreme scenario data difficult to obtain in real environments.

**Limitation:** Sim-to-real gap still exists — physical dynamics in simulation (contact mechanics, friction, deformation) don't perfectly match the real world. The distribution of simulation data and real data have mismatches, and direct use can cause policy performance degradation in real environments.

NVIDIA Isaac Sim, MuJoCo, and other simulation platforms are being widely used to generate training data. But simulation data typically needs to be combined with domain randomization, system identification, or real-world fine-tuning to bridge the gap.

### Synthetic Data: World Models as Experience Generators

An increasingly important direction is: **using trained world models to expand agent experience.**

But here two different mechanisms need to be distinguished:

**Model-based RL (e.g., Dreamer):** The world model serves as a **latent experience generator**, producing imagined trajectories in latent space. Actor/critic trains in latent imagination: $z_t \rightarrow a_t \rightarrow z_{t+1}$, without needing to generate photorealistic RGB frames.

**Generative world models (e.g., video-generative world models, NVIDIA Cosmos):** Further attempt to generate synthetic data (synthetic observations / videos / trajectories) close to real observations, serving as data sources for downstream training.

Both are "using models to expand experience," but the data forms are completely different. The former is a planning and training interface in latent space; the latter is closer to the traditional sense of "synthetic data generation."

The "world model" here includes two related but different concepts: dynamics models for latent planning / imagination, and generative world models for generating or predicting visual worlds.

## Data Interfaces for Different Paradigms

This is an easily overlooked but very important dimension: **different technical approaches don't need the same kind of data.**

### VLA's Data Interface

The most basic VLA training sample can be abstracted as $(o_t, l, a_{t:t+k})$, where $l$ is the language instruction and $a_{t:t+k}$ is an action chunk.

But actual systems may also include: proprioception, historical observation windows, task metadata, embodiment information. Action output is also not simply $(o,l) \rightarrow a$ — modern VLAs may use action chunks, diffusion / flow action heads, discrete action tokens or continuous action, and heterogeneous action representations.

This means VLA's core data requirement is: **high-quality observation-action pairs, covering sufficiently diverse tasks and objects, while needing to adapt to different embodiment action representations.**

### World Model's Data Interface

The world model's core interface is:

```
Input: (o_{≤t}, a_{≤t})
Output: predicted future latent / observation
Optional: reward / termination / task outcome
```

It's important to note that **world models don't necessarily require reward.** In Dreamer, reward prediction and continuation prediction are important components needed for training actor-critic, but they belong to other modules of the overall agent architecture, not required outputs of the world model itself. The world model's core function is learning action-conditioned dynamics — predicting how future states change given action sequences.

For approaches centered on latent dynamics + model-based control (such as Dreamer's RSSM, TD-MPC2), data needs to be temporally coherent, action-annotated interaction trajectories.

### RL's Data Interface

RL's data requirements depend on the specific paradigm:

- **On-policy** (e.g., PPO): needs data produced by the current policy; data "freshness" matters
- **Off-policy** (e.g., SAC): can reuse historical data, but needs sufficient diversity to avoid overfitting
- **Offline RL**: relies entirely on pre-collected datasets; extremely high requirements for data distribution coverage
- **Imitation + RL**: first pre-train with demonstrations, then fine-tune with online interaction

From a data perspective, different RL paradigms have very different requirements for replay buffer or dataset quality and diversity.

### Data Interface Incompatibility

A commonly encountered practical problem is: **data from different embodiments, different sensor configurations, different action spaces typically cannot be directly used for the same low-level policy without processing.**

Data collected on a Franka arm, due to differences in action space dimensions, observation viewpoints, and dynamic characteristics, typically needs action retargeting, action normalization, or embodiment conditioning before it can be used for other robots.

This is why cross-embodiment data is an important research direction. But terminology precision is needed: **multi-task** (same robot completing multiple tasks), **multi-embodiment** (training data from multiple robots but handled separately), and **cross-embodiment** (model generalizes to unseen robots) are three different levels of problem.

TD-MPC2's multi-task / multi-domain capability is primarily achieved through task embeddings — but task conditioning ≠ embodiment conditioning. Embodiment differences involve action space, observation space, morphology, dynamics, control frequency, and multiple other dimensions, which cannot be simply solved with a task embedding. π₀ series's cross-embodiment capability comes more from large-scale diverse data training, rather than some specific conditioning mechanism.

## Data Is Not a Dataset, It's a Distribution

"Data volume" is an easily quantified metric, but in embodied AI, **data's effective scale cannot be simply measured by trajectory count.**

### Data Quality > Data Quantity

A common observation is that high-quality demonstrations combined with large-scale vision-language pre-training can significantly improve robot policy generalization. But "quality" needs a more precise definition — systematic suboptimal behavior or erroneous actions in demonstrations will change the behavior policy's target distribution; without filtering or weighting mechanisms, these patterns may be learned by the model.

Modern policy learning contains multiple response mechanisms: augmentation, trajectory weighting, filtering, robust loss, advantage weighting, diffusion policy smoothing, etc. But the core challenge remains: **"noise" in robot data is not just annotation errors — it also includes systematic problems like non-smooth manipulation, suboptimal strategies, and sensor noise.**

### Data Diversity and Curriculum Learning: Two Orthogonal Dimensions

Data diversity and curriculum learning are two orthogonal dimensions:

- **Diversity:** How many different situations have I seen? — determines coverage
- **Curriculum:** In what order do I see these situations? — determines optimization path

If training data only covers one type of cup, one lighting condition, one table surface, the policy will fail when encountering variation — this is insufficient diversity. Curriculum learning (simple to complex) is a training strategy that affects the optimization path, not coverage itself. **Diversity determines coverage; curriculum determines optimization path.**

### Data Curation: From Trend to Technical

"More data" doesn't automatically equal "better performance." Data curation can be broken into several operational dimensions:

```
Curation = Quality + Diversity + Coverage + Deduplication + Relevance + Balance
```

For robot data, each dimension has specific technical challenges:

- **Quality:** success rate, action smoothness (jerk), collision-free, action consistency
- **Diversity:** scene diversity, object variety, lighting variation
- **Coverage:** task coverage, failure mode coverage, edge case coverage
- **Deduplication:** deduplicate similar trajectories to avoid overfitting
- **Relevance:** whether data is relevant to the target task
- **Balance:** data ratios across different tasks and scenes

These problems currently have no standardized solutions, but are becoming an independent technical direction.

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

Different teams' choices in these areas can vary significantly, and these choices often have significant impact on final performance — sometimes even exceeding the choice of model architecture. This is also why training recipes are difficult to fully transmit through a single paper — they are an entire engineering practice, not a set of hyperparameters.

## Sim-to-Real: Four Different Strategies

Simulation data cannot directly replace real data, but there are multiple strategies for handling the sim-to-real problem. The logic of these four strategies differs:

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

## Robot Data Scaling: Not Just "More Trajectories"

The LLM domain has established relatively clear scaling laws (more data + larger models + more compute → predictable performance improvement). Does a similar scaling law exist for robotics?

Robot data scale has at least 5 dimensions:

```
D = (D_trajectory, D_task, D_environment, D_embodiment, D_quality)
```

Therefore, robotics' scaling law may not be:

$$Performance = f(N)$$

but rather:

$$Performance = f(N_{steps}, N_{tasks}, N_{scenes}, N_{embodiments}, Q)$$

This means: **what robotics truly needs to scale is not just data volume, but interaction distribution.**

LLMs can roughly ask "how many tokens do I have?"; robotics should rather ask "how many tasks, states, environments, actions, failure modes, and embodiments have I covered?"

```
Robot Data Scaling ≠ More Trajectories

Data Scaling = Volume × Quality × Diversity × Coverage × Embodiment
```

This is a hypothesis worth testing: **scaling on interaction distribution (rather than pure trajectory count) may be the more effective scaling direction for robotics.**

## What Does This Mean?

If we connect the threads from previous articles:

- [The world model series](/en/articles/2026-09-01-world-model-h2-review/) established the "prediction interface" concept
- [The VLA series](/en/articles/2026-09-03-vla-deep-dive/) analyzed "semantics + action" interface design
- [RSSM evolution](/en/articles/2026-09-04-rssm-beyond/) discussed different latent dynamics' data requirements
- [The industry landscape](/en/articles/2026-09-06-embodied-ai-landscape/) pointed out data is becoming a key differentiator

What this article wants to say is: **data and training recipes may be becoming embodied AI's most underestimated competitive moat.**

Model architectures can be disseminated through papers and open-source code; simulation platforms are being standardized by a few players; but **high-quality robot interaction data, effective data curation processes, and repeatedly refined training recipes — these are difficult to fully transmit through a single paper.**

And the core question is not "who has more data," but "who covers a broader interaction distribution."

---

*This article extends the embodied AI series — from "who is doing what" to "what is driving performance." The next article may discuss sim-to-real methodology in detail.*
