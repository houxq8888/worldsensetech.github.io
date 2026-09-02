---
title: "The Data Problem in Embodied AI: Why 'What Data You Use' May Matter More Than 'What Model You Use'"
slug: "2026-09-08-data-and-training-recipes"
date: 2026-09-08
draft: false
categories: ["Embodied Intelligence", "Training Methods"]
tags: ["Embodied Intelligence", "Robot Data", "Training Recipe", "Teleoperation", "Synthetic Data", "Sim-to-Real", "Data Curation", "VLA", "World Model"]
description: "Model architecture differentiation is shrinking; data and training recipe differentiation is growing. But robot data is not simply 'the more the better' — different paradigms need different data interfaces, the data quality vs diversity tradeoff, sim-to-real distribution mismatch, and training recipe design are becoming the true competitive moats in embodied AI."
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

This article wants to expand on that problem: **why is robot data fundamentally difficult? How do different technical approaches have different data requirements? And why might training recipes be more critical than model selection?**

## Why Robot Data Is Not Like Internet Data

The success of large language models is largely built on internet-scale text data — trillion-token pre-training corpora are nearly a public resource. Vision models are similar — large-scale image-text datasets like LAION-5B provide the foundation for VLMs.

But robot data has a fundamental difference: **it's not just "observation," it's "interaction."**

A piece of internet text only needs text; a web image only needs pixels. But a single robot data point typically needs to include simultaneously: multi-view visual observations, proprioception (joint angles, torques, end-effector poses), action commands, language instructions (if applicable), and — in many cases — reward signals or task success annotations.

This means:

```
Internet data:
  observation (text/image) → large scale, low acquisition cost

Robot data:
  observation + action + state + (reward)
  → small scale, high acquisition cost, non-uniform formats
```

This difference is not a detail — it's a fundamental data structure difference. It determines that embodied AI cannot simply replicate LLM's "data scaling" approach.

## Several Data Source Approaches

Currently, data sources for embodied AI can be roughly divided into several categories, each with its own advantages and limitations.

### Teleoperation Data

The most direct source is having humans control robots to complete tasks, recording observation-action trajectory pairs.

**Advantage:** High data quality, directly demonstrating "successful task completion" behavior patterns; naturally contains human manipulation strategies and common sense.

**Limitation:** Slow collection speed, high cost; operator skill level directly affects data quality; task diversity and environmental diversity are limited by the operator's time and imagination.

Current mainstream teleoperation systems include VR controller-based control, SpaceMouse, and vision-based imitation systems. Some companies (such as Physical Intelligence, Figure AI) have established large-scale teleoperation data collection pipelines.

### Autonomous Data

Letting robots autonomously explore in real or simulated environments, collecting successful or failed interaction data.

**Advantage:** Can be collected at scale in parallel; doesn't require continuous human operator involvement.

**Limitation:** In real environments, autonomous exploration is usually very inefficient (random exploration → most data is useless); in simulation, exploration efficiency is higher but sim-to-real gap exists.

RL training typically relies on this type of data — agents repeatedly trial-and-error in environments, collecting on-policy or off-policy interaction data.

### Simulation Data

Generating training data in simulated environments.

**Advantage:** Can be大规模 parallelized, precisely controlled environment parameters, automatic annotation; can generate extreme scenario data difficult to obtain in real environments.

**Limitation:** Sim-to-real gap still exists — physical dynamics in simulation (contact mechanics, friction, deformation) don't perfectly match the real world. The distribution of simulation data and real data have mismatches, and direct use can cause policy performance degradation in real environments.

NVIDIA Isaac Sim, MuJoCo, and other simulation platforms are being widely used to generate training data. But simulation data typically needs to be combined with domain randomization, system identification, or real-world fine-tuning to bridge the gap.

### Synthetic Data: World Models as Data Generators

An increasingly important direction is: **using trained world models to generate synthetic training data.**

This is already a core mechanism in the Dreamer series — RSSM "imagines" trajectories in latent space, then trains actor-critic policies on these imagined trajectories. From a functional perspective, this is using the world model as a data source.

NVIDIA Cosmos's positioning also includes this direction — using world foundation models to generate physical world predictions and synthetic data for downstream policy model training.

**Advantage:** Doesn't require additional real interaction; can learn a world model from limited real data, then use the world model to "amplify" data volume.

**Limitation:** Synthetic data quality depends entirely on the world model's prediction accuracy. If the world model's predictions are inaccurate in certain regions (typically out-of-distribution regions), synthetic data may introduce erroneous signals, causing policies to learn incorrect behavior patterns.

### Internet Pre-training Data

A unique advantage of the VLA approach is the ability to leverage internet-scale pre-training data. RT-2's core approach is fine-tuning a VLM (pre-trained on internet image-text data) as a robot policy — meaning VLA inherits substantial semantic knowledge and visual understanding from internet data.

**But internet data cannot replace robot interaction data.** Internet data provides semantic priors ("what a cup is," "what grasping means"), not manipulation skills ("how to stably grasp a cup filled with water"). This is why all VLA systems ultimately need robot data for fine-tuning.

## Data Interfaces for Different Paradigms

This is an easily overlooked but very important dimension: **different technical approaches don't need the same kind of data.**

### VLA's Data Interface

VLA needs **(observation, instruction, action)** triplets.

```
Input: visual observation + language instruction
Output: action (discrete tokens or continuous vectors)
```

VLA training typically has two phases: first pre-training on internet image-text data (acquiring semantic capabilities), then fine-tuning on robot interaction data (acquiring manipulation capabilities). The fine-tuning phase's data interface is observation + language → action.

This means VLA's core data requirement is: **high-quality observation-action pairs, covering sufficiently diverse tasks and objects.**

### World Model's Data Interface

World models need **action-conditioned trajectories:**

```
Input: historical observations + action sequences
Output: future observations (and/or latent states) + reward
```

Dreamer's RSSM needs complete (observation, action, reward, terminal) sequences to learn latent dynamics. TD-MPC2 needs (observation, action) sequences to learn latent consistency and reward prediction.

This means world models' core data requirement is: **temporally coherent, action-annotated interaction trajectories, covering sufficiently diverse state transitions.**

### RL's Data Interface

RL's data requirements depend on whether on-policy or off-policy:

- **On-policy** (e.g., PPO): needs data produced by the current policy; data "freshness" matters
- **Off-policy** (e.g., SAC): can reuse historical data, but needs sufficient diversity to avoid overfitting

From a data perspective, off-policy methods have higher requirements for replay buffer quality and diversity.

### Data Interface Incompatibility

A commonly encountered practical problem is: **data from different embodiments, different sensor configurations, different action spaces cannot be directly used together.**

Data collected on a Franka arm cannot directly train a UR5 policy — even if the task is completely identical. Action space dimensions differ, observation viewpoints differ, dynamic characteristics differ.

This is why cross-embodiment data is an important research direction — TD-MPC2 handles different embodiment differences through task embeddings, and the Open X-Embodiment dataset attempts to unify data formats across multiple robots. But from a data engineering perspective, "unified format" doesn't equal "unified distribution" — data from different embodiments still has fundamental differences in statistical characteristics.

## Training Recipes: How You Use Data May Matter More Than How Much Data You Have

"Data volume" is an easily quantified metric, but in embodied AI, **how data is used (training recipe) may be more critical than how much data there is.**

### Data Quality > Data Quantity

This has been validated in multiple works. The RT-2 team found that a small amount of high-quality robot demonstration data (a few thousand trajectories) combined with large-scale internet pre-training often outperforms large amounts of low-quality data.

Intuitively this isn't hard to understand: "noise" in robot data isn't just annotation errors — it also includes non-smooth manipulation, suboptimal strategies, sensor noise. This noise gets directly learned during supervised fine-tuning, causing policy quality degradation.

### Data Diversity and Curriculum Learning

Another key dimension is data diversity. If training data only covers one type of cup, one lighting condition, one table surface, the policy will fail when encountering new cups, new lighting, new surfaces.

Curriculum learning is a commonly used strategy: start training from simple tasks, progressively increasing difficulty. This is especially common in simulation — first letting agents learn basic skills in simple environments, then progressively increasing environment complexity.

### Data Mixing Strategies

VLA training typically involves mixing multiple data sources: internet image-text data (semantic capabilities), robot demonstration data (manipulation skills), and possibly synthetic data (data augmentation).

**The mixing ratios of these data sources, training order, and learning rate scheduling constitute the core of the training recipe.** Different teams' choices in these areas can vary significantly, and these choices often have significant impact on final performance — sometimes even exceeding the choice of model architecture.

### Sim-to-Real Adaptation Strategies

Simulation data cannot directly replace real data, but the gap can be bridged through various strategies:

- **Domain randomization:** Randomize visual appearance, physical parameters, environment layout in simulation, making policies more robust to variation
- **System identification:** Precisely calibrate simulator physical parameters, making simulation closer to reality
- **Real-world fine-tuning:** First pre-train in simulation, then fine-tune with a small amount of real data
- **Domain adaptation:** Learn mappings between simulation and reality, reducing distribution differences

These strategies are typically not mutually exclusive — practical systems often combine them.

## Several Directions Worth Watching

### Data Curation Is Becoming an Independent Direction

"More data" doesn't automatically equal "better performance." Data filtering, deduplication, quality scoring, diversity assurance — these "data curation" tasks are becoming an independent technical direction.

For robot data, curation challenges are especially large: how to automatically judge the quality of a teleoperation trajectory? How to measure a dataset's task coverage? How to detect and handle anomalous data? These problems currently have no standardized solutions.

### Cross-embodiment Data: Opportunities and Challenges

Datasets like Open X-Embodiment attempt to integrate data from multiple robots. Ideally, this could let models learn general manipulation principles from multiple embodiments' experiences.

But the practical challenge is: data from different embodiments differs in action space, observation format, and task distribution. Simply mixing this data may not produce the expected generalization effects — the model may spend significant capacity "memorizing" differences between embodiments rather than learning common manipulation principles.

TD-MPC2's task-conditioned approach and π₀ series's cross-embodiment experiments are both exploring how to more effectively utilize multi-embodiment data.

### Data Scaling Laws: Do They Exist for Robotics?

The LLM domain has established relatively clear scaling laws (more data + larger models + more compute → predictable performance improvement). Does a similar scaling law exist for robotics?

From the current situation, this question has no clear answer. The reasons are:

- Robot data "volume" isn't easy to define (number of trajectories? number of timesteps? task diversity? embodiment diversity?)
- Data quality and diversity's impact may be greater than pure data volume
- Scaling behavior may differ completely across different tasks and embodiments

One hypothesis worth testing is: **scaling in data diversity (rather than pure data volume) may be the more effective scaling direction for robotics.**

## What Does This Mean?

If we connect the threads from previous articles:

- [The world model series](/en/articles/2026-09-01-world-model-h2-review/) established the "prediction interface" concept
- [The VLA series](/en/articles/2026-09-03-vla-deep-dive/) analyzed "semantics + action" interface design
- [RSSM evolution](/en/articles/2026-09-04-rssm-beyond/) discussed different latent dynamics' data requirements
- [The industry landscape](/en/articles/2026-09-06-embodied-ai-landscape/) pointed out data is becoming a key differentiator

What this article wants to say is: **data and training recipes may be becoming embodied AI's most underestimated competitive moat.**

Model architectures can be disseminated through papers and open-source code; simulation platforms are being standardized by a few players; but **high-quality robot interaction data, effective data curation processes, and repeatedly refined training recipes — these are difficult to fully transmit through a single paper.**

This is also why I tend to think: in the near term, embodied AI differentiation will come more from "who has better data and training recipes" rather than "who has a bigger model."

---

*This article extends the embodied AI series — from "who is doing what" to "what is driving performance." The next article may discuss sim-to-real methodology in detail.*
