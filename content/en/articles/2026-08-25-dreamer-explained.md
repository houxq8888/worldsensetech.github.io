---
title: "Understanding Dreamer: How World Models Learn to Imagine"
slug: "2026-08-25-dreamer-explained"
date: 2026-08-25
draft: false
categories: ["World Models"]
tags: ["DreamerV3", "World Models", "RSSM", "Reinforcement Learning", "imagination", "Dreamer Series"]
description: "From RSSM architecture to the imagination mechanism: a complete breakdown of how Dreamer builds world models in latent space, generates training data, and optimizes policies — with source code walkthrough."
toc: true
related_articles:
  - rssm-deep-dive
  - 2026-08-27-dreamer-actor-critic
  - 2026-08-28-dreamerv3-training-tips
  - world-model-intro
  - vla-vs-world-model
  - td-mpc-world-model-control
aliases:
  - /en/articles/2026-08-25-dreamer-explained.html
---

> **Dreamer Series · Part 1**
>
> This article covers Dreamer's overall architecture at a conceptual level. If you've already read the [RSSM Code Walkthrough Series](/en/articles/2026-08-19-rssm-code-walkthrough/), this article will help you connect the scattered code details into a coherent architectural understanding.

Dreamer's most important idea is not "training a model that generates future frames," but rather **training a latent world model sufficient to support decision-making, then letting policies learn inside that internal world.**

This article walks through Dreamer's design logic from start to finish, organized around this central thesis.

## 1. What Problem Does Dreamer Solve?

The core loop of reinforcement learning is: agent interacts with environment → receives reward → improves policy. The problem with this loop is that every interaction requires real environment computation—for complex tasks like robot control, this means significant simulation time or even real robot time.

Dreamer's approach is: instead of repeatedly trial-and-erroring in the real environment, first **learn a world model**, then train the policy in the model's "imagination."

This sounds straightforward, but making it work requires solving several key problems: How does the world model represent environment state? How do you learn useful policies from imagination? How do you prevent imagination errors from biasing the policy?

From V1 to V3, Dreamer's evolution can be roughly understood along three directions: better latent representation, more stable imagination learning, and stronger cross-task generalization.

## 2. Why Does Dreamer Primarily Predict in Latent Space?

Before diving into the architecture, let's establish the most important intuition.

Dreamer does not completely avoid predicting observations; rather, it **does not require pixel-level future video as the primary vehicle for policy learning.** Dreamer's key is learning dynamics in latent space that are useful for prediction and decision-making.

Suppose a robot currently sees a 1024×1024 RGB image. If the world model had to precisely predict every pixel of the next frame at each step, enormous computation would be wasted on visual details irrelevant to decision-making. Dreamer's approach is to first compress observations into latent states, then predict the future in latent space. As long as the latent representation retains information relevant to prediction and decision-making, there's no need to require the world model to precisely reconstruct every pixel of future frames.

In other words, **Dreamer doesn't predict what the world "looks like"—it predicts the world state needed for decision-making.**

This is the core starting point for understanding all subsequent design choices.

## 3. Global Architecture Diagram

Dreamer's training can be understood as two loops: **Observe** learns the world model from real environment data; **Imagine** generates future trajectories in latent space and uses them to train policies.

It's important to note that actions in imagination and actions in the real environment follow two different paths: actions in imagination are used to advance the world model to predict the next latent state, while actions in the real environment actually change the environment.

```text
              ┌─────────────────────────────────────┐
              │          OBSERVE Phase               │
              │                                     │
              │   Real Environment                  │
              │        │                            │
              │   observation o_t                   │
              │        │                            │
              │        ▼                            │
              │    Encoder                          │
              │        │                            │
              │        ▼                            │
              │   ┌──────────────────────┐          │
              │   │        RSSM          │          │
              │   │                      │          │
              │   │  deterministic h     │          │
              │   │        +             │          │
              │   │  posterior z         │ ← obs    │
              │   └──────────┬───────────┘          │
              │              │                      │
              │         latent state                │
              └──────────────┼──────────────────────┘
                             │
              ┌──────────────┼──────────────────────┐
              │          IMAGINE Phase               │
              │              ▼                       │
              │   ┌──────────────────────┐           │
              │   │    IMAGINATION       │           │
              │   │                      │           │
              │   │    Prior rollout     │           │
              │   │         ↓            │           │
              │   │    latent states     │           │
              │   └──────────┬───────────┘           │
              │              │                       │
              │       ┌──────┴──────┐                │
              │       ▼             ▼                │
              │     Actor         Critic             │
              │       │                            │
              │       ▼                            │
              │  imagined action                   │
              │       │                            │
              │       ▼                            │
              │  Prior dynamics                    │
              │       │                            │
              │       ▼                            │
              │  next latent state                 │
              │  (continue rollout)                │
              └────────────────────────────────────┘

    Meanwhile, the Actor's policy also executes in the real environment:

              Actor (current policy)
                   │
                   ▼
                 action
                   │
                   ▼
            Real Environment
                   │
                   ▼
             observation
                   │
                   ▼
            RSSM / Posterior
            (update world model)
```

During the Observe phase, Posterior and Prior are constrained through KL loss:

```text
Prior:      p(z_t | h_t)
                ↑
              KL loss (two-sided stop-gradient)
                ↓
Posterior:  q(z_t | h_t, o_t)
```

**Posterior is the "observation corrector" during world model training; Prior is the model actually rolled out during imagination.** This relationship is critical.

## 4. Observe: How Does the World Model Learn?

The Observe phase aims to learn the environment's latent dynamics from real environment data.

```text
Observation o_t → Encoder → o_t^emb
                                ↓
                    RSSM: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
                                ↓
                    Posterior: q(z_t | h_t, o_t^emb)  ← sees observations
                    Prior:     p(z_t | h_t)            ← no observations
                                ↓
                    feature_t = concat(h_t, z_t)
                              → deterministic state + stochastic state
                              → specific dimensions determined by model config
```

Two key components here:

**Posterior sees observations.** It uses real observations to "correct" the latent state—essentially saying: "Given what I see and what I remember, what is the current state most likely to be?"

**Prior doesn't see observations.** It can only rely on past latent states and actions—essentially saying: "Based on my memory alone, what do I predict the next state should be?"

During training, KL loss constrains the distributional difference between Prior and Posterior, and KL balancing controls the learning intensity on both sides, enabling the model to learn reliable posterior representations while also allowing the prior to reproduce this latent dynamics without observations. This way, during the imagination phase (when no observations are available), Prior can work independently.

Using DreamerV3's default configuration as an example, deterministic state (8192 dimensions) and stochastic state (32×64=2048 dimensions) are concatenated to form a 10240-dimensional feature. But this is not a fixed value—dimensions are determined by model configuration.

For code-level details on RSSM's deterministic transition, Block GRU, categorical latent, etc., refer to the [RSSM Code Walkthrough Series](/en/articles/2026-08-19-rssm-code-walkthrough/).

## 5. Imagine: How Does the World Model "Dream"?

The Imagine phase is Dreamer's most core design.

In Dreamer, "imagination" essentially means letting RSSM's prior dynamics continuously predict subsequent latent states without real observations, based on the current latent state and actions produced by the Actor.

```text
Start from posterior latent states corresponding to sampled real observation sequences in replay buffer
         ↓
    Rollout using only Prior (no observations)
         ↓
    h_t, z_t ~ Prior → feature_t → Actor → action_t
         ↓
    feature_t → Critic → value_t
    feature_t → Reward predictor → reward_t
         ↓
    Update Actor and Critic using imagined (reward, value)
```

The reward predictor's role is to predict the corresponding reward from imagined latent states; this way, even without the real environment, imagination can obtain the reward signals needed to train the Actor.

It's important to emphasize: **Actor and Critic parameter updates happen primarily on imagined trajectories, but both the starting points of these imaginations and the world model itself come from data collected in the real environment.**

Dreamer's complete loop is: collect data from real environment → learn world model → start imagination from real data's latent states → train Actor/Critic → Actor returns to real environment to execute → produce new data → continue training world model. It's important to emphasize that Observe and Imagine are not two one-time sequential training phases, but rather alternate repeatedly throughout the entire training process.

This is fundamentally different from "pure model self-training."

## 6. Why Doesn't Training in the "Dream" Immediately Go Off Track?

A natural question is: training policies on trajectories imagined by the model—won't it drift further and further off?

Dreamer uses several mechanisms to control this:

**Finite horizon + value bootstrap.** Dreamer uses a pre-set finite imagination horizon, avoiding unlimited forward rollout of the latent model; at the horizon endpoint, it bootstraps further returns through value estimates. This way, the policy can leverage short-term imagined futures provided by the model, without requiring the world model to take on an infinitely long prediction task.

**KL loss constrains Prior-Posterior consistency.** During training, KL loss constrains the distributional difference between Prior and Posterior, and KL balancing controls the learning intensity on both sides. This allows Prior to approximate Posterior as closely as possible in the state distribution covered by real data, providing more reliable latent dynamics for imagination; however, it cannot eliminate error accumulation in long-horizon rollouts.

**Continuous real data collection.** The Actor executes the current policy in the real environment and continuously collects new data, which enters the replay buffer for further updating the world model and policy. This prevents the world model from "closing itself off" in its own imagination.

Of course, limitations of imagination training do exist. For tasks requiring very long-horizon planning, or environments with very complex dynamics (such as contact-rich manipulation), world model imagination errors can significantly affect policy quality. This remains an active area of exploration in model-based RL.

## 7. What Did V1 → V2 → V3 Actually Change?

### DreamerV1: Demonstrating "Imagination Training" Works

One of DreamerV1's core contributions was demonstrating an important approach: using learned latent dynamics to generate imagined trajectories, and training Actor-Critic on those trajectories, can effectively solve continuous control tasks.

### DreamerV2: Introducing Categorical Latent

V2 changed RSSM's stochastic state to categorical latent and used a straight-through estimator, enabling discrete latent to participate in end-to-end training. Combinations of multiple categorical variables provide a compact discrete representation of state, and this discrete latent representation is an important component of V2's improvements on visual control tasks like Atari.

### DreamerV3: Toward Stronger Generalization

V3 made several important engineering improvements, each addressing a specific problem:

| Improvement | Problem Addressed |
|-------------|------------------|
| **symlog** | Reward/value numerical scale differences across tasks |
| **unimix** | Categorical distribution becoming too sharp too early |
| **KL balancing** | Prior/posterior learning imbalance |
| **free_nats** | KL term overly influencing latent learning |
| **Block GRU + large deter** | Improving deterministic memory capacity while controlling compute cost |

Specific details:

* **symlog prediction**: Performing scale compression on reward, value, and other numerical targets, enabling the model to more stably handle widely varying numerical ranges across different tasks, thereby reducing dependence on task-specific reward/value scaling.
* **unimix**: Mixing a small amount of uniform distribution into the categorical distribution; DreamerV3's default configuration uses `unimix=0.01`, used to prevent the categorical distribution from becoming too sharp too early.
* **KL balancing and free_nats**: KL balancing controls gradient contributions from both prior and posterior sides; free-nats limits the effective optimization pressure on KL terms, preventing KL terms from continuing to dominate latent representation learning in regions where they are already small. For the specifics of DreamerV3's implementation, the computation location and stop-gradient approach of free-nats deserve separate discussion, which is also covered in detail in the [RSSM Series](/en/articles/2026-08-22-rssm-kl-balancing/).
* **Larger deterministic state + Block GRU**: DreamerV3 uses a larger deterministic state (default `deter=8192`), and adopts Block GRU structure to control the computational cost brought by large hidden states.

These improvements enable DreamerV3 to demonstrate strong generalization across many different types of tasks, making it a highly representative open-source implementation for research on world models and model-based RL.

## 8. Dreamer's Position in the World Model Landscape

World models are a broad concept, and Dreamer represents one specific approach. The table below compares several major paradigms:

| Paradigm | Representative | Core Idea |
|----------|---------------|-----------|
| **Latent dynamics + RL** | DreamerV1/V2/V3 | Learning dynamics in latent state space, optimizing policies through imagination |
| **Generative / interactive world model** | Genie, etc. | Learning to generate/predict future environment states |
| **Video / diffusion world model** | DIAMOND, etc. | Generating future observations in visual space |
| **VLA policy** | RT-2, OpenVLA, π0 | Directly learning vision/language-to-action mappings |

It's worth noting that VLA and the first three are not strictly the same classification dimension; this categorization is primarily used to build intuition and is not a strictly mutually exclusive taxonomy.

Dreamer's distinctive feature is that it doesn't directly predict pixel-level future frames, but instead predicts future states in a compressed latent space. This typically can significantly reduce the representational space cost of prediction and planning, but the trade-off is that latent representations may not retain all pixel-level details.

For comparison of different architectural approaches, see [World Model Architecture Evolution: RSSM, Transformer, and Unified World Models](/en/articles/world-model-transformer/).

## 9. An Intuitive Example

To make the "world model → imagination → policy" pipeline more concrete, consider a simple scenario:

```text
Robot sees: "wall ahead"

Posterior (sees observation):
  "I now know there's a wall ahead"

Prior (no observation, just memory):
  "Based on the past few steps, I predict there's still likely a wall ahead"

Imagine (rollout future with Prior):

  Action: go left
    latent state → reward/value
    "Higher future returns"

  Action: go right
    latent state → reward/value
    "Lower future returns"

Actor:
  "Then I choose to go left"
```

Note that it's not just "Prior knows which side is safe"—rather, each component has its specific role:

**Prior is responsible for "predicting what will happen," Reward model is responsible for "what reward will I get," Critic is responsible for "how valuable is this future in the long run," and Actor is responsible for "choosing what to do."**

```text
World Model   → What will happen?
Reward Model  → What reward will I get?
Critic        → How valuable is this future?
Actor         → What should I do?
```

This is Dreamer's most core division of responsibilities.

This is Dreamer's core loop: Posterior obtains reliable latent from real observations → Prior learns to predict latent without observations → latent dynamics supports imagination → Reward/Critic evaluate imagined futures → Actor learns from evaluations → new policy returns to real environment to collect data → world model continues improving.

## 10. Connecting the Articles

At this point, the world model content on this blog forms the following reading path:

```text
World Model Intro → RSSM Deep Dive → RSSM Code Series (6 parts)
                                              ↓
                                     This article: Dreamer Architecture
                                              ↓
                                Dreamer Training Tips → GPU Setup
```

If you're new to world models, start with [What Is a World Model in Robotics?](/en/articles/world-model-intro/).

If you want code-level RSSM dissection, the [RSSM Code Walkthrough Series](/en/articles/2026-08-19-rssm-code-walkthrough/) goes from stochastic state all the way through KL balancing and imagine reset.

If you're interested in practical training issues, [DreamerV3 Training Tips](/en/articles/dreamerv3-training-tips/) summarizes hands-on experience from environment setup to hyperparameter tuning.

## 11. Summary: What Does Dreamer Actually "Dream" About?

Dreamer's "dream" is not about predicting a future video, but rather enabling the agent to perform low-cost counterfactual trial-and-error in a learned latent world:

> "If I take this action now, what might happen next? Is that future worth pursuing?"

The world model is responsible for prediction, the Critic for evaluation, and the Actor for selection. The real environment continuously provides new data to correct this internal world.

**So Dreamer's core is not "how vividly it imagines," but "whether its imagination is sufficient to support correct decisions."**

In the next article, we'll dive deeper into Dreamer's Actor-Critic design—how they work in imagination space, and why the symlog transformation is so important for value learning.
