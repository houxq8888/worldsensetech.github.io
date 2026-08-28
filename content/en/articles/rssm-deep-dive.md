---
title: "Deep Dive into RSSM: The Core Engine of World Models"
slug: "rssm-deep-dive"
aliases:
  - /en/articles/rssm-deep-dive.html
date: 2026-08-02
draft: false
categories: ["World Models"]
tags: ["RSSM", "State Space Model", "DreamerV3", "Latent State", "Reinforcement Learning", "World Model", "Dreamer Series"]
description: "Deep dive into RSSM, the core component of Dreamer world models: dual-track deterministic/stochastic design, latent dynamics prediction, KL balancing, and imagination training — with source code walkthrough."
toc: true
related_articles:
  - 2026-08-25-dreamer-explained
  - world-model-transformer
  - td-mpc-world-model-control
  - 2026-08-22-rssm-kl-balancing
  - world-model-intro
  - vla-vs-world-model
---


In the previous article, we covered the basic concepts of world models and the overall architecture of DreamerV3. Some readers asked for a deeper explanation of how RSSM actually works. This article dissects the core component of the Dreamer family of world models.
 

I'll try to make the math clear without being overly formal. After all, our goal is to understand the principles, not prove theorems.
 
## Why We Need State-Space Models
 

Before discussing RSSM, let's step back and ask: why do we need state-space models at all? Can't we just use an RNN or Transformer directly?
 

Take RNNs first. The problem with traditional RNNs is that they compress all information into a single deterministic hidden state vector. This is like asking you to describe "how's the weather today" using a single number — no matter how you choose that number, you'll lose a tremendous amount of information. More critically, RNNs have no ability to express uncertainty. If the environment has stochastic elements (like a gust of wind), an RNN cannot express "I'm not sure what will happen next."
 

Now consider Transformers. The attention mechanism is indeed powerful, but it has two problems: first, computational complexity grows quadratically with sequence length, which is too expensive for robotic tasks requiring long-horizon memory; second, it is essentially an open-loop model with no explicit state transition structure, making it ill-suited for control.
 

State-space models (SSMs) offer a middle ground. They have explicit state transition equations that capture the essence of dynamical systems, while through carefully designed state representations, they balance expressiveness and computational efficiency.
 
## The Core Idea of RSSM: Dual-Track State
 

RSSM stands for Recurrent State-Space Model. Its core innovation is splitting the hidden state into two tracks:
 

Deterministic state h_t (Deterministic State)
 

This state is analogous to the hidden state in a traditional RNN, updated through a GRU (Gated Recurrent Unit):
 
```
h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
```
 

It captures the deterministic regularities in the environment — the parts that can be predicted precisely. For example, "if the robotic arm moves 10 cm to the right, the end-effector's x-coordinate increases by 10 cm" — a deterministic physical relationship.
 

Stochastic state z_t (Stochastic State)
 

This state is sampled from a distribution:
 
```
z_t ~ p(z_t | h_t)     # Prior (during imagination)
z_t ~ q(z_t | h_t, o_t)  # Posterior (during training)
```
 

It captures the uncertainty in the environment — the parts that cannot be predicted precisely. For example, "the friction between two surfaces might be between 0.2 and 0.4" — a fuzzy physical property.
 

Why split them this way? Because the real world is composed of both deterministic laws and uncertain factors. By modeling them separately, the model can more accurately understand the environment.
 
## The Complete State Transition Process
 

Let's walk through the RSSM state transition step by step. Suppose at time t we have an observation o_t (e.g., a camera image) and an action a_t (e.g., joint torques):
 

Step 1: Update the deterministic state
 
```
h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
```
 

We concatenate the previous deterministic state h_{t-1}, stochastic state z_{t-1}, and action a_{t-1}, then pass them through the GRU to obtain the new deterministic state. This step is similar to a traditional RNN update.
 

Step 2: Compute the stochastic state
 

There are two modes here:
 

During training (with real observations), we use the posterior distribution:
 
```
mu_t, sigma_t = Encoder(h_t, o_t)
z_t = mu_t + sigma_t * epsilon,  where epsilon ~ N(0, I)
```
 

During inference (no real observations, during imagination), we use the prior distribution:
 
```
mu_t, sigma_t = Prior(h_t)
z_t = mu_t + sigma_t * epsilon,  where epsilon ~ N(0, I)
```
 

This design is quite clever. During training, the posterior distribution leverages real observations to correct predictions, improving training efficiency. During inference, only the prior distribution is used, allowing the model to make predictions purely based on internal state without requiring external observations.
 

Step 3: Decode outputs
 

With the state (h_t, z_t), we can decode various outputs:
 
```
Observation prediction: o_hat_t = Decoder(h_t, z_t)
Reward prediction: r_hat_t = RewardHead(h_t, z_t)
Discount prediction: gamma_hat_t = DiscountHead(h_t, z_t)
```
 

### Training Loss
 

The RSSM training loss consists of three components:
 

1. Reconstruction Loss
 
```
L_recon = -log p(o_t | h_t, z_t)
```
 

This makes the model's predicted observations as close to the real observations as possible. It drives the model to learn the environment dynamics.
 

2. KL Divergence Loss
 
```
L_kl = KL[q(z_t|h_t,o_t) || p(z_t|h_t)]
```
 

This pushes the posterior distribution close to the prior distribution. The purpose of this loss is to ensure the model performs well during inference (when only the prior is used). Without this constraint, the posterior and prior could diverge significantly, leading to good training performance but inference-time collapse.
 

In practice, the KL divergence is typically weighted with a coefficient, and the "free bits" trick is used — when the KL falls below a certain threshold (e.g., 3 nats), no further penalty is applied, preventing the model from over-compressing information.
 

3. Reward / Discount Loss
 
```
L_reward = -log p(r_t | h_t, z_t)
L_discount = -log p(gamma_t | h_t, z_t)
```
 

This makes the model accurately predict rewards and termination conditions.
 
## Physics-Aware RSSM: An Improvement Direction
 

The standard RSSM treats the state as a "black box" vector, without concern for whether it corresponds to real physical quantities. But in robotic tasks, many states have clear physical meanings — position, velocity, force, temperature, etc.
 

A natural improvement direction is to give the state space physical meaning.
 

Specifically, this could be done by:
 

- Partitioning the state vector into subspaces, each corresponding to a category of physical quantities (position, velocity, force, thermodynamics, etc.) 
- Introducing physical constraints into the state transition equations (e.g., conservation of energy, conservation of momentum) 
- Using physical priors to initialize certain parameters (e.g., knowing that gravitational acceleration is 9.8 m/s^2) 
 

Such a "physics-aware" RSSM offers several benefits:
 

- Better generalization: Physical laws are universal, so policies based on physical modeling transfer more easily to new scenarios 
- Stronger constraints: Physical constraints reduce the degrees of freedom the model needs to learn, accelerating training 
- Better interpretability: Each state dimension has a physical meaning, making debugging and analysis easier 
 

Of course, this also increases model complexity and requires deep physical understanding of the task. Not all scenarios are suitable for this improvement.
 
## Engineering Implementation Notes
 

If you plan to implement RSSM yourself, here are some practical engineering considerations:
 

1. Network Architecture
 

The specific implementation of RSSM in DreamerV3:
 

- Deterministic state h: dimension 1024, updated through a 4-layer MLP GRU 
- Stochastic state z: dimension 32 (note: much smaller than h), parameterized Gaussian distribution through a 2-layer MLP 
- Both prior and posterior networks are 2-layer MLPs, outputting mean and log standard deviation 
 

2. Numerical Stability
 

If the standard deviation of the Gaussian distribution is too small, it can cause the KL divergence to explode; if too large, the model cannot make precise predictions. In practice, the standard deviation is typically clamped:
 
```
sigma = torch.clamp(sigma, min=1e-4, max=1.0)
```
 

3. Gradient Propagation
 

During imagination training, gradients need to backpropagate through the world model to the Actor network. Since imagination trajectories can be long (15 steps), gradient vanishing/explosion must be handled. DreamerV3 uses gradient clipping to address this.
 

4. Experience Replay
 

Training data comes from an experience replay buffer. In implementation, complete episode sequences are typically stored, and 64 sequence segments are randomly sampled during training. The replay buffer size is usually set to around 1 million steps.
 
## Comparison with Other State-Space Models
 

To understand RSSM more comprehensively, let's compare it with several other common state-space models:
 

RSSM vs. Classical Kalman Filter
 

The Kalman Filter assumes linear dynamics and Gaussian noise; RSSM uses neural networks to learn nonlinear dynamics. The Kalman Filter's state transition is matrix multiplication; RSSM uses GRU + MLP. But the core idea is similar: maintain a belief about the environment state and update it through observations.
 

RSSM vs. Transformer
 

Transformers process the entire history sequence through attention; RSSMs compress historical information through recurrent state. Transformer computational complexity is O(n^2); RSSM is O(n). For robotic tasks requiring long-horizon memory, RSSM is more efficient. However, Transformers are stronger at capturing long-range dependencies.
 

RSSM vs. Mamba/S4
 

Mamba and S4 are newer state-space models that improve expressiveness through selective mechanisms and structured parameterization. They are primarily used for sequence modeling (similar to Transformers), while RSSM focuses on control scenarios. The two are potentially complementary — using Mamba's selective mechanism to improve RSSM's state updates is a direction worth exploring.
 
## Summary
 

The core contribution of RSSM is that it provides an effective way to simultaneously model the deterministic regularities and uncertain factors of an environment. Through its dual-track state design, it can handle sequential data like an RNN while expressing uncertainty like a probabilistic model.
 

In DreamerV3, RSSM has been shown to achieve high-level performance across 55 Atari games with a single set of parameters, and to enable efficient policy learning in robotic manipulation tasks. This suggests that the RSSM design captures some essential features of environment modeling.
 

If you're interested in the RSSM implementation, I recommend checking out the official DreamerV3 code (searchable on GitHub). Focus on the RSSM class implementation — the code is not large, but the design is elegant.
 

In the next article, we'll discuss a more macro-level topic: in 2026, will world models be the next big AI trend? From technology trends to career directions — see you next time.
