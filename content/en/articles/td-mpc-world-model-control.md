---
title: "TD-MPC: How World Models Enable Robot Control"
slug: "td-mpc-world-model-control"
date: 2026-08-07
draft: false
categories: ["World Models"]
tags: ["TD-MPC", "Model Predictive Control", "World Models", "Robot Control"]
description: "TD-MPC: How World Models Enable Robot Control - WorldSense Tech Blog"
toc: true
aliases:
  - /en/articles/td-mpc-world-model-control.html
---


In the previous article, we broke down RSSM's dual-track state design and understood how the Dreamer series "imagines" the future in latent space. But RSSM is not the only approach to using world models for robot control. Today we discuss another important route: TD-MPC (Temporal Difference Model Predictive Control).
 

If Dreamer's core idea is to leverage a learned world model to perform imagined rollouts in latent space and optimize the policy via actor-critic methods, then TD-MPC takes a more direct approach: learn a world model, then plan within that model to select the optimal action sequence for execution. Rather than relying on a policy network as the sole decision-making mechanism, it uses the world model for online planning, combined with a learned policy prior to improve search efficiency.
 
## Why We Need Model Predictive Control
 

Let's take a step back. Traditional reinforcement learning methods (such as DQN and PPO) learn a policy pi(a|s) — given the current state, directly output an action. This policy is trained through extensive trial and error, and once training is complete, inference only requires a single forward pass.
 

But this approach has two problems:
 

First, low sample efficiency. The policy requires a large amount of interaction data to learn well, and interaction with real robots is expensive.
 

Second, lack of "foresight." The policy network decides on an action based only on the current state — it doesn't proactively "think about what would happen in the next few steps if I do this."
 

Model Predictive Control (MPC) takes a completely different approach: at each time step, based on the current state, simulate multiple future trajectories within the model, evaluate the cumulative return of each trajectory, and execute the first action of the trajectory with the highest return. Then at the next time step, replan from scratch.
 

This is like playing chess: instead of making a move on intuition alone, you look ahead several moves and pick the best one.
 
## Core Design of TD-MPC
 

The key contribution of TD-MPC (Hansen et al., 2022) is demonstrating that a latent-space dynamics model learned for control tasks can be combined with MPC planning, and that efficient control can be achieved through TD learning objectives. Note that combining MPC with model-based RL is not a new concept — TD-MPC's innovation lies in replacing reconstruction loss with TD targets to train the world model.
 
### Structure of the World Model
 

The core components of TD-MPC include a state encoder, a latent-space dynamics model, a reward prediction model, and a Q-network for value estimation:
 

1. State Encoder
 
```
`z_t = Encoder(o_t)`
```
 
This compresses high-dimensional observations (e.g., images) into a low-dimensional latent state z_t. Unlike RSSM, this latent state does not need to be split into deterministic and stochastic components — TD-MPC uses a deterministic latent state, resulting in a simpler architecture.
 

2. Dynamics Model
 
```
`z_{t+1} = f(z_t, a_t)`
```
 

Given the current latent state and action, predict the next latent state. This is the core of the world model — it has learned the environment's state transition dynamics.
 

3. Reward Model
 
```
`r_t = R(z_t, a_t)`
```
 

Given the latent state and action, predict the immediate reward. With the reward model, planning no longer requires reward signals from the real environment.
 

4. Q-Network (Q-function)
 
```
`Q(z_t, a_t) ≈ long-term cumulative return from a state-action pair`
```
 

The Q-network evaluates the long-term value of latent state-action pairs. At the end of the planning horizon, the Q-network provides bootstrap estimates, avoiding reliance solely on reward accumulation within a limited horizon.
 
### The Planning Process
 

At each control time step, TD-MPC does the following:
 

First, encode the current observation into a latent state z_0 using the encoder.
 

Second, sample K action sequences (each of length H), roll them out in the dynamics model, and obtain K latent-state trajectories.
 

Third, evaluate the cumulative return of each trajectory using the reward model:
 
```
`G_k = sum_{t=0}^{H-1} gamma^t * R(z_t^k, a_t^k) + gamma^H * Q(z_H^k, a_H^k)`
```
 

In practice, Q-values are typically also combined to estimate the value at the end of the planning horizon, avoiding reliance solely on short-term rewards. This is equivalent to adding a "long-range assessment" at the endpoint of the rollout, so that planning considers not just the next few steps but also longer-term consequences.
 

Fourth, select and execute the first action of the trajectory with the highest cumulative return.
 

This process repeats at every time step — this is what's known as "receding horizon control."
 
## TD Learning vs. Reconstruction Loss
 

This is one of the biggest differences between TD-MPC and the Dreamer series.
 

The Dreamer series typically constrains world model learning by predicting observations, rewards, and other signals, ensuring the latent state retains information about environment dynamics. This requires the model to learn a relatively complete representation of the environment.
 

TD-MPC, on the other hand, uses temporal difference (TD) targets for training. The core idea is:
 
```
`Q(z_t, a_t) = R(z_t, a_t) + gamma * max_a' Q(z_{t+1}, a')`
```
 

The model only needs to learn state representations relevant to control — it doesn't need to reconstruct pixels. This brings two benefits:
 

First, more efficient training. No decoder is needed, eliminating a major neural network component.
 

Second, more compact representations. The latent state encodes only control-relevant information, filtering out irrelevant visual details.
 

The trade-off, of course, is that TD-MPC's latent state cannot "generate" interpretable observations the way Dreamer can — it is more of a purely control-oriented representation. This is a common trade-off in model-based control: stronger generative capability vs. more efficient control representations.
 
## Improvements in TD-MPC2
 

Released in 2024, TD-MPC2 introduces several important improvements over the original version:
 

1. Multi-task support. By conditioning on goal vectors, the same model can handle multiple different tasks. This is highly practical in robotic scenarios — a single model can both grasp objects and push away obstacles.
 

2. Larger model scale. TD-MPC2 uses a larger dynamics model and more planning samples, achieving significantly improved performance on complex manipulation tasks.
 

3. Improved sampling strategy. Instead of purely random sampling of action sequences, it combines a learned prior to guide sampling, making planning more efficient.
 

TD-MPC2 achieves very strong results on multiple continuous control benchmarks including Meta-World and DMControl, demonstrating the competitiveness of the "world model + MPC planning" approach.
 
## TD-MPC vs. Dreamer: Comparing the Two Approaches
 

To better understand where TD-MPC stands, let's make a systematic comparison with Dreamer:

| Dimension | Dreamer | TD-MPC |
| --- | --- | --- |
| State Representation | RSSM dual-track state (deterministic + stochastic), can express uncertainty | Deterministic latent state, more compact structure |
| Learning Objective | Observation/reward prediction + KL divergence | TD targets / value learning |
| Control Method | Actor policy network execution | MPC online planning |
| Strengths | High sample efficiency, suitable for long-horizon learning | Strong planning capability, supports constraint handling |
| Weaknesses | Policy generalization depends on training data distribution | High inference cost (requires rolling out multiple trajectories) |

The two are not substitutes — they are complementary. Dreamer leans more toward learning policies through the model, suitable for tasks that require learning from environment interaction and leveraging imagination to improve sample efficiency. TD-MPC leans more toward online planning, with advantages in robotic tasks requiring fine-grained control, real-time planning, and constraint handling.
 
## Key Engineering Implementation Notes
 

If you plan to implement TD-MPC yourself, here are some practical points to keep in mind:
 

1. Planning parameter selection. The number of trajectories K and the trajectory length H are the two most critical hyperparameters. If K is too small, sampling is insufficient; if K is too large, inference becomes too slow. If H is too small, the planning horizon is too short; if H is too large, model error accumulation becomes severe. The TD-MPC series of papers typically uses dozens to hundreds of candidate trajectories, adjusting the planning horizon based on task type.
 

2. Action sampling strategy. Pure random sampling is very inefficient. TD-MPC2 uses a learned prior to guide sampling — first use a small policy network to generate an initial action sequence, then add noise around it for sampling. This is far more efficient than purely random sampling.
 

3. Model capacity. The dynamics model doesn't need to be large. The original TD-MPC paper uses a 3-layer MLP with 512 dimensions per layer. Models that are too large tend to overfit, especially when data is limited.
 

4. Target propagation. The number of propagation steps (n-step) for TD targets affects training stability. If n is too small, the target has high variance; if n is too large, the target has high bias. Typically n=3 to n=5 works well.
 
## The Big Picture of World Model Control Approaches
 

So far, we have seen three main routes for using world models in robot control:
 

Route One: Dreamer series (RSSM + Actor-Critic). Train the policy network in latent space — the policy learns in imagination, then executes in the real world. Suitable for tasks that require learning policies through environment interaction and leveraging model-based imagination to improve sample efficiency.
 

Route Two: TD-MPC series (deterministic world model + MPC planning). Perform online planning in latent space, selecting the optimal action at each step. Suitable for tasks requiring precise planning.
 

Route Three: VLA (Vision-Language-Action models). Typically do not explicitly construct a world model, but instead learn the mapping from perception to action through large-scale vision-language data. Suitable for tasks requiring natural language instruction understanding.
 

These three routes are not mutually exclusive. In fact, the latest research trend is to combine them — for example, using VLA for high-level task planning, TD-MPC for low-level action planning, and Dreamer for exploration and data augmentation. A possible future architecture would be: LLM/VLM handles task understanding, VLA handles vision-language-conditioned action generation, the world model handles future state prediction, and TD-MPC handles real-time control and constraint processing. Each layer has its own role, forming a complete embodied intelligence system.
 
## Summary
 

The core contribution of TD-MPC is demonstrating that you don't need to reconstruct pixels to learn an effective world model for control. Through TD learning objectives, it enables the model to focus on control-relevant state representations rather than the full dynamics of the environment.
 

For robot control, TD-MPC's "planning as control" paradigm has an intuitive advantage: you can inject constraints during planning (such as joint limits and collision avoidance), constraints that are difficult to express explicitly in a policy network.
 

If you're interested in implementing TD-MPC, the official TD-MPC2 implementation is open-sourced on GitHub: [GitHub/nicklash/td-mpc2](https://github.com/nicklash/td-mpc2). The code is well-structured and serves as a great entry point for learning model-based RL and robot control. In the next article, we'll discuss the most critical technology in Sim-to-Real transfer: domain randomization.
