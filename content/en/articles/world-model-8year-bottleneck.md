---
title: "World Models: 8 Years and the Same Bottleneck"
slug: "world-model-8year-bottleneck"
date: 2026-08-03
draft: false
categories: ["World Models"]
tags: ["World Models", "Bottleneck", "Sim-to-Real", "Challenges", "Generalization", "DreamerV3", "Embodied AI"]
description: "From 2018 to 2026, world models have yet to fundamentally break through in generalization, Sim-to-Real transfer, and long-horizon prediction. Analyzing where the bottlenecks are and why they persist."
toc: true
related_articles:
  - world-model-2026-trend
  - world-model-good-direction
  - vla-vs-world-model
  - rssm-deep-dive
  - world-model-intro
  - 2026-08-31-world-model-future
aliases:
  - /en/articles/world-model-8year-bottleneck.html
---


After working in the world models direction for over half a year, my biggest takeaway is this: there are plenty of papers, but very few that truly land in practice. Recently I read a 2026 survey from the Chinese Academy of Sciences and several universities that does an excellent job of mapping out eight years of progress. Drawing on my own hands-on experience, I want to discuss a few bottlenecks that still haven't been fundamentally broken.

Let's start with the progress. From Ha and Schmidhuber's original world model concept in 2018, to DreamerV3 achieving strong results across 55 Atari games with a single set of parameters, to the explosion of video generation models like Sora, world models have genuinely moved from "proof of concept" to "proof of technology." Four major technical paradigms — observation-level generation, latent-space modeling, reinforcement-learning-driven approaches, and object-centric representations — are now well established, and application scenarios have expanded from games to robotics, autonomous driving, scientific discovery, and more.

But several core bottlenecks remain. Eight years on, none of them has seen a fundamental breakthrough.

## Bottleneck 1: The Sim-to-Real Gap

This is the most painful bottleneck. World models perform well in simulation — DreamerV3's sample efficiency is 10–100x that of traditional RL. But policies trained in simulation typically suffer a 30–50% performance drop when deployed on real robots.

Domain randomization can mitigate part of the problem, but it is far from a solution. The root cause is that simulators cannot perfectly replicate real-world physical dynamics — contact friction, flexible deformation, sensor noise, unmodeled disturbances. The "laws of the world" a world model learns are the laws of the simulation, not the laws of reality.

Eight years later, there is still no elegant solution. You either spend enormous effort on system identification, or you accept degraded performance.

## Bottleneck 2: Long-Horizon Prediction Error Accumulation

A world model's core capability is "imagining the future." But predictions are never perfect; every step carries a small error, and over multiple steps those errors compound exponentially.

DreamerV3 limits its imagination rollouts to roughly 15 steps; beyond that, predictions become unreliable. For tasks that require long-term planning — such as a robot completing a multi-step assembly task — 15 steps fall far short.

Hierarchical temporal modeling and explicit causal representations are promising directions, but they remain in the research stage with no mature engineering solutions yet.

## Bottleneck 3: Physical Consistency

Today's models can generate visually plausible scenes, yet they frequently violate the laws of physics. Objects pass through each other in generated videos, gravity points the wrong way, momentum is not conserved after collisions.

This is because models learn statistical regularities from data distributions, not physical laws. A model "knows" that a glass shatters when it hits the floor because it has seen that pattern in training data, not because it understands gravity and material mechanics.

Embedding physical constraints — conservation of energy, conservation of momentum, causal rules — into models is one direction, but how to embed them, how much to embed, and whether doing so limits the model's expressive power are questions that have yet to reach consensus.

## Bottleneck 4: Computational Cost

Training a decent world model requires multiple A100 GPUs running for days to weeks. Inference is faster than training, but for real-time control scenarios — such as a robot that needs to make 100 decisions per second — latency remains too high.

Deployment on edge devices is even harder. Industrial robot controllers have limited compute and cannot run large models. Model compression and knowledge distillation can help, but they come at the cost of prediction accuracy.

## Why These Bottlenecks Are Hard to Break

The fundamental reason is this: world models attempt to solve a problem that requires mechanistic understanding using data-driven methods.

Humans understand the world not just through statistical observation, but also through physical intuition, causal reasoning, and abstract modeling. Current models possess only the first of these capabilities; they lack the latter two.

Over the next five years, if we can bridge the gap from "statistical fitting" to "mechanistic understanding," world models can truly become the cornerstone of general artificial intelligence and embodied intelligence. Otherwise, they may remain stuck in the state of *"strong in simulation, underwhelming in the real world."*
