---
title: "What Does a World Model Actually Do in a Robot? From Perception to Action"
slug: "2026-08-26-world-model-in-robotics"
date: 2026-08-26
draft: false
categories: ["World Models"]
tags: ["World Model", "Robotics", "DreamerV3", "RSSM", "Perception", "Sim-to-Real"]
description: "From sensors to actuators: the complete pipeline of world models in robotic systems — perception fusion, latent dynamics prediction, policy learning, Sim-to-Real, and why different robots need different world models."
toc: true
---

> **Dreamer Series · Part 2**
>
> [Part 1](/en/articles/2026-08-25-dreamer-explained/) broke down Dreamer's overall architecture. This article zooms out one step further: where exactly does a world model sit in a robotic system? What happens between sensor data and the final action? Rather than discussing how to train a general-purpose robot world model, this article focuses on its functional position within robotic systems.

## 1. A Fundamental Difference: Robot World Models Are Nothing Like Language Models

Over the past few years, the word "model" has been used repeatedly across AI. Language models predict the next token, vision models predict the next frame, autonomous driving models predict the behavior of traffic participants. They all do some form of "prediction," but the objects and constraints differ enormously.

Robot world models face a fundamentally different problem: **they need to learn action-conditioned future dynamics** — given the current observation and action, predict what future states, observations, and task-relevant outcomes may appear. Note the use of "observations" rather than "states" here, because different types of world models predict different things: Dreamer RSSM predicts in latent space (latent → latent), video world models predict in pixel space (action + video → future video) — "state" is not a universal interface.

Imagine a scenario: a robotic arm on a desk wants to push a cup. Before actually executing, it needs to predict: after my hand contacts the cup, which direction will it slide? What if the cup is too heavy to push? What if the desk is slippery — will the cup fly off?

This kind of prediction is not semantic knowledge like "cups usually sit on tables" learned from text. It's action-conditioned future prediction like "my hand contacting the cup at this angle with this force will cause it to slide about 10cm across the desk."

I think this is the most important starting point for understanding robot world models. Language models learn language structure from massive text, vision models learn visual patterns from massive images, but robot world models need to learn the relationship between actions and consequences from interaction experience.

But here's an important distinction: a world model doesn't need to explicitly recover complete physical laws. It may only need to predict in latent space:

```text
latent_t + action
        ↓
latent_{t+1}
        ↓
Future task-relevant outcomes
```

Without actually knowing:

```text
friction coefficient = 0.31
mass = 247g
contact force = 1.8N
```

**A world model is not about "reconstructing the world" — it's about predicting action-relevant futures.** This thread will run through the entire article.

## 2. What Exactly Does a Robot World Model Predict?

In one sentence: a world model learns "if I execute a certain action, how will the decision-relevant state change."

Technically, a world model learns state transition dynamics, and typically additionally learns task-related prediction heads for reward, continuation, etc.:

```text
WorldModel:
  dynamics:  (s_t, a_t) → s_{t+1}
  heads:     s_{t+1} → r_t, γ_t, ...
```

In practice, world models typically don't predict the complete raw state directly, but instead predict latent dynamics in a hidden space.

Looks simple, but several key questions in this definition become particularly complex in robotics.

**What is "state"?** For a language model, state is a token sequence. For a robot, state comes from multiple heterogeneous sources: camera images (high-dimensional vision), joint angles and torques (low-dimensional proprioception), possibly tactile sensors, depth cameras, and more. The world model needs to fuse all this information into a unified state representation before it can predict.

**What does the world model need to maintain?** A world model needs to maintain an internal representation sufficient to support future prediction and decision-making. This internal representation may encode the robot's own state, environment information, historical context, and latent variables that cannot be explicitly explained but are helpful for prediction — it is closer to a belief state in a POMDP than a simple compressed physical state. For example, a robotic arm needs to predict "where will the cup slide after I push it," and a mobile robot needs to predict "will the obstacle ahead move." Note that a latent world model is not a compressed physical state estimator — it's more like an "information container sufficient for prediction," which may include latent variables that don't directly correspond to physical quantities.

**How reliable does the prediction need to be?** Language models can tolerate occasional "hallucinations" — generating text that's not quite accurate but fluent. But for a world model, the key is not predicting every future pixel with extreme accuracy, but reliably predicting the future differences that actually matter to the policy. Predicting the grip force is sufficient when it's actually not, and the object slips — this requirement for decision-relevant prediction quality is a key difference between robot world models and general generative models.

## 3. From Sensors to Actions: The Complete Pipeline

Now that we understand what the world model predicts, let's look at where it sits in a robotic system.

It's important to clarify first: **a world model is not a mandatory middle layer in robotic systems.** Model-free RL can go directly from observation → policy → action, without any world model. A world model is a model-based component that enables policies to learn and plan in "predicted futures," not a standard requirement for all robotic systems.

A typical world-model-based robotic system follows this complete pipeline:

```text
                 ┌─────────────────────┐
                 │   Perception Module  │
                 │                     │
                 │  Camera → Encoder    │
                 │       ↓              │
                 │  Proprio → Encoder   │
                 │       ↓              │
                 │  latent state (s_t)  │
                 └──────────┬──────────┘
                            │
               ┌────────────┴────────────┐
               ↓                         ↓
         World Model           Direct Policy
               ↓                         │
      imagined futures                   │
               ↓                         │
       Policy / Planner ◄────────────────┘
               ↓
            Action
               ↓
         Real Robot
               ↓
         Observation
               │
               └────────→ Back to perception
```

Note the key structure in this diagram: after perception provides the latent state, there are two paths leading to decision-making. One goes through the world model: first predicting imagined futures, then having the Policy/Planner make decisions based on those predictions. The other is a direct policy path (like model-free RL), outputting actions directly from the latent state. **A world model augments decision-making; it is not a mandatory path.**

At the same time, world model training runs a parallel imagination path:

```text
Sample posterior latent state from replay buffer
         ↓
    Roll out with Prior only (no observations)
         ↓
    (h_t, z_t) → Prior → (h_{t+1}, z_{t+1})
         ↓
    Actor samples action_t
         ↓
    Reward predictor → r_t
    Continuation predictor → γ_t
    Critic → V(h_{t+1}, z_{t+1})
         ↓
    Update Actor and Critic with imagined (r_t, γ_t, V_t)
```

Continuation is used to determine whether the imagined trajectory is still within a valid rollout range — at episode boundaries, γ_t decays to 0, preventing imagination from extending beyond reasonable bounds.

Here's a key distinction: **during Observe, real observations correct the latent state; during Imagine, there are no real observations.** **The key point of Imagine is not "re-simulating the real world," but starting from latent states that have already been corrected by real observations and letting learned dynamics unfold the future on their own.** This is not a pixel-level mental simulation — it is a latent rollout. For details on the Observe and Imagine mechanisms in Dreamer, see [Understanding Dreamer](/en/articles/2026-08-25-dreamer-explained/).

The relationship between these two paths: the real environment provides data to learn the world model, the world model provides imagination space to train the policy, and the policy executes in the real environment to produce new data. Observe and Imagine alternate repeatedly throughout the entire training process.

Several key design choices appear in this pipeline:

**Perception fusion.** In robotic systems, a common approach is to encode vision, proprioception, touch, and other sensors separately, then fuse them in latent space. Dreamer-style RSSM can further compress this information into a latent state for dynamics prediction.

**Action conditioning.** The world model's prediction needs to know explicitly "which action is being evaluated." For multi-DoF robots, different joint action combinations dramatically affect state transitions, and the world model must accurately condition its predictions on the action.

**Imagination starting point.** Imagination trajectories don't start from random states, but from posterior latent states corresponding to real observations sampled from the replay buffer. This ensures imagination trajectory starting points lie on the real data distribution.

### Prediction ≠ Planning

Finally, it's worth emphasizing: **a world model predicting the future is not the same as a world model completing planning on its own.**

```text
Current state s_t

        ├── action A → predicted future A → value 0.8
        ├── action B → predicted future B → value 0.4
        └── action C → predicted future C → value 0.9

                         ↓

                    choose C
```

In this example: the world model answers "what happens if I take this action?"; reward/value answers "is this future good?"; actor/planner answers "so what should I do?" **Prediction and planning are separate** — the world model provides predictions, and the policy/planning module makes decisions based on those predictions. This is also the starting point for the Actor-Critic design discussion in the next article.

### Why Do Robots Need World Models?

Back to a fundamental question: why do robots need world models?

The biggest problem with real robots is not that they can't learn, but that **every failure is expensive.**

If reinforcement learning relies entirely on real interaction, it requires massive trial and error:

```text
1 million attempts
× mechanical wear
× human supervision
× safety risks
```

The cost is enormous. The value of a world model is shifting this process into imagination space:

```text
Small amount of real experience
        ↓
Learn dynamics
        ↓
Massive latent imagination
        ↓
Train policy
```

It transforms robot learning from "trial and error in the real world" to "practice first in a learned world." This is also one of the core contributions of the Dreamer series — dramatically reducing dependence on real interaction through imagined trajectories.

## 4. Different Robots, Different Requirements

"Robot" is a broad category. Different types of robots have very different world model requirements.

### Robotic Arm Manipulation

Robotic arms have relatively fixed DoF (typically 6-7), limited workspace, and usually operate in structured environments (desks, known objects). The main challenge is precise prediction of **contact dynamics** — pushing, grasping, and placing involve complex rigid body contact and friction.

What the world model needs: high-precision short-term contact prediction (a few steps to a few dozen). The Dreamer series has validated the effectiveness of latent dynamics + imagination RL in continuous control benchmarks, but real robotic arm manipulation still faces challenges in contact modeling, visual closed-loop control, and sim-to-real transfer.

### Mobile Robot Navigation

Mobile robots (AGVs, drones, etc.) need to navigate larger spaces, handle dynamic obstacles, and often deal with noisy, incomplete sensor data. The main challenges are **long-term prediction** and **uncertainty management** — "if I go that way, what will I see a few seconds later?"

What the world model needs: prediction capability over longer time horizons, and reasonable expression of "I don't know what will happen." RSSM's stochastic state allows the model to retain information that cannot be fully determined by the deterministic history, giving it an advantage in partially observable tasks with multiple possible future evolutions. This stochasticity provides latent-level stochastic modeling capability and is not equivalent to a calibrated future probability prediction system. For details on RSSM's dual-track state design, see [RSSM Deep Dive](/en/articles/rssm-deep-dive/).

### Humanoid Robots

Humanoid robots have dozens or even more degrees of freedom, and maintaining gait balance is itself a continuous control problem, involving complex ground contact. The main challenge is modeling **high-dimensional dynamic systems** — the world model needs to simultaneously predict the robot's own high-dimensional state and the external environment's response.

What the world model needs: efficient representation capable of handling high-dimensional state spaces. DreamerV3 uses Block GRU to process large-scale deterministic state in blocks, reducing the computational cost of recurrent computation. This structure demonstrates how to improve computational efficiency in large-scale latent dynamics modeling, and is instructive for future high-DoF robotic systems. For details on Block GRU implementation, see [RSSM Code Walkthrough (3)](/en/articles/2026-08-21-rssm-deterministic-core/).

### Dexterous Manipulation

Dexterous hand manipulation is one of the most challenging scenarios in robot manipulation today. Multi-fingered hands can have 20+ DoF, and combined with object shape uncertainty, contact dynamics become extremely complex. The main challenges are **fine force control** and **deformation prediction** — tasks like unscrewing caps, flipping objects, or manipulating flexible materials require the world model to capture subtle force-deformation relationships.

What the world model needs: modeling capability for fine contact and deformation. This is currently a weak point for world models — for scenarios involving fluids, soft bodies, or complex contact, prediction accuracy is not yet sufficient.

These four types illustrate a core tension in world model design: the higher the robot's DoF and the more uncertain the environment, the larger the prediction space the world model needs, and the higher the requirements for accuracy and generalization.

## 5. A World Model Is Not Another Simulator: Its Relationship to Simulators

Discussing robot world models inevitably leads to simulators.

Real robot interaction is expensive — hardware wear, time consumption, safety risks. Simulators (MuJoCo, Isaac Sim, Gazebo) provide a low-cost alternative environment. But the relationship between simulators and world models is more complex than it appears on the surface.

There are currently three main patterns:

**Pattern 1: Physics model-driven.** Train policies directly in a physics simulator (model-free RL), then transfer to real robots via sim-to-real techniques. Here "model-free" means the policy learning process doesn't use an environment dynamics model, not that a training environment doesn't exist — the simulator itself is the environment. Domain Randomization is the representative method — randomizing physical parameters (friction, mass, latency, etc.) in simulation so the policy learns robustness to parameter changes.

```text
Physics Simulator → RL → Sim-to-Real → real robot
```

**Pattern 2: Data-driven world model.** Learn a world model from real or simulated interaction data, train the policy through imagination in the world model, then transfer to the real environment.

```text
Real / Sim Data → World Model → Imagination → Policy → real robot
```

**Pattern 3: Hybrid.** First use the simulator to generate synthetic data, pre-train in the world model, then adapt and fine-tune with real interaction data. This pattern is increasingly common, because real robotics research is likely moving more and more toward a physics + learned dynamics fusion, rather than a physics vs world model either-or.

```text
Physics Simulator → synthetic data
                         ↓
                   World Model
                         ↑
Real data ───────────────┘
         ↓
     adapted World Model
         ↓
       Policy
```

From a training mechanism perspective, Dreamer can treat RSSM as an "internal simulator" for latent imagination; but it is not an explicit physics simulator like MuJoCo / Isaac Sim, nor does it aim to reproduce all environment states. The fundamental difference is:

```text
Physics Simulator
    Goal: simulate the physical world as accurately as possible
    Level: physical state → physical state

World Model (e.g., RSSM)
    Goal: learn future predictions useful for decision-making
    Level: latent state → latent state
```

They can even be combined: the physics simulator provides data distributions, and the world model learns abstract dynamics better suited for decision-making. This is also the core idea behind Pattern 3 (Hybrid).

These three patterns are not mutually exclusive. For choosing between MuJoCo and Isaac Sim, see [MuJoCo vs Isaac Sim](/en/articles/mujoco-vs-isaac-sim/); for Sim-to-Real transfer methods, see [Sim-to-Real Adaptive Transfer](/en/articles/sim-to-real-transfer/).

The core distinction behind these three patterns is: **a simulator provides a world designed by humans; a world model learns a world constrained by data.** Of course, the world model is still subject to constraints from model architecture, data distribution, and training objectives — it doesn't create something from nothing, but rather learns the dynamics that matter for decision-making under data guidance.

## 6. Sim-to-Real: The World Model's Last Mile

No matter how well a world model trains in simulation, it ultimately needs to work on a real robot. This raises the Sim-to-Real problem.

For robot world models, a core question is: **should you transfer the policy, or the world model?**

```text
Approach A: Transfer policy
Sim → Policy → Deploy on Real Robot

Approach B: Transfer world model
Sim → World Model + Policy → Deploy on Real Robot

Approach C: Joint transfer
Sim → World Model
Real data ───────┘
       ↓
 adapted World Model + Policy → Deploy on Real Robot
```

**Approach A** is the most traditional: train a policy in simulation, deploy directly on the real robot. The problem is that sim-to-real gaps can cause the policy to fail.

**Approach B** World model pretraining + adaptation: pre-learn dynamics from simulation data, then adapt and fine-tune with real data, then train and deploy the policy. This is the current research trend — directly transferring a sim world model to real often has limited effectiveness due to large dynamics gaps; pretraining + adaptation is a more practical path.

**Approach C** Joint sim-real learning: simultaneously leverage simulation and real data during training, letting the world model continuously adapt between both. This is the future direction — Dreamer's alternating Observe-Imagine cycle naturally supports this approach.

Several common approaches address these challenges:

System Identification: Calibrate simulator parameters through experiments to make simulation match reality as closely as possible. This is the most traditional approach, but for complex scenarios (like contact-rich manipulation), precise modeling is very difficult.

Domain Randomization: Instead of trying to precisely match real parameters, randomize various parameters in simulation so the policy/world model learns robustness to parameter changes. For detailed domain randomization techniques, see [Domain Randomization: The Bridge from Simulation to Reality](/en/articles/domain-randomization-sim-to-real/).

Online Adaptation: After deployment, continuously update the world model with real environment data, letting it gradually adapt to real dynamics. Dreamer's alternating Observe-Imagine cycle naturally supports this — new data from each real interaction can update the world model, and then the policy re-trains through imagination in the adapted world model. This corresponds exactly to Approach C above.

## 7. World Models in Relation to Other Methods

World models are not the only approach to robot learning. Understanding their relationship with other methods helps determine when to use a world model.

**Relationship to Model-Free RL.** Model-free methods like PPO and SAC learn policies directly from interaction without an explicit environment model. The advantage is theoretically approximating any policy; the disadvantage is low sample efficiency. World model methods can be understood as: first learn a model, then use imagined data from the model to assist policy learning. They're not a replacement but complementary.

**Relationship to Imitation Learning.** Imitation Learning (IL) learns from expert demonstrations without requiring a reward function. But pure IL has a generalization problem — it struggles when encountering situations not covered in training data. A world model can provide "imagination" capability: when facing new situations, predict the consequences of different actions through the world model, rather than only reproducing expert behavior.

**Relationship to VLA.** VLA (Vision-Language-Action) models like RT-2, OpenVLA map directly from vision and language instructions to actions. VLA's advantage lies in leveraging large language models' visual understanding and instruction-following capabilities. The core training objective of typical VLAs is to predict actions directly from vision, language, and robot state, rather than explicitly learning an action-conditioned dynamics model available for rollout. World models provide the "what happens if I do this" prediction capability. A possible future combination is to let VLA provide semantic conditions and task-level behavioral priors, while the world model handles action-conditioned dynamics prediction and evaluates/plans over local futures. The key challenge of this combination is establishing a stable interface between task goals in language space and physical predictions in latent dynamics space. Their combination is, at its essence, bridging the gap between "knowing what to do" and "knowing how to do it." For a detailed VLA vs world model comparison, see [VLA vs World Models](/en/articles/vla-vs-world-model/).

### An Important Misconception: World Model ≠ Dreamer

Finally, an important distinction. This article has used many Dreamer / RSSM examples, but **Dreamer is just one specific implementation of "latent world model + imagination-based RL," not the world model itself.** Robot world models are a much broader concept:

```text
Robot World Model
        │
        ├── latent dynamics
        │      ├── RSSM / Dreamer
        │      ├── Transformer world model
        │      └── other latent models
        │
        ├── pixel / video prediction
        │
        └── action-conditioned predictive models
```

In upcoming articles, we'll discuss TD-MPC (model predictive control in world models) and the Transformer route, which are equally important relationships to world models. Don't equate World Model with RSSM or Dreamer.

## 8. Connecting the Article Cluster

This article maps out where world models sit in robotic systems. Combined with other content on this blog, here's the reading path:

```text
What is a World Model (conceptual intro)
        ↓
RSSM Math Deep Dive → RSSM Code Series (6 parts)
        ↓                    ↓
Dreamer Architecture    World Model Representations
        ↓                    ↓
Robot World Models (this article) ←→ Simulator Comparison
        ↓                    ↓
TD-MPC / VLA          Sim-to-Real
        ↓
World Models + VLA Fusion
```

Related articles by topic:

**Introductory concepts:** [What Is a Robot World Model?](/en/articles/world-model-intro/) starts from "robots also need imagination" to introduce the basic concept and DreamerV3 overview.

**RSSM deep analysis:** [RSSM Deep Dive](/en/articles/rssm-deep-dive/) covers the math; [RSSM Code Walkthrough Series](/en/articles/2026-08-19-rssm-code-walkthrough/) breaks down stochastic state, deterministic core, KL balancing, and imagine reset at the code level.

**Dreamer architecture:** [Understanding Dreamer](/en/articles/2026-08-25-dreamer-explained/) explains Dreamer's complete Observe → RSSM → Imagine → Actor/Critic design at the architecture level.

**Representations and architecture:** [Four Paradigms of World Model Representations](/en/articles/world-model-representations/) compares four representation routes; [World Model Architecture Evolution](/en/articles/world-model-transformer/) discusses RSSM and Transformer fusion trends.

**Control methods:** [TD-MPC: How World Models Enable Robot Control](/en/articles/td-mpc-world-model-control/) introduces another important route — model predictive control in the world model.

**Simulation and transfer:** [MuJoCo vs Isaac Sim](/en/articles/mujoco-vs-isaac-sim/) for simulator selection; [Domain Randomization](/en/articles/domain-randomization-sim-to-real/) for transfer techniques; [Sim-to-Real Adaptive Transfer](/en/articles/sim-to-real-transfer/) for world model-driven transfer.

**Future directions:** [World Models as Synthetic Data Engines for VLA Training](/en/articles/world-model-synthetic-data-for-vla/) discusses combining world models with VLA.

## 9. Summary

The core work of a robot world model can be summarized in one sentence: **learn the relationship between actions and consequences, then train policies through prediction.**

Returning to the thread from the beginning: a world model is not about "reconstructing the world," but about predicting action-relevant futures. In a complete robotic system, this goal is achieved through three core questions:

**1. What is happening now?** → The perception module turns sensor data into a latent state.

**2. If I do this, what will happen?** → The world model predicts multiple possible futures in latent space.

**3. Which future is worth pursuing?** → Reward/value evaluates these futures, policy/planner makes the choice.

```text
         What is it now?
              ↓
          Perception
              ↓
         latent state
              ↓
     What happens if I do this?
              ↓
        World Model
              ↓
      Multiple possible futures
         ↙    ↓    ↘
       A      B      C
       ↓      ↓      ↓
     value  value  value
       ↘      ↓      ↙
          Policy
             ↓
          Action
             ↓
        Real Robot
             ↓
      New Observation
             ↺
```

This world model doesn't exist in isolation. Upstream it connects to the perception module (turning sensor data into usable state representations), downstream to the policy/planning module (turning predictions into actions), alongside simulators providing training data, and Sim-to-Real techniques transferring learned capabilities to real hardware.

Different robot types have different world model requirements — robotic arms need precise contact prediction, mobile robots need long-term prediction capability, humanoids need to handle high-dimensional dynamic systems, and dexterous manipulation needs fine force-deformation modeling. **No single unified world model architecture can simultaneously optimize all robot tasks.** And Dreamer is just one implementation of "latent world model + imagination-based RL," not the entirety of world models.

I think understanding the world model's "position" in a robotic system is more important than understanding its mathematical formulas alone. Because the world model's value lies not in how precise it is, but in whether it can provide a good enough imagination space for policy learning.

**Perception tells me what is now; the world model tells me what might happen after I act; value/policy tells me which future is worth pursuing.**

Next, we'll discuss the specific design of Actor-Critic in imagination space — how they turn world model predictions into useful policies, and why the symlog transformation matters so much for value learning.
