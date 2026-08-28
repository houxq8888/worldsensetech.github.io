---
title: "Is Sim-to-Real Too Hard? World Model-Driven Adaptive Transfer Methods"
slug: "sim-to-real-transfer"
aliases:
  - /en/articles/sim-to-real-transfer.html
date: 2026-08-05
draft: false
categories: ["Sim-to-Real"]
tags: ["Sim-to-Real", "World Models", "Domain Adaptation", "DreamerV3", "Domain Randomization", "Robotics"]
description: "Policies trained in simulation often see 30-50% performance drops when deployed on real robots — the Sim-to-Real Gap. World models are providing new solutions through synthetic data generation and adaptive transfer."
toc: true
related_articles:
  - domain-randomization-sim-to-real
  - mujoco-vs-isaac-sim
  - world-model-intro
  - 2026-08-30-dreamer-applications
  - td-mpc-world-model-control
  - isaac-lab-robot-rl
---

This is the fourth article in the World Models series. The first three covered the basic concepts of world models, the core principles of RSSM, and an introduction to embodied AI. Today, we're diving into a more practical topic: how do we actually deploy policies trained in simulation onto real robots?

This problem has plagued the robotics AI field for nearly a decade. World models perform brilliantly in simulated environments — DreamerV3's sample efficiency is 10-100x higher than traditional RL. But once deployed on real robots, performance often drops by 30-50%. This gap is the famous "Sim-to-Real Gap".

In this article, I want to discuss from an engineer's perspective why this problem is so hard, and what new solutions world models offer.

## Three Major Challenges of Sim-to-Real

Let's first understand where the problems come from. The gap between simulation and the real world mainly comes from three aspects:

### 1. Domain Shift

The physical parameters of simulators can never perfectly match the real world. A friction coefficient off by 0.1, a mass difference of 5 grams, motor response delayed by 10 milliseconds — these tiny differences are invisible in simulation but get amplified after policy deployment.

What makes it worse is that these parameters are often unknown. You don't know the exact friction coefficient of the real world, you can only guess a range.

### 2. Unmodeled Dynamics

Simulators can only model known physical laws. But the real world has many "surprises":

- Deformation of flexible objects (like grabbing a piece of cloth — the wrinkles cannot be precisely predicted)
- Contact nonlinearities (stick-slip effects, micro-vibrations when two surfaces contact)
- Sensor noise (illumination changes in camera images, drift in force sensors)
- External disturbances (someone bumped the table, slight ground vibration)

These dynamics are either simplified or completely ignored in simulation. Policies learn the rules of an "ideal world" in simulation, then get confused in the real world.

### 3. Safety Constraints

In simulation, policies can try and fail freely — if they crash, just restart. But in the real world, if a robot crashes, it's actually broken, and could even hurt someone.

This means policies deployed to the real world must satisfy safety constraints: cannot exceed joint limits, cannot produce excessive force, cannot enter dangerous areas. These constraints are often not fully considered during simulation training.

## Traditional Approach: Limitations of Domain Randomization

In recent years, the mainstream approach for Sim-to-Real transfer has been Domain Randomization.

The idea is intuitive: since we don't know the exact parameters of the real world, let's randomize these parameters in simulation. During training, each episode uses different friction coefficients, masses, lighting conditions, sensor noise. The policy trains in so many "different worlds" — hopefully one of them is close to the real world.

This method does work. OpenAI's Rubik's Cube robot (2019) was achieved using domain randomization — trained in simulation, directly deployed to the real world with over 90% success rate.

But domain randomization has several fundamental limitations:

First, the randomization range is hard to determine. If the range is too small, it won't cover the real world's parameters; if too large, the policy becomes overly conservative and can't do anything well.

Second, it can only handle known uncertainties. Domain randomization can only randomize parameters you "know to randomize". But unmodeled dynamics (like flexible deformation) — you don't even know how to randomize them.

Third, low sample efficiency. To cover a large enough parameter space, massive training data is needed. For complex robot tasks, training time can stretch to weeks.

## New Approach: Using World Models as a "Bridge"

World models provide a completely new approach to Sim-to-Real transfer: instead of having the policy jump directly from simulation to the real world, use the world model as an intermediate "bridge".

The core idea is: the world model learns the dynamics of the environment. If this world model is good enough, it should be able to simulate both the simulated environment and the real world. The policy trains in the world model's "imagined space" rather than in the simulator. This way, what the policy learns is not "the rules of the simulation world" but "more general environmental dynamics".

Specifically, there are three key mechanisms for world model-driven Sim-to-Real transfer:

### Mechanism 1: World Model-Driven Domain Adaptation

Traditional domain randomization randomizes parameters at the simulator level. The world model approach does adaptation at the model level.

The process is:

1. Collect large amounts of data in the simulated environment, train a world model
2. Collect a small amount of data in the real world (possibly just a few minutes of demonstrations)
3. Fine-tune the world model with real data to make it "adapt" to real-world dynamics
4. Retrain the policy in the adapted world model

The advantage of this approach: the world model's parameters are far fewer than the simulator's parameters. Fine-tuning the world model is much faster than recalibrating the simulator, and only requires a small amount of real data.

For example: in a robotic arm grasping task, the world model trained in simulation predicts friction of 0.3, but the real world's friction is 0.35. Traditional methods need to recalibrate the simulator's friction parameters, which might take hours. The world model approach only needs a few rounds of fine-tuning with real data, and the model can learn the correct friction, possibly in just minutes.

### Mechanism 2: Progressive Trust Transfer

Even if the world model is well-adapted, directly deploying the policy to the real world still has risks. The idea of progressive trust transfer is: deploy in stages, with safety constraints at each step.

Specifically divided into three phases:

**Phase 1: Pure Simulation Training**

The policy trains entirely in the world model's imagined space. The policy might not be stable yet at this stage, but that's okay because it's in simulation.

**Phase 2: Mixed Environment Training**

The policy alternates between simulation and real world training. For example, odd episodes run in simulation, even episodes run in the real world. In real-world episodes, add safety constraints (like limiting speed, limiting force range).

The key at this stage is trust evaluation: the world model predicts "the consequences of executing this action in the real world". If the predicted uncertainty is too high, reduce the policy's exploration intensity, or switch to more conservative behavior.

**Phase 3: Real World Deployment**

The policy runs entirely in the real world. But the world model still runs in the background, continuously predicting environmental dynamics. If the prediction deviates too much from actual observations, trigger safety mechanisms (like slowing down, stopping, requesting human intervention).

The core advantage of this progressive approach is safety. Every step has safety guarantees, avoiding the situation of "works fine in simulation, crashes immediately when deployed".

### Mechanism 3: Online Model Adaptation

Even after successful deployment, the real world keeps changing. Temperature changes, parts wear out, the table gets moved — all these affect policy performance.

The idea of online model adaptation is: after deployment, the world model continues learning from real data, constantly correcting its predictions.

Specific approach:

- While the policy executes actions, the world model simultaneously predicts the next state
- Real sensors return actual observations
- Compare predictions with actual observations, calculate error
- Use this error to update the world model's parameters (online learning)
- The updated world model is used for next-step predictions and policy updates

This mechanism allows the policy to continuously adapt to environmental changes. For example, a robotic arm's joints wear with use, friction gradually changes. An online-adapting world model can capture these changes, and the policy can adjust accordingly.

## Safety Constraints: Control Barrier Functions

I mentioned safety constraints earlier, let me expand on the specific implementation method.

Control Barrier Functions (CBF) are currently the mainstream method for robot safety control. The core idea is: define a "safe set", and the policy's output must ensure the system state always stays within the safe set.

Formally, if h(x) is the barrier function, the safe set is {x | h(x) ≥ 0}, then policy π must satisfy:

```
h(xₜ₊₁) ≥ (1-α) * h(xₜ)

where α ∈ (0, 1) is the safety parameter
```

The meaning of this constraint is: the next step's safety measure h(xₜ₊₁) cannot decrease too much compared to the current safety measure h(xₜ). If the policy's output violates this constraint, use an optimization problem to find the "closest action to the policy output that satisfies the safety constraint".

In the world model framework, CBF can be integrated like this:

1. World model predicts next state xₜ₊₁
2. Check if h(xₜ₊₁) satisfies safety constraints
3. If not satisfied, use quadratic programming (QP) to find the optimal action satisfying constraints
4. Execute the corrected action

The advantage of this approach: the policy can explore freely (in the world model), but the actually executed actions always satisfy safety constraints. Even if the world model's predictions have errors, CBF can guarantee the system won't enter dangerous states.

## A Complete Example: Robotic Arm Grasping

Finally, let's use a specific example to tie all the methods together.

Suppose the task is to make a robotic arm learn to grasp objects of different shapes (cubes, cylinders, spheres).

**Step 1: Simulation Training**

Build the robotic arm and object simulation environment in MuJoCo. Collect 100,000 steps of interaction data, train the world model. Then train the grasping policy in the world model's imagined space (using Actor-Critic method). At this stage, the policy learns the basic skills of "how to grasp".

**Step 2: Real Data Collection**

On the real robotic arm, use teleoperation to collect 100 grasping demonstrations (about 30 minutes). This data contains real-world friction, mass, sensor noise, and other information.

**Step 3: World Model Adaptation**

Fine-tune the world model with the 100 demonstrations. The model learns the real world's dynamic characteristics — like the real friction coefficient is 0.05 higher than simulation, real object mass is 2 grams heavier than nominal.

**Step 4: Mixed Environment Training**

The policy alternates between the adapted world model and real environment training. Add CBF safety constraints in real environment episodes — limit end-effector speed to no more than 0.5m/s, limit gripping force to no more than 20N.

**Step 5: Deployment and Online Adaptation**

Deploy the policy to the real robotic arm. The world model continues running, predicting the next state at each step. If prediction error exceeds threshold, trigger safety mechanisms (slow down or stop). Meanwhile, the world model updates online with real data, continuously adapting to environmental changes.

After this process, the policy's grasping success rate in the real world typically reaches 85-95%, with performance gap from simulation controlled within 10%.

## Limitations and Future Directions

Although world model-driven Sim-to-Real transfer is promising, there are still limitations:

**Computational cost**. The world model itself needs training, and fine-tuning also requires data. For resource-limited teams, this can be a barrier.

**Complex scenarios**. For scenarios involving flexible objects, multi-object interaction, dynamic environments, the world model's prediction accuracy is not yet high enough. Sim-to-Real transfer for these scenarios remains an open problem.

**Theoretical guarantees**. Current progressive trust transfer and online adaptation methods lack rigorous theoretical guarantees (like convergence, stability proofs). This is a direction academia is working on.

In the next 2-3 years, I look forward to seeing the following developments:

- More efficient world model adaptation methods (few-shot or even zero-shot adaptation)
- Stricter safety guarantees (combining formal verification with CBF)
- Multi-robot collaborative Sim-to-Real transfer (scaling from single robot to multi-robot systems)

## Final Thoughts

Sim-to-Real transfer is the key step for robotics AI to go from "lab demo" to "real application". World models offer new solutions — not having the policy directly cross the gap between simulation and reality, but using the world model as a bridge, gradually closing the gap through domain adaptation, progressive transfer, and online adaptation.

The core advantages of this approach are safety and efficiency: safety in that every step has constraint guarantees, efficiency in that only a small amount of real data is needed to complete adaptation.

For engineers wanting to practice in this direction, my advice is: start with simple tasks (like robotic arm reaching), run through the entire process, then gradually increase task complexity. Don't start by challenging flexible object grasping or humanoid robot walking — Sim-to-Real transfer for those tasks is still at the research frontier.

Next, let's look at a very interesting case: how Amap ABot-World-0 extends interactive world model inference time from 1 minute to 24 hours. This is an important breakthrough in long-term stability for world models. Stay tuned.
