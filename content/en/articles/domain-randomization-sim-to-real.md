---
title: "Domain Randomization: The Bridge from Simulation to Reality"
slug: "domain-randomization-sim-to-real"
date: 2026-08-08
draft: false
categories: ["Sim-to-Real"]
tags: ["Domain Randomization", "Sim-to-Real", "Simulation", "Robotics", "DreamerV3", "MuJoCo", "Isaac Sim"]
description: "Why do policies trained in simulation fail on real robots? Domain randomization randomizes physics parameters, visual appearance, and sensor noise to make policies robust to the sim-to-real gap."
toc: true
related_articles:
  - sim-to-real-transfer
  - mujoco-vs-isaac-sim
  - td-mpc-world-model-control
  - isaac-lab-install-guide
  - 2026-08-30-dreamer-applications
  - world-model-intro
aliases:
  - /en/articles/domain-randomization-sim-to-real.html
---


In the previous post, we discussed how TD-MPC uses world models for robot control. But whether you're using Dreamer, TD-MPC, or any other method, the learned policy ultimately needs to be deployed on a real robot. This inevitably leads to Sim-to-Real transfer — and Domain Randomization is the most fundamental technique on this path.
 

This article systematically breaks down domain randomization: what problem it solves, what types exist, how to implement it in engineering, and the latest advances.
 
## The Root Causes of the Sim-to-Real Gap
 

Let's first revisit the problem. There are systematic discrepancies between simulation environments and the real world, collectively known as the Sim-to-Real Gap. Some of these stem from unmodeled dynamics, such as friction variations, contact errors, and sensor noise:
 

Dynamics level: friction coefficients, mass distribution, joint damping, motor response delays. The physical parameters in simulation are often approximations that deviate from the real robot.
 

Visual level: lighting conditions, camera noise, texture details, background variations. Simulation rendering differs significantly from what a real camera captures.
 

Sensor level: IMU drift, force sensor noise, encoder accuracy. Sensors in simulation are idealized, while real sensors exhibit various non-ideal characteristics.
 

These discrepancies lead to a core problem: a policy trained in simulation suffers performance degradation when deployed directly in the real world. The idea behind domain randomization is straightforward — since the real world cannot be fully modeled, let the training environment cover a reasonable range of possible conditions, so that the real world becomes just one possible instance within the training distribution.
 
## Why Is Domain Randomization Gaining Attention Again?
 

Domain randomization is not a new technique — systematic work dates back to around 2017. But its popularity has clearly risen in recent years, driven by broader shifts in the embodied AI research paradigm.
 

In the past, robot control relied primarily on hand-crafted models and task-specific controllers. Engineers designed dynamics models, tuned parameters, and wrote control logic for each specific task. This approach worked well in structured environments but scaled poorly — every new task required starting over.
 

The current paradigm is: large models + reinforcement learning + simulation training. Policies are no longer hand-designed by humans but automatically learned through massive trial and error in simulation. This means the demand for training data has surged. But real-world robot data is too expensive, too slow, and too dangerous to collect — you can't have a real robot running millions of trial-and-error episodes per day.
 

As a result, simulation training has become the mainstream choice. But the core bottleneck of simulation training is: how do you transfer the capabilities learned in simulation to the real world? Sim-to-Real is no longer just a robot control problem — it has become a central challenge in scaling embodied AI training. Domain randomization, as the most fundamental and general-purpose technique in this space, has naturally regained the spotlight.
 
## The Core Idea of Domain Randomization
 

The intuition behind domain randomization comes from a simple observation: if a policy performs well across a sufficiently diverse set of simulation environments, it should also work in the real world — because the real world is just one of many possible environments.
 

Formally, domain randomization treats the simulation parameters θ as random variables, sampling from a distribution p(θ). During training, each episode uses a different set of θ values. What the policy learns is not "how to perform in one specific simulation," but "how to perform across a wide range of possible environments."
  Key insight: Domain randomization is essentially injecting diversity into the training data distribution, enabling the policy to learn robust behaviors that do not depend on specific simulation parameters.  
## Taxonomy of Domain Randomization
 

Based on what is being randomized, domain randomization can be divided into four major categories:
 
### 1. Dynamics Domain Randomization (Dynamics DR)
 

Randomizing physical parameters:
 
 
- Mass: object mass, robot link mass. In practice, engineers typically start with a narrow range (e.g., perturbing around nominal values) and gradually expand based on real-world transfer results. 
- Friction: contact friction coefficients, joint friction. This is one of the most critical factors affecting manipulation tasks. 
- Damping: joint damping coefficients. Affects the robot's dynamic response. 
- Inertia: moments of inertia. Affects the dynamics of rotational motion. 
- Delay: control delay, observation delay. Simulates the response lag of real systems. 
 

Dynamics DR is the most commonly used type and has a significant impact on most tasks.
 
### 2. Visual Domain Randomization (Visual DR)
 

Randomizing visual rendering parameters:
 
 
- Lighting: light source position, intensity, color. This is the most important factor in visual DR. 
- Texture: object surface textures and colors. Can use procedural textures or random image maps. 
- Background: scene backgrounds. Can use random colors, random images, or random 3D scenes. 
- Camera: camera position, angle, focal length, noise. 
- Material: reflectance, transparency, roughness. 
 

Visual DR is especially important for vision-based policies. If the policy directly uses camera images as input, visual DR is almost mandatory.
 
### 3. Sensor Domain Randomization (Sensor DR)
 

Randomizing sensor characteristics:
 
 
- Noise: Gaussian noise, bias drift. 
- Accuracy: quantization error, resolution limits. 
- Frame drops: randomly dropping observation data to simulate communication delays. 
 

### 4. Task Domain Randomization (Task DR)
 

Randomizing task-related parameters:
 
 
- Object pose: initial position and orientation of target objects. 
- Object shape: variations in object size and shape. 
- Goal position: variations in target placement locations. 
 

Strictly speaking, Task DR does not fall under the Sim-to-Real category — it primarily addresses task generalization rather than the simulation-reality gap per se. However, many papers include it under the broader umbrella of domain randomization (e.g., object pose randomization, goal randomization), because it shares the same technical framework as Dynamics DR and is typically used in conjunction with it.
 
## Engineering Implementation: Choosing the Randomization Range
 

The most critical engineering decision in domain randomization is: how large should the randomization range be?
 

Range too small: it fails to cover real-world variations, and Sim-to-Real transfer fails.
 

Range too large: the training environment becomes overly diverse, the policy fails to learn effective behaviors, and even simulation performance suffers.
 

This is a classic bias-variance tradeoff. In practice, there are several rules of thumb:
 

1. Start from nominal values and expand gradually. First train with nominal parameters (simulation defaults) to confirm the policy can learn effectively in simulation. Then progressively expand the randomization range while monitoring changes in simulation performance and transfer performance.
 

2. Prioritize sensitive parameters. Not all parameters are equally important. For grasping tasks, friction coefficients and object mass are typically the most sensitive. Sensitivity analysis (varying one parameter at a time and observing performance changes) can help determine priorities.
 

3. Calibrate the range using real data. If you have real-world data (e.g., sensor readings from actual robot runs), you can use it to estimate the true distribution of parameters and then set the randomization range accordingly. This is far more reliable than guessing.
 
## Advanced Techniques
 ### Automatic Domain Randomization (AutoDR)
 

Manually setting randomization ranges is time-consuming. The idea behind AutoDR is to use optimization algorithms to automatically search for the best randomization distribution. The core concept is to treat the parameters of the randomization distribution (e.g., the bounds of a uniform distribution) as hyperparameters and optimize them using Bayesian optimization or gradient-based methods.
 

Work from OpenAI and others has demonstrated the feasibility of automatically adjusting randomization distributions, showing that automated search can reduce tuning costs compared to fully manual range design.
 
### Curriculum Domain Randomization (Curriculum DR)
 

A common practice is to progressively increase randomization difficulty during training — for example, starting with a narrow range to let the policy learn basic behaviors first, then gradually expanding. This follows the philosophy of curriculum learning.
 

The advantage of Curriculum DR is more stable training — if you start with a very large randomization range from the beginning, the policy may fail to learn anything at all.
 
### Adversarial Domain Randomization (Adversarial DR)
 

An adversarial network is used to generate "worst-case" simulation parameters — specifically targeting the environment parameters where the policy performs worst. The policy is then trained on these difficult environments, resulting in stronger robustness.
 

This approach is theoretically elegant but practically unstable, and is not widely used at present.
 
## Limitations of Domain Randomization
 

While effective, domain randomization is not a panacea. It has several known limitations:
 

1. Cannot cover all discrepancies. Some Sim-to-Real gaps are difficult to bridge through randomization alone. For example, the contact models in simulation and real-world contact mechanics may differ fundamentally — this cannot be resolved by simply tuning a few parameters.
 

2. Can be overly conservative. To cover all possible conditions, the policy may learn excessively conservative behaviors. For instance, in a grasping task, if a wide friction range is randomized, the policy might grasp with excessive force, which could damage objects in the real world.
 

3. Increased training cost. Randomization means every episode has a different environment, so the policy requires more training steps to converge.
 

These limitations are precisely why the Sim-to-Real field has other technical approaches (such as system identification, online adaptation, and world model-assisted methods).
 
## Domain Randomization + World Models
 

An emerging direction is combining domain randomization with world models. The idea is to have the model learn the dynamics under different environment parameters, thereby improving the policy's adaptability to unseen variations.
 

If this direction proves viable, world models could explore different domain parameter configurations in latent space without needing to re-render each one in simulation. For vision-based tasks, this could significantly reduce training costs. However, this is not yet an industrial mainstream practice and remains largely in the research validation stage.
 
## Practical Checklist
 

If you plan to use domain randomization in your project, here is a practical checklist:
 
 
- Verify that the nominal parameters in simulation are reasonable (compare against the real robot) 
- Start with Dynamics DR (mass, friction, damping) — this gives the best return on investment 
- If using a vision-based policy, add lighting and texture DR 
- Start with small perturbation ranges and gradually expand based on transfer results 
- Use a Curriculum DR strategy to progressively widen the range 
- Monitor simulation performance — if it degrades too much, the range is too large 
- If you have real-world data, use it to calibrate the randomization range 
 
## Summary
 

Domain randomization is the most fundamental technique in Sim-to-Real transfer. Its core idea is simple: let the training environment cover a reasonable range of parameter variations, and the policy will work robustly in the real world. But there are many details in engineering implementation — what to randomize, how large the range should be, how to train — and these details often determine the success or failure of the transfer.
 

Domain randomization is not the only Sim-to-Real method, but it is the most universal one. Regardless of your task, algorithm, or simulation platform, domain randomization can be applied. This is why it has become a standard component in nearly all Sim-to-Real work.
 

In the next post, we'll compare two mainstream robot simulation platforms: MuJoCo and Isaac Sim, and look at what scenarios each is best suited for.
