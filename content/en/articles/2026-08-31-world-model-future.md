---
title: "From Dreamer to World Model Agents: Future Directions and Research Trends"
slug: "2026-08-31-world-model-future"
date: 2026-08-31
draft: false
categories: ["World Models"]
tags: ["DreamerV3", "World Models", "Transformer", "V-JEPA", "Genie", "LLM Agent", "Dreamer Series"]
description: "Starting from DreamerV3, a comprehensive roadmap of Transformer world models, V-JEPA, Genie, LLM Agents, and robotics foundation models — toward modular agent architectures."
toc: true
related_articles:
  - vla-vs-world-model
  - 2026-08-30-dreamer-applications
  - world-model-transformer
  - world-model-representations
  - rssm-deep-dive
  - world-model-8year-bottleneck
---

> **Dreamer Series · Part 6**
>
> Series directory (currently at Part 6):
> 1. [(1) Understanding Dreamer: How World Models Learn to 'Imagine'](/en/articles/2026-08-25-dreamer-explained/)
> 2. [(2) Dreamer's Actor-Critic: Policy Optimization in Imagination](/en/articles/2026-08-27-dreamer-actor-critic/)
> 3. [(3) DreamerV3 Training Engineering: From GPU Setup to Hyperparameter Tuning](/en/articles/2026-08-28-dreamerv3-training-tips/)
> 4. [(4) DreamerV3 GPU Guide: From VRAM Requirements to Cost-Performance Analysis](/en/articles/2026-08-29-dreamerv3-gpu-guide/)
> 5. [(5) Dreamer in Practice: From Simulation Control to Sim-to-Real](/en/articles/2026-08-30-dreamer-applications/)
> 6. **(6) From Dreamer to World Model Agents: Future Directions and Research Trends**

The previous five articles covered Dreamer's architecture, principles, training, hardware, and applications. But a broader question remains: **where is the future of world models? How will Dreamer-like methods evolve?**

As the finale of the Dreamer series, this article starts from Dreamer and looks at several important research directions: Transformer world models, predictive visual models (V-JEPA/Genie), World Model + LLM Agents, and robotics foundation models.

### Three Levels of World Models

Before diving in, it's worth clarifying what "world model" means. The methods discussed below actually operate at different levels:

| Level | Goal | Examples |
|-------|------|----------|
| **Representation WM** | Learn useful world representations | V-JEPA, Slot Attention |
| **Dynamics WM** | Learn state transitions for prediction and decision-making | Dreamer, MuZero |
| **Simulation WM** | Generate possible future states, support multi-modal prediction | Genie, Diffusion WM |

```text
            World Model
                |
    ┌───────────┼───────────┐
    |           |           |
Representation Dynamics   Simulation
 (V-JEPA)     (Dreamer)    (Genie)
    |           |           |
 Representation  Policy    Multi-future
  Learning    Optimization  Generation
```

These three levels are complementary, not competing. Dreamer belongs to dynamics world models, V-JEPA is closer to the representation level, and Genie and diffusion models belong to the simulation level. Understanding this classification helps avoid conflating methods at different levels.

## I. Dreamer's Limitations and Evolution Directions

Before discussing the future, let's review Dreamer's core limitations:

**Structural Limitations of RSSM**

- No explicit 3D structure — a prediction model based on latent variable statistics
- No explicit high-level causal structure — low-level action→state transitions implicitly contain causality, but lack composable high-level causal mechanisms like "push object → object moves → collision occurs"
- Predictions blur in imagination space — details are lost after long rollouts
- Insufficient uncertainty modeling — RSSM includes stochastic latent states that can model some degree of random dynamics, but due to training objectives and latent representation constraints, modeling complex, multi-modal future distributions remains limited
- **Representation Bottleneck** — one of RSSM's core limitations is not the prediction network capacity, but whether the latent state forms a rich enough world representation for long-horizon planning. The visual encoder may only encode color and texture while ignoring cup position, mass, and graspability — information critical for control

These are not engineering problems but fundamental architectural constraints. RSSM doesn't explicitly model 3D world structure or causal relationships — it learns action-conditioned latent dynamics from data. DreamerV3 pushed this architecture to high engineering maturity through symlog, two-hot, KL balancing, and other techniques, but from a research perspective, the RSSM paradigm has its ceiling.

**Evolution Logic of the Dreamer Series**

Looking at Dreamer's development:

- **Dreamer (V1)**: Proposed the basic RSSM + imagination framework
- **DreamerV2**: Replaced Gaussian latent with categorical latent, improving discrete action tasks
- **DreamerV3**: Significantly improved training stability through symlog, two-hot, KL balancing

The evolution from V1 to V3 focused on continuous improvements in latent dynamics, training objectives, and optimization stability, without changing the overall RSSM + imagination + actor-critic paradigm.

Dreamer's success proved that "small-scale, task-driven world models" are feasible, but didn't prove that "general world models" are solved. The gap from task-specific world models to general world models is exactly what the directions discussed below aim to bridge.

It's also worth mentioning another important world model route beyond Dreamer: **MuZero**. Rather than generating realistic world representations, MuZero learns abstract models in task-relevant spaces for planning — using learned dynamics + MCTS to achieve planning, with outstanding results on Atari and Go.

The two represent different world model philosophies:

```text
Dreamer (representation-oriented):
Learn world model → generate imagined trajectories → optimize policy

MuZero (decision-oriented):
Learn task-relevant dynamics → search future actions → select decisions
```

The former emphasizes "learning through imagination," the latter emphasizes "model-assisted planning." Dreamer pursues representations that "look like the real world," while MuZero only cares about abstract models that "help make good decisions" — no need for realistic representations, only useful predictions.

So where will architectural breakthroughs come from?

## II. Transformer World Models

### From RSSM to Attention

RSSM uses GRU as the deterministic model — a recursive sequence-to-sequence structure. But given Transformer's success in sequence modeling, a natural question arises: **can Transformer replace RSSM?**

Relevant work already exists:

- **TransDreamer**: Explored introducing Transformer attention into the RSSM framework to enhance latent sequence modeling
- **IRIS**: Uses Transformer for latent dynamics modeling, closer to a pure Transformer world model
- Related explorations include **Trajectory Transformer** and other offline RL methods based on trajectory sequence modeling — not strictly world models, but demonstrating Transformer's potential in decision sequence modeling
- **Later Dreamer explorations**: Hafner et al.'s research also explores attention mechanisms

Advantages of Transformer world models:

- **Parallel computation**: attention can process sequences in parallel, improving training efficiency
- **Long-range dependency**: attention directly models dependencies at any distance, without needing to pass through RNN hidden states
- **Scalability**: Transformer architecture more easily scales up through increased parameters and data

It's important to note that Transformer's advantages come primarily from sequence modeling and scalable training capabilities, not from being inherently better suited as a dynamics model. It represents a different dimension of exploration alongside RSSM, JEPA, and Diffusion — not a simple replacement.

But Transformer world models also face challenges:

- **Computational complexity**: attention's O(n²) complexity is costly for long sequences
- **Inductive bias**: Transformer lacks RNN's temporal inductive bias, potentially requiring more data to learn temporal structure
- **Data efficiency**: compared to language models, RL data is typically more limited and constantly shifting in distribution, making it challenging to directly replicate LLM's scaling recipe
- **Integration with RL**: how to combine Transformer's prediction capability with RL's decision framework remains an open question

It's worth noting that world models need more than just longer context — they need **better state compression and long-term memory mechanisms**. Language models process discrete token sequences, while the world is a continuously evolving system — gravity, mass, friction, geometric relationships don't disappear over time. Simply increasing context length isn't enough; hierarchical latent models, recurrent memory, and neural compression mechanisms are needed for effective long-horizon world modeling.

Currently, Transformer world models aren't simply replacing RSSM but exploring different temporal modeling inductive biases. One possible direction is **hybrid architectures**: using Transformer for long-range dependencies and RSSM-like structures for temporal dynamics.

Another important trend is **tokenization**. Unlike RSSM which models directly in continuous latent space, an increasing number of generative world models choose to first discretize visual states into tokens, then use Transformer to learn token dynamics:

```text
Observation → Tokenizer → Discrete tokens → Transformer dynamics → Future tokens
```

This tokenized world model is at the core of Genie and Sora-like approaches. It brings world models closer to the language model training paradigm — predicting the future via next token prediction. This provides a technical evolution thread from Dreamer → Genie → Sora → Agent.

### Possibility of Large-Scale Pretraining

Another advantage of Transformer is pretraining. The success of GPT, BERT, and other models demonstrates the power of the "large-scale pretraining + fine-tuning" paradigm. Could world models follow this path?

Current explorations:

- **General world model pretraining**: pretrain a general world model on large-scale datasets, then fine-tune to specific tasks
- **Multi-task learning**: use the same world model for multiple tasks, distinguished through task embeddings

But world model pretraining differs fundamentally from language models:

- Language model tokens are discrete; world model latent states are continuous
- Language model pretraining data volumes far exceed RL environment data volumes
- World models need action-conditioned prediction; language models only need next token prediction

World model pretraining is a promising direction, but requires finding appropriate representations and training objectives. Directly applying the language model pretrain-finetune paradigm may not suffice.

## III. Predictive vs. Generative World Models

### V-JEPA: Predictive Representation Learning

Meta's **V-JEPA (Visual Joint-Embedding Predictive Architecture)** represents a different approach — a predictive representation learning method inspired by world model ideas:

- Doesn't predict pixel-level futures, but predicts **abstract representations** of the future
- Makes predictions in latent space, avoiding the computational cost of pixel-level reconstruction
- Uses masking strategies to make the model learn meaningful representations

V-JEPA's core idea:

```text
Input: current frame + masked future frames
Objective: predict latent representations of masked regions
```

How this differs from Dreamer:

- Dreamer is action-conditioned; V-JEPA is mainly prediction-only
- Dreamer's latent space is designed for RL; V-JEPA leans more toward self-supervised representation learning
- V-JEPA doesn't directly output policies but learns general visual representations

V-JEPA represents an important direction: **world models don't necessarily need to be directly used for control — they can serve as general visual representation learning frameworks**. This "representation-first" approach may be more suitable for large-scale pretraining. V-JEPA is therefore closer to the perception layer of world models, not a complete agent world model — it provides agents with the ability to understand the world, but doesn't directly provide planning and decision-making capabilities.

Notably, the JEPA series is not just a visual representation learning method but part of Yann LeCun's broader **Advanced Machine Intelligence (AMI)** world model architecture. AMI's complete framework is Perception → World Model → Cost/Planning → Action, highly consistent with the Reasoning → World Model → Action architecture discussed at the end of this article.

### Genie: Generative Interactive Environment Model

Google DeepMind's **Genie** is another direction worth attention:

- Learns interactive environments from unlabelled internet videos
- Can generate the next frame based on user input
- Closer to a **Generative Interactive Environment Model**, borrowing video generation techniques to learn interactive environments

Genie's technical approach:

- Uses a tokenizer to discretize video frames
- Uses Transformer for autoregressive prediction
- Achieves interaction through latent space action inference

The differences from Dreamer are even more pronounced:

- Dreamer learns latent dynamics; Genie ultimately generates visual futures, but core dynamics modeling occurs in discrete latent token space, not directly in raw pixel space
- Dreamer's imagination space is latent space; Genie's is discrete token space
- Dreamer directly outputs actions; Genie needs an additional policy layer

The value of Genie-like models lies not in direct control but in:

- **Data augmentation**: generating synthetic data for training
- **Environment simulation**: providing training signals when real environments aren't available
- **Learning regularities**: learning environmental change patterns through large-scale video, exhibiting some physical consistency

Genie's greatest breakthrough is not its video generation capability but demonstrating that **internet video data could become a data source for training interactive world models** — learning interactive environment models from unlabelled videos alone, without action labels. This provides a possible path for future large-scale world model pretraining.

But Genie faces a fundamental challenge: **latent action discovery**. Internet videos only contain observation sequences, not control signals — "person opens door" in a video is an observation, but control variables like "apply 20N force, rotate 30 degrees" are missing. Learning controllable world models from action-free videos requires inferring latent control variables from observation sequences. Observation sequences are not control trajectories — this is the core challenge of learning interactive world models from video.

The computational cost and prediction error of pixel-level generation remain fundamental challenges.

### Probabilistic World Models: From Single Prediction to Multi-Future Generation

Unlike Dreamer's single-step latent prediction, **Diffusion World Models** learn future state distributions, representing multiple possible futures, suitable for prediction in high-uncertainty environments:

- **Diffuser**: models trajectory planning as a diffusion process, performing denoising generation in trajectory space
- **Dreamer Diffusion**: introduces diffusion models into the Dreamer framework, using diffusion processes to model latent dynamics
- **Video Diffusion World Model**: uses video diffusion models to generate future visual observations

The core advantage of diffusion world models is **multi-modal future prediction**:

```text
Traditional world model:
Current state → predict single future

Diffusion world model:
Current state → sample multiple possible futures
```

The real world has high uncertainty — a robot pushing a cup could result in it moving, tipping over, or sliding. A good world model needs not only to predict "what's most likely to happen" but also to represent "what could happen." Diffusion models are naturally suited for modeling such multi-modal distributions.

But the computational cost of diffusion models is significantly higher than latent prediction methods; balancing prediction quality and inference efficiency remains an open question.

## IV. World Model + LLM Agent

### Large Language Models as High-Level Planners

In recent years, LLMs have demonstrated remarkable capabilities in reasoning and planning. A natural direction is: **use LLMs as high-level planners and world models as low-level dynamic predictors**.

The architecture works as follows:

```text
User instruction
    ↓
LLM (high-level planning)
    ↓
Sub-goal sequence
    ↓
World Model (low-level prediction)
    ↓
Policy execution
```

Advantages:

- LLM handles semantic understanding and long-horizon planning
- World model handles physical dynamics and short-term prediction
- Complementary: LLM excels at abstract semantics and task decomposition but lacks reliable environment state prediction; world models excel at dynamics modeling but typically lack open-domain semantic knowledge

Existing explorations:

- **SayCan**: Google's work, using LLM for task planning and RL policies for execution
- **Code as Policies**: using LLM to generate robot control code
- **VoxPoser**: using LLM to generate 3D value functions for robot manipulation

But most of these works don't have explicit world models. If we introduce world models into this framework:

- LLM provides semantic understanding and task decomposition
- World model provides physical prediction and feasibility verification
- Policy layer makes decisions based on both outputs

World Model + LLM is a promising direction. Combining LLM's semantic knowledge with world model's dynamics prediction could produce more powerful agents.

With the development of reasoning models, future architectures may not be simple LLM + World Model but a **Reasoning Model + World Model + Action Model** three-layer structure:

```text
Reasoning Model (reasoning + planning)
       ↓
World Model (dynamics prediction + imagination)
       ↓
Action Model / VLA (policy execution)
```

One development direction for large language models is evolving from language generators into cognitive modules with reasoning, planning, and tool-calling capabilities. This layered architecture lets each layer focus on different levels of capability.

### Challenges and Open Questions

But this direction also faces challenges:

**Interface Design**

LLMs output text or symbols; world models need continuous actions or goals. How do we design the interface between them?

**Grounding**

LLM "common sense" is learned from text and may not align with the real physical world. How do we align LLM planning with physical world constraints?

**Real-time Performance**

LLM inference is relatively slow. How do we integrate it with control systems that need fast responses?

These questions have no standard answers, but the research direction is clear: **make language models understand physics, and make world models understand semantics**.

### Belief State and Memory

In the LLM + World Model architecture, there's another easily overlooked key component: the **belief state**. The real world is not fully observable — agents only see partial observations and need to maintain an internal belief state:

```text
Agent = Planner + World Model + Belief State

LLM (Planner)
  ↓
goal
  ↓
World Model (belief update)
  ↓
policy
```

The world model's role here is not just "predicting the future" but also **state estimation** — continuously updating internal beliefs based on new observations. This is the same problem as belief update in POMDPs (Partially Observable Markov Decision Processes).

Additionally, long-term memory is an indispensable component in complete agent architectures — agents need to remember past experiences for future decisions. Future world models may need to integrate with explicit memory mechanisms rather than relying solely on implicit recurrent states.

## V. Robotics Foundation Models

### From Specialized to General

Current robot learning is mostly **task-specific**: each task requires separate data collection and policy training. This is inefficient and hard to scale.

A more important direction is the **Robotics Foundation Model**:

- Pretrained on large-scale, multi-task, multi-robot data
- Able to generalize to new tasks, new environments, new robots
- Similar to LLMs' generalization ability in language tasks

The world model's role in this direction:

- **Unified representation**: world models can serve as a unified representation framework for different robots and tasks
- **Data efficiency**: through world model prediction, reduce the need for real data
- **Safety verification**: verify policy safety in the world model's imagination space

### Existing Explorations

- **RT-2**: Google's work, transforming robot control into a language modeling problem
- **Octo**: Berkeley's general robot policy
- **Open X-Embodiment**: multi-robot dataset and pretrained models
- **π0 (Physical Intelligence)**: Vision-Language-Action (VLA) model, representing a new direction for general robot policies

These models mainly address policy generalization, not complete world modeling. Most are based on imitation learning or reinforcement learning; integration with world models is still in early stages.

### How World Models Integrate with Robotics Foundation Models

Current robotics foundation models are mainly **robot foundation policies** — solving policy generalization. But complete robot agents also need predictive capability:

```text
Robot Foundation Model
        +
World Model (predict dynamics)
        ↓
Predictive Robot Agent
```

Possible ways world models integrate with robotics foundation models:

- **Predictive representation**: world models provide robotics foundation models with the ability to predict future states, not just reactive policies
- **Imagination training**: incorporating world model imagination training into robotics foundation model pretraining for improved data efficiency
- **Safety constraints**: world models predict consequences before execution, providing safety verification

Particularly the combination of VLA models with world models:

```text
VLA Model (perception + language understanding + action generation)
    +
World Model (dynamics prediction + feasibility verification)
    ↓
Predictive VLA Agent
```

VLA models excel at generating actions from multimodal inputs but lack the ability to predict future dynamics; world models fill this gap. Their combination represents the evolution from "reactive policies" to "predictive agents."

This leads to the concept of an **Active World Model** — future robots won't just observe → act, but:

```text
observe
  ↓
imagine futures (world model imagines multiple possible futures)
  ↓
choose action (select optimal action)
  ↓
observe result (observe execution outcome)
  ↓
update model (update world model)
```

This is essentially an extension of Dreamer's "imagination training" philosophy to real robot systems — the world model is no longer just an offline training tool but a cognitive core that continuously perceives, predicts, and acts at runtime.

### World Model as Data Engine

The most critical aspect for future robots is not just "reducing data requirements" but forming an **automatic data loop**:

```text
Robot acts → collect experience → world model update → better policy → more capable robot → collect richer experience → ...
```

The world model's role in this loop goes beyond sample efficiency — it serves as a **data engine**:

- **Synthetic data generation**: generate training data in imagination space
- **Failure simulation**: predict which operations might fail, avoid them proactively
- **Exploration guidance**: world model guides agents to explore unknown regions
- **Active learning**: identify uncertain regions, prioritize data collection

This self-improving loop goes beyond simple sample efficiency — the world model doesn't just "learn with less data" but "drive automatic data growth."

This direction is still early but represents the possible evolution of robotics foundation models from "policy generalization" to "prediction + planning."

Robotics foundation models are likely to become one of the most important application directions for world models, because robot data is expensive to acquire (world model sample efficiency is valuable), robot tasks are highly diverse (unified representation is valuable), and robot safety requirements are high (imagination verification is meaningful).

But challenges are also clear: robot data diversity and scale are far smaller than language or image data; different robots have vastly different morphologies, sensors, and control methods; and the real world's complexity and sim-to-real gap remain significant.

## VI. Several Trends in World Model Research

Synthesizing the above discussion, several clear trends emerge in world model research:

### Trend 1: From RL World Models to General World Models

Dreamer-like methods belong to **RL world models** — serving reinforcement learning with the goal of improving sample efficiency. But world models' potential goes far beyond this:

- **Visual understanding**: learning visual representations by predicting future frames
- **Physical reasoning**: understanding physical laws by learning dynamics
- **Planning and decision-making**: assisting decisions by imagining futures

Future **general world models** may emerge — not limited to RL, but serving as general prediction and planning tools.

### Trend 2: From Single Modality to Multi-Modal Fusion

Dreamer primarily handles vision and proprioception. But the real world is multi-modal:

- Vision, touch, hearing, language
- Different modalities provide complementary information
- Language can provide high-level semantics; vision provides low-level details

Future world models need to **fuse multi-modal information** to build more complete world representations.

### Trend 3: From Prediction to Generation

Dreamer's imagination space is latent space, predicting latent dynamics. But Genie-like models demonstrate another possibility: **directly generating pixel-level futures**.

Both routes have pros and cons:

- Latent prediction: computationally efficient but lacks detail
- Pixel generation: rich in detail but computationally expensive

Future **hybrid architectures** may emerge: planning in latent space, verification in pixel space.

### Trend 4: From Single Agent to Multi-Agent

Dreamer primarily handles single-agent tasks. But the real world is multi-agent:

- Multi-robot collaboration
- Human-robot interaction
- Social agents

Future world models need to model **other agents' behaviors and intentions** — a more complex challenge.

### Trend 5: From Unstructured Latent Variables to Structured World Models

Dreamer's latent state is an unstructured vector — all information mixed together. But the real world's structure is **objects + relations**:

```text
Unstructured representation:
pixels → latent vector (all information mixed)

Structured representation:
pixels → objects → relations → dynamics
```

**Object-centric World Models** are an important direction for addressing the representation bottleneck:

- **Slot Attention**: decomposes scenes into independent object slots
- **Object-centric learning**: each object has independent representation and dynamics
- **Structured prediction**: predict object state transitions, not pixel changes

For example, when a robot sees "a cup on the table," a structured world model would separately model cup position, mass, and relationship to the hand, rather than compressing all information into a single vector. This directly addresses the "no explicit 3D structure" problem mentioned earlier — through object-centric representations, world models can achieve more structured and interpretable world understanding.

## VII. Possible Future Convergence Directions

Synthesizing the above discussion, the future of world models may not be a "Dreamer vs Genie vs LLM" competition but a convergence of multiple routes:

```text
                 Human Goal
                     ↓
           Reasoning Model (reasoning + planning)
              ┌──────────┴──────────┐
              ↓                     ↓
          Memory              World Model
      (experiential)        (dynamics prediction)
              └──────────┬──────────┘
                         ↓
                   Belief State
                         ↓
                     Planner
                   (optimization)
                         ↓
                Policy / VLA Model
                         ↓
                      Action
                         ↓
                   Environment
                         ↓
                  New Experience
                    (→ Memory update)
```

Core ideas of this framework:

- **Reasoning Model** provides reasoning, reflection, and high-level task planning
- **Memory** stores past experiences, supporting long-term learning and recall
- **World Model** provides physical dynamics prediction and imagination training space
- **Belief State** fuses memory and prediction, maintaining current world understanding
- **Planner** makes decision optimizations based on belief state
- **Policy / VLA Model** handles low-level control execution
- **Environment** generates new experiences, feeding back to update memory and models

What may emerge in the future is not a single world model but a **modular agent architecture** — each component responsible for different levels of capability, with the world model as the prediction and planning core.

Challenges facing this convergent architecture:

- Interface design and information flow between components
- Training coordination and end-to-end optimization of the overall system
- Computational efficiency for real-time inference

But the direction is clear: from single models toward **system-level agent architectures**.

## VIII. What Challenges Remain for World Models on the Path to General Intelligence?

After discussing future directions, it's necessary to clarify world models' current boundaries. Current world models still face:

- **Insufficient long-term consistency**: prediction quality degrades after long rollouts, making it hard to maintain long-term consistency
- **Insufficient physical law generalization**: limited generalization to unseen physical scenarios
- **Insufficient data scale**: compared to language models, world model training data scales remain small
- **Non-unified evaluation standards**: lack of unified benchmarks and metrics
- **Gap between prediction and action**: good prediction doesn't equal good decision-making

These limitations mean world models are still quite far from AGI. World models are important components on the path to more intelligent systems, but not the only components. They need to combine with perception, reasoning, planning, language, and other capabilities to build truly general agents.

## IX. Frequently Asked Questions

### Q1: What's the difference between Dreamer's and ChatGPT's world models?

ChatGPT-like models learn language world models from large-scale text — they understand statistical patterns of language but don't understand the physical world. Dreamer's world model learns dynamics models from environment interaction — it understands task-relevant dynamic changes but lacks open-domain semantic knowledge. They're complementary: one excels at semantic reasoning, the other at physical prediction.

### Q2: Are world models a necessary condition for AGI?

World models may be one important component toward AGI, but not the only one. Complete general intelligence also requires perception, reasoning, planning, language, social skills, and more. Good prediction doesn't equal good decision-making; world models solve the "predicting the future" problem, but AGI also needs "understanding goals" and "value alignment."

### Q3: Why do robots need world models?

Robot data is expensive to acquire and trial-and-error is costly. The core value of world models is **sample efficiency** — reducing dependence on real interaction by repeatedly practicing in imagination space. Additionally, world models can predict action consequences before deployment, providing safety verification.

### Q4: Will Genie replace Dreamer?

Unlikely. Genie and Dreamer solve problems at different levels: Dreamer learns action-conditioned latent dynamics, directly serving policy optimization; Genie learns visual environment generation, mainly used for data augmentation and environment simulation. The future is more likely a fusion — Genie-like methods for the perception layer, Dreamer-like methods for the decision layer.

## X. Connecting the Series

```text
World Model Intro → RSSM Deep Dive → RSSM Code Series (6 parts)
                                              ↓
                                     Dreamer Series #1: Architecture
                                              ↓
                                     Dreamer Series #2: Actor-Critic
                                              ↓
                                     Dreamer Series #3: Training Tips
                                              ↓
                                     Dreamer Series #4: GPU Guide
                                              ↓
                                     Dreamer Series #5: Applications
                                              ↓
                                     Dreamer Series #6: Future Directions (this article)
```

Future directions is the outlook piece of the Dreamer series. Starting from Dreamer's limitations, it explores world model evolution directions and research trends.

If you haven't read the previous articles, I recommend starting with [Dreamer Architecture](/en/articles/2026-08-25-dreamer-explained/), [Actor-Critic](/en/articles/2026-08-27-dreamer-actor-critic/), [Training Tips](/en/articles/2026-08-28-dreamerv3-training-tips/), [GPU Guide](/en/articles/2026-08-29-dreamerv3-gpu-guide/), and [Applications](/en/articles/2026-08-30-dreamer-applications/) before reading this future directions article.

## XI. Summary

Looking at world model development, one possible evolution path is:

```text
2020-2024  RL World Model (Dreamer, MuZero)
                ↓ representation learning + imagination training
2024-2026  Generative World Model (Genie, Diffusion WM)
                ↓ multi-modal generation + uncertainty modeling
2026+      Agentic World Model (WM + Reasoning + VLA)
                ↓ modular agent architecture
```

From Dreamer to world model agents, future directions can be summarized as:

- **Transformer world models**: enhancing long-range dependencies with attention, exploring pretraining paradigms
- **Predictive and generative world models**: V-JEPA's representation learning, Genie's interactive environment modeling, diffusion models' uncertainty prediction
- **World Model + LLM/Reasoning Model**: reasoning model's high-level planning + world model's physical prediction
- **Robotics foundation models**: from task-specific to general generalization, VLA + World Model fusion
- **Research trends**: from RL to general, from single to multi-modal, from prediction to generation, from single agent to multi-agent

Dreamer is not the endpoint of world models but an important starting point. It proved that **learned world models can be used for effective policy learning** — this core idea will influence many future research directions.

World model research is just beginning. In the coming years, we may see more powerful world models emerge. They may no longer be called "Dreamer," but they will inherit Dreamer's core idea: **learning through imagination**.

I hope this series helps you build a comprehensive understanding of world models. If you have specific research questions, feel free to discuss them in the comments.
