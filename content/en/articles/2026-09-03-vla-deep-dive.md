---
title: "VLA Deep Dive: From RT-2 to OpenVLA to pi0 -- How End-to-End Policies Connect Language and Action"
slug: "2026-09-03-vla-deep-dive"
date: 2026-09-03
draft: false
categories: ["Embodied Intelligence", "Paper Analysis"]
tags: ["VLA", "RT-2", "OpenVLA", "pi0", "Vision-Language-Action", "Robot Foundation Model", "End-to-End Policy", "Flow Matching", "Embodied Intelligence", "Physical Intelligence"]
description: "VLA (Vision-Language-Action) models are reshaping the technical trajectory of robot learning -- from RT-2 injecting internet-scale knowledge into robot control, to OpenVLA surpassing a 55B closed-source model on 29 tasks with just 7B parameters, to pi0 using flow matching for continuous action generation and high-frequency control. This article provides a complete technical breakdown of the VLA evolution across three papers, distinguishing four axes of progress -- action representation, temporal abstraction, prediction/planning capability, and data heterogeneity -- and discusses the relationship between VLA and world models. This article's timeline focuses on pi0.7; subsequent model variants are not covered."
toc: true
related_articles:
  - 2026-09-02-jepa-deep-dive
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - world-model-intro
  - 2026-08-25-dreamer-explained
---

At the end of the [previous JEPA deep dive](/en/articles/2026-09-02-jepa-deep-dive/), I mentioned an open question: the JEPA approach currently operates primarily in visual and action spaces -- how does it integrate with language capability?

This question points to another core trajectory in embodied intelligence -- VLA (Vision-Language-Action).

If the core question of JEPA is "what should a world model predict?", then the core question of VLA is "how should a robot transform what it sees and hears into actions?" The two approaches start from different premises, but ultimately must answer the same underlying question: **how does perceptual information translate into physical action?**

This article provides a complete technical breakdown of VLA from RT-2 to OpenVLA to pi0. But I want to emphasize more than just a timeline -- **the evolution of VLA actually unfolds simultaneously along four dimensions: action representation, temporal abstraction level, prediction/planning capability, and data heterogeneity.** Only by separating these four dimensions can we understand what each model truly solves.

*Note: This article's technical timeline focuses on pi0.7 (April 2026). The VLA field is developing rapidly; subsequent model variants are not covered here.*

## 1. The Core Idea of VLA

**VLA (Vision-Language-Action) is a class of foundation-model policies that unify visual observations, language/task conditioning, and robot action policies into a single modeling framework.** It emphasizes unified representation and joint learning between perception, language grounding, and action generation, and **does not require the implementation to be a single neural network.** The concrete implementation can still include multiple specialized modules -- such as an action expert, a history encoder, or a hierarchical action head -- but they share the same representational foundation.

Traditional robot control is staged: a perception module performs object detection and segmentation, a planning module handles task decomposition and path planning, and a control module executes PID or impedance control. Each module is designed independently, and modules are connected through manually defined interfaces.

The VLA approach folds these stages into a single end-to-end learning framework. The inputs are camera images and natural language instructions; the outputs are robot actions -- end-effector pose deltas, joint angles, gripper open/close. There is no explicit perception-planning-control separation, no manually designed intermediate representations.

> **A note on "end-to-end":** When this article says "end-to-end," it means that the task conditioning and robot action policy are directly connected through a unified training system -- not that the model internally lacks modular structure or intermediate representations. For example, pi0 includes a VLM backbone -> action expert -> flow matching -> action multi-stage structure, and pi0.5 even has a semantic subtask -> action generation hierarchical inference chain -- they are still end-to-end learned policies, just not "from pixels straight through to motor commands in a single undifferentiated pass."

The key turning point for this idea occurred in 2023.

## 2. Technical Evolution Roadmap

Before breaking down specific models, let us look at two parallel trajectories and four evolution dimensions. The VLA / policy line and the predictive / world model line are **not different points in the same coordinate system** -- they approach the same goal from different directions.

```
Main Line A: VLA / Policy Line
--------------------------------------------------------->
RT-2 -> OpenVLA -> OpenVLA-OFT -> pi0 -> pi0.5 -> pi0.7


Sub Line B: Predictive / World Model Line
--------------------------------------------------------->
V-JEPA -> V-JEPA 2 -> V-JEPA 2-AC -> ???
                                          \
                                           \
                                      Future Unified Model?
                /
               /
    Convergence Point of Both Lines:
    shared representation
         |        |
     policy    prediction
```

### The Core Question Each Model Answers

Rather than only comparing parameters and data sizes, it is more revealing to look at what each model truly answers in the technical evolution:

| Model | Core Question Addressed |
|-------|----------------------|
| RT-2 | Can VLM semantic knowledge be transferred to robot actions? |
| OpenVLA | Can an open VLA be trained on large-scale cross-robot data? |
| OpenVLA-OFT | Can a discrete AR action decoder become a high-throughput policy? |
| pi0 | Can continuous action generation scale to a generalist robot policy? |
| pi0.5 | Can heterogeneous data and hierarchical semantics handle long-horizon tasks? |
| pi0.7 | Can strategy conditioning achieve a steerable generalist policy? |

This table reveals the internal logic of technical evolution better than a pure parameter comparison.

### Four Evolution Dimensions

```
                         VLA / Robot Foundation Model
                                      |
          +---------------------------+---------------------------+
          |                           |                           |
      Action Representation      Temporal Abstraction       Prediction
          |                           |                           |
   discrete token              action chunk                implicit
          |                           |                           |
   continuous regression       semantic subtask             subgoal
          |                           |                           |
   flow matching               hierarchical policy        world model
          |                           |                           |
          +---------------------------+---------------------------+
                                      |
                              Data Heterogeneity
                                      |
               single robot -> multi-robot -> web + robot
                         -> heterogeneous + suboptimal
                                      |
                                      v
                         Generalist Robot Policy
```

Under this framework, each model's positioning becomes clear:

- RT-2, OpenVLA, OpenVLA-OFT, and pi0 belong to the VLA / policy line, progressively solving action representation and inference efficiency problems
- pi0.5 and pi0.7 introduce hierarchical semantic structure and multimodal steering into the VLA line
- V-JEPA, V-JEPA 2, and V-JEPA 2-AC belong to the predictive / world model line, progressively acquiring action-conditioned prediction and planning capabilities
- The two lines have not yet fully converged, but they are approaching each other

## 3. RT-2: Injecting Internet Knowledge into Robot Control

### Paper Information

*RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, Google DeepMind, July 2023. arXiv:2307.15818.

### Core Idea

RT-2's core finding is: **a vision-language model pretrained on large-scale internet text and images can be directly fine-tuned into a robot policy -- and the fine-tuned policy can leverage semantic knowledge learned during pretraining for zero-shot generalization.**

The significance of this finding is that it systematically demonstrated for the first time that "internet-scale knowledge transfer to robot control" is feasible. Its core contribution can be summarized as a chain:

```
web knowledge -> language/vision representation -> robot action token
```

### Architecture

RT-2 did not build a new architecture from scratch. It directly fine-tuned existing VLMs:

| Variant | Backbone | Parameters |
|---------|----------|------------|
| RT-2 PaLI-X | PaLI-X (ViT visual encoder) | 55B |
| RT-2 PaLM-E | PaLM-E (multimodal embeddings) | 12B |

Images are processed through the backbone's built-in visual encoder, and language is processed through the standard VLM tokenizer. The key innovation is on the action side.

### Action Discretization: Turning Actions into "Another Language"

RT-2 represents robot actions as **7 continuous quantities**: end-effector 3D position deltas (dx, dy, dz), 3D rotation deltas (roll, pitch, yaw), and gripper state. Each dimension is uniformly quantized into **256 discrete bins**. The termination signal belongs to the control logic of the action sequence, not to one of the 7 continuous action dimensions.

The key engineering trick is **symbol tuning** -- discretizing actions into tokens so that actions can share the same autoregressive output interface as language. The concrete implementation differs by backbone: **PaLI-X directly uses existing token IDs to represent action bins, while PaLM-E overwrites 256 low-frequency tokens in the vocabulary, redefining them as action tokens.** Regardless of the approach, the effect is the same: the model architecture remains unchanged, the pretrained language and visual knowledge is fully preserved, and actions simply become "another language the model speaks."

During inference, when the model enters the action output phase, a decoding mask ensures that only these 256 action tokens have non-zero probability.

### Training

RT-2 performs co-finetuning on top of RT-1 / Everyday Robots' existing robot data and VLM web data to prevent catastrophic forgetting. The training recipes for the two backbone variants are not identical -- the paper reports different data mixing strategies for each. The core idea is consistent: retain a portion of internet image-text data to maintain the VLM's general semantic capabilities, while mixing in robot manipulation demos to learn action outputs.

### Key Results

RT-2's core numbers need to be distinguished across different evaluation protocols:

- **Performance on the original task distribution**: similar to RT-1 (approximately 91-93%), indicating that co-finetuning did not sacrifice existing capabilities
- **On generalization evaluation with novel objects, novel scenes, etc.**: RT-2 achieves approximately **62%**, significantly higher than RT-1's approximately **32%** -- nearly a 2x improvement
- **Emergent semantic capabilities** (capabilities completely absent from robot training):
  - Symbol understanding 82% (identifying and manipulating abstract symbols -- numbers, shapes, logos)
  - Person recognition 53% ("move the object to Taylor Swift")
  - Logical reasoning 46% ("place the object on top of 2+1")

Emergent capabilities are RT-2's most exciting result. These capabilities were not learned from robot training data but transferred from the VLM's internet pretraining knowledge. RT-1 performs near zero on these tasks.

### What Did RT-2 Actually Transfer?

This deserves further discussion. RT-2's emergent capabilities reveal an important hierarchical structure -- the knowledge that VLA transfers from internet pretraining and the capabilities learned from robot data are fundamentally different levels of things:

**Semantic knowledge.** "Cup," "red," "Taylor Swift," "inside," "on top of," "2+1" -- these concepts and relationships come from internet text and images. VLM pretraining provides abundant such priors.

**Visual grounding.** Seeing a completely novel object and being able to judge what semantic category it belongs to, understanding its relationship to language instructions. The VLM's visual encoder, trained on large-scale image-text pairs, provides powerful visual generalization capability.

**Physical skills.** Seeing a cup -> how to reach -> how to grasp firmly -> how to control force -> how to avoid collisions -- this entire chain from perception to force control, **internet data fundamentally cannot provide directly**.

This precisely explains a core phenomenon of RT-2: semantic generalization improves dramatically, but it cannot spontaneously generate new physical skills. Internet pretraining gives VLA a powerful "semantic engine," but physical manipulation capability still depends entirely on robot demo data.

### Limitations

RT-2's limitations are also clear:

- **Cannot invent new physical skills**. It can apply existing manipulation skills to new objects and new scenes, but cannot learn entirely new motor capabilities from internet knowledge.
- **Inference speed is constrained**. The 55B model requires cloud TPU inference, with a control frequency of only 1-3 Hz, far below the needs of actual robot control.
- **Action precision is limited by quantization**. 256-bin discretization may not be sufficient for fine manipulation.

### Relationship with World Models

RT-2 can be viewed as a typical **model-free / direct policy** method: it does not explicitly learn an action-conditioned dynamics model, but instead directly learns the mapping from observation + instruction to action. Chain-of-thought reasoning is the closest RT-2 gets to "planning," but that is language-level reasoning, not physical simulation.

However, it is important to note that this description primarily applies to the RT-2 generation of VLA. Subsequent VLAs have begun introducing action chunking, hierarchical semantic actions, subgoal conditioning, and other mechanisms, progressively adding more temporal structure on top of the "direct mapping" foundation.

## 4. OpenVLA: Open-Source 7B VLA

### Paper Information

*OpenVLA: An Open-Source Vision-Language-Action Model*, Stanford / UC Berkeley, June 2024, CoRL 2024. arXiv:2406.09246. Authors: Moo Jin Kim, Karl Pertsch, Chelsea Finn, et al.

### Core Idea

RT-2 proved the feasibility of VLA, but it is entirely closed-source -- the backbone is Google's internal PaLM/PaLI-X, and external researchers cannot reproduce or extend it. OpenVLA's goal is: **to build a reproducible VLA using open-source components and see how far the open-source approach can go.**

The answer is: farther than expected.

### Architecture

OpenVLA is a 7B parameter model based on the Prismatic VLM architecture:

**Visual Side -- Dual Encoder:**
- SigLIP (1152-dimensional features) + DINOv2 (1024-dimensional features), two visual encoders providing complementary visual features
- 224x224 images processed through a patch size 14 visual encoder yield 16x16 = 256 patch embeddings
- Features from both encoders are concatenated along the channel dimension, yielding a 2176-dimensional representation

**Projection Layer:** A 3-layer MLP (GELU activation) maps the 2176-dimensional visual features to the LLM's 4096-dimensional embedding space. OpenVLA's VLA training does not simply freeze the visual side -- it adapts visual representations using robot data -- this is one of the more counterintuitive design choices in the paper.

**Language Backbone:** Llama-2 7B (32-layer Transformer decoder), with an input sequence of approximately 280 tokens (BOS + visual patches + language instruction + action tokens).

**Action Output:** Following RT-2's approach -- **overwriting the 256 lowest-frequency tokens in the Llama tokenizer** as action bin IDs (the Llama tokenizer has only about 100 reserved special tokens, which is insufficient), with each action dimension autoregressively generating one token. Outputs 7-DoF actions for the WidowX robot.

### Training

- **Data**: Open X-Embodiment dataset, 970,000 real robot manipulation demos
- **Compute**: 64 A100 GPUs, training for approximately 15 days
- **Fine-tuning strategy**: Among the adaptation settings tested in the paper, LoRA updates only about 1.4% of parameters yet achieves downstream performance comparable to full fine-tuning

### Key Results

On the **29 tasks across multiple robot embodiments** reported in the paper (average task success rate), OpenVLA outperforms RT-2-X by 16.5 percentage points. Since the two systems differ in pretraining framework, robot data, and training recipe, this result is better interpreted as an overall system-level comparison rather than a pure parameter-efficiency comparison:

| Model | Parameters | Comparison |
|-------|------------|------------|
| **OpenVLA** | **7B** | **+16.5pp over RT-2-X (29-task avg, system-level comparison)** |
| RT-2-X | 55B | Baseline |
| OpenVLA vs Diffusion Policy | -- | +20.4% |
| OpenVLA vs RT-1-X / Octo | -- | Surpasses both |

**On difficult semantic generalization tasks (requiring internet-scale knowledge to understand the concepts), RT-2-X remains stronger.** OpenVLA's Open X-Embodiment training data does not include internet-scale image-text pretraining, so in "understanding completely unseen semantic concepts," it falls short of RT-2.

### Limitations

- **Autoregressive decoding bottleneck**: generating actions token by token, inference speed is approximately 4.2 Hz
- **The original model's inputs and training recipe center primarily on single-frame visual observations**, with limited capability for modeling historical visual information and complex multi-view scenes
- **Real robot deployment still requires embodiment / task-specific adaptation**; the original model's generalization capability cannot be directly equated with production-grade reliability
- **Discretization precision loss**: 256-bin quantization remains insufficient for fine manipulation

## 5. OpenVLA-OFT: The Real Bottleneck May Be the Action Interface

OpenVLA-OFT (2025, arXiv:2502.19645) should not be viewed merely as an optimization of OpenVLA. From a technical evolution perspective, it answers a critical question:

> **"Is the performance bottleneck of VLA the foundation model itself, or the action decoding?"**

OpenVLA's action decoding generates discrete tokens autoregressively, one token at a time. OFT performed a thorough overhaul:

- **Parallel decoding** replacing autoregressive: generating all action tokens simultaneously
- **Action chunking**: predicting 8-step actions in a single forward pass
- **Continuous action head**: using MLP + L1 regression instead of discrete tokens -- note, this is **continuous regression**, not flow matching
- **LoRA fine-tuning**

Results:

| Metric | Original OpenVLA | OpenVLA-OFT |
|--------|-----------------|-------------|
| LIBERO success rate | 76.5% | **97.1%** |
| Inference throughput | 4.2 Hz | **109.7 Hz** |
| Speed improvement | -- | **26x** |

The significance of OFT lies in demonstrating an important conclusion: **the scaling of the model backbone is not the only bottleneck for real-time robot control performance -- the action interface itself is equally important.** Changing the action decoding from token-by-token autoregressive to parallel decoding + action chunking + continuous action head -- modifying only the action interface -- yielded a 26x speed improvement and a substantial success rate increase.

This aligns closely with the main thread we will see later: "discrete handles unification, continuous handles control."

There is a point of confusion that needs clarification here: **"continuous action" and "flow matching" are not the same concept.** "Discrete tokens -> continuous action" is an evolution along one dimension (action representation), while "regression / diffusion / flow matching" is a choice along another dimension (the specific generation mechanism for continuous actions). OpenVLA-OFT takes the continuous regression path, while pi0 takes the flow matching path -- both belong to "continuous action" but with different generation mechanisms.

## 6. pi0: Flow Matching + Action Chunking

### Paper Information

*pi0: A Vision-Language-Action Flow Model for General Robot Control*, Physical Intelligence, October 2024. arXiv:2410.24164.

### Physical Intelligence Background

Physical Intelligence (also known as pi) was founded in 2023 in San Francisco, with the mission of "building general-purpose robot brains." The five co-founders include Karol Hausman (CEO, former Google DeepMind, core member of SayCan / RT-2), Chelsea Finn (Stanford, inventor of MAML), Sergey Levine (UC Berkeley, co-author of SAC), Brian Ichter, and Jasmine Hsu (both from Google Brain). As of 2026, Physical Intelligence has raised approximately $2.1 billion in total funding, with the most recent round corresponding to a valuation of approximately $11 billion.

### Core Architecture

pi0 made a different choice from RT-2 / OpenVLA: **instead of discrete tokens, it uses flow matching to generate continuous actions.**

The architecture has two parts:

| Component | Description |
|-----------|-------------|
| VLM Backbone | PaliGemma (3B parameter vision-language model) |
| Action Expert | 300M parameter specialized network, attached to the VLM |
| **Total Parameters** | **Approximately 3.3B** |

It is important to note that **the action expert is not an "add-on controller."** It is jointly trained and conditionally coupled with the VLM backbone -- the VLM provides language and vision conditioning, and the action expert generates continuous actions based on these conditions. A more accurate description is: **a language-vision backbone + a continuous action generation expert sharing conditional information.**

The inputs include image tokens, language tokens, and proprioception, which pass through the shared representation and then the action expert outputs an action chunk via flow matching.

### Three Mechanisms for Continuous Action Generation

There are three main mechanisms for continuous action generation, which need to be clearly distinguished. They are not a sequence of replacements, but rather **two parallel lines under generative action modeling**:

```
              Generative Action Modeling
                        |
                   +----+----+
              Diffusion   Flow Matching
                   |             |
             score/noise   vector field
                   |             |
             iterative     ODE integration
             denoising     / transport
                   |             |
                   +----+--------+
                        |
                  Regression
                        |
                  direct prediction
                  (no sampling)
```

Flow matching trains a vector field / velocity field over a probability path, enabling the model to transport from a simple prior distribution to the target data distribution through ODE integration. What it does is:

- Define a probability path from a pure noise distribution to a target action distribution (linear-Gaussian probability path)
- Train a network to predict the velocity field along this path
- During inference, start from noise and integrate along the learned vector field to obtain continuous actions

The key difference from diffusion is that their training objectives and inference forms differ. In pi0's specific implementation, flow matching combined with fewer Euler integration steps forms a continuous policy suitable for real-time action chunk generation.

### Action Chunks, Temporal Abstraction, and Planning: Three Easily Confused Concepts

pi0 generates an action chunk containing **50 future actions** each time; at a maximum **50 Hz system control frequency**, this corresponds to approximately 1 second of future trajectory. The paper reports that its system can achieve up to 50 Hz system control frequency on dexterous tasks.

It is important to note that flow matching inference itself still requires multi-step integration. The 50 Hz figure is the system-level control frequency including action chunk execution, not the speed of a single flow matching inference call.

**Three easily confused concepts need special clarification here:**

**Action chunking** -- outputting a_t, a_{t+1}, ..., a_{t+H-1} at once. This addresses reducing decision frequency and improving motion coherence. pi0's 50-step chunk belongs to this level.

**Temporal abstraction** -- compressing long tasks to a higher-level timescale. For example, "pick up the cup" is a high-level behavior, not 50 joint actions. pi0.5's semantic subtask belongs to this level.

**Planning** -- requires considering what state changes an action sequence will produce, and evaluating "which future is better." This is the core of world-model planning.

**Action chunk does not equal planning.** pi0 generates a 50-step action chunk, executes the first few steps, then re-observes and generates the next chunk. This is **receding-horizon policy execution**, not internally simulating multiple candidate futures and comparing them for selection.

### Cross-Embodiment Action Space

pi0 unifies the action representations of different embodiments into a common space of **up to 18 DoF**, with different robots performing padding / masking according to their own action spaces:

```
Franka single arm    -> 7 DoF
ALOHA dual arms      -> 14 DoF
Mobile manipulation  -> up to 18 DoF
```

This allows the same model to control robots of different morphologies -- this is one of pi0's important contributions.

### Training

Two stages:

**Stage 1 -- Broad pretraining:**
- Internet-scale image-text data (inherited from PaliGemma)
- Open X-Embodiment "Magic Soup" subset
- Proprietary multi-robot data: approximately **10,000 hours**, approximately **900 million timesteps**, covering **68 tasks** across **7 hardware configurations**

**Stage 2 -- Targeted post-training:**
- Fine-tuning on high-quality, curated task demos to acquire complex manipulation skills

### Key Results

Under the zero-shot real-world evaluation protocol reported in the paper, pi0 demonstrated capabilities that discrete token approaches did not possess on complex manipulation tasks:

| Task Type | Observations from pi0 Paper |
|-----------|---------------------------|
| Long-horizon cloth manipulation | Able to complete zero-shot manipulation |
| Desk clearing | Reports 97.1% success |
| Comparison with discrete VLAs | Clearly stronger on these specific zero-shot protocols |

Note that these observations are highly dependent on the specific task protocol (number of trials, robot embodiment, task definition, the exact meaning of "zero-shot," etc.), and therefore should not be interpreted as a universal cross-model performance ranking. However, they clearly show that tasks like cloth folding and desk clearing -- which require long-horizon, high-precision continuous manipulation -- are precisely the weakness of discrete token approaches.

### Understanding pi0's Performance Gains

From RT-2 to pi0, continuous action generation has indeed become an increasingly important direction. However, we cannot simply attribute the performance gains to "continuous representation being superior to discrete tokens." pi0 simultaneously introduced action chunking, flow matching, cross-embodiment training data, and larger-scale robot data. A more accurate conclusion is: **continuous action representation solves some structural problems of quantization and autoregressive output, but its ultimate benefit is strongly coupled with data scale, action chunking, and training recipe.**

### Limitations

- **Code and some model checkpoints have been open-sourced through openpi, but the training data, complete internal training system, and all production/experimental variants are not fully open**
- Some tasks requiring precise force control remain unreliable
- Generalization to entirely new physical domains (autonomous driving, aerial vehicles) is unknown
- VLM fine-tuning may lead to degradation of language/vision capabilities (catastrophic forgetting)

## 7. pi0.5 and pi0.7: From Action Generation to Policy Steering

### pi0.5: Data Composition + Hierarchical Policy

pi0.5's core contribution (April 2025, arXiv:2504.16054) is not simply "adding high-level reasoning," but introducing a **hierarchical architecture**, along with a very important technical finding: **discrete and continuous action representations can serve different roles within the same foundation model.**

pi0.5's training is divided into two stages, using different action representations and prediction targets:

```
Pretraining Stage (discrete)          Post-training Stage (continuous)
+---------------------+      +---------------------+
| FAST action tokenizer|      | Flow matching action |
| Discrete action token|  ->  | head                 |
| Semantic subtask     |      | Continuous high-freq |
|   prediction         |      |   control            |
| Multi-robot + web    |      | Target domain fine-  |
|   data               |      |   tuning             |
+---------------------+      +---------------------+
         |                              |
    Unified backbone               At inference: first predict
    representation                 semantic subtasks
                                   then generate continuous actions
                                   conditioned on subtasks
```

**pi0.5 does not simply replace a discrete policy with a continuous one; rather, it assigns different roles to discrete sequence modeling and continuous action generation at different levels and in different training stages.**

The **pretraining stage** uses **FAST action tokenizer** to discretize robot actions, enabling actions to share next-token prediction training with language and semantic subtask tokens. This allows large-scale heterogeneous data (different robots, different tasks, even web data) to be utilized within a unified sequence modeling framework. High-level **hierarchical task decomposition / semantic subtask prediction** is an important component of this stage.

The **post-training stage** introduces a **flow-matching action head** for high-frequency continuous control, targeting mobile manipulation post-training. During inference, the model first infers a high-level semantic subtask (e.g., "pick up the pillow"), then generates a continuous action chunk using flow matching conditioned on this subtask.

One notable data point: in pi0.5's first-stage training data, approximately **97.6% is not mobile-manipulator household data**. It gains generalization capability through mixing large amounts of heterogeneous data, then fine-tunes on a small amount of target-domain data.

pi0.5 therefore demonstrates that "discrete vs continuous" is not a simple substitution relationship. Its actual trajectory is not "discrete -> continuous" but rather **discrete for scalable multimodal pretraining + continuous for fine-grained control**.

pi0.5 is already capable of executing 10-15 minute long-horizon tasks in household environments not seen during training, but its success rate remains notably lower than for short tasks in controlled environments. This shows that error accumulation during long-horizon tasks remains a major bottleneck for VLA generalization.

### pi0-FAST: Why Discrete Tokens Have Not Disappeared?

Physical Intelligence subsequently explored **pi0-FAST**, using the FAST action tokenizer to compress continuous action chunks into discrete tokens, bringing robot actions back into the autoregressive language-modeling framework.

This shows that Physical Intelligence itself has not concluded that discrete action tokenization is a dead end. In fact, **discrete tokens still have enormous value for multimodal pretraining** -- they allow robot actions to share the same sequence modeling interface with language, vision, and semantic subtasks.

Here it is worth making an explicit comparison of the respective advantages of discrete and continuous:

**Advantages of discrete action tokens:**
- Naturally aligned with LLM/VLM autoregressive objectives
- Mixing language, vision, and actions into the same token space
- Leveraging large-scale multimodal pretraining
- Unified data format

**Advantages of continuous action (a in R^D):**
- Preserves spatial precision, avoiding 256-bin quantization loss
- Can model multi-modal action distributions
- Generates continuous action chunks
- Suitable for high-frequency fine control

So the conclusion is not "discrete -> continuous" but rather **tokenization and continuous generation may serve different stages / different levels.** The core thesis of this entire article can be compressed into one sentence:

> **Discrete handles unification; continuous handles control.**

### pi0.7: From Task Conditioning to Strategy Conditioning

pi0.7 (April 2026, arXiv:2604.15483) further explores VLA's "steerability."

From RT-2 to pi0.7, the policy's inputs underwent a qualitative change:

```
RT-2:   language -> action
pi0:    language + image + proprioception -> action chunk
pi0.5:  language + observation -> semantic subtask -> action chunk
pi0.7:  language + episode metadata + strategy + subgoal image + history -> policy -> action chunk
```

pi0.7's true advance is not simply "more inputs," but rather: **the prompt changed from "describing what to do" to "describing how to do it."** That is, a shift from **task conditioning** toward **strategy conditioning / policy steering**. This is precisely why pi0.7's title uses "Steerable Generalist."

The model no longer conditions only on language instructions, but unifies language, episode metadata, execution strategy information, visual subgoals, and observation history as multimodal context inputs to the policy.

pi0.7's model scale is approximately 5B, consisting of an approximately 4B VLM backbone, a video history encoding module (MEM-style video history encoder), and an 860M parameter action expert. It still follows the flow matching continuous action generation approach.

Notably, **pi0.7 itself is not an action-conditioned world model; however, its inference system can use subgoal images produced by an external lightweight visual generative model as future visual targets, so the combination of policy and predictive/generative model has begun to emerge.**

Results reported by pi0.7: achieving 85.6% task progress and 80% success rate on unseen robot embodiments, approaching the human teleoperator's 90.9% / 80.6%.

### Visual Subgoal is Not a World Model

With pi0.7 introducing visual subgoals, the boundary between VLA and world models begins to blur. However, it is important to note that **"using future visual subgoals" is not automatically equivalent to "having an explicit world model."** A true world model typically needs to learn action-conditioned transition dynamics and be able to perform future state prediction or rollout internally. A more accurate description of pi0.7 is that it introduces future visual goals as a conditioning signal in the policy.

This distinction matters: a model "seeing a future goal" and a model "being able to predict how the world will change after executing actions" are two different capabilities.

## 8. VLA and World Models: Policy Learning vs Predictive Modeling

This is the part I think deserves the deepest discussion. The technical evolution above has already touched on this question multiple times -- from RT-2's model-free policy, to pi0's action chunk not equaling planning, to pi0.7's visual subgoal not equaling a world model -- now let us bring these threads together.

### What VLA Lacks Is Not "Prediction Capability" but a Queryable Action-Conditioned Prediction Interface

A common simplification is: VLA can only do actions, world models can only do predictions. But this is not precise enough.

VLA can certainly make predictions -- a large enough autoregressive model can perfectly well predict the next frame. The real distinction is not about "whether there is prediction capability," but rather: **is prediction an explicit, queryable, action-conditioned interface of the model?**

Specifically:

- **VLA learns the action distribution**: pi(a_t | o_{<=t}, l) -- only needs to answer "what action should I take now?"
- **World model learns the future distribution**: p(z_{t+1:t+H} | z_t, a_{t:t+H-1}) -- answers "if I execute these actions, what will the future look like?"

With the latter, one can naturally form:

```
Candidate action a(1) -> predicted future o^(1) -> evaluate J(a(1))
Candidate action a(2) -> predicted future o^(2) -> evaluate J(a(2))
...
Select the action sequence with the highest J
```

This is the key to planning -- **generating multiple candidate futures internally, comparing them, and then selecting.** VLA's end-to-end policy does not have this queryable interface.

Note: a typical imitation-learning VLA does not explicitly learn a queryable action-conditioned dynamics model -- but the policy itself can implicitly encode dynamic priors. This is very different from "having no internal representation of the physical world at all."

### A More Accurate Distinction Framework

| Dimension | VLA / Policy | Action-conditioned World Model |
|-----------|-------------|-------------------------------|
| **Core Question** | What should I do now? | What will happen after I do it? |
| **Learning Objective** | pi(a \| o, l) | p(o_{t+1:t+H} \| o_t, a_{t:t+H}) |
| **Output** | Action commands | Predicted future states / representations |
| **Strengths** | Direct control, reaction speed | Prediction, comparing candidate futures |
| **Excels at** | execution | planning |
| **Weaknesses** | error accumulation | model bias / compute |
| **Requires action labels?** | Policy needs them | Action-conditioned version needs them |
| **Naturally requires search?** | No | Typically combines with search / MPC / optimization |
| **Ultimate role** | actor | predictor / planner |

Simply put: a VLA answers "what action should I take?"; a world model answers "what will the world look like after executing this action?"

### Passive World Models vs Action-conditioned World Models

There is another easily confused concept that needs distinguishing.

**Passive world models** can learn "how the world changes" using only video -- predicting o_{t+1} from o_t, without action labels.

**Action-conditioned world models** require (o_t, a_t, o_{t+1}) triplets, learning "**what results different actions will produce.**"

So what truly needs action-labeled interaction data is the **world model used for action-conditioned planning**. This also explains why V-JEPA 2 (passive video prediction) and V-JEPA 2-AC (action-conditioned) need to be separated in the technology stack -- JEPA is a predictive representation learning paradigm; only when its prediction process is further explicitly conditioned on actions and can be used to predict future states does it constitute an action-conditioned world model.

### The Unified Model Technical Framework

A true unified model needs to simultaneously answer two questions:

p(a_{t:t+H}, z_{t+1:t+H} | z_t, l, g)

That is, simultaneously learning:
- **What should I do?** (policy)
- **What will happen after I do it?** (prediction)

This is more precise than simply saying "VLA + world model" -- it defines a joint model that possesses both an action distribution and a future distribution.

### The Two Lines Are Converging

A common misconception needs correction: the world model line is not "without language" or "unable to do actions." V-JEPA 2 has already demonstrated the complete technology stack of web-scale video pretraining + action-conditioned world model + V-JEPA 2-AC, including zero-shot robot deployment and image-goal planning. World models themselves can also acquire semantic capabilities through language alignment.

Planning in the JEPA line is also not necessarily "generate multiple trajectories and select." It can be latent prediction -> goal-conditioned planning, implemented via search, optimization, or policy guidance.

So the more accurate picture is: **VLA and world models are approaching the same goal from two directions -- a robot foundation model that simultaneously possesses policy, prediction, and planning capabilities.** Future systems are more likely to be Actor + Predictor, not one or the other.

```
                 Robot Foundation Model
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       Language       Perception      Robot Data
          |              |              |
          +--------------+--------------+
                         |
                 Shared Representation
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       Semantic       Policy        Prediction
       Subtask          |              |
          |             v              v
          |       Action Chunk    Future State
          |             |              |
          |             v              v
          |       Flow Matching    World Model
          |             |              |
          +-------------+--------------+
                        |
                  Physical Action
```

**The future robot foundation model is likely neither a pure VLA nor a pure world model, but a unified system that simultaneously supports semantic conditioning, policy execution, and predictive modeling.**

### Are They Complementary?

I believe the answer is yes, but the complementarity is more subtle than "putting two modules together."

**What does a typical direct-policy VLA lack?** It does not explicitly learn a queryable action-conditioned dynamics model. When encountering novel situations not seen during training, it can only rely on the generalization capabilities learned during pretraining -- the policy can implicitly encode dynamic priors, but cannot explicitly simulate the consequences of actions the way a world model can.

**What does a world model lack?** While world models are gaining language and action capabilities, they still fall short of the VLA line in terms of the efficiency of end-to-end policy learning and the naturalness of language grounding.

So a natural idea is: **use the world model for physical prediction, and the VLA for action execution and language understanding.**

## 9. Open Questions

**Data bottleneck -- from "hours" to "data value."** A more meaningful question may not be "how to obtain a million hours of robot data," but rather: **should robot data really continue to be measured in "hours"?** One hour of a human continuously folding 300 garments successfully, and one hour of a robot encountering 50 failures, 20 recoveries, 10 different strategies, and 5 embodiments -- the information content is completely different. The value function for future data scaling may look more like:

Data Value = f(diversity, failure, recovery, embodiment, task coverage)

rather than simply Data Value proportional to hours. This connects directly to pi0.7's exploration of heterogeneous / suboptimal data. Many mainstream VLA datasets consist primarily of successful demonstrations, with failure / recovery data being relatively scarce. **How do we move from "successful demo datasets" to experience datasets that include failures, recoveries, and policy variations?** is the more critical question.

**Long-horizon tasks -- error propagation, not step count.** The real problem with long-horizon tasks is not H > 5 or H > 50, but the probabilistic effect of error accumulation:

P(success over T) approximately equals the product of P(correct_t)

Even if single-step success rate is P = 0.98, after 100 critical decisions, 0.98^100 approximately equals 13%. Of course, real robot tasks do not strictly follow this independent-events model, but it illustrates the point well: **the essence of long-horizon difficulty is error accumulation, not simply sequence length.** This also explains why hierarchical policy, recovery policy, replanning, world models, and memory are all natural directions for addressing long-horizon problems.

**Safety -- three levels.** VLA safety issues can be divided into three levels:

*Policy safety*: Will pi(a|o) output dangerous actions?

*Predictive safety*: p(o_future|o,a) -- will this action cause danger when executed?

*Runtime safety*: Even if the model is wrong, is there an independent safety layer to intercept?

A more complete system could be:

```
VLA
 |
candidate action
 |
world model / safety critic
 |
constraint checker
 |
robot
```

This has much more technical depth than simply saying "VLA needs safety mechanisms." Safety constraints cannot be treated merely as a language-level alignment problem -- they are hard engineering constraints.

**Missing modalities.** Current VLAs rely almost exclusively on vision and language. Touch, force feedback, and audio are severely underrepresented in training data. Yet for fine manipulation (screwing in bolts, inserting keys, folding soft objects), these modalities may be critical information sources.

**Does VLA need a world model?** I think this question does not yet have a definitive answer. pi0.7's introduction of visual subgoals as a conditioning signal does improve generalization, but this is not the same as "having an explicit world model." True integration may require a single model to simultaneously achieve: language grounding, action-conditioned prediction, and high-frequency continuous control. **There is currently no publicly demonstrated solution that addresses all three in a mature, unified manner and has been thoroughly validated on large-scale real robot tasks.**

**The final verdict on discrete vs continuous.** From RT-2's discrete tokens to OpenVLA-OFT's continuous regression to pi0's flow matching, continuous methods have demonstrated advantages in control precision and inference speed. But pi0.5 and pi0-FAST show that **discrete and continuous likely serve different roles: discrete handles "unification" -- enabling actions to share the sequence modeling interface with language, vision, and semantic subtasks; continuous handles "control" -- providing high-frequency, fine-grained continuous action output at the final execution stage.**

```
         Foundation-model pretraining
                    |
                    |
           discrete tokens
                    |
                    |
              shared LM space
                    |
                    v
       continuous action generation
                    |
                    v
           high-frequency control
```

This judgment is much stronger than "continuous will eventually replace discrete," and it better explains why pi0-FAST, pi0.5, and pi0.7 -- which might appear to be "going backwards" -- are actually exploring different model interfaces.

---

## 10. Three Judgments

Finally, let me distill the article's argument into three judgments.

**Judgment 1: The core progress of VLA is not that parameters are getting larger, but that the action interface is becoming better suited for robots.** From RT-2's 55B to OpenVLA's 7B to pi0's 3.3B, parameter counts are shrinking; but from 256-bin discrete tokens to parallel continuous regression to flow matching + 50-step action chunks, the action interface is continuously evolving. OFT achieved a 26x speed improvement by changing only the action interface, demonstrating that the bottleneck is not in the backbone but in the action interface.

**Judgment 2: The key bottleneck for generalist robot capability is shifting from representation scaling to data scaling, temporal abstraction, and recovery.** pi0.5's 97.6% non-target-domain data, pi0.7's utilization of suboptimal data, and the structural difficulty of error accumulation in long-horizon tasks all point in the same direction -- the next breakthrough depends not on making models larger, but on making data more diverse, temporal structures more robust, and failure recovery stronger.

**Judgment 3: The real next stage may not be "VLA or World Model," but the unification of policy, predictor, and planner.** The technology map for the future robot foundation model can be drawn as:

```
                         Robot Foundation Model
                                  |
              +-------------------+-------------------+
              |                   |                   |
          Representation      Action Interface     Temporal Structure
              |                   |                   |
          VLM / VLA         discrete token       single action
              |                   |                   |
       cross-modal FM       continuous action     action chunk
                                  |                   |
                           flow matching       semantic subtask
                                  |                   |
              +-------------------+-------------------+
              |
              v
       Generalist Policy
              |
              |        + predictive modeling
              |
              v
       Action-conditioned
         World Model
              |
              v
        Future prediction
              |
              v
        Planning / Safety
```

To summarize in one sentence:

> Robot Foundation Model = Perception + Language + Policy + Prediction + Planning

But immediately followed by a caveat: **today's public systems typically cover only a portion of this, and work like pi0.5/pi0.7 is more like gradually expanding this closed loop rather than having completed the unification.**

As I mentioned in the [world model survey](/en/articles/2026-09-01-world-model-h2-review/), "world model" is losing its singular meaning. The addition of VLA makes this picture more complex -- and more interesting.

*Next, I plan to dive deep into Sim-to-Real -- just how wide the deployment gap is from simulation to real robots, and what the current best transfer methods are.*
