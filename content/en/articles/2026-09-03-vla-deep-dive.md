---
title: "VLA Deep Dive (Part 1): From RT-2 to OpenVLA -- Foundations and Early Evolution of End-to-End Policies"
slug: "2026-09-03-vla-deep-dive"
date: 2026-09-03
draft: false
categories: ["Embodied Intelligence", "Paper Analysis"]
tags: ["VLA", "RT-2", "OpenVLA", "Vision-Language-Action", "Robot Foundation Model", "End-to-End Policy", "Embodied Intelligence"]
description: "Part 1 of a 3-part VLA series. From RT-2 injecting internet-scale knowledge into robot control, to OpenVLA surpassing a 55B closed-source model on 29 tasks with just 7B parameters, to OFT revealing the action interface as the bottleneck -- a complete technical breakdown of VLA foundations and early evolution across six axes."
toc: true
related_articles:
  - 2026-09-05-vla-pi-family
  - 2026-09-07-vla-world-models
  - 2026-09-02-jepa-deep-dive
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
---

> **VLA Series (3 parts):** Part 1 (this article) | [Part 2: The pi0 Family](/en/articles/2026-09-05-vla-pi-family/) | [Part 3: VLA and World Models](/en/articles/2026-09-07-vla-world-models/)
This article is **Part 1 of a 3-part series**, covering VLA from RT-2 to OpenVLA to OFT -- the core ideas, six evolution axes, and three representative early models. But I want to emphasize more than just a timeline -- **the evolution of VLA actually unfolds simultaneously along six axes: action representation, action generation mechanism, temporal abstraction level, context/embodiment conditioning, predictive/planning capability, and data heterogeneity.** Only by separating these six dimensions can we understand what each model truly solves.

*Note: This series' technical timeline focuses on pi0.7 (April 2026). The VLA field is developing rapidly; subsequent model variants are not covered here.*


## 1. The Core Idea of VLA

**VLA (Vision-Language-Action) is a class of foundation-model policies that unify visual observations, language/task conditioning, and robot action policies into a single modeling framework.** It emphasizes unified representation and joint learning between perception, language grounding, and action generation, and **does not require the implementation to be a single neural network.** The concrete implementation can still include multiple specialized modules -- such as an action expert, a history encoder, or a hierarchical action head -- but they share the same representational foundation.

Traditional robot control is staged: a perception module performs object detection and segmentation, a planning module handles task decomposition and path planning, and a control module executes PID or impedance control. Each module is designed independently, and modules are connected through manually defined interfaces.

The VLA approach folds these stages into a single end-to-end learning framework. The inputs are camera images and natural language instructions; the outputs are robot actions -- end-effector pose deltas, joint angles, gripper open/close. There is no explicit perception-planning-control separation, no manually designed intermediate representations.

> **A note on "end-to-end":** When this article says "end-to-end," it means that the task conditioning and robot action policy are directly connected through a unified training system -- not that the model internally lacks modular structure or intermediate representations. For example, pi0 includes a VLM backbone -> action expert -> flow matching -> action multi-stage structure, and pi0.5 even has a semantic subtask -> action generation hierarchical inference chain -- they are still end-to-end learned policies, just not "from pixels straight through to motor commands in a single undifferentiated pass."

The key turning point for this idea occurred in 2023.

## 2. Technical Evolution Roadmap

Before breaking down specific models, let us look at two parallel trajectories and six evolution dimensions. The VLA / policy line and the predictive / world model line are **not different points in the same coordinate system** -- they approach the same goal from different directions.

```
Main Line A: VLA / Policy Line
--------------------------------------------------------->
RT-2 -> OpenVLA -> OpenVLA-OFT -> pi0 -> pi0.5 -> pi0.7


Sub Line B: Predictive / World Model Line
--------------------------------------------------------->
V-JEPA -> V-JEPA 2 -> V-JEPA 2-AC -> ???
                   +---> (action-conditioned extension)
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
| RT-2 | Can actions become tokens? |
| OpenVLA | Can multi-robot data fit in one open VLA? |
| OpenVLA-OFT | Is the action decoder the bottleneck? |
| pi0 | Can continuous generation be a universal action interface? |
| pi0.5 | Can we separate the semantic layer and the control layer? |
| pi0.7 | Can we tell the policy not just WHAT but HOW / UNDER WHAT CONTEXT? |

This table reveals an "interface evolution" narrative -- from RT-2 to pi0.7, each step extends the interface boundary of robot foundation models.

### Six Evolution Dimensions

The earlier analysis used four axes. But upon closer examination, "action representation" and "action generation" are actually two independent dimensions, and "embodiment interface" runs through everything yet was not singled out. Upgrading to six axes:

| Evolution Axis | Core Question |
|----------------|--------------|
| **Action Representation** | How are actions represented? (discrete / continuous) |
| **Action Generation** | How are actions generated? (AR / parallel regression / flow matching) |
| **Temporal Abstraction** | How much time does one decision cover? (single action / chunk / semantic subtask) |
| **Context / Embodiment Conditioning** | How does the model know "what to do, how to do it, and for whom"? |
| **Predictive Modeling** | Can the model predict the consequences of actions? |
| **Data Heterogeneity** | Can data of different quality, from different robots and tasks, be utilized? |

```
              Action Representation        Action Generation
              discrete -> continuous        AR -> parallel -> flow matching
                       |                         |
                       +----------+--------------+
                                  |
              +-------------------+-------------------+
              |                   |                   |
     Temporal Abstraction   Context / Embodiment   Predictive Modeling
     single -> chunk ->       language -> language    implicit -> visual
     semantic subtask       + proprio + subtask    subgoal generator
                            + metadata + control   -> external world
                            mode + subgoal         model component
                                   |
                            Data Heterogeneity
                            single robot -> multi-robot
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

The significance of this finding is that it was among the early, representative systematic demonstrations proving that "internet-scale knowledge transfer to robot control" is feasible. Its core contribution can be summarized as a chain:

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
- **Emergent semantic capabilities** (capabilities completely absent from robot training scope):
  - Symbol understanding 82% (identifying and manipulating abstract symbols -- numbers, shapes, logos)
  - Person recognition 53% ("move the object to Taylor Swift")
  - Logical reasoning 46% ("place the object on top of 2+1")

Emergent capabilities are RT-2's most exciting result. These capabilities were not learned from robot training data but transferred from the VLM's internet pretraining knowledge. RT-1 performs near zero on these tasks.

### What Did RT-2 Actually Transfer?

This deserves further discussion. RT-2's emergent capabilities reveal an important hierarchical structure -- the knowledge that VLA transfers from internet pretraining and the capabilities learned from robot data are fundamentally different levels of things:

**Semantic knowledge.** "Cup," "red," "Taylor Swift," "inside," "on top of," "2+1" -- these concepts and relationships come from internet text and images. VLM pretraining provides abundant such priors.

**Visual grounding.** Seeing a completely novel object and being able to judge what semantic category it belongs to, understanding its relationship to language instructions. The VLM's visual encoder, trained on large-scale image-text pairs, provides powerful visual generalization capability.

**Physical skills.** Seeing a cup -> how to reach -> how to grasp firmly -> how to control force -> how to avoid collisions -- this entire chain from perception to force control, **internet data can provide certain physical interaction priors, but typically cannot directly provide high-frequency action supervision aligned with the target robot embodiment, action space, and control interface.**

This precisely explains a core phenomenon of RT-2: semantic generalization improves dramatically, but it cannot spontaneously generate new physical skills. Internet pretraining gives VLA a powerful "semantic engine," but precise robot manipulation skills still depend primarily on robot interaction data or other embodiment-aligned control data.

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
- Each visual encoder produces 256 spatial patch tokens at 224x224 input with 14x14 patch configuration
- Features from both encoders are concatenated along the channel dimension, yielding a 2176-dimensional representation

**Projection Layer:** A 3-layer MLP (GELU activation) maps the 2176-dimensional visual features to the LLM's 4096-dimensional embedding space. OpenVLA's VLA training does not simply freeze the visual side -- it adapts visual representations for robot data -- this is one of the more counterintuitive design choices in the paper.

**Language Backbone:** Llama-2 7B (32-layer Transformer decoder), with visual patch tokens, language instruction tokens, and action tokens forming the input sequence together.

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

**On some zero-shot generalization evaluations that rely on web-scale semantic knowledge, RT-2-X retains an advantage.** This conclusion comes from the zero-shot evaluations reported in the OpenVLA paper, not the 29-task main benchmark. RT-2 directly inherits backbones like PaLI-X / PaLM-E that underwent large-scale internet multimodal pretraining, while OpenVLA's training focus is robot data -- this relates to differences in their pretraining data and backbone origins.

### Limitations

- **Autoregressive decoding bottleneck**: generating actions token by token, inference speed is approximately 4.2 Hz
- **The original model's inputs and training recipe center primarily on single-frame visual observations**, with limited capability for modeling historical visual information and complex multi-view scenes
- **Real robot deployment still requires embodiment / task-specific adaptation**; the original model's generalization capability cannot be directly equated with production-grade reliability
- **Potential discretization precision limitation**: fixed 256-bin representation introduces quantization error, which may become a limitation for tasks requiring high-precision continuous control

## 5. OpenVLA-OFT: The Real Bottleneck May Be the Action Interface

OpenVLA-OFT (2025, arXiv:2502.19645) should not be viewed merely as an optimization of OpenVLA in the article's technical timeline. From a technical evolution perspective, it answers a critical question:

> **"Is the performance bottleneck of VLA the foundation model itself, or the action decoding?"**

OpenVLA's action decoding generates discrete tokens autoregressively, one token at a time. OFT performed a thorough overhaul:

- **Parallel decoding** replacing autoregressive: generating all action tokens simultaneously
- **Action chunking**: predicting 8-step actions in a single forward pass
- **Continuous action head**: using MLP + L1 regression instead of discrete tokens -- note, this is **continuous regression**, not flow matching
- **LoRA fine-tuning**

Results:

| Metric | Original OpenVLA | OpenVLA-OFT |
|--------|-----------------|-------------|
| policy/action-generation throughput | 4.2 Hz | **109.7 Hz** |
| LIBERO success rate | 76.5% | **97.1%** |
| Speed improvement | -- | **26x** |

It is important to note that these frequency numbers are not the same metric. The following table lists the specific meaning of each model's reported number:

| Model | Reported Number | Metric Meaning |
|-------|----------------|----------------|
| RT-2 | 1-3 Hz | policy inference / control frequency |
| OpenVLA | ~4.2 Hz | autoregressive action generation |
| OpenVLA-OFT | 109.7 Hz | action-generation throughput under benchmark configuration |
| pi0 | up to 50 Hz | reported robot control frequency |

The pi0 paper explicitly states its action chunk is H=50 with 10 Euler integration steps; the paper describes the maximum 50 Hz as robot control frequency. One should not directly compare 109.7 Hz with 50 Hz -- the two have different measurement scopes.

OFT's results demonstrate that the **action interface, decoding strategy, and temporal chunking** of a robot VLA are themselves important system design axes, not merely secondary concerns to backbone scaling. Primarily by restructuring the action interface -- from token-by-token autoregressive to parallel decoding + action chunking + continuous action head -- OFT achieved a dramatic speed improvement and success rate increase.

This aligns closely with the main thread we will see later: "Discrete handles unification; continuous handles control."

There is a point of confusion that needs clarification here: **"continuous action" and "flow matching" are not the same concept.** "Discrete tokens -> continuous action" is an evolution along one dimension (action representation), while "regression / diffusion / flow matching" is a choice along another dimension (the specific generation mechanism for continuous actions). OpenVLA-OFT takes the continuous regression path, while pi0 takes the flow matching path -- both belong to "continuous action" but with different generation mechanisms.


---

**Next (Part 2)** enters the most technically dense part of the VLA story: pi0's flow matching architecture, pi0.5's discrete-continuous hybrid recipe, pi0.7's context-rich policy steering, and the Training Interface vs Execution Interface decoupling meta-axis.

> **VLA Series:** Part 1 (this article) | [Part 2: The pi0 Family](/en/articles/2026-09-05-vla-pi-family/) | [Part 3: VLA and World Models](/en/articles/2026-09-07-vla-world-models/)
