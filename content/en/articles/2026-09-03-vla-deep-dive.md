---
title: "VLA Deep Dive: From RT-2 to OpenVLA to π₀ — How End-to-End Policies Connect Language and Action"
slug: "2026-09-03-vla-deep-dive"
date: 2026-09-03
draft: false
categories: ["Embodied AI", "Paper Analysis"]
tags: ["VLA", "RT-2", "OpenVLA", "π₀", "Vision-Language-Action", "Robot Foundation Models", "End-to-End Policy", "Flow Matching", "Embodied AI", "Physical Intelligence"]
description: "VLA (Vision-Language-Action) models are reshaping the robotics learning pipeline — from RT-2 injecting internet knowledge into robot control, to OpenVLA surpassing a 55B closed-source model with 7B parameters on 29 tasks, to π₀ achieving continuous action generation and high-frequency control via flow matching. This article breaks down the VLA technology evolution across three independent dimensions, compares discrete tokens with continuous flow matching, and discusses the relationship between VLA and world models."
toc: true
related_articles:
  - 2026-09-02-jepa-deep-dive
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - world-model-intro
  - 2026-08-25-dreamer-explained
---

At the end of the [previous JEPA deep dive](/en/articles/2026-09-02-jepa-deep-dive/), I raised an open question: the JEPA path currently works primarily in visual and action space — how does it integrate with language capabilities?

That question points to another core path in embodied AI — VLA (Vision-Language-Action).

If JEPA's core question is "what should a world model predict?", then VLA's core question is "how should a robot turn what it sees and hears into physical action?" The two paths start from different premises, but both must ultimately answer the same foundational question: **how does perceptual information translate into physical action?**

This article provides a complete technical breakdown of VLA from RT-2 to OpenVLA to π₀. But what I want to emphasize is not just a timeline — **VLA's evolution actually unfolds simultaneously across three independent dimensions: action representation, temporal abstraction, and prediction/planning capability.** Separating these three dimensions is the key to understanding what each model actually solved.

## 1. The Core Idea of VLA

**One neural network: see the image, understand the language, output the action.**

Traditional robot control is staged: a perception module does object detection and segmentation, a planning module does task decomposition and path planning, and a control module executes PID or impedance control. Each module is engineered separately, connected by hand-designed interfaces.

VLA collapses all these stages into a single end-to-end model. The inputs are camera images and natural language instructions; the outputs are robot actions — end-effector pose deltas, joint angles, gripper state. There is no explicit perception-planning-control separation, no hand-designed intermediate representation.

The key turning point for this approach happened in 2023.

## 2. Technology Evolution Roadmap

Before diving into specific models, consider the evolution across two dimensions. A simple timeline can conflate progress on different axes — in reality, VLA has advanced simultaneously on **action representation** and **decision horizon** axes.

```
                        Decision / Planning horizon
                              ↑
                              │
                   π₀.5       │      V-JEPA 2-AC
                   (semantic  │     (action-conditioned
                    hierarchy)│        planning)
                              │
                   π₀.7       │
                  (multimodal │
                   steering)  │
                              │
   RT-2 ────── OpenVLA ───────┼────── OpenVLA-OFT
  (discrete    (discrete      │     (continuous regression +
   tokens)      tokens)       │      parallel decoding + chunk)
                              │
   ───────────────────────────┼────────────────────→
                              │        Action representation
                    discrete tokens → continuous regression → flow matching
```

Placing models in this diagram reveals several key observations:

- RT-2 and OpenVLA belong to the same generation in action representation (discrete tokens), but OpenVLA achieved better cost-effectiveness with an open-source approach
- OpenVLA-OFT solved continuous actions and inference speed, but through continuous regression rather than flow matching
- π₀ chose flow matching as its continuous action generation mechanism, while also introducing action chunking
- π₀.5 and π₀.7 primarily advance the decision horizon dimension — from single-step actions to semantic-level subgoals and multimodal steering
- V-JEPA 2-AC advances action-conditioned prediction and planning from the other direction

## 3. RT-2: Transferring Internet Knowledge to Robot Control

### Paper Info

*RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, Google DeepMind, July 2023. arXiv:2307.15818.

### Core Insight

RT-2's core finding: **a vision-language model pretrained on large-scale internet text and images can be directly fine-tuned into a robot policy — and the resulting policy can leverage semantic knowledge from pretraining for zero-shot generalization.**

The significance of this finding is that it systematically demonstrated, for the first time, that "internet-scale knowledge transfer to robot control" is feasible.

### Architecture

RT-2 did not build a new architecture from scratch. It took existing VLMs and fine-tuned them:

| Variant | Backbone | Parameters |
|---------|----------|------------|
| RT-2 PaLI-X | PaLI-X (ViT visual encoder) | 55B |
| RT-2 PaLM-E | PaLM-E (multimodal embedding) | 12B |

Images are processed through the backbone's native visual encoder; language goes through the standard VLM tokenizer. The key innovation is on the action side.

### Action Discretization: Actions as "Another Language"

RT-2 represents robot actions as **7 continuous quantities**: end-effector 3D position delta (Δx, Δy, Δz), 3D rotation delta (roll, pitch, yaw), and gripper state. Each dimension is uniformly quantized into **256 discrete bins**. The termination signal belongs to the control logic of the action sequence, not to these 7 continuous action dimensions.

The key engineering trick is **symbol tuning** — discretizing actions into tokens so that actions can share the same autoregressive output interface as language. The specific implementation differs by backbone: **PaLI-X directly uses existing token IDs to represent action bins, while PaLM-E overwrites 256 low-frequency tokens in its vocabulary, redefining them as action tokens.** Either way, the effect is the same: the model architecture stays unchanged, the pretrained language and visual knowledge is fully preserved, and actions simply become "another language the model speaks."

During inference, when the model enters the action output phase, a decoding mask ensures only these 256 action tokens carry non-zero probability.

### Training

RT-2 performs co-finetuning on top of existing RT-1 / Everyday Robots robot data and VLM web data to prevent catastrophic forgetting. The two backbone variants do not have identical training recipes — the paper reports different data mixing strategies for each. The core idea is consistent: retain some internet image-text data to maintain the VLM's general semantic capabilities, while mixing in robot manipulation demos to learn action output.

### Key Results

RT-2's core numbers:

- **Trained task success rate**: 91-93%, comparable to RT-1's 92%
- **Novel scenario generalization** (new objects, backgrounds, environments): average **62%**, vs RT-1's **32%** — nearly 2× improvement
- **Emergent semantic capabilities** (abilities completely absent from robot training data):
  - Symbol understanding: 82% (identifying and manipulating abstract symbols — numbers, shapes, logos)
  - Person recognition: 53% ("move the object to Taylor Swift's side")
  - Logical reasoning: 46% ("place the object on the sum of 2 + 1")

Emergent capabilities are RT-2's most exciting result. These abilities were not learned from robot training data but transferred from the VLM's internet pretraining knowledge. RT-1 scored near zero on all of these.

### What Did RT-2 Actually Transfer?

This deserves further discussion. RT-2's emergent capabilities reveal an important layered structure — the knowledge VLA migrates from internet pretraining and the capabilities it learns from robot data are fundamentally different kinds of things:

**Semantic knowledge.** "Cup," "red," "Taylor Swift," "inside," "above," "two plus one" — these concepts and relationships come from internet text and images. VLM pretraining provides extensive priors of this kind.

**Visual grounding.** Seeing a never-before-seen object and being able to judge what semantic category it belongs to, understanding its relationship to language instructions. The VLM's visual encoder, trained on large-scale image-text pairs, provides strong visual generalization capability.

**Physical skills.** Seeing a cup → how to reach → how to grasp firmly → how to control force → how to avoid collision — this entire chain from perception to force control, **internet data basically cannot provide directly.**

This precisely explains a core phenomenon in RT-2: semantic generalization improves dramatically, but it cannot spontaneously generate new physical skills. Internet pretraining gave VLA a powerful "semantic engine," but physical manipulation capability still depends entirely on robot demo data.

### Limitations

RT-2's limitations are equally clear:

- **Cannot invent new physical skills.** It applies existing manipulation skills to new objects and scenes but cannot learn genuinely new motor capabilities from internet knowledge alone.
- **Inference speed is constrained.** The 55B model requires cloud TPU inference at only 1-3 Hz control frequency, far below real robot control needs.
- **Action precision is limited by quantization.** 256 bins may not be sufficient for fine manipulation.

### Relationship to World Models

RT-2 can be viewed as a typical **model-free / direct policy** method: it does not explicitly learn an action-conditioned dynamics model, but instead directly learns the observation + instruction → action mapping. Chain-of-thought reasoning is the closest RT-2 gets to "planning," but that is linguistic reasoning, not physical simulation.

Note, however, that this description applies primarily to RT-2's generation of VLA. Subsequent VLAs have begun introducing action chunking, hierarchical semantic actions, subgoal conditioning, and other mechanisms, gradually adding more temporal structure on top of the "direct mapping" foundation.

## 4. OpenVLA: Open-Source 7B VLA

### Paper Info

*OpenVLA: An Open-Source Vision-Language-Action Model*, Stanford / UC Berkeley, June 2024, CoRL 2024. arXiv:2406.09246. Authors: Moo Jin Kim, Karl Pertsch, Chelsea Finn, et al.

### Core Insight

RT-2 proved VLA feasibility but was completely closed-source — backbone models were Google's internal PaLM/PaLI-X, unreproducible by outside researchers. OpenVLA's goal: **build a reproducible VLA from open-source components and see how far an open approach can go.**

The answer: further than expected.

### Architecture

OpenVLA is a 7B-parameter model built on the Prismatic VLM architecture:

**Vision side — dual encoders:**
- SigLIP (1152-dim features) + DINOv2 (1024-dim features), two frozen visual encoders working in parallel
- 224×224 images split into 14×14 patches, yielding 256 visual patch tokens + 1 CLS token
- Features from both encoders concatenated along the channel dimension, producing 2176-dim representations

**Projection layer:** 3-layer MLP (GELU activation) maps 2176-dim visual features into the LLM's 4096-dim embedding space.

**Language backbone:** Llama-2 7B (32-layer Transformer decoder), approximately 280-token input sequence (BOS + visual patches + language instruction + action tokens).

**Action output:** Similar approach to RT-2 — 256 special tokens added to the LLM tokenizer for action discretization, each action dimension generated autoregressively. Outputs 7-DoF actions for the WidowX robot.

### Training

- **Data**: Open X-Embodiment dataset, 970K real robot manipulation demonstrations
- **Compute**: 64 A100 GPUs, approximately 15 days
- **Fine-tuning strategy**: LoRA modifying only 1.4% of parameters matches full fine-tuning performance

### Key Results

On the **29 tasks across multiple robot embodiments' average task success rate** reported in the paper, OpenVLA exceeds RT-2-X by 16.5 percentage points, with only about 1/7 the parameters:

| Model | Parameters | Comparison |
|-------|------------|------------|
| **OpenVLA** | **7B** | **+16.5pp over RT-2-X (29-task avg)** |
| RT-2-X | 55B | Baseline |
| OpenVLA vs Diffusion Policy | — | +20.4% |
| OpenVLA vs RT-1-X / Octo | — | Outperforms both |

**This "surpassing" refers to 29-task average success, not comprehensive superiority over RT-2-X.** There is one important exception: **on high-difficulty semantic generalization tasks (requiring internet-scale knowledge to understand concepts), RT-2-X remains stronger.** OpenVLA's Open X-Embodiment training data does not include internet-scale image-text pretraining, so for "understanding semantic concepts never seen in robot data," it falls short of RT-2.

### Limitations

- **Autoregressive decoding bottleneck**: token-by-token generation limits inference to approximately 4.2 Hz
- **Single-image input**: no multi-view or stereo vision support
- **Insufficient zero-shot reliability**: sub-90% success rate without fine-tuning, not enough for real deployment
- **Quantization precision loss**: 256-bin quantization remains insufficient for fine manipulation

### Follow-up: OpenVLA-OFT

OpenVLA-OFT (2025, arXiv:2502.19645) made a thorough overhaul targeting the autoregressive bottleneck:

- **Parallel decoding** replacing autoregressive generation: all action tokens generated simultaneously
- **Action chunking**: 8 action steps predicted in one forward pass
- **Continuous action head**: MLP + L1 regression replacing discrete tokens — note, this is **continuous regression**, not flow matching
- **LoRA fine-tuning**

Results:

| Metric | Original OpenVLA | OpenVLA-OFT |
|--------|-----------------|-------------|
| LIBERO success rate | 76.5% | **97.1%** |
| Inference speed | 4.2 Hz | **109.7 Hz** |
| Speed improvement | — | **26×** |

An important point about OpenVLA-OFT: it demonstrates that **without tokenizing actions and without using flow matching, VLA can still achieve high-speed control through parallel decoding + action chunking + continuous action head.**

There is an easily confused distinction that needs clarification here: **"continuous actions" and "flow matching" are not the same concept.** "Discrete tokens → continuous actions" is one dimension of evolution (action representation), while "regression / diffusion / flow matching" is another dimension (the specific generation mechanism for continuous actions). OpenVLA-OFT uses continuous regression; π₀ uses flow matching — both belong to "continuous actions" but differ in generation mechanism.

## 5. π₀: Flow Matching + Action Chunking

### Paper Info

*π₀: A Vision-Language-Action Flow Model for General Robot Control*, Physical Intelligence, October 2024. arXiv:2410.24164.

### Physical Intelligence Background

Physical Intelligence (also "π") was founded in 2023 in San Francisco, with the mission of "building a generalist robot brain." Five co-founders include Karol Hausman (CEO, former Google DeepMind, core contributor to SayCan / RT-2), Chelsea Finn (Stanford, inventor of MAML), Sergey Levine (UC Berkeley, co-inventor of SAC), Brian Ichter, and Jasmine Hsu (both from Google Brain). As of 2026, Physical Intelligence has raised approximately $2.1 billion cumulatively, with the most recent round corresponding to a valuation of approximately $11 billion.

### Core Architecture

π₀ made a different choice from RT-2 / OpenVLA: **no discrete tokens — use flow matching to generate continuous actions.**

The architecture has two parts:

| Component | Description |
|-----------|-------------|
| VLM backbone | PaliGemma (3B-parameter vision-language model) |
| Action expert | 300M-parameter dedicated network appended to the VLM |
| **Total parameters** | **approximately 3.3B** |

### What Flow Matching Does

Flow matching works completely differently from discrete tokens. The action expert learns a vector field that progressively transforms Gaussian noise into continuous action trajectories. Specifically:

- Define a probability path from a pure noise distribution to the target action distribution
- Train a network to predict the velocity field along this path
- At inference, start from noise and integrate along the learned vector field to obtain continuous actions

Robot actions are inherently continuous. Quantizing a continuous joint angle into 256 bins is like representing a curve with pixels — it works, but it is not the most natural representation. Flow matching works directly in continuous space, avoiding quantization error.

### Action Chunking and High-Frequency Control

π₀ generates an action chunk containing **50 future actions** per inference; at a control frequency of **up to 50 Hz**, this corresponds to approximately 1 second of future trajectory. The paper reports its system can achieve up to 50 Hz control frequency on dexterous tasks.

Note that flow matching inference itself requires multiple integration steps. The 50 Hz figure is the system-level control frequency including action chunk execution, not the speed of a single flow matching inference.

### Cross-Embodiment Action Space

π₀ supports action dimensions up to 18 — this is not simply "more dimensions" but addresses an important engineering problem: **different robots have completely different action spaces.**

```
Franka single arm  → 7 DoF
ALOHA dual arm     → 14 DoF
Mobile manipulator  → 18 DoF
```

π₀ needs to align different embodiments' action spaces into a unified interface. Its approach uses 18 dimensions as the maximum common space, with zero-padding and masking for simpler robots. This allows the same model to control robots of different form factors — this is an important contribution of π₀ that should not be glossed over as just a number.

### Training

Two stages:

**Stage 1 — Broad pretraining:**
- Internet-scale image-text data (inherited from PaliGemma)
- Open X-Embodiment "Magic Soup" subset
- Proprietary multi-robot data: approximately **10,000 hours**, approximately **900M timesteps** across **68 tasks** and **7 hardware configurations**

**Stage 2 — Targeted post-training:**
- Fine-tuning on high-quality, curated task demonstrations to master complex manipulation skills

### Key Results

π₀ dramatically outperforms discrete token approaches on complex manipulation tasks:

| Task | π₀ | OpenVLA | Octo |
|------|-----|---------|------|
| Zero-shot garment folding | ~100% | ~0% | ~0% |
| Simple table clearing | 97.1% | ~0% | ~0% |

Tasks like garment folding and table clearing require long sequences of precise continuous operations — exactly where discrete token approaches struggle.

### How to Understand π₀'s Performance Gains

From RT-2 to π₀, continuous action generation has indeed become an increasingly important direction. However, the performance improvement cannot be simply attributed to "continuous representation is better than discrete tokens." π₀ simultaneously introduced action chunking, flow matching, cross-embodiment training data, and larger-scale robot data. A more accurate conclusion: **continuous action representation solves some structural problems of quantization and autoregressive output, but its ultimate benefits are strongly coupled with data scale, action chunking, and training recipe.**

### Limitations

- Full model weights are not open-sourced (only the openpi research package supports DROID/Franka and ALOHA platforms)
- Some tasks requiring precise force control remain unreliable
- Generalization to fundamentally different physical domains (autonomous driving, aerial vehicles) is unknown
- VLM fine-tuning may cause language/vision capability degradation (catastrophic forgetting)

## 6. π₀.5 and π₀.7: From Action Generation to Policy Steering

### π₀.5: Hierarchical Semantic Action Prediction

π₀.5's core contribution (April 2025, arXiv:2504.16054) is not simply "adding high-level reasoning" but introducing a **hierarchical architecture**:

```
observation + instruction ("clean the bedroom")
        ↓
High-level semantic subtask ("pick up the pillow")
        ↓
Low-level continuous action chunk (flow matching generation)
```

This semantic subtask → low-level action hierarchy enables the model to handle 10-15 minute long-horizon tasks.

A noteworthy data point: approximately **97.6% of π₀.5's Stage 1 training data is not mobile-manipulator household data.** It achieves generalization through mixed training on large amounts of heterogeneous data (different robots, different task types), then fine-tunes on a small amount of target domain data.

π₀.5 can already execute 10-15 minute long-horizon tasks in home environments not seen during training, but its success rate remains significantly lower than for short tasks in controlled environments. This indicates that error accumulation in long-horizon tasks remains the primary bottleneck for VLA generalization.

### π₀.7: Steering a Generalist Policy Through Multimodal Context

π₀.7 (April 2026, arXiv:2604.15483) further explores VLA's "steerability." Its core innovation is not simply "adding a world model," but rather:

**The model no longer takes only language instructions as conditioning, but unifies language, episode metadata, execution strategy information, visual subgoals, and observation history as multimodal context input to the policy.**

π₀.7's model scale is approximately 5B, consisting of an approximately 4B VLM backbone, a video history encoding module (MEM-style video history encoder), and an 860M-parameter action expert. It continues to use flow matching for continuous action generation, but the focus shifts from "how to generate actions" to "**how to tell the model what strategy to adopt through rich context.**"

Therefore, rather than simply understanding π₀.7 as "π₀ with a world model added," a more accurate characterization is: **it explores how a generalist robot policy can be steered through multimodal context, achieving cross-task, cross-environment, and cross-embodiment generalization by leveraging heterogeneous data.**

π₀.7's reported results: 85.6% task progress and 80% success rate on unseen robot embodiments, approaching human teleoperators' 90.9% / 80.6%.

### Visual Subgoal ≠ World Model

With π₀.7 introducing visual subgoals, the boundary between VLA and world models begins to blur. However, note that **"using future visual subgoals" is not automatically equivalent to "having an explicit world model."** A true world model typically needs to learn action-conditioned transition dynamics and be able to perform future state prediction or rollout internally. π₀.7 more accurately introduces future visual goals as conditioning signals in the policy.

This distinction matters: a model that "sees a future goal" and a model that "can predict how the world will change after executing an action" are two different capabilities.

## 7. Core Technical Comparison

| Dimension | RT-2 (2023) | OpenVLA (2024) | OpenVLA-OFT (2025) | π₀ (2024) |
|-----------|-------------|----------------|--------------------|-----------| 
| **Core approach** | VLM → VLA | Open-source VLA | VLA optimization | VLA + flow matching |
| **Parameters** | 5B / 12B / 55B | 7B | 7B backbone + heads | ~3.3B |
| **Action representation** | Discrete 256-bin token | Discrete token | **Continuous regression** | **Continuous flow matching** |
| **Action chunk** | No | No | **Yes, 8 steps** | **Yes, 50 steps** |
| **Decoding** | Autoregressive | Autoregressive | **Parallel** | Flow integration |
| **Control/inference speed** | 1-3 Hz (55B) | ~4.2 Hz | **109.7 Hz** | **Up to 50 Hz** |
| **Action dimensions** | 7 | 7 | 7 | 18 (cross-embodiment) |
| **Main contribution** | Internet knowledge transfer | Open scalable VLA | Speed/success optimization | Continuous dexterous control |
| **Training data** | Robot demos + web VLM | 970K episodes | OpenVLA fine-tuning | 10K h + OXE/DROID/Bridge |
| **Open-source** | No | Yes | Yes | Partial (openpi) |

The most noteworthy aspect of this table is not who is "best," but the **evolution across two independent dimensions**:

**Action representation dimension**: discrete tokens (RT-2, OpenVLA) → continuous regression (OpenVLA-OFT) → continuous flow matching (π₀). OpenVLA-OFT and π₀ both belong to "continuous actions" but differ in generation mechanism — regression is simpler and more direct, flow matching is better suited for complex multimodal distributions.

**Inference efficiency dimension**: autoregressive token-by-token (RT-2, OpenVLA) → parallel decoding (OpenVLA-OFT) → flow integration + action chunk (π₀). OpenVLA-OFT's 109.7 Hz shows that parallel decoding + continuous regression alone can achieve very high inference frequency — flow matching is not required.

## 8. VLA vs World Models: Policy Learning vs Predictive Modeling

This is the part I think deserves the deepest discussion.

### A More Accurate Framing

The difference between the VLA path and the world model path lies not in "whether they have language" or "whether they have actions," but rather in their **learning objectives**:

- **VLA's core objective** is learning a policy — observation + instruction → action
- **World model's core objective** is learning action-conditioned state transitions / future representations — enabling the system to predict action consequences and perform planning

| Dimension | VLA | World Models (Dreamer / JEPA etc.) |
|-----------|-----|-------------------------------------|
| **Core function** | Learn policy | Learn dynamics |
| **Output** | Action commands | Predicted future states / representations |
| **Planning approach** | Implicit (learned end-to-end) | Explicit (imagine multiple trajectories, evaluate, select) |
| **Analogy** | Reflexive policy | Mental simulation |
| **Data needs** | Manipulation trajectory demos | State transitions / physical dynamics data |
| **Strengths** | Fast reactive control, language grounding | Imagination for novel situations, synthetic data |

Simply put: VLA is a highly skilled "reflex arc" — see the scene, directly output the action. A world model is an "inner theater" — first simulate different action consequences in imagination, then choose the best one.

### The Two Paths Are Converging

A common misconception needs correcting: the world model path is not "without language" or "cannot do actions." V-JEPA 2 has already demonstrated a complete technology stack of web-scale video pretraining + action-conditioned world model + V-JEPA 2-AC, including zero-shot robot deployment and image-goal planning. World models themselves can also gain semantic capabilities through language alignment.

So the more accurate picture is: **VLA and world models are approaching the same goal from two directions — a robot foundation model that simultaneously possesses policy, prediction, and planning capabilities.**

### Are They Complementary?

I believe the answer is yes, but the complementarity is more subtle than "plugging two modules together."

**What does VLA lack?** VLA has no explicit physical prediction capability. It cannot "imagine" what the world will look like after executing a particular action. When encountering novel situations not seen in training, it can only rely on generalization learned during pretraining — it cannot evaluate different options through internal simulation.

**What do world models lack?** Although world models are gaining language and action capabilities, in terms of end-to-end policy learning efficiency and naturalness of language grounding, they currently still fall short of the VLA path.

So a natural idea emerges: **use world models for physical prediction, use VLA for action execution and language understanding.**

## 9. Open Problems

**Data bottleneck.** Current VLA training data quality is uneven. Most datasets contain only successful demos, no failure trajectories — models cannot learn "what not to do." Physical data collection costs two to three orders of magnitude more than text/image data. I think future VLA progress will depend more on data quality than model architecture.

**Long-horizon tasks.** π₀.5 can already execute 10-15 minute long-horizon tasks in home environments not seen during training, but its success rate remains significantly lower than for short tasks in controlled environments. Manipulation chains beyond five steps remain challenging for all VLAs. This is not a model scale problem but a structural issue of long-range dependencies and error accumulation.

**Safety.** VLAs are unsupervised physical agents — they can directly harm people. RL-based safety alignment is still in early stages. When VLA models are deployed in home environments, safety will shift from academic discussion to engineering hard constraint.

**Missing modalities.** Current VLAs rely almost exclusively on vision and language. Tactile sensing, force feedback, and audio are severely underrepresented in training data. Yet for fine manipulation (screwing bolts, inserting keys, folding soft objects), these modalities may be critical information sources.

**Does VLA need world models?** I think this question has no definitive answer yet. π₀.7 introducing visual subgoals as conditioning signals did improve generalization, but this is not the same as "having an explicit world model." True convergence may require a model that can simultaneously achieve: language grounding, action-conditioned prediction, and high-frequency continuous control. No such model exists yet.

**Discrete vs continuous.** From RT-2's discrete tokens to OpenVLA-OFT's continuous regression to π₀'s flow matching, continuous methods have shown clear advantages in control precision and inference speed. But discrete methods retain a persistent advantage: unity with language models. Actions and language sharing the same token space means the entire language model infrastructure can be directly reused. Continuous methods are better for control precision and speed, but still have gaps in architectural unity. This trade-off currently has no definitive answer.

---

The VLA path's core insight — **a sufficiently large end-to-end model can learn robot actions directly from vision and language** — has been progressively validated through experiments from RT-2 to π₀.7. From RT-2's large-scale VLM, to OpenVLA's open-source 7B model, to π₀'s approximately 3.3B VLM+action-expert architecture, researchers are increasingly focused on **how to achieve stronger actual control capability with smaller policy models combined with better pretraining, robot data, and action generation mechanisms.**

But VLA's evolution is not simply a march from "language model outputs action tokens" to "flow matching." It actually unfolds across three dimensions simultaneously:

**Action representation**: discrete tokens → continuous regression → flow matching

**Temporal abstraction**: single-step action → action chunk → hierarchical semantic subgoals

**Prediction / planning**: direct policy → subgoal-conditioned policy → world model / planning

RT-2 primarily solved an early version of the first question: enabling VLMs to "speak actions." OpenVLA proved this path can be open-sourced and scaled. OpenVLA-OFT further solved the inference bottleneck — note it uses continuous regression, not flow matching. π₀ pushes VLA toward high-frequency fine continuous control through flow matching and action chunking. π₀.5 tackles long-horizon tasks and semantic hierarchy in unfamiliar environments. π₀.7 investigates how to steer a generalist policy through multimodal context. Meanwhile, V-JEPA 2-AC and related work advance action-conditioned prediction and planning from the other direction.

**Therefore, the real question is not "will VLA eventually become a world model," but whether future robot foundation models will simultaneously possess policy, prediction, and planning capabilities.**

As I concluded in the [world model roundup](/en/articles/2026-09-01-world-model-h2-review/), "world model" is losing its singular meaning. VLA's entry makes this picture more complex — and more interesting.

*Next up, I plan to dive into Sim-to-Real — just how wide the deployment gap is from simulation to real robots, and what the current best transfer methods are.*
