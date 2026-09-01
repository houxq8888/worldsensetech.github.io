---
title: "VLA Deep Dive: From RT-2 to OpenVLA to π₀ — How End-to-End Policies Connect Language and Action"
slug: "2026-09-03-vla-deep-dive"
date: 2026-09-03
draft: false
categories: ["Embodied AI", "Paper Analysis"]
tags: ["VLA", "RT-2", "OpenVLA", "π₀", "Vision-Language-Action", "Robot Foundation Models", "End-to-End Policy", "Flow Matching", "Embodied AI", "Physical Intelligence"]
description: "VLA (Vision-Language-Action) models are reshaping the robotics learning pipeline — from RT-2 injecting internet knowledge into robot control, to OpenVLA beating a 55B closed-source model with 7B parameters, to π₀ achieving continuous action generation and 50Hz control via flow matching. This article breaks down the VLA technology evolution, compares discrete tokens vs continuous flow matching, and discusses the fundamental relationship between VLA and world models."
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

This article provides a complete technical breakdown of VLA from RT-2 to OpenVLA to π₀, focusing on the fundamental divergence in action representation — discrete tokens vs continuous generation — and the relationship between VLA and world models.

## 1. The Core Idea of VLA

**One neural network: see the image, understand the language, output the action.**

Traditional robot control is staged: a perception module does object detection and segmentation, a planning module does task decomposition and path planning, and a control module executes PID or impedance control. Each module is engineered separately, connected by hand-designed interfaces.

VLA collapses all these stages into a single end-to-end model. The inputs are camera images and natural language instructions; the outputs are robot actions — end-effector pose deltas, joint angles, gripper state. There is no explicit perception-planning-control separation, no hand-designed intermediate representation.

The key turning point for this approach happened in 2023.

## 2. Technology Evolution Roadmap

Before diving into specific models, here is a timeline:

```
RT-1 (2022.12)     → Validate Transformer + action tokens for robot control
    ↓
RT-2 (2023.07)     → Transfer VLM internet knowledge to robot control; first use of "VLA"
    ↓
OpenVLA (2024.06)  → Open-source 7B VLA outperforms 55B RT-2-X on most tasks
    ↓
π₀ (2024.10)       → Abandon discrete tokens; flow matching for continuous action generation
    ↓
π₀.5 (2025.04)     → Add high-level semantic reasoning; zero-shot generalization across environments
    ↓
π₀.7 (2026.04)     → Introduce visual world model component; move toward World-Action Model
```

Two key dimensions of change along this path. First, action representation moved from discrete tokens to continuous generation. Second, model capability expanded from "executing trained tasks" to "zero-shot generalization to entirely new environments."

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

RT-2 represents robot actions as an 8-dimensional tuple: termination signal, 3D translation (Δx, Δy, Δz), 3D rotation (roll, pitch, yaw), gripper state. Each continuous dimension is uniformly quantized into **256 discrete bins**.

The key engineering trick is **symbol tuning**: rather than expanding the VLM vocabulary (which would break pretrained weights), RT-2 reuses the **256 least-frequently-used tokens** in the VLM's vocabulary and redefines them as action bin IDs. This way the model architecture stays completely unchanged, the pretrained language and visual knowledge is fully preserved, and actions simply become "another language the model speaks."

During inference, when the model enters the action output phase, a decoding mask ensures only these 256 action tokens carry non-zero probability.

### Training

RT-2 uses mixed training to prevent catastrophic forgetting:

- **Robot data**: approximately 100K episodes from Everyday Robots teleoperation in kitchen settings, covering about 700 language instructions
- **Internet data**: millions of image-text pairs to maintain the VLM's general visual-language capabilities

The mixing ratio is roughly 50:50 (PaLI-X variant) to 66:34 (PaLM-E variant).

### Key Results

RT-2's core numbers:

- **Trained task success rate**: 91-93%, comparable to RT-1's 92%
- **Novel scenario generalization** (new objects, backgrounds, environments): average **62%**, vs RT-1's **32%** — nearly 2× improvement
- **Emergent semantic capabilities** (abilities completely absent from robot training data):
  - Symbol understanding: 82% (identifying and manipulating abstract symbols — numbers, shapes, logos)
  - Person recognition: 53% ("move the object to Taylor Swift's side")
  - Logical reasoning: 46% ("place the object on the sum of 2 + 1")

Emergent capabilities are RT-2's most exciting result. These abilities were not learned from robot training data but transferred from the VLM's internet pretraining knowledge. RT-1 scored near zero on all of these.

### Limitations

RT-2's limitations are equally clear:

- **Cannot invent new physical skills.** It applies existing manipulation skills to new objects and scenes but cannot learn genuinely new motor capabilities from internet knowledge alone.
- **Inference speed is constrained.** The 55B model requires cloud TPU inference at only 1-3 Hz control frequency, far below real robot control needs.
- **Action precision is limited by quantization.** 256 bins may not be sufficient for fine manipulation.

### Relationship to World Models

RT-2 is a **model-free** method. It builds no world model, predicts no future observations, simulates no physical dynamics. It is a direct perception-to-action mapping — except that this mapping is enriched by internet-scale semantic knowledge.

Chain-of-thought reasoning is the closest RT-2 gets to "planning," but that is linguistic reasoning, not physical simulation.

## 4. OpenVLA: Open-Source 7B Beats 55B

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

**Action output:** Similar approach to RT-2 — 256 special tokens added to the LLM tokenizer for action discretization (replacing lowest-frequency tokens), each action dimension generated autoregressively. Outputs 7-DoF actions for the WidowX robot.

### Training

- **Data**: Open X-Embodiment dataset, 970K real robot manipulation demonstrations
- **Compute**: 64 A100 GPUs, approximately 15 days
- **Fine-tuning strategy**: LoRA modifying only 1.4% of parameters matches full fine-tuning performance

### Key Results

This is where OpenVLA delivers its most surprising result:

| Model | Parameters | 29-task avg success rate |
|-------|------------|-------------------------|
| **OpenVLA** | **7B** | **+16.5% over RT-2-X** |
| RT-2-X | 55B | Baseline |
| OpenVLA vs Diffusion Policy | — | +20.4% |
| OpenVLA vs RT-1-X / Octo | — | Outperforms both |

A 7B-parameter model beats the 55B closed-source model on most tasks.

But there is one important exception: **on high-difficulty semantic generalization tasks (requiring internet-scale knowledge to understand concepts), RT-2-X remains stronger.** OpenVLA's Open X-Embodiment training data does not include internet-scale image-text pretraining, so for "understanding semantic concepts never seen in robot data," it falls short of RT-2.

### Limitations

- **Autoregressive decoding bottleneck**: token-by-token generation limits inference to approximately 4.2 Hz
- **Single-image input**: no multi-view or stereo vision support
- **Insufficient zero-shot reliability**: sub-90% success rate without fine-tuning, not enough for real deployment
- **Quantization precision loss**: 256-bin quantization remains insufficient for fine manipulation

### Follow-up: OpenVLA-OFT

OpenVLA-OFT (2025, arXiv:2502.19645) made a thorough overhaul targeting the autoregressive bottleneck:

- **Parallel decoding** replacing autoregressive generation: all action tokens generated simultaneously
- **Action chunking**: multiple action steps predicted in one forward pass
- **Continuous action head**: MLP + L1 regression replacing discrete tokens
- **LoRA fine-tuning**

Results:

| Metric | Original OpenVLA | OpenVLA-OFT |
|--------|-----------------|-------------|
| LIBERO success rate | 76.5% | **97.1%** |
| Inference speed | 4.2 Hz | **109.7 Hz** |
| Speed improvement | — | **26×** |

This improvement direction is already very close to π₀'s approach — continuous actions, parallel generation, high-frequency control.

## 5. π₀: Continuous Actions + Flow Matching — A New Paradigm

### Paper Info

*π₀: A Vision-Language-Action Flow Model for General Robot Control*, Physical Intelligence, October 2024. arXiv:2410.24164.

### Physical Intelligence Background

Physical Intelligence (also "π") was founded in 2023 in San Francisco, with the mission of "building a generalist robot brain." Five co-founders include Karol Hausman (CEO, former Google DeepMind, core contributor to SayCan / RT-2), Chelsea Finn (Stanford, inventor of MAML), Sergey Levine (UC Berkeley, co-inventor of SAC), Brian Ichter, and Jasmine Hsu (both from Google Brain).

As of 2026, the company has raised over $2 billion cumulatively, with valuation exceeding $10 billion.

### Core Architecture

π₀ made a fundamentally different choice from RT-2 / OpenVLA: **no discrete tokens — use flow matching to generate continuous actions.**

The architecture has two parts:

| Component | Description |
|-----------|-------------|
| VLM backbone | PaliGemma (3B-parameter vision-language model) |
| Action expert | 300M-parameter dedicated network appended to the VLM |
| **Total parameters** | **approximately 3.3B** |

### What Flow Matching Does

The discrete token approach quantizes each action dimension into one of 256 bins, then uses the language model's next-token prediction to generate them. This is conceptually simple but has two problems: quantization error and autoregressive latency.

Flow matching works completely differently. The action expert learns a vector field that progressively transforms Gaussian noise into continuous action trajectories. Specifically:

- Define a probability path from a pure noise distribution to the target action distribution
- Train a network to predict the velocity field along this path
- At inference, start from noise and integrate along the learned vector field to obtain continuous actions

This is more natural than discrete tokens because robot actions are inherently continuous. Quantizing a continuous joint angle into 256 bins is like representing a curve with pixels — it works, but it is not the most natural representation.

### Action Chunking and High-Frequency Control

π₀ generates **50 action steps** per forward pass (action chunk), achieving **50 Hz** control frequency.

This is an order of magnitude faster than RT-2's 1-3 Hz and OpenVLA's 4.2 Hz. The reason is straightforward: no token-by-token autoregressive decoding — the action expert generates the entire chunk in parallel.

Action dimensions support up to 18 (dual arms + mobile base), with zero-padding for simpler robots.

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

### Limitations

- Full model weights are not open-sourced (only the openpi research package supports DROID/Franka and ALOHA platforms)
- Some tasks requiring precise force control remain unreliable
- Generalization to fundamentally different physical domains (autonomous driving, aerial vehicles) is unknown
- VLM fine-tuning may cause language/vision capability degradation (catastrophic forgetting)

## 6. Core Technical Comparison Across the Three Approaches

| Dimension | RT-2 (2023) | OpenVLA (2024) | π₀ (2024) |
|-----------|-------------|----------------|-----------|
| **Parameters** | 5B / 55B | 7B | 3.3B |
| **VLM backbone** | PaLI-X / PaLM-E | Prismatic (Llama-2 7B) | PaliGemma (3B) |
| **Visual encoder** | ViT (single) | SigLIP + DINOv2 (dual) | PaliGemma built-in |
| **Action representation** | Discrete 256 bins | Discrete 256 bins | **Continuous flow matching** |
| **Action dimensions** | 8 | 7 | 18 |
| **Action chunk** | None | None (added in OFT) | 50 steps |
| **Control frequency** | 1-3 Hz | 4.2 Hz (OFT: 109.7 Hz) | 50 Hz |
| **Training data** | 100K episodes + internet | 970K demos (Open X) | 10K hours + 900M timesteps |
| **Open-source** | No | Yes | Partial (openpi) |
| **Core advantage** | Internet semantic knowledge transfer | Reproducible, cost-effective | Continuous actions, high-frequency control |
| **Core limitation** | Slow inference, closed-source | Autoregressive bottleneck | Closed-source, large data requirements |

The most noteworthy aspect of this table is not who is "best," but that **the action representation dimension underwent a directional shift over three years**: from RT-2's discrete tokens, to OpenVLA-OFT's continuous action head, to π₀'s native flow matching.

I think the logic behind this shift is clear: robot actions are inherently continuous, and discretization is an engineering simplification. Once model scale and data scale are large enough, working directly with continuous representations becomes more efficient.

## 7. VLA vs World Models: The Fundamental Difference

This is the part I think deserves the deepest discussion.

### Core Distinction

| Dimension | VLA | World Models (Dreamer / JEPA etc.) |
|-----------|-----|-------------------------------------|
| **Core function** | Perception → action direct mapping | Learn dynamics, imagine futures, then plan |
| **Output** | Action commands | Predicted future states |
| **Planning approach** | Implicit (learned end-to-end) | Explicit (imagine multiple trajectories, evaluate, select) |
| **Analogy** | Reflexive policy | Mental simulation |
| **Data needs** | Manipulation trajectory demos | State transitions / physical dynamics data |
| **Strengths** | Fast reactive control, language grounding | Imagination for novel situations, synthetic data |

Simply put: VLA is a highly skilled "reflex arc" — see the scene, directly output the action. A world model is an "inner theater" — first simulate different action consequences in imagination, then choose the best one.

### Are They Complementary?

I believe the answer is yes, but the complementarity is more subtle than "plugging two modules together."

**What does VLA lack?** VLA has no explicit physical prediction capability. It cannot "imagine" what the world will look like after executing a particular action. When encountering novel situations not seen in training, it can only rely on generalization learned during pretraining — it cannot evaluate different options through internal simulation.

**What do world models lack?** World models (especially the JEPA path) currently lack a natural interface with language. V-JEPA 2-AC can do action-conditioned prediction and planning, but it cannot understand instructions like "put the red cup on the blue plate."

So a natural idea emerges: **use world models for physical prediction, use VLA for action execution and language understanding.**

### π₀.7's Convergence Attempt

π₀.7 (April 2026, arXiv:2604.15483) is the latest attempt along these lines. It adds a **14B-parameter visual generator** (mixture-of-transformers) to the π₀ foundation, synthesizing multi-view future state images as visual subgoals to guide the action expert.

The action expert also grew from 300M to **860M parameters**, still using flow matching for 50-step continuous trajectories.

π₀.7's reported results: 85.6% task progress and 80% success rate on unseen robot embodiments, approaching human teleoperators' 90.9% / 80.6%.

NVIDIA calls this class of models **World-Action Models (WAM)** — combining world model prediction capabilities with VLA action capabilities.

### But the Deeper Question Remains

π₀.7's visual generator is a pixel-level prediction model — it generates future images. This creates an interesting contrast with the JEPA path: JEPA argues prediction should happen in representation space, not pixel space.

So the real question is: **when world models provide "imagination" to VLA, should it be pixel-level images or abstract representations?**

There is no definitive answer yet. But I tend to think that for tasks requiring precise physical manipulation, pixel-level visual subgoals may be more directly useful than abstract representations — because robots ultimately need to act in a pixel world. For tasks requiring long-horizon reasoning, representation-level prediction may be more efficient.

## 8. Open Problems

**Data bottleneck.** Current VLA training data quality is uneven. Most datasets contain only successful demos, no failure trajectories — models cannot learn "what not to do." Physical data collection costs two to three orders of magnitude more than text/image data. I think future VLA progress will depend more on data quality than model architecture.

**Long-horizon tasks.** Even the strongest π₀.5 achieves only about 65% success rate for kitchen + bedroom cleaning in completely unfamiliar homes. Manipulation chains beyond five steps remain challenging for all VLAs. This is not a model scale problem but a structural issue of long-range dependencies and error accumulation.

**Safety.** VLAs are unsupervised physical agents — they can directly harm people. RL-based safety alignment is still in early stages. When VLA models are deployed in home environments, safety will shift from academic discussion to engineering hard constraint.

**Missing modalities.** Current VLAs rely almost exclusively on vision and language. Tactile sensing, force feedback, and audio are severely underrepresented in training data. Yet for fine manipulation (screwing bolts, inserting keys, folding soft objects), these modalities may be critical information sources.

**Does VLA need world models?** I think this question has no definitive answer yet. From π₀ to π₀.7, adding world model components did improve generalization. But whether VLA's core advantages — fast, end-to-end, language-grounded — will be diluted by adding world models requires more experimental validation.

**Discrete vs continuous final answer.** From RT-2's discrete tokens to π₀'s flow matching, the trend seems to point toward continuous methods. But discrete methods retain a persistent advantage: unity with language models. Actions and language sharing the same token space means the entire language model infrastructure can be directly reused. Continuous methods are better for control precision and speed, but still have gaps in architectural unity.

---

The VLA path's core insight — **a sufficiently large end-to-end model can learn robot actions directly from vision and language** — has been progressively validated through experiments from RT-2 to π₀.7. From 55B closed-source giants to 7B open-source alternatives to 3.3B flow matching models, parameter efficiency is improving, action quality is rising, and generalization capability is expanding.

But the relationship between VLA and world models is only beginning to unfold. π₀.7 introducing visual world model components, WA-JEPA re-examining JEPA's applicability to action modeling — these developments all point in the same direction: **pure perception-to-action mapping and pure world model prediction may not be the final answer; their convergence is.**

As I concluded in the [world model roundup](/en/articles/2026-09-01-world-model-h2-review/), "world model" is losing its singular meaning. VLA's entry makes this picture more complex — and more interesting.

*Next up, I plan to dive into Sim-to-Real — just how wide the deployment gap is from simulation to real robots, and what the current best transfer methods are.*
