---
title: "JEPA Deep Dive: From I-JEPA to V-JEPA 2 — LeCun's Predictive Representation Learning Path"
slug: "2026-09-02-jepa-deep-dive"
date: 2026-09-02
draft: false
categories: ["World Models", "Paper Analysis"]
tags: ["JEPA", "I-JEPA", "V-JEPA", "V-JEPA 2", "AMI Labs", "LeCun", "Predictive Representation Learning", "Self-Supervised Learning", "World Models", "Embodied AI"]
description: "From the 2022 theoretical blueprint to I-JEPA in 2023, V-JEPA in 2024, V-JEPA 2 in 2025, and AMI Labs' $1.03B funding in 2026 — LeCun's JEPA path took four years to go from 'predicting representations beats predicting pixels' to 'zero-shot robot manipulation.' This article breaks down the core ideas and key experiments across the entire JEPA series."
toc: true
related_articles:
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - world-model-intro
  - 2026-08-25-dreamer-explained
  - world-model-transformer
---

In my [previous world model roundup](/en/articles/2026-09-01-world-model-h2-review/), I covered Cosmos, Genie 3, and Marble in detail but only sketched JEPA.

Not because JEPA isn't important. Quite the opposite — **JEPA may be the most theoretically deep path in the current world model landscape.** It's backed by Yann LeCun, has a complete technical evolution from I-JEPA to V-JEPA 2, has been validated by AMI Labs' $1.03 billion funding round, and represents a fundamentally different technical philosophy from today's dominant generative world models.

This article provides a complete technical breakdown of the JEPA series from the first paper to the latest.

## 1. JEPA's Core Idea: The One-Sentence Version

**Don't predict pixels — predict representations.**

This sounds simple, but it directly opposes the fundamental assumption of most current world models (including Cosmos and the Genie series).

Generative world models work like this: given historical observations, predict future pixels. JEPA works like this: given historical observations, predict **abstract representations** of future observations — make predictions in representation space, not pixel space.

Why does this distinction matter? Because pixel space contains a large amount of task-irrelevant details: lighting changes, texture details, random noise. Requiring the model to precisely predict all these details wastes model capacity and introduces useless gradient signals that slow down learning.

LeCun made this point clearly in his [2022 position paper](https://openreview.net/forum?id=BZ5a1r-kVsf): **for learning high-level semantics and world state representations, requiring the model to precisely predict all pixels is not an ideal learning objective.**

This isn't to say pixel-level prediction has no value — video generation is certainly impressive. JEPA's argument is: if your goal is to learn world representations useful for downstream tasks, then predicting in representation space is a more efficient learning strategy.

## 2. I-JEPA: Proving It Works on Images (2023)

### Paper Info

*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, Meta AI, CVPR 2023. [Paper](https://arxiv.org/abs/2301.08243), [Code](https://github.com/facebookresearch/ijepa).

### Architecture

I-JEPA has three core components:

**Context Encoder**: A Vision Transformer that receives visible image patches and outputs their representations.

**Target Encoder**: Same architecture as the Context Encoder, but parameters are updated via exponential moving average (EMA) — it doesn't directly participate in gradient computation but slowly tracks the Context Encoder's parameter changes. It encodes masked regions into target representations.

**Predictor**: A much lighter Transformer (embedding dimension 384 vs the encoder's larger dimension) that receives the Context Encoder's visible representations and predicts the masked regions' representations in the Target Encoder's space.

### Masking Strategy

This is where I-JEPA differs fundamentally from MAE (Masked Autoencoder).

MAE masks at the pixel level — randomly masking 75% of patches and asking the model to reconstruct those pixels.

I-JEPA masks at the **representation level**. Specifically: it uses a multi-block strategy, selecting 4 target regions (each 15%-20% of the image) and 1 context region (85%-100%), removing overlaps. The key difference is that masking happens at the Target Encoder's output, not at the input pixels.

This means: the model doesn't need to reconstruct masked content from the pixel level. Instead, it needs to **infer** what the masked regions should look like in abstract representation space from the visible regions' representations.

### Key Results

I-JEPA with ViT-H/14 on ImageNet:

- **Linear probing**: 79.3% top-1 accuracy (vs MAE's 77.2%)
- **448 resolution**: 81.1% top-1
- **1% low-shot**: 73.3% (vs MAE's 59.8%) — this gap is enormous
- **Full fine-tuning**: 87.1%, using only 1/5.3 the epochs of MAE

The last point is particularly noteworthy: **I-JEPA reaches comparable performance to MAE but with over 10x better training efficiency.** This directly validates LeCun's core thesis — predicting in representation space is more efficient than predicting in pixel space.

### Why It Matters

I-JEPA's significance isn't about SOTA numbers. It cleanly demonstrates one thing: **self-supervised learning that doesn't generate pixels and only predicts in representation space not only works but is more efficient than generative methods.**

This is the first cornerstone of the JEPA path.

## 3. V-JEPA: From Images to Video (2024)

### Paper Info

*V-JEPA: Latent Video Prediction for Visual Representation Learning*, Meta AI, ICLR 2024. Authors: Quentin Garrido et al.

### Core Extension

If I-JEPA proved "predicting image patches in representation space works," V-JEPA asks: **can we predict future video frames in representation space?**

This is much harder than images. Masked regions in images are spatially fixed — the model just needs to understand spatial structure. But predicting future frames in video involves temporal dynamics — the model needs to understand motion, causality, and temporal evolution.

V-JEPA extends the I-JEPA framework to the spatiotemporal dimension. The Context Encoder receives spatiotemporal patches from historical video frames, and the Predictor predicts future frames' abstract representations in representation space.

### Technical Highlights

V-JEPA maintains JEPA's core design philosophy:

- Target Encoder still updated via EMA
- Prediction still happens in representation space, never returning to pixel space
- Masking strategy extends from spatial to spatiotemporal — masking parts of future frames

### Key Contribution

V-JEPA's core contribution is proving the JEPA framework can naturally extend to video, and the learned representations perform well on downstream tasks like action recognition and video understanding.

But honestly, V-JEPA is more of a "feasibility proof" — it demonstrated the path works but hadn't yet shown game-changing capabilities. The real breakthrough came with V-JEPA 2.

## 4. V-JEPA 2: From Representation Learning to World Model (2025)

### Paper Info

*V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction, and Planning*, Meta AI, June 2025. [Paper](https://arxiv.org/abs/2506.09985).

### This Is the Most Important One

If I-JEPA and V-JEPA are basic research, V-JEPA 2 is where the JEPA path first truly demonstrates "I can do what world models do."

### Architecture Upgrade

V-JEPA 2's architecture has major upgrades:

**Encoder**: Vision Transformer, up to **1 billion parameters**. Video input is split into 2x16x16 tubelets (2 frames × 16 × 16 pixels). Positional encoding uses **3D-RoPE** (3D Rotary Position Embedding), a key design for processing spatiotemporal sequences.

**Action-conditioned variant**: This is V-JEPA 2's most important addition. A **300M parameter Transformer** using **block-causal attention**, receiving action sequences as conditional input to predict future representations.

This means: the model can not only "understand" video but also "imagine" what the world would look like if a certain action were executed. This is the core capability of a world model.

### Pretraining

The pretraining scale is substantial:

- **22 million samples**, over 1 million hours of video data
- **252K iterations**
- Progressive resolution strategy (starting from low resolution, gradually increasing)
- Mask-denoising feature prediction objective

### Key Results

**Video understanding**:

- **88.2 average** across 6 benchmarks
- Motion understanding **77.3%** top-1
- Action prediction recall@5 **39.7**
- Video QA (PerceptionTest) **84.0**
- Surpasses InternVideo2 and DINOv2

**Robot manipulation (this is the most exciting part)**:

V-JEPA 2 was post-trained on 62 hours of unlabeled robot manipulation data, then tested for zero-shot real robot manipulation:

- Target reaching accuracy: **less than 4 cm**
- Cup pick-and-place: **80% success rate**
- Planning time per action: **16 seconds**

For comparison:

- **Cosmos baseline**: 4 minutes to compute a single action, and fails object manipulation tasks
- **Octo baseline**: can reach but can't grasp, 0% success on box grasp

This comparison is very compelling. V-JEPA 2 is not only two orders of magnitude faster than the generative approach (16 seconds vs 4 minutes) but also succeeds at manipulation tasks from zero.

### Why V-JEPA 2 Is the JEPA Path's Turning Point

V-JEPA 2 demonstrates for the first time:

1. The JEPA framework can scale to the billion-parameter level
2. Action-conditioned prediction can serve real robot control
3. Predicting in representation space is not just theoretically elegant but also more efficient in practice than pixel-level prediction
4. Zero-shot robot manipulation — no need to retrain for each task

This is exactly the blueprint LeCun drew in his 2022 paper: **using predictive representation learning to build AI systems that understand the physical world.** Three years later, V-JEPA 2 turned that blueprint into a running system.

## 5. AMI Labs: From Papers to Company (2026)

### Basic Info

In March 2026, Yann LeCun founded [AMI Labs](https://amilabs.xyz/) (Advanced Machine Intelligence Labs) in Paris, completing approximately **$1.03 billion** (approximately €890 million) in seed funding. This is one of the largest seed rounds in European AI foundation model history.

### Team

AMI Labs' team is remarkably strong:

- **Yann LeCun**: Executive Chairman
- **Saining Xie**: Chief Science Officer — NYU professor, one of the most influential computer vision researchers alive, author of MAE
- **Alex LeBrun**: CEO
- **Michael Rabbat**: VP of World Models
- **Laurent Solly**: COO
- **Pascale Fung**: Chief Research and Innovation Officer

Offices in Paris, New York, Montreal, and Singapore.

### Investors

The seed round was co-led by Cathay Innovation, Greycroft, Hiro Capital, and HV Capital. Individual investors include **Jeff Bezos, NVIDIA, Eric Schmidt, Mark Cuban**.

### Technical Direction

AMI Labs officially states it builds world models that understand physical environments, retain long-term information, execute logical planning, and maintain secure operations. They believe true intelligence originates from physical environments rather than text, focusing on learning abstract representations from multimodal sensor inputs, filtering out unpredictable details, and predicting outcomes in conceptual space.

Notably, AMI Labs explicitly emphasizes **action-conditioned world models** — enabling autonomous agents to predict action consequences and plan subsequent steps.

This aligns closely with V-JEPA 2's technical direction. Given that Saining Xie (CSO) is one of the core authors of the JEPA paper series, it's reasonable to infer AMI Labs' technical path is the industrialization extension of JEPA.

### Target Domains

Industrial process control, automation, robotics, personal health monitoring, healthcare. These are all domains requiring high controllability and safety — precisely where generative models struggle most with reliability.

## 6. JEPA vs Dreamer/RSSM: Both Predict in Latent Space — What's the Difference?

If you've read my [RSSM deep dive](/en/articles/rssm-deep-dive/), you'll notice an obvious similarity between JEPA and RSSM/Dreamer: **both predict in latent space rather than directly predicting pixels.**

But they're not the same thing.

### Similarities

Both recognize pixel space isn't a good prediction target. RSSM compresses observations into latent states via an encoder, then uses a dynamics model to predict the next state in latent space; JEPA encodes observations into representation space via an encoder, then uses a predictor to predict future representations.

### Core Differences

**Training objectives differ.** RSSM/Dreamer's latent dynamics model ultimately serves RL — it needs an explicit state → action → next state structure to support planning and control. JEPA's training objective is self-supervised representation learning — it doesn't explicitly model the separation of state and action, instead learning useful representations through masking and prediction.

**Action's role differs.** In Dreamer, action is an explicit input to the dynamics model — state transitions directly depend on action. In the original I-JEPA and V-JEPA, there's no concept of action. Only V-JEPA 2 introduced an action-conditioned variant, but its action conditioning differs from Dreamer's RSSM — V-JEPA 2 uses block-causal attention to process action sequences and visual representations together.

**Application scenarios differ.** Dreamer's core scenario is RL and control — imagining futures, evaluating actions, selecting optimal policies. JEPA's core scenario is representation learning and understanding — learning world representations useful for downstream tasks. V-JEPA 2's robot experiments show JEPA can also do control, but differently from Dreamer.

### One-Sentence Summary

Dreamer is "using latent dynamics models for planning"; JEPA is "using predictive representation learning for understanding." Both are clever but solve different problems.

## 7. JEPA's Position in the World Model Landscape

Going back to the four technology paths from my [roundup article](/en/articles/2026-09-01-world-model-h2-review/):

- **Latent Dynamics** (Dreamer/RSSM): state → action → next state
- **Generative Video** (Cosmos): condition → future video frames
- **Interactive** (Genie 3): state + action → interactive future
- **Spatial/3D** (Marble): persistent spatial representation

JEPA doesn't cleanly fit into any of these. It's closest to Latent Dynamics since it also predicts in latent space. But unlike Dreamer, it doesn't have an explicit state transition structure, nor is RL and planning its primary goal.

If I had to categorize it, I'd say **JEPA represents a fifth path: Predictive Representation Learning.** Its core contribution isn't "how to model world dynamics" but "what objective function to use for learning world representations."

This is also why I said in the roundup that JEPA isn't "a complete world-model definition" — it's more of a **learning philosophy**: rather than generating all details, predict what's useful in abstraction.

## 8. What This Means for Practitioners

### If You're Doing Robot Control

V-JEPA 2's zero-shot manipulation results deserve serious attention. 16 seconds/action vs Cosmos' 4 minutes/action — this efficiency gap is decisive in real-time control scenarios. If your system needs fast inference rather than beautiful video generation, the JEPA path may be more suitable.

### If You're Doing Self-Supervised Learning

I-JEPA's training efficiency advantage (over 10x faster than MAE) is already attention-worthy. The JEPA framework provides a self-supervised learning paradigm that doesn't rely on pixel reconstruction, which is particularly valuable as compute costs receive increasing scrutiny.

### If You're Doing World Model Research

The JEPA path provides an important alternative: not all world models need to generate pixels. If your downstream tasks require understanding rather than generation, predicting in representation space may be the better choice.

### If You're Following AMI Labs

AMI Labs' team and funding scale tell us one thing: investors have strong conviction about the JEPA path's industrial value. But stay clear-eyed — the distance from papers to products remains vast. AMI Labs hasn't yet released public products or benchmark results beyond V-JEPA 2's scope.

## 9. Open Questions for the JEPA Path

Finally, a few questions the JEPA path hasn't fully answered yet:

**Representation collapse.** JEPA uses EMA target encoder to avoid collapse, but this isn't a theoretically perfect solution. Like contrastive learning, JEPA needs careful training objective design to ensure representations don't degenerate to constants.

**Long-range prediction stability.** V-JEPA 2 demonstrated short-range action prediction capability, but for tasks requiring long-range planning (like multi-step robot manipulation), will JEPA's predictions gradually diverge? Dreamer addresses this through KL balancing and imagined rollout; JEPA currently has no equivalent mechanism.

**Language alignment.** A core capability of current VLA (Vision-Language-Action) models is language grounding. The JEPA path currently works primarily in visual and action space — how to integrate with language capabilities is an important open question.

**Scalability ceiling.** V-JEPA 2 reached 1 billion parameters, but compared to LLMs' hundreds of billions, there's still a large gap. What does JEPA's scaling law look like? Can it scale continuously like LLMs? These questions remain unanswered.

---

The JEPA path's core insight — **predicting representations is more efficient than predicting pixels** — has been validated both theoretically and experimentally. From I-JEPA to V-JEPA 2 to AMI Labs, this path is moving from academic hypothesis to engineering practice.

But it's not the only answer for world models. As I concluded in my [roundup article](/en/articles/2026-09-01-world-model-h2-review/), "world model" is losing its singular meaning. JEPA answers "how to learn world representations," not "how to do world planning" or "how to generate training data." Different questions require different tools.

*Next up, I'll discuss team configuration for starting a company in embodied AI — not a business plan, but a pragmatic analysis from an engineer's perspective.*
