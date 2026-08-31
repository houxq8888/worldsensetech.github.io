---
title: "JEPA Deep Dive: From I-JEPA to V-JEPA 2-AC — How Predictive Representation Learning Leads to World Models"
slug: "2026-09-02-jepa-deep-dive"
date: 2026-09-02
draft: false
categories: ["World Models", "Paper Analysis"]
tags: ["JEPA", "I-JEPA", "V-JEPA", "V-JEPA 2", "V-JEPA 2-AC", "AMI Labs", "LeCun", "Predictive Representation Learning", "Self-Supervised Learning", "World Models", "Embodied AI"]
description: "From the 2022 theoretical blueprint to I-JEPA in 2023, V-JEPA in 2024, V-JEPA 2 and V-JEPA 2-AC in 2025, and AMI Labs' $1.03B funding in 2026 — LeCun's JEPA path took four years to go from 'predicting representations beats predicting pixels' to 'action-conditioned robot manipulation.' This article breaks down the core ideas, key experiments, and open questions across the entire JEPA series."
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

But this strategy leaves a core question that runs through this entire article: **the "unpredictable pixel details" that JEPA actively filters out — could they be precisely the information that robot control needs?** I'll return to this in section 9's open questions.

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

I-JEPA's masking strategy shares a similar "mask-and-predict" structure with MAE, but the key difference lies not in "where to mask" but in **what the prediction target is**. Specifically: I-JEPA uses a multi-block strategy, selecting 4 target regions (each 15%-20% of the image) and 1 context region (85%-100%), removing overlaps. The Context Encoder only receives patches from the context region, the Target Encoder encodes patches from the target region, and the Predictor maps context representations to target representations. The final optimization uses representation-space prediction loss, not pixel reconstruction loss.

This means: the model doesn't need to reconstruct masked content from the pixel level. Instead, it needs to **infer** what the masked regions should look like in abstract representation space from the visible regions' representations.

### Key Results

I-JEPA with ViT-H/14 on ImageNet:

- **Linear probing**: 79.3% top-1 accuracy (vs MAE's 77.2%)
- **448 resolution**: 81.1% top-1
- **1% low-shot**: 73.3% (vs MAE's 59.8%) — this gap is enormous
- **Full fine-tuning**: 87.1% (300 epochs), compared to MAE's 1600 epochs at 87.8%

The last point is particularly noteworthy: at comparable final performance, I-JEPA requires approximately **5.3× fewer fine-tuning epochs**. This suggests strong training efficiency potential for the representation prediction objective, though it cannot be simply equated to 5.3× end-to-end training time savings — different methods may have different batch sizes, data augmentation, GPU utilization, and other factors.

### Why It Matters

I-JEPA's significance isn't about SOTA numbers. It cleanly demonstrates one thing: **self-supervised learning that doesn't generate pixels and only predicts in representation space not only works but shows strong training efficiency in visual representation learning tasks.**

This is the first cornerstone of the JEPA path.

## 3. V-JEPA: From Images to Video (2024)

### Paper Info

*V-JEPA: Latent Video Prediction for Visual Representation Learning*, Meta AI, ICLR 2024. Authors: Quentin Garrido et al.

### Core Extension

If I-JEPA proved "predicting image regions in representation space works," V-JEPA goes further: **can we predict masked spatiotemporal regions in video within representation space, and thereby learn dynamics-relevant visual representations?**

This is much harder than images. Masked regions in images are spatially fixed — the model just needs to understand spatial structure. But spatiotemporal prediction in video involves motion patterns, temporal dependencies, and scene evolution, requiring modeling of far more complex temporal structure than spatial prediction on images.

V-JEPA extends the I-JEPA framework to the spatiotemporal dimension. The Context Encoder receives spatiotemporal patches from historical video frames, and the Predictor predicts abstract representations of the masked spatiotemporal regions in representation space.

### Technical Highlights

V-JEPA maintains JEPA's core design philosophy:

- Target Encoder still updated via EMA
- Prediction still happens in representation space, never returning to pixel space
- Masking strategy extends from spatial to spatiotemporal — masking parts of spatiotemporal regions

### Key Contribution

V-JEPA's core contribution is proving the JEPA framework can naturally extend to video, and the learned representations perform well on downstream tasks like action recognition and video understanding. Under frozen representation evaluation, V-JEPA achieved strong results: 82.1% on Kinetics-400, 71.2% on Something-Something-v2, and 77.9% on ImageNet.

From the perspective of the JEPA path's evolution, V-JEPA serves as a key transition from image representation learning to video temporal modeling: it demonstrated that latent video prediction can learn powerful spatiotemporal representations. What pushed this path further into "action-conditioned prediction + planning" was V-JEPA 2.

## 4. V-JEPA 2: From Representation Learning to World Model (2025)

### Paper Info

*V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction, and Planning*, Meta AI, June 2025. [Paper](https://arxiv.org/abs/2506.09985).

### This Is the Most Important One

If I-JEPA and V-JEPA are basic research, V-JEPA 2 is where the JEPA path first fairly completely demonstrates that action-conditioned prediction can serve real robot planning.

### Architecture Upgrade

V-JEPA 2's architecture has major upgrades:

**Encoder**: Vision Transformer, available in multiple sizes — ViT-L (~300M parameters), ViT-H (~600M parameters), ViT-g (~**1B parameters**). Video input is split into 2x16x16 tubelets (2 frames × 16 × 16 pixels). Positional encoding uses **3D-RoPE** (3D Rotary Position Embedding), a key design for processing spatiotemporal sequences.

**Action-conditioned variant (V-JEPA 2-AC)**: This is V-JEPA 2's most important addition. A ~**300M parameter predictor Transformer** using **block-causal attention**, receiving action sequences as conditional input to predict future representations.

This means: the model can not only "understand" video but also "imagine" what the world would look like if a certain action were executed. This is a key step toward world model capability.

### Pretraining

The pretraining scale is substantial:

- Uses the **VideoMix22M** data construction scheme, approximately **22 million video/image samples**; the paper describes its data sources as covering over a million hours of internet video
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

V-JEPA 2-AC was post-trained on approximately **62 hours of robot trajectory data** with action-conditioned post-training on top of the pretrained V-JEPA 2, then tested for robot manipulation without task-specific training or environment adaptation. The "zero-shot" here should be understood as **task/environment-level zero-shot generalization**, not as learning robot control from scratch without any robot interaction data.

The specific task results reported in the paper:

| Task | V-JEPA 2-AC | Cosmos | Octo |
|---|---|---|---|
| Reach | 100% | 80% | — |
| Grasp cup | 60% | 0% | — |
| Grasp box | 20% | 20% | — |
| Pick-and-place cup | 80% | 0% | — |
| Pick-and-place box | 50% | 0% | 0% |

Target reaching accuracy is **less than 4 cm**. Under the same RTX 4090, CEM planning setup reported in the paper, Cosmos baseline requires approximately 4 minutes per action, while V-JEPA 2-AC, even using **10× more candidate samples** (800 vs 80), requires only about 16 seconds — approximately **15× faster**.

This comparison is very compelling. V-JEPA 2-AC performs comparably to Cosmos on the reach task (100% vs 80%), but shows clear advantages on object interaction tasks — Cosmos achieves 0% success on cup/box pick-and-place, while V-JEPA 2-AC reaches 80% and 50% respectively.

### Why V-JEPA 2-AC Is the JEPA Path's Turning Point

V-JEPA 2-AC demonstrates for the first time:

1. The JEPA framework can scale to the billion-parameter level
2. Action-conditioned prediction can serve real robot control
3. Predicting in representation space is not just theoretically elegant but also more efficient in practice than pixel-level prediction
4. After post-training on a small amount of robot data, the model can zero-shot generalize across multiple manipulation tasks — no need to retrain for each specific task

This is exactly the blueprint LeCun drew in his 2022 paper: **using predictive representation learning to build AI systems that understand the physical world.** Three years later, V-JEPA 2-AC turned that blueprint into a running system.

### What Does V-JEPA 2-AC Actually Solve?

It's worth breaking down V-JEPA 2-AC's capabilities into three layers:

**Layer 1: representation.** Compress visual observations into task-relevant latent representations — filtering out unpredictable pixel details while retaining abstract information useful for downstream tasks.

**Layer 2: prediction.** Given action sequences, predict future latent states in representation space — not predicting pixels, but predicting "how the task-relevant state of the world will change."

**Layer 3: planning.** Perform latent rollout over multiple candidate actions and select the action with highest expected return — this is what CEM planning does in the paper.

Connecting these three layers:

```
observation
    ↓
representation (encode current state)
    ↓
action-conditioned dynamics (predict action consequences)
    ↓
latent rollout (imagine multiple future trajectories)
    ↓
planning (select best action)
    ↓
action
```

This is actually very close to Dreamer's core loop. But the starting points differ: Dreamer's latent dynamics was designed from the beginning as the core loop of model-based RL; V-JEPA 2-AC started from a large-scale visual representation model and acquired action-conditioned prediction capability through robot trajectory data. One path goes "from control, learn good representations"; the other goes "from representations, acquire control capability."

## 5. AMI Labs: An Industrialization Observation (2026)

Yann LeCun founded [AMI Labs](https://amilabs.xyz/) (Advanced Machine Intelligence Labs) in Paris in late 2025. In March 2026, AMI Labs announced approximately **$1.03 billion** (approximately €890 million) in seed funding at a pre-money valuation of approximately **$3.5 billion** — one of the largest seed rounds in European AI foundation model history. The round was co-led by Cathay Innovation, Greycroft, Hiro Capital, HV Capital, and others, with participation from NVIDIA, Temasek, Samsung and other institutional investors, as well as individual investors including Jeff Bezos and Eric Schmidt.

The core team includes Yann LeCun (Executive Chairman), Alex LeBrun (CEO), Saining Xie (Chief Science Officer), Michael Rabbat (VP of World Models), and Pascale Fung (Chief Research and Innovation Officer).

AMI Labs' stated technical direction is: building world models that understand physical environments, retain long-term information, and execute logical planning — focusing on learning abstract representations from multimodal sensor inputs, filtering out unpredictable details, and predicting outcomes in conceptual space. Notably, AMI Labs explicitly emphasizes **action-conditioned world models**.

Based on public personnel backgrounds and technical vision, it is reasonable to speculate that JEPA-style predictive representation learning will be one of AMI Labs' important research directions; however, this is inference based on publicly available information, not a confirmed architectural conclusion officially announced by AMI.

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

### When Does JEPA Become a World Model?

This question deserves separate discussion because it directly affects how we understand the entire technology path.

**I-JEPA** itself is better described as predictive representation learning, not a complete world model. Its goal is to learn image representations useful for downstream tasks — no temporal dynamics modeling, no concept of action.

**V-JEPA** introduces temporal prediction, but its primary goal remains learning video representations — it predicts future frame representations but doesn't support action-conditioned planning.

**V-JEPA 2-AC** goes further by adding action-conditioned prediction and using model-based planning to predict action consequences on robots. If we adopt the looser definition that "a world model should be able to predict action consequences and support planning," then V-JEPA 2-AC already possesses key world model capabilities; however, it still differs structurally from Dreamer/RSSM's explicit latent state transition model.

Therefore, to be precise, I would call I-JEPA / V-JEPA "JEPA representation learning" and V-JEPA 2-AC a "JEPA-based latent world model" under the looser definition. This distinction matters — it lets us understand more precisely which parts of the JEPA path are representation learning and which parts are world modeling.

## 8. What This Means for Practitioners

### If You're Doing Robot Control

V-JEPA 2-AC's manipulation results deserve serious attention. Under the same RTX 4090 + CEM planning setup reported in the paper, V-JEPA 2-AC's single planning time is approximately 16 seconds, while Cosmos baseline requires approximately 4 minutes. Even with **10× more candidate samples** (800 vs 80), V-JEPA 2-AC still maintains approximately **15× planning latency advantage**. This shows latent prediction has clear computational efficiency advantages, but **16 seconds/action itself is still far from real-time robot control**. If your system can tolerate some planning delay rather than generating beautiful video, the JEPA path may be more suitable.

### If You're Doing Self-Supervised Learning

I-JEPA's training efficiency performance (5.3× fewer epochs achieving comparable performance to MAE) is already attention-worthy. The JEPA framework provides a self-supervised learning paradigm that doesn't rely on pixel reconstruction, which is particularly valuable as compute costs receive increasing scrutiny.

### If You're Doing World Model Research

The JEPA path provides an important alternative: not all world models need to generate pixels. If your downstream tasks require understanding rather than generation, predicting in representation space may be the better choice.

### If You're Following AMI Labs

AMI Labs' team and funding scale tell us one thing: investors have strong conviction about the JEPA path's industrial value. But stay clear-eyed — the distance from papers to products remains vast. AMI Labs hasn't yet released public products or benchmark results beyond V-JEPA 2's scope.

## 9. Open Questions for the JEPA Path

Finally, a few questions the JEPA path hasn't fully answered yet:

**Representation sufficiency.** This may be one of the most critical questions for JEPA's transition from representation learning to a true world model. JEPA's greatest potential strength is also its greatest risk: the model actively filters out unpredictable pixel details. But for robot control, some seemingly low-level details (precise geometry, contact states, friction, fine object states, affordances) may be precisely what determines whether an action succeeds. A representation that works very well for video classification is not necessarily a sufficiently complete state representation for control. Therefore, the real question to answer is not "are representations better than pixels" but: **does this representation retain all the information needed for downstream planning?**

**Representation degeneration and objective design.** JEPA avoids representation degeneration through target encoder, EMA updates, and prediction objective design, but "why this mechanism can stably learn informative representations" remains an open research question. Especially as the predictor, target encoder, and masking strategy extend further into long temporal sequences and action-conditioned prediction, whether representations remain sufficient and stable is a more important question than simply avoiding collapse.

**Long-range prediction stability.** V-JEPA 2-AC demonstrated action-conditioned prediction and planning capability, but whether its long-range rollout can stably maintain task-relevant information requires more experimental validation. In comparison, Dreamer and other model-based RL methods place multi-step latent rollout and policy/value learning at the core of their training loop from the beginning, so the two approaches differ fundamentally in their technical paths for long-horizon planning.

**Language alignment.** A core capability of current VLA (Vision-Language-Action) models is language grounding. The JEPA path currently works primarily in visual and action space — how to integrate with language capabilities is an important open question.

**Scalability.** V-JEPA 2 has scaled to the billion-parameter level, but whether JEPA-style models have a stable, predictable scaling law still lacks sufficient evidence. The more critical question is not simply pursuing parameter scale, but whether representation quality, action-conditioned prediction accuracy, and downstream planning performance continue to improve as model scale, video data volume, and prediction horizon increase.

---

The JEPA path's core insight — that in tasks targeting visual representation learning, prediction, and planning, latent representation prediction can avoid the computational burden of pixel-level generation while achieving strong downstream performance — has been progressively validated through experiments from I-JEPA to V-JEPA 2-AC. From academic hypothesis to AMI Labs' industrialization attempt, this path is moving toward broader applications.

But it's not the only answer for world models. As I concluded in my [roundup article](/en/articles/2026-09-01-world-model-h2-review/), "world model" is losing its singular meaning. Different questions require different tools.

What truly makes JEPA worth attention is not that it proposes yet another "world model architecture," but that it redefines what world models should predict. The pixel generation path asks "what will the future look like?" JEPA asks "which state changes in the future are worth predicting?" And V-JEPA 2-AC takes a further step: "after executing this action, how will the task-relevant state of the world change?" This is the true turning point of JEPA's journey from representation learning to world modeling.

*Next up, I'll discuss team configuration for starting a company in embodied AI — not a business plan, but a pragmatic analysis from an engineer's perspective.*
