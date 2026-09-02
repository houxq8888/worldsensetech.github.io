---
title: "Embodied AI in 2026: A Technical Landscape of Who's Doing What and How"
slug: "2026-09-06-embodied-ai-landscape"
date: 2026-09-06
draft: false
categories: ["Embodied Intelligence", "Industry Observation"]
tags: ["Embodied Intelligence", "Humanoid Robot", "VLA", "World Model", "Sim-to-Real", "Physical Intelligence", "Gemini Robotics", "GR00T"]
description: "In 2026, the embodied AI track enters an acceleration phase. From Physical Intelligence's π₀.7 to Google DeepMind's Gemini Robotics, from NVIDIA's GR00T platform to the collective sprint of Chinese humanoid robot companies — this article surveys the technical approaches, progress stages, and core differentiators of the major players, attempting to answer a fundamental question: where exactly is this field right now?"
toc: true
related_articles:
  - 2026-09-03-vla-deep-dive
  - 2026-09-01-world-model-h2-review
  - 2026-09-02-jepa-deep-dive
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
---

Over the past six months, the pace of the embodied AI field has clearly accelerated.

Physical Intelligence released π₀.7, Google DeepMind launched the Gemini Robotics series, NVIDIA's GR00T platform moved from concept to open source, and Figure AI's humanoid robot completed over 11 months of deployment testing at the BMW factory. On the Chinese side, companies like Unitree, Agibot, and Galaxy General have seen financing reach record highs, with humanoid robots beginning to move from lab prototypes toward small-batch production.

This article attempts a foundational survey: **what the major players are doing, what technical approaches they are using, and what stage they have reached.** No predictions, no rankings — just drawing the technology map clearly.

## Three Schools of Technical Approach

Based on underlying technical choices, the major players in embodied AI can be roughly divided into three schools.

### School One: VLA End-to-End Policy

**Core idea:** Use a Vision-Language-Action (VLA) model to map directly from perception to action, without explicitly building a world model.

**Representatives:** Physical Intelligence (π₀ series), Google DeepMind (RT-2 → Gemini Robotics)

Physical Intelligence's π₀ series is currently the most representative work in the VLA approach. From π₀'s flow matching continuous action generation, to π₀.5's discrete-continuous hybrid recipe, to π₀.7's context-rich steering + visual sub-goals — this technical line has been continuously iterating on action interface design. π₀.7's cross-embodiment zero-shot T-shirt folding is a landmark result, but it should be noted that this is still a result under a specific evaluation protocol, and there is still considerable distance to general household manipulation.

Google DeepMind's approach evolved from RT-2 (a 55B VLM fine-tuned as a robot policy) to Gemini Robotics. The core change in the Gemini Robotics series is: instead of training a robot-specific model from scratch, they directly adapt the Gemini multimodal foundation model for robotics. This is a "general to specific" approach, in the same lineage as RT-2's "from VLM to robot policy" but at a larger scale.

**The advantage of this approach** is strong semantic generalization — the semantic knowledge VLA inherits from internet pre-training enables it to handle novel objects and novel instructions. **The limitation** is: a typical VLA does not have an explicit action-conditioned prediction interface (this issue was discussed in [a previous article](/en/articles/2026-09-07-vla-world-models/)), resulting in limited physical prediction capability.

### School Two: World Model + Imagination Training

**Core idea:** First learn a world model that can predict the future, then train the policy in "imagination."

**Representatives:** DreamerV3 (RSSM + imagination), TD-MPC2 (latent dynamics + MPC)

DreamerV3 learns a dynamics model in latent space through RSSM, then rolls out trajectories in imagination to train an actor-critic policy. This approach has a clear advantage in sample efficiency — complex skills can be learned from a small amount of real interaction data.

TD-MPC2 takes a more concise approach: encoder + MLP ensemble dynamics + latent-space MPC. Without the dual-track structure of RSSM, it demonstrates potential in cross-task scaling — training a single world model across 139 tasks.

**The advantage of this approach** is sample efficiency and physical prediction capability. **The limitation** is: world model prediction quality degrades with rollout length, and reliability for long-horizon tasks remains a bottleneck. Additionally, the world model approach typically lags behind the VLA approach in semantic understanding (language grounding).

### School Three: Platform + Simulation + Foundation Models

**Core idea:** Instead of directly building end-to-end robot policies, provide the infrastructure — simulation platforms, foundation models, data toolchains.

**Representatives:** NVIDIA (Isaac + GR00T + Cosmos), World Labs (3D scene generation)

NVIDIA's strategy is to be the "infrastructure layer" for embodied AI. Isaac Sim provides the physics simulation environment, GR00T N1/N2 provides open-source humanoid robot foundation models, and Cosmos provides a world foundation model for generating synthetic training data. NVIDIA does not build its own robot products; instead, it enables ecosystem partners to build on its platform.

World Labs focuses on 3D scene generation — producing 3D environments suitable for simulation training from images or text. This addresses a critical link in the sim-to-real chain: the diversity and realism of simulation environments.

**The advantage of this approach** is the leverage effect — one platform can serve multiple robot companies. **The limitation** is: the platform's value depends on ecosystem maturity, and the embodied AI ecosystem is still in its early stages.

## Humanoid Robots: The Bet on Hardware Form Factor

One of the most prominent trends in 2026 is the collective sprint in humanoid robots.

**Figure AI's** Figure 02/03 has completed over 11 months of deployment testing at the BMW Spartanburg factory, reportedly participating in the production process of over 30,000 vehicles. Figure's technical approach combines end-to-end learning (in partnership with OpenAI) and traditional control.

**Tesla Optimus** continues to iterate on hardware design, targeting internal factory deployment. Tesla's advantage lies in its vertical integration capability — its own factories provide testing environments, its own chips (Dojo/FSD) provide training compute, and its own AI team provides algorithms.

**1X Technologies'** NEO series targets general-purpose service scenarios, with a technical approach leaning toward end-to-end learning.

**Chinese companies** are investing particularly intensively in this direction. Unitree pivoted from quadruped robots to humanoid, with fast hardware iteration and strong cost control. Agibot and Galaxy General have repeatedly broken records in financing — Galaxy General's valuation has reportedly reached the $3 billion level.

Humanoid robots are a high-risk, high-reward bet. The advantage is: humanoids can adapt to environments designed for humans (stairs, door handles, tools). The risk is: the engineering complexity of humanoids is far greater than that of specialized-form-factor robots, and most current "humanoid robot demos" are still using teleoperation or simple policies to complete relatively simple tasks.

## Chinese Embodied AI: Fast Catching Up and Differentiation

The Chinese embodied AI track has shown several characteristics over the past six months.

**Explosive financing scale.** Multiple companies have completed financing in the hundreds of millions of RMB range, with leading companies like Galaxy General and Unitree entering unicorn territory. Capital is shifting from "investing in concepts" to "investing in deployment."

**Outstanding hardware capability.** China's supply chain advantage in robot hardware (motors, reducers, sensors) is translating into whole-machine advantage. Unitree's cost control capability is competitive on a global scale.

**Still catching up in software/algorithms.** In areas like VLA foundation models, world models, and large-scale robot data, Chinese companies still have a gap compared to Physical Intelligence and Google DeepMind. But this gap is narrowing — partly because open papers and open-source code have lowered the technical barrier.

**Application scenario differentiation.** Compared to American companies leaning toward general-purpose humanoids, Chinese companies are focusing more on specific scenarios — warehouse logistics, industrial assembly, commercial services. This is a more pragmatic strategy, but it also means that accumulation of generalizability may be slower.

## Several Technical Trends Worth Watching

Setting aside specific companies, there are several technical trends that deserve continued attention.

### Trend One: VLA and World Models Are Converging

From π₀.7 introducing visual sub-goals and Gemini Robotics introducing multimodal reasoning, the VLA approach is gaining more and more "predictive" capability. Conversely, the world model approach is also gaining language and semantic capabilities. The two approaches are converging from both ends toward the middle — but as of now, no single system simultaneously possesses mature language grounding, action-conditioned prediction, and high-frequency continuous control.

### Trend Two: Simulation Becomes Standard

Nearly all major players are using simulation at scale for training or data augmentation. The use of simulation platforms like NVIDIA Isaac Sim, MuJoCo, and Isaac Lab has become an industry standard. The sim-to-real gap is narrowing, but has not yet been eliminated — particularly in fine manipulation and contact-rich tasks.

### Trend Three: Data Is Becoming the Bottleneck

Differentiation in model architecture is shrinking; differentiation in data and training recipes is growing. Whoever has more, more diverse, and higher-quality robot interaction data has the advantage. This explains why teleoperation data collection, synthetic data generation, data quality filtering, and other "data engineering" directions are receiving more attention.

### Trend Four: The Gap from Demo to Deployment

Most publicly demonstrated robot capabilities remain "demo-level" — completing specific tasks in controlled environments. There is a massive engineering gap between demo and reliable deployment (handling failure recovery, adapting to environmental changes, running stably over long periods). Figure AI's 11-month deployment test at BMW is one of the closest cases to "real deployment" in publicly available information.

## What Does This Map Mean?

If we summarize the current technology map of embodied AI:

```
                    Embodied AI 2026
                          │
           ┌──────────────┼──────────────┐
           │              │              │
       VLA Policy    World Model     Platform
    (π₀, Gemini)   (Dreamer,      (NVIDIA,
                    TD-MPC)        World Labs)
           │              │              │
           └──────────────┼──────────────┘
                          │
                    Humanoid Robots
                 (Figure, Tesla, 1X,
                  Unitree, Agibot, ...)
                          │
                    Real Deployment
                  (still early stage)
```

Several assessments:

**First, technical approaches have not yet converged.** The three approaches — VLA, world models, and platformization — each have their advantages and each have unsolved problems. The final system will likely be a hybrid architecture, but a clear convergence direction is not yet visible.

**Second, hardware and software are decoupling.** Humanoid robot hardware is becoming a "platform commodity," with differentiation increasingly concentrated in software (foundation models, data, training recipes). This mirrors the evolution path of the electric vehicle industry.

**Third, the gap from demo to deployment is the biggest current challenge.** Most public results remain "it works under specific conditions" rather than "it runs reliably in real environments." The key to solving this problem may not be larger models, but better data, more robust policies, and a more mature sim-to-real pipeline.

**Fourth, China has advantages in hardware and deployment, and is still catching up in foundation models.** This landscape may change in the next 1-2 years — the open-source ecosystem and returning talent are narrowing the gap.

---

*This is a snapshot at a point in time. The embodied AI field is changing rapidly; this map may need to be redrawn in six months.*

*Next up is the [follow-up to the VLA series](/en/articles/2026-09-07-vla-world-models/) — an analysis of the relationship between VLA and world models, open questions, and three assessments.*
