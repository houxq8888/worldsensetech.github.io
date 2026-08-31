---
title: "World Models 2026 H2 Review: Papers and Projects Worth Watching"
slug: "2026-09-01-world-model-h2-review"
date: 2026-09-01
draft: false
categories: ["World Models"]
tags: ["World Models", "2026 Review", "NVIDIA Cosmos", "Genie 3", "AMI Labs", "Embodied AI", "Robot AI", "Paper Recommendations"]
description: "From NVIDIA Cosmos to Google Genie 3, from LeCun's AMI Labs to Fei-Fei Li's World Labs — a curated review of the most important world model papers, projects, and trends in 2026 H2."
toc: true
related_articles:
  - world-model-2026-trend
  - vla-vs-world-model
  - world-model-intro
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
  - world-model-transformer
---

The second half of 2026 has seen a noticeable acceleration in the world models space.

Over the past few months, I've written a lot of foundational content on this blog — from the mathematics of RSSM to hands-on DreamerV3 training, from the VLA vs world models comparison to Sim-to-Real engineering pitfalls. A reader recently asked: "If I want to keep up with the latest in world models, what should I be reading?"

In this article, I'll walk through the most noteworthy papers and projects from the past few months. This isn't an exhaustive survey list — it's a curated selection from the perspective of an engineer working in this field, filtered by "how useful is this for my actual work?"

## 1. Industrial-Grade World Foundation Models: From Papers to Platforms

### NVIDIA Cosmos: A Data Engine for Physical AI

NVIDIA officially launched the [Cosmos World Foundation Model Platform](https://www.nvidia.com/en-us/ai/cosmos/) at CES this year, and by January 2026 it had already surpassed 2 million downloads.

This isn't an academic demo — it's an industrial-grade platform. Its core capability: using physics-aware video generation models to produce large-scale synthetic training data for autonomous driving and robotics.

Why does this matter? I analyzed this in detail in my [earlier article on synthetic data](/en/articles/world-model-synthetic-data-for-vla/) — real robot data collection is expensive and narrow in coverage, which is the core bottleneck for world model deployment. Cosmos's approach is: since world models have already learned physical laws, let them "manufacture data" — generating videos of diverse scenarios, lighting conditions, and object interactions to train downstream perception and control models.

For practitioners: If you're working on robotics or autonomous driving, Cosmos's open-source models and toolchain deserve a serious look. It could become the "Llama" of physical AI — a foundational base that everyone can build on.

### Google DeepMind Genie 3: Real-Time Interactive World Models

The Genie series represents Google DeepMind's important investment in world models. [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/), released in August 2025, is positioned as "the first real-time interactive general-purpose world model."

What can it do? Given an image or text description, it renders navigable 3D spaces at 24fps using self-taught physics — no hard-coded rules. The model learns on its own how objects move and how collisions occur.

How it differs from predecessors: Genie 1 proved the feasibility of learning interactive environments from video, Genie 2 scaled to a large foundation model, and Genie 3 pushes real-time performance and generality to new heights. 24fps means it's approaching the "usable" threshold — while it can't match game engine precision yet, it's already sufficient as a training environment and prototyping tool.

### World Labs Marble: Commercializing Spatial Intelligence

Fei-Fei Li's [World Labs](https://www.worldlabs.ai/) released Marble in November 2025, generating persistent, downloadable 3D spaces from various media inputs (images, video, text).

This is an important signal of world models moving toward commercialization. Marble's approach differs from Cosmos — Cosmos targets data generation and training, while Marble targets content creation and spatial computing. But at the technical foundation, they're solving the same problem: teaching AI to understand the 3D physical world.

## 2. A Major New Player: AMI Labs

### LeCun's €500 Million Bet

In January 2026, Yann LeCun founded [AMI Labs](https://amigroup.ai/) (Advanced Machine Intelligence Labs) in Paris, backed by €500 million in funding. This is by far the largest industrial investment in the world models direction to date.

AMI Labs's technical approach is based on the [JEPA (Joint Embedding Predictive Architecture)](https://openreview.net/pdf?id=BZ5a1r-kVsf) that LeCun has championed for years. Simply put, JEPA's core idea is: don't make predictions in pixel space (that's the generative model approach) — make predictions in abstract representation space.

This shares common ground with the Dreamer series — as I discussed in the [RSSM deep dive](/en/articles/rssm-deep-dive/), RSSM also makes dynamic predictions in latent space rather than directly predicting the next image frame. But JEPA goes further: it argues that even the objective of "reconstructing pixels" is wrong — the model should only focus on "abstract features useful for the task."

Why pay attention? Because LeCun isn't just any researcher. He's one of the founding figures of deep learning, and his judgment on technical directions carries significant weight. His decision to go all-in on world models, backed by €500M, signals that he believes the field has reached an industrialization tipping point.

Of course, whether €500M produces results depends on execution. LeCun's academic ability is unquestionable, but building a company and doing research are very different things.

## 3. Key Survey Papers: Building a Global Perspective

If you only want to read one paper to build a comprehensive understanding of world models, I recommend these 2026 surveys:

### "A Definition and Roadmap for World Models"

This [arxiv paper (2607.06401)](https://arxiv.org/html/2607.06401v1) does something very valuable: it provides a formal definition of world models and draws a technology roadmap.

The concept of "world model" is being used too loosely now — video generation models call themselves world models, game engines call themselves world models, even simple prediction models claim the title. This paper attempts to clarify: what qualifies as a world model, what doesn't, and what's still missing between the current state and true world models.

### "World Model for Robot Learning: A Comprehensive Survey"

This [paper on Hugging Face](https://huggingface.co/papers/2605.00080) focuses specifically on world models in robot learning. If your question is "how do I use world models on real robots?", this is more targeted than the one above.

It systematically covers how world models apply to perception, planning, and control, and how different robot form factors (robotic arms, mobile robots, humanoids) have different requirements for world models.

### "A Comprehensive Survey on World Models for Embodied AI"

This [survey with a maintained paper list on GitHub](https://github.com/Li-Zn-H/AwesomeWorldModels) analyzes world models from the embodied AI perspective. Its strength is placing world models within the larger embodied AI framework, discussing the relationships between world models, VLAs, reinforcement learning, and simulators.

## 4. Technology Trends: Three Directions to Watch

### Trend 1: World Models + Transformer Scaling

I discussed this topic in [a previous article](/en/articles/world-model-transformer/). The 2026 trend is clear: more and more world models are adopting Transformer architectures instead of traditional RNN/GRU.

Cosmos uses a Transformer-based diffusion model, Genie 3 is also Transformer-based underneath, and even Dreamer's successors are exploring Transformer replacements for the GRU components in RSSM.

The reason is straightforward: Transformer's scaling capability far exceeds RNNs. When world models need to move from "lab-level simple tasks" to "real-world complex scenarios," both model parameters and training data need substantial increases — and Transformer is currently the only architecture proven to scale effectively.

### Trend 2: Accelerating VLA + World Model Convergence

I predicted in my [VLA vs World Models article](/en/articles/vla-vs-world-model/) that 2026-2027 would see increasing hybrid architectures. The trend is now accelerating.

Google's Gemini Robotics On-Device is a prime example — using VLA for perception and language understanding, with world models for planning and prediction. NVIDIA's GR00T follows a similar path.

This convergence isn't simply "two modules bolted together" — it's deep architectural integration. For instance, the world model's latent state representations can directly serve as conditional inputs for the VLA, while the VLA's language grounding ability can guide the world model's imagination direction.

### Trend 3: From "Can It Work?" to "How to Make It Work Well"

In 2024-2025, the primary question for world models was "can it work?" — can it learn environment dynamics, can it generate useful data through imagination?

In 2026, the question has become "how to make it work well" — how to improve sample efficiency, how to reduce training costs, how to deploy stably on real robots. This is why industrial platforms like Cosmos are emerging, and why DreamerV3 training engineering practices (which I covered in detail in [this article](/en/articles/2026-08-28-dreamerv3-training-tips/)) have become as important as the algorithms themselves.

## 5. My Reading Recommendations

Finally, a practical reading list. Based on your background and goals, I suggest the following priority:

**If you're new to world models:**

Start with any one of the three surveys mentioned above to build a global perspective. Then work through this blog's foundational series — from [What is a World Model](/en/articles/world-model-intro/) to the [RSSM Deep Dive](/en/articles/rssm-deep-dive/) to the [Dreamer Explained](/en/articles/2026-08-25-dreamer-explained/) — to solidify core concepts.

**If you're already doing world model research:**

Focus on the Cosmos technical report and Genie 3 paper to understand how industry approaches large-scale world models. Then follow AMI Labs's progress — if LeCun's JEPA approach succeeds, it could reshape the entire field's technical paradigm.

**If you care about engineering deployment:**

Cosmos's open-source toolchain is priority one. Then look at the Sim-to-Real chapters in the World Model for Robot Learning survey. Finally, follow NVIDIA and Google's latest sharing on robot deployment.

The world models direction is at a critical transition from "academic exploration" to "industrial deployment" in 2026. Each paper and project above represents a facet of this trend. You don't need to read them all — but understanding the landscape, then going deep in your area of interest, is essential.

---

*In the next article, I'll discuss what kind of team you need to start a company in embodied AI — not a generic business plan, but a pragmatic analysis from an engineer's perspective. Stay tuned.*
