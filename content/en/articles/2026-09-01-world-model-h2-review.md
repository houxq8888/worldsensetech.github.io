---
title: "World Models 2026 Mid-Year Review: From Cosmos, Genie to JEPA — The Divergence of Routes"
slug: "2026-09-01-world-model-h2-review"
date: 2026-09-01
draft: false
categories: ["World Models"]
tags: ["World Models", "2026 Review", "NVIDIA Cosmos", "Genie 3", "AMI Labs", "JEPA", "Embodied AI", "Robot AI", "Paper Recommendations"]
description: "As of late August 2026, the world model field is undergoing a deep divergence. From NVIDIA Cosmos to Google Genie 3, from LeCun's AMI Labs to Fei-Fei Li's World Labs, different technology paths are heading toward different applications. This article attempts to clarify the relationships between these projects and papers."
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

In this article, I'll walk through the most noteworthy papers and projects as of late August 2026. "Noteworthy" here doesn't mean all of these projects launched in the second half of the year — some (like Cosmos and Genie) were first released in 2025 but continued to develop and exert significant influence in 2026. What matters more to me is: as of now, they still represent technology paths worth tracking over the next six months.

This isn't an exhaustive survey list — it's a curated selection from the perspective of an engineer working in this field, filtered by "how useful is this for my actual work?"

But before diving into specific projects, there's something important that needs to be clarified first.

## 1. Let's Distinguish Between Types of "World Models"

The term "world model" is being used too loosely now — video generation models call themselves world models, game engines call themselves world models, even simple prediction models claim the title. But the projects discussed below, while all called "world models," are not the same thing.

After reading [A Definition and Roadmap for World Models (arxiv 2607.06401)](https://arxiv.org/html/2607.06401v1), and combining the paper's classification framework with my own engineering experience, I categorize current world models into four types:

**A. Latent Dynamics World Model**

Examples: Dreamer / RSSM

Core logic: state → action → next state, learning environment dynamics in latent space.

Goal: supporting planning, RL, control. This is the type I've written about most on this blog.

**B. Generative Video World Model**

Examples: NVIDIA Cosmos, etc.

Core logic: condition + history → future observations, generating future video frames.

Goals lean toward: data generation, simulation, prediction, perception.

**C. Interactive World Model**

Examples: Google Genie series

Core logic: state + action → interactive future, generating interactive futures based on actions.

Key capabilities: action-conditioned generation, temporal consistency, controllability.

**D. Spatial / 3D World Model**

Examples: World Labs Marble, etc.

Focus: persistent scene, geometry, spatial consistency, navigability, 3D representation.

With this taxonomy in place, we won't conflate different things when discussing specific projects below.

The biggest change in "world models" in 2026 is not the emergence of a unified World Model, but rather **the divergence of different world-model paradigms** — each heading toward different application scenarios and evaluation criteria.

Let's look at each category in turn.

## 2. Generative Video World Model: NVIDIA Cosmos

### More Than Just "Video Generation"

NVIDIA first unveiled the [Cosmos World Foundation Model Platform](https://www.nvidia.com/en-us/ai/cosmos/) at CES 2025, and by early 2026 it had surpassed 2 million downloads.

An important distinction: Cosmos is not simply a "video generation model." It's a development platform for Physical AI, covering video generation, world state understanding, data processing, and synthetic data generation. Reducing it to "video generation" underestimates its technical ambition.

Why does this matter? I analyzed this in detail in my [earlier article on synthetic data](/en/articles/world-model-synthetic-data-for-vla/) — real robot data collection is expensive and narrow in coverage, which is the core bottleneck for world model deployment. Cosmos's approach is: use physics-aware generative models to produce large-scale synthetic training data for autonomous driving and robotics.

For practitioners: If you're working on robotics or autonomous driving, Cosmos's open-source models and toolchain deserve a serious look. If its models, data tools, and deployment ecosystem continue to mature, it has the potential to become important open-source infrastructure for Physical AI.

## 3. Interactive World Model: Google DeepMind Genie 3

### Real-Time Generation ≠ Controllability

The Genie series represents Google DeepMind's important investment in world models. [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/), released in August 2025, is positioned as "the first real-time interactive general-purpose world model."

What can it do? Given an image or text description, it generates an interactive world simulation at 24fps — continuously evolving visually coherent scenes based on user action inputs. This is not pre-rendered video, nor is it a traditional exportable 3D world with complete geometry and physics engines. It's closer to: generating an interactive, action-conditional, continuously evolving world simulation from visual/text conditions.

How it differs from predecessors: Genie 1 proved the feasibility of learning interactive environments from video, Genie 2 scaled to a large foundation model, and Genie 3 pushes real-time performance and generality to new heights.

But there's an important technical issue worth highlighting here: **real-time generation ≠ controllability for control**.

24fps frame rate is just one of many metrics. For robotic world models, what matters more includes: temporal consistency, action controllability, long-horizon stability, spatial consistency, object permanence, and more. A high frame rate doesn't mean it's ready to serve as a robot training environment.

This distinction is critical and represents one of the core challenges facing interactive world models today.

## 4. Spatial / 3D World Model: World Labs Marble

### A Commercial Path for Spatial Intelligence

Fei-Fei Li's [World Labs](https://www.worldlabs.ai/) released Marble in November 2025, generating persistent, downloadable 3D spaces from various media inputs (images, video, text).

Marble and Cosmos both involve world modeling, but their product goals and technical focuses differ significantly. Cosmos leans more toward Physical AI / simulation / synthetic data / robotics, while Marble leans more toward 3D world generation / spatial intelligence / world reconstruction / content creation. Both are advancing toward "generable, interactive, spatially consistent world representations," but in different directions.

This is an important signal of world models moving toward commercialization — the spatial intelligence path is closer to content creation and AR/VR applications. But it's worth noting that World Labs has also been expanding into robotics simulation in 2026, with its Marble technology being used to generate 3D scene data for robot training. This means the boundary between Spatial/3D world models and Physical AI is blurring.

## 5. Predictive Representation Learning: AMI Labs and JEPA

### LeCun's Technology Bet

Yann LeCun founded [AMI Labs](https://amigroup.ai/) (Advanced Machine Intelligence Labs) in Paris, completing approximately $1.03 billion (approximately €890 million) in funding in March 2026, making it one of the most watched new companies in European AI foundation models in recent years.

AMI Labs's technical approach is based on the [JEPA (Joint Embedding Predictive Architecture)](https://openreview.net/pdf?id=BZ5a1r-kVsf) that LeCun has championed for years. JEPA's core idea is: **predict representations rather than raw observations** — make predictions in abstract representation space, not pixel space.

This shares some common ground with the Dreamer series — as I discussed in the [RSSM deep dive](/en/articles/rssm-deep-dive/), RSSM also makes dynamic predictions in latent space rather than directly predicting the next image frame. But the two are not at the same level: RSSM is a latent dynamics model aimed at supporting planning and control; JEPA is a predictive representation architecture aimed at learning world state representations useful for tasks.

JEPA's core argument is not that "pixel reconstruction is wrong." More precisely: **for learning high-level semantics and world state representations, requiring the model to precisely predict all pixels is not an ideal learning objective** — because pixel space contains a large amount of task-irrelevant details and randomness.

Why pay attention? LeCun is one of the founding figures of deep learning, and his judgment on technical directions carries significant weight. His decision to go all-in on a JEPA-based world model approach at least indicates that investors and the founding team have a very strong conviction about the future industrial value of this technology path. Of course, whether it produces results depends on execution.

## 6. Key Survey Papers: Building a Global Perspective

If you want to read a few papers to build a comprehensive understanding of world models, I recommend these:

### "A Definition and Roadmap for World Models"

This [arxiv paper (2607.06401)](https://arxiv.org/html/2607.06401v1) does something very valuable: it provides a formal definition of world models and draws a technology roadmap. The four-category taxonomy above draws primarily from this paper.

### "World Model for Robot Learning: A Comprehensive Survey"

This [paper](https://huggingface.co/papers/2605.00080) focuses specifically on world models in robot learning. If your question is "how do I use world models on real robots?", this is more targeted. It systematically covers how world models apply to perception, planning, and control.

### "A Comprehensive Survey on World Models for Embodied AI"

This [paper](https://arxiv.org/abs/2510.16732) analyzes world models from the embodied AI perspective. [A maintained paper list is available on GitHub](https://github.com/Li-Zn-H/AwesomeWorldModels) for extended reading.

## 7. A Comparison Table

Placing the projects discussed above side by side:

| Project | Core Paradigm | What It Predicts | How to Verify Usefulness | Current Maturity | Primary Applications |
|---|---|---|---|---|---|
| Cosmos | Generative World Model | Future video frames / world states | Synthetic data improvement on downstream perception/control | Open-source available | Autonomous driving / robot training data |
| Genie 3 | Interactive World Model | Action-conditioned future visuals | Interaction consistency and long-horizon stability | Research preview | Simulation / prototype validation |
| Marble | 3D World Generation | Persistent spatial geometry representations | 3D reconstruction accuracy and spatial consistency | Commercial product | Spatial intelligence / robot simulation |
| Dreamer | Latent Dynamics | Next state in latent space | RL task scores and sample efficiency | Academically mature | Robot control / RL |
| JEPA series | Predictive Representation | Abstract representations (non-pixel) | Downstream task representation quality | Early research | Representation learning / world understanding |

This table is more valuable than paragraphs of adjectives. When you encounter a new "world model," place it in this framework first to quickly understand its relationship to other work.

## 8. The Biggest Gap in World Models: Evaluation

After discussing all these projects and technologies, there's a question we can't avoid: **how do we prove that a world model is "good"?**

This is the weakest link in the current world model field, and the dimension I pay closest attention to when reading papers. Multiple surveys from 2025-2026 have listed benchmarks, metrics, physical consistency, computational efficiency, and long-horizon consistency as core open problems.

I believe world model evaluation can be organized as a ladder:

```
Generation quality
       ↓
Temporal consistency
       ↓
Physical consistency
       ↓
Action controllability
       ↓
Counterfactual accuracy
       ↓
Long-horizon stability
       ↓
Downstream task improvement
```

The further down you go, the closer you get to a truly useful world model.

Most current work remains at the upper layers — video generation models are impressive at generation quality, but can they achieve physical consistency? Can they be precisely controlled by actions? Do they actually help downstream tasks? These questions often go unanswered.

This is why I've repeatedly emphasized "real-time generation ≠ controllability" and "you can't call something VLA + world model just because it has predictive ability." **Evaluation criteria are shifting from "does it look realistic?" to "are predictions accurate, can it be controlled by actions, and is it actually useful for downstream tasks"** — yet most projects are still proving themselves with upper-layer metrics.

What truly deserves attention is work that reaches the bottom of this ladder and demonstrates world model value through downstream task improvement.

## 9. On VLA + World Model Convergence

I discussed in my [VLA vs World Models article](/en/articles/vla-vs-world-model/) that VLA and world models are not competing approaches but complementary ones. In 2026, we are indeed seeing more work combining the two, but this requires particular caution.

You can't simply label a robot system as "VLA + world model" just because it has prediction or planning capabilities. You need to specify: which module is the world model? Is it an explicit dynamics model, or latent prediction within the policy? Is it training-time simulation, or inference-time planning?

If this convergence is truly achieved — with world model latent state representations directly serving as conditional inputs for the VLA, and the VLA's language grounding ability guiding the world model's imagination direction — it would be a very powerful architecture. But most current work is still in the exploration phase and requires more specific technical validation.

## 10. Technology Trends: Three Important Shifts

### Trend 1: Transformers Entering World Models

I discussed this topic in [a previous article](/en/articles/world-model-transformer/). The 2026 trend is clear: Transformers have demonstrated clear scaling advantages in large-scale sequence modeling and generative world models, and are gradually entering positions previously held by traditional RSSM/GRU world models.

However, it's important to note that world models contain multiple distinct modules — observation encoder, latent dynamics, action-conditioned transition, video generation, planner, policy, value model — and Transformers play completely different roles in each. We can't simply say "world models are becoming Transformers."

### Trend 2: Evaluation Criteria Are Changing

In 2024-2025, the primary question for world models was "can it work?" — can it learn environment dynamics, can it generate useful data through imagination?

In 2026, the question has become "how to make it work well" — how to improve sample efficiency, how to reduce training costs, how to deploy stably on real robots.

In other words, **the evaluation criteria for world models are shifting from "does it generate realistically?" to "are predictions accurate, can it be controlled by actions, and is it actually useful for downstream tasks?"** The real watershed has moved from generation quality to controllability, predictive accuracy, and downstream utility.

This is why industrial platforms like Cosmos are emerging, and why DreamerV3 training engineering practices (which I covered in detail in [this article](/en/articles/2026-08-28-dreamerv3-training-tips/)) have become as important as the algorithms themselves.

### Trend 3: Different Paradigms Heading Toward Different Applications

As the taxonomy in Section 1 shows, world models in 2026 are not a single unified path but multiple paradigms each finding their own application scenarios: Latent Dynamics heading toward RL and control, Generative Video toward synthetic data, Interactive World toward simulation environments, and Spatial/3D toward spatial intelligence and content creation.

This divergence is healthy. It means world models are no longer a vague concept but are forming concrete technology stacks and product forms.

## 11. Reading Recommendations

Finally, a practical reading framework.

**When reading any world model paper, ask 6 questions first:**

1. What is the state representation?
2. What does the dynamics model predict?
3. Does action enter the dynamics?
4. How long is the prediction horizon?
5. How do they verify the predictions are actually useful?
6. What is the final downstream task?

Questions 5 and 6 are especially critical. Otherwise you risk the common pattern: video prediction benchmarks look impressive, but they don't help with robot control. This is precisely one of the most important issues to discuss in the current world model field.

**If you're new to world models:**

Start with any one of the surveys mentioned above to build a global perspective. Then work through this blog's foundational series — from [What is a World Model](/en/articles/world-model-intro/) to the [RSSM Deep Dive](/en/articles/rssm-deep-dive/) to [Dreamer Explained](/en/articles/2026-08-25-dreamer-explained/) — to solidify core concepts.

**If you're already doing world model research:**

Focus on the Cosmos technical report and Genie 3 paper to understand how industry approaches large-scale world models. Then follow AMI Labs's progress — if the JEPA approach succeeds, it could reshape the field's technical paradigm.

**If you care about engineering deployment:**

Cosmos's open-source toolchain is priority one. Then look at the Sim-to-Real chapters in the World Model for Robot Learning survey. Finally, follow NVIDIA and Google's latest sharing on robot deployment.

---

The biggest change in world models in 2026 is not any single breakthrough, but **divergence** — different paradigms heading toward different scenarios, evaluation criteria shifting from "can it generate?" to "is it useful?" Understanding this divergence matters more than chasing any single project.

*In the next article, I'll discuss what kind of team you need to start a company in embodied AI — not a generic business plan, but a pragmatic analysis from an engineer's perspective. Stay tuned.*
