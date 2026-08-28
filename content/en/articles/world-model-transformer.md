---
title: "When World Models Meet Transformers: From RSSM to Large-Scale Sequence Modeling"
slug: "world-model-transformer"
date: 2026-08-12
draft: false
categories: ["World Models"]
tags: ["Transformer", "World Models", "RSSM", "DreamerV3", "UniSim", "Cosmos", "Attention", "Sequence Modeling", "Video Prediction"]
description: "Tracing the convergence of world models and Transformer architectures — from RSSM's recurrent state space to UniSim and Cosmos, and what large-scale sequence modeling means for next-generation world simulators."
toc: true
related_articles:
  - rssm-deep-dive
  - world-model-representations
  - 2026-08-31-world-model-future
  - 2026-08-25-dreamer-explained
  - vla-vs-world-model
  - td-mpc-world-model-control
aliases:
  - /en/articles/world-model-transformer.html
---


In previous articles, we covered the RSSM architecture and training techniques in DreamerV3 in depth. RSSM is a classic design in reinforcement learning world models, but if you follow recent research, you'll notice a clear trend: world models are becoming Transformer-based.

From Google's UniSim to Wayve's GAIA-1, from NVIDIA's Cosmos to solutions from domestic embodied AI teams, the Transformer is emerging as a key technical approach for large-scale world models.

Why is this happening? What are the limitations of RSSM? What advantages do Transformers bring? What is the architectural evolution path for world models?

This article traces this technical trajectory.

Before diving in, let's clarify one concept. A so-called unified world model does not mean a model that precisely replicates the entire physical world. Rather, it aims to simultaneously understand vision, language, action, and environmental dynamics through a unified representation space — enabling the same model to both "see" and understand scenes, "think" through causal relationships, and "act" to make decisions. This is a frontier direction in current world model research.

## The Design Logic and Limitations of RSSM

Let's first review why RSSM was designed in the first place.

The core idea of the RSSM (Recurrent State-Space Model) is to split the hidden state into two tracks: a deterministic state and a stochastic state. The deterministic state is updated with a GRU, capturing regular dynamics; the stochastic state is sampled through a discrete categorical distribution — DreamerV3 uses 32 discrete categorical variables, each with 64 categories, combining to form a stochastic latent space with a capacity of 64^32, capturing environmental uncertainty.

This design has several advantages:

Computational efficiency. The GRU is a recurrent network — each step only requires the hidden state from the previous step, without needing to process the entire sequence. Memory overhead for both training and inference is relatively small.

Well-structured latent space. The dual-track design of deterministic + stochastic allows the model to learn both "regularities" and "uncertainty." This is important for modeling the physical world — macroscopic physical laws are typically quite deterministic, while observations and environmental perturbations are stochastic.

Imagination-friendly training. In the latent space, it's convenient to "imagine" future trajectories: starting from the current state, the GRU advances step by step, sampling a stochastic state at each step to produce a sequence of hidden states, which are then decoded into observations.

However, RSSM also has several notable limitations:

Long-range dependency issues. RSSM maintains historical information through the deterministic hidden state, but since state updates are still recursive, information over very long time horizons must be compressed into a fixed-dimension hidden state, potentially creating an information bottleneck. When tasks require remembering information from the distant past (e.g., delayed rewards, long-range causal chains), RSSM's performance may fall short.

Limited parallelization. Recurrent networks are sequential along the time dimension — each step depends on the output of the previous step. However, RSSM training is not entirely serial: the observation encoder can process in parallel, and rollout imagination can also be parallelized along the batch dimension. The real bottleneck lies in the dependencies along the time dimension, which ties training speed to sequence length.

Scaling gap. Transformers more easily inherit the large-scale training paradigms from the language model era — large-scale pretraining experience, data scale advantages, and mature industrial infrastructure. While RSSM can also be scaled up (e.g., DreamerV3's 4096-dimensional deterministic state), it has yet to form a scaling ecosystem of comparable magnitude.

## The Rise of Transformer World Models

The success of Transformers in LLMs naturally leads to applying them to world models. The core idea behind Transformer-based world models is:

Take past sequences of observations and actions as input, and use a Transformer to predict the next observation (or hidden state) — essentially a sequence-to-sequence prediction problem.

This approach has several inherent advantages:

Long-range dependencies. Self-attention can directly establish connections between any temporal positions, thereby alleviating the long-range information transfer bottleneck in recurrent structures. For tasks requiring long-range memory, Transformers are theoretically stronger — though in practice they are still constrained by context length limits and the computational cost of attention.

Parallel training. Transformers can achieve highly parallel computation along the sequence dimension during training, reducing the serial bottleneck caused by temporal dependencies compared to recurrent networks. This means faster training on GPUs and easier utilization of large-scale compute.

Scaling laws. Transformers have demonstrated favorable scaling laws in LLMs. If world models also use Transformers, then in theory they can leverage LLM scaling experience — increase model size, add more data, and continue improving performance.

Unification with LLMs. If world models also use Transformers, then their architecture is unified with LLMs. This means LLM infrastructure (training frameworks, inference optimizations) can be reused, and language understanding and physical prediction can even be unified within a single model.

Beyond these general advantages, there are several unique reasons why world models specifically benefit from Transformers:

Spatial dimensions. World model inputs are typically video (time x height x width), and Transformers are naturally suited to splitting images into patch tokens for modeling. Compared to recurrent networks processing frame by frame, Transformers can simultaneously attend to dependencies along both spatial and temporal dimensions.

Multimodal unification. World model inputs include images, language instructions, actions, proprioception, and other modalities. The tokenization framework of Transformers is naturally suited to unifying these heterogeneous inputs — each modality is encoded as a token sequence, then processed by the same attention mechanism.

Long-horizon planning. Robotic tasks often require long-term planning across time — a current action may affect goal achievement minutes later. The cross-temporal attention mechanism of Transformers can directly establish associations between "current actions" and "future goals," which is difficult for recurrent networks to achieve efficiently.

Current Transformer world models can be classified along two dimensions:

By prediction space: pixel space (directly generating in visual space, as in early video prediction models), latent space (predicting in a compressed representation space — the Dreamer series belongs to this category of latent dynamics models), token space (discretizing observations into tokens before prediction, as in Genie).

By generation method: autoregressive (step-by-step generation, as in Genie, VideoPoet), diffusion (conditional diffusion generation, as in DIAMOND, Cosmos), masked prediction (masked prediction, similar to BERT's bidirectional approach).

These two dimensions can be combined — for example, Cosmos represents a line of research combining diffusion transformers with latent video modeling. Works like GAIA-1 and Genie fall into the latent/token space + autoregressive combination. The choice of combination depends on the task's requirements for real-time performance, generation quality, and computational efficiency.

## Representative Works

Below we introduce several representative Transformer world models.

### 1. UniSim (Google, 2023-2024)

UniSim is Google's Universal Simulator project. Its core idea is: use large-scale generative models to build a general-purpose environment simulator — based on video generation models and conditional generation, taking "observation + action + condition" as input and outputting "the next observation." Conditions can be text instructions, goal images, or other control signals.

Unlike explicit latent dynamics models like RSSM, UniSim leans more toward generative environment simulation — it does not explicitly model hidden state transitions, but instead implicitly "learns" environmental dynamics through large-scale video generation. UniSim demonstrated the possibility of building a unified environment simulator using large-scale generative models, though it has not yet reached the level of "one model covering all of the physical world" — it is more of a directional exploration.

### 2. GAIA-1 (Wayve, 2023)

GAIA-1 is a generative world model proposed by the autonomous driving company Wayve. It comprises three core modules: a multimodal tokenizer, a generative model, and a world representation. It takes past video frames and actions as input and predicts future video frames.

What makes GAIA-1 notable is its scale — publicly available information indicates approximately 9 billion parameters, trained on large amounts of real driving data. The generated video quality is quite high, capturing complex dynamics such as weather changes, lighting variations, and traffic participant behaviors.

GAIA-1 demonstrated the scaling potential of generative world models on real driving data. This is an attempt to apply LLM-style scaling thinking to world models.

### 3. Cosmos (NVIDIA, 2025)

Cosmos is NVIDIA's foundation model platform for world models. The Cosmos series adopts a Transformer-based generative architecture, including diffusion transformer and autoregressive transformer approaches, providing pretrained video generation and physical simulation capabilities with support for fine-tuning on custom data. Cosmos's goal is not simply to predict videos, but to build foundational world models for Physical AI, providing learnable environmental priors for robotics and autonomous driving.

The launch of Cosmos signals that in industry, world models are moving from research prototypes toward engineered platforms.

### 4. DIAMOND (2024)

DIAMOND (DIffusion As a Model of eNvironment Dreams) introduces diffusion models into world models. It models environmental dynamics as a conditional diffusion process — given past observations and actions, a diffusion model generates future environment states. DIAMOND achieved strong results on Atari games, demonstrating the potential of diffusion-based world models.

## RSSM vs. Transformers: Not a Simple Replacement

Although Transformers offer many advantages, saying they "replace" RSSM may be an oversimplification. Each has its appropriate use cases. Below is a comparison across several key dimensions:

| Dimension | RSSM (DreamerV3) | Transformer World Models |
| --- | --- | --- |
| Sequence modeling | GRU recurrent, step-by-step hidden state updates | Self-attention, global parallel computation |
| Parallel training | Sequential along time dimension (encoder/rollout can parallelize) | Highly parallel along sequence dimension |
| Long-range dependencies | Limited by fixed hidden state capacity | Strong (direct attention) |
| Hidden representations | Structured latent state (deterministic + stochastic) | Token / latent representation |
| Data requirements | Friendly to small-to-medium data | Typically requires large-scale pretraining |
| Representative works | DreamerV3 | UniSim, GAIA-1, Cosmos |

Data volume. Transformers typically require large amounts of data to realize their advantages. If data is limited (e.g., only a few thousand robot manipulation trajectories), RSSM's inductive biases (recurrence + hidden state) may be more effective — it can learn reasonable dynamics with fewer parameters.

Real-time performance. RSSM's recurrent structure advances step by step during inference, with each step only requiring the previous step's hidden state. Transformer inference typically requires maintaining historical context, and practical systems reduce costs through KV cache, windowed attention, or memory tokens. In scenarios with strict real-time requirements, RSSM may be more suitable.

Task type. If a task requires long-range memory or global reasoning (e.g., navigation, strategic planning), Transformer self-attention is more advantageous. If the task is short-term and local (e.g., rapid manipulation reactions), RSSM may suffice.

Computational resources. Transformer training demands significant GPU resources. Under the same compute budget, RSSM is generally easier to deploy on resource-constrained devices — which is why DreamerV3 can be trained on a single GPU.

So a more accurate statement would be: Transformers are becoming an important direction for world models, but RSSM still has its value. Which architecture to choose depends on the specific task, data, and resource conditions.

## Hybrid Architectures: The Best of Both Worlds?

Since RSSM and Transformers each have their strengths and weaknesses, a natural question arises: can the two be combined?

Several works are already exploring this direction:

Transformer + RSSM. Replace the GRU in RSSM (deterministic state update) with a Transformer, while retaining the discrete categorical distribution design for the stochastic state. The Dreamer series has always used CNN/MLP encoder + RSSM at its core, but subsequent explorations are attempting to replace the GRU with a Transformer for sequence modeling. This gains Transformer's long-range modeling capability while preserving RSSM's latent space structure.

Hierarchical architecture. Use RSSM at the lower level for short-term dynamics prediction, and a Transformer at the higher level for long-range planning and reasoning. This hierarchical design has analogs in LLMs (e.g., local attention + global attention), and is especially promising in robotic control — short-term reactions handled by efficient recurrent networks, long-term planning handled by powerful Transformers.

Tokenization + Transformer. Discretize observations (e.g., video frames) into tokens, then use a Transformer to make predictions over the token sequence. DeepMind's Genie (2024) is a representative of this approach — it comprises four modules: a video tokenizer, a latent action model, a dynamics model, and a decoder, where the dynamics model uses a Transformer to predict the token sequence for the next frame, enabling the generation of interactive 2D environments from a single image. This approach unifies the architecture of world models and LLMs, representing a possible path toward a "general-purpose world model."

VLM + World Model. Future world models may not exist independently, but instead serve as internal prediction modules within Vision-Language-Action models (VLA). The model would simultaneously possess language understanding, scene understanding, dynamics prediction, and action planning capabilities — taking images, language instructions, and state as input, predicting future states through a foundation model, then outputting actions via a planning module or policy model. This is an important direction for Physical AI, and a key step in world models evolving from standalone modules toward unified architectures.

Hybrid architectures are an active research direction, and there is no clear optimal solution yet. But the trend shows Transformer elements appearing with increasing frequency in world models.

## Implications for Practitioners

The Transformer trend in world models offers several implications for practitioners:

Learn Transformers. If you've only worked with RSSM so far, it's time to study Transformers in depth. Concepts like self-attention, positional encoding, and KV cache are becoming increasingly important in world models.

Pay attention to scaling. The performance of Transformer world models depends heavily on model scale and data volume. Follow how scaling laws manifest in world models — understand "how much model + how much data = what performance."

Don't abandon RSSM. RSSM still has advantages in resource-constrained, data-limited scenarios. DreamerV3's training techniques and experience also have reference value for Transformer world models (e.g., imagination training, KL regularization).

Watch for unified architectures. The boundary between world models and LLMs is blurring. In the future, we may see "unified Physical AI models" that simultaneously understand language, vision, and action. Understanding this trend helps in grasping the long-term technical direction.

## Summary

World models are entering a phase of coexisting approaches: compact latent dynamics represented by RSSM, large-scale sequence modeling represented by Transformers, generative prediction represented by Diffusion, and token-based foundation models are converging. Transformers' advantages in long-range modeling, parallel training, and scaling laws make them an important technical direction for large-scale world models. Works like UniSim, GAIA-1, and Cosmos have already demonstrated the potential of this direction.

But this does not mean RSSM is obsolete. In scenarios with limited data, constrained resources, and high real-time requirements, RSSM still has its value. Hybrid architectures may be the future direction — combining the strengths of both to balance efficiency and performance.

For practitioners, the key is to understand the design logic and applicable scenarios of different architectures, rather than simply chasing the latest trends. The core problem of world models — how to efficiently learn and predict physical dynamics — remains unchanged; architectures are merely tools for solving that problem.
