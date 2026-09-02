---
title: "From RSSM to Modern State-Space Models: How the Engine of World Models Evolves"
slug: "2026-09-04-rssm-beyond"
date: 2026-09-04
draft: false
categories: ["World Models", "Paper Analysis"]
tags: ["RSSM", "State-Space Model", "TD-MPC", "Mamba", "DreamerV3", "World Model", "Latent Dynamics"]
description: "RSSM is the core engine of the Dreamer family of world models, but the landscape of state-space modeling has changed significantly over the past few years. From S4 to Mamba to TD-MPC2, this article places RSSM within the broader evolution of state-space modeling, discussing the tension between specialized dynamics models and general-purpose sequence architectures, as well as the possible future directions for world model engines."
toc: true
related_articles:
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
  - 2026-09-01-world-model-h2-review
  - 2026-09-02-jepa-deep-dive
  - 2026-08-24-rssm-recap
---

In the [previous RSSM series](/en/articles/rssm-deep-dive/), I broke down the architectural details of RSSM in depth — the dual-track design of deterministic and stochastic paths, the choice of categorical latent, the training tricks of KL balancing, and the reset strategy during the imagination phase.

But that series of articles focused primarily on RSSM itself. Today, I'd like to take a different perspective: **where does RSSM stand when placed within the broader evolution of state-space modeling? And after RSSM, which direction is the "engine" of world models heading?**

## Core Design Choices of RSSM

Let's start with a quick review. The core of RSSM (Recurrent State-Space Model) is a dual-track latent state structure:

```
              h_t (deterministic)     s_t (stochastic)
                   │                        │
              GRU update              categorical prior
                   │                        │
                   └────────┬───────────────┘
                            ↓
                    p(o_t | h_t, s_t)
                     observation model
```

The **deterministic path h_t** uses a GRU to capture long-range dependencies, while the **stochastic path s_t** uses a categorical distribution to model environmental uncertainty. The two paths together condition the observation model.

This design has several noteworthy characteristics:

**First, it is not a pure neural state-space model.** The classical state-space model takes the form z_{t+1} = f(z_t, a_t), o_t = g(z_t). RSSM is closer to a **partially observable** latent variable model — h_t takes on part of the role of observable history encoding, while s_t is the "true" latent state.

**Second, the categorical latent is an engineering-driven choice.** Unlike continuous Gaussian latents, the categorical latent allows the KL divergence between the prior and posterior to be computed exactly, avoiding the KL estimation variance problems common in continuous distributions. This is a pragmatic design decision, not a theoretically optimal solution.

**Third, imagination is the core innovation.** RSSM is not only used to fit historical data, but also to "imagine" future trajectories in latent space — starting from the current posterior, using the prior to .rollout multiple future paths, and then learning policies within imagination. This has kept the Dreamer family consistently ahead in sample efficiency.

These design choices were progressively validated through the evolution from DreamerV1 → V2 → V3. But are they the only possible path?

## Another Line in State-Space Modeling: From S4 to Mamba

During the same period as RSSM's development, a parallel line of state-space models emerged in the NLP and sequence modeling domains.

**S4 (Structured State Space for Sequences, 2022)** introduced a structured-parameterized continuous-time state-space model. Its core formulation is the classical linear SSM:

```
h'(t) = A h(t) + B x(t)
y(t) = C h(t) + D x(t)
```

But the key innovation lies in the structured constraints on the A matrix — through diagonalization (HiPPO initialization), the model is able to capture ultra-long-range dependencies. S4 demonstrated Transformer-level performance on long-sequence benchmarks, but with higher computational efficiency (linear complexity).

**Mamba (2024)** built upon S4 by introducing **selectivity** — allowing the SSM parameters to change dynamically based on the input. This broke the "time-invariance" limitation of classical SSMs, enabling the model to selectively remember or forget information. Mamba achieved near-Transformer performance on language modeling while maintaining the linear inference efficiency of SSMs.

What is the relationship between these two lines and RSSM?

**Formally similar, but with different goals.** Both RSSM and S4/Mamba use a "latent state + state transition" framework, but RSSM's latent state is a **low-dimensional representation of environment dynamics** (the latent state of a world model), whereas S4/Mamba's latent state is a **compressed representation of sequence information** (the hidden state of a sequence model).

In other words:

```
RSSM:      latent state ≈ compressed description of the world (physical state, dynamics)
S4/Mamba:  latent state ≈ compressed description of the sequence (context, semantics)
```

This distinction is crucial. RSSM's latent state is designed to answer "what state is the world in now, and how will it change next?"; S4/Mamba's latent state is designed to answer "what is the context of this sequence, and what should the next token be?"

## TD-MPC2: Building World Models Without RSSM

TD-MPC2 (2024, arXiv:2310.16828) represents an entirely different design philosophy for world models.

TD-MPC2's world model does not use RSSM's dual-track structure, but instead employs a more concise architecture:

```
Encoder:     e_t = E(o_t)                    → encode observations into latent
Dynamics:    z_{t+1} = f_θ(z_t, a_t)         → MLP ensemble predicts next latent
Reward:      r_t = R(z_t, a_t)               → predict reward
Termination: d_t = D(z_t)                     → predict termination
```

**No deterministic/stochastic dual track, no categorical latent, no KL balancing.** It uses a more direct approach: the encoder maps observations to latent space, the MLP ensemble performs dynamics prediction directly in latent space, and then MPC (Model Predictive Control) selects actions in imagination.

TD-MPC2's key innovations lie not in the world model architecture itself, but in three areas:

**First, MLP ensemble uncertainty estimation.** Using the prediction disagreement among multiple MLPs to estimate model uncertainty, then leveraging this uncertainty within MPC to guide exploration.

**Second, cross-task/cross-embodiment scalability.** TD-MPC2 demonstrated scalability across 139 tasks and multiple robot morphologies — something the Dreamer family did not systematically showcase.

**Third, latent space consistency.** Through a consistency loss, the encoder and dynamics model are kept aligned in latent space, eliminating the hyperparameter tuning burden of KL balancing in RSSM.

From RSSM to TD-MPC2, a clear trend emerges: **world model architectures are becoming more concise, but training objectives and scaling strategies are becoming more important.**

## Comparing the Three Approaches

Placing RSSM, S4/Mamba, and TD-MPC2 side by side, we can see three distinct design philosophies for "world model engines":

| Dimension | RSSM (Dreamer) | S4 / Mamba | TD-MPC2 |
|------|---------------|------------|---------|
| **Core Formulation** | Dual-track latent state (deterministic + stochastic) | Continuous/discrete linear SSM | Encoder + MLP dynamics |
| **Latent State Meaning** | Environment dynamics representation | Sequence context compression | Task-relevant latent |
| **Primary Training Objective** | reconstruction + KL + imagination reward | next-token / sequence prediction | reconstruction + consistency + MPC |
| **Inference Method** | imagination → actor-critic | sequence forward pass | latent-space MPC |
| **Strengths** | sample efficiency, long-horizon imagination | long-sequence efficiency, scalability | architectural simplicity, cross-task scaling |
| **Limitations** | complex tuning, limited categorical precision | does not directly model dynamics | relies on MPC, limited long-range planning |

These three approaches are not mutually substitutable. They address problems at different levels:

- **RSSM** addresses "how to simultaneously capture deterministic dynamics and stochastic uncertainty in latent space, and support imagination-based policy learning"
- **S4/Mamba** addresses "how to efficiently process ultra-long sequences while maintaining selective memory of key information"
- **TD-MPC2** addresses "how to achieve cross-task, cross-embodiment world model scaling with a concise architecture"

## Directions of Convergence: What Kind of Sequence Architecture Do World Models Need?

A natural question arises: will these lines converge?

Looking at recent work, there are several convergence trends worth noting.

### Trend 1: VLAs Are Borrowing SSM Architectures

Some new VLA work has begun exploring the use of Mamba-style architectures to replace Transformer backbones for processing vision-language-action sequences. The motivation is straightforward: the self-attention mechanism of Transformers has O(n²) complexity, which is computationally expensive for robot policies that need to process long observation histories.

But there is a conceptual tension here: **Mamba's latent state is sequence context compression, not an environment dynamics representation.** Using Mamba as a VLA backbone can make the model more efficient at processing long sequences, but it will not automatically gain RSSM's ability to "simulate physical dynamics in latent space."

From a modeling perspective, this connects precisely to the problem discussed in the VLA series: what VLAs lack is not sequence processing capability, but an explicit action-conditioned prediction interface. Placing Mamba inside a VLA can improve sequence processing efficiency, but it will not automatically turn a VLA into a world model.

### Trend 2: World Models Are Adopting Transformer Architectures

Conversely, some world model work has begun using Transformers to replace RSSM's GRU + categorical latent structure. For example, IRIS (Implicit Representation of Images with Self-supervised Transformers) uses Transformers for world modeling in a discrete token space.

The advantage of this direction is that Transformer attention mechanisms naturally support "attending to key past time steps," without needing to compress all historical information through h_t as in RSSM. The disadvantage is that computational cost grows quadratically with sequence length, which is unfriendly for real-time control.

### Trend 3: Hybrid Architectures of Latent Dynamics + Foundation Models

An even more interesting direction is: **using foundation models for semantic understanding and representation, and latent dynamics models for physical prediction.**

```
Foundation Model (Transformer / Mamba)
      ↓
  semantic representation
      ↓
Latent Dynamics Model (RSSM-style / TD-MPC-style)
      ↓
  action-conditioned future prediction
      ↓
  planning / policy
```

This hybrid architecture attempts to combine the strengths of both: foundation models provide powerful semantic generalization capability, while latent dynamics models provide physical prediction capability.

## The Legacy of RSSM

Returning to the original question: where does RSSM stand in the evolution of state-space modeling?

I think RSSM's contributions can be summarized at three levels.

**First, it demonstrated the practicality of latent imagination.** Before Dreamer, "learning policies in imagination" was more of a theoretical concept. The Dreamer series, through the complete engineering implementation of RSSM + imagination, proved that this approach is viable and even sample-efficient on real robot tasks.

**Second, it provided a reference architecture.** RSSM's dual-track design (deterministic + stochastic) was not an accidental choice, but rather reflects a deep design principle: **world models need to simultaneously capture predictable dynamics and unpredictable uncertainty.** This principle will not become outdated, even if the specific implementation (GRU, categorical latent) may be superseded.

**Third, it delineated a design space.** RSSM proved that "performing dynamics prediction in latent space + training policies in imagination" is a viable path. Subsequent work — whether TD-MPC2's simplification or the architectural substitution of Transformer-based world models — has been exploring within the design space that RSSM delineated.

From a broader perspective, RSSM's relationship to world models may be analogous to LSTM's relationship to sequence modeling — it is not the final architecture, but it proved a key concept and provided a design template for subsequent work.

## Open Questions

There are several questions that I think do not yet have clear answers.

**What should the latent state be?** RSSM uses categorical latents, TD-MPC2 uses continuous latents, S4/Mamba use deterministic hidden states. Which representation is best suited for world models? There may be no single answer — different tasks and different embodiments may require different latent state structures.

**Should dynamics models be specialized or general-purpose?** RSSM is a specialized dynamics model (designed for environment modeling), while Mamba is a general-purpose sequence model. Which should the engine of a world model be? Hybrid architectures may be the answer, but the optimal combination remains unclear.

**Will scaling change architectural choices?** In the small-data regime, RSSM's inductive biases (dual-track structure, categorical prior) may be important. But in the large-data regime, more concise architectures (like TD-MPC2) may win out because they are easier to scale. This is a question worth systematically investigating.

**What are the limits of imagination?** RSSM's imagination in DreamerV3 can already rollout very long trajectories. But the quality of imagination degrades with rollout length — is this a fundamental limitation, or can it be resolved through better architectures?

---

*This article is a complementary perspective to the RSSM series. If you want to see the specific architectural details of RSSM, refer to the [previous RSSM deep dive](/en/articles/rssm-deep-dive/) and the [source code walkthrough](/en/articles/2026-08-19-rssm-code-walkthrough/).*

*The next article is the middle installment of the VLA series — the Pi Family and the Evolution of Action Interfaces.*
