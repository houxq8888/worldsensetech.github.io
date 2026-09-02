---
title: "From RSSM to Modern Latent Dynamics: How the 'Engine' of World Models Evolves"
slug: "2026-09-04-rssm-beyond"
date: 2026-09-04
draft: false
categories: ["World Models", "Paper Analysis"]
tags: ["RSSM", "State-Space Model", "TD-MPC", "Mamba", "DreamerV3", "World Model", "Latent Dynamics"]
description: "RSSM is the core engine of the Dreamer family of world models, but the landscape of state-space modeling has changed significantly in recent years. This article places RSSM within the broader evolution of state-space modeling -- distinguishing between latent dynamics and sequence backbone layers, discussing TD-MPC2's planning + value fusion approach, and exploring possible future directions for world model engines."
toc: true
related_articles:
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
  - 2026-09-01-world-model-h2-review
  - 2026-09-02-jepa-deep-dive
  - 2026-08-24-rssm-recap
---

> **On the scope of this article:** The "state-space modeling" discussed here encompasses two related but distinct layers: latent state-space models for environment dynamics modeling (such as RSSM and TD-MPC2's latent dynamics), and SSM architectures for efficient sequence processing (such as S4 and Mamba). The two are formally similar but should not be directly equated.

In the [previous RSSM series](/en/articles/rssm-deep-dive/), I broke down the architectural details of RSSM in depth -- the dual-track design of deterministic and stochastic paths, the choice of categorical latent, the training tricks of KL balancing, and the reset strategy during the imagination phase.

But that series focused primarily on RSSM itself. Today, I'd like to take a different perspective: **where does RSSM stand when placed within the broader evolution of state-space modeling? And after RSSM, which direction is the "engine" of world models heading?**

## Core Design Choices of RSSM

Let's start with a quick review. The core of RSSM (Recurrent State-Space Model) is a dual-track latent state structure:

```
              h_t (deterministic)     z_t (stochastic)
                   │                        │
              GRU update              categorical prior
                   │                        │
                   └────────┬───────────────┘
                            ↓
                    p(o_t | h_t, z_t)
                     observation model
```

The **deterministic path h_t** is recursively updated by h_t = f(h_{t-1}, z_{t-1}, a_{t-1}), responsible for accumulating historical information. The **stochastic path z_t** models the uncertainty of the current state through a categorical distribution's prior/posterior. Together, they constitute RSSM's **belief state** s_t = (h_t, z_t), which conditions the observation model.

This design has several noteworthy characteristics:

**First, RSSM is more accurately described as a belief-state model.** The classical state-space model takes the form z_{t+1} = f(z_t, a_t), o_t = g(z_t). RSSM is closer to a **partially observable** latent variable model. From a POMDP perspective, RSSM's recurrent state (h_t, z_t) can be understood as a parameterization of the belief state formed from historical observations and actions -- it does not directly recover the environment's "true physical state," but rather learns a latent belief sufficient to support prediction and control.

**Second, the categorical latent is an engineering-driven choice.** An engineering advantage of the categorical latent is that both prior and posterior are explicit discrete distributions, so KL divergence can be computed analytically; at the same time, it provides a more flexible discrete stochastic representation than a single continuous Gaussian latent. DreamerV2/V3 further combined this latent with a straight-through estimator. This is a pragmatic design decision, not a theoretically optimal solution.

**Third, one of Dreamer's key contributions is combining learned latent dynamics with actor-critic learning, enabling policy/value to be trained primarily on imagined latent trajectories.** The concept of imagination / model rollout in model-based RL predates Dreamer. Dreamer's innovation lies in: RSSM is not only used to fit historical data, but also to "imagine" future trajectories in latent space -- starting from the current posterior, using the prior to rollout multiple future paths, and then training the actor and critic on these imagined trajectories. This has kept the Dreamer family consistently ahead in sample efficiency.

These design choices were progressively validated through the evolution from DreamerV1 → V2 → V3. But are they the only possible path?

## Another Line in Sequence Modeling: From S4 to Mamba

During the same period as RSSM's development, a parallel line of state-space models emerged in the NLP and sequence modeling domains. **It is important to emphasize: S4/Mamba are first and foremost sequence models, not world models.** They address efficient sequence processing, not action-conditioned environment dynamics.

**S4 (Structured State Space for Sequences, 2022)** introduced a structured-parameterized continuous-time state-space model. Its core formulation is the classical linear SSM:

```
h'(t) = A h(t) + B x(t)
y(t) = C h(t) + D x(t)
```

The key to S4 is not simply "doing a diagonalization," but rather a **structured parameterization** of the classical SSM's state matrix A -- combining HiPPO initialization, low-rank correction, and normalized/diagonal parameterization, implemented through efficient Cauchy kernel computation -- making long-range memory both expressive and computationally efficient. S4 demonstrated Transformer-level performance on long-sequence benchmarks, but with higher computational efficiency.

**Mamba (2024)** built upon S4 by introducing **selectivity** -- making the SSM's key parameters (B_t, C_t, Δ_t) input-dependent, thereby obtaining a **selective state space**. This means that state transition / information retention can vary based on the current token content, enabling the model to selectively remember or forget information. Mamba achieved near-Transformer performance on language modeling, while exhibiting linear scaling in sequence length and efficient hardware execution through selective scan.

What is the relationship between these two lines and RSSM?

**Formally similar, but with different goals.** Both RSSM and S4/Mamba use a "latent state + state transition" framework, but RSSM's latent state is a **task-relevant latent belief about environment history** -- sufficient to support observation prediction, reward prediction, and control; whereas S4/Mamba's latent state is a **contextual compression of input sequence history** -- serving sequence prediction.

In other words:

```
RSSM:
  latent state ≈ task-relevant latent belief about environment history
               → sufficient for observation / reward / control prediction

S4/Mamba:
  latent state ≈ contextual compression of input sequence history
               → serving sequence prediction
```

This distinction is crucial. RSSM's latent state is designed to answer "what state is the world in now, and how will it change next?"; S4/Mamba's latent state is designed to answer "what is the context of this sequence, and what should the next token be?"

## TD-MPC2: An Alternative Latent Dynamics Approach

TD-MPC2 (2024, arXiv:2310.16828) represents a different design philosophy for world models compared to RSSM.

TD-MPC2 does not use RSSM's dual-track structure, but instead employs a more concise architecture:

```
Encoder:     e_t = E(o_t)                    → encode observations into latent
Dynamics:    z_{t+1} = f_θ(z_t, a_t)         → predict next latent
Reward:      r_t = R(z_t, a_t)               → predict reward
Q-function:  Q(z_t, a_t)                     → long-term value estimation (5 Q ensemble)
Policy:      π(a_t | z_t)                    → policy prior
```

**No deterministic/stochastic dual track, no categorical latent, no KL balancing.** It uses a more direct approach: the encoder maps observations to latent space, performs dynamics prediction in latent space, and then combines short-horizon MPC with long-term Q-value estimation to select actions.

The core of TD-MPC2 is not a complex latent-state decomposition, but rather the combination of **concise latent dynamics, task-conditioned representation, short-horizon MPC, and long-horizon Q-value estimation.** Its three key innovations:

**First, the combination of latent-space MPC and Q-function ensemble.** TD-MPC2 performs short-horizon rollouts on latent dynamics and uses a Q-function ensemble (defaulting to 5 Q-functions, with TD target using the minimum of a randomly subsampled Q-function) to provide long-term value estimation, thereby bridging short-horizon MPC planning with long-horizon TD bootstrapping. This is what makes TD-MPC2 truly elegant.

**Second, task-conditioned cross-task / cross-embodiment scaling.** TD-MPC2 demonstrated scalability across 139 tasks and multiple robot morphologies. This is not "one dynamics model automatically understanding all embodiments," but rather using **task embeddings / task-conditioned components** to adapt the same network to different tasks/morphologies -- the encoder, dynamics, reward, policy prior, and Q components are all linked to task embeddings.

**Third, stabilization of latent representations.** TD-MPC2 uses SimNorm to normalize latent states, and jointly trains the encoder, dynamics, reward, policy prior, and Q-functions, so that the latent representation serves both prediction and control -- without requiring the stochastic prior/posterior KL constraints of RSSM.

From RSSM to TD-MPC2, a clear trend emerges: **TD-MPC2 demonstrates an alternative path: instead of relying on complex stochastic recurrent states, it combines dynamics prediction, short-horizon MPC, and long-horizon value estimation within a compact latent space.**

## Four-Layer Architecture: Not Just "Three Approaches"

Placing the models discussed above together reveals that they do not operate at the same level. A more accurate understanding is a four-layer architecture:

```
                    Sequence / State Modeling
                              │
              ┌───────────────┴───────────────┐
              ↓                               ↓
      Sequence Backbone                 Latent Dynamics
      S4 / Mamba / Transformer          RSSM / MLP / Transformer
              │                               │
              │                               ↓
              │                     Action-conditioned prediction
              │                               │
              └───────────────┬───────────────┘
                              ↓
                         World Model
                              │
                    ┌─────────┴─────────┐
                    ↓                   ↓
                 Imagination            MPC + Value
                 Dreamer             TD-MPC2
                    ↓                   ↓
                    └─────────┬─────────┘
                              ↓
                           Policy
```

This layering matters:

- **S4/Mamba** are sequence engines -- solving how to efficiently maintain state/history
- **RSSM** is a latent dynamics engine -- solving how to model environment dynamics in latent space
- **TD-MPC2** is a latent dynamics + value + MPC system -- bridging short-horizon planning with long-horizon value
- **Dreamer** is a latent dynamics + imagination + actor-critic system -- training policies in imagination

The relationship between them is not "three parallel approaches," but rather **components at different levels that can be combined.** For example, Mamba can serve as a VLA's sequence backbone, but that does not automatically make the VLA a world model -- only when it is trained as a queryable predictive model about the relationships between environment states, actions, and future states does it truly become a world model.

### Comparison Table

| Dimension | RSSM / Dreamer | S4 / Mamba | TD-MPC2 |
|------|---------------|------------|---------|
| **Core Formulation** | Dual-track latent state (deterministic + stochastic) | Continuous/discrete linear SSM | Encoder + MLP dynamics + Q ensemble |
| **Latent State Meaning** | Task-relevant latent belief about environment history | Sequence context compression | Task-conditioned latent |
| **Primary Training Mechanism** | observation/reward/continuation prediction + KL regularization; actor-critic trained on imagined trajectories | sequence modeling objective | latent dynamics/reward + TD/Q learning + MPC |
| **Inference Method** | imagination → actor-critic | sequence forward pass | latent-space MPC + Q-value bootstrapping |
| **Strengths** | sample efficiency, long-horizon imagination | long-sequence efficiency, scalability | planning + value fusion, task-conditioned cross-task scaling |
| **Limitations** | complex tuning, limited categorical precision | does not directly model dynamics | planning relies on limited-horizon latent MPC; long-term decisions depend on learned Q-function bootstrapping |

## Directions of Convergence: What Kind of Sequence Architecture Do World Models Need?

A natural question arises: will these lines converge?

Looking at recent work, there are several convergence trends worth noting.

### Trend 1: VLAs Are Borrowing SSM Architectures

Some new VLA work has begun exploring the use of Mamba-style architectures to replace Transformer backbones for processing vision-language-action sequences. The motivation is straightforward: under standard global self-attention, Transformer computational/memory costs grow quadratically with sequence length; for robot policies that need to process long observation histories, this is computationally expensive. Practical world models often mitigate this through token compression, local attention, or other structural choices.

But there is a conceptual tension here: **Mamba's latent state is sequence context compression, not an environment dynamics representation.** Using Mamba as a VLA backbone can make the model more efficient at processing long sequences, but it will not automatically gain RSSM's ability to "simulate physical dynamics in latent space."

From a modeling perspective, this connects precisely to the problem discussed in the VLA series: what VLAs lack is not sequence processing capability, but an explicit action-conditioned prediction interface. Placing Mamba inside a VLA can improve sequence processing efficiency, but it will not automatically turn a VLA into a world model.

### Trend 2: World Models Are Adopting Transformer Architectures

Conversely, some world model work has begun using Transformers to replace RSSM's GRU + categorical latent structure. For example, IRIS (*Transformers are Sample-Efficient World Models*) uses a discrete image tokenizer + autoregressive Transformer to build a world model, turning image dynamics into token sequence modeling.

The advantage of this direction is that Transformer attention mechanisms naturally support "attending to key past time steps," without needing to compress all historical information through h_t as in RSSM. Under standard global self-attention, computational/memory costs grow quadratically with sequence length; but practical world models often mitigate this through token compression, local attention, or other structural choices.

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

**First, it demonstrated the practicality of latent imagination for behavior learning.** The Dreamer series, through the complete engineering implementation of RSSM + imagination, proved that the combination of learned latent dynamics with actor-critic learning is viable and even sample-efficient on real robot tasks.

**Second, it provided a reference architecture.** RSSM's dual-track design (deterministic + stochastic) was not an accidental choice, but reflects a deep design principle: **world models need to simultaneously capture predictable dynamics and unpredictable uncertainty.** This principle will not become outdated, even if the specific implementation (GRU, categorical latent) may be superseded.

**Third, it represented and systematized an important design space of "latent dynamics + imagination-based control."** RSSM proved that "performing dynamics prediction in latent space + training policies in imagination" is a viable path. Subsequent work can either replace the dynamics engine within this space (as TD-MPC2 does with MLP dynamics replacing RSSM), or adopt entirely different predictive representations (such as JEPA's latent prediction, video generative models, or diffusion world models).

If one were to make a somewhat metaphorical analogy, RSSM's position in latent-dynamics world models is somewhat analogous to LSTM's position in classical sequence modeling: it may not be the final destination, but it proved that an important structure can work at scale.

## Open Questions

There are several questions that I think do not yet have clear answers.

**What should the latent state be?** RSSM uses categorical latents, TD-MPC2 uses continuous latents, S4/Mamba use deterministic hidden states. Which representation is best suited for world models? There may be no single answer -- different tasks and different embodiments may require different latent state structures.

**Should dynamics models be specialized or general-purpose?** RSSM is a specialized latent dynamics model (designed for environment modeling), while Mamba is a general-purpose sequence model. Which should the sequence engine of a world model be? Hybrid architectures may be the answer, but the optimal combination remains unclear.

**Will scaling change architectural choices?** A hypothesis worth testing is: in regimes where data is limited, observations are complex, or environments are highly stochastic, explicit stochastic latent structure may provide valuable inductive bias; while after data and task scales expand, whether more concise and unified latent dynamics architectures scale more easily requires systematic experimental validation.

**What are the limits of imagination?** RSSM's imagination in DreamerV3 can already rollout very long trajectories. But the quality of imagination degrades with rollout length -- is this a fundamental limitation, or can it be resolved through better architectures?

---

*This article is a complementary perspective to the RSSM series. If you want to see the specific architectural details of RSSM, refer to the [previous RSSM deep dive](/en/articles/rssm-deep-dive/) and the [source code walkthrough](/en/articles/2026-08-19-rssm-code-walkthrough/).*

*The next article is the middle installment of the VLA series -- the Pi Family and the Evolution of Action Interfaces.*
