---
title: "From RSSM to Modern Latent Dynamics: How the 'Engine' of World Models Evolves"
slug: "2026-09-04-rssm-beyond"
date: 2026-09-04
draft: false
categories: ["World Models", "Paper Analysis"]
tags: ["RSSM", "State-Space Model", "TD-MPC", "Mamba", "DreamerV3", "World Model", "Latent Dynamics"]
description: "RSSM is the core engine of the Dreamer family of world models, but the landscape of state-space modeling has changed significantly in recent years. This article places RSSM within the broader evolution of state-space modeling -- distinguishing between latent dynamics and sequence backbone layers, discussing TD-MPC2's decoder-free latent world model approach and planning + value fusion, and exploring the design trend from 'generating the world' to 'providing predictive interfaces' for world model engines."
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

The **deterministic path h_t** is recursively updated by h_t = f(h_{t-1}, z_{t-1}, a_{t-1}), responsible for accumulating historical information. The **stochastic path z_t** models latent state uncertainty through categorical prior/posterior. Together, they constitute RSSM's latent state s_t = (h_t, z_t), providing conditions for prediction heads such as the observation model. From a POMDP perspective, this latent state can be understood as a parameterized approximation of the belief state formed from historical observations and actions; it does not require recovering the environment's "true physical state," but rather learning a latent representation sufficient to support prediction and control.

This design has several noteworthy characteristics:

**First, the categorical latent is an engineering-driven choice.** An engineering advantage of the categorical latent is that both prior and posterior are explicit discrete distributions, so KL divergence can be computed analytically; at the same time, it provides a discrete stochastic representation, allowing prior/posterior to perform KL computation directly on categorical distributions. DreamerV2 introduced the categorical stochastic latent combined with a straight-through estimator; DreamerV3 continued this representation. This is a pragmatic design decision, not a theoretically optimal solution.

**Second, Dreamer's core training mechanism is combining learned latent dynamics with actor-critic learning, enabling policy/value to be trained primarily on imagined latent trajectories.** The concept of imagination / model rollout in model-based RL predates Dreamer. Dreamer's innovation lies in: RSSM is not only used to fit historical data, but also to "imagine" future trajectories in latent space -- starting from the current posterior, using the prior to rollout multiple future paths, and then training the actor and critic on these imagined trajectories. This has made the Dreamer series one of the most representative works in the latent imagination + actor-critic line, demonstrating strong sample efficiency across multiple visual control and reinforcement learning benchmarks.

These design choices were progressively validated through the evolution from DreamerV1 → V2 → V3. But are they the only possible path?

## Another Line in Sequence Modeling: From S4 to Mamba

During the same period as RSSM's development, a parallel line of state-space models emerged in the NLP and sequence modeling domains. **It is important to emphasize: S4/Mamba are first and foremost sequence models, not world models.** They address efficient sequence processing, not action-conditioned environment dynamics.

**S4 (Structured State Spaces, 2021/2022)** introduced a structured-parameterized continuous-time state-space model. Its core formulation is the classical linear SSM:

```
h'(t) = A h(t) + B x(t)
y(t) = C h(t) + D x(t)
```

The key to S4 lies in structured parameterization of the state matrix A, combined with HiPPO initialization and low-rank correction, enabling stable and efficient computation of long-range convolutions. S4 demonstrated competitive performance on long-sequence benchmarks, translating long-sequence computation into more efficient convolutional/state-space operations through structured SSMs.

**Mamba (2023/2024)** built upon S4 by introducing **selectivity** -- making the SSM's key parameters (B_t, C_t, Δ_t) input-dependent, thereby obtaining a **selective state space**. This means that state transition / information retention can vary based on the current token content, enabling the model to selectively remember or forget information. Mamba demonstrated competitive performance with same-scale Transformers on the language modeling experiments reported in its paper, while exhibiting linear scaling in sequence length and efficient hardware execution through selective scan.

### RSSM vs S4/Mamba: Not "Representing the World vs Representing Context"

What is the relationship between these two lines and RSSM?

**Formally similar, but with different goals and training interfaces.** Both RSSM and S4/Mamba use a "latent state + state transition" framework, but RSSM explicitly defines action-conditioned latent transitions and prediction models related to observation/reward, so its latent state is trained to support environment prediction and control. S4/Mamba, on the other hand, are first and foremost a general-purpose sequence architecture; their hidden states do not inherently possess fixed semantics of "context" or "world state" -- rather, this is determined by the specific training objective.

Therefore, a more accurate distinction is not "RSSM represents the world, Mamba represents context," but rather:

```
RSSM:
  latent state + action-conditioned transition
  → environment prediction / imagination / control

S4/Mamba:
  recurrent/SSM hidden state
  → sequence processing
```

Mamba's hidden state can serve as efficient sequence history compression, but it does not inherently presuppose that this state carries "context" or "world state" semantics. If Mamba is trained as an action-conditioned latent dynamics model, it too can become a world-model engine; conversely, RSSM's recurrent state can also be viewed as a sequence state representation, except that it is further situated within action-conditioned latent dynamics and predictive objectives. **The architecture itself does not determine semantics; the training interface does.**

## TD-MPC2: An Alternative Latent Dynamics Approach

TD-MPC2 (2024, arXiv:2310.16828) is a model-based RL system whose core world model adopts decoder-free latent dynamics.

TD-MPC2 does not use RSSM's dual-track structure, but instead employs a more concise architecture:

```
Encoder:     e_t = E(o_t)                    → encode observations into latent
Dynamics:    z_{t+1} = f_θ(z_t, a_t)         → deterministic latent transition
Reward:      r_t = R(z_t, a_t)               → predict reward
Q-function:  Q(z_t, a_t)                     → long-term value estimation (Q ensemble)
Policy:      π(a_t | z_t)                    → policy prior
```

**No deterministic/stochastic dual track, no categorical latent, no KL balancing.** It uses a more direct approach: the encoder maps observations to latent space, performs dynamics prediction in latent space, and then combines short-horizon MPC with long-term Q-value estimation to select actions.

### Decoder-free: From "Generating the World" to "Serving Control"

A very important divergence emerges here.

The Dreamer approach of constraining latent representations through observation / reward prediction tasks is:

```
observation → latent → dynamics → predict observation / reward / continuation
                                        ↓
                                   imagination
```

The TD-MPC2 approach is:

```
observation → encoder → z_t → latent dynamics → ẑ_{t+1}
                                │                    ↕
                          ┌──────┼──────        consistency
                          ↓      ↓      ↓         loss
                        reward   Q    policy    ↕
                                          encoder(o_{t+1})
```

**Unlike Dreamer's approach of constraining latent representations through observation / reward prediction tasks, TD-MPC2 explicitly adopts a decoder-free implicit world model -- here "implicit" means it does not define a pixel-generating world model through an explicit observation decoder, but instead constrains latent space through task-relevant objectives such as latent dynamics, reward/value prediction.**

### A Deeper Perspective: Decision-sufficient vs Observation-sufficient

This distinction deserves further elaboration.

```
Generative world model (Dreamer)

z_t
 ↓
p(o_{t+1:t+H} | z_t, a_{t:t+H})
 ↓
future observations
 → imagination → actor-critic


Control-oriented world model (TD-MPC2)

z_t
 ↓
f(z_t, a_t)
 ↓
z_{t+1}
 ↓
reward / value / terminal
 ↓
action selection (MPC)
```

In typical generative world models, the latent needs to retain sufficient information to support the generation or prediction of future observations; the latter only requires the latent to retain information useful for decision-making. The representation pressures are not the same. TD-MPC2 optimizes for a **decision-sufficient representation**, not an observation-sufficient representation -- which explains why it can forgo a decoder.

From the functional perspective adopted in this article, this can be understood as a design trend from "generating the world" toward "providing predictive interfaces." This is a design trend, not a field-wide consensus -- generative and control-oriented lines clearly coexist. But it raises a core question: **a world model does not necessarily need to become a more powerful "video generator." The more critical question is: what kind of action-conditioned predictive interface does it need to provide to support planning, value estimation, or policy learning at the lowest computational and data cost?**

### TD-MPC2's Design Priorities

The core of TD-MPC2 is not a complex latent-state decomposition, but rather the combination of **concise latent dynamics, task-conditioned representation, short-horizon MPC, and long-horizon Q-value estimation.** The design priorities of TD-MPC2 can be summarized in three aspects:

**First, the combination of latent-space MPC and Q-function ensemble.** TD-MPC2's explicit MPC planning horizon is very short (default 3 steps), so it does not complete long-term planning through extended rollouts; instead, it lets the learned Q-function provide long-term value bootstrap at the planning boundary. **In other words, TD-MPC2 transforms "a portion of long-horizon planning problems" from a model rollout problem into a value estimation problem.** Specifically, TD-MPC2 performs short-horizon rollouts on latent dynamics and uses a Q-function ensemble (defaulting to 5 Q-functions, with TD target using the minimum of a randomly subsampled Q-function) to provide long-term value estimation, thereby bridging short-horizon MPC planning with long-horizon TD bootstrapping.

**Second, task-conditioned multi-task and cross-embodiment scaling.** TD-MPC2 was evaluated on 104 continuous control tasks, and further demonstrated that a single 317M-parameter agent can be trained on 80 tasks, covering different tasks, embodiments, and action spaces. This is not "one dynamics model automatically understanding all embodiments," but rather using **task embeddings / task-conditioned components** to adapt the same set of models to different tasks -- the encoder, dynamics, reward, policy prior, and Q components are all linked to task embeddings.

**Third, stabilization of latent representations.** TD-MPC2 uses SimNorm to normalize latent states, and jointly trains the encoder, dynamics, reward, policy prior, and Q-functions, so that the latent representation serves both prediction and control -- without requiring the stochastic prior/posterior KL constraints of RSSM.

From RSSM to TD-MPC2, a clear trend emerges: **TD-MPC2 demonstrates an alternative path: instead of relying on complex stochastic recurrent states, it combines dynamics prediction, short-horizon MPC, and long-horizon value estimation within a compact latent space.**

## Architecture ≠ Function: From Sequence Models to World Models

Placing the models discussed above together reveals that they do not operate at the same level. More importantly, **the same architecture can occupy different functional roles.** Rather than describing a "four-layer architecture," a two-dimensional taxonomy is more accurate:

**Horizontal axis: architecture** (Recurrent / SSM / Transformer / MLP)

**Vertical axis: functional role** (sequence backbone → latent dynamics → prediction interface → planning / policy)

| Architecture | Sequence backbone | Latent dynamics | Planning / policy |
| ------------ | ----------------- | --------------- | ----------------- |
| GRU | ✓ | RSSM | can serve as policy/value backbone |
| MLP | — | TD-MPC2 | can serve as policy/value network |
| Transformer | ✓ | IRIS / other WMs | can serve as policy / planner |
| Mamba | ✓ | can be constructed | can serve as policy backbone |

> **The same architecture can occupy different functional layers; what truly determines whether it constitutes a world model is the training objective, input interface, and what predictive interface it can provide.**

This table expresses the article's core point: **Architecture ≠ function.** RSSM uses GRU for latent dynamics; Mamba can equally serve as a latent dynamics backbone -- the difference lies not in the architecture, but in the training interface and functional role.

Therefore, "world model" is better understood as a functional interface rather than a fixed network structure: it needs to provide at least some queryable prediction interface related to the future evolution of the environment that can be consulted by decision processes, such as predictions of future latent states, observations, rewards, or termination; value functions can further serve as a downstream bootstrap mechanism for this predictive model. From this perspective, RSSM, TD-MPC2, Transformer world models, and even certain JEPA-style predictive models can all belong to the world-model family, but the prediction interfaces they provide are not the same.

### Comparison Table

| Dimension | RSSM-based Dreamer | S4 / Mamba | TD-MPC2 |
|------|-------------------|------------|---------|
| **World-model role** | explicit latent dynamics | not a world model per se | implicit latent dynamics |
| **State Structure** | deterministic + stochastic latent | SSM hidden state | continuous latent + task conditioning |
| **Action-conditioned** | Yes | Not by default | Yes |
| **Observation Decoder** | typically present in Dreamer | depends on task | none (decoder-free) |
| **Core Prediction Interface** | observation / reward + latent transition | depends on training objective | latent transition + reward/value |
| **Control Method** | imagined actor-critic | not a control algorithm itself | latent MPC + Q bootstrap |
| **Key Strengths** | imagination / sample efficiency | long sequence efficiency | planning + value + multitask scaling |
| **Key Limitations** | stochastic latent / training complexity | does not naturally provide dynamics semantics | short MPC horizon + Q bootstrap |

## Directions of Convergence: What Kind of Sequence Architecture Do World Models Need?

A natural question arises: will these lines converge?

Looking at recent work, there are several convergence trends worth noting.

### Trend 1: VLAs Are Borrowing SSM Architectures

The VLA domain has already seen work directly adopting Mamba/SSM backbones. RoboMamba (NeurIPS 2024) is one representative case -- it combines visual encoding with Mamba for vision-language-action reasoning, validating its efficiency in both simulation and real robot experiments; recent work has also begun further exploring the use of selective SSMs for VLA action experts.

The motivation is straightforward: under standard full self-attention, the attention computation during training scales as O(L²) with context length; while autoregressive inference can avoid recomputing the entire attention at each step through KV cache, the KV cache memory still grows with context length. For robot policies that need to continuously process long observation histories, this creates significant computational and memory pressure.

But from a modeling perspective, this connects precisely to the problem discussed in the VLA series: what VLAs lack is not sequence processing capability, but an explicit action-conditioned prediction interface. Placing Mamba inside a VLA can improve sequence processing efficiency, but it will not automatically turn a VLA into a world model.

### Trend 2: World Models Are Adopting Transformer Architectures

Conversely, some world model work has begun using Transformers to replace RSSM's GRU + categorical latent structure. For example, IRIS (*Transformers are Sample-Efficient World Models*) uses a discrete image tokenizer + autoregressive Transformer to build a world model. IRIS does not simply replace RSSM's recurrent state with a Transformer; rather, it changes both the representation and the dynamics simultaneously: first compressing images into discrete tokens, then modeling dynamics over the token sequence with an autoregressive Transformer -- a fundamentally different world-model design. IRIS was primarily validated on Atari environments, so it serves better as a representative of the "Transformer world model" architectural line rather than a direct robot world-model benchmark.

The advantage of this direction is that Transformer's explicit attention provides direct interaction capability across temporal positions, allowing the model to dynamically utilize information from different historical positions based on the current prediction target, without needing to compress history into a single recurrent state of fixed dimension as in RSSM. Under standard global self-attention, attention computation during training scales as O(L²) with context length; but practical world models often mitigate this through token compression, local attention, or other structural choices.

### Trend 3: Hybrid Architectures of Latent Dynamics + Foundation Models

An even more interesting direction is: **using foundation models for semantic/visual representation, with a dedicated latent dynamics module handling action-conditioned future prediction.**

```
Foundation Model (Transformer / Mamba)
      ↓
  semantic / visual representation
      ↓
Latent Dynamics Model (RSSM-style / TD-MPC-style)
      ↓
  action-conditioned future prediction
      ↓
  planning / policy
```

This hybrid architecture attempts to let foundation models provide stronger semantic/visual representations, with a dedicated latent dynamics module handling action-conditioned future prediction; but how the two interface, and what information needs to be preserved from the foundation model into the dynamics latent, remain open questions. A representation well-suited for semantic understanding is not necessarily a dynamics state well-suited for action-conditioned prediction -- this is precisely the core challenge that hybrid architectures must address.

## The Legacy of RSSM

Returning to the original question: where does RSSM stand in the evolution of state-space modeling?

I think RSSM's contributions can be summarized at three levels.

**First, it demonstrated the practicality of latent imagination for behavior learning.** The Dreamer series, through the complete engineering implementation of RSSM + imagination, proved that the combination of learned latent dynamics with actor-critic learning can achieve high sample efficiency across a variety of visual control and reinforcement learning tasks.

**Second, it provided a reference architecture.** RSSM's dual-track design embodies an important modeling principle: **for partially observable environments or those with multimodal futures, explicitly representing uncertainty is often valuable.** This design dimension will not become outdated, even if the specific implementation (GRU, categorical latent) may be superseded.

**Third, it represented and systematized an important design space of "latent dynamics + imagination-based control."** RSSM proved that "performing dynamics prediction in latent space + training policies in imagination" is a viable path. Subsequent work can either replace the dynamics engine within this space (as TD-MPC2 does with MLP dynamics replacing RSSM), or adopt entirely different predictive representations (such as JEPA's latent prediction, video generative models, or diffusion world models).

If one were to make a somewhat metaphorical analogy, RSSM is more like a **canonical reference architecture** in latent-dynamics world models: it may not be the final form, but it concretized the entire paradigm of "stochastic latent + recurrent state + learned dynamics + imagination" and proved its engineering feasibility.

## Open Questions

There are several questions that I think do not yet have clear answers.

**What should the latent state be?** RSSM uses categorical latents, TD-MPC2 uses continuous latents, S4/Mamba use deterministic hidden states. Which representation is best suited for world models? There may be no single answer -- different tasks and different embodiments may require different latent state structures.

**Should dynamics models be specialized or general-purpose?** RSSM is a specialized latent dynamics model (designed for environment modeling), while Mamba is a general-purpose sequence model. Which should the sequence engine of a world model be? Hybrid architectures may be the answer, but the optimal combination remains unclear.

**Will scaling change architectural choices?** A hypothesis worth testing is: in regimes where data is limited, observations are complex, or environments are highly stochastic, explicit stochastic latent structure may provide valuable inductive bias; while after data and task scales expand, whether more concise and unified latent dynamics architectures exhibit better scaling efficiency under large-scale data and multi-task training requires systematic experimental validation.

**What are the limits of imagination?** DreamerV3 does not solve long-horizon decision-making through a single infinitely extended latent rollout; its default imagination horizon is 15 steps, with long-term returns primarily propagated through value bootstrap. Interestingly, TD-MPC2 adopts a similar strategy -- with a default MPC horizon of only 3 steps, also handling long-term effects beyond the explicit planning horizon through Q-function bootstrap. **Neither has attempted to solve long-horizon decision-making solely by infinitely extending latent rollouts; both use finite-horizon model-based computation, with value functions bootstrapping longer-range return information back.** The specific implementations differ: Dreamer performs actor-critic learning on imagination trajectories, while TD-MPC2 uses Q-functions at the boundary of short-horizon MPC to provide long-term value estimation. Therefore, a more accurate question is: **to what extent can finite-horizon imagination/planning + value bootstrap reliably solve long-horizon tasks? If the rollout horizon is further increased, at what rate does model error accumulate?**

---

*This article is a complementary perspective to the RSSM series. If you want to see the specific architectural details of RSSM, refer to the [previous RSSM deep dive](/en/articles/rssm-deep-dive/) and the [source code walkthrough](/en/articles/2026-08-19-rssm-code-walkthrough/).*

*The next article is the middle installment of the VLA series -- the Pi Family and the Evolution of Action Interfaces.*
