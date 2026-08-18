---
title: "Understanding RSSM Through Code (6): Default Config, Four Formulas, and the Code↔Math↔Semantics Map"
slug: "2026-08-24-rssm-recap"
date: 2026-08-24
draft: false
categories: ["World Models"]
tags: ["RSSM", "DreamerV3", "World Model", "State Space Model", "Code Walkthrough", "RSSM Series"]
description: "Wrap-up: split the config into architecture vs. training tables, compress RSSM into four core formulas, present the code↔math↔semantics mapping table, and re-understand RSSM as Memory / State estimation / Prediction."
toc: true
---

> **Understanding RSSM Through Code · 第 6 篇 / 共 6 篇**
>
> Series contents (you are on part 6, bolded; prev/next at the bottom):
> 1. [(1) Where RSSM Sits & the Stochastic State](/en/articles/2026-08-19-rssm-code-walkthrough/)
> 2. [(2) Prior/Posterior, Straight-Through & unimix](/en/articles/2026-08-20-rssm-stochastic-state/)
> 3. [(3) _core(), deter=8192 & Block GRU](/en/articles/2026-08-21-rssm-deterministic-core/)
> 4. [(4) KL Balancing, Free Nats & Final KL](/en/articles/2026-08-22-rssm-kl-balancing/)
> 5. [(5) Imagine, Observe vs. Imagine & Reset](/en/articles/2026-08-23-rssm-imagine-reset/)
> **6. [(6) Default Config, Four Formulas & the Map](/en/articles/2026-08-24-rssm-recap/)**

## 25. Default Config: Split into "RSSM Architecture" and "World-Model Training"

To avoid mixing "RSSM architecture parameters" with "world-model / agent training config," the configuration is split into two tables below.

### RSSM architecture

| Parameter | Default | Meaning |
| :--- | ---: | :--- |
| `deter` | 8192 | deterministic state dimension (split by blocks) |
| `stoch` | 32 | number of categorical variables |
| `classes` | 64 | classes per categorical variable |
| `blocks` | 8 | Block GRU group count (8192 / 8 = 1024) |
| `hidden` | 1024 | RSSM internal MLP hidden dimension |
| `imglayers` | 2 | prior network layers |
| `obslayers` | 1 | posterior network layers |
| `dynlayers` | 1 | dynamics network layers |

### RSSM / world-model training

| Parameter | Default | Meaning | Attribution |
| :--- | ---: | :--- | :--- |
| `unimix` | 0.01 | uniform mixing ratio of categorical distribution | inside RSSM distribution |
| `free_nats` | 1.0 | KL loss floor (`max(KL, 1)`) | inside RSSM |
| `dyn_scale` | 1.0 | weight of dynamics KL in final loss | agent loss scale |
| `rep_scale` | 0.1 | weight of representation KL in final loss | agent loss scale |
| `imag_length` | 15 | imagination rollout training steps | agent training config (not RSSM structure) |

**Key points:**

* `deter / stoch / classes / blocks` are the RSSM **architecture**;
* `unimix / free_nats` are used inside the RSSM to construct distributions and KL;
* `dyn_scale / rep_scale / imag_length` are **agent / world-model training config**, not part of the RSSM structure itself.

These are **default configurations**, not fixed parameters for all DreamerV3 model scales. Different model scales (1M to 400M) change `deter` from 512 to 12288 and `classes` from 4 to 96.

> `8192 / 32 / 64` should be understood as a specific instance under DreamerV3's default configuration, not RSSM's fixed structure.

---

## 26. Compressing the Entire RSSM Into Four Formulas

### ① Deterministic transition

```text
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})    (_core(), no observation)
```

Historical state and action determine the deterministic state.

### ② Prior

```text
p(z_t | h_t)
```

Depends only on deterministic state. Handles imagination.

### ③ Posterior

```text
q(z_t | h_t, o_t)
```

Uses observation to correct the latent state. Handles state inference on real trajectories.

### ④ KL balancing

```text
L_dyn = KL[sg(q(z_t|h_t,o_t)) || p(z_t|h_t)]
L_rep = KL[q(z_t|h_t,o_t) || sg(p(z_t|h_t))]
L_KL  = dyn_scale × L_dyn + rep_scale × L_rep
```

KL also has `free_nats = max(KL, 1)` applied before entering the final loss; `dyn_scale / rep_scale` come from the agent's loss config, not from inside the RSSM class.

---

## 27. Code ↔ Math ↔ Semantics Mapping Table

To tie together the previous twenty-some sections, here is a table aligning "source symbol / math symbol / semantics":

| Source concept | Math | Semantics |
| :--- | :--- | :--- |
| `deter` | `h_t` | deterministic memory (long-term temporal memory) |
| `stoch` | `z_t` | stochastic latent (current-state uncertainty) |
| `_core()` | `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})` | latent dynamics (deterministic transition) |
| `prior` | `p(z_t | h_t)` | imagination prediction (no observation) |
| `posterior` | `q(z_t | h_t, o_t)` | observation-conditioned inference |
| `observe()` | filtering | state inference on real trajectories (via `nj.scan` → `_observe`) |
| `imagine()` | latent rollout | future simulation without observation (prior loop) |
| `dyn loss` | `KL[sg(q) || p]` | train dynamics / prior |
| `rep loss` | `KL[q || sg(p)]` | train representation / posterior |
| `feature` | `concat(h_t, z_t)` | representation fed to decoder / reward / actor-critic |

This table is the "index" of the whole article: any source symbol can be traced here to its mathematical meaning and role.

---

## 28. Re-Understanding RSSM From Code

Without looking at code, RSSM is easily understood as "a GRU + a Gaussian."

But DreamerV3's actual implementation is far richer:

```text
                 ┌──────────────────────┐
                 │      RSSM            │
 action ────────►│    Block GRU         │
 z_{t-1} ───────►│        │             │
 h_{t-1} ───────►│       h_t            │
                 │      /   \            │
                 │  Prior   Posterior    │
                 │    │        ▲         │
                 │    │      token_t     │
                 │    ▼        ▼         │
                 │   p(z)     q(z)       │
                 │     └── KL ──┘        │
                 └──────────────────────┘
```

From the perspective of DreamerV3's implementation, RSSM can be understood as **simultaneously playing three functional roles**:

**Memory**: `deter` preserves history through recurrent dynamics.
**State estimation**: posterior corrects current latent state using observations.
**Prediction**: prior predicts future latent states without observations.

> **Note:** The above "RSSM plays three roles" is a **functional generalization**, not a strict definition of the RSSM. Precisely, it should be phrased as "from the … implementation perspective, it can be understood as …," to avoid taking an implementation-level functional division as the mathematical definition of the RSSM.

> **During training, RSSM leverages real observations to learn latent dynamics; during imagination, it drops observations and relies solely on the prior to rollout forward in latent space.**

This is the foundation enabling DreamerV3's latent imagination.

---

## 29. Back to Source Code: Why Do These Details Matter?

From the paper alone, RSSM might be understood as `RNN + latent distribution`. But reading the code reveals extensive engineering:

```text
Categorical latent + OneHot distribution + Unimix (p' = (1-ε)p + ε/K)
+ Block GRU (BlockLinear structured parameterization) + RMSNorm
+ Prior / Posterior + KL balancing (gradient routing: sg(q)||p and q||sg(p))
+ Stop Gradient + Free Nats (max(KL, 1)) + Scan (observe → nj.scan → _observe)
```

Individually none are complex. What matters is how they form a complete pipeline:

```text
real observation → Encoder → observation token → RSSM → latent state
→ imagination → future latent states → reward / value / policy
```

From a code perspective, the real problem RSSM solves isn't "how to predict the next observation."

It's:

> **What RSSM truly solves is not how to predict the next observation, but how to learn a latent state that can both be corrected by observations and, without observations, roll autonomously into the future on its own.**

This is the core of DreamerV3's RSSM.

---

## Source Code References

* [DreamerV3 GitHub Repository](https://github.com/danijar/dreamerv3)
* [dreamerv3/rssm.py](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py)
* [dreamerv3/configs.yaml](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml)
