---
title: "Understanding RSSM Through Code (2): Prior/Posterior, Straight-Through Sampling, and unimix"
slug: "2026-08-20-rssm-stochastic-state"
date: 2026-08-20
draft: false
categories: ["World Models"]
tags: ["RSSM", "DreamerV3", "World Model", "State Space Model", "Code Walkthrough", "RSSM Series"]
description: "Translate the code into math with correct time indexing; understand the prior/posterior pair, straight-through categorical sampling, and how unimix adds a probability floor per class."
toc: true
---

> **Understanding RSSM Through Code · 第 2 篇 / 共 6 篇**
>
> Series contents (you are on part 2, bolded; prev/next at the bottom):
> 1. [(1) Where RSSM Sits & the Stochastic State](/en/articles/2026-08-19-rssm-code-walkthrough/)
> **2. [(2) Prior/Posterior, Straight-Through & unimix](/en/articles/2026-08-20-rssm-stochastic-state/)**
> 3. [(3) _core(), deter=8192 & Block GRU](/en/articles/2026-08-21-rssm-deterministic-core/)
> 4. [(4) KL Balancing, Free Nats & Final KL](/en/articles/2026-08-22-rssm-kl-balancing/)
> 5. [(5) Imagine, Observe vs. Imagine & Reset](/en/articles/2026-08-23-rssm-imagine-reset/)
> 6. [(6) Default Config, Four Formulas & the Map](/en/articles/2026-08-24-rssm-recap/)

## 4. Translating the Source Code into Mathematical Formulas (Watch the Time Index)

DreamerV3's recurrence relation:

```text
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
```

Then compute separately:

```text
p(z_t | h_t)      [prior, no observation]
q(z_t | h_t, o_t) [posterior, with observation]
```

Where `p` is the prior, `q` is the posterior, `o_t` is the current observation, and `h_t` is the deterministic state.

**A time-index note that is critical but where beginners often get stuck:**

> In `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`, the `h_t` already fuses the latent/action history up to `t-1`, but **it does not contain the current `o_t`**. The current observation only enters the **posterior branch** to infer `z_t`; it never enters the deterministic transition.

In other words, beginners often ask: "Since `h_t` is the current state, why doesn't it contain `o_t`?" The answer: `h_t` here is the "current deterministic memory derived from history `(h_{t-1}, z_{t-1}, a_{t-1})`"; `o_t` is a separate stream of information at the same timestep, used only by the posterior to correct `z_t`, and never fed into `_core()`. Understanding this index is the key to reading Dreamer's RSSM.

The entire RSSM can be understood as:

```text
┌─────────────────────────────────────────────────┐
│  (h_{t-1}, z_{t-1}, a_{t-1}) → h_t →           │
│                                    ┌──────────┐  │
│                                    │ p(z_t|h_t)│  │
│                                    │ q(z_t|h_t,│  │
│                                    │   o_t)    │  │
│                                    └──────────┘  │
└─────────────────────────────────────────────────┘
```

This is also the key to understanding imagination and KL loss later.

---

## 5. Why Two Distributions — Prior and Posterior?

These two distributions actually solve two different problems.

### 1. Posterior: After seeing the observation, what do I think the current state is?

```text
q(z_t | h_t, o_t)
```

It can see both historical information `h_t` and the current observation `o_t`, so it has more information.

During training on real trajectories, we use the posterior to get the stochastic state:

```python
logit = posterior(deter, token)
stoch = sample(logit)
```

The posterior can be understood as:

> **"After seeing the real world, the estimate of the current latent state."**

### 2. Prior: If there were no observation, what do I predict the state would be?

```text
p(z_t | h_t)
```

It only sees `h_t`, so it doesn't know the current real observation.

It expresses:

> **"Based only on historical state and action, what I predict the next latent state will be."**

This is exactly the capability needed during imagination.

---

## 6. How Is Categorical Latent Sampled? (Straight-Through)

Both posterior and prior ultimately output logits:

```text
logits
[B, 32, 64]
```

Each `[64]` corresponds to a categorical distribution for the i-th categorical variable (64 class probabilities).

In the source code:

```python
def _dist(self, logits):
    out = embodied.jax.outs.OneHot(
        logits,
        self.unimix
    )
    out = embodied.jax.outs.Agg(
        out, 1, jnp.sum
    )
    return out
```

This is not a Gaussian distribution, but a one-hot categorical distribution.

A precise description of the sampling is needed here, to avoid a common misunderstanding:

> **`z_t` is NOT "first sample an integer `[B, 32]`, then convert to one-hot."**

More accurately:

* Each stochastic variable corresponds to a 64-class categorical distribution;
* The sampled stochastic state participates in downstream networks in the forward pass as a **one-hot categorical representation** (a one-hot-like representation);
* The implementation uses a **straight-through estimator** so that discrete sampling can participate in backpropagation — i.e., the forward pass uses the discrete one-hot sample, while the backward pass approximates the gradient back to the network parameters that produced the logits.

So in the forward computation `z_t` is a one-hot-like tensor of shape `[B, 32, 64]`. It "looks like one-hot," but its gradient path is straight-through, not the "round to integer then one-hot" operation that would completely cut the gradient.

```text
logits
  │
  ▼
categorical distribution (each [64] a 64-class dist)
  │
  ▼
sample  ── forward:  one-hot-like representation
  │        backward: straight-through estimator
  ▼
stoch.shape = [B, 32, 64]
```

---

## 7. What Does `unimix=0.01` Do?

DreamerV3's categorical distribution has another easily overlooked parameter:

```yaml
unimix: 0.01
```

Its purpose is to keep a small portion of uniform distribution in the categorical distribution.

If the network is already very confident:

```text
class 7: 0.9999
other:   0.0001
```

The distribution becomes very sharp.

`unimix` mixes it slightly with a uniform distribution, ensuring every class retains some probability. Written as a formula:

```text
U_i = 1 / K            (K = number of classes, here K = 64)
p'_i = (1 - ε) × p_i + ε × U_i
     = (1 - ε) × p_i + ε / 64
```

Where `ε = 0.01`.

So it adds a small probability floor `ε/K` to **every single class**, rather than just "blunting the distribution overall."

Its role has two layers:

1. Improving categorical latent training stability, preventing the distribution from becoming too sharp too early;
2. Providing numerical stability for categorical KL / entropy computations (avoiding log(0) and division-by-zero caused by zero probabilities).

---

## 8. Why Does DreamerV3 Make the Stochastic State So Many Categorical Variables?

This is an important design in DreamerV3's latent representation.

Default: `stoch=32, classes=64`

Not `z ∈ R^32`, but:

```text
z = [
    categorical(64),
    categorical(64),
    ...
    categorical(64)
]
        × 32
```

An important benefit is forming a very rich discrete combinatorial space.

From the perspective of the combinatorial space, 32 variables with 64 classes each correspond to:

```text
64^32
```

discrete combinations.

Mathematically, this factorized categorical distribution is the product of independent per-variable distributions:

```text
p(z_t | h_t) = Π_{i=1}^{32} p(z_t^i | h_t)
```

That is, the joint distribution is the product of 32 independent categorical distributions.

**An important qualification is needed:**

> `64^32` describes the size of the combinatorial space that this factorized categorical distribution "covers." It does **NOT** mean the model explicitly enumerates `64^32` states. The model merely parameterizes this space using `32 × 64` logits (a factorized categorical distribution); it does not maintain a lookup table or state set of size `64^32`.

So a more precise statement is: "this structure provides combinatorial representational capacity on the order of `64^32`," not "the model uses a giant discrete state space with `64^32` explicit states." This is why DreamerV3 can express complex environment states in a relatively compact latent space.

---

