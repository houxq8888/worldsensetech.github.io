---
title: "Understanding RSSM Through Code (1): Where RSSM Sits in DreamerV3 and the Stochastic State"
slug: "2026-08-19-rssm-code-walkthrough"
date: 2026-08-19
draft: false
categories: ["World Models"]
tags: ["RSSM", "DreamerV3", "World Model", "State Space Model", "Code Walkthrough", "RSSM Series"]
description: "Series opener: where RSSM fits in DreamerV3, why the stochastic state is not 'an integer then one-hot', and how real observations enter RSSM (the observe call hierarchy)."
toc: true
---

> **Understanding RSSM Through Code · 第 1 篇 / 共 6 篇**
>
> Series contents (you are on part 1, bolded; prev/next at the bottom):
> **1. [(1) Where RSSM Sits & the Stochastic State](/en/articles/2026-08-19-rssm-code-walkthrough/)**
> 2. [(2) Prior/Posterior, Straight-Through & unimix](/en/articles/2026-08-20-rssm-stochastic-state/)
> 3. [(3) _core(), deter=8192 & Block GRU](/en/articles/2026-08-21-rssm-deterministic-core/)
> 4. [(4) KL Balancing, Free Nats & Final KL](/en/articles/2026-08-22-rssm-kl-balancing/)
> 5. [(5) Imagine, Observe vs. Imagine & Reset](/en/articles/2026-08-23-rssm-imagine-reset/)
> 6. [(6) Default Config, Four Formulas & the Map](/en/articles/2026-08-24-rssm-recap/)


> **Source Code Reading Guide**
>
> This article intentionally does not start from "standard RSSM pseudocode." Instead, it follows the actual execution path of `rssm.py`. When reading, focus on these functions:
>
> ```text
> observe()   → sequence-level wrapper that calls _observe() along the time axis
> _observe()  → single-timestep state inference (real trajectories)
> _core()     → single-timestep deterministic dynamics
> imagine()   → latent rollout without observation
> ```
>
> The most common confusion: **`observe()` is not a function that processes a single transition.** It is only a sequence-level wrapper that repeatedly calls `_observe()` along the time dimension via Ninjax's `nj.scan(...)`. The code that actually handles a single timestep is `_observe()` and `_core()`. Connecting `observe() / _observe() / _core() / imagine()` gives you the core of DreamerV3's RSSM.
>
> (You'll see how `nj.scan` unrolls the sequence in the "Sequence training" section. For now, remember one sentence: **the first time `observe()` appears, it, `_observe()`, and `nj.scan` are three layers of the same thing.**)

The previous two articles covered RSSM's basic principles and the evolution of world models.

Theory answers "why RSSM is designed this way," but once you open DreamerV3's actual source code, you'll find many details that the paper's formulas don't directly tell you:

* Why isn't the stochastic state a plain Gaussian vector?
* What inputs does the `GRU` actually receive?
* What do prior and posterior correspond to in the code?
* Why does DreamerV3 compute two KL terms?
* What role does `stop_gradient` play in KL balancing?
* How does RSSM continue running during imagination without observations?
* What does `stoch=32, classes=64` actually mean?
* Why is `deter=8192` so large, and how can it still be trained effectively?

Instead of starting from generic Gaussian RSSM pseudocode, this article **directly traces the data flow through `dreamerv3/rssm.py`** in the DreamerV3 repository, mapping computation paths in the source code to RSSM's mathematical formulas.

> **Source Code Note**
>
> This article references `dreamerv3/rssm.py` and `configs.yaml` from the current `main` branch of the DreamerV3 open-source repository. The repository's README describes itself as a reimplementation of DreamerV3, so this article consistently refers to it as the "DreamerV3 open-source implementation" rather than Google/DeepMind's official code.
>
> For readability, JAX, Ninjax, dtype, and `scan` engineering code is appropriately simplified, but the core computation logic follows the source code. Wherever "default configuration" appears, this article explicitly distinguishes **RSSM architecture parameters** from **world-model / agent loss configuration**, so that no agent hyperparameter is misread as an inherent property of the RSSM.

---

## 1. Where RSSM Sits in DreamerV3

DreamerV3's world model can be roughly understood as:

```text
Observation
     │
     ▼
  Encoder
     │
     ▼
 observation token
     │
     ▼
 ┌─────────────────────────────┐
 │            RSSM             │
 │                             │
 │  deterministic state h_t    │
 │            +                │
 │  stochastic state z_t       │
 └─────────────────────────────┘
     │
     ▼
 latent feature
     │
     ├──► Decoder       reconstruct observation
     ├──► Reward Head   predict reward
     ├──► Continue Head
     └──► Actor/Critic  policy and value on imagined trajectories
```

The problem RSSM solves can be summarized in one sentence:

> **Maintain a hidden state that can continuously roll into the future, based on past latent states and actions.**

This state consists of two parts:

```text
s_t = (h_t, z_t)
```

Where:

* `h_t`: deterministic state, responsible for maintaining temporal context;
* `z_t`: stochastic state, representing the random latent in the current state.

An important distinction to make up front (expanded in "Why Is `deter=8192`?" below): **what is fed to the Decoder / Reward / Actor-Critic is not `h_t` or `z_t` alone, but their concatenation `feature = concat(h_t, z_t)`.** So `8192` is NOT the "total latent state dimension."

DreamerV3 has a key difference right here:

> **`z_t` is not a plain vector from a traditional continuous Gaussian RSSM, but multiple categorical latent variables.**

---

## 2. The Most Common Misconception: What Exactly Is `stoch`?

Many RSSM tutorials write:

```text
z_t ~ Normal(μ_t, σ_t)
```

Then sample via:

```text
z = μ + σ × ε
```

This notation helps explain classical continuous RSSMs, but **cannot be directly applied to DreamerV3's implementation**.

DreamerV3 uses a categorical latent.

Default configuration (note: this mixes "RSSM architecture" and "world-model training" parameters; the formal split is in the "RSSM or World-Model Config?" section):

```yaml
rssm:
  deter: 8192
  hidden: 1024
  stoch: 32
  classes: 64
  unimix: 0.01
  blocks: 8
```

So the stochastic state's shape is:

```text
[B, 32, 64]
```

Meaning:

* There are `32` categorical variables;
* Each variable has `64` classes;
* Each variable corresponds, in the forward computation, to a 64-dimensional **one-hot categorical representation**.

Flattened:

```text
32 × 64 = 2048
```

So you can think of the entire stochastic state as a 2048-dimensional vector, but **semantically it cannot be simply understood as a 2048-dimensional plain categorical variable**.

More accurately:

```text
z_t = [z_t^1, z_t^2, ..., z_t^32]
```

Where:

```text
z_t^i ∈ {1, ..., 64}
```

Each `z_t^i` is a 64-class categorical variable, and together they form a **factorized categorical distribution** — not a single 2048-class distribution.

The `_logit()` in the source code does exactly this:

```python
x = Linear(..., self.stoch * self.classes)(x)
return x.reshape(
    x.shape[:-1] + (self.stoch, self.classes)
)
```

That is:

```text
Linear
  │
  ▼
32 × 64 logits
  │
  ▼
[B, 32, 64]
```

---

## 3. Observe: How Does the Real Observation Enter RSSM?

The most important entry point for understanding DreamerV3's RSSM is:

```python
observe(...)
```

But recall the reminder above: the public `observe()` does not process a single transition. Its role is a sequence-level wrapper that repeatedly calls the `_observe()` below along the time axis via `nj.scan`. So when you read the source, the code that "handles a single timestep" is `_observe()`, not `observe()`.

`_observe()`'s core logic simplified:

```python
def _observe(carry, tokens, action, reset, training):

    deter, stoch, action = mask(
        carry["deter"],
        carry["stoch"],
        action,
        ~reset
    )

    action = preprocess_action(action)

    # Key: no observation here
    deter = self._core(
        deter,
        stoch,
        action
    )

    # observation token enters here
    x = concat([deter, tokens])

    logit = posterior_network(x)

    stoch = sample(logit)

    return {
        "deter": deter,
        "stoch": stoch,
    }
```

There's a crucial detail here:

> **`_core()` does not read the current observation.**

The current observation embedding (`tokens`) enters the posterior network only after the deterministic transition is complete.

So the entire process is actually:

```text
z_{t-1} ─────┐
             │
h_{t-1} ─────┼──► RSSM Core ──► h_t
             │
a_{t-1} ─────┘

                         │
                         ▼
                  posterior network
                         ▲
                         │
                      token_t
                         │
                         ▼
                       z_t
```

This is different from the simplified RSSM notation where "GRU inputs = observation + action."

---

