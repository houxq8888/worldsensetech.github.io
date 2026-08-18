---
title: "Understanding RSSM Through Code (5): Imagine, Observe vs. Imagine, Sequence Training, and Reset"
slug: "2026-08-23-rssm-imagine-reset"
date: 2026-08-23
draft: false
categories: ["World Models"]
tags: ["RSSM", "DreamerV3", "World Model", "State Space Model", "Code Walkthrough", "RSSM Series"]
description: "Future simulation without observation: the full imagine loop, the essential difference between Observe and Imagine, why imagination cannot be infinitely long, and the roles of sequence training and reset."
toc: true
---

> **Understanding RSSM Through Code · 第 5 篇 / 共 6 篇**
>
> Series contents (you are on part 5, bolded; prev/next at the bottom):
> 1. [(1) Where RSSM Sits & the Stochastic State](/en/articles/2026-08-19-rssm-code-walkthrough/)
> 2. [(2) Prior/Posterior, Straight-Through & unimix](/en/articles/2026-08-20-rssm-stochastic-state/)
> 3. [(3) _core(), deter=8192 & Block GRU](/en/articles/2026-08-21-rssm-deterministic-core/)
> 4. [(4) KL Balancing, Free Nats & Final KL](/en/articles/2026-08-22-rssm-kl-balancing/)
> **5. [(5) Imagine, Observe vs. Imagine & Reset](/en/articles/2026-08-23-rssm-imagine-reset/)**
> 6. [(6) Default Config, Four Formulas & the Map](/en/articles/2026-08-24-rssm-recap/)

## 20. Imagine: How Does RSSM Run Without Observations?

This is the most elegant part of RSSM.

During imagination, there are no real observations, so we use `p(z_t|h_t)` — the prior:

```text
z_t ~ p(z_t | h_t)
```

Inside `imagine()`, it also first runs `_core()` then uses the prior to produce the latent:

```python
deter = self._core(carry['deter'], carry['stoch'], actemb)
logit = self._prior(deter)
stoch = self._dist(logit).sample(...)
```

> **There is completely no observation here, and no zero-filling of observations.**

Because `_core()` doesn't need observations in the first place.

But writing only a single step is not enough — **the real core of imagination is a closed loop**: the actor produces an action, the RSSM takes one transition, the prior produces the next latent, then the actor produces a new action… and the loop continues to roll out.

The complete loop:

```text
z_t ~ p(z_t | h_t)
  │
  ▼
actor → a_t                         (policy outputs action from current feature)
  │
  ▼
h_{t+1} = f(h_t, z_t, a_t)          (_core advances deterministic state)
  │
  ▼
z_{t+1} ~ p(z_{t+1} | h_{t+1})      (prior predicts next latent)
  │
  ▼
actor → a_{t+1} → ...               (loop continues)
```

A more compact loop diagram:

```text
(h_t, z_t)
    │
    ├── actor → a_t
    │
    ▼
core(h_t, z_t, a_t)
    │
    ▼
h_{t+1}
    │
    ▼
prior
    │
    ▼
z_{t+1}
    │
    └──► back to actor, loop continues
```

**Actor produces action → RSSM transition → prior produces latent → produces action again.** If the article's theme is RSSM, this closed loop is the heart of Dreamer's "latent imagination": the world model "dreams" on its own, and the policy tries and errs inside the dream.

---

## 21. Observe vs. Imagine

Putting the two paths side by side makes it clear (they share the same `_core()` transition; the only difference is whether `z_t` comes from the posterior or the prior).

### Observe

```text
previous state + action → _core → h_t
                                    ├──► prior (also computed, for KL)
                                    │
                                    ▼
                              observation token
                                    │
                                    ▼
                               posterior → z_t   ← posterior(h_t, o_t)
```

### Imagine

```text
previous state + action → _core → h_t → prior → z_t   ← prior(h_t)
```

> **Observe is "updating state while watching reality"; Imagine is "predicting state with eyes closed based on the model."**

---

## 22. Why Can't Imagination Be Infinitely Long?

Because imagination uses `p(z_t|h_t)` instead of `q(z_t|h_t,o_t)`, every prediction step is affected by model error.

```text
prediction error → next step input → new prediction error → continues accumulating
```

This is typical rollout error accumulation.

> **Attribution matters:** `imag_length: 15` is an **agent imagination rollout training config**, not a structural parameter of the RSSM. It belongs to the world-model / agent training hyperparameters, deciding how many steps the policy unrolls on the imagined trajectory — it is not the same layer of concept as RSSM architecture parameters like `deter / stoch / classes / blocks`.

Not that the world model suddenly fails after 15 steps, but an engineering trade-off: too short limits planning ability, too long causes severe error accumulation.

---

## 23. Sequence Training: How Can RSSM Process Entire Sequences?

Back to the reminder at the top: `observe()` does not process just a single timestep — it is only a sequence-level wrapper.

The source code uses Ninjax's `nj.scan(...)` to unroll RSSM along the time dimension — **this is exactly the `observe() → (via `nj.scan`) → _observe()` hierarchy that should have been connected the first time `observe()` appeared.**

State continuously passes along the sequence:

```text
t=0 → (h0, z0) → t=1 → (h1, z1) → t=2 → (h2, z2) → ...
```

The temporal recurrence still exists: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`. `scan` is just the JAX/Ninjax engineering implementation for more efficiently expressing this recurrence.

---

## 24. Why Is Reset Important?

RSSM has memory. If an episode ends without reset, Episode B's initial state would carry Episode A's history — state contamination.

The source code masks `deter`, `stoch`, `action` based on `reset` at the beginning of `_observe()`.

> **Reset essentially cuts off the recurrent state between episodes — telling the model: "A new world begins; don't bring memories from the last episode."**

---

