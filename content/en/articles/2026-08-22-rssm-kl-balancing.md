---
title: "Understanding RSSM Through Code (4): KL Balancing, Free Nats, and the Final KL Combination"
slug: "2026-08-22-rssm-kl-balancing"
date: 2026-08-22
draft: false
categories: ["World Models"]
tags: ["RSSM", "DreamerV3", "World Model", "State Space Model", "Code Walkthrough", "RSSM Series"]
description: "The core of training the world model: the complete Observe-phase data flow, the gradient routing of dyn/rep KLs, free_nats as a loss floor, and how the final KL loss is combined."
toc: true
---

> **Understanding RSSM Through Code · 第 4 篇 / 共 6 篇**
>
> Series contents (you are on part 4, bolded; prev/next at the bottom):
> 1. [(1) Where RSSM Sits & the Stochastic State](/en/articles/2026-08-19-rssm-code-walkthrough/)
> 2. [(2) Prior/Posterior, Straight-Through & unimix](/en/articles/2026-08-20-rssm-stochastic-state/)
> 3. [(3) _core(), deter=8192 & Block GRU](/en/articles/2026-08-21-rssm-deterministic-core/)
> **4. [(4) KL Balancing, Free Nats & Final KL](/en/articles/2026-08-22-rssm-kl-balancing/)**
> 5. [(5) Imagine, Observe vs. Imagine & Reset](/en/articles/2026-08-23-rssm-imagine-reset/)
> 6. [(6) Default Config, Four Formulas & the Map](/en/articles/2026-08-24-rssm-recap/)

## 14. Complete Observe Phase Data Flow (Observe / Imagine in One Framework)

We can now draw the RSSM state-transition framework and put **Observe and Imagine inside the same `_core()` transition framework**:

```text
                 ┌──────────────────────────┐
                 │      RSSM transition     │
                 │                          │
(h_{t-1},z_{t-1}) + a_{t-1}
                 │
                 ▼
              _core()
                 │
                 ▼
                h_t
                 │
          ┌──────┴──────┐
          │             │
          ▼             ▼
       Prior        Observation
   p(z_t | h_t)         │
          │             ▼
          │         Posterior
          │      q(z_t|h_t,o_t)
          │             │
          │             ▼
          │            z_t
          └───────┬─────┘
                  │
                  ▼
              latent state
                  │
                  ▼
               next step
```

With the two usages annotated on the side — these are essentially the two most important sentences of the whole article:

```text
OBSERVE:  z_t ← posterior(h_t, o_t)
IMAGINE:  z_t ← prior(h_t)
```

The complete Observe data flow takes the `OBSERVE` branch in the framework above (observation enters the posterior); Imagine takes the `IMAGINE` branch (no observation, use the prior directly). This diagram is essentially the core of DreamerV3's RSSM.

---

## 15. KL Balancing: Why Does the Source Code Compute Two KL Terms? (Focus on Gradient Routing)

This is the most easily glossed-over but critically important part of DreamerV3 RSSM.

```python
dyn = self._dist(sg(post)).kl(self._dist(prior))
rep = self._dist(post).kl(self._dist(sg(prior)))
```

### 1. Dynamics KL

```text
L_dyn = KL[sg(q(z_t|h_t,o_t)) || p(z_t|h_t)]
```

`sg(post)` means posterior is stop-gradiented. This loss primarily:

> **Makes the prior / dynamics model learn to predict (approximate) the posterior.**

That is, **the extra information from the observation acts as a teacher, but gradient does not flow back through the posterior** — the prior is pulled toward the posterior's distribution.

```text
posterior = teacher → target → prior learns (gradient only updates prior / dynamics)
```

---

## 16. Representation KL

```text
L_rep = KL[q(z_t|h_t,o_t) || sg(p(z_t|h_t))]
```

Now `sg(prior)` means the prior is the fixed target. This loss:

> **Keeps the posterior's representation from becoming completely disconnected from the learned dynamics (the already-trained prior).**

An important precision point: the sentence above is intuitively correct but not precise enough. The more essential understanding is — **the two KLs have different gradient routing**:

```text
L_dyn:  sg(q) || p      → gradient mainly trains dynamics / prior
L_rep:  q || sg(p)      → gradient mainly trains representation / posterior
```

That is, **the real point of KL balancing is NOT the vague phrase "constrain the posterior from drifting infinitely far from the prior," but: `stop_gradient` splits the optimization target into two different gradient paths — one trains dynamics (prior chases posterior), the other trains representation (posterior stays connected to dynamics).** This is the core of KL balancing.

---

## 17. Why Can't We Just Write a Single KL?

If we simply wrote `kl = KL(post || prior)`, both networks would receive gradients simultaneously.

But DreamerV3 wants to separate the two roles:

```text
Dynamics KL
  sg(q) ──► p        trains dynamics / prior
            (posterior as teacher, no gradient back)

Representation KL
  q ──► sg(p)         trains representation / posterior
            (prior as fixed target)
```

**The key insight isn't "there are two KLs" but that `stop_gradient` splits the optimization targets into two directions.**

---

## 18. What Does Free Nats Actually Do?

```python
if self.free_nats:
    dyn = jnp.maximum(dyn, self.free_nats)
    rep = jnp.maximum(rep, self.free_nats)
```

Default `free_nats: 1.0`: `L_dyn = max(L_dyn, 1.0)`, `L_rep = max(L_rep, 1.0)`

An important correction:

> **Don't interpret DreamerV3's `free_nats` as "each stochastic dimension must retain at least 1 nat of information."**

The source code doesn't set an individual information floor per categorical variable. It applies `maximum(kl, free_nats)` directly on the KL tensor from `_dist(...).kl(...)`.

**More precisely (numerical form only): it does not make KL below 1 become 0; it floors the loss at 1:**

```text
L' = max(L, 1.0)
```

> **Strict limit:** The above only states that `free_nats` is a **loss floor** (applying `max(kl, free_nats)` on the KL tensor), not a per-latent-dimension information floor. From this `max` form alone, one must not conclude training-effect claims such as "no gradient when KL < 1" — unless the specific JAX implementation and how the KL tensor is subsequently aggregated, scaled, and combined into the final loss have been confirmed. This article makes no further inference.

Therefore:

```text
KL = 0.2  →  loss = 1.0
KL = 0.8  →  loss = 1.0
KL = 1.5  →  loss = 1.5
```

This is worth stating clearly, because the term "free nats" easily makes readers auto-apply definitions from other world-model implementations — for example, many implementations use:

```text
max(KL - τ, 0)
```

This is **NOT the same form** as DreamerV3's `max(KL, 1)`. DreamerV3 "raises the lower bound of KL to 1," rather than "subtracting a threshold then taking max with 0."

---

## 19. How Is the Final KL Loss Combined?

Default configuration (note: this belongs to the **agent's loss scale configuration**, not the RSSM architecture itself):

```yaml
loss_scales:
  dyn: 1.0
  rep: 0.1
```

So the RSSM's KL part can be written as:

```text
L_KL = 1.0 × L_dyn + 0.1 × L_rep
```

Note (very important; this distinction should be kept throughout the article):

> `1.0` and `0.1` are not hardcoded in the RSSM class — they are the **agent's loss scale configuration** (part of world-model / agent training config), and should not be counted as an inherent property of the RSSM architecture.

So if emphasizing "source-code-level analysis," it's better to distinguish:

```text
RSSM architecture (rssm.py)
    ├── compute dyn KL
    ├── compute rep KL
    └── free_nats (inside RSSM)

World model / agent loss config (configs.yaml)
    ├── dyn_scale  = 1.0
    └── rep_scale  = 0.1
```

This way readers won't mistake `0.1` as a fixed formula of the RSSM, nor treat "default RSSM config" as one mixed table.

---

