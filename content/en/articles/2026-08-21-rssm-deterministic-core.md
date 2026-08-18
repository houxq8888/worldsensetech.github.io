---
title: "Understanding RSSM Through Code (3): Deterministic Transition _core(), deter=8192, and Block GRU"
slug: "2026-08-21-rssm-deterministic-core"
date: 2026-08-21
draft: false
categories: ["World Models"]
tags: ["RSSM", "DreamerV3", "World Model", "State Space Model", "Code Walkthrough", "RSSM Series"]
description: "Trace the real deterministic dynamics: how _core() does one-step transition, why deter is 8192, the block-wise parameterization of Block GRU, and what the posterior/prior networks do."
toc: true
---

> **Understanding RSSM Through Code · 第 3 篇 / 共 6 篇**
>
> Series contents (you are on part 3, bolded; prev/next at the bottom):
> 1. [(1) Where RSSM Sits & the Stochastic State](/en/articles/2026-08-19-rssm-code-walkthrough/)
> 2. [(2) Prior/Posterior, Straight-Through & unimix](/en/articles/2026-08-20-rssm-stochastic-state/)
> **3. [(3) _core(), deter=8192 & Block GRU](/en/articles/2026-08-21-rssm-deterministic-core/)**
> 4. [(4) KL Balancing, Free Nats & Final KL](/en/articles/2026-08-22-rssm-kl-balancing/)
> 5. [(5) Imagine, Observe vs. Imagine & Reset](/en/articles/2026-08-23-rssm-imagine-reset/)
> 6. [(6) Default Config, Four Formulas & the Map](/en/articles/2026-08-24-rssm-recap/)

## 9. The Real Deterministic Transition: `_core()`

This is the most worthwhile part of `rssm.py` to read.

The source code doesn't simply call `nn.GRUCell(...)`. Instead, it constructs a block-wise GRU.

Core inputs: `deter`, `stoch`, `action`.

First, stoch is flattened:

```text
[B, 32, 64] → [B, 2048]
```

Then each of `deter`, `stoch`, `action` is passed through its own input projection (note: "separate Linear then concat" is a conceptual sketch — **the real core is the structured parameterization of BlockLinear**, see next section):

```text
deter ──► Linear ──┐
                   │
stoch ──► Linear ──┼──► concat ──► Block GRU
                   │
action ─► Linear ──┘
```

Note: **No observation here.**

This again confirms the deterministic transition core: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`

---

## 10. Why Is `deter=8192`?

The default `deter: 8192` looks very large at first glance.

But DreamerV3 doesn't use a plain 8192-dimensional dense GRU. It also has `blocks: 8`.

The deterministic state is split into 8 blocks: `8192 / 8 = 1024` per block.

```text
8192
 │
 ├── block 1: 1024
 ├── block 2: 1024
 ├── ...
 └── block 8: 1024
```

Then `nn.BlockLinear` transforms these blocks. This is the **Block GRU**.

Its core goal: **maintain the representational power of a large deterministic state while avoiding the huge computation and parameter cost of a full dense GRU.**

Why is this worth doing specially? A scale intuition:

A plain GRU's recurrent projection, if applied directly to an 8192-dimensional hidden state with a dense transformation, would have an enormous parameter count (the recurrent weight alone is on the order of `8192 × 8192`). **BlockLinear organizes this large linear transformation into a block-wise parameter structure**, thereby avoiding a fully dense `8192 × 8192` recurrent transformation. A caution: one must not infer from the class name alone that the blocks are mathematically completely independent — whether (and how) blocks interact should be confirmed against the actual `nn.BlockLinear` implementation. This article only describes the engineering intent ("use a structured block parameterization instead of a fully dense transformation"), and avoids over-inferring the exact connection structure between blocks.

### A Dimension Distinction Easily Missed

This clarifies an important conceptual gap in the article: `8192` is NOT the "total latent state dimension."

| state | role |
| :--- | :--- |
| `deter` (8192) | long-term temporal memory / deterministic dynamics |
| `stoch` (32×64 = 2048) | current-state uncertainty / observation-conditioned information |
| `feature = concat(deter, stoch)` | used by Decoder, Reward, Actor-Critic |

In other words, the latent feature the model ultimately uses is `concat(h_t, z_t)`, whose dimension is `8192 + 2048`, not simply `8192`. `deter` is large to give deterministic dynamics enough memory capacity; `stoch` retains the uncertainty information brought by observations.

Writing the dimensions explicitly:

```text
dim(h_t)        = 8192
dim(z_t)        = 32 × 64 = 2048
dim(feature_t)  = 8192 + 2048 = 10240
```

i.e.:

```text
deter   = 8192
stoch   = 2048
feature = 10240
```

This is actually an important bridge for understanding the inputs to DreamerV3's later decoder / actor / critic: what they receive is not `h_t` or `z_t` alone, but the 10240-dimensional concatenated feature.

---

## 11. What Is Block GRU Actually Computing? (Variable Names ≠ Standard GRU)

The source code ultimately:

```python
x = BlockLinear(...)(x)
gates = split(x, 3)
reset, cand, update = gates

reset = sigmoid(reset)
cand = tanh(reset * cand)
update = sigmoid(update - 1)

deter = update * cand + (1 - update) * deter
```

**A confusing point must be clarified first:**

> The `reset`, `cand`, `update` in the code above are **implementation-internal intermediate variable / chunk names**. They do **NOT** directly equal the textbook GRU's `reset gate / update gate / candidate`. The implementation first obtains several gate chunks through a projection, then updates the deterministic state following the GRU's gating structure. When reading this code, do not mistake it for a direct rewriting of the "standard GRU formula."

If we map these internal chunks to the standard GRU gating structure, the core can still be abstracted into the standard GRU's three parts `r, z, h̃`:

```text
r_t   = σ(W_r · x_t + U_r · h_{t-1} + b_r)   [reset gate]
z_t   = σ(W_z · x_t + U_z · h_{t-1} + b_z)   [update gate]
h̃_t   = tanh(W_h · x_t + U_h · (r_t ⊙ h_{t-1}) + b_h)   [candidate]
h_t   = z_t ⊙ h̃_t + (1 - z_t) ⊙ h_{t-1}
```

The "source ↔ math" correspondence:

* Source chunk `update` ≈ standard GRU update gate `z_t` (controls "how much old state to keep");
* Source chunk `reset` ≈ standard GRU reset gate `r_t` (controls "how much old state enters the candidate");
* Source chunk `cand` + interaction with `reset` ≈ standard GRU candidate `h̃_t`.

So the conclusion remains: **Block GRU is semantically a GRU**, just with:

* Inputs through independent projections;
* Hidden transformation using block structure (BlockLinear);
* Gate computation also using block-wise transformation;
* Internal variable naming is an engineering detail — don't rigidly map it one-to-one to textbook symbols.

---

## 12. What Does the Posterior Network Specifically Do?

Back to `_observe()`. After getting the deterministic state:

```python
x = tokens if self.absolute else concat([deter, tokens])
```

Default: `absolute: False`, so by default: `x_t = [h_t, o_t^emb]`

**Why does this switch deserve a separate note?**

It essentially decides how the observation encoder token participates in the posterior input (whether the posterior input is constructed in an "absolute/relative" way). What is discussed here is the current default configuration `absolute=False` execution path: the posterior input is `concat([deter, tokens])`. **If this config is changed, with `absolute=True` the posterior input becomes `tokens` alone, and the construction changes.** An important clarification: here `absolute` is merely an RSSM-internal config switch that controls how the posterior input is constructed — **it is NOT the Transformer-style absolute positional encoding**. Readers should not associate this name with positional encodings. Therefore, `x_t = [h_t, o_t^emb]` should not be taken as the mathematical definition of the RSSM itself — it is just the execution path under the default config branch.

Then through `obslayers` MLP layers (default: `obslayers: 1, hidden: 1024`).

Finally: `logit = self._logit('obslogit', x)` → `[B, 32, 64]`

```text
deter_t + observation_token_t
              │
              ▼
         obs network
              │
              ▼
           logits → categorical q(z_t)
```

---

## 13. What About the Prior Network?

The prior code is actually very simple:

```python
def _prior(self, feat):
    x = feat
    for i in range(self.imglayers):
        x = Linear(hidden)(x)
        x = activation(norm(x))
    return self._logit('priorlogit', x)
```

Input is only `deter_t`. Default `imglayers: 2`.

```text
h_t → MLP (2 layers) → 32 × 64 logits → p(z_t | h_t)
```

Again confirming: the prior does not read observations.

---

