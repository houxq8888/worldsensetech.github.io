---
title: "Understanding RSSM Through Code: Implementation Details in DreamerV3"
slug: "2026-08-19-rssm-code-walkthrough"
aliases:
  - /en/articles/2026-08-19-rssm-code-walkthrough.html
date: 2026-08-19
draft: false
categories: ["World Models"]
tags: ["RSSM", "DreamerV3", "World Model", "State Space Model", "Code Walkthrough"]
description: "Tracing the data flow through DreamerV3's open-source rssm.py: categorical latent, Block GRU, dual KL balancing (dyn + rep), observe vs. imagine."
toc: true
---

> **Source Code Reading Guide**
>
> This article intentionally does not start from "standard RSSM pseudocode." Instead, it follows the actual execution path of `rssm.py`. When reading, focus on these three key functions:
>
> ```text
> observe()  → state inference on real trajectories
> _core()    → deterministic dynamics
> imagine()  → latent rollout without observation
> ```
>
> Connecting these three functions gives you the core of DreamerV3's RSSM.

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
> For readability, JAX, Ninjax, dtype, and `scan` engineering code is appropriately simplified, but the core computation logic follows the source code.

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

Default configuration:

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
* Each variable ultimately corresponds to a 64-dimensional one-hot vector.

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

Each `z_t^i` is a 64-class categorical variable.

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

The core logic simplified:

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

## 4. Translating the Source Code into Mathematical Formulas

DreamerV3's recurrence relation:

```text
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
```

Then compute separately:

```text
p(z_t | h_t)    [prior]
q(z_t | h_t, o_t)  [posterior]
```

Where `p` is the prior, `q` is the posterior, `o_t` is the current observation, and `h_t` is the deterministic state.

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

## 6. How Is Categorical Latent Sampled?

Both posterior and prior ultimately output logits:

```text
logits
[B, 32, 64]
```

Each `[64]` corresponds to a categorical distribution.

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

The sampling process:

```text
logits
  │
  ▼
categorical distribution
  │
  ▼
sample
  │
  ▼
one-hot vector
```

Result: `stoch.shape = [B, 32, 64]`

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

`unimix` mixes it slightly with a uniform distribution, ensuring every class retains some probability:

```text
p' = (1 - ε) × p + ε × U
```

Where `ε = 0.01`. Its main role is improving categorical latent training stability, preventing the distribution from becoming too sharp too early.

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

Theoretically, 32 variables with 64 classes each can form `64^32` combinations.

Of course, the actual model won't utilize all combinations, but this structure provides enormous representational capacity. This is why DreamerV3 can express complex environment states in a relatively compact latent space.

---

## 9. The Real Deterministic Transition: `_core()`

This is the most worthwhile part of `rssm.py` to read.

The source code doesn't simply call `nn.GRUCell(...)`. Instead, it constructs a block-wise GRU.

Core inputs: `deter`, `stoch`, `action`.

First, stoch is flattened:

```text
[B, 32, 64] → [B, 2048]
```

Then three separate input projections:

```python
x0 = Linear(hidden)(deter)
x1 = Linear(hidden)(stoch)
x2 = Linear(hidden)(action)
```

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

---

## 11. What Is Block GRU Actually Computing?

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

In mathematical form:

```text
r_t = σ(W_r × x_t + b_r)
h̃_t = tanh(r_t ⊙ W_h × x_t)
u_t = σ(W_u × x_t - 1)
h_t = u_t ⊙ h̃_t + (1 - u_t) ⊙ h_{t-1}
```

It's essentially still a GRU, just with:

* Inputs through independent projections;
* Hidden transformation using block structure;
* Gate computation also using block-wise transformation.

---

## 12. What Does the Posterior Network Specifically Do?

Back to `_observe()`. After getting the deterministic state:

```python
x = tokens if self.absolute else concat([deter, tokens])
```

Default: `absolute: False`, so by default: `x_t = [h_t, o_t^emb]`

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

---

## 14. Complete Observe Phase Data Flow

```text
                 observation_t
                       │
                       ▼
                    Encoder
                       │
                       ▼
                     token_t
                       │
z_{t-1} ───────┐       │
               │       │
h_{t-1} ───────┼───────┼──► RSSM Core
               │       │        ▲
a_{t-1} ───────┘       │        │
                       ▼
                      h_t
                       │
                ┌──────┴──────┐
                ▼             ▼
             Prior        Posterior
          p(z_t|h_t)   q(z_t|h_t,o_t)
                │             │
                │             ▼ → sample → z_t
                └────── KL ───┘
```

---

## 15. KL Balancing: Why Does the Source Code Compute Two KL Terms?

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

> **Trains the prior / dynamics model to approximate the posterior.**

```text
posterior = teacher → target → prior learns
```

---

## 16. Representation KL

```text
L_rep = KL[q(z_t|h_t,o_t) || sg(p(z_t|h_t))]
```

Now `sg(prior)` means prior is the fixed target. This loss:

> **Constrains the posterior's representation from drifting infinitely far from the prior.**

The two KL directions aren't mathematically "redundant computation" — `stop_gradient` **explicitly specifies the optimization direction**.

---

## 17. Why Can't We Just Write a Single KL?

If we simply wrote `kl = KL(post || prior)`, both networks would receive gradients simultaneously.

But DreamerV3 wants to separate the two roles:

```text
Dynamics KL:       posterior ──► prior    (train dynamics)
Representation KL: prior ──► posterior    (constrain representation)
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

---

## 19. How Is the Final KL Loss Combined?

Default config: `loss_scales: dyn=1.0, rep=0.1`

```text
L_KL = 1.0 × L_dyn + 0.1 × L_rep
```

Note: `1.0` and `0.1` are not hardcoded in the RSSM class — they're the agent's loss scale configuration.

```text
rssm.py
    ├── compute dyn KL
    ├── compute rep KL
    └── free_nats

configs.yaml / agent loss scale
    ├── dyn = 1.0
    └── rep = 0.1
```

---

## 20. Imagine: How Does RSSM Run Without Observations?

This is the most elegant part of RSSM.

During imagination, there are no real observations, so we use `p(z_t|h_t)` — the prior.

```python
deter = self._core(carry['deter'], carry['stoch'], actemb)
logit = self._prior(deter)
stoch = self._dist(logit).sample(...)
```

> **There is completely no observation here, and no zero-filling of observations.**

Because `_core()` doesn't need observations in the first place.

```text
h_{t-1}, z_{t-1}, a_{t-1} → RSSM Core → h_t → Prior → z_t → continue rollout
```

This is where the world model truly starts "dreaming."

---

## 21. Observe vs. Imagine

### Observe

```text
previous state + action → _core → h_t → posterior (with observation token) → z_t
                                    └──► prior (also computed)
```

### Imagine

```text
previous state + action → _core → h_t → prior → z_t
```

> **Observe is "updating state while watching reality"; Imagine is "predicting state with eyes closed based on the model."**

---

## 22. Why Can't Imagination Be Infinitely Long?

Because imagination uses `p(z_t|h_t)` instead of `q(z_t|h_t,o_t)`, every prediction step is affected by model error.

```text
prediction error → next step input → new prediction error → continues accumulating
```

This is typical rollout error accumulation.

Default `imag_length: 15` — not that the world model suddenly fails after 15 steps, but an engineering trade-off: too short limits planning ability, too long causes severe error accumulation.

---

## 23. Sequence Training: How Can RSSM Process Entire Sequences?

`observe()` doesn't process just a single timestep. The source code uses Ninjax's `nj.scan(...)` to unroll RSSM along the time dimension.

State continuously passes along the sequence. The temporal recurrence still exists: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`. `scan` is just the JAX/Ninjax engineering implementation for more efficiently expressing this recurrence.

---

## 24. Why Is Reset Important?

RSSM has memory. If an episode ends without reset, Episode B's initial state would carry Episode A's history — state contamination.

The source code masks `deter`, `stoch`, `action` based on `reset` at the beginning of `_observe()`.

> **Reset essentially tells the model: "A new world begins; don't bring memories from the last episode."**

---

## 25. Default RSSM Configuration

| Parameter | Default | Meaning |
|:---|---:|:---|
| `deter` | 8192 | deterministic state dimension |
| `hidden` | 1024 | RSSM MLP hidden dimension |
| `stoch` | 32 | number of categorical variables |
| `classes` | 64 | classes per categorical variable |
| `unimix` | 0.01 | uniform mixing ratio |
| `blocks` | 8 | Block GRU group count |
| `free_nats` | 1.0 | KL free-nats threshold |
| `imglayers` | 2 | prior network layers |
| `obslayers` | 1 | posterior network layers |
| `dynlayers` | 1 | dynamics network layers |

These are **default configurations**, not fixed parameters for all DreamerV3 model scales. Different model scales (1M to 400M) change `deter` from 512 to 12288 and `classes` from 4 to 96.

> `8192 × 32 × 64` should be understood as a specific instance under DreamerV3's default configuration, not RSSM's fixed structure.

---

## 26. Compressing the Entire RSSM Into Four Formulas

### ① Deterministic transition

```text
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
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
L_KL  = 1.0 × L_dyn + 0.1 × L_rep
```

KL also has `free_nats` applied before entering the final loss.

---

## 27. Re-Understanding RSSM From Code

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

RSSM actually accomplishes three things:

**Memory**: `deter` preserves history through recurrent dynamics.
**State estimation**: posterior corrects current latent state using observations.
**Prediction**: prior predicts future latent states without observations.

> **During training, RSSM leverages real observations to learn latent dynamics; during imagination, it drops observations and relies solely on the prior to rollout forward in latent space.**

This is the foundation enabling DreamerV3's latent imagination.

---

## 28. Back to Source Code: Why Do These Details Matter?

From the paper alone, RSSM might be understood as `RNN + latent distribution`. But reading the code reveals extensive engineering:

```text
Categorical latent + OneHot distribution + Unimix + Block GRU + RMSNorm
+ Prior / Posterior + KL balancing + Stop Gradient + Free Nats + Scan
```

Individually none are complex. What matters is how they form a complete pipeline:

```text
real observation → Encoder → observation token → RSSM → latent state
→ imagination → future latent states → reward / value / policy
```

From a code perspective, the real problem RSSM solves isn't "how to predict the next observation."

It's:

> **How to learn a latent state that can both explain real observations and, without observations, roll into the future using only historical state and actions.**

This is the core of DreamerV3's RSSM.

---

## Source Code References

* [DreamerV3 GitHub Repository](https://github.com/danijar/dreamerv3)
* [dreamerv3/rssm.py](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py)
* [dreamerv3/configs.yaml](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml)

> **Version note:** If keeping this article as a long-term technical blog post, it's recommended to record a specific commit hash at publication time rather than just referencing `main`, as the source code and default configurations may continue to evolve.
