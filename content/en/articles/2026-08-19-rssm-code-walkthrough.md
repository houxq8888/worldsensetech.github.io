---
title: "Understanding RSSM Through Code: Implementation Details in DreamerV3"
slug: "2026-08-19-rssm-code-walkthrough"
aliases:
  - /en/articles/2026-08-19-rssm-code-walkthrough.html
date: 2026-08-19
draft: false
categories: ["World Models"]
tags: ["RSSM", "DreamerV3", "World Model", "State Space Model", "Code Walkthrough"]
description: "A code-level walkthrough of DreamerV3's official RSSM implementation: categorical latent, Block GRU, dual KL balancing, observe vs. imagine."
toc: true
---

The previous two articles covered RSSM's basic principles and the evolution of world models. Theory answers "why RSSM is designed this way," but when you actually read the DreamerV3 source code, you'll find many remaining questions:

* Why isn't the stochastic state a plain Gaussian vector?
* Does the GRU consume the observation, or the previous timestep's latent state?
* What do prior and posterior correspond to in the code?
* Why does DreamerV3 compute KL twice?
* What role does `stop_gradient` play in KL balancing?
* How does RSSM run during imagination when there's no observation?
* What does `stoch=32, classes=64` actually mean?

Instead of starting from a generic Gaussian RSSM pseudocode, this article **starts directly from DreamerV3's official implementation**, mapping the computation paths in the source code to RSSM's mathematical formulas one by one.

> This article references `dreamerv3/rssm.py` and the default configuration from the official DreamerV3 repository. For readability, JAX, Ninjax, scan, dtype, and other engineering code is appropriately simplified below, but the core computation logic is preserved.

---

## 1. Where RSSM Sits in DreamerV3

DreamerV3 can be roughly understood as:

```text
Observation
    │
    ▼
 Encoder
    │
    ▼
 observation embedding
    │
    ▼
 ┌──────────────────────────────┐
 │             RSSM             │
 │                              │
 │  deterministic state h_t     │
 │          +                   │
 │  stochastic state z_t        │
 └──────────────────────────────┘
    │
    ▼
 latent feature
    │
    ├──► Decoder: reconstruct observation
    ├──► Reward Head: predict reward
    ├──► Continue Head: predict episode continuation
    └──► Actor / Critic: policy and value on imagined trajectories
```

RSSM is the core of the entire world model.

The problem it solves can be summarized in one sentence:

> **Maintain a hidden state that can continuously roll into the future, based on historical latent states and actions.**

This hidden state consists of two parts:

```text
s_t = (h_t, z_t)

h_t: deterministic state
z_t: stochastic state
```

Where:

* `h_t` is responsible for maintaining long-term temporal context;
* `z_t` is responsible for representing the stochastic part of the current state.

In DreamerV3's official code, RSSM's state space is:

```python
deter = [B, deter]
stoch = [B, stoch, classes]
```

In other words, `stoch` **is not a plain 1D Gaussian vector**, but multiple categorical random variables.

---

## 2. What Exactly Is DreamerV3's Stochastic State?

This is the most important step in understanding DreamerV3's RSSM.

Many RSSM tutorials write:

```text
z_t ~ Normal(μ_t, σ_t)
```

And then:

```python
z = mean + std * eps
```

This notation works for explaining some classical continuous RSSMs, but **cannot be directly used as DreamerV3's implementation**.

DreamerV3 uses a categorical latent.

In the default configuration:

```yaml
rssm:
  deter: 8192
  hidden: 1024
  stoch: 32
  classes: 64
```

Therefore:

```text
stoch = 32
classes = 64
```

Means:

> At each timestep, there are 32 categorical random variables, each with 64 classes.

So the stochastic state's shape is:

```text
[B, 32, 64]
```

Not:

```text
[B, 32]
```

If flattened:

```text
32 × 64 = 2048
```

So the stochastic state, before entering the decoder and other modules, can be understood as a 2048-dimensional one-hot / straight-through representation.

The official source code definition is also straightforward:

```python
@property
def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))
```

In other words, RSSM's complete latent state is:

```text
deter:  [B, 8192]
stoch:  [B, 32, 64]
```

The 8192 here is the default large model configuration, not a fixed value for RSSM.

DreamerV3 provides multiple model scales, for example:

```text
1M   → deter 512
12M  → deter 2048
25M  → deter 3072
50M  → deter 4096
100M → deter 6144
200M → deter 8192
400M → deter 12288
```

Therefore, when discussing DreamerV3, a more accurate statement would be:

> `deter` and `classes` vary with model scale, while `stoch=32` remains constant in the default configuration.

---

## 3. Observe: What Happens When Real Observations Enter RSSM?

The best way to understand DreamerV3's RSSM is to directly trace `observe()`.

Simplified, its core flow can be written as:

```python
def observe(carry, token, action, reset):

    deter = core(
        carry["deter"],
        carry["stoch"],
        action
    )

    logit = posterior(
        deter,
        token
    )

    stoch = sample(logit)

    return {
        "deter": deter,
        "stoch": stoch,
    }
```

There's a very important point here:

**The GRU does not directly read the current observation embedding.**

What it reads is:

```text
Previous timestep deter
Previous timestep stoch
Current action
```

Then produces:

```text
Current deter
```

That is:

```text
(h_{t-1}, z_{t-1}, a_{t-1})
            │
            ▼
          GRU
            │
            ▼
           h_t
```

Only then is the current observation embedding used for inference:

```text
q(z_t | h_t, o_t)
```

This is precisely the division of labor between deterministic transition and stochastic inference in RSSM.

---

## 4. RSSM's True Recurrence Relation

Translating the code into mathematical formulas:

### 1. Deterministic transition

```text
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
```

In DreamerV3, `f` is a GRU-like transition.

Note:

```text
Not:

h_t = GRU(h_{t-1}, observation_t, action_t)

But:

h_t = GRU(h_{t-1}, z_{t-1}, action_{t-1})
```

This is a critical point for understanding RSSM.

### 2. Posterior

After obtaining the current observation encoder output `o_t`:

```text
q(z_t | h_t, o_t)
```

In DreamerV3's code, the concatenation of:

```text
h_t + observation embedding
```

is fed into the observation model, which produces:

```text
logits
```

The logits are then converted into a categorical distribution.

### 3. Prior

On the other hand, using only the deterministic state:

```text
p(z_t | h_t)
```

That is:

```python
prior_logit = prior(deter)
```

So the same `h_t` corresponds to two distributions:

```text
                 ┌──► posterior q(z_t | h_t, o_t)
h_t ─────────────┤
                 └──► prior     p(z_t | h_t)
```

This is the core of RSSM.

---

## 5. Why Two Distributions — Prior and Posterior?

We can understand this from the "training" and "imagination" phases.

### Training Phase

During training, we have real observations:

```text
o_1, o_2, o_3, ...
```

So we can use:

```text
q(z_t | h_t, o_t)
```

That is, the posterior.

It has seen the real observation, so it can obtain a more accurate latent state.

### Imagination Phase

During Dreamer's imagination:

```text
No more observations
```

We only have:

```text
Current latent state
+
action
```

So we can no longer compute:

```text
q(z_t | h_t, o_t)
```

We can only use:

```text
p(z_t | h_t)
```

That is, the prior.

Therefore:

```text
Training:

observation
    ↓
posterior
    ↓
z_t


Imagination:

h_t
 ↓
prior
 ↓
z_t
```

One of the training objectives is to make the prior gradually learn to approximate the posterior.

This enables the model to:

> Learn world states using real observations during training, and still predict the future without observations during imagination.

This is the foundation that allows world models to perform imagination.

---

## 6. Why Does DreamerV3 Use a Categorical Latent?

Now let's look at `_logit()`:

```python
def _logit(self, name, x):
    x = Linear(
        self.stoch * self.classes
    )(x)

    return x.reshape(
        x.shape[:-1],
        self.stoch,
        self.classes
    )
```

Assuming:

```text
stoch = 32
classes = 64
```

The Linear layer ultimately outputs:

```text
32 × 64 = 2048
```

Then reshapes to:

```text
[32, 64]
```

So each stochastic variable has a 64-class categorical distribution.

You can think of it as:

```text
z_1 → 64 classes
z_2 → 64 classes
z_3 → 64 classes
...
z_32 → 64 classes
```

Ultimately:

```text
z = [z_1, z_2, ..., z_32]
```

---

## 7. How Is Categorical Sampling Done?

In the source code:

```python
def _dist(self, logits):
    out = embodied.jax.outs.OneHot(
        logits,
        self.unimix
    )
    out = embodied.jax.outs.Agg(
        out,
        1,
        jnp.sum
    )
    return out
```

The `OneHot` here is crucial.

It is not:

```text
Gaussian sampling
```

But rather:

```text
Categorical distribution
→ sample one category
→ represent it as one-hot
```

For example, a categorical variable:

```text
[0.05, 0.10, 0.70, 0.15]
```

After sampling might become:

```text
[0, 0, 1, 0]
```

After independently sampling 32 variables:

```text
[32, 64]
```

That's the current stochastic state.

---

## 8. Why Is There a `unimix` in the Code?

DreamerV3 defaults to:

```yaml
unimix: 0.01
```

Its effect can be simply understood as:

> Preventing the categorical distribution from becoming too sharp too early.

Suppose the model predicts:

```text
[0.999, 0.001, 0, 0, ...]
```

This kind of distribution could cause some class probabilities to rapidly approach 0.

unimix mixes the distribution with a uniform distribution, ensuring every class retains a minimum probability.

Intuitively:

```text
Original distribution
       │
       ▼
  + small amount of uniform
       │
       ▼
Smoother categorical distribution
```

The `0.01` here is the default mixing ratio.

---

## 9. DreamerV3's GRU Is Not a Standard GRUCell

If you're used to PyTorch:

```python
nn.GRUCell(...)
```

Then looking at DreamerV3's source code, you'll find:

**It doesn't directly call a standard GRUCell.**

Instead, it constructs the GRU itself in `_core()`.

The core logic is similar to:

```python
reset, cand, update = split(x)

reset = sigmoid(reset)
cand = tanh(reset * cand)

update = sigmoid(update - 1)

deter = update * cand + (1 - update) * deter
```

In other words, it computes the GRU gates itself.

This is also why you can't simply understand DreamerV3's RSSM as:

```python
self.gru = nn.GRUCell(...)
```

The source code also includes another very important design:

```text
Block GRU
```

---

## 10. Why Block GRU?

DreamerV3 defaults to:

```yaml
blocks: 8
```

And:

```text
deter = 8192
```

So the deterministic state is split into multiple blocks.

You can roughly think of it as:

```text
8192
 ↓
8 blocks
 ↓
1024 per block
```

In the source code:

```python
flat2group = lambda x: einops.rearrange(
    x,
    '... (g h) -> ... g h',
    g=g
)
```

This is doing exactly this kind of grouping.

Then through:

```python
nn.BlockLinear
```

A block-wise transformation is performed.

The purpose of this design is to control computation and parameter scale while maintaining the expressive power of a large deterministic state.

Therefore, DreamerV3's deterministic path is not simply:

```text
8192-dimensional GRU
```

But rather a GRU-like transition with block structure.

This is also a very noteworthy engineering design at the source code level.

---

## 11. What Is `_core()` Actually Computing?

Stripping away the extensive network details from the source code, `_core()` can be simplified to:

```python
def core(deter, stoch, action):

    stoch = flatten(stoch)

    x_deter = linear(deter)
    x_stoch = linear(stoch)
    x_action = linear(action)

    x = concat(
        x_deter,
        x_stoch,
        x_action
    )

    x = block_gru(x, deter)

    return deter
```

So what it really expresses is:

```text
Previous timestep:

deter_{t-1}
stoch_{t-1}
action_{t-1}

        │
        ▼

      RSSM Core

        │
        ▼

deter_t
```

This step **does not require the current observation at all**.

The current observation is an input to the posterior, not a direct input to the deterministic transition.

---

## 12. Complete Data Flow of the Observe Phase

Now let's put everything together:

```text
                  observation_t
                       │
                       ▼
                    Encoder
                       │
                       ▼
                    token_t
                       │
                       │
                       ▼
z_{t-1} ───────┐
               │
h_{t-1} ───────┼──► RSSM Core ◄── action_{t-1}
               │
               ▼
              h_t
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
      Prior        Posterior
    p(z_t|h_t)   q(z_t|h_t,o_t)
        │             │
        │             ▼
        │          sample
        │             │
        │             ▼
        │            z_t
        │
        └──────► KL ◄──────┘
```

Here we can see:

**RSSM actually has two stochastic paths:**

```text
Prior path
h_t → p(z_t | h_t)

Posterior path
h_t + observation → q(z_t | h_t, observation)
```

During training, the posterior is used to obtain latent states on real trajectories, while KL is used to constrain the prior.

---

## 13. KL Balancing: The Most Commonly Misunderstood Part of DreamerV3

Now let's look at the official loss:

```python
prior = self._prior(feat['deter'])
post = feat['logit']

dyn = self._dist(sg(post)).kl(
    self._dist(prior)
)

rep = self._dist(post).kl(
    self._dist(sg(prior))
)
```

The most important thing here is:

```python
sg(...)
```

That is:

```text
stop_gradient
```

### 1. Dynamics loss

The first term:

```python
dyn = KL(
    sg(posterior)
    || prior
)
```

The posterior is stop-gradiented.

So when optimizing this term:

```text
posterior: serves as target
prior: is trained
```

In other words:

> **Make the prior chase the posterior.**

### 2. Representation loss

The second term:

```python
rep = KL(
    posterior
    || sg(prior)
)
```

This time the prior is stop-gradiented.

So:

```text
prior: serves as target
posterior: is trained
```

In other words:

> **Keep the posterior's representation from drifting too far from the prior.**

### 3. Why both terms?

If we simply had:

```text
KL(posterior || prior)
```

Both networks would receive gradients.

This makes it hard to control:

> Who should move toward whom?

KL balancing explicitly specifies through stop-gradient:

```text
dyn:
prior → posterior

rep:
posterior → prior
```

So it's not simply:

```text
KL × a coefficient
```

But rather two KL losses in different directions, separately constraining the dynamics model and the representation.

---

## 14. What Are Free Nats?

Official configuration:

```yaml
free_nats: 1.0
```

Code:

```python
if self.free_nats:
    dyn = jnp.maximum(dyn, self.free_nats)
    rep = jnp.maximum(rep, self.free_nats)
```

That is:

```text
dyn = max(dyn, 1.0)
rep = max(rep, 1.0)
```

The idea here is not:

> "Each stochastic dimension must retain at least 1 bit of information."

More accurately:

> Provide a free zone for KL. When KL is small enough, stop applying strong constraints through the KL term.

This prevents the model from excessively pursuing:

```text
posterior ≈ prior
```

too early in training, which would cause the posterior to lose its ability to utilize observations.

---

## 15. Final Weights of KL Balancing

In the default configuration:

```yaml
loss_scales:
  dyn: 1.0
  rep: 0.1
```

So RSSM's KL-related loss can be understood as:

```text
L_RSSM
=
1.0 × dyn
+
0.1 × rep
```

Combined with:

```text
dyn = max(
    KL(stopgrad(post) || prior),
    free_nats
)

rep = max(
    KL(post || stopgrad(prior)),
    free_nats
)
```

This is much more accurate than simply saying:

> "DreamerV3 uses a single KL loss to make the prior approximate the posterior."

---

## 16. Imagine: What to Do Without Observations?

This is the most elegant part of Dreamer.

Training phase:

```text
observation → posterior → z
```

But during imagination:

```text
No observations
```

So we can only use:

```text
prior
```

The core logic in the source code can be simplified to:

```python
def imagine(carry, action):

    deter = core(
        carry["deter"],
        carry["stoch"],
        action
    )

    logit = prior(deter)

    stoch = sample(logit)

    return {
        "deter": deter,
        "stoch": stoch,
    }
```

Note:

**There is no zero-filling of observation embeddings here.**

This is a very important correction to the earlier draft.

Because DreamerV3's `_core()` does not depend on the current observation in the first place.

So the computation path for imagination is naturally:

```text
(h_t, z_t)
      │
      + action_t
      │
      ▼
    RSSM Core
      │
      ▼
    h_{t+1}
      │
      ▼
    Prior
      │
      ▼
    z_{t+1}
```

Then continues:

```text
(h_{t+1}, z_{t+1})
      +
    action_{t+1}
      ↓
...
```

This allows generating a latent trajectory without any real observations.

---

## 17. Why Can Imagination Be Used for Planning?

Suppose the real environment has given us:

```text
(h_t, z_t)
```

Next, the Actor produces:

```text
a_t
```

RSSM can then predict:

```text
(h_{t+1}, z_{t+1})
```

Then the reward head predicts:

```text
r_{t+1}
```

The value head predicts:

```text
V_{t+1}
```

So the entire process becomes:

```text
Current real state
     │
     ▼
   RSSM
     │
     ▼
  latent state
     │
     ├──► Actor → action
     │
     ├──► Reward Head → reward
     │
     └──► Value Head → value
              │
              ▼
          next latent
              │
              ▼
             ...
```

This is Dreamer's imagination.

It doesn't need to actually interact with the environment to "imagine" the future within a learned world model.

---

## 18. Why Does RSSM's Imagination Error Accumulate?

Because imagination is recursive.

Suppose:

```text
z_1 = real state
```

Then:

```text
z_2 = model(z_1, a_1)

z_3 = model(z_2, a_2)

z_4 = model(z_3, a_3)
```

If:

```text
z_2
```

already has error, then:

```text
z_3
```

is making predictions based on a state with error.

So:

```text
One-step prediction error
      ↓
Enters next step
      ↓
Continues accumulating
      ↓
Long rollouts gradually deviate from the real environment
```

This is also why DreamerV3 doesn't simply rely on infinitely long imagination, but uses a finite imagination horizon.

In the default configuration:

```yaml
imag_length: 15
```

So one imagination rollout typically only unfolds a limited number of steps forward.

---

## 19. RSSM's Sequence Training

In practice, training doesn't process just one timestep at a time.

Suppose:

```text
batch_size = B
sequence_length = T
```

Then the inputs are roughly:

```text
tokens: [B, T, embed_dim]
actions: [B, T, action_dim]
```

RSSM recurs along the time dimension:

```text
t=1:
(h_0, z_0, a_0)
       ↓
      h_1
       ↓
      z_1

t=2:
(h_1, z_1, a_1)
       ↓
      h_2
       ↓
      z_2

...

t=T:
(h_{T-1}, z_{T-1}, a_{T-1})
       ↓
      h_T
       ↓
      z_T
```

DreamerV3 uses JAX/Ninjax's `scan` to complete this recursion over the time dimension.

The code structure can be simplified to:

```python
carry, entries = scan(
    rssm_step,
    carry,
    (tokens, actions, resets)
)
```

Here `carry` is:

```text
{
    deter,
    stoch
}
```

So RSSM's state is not reinitialized at every timestep, but continuously passed through the entire sequence.

---

## 20. Why Does Reset Also Enter RSSM?

At the beginning of `_observe()` in the source code:

```python
deter, stoch, action = nn.mask(
    (carry['deter'], carry['stoch'], action),
    ~reset
)
```

This indicates that after an episode ends, you can't carry the previous episode's latent state into the next episode.

Otherwise:

```text
Episode A latent
       ↓
Episode B
```

Two completely different episodes would have state contamination.

So during reset:

```text
deter → reset
stoch → reset
action → reset
```

Then restart from the new episode's state.

This is also an engineering detail that is often overlooked between theoretical formulas and actual implementation.

---

## 21. DreamerV3's Default RSSM Configuration

According to the official configuration, the default large model RSSM parameters are:

```yaml
rssm:
  deter: 8192
  hidden: 1024
  stoch: 32
  classes: 64
  act: silu
  norm: rms
  unimix: 0.01
  outscale: 1.0
  imglayers: 2
  obslayers: 1
  dynlayers: 1
  absolute: False
  blocks: 8
  free_nats: 1.0
```

The key parameters can be understood as:

| Parameter | Meaning |
|:---|:---|
| `deter` | Dimension of the deterministic state |
| `hidden` | RSSM internal MLP hidden dimension |
| `stoch` | Number of categorical variables |
| `classes` | Number of classes per categorical variable |
| `unimix` | Uniform mixing ratio for categorical distributions |
| `blocks` | Number of groups in Block GRU |
| `free_nats` | KL free-nats threshold |
| `imglayers` | Number of prior network layers |
| `obslayers` | Number of posterior network layers |
| `dynlayers` | Number of dynamics network layers |

A particularly confusing point:

```text
stoch = 32
```

Does not mean the stochastic state is 32-dimensional.

It's actually:

```text
32 × 64 = 2048
```

Dimensions of categorical representation.

---

## 22. Compressing the Entire RSSM Into One Diagram

At this point, DreamerV3's RSSM can be summarized as:

```text
                  Observation o_t
                        │
                        ▼
                     Encoder
                        │
                        ▼
                    embedding
                        │
                        │
                        ▼
z_{t-1} ───────┐    ┌─────────┐
               ├───►│  Core   │◄──── action_{t-1}
h_{t-1} ───────┘    └────┬────┘
                          │
                          ▼
                         h_t
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
              Prior           Posterior
           p(z_t | h_t)    q(z_t | h_t,o_t)
                 │                 │
                 │                 ▼
                 │               sample
                 │                 │
                 │                 ▼
                 │                z_t
                 │                 │
                 └─────── KL ──────┘
```

Training phase:

```text
Real observation
      ↓
posterior
      ↓
z_t
```

Imagination phase:

```text
No observation
      ↓
prior
      ↓
z_t
```

And the two are connected through KL balancing.

---

## 23. Re-Understanding "World Model" From a Code Perspective

Looking back at DreamerV3's world model now, you'll find RSSM actually does three things.

### First: Memory

```text
(h_{t-1}, z_{t-1}, a_{t-1})
                ↓
               h_t
```

The deterministic state is responsible for maintaining historical context.

### Second: Inference

```text
(h_t, observation_t)
          ↓
        posterior
          ↓
         z_t
```

When there are real observations, the model can correct its state estimates.

### Third: Prediction

```text
(h_t, z_t, action_t)
          ↓
        prior
          ↓
     (h_{t+1}, z_{t+1})
```

When there are no observations, the model can rely on its learned dynamics to roll into the future.

So RSSM essentially unifies:

```text
State Estimation
        +
World Dynamics
```

And Dreamer's imagination further connects:

```text
World Dynamics
        ↓
Future Prediction
        ↓
Planning / Policy Learning
```

---

## 24. Revisiting RSSM's Core Formulas

If we compress DreamerV3's implementation into its most core mathematical form, it can be written as:

### Deterministic transition

```text
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
```

### Prior

```text
p(z_t | h_t)
```

### Posterior

```text
q(z_t | h_t, o_t)
```

### Dynamics KL

```text
L_dyn =
KL[
    sg(q(z_t | h_t, o_t))
    ||
    p(z_t | h_t)
]
```

### Representation KL

```text
L_rep =
KL[
    q(z_t | h_t, o_t)
    ||
    sg(p(z_t | h_t))
]
```

### Free Nats

```text
L_dyn = max(L_dyn, free_nats)

L_rep = max(L_rep, free_nats)
```

Then:

```text
L_KL =
λ_dyn L_dyn
+
λ_rep L_rep
```

Defaults:

```text
λ_dyn = 1.0
λ_rep = 0.1
```

After RSSM is fully trained:

```text
posterior
    ↓
Latent on real trajectories

prior
    ↓
Latent prediction without observations
```

This is the key to enabling DreamerV3's world model to perform imagination.

---

## 25. Summary

If you only remember a few key points about DreamerV3's RSSM, here's a summary table:

| Component | Role | DreamerV3 Implementation |
|:---|:---|:---|
| `deter` | Maintains temporal context | Block GRU |
| `stoch` | Represents stochastic latent | categorical |
| `classes` | Classes per categorical | Default 64 |
| Prior | Predicts latent without observation | `p(z_t \| h_t)` |
| Posterior | Infers latent with observation | `q(z_t \| h_t, o_t)` |
| `unimix` | Prevents categorical from becoming too sharp | Default 0.01 |
| `dyn KL` | Trains prior | posterior stop-gradient |
| `rep KL` | Constrains posterior | prior stop-gradient |
| Free Nats | Prevents KL constraints from being too strong too early | Default 1.0 |
| Observe | Updates state using real observations | posterior |
| Imagine | Observation-free rollout | prior |

From a code perspective, DreamerV3's RSSM is not simply:

```text
GRU + Gaussian
```

But rather:

```text
                 ┌──────────────┐
                 │  Block GRU   │
                 └──────┬───────┘
                        │
                      h_t
                    ┌───┴───┐
                    │       │
                  prior  posterior
                    │       │
                    │       │
                    ▼       ▼
               categorical latent
                    │
                    ▼
                   z_t
```

Where the deterministic state is responsible for "remembering history," the categorical stochastic state is responsible for "expressing state uncertainty," the posterior is responsible for using real observations for state inference, and the prior is responsible for predicting the future when there are no observations.

And KL balancing connects these two paths:

```text
Real world
   │
   ▼
posterior ───────► latent
   │                 ▲
   │                 │
   └────── KL ───── prior
                     ▲
                     │
                  dynamics
```

Ultimately, what's learned during training:

> **How to infer world states from observations.**

And what's used during imagination:

> **How to predict the world's future using only latent states and actions.**

This is also one of DreamerV3's most core ideas:

**The model doesn't imagine the future directly in observation space, but runs its world model within a learned latent space.**

---

## Source Code References

The code analysis in this article primarily corresponds to `dreamerv3/rssm.py` and `dreamerv3/configs.yaml` from the official DreamerV3 repository. The source code functions `RSSM._core()`, `_prior()`, `_dist()`, `observe()`, `imagine()`, and `loss()` correspond to the deterministic transition, prior/posterior, categorical sampling, observe/imagine rollout, and KL balancing discussed in this article, respectively.

* [DreamerV3 Official GitHub Repository](https://github.com/danijar/dreamerv3)
* [DreamerV3 RSSM Source Code rssm.py](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py)
* [DreamerV3 Default Configuration configs.yaml](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml)

> **Source code version note:** The DreamerV3 repository's configuration evolves with versions. If you need "code reproducibility" in your articles, it's recommended to also record the corresponding commit hash, rather than just referencing the `main` branch.
