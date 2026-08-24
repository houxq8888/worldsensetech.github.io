---
title: "Dreamer's Actor-Critic: How Policy Optimization Works in Imagination"
slug: "2026-08-27-dreamer-actor-critic"
date: 2026-08-27
draft: false
categories: ["World Models"]
tags: ["DreamerV3", "World Models", "RSSM", "Actor-Critic", "Imagination Training", "Dreamer Series"]
description: "Understanding Dreamer's Actor-Critic design from source code: imagine loop, lambda-return, two-hot value prediction, and symlog transformation."
toc: true
---

> **Dreamer Series - Part 2**
>
> Series directory (currently at Part 2):
> 1. [(Part 1) Understanding Dreamer: How World Models Learn to 'Imagine'](/en/articles/2026-08-25-dreamer-explained/)
> 2. **[(Part 2) Dreamer's Actor-Critic: Policy Optimization in Imagination](/en/articles/2026-08-27-dreamer-actor-critic/)**

The previous article clarified Dreamer's overall architecture: the world model predicts the future in latent space, and the policy learns on imagined trajectories. But this leaves a core question: **how exactly does the policy extract learning signals from imagined trajectories?**

This article dissects the complete Actor-Critic workflow from DreamerV3 source code: how imagination rollout generates features, how lambda-return is computed, how two-hot value prediction works, and why symlog transformation is indispensable.

This article is based on the [danijar/dreamerv3@e3f02248](https://github.com/danijar/dreamerv3) JAX reference implementation snapshot. Code references come from `dreamerv3/agent.py`, `dreamerv3/rssm.py`, `embodied/jax/`, and other files. Note that there are some engineering-level differences between the open-source implementation and the paper description; this article follows source code behavior.

## 1. From Imagination to Policy: Complete Pipeline Overview

Let's first establish a global perspective. One policy update in Dreamer goes through the following steps:

```text
Real sequences from replay buffer
         |
    encoder + posterior -> initial latent state (h0, z0)
         |
    imagine loop (15 steps):
        actor -> action -> RSSM deterministic transition -> prior stochastic state -> next latent feature
         |
    (No observation during imagination; stochastic state is entirely produced by RSSM prior rollout)
         |
    Each step's latent feature -> reward predictor / continuation predictor / value predictor
         |
    lambda-return computation -> Actor loss + Critic loss
         |
    Gradient update
```

The key insight: imagined trajectories are just sequences of latent features. For the policy to learn from them, the reward model provides immediate feedback, the continuation model predicts whether the episode continues, and the value model estimates long-term value—together they form the lambda-return, then Actor and Critic update separately.

It's worth emphasizing that imagination's essence is not "predicting the future for display," but rather **transforming the world model into a simulator that generates large-scale low-cost training data**. Dreamer's biggest innovation is not the actor-critic form, but this data generation paradigm:

```text
environment interaction
        |
world model (learns latent dynamics)
        |
large-scale imagined transitions
        |
actor-critic learning
```

Real environment interaction is expensive, but from a learned world model, we can sample large quantities of low-cost imagined trajectories (limited by imagination horizon, number of starting points, and model capacity). Policy optimization is no longer completely constrained by real sample size—this is the core idea that distinguishes the Dreamer series from traditional RL.

## 2. Imagine Loop: From Latent to Feature Sequences

The specific imagination process is implemented in RSSM's `imagine()` method (`rssm.py`):

```python
def imagine(self, carry, policy, length, training, single=False):
    # When single=True, execute single step
    action = policy(sg(carry))
    actemb = nn.DictConcat(self.act_space, 1)(action)
    deter = self._core(carry['deter'], carry['stoch'], actemb)
    logit = self._prior(deter)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    carry = nn.cast(dict(deter=deter, stoch=stoch))
    feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
    return carry, (feat, action)
```

Multi-step rollout is unfolded along the time dimension via `nj.scan`. Each step's output is a feature (deterministic + stochastic state) and the corresponding action.

In `Agent.loss()`, imagination is called as follows:

```python
K = min(self.config.imag_last or T, T)
H = self.config.imag_length  # Default 15
starts = self.dyn.starts(dyn_entries, dyn_carry, K)
policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
_, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
```

Several implementation details worth noting:

**Starting points come from real data.** `starts` are posterior states corresponding to real sequences sampled from the replay buffer. This ensures imagination begins from latent states encoded from real experience, not from random latent initialization.

**Imagination length is a hyperparameter.** `imag_length: 15` is the agent's training configuration, not an RSSM structural parameter. Common official configurations (DreamerV1/V2/V3) all use 15, but the specific value depends on the task.

**Actor-Critic loss does not update world model representation by default.** The source code `sg(first, skip=self.config.ac_grads)` controls whether to block gradients at the imagined rollout's starting point (`ac_grads` defaults to `False`). By default, actor-critic loss does not backpropagate through imagination dynamics to update RSSM, encoder, and other world model parameters. The world model still updates independently through its own reconstruction loss, KL loss, reward prediction loss, and continuation prediction loss—the stop-gradient only isolates the impact of policy optimization on the world model; the world model itself continues learning from real data.

## 3. Actor and Critic Network Structure

Both Actor and Critic are built using the MLPHead template—independent MLP backbones + task-specific output heads, with no shared parameters. Both receive the same input (RSSM feature), but hidden layers are completely independent.

```python
# In Agent.__init__()
self.pol = embodied.jax.MLPHead(act_space, outs, **config.policy, name='pol')
self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
```

The default MLP backbone configuration is 3 layers, 1024 units, SiLU activation, RMS normalization:

```python
# configs.yaml
policy: {layers: 3, units: 1024, act: silu, norm: rms, minstd: 0.1, maxstd: 1.0, outscale: 0.01, unimix: 0.01}
value:  {layers: 3, units: 1024, act: silu, norm: rms, output: symexp_twohot, outscale: 0.0, bins: 255}
```

Note that Actor and Critic have different output heads:

**Actor** output depends on action space type. Discrete actions use categorical distribution; continuous actions use squashed normal distribution—the network predicts mean and standard deviation (std constrained to `[minstd, maxstd]` range), samples are transformed through tanh and scaled to the environment's action range.

**Critic** uses `symexp_twohot` output—one of DreamerV3's most critical designs, expanded separately below.

## 4. Lambda-Return: Where Policy Learning Signals Come From

Dreamer does not directly accumulate immediate rewards on imagined trajectories. It uses lambda-return as the policy optimization objective—balancing bias and variance.

### Lambda-Return Recursive Formula

The `lambda_return()` function in source code (`agent.py`):

```python
def lambda_return(last, term, rew, val, boot, disc, lam):
    rets = [boot[:, -1]]
    live = (1 - f32(term))[:, 1:] * disc
    cont = (1 - f32(last))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return jnp.stack(list(reversed(rets))[:-1], 1)
```

This function recurses backward from the end of the imagined trajectory. The simplified form is:

```text
R_t^lambda = r_t + gamma * ((1-lambda) * V(s_{t+1}) + lambda * R_{t+1}^lambda)
```

Where `gamma` is the discount factor, `lambda` controls the tradeoff between Monte Carlo and TD. The above omits the continuation factor. In DreamerV3 implementation, the continuation model outputs `p(not terminal)`, which is used in two places during imagination: first, converted to `term = 1 - con` to control whether lambda-return continues bootstrapping; second, as part of discount weighting (`weight = cumprod(disc * con)`) to reduce contribution from unreliable future steps. `term` and `con` essentially come from the same continuation prediction, just used differently. `last` represents the boundary of the imagined sequence (imagination horizon truncation), controlling lambda recursion propagation. When continuation approaches 0, `term` approaches 1, bootstrap is blocked, and subsequent returns no longer propagate. Real environment terminal signals are only used for training the continuation predictor.

### Default Discount and Lambda Configuration

```yaml
imag_loss: {slowtar: False, lam: 0.95, actent: 3e-4, slowreg: 1.0}
horizon: 333
contdisc: True
```

DreamerV3 does not use pure fixed discount. It multiplies fixed horizon discount (`disc = 1 - 1/horizon ~ 0.997`) with learned continuation prediction to form effective discount (`weight = cumprod(disc * con)`), making value propagation consider both temporal decay and model-predicted episode continuation.

`lam=0.95` means the return estimator leans more toward using actual returns from multi-step imagined rollouts, while still retaining value bootstrap to control variance. Lambda=0 corresponds to pure TD(0), lambda=1 corresponds to Monte Carlo return, 0.95 is a high-order TD mixture.

### Role of Continuation Model

The `con` parameter is the continuation model's prediction—`p(not terminal)`, i.e., whether the model believes the future is still in a valid rollout state. In imagination rollout, this prediction is used in two places: first, converted to `term = 1 - con` to control whether lambda-return continues bootstrapping; second, as part of discount weighting (`weight = cumprod(disc * con)`) to reduce contribution from unreliable future steps. If continuation prediction approaches 0, `term` approaches 1, bootstrap is blocked, and subsequent returns no longer propagate. Note that real environment terminal signals are only used for training the continuation predictor; both term and con in imagination come from model predictions. This is especially important for robot tasks: when a robotic arm falls or collides causing episode termination, the continuation prediction drops, and value estimation must account for this.

When calling `imag_loss`:

```python
los, imgloss_out, mets = imag_loss(
    imgact,
    self.rew(inp, 2).pred(),     # reward prediction
    self.con(inp, 2).prob(1),    # continuation probability
    self.pol(inp, 2),            # policy distribution
    self.val(inp, 2),            # value prediction
    self.slowval(inp, 2),        # slow (target) value prediction
    ...)
```

## 5. Actor Loss: REINFORCE-Style Policy Gradient

Before diving into code, let's understand why Actor needs Critic. If we directly use return as the policy gradient weight, all actions in the same state would share enormous variance—good and bad actions would be weighted by the same return value. The value network provides a baseline; advantage = return - value measures "how much better is this action than average," significantly reducing policy gradient variance.

Now let's look at the complete Actor-Critic information flow. A common misconception is "Dreamer uses the world model to directly optimize actions," but the actual information flow is:

```text
imagined latent state z_t
          |
       policy (Actor)
          |
       action a_t
          |
       RSSM prior
          |
      imagined future z_{t+1}
          |
 reward predictor + value predictor
          |
     lambda return
          |
     advantage
          |
 policy gradient (update Actor)
```

The policy network samples actions from imagined states, actions pass through RSSM prior to produce imagined futures, reward and value predictors give feedback on imagined futures, lambda-return aggregates into long-term signals, advantage evaluates how much better this action is than baseline, and finally policy gradient updates Actor. **The world model does not directly output actions—it provides training data, and the policy network learns from it.**

Actor loss computation in `imag_loss`:

```python
voffset, vscale = valnorm.stats()
val = value.pred() * vscale + voffset
slowval = slowvalue.pred() * vscale + voffset
tarval = slowval if slowtar else val

disc = 1 if contdisc else 1 - 1 / horizon
weight = jnp.cumprod(disc * con, 1) / disc

ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

roffset, rscale = retnorm(ret, update)
adv = (ret - tarval[:, :-1]) / rscale
aoffset, ascale = advnorm(adv, update)
adv_normed = (adv - aoffset) / ascale

logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}

policy_loss = sg(weight[:, :-1]) * -(
    logpi * sg(adv_normed) + actent * sum(ents.values()))
```

This code does several things:

**1. Compute advantage.** `adv = (ret - tarval) / rscale`—lambda-return minus baseline (value prediction), then normalized by dividing by return's standard deviation.

**2. Normalize advantage.** Further standardization through `advnorm`, subtracting mean and dividing by standard deviation.

**3. Policy gradient.** `logpi * stop_gradient(adv_normed)`—standard REINFORCE style: log probability multiplied by normalized advantage estimate. Advantage estimate has stop-gradient, not backpropagating to value network.

**4. Entropy regularization.** `actent * entropy`, default `actent=3e-4`, encouraging policy to maintain exploration.

**5. Time step weighting.** `weight = cumprod(disc * con)`—each time step's loss is weighted by discount factor and continuation probability. Further future steps have lower weight; after episode termination, weight becomes zero.

Note an important design here: **DreamerV3's default Actor update primarily uses advantage-weighted log probability, i.e., REINFORCE-style score-function gradient.** The source code retains the possibility of letting Actor-Critic loss pass through imagination dynamics, controlled by `ac_grads` determining whether to block gradients at the imagined rollout's starting point. Default is off, so Actor-Critic updates don't affect encoder/RSSM latent state. But DreamerV3's default Actor objective itself is still advantage-weighted log probability (score-function style), not pure dynamics backprop relying on action gradients. Actor's gradients backpropagate to the policy network through `logpi`, but advantage is stop-gradient. Meanwhile, the entropy term's gradient directly differentiates with respect to policy parameters, not relying on score-function estimator.

It's worth clarifying that the Dreamer series did not start using REINFORCE from V2. Actually, DreamerV1 already explored both gradient paths simultaneously:

* **dynamics gradients (pathwise / analytic gradients)**: gradients pass directly through RSSM's differentiable dynamics
* **reinforce gradients**: score-function gradients through log pi(a|s) x advantage

Both were implemented and compared in the DreamerV1 paper. DreamerV2 primarily used analytic gradients through stochastic latent imagination. By DreamerV3, the source code defaults to `ac_grads=False`, meaning actor loss does not pass through imagination dynamics, using score-function objective, but retaining the interface. DreamerV3's choice is not because dynamics gradients are theoretically infeasible, but rather that in large-scale, multi-task training settings, score-function objective more easily achieves stable behavior—the discrete categorical latent makes pathwise gradients more difficult, and longer imagination horizons make dynamics gradients more sensitive to model errors; these are all comprehensive considerations.

### Why Not Directly Backprop Through Model?

A natural question: since the world model is differentiable, why not directly use `nabla_a (r + gamma V)` to backpropagate to action, letting Actor maximize imagined reward?

Early Dreamer versions did try this dynamics gradient:

```text
policy -> action -> RSSM -> reward -> gradient
```

The advantage is high sample efficiency—gradients pass directly through the model, short path. But disadvantages are also clear: sensitive to model error, gradients easily become unstable over long horizons. World model prediction errors amplify along the backpropagation path, causing policy update directions to deviate.

DreamerV3's default configuration primarily uses score-function style objective:

```text
policy -> sample action -> log probability -> advantage
```

Meanwhile, the code retains the pathwise dynamics gradient option (controlled through `ac_grads`). Default is off, trading some model gradient utilization for training stability. Advantage has stop-gradient, not backpropagating through the world model—policy updates only rely on "how well this action performed on imagined trajectories," not "what precise state the world model predicts this action will lead to."

## 6. Critic Loss: Two-Hot Value Prediction

Critic's loss computation follows Actor:

```python
voffset, vscale = valnorm(ret, update)
tar_normed = (ret - voffset) / vscale
tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)

losses['value'] = sg(weight[:, :-1]) * (
    value.loss(sg(tar_padded)) +
    slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]
```

The key here is `value.loss()`—it's not simple MSE, but two-hot categorical cross-entropy loss.

### Two-Hot Encoding Principle

DreamerV3's Critic does not directly output a scalar value, but predicts value distribution over a set of discrete bins. Default configuration uses 255 bins, covering an extremely large dynamic range.

Given a target value `v`, two-hot encoding finds two adjacent bins `b_below` and `b_above`, assigning weights by inverse distance:

```text
weight_below = (b_above - v) / (b_above - b_below)
weight_above = (v - b_below) / (b_above - b_below)
```

Then cross-entropy loss is applied to predicted logits. This is a form of distributional regression: the network predicts a probability distribution over discrete support, while the continuous scalar target obtains a soft label through linear interpolation of two adjacent bins.

Source code `TwoHot.loss()` (`embodied/jax/outs.py`):

```python
def loss(self, target):
    target = sg(self.squash(target))
    below = (self.bins <= target[..., None]).astype(i32).sum(-1) - 1
    above = len(self.bins) - (self.bins > target[..., None]).astype(i32).sum(-1)
    below = jnp.clip(below, 0, len(self.bins) - 1)
    above = jnp.clip(above, 0, len(self.bins) - 1)
    equal = (below == above)
    dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - target))
    dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - target))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target = (
        jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None] +
        jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None])
    log_pred = self.logits - jax.scipy.special.logsumexp(
        self.logits, -1, keepdims=True)
    return -(target * log_pred).sum(-1)
```

### symexp_twohot: Bin Construction

Bin spacing is not uniform. The `symexp_twohot` head samples uniformly on `[-20, 0]`, then applies symexp transformation to get exponentially-spaced bins, then mirrors to the positive half:

```python
def symexp_twohot(self, x):
    shape = (*self.space.shape, self.bins)
    logits = self.sub('logits', nets.Linear, shape, **self.kw)(x)
    if self.bins % 2 == 1:
        half = jnp.linspace(-20, 0, (self.bins - 1) // 2 + 1, dtype=f32)
        half = nets.symexp(half)
        bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
    else:
        half = jnp.linspace(-20, 0, self.bins // 2, dtype=f32)
        half = nets.symexp(half)
        bins = jnp.concatenate([half, -half[::-1]], 0)
    return outs.TwoHot(logits, bins)
```

The 255 bins are uniformly distributed in symlog domain, mapped to raw scale after symexp, thus exponentially spaced in raw value space. Theoretical coverage reaches approximately +/-4.8x10^8, but actual training mainly occurs near symlog space; these extreme bins are more for improving robustness under abnormal scales—because `TwoHot.loss()` internally first does `target = sg(self.squash(target))`, i.e., symlog transformation. **Bins cover symlog space, not raw value space.** Therefore, two-hot loss's actual working space is symlog(value), and the network mainly learns the compressed value distribution.

During prediction, take the distribution's expected value. In source code, `TwoHot.pred()` leverages the symmetric structure since bins are symmetric around 0—for odd bins, the middle bin is handled separately, both sides added symmetrically, making predicted value 0 when initialized with uniform logits.

### Slow Value Network

Critic loss also has a `slowreg` term:

```python
slowreg * value.loss(sg(slowvalue.pred()))
```

`slowvalue` is a target value network updated via EMA (exponential moving average):

```yaml
slowvalue: {rate: 0.02, every: 1}
```

Each step slides online value network parameters to slow network at `rate=0.02`. `slowreg=1.0` requires online value network predictions to simultaneously fit lambda-return target and slow network predictions. It's more like an EMA teacher providing stable value reference, not a target network in traditional TD algorithms (like DQN's periodic hard copy).

## 7. DreamerV3's Scale Robustness Design

DreamerV3 can train on vastly different tasks like Atari, MuJoCo, and robot manipulation with the same hyperparameters, relying not on a single trick but a combination of multi-layer scale normalization designs.

### symlog Transformation

First, the basic mathematical tools:

```python
# embodied/jax/nets.py
def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))
```

symlog compresses large values while preserving small value precision. These two functions are inverses: `symexp(symlog(x)) = x`.

### Both Reward and Value Use Two-Hot Heads

A common misconception is "reward uses symlog regression, value uses two-hot distribution." Actually, **both reward and value predictions in DreamerV3 use two-hot distribution heads.** The difference lies in target processing:

* **Reward prediction**: reward head uses two-hot distribution, target directly undergoes symlog transformation (`reward -> symlog -> two-hot`), making immediate rewards from different tasks fall into similar numerical ranges
* **Value prediction**: lambda-return first undergoes percentile-based return normalization, then enters value head's two-hot loss. Inside two-hot loss, symlog squash is applied again (`lambda-return -> return normalization -> symlog squash -> two-hot`), further compressing dynamic range

Both heads use two-hot distribution, but target scale processing paths differ: reward trains directly in symlog reward space, value trains in symlog space after return normalization. Note that reward normalization and return normalization are not the same mechanism—the former handles immediate reward scale, the latter handles long-term cumulative return scale. Network outputs logits, softmax gives discrete distribution, expectation is taken then symexp inverse transformation returns to raw scale—the inverse transformation is mainly for obtaining interpretable predictions; training loss itself is computed on symlog support.

### Return Normalization

Besides symlog and two-hot, DreamerV3 also applies percentile-based normalization to lambda-return:

```yaml
retnorm: {impl: perc, rate: 0.01, limit: 1.0, perclo: 5.0, perchi: 95.0, debias: False}
```

This uses 5th and 95th percentiles for normalization:

```python
roffset, rscale = retnorm(ret, update)
adv = (ret - tarval[:, :-1]) / rscale
```

Advantage first divides by return's scale, then undergoes further mean-std normalization. This ensures advantages across different tasks and training stages are in similar numerical ranges, avoiding gradient explosion or vanishing.

### How Three-Layer Design Cooperates

```text
reward
  |
symlog compression
  |
reward two-hot head

reward + continuation
  |
lambda-return
  |
return normalization
  |
symlog squash (inside two-hot loss)
  |
value two-hot head
```

symlog solves immediate reward scale differences, return normalization solves long-term cumulative return scale changes, two-hot distribution provides flexible value distribution representation. Together they enable the same hyperparameters to train stably across tasks.

## 8. Dreamer vs MPC: Amortized Decision Making

Readers familiar with Model Predictive Control (MPC) might ask: with a world model, why not directly search for optimal action sequences at every step?

The core differences are:

* **MPC is online planning**: each decision re-searches for optimal action sequences in the world model, executes the optimal first step, then re-searches.
* **Dreamer is amortized planning**: first train the Actor network on imagined trajectories, deployment only needs one policy forward pass to output actions, no model search needed.

Optimization objects also differ: MPC online optimizes action sequences, Dreamer offline optimizes policy parameters.

MPC's advantage is not needing to train a policy network; disadvantage is each decision requires massive search, high computational cost. Dreamer transforms online planning into policy optimization completed in one training phase, more efficient at deployment, advantageous when high-frequency real-time decisions and large-scale policy learning are needed. More strictly, Dreamer transforms future predictions in the model into training data, amortizing future decisions through the policy network, rather than re-searching action sequences at each execution—this is amortized model-based reinforcement learning, a more accurate positioning than planning. However, MPC remains very important in manipulation, locomotion control, and safety-critical scenarios; the two are not substitutes.

## 9. Complete Training Loop

Putting all components together, Dreamer's complete training loop is:

```text
Real environment execution -> collect (obs, action, reward) -> store in replay buffer
         |
Sample sequences -> encoder + RSSM observe -> world model training
    (reconstruction loss + KL loss + reward/continuation prediction loss)
         |
Sample posterior state -> imagine loop (15 steps)
         |
Each step feature -> reward/continuation/value prediction
         |
lambda-return -> Actor loss (REINFORCE + entropy) + Critic loss (two-hot CE + slowreg)
         |
Gradient update Actor + Critic
         |
Actor returns to real environment execution -> new data collection round
```

This world model is not a complete simulator of the physical world—it learns task-relevant latent dynamics. Dreamer's core is not "how vivid the imagination is," but "whether the imagination is sufficient to support correct decisions."

## 10. Key Hyperparameter Quick Reference

Actor-Critic related default configurations extracted from `configs.yaml`:

| Parameter | Default | Meaning |
|------|--------|------|
| `imag_length` | 15 | Imagination trajectory steps |
| `horizon` | 333 | Continuous discount horizon (disc ~ 0.997) |
| `lam` | 0.95 | Lambda in lambda-return |
| `actent` | 3e-4 | Entropy regularization coefficient |
| `slowreg` | 1.0 | Slow value network regularization weight |
| `slowvalue.rate` | 0.02 | EMA update rate |
| `policy.layers/units` | 3/1024 | Actor MLP structure |
| `value.layers/units` | 3/1024 | Critic MLP structure |
| `value.bins` | 255 | Two-hot bin count |
| `value.output` | symexp_twohot | Value output distribution type |
| `lr` | 4e-5 | Default optimizer learning rate |
| `retnorm` | perc (5-95) | Return normalization method |

## 11. Connecting Previous Articles

```text
World model intro -> RSSM deep dive -> RSSM code series (6 articles)
                                       |
                              Dreamer Series #1: Overall architecture
                                       |
                              Dreamer Series #2: Actor-Critic (this article)
                                       |
                              DreamerV3 training tips -> GPU selection
```

If you're new to world models, start with [What is a Robot World Model?](/en/articles/world-model-intro/).

If you want code-level RSSM breakdown, the [RSSM Code Walkthrough Series](/en/articles/2026-08-19-rssm-code-walkthrough/) covers everything from stochastic state to KL balancing and imagine reset.

If you're interested in practical training issues, [DreamerV3 Training Tips](/en/articles/dreamerv3-training-tips/) summarizes practical experience from environment configuration to hyperparameter tuning.

## 12. Summary

Dreamer's Actor-Critic design can be summarized as:

* **Actor** uses REINFORCE-style policy gradient based on imagined trajectories to optimize policy. Since rollouts are generated from replay buffer latent states by the world model (policy-conditioned imagination trajectories), not online collected from real environment, advantage estimates are normalized, with entropy regularization encouraging exploration.
* **Critic** uses two-hot categorical distribution to predict value, with symexp-spaced bins covering extremely large dynamic range. Slow value network provides stable value targets.
* **Lambda-return** balances between Monte Carlo (actual returns on imagined trajectories) and TD (Critic's value bootstrap), default lambda=0.95 leaning toward multi-step returns.
* **symlog + two-hot + return normalization** together form multi-layer scale robustness design, a key engineering contribution for DreamerV3's cross-task stable training. Both reward and value use two-hot distribution heads; the difference is reward target undergoes symlog compression, value target undergoes return normalization.

These designs make DreamerV3 one of the most influential general model-based RL open-source implementations—the same hyperparameters work on vastly different tasks like Atari games, MuJoCo control, and robot manipulation.

Next article will likely discuss DreamerV3's training engineering practice—from GPU configuration to hyperparameter tuning practical experience.
