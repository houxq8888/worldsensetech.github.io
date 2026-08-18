---
title: "Understanding RSSM Through Code: Implementation Details in DreamerV3"
slug: "2026-08-19-rssm-code-walkthrough"
aliases:
  - /en/articles/2026-08-19-rssm-code-walkthrough.html
date: 2026-08-19
draft: false
categories: ["World Models"]
tags: ["RSSM", "DreamerV3", "World Model", "State Space Model", "Code Walkthrough"]
description: "A code-level breakdown of RSSM in DreamerV3: deterministic path, stochastic path, prior/posterior networks, reparameterization trick, and KL balancing."
toc: true
---

The previous two RSSM articles ([RSSM Deep Dive](/en/articles/rssm-deep-dive/) and [Architecture Evolution](/en/articles/world-model-transformer/)) covered the theoretical framework. But reading a paper and reading code are two different things — many design details only become clear when you look at the implementation. This article dissects the RSSM implementation in DreamerV3 directly at the code level, helping you build a mapping from "math formulas" to "runnable code."

## Where RSSM Sits in DreamerV3

DreamerV3's overall architecture can be divided into three blocks:

1. **Encoder**: compresses observations (images/states) into feature vectors
2. **RSSM**: performs temporal modeling in latent space — given historical observations and actions, predicts future states
3. **Decoder / Heads**: decodes hidden states into predicted observations, rewards, values, etc.

RSSM is the middle block, and the most critical one. Its inputs are the encoder-output features and the agent's actions; its output is a latent state sequence — all subsequent predictions (reconstructed observations, reward predictions, value computation) are based on this latent state.

To summarize RSSM's operation in one sentence: **it maintains a deterministic hidden state (GRU), and at each time step outputs a Gaussian distribution parameterized by that deterministic state, then samples a stochastic hidden state from that distribution. The deterministic state + stochastic state together form the complete latent representation.**

## RSSM Class Structure Overview

Let's start with the overall skeleton of the RSSM class (PyTorch-style pseudocode, as close to DreamerV3's actual implementation as possible):

```python
class RSSM(nn.Module):
    def __init__(self, obs_dim, act_dim, deter_dim, stoch_dim, hidden_dim):
        super().__init__()
        self.deter_dim = deter_dim  # deterministic hidden state dim, typically 1024
        self.stoch_dim = stoch_dim  # stochastic hidden state dim, typically 32

        # Deterministic path: GRU
        self.gru = nn.GRUCell(act_dim + obs_feat_dim, deter_dim)

        # Prior network: predicts (mean, std) from (deter)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)  # outputs mean and log_std
        )

        # Posterior network: predicts (mean, std) from (deter + obs_feat)
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + obs_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )

    def forward(self, obs_feat, action, prev_state):
        # obs_feat: encoder output features [B, obs_feat_dim]
        # action: actions [B, act_dim]
        # prev_state: (prev_deter, prev_stoch) previous hidden state

        prev_deter, prev_stoch = prev_state

        # Step 1: Update deterministic state
        gru_input = torch.cat([obs_feat, action], dim=-1)
        deter = self.gru(gru_input, prev_deter)

        # Step 2: Prior network prediction (without current observation)
        prior_mean, prior_log_std = self._split_dist(self.prior_net(deter))

        # Step 3: Posterior network prediction (depends on current observation features)
        post_input = torch.cat([deter, obs_feat], dim=-1)
        post_mean, post_log_std = self._split_dist(self.post_net(post_input))

        # Step 4: Sample from posterior distribution (during training)
        stoch = self._sample(post_mean, post_log_std)

        # Return current state and distribution info (for KL computation)
        state = (deter, stoch)
        dist = {
            'prior': (prior_mean, prior_log_std),
            'post': (post_mean, post_log_std)
        }
        return state, dist

    def _split_dist(self, params):
        mean, log_std = params.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)  # numerical stability
        return mean, log_std.exp()

    def _sample(self, mean, std):
        # Reparameterization trick
        eps = torch.randn_like(std)
        return mean + std * eps
```

This skeleton covers RSSM's core logic. Let's break down each key design below.

## The Deterministic Path: What the GRU Actually Does

```python
gru_input = torch.cat([obs_feat, action], dim=-1)
deter = self.gru(gru_input, prev_deter)
```

This line is straightforward: concatenate the encoder's observation features with the current action, feed them as GRU input, and update the deterministic hidden state.

**Why GRU instead of LSTM?** Both DreamerV3's paper and code chose GRU. It's not that LSTM wouldn't work — GRU has fewer parameters (one fewer gate), which means lower overfitting risk in latent space models. The deterministic path of RSSM is essentially performing "temporal integration in latent space" — given compressed information from all past observations and actions, it maintains a continuously updating context vector. GRU is sufficient for this task.

**Why is `deter_dim` 1024?** This dimension needs to encode two things simultaneously: (1) the temporal context of historical observations, and (2) sufficient information capacity for the downstream prior/posterior networks. DreamerV3's ablation shows that increasing deter_dim from 512 to 1024 gives a clear improvement, while going further to 2048 yields diminishing returns. 1024 is the sweet spot.

**An easily overlooked detail:** The GRU's input is `obs_feat` (encoder output), not the raw observation. This means the encoder's quality directly determines how much information the GRU can access. If the encoder loses critical information, no amount of GRU capacity can recover it.

## The Stochastic Path: Why Two Types of States

```python
# Prior: only looks at deterministic state
prior_mean, prior_std = self._split_dist(self.prior_net(deter))

# Posterior: looks at deterministic state + observation features
post_input = torch.cat([deter, obs_feat], dim=-1)
post_mean, post_std = self._split_dist(self.post_net(post_input))

# Sample from posterior during training
stoch = self._sample(post_mean, post_std)
```

This is RSSM's most critical design, and also the most counterintuitive part: **why two networks, two distributions?**

Let's understand the role of each network:

**Prior network**: predicts the next stochastic state based only on the deterministic state `deter`. It represents "the model's prediction of the future without seeing new observations." This network is used during training to compute KL divergence, and during inference/imagination to generate imagined trajectories (since no real observations are available).

**Posterior network**: predicts the next stochastic state based on the deterministic state `deter` and the current observation features `obs_feat`. It represents "the model's estimate of the current state after seeing the current observation." The posterior is used for sampling during training because it has observation information, making its estimates more accurate.

**Why can't we use just one?** If we only had the posterior network, inference would be impossible — during inference there are no real observations, so you can't compute the posterior. If we only had the prior network, during training the model would have to learn solely from its own predictions without any observation signal to correct it, which would lead to drift.

**The relationship between prior and posterior:** The training objective is to make the prior network gradually approach the posterior network. Ideally, if the model has learned a good enough world dynamics, the prior network's predictions should be close to the posterior network's — meaning the model can accurately predict future states using only historical information and actions, without needing observation corrections at every step. KL divergence measures this gap.

## The Reparameterization Trick: How Gradients Flow Through Sampling

```python
def _sample(self, mean, std):
    eps = torch.randn_like(std)
    return mean + std * eps
```

This code looks simple, but it solves a critical problem: **how to let gradients flow through a random sampling operation?**

If you wrote `stoch = torch.normal(mean, std)` directly, the sampling operation would be non-differentiable — gradients couldn't flow back from the loss to the computation of mean and std. The reparameterization trick rewrites the sampling as `mean + std * eps`, where `eps` is fixed noise sampled from a standard normal distribution. Now, the gradient of `stoch` with respect to `mean` is 1, and with respect to `std` is `eps`, so gradients can flow normally.

**Numerical pitfalls in implementation:**

```python
log_std = torch.clamp(log_std, min=-20, max=2)
return mean, log_std.exp()
```

The network outputs `log_std` rather than `std`, for two reasons: (1) `log_std` has a range over the entire real line, making it easier to output from a linear layer; (2) after taking `exp`, `std` is guaranteed to be positive without additional constraints. But `exp` can overflow when `log_std` is large, so clamping is essential. In DreamerV3, the clamp range is `[-20, 2]`, corresponding to a `std` range of approximately `[2e-9, 7.4]`.

## KL Divergence and Free Bits

```python
def kl_divergence(prior_mean, prior_std, post_mean, post_std):
    # Analytical KL divergence between two Gaussians
    var_ratio = (prior_std / post_std).pow(2)
    mean_diff = (post_mean - prior_mean).pow(2) / post_std.pow(2)
    kl = 0.5 * (var_ratio + mean_diff - 1.0 + torch.log(post_std.pow(2) / prior_std.pow(2)))
    return kl.sum(dim=-1)  # sum over stochastic dimensions
```

KL divergence measures the gap between the posterior and prior distributions. During training, we want this gap to be as small as possible — the closer the prior is to the posterior, the better the model can predict the future using only historical information.

**But KL can't simply be added to the loss.** If the KL weight is too large, the model will tend to make the prior and posterior identical, meaning the posterior no longer uses observation information — it degenerates into using only the prior. This causes reconstruction quality to drop.

DreamerV3 uses a technique called **KL balancing + free bits**:

```python
# KL balancing: not simple weighting, but dynamic adjustment
kl_loss = kl_divergence(prior, posterior)

# Free bits: each stochastic dimension retains at least `info` amount of information
# Prevents KL from being too small, which would cause the posterior to degenerate
# into deterministic encoding
kl_loss = torch.clamp(kl_loss, min=free_bits)  # free_bits typically around 1.0

# Final KL term in the loss
total_loss = recon_loss + kl_loss * kl_scale - reward_loss - value_loss
```

**The role of free bits:** If KL is too small (prior and posterior are too close), it means the model isn't fully utilizing observation information. Free bits set a lower bound, forcing each stochastic dimension to retain at least a certain amount of information. This is equivalent to telling the model: "You can make the prior close to the posterior, but not too close — the posterior must know at least `free_bits` more than the prior."

**KL scale scheduling:** DreamerV3 doesn't use a fixed KL weight. Instead, it uses progressive scheduling — early in training the KL weight is small (letting the model first learn good reconstructions), then gradually increases (making the prior progressively approach the posterior). This avoids the KL constraint being too strong early in training, which would prevent the model from learning anything.

## Unrolling RSSM Over Sequences

The code above is for a single time step. In practice, RSSM needs to be unrolled over a complete episode sequence during training:

```python
def forward_sequence(self, obs_feats, actions):
    # obs_feats: [T, B, obs_feat_dim]
    # actions: [T, B, act_dim]
    T, B = obs_feats.shape[:2]

    # Initialize state
    deter = torch.zeros(B, self.deter_dim, device=obs_feats.device)
    stoch = torch.zeros(B, self.stoch_dim, device=obs_feats.device)

    prior_means, prior_stds = [], []
    post_means, post_stds = [], []
    stochs = []

    for t in range(T):
        state, dist = self.forward(obs_feats[t], actions[t], (deter, stoch))
        deter, stoch = state

        prior_means.append(dist['prior'][0])
        prior_stds.append(dist['prior'][1])
        post_means.append(dist['post'][0])
        post_stds.append(dist['post'][1])
        stochs.append(stoch)

    # Stack into sequences
    return {
        'deter': deter,  # last step's deterministic state
        'stoch': torch.stack(stochs, dim=0),  # [T, B, stoch_dim]
        'prior': (torch.stack(prior_means), torch.stack(prior_stds)),
        'post': (torch.stack(post_means), torch.stack(post_stds))
    }
```

**Several engineering details:**

1. **Zero initial state.** The first time step has no prev_state, so `deter` and `stoch` are initialized to zero vectors. This means the first few predictions will be poor — the model has no historical information to work with. In practice, the loss for the first few steps is often discarded, or longer sequences are used to dilute the effect of the initial state.

2. **Step-by-step unrolling vs. parallelization.** The code above uses a for loop to unroll step by step, because each time step depends on the previous step's state. This means the sequence dimension of RSSM cannot be parallelized like a Transformer. This is an inherent limitation of RSSM — the recurrent nature of the GRU requires sequential processing. For long sequences (T > 200), this becomes a training bottleneck.

3. **How stoch is combined.** The final latent representation is the concatenation of `deter` and `stoch` (`[deter; stoch]`), which is fed to the decoder and various heads (reward head, value head, etc.). So the decoder's input dimension is `deter_dim + stoch_dim`, typically 1024 + 32 = 1056.

## Inference / Imagination Mode: What to Do Without Observations

During training, RSSM samples from the posterior. But during inference (or DreamerV3's "imagination" phase), there are no real observations — only the prior can be used:

```python
def imagine(self, prev_state, actions):
    # actions: [T, B, act_dim]
    T, B = actions.shape[:2]
    deter, stoch = prev_state

    imagined = []
    for t in range(T):
        # No observations, use prior only
        deter = self.gru(
            torch.cat([torch.zeros(B, obs_feat_dim), actions[t]], dim=-1),
            deter
        )
        prior_mean, prior_std = self._split_dist(self.prior_net(deter))
        stoch = self._sample(prior_mean, prior_std)
        imagined.append(torch.cat([deter, stoch], dim=-1))

    return torch.stack(imagined, dim=0)  # [T, B, deter_dim + stoch_dim]
```

**Note two changes:**

1. **obs_feat is replaced with zeros.** The GRU's input is normally `[obs_feat; action]`, but during imagination there are no observations, so zero vectors are used as placeholders. This means in imagination mode, the GRU receives information only from actions. This is a somewhat controversial design — some implementations directly modify the GRU's input structure so it only receives actions in imagination mode.

2. **Sampling from the prior.** The posterior network is not involved at all; the model relies entirely on the prior network's predictions. This means the quality of imagined trajectories depends entirely on how well the prior network has learned the world dynamics. If the prior network's predictions have errors, those errors accumulate during imagination — the longer the imagination, the larger the deviation.

## Engineering Details in the Actual Code

**LayerNorm placement:** DreamerV3 uses LayerNorm extensively in the prior/posterior networks. This isn't decorative — the training stability of latent space models depends heavily on normalization. Without LayerNorm, RSSM's KL divergence tends to explode early in training, causing the entire training to collapse.

**GELU vs ReLU:** DreamerV3 consistently uses GELU activation. Compared to ReLU, GELU has non-zero gradients in the negative region, which helps avoid the "dead neuron" problem. In latent space models, this is especially sensitive — because RSSM's outputs are used repeatedly (sequence unrolling), the impact of a single dead neuron gets amplified.

**Gradient clipping:** DreamerV3 applies global gradient clipping to RSSM (max norm = 1000). This value looks large, but RSSM's gradients can indeed be large — BPTT (Backpropagation Through Time) across the sequence unrolling causes gradient accumulation. Without clipping, occasional gradient spikes would directly cause training to collapse.

**Symmetric weight initialization:** DreamerV3 uses symmetric weight initialization, where the eigenvalues of the weight matrix are distributed uniformly on the unit circle. This helps with learning long-term dependencies in the GRU — avoiding gradient vanishing or explosion in the initial state.

## Summary

From a code perspective, RSSM's core design can be summarized as:

| Component | Role | Key Design |
|:---|:---|:---|
| GRU | Maintains deterministic temporal context | Input = encoder features + actions |
| Prior network | State prediction without observations | Input = deter, outputs Gaussian |
| Posterior network | State estimation with observations | Input = deter + obs_feat |
| Reparameterization | Lets gradients flow through sampling | mean + std * eps |
| KL balancing | Constrains prior to approximate posterior | Free bits prevent degeneration |
| Imagination mode | Observation-free rollout | Prior only, obs zero-filled |

RSSM's elegance lies in how it unifies two seemingly different tasks — "filtering with sequential observations" and "prediction without observations" — in a concise way. The prior and posterior share the same deterministic state (GRU), align their distributions through KL divergence, and ultimately enable the prior network to achieve prediction accuracy close to the posterior — this is the foundation that allows the "world model" to plan future actions in imagination.

Once you understand these code details, many formulas in the DreamerV3 paper stop being abstract symbols and start having concrete implementations.
