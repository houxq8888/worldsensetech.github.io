---
title: "DreamerV3 Training Engineering: From GPU Setup to Hyperparameter Tuning"
slug: "2026-08-28-dreamerv3-training-tips"
date: 2026-08-28
draft: false
categories: ["World Models"]
tags: ["DreamerV3", "Training Tips", "GPU", "Hyperparameters", "Engineering Practice", "Dreamer Series"]
description: "Practical engineering experience training DreamerV3: GPU memory optimization, hyperparameter tuning, common pitfalls and solutions."
toc: true
---

> **Dreamer Series - Part 3**
>
> Series directory (currently at Part 3):
> 1. [(Part 1) Understanding Dreamer: How World Models Learn to 'Imagine'](/en/articles/2026-08-25-dreamer-explained/)
> 2. [(Part 2) Dreamer's Actor-Critic: Policy Optimization in Imagination](/en/articles/2026-08-27-dreamer-actor-critic/)
> 3. **(Part 3) DreamerV3 Training Engineering: From GPU Setup to Hyperparameter Tuning**

The previous two articles covered Dreamer's architecture design and Actor-Critic工作原理 at the theoretical level. But getting DreamerV3 to actually run well requires solving a bunch of engineering problems: What do you do when GPU memory runs out? How do you tune hyperparameters? How do you debug training instability?

This article takes a practical perspective, summarizing engineering experience and common pitfalls in DreamerV3 training. Content is based on the [danijar/dreamerv3@e3f02248](https://github.com/danijar/dreamerv3) JAX reference implementation.

## 1. Environment Setup: JAX + GPU

### JAX GPU Support

DreamerV3 is implemented in JAX. JAX's GPU support depends on CUDA and cuDNN, and version matching matters during installation:

```bash
# Recommended installation (for CUDA 12.x)
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify that the GPU is correctly detected:

```python
import jax
print(jax.devices())  # Should show GPU devices
```

If the output is only `[CpuDevice]`, JAX hasn't correctly detected the GPU. Common causes include CUDA version mismatch, missing cuDNN, or environment variable issues.

Note that `nvidia-smi` showing a GPU doesn't guarantee JAX can use it — the CUDA driver version must be ≥ the runtime version, and multi-version CUDA environments can easily conflict. For more thorough diagnostics:

```python
import jax
jax.print_environment_info()  # Prints JAX, CUDA, cuDNN version info
```

### Memory Management

JAX pre-allocates nearly all GPU memory by default. This can be problematic when running multiple tasks or debugging. You can control this via environment variables:

| Variable | Effect |
|----------|--------|
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | Disables startup pre-allocation |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` | Limits pre-allocation to 90% |

```bash
# Disable pre-allocation (on-demand allocation)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Or limit pre-allocation ratio (still pre-allocates, but only 90%)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

Note: `MEM_FRACTION=0.9` is not "on-demand allocation" — it means "pre-allocate up to 90% of GPU memory". To truly disable pre-allocation, use `PREALLOCATE=false`.

For GPUs with 24GB memory (like RTX 3090/4090), DreamerV3's default configuration usually works. If memory is tight, you'll need to adjust batch size or imagination length.

## 2. Key Hyperparameters

DreamerV3's hyperparameters are defined in `configs.yaml`. Understanding what each parameter does is essential for tuning on specific tasks.

### World Model

```yaml
# RSSM structure
rssm:
  deter: 4096      # deterministic state dimension
  stoch: 32        # number of categorical variables
  classes: 32      # number of classes per categorical variable
  hidden: 4096     # RSSM internal MLP hidden dimension

# imagination
imag_length: 15    # imagination trajectory length
```

Note: `stoch=32` and `classes=32` mean the stochastic state consists of 32 categorical variables, each with 32 classes. The actual latent is a `32 × 32` categorical distribution, not a 32-dimensional continuous vector.

KL divergence appears in two forms on this categorical distribution:

- **Representation KL**: $D_{KL}[q(z_t | h_t, o_t) \| p(z_t | h_t)]$, measuring the gap between posterior (given observation) and prior (history-only)
- **Dynamics KL**: $D_{KL}[p(z_t | h_t) \| q(z_t | h_t, o_t)]$, opposite direction, used for stabilizing training

The key difference between the two is not just the KL direction, but also **what gets gradient updates**: dynamics loss primarily updates the prior network, while representation loss primarily updates the posterior encoder. This separated gradient flow design is a crucial mechanism for DreamerV3 to prevent latent collapse and maintain predictive capability.

**`imag_length`** is one of the most important hyperparameters. 15 steps is DreamerV3's default and works well on most tasks.

But beware a common misconception: **longer imagination does not equal better long horizon credit assignment**. The world model's prediction error $p(s_{t+k})$ accumulates rapidly as steps $k$ increase, so:

- **Longer imagination** (20-25 steps): Improves long-term planning ability, but increases risk of model bias accumulation
- **Shorter imagination** (10-15 steps): Relies more on critic's bootstrap estimates, lower variance, more stable training

If the task requires longer horizon to see reward signals, you can increase to 20-25, but only if **the world model itself is accurate enough**. Otherwise, long imagination will lead the policy astray.

### Actor-Critic

```yaml
# Actor
policy:
  layers: 3
  units: 1024
  act: silu
  norm: rms
  minstd: 0.1      # continuous action minimum standard deviation
  maxstd: 1.0      # continuous action maximum standard deviation
  outscale: 0.01   # output layer weight scaling
  unimix: 0.01     # discrete action uniform mixing coefficient

# Critic
value:
  layers: 3
  units: 1024
  output: symexp_twohot
  bins: 255        # number of bins in two-hot distribution

# Training
imag_loss:
  lam: 0.95        # lambda for lambda-return
  actent: 3e-4     # entropy regularization coefficient
  slowtar: False   # whether to use slow value as target
  slowreg: 1.0     # slow value network regularization weight

slowvalue:
  rate: 0.02       # EMA update rate
  every: 1         # update every step
```

**`actent`** controls exploration level. The default `3e-4` works well on most tasks. But note that in DreamerV3, exploration isn't controlled solely by `actent` — the Actor loss is imagination return + entropy, and the policy distribution itself is controlled by `minstd`/`maxstd` (continuous actions) and `unimix` (discrete actions).

If you need to adjust exploration behavior, here's the **empirical priority order** (not official recommendation, may differ by task):

1. **minstd / maxstd** (largest impact on continuous actions): Directly changes the standard deviation range of the policy distribution
2. **unimix** (discrete actions): Controls uniform mixing coefficient, affecting discrete action randomness
3. **actent** (entropy coefficient): As a final fine-tuning knob

In many cases, the default `actent` is sufficient — problems usually lie in `minstd`/`maxstd` settings.

**`lam`** controls the tradeoff between TD and Monte Carlo. 0.95 favors Monte Carlo (multi-step return), suitable when imagination is fairly accurate. If the world model isn't accurate enough, you can reduce to 0.9 or 0.85 to increase TD bootstrap weight and reduce variance.

### Optimizer

```yaml
lr: 4e-5           # learning rate
opt: {eps: 1e-20, clip: 1000.0}  # Adam optimizer parameters
```

DreamerV3 uses a relatively small learning rate `4e-5`, which is key to training stability. `clip: 1000.0` is the gradient clipping threshold to prevent gradient explosion.

## 3. Memory Optimization in Practice

### Problem: OOM (Out of Memory)

When batch size is large or imagination length is long, you may run out of memory. Here are several solutions:

**Solution 1: Reduce batch size**

```yaml
batch_size: 8      # reduce from 16 to 8
batch_length: 512  # sequence length can also be reduced
```

**Solution 2: Reduce imagination length**

```yaml
imag_length: 10    # reduce from 15 to 10
```

This reduces computation in the imagination phase but may affect long-term credit assignment.

**Solution 3: Use gradient checkpointing**

JAX supports gradient checkpointing, trading compute for memory:

```python
# Add gradient checkpointing to key functions
import jax

@jax.remat
def expensive_function(x):
    # ...
```

But this increases training time by about 20-30%.

**Solution 4: Mixed precision training (use with caution)**

DreamerV3 defaults to float32. If you need mixed precision, **strongly recommend bfloat16 over float16**, because bfloat16 has a larger exponent range and is more numerically stable.

But even with bfloat16, be cautious: DreamerV3 is highly sensitive to numerical stability, involving KL balancing, categorical logits, two-hot value distribution, symlog/symexp, and other components.

**Recommended strategy:**

- Only use low precision for encoder/decoder and large matrix multiplications
- Don't use low precision for RSSM latent dynamics, distribution logits, value head
- If training produces NaN or KL collapse, immediately fall back to float32

```python
# Set matmul precision in JAX (use with caution)
from jax import config
config.update("jax_default_matmul_precision", "bfloat16")
```

Note: `jax_default_matmul_precision` controls **matrix multiplication internal precision selection**, not model dtype. It doesn't automatically perform dtype casting and **is not equivalent to PyTorch's AMP (Automatic Mixed Precision)**. JAX's precision control (X64/X32/bf16) uses global or per-operation explicit settings, while PyTorch AMP automatically selects appropriate precision for different operations via autocast. Setting `jax_default_matmul_precision` doesn't mean "bf16 training is enabled" — the actual effect is limited to matmul operation internal precision.

Note: float16 easily causes KL collapse, value explosion, or NaN in DreamerV3 and is not recommended.

### Memory Monitoring

Monitor memory usage during training:

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use a more detailed tool
nvtop
```

If memory usage gradually increases during training, there may be a memory leak. JAX's memory management is usually quite stable, but certain custom operations can cause issues.

## 4. Training Stability Debugging

### Problem 1: Reward Not Increasing

If cumulative reward in the real environment doesn't increase during training, possible causes:

**World model hasn't learned well**

Check the world model loss curves:
- reconstruction loss should decrease steadily
- KL loss should fluctuate within a reasonable range
- reward prediction loss should decrease

If world model loss doesn't decrease, the learning rate may be too large or too small, or the encoder/decoder structure may not suit the current task.

**Actor-Critic isn't extracting useful signals from imagination**

Check Actor-Critic losses:
- advantage distribution should be reasonable (not all zeros or all extreme values)
- value prediction should be close to lambda-return
- entropy should be in a reasonable range

If advantage variance is very large, the value network may not have learned well. You can check slow value regularization (`slowreg`), `lam`, reward scale, and other factors. `slowreg` is not the first thing to adjust — first verify that reward scale and lambda settings are reasonable.

### Problem 2: Training Suddenly Crashes

Training is normal initially, but suddenly loss becomes NaN or Inf at some step:

**Gradient explosion**

Check gradient norms. DreamerV3 has gradient clipping, but if the `clip` value is set too high, it may not effectively prevent explosion. Try:

```yaml
opt: {clip: 100.0}  # reduce from 1000 to 100
```

**Numerical overflow**

The two-hot distribution bins cover symlog space, mapping back to original scale covering an extremely large range (about ±4.8×10⁸), but **training primarily happens in the compressed symlog space**. DreamerV3 compresses dynamic range through symlog, so value prediction doesn't need to directly predict huge raw returns.

If value prediction shows NaN, check:
- Whether real returns go through symlog transformation (`symlog(x) = sign(x) * log(|x| + 1)`)
- Whether two-hot bin range is sufficient
- Whether value prediction distribution is reasonable

**Poor imagination starting point quality**

If some sequences in the replay buffer are poor quality (e.g., very short episodes, or abnormally large rewards), imagination may produce abnormal values. Check the distribution of sequences in the replay buffer.

### Problem 3: Large Variation Across Seeds

If training results vary greatly after changing random seeds, training is unstable:

**Increase replay buffer**

```yaml
replay_size: 5e6   # increase from 1e6 to 5e6
```

A larger replay buffer smooths the sampling distribution and reduces variance.

**Reduce learning rate**

```yaml
lr: 2e-5           # reduce from 4e-5 to 2e-5
```

**Increase batch size**

```yaml
batch_size: 32     # increase from 16 to 32
```

Larger batch size provides more stable gradient estimates.

**Multi-seed evaluation**

Seed sensitivity in RL experiments is a known issue. If you're doing serious experimental comparisons:

- Run at least 3-5 different seeds
- Report median and confidence intervals (not single curves)
- Single-seed results have significantly reduced credibility

### Problem 4: Replay Buffer Sampling and Training Rhythm

Dreamer is very sensitive to replay buffer sampling strategy, but this is often overlooked. Just focusing on `replay_size` isn't enough — you also need to consider:

**Replay ratio (train/collect ratio)**

Dreamer doesn't "first train world model then train Actor" — it uses **joint updates**:

```text
Collect data → Joint update: world model + Actor-Critic → Collect more data → ...
```

DreamerV3 typically uses a high replay ratio, but specifics depend on environment speed and configuration — for fast simulation environments, ratio may be much greater than 1; for slow environments like real robots, ratio may be close to 1. This ratio directly affects training efficiency:

- ratio too low: model doesn't fully utilize existing data, training is slow
- ratio too high: overfits to old data in replay buffer, policy degrades

DreamerV3's default ratio is already reasonable for most tasks, but can be adjusted if training efficiency isn't ideal.

**Warmup phase**

Early in training, there's little data in the replay buffer and the world model hasn't learned well. Things to note:

- The first few hundred steps are mainly random exploration, filling the replay buffer
- You should typically see world model loss start decreasing in the early training phase, but exact steps vary by environment complexity — simple environments may start decreasing after a few hundred steps, complex environments (Atari, robotics, sparse rewards) may need more steps
- Actor-Critic starts meaningfully updating only after the world model is initially stable
- If warmup phase is too long with no improvement, check environment interface and hyperparameters

**Train/eval ratio**

The ratio of training to evaluation is also important. Evaluating too frequently wastes training time; evaluating too infrequently means problems aren't caught in time. Recommend evaluating every 1000-5000 steps.

### Problem 5: World Model Collapse Diagnosis

World model collapse is Dreamer's most insidious failure mode. Unlike reward not increasing, loss curves may look completely normal during collapse.

**What is collapse?**

The RSSM's latent space gradually loses information representation capability. Specific manifestations:

- posterior entropy continuously decreases, approaching zero
- KL between prior and posterior approaches zero (model "can't be bothered" to use observation information)
- reconstruction loss may still be decreasing, but latent state no longer contains useful information

**Diagnostic methods:**

1. Monitor `posterior entropy` and `prior entropy` trends
2. If both continuously decrease and the gap narrows, collapse may be occurring
3. Latent embedding visualization (like PCA/t-SNE) serves only as auxiliary reference — RSSM's categorical latent and temporal structure make t-SNE not entirely reliable. More recommended diagnostic methods are latent probing, linear probe, or rollout reconstruction

**Repair strategies:**

- When adjusting KL free scale / free nats, balance reconstruction and dynamics learning — free bits' role is to ignore optimization pressure in small KL intervals, not to force posterior to carry more information. Setting too large causes posterior/prior divergence, prior can't track posterior, dynamics learning becomes difficult
- Check if encoder capacity is sufficient (weak encoder leads to uninformative posterior)
- Reduce KL weight (if any) to give posterior more freedom
- Check reconstruction loss weight — if recon weight is too low, encoder has no motivation to extract useful information

## 5. Task-Specific Tuning Advice

### Atari Games

Atari game observations are pixel images, and rewards are usually sparse (only when score changes).

**Recommended configuration adjustments:**

```yaml
# Can try increasing imagination length (need to confirm world model prediction quality)
imag_length: 20    # check imagined rollout quality before deciding to increase

# Use more aggressive exploration
policy:
  unimix: 0.05     # increase uniform mixing, add randomness

# Adjust lambda, as world model may not be accurate enough in pixel space
imag_loss:
  lam: 0.9         # reduce lambda, increase TD weight
```

**Notes:**

- DreamerV3 primarily relies on encoder's temporal modeling and RSSM hidden state to capture temporal information, unlike early DQN which relied on frame stacking. However, some implementations may still use multi-frame input, depending on environment wrapper
- Some games have very large reward ranges; symlog transformation can help handle this
- If training is slow, you can reduce `batch_length`, since Atari episodes are usually very long

### MuJoCo Control

MuJoCo is a continuous control task where observations are low-dimensional state vectors and actions are continuous.

**Recommended configuration adjustments:**

```yaml
# MuJoCo usually doesn't need very long imagination
imag_length: 15    # default is usually sufficient

# Continuous action standard deviation range
policy:
  minstd: 0.1
  maxstd: 1.0

# Default entropy is usually sufficient, only increase if exploration is insufficient
imag_loss:
  actent: 3e-4     # default value, consider increasing to 1e-3 only if exploration is insufficient
```

**Notes:**

- Many continuous control tasks have sufficient default entropy; blindly increasing `actent` may delay convergence and reduce final performance
- MuJoCo episodes usually have fixed time length, not requiring much from the continuation model
- If policy converges prematurely, prioritize adjusting `minstd`/`maxstd`, then consider increasing `actent`
- If training is unstable, check if action scales need normalization

### Robot Manipulation

Robot manipulation tasks typically have high state dimensions and complex dynamics.

**Recommended configuration adjustments:**

```yaml
# Robot tasks may need longer imagination to understand causality
imag_length: 20

# Increase RSSM capacity
rssm:
  deter: 4096
  hidden: 4096

# If episodes frequently terminate early, continuation model is important
# Ensure continuation prediction loss decreases normally
```

**Notes:**

- Robot task simulations are usually slow, replay buffer fill rate may be slow
- If episodes frequently terminate early due to collisions or drops, continuation model accuracy is important
- Real robot experiments need to consider sim-to-real gap; DreamerV3's imagination learns in simulation and may need fine-tuning when deployed on real robots

**Engineering problems specific to robot tasks:**

- **Action delay / repeat**: Real robots usually have action delay, which needs to be handled in environment wrapper. Action repeat can mitigate delay issues but affects control frequency
- **Proprioception and observation normalization**: Robot proprioception (joint angles, velocities, etc.) and visual observations have very different scales; observation normalization is essential
- **Contact dynamics**: Contact dynamics is the hardest part to model in robot tasks; world model predictions are often inaccurate in contact regions
- **Domain randomization**: For sim-to-real transfer, randomization needs to be introduced in simulation (textures, lighting, friction coefficients, etc.) to improve policy robustness
- **Observation latency**: Multi-sensor synchronization and timestamp alignment is an easily overlooked but impactful engineering issue

## 6. Training Monitoring and Logging

### Key Metrics

You should monitor these metrics during training:

**World model metrics:**
- `loss/recon`: reconstruction loss, should decrease steadily
- `loss/kl`: KL divergence, should fluctuate within reasonable range
- `loss/dyn`: dynamics loss (prior → posterior KL), reflects gap between RSSM's prior and posterior
- `loss/rep`: representation loss (posterior → prior KL), reflects encoder representation quality
- `loss/reward`: reward prediction loss, should decrease
- `loss/cont`: continuation prediction loss, should decrease

**Latent space health (critical!):**
- `posterior entropy`: posterior entropy, reflects latent state information content
- `prior entropy`: prior entropy, reflects world model prediction uncertainty

Many Dreamer training failures aren't about reward loss not decreasing, but **latent collapse** — latent space information content gradually collapses to near zero, posterior entropy continuously decreases. In this case, world model loss may look normal, but the policy can no longer extract useful information from latent state.

If latent collapse signs are detected:
- Note: a single metric cannot diagnose collapse; you need to jointly observe latent information content and task performance. During successful training, latent posterior may also become more deterministic — the key is whether this accompanies task performance degradation
- When adjusting KL free scale / free nats, balance reconstruction and dynamics learning; too large may cause prior to be unable to track posterior
- Check if encoder converges too early (recon loss too low may cause posterior and prior to become too close)

**Actor-Critic metrics:**
- `loss/actor`: Actor loss, should decrease steadily
- `loss/value`: Critic loss, should decrease steadily
- `loss/entropy`: policy entropy, should decrease slowly (exploration reducing)
- `stats/adv_max`, `stats/adv_min`: advantage range, shouldn't be too extreme
- `stats/ret_mean`: average return, should gradually increase

**System metrics:**
- `steps/sec`: training speed
- `gpu_mem`: memory usage
- `buffer_size`: replay buffer size

**Advanced diagnostic metrics:**

- **KL balance ratio**: `dynKL / repKL` ratio, reflects whether prior and posterior learning is balanced. If ratio severely deviates from 1, one side is having learning difficulties
- **Decoder prediction quality**: Don't just look at recon loss value; also visualize reconstruction image samples. If recon loss is decreasing but images are blurry or missing key details, encoder/decoder capacity may be insufficient
- **Imagined rollout sanity check**: This is a Dreamer-specific diagnostic method — directly visualize `latent rollout → decoded observation` to see if imagined "future frames" look reasonable. This is more intuitive than looking at loss values and can quickly reveal whether the world model has learned meaningful dynamics

### Using TensorBoard

DreamerV3 supports TensorBoard logging:

```bash
# Specify log directory when starting training
python dreamerv3/train.py --logdir ./logs/my_experiment

# Start TensorBoard
tensorboard --logdir ./logs
```

TensorBoard provides intuitive visualization of metric trends, convenient for debugging.

### Regular Checkpoint Saving

Regularly save models during training to prevent unexpected interruptions:

```yaml
# configs.yaml
save_every: 10000  # save every 10000 steps
```

Saved checkpoints can be used for:
- Resuming training from breakpoints
- Evaluating policy performance at different training stages
- Analyzing behavioral changes during training

## 7. Common Pitfalls and Solutions

### Pitfall 1: JAX Compilation is Slow

JAX uses the XLA compiler; the first run compiles the computation graph, which may take several minutes. This is normal; subsequent runs will be much faster.

**Solutions:**

- Be patient on the first run
- Use `jax.config.update("jax_compilation_cache_dir", "/path/to/cache")` to cache compilation results
- If frequent code changes cause recompilation, consider using `jax.jit`'s `static_argnums` parameter

### Pitfall 2: Multi-GPU Training

DreamerV3's official JAX reference implementation's default configuration primarily targets a single accelerator, but this doesn't mean multi-GPU is "not supported". JAX itself provides `jax.pmap`, `jax.shard_map`, PJRT, and other parallel mechanisms for multi-GPU scaling.

**Reality:**

- For most tasks, single GPU training is sufficient
- Official implementation doesn't provide out-of-the-box multi-GPU configuration; requires custom implementation based on JAX parallel APIs
- More practical approach is data parallelism: run different experiments on different GPUs (different hyperparameters or seeds)
- For scaling, refer to JAX's official [sharding guide](https://jax.readthedocs.io/en/latest/sharding.html)

### Pitfall 3: Replay Buffer Uses Lots of Memory

DreamerV3's replay buffer stores raw trajectory data (observation, action, reward, continuation, episode information), not trained latent sequences. Data goes through the encoder during training to get latents. But storing raw observations (especially pixel images) can still use a lot of memory.

**Solutions:**

```yaml
replay_size: 1e6   # control replay buffer size
```

- Use `replay_size` to limit buffer size
- If memory is tight, reduce `batch_length` to decrease sequence length
- Consider using disk storage instead of memory storage (requires code changes)

### Pitfall 4: Slow Training Speed

DreamerV3's training speed is affected by multiple factors:

**Possible causes:**
- Insufficient GPU performance
- Batch size too large
- Imagination length too long
- World model capacity too large

**Solutions:**
- Upgrade GPU
- Reduce batch size or imagination length
- Reduce RSSM `deter` and `hidden` dimensions
- Use smaller encoder/decoder networks

### Pitfall 5: symlog/symexp Numerical Issues

symlog and symexp are key to how DreamerV3 handles scale, but extreme input values can cause numerical issues.

**Solutions:**

- Ensure input data is properly normalized
- Check reward range; if rewards are particularly large, consider manual scaling
- If value prediction shows NaN, check if two-hot bin range is sufficient

## 8. Practical Experience Summary

### Tuning Priority

When tuning, try in this priority order:

1. **Environment interface correctness**: reward scale, action normalization, observation preprocessing, continuation settings — these errors directly cause training failure
2. **Replay / warmup configuration**: ensure replay buffer fills normally, warmup phase is reasonable
3. **Learning rate**: affects training stability; DreamerV3's default `4e-5` usually doesn't need major changes
4. **Batch size**: affects gradient estimate stability
5. **RSSM capacity**: `deter` and `hidden` dimensions, affecting world model representation ability
6. **Imagination length**: affects long-term credit assignment, but many failures aren't imagination length problems
7. **Entropy / exploration**: adjust through minstd/maxstd → unimix → actent order

Many training failures stem from basic issues like reward scale errors, action normalization errors, continuation configuration errors, not imagination length being too short. Debug basic configuration first, then tune hyperparameters.

### Training Checklist

Before starting training, check:

- [ ] GPU driver and CUDA correctly installed
- [ ] JAX can detect GPU
- [ ] Environment interface matches expectations (observation, action, reward shapes and types)
- [ ] Replay buffer fills normally
- [ ] World model loss starts decreasing in first few hundred steps
- [ ] Actor-Critic loss starts decreasing after world model stabilizes
- [ ] Memory usage is within safe range
- [ ] Training speed matches expectations

### When to Stop Training

Training isn't always better the longer it runs. Consider stopping when:

- Cumulative reward in real environment has saturated
- No significant improvement for several thousand steps
- Policy shows instability signs (reward fluctuates greatly)
- Training time has reached budget limit

## 9. Experiment Reproducibility Guide

Dreamer-type algorithms are highly dependent on experimental conditions; reproducibility is a common issue. When doing serious experiments, establish these habits:

**Seed management**

- Run at least 3-5 different seeds, report median and confidence intervals
- Seed should be recorded in experiment name, e.g., `dreamerv3_atari_pong_seed42`
- Note JAX's seed setting method: `jax.random.PRNGKey(seed)`

**Config saving**

- At the start of each experiment, save complete `configs.yaml` to logdir
- If command-line parameters override, also record the complete final configuration
- Recommend auto-saving in JSON or YAML format

**Checkpoint naming**

- Checkpoint filenames include step count, e.g., `checkpoint_100000.pkl`
- Keep checkpoints from multiple time points for retrospective analysis
- Record criteria for best checkpoint (eval reward, train reward, etc.)

**Git commit recording**

- Experiment code must have corresponding git commit hash
- If there are uncommitted changes, record `git diff` or `git stash`
- Recommend using experiment tracking tools (like MLflow, W&B) for automatic recording

**Hardware recording**

- Record GPU model, memory size, CUDA version, JAX version
- Training speed and final performance may differ across hardware
- For multi-machine experiments, ensure environment consistency

## 10. Connecting the Previous Articles

```text
World Model Intro → RSSM Deep Dive → RSSM Code Series (6 articles)
                                              ↓
                                     Dreamer Series #1: Overall Architecture
                                              ↓
                                     Dreamer Series #2: Actor-Critic
                                              ↓
                                     Dreamer Series #3: Training Tips (this article)
                                              ↓
                                     GPU Selection Guide
```

If you haven't read the first two, I recommend reading [Dreamer Overall Architecture](/en/articles/2026-08-25-dreamer-explained/) and [Actor-Critic Detailed Explanation](/en/articles/2026-08-27-dreamer-actor-critic/) before this training tips article — you'll get more out of it.

## 11. Summary

DreamerV3 training engineering practice can be summarized as:

- **Environment setup**: Correct JAX + GPU installation and memory management is foundational
- **Hyperparameter tuning**: Understand what each parameter does, adjust by priority
- **Stability debugging**: From world model to Actor-Critic, locate problems layer by layer
- **Task adaptation**: Different task types require different tuning strategies
- **Monitoring and logging**: Catch problems early, avoid wasting time

DreamerV3's design is already very robust; default configuration works on most tasks. But to achieve optimal performance, or when encountering training issues, you need deep understanding of each component's role to tune effectively.

Hope this practical guide helps you train DreamerV3 more smoothly. If you encounter problems, feel free to discuss in the comments.
