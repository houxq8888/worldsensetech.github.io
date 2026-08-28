---
title: "Dreamer 的 Actor-Critic：想象空间里的策略优化是怎么工作的？"
slug: "2026-08-27-dreamer-actor-critic"
date: 2026-08-27
draft: false
categories: ["世界模型"]
tags: ["DreamerV3", "世界模型", "RSSM", "Actor-Critic", "想象训练", "Dreamer系列"]
description: "从源码理解 Dreamer 的 Actor-Critic 设计：imagine loop、λ-return、two-hot value prediction 和 symlog 变换。"
toc: true
related_articles:
  - 2026-08-25-dreamer-explained
  - 2026-08-28-dreamerv3-training-tips
  - rssm-deep-dive
  - world-model-intro
  - vla-vs-world-model
  - 2026-08-30-dreamer-applications
---

> **Dreamer 系列 · 第 2 篇**
>
> 系列目录（当前在第 2 篇）：
> 1. [（一）读懂 Dreamer：世界模型是怎么学会'想象'的？](/zh/articles/2026-08-25-dreamer-explained/)
> 2. **[（二）Dreamer 的 Actor-Critic：想象空间里的策略优化](/zh/articles/2026-08-27-dreamer-actor-critic/)**

上一篇从架构层面讲清楚了 Dreamer 的整体设计：世界模型在隐空间中预测未来，策略在想象轨迹上学习。但这留下了一个核心问题：**策略具体怎么从想象轨迹中提取学习信号？**

这篇文章从 DreamerV3 源码出发，把 Actor-Critic 的完整工作链路拆开来看：imagination rollout 怎么生成特征、λ-return 怎么计算、two-hot value prediction 怎么工作、symlog 变换为什么不可或缺。

本文基于 [danijar/dreamerv3@e3f02248](https://github.com/danijar/dreamerv3) JAX reference implementation 开源快照，代码引用来自 `dreamerv3/agent.py`、`dreamerv3/rssm.py`、`embodied/jax/` 等文件。需要注意的是，开源实现与论文描述之间存在一些工程层面的差异，本文以源码行为为准。

## 一、从想象到策略：完整链路概览

先建立全局视角。Dreamer 的一次策略更新经过以下步骤：

```text
replay buffer 中的真实序列
         ↓
    encoder + posterior → 初始 latent state (h₀, z₀)
         ↓
    imagine loop（15 步）：
        actor → action → RSSM deterministic transition → prior stochastic state → next latent feature
         ↓
    （imagination 阶段没有 observation，stochastic state 完全由 RSSM prior rollout 产生）
         ↓
    每步的 latent feature → reward predictor / continuation predictor / value predictor
         ↓
    λ-return 计算 → Actor loss + Critic loss
         ↓
    梯度更新
```

这里的关键是：想象轨迹本身只是一串 latent features。要让策略从中学习，需要 reward model 给出即时反馈、continuation model 判断 episode 是否继续、value model 估计长期价值——三者共同构成 λ-return，然后 Actor 和 Critic 各自更新。

值得强调的是，imagination 的本质不是"预测未来用于展示"，而是**把世界模型转换成一个可生成大量低成本训练数据的 simulator**。Dreamer 最大的创新不在于 actor-critic 的形式，而在于这个数据生成范式：

```text
environment interaction
        ↓
world model（学会隐空间动力学）
        ↓
large-scale imagined transitions
        ↓
actor-critic learning
```

真实环境交互是昂贵的，但从一个学到的世界模型中可以采样出大量低成本的想象轨迹（受限于 imagination horizon、起点数量和模型容量）。策略优化不再完全受限于真实样本量——这是 Dreamer 系列区别于传统 RL 的核心思想。

## 二、Imagine loop：从 latent 到特征序列

imagination 的具体过程在 RSSM 的 `imagine()` 方法中实现（`rssm.py`）：

```python
def imagine(self, carry, policy, length, training, single=False):
    # single=True 时执行单步
    action = policy(sg(carry))
    actemb = nn.DictConcat(self.act_space, 1)(action)
    deter = self._core(carry['deter'], carry['stoch'], actemb)
    logit = self._prior(deter)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    carry = nn.cast(dict(deter=deter, stoch=stoch))
    feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
    return carry, (feat, action)
```

多步 rollout 通过 `nj.scan` 沿时间维度展开。每一步的输出是 feature（deterministic + stochastic state）和对应的 action。

在 `Agent.loss()` 中，imagination 的调用方式是：

```python
K = min(self.config.imag_last or T, T)
H = self.config.imag_length  # 默认 15
starts = self.dyn.starts(dyn_entries, dyn_carry, K)
policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
_, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
```

几个值得注意的实现细节：

**起点来自真实数据。** `starts` 是从 replay buffer 中采样的真实序列对应的 posterior state。这保证 imagination 从真实经验编码得到的 latent state 开始，而不是从随机 latent 初始化。

**imagination 的长度是超参数。** `imag_length: 15` 是 agent 的训练配置，不是 RSSM 的结构参数。常见官方配置（DreamerV1/V2/V3）均采用 15，但具体取值取决于任务。

**Actor-Critic loss 默认不更新 world model representation。** 源码中 `sg(first, skip=self.config.ac_grads)` 控制 imagined rollout 的起点是否阻断梯度（`ac_grads` 默认为 `False`）。默认配置下，actor-critic loss 不会通过 imagination dynamics 回传更新 RSSM、encoder 等世界模型参数。世界模型仍通过自身的 reconstruction loss、KL loss、reward prediction loss、continuation prediction loss 独立更新——stop-gradient 只是隔离了策略优化对世界模型的影响，世界模型本身一直在从真实数据中学习。

## 三、Actor 和 Critic 的网络结构

Actor 和 Critic 都使用 MLPHead 模板构建——各自独立的 MLP 骨干 + 任务特定的输出头，参数不共享。两者输入相同（都是 RSSM feature），但隐藏层完全独立。

```python
# Agent.__init__() 中
self.pol = embodied.jax.MLPHead(act_space, outs, **config.policy, name='pol')
self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
```

MLP 骨干的默认配置是 3 层、1024 单元、SiLU 激活、RMS 归一化：

```python
# configs.yaml
policy: {layers: 3, units: 1024, act: silu, norm: rms, minstd: 0.1, maxstd: 1.0, outscale: 0.01, unimix: 0.01}
value:  {layers: 3, units: 1024, act: silu, norm: rms, output: symexp_twohot, outscale: 0.0, bins: 255}
```

注意 Actor 和 Critic 的输出头不同：

**Actor** 的输出取决于动作空间类型。离散动作使用 categorical 分布，连续动作使用 squashed normal 分布——网络预测均值和标准差（标准差限制在 `[minstd, maxstd]` 范围内），采样后经 tanh 变换并缩放到环境 action range。

**Critic** 使用 `symexp_twohot` 输出——这是 DreamerV3 最关键的设计之一，后面单独展开。

## 四、λ-return：策略学习信号从哪来

Dreamer 不是直接累加想象轨迹上的即时奖励。它使用 λ-return 作为策略优化的目标——在偏差和方差之间取得平衡。

### λ-return 的递推公式

源码中的 `lambda_return()` 函数（`agent.py`）：

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

这个函数从想象轨迹的末端向前递推。简化形式为：

```text
R_t^λ = r_t + γ · ((1-λ) · V(s_{t+1}) + λ · R_{t+1}^λ)
```

其中 `γ` 是折扣因子，`λ` 控制 Monte Carlo 和 TD 之间的权衡。上式省略了 continuation factor。DreamerV3 实现中，continuation model 输出 `p(not terminal)`，该预测在 imagination 阶段被用于两个地方：一是转换成 `term = 1 - con`，控制 λ-return 是否继续 bootstrap；二是作为 discount weighting 的一部分（`weight = cumprod(disc * con)`），降低不可靠未来步骤的贡献。`term` 和 `con` 本质来自同一个 continuation prediction，只是用途不同。`last` 表示 imagined sequence 的边界（imagination horizon 截断），控制 λ 递推传播。当 continuation 接近 0 时，`term` 接近 1，bootstrap 被阻止，后续 return 不再传播。真实环境 terminal 信号只用于训练 continuation predictor。

### 默认配置中的折扣和 λ

```yaml
imag_loss: {slowtar: False, lam: 0.95, actent: 3e-4, slowreg: 1.0}
horizon: 333
contdisc: True
```

DreamerV3 不使用纯固定 discount。它将固定 horizon discount（`disc = 1 - 1/horizon ≈ 0.997`）与 learned continuation prediction 相乘，形成有效 discount（`weight = cumprod(disc * con)`），使价值传播同时考虑时间衰减和模型预测的 episode continuation。

`lam=0.95` 表示 return estimator 更偏向利用多步 imagined rollout 的实际回报，但仍保留 value bootstrap 来控制方差。λ=0 对应纯 TD(0)，λ=1 对应 Monte Carlo return，0.95 是高阶 TD mixture。

### continuation model 的作用

`con` 参数是 continuation model 的预测——`p(not terminal)`，即模型认为未来是否仍处于有效 rollout 状态。在 imagination rollout 中，该预测被用于两个地方：一是转换成 `term = 1 - con`，控制 λ-return 是否继续 bootstrap；二是作为 discount weighting 的一部分（`weight = cumprod(disc * con)`），降低不可靠未来步骤的贡献。如果 continuation prediction 接近 0，`term` 接近 1，bootstrap 被阻止，后续 return 不再传播。需要注意的是，真实环境的 terminal 信号只用于训练 continuation predictor，imagination 中的 term 和 con 都来自模型预测。这对机器人任务尤其重要：机械臂掉落或碰撞导致 episode 终止时，continuation 预测下降，价值估计必须知道这一点。

在调用 `imag_loss` 时：

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

## 五、Actor loss：REINFORCE 风格的策略梯度

在展开代码之前，先理解 Actor 为什么需要 Critic。如果直接使用 return 作为 policy gradient 的权重，相同状态下所有动作都会共享巨大的方差——好的动作和坏的动作都被同样的 return 值加权。value network 提供 baseline，advantage = return - value 衡量"这个动作比平均水平好多少"，从而大幅降低策略梯度的方差。

再看 Actor-Critic 的完整信息流。很多读者最大的误解是"Dreamer 用世界模型直接优化 action"，实际上信息流是这样的：

```text
imagined latent state z_t
          ↓
       policy (Actor)
          ↓
       action a_t
          ↓
       RSSM prior
          ↓
      imagined future z_{t+1}
          ↓
 reward predictor + value predictor
          ↓
     lambda return
          ↓
     advantage
          ↓
 policy gradient (update Actor)
```

策略网络从 imagined state 采样动作，动作经过 RSSM prior 产生 imagined future，reward 和 value predictor 在 imagined future 上给出反馈，λ-return 汇总成长期信号，advantage 评估这个动作比 baseline 好多少，最后策略梯度更新 Actor。**世界模型不直接输出 action——它提供训练数据，策略网络从中学习。**

`imag_loss` 中的 Actor 损失计算：

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

这段代码做了以下几件事：

**1. 计算 advantage。** `adv = (ret - tarval) / rscale`——λ-return 减去 baseline（value prediction），再除以 return 的标准差做归一化。

**2. 归一化 advantage。** 通过 `advnorm` 进一步做标准化，减去均值除以标准差。

**3. 策略梯度。** `logpi * stop_gradient(adv_normed)`——这是标准的 REINFORCE 风格：对数概率乘以归一化的优势估计。优势估计做了 stop-gradient，不回传到 value network。

**4. 熵正则。** `actent * entropy`，默认 `actent=3e-4`，鼓励策略保持探索。

**5. 时间步权重。** `weight = cumprod(disc * con)`——每个时间步的 loss 按折扣因子和 continuation 概率加权。越远的未来步骤权重越低，episode 终止后权重归零。

注意这里的一个重要设计：**DreamerV3 默认 Actor 更新以 advantage-weighted log probability 为主体，即 REINFORCE 风格的 score-function gradient。** 源码中保留了让 Actor-Critic loss 穿过 imagination dynamics 的可能性，通过 `ac_grads` 控制 imagined rollout 起点是否阻断梯度。默认关闭，使 Actor-Critic 更新不影响 encoder/RSSM latent state。但 DreamerV3 默认 Actor objective 本身仍然是 advantage-weighted log probability（score-function style），而不是依赖 action gradient 的纯 dynamics backprop。Actor 的梯度通过 `logpi` 回传到 policy 网络，但 advantage 是 stop-gradient 的。而 entropy term 的梯度直接对 policy 参数求导，不依赖 score-function estimator。

需要澄清的是，Dreamer 系列并非从 V2 才开始使用 REINFORCE。实际上 DreamerV1 已经同时探索了两种梯度路径：

* **dynamics gradients（pathwise / analytic gradients）**：梯度直接穿过 RSSM 的可微动力学
* **reinforce gradients**：通过 log π(a|s) × advantage 的 score-function 梯度

DreamerV1 论文中两种都有实现和对比。DreamerV2 主要使用 analytic gradients through stochastic latent imagination。到了 DreamerV3，源码默认 `ac_grads=False`，意味着 actor loss 不穿过 imagination dynamics，使用 score-function objective，但保留接口。DreamerV3 的选择不是因为 dynamics gradient 理论上不可行，而是在大规模、多任务训练设置下，score-function objective 更容易获得稳定行为——离散 categorical latent 使 pathwise gradient 变得更加困难，更长 imagination horizon 下 dynamics gradient 对模型误差更敏感，这些都是综合考量。

### 为什么不直接 backprop through model？

一个自然的问题是：既然世界模型可微，为什么不直接用 `∇_a (r + γV)` 反传到 action，让 Actor 最大化 imagined reward？

Dreamer 早期版本确实尝试过这种 dynamics gradient：

```text
policy → action → RSSM → reward → gradient
```

优点是样本效率高——梯度直接穿过模型，路径短。但缺点也很明显：对 model error 敏感，长 horizon 上 gradient 容易不稳定。世界模型的预测误差会沿着反向传播路径放大，导致策略更新方向偏离。

DreamerV3 默认配置主要采用 score-function style objective：

```text
policy → sample action → log probability → advantage
```

同时代码保留 pathwise dynamics gradient 选项（通过 `ac_grads` 控制）。默认关闭以牺牲一点模型梯度的利用，换取训练稳定性。advantage 做了 stop-gradient，不经过世界模型反传——策略更新只依赖"这个动作在想象轨迹上表现如何"，而不是"世界模型预测这个动作会导致什么精确状态"。

## 六、Critic loss：two-hot 价值预测

Critic 的损失计算紧接 Actor：

```python
voffset, vscale = valnorm(ret, update)
tar_normed = (ret - voffset) / vscale
tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)

losses['value'] = sg(weight[:, :-1]) * (
    value.loss(sg(tar_padded)) +
    slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]
```

这里的关键是 `value.loss()`——它不是简单的 MSE，而是 two-hot categorical 的交叉熵损失。

### Two-hot 编码的原理

DreamerV3 的 Critic 不直接输出一个标量 value，而是在一组离散的 bin 上预测价值分布。默认配置使用 255 个 bin，范围覆盖极大的动态区间。

给定一个目标值 `v`，two-hot 编码找到相邻的两个 bin `b_below` 和 `b_above`，按距离反比分配权重：

```text
weight_below = (b_above - v) / (b_above - b_below)
weight_above = (v - b_below) / (b_above - b_below)
```

然后对预测的 logits 做交叉熵损失。这是一种 distributional regression：网络预测离散 support 上的概率分布，而 continuous scalar target 通过邻近两个 bin 的线性插值得到 soft label。

源码中 `TwoHot.loss()`（`embodied/jax/outs.py`）：

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

### symexp_twohot：bin 的构造

bin 的间距不是均匀的。`symexp_twohot` 头在 `[-20, 0]` 上均匀取点，然后做 symexp 变换得到指数级间距的 bin，再镜像到正半轴：

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

255 个 bin 在 symlog domain 中均匀分布，经过 symexp 后映射到原始尺度，因此在 raw value space 中呈指数间隔。理论覆盖范围达到约 ±4.8×10⁸，但实际训练主要集中在 symlog 空间附近，这些极端 bin 更多用于提高异常尺度下的鲁棒性——因为 `TwoHot.loss()` 内部先做 `target = sg(self.squash(target))`，即 symlog 变换。**bins 覆盖的是 symlog 空间，而不是原始 value 空间**。因此 two-hot loss 实际工作空间是 symlog(value)，网络主要学习压缩后的价值分布。

预测时取分布的期望值。源码中 `TwoHot.pred()` 由于 bins 关于 0 对称，利用对称结构计算期望值——对奇数 bins，中间 bin 单独处理，两侧对称相加，使初始化 logits 均匀时预测 value 为 0。

### Slow value network

Critic 损失中还有一个 `slowreg` 项：

```python
slowreg * value.loss(sg(slowvalue.pred()))
```

`slowvalue` 是一个 EMA（指数移动平均）更新的 target value network：

```yaml
slowvalue: {rate: 0.02, every: 1}
```

每步以 `rate=0.02` 的速率将在线 value network 的参数滑动平均到 slow network。`slowreg=1.0` 要求在线 value network 的预测同时拟合 λ-return target 和 slow network 的预测。它更像一个 EMA teacher 提供稳定的价值参考，而不是传统 TD 算法中的 target network（如 DQN 中周期性硬拷贝的那种）。

## 七、DreamerV3 的尺度鲁棒性设计

DreamerV3 能在 Atari、MuJoCo、机器人操控等差异极大的任务上用同一套超参数训练，靠的不是单一技巧，而是多层尺度归一化设计的组合。

### symlog 变换

首先看基础的数学工具：

```python
# embodied/jax/nets.py
def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))
```

symlog 把大数值压缩、保留小数值精度。这两个函数互为反函数：`symexp(symlog(x)) = x`。

### Reward 和 Value 都用 two-hot head

一个常见的误解是"reward 用 symlog 回归，value 用 two-hot 分布"。实际上，**DreamerV3 中 reward 和 value 预测都采用 two-hot distribution head**。区别在于 target 的处理方式：

* **Reward prediction**：reward head 使用 two-hot distribution，target 直接做 symlog 变换（`reward → symlog → two-hot`），使不同任务的即时奖励落在相近的数值范围
* **Value prediction**：λ-return 首先经过 percentile-based return normalization，然后进入 value head 的 two-hot loss。two-hot loss 内部再做 symlog squash（`λ-return → return normalization → symlog squash → two-hot`），进一步压缩动态范围

两类 head 都使用 two-hot distribution，但 target 的尺度处理路径不同：reward 直接在 symlog reward 空间训练，value 则在 return normalization 后的 symlog 空间训练。需要注意的是，reward normalization 和 return normalization 不是同一个机制——前者处理即时奖励的尺度，后者处理长期累积回报的尺度。网络输出 logits，经 softmax 得到离散分布，取期望后做 symexp 反变换回到原始尺度——反变换主要用于得到可解释的预测值，训练 loss 本身是在 symlog support 上计算的。

### Return normalization

除了 symlog 和 two-hot，DreamerV3 还对 λ-return 做 percentile-based normalization：

```yaml
retnorm: {impl: perc, rate: 0.01, limit: 1.0, perclo: 5.0, perchi: 95.0, debias: False}
```

这使用 5th 和 95th 百分位数做归一化：

```python
roffset, rscale = retnorm(ret, update)
adv = (ret - tarval[:, :-1]) / rscale
```

advantage 先除以 return 的 scale，再做进一步的 mean-std normalization。这确保了不同任务、不同训练阶段的 advantage 在相似的数值范围内，避免梯度爆炸或消失。

### 三层设计如何配合

```text
reward
  ↓
symlog 压缩
  ↓
reward two-hot head

reward + continuation
  ↓
λ-return
  ↓
return normalization
  ↓
symlog squash（two-hot loss 内部）
  ↓
value two-hot head
```

symlog 解决即时奖励的尺度差异，return normalization 解决长期累积回报的尺度变化，two-hot distribution 提供灵活的价值分布表征。三者共同使同一套超参数能跨任务稳定训练。

## 八、Dreamer vs MPC：amortized decision making

熟悉模型预测控制（MPC）的读者可能会问：既然有了世界模型，为什么不直接每一步都搜索最优动作序列？

两者的核心区别在于：

* **MPC 是 online planning**：每次决策都在世界模型中重新搜索最优动作序列，选择最优的执行一步，然后重新搜索。
* **Dreamer 是 amortized planning**：先在想象轨迹上训练好 Actor 网络，部署时只需要一次 policy forward pass 即可输出动作，不需要再做模型搜索。

两者优化对象也不同：MPC 在线优化 action sequence，Dreamer 离线优化 policy parameters。

MPC 的优点是不需要训练策略网络，缺点是每次决策都需要大量搜索，计算成本高。Dreamer 将在线规划过程转化为一次训练阶段完成的策略优化，部署效率更高，在需要高频实时决策和大规模策略学习时具有优势。更严格地说，Dreamer 将模型中的未来预测转化为训练数据，通过策略网络摊销（amortize）未来决策，而不是每次执行时重新搜索动作序列——这是 amortized model-based reinforcement learning，比 planning 更准确的定位。不过 MPC 在操控、运动控制和安全关键场景中仍然非常重要，两者并非替代关系。

## 九、完整的训练循环

把以上所有组件放在一起，Dreamer 的完整训练循环是：

```text
真实环境执行 → 收集 (obs, action, reward) → 存入 replay buffer
         ↓
采样序列 → encoder + RSSM observe → 世界模型训练
    （reconstruction loss + KL loss + reward/continuation prediction loss）
         ↓
采样 posterior state → imagine loop（15 步）
         ↓
每步 feature → reward/continuation/value prediction
         ↓
λ-return → Actor loss（REINFORCE + entropy）+ Critic loss（two-hot CE + slowreg）
         ↓
梯度更新 Actor + Critic
         ↓
Actor 回到真实环境执行 → 新一轮数据采集
```

这个世界模型不是物理世界的完整模拟器——它学习的是任务相关的隐空间动力学。Dreamer 的核心不是"想象得多逼真"，而是"想象得是否足以支持正确决策"。

## 十、关键超参数速查

从 `configs.yaml` 中提取的 Actor-Critic 相关默认配置：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `imag_length` | 15 | 想象轨迹步数 |
| `horizon` | 333 | 连续折扣 horizon（disc ≈ 0.997） |
| `lam` | 0.95 | λ-return 的 λ |
| `actent` | 3e-4 | 熵正则系数 |
| `slowreg` | 1.0 | slow value network 正则权重 |
| `slowvalue.rate` | 0.02 | EMA 更新速率 |
| `policy.layers/units` | 3/1024 | Actor MLP 结构 |
| `value.layers/units` | 3/1024 | Critic MLP 结构 |
| `value.bins` | 255 | two-hot bin 数量 |
| `value.output` | symexp_twohot | value 输出分布类型 |
| `lr` | 4e-5 | 默认 optimizer learning rate |
| `retnorm` | perc (5-95) | return 归一化方式 |

## 十一、把之前的文章串起来

```text
世界模型入门 → RSSM 深度解析 → RSSM 代码系列（6篇）
                                       ↓
                              Dreamer 系列 #1：整体架构
                                       ↓
                              Dreamer 系列 #2：Actor-Critic（本篇）
                                       ↓
                              DreamerV3 训练技巧 → GPU 选型
```

如果你刚接触世界模型，建议从 [什么是机器人世界模型？](/zh/articles/world-model-intro/) 开始。

如果你想看代码级别的 RSSM 拆解，[RSSM 代码解析系列](/zh/articles/2026-08-19-rssm-code-walkthrough/) 从 stochastic state 一路讲到 KL balancing 和 imagine reset。

如果你对训练过程中的实际问题感兴趣，[DreamerV3 训练技巧](/zh/articles/dreamerv3-training-tips/) 总结了从环境配置到超参调优的实战经验。

## 十二、总结

Dreamer 的 Actor-Critic 设计可以概括为：

* **Actor** 使用基于 imagined trajectory 的 REINFORCE 风格策略梯度优化策略。由于 rollout 由世界模型从 replay buffer 的 latent states 出发生成（policy-conditioned imagination trajectories），而非真实环境在线采集，优势估计经过归一化，配合熵正则鼓励探索。
* **Critic** 使用 two-hot categorical 分布预测价值，配合 symexp-spaced bins 覆盖极大动态范围。slow value network 提供稳定的价值目标。
* **λ-return** 在 Monte Carlo（想象轨迹上的实际回报）和 TD（Critic 的 value bootstrap）之间取得平衡，默认 λ=0.95 偏向多步回报。
* **symlog + two-hot + return normalization** 共同构成多层尺度鲁棒性设计，是 DreamerV3 跨任务稳定训练的关键工程贡献。reward 和 value 都使用 two-hot distribution head，区别在于 reward target 经过 symlog 压缩，value target 经过 return normalization。

这些设计让 DreamerV3 成为目前影响力较大的通用 model-based RL 开源实现之一——同一套超参数能在 Atari 游戏、MuJoCo 控制和机器人操控等差异极大的任务上工作。

下一篇我们可能会讨论 DreamerV3 的训练工程实践——从 GPU 配置到超参调优的实战经验。
