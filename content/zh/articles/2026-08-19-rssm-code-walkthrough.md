---
title: "从代码理解 RSSM：DreamerV3 中 RSSM 的实现细节"
slug: "2026-08-19-rssm-code-walkthrough"
date: 2026-08-19
draft: false
categories: ["世界模型"]
tags: ["RSSM", "DreamerV3", "世界模型", "状态空间模型", "代码解析"]
description: "从 DreamerV3 官方源码出发，把 RSSM 的计算路径和数学公式一一对应：categorical latent、Block GRU、双 KL balancing、observe 与 imagine。"
toc: true
---

前面两篇文章分别介绍了 RSSM 的基本原理和世界模型的发展路线。理论解决的是"RSSM 为什么这样设计"，但真正读 DreamerV3 源码时，会发现还有很多问题：

* 为什么 stochastic state 不是一个普通的 Gaussian 向量？
* GRU 到底吃的是 observation，还是上一时刻的 latent state？
* prior 和 posterior 在代码里分别对应什么？
* DreamerV3 为什么要做两次 KL？
* `stop_gradient` 到底在 KL balancing 里起什么作用？
* imagination 阶段没有 observation，RSSM 又是怎么跑起来的？
* `stoch=32, classes=64` 到底意味着什么？

这篇文章不再从一个通用的 Gaussian RSSM 伪代码开始，而是**直接从 DreamerV3 官方实现出发**，把源码中的计算路径和 RSSM 的数学公式一一对应起来。

> 本文以 DreamerV3 官方仓库中的 `dreamerv3/rssm.py` 和默认配置为参考。为了方便阅读，下面会对 JAX、Ninjax、scan、dtype 等工程代码做适当简化，但不会改变核心计算逻辑。

---

## 一、先看 RSSM 在 DreamerV3 中的位置

DreamerV3 可以粗略理解成：

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
    ├──► Decoder：重建 observation
    ├──► Reward Head：预测 reward
    ├──► Continue Head：预测 episode 是否继续
    └──► Actor / Critic：想象轨迹上的策略与价值
```

RSSM 是整个世界模型的核心。

它要解决的问题其实可以概括成一句话：

> **根据历史 latent state 和 action，维护一个可以不断向未来滚动的隐状态。**

这个隐状态由两部分组成：

```text
s_t = (h_t, z_t)

h_t：deterministic state
z_t：stochastic state
```

其中：

* `h_t` 负责保存长期的时序上下文；
* `z_t` 负责表示当前状态中具有随机性的部分。

DreamerV3 官方代码中，RSSM 的状态空间就是：

```python
deter = [B, deter]
stoch = [B, stoch, classes]
```

也就是说，`stoch` **不是一个普通的一维 Gaussian 向量**，而是多个 categorical 随机变量。

---

## 二、DreamerV3 的 stochastic state 到底是什么？

这是理解 DreamerV3 RSSM 最重要的一步。

很多 RSSM 教程会写成：

```text
z_t ~ Normal(μ_t, σ_t)
```

然后：

```python
z = mean + std * eps
```

这种写法适合解释一些经典的连续 RSSM，但**不能直接当成 DreamerV3 的实现**。

DreamerV3 使用的是 categorical latent。

官方默认配置中：

```yaml
rssm:
  deter: 8192
  hidden: 1024
  stoch: 32
  classes: 64
```

因此：

```text
stoch = 32
classes = 64
```

意味着：

> 每一个时间步有 32 个 categorical 随机变量，每个变量有 64 个类别。

所以 stochastic state 的形状是：

```text
[B, 32, 64]
```

而不是：

```text
[B, 32]
```

如果把它展平：

```text
32 × 64 = 2048
```

因此 stochastic state 在进入 decoder 等模块之前，可以理解成一个 2048 维的 one-hot / straight-through 表示。

官方源码中的定义也非常直接：

```python
@property
def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))
```

也就是说，RSSM 的完整 latent state 是：

```text
deter:  [B, 8192]
stoch:  [B, 32, 64]
```

这里的 8192 是默认大模型配置，并不是 RSSM 固定使用 8192。

DreamerV3 提供了多个模型规模，例如：

```text
1M   → deter 512
12M  → deter 2048
25M  → deter 3072
50M  → deter 4096
100M → deter 6144
200M → deter 8192
400M → deter 12288
```

因此，在讲 DreamerV3 时，更准确的说法应该是：

> `deter` 和 `classes` 会随着模型规模变化，`stoch=32` 在默认配置中保持不变。

---

## 三、Observe：真实观测进入 RSSM 后发生了什么？

理解 DreamerV3 RSSM 最好的方法，是直接追踪 `observe()`。

简化之后，它的核心流程可以写成：

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

这里有一个非常重要的地方：

**GRU 并不是直接读取当前 observation embedding。**

它读取的是：

```text
上一时刻 deter
上一时刻 stoch
当前 action
```

然后得到：

```text
当前 deter
```

也就是：

```text
(h_{t-1}, z_{t-1}, a_{t-1})
            │
            ▼
          GRU
            │
            ▼
           h_t
```

之后才使用当前 observation embedding 来推断：

```text
q(z_t | h_t, o_t)
```

这正是 RSSM 中 deterministic transition 和 stochastic inference 的分工。

---

## 四、RSSM 的真正递推关系

把代码翻译成数学公式，大致就是：

### 1. Deterministic transition

```text
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
```

DreamerV3 中 `f` 是一个 GRU-like transition。

注意：

```text
不是：

h_t = GRU(h_{t-1}, observation_t, action_t)

而是：

h_t = GRU(h_{t-1}, z_{t-1}, action_{t-1})
```

这是理解 RSSM 非常关键的一点。

### 2. Posterior

拿到当前 observation encoder 的输出 `o_t` 后：

```text
q(z_t | h_t, o_t)
```

DreamerV3 代码中会把：

```text
h_t + observation embedding
```

送入 observation model，最后产生：

```text
logits
```

然后把 logits 转换成 categorical distribution。

### 3. Prior

另一方面，只使用 deterministic state：

```text
p(z_t | h_t)
```

也就是：

```python
prior_logit = prior(deter)
```

于是同一个 `h_t` 同时对应两个分布：

```text
                 ┌──► posterior q(z_t | h_t, o_t)
h_t ─────────────┤
                 └──► prior     p(z_t | h_t)
```

这就是 RSSM 的核心。

---

## 五、为什么需要 prior 和 posterior 两套分布？

可以从"训练"和"想象"两个阶段理解。

### 训练阶段

训练时我们有真实 observation：

```text
o_1, o_2, o_3, ...
```

所以可以使用：

```text
q(z_t | h_t, o_t)
```

也就是 posterior。

它看到了真实 observation，因此可以得到更准确的 latent state。

### 想象阶段

到了 Dreamer 的 imagination：

```text
observation 没了
```

我们只有：

```text
当前 latent state
+
action
```

所以不能再计算：

```text
q(z_t | h_t, o_t)
```

只能使用：

```text
p(z_t | h_t)
```

也就是 prior。

因此：

```text
训练：

observation
    ↓
posterior
    ↓
z_t


想象：

h_t
 ↓
prior
 ↓
z_t
```

训练的目标之一，就是让 prior 逐渐学会逼近 posterior。

这样模型才能做到：

> 训练时借助真实 observation 学习世界状态，想象时脱离 observation 仍然能够预测未来。

这正是世界模型能够进行 imagination 的基础。

---

## 六、DreamerV3 为什么使用 categorical latent？

现在来看 `_logit()`：

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

假设：

```text
stoch = 32
classes = 64
```

那么 Linear 最终输出：

```text
32 × 64 = 2048
```

然后 reshape 成：

```text
[32, 64]
```

于是每个 stochastic variable 都拥有一个 64 类 categorical distribution。

可以理解成：

```text
z_1 → 64 classes
z_2 → 64 classes
z_3 → 64 classes
...
z_32 → 64 classes
```

最终：

```text
z = [z_1, z_2, ..., z_32]
```

---

## 七、Categorical sampling 是怎么做的？

源码里：

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

这里的 `OneHot` 非常关键。

它并不是：

```text
Gaussian sampling
```

而是：

```text
Categorical distribution
→ sample one category
→ represent it as one-hot
```

例如某个 categorical variable：

```text
[0.05, 0.10, 0.70, 0.15]
```

采样之后可能得到：

```text
[0, 0, 1, 0]
```

32 个变量分别采样后：

```text
[32, 64]
```

就是当前 stochastic state。

---

## 八、为什么代码里还有 unimix？

DreamerV3 默认：

```yaml
unimix: 0.01
```

它的作用可以简单理解成：

> 不让 categorical distribution 过早变得过于尖锐。

假设模型预测：

```text
[0.999, 0.001, 0, 0, ...]
```

这种分布可能导致某些类别概率迅速接近 0。

unimix 会把分布和均匀分布做一点混合，让每个类别保留最小概率。

直观上：

```text
原始 distribution
       │
       ▼
  + 少量 uniform
       │
       ▼
更平滑的 categorical distribution
```

这里的 `0.01` 就是默认混合比例。

---

## 九、DreamerV3 的 GRU 其实不是普通的 GRUCell

如果你原来习惯 PyTorch：

```python
nn.GRUCell(...)
```

那么看 DreamerV3 源码会发现：

**它没有直接调用一个标准 GRUCell。**

而是在 `_core()` 中自己构造了 GRU。

核心逻辑类似：

```python
reset, cand, update = split(x)

reset = sigmoid(reset)
cand = tanh(reset * cand)

update = sigmoid(update - 1)

deter = update * cand + (1 - update) * deter
```

也就是说，它自己计算 GRU 的 gates。

这也是为什么不能简单地把 DreamerV3 的 RSSM 理解成：

```python
self.gru = nn.GRUCell(...)
```

源码中还加入了一个非常重要的设计：

```text
Block GRU
```

---

## 十、为什么要 Block GRU？

DreamerV3 默认：

```yaml
blocks: 8
```

而：

```text
deter = 8192
```

因此 deterministic state 会被拆成多个 block。

可以粗略理解为：

```text
8192
 ↓
8 个 block
 ↓
每个 block 1024
```

源码中：

```python
flat2group = lambda x: einops.rearrange(
    x,
    '... (g h) -> ... g h',
    g=g
)
```

就是在做这种分组。

之后通过：

```python
nn.BlockLinear
```

进行 block-wise transformation。

这样设计的目的，是在保持大 deterministic state 表达能力的同时，控制计算和参数规模。

因此，DreamerV3 的 deterministic path 并不是简单的：

```text
8192-dimensional GRU
```

而是一个带 block structure 的 GRU-like transition。

这也是源码层面非常值得关注的工程设计。

---

## 十一、`_core()` 到底在计算什么？

把源码大量的网络细节去掉，可以把 `_core()` 简化成：

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

所以它真正表达的是：

```text
上一时刻：

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

这一步**完全不需要当前 observation**。

当前 observation 是 posterior 的输入，而不是 deterministic transition 的直接输入。

---

## 十二、Observe 阶段的完整数据流

现在把前面的东西串起来：

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

这里可以看到：

**RSSM 实际上存在两条 stochastic path：**

```text
Prior path
h_t → p(z_t | h_t)

Posterior path
h_t + observation → q(z_t | h_t, observation)
```

训练时利用 posterior 得到真实轨迹上的 latent state，同时用 KL 约束 prior。

---

## 十三、KL balancing：DreamerV3 最容易被讲错的地方

现在来看官方 loss：

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

这里最重要的是：

```python
sg(...)
```

也就是：

```text
stop_gradient
```

### 1. Dynamics loss

第一项：

```python
dyn = KL(
    sg(posterior)
    || prior
)
```

posterior 被 stop-gradient。

因此优化这项时：

```text
posterior：作为目标
prior：被训练
```

也就是说：

> **让 prior 去追 posterior。**

### 2. Representation loss

第二项：

```python
rep = KL(
    posterior
    || sg(prior)
)
```

这一次 prior 被 stop-gradient。

因此：

```text
prior：作为目标
posterior：被训练
```

也就是说：

> **让 posterior 的表示不要偏离 prior 太远。**

### 3. 为什么需要两项？

如果直接：

```text
KL(posterior || prior)
```

两个网络都会收到梯度。

这样很难控制：

> 到底是谁应该向谁靠近？

KL balancing 就通过 stop-gradient 明确指定：

```text
dyn：
prior → posterior

rep：
posterior → prior
```

因此它并不是简单的：

```text
KL × 一个系数
```

而是通过两个方向不同的 KL loss，分别约束 dynamics model 和 representation。

---

## 十四、Free Nats 又是什么？

官方配置：

```yaml
free_nats: 1.0
```

代码：

```python
if self.free_nats:
    dyn = jnp.maximum(dyn, self.free_nats)
    rep = jnp.maximum(rep, self.free_nats)
```

也就是：

```text
dyn = max(dyn, 1.0)
rep = max(rep, 1.0)
```

这里的思想不是：

> "每个 stochastic dimension 必须保存至少 1 bit 信息。"

更准确地说：

> 给 KL 提供一个免费区间。当 KL 足够小时，不继续通过 KL 项施加强约束。

这样可以避免模型在训练早期就过度追求：

```text
posterior ≈ prior
```

从而让 posterior 失去利用 observation 的能力。

---

## 十五、KL balancing 的最终权重

默认配置里：

```yaml
loss_scales:
  dyn: 1.0
  rep: 0.1
```

因此 RSSM 的 KL 相关损失可以理解成：

```text
L_RSSM
=
1.0 × dyn
+
0.1 × rep
```

再结合：

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

这比简单说：

> "DreamerV3 用一个 KL loss 让先验逼近后验"

要准确得多。

---

## 十六、Imagine：没有 observation 时怎么办？

这是 Dreamer 最漂亮的地方。

训练阶段：

```text
observation → posterior → z
```

但 imagination 阶段：

```text
没有 observation
```

所以只能使用：

```text
prior
```

源码中的核心逻辑可以简化为：

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

注意：

**这里没有把 observation embedding 填零。**

这是对原稿非常重要的一处修正。

因为 DreamerV3 的 `_core()` 本来就不依赖当前 observation。

所以 imagination 的计算路径天然就是：

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

然后继续：

```text
(h_{t+1}, z_{t+1})
      +
    action_{t+1}
      ↓
...
```

这样就可以在完全没有真实 observation 的情况下生成一条 latent trajectory。

---

## 十七、Imagine 为什么可以用于规划？

假设当前真实环境已经给了我们：

```text
(h_t, z_t)
```

接下来 Actor 产生：

```text
a_t
```

RSSM 就可以预测：

```text
(h_{t+1}, z_{t+1})
```

然后 reward head 预测：

```text
r_{t+1}
```

value head 预测：

```text
V_{t+1}
```

于是整个过程变成：

```text
当前真实状态
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
          下一 latent
              │
              ▼
             ...
```

这就是 Dreamer 的 imagination。

它不需要真的与环境交互，就可以在 learned world model 中"想象"未来。

---

## 十八、为什么 RSSM 的 imagination 误差会累积？

因为 imagination 是递归的。

假设：

```text
z_1 = 真实状态
```

那么：

```text
z_2 = model(z_1, a_1)

z_3 = model(z_2, a_2)

z_4 = model(z_3, a_3)
```

如果：

```text
z_2
```

已经存在误差，那么：

```text
z_3
```

是在一个有误差的状态上继续预测。

所以：

```text
一步预测误差
      ↓
进入下一步
      ↓
继续累积
      ↓
长 rollout 逐渐偏离真实环境
```

这也是为什么 DreamerV3 不会简单地依赖无限长 imagination，而是使用有限的 imagination horizon。

默认配置中：

```yaml
imag_length: 15
```

也就是一次 imagination rollout 通常只向前展开有限步数。

---

## 十九、RSSM 的 sequence training

实际训练不会一次只处理一个时间步。

假设：

```text
batch_size = B
sequence_length = T
```

那么输入大致是：

```text
tokens: [B, T, embed_dim]
actions: [B, T, action_dim]
```

RSSM 沿时间维递推：

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

DreamerV3 使用 JAX/Ninjax 的 `scan` 来完成这个时间维度上的递推。

代码结构可以简化成：

```python
carry, entries = scan(
    rssm_step,
    carry,
    (tokens, actions, resets)
)
```

这里的 `carry` 就是：

```text
{
    deter,
    stoch
}
```

因此 RSSM 的状态不会在每一个时间步重新初始化，而是在整个 sequence 中不断传递。

---

## 二十、Reset 为什么也进入 RSSM？

源码中的 `_observe()` 开头有：

```python
deter, stoch, action = nn.mask(
    (carry['deter'], carry['stoch'], action),
    ~reset
)
```

这说明 episode 结束之后，不能继续把上一 episode 的 latent state 带到下一 episode。

否则：

```text
Episode A latent
       ↓
Episode B
```

两个完全不同的 episode 就会发生状态污染。

所以 reset 时：

```text
deter → reset
stoch → reset
action → reset
```

然后从新的 episode 状态重新开始。

这也是实际实现和理论公式之间经常被忽略的工程细节。

---

## 二十一、DreamerV3 默认 RSSM 配置

根据官方配置，默认的大模型 RSSM 参数是：

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

因此可以把几个关键参数理解成：

| 参数 | 含义 |
|:---|:---|
| `deter` | deterministic state 的维度 |
| `hidden` | RSSM 内部 MLP hidden dimension |
| `stoch` | categorical variable 数量 |
| `classes` | 每个 categorical variable 的类别数 |
| `unimix` | categorical distribution 的均匀混合比例 |
| `blocks` | Block GRU 的分组数量 |
| `free_nats` | KL 的 free-nats 阈值 |
| `imglayers` | prior 网络层数 |
| `obslayers` | posterior 网络层数 |
| `dynlayers` | dynamics 网络层数 |

这里尤其容易搞混：

```text
stoch = 32
```

并不意味着 stochastic state 是 32 维。

实际是：

```text
32 × 64 = 2048
```

维 categorical representation。

---

## 二十二、把整个 RSSM 压缩成一张图

到这里，可以把 DreamerV3 RSSM 总结成：

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

训练阶段：

```text
真实 observation
      ↓
posterior
      ↓
z_t
```

想象阶段：

```text
没有 observation
      ↓
prior
      ↓
z_t
```

而两者通过 KL balancing 联系起来。

---

## 二十三、从代码角度重新理解"世界模型"

现在再回头看 DreamerV3 的世界模型，会发现 RSSM 实际上做了三件事情。

### 第一件事：记忆

```text
(h_{t-1}, z_{t-1}, a_{t-1})
                ↓
               h_t
```

deterministic state 负责维护历史上下文。

### 第二件事：推断

```text
(h_t, observation_t)
          ↓
        posterior
          ↓
         z_t
```

有真实 observation 时，模型可以修正自己的状态估计。

### 第三件事：预测

```text
(h_t, z_t, action_t)
          ↓
        prior
          ↓
     (h_{t+1}, z_{t+1})
```

没有 observation 时，模型可以依靠自己学习到的 dynamics 向未来滚动。

所以 RSSM 本质上统一了：

```text
State Estimation
        +
World Dynamics
```

而 Dreamer 的 imagination 又进一步把：

```text
World Dynamics
        ↓
Future Prediction
        ↓
Planning / Policy Learning
```

连接起来。

---

## 二十四、最后重新看 RSSM 的核心公式

如果把 DreamerV3 的实现压缩成最核心的数学形式，可以写成：

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

然后：

```text
L_KL =
λ_dyn L_dyn
+
λ_rep L_rep
```

默认：

```text
λ_dyn = 1.0
λ_rep = 0.1
```

最终 RSSM 训练出来以后：

```text
posterior
    ↓
真实轨迹上的 latent

prior
    ↓
没有 observation 时的 latent prediction
```

这就是 DreamerV3 世界模型可以进行 imagination 的关键。

---

## 二十五、小结

如果只记住 DreamerV3 RSSM 的几个关键点，可以记成下面这张表：

| 组件 | 作用 | DreamerV3 实现 |
|:---|:---|:---|
| `deter` | 保存时序上下文 | Block GRU |
| `stoch` | 表示随机 latent | categorical |
| `classes` | 每个 categorical 的类别数 | 默认 64 |
| Prior | 没有 observation 时预测 latent | `p(z_t \| h_t)` |
| Posterior | 有 observation 时推断 latent | `q(z_t \| h_t, o_t)` |
| `unimix` | 防止 categorical 过早尖锐 | 默认 0.01 |
| `dyn KL` | 训练 prior | posterior stop-gradient |
| `rep KL` | 约束 posterior | prior stop-gradient |
| Free Nats | 避免 KL 约束过早过强 | 默认 1.0 |
| Observe | 使用真实 observation 更新状态 | posterior |
| Imagine | 无 observation rollout | prior |

从代码角度看，DreamerV3 的 RSSM 并不是一个简单的：

```text
GRU + Gaussian
```

而是：

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

其中 deterministic state 负责"记住历史"，categorical stochastic state 负责"表达状态不确定性"，posterior 负责利用真实观测进行状态推断，prior 则负责在没有观测的时候预测未来。

而 KL balancing 把这两条路径连接起来：

```text
真实世界
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

最终，训练阶段学习的是：

> **如何根据 observation 推断世界状态。**

而 imagination 阶段使用的是：

> **如何仅根据 latent state 和 action 预测世界未来。**

这也是 DreamerV3 最核心的思想之一：

**模型不是直接在 observation space 里想象未来，而是在一个学出来的 latent space 里运行自己的世界模型。**

---

## 参考源码

本文代码分析主要对应 DreamerV3 官方仓库中的 `dreamerv3/rssm.py` 和 `dreamerv3/configs.yaml`。源码中的 `RSSM._core()`、`_prior()`、`_dist()`、`observe()`、`imagine()` 和 `loss()` 分别对应本文介绍的 deterministic transition、prior/posterior、categorical sampling、observe/imagine rollout 以及 KL balancing。

* [DreamerV3 官方 GitHub 仓库](https://github.com/danijar/dreamerv3)
* [DreamerV3 RSSM 源码 rssm.py](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py)
* [DreamerV3 默认配置 configs.yaml](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml)

> **源码版本说明：** DreamerV3 仓库的配置会随版本演进。如果你要在文章中做到"代码可复现"，建议同时记录本文对应的 commit hash，而不要只写 `main` 分支。
