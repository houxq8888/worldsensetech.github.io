---
title: "从代码理解 RSSM：DreamerV3 中 RSSM 的实现细节"
slug: "2026-08-19-rssm-code-walkthrough"
date: 2026-08-19
draft: false
categories: ["世界模型"]
tags: ["RSSM", "DreamerV3", "世界模型", "状态空间模型", "代码解析"]
description: "从 DreamerV3 开源实现出发，沿着 rssm.py 追踪数据流：categorical latent、Block GRU、双 KL balancing（dyn + rep）、observe 与 imagine。"
toc: true
---

> **源码阅读提示**
>
> 本文有意不从"标准 RSSM 伪代码"开始，而是按照 `rssm.py` 的实际执行路径展开。阅读时可以重点关注三个函数：
>
> ```text
> observe()  → 真实轨迹上的状态推断
> _core()    → deterministic dynamics
> imagine()  → 无 observation 的 latent rollout
> ```
>
> 把这三个函数串起来，基本就能理解 DreamerV3 RSSM 的主体。

前面两篇文章分别介绍了 RSSM 的基本原理和世界模型的发展路线。

理论解决的是"RSSM 为什么这样设计"，但真正打开 DreamerV3 的源码后，会发现还有很多细节是论文公式没有直接告诉你的：

* 为什么 stochastic state 不是普通的 Gaussian 向量？
* `GRU` 到底接收什么输入？
* prior 和 posterior 在代码里分别对应什么？
* 为什么 DreamerV3 要计算两个 KL？
* `stop_gradient` 在 KL balancing 中到底起什么作用？
* imagination 阶段没有 observation，RSSM 是怎么继续运行的？
* `stoch=32, classes=64` 到底意味着什么？
* `deter=8192` 为什么这么大，还能训练得动？

这篇文章不再从一个通用的 Gaussian RSSM 伪代码出发，而是**直接沿着 DreamerV3 仓库中的 `dreamerv3/rssm.py` 追踪数据流**，把源码中的计算路径和 RSSM 数学公式对应起来。

> **源码说明**
>
> 本文参考的是 DreamerV3 开源仓库当前 `main` 分支中的 `dreamerv3/rssm.py` 与 `configs.yaml`。该仓库 README 将自身描述为 DreamerV3 的 reimplementation，因此本文统一称其为"DreamerV3 开源实现"，而不是 Google/DeepMind 官方代码。
>
> 为了方便阅读，文中会对 JAX、Ninjax、dtype 和 `scan` 等工程代码进行适当简化，但核心计算逻辑以源码为准。

---

## 一、先看 RSSM 在 DreamerV3 中的位置

DreamerV3 的世界模型可以粗略理解为：

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
     ├──► Decoder       重建 observation
     ├──► Reward Head   预测 reward
     ├──► Continue Head
     └──► Actor/Critic  想象轨迹上的策略与价值
```

RSSM 要解决的问题可以概括成一句话：

> **根据过去的 latent state 和 action，维护一个可以不断向未来滚动的隐状态。**

这个状态由两部分组成：

```text
s_t = (h_t, z_t)
```

其中：

* `h_t`：deterministic state，负责保存时序上下文；
* `z_t`：stochastic state，表示当前状态中的随机 latent。

DreamerV3 的一个关键变化就在这里：

> **`z_t` 不是传统连续 Gaussian RSSM 中的一个普通向量，而是多个 categorical latent variable。**

---

## 二、先解决一个最容易误解的问题：`stoch` 到底是什么？

很多 RSSM 教程会直接写：

```text
z_t ~ Normal(μ_t, σ_t)
```

然后通过：

```text
z = μ + σ × ε
```

完成采样。

这种写法可以帮助理解经典连续 RSSM，但**不能直接套到 DreamerV3 的实现上**。

DreamerV3 使用的是 categorical latent。

默认配置：

```yaml
rssm:
  deter: 8192
  hidden: 1024
  stoch: 32
  classes: 64
  unimix: 0.01
  blocks: 8
```

因此 stochastic state 的形状是：

```text
[B, 32, 64]
```

也就是说：

* 一共有 `32` 个 categorical variable；
* 每个 variable 有 `64` 个类别；
* 每个 variable 最终对应一个 64 维 one-hot 向量。

如果把它展平：

```text
32 × 64 = 2048
```

所以可以把整个 stochastic state 看成一个 2048 维向量，但**语义上不能简单把它理解成一个 2048 维普通 categorical variable**。

更准确地说：

```text
z_t = [z_t^1, z_t^2, ..., z_t^32]
```

其中：

```text
z_t^i ∈ {1, ..., 64}
```

每个 `z_t^i` 都是一个 64 类 categorical variable。

源码中的 `_logit()` 正是在做这件事：

```python
x = Linear(..., self.stoch * self.classes)(x)
return x.reshape(
    x.shape[:-1] + (self.stoch, self.classes)
)
```

也就是：

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

## 三、Observe：真实 observation 是怎么进入 RSSM 的？

理解 DreamerV3 RSSM，最重要的入口就是：

```python
observe(...)
```

源码的核心逻辑可以简化成：

```python
def _observe(carry, tokens, action, reset, training):

    deter, stoch, action = mask(
        carry["deter"],
        carry["stoch"],
        action,
        ~reset
    )

    action = preprocess_action(action)

    # 关键：这里没有 observation
    deter = self._core(
        deter,
        stoch,
        action
    )

    # observation token 在这里才进入
    x = concat([deter, tokens])

    logit = posterior_network(x)

    stoch = sample(logit)

    return {
        "deter": deter,
        "stoch": stoch,
    }
```

这里有一个非常关键的细节：

> **`_core()` 不读取当前 observation。**

当前 observation embedding，也就是 `tokens`，是在 deterministic transition 完成以后，才进入 posterior 网络。

所以整个过程实际上是：

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

这和很多"GRU 输入 observation + action"的简化 RSSM 写法是不一样的。

---

## 四、把源码翻译成数学公式

DreamerV3 的递推关系可以写成：

```text
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
```

然后分别计算：

```text
p(z_t | h_t)
```

以及：

```text
q(z_t | h_t, o_t)
```

其中：

* `p`：prior；
* `q`：posterior；
* `o_t`：当前 observation；
* `h_t`：deterministic state。

因此整个 RSSM 可以理解成：

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

这也是后面理解 imagination 和 KL loss 的关键。

---

## 五、为什么要有 prior 和 posterior 两套分布？

这两个分布实际上解决的是两个不同的问题。

### 1. Posterior：看到 observation 后，我认为现在是什么状态？

Posterior：

```text
q(z_t | h_t, o_t)
```

它可以同时看到：

* 历史信息 `h_t`
* 当前 observation `o_t`

所以它拥有更多信息。

训练真实轨迹时，我们利用 posterior 得到 stochastic state：

```python
logit = posterior(deter, token)
stoch = sample(logit)
```

因此可以把 posterior 理解成：

> **"看到了真实世界以后，对当前 latent state 的估计。"**

### 2. Prior：如果没有 observation，我预测会是什么状态？

Prior：

```text
p(z_t | h_t)
```

它只看 `h_t`，因此它不知道当前真实 observation。

它表达的是：

> **"只根据历史状态和动作，我预测接下来会进入什么 latent state。"**

这正是 imagination 阶段需要的能力。

---

## 六、Categorical latent 是怎么采样的？

Posterior 和 prior 最终都会输出 logits。

例如：

```text
logits
[B, 32, 64]
```

每一个 `[64]` 都对应一个 categorical distribution。

源码中的：

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

这里不是 Gaussian distribution，而是 one-hot categorical distribution。

因此 sampling 过程可以理解成：

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

最终：

```text
stoch.shape = [B, 32, 64]
```

---

## 七、`unimix=0.01` 是干什么的？

DreamerV3 的 categorical distribution 还有一个容易忽略的参数：

```yaml
unimix: 0.01
```

它的目的，是让 categorical distribution 保留一小部分均匀分布。

直观理解：

如果网络已经非常确定：

```text
class 7: 0.9999
other:   0.0001
```

那么分布会非常尖锐。

`unimix` 会把它和 uniform distribution 做少量混合，让每个类别始终保留一点概率。

可以粗略理解成：

```text
p' = (1 - ε) × p + ε × U
```

其中 `ε = 0.01`。

它的作用主要是改善 categorical latent 的训练稳定性，避免分布过早变得过于尖锐。

---

## 八、DreamerV3 为什么把 stochastic state 做成这么多 categorical variable？

这是 DreamerV3 latent representation 的重要设计。

默认：

```text
stoch = 32
classes = 64
```

不是：

```text
z ∈ R^32
```

而是：

```text
z = [
    categorical(64),
    categorical(64),
    ...
    categorical(64)
]
        × 32
```

这样做的一个重要好处，是可以形成非常丰富的离散组合空间。

理论上 32 个变量、每个 64 个类别，可以形成：

```text
64^32
```

种组合。

当然，实际模型并不会把这些组合全部利用起来，但这个结构提供了很大的表示能力。

这也是为什么 DreamerV3 可以在 relatively compact 的 latent space 中表达复杂环境状态。

---

## 九、真正的 deterministic transition：`_core()`

这是整个 `rssm.py` 中最值得读的部分。

源码并不是简单调用：

```python
nn.GRUCell(...)
```

而是自己构造了一个 block-wise GRU。

核心输入包括：

```python
deter
stoch
action
```

首先：

```python
stoch = stoch.reshape((stoch.shape[0], -1))
```

也就是：

```text
[B, 32, 64]
       │
       ▼
[B, 2048]
```

然后分别经过三个输入映射：

```python
x0 = Linear(hidden)(deter)
x1 = Linear(hidden)(stoch)
x2 = Linear(hidden)(action)
```

所以可以理解为：

```text
deter ──► Linear ──┐
                   │
stoch ──► Linear ──┼──► concat ──► Block GRU
                   │
action ─► Linear ──┘
```

注意：

> **这里没有 observation。**

这再次说明 deterministic transition 的核心是：

```text
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
```

---

## 十、为什么 `deter=8192`？

默认配置中的：

```yaml
deter: 8192
```

乍看非常大。

但 DreamerV3 并不是直接使用一个普通的 8192 维 dense GRU。

它还有：

```yaml
blocks: 8
```

因此 deterministic state 会被拆成 8 个 block。

```text
8192 / 8 = 1024
```

也就是说：

```text
8192
 │
 ├── block 1: 1024
 ├── block 2: 1024
 ├── ...
 └── block 8: 1024
```

然后使用 `nn.BlockLinear` 对这些 block 做变换。

这就是所谓的 **Block GRU**。

它的核心目标是：

> **保留大 deterministic state 的表示能力，同时避免一个完整 dense GRU 带来的巨大计算量和参数量。**

---

## 十一、Block GRU 到底在计算什么？

源码最终得到：

```python
x = BlockLinear(...)(x)

gates = split(x, 3)

reset, cand, update = gates
```

然后：

```python
reset = sigmoid(reset)
cand = tanh(reset * cand)
update = sigmoid(update - 1)

deter = update * cand + (1 - update) * deter
```

如果写成数学形式：

```text
r_t = σ(W_r × x_t + b_r)
h̃_t = tanh(r_t ⊙ W_h × x_t)
u_t = σ(W_u × x_t - 1)
h_t = u_t ⊙ h̃_t + (1 - u_t) ⊙ h_{t-1}
```

所以它本质上仍然是 GRU，只是：

* 输入经过独立映射；
* hidden transformation 使用 block structure；
* gate 计算也采用 block-wise transformation。

---

## 十二、Posterior 网络具体做了什么？

回到 `_observe()`。

确定性状态得到以后：

```python
x = tokens if self.absolute else concat([deter, tokens])
```

默认：

```yaml
absolute: False
```

因此默认情况下：

```text
x_t = [h_t, o_t^emb]
```

然后经过 `obslayers` 层 MLP。

默认配置：

```yaml
obslayers: 1
hidden: 1024
```

最后：

```python
logit = self._logit('obslogit', x)
```

得到 `[B, 32, 64]`。

所以 posterior 的计算可以写成：

```text
deter_t + observation_token_t
              │
              ▼
         obs network
              │
              ▼
           logits
              │
              ▼
       categorical q(z_t)
```

---

## 十三、Prior 网络又是什么？

Prior 的代码反而非常简单：

```python
def _prior(self, feat):
    x = feat

    for i in range(self.imglayers):
        x = Linear(hidden)(x)
        x = activation(norm(x))

    return self._logit('priorlogit', x)
```

输入只有 `deter_t`。

默认 `imglayers: 2`。

因此可以简单理解成：

```text
h_t
 │
 ▼
MLP (2 layers)
 │
 ▼
32 × 64 logits
 │
 ▼
p(z_t | h_t)
```

---

## 十四、Observe 阶段完整数据流

到这里，可以把整个 observe 过程串起来：

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
z_{t-1} ───────┐       │
               │       │
h_{t-1} ───────┼───────┼──► RSSM Core
               │       │        ▲
a_{t-1} ───────┘       │        │
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
                └────── KL ───┘
```

这张图基本就是 DreamerV3 RSSM 的核心。

---

## 十五、KL balancing：源码为什么计算两个 KL？

这是 DreamerV3 RSSM 最容易被简单带过、但实际上非常关键的一部分。

源码：

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

注意这里有两个 KL。

### 1. Dynamics KL

```python
dyn = KL(sg(post) || prior)
```

也就是：

```text
L_dyn = KL[sg(q(z_t|h_t,o_t)) || p(z_t|h_t)]
```

这里 `sg(post)` 意味着 posterior 被 stop-gradient。

所以这个 loss 的梯度主要用于：

> **训练 prior / dynamics model 去逼近 posterior。**

可以把它理解成：

```text
posterior = 老师
     │
     ▼
   target
     │
     ▼
  prior 学习
```

---

## 十六、Representation KL

第二项：

```python
rep = KL(post || sg(prior))
```

数学上：

```text
L_rep = KL[q(z_t|h_t,o_t) || sg(p(z_t|h_t))]
```

这里反过来了：`sg(prior)` 因此 prior 被当成固定 target。

这个 loss 主要用于：

> **约束 posterior 的 representation 不要无限偏离 prior。**

所以两个 KL 的方向不是为了数学上"重复计算一次"，而是通过 `stop_gradient` **明确指定了优化方向**。

---

## 十七、为什么不能直接写一个 KL？

如果简单写：

```python
kl = KL(post || prior)
```

然后一起优化，会让 prior 和 posterior 同时受到梯度影响。

但 DreamerV3 希望把两个角色分开：

```text
Dynamics KL
posterior ──► prior
       训练 dynamics

Representation KL
prior ──► posterior
       约束 representation
```

因此：

```text
dyn = KL(sg(post) || prior)
rep = KL(post || sg(prior))
```

这就是 KL balancing 的核心。

**真正值得注意的不是"有两个 KL"，而是 `stop_gradient` 把优化目标拆成了两个方向。**

---

## 十八、Free Nats 到底做了什么？

源码：

```python
if self.free_nats:
    dyn = jnp.maximum(dyn, self.free_nats)
    rep = jnp.maximum(rep, self.free_nats)
```

默认 `free_nats: 1.0`，所以：

```text
L_dyn = max(L_dyn, 1.0)
L_rep = max(L_rep, 1.0)
```

这里有一个需要特别纠正的常见说法：

> **不要把 DreamerV3 的 `free_nats` 解释成"每个 stochastic dimension 至少保留 1 nat 信息"。**

源码并不是对每个 categorical variable 单独设置一个 information floor。

它是在 `_dist(...).kl(...)` 得到的 KL 张量上直接做 `maximum(kl, free_nats)`。

因此文章里最好直接按照源码描述。

---

## 十九、最终 KL loss 是怎么组合的？

默认配置：

```yaml
loss_scales:
  dyn: 1.0
  rep: 0.1
```

所以 RSSM 的 KL 部分可以写成：

```text
L_KL = 1.0 × L_dyn + 0.1 × L_rep
```

注意：

> `1.0` 和 `0.1` 并不是 RSSM 类内部写死的，而是 agent 的 loss scale 配置。

因此如果强调"源码级解析"，最好区分：

```text
rssm.py
    ├── 计算 dyn KL
    ├── 计算 rep KL
    └── free_nats

configs.yaml / agent loss scale
    ├── dyn = 1.0
    └── rep = 0.1
```

这样读者就不会误以为 `0.1` 是 RSSM 本身的固定公式。

---

## 二十、Imagine 阶段：没有 observation，RSSM 怎么跑？

这是 RSSM 最漂亮的地方。

训练真实轨迹时需要 observation 来计算 `q(z_t|h_t,o_t)`。但 imagination 阶段没有真实 observation。

直接使用 `p(z_t|h_t)`，也就是 prior。

源码中的 `imagine()`：

```python
deter = self._core(
    carry['deter'],
    carry['stoch'],
    actemb
)

logit = self._prior(deter)

stoch = self._dist(logit).sample(...)
```

注意：

> **这里完全没有 observation，也没有用零向量填 observation。**

这是因为 `_core()` 本来就不需要 observation。

因此 imagination：

```text
h_{t-1}, z_{t-1}, a_{t-1}
              │
              ▼
          RSSM Core
              │
              ▼
             h_t
              │
              ▼
           Prior
              │
              ▼
             z_t
              │
              └──────────────┐
                             │
                             ▼
                    下一时间步继续 rollout
```

这就是世界模型真正开始"做梦"的地方。

---

## 二十一、Observe 和 Imagine 的区别

把两条路径放在一起就非常清楚了。

### Observe

```text
previous state + action
          │
          ▼
        _core
          │
          ▼
         h_t
          │
          ├─────────────► prior
          │
          ▼
 observation token
          │
          ▼
      posterior
          │
          ▼
         z_t
```

### Imagine

```text
previous state + action
          │
          ▼
        _core
          │
          ▼
         h_t
          │
          ▼
        prior
          │
          ▼
         z_t
```

所以可以把两者理解成：

> **Observe 是"看着现实更新状态"，Imagine 是"闭着眼睛根据模型预测状态"。**

---

## 二十二、为什么 imagination 不能无限长？

因为 imagination 使用的是 `p(z_t|h_t)` 而不是 `q(z_t|h_t,o_t)`，因此每一步预测都会受到模型误差影响。

```text
预测误差
   │
   ▼
下一步输入
   │
   ▼
新的预测误差
   │
   ▼
继续累积
```

这就是典型的 rollout error accumulation。

DreamerV3 默认 `imag_length: 15`，也就是 imagination horizon 为 15。

这并不是说 15 步之后世界模型突然失效，而是一个工程上的 trade-off：

* 想象太短：规划能力不足；
* 想象太长：模型误差累积严重。

---

## 二十三、Sequence training：RSSM 为什么可以处理整段序列？

`observe()` 并不是只处理单个 timestep。

源码使用 Ninjax 的 `nj.scan(...)` 沿着时间维度展开 RSSM。

可以理解成：

```text
t=0
  │
  ▼
(h0, z0)
  │
  ▼
t=1
  │
  ▼
(h1, z1)
  │
  ▼
t=2
  │
  ▼
(h2, z2)
  │
  ▼
...
```

状态会沿着 sequence 不断传递。

所以 RSSM 的时间递推依然存在：

```text
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
```

`scan` 只是 JAX/Ninjax 层面的工程实现，用来更高效地表达这个递推过程。

---

## 二十四、Reset 为什么很重要？

RSSM 是有记忆的。

如果一个 episode 结束：

```text
Episode A
    │
    ▼
RSSM state
    │
    X reset
    │
    ▼
Episode B
```

如果不 reset，那么 Episode B 的初始状态会携带 Episode A 的历史。

源码在 `_observe()` 开头会根据 `reset` 对 `deter`、`stoch`、`action` 进行 mask。

因此 episode 边界会切断之前的 latent state。

这不是一个小细节。

对于 recurrent world model 来说：

> **reset 本质上就是告诉模型："新的世界开始了，不要把上一个 episode 的记忆带过来。"**

---

## 二十五、默认 RSSM 配置

当前 `configs.yaml` 中默认 RSSM 配置为：

| 参数 | 默认值 | 含义 |
|:---|---:|:---|
| `deter` | 8192 | deterministic state 维度 |
| `hidden` | 1024 | RSSM MLP hidden dimension |
| `stoch` | 32 | categorical variable 数量 |
| `classes` | 64 | 每个 categorical variable 的类别数 |
| `unimix` | 0.01 | categorical distribution 的均匀混合比例 |
| `blocks` | 8 | Block GRU 分组数量 |
| `free_nats` | 1.0 | KL free-nats 阈值 |
| `imglayers` | 2 | prior 网络层数 |
| `obslayers` | 1 | posterior 网络层数 |
| `dynlayers` | 1 | dynamics 网络层数 |

这些是**默认配置**，不是所有 DreamerV3 模型规模都固定使用的参数。

例如配置文件中的不同模型规模会改变 `deter`、`hidden` 和 `classes`。从 1M 到 400M，`deter` 从 512 增加到 12288，`classes` 从 4 增加到 96。

所以：

> `8192 × 32 × 64` 应该理解成 DreamerV3 默认配置下的具体实例，而不是 RSSM 的固定结构。

---

## 二十六、把整个 RSSM 压缩成四条公式

读完代码以后，其实整个 RSSM 可以浓缩成四步。

### ① Deterministic transition

```text
┌─────────────────────────────────────────┐
│  h_t = f(h_{t-1}, z_{t-1}, a_{t-1})    │
└─────────────────────────────────────────┘
```

历史状态和动作决定 deterministic state。

### ② Prior

```text
┌─────────────────────────┐
│  p(z_t | h_t)           │
└─────────────────────────┘
```

只依赖 deterministic state。它负责 imagination。

### ③ Posterior

```text
┌─────────────────────────────┐
│  q(z_t | h_t, o_t)          │
└─────────────────────────────┘
```

利用 observation 对 latent state 进行修正。它负责真实轨迹上的状态推断。

### ④ KL balancing

```text
┌─────────────────────────────────────────────────────┐
│  L_dyn = KL[sg(q(z_t|h_t,o_t)) || p(z_t|h_t)]      │
│  L_rep = KL[q(z_t|h_t,o_t) || sg(p(z_t|h_t))]      │
│                                                      │
│  L_KL = 1.0 × L_dyn + 0.1 × L_rep                  │
└─────────────────────────────────────────────────────┘
```

其中 KL 在进入最终 loss 前还会应用 `free_nats`。

---

## 二十七、从代码角度重新理解 RSSM

如果不看代码，RSSM 很容易被理解成：

> "一个 GRU + 一个 Gaussian。"

但 DreamerV3 的实际实现要丰富得多：

```text
                 ┌──────────────────────┐
                 │      RSSM            │
                 │                      │
 action ────────►│                      │
                 │    Block GRU         │
 z_{t-1} ───────►│        │             │
                 │        ▼             │
 h_{t-1} ───────►│       h_t            │
                 │      /   \            │
                 │     /     \           │
                 │  Prior   Posterior    │
                 │    │        ▲         │
                 │    │        │         │
                 │    │      token_t     │
                 │    │        │         │
                 │    ▼        ▼         │
                 │   p(z)     q(z)       │
                 │     │        │        │
                 │     └── KL ──┘        │
                 └──────────────────────┘
```

所以 RSSM 实际上统一完成了三件事情：

### 1. Memory

`deter` 通过 recurrent dynamics 保存历史。

### 2. State estimation

posterior 根据 observation 修正当前 latent state。

### 3. Prediction

prior 在没有 observation 的情况下预测未来 latent state。

最终：

> **训练阶段，RSSM 可以借助真实 observation 学习 latent dynamics；想象阶段，则丢掉 observation，仅依赖 prior 在 latent space 中向未来 rollout。**

这就是 DreamerV3 世界模型能够进行 latent imagination 的基础。

---

## 二十八、最后回到源码：为什么这些细节值得关注？

如果只看论文，我们可能会把 RSSM 理解成 `RNN + latent distribution`。

但真正看代码以后，会发现 DreamerV3 在这上面做了大量工程设计：

```text
Categorical latent
       +
OneHot distribution
       +
Unimix
       +
Block GRU
       +
RMSNorm
       +
Prior / Posterior
       +
KL balancing
       +
Stop Gradient
       +
Free Nats
       +
Scan
```

这些东西单独看都不复杂。

真正重要的是它们组合起来之后形成了一条完整的链：

```text
真实 observation
       │
       ▼
   Encoder
       │
       ▼
 observation token
       │
       ▼
┌────────────────┐
│     RSSM       │
│                │
│ history ──► h  │
│          │     │
│          ├──► prior
│          │
│ obs ─────┴──► posterior
│                │
│        KL align│
└────────────────┘
       │
       ▼
 latent state
       │
       ▼
 imagination
       │
       ▼
 future latent states
       │
       ▼
 reward / value / policy
```

所以，从代码层面看，RSSM 真正解决的问题并不是"如何预测下一个 observation"。

而是：

> **如何学习一个 latent state，使它既能够解释真实 observation，又能够在没有 observation 的情况下，仅凭历史状态和 action 自己向未来滚动。**

这才是 DreamerV3 RSSM 的核心。

---

## 参考源码

* [DreamerV3 GitHub 仓库](https://github.com/danijar/dreamerv3)
* [dreamerv3/rssm.py](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py)
* [dreamerv3/configs.yaml](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml)

> **版本建议：** 如果这篇文章作为长期技术博客保存，建议发布时记录具体 commit hash，而不是只写 `main`。因为源码和默认配置可能继续演进。当前仓库的默认分支是 `main`。
