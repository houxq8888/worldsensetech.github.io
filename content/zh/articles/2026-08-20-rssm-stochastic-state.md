---
title: "从代码理解 RSSM（二）：先验/后验、straight-through 采样与 unimix"
slug: "2026-08-20-rssm-stochastic-state"
date: 2026-08-20
draft: false
categories: ["世界模型"]
tags: ["RSSM", "DreamerV3", "世界模型", "状态空间模型", "代码解析", "RSSM系列"]
description: "把源码翻译成数学，明确时间索引；理解 prior 与 posterior 两套分布、categorical 的 straight-through 采样，以及 unimix 给每个类别加概率下界的作用。"
toc: true
---

> **《从代码理解 RSSM》系列 · 第 2 篇 / 共 6 篇**
>
> 系列目录（当前在第 2 篇，已加粗；上/下一篇见文末导航）：
> 1. [（一）RSSM 的位置与 stochastic 状态](/zh/articles/2026-08-19-rssm-code-walkthrough/)
> **2. [（二）先验/后验、straight-through 与 unimix](/zh/articles/2026-08-20-rssm-stochastic-state/)**
> 3. [（三）_core()、deter=8192 与 Block GRU](/zh/articles/2026-08-21-rssm-deterministic-core/)
> 4. [（四）KL balancing、Free Nats 与最终 KL](/zh/articles/2026-08-22-rssm-kl-balancing/)
> 5. [（五）Imagine、Observe/Imagine 区别与 Reset](/zh/articles/2026-08-23-rssm-imagine-reset/)
> 6. [（六）默认配置、四条公式与对照表](/zh/articles/2026-08-24-rssm-recap/)

## 四、把源码翻译成数学公式（注意时间索引）

DreamerV3 的递推关系可以写成：

```text
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
```

然后分别计算：

```text
p(z_t | h_t)      [prior，不读 observation]
q(z_t | h_t, o_t) [posterior，读 observation]
```

其中：

* `p`：prior；
* `q`：posterior；
* `o_t`：当前 observation；
* `h_t`：deterministic state。

**一个非常重要、但初学者很容易卡住的时间索引说明：**

> `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})` 中的 `h_t` 已经融合了截至 `t-1` 的 latent / action 历史，但**它并不包含当前 `o_t`**。当前 observation 只在 **posterior 分支**中用于推断 `z_t`，而不进入 deterministic transition。

换句话说，初学者常问："既然 `h_t` 是当前状态，为什么它不包含 `o_t`？"——答案是：这里的 `h_t` 是"用历史 `(h_{t-1}, z_{t-1}, a_{t-1})` 推出来的当前 deterministic 记忆"，`o_t` 是同一时刻的另一路信息，只通过 posterior 修正 `z_t`，从不进入 `_core()`。理解这个索引，是读懂 Dreamer RSSM 的关键。

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

## 六、Categorical latent 是怎么采样的？（straight-through）

Posterior 和 prior 最终都会输出 logits。

例如：

```text
logits
[B, 32, 64]
```

每一个 `[64]` 都对应一个 categorical distribution，对应第 `i` 个 categorical variable 的 64 类概率。

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

这里需要特别精确地说明采样方式，避免一个常见误解：

> **`z_t` 并不是"先采样出一个整数 `[B, 32]`，再转成 one-hot"。**

更准确的描述是：

* 每个 stochastic variable 对应一个 64 类 categorical distribution；
* 采样得到的 stochastic state 在前向计算中以 **one-hot categorical representation**（或者说 one-hot-like 的表示）参与后续网络；
* 同时实现上使用 **straight-through estimator**，让离散采样能够参与反向传播——即前向走的是离散 one-hot 采样，反向时把梯度近似地回传到产生 logits 的网络参数上。

所以前向计算里 `z_t` 是一个形状为 `[B, 32, 64]` 的 one-hot-like 张量；它"看起来像 one-hot"，但其梯度通路是 straight-through 的，而不是"先取整再 one-hot"那种会把梯度完全切断的操作。

```text
logits
  │
  ▼
categorical distribution (each [64] a 64-class dist)
  │
  ▼
sample  ── 前向：one-hot-like representation
  │       反向：straight-through estimator
  ▼
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

`unimix` 会把它和 uniform distribution 做少量混合，让每个类别始终保留一点概率。写成公式：

```text
U_i = 1 / K            （K 为类别数，这里 K = 64）
p'_i = (1 - ε) × p_i + ε × U_i
     = (1 - ε) × p_i + ε / 64
```

其中 `ε = 0.01`。

也就是说，它对**每一个类别**都增加了一个很小的概率下界 `ε/K`，而不是只"让分布整体变钝"。

它的作用有两层：

1. 改善 categorical latent 的训练稳定性，避免分布过早变得过于尖锐；
2. 对 categorical 的 KL / entropy 计算提供数值稳定性（避免 0 概率导致的 log(0)、除零等问题）。

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

从组合空间的角度，32 个 categorical variable、每个 64 类，对应：

```text
64^32
```

种离散组合。

从数学上，这个 factorized categorical distribution 可以写成各变量独立分布的连乘：

```text
p(z_t | h_t) = Π_{i=1}^{32} p(z_t^i | h_t)
```

也就是说，联合分布由 32 个变量各自独立的类别分布相乘得到。

**但需要加一个重要限定：**

> `64^32` 描述的是这个 factorized categorical distribution 所"覆盖"的组合空间大小，**并不代表模型真的显式枚举了 `64^32` 个状态**。模型只是用 `32 × 64` 的 logits 对这个组合空间做了参数化（factorized categorical distribution），并不维护一张 `64^32` 规模的查找表或状态集合。

因此更严谨的表述是："该结构提供了 `64^32` 量级的组合表示能力"，而不是"模型在使用一个拥有 `64^32` 个显式状态的巨大离散状态空间"。这也是为什么 DreamerV3 可以在 relatively compact 的 latent space 中表达复杂环境状态。

---

