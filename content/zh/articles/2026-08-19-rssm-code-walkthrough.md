---
title: "从代码理解 RSSM（一）：RSSM 在 DreamerV3 中的位置与 stochastic 状态"
slug: "2026-08-19-rssm-code-walkthrough"
date: 2026-08-19
draft: false
categories: ["世界模型"]
tags: ["RSSM", "DreamerV3", "世界模型", "状态空间模型", "代码解析", "RSSM系列"]
description: "系列开篇：先定位 RSSM 在 DreamerV3 中的角色，澄清 stochastic state 不是“整数再 one-hot”，再看真实 observation 如何进入 RSSM（observe 的调用层次）。"
toc: true
---

> **《从代码理解 RSSM》系列 · 第 1 篇 / 共 6 篇**
>
> 系列目录（当前在第 1 篇，已加粗；上/下一篇见文末导航）：
> **1. [（一）RSSM 的位置与 stochastic 状态](/zh/articles/2026-08-19-rssm-code-walkthrough/)**
> 2. [（二）先验/后验、straight-through 与 unimix](/zh/articles/2026-08-20-rssm-stochastic-state/)
> 3. [（三）_core()、deter=8192 与 Block GRU](/zh/articles/2026-08-21-rssm-deterministic-core/)
> 4. [（四）KL balancing、Free Nats 与最终 KL](/zh/articles/2026-08-22-rssm-kl-balancing/)
> 5. [（五）Imagine、Observe/Imagine 区别与 Reset](/zh/articles/2026-08-23-rssm-imagine-reset/)
> 6. [（六）默认配置、四条公式与对照表](/zh/articles/2026-08-24-rssm-recap/)

> **源码阅读提示**
>
> 本文有意不从"标准 RSSM 伪代码"开始，而是按照 `rssm.py` 的实际执行路径展开。阅读时可以重点关注下面几个函数：
>
> ```text
> observe()   → sequence-level wrapper，沿时间维调用 _observe()
> _observe()  → 单个时间步的状态推断（真实轨迹）
> _core()     → 单个时间步的 deterministic dynamics
> imagine()   → 无 observation 的 latent rollout
> ```
>
> 这里最容易混淆的一点是：**`observe()` 并不是一个"处理单步 transition"的函数**。它只是 sequence-level 的封装，内部通过 Ninjax 的 `nj.scan(...)` 沿着时间维度反复调用 `_observe()`；真正处理单个时间步的是 `_observe()` 与 `_core()`。把 `observe() / _observe() / _core() / imagine()` 这一层调用关系理清，基本就能理解 DreamerV3 RSSM 的主体。
>
> （你后面会在"Sequence training"一节看到 `nj.scan` 的具体展开；这里先记一句话：**第一次出现 `observe()` 时，它和 `_observe()`、`nj.scan` 是同一件事的三个层次**。）

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
> 为了方便阅读，文中会对 JAX、Ninjax、dtype 和 `scan` 等工程代码进行适当简化，但核心计算逻辑以源码为准。文中凡是涉及"默认配置"的地方，都明确区分了 **RSSM 架构参数** 与 **World model / agent 的 loss 配置**，避免把某个 agent 的超参误读成 RSSM 本身的结构。

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

这里要先强调一个重要区分（后面"为什么 8192"一节还会展开）：**最终喂给 Decoder / Reward / Actor-Critic 的不是单独的 `h_t` 或 `z_t`，而是二者的拼接 `feature = concat(h_t, z_t)`**。所以 `8192` 并不是"latent state 总维度"。

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

默认配置（注意：下面混排了"RSSM 架构"与"world-model 训练"两类参数，正式拆分见"默认 RSSM 配置"一节）：

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
* 每个 variable 在网络前向计算中对应一个 64 维的 **one-hot categorical representation**。

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

每个 `z_t^i` 都是一个 64 类 categorical variable，它们构成的是一个 **factorized categorical distribution**（因子化的类别分布），而不是一个 2048 类的单一分布。

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

但回到开头那句提醒：**公开接口 `observe()` 本身不处理单步 transition**。它的核心作用是 sequence-level 的封装，通过 `nj.scan` 把下面这段 `_observe()` 沿着时间维反复调用。所以读者继续往下看源码时，应当把"处理单个时间步逻辑"的那段代码理解为 `_observe()`，而不是 `observe()`。

`_observe()` 的核心逻辑可以简化成：

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

