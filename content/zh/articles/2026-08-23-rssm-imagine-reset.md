---
title: "从代码理解 RSSM（五）：Imagine、Observe/Imagine 区别、序列训练与 Reset"
slug: "2026-08-23-rssm-imagine-reset"
date: 2026-08-23
draft: false
categories: ["世界模型"]
tags: ["RSSM", "DreamerV3", "世界模型", "状态空间模型", "代码解析", "RSSM系列"]
description: "脱离 observation 的未来模拟：imagine 的完整闭环、Observe 与 Imagine 的本质区别、为什么 imagination 不能无限长，以及 sequence training 与 reset 的作用。"
toc: true
---

> **《从代码理解 RSSM》系列 · 第 5 篇 / 共 6 篇**
>
> 系列目录（当前在第 5 篇，已加粗；上/下一篇见文末导航）：
> 1. [（一）RSSM 的位置与 stochastic 状态](/zh/articles/2026-08-19-rssm-code-walkthrough/)
> 2. [（二）先验/后验、straight-through 与 unimix](/zh/articles/2026-08-20-rssm-stochastic-state/)
> 3. [（三）_core()、deter=8192 与 Block GRU](/zh/articles/2026-08-21-rssm-deterministic-core/)
> 4. [（四）KL balancing、Free Nats 与最终 KL](/zh/articles/2026-08-22-rssm-kl-balancing/)
> **5. [（五）Imagine、Observe/Imagine 区别与 Reset](/zh/articles/2026-08-23-rssm-imagine-reset/)**
> 6. [（六）默认配置、四条公式与对照表](/zh/articles/2026-08-24-rssm-recap/)

## 二十、Imagine 阶段：没有 observation，RSSM 怎么跑？

这是 RSSM 最漂亮的地方。

训练真实轨迹时需要 observation 来计算 `q(z_t|h_t,o_t)`。但 imagination 阶段没有真实 observation。

直接使用 `p(z_t|h_t)`，也就是 prior：

```text
z_t ~ p(z_t | h_t)
```

源码中的 `imagine()` 内部同样先走 `_core()`，再用 prior 产生 latent：

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

不过只写出单步还不够，**imagination 真正的核心是一个闭环**：actor 产生 action 后，RSSM 做一步 transition，prior 再产生下一时刻的 latent，然后 actor 再产生新的 action……如此循环 rollout。

完整循环写成：

```text
z_t ~ p(z_t | h_t)
  │
  ▼
actor → a_t                         （策略根据当前 feature 产出动作）
  │
  ▼
h_{t+1} = f(h_t, z_t, a_t)          （_core 推进 deterministic state）
  │
  ▼
z_{t+1} ~ p(z_{t+1} | h_{t+1})      （prior 预测下一 latent）
  │
  ▼
actor → a_{t+1} → ...               （闭环继续）
```

用一张更紧凑的闭环图表示：

```text
(h_t, z_t)
    │
    ├── actor → a_t
    │
    ▼
core(h_t, z_t, a_t)
    │
    ▼
h_{t+1}
    │
    ▼
prior
    │
    ▼
z_{t+1}
    │
    └──► 再回到 actor，闭环继续
```

**Actor 产生 action → RSSM transition → prior 产生 latent → 再产生 action。** 如果文章主题是 RSSM，这一段闭环正是 Dreamer "latent imagination" 的心脏：世界模型自己"做梦"，策略在梦里试错。

---

## 二十一、Observe 和 Imagine 的区别

把两条路径放在一起就非常清楚了（它们共享同一个 `_core()` 转移，区别只在 `z_t` 来自 posterior 还是 prior）。

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
          ├─────────────► prior（也计算，用于 KL）
          │
          ▼
 observation token
          │
          ▼
      posterior
          │
          ▼
         z_t   ← posterior(h_t, o_t)
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
         z_t   ← prior(h_t)
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

> **需要明确归属：** `imag_length: 15` 是 **agent 的 imagination rollout 训练配置**，不是 RSSM 本身的结构参数。它属于 world-model / agent 的训练超参，决定策略在想象轨迹上展开多少步，与 `deter / stoch / classes / blocks` 这类 RSSM 架构参数不是同一层概念。

这并不是说 15 步之后世界模型突然失效，而是一个工程上的 trade-off：

* 想象太短：规划能力不足；
* 想象太长：模型误差累积严重。

---

## 二十三、Sequence training：RSSM 为什么可以处理整段序列？

回到开头那句提醒：`observe()` 并不是只处理单个 timestep，它只是 sequence-level 的封装。

源码使用 Ninjax 的 `nj.scan(...)` 沿着时间维度展开 RSSM——**这正是第三节开头说过的：`observe()` →（通过 `nj.scan`）→ `_observe()` 的调用层次在第一次出现 `observe()` 时就应当连起来。**

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

> **reset 本质上就是切断 episode 之间的 recurrent state——告诉模型："新的世界开始了，不要把上一个 episode 的记忆带过来。"**

---

