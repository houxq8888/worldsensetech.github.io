---
title: "从代码理解 RSSM（四）：KL balancing、Free Nats 与最终 KL 组合"
slug: "2026-08-22-rssm-kl-balancing"
date: 2026-08-22
draft: false
categories: ["世界模型"]
tags: ["RSSM", "DreamerV3", "世界模型", "状态空间模型", "代码解析", "RSSM系列"]
description: "训练世界模型的核心：Observe 阶段完整数据流、dyn/rep 两个 KL 的 gradient routing、free_nats 作为 loss floor，以及最终 KL loss 如何组合。"
toc: true
---

> **《从代码理解 RSSM》系列 · 第 4 篇 / 共 6 篇**
>
> 系列目录（当前在第 4 篇，已加粗；上/下一篇见文末导航）：
> 1. [（一）RSSM 的位置与 stochastic 状态](/zh/articles/2026-08-19-rssm-code-walkthrough/)
> 2. [（二）先验/后验、straight-through 与 unimix](/zh/articles/2026-08-20-rssm-stochastic-state/)
> 3. [（三）_core()、deter=8192 与 Block GRU](/zh/articles/2026-08-21-rssm-deterministic-core/)
> **4. [（四）KL balancing、Free Nats 与最终 KL](/zh/articles/2026-08-22-rssm-kl-balancing/)**
> 5. [（五）Imagine、Observe/Imagine 区别与 Reset](/zh/articles/2026-08-23-rssm-imagine-reset/)
> 6. [（六）默认配置、四条公式与对照表](/zh/articles/2026-08-24-rssm-recap/)

## 十四、Observe 阶段完整数据流（Observe / Imagine 同一框架）

到这里，可以把整个 RSSM 的状态转移框架画出来，并且把 **Observe 与 Imagine 放在同一个 `_core()` 转移框架里**：

```text
                 ┌──────────────────────────┐
                 │      RSSM transition     │
                 │                          │
(h_{t-1},z_{t-1}) + a_{t-1}
                 │
                 ▼
              _core()
                 │
                 ▼
                h_t
                 │
          ┌──────┴──────┐
          │             │
          ▼             ▼
       Prior        Observation
   p(z_t | h_t)         │
          │             ▼
          │         Posterior
          │      q(z_t|h_t,o_t)
          │             │
          │             ▼
          │            z_t
          └───────┬─────┘
                  │
                  ▼
              latent state
                  │
                  ▼
               next step
```

在旁边注明两种用法，其实这就是全文最重要的两句话：

```text
OBSERVE:  z_t ← posterior(h_t, o_t)
IMAGINE:  z_t ← prior(h_t)
```

Observe 的完整数据流就是在上面的框架里走 `OBSERVE` 这一支（observation 进入 posterior）；而 Imagine 走的是 `IMAGINE` 那一支（没有 observation，直接用 prior）。这张图基本就是 DreamerV3 RSSM 的核心。

---

## 十五、KL balancing：源码为什么计算两个 KL？（重点看 gradient routing）

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

```text
L_dyn = KL[sg(q(z_t|h_t,o_t)) || p(z_t|h_t)]
```

这里 `sg(post)` 意味着 posterior 被 stop-gradient。

所以这个 loss 的梯度主要用于：

> **让 prior / dynamics model 学会预测（逼近）posterior。**

也就是说，**observation 提供的额外信息充当 teacher，但不让梯度穿过 posterior 回去**——prior 被拉近到 posterior 的分布。

可以把它理解成：

```text
posterior = teacher
     │
     ▼
   target
     │
     ▼
  prior 学习（梯度只更新 prior / dynamics）
```

---

## 十六、Representation KL

第二项：

```text
L_rep = KL[q(z_t|h_t,o_t) || sg(p(z_t|h_t))]
```

这里反过来了：`sg(prior)` 因此 prior 被当成固定 target。

这个 loss 主要用于：

> **让 posterior 的 representation 不要与 learned dynamics（已经学好的 prior）完全脱节。**

这里要做一个重要的精确化：上面这句话直觉上是对的，但还不够精确。更本质的理解是——**两个 KL 的 gradient routing 不同**：

```text
L_dyn:  sg(q) || p      → 梯度主要训练 dynamics / prior
L_rep:  q || sg(p)      → 梯度主要训练 representation / posterior
```

也就是说，**KL balancing 真正重点不是"约束 posterior 不要无限偏离 prior"这种模糊表述，而是：通过 `stop_gradient` 把优化目标拆成两条不同的梯度通路——一条训练 dynamics（让 prior 追 posterior），一条训练 representation（让 posterior 不脱离 dynamics）。** 这才是 KL balancing 的核心。

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
  sg(q) ──► p        训练 dynamics / prior
            （posterior 当 teacher，不回传梯度）

Representation KL
  q ──► sg(p)        训练 representation / posterior
            （prior 当固定 target）
```

因此：

```text
dyn = KL(sg(post) || prior)
rep = KL(post || sg(prior))
```

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

**更精确地说（仅描述数值形式）：它不是让 KL 低于 1 时变成 0，而是把 loss 在 1 这个位置 floor 住：**

```text
L' = max(L, 1.0)
```

> **严格限定：** 上面只说明 `free_nats` 是一个 **loss floor**（在 KL 张量上做 `max(kl, free_nats)`），而不是逐 latent dimension 的 information floor。仅从这个 `max` 形式，不能推出"KL 低于 1 时无梯度"之类的训练效果结论——除非已确认具体的 JAX 实现，以及 KL 张量后续如何聚合、缩放并进入最终 loss。本文不对此做进一步推断。

因此：

```text
KL = 0.2  →  loss = 1.0
KL = 0.8  →  loss = 1.0
KL = 1.5  →  loss = 1.5
```

这一点非常值得写清楚，因为"free nats"这个术语很容易让读者自动套用其他 world model 实现里的定义——例如很多实现里用的是：

```text
max(KL - τ, 0)
```

这与 DreamerV3 的 `max(KL, 1)` **并不是同一个形式**。DreamerV3 是"把 KL 的下界抬高到 1"，而不是"减去一个阈值再和 0 取大"。

---

## 十九、最终 KL loss 是怎么组合的？

默认配置（注意：下面属于 **agent 的 loss scale 配置**，不是 RSSM 架构本身）：

```yaml
loss_scales:
  dyn: 1.0
  rep: 0.1
```

所以 RSSM 的 KL 部分可以写成：

```text
L_KL = 1.0 × L_dyn + 0.1 × L_rep
```

注意（这一点非常重要，全文都应保持这个区分）：

> `1.0` 和 `0.1` 并不是 RSSM 类内部写死的，而是 **agent 的 loss scale 配置**（属于 world-model / agent 训练配置），不该被算作 RSSM 架构的固有属性。

因此如果强调"源码级解析"，最好区分：

```text
RSSM architecture (rssm.py)
    ├── 计算 dyn KL
    ├── 计算 rep KL
    └── free_nats（RSSM 内部）

World model / agent loss config (configs.yaml)
    ├── dyn_scale  = 1.0
    └── rep_scale  = 0.1
```

这样读者就不会误以为 `0.1` 是 RSSM 本身的固定公式，也不会把"RSSM 默认配置"理解成一个混在一起的大表。

---

