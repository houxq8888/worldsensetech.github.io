---
title: "从代码理解 RSSM（六）：默认配置、四条公式与代码↔数学↔语义对照"
slug: "2026-08-24-rssm-recap"
date: 2026-08-24
draft: false
categories: ["世界模型"]
tags: ["RSSM", "DreamerV3", "世界模型", "状态空间模型", "代码解析", "RSSM系列"]
description: "收尾：把 RSSM 拆成架构/训练两张配置表，压缩成四条核心公式，给出代码↔数学↔语义对照表，并从 Memory / State estimation / Prediction 重新理解 RSSM。"
toc: true
---

> **《从代码理解 RSSM》系列 · 第 6 篇 / 共 6 篇**
>
> 系列目录（当前在第 6 篇，已加粗；上/下一篇见文末导航）：
> 1. [（一）RSSM 的位置与 stochastic 状态](/zh/articles/2026-08-19-rssm-code-walkthrough/)
> 2. [（二）先验/后验、straight-through 与 unimix](/zh/articles/2026-08-20-rssm-stochastic-state/)
> 3. [（三）_core()、deter=8192 与 Block GRU](/zh/articles/2026-08-21-rssm-deterministic-core/)
> 4. [（四）KL balancing、Free Nats 与最终 KL](/zh/articles/2026-08-22-rssm-kl-balancing/)
> 5. [（五）Imagine、Observe/Imagine 区别与 Reset](/zh/articles/2026-08-23-rssm-imagine-reset/)
> **6. [（六）默认配置、四条公式与对照表](/zh/articles/2026-08-24-rssm-recap/)**

## 二十五、默认配置：拆成"RSSM 架构"与"World-model 训练"两张表

为了避免把"RSSM 架构参数"和"world-model / agent 训练配置"混在一起，下面把配置拆成两张表。

### RSSM architecture（RSSM 架构）

| 参数 | 默认值 | 含义 |
| :--- | ---: | :--- |
| `deter` | 8192 | deterministic state 维度（会被 blocks 切分） |
| `stoch` | 32 | categorical variable 数量 |
| `classes` | 64 | 每个 categorical variable 的类别数 |
| `blocks` | 8 | Block GRU 分组数量（8192 / 8 = 1024） |
| `hidden` | 1024 | RSSM 内部 MLP hidden dimension |
| `imglayers` | 2 | prior 网络层数 |
| `obslayers` | 1 | posterior 网络层数 |
| `dynlayers` | 1 | dynamics 网络层数 |

### RSSM / world-model training（RSSM 与世界模型训练配置）

| 参数 | 默认值 | 含义 | 归属 |
| :--- | ---: | :--- | :--- |
| `unimix` | 0.01 | categorical distribution 的均匀混合比例 | RSSM 内部分布 |
| `free_nats` | 1.0 | KL 的 loss floor（`max(KL, 1)`） | RSSM 内部 |
| `dyn_scale` | 1.0 | dynamics KL 在最终 loss 中的权重 | agent loss scale |
| `rep_scale` | 0.1 | representation KL 在最终 loss 中的权重 | agent loss scale |
| `imag_length` | 15 | imagination rollout 的训练步数 | agent 训练配置（非 RSSM 结构） |

**关键点：**

* `deter / stoch / classes / blocks` 才是 RSSM 的**架构**；
* `unimix / free_nats` 是 RSSM 内部用来构造分布与 KL 的；
* `dyn_scale / rep_scale / imag_length` 是 **agent / world-model 训练配置**，不属于 RSSM 结构本身。

这些是**默认配置**，不是所有 DreamerV3 模型规模都固定使用的参数。

例如配置文件中的不同模型规模会改变 `deter`、`hidden` 和 `classes`。从 1M 到 400M，`deter` 从 512 增加到 12288，`classes` 从 4 增加到 96。

所以：

> `8192 / 32 / 64` 应该理解成 DreamerV3 默认配置下的具体实例，而不是 RSSM 的固定结构。

---

## 二十六、把整个 RSSM 压缩成四条公式

读完代码以后，其实整个 RSSM 可以浓缩成四步。

### ① Deterministic transition

```text
┌─────────────────────────────────────────┐
│  h_t = f(h_{t-1}, z_{t-1}, a_{t-1})    │
│  （_core()，不读取 observation）        │
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
│  L_KL = dyn_scale × L_dyn + rep_scale × L_rep       │
└─────────────────────────────────────────────────────┘
```

其中 KL 在进入最终 loss 前还会应用 `free_nats = max(KL, 1)`；`dyn_scale / rep_scale` 来自 agent 的 loss 配置，而非 RSSM 类内部。

---

## 二十七、代码 ↔ 数学 ↔ 语义 对照表

把前面二十多节的内容串起来，可以用一张表对齐"源码变量名 / 数学符号 / 语义"三者：

| 源码概念 | 数学 | 语义 |
| :--- | :--- | :--- |
| `deter` | `h_t` | deterministic memory（长期时序记忆） |
| `stoch` | `z_t` | stochastic latent（当前状态的不确定性） |
| `_core()` | `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})` | latent dynamics（确定性转移） |
| `prior` | `p(z_t | h_t)` | imagination prediction（不读 observation） |
| `posterior` | `q(z_t | h_t, o_t)` | observation-conditioned inference |
| `observe()` | filtering | 现实轨迹上的状态推断（经 `nj.scan` 调 `_observe`） |
| `imagine()` | latent rollout | 无 observation 的未来模拟（prior 闭环） |
| `dyn loss` | `KL[sg(q) || p]` | 训练 dynamics / prior |
| `rep loss` | `KL[q || sg(p)]` | 训练 representation / posterior |
| `feature` | `concat(h_t, z_t)` | 喂给 decoder / reward / actor-critic 的表征 |

这张表是全文的"索引"：任何一个源码符号，都能在这里找到它对应的数学含义与作用。

---

## 二十八、从代码角度重新理解 RSSM

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

从 DreamerV3 的实现角度，可以把 RSSM 理解为**同时承担三类功能**：

### 1. Memory

`deter` 通过 recurrent dynamics 保存历史。

### 2. State estimation

posterior 根据 observation 修正当前 latent state。

### 3. Prediction

prior 在没有 observation 的情况下预测未来 latent state。

最终：

> **训练阶段，RSSM 可以借助真实 observation 学习 latent dynamics；想象阶段，则丢掉 observation，仅依赖 prior 在 latent space 中向未来 rollout。**

这就是 DreamerV3 世界模型能够进行 latent imagination 的基础。

> 注：上面"RSSM 承担三类功能"是**功能性概括**，不是 RSSM 的严格定义。严谨地说，应表述为"从……实现角度，可以理解为……"，避免把实现层面的功能分工当成 RSSM 的数学定义。

---

## 二十九、最后回到源码：为什么这些细节值得关注？

如果只看论文，我们可能会把 RSSM 理解成 `RNN + latent distribution`。

但真正看代码以后，会发现 DreamerV3 在这上面做了大量工程设计：

```text
Categorical latent
       +
OneHot distribution
       +
Unimix  (p' = (1-ε)·p + ε/K)
       +
Block GRU  (BlockLinear 结构化参数化)
       +
RMSNorm
       +
Prior / Posterior
       +
KL balancing  (gradient routing: sg(q)||p 与 q||sg(p))
       +
Stop Gradient
       +
Free Nats  (max(KL, 1))
       +
Scan  (observe → nj.scan → _observe)
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
│          ├──► prior  ──► imagine（无 observation）
│          │             │
│ obs ─────┴──► posterior（有 observation）
│                │
│        KL align│
└────────────────┘
       │
       ▼
 latent state  (feature = concat(h_t, z_t))
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

所以，从代码层面看，RSSM 真正解决的并不是"如何预测下一个 observation"。

而是：

> **RSSM 真正解决的不是如何预测下一个 observation，而是如何学习一个既能被 observation 修正、又能脱离 observation 自主向未来滚动的 latent state。**

这才是 DreamerV3 RSSM 的核心。

---

## 参考源码

* [DreamerV3 GitHub 仓库](https://github.com/danijar/dreamerv3)
* [dreamerv3/rssm.py](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py)
* [dreamerv3/configs.yaml](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml)
