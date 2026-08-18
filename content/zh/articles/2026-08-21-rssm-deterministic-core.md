---
title: "从代码理解 RSSM（三）：确定性转移 _core()、deter=8192 与 Block GRU"
slug: "2026-08-21-rssm-deterministic-core"
date: 2026-08-21
draft: false
categories: ["世界模型"]
tags: ["RSSM", "DreamerV3", "世界模型", "状态空间模型", "代码解析", "RSSM系列"]
description: "追踪真正的确定性动力学：_core() 如何做单步转移、为什么 deter 取 8192、Block GRU 的 block-wise 参数化，以及 posterior/prior 网络各自做什么。"
toc: true
---

> **《从代码理解 RSSM》系列 · 第 3 篇 / 共 6 篇**
>
> 系列目录（当前在第 3 篇，已加粗；上/下一篇见文末导航）：
> 1. [（一）RSSM 的位置与 stochastic 状态](/zh/articles/2026-08-19-rssm-code-walkthrough/)
> 2. [（二）先验/后验、straight-through 与 unimix](/zh/articles/2026-08-20-rssm-stochastic-state/)
> **3. [（三）_core()、deter=8192 与 Block GRU](/zh/articles/2026-08-21-rssm-deterministic-core/)**
> 4. [（四）KL balancing、Free Nats 与最终 KL](/zh/articles/2026-08-22-rssm-kl-balancing/)
> 5. [（五）Imagine、Observe/Imagine 区别与 Reset](/zh/articles/2026-08-23-rssm-imagine-reset/)
> 6. [（六）默认配置、四条公式与对照表](/zh/articles/2026-08-24-rssm-recap/)

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

然后把 `deter`、`stoch`、`action` 各自通过一个输入映射（注意：这里的"分别 Linear 映射再 concat"是概念示意图，**真正的核心在于 BlockLinear 的结构化参数化**，见下一节）：

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

为什么这件事值得专门做？可以给一个量级直觉：

一个普通 GRU 的 recurrent projection 如果直接对一个 8192 维的 hidden state 做 dense 变换，其参数规模会非常夸张（仅 recurrent 权重就接近 `8192 × 8192` 量级）。**BlockLinear 将大维度的线性变换组织成 block-wise 的参数结构**，从而避免使用完全 dense 的 `8192 × 8192` recurrent transformation。需要谨慎：不能仅凭类名就推断各 block 之间在数学上完全独立——具体 block 之间是否存在交互（以及以何种方式交互），应当以 `nn.BlockLinear` 的具体实现为准。本文只描述其"用结构化 block 参数化替代完全 dense 变换"这一工程意图，而不对 block 间的连接结构做过度推断。

### 一个容易遗漏的维度区分

这里顺便澄清文章里一个重要的概念缺口：`8192` 并不是"latent state 总维度"。

| state | 作用 |
| :--- | :--- |
| `deter`（8192） | 长期时序记忆 / deterministic dynamics |
| `stoch`（32×64=2048） | 当前状态的不确定性 / observation-conditioned 信息 |
| `feature = concat(deter, stoch)` | 给 Decoder、Reward、Actor-Critic 使用 |

也就是说，模型最终使用的 latent feature 是 `concat(h_t, z_t)`，其维度是 `8192 + 2048`，而不是单纯的 `8192`。`deter` 大是为了给 deterministic dynamics 足够的记忆容量；`stoch` 则保留 observation 带来的不确定性信息。

把维度单独写出来：

```text
dim(h_t)        = 8192
dim(z_t)        = 32 × 64 = 2048
dim(feature_t)  = 8192 + 2048 = 10240
```

即：

```text
deter   = 8192
stoch   = 2048
feature = 10240
```

这其实是理解 DreamerV3 后续 decoder / actor / critic 输入的重要桥梁：它们拿到的不是单独的 `h_t` 或 `z_t`，而是 10240 维的拼接 feature。

---

## 十一、Block GRU 到底在计算什么？（变量名 ≠ 标准 GRU）

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

**这里必须先强调一个容易让人困惑的点：**

> 上面代码里的 `reset`、`cand`、`update` 是**该实现内部的中间变量 / chunk 名称**，它们**不**直接等同于教科书 GRU 里的 `reset gate / update gate / candidate`。实现首先通过 projection 得到多个 gate 分量，再按照 GRU 的门控结构更新 deterministic state。读这段代码时，不要把它误认为"标准 GRU 公式的直接改写"。

如果把这些内部 chunk 与标准 GRU 门控结构对应起来，核心仍然可以抽象成标准 GRU 的三部分 `r、z、h̃`：

```text
r_t   = σ(W_r · x_t + U_r · h_{t-1} + b_r)   [reset gate]
z_t   = σ(W_z · x_t + U_z · h_{t-1} + b_z)   [update gate]
h̃_t   = tanh(W_h · x_t + U_h · (r_t ⊙ h_{t-1}) + b_h)   [candidate]
h_t   = z_t ⊙ h̃_t + (1 - z_t) ⊙ h_{t-1}
```

把它翻译成"源码 ↔ 数学"的对应关系就是：

* 源码 chunk `update` ≈ 标准 GRU 的 update gate `z_t`（控制"保留多少旧状态"）；
* 源码 chunk `reset` ≈ 标准 GRU 的 reset gate `r_t`（控制"多少旧状态进入 candidate"）；
* 源码 chunk `cand` + 与 `reset` 的交互 ≈ 标准 GRU 的 candidate `h̃_t`。

所以结论仍是：**Block GRU 在门控语义上就是 GRU**，只是：

* 输入经过独立映射；
* hidden transformation 使用 block structure（BlockLinear）；
* gate 计算也采用 block-wise transformation；
* 内部变量命名是工程实现细节，不要和教科书符号一一死板对应。

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

**为什么这个开关值得单独注明？**

它本质是在决定 observation encoder token 以何种方式参与 posterior 输入（是否"绝对/相对"地构造 posterior 的输入）。这里讨论的是当前默认配置 `absolute=False` 的执行路径：posterior 输入是 `concat([deter, tokens])`。**如果改变该配置，`absolute=True` 时 posterior 输入就只是 `tokens` 本身，构造方式会发生变化。** 需要特别说明：这里的 `absolute` 只是 RSSM 实现内部用来控制 posterior 输入如何构造的一个配置项，**并不是 Transformer 意义上的 absolute positional encoding**——读者看到这个命名时不要联想到位置编码。因此，也不能把 `x_t = [h_t, o_t^emb]` 这一形式当成 RSSM 的数学定义本身——它只是默认 config 分支下的执行路径。

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

再次印证：prior 不读取 observation。

---

