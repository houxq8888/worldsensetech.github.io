---
title: "从代码理解 RSSM：DreamerV3 中 RSSM 的实现细节"
slug: "2026-08-19-rssm-code-walkthrough"
date: 2026-08-19
draft: false
categories: ["世界模型"]
tags: ["RSSM", "DreamerV3", "世界模型", "状态空间模型", "代码解析"]
description: "从代码层面拆解 DreamerV3 中 RSSM 的实现：确定性路径、随机性路径、先验/后验网络、重参数化技巧与 KL balancing。"
toc: true
---

前面两篇 RSSM 文章（[RSSM 状态空间模型详解](/zh/articles/rssm-deep-dive/)和[架构演进](/zh/articles/world-model-transformer/)）讲了 RSSM 的理论框架。但看论文和看代码是两回事——很多设计细节只有读代码才能理解。这篇文章直接从代码层面拆解 DreamerV3 中 RSSM 的实现，帮你建立从「数学公式」到「可运行代码」的映射。

## 先搞清楚 RSSM 在 DreamerV3 中的位置

DreamerV3 的整体架构可以分成三块：

1. **Encoder**：把观测（图像/状态）压缩成特征向量
2. **RSSM**：在隐空间中做时序建模——给定历史观测和动作，预测未来状态
3. **Decoder / Head**：从隐状态解码出预测的观测、奖励、价值等

RSSM 是中间那块，也是最核心的那块。它的输入是 encoder 输出的特征和 agent 的动作，输出是一个隐状态序列——后续所有预测（重建观测、预测奖励、计算价值）都基于这个隐状态。

用一句话概括 RSSM 的工作：**维护一个确定性隐状态（GRU），在每个时间步基于这个确定性状态输出一个随机采样的高斯分布，再从分布中采样得到随机隐状态。确定性状态 + 随机状态共同构成完整的隐表示。**

## RSSM 类结构总览

先看 RSSM 类的整体骨架（PyTorch 风格伪代码，尽量贴近 DreamerV3 的实际实现）：

```python
class RSSM(nn.Module):
    def __init__(self, obs_dim, act_dim, deter_dim, stoch_dim, hidden_dim):
        super().__init__()
        self.deter_dim = deter_dim  # 确定性隐状态维度，通常 1024
        self.stoch_dim = stoch_dim  # 随机隐状态维度，通常 32

        # 确定性路径：GRU
        self.gru = nn.GRUCell(act_dim + obs_feat_dim, deter_dim)

        # 先验网络：从 (deter) 预测 (mean, std)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)  # 输出 mean 和 log_std
        )

        # 后验网络：从 (deter + obs_feat) 预测 (mean, std)
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + obs_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )

    def forward(self, obs_feat, action, prev_state):
        # obs_feat: encoder 输出的特征 [B, obs_feat_dim]
        # action: 动作 [B, act_dim]
        # prev_state: (prev_deter, prev_stoch) 上一步的隐状态

        prev_deter, prev_stoch = prev_state

        # Step 1: 更新确定性状态
        gru_input = torch.cat([obs_feat, action], dim=-1)
        deter = self.gru(gru_input, prev_deter)

        # Step 2: 先验网络预测（不依赖当前观测）
        prior_mean, prior_log_std = self._split_dist(self.prior_net(deter))

        # Step 3: 后验网络预测（依赖当前观测特征）
        post_input = torch.cat([deter, obs_feat], dim=-1)
        post_mean, post_log_std = self._split_dist(self.post_net(post_input))

        # Step 4: 从后验分布采样（训练时）
        stoch = self._sample(post_mean, post_log_std)

        # 返回当前状态和分布信息（用于计算 KL）
        state = (deter, stoch)
        dist = {
            'prior': (prior_mean, prior_log_std),
            'post': (post_mean, post_log_std)
        }
        return state, dist

    def _split_dist(self, params):
        mean, log_std = params.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)  # 数值稳定
        return mean, log_std.exp()

    def _sample(self, mean, std):
        # 重参数化技巧
        eps = torch.randn_like(std)
        return mean + std * eps
```

这个骨架涵盖了 RSSM 的核心逻辑。下面逐个拆解关键设计。

## 确定性路径：GRU 到底在做什么

```python
gru_input = torch.cat([obs_feat, action], dim=-1)
deter = self.gru(gru_input, prev_deter)
```

这行代码做的事情很简单：把 encoder 输出的观测特征和当前动作拼接，作为 GRU 的输入，更新确定性隐状态。

**为什么用 GRU 而不是 LSTM？** DreamerV3 的论文和代码都选择了 GRU。原因不是 LSTM 不行，而是 GRU 参数更少（少一个门），在隐空间模型中过拟合风险更低。RSSM 的确定性路径本质上是在做「隐空间的时序积分」——给定历史所有观测和动作的压缩信息，维护一个不断更新的上下文向量。GRU 足够完成这个任务。

**`deter_dim` 为什么是 1024？** 这个维度需要同时编码两件事：(1) 历史观测的时序上下文，(2) 为后续的先验/后验网络提供足够的信息容量。DreamerV3 论文中的 ablation 显示，deter_dim 从 512 增加到 1024 有明显提升，再增加到 2048 提升不大。1024 是性价比的平衡点。

**一个容易忽略的细节：** GRU 的输入是 `obs_feat`（encoder 输出），不是原始观测。这意味着 encoder 的质量直接决定了 GRU 能拿到多少信息。如果 encoder 把关键信息丢了，GRU 再强也补不回来。

## 随机性路径：为什么需要两种状态

```python
# 先验：只看确定性状态
prior_mean, prior_std = self._split_dist(self.prior_net(deter))

# 后验：看确定性状态 + 观测特征
post_input = torch.cat([deter, obs_feat], dim=-1)
post_mean, post_std = self._split_dist(self.post_net(post_input))

# 训练时用后验采样
stoch = self._sample(post_mean, post_std)
```

这是 RSSM 最核心的设计，也是最反直觉的部分：**为什么需要两个网络、两个分布？**

先理解两个网络的角色：

**先验网络（prior）**：只根据确定性状态 `deter` 预测下一个随机状态。它代表的是「在没有看到新观测的情况下，模型对未来的预测」。这个网络在训练时用于计算 KL 散度，在推理/想象时用于生成想象轨迹（因为没有真实观测可用）。

**后验网络（posterior）**：根据确定性状态 `deter` 和当前观测特征 `obs_feat` 预测下一个随机状态。它代表的是「在看到当前观测之后，模型对当前状态的估计」。训练时用后验采样，因为它有观测信息，估计更准确。

**为什么不能只用一个？** 如果只有后验网络，推理时就无法工作——推理时没有真实观测，你没法计算后验。如果只有先验网络，训练时模型只能靠自己的预测来学习，没有观测信号修正，容易漂移。

**先验和后验的关系：** 训练的目标是让先验网络逐渐逼近后验网络。理想情况下，如果模型学到了足够好的世界动态，先验网络的预测应该和后验网络差不多——也就是说，模型能仅凭历史信息和动作就准确预测未来状态，不需要每次都依赖观测修正。KL 散度就是用来度量这个差距的。

## 重参数化技巧：梯度怎么穿过采样

```python
def _sample(self, mean, std):
    eps = torch.randn_like(std)
    return mean + std * eps
```

这段代码看起来简单，但它解决了一个关键问题：**如何让梯度穿过随机采样操作？**

如果你直接写 `stoch = torch.normal(mean, std)`，采样操作是不可导的——梯度无法从 loss 回传到 mean 和 std 的计算过程中。重参数化技巧把采样改写成 `mean + std * eps`，其中 `eps` 是从标准正态分布采样的固定噪声。这样，`stoch` 对 `mean` 的梯度是 1，对 `std` 的梯度是 `eps`，梯度可以正常回传。

**实现中的数值陷阱：**

```python
log_std = torch.clamp(log_std, min=-20, max=2)
return mean, log_std.exp()
```

网络输出的是 `log_std` 而不是 `std`，原因有两个：(1) `log_std` 的值域是整个实数轴，更容易用线性层输出；(2) 取 `exp` 后 `std` 一定是正数，不需要额外约束。但 `exp` 操作在 `log_std` 很大时会溢出，所以必须 clamp。DreamerV3 中 clamp 的范围是 `[-20, 2]`，对应 `std` 的范围大约是 `[2e-9, 7.4]`。

## KL 散度与 Free Bits

```python
def kl_divergence(prior_mean, prior_std, post_mean, post_std):
    # 两个高斯分布之间的 KL 散度（解析解）
    var_ratio = (prior_std / post_std).pow(2)
    mean_diff = (post_mean - prior_mean).pow(2) / post_std.pow(2)
    kl = 0.5 * (var_ratio + mean_diff - 1.0 + torch.log(post_std.pow(2) / prior_std.pow(2)))
    return kl.sum(dim=-1)  # 对随机维度求和
```

KL 散度衡量的是后验分布和先验分布之间的差距。训练时我们希望这个差距尽量小——先验越接近后验，说明模型仅凭历史信息就能准确预测未来。

**但 KL 不能直接加到 loss 里。** 如果 KL 权重太大，模型会倾向于让先验和后验完全一样，这意味着后验不再利用观测信息——退化成只用先验。这会导致重建质量下降。

DreamerV3 用了一个叫 **KL balancing + free bits** 的技巧：

```python
# KL balancing：不是简单加权，而是动态调整
kl_loss = kl_divergence(prior, posterior)

# Free bits：每个随机维度至少保留 info 量的信息
# 防止 KL 太小导致后验退化为确定性编码
kl_loss = torch.clamp(kl_loss, min=free_bits)  # free_bits 通常约 1.0

# 最终 loss 中的 KL 项
total_loss = recon_loss + kl_loss * kl_scale - reward_loss - value_loss
```

**Free bits 的作用：** 如果 KL 太小（先验和后验太接近），说明模型没有充分利用观测信息。Free bits 设置了一个下限，强制每个随机维度至少保留一定量的信息。这相当于告诉模型：「你可以让先验接近后验，但不能太接近——后验必须比先验多知道至少 `free_bits` 的信息。」

**kl_scale 的调度：** DreamerV3 不是固定 KL 权重，而是用了一个渐进式调度——训练初期 KL 权重小（让模型先学好重建），后期逐渐增大（让先验逐渐逼近后验）。这避免了训练初期 KL 约束太强导致模型学不到东西。

## RSSM 在序列上的展开

上面的代码是单个时间步的。实际训练中，RSSM 需要在一个完整的 episode 序列上展开：

```python
def forward_sequence(self, obs_feats, actions):
    # obs_feats: [T, B, obs_feat_dim]
    # actions: [T, B, act_dim]
    T, B = obs_feats.shape[:2]

    # 初始化状态
    deter = torch.zeros(B, self.deter_dim, device=obs_feats.device)
    stoch = torch.zeros(B, self.stoch_dim, device=obs_feats.device)

    prior_means, prior_stds = [], []
    post_means, post_stds = [], []
    stochs = []

    for t in range(T):
        state, dist = self.forward(obs_feats[t], actions[t], (deter, stoch))
        deter, stoch = state

        prior_means.append(dist['prior'][0])
        prior_stds.append(dist['prior'][1])
        post_means.append(dist['post'][0])
        post_stds.append(dist['post'][1])
        stochs.append(stoch)

    # 堆叠成序列
    return {
        'deter': deter,  # 最后一步的确定性状态
        'stoch': torch.stack(stochs, dim=0),  # [T, B, stoch_dim]
        'prior': (torch.stack(prior_means), torch.stack(prior_stds)),
        'post': (torch.stack(post_means), torch.stack(post_stds))
    }
```

**几个工程细节：**

1. **初始状态全零。** 第一个时间步没有 prev_state，所以 `deter` 和 `stoch` 初始化为零向量。这意味着前几步的预测质量会比较差——模型没有历史信息可用。实际训练中，通常会丢弃前几步的 loss，或者用较长的 sequence 来稀释初始状态的影响。

2. **逐步展开 vs 并行。** 上面的代码用 for 循环逐步展开，因为每个时间步依赖上一步的状态。这意味着 RSSM 的序列维度不能像 Transformer 那样并行计算。这是 RSSM 的一个固有限制——GRU 的递推性质决定了它必须串行。对于长序列（T > 200），这会成为训练瓶颈。

3. **stoch 的拼接方式。** 最终的隐表示是 `deter` 和 `stoch` 的拼接（`[deter; stoch]`），这个拼接向量会被送入 decoder 和各个 head（reward head、value head 等）。所以 decoder 的输入维度是 `deter_dim + stoch_dim`，通常是 1024 + 32 = 1056。

## 推理/想象模式：没有观测怎么办

训练时 RSSM 用后验采样。但到了推理阶段（或者 DreamerV3 的「想象」阶段），没有真实观测，只能用先验：

```python
def imagine(self, prev_state, actions):
    # actions: [T, B, act_dim]
    T, B = actions.shape[:2]
    deter, stoch = prev_state

    imagined = []
    for t in range(T):
        # 没有观测，只用先验
        deter = self.gru(
            torch.cat([torch.zeros(B, obs_feat_dim), actions[t]], dim=-1),
            deter
        )
        prior_mean, prior_std = self._split_dist(self.prior_net(deter))
        stoch = self._sample(prior_mean, prior_std)
        imagined.append(torch.cat([deter, stoch], dim=-1))

    return torch.stack(imagined, dim=0)  # [T, B, deter_dim + stoch_dim]
```

**注意两个变化：**

1. **obs_feat 用零向量替代。** GRU 的输入本来是 `[obs_feat; action]`，想象时没有观测，所以用零向量填充。这意味着 GRU 在想象模式下只从 action 获取信息。这是一个有争议的设计——有些实现会直接修改 GRU 的输入结构，让它在想象模式下只接收 action。

2. **从先验采样。** 没有后验网络参与，完全依赖先验网络的预测。这意味着想象轨迹的质量完全取决于先验网络学到了多好的世界动态模型。如果先验网络的预测有误差，误差会在想象过程中逐步累积——想象越长，偏差越大。

## 实际代码中的工程细节

**LayerNorm 的位置：** DreamerV3 在先验/后验网络中大量使用 LayerNorm。这不是装饰——隐空间模型的训练稳定性高度依赖归一化。没有 LayerNorm，RSSM 的 KL 散度容易在训练初期就爆炸，导致整个训练崩溃。

**GELU vs ReLU：** DreamerV3 统一使用 GELU 激活函数。相比 ReLU，GELU 在负值区域有非零梯度，有助于避免「dead neuron」问题。在隐空间模型中，这个问题尤其敏感——因为 RSSM 的输出会被反复使用（序列展开），一个 dead neuron 的影响会被放大。

**梯度裁剪：** DreamerV3 对 RSSM 的梯度做了全局裁剪（max norm = 1000）。这个值看起来很大，但实际上 RSSM 的梯度确实可能很大——因为序列展开的 BPTT（Backpropagation Through Time）会导致梯度累积。不设裁剪的话，偶尔的梯度尖峰会直接导致训练崩溃。

**对称权重初始化：** DreamerV3 使用对称权重初始化（symmetric weights initialization），即让权重矩阵的特征值均匀分布在单位圆上。这对 GRU 的长期依赖学习有帮助——避免初始状态下梯度消失或爆炸。

## 小结

从代码角度看，RSSM 的核心设计可以总结为：

| 组件 | 作用 | 关键设计 |
|:---|:---|:---|
| GRU | 维护确定性时序上下文 | 输入 = encoder 特征 + 动作 |
| 先验网络 | 无观测时的状态预测 | 输入 = deter，输出高斯分布 |
| 后验网络 | 有观测时的状态估计 | 输入 = deter + obs_feat |
| 重参数化 | 让梯度穿过采样 | mean + std * eps |
| KL balancing | 约束先验逼近后验 | free bits 防止退化 |
| 想象模式 | 无观测的 rollout | 只用先验，obs 用零填充 |

RSSM 的精妙之处在于它用一种简洁的方式统一了「有序观测时的滤波」和「无观测时的预测」这两个看似不同的任务。先验和后验共享同一个确定性状态（GRU），通过 KL 散度对齐两者的分布，最终让先验网络也能达到接近后验的预测精度——这就是「世界模型」能在想象中规划未来的基础。

理解了这些代码细节，再回去看 DreamerV3 的论文，很多公式就不再是抽象的符号，而是有具体对应的实现。
