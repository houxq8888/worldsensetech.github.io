---
title: "DreamerV3 训练工程实践：从 GPU 配置到超参调优"
slug: "2026-08-28-dreamerv3-training-tips"
date: 2026-08-28
draft: false
categories: ["世界模型"]
tags: ["DreamerV3", "训练技巧", "GPU", "超参数", "工程实践", "Dreamer系列"]
description: "DreamerV3 训练中的工程实战经验：GPU 显存优化、超参数调优、常见坑与解决方案。"
toc: true
---

> **Dreamer 系列 · 第 3 篇**
>
> 系列目录（当前在第 3 篇）：
> 1. [（一）读懂 Dreamer：世界模型是怎么学会'想象'的？](/zh/articles/2026-08-25-dreamer-explained/)
> 2. [（二）Dreamer 的 Actor-Critic：想象空间里的策略优化](/zh/articles/2026-08-27-dreamer-actor-critic/)
> 3. **（三）DreamerV3 训练工程实践：从 GPU 配置到超参调优**

前两篇文章从理论层面讲清楚了 Dreamer 的架构设计和 Actor-Critic 工作原理。但要把 DreamerV3 跑起来、训得好，还需要解决一堆工程问题：GPU 显存不够怎么办？超参数怎么调？训练不稳定怎么排查？

这篇文章从实战角度出发，总结 DreamerV3 训练中的工程经验和常见坑。内容基于 [danijar/dreamerv3@e3f02248](https://github.com/danijar/dreamerv3) JAX reference implementation。

## 一、环境配置：JAX + GPU

### JAX 的 GPU 支持

DreamerV3 使用 JAX 实现。JAX 的 GPU 支持依赖 CUDA 和 cuDNN，安装时需要注意版本匹配：

```bash
# 推荐安装方式（以 CUDA 12.x 为例）
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

验证 GPU 是否被正确识别：

```python
import jax
print(jax.devices())  # 应该看到 GPU 设备
```

如果输出只有 `[CpuDevice]`，说明 JAX 没有正确检测到 GPU。常见原因包括 CUDA 版本不匹配、cuDNN 未安装、或者环境变量配置问题。

需要注意的是：`nvidia-smi` 能看到 GPU 不等于 JAX 一定可用——CUDA driver 版本必须 ≥ runtime 版本，而且多版本 CUDA 环境容易互相污染。建议用更详细的诊断命令：

```python
import jax
jax.print_environment_info()  # 打印 JAX、CUDA、cuDNN 版本信息
```

### 显存管理

JAX 默认会预分配几乎所有 GPU 显存。在多任务训练或调试时，这可能造成问题。可以通过环境变量控制：

| 变量 | 作用 |
|------|------|
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | 关闭启动时大块预分配 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` | 限制预分配比例为 90% |

```bash
# 关闭预分配（按需分配）
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 或者限制预分配比例（仍预分配，但只占 90%）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

注意：`MEM_FRACTION=0.9` 不是"按需分配"，而是"预分配最多 90% 显存"。真正关闭预分配需要用 `PREALLOCATE=false`。

对于 24GB 显存的 GPU（如 RTX 3090/4090），DreamerV3 的默认配置通常可以跑通。如果显存紧张，需要调整 batch size 或 imagination length。

## 二、关键超参数

DreamerV3 的超参数在 `configs.yaml` 中定义。理解每个参数的作用，才能在具体任务上调得好。

### 世界模型相关

```yaml
# RSSM 结构
rssm:
  deter: 4096      # deterministic state 维度
  stoch: 32        # categorical variables 数量
  classes: 32      # 每个 categorical variable 的类别数
  hidden: 4096     # RSSM 内部 MLP 隐藏层维度

# imagination
imag_length: 15    # 想象轨迹长度
```

注意：`stoch=32` 和 `classes=32` 表示 stochastic state 由 32 个 categorical variables 组成，每个变量有 32 个类别。实际 latent 是 `32 × 32` 的 categorical distribution，而不是 32 维的连续向量。

KL divergence 在这个 categorical distribution 上以两种形式出现：

- **Representation KL**：$D_{KL}[q(z_t | h_t, o_t) \| p(z_t | h_t)]$，衡量 posterior（给定观测）和 prior（仅依赖历史）的差距
- **Dynamics KL**：$D_{KL}[p(z_t | h_t) \| q(z_t | h_t, o_t)]$，方向相反，用于 stabilizing training

两者的主要区别不仅在 KL 方向，还在于**梯度更新对象不同**：dynamics loss 主要更新 prior 网络，representation loss 主要更新 posterior encoder。这种分离的梯度流设计是 DreamerV3 防止 latent collapse 和保持预测能力的重要机制。

**`imag_length`** 是最重要的超参数之一。15 步是 DreamerV3 的默认值，在大多数任务上表现良好。

但要注意一个常见误解：**长 imagination 不等于更好的 long horizon credit assignment**。世界模型的预测误差 $p(s_{t+k})$ 随步数 $k$ 增大而快速累积，所以：

- **较长 imagination**（20-25 步）：提高长期规划能力，但增加模型偏差累积的风险
- **较短 imagination**（10-15 步）：更依赖 critic 的 bootstrap 估计，方差更低，训练更稳定

如果任务需要更长的 horizon 才能看到奖励信号，可以适当增加到 20-25，但前提是**世界模型本身足够准确**。否则，长 imagination 反而会把策略带偏。

### Actor-Critic 相关

```yaml
# Actor
policy:
  layers: 3
  units: 1024
  act: silu
  norm: rms
  minstd: 0.1      # 连续动作最小标准差
  maxstd: 1.0      # 连续动作最大标准差
  outscale: 0.01   # 输出层权重缩放
  unimix: 0.01     # 离散动作均匀混合系数

# Critic
value:
  layers: 3
  units: 1024
  output: symexp_twohot
  bins: 255        # two-hot distribution 的 bin 数量

# 训练
imag_loss:
  lam: 0.95        # lambda-return 的 lambda
  actent: 3e-4     # 熵正则系数
  slowtar: False   # 是否使用 slow value 作为 target
  slowreg: 1.0     # slow value network 正则权重

slowvalue:
  rate: 0.02       # EMA 更新速率
  every: 1         # 每步更新
```

**`actent`** 控制探索程度。默认 `3e-4` 在大多数任务上表现良好。但要注意，DreamerV3 中探索不是简单靠 `actent` 一个参数控制的——Actor loss 是 imagination return + entropy，而 policy distribution 本身受 `minstd`/`maxstd`（连续动作）和 `unimix`（离散动作）控制。

如果需要调整探索行为，**经验优先级**（非官方推荐顺序，不同任务可能不同）：

1. **minstd / maxstd**（连续动作影响最大）：直接改变策略分布的标准差范围
2. **unimix**（离散动作）：控制均匀混合系数，影响离散动作的随机性
3. **actent**（entropy coefficient）：作为最后微调手段

很多情况下，默认 `actent` 已经足够，问题往往出在 `minstd`/`maxstd` 的设置上。

**`lam`** 控制 TD 和 Monte Carlo 之间的权衡。0.95 偏向 Monte Carlo（多步回报），适合 imagination 比较准确的场景。如果世界模型不够准确，可以减小到 0.9 或 0.85，增加 TD bootstrap 的比重来降低方差。

### 优化器相关

```yaml
lr: 4e-5           # 学习率
opt: {eps: 1e-20, clip: 1000.0}  # Adam 优化器参数
```

DreamerV3 使用较小的学习率 `4e-5`，这是训练稳定的关键之一。`clip: 1000.0` 是梯度裁剪阈值，防止梯度爆炸。

## 三、显存优化实战

### 问题：OOM（Out of Memory）

当 batch size 较大或 imagination length 较长时，容易遇到显存不足。以下是几种解决方案：

**方案 1：减小 batch size**

```yaml
batch_size: 8      # 从 16 减到 8
batch_length: 512  # 序列长度也可以适当减小
```

**方案 2：减小 imagination length**

```yaml
imag_length: 10    # 从 15 减到 10
```

这会减少 imagination 阶段的计算量，但可能影响长期信用的分配。

**方案 3：使用梯度检查点**

JAX 支持梯度检查点（gradient checkpointing），用计算换显存：

```python
# 在关键函数上添加 gradient checkpointing
import jax

@jax.remat
def expensive_function(x):
    # ...
```

但这会增加约 20-30% 的训练时间。

**方案 4：混合精度训练（谨慎使用）**

DreamerV3 默认使用 float32。如果需要混合精度，**强烈推荐 bfloat16 而非 float16**，因为 bfloat16 的指数范围更大，数值更稳定。

但即使使用 bfloat16，也要谨慎：DreamerV3 对数值稳定高度敏感，涉及 KL balancing、categorical logits、two-hot value distribution、symlog/symexp 等组件。

**推荐策略：**

- 只在 encoder/decoder 和大型矩阵乘法使用低精度
- 不要在 RSSM latent dynamics、distribution logits、value head 使用低精度
- 如果训练出现 NaN 或 KL collapse，立即回退到 float32

```python
# JAX 中设置 matmul 精度（谨慎使用）
from jax import config
config.update("jax_default_matmul_precision", "bfloat16")
```

注意：`jax_default_matmul_precision` 控制的是**矩阵乘法内部精度选择**，不是模型 dtype。它不会自动完成 dtype casting，**不等同于 PyTorch 的 AMP（Automatic Mixed Precision）**。JAX 的精度控制（X64/X32/bf16）是全局或逐操作的显式设置，而 PyTorch AMP 通过 autocast 自动为不同操作选择合适精度。开了 `jax_default_matmul_precision` 不等于"开启了 bf16 training"，实际效果仅限于 matmul 运算的内部精度。

注意：float16 在 DreamerV3 中很容易导致 KL collapse、value explosion 或 NaN，不建议使用。

### 显存监控

训练时监控显存使用情况：

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者使用更详细的工具
nvtop
```

如果发现显存使用量在训练过程中逐渐增加，可能存在显存泄漏。JAX 的显存管理通常比较稳定，但某些自定义操作可能导致问题。

## 四、训练稳定性排查

### 问题 1：Reward 不增长

如果训练过程中，真实环境的累积奖励不增长，可能原因：

**世界模型没有学好**

检查世界模型的 loss 曲线：
- reconstruction loss 应该稳定下降
- KL loss 应该在合理范围内波动
- reward prediction loss 应该下降

如果世界模型 loss 不下降，可能是学习率太大或太小，或者 encoder/decoder 结构不适合当前任务。

**Actor-Critic 没有从想象中提取到有效信号**

检查 Actor-Critic 的 loss：
- advantage 的分布是否合理（不应该全是 0 或全是极大值）
- value prediction 是否和 lambda-return 接近
- entropy 是否在合理范围内

如果 advantage 方差很大，可能是 value network 没有学好，可以检查 slow value regularization（`slowreg`）、`lam`、reward scale 等因素。`slowreg` 不是第一调节项——先排查 reward scale 和 lambda 设置是否合理。

### 问题 2：训练突然崩溃

训练前期正常，但某个 step 突然 loss 变成 NaN 或 Inf：

**梯度爆炸**

检查梯度范数。DreamerV3 有梯度裁剪，但如果 `clip` 值设置太大，可能无法有效防止爆炸。可以尝试：

```yaml
opt: {clip: 100.0}  # 从 1000 减到 100
```

**数值溢出**

two-hot distribution 的 bins 覆盖的是 symlog 空间，映射回原始尺度后覆盖极大范围（约 ±4.8×10⁸），但**训练主要发生在压缩后的 symlog 空间**。DreamerV3 通过 symlog 压缩动态范围，使得 value prediction 不需要直接预测巨大的原始 return。

如果 value prediction 出现 NaN，检查：
- 真实 return 是否经过 symlog 变换（`symlog(x) = sign(x) * log(|x| + 1)`）
- two-hot 的 bin 范围是否足够
- value prediction 的分布是否合理

**imagination 起点质量差**

如果 replay buffer 中某些序列质量很差（比如 episode 很短，或者 reward 异常大），可能导致 imagination 产生异常值。可以检查 replay buffer 中序列的分布。

### 问题 3：不同 seed 结果差异大

如果换随机 seed 后训练结果差异很大，说明训练不稳定：

**增大 replay buffer**

```yaml
replay_size: 5e6   # 从 1e6 增大到 5e6
```

更大的 replay buffer 可以平滑采样分布，减少方差。

**减小学习率**

```yaml
lr: 2e-5           # 从 4e-5 减到 2e-5
```

**增大 batch size**

```yaml
batch_size: 32     # 从 16 增大到 32
```

更大的 batch size 可以提供更稳定的梯度估计。

**多 seed 评估**

RL 实验的 seed 敏感性是已知问题。如果你在做严肃的实验对比，建议：

- 至少跑 3-5 个不同 seed
- 报告中位数和置信区间（而非单条曲线）
- 如果只有单 seed 结果，结论的可信度会大打折扣

### 问题 4：Replay buffer 采样与训练节奏

Dreamer 对 replay buffer 的采样策略非常敏感，但这一点经常被忽略。只关注 `replay_size` 是不够的，还需要注意：

**Replay ratio（训练/采集比）**

Dreamer 不是"先训世界模型再训 Actor"，而是**联合更新**：

```text
采集数据 → 联合更新：世界模型 + Actor-Critic → 采集更多数据 → ...
```

DreamerV3 通常采用较高 replay ratio，但具体取决于环境速度和配置——对于仿真速度快的环境，ratio 可能远大于 1；对于真实机器人等慢速环境，ratio 可能接近 1。这个比例直接影响训练效率：

- ratio 太低：模型没有充分利用已有数据，训练慢
- ratio 太高：过拟合 replay buffer 中的旧数据，策略退化

DreamerV3 默认的 ratio 在大多数任务上已经比较合理，但如果训练效率不理想，可以调整。

**Warmup 阶段**

训练初期，replay buffer 中的数据量很少，世界模型还没有学好。这个阶段需要注意：

- 前几百步主要是随机探索，填充 replay buffer
- 通常应该在早期训练阶段看到世界模型 loss 的下降趋势，但具体步数依环境复杂度而变化——简单环境可能几百步就开始下降，复杂环境（Atari、机器人、稀疏奖励）可能需要更多步
- Actor-Critic 在世界模型初步稳定后才开始有意义地更新
- 如果 warmup 阶段过长仍无改善，检查环境接口和超参数

**Train/eval ratio**

训练和评估的比例也很重要。评估太频繁会浪费训练时间，评估太少则无法及时发现问题。建议每 1000-5000 步评估一次。

### 问题 5：世界模型坍缩诊断

世界模型坍缩（world model collapse）是 Dreamer 训练中最隐蔽的失败模式。与 reward 不增长不同，坍缩时 loss 曲线可能看起来完全正常。

**什么是坍缩？**

RSSM 的 latent space 逐渐失去信息表征能力。具体表现为：

- posterior entropy 持续下降，接近零
- prior 和 posterior 之间的 KL 趋近于零（模型"懒得"用 observation 信息）
- reconstruction loss 可能还在下降，但 latent state 已经不包含有用信息

**诊断方法：**

1. 监控 `posterior entropy` 和 `prior entropy` 的变化趋势
2. 如果两者都持续下降且差距缩小，可能正在坍缩
3. Latent embedding visualization（如 PCA/t-SNE）仅作为辅助参考——RSSM 的 categorical latent 和 temporal structure 使得 t-SNE 不一定可靠。更推荐的诊断手段是 latent probing、linear probe、或 rollout reconstruction

**修复策略：**

- 调整 KL free scale / free nats 时需要平衡 reconstruction 和 dynamics learning——free bits 的作用是忽略小 KL 区间内的优化压力，不是强迫 posterior 携带更多信息。设置过大会导致 posterior/prior 差异过大，prior 无法追踪 posterior，dynamics 学习困难
- 检查 encoder 容量是否足够（encoder 太弱会导致 posterior 没有信息）
- 减小 KL 权重（如果有的话），让 posterior 有更多自由度
- 检查 reconstruction loss 的权重——如果 recon 权重太低，encoder 没有动力提取有用信息

## 五、任务特定的调优建议

### Atari 游戏

Atari 游戏的 observation 是像素图像，reward 通常是稀疏的（只有得分变化时才有奖励）。

**推荐配置调整：**

```yaml
# 可以尝试增大 imagination length（需要确认 world model prediction quality）
imag_length: 20    # 先检查 imagined rollout 质量再决定是否增大

# 使用更激进的探索
policy:
  unimix: 0.05     # 增大均匀混合，增加随机性

# 调整 lambda，因为世界模型在像素空间可能不够准确
imag_loss:
  lam: 0.9         # 减小 lambda，增加 TD 比重
```

**注意事项：**

- DreamerV3 主要依赖 encoder 的 temporal modeling 和 RSSM hidden state 来捕捉时序信息，不像早期 DQN 那样依赖 frame stacking。不过某些实现仍可能使用多帧输入，具体看环境封装
- 某些游戏的 reward 范围很大，symlog 变换可以帮助处理
- 如果训练很慢，可以减小 `batch_length`，因为 Atari 的 episode 通常很长

### MuJoCo 控制

MuJoCo 是连续控制任务，observation 是低维状态向量，action 也是连续的。

**推荐配置调整：**

```yaml
# MuJoCo 通常不需要太长的 imagination
imag_length: 15    # 默认值通常够用

# 连续动作的标准差范围
policy:
  minstd: 0.1
  maxstd: 1.0

# 默认 entropy 通常足够，只有在探索不足时才增大
imag_loss:
  actent: 3e-4     # 默认值，如果探索不足再考虑增大到 1e-3
```

**注意事项：**

- 很多连续控制任务的默认 entropy 已经足够，盲目增大 `actent` 可能延迟收敛、降低最终性能
- MuJoCo 的 episode 通常有固定的时间长度，不需要 continuation model 做太多工作
- 如果策略过早收敛，优先调整 `minstd`/`maxstd`，其次才考虑增大 `actent`
- 如果训练不稳定，检查动作尺度是否需要归一化

### 机器人操控

机器人操控任务通常有较高的状态维度和较复杂的动力学。

**推荐配置调整：**

```yaml
# 机器人任务可能需要更长的 imagination 来理解因果
imag_length: 20

# 增大 RSSM 容量
rssm:
  deter: 4096
  hidden: 4096

# 如果 episode 经常提前终止，continuation model 很重要
# 确保 continuation prediction loss 正常下降
```

**注意事项：**

- 机器人任务的仿真通常很慢，replay buffer 的填充速度可能较慢
- 如果 episode 经常因为碰撞或掉落而提前终止，continuation model 的准确性很重要
- 真实机器人实验需要考虑 sim-to-real gap，DreamerV3 的 imagination 是在仿真中学习，部署到真实机器人时可能需要微调

**机器人任务特有的工程问题：**

- **Action delay / repeat**：真实机器人通常有 action delay，需要在环境封装中处理。Action repeat 可以缓解延迟问题，但会影响控制频率
- **Proprioception 和 observation normalization**：机器人本体感知（关节角度、速度等）和视觉观测的尺度差异很大，必须做 observation normalization
- **Contact dynamics**：接触动力学是机器人任务中最难建模的部分，世界模型在接触区域的预测往往不准确
- **Domain randomization**：如果要做 sim-to-real transfer，需要在仿真中引入随机化（纹理、光照、摩擦系数等），提高策略的鲁棒性
- **Observation latency**：多传感器同步和时间戳对齐是工程上容易忽略但影响很大的问题

## 六、训练监控与日志

### 关键指标

训练时应该监控以下指标：

**世界模型指标：**
- `loss/recon`：reconstruction loss，应该稳定下降
- `loss/kl`：KL divergence，应该在合理范围内波动
- `loss/dyn`：dynamics loss（prior → posterior 的 KL），反映 RSSM 的 prior 和 posterior 之间的差距
- `loss/rep`：representation loss（posterior → prior 的 KL），反映 encoder 表征质量
- `loss/reward`：reward prediction loss，应该下降
- `loss/cont`：continuation prediction loss，应该下降

**latent space 健康度（关键！）：**
- `posterior entropy`：后验熵，反映 latent state 的信息量
- `prior entropy`：先验熵，反映世界模型的预测不确定性

很多 Dreamer 训练失败不是 reward loss 不下降，而是 **latent collapse**——latent space 的信息量逐渐坍缩到接近零，posterior entropy 持续下降。这种情况下，世界模型 loss 可能看起来正常，但策略已经无法从 latent state 中提取有用信息。

如果发现 latent collapse 迹象：
- 注意：单个指标不能判断 collapse，需要联合观察 latent 信息量和任务性能。训练成功时 latent posterior 也可能变得更加确定，关键看是否伴随任务性能下降
- 调整 KL free scale / free nats 时需要平衡 reconstruction 和 dynamics learning，过大可能导致 prior 无法追踪 posterior
- 检查 encoder 是否过早收敛（recon loss 太低可能导致 posterior 和 prior 过度接近）

**Actor-Critic 指标：**
- `loss/actor`：Actor loss，应该稳定下降
- `loss/value`：Critic loss，应该稳定下降
- `loss/entropy`：策略熵，应该缓慢下降（探索减少）
- `stats/adv_max`、`stats/adv_min`：advantage 范围，不应该太极端
- `stats/ret_mean`：平均 return，应该逐渐增长

**系统指标：**
- `steps/sec`：训练速度
- `gpu_mem`：显存使用量
- `buffer_size`：replay buffer 大小

**进阶诊断指标：**

- **KL balance 比例**：`dynKL / repKL` 的比值，反映 prior 和 posterior 的学习是否平衡。如果比值严重偏离 1，说明某一侧学习困难
- **Decoder 预测质量**：不仅看 recon loss 数值，还应该可视化 reconstruction 图像样本。如果 recon loss 在下降但图像模糊或丢失关键细节，说明 encoder/decoder 容量可能不足
- **Imagined rollout sanity check**：这是 Dreamer 特有的诊断手段——直接可视化 `latent rollout → decoded observation`，观察想象出来的"未来画面"是否合理。这比看 loss 数值更直观，能快速发现世界模型是否学到了有意义的 dynamics

### 使用 TensorBoard

DreamerV3 支持 TensorBoard 日志：

```bash
# 启动训练时指定日志目录
python dreamerv3/train.py --logdir ./logs/my_experiment

# 启动 TensorBoard
tensorboard --logdir ./logs
```

在 TensorBoard 中可以直观地看到各项指标的变化趋势，方便调试。

### 定期保存 checkpoint

训练过程中定期保存模型，防止意外中断：

```yaml
# configs.yaml
save_every: 10000  # 每 10000 步保存一次
```

保存的 checkpoint 可以用于：
- 从断点恢复训练
- 评估不同训练阶段的策略性能
- 分析训练过程中的行为变化

## 七、常见坑与解决方案

### 坑 1：JAX 编译很慢

JAX 使用 XLA 编译器，第一次运行时会编译计算图，可能需要几分钟。这是正常的，后续运行会快很多。

**解决方案：**

- 第一次运行时耐心等待
- 使用 `jax.config.update("jax_compilation_cache_dir", "/path/to/cache")` 缓存编译结果
- 如果频繁修改代码导致重新编译，可以考虑使用 `jax.jit` 的 `static_argnums` 参数

### 坑 2：多 GPU 训练

DreamerV3 的官方 JAX 参考实现默认配置主要针对单个 accelerator，但这并不意味着"不支持"多 GPU。JAX 本身提供了 `jax.pmap`、`jax.shard_map`、PJRT 等并行机制，理论上可以进行多 GPU 扩展。

**实际情况：**

- 对于大多数任务，单 GPU 训练已经足够
- 官方没有提供开箱即用的多 GPU 配置，需要自行基于 JAX 并行 API 改写
- 更实用的做法是用数据并行：在不同 GPU 上运行不同的实验（不同超参或 seed）
- 如果需要扩展，参考 JAX 官方文档中的 [sharding 指南](https://jax.readthedocs.io/en/latest/sharding.html)

### 坑 3：Replay buffer 占用大量内存

DreamerV3 的 replay buffer 保存的是原始 trajectory 数据（observation、action、reward、continuation、episode 信息），而不是训练后的 latent sequence。训练时数据经过 encoder 才得到 latent。但如果存储原始 observation（尤其是像素图像），内存占用仍然可能很大。

**解决方案：**

```yaml
replay_size: 1e6   # 控制 replay buffer 大小
```

- 使用 `replay_size` 限制 buffer 大小
- 如果内存紧张，可以减小 `batch_length`，减少每个序列的长度
- 考虑使用磁盘存储而不是内存存储（需要改写代码）

### 坑 4：训练速度慢

DreamerV3 的训练速度受多个因素影响：

**可能原因：**
- GPU 性能不足
- batch size 太大
- imagination length 太长
- 世界模型容量太大

**解决方案：**
- 升级 GPU
- 减小 batch size 或 imagination length
- 减小 RSSM 的 `deter` 和 `hidden` 维度
- 使用更小的 encoder/decoder 网络

### 坑 5：symlog/symexp 数值问题

symlog 和 symexp 是 DreamerV3 处理尺度的关键，但如果输入值极端，可能导致数值问题。

**解决方案：**

- 确保输入数据已经适当归一化
- 检查 reward 的范围，如果 reward 特别大，考虑手动缩放
- 如果 value prediction 出现 NaN，检查 two-hot 的 bin 范围是否足够

## 八、实战经验总结

### 调参优先级

当需要调优时，建议按以下优先级尝试：

1. **环境接口正确性**：reward scale、action normalization、observation preprocessing、continuation 设置——这些错误会直接导致训练失败
2. **Replay / warmup 配置**：确保 replay buffer 正常填充，warmup 阶段合理
3. **Learning rate**：影响训练稳定性，DreamerV3 默认 `4e-5` 通常不需要大改
4. **Batch size**：影响梯度估计的稳定性
5. **RSSM 容量**：`deter` 和 `hidden` 维度，影响世界模型表征能力
6. **Imagination length**：影响长期信用的分配，但很多失败并不是 imagination length 的问题
7. **Entropy / 探索**：通过 minstd/maxstd → unimix → actent 的顺序调整

很多训练失败的根因是 reward scale 错误、action normalization 错误、continuation 配置错误等基础问题，而不是 imagination length 不够长。先排查基础配置，再调超参数。

### 训练 checklist

开始训练前，检查以下事项：

- [ ] GPU 驱动和 CUDA 正确安装
- [ ] JAX 能够检测到 GPU
- [ ] 环境接口符合预期（observation、action、reward 的 shape 和类型）
- [ ] replay buffer 能够正常填充
- [ ] 世界模型 loss 在前几百步开始下降
- [ ] Actor-Critic loss 在世界模型稳定后开始下降
- [ ] 显存使用量在安全范围内
- [ ] 训练速度符合预期

### 何时停止训练

训练不是越久越好。以下情况可以考虑停止：

- 真实环境的累积奖励已经饱和
- 连续几千步没有明显改善
- 策略开始出现不稳定的迹象（reward 波动很大）
- 训练时间已经达到预算上限

## 九、实验复现指南

Dreamer 类算法非常依赖实验条件，复现性是一个常见问题。做严肃实验时，建议建立以下习惯：

**Seed 管理**

- 至少跑 3-5 个不同 seed，报告中位数和置信区间
- Seed 应该记录在实验名称中，例如 `dreamerv3_atari_pong_seed42`
- 注意 JAX 的 seed 设置方式：`jax.random.PRNGKey(seed)`

**Config 保存**

- 每次实验开始时，将完整的 `configs.yaml` 保存到 logdir
- 如果有命令行参数覆盖，也要记录完整的最终配置
- 建议用 JSON 或 YAML 格式自动保存

**Checkpoint 命名**

- Checkpoint 文件名包含 step 数，例如 `checkpoint_100000.pkl`
- 保留多个时间点的 checkpoint，方便回溯分析
- 记录 best checkpoint 的评判标准（eval reward、train reward 等）

**Git commit 记录**

- 实验代码必须有对应的 git commit hash
- 如果有未提交的修改，记录 `git diff` 或 `git stash`
- 建议使用 experiment tracking 工具（如 MLflow、W&B）自动记录

**Hardware 记录**

- 记录 GPU 型号、显存大小、CUDA 版本、JAX 版本
- 不同硬件上的训练速度和最终性能可能有差异
- 多机实验时，确保环境一致性

## 十、把之前的文章串起来

```text
世界模型入门 → RSSM 深度解析 → RSSM 代码系列（6篇）
                                       ↓
                              Dreamer 系列 #1：整体架构
                                       ↓
                              Dreamer 系列 #2：Actor-Critic
                                       ↓
                              Dreamer 系列 #3：训练技巧（本篇）
                                       ↓
                              GPU 选型指南
```

如果你还没读过前两篇，建议先看 [Dreamer 整体架构](/zh/articles/2026-08-25-dreamer-explained/) 和 [Actor-Critic 详解](/zh/articles/2026-08-27-dreamer-actor-critic/)，再来读这篇训练技巧，会更有收获。

## 十一、总结

DreamerV3 的训练工程实践可以概括为：

- **环境配置**：JAX + GPU 的正确安装和显存管理是基础
- **超参数调优**：理解每个参数的作用，按优先级逐步调整
- **稳定性排查**：从世界模型到 Actor-Critic，逐层定位问题
- **任务适配**：不同任务类型需要不同的调优策略
- **监控与日志**：及时发现问题，避免浪费时间

DreamerV3 的设计已经非常鲁棒，默认配置在大多数任务上都能工作。但如果想要达到最佳性能，或者遇到训练问题，就需要深入理解每个组件的作用，才能有针对性地调优。

希望这篇实战指南能帮助你更顺利地训练 DreamerV3。如果遇到问题，欢迎在评论区讨论。
