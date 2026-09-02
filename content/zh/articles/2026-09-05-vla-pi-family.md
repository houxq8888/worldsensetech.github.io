---
title: "VLA 深度解读（中）：π₀ 家族与动作接口的演进"
slug: "2026-09-05-vla-pi-family"
date: 2026-09-05
draft: false
categories: ["具身智能", "论文解读"]
tags: ["VLA", "π₀", "π₀.5", "π₀.7", "Flow Matching", "Action Chunking", "Physical Intelligence", "具身智能"]
description: "VLA 系列三篇的中篇。π₀ 用 flow matching 实现连续动作生成，π₀.5 引入离散-连续混合 recipe，π₀.7 用 context-rich steering 和视觉子目标扩展泛化能力——拆解动作接口演进的核心技术密度，并提出 Training Interface vs Execution Interface 解耦这条 meta-axis。"
toc: true
related_articles:
  - 2026-09-03-vla-deep-dive
  - 2026-09-07-vla-world-models
  - 2026-09-02-jepa-deep-dive
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
---

> **VLA 系列共三篇：** [上篇：RT-2 到 OpenVLA](/zh/articles/2026-09-03-vla-deep-dive/) · 中篇（本文）· [下篇：VLA 与世界模型](/zh/articles/2026-09-07-vla-world-models/)

在[上篇](/zh/articles/2026-09-03-vla-deep-dive/)中，我们走过了 RT-2、OpenVLA 和 OFT，看到了 VLA 的基础架构和六条演进轴线。OFT 的结果揭示了一个关键信号：action interface 本身就是重要的系统设计轴。从这篇开始，我们进入技术密度更高的部分——π₀ 家族的三代模型。

## 一、π₀：Flow Matching + Action Chunking

### 论文信息

*π₀: A Vision-Language-Action Flow Model for General Robot Control*，Physical Intelligence，2024 年 10 月。arXiv:2410.24164。

### Physical Intelligence 背景

Physical Intelligence 是一家总部位于旧金山、专注通用机器人基础模型的公司。其 π 系列代表了目前连续动作 VLA 的重要技术路线之一。

### 核心架构

π₀ 做了一个和 RT-2 / OpenVLA 不同的选择：**不用离散 token，用 flow matching 生成连续动作。**

架构分为两部分：

| 组件 | 说明 |
|------|------|
| VLM 骨干 | PaliGemma（3B 参数视觉语言模型） |
| 动作专家 | 300M 参数的专用网络，挂在 VLM 后面 |
| **总参数** | **约 3.3B** |

需要注意的是，**action expert 不是传统意义上的外挂 controller。** π₀ 实际上采用了类似 two-expert mixture 的结构：图像/文本主要使用 VLM backbone 的第一套 weights，而机器人 state/action token 使用独立的一套 action-expert weights；两者通过 attention 共享上下文。更准确的描述是：**一个语言视觉 backbone + 一个连续动作生成专家，通过 attention 机制共享条件信息。**

输入包括图像 token、语言 token 和本体感知（proprioception），经过共享表征后由 action expert 通过 flow matching 输出 action chunk。

### 连续动作生成的三种机制

连续动作生成有三种主要机制，需要区分清楚。它们**不是先后替代关系，而是三种平行的连续动作建模方式**：

```
             Continuous Action Modeling
                        │
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
     Regression      Diffusion    Flow Matching
          │             │             │
    direct output   denoising      vector field
                    sampling       + ODE
```

Flow matching 训练的是概率路径上的 vector field / velocity field，使模型能够通过 ODE integration 从简单先验分布向目标动作分布 transport。它的做法是：

- 定义一条从简单先验分布到目标动作分布的概率路径（linear-Gaussian probability path）
- 训练一个网络来预测这条路径上的速度场
- 推理时从先验出发，沿学到的向量场积分，得到连续动作

和 diffusion 的关键区别在于：两者的训练目标和推理形式不同。π₀ 采用 flow matching，并在具体实现中使用少量 Euler integration steps 进行 action generation，从而将连续生成过程控制在适合机器人实时执行的计算预算内。

### 动作 Chunk、时序抽象与规划：三个容易混淆的概念

π₀ 每次预测一个长度为 **50 steps 的 action chunk**，每个 step 对应当前 embodiment 的动作向量（最多 18 DoF）。如果以 50 Hz 的控制频率计算，50-step chunk 在时间尺度上相当于约 1 秒——但 chunk length 本身不是时间长度，真正的时间跨度 T = H / f，换 embodiment 或控制频率后同样 50 steps 的时间跨度会不同。论文报告其系统在 dexterous tasks 上可以达到最高 50 Hz 的系统控制频率。

需要注意的是，flow matching 推理本身还需要进行多步 integration。50 Hz 是包含 action chunk 执行在内的系统级控制频率，不是单次 flow matching 推理的速度。

实际执行采用 receding-horizon / action chunk execution：模型生成 50 步 chunk 后执行前几步，再重新观测、生成下一个 chunk——并非 open-loop 执行整段 1 秒的动作。

**这里需要特别澄清三个容易混淆的概念：**

**Action chunking**——一次输出 a_t, a_{t+1}, ..., a_{t+H-1}。解决的是减少决策频率、提高动作连贯性。π₀ 的 50-step chunk 属于这个层面。

**Temporal abstraction**——把长任务压缩到更高层的时间尺度。比如"拿起杯子"是一个高层行为，而不是 50 个关节动作。π₀.5 的 semantic subtask 属于这个层面。

**Planning**——需要考虑动作序列会导致什么状态变化，并评估"哪个未来更好"。这才是 world-model planning 的核心。

**Action chunk ≠ planning。** π₀ 生成 50 步动作 chunk 后执行前几步，再重新观测、生成下一个 chunk。这是 **receding-horizon policy execution**，不等于在内部模拟多个候选未来然后比较选择。**Chunking 改变的是 policy 的输出时间尺度，并没有改变 policy 的决策准则**——从 π(a_t|o_t) 变成 π(a_{t:t+H-1}|o_t)，并不自动变成 argmax_a E[J(z_{t+H})|z_t,a]。后者才开始进入 prediction/planning。

### Cross-Embodiment Action Space

π₀ 将不同 embodiment 的动作映射到**最多 18 DoF 的公共 action representation**，不足部分通过 padding / masking 处理。例如单臂和双臂系统可以分别占用其中不同数量的 action dimensions：

```
不同 embodiment
      ↓
映射到最多 18 DoF 的公共 action representation
      ↓
不足部分 padding / masking
```

这使得同一个模型可以控制不同形态的机器人——这是 π₀ 的一个重要贡献，也是 embodiment interface 维度的早期实践。

### 训练

可以粗略理解为"初始化 + 机器人预训练 + 后训练"的多阶段 recipe，但具体数据混合和训练过程比这一简化图更复杂：

**VLM 初始化阶段：** 继承 PaliGemma 的互联网视觉语言预训练知识。π₀ 并非自己重新做互联网预训练，而是从已经 pretrained 的 PaliGemma 初始化。

**π₀ robot pre-training：**
- Open X-Embodiment "Magic Soup" 子集
- 自有多机器人数据：约 **10,000 小时**，约 **903M timesteps**（其中 106M 来自 single-arm，797M 来自 dual-arm），覆盖 **68 个任务**、**7 种硬件配置**。最大 action/configuration dimension 为 **18**，对应两只 6-DoF arm + 2 grippers + mobile base + vertically actuated torso。

**阶段二——定向后训练：**
- 在高质量、精选的任务 demo 上微调，掌握复杂操作技能

### 关键结果

在论文报告的 zero-shot real-world evaluation protocol 下，π₀ 在复杂操作任务上展现了离散 token 方案所不具备的能力：

| 任务类型 | π₀ 论文中的观察 |
|---------|---------------|
| 衣物折叠 | 展示了复杂、多物体、长时序的 dexterous manipulation |
| 桌面清理 / bussing | 展示了跨 embodiment 的复杂操作与语言条件执行 |
| 新技能 | 通过 post-training 学习 pre-training 中未覆盖或差异较大的任务 |
| 对比离散 VLA | 在这些特定 zero-shot protocol 上明显更强 |

需要注意，这些观察高度依赖具体的 task protocol（trial 数量、机器人形态、task definition、"zero-shot"的具体含义等），因此不宜理解成跨模型的普适性能排序。这些结果说明连续生成路线在这类任务上具有潜力，但不能单独归因于"连续表示"本身；动作表示、生成机制、chunking、数据规模和训练 recipe 是耦合变化的。

### 如何理解 π₀ 的性能提升

从 RT-2 到 π₀，连续动作生成确实成为越来越重要的方向。但目前还不能把性能提升简单归因于"连续表示优于离散 token"。π₀ 同时引入了 action chunking、flow matching、跨 embodiment 训练数据以及更大规模的机器人数据。更准确的结论是：**连续动作表示解决了量化和自回归输出的一部分结构性问题，但它的最终收益与数据规模、动作 chunking 和训练 recipe 强烈耦合。**

### 局限

- **代码和部分模型 checkpoint 已通过 openpi 开源，但训练数据、完整内部训练体系以及所有生产/实验变体并未全部开放**
- 某些需要精确力控的任务仍不可靠
- 对全新物理域（自动驾驶、飞行器）的泛化能力未知
- VLM 微调可能导致语言/视觉能力退化（catastrophic forgetting）

## 二、π₀.5 与 π₀.7：从动作生成到策略引导

### π₀.5：数据组成 + 层次化策略

π₀.5（2025 年 4 月，arXiv:2504.16054）的核心贡献不是简单地"加入高层推理"，而是引入了一个**层次化架构**，以及一个非常重要的技术事实：**离散和连续动作表示可以在同一个 foundation model 中承担不同角色。**

π₀.5 的训练分为两个阶段，使用了不同的动作表示和 prediction target：

```
                 π₀.5

Pre-training
┌───────────────────────────┐
│ VLM + FAST autoregressive │
│ action token prediction   │
│ α = 0 (无 flow loss)      │
└──────────────┬────────────┘
               ↓
Post-training
┌───────────────────────────┐
│ 保留 FAST sequence model  │
│ + 加入 flow action expert │
│ α > 0 (flow loss 开启)    │
└──────────────┬────────────┘
               ↓
Inference
language/subtask
      ↓
semantic prediction (FAST AR)
      ↓
flow matching
      ↓
continuous action chunk
```

**π₀.5 不是简单地把离散策略替换成连续策略。** 同一个模型在 pretraining 时同时有 FAST autoregressive action prediction 和 flow-field prediction，但 α=0 即 flow loss 未开启；post-training 时 α>0，在保留 FAST sequence model 的基础上加入 flow action expert。也就是说，**FAST 离散建模并没有被替换，而是在 post-training 阶段与 flow matching 共存。**

**预训练阶段**使用 **FAST action tokenizer** 将机器人动作离散化，使动作可以和语言、语义子任务等 token 共享 next-token prediction 训练。这使得大规模异构数据（不同机器人、不同任务、甚至 web 数据）可以在统一的 sequence modeling 框架中被利用。高层 **hierarchical task decomposition / semantic subtask prediction** 是这一阶段的重要组成部分。

**后训练阶段**在保留 FAST sequence model 的基础上，**加入 flow-matching action head**（α>0），负责高频连续控制，针对 mobile manipulation 做 post-training。推理时，模型先通过 FAST 推断一个高层语义子任务（如 "pick up the pillow"），然后基于这个子任务用 flow matching 生成连续动作 chunk。

一个值得注意的数据是：π₀.5 的第一阶段训练数据中，约 **97.6% 并不是 mobile-manipulator household data**。它通过大量异质数据的混合训练来获得泛化能力，然后在少量目标域数据上微调。

π₀.5 因此说明"离散 vs 连续"不是一个简单的替代关系。它的实际路线不是 "discrete → continuous"，而是 **discrete for scalable multimodal pretraining + continuous for fine-grained control**。

从 π₀.5 和 π₀-FAST 的设计来看，一个越来越清晰的工程分工是：**离散表示更适合 foundation-model pretraining 和 multimodal sequence modeling，而连续生成更适合最终的高精度控制。** 如果压缩成一句话，可以概括为：

> **离散负责统一，连续负责控制。**

这是本文对 π₀ / π₀.5 / π₀-FAST 技术演进的总结性解释，而非某一篇论文已经证明的普适设计原则。

### 一个更深的变化：Training Interface 与 Execution Interface 开始解耦

回顾 RT-2 到 π₀.5 的演进，有一条比"六条轴"更深的 meta-axis 正在浮现：

```
                 Foundation Model
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
       Training Interface   Execution Interface
             │                   │
       discrete tokens      continuous actions
       FAST / AR             flow / regression
             │                   │
       scalable learning     high-frequency control
```

具体来看：

- **RT-2**：training = discrete action tokens，execution = autoregressive action tokens
- **OpenVLA**：training = autoregressive action tokens，execution = autoregressive action tokens
- **OpenVLA-OFT**：foundation model representation → continuous action head → parallel execution
- **π₀**：VLM semantic representation → continuous flow action expert → high-frequency execution
- **π₀.5**：最有意思——pretraining interface = FAST discrete tokens，post-training / execution interface = flow continuous action

这说明真正发生的可能不是简单的 "discrete → continuous"，而是 **foundation-model learning interface 和 robot execution interface 开始解耦**。离散 token 服务于 scalable multimodal learning，连续生成服务于 high-frequency control——两者可以共存于同一个系统。

这个框架可以把 RT-2 → OpenVLA → OFT → π₀ → π₀.5 的整个故事串起来，也为理解后续模型提供了更有解释力的视角。

π₀.5 展示了分钟级长时序 household manipulation，并在论文中讨论了 10–15 分钟级别的长任务能力；其定量 real-home evaluation 中的具体任务则主要持续约 2–5 分钟。需要注意的是，"能够执行 10–15 分钟任务"与"在 10–15 分钟任务上有系统性 benchmark"不是一回事。长程任务中的错误累积仍然是 VLA 泛化的主要瓶颈。

### π₀-FAST：为什么离散 token 并没有消失？

Physical Intelligence 后续还探索了 **π₀-FAST**，用 FAST action tokenizer 把连续 action chunk 压缩成离散 token，使机器人动作重新进入 autoregressive language-modeling framework。

这说明 Physical Intelligence 自己也没有认为 discrete action tokenization 是一条死路。实际上，**离散 token 对 multimodal pretraining 仍然有巨大价值**——它使机器人动作可以和语言、视觉、语义子任务共享同一个 sequence modeling 接口。

这里值得把离散和连续各自的优势做一个显式对比：

**离散 action token 的优势：**
- 与 LLM/VLM 的 autoregressive objective 天然对齐
- 混合语言、视觉、动作到同一 token 空间
- 使用大规模 multimodal pretraining
- 数据格式统一

**连续 action 的优势（a ∈ ℝ^D）：**
- 避免离散 bin 的量化误差
- 与 diffusion / flow matching 等连续分布建模方法兼容
- 可以直接生成连续 action chunk
- 更适合高精度控制

所以最终不是 "discrete → continuous"，而是 **tokenization 和 continuous generation 可能服务于不同阶段/不同层级。**

### π₀.7：从 task conditioning 到 context-rich policy steering

π₀.7（2026 年 4 月，arXiv:2604.15483）进一步探索了 VLA 的"可引导泛化"（steerability）。

从 RT-2 到 π₀.7，policy 的输入发生了质的变化：

```
RT-2：  language → action
π₀：    language + image + proprioception → action chunk
π₀.5：  language + observation → semantic subtask → action chunk
π₀.7：  language + subtask + episode metadata + subgoal image + observation history + proprioception + control mode → action chunk
```

π₀.7 的真正进步不只是"多了一些输入"，而是：**prompt 从"描述我要做什么"变成了"描述我应该如何做"。** 也就是从 **task conditioning** 逐渐走向 **context-rich policy steering**。这恰好也是为什么 π₀.7 的标题用了 "Steerable Generalist"。

模型不再只把语言指令作为条件，而是把语言、episode metadata、执行策略信息、视觉子目标以及观测历史等多模态上下文统一作为 policy 的条件输入。

**从建模角度看，π₀.7 可以理解为试图解决异质数据带来的条件冲突问题。** 同一个任务的不同 demo 可能呈现完全不同的行为模式——快速但粗糙、慢但高质量、包含错误、不同策略、不同机器人。如果训练数据只有 (task, observation) → action，那么这些行为模式对于 policy 来说可能是**相互冲突的 supervision**。π₀.7 的解决方案是增加条件变量（subtask、strategy/metadata、quality、speed、subgoal image、control mode 等），**不是单纯增加数据，而是增加"数据为什么不同"的条件变量，使原本相互冲突的 action supervision 变得 conditionally consistent。** 这是一种对 π₀.7 设计动机的机制性解释，而不是论文通过因果实验单独证明的定理。

π₀.7 的模型规模约为 5B，由约 4B 的 VLM backbone、视频历史编码模块（MEM-style video history encoder）以及 860M 参数的 action expert 组成。**π₀.7 VLA 本体是约 5B 的模型。** 完整的实验推理栈还可以在它之外加入一个 high-level semantic policy，以及一个基于 14B BAGEL 初始化的 subgoal-image world model，用于生成视觉子目标。它仍然沿用 flow matching 连续动作生成路线。

值得注意的是，**π₀.7 的核心 VLA 本身并不是 action-conditioned world model**。但完整推理系统在 VLA 外部增加了一个**基于 BAGEL 初始化的视觉生成世界模型**（约 14B 参数），用于根据当前观察、子任务和上下文生成候选视觉子目标（candidate visual subgoal），再将其作为低层 VLA 的条件输入。整个系统可以画成：

```
                  π₀.7 System
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
 High-level        World Model       VLA
 Semantic Policy   BAGEL-based       ~5B
        │           visual generator  │
        ↓              │              ↓
    subtask            ↓          action chunk
        │          subgoal image       │
        └──────────────┬──────────────┘
                       ↓
                  Robot Action
```

这里有一个非常重要的边界需要钉死：**π₀.7 系统中的 world model 做的是 visual subgoal generation（"未来应该看到什么"），而不是 action-conditioned dynamics prediction（"执行某个动作后会发生什么"）。** 它的信息流是：

```
current observation + subtask / context
       ↓
visual generative model
       ↓
candidate visual subgoal
       ↓
VLA conditioning → action
```

而不是：

```
current state + candidate action
       ↓
predicted future
       ↓
evaluate → select action
```

这两种 world model 的功能完全不同。本文讨论 VLA 和 world model 的关系，π₀.7 的"VLA 有一个 world model"与"VLA 本身是一个 action-conditioned world model"必须成为全文最清楚的一条边界。

因此，π₀.7 更准确的定位不是"VLA 已经变成 world model"，而是：**VLA 开始把预测模型产生的未来目标作为 policy conditioning signal。** 这意味着 policy 与 prediction 的接口开始出现，但两者仍然承担不同职责：world model 负责生成"希望未来看起来怎样"的视觉目标，VLA 负责学习"在当前状态下如何行动才能完成这个目标"。它还不是一个统一的、可查询 action-conditioned dynamics model。

在一个特定的 **π₀.7 (GC) zero-shot cross-embodiment T-shirt folding evaluation** 中——使用 generated subgoal / visual goal conditioning 的配置，在此前没有见过的双臂 UR5e folding setting 上（没有使用 UR5e folding data 训练）——**π₀.7 (GC)** 达到 85.6% task progress 和 80% success rate；10 名经验丰富的遥操作员在同样的陌生 UR5e 双臂平台上分别达到 90.9% 和 80.6%。更有意思的是，这并不是简单复制源机器人轨迹，而是出现了针对目标机器人运动学重新组织行为策略的现象。

### Visual Subgoal ≠ World Model

π₀.7 引入视觉子目标后，VLA 与 world model 的边界开始变得模糊。但需要注意，**"使用未来视觉子目标"并不自动等价于"拥有一个显式世界模型"**。如果讨论的是用于机器人规划的 world model，那么关键能力通常是 action-conditioned prediction：给定当前状态和候选动作，预测未来状态或 latent state。π₀.7 更准确地说是在 policy 中引入了未来视觉目标作为 conditioning signal。

这一区分很重要：一个模型"看到未来目标"与一个模型"能够预测执行动作后世界会如何变化"，是两个不同能力。


---

**下一篇（下篇）** 进入全系列最概念化的部分：VLA 与世界模型的关系辨析、开放问题、以及三个判断。

> **VLA 系列：** [上篇：RT-2 到 OpenVLA](/zh/articles/2026-09-03-vla-deep-dive/) · [中篇：π₀ 家族](/zh/articles/2026-09-05-vla-pi-family/) · [下篇：VLA 与世界模型](/zh/articles/2026-09-07-vla-world-models/)
