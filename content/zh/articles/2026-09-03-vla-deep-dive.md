---
title: "VLA 深度解读：从 RT-2 到 OpenVLA 到 π₀，端到端策略如何连接语言与动作"
slug: "2026-09-03-vla-deep-dive"
date: 2026-09-03
draft: false
categories: ["具身智能", "论文解读"]
tags: ["VLA", "RT-2", "OpenVLA", "π₀", "Vision-Language-Action", "机器人基础模型", "端到端策略", "Flow Matching", "具身智能", "Physical Intelligence"]
description: "VLA（Vision-Language-Action）模型正在重塑机器人学习的技术路径——从 RT-2 把互联网知识注入机器人控制，到 OpenVLA 用 7B 参数在 29 个任务上超过 55B 闭源模型，再到 π₀ 用 flow matching 实现连续动作生成与高频控制。这篇文章逐篇拆解 VLA 的技术演进，区分动作表示、时序抽象、预测能力和数据异质性四条轴线，并讨论 VLA 与世界模型之间的关系。本文时间线以 π₀.7 为主线，后续模型变体不展开。"
toc: true
related_articles:
  - 2026-09-02-jepa-deep-dive
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - world-model-intro
  - 2026-08-25-dreamer-explained
---

在[上一篇 JEPA 深度解读](/zh/articles/2026-09-02-jepa-deep-dive/)的结尾，我提到了一个开放问题：JEPA 路线目前主要在视觉和动作空间工作，怎么和语言能力结合？

这个问题指向了具身智能的另一条核心路线——VLA（Vision-Language-Action）。

如果说 JEPA 的核心问题是"世界模型应该预测什么"，那么 VLA 的核心问题就是"机器人应该怎么把看到的东西和听到的指令变成动作"。两条路线的出发点不同，但最终都要回答同一个底层问题：**感知信息如何转化为物理行动？**

这篇文章把 VLA 从 RT-2 到 OpenVLA 到 π₀ 做一次完整的技术拆解。但我想强调的不只是一条时间线——**VLA 的演进实际上同时在四个维度上展开：动作表示方式、时序抽象层级、预测/规划能力、以及数据异质性。** 把这四个维度分开看，才能理解每个模型真正解决了什么问题。

*注：本文技术时间线以 π₀.7（2026 年 4 月）为主线。VLA 领域发展迅速，后续模型变体不在此展开。*

## 一、VLA 的核心思想

**VLA（Vision-Language-Action）是一类将视觉观察、语言/任务条件与机器人动作策略统一建模的 foundation-model policy。** 它强调的是感知、语言接地与动作生成之间的统一表示和联合学习，而**不要求实现上必须是单一神经网络**。具体实现可以仍然包含多个专用模块——例如 action expert、history encoder 或 hierarchical action head——但它们共享同一个表征基础。

传统机器人控制是分阶段的：感知模块做目标检测和分割，规划模块做任务分解和路径规划，控制模块执行 PID 或阻抗控制。每个模块单独设计，模块之间靠手工定义的接口连接。

VLA 的做法是把这些阶段折叠进一个端到端的学习框架。输入是相机图像和自然语言指令，输出是机器人的动作——末端执行器的位姿增量、关节角度、夹爪开合。没有显式的感知-规划-控制分离，没有手工设计的中间表示。

> **关于"端到端"的说明：** 本文所说的"端到端"，指任务条件到机器人动作策略由统一训练体系直接连接，而不是指模型内部不存在模块化结构或中间表示。例如 π₀ 包含 VLM backbone → action expert → flow matching → action 的多级结构，π₀.5 甚至有 semantic subtask → action generation 的层次化推理链——它们仍然是端到端学习的 policy，但不是"从像素一层不分地直接输出 motor command"。

这个思路的关键转折发生在 2023 年。

## 二、技术演进路线图

在拆解具体模型之前，先看两条平行路线和四个演进维度。VLA / policy 路线和 predictive / world model 路线**不是同一个坐标系里的不同点**，而是从不同方向靠近同一个目标。

```
主线 A：VLA / Policy 路线
───────────────────────────────────────────────→
RT-2 → OpenVLA → OpenVLA-OFT → π₀ → π₀.5 → π₀.7

支线 B：Predictive / World Model 路线
───────────────────────────────────────────────→
V-JEPA → V-JEPA 2 → V-JEPA 2-AC → ???
                                          ↘
                                           ↘
                                      未来统一模型？
                ↗
               ↗
    两条路线的交汇点：
    shared representation
         ↕        ↕
     policy    prediction
```

### 每个模型真正回答的核心问题

与其只看参数量和数据量，不如看每个模型在技术演进中真正回答的问题：

| 模型 | 主要回答的问题 |
|------|--------------|
| RT-2 | VLM 的语义知识能否迁移到机器人动作？ |
| OpenVLA | 能否在大规模跨机器人数据上训练开放 VLA？ |
| OpenVLA-OFT | 离散 AR action decoder 能否变成高吞吐 policy？ |
| π₀ | 连续动作生成能否扩展到 generalist robot policy？ |
| π₀.5 | 能否利用异质数据和层次化语义处理长时序任务？ |
| π₀.7 | 能否通过策略条件实现 steerable generalist policy？ |

这张表比单纯的参数对比更能揭示技术演进的内在逻辑。

### 四条演进维度

```
                         VLA / Robot Foundation Model
                                      │
          ┌───────────────────────────┼──────────────────────────┐
          │                           │                          │
      Action Representation      Temporal Abstraction       Prediction
          │                           │                          │
   discrete token              action chunk                implicit
          ↓                           ↓                          ↓
   continuous regression       semantic subtask             subgoal
          ↓                           ↓                          ↓
   flow matching               hierarchical policy        world model
          │                           │                          │
          └───────────────────────────┼──────────────────────────┘
                                      │
                              Data Heterogeneity
                                      │
               single robot → multi-robot → web + robot
                         → heterogeneous + suboptimal
                                      │
                                      ↓
                         Generalist Robot Policy
```

在这个框架下，每个模型的定位就清楚了：

- RT-2、OpenVLA、OpenVLA-OFT、π₀ 属于 VLA / policy 路线，逐步解决动作表示和推理效率问题
- π₀.5 和 π₀.7 在 VLA 路线上引入了层次化语义结构和多模态 steering
- V-JEPA、V-JEPA 2、V-JEPA 2-AC 属于 predictive / world model 路线，逐步获得 action-conditioned prediction 和 planning 能力
- 两条路线目前还没有完全汇合，但正在靠近

## 三、RT-2：把互联网知识注入机器人控制

### 论文信息

*RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*，Google DeepMind，2023 年 7 月。arXiv:2307.15818。

### 核心思路

RT-2 的核心发现是：**一个在大规模互联网文本和图像上预训练好的视觉语言模型，可以直接微调成机器人策略——而且微调后的策略能利用预训练阶段学到的语义知识来做零样本泛化。**

这个发现的意义在于，它第一次系统性地展示了"互联网规模的知识迁移到机器人控制"是可行的。其核心贡献可以用一条链路概括：

```
web knowledge → language/vision representation → robot action token
```

### 架构

RT-2 没有从零构建新架构。它直接拿已有的 VLM 做微调：

| 变体 | 骨干网络 | 参数量 |
|------|---------|--------|
| RT-2 PaLI-X | PaLI-X（ViT 视觉编码器） | 55B |
| RT-2 PaLM-E | PaLM-E（多模态嵌入） | 12B |

图像通过骨干网络自带的视觉编码器处理，语言通过标准的 VLM tokenizer 处理。关键创新在动作端。

### 动作离散化：把动作变成"另一种语言"

RT-2 将机器人动作表示为 **7 个连续量**：末端执行器三维位置增量（Δx, Δy, Δz）、三维旋转增量（roll, pitch, yaw）以及夹爪状态。每个维度被均匀量化为 **256 个离散 bin**。终止信号属于动作序列的控制逻辑，而不是这 7 个连续动作维度中的一个。

关键的工程技巧是 **symbol tuning**——把动作离散化为 token，使动作能够和语言共享同一个自回归输出接口。具体实现因 backbone 不同而有所区别：**PaLI-X 直接利用已有 token ID 表示 action bin，而 PaLM-E 则覆盖词汇表中 256 个低频 token，将它们重新定义为动作 token。** 无论哪种方式，效果是一样的：模型架构不变，预训练的语言和视觉知识完整保留，动作只是变成了模型"说的另一种语言"。

推理时，当模型进入动作输出阶段，一个 decoding mask 确保只有这 256 个动作 token 有非零概率。

### 训练

RT-2 在 RT-1 / Everyday Robots 已有的机器人数据以及 VLM web data 的基础上做 co-finetuning，以防止灾难性遗忘。两个 backbone 变体的训练 recipe 并不完全一样——论文中分别报告了不同的数据混合策略。核心思路是一致的：保留一部分互联网图文数据来维持 VLM 的通用语义能力，同时混入机器人操作 demo 来学习动作输出。

### 关键结果

RT-2 的核心数字需要区分不同的 evaluation protocol：

- **在原有任务分布上的表现**：与 RT-1 相近（约 91-93%），说明 co-finetuning 没有牺牲已有能力
- **在新物体、新场景等 generalization evaluation 上**：RT-2 达到约 **62%**，明显高于 RT-1 的约 **32%**——接近两倍提升
- **涌现语义能力**（完全不在机器人训练范围内的能力）：
  - 符号理解 82%（识别并操作抽象符号——数字、形状、logo）
  - 人物识别 53%（"把东西移到 Taylor Swift 那边"）
  - 逻辑推理 46%（"把东西放在 2+1 的和上面"）

涌现能力是 RT-2 最令人兴奋的结果。这些能力不是从机器人训练数据中学到的，而是从 VLM 的互联网预训练知识中迁移过来的。RT-1 在这些任务上基本为零。

### RT-2 真正迁移了什么？

这里值得展开讨论。RT-2 的涌现能力揭示了一个重要的分层结构——VLA 从互联网预训练中迁移的知识，和从机器人数据中学到的能力，本质上是不同层次的东西：

**语义知识。** "杯子"、"红色"、"Taylor Swift"、"里面"、"上面"、"2+1"——这些概念和关系来自互联网文本和图像。VLM 预训练提供了大量此类先验。

**视觉接地。** 看到一个从未见过的新物体，能判断它属于什么语义类别，理解它和语言指令之间的关系。VLM 的视觉编码器在大规模图文对上训练过，提供了强大的视觉泛化能力。

**物理技能。** 看到杯子→如何伸手→如何抓稳→如何控制力度→如何避免碰撞——这一整套从感知到力控的运动能力，**互联网数据基本无法直接提供**。

这正好解释了 RT-2 的一个核心现象：语义泛化提升巨大，但它不会凭空产生新的物理技能。互联网预训练给了 VLA 一个强大的"语义引擎"，但物理操作能力仍然完全依赖机器人 demo 数据。

### 局限

RT-2 的局限也很明确：

- **不能发明新的物理技能**。它能做的是把已有的操作技能应用到新物体和新场景，但不能从互联网知识中学到全新的运动能力。
- **推理速度受限**。55B 模型需要云端 TPU 推理，控制频率只有 1-3 Hz，远低于实际机器人控制的需求。
- **动作精度受量化限制**。256 bin 的离散化对精细操作来说可能不够。

### 与世界模型的关系

RT-2 可以视为典型的 **model-free / direct policy** 方法：它不显式学习一个 action-conditioned dynamics model，而是直接学习 observation + instruction → action 的映射。chain-of-thought 推理是 RT-2 最接近"规划"的东西，但那是语言层面的推理，不是物理模拟。

不过需要注意，这个描述主要适用于 RT-2 这一代 VLA。后续的 VLA 已经开始引入 action chunking、层次化语义动作、子目标条件化等机制，逐步在"直接映射"的基础上增加了更多时序结构。

## 四、OpenVLA：开源 7B VLA

### 论文信息

*OpenVLA: An Open-Source Vision-Language-Action Model*，Stanford / UC Berkeley，2024 年 6 月，CoRL 2024。arXiv:2406.09246。作者 Moo Jin Kim, Karl Pertsch, Chelsea Finn 等。

### 核心思路

RT-2 证明了 VLA 的可行性，但它完全闭源——骨干网络是 Google 内部的 PaLM/PaLI-X，外部研究者无法复现或扩展。OpenVLA 的目标是：**用开源组件构建一个可复现的 VLA，看看开源方案能走多远。**

答案是：比想象中更远。

### 架构

OpenVLA 是一个 7B 参数的模型，基于 Prismatic VLM 架构：

**视觉端——双编码器：**
- SigLIP（1152 维特征）+ DINOv2（1024 维特征），两个视觉编码器提供互补的视觉特征
- 224×224 图像经过 patch size 14 的视觉编码器得到 16×16 = 256 个 patch embeddings
- 两个编码器的特征在通道维度拼接，得到 2176 维表示

**投影层：** 3 层 MLP（GELU 激活）把 2176 维视觉特征映射到 LLM 的 4096 维嵌入空间。OpenVLA 的 VLA 训练并非简单冻结视觉端，而是对视觉表示进行针对机器人数据的适配——这是论文中比较反直觉的设计选择之一。

**语言骨干：** Llama-2 7B（32 层 Transformer decoder），约 280 个 token 的输入序列（BOS + 视觉 patch + 语言指令 + 动作 token）。

**动作输出：** 沿用 RT-2 的思路——**覆盖 Llama tokenizer 中 256 个最低频 token** 作为动作 bin 的 ID（Llama tokenizer 只有约 100 个 reserved special tokens，不够用），每个动作维度自回归生成一个 token。对 WidowX 机器人输出 7-DoF 动作。

### 训练

- **数据**：Open X-Embodiment 数据集，97 万个真实机器人操作 demo
- **算力**：64 块 A100，训练约 15 天
- **微调策略**：在论文所测试的适配设置中，LoRA 仅更新约 1.4% 参数，就能取得与全量微调相当的下游表现

### 关键结果

在论文报告的 **29 个任务、多个 robot embodiment 的平均 task success rate** 上，OpenVLA 比 RT-2-X 高 16.5 个百分点。由于两者的预训练体系、机器人数据和训练 recipe 并不相同，这个结果更适合作为整体系统性能比较，而不是纯粹的参数效率对照：

| 模型 | 参数量 | 对比 |
|------|--------|------|
| **OpenVLA** | **7B** | **+16.5pp over RT-2-X（29-task avg，系统级比较）** |
| RT-2-X | 55B | 基线 |
| OpenVLA vs Diffusion Policy | — | 高出 20.4% |
| OpenVLA vs RT-1-X / Octo | — | 均超过 |

**在高难度语义泛化任务上（需要互联网规模知识才能理解的概念），RT-2-X 仍然更强。** OpenVLA 的 Open X-Embodiment 训练数据不包含互联网规模的图文预训练，所以在"理解从未见过的语义概念"这件事上，它不如 RT-2。

### 局限

- **自回归解码瓶颈**：逐 token 生成动作，推理速度约 4.2 Hz
- **原始模型的输入和训练 recipe 主要围绕单帧视觉观察展开**，对历史视觉信息和复杂多视角场景的建模能力有限
- **真实机器人部署仍需要针对 embodiment / task 做适配**，原始模型的泛化能力不能直接等同于生产级可靠性
- **离散化精度损失**：256 bin 量化对精细操作仍然不够

## 五、OpenVLA-OFT：真正的瓶颈可能在 Action Interface

OpenVLA-OFT（2025 年，arXiv:2502.19645）在全文技术时间线中的地位不应仅仅被视为 OpenVLA 的一个优化版本。从技术演进角度，它回答了一个关键问题：

> **"VLA 的性能瓶颈究竟是 foundation model 本身，还是 action decoding？"**

OpenVLA 的 action decoding 是逐 token 自回归生成离散 token。OFT 对此做了彻底的改造：

- **并行解码**替代自回归：同时生成所有动作 token
- **动作 chunk**：一次前向传播预测 8 步动作
- **连续动作头**：用 MLP + L1 回归替代离散 token——注意，这是 **continuous regression**，不是 flow matching
- **LoRA 微调**

结果：

| 指标 | 原始 OpenVLA | OpenVLA-OFT |
|------|-------------|-------------|
| LIBERO 成功率 | 76.5% | **97.1%** |
| 推理 throughput | 4.2 Hz | **109.7 Hz** |
| 速度提升 | — | **26 倍** |

OFT 的意义在于证明了一个重要结论：**模型 backbone 的 scaling 并不是机器人实时控制性能的唯一瓶颈，action interface 本身同样重要。** 将 action decoding 从逐 token 自回归改为并行解码 + action chunking + continuous action head，仅改变 action interface 就获得了 26 倍速度提升和大幅成功率提升。

这和我们后面会看到的"离散负责统一，连续负责控制"这条主线高度吻合。

这里有一个容易混淆的地方需要澄清：**"连续动作"与"flow matching"不是同一个概念。** "离散 token → 连续 action"是一个维度上的演进（动作表示方式），而"regression / diffusion / flow matching"则是另一个维度上的选择（连续动作的具体生成机制）。OpenVLA-OFT 走的是 continuous regression，π₀ 走的是 flow matching——两者都属于"连续动作"，但生成机制不同。

## 六、π₀：Flow Matching + Action Chunking

### 论文信息

*π₀: A Vision-Language-Action Flow Model for General Robot Control*，Physical Intelligence，2024 年 10 月。arXiv:2410.24164。

### Physical Intelligence 背景

Physical Intelligence（也叫 π）2023 年成立于旧金山，使命是"构建通用机器人大脑"。五位联合创始人包括 Karol Hausman（CEO，前 Google DeepMind，SayCan / RT-2 核心成员）、Chelsea Finn（Stanford，MAML 发明者）、Sergey Levine（UC Berkeley，SAC 共同作者）、Brian Ichter 和 Jasmine Hsu（均出自 Google Brain）。截至 2026 年，Physical Intelligence 已累计融资约 21 亿美元，最近一轮融资对应估值约 110 亿美元。

### 核心架构

π₀ 做了一个和 RT-2 / OpenVLA 不同的选择：**不用离散 token，用 flow matching 生成连续动作。**

架构分为两部分：

| 组件 | 说明 |
|------|------|
| VLM 骨干 | PaliGemma（3B 参数视觉语言模型） |
| 动作专家 | 300M 参数的专用网络，挂在 VLM 后面 |
| **总参数** | **约 3.3B** |

需要注意的是，**action expert 不是一个"外挂 controller"**。它和 VLM backbone 是联合训练、条件耦合的——VLM 提供语言视觉条件，action expert 基于这些条件生成连续动作。更准确的描述是：**一个语言视觉 backbone + 一个连续动作生成专家共享条件信息。**

输入包括图像 token、语言 token 和本体感知（proprioception），经过共享表征后由 action expert 通过 flow matching 输出 action chunk。

### 连续动作生成的三种机制

连续动作生成有三种主要机制，需要区分清楚。它们不是"先后替代"的关系，而是**生成式动作建模（generative action modeling）下的两条平行路线**：

```
              Generative Action Modeling
                        │
                   ┌────┴────┐
              Diffusion   Flow Matching
                   │             │
             score/noise   vector field
                   │             │
             iterative     ODE integration
             denoising     / transport
                   │             │
                   └────┬────────┘
                        │
                  Regression
                        │
                  direct prediction
                  (no sampling)
```

Flow matching 训练的是概率路径上的 vector field / velocity field，使模型能够通过 ODE integration 从简单先验分布 transport 到目标数据分布。它的做法是：

- 定义一条从纯噪声分布到目标动作分布的概率路径（linear-Gaussian probability path）
- 训练一个网络来预测这条路径上的速度场
- 推理时从噪声出发，沿学到的向量场积分，得到连续动作

和 diffusion 的关键区别在于：两者的训练目标和推理形式不同。在 π₀ 的具体实现中，flow matching 配合较少的 Euler integration steps，形成了适合实时 action chunk generation 的连续策略。

### 动作 Chunk、时序抽象与规划：三个容易混淆的概念

π₀ 每次生成一个包含 **50 个未来动作**的 action chunk；在最高 **50 Hz 系统控制频率**下，这相当于约 1 秒的未来轨迹。论文报告其系统在 dexterous tasks 上可以达到最高 50 Hz 的系统控制频率。

需要注意的是，flow matching 推理本身还需要进行多步 integration。50 Hz 是包含 action chunk 执行在内的系统级控制频率，不是单次 flow matching 推理的速度。

**这里需要特别澄清三个容易混淆的概念：**

**Action chunking**——一次输出 a_t, a_{t+1}, ..., a_{t+H-1}。解决的是减少决策频率、提高动作连贯性。π₀ 的 50 步 chunk 属于这个层面。

**Temporal abstraction**——把长任务压缩到更高层的时间尺度。比如"拿起杯子"是一个高层行为，而不是 50 个关节动作。π₀.5 的 semantic subtask 属于这个层面。

**Planning**——需要考虑动作序列会导致什么状态变化，并评估"哪个未来更好"。这才是 world-model planning 的核心。

**Action chunk ≠ planning。** π₀ 生成 50 步动作 chunk 后执行前几步，再重新观测、生成下一个 chunk。这是 **receding-horizon policy execution**，不等于在内部模拟多个候选未来然后比较选择。

### Cross-Embodiment Action Space

π₀ 将不同 embodiment 的动作表示统一到**最多 18 DoF** 的公共空间，不同机器人根据自身动作空间进行 padding / masking：

```
Franka 单臂    → 7 DoF
ALOHA 双臂     → 14 DoF
移动操作平台   → 最多 18 DoF
```

这使得同一个模型可以控制不同形态的机器人——这是 π₀ 的一个重要贡献。

### 训练

两阶段：

**阶段一——广泛预训练：**
- 互联网规模图文数据（继承自 PaliGemma）
- Open X-Embodiment "Magic Soup" 子集
- 自有多机器人数据：约 **10,000 小时**，约 **9 亿个时间步**，覆盖 **68 个任务**、**7 种硬件配置**

**阶段二——定向后训练：**
- 在高质量、精选的任务 demo 上微调，掌握复杂操作技能

### 关键结果

在论文报告的 zero-shot real-world evaluation protocol 下，π₀ 在复杂操作任务上展现了离散 token 方案所不具备的能力：

| 任务类型 | π₀ 论文中的观察 |
|---------|---------------|
| 长序列衣物操作 | 能够完成 zero-shot manipulation |
| 桌面清理 | 报告 97.1% success |
| 对比离散 VLA | 在这些特定 zero-shot protocol 上明显更强 |

需要注意，这些观察高度依赖具体的 task protocol（trial 数量、机器人形态、task definition、"zero-shot"的具体含义等），因此不宜理解成跨模型的普适性能排序。但它们清楚地表明：衣物折叠和桌面清理这类需要长序列、高精度连续操作的任务，正好是离散 token 方案的弱点。

### 如何理解 π₀ 的性能提升

从 RT-2 到 π₀，连续动作生成确实成为越来越重要的方向。但目前还不能把性能提升简单归因于"连续表示优于离散 token"。π₀ 同时引入了 action chunking、flow matching、跨 embodiment 训练数据以及更大规模的机器人数据。更准确的结论是：**连续动作表示解决了量化和自回归输出的一部分结构性问题，但它的最终收益与数据规模、动作 chunking 和训练 recipe 强烈耦合。**

### 局限

- **代码和部分模型 checkpoint 已通过 openpi 开源，但训练数据、完整内部训练体系以及所有生产/实验变体并未全部开放**
- 某些需要精确力控的任务仍不可靠
- 对全新物理域（自动驾驶、飞行器）的泛化能力未知
- VLM 微调可能导致语言/视觉能力退化（catastrophic forgetting）

## 七、π₀.5 与 π₀.7：从动作生成到策略引导

### π₀.5：数据组成 + 层次化策略

π₀.5（2025 年 4 月，arXiv:2504.16054）的核心贡献不是简单地"加入高层推理"，而是引入了一个**层次化架构**，以及一个非常重要的技术事实：**离散和连续动作表示可以在同一个 foundation model 中承担不同角色。**

π₀.5 的训练分为两个阶段，使用了不同的动作表示和 prediction target：

```
预训练阶段（discrete）           后训练阶段（continuous）
┌─────────────────────┐      ┌─────────────────────┐
│ FAST action tokenizer│      │ Flow matching action │
│ 离散 action token    │  →   │ head                 │
│ 语义子任务预测       │      │ 连续高频控制         │
│ 多机器人 + web 数据  │      │ 目标域微调           │
└─────────────────────┘      └─────────────────────┘
         ↓                              ↓
    统一的 backbone 表征           推理时：先推断语义子任务
                                   再基于子任务生成连续动作
```

**π₀.5 不是简单地把一个离散策略替换成连续策略，而是让离散序列建模和连续动作生成在不同层级、不同训练阶段承担不同职责。**

**预训练阶段**使用 **FAST action tokenizer** 将机器人动作离散化，使动作可以和语言、语义子任务等 token 共享 next-token prediction 训练。这使得大规模异构数据（不同机器人、不同任务、甚至 web 数据）可以在统一的 sequence modeling 框架中被利用。高层 **hierarchical task decomposition / semantic subtask prediction** 是这一阶段的重要组成部分。

**后训练阶段**引入 **flow-matching action head** 负责高频连续控制，针对 mobile manipulation 做 post-training。推理时，模型先推断一个高层语义子任务（如 "pick up the pillow"），然后基于这个子任务用 flow matching 生成连续动作 chunk。

一个值得注意的数据是：π₀.5 的第一阶段训练数据中，约 **97.6% 并不是 mobile-manipulator household data**。它通过大量异质数据的混合训练来获得泛化能力，然后在少量目标域数据上微调。

π₀.5 因此说明"离散 vs 连续"不是一个简单的替代关系。它的实际路线不是 "discrete → continuous"，而是 **discrete for scalable multimodal pretraining + continuous for fine-grained control**。

π₀.5 已经能够在训练中未出现的家庭环境执行 10-15 分钟级别的长时序任务，但其成功率仍明显低于受控环境中的短任务。这说明长程任务中的错误累积仍然是 VLA 泛化的主要瓶颈。

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
- 保留空间精度，避免 256-bin 量化损失
- 可建模多峰 action distribution
- 生成连续 action chunk
- 适合高频精细控制

所以最终不是 "discrete → continuous"，而是 **tokenization 和 continuous generation 可能服务于不同阶段/不同层级。** 全文最核心的判断可以压成一句：

> **离散负责统一，连续负责控制。**

### π₀.7：从 task conditioning 到 strategy conditioning

π₀.7（2026 年 4 月，arXiv:2604.15483）进一步探索了 VLA 的"可引导泛化"（steerability）。

从 RT-2 到 π₀.7，policy 的输入发生了质的变化：

```
RT-2：  language → action
π₀：    language + image + proprioception → action chunk
π₀.5：  language + observation → semantic subtask → action chunk
π₀.7：  language + episode metadata + strategy + subgoal image + history → policy → action chunk
```

π₀.7 的真正进步不只是"多了一些输入"，而是：**prompt 从"描述我要做什么"变成了"描述我应该如何做"。** 也就是从 **task conditioning** 逐渐走向 **strategy conditioning / policy steering**。这恰好也是为什么 π₀.7 的标题用了 "Steerable Generalist"。

模型不再只把语言指令作为条件，而是把语言、episode metadata、执行策略信息、视觉子目标以及观测历史等多模态上下文统一作为 policy 的条件输入。

π₀.7 的模型规模约为 5B，由约 4B 的 VLM backbone、视频历史编码模块（MEM-style video history encoder）以及 860M 参数的 action expert 组成。它仍然沿用 flow matching 连续动作生成路线。

值得注意的是，**π₀.7 本身不是一个 action-conditioned world model；不过其推理系统可以使用由外部轻量视觉生成模型产生的 subgoal image 作为未来视觉目标，因此已经出现了 policy 与 predictive/generative model 组合的雏形。**

π₀.7 报告的结果：在未见过的机器人形态上达到 85.6% 任务进度和 80% 成功率，接近人类遥操作员的 90.9% / 80.6%。

### Visual Subgoal ≠ World Model

π₀.7 引入视觉子目标后，VLA 与 world model 的边界开始变得模糊。但需要注意，**"使用未来视觉子目标"并不自动等价于"拥有一个显式世界模型"**。真正的世界模型通常需要学习 action-conditioned transition dynamics，并能够在内部进行未来状态预测或 rollout。π₀.7 更准确地说是在 policy 中引入了未来视觉目标作为 conditioning signal。

这一区分很重要：一个模型"看到未来目标"与一个模型"能够预测执行动作后世界会如何变化"，是两个不同能力。

## 八、VLA 与世界模型：Policy Learning vs Predictive Modeling

这是我认为最值得深入讨论的部分。前面技术演进中已经多次触及这个问题——从 RT-2 的 model-free policy，到 π₀ 的 action chunk 不等于 planning，再到 π₀.7 的 visual subgoal 不等于 world model——现在把这条线索集中展开。

### VLA 缺的不是"预测能力"，而是可查询的 action-conditioned prediction interface

一个常见的简化是：VLA 只能做动作，世界模型才能做预测。但这个说法不够精确。

VLA 当然可以做预测——一个足够大的自回归模型完全可以预测下一帧图像。真正的区别不在于"有没有预测能力"，而在于：**预测是不是模型的显式、可查询、action-conditioned interface？**

具体来说：

- **VLA 学的是 action distribution**：π(a_t | o_{≤t}, l)——只需要回答"现在该做什么动作"
- **世界模型学的是 future distribution**：p(z_{t+1:t+H} | z_t, a_{t:t+H-1})——回答"如果执行这些动作，未来会变成什么样"

有了后者，才能自然地形成：

```
候选动作 a⁽¹⁾ → 预测未来 ô⁽¹⁾ → 评估 J(a⁽¹⁾)
候选动作 a⁽²⁾ → 预测未来 ô⁽²⁾ → 评估 J(a⁽²⁾)
...
选择 J 最大的动作序列
```

这才是 planning 的关键——**在内部生成多个候选未来，比较它们，然后选择**。VLA 的端到端 policy 不具备这个可查询的 interface。

需要注意：典型的 imitation-learning VLA 并不显式学习可查询的 action-conditioned dynamics model——但 policy 本身可以隐式编码动态先验。这和"完全没有关于物理世界的内部表征"是两回事。

### 更准确的区分框架

| 维度 | VLA / Policy | Action-conditioned World Model |
|------|-------------|-------------------------------|
| **核心问题** | 现在应该做什么？ | 做了之后会发生什么？ |
| **学习目标** | π(a \| o, l) | p(o_{t+1:t+H} \| o_t, a_{t:t+H}) |
| **输出** | 动作指令 | 预测的未来状态/表征 |
| **强项** | 直接控制、反应速度 | 预测、比较候选未来 |
| **长处** | execution | planning |
| **弱点** | error accumulation | model bias / compute |
| **是否必须 action label** | policy 需要 | action-conditioned 版本需要 |
| **是否天然需要搜索** | 不需要 | 通常可与搜索/MPC/optimization 结合 |
| **最终角色** | actor | predictor / planner |

简单来说：VLA 回答"我要做什么动作"；世界模型回答"执行这个动作后世界会变成什么样"。

### 被动世界模型 vs Action-conditioned 世界模型

这里还有一个容易混淆的概念需要区分。

**被动世界模型（passive world model）** 可以只用视频学习"世界如何变化"——从 o_t 预测 o_{t+1}，不需要动作标签。

**Action-conditioned 世界模型** 则需要 (o_t, a_t, o_{t+1}) 三元组数据，学习的是"**采取不同动作会导致什么结果**"。

所以真正需要 action-labeled interaction data 的，是**用于 action-conditioned planning 的 world model**。这也正好解释了为什么 V-JEPA 2（被动视频预测）和 V-JEPA 2-AC（action-conditioned）需要在技术栈上分开——JEPA 是一种预测式 representation learning 范式；当其预测过程进一步显式条件于动作，并能够用于预测未来状态时，才构成 action-conditioned world model。

### 统一模型的技术框架

真正的统一模型其实需要同时回答两个问题：

p(a_{t:t+H}, z_{t+1:t+H} | z_t, l, g)

也就是同时学到：
- **我要做什么？**（policy）
- **这么做以后会发生什么？**（prediction）

这比单纯说"VLA + world model"更精确——它定义了一个同时具备 action distribution 和 future distribution 的联合模型。

### 两条路线正在靠近

需要纠正一个常见误解：世界模型路线并不是"没有语言"或"不能做 action"。V-JEPA 2 已经展示了 web-scale video pretraining + action-conditioned world model + V-JEPA 2-AC 的完整技术栈，包括 zero-shot robot deployment 和 image-goal planning。世界模型本身也可以通过语言对齐获得语义能力。

JEPA 路线的规划也不一定是"生成多条轨迹再选择"。它可以是 latent prediction → goal-conditioned planning，可以是 search、optimization 或 policy guidance。

所以更准确的图景是：**VLA 和世界模型正在从两个方向靠近同一个目标——一个同时具备 policy、prediction 和 planning 能力的机器人基础模型。** 未来系统更可能是 Actor + Predictor，而不是二选一。

```
                 Robot Foundation Model
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ↓              ↓              ↓
       Language       Perception      Robot Data
          │              │              │
          └──────────────┼──────────────┘
                         ↓
                 Shared Representation
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       Semantic       Policy        Prediction
       Subtask          │              │
          │             ↓              ↓
          │       Action Chunk    Future State
          │             │              │
          │             ↓              ↓
          │       Flow Matching    World Model
          │             │              │
          └─────────────┼──────────────┘
                        ↓
                  Physical Action
```

**未来的机器人 foundation model 很可能不是一个纯 VLA，也不是一个纯 world model，而是一个同时支持 semantic conditioning、policy execution 和 predictive modeling 的统一系统。**

### 它们互补吗？

我觉得答案是肯定的，但互补的方式比"把两个模块拼在一起"更微妙。

**典型的直接 policy VLA 缺什么？** 它不显式学习可查询的 action-conditioned dynamics model。当遇到训练中没见过的新情况时，它只能依赖预训练阶段学到的泛化能力——policy 可以隐式编码动态先验，但不能像世界模型那样显式模拟行动后果。

**世界模型缺什么？** 虽然世界模型正在获得语言和 action 能力，但在端到端策略学习的效率和语言接地的自然度上，目前仍不如 VLA 路线。

所以一个自然的想法是：**用世界模型做物理预测，用 VLA 做动作执行和语言理解。**

## 九、开放问题

**数据瓶颈——从"小时数"到"数据价值"。** 更有意义的问题可能不是"如何获得百万小时机器人数据"，而是：**机器人数据是否真的应该继续按"小时"计量？** 1 小时人类连续成功折 300 件衣服，和 1 小时机器人遇到 50 次失败、20 次恢复、10 种不同策略、5 种 embodiment——信息量完全不同。未来数据 scaling 的价值函数可能更像：

Data Value = f(diversity, failure, recovery, embodiment, task coverage)

而不只是 Data Value ∝ hours。这正好连接到 π₀.7 对 heterogeneous / suboptimal data 的探索。许多主流 VLA 数据集以成功 demonstration 为主，失败 / recovery 数据相对稀缺。**如何从"成功 demo 数据集"走向包含失败、恢复和策略变化的 experience dataset？** 是更关键的问题。

**长时序任务——错误传播而非步数。** 长时序真正的问题不是 H > 5 或 H > 50，而是错误累积的概率效应：

P(success over T) ≈ ∏ P(correct_t)

哪怕单步成功率 P = 0.98，100 个关键决策之后 0.98^100 ≈ 13%。当然真实机器人任务并不严格服从独立事件模型，但它很好地说明：**long-horizon difficulty 的本质是 error accumulation，而不是简单的 sequence length。** 这也解释了为什么 hierarchical policy、recovery policy、replanning、world model 和 memory 都是应对长时序问题的自然方向。

**安全——三个层次。** VLA 的安全问题可以分成三个层次：

*Policy safety*：π(a|o) 会不会输出危险动作？

*Predictive safety*：p(o_future|o,a)——这个动作执行后会不会造成危险？

*Runtime safety*：即使模型错了，有没有独立 safety layer 拦截？

一个更完整的系统可以是：

```
VLA
 ↓
candidate action
 ↓
world model / safety critic
 ↓
constraint checker
 ↓
robot
```

这比单纯说"VLA 需要安全机制"有技术深度得多。safety constraint 不能只作为语言层面的 alignment 问题——它是工程硬约束。

**缺失的模态。** 当前 VLA 几乎只依赖视觉和语言。触觉、力反馈、听觉在训练数据中严重缺失。但对于精细操作（拧螺丝、插钥匙、折叠柔软物体），这些模态可能是关键信息源。

**VLA 是否需要世界模型？** 这个问题我觉得还没有确定的答案。π₀.7 引入视觉子目标作为 conditioning signal 确实提升了泛化能力，但这和"拥有一个显式世界模型"是两回事。真正的融合可能需要一个模型同时做到：语言接地、action-conditioned prediction、高频连续控制。**目前还没有一个在公开实验中同时以成熟、统一的方式解决这三者，并在大规模真实机器人任务上得到充分验证的方案。**

**离散 vs 连续的最终判断。** 从 RT-2 的离散 token 到 OpenVLA-OFT 的 continuous regression 再到 π₀ 的 flow matching，连续方法在控制精度和推理速度上展现了优势。但 π₀.5 和 π₀-FAST 表明，**离散和连续很可能承担不同的角色：离散负责"统一"——使动作可以和语言、视觉、语义子任务共享 sequence modeling 接口；连续负责"控制"——在最终执行阶段提供高频精细的连续动作输出。**

```
         Foundation-model pretraining
                    ↑
                    │
           discrete tokens
                    │
                    │
              shared LM space
                    │
                    ↓
       continuous action generation
                    │
                    ↓
           high-frequency control
```

这个判断比"continuous 最终取代 discrete"强得多，也更能解释为什么 π₀-FAST、π₀.5、π₀.7 看起来像是在"走回头路"，实际上是在探索不同的模型接口。

---

## 十、三个判断

最后把全文的论证收束为三个判断。

**判断一：VLA 的核心进步不是参数越来越大，而是动作接口越来越适合机器人。** 从 RT-2 的 55B 到 OpenVLA 的 7B 到 π₀ 的 3.3B，参数量在缩小；但从 256-bin 离散 token 到 parallel continuous regression 到 flow matching + 50-step action chunk，动作接口在不断进化。OFT 仅改变 action interface 就获得 26 倍速度提升，说明瓶颈不在 backbone，而在 action interface。

**判断二：通用机器人能力的关键瓶颈正在从 representation scaling 转向 data scaling、temporal abstraction 和 recovery。** π₀.5 的 97.6% 非目标域数据、π₀.7 对 suboptimal data 的利用、以及长时序任务中错误累积的结构性困难，都指向同一个方向——下一步突破的关键不只是模型更大，而是数据更多样、时序结构更鲁棒、失败恢复能力更强。

**判断三：真正的下一阶段可能不是"VLA 还是 World Model"，而是 policy、predictor 和 planner 的统一。** 未来机器人基础模型的技术地图可以画成：

```
                         Robot Foundation Model
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
          Representation      Action Interface     Temporal Structure
              │                   │                   │
          VLM / VLA         discrete token       single action
              ↓                   ↓                   ↓
       cross-modal FM       continuous action     action chunk
                                  ↓                   ↓
                           flow matching       semantic subtask
                                  │                   │
              ┌───────────────────┴───────────────────┘
              │
              ↓
       Generalist Policy
              │
              │        + predictive modeling
              │
              ↓
       Action-conditioned
         World Model
              │
              ↓
        Future prediction
              │
              ↓
        Planning / Safety
```

如果用一句话概括：

> Robot Foundation Model = Perception + Language + Policy + Prediction + Planning

但必须马上补一句：**今天的公开系统通常只覆盖其中的一部分，π₀.5/π₀.7 等工作更像是在逐步扩大这个闭环，而不是已经完成统一。**

正如我在[世界模型盘点](/zh/articles/2026-09-01-world-model-h2-review/)里说的，"world model"正在失去单一含义。VLA 的加入让这个图景更复杂，也更有趣。

*下一篇，我打算深入聊 Sim-to-Real——从仿真到真实机器人的部署鸿沟到底有多宽，以及当前最好的迁移方法是什么。*
