---
title: "VLA 深度解读：从 RT-2 到 OpenVLA 到 π₀，端到端策略如何连接语言与动作"
slug: "2026-09-03-vla-deep-dive"
date: 2026-09-03
draft: false
categories: ["具身智能", "论文解读"]
tags: ["VLA", "RT-2", "OpenVLA", "π₀", "Vision-Language-Action", "机器人基础模型", "端到端策略", "Flow Matching", "具身智能", "Physical Intelligence"]
description: "VLA（Vision-Language-Action）模型正在重塑机器人学习的技术路径——从 RT-2 把互联网知识注入机器人控制，到 OpenVLA 用 7B 参数在 29 个任务上超过 55B 闭源模型，再到 π₀ 用 flow matching 实现连续动作生成与高频控制。这篇文章逐篇拆解 VLA 的技术演进，区分动作表示、动作生成、时序抽象、上下文/embodiment 条件化、预测建模和数据异质性六条轴线，提出 Training Interface 与 Execution Interface 解耦的 meta-axis，并区分被动预测、action-conditioned 和 subgoal generator 三类世界模型。"
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

这篇文章把 VLA 从 RT-2 到 OpenVLA 到 π₀ 做一次完整的技术拆解。但我想强调的不只是一条时间线——**VLA 的演进实际上同时在六条轴上展开：动作表示方式、动作生成机制、时序抽象层级、上下文/embodiment 条件化、预测/规划能力、以及数据异质性。** 把这六个维度分开看，才能理解每个模型真正解决了什么问题。

*注：本文技术时间线以 π₀.7（2026 年 4 月）为主线。VLA 领域发展迅速，后续模型变体不在此展开。*

## 一、VLA 的核心思想

**VLA（Vision-Language-Action）是一类将视觉观察、语言/任务条件与机器人动作策略统一建模的 foundation-model policy。** 它强调的是感知、语言接地与动作生成之间的统一表示和联合学习，而**不要求实现上必须是单一神经网络**。具体实现可以仍然包含多个专用模块——例如 action expert、history encoder 或 hierarchical action head——但它们共享同一个表征基础。

传统机器人控制是分阶段的：感知模块做目标检测和分割，规划模块做任务分解和路径规划，控制模块执行 PID 或阻抗控制。每个模块单独设计，模块之间靠手工定义的接口连接。

VLA 的做法是把这些阶段折叠进一个端到端的学习框架。输入是相机图像和自然语言指令，输出是机器人的动作——末端执行器的位姿增量、关节角度、夹爪开合。没有显式的感知-规划-控制分离，没有手工设计的中间表示。

> **关于"端到端"的说明：** 本文所说的"端到端"，指任务条件到机器人动作策略由统一训练体系直接连接，而不是指模型内部不存在模块化结构或中间表示。例如 π₀ 包含 VLM backbone → action expert → flow matching → action 的多级结构，π₀.5 甚至有 semantic subtask → action generation 的层次化推理链——它们仍然是端到端学习的 policy，但不是"从像素一层不分地直接输出 motor command"。

这个思路的关键转折发生在 2023 年。

## 二、技术演进路线图

在拆解具体模型之前，先看两条平行路线和六个演进维度。VLA / policy 路线和 predictive / world model 路线**不是同一个坐标系里的不同点**，而是从不同方向靠近同一个目标。

```
主线 A：VLA / Policy 路线
───────────────────────────────────────────────→
RT-2 → OpenVLA → OpenVLA-OFT → π₀ → π₀.5 → π₀.7

支线 B：Predictive / World Model 路线
───────────────────────────────────────────────→
V-JEPA → V-JEPA 2 → V-JEPA 2-AC → ???
                   └──→ (action-conditioned extension)
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
| RT-2 | 动作能不能成为 token？ |
| OpenVLA | 能不能把多机器人数据放进一个开放 VLA？ |
| OpenVLA-OFT | 是不是 action decoder 才是瓶颈？ |
| π₀ | 连续生成能不能成为通用 action interface？ |
| π₀.5 | 能不能把语义层和控制层分开？ |
| π₀.7 | 能不能告诉 policy 不只是 WHAT，而是 HOW / UNDER WHAT CONTEXT？ |

这张表揭示了一条"接口演进"的叙事线——从 RT-2 到 π₀.7，每一步都在扩展机器人基础模型的接口边界。

### 六条演进维度

之前的分析用了四条轴。但仔细审视后会发现，"动作表示"和"动作生成"其实是两个独立维度，而"embodiment interface"贯穿始终却未被单独标出。升级为六条轴：

| 演进轴 | 核心问题 |
|--------|---------|
| **Action Representation** | 动作怎么表示？（discrete / continuous） |
| **Action Generation** | 动作怎么生成？（AR / parallel regression / flow matching） |
| **Temporal Abstraction** | 一次决定多长时间？（single action / chunk / semantic subtask） |
| **Context / Embodiment Conditioning** | 模型如何知道"做什么、怎么做、对谁做"？ |
| **Predictive Modeling** | 模型能否预测动作后果？ |
| **Data Heterogeneity** | 能否利用不同质量、机器人、任务的数据？ |

```
              Action Representation        Action Generation
              discrete → continuous        AR → parallel → flow matching
                       │                         │
                       └────────┬────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
     Temporal Abstraction   Context / Embodiment   Predictive Modeling
     single → chunk →       language → language    implicit → visual
     semantic subtask       + proprio + subtask    subgoal generator
                            + metadata + control   → external world
                            mode + subgoal         model component
                                   │
                            Data Heterogeneity
                            single robot → multi-robot
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

这个发现的意义在于，它是早期、具有代表性的系统性展示之一，证明互联网规模的视觉语言预训练知识可以迁移到机器人控制。其核心贡献可以用一条链路概括：

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

**物理技能。** 看到杯子→如何伸手→如何抓稳→如何控制力度→如何避免碰撞——这一整套从感知到力控的运动能力，**互联网数据可以提供一定的物理交互先验，但通常不能直接提供与目标机器人本体、动作空间和控制接口对齐的高频动作监督。**

这正好解释了 RT-2 的一个核心现象：语义泛化提升巨大，但它不会凭空产生新的物理技能。互联网预训练给了 VLA 一个强大的"语义引擎"，但精确的机器人操作技能仍然主要依赖机器人交互数据或其他与 embodiment 对齐的控制数据。

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
- 每个视觉编码器在 224×224 输入、14×14 patch 设置下产生 256 个 spatial patch tokens
- 两个编码器的特征在通道维度拼接，得到 2176 维表示

**投影层：** 3 层 MLP（GELU 激活）把 2176 维视觉特征映射到 LLM 的 4096 维嵌入空间。OpenVLA 的 VLA 训练并非简单冻结视觉端，而是对视觉表示进行针对机器人数据的适配——这是论文中比较反直觉的设计选择之一。

**语言骨干：** Llama-2 7B（32 层 Transformer decoder），视觉 patch token、语言 instruction token 和 action token 共同组成输入序列。

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

**在部分依赖 web-scale semantic knowledge 的 zero-shot generalization evaluation 上，RT-2-X 仍保有优势。** 这一结论来自 OpenVLA 论文所报告的相关 zero-shot evaluation，而非 29-task 主 benchmark。RT-2 直接继承 PaLI-X / PaLM-E 等经过大规模互联网多模态预训练的 backbone，而 OpenVLA 的训练重点是机器人数据，这与两者预训练数据和 backbone 的来源差异有关。

### 局限

- **自回归解码瓶颈**：逐 token 生成动作，推理速度约 4.2 Hz
- **原始模型的输入和训练 recipe 主要围绕单帧视觉观察展开**，对历史视觉信息和复杂多视角场景的建模能力有限
- **真实机器人部署仍需要针对 embodiment / task 做适配**，原始模型的泛化能力不能直接等同于生产级可靠性
- **离散化的潜在精度限制**：固定的 256-bin 表示会引入量化误差，对需要高精度连续控制的任务可能成为限制

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
| policy/action-generation throughput | 4.2 Hz | **109.7 Hz** |
| LIBERO 成功率 | 76.5% | **97.1%** |
| 速度提升 | — | **26 倍** |

需要注意，这些频率数字并不是同一个指标。下表列出了各模型报告数字的具体含义：

| 模型 | 报告数字 | 指标含义 |
|------|----------|----------|
| RT-2 | 1–3 Hz | policy inference / control frequency |
| OpenVLA | ~4.2 Hz | autoregressive action generation |
| OpenVLA-OFT | 109.7 Hz | benchmark configuration 下的 action-generation throughput |
| π₀ | up to 50 Hz | reported robot control frequency |

π₀ 原论文明确说其 action chunk 为 H=50，并使用 10 次 Euler integration；论文把最高 50 Hz 描述为机器人控制频率。不应将 109.7 Hz 与 50 Hz 直接对比——两者的测量口径不同。

OFT 的结果表明，机器人 VLA 的 **action interface、decoding strategy 和 temporal chunking 本身就是重要的系统设计轴**，而不仅仅是 backbone scaling 的附属问题。主要通过重构 action interface——从逐 token 自回归改为并行解码 + action chunking + continuous action head——OFT 获得了大幅速度提升和成功率提升。

这和我们后面会看到的"离散负责统一，连续负责控制"这条主线高度吻合。

这里有一个容易混淆的地方需要澄清：**"连续动作"与"flow matching"不是同一个概念。** "离散 token → 连续 action"是一个维度上的演进（动作表示方式），而"regression / diffusion / flow matching"则是另一个维度上的选择（连续动作的具体生成机制）。OpenVLA-OFT 走的是 continuous regression，π₀ 走的是 flow matching——两者都属于"连续动作"，但生成机制不同。

## 六、π₀：Flow Matching + Action Chunking

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

## 七、π₀.5 与 π₀.7：从动作生成到策略引导

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

## 八、VLA 与世界模型：Policy Learning vs Predictive Modeling

这是我认为最值得深入讨论的部分。前面技术演进中已经多次触及这个问题——从 RT-2 的 model-free policy，到 π₀ 的 action chunk 不等于 planning，再到 π₀.7 的 visual subgoal 不等于 world model——现在把这条线索集中展开。

### VLA 缺的不是"预测能力"，而是显式、可查询的 action-conditioned prediction interface

一个常见的简化是：VLA 只能做动作，世界模型才能做预测。但这个说法不够精确。

VLA 当然可以做预测——一个足够大的自回归模型完全可以预测下一帧图像。真正的区别不在于"有没有预测能力"，而在于：**预测是不是模型的显式、可查询、action-conditioned interface？**

具体来说：

- **VLA 学的是 action distribution**：π(a_t | o_{≤t}, l)——只需要回答"现在该做什么动作"
- **世界模型学的是 future distribution**：一个典型 action-conditioned world model 可以表示为 p_θ(z_{t+1:t+H} | z_t, a_{t:t+H-1})，也可以是确定性的 latent transition function f_θ(z_t, a_t) → z_{t+1}——回答"如果执行这些动作，未来会变成什么样"

有了后者，才能自然地形成一种典型的 planning 形式：

```
候选动作 a⁽¹⁾ → 预测未来 ô⁽¹⁾ → 评估 J(a⁽¹⁾)
候选动作 a⁽²⁾ → 预测未来 ô⁽²⁾ → 评估 J(a⁽²⁾)
...
选择 J 最大的动作序列
```

**这是一种典型的 planning 形式：利用预测模型评估候选动作或轨迹的后果，再进行选择或优化。** Planning 还可以通过 trajectory optimization、MPC、gradient-based optimization、tree search、latent-space optimization、goal-conditioned planning 等多种方式实现，不一定必须显式生成离散的多个候选轨迹。**典型的 imitation-learning VLA 并没有把 action-conditioned future prediction 作为显式、可查询的模型接口。**

需要注意：典型的 imitation-learning VLA 并不显式学习可查询的 action-conditioned dynamics model——但 policy 本身可以隐式编码动态先验。这和"完全没有关于物理世界的内部表征"是两回事。

### 更准确的区分框架

| 维度 | VLA / Policy | Action-conditioned World Model |
|------|-------------|-------------------------------|
| **核心问题** | 现在应该做什么？ | 做了之后会发生什么？ |
| **学习目标** | π(a \| o, l) | p(z_future \| z, a) 或 f(z, a) → z' |
| **数据关系** | observation → action | observation + action → future |
| **输出** | 动作指令 | 预测的未来状态/latent |
| **典型用途** | execution | prediction / planning |
| **主要风险** | policy error / distribution shift | model bias / compounding prediction error |
| **是否需要搜索** | 不需要 | 可与搜索/MPC/optimization 结合 |
| **典型角色** | 偏 execution | 偏 prediction / planning |

简单来说：VLA 回答"我要做什么动作"；世界模型回答"执行这个动作后世界会变成什么样"。典型角色上，VLA 更偏 execution，world model 更偏 prediction/planning——但 VLA 也可以做 implicit planning、hierarchical policy、chain-of-thought，world model 也可以直接支持 policy learning。

### 被动世界模型 vs Action-conditioned 世界模型 vs Subgoal Generator

这里有一个容易混淆的概念需要区分。**本文所说的"用于机器人 planning 的 world model"，重点指具有可查询 action-conditioned prediction interface 的预测模型。** 广义的 world model 可以包括被动视频预测、latent dynamics、reward prediction、object-centric models、generative simulators、goal-conditioned models 等，但这里聚焦于与 planning 直接相关的类型。

**被动世界模型（passive world model）** 可以只用视频学习"世界如何变化"——从 o_t 预测 o_{t+1}，不需要动作标签。

**Action-conditioned 世界模型** 则需要 (o_t, a_t, o_{t+1}) 三元组数据，学习的是"**采取不同动作会导致什么结果**"。这里的 action 不一定非得是机器人低层 motor command——可以是 end-effector action、semantic action、甚至 high-level skill。"action-conditioned"真正要求的是：模型知道导致状态转移的 intervention / control variable 是什么。

**Subgoal generator（如 π₀.7 的 BAGEL-based world model）** 则是第三类：它不预测"执行某个动作后会发生什么"，而是根据任务条件和上下文生成"未来应该看到什么"的候选视觉子目标。

| 模型类型 | 条件 | 输出 | 是否直接回答"动作后果" |
|---------|------|------|------|
| Passive predictive | 当前状态 | future state | 否 |
| Action-conditioned WM | 当前状态 + action | future state | 是 |
| Subgoal generator | 当前状态 + task/context | visual goal | 否 |
| Policy | 当前状态 + task | action | — |

这张表可以作为理解 VLA 与 world model 关系的一个核心参照。

所以真正需要 action-labeled interaction data 的，是**用于 action-conditioned planning 的 world model**。这也正好解释了为什么 V-JEPA 2（被动视频预测）和 V-JEPA 2-AC（action-conditioned）需要在技术栈上分开——JEPA 本身是一类 predictive representation learning 方法；V-JEPA 2-AC 则是 V-JEPA 2 系列上的 **action-conditioned extension**，而不是简单意义上的下一代独立模型——它进一步把 action conditioning 引入预测过程，使其能够承担 action-conditioned world modeling 的角色。

### 统一模型的技术框架

真正的统一模型其实需要同时回答两个问题。一个概念化的联合建模形式可以写成：

p(a_{t:t+H}, z_{t+1:t+H} | z_t, l, g)

也就是同时学到：
- **我要做什么？**（policy）
- **这么做以后会发生什么？**（prediction）

这比单纯说"VLA + world model"更精确——它定义了一个同时具备 action distribution 和 future distribution 的联合模型。**联合建模 policy 与 future state 是 planning 的基础，但真正的 planning 仍需要目标函数、搜索、优化、MPC 或其他 action selection mechanism。** 联合分布本身不等于 planning。

### 两条路线正在靠近

需要纠正一个常见误解：世界模型路线并不是"没有语言"或"不能做 action"。V-JEPA 2 系列已经展示了从 web-scale 视频预训练到 action-conditioned latent prediction、再到机器人规划/控制的技术链条，包括 zero-shot robot deployment 和 image-goal planning。世界模型本身也可以通过语言对齐获得语义能力。

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

**数据瓶颈——从"小时数"到"有效数据"。** 更有意义的问题可能不是"如何获得百万小时机器人数据"，而是：**机器人数据是否真的应该继续按"小时"计量？** 1 小时人类连续成功折 300 件衣服，和 1 小时机器人遇到 50 次失败、20 次恢复、10 种不同策略、5 种 embodiment——信息量完全不同。未来数据 scaling 的价值函数可能更像：

Data Value = f(diversity, failure, recovery, embodiment, task coverage)

而不只是 Data Value ∝ hours。这正好连接到 π₀.7 对 heterogeneous / suboptimal data 的探索。许多主流 VLA 数据集以成功 demonstration 为主，失败 / recovery 数据相对稀缺。**如何从"成功 demo 数据集"走向包含失败、恢复和策略变化的 experience dataset？** 是更关键的问题。π₀ 本身已经给出了一个直接的证据：只用高质量数据会让策略更 fluent，但容易缺少错误恢复能力；多样、较低质量的预训练数据则提供了 recovery / correction repertoire。下一阶段不是单纯 scale hours，而是 scale useful experience。

**长时序任务——错误传播而非步数。** 长时序真正的问题不是 H > 5 或 H > 50，而是错误累积的概率效应。在一个独立同分布、无恢复机制的极简近似下：

P(success over T) ≈ ∏_{t=1}^{T} p_t

如果每步成功率 p_t = p = 0.98，则 0.98^100 ≈ 13%。这当然不是机器人任务成功率的真实统计模型——真实任务中 decisions 不是独立的，recovery 可以改变后续概率，有些错误可恢复，有些关键动作比其他动作重要得多——但它很好地说明了 error accumulation 的指数性直觉：**long-horizon difficulty 的本质是 error accumulation，而不是简单的 sequence length。** 这也解释了为什么 hierarchical policy、recovery policy、replanning、world model 和 memory 都是应对长时序问题的自然方向。

**安全——三个层次。** VLA 的安全问题可以分成三个层次：

*Policy safety*：π(a|o) 会不会输出危险动作？

*Predictive safety*：p(o_future|o,a)——这个动作执行后会不会造成危险？

*Runtime safety*：即使模型错了，有没有独立 safety layer 拦截？

一种可能的 runtime safety architecture 是：

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

safety constraint 不能只作为语言层面的 alignment 问题——它是工程硬约束。

**缺失的模态。** 视觉和语言仍然是主流 VLA 的核心条件模态，proprioception 已经成为不少系统的标准输入（π₀ 和 π₀.7 均使用），而触觉、力/扭矩、听觉等模态在当前大规模 VLA 预训练体系中的覆盖仍明显不足。但对于精细操作（拧螺丝、插钥匙、折叠柔软物体），这些模态可能是关键信息源。

**VLA 是否需要世界模型？** 这个问题我觉得还没有确定的答案。π₀.7 引入视觉子目标作为 conditioning signal 确实提升了泛化能力，但这和"拥有一个显式世界模型"是两回事。真正的融合可能需要一个模型同时做到：语言接地、action-conditioned prediction、高频连续控制。**就本文讨论的公开代表性工作而言，还没有一个系统在大规模真实机器人任务上，以成熟且统一的方式同时解决语言接地、action-conditioned prediction 和高频连续控制。**

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

最后把全文的论证收束为三个判断。前半部分是从已有论文归纳出的技术趋势，后半部分是基于这些趋势提出的未来架构判断。

**判断一：VLA 的进步不能用参数规模单轴解释，动作接口正在成为与 backbone scaling 并列的重要设计轴。** RT-2 → OpenVLA → OFT → π₀ 证明：backbone scaling + data scaling + action interface 共同决定性能。从 RT-2 的 55B 到 OpenVLA 的 7B 到 π₀ 的 3.3B，参数量在缩小；但从 256-bin 离散 token 到 parallel continuous regression 到 flow matching + 50-step action chunk，动作接口在不断进化。OFT 的结果表明，action interface、decoding strategy 和 temporal chunking 本身就是重要的系统设计轴，而不仅仅是 backbone scaling 的附属问题。

**判断二：通用机器人能力的关键瓶颈正在从 representation scaling 转向 effective data scaling、temporal abstraction 和 recovery。** π₀.5 的 97.6% 非目标域数据、π₀.7 对 suboptimal data 的利用、以及长时序任务中错误累积的结构性困难，这些工作共同提示——模型规模之外，effective data scaling、temporal abstraction 和 recovery 正逐渐成为独立的性能决定因素。

**判断三：真正的下一阶段可能不是"VLA 还是 World Model"，而是 policy、predictor 和 planner 的统一。** 未来机器人基础模型的技术地图可以画成：

```
                 Robot Foundation Models
                           │
          ┌────────────────┼────────────────┐
          │                │                │
       Backbone       Action Interface   Temporal Structure
          │                │                │
      VLM / VLA       discrete token      action
          ↓                ↓              chunk
      semantic        continuous            ↓
      grounding       regression       semantic subtask
          │                ↓                │
          │           flow matching         │
          │                │                │
          └────────────────┼────────────────┘
                           ↓
                    Generalist Policy
                           │
                ┌──────────┴──────────┐
                ↓                     ↓
             Action              Prediction
                │                     │
                │              future state /
                │              visual subgoal
                │                     │
                └──────────┬──────────┘
                           ↓
                   Planning / Recovery
                           │
                           ↓
                    Physical Robot
```

如果把这条技术主线压缩：

```
RT-2
│
├─ web knowledge → action token
│
↓
OpenVLA
│
├─ open multi-robot scaling
│
↓
OpenVLA-OFT
│
├─ action interface
│
↓
π₀
│
├─ continuous generative action
│
↓
π₀.5
│
├─ heterogeneous data
├─ semantic hierarchy
│
↓
π₀.7
│
├─ context-rich steering
├─ subgoal conditioning
├─ heterogeneous / suboptimal experience
│
↓
???
├─ policy
├─ prediction
└─ planning
```

如果要用一个方向性描述：

> Robot Foundation Model 的可能形态 = 在统一表征基础上，组合 Perception + Language + Policy + Prediction + Planning 能力

但必须马上补两句：**今天的公开系统通常只覆盖其中的一部分，π₀.5/π₀.7 等工作更像是在逐步扩大这个闭环，而不是已经完成统一。** 并且，这并不意味着每个 robot foundation model 都必须同时包含所有这些能力——更可能的未来形态，是一个能够在统一表征基础上灵活组合这几个能力的系统。

正如我在[世界模型盘点](/zh/articles/2026-09-01-world-model-h2-review/)里说的，"world model"正在失去单一含义。VLA 的加入让这个图景更复杂，也更有趣。

*下一篇，我打算深入聊 Sim-to-Real——从仿真到真实机器人的部署鸿沟到底有多宽，以及当前最好的迁移方法是什么。*
