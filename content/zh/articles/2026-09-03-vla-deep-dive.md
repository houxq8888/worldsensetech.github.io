---
title: "VLA 深度解读（上）：从 RT-2 到 OpenVLA，端到端策略的基础与早期演进"
slug: "2026-09-03-vla-deep-dive"
date: 2026-09-03
draft: false
categories: ["具身智能", "论文解读"]
tags: ["VLA", "RT-2", "OpenVLA", "Vision-Language-Action", "机器人基础模型", "端到端策略", "具身智能"]
description: "VLA 系列三篇的上篇。从 RT-2 把互联网知识注入机器人控制，到 OpenVLA 用 7B 参数在 29 个任务上超过 55B 闭源模型，再到 OFT 揭示 action interface 才是瓶颈——逐篇拆解 VLA 的基础架构与早期演进，区分动作表示、动作生成、时序抽象、上下文/embodiment 条件化、预测建模和数据异质性六条轴线。"
toc: true
related_articles:
  - 2026-09-05-vla-pi-family
  - 2026-09-07-vla-world-models
  - 2026-09-02-jepa-deep-dive
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
---

> **VLA 系列共三篇：** 上篇（本文）· [中篇：π₀ 家族与动作接口演进](/zh/articles/2026-09-05-vla-pi-family/) · [下篇：VLA 与世界模型](/zh/articles/2026-09-07-vla-world-models/)

在[上一篇 JEPA 深度解读](/zh/articles/2026-09-02-jepa-deep-dive/)的结尾，我提到了一个开放问题：JEPA 路线目前主要在视觉和动作空间工作，怎么和语言能力结合？

这个问题指向了具身智能的另一条核心路线——VLA（Vision-Language-Action）。

如果说 JEPA 的核心问题是"世界模型应该预测什么"，那么 VLA 的核心问题就是"机器人应该怎么把看到的东西和听到的指令变成动作"。两条路线的出发点不同，但最终都要回答同一个底层问题：**感知信息如何转化为物理行动？**

这篇文章是三篇系列的**上篇**，把 VLA 从 RT-2 到 OpenVLA 到 OFT 做一次完整的技术拆解，覆盖 VLA 的核心思想、六条演进轴线、以及早期三个代表性模型。但我想强调的不只是一条时间线——**VLA 的演进实际上同时在六条轴上展开：动作表示方式、动作生成机制、时序抽象层级、上下文/embodiment 条件化、预测/规划能力、以及数据异质性。** 把这六个维度分开看，才能理解每个模型真正解决了什么问题。

*注：本系列技术时间线以 π₀.7（2026 年 4 月）为主线。VLA 领域发展迅速，后续模型变体不在此展开。*

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

---

**下一篇（中篇）** 进入 VLA 技术密度最高的部分：π₀ 的 flow matching 架构、π₀.5 的离散-连续混合 recipe、π₀.7 的 context-rich policy steering，以及贯穿演进的 Training Interface vs Execution Interface 解耦这条 meta-axis。

> **VLA 系列：** 上篇（本文）· [中篇：π₀ 家族与动作接口演进](/zh/articles/2026-09-05-vla-pi-family/) · [下篇：VLA 与世界模型](/zh/articles/2026-09-07-vla-world-models/)

