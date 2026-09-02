---
title: "VLA 深度解读：从 RT-2 到 OpenVLA 到 π₀，端到端策略如何连接语言与动作"
slug: "2026-09-03-vla-deep-dive"
date: 2026-09-03
draft: false
categories: ["具身智能", "论文解读"]
tags: ["VLA", "RT-2", "OpenVLA", "π₀", "Vision-Language-Action", "机器人基础模型", "端到端策略", "Flow Matching", "具身智能", "Physical Intelligence"]
description: "VLA（Vision-Language-Action）模型正在重塑机器人学习的技术路径——从 RT-2 把互联网知识注入机器人控制，到 OpenVLA 用 7B 参数在 29 个任务上超过 55B 闭源模型，再到 π₀ 用 flow matching 实现连续动作生成与高频控制。这篇文章逐篇拆解 VLA 的技术演进，区分动作表示、时序抽象、预测能力和数据异质性四条轴线，并讨论 VLA 与世界模型之间的关系。"
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

## 一、VLA 的核心思想

**VLA 的核心不是"所有东西必须只有一个网络"，而是把感知、语言接地和策略学习放进一个共享的 foundation-model 表示体系中。** 具体实现可以仍然包含多个专用模块——例如 action expert、history encoder 或 hierarchical action head——但它们共享同一个表征基础。

传统机器人控制是分阶段的：感知模块做目标检测和分割，规划模块做任务分解和路径规划，控制模块执行 PID 或阻抗控制。每个模块单独设计，模块之间靠手工定义的接口连接。

VLA 的做法是把这些阶段折叠进一个端到端的学习框架。输入是相机图像和自然语言指令，输出是机器人的动作——末端执行器的位姿增量、关节角度、夹爪开合。没有显式的感知-规划-控制分离，没有手工设计的中间表示。

这个思路的关键转折发生在 2023 年。

## 二、技术演进路线图

在拆解具体模型之前，先看两条平行路线。VLA / policy 路线和 predictive / world model 路线**不是同一个坐标系里的不同点**，而是从不同方向靠近同一个目标。

```
VLA / Policy 路线
───────────────────────────────────────────────→
RT-2 → OpenVLA → OpenVLA-OFT → π₀ → π₀.5 → π₀.7


Predictive / World Model 路线
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

这个发现的意义在于，它第一次系统性地展示了"互联网规模的知识迁移到机器人控制"是可行的。

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

在论文报告的 **29 个任务、多个 robot embodiment 的平均 task success rate** 上，OpenVLA 比 RT-2-X 高 16.5 个百分点，同时参数量只有约 1/7：

| 模型 | 参数量 | 对比 |
|------|--------|------|
| **OpenVLA** | **7B** | **比 RT-2-X 高 16.5pp（29-task avg）** |
| RT-2-X | 55B | 基线 |
| OpenVLA vs Diffusion Policy | — | 高出 20.4% |
| OpenVLA vs RT-1-X / Octo | — | 均超过 |

**这里的"超过"指的是 29-task average success，而不是全面超越 RT-2-X。** 有一个重要的例外：**在高难度语义泛化任务上（需要互联网规模知识才能理解的概念），RT-2-X 仍然更强。** OpenVLA 的 Open X-Embodiment 训练数据不包含互联网规模的图文预训练，所以在"理解从未见过的语义概念"这件事上，它不如 RT-2。

### 局限

- **自回归解码瓶颈**：逐 token 生成动作，推理速度约 4.2 Hz
- **单目图像输入**：不支持多视角或立体视觉
- **零样本可靠性不足**：未微调时成功率不到 90%，不够做实际部署
- **离散化精度损失**：256 bin 量化对精细操作仍然不够

### 后续：OpenVLA-OFT

2025 年的 OpenVLA-OFT（arXiv:2502.19645）针对自回归瓶颈做了一次彻底的改造：

- **并行解码**替代自回归：同时生成所有动作 token
- **动作 chunk**：一次前向传播预测 8 步动作
- **连续动作头**：用 MLP + L1 回归替代离散 token——注意，这是 **continuous regression**，不是 flow matching
- **LoRA 微调**

结果：

| 指标 | 原始 OpenVLA | OpenVLA-OFT |
|------|-------------|-------------|
| LIBERO 成功率 | 76.5% | **97.1%** |
| 推理速度 | 4.2 Hz | **109.7 Hz** |
| 速度提升 | — | **26 倍** |

OpenVLA-OFT 很重要的一点是，它证明了**不需要把 action token 化，也不需要采用 flow matching，VLA 同样可以通过并行解码 + action chunking + continuous action head 获得高速控制。**

这里有一个容易混淆的地方需要澄清：**"连续动作"与"flow matching"不是同一个概念。** "离散 token → 连续 action"是一个维度上的演进（动作表示方式），而"regression / diffusion / flow matching"则是另一个维度上的选择（连续动作的具体生成机制）。OpenVLA-OFT 走的是 continuous regression，π₀ 走的是 flow matching——两者都属于"连续动作"，但生成机制不同。

## 五、π₀：Flow Matching + Action Chunking

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

### Flow Matching 做什么

连续动作生成有三种主要机制，需要区分清楚：

```
连续动作生成
├── regression：直接预测 action 值
├── diffusion：学习 score / denoising process
└── flow matching：学习 velocity field / transport path
```

Flow matching 不是传统意义上"逐步去噪的 diffusion policy"的替代说法。它的做法是：

- 定义一条从纯噪声分布到目标动作分布的概率路径（linear-Gaussian probability path）
- 训练一个网络来预测这条路径上的速度场
- 推理时从噪声出发，沿学到的向量场积分，得到连续动作

和 diffusion 的关键区别在于：flow matching 学习的是确定性的 transport path，而不是随机去噪过程。这使得推理更高效，轨迹更平滑。

### 动作 Chunk 与高频控制

π₀ 每次生成一个包含 **50 个未来动作**的 action chunk；在最高 **50 Hz** 的控制频率下，这相当于约 1 秒的未来轨迹。论文报告其系统在 dexterous tasks 上可以达到最高 50 Hz 的控制频率。

需要注意的是，flow matching 推理本身还需要进行多步 integration。50 Hz 是包含 action chunk 执行在内的系统级控制频率，不是单次 flow matching 推理的速度。

**这里需要特别澄清：action chunk ≠ planning。** π₀ 生成 50 步动作 chunk 后执行前几步，再重新观测、生成下一个 chunk。这是 **receding-horizon policy execution**，不等于在内部模拟多个候选未来然后比较选择。Planning 更典型的做法是：生成多个候选动作序列 → 用世界模型 rollout → 比较 → 选择。Action chunk 是时序抽象，不是规划。

### Cross-Embodiment Action Space

π₀ 的动作维度支持到 18 维，这不只是"更多维度"这么简单——它实际上解决了一个重要的工程问题：**不同机器人的动作空间完全不同。**

```
Franka 单臂    → 7 DoF
ALOHA 双臂     → 14 DoF
移动操作平台   → 18 DoF
```

π₀ 需要把不同 embodiment 的 action space 对齐到统一接口。它的做法是用 18 维作为最大公共空间，简单机器人用零填充和 mask。这使得同一个模型可以控制不同形态的机器人——这是 π₀ 的一个重要贡献。

### 训练

两阶段：

**阶段一——广泛预训练：**
- 互联网规模图文数据（继承自 PaliGemma）
- Open X-Embodiment "Magic Soup" 子集
- 自有多机器人数据：约 **10,000 小时**，约 **9 亿个时间步**，覆盖 **68 个任务**、**7 种硬件配置**

**阶段二——定向后训练：**
- 在高质量、精选的任务 demo 上微调，掌握复杂操作技能

### 关键结果

在论文报告的 zero-shot real-world evaluation protocol 下，π₀ 在复杂操作任务上的表现远超离散 token 方案：

| 任务 | π₀ | OpenVLA | Octo |
|------|-----|---------|------|
| 零样本衣物折叠 | 显著成功 | 基本失败 | 基本失败 |
| 简单桌面清理 | 97.1% | 基本失败 | 基本失败 |

需要注意，这些数字高度依赖具体的 task protocol（trial 数量、机器人形态、task definition、"zero-shot"的具体含义等），因此不宜理解成跨模型的普适性能排序。但它们清楚地表明：衣物折叠和桌面清理这类需要长序列、高精度连续操作的任务，正好是离散 token 方案的弱点。

### 如何理解 π₀ 的性能提升

从 RT-2 到 π₀，连续动作生成确实成为越来越重要的方向。但目前还不能把性能提升简单归因于"连续表示优于离散 token"。π₀ 同时引入了 action chunking、flow matching、跨 embodiment 训练数据以及更大规模的机器人数据。更准确的结论是：**连续动作表示解决了量化和自回归输出的一部分结构性问题，但它的最终收益与数据规模、动作 chunking 和训练 recipe 强烈耦合。**

### 局限

- 完整模型权重未开源（仅 openpi 研究包支持 DROID/Franka 和 ALOHA 平台）
- 某些需要精确力控的任务仍不可靠
- 对全新物理域（自动驾驶、飞行器）的泛化能力未知
- VLM 微调可能导致语言/视觉能力退化（catastrophic forgetting）

## 六、π₀.5 与 π₀.7：从动作生成到策略引导

### π₀.5：离散 + 连续的混合路线

π₀.5（2025 年 4 月，arXiv:2504.16054）的核心贡献不是简单地"加入高层推理"，而是引入了一个**层次化架构**，以及一个非常重要的技术事实：**离散和连续动作表示可以在同一个 foundation model 中承担不同角色。**

π₀.5 的训练分为两个阶段，使用了不同的动作表示：

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

**预训练阶段**使用 **FAST action tokenizer** 将机器人动作离散化，使动作可以和语言、语义子任务等 token 共享 next-token prediction 训练。这使得大规模异构数据（不同机器人、不同任务、甚至 web 数据）可以在统一的 sequence modeling 框架中被利用。

**后训练阶段**引入 **flow-matching action head** 负责高频连续控制。推理时，模型先推断一个高层语义子任务（如 "pick up the pillow"），然后基于这个子任务用 flow matching 生成连续动作 chunk。

一个值得注意的数据是：π₀.5 的第一阶段训练数据中，约 **97.6% 并不是 mobile-manipulator household data**。它通过大量异质数据的混合训练来获得泛化能力，然后在少量目标域数据上微调。

π₀.5 因此说明"离散 vs 连续"不是一个简单的替代关系。它的实际路线不是 "discrete → continuous"，而是 **discrete for scalable multimodal pretraining + continuous for fine-grained control**。

π₀.5 已经能够在训练中未出现的家庭环境执行 10-15 分钟级别的长时序任务，但其成功率仍明显低于受控环境中的短任务。这说明长程任务中的错误累积仍然是 VLA 泛化的主要瓶颈。

### π₀-FAST：为什么离散 token 并没有消失？

Physical Intelligence 后续还探索了 **π₀-FAST**，用 FAST action tokenizer 把连续 action chunk 压缩成离散 token，使机器人动作重新进入 autoregressive language-modeling framework。

这说明 Physical Intelligence 自己也没有认为 discrete action tokenization 是一条死路。实际上，**离散 token 对 multimodal pretraining 仍然有巨大价值**——它使机器人动作可以和语言、视觉、语义子任务共享同一个 sequence modeling 接口。

这引出了一个更重要的技术判断：**不同阶段可能需要不同的动作表示。** 离散 token 更适合统一的 sequence modeling 和大规模异构数据预训练；连续 flow matching 更适合最终的高频精细控制。

### π₀.7：通过多模态上下文引导通用策略

π₀.7（2026 年 4 月，arXiv:2604.15483）进一步探索了 VLA 的"可引导泛化"（steerability）。它的核心创新不是简单地"加入一个世界模型"，而是：

**模型不再只把语言指令作为条件，而是把语言、episode metadata、执行策略信息、视觉子目标以及观测历史等多模态上下文统一作为 policy 的条件输入。**

π₀.7 的模型规模约为 5B，由约 4B 的 VLM backbone、视频历史编码模块（MEM-style video history encoder）以及 860M 参数的 action expert 组成。它仍然沿用 flow matching 连续动作生成路线，但重点从"如何生成动作"进一步转向"**如何通过丰富上下文告诉模型应该采用什么策略**"。

因此，与其把 π₀.7 简单理解成"加入世界模型的 π₀"，更准确的说法是：**它在探索如何让一个通用机器人策略通过多模态上下文被 steer，从而利用异质数据获得跨任务、跨环境和跨 embodiment 的泛化能力。**

π₀.7 报告的结果：在未见过的机器人形态上达到 85.6% 任务进度和 80% 成功率，接近人类遥操作员的 90.9% / 80.6%。

### Visual Subgoal ≠ World Model

π₀.7 引入视觉子目标后，VLA 与 world model 的边界开始变得模糊。但需要注意，**"使用未来视觉子目标"并不自动等价于"拥有一个显式世界模型"**。真正的世界模型通常需要学习 action-conditioned transition dynamics，并能够在内部进行未来状态预测或 rollout。π₀.7 更准确地说是在 policy 中引入了未来视觉目标作为 conditioning signal。

这一区分很重要：一个模型"看到未来目标"与一个模型"能够预测执行动作后世界会如何变化"，是两个不同能力。

## 七、核心技术对比

| 维度 | RT-2 (2023) | OpenVLA (2024) | OpenVLA-OFT (2025) | π₀ (2024) |
|------|-------------|----------------|--------------------|-----------| 
| **核心路线** | VLM → VLA | 开源 VLA | VLA 优化 | VLA + flow matching |
| **参数量** | 5B / 12B / 55B | 7B | 7B backbone + heads | ~3.3B |
| **动作表示** | 离散 256-bin token | 离散 token | **连续 regression** | **连续 flow matching** |
| **动作 chunk** | 否 | 否 | **是，8 步** | **是，50 步** |
| **解码方式** | 自回归 | 自回归 | **并行** | flow integration |
| **控制/推理速度** | 1-3 Hz（55B） | ~4.2 Hz | **109.7 Hz** | **最高 50 Hz** |
| **动作维度** | 7 | 7 | 7 | 18（cross-embodiment） |
| **主要贡献** | 互联网知识迁移 | 开源可扩展 VLA | 速度/成功率优化 | 连续精细操作 |
| **训练数据** | 机器人 demo + web VLM | 97 万 episode | OpenVLA 微调 | 1 万 h + OXE/DROID/Bridge |
| **开源** | 否 | 是 | 是 | 部分（openpi） |

这张表里最值得注意的不是谁"最好"，而是**多个维度的独立演进**：

**动作表示维度**：离散 token（RT-2, OpenVLA）→ 连续 regression（OpenVLA-OFT）→ 连续 flow matching（π₀）。但 π₀.5 又表明离散 token 在预训练阶段仍有不可替代的价值。

**推理效率维度**：自回归逐 token（RT-2, OpenVLA）→ 并行解码（OpenVLA-OFT）→ flow integration + action chunk（π₀）。OpenVLA-OFT 的 109.7 Hz 说明，仅靠并行解码 + 连续回归就可以达到非常高的推理频率。

**数据异质性维度**：这条贯穿始终但容易被忽视的轴线——RT-2 用单一机器人的厨房数据 + web VLM；OpenVLA 用 Open X-Embodiment 的多机器人数据；π₀ 进一步加入自有 1 万小时多 embodiment 数据；π₀.5 / π₀.7 则把异质数据推到极致，通过 diverse context conditioning 让不同来源、不同质量、不同 embodiment 的数据进入同一个 policy。

## 八、VLA 与世界模型：Policy Learning vs Predictive Modeling

这是我认为最值得深入讨论的部分。

### 更准确的区分框架

VLA 路线和世界模型路线的差异，不在于"有没有语言"或"有没有 action"，而更在于**学习目标不同**：

- **VLA 的核心目标**是学习一个 policy——observation + instruction → action
- **世界模型的核心目标**是学习 action-conditioned state transition / future representation——使系统能够预测行动后果并进行 planning

| 维度 | VLA | 世界模型（Dreamer / JEPA 等） |
|------|-----|---------------------------|
| **核心功能** | 学习 policy | 学习 dynamics |
| **输出** | 动作指令 | 预测的未来状态 / 表征 |
| **规划方式** | 隐式（端到端学习到的） | 通常通过显式预测模型进行 trajectory evaluation / goal-conditioned planning，具体实现可以是 search、optimization、MPC 或其他 planner |
| **数据需求** | 操作轨迹 demo | 视频 / 状态转移数据；action-conditioned world model 额外需要 action-observation 对应关系 |
| **优势** | 快速反应控制，语言接地 | 对新情况的想象力，合成数据；可从大量无动作标签视频中学习 |

简单来说：VLA 是一个高度熟练的"反射弧"——看到场景，直接输出动作。世界模型是一个"内心剧场"——先在想象中模拟不同行动的后果，再选择最好的那个。

但需要注意：典型的 imitation-learning VLA 并不显式学习可查询的 action-conditioned dynamics model——但 policy 本身可以隐式编码动态先验。这和"完全没有关于物理世界的内部表征"是两回事。

### 两条路线正在靠近

需要纠正一个常见误解：世界模型路线并不是"没有语言"或"不能做 action"。V-JEPA 2 已经展示了 web-scale video pretraining + action-conditioned world model + V-JEPA 2-AC 的完整技术栈，包括 zero-shot robot deployment 和 image-goal planning。世界模型本身也可以通过语言对齐获得语义能力。

JEPA 路线的规划也不一定是"生成多条轨迹再选择"。它可以是 latent prediction → goal-conditioned planning，可以是 search、optimization 或 policy guidance。

所以更准确的图景是：**VLA 和世界模型正在从两个方向靠近同一个目标——一个同时具备 policy、prediction 和 planning 能力的机器人基础模型。**

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

**数据瓶颈。** 许多主流 VLA 数据集以成功 demonstration 为主，失败 / recovery 数据相对稀缺。机器人真实数据的采集成本远高于互联网文本和图像数据，因此机器人 foundation model 的 scaling law 受到明显的数据获取约束。更有意义的问题可能是：**如何从"成功 demo 数据集"走向包含失败、恢复和策略变化的 experience dataset？** π₀.7 已经开始明确利用 potentially suboptimal autonomous data 作为数据来源之一。

**长时序任务。** π₀.5 已经能够在训练中未出现的家庭环境执行 10-15 分钟级别的长时序任务，但其成功率仍明显低于受控环境中的短任务。五步以上的操作链对所有 VLA 来说仍然是挑战。这不是模型规模的问题，而是长程依赖和错误累积的结构性问题。

**安全。** VLA 是能够直接作用于物理环境的学习型智能体。与纯语言模型不同，它的预测错误可能直接转化为真实世界中的碰撞、夹伤或设备损坏。因此 safety constraint 不能只作为语言层面的 alignment 问题——它是工程硬约束。

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

VLA 的技术演进不是"离散 token 被连续 flow matching 淘汰"，而是**动作表示、时序抽象、预测能力和数据异质性四条轴同时演进**。π₀.5 甚至表明，离散 token 和连续 action 可以在同一个 foundation model 中承担不同角色：前者服务于统一预训练，后者服务于精细控制。

从 RT-2 的大规模 VLM，到 OpenVLA 的开源 7B 模型，再到 π₀ 的约 3.3B VLM+action-expert 架构，研究者开始越来越关注**如何用更小的策略模型配合更好的预训练、机器人数据和动作生成机制获得更强的实际控制能力**。

RT-2 主要解决了让 VLM 能够"说动作"。OpenVLA 证明这条路线可以开源化并规模化。OpenVLA-OFT 进一步解决推理瓶颈。π₀ 通过 flow matching 和 action chunking 把 VLA 推向高频精细连续控制。π₀.5 用离散+连续的混合路线处理长时序和陌生环境。π₀.7 研究如何通过多模态上下文 steer 通用策略。与此同时，V-JEPA 2-AC 等工作从另一侧推进 action-conditioned prediction 和 planning。

**因此，真正值得关注的不是"VLA 最终会不会变成世界模型"，而是未来机器人基础模型是否会同时具备 policy、prediction 和 planning 三种能力。**

正如我在[世界模型盘点](/zh/articles/2026-09-01-world-model-h2-review/)里说的，"world model"正在失去单一含义。VLA 的加入让这个图景更复杂，也更有趣。

*下一篇，我打算深入聊 Sim-to-Real——从仿真到真实机器人的部署鸿沟到底有多宽，以及当前最好的迁移方法是什么。*
