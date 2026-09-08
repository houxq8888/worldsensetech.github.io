---
title: '拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口'
slug: "2026-09-12-multimodal-fusion-interface"
date: 2026-09-12
draft: false
categories: ["具身智能", "多模态感知"]
tags: ["具身智能", "多模态融合", "视觉-触觉", "力觉", "本体感觉", "表示接口", "多模态状态估计", "Hybrid State Estimator", "Structured Belief", "Registration", "Cross-attention", "Modality Dropout", "VLA", "世界模型", "坐标系对齐", "时间对齐", "不确定性"]
description: '多模态融合常被讲成"选哪种 attention 结构"的问题、但机器人场景真正的瓶颈在更上游：Vision / Tactile / Force-torque / Proprioception 四路信号在时间基、坐标系基、任务语义、有效性与来源上从来没对齐过。本文把"融合"拆成 perception → registration → semantic projection → hybrid multimodal state estimation → structured belief interface，指出缺的不是一个更强的 fusion operator、而是一个横跨 heterogeneous observation 与下游 policy / world model / controller / diagnostics 的 structured belief state——带 value / semantics / frame / time / uncertainty / provenance / validity。Contact set 只是这套接口在 contact-rich manipulation 场景下的一个实例、不是接口本身的定义；interface 不规定 Bayesian 推理算法、belief 只是"uncertainty-aware state"、EKF / factor graph / learned filter 都是可选实现。Benchmark 部分把 Interface swap 提到第一主实验、并补上 oracle-slot / estimated-slot / end-to-end 三组 baseline、consistency graph 与 cross-modal contradiction。三条设计原则：Register before compose · Expose belief at the interface · Design for disagreement。'
toc: true
related_articles:
  - 2026-09-11-tactile-force-sensing
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-09-robot-data-scaling
  - 2026-09-07-vla-world-models
  - 2026-09-03-vla-deep-dive
  - 2026-08-26-world-model-in-robotics
---

> 接 [只会看、不会摸：机器人为什么缺一双"手感"的手](/zh/articles/2026-09-11-tactile-force-sensing/)：那一篇钉下三个标签——Action-conditioned observation · Contact-state representation · Closed-loop value——并留下一个明显的问号：**触觉/力觉/本体感觉既然各自都有价值、为什么没有像视觉那样形成一个可复用的共同表示？** 这一篇正面回答这个问号。答案不在模型结构、**在接口**。

想象一个把 RGB 相机、GelSight 指尖、腕部六维力/力矩、关节编码器全部装齐的双臂机器人。硬件清单看起来很"多模态"、但把它交给一个 policy、很多工作会告诉你"用 cross-attention 融合一下"就完事了。这在 demo 里能跑通、**但在真实部署里常常会遇到四类翻车**：某一路掉帧、某个传感器坏掉、坐标系漂移、模型悄悄把某一路当成主导路径而其他路变成噪声。这四类不是工程细节、**它们指向的是同一个根因：模态之间没有先约定好一个可被 policy、world model、controller、diagnostics 共同消费的、显式表达时间、坐标、语义、不确定性、来源、有效性的 structured belief interface**。

这一篇聊具身智能里最容易讲虚的一块：**多模态融合**。不打算推销某个具体网络、也不打算反对 cross-attention、更不打算反对 end-to-end learning。本文真正反对的是——**让 state estimation、cross-modal composition 与 control 三件事全部隐式发生在一个不可诊断、不可复用的 latent interface 里**。文章会先把"融合"拆到最底层的三个基（时间、坐标、任务语义）上、看清楚**视觉为什么先跑通了、触觉/力觉/本体感觉还差在哪、以及如果非要现在动手、最小可用的接口长什么样**。

## 0. 全文框架：从 "fusion layer" 到 "hybrid state estimation + structured belief interface"

先把整篇文章的分析框架放在开头、后面每一节都会回到这张图。

```text
                     ┌────────────────────────────────────┐
                     │         Raw sensor streams         │
                     │  V · T · F · P  (four channels)    │
                     └──────────────────┬─────────────────┘
                                        │
                                        ▼
                        Modality-specific perception
                        (sensor-native encoder、raw → 局部结构观测)
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
      Temporal registration     Spatial registration     Semantic projection
        (clock / timestamp        (SE(3)、hand-eye、
         alias / event bus)        sensor-mount、wrench transform)
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        ▼
                     ┌────────────────────────────────────┐
                     │  Hybrid multimodal state estimator │
                     │  x_t = (q_t, q̇_t, C_t, w_t, m_t, g_t) │
                     │  continuous · set-valued · discrete  │
                     │  mode · task goal                    │
                     └──────────────────┬─────────────────┘
                                        ▼
                     ┌────────────────────────────────────┐
                     │  Structured belief interface       │
                     │  Value · Semantics · Frame · Time  │
                     │  Uncertainty · Provenance ·        │
                     │  Validity / Availability / Lifecycle│
                     └──────────────────┬─────────────────┘
                                        │
              ┌─────────────┬───────────┼───────────┬─────────────┐
              ▼             ▼           ▼           ▼             ▼
           Policy      World Model  Controller  Diagnostics  Safety layer
              │             │           │           │             │
              └─────────────┴───────────┼───────────┴─────────────┘
                                        ▼
                                   Action a_t
```

这张图强调三件事、也是全文真正想立住的架构主张：**fusion 不是一个网络模块、是一整套 hybrid state estimation + interface contract**（所谓 hybrid、是因为待估计的 $x_t$ 天然同时含连续量 $q, \dot q$、集合量 $C_t$、离散 mode $m_t$ 与任务 goal $g_t$）；**registration 与 semantic projection 是 state estimation 之前两段不同性质的准备**（前者是 measurement-layer 的显式时间/几何/标定变量、可通过 calibration 或 estimation 求解、不应默认交给下游 fusion network 隐式学习；后者才是任务相关的表示学习）；**接口层不是 learned embedding、是一份 contract**、里面每一格都有单位、坐标系、时间戳、不确定性、来源、有效性、优化的目标是**可互操作**、不是"对某个模型友好"。

一句可以站得住的中心命题：

> **The missing abstraction is not a better fusion operator, but a structured belief state for heterogeneous, multi-rate observations.**

或者用图表示全文最核心的这条 boxed claim：

$$
\boxed{\;\text{Sensor-specific observations} \;\rightarrow\; \underbrace{\text{Structured Belief}}_{\text{stable interface}} \;\rightarrow\; \{\text{Policy, World Model, Controller, Diagnostics}\}\;}
$$

一旦这么理解、文章里 VLA、world model、tactile foundation model、modality dropout、cross-modal contradiction、F/T constraint、contact set **全部变成同一条主线上的不同实例、而不是六七个并列观点**。

**本文不提出新的 fusion operator**、只提出三条系统层面的设计原则、后面每一节都可以视为这三条的具体化：

1. **Register before compose**——把 temporal / spatial registration 与 semantic projection 从 learned fusion 里剥离出来、作为可标定、可估计、可单元测试的系统变量。
2. **Expose belief at the interface**——用带 semantics、frame、time、uncertainty、provenance、validity 的 structured state 作为多消费者之间的稳定边界；belief 只表示"uncertainty-aware state"、Bayesian posterior 是可选实现之一、不是接口的必要条件。
3. **Design for disagreement**——把 missing modality、cross-modal contradiction、consistency reasoning 当作接口设计目标、而不是部署后的异常情况。

**Contact set 是这套接口在 contact-rich manipulation 场景下的核心实例、不是接口的定义本身**——这一句会贯穿全文、也是本文把自己定位为 *architecture position paper* 而不是 *tactile survey* 的原因。

## 1. "多模态"不等于"多传感器"

这个话题最容易一开始就跑偏——很多人（很多论文引言也一样）把"多装了几个传感器"直接当成"多模态"。这是不成立的。

**"modality" 本身没有一个唯一的物理学定义、它是分析视角的产物**。为了讨论方便、本文的 taxonomy 用一个四要素约定：**measurement space（信号值域）、physical origin（物理来源）、noise model（噪声结构）、update semantics（触发 / 采样 / 时钟语义）**。这四件事一起决定了一个数据流能不能被"当成同一种东西"处理。不同作者可以把边界画得略松或略紧、但只要一次性约定清楚、后面的讨论就不会飘。

按这套 taxonomy、几个常被误当作"多模态"的例子：腕部相机 + 头顶相机是**同模态多视角**；GelSight 指尖的 4 个小摄像头是**一个触觉模态的内部结构**、measurement space 是弹性体表面形变场、下游语义是 contact geometry 而不是 object detection；关节编码器 + 电机电流是**同一 robot-state modality 下的两个 observation channels**、共享同一 latent 物理状态、当作两个模态融合大概率只学到冗余。反过来、**Vision / Tactile / Force-torque / Proprioception** 是本文 taxonomy 下真正的四个不同模态——各自的 measurement space、坐标系、effective observation rate 见 §5 表格。有些工作还会加 audio、thermal、gas、ultrasound 当第五 / 第六模态、分类逻辑一致。9/11 §2.1 用一棵 taxonomy 树把 contact sensing 分成 Tactile / Force-torque / Proprioceptive 三支、本文沿用同样的三分法、把讨论限定在 V + T + F + P。

## 2. 融合之前的三小段：Registration 与 Semantic projection

这一节是全文最技术、也最容易被略过的一节。先做一个**术语收紧**：**"配准（registration）" 与 "语义投影（semantic projection）" 是两个不同性质的事**、把它们都塞进 alignment 会让读者以为只要做 math 变换就够了。

```text
Pre-fusion pipeline
├── Registration        (measurement-layer、显式时间/几何/标定变量)
│   ├── Temporal registration
│   └── Spatial registration
└── Semantic projection (学习任务相关的状态表示)
    └── Raw observation → task-relevant state
```

**Registration 主要属于 measurement layer：它应尽可能由显式的时间、几何与标定变量描述、并优先通过 calibration 或 estimation 求解、而不是默认交给下游 fusion network 隐式学习**。这一句是本文对 registration 的正式定义、比"硬约束 / 有闭式解"要弱一档——因为真实机器人里 registration 常常也是 estimation problem（online calibration、time-varying extrinsics、compliant sensor mounting、tactile elastomer deformation、thermal drift、synchronization / latency estimation）——但**性质仍然是 measurement-layer 的显式变量、不是任务语义**。**Semantic projection** 更接近 perception 与 state estimation、是这一篇后面 §6 structured belief interface 的主角。两者**都在 fusion 之前**、性质完全不同、分工也不同。

### 2.1 Temporal registration（时间配准）

四路信号的默认时间常数差好几个数量级：vision ~15–60 Hz、tactile（相机型）~30–200 Hz、tactile（阵列型）~500 Hz – kHz、F/T ~500 Hz – 1 kHz、proprio ~几百 Hz – kHz。这里有个常被混淆的点：上面这些是 **effective observation rate**、不是 sensor 内部采样率；相机型触觉的 "kHz taxel 采样"、如果它的输出图像仍然只有 30 Hz、那融合层能拿到的有效速率就是 30 Hz。

把各模态重采样到共同低频 policy clock（通常是视觉那一档）是常见 baseline、但**对于 contact-rich control、这会把部分高频事件压缩成不可见的 alias**——滑移检测、瞬态接触力峰、关节冲击这些"事件性"信号往往就发生在两次 vision 帧之间、降采样等于扔掉。更稳健的架构是**多速率共存**：视觉 policy 可以 30 Hz、力/触觉驱动的柔顺控制保持 native rate、事件信号走单独 event bus。三个必须写进接口层的时间细节：**时间戳语义**（sensor timestamp vs arrival timestamp vs host timestamp、三者差 5–20 ms 就足以让"抓杯子的接触瞬间"被错位到"合指之前"）、**时钟漂移**（相机 / controller / 上位机三套独立晶振、几分钟漂几十毫秒）、**事件型 vs 周期型**（slip 触发本质上是 sparse event、塞进均匀分布 buffer 事件密度就丢了）。**判断标准**：**把时间常数差异写成建模假设、不要靠降采样掩盖**。这条与 9/11 §4.1 是同一个判断。

### 2.2 Spatial registration（空间配准）

四路信号天然挂在四个不同坐标系上：Vision camera frame、Tactile sensor frame、F/T sensor frame + tool frame、Proprio base / world frame。想把它们放进同一个融合层、至少要做三件事：**手眼标定**（$T^{cam}_{base} \in SE(3)$、任何一次碰撞都可能让它漂掉零点几度到几度）、**传感器安装标定**（fingertip sensor frame → link frame 的偏移 $T^{\text{sens}}_{\text{link}}$）、**力旋量变换与重力 / 惯性补偿**（把 sensor frame 下的 wrench 变到 base 或 world、扣掉工具重力与惯性项；具体公式见 §5.2）。

空间配准没做好的典型症状：模型在换工具、换相机角度、或者机器人被推歪之后立刻掉点——因为模型学到的其实是"某种传感器 frame 下的相关性"、而不是任务本身。一个更微妙的问题是**相对位姿 vs 绝对位姿**：接触物理本质只依赖"谁碰谁"的相对关系、但很多 policy 直接把末端在 world frame 下的绝对位置喂进去、于是模型学了一堆不该学的自由度。

### 2.3 Semantic projection（语义投影）

这一段严格来说不是"配准"、而是"把观测映射到共同的任务语义坐标"：Vision → object / scene 语义、Tactile → local contact 语义、F/T → aggregate contact 语义、Proprio → self-state 语义。9/11 §2.1 里把 signal → meaning 分成九层（Sensor → Calibration → Raw obs → Contact perception → Contact geometry & wrench → Contact mode & physical state → Task-relevant belief → Policy / controller → Action → New contact）。**多模态融合的关键问题就是：到底在哪一层做 composition？** 常见错误是在 raw 层就 composition（把四路 tensor 拼起来喂进网络）——这相当于强迫 policy 自己学完 §2.1 / §2.2 / §2.3 全部、代价极高、样本效率极低。相对合理的做法是**让每一路各自走完 raw → perception → contact geometry 这几层、然后在"接触事件、力旋量、机器人状态、任务 belief"这四个语义已经收敛的位置上做 composition**、也就是本文后面 §6 的 structured belief interface。

**这一节的落点**：temporal / spatial 两小段 registration 是 measurement-layer 的显式变量、semantic projection 是 perception 与 state estimation 的分工。三件事任何一件糊过去、后面的 fusion architecture 再华丽也是在补前面的锅。

## 3. 现有融合范式的谱系：机制不同、层次不同

Baltrusaitis 等的经典综述 [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) 把多模态 ML 整理为 representation / learning / feature selection / fusion / application 五层、这一节借用其中 fusion 那一层的分类。**先做一个重要的定位**：下面这几种范式**不是互斥的架构选择、而是不同轴上的机制**——一个成熟系统往往同时用其中几种：

| 机制 | 解决什么问题 | 属于本文 §0 哪一层 |
| --- | --- | --- |
| Modality-specific encoder | 各自 raw → 局部观测 | Perception |
| Temporal / spatial registration | 时钟、SE(3)、坐标系 | Registration |
| Cross-attention | 学出来的 composition | 主要在 state interface 之内 |
| Shared latent / VLT-style | 表示层跨模态对齐 | Semantic projection |
| Late / decision fusion | 决策层合成 | Policy / controller |
| Structured state interface | 语义 schema 与 contract | Belief interface |

本文不反对 cross-attention、**限定它的职责**：**cross-attention is not guaranteed to recover explicit temporal / spatial / semantic registration from raw data**——Attention 可以把已经对齐好的表示组合起来、但把 registration 也"顺手学出来"在缺乏显式约束时通常不可靠。这一句是本节与整篇文章的枢纽。

**Early concat**（raw / embedding 层拼接）：把四路 encoder 之后 concat 成 state vector、扔 MLP 或 Transformer。门槛低、样本够多时深度网络会自己学出一些对齐关系。缺点：时间常数差异被 concat 掩盖、容易学到虚假 lead/lag；任何一路掉帧就要么补零、要么 hold-last-value、都是分布外；触觉与视觉 encoder 结构差异大、梯度互相污染；训练时没见过的模态缺失组合一出现、policy 就崩。

**关于 VLA 的一个必须小心的段落**：把 RT-2 / OpenVLA / π0 都归为 "vision + proprio + language concat" 不准确。**RT-2** [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) 的核心是把动作直接**表达成文本 token**、与 VLM 联合 fine-tune；**π0** [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) 是 VLM backbone + **proprioception token** + **noisy action chunk**、通过 flow matching 出 action；**OpenVLA** [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) 的公开配置里明确有多相机、depth、proprioceptive state encoding 分支。**以这些代表性系统为例、可以观察到：现有通用 VLA 的共同点是"以 vision-language 预训练作为主要规模入口、把机器人状态作为额外 representation 注入 policy"、而 tactile / force-torque 尚未形成与 vision-language 相当的公开、跨任务、跨 embodiment 的规模化预训练生态。** 这个缺席不是"忘了加"——加了以后 §2 那三小段都要重讨论、且触觉与 F/T 目前没有对应的预训练数据规模。这个判断在 §9.1 会再展开。

**Cross-attention / Transformer fusion**：把每一路当作 token 序列、上层跑 self / cross-attention。Tsai 等的 Multimodal Transformer [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) 是这条路线的经典起点、显式处理"不同模态时间粒度不一致"。现代 multi-modal transformer 完全可以搭配 timestamp / positional embedding、relative temporal encoding、modality embedding、frame-aware features、modality-specific encoder、modality dropout / masking、auxiliary per-modality loss、attention masks——**真正的问题不是"attention 会翻车"、而是如果没有把这些当接口契约写清楚、attention 会顺手把配准也"学"掉、结果学到的是数据分布里的巧合、不是可移植的接口**。成熟系统里、cross-attention 应该在 **perception 与 registration 已经把输入送到 state interface 之后、只在 state interface 之内做 composition**。

**Late / decision-level fusion**：每一路各出一个子 policy、决策层再加权投票或按 confidence gating。更接近传统 robotics——视觉给粗调、F/T 给细调、触觉给 slip recovery、proprioception 给 nominal trajectory tracking。**一个常见的工程 pattern** 是——任何一路掉线只是掉一个 proposal、不会全局崩、容易加 safety layer。缺点是底层耦合信息丢了（"从触觉和 F/T 合起来看、这个接触是稳定滑动还是 stick-slip"这类跨模态联合信息、分开提 proposal 之后在决策层很难重建）；weighting 规则要么手写不可扩展、要么学又回到 early concat 的问题。9/11 里阻抗控制 + 视觉粗定位 + 触觉滑移检测的组合、本质就是这种 late fusion。

**Shared latent / VLT-style alignment**：用对比学习 / 蒸馏 / CLIP-style 目标、把不同模态的表示拉进同一 latent 空间。代表工作：**TVL / Binding Touch to Everything** [arXiv:2402.13232](https://arxiv.org/abs/2402.13232)（Zhao 等, ICML 2024）用约 44K vision-touch pairs 通过语言建立跨模态 alignment；**AnyTouch** [arXiv:2502.12191](https://arxiv.org/abs/2502.12191)（Feng 等, 2025）针对 heterogeneous visuo-tactile sensors 学统一的 static-dynamic 表示；**3D-ViTac** [arXiv:2410.24091](https://arxiv.org/abs/2410.24091)（Huang 等, CoRL 2024）在该论文的实验中报告 visuo-tactile representation 相对 vision-only 的显著增益；**Lee 等 Making Sense of Vision and Touch** [arXiv:1810.10191](https://arxiv.org/abs/1810.10191)（ICRA 2019）是这条路线最早的自监督锚点。缺点：**训练监督信号哪里来？** 视觉-语言的对比学习能吃海量网络图文对、四元组对齐没有天然监督源；目前主要靠 (a) 遥操作日志把四路强行同采（Calandra 等 *More Than a Feeling* [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) 是这条思路在触觉抓取上的早期实证）、或 (b) 用语言 / 视觉作为桥梁把触觉拉进 V-L 空间（TVL 走的就是这条）。

**这里要显式说明**：上述工作证明了共享 latent 在特定任务族上可行、但它们的表示仍是 policy-specific 或 dataset-specific 的。本文据此提出——如果想让这些表示跨 policy / 跨 world model / 跨 controller / 跨 sensor 复用、需要在 latent 之下再加一层带单位、坐标系、时间戳、不确定性、来源、有效性的 **structured belief interface**。这一层是本文的核心主张、不是那些论文的结论。

**Structured belief interface**（本文推荐路线）：不在 raw 层融合、不在 latent 层融合、而是在一个显式约定的中间表示上组合。这个中间表示不是 learned embedding、是一组语义已经收敛、且带 uncertainty / provenance / timestamp / validity 的 belief 槽位。具体 schema 设计放到 §6 展开。优点：缺一路信号只影响它对应的 key、其他 key 不受污染；时间基、坐标系基、语义基、有效性都收敛在 slot 定义里；slot 一旦定下、fusion 结构可以随便换。缺点：slot 本身要先设计、这活比"套一个 Transformer"重；slot 精度不够时、下游 policy 也学不出超出 slot 的能力；**并且 slot 不解决 vision 提供的 object identity / geometry / free-space / occlusion / scene context 这类"非接触语义"**——本文推荐的是 contact-rich manipulation 场景下的 state interface、不是一般意义上的 multimodal interface。

**这一节落点**：融合范式的谱系里、early concat / attention / late fusion / shared latent 都各有用武之地、但它们的失效模式绝大多数都能追溯到 §2 那三小段没做；本文更倾向于把工程重心从"选哪种 attention"移回"先把 state interface 定下来"。

## 4. 为什么视觉先形成了可复用的数据与表示生态

一个自然的问题：§2 那三小段对视觉同样存在、为什么视觉就能撑起一整个共同数据接口？

**先做一个修正**：**这不是单因果**、是三基固化 + 生态条件共同作用——标准化硬件（CMOS sensor + 统一 lens mount）、统一文件格式（JPEG / PNG / MP4 / HDF5）、坐标模型（针孔 + SE(3)）、大规模互联网数据、成熟标注任务、公开 benchmark（ImageNet / COCO / ADE20K / LVIS）、成熟 encoder、GPU scaling law、可获得的算力。本文只强调三基这一维度、因为这条恰好是触觉目前最缺的、**不是要宣称"三基固化 = 视觉成功"、也不是"视觉先解决了三小段、所以才能成为 foundation model"**。

**时间基上**，视频天然是 frame-indexed、30 或 60 Hz、所有下游任务都约定俗成"以帧为单位"——这个约定消灭了融合讨论里最麻烦的一类问题。**坐标基上**，针孔相机模型 + 内外参把 image pixel ↔ world point 变成一条公式（$s \cdot m = K [R | t] \cdot M$），3D 视觉、SLAM、NeRF、3D Gaussian Splatting、多视角立体都在这个约定上生长；标定错误存在、但"错误长什么样"是可预期、可复现的。**语义基上**，RGB 本身没有语义、视觉社区用 COCO / ImageNet / ADE20K / LVIS 形成了一组**高度互操作的任务级 representation conventions**——object class / bounding box / instance mask / depth / affordance / caption。**更准确地说，这一批数据集并没有形成一个统一 semantic schema**：ImageNet 是 classification ontology、COCO 是 detection + instance + caption、ADE20K 是 scene parsing、LVIS 是 long-tail instance 分布、四者各自定义、只是彼此可组合。

对比触觉：时间基上不同 sensor family 从 30 Hz 到 kHz、事件触发与轮询混合、没有共识；坐标基上 GelSight 是 pixel + 弹性体形变、9DTact 是 pixel + 3D 形变场、阵列 taxel 是一维 / 二维力分布、光学触觉又是另一套表示、连"一个触觉读数到底长什么 shape"都没统一；语义基上接触点、法向、切向、slip、mode 这些概念在论文里都有人用、但缺乏**跨数据集统一的任务级 conventions**。这里也顺便修正一个常见说法——"9DTact 里叫 6D force、GelSight 里叫 shear map、阵列里叫 taxel load、都是同一个物理量不同投影"——**它们包含重叠但不同层级的物理信息**：GelSight 的 shear / deformation map 更接近原始局部形变观测、9DTact 的 6D force 是经过模型反演得到的全局 wrench estimate。两者不是同一个物理量的两种投影、是**观测层与估计层**的差异。这个区别对 §6 slot schema 有直接影响。

**这一节落点**：视觉生态的成功并不是因为 ImageNet / COCO 提供了一份统一的 multimodal state contract、而是因为它们逐渐形成了一组**高度互操作的任务级 representation conventions**；相机模型、标定规范、图像格式、时间轴、成熟 benchmark 又进一步降低了不同系统之间的接口摩擦。触觉 / 力觉 / 本体感觉要形成同规模的融合生态、先要做的是**约定接口**、而不是卷模型结构。

## 5. 每一路信号各自的融合难点

这一节把 §2 已经立起来的时间/坐标/语义先放在一边、只谈每个 sensor 特有的建模假设与难点、给 §6 的接口设计做铺垫。

| Modality | Native evidence | Main ambiguity | Canonical slot output |
| --- | --- | --- | --- |
| Vision | Scene geometry, appearance, semantics | Occlusion, view-dependent pose, identity ambiguity | Geometric prior / contact hypothesis |
| Tactile | Local deformation / force distribution at contact interface | Sensor-family heterogeneity, patch vs point, mounting offset | Local contact observation (patch-aware) |
| Force/torque | Aggregate 6D wrench at sensor frame | Contact decomposition is under-determined; requires model-based compensation | Aggregate wrench constraint + residual |
| Proprioception | Robot internal state $(q, \dot{q}, \tau)$, FK-based EE pose | Model mismatch, friction / backlash, no external-world signal | Kinematic / dynamic constraint |

### 5.1 Tactile：raw 层就没统一

同一类物理量（局部接触界面的形变或力分布）、目前至少有四类**表示完全不同**的实现：

```
Image-based      GelSight / 9DTact [arXiv:2308.14277] / GelSlim / TacTip
                 → RGB 形变图 / 多视角图 / 光流场 / 末端位姿

Taxel array      BioTac / 1D-2D 电容阵列 / Piezoresistive array
                 → 阵列 pressure / 局部法向力分布 / 局部应力图

Optical waveguide AnySkin / DigiTact
                 → 光在弹性体内传播、边缘触发接触点

Proprioceptive-inferred  F/T + kinematics
                 → 反推 contact（"soft tactile"）
```

四路 encoder 结构差异巨大：CNN for images、MLP for taxel arrays、Spline model for waveguide、IK for proprioception-inferred。一个常见误解是"做一个 tactile foundation model 统一处理所有触觉传感器"——这正是 AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) 在做的事、方向对、**但它解决的是 learned representation 层的统一、不是 raw 层的统一、也不是 state interface 层的统一**。即便有 AnyTouch、输出仍然是 embedding、还需要一层 "embedding → contact slot" 的显式约定才能进 state interface。**缓解**：raw 到 slot 之间引入一层 **sensor-specific decoder**、把异构触觉输出统一解码到 §6.1 的 slot schema；这一层可以是解析、也可以是学习的、但它必须是**接口的一部分**、而不是藏在 policy 里。

### 5.2 Force/torque：aggregate、依赖大量前置补偿、且不能反推 geometry

F/T 传感器输出一个 6D wrench、格式统一、但它的**语义**要复杂很多：

$$
w_{\text{raw}} \;=\; w_{\text{contact}} + w_{\text{gravity}} + w_{\text{inertial}} + w_{\text{friction}} + b_{\text{bias}}
$$

要提取"外部接触力"、至少要做：零漂补偿（$b_{\text{bias}}$、tare 归零、真机上很难完美）、重力补偿（$w_g = g(q)$、包含工具 + 夹爪质量分布、工具一换整条曲线全变）、惯性补偿、坐标系变换（把 sensor frame 下的 wrench 变到 base 或 world）。

**关于惯性补偿这里做一个重要的公式收紧**（v3 版本把 joint-space 动力学与 sensor-frame wrench 混在一条公式里、方向也写反了、reviewer 正确地指出：$\tau = J^T F$ 是从 wrench 映到 joint torque、反过来做需要 pseudoinverse 并且依赖 rank / model assumptions）：

- **joint-side estimation chain**：先算 joint residual torque
$$\tau_{\mathrm{res}} \;=\; \tau_{\mathrm{meas}} - \hat{\tau}_{\mathrm{model}}(q,\dot q,\ddot q)$$
  其中 $\hat{\tau}_{\mathrm{model}}$ 通常包含 $M(q)\ddot{q} + C(q,\dot q)\dot{q} + g(q) + \tau_{\mathrm{friction}}$。若模型合理、joint-space 满秩、且无 null-space torque 干扰、则 external wrench hypothesis 可以写成
$$\hat{F}_{\mathrm{ext}} \;=\; (J^T)^{\dagger}\, \tau_{\mathrm{res}}$$
  这里 $(\cdot)^{\dagger}$ 是 weighted pseudoinverse、解不唯一、需要额外的最小 norm 或 task-space 加权假设。
- **wrist F/T 侧是另一条 chain**：sensor 直接输出 $w_{\text{raw}}$、通过 SE(3) 伴随变换与 sensor-frame 的 bias / gravity / inertial 补偿得到 external wrench $w_{\text{ext}}^{\text{sens}}$、再变到 base 或 tool frame。**不要**把两条 chain 混成一条公式——joint-space dynamics term 与 wrist sensor 看到的 inertial wrench 是两类不同的物理量、mapping 方向也不同。

慢速操作时、joint-side residual 常常足够；高速操作时、inertial 与 actuator dynamics 都不可忽略。**这一层任何一步出错、下游的"法向力 / 切向力分解"就整个错**。

F/T 还有一个更本质的限制——**它给出的是"整体"力旋量、不是"局部"接触**。这个判断可以凝练成一条 boxed principle：

$$
\boxed{\;\text{Force/torque measures an aggregate wrench; it does not, on its own, uniquely decompose into individual contacts.}\;}
$$

$$
w \;=\; \sum_{i=1}^{N} \begin{bmatrix} f_i \\ (p_i - p_0) \times f_i \end{bmatrix}
$$

无穷多组 $\{p_i, f_i\}$ 可以给出同一个 $w$。**需要限定**：在给定 contact geometry、robot kinematics、object geometry 与 contact model 的条件下、F/T 可以**间接**产生较强的 localization constraint（例如"只有一个候选接触点时、F/T 就能把力值定位到该点"）；问题不是"F/T 不能定位"、而是**"F/T 单独不提供唯一的 contact decomposition"**。这一条对 §6.2 的 slot 语义有直接影响。

**缓解**：F/T 在 slot 层应该输出两件事——(a) 已补偿的 external wrench（用于 force-aware policy）、(b) 残差 magnitude（用于检测"是不是又漂了 / 是不是又撞到不该撞的东西"）。第二件事常常被忽略。

### 5.3 Proprioception：最直接、也最容易被过度依赖的一路

Proprio 是四路里默认时间常数最快的、也**通常是机器人系统中最直接、最稳定、最容易获得的一类内部状态观测**——这一句需要限定、"完整通道"是不准确的：很多机器人没有直接 torque sensing、$\dot q$ 往往是数值微分、EE pose 往往是 FK 估计、motor current 与 joint torque 之间有摩擦 / 齿隙 / 传动比映射、某些软体机器人或低成本平台的 proprio 甚至非常残缺。所以更稳的表述是——它是**最容易标准化、也最容易当作坐标系与时间基 anchor 的一路**。

Proprio 在融合里通常有两个作用：**作为 contact hypothesis 的输入**——给定期望末端轨迹 + 关节力矩反馈、可以按 §5.2 的 joint-side chain 反推 external wrench hypothesis（contact inference、Hogan 早年阻抗控制一脉的经典工具）；**作为时间基与坐标系的锚**——融合层需要一个稳定的 frame、proprio 天然跑在硬件 servo 允许的速率上限、EE pose 也可以当作所有其他坐标系的 anchor。**融合难点**：proprio 的信息量太"干净"——它是机器人自己测自己的状态、没有外部世界的不确定性、模型很容易**过度依赖 proprio**、把它当作 shortcut、结果在没有接触的场景表现好、有接触的场景反而没学会用触觉 / F/T。这一条正是 9/11 §5.2.1 feedback-value ablation 想避免的。

### 5.4 时间常数差异本身就是建模假设

把 §5.1–5.3 与 §2.1 合起来看、可以提炼出这条判断：**不同模态的 effective observation rate 差异、不是"融合层要不要处理一下"的工程问题、而是"任务需要什么控制带宽"的建模假设**。擦拭 / 打磨这类任务力控带宽至少 100–500 Hz、视觉 30 Hz 完全够用、触觉与 F/T 得走 native rate、proprio 也得到力矩层——这一类任务、融合层不能把所有信号降到 30 Hz；抓取-放置 / 装配这类任务视觉主导、接触事件稀疏、把触觉降采样到 vision rate 是可接受的近似。具体数值随硬件、任务、controller 架构变化、这里不写死。

## 6. 一个最小可用的 structured belief interface

到这里可以正面回答"到底怎么办"了。这一节给一个**可实施、可 benchmark、可增量演进**的最小接口。

**先做一个范围收紧**：本节要提的不是 "所有机器人多模态信息都应该压缩成 contact set"、也不是 "structured belief interface ≈ contact set"。**Contact set 是这套接口在 contact-rich manipulation 场景下的核心实例、不是接口本身的定义**——这一句决定了 locomotion / navigation / whole-body manipulation 的读者能不能把自己的状态塞进同一份 contract。接口本身至少包含：

```text
Structured belief interface
├── robot_state      ── (q, q̇, τ, EE pose) + availability
├── scene_state      ── free-space / object pose / occlusion …（可选）
├── task_context     ── language token / goal embedding
├── wrench_ext       ── 6D + covariance + validity
├── contact_set      ── 本文 §6.1 详细展开、contact-rich manipulation 的核心 object
└── belief (optional)── task-relevant latent / explicit
```

对没有接触语义的任务（例如纯导航）、contact_set 字段可以整块缺席、接口仍然成立。**contact set 是实例、接口是 contract**。

### 6.1 Contact slot schema

先定义跨模态共享的**接触事件记录**（升级版、带 uncertainty / provenance / timestamp / validity）：

$$
C_t = \big\{\, \big(\, \mathrm{track\_id}_i,\; G_i,\; \mathcal{F}_i,\; m_i,\; \mathcal{U}_i,\; \mathcal{P}_i,\; \mathcal{I}_i,\; \mathcal{L}_i \,\big) \,\big\}_{i=1}^{N_t}
$$

$G$ 几何、$\mathcal{F}$ 力旋量、$m$ mode、$\mathcal{U}$ uncertainty（拆开写）、$\mathcal{P}$ provenance（对象化）、$\mathcal{I}$ evidence mask、$\mathcal{L}$ lifecycle。逐字段展开：

```python
contact = {
    # 身份（跨帧一致只是"假设"、不是内禀属性）
    "track_id":            Optional[int],
    "track_confidence":    float,           # ∈ [0, 1]；track_id=17 与
                                            # "track_id=17 + confidence=0.97"
                                            # 在接口语义上完全不同

    # 几何：不假设一定是 point contact
    "geometry": {
        "type":            enum,            # point | patch | region
        "position":        Quantity,        # {value: Vector3, unit: "m",
                                            #  frame_id: "base"}
        "normal":          Quantity,        # {value: Vector3, unit: "-",
                                            #  frame_id: "base"}
        "extent":          Optional[Quantity],  # 若 type ≠ point
        "center_of_pressure": Optional[Quantity],
    },

    # 力旋量（canonical point / patch summary）
    "wrench": {
        "f_perp":          Quantity,        # {value: float, unit: "N", frame_id}
        "f_parallel":      Quantity,        # {value: Vector2, unit: "N", frame_id}
        "moment":          Optional[Quantity],  # 若 type ≠ point
    },

    # 模态与事件
    "slip_probability":    float,           # ∈ [0, 1]
    "mode":                enum,            # free / touch / sticking /
                                            # sliding / rolling / separating

    # 不确定性：连续与类别必须分开、不能塞同一个矩阵
    "uncertainty": {
        "pose_covariance":      Matrix,     # Σ on ℝ³ or SE(3)
        "force_covariance":     Matrix,
        "slip_probability":     float,      # 标量、非 covariance
        "mode_probability":     Vector,     # categorical distribution
        "calibration_quality":  enum,       # nominal / degraded / unknown
        # 备注：covariance 本身并不区分 measurement noise /
        # model uncertainty / calibration uncertainty。若需要
        # 更精细归因、可显式拆 aleatoric / epistemic / calibration
        # 三个源、或至少在训练侧记录 σ_sensor、σ_model、σ_calibration。
    },

    # 来源（provenance）：对象化、不再是单个 enum
    "provenance": {
        "primary_sources":      ["tactile", "force_torque"],   # 主贡献源
        "contributing_mask":    [V, T, F, P] → [0/1, 0/1, 0/1, 0/1],
        "estimator":            "contact_estimator_v2",
        "calibration_version":  "...",
    },

    # 时间与生命周期（§6.1.1）
    "timestamp":           float,           # host clock、seconds
    "age":                 float,           # seconds
    "valid_from":          float,
    "valid_until":         float,
    "availability":        enum,            # present / stale / delayed /
                                            # unavailable / corrupt
    "lifecycle":           enum,            # new / tracked / occluded /
                                            # lost / merged / split
}
```

Schema 里所有 numeric 字段类型都是 `Quantity = {value, unit, frame_id, ...}`、unit 用 SI canonical 记法（米、牛顿、牛·米、秒、弧度）、frame_id 引用注册过的 frame 命名空间。这一层不是"语义上要求有 unit"、是"schema 可以机器验证 unit 与 frame"——§6.8 的 runtime validator 会 assert 每一条 slot 的 unit / frame 组合。

这份 schema 刻意做到五件事：**sensor-agnostic**（GelSight、taxel 阵列、F/T——只要能填这些 key、就能进 state interface）；**物理可解释 + 单位明确**（每一个数字有明确 SI 量纲与坐标系）；**带 uncertainty**（连续 covariance 与类别概率分开、并保留 aleatoric / epistemic / calibration 的进一步拆分接口；**没有 uncertainty 的 slot 不是接口、是"事实"、真机上永远不成立**）；**带 provenance**（primary_sources + contributing_mask + estimator + calibration_version——与 §7.5 modality dropout、§8.6 contradiction、§8.7 consistency graph 直接对应）；**带 validity / availability / lifecycle**（下面三小节展开）。

#### 6.1.1 Validity / Availability / Lifecycle 为什么是一等公民

同一个 slot 值、可能对应三种完全不同的状态：

```text
value 相似、age 很大        ── historically valid + temporally stale
value 新、covariance 很大   ── fresh + uncertain
value 缺失                  ── unavailable
```

对 controller 而言这三态的处理也不同：stale 时应做 hold + covariance inflation + fallback；fresh-uncertain 时继续消费但降权；unavailable 时走 §6.6 的显式退化路径。这里 v3 有一处措辞可以更严谨——严格来说、"stale observation 对过去的状态仍然 valid、但对当前状态未必 valid"、所以本文统一改成 **"staleness is not invalidity; it is validity relative to a past timestamp"**。如果接口层只能表达"有值 / 没值"、下游就只能靠猜；**这一层是接口设计本身的要求、不是 benchmark 技巧**。

#### 6.1.2 track_id 是一个 hypothesis、不是内禀属性

v2 把 id 写成"tracking ID、跨帧一致"过乐观。contact set 本质上是 **permutation-invariant set**、不天然存在稳定 identity：多指操作、rolling contact、contact patch splitting / merging、遮挡、瞬态接触都会让 identity assignment 本身成为 inference problem。v3 起把字段改成 `track_id: Optional[int]` + `track_confidence: float` 并明确：

> **track_id is a hypothesis maintained by temporal association, not an intrinsic physical property of a contact.**

没有 track_id 时 slot 依然有效；有 track_id 时它是一个可被下游消费的、带 confidence 的假设。`track_confidence` 也可以被 lifecycle 字段部分覆盖（例如 `lifecycle = "split"` 意味着两条子 slot 的 track_confidence 应该同步下调）。

#### 6.1.3 Point contact 是一个建模假设、不是物理事实

$p$ / $n$ / $f_\perp$ / $f_\parallel$ 这一组经典描述其实是 **point contact 或 local contact patch 收缩到一个代表点** 的 canonical summary。很多触觉任务不是 point contact——finger pad ↔ object 实际是一个 contact patch $\mathcal{A}_i$、上面有 pressure distribution、shear distribution、normal force distribution、center of pressure、torsional moment。GelSight 类 sensor 的 raw observation 更自然对应 contact patch geometry、而不是单点 $p + n$。接口层显式带 `geometry.type ∈ {point, patch, region}` + `extent` + `center_of_pressure` + `moment`。退一步说、$p, n, f_\perp, f_\parallel$ 至少被称为 **"canonical point/patch summary"**、不要暗示这是 contact 的完整物理描述。

### 6.2 每一路信号怎么映射到 slot · F/T 是 constraint、不是 detector

四路合并的信号 → slot 语义映射：

```text
Tactile         → local contact observations  （直接观测、有 sensor noise）
F/T             → global wrench constraint     （aggregate 层约束、不定位、不分解）
Proprioception  → kinematic / dynamic constraint  （J^T 反推、依赖模型精度）
Vision          → geometric prior              （预测 contact hypothesis、非观测）
```

一个自然的 formulation 是 Bayesian：

$$
p(C_t \mid V, T, F, P) \;\propto\; p(V \mid C_t)\, p(T \mid C_t)\, p(F \mid C_t)\, p(P \mid C_t)\, p(C_t)
$$

其中 $p(T|C_t)$ 是局部观测 likelihood、$p(F|C_t)$ 是 wrench consistency likelihood、$p(P|C_t)$ 是动力学 consistency likelihood（$\tau_{\mathrm{res}}$ 残差）、$p(V|C_t)$ 是几何 / 视觉预测 likelihood、$p(C_t)$ 是先验。

**两个必须写下来的 caveat**——

**(i) Conditional independence 是简化**。上面这个 factorization 隐含 $p(V,T,F,P|C) = p(V|C)p(T|C)p(F|C)p(P|C)$。真实机器人里 Vision / Proprio / F/T / Tactile 之间存在大量共享变量：robot pose、object pose、contact geometry、dynamics、calibration、actuator state（例如 F/T 与 proprio 共同依赖 $(q,\dot q,\tau)$）。**这里采用条件独立的简化 factorization、只是为了把证据融合关系写清楚；实际系统需要显式建模 cross-modal correlations**（joint Gaussian likelihood + shared covariance、或者 graphical model / message passing over a coupled factor graph）。

**(ii) Bayesian 是可选实现、不是接口要求**。**The interface does not prescribe a Bayesian inference algorithm. "Belief" here denotes uncertainty-aware state information; Bayesian posterior inference is one implementation, not a requirement of the schema.** 同一份 §6.1 slot schema 可以由 EKF / UKF、factor graph、particle filter、learned filter、diffusion state estimator、Transformer state estimator、hybrid neural-symbolic estimator 产生——只要输出满足 §6.4 那条 7-tuple 公式（Value / Semantics / Frame / Time / Uncertainty / Provenance / Validity）。这一句让接口从"某个 probabilistic architecture"变成真正的"interface proposal"。

### 6.3 Evidence ranking：hypothesis-dependent、不是固定排序

一个固定排序 "tactile > F/T > proprio > vision" 是错的。正确的表述是 **evidence ranking should be hypothesis-dependent**：

```text
Contact location:    Tactile > Vision > F/T ≈ Proprio
Global wrench:       F/T > Tactile > Proprio > Vision
Object pose:         Vision > Tactile > F/T ≈ Proprio
Joint state / τ_res: Proprio >> others
Contact mode:        Tactile > F/T ≈ Proprio > Vision
Slip probability:    Tactile > F/T (rate) > Proprio (residual) > Vision
```

**同一个 $C_t$ record 里不同字段的"主导模态"可以完全不同**——这也是为什么 §6.1 provenance 要同时记 `primary_sources` 与 `contributing_mask`。

### 6.4 Interface ≠ learned latent representation

这一小节是全文架构主张最核心的一段。

```text
Latent representation             Structured belief interface
─────────────────────────         ─────────────────────────────
优化目标：对下游模型友好            优化目标：对多个消费者互操作
单位：无                             单位：Quantity 里显式记 canonical unit
坐标系：无（隐含在数据流里）         坐标系：显式声明、字段附 frame_id
时间：无                             时间：显式 timestamp + age
有效性：无                             有效性：availability / lifecycle / valid_until
不确定性：无（或仅在损失里体现）     不确定性：连续 covariance + 类别分布 + calibration metadata
来源：无                             来源：primary_sources + contributing_mask + estimator
消费者：一个 model                   消费者：policy / world model / controller /
                                       diagnostic tool / safety layer / 另一个 sensor
```

一句可以直接留下来的公式：

$$
\text{Interface} \;=\; \big(\, \text{Value},\;\text{Semantics},\;\text{Frame},\;\text{Time},\;\text{Uncertainty},\;\text{Provenance},\;\text{Validity} \,\big)
$$

**一个 latent 可以非常适合 policy、却不适合作为 world model、controller、diagnostic tool 或另一个 sensor 的输入接口**——这是接口存在的必要性、也是 §3.4 那些 shared latent 工作真正缺的那一环。**视觉生态之所以能撑起跨系统复用、并不是因为 ImageNet / COCO 提供了一份完整的 multimodal state contract（它们并没有定义 SI unit、frame、timestamp、covariance 与 validity lifecycle），而是因为它们形成了一组高度互操作的任务级 representation conventions、再叠加相机模型 / 标定规范 / 文件格式 / 时间轴约定、把跨系统的接口摩擦降到了很低的水平**。触觉、力觉、本体感觉现在需要的、是在这一层 convention 之上更进一步——一份**跨 sensor family / 跨 embodiment / 跨 consumer 的显式 state contract**。

### 6.5 融合的时机：不在 raw、不在决策、**在 structured belief interface 上**

有了 §6.1 的 slot schema 之后、fusion 变成三步：

```python
# Step 1: per-modality perception + registration
raw_v  → enc_v → pred_contact_hypothesis      （视觉的 contact 预测、附 confidence）
raw_t  → enc_t → detected_contact             （tactile 观测、附 pose_covariance）
raw_ft → enc_ft → external_wrench + residual  （补偿后 wrench、附 force_covariance）
raw_p  → proprio_state → τ_res → (J^T)† τ_res → inferred_contact  （附模型误差）

# Step 2: hybrid state estimation → 写到 belief interface
# （Bayesian / EKF / factor graph / learned filter 都是可选实现）
C_t         = state_estimator(V, T, F, P)   # 输出的是 belief、不是 hard fact
wrench_ext  = compensate_ft(raw_ft)
robot_state = (q, q̇, τ, EE_pose) + availability
belief_t    = belief_update(C_t, wrench_ext, robot_state, task_ctx)

# Step 3: policy / world model / controller / diagnostics 消费 belief interface
a_t = policy(belief_interface)
```

关键判断——**Step 2 的 interface schema 才是整个多模态系统的接口**、Step 1 里各模态可以任意换 encoder、Step 3 里 policy 可以任意换结构、只要 Step 2 的 schema 稳定。同时——**structured belief interface 是系统的稳定互操作边界、但不必成为所有下游模型唯一的信息通路**：policy 完全可以同时接收 structured slots + 额外的 raw visual feature / language token / task embedding。真正需要守住的是**跨消费者共享的那部分状态**走这份 contract、而不是"所有信息必须穿过这里"。

### 6.6 缺模态时的 fallback：把"少一路"当成训练分布的一部分

真实部署里、传感器坏掉、掉帧、超时是常态。接口层要显式支持"某一路暂时没数据"：

```python
if tactile_missing:
    C_t = state_estimator(V, F, P)         # 无 T、p(T|C_t) 换成 uniform likelihood
    for r in C_t:
        r.uncertainty.pose_covariance  *= inflation_T_missing
        r.uncertainty.mode_probability  = uniform_over_modes
        r.provenance.contributing_mask.T = False
        r.provenance.primary_sources.remove("tactile")
        r.availability     = "unavailable"
```

**这不是工程补丁、这是训练时的核心约束**。见 §7.5 modality dropout、§8.5 degradation modes、§8.6 cross-modal contradiction test。

### 6.7 wrench_ext 与 contacts[]：aggregate observation + consistency constraint、不是冗余

一个 reviewer 会立刻问的问题：如果 `contacts[]` 里已经有每个 contact 的 force 与 moment、为什么还需要一个 global wrench？答案是——

$$
w_{\text{ext}} \;\neq\; \sum_i w_i^{\text{reconstructed}}
$$

`wrench_ext` 是 **独立的 aggregate measurement**（F/T 传感器直接给出的整体力旋量）、`contacts[]` 是 **structured hypothesis**（每路感知 + state estimator 联合推断的结果）。二者是两类不同的观测、相互约束：

$$
\mathcal{C}_{\text{wrench}} \;=\; \big\| w_{\text{ext}} - \textstyle\sum_i \big[f_i;\, (p_i - p_0) \times f_i\big] \big\|_{\Sigma^{-1}}
$$

这个 residual 越大、说明 contact hypothesis 与 F/T 观测越不一致、系统应该 raise 一条 consistency alarm、或者把 `mode_probability` 拉向 uniform。**`wrench_ext` 不是 `contacts[]` 的冗余字段、而是一个独立的 aggregate observation、可以作为 contact-set reconstruction 的 consistency constraint**——这也是 §8.7 consistency graph 里 F↔T、F↔V、F↔P 三条边的物理基础。

### 6.8 Schema evolution / interface compatibility（unit / frame validator 的落脚点）

如果接口真的存在、它必须回答一个工程上非常现实的问题：**旧 policy 能不能消费新 sensor 版本？**

```text
v1  contact: p, n, force
v2  contact: p, n, force, slip_probability, covariance
v3  contact: geometry{type, pose, extent}, wrench{f, m}, mode_distribution
```

只要接口是 API、就要处理 API versioning。最小的一组约定：

```text
Interface compatibility
├── version tag            # 每一条 slot 附 schema version
├── optional fields        # 老消费者可以忽略新字段、新消费者用 default 处理缺失
├── backward compatibility # v2 消费者能读 v1 slot、v1 消费者能安全降级读 v2
├── unit / frame validator # schema metadata 定义每一个字段的 canonical unit 与
│                          # frame namespace、runtime assert；Quantity{value, unit,
│                          # frame_id} 是这份 validator 的落脚点
└── graceful degradation   # 缺字段时的 inflation / default / alarm 策略
```

没有这一层、"state interface = API" 的类比就不成立。这一层也是 §8.1 interface swap benchmark 的前提。

## 7. 融合失败模式与诊断

reviewer 视角、下面这五类失败在机器人多模态工作里最常见、也最容易被"我们用了 cross-attention 所以鲁棒"这类话糊过去。每一类统一四段：**Failure mode → Observable symptom → Diagnostic test → Mitigation**。

### 7.1 Modality collapse：policy 悄悄退回视觉 / proprio 主导

- **Failure mode**：某些模态在训练里被边缘化、变成事实上的 dead input。
- **Observable symptom**：训练 loss 与 validation success rate 都正常；一旦评测遮住视觉、policy 表现几乎不变。
- **Diagnostic test**：主指标 **modality ablation gain**（§8.4）——若 $\Delta_{\text{mod}} \approx 0$、说明这一路在 policy 里其实没被消费。3D-ViTac [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) 的 ablation 之所以重要、就是给这个指标提供了一个明确的参照（**该论文实验中报告的 visuo-tactile > vision-only 增益、是这种 gain 存在性的证据、不是普遍定律**）。Attention 权重可视化只能算 **辅助 diagnostic visualization**、不能拿来判断 modality contribution——**attention ≠ causal importance**、一个模态 attention 权重低不代表它没有贡献、一个模态 attention 高也不代表去掉它性能一定下降；这是经典 interpretability 陷阱。
- **Mitigation**：(a) 训练时显式跑 modality dropout、见 §7.5；(b) 每个模态加 auxiliary supervision（例如 tactile encoder 除了给 policy 用、还得独立预测 slip 事件）；(c) 用 9/11 §5.2.1 的 feedback-value benchmark 而不是纯 success rate 逼出真实贡献。

### 7.2 Temporal smearing：所有信号被强行插值到 30 Hz

- **Failure mode**：高频物理事件被降采样抹掉。
- **Observable symptom**：滑移检测、瞬态接触、冲击类任务的 policy 学不出来。
- **Diagnostic test**：对比"以 30 Hz 融合 vs 以 native rate 融合"的成功率差；或者用 $\Delta_{\text{tail}}$ 只看困难接触条件下的表现。
- **Mitigation**：见 §5.4——把时间常数差异当**建模假设**；不同层用不同频率、事件走 event bus；不要在接口层把高频信号降采样掉。

### 7.3 Frame confusion：模型学到坐标系伪相关

- **Failure mode**：模型学到的是某种特定传感器 frame 下的相关性、不是任务本身。
- **Observable symptom**：模型在训练位姿、训练相机摆放、训练工具型号下表现优秀、稍微动一下就崩。
- **Diagnostic test**：评测集主动做**坐标系扰动**（四类细分见 §8.8）。
- **Mitigation**：state interface 里显式带 `frame_id`、下游消费者必须做变换才能读；训练时把坐标系扰动当 domain randomization 的一部分（Tobin 等 [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)）。

### 7.4 Semantic leakage：raw 视觉路径绕过 slot 定义接触语义

- **Failure mode**：接触状态的定义被未约束的视觉 latent 悄悄改写。
- **Observable symptom**：slot 明明定义成"接触几何"、但 policy 表现严重依赖视觉外观（同一个物体的照片换背景就掉点）。
- **Diagnostic test**：把 slot 里的视觉贡献换成一个"最小充分"的合成 slot（例如把预测 contact 用真值 contact 替）、看性能变化。
- **Mitigation**：这里 v3 的措辞过绝对、需要收紧——本文**不是**主张"raw pixel 不允许直接进 policy"（现代 VLA 本来就是 image → vision encoder → token representation → policy、raw visual latent 进 policy 是常规操作）；本文主张的是——**对于由 state interface 定义的 contact-related state、不应存在绕过 slot encoder 的 raw-visual shortcut**。也就是说"contact 语义的唯一通路必须是 slot encoder"、而不是"视觉特征不许进 policy"。这条与 9/11 §3.5 一致、也更可实施。

### 7.5 Missing / degraded modalities：一坏就崩、或者悄悄退化没报警

- **Failure mode**：训练分布里模态永远齐全、上线时的退化不在分布内。
- **Observable symptom**：触觉掉帧一次整个 policy 行为剧烈变化、或悄悄退化没报警。
- **Diagnostic test**：跑 **modality masking / dropout test**（按 (1-p) 概率随机丢掉某一路）；modality masking / dropout 是 missing-modality robustness 中**常见的一类训练策略**、可参 Maiga 等 MMP [arXiv:2410.03010](https://arxiv.org/abs/2410.03010)。**但 dropout ≠ sensor failure**。真实部署里的模态退化通常不是 Bernoulli(p) 独立丢、更常见的是 missing / stale / delayed / corrupted / biased / noisy 六类。
- **Mitigation**：接口层要能表达这六种状态、不能只有"有 / 无"两个 flag。§6.1 的 `timestamp / age / availability / lifecycle / uncertainty` 就是为了让 stale / delayed / biased 在数据格式上就是可检测的——**这是接口设计本身的要求、不是 benchmark 的技巧**。

## 8. Benchmark 与评估

这一节把 §7 里散在各处的诊断指标整合成一个可执行的 benchmark 骨架。**排序上把 interface swap 提到 8.1**——因为 dropout 类指标证明的是"模态鲁棒性"、而 swap 证明的是"接口互操作性"、后者更贴近本文核心 claim。整体思路沿用 9/11 §5.2.1 的 feedback-value 主张：**benchmark 应该测接口是不是真的抽象边界、模态是否真的不可替代、系统能否正确处理 disagreement、不是"这个模型能不能记住训练分布"**。

### 8.1 Interface swap（interoperability benchmark、本文第一主实验）

如果 interface 是 abstraction boundary、那么换 encoder / 换 consumer 应该不需要重新设计中间层。三组具体实验：

1. **Sensor encoder swap**：tactile encoder A → encoder B（不同 backbone、甚至不同 sensor family）、slot schema 不变、policy 权重不变、看性能迁移。
2. **Policy swap**：MLP policy → Transformer policy、slot 不变、看下游训练成本。
3. **Consumer swap**：policy ↔ world model ↔ controller ↔ diagnostic tool 四路消费者共享同一份 slot、各自独立训练、看接口是不是真的"可复用"。

若三组都能做到"换 encoder / 换 consumer 不需要重新设计中间层"、这个实验比再跑一个 success rate 强得多——**它把"接口"从 metaphor 变成 measurable property**。§8.1 与 §6.8 是一对：schema 保证向前兼容、swap 保证横向可插拔。

### 8.2 Oracle-slot / Estimated-slot / End-to-end：三组对照 baseline

要证明"state interface 是不是必要"、必须做三组 baseline：

```text
A · Oracle interface   : ground-truth contact slots → policy
B · Estimated interface: sensor → state estimator → estimated slots → policy
C · End-to-end fusion  : sensor → fusion policy（不显式接口）
```

于是 $S_A$ = 接口质量上限、$S_B$ = 接口 + 估计、$S_C$ = 端到端。至少实验上能区分：性能损失发生在 **perception / interface**（$S_A$ 高、$S_B$ 掉）、发生在 **policy**（$S_A$ 与 $S_B$ 都掉）、还是"接口这条抽象根本就没帮上"（$S_C \geq S_B$）。**$S_{\text{total}} \approx S_{\text{interface}} \times S_{\text{downstream}}$ 这个因式分解、是本文核心 claim 最直接的实验检验**。少了这组对照、其他 benchmark 都只是在描述"接口好不好用"、而不是在检验"接口是不是必要"。

### 8.3 Slot 层的 fidelity 单独测

**slot 本身的准确度就应该独立于 policy 测**：contact position 与真值的 IoU / distance error；slip detection 的 AUROC；contact mode 分类的 macro-F1；**uncertainty calibration**——连续部分用 reliability diagram 或 negative log-likelihood、类别部分用 Brier score 或 expected calibration error（ECE）；patch 场景加 extent / center-of-pressure error；track_id 加 identity preservation（ID-switch 次数、MOTA / IDF1）。这些是"融合层之前的质量指标"、它们不合格、policy 层的成功率高只可能是过拟合。

### 8.4 Modality information gain 与 graceful degradation

**信息增益**（每个模态的贡献上限）：

$$
\Delta_m \;=\; S(M) - S(M \setminus m)
$$

**Graceful degradation**（部分损失下的退化形状）：

$$
G_m(p) \;=\; \frac{S(M, p) - S(M \setminus m)}{\Delta_m}
$$

**符号定义先明确**——$S(M, p)$ 表示"模态 $m$ 以独立 dropout 率 $p$ 被降级时的性能"、并规定 $S(M, 0) = S(M)$；于是 $G_m(0) = 1$、$G_m(1) = 0$、中间值刻画退化曲线形状。

一个"好系统"可以是 $\Delta_m$ 很大（这个模态不可替代）、但 $G_m(p)$ 从 1 平滑降到 0（部分丢失不会立刻崩）——这比"越平越健康"更贴合真实的 sensor 重要性分布。**同时要明确一点**：不要暗示好系统的 $G_m(p)$ 曲线一定单调或平滑——现实中常有 threshold effect（触觉分辨率降到某个阈值以下 slip 就完全检测不了）。更准确的表述是——**graceful degradation should be characterized rather than assumed monotonic or smooth**、报告形状、不预设形状。

### 8.5 Degradation modes beyond dropout

对应 §7.5、把测试从 Bernoulli dropout 扩到 missing / stale / delayed / corrupted / biased / noisy 六类。对每一类退化分别报成功率-曲线。**这才是"接口有没有真的处理 uncertainty / provenance / validity"的直接检验**。

### 8.6 Cross-modal contradiction test：$D_{ij}(z)$ 定义

**Good fusion $\neq$ agreement**。这一节是全文最值得强调的 benchmark 维度、也是最有可能形成独立品牌的一块。Interface 存在的意义就是**当不同模态说不同话时、系统能不能"正确地不确定"而不是"平均成一个错误结果"**——那 benchmark 就应该主动制造跨模态矛盾。

**Definition**：contradiction 是**两个模态对同一个可验证 latent 变量 $z$ 的预测不一致**：

$$
D_{ij}(z) \;=\; d\!\big(p_i(z),\, p_j(z)\big)
$$

例如 $\mathcal{C}_{VT}$ 是 vision 与 tactile 在 contact position 上的距离、$\mathcal{C}_{TF}$ 是 tactile 与 F/T 在 normal force 上的距离、$\mathcal{C}_{VP}$ 是 proprio 与 vision 在 EE pose 上的距离。三类常见实例：

```text
Spatial disagreement    vision 报 contact 在 A、tactile 报在 B（5/10/20 mm 三档）
Temporal disagreement   F/T 报 contact start 在 t、tactile 报 t + Δ
Force disagreement      tactile 报 2 N 法向、F/T-consistent reconstruction 需要 8 N
Kinematic disagreement  proprio 报 EE pose X、vision registration 报 Y
```

（v2 里"vision 说 rigid · tactile 说 compliant"这种跨抽象层的表述不算 contradiction——一个物体可以整体刚性 + 表面柔性——已删。）

然后测四件事：(1) **Contradiction detection**——模型有没有报出"这里跨模态不一致"；(2) **Uncertainty calibration**——报出的 uncertainty 大小是否与实际偏差成正比；(3) **Source attribution**——事后能不能定位到是哪一路出问题；(4) **Recovery**——处理完之后成功率退化多少。

### 8.7 Consistency graph：从"矛盾检测"到"哪条边可疑"

$\mathcal{L}_{consistency}$ 不是 scalar、是图：

```text
           Vision
          /      \
      Tactile --- F/T
          \      /
         Proprio
```

每条 $(i, j)$ edge 是一个 consistency constraint $\mathcal{C}_{ij} = D_{ij}(z_{ij})$、$\mathcal{C}_{VT}$ 在 contact position、$\mathcal{C}_{TF}$ 在 normal force、$\mathcal{C}_{FP}$ 在 $J^T F$ 残差、$\mathcal{C}_{VP}$ 在 EE pose。总损失：

$$
\mathcal{L}_{\text{consistency}} \;=\; \sum_{(i,j)} w_{ij}\, \mathcal{C}_{ij}
$$

再进一步、把每条 edge 的 residual 记成 $r_{ij} = \mathcal{C}_{ij}$、按边排 rank：**edge residual ranking → candidate faulty modality**、或者用更原则化的 $P(\text{fault} = i \mid \{r_{ij}\}_j)$。这样系统不仅能说"有矛盾"、还能回答"**哪一条 observation graph edge 最可疑**"、自然连接到 provenance + diagnostics + sensor fault isolation。这条升级让 §6.4 那条 7-tuple 公式里的 **Provenance 与 Validity 两项变得可 benchmark**。

### 8.8 Registration perturbation（相对化、四类分开）

时间扰动相对化：$\delta t \in \{0.25\,\Delta t,\; 0.5\,\Delta t,\; \Delta t,\; 2\,\Delta t\}$、$\Delta t$ 是**该任务的 time-relevant bandwidth**（contact-mode 转移的特征时间、或 policy control period）；不写具体毫秒数字、避免 benchmark 不能迁移。

坐标系扰动分成四类：

```text
Registration robustness
├── calibration noise        ── T̂ = T · ΔT、ΔT 是小扰动（高斯 / 均匀）
├── calibration drift        ── 长时程慢漂、模拟温度 / 机械蠕变
├── sensor remounting        ── 换工具 / 换相机 / 换 sensor mounting、几何重定义
└── frame convention mismatch── base ↔ world、sensor ↔ tool、extrinsic sign flip
```

前三类是"参数错"、第四类是"约定错"、下游消费者的失败模式完全不同。这样 benchmark 比简单 domain randomization 更有说服力。

### 8.9 现有 benchmark 的适配度

RoboCasa / LIBERO / ManiSkill3 / BEHAVIOR 这一批主流 manipulation benchmark **大多以视觉为主、触觉缺席**。这一节不下结论、但**建议**：把上面八类指标做成一个可选插件、给现有 benchmark 加一层 "feedback-value + interface quality" 视角；9/11 §5.2.1 的 A/B/C/D 四臂消融可以直接借用。

## 9. 与 VLA、世界模型的关系

### 9.1 VLA：目前尚缺的不是"concat 通道"、而是跨 sensor family 的稳定 state contract

§3.1 已经修正过——把 RT-2 / OpenVLA / π0 都归为 "concat" 不准确。**以 RT-2、OpenVLA、π0 等代表性公开系统为例可以观察到**：现有通用 VLA 的规模化预训练生态主要围绕 vision-language observation 与 robot state / action 展开；tactile 与 F/T **尚未形成与 vision-language 相当的、公开、跨任务、跨 embodiment 的规模化预训练生态**。这是一个生态级判断、证据链是"具体系统 → 观察 → 本文归纳"、不是"几篇 paper 证明整个 VLA community 都如此"。

需要限定的是——这个缺席**不等于"没人做过"**：tactile foundation model 正在出现（TVL、AnyTouch）、visuo-tactile representation learning 已经很多（Lee、Calandra、3D-ViTac）、部分 robot foundation model 工作确实把 proprio 加入输入、hybrid state representation 也不是空白。真正尚缺的是——**跨 sensor family、跨 embodiment、跨 downstream consumer 的一份稳定 structured state contract**。这是一个更难被反驳、也更准确的判断。

真正的开放问题是：**VLA 的下一波扩展、是加更多 vision+language、还是补上 tactile / F/T 的 state interface？** 这一篇押后者——但不是把 tactile 当作"再一个 channel"、而是把 §6 的 structured belief interface 塞进 VLA 的输入层。Qi 等 T-Dex [arXiv:2309.09979](https://arxiv.org/abs/2309.09979)（CoRL 2023）与 Lee 等 [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) 可以视为这条路线的早期形态。

### 9.2 世界模型：如果 contact / event state 显著影响任务转移与控制、world model 通常能受益

RSSM / DreamerV3 [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) 一脉的 latent dynamics 通常假设 state transition 相对平滑、梯度可导。9/11 §5.1 讨论过、接触事件本质上是 hybrid dynamics 的 mode switch。

**这里给一个更严谨、也更"推荐"而不是"断言"的表述**：现代 world model 完全可以用 discrete latent / categorical latent / hybrid state / event-conditioned dynamics / multiple latent heads / mode-conditioned transition 来表示——因此本文**不是**在说"world model 必须直接预测 `contact[]`"。世界模型可以分别预测：

$$
z_{t+1}, \quad p(m_{t+1} \mid z_t, a_t), \quad p(C_{t+1} \mid z_t, a_t)
$$

或者用 hybrid latent（`z_continuous + z_discrete + contact state`）。本文的建议是——**对于 contact-rich task、如果 contact / event state 会显著改变任务转移与控制行为、我们认为 world model 通常能受益于保留某种显式、可辨识的 contact / event representation、而不是强迫其完全隐式地存在于一个不可解释的 continuous latent 中**。这是一个 architectural recommendation、不是事实性结论。

这条分工与 §6 的 structured belief interface 自然对接：**world model 不必消费 slot 全部字段、但通常值得让某个 head 显式预测 slot 里 mode / event 的部分**（可以是离散 latent、也可以是 $p(m_{t+1})$）；policy 层可以用 learned representation（从 slot 学 embedding 给 policy 消费是合理的、但这个 embedding 是"下游消费者"、不是"上游数据格式"）；两者之间用 §6 的接口连接、world model 预测下一时刻的 slot、policy 消费当前 slot。

### 9.3 一句话总结

**VLA 缺的不是 tactile channel 本身、而是跨 sensor family / 跨 embodiment / 跨 consumer 的稳定 state contract；world model 缺的不是 continuous latent、而是在 contact-rich 任务上对 contact / event 的显式可辨识表征。这两件事的共同根因、是 heterogeneous observation 与 downstream models 之间没有一层可以承载 belief / uncertainty / provenance / validity 的 structured interface。**

## 10. 结论

这一篇从"多模态融合常被讲成模型结构问题"这个误解出发、把讨论拉回到底层：真正的缺的不是一个更强的 fusion operator、而是一个横跨 heterogeneous observation 与 downstream models 之间、**能够显式表达 value / semantics / frame / time / uncertainty / provenance / validity 的 structured belief interface**。视觉之所以撑起一整个共同数据接口、不是因为它的 encoder 更聪明、是因为它先形成了**高度互操作的任务级 representation conventions**、并叠加了相机模型 / 标定规范 / 文件格式 / 时间轴 / GPU / 廉价传感器 / 互联网数据 / 标注生态等一整套生态条件——**它形成的是任务级 conventions、不是唯一 schema、也不是完整的 multimodal state contract**。触觉 / 力觉 / 本体感觉各自的 effective observation rate、坐标系、raw 表示、语义 convention 都还没收敛、于是 cross-attention / shared latent 的努力很容易在 §7 那五类失败模式上翻车。

本文不反对 cross-attention、不反对 shared latent、**也不反对 end-to-end learning**；本文反对的是——**让 state estimation、cross-modal composition 与 control 三件事全部隐式发生在一个不可诊断、不可复用的 latent interface 里**。给出的最小可用路线是：以 contact set 作为核心实例的 structured belief interface（**contact set 是实例、不是接口的定义**）、raw→slot 的映射放在 sensor-specific perception 里、slot 带 **point / patch / region 三类几何**、**连续 covariance 与类别 probability 分开**、**uncertainty 保留 aleatoric / epistemic / calibration 的拆分接口**、**track_id 是带 confidence 的 hypothesis、不是内禀 identity**、**unit / frame 通过 `Quantity` 类型进入 schema、可被 runtime validator 校验**、**staleness 是"相对于过去时刻的 validity"、不等于 invalidity**；接口层带 `wrench_ext` 作为 aggregate observation 与 consistency constraint；接口本身有 versioning 与 graceful degradation；policy / world model / controller / diagnostics 通过接口互操作、但不必把接口当作唯一信息通路；**Bayesian 是可选实现、不是接口的必要条件**；并把 interface swap、oracle / estimated / end-to-end 三组 baseline、cross-modal contradiction、consistency graph、modality dropout 与 degradation modes 作为一等公民写进 benchmark。

**这条路线不是终态答案、它更像 9/11 里那条"触觉缺一条可复用的中间表示"的正面回答——先约定接口、再谈架构**。回到本文的三条设计原则：**Register before compose · Expose belief at the interface · Design for disagreement**。

最后留三条可以贴到墙上的公式作为整篇文章的锚点：

$$
\boxed{\;\text{Sensor-specific observations} \;\rightarrow\; \underbrace{\text{Structured Belief}}_{\text{stable interface}} \;\rightarrow\; \{\text{Policy, World Model, Controller, Diagnostics}\}\;}
$$

$$
\boxed{\;\text{Good fusion} \;\neq\; \text{agreement}\;}
$$

$$
\boxed{\;\text{Good fusion} \;=\; \text{evidence} + \text{uncertainty} + \text{provenance} + \text{disagreement handling}\;}
$$

下一篇（9/13）打算从"接口"往下游走一步：**灵巧手与 in-hand manipulation**——把这一篇的 state interface 放到 T-Dex / DextrAH / LEAP 这一批近期工作的具体场景里、看它能不能撑起这些系统的架构、以及为什么"能装手的机器人很多、真正在做 dexterous 的少"这个反差背后的成本结构。

## Sources

本文的引用不追求"堆 sensor paper"、而是按**四条主要 thesis 判断的证据链**分组：

### A · 跨模态表示与融合范式（支撑 §3 / §4）

- Baltrusaitis, Ahuja, Morency, *Multimodal Machine Learning: A Survey and Taxonomy*, TPAMI 2019 · [arXiv:1705.09406](https://arxiv.org/abs/1705.09406)（多模态融合的经典 taxonomy · §3 谱系分层的参照）
- Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language Sequences*, ACL 2019 · [arXiv:1906.00295](https://arxiv.org/abs/1906.00295)（早期显式处理"跨模态时间不对齐"的代表 · §3.2）

### B · Visuo-tactile 融合的实证线（支撑 §4 / §7.1 / §9.1）

- Calandra et al., *More Than a Feeling: Learning to Grasp and Regrasp using Vision and Touch*, RA-L 2018 · [arXiv:1805.11085](https://arxiv.org/abs/1805.11085)（较早的 visuo-tactile regrasp 实证、"触觉在长尾场景里更值钱"的最初证据之一）
- Lee et al., *Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks*, ICRA 2019 · [arXiv:1810.10191](https://arxiv.org/abs/1810.10191)（visuotactile 自监督表示、§3.4 shared latent 路线的早期锚点）
- Huang et al., *3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing*, CoRL 2024 · [arXiv:2410.24091](https://arxiv.org/abs/2410.24091)（该论文实验报告 visuo-tactile 相对 vision-only 的显著增益 · §7.1 modality-collapse 诊断的参照）
- Qi et al., *General In-Hand Object Rotation with Vision and Touch* (T-Dex), CoRL 2023 · [arXiv:2309.09979](https://arxiv.org/abs/2309.09979)（主动触觉探索 + 视觉–触觉融合 · §9.1 VLA 补触觉的具体形态）

### C · 跨传感器 / 跨模态的统一表示（支撑 §3.4 / §5.1）

- Zhao et al., *A Touch, Vision, and Language Dataset for Multimodal Alignment*（TVL / Binding Touch to Everything）, ICML 2024 · [arXiv:2402.13232](https://arxiv.org/abs/2402.13232)（约 44K vision-touch pairs、tactile-VL 对齐、"触觉版 CLIP"路线的代表）
- Feng et al., *AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors*, 2025 · [arXiv:2502.12191](https://arxiv.org/abs/2502.12191)（跨异构 visuo-tactile sensor 统一表示 · §5.1 raw 层难统一的对照）
- Lin et al., *9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation*, ICRA 2023 · [arXiv:2308.14277](https://arxiv.org/abs/2308.14277)（多模态触觉的一个具体形态、3D shape reconstruction + 6D force 是**估计层** · §5.1 异构性证据）

### D · 鲁棒性与 modality masking / dropout（支撑 §7.5 / §8.4 / §8.5）

- Maiga et al., *MMP: Towards Robust Multi-Modal Learning with Masked Modality Prior Fine-Tuning*, 2024 · [arXiv:2410.03010](https://arxiv.org/abs/2410.03010)（把 modality masking 当训练策略、missing-modality robustness 中的常见做法之一 · §7.5 缓解方案参照）
- *Robust Multimodal Learning with Missing Modalities via Parameter Projection*, 2023 · [arXiv:2310.03986](https://arxiv.org/abs/2310.03986)（缺模态条件下的表示对齐 · §8.4 robustness 指标参照）

### E · VLA 与 World Model（支撑 §9）

- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)（VLM backbone + proprio token + noisy action chunk + flow matching · §3 与 §9.1 修正描述）
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)（开源 VLA 基线 · 公开配置含多相机 / depth / proprioceptive state encoding、并非简单 V+L concat）
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)（把 action 表达成 text token 与 VLM 联合 fine-tune · §3 修正描述）
- Hafner et al., *Mastering Diverse Control Tasks through World Models*（DreamerV3）, Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)（latent dynamics 的世界模型代表 · §9.2 "若仅用连续 latent 且无显式 mode / event variable、contact 事件更容易被平滑" 的对照面）

### F · Sim-to-Real / Domain Randomization 背景（支撑 §7.3 / §8.8）

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)（把坐标系扰动当 domain randomization 一部分的经典做法）

### G · 承接 9/11 · Contact state 与 impedance（背景）

- 9/11 那篇里已经引用过、这一篇继续沿用的：Hogan 阻抗控制三部曲、Posa-Cantu-Tedrake IJRR 2014（hybrid contact-mode trajectory optimization）、Lee 1810.10191、Qi 2309.09979、Huang 2410.24091、Zhao 2402.13232、Feng 2502.12191。这一篇不重复贴链接、需要精确出处请直接看 9/11 的 Sources 部分。

---

> **相关阅读**
>
> - [只会看、不会摸：机器人为什么缺一双"手感"的手](/zh/articles/2026-09-11-tactile-force-sensing/)——这一篇的前作、把触觉与力控单独拆开讲
> - [Sim-to-Real 方法论](/zh/articles/2026-09-10-sim-to-real-methodology/)——§7.3 坐标系扰动、§8.8 registration perturbation 都可以借用它的 domain randomization 视角
> - [机器人数据为什么比大模型数据更难](/zh/articles/2026-09-09-robot-data-scaling/)——§3 shared latent 路线的"监督信号哪里来"问题、其实是数据 scaling 问题
> - [VLA 与世界模型](/zh/articles/2026-09-07-vla-world-models/)——§9 那一节是它的一个具体侧面：VLA 尚缺跨 sensor family / embodiment / consumer 的稳定 state contract；世界模型在 contact-rich 任务上通常能受益于显式的 contact / event representation
