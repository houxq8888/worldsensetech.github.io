---
title: '拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口'
slug: "2026-09-12-multimodal-fusion-interface"
date: 2026-09-12
draft: false
categories: ["具身智能", "多模态感知"]
tags: ["具身智能", "多模态融合", "视觉-触觉", "力觉", "本体感觉", "表示接口", "多模态状态估计", "Registration", "Cross-attention", "Modality Dropout", "VLA", "世界模型", "坐标系对齐", "时间对齐", "不确定性"]
description: '多模态融合常被讲成"选哪种 attention 结构"的问题、但机器人场景真正的瓶颈在更上游：四路信号（Vision / Tactile / Force-torque / Proprioception）在时间基、坐标系基、任务语义上从来没对齐过。本文把"融合"拆成 *perception → registration → semantic projection → multimodal state estimation → structured state interface* 五级、指出视觉之所以能撑起共同数据接口、是因为它把这三个基一次性固化了下来（虽然固化的是"高度互操作的任务级约定"、不是唯一 schema）、而触觉/力觉/本体感觉各自的 effective observation rate、坐标语义、raw 层结构都还没收敛。文章不是反对 cross-attention、而是限定它的职责：attention 只能负责 composition、不能替代显式的 registration 与 state estimation。给出的最小可用路线是：以 contact set 作为 contact-rich manipulation 的核心结构化 state interface（而非"所有模态都要压缩成 contact set"）、raw→slot 的映射放在 sensor-specific perception 里、slot 层带 uncertainty / provenance / timestamp / age、policy / world model / controller 只在 state interface 上组合、并把 modality dropout 与 cross-modal contradiction 作为一等公民写进训练与 benchmark。'
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

想象一个把 RGB 相机、GelSight 指尖、腕部六维力/力矩、关节编码器全部装齐的双臂机器人。硬件清单看起来很"多模态"、但把它交给一个 policy、绝大多数工作会告诉你"用 cross-attention 融合一下"就完事了。这在 demo 里能跑通、**但在真实部署里几乎一定会遇到四类翻车**：某一路掉帧、某个传感器坏掉、坐标系漂移、模型悄悄把某一路当成主导路径而其他路变成噪声。这四类不是工程细节、**它们指向的是同一个根因：模态之间没有先约定好一个可被 policy、world model、controller 共同消费的、带时间戳 / 坐标系 / 不确定性 / 来源信息的结构化状态接口**。

这一篇聊具身智能里最容易讲虚的一块：**多模态融合**。不打算推销某个具体网络、也不打算反对 cross-attention——本文真正反对的是**让一个 policy network 同时承担 state estimation + sensor fusion + control 三件事**。文章会先把"融合"拆到最底层的三个基（时间、坐标、任务语义）上、看清楚**视觉为什么先跑通了、触觉/力觉/本体感觉还差在哪、以及如果非要现在动手、最小可用的接口长什么样**。

## 0. 全文框架：从 "fusion layer" 到 "multimodal state estimation + state interface"

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
                     │  Multimodal state estimator         │
                     │  (Bayesian fusion over $C_t$,       │
                     │   wrench_ext, robot_state, belief)  │
                     └──────────────────┬─────────────────┘
                                        ▼
                     ┌────────────────────────────────────┐
                     │  Structured state interface (API)   │
                     │  contacts[]  · wrench  · robot_state│
                     │  task_context · belief              │
                     │  + uncertainty · provenance         │
                     │  + timestamp · age                  │
                     └──────────────────┬─────────────────┘
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
                   Policy          World Model         Controller
                     │                  │                  │
                     └──────────────────┼──────────────────┘
                                        ▼
                                   Action a_t
```

这张图想强调三件事、也是全文真正想立住的架构主张：

1. **fusion 不是一个网络模块、是一整套 "state-estimation + interface contract"**。让 policy 网络顺手把 state estimation 也做了、是当前大多数多模态机器人论文最深的隐患。
2. **alignment 只是 state estimation 之前的一段**（temporal registration + spatial registration + semantic projection 三小段）、**不等于 fusion 本身**；把 semantic projection 混进 alignment、会让"配准"这个原本可以硬约束的问题被稀释成"任务语义表示学习"。
3. **接口层（Structured State API）不是 learned embedding、是一份 contract**：里面每一格都有单位、坐标系、时间戳、不确定性、来源。它优化的目标是**可互操作**、不是"对某个模型友好"。

用一句 reviewer 视角能站得住的话总结全文的中心命题：

> **Multimodal fusion should *compose already-aligned modality representations*, not *discover the alignment contract from raw observations*.**

如果 9/11 那篇的三个标签是"触觉自己的三条硬骨头"、这一篇给"多模态融合"钉三条：

> **Alignment before fusion · Interface before architecture · Missing modalities by design**

后面每一节围绕这三条展开。

## 1. "多模态"不等于"多传感器"

这个话题最容易一开始就跑偏——很多人（很多论文引言也一样）把"多装了几个传感器"直接当成"多模态"。这是不成立的。

**"modality" 本身没有一个唯一的物理学定义、它是分析视角的产物**。为了讨论方便、本文的 taxonomy 用一个四要素约定：**measurement space（信号值域）、physical origin（物理来源）、noise model（噪声结构）、update semantics（触发 / 采样 / 时钟语义）**。这四件事一起决定了一个数据流能不能被"当成同一种东西"处理。不同的作者、不同的任务、可以把边界画得略松或略紧、但只要一次性约定清楚、后面的讨论就不会飘。

按本文这套 taxonomy、有几个常被误当作"多模态"的例子：

- 腕部相机 + 头顶相机 → 是**同模态多视角**、不是多模态；measurement space、physical origin、noise model 基本一致、只有外参不同、可以走同一条 encoder。
- GelSight 指尖的 4 个小摄像头 → 是**一个触觉模态的内部结构**、不是 4 个视觉模态。它的 measurement space 是"弹性体表面形变场"、不是场景 RGB；即便中间用了 CNN、它的下游语义也是 contact geometry、不是 object detection。
- 关节编码器 + 电机电流 → 在本文的 taxonomy 里、可以把它们视为**同一 robot-state modality 下的不同 observation channels**（一个是直接位置测量、一个是间接力矩推断、共享同一个 latent 物理状态）。把它们当作两个独立模态去融合、attention 大概率只会学到冗余。

反过来、下面这几个是**在本文 taxonomy 下的真正不同模态**：

- **Vision**（RGB / RGB-D / 事件相机）——光辐射场、坐标系是相机、effective observation rate 常见 15–60 Hz。
- **Tactile**（GelSight / GelSlim / TacTip / 9DTact / 阵列 taxel）——接触界面形变或力分布、坐标系是 sensor frame；effective observation rate 依 sensor family 从 30 Hz 到 kHz 不等、相机型受帧率限制、阵列型受扫描限制、事件型只看事件密度。
- **Force/torque**（腕部 F/T、六维力/力矩）——接触界面的**整体**力旋量、坐标系是 sensor frame + tool frame、常见 500 Hz – 1 kHz。
- **Proprioception**（关节角 q、角速度 q̇、关节力矩 τ、末端位姿）——机器人自身状态、坐标系是 base / world frame；典型范围从数百 Hz 到 kHz 级、工业伺服 loop 通常更高、手臂式教学机器人则常在几百 Hz。

有些工作还会把 audio（麦克风阵列）、thermal、gas、ultrasound 加进来当第五/第六模态。它们的分类逻辑与上面一致。

**这一节的落点**：讨论"融合"之前、先把模态清单和它们在**本文 taxonomy** 下的四要素约定列清楚。这一列没做、后面所有的 fusion architecture 都是空中楼阁。9/11 §2.1 用一棵 taxonomy 树把 contact sensing 分成 Tactile / Force-torque / Proprioceptive 三支、这一篇沿用同样的三分法、把讨论限定在 **V + T + F + P** 这四个模态上。

## 2. 融合之前的三小段：Registration 与 Semantic projection

这一节是全文最技术、也最容易被略过的一节。这里先做一个重要的**术语收紧**：上一版把这三段叫 "alignment 的三个基"、reviewer 正确地指出——**"配准（registration）" 与 "语义投影（semantic projection）" 是两个不同性质的事**、把它们都塞进 alignment 会让读者以为只要做 math 变换就够了。本文后续统一改成：

```text
Pre-fusion pipeline
├── Registration        (硬约束、有闭式解或标准算法)
│   ├── Temporal registration
│   └── Spatial registration
└── Semantic projection (学习任务相关的状态表示)
    └── Raw observation → task-relevant state
```

**Registration** 是 measurement 层的问题、可标准化、可标定、可单元测试；**Semantic projection** 更接近 perception 与 state estimation、是这一篇后面 §6 state interface 的主角。两者**都是 fusion 之前**、但性质完全不同。

### 2.1 Temporal registration（时间配准）

四路信号的默认时间常数是差好几个数量级的：

```text
Vision (image-based)      ~15–60 Hz            Δt ≈ 17–67 ms
Tactile (image-based)     ~30–200 Hz           Δt ≈ 5–33 ms
Tactile (taxel / array)   ~500 Hz – kHz        Δt ≈ 1 ms
Force/torque              ~500 Hz – 1 kHz      Δt ≈ 1–2 ms
Proprioception            ~几百 Hz – kHz       Δt ≈ 0.5–2 ms
```

这里有个常被混淆的点：上面这些是 **effective observation rate**、不是 sensor 内部采样率。相机型触觉的 "kHz taxel 采样"、如果它的输出图像仍然只有 30 Hz、那融合层能拿到的有效速率就是 30 Hz。

一种常见但过于简单的 baseline、是把各模态重采样到共同的低频 policy clock（通常是视觉那一档）。**对于 contact-rich control、这会把部分高频事件压缩成不可见的 alias**——滑移检测、瞬态接触力峰、关节冲击这些"事件性"信号往往就发生在两次 vision 帧之间、降采样等于扔掉。

现实里更常见、也更稳健的架构是**多速率共存**：

```text
                    ┌── high-rate state estimator / controller
                    │       (1 kHz 或更高、由 robot 硬件 servo loop 决定)
                    │            ↑
                    │       latest proprio · latest F/T
                    │       tactile history · latest vision
                    │
   multi-rate observations
     ↑        ↑        ↑        ↑
  camera   tactile    F/T     proprio
```

融合层应该**承认时间常数差异、并按物理任务的敏感度选择对齐层**。视觉 policy 可以 30 Hz、力/触觉驱动的柔顺控制必须保持 native rate、事件信号要单独走一条 event bus。

三个必须写进接口层的时间细节：

- **时间戳语义**：sensor timestamp（曝光 / 采样瞬间）vs arrival timestamp（软件层拿到）vs host timestamp（融合那一刻）。三者差 5–20 ms、足以让"抓杯子的接触瞬间"被错位到"合指之前"。
- **时钟漂移**：相机走自己的晶振、机器人 controller 走另一套时钟、上位机走第三套；几分钟内漂几十毫秒、跑一夜就足以让融合模型学到错误的 lead/lag 相关性。NTP/PTP 只在系统层能压、传感器层还得靠硬触发。
- **事件型 vs 周期型**：触觉里的事件检测（例如 slip 触发）本质上是 sparse event、不是周期采样；把它塞进时间轴上均匀分布的 buffer、事件密度信号就丢了。

**判断标准**：**把时间常数差异写成建模假设、不要靠降采样掩盖**。这条与 9/11 §4.1 里"具体频率由 policy arch / hardware servo / controller / compute budget 决定、不写死数字"是同一个判断。

### 2.2 Spatial registration（空间配准）

四路信号天然挂在四个不同的坐标系上：

```text
Vision           camera frame (or: RGB in image px + depth in cam)
Tactile          sensor frame (贴在指尖 surface 上的 local frame)
Force/torque     sensor frame (通常装在腕部、需要变换到 tool/base)
Proprioception   base frame / world frame
```

想把它们放进同一个融合层、至少要做三件事：

- **手眼标定（eye-in-hand / eye-to-hand）**：$T^{cam}_{base}$ 是 SE(3)、任何一次碰撞都可能让它漂掉零点几度到几度——对视觉抓取也许无所谓、对精细接触操作就直接毁掉下游。
- **传感器安装标定**：指尖 sensor frame → 连杆 frame 的偏移 $T^{sensor}_{link}$；这块常常被简化成"我假设装得很正"、但 GelSight 类光学触觉的形变场对安装偏差极敏感。
- **力旋量变换与重力补偿**：F/T 传感器读到的是 sensor frame 下的 wrench、要变换到 base 或 world、并且扣掉工具重力 + 惯性项。这一层任何一步出错、下游的"法向力/切向力分解"就整个错。

**空间配准没做好的典型症状**：训练好的模型在换工具、换相机角度、或者机器人被推歪之后立刻掉点——因为模型学到的其实是"某种传感器 frame 下的相关性"、而不是任务本身。

一个更微妙的问题是**相对位姿 vs 绝对位姿**：接触物理本质只依赖"谁碰谁"的相对关系、但很多 policy 直接把末端在 world frame 下的绝对位置喂进去、于是模型学了一堆不该学的自由度。这一条与 9/11 §3.3 的 "policy 需要一致的是 contact-mode 转移、不是每一个物理参数的绝对值" 是同一种判断。

### 2.3 Semantic projection（语义投影）

这一段严格来说不是"配准"、而是"把观测映射到共同的任务语义坐标"：

```text
Vision           ──► object / scene 语义（这是什么、在哪里）
Tactile          ──► local contact 语义（谁在碰、怎么碰、有没有滑）
Force/torque     ──► aggregate contact 语义（这一瞬间整体力旋量）
Proprioception   ──► self-state 语义（我自己现在什么姿态、什么速度）
```

9/11 §2.1 里把 signal → meaning 分成九层：

```
Sensor → Calibration → Raw obs
       → Contact perception
       → Contact geometry & wrench
       → Contact mode & physical state
       → Task-relevant belief
       → Policy / controller
       → Action → New contact
```

**多模态融合的关键问题就是：到底在哪一层做 composition？** 常见错误是在 raw 层就 composition（把四路 tensor 拼起来喂进网络）——这相当于强迫 policy 自己学完 §2.1 / §2.2 / §2.3 全部、代价极高、样本效率极低。相对合理的做法是**让每一路各自走完 raw → perception → contact geometry 这几层、然后在"接触事件、力旋量、机器人状态、任务 belief"这四个语义已经收敛的位置上做 composition**、也就是本文后面 §6 的 state interface。

**这一节的落点**：temporal / spatial 两小段 registration 是 measurement 层的硬约束、semantic projection 是 perception 与 state estimation 的分工。三件事任何一件糊过去、后面的 fusion architecture 再华丽也是在补前面的锅。

## 3. 现有融合范式的谱系：不同机制解决不同问题

学界讲"多模态融合"、其实主要讲的是**在 composition 这一步怎么组织信号**。这一节按"融合发生的时机"把主流做法排一遍、顺便指出各自的失效模式。Baltrusaitis 等的经典综述 [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) 把这条谱系整理为 representation / learning / feature selection / fusion / application 五层、这一节借用其中 fusion 那一层的分类。

**先做一个重要的定位**：下面这几种范式**不是互斥的架构选择**、而是不同轴上的机制。一个成熟系统往往同时用其中几种：

| 机制 | 解决什么问题 | 属于本文 §0 哪一层 |
| --- | --- | --- |
| Modality-specific encoder | 各自 raw → 局部观测 | Perception |
| Temporal / spatial registration | 时钟、SE(3)、坐标系 | Registration |
| Cross-attention | 学出来的 composition | 主要在 state interface 之后 |
| Shared latent / VLT-style | 表示层跨模态对齐 | Semantic projection |
| Late / decision fusion | 决策层合成 | Policy / controller |
| Structured state interface | 语义 schema 与 contract | State interface |

也就是说、本文不是反对 cross-attention、而是**限定它的职责**：

> **Cross-attention cannot substitute for explicit temporal / spatial / semantic registration。**

Attention 可以把已经对齐好的表示组合起来、但不能从 raw 里"学出"配准。这一句话是本节与整篇文章的枢纽。

### 3.1 Early concat：raw / embedding 层拼接

最朴素的做法：把四路信号 encoder 之后 concat 成一个大 state vector、扔给 MLP 或 Transformer。伪代码：

```python
state_t = concat([
    enc_v(img_t),
    enc_tac(tac_t),
    enc_ft(wrench_t),
    proprio_t           # (q, q̇, τ, EE pose)
])
a_t = policy(state_t)
```

优点：实现门槛低、不需要额外的语义层设计；样本够多时、深度网络可以自己"学"出一些对齐关系。

缺点（也是几乎必然踩到的）：

- 时间常数差异被 concat 掩盖、模型很容易学到"vision 帧触发的那一刻其他信号取到什么值"这种虚假 lead/lag。
- 任何一路掉帧就要么补零、要么 hold-last-value、这两种都是分布外。
- 缺少 modality-specific prior——触觉 encoder 与视觉 encoder 结构差异很大、被强行拉平之后、梯度互相污染。
- 一旦训练时没见过的模态缺失组合出现、policy 立刻崩。

**这一节必须小心的一段是 VLA**。上一版把 RT-2 / OpenVLA / π0 都归为 "vision + proprio + language concat"、reviewer 指出这不准确：

- **RT-2** [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) 的核心是把机器人动作直接**表达成文本 token**、和 VLM 一起 fine-tune、并不是 state vector concat。
- **π0** [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) 是 VLM backbone + **proprioception token**（线性投影进 transformer embedding 空间）+ **noisy action chunk**、通过 flow matching 出 action。
- **OpenVLA** [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) 的公开配置里明确有多相机、depth、proprioceptive state encoding 等分支、也不是简单的 V+L concat。

更准确的表述是：

> **现有通用 VLA 的共同点不是"简单 concat"、而是"以 vision-language 预训练作为主要规模入口、把机器人状态作为额外 representation 注入 policy"**。三者在输入组织与 action 表示上差异很大、但 **tactile / force-torque 并没有成为与 vision 同等级的、经过规模化预训练生态支持的输入模态**。

这个缺席不是"忘了加"——是加了以后 §2 那三小段（temporal / spatial / semantic）都要重讨论、而且**触觉与 F/T 目前没有对应的预训练数据规模**。这个精确判断在 §9.1 会再展开。

### 3.2 Cross-attention / Transformer fusion

比 concat 进一步、把每一路当作一个 token（或一小段 token）序列、上层跑 self-attention / cross-attention。视觉-语言-动作那一支大部分用这个模板；Tsai 等的 Multimodal Transformer [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) 是这条路线的经典起点、显式处理"不同模态时间粒度不一致"的问题。

优点：结构上承认了模态间的对齐是**学出来的**、给了 soft alignment 的空间；能吸收不规整的时间戳。而且现代 multi-modal transformer 完全可以搭配：

- **timestamp / positional embedding** 吸收时间差
- **relative temporal encoding** 处理 lead/lag
- **modality embedding** 区分通道身份
- **frame-aware features**（把 SE(3) 变换作为输入而非"学出来"）
- **modality-specific encoder** 保留各自归纳偏置
- **modality dropout / masking** 强制不依赖单一路
- **auxiliary per-modality loss** 逼每一路自己就有信息
- **attention masks** 显式处理缺模态

所以真正的问题不是"attention 会翻车"、而是：**如果没有把上面这些当接口契约的一部分写清楚、attention 会顺手把配准也"学"掉、结果学到的是数据分布里的巧合、不是可移植的接口**。

一个成熟的多模态系统里、cross-attention 的位置应该是：**perception 与 registration 已经把输入送到 state interface 上、attention 只在 state interface 之内做 composition**。

### 3.3 Late / decision-level fusion

每一路各出一个子 policy（或子 value / 子动作提议）、决策层再加权投票、或者按 confidence gating。这条路线更接近传统 robotics：视觉给粗调、F/T 给细调、触觉给 slip recovery、proprioception 给 nominal trajectory tracking。

优点：**工程稳健**——任何一路掉线只是掉一个 proposal、不会全局崩；容易加 safety layer、也容易解释。

缺点：

- 底层耦合信息丢了。很多任务的信号本质上是跨模态的（例如"从触觉和 F/T 合起来看、这个接触是稳定滑动还是 stick-slip"）、分开提 proposal 之后在决策层很难重建。
- Weighting 规则要么手写要么学——手写不可扩展、学又回到 §3.1 的问题。
- 对"多模态共同支持某个高维 belief"这种任务不利。

**这一支在工业界的接触丰富操作里其实最常见**——但通常被叫做"controller 分层"、不叫"融合"、所以论文里少。9/11 的阻抗控制 + 视觉粗定位 + 触觉滑移检测的组合、本质就是这种 late fusion。

### 3.4 Shared latent / VLT-style alignment

用对比学习 / 蒸馏 / CLIP-style 目标、把不同模态的表示拉进同一个 latent 空间。这条路线的最新例子包括：

- TVL / Binding Touch to Everything [arXiv:2402.13232](https://arxiv.org/abs/2402.13232)（Zhao 等, ICML 2024）——把 tactile 与 vision-language 在 latent 层对齐、约 44K vision-touch pairs、通过语言标签建立跨模态 alignment、是当前"触觉 foundation model"讨论里绕不开的一个节点。
- AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191)（Feng 等, 2025）——针对 **heterogeneous visuo-tactile sensors** 的统一 static-dynamic 表示、相当于在 tactile 内部先做一次 latent 融合。
- 3D-ViTac [arXiv:2410.24091](https://arxiv.org/abs/2410.24091)（Huang 等, CoRL 2024）——visuo-tactile 联合表示学 fine-grained 插拔、摘要里明确报告了 visuo-tactile representation 相对 vision-only 的提升。
- Lee 等 Making Sense of Vision and Touch [arXiv:1810.10191](https://arxiv.org/abs/1810.10191)（ICRA 2019）——最早的 visuotactile 自监督对齐代表之一。

优点：表示层可复用、跨任务/跨 embodiment 有理论上限。

缺点：**训练监督信号哪里来？** 视觉-语言的对比学习能吃海量网络图文对、**视觉-触觉-力觉-本体感觉的四元组对齐没有天然监督源**。目前的解决路径主要有两条——(a) 用机器人遥操作日志把四路强行同采、把"时间戳接近"当 weak alignment supervision（Calandra 等 More Than a Feeling [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) 是这个思路在触觉抓取上的早期实证）；(b) 用语言/视觉作为桥梁、把触觉拉进 V-L 空间（TVL 走的就是这条）。两条都还远没到规模化阶段。

**这里要显式说明**：上一版把这条路线接着推成"shared latent 天然会需要 structured interface"、reviewer 提醒这是**本文据此提出的推论**、不是上述论文共同得出的结论。以下按更严谨的口径重写：

> 上述工作证明了**共享 latent 在特定任务族上可行**、但它们的表示是 policy-specific 或 dataset-specific 的。本文据此提出：如果想让这些表示跨 policy / 跨 world model / 跨 controller / 跨 sensor 复用、需要在 latent 之下再加一层**带单位、带坐标系、带时间戳、带不确定性**的 state interface。这一层是本文的核心主张、不是那些论文的结论。

9/11 §3.5 里把 representation 分成"接口"与"learned representation"两支——shared latent 严格说是后一支、本文推的是前一支。

### 3.5 Structured state interface：本文推荐的接口路线

最后一种：**不在 raw 层融合、不在 latent 层融合、而是在一个显式约定的中间表示上组合**。这个中间表示不是 learned embedding、是一组**语义已经收敛、且带 uncertainty / provenance / timestamp 的状态槽位**。这一条路线与 9/11 §3.4 "视觉有共同数据接口、触觉没有" 直接对应、但范围收紧到 contact-rich manipulation。

具体 schema 设计放到 §6 展开、这里先给个粗线条：

```text
state_slots = {
    contact_set      : list of contact records (§6.1 详列)
    wrench_ext       : 6D + covariance
    robot_state      : (q, q̇, τ, EE pose)
    task_context     : language token / goal embedding
    belief           : task-relevant latent / explicit
}
```

四路信号各自负责往这些槽里填内容、fusion 变成"往同一个 dict 里填 key"、policy 与 world model 都只在 dict 层读。这条路线的**优点**：

- 缺一路信号只影响它对应的 key、其他 key 不受污染；modality dropout 天然好加。
- 时间基、坐标系基、语义基都收敛在 slot 定义里；slot 一旦定下、fusion 结构可以随便换。
- 与 9/11 §5.2.1 的 feedback-value ablation 直接对应——A/B/C/D 四组实验就是往 dict 里少填一些 key。

**缺点**：slot 本身要先设计、这活比"套一个 Transformer"重；且 slot 精度不够时、下游 policy 也学不出超出 slot 的能力。**并且 slot 不解决 vision 提供的 object identity / geometry / free-space / occlusion / scene context 这类"非接触语义"**——本文推荐的是 **contact-rich manipulation 场景下的 state interface**、不是一般意义上的 multimodal interface。这个范围收紧很重要、下一节 §6 会正式讲。

**这一节落点**：融合范式的谱系里、early concat / attention / late fusion / shared latent 都各有用武之地、但**它们的失效模式绝大多数都能追溯到 §2 那三小段没做**；本文更倾向于把工程重心从"选哪种 attention"移回"先把 state interface 定下来"。

## 4. 视觉为什么"先"跑通了共同数据接口

一个自然的问题：§2 那三小段对视觉同样存在、为什么视觉就能撑起一整个共同数据接口？这一节给一个偏历史-工程化的回答、也是给触觉/力觉/本体感觉一面镜子。

**先做一个重要的修正**：**这不是单因果、是三基固化 + 生态条件共同作用**。视觉跑通除了本文强调的三个基、还叠加了 GPU/CNN scaling law、廉价 CMOS 相机、互联网规模数据、标注生态、标准化文件格式（JPEG / PNG / MP4 / HDF5）等一系列工程史条件。本文只强调三基这一维度、因为这条恰好是触觉目前最缺的、**不是要宣称"三基固化 = 视觉成功"**。

**Temporal registration 上**：视频天然是 frame-indexed、30 Hz 或 60 Hz、所有下游任务（分类 / 检测 / 分割 / SLAM）都约定俗成"以帧为单位"。这个约定看似 trivial、但它**消灭了融合讨论里最麻烦的一类问题**——所有基于图像的下游算法都用同一条时间轴。

**Spatial registration 上**：针孔相机模型 + 相机内外参这套约定、把 image pixel ↔ world point 的映射变成一条公式（$s \cdot m = K [R | t] \cdot M$）。所有 3D 视觉、SLAM、NeRF、3D Gaussian Splatting、多视角立体都在这个约定上生长。标定错误当然存在、但**"错误长什么样"是可预期、可复现的**。

**Semantic projection 上**：RGB 图像本身没有语义、但视觉社区在过去十几年用 COCO / ImageNet / ADE20K / LVIS 这一批大规模标注数据集、**形成了一组高度互操作的任务级 representation conventions**——object class / bounding box / instance mask / depth / affordance / caption。更准确地说、这一批数据集**并没有形成一个统一 semantic schema**：ImageNet 是 classification ontology、COCO 是 detection + instance + caption、ADE20K 是 scene parsing、LVIS 是 long-tail instance 分布、四者各自定义、只是彼此可组合。**"高度互操作的任务级约定" 比 "统一 schema" 更接近事实**、也更能解释为什么一个 CV 模型能迅速与另一个 CV 模型串起来。

对比触觉：

- Temporal registration：不同 sensor family 从 30 Hz 到 kHz、事件触发与轮询混合、没有共识。
- Spatial registration：GelSight 是 pixel + 弹性体形变、9DTact 是 pixel + 3D 形变场、阵列 taxel 是一维/二维力分布、光学触觉（AnySkin / DigiTact）又是另一套表示；连"一个触觉读数到底长什么 shape"都没统一。
- Semantic projection：接触点、法向、切向、slip、mode 这些概念在论文里都有人用、**但没有跨数据集统一的任务级 conventions**。这里也顺便修正上一版的一个说法——"9DTact 里叫 6D force、GelSight 里叫 shear map、阵列里叫 taxel load、都是同一个物理量不同投影"——**它们包含重叠但不同层级的物理信息**：GelSight 的 shear / deformation map 更接近原始局部形变观测、9DTact 的 6D force 是经过模型反演得到的全局 wrench estimate。两者不是同一个物理量的两种投影、是**观测层与估计层**的差异。这个区别对 state interface 的 slot schema 有直接影响（见 §6）。

**这一节落点**：视觉之所以赢、不是因为它的 encoder 更聪明、是因为它**先解决掉了 §2 那三小段、并叠加了一整套生态条件**、让所有下游 fusion 讨论都跑在同一个地基上。触觉/力觉/本体感觉要形成同规模的融合生态、先要做的是**约定接口**、而不是卷模型结构。

## 5. 每一路信号各自的融合难点

这一节按模态过一遍具体的难点、给 §6 的接口设计做铺垫。

### 5.1 Tactile：图像 / taxel / 光学 / 电容、raw 层就没统一

触觉是这一篇里最"分裂"的一路。同一类物理量（局部接触界面的形变或力分布）、目前至少有以下**表示完全不同**的实现：

```
Image-based      GelSight  · 3 个彩色 LED + 相机 → RGB 形变图
                 9DTact    · 环形彩色 LED + 相机 → 多视角图 [arXiv:2308.14277]
                 GelSlim   · 3 相机 + 弹性体散斑 → 3 路光流场
                 TacTip    · 硅胶锥 + 光纤 + 相机 → 末端位姿

Taxel array      BioTac · 阵列 pressure / temperature / EDS
                 1D/2D 电容阵列 · 局部法向力分布
                 Piezoresistive array · 局部应力图

Optical waveguide AnySkin · DigiTact · 光在弹性体内传播、边缘触发接触点

Proprioceptive-  F/T + kinematics 反推 contact（"soft tactile"）
 inferred
```

四路信号进融合层之前、至少需要**各自走完一条 encoder**——而且这些 encoder 结构差异巨大：CNN for images、MLP for taxel arrays、Spline model for waveguide、IK for proprioception-inferred。

一个常见误解是："那我做一个 tactile foundation model、统一处理所有触觉传感器不就行了？"——这正是 AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) 在做的事、方向对、但**它解决的是 learned representation 层的统一、不是 raw 层的统一、也不是 state interface 层的统一**。而且即便有 AnyTouch、它的输出仍然是 embedding、还需要一层 "embedding → contact slot" 的显式约定才能进 state interface。9/11 §3.5 把 representation 分成"接口"与"learned representation"两支、就是为了避免这一层被压在一起。

**融合难点**：

- 不同 sensor family 的时间常数、采样语义都不一样、temporal registration 得 case-by-case 处理。
- Sensor frame 定义不统一（GelSight 是相机像素、taxel array 是阵列索引、waveguide 是光路拓扑）。
- 输出语义不统一（有的给图像、有的给力分布、有的给 3D 形变场）——注意这一条不是"同一物理量的不同投影"、而是**观测层与估计层混在一起**。

**缓解**：在 raw 到 slot 之间引入一层 **sensor-specific decoder**、把异构触觉输出统一解码到 §6.1 的 slot schema；这一层可以是解析、也可以是学习的、但它必须是**接口的一部分**、而不是藏在 policy 里。

### 5.2 Force/torque：看起来最"标准"、其实坑最多

F/T 传感器输出一个 6D wrench、格式统一（相对 tactile 好得多）、但它的**语义**要复杂很多：

```
raw_wrench_sensor = contact_wrench + gravity_wrench
                  + inertial_wrench + friction_wrench + bias
```

要提取"外部接触力"、至少要做：

- **零漂补偿**：bias 项、传感器自身温度与老化都会漂；一般用 tare 归零、但真机上很难完美。
- **重力补偿**：$\tau_g = g(q)$ 里包含工具 + 夹爪质量分布；工具一换、整条曲线全变；这个补偿在 base frame / sensor frame 下写法不一样、容易搞错。
- **惯性补偿**：$\tau_i = M(q)\ddot{q} + C(q,\dot{q})\dot{q}$；高速操作时这一项不可忽略、慢速时可以。
- **坐标系变换**：把 sensor frame 下的 wrench 变到 base 或 world；每一步都要 SE(3) 伴随矩阵。

F/T 还有一个更本质的限制：**它给出的是"整体"力旋量、不是"局部"接触**。这个 "aggregate vs field-level" 的区别正是 9/11 §2.1 taxonomy 树里 tactile 与 force/torque 被分成两支的原因。这一条对 state interface 设计有非常直接的影响、放到 §6.2 展开。

**融合难点**：F/T 的语义"干净"（6 个数）但**依赖大量前置补偿**、而补偿的精度直接依赖工具模型 + 动力学模型。工具一换、下游融合层看到的分布就变了。

**缓解**：F/T 在 slot 层应该输出**两件事**——(a) 已补偿的 external wrench（用于 force-aware policy）、(b) 残差 magnitude（用于检测"是不是又漂了 / 是不是又撞到不该撞的东西"）。第二件事常常被忽略。

### 5.3 Proprioception：最容易被当成"背景信号"、其实是最重要的一路

四路里 proprioception 是默认时间常数最快的、也是**唯一机器人天生自带完整通道**的一路。但它有一个尴尬的定位：**很多 policy 网络把 (q, q̇, τ) 直接拼进 state、不当它是"一个模态"**、于是讨论多模态融合时它常常缺席。这是错的。

Proprioception 在融合里有两个作用：

- **作为 contact hypothesis 的输入**：给定期望末端轨迹 + 关节力矩反馈、可以通过 $J^T \hat{F}_{ext} = \tau_{residual}$ 反推外部接触力（所谓 contact inference、Hogan 早年阻抗控制一脉的经典工具）。这条反推路径让 proprio 也能"贡献"到 contact slot 里、而不只是自己那一路。
- **作为时间基与坐标系的锚**：融合层需要一个稳定的 frame、proprio 天然跑在硬件 servo 允许的速率上限（几百 Hz 到 kHz 级）、是四个模态里最像 "时钟" 的一路；EE pose 也可以当作所有其他坐标系（camera / sensor）的 anchor。

**融合难点**：proprio 的信息量太"干净"——它是机器人自己测自己的状态、没有外部世界的不确定性。模型很容易**过度依赖 proprio**、把它当作 shortcut、结果在没有接触的场景表现好、有接触的场景反而没学会用触觉/F/T。这一条正是 9/11 §5.2.1 feedback-value ablation 想避免的——**评估要看"在同样有 proprio 的条件下、加入触觉/F/T 到底带来多少增量"**、不是"只用触觉 vs 只用视觉"。

### 5.4 时间常数差异本身就是建模假设

把 §5.1–5.3 与 §2.1 合起来看、可以提炼出这条判断：

> **不同模态的 effective observation rate 差异、不是"融合层要不要处理一下"的工程问题、而是"任务需要什么控制带宽"的建模假设。**

举两个例子：

- **擦拭 / 打磨**：力控带宽至少 100–500 Hz、视觉 30 Hz 完全够用、触觉与 F/T 得走 native rate、proprio 也得到力矩层。这一类任务、融合层不能把所有信号降到 30 Hz。
- **抓取-放置 / 装配**：视觉主导、接触事件稀疏；把触觉降采样到 vision rate 是可接受的近似。

具体数值随硬件、任务、controller 架构变化、这里不写死。这条与 9/11 §4.1 是同一个判断。

## 6. 一个最小可用的 state interface

到这里可以正面回答"到底怎么办"了。这一节给一个**可实施、可 benchmark、可增量演进**的最小接口。

**先做一个重要的范围收紧**：本节要提的不是 "所有机器人多模态信息都应该压缩成 contact set"、而是 **在 contact-rich manipulation 这一类任务上、contact set 应该成为跨模态组合的核心 state interface**。Vision 的 object identity / geometry / free-space / occlusion / scene context 不会全部塞进 contact set、它们各自有别的槽位（例如 task_context、world_model_state）。本文推荐的完整 slot 结构在 §6.1 明确列。

### 6.1 以 contact set 为核心 · 但不止 contact set

先定义跨模态共享的**接触事件记录**（升级版、带 uncertainty / provenance / timestamp）：

$$
C_t = \big\{\, \big(\, \mathrm{id}_i,\; p_i,\; n_i,\; f_i^{\perp},\; f_i^{\parallel},\; \phi_i^{\text{slip}},\; m_i,\; \Sigma_i,\; s_i,\; t_i^{\text{obs}},\; a_i \,\big) \,\big\}_{i=1}^{N_t}
$$

其中每一个字段：

```python
contact = {
    # 身份与几何
    "id":                int,           # tracking ID、跨帧一致
    "p":                 Vector3,       # base frame 下的接触位置
    "n":                 Vector3,       # 单位法向
    # 物理量
    "f_perp":            float,         # 法向力
    "f_parallel":        Vector2,       # 切向力（在切平面内）
    "slip_probability":  float,         # φ ∈ [0, 1]
    "mode":              enum,          # free / touch / sticking / sliding / rolling / separating
    # 不确定性（本文强调必须显式带）
    "covariance":        Matrix,        # Σ、位置/力/mode 的联合协方差
    "confidence":        float,         # 综合置信度、可选
    # 来源（provenance）
    "source":            enum,          # tactile / FT / proprio / vision / fused
    "evidence_mask":     [V, T, F, P],  # 哪些通道贡献了这一格
    # 时间
    "timestamp":         float,         # 观测时刻、host clock
    "age":               float,         # 距当前时刻的时长
    "valid_from":        float,         # 若为区间量（例如 vision 预测）
    "valid_until":       float,
}
```

这个定义刻意做到四件事：

- **sensor-agnostic**：不管下游用的是 GelSight 还是 taxel 阵列还是 F/T、只要能填这些 key、就能进 state interface。
- **物理可解释 + 单位明确**：每一个数字有明确物理量纲与坐标系。
- **带 uncertainty**：给下游 Bayesian fusion、safety layer、fallback 提供可消费的分布信息；**没有 uncertainty 的 slot 不是接口、是"事实"、真机上永远不成立**。
- **带 provenance**：这一格是谁贡献的、缺哪一路时怎么退化——这一条与 §7.5 modality dropout、§8.6 cross-modal contradiction 直接对应。

然后 **contact set 之外**、state interface 至少还要包括：

```text
wrench_ext          : 6D + covariance              # F/T 独立产出、aggregate 层
robot_state         : (q, q̇, τ, EE pose)          # proprioception
task_context        : language token / goal embed  # 高层指令、object identity 塞在这里
scene_state         : free-space / occlusion / …   # vision 的非接触语义、可选
belief              : task-relevant latent / explicit
```

**contact set 是这套接口的核心结构化对象、不是它的唯一内容**——这一句是全文最重要的措辞收紧。

### 6.2 每一路信号怎么映射到 slot · 关键修正：F/T 是 constraint、不是 detector

**先把 F/T → contact set 的映射讲准确**：上一版说 "F/T 分解到已有 $C_t$ 上、若无 tactile 可合成一个整体接触条目"——**这容易被误解成 6D wrench 足以恢复 contact geometry**。实际上：

$$
w = \sum_{i=1}^{N} \begin{bmatrix} f_i \\ (p_i - p_0) \times f_i \end{bmatrix}
$$

多个 contact point 的 $\{p_i, f_i\}$ 对**同一个** resultant wrench $w$ 可以有**无穷多解**。这是一个典型的 **inverse problem、不是 deterministic decoder**。

正确的 slot 语义应该是——**每一路提供的是 likelihood / constraint / proposal 中的一类、而不是"自己填好一个事实"**：

```text
Tactile         → local contact observations  （直接观测、有 sensor noise）
F/T             → global wrench constraint     （aggregate 层约束、不定位、不分解）
Proprioception  → kinematic / dynamic constraint  （J^T 反推、依赖模型精度）
Vision          → geometric prior              （预测 contact hypothesis、非观测）
```

四路合并变成一个 Bayesian 问题：

$$
p(C_t \mid V, T, F, P) \;\propto\; p(V \mid C_t)\, p(T \mid C_t)\, p(F \mid C_t)\, p(P \mid C_t)\, p(C_t)
$$

其中：

- $p(T \mid C_t)$ 是**局部观测似然**（tactile sensor model）
- $p(F \mid C_t)$ 是**整体 wrench 一致性**似然（wrench reconstruction）
- $p(P \mid C_t)$ 是**动力学一致性**似然（$\tau_{residual} = J^T F$ 残差）
- $p(V \mid C_t)$ 是**几何 / 视觉预测**似然（affordance、pose estimation）
- $p(C_t)$ 是先验（例如 contact 应贴着物体表面）

这个公式给 §6.3 的 evidence ranking 提供了正确框架：**evidence hierarchy 是 hypothesis-dependent、不存在全局排序**。

**Reviewer 视角的一个具体后果**：state interface 存的**不该只是"事实"、应允许存 belief + uncertainty**。$C_t$ 里每一个 record 的 covariance 就是这一层的量化表达。

### 6.3 Evidence ranking：hypothesis-dependent、不是固定排序

上一版给了一个固定排序 "tactile > F/T > proprio > vision"、这是错的。正确的表述是：

> **Evidence ranking should be hypothesis-dependent.**

不同的"待估计对象"、四路的直接性排序不同：

```text
Contact location:    Tactile > Vision > F/T ≈ Proprio
Global wrench:       F/T > Tactile > Proprio > Vision
Object pose:         Vision > Tactile > F/T ≈ Proprio
Joint state / τ_res: Proprio >> others
Contact mode:        Tactile > F/T ≈ Proprio > Vision
Slip probability:    Tactile > F/T (rate) > Proprio (residual) > Vision
```

**同一个 $C_t$ record 里不同字段的"主导模态"可以完全不同**——这也是为什么接口要带 `evidence_mask`。这个升级让本文的接口从"规则系统"变成"概率证据融合"、直接对应 §6.2 的 Bayesian 表达。

### 6.4 Interface ≠ learned latent representation（新增子节）

这一小节是全文架构主张最核心的一段。

上一版说了 "structured slots" 与 "shared latent" 是两支、但没有把差别讲透。这一版正面讲清：

```text
Latent representation             State interface
─────────────────────────         ─────────────────────────
优化目标：对下游模型友好            优化目标：对多个消费者互操作
单位：无                             单位：每一个字段有明确 SI / 局部量纲
坐标系：无（隐含在数据流里）         坐标系：显式声明、字段附 frame_id
时间：无                             时间：显式 timestamp + age
不确定性：无（或仅在训练损失里体现） 不确定性：字段级 covariance / confidence
来源：无                             来源：source + evidence_mask
消费者：一个 model                   消费者：policy / world model / controller /
                                       diagnostic tool / safety layer / 另一个 sensor
```

一句可以直接留下来的公式：

$$
\text{Interface} \;=\; \text{Semantics} + \text{Units} + \text{Frame} + \text{Time} + \text{Uncertainty} + \text{Provenance}
$$

**一个 latent 可以非常适合 policy、却不适合作为 world model、controller、diagnostic tool 或另一个 sensor 的输入接口**——这是接口存在的必要性、也是 §3.4 那些 shared latent 工作真正缺的那一环。CLIP / ImageNet / COCO 之所以撑起视觉生态、不是因为它们的 embedding 最好、是因为**它们把语义、单位、坐标、时间这些约定做成了所有下游消费者都能读的 contract**。触觉、力觉、本体感觉现在需要的就是这样一份 contract、**不是更大的模型**。

### 6.5 融合的时机：不在 raw、不在决策、**在 state interface 上**

有了 §6.1 的 slot schema 之后、fusion 变成三步：

```python
# Step 1: per-modality perception + registration
#         时间 / 空间 / 语义三小段各自收敛
raw_v  → enc_v → pred_contact_hypothesis      （视觉的 contact 预测、附 confidence）
raw_t  → enc_t → detected_contact             （tactile 观测、附 covariance）
raw_ft → enc_ft → external_wrench + residual  （补偿后 wrench、附 covariance）
raw_p  → proprio_state → J^T · τ_res → inferred_contact  （附模型误差）

# Step 2: multimodal state estimation → 写到 state interface
#         §6.2 的 Bayesian fusion、§6.3 hypothesis-dependent ranking
C_t         = state_estimator(V, T, F, P)   # 输出的是 belief、不是 hard fact
wrench_ext  = compensate_ft(raw_ft)
robot_state = (q, q̇, τ, EE_pose)
belief_t    = belief_update(C_t, wrench_ext, robot_state, task_ctx)

# Step 3: policy / world model / controller 消费 state interface
a_t = policy(state_interface)
```

关键的判断是——**Step 2 的 state interface schema 才是整个多模态系统的接口**、Step 1 里各模态可以任意换 encoder、Step 3 里 policy 可以任意换结构、只要 Step 2 的 schema 稳定。

### 6.6 缺模态时的 fallback：把"少一路"当成训练分布的一部分

真实部署里、传感器坏掉、掉帧、超时是常态。接口层要显式支持"某一路暂时没数据"：

```python
if tactile_missing:
    # §6.2 里 p(T | C_t) 换成 uniform likelihood
    C_t = state_estimator(V, F, P)         # 无 T
    for r in C_t:
        r.covariance *= inflation_T_missing   # covariance 拉大
        r.evidence_mask.T = False
        r.provenance  = "no-tactile"
```

**这不是工程补丁、这是训练时的核心约束**。见 §7.5 modality dropout、以及 §8.6 cross-modal contradiction test。

## 7. 融合失败模式与诊断

reviewer 视角、下面这五类失败在机器人多模态工作里最常见、也最容易被"我们用了 cross-attention 所以鲁棒"这类话糊过去。每一类给出**症状 / 诊断指标 / 缓解**。

### 7.1 Modality collapse：attention 权重悄悄退回视觉主导

**症状**：训练 loss 一切正常、validation success rate 正常；一旦在评测里遮住视觉、policy 表现几乎不变——说明其他模态其实没被"用起来"。

**诊断**：可视化 attention 权重按模态聚合的时间曲线；或者更狠、直接跑 **modality ablation gain**（§8.1）——$\Delta_{\text{modality}}$ 若接近零、说明这一路在 policy 里其实是死的。3D-ViTac [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) 的 ablation 之所以重要、就是给这个指标提供了一个明确的参照。

**缓解**：(a) 训练时显式跑 modality dropout、见 §7.5；(b) 给每个模态加 auxiliary supervision（例如 tactile encoder 除了给 policy 用、还得独立预测 slip 事件）；(c) 用 9/11 §5.2.1 的 feedback-value benchmark 而不是纯 success rate 逼出真实贡献。

### 7.2 Temporal smearing：所有信号被强行插值到 30 Hz

**症状**：滑移检测、瞬态接触、冲击类任务的 policy 学不出来——因为高频物理细节已经被 §2.1 讨论过的降采样抹掉了。

**诊断**：在评测里对比"以 30 Hz 融合 vs 以 native rate 融合"的成功率差；或者用 $\Delta_{\text{tail}}$ 只看困难接触条件下的表现。

**缓解**：见 §5.4，承认时间常数差异是**建模假设**；不同层用不同频率、事件走 event bus；不要在接口层把高频信号降采样掉。

### 7.3 Frame confusion：模型学到坐标系伪相关

**症状**：模型在训练位姿、训练相机摆放、训练工具型号下表现优秀、稍微动一下就崩。

**诊断**：评测集主动做**坐标系扰动**——把相机外参旋转几度、把工具重心偏移几克、把 base frame 平移几厘米、看成功率掉多少。

**缓解**：state interface 里显式带 `frame_id`、下游消费者必须做变换才能读；训练时把坐标系扰动当 domain randomization 的一部分（Tobin 等 [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)）。

### 7.4 Semantic leakage：raw 图像偷偷污染 slot 语义

**症状**：slot 明明定义成"接触几何"、但 policy 表现严重依赖视觉外观（同一个物体的照片换背景就掉点）。

**诊断**：把 slot 里的视觉贡献换成一个"最小充分"的合成 slot（例如把预测 contact 用真值 contact 替）、看性能变化。

**缓解**：slot 层用**独立 encoder** 输出到 slot 的**唯一通路**；不允许 raw pixel 直接进 policy。这条与 9/11 §3.5 "representation interface vs learned representation" 一致。

### 7.5 Missing / degraded modalities：一坏就崩、或者悄悄退化没报警

**症状**：训练时永远所有模态齐全、上线时触觉掉帧一次整个 policy 行为剧烈变化。

**诊断**：跑 **modality masking / dropout test**——评测时按 (1-p) 概率随机丢掉某一路、看成功率-曲线。modality masking / dropout 已经成为 missing-modality robustness 中**常见的一类训练策略**、可参 Maiga 等 MMP [arXiv:2410.03010](https://arxiv.org/abs/2410.03010) 与相关文献。

**但 dropout ≠ sensor failure**。真实部署里的模态退化不是 Bernoulli(p) 独立丢、更常见的是：

```text
Modality missing     ── 传感器拔了
Modality stale       ── 数值仍在、但对应的是几十毫秒之前的观测
Modality delayed     ── 时间戳正确、到达时刻滞后
Modality corrupted   ── 单帧异常值（例如相机糊了、触觉接触区外杂散信号）
Modality biased      ── 系统性漂移（F/T tare 失效、IMU 温漂）
Modality noisy       ── 高斯或非高斯噪声显著上升
```

接口层要能表达这六种状态、不能只有"有 / 无"两个 flag。§6.1 的 `timestamp / age / covariance / provenance` 就是为了让 stale / delayed / biased 在数据格式上就是可检测的。

## 8. Benchmark 与评估

这一节把 §7 里散在各处的诊断指标整合成一个可执行的 benchmark 骨架。整体思路直接沿用 9/11 §5.2.1 的 feedback-value 主张——**benchmark 应该测"这个模态真的提供了多少不可替代的信号"、不是"这个模型能不能记住训练分布"**。

### 8.1 Modality ablation gain

$$
\Delta_{\text{mod}} = S(\text{base} + \text{mod}) - S(\text{base})
$$

其中 $S$ 是任务成功率、base 是一个不含该模态的固定 policy。$\Delta_{\text{mod}}$ 要按任务族分别报——"稳定抓取"这一类里 $\Delta_{\text{tactile}}$ 小是合理的、"滑移恢复 / 精细插拔 / 擦拭"这一类里 $\Delta_{\text{tactile}}$ 就应该显著。

### 8.2 Missing-modality robustness

$$
R_{\text{rob}}(p) = \mathbb{E}_{\mathcal{D}}\big[\, S \,\big|\, \text{each modality dropped with prob } p \,\big]
$$

扫 $p \in \{0, 0.1, 0.25, 0.5\}$、画出**每个模态被单独 dropout** 的成功率退化曲线。**一条健康的多模态系统、这些曲线应该是相对平坦的**。

### 8.3 Degradation modes beyond dropout

对应 §7.5、把测试从 Bernoulli dropout 扩到：

```text
missing      ── 完全拔掉
stale        ── 保留数值、时间戳不刷新
delayed      ── 时间戳正确、arrival 晚 20/50/100 ms
corrupted    ── 单帧 outlier injection
biased       ── 加固定 offset 或 slow drift
noisy        ── σ 逐步加大
```

对每一类退化分别报成功率-曲线。**这才是"接口有没有真的处理 uncertainty / provenance"的直接检验**。

### 8.4 Temporal & frame perturbation

给视觉加 ± 5 ms 时间抖动、给触觉加 ± 20 ms、给 F/T 加 ± 5 ms、观察 $\Delta_{\text{tail}}$（只看困难接触条件下的成功率）。坐标系扰动同理。这一栏是 §7.2 / 7.3 的直接量化。

### 8.5 Slot 层的 fidelity 单独测

既然本文主张 state interface 是核心、**slot 本身的准确度就应该独立于 policy 测**：

- contact position 与真值（在仿真里）的 IoU / distance error；
- slip detection 的 AUROC；
- contact mode 分类的 macro-F1；
- **covariance calibration**（例如 reliability diagram：预测的 Σ 是否与实际误差分布相符）。

这四个是"融合层之前的质量指标"、它们不合格、policy 层的成功率高只可能是过拟合。

### 8.6 Cross-modal contradiction test（新增、本文最推荐的 benchmark 维度）

这一节是全文最值得强调的一条 benchmark。既然 interface 存在的意义就是**当不同模态说不同话时、系统能不能"正确地不确定"而不是"平均成一个错误结果"**——那 benchmark 就应该主动制造跨模态矛盾。

$$
\mathcal{L}_{consistency} = d\big(\hat{C}^{T}, \hat{C}^{F}, \hat{C}^{V}, \hat{C}^{P}\big)
$$

具体做三类主动矛盾注入：

```text
Spatial disagreement   ── 视觉说 contact 在 A、tactile 说在 B（相差 5/10/20 mm）
Temporal disagreement  ── F/T 已见 wrench 下降、tactile 仍在报 sliding
Semantic disagreement  ── vision 分类为"刚性物体"、tactile 局部形变符合"柔性物体"
```

然后测四件事：

1. **Contradiction detection**：模型有没有报出"这里跨模态不一致"
2. **Uncertainty calibration**：报出的 uncertainty 大小是否与实际偏差成正比
3. **Source attribution**：事后能不能定位到是哪一路出问题
4. **Recovery**：处理完之后成功率退化多少

这个指标与本文 interface 主张的契合度**比 modality dropout 更高**——因为它直接检验 §6.4 那条公式 Interface = Semantics + Units + Frame + Time + Uncertainty + Provenance 里的 Uncertainty 与 Provenance 两项。**建议本文推荐的 state interface 上、§8.6 而不是 §8.2 才是一等公民**。

### 8.7 现有 benchmark 的适配度

RoboCasa / LIBERO / ManiSkill3 / BEHAVIOR 这一批主流 manipulation benchmark **大多以视觉为主、触觉缺席**。这一节不下结论、但**建议**：把上面六类指标（ablation / dropout / degradation modes / temporal+frame perturbation / slot fidelity / contradiction）做成一个可选插件、给现有 benchmark 加一层 "feedback-value 视角"；9/11 §5.2.1 的 A/B/C/D 四臂消融可以直接借用。

## 9. 与 VLA、世界模型的关系

### 9.1 VLA 现状：缺席的是"与 vision 同级的预训练生态"、不是"concat 通道"

§3.1 已经修正过——把 RT-2 / OpenVLA / π0 都归为 "vision + proprio + language concat" 是不准确的。三者输入结构各自不同、π0 明确有 proprio token 与 action chunk、OpenVLA 有多相机与 state encoding、RT-2 把 action 表达成文本 token。

更准确的判断是：

> **现有通用 VLA 的规模化预训练生态主要围绕 vision-language observation 与 robot state / action 展开、tactile / F/T 尚未形成同等规模、同等标准化的预训练生态。**

这个缺席不是"忘了加"、是**数据与接口都还没到位**——触觉与 F/T 目前既没有 ImageNet 级别的公共大规模标注集、也没有 §6.1 这种 state interface 的标准 schema。真正的开放问题是：**VLA 的下一波扩展、是加更多 vision+language、还是补上 tactile / F/T state interface？** 这一篇押后者——但不是把 tactile 当作"再一个 channel"、而是把 §6 的 state interface 塞进 VLA 的输入层。Qi 等 T-Dex [arXiv:2309.09979](https://arxiv.org/abs/2309.09979)（CoRL 2023）与 Lee 等 [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) 可以视为这条路线的早期形态。

### 9.2 世界模型：不要把接触事件"平滑"进单一连续 latent

RSSM / DreamerV3 [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) 一脉的 latent dynamics 通常假设 state transition 相对平滑、梯度可导。9/11 §5.1 讨论过、接触事件本质上是 hybrid dynamics 的 mode switch。

**这里给一个更严谨的表述**（上一版说"RSSM 会把 mode switch 学成连续变化"过于绝对、reviewer 指出：现代 world model 完全可以用 discrete latent / categorical latent / hybrid state / event-conditioned dynamics / multiple latent heads / mode-conditioned transition 来表示）：

> **如果一个 world model 仅使用单一连续 latent dynamics 来表示接触状态、且没有显式的 mode / event variable、那么 contact-induced discontinuities 就更容易被"平滑地"表示成 latent 空间的连续变化。**

这是一个条件性陈述、不是"RSSM 一定如何"。规避方案就是本文 §6 的 state interface：

- **World model 层保留结构化 contact slot**：mode 是离散、$C_t$ 是集合、$f^{\perp}/f^{\parallel}$ 是实数；这三层不要塞进一个连续 latent。
- **Policy 层可以用 learned representation**：从 slot 学一个 embedding 给 policy 消费是合理的、但这个 embedding 是"下游消费者"、不是"上游数据格式"。
- **两者之间用 §6 的接口连接**、world model 预测下一时刻的 slot、policy 消费当前 slot。

这条分工与 9/11 §5.1 belief update 是同一个思路、只是把它从触觉内部推广到跨模态。

### 9.3 一句话总结

**VLA 缺 tactile state interface、world model 缺 contact structure；两件事其实是同一件事——大家都没在接口层做过事情、都寄希望于"多堆点数据、模型自己会学出来"。**

## 10. 结论

这一篇从"多模态融合常被讲成模型结构问题"这个误解出发、把讨论拉回到底层：融合真正的难点不是**选哪种 attention**、而是**能不能有一个可被 policy / world model / controller 共同消费的、带时间戳 / 坐标系 / 不确定性 / 来源信息的结构化状态接口**。视觉之所以撑起一整个共同数据接口、不是因为它的 encoder 更聪明、是因为它先解决掉了 §2 那三小段 registration + semantic projection、并叠加了 GPU / 廉价传感器 / 互联网数据 / 标注生态等一整套生态条件、**它形成的是"高度互操作的任务级 representation conventions"、不是唯一 schema**。触觉 / 力觉 / 本体感觉各自的 effective observation rate、坐标系、raw 表示都还没收敛、于是所有 cross-attention / shared latent 的努力都会在 §7 那五类失败模式上翻车。

本文不是反对 cross-attention、也不是反对 shared latent。本文反对的是**让一个 policy network 同时承担 state estimation + sensor fusion + control 三件事**。给出的最小可用路线是：以 contact set 为核心的 structured state interface、raw→slot 的映射放在 sensor-specific perception 里、state interface 显式带 uncertainty / provenance / timestamp / age、policy 与 world model 都只在 state interface 层组合、并把 modality masking / dropout 与 cross-modal contradiction test 作为一等公民写进训练与 benchmark。

**这条路线不是终态答案、它更像 9/11 里那条"触觉缺一条可复用的中间表示"的正面回答——先约定接口、再谈架构**。如果 9/11 那篇的三个标签是 *Action-conditioned observation · Contact-state representation · Closed-loop value*、这一篇的三个标签是：

> **Alignment before fusion · Interface before architecture · Missing modalities by design**

三句话把它们串起来：**融合之前先做 registration 与 semantic projection、架构之前先把 state interface 定下来、训练与 benchmark 里把缺模态与跨模态矛盾当分布的一部分**。

下一篇（9/13）打算从"接口"往下游走一步：**灵巧手与 in-hand manipulation**——把这一篇的 state interface 放到 T-Dex / DextrAH / LEAP 这一批近期工作的具体场景里、看它能不能撑起这些系统的架构、以及为什么"能装手的机器人很多、真正在做 dexterous 的少"这个反差背后的成本结构。

## Sources

本文的引用不追求"堆 sensor paper"、而是按**四条主要 thesis 判断的证据链**分组：

### A · 跨模态表示与融合范式（支撑 §3 / §4）

- Baltrusaitis, Ahuja, Morency, *Multimodal Machine Learning: A Survey and Taxonomy*, TPAMI 2019 · [arXiv:1705.09406](https://arxiv.org/abs/1705.09406)（多模态融合的经典 taxonomy · §3 谱系分层的参照）
- Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language Sequences*, ACL 2019 · [arXiv:1906.00295](https://arxiv.org/abs/1906.00295)（早期显式处理"跨模态时间不对齐"的代表 · §3.2）

### B · Visuo-tactile 融合的实证线（支撑 §4 / §7.1 / §9.1）

- Calandra et al., *More Than a Feeling: Learning to Grasp and Regrasp using Vision and Touch*, RA-L 2018 · [arXiv:1805.11085](https://arxiv.org/abs/1805.11085)（最早的 visuo-tactile regrasp 实证之一、"触觉在长尾场景里更值钱"的最初证据）
- Lee et al., *Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks*, ICRA 2019 · [arXiv:1810.10191](https://arxiv.org/abs/1810.10191)（visuotactile 自监督表示、§3.4 shared latent 路线的早期锚点）
- Huang et al., *3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing*, CoRL 2024 · [arXiv:2410.24091](https://arxiv.org/abs/2410.24091)（明确的 "vision + tactile > vision-only" 实验证据 · §7.1 modality-collapse 诊断的参照）
- Qi et al., *General In-Hand Object Rotation with Vision and Touch* (T-Dex), CoRL 2023 · [arXiv:2309.09979](https://arxiv.org/abs/2309.09979)（主动触觉探索 + 视觉–触觉融合 · §9.1 VLA 补触觉的具体形态）

### C · 跨传感器 / 跨模态的统一表示（支撑 §3.4 / §5.1）

- Zhao et al., *A Touch, Vision, and Language Dataset for Multimodal Alignment*（TVL / Binding Touch to Everything）, ICML 2024 · [arXiv:2402.13232](https://arxiv.org/abs/2402.13232)（约 44K vision-touch pairs、tactile-VL 对齐、"触觉版 CLIP"路线的代表）
- Feng et al., *AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors*, 2025 · [arXiv:2502.12191](https://arxiv.org/abs/2502.12191)（跨异构 visuo-tactile sensor 统一表示 · §5.1 raw 层难统一的对照）
- Lin et al., *9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation*, ICRA 2023 · [arXiv:2308.14277](https://arxiv.org/abs/2308.14277)（多模态触觉的一个具体形态、3D shape reconstruction + 6D force 是**估计层** · §5.1 异构性证据）

### D · 鲁棒性与 modality masking / dropout（支撑 §7.5 / §8.2 / §8.3）

- Maiga et al., *MMP: Towards Robust Multi-Modal Learning with Masked Modality Prior Fine-Tuning*, 2024 · [arXiv:2410.03010](https://arxiv.org/abs/2410.03010)（把 modality masking 当训练策略、missing-modality robustness 中的常见做法之一 · §7.5 缓解方案参照）
- *Robust Multimodal Learning with Missing Modalities via Parameter Projection*, 2023 · [arXiv:2310.03986](https://arxiv.org/abs/2310.03986)（缺模态条件下的表示对齐 · §8.2 robustness 指标参照）

### E · VLA 与 World Model（支撑 §9）

- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)（VLM backbone + proprio token + noisy action chunk + flow matching · §3.1 与 §9.1 修正描述）
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)（开源 VLA 基线 · 公开配置含多相机 / depth / proprioceptive state encoding、并非简单 V+L concat）
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)（把 action 表达成 text token 与 VLM 联合 fine-tune · §3.1 修正描述）
- Hafner et al., *Mastering Diverse Control Tasks through World Models*（DreamerV3）, Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)（latent dynamics 的世界模型代表 · §9.2 "若仅用连续 latent 且无显式 mode / event variable、contact 事件更容易被平滑" 的对照面）

### F · Sim-to-Real / Domain Randomization 背景（支撑 §7.3）

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)（把坐标系扰动当 domain randomization 一部分的经典做法）

### G · 承接 9/11 · Contact state 与 impedance（背景）

- 9/11 那篇里已经引用过、这一篇继续沿用的：Hogan 阻抗控制三部曲、Posa-Cantu-Tedrake IJRR 2014（hybrid contact-mode trajectory optimization）、Lee 1810.10191、Qi 2309.09979、Huang 2410.24091、Zhao 2402.13232、Feng 2502.12191。这一篇不重复贴链接、需要精确出处请直接看 9/11 的 Sources 部分。

---

> **相关阅读**
>
> - [只会看、不会摸：机器人为什么缺一双"手感"的手](/zh/articles/2026-09-11-tactile-force-sensing/)——这一篇的前作、把触觉与力控单独拆开讲
> - [Sim-to-Real 方法论](/zh/articles/2026-09-10-sim-to-real-methodology/)——§7.3 坐标系扰动、§8.4 temporal perturbation 都可以借用它的 domain randomization 视角
> - [机器人数据为什么比大模型数据更难](/zh/articles/2026-09-09-robot-data-scaling/)——§3.4 shared latent 路线的"监督信号哪里来"问题、其实是数据 scaling 问题
> - [VLA 与世界模型](/zh/articles/2026-09-07-vla-world-models/)——§9 那一节是它的一个具体侧面：VLA 缺席 tactile 的规模化预训练生态、世界模型缺席显式 contact structure
