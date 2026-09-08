---
title: '拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口'
slug: "2026-09-12-multimodal-fusion-interface"
date: 2026-09-12
draft: false
categories: ["具身智能", "多模态感知"]
tags: ["具身智能", "多模态融合", "视觉-触觉", "力觉", "本体感觉", "表示接口", "Cross-attention", "Modality Dropout", "VLA", "世界模型", "坐标系对齐", "时间对齐"]
description: '多模态融合常被讲成"选哪种 attention 结构"的问题、但机器人场景真正的瓶颈在更上游：四路信号（Vision / Tactile / Force-torque / Proprioception）在时间基、坐标系基、语义基上从来没对齐过。本文把"融合"拆成 *alignment* 与 *composition* 两步、指出视觉之所以能撑起共同数据接口是因为它把这三个基一次性固化了、而触觉/力觉/本体感觉各自的时间常数与坐标语义都还没收敛、于是所有 cross-attention / shared latent 的努力都会在缺失模态、时间错位、坐标系伪相关上翻车。给出的最小可用路线是：以 contact set 作为跨模态的**表示接口**、raw→slot 的映射放在各模态内部、policy 与 world model 只在 slot 层组合、并用 modality dropout 把"缺模态"作为一等公民写进训练与 benchmark。'
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

想象一个把 RGB 相机、GelSight 指尖、腕部六维力/力矩、关节编码器全部装齐的双臂机器人。硬件清单看起来很"多模态"、但把它交给一个 policy、绝大多数工作会告诉你"用 cross-attention 融合一下"就完事了。这在 demo 里能跑通、**但在真实部署里几乎一定会遇到四类翻车**：某一路掉帧、某个传感器坏掉、坐标系漂移、模型悄悄把某一路当成主导路径而其他路变成噪声。这四类不是工程细节、**它们指向的是同一个根因：模态之间没有先约定好可对齐、可组合、可下传控制器的表示接口**。

这一篇聊具身智能里最容易讲虚的一块：**多模态融合**。不打算推销某个具体网络、而是把"融合"这件事拆到最底层的三个基（时间、坐标、语义）上看清楚：**为什么视觉可以撑起一整个共同数据接口、触觉/力觉/本体感觉不行、以及如果非要现在动手、最小可用的接口长什么样**。

## 0. 全文框架：把"融合"拆成 alignment 与 composition 两步

先把整篇文章的分析框架放在开头、后面每一节都会回到这张图。

```text
                    ┌───────────────────────────────┐
                    │         Raw sensor streams    │
                    │  V · T · F · P  (4 channels)  │
                    └──────────────┬────────────────┘
                                   │  ①  alignment
                       ┌───────────┼───────────┐
                       ▼           ▼           ▼
                    Time base   Frame base  Semantic base
                   (t, Δt, τ)  (SE(3) +   (raw → contact
                                camera FK)  → mode → belief)
                       └───────────┼───────────┘
                                   │  ②  composition
                                   ▼
                       Structured interface  (slots)
                     {contact_set, wrench, robot_state,
                      task_context, belief}
                                   │
                       ┌───────────┼───────────┐
                       ▼           ▼           ▼
                     Policy    World Model   Controller
                       │           │           │
                       └───────────┼───────────┘
                                   ▼
                              Action a_t
```

上半部分 **alignment**（时间 / 坐标 / 语义）是这一篇的**主角**、也是绝大多数多模态论文里被略过的一步；下半部分 **composition**（attention、shared latent、structured slots）反而是相对成熟的领域。整篇文章的判断是：**当前机器人多模态融合之所以还没跑出通用范式、瓶颈在上半部分、而不在下半部分**。

如果 9/11 那篇的三个标签是"触觉自己的三条硬骨头"、那么这一篇要给"多模态融合"也钉三条：

> **Alignment before fusion · Structured contact slots · Modality dropout as first-class**

后面每一节围绕这三条展开。

## 1. "多模态"不等于"多传感器"

这个话题最容易一开始就跑偏——很多人（很多论文引言也一样）把"多装了几个传感器"直接当成"多模态"。这是不成立的。

**模态（modality）的正确定义要同时锁定四件事**：measurement space（信号值域）、physical origin（物理来源）、noise model（噪声结构）、update semantics（触发/采样/时钟语义）。这四件事一起决定了一个数据流能不能被"当成同一种东西"处理。

举几个反例：

- 腕部相机 + 头顶相机 → 是**同模态多视角**、不是多模态；measurement space、physical origin、noise model 基本一致、只有外参不同、可以走同一条 encoder。
- GelSight 指尖的 4 个小摄像头 → 是**一个触觉模态的内部结构**、不是 4 个视觉模态。它的 measurement space 是"弹性体表面形变场"、不是场景 RGB；即便中间用了 CNN、它的下游语义也是 contact geometry、不是 object detection。
- 关节编码器 + 电机电流 → 严格说这两者是**同模态、不同观测点**（都是 robot state、一个是直接测量、一个是间接推断）；把它们当作两个模态融合、只会让 attention 学到冗余。

反过来、下面这几个是**真正的不同模态**：

- Vision（RGB / RGB-D / 事件相机）——光辐射场、坐标系是相机、采样 15–60 Hz。
- Tactile（GelSight / GelSlim / TacTip / 9DTact / 阵列 taxel）——接触界面形变或力分布、坐标系是 sensor frame、100 Hz – 几 kHz。
- Force/torque（腕部 F/T、六维力/力矩）——接触界面的**整体**力旋量、坐标系是 sensor frame + tool frame、500 Hz – 1 kHz。
- Proprioception（关节角 q、角速度 q̇、关节力矩 τ、末端位姿）——机器人自身状态、坐标系是 base frame / world frame、1 kHz+。

有些工作还会把 audio（麦克风阵列）、thermal、gas、ultrasound 加进来当第五/第六模态。它们的分类逻辑与上面一致。

**这一节的落点**：讨论"融合"之前、先把模态清单和它们的四要素定义列清楚。这一列没做、后面所有的 fusion architecture 都是空中楼阁。9/11 的 §2.1 用一棵 taxonomy 树把 contact sensing 分成 Tactile / Force-torque / Proprioceptive 三支、这一篇沿用同样的三分法、把讨论限定在 **V + T + F + P** 这四个模态上。

## 2. 融合之前的三个基：时间、坐标、语义

这一节是全文最技术、也最容易被略过的一节。但**只要这三件事没定、后面的 fusion 就是没有地基的楼阁**。

### 2.1 时间基（Time base）

四路信号的默认时间常数是**差好几个数量级**的：

```text
Vision           ~30 Hz       Δt ≈ 33 ms      帧触发/轮询
Tactile (image)  30–200 Hz    Δt ≈ 5–33 ms    帧触发
Tactile (taxel)  500 Hz – kHz Δt ≈ 1 ms       扫描触发
Force/torque     500 Hz–1 kHz Δt ≈ 1–2 ms     轮询
Proprioception   1 kHz+       Δt ≈ 0.5–1 ms   硬实时
```

工程上常见的做法是把所有信号插值到最低频率那一路（通常就是 vision）。**这一步看似无害、实际上是把触觉、力觉、本体感觉的高频物理细节抹平了**。9/11 里讨论过、滑移检测、瞬态接触力峰、关节冲击这些"事件性"信号、往往就发生在两次 vision 帧之间——你插值到 30 Hz 就等于把它们扔了。

更细的问题还有三类：

- **时间戳语义**：sensor timestamp（曝光/采样瞬间）vs arrival timestamp（软件层拿到）vs host timestamp（融合那一刻）。这三者之间往往差 5–20 ms、足以让"抓杯子的接触瞬间"被错位到"合指之前"。
- **时钟漂移**：相机走自己的晶振、机器人 controller 走另一套时钟、上位机走第三套；几分钟内漂几十毫秒、跑一夜就足以让融合模型学到错误的 lead/lag 相关性。NTP/PTP 只在系统层能压、传感器层还得靠硬触发。
- **事件型 vs 周期型**：触觉里的事件检测（例如 slip 触发）本质上是 sparse event、不是周期采样；把它塞进时间轴上均匀分布的 buffer、事件密度信号就丢了。

**判断标准**：融合层应该**承认时间常数差异、并按物理任务的敏感度选择对齐层**。视觉 policy 可以 30 Hz、力/触觉驱动的柔顺控制必须保持 native rate、事件信号要单独走一条 event bus。把这条写死成一个统一时间基、是**建模假设**、不是工程细节。

### 2.2 坐标系基（Frame base）

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

**坐标系基没做好的典型症状**：训练好的模型在换工具、换相机角度、或者机器人被推歪之后立刻掉点——因为模型学到的其实是"某种传感器 frame 下的相关性"、而不是任务本身。

一个更微妙的问题是**相对位姿 vs 绝对位姿**：接触物理本质只依赖"谁碰谁"的相对关系、但很多 policy 直接把末端在 world frame 下的绝对位置喂进去、于是模型学了一堆不该学的自由度。这一条与 9/11 §3.3 的 "policy 需要一致的是 contact-mode 转移、不是每一个物理参数的绝对值" 是同一种判断。

### 2.3 语义基（Semantic base）

四路信号在语义层级上**天然不对齐**：

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

**多模态融合的关键问题就是：到底在哪一层融合？** 常见错误是在 raw 层就融合（把四路 tensor 拼起来喂进网络）——这相当于强迫 policy 自己学完 2.1/2.2/2.3 全部对齐、代价极高、样本效率极低。相对合理的做法是**在 slot 层融合**、也就是让每一路各自走完 raw → perception → contact geometry 这几层、然后在"接触事件、力旋量、机器人状态、任务 belief"这四个**语义基已经收敛的位置**上做 composition。

**这一节落点**：时间基、坐标系基、语义基是**融合之前**必须先解决的问题；三件事任何一件糊过去、后面的 fusion architecture 再华丽也是在补前面的锅。

## 3. 现有融合范式的谱系

学界讲"多模态融合"、其实主要讲的是**在 composition 这一步怎么组织信号**。这一节按"融合发生的时机"把主流做法排一遍、顺便指出各自的失效模式。Baltrusaitis 等的经典综述 [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) 把这条谱系整理为 representation / learning / feature selection / fusion / application 五层、这一节借用其中 fusion 那一层的分类。

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

Early concat 依然是当前**绝大多数 VLA 的默认做法**——只不过它们通常只 concat vision + proprioception + language token、把 tactile / F/T 整路省了。π0 [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)、OpenVLA [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)、RT-2 [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) 都是这种"vision+proprio+language" concat。这个事实本身就说明问题：**主流 VLA 到今天还是没有认真把 tactile / F/T 加进来**、而且不是"忘了加"、是加了以后 §2 的三个基全都得重新讨论。

### 3.2 Cross-attention / Transformer fusion

比 concat 进一步、把每一路当作一个 token（或一小段 token）序列、上层跑 self-attention / cross-attention。视觉-语言-动作那一支大部分用这个模板；Tsai 等的 Multimodal Transformer [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) 是这条路线的经典起点、显式处理"不同模态时间粒度不一致"的问题。

优点：结构上承认了模态间的对齐是**学出来的**、给了 soft alignment 的空间；能吸收不规整的时间戳。

缺点：

- Attention 的隐式对齐在**样本不足**时会退化成 dominant-modality collapse——attention 权重最后基本全落在 vision、其他模态贡献变成噪声。这在 3D-ViTac、TVL 这类工作里都被明确观察到（后文 §7 展开）。
- Sample hungry。视觉预训练帮了大忙、但触觉/力觉没有同规模的预训练、attention 层拿不到稳定的 modality-specific representation。
- 时间对齐问题被"学习"部分掩盖、但**没有消失**；只是从建模问题变成了训练问题——工程上更贵。

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

- TVL / Binding Touch to Everything [arXiv:2402.13232](https://arxiv.org/abs/2402.13232)（Zhao 等, ICML 2024）——把 tactile 与 vision-language 在 latent 层对齐、是"触觉版 CLIP"讨论里绕不开的节点。
- AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191)（Feng 等, 2025）——试图给**多种异构触觉传感器**做一个统一 static-dynamic 表示、相当于在 tactile 内部先做一次 latent 融合。
- 3D-ViTac [arXiv:2410.24091](https://arxiv.org/abs/2410.24091)（Huang 等, CoRL 2024）——visuo-tactile 联合表示 + fine-grained 插拔操作、给出了明确的 "vision + tactile > vision-only" 实验证据。
- Lee 等 Making Sense of Vision and Touch [arXiv:1810.10191](https://arxiv.org/abs/1810.10191)（ICRA 2019）——最早的 visuotactile 自监督对齐代表之一。

优点：表示层可复用、跨任务/跨 embodiment 有理论上限。

缺点：**训练监督信号哪里来？** 视觉-语言的对比学习能吃海量网络图文对、**视觉-触觉-力觉-本体感觉的四元组对齐没有天然监督源**。目前的解决路径主要有两条——(a) 用机器人遥操作日志把四路强行同采、把"时间戳接近"当 weak alignment supervision（Calandra 等 More Than a Feeling [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) 是这个思路在触觉抓取上的早期实证）；(b) 用语言/视觉作为桥梁、把触觉拉进 V-L 空间（TVL 走的就是这条）。两条都还远没到规模化阶段。

**9/11 §3.5 里把 representation 分成"接口"与"learned representation"两支**——shared latent 严格说是后一支、它需要前一支已经存在才能稳定训练。这一点在下一篇的 §6 会展开。

### 3.5 Structured contact slots：本文推荐的接口路线

最后一种：**不在 raw 层融合、不在 latent 层融合、而是在一个显式约定的中间表示上组合**。这个中间表示不是 learned embedding、是一组**语义已经收敛的槽位**。这一条路线与 9/11 §3.4 "视觉有共同数据接口、触觉没有" 直接对应。

具体槽位设计放到 §6 展开、这里先给个粗线条：

```text
state_slots = {
    contact_set      : list of { p, n, f⊥, f∥, φ_slip, mode }   # 触觉 + F/T 共同产出
    wrench_ext       : 6D                                       # F/T 独立产出
    robot_state      : (q, q̇, τ, EE pose)                       # proprioception
    task_context     : language token / goal embedding           # 高层指令
    belief           : task-relevant latent / explicit           # 状态估计
}
```

四路信号各自负责往这些槽里填内容、fusion 变成"往同一个 dict 里填 key"、policy 与 world model 都只在 dict 层读。这条路线的**优点**：

- 缺一路信号只影响它对应的 key、其他 key 不受污染；modality dropout 天然好加。
- 时间基、坐标系基、语义基都收敛在 slot 定义里；slot 一旦定下、fusion 结构可以随便换。
- 与 9/11 §5.2.1 的 feedback-value ablation 直接对应——A/B/C/D 四组实验就是往 dict 里少填一些 key。

**缺点**：slot 本身要先设计、这活比"套一个 Transformer"重；且 slot 精度不够时、下游 policy 也学不出超出 slot 的能力。

**这一节落点**：融合范式的谱系里、early concat / attention / late fusion / shared latent 都各有用武之地、但**它们的失效模式绝大多数都能追溯到 §2 那三个基**；本文更倾向于把工程重心从"选哪种 attention"移回"先把 slot 定下来"。

## 4. 视觉为什么"先"撑起了共同数据接口

一个自然的问题：§2 那三个基对视觉同样存在、为什么视觉就能撑起一整个共同数据接口？这一节给一个偏历史-工程化的回答、也是给触觉/力觉/本体感觉一面镜子。

**时间基上**：视频天然是 frame-indexed、30 Hz 或 60 Hz、所有下游任务（分类 / 检测 / 分割 / SLAM）都约定俗成"以帧为单位"。这个约定看似 trivial、但它**消灭了融合讨论里最麻烦的一类问题**——所有基于图像的下游算法都用同一条时间轴。

**坐标系基上**：针孔相机模型 + 相机内外参这套约定、把 image pixel ↔ world point 的映射变成一条公式（$s \cdot m = K [R | t] \cdot M$）。所有 3D 视觉、SLAM、NeRF、3D Gaussian Splatting、多视角立体都在这个约定上生长。标定错误当然存在、但**"错误长什么样"是可预期、可复现的**。

**语义基上**：RGB 图像本身没有语义、但视觉社区在过去十几年用 COCO / ImageNet / ADE20K / LVIS 这一批大规模标注数据集、把"语义槽"固化下来了：object class、bounding box、instance mask、depth map、affordance、caption……这些槽位不是学出来的、**是社区共识出来的**。这就是为什么任何一个新视觉模型都能立刻与老模型互操作。

对比一下触觉：

- 时间基：不同 sensor family 从 30 Hz 到 kHz、事件触发与轮询混合、没有共识。
- 坐标系基：GelSight 是 pixel + 弹性体形变、9DTact 是 pixel + 3D 形变场、阵列 taxel 是一维/二维力分布、光学触觉（AnySkin / DigiTact）又是另一套表示；连"一个触觉读数到底长什么 shape"都没统一。
- 语义基：接触点、法向、切向、slip、mode 这些概念在论文里都有人用、**但没有跨数据集统一的 schema**；9DTact 里叫 6D force、GelSight 里叫 shear map、阵列里叫 taxel load——都是同一个物理量的不同投影。

**这一节落点**：视觉之所以赢、不是因为它的 encoder 更聪明、是因为它**先解决掉了 §2 那三个基**、让所有下游 fusion 讨论都跑在同一个地基上。触觉/力觉/本体感觉要形成同规模的融合生态、先要做的是**约定接口**、而不是卷模型结构。

## 5. 每一路信号各自的融合难点

这一节按模态过一遍具体的难点、给 §6 的接口设计做铺垫。

### 5.1 Tactile：图像 / taxel / 光学 / 电容、raw 层就没统一

触觉是这一篇里最"分裂"的一路。同一类物理量（局部接触界面的形变或力分布）、目前至少有以下**表示完全不同**的实现：

```
Image-based      GelSight  · 3 个彩色 LED + 相机 → RGB 形变图
                 9DTact    · 环形彩色 LED + 相机 → 多视角图 [arXiv:2308.14277]
                 GelSlim   · 3 相机 + 弹性体散斑 → 3 路光流场
                 TacTip    · 硅胶锥 + 光纤 + 相机 → 末端位姿

Taxel array      BioTac · 阵列 pressure/temperature/EDS
                 1D/2D 电容阵列 · 局部法向力分布
                 Piezoresistive array · 局部应力图

Optical waveguide AnySkin · DigiTact · 光在弹性体内传播、边缘触发接触点

Proprioceptive-  F/T + kinematics 反推 contact（"soft tactile"）
 inferred
```

四路信号进融合层之前、至少需要**各自走完一条 encoder**——而且这些 encoder 结构差异巨大：CNN for images、MLP for taxel arrays、Spline model for waveguide、IK for proprioception-inferred。

一个常见误解是："那我做一个 tactile foundation model、统一处理所有触觉传感器不就行了？"——这正是 AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) 在做的事、方向对、但**它解决的是 learned representation 层的统一、不是 raw 层的统一**。而且即便有 AnyTouch、它的输出仍然是 embedding、还需要一层"embedding → contact slot"的显式约定才能进融合层。9/11 §3.5 把 representation 分成"接口"与"learned representation"两支、就是为了避免这一层被压在一起。

**融合难点**：

- 不同 sensor family 的时间常数、采样语义都不一样、时间基得 case-by-case 处理。
- Sensor frame 定义不统一（GelSight 是相机像素、taxel array 是阵列索引、waveguide 是光路拓扑）。
- 输出语义不统一（有的给图像、有的给力分布、有的给 3D 形变场）。

**缓解**：在 raw 到 slot 之间引入一层**sensor-specific decoder**、把异构触觉输出统一解码到 (p, n, f⊥, f∥, φ_slip, mode)；这一层可以是解析、也可以是学习的、但它必须是**接口的一部分**、而不是藏在 policy 里。

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
- **坐标系变换**：把 sensor frame 下的 wrench 变到 tool / base / world；每一步都要 SE(3) 伴随矩阵。

F/T 还有一个常被忽略的问题：**它给出的是"整体"力旋量、不是"局部"接触**。一次抓取可能有 3 个接触点、每个点的法向/切向不同、F/T 只能告诉你合力。这个"aggregate vs field-level"的区别正是 9/11 §2.1 taxonomy 树里 tactile 与 force/torque 被分成两支的原因。

**融合难点**：F/T 的语义"干净"（6 个数）但**依赖大量前置补偿**、而补偿的精度直接依赖工具模型 + 动力学模型。工具一换、下游融合层看到的分布就变了。

**缓解**：F/T 在 slot 层应该输出**两件事**——(a) 已补偿的 external wrench（用于 force-aware policy）、(b) 残差 magnitude（用于检测"是不是又漂了 / 是不是又撞到不该撞的东西"）。第二件事常常被忽略。

### 5.3 Proprioception：最容易被当成"背景信号"、其实是最重要的一路

四路里 proprioception 是最快的、也是**唯一机器人天生自带完整通道**的一路。但它有一个尴尬的定位：**很多 policy 网络把 (q, q̇, τ) 直接拼进 state、不当它是"一个模态"**、于是讨论多模态融合时它常常缺席。这是错的。

Proprioception 在融合里有两个作用：

- **作为 contact hypothesis 的输入**：给定期望末端轨迹 + 关节力矩反馈、可以通过 $J^T \hat{F}_{ext} = \tau_{residual}$ 反推外部接触力（所谓 contact inference、Hogan 早年阻抗控制一脉的经典工具）。这条反推路径让 proprio 也能"贡献"到 contact slot 里、而不只是自己那一路。
- **作为时间基与坐标系的锚**：融合层需要一个稳定的 frame、proprio 天然是 1 kHz 硬实时、是四个模态里最像 "时钟" 的一路；EE pose 也可以当作所有其他坐标系（camera / sensor）的 anchor。

**融合难点**：proprio 的信息量太"干净"——它是机器人自己测自己的状态、没有外部世界的不确定性。模型很容易**过度依赖 proprio**、把它当作 shortcut、结果在没有接触的场景表现好、有接触的场景反而没学会用触觉/F/T。这一条正是 9/11 §5.2.1 feedback-value ablation 想避免的——**评估要看"在同样有 proprio 的条件下、加入触觉/F/T 到底带来多少增量"**、不是"只用触觉 vs 只用视觉"。

### 5.4 四路信号的时间常数差异本身就是建模假设

把 §5.1–5.3 与 §2.1 合起来看、可以提炼出这条判断：

> **不同模态的时间常数差异、不是"融合层要不要处理一下"的工程问题、而是"任务需要什么控制带宽"的建模假设。**

举两个例子：

- **擦拭 / 打磨**：力控带宽至少 100–500 Hz、视觉 30 Hz 完全够用、触觉与 F/T 得走 native rate、proprio 也得到力矩层。这一类任务、融合层不能把所有信号降到 30 Hz。
- **抓取-放置 / 装配**：视觉主导、接触事件稀疏；把触觉降采样到 vision rate 是可接受的近似。

这一条与 9/11 §4.1 阻抗控制里"具体频率由 policy arch / hardware servo / controller / compute budget 决定、不写死数字"是同一个判断。

## 6. 一个最小可用的融合接口

到这里可以正面回答"到底怎么办"了。这一节给一个**可实施、可 benchmark、可增量演进**的最小接口。

### 6.1 以 contact set 为核心的中间层

先定义一个跨模态共享的**接触事件集合**：

$$
C_t = \big\{\, \big(\, p_i,\; n_i,\; f_i^{\perp},\; f_i^{\parallel},\; \phi_i^{\text{slip}},\; m_i \,\big) \,\big\}_{i=1}^{N_t}
$$

其中 $p_i$ 是接触位置（在 base frame 下）、$n_i$ 是法向、$f_i^{\perp}$ / $f_i^{\parallel}$ 分别是法向力与切向力大小、$\phi_i^{\text{slip}} \in [0,1]$ 是滑移概率、$m_i$ 是离散接触模式（free / touch / sticking / sliding / rolling / separating）。$N_t$ 是当前活跃接触点数、随任务动态变化。

这个定义刻意做到三件事：

- **sensor-agnostic**：不管下游用的是 GelSight 还是 taxel 阵列还是 F/T、只要能填这几个 key、就能进融合层。
- **物理可解释**：6 个数每一个都有明确物理含义、便于诊断、便于 safety 层挂钩。
- **可下传控制器**：阻抗 / 力位混合 / 抓取力优化 / QP-based whole-body controller 都能直接消费 $C_t$、不需要 policy 再解码。

### 6.2 每一路信号怎么映射到 slot

```text
Vision          ──► predicted contact hypothesis (soft、来自 affordance + grasp planner)
                      填 C_t 的"预测槽"、置信度低
Tactile         ──► detected contact instance (hard、来自 sensor-specific decoder)
                      填 C_t 的"观测槽"、置信度高、位置精度取决于 sensor
Force/torque    ──► aggregate wrench → 分解到已有 C_t 上
                      如果只检测到一个 F/T、无 tactile、可以合成一个"整体接触"条目
Proprioception  ──► 通过 J^T 反推 external contact hypothesis
                      填 C_t 的"推断槽"、精度取决于动力学模型
```

三路填同一个 slot、有冲突时按 (tactile detected) > (F/T-derived) > (proprio inferred) > (vision predicted) 的**证据等级**合并；这个等级不是绝对真理、但给了一个明确、可解释的默认。

### 6.3 融合的时机：不在 raw、不在决策、**在 slot 层**

有了 §6.1 的 slot 之后、fusion 变成三步：

```python
# Step 1: per-modality preprocessing → sensor-specific encoder
#         （时间/坐标系/语义在这一步之内收敛）
raw_v  → enc_v   → pred_contact_hypothesis
raw_t  → enc_t   → detected_contact
raw_ft → enc_ft  → external_wrench + residual
raw_p  → proprio_state → J^T · τ_res → inferred_contact

# Step 2: composition into slots（本文推荐用 structured slots）
C_t        = merge(pred, detected, wrench_split, inferred)
wrench_ext = raw_ft.compensated
robot_state= (q, q̇, τ, EE_pose)
belief_t   = belief_update(C_t, wrench_ext, robot_state, task_ctx)

# Step 3: policy / controller 消费 slot 层
a_t = policy(C_t, wrench_ext, robot_state, belief_t, task_ctx)
```

关键的判断是**"Step 2 的输出格式**才是整个多模态系统的接口"、Step 1 里各模态可以任意换 encoder、Step 3 里 policy 可以任意换结构、只要 Step 2 的 schema 稳定。

### 6.4 缺模态时的 fallback：把"少一路"当成训练分布的一部分

真实部署里、传感器坏掉、掉帧、超时是常态。接口层要显式支持"某一路暂时没数据"：

```python
if tactile_missing:
    C_t = C_t without detected_slot
         + fill_from_ft_and_proprio()
         + inflate_uncertainty()
```

**这不是工程补丁、这是训练时的核心约束**。见 §7.1 modality dropout。

## 7. 融合失败模式与诊断

reviewer 视角、下面这五类失败在机器人多模态工作里最常见、也最容易被"我们用了 cross-attention 所以鲁棒"这类话糊过去。每一类给出**症状 / 诊断指标 / 缓解**。

### 7.1 Modality collapse：attention 权重悄悄退回视觉主导

**症状**：训练 loss 一切正常、validation success rate 正常；一旦在评测里遮住视觉、policy 表现几乎不变——说明其他模态其实没被"用起来"。

**诊断**：可视化 attention 权重按模态聚合的时间曲线；或者更狠、直接跑 **modality ablation gain**（§8.1）——$\Delta_{\text{modality}}$ 若接近零、说明这一路在 policy 里其实是死的。3D-ViTac [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) 的 ablation 之所以重要、就是给这个指标提供了一个明确的参照。

**缓解**：(a) 训练时显式跑 modality dropout、见 §7.5；(b) 给每个模态加 auxiliary supervision（例如 tactile encoder 除了给 policy 用、还得独立预测 slip 事件）；(c) 用 9/11 §5.2.1 的 feedback-value benchmark 而不是纯 success rate 逼出真实贡献。

### 7.2 Temporal smearing：所有信号被强行插值到 30 Hz

**症状**：滑移检测、瞬态接触、冲击类任务的 policy 学不出来——因为高频物理细节已经被 §2.1 讨论过的插值抹掉了。

**诊断**：在评测里对比"以 30 Hz 融合 vs 以 native rate 融合"的成功率差；或者用 $\Delta_{\text{tail}}$ 只看困难接触条件下的表现。

**缓解**：见 §5.4，承认时间常数差异是**建模假设**；不同层用不同频率、事件走 event bus；不要在接口层把高频信号降采样掉。

### 7.3 Frame confusion：模型学到坐标系伪相关

**症状**：模型在训练位姿、训练相机摆放、训练工具型号下表现优秀、稍微动一下就崩。

**诊断**：评测集主动做**坐标系扰动**——把相机外参旋转几度、把工具重心偏移几克、把 base frame 平移几厘米、看成功率掉多少。

**缓解**：slot 层显式使用相对量（$p_i$ 用 EE frame 而不是 world frame、wrench 用 tool frame 而不是 sensor frame）；训练时把坐标系扰动当 domain randomization 的一部分（Tobin 等 [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)）。

### 7.4 Semantic leakage：raw 图像偷偷污染 slot 语义

**症状**：slot 明明定义成"接触几何"、但 policy 表现严重依赖视觉外观（同一个物体的照片换背景就掉点）。

**诊断**：把 slot 里的视觉贡献换成一个"最小充分"的合成 slot（例如把预测 contact 用真值 contact 替）、看性能变化。

**缓解**：slot 层用**独立 encoder** 输出到 slot 的**唯一通路**；不允许 raw pixel 直接进 policy。这条与 9/11 §3.5 "representation interface vs learned representation" 一致。

### 7.5 Missing-modality brittleness：一坏就崩

**症状**：训练时永远所有模态齐全、上线时触觉掉帧一次整个 policy 行为剧烈变化。

**诊断**：跑 **modality dropout test**——评测时按 (1-p) 概率随机丢掉某一路、看成功率-曲线。这条已经是当前 robust multimodal learning 的标准做法、可以参 Maiga 等 MMP [arXiv:2410.03010](https://arxiv.org/abs/2410.03010) 与相关 missing-modality robustness 文献。

**缓解**：训练时把 modality dropout 当一等公民；接口层显式带 uncertainty flag（§6.4）；benchmark 指标把"缺模态恢复速度"作为独立一栏（§8.2）。

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

### 8.3 Temporal & frame perturbation

给视觉加 ± 5 ms 时间抖动、给触觉加 ± 20 ms、给 F/T 加 ± 5 ms、观察 $\Delta_{\text{tail}}$（只看困难接触条件下的成功率）。坐标系扰动同理。这一栏是 §7.2/7.3 的直接量化。

### 8.4 Slot 层的 fidelity 单独测

既然本文主张 slot 是接口、**slot 本身的准确度就应该独立于 policy 测**：

- contact position 与真值（在仿真里）的 IoU / distance error；
- slip detection 的 AUROC；
- contact mode 分类的 macro-F1。

这三个是"融合层之前的质量指标"、它们不合格、policy 层的成功率高只可能是过拟合。

### 8.5 现有 benchmark 的适配度

RoboCasa / LIBERO / ManiSkill3 / BEHAVIOR 这一批主流 manipulation benchmark **大多以视觉为主、触觉缺席**。这一节不下结论、但**建议**：把上面四类指标（ablation / dropout / temporal / slot fidelity）做成一个可选插件、给现有 benchmark 加一层"feedback-value 视角"；9/11 §5.2.1 的 A/B/C/D 四臂消融可以直接借用。

## 9. 与 VLA、世界模型的关系

### 9.1 VLA 现状：架构性缺席、不是工程问题

前面 §3.1 已经点破：主流 VLA（π0 [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)、OpenVLA [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)、RT-2 [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)）几乎都是 **vision + proprioception + language** concat、tactile 与 F/T **缺席**。这个缺席不是"忘了加"、而是**加了以后 §2 那三个基都要重讨论**——VLA 的规模化优势建立在"数据可以海量同采"上、触觉与 F/T 目前的采集成本与统一 schema 都跟不上。

真正的开放问题是：**VLA 的下一波扩展、是加更多 vision+language、还是补上 tactile/F/T slot？** 这一篇押后者——但不是把 tactile 当作"再一个 channel"、而是把 §6 的 slot 接口塞进 VLA 的输入层。Qi 等 T-Dex [arXiv:2309.09979](https://arxiv.org/abs/2309.09979)（CoRL 2023）与 Lee 等 [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) 可以视为这条路线的早期形态。

### 9.2 世界模型：不要在 latent 里糊在一起

RSSM / DreamerV3 [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) 一脉的 latent dynamics 假设 state transition 相对平滑、梯度可导。9/11 §5.1 讨论过、接触事件本质上是 hybrid dynamics 的 mode switch、把它直接塞进 RSSM latent、**模型会倾向于把 mode switch 学成"某种连续变化"、结果就是 contact 事件被抹平**。

建议的架构分工：

- **World model 层保留结构化 contact slot**：mode 是离散、$C_t$ 是集合、$f^{\perp}/f^{\parallel}$ 是实数；这三层不要塞进一个 latent vector。
- **Policy 层可以用 learned representation**：从 slot 学一个 embedding 给 policy 消费是合理的、但这个 embedding 是"下游消费者"、不是"上游数据格式"。
- **两者之间用 §6 的接口连接**、world model 预测下一时刻的 slot、policy 消费当前 slot。

这条分工与 9/11 §5.1 belief update 是同一个思路、只是把它从触觉内部推广到跨模态。

### 9.3 一句话总结

**VLA 缺 tactile slot、world model 缺 contact structure；两件事其实是同一件事——大家都没在接口层做过事情、都寄希望于"多堆点数据、模型自己会学出来"。**

## 10. 结论

这一篇从"多模态融合常被讲成模型结构问题"这个误解出发、把讨论拉回到底层：融合真正的难点是**时间基 / 坐标系基 / 语义基这三件事**、任何 fusion architecture 都不能替代它们。视觉之所以撑起一整个共同数据接口、不是 encoder 更聪明、而是它先把这三个基固化了下来。触觉/力觉/本体感觉各自的时间常数、坐标系、raw 表示都还没收敛、于是所有 cross-attention / shared latent 的努力都会在 §7 那五类失败模式上翻车。

给出的最小可用路线是：**以 contact set 为核心的 structured slots、raw→slot 的映射放在 sensor-specific decoder 里、policy 与 world model 都只在 slot 层组合、并把 modality dropout 当一等公民写进训练与 benchmark**。这条路线不是终态答案、**它更像 9/11 里那条"触觉缺一条可复用的中间表示"的正面回答**——先约定接口、再谈架构。

如果 9/11 那篇钉的三个标签是 Action-conditioned observation · Contact-state representation · Closed-loop value、这一篇的三个标签是：

> **Alignment before fusion · Structured contact slots · Modality dropout as first-class**

三句话把它们串起来：**融合之前先对齐时间/坐标/语义、对齐之后先约定 slot 而不是先选架构、slot 之上把缺模态当分布的一部分而不是异常**。

下一篇（9/13）打算从"接口"往下游走一步：**灵巧手与 in-hand manipulation**——把这一篇的 slot 接口放到多指手的具体场景里、看它能不能撑起 T-Dex / DextrAH / LEAP 这一批近期工作、以及为什么"能装手的机器人很多、真正在做 dexterous 的少"这个反差背后的成本结构。

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

- Zhao et al., *A Touch, Vision, and Language Dataset for Multimodal Alignment*（TVL / Binding Touch to Everything）, ICML 2024 · [arXiv:2402.13232](https://arxiv.org/abs/2402.13232)（tactile-VL 对齐、"触觉版 CLIP"路线的代表）
- Feng et al., *AnyTouch: Learning Unified Static-Dynamic Representation across Multimodal Tactile Sensors*, 2025 · [arXiv:2502.12191](https://arxiv.org/abs/2502.12191)（跨异构触觉传感器统一表示 · §5.1 raw 层难统一的对照）
- Lin et al., *9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation*, ICRA 2023 · [arXiv:2308.14277](https://arxiv.org/abs/2308.14277)（多模态触觉的一个具体形态 · §5.1 异构性证据）

### D · 鲁棒性与 modality dropout（支撑 §7.5 / §8.2）

- Maiga et al., *MMP: Towards Robust Multi-Modal Learning with Masked Modality Prior Fine-Tuning*, 2024 · [arXiv:2410.03010](https://arxiv.org/abs/2410.03010)（把 modality dropout 当一等公民的近期做法 · §7.5 缓解方案参照）
- *Robust Multimodal Learning with Missing Modalities via Parameter Projection*, 2023 · [arXiv:2310.03986](https://arxiv.org/abs/2310.03986)（缺模态条件下的表示对齐 · §8.2 robustness 指标参照）

### E · VLA 与 World Model（支撑 §9）

- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)（VLA 现状缺席 tactile 的实证）
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)（开源 VLA 基线 · 输入侧同样以 vision+proprio+language 为主）
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)（VLA 路线的奠基工作 · 缺席 tactile 的架构起点）
- Hafner et al., *Mastering Diverse Control Tasks through World Models*（DreamerV3）, Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)（latent dynamics 的世界模型代表 · §9.2 "不要在 latent 里糊接触事件" 的对照面）

### F · Sim-to-Real / Domain Randomization 背景（支撑 §7.3）

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)（把坐标系扰动当 domain randomization 一部分的经典做法）

### G · 承接 9/11 · Contact state 与 impedance（背景）

- 9/11 那篇里已经引用过、这一篇继续沿用的：Hogan 阻抗控制三部曲、Posa-Cantu-Tedrake IJRR 2014（hybrid contact-mode trajectory optimization）、Lee 1810.10191、Qi 2309.09979、Huang 2410.24091、Zhao 2402.13232、Feng 2502.12191。这一篇不重复贴链接、需要精确出处请直接看 9/11 的 Sources 部分。

---

> **相关阅读**
>
> - [只会看、不会摸：机器人为什么缺一双"手感"的手](/zh/articles/2026-09-11-tactile-force-sensing/)——这一篇的前作、把触觉与力控单独拆开讲
> - [Sim-to-Real 方法论](/zh/articles/2026-09-10-sim-to-real-methodology/)——§7.3 坐标系扰动、§8.3 temporal perturbation 都可以借用它的 domain randomization 视角
> - [机器人数据为什么比大模型数据更难](/zh/articles/2026-09-09-robot-data-scaling/)——§3.4 shared latent 路线的"监督信号哪里来"问题、其实是数据 scaling 问题
> - [VLA 与世界模型](/zh/articles/2026-09-07-vla-world-models/)——§9 那一节是它的一个具体侧面：VLA 缺席 tactile、世界模型缺席 contact structure
