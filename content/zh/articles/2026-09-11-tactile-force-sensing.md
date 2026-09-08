---
title: '只会看、不会摸：机器人为什么缺一双"手感"的手'
slug: "2026-09-11-tactile-force-sensing"
date: 2026-09-11
draft: false
categories: ["具身智能", "多模态感知"]
tags: ["具身智能", "触觉", "力控", "阻抗控制", "导纳控制", "接触丰富操作", "Active Perception", "VLA", "Sim-to-Real", "机器人数据", "GelSight", "接触状态估计"]
description: '通用机器人真正稀缺的、不是"再多一个触觉传感器"、而是一条把异构接触信号稳定转换成任务相关接触状态、并实时进入控制闭环的能力。本文以"闭环"为贯穿框架、把 sensor / raw observation / contact perception / state estimation / task-relevant latent / policy / controller / action 分层拆开、澄清视觉与触觉的"弱可观测 vs 直接可观测"边界、把力控写成四种并列控制范式而非等级、并把触觉价值同时定位在"接触状态感知本身"与"最容易体现工程 ROI 的长尾场景"。'
toc: true
related_articles:
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-09-robot-data-scaling
  - 2026-08-26-world-model-in-robotics
  - 2026-09-07-vla-world-models
  - 2026-09-03-vla-deep-dive
  - 2026-09-06-embodied-ai-landscape
---

> 接 [Sim-to-Real 方法论](/zh/articles/2026-09-10-sim-to-real-methodology/) 与 [机器人数据 scaling](/zh/articles/2026-09-09-robot-data-scaling/)：那两篇分别在讲"训练分布与 evaluation 分布之间的错配怎么预算、怎么干预"、"下一份数据该采什么"。这一篇想往下再切一层——**真实世界不是纯视觉世界，很多关键状态发生在接触界面上；而通用机器人恰恰在"接触 → 反馈 → 调整动作"这条闭环上还远没有像视觉那样成熟**。

闭着眼睛，你也能拧开一瓶没喝完的水、把一颗鸡蛋从一堆里捡起来而不捏碎——靠的不是眼睛、是手上的"感觉"。而在许多**通用机器人**上、恰恰"眼睛"（相机）很发达、"手"的接触感知却还没有形成像视觉那样可复用的数据、表示、模型和硬件接口生态——这并不是说机器人完全没触觉（腕部六维力/力矩、关节力矩、指尖触觉阵列现在都有人用）；而是它*真正稀缺的、是一条可规模化部署的"感知接触 → 调整动作"的闭环*。这一篇聊聊具身智能里最容易被忽略、却又最要命的一块：**触觉与力控**。

## 0. 全文框架：把"接触闭环"当作贯穿主线

先把整篇文章的分析框架放在开头、后面每一节都会回到这张图。

```text
                        WORLD
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
            Vision                  Contact
              │                       │
              ▼                       ▼
        World state            Contact state
              │                       │
              └───────────┬───────────┘
                          ▼
                    State estimator
                          ▼
                 Task-relevant latent
                          ▼
                        Policy
                          ▼
                Motion / Force reference
                          ▼
                 High-frequency controller
                          ▼
                        Robot
                          ▼
                       Contact  ────┐
                          │         │
                    tactile / force │
                          └─────────┘  ↺

     ── 一条横跨全链路的隐性层 ──
     Calibration · Synchronization · Registration
     时间戳对齐 · 坐标系对齐 · sensor-to-robot 标定 ·
     tactile-to-contact-frame 注册
```

这条链路里、"触觉传感器测到了东西" ≠ "机器人知道发生了什么" ≠ "机器人据此改变了动作"。这三步是三种不同的能力、缺哪一步、闭环就断在哪一步。而在它们之下、还压着一层常被论文忽略的工程层——*不同模态之间的时间戳对齐、坐标系对齐、sensor-to-robot 外参标定、tactile-to-contact-frame 注册*。缺这一层、再漂亮的 tactile encoder 也很难稳定进入 policy / controller。全文其余章节其实都是在解释：**为什么这条链路里从 raw signal 到 task-relevant latent 的那一段、目前还没有像视觉那样形成统一技术栈。**

## 一、为什么"看见"不够用

这些年机器人进步最明显的地方是"看"——相机越来越清、视觉模型越来越强。于是大家默认：只要看得够准、动作就能做对。但一旦涉及到**接触**、"看"就开始力不从心。

先把措辞收紧一点。不是"视觉看不到接触、只有触觉看得到"——这个说法容易被机器人/控制方向的人挑刺。*更严谨的表述是*：**某些接触状态对视觉是不可辨识的（unidentifiable）、或只能通过长时间序列 + 物理先验间接推断；而触觉/力觉提供的是更直接的局部观测**。

再拆细一点、"视觉弱可观测"其实同时包含至少 5 类不同性质的问题、混着说容易失焦：

- **Occlusion（遮挡）**：接触面常常正被物体本身或机械手挡住。
- **Insufficient resolution（分辨率不足）**：局部形变、微滑移、边缘接触点在像素层面被平均掉。
- **Temporal aliasing（时间走样）**：滑动、颤振、冲击发生在毫秒级、远低于常见相机帧率。
- **Hidden internal force / stress（不可见的内力）**：接触法向力、剪切力、物体内部应力、不在表面成像里。
- **Friction & material state（摩擦与材质状态）**：湿滑、油污、粘弹性、粗糙度、只在被"按下去"时才显现。

所以更精确的一句话不是"vision cannot measure force"、而是：**在遮挡、低分辨率、反光/透明表面、快速运动等条件下、vision-only 对接触力/滑移/接触状态的估计更间接、也更脆弱**——视觉*确实*可以通过形变、光流、物体位移、marker tracking、inverse dynamics 等间接估计力、只是这条推断链长、假设多、鲁棒性差。几个例子：

- **捏鸡蛋**：鸡蛋能承受多大夹持力、很难从外观直接确定。真正起作用的是一个小闭环：视觉定位置 → 建立初始接触 → 触觉 / 力觉读出接触面积与是否滑移 → 控制器增减夹持力 → 再看有没有继续滑。**关键不是"感觉到了"、而是"能根据感觉改变下一步动作"**。
- **插 USB / 插销孔**：不是"视觉完全没用"——视觉把末端带到大致位置、随后进入一个非常典型的**接触丰富任务（contact-rich manipulation）**：一旦接触发生、环境对运动的约束突然变强、此时靠接触力和力矩告诉机器人"抵在哪、滑没滑进去"、比看像素更直接。
- **拧瓶盖**：拧到多紧算到位？打滑了要不要再加点力？这些都是接触面在"说"的话。
- **从一叠纸里拿一张**：抓多了、带起来了、滑走了——很多失败在单帧画面上几乎无感、手里却一清二楚。

> *在很多接触丰富任务中*、视觉负责把机器人带到"接触附近"、触觉与力觉负责在接触之后继续提供局部观测——这不是说视觉在接触后就完全退场、而是它的信号密度与可辨识度显著下降。做动作尤其是精细操作、后半段往往才是决定成败的那一维。

## 二、机器人的"感觉"缺在哪

### 2.1 先把几个词分清：tactile / force / proprioception / force control

这一篇里它们容易被混着说、但严格讲是几层。*先说"感觉"这一侧*：**视觉**告诉你外部世界状态；**触觉（tactile）** 告诉你局部接触状态——接触位置、压力分布、剪切、滑移、纹理；**力 / 力矩感知（force sensing）** 告诉你机器人整体受到多大外力、外力矩；**本体感觉（proprioception）** 告诉你机器人自身关节、末端当前在哪、以什么姿态运动。*再说"动作"这一侧*：**力控（force control）** 则是拿这些反馈去主动调节行为。

一个末端装了六维力/力矩传感器的机械臂、直接得到的是"末端受到的合力与合力矩"；一块高分辨率指尖触觉阵列、直接得到的是"接触位置与压力空间分布"。两者*并非等价*、也*并非互斥*——前者是 wrench-level 观测、后者是 field-level 观测。对多接触点、局部滑移、接触几何重构这类问题、field-level 观测通常更丰富；对整体负载、碰撞检测这类问题、wrist F/T 反而更稳、更便宜、更成熟。

所以全文想强调的链条是：

```text
Sensor
  ↓
Calibration · Synchronization · Registration
                           (时间戳对齐、坐标系对齐、外参标定)
  ↓
Raw observation            (压力阵列 / RGB / 6-axis wrench / joint current)
  ↓
Contact perception         (哪里有接触、接触面积、法向/剪切分离)
  ↓
State estimation           (接触几何、滑移速度、刚度、摩擦系数、contact mode)
  ↓
Task-relevant latent state (抓取稳不稳、插没插进去、拧到位没有)
  ↓
Policy / controller
  ↓
Action
  ↓
New contact                ↺
```

**至少 7 层"从信号到意义"的工程/学习问题**——把触觉当成"传感器"是不够的、把它当成"从 raw signal 到 task-relevant latent 的一整条链路"、才更接近工程现实。

人的手掌是全身触觉最密集的地方之一：压强、纹理、温度、滑动、本体感觉一起工作。相比之下、大多数机械手能读到的信号要稀疏得多：

| 感觉维度 | 人手 | 常见机器人 |
|---|---|---|
| 位置 / 姿态 | 有（本体感觉） | 由关节编码器 + 电机电流 + 力矩 + 运动学估计共同给出、**在自身位姿这一维往往比人更精确** |
| 关节力矩 | 并非直接"读"出关节力矩、而是通过肌梭、腱器官等间接感知肌肉力量与关节负荷 | 部分型号能读（关节电流或专用力矩传感器） |
| 接触压强分布 | 非常密集 | 大多缺失或很粗 |
| 接触纹理 / 局部材质属性 | 按下去 → 感觉阻力 → 感觉回弹 → 感觉表面摩擦、**在接触过程中动态获得** | 视觉能识别大量"看起来"的材质；缺的是接触过程中那种**局部、动态**的材质信息（往往要靠视觉 / 力 / 振动 / 电流 / 运动响应**综合推断**） |
| 滑动 / 摩擦 | 立刻察觉 | 可以检测（触觉阵列、剪切力、接触点运动、视觉等）、难在**稳定、低延迟、泛化地估计并用于控制** |
| 温度 | 有 | 基本没有 |

这里想强调一点：**"人类触觉"和"机器人传感器"并不是一一对应的关系**。机器人并不是一没触觉阵列就完全"摸不出来"——通过末端力矩、关节电流、视觉形变、振动等多通道、也能推断出不少东西。真正的问题不在"有没有传感器"、而在：

> 机器人缺少**高密度、多模态、低延迟、覆盖广、而且能提供局部接触信息的接触感知、以及把它稳定转成"接触状态"的那条估计链路**。"局部性"是这里特别容易被忽略、却又最关键的一维——同样是"两个指尖一起把东西端起来"：*腕部六维力/力矩传感器直接提供的是末端合力与合力矩*；*而指尖触觉阵列进一步提供接触位置与压力空间分布*。对多接触点、局部滑移、接触几何重构这类问题、后者通常提供更丰富的观测。需要小心的是：*给定运动学、接触几何假设、动力学与执行器状态、wrist F/T 也可以做 contact localization / wrench decomposition / external force estimation*、只是那条反演的适定性和可扩展性远不如直接读场分布——*两种信息在"能否直接支撑下游决策"这一维上并不等价*。
>
> 机器人现在的样子、更像一只裹着厚手套、只在几个点上装了稀疏感受器的手：能感到"碰到了"、却很难稳定地感到"哪里、怎么碰的、碰得对不对"。

### 2.2 一句更硬的技术结论

如果说这一节要浓缩成一个可被引用的判断：

> **机器人缺的不是触觉信号本身、而是能够把异构接触信号（压力场 / RGB / 剪切 / 振动 / 力矩 / 电流）稳定转换成任务相关接触状态、并实时进入控制闭环的能力。**

这句话把"触觉"从"再加一个传感器"降权、把"感知–估计–控制"整条链路升权、也解释了为什么"给机器人装上 GelSight 就万事大吉"是不成立的。

## 三、为什么这块一直难

### 3.1 传感器本身：一个多维 trade-off 空间、不是一个"越密越好"的单变量

触觉要贴在接触面上、就得薄、要耐磨、能重复承受挤压还不坏。但真正让工程师头疼的、往往不是"能不能做出来"、而是*这些目标彼此冲突*。

*需要提醒一句*：vision-based tactile（GelSight / 9DTact 这一类 *camera + elastomer + illumination + reconstruction*）、capacitive / resistive tactile array、piezoresistive skin、磁通量方案、wrist 6-axis F/T——它们面对的工程约束*完全不同*。把下面这 7 维放进同一个 trade-off 空间、只在"跨模态做系统设计"这一层才成立；*单个具体传感器方案内部、往往只撞其中两三面墙*。

在跨模态这一层、至少 7 个维度彼此冲突：

```text
Spatial resolution  ↔  Force range
        ↔  Bandwidth  ↔  Latency
        ↔  Durability ↔  Calibration stability
        ↔  Mechanical compliance
        ↔  Cost
```

方向性的 trade-off 举几条：

- 更高的空间分辨率、更高的动态带宽、与更大的量程、通常难以同时实现；高频采样还会进一步放大噪声、带宽、数据量与长期稳定性的工程约束。
- 柔软 sensing skin 更适合贴合曲面抓取、却不一定适合高载荷或长时间工业节拍。
- 传感器装在指尖会改变 fingertip geometry、增加一层 sensing 就减少一层机械顺应——**"加了触觉"这件事本身可能就在改变操作动力学**。
- 越接近真正人手密度/柔性一致性的方案、越难做成工业产品——既要测得准、又要长期稳定可标定可重复、还不能因为加了一层传感器就把机械手原本的抓取性能破坏掉。

顺带一条经典综述：Dahiya 等人 2010 年在 IEEE T-RO 上把"从人类触觉到 humanoid 机器人触觉"的鸿沟系统梳理过（[Tactile Sensing—From Humans to Humanoids](https://ieeexplore.ieee.org/document/5339133)）、十五年后读仍然不过时——这也侧面说明这块进展不快。

### 3.2 视觉能 scale、不只是因为"RGB 统一表示"；触觉数据的本质是 action-conditioned observation

严格说、触觉数据并不是空白。学界和工业界已经有一批触觉数据集、视觉-触觉数据集、触觉预训练、以及所谓"触觉基础模型"的早期探索；GelSight、[GelSlim](https://github.com/Antaoyu/GelSlim_4Gen_Curvature-Based_Fingertip_Sensor)、[TacTip](https://www.tacpix.com/products)、[9DTact](https://arxiv.org/abs/2308.14277) 等传感方案也在被反复使用。跨模态对齐方向、[TVL / Binding Touch to Everything](https://arxiv.org/abs/2402.13232)（Zhao et al., ICML 2024）把触觉和视觉-语言在表示层对齐、是当前"触觉 foundation model"讨论里绕不开的一个节点。操作侧、[3D-ViTac](https://arxiv.org/abs/2410.24091)（Huang et al., CoRL 2024）用 visuo-tactile 表示学 fine-grained 插拔、给出了一个明确的"vision + tactile > vision-only"的实验证据。

但真正的问题不是"没有触觉数据"、而是*数据的规模、跨传感器可比性、任务覆盖与跨 embodiment 可迁移性都远远没有达到视觉数据的程度*。

先承认一个直觉——"像素 / RGB / 视频这一层通用表示把差异吸收掉了"——是对的、但*还不够*。视觉之所以能 scale、其实是多条件同时成熟：

- **廉价、通用的传感器**（CMOS 图像传感器一条产业链）；
- **标准化的图像/视频格式与色彩约定**；
- **互联网上天然存在海量图像数据**（不需要谁专门去采）；
- **弱监督 / 自动标注友好**（alt-text、点击、社交信号）；
- **空间局部性 & CNN/ViT 可复用架构**；
- **成熟的 benchmark 生态**（ImageNet / COCO / Kinetics / LAION ...）；
- **预训练权重可以跨任务复用**。

触觉几乎没有同时具备上面任何一条——但最要命的是其中一条*结构性差异*：

> **视觉数据主要是 passive observation、而触觉数据天然是 action-conditioned observation。**

```text
Vision：   world  ─────►  observation

Tactile：  action  ──►  contact  ──►  observation
                          ▲              │
                          └── next action ◄┘
```

相机可以"放在那里就自动获得大量数据"、而触觉*必须"机器人主动接触才能产生数据"*——于是触觉数据集从来就不只是一个 $X$、它更像：

$$\mathcal{D} \;=\; \big\{\,\big(s_t,\ a_t,\ c_t,\ o^{\mathrm{tac}}_t,\, y_t\big)\,\big\}_t$$

其中 $c_t$ 是接触状态、$o^{\mathrm{tac}}_t$ 是触觉观测、$y_t$ 是任务结果。更麻烦的是——*采集数据的策略 $\pi(a\mid o)$ 本身就在决定数据分布*：换一条策略、看到的接触分布就变了。这条逻辑也直接接上 [机器人数据 scaling](/zh/articles/2026-09-09-robot-data-scaling/) 里 interaction distribution 的观点：*机器人不只是缺"动作数据"、还缺大量"带接触状态、并且由合理采集策略生成的动作数据"*。

沿着 action-conditioned 这条判断、"触觉数据 heterogeneity"可以再拆成三轴：

- **Sensor heterogeneity**：pressure / RGB deformation / shear / vibration / wrench / current 是不同物理量、彼此没有天然可比性；
- **Embodiment heterogeneity**：不同 hand、finger geometry、材料、执行器、运动学、接触面；
- **Interaction-policy heterogeneity**：你*摸哪里、怎么摸、用多大力、速度多快*、本身就决定了你*获得什么*触觉信息。

第三轴是这一节最想强调的、也最容易被忽略——**它是前两条之外的独立维度、而且直接决定"触觉数据集能不能被复用"**。

### 3.3 Sim-to-Real：接触模型误差比视觉模型误差更容易击穿策略

光和碰撞在仿真里相对好模拟、但"接触—形变—摩擦—滑移"这一套接触力学非常复杂——摩擦系数、接触刚度、材料粘弹性、微观表面结构、接触面积变化、传感器噪声、软材料与传感器之间的耦合……每一环都可能引入偏差。

需要收窄一句：并不是"触觉 Sim-to-Real 就是比视觉 Sim-to-Real 更难"这么笼统——视觉 Sim-to-Real 也有大量未解问题（照明、材质、渲染 gap）。更准确的是：**接触丰富任务的 Sim-to-Real 往往对接触动力学、摩擦、材料、传感器结构与接触几何更敏感、因此 domain gap 更难仅靠简单的视觉 domain randomization 覆盖**。

再加一层更技术的说法：对大多数 manipulation policy、*真正需要被仿准的不是每一个物理参数的绝对值、而是策略实际依赖的接触模式（contact mode）*及其状态转移与失效边界：

```text
free space
   ↓ touch
contact (sticking)
   ↓ shear force ↑
sliding  ⇄  sticking
   ↓ rolling
rolling
   ↓ release
separation
```

μ = 0.42 还是 μ = 0.38、很多策略其实并不敏感；但*"现在是 sticking 还是 sliding"、"接触什么时候发生分离"这些mode boundary*、一旦 sim 判错、policy 就会瞬间失灵。也正因如此、sim-to-real 校准的合理目标常常不是"把每一个物理量估准"、而是*把 contact mode 的转移边界推到 policy 可承受的范围里*——这条逻辑也直接接上 §2.1 链路里 "State estimation" 那一环、以及 [Sim-to-Real 方法论](/zh/articles/2026-09-10-sim-to-real-methodology/) 的 policy-conditioned mismatch 观点：**reality gap 只要落在 policy 不敏感的方向上、就不是问题**。

而且要避免一个走极端的判断：*"仿真不准" 不等于 "只能全部靠真实数据"*。工程上更常见的是把它当成一个可校准的近似——而且是一条*循环*、不是一次跑完的单向 pipeline：

```text
Simulation
    ↓
Policy
    ↓
Real robot
    ↓
Failure / residual
    ↓
Parameter estimation (system identification)
    ↓
Simulator calibration
    ↓
Retrain
    ↺
```

再配合域随机化（domain randomization，[Tobin et al., 2017](https://arxiv.org/abs/1703.06907)）、真机微调（real-world fine-tuning）与残差学习 / action transformation 一类的方法来补上剩余误差。**仿真仍然是训练和验证的强力工具**；只是触觉/接触任务的校准循环走得比视觉任务更"手工"、也更依赖 SI → calibration → retrain 这条闭环——本质上还是 [Sim-to-Real 方法论](/zh/articles/2026-09-10-sim-to-real-methodology/) 那一篇讲的 *data engine / feedback loop*。

### 3.4 标准化：不是缺一个 hdf5 schema、是缺 5 层基础设施

视觉真正占优势的地方、其实并不是"相机完全一样"、而是**图像的基础表示高度统一**、并围绕它形成了一整套共享接口。触觉麻烦得多：不同传感器输出的可能是压力阵列、3D 几何形变、RGB 图像、剪切力、法向力、六维力/力矩、电阻/电容变化、高频振动信号……甚至**同一个"压力"在不同传感器上都未必具有直接可比性**。

把"标准化缺失"再拆一层、更清楚：

| 层 | 视觉（已形成相对成熟的共享接口与 benchmark 生态） | 触觉（目前状态） |
|---|---|---|
| **硬件层** | 相机接口、镜头规格、内参/外参约定 | 传感几何、安装方式、坐标系、采样率、延迟、标定协议都还在各做各的 |
| **数据层** | 图像/视频格式、时间戳、metadata、色彩空间 | raw representation、多模态时间同步、contact labels、跨数据集 metadata 都不统一 |
| **表征层** | pixels → CNN/ViT features（跨任务复用） | 接触位置、法向/剪切分离、滑移、形变、材质——*这些"中间表示"本身还没形成公认约定* |
| **任务层** | detection / segmentation / depth 等标准任务 | 抓取稳定性、插入到位判定、接触模式、失败态——定义分散 |
| **benchmark 层** | COCO / ImageNet / Waymo / nuScenes 等 | cross-sensor transfer / cross-hand transfer / sim-to-real / closed-loop success / recovery rate 等指标各自为战 |

一句限定：**视觉本身也远没有把标准化问题"做完"**——camera intrinsics / extrinsics / color calibration、depth sensor heterogeneity、event camera、multispectral、rolling shutter、不同帧率、都是尚未收敛的开放问题。这里说视觉成熟、*指的是"相对成熟的共享接口 + benchmark 生态"*、而不是"标准化问题已被解决"——两条不要混着读。

一张表看下来、"触觉生态还不成熟"就不再只是一句口号——而是"每一层都还差一段"。

### 3.5 触觉很难"说清楚自己感到了什么"——问题在缺一条中间表示链路

先别急着说"触觉没有语义"。恰恰相反、**触觉完全可以携带很丰富的语义**——材质、纹理、是否空心、有没有形变、抓得稳不稳、都可以通过触摸判断出来。真正的问题是：*图像到高层语义之间、已经有一套非常成熟的中间表示*（像素 → 特征 → 目标 / 场景）；而*触觉原始信号到高层语义之间、还缺少这样一条被广泛复用、跨传感器可用的通用表征链路*。

这条链路就是 §2.1 里那 5 层的核心：

```text
raw tactile signal  →  contact perception  →  contact state estimation
                   →  task-relevant latent state  →  action
```

相机天然给你一个二维空间结构：这里有个杯子、那里有只手、这是红色、那是桌面——表示层面就离语义近一些。而触觉给你的往往是一片压力分布、一组时序信号、一次剪切力变化——语义并不是没有、只是机器人必须自己学会："这不是一团压力变化、而是**物体正在从我的指尖滑出去**。"

一个值得关注的研究方向是把这条链路做成"触觉的基础模型"——TVL / AnyTouch / Binding Touch 这类工作已经在做（对齐到 vision-language 表示空间）、但目前还没有出现"触觉版 CLIP"级别的公共底座、更没有出现"触觉版 ImageNet"级别的训练语料。*这里所谓"触觉版 CLIP"、并不是指必须复刻 CLIP 的模型结构、而是指一种能跨传感器、跨 embodiment、跨任务*复用的公共表示空间。真正缺的是*representation standard*（"raw signal → contact state → action-conditioned representation" 这一整条约定）、而不一定是一种 foundation-model architecture——这个区分很关键、别把"造一个大模型"错当成"造出统一表示"。

> **我个人的判断（带条件、非绝对）**：如果目标是"构建跨硬件、可规模化学习的通用触觉能力"、那么*数据可扩展性、跨传感器表示、以及从 raw signal 到 task-relevant latent 的中间表征*、是**最基础的几个瓶颈之一**。当然、不同路线最先撞到的墙并不一样：*做传感器的先受限于耐久性与一致性、做整机的先卡在机械手本体、做控制的先撞上传感器带宽和执行延迟、做学习的才被数据量和标注成本压住*。但如果不定义"跨硬件、可规模化"这个目标、那这五条难点之间就没有客观排序；如果定义了、那"数据 + 中间表征 + 生态"这条组合最决定别人能不能站在你上面继续做。

## 四、力控：四种并列控制范式、不是能力等级

有了传感只是第一步、更难的是让机器人**根据力去调整动作**。这里有个基础的概念分野、值得讲清楚——*它们不是从"初级"到"高级"的四段台阶、而是四种并列的控制范式*：

```text
                     ┌─ Position control     ：规定"末端到哪里"
                     │
Task command ────────┼─ Force control        ：规定"施加多大力 / wrench"
                     │
                     ├─ Impedance control    ：通过控制器实现期望的
                     │                        力–运动动态关系
                     │                        F = Mẍ + Bẋ + K(x − x_d)
                     │
                     └─ Admittance control   ：由测得的力解出期望运动
                                              ẍ = M⁻¹(F − Bẋ − K(x − x_d))
                                              再作为位置/速度参考下发
```

- **位置控制**：命令"手走到某个坐标"。硬、准、但一旦撞到意料之外的阻力、容易"较劲"、把东西顶坏。
- **力控制**：命令"施加多大的力"。适合"贴着表面走"这类以接触力为主的任务。
- **阻抗控制（impedance control）**：让末端*表现出期望的动态关系*、而不是简单地"输入位移输出力"。控制器主动闭合这条 $F = M\ddot{x} + B\dot{x} + Kx$ 的关系式、让机器人在扰动下按设定的质量-阻尼-刚度行为响应。
- **导纳控制（admittance control）**：先测外力、再解出期望运动、把它作为位置/速度参考下发给一个高刚度的位置/速度控制器。

这里要加一个技术限定、避免被控制方向的读者挑刺：*严格来说、阻抗与导纳都在实现某种"期望的力–运动动态关系"*、两者的实际差别更接近*"控制结构里谁是指令、谁是测量"*——阻抗以运动为输入、以力为输出、通常需要良好的力矩控制带宽；导纳以力为输入、以运动为输出、更常搭配高刚度位置环使用。工业实践中两者也常常混用、由传感器与执行器带宽共同决定哪一侧更稳。

"把手变软"这个流行的比喻需要加一个限定：**"变软"并不是简单地降低刚度**、而是让机器人按照设定的力—位移关系、对外界扰动产生**可控的顺应**。真正把刚度一路降到很低、机器人只会软趴趴、精度全丢——所谓"调准手感"、调的是弹簧-阻尼曲线的形状、不是把它关掉。所以许多接触不确定性较高的装配、插拔、擦拭和精细操作、需要的是**一定程度的主动或被动顺应**——并不是所有任务都得靠主动力控、高精度定位、专用夹具、柔顺机构、被动顺应、预定义轨迹也都是常见工程答案。**力控并不等于"机器人自己学会控制力"**、工程系统里往往是*机械柔顺 + 阻抗控制 + 力反馈 + 策略学习*共同完成。

更进一步的想法、是把力/触觉和视觉、语言、本体感觉**放进同一个策略与学习框架**：不只把触觉"多加一个 sensor channel"、而是给策略**增加了一个关于接触状态的高价值观测**。这条方向是对的、也已经有工作证明它有效（例如 [3D-ViTac](https://arxiv.org/abs/2410.24091)）；真正还没解决的、是*还没形成像视觉语言模型那样规模化、统一、跨硬件泛化的范式*。所以更精确的表述不是"视觉-触觉学习跑没跑通"、而是"跑通了、但离统一还很远"。

### 4.1 一个关键的时间尺度事实：VLA ≠ 控制器

这是 AI 读者和机器人控制读者的交界处、也是很多泛 AI 文章会犯错的地方。把触觉接进 VLA、**不等于 VLA 直接输出电机电流或力矩**。更常见的分工是：

```text
Vision / Language / Tactile / Proprioception
                    ↓
              State / Policy  (低频、几十~几百 ms)
                    ↓
       ┌────────────┴────────────┐
       ↓                         ↓
 task-space command        impedance / force target
 (pose / velocity)         (stiffness / damping)
       └────────────┬────────────┘
                    ↓
       High-frequency controller (高频、~100 µs ~ ms)
                    ↓
                  Motor
                    ↓
                 Contact
                    ↓
             tactile / force
                    ↺
```

需要被单独点出来的一个物理事实：**VLA 的时间尺度和 contact control 的时间尺度不是一回事**。高层策略通常是低频更新、低层控制器需要在更高频率下闭环运行——*典型系统中两者往往相差一个甚至多个数量级、具体倍差取决于 policy architecture、action chunking、inference accelerator、motor controller 与 impedance loop*。中间那层控制器才是让"决策"落到"电机"上的那座桥。也正因如此、即便 VLA 再大、它也不会自动变成一个能扛住接触扰动的控制器——**"触觉进入 VLA" ≠ "VLA 就是控制器"**。

## 五、真正的价值是"闭环"

到这里其实可以亮出一个更本质的判断：**触觉的价值不是"感知更多"、而是让机器人能够根据接触结果实时改变动作**。

需要小心的是、这一步很容易被说成"触觉第一次让操作闭上了环"——严格讲并不成立。*视觉伺服（visual servoing）*本身就是闭环（Hutchinson、Hager、Corke 的经典综述 *A tutorial on visual servo control*、IEEE Transactions on Robotics and Automation, 1996 · [Semantic Scholar](https://www.semanticscholar.org/paper/A-tutorial-on-visual-servo-control-Hutchinson-Hager/4676d81d6a2477f6ea1de5723fc81e431fbaa96f)）、很多机器人系统在"看 → 估计状态 → 动作 → 再看"这条路上早就转得起来了。触觉真正带来的、不是"从开环变成闭环"、而是：

> **让原本主要依赖视觉的闭环、进一步覆盖到"接触发生之后"那些视觉难以可靠观测的局部状态。**
>
> 换句话说、*触觉的意义不是让机器人多装了一个传感器、而是让它能在接触发生之后、知道发生了什么、并据此改变下一步动作*——这也是整篇最想把一句话留给你。

```text
只看 · 视觉闭环：
       看 ──► 估计 ──► 动作 ──┐
        ▲                      │
        └────── 再看 ◄─────────┘
       （接触一旦发生、被遮住、
         反馈就断了）

加入 · 接触反馈的闭环：
       看 ──► 动作 ──► 接触
              ▲           │
              └─ 触觉/力 · 修正 ◄┘
```

这也是"会看又会摸"和"只会看"最本质的差别。 [Sim-to-Real](/zh/articles/2026-09-10-sim-to-real-methodology/) 那一篇讲过的分布漂移、误差累积、recovery——它们中的很大一部分、其实都是**"闭环里没有接触反馈这一环"**在最后一厘米处的表现。

再加一步收敛：**进入闭环 ≠ 闭环受益**。一条触觉信号即便已经接进 observation space、只要它 *latency 高、noise 大、drift 明显、可辨识性差、或者 localization 精度不够*、下游 policy 也可能学不会怎样用它、系统收益接近 0。所以更硬的一句话是：

> **触觉是否有价值、不取决于它有没有进入 observation space、而取决于它是否提供了足够及时、可辨识、与任务相关的 information、使闭环策略能够改变动作并提高系统稳定性。**

这句话也把 §3.1 的 trade-off、§4.1 的时间尺度、以及后面 §6 的 ROI 判断串起来了——*"接进闭环"是必要条件、"接得对"才是充分条件*。

### 5.1 主动感知：触觉不只是"避免失败"、它改变了 information acquisition 的机制

触觉不只负责"避免失败"、还负责**探索**。人看不清杯子里有什么、会伸手摸一摸；不知道物体有多重、会拿起来掂一掂；不确定零件是不是滑了、会轻推试探一下；不知道材质软硬、会按压摩擦。这条逻辑在机器人里有个更宽的名字：**主动感知（active perception / active sensing）**。

它的范围其实比"主动触觉"要广——*移动相机、改变视角、绕着物体转一圈、伸进去探一下、推一推动一动*、都算 active perception；**触觉只是其中一种非常重要的实现方式**。它的核心是：*机器人不只是"通过传感器被动感知世界"、而是"主动做动作来换取想要的信息"*。

如果把这一层再往前推一步、会得出一个更硬的判断：*触觉不只是多了一种 observation modality、它改变了"信息是怎么被拿到的"这件事的机制本身*：

```text
Vision   ：  observe  ─────►  act

Tactile  ：  act  ──►  contact  ──►  observe  ──►  act
```

也就是说、*在触觉这一侧、action 本身就同时是 sensing operation*——探索与操作开始耦合、最优策略不再只优化 task reward、还要优化 **information gain**。这条逻辑天然把话题接进 robotics 的经典框架：**POMDP / active sensing / information gathering**——它们处理的正是"当状态部分可观测时、如何通过动作换取有用观测、再做决策"的问题。触觉任务几乎是一个天然的 POMDP：接触状态在被摸之前是 hidden 的、只有 action 才能让 belief 收缩。

这也让 §3.2 的 "action-conditioned observation" 判断在这里闭环了：*因为数据本身由 action 决定、采集策略与执行策略就必然耦合在一起*——这是触觉和视觉在数据集定义上的一个非常结构性的差异、不是"多加点数据就好"的问题。

一个值得关注的研究方向是：*把主动触觉、VLA、世界模型放到同一套策略里联合优化*——VLA 把视觉语言映射到动作；下一步很可能是"视觉 + 语言 + 触觉 + 本体感觉 → 接触状态理解 → 动作闭环"。*这是预测性判断、不是技术事实*、但已经有像 TVL / 3D-ViTac / AnyTouch 这类工作在做早期对齐。

### 5.2 闭环到底怎么量：一套评价指标体系

上面所有关于"闭环"的判断、都需要能被测量才算立得住。这里给出一个跨层的评价指标表——**它同时也是 §3.4 "标准化缺失"那一张表的镜像**：既然标准化每一层都缺、评价指标也就每一层都得各自补齐。

| 层级 | 关键指标 |
|---|---|
| **Sensor** | spatial resolution · force range · bandwidth · latency · drift · calibration stability · cost |
| **Contact perception** | contact detection accuracy · contact localization error · normal/shear decomposition error |
| **State estimation** | slip detection AUROC · contact-mode classification · force / pose / stiffness estimation error |
| **Policy** | task success rate · sample efficiency · **failure recovery rate** · **time-to-recovery** |
| **Control** | wrench / pose tracking error · contact-force tracking error · overshoot on transition |
| **System** | cycle time · long-run success rate · **contact-induced failure rate** · **success under perturbation** |
| **Generalization** | **cross-sensor transfer** · cross-hand transfer · cross-object · **performance degradation under sensor shift** |

其中*最后三项加粗指标是最应该被主流论文与工业评估采用的*——因为它们直接对应 §6 的 "触觉价值在长尾 / 条件性收益" 论点、也才是真正回答"闭环有没有形成"的问题。只看 task success rate、会把"触觉带来的稳定性收益"完全平均掉、看不到 §6 想讲的那部分。

## 六、触觉最容易体现工程 ROI 的场景：Last Mile 长尾

先说明一个限定：*触觉的价值并不止于长尾 / recovery*。触觉本身就是一等公民的感知通道、直接提供：

- **Object identity / material cues**（材质、软硬度、是否空心）
- **Grasp state**（接触面积、力分布、稳定性）
- **Contact geometry**（法向 vs 剪切分离、接触位置）
- **Pose refinement**（把物体在手里"搓"到目标位姿）
- **Texture recognition**（表面粗糙度、纹路）
- **Active exploration**（看不清时伸手探一下）

上面这一部分、属于"触觉作为感知通道本身"的价值、不依赖"闭环"就已经成立。

**什么任务其实不太需要触觉**——把这条边界画出来、比继续论证"触觉很重要"更重要：

- 大范围视觉定位（AGV / AMR 在结构化仓库里靠 SLAM 跑）；
- 结构化环境里的粗粒度 pick-and-place（正对摆放、公差宽松、单一材质）；
- Open-space navigation（避开人和障碍、不涉及精细接触）；
- 完全不接触物体的操作（扫描、喷印、演示、纯显示）；
- 高精度夹具 / 治具 / 视觉伺服已经能覆盖的装配。

所以真正需要被写下来的一句话是：

> **触觉的价值与 *contact richness* 正相关、而不是与"任务复杂度"简单正相关。复杂任务 ≠ 一定需要触觉。**

工程上更可操作的一个粗略判据是：

$$\text{Tactile ROI} \;\propto\; \underbrace{\text{contact uncertainty}}_{\text{接触状态有多难预测}} \;\times\; \underbrace{\text{contact sensitivity}}_{\text{性能对接触状态有多敏感}} \;\times\; \underbrace{\text{failure cost}}_{\text{一次接触失败多贵}}$$

*这一乘积越大、触觉越有 ROI*——精细装配、易碎物体、柔性物体、脏污 / 湿滑 / 摩擦变异大的场景都在高值区；反之，即便任务在其它维度上很难（长时间导航、复杂决策树），只要 contact uncertainty 低，触觉也帮不上多少忙。

回到 Last Mile：*它是触觉最容易量化 ROI 的场景*。前面几篇 [Sim-to-Real](/zh/articles/2026-09-10-sim-to-real-methodology/) 与 [机器人数据 scaling](/zh/articles/2026-09-09-robot-data-scaling/) 反复说过一句话：落地拼的是系统。触觉带来的收益、往往*不体现在"成功率从 80% 变成 90%"*、而体现在那些视觉难以处理的**长尾接触状态**上——稍微歪了、稍微滑了、卡住了、接触点变了、摩擦忽然变化、公差跑到边、物体被捏变形。这些恰恰就是"最后一段路"里那一片"看起来都是小概率、合起来占比不小"的分布。

再补一句更贴近工程直觉的：**触觉带来的收益往往是"条件性的"**。常规条件下没有触觉也能 95% 成功、加了触觉变成 99%——单看 Demo、两个机器人都把东西抓起来了、差异并不显眼。真正的分岔出现在：物体稍微湿一点、夹持位置偏一点、摩擦系数变了一点、物体开始形变、接触点悄悄滑动、轻微滑移刚开始还没掉下去……*触觉不一定让 Demo 更惊艳、却很可能让失败长尾变短*——这正好接得上 [Sim-to-Real 方法论](/zh/articles/2026-09-10-sim-to-real-methodology/) 的核心命题：**系统能不能被信任、看的从来不是高光时刻、而是那些"看起来都是小概率、合起来占比不小"的时刻**。

> 视觉模型解决"看懂世界"的大量常规情况；触觉与力控一方面在"接触状态感知"这一维有独立的 sensing value、另一方面在"接触发生之后的局部异常"这一维最容易量化 ROI。而机器人真正难的地方、恰恰是那些**低概率、组合复杂、难以提前穷举**的接触状态。

## 七、一个更硬的问题

如果只允许我把这一篇压缩成一个开放问题、我会写成：

> **为什么触觉至今仍没有形成一套能够跨传感器、跨 embodiment、跨任务、把 action-conditioned contact observations 稳定映射为可泛化 contact state、并最终进入实时控制闭环的统一技术栈？**

这个问题一旦立起来、这篇文章要回答的就不再是"**为什么机器人需要触觉**"、而是一个明显更硬的：*为什么 tactile robotics 到今天还没有出现自己的"视觉时刻"。*

拆开来正好是 §0 那张图里的四段：

1. **Sensor → Calibration → Raw observation**：硬件 trade-off 空间太大、没有像"CMOS 图像传感器"这样的通用形态；跨模态的时间戳对齐、坐标系注册、sensor-to-robot 标定*至今没有一个可复用的公共基础设施*（§3.1、§2.1）。
2. **Raw observation → Task-relevant latent state**：缺一条被广泛复用的中间表示链路、导致每个团队都在自己造 contact perception / state estimation 层（§3.5）。
3. **Task-relevant latent → Control loop**：VLA 的时间尺度与 contact control 的时间尺度还没接上、"感知–估计–控制"这三段没有共享同一套 benchmark 与评价指标（§4.1、§3.4、§5.2）。
4. **Data acquisition policy 本身是问题**：触觉数据是 action-conditioned observation、采集策略与执行策略天然耦合、无法像视觉那样"放个相机就自动攒数据"（§3.2、§5.1）。

*把这四段解释清楚、这一篇从"触觉科普"就往"触觉技术评审"再往前挪了一步*。

## 小结：四层技术架构

回到 §0 那张图、把这条链路压缩成一个四层技术主线：

```text
视觉             →  世界理解 (world state)
触觉 / 力觉       →  接触理解 (contact state)
力控              →  接触闭环 (constrained contact behavior)
策略学习          →  把这些局部反馈变成可泛化的行为
```

这四层不是"缺一个机器人就什么都干不了"——视觉分拣、视觉定位、开放空间导航、许多无接触操作、没有触觉也照样跑得很好。但*对于接触丰富、精细操作、接触状态变化明显的任务*——装配、插拔、擦拭、柔性物体操作——这四层里缺任何一层、都很可能成为机器人从"能做"走到"稳定做好"的瓶颈。**真正从"会看"走到"会干活"、中间少的那一段往往不在感知模型有多大、而在接触—反馈—调整这条闭环有没有被认真做出来。**

所以下一波真正让人眼前一亮的机器人进展、也许不在"看得更懂"、而在——它终于把这条"感知接触 → 调整动作"的闭环建起来了。

---

**延伸阅读 / Sources**

*触觉传感 / 综述*

- Dahiya, Meta, Schmitt, Cowley, *Tactile Sensing—From Humans to Humanoids*, IEEE Transactions on Robotics, 2010（经典综述）· <https://ieeexplore.ieee.org/document/5339133>
- GelSight 高精度视觉触觉传感器 · <https://gelsight.com/>
- GelSlim 开源指尖触觉传感器 · <https://github.com/Antaoyu/GelSlim_4Gen_Curvature-Based_Fingertip_Sensor>
- TacTip 3D 打印触觉指尖 · <https://www.tacpix.com/products>
- Lin et al., *9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation*, ICRA 2023 · [arXiv:2308.14277](https://arxiv.org/abs/2308.14277)

*Visuotactile learning / 触觉表示与基础模型*

- Huang et al., *3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing*, CoRL 2024 · [arXiv:2410.24091](https://arxiv.org/abs/2410.24091)
- Zhao et al., *A Touch, Vision, and Language Dataset for Multimodal Alignment*（TVL / Binding Touch to Everything）, ICML 2024 · [arXiv:2402.13232](https://arxiv.org/abs/2402.13232)
- Feng et al., *AnyTouch: Learning Unified Static-Dynamic Representation across Multimodal Tactile Sensors*, 2025（跨传感器统一表示 · 对应 §3.5 "representation standard"）· [arXiv:2502.12191](https://arxiv.org/abs/2502.12191)

*接触丰富操作与力控（经典）*

- Hogan, *Impedance Control: An Approach to Manipulation* (Parts I–III), ASME Journal of Dynamic Systems, Measurement, and Control, 1985 · [Semantic Scholar](https://www.semanticscholar.org/paper/Impedance-Control%3A-An-Approach-to-Manipulation-Hogan/f5f8a6aa4adc13070224c2bd43a255c4e0844c15)
- Hutchinson, Hager, Corke, *A tutorial on visual servo control*, IEEE Transactions on Robotics and Automation, 1996（视觉伺服作为闭环的经典综述）· [Semantic Scholar](https://www.semanticscholar.org/paper/A-tutorial-on-visual-servo-control-Hutchinson-Hager/4676d81d6a2477f6ea1de5723fc81e431fbaa96f)

*Sim-to-Real*

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
