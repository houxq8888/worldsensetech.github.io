---
title: '拼起来不等于看懂：机器人多模态融合缺的不是模型、是接口'
slug: "2026-09-14-multimodal-fusion-interface"
date: 2026-09-14
draft: false
categories: ["具身智能", "多模态感知"]
tags: ["具身智能", "多模态融合", "视觉-触觉", "力觉", "本体感觉", "表示接口", "多模态状态估计", "Hybrid State Estimator", "Structured State Contract", "Observability", "Identifiability", "Registration", "Cross-attention", "Modality Dropout", "VLA", "世界模型", "坐标系对齐", "时间对齐", "不确定性"]
description: '多模态融合常被讲成"选哪种 attention 结构"的问题、但机器人场景真正的瓶颈在更上游：Vision / Tactile / Force-torque / Proprioception 四路信号在时间基、坐标系基、任务语义、有效性与来源上从来没对齐过。本文把系统拆成四层——measurement layer（时间/坐标/标定）→ perception & semantic projection → state estimation → structured state contract——再交给 policy / world model / controller / diagnostics 多消费者。核心主张不是"再要一个 fusion operator"、而是 heterogeneous observation 与多消费者之间缺一层稳定的 structured state contract：它规定"什么信息必须以什么语义、单位、frame、时间、不确定性、来源、有效性暴露出来"、但把推理算法与下游表示学习完全留白。interface ≠ inference algorithm ≠ latent representation；belief 只是这份 contract 可能携带的一类内容。Contact set 是一个 manipulation-specific 实例、不是接口定义。Benchmark 部分给出 Interface Property Benchmark：schema/encoder/consumer swap、oracle/estimated/end-to-end 三组受控 baseline、uncertainty-aware disagreement、consistency graph 与 fault isolation、representation bottleneck ablation。三条设计原则：Register before compose · Expose belief at the interface · Design for disagreement。'
toc: true
related_articles:
  - 2026-09-13-tactile-force-sensing
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-11-sim-to-real-intervention-lenses
  - 2026-09-12-sim-to-real-evaluation-protocol
  - 2026-09-09-robot-data-scaling
  - 2026-09-07-vla-world-models
  - 2026-09-03-vla-deep-dive
  - 2026-08-26-world-model-in-robotics
---

> 接 [只会看、不会摸：机器人为什么缺一双"手感"的手](/zh/articles/2026-09-13-tactile-force-sensing/)：那一篇钉下三个标签——Action-conditioned observation · Contact-state representation · Closed-loop value——并留下一个明显的问号：**触觉/力觉/本体感觉既然各自都有价值、为什么没有像视觉那样形成一个可复用的共同表示？** 这一篇正面回答这个问号。答案不在模型结构、**在接口**。

想象一个把 RGB 相机、GelSight 指尖、腕部六维力/力矩、关节编码器全部装齐的双臂机器人。硬件清单看起来很"多模态"、但把它交给一个 policy、很多工作会告诉你"用 cross-attention 融合一下"就完事了。这在 demo 里能跑通、**但在真实部署里常常会遇到四类翻车**：某一路掉帧、某个传感器坏掉、坐标系漂移、模型悄悄把某一路当成主导路径而其他路变成噪声。这四类不是工程细节、**它们指向的是同一个根因：模态之间没有先约定好一份可被 policy、world model、controller、diagnostics 共同消费的、显式表达时间、坐标、语义、不确定性、来源、有效性的 structured state contract**。

这一篇聊具身智能里最容易讲虚的一块：**多模态融合**。不打算推销某个具体网络、也不打算反对 cross-attention、更不打算反对 end-to-end learning。本文真正反对的是——**让 state estimation、cross-modal composition 与 control 三件事全部隐式发生在一个不可诊断、不可复用的 latent 里、而没有一个显式的、跨消费者的状态边界**。文章会先把"融合"拆到最底层的三个基（时间、坐标、任务语义）上、看清楚**视觉为什么先跑通了、触觉/力觉/本体感觉还差在哪、以及如果非要现在动手、最小可用的接口长什么样**。

## 0. 全文框架：四层系统与 structured state contract

先把整篇文章的分析框架放在开头、后面每一节都会回到这张图。**这一版最重要的一个概念收紧**：全文要立的东西不叫"一个更好的 fusion 层"、也不完全等于上一稿口径里的"structured belief interface"——更准确的名字是 **structured state contract（结构化状态契约）**。belief / uncertainty-aware state 只是这份契约**可能承载的一类内容**、不是契约本身；契约本身规定"暴露什么、以什么语义暴露"、而**不规定下游用什么算法去推断、也不规定 policy 内部用什么表示**。

```text
              Heterogeneous observations
              V / T / F / P / audio / ...   (raw sensor streams)
                              │
                              ▼
            ┌───────────────────────────────────┐
            │  Layer 1 · Measurement layer      │  ← registration 的家
            │  time / frame / calibration /     │     (显式、可标定、可估计)
            │  latency / sensor noise model     │
            └─────────────────┬─────────────────┘
                              ▼
            ┌───────────────────────────────────┐
            │  Layer 2 · Perception &           │  ← semantic projection
            │  semantic projection              │     (observation →
            │  sensor-native encoder            │      task-relevant state)
            └─────────────────┬─────────────────┘
                              ▼
            ┌───────────────────────────────────┐
            │  Layer 3 · State estimation       │  ← estimator 的家
            │  hybrid / uncertain               │     (EKF / factor graph /
            │  x̂_t, Σ_t  or general B(x_t)       │      learned filter 皆可选)
            └─────────────────┬─────────────────┘
                              ▼
            ┌───────────────────────────────────┐
            │  Layer 4 · Structured State       │  ← 本文的贡献所在
            │  Contract (interface)             │     value · semantics ·
            │  value·semantics·frame·time·      │     frame · time ·
            │  uncertainty·provenance·validity  │     uncertainty · provenance
            └───────┬────────┬────────┬─────────┘  · validity
                    ▼        ▼        ▼
        Policy   World Model  Controller  Diagnostics / Safety
                    │        ▼        ▼
                    └───── Action a_t ─────┘
```

这张图强调三件事、也是全文真正想立住的架构主张：**fusion 不是一个网络模块、而是一个横跨 measurement / perception / estimation / contract 四层的系统问题**；**state estimator 与 state contract 是两层不同的东西**——estimator 负责"算出 $\hat{x}$、$\Sigma$ 或一般化的 $\mathcal{B}(x)$"、contract 负责"规定算出来的东西以什么语义、单位、frame、时间、不确定性、来源、有效性暴露给多个消费者"、两者可以各自独立演进；**接口层不是 learned embedding、是一份 contract**、里面每一格都有单位、坐标系、时间戳、不确定性、来源、有效性、优化的目标是**可互操作**、不是"对某个模型友好"。

一句可以站得住的中心命题（这一版把它从"belief state"改成"state contract"、以避开"这不就是 state estimator 输出吗"的质疑）：

> **The missing abstraction is not necessarily another fusion operator, but a stable structured state contract between heterogeneous observations and multiple downstream consumers. This contract should expose task-relevant state together with its semantics, frame, time, uncertainty, provenance, and validity, while leaving the inference algorithm and downstream representation learning unconstrained.**

或者用三张 boxed 公式表示全文最核心的三条 claim：

$$
\boxed{\;\text{Sensor-specific observations} \;\rightarrow\; \underbrace{\text{Structured State Contract}}_{\text{stable interface}} \;\rightarrow\; \{\text{Policy, World Model, Controller, Diagnostics}\}\;}
$$

$$
\boxed{\;\text{Interface} \;\neq\; \text{Inference algorithm}\qquad \text{Belief} \;\neq\; \text{(necessarily) Bayesian posterior}\;}
$$

$$
\boxed{\;\text{representation compatibility} \;\neq\; \text{interface compatibility}\;}
$$

一旦这么理解、文章里 VLA、world model、tactile foundation model、modality dropout、cross-modal contradiction、F/T constraint、contact set **全部变成同一条主线上的不同实例、而不是六七个并列观点**。

**本文不提出新的 fusion operator、也不提出新的 state estimator**——它提出的是一份**关于 estimator 输出的契约（a contract over estimator outputs）**、以及检验这份契约的实验方法学。全文贡献收敛成三条、后面每一节都可以视为这三条的具体化：

1. **Abstraction**——为异构机器人 observation 提出 **structured state contract** 这一层：规定跨消费者共享的状态如何以显式语义、单位、frame、时间、不确定性、来源、有效性暴露、而不是把它留给某个下游网络的隐式 latent。
2. **Representation**——给出一份 contact-rich manipulation 下的具体实例化：contact set + wrench + robot state + uncertainty + provenance + validity + lifecycle（并强调 contact set 只是实例之一）。
3. **Evaluation**——提出一套 **Interface Property Benchmark**：schema / encoder / consumer swap、oracle / estimated / end-to-end 三组受控 baseline、uncertainty-aware disagreement、consistency graph 与 fault isolation、representation bottleneck ablation、registration perturbation。

三条设计原则保持不变、它们已经覆盖 measurement → representation → robustness 的完整链路、本文**不打算再加第四条**：

> **Register before compose · Expose belief at the interface · Design for disagreement**

**Contact set 是这套接口在 contact-rich manipulation 场景下的一个实例、不是接口本身的定义**——这一句会贯穿全文、也是本文把自己定位为 *architecture position paper* 而不是 *tactile survey* 的原因。

## 1. "多模态"不等于"多传感器"

这个话题最容易一开始就跑偏——很多人（很多论文引言也一样）把"多装了几个传感器"直接当成"多模态"。这是不成立的。

**"modality" 本身没有一个唯一的物理学定义、它是分析视角的产物**。为了讨论方便、本文的 taxonomy 用一个四要素约定：**measurement space（信号值域）、physical origin（物理来源）、noise model（噪声结构）、update semantics（触发 / 采样 / 时钟语义）**。这四件事一起决定了一个数据流能不能被"当成同一种东西"处理。不同作者可以把边界画得略松或略紧、但只要一次性约定清楚、后面的讨论就不会飘。

按这套 taxonomy、几个常被误当作"多模态"的例子：腕部相机 + 头顶相机是**同模态多视角**；GelSight 指尖的 4 个小摄像头是**一个触觉模态的内部结构**、measurement space 是弹性体表面形变场、下游语义是 contact geometry 而不是 object detection；关节编码器 + 电机电流是**同一 robot-state modality 下的两个 observation channels**、共享同一 latent 物理状态、当作两个模态融合大概率只学到冗余。反过来、**Vision / Tactile / Force-torque / Proprioception** 是本文 taxonomy 下真正的四个不同模态——各自的 measurement space、坐标系、effective observation rate 见 §5 表格。有些工作还会加 audio、thermal、gas、ultrasound 当第五 / 第六模态、分类逻辑一致。《触觉·力控》 §2.1 用一棵 taxonomy 树把 contact sensing 分成 Tactile / Force-torque / Proprioceptive 三支、本文沿用同样的三分法、把讨论限定在 V + T + F + P。

## 2. 四层系统的最底两层：Measurement layer 与 Semantic projection

这一节对应 §0 图的 Layer 1（measurement layer / registration）与 Layer 2（perception + semantic projection）。先做一个**术语收紧**：**"配准（registration）" 与 "语义投影（semantic projection）" 是两个不同性质的事**、把它们都塞进 alignment 会让读者以为只要做 math 变换就够了。

```text
Pre-fusion pipeline
├── Registration        (Layer 1 · measurement-layer、显式时间/几何/标定变量)
│   ├── Temporal registration
│   └── Spatial registration
└── Semantic projection (Layer 2 · observation → task-relevant state variables)
```

**Registration 主要属于 measurement layer：它应尽可能由显式的时间、几何与标定变量描述、并优先通过 calibration 或 estimation 求解、而不是默认交给下游 fusion network 隐式学习**。这一句是本文对 registration 的正式定义、比"硬约束 / 有闭式解"要弱一档——因为真实机器人里 registration 常常也是 estimation problem（online calibration、time-varying extrinsics、compliant sensor mounting、tactile elastomer deformation、thermal drift、synchronization / latency estimation）——但**性质仍然是 measurement-layer 的显式变量、不是任务语义**。**Semantic projection** 更接近 perception 与 state estimation、是这一篇后面 §6 state contract 的主角。两者**都在 fusion 之前**、性质完全不同、分工也不同。

### 2.1 Temporal registration（时间配准）

四路信号的默认时间常数差好几个数量级：vision ~15–60 Hz、tactile（相机型）~30–200 Hz、tactile（阵列型）~500 Hz – kHz、F/T ~500 Hz – 1 kHz、proprio ~几百 Hz – kHz。这里有个常被混淆的点：上面这些是 **effective observation rate**、不是 sensor 内部采样率。**effective observation rate 应显式定义为**——"对某个消费模块而言、随时间有意义（temporally meaningful）的观测真正可用的速率"、粗略地可写成

$$
f_{\text{effective}} \;\approx\; \min\!\big(\,f_{\text{sampling}},\; f_{\text{processing}},\; f_{\text{transport}},\; f_{\text{consumer}}\,\big)
$$

相机型触觉的 "kHz taxel 采样"、如果它的输出图像仍然只有 30 Hz、那融合层能拿到的有效速率就是 30 Hz。

把各模态重采样到共同低频 policy clock（通常是视觉那一档）是常见 baseline、但**对于 contact-rich control、这会把部分高频事件压缩成不可见的 alias**——滑移检测、瞬态接触力峰、关节冲击这些"事件性"信号往往就发生在两次 vision 帧之间、降采样等于扔掉。更稳健的架构是**多速率共存**：视觉 policy 可以 30 Hz、力/触觉驱动的柔顺控制保持 native rate、事件信号走单独 event bus。

时间配准里必须写进接口层的细节不止 timestamp、**还包括 latency**。一个观测被消费的时刻是

$$
t_{\text{effective}} \;=\; t_{\text{sens}} + \Delta_{\text{processing}} + \Delta_{\text{transport}} + \Delta_{\text{queue}}
$$

而 **timestamp synchronization $\neq$ causal synchronization**：把 camera timestamp 对齐到 tactile timestamp、不代表两者描述的是同一个 physical state 时刻——中间那段 processing / transport 延迟才是真正决定"这条观测对应哪一刻世界状态"的东西。三个必须写进接口层的时间细节：**时间戳语义**（sensor / arrival / host timestamp、三者差 5–20 ms 就足以让"抓杯子的接触瞬间"被错位到"合指之前"）、**latency 与因果对齐**（上面的 $t_{\text{effective}}$、以及它带来的 lead/lag）、**事件型 vs 周期型**（slip 触发本质上是 sparse event、塞进均匀分布 buffer 事件密度就丢了）。**判断标准**：**把时间常数与延迟差异写成建模假设、不要靠降采样掩盖**。这条与 《触觉·力控》 §4.1 是同一个判断。

### 2.2 Spatial registration（空间配准）

四路信号天然挂在四个不同坐标系上：Vision camera frame、Tactile sensor frame、F/T sensor frame + tool frame、Proprio base / world frame。想把它们放进同一个融合层、至少要做三件事：**手眼标定**（$T^{cam}_{base} \in SE(3)$、任何一次碰撞都可能让它漂掉零点几度到几度）、**传感器安装标定**（fingertip sensor frame → link frame 的偏移 $T^{\text{sens}}_{\text{link}}$）、**力旋量变换与重力 / 惯性补偿**（把 sensor frame 下的 wrench 变到 base 或 world、扣掉工具重力与惯性项；具体公式见 §5.2）。

**这里再补一个区分**：registration 参数本身可能是**静态**的（出厂标定好的 extrinsic 常数 $T$）、也可能是**动态**的（时变隐变量）。对触觉尤其明显——$T_{\text{sens}\rightarrow\text{link}}(t)$ 在弹性体形变、柔性安装、温漂下并不是常数。**把 dynamic registration 当成"标定一次就完事"是常见的失败源**；更合理的建模是把 $T_{\text{sens}\rightarrow\text{link}}(t)$ 当作随时间漂移的估计量、纳入 measurement layer 的在线估计。这恰好支持本文的核心判断——**registration is measurement-layer estimation**、而不是"配一次就冻结的常数"。

空间配准没做好的典型症状：模型在换工具、换相机角度、或者机器人被推歪之后立刻掉点——因为模型学到的其实是"某种传感器 frame 下的相关性"、而不是任务本身。一个更微妙的问题是**相对位姿 vs 绝对位姿**：接触物理本质只依赖"谁碰谁"的相对关系、但很多 policy 直接把末端在 world frame 下的绝对位置喂进去、于是模型学了一堆不该学的自由度。

### 2.3 Semantic projection（语义投影）

这一段严格来说不是"配准"、而是**"把观测投影到与任务相关的状态变量"**：这里 "semantic" 指的是 **task-relevant state abstraction、不是 human-readable semantics**。Vision → object / scene 状态、Tactile → local contact 状态、F/T → aggregate wrench 状态、Proprio → self-state——可以写成一组具体映射：

```text
RGB        → object pose / scene geometry
GelSight   → contact patch
F/T        → external wrench
(q, q̇)     → kinematic state
```

《触觉·力控》 §2.1 里把 signal → meaning 分成九层（Sensor → Calibration → Raw obs → Contact perception → Contact geometry & wrench → Contact mode & physical state → Task-relevant belief → Policy / controller → Action → New contact）。**多模态融合的关键问题就是：到底在哪一层做 composition？** 常见错误是在 raw 层就 composition（把四路 tensor 拼起来喂进网络）——这相当于强迫 policy 自己学完 §2.1 / §2.2 / §2.3 全部、代价极高、样本效率极低。相对合理的做法是**让每一路各自走完 raw → perception → contact geometry 这几层、然后在"接触事件、力旋量、机器人状态、任务 belief"这四个语义已经收敛的位置上做 composition**、也就是本文后面 §6 的 structured state contract。

**这一节的落点**：temporal / spatial 两小段 registration 是 measurement-layer 的显式变量（Layer 1）、semantic projection 是 perception 与 state estimation 的分工（Layer 2）。三件事任何一件糊过去、后面的 fusion architecture 再华丽也是在补前面的锅。

## 3. 现有融合范式的谱系：机制不同、层次不同

Baltrusaitis 等的经典综述 [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) 把多模态 ML 整理为 representation / learning / feature selection / fusion / application 五层、这一节借用其中 fusion 那一层的分类。**先做一个重要的定位**：下面这几种范式**不是互斥的架构选择、而是不同轴上的机制**——一个成熟系统往往同时用其中几种：

| 机制 | 解决什么问题 | 属于本文 §0 哪一层 |
| --- | --- | --- |
| Modality-specific encoder | 各自 raw → 局部观测 | Layer 2 · Perception |
| Temporal / spatial registration | 时钟、SE(3)、坐标系、latency | Layer 1 · Measurement |
| Cross-attention | 学出来的 composition | 可在 Layer 2 / 3 / 4 任一处 |
| Shared latent / VLT-style | 表示层跨模态对齐 | Layer 2 · Semantic projection |
| Late / decision fusion | 决策层合成 | Layer 4 下游 · policy |
| Structured state contract | 语义 schema 与跨消费者边界 | Layer 4 · Contract |

本文不反对 cross-attention、**限定它的职责**：**cross-attention is not guaranteed to recover explicit temporal / spatial / semantic registration from raw data**——Attention 可以把已经对齐好的表示组合起来、但把 registration 也"顺手学出来"在缺乏显式约束时通常不可靠。这一句是本节与整篇文章的枢纽之一。

**Early concat**（raw / embedding 层拼接）：把四路 encoder 之后 concat 成 state vector、扔 MLP 或 Transformer。门槛低、样本够多时深度网络会自己学出一些对齐关系。缺点：时间常数差异被 concat 掩盖、容易学到虚假 lead/lag；任何一路掉帧就要么补零、要么 hold-last-value、都是分布外；触觉与视觉 encoder 结构差异大、梯度互相污染；训练时没见过的模态缺失组合一出现、policy 就崩。

**关于 VLA 的一个必须小心的段落**：把 RT-2 / OpenVLA / π0 都归为 "vision + proprio + language concat" 不准确。**RT-2** [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) 的核心是把动作直接**表达成文本 token**、与 VLM 联合 fine-tune；**π0** [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) 是 VLM backbone + **proprioception token** + **noisy action chunk**、通过 flow matching 出 action；**OpenVLA** [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) 的公开配置里明确有多相机、depth、proprioceptive state encoding 分支。**以这些代表性公开系统为例、可以观察到：它们的共同点是"以 vision-language 预训练作为主要规模入口、把机器人状态作为额外 representation 注入 policy"。** 这里要特别注意区分两件事：**"支持某类输入"与"建立了跨消费者共享的 structured state contract"是两回事**——OpenVLA 能吃 proprio 输入、不等于它定义了可被 world model / controller / diagnostics 复用、带 unit / frame / validity 的状态契约。这个判断在 §9.1 会再展开。

**Cross-attention / Transformer fusion**：把每一路当作 token 序列、上层跑 self / cross-attention。Tsai 等的 Multimodal Transformer [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) 是这条路线的经典起点、显式处理"不同模态时间粒度不一致"。现代 multi-modal transformer 完全可以搭配 timestamp / positional embedding、relative temporal encoding、modality embedding、frame-aware features、modality-specific encoder、modality dropout / masking、auxiliary per-modality loss、attention masks——**真正的问题不是"attention 会翻车"、而是如果没有把这些当接口契约写清楚、attention 会顺手把配准也"学"掉、结果学到的是数据分布里的巧合、不是可移植的接口**。成熟系统里、cross-attention 更适合在 **perception 与 registration 已经把输入送到 Layer 3 / 4 之后、围绕 state contract 做 composition**（但这不是唯一合法用法、见 §6.5）。

**Late / decision-level fusion**：每一路各出一个子 policy、决策层再加权投票或按 confidence gating。更接近传统 robotics——视觉给粗调、F/T 给细调、触觉给 slip recovery、proprioception 给 nominal trajectory tracking。**一个常见的工程 pattern** 是——任何一路掉线只是掉一个 proposal、不会全局崩、容易加 safety layer。缺点是底层耦合信息丢了（"从触觉和 F/T 合起来看、这个接触是稳定滑动还是 stick-slip"这类跨模态联合信息、分开提 proposal 之后在决策层很难重建）；weighting 规则要么手写不可扩展、要么学又回到 early concat 的问题。《触觉·力控》 里阻抗控制 + 视觉粗定位 + 触觉滑移检测的组合、本质就是这种 late fusion。

**Shared latent / VLT-style alignment**：用对比学习 / 蒸馏 / CLIP-style 目标、把不同模态的表示拉进同一 latent 空间。代表工作：**TVL / Binding Touch to Everything** [arXiv:2402.13232](https://arxiv.org/abs/2402.13232)（Zhao 等, ICML 2024）用约 44K vision-touch pairs 通过语言建立跨模态 alignment；**AnyTouch** [arXiv:2502.12191](https://arxiv.org/abs/2502.12191)（Feng 等, 2025）针对 heterogeneous visuo-tactile sensors 学统一的 static-dynamic 表示；**3D-ViTac** [arXiv:2410.24091](https://arxiv.org/abs/2410.24091)（Huang 等, CoRL 2024）在该论文的实验中报告 visuo-tactile representation 相对 vision-only 的显著增益；**Lee 等 Making Sense of Vision and Touch** [arXiv:1810.10191](https://arxiv.org/abs/1810.10191)（ICRA 2019）是这条路线最早的自监督锚点。缺点：**训练监督信号哪里来？** 视觉-语言的对比学习能吃海量网络图文对、四元组对齐没有天然监督源；目前主要靠 (a) 遥操作日志把四路强行同采（Calandra 等 *More Than a Feeling* [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) 是这条思路在触觉抓取上的早期实证）、或 (b) 用语言 / 视觉作为桥梁把触觉拉进 V-L 空间（TVL 走的就是这条）。

**这里要显式说明**：上述工作证明了共享 latent 在特定任务族上可行、但它们的表示仍是 policy-specific 或 dataset-specific 的。本文据此提出——如果想让这些表示跨 policy / 跨 world model / 跨 controller / 跨 sensor 复用、需要在 latent 之上再加一层带单位、坐标系、时间戳、不确定性、来源、有效性的 **structured state contract**。这一层是本文的核心主张、不是那些论文的结论。

**Structured state contract**（本文推荐路线）：不在 raw 层融合、不在 latent 层融合、而是在一个显式约定的中间表示上组合。这个中间表示不是 learned embedding、是一组语义已经收敛、且带 uncertainty / provenance / timestamp / validity 的状态槽位。具体 schema 设计放到 §6 展开。优点：缺一路信号只影响它对应的 key、其他 key 不受污染；时间基、坐标系基、语义基、有效性都收敛在 slot 定义里；slot 一旦定下、fusion 结构可以随便换。缺点：slot 本身要先设计、这活比"套一个 Transformer"重；slot 精度不够时、下游 policy 也学不出超出 slot 的能力；**并且 slot 不解决 vision 提供的 object identity / geometry / free-space / occlusion / scene context 这类"非接触语义"**——本文推荐的是 contact-rich manipulation 场景下的 state contract、不是一般意义上的 multimodal interface。

**这一节落点**：融合范式的谱系里、early concat / attention / late fusion / shared latent 都各有用武之地、但它们的失效模式绝大多数都能追溯到 §2 那三小段没做；本文更倾向于把工程重心从"选哪种 attention"移回"先把 state contract 定下来"。

## 4. 为什么视觉先形成了可复用的数据与表示生态

一个自然的问题：§2 那三小段对视觉同样存在、为什么视觉就能撑起一整个共同数据接口？

**先做一个修正**：**这不是单因果**、是三基固化 + 生态条件共同作用——标准化硬件（CMOS sensor + 统一 lens mount）、统一文件格式（JPEG / PNG / MP4 / HDF5）、坐标模型（针孔 + SE(3)）、大规模互联网数据、成熟标注任务、公开 benchmark（ImageNet / COCO / ADE20K / LVIS）、成熟 encoder、GPU scaling law、可获得的算力。本文只强调三基这一维度、因为这条恰好是触觉目前最缺的。

**时间基上**，视频天然是 frame-indexed、30 或 60 Hz、所有下游任务都约定俗成"以帧为单位"——这个约定消灭了融合讨论里最麻烦的一类问题。**坐标基上**，针孔相机模型 + 内外参把 image pixel ↔ world point 变成一条公式（$s \cdot m = K [R | t] \cdot M$），3D 视觉、SLAM、NeRF、3D Gaussian Splatting、多视角立体都在这个约定上生长；标定错误存在、但"错误长什么样"是可预期、可复现的。**语义基上**，RGB 本身没有语义、视觉社区用 COCO / ImageNet / ADE20K / LVIS 形成了一组**高度互操作的任务级 representation conventions**——object class / bounding box / instance mask / depth / affordance / caption。**更准确地说，这一批数据集并没有形成一个统一 semantic schema**：ImageNet 是 classification ontology、COCO 是 detection + instance + caption、ADE20K 是 scene parsing、LVIS 是 long-tail instance 分布、四者各自定义、只是彼此可组合。

对比触觉：时间基上不同 sensor family 从 30 Hz 到 kHz、事件触发与轮询混合、没有共识；坐标基上 GelSight 是 pixel + 弹性体形变、9DTact 是 pixel + 3D 形变场、阵列 taxel 是一维 / 二维力分布、光学触觉又是另一套表示、连"一个触觉读数到底长什么 shape"都没统一；语义基上接触点、法向、切向、slip、mode 这些概念在论文里都有人用、但缺乏**跨数据集统一的任务级 conventions**。这里也顺便修正一个常见说法——"9DTact 里叫 6D force、GelSight 里叫 shear map、阵列里叫 taxel load、都是同一个物理量不同投影"——**它们包含重叠但不同层级的物理信息**：GelSight 的 shear / deformation map 更接近原始局部形变观测、9DTact 的 6D force 是经过模型反演得到的全局 wrench estimate。两者不是同一个物理量的两种投影、是**观测层与估计层**的差异。这个区别对 §6 slot schema 有直接影响。

**这一节落点（也是论证方式的一处修正）**：视觉生态的成功并不是因为 ImageNet / COCO 提供了一份统一的 multimodal state contract、**也不是因为"视觉先解决了 registration"**——恰恰相反、视觉今天仍然没解决多相机时间同步、外参标定、语义对齐这些问题。真正发生的是——**视觉积累的 convention 足够多、使得那些没被解决的 registration 问题退化成了局部的、可各自为战的工程麻烦、而不是一个系统级的表示障碍**。触觉 / 力觉 / 本体感觉要形成同规模的融合生态、先要做的是**约定接口**、而不是卷模型结构。

## 5. 每一路信号各自的融合难点

这一节把 §2 已经立起来的时间/坐标/语义先放在一边、只谈每个 sensor 特有的建模假设与难点、给 §6 的接口设计做铺垫。

| Modality | Native evidence | Main ambiguity | Canonical slot output |
| --- | --- | --- | --- |
| Vision | Scene geometry, appearance, semantics | Occlusion, view-dependent pose, identity ambiguity | Geometric / semantic observation → contact hypothesis |
| Tactile | Local deformation / force distribution at contact interface | Sensor-family heterogeneity, patch vs point, mounting offset | Local contact observation (patch-aware) |
| Force/torque | Aggregate 6D wrench at sensor frame + reference point | Contact decomposition is non-identifiable; requires model-based compensation | Aggregate wrench measurement + residual |
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

四路 encoder 结构差异巨大：CNN for images、MLP for taxel arrays、Spline model for waveguide、IK for proprioception-inferred。一个常见误解是"做一个 tactile foundation model 统一处理所有触觉传感器"——这正是 AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) 在做的事、方向对、**但它解决的是 learned representation 层的统一、不是 raw 层的统一、也不是 state contract 层的统一**。即便有 AnyTouch、输出仍然是 embedding、还需要一层 "embedding → contact slot" 的显式约定才能进 state contract。**缓解**：raw 到 slot 之间引入一层 **sensor-specific decoder**、把异构触觉输出统一解码到 §6.1 的 slot schema；这一层可以是解析、也可以是学习的、但它必须是**接口的一部分**、而不是藏在 policy 里。

### 5.2 Force/torque：aggregate、依赖大量前置补偿、且不能反推 geometry

F/T 传感器输出一个 6D wrench、格式统一、但它的**语义**要复杂很多：

$$
w_{\text{raw}} \;=\; w_{\text{contact}} + w_{\text{gravity}} + w_{\text{inertial}} + w_{\text{friction}} + b_{\text{bias}}
$$

要提取"外部接触力"、至少要做：零漂补偿（$b_{\text{bias}}$、tare 归零、真机上很难完美）、重力补偿（$w_g = g(q)$、包含工具 + 夹爪质量分布、工具一换整条曲线全变）、惯性补偿、坐标系变换（把 sensor frame 下的 wrench 变到 base 或 world）。

**关于力旋量本身还有一个 frame semantics 的坑**：一个 wrench 不只是"一个 unit + 一个 frame_id"就能定死的。力矩依赖**力矩参考点**：

$$
\tau_{p_2} \;=\; \tau_{p_1} + (p_1 - p_2) \times f
$$

同一个力作用线、换参考点、力矩就变。所以一个 wrench 字段除了 orientation frame、还必须显式带 **reference point**、以及 **wrench 元素顺序**（$[f;\tau]$ 还是 $[\tau;f]$）、**active/passive 变换约定**。§6.1 的 `Quantity` 会为此把 `frame` 展开成一个带 `orientation_frame / reference_point / convention` 的小结构。

**关于惯性补偿这里做一个重要的公式收紧**（v3 版本把 joint-space 动力学与 sensor-frame wrench 混在一条公式里、方向也写反了、reviewer 正确地指出：$\tau = J^T F$ 是从 wrench 映到 joint torque、反过来做是病态的）。更一般、也更诚实的写法不是直接给一个伪逆闭式解、而是把 joint-side 的外部力估计写成**一个带正则的最小二乘**：

- **joint-side estimation chain**：残差力矩

$$\tau_{\mathrm{res}} \;=\; \tau_{\mathrm{meas}} - \hat{\tau}_{\mathrm{model}}(q,\dot q,\ddot q)$$

  其中 $\hat{\tau}_{\mathrm{model}}$ 通常包含 $M(q)\ddot{q} + C(q,\dot q)\dot{q} + g(q) + \tau_{\mathrm{friction}}$。但真实情况是

$$\tau_{\mathrm{res}} \;=\; J^T F_{\mathrm{ext}} + \tau_{\mathrm{null}} + \epsilon$$

  $\tau_{\mathrm{null}}$ 是落在 $J^T$ 零空间、joint torque 观测不到的自运动内力、$\epsilon$ 汇总了 unmodeled friction、actuator / transmission dynamics、compliance、以及关节力矩估计误差。因此**外部力不是一个被"恢复"出来的确定值、而是一个 hypothesis**、写成优化更稳妥：

$$\hat{F}_{\mathrm{ext}} \;=\; \arg\min_{F}\; \big\|\, \tau_{\mathrm{res}} - J^T F \,\big\|_{W}^{2} \;+\; \lambda\, R(F)$$

  在满秩良态时它退化到加权伪逆 $(J^T)^{\dagger}\tau_{\mathrm{res}}$、但在冗余 / 病态构型下需要 regularization 或额外的接触假设、解不唯一。

- **wrist F/T 侧是另一条 chain**：sensor 直接输出 $w_{\text{raw}}$、通过 SE(3) 伴随变换与 sensor-frame 的 bias / gravity / inertial 补偿得到 external wrench $w_{\text{ext}}^{\text{sens}}$、再变到 base 或 tool frame（带 §5.2 上面说的 reference point 变换）。**不要**把两条 chain 混成一条公式——joint-space dynamics term 与 wrist sensor 看到的 inertial wrench 是两类不同的物理量、mapping 方向也不同。

慢速操作时、joint-side residual 常常足够；高速操作时、inertial 与 actuator dynamics 都不可忽略。**这一层任何一步出错、下游的"法向力 / 切向力分解"就整个错**。

F/T 还有一个更本质的限制——**它给出的是"整体"力旋量、不是"局部"接触**。本文把这条列为**三条 physics principle 之一**：

$$
\boxed{\;\text{Physics principle 1 — Force/torque measures an aggregate wrench; it does not, on its own, uniquely decompose into individual contacts.}\;}
$$

$$
w \;=\; \sum_{i=1}^{N} \begin{bmatrix} f_i \\ (p_i - p_0) \times f_i \end{bmatrix}
$$

无穷多组 $\{p_i, f_i\}$ 可以给出同一个 $w$。这不是"估计器不够强"、而是**逆问题从 F/T 单独看是非可辨识的（non-identifiable）**（§6.1.4 会展开 identifiability）。**需要限定**：在给定 contact geometry、robot kinematics、object geometry 与 contact model 的条件下、F/T 可以**间接**产生较强的 localization constraint（例如"只有一个候选接触点时、F/T 就能把力值定位到该点"）；问题不是"F/T 不能定位"、而是**"F/T 单独不提供唯一的 contact decomposition"**。这一条对 §6.2 的 slot 语义有直接影响。

**缓解**：F/T 在 slot 层应该输出两件事——(a) 已补偿的 external wrench（用于 force-aware policy）、(b) 残差 magnitude（用于检测"是不是又漂了 / 是不是又撞到不该撞的东西"）。第二件事常常被忽略。

### 5.3 Proprioception：最直接、也最容易被过度依赖的一路

Proprio 是四路里默认时间常数最快的、也**通常在常规刚性机器人上是最容易获得的高速率内部参考**——这一句需要限定、"完整通道 / 天然主时钟"是不准确的：很多机器人没有直接 torque sensing、$\dot q$ 往往是数值微分、EE pose 往往是 FK 估计、motor current 与 joint torque 之间有摩擦 / 齿隙 / 传动比映射、某些软体机器人、分布式架构或网络化平台甚至有多个 control clock、proprio 也可能被 delay / estimate / filter 过。更稳的表述是——**proprioceptive state is often the most readily available high-rate internal reference on conventional rigid robots**、它是**最容易标准化、也最常拿来当作坐标系与时间基 anchor 的一路**、而不是"天然的全系统主时钟"。

Proprio 在融合里通常有两个作用：**作为 contact hypothesis 的输入**——给定期望末端轨迹 + 关节力矩反馈、可以按 §5.2 的 joint-side chain 反推 external wrench hypothesis（contact inference、Hogan 早年阻抗控制一脉的经典工具）；**作为时间基与坐标系的锚**——融合层需要一个稳定的 frame、proprio 通常跑在硬件 servo 允许的高速率、EE pose 也常当作其他坐标系的 anchor。**融合难点**：proprio 的信息量太"干净"——它是机器人自己测自己的状态、没有外部世界的不确定性、模型很容易**过度依赖 proprio**、把它当作 shortcut、结果在没有接触的场景表现好、有接触的场景反而没学会用触觉 / F/T。这一条正是 《触觉·力控》 §5.2.1 feedback-value ablation 想避免的。

### 5.4 时间常数差异本身就是建模假设

把 §5.1–5.3 与 §2.1 合起来看、可以提炼出这条判断：**不同模态的 effective observation rate 差异、不是"融合层要不要处理一下"的工程问题、而是"任务需要什么控制带宽"的建模假设**。擦拭 / 打磨这类任务力控带宽至少 100–500 Hz、视觉 30 Hz 完全够用、触觉与 F/T 得走 native rate、proprio 也得到力矩层——这一类任务、融合层不能把所有信号降到 30 Hz；抓取-放置 / 装配这类任务视觉主导、接触事件稀疏、把触觉降采样到 vision rate 是可接受的近似。具体数值随硬件、任务、controller 架构变化、这里不写死。

## 6. 一个最小可用的 structured state contract

到这里可以正面回答"到底怎么办"了。这一节给一个**可实施、可 benchmark、可增量演进**的最小接口。

**先做一个范围收紧**：本节要提的不是 "所有机器人多模态信息都应该压缩成 contact set"、也不是 "structured state contract ≈ contact set"。**Contact set 是这份契约在 contact-rich manipulation 场景下的一个核心实例、不是契约本身的定义**——这一句决定了 locomotion / navigation / whole-body manipulation 的读者能不能把自己的状态塞进同一份 contract。契约本身至少包含（注意 contact_state 只是其中一个 manipulation-specific 分支）：

```text
Structured State Contract
├── robot_state      ── (q, q̇, τ, EE pose) + availability
├── scene_state      ── free-space / object pose / occlusion …（可选）
├── task_context     ── language token / goal embedding
├── events           ── contact / slip / impact / mode-switch 等稀疏事件
├── wrench           ── 6D + reference point + covariance + validity
├── contact_state    ── §6.1 详细展开、manipulation-specific 实例
└── uncertainty / metadata ── 贯穿所有字段的 unit / frame / provenance / validity
```

用集合语言表达就是 $\text{ContactSet} \subset \text{StructuredState}$、而不是 $\text{StructuredState} \approx \text{ContactSet}$。对没有接触语义的任务（例如纯导航）、contact_state 分支可以整块缺席、契约仍然成立。**contact set 是实例、契约是 contract。**

### 6.1 Contact slot schema（一个 manipulation-specific 实例）

先定义跨模态共享的**接触事件记录**（带 uncertainty / provenance / timestamp / validity）：

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
        "position":        Quantity,         # {value: Vector3, unit: "m", frame: ...}
        "normal":          Quantity,         # {value: Vector3, unit: "-", frame: ...}
        "extent":          Optional[Quantity],  # 若 type ≠ point
        "center_of_pressure": Optional[Quantity],
    },

    # 力旋量（canonical point / patch summary）—— 必须带 reference point + convention
    "wrench": {
        "f_perp":          Quantity,         # {value: float, unit: "N", frame: ...}
        "f_parallel":      Quantity,         # {value: Vector2, unit: "N", frame: ...}
        "moment":          Optional[Quantity],  # 若 type ≠ point
    },

    # 模态与事件
    "slip_probability":    float,           # ∈ [0, 1]
    "mode":                enum,            # free / touch / sticking /
                                            # sliding / rolling / separating

    # 不确定性：连续与类别必须分开、且要匹配状态变量的拓扑
    "uncertainty": {
        "pose_covariance":      Matrix,     # Σ on ℝ³ or SE(3)
        "force_covariance":     Matrix,
        "slip_probability":     float,      # 标量、非 covariance
        "mode_probability":     Vector,     # categorical distribution
        "count_uncertainty":    Optional[Distribution],  # N_t ∈ {0,1,2,...} 是
                                            # combinatorial、不是单个 covariance 能表达
        "calibration_quality":  enum,       # nominal / degraded / unknown
        # 备注：单个 Gaussian covariance 不足以表达 multimodal posterior
        # hypothesis；covariance 也不区分 measurement noise / model
        # uncertainty / calibration uncertainty。需要更细归因时拆
        # aleatoric / epistemic / calibration、或至少记录 σ_sensor、
        # σ_model、σ_calibration。
    },

    # 可观测性：不是所有 state 都从四路 observation 可观测
    "observability":       enum,            # observable | partially_observable
                                            # | unobservable

    # 来源（provenance）：对象化、不再是单个 enum
    "provenance": {
        "primary_sources":      ["tactile", "force_torque"],   # 主贡献源
        "contributing_mask":    [V, T, F, P] → [0/1, 0/1, 0/1, 0/1],
        "negative_evidence":    ["vision"],   # 明确"排除了某 hypothesis"的路、
                                            # 也是证据（见 §6.1.3）
        "correlated_with":      ["proprio"],  # 与本 slot 证据共享 latent、
                                            # 防止 double counting（见 §6.2.1）
        "estimator":            "contact_estimator_v2",
        "calibration_version":  "...",
    },

    # 时间与生命周期（§6.1.1）
    "timestamp":           float,           # host clock、seconds
    "age":                 float,           # seconds、freshness
    "valid_from":          float,
    "valid_until":         float,
    "availability":        enum,            # present / stale / delayed /
                                            # unavailable / corrupt（= 能不能拿到）
    "validity":            enum,            # valid_now / valid_for_past /
                                            # conditionally_valid / invalid（= 能不能信）
    "lifecycle":           enum,            # new / tracked / occluded /
                                            # lost / merged / split
}
```

**`Quantity` 的 frame 展开**：schema 里所有 numeric 字段类型都是 `Quantity = {value, unit, frame, ...}`、unit 用 SI canonical 记法（米、牛顿、牛·米、秒、弧度）。但对 wrench / moment 这类几何量、只给一个 `frame_id` 不够、`frame` 需要是一个小结构 `{orientation_frame, reference_point, convention}`——因为力矩依赖参考点（§5.2 的 $\tau_{p_2} = \tau_{p_1} + (p_1-p_2)\times f$）与 $[f;\tau]$ / $[\tau;f]$ 顺序约定。这一层不是"语义上要求有 unit"、是"schema 可以机器验证 unit / frame / reference point 组合"——§6.8 的 runtime validator 会 assert 每一条 slot 的这套元数据。

这份 schema 刻意做到几件事：**sensor-agnostic**（GelSight、taxel 阵列、F/T——只要能填这些 key、就能进 state contract）；**物理可解释 + 单位明确**（每一个数字有明确 SI 量纲与 frame / reference point）；**带 uncertainty**（连续 covariance 与类别概率分开、uncertainty 表征要**匹配状态变量的拓扑**——contact count 是 combinatorial、multimodal 假设不能塞进单个 Gaussian、并保留 aleatoric / epistemic / calibration 拆分接口；**没有 uncertainty 的 slot 不是接口、是"事实"、真机上永远不成立**）；**带 observability**（下面 §6.1.4）；**带 provenance**（primary_sources + contributing_mask + negative_evidence + correlated_with + estimator + calibration_version——与 §7.5 modality dropout、§8.6 contradiction、§8.7 consistency graph 直接对应）；**带 availability / validity / lifecycle**（下面 §6.1.1 展开）。

#### 6.1.1 availability 与 validity 是两条不同的轴

上一版把 staleness 塞进 `availability` 一个字段、容易语义冲突。这一版明确拆成三条正交的轴：

```text
availability = 能不能拿到这条观测    (present / stale / delayed / unavailable / corrupt)
validity     = 在什么时间/模型假设下能信它 (valid_now / valid_for_past / conditionally_valid / invalid)
age / freshness = 它有多旧          (seconds)
```

于是 "present + stale + valid_for_past" 是一个自洽组合：**一条 stale 观测现在仍然拿得到（availability=present）、对过去的状态仍然 valid（validity=valid_for_past）、只是对当前状态未必 valid**。本文统一口径——**staleness is not invalidity; it is validity relative to a past timestamp**。对 controller 而言三态处理也不同：stale-but-valid-for-past 应做 hold + covariance inflation + fallback；fresh-uncertain 继续消费但降权；unavailable 走 §6.6 的显式退化路径。**这一层是接口设计本身的要求、不是 benchmark 技巧**。

#### 6.1.2 track_id 是一个 hypothesis、不是内禀属性

v2 把 id 写成"tracking ID、跨帧一致"过乐观。contact set 本质上是 **permutation-invariant set**、不天然存在稳定 identity：多指操作、rolling contact、contact patch splitting / merging、遮挡、瞬态接触都会让 identity assignment 本身成为 inference problem。从 v3 起把字段改成 `track_id: Optional[int]` + `track_confidence: float` 并明确：

> **track_id is a hypothesis maintained by temporal association, not an intrinsic physical property of a contact.**

没有 track_id 时 slot 依然有效；有 track_id 时它是一个可被下游消费的、带 confidence 的假设。`track_confidence` 也可以被 lifecycle 字段部分覆盖（例如 `lifecycle = "split"` 意味着两条子 slot 的 track_confidence 应该同步下调）。

#### 6.1.3 provenance 还要能表达"负面证据"

`primary_sources` 记的是"支持这个 contact 存在"的路、但某一路明确**排除了某个 hypothesis** 时它同样是证据。例如 vision 报 "no visible contact"、tactile 报 contact、F/T 报 wrench inconsistent——这里 vision 并不是 contributing source、但它提供了 negative evidence。因此 provenance 里保留 `negative_evidence` 字段（概念上、字段可后置）。它让"证据"不再是"谁赞同"、而是"谁提供了对某假设的似然更新（正或负）"。

#### 6.1.4 observability、identifiability 与"别把欠定当成一个点估计"

接口定义得再漂亮、也要承认一个事实：**不是所有 state 都能从四路 observation 观测到**。F/T 单独看、contact decomposition 是 unobservable / non-identifiable（§5.2）；视觉遮挡下 contact location 只是 partially observable；触觉对 object identity 往往 weakly observable。所以 structured state 不只是 $(\hat{x}, \Sigma)$、还隐含**observability / identifiability**——§6.1 里用一个 `observability ∈ {observable, partially_observable, unobservable}` 至少把它记下来。

更进一步：**inverse problem 不可辨识时、接口应允许多个 hypothesis 并存、而不是强迫输出一个单一 contact point**。这正是 contact set（一个 set、可以是多解）相对"单点估计"的价值所在。要显式写一句以免被误读：

> **A single Gaussian covariance is not sufficient to represent multimodal posterior hypotheses; the uncertainty representation should match the topology of the state variable (continuous, categorical, combinatorial, or set-valued).**

这条也强化了 §5.2 那条 physics principle 的另一面——**structured belief ≠ hallucinated point estimate**。

### 6.2 每一路信号怎么映射到 slot · F/T 是 constraint、不是 detector

四路合并的信号 → slot 语义映射：

```text
Tactile         → local contact observation      （直接观测、有 sensor noise）
F/T             → global wrench measurement      （aggregate 独立测量、不定位、不分解）
Proprioception  → kinematic / dynamic constraint （经 §5.2 argmin、依赖模型精度）
Vision          → geometric / semantic observation
                  → contact hypothesis / prior    （既是观测、也产生预测）
```

一个**说明性（illustrative）** 的 formulation 是 Bayesian 分解——但请注意这只是把证据融合关系写清楚的脚手架、不是本文要主张的算法：

$$
p(C_t \mid O_{1:N}) \;\propto\; p(O_{1:N} \mid C_t)\, p(C_t)
$$

其中 $p(O_{1:N}\mid C_t)$ 里的 dependency structure 可以按系统需要选成下面这种朴素分解、或任意图模型 / 学习式结构：

$$
p(O_{1:N}\mid C_t) \;=\; p(V\mid C_t)\,p(T\mid C_t)\,p(F\mid C_t)\,p(P\mid C_t)\quad (\text{one illustrative factorization})
$$

**三个必须写下来的 caveat**——

**(i) 本文不是在提 Bayesian fusion。** 上面这组公式只是 illustrative factorization。真正的一般写法是 $p(C_t\mid O_{1:N})\propto p(O_{1:N}\mid C_t)p(C_t)$、其中 $p(O_{1:N}\mid C_t)$ 可采用任意 graphical / learned dependency structure。把它写成条件独立乘积只是为了让证据融合关系一眼可读、并不意味着本文的贡献是"probabilistic contact estimation"。

**(ii) Conditional independence 是简化、而且会 double counting。** 上面这个 factorization 隐含 $p(V,T,F,P|C) = p(V|C)p(T|C)p(F|C)p(P|C)$。真实机器人里四路之间存在大量共享 latent：robot pose、object pose、contact geometry、dynamics、calibration、actuator state。**最典型的一个坑**：proprio 反推出来的 external wrench（$\hat F_{\mathrm{ext}}$ from $\tau_{\mathrm{res}}$）和 F/T 直接测的 external wrench、共享同一批物理证据、把 $p(F\mid C)\,p(P\mid C)$ 直接相乘很可能**把同一份证据数了两遍（double counting）**。正确的做法是**显式追踪 correlated evidence / shared latent variables**——要么用 joint Gaussian likelihood + shared covariance、要么在 graphical model / factor graph 上让这两条 observation 连到同一个 wrench latent、并在 §6.1 provenance 的 `correlated_with` 字段里标注出来。这也正是 §8.7 consistency graph 存在的理由之一。

**(iii) 接口不规定推理算法。** **The interface does not prescribe a Bayesian inference algorithm. "Belief" here denotes uncertainty-aware state information; Bayesian posterior inference is one implementation, not a requirement of the schema. The contribution is not a new estimator — it is a contract over estimator outputs.** 同一份 §6.1 slot schema 可以由 EKF / UKF、factor graph、particle filter、learned filter、diffusion state estimator、Transformer state estimator、hybrid neural-symbolic estimator 产生——只要输出满足 §6.4 那条 7-tuple 公式（Value / Semantics / Frame / Time / Uncertainty / Provenance / Validity）。这一句让接口从"某个 probabilistic architecture"变成真正的"contract proposal"。

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
Latent representation             Structured state contract
─────────────────────────         ─────────────────────────────
优化目标：对下游模型友好            优化目标：对多个消费者互操作
单位：隐式                          单位：Quantity 里显式记 canonical unit
坐标系：隐式（藏在数据流里）        坐标系：显式声明、字段附 frame + reference point
时间：隐式                          时间：显式 timestamp + age + latency
有效性：无                           有效性：availability / validity / lifecycle
不确定性：隐式或仅在损失里          不确定性：连续 covariance + 类别分布 + 拓扑匹配 + calibration metadata
来源：无                             来源：primary + contributing + negative + correlated
消费者：一个 model                   消费者：policy / world model / controller /
                                       diagnostic tool / safety layer / 另一个 sensor
```

一句可以直接留下来的公式：

$$
\text{Contract} \;=\; \big(\, \text{Value},\;\text{Semantics},\;\text{Frame},\;\text{Time},\;\text{Uncertainty},\;\text{Provenance},\;\text{Validity} \,\big)
$$

**这一节的两处重要限定**——

**(a) 不是说 "latent 天生没有 uncertainty / provenance"。** 现在完全可以有 probabilistic latent、uncertainty-aware latent、timestamped latent、multimodal latent。本文要区分的是：

$$
\text{latent} \;\not\Rightarrow\; \text{contract}\qquad(\text{而不是}\quad \text{latent} \Rightarrow \text{no uncertainty})
$$

准确的说法是——**a generic learned latent does not *guarantee* these semantics unless they are explicitly encoded and standardized across producers and consumers**。一个 latent 可以非常适合某个 policy、却不被保证能作为 world model、controller、diagnostic tool、另一个 sensor 的稳定输入接口——这才是"接口"存在的必要性、也是 §3.4 那些 shared latent 工作真正缺的那一环。

**(b) structured ≠ symbolic-only。** 这份 contract 的消费者完全可能是神经网络。structured state 可以被再 embedding 之后喂给 policy、也可以直接喂给 controller：

```text
structured state → neural embedding → policy
structured state → controller
```

明确这一点、是为了挡掉"你是不是把 learned robotics 拉回 hand-designed symbolic state"这种攻击——**本文主张的是"跨消费者共享状态要有显式、标准化的边界"、而不是"这个边界必须是人类可读的符号"**。

### 6.5 融合的时机：reusable state 应在 contract 上暴露（而不是"所有 fusion 必须发生在这里"）

**这一节是相对上一稿口径的一次重要降级**。上一稿写"融合的时机：不在 raw、不在决策、在 structured state contract 上"——这说得过绝对了。现代多模态学习完全可以在 raw / latent / decision 多个层级同时融合：learned latent fusion、neural implicit state estimator、end-to-end visuomotor policy、diffusion policy、world-model latent fusion 都可能比 handcrafted slots 更强、而且它们并不需要先把 $C_t$ 收敛成显式 contact position 才能拿到性能。

所以本文真正主张的是一个**弱得多、但强得多**的命题：

> **Reusable multimodal state should be exposed at the structured state contract, while learned fusion may still occur before, within, or after that interface.**

中文：**本文并不规定所有 fusion 必须发生在 structured state contract；本文主张的是——凡是需要被多个下游消费者共享、诊断、校验、或跨 embodiment 复用的 multimodal state、应当在这一层显式暴露。** 有了 §6.1 的 slot schema 之后、一个典型的组合是三步：

```python
# Layer 1+2: per-modality perception + registration + semantic projection
raw_v  → enc_v → pred_contact_hypothesis      （视觉的 contact 预测、附 confidence）
raw_t  → enc_t → detected_contact             （tactile 观测、附 pose_covariance）
raw_ft → enc_ft → external_wrench + residual  （补偿后 wrench、附 force_covariance）
raw_p  → τ_res → argmin_F ‖τ_res − JᵀF‖²_W + λR(F) → inferred_contact（带模型误差）

# Layer 3+4: hybrid state estimation → 写到 state contract
# （Bayesian / EKF / factor graph / learned filter 都是可选实现）
C_t         = state_estimator(V, T, F, P)   # 输出的是 belief、不是 hard fact
wrench      = compensate_ft(raw_ft)
robot_state = (q, q̇, τ, EE_pose) + availability
contract_t  = assemble(C_t, wrench, robot_state, task_ctx)   # 一份 contract

# consumers: policy / world model / controller / diagnostics 消费 contract
a_t = policy(contract_t)   # policy 内部仍可再做 learned fusion
```

关键判断——**Layer 4 的 contract schema 才是整个多模态系统的跨消费者接口**。Layer 1–2 各模态可以任意换 encoder、Layer 3 可以任意换 estimator、消费端可以任意换 policy 结构、只要 Layer 4 的 schema 稳定。同时——**state contract 是系统的稳定互操作边界、但不必成为所有下游模型唯一的信息通路**：policy 完全可以同时接收 structured slots + 额外的 raw visual feature / language token / task embedding。真正需要守住的是**跨消费者共享的那部分状态**走这份 contract、而不是"所有 fusion 都必须穿过这里"。

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
        r.validity         = "invalid"        # 该路本身没有可信值
```

**这不是工程补丁、这是训练时的核心约束**。见 §7.5 modality dropout、§8.5 degradation modes、§8.6 cross-modal contradiction test。

### 6.7 wrench（measurement）与 contacts[]（hypothesis）：不是冗余、是 measurement ↔ hypothesis 的 consistency layer

一个 reviewer 会立刻问的问题：如果 `contacts[]` 里已经有每个 contact 的 force 与 moment、为什么还需要一个 global wrench？答案是——

$$
w_{\text{ext}} \;\neq\; \sum_i w_i^{\text{reconstructed}}
$$

**`wrench` 是独立的 aggregate measurement**（F/T 传感器直接给出的整体力旋量）、**`contacts[]` 是 structured hypothesis**（每路感知 + state estimator 联合推断的结果）。二者是两类不同的量、之间产生残差：

$$
\mathcal{C}_{\text{wrench}} \;=\; \big\| w_{\text{ext}} - \textstyle\sum_i \big[f_i;\, (p_i - p_0) \times f_i\big] \big\|_{\Sigma^{-1}}
$$

这个 residual 越大、说明 contact hypothesis 与 F/T 观测越不一致、系统应该 raise 一条 consistency alarm、或者把 `mode_probability` 拉向 uniform。这一点很值得强调：正是这条 measurement ↔ hypothesis 的残差、把 §0 里那句 **Design for disagreement** 从一句口号、变成了一个**数学对象**。所以——**`wrench` 不是 `contacts[]` 的冗余字段、而是一个独立的 aggregate measurement、可以作为 contact-set reconstruction 的 consistency constraint**；这也是 §8.7 consistency graph 里 F↔T、F↔V、F↔P 三条边的物理基础。

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
├── unit / frame validator # schema metadata 定义每一个字段的 canonical unit、
│                          # frame、reference point；runtime assert；Quantity
│                          # {value, unit, frame{orientation, ref_point, conv}}
│                          # 是这份 validator 的落脚点
└── graceful degradation   # 缺字段时的 inflation / default / alarm 策略
```

没有这一层、"state contract = API" 的类比就不成立。这一层也是 §8.1 schema-swap benchmark 的前提。

## 7. 融合失败模式与诊断

reviewer 视角、下面这五类失败在机器人多模态工作里最常见、也最容易被"我们用了 cross-attention 所以鲁棒"这类话糊过去。每一类统一四段：**Failure mode → Observable symptom → Diagnostic test → Mitigation**。

### 7.1 Modality collapse：policy 悄悄退回视觉 / proprio 主导

- **Failure mode**：某些模态在训练里被边缘化、变成事实上的 dead input。
- **Observable symptom**：训练 loss 与 validation success rate 都正常；一旦评测遮住视觉、policy 表现几乎不变。
- **Diagnostic test**：主指标 **modality marginal contribution**（§8.4）——若 $\Delta_{\text{mod}} \approx 0$、说明这一路在 policy 里其实没被消费。3D-ViTac [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) 的 ablation 之所以重要、就是给这个指标提供了一个明确的参照（**该论文实验中报告的 visuo-tactile > vision-only 增益、是这种贡献存在性的证据、不是普遍定律**）。Attention 权重可视化只能算 **辅助 diagnostic visualization**、不能拿来判断 modality contribution——**attention ≠ causal importance**、一个模态 attention 权重低不代表它没有贡献、一个模态 attention 高也不代表去掉它性能一定下降；这是经典 interpretability 陷阱。
- **Mitigation**：(a) 训练时显式跑 modality dropout、见 §7.5；(b) 每个模态加 auxiliary supervision（例如 tactile encoder 除了给 policy 用、还得独立预测 slip 事件）；(c) 用 《触觉·力控》 §5.2.1 的 feedback-value benchmark 而不是纯 success rate 逼出真实贡献。

### 7.2 Temporal smearing：所有信号被强行插值到 30 Hz

- **Failure mode**：高频物理事件被降采样抹掉。
- **Observable symptom**：滑移检测、瞬态接触、冲击类任务的 policy 学不出来。
- **Diagnostic test**：对比"以 30 Hz 融合 vs 以 native rate 融合"的成功率差；或者用 $\Delta_{\text{tail}}$ 只看困难接触条件下的表现。
- **Mitigation**：见 §5.4——把时间常数差异当**建模假设**；不同层用不同频率、事件走 event bus；不要在接口层把高频信号降采样掉。

### 7.3 Frame confusion：模型学到坐标系伪相关

- **Failure mode**：模型学到的是某种特定传感器 frame 下的相关性、不是任务本身。
- **Observable symptom**：模型在训练位姿、训练相机摆放、训练工具型号下表现优秀、稍微动一下就崩。
- **Diagnostic test**：评测集主动做**坐标系扰动**（四类细分见 §8.9）。
- **Mitigation**：state contract 里显式带 `frame`（含 reference point）、下游消费者必须做变换才能读；训练时把坐标系扰动当 domain randomization 的一部分（Tobin 等 [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)）。

### 7.4 Semantic leakage：raw 视觉路径绕过 slot 定义接触语义

- **Failure mode**：接触状态的定义被未约束的视觉 latent 悄悄改写。
- **Observable symptom**：slot 明明定义成"接触几何"、但 policy 表现严重依赖视觉外观（同一个物体的照片换背景就掉点）。
- **Diagnostic test**：把 slot 里的视觉贡献换成一个"最小充分"的合成 slot（例如把预测 contact 用真值 contact 替）、看性能变化。
- **Mitigation**：本文**不是**主张"raw pixel 不允许直接进 policy"（现代 VLA 本来就是 image → vision encoder → token representation → policy、raw visual latent 进 policy 是常规操作）；本文主张的是——**对于由 state contract 定义的 contact-related state、不应存在一条绕过 slot encoder、且不可诊断的 raw-visual shortcut**。也就是说"contact 语义的规范通路是 slot encoder"、而不是"视觉特征不许进 policy"。这条与 《触觉·力控》 §3.5 一致、也更可实施。

### 7.5 Missing / degraded modalities：一坏就崩、或者悄悄退化没报警

- **Failure mode**：训练分布里模态永远齐全、上线时的退化不在分布内。
- **Observable symptom**：触觉掉帧一次整个 policy 行为剧烈变化、或悄悄退化没报警。
- **Diagnostic test**：跑 **modality masking / dropout test**（按 (1-p) 概率随机丢掉某一路）；modality masking / dropout 是 missing-modality robustness 中**常见的一类训练策略**、可参 Maiga 等 MMP [arXiv:2410.03010](https://arxiv.org/abs/2410.03010)。
- **Mitigation**：接口层要能表达下面 §8.5 那六类退化、不能只有"有 / 无"两个 flag。§6.1 的 `availability / validity / age / lifecycle / uncertainty` 就是为了让这些状态在数据格式上可检测——**这是接口设计本身的要求、不是 benchmark 的技巧**。

## 8. Interface Property Benchmark 与评估

这一节把 §7 里散在各处的诊断指标整合成一个可执行的 benchmark 骨架。本文把它命名为 **Interface Property Benchmark**——因为它测的不是"这个 fusion 模型成功率多高"、而是"这份接口作为抽象边界成立不成立"。整体思路沿用 《触觉·力控》 §5.2.1 的 feedback-value 主张：**benchmark 应该测接口是不是真的抽象边界、模态是否真的不可替代、系统能否正确处理 disagreement、不是"这个模型能不能记住训练分布"**。

一个贯穿本节的方法学立场要放在最前面：**接口价值的证据不应只是 task success**。$\Delta_m$、success rate 这些只能说明"某个模态有用 / 某个 pipeline 跑得动"；真正证明"这是一份接口、不是命名游戏"的、是 **interchangeability（可替换）、degradation（可退化表达）、diagnostics（可诊断）、cross-consumer reuse（可复用）** 这四类属性。下面 §8.1 / §8.2 分别对应这些属性。

### 8.1 Interface & schema swap（interoperability benchmark、本文第一主实验）

如果 interface 是 abstraction boundary、那么换 encoder / 换 consumer 应该不需要重新设计中间层。四组具体实验：

1. **Sensor encoder swap**：tactile encoder A → encoder B（不同 backbone、甚至不同 sensor family）、slot schema 不变、policy 权重不变、看性能迁移。
2. **Policy swap**：MLP policy → Transformer policy、slot 不变、看下游训练成本。
3. **Consumer swap**：policy ↔ world model ↔ controller ↔ diagnostic tool 四路消费者共享同一份 slot、各自独立训练、看接口是不是真的"可复用"。
4. **Schema swap / evolution**：这一组是 v5 新增、也最关键——不只是换 encoder、而是换 **schema 版本**。例如 v1（position + force）↔ v2（position + force + covariance + provenance）；分两个方向测：**old consumer + new producer**（旧消费者能不能安全降级读新 producer 的字段）与 **new consumer + old producer**（新消费者能不能对缺失字段补 default / inflation）。这才是真正的 **interface compatibility** 检验；只跑 encoder swap 的话、证明的其实是 representation transfer、不是 interface compatibility。

若这四组都能做到"换 encoder / 换 consumer / 换 schema 版本不需要重新设计中间层"、这个实验比再跑一个 success rate 强得多——**它把"接口"从 metaphor 变成 measurable property**。§8.1 与 §6.8 是一对：schema versioning 保证向前兼容、swap 保证横向可插拔。

### 8.2 Oracle-slot / Estimated-slot / End-to-end：三组受控 baseline（明确各自测什么）

要检验"state contract 是否必要"、必须做三组 baseline——**但更要把它们的角色与 information budget 说死**、否则 $\Delta$ 会被误读：

```text
A · Oracle interface   : ground-truth contact slots → policy
B · Estimated interface: sensor → state estimator → estimated slots → policy
C · End-to-end fusion  : sensor → fusion policy（不显式接口）
```

**正式定义三个量的语义**：$S_A$ = "以理想 state 为条件时、下游 policy 的性能上限"（upper bound conditioned on ideal state）；$S_B$ = 真实接口 pipeline（estimation + interface）；$S_C$ = unconstrained end-to-end baseline。于是应当分别读：

$$
S_A - S_B \;=\; \text{state / interface estimation gap}\qquad
S_B - S_C \;=\; \text{explicit interface 相对 end-to-end 的净价值}
$$

**这里要主动堵一个 reviewer 会打的洞**：$\;S_B > S_C\;$ **本身并不证明接口"必要"**——也可能只是 slot 更易学、oracle supervision 更强、architecture capacity 不公平、或 end-to-end baseline 没调好。所以三组对照必须**控信息预算**（尽量对齐 trainable params / 训练样本 / supervision access）、并且把接口价值的主要证据放在 §8.1 的 interchangeability 与 §8.5–§8.7 的 degradation / diagnostics 上、而不是压在一个 success-rate 差值上。**$S_{\text{total}} \approx S_{\text{interface}} \times S_{\text{downstream}}$ 这个因式分解、是把 estimation gap 与 downstream 能力分开的工具、不是"接口更好"的单一证明。**

### 8.3 Slot 层的 fidelity 单独测

**slot 本身的准确度就应该独立于 policy 测**：contact position 与真值的 IoU / distance error；slip detection 的 AUROC；contact mode 分类的 macro-F1；**uncertainty calibration**——连续部分用 reliability diagram 或 negative log-likelihood、类别部分用 Brier score 或 expected calibration error（ECE）；patch 场景加 extent / center-of-pressure error；track_id 加 identity preservation（ID-switch 次数、MOTA / IDF1）。这些是"融合层之前的质量指标"、它们不合格、policy 层的成功率高只可能是过拟合。

### 8.4 Modality marginal contribution 与 graceful degradation

**边际贡献**（每个模态带来的性能增量）：

$$
\Delta_m \;=\; S(M) - S(M \setminus m)
$$

**术语修正**：$\Delta_m$ 测的是 **marginal performance contribution**、不是信息论意义上的 information gain。真正的 information gain 应是条件互信息 $I(X_m; Y \mid X_{-m})$；若要报 MI、另行定义。本节把它正式命名为 **Modality marginal contribution**。

**Graceful degradation**（部分损失下的退化形状）：

$$
G_m(p) \;=\; \frac{S(M, p) - S(M \setminus m)}{\Delta_m}
$$

**符号与适用域先明确**——$S(M, p)$ 表示"模态 $m$ 以独立 **masking/dropout** 率 $p$ 被丢时的性能"、规定 $S(M, 0) = S(M)$；于是 $G_m(0) = 1$、$G_m(1) = 0$。**这条归一化曲线只对 pure masking / dropout 定义**——它隐含 $S(M, p{=}1) = S(M\setminus m)$、而 stale / biased / corrupted 都不等价于 missing、所以那些退化模式在 §8.5 里单独定义、不套进这条 $G_m(p)$。一个"好系统"可以是 $\Delta_m$ 很大（这个模态不可替代）、但 $G_m(p)$ 从 1 平滑降到 0（部分丢失不会立刻崩）。**同时要明确**：不要暗示好系统的 $G_m(p)$ 一定单调或平滑——现实中常有 threshold effect。更准确的表述是——**graceful degradation should be characterized rather than assumed monotonic or smooth**、报告形状、不预设形状。

### 8.5 Degradation modes beyond dropout：把 masking ≠ missing ≠ stale ≠ latency ≠ bias ≠ corruption 分开测

$\Delta_m$ 与 $G_m(p)$ 都建立在"随机独立丢一路"上、但**真实 sensor 退化远不止 dropout**。本节把测试从 Bernoulli masking 扩成一条明确的层级、因为**它们对 estimator 的统计意义完全不同**：

$$
\text{Random masking} \;\neq\; \text{Missingness} \;\neq\; \text{Staleness} \;\neq\; \text{Latency} \;\neq\; \text{Bias} \;\neq\; \text{Corruption}
$$

举个最尖的对比：$p(y_m)=0$（该路彻底没了）与 $y_m = y_{\text{true}} + b$（该路有系统性偏置、但看起来"正常")对 inference 是两个完全不同的问题。接口真正有价值的地方、恰恰是它的 `availability / validity / age / uncertainty / provenance` 能**把这六种区分表达出来**。对每一类退化分别报成功率-曲线。**这才是"接口有没有真的处理 uncertainty / provenance / validity"的直接检验**（对应 §6.1.1、§7.5）。

### 8.6 Cross-modal contradiction：用 uncertainty-aware disagreement、不要只用裸距离

**Good fusion $\neq$ agreement**。这一节是全文最有原创潜力的一块：本文评价融合、不是看"它会不会把信息合起来"、而是看"**当信息彼此矛盾时、它的行为是否正确**"。

上一稿把 contradiction 定义成两个预测的裸距离 $D_{ij}(z) = d(p_i(z), p_j(z))$——但**两个 posterior 的均值不一致、并不等于有 fault**。例如 vision 不确定性很大、tactile 很小、两者 mean 差一点——这也许完全在预期噪声范围内、不是矛盾。因此本文把 disagreement 改成**不确定性归一化的度量**、Gaussian 情形下就是 Mahalanobis 型：

$$
D_{ij} \;=\; (\mu_i - \mu_j)^{\top}\,\big(\Sigma_i + \Sigma_j\big)^{-1}\,(\mu_i - \mu_j)
$$

只有当 $D_{ij} \gg 1$、才构成**统计上有意义的 contradiction**。非 Gaussian / 类别情形用对应的 predictive divergence（比如 $\mathrm{KL}$ 对称化、或 Bhattacharyya / Hellinger）替代 $d(\cdot,\cdot)$。三类常见实例（作用在归一化度量上）：

```text
Spatial disagreement    vision vs tactile 在 contact position（5/10/20 mm 三档、按 Σ 归一）
Force disagreement      tactile 报 2 N 法向、F/T-consistent reconstruction 需要 8 N
Temporal disagreement   F/T 报 contact start 在 t、tactile 报 t + Δ（按 latency 归一）
```

（"vision 说 rigid · tactile 说 compliant"这种跨抽象层的表述不算 contradiction——一个物体可以整体刚性 + 表面柔性。）

然后测四件事：(1) **Contradiction detection**——模型有没有报出"这里跨模态在统计上不一致"；(2) **Uncertainty calibration**——报出的 uncertainty 大小是否与实际偏差成正比；(3) **Source attribution**——事后能不能定位到是哪一路出问题；(4) **Recovery**——处理完之后成功率退化多少。

### 8.7 Consistency graph → 可量化的 fault isolation benchmark

$\mathcal{L}_{consistency}$ 不是 scalar、是图：

```text
           Vision
          /      \
      Tactile --- F/T
          \      /
         Proprio
```

每条 $(i, j)$ edge 是一个 consistency constraint $\mathcal{C}_{ij} = D_{ij}(z_{ij})$（$D$ 用 §8.6 的归一化度量）、$\mathcal{C}_{VT}$ 在 contact position、$\mathcal{C}_{TF}$ 在 normal force、$\mathcal{C}_{FP}$ 在 $\tau_{\mathrm{res}} - J^T F$ 残差、$\mathcal{C}_{VP}$ 在 EE pose。总损失：

$$
\mathcal{L}_{\text{consistency}} \;=\; \sum_{(i,j)} w_{ij}\, \mathcal{C}_{ij}
$$

再进一步、把每条 edge 的 residual 记成 $r_{ij} = \mathcal{C}_{ij}$、按边排 rank：**edge residual ranking → candidate faulty modality**、或者用更原则化的 $P(\text{fault} = i \mid \{r_{ij}\}_j)$。到这里可以把它从"检测矛盾"升级成一个**真正可量化的 fault isolation benchmark**、给 provenance 第一次一个 quantitative metric：

```text
Fault isolation metrics
├── fault detection AUROC
├── source attribution accuracy / F1_fault
├── top-1 faulty modality / top-k coverage
├── recovery success（隔离 + 重估后性能回升）
└── time-to-diagnosis（从注入到报警的延迟）
```

这条升级让 §6.4 那条 7-tuple 公式里的 **Provenance 与 Validity 两项变得可 benchmark**、也把 §6.2.1 的 correlated-evidence 问题（比如 proprio / F/T 共享 wrench latent）变成可检验的对象。

### 8.8 Representation bottleneck ablation：接口到底靠"结构化"还是"多一个 bottleneck"

这是 v5 新增、也是回答 reviewer "接口价值会不会只是一个额外 bottleneck"的关键一组。比较四种从 raw 到 policy 的通路：

```text
A  raw → fusion → policy                         （无结构、无接口）
B  raw → learned latent → policy                 （隐式表示、无显式契约）
C  raw → structured slots → learned embedding → policy
                                                 （有接口、接口后再 embedding 给 policy）
D  raw → structured slots → controller           （有接口、直接给非学习消费者）
```

在 IID task success、**unseen sensor、unseen mounting、missing modality、contradiction、consumer swap** 六个条件下分别对比。若 C / D 在 IID 上不输 A / B、却在 unseen / missing / contradiction / consumer swap 上明显更强、那就说明价值确实来自"结构化契约"而非单纯 bottleneck；反之若 IID 明显更差而 robustness 无提升、本文的推荐就该降级为"仅在需要多消费者复用时才值得"。

### 8.9 Registration perturbation（相对化、四类分开、parameter vs convention 分别报）

时间扰动相对化：$\delta t \in \{0.25\,\Delta t,\; 0.5\,\Delta t,\; \Delta t,\; 2\,\Delta t\}$、$\Delta t$ 是**该任务的 time-relevant bandwidth**；不写具体毫秒数字、避免 benchmark 不能迁移。坐标系扰动分成四类：

```text
Registration robustness
├── calibration noise        ── T̂ = T · ΔT、ΔT 是小扰动（高斯 / 均匀）
├── calibration drift        ── 长时程慢漂、模拟温度 / 机械蠕变
├── sensor remounting        ── 换工具 / 换相机 / 换 sensor mounting、几何重定义
└── frame convention mismatch── base ↔ world、sensor ↔ tool、extrinsic sign flip
```

**前三类是 parameter perturbation（$T \rightarrow T\,\Delta T$ 的连续扰动）、第四类 convention violation 不是连续扰动、而是 semantic / convention failure**、下游消费者的失败模式完全不同。两类应**分别报告**、不要混在一个 domain-randomization 指标里、否则会把"参数错"和"约定错"的平均效果混同。

### 8.10 现有 benchmark 的适配度

RoboCasa / LIBERO / ManiSkill3 / BEHAVIOR 这一批主流 manipulation benchmark **大多以视觉为主、触觉缺席**。这一节不下结论、但**建议**：把上面九类指标做成一个可选插件、给现有 benchmark 加一层 "feedback-value + interface quality" 视角；《触觉·力控》 §5.2.1 的 A/B/C/D 四臂消融可以直接借用。

## 9. 与 VLA、世界模型的关系

### 9.1 VLA：目前尚缺的不是"concat 通道"、而是跨 sensor family 的稳定 state contract

§3.1 已经修正过——把 RT-2 / OpenVLA / π0 都归为 "concat" 不准确、也要区分"支持输入"与"建立契约"。**以 RT-2、OpenVLA、π0 等代表性公开系统为例可以观察到**：现有通用 VLA 的规模化预训练生态主要围绕 vision-language observation 与 robot state / action 展开。关于 tactile / F/T 缺席、本文把它当作一个 **ecosystem observation（文献定位式判断）** 来陈述、而不是一个可被几篇 paper 证实 / 证伪的事实定律、并尽量用可反驳的措辞：

> **To our knowledge, no publicly established ecosystem currently matches vision-language-action pretraining in scale, sensor diversity, task coverage, and embodiment diversity for tactile / force observations.**

需要限定的是——这个缺席**不等于"没人做过"**：tactile foundation model 正在出现（TVL、AnyTouch）、visuo-tactile representation learning 已经很多（Lee、Calandra、3D-ViTac）、部分 robot foundation model 工作确实把 proprio 加入输入、hybrid state representation 也不是空白。真正尚未成形的是——**跨 sensor family、跨 embodiment、跨 downstream consumer 的一份稳定 structured state contract**；而"某个 VLA 能吃 proprio 输入"这件事、离"它定义了跨消费者可复用的 state contract"还差着 §6.4 说的那一整层。

真正的开放问题是：**VLA 的下一波扩展、是加更多 vision+language、还是补上 tactile / F/T 的 state contract？** 这一篇押后者——但不是把 tactile 当作"再一个 channel"、而是把 §6 的 structured state contract 塞进 VLA 的输入层。Qi 等 T-Dex [arXiv:2309.09979](https://arxiv.org/abs/2309.09979)（CoRL 2023）与 Lee 等 [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) 可以视为这条路线的早期形态。

### 9.2 世界模型：不批评 latent world model、只建议 contact-rich 动力学保留一个可辨识的 event / mode channel

RSSM / DreamerV3 [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) 一脉的 latent dynamics 通常假设 state transition 相对平滑、梯度可导。《触觉·力控》 §5.1 讨论过、接触事件本质上是 hybrid dynamics 的 mode switch。

**本文不是批评 latent world model、也不是说 "continuous latent 会把 contact 平滑掉"**——现代 latent dynamics 完全可以用 discrete latent / stochastic latent / event latent / hierarchical latent / latent mode switching / multiple heads 来表示、这些都能承载事件与 mode。本文的表述收敛成一条**很窄、因此也很难被反驳**的建议：

> **对 contact-rich dynamics、我们建议 world model 至少保留一个可辨识（identifiable）的 event / mode channel、而不是强迫它完全隐式地存在于一个不可区分的 continuous latent 中。**

世界模型可以分别预测 $z_{t+1}$、$p(m_{t+1}\mid z_t, a_t)$、$p(C_{t+1}\mid z_t, a_t)$、或用 hybrid latent（`z_continuous + z_discrete + contact state`）。这是一个 architectural recommendation、不是事实性结论。

这条分工与 §6 的 state contract 自然对接：**world model 不必消费 slot 全部字段、但通常值得让某个 head 显式预测 slot 里 mode / event 的部分**；policy 层可以用 learned representation（从 slot 学 embedding 给 policy 消费是合理的、但这个 embedding 是"下游消费者"、不是"上游数据格式"、见 §6.4(b)）；两者之间用 §6 的接口连接、world model 预测下一时刻的 slot、policy 消费当前 slot。

### 9.3 一句话总结

**VLA 缺的不是 tactile channel 本身、而是跨 sensor family / 跨 embodiment / 跨 consumer 的稳定 state contract；world model 缺的不是 continuous latent、而是在 contact-rich 任务上至少保留一个可辨识的 event / mode channel。这两件事的共同根因、是 heterogeneous observation 与 downstream models 之间没有一层可以承载 value / semantics / frame / time / uncertainty / provenance / validity 的 structured state contract。**

## 10. 结论

这一篇从"多模态融合常被讲成模型结构问题"这个误解出发、把讨论拉回到底层。本文真正的论点、比"融合要加 uncertainty"更进一步：**这不是在提议一个新的 fusion operator、也不是在提议一个新的 state estimator、而是在提议一份关于 estimator 输出的契约（a contract over estimator outputs）**——一份横跨 heterogeneous observation 与多个 downstream consumer、能显式表达 value / semantics / frame / time / uncertainty / provenance / validity 的 **structured state contract**。它刻意只规定"暴露什么"、而把"用什么算法推断"（Layer 3、Bayesian / EKF / factor graph / learned filter 皆可）与"下游用什么表示消费"（可以是 symbol、也可以是 neural embedding）完全留白。

$$
\boxed{\;\text{Interface} \;\neq\; \text{Inference algorithm} \;\neq\; \text{Latent representation}\;}
$$

视觉之所以撑起一整个共同数据接口、不是因为它的 encoder 更聪明、也不是因为它"解决了 registration"（并没有）、而是因为它积累的 **任务级 representation conventions** 足够多、把那些仍未解决的 registration 问题压成了局部工程麻烦、再加上相机模型 / 标定规范 / 文件格式 / 时间轴 / GPU / 廉价传感器 / 互联网数据 / 标注生态等一整套生态条件。触觉 / 力觉 / 本体感觉各自的 effective observation rate、坐标系、raw 表示、语义 convention 都还没收敛、于是 cross-attention / shared latent 的努力很容易在 §7 那五类失败模式上翻车。

本文不反对 cross-attention、不反对 shared latent、**也不反对 end-to-end learning**；本文反对的是——**让 state estimation、cross-modal composition 与 control 三件事全部隐式发生在一个不可诊断、不可复用的 latent 里**。给出的最小可用路线是：以 **structured state contract** 为 Layer 4、raw→slot 映射放在 sensor-specific perception（Layer 1–2）、estimator 可任选（Layer 3）；contact set 是这份契约在 manipulation 场景下的一个实例（$\text{ContactSet}\subset\text{StructuredState}$）、slot 带 **point / patch / region 三类几何**、**连续 covariance 与类别 probability 分开、且 uncertainty 表征匹配状态拓扑（count 是 combinatorial、multimodal 假设不能塞进单个 Gaussian）**、**保留 aleatoric / epistemic / calibration 拆分接口**、**track_id 是带 confidence 的 hypothesis、不是内禀 identity**、**unit / frame / reference point 通过 `Quantity` 进入 schema、可被 runtime validator 校验**、**availability（能不能拿到）与 validity（能不能信）拆成两条轴、staleness 是"相对过去时刻的 validity"、不等于 invalidity**、**显式记录 observability / identifiability 与 correlated evidence（防 double counting）**；`wrench` 作为独立 aggregate measurement 与 `contacts[]` 假设之间产生 consistency residual、把 Design for disagreement 变成数学对象；接口本身有 versioning 与 graceful degradation；**本文并不规定所有 fusion 必须发生在接口、只要求需要被多消费者共享 / 诊断 / 复用的状态在接口暴露**；并把 Interface Property Benchmark（schema / encoder / consumer swap、oracle / estimated / end-to-end 三组受控 baseline、uncertainty-aware disagreement、consistency graph 与 fault isolation、representation bottleneck ablation、registration perturbation）作为一等公民写进评测。

回到本文的三条设计原则（它们已覆盖 measurement → representation → robustness、不再加第四条）：**Register before compose · Expose belief at the interface · Design for disagreement**。

最后留三条可以贴到墙上的公式作为整篇文章的锚点：

$$
\boxed{\;\text{Sensor-specific observations} \;\rightarrow\; \underbrace{\text{Structured State Contract}}_{\text{stable interface}} \;\rightarrow\; \{\text{Policy, World Model, Controller, Diagnostics}\}\;}
$$

$$
\boxed{\;\text{Good fusion} \;\neq\; \text{agreement}\;}
$$

$$
\boxed{\;\text{A good multimodal system must represent disagreement, not merely resolve it.}\;}
$$

下一篇讲灵巧手与 in-hand manipulation打算从"接口"往下游走一步：**灵巧手与 in-hand manipulation**——把这一篇的 state contract 放到 T-Dex / DextrAH / LEAP 这一批近期工作的具体场景里、看它能不能撑起这些系统的架构、以及为什么"能装手的机器人很多、真正在做 dexterous 的少"这个反差背后的成本结构。

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
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)（开源 VLA 基线 · 公开配置含多相机 / depth / proprioceptive state encoding；但"支持输入" ≠ "建立跨消费者 state contract"）
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)（把 action 表达成 text token 与 VLM 联合 fine-tune · §3 修正描述）
- Hafner et al., *Mastering Diverse Control Tasks through World Models*（DreamerV3）, Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)（latent dynamics 的世界模型代表 · §9.2 "可保留 event / mode channel、但不批评 latent world model 本身" 的参照）

### F · Sim-to-Real / Domain Randomization 背景（支撑 §7.3 / §8.9）

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)（把坐标系扰动当 domain randomization 一部分的经典做法）

### G · 承接 《触觉·力控》 · Contact state 与 impedance（背景）

- 《触觉·力控》 那篇里已经引用过、这一篇继续沿用的：Hogan 阻抗控制三部曲、Posa-Cantu-Tedrake IJRR 2014（hybrid contact-mode trajectory optimization）、Lee 1810.10191、Qi 2309.09979、Huang 2410.24091、Zhao 2402.13232、Feng 2502.12191。这一篇不重复贴链接、需要精确出处请直接看 《触觉·力控》 的 Sources 部分。

---

> **相关阅读**
>
> - [只会看、不会摸：机器人为什么缺一双"手感"的手](/zh/articles/2026-09-13-tactile-force-sensing/)——这一篇的前作、把触觉与力控单独拆开讲
> - [Sim-to-Real 方法论](/zh/articles/2026-09-10-sim-to-real-methodology/)——§7.3 坐标系扰动、§8.9 registration perturbation 都可以借用它的 domain randomization 视角
> - [机器人数据为什么比大模型数据更难](/zh/articles/2026-09-09-robot-data-scaling/)——§3 shared latent 路线的"监督信号哪里来"问题、其实是数据 scaling 问题
> - [VLA 与世界模型](/zh/articles/2026-09-07-vla-world-models/)——§9 那一节是它的一个具体侧面：VLA 尚缺跨 sensor family / embodiment / consumer 的稳定 state contract；世界模型在 contact-rich 任务上建议保留一个可辨识的 event / mode channel
