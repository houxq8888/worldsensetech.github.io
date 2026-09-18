---
title: "具身智能的工程架构：一个能跑、能测、能换组件的 Agent 骨架"
slug: "2026-09-19-embodied-agent-architecture"
date: 2026-09-19
draft: false
categories: ["具身智能", "教程"]
tags: ["具身智能", "软件架构", "机器人", "VLA", "世界模型", "Python", "系统设计", "工程架构", "Sim-to-Real"]
description: "前面几篇把'具身智能缺的是接口不是模型'讲成了契约和评估协议，这篇落到工程：一个具身 Agent 的代码到底怎么组织。给出运行时栈与训练时栈的分层、StateContract / Action / Env 三条核心契约、时间化动作与 active/scheduled 双槽、epoch 屏障的 ActionBuffer 调度协议、三条线程双频率解耦的 runtime loop、Temporal & Failure Contract（安全门与 command authority、故障状态机与 FAULT_LATCHED、延迟预算与动作覆盖率、Supervisor），以及文末一个纯 stdlib、可跑可测的最小闭环。"
toc: true
related_articles:
  - 2026-09-22-agent-deployment-rollback
  - 2026-09-16-policy-side-evaluation
  - 2026-09-15-policy-side-interface
  - 2026-09-14-multimodal-fusion-interface
  - 2026-09-12-sim-to-real-evaluation-protocol
  - 2026-09-06-embodied-ai-landscape
  - world-model-lab-setup
---

前面几篇把"具身智能缺的是接口、不是模型"这个判断一路推到了契约对象和评估协议：多模态融合要交付 [Structured State Contract](/zh/articles/2026-09-14-multimodal-fusion-interface/)，policy 侧要按 [Consumer Contract](/zh/articles/2026-09-15-policy-side-interface/) 消费它，落地要靠 [四类 compliance evidence](/zh/articles/2026-09-16-policy-side-evaluation/)。理论到这里已经闭环了，但留言里问得最多的问题始终是同一个：

**道理懂了，代码长什么样？**

这篇回答这个问题。不讲框架选择（那是另一篇的事），讲骨架：一个具身 Agent 的代码怎么分层、模块之间靠什么接缝、训练代码和部署代码怎么共用同一份定义。

先交代定位：**我给出的不是某个模型的 demo，而是一套面向工程化的架构骨架。** 示例代码刻意压缩了实现细节（schematic：省略导入、类型定义和错误处理），重点展示的是组件之间的接缝、时间语义和失败边界；文末再用一个纯 stdlib 的最小闭环把其中最小的一部分真正跑起来。跑通它离生产可用还差着标定、驱动和硬件那一整层，但接缝的形状是认真的。也要把边界说在前面：本文的 Protocol 定义的是软件组件之间的语义边界，不等价于实时性、安全认证或驱动器级保证——hard real-time 与功能安全活在控制器和硬件那一层，软件骨架只负责不挡它们的路。

## 先立一个总原则：架构是"接缝"的设计

写具身 Agent 最容易犯的错，是从模型出发往下堆代码：先有个大的 policy 网络，然后把相机驱动、动作后处理、安全裁剪一股脑塞进训练脚本。三个月后代码变成一坨，换相机要动 policy，换机器人要动数据管线，谁也不敢重构。

正确的出发点反过来：**先设计接缝，再填实现**。具身系统天然分成两个栈：

| 栈 | 职责 | 运行时机 | 关键约束 |
|---|---|---|---|
| 运行时栈（runtime） | 感知 → 状态估计 → 决策 → 动作 → 控制 → 反馈 | 机器人上，实时 | 延迟确定性、安全、可降级 |
| 训练时栈（training） | 数据采集 → 学习（策略/世界模型）→ 评估 → 部署 | 离线 / 仿真集群 | 吞吐、可复现、数据质量 |

两个栈共享同一套**核心数据定义**（状态、动作、观测的 schema），但执行节奏、失败模式、资源画像完全不同。工程架构的第一课就是：**不要试图用一个进程、一份代码同时伺候两个栈**。这篇的骨架也因此拆成两半讲，但它们的接缝——`StateContract`、`Action`、`Env`——是同一份契约定义。整幅地图长这样：

```text
┌───────────── 训练时栈（离线） ─────────────┐
│ Dataset → Preprocess → Policy/WM → Eval → Artifact │
└──────────────────────┬───────────────────────┘
      共享契约：State / Action Schema · ObsTransform · Artifact Manifest
┌──────────────────────┴───────────────────────┐
│                运行时栈（实时）                │
│ Sensor → Estimator → StateContract → Policy → Action │
│                       ↓ 缓冲 / 插值 / 保持             │
│               Safety Gate → Controller → Robot       │
└──────────────────────────────────────────────┘
```

## 运行时栈：六层，每层只做一件事

把"感知到行动"这条链切开，每一层的职责可以用一句话说清，说不清楚的那一层就是你代码里的泥球。

| 层 | 职责（一句话） | 输入 → 输出 | 典型实现 |
|---|---|---|---|
| 1. Sensing | 把物理世界读成原始观测 | 硬件时间戳 → `RawObservation` | 相机驱动、力/触觉驱动、本体感受 |
| 2. State Estimation | 把多路原始观测融合成结构化状态 | `RawObservation` → `StateContract` | 多模态 estimator（呼应 9/14） |
| 3. Policy | 根据状态决定"做什么" | `StateContract` → `Action` | VLA / Diffusion / MPC |
| 4. Action Interface | 缓冲策略输出（chunk）、插值保持、翻译成控制器指令 | `Action` → `ControlCommand` | action buffer、chunk 调度、语义转换 |
| 5. Control | 把指令落到电机/关节 | `ControlCommand` → 电机力矩 | 厂商 SDK、ROS2 controller、力控环 |
| 6. Feedback | 读回执行结果，闭环 | proprioception → 状态更新 | 与第 1 层共享驱动层 |

两个容易被忽视的工程事实，先钉在这里：

**第一，Policy 频率 ≠ 控制频率。** VLA 以 5~10 Hz 出决策很常见，而关节力控环要跑在 200 Hz~1 kHz。第 3 层和第 5 层之间必须有一层（第 4 层）做"决策保持 + 插值 + 安全接管"，否则 policy 一帧卡顿，机器人就僵在半空。VLA 深度解读里强调过"不同频率口径不可直接对比"，工程上对应的就是这一层。由此得出一个结构性结论：运行时 loop 必须是双频率的——policy 线程按自己的节奏产 chunk，控制线程按固定 dt 消费，后者永远不阻塞在前者上。下文的最小 loop 就是这个形状。

**第二，安全不是 policy 的一个 if 分支，是独立的一层。** 第 4 层做硬约束（速度限幅、工作空间边界、力上限），第 5 层之上还要有 watchdog：心跳丢失、状态过期（staleness）、质量低于阈值时，robot 进入预定义的安全状态——注意是"安全状态"而不是特指"保持位置"：机械臂可以 hold，无人机该 land，车该 brake，具体动作由机器人的 SafetyPolicy 定义（后面"Temporal & Failure Contract"一节展开）。把安全埋进 policy 内部，等于让"会犯错的学习系统"兼任安全仲裁者。边界也要划清：这层是软件控制链上一道独立的防线，硬件限位、驱动器保护和独立急停链不在它的射程内——Python 进程挂了，硬件仍要兜住机器人。

## 训练时栈：数据进，模型出，和运行时共用 schema

训练栈的形状比运行时简单，坑都在细节上：

| 阶段 | 职责 | 关键工程问题 |
|---|---|---|
| 数据采集 | 从真机/仿真/遥操作攒 episode | episode 格式统一、时间戳对齐、失败案例不丢 |
| 数据管理 | 存储、版本、去重、回放 | 数据即资产，需要 lineage 和 split 协议 |
| 学习 | 训 policy / world model | 与部署共用 preprocessing（下面详谈） |
| 评估 | 离线指标 + 仿真 rollout + HIL | 评估协议见 9/12 那篇 |
| 部署 | 导出 checkpoint + 配置 | artifact 版本化，配置与代码同仓 |

这里埋着具身工程最贵的 bug 之一：**train/serve skew**——训练时用一套 preprocessing（resize、normalize、坐标变换），部署时用另一套，数值对不上，模型性能凭空打折。解法是强迫两个栈 import 同一份代码：

```python
# agent/state/obs_transform.py
# 训练时和部署时都从这一个模块 import，禁止各自实现
import numpy as np

class ObsTransform:
    """相机观测的唯一权威预处理，训练/部署共用。"""

    IMG_SIZE = (224, 224)

    def __call__(self, raw_bgr: np.ndarray) -> np.ndarray:
        img = raw_bgr[..., ::-1]              # BGR -> RGB
        img = resize(img, self.IMG_SIZE)      # 双线性，对齐训练配置
        return (img / 127.5 - 1.0).astype(np.float32)  # 与 checkpoint 一致
```

比 preprocessing 更容易漏的是 **artifact 清单**。`checkpoint_v17.pt` 这个文件名本身不足以复现一次部署——同一个 ckpt 配上不同的 state schema、归一化统计或机器人标定，行为完全不同，train/serve skew 也会从这条缝钻回来。把复现所需的一切绑成一个版本化 manifest，与 ckpt 同仓同版本：

```yaml
# deploy/artifact_manifest.yaml —— 复现一次部署所需的完整清单
artifact:
  checkpoint: ckpt/diffusion_v17.pt
  code_commit: 8f3a2c1           # 训练代码的 commit
  state_schema: v3              # StateContract 字段定义版本
  action_schema: v2             # ActionSpace / 坐标系 / dt 约定版本
  obs_transform: v5             # 训练/部署共用的预处理版本
  normalization: stats/norm_v17.npz   # 归一化统计量
  robot_model: calib/ur5e_2026-08.yaml  # 机器人标定
runtime:
  container_digest: sha256:71c0...  # 推理镜像 digest，不是 tag
  python: "3.11"
  pytorch: "2.4"
  cuda: "12.4"
compat:                        # 这份 ckpt 兼容的 schema 范围，启动时校验
  state_schema: [v3, v4]
  action_schema: [v2]
```

两个容易漏的扩展：`runtime` 把"哪份代码跑在哪个环境里"钉死——tag 可变、digest 不可变，跨机器复现先对 digest；`compat` 是 schema 演进的防火墙，policy ckpt 加载时先校验当前 schema 是否落在兼容区间内，**不兼容就 fail-before-motion**，而不是跑到一半才发现字段对不上。评估协议（9/12）里的 traceability 要求落到工程上，多半就是这样一张清单。

版本校验不能停在"字段还在不在"。schema 的兼容性判据是**语义兼容**：单位换了（rad→deg）、坐标系换了（base→tool）、关节顺序换了、归一化统计量换了、时间口径换了——字段层面一个不缺，行为层面全是 breaking。一个具体例子：v3 的 proprio 元组按 `[j1, j2, j3]` 排序，v4 按 `[j2, j1, j3]`，字段名和类型完全相同，逐元素语义却整个反了；这种变更任何"字段存在性"检查都会放行。所以 compat 区间写的是语义版本，加载校验要把 units / frame / 关节序 / normalization 指纹一起对——对不上就是 fail-before-motion，没有"先跑起来看看"这个选项。

指纹管的是"定义一致"，还缺一层"实现一致"：**golden vectors**。取一组固定输入（golden input）跑一遍共用的 `ObsTransform`，把输出连同环境信息（numpy 版本、resize 后端、dtype 转换路径）存成基准向量（golden output），部署启动与 CI 时重跑比对——任何一处数值漂移，比对失败即拒绝加载。同一手法适用于 tokenizer 往返、action denormalization、坐标变换：train 和 serve 即使 import 同一份代码，不同 numpy 版本、不同 resize 后端照样能造出 skew。到这一步，train/serve skew 就不只是"别复制两份代码"的组织纪律，而是一道可验证的 contract evidence。

## 目录结构：骨架先于血肉

按上面的分层，一个可维护的仓库长这样：

```text
embodied_agent/
├── agent/
│   ├── core/            # loop、时钟、生命周期管理
│   ├── perception/      # 第 1 层：Sensor 驱动（read → RawObservation）
│   ├── state/           # 第 2 层：StateContract + StateEstimator
│   ├── policy/          # 第 3 层：Protocol + 各路线实现
│   ├── action/          # 第 4 层：tokenizer / chunking / safety gate
│   ├── control/         # 第 5 层：机器人 SDK 适配
│   └── safety/          # watchdog、限幅、安全位姿
├── train/
│   ├── data/            # episode 格式、dataset、replay
│   ├── sim/             # 仿真后端（与真机同一 RobotInterface）
│   └── learn/           # 训练入口、loss、checkpoint 导出
├── configs/             # 每机器人/每任务一份 yaml
├── tests/               # 分层测试（见下文）
└── deploy/              # 导出 artifact、版本清单
```

注意三个刻意的设计：`agent/core` 不 import 任何具体实现；`configs/` 与代码同仓——凡是影响模型输入、动作语义、控制参数、安全边界和实验结果的配置，都视为版本化 artifact、走同一个 review，而不是运行时随手改的环境变量；`tests/` 按层建目录，与 `agent/` 的层级一一对应。

配置的权威链也要写死：仓库里的 yaml 是不可变源，进程启动时解析成一份**运行时配置**（含默认值展开、环境覆盖、路径解析），并打印它的 hash。运行时任何组件想改参数，只能改自己内存里的副本，源头不变——这样"这次实验到底用了什么参数"永远可以事后回答。

## 核心契约与接缝

整个骨架的接缝不止一处——Sensor、StateEstimator、Controller、Safety 各有各的 Protocol——但承重的核心契约是三条：状态的形状（StateContract）、动作的形状（Action）、环境的形状（Env）。Policy 是挂在状态与动作之间的策略接缝，让三条路线成为可替换实现。它们都是 Python Protocol（结构化类型 duck typing），不强制继承、不绑架实现，但规定了接缝的形状。数据契约之外，运行时还有一层**时间与故障契约**（调度协议、安全门、故障状态机）——前三条管"组件之间交换什么形状"，后一条管"组件不按节奏、不按预期出牌时系统怎么收场"，放在测试策略之后单独展开。

### 契约一：StateContract

这是 9/14 那篇"Structured State Contract"的直接代码化：状态不是一坨 tensor，而是携带假设、来源、时效和质量的结构化对象。反模式里我们批评"裸 dict 满天飞"，那核心契约自己就不能还是 string key + object——观测、本体感受、质量都要写成 schema：

```python
# agent/state/contract.py
from dataclasses import dataclass
from enum import Enum

class Hypothesis(Enum):
    """多模态 estimator 可能给出多个假设，禁止无声折叠（呼应 9/15 Failure 1）。"""
    SINGLE = "single"
    MULTI = "multi"          # 需要 policy 侧做 mode_select

@dataclass(frozen=True)
class Provenance:
    source: str              # 哪个传感器/estimator 产出
    stamp: float             # 物理世界采样时刻（与 monotonic() 同时钟域）
    frame_id: str            # 坐标系

@dataclass(frozen=True)
class ObservationField:
    """一路观测的 schema 化封装：值之外，语义元数据齐全。"""
    name: str                # "image" / "depth" / "tactile" / ...
    value: object
    dtype: str
    shape: tuple
    unit: str | None
    frame_id: str | None
    stamp: float

@dataclass(frozen=True)
class Observation:
    """开放式多模态容器：加传感器 = 加 field，不动 contract。"""
    fields: tuple            # tuple[ObservationField, ...]

    def get(self, name: str) -> ObservationField | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None          # 传感器缺席是正常情况，policy 必须能处理

@dataclass(frozen=True)
class ProprioState:
    """本体感受：字段名是 contract，顺序不是；单位也是 contract。
    约定 position=rad、velocity=rad/s、effort=Nm，换单位走 schema 版本，不走口头通知。"""
    joint_names: tuple       # ("shoulder_pan", "shoulder_lift", ...)
    position: tuple
    velocity: tuple
    effort: tuple | None
    stamp: float

@dataclass(frozen=True)
class StateQuality:
    """状态可用性，不是概率。分来源给，Safety 按任务定义 gate。"""
    proprio: float
    vision: float
    localization: float
    temporal: float          # 多传感器对齐质量（见下文）

    def min(self) -> float:
        return min(self.proprio, self.vision, self.localization, self.temporal)

@dataclass(frozen=True)
class StateContract:
    """policy 唯一允许消费的状态对象（呼应 9/15：policy 是 contract consumer）。"""
    observation: Observation
    proprio: ProprioState
    task: object                       # 指令/目标表征（MPC 等无语言输入的可为 None）
    hypothesis: Hypothesis
    provenance: Provenance
    validity_sec: float                # 数据有效期（过期必须 gate 掉）
    quality: StateQuality
    schema_version: str                # 字段定义版本，启动时与 artifact manifest 的
                                       # state_schema 校验：对象要知道自己是哪一版

    def age_sec(self, now: float) -> float:
        """数据年龄 = now - 物理采样时刻。estimator 算了 100ms，年龄就长 100ms。
        now 由调用方显式传入：contract 内部不读 wall clock（见下文的 Clock Protocol）。"""
        return now - self.provenance.stamp

    def is_fresh(self, now: float) -> bool:
        return self.age_sec(now) < self.validity_sec
```

`frozen=True` 不是洁癖：状态对象一旦构造就不该被下游改写，否则"谁在哪个环节改了状态"会成为无法定位的 bug 源。但要钉准 `frozen` 的语义边界：它保证的是**字段引用不可重新绑定**，不是 deep immutability——`ObservationField.value` 里若装着一个 `ndarray`，`field.value[0] = 123` 照样生效。真正的下游不可变还要靠只读视图、ownership 规则或 copy-on-write 补上，这一条漏了，`frozen=True` 只是给了人一种已经安全的错觉。

同属最小骨架的还有 `validity_sec` 这个单一数字：它对闭环够用，但 multi-modal runtime 里 vision 30 Hz、IMU 500 Hz、joint 1 kHz 的合法年龄天然不同（几十毫秒对几毫秒），生产系统通常需要 per-field freshness 或 modality-specific validity policy——否则一个整体的 `is_fresh(now)` 会被误读成"整个状态整体新鲜"，而其实只有 proprio 那一路还新鲜。

**时间语义是这里最容易写错的地方。** freshness 必须按**物理采样时刻**算，而不是按对象创建时间算——否则 estimator 跑了 100ms 计算，`is_fresh` 反而认为状态"刚出炉很新鲜"，实际上它已经是 100ms 之前的世界了。对机器人要紧的是数据年龄（data age），不是软件对象的年龄。前提是所有时间戳与 `monotonic()` 在同一时钟域；如果传感器时间戳来自 ROS / system wall clock，先做时钟域对齐再比较。

由这条推出一条纪律：**clock 是依赖，不是全局变量**。contract、buffer、调度层都不许自己读 `time.monotonic()`——所有 `now` 都从唯一一个注入的 Clock 流进来。生产注入 `SystemClock`，测试注入 `FakeClock`；不注入时钟，确定性回放就无从谈起：

```python
# agent/core/clock.py
from typing import Protocol
import time

class Clock(Protocol):
    """全系统唯一的时间来源。业务代码里散读 time.monotonic() 的位置，
    每一个都是测试里无法冻结的时间裂缝。"""
    def monotonic(self) -> float: ...

class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()
```

**多传感器同步是这里最难的另一半。** camera 30 Hz、IMU 500 Hz、joint 1 kHz，各自的时间戳天然不同——真正干活的 StateEstimator 要写明：对齐窗口、插值/外推策略、超时丢弃规则。所以 `RawObservation` 把时间戳做成一等公民，estimator 的产出质量用 `quality.temporal` 回报：

```python
@dataclass(frozen=True)
class RawObservation:
    """单路传感器的原始读数。"""
    sensor_id: str
    sensor_timestamp: float    # 硬件采样时刻（注明属于哪个 clock domain）
    receive_timestamp: float   # 进入软件的时刻
    sequence_id: int
    clock_domain: str          # "monotonic" / "ros_wall" / ...
    calibration_version: str
    payload: object
```

多时钟域要落成一条显式管线，而不是埋在 estimator 里的隐式约定——不归一，两个域的时间戳根本没有可比性，`data age` 也就无从谈起：

```text
原始传感器时间戳（各自 clock domain）
    ↓ clock normalization：统一换算进 monotonic 域，记录换算版本
带统一时间戳的样本缓冲
    ↓ temporal alignment：按窗口对齐、插值/外推、超时丢弃
StateContract（provenance.stamp 已在 monotonic 域）
```

另一个容易糊在一起的接缝是"读"和"估"。第 1 层和第 2 层应该是两个 Protocol，而不是一个 perception 对象上的两个方法：

```python
# agent/perception/sensor_buffer.py、agent/state/estimator.py
@dataclass(frozen=True)
class SensorSnapshot:
    """各源在 now 时刻的最新样本集合：estimator 做时间对齐的输入，不是单路读数。"""
    samples: tuple          # tuple[RawObservation, ...]，每源至多一条
    window_sec: float       # 对齐窗口：早于 now - window 的样本不允许参与对齐

class SensorBuffer(Protocol):
    """第 1 层：驱动线程持续把带时间戳的样本写进环形缓冲，消费方非阻塞取最新。
    生产形态绝不是 blocking read()——runtime 线程没有资格阻塞在传感器上。"""
    def latest(self, now: float) -> SensorSnapshot: ...

class StateEstimator(Protocol):
    """第 2 层：系统里唯一允许产出 StateContract 的地方。
    职责：时钟域归一、时间对齐、插值/外推、drop policy、最大同步窗口。"""
    def estimate(self, snapshot: SensorSnapshot, now: float) -> StateContract: ...
```

### 契约二：Action——时间化的动作

VLA 吐 joint delta、MPC 吐目标位姿、Diffusion 吐动作 chunk——这些输出的**语义、单位、坐标系、时间间隔、执行时长**都不同。接缝要真能承载差异，Action 必须自带元数据，否则"三种路线吐同一种 Action"只是类型层面统一，还没达到真正的 contract。

比元数据更容易缺的是**时间口径**。假设 state 在 t=10.000 采样，VLA 推理 150ms 后吐出一个 dt=0.02、horizon=10 的 chunk——它究竟对应世界时刻的 `[10.150, 10.350]`，还是"从现在开始执行 200ms"，还是"对应 state 10.000 之后的未来 200ms"？三者不是一回事，policy 又是异步线程，含糊的口径会让下游缓冲无法严格定义。所以 Action 必须回答五个问题：**这是哪个世界时刻的决策、从什么时候开始有效、什么时候失效、属于哪一次 episode（epoch）、在这次 episode 内是第几号决策：**

```python
# agent/action/contract.py
class ActionSpace(Enum):
    JOINT_POS = "joint_pos"
    JOINT_DELTA = "joint_delta"
    EE_POSE_DELTA = "ee_pose_delta"   # 末端位姿增量
    # ...

@dataclass(frozen=True)
class ActionSchema:
    """空间语义与单位：space 定语义，representation 定编码，单位与坐标系是 contract。"""
    space: ActionSpace
    representation: str         # 编码约定：joint_rad / ee_se3_log / ee_6d_rot / ...
    translation_unit: str|None  # 平移单位（关节动作为 None）；EE 动作必须显式
    rotation_unit: str|None     # 旋转表示：se3_log / quat / euler_xyz / 6d ...
    frame_id: str               # 坐标系：joint / base / ee ...

@dataclass(frozen=True)
class Action:
    """策略输出不是一坨裸 tensor：语义、坐标系、时间口径都写在对象里。"""
    values: object        # 动作值：chunk (H, D) 或单步 (D,)
    schema: ActionSchema  # 空间语义、编码、单位、坐标系
    schema_version: str   # schema 指纹：consumer 侧据此 fail-before-motion
    dt: float             # 相邻动作点的时间间隔
    horizon: int          # chunk 长度，单步动作为 1

    generated_at: float   # 推理完成、提交给第 4 层的时刻
    valid_from: float     # 生效起点
    valid_until: float    # 失效时刻：超过即 stale，不允许再被消费
    state_stamp: float    # 决策所依据状态的物理采样时刻
    epoch: int            # 生命周期身份：属于哪一次 episode / runtime epoch
    sequence_id: int      # epoch 内决策序号：异步提交不许乱序（见调度协议）
```

同一个 `EE_POSE_DELTA`，旋转用 se(3) 对数、四元数还是 6D rotation matrix，值的形状和插值语义完全不同——这正是 `representation` 字段存在的理由。

有了这五个时间字段加 epoch 身份，"抢占时机、过期丢弃、决策溯源"才都有据可依：`state_stamp` 把它钉回它看见的世界，`valid_until` 给消费方一个硬截止，`sequence_id` 在 epoch 内给异步提交定序，`epoch` 则让 reset 之后晚到的旧决策永远失去覆盖权——episode 41 的最后一号决策再晚到，也进不了 episode 42 的缓冲。

五个字段还暗含一条值得写明的不变量：**`state_stamp <= generated_at <= valid_from < valid_until`**。它把两个常被混为一谈的延迟拆开了——决策延迟 `generated_at - state_stamp`（policy 看见世界到提交决策的耗时，VLA 那 150ms 花在这里），调度提前量 `valid_from - generated_at`（决策提交到允许生效的缓冲，给传输和排队留的余量）。不变量之外还有迟到决策的处置协议：新 chunk 到达时 `valid_from` 已过、但 `valid_until` 未到，是截断头部立即执行剩余部分、整段丢弃、还是 ASAP 从当前时刻重新铺开？三种选择对应三种动作连续性语义，必须在调度协议里显式声明，不能留给实现临场发挥。

另一条口径论文里常默认、实现里必须钉死：**chunk 是零阶保持，不是点采样**。`values[i]` 作用于世界时刻区间 `[valid_from + i*dt, valid_from + (i+1)*dt)`——左闭右开。所以 `dt=0.02、horizon=16` 的 chunk 实际覆盖 `[valid_from, valid_from + 0.32)`；控制环在区间尾部取到点 15 并执行到末尾，是这个语义的自然结果，不是 off-by-one。把 chunk 误读成"一串时刻上的目标值"，是这套协议最常见的实现 bug。取点的索引是 `int((now - valid_from) / dt)`——floor，不是 round：t=0.011、dt=0.02 时 round 会把 0.55 拍错取成第 1 点，而按左闭右开的定义它仍属于第 0 点。

时间语义之外，chunk 切换处还横着一条**轨迹连续性**契约：A(t_end⁻) ≈ B(t_start⁺)——新 chunk 的起点要和旧 chunk 的终点在 position 上衔接，最好 velocity、acceleration 也连续。这条契约不在 ActionBuffer 的射程内：**ActionBuffer 负责时间选择**（哪一拍的哪个点生效），**trajectory feasibility / continuity 属于 Safety 与 Controller 的 contract**。同样要钉准的是 rate-limit 的语义边界：它只保证每拍 `Δq ≤ vmax·dt`（位置增量上限），既不普遍保证 velocity 上限，更不管 acceleration 与 jerk——把"限速"说成"保证轨迹连续"，是这两层职责最常见的混淆。

### 接缝：Policy

```python
# agent/policy/base.py
from typing import Protocol

class Policy(Protocol):
    """第 3 层的接缝。VLA / Diffusion / MPC 都只是它的实现。"""

    def reset(self, epoch: int, alloc: "SequenceAllocator") -> None:
        """episode 边界：清空 KV cache / 隐藏状态 / 规划缓存，
        同时领取本 epoch 的身份与决策序号——(epoch, sequence_id) 才是决策的真实身份。
        全局 next_seq() 这类进程级可变状态会污染回放与 reset，禁止使用。"""
        ...

    def act(self, state: StateContract, now: float) -> Action:
        """给定状态与世界时刻，输出一个动作（通常是一个动作 chunk）。now 由 Clock 显式传入。"""
        ...
```

三个实现各自很短——短是因为骨架把"吃状态、吐动作"之外的脏活全推给了别的层。它们都是 schematic，省略了导入、错误处理和各自的推理细节，读接缝即可：

```python
# agent/policy/vla.py（示意，框架细节从略）
class VLAPolicy:
    def __init__(self, ckpt, action_tokenizer, device="cuda"):
        self.model = load_vla(ckpt).to(device).eval()
        self.tok = action_tokenizer
        self._past = None

    def reset(self, epoch=0, alloc=None):
        self._past = None     # KV cache 是 epoch 的从属物，随 reset 一并清零

    @torch.no_grad()
    def act(self, state: StateContract, now: float) -> Action:
        image = state.observation.get("image")   # -> ObservationField | None，缺席要会降级
        ids = self.model.generate(
            image=None if image is None else image.value,
            instruction=state.task,
            past=self._past,
        )
        # schematic：KV cache 能否跨帧复用取决于模型架构
        # （visual token 是否逐帧替换、位置编码、causal attention 结构），不能默认
        self._past = ids.kv_cache
        return self.tok.decode(ids.action_bins)  # -> Action(chunk)，第 4 层负责按时序放出
```

```python
# agent/policy/mpc.py（世界模型在环，呼应"混合架构"趋势）
class MPCPolicy:
    """不学显式策略，用世界模型 rollout 挑动作——TD-MPC 路线的运行时形态。"""

    def __init__(self, world_model, candidate_sampler, horizon=5, n_candidates=32):
        self.wm = world_model
        self.sampler = candidate_sampler
        self.horizon = horizon
        self.n = n_candidates

    def reset(self, epoch=0, alloc=None):
        pass  # 无跨帧状态：每帧从当前观测重新编码（stateless re-plan）

    def act(self, state: StateContract, now: float) -> Action:
        z = self.wm.encode(state)
        cands = self.sampler(z, n=self.n, horizon=self.horizon)
        # schematic：把 dynamics rollout / cost / constraint / uncertainty
        # 压缩进 estimate_return() 只为突出 Policy 接缝；真实 MPC 不该把这些语义
        # 藏成一个无约束的 scalar scorer
        scores = [self.wm.estimate_return(z, c) for c in cands]
        best = cands[int(np.argmax(scores))]
        u0 = best[0]  # 只执行第一步，下一帧重新规划（receding horizon）
        return Action(values=u0, horizon=1, ...)   # Policy.act 的返回类型是 Action，不是裸控制向量
```

```python
# agent/policy/diffusion.py（示意：省略了 timestep / noise schedule / CFG 等采样细节）
class DiffusionPolicy:
    """生成式策略：多步去噪出一个动作 chunk，天然输出多假设（呼应 9/15 Failure 1）。"""

    def __init__(self, denoiser, n_steps=10, clock=None):
        self.den = denoiser
        self.n_steps = n_steps
        self.clock = clock if clock is not None else SystemClock()   # Clock Protocol

    def reset(self):
        pass  # 无状态，或按实现清空条件队列

    def act(self, state: StateContract) -> Action:
        x = torch.randn(1, CHUNK_LEN, ACTION_DIM)   # 去噪起点是噪声，终点不是
        image = state.observation.get("image")
        for _ in range(self.n_steps):
            x = self.den(x, None if image is None else image.value, state.task)
        values = self.denormalize(x[0])   # 模型输出 -> 反归一化 -> 动作值
        now = self.clock.monotonic()      # 唯一时间来源：不散读 time.monotonic()
        return Action(values=values,
                      schema=ActionSchema(space=ActionSpace.JOINT_POS,
                                          representation="joint_rad",
                                          translation_unit=None,
                                          rotation_unit=None,   # 关节空间没有旋转单位：它属于 SE(3) 动作
                                          frame_id="joint"),
                      schema_version="joint_pos@v2",
                      dt=0.02, horizon=CHUNK_LEN,
                      generated_at=now, valid_from=now,
                      valid_until=now + CHUNK_LEN * 0.02,
                      state_stamp=state.provenance.stamp,
                      sequence_id=next_seq())  # chunk 交给第 4 层按时序放出
```

这条链的每一环都要是可审计的：**model output → denormalization → action schema 转换 → limits/projection → ActionBuffer → Safety**。反归一化用的统计量必须和训练时同一份——normalization 的 train/serve 一致性就卡在这条链上，任何一环换了实现，skew 就从这里回来。顺带一个 schema 设计上的教训：`ActionSchema` 把关节动作和 SE(3) 动作塞进同一组 translation/rotation 单位字段，joint 动作只能两个都填 `None` 兜底——继续演进时应当拆成 `JointActionSchema` / `CartesianActionSchema`（或 `units: tuple[str, ...]`），别让类型系统靠约定补洞。

三个实现摆在一起，9/15 那张"conditioning representation × action head"的 grid 就落了地：它们吃同一种状态、吐同一种 `Action`，区别只在"状态如何进网络"和"动作如何生成"。**路线之争在骨架层面被降级为可替换实现**——这正是接口设计的价值。

一句理论上的限定：在本文的软件抽象里，凡是接收当前状态并产生下一控制决策的模块，都统一视为 policy-like decision module，所以 MPC 在接口层与学习策略共享同一个插槽——但这不意味着它们在控制理论上是同一类对象。MPC 更准确的定位是 online receding-horizon 决策器，"挂进 Policy 插槽"描述的是工程接缝，不是理论等价。

### 第 4 层的调度协议（ActionBuffer）

Action 的时间字段把最难的问题交给了第 4 层：**新 chunk 到了以后，到底发生什么？** chunk A 覆盖 `[0.00, 0.20]`，chunk B（sequence_id 更大）在 `t=0.10` 到达、覆盖 `[0.10, 0.30]`——是 A 让位、A 执行完再轮到 B、立即抢占，还是两者混合？对机器人这是**动作连续性**问题，不是普通缓存问题。答案必须写成明确的调度协议：

```text
ActionBuffer semantics（active + scheduled 双槽，epoch 屏障）:
- 每个 epoch 一份序号空间：put() 只接受本 epoch 且 sequence_id > last_accepted 的 chunk，
  旧 epoch / 乱序晚到一律拒绝——reset 之后，迟到的旧决策永远失去覆盖权
- stale chunk（valid_until 已过）: 丢弃，绝不沿用旧决策
- future chunk（valid_from 在未来）: 进 scheduled 槽暂存，到 valid_from 在控制边界接管；
  期间只有 sequence_id 更大的 chunk 可以替换它
- 已生效 chunk: sequence_id 更大且 valid_from 已到的后来者立即接管 active 槽
- 接管只发生在控制边界（control boundary），不在控制拍中间切换；
  A 执行中 B 未到生效点，继续执行 A——不允许出现执行空洞
- missing chunk（无可用决策）: 进入 safe state，而不是静默沿用
- action discontinuity（A→B 跳变过大）: project / rate-limit 后再放行
```

还有一条是异步系统的纪律：**policy 输出不允许乱序提交。** 换成异步推理服务后，request #42 可能比 #41 先完成——如果 buffer 照单全收，旧决策就会晚到、覆盖新决策。所以 `put(action, now)` 先做 epoch 屏障判定（`action.epoch != current_epoch` 的直接拒绝），再只接受 `sequence_id > last_accepted_sequence` 的 chunk，其余直接丢弃。这也回答了"换组件"的一个重要边界：任何接进 Policy 插槽的实现，都必须遵守同一套提交纪律。

这层要薄，薄到不含任何隐藏策略：**默认不做跨 chunk 融合**，A→B 的跳变原样交出，连续性由 SafetyGate 的 project / rate-limit 兜底；重规划随时允许——B 在 `t=0.10` 到达、生效点是 `t=0.20`，就从 `t=0.20` 的控制边界接管，A 的剩余部分作废，不存在"先把 A 执行完"的队列语义；若 B 的生效点已过，则立即接管 active 槽。名字叫 buffer，行为上是 **active + scheduled 双槽寄存器**：FIFO 的直觉（排队、溢出、丢最旧）在这层全部不成立，`put()` 的屏障判定与覆写就是全部语义。

### 契约三：Env——拆成"机器人接口"和"任务环境"两个世界

先分清它能做什么、不能做什么：统一的接口消灭的是**人为制造的接口差异**——两套代码、两套字段名、两套坐标约定；它消灭不了 **sim-to-real gap 本身**——dynamics mismatch、执行器延迟、摩擦、传感器噪声、标定误差，这些要靠 domain randomization、延迟/噪声建模、system identification 和真机验证去解决。把接口统一当 gap 的解药，是这类架构最常见的高估。

但"统一接口"本身也藏着一个常见的设计错误：`reset() -> RawObservation`、`step() -> (RawObservation, reward, done, info)` 这个三件套对 RL simulator 很自然，却把一个接口塞进了**三个角色**——机器人接口、RL 环境、任务评估。真机器人根本不天然拥有 `reward` 和 `done`：那是任务定义，不是硬件属性。拆开之后 sim-to-real 的边界反而更干净：

```python
# agent/control/robot.py、train/sim/task_env.py
class RobotInterface(Protocol):
    """真机与仿真后端实现同一个接口：只会观察、发送、复位。"""
    def observe(self) -> RawObservation: ...
    def send(self, cmd: ControlCommand) -> None: ...
    def reset_robot(self) -> None: ...

class TaskEnv(Protocol):
    """任务定义：把 reward / done / 成功判据从硬件里剥出来。"""
    def reset(self) -> StateContract: ...
    def step(self, action: Action) -> Transition: ...   # Transition 里才有 reward/done
```

```python
class SimBackend:
    """MuJoCo / Isaac 适配 RobotInterface。proprio 字段名与真机驱动严格一致。"""
    ...

class RealRobot:
    """真机适配 RobotInterface。所有阻塞 IO 移到后台线程，observe() 只取最新。"""
    ...

class PickPlaceTask:
    """TaskEnv 的一种：包着一个 RobotInterface 或 SimBackend，负责评 reward。"""
    ...
```

训练时 `TaskEnv` 包着 `SimBackend`；部署时没有 `TaskEnv` 和 reward——policy 直接消费 `StateContract`。两个世界靠 `RobotInterface` 这一个接缝对齐，sim-to-real 要管的就只剩这一个实现差异。

另外 `send() -> None` 是示意性的最小同步接口：`send()` 被调用既不代表控制器接受了命令，更不代表执行器执行了它——生产 adapter 通常还需要 command acknowledgement、controller mode、health 与 fault status，把 submitted / accepted / executing / rejected 分开建模。这个接口的最小性是有意的：接缝先对齐，可靠性语义按机器人逐个补齐。

## 最小运行时 loop：三条线程、双频率解耦

把上面拼起来。运行时**不是单循环**：estimator 线程按自己的高频率持续发布状态，policy 线程按 5~20 Hz 产 chunk，控制线程按 200~1000 Hz 消费——中间靠两个寄存器解耦：状态进 StateBuffer，动作进 ActionBuffer。一次 150ms 的 VLA 推理只会拖慢 policy 线程，控制环无感：

```python
# agent/core/loop.py（schematic）
CONTROL_DT = 1.0 / 200     # 命令更新节拍 200 Hz；policy 线程按模型自身节奏跑
POLICY_DT = 1.0 / 20       # policy 产出的上限节拍：慢由模型决定，快不许越过它
MIN_QUALITY = 0.5          # 状态可用性下限：低于此值不产新决策

class AgentLoop:
    def __init__(self, sensor, estimator, policy, state_buffer,
                 action_buffer, safety, sink, clock):
        self.sensor = sensor                # Sensor Protocol（第 1 层）
        self.estimator = estimator          # StateEstimator Protocol（第 2 层）
        self.policy = policy                # Policy Protocol（第 3 层）
        self.state_buffer = state_buffer    # 第 2 层出口：状态寄存器
        self.action_buffer = action_buffer  # 第 4 层：chunk 寄存器 + 调度协议
        self.safety = safety                # 独立安全层（SafetyGate）
        self.sink = sink                    # 唯一出口：ApprovedCommandSink
        self.clock = clock                  # Clock Protocol：不散读 time.monotonic()

    def _estimator_thread(self):
        while self._running:
            now = self.clock.monotonic()
            snapshot = self.sensor.latest(now)   # 非阻塞：取各源最新样本集
            state = self.estimator.estimate(snapshot, now)
            self.state_buffer.publish(state)     # 唯一写者；消费方各自取最新且新鲜

    def _policy_thread(self):
        while self._running:
            now = self.clock.monotonic()
            state = self.state_buffer.latest_valid(now)  # 与数据生产解耦
            if state is None or state.quality.min() < MIN_QUALITY:
                self.safety.enter_safe_state("stale_state")   # 数据过期/质量不足
                self._sleep_until(now + POLICY_DT)   # 错误路径也守节拍，不许忙等
                continue
            action = self.policy.act(state, now)   # 允许 100ms+，不背控制的 deadline
            self.action_buffer.put(action, now)    # chunk 交给第 4 层按调度协议放出

    def _control_thread(self):
        next_tick = self.clock.monotonic()
        while self._running:
            now = self.clock.monotonic()
            state = self.state_buffer.latest_valid(now)   # 控制侧独立再判新鲜度
            if state is None:
                self.safety.enter_safe_state("stale_state")
            else:
                cmd = self.action_buffer.current(now)  # 无可用决策则返回 None（协议已定）
                if cmd is None:
                    self.safety.enter_safe_state("no_action")
                else:
                    decision = self.safety.evaluate(state=state, action=cmd,
                                                    context={"now": now})  # 生产形态，见下节
                    if decision.allow:
                        self.sink.submit(decision.command)  # 唯一出口：批准的命令才进 RobotInterface
                    else:
                        self.safety.enter_safe_state(decision.reason)
            next_tick += CONTROL_DT      # 节拍推进与单次耗时解耦，不漂移
            if now > next_tick:
                self.safety.record_deadline_miss(now - next_tick)  # 违约记账，按 miss policy 处置
                next_tick = now          # 重对齐：skip tick / degraded / safe stop
            self._sleep_until(next_tick)

    def _sleep_until(self, deadline):
        # monotonic 时钟对齐节拍；超时时记录 jitter 指标而非静默吞掉
        ...
```

几个刻意的设计：`safety.evaluate` 在动作缓冲之后、控制器之前，软件路径的任何分支都绕不过去——但它是**软件控制链上的一道独立防线**，不是系统的最后一道：硬件限位、驱动器保护和独立急停链必须独立存在。这里有两个升级相对初稿值得说明：状态门槛从单个 `confidence` 标量换成 `quality.min()`，因为"视觉 0.9、定位 0.2"折成一个标量本身就是语义偷换；`hold_position()` 换成 `enter_safe_state(reason)`，因为 hold 只是机械臂的安全态，无人机的安全态是 land、车是 brake——动作由机器人自己的 SafetyPolicy 定义。节拍对齐单独成函数，超时记指标——具身系统的延迟问题往往先表现为"jitter 增大"，把它变成可观测信号比事后抓栈有用得多。

还有两个接缝升级值得点名。状态不再经 `self._last_state` 这类跨线程私有缓存传递：estimator 线程是 StateBuffer 的唯一写者，policy 与控制两侧各自调用 `latest_valid(now)` 独立判断新鲜度——policy 侧的 stale gate 和控制侧的 stale gate 各自记账，谁也替谁做不了主。命令的 authority 也只有一条通路：policy 的 `Action` 只是提案，**只有 Safety 批准的 ControlCommand 能进入 RobotInterface**——正常 authority 走 `SafetyGate → CommandSink`（见安全门一节）；急停属于另一条独立的硬件 authority，不经过软件链的任何一层。

两处节拍纪律值得单独点名，因为它们都是 runtime correctness 的小坑：错误路径不许忙等——`continue` 之前同样要睡到下一个节拍点，否则传感器一挂，policy 线程就在 `latest_valid → safe → continue` 里把 CPU 打满；节拍推进用 `next_tick += CONTROL_DT` 而不是每轮重取 `t0 + CONTROL_DT`——后者一旦某拍超时，睡眠基准就被拖后，漂移会滚进之后的每一拍。超时不静默吞掉：记录 deadline miss 并按 miss policy（skip tick / degraded / safe stop）处置，这和"控制的 deadline 是硬契约"的叙述是同一件事。

最后把实时性边界说死：文中的 200–1000 Hz 指**控制接口的更新频率**，Python 这一层承担的是 200 Hz 级别的 command scheduling 与监督；真正的硬实时 servo loop（1 kHz 及以上）由 RT controller、RTOS、驱动器或固件承担，**Python runtime 不作为 hard-real-time safety loop**。

## 补到能跑：一个纯 stdlib 的最小闭环

上面都是 schematic。这一段把接缝补到能真跑真测：不用 torch、不用 GPU，一套假实现加一个 RuntimeCore、九个 pytest，`python -m pytest tests/test_loop.py -q` 直接通过（本文付印前实跑：9 passed）。生产和测试共用同一个循环体——`RuntimeCore` 只暴露 `prime(now)`、`policy_tick(now)`、`control_tick(now)` 三个入口：生产里由三条线程各自按节奏驱动，最小形态则把 estimator 折进控制拍、每拍先发布到 StateBuffer 再消费，接缝不变；测试里换成一个 FakeScheduler 逐拍驱动、注入时钟——不注入时钟，回放就无从确定性谈起：

```python
# tests/fakes.py —— 纯 stdlib 的假实现，pytest 直接跑
# 契约字段与 agent/state/contract.py、agent/action/contract.py 同形；
# 为保持最小，quality / SafeState 等字段从略，只保留本组测试要钉住的接缝。
from dataclasses import dataclass
from typing import Optional, Tuple
import math

CHUNK_LEN = 16
DT = 0.02


@dataclass(frozen=True)
class Provenance:
    source: str
    stamp: float                 # 物理采样时刻（与注入时钟同时钟域）
    frame_id: str


@dataclass(frozen=True)
class StateContract:
    proprio: Tuple[float, ...]
    provenance: Provenance
    validity_sec: float

    def is_fresh(self, now: float) -> bool:
        return (now - self.provenance.stamp) < self.validity_sec


class SequenceAllocator:
    """决策序号分配器：每个 runtime epoch 一份，epoch 内单调、reset 即清零。
    全局 next_seq() 会污染 replay 与 reset——(epoch, sequence_id) 才是决策的真实身份。"""

    def __init__(self):
        self._next = 0

    def next(self) -> int:
        self._next += 1
        return self._next


@dataclass(frozen=True)
class Action:
    values: Tuple[float, ...]    # 动作 chunk（CHUNK_LEN,）
    dt: float
    horizon: int
    generated_at: float          # 推理完成、提交给第 4 层的时刻
    valid_from: float
    valid_until: float           # 失效时刻：超过即 stale，不允许再被消费
    state_stamp: float           # 决策所依据状态的物理采样时刻
    epoch: int                   # 生命周期身份：属于哪一次 episode / runtime epoch
    sequence_id: int             # 决策序号：epoch 内单调，异步提交不许乱序


class FakeClock:
    """测试注入的时钟（生产对应 SystemClock；二者实现同一个 Clock Protocol）。"""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeSensor:
    """第 1 层 buffer 的最小形态：latest(now) 只取最新样本，绝不阻塞调用方。"""

    def __init__(self, clock):
        self.clock = clock

    def latest(self, now):
        t = now
        return {"angle": math.sin(t), "stamp": t, "frame_id": "joint"}


class DelayedSensor(FakeSensor):
    """模拟延迟 0.2s 的传感器：数据年龄超过 validity 时应触发 safe state。"""

    def latest(self, now):
        raw = super().latest(now)
        return {**raw, "stamp": raw["stamp"] - 0.2}


class FailingSensor(FakeSensor):
    """模拟 fail_at 时刻死掉的传感器：stamp 冻结在死前最后一帧，数据年龄持续增长。"""

    def __init__(self, clock, fail_at):
        super().__init__(clock)
        self.fail_at = fail_at

    def latest(self, now):
        t = min(now, self.fail_at)
        return {"angle": math.sin(t), "stamp": t, "frame_id": "joint"}


class NaiveEstimator:
    def estimate(self, snapshot, now):
        return StateContract(
            proprio=(snapshot["angle"],),
            provenance=Provenance("fake_sensor", snapshot["stamp"], snapshot["frame_id"]),
            validity_sec=0.05,
        )


class StateBuffer:
    """跨线程接缝的最小形态：estimator 发布最新状态，policy / 控制两侧各自取'最新且仍新鲜'的。

    两个消费侧各自做 freshness 判定：policy 侧的 gate 只保证'进入决策的状态是新鲜的'；
    控制侧必须重新判定——决策所依据的状态到执行时可能已经过期。
    """

    def __init__(self):
        self._state = None

    def publish(self, state):
        self._state = state

    def latest_valid(self, now):
        s = self._state
        if s is None or not s.is_fresh(now):
            return None        # 过期/缺失：交给 Safety 进入 safe state，绝不静默沿用
        return s


class SinePolicy:
    """deterministic policy：输出只由状态决定，episode 回放可逐位一致。"""

    def reset(self, epoch=1, allocator=None):
        self._epoch = epoch
        self._alloc = allocator if allocator is not None else SequenceAllocator()

    def act(self, state, now):
        return Action(
            values=tuple(state.proprio[0] for _ in range(CHUNK_LEN)),
            dt=DT, horizon=CHUNK_LEN,
            generated_at=now, valid_from=now,
            valid_until=now + CHUNK_LEN * DT,
            state_stamp=state.provenance.stamp,
            epoch=self._epoch,
            sequence_id=self._alloc.next(),
        )


class SlowPolicy(SinePolicy):
    """推理延迟注入：决策在发起时刻看见状态，latency 秒后才提交生效——
    生效前的空窗正是'动作覆盖率'缺口的来源。"""

    def __init__(self, latency=0.15):
        self.latency = latency

    def act(self, state, now):
        ready = now + self.latency
        return Action(
            values=tuple(state.proprio[0] for _ in range(CHUNK_LEN)),
            dt=DT, horizon=CHUNK_LEN,
            generated_at=ready, valid_from=ready,
            valid_until=ready + CHUNK_LEN * DT,
            state_stamp=state.provenance.stamp,
            epoch=self._epoch,
            sequence_id=self._alloc.next(),
        )


class ActionBuffer:
    """第 4 层最小形态：active + scheduled 两槽 + epoch 屏障。
    latest-wins 的准确语义：future chunk 暂存到点接管，绝不制造执行空洞。"""

    def __init__(self, epoch=1):
        self._epoch = epoch
        self._active: Optional[Action] = None
        self._scheduled: Optional[Action] = None
        self._last_accepted_sequence = 0
        self._active_sequence = 0
        self.events = []         # evidence 钩子：accepted/activated/preempted/expired/rejected_*

    def sync_epoch(self, epoch):
        """reset_episode 的屏障：旧 epoch 的决策立即作废，序号重新起算。"""
        self._epoch = epoch
        self._active = None
        self._scheduled = None
        self._last_accepted_sequence = 0
        self._active_sequence = 0
        self.events.clear()

    def put(self, action, now) -> bool:
        """只接受本 epoch 且更新的决策；旧 epoch / 乱序晚到直接拒。"""
        if action.epoch != self._epoch:
            self.events.append(("rejected_stale_epoch", action.sequence_id))
            return False
        if action.sequence_id <= self._last_accepted_sequence:
            self.events.append(("rejected_out_of_order", action.sequence_id))
            return False
        self.events.append(("accepted", action.sequence_id))
        self._last_accepted_sequence = action.sequence_id
        if action.valid_from <= now:
            if self._active is not None:
                self.events.append(("preempted", self._active.sequence_id))
            self._active = action            # 立即生效：latest-wins
            self._active_sequence = action.sequence_id
            self._scheduled = None
            self.events.append(("activated", action.sequence_id))
        else:
            self._scheduled = action         # future：暂存，到 valid_from 在控制边界接管
        return True

    def current(self, now: float) -> Optional[float]:
        """按时刻取 chunk 点；scheduled 到点提升，过期作废，绝不沿用。"""
        s = self._scheduled
        if s is not None and now >= s.valid_from:
            if self._active is not None:
                self.events.append(("preempted", self._active.sequence_id))
            self._active = s
            self._active_sequence = s.sequence_id
            self._scheduled = None
            self.events.append(("activated", s.sequence_id))
        a = self._active
        if a is None:
            return None
        if now >= a.valid_until:
            self.events.append(("expired", a.sequence_id))
            self._active = None              # 过期决策作废，不沿用
            return None
        idx = min(int((now - a.valid_from) / a.dt), a.horizon - 1)   # ZOH：floor，不是 round
        return a.values[idx]


class SafetyLimiter:
    """最小安全层：速度上限（物理量，rad/s）+ safe-state 入口记录。
    限幅基于实测位置，不是上一拍 command：发了 1.0 不代表机器人到了 1.0。"""

    def __init__(self, max_velocity=5.0, dt=DT):
        self.max_velocity = max_velocity     # 限的是速度，不是位置
        self._dt = dt
        self.safe_entries = []

    def check(self, state, cmd):
        pos, = cmd
        measured, = state.proprio            # 速度是否合法要对实测算，不是对 last command
        max_delta = self.max_velocity * self._dt   # 速度上限换算成本拍位置增量
        delta = max(-max_delta, min(max_delta, pos - measured))
        return (measured + delta,)           # 位置合法 ≠ 速度合法

    def enter_safe_state(self, reason):
        self.safe_entries.append(reason)

    def reset(self):
        self.safe_entries.clear()


class FakeController:
    def __init__(self):
        self.sent = []

    def send(self, cmd):
        self.sent.append(cmd)


class ControllerSink:
    """测试版 CommandSink：唯一允许调用 controller.send() 的对象。
    RuntimeCore 只拿得到 sink——绕过 SafetyLimiter 直写 controller 在结构上不可能，
    与生产同一条 authority 约束（见正文安全门一节）。"""

    def __init__(self, controller):
        self._controller = controller

    def submit(self, cmd):
        self._controller.send(cmd)


class RuntimeCore:
    """与生产 loop 同一个循环体：两个 tick 由调度器驱动，而不是由线程驱动。
    epoch 由 core 拥有：reset_episode 递增并同步给 policy 与 ActionBuffer。"""

    def __init__(self, sensor, estimator, policy, action_buffer, safety, sink,
                 state_buffer):
        self.sensor = sensor
        self.estimator = estimator
        self.policy = policy
        self.action_buffer = action_buffer
        self.safety = safety
        self.sink = sink   # 唯一出口：批准的命令经 CommandSink 进 controller
        self.state_buffer = state_buffer
        self.epoch = 0

    def reset_episode(self):
        self.epoch += 1
        self.policy.reset(self.epoch, SequenceAllocator())
        self.safety.reset()
        self.action_buffer.sync_epoch(self.epoch)
        # 生产还要重置：estimator 滤波器、StateBuffer、controller tracking、种子

    def prime(self, now):
        """episode 起点先把状态缓冲灌起来，policy 首拍才有状态可消费。"""
        state = self.estimator.estimate(self.sensor.latest(now), now)
        if state.is_fresh(now):
            self.state_buffer.publish(state)

    def policy_tick(self, now):
        state = self.state_buffer.latest_valid(now)
        if state is None:
            self.safety.enter_safe_state("stale_state")  # policy 侧 freshness gate
            return
        self.action_buffer.put(self.policy.act(state, now), now)

    def control_tick(self, now):
        # 生产里 estimator 跑在自己的高频 loop 里向 StateBuffer 发布；
        # 最小 loop 把它折进控制节拍，二者对 latest_valid 的判据完全一致。
        state = self.estimator.estimate(self.sensor.latest(now), now)
        if state.is_fresh(now):
            self.state_buffer.publish(state)
        state = self.state_buffer.latest_valid(now)
        if state is None:
            self.safety.enter_safe_state("stale_state")  # 控制侧 freshness gate，独立于 policy 侧
            return
        point = self.action_buffer.current(now)
        if point is None:
            self.safety.enter_safe_state("no_action")   # missing：不静默沿用
            return
        self.sink.submit(self.safety.check(state, (point,)))   # 唯一出口：approved command 经 CommandSink


def run_episode(clock, n_steps, hz=50, sensor=None, policy=None, policy_every=5):
    """FakeScheduler：policy 节奏用 policy_every 拍近似，时钟注入。"""
    controller = FakeController()
    core = RuntimeCore(sensor if sensor is not None else FakeSensor(clock),
                       NaiveEstimator(),
                       policy if policy is not None else SinePolicy(),
                       ActionBuffer(), SafetyLimiter(), ControllerSink(controller),
                       StateBuffer())
    core.reset_episode()
    core.prime(clock.monotonic())
    dt = 1.0 / hz
    for i in range(n_steps):
        now = clock.monotonic()
        if i % policy_every == 0:
            core.policy_tick(now)
        core.control_tick(now)
        clock.advance(dt)
    return controller, core.safety
```

九个测试分别钉住九件事：确定性回放、staleness 双门、限速、乱序拒绝、stale 状态绝不执行缓冲决策、真·延迟注入、epoch 屏障、scheduled 到点接管、floor 取点：

```python
# tests/test_loop.py
import pytest

from fakes import (CHUNK_LEN, DT, Action, ActionBuffer, FakeClock, FakeSensor,
                   DelayedSensor, FailingSensor, NaiveEstimator, Provenance,
                   SafetyLimiter, SinePolicy, SlowPolicy, StateContract, run_episode)


def _state(angle):
    """构造一个实测关节位置为 angle 的状态：限速测试的输入。"""
    return StateContract(proprio=(angle,),
                         provenance=Provenance("test", 0.0, "joint"),
                         validity_sec=0.05)


def test_episode_replay_bitwise_identical():
    a, _ = run_episode(FakeClock(), n_steps=100)
    b, _ = run_episode(FakeClock(), n_steps=100)
    assert a.sent == b.sent   # deterministic policy + 注入时钟 => 逐位一致


def test_stale_state_enters_safe_state():
    clock = FakeClock()
    _, safety = run_episode(clock, n_steps=10, sensor=DelayedSensor(clock),
                            policy_every=1)
    # 双 gate 各自记账：policy 侧 freshness gate 触发 10 次（从不发布），
    # 控制侧 latest_valid 触发 10 次（寄存器为空）；stale 优先于 missing 报告
    assert safety.safe_entries.count("stale_state") == 20
    assert safety.safe_entries.count("no_action") == 0


def test_safety_limits_against_measured_state():
    # 速度限幅必须对实测位置做，不是对上一拍 command：
    # 发了 1.0 不代表机器人到了 1.0——基于 command 算 Δq 会系统性低估真实速度
    safety = SafetyLimiter(max_velocity=5.0, dt=DT)   # 5 rad/s @ 50 Hz => 单拍上限 0.1 rad
    assert safety.check(_state(0.7), (5.0,)) == pytest.approx((0.8,))    # 目标 5.0：Δq=4.3，截到 0.7+0.1
    assert safety.check(_state(0.75), (5.0,)) == pytest.approx((0.85,))  # 实测只走到 0.75：重新对实测限幅


def test_out_of_order_commit_rejected():
    buf = ActionBuffer(epoch=1)
    clock = FakeClock()
    p = SinePolicy(); p.reset(epoch=1)
    state = NaiveEstimator().estimate(FakeSensor(clock).latest(0.0), 0.0)
    older = p.act(state, 0.0)
    newer = p.act(state, 0.0)     # 模拟 request #42 比 #41 先完成
    assert buf.put(newer, 0.0)
    assert not buf.put(older, 0.0)                # 旧决策晚到，必须被拒
    assert buf.current(0.0) == newer.values[0]


def test_stale_state_cannot_execute_buffered_action():
    # t=0.0 状态新鲜，policy 产出 A（valid_until=0.32）；t=0.1 传感器死掉。
    # t=0.16~0.30 期间 A 仍未过期，但状态已 stale——控制侧必须拒绝执行 A。
    clock = FakeClock()
    ctrl, safety = run_episode(clock, n_steps=20, sensor=FailingSensor(clock, fail_at=0.10),
                               policy_every=100)
    assert len(ctrl.sent) == 8                 # t=0.00~0.14：状态新鲜，正常执行
    # t=0.16~0.38：状态 stale，A 虽在缓冲里未过期也绝不执行
    assert safety.safe_entries.count("stale_state") == 12
    assert safety.safe_entries.count("no_action") == 0


def test_slow_policy_yields_coverage_gap_then_recovers():
    # 真·延迟注入：决策在 t 发起、t+150ms 才生效。控制节拍不被推理拖慢，
    # 但生效前的空窗里没有可执行决策——coverage 缺口显式落成 safe state。
    clock = FakeClock()
    ctrl, safety = run_episode(clock, n_steps=40, policy_every=10,
                               policy=SlowPolicy(latency=0.15))
    assert len(ctrl.sent) == 32                # 前 8 拍是缺口：150ms 延迟里没有决策
    assert safety.safe_entries.count("no_action") == 8
    assert ctrl.sent                            # 推理完成后控制环恢复执行


def test_epoch_barrier_rejects_stale_episode_action():
    # Episode 41 的迟到决策（seq 更大）绝不能覆盖 Episode 42 的当前决策
    buf = ActionBuffer(epoch=1)
    clock = FakeClock()
    p41 = SinePolicy(); p41.reset(epoch=1)
    state = NaiveEstimator().estimate(FakeSensor(clock).latest(0.0), 0.0)
    p41.act(state, 0.0)
    late = p41.act(state, 0.0)          # episode 41 的第 2 号决策，异步晚到
    buf.sync_epoch(2)                    # reset_episode：episode 42 开始
    p42 = SinePolicy(); p42.reset(epoch=2)
    fresh = p42.act(state, 0.0)          # episode 42 的第 1 号决策
    assert buf.put(fresh, 0.0)
    assert not buf.put(late, 0.0)        # 旧 epoch 晚到：即使 seq 更大也拒绝
    assert buf.current(0.0) == fresh.values[0]
    assert ("rejected_stale_epoch", late.sequence_id) in buf.events


def test_scheduled_chunk_takes_over_at_its_valid_from():
    # A 执行中，B（seq 更大）提前到达但 valid_from 在未来：
    # 0.10~0.20 必须继续执行 A，到 0.20 由 B 在控制边界接管——中间不允许出现空洞
    buf = ActionBuffer(epoch=1)
    a = Action(values=tuple(float(i) for i in range(CHUNK_LEN)), dt=DT, horizon=CHUNK_LEN,
               generated_at=0.0, valid_from=0.0, valid_until=0.32,
               state_stamp=0.0, epoch=1, sequence_id=10)
    b = Action(values=(2.0,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
               generated_at=0.10, valid_from=0.20, valid_until=0.52,
               state_stamp=0.10, epoch=1, sequence_id=11)
    assert buf.put(a, 0.10)
    assert buf.current(0.10) == 5.0      # A 的第 5 点：floor((0.10-0)/0.02)
    assert buf.put(b, 0.10)
    assert buf.current(0.10) == 5.0      # B 还没生效：继续 A，不得返回 None
    assert buf.current(0.19) == 9.0      # floor(9.5)=9：仍是 A
    assert buf.current(0.20) == 2.0      # 到点：B 接管控制边界
    assert buf.current(0.21) == 2.0      # floor(0.5)=0：B 的第 0 点


def test_chunk_indexing_uses_floor_not_round():
    # ZOH：values[i] 覆盖 [valid_from + i*dt, valid_from + (i+1)*dt)，左闭右开
    buf = ActionBuffer(epoch=1)
    a = Action(values=tuple(float(i) for i in range(CHUNK_LEN)), dt=DT, horizon=CHUNK_LEN,
               generated_at=0.0, valid_from=0.0, valid_until=0.32,
               state_stamp=0.0, epoch=1, sequence_id=1)
    buf.put(a, 0.0)
    assert buf.current(0.015) == 0.0     # 0.75 拍：round 会错取 1，floor 留在 0
    assert buf.current(0.020) == 1.0     # 区间边界：进入下一点
```

注意第一个测试的断言强度是"逐位一致"——因为 `SinePolicy` 是 deterministic。把它换成带采样的策略，同一条断言就要按测试策略一节的分级规则降级为容差或分布级。两百多行假实现换来的是：每一次改 contract，这九个测试都会替你盯住接缝有没有破——尤其后四件：乱序提交、真·延迟注入、stale 状态绝不执行缓冲决策、epoch 屏障，是异步推理服务和频繁 reset 接上之后最先咬人的四件事。

## 测试策略：按层建金字塔

具身系统的测试成本从仿真到真机指数上升，所以要让 bug 尽可能死在便宜的层：

| 层 | 测什么 | 手段 |
|---|---|---|
| 单元 | StateContract 序列化/过期逻辑、action 边界、tokenizer 往返一致 | 普通 pytest，毫秒级 |
| 组件 | 单层替换：假观测喂 policy、假 policy 喂控制器 | fake Protocol 实现，内存中跑 |
| 仿真集成 | 整条 loop 在 SimEnv 跑 episode，episode 回放回归 | seed 固定，分级断言（见下文） |
| 运行时故障注入 | policy 延迟/超时/异常、buffer 空/过期/乱序、时钟跳变 | fake scheduler + 时钟注入，断言 deadline 不变量 |
| HIL | 真机 + 安全位姿 + 限速，验接口时序与 watchdog 触发 | 小规模、人在环 |

其中"录 episode 回放"是最值得投资的一件基础设施：把一帧状态存下来，反复喂给 policy，按策略类型分级断言——deterministic policy 要求逐位一致；带采样的策略（diffusion、多数 VLA）固定随机种子后给数值容差（atol/rtol），或做分布级、轨迹级回归。GPU kernel 非确定、量化、编译优化都会引入数值差异，"一律逐位一致"对现代策略并不成立。这件基础设施能拦住大量"代码没变但行为变了"的回归。

故障注入里最值得先做的一件是 **policy 延迟扫描**：用注入时钟把 policy 的产出节奏依次拖到 0 / 50 / 100 / 150 / 500 ms 直至超时，每一档都断言同一条不变量——控制环的 deadline 不超时、过期的 chunk 不被静默沿用、missing 时进入 safe state。VLA 的推理延迟是生产环境的常态而不是异常，这条扫描能把"双频率解耦是否真的成立"从信心变成回归测试。文末最小闭环的最后一个测试就是它的一档简化版。

## Temporal & Failure Contract：机器人 runtime 真正的边界在哪里？

前三条契约管的是"组件之间交换什么形状"，Action 的时间字段和调度协议管的是"组件按什么节奏交换"。这一节处理最后两个问题：**安全判定归谁做、按什么做**，以及**出故障时系统怎么收场**。这两个问题在仿真里几乎免费，在生产里全是真金白银——它们区分的正是"demo 能跑"和"系统能长期跑"。

把这一层的契约再抬高一步看：一个 `Action`（以及一个 `StateContract`）穿过系统边界时，要带齐**四种身份**——**语义**（什么空间、什么单位，`ActionSchema`）、**时间**（哪个世界时刻的决策、何时生效何时失效）、**坐标**（在哪个系里表达，`frame_id`）、**版本**（`schema_version` 指纹，对不上就拒绝消费）。缺任何一种，接缝的另一侧就要靠猜测补全——而猜测是 bug 的别名。前三条契约各自钉住了其中几种，本节把最后两种连同失败语义一起补齐。

### 安全门：一个 runtime gate，不是 policy 里的 if

第 4 层例行的"裁剪一下数值"不叫安全。考虑一个具体例子：目标位置 `q_target = 1.0`，当前 `q_current = 0.1`，控制拍 `dt = 0.001 s`——位置本身完全在工作空间内，但这一步隐含速度是 **900 rad/s**。位置合法不等于速度合法，而速度是否合法取决于**上一拍 committed command**、dt、以及机器人当前健康状态（温度、力矩余量）。所以安全判定的输入天然是 `(state, action, context)` 三元组，不是 action 一个：

```python
# agent/safety/gate.py
@dataclass(frozen=True)
class SafetyDecision:
    allow: bool
    command: ControlCommand | None   # allow 时放行（可能经 project/rate-limit）
    reason: str                      # 拒绝时进 safe state 的原因码

class SoftwareSafeAction(Enum):
    """软件安全态：由机器人自己定义——机械臂 hold 是安全，无人机 land 才是。"""
    HOLD = "hold"                    # 机械臂：保持当前位姿
    BRAKE = "brake"                  # 轮式底盘：制动
    ZERO_TORQUE = "zero_torque"      # 协作臂/四足：卸力
    SIT = "sit"                      # 四足：趴下
    LAND = "land"                    # 无人机：降落
    RETURN_HOME = "return_home"
    # 注意：EMERGENCY_STOP 不在这里——急停是硬件 authority，见 HardwareSafety

class HardwareSafety(Protocol):
    """硬件 authority：急停链、驱动器 STO、独立限位。它不属于软件栈——
    SafetyGate 全体通过时它也在线，软件全挂时它仍然兜得住机器人。"""
    def assert_emergency_stop(self) -> None: ...

class SafetyGate(Protocol):
    """软件控制链上独立的判定层：任何分支都绕不过去。"""
    def evaluate(self, state: StateContract, action: Action,
                 context: dict) -> SafetyDecision: ...
    def enter_safe_state(self, reason: str) -> None: ...
```

`SoftwareSafeAction` 到具体动作的映射由每台机器人的 `SafetyPolicy` 实现——这层抽象写对了，同一套骨架才能从机械臂搬到四足、无人机。两条边界要钉死：急停实现成软件枚举里的一个值，等于假定 Python 进程永远活着，所以 `EMERGENCY_STOP` 归 `HardwareSafety`；`SafetyGate` 是 runtime 的安全判定层，不是认证意义上的 functional safety（ISO 13849 / IEC 62061 那一套）——认证的安全回路必须在软件之外独立存在，软件层只做运行时裁决。

还有一层比"判定层放哪"更容易被忽略：**authority 的写者唯一性**。Protocol 定义的是接缝形状，不是 authority 本身——如果 loop 里每个组件都拿得到 controller 句柄，将来任何一行 `controller.send(raw)` 都悄悄绕过 SafetyGate，"任何分支都绕不过去"就退化成 code review 约定。所以 command authority 的唯一软件 owner 是 Safety：批准的命令经 `CommandSink` 进入 RobotInterface，软件栈里没有任何组件（包括 policy 与 controller 自己）持有 `RobotInterface` 的直连句柄：

```python
# agent/control/sink.py
class CommandSink(Protocol):
    """command authority 的唯一入口：Safety 批准的命令从这里进入 RobotInterface。
    软件栈里没有任何组件（包括 policy 与 controller）持有 RobotInterface 的直连句柄。"""
    def submit(self, cmd: ControlCommand) -> None: ...
```

`Protocol ≠ authority enforcement`：前者靠类型系统表达，后者靠"只有一个写者"的结构保证。链路是 `Policy → Action → ActionBuffer → SafetyGate → CommandSink → RobotInterface`，缺了 sink 这一环，Safety 前面的一切都还有被旁路的可能。

**Fail-closed 是这条链的默认语义。** 安全层自己挂了、心跳丢了、状态过期了，系统的默认行为必须是"禁止 command"，而不是"继续执行最后一条 command"——后者把命运交给了恰好停在缓冲里的旧决策。但 fail-closed 的确切含义要钉准：它撤销的是 **command authority**——禁止新的 command 进入 RobotInterface，**不是把执行器归零**。无人机悬停或缓降是安全的，电机骤停反而可能摔机；机械臂 hold 是安全的，zero torque 可能让它在重力下坠落。fail-closed 的 "closed" 落在"谁还有权下命令"上，不落在"命令内容是什么"上。这条原则有一个推论：任何组件的失效都要能落到预定义的安全态上，Python 进程死了，硬件限位和独立急停链仍然兜得住机器人。

### 故障状态机：不是所有异常都叫 hold_position()

"出问题了先停住再说"在 demo 里够用，在生产里会把系统锁死在一种粗暴行为上。生产系统要的是显式的故障状态机，每个状态有明确的进入条件、退出条件和责任人：

```text
                 ┌─────────────┐
                 │   STARTING  │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │   RUNNING   │
                 └──────┬──────┘
                        │
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
    stale state   policy timeout   controller fault
          │             │             │
          └─────────────┼─────────────┘
                        ↓
                 ┌─────────────┐
                 │  DEGRADED   │
                 └──────┬──────┘
                        │ fault persists
                        ↓
                 ┌─────────────┐   fault persists / 反复未愈   ┌──────────────┐
                 │  SAFE_STOP  │ ───────────────────────────→ │ FAULT_LATCHED │
                 └──────┬──────┘                              └──────┬───────┘
                        │ operator / recovery                       │ manual reset only
                        ↓                                           ↓
                 ┌─────────────┐                              （人工介入，
                 │   RECOVER   │                               不允许自动退出）
                 └─────────────┘
```

三条语义要钉死：**DEGRADED → SAFE_STOP 是"故障持续超时"的时钟，不是"再试一次"的循环**——降级状态是给恢复留的窗口，不是给重试冲量用的；**SAFE_STOP → RECOVER → RUNNING 必须经过 operator 或显式 recovery 流程**，不允许无人值守地自动回到 RUNNING，否则一个间歇性故障就能把系统永动在"故障—恢复—再故障"的振荡里；**SAFE_STOP → FAULT_LATCHED 是单向的**——故障持续超时或反复未愈，系统锁存到人工复位。latch 和 safe stop 的区别正是"还能自动恢复"与"只能人来复位"：前者留窗口，后者要求人先看清发生了什么。

### 延迟预算与动作覆盖率：把"快不快"变成可测的账

故障语义之外，Temporal Contract 还有一本更日常的账：延迟。从传感器到执行器的全链路预算是

```text
L_total = L_sensor + L_transport + L_sync + L_estimator
        + L_policy + L_queue + L_safety + L_controller  <  L_budget
```

其中 `L_budget` 是控制周期（200 Hz 即 5ms）。每一项都要能单独测量、单独报警——"系统卡了"不是一个可操作的诊断，"L_policy 的 p99 从 80ms 涨到 210ms"才是。预算被突破时该报警的是哪一环，这张图就是答案。

比 policy 帧率更值得盯的指标是**动作覆盖率**：控制环每个节拍记账 `valid_until - now`——当前消费的决策距离失效还有多久。它的分布（尤其 p05）比 policy FPS 更能预测事故：FPS 高但 chunk 短，覆盖率照样可能见底；覆盖率长期贴着零走，说明 policy 的产出节奏已经在贴着消费端跑，任何抖动都会直接变成 missing。配套地要区分两种 deadline 的硬度：**policy 的 deadline 是软契约**——超时意味着决策变旧、质量降级，由 staleness gate 兜底；**控制的 deadline 是硬契约**——超时直接违约，safe state 是它的违约处理。把两种 deadline 混为一谈，是"优化 policy 延迟"时最常见的目标函数错位。

### Supervisor：六层之外横切的一层

上面的状态机不会自己运转。生命周期、健康心跳、deadline 监控、故障转移、metrics、日志、artifact 校验——这些关注点不属于任何一层，又横切所有层，归一个独立的 **Runtime Supervisor**：

```text
┌──────────────── Runtime Supervisor ────────────────┐
│ lifecycle · heartbeat · deadline · fault · metrics │
└──┬─────────┬──────────┬─────────┬──────────┬───────┘
 Sensor    Estimator   Policy   ActionBuffer Controller
```

它回答的是这一类问题：policy 线程 200ms 没产出，是 VLA 本来就慢还是已经死锁？estimator 的 quality 降级有没有被记录成指标？重启一个组件之后，artifact manifest 和 config hash 还对不对得上？双频率 loop 的示意图里没有画它，不是因为不需要，而是因为把它画进去图就乱了——横切关注点和分层架构是正交的两件事。范围也要收窄：Supervisor 只做看护、不做决策——它发现 policy 超时并把系统搬进 DEGRADED，但不替任何一层决定行为，决策权永远留在层内；把它做成"中央大脑"，只是换了一种形式的大泥球。但故障状态机本身必须有唯一 owner，否则 policy 线程写 DEGRADED、控制线程写 SAFE_STOP、Supervisor 写 FAULT_LATCHED，三个写者就又回到了并发 authority 的老问题——**Supervisor 是 RUNNING / DEGRADED / SAFE_STOP / FAULT_LATCHED 的唯一状态 owner，其他组件只能 `report_fault(event)`**（`POLICY_TIMEOUT` / `STATE_STALE` / `CONTROLLER_FAULT` / `HEARTBEAT_LOST`），不能自己 `set_state(...)`。

最后一件容易被忽略的事是 **episode 生命周期**。一次正式实验的边界不是 `for` 循环的起止，而是显式的状态转移：`initialize → start → reset_episode → run → stop → fault/recover → shutdown`。其中 `reset_episode` 有一个容易被漏光的重置清单：**policy 的隐藏状态、estimator 的滤波器、action buffer、controller 的 tracking 状态、随机种子**。漏掉任何一项，上一个 episode 的残留就会污染下一个 episode 的决策——回放不复现，往往先是这里出了问题。

## 六个常见的工程反模式

都是真实项目里反复出现的坑，按出现频率排序：

1. **大泥球 import**：模块互相 import 实现，换任一组件牵一发动全身。解法是强迫每层只依赖 Protocol，import 方向单向。
2. **裸 dict / 裸 tensor 满天飞**：状态没有 schema，字段名靠口头约定。解法只有一个——StateContract，并在边界处校验。
3. **训练/部署两套 preprocessing**：skew 的根源。解法前面给了：共用 `ObsTransform` 模块，禁止复制实现。
4. **安全逻辑散在 policy 里**：把安全决策交给了学习系统。解法是独立 safety 层 + watchdog，policy 永远可以被旁路；但软件安全层之上还有硬件限位和独立急停，两者别互相替代。
5. **把 policy 和控制塞进一个循环**：VLA 一次 150ms 推理就拖垮整个 200 Hz 控制环。解法是分线程、分频率——policy 产 chunk 进动作缓冲，控制环只消费缓冲，两个频率在配置里分开声明。
6. **仿真和真机各写一套环境代码**：接口差异被人为放大。解法是同一个 RobotInterface，proprio 字段名都从一份 schema 生成——它管接口统一，gap 本身另靠 randomization 和真机验证（见 Env 一节）。

## 总结

这篇把"具身智能缺的是接口"从判断落成了骨架：六层运行时栈 + 一条训练管线，靠四组契约接缝——**State**（状态的形状：观测、本体感受、质量分）、**Action**（时间化的动作：ActionSchema 语义指纹、valid_from/valid_until/epoch/sequence_id 与 active/scheduled 双槽 + epoch 屏障的 ActionBuffer 调度协议）、**Env**（RobotInterface + TaskEnv 拆开 sim 与任务）、**Temporal & Failure**（四种身份、安全门与 command authority、fail-closed、故障状态机与 FAULT_LATCHED、延迟预算与动作覆盖率、Supervisor、episode 生命周期）。VLA、Diffusion、MPC 在同一份代码里只是 Policy 插槽的三个实现类。骨架的价值不在代码本身，在于它把"换路线、换传感器、换机器人"从外科手术降级为换插件——而这恰恰是评估协议（9/12~9/16 那一串）能落地的前提：只有接缝清晰，compliance 才有可插拔的测量点。

下一步可以往两个方向展开：一是把这套骨架的某个 Protocol 做成完整可运行实现（比如 MPC 那一支，配世界模型 rollout）；二是按 9/12 的评估协议，给每一层接上对应的 evidence 采集。需要哪个，评论区告诉我。
