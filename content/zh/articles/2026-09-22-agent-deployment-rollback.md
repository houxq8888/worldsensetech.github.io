---
title: "具身智能的部署与运维：换组件不是发一次版本，是养一台状态机"
slug: "2026-09-22-agent-deployment-rollback"
date: 2026-09-22
draft: false
categories: ["具身智能", "教程"]
tags: ["具身智能", "软件架构", "机器人", "部署运维", "VLA", "Python", "系统设计", "工程架构", "Sim-to-Real"]
description: "9/17 给出了能跑、能测、能换组件的 Agent 骨架，这篇回答下一问：换下去的那一瞬间靠什么兜底。Web 服务回滚是撤销，机器人回滚是让旧决策失权——发布身份（分层 manifest + 签名语义）、code×schema×config 的兼容性网格（包含而非相交）、shadow 五维分歧台账与限幅差异、带晋升门槛的 canary、先失权再切指针的 epoch 屏障回滚、fail-before-motion 的配置热加载，附一个纯 stdlib、十五个 invariant 的发布状态机最小闭环。"
toc: true
related_articles:
  - 2026-09-19-embodied-agent-architecture
  - 2026-09-16-policy-side-evaluation
  - 2026-09-12-sim-to-real-evaluation-protocol
  - 2026-09-15-policy-side-interface
  - 2026-09-09-robot-data-scaling
---

Web 服务的回滚是撤销——把流量指回旧版本，错误率掉下去就算完。机器人的回滚是让旧决策失权——把 manifest 指针切回去，那个版本还有一串在途的 chunk、排队中的异步推理、没清零的隐藏状态攥着控制权不放。这一篇就写这层落差。

上一篇把骨架立起来了：[六层运行时栈、三条核心契约、能跑能测的最小闭环](/zh/articles/2026-09-19-embodied-agent-architecture/)，结论是"换路线、换传感器、换机器人从外科手术降级为换插件"。但"可以插拔"和"敢拔敢插"之间还隔着一整层没写出来的东西——**拔下去的那一刻，谁兜底？**

换一块板子，焊错了可以返工；换一份 policy 权重，机器人正在端着一条玻璃杯。它和 9/17 是同一套写法——先把概念给到能讨论，文末再用一套纯 stdlib 的假实现把发布状态机真正跑起来（十五个 invariant，本文付印前实跑：15 passed）。这篇区别于普通 DevOps 的地方，一句话能说完：**它把 deployment 当作机器人控制安全边界的一部分**，所以整篇其实是一条链——

```text
release_id -> compatibility -> shadow evidence -> canary -> epoch barrier -> rollback
   身份           能不能装         装了动不动         给谁跑      旧决策失权       出事退得回去
```

身份对不齐，后面全没有可复盘的账；兼容性没实测，装上去就是带病运行；没有 shadow 台账，晋升靠的是胆子；没有 epoch 屏障，回滚就是新旧两个版本抢同一台机器人。下面一节一环。

## 先立一个总原则：部署不是一个动作，是一台带证据的状态机

Web 后端的部署近似于"把新代码放上去，看错误率"。这个直觉搬到机器人上是危险的，危险有三层。第一，**回滚窗口不对等**：web 回滚是撤销，机器人回滚是从一个已经执行了错误动作的物理状态里恢复——杯子已经掉了。第二，**观测面不对等**：错误率和延迟曲线是集中式的，机器人的"对不对"分散在每台机器的每个 episode 里，等你看到现场事故，样本早已消费掉了。第三，也是本篇要反复回来的：**一次部署改变的不只是代码**——ckpt 换一版，归一化统计、ObsTransform、标定、控制阈值往往都在动，这些"看不见的版本"没有任何一行报错，只会让行为静静地漂移。

所以部署的正确建模不是动作，是**状态机 + 证据链**：每一次版本移动都要留下可复查的凭证（谁构建的、对着哪份 schema、跑过哪些校验、和旧版对拍过分歧没有），状态只允许沿合法边转移，非法转移被结构拒绝而不是靠流程自觉。眼熟吗？这正是 9/17 给运行时立的规矩——command authority 单写者、故障状态机单 owner、epoch 屏障——现在原样搬到了发布通道上。架构篇的接缝设计得越好，这一篇的运维才越便宜，这是两篇真正的咬合处。

## 发布身份：manifest 里每个字段都是防呆开关

9/17 给过一张 `deploy/artifact_manifest.yaml`：checkpoint、code_commit、schema 版本、归一化统计、标定文件、容器 digest。那时它的职责是"复现一次部署"；这一篇它要多干一份活——**充当发布状态机的输入**。字段本身不新鲜，新鲜的是给每个字段配的那道闸：

```yaml
# deploy/release_manifest.yaml —— artifact_manifest 的运维增强版
release:
  release_id: 2026.09-r7        # 单调发布序号：所有事件流拿它对齐（见下文）
  checkpoint: ckpt/diffusion_v17.pt
  artifact_sha256: 9b41...      # 权重哈希：tag 会变，哈希不会
  code_commit: 8f3a2c1
  signature: minisig:r7.ok      # 来路签名：覆盖 canonical(manifest − signature)，不只是签 artifact
compat:
  state_schema: [v3, v4]        # 声明支持的 schema 区间：必须整体落在 runtime 接受范围内（包含，非相交）
  action_schema: [v2]
  obs_fingerprint: fp-v7        # 语义指纹 + golden vectors 的合成哈希（9/17）
  config_range: [3.1.0, 3.2.0]  # 前向 + 后向兼容区间：min/max 都有立场，过旧也可能不兼容
runtime:
  container_digest: sha256:71c0...
  python: "3.11"
policy:                          # 发布策略入 manifest：晋升门槛是数据，不是散在代码里的常量
  baseline_release: 2026.09-r6  # 与谁对拍
  divergence_budget: 0.02       # 晋升门槛：分歧率上限
  min_ticks: 10000              # 晋升门槛：最小样本量
```

这张清单在代码里不是一坨平铺字段，而是**分成三层、和 YAML 的三段一一对应**：`ArtifactIdentity`（哈希 + 签名）管"这是哪个件、谁造的"，`CompatibilityContract`（schema 区间 + config 区间 + 观测指纹）管"能不能和当前 runtime 一起跑"，`ReleasePolicy`（min_ticks + divergence_budget）管"要攒够多少证据才许晋升"。分层不是洁癖——它让 `boot_check`、`promote_allowed`、`rollback` 各自只读自己该读的那一段，晋升门槛从 manifest 的 policy 字段里取，而不是三个魔法常量散落在状态机里。

三个身份字段各自挡一类事故，而且要分清它们**不是一回事**。`artifact_sha256` 是**内容身份**，防"名字对了东西不对"：`diffusion_v17.pt` 是个名字不是身份，同名不同内容的 ckpt 是所有"见鬼了怎么行为不一样"案发现场的常客。`signature` 是**来路身份**（authenticity / provenance），防"内容没被换、但被谁签的不明不白"——关键在于签名覆盖的应当是 `canonical(manifest − signature)` 这一整段规范化字节，而不只是 artifact 的哈希；否则会出现"哈希没变、但有人偷偷改了 compat 里的 schema 区间"这种绕过。`release_id` 是**发布事件身份**，防事件流对不齐：运维事件（部署、对拍、灰度、回滚）跨 CI、注册中心、每台机器人三个时钟域，没有单调序号就没有可复盘的账。`obs_fingerprint` 则防定义漂移：9/17 里它管 train/serve skew，在这里它管 release/serve skew——新 ckpt 配旧预处理，字段一个不缺、行为整个反了，靠逐元素比对数值当场就拦。评估协议（[9/12](/zh/articles/2026-09-12-sim-to-real-evaluation-protocol/)）里 traceability 的落地形态，落到部署层就是这一张清单加这几道闸。

还有一组容易混的身份刻度，值得在这里一次性钉死，因为它们各自解决一种对齐问题：`release_id`（哪一次发布）、`robot_id`（哪台机器）、`episode_id`（哪一次作业）、`epoch`（这次作业里第几段被屏障划开的生命周期）、`sequence_id`（段内第几个决策）。特别要分清 **`release_id ≠ epoch`**：一次发布可以横跨很多个 episode，一个 episode 里也可能因为回滚连升好几个 epoch。事故复盘时它们是嵌套的坐标——`release r7 / episode 42 / epoch 8 / sequence 183`，一旦回滚就跳到 `release r6 / episode 43 / epoch 9 / sequence 0`。少了任何一层，你都没法在日志里精确指认"到底是哪一个决策把杯子碰掉的"。

## 兼容性网格：不是向后兼容，是"哪些格子能一起跑"

版本管理最常见的错误假设是线性：新代码兼容旧数据，旧代码兼容新数据，一路向后。具身系统的版本至少在三个独立轴上移动，彼此不正交：

```text
            schema  v3      v4      v5
code        ─────────────────────────────
  1.4.x      ✓        ✓      ✗      ✗
  1.5.x      ✗        ✓      ✓      ✗
config      ─────────────────────────────
  3.1        ✓        ✓      ✗      ✗
  3.2        ✗        ✓      ✓      ✗
             ↑ 每个格子 = 一次真实跑过的 boot 校验，没测过的格子默认是 ✗
```

三点纪律。**一，网格只给"真实存在过的组合"**。总想维持全兼容矩阵的团队，最后谁也没测全。机器人 fleet 的典型真相是：一台机器上 runtime 落后注册中心一个版本是常态——所以兼容区间必须显式建模（`compat` 的 `[v3, v4]`），而不是靠"应该没问题吧"。**二，没测过的格子默认是 ✗**。兼容性是测出来的属性，不是声明出来的属性；每个 ✓ 背后要有一次真跑过的 `boot_check` 加一轮 shadow 对拍。这句话还得追一句现实的坑：**那个 ✓ 到底记在谁头上？** 如果 ✓ 只是注册中心里一个人手敲进 YAML 的布尔值，它迟早和现实脱节。可信的做法是让 ✓ 携带证据引用（哪次 CI、哪轮 shadow、哪个测试套件跑过），由流水线写、不给人手改——这也正是下一篇 fleet 版本治理要接的接口。**三，配置也要做前向兼容**。灰度回滚时，旧版 runtime 往往要加载新版已经改过的配置（episode 42），字段改名或语义变化，旧代码就地崩——这是最容易在回滚当晚补第二个故障的路径。所以 config 变更遵守标准的 expand-migrate-contract：只加不删、新字段给默认值、删除排在所有机器都升级之后；manifest 里的 `config_range` 有 min 也有 max，就是这条纪律的强制闸——过旧和过界同样可疑。

一个建模上的坦白：`compat_within` 把版本当成**单调连续的语义区间**做包含判定（声明支持的整段必须都在 runtime 接受段内），这在 schema 确实是线性演进时成立。但真实世界里版本往往是**离散的 capability**——支持 `{v3, v4}` 不代表支持一个介于其间、字段却不同的伪版本。那种情况要把它换成 capability 集合的交集判断，而不是区间。本文为了最小闭环选了区间模型，并在代码注释里留了这个口子；读者若你的 schema 是离散 ID，请把"区间包含"替换成"集合包含"。注意我特意用的是**包含而非相交**：manifest 声明支持 `[1.2, 2.0]`、runtime 只吃 `[1.8, 3.0]`，两者相交但绝不能判兼容——因为 runtime 根本不认 manifest 承诺的 `1.2~1.8` 那半段，相交判定会放它过闸。

启动时的校验顺序也值得定死，从便宜到贵：**先结构**——manifest 字段齐不齐、有没有 `release_id`、schema 区间是否被 runtime 完整覆盖、config 版本是否越界，纯比对，毫秒级；**再数值**——用 release manifest 里的 golden vectors 重跑一遍共用 `ObsTransform`，比对语义指纹（9/17 那套手法在部署时点的复用）；**最后行为**——进 shadow mode 对拍（下一节）。宁可拒绝启动，不可带病运行：fail-before-motion 说了一万遍，败就败在"先跑起来看看"。

## Shadow mode：决策对拍，永不落地

发布系统里最便宜的保险，是让新 policy **看得到一切、碰不到任何东西**。对同一份 `StateContract` 流，新旧两个 policy 各自产 `Action`，旧的照常进 SafetyGate → CommandSink 落地执行，新的进影子台账：

```text
StateContract 流 ──→ active policy ──→ ActionBuffer → SafetyGate → CommandSink → 真机
        └──────────→ candidate policy ─→ 分歧台账（只记账，sink 里没有它的路）
```

影子模式的实现纪律恰好是 9/17 两条老规矩的复用。其一，**authority 唯一**：candidate 的输出没有通向 RobotInterface 的路径——准确说，在 Python 这种动态类型下它不是"类型上进不去"，而是"结构上没有那条调用路径"：`ShadowRunner.tick()` 里 `candidate.act()` 的返回值只流向分歧台账，唯一调用 `sink.submit()` 的地方喂的是 `active` 经 `safety.check()` 的产物。类型级强制（把 `CandidateAction` 和 `Action` 分成不同类型、让 sink 只接受后者）是生产版该补的，这里不谎称已经做到。其二，**对拍要同状态同时钟**：两边喂的是同一拍 `latest_valid(now)` 的同一个状态对象，用同一个注入时钟；影子跑在旧状态上算出来的分歧全是噪声。

台账记的不该只有一个数。分歧是一维以上的事：**数值一致 ≠ 行为一致**。真正的危险常常不是 `0.50` 对 `0.52`（那可能只是噪声），而是同一个动作 `valid_from` 晚了 100ms——那是**另一个 chunk**，接管的物理时刻错了。所以 `Divergence` 拆成五个维度：`value`（数值超 ε）、`validity`（valid_from / chunk 时长错位）、`horizon`（chunk 覆盖时长不一致）、`sequence`（接管节奏，sequence_id 步进是否一致）、`safety`（限幅触发差异）。最容易被漏的是最后一维，也是最该盯的：**限幅触发差异**——同一条命令，SafetyGate 对老 policy 放行、对新 policy 夹了钳，哪怕任务成功率暂时看不出来，它就是新 policy 更激进的直接证据。实现上让 active 和 candidate 都过一次 `safety.preview()`（只判定、不落地），比较两者的 `clamped` 标志，`clamp_diffs` 就真在计数了；candidate 看得到安全门的裁决，却依旧碰不到 sink——这恰好同时演示了"能观察、不能执行"。分歧率是风险读数，不看它的灰度等于裸奔。

shadow 阶段也有两个失效条件要提前防。**候选不能碰状态**：影子 policy 不许往 estimator / StateBuffer 回写任何东西，否则"只观察"就变成了共同决策。**对拍要挑边界工况**：均匀时间轴上两个 policy 当然大部分时候一致——分歧台账要按 manifest 里声明的任务标签分层（接触、遮挡、标定漂移），对拍窗口盖不住边界工况，等于没拍。

## 灰度与晋升门槛：分歧台账说了算

对拍通过，才轮到真实执行。机器人场景的 canary 和 web 的百分比流量是两回事——**流量的天然切片不是用户，是任务、机器人和时段**：先影子，再一台机器的单一任务，再同型号 fleet，再跨型号；每一步都是"真实执行 + 故障预算"，区别只在爆炸半径。

晋升的规矩要定在前面：**门槛是数据不是勇气**。`release_manifest.yaml` 里那两个字段——`min_ticks` 和 `divergence_budget`——就是晋升闸门的门轴：样本量不够不许升（哪怕分歧率是零），分歧率超预算不许升（哪怕样本已经足够）。最忌讳的形态是"再观察观察"：没有量化门槛的观察，最后都会滑向周五下午的"看起来没问题，上吧"。回滚判据同理，而且要分硬度：安全门频繁介入、分歧率越过 hard threshold 是**立即回滚**，不讨论；覆盖率、成功率缓慢劣化是**预算内回滚**，按窗口计。两类判据都要在发布之前写成可计算的表达式，而不是事后开会找共识。

一句范围声明，免得读者误会：本文那套 pytest 验证的是**发布状态机的阶段约束与证据结构**——哪些状态之间能合法转移、晋升要不要攒够证据、回滚要不要重过闸。它**不模拟 fleet 级 canary 调度器**：机器选择、任务配比、观察窗口到点自动晋升，这些是 fleet registry 的对象，属于下一篇。这里出现的 `CANARY` 只是一个被状态机保护着的阶段名，不是一台调度器。

## 回滚：9/17 的 epoch 屏障在这里第二次上岗

回滚难，不在切配置。配置秒切，切完真正的麻烦浮出来：**在途的东西**——ActionBuffer 里旧版本产出的 chunk（scheduled 槽里还躺着一份）、policy 服务排队中的异步推理、正在执行的轨迹片段、以及 policy 的隐藏状态。只切 manifest 指针的回滚，等于是新旧两个版本共同驾驶同一台机器人——旧版本的最后一个决策还在缓冲里排队，新版本的第一个决策已经进场，两边的 sequence_id 还各算各的。

所以回滚不是一行赋值，是一台**顺序即语义**的小型事务，五步一步都不能换：① 停发新命令，关掉 command authority；② `epoch + 1`，让旧版本的在途决策**结构性失权**——不等"它们执行完"，也轮不到它们覆盖回滚后的版本；③ 进入 `SAFE_STOP`，机器人不悬在中间态；④ 旧 manifest **必须重新过一遍 `boot_check`**；⑤ 才切指针 + 走一次完整的 `reset_episode`（清 policy 隐藏状态、滤波器、tracking），回 `VERIFYING` 重走证据链。9/17 设计的 epoch 屏障正是为第②步准备的：epoch 加一再同步给 policy 与 ActionBuffer 之后，旧版本所有在途决策——不管它的 sequence_id 多大、离生效还有多远——在 `put()` 的 epoch 屏障前一律拒绝；序号空间重新起算，不存在跨版本的序号纠缠。"等旧 chunk 执行完再切"是不需要的，也是不允许的：屏障让旧版本失权，这不是快，这是干净。

第④步是全文最容易漏、也最要命的一条：**"旧版本"不等于"可回滚版本"**。你回滚到的那个 release，它的 schema、config、obs_fingerprint 未必还兼容**当前**的 runtime——尤其如果你已经先升了 runtime。回退前对旧 manifest 重新跑一次兼容性校验，过不了就停在 `SAFE_STOP` 求助，绝不硬切：硬切一个不兼容的旧版本，等于用一个新 bug 替换旧 bug。代码里这一步失败会抛 `RollbackRejected`，状态机稳稳停在 `SAFE_STOP`，指针一个字没动。

这里还牵出一条贯穿两篇的架构规矩：**epoch 的唯一发布者是 RuntimeCore**。supervisor、policy、ActionBuffer 只**接受** epoch、同步镜像，谁都不许自己 `epoch += 1`。否则就会出现 supervisor 看见 7、policy 还停在 7、buffer 停在 6 的半同步地狱——回滚恰恰是在这种不一致里最容易翻车的时刻。`rollback()` 里递增 epoch 的只有 `runtime_core`，supervisor 通过 `note_epoch()` 事后取镜像写进事件，正是这条 single-owner 纪律的落地。

为了审计，我还给状态机补了一个 `ROLLING_BACK` 态：光是 `ACTIVE → SAFE_STOP` 分不清"真故障""人工 E-stop""回滚""watchdog""controller timeout"。有了显式回滚态（半途失败还能落回 `SAFE_STOP`），再配上每条转移都带 `reason` 的事件，事后才问得出"这台机器昨晚到底经历了什么"。而"回滚路径本身要测"这条反直觉的纪律——没执行过回滚的发布计划，等价于没有回滚——在文末就是一行 pytest。

## 配置与阈值热加载：改阈值不是改代码，但比改代码更危险

部署要换的是 artifact，运维要换的常常只是阈值：速度上限、validity 窗口、控制频率。热加载配置诱人，因为它绕过了整个发布通道；它危险，因为**阈值就是安全参数**。三条规矩：配置版本进 manifest，改一个阈值也算一次发布（走同一台状态机，没有捷径）；热加载走 **prepare → validate → commit**，校验失败保持旧值并记事件，绝不允许半新半旧的状态被控制环读到；所有阈值标注生效时点（下一拍 / 下一个 episode），控制中段的参数切换要像 ActionBuffer 的接管一样只发生在边界上。"小改动"三个字在事故复盘里的出现频率，高得离谱。这一条文末也用 `ConfigManager` 落成了能跑的断言：非法阈值 commit 之后，`current` 必须一字不动。

## 补到能跑：一台发布状态机的最小闭环

上面都是设计。这一节把发布通道做成能跑能测的最小形态——仍然不用 torch、不用 GPU，复用 9/17 那套 fakes（时钟注入、确定性 policy、epoch 屏障、CommandSink 单写者全部原样在场），只加几样东西：分层的 `Manifest` / `RuntimeVersion`（发布身份与兼容区间）、带证据链的 `ReleaseSupervisor`（唯一写者，每条转移落成一行 `ReleaseEvent`）、按性质分流的 `Fault`、同状态五维对拍且限幅可观察的 `ShadowRunner`、`boot_check` / `promote_allowed` / `ConfigManager` / `rollback`（从装到退的四道闸）。先看骨架代码：

```python
# tests/deploy_fakes.py —— 部署与运维的最小闭环：纯 stdlib，pytest 直接跑
# 与 9/17 的 fakes.py 同一套纪律：时钟注入、确定性 policy、epoch 屏障、单写者状态机。
# 范围声明：本文件只验证发布状态机的阶段约束与证据结构，不模拟 fleet 级 canary 调度器
# （机器选择、任务配比、观察窗口自动晋升都是 fleet registry 的对象，见文末）。
from dataclasses import dataclass, field
from typing import Optional, Tuple


def parse_version(s: str) -> Tuple[int, ...]:
    """int.int.int 的最小版本解析：刻意不叫 semver——不处理 -rc1 / +build7
    预发布元数据，叫对名字，别向规范碰瓷。"""
    return tuple(int(x) for x in s.split("."))


def compat_within(declared: Tuple[str, str], accepted: Tuple[str, str]) -> bool:
    """包含判定，不是相交判定：产物声明支持的整段区间必须都在 runtime 可接受范围内。
    相交会把 manifest 声称支持、但 runtime 根本不吃的那半段（如 manifest [1.2,2.0] ∩
    runtime [1.8,3.0] 里的 1.2~1.8）误判为兼容。连续语义版本假设见正文：这里把版本
    建模成单调连续区间——真实的离散 capability 协商应换成 frozenset 交集（见正文）。"""
    return (parse_version(accepted[0]) <= parse_version(declared[0])
            and parse_version(declared[1]) <= parse_version(accepted[1]))


# ---------------------------------------------------------------- 发布身份与 manifest

@dataclass(frozen=True)
class ArtifactIdentity:
    artifact_hash: str           # 内容身份：同名不同物由它拦
    signature: str               # 来路身份：签名覆盖 canonical(manifest − signature) 字段，
                                 # 防"哈希没变、compat 被改"——hash 认内容，signature 认发布者


@dataclass(frozen=True)
class CompatibilityContract:
    schema_range: Tuple[str, str]    # 必须整体落在 runtime 的 accepts_schema 内（包含，非相交）
    config_range: Tuple[str, str]    # min/max 都有立场：过旧配置同样可能不兼容
    obs_fingerprint: str             # 语义指纹 + golden vectors 的合成哈希（9/17）


@dataclass(frozen=True)
class ReleasePolicy:
    min_ticks: int                   # 晋升门槛①：样本量
    divergence_budget: float         # 晋升门槛②：分歧率上限


@dataclass(frozen=True)
class Manifest:
    """deploy/release_manifest.yaml 的内存形态：与 YAML 的 release/compat/policy 三段一一对应。
    manifest 是发布状态机的输入——晋升门槛从 policy 字段里读，不是散在代码里的常量。"""
    release_id: str                  # 单调发布序号：事件流的对齐轴
    code_version: str
    artifact: ArtifactIdentity
    compat: CompatibilityContract
    policy: ReleasePolicy


@dataclass(frozen=True)
class RuntimeVersion:
    """机器人上这一版 runtime 能接受什么，说清楚。"""
    code_version: str
    accepts_schema: Tuple[str, str]
    accepts_config: Tuple[str, str]
    expected_fingerprint: str


def boot_check(manifest: Manifest, runtime: RuntimeVersion) -> Optional[str]:
    """启动与回滚共用的一道闸：失败返回原因，通过返回 None。宁可拒绝，不可带病运行。"""
    if not manifest.release_id:
        return "missing_release_id"          # 没有身份的事件流不可复盘，直接拒
    if not compat_within(manifest.compat.schema_range, runtime.accepts_schema):
        return "schema_incompatible"
    if manifest.compat.obs_fingerprint != runtime.expected_fingerprint:
        return "fingerprint_mismatch"
    if not compat_within(manifest.compat.config_range, runtime.accepts_config):
        return "config_incompatible"
    return None


# ---------------------------------------------------------------- 发布状态机与证据链

TRANSITIONS = {
    "INSTALLED": {"VERIFYING"},
    "VERIFYING": {"SHADOW", "SAFE_STOP"},
    "SHADOW":    {"CANARY", "SAFE_STOP"},
    "CANARY":    {"ACTIVE", "SAFE_STOP"},
    "ACTIVE":    {"DEGRADED", "SAFE_STOP", "ROLLING_BACK"},
    "DEGRADED":  {"ACTIVE", "SAFE_STOP", "ROLLING_BACK"},
    "ROLLING_BACK": {"VERIFYING", "SAFE_STOP"},   # 半途失败允许落回 SAFE_STOP
    "SAFE_STOP": {"VERIFYING"},         # 故障/回滚之后重走证据链，不许直接回 ACTIVE
}


class Fault:
    SAFETY = "safety"                  # 过限幅 / stale / controller timeout / e-stop → SAFE_STOP
    RELEASE = "release"                # 分歧率抬头 / 成功率下滑 / P95 劣化 → DEGRADED + 回滚预算
    OBSERVABILITY = "observability"    # 指标断流 / 台账写失败 → 冻结晋升，不动当前执行态


@dataclass(frozen=True)
class ReleaseEvent:
    """证据链的最小单元：每条转移都带身份、时间、原因和 epoch——
    审计问"哪个版本在什么时候以什么理由进了哪个状态"，这一行就是答案。"""
    release_id: str
    timestamp: float                   # 注入时钟：与 9/17 同一条 clock 纪律
    from_state: str
    to_state: str
    reason: str                        # 操作意图或故障事件："rollback" / "policy_timeout"…
    epoch: int                         # 发生时 runtime 的 epoch：复盘时串起动作流
    evidence_ref: Optional[str] = None # 指向对拍台账 / CI run 的凭证引用


class ReleaseSupervisor:
    """发布状态机的唯一写者（9/17 Supervisor 纪律的部署版）：
    其他组件只能 request_transition / report_fault，不许自己改状态。
    每条合法转移落成一行 ReleaseEvent；非法转移拒绝且不记账。"""

    def __init__(self, clock, release_id, epoch=0):
        self._clock = clock
        self.release_id = release_id
        self._epoch = epoch            # 只读镜像：epoch 的唯一发布者是 RuntimeCore（见 rollback）
        self.state = "INSTALLED"
        self.events = []

    def note_epoch(self, epoch):
        self._epoch = epoch            # RuntimeCore 递增后同步镜像，供下一条事件使用

    def _move(self, to: str, reason: str, evidence_ref=None) -> bool:
        if to not in TRANSITIONS[self.state]:
            return False
        self.events.append(ReleaseEvent(
            self.release_id, self._clock.monotonic(), self.state, to,
            reason, self._epoch, evidence_ref))
        self.state = to
        return True

    def request_transition(self, to: str, reason: str, evidence_ref=None) -> bool:
        return self._move(to, reason, evidence_ref)

    def report_fault(self, kind: str, reason: str, evidence_ref=None) -> str:
        """故障按性质分流，不是一律 SAFE_STOP：
        safety 碰物理边界，必须停；release 是质量读数，先 DEGRADED 再走回滚预算；
        observability 只是眼睛瞎了——冻结晋升、不打断正在安全执行的控制环，但必须留痕。"""
        if kind == Fault.SAFETY:
            self._move("SAFE_STOP", reason, evidence_ref)
        elif kind == Fault.RELEASE:
            self._move("DEGRADED", reason, evidence_ref)
        else:                          # OBSERVABILITY：状态不动，事件照记
            self.events.append(ReleaseEvent(
                self.release_id, self._clock.monotonic(), self.state, self.state,
                reason, self._epoch, evidence_ref))
        return self.state


# ---------------------------------------------------------------- 影子对拍：分歧台账

@dataclass
class Divergence:
    """分歧不是一维的：数值一致 ≠ 行为一致。
    0.50 vs 0.52 可能是噪声，valid_from 晚 100ms 是另一个 chunk——危险藏在时间语义里。"""
    value: bool = False                    # 动作数值超 ε
    validity: bool = False                 # valid_from / chunk 时长错位
    horizon: bool = False                  # chunk 时长（horizon×dt）不一致
    sequence: bool = False                 # 接管节奏（sequence_id 步进）不一致
    safety: bool = False                   # 限幅触发差异：active 放行、candidate 被夹


@dataclass
class ShadowStats:
    ticks: int = 0
    divergences: int = 0
    clamp_diffs: int = 0                 # safety 维度计数：新 policy 更激进的直接证据
    validity_diffs: int = 0
    ledger: list = field(default_factory=list)   # 逐拍 (now, Divergence)：台账先于汇总

    @property
    def divergence_rate(self) -> float:
        return self.divergences / self.ticks if self.ticks else 0.0


class ShadowRunner:
    """同一份状态喂两个 policy，candidate 的输出永不进 sink——影子模式的判定：
    只记账，不落地。落地的那一条走 safety.check → sink.submit，与生产同一条 authority 路。
    说明：Python 的动态类型不强制 candidate 进不了 sink，本实现靠的是"唯一调用路径"
    结构——candidate.act() 的返回值只流向台账；类型级强制留给生产版（见正文）。"""

    def __init__(self, active, candidate, safety, sink, divergence_eps=1e-6):
        self.active = active
        self.candidate = candidate
        self.safety = safety
        self.sink = sink
        self.eps = divergence_eps
        self.stats = ShadowStats()
        self._last = None                # (active_seq, candidate_seq)：接管节奏比对基线

    def tick(self, state, now) -> bool:
        a = self.active.act(state, now)
        c = self.candidate.act(state, now)
        d = Divergence()
        d.value = max(abs(x - y) for x, y in zip(a.values, c.values)) > self.eps
        chunk_a, chunk_c = a.horizon * a.dt, c.horizon * c.dt
        d.validity = abs(a.valid_from - c.valid_from) > self.eps or abs(chunk_a - chunk_c) > self.eps
        d.horizon = abs(chunk_a - chunk_c) > self.eps
        d.sequence = (self._last is not None
                      and (a.sequence_id - self._last[0]) != (c.sequence_id - self._last[1]))
        self._last = (a.sequence_id, c.sequence_id)
        _, a_clamp = self.safety.preview(state, (a.values[0],))   # preview 只判不发
        _, c_clamp = self.safety.preview(state, (c.values[0],))
        d.safety = a_clamp != c_clamp
        self.stats.ticks += 1
        self.stats.ledger.append((now, d))
        if any((d.value, d.validity, d.horizon, d.sequence, d.safety)):
            self.stats.divergences += 1
        if d.safety:
            self.stats.clamp_diffs += 1
        if d.validity:
            self.stats.validity_diffs += 1
        self.sink.submit(self.safety.check(state, (a.values[0],)))   # 只有 active 落地
        return d.value


def promote_allowed(stats: ShadowStats, policy: ReleasePolicy) -> bool:
    """晋升门槛是数据不是勇气：门槛本身住在 manifest 的 policy 段里。"""
    return stats.ticks >= policy.min_ticks and stats.divergence_rate <= policy.divergence_budget


# ---------------------------------------------------------------- 配置热加载：prepare → validate → commit

class ConfigRejected(Exception):
    pass


class ConfigManager:
    """阈值也是发布物：改一个阈值走一遍状态机，不许绕过发布通道直写现场参数。
    validate 不过 → current 一字不动 + 留事件；生效时点标注为下一个控制边界。"""

    def __init__(self, current, runtime: RuntimeVersion):
        self.current = current
        self.runtime = runtime
        self._staged = None
        self.events = []

    def prepare(self, candidate):
        if self._staged is not None:
            raise ConfigRejected("already_staged")      # 一次只允许一个在途变更
        self._staged = candidate

    def validate(self) -> Optional[str]:
        c = self._staged
        if c is None:
            return "nothing_staged"
        v = parse_version(c["config_version"])
        if v > parse_version(self.runtime.accepts_config[1]):
            return "config_too_new"
        if v < parse_version(self.runtime.accepts_config[0]):
            return "config_too_old"                     # 前向兼容同样要声明，不是默认成立
        if float(c["max_velocity"]) <= 0:               # 阈值必须是合法物理量
            return "unsafe_threshold"
        return None

    def commit(self, now: float) -> bool:
        reason = self.validate()
        staged, self._staged = self._staged, None
        if reason is not None:
            self.events.append(("rejected", staged["config_version"], reason, now))
            return False                                # 失败：current 保持不变，绝不半新半旧
        self.events.append(("committed", staged["config_version"], None, now))
        self.current = staged                           # 生效点：下一拍控制读到的是新值
        return True


# ---------------------------------------------------------------- 回滚：先失权，再切指针，再重验证

class RollbackRejected(Exception):
    pass


def rollback(supervisor: ReleaseSupervisor, runtime_core) -> bool:
    """回滚不是切指针。顺序即语义，一步都不能换：
    ① STOP：新命令停发（authority 关闭）
    ② invalidate：epoch+1——旧版本在途决策被结构性失权，不等"执行完"
    ③ SAFE_STOP：进安全态，机器人不悬在中间
    ④ 旧 manifest 必须重新过 boot_check——"旧版本"不等于"可回滚版本"
    ⑤ 切指针 + 完整 reset_episode（清隐藏状态/filter/tracking），回 VERIFYING 重走证据链
    epoch 的唯一发布者是 RuntimeCore：supervisor 与台账只消费镜像、不自行递增。"""
    supervisor.request_transition("ROLLING_BACK", "rollback")
    runtime_core.stop_new_commands()                    # ①
    runtime_core.bump_epoch()                           # ②
    supervisor.note_epoch(runtime_core.epoch)
    supervisor.request_transition("SAFE_STOP", "rollback_safe")   # ③
    reason = boot_check(runtime_core.previous_manifest, runtime_core.version)   # ④
    if reason is not None:
        raise RollbackRejected(reason)                  # 旧版本不兼容：停在 SAFE_STOP 求助，绝不硬切
    runtime_core.install_manifest(runtime_core.previous_manifest)  # ⑤ 切指针 + 完整 reset
    supervisor.note_epoch(runtime_core.epoch)
    supervisor.request_transition("VERIFYING", "rollback_complete")
    return True
```

然后是十五个 pytest，我不按"测了哪个函数"排，而按它钉住的**不变量**排——这样它和 9/17 的"契约 + invariant"风格对得上：

| # | 不变量 | 钉住它的测试 |
| --- | --- | --- |
| I1 | 不兼容 / 缺身份的 artifact 不得进入 motion path | `test_i1_boot_rejects_identity_and_drift` |
| I2 | 兼容性是包含判定，不是相交判定 | `test_i2_compatibility_is_containment_not_intersection` |
| I3 | 状态机不得跳过证据阶段 | `test_i3_state_machine_rejects_illegal_transition` |
| I4 | SAFE_STOP 后必须重新验证，不许"重启大法" | `test_i4_safe_stop_forces_reverification` |
| I5 | candidate 永远拿不到 command authority | `test_i5_shadow_never_lands_candidate` |
| I6 | 晋升必须满足 evidence budget | `test_i6_promotion_gate_requires_evidence` |
| I7 | epoch 屏障拒绝旧 epoch 决策（缓冲原语） | `test_i7_action_buffer_rejects_stale_epoch` |
| I8 | rollback 执行后旧在途决策永久失权 | `test_i8_rollback_executes_and_invalidates_inflight` |
| I9 | 限幅差异可观察、可计数 | `test_i9_clamp_divergence_detected` |
| I10 | 数值一致但时间语义分歧要被抓到 | `test_i10_temporal_divergence_without_value_divergence` |
| I11 | 故障按严重度分流，不一律急停 | `test_i11_fault_severity_routing` |
| I12 | 每条合法转移都留下完整证据 | `test_i12_events_form_evidence_chain` |
| I13 | 配置热加载 fail-before-motion | `test_i13_config_reload_is_fail_before_motion` |
| I14 | 旧版本 ≠ 可回滚版本：回滚要重过闸 | `test_i14_rollback_must_reverify_old_manifest` |
| I15 | 替换 policy 不改变 runtime seam | `test_i15_swap_policy_keeps_contract_shape` |

```python
# tests/test_deploy.py —— 部署与运维最小闭环：把 9/18 正文的每条不变量钉成一个可跑断言
# 与 9/17 同一套纪律：时钟注入、确定性 policy、epoch 屏障、单写者状态机。
# 范围：这些 pytest 验证的是"发布状态机的阶段约束 + 证据结构"，不模拟 fleet 级 canary 调度器。
import pytest

from deploy_fakes import (ArtifactIdentity, CompatibilityContract, ConfigManager,
                          ConfigRejected, Divergence, Fault, Manifest,
                          ReleaseEvent, ReleasePolicy, ReleaseSupervisor,
                          RollbackRejected, RuntimeVersion, ShadowRunner,
                          ShadowStats, boot_check, compat_within, parse_version,
                          promote_allowed, rollback)
from fakes import (CHUNK_LEN, DT, Action, ActionBuffer, ControllerSink, FakeClock,
                   FakeController, FakeSensor, NaiveEstimator, RuntimeCore,
                   SafetyLimiter, SequenceAllocator, SinePolicy, SlowPolicy,
                   StateBuffer, StateContract, Provenance, run_episode)


# ---------------------------------------------------------------- 构造小工具

def _manifest(release_id="r7", code_version="1.4.0", **over):
    art = over.pop("artifact", None) or ArtifactIdentity("sha256:aaa", "minisig:r7")
    compat = over.pop("compat", None) or CompatibilityContract(
        schema_range=("1.8", "2.0"), config_range=("3.1.0", "3.2.0"),
        obs_fingerprint="fp-v7")
    policy = over.pop("policy", None) or ReleasePolicy(min_ticks=100, divergence_budget=0.02)
    return Manifest(release_id=release_id, code_version=code_version,
                    artifact=art, compat=compat, policy=policy)


def _runtime(**over):
    base = dict(code_version="1.4.2", accepts_schema=("1.2", "3.0"),
                accepts_config=("3.0.0", "4.0.0"), expected_fingerprint="fp-v7")
    base.update(over)
    return RuntimeVersion(**base)


class OffsetPolicy(SinePolicy):
    """与 SinePolicy 差一个常量偏移的影子候选：用来制造确定性分歧。"""

    def __init__(self, offset):
        self.offset = offset

    def act(self, state, now):
        a = super().act(state, now)
        return Action(values=tuple(v + self.offset for v in a.values),
                      dt=a.dt, horizon=a.horizon, generated_at=a.generated_at,
                      valid_from=a.valid_from, valid_until=a.valid_until,
                      state_stamp=a.state_stamp, epoch=a.epoch,
                      sequence_id=a.sequence_id)


class PreviewSafety(SafetyLimiter):
    """给 9/17 的 SafetyLimiter 补一个 preview：只判定不发，让影子能观察限幅差异。
    check()（会落地）继承父类；deploy 侧唯一落地路径仍走 check→sink.submit。"""

    def preview(self, state, cmd):
        approved = self.check(state, cmd)
        return approved, approved != cmd


# 只为回滚接线准备的最小 runtime 替身：真实现里 RuntimeCore 本身就持有这些引用。
class RollbackCore:
    def __init__(self, active_manifest, previous_manifest, version, clock):
        self._epoch = 1
        self.active_manifest = active_manifest
        self.previous_manifest = previous_manifest
        self.version = version
        self.action_buffer = ActionBuffer(epoch=self._epoch)
        self.authority_open = True

    @property
    def epoch(self):
        return self._epoch

    def stop_new_commands(self):
        self.authority_open = False          # ① 停发新命令

    def bump_epoch(self):
        self._epoch += 1                     # ② 旧在途决策结构性失权
        self.action_buffer.sync_epoch(self._epoch)

    def install_manifest(self, manifest):
        self.active_manifest = manifest      # ⑤ 切指针
        self._epoch += 1                     # 完整 reset：epoch 再进一格，清隐藏状态
        self.action_buffer.sync_epoch(self._epoch)
        self.authority_open = True


# ================================================================ I1
def test_i1_boot_rejects_identity_and_drift():
    # 一切合规：放行
    assert boot_check(_manifest(), _runtime()) is None
    # 缺 release_id：事件流不可复盘，直接拒
    assert boot_check(_manifest(release_id=""), _runtime()) == "missing_release_id"
    # 观测语义指纹漂移：换 ckpt 最常见的静默 bug
    bad_fp = _manifest(compat=CompatibilityContract(
        schema_range=("1.8", "2.0"), config_range=("3.1.0", "3.2.0"),
        obs_fingerprint="fp-v6"))
    assert boot_check(bad_fp, _runtime()) == "fingerprint_mismatch"
    # schema 落在 runtime 接受区间之外
    bad_schema = _manifest(compat=CompatibilityContract(
        schema_range=("3.1", "4.0"), config_range=("3.1.0", "3.2.0"),
        obs_fingerprint="fp-v7"))
    assert boot_check(bad_schema, _runtime()) == "schema_incompatible"
    # config 太新
    bad_cfg = _manifest(compat=CompatibilityContract(
        schema_range=("1.8", "2.0"), config_range=("5.0.0", "6.0.0"),
        obs_fingerprint="fp-v7"))
    assert boot_check(bad_cfg, _runtime()) == "config_incompatible"


# ================================================================ I2（评审 #2：包含 ≠ 相交）
def test_i2_compatibility_is_containment_not_intersection():
    # manifest 声称支持 1.2~2.0，runtime 只吃 1.8~3.0：相交但半段不被覆盖，必须拒
    partial = _manifest(compat=CompatibilityContract(
        schema_range=("1.2", "2.0"), config_range=("3.1.0", "3.2.0"),
        obs_fingerprint="fp-v7"))
    narrow_runtime = _runtime(accepts_schema=("1.8", "3.0"))
    assert boot_check(partial, narrow_runtime) == "schema_incompatible"
    # 直接钉住判定函数：包含才真，相交不算
    assert compat_within(("1.8", "2.0"), ("1.2", "3.0")) is True
    assert compat_within(("1.2", "2.0"), ("1.8", "3.0")) is False
    assert parse_version("1.4.0") == (1, 4, 0)


# ================================================================ I3
def test_i3_state_machine_rejects_illegal_transition():
    sup = ReleaseSupervisor(FakeClock(), "r7")
    assert sup.request_transition("VERIFYING", "boot")
    assert sup.request_transition("SHADOW", "shadow_start")
    assert not sup.request_transition("ACTIVE", "skip")     # 跳级晋升：非法
    assert sup.state == "SHADOW"
    assert sup.request_transition("CANARY", "canary_start")
    assert sup.request_transition("ACTIVE", "promote")
    assert sup.request_transition("DEGRADED", "health_dip")
    assert sup.request_transition("ACTIVE", "recover")      # 恢复也走合法边


# ================================================================ I4
def test_i4_safe_stop_forces_reverification():
    sup = ReleaseSupervisor(FakeClock(), "r7")
    sup.request_transition("VERIFYING", "boot")
    sup.request_transition("SHADOW", "shadow_start")
    assert sup.report_fault(Fault.SAFETY, "policy_timeout") == "SAFE_STOP"
    assert not sup.request_transition("ACTIVE", "reboot_magic")  # "重启大法"被拒
    assert sup.request_transition("VERIFYING", "reverify")       # 唯一出路：回证据链起点


# ================================================================ I5（评审 #1：candidate 结构上进不了 sink）
def test_i5_shadow_never_lands_candidate():
    clock = FakeClock()
    active, candidate = SinePolicy(), OffsetPolicy(0.5)
    active.reset(1, SequenceAllocator())
    candidate.reset(1, SequenceAllocator())
    sink = ControllerSink(FakeController())
    runner = ShadowRunner(active, candidate, PreviewSafety(), sink)
    est, sensor = NaiveEstimator(), FakeSensor(clock)
    for _ in range(20):
        clock.advance(DT)
        now = clock.monotonic()
        state = est.estimate(sensor.latest(now), now)
        runner.tick(state, now)          # 落地只发生在 tick 内部：active→safety.check→sink
    assert runner.stats.ticks == 20
    assert runner.stats.divergences == 20                    # 常量偏移：每拍数值分歧
    assert len(sink._controller.sent) == 20                  # 落地的每一拍都来自 active 分支
    # candidate 的产物从未出现在 sink：它只进了台账。测试不替 runner 调 submit。


# ================================================================ I6
def test_i6_promotion_gate_requires_evidence():
    policy = ReleasePolicy(min_ticks=100, divergence_budget=0.02)
    few = ShadowStats(ticks=50)                              # 样本不足
    assert promote_allowed(few, policy) is False
    ok = ShadowStats(ticks=200, divergences=2)               # 1% < 2%
    assert promote_allowed(ok, policy) is True
    hot = ShadowStats(ticks=200, divergences=30)             # 15% > 2%
    assert promote_allowed(hot, policy) is False


# ================================================================ I7（评审 #13：拆开缓冲原语与真回滚）
def test_i7_action_buffer_rejects_stale_epoch():
    buf = ActionBuffer(epoch=2)
    stale = Action(values=(1.0,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                   generated_at=0.2, valid_from=0.2, valid_until=0.52,
                   state_stamp=0.2, epoch=1, sequence_id=99)
    assert buf.put(stale, 0.2) is False                      # epoch 1 的在途决策：屏障拒绝
    assert ("rejected_stale_epoch", 99) in buf.events


# ================================================================ I8（评审 #13：真正执行 rollback）
def test_i8_rollback_executes_and_invalidates_inflight():
    clock = FakeClock()
    core = RollbackCore(_manifest(release_id="r7"),
                        _manifest(release_id="r6", code_version="1.3.9"),
                        _runtime(), clock)
    sup = ReleaseSupervisor(clock, "r7")
    sup.request_transition("VERIFYING", "boot")
    sup.request_transition("CANARY", "canary_start")
    sup.note_epoch(core.epoch)
    sup.request_transition("ACTIVE", "promote")
    inflight = Action(values=(0.5,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                      generated_at=0.0, valid_from=0.0, valid_until=0.32,
                      state_stamp=0.0, epoch=core.epoch, sequence_id=1)
    assert core.action_buffer.put(inflight, 0.0) is True     # 回滚前：旧 epoch 决策可入队
    assert rollback(sup, core) is True
    assert sup.state == "VERIFYING"                          # 停在证据链起点，不直接回 ACTIVE
    assert core.active_manifest.release_id == "r6"
    assert core.epoch == 3                                   # 1 →(bump) 2 →(reset) 3
    stale = Action(values=(0.5,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                   generated_at=0.0, valid_from=0.0, valid_until=0.32,
                   state_stamp=0.0, epoch=2, sequence_id=5)
    assert core.action_buffer.put(stale, 0.0) is False       # 回滚期 epoch 的在途决策：已失权


# ================================================================ I9（评审 #9：clamp_diffs 真的在数）
def test_i9_clamp_divergence_detected():
    clock = FakeClock()
    active = OffsetPolicy(0.0)                               # ~0.02，不触发限幅
    candidate = OffsetPolicy(6.0)                            # 超 max_velocity*dt，被夹
    active.reset(1, SequenceAllocator())
    candidate.reset(1, SequenceAllocator())
    safety = PreviewSafety(max_velocity=5.0, dt=DT)
    sink = ControllerSink(FakeController())
    runner = ShadowRunner(active, candidate, safety, sink)
    est, sensor = NaiveEstimator(), FakeSensor(clock)
    for _ in range(5):
        clock.advance(DT)
        now = clock.monotonic()
        state = est.estimate(sensor.latest(now), now)
        runner.tick(state, now)
    assert runner.stats.clamp_diffs == 5                     # 每拍 active 放行、candidate 被夹
    assert all(d.safety for _, d in runner.stats.ledger)
    # candidate 看得到限幅裁决，但它的产物永不落地：sink 收到的仍是 active 的放行值
    assert all(cmd[0] <= 5.0 * DT + 1e-9 for cmd in sink._controller.sent)


# ================================================================ I10（评审 #8：时间语义分歧）
def test_i10_temporal_divergence_without_value_divergence():
    clock = FakeClock()
    active = SinePolicy()
    candidate = SlowPolicy(latency=0.15)                     # 数值相同，生效时刻晚 0.15s
    active.reset(1, SequenceAllocator())
    candidate.reset(1, SequenceAllocator())
    sink = ControllerSink(FakeController())
    runner = ShadowRunner(active, candidate, PreviewSafety(), sink)
    est, sensor = NaiveEstimator(), FakeSensor(clock)
    for _ in range(5):
        clock.advance(DT)
        now = clock.monotonic()
        state = est.estimate(sensor.latest(now), now)
        value_diverged = runner.tick(state, now)
    assert value_diverged is False                           # 数值维度：零分歧
    assert runner.stats.validity_diffs == 5                  # 时间语义维度：每拍都分歧
    assert runner.stats.divergences == 5                     # 若只比数值，这台状态机会"全绿"错过它
    assert all(d.validity and not d.value for _, d in runner.stats.ledger)


# ================================================================ I11（评审 #5：故障分级，不一律急停）
def test_i11_fault_severity_routing():
    clock = FakeClock()
    sup = ReleaseSupervisor(clock, "r7")
    sup.request_transition("VERIFYING", "boot")
    sup.request_transition("SHADOW", "shadow_start")
    sup.request_transition("CANARY", "canary_start")
    sup.request_transition("ACTIVE", "promote")
    # 可观测性故障：指标断流——冻结晋升、不打断正在安全执行的控制环，但留痕
    assert sup.report_fault(Fault.OBSERVABILITY, "metrics_gap") == "ACTIVE"
    assert sup.events[-1].to_state == "ACTIVE"
    assert sup.events[-1].reason == "metrics_gap"
    # 发布健康退化：分歧率抬头——进 DEGRADED，不是 SAFE_STOP
    assert sup.report_fault(Fault.RELEASE, "divergence_rise") == "DEGRADED"
    # 安全故障：碰物理边界——必须 SAFE_STOP
    assert sup.report_fault(Fault.SAFETY, "command_over_limit") == "SAFE_STOP"


# ================================================================ I12（评审 #4/#17/#18：证据链完整）
def test_i12_events_form_evidence_chain():
    clock = FakeClock()
    sup = ReleaseSupervisor(clock, "r7", epoch=4)
    sup.request_transition("VERIFYING", "boot", evidence_ref="ci:1234")
    clock.advance(1.0)
    sup.request_transition("SHADOW", "shadow_start", evidence_ref="shadow:88")
    assert sup.state == "SHADOW"
    assert len(sup.events) == 2
    e0, e1 = sup.events
    assert isinstance(e0, ReleaseEvent)
    assert e0.release_id == "r7" and e0.epoch == 4
    assert e0.to_state == "VERIFYING" and e0.evidence_ref == "ci:1234"
    assert e1.timestamp > e0.timestamp                       # 注入时钟：时间可复盘
    assert e1.reason == "shadow_start" and e1.from_state == "VERIFYING"
    # 非法转移不记账：证据链只记真实发生过的移动
    n_before = len(sup.events)
    assert not sup.request_transition("ACTIVE", "illegal")
    assert len(sup.events) == n_before


# ================================================================ I13（评审 #11：热加载 fail-before-motion）
def test_i13_config_reload_is_fail_before_motion():
    runtime = _runtime()
    cfg = ConfigManager({"config_version": "3.1.0", "max_velocity": 5.0}, runtime)
    cfg.prepare({"config_version": "3.1.5", "max_velocity": 4.0})
    assert cfg.validate() is None
    assert cfg.commit(now=1.0) is True
    assert cfg.current["config_version"] == "3.1.5"          # 合法：下一拍读到新值
    # 非法阈值：prepare→validate 失败→current 一字不动
    cfg.prepare({"config_version": "3.1.6", "max_velocity": -1.0})
    assert cfg.commit(now=2.0) is False
    assert cfg.current["max_velocity"] == 4.0                # 绝不半新半旧
    assert ("rejected", "3.1.6", "unsafe_threshold", 2.0) in cfg.events
    # 前向兼容同样要声明：过旧配置也拒（不是默认成立）
    cfg.prepare({"config_version": "2.0.0", "max_velocity": 3.0})
    assert cfg.validate() == "config_too_old"
    cfg.commit(now=3.0)
    # 一次只允许一个在途变更
    cfg2 = ConfigManager({"config_version": "3.1.0", "max_velocity": 5.0}, runtime)
    cfg2.prepare({"config_version": "3.1.1", "max_velocity": 5.0})
    with pytest.raises(ConfigRejected):
        cfg2.prepare({"config_version": "3.1.2", "max_velocity": 5.0})


# ================================================================ I14（评审 #20：旧版本 ≠ 可回滚版本）
def test_i14_rollback_must_reverify_old_manifest():
    clock = FakeClock()
    # 上一版 r6 的观测指纹与当前 runtime 不匹配：不能硬回滚
    incompatible_old = _manifest(release_id="r6", compat=CompatibilityContract(
        schema_range=("1.8", "2.0"), config_range=("3.1.0", "3.2.0"),
        obs_fingerprint="fp-v5"))
    core = RollbackCore(_manifest(release_id="r7"), incompatible_old, _runtime(), clock)
    sup = ReleaseSupervisor(clock, "r7")
    sup.request_transition("VERIFYING", "boot")
    sup.request_transition("CANARY", "canary_start")
    sup.request_transition("ACTIVE", "promote")
    with pytest.raises(RollbackRejected) as ei:
        rollback(sup, core)
    assert ei.value.args[0] == "fingerprint_mismatch"
    assert sup.state == "SAFE_STOP"                          # 停在安全态求助，绝不带病切回
    assert core.active_manifest.release_id == "r7"           # 指针没被污染


# ================================================================ I15（评审 #12：同 episode 直接比 A/B）
def test_i15_swap_policy_keeps_contract_shape():
    # 换一份数值同形、类不同的实现：接缝与 sent 流由契约钉住，回放逐位一致
    ctrl_a, safety_a = run_episode(FakeClock(), n_steps=20, policy=SinePolicy())
    ctrl_b, safety_b = run_episode(FakeClock(), n_steps=20, policy=OffsetPolicy(0.0))
    assert ctrl_a.sent == ctrl_b.sent                        # 同一 episode 下直接比，不另跑一次
    assert safety_a.safe_entries == safety_b.safe_entries
```

`python -m pytest tests/test_deploy.py -q` 直接通过（本文付印前实跑：15 passed）。这一版比初稿多做的事，恰恰是把评审会一眼看穿的"说了没做"补齐了：`clamp_diffs` 真的在数（I9）、candidate 的零落地由 runner 内部唯一路径保证而非测试替它调 `submit`（I5）、回滚测的是 `rollback()` 本身而不是缓冲原语（I7/I8 拆开）、config 热加载有了能跑的 `ConfigManager`（I13）、`release_id` 进了事件与状态机（I12）。这一篇的心脏是 I4、I8、I14 三块：I4 拒绝"重启大法"——SAFE_STOP 之后唯一出路是回到证据链起点重走一遍；I8 把回滚落成一次真正的失权，旧版本的在途决策被 epoch 屏障结构性拒绝，而不是靠"等一下让它执行完"；I14 补上最反直觉的一刀——回滚的目标版本也得先过兼容性闸，"旧"不等于"可回滚"。

## 部署层的六个反模式

延续 9/17 的清单，这里只列发布通道上的坑，还是按出现频率排序：

1. **同名不同物的 artifact**：靠文件名对版本，`v17_final_v2.pt` 式的命名是 skew 的头号温床。解法：哈希做内容身份、签名做来路身份（且签名要覆盖整段规范化 manifest），release_id 做事件身份，三者别混为一谈。
2. **回滚只切配置**：指针切了，旧版本的在途 chunk 和异步推理还在飞。解法：回滚 = 停命令 + epoch 失权 + 重过 boot_check + 切指针 + 完整 reset，五步顺序即语义。
3. **"看起来没问题就晋升"**：没有量化门槛的观察都会滑向侥幸。解法：`min_ticks` + `divergence_budget` 写进 manifest 的 policy 段，晋升闸门只认这两个数。
4. **回滚路径从没测过、回滚目标从没验过兼容**：事故当晚第一次跑回滚脚本，切回去才发现旧版本不兼容当前 runtime。解法：回滚演练进流水线，且回滚前对旧 manifest 重跑 boot_check——旧版本不等于可回滚版本。
5. **热改阈值走捷径**：绕过发布通道直接改现场参数。解法：配置版本进 manifest，热加载走 prepare → validate → commit，校验不过旧值一字不动，改一个阈值也算一次发布。
6. **兼容矩阵靠声明**："支持 v3 及以上"从没测过 v5，✓ 还是人手敲进 YAML 的。解法：没跑过 boot 校验的格子默认 ✗，每个 ✓ 携带证据引用、由流水线写；兼容性是测出来的属性。

## 总结

9/17 说骨架要"能换组件"，这一篇补的是换组件的那一层：**发布身份**（分层的 manifest：内容哈希、来路签名、事件序号，外加 release_id 与 epoch 的身份刻度）、**兼容性网格**（code × schema × config，包含而非相交，没测过的格子默认是 ✗ 且 ✓ 要带证据）、**shadow 对拍**（只记账、不落地，分歧拆成数值/时间/幅度/序号/安全五维，限幅差异真在计数）、**带门槛的灰度晋升**（数据说了算，且明确本文不模拟 fleet 调度器）、**epoch 屏障撑腰的回滚**（先停命令、再让旧决策失权、重过兼容性闸、才切指针，回滚路径本身要演练）、**fail-before-motion 的配置热加载**。十五个 invariant 把这台状态机钉在了能跑的程度上。

回头看这个系列的分工：评估协议（9/12）定义了什么算证据，架构篇（9/17）定义了证据从哪条接缝里长出来，这一篇定义了证据如何闸门下一次发布。三层拼起来，"换路线、换传感器、换机器人"才第一次成为一句有工程含义的话——拔下去有台账，插回来有屏障，每一步有证据。

下一步我更倾向第二个方向：**多机 fleet 的版本治理**。因为这一篇已经自然长出了 fleet registry 需要的基本原语——`release_id + compatibility matrix + evidence + epoch + rollback`。把它们提升成 registry / desired-state / reconciliation loop（N 台机器人 × code/schema/config 三轴，注册中心怎么记每个 ✓ 的证据、怎么收敛漂移），整个系列的架构线就闭合了。第一个方向（把 shadow 展开成完整发布流水线：对拍样本配比、边界工况、台账 schema）仍在候选里。想要哪个，评论区告诉我。
