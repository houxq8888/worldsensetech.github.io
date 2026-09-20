---
title: "具身智能的部署与运维：换组件不是发一次版本，是养一台状态机"
slug: "2026-09-22-agent-deployment-rollback"
date: 2026-09-22
draft: false
categories: ["具身智能", "教程"]
tags: ["具身智能", "软件架构", "机器人", "部署运维", "VLA", "Python", "系统设计", "工程架构", "Sim-to-Real"]
description: "9/17 给出了能跑、能测、能换组件的 Agent 骨架，这篇回答下一问：换下去的那一瞬间靠什么兜底。Web 服务回滚是撤销，机器人回滚是让旧决策失权——发布身份（分层 manifest + 签名语义）、runtime 的 schema×config×观测指纹兼容性网格（包含而非相交）、shadow 五维分歧台账与限幅差异、带晋升门槛的 canary、先失权再切指针的 epoch 屏障回滚、收在唯一命令入口的 authority/epoch/release 三道闸（发布身份必填、active 与异步晚到结果同门准入、rollback 与准入无竞态窗口）、fail-before-activation 的配置热加载，附一个纯 stdlib、十七个 invariant 的发布状态机最小闭环。"
toc: true
related_articles:
  - 2026-09-23-agent-release-state-machine-runnable
  - 2026-09-19-embodied-agent-architecture
  - 2026-09-16-policy-side-evaluation
  - 2026-09-12-sim-to-real-evaluation-protocol
  - 2026-09-15-policy-side-interface
  - 2026-09-09-robot-data-scaling
---

Web 服务的回滚是撤销——把流量指回旧版本，错误率掉下去就算完。机器人的回滚是让旧决策失权——把 manifest 指针切回去，那个版本还有一串在途的 chunk、排队中的异步推理、没清零的隐藏状态攥着控制权不放。这一篇就写这层落差。

上一篇把骨架立起来了：[六层运行时栈、三条核心契约、能跑能测的最小闭环](/zh/articles/2026-09-19-embodied-agent-architecture/)，结论是"换路线、换传感器、换机器人从外科手术降级为换插件"。但"可以插拔"和"敢拔敢插"之间还隔着一整层没写出来的东西——**拔下去的那一刻，谁兜底？**

换一块板子，焊错了可以返工；换一份 policy 权重，机器人正在端着一条玻璃杯。它和 9/17 是同一套写法——先把概念给到能讨论，代码落地与十七个 invariant 的实跑拆到了姊妹篇（[《把发布状态机跑起来》](/zh/articles/2026-09-23-agent-release-state-machine-runnable/)，纯 stdlib、17 passed）。这篇区别于普通 DevOps 的地方，一句话能说完：**它把 deployment 当作机器人控制安全边界的一部分**，所以整篇其实是一条链——

```text
release_id -> compatibility -> shadow evidence -> canary -> epoch barrier -> rollback
   身份           能不能装         装了动不动         给谁跑      旧决策失权       出事退得回去
```

身份对不齐，后面全没有可复盘的账；兼容性没实测，装上去就是带病运行；没有 shadow 台账，晋升靠的是胆子；没有 epoch 屏障，回滚就是新旧两个版本抢同一台机器人。下面一节一环。

一句贯穿全文的用词纪律，先把尺子立起来，因为这篇会反复跨这三层：我把话分成三种强度——**demo 实现保证的**（这套 stdlib 状态机的当前代码路径真跑出来的行为）、**架构要求的**（设计里必须成立、这一版未必落成）、**生产必须补的**（跨到 registry / 类型系统 / 并发才立得住）。姊妹篇里十七个 pytest 全绿，证明的是这个最小模型上局部 invariant 自洽，**不等于生产部署安全已经成立**——把"设计意图"读成"代码保证"，是这类文章最容易埋的坑，这篇尽量把它们分开写。

## 先立一个总原则：部署不是一个动作，是一台带证据的状态机

Web 后端的部署近似于"把新代码放上去，看错误率"。这个直觉搬到机器人上是危险的，危险有三层。第一，**回滚窗口不对等**：web 回滚是撤销，机器人回滚是从一个已经执行了错误动作的物理状态里恢复——杯子已经掉了。第二，**观测面不对等**：错误率和延迟曲线是集中式的，机器人的"对不对"分散在每台机器的每个 episode 里，等你看到现场事故，样本早已消费掉了。第三，也是本篇要反复回来的：**一次部署改变的不只是代码**——ckpt 换一版，归一化统计、ObsTransform、标定、控制阈值往往都在动，这些"看不见的版本"没有任何一行报错，只会让行为静静地漂移。

所以部署的正确建模不是动作，是**状态机 + 证据链**：每一次版本移动都要留下可复查的凭证（谁构建的、对着哪份 schema、跑过哪些校验、和旧版对拍过分歧没有），状态只允许沿合法边转移，非法转移被结构拒绝而不是靠流程自觉。眼熟吗？这正是 9/17 给运行时立的规矩——command authority 单写者、故障状态机单 owner、epoch 屏障——现在原样搬到了发布通道上。架构篇的接缝设计得越好，这一篇的运维才越便宜，这是两篇真正的咬合处。

## 发布身份：manifest 里每个字段都是防呆开关

9/17 给过一张 `deploy/artifact_manifest.yaml`：checkpoint、code_commit、schema 版本、归一化统计、标定文件、容器 digest。那时它的职责是"复现一次部署"；这一篇它要多干一份活——**充当发布状态机的输入**。字段本身不新鲜，新鲜的是给每个字段配的那道闸：

```yaml
# deploy/release_manifest.yaml —— artifact_manifest 的运维增强版
release:
  release_id: 2026.09-r7        # 发布事件标识：事件流对齐轴（单调性由 registry/CI 保证，非本机校验）
  checkpoint: ckpt/diffusion_v17.pt
  artifact_sha256: 9b41...      # 权重哈希：tag 会变，哈希不会
  code_commit: 8f3a2c1
  signature: minisig:r7.ok      # 来路签名：覆盖 canonical(manifest − signature)，不只是签 artifact
compat:
  state_schema: [v3, v4]        # 声明支持的 schema 区间：必须整体落在 runtime 接受范围内（包含，非相交）
  action_schema: [v2]
  obs_fingerprint: fp-v7        # 语义指纹 + golden vectors 的合成哈希（9/17）：evidence identifier，非 correctness proof
  config_range: [3.1.0, 3.2.0]  # 版本层面的兼容闸：min/max 都有立场，过旧也可能不兼容；字段级 schema/语义由 config validator 判
runtime:
  container_digest: sha256:71c0...
  python: "3.11"
policy:                          # 发布策略入 manifest：晋升门槛是数据，不是散在代码里的常量
  baseline_release: 2026.09-r6  # 与谁对拍
  divergence_budget: 0.02       # 晋升门槛：分歧率上限
  min_ticks: 10000              # 晋升门槛：最小样本量
```

这张清单在代码里不是一坨平铺字段，而是**分成三层、和 YAML 的三段一一对应**：`ArtifactIdentity`（哈希 + 签名）管"这是哪个件、谁造的"，`CompatibilityContract`（schema 区间 + config 区间 + 观测指纹）管"能不能和当前 runtime 一起跑"，`ReleasePolicy`（min_ticks + divergence_budget）管"要攒够多少证据才许晋升"。分层不是洁癖——它让 `boot_check`、`promote_allowed`、`rollback` 各自只读自己该读的那一段，晋升门槛从 manifest 的 policy 字段里取，而不是三个魔法常量散落在状态机里。

三个身份字段各自挡一类事故，而且要分清它们**不是一回事**。`artifact_sha256` 是**内容身份**，防"名字对了东西不对"：`diffusion_v17.pt` 是个名字不是身份，同名不同内容的 ckpt 是所有"见鬼了怎么行为不一样"案发现场的常客。`signature` 是**来路身份**（authenticity / provenance），防"内容没被换、但被谁签的不明不白"——关键在于签名覆盖的应当是 `canonical(manifest − signature)` 这一整段规范化字节，而不只是 artifact 的哈希；否则会出现"哈希没变、但有人偷偷改了 compat 里的 schema 区间"这种绕过。`release_id` 是**发布事件身份**，防事件流对不齐：运维事件（部署、对拍、灰度、回滚）跨 CI、注册中心、每台机器人三个时钟域，没有对齐序号就没有可复盘的账。`obs_fingerprint` 则防定义漂移：9/17 里它管 train/serve skew，在这里它管 release/serve skew——新 ckpt 配旧预处理，字段一个不缺、行为整个反了，靠逐元素比对数值当场就拦。评估协议（[9/12](/zh/articles/2026-09-12-sim-to-real-evaluation-protocol/)）里 traceability 的落地形态，落到部署层就是这一张清单加这几道闸。一句话把边界钉死：上面说的是**架构要求那道闸长什么样**；而姊妹篇那套 demo 状态机里，`artifact_sha256` 与 `signature` 只是被携带的 manifest 元数据——`boot_check` 既不会 `sha256(实际 artifact 字节)` 去复核哈希，也不会做 canonicalization + 验签。这两件事在生产版必须发生在 manifest 进状态机**之前**，边界在 registry / artifact-loader，不在这台状态机里。`release_id` 的"单调"同理：它是给事件流对齐用的**外部序号键**，单调性由 registry/CI 颁发方保证，本状态机连 `r7 > r6` 都不校验。

还有一组容易混的身份刻度，值得在这里一次性钉死，因为它们各自解决一种对齐问题：`release_id`（哪一次发布）、`robot_id`（哪台机器）、`episode_id`（哪一次作业）、`epoch`（这次作业里第几段被屏障划开的生命周期）、`sequence_id`（段内第几个决策）。特别要分清 **`release_id ≠ epoch`**：一次发布可以横跨很多个 episode，一个 episode 里也可能因为回滚连升好几个 epoch。事故复盘时它们是嵌套的坐标——`release r7 / episode 42 / epoch 8 / sequence 183`，一旦回滚就跳到 `release r6 / episode 43 / epoch 9 / sequence 0`。少了任何一层，你都没法在日志里精确指认"到底是哪一个决策把杯子碰掉的"。再往生产版推一格，事件流还该带上 `config_version` 和 `policy_instance_id`：同一份 `release_id` 可能配着不同阈值在跑（阈值也是发布物），而 shadow / 异步推理下、一次 release 里更可能同时驻留多个 candidate policy 进程——`release r7 / epoch 8` 分不清"是实例 A 还是实例 B 做的这个决策"。`release_id` 不足以唯一标识一个正在运行的 policy 实例，事故复盘要能指认到实例，这两个字段就得提前进 schema，而不是出事后再补。

再往前一步，把命令准入真正读的那三个刻度并列出来——它们常被当成"都是序号"，其实各管一件事，谁也别替谁：

| 字段 | 语义 | 在命令准入里承担什么 |
| --- | --- | --- |
| `release_id` | **谁产生的**（provenance） | 发布身份屏障：旧 release / 缺失 release 的决策失权 |
| `epoch` | **属于哪个生命周期**（authority generation） | 生命周期屏障：旧 epoch 的在途决策失权 |
| `sequence_id` | **这个生命周期里的第几个**（ordering） | 顺序约束，只在同一段内比先后 |

这解释了一个评审常问的点：**准入为什么用 `epoch + release` 判，而不用 `sequence` 判？** 因为 `sequence_id` 是 policy-local 的——shadow 对拍时比的是 `delta(sequence)` 而不是绝对值，回滚后它会从 0 重新起算。它只保证"同一段生命周期里第 183 个决策排在第 182 个之后"，**不承担跨 release 的身份认证**：一个 sequence 很大、epoch 又碰巧对上的旧 release 决策，光看序号是放它进来的漏网鱼（姊妹篇 I16 演的正是这条）。所以顺序归 sequence、来路归 release、生命周期权威归 epoch，三者不互相替代。顺带把 `epoch` 是什么钉死：它是**生命周期 / 权威的代际号（generation）**，不是版本号。schema 层的不变量是——`release_id` 变了**不必** `epoch + 1`（一次发布横跨好几个 episode，各自 epoch），`epoch + 1` 也**不必**换 release（同一次发布里回滚连升两格）。像 `r7 / epoch 8`、`r7 / epoch 9`、`r6 / epoch 10` 都是合法坐标；把 epoch 当版本指针，就会漏掉"同一 release 内多次生命周期重建"这一整类历史。

## 兼容性网格：不是向后兼容，是"哪些格子能一起跑"

版本管理最常见的错误假设是线性：新代码兼容旧数据，旧代码兼容新数据，一路向后。真正会各自移动、彼此不正交的是**运行时兼容三轴**——`schema`（状态/动作契约）、`config`（阈值/参数），以及 9/17 那套手法里的 `obs_fingerprint`（语义指纹 + golden vectors，防止预处理定义悄悄漂移）。至于 `code_version`：它当然也在动，但它更适合当**发布物身份**（`artifact_sha256 + code_commit`）由 registry 记着，而不是当一轴去做连续区间兼容判定——代码兼容性极少真的能用"某段版本区间内全兼容"表达，所以本文的 `boot_check` **携带** code_version、却不拿它比对（见姊妹篇代码）。下面这张格子图就是沿这三轴铺开的：

```text
        schema   v3      v4      v5
runtime  ──────────────────────────────  ← 一格 = 一次真实跑过的 boot 校验
  1.4.x    ✓      ✓      ✗      ✗
  1.5.x    ✗      ✓      ✓      ✗
config   ──────────────────────────────
  3.1      ✓      ✓      ✗      ✗
  3.2      ✗      ✓      ✓      ✗
           ↑ 没测过的格子默认是 ✗
           ↑（这张格子的"✓ + 证据引用"由 registry/流水线持有，本文最小状态机不落库）
```

三点纪律。**一，网格只给"真实存在过的组合"**。总想维持全兼容矩阵的团队，最后谁也没测全。机器人 fleet 的典型真相是：一台机器上 runtime 落后注册中心一个版本是常态——所以兼容区间必须显式建模（`compat` 的 `[v3, v4]`），而不是靠"应该没问题吧"。**二，没测过的格子默认是 ✗**。兼容性是测出来的属性，不是声明出来的属性；每个 ✓ 背后要有一次真跑过的 `boot_check` 加一轮 shadow 对拍。这句话还得追一句现实的坑：**那个 ✓ 到底记在谁头上？** 如果 ✓ 只是注册中心里一个人手敲进 YAML 的布尔值，它迟早和现实脱节。可信的做法是让 ✓ 携带证据引用（哪次 CI、哪轮 shadow、哪个测试套件跑过），由流水线写、不给人手改——这也正是下一篇 fleet 版本治理要接的接口。**三，配置也要做前向兼容**。灰度回滚时，旧版 runtime 往往要加载新版已经改过的配置（episode 42），字段改名或语义变化，旧代码就地崩——这是最容易在回滚当晚补第二个故障的路径。所以 config 变更遵守标准的 expand-migrate-contract：只加不删、新字段给默认值、删除排在所有机器都升级之后；manifest 里的 `config_range` 有 min 也有 max，就是这条纪律的强制闸——过旧和过界同样可疑。

一个建模上的坦白：`compat_within` 把版本当成**单调连续的语义区间**做包含判定（声明支持的整段必须都在 runtime 接受段内），这在 schema 确实是线性演进时成立。但真实世界里版本往往是**离散的 capability**——支持 `{v3, v4}` 不代表支持一个介于其间、字段却不同的伪版本。那种情况要把它换成 capability 集合的交集判断，而不是区间。本文为了最小闭环选了区间模型，并在代码注释里留了这个口子；读者若你的 schema 是离散 ID，请把"区间包含"替换成"集合包含"。再坦白一层：这里用的是**自定义的 numeric tuple 版本**（`int.int.int`），刻意不叫 SemVer——`1.9 < 1.10` 它能比，且解析时**接受 1~3 段、右侧补零归一到三段**，于是 `"2.0"` 与 `"2.0.0"` 相等、`"1"`/`"1.2"` 也各自补齐成 `1.0.0`/`1.2.0`，不会因为段数不同而被 Python 元组序判成 `"1.2" < "1.2.0"` 那种乱序。但它对格式是**严格**的：段数一旦超过三段（`"1.2.3.4"`）或任一段非纯数字（`"-1.2.0"`、`"1.2.x"`），直接 `ValueError` 拒绝——早先那版 `parts + [0]*(3-len(parts))` 会因为 `[0]*负数 == []` 把 `"1.2.3.4"` 悄悄放行（评审 §3 修的正是这个）。它仍不处理 `-rc1`/`+build7` 这类 pre-release/build 元数据。真要拿去判 SemVer，得处理这些元数据或干脆交给成熟的版本库；本文的区间模型只在"版本都是纯数字、且至多三段"这个前提被外部保证时才是可信的。注意我特意用的是**包含而非相交**：manifest 声明支持 `[1.2, 2.0]`、runtime 只吃 `[1.8, 3.0]`，两者相交但绝不能判兼容——因为 runtime 根本不认 manifest 承诺的 `1.2~1.8` 那半段，相交判定会放它过闸。

启动时的校验顺序也值得定死，从便宜到贵：**先结构**——manifest 字段齐不齐、有没有 `release_id`、schema 区间是否被 runtime 完整覆盖、config 版本是否越界，纯比对，毫秒级（注意这一层连 artifact 哈希都不重算、更不验签，那是进状态机前 loader 的活）；**再指纹**——比对 `obs_fingerprint` 是否一致。这一步要说准：它证明的只是"release 端和 runtime 端用的是同一份指纹定义 / 同一个证据标识"，是个 **evidence identifier，不是 correctness proof**——指纹相等并不代表 `ObsTransform(x)` 真算对了，只代表两边对"用什么哈希来表征预处理"这件事口径一致；**再数值**——这才是真正的正确性闸：拿 release manifest 里那组 golden vectors **真跑一遍**共用的 `ObsTransform`，逐条把实际输出和预期输出比对（golden vectors → actual transform → expected output → compare），对不上就拒。指纹门和 golden-vector 门是**两道不同的闸**，别把前者当后者；最后**行为**——进 shadow mode 对拍（下一节）。宁可拒绝启动，不可带病运行：**fail-before-activation** 说了一万遍，败就败在"先跑起来看看"。本文的 fake 能证到的只是"拒绝装载"这一步（boot_check 返回原因、不进后续），生产版要把同一条纪律一路推到"拒绝生效"——非法配置在控制边界上绝不接管。（"fail-before-motion" 这个词我早先写得偏重了，严格讲证据只到 activation 前，见姊妹篇 I13。）

## Shadow mode：决策对拍，永不落地

发布系统里最便宜的保险，是让新 policy **看得到一切、碰不到任何东西**。对同一份 `StateContract` 流，新旧两个 policy 各自产 `Action`，旧的照常进 SafetyGate → CommandSink 落地执行，新的进影子台账：

```text
StateContract 流 ──→ active policy ──→ ActionBuffer → SafetyGate → CommandSink → 真机
        └──────────→ candidate policy ─→ 分歧台账（只记账，sink 里没有它的路）
```

影子模式的实现纪律恰好是 9/17 两条老规矩的复用。其一，**authority 唯一**：candidate 的输出没有通向 RobotInterface 的路径——准确说，在 Python 这种动态类型下它不是"类型上进不去"，而是"结构上没有那条调用路径"：`ShadowRunner.tick()` 里 `candidate.act()` 的返回值只流向分歧台账，唯一调用 `sink.submit()` 的地方喂的是 `active` 经 `safety.check()` 的产物。类型级强制（把 `CandidateAction` 和 `Action` 分成不同类型、让 sink 只接受后者）是生产版该补的，这里不谎称已经做到。其二，**对拍要同状态同时钟**：两边喂的是同一拍 `latest_valid(now)` 的同一个状态对象，用同一个注入时钟；影子跑在旧状态上算出来的分歧全是噪声。

台账记的不该只有一个数。分歧是一维以上的事：**数值一致 ≠ 行为一致**。真正的危险常常不是 `0.50` 对 `0.52`（那可能只是噪声），而是同一个动作 `valid_from` 晚了 100ms——那是**另一个 chunk**，接管的物理时刻错了。所以 `Divergence` 拆成五个**彼此独立**的维度：`value`（数值超 ε）、`validity`（`valid_from` 生效时刻错位，管"什么时候开始接管"）、`horizon`（chunk 覆盖时长 `horizon×dt` 不一致，管"一次管多远"）、`sequence`（接管节奏，`sequence_id` 步进是否一致）、`safety`（限幅触发差异）。validity 与 horizon 是两回事，别把它们合成一轴——否则 horizon 一动就顺带把 validity 也点亮，维度就塌回四个了。`sequence` 这里还有个契约要先钉死：本文选的是 **policy-local 序号**——active 与 candidate 各持自己的 `SequenceAllocator`，绝对值天然对不齐，只能比步进 delta；若你的语义是全局共享序号，判定就该换成 `a.sequence_id != c.sequence_id` 的绝对比较。两种契约不能混用，正文和代码得选同一边。最容易被漏的是最后一维，也是最该盯的：**限幅触发差异**——同一条命令，SafetyGate 对老 policy 放行、对新 policy 夹了钳，哪怕任务成功率暂时看不出来，它就是新 policy 更激进的直接证据。实现上让 active 和 candidate 都过一次 `safety.preview()`（只判定、不落地），比较两者的 `clamped` 标志，`clamp_diffs` 就真在计数了；candidate 看得到安全门的裁决，却依旧碰不到 sink——这恰好同时演示了"能观察、不能执行"。这条不变量不能再靠运气守住：旧写法 `preview()` 直接复用 `check()`，"影子不产生副作用"只是因为 `check()` 恰好是个纯函数的**偶然性质**——真实 SafetyGate 常带状态突变、计数、watchdog、限流记账，那样的 `check()` 一旦被 preview 调用，candidate 就会偷偷改到门内的 active 状态，authority boundary 当场破功。修法是把纯判定收敛成唯一来源 `evaluate()`：`preview()` 只读 `evaluate()`、绝不碰提交态，`check()` = `evaluate()` 之后再 `commit`（写 `last_command`、累加 `clamp_count`）。这样"candidate 永远拿不到 authority"就从代码路径性质升级成接口性质——姊妹篇里的 `PreviewSafety` 就是这么拆的。分歧率是风险读数，不看它的灰度等于裸奔——但也要说清：`divergence_rate` 是**aggregate 计数**（五维任一命中记一次分歧），把"一次限幅"和"一次 1e-5 数值差"算成同权的一票，会丢风险权重；生产判据至少该把 safety/clamp 拆成硬门槛、把 numeric 留作软门槛，更讲究的按维度各自算率。

shadow 阶段也有两个失效条件要提前防。**候选不能碰状态**：影子 policy 不许往 estimator / StateBuffer 回写任何东西，否则"只观察"就变成了共同决策。**对拍要挑边界工况**：均匀时间轴上两个 policy 当然大部分时候一致——分歧台账要按 manifest 里声明的任务标签分层（接触、遮挡、标定漂移），对拍窗口盖不住边界工况，等于没拍。

## 灰度与晋升门槛：分歧台账说了算

对拍通过，才轮到真实执行。机器人场景的 canary 和 web 的百分比流量是两回事——**流量的天然切片不是用户，是任务、机器人和时段**：先影子，再一台机器的单一任务，再同型号 fleet，再跨型号；每一步都是"真实执行 + 故障预算"，区别只在爆炸半径。

晋升的规矩要定在前面：**门槛是数据不是勇气**。`release_manifest.yaml` 里那两个字段——`min_ticks` 和 `divergence_budget`——就是晋升闸门的门轴：样本量不够不许升（哪怕分歧率是零），分歧率超预算不许升（哪怕样本已经足够）。最忌讳的形态是"再观察观察"：没有量化门槛的观察，最后都会滑向周五下午的"看起来没问题，上吧"。回滚判据同理，而且要分硬度：安全门频繁介入、分歧率越过 hard threshold 是**立即回滚**，不讨论；覆盖率、成功率缓慢劣化是**预算内回滚**，按窗口计。两类判据都要在发布之前写成可计算的表达式，而不是事后开会找共识。还有一类容易被一锅炖的是"可观测性故障"——本文的 `Fault.OBSERVABILITY` 为了最小闭环只留了一格，但生产里至少该拆两半：**指标断流**（`OBSERVABILITY_DEGRADED`：眼睛暂时瞎了，控制环还在安全跑，冻结晋升即可）和**台账写失败**（`EVIDENCE_LOSS`：系统失去证明自己安全的能力，promotion evidence 断了，光"冻结晋升"未必够，可能得把 candidate 直接隔离）。两者严重程度不是一回事，把 ledger 写失败也归进"observability"是本文 fake 的一处简化，值得在真实 deployment policy 里钉死。

一句范围声明，免得读者误会：本文那套 pytest 验证的是**发布状态机的阶段约束与证据结构**——哪些状态之间能合法转移、晋升要不要攒够证据、回滚要不要重过闸。它**不模拟 fleet 级 canary 调度器**：机器选择、任务配比、观察窗口到点自动晋升，这些是 fleet registry 的对象，属于下一篇。这里出现的 `CANARY` 只是一个被状态机保护着的阶段名，不是一台调度器。同理别高估 `promote_allowed()`：它只实现两个最小门槛——`min_ticks`（样本量）加 `divergence_budget`（aggregate 分歧率）；正文提到的安全硬门槛（限幅/急停频次）、任务分层覆盖、成功率与 P95 劣化、fleet 级 policy evaluation，都不在这台 demo 里，是下一层调度器的活。这篇能替你保住的是"晋升要有可计算门槛"这条纪律，至于门槛该看哪些维度，代码只开了个头。

顺着这条"合法边 ≠ 放行"的线，把"**回到 ACTIVE**"这件事一次性拆干净。状态机里其实有三条"回来"的边，看着相似，吃的证据完全不是一回事——这也正是该把 **`ReleaseStateMachine`（判拓扑合法性）** 和 **`PromotionController`（判证据）** 分成两层的根因。demo 里 `request_transition()` 只看边、`promote_allowed()` 只看门槛，**二者当前没有接线**（前面状态机 docstring 标得很清楚：这是"保持最小模型"的 A 方案）；生产实现必须让 PromotionController 先把证据闸接上状态转移，调用方**绝不能绕过门槛直接 `request_transition("ACTIVE", "promote")`**。三条边的语义与严格程度递增：

| 来源态 | 回到 ACTIVE / 继续，需要什么 | 本文 demo 钉到哪一步 |
| --- | --- | --- |
| `CANARY` | **promotion evidence**：样本量够 + 分歧率在预算内 | `promote_allowed()` 两个最小门槛（I6）；但**未**接进状态转移 |
| `DEGRADED` | **recovery evidence**：health 回来、divergence 回到预算内、shadow 证据新鲜 | 只有 `DEGRADED → ACTIVE` 这条合法边；恢复条件**未落**，属架构要求 |
| `SAFE_STOP` | **full re-verification**：不许直接回 ACTIVE，必须回 `VERIFYING` 从证据起点重走 | `SAFE_STOP → VERIFYING` 唯一出边（I4）——最严的一条 |

一句话对齐三者：canary 靠**数据**转正、degraded 靠**恢复证据**回来、而 SAFE_STOP 之后压根没有"直接回 ACTIVE"这条路。把这三条写成分层的 guard（`promote_allowed` / `recover_allowed` / `SAFE_STOP→VERIFYING`），而不是一句 `if state_ok: go ACTIVE`，才配得上"顺序即语义"这四个字。

## 回滚：9/17 的 epoch 屏障在这里第二次上岗

回滚难，不在切配置。配置秒切，切完真正的麻烦浮出来：**在途的东西**——ActionBuffer 里旧版本产出的 chunk（scheduled 槽里还躺着一份）、policy 服务排队中的异步推理、正在执行的轨迹片段、以及 policy 的隐藏状态。只切 manifest 指针的回滚，等于是新旧两个版本共同驾驶同一台机器人——旧版本的最后一个决策还在缓冲里排队，新版本的第一个决策已经进场，两边的 sequence_id 还各算各的。

所以回滚不是一行赋值，是一台**顺序即语义**的小型事务，五步一步都不能换：① 停发新命令，关掉 command authority；② `epoch + 1`，让旧版本的在途决策**结构性失权**——不等"它们执行完"，也轮不到它们覆盖回滚后的版本；③ 进入 `SAFE_STOP`，下达安全停命令、机器人不悬在中间态（注意这只是**软件下了停令**，不等于"物理已经停稳"——requested / acknowledged / physically-confirmed 是三件事，见下文"三道边界"一节）；④ 旧 manifest **必须重新过一遍 `boot_check`**；⑤ 才切指针 + 走一次完整的 `reset_episode`——回 `VERIFYING` 重走证据链。关于第⑤步得说实在的：demo 里的 `reset_episode()` 只做两件事，置一个可断言的 `reset_called` 标志、再靠 `sync_epoch` 把 ActionBuffer 里的 active/scheduled 连同序号一并清掉（I8 正是拿这两条来验"完整 reset 不是空话"）；生产版还得在里面清 policy 隐藏状态、estimator 滤波器、controller tracking、随机种子——这些 fake 没实现，是明写的边界，别把"清了缓冲"读成"清了整条状态"。9/17 设计的 epoch 屏障正是为第②步准备的：epoch 加一再同步给 policy 与 ActionBuffer 之后，旧版本所有在途决策——不管它的 sequence_id 多大、离生效还有多远，**包括已经排进缓冲还没派发的那一份**——在 `put()` 的屏障前一律被拒、在 `current()` 派发时也取不到；序号空间重新起算，不存在跨版本的序号纠缠。"等旧 chunk 执行完再切"是不需要的，也是不允许的：屏障让旧版本失权，这不是快，这是干净。

第④步是全文最容易漏、也最要命的一条：**"旧版本"不等于"可回滚版本"**。你回滚到的那个 release，它的 schema、config、obs_fingerprint 未必还兼容**当前**的 runtime——尤其如果你已经先升了 runtime。回退前对旧 manifest 重新跑一次兼容性校验，过不了就停在 `SAFE_STOP` 求助，绝不硬切：硬切一个不兼容的旧版本，等于用一个新 bug 替换旧 bug。代码里这一步失败会抛 `RollbackRejected`，状态机稳稳停在 `SAFE_STOP`，指针一个字没动。

这里还牵出一条贯穿两篇的架构规矩：**epoch 的唯一发布者是 RuntimeCore**。supervisor、policy、ActionBuffer 只**接受** epoch、同步镜像，谁都不许自己 `epoch += 1`。否则就会出现 supervisor 看见 7、policy 还停在 7、buffer 停在 6 的半同步地狱——回滚恰恰是在这种不一致里最容易翻车的时刻。`rollback()` 里递增 epoch 的只有 `runtime_core`，supervisor 通过 `note_epoch()` 事后取镜像写进事件，正是这条 single-owner 纪律的落地。顺带解释一个容易被当成手滑的细节：一次回滚里 epoch 会 `+2`，不是重复自增。本文把 epoch 定义在**每个生命周期边界**上，而回滚恰好跨过两道边界——`epoch N → N+1` 是**撤销屏障**（第②步 bump，让旧世界失权），`epoch N+1 → N+2` 是**新世界起始屏障**（第⑤步 install 时再 bump，序号从 0 重新起算）。生产的事件日志最好把这两次 bump 标成不同 reason（例如 `rollback_invalidate` / `rollback_install`），否则复盘时会误以为有人多推了一格。

为了审计，我还给状态机补了一个 `ROLLING_BACK` 态：光是 `ACTIVE → SAFE_STOP` 分不清"真故障""人工 E-stop""回滚""watchdog""controller timeout"。有了显式回滚态（半途失败还能落回 `SAFE_STOP`），再配上每条转移都带 `reason` 的事件，事后才问得出"这台机器昨晚到底经历了什么"。而"回滚路径本身要测"这条反直觉的纪律——没执行过回滚的发布计划，等价于没有回滚——在姊妹篇里就是一行 pytest。这里也得给自己正个名：正文我几处顺口说了"证据链"，但**姊妹篇里的 `supervisor.events` 严格讲只是一份 append-only 的事件日志（event log）**，每条记录携带的是**证据引用**（哪次 CI、哪轮 shadow、哪个 `reason`），而不是证据本身。真正的 tamper-evident 证据链是四段——`事件日志 → 证据引用 → 外部证据库 → 防篡改审计（哈希链 / 签名 / WORM 存储）`——后三段都在 registry 侧，不在这台 stdlib 状态机里，demo 既不签名也不哈希、任何人都能改内存里的 `events`。所以读到"证据链"三个字，请按"事件日志 + 证据引用"理解，把 tamper-evident 留给加固版——这也是全文那条三层用词纪律的一次自查。

### 回滚之后：三道常被漏掉的边界

上面那五步把"回滚"讲圆了，但真要拿去兜底，还有三道边界容易被"代码没写这条路"当成"结构上不可能"——而 demo 恰恰最该把后者显式立起来。

**第一道：命令入口（command admission）——所有安全不变量的那扇门。** 回滚第①步说"停发新命令"、第②步说"旧 epoch 决策失权"，可如果 authority 闸、epoch 闸、release 闸、序号闸散落在 `put()` 前后的几处判断里，将来任何一条新路径（异步回调、candidate 转正、调试后门）绕过某一句，屏障就漏了。所以要把它们收进**唯一入口**：决策不再直接 `buffer.put()`，而是先过一道 admission，全绿才允许进缓冲。这里有个容易被忽略的取舍：发布身份必须是**准入的必填输入**，而不是可选元数据——如果门里写的是 `release = getattr(action, "release_id", 当前版本)`，那么一条**没带 `release_id`** 的旧命令会被悄悄归成"当前 release"、混过身份闸。所以这一版把身份抽成一个显式的 `ActionContext(release_id, epoch, sequence_id)` 随决策一起进门，缺了 release 直接拒，不再兜底。而且走的正是"**同步 active、异步晚到、candidate 转正三类决策同门准入**"——active 也不再抄近路直接 `sink.submit`，它和迟到结果过同一道闸。

```text
   Policy 同步产出的 active    ┐
   异步推理的晚到结果          ├──► Command Admission    ← 唯一入口，三类决策同门准入
   candidate 转正后首次上跑    ┘        │   准入输入 = ActionContext(release_id, epoch, sequence_id)
                                       │   发布身份必填：缺 release 直接拒，不再 getattr 兜底成"当前版本"
                                       ▼
                              authority_open ?    回滚①关闸：停发新命令
                              release   match ?   发布身份屏障：旧 / 缺失 release 失权
                              epoch     match ?   回滚②生命周期屏障：旧 epoch 失权
                              ctx ↔ action 自洽 ? 序号 / 时窗交给 ActionBuffer
                                       │  prepare() 判闸 → commit() 复核 generation 才 put()
                                       │  两步之间不假装"检查即提交"原子，靠 generation 屏障关 TOCTOU
                                       ▼
                              ActionBuffer
                                       ▼
                              SafetyGate (evaluate 纯判定 / check = evaluate + commit)
                                       ▼
                              CommandSink → 真机

   ── candidate 在 shadow 里只走 preview()：既不进 admission，也不进 sink
      （转正前拿不到 authority 是接口性质，不是"恰好没那条调用路径"）
```

姊妹篇里这道门落成一个 `RollbackCore` 与 `ShadowRunner` **共用**的 `CommandAdmission`：准入输入是显式的 `ActionContext(release_id, epoch, sequence_id)`，缺 release 直接拒、不再兜底；`authority_open → release → epoch → ctx↔action 自洽` 四步判完，`prepare()` 只判闸、`commit()` 复核 generation 后才 `action_buffer.put()`。关键点是**连 active 的落地也从这道门进**——shadow 里 active 决策现在走 `admission.admit()` 过闸才 `sink.submit`，不再是"`safety.check → sink`"抄近路。于是"candidate 进不了 sink"不再依赖"恰好没有那条调用路径"，"active 之外没有旁门"也不再是文档承诺——都变成进门先被闸掉的接口性质。

**第二道：异步晚到——epoch 挡得住，还得 release 来补。** policy 服务是异步的：一个推理请求可以在 r7 / epoch 1 时发出，回滚全跑完、机器人已经换到 r6 / epoch 3 了，它的结果才姗姗返回。光靠 epoch 屏障够不够？大部分够——旧 epoch 对不上新 epoch，直接拒。但有一种漏网鱼：晚到结果的 epoch **碰巧**等于当前 epoch（队列错位、或回滚又 bump 回了同一个数）。这时只有**发布身份屏障**能拦住它——决策身上除了 `epoch`，还得带一个 `release_id`，进门时和当前权威 release 比对。**epoch 是生命周期屏障，release 是发布身份屏障，两者职责不重叠、缺一不可。** 姊妹篇里 I16 演的就是这个：晚到结果随决策带进门的 `ActionContext` 里，一条 epoch 对上、但 `release_id` 还是 r7 的决策，被发布身份闸稳稳拒掉；再补一条 `release_id` 为空（缺身份、指望 getattr 兜底成"当前版本"）的决策——同样被拒，因为这一版根本没有那条兜底。这也是前面"身份刻度"里说 `release_id` 该提前进 action schema 的兑现处。

**第三道：物理边界——软件失权 ≠ 世界已经停下。** epoch 屏障管的是**软件侧**的三种在途：queued（还没进缓冲的）、buffered（排进缓冲未派发的）、late-produced（异步晚到的）——这三类它能一律作废。但它管不到**已经派发进物理执行器**的那一条：命令一旦被 MCU 接受、关节已经在动了，这不是软件能"撤销"的历史。要真正停下，得靠控制器层的**取消 / 制动 / hold / 轨迹 abort**，那是 9/17"安全是独立一层、硬件兜底"的延伸。同理，第③步的 `SAFE_STOP` 是软件**下了停令**，要分清 `requested / acknowledged / physically-confirmed` 三个时刻——`supervisor.state == SAFE_STOP` 只代表命令通道关了，不代表玻璃杯已经稳稳放回桌面。**这条边界不是 epoch 的 bug，是物理现实**：也正因为它，"机器人回滚 ≠ web 回滚"才成立——web 撤销的是流量，机器人失权之后还要等世界真的停下来。这三道边界在姊妹篇里都只做到"能讲清、能各钉一个断言"（I8 钉 authority + 派发失权，I16 钉异步晚到，I17 钉 admission 与 rollback 之间没有竞态窗口——用 `prepare()` → `rollback()` → `commit()` 的确定性交错，证明 barrier 之前判过闸、之后才想提交的旧动作被 generation 屏障挡回），真多线程 / 跨进程的并发加固、物理停稳的确认闭环、release 屏障跨进程传递，都是明写的生产债。

## 配置与阈值热加载：改阈值不是改代码，但比改代码更危险

部署要换的是 artifact，运维要换的常常只是阈值：速度上限、validity 窗口、控制频率。热加载配置诱人，因为它绕过了整个发布通道；它危险，因为**阈值就是安全参数**。三条规矩：配置版本进 manifest，改一个阈值也算一次发布（走同一台状态机，没有捷径）；热加载走 **prepare → validate → commit**，校验失败保持旧值并记事件，绝不允许半新半旧的状态被控制环读到；所有阈值标注生效时点（下一拍 / 下一个 episode），控制中段的参数切换要像 ActionBuffer 的接管一样只发生在边界上。"小改动"三个字在事故复盘里的出现频率，高得离谱。这一条在姊妹篇里也用 `ConfigManager` 落成了能跑的断言：非法阈值 commit 之后，`current` 必须一字不动。但把话说回来——`commit()` 里那句 `self.current = staged`，在 demo 里只是**单线程下的一次对象引用替换**，它演示的是"校验不过就不动 current"的**提交纪律**，不是并发原子性。真正要兑现"控制环读不到半新半旧"，生产得靠**不可变快照 + 原子指针 swap / generation 号 / 控制边界 latch**（或直接复用文中的 epoch 屏障）来保证读写不会各读到一半——那是架构要求，这套 stdlib fake 没证明、也不打算假装证明。

## 补到能跑：一台发布状态机的最小闭环（挪到了姊妹篇）

上面都是设计。真正把它落成能跑、能测的最小形态——完整 `deploy_fakes.py`、逐条钉住 I1–I17 的 `test_deploy.py`、以及一次实跑（17 passed）——篇幅足够独立成篇，我把它拆到了姊妹篇 [《把发布状态机跑起来：一台纯 stdlib 最小闭环与十七个 invariant》](/zh/articles/2026-09-23-agent-release-state-machine-runnable/)。那篇复用 [9/19 的 fakes.py](/zh/articles/2026-09-19-embodied-agent-architecture/)（`ActionBuffer`、epoch 屏障、时钟注入、单写者 CommandSink 全部原样在场），本篇只留设计与边界，代码与断言都在那篇里。想看"每条不变量到底被哪个断言钉住、`CommandAdmission` 的 `prepare/commit` 怎么在代码里落闸门"，直接跳到那篇；想先弄清"为什么要有这道闸、回滚为什么 ≠ web 回滚"，留在这篇读完再走也不迟。

## 部署层的六个反模式

延续 9/17 的清单，这里只列发布通道上的坑，还是按出现频率排序：

1. **同名不同物的 artifact**：靠文件名对版本，`v17_final_v2.pt` 式的命名是 skew 的头号温床。解法：哈希做内容身份、签名做来路身份（且签名要覆盖整段规范化 manifest），release_id 做事件身份，三者别混为一谈。补一句边界：本文 demo 只是把 `artifact_sha256`/`signature` 当字段**带**进状态机，真正重算哈希、canonicalize + 验签发生在生产版的 registry / artifact-loader，不进这台状态机——别把"字段在场"读成"验签完成"。
2. **回滚只切配置**：指针切了，旧版本的在途 chunk 和异步推理还在飞。解法：回滚 = 停命令 + epoch 失权 + 重过 boot_check + 切指针 + 完整 reset，五步顺序即语义。
3. **"看起来没问题就晋升"**：没有量化门槛的观察都会滑向侥幸。解法：`min_ticks` + `divergence_budget` 写进 manifest 的 policy 段，晋升闸门只认这两个数。同样标个边界：demo 的 `promote_allowed` 就吃这两个数、且分歧率是 aggregate 一维读数；按维度拆率、限幅/急停的硬门槛、任务分层覆盖这些更强的判据是生产 / 下一层调度器的活，本文只把"晋升必须有可计算门槛"这条纪律立住。
4. **回滚路径从没测过、回滚目标从没验过兼容**：事故当晚第一次跑回滚脚本，切回去才发现旧版本不兼容当前 runtime。解法：回滚演练进流水线，且回滚前对旧 manifest 重跑 boot_check——旧版本不等于可回滚版本。
5. **热改阈值走捷径**：绕过发布通道直接改现场参数。解法：配置版本进 manifest，热加载走 prepare → validate → commit，校验不过旧值一字不动，改一个阈值也算一次发布。
6. **兼容矩阵靠声明**："支持 v3 及以上"从没测过 v5，✓ 还是人手敲进 YAML 的。解法：没跑过 boot 校验的格子默认 ✗，每个 ✓ 携带证据引用、由流水线写；兼容性是测出来的属性。

## 总结

9/17 说骨架要"能换组件"，这一篇补的是换组件的那一层：**发布身份**（分层的 manifest：内容哈希、来路签名、事件序号，外加 release_id 与 epoch 的身份刻度——哈希与签名在本 demo 里只是被携带的元数据，验签/重算是生产 loader 的边界）、**兼容性网格**（runtime 的 schema × config × 观测指纹，包含而非相交，code 只作发布物身份不作兼容轴，没测过的格子默认是 ✗ 且 ✓ 要带 registry 里的证据；指纹门与 golden-vector 门是两道不同的闸）、**shadow 对拍**（只记账、不落地，分歧拆成数值/时间/幅度/序号/安全五个独立维度，限幅差异真在计数，且 preview 收敛到纯 `evaluate`、副作用只留在 check 的 commit 分支）、**带门槛的灰度晋升**（数据说了算，且明确本文不模拟 fleet 调度器、promote 只吃两门槛、合法边 ≠ 满足业务条件；CANARY / DEGRADED / SAFE_STOP 三条"回来"的边各吃不同证据——promotion / recovery / full re-verification）、**epoch 屏障撑腰的回滚**（所有决策——同步 active、异步晚到、candidate 转正——统一走同一道 `CommandAdmission`，`ActionContext` 里的 `release_id` **必填**、不再 getattr 兜底；authority / release / epoch 三道闸收在一处；先停命令、再让旧决策含排队未派发者与异步晚到的旧 release / 缺身份结果失权、重过兼容性闸、才切指针，回滚路径本身要演练；且 `prepare()` → `rollback()` → `commit()` 之间由 generation 复核挡掉 TOCTOU 窗口，真并发的线性化仍是生产债）、**fail-before-activation 的配置热加载**（提交原子、非法候选生效前被拒；不声称并发原子）。十七个 invariant 把这台状态机钉在了"能跑、局部自洽"的程度上——但请记住全文那条三层用词纪律：demo 实现保证的 ≠ 架构要求的 ≠ 生产必须补的，绿灯只证明了这个最小模型自洽，不等于生产部署安全已经成立。

回头看这个系列的分工：评估协议（9/12）定义了什么算证据，架构篇（9/17）定义了证据从哪条接缝里长出来，这一篇定义了证据如何闸门下一次发布。三层拼起来，"换路线、换传感器、换机器人"才第一次成为一句有工程含义的话——拔下去有台账，插回来有屏障，每一步有证据。

下一步我更倾向第二个方向：**多机 fleet 的版本治理**。因为这一篇已经自然长出了 fleet registry 需要的基本原语——`release_id + compatibility matrix + evidence + epoch + rollback`。把它们提升成 registry / desired-state / reconciliation loop（N 台机器人 × schema/config/观测指纹三轴、code 作发布物身份挂边，注册中心怎么记每个 ✓ 的证据、怎么收敛漂移），整个系列的架构线就闭合了。第一个方向（把 shadow 展开成完整发布流水线：对拍样本配比、边界工况、台账 schema）仍在候选里。想要哪个，评论区告诉我。
