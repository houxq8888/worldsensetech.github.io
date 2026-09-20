---
title: "把发布状态机跑起来：一台纯 stdlib 最小闭环与十七个 invariant"
slug: "2026-09-25-agent-release-state-machine-runnable"
date: 2026-09-25
draft: false
categories: ["具身智能", "教程"]
tags: ["具身智能", "软件架构", "机器人", "部署运维", "VLA", "Python", "系统设计", "工程架构", "测试"]
description: "9/22 概念篇把部署与回滚讲成一台带证据的状态机，但只给设计、没摊开代码。这一篇是那半句“文末再用一套纯 stdlib 假实现把它跑起来”的兑现：完整贴出 deploy_fakes.py、逐条钉住 I1–I17 的 test_deploy.py，并实跑（17 passed）。前置：复用 9/19 的 fakes.py（ActionBuffer、epoch 屏障、时钟注入、单写者 CommandSink），概念与边界见 9/22 概念篇。"
toc: true
related_articles:
  - 2026-09-22-agent-deployment-rollback
  - 2026-09-19-embodied-agent-architecture
  - 2026-09-16-policy-side-evaluation
  - 2026-09-12-sim-to-real-evaluation-protocol
---

[9/22 那篇](/zh/articles/2026-09-22-agent-deployment-rollback/)把部署与回滚讲成一台带证据的状态机——发布身份、兼容性网格、shadow 对拍、带门槛的灰度晋升、先失权再切指针的 epoch 屏障回滚、收在唯一命令入口的 authority/release/epoch 三道闸，它只给了设计，把那半句"文末用一套纯 stdlib 的假实现真正跑起来"留给了这一篇。这里就是兑现那半句的地方：完整贴出 `deploy_fakes.py`，把每条不变量钉成一个可跑断言的 `test_deploy.py`，以及一次实跑（17 passed）。

先把前置摆清楚，免得读者对着跑不起来：这一篇**复用 [9/19 那套 fakes.py](/zh/articles/2026-09-19-embodied-agent-architecture/)**——`from fakes import ActionBuffer`、以及测试里的 `Action / ControllerSink / FakeClock` 等，都来自那篇文末的同一套假件；时钟注入、确定性 policy、epoch 屏障、单写者 CommandSink 全部原样在场，本篇一行都不改它们。至于**为什么**要这么设计——三道常被漏掉的边界、六个部署层反模式、以及贯穿全文的三层用词纪律——都在 [9/22 概念篇](/zh/articles/2026-09-22-agent-deployment-rollback/)，那篇现在只留设计、不再摊开代码。本篇只做一件事：把状态机摆到能跑，再一条一条钉给你看。

一句用词纪律先立起来：下面这十七个 pytest 全绿，证明的只是"这套 stdlib 最小模型里定义的不变量可以重复执行、局部自洽"，**不等于生产部署安全已经成立**——哪些是 demo 真跑出来的、哪些只是架构要求、哪些还得生产补，逐条标在下面。

## 这台状态机的最小闭环长什么样

把概念篇里那条链落成能跑能测的最小形态——仍然不用 torch、不用 GPU，复用 9/19 那套 fakes（时钟注入、确定性 policy、epoch 屏障、CommandSink 单写者全部原样在场），只加几样东西：分层的 `Manifest` / `RuntimeVersion`（发布身份与兼容区间）、带证据链的 `ReleaseSupervisor`（唯一写者，每条转移落成一行 `ReleaseEvent`）、按性质分流的 `Fault`、同状态五维对拍且限幅可观察的 `ShadowRunner`、`boot_check` / `promote_allowed` / `ConfigManager` / `rollback`（从装到退的四道闸）。先看骨架代码：

```python
# tests/deploy_fakes.py —— 部署与运维的最小闭环：纯 stdlib，pytest 直接跑
# 与 9/17 的 fakes.py 同一套纪律：时钟注入、确定性 policy、epoch 屏障、单写者状态机。
# 范围声明：本文件只验证发布状态机的阶段约束与证据结构，不模拟 fleet 级 canary 调度器
# （机器选择、任务配比、观察窗口自动晋升都是 fleet registry 的对象，见文末）。
from dataclasses import dataclass, field
from typing import Optional, Tuple

from fakes import ActionBuffer

_SHADOW_RELEASE = "shadow-active"        # 纯 shadow 对拍时 active 落地所挂的固定 release 标


def parse_version(s: str) -> Tuple[int, int, int]:
    """MAJOR.MINOR.PATCH 的最小版本解析：支持 1~3 段数字，右侧补零归一到三段，于是
    "2.0" 与 "2.0.0" 相等、(1,2) 不再被 Python 元组序判成小于 (1,2,0)。段数 > 3 或
    任一段非纯数字直接 ValueError——于是 "1.2.3.4" 与 "-1.2.0" 这类越界输入不再被静默接受
    （评审 §3：文档说三段，代码就别再吞四段/负号）。刻意不叫 semver——不处理 -rc1 /
    +build7 预发布元数据。叫对名字，别向规范碰瓷。"""
    parts = s.split(".")
    if not (1 <= len(parts) <= 3):
        raise ValueError("version must have 1..3 numeric segments")
    if not all(p.isdigit() for p in parts):
        raise ValueError("version segments must be numeric")
    nums = [int(p) for p in parts] + [0, 0, 0]
    return (nums[0], nums[1], nums[2])


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
    """内容身份 + 来路身份在 manifest 里的**字段承载**。范围：本 fake 只携带这两个字段，
    不重算 artifact 字节、也不做密码学验签——canonicalization + signature verification
    与 sha256(actual_artifact)==artifact_hash 的复核，都是生产版 registry / artifact-loader
    边界的活，manifest 进状态机之前必须完成（见正文）。"""
    artifact_hash: str           # 内容身份：同名不同物由它拦（本 fake 不重算，只携带）
    signature: str               # 来路身份：签名应覆盖 canonical(manifest − signature)，防"哈希没变、
                                 # compat 被改"（本 fake 不验签，只建模字段及其应有语义）


@dataclass(frozen=True)
class CompatibilityContract:
    """兼容声明的三段。两条边界要先钉清，免得被读大：
    ① config_range 是**版本号层面的** compatibility gate——它只判"版本落在区间内"，
       字段级的 schema / 语义兼容（改字段名、换单位、换默认值）由 config validator 负责，
       是另一层，别把"版本在范围里"读成"配置真兼容"（评审 §7）。
    ② obs_fingerprint 是 **evidence identifier，不是 correctness proof**——相等只证明
       "预处理定义一致"，不证明"实现正确"：两个都写错的 transform 照样能生成同一个
       fp-v7。实现一致那半靠 golden vectors 的数值比对兜底（9/17），不靠这根字符串相等（评审 §8）。"""
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
    release_id: str                  # 发布事件身份：事件流的对齐轴；单调性由 registry/CI 颁发方
                                     # 保证，本状态机不校验 r7>r6（见正文）
    code_version: str
    artifact: ArtifactIdentity
    compat: CompatibilityContract
    policy: ReleasePolicy


@dataclass(frozen=True)
class RuntimeVersion:
    """机器人上这一版 runtime 能接受什么，说清楚。
    注意 code_version 在这里只是身份载体：本 fake 的 boot_check 不拿 code_version 做兼容判定，
    实际的兼容三轴是 accepts_schema / accepts_config / expected_fingerprint。代码身份由
    artifact_hash + code_version 承担（见正文 code 是不是独立轴那段）。"""
    code_version: str
    accepts_schema: Tuple[str, str]
    accepts_config: Tuple[str, str]
    expected_fingerprint: str


def boot_check(manifest: Manifest, runtime: RuntimeVersion) -> Optional[str]:
    """启动与回滚共用的一道**结构闸**：只查 manifest 里的 release_id 在不在、
    schema/config 区间是否被 runtime 完整覆盖、obs_fingerprint 是否相等；失败返回原因，
    通过返回 None。宁可拒绝，不可带病运行。
    范围：不重算 artifact 字节、不验签——artifact digest / signature verification
    在生产版是 registry / artifact-loader 的前置动作（见正文与 ArtifactIdentity）。
    对 release_id 也**只验 presence**：syntax / issuer / uniqueness / monotonicity
    （r7>r6 之类）都属于 registry admission，不在这台本机状态机里判（评审 §13）。"""
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
    OBSERVABILITY = "observability"    # 指标断流 → 冻结晋升，不动当前执行态
    # 范围：本 fake 把 OBSERVABILITY 归成一格。生产里还应从 OBSERVABILITY 里拆出 EVIDENCE_LOSS：
    # 台账写失败意味着系统暂时失去"证明自己安全"的能力，其处置可能不止冻结晋升、还要隔离候选。
    # 具体走 OBSERVABILITY_DEGRADED 还是 EVIDENCE_LOSS 由 deployment policy 明确定义（见正文）。


@dataclass(frozen=True)
class ReleaseEvent:
    """证据链的最小单元：每条转移都带身份、时间、原因和 epoch——
    审计问"哪个版本在什么时候以什么理由进了哪个状态"，这一行就是答案。
    范围（评审 §14）：本 demo 里 `supervisor.events` 只是一个普通 list——这是 **event log**，
    还不是 **tamper-evident evidence chain**。生产版的 registry / audit log 要再加
    event_id、previous_event_hash 串成哈希链、append-only 持久存储与签名者身份，
    记录才谈得上不可抵赖；这几样都不在这台状态机里。"""
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
    每条合法转移落成一行 ReleaseEvent；非法转移拒绝且不记账。

    一条必须钉住的边界（评审 §9/§10，本文取"保持最小模型"的 A 方案）：这台状态机判的是
    **转移图的合法性**——`CANARY → ACTIVE`、`DEGRADED → ACTIVE` 是**合法的边**，但合法边 ≠
    满足走这条边的**业务条件**。真正的晋升判据（`promote_allowed`：样本量 + 分歧预算）与恢复判据
    （health 回来了、divergence 回到预算内、shadow 证据新鲜）是**上层 promotion / recovery controller
    的 guard**，本 demo **刻意不把它们接进状态机**：`request_transition("ACTIVE", "promote")` 只看边、
    不看 guard。别把"状态机放行了"读成"证据够了"。"""

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
    validity: bool = False                 # valid_from 生效时刻错位（时间对齐轴）
    horizon: bool = False                  # chunk 时长（horizon×dt）不一致（覆盖窗口轴）
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
        # 范围：aggregate——五维任一命中即记一次分歧，把 value/validity/horizon/sequence/safety
        # 压成同一读数，会丢风险权重（一次限幅差异 ≠ 一次 1e-5 数值差）。生产的晋升判据
        # 应至少拆出 safety/clamp 硬门槛与 numeric 软门槛，或干脆按维度各自算率（见正文）。
        return self.divergences / self.ticks if self.ticks else 0.0


@dataclass(frozen=True)
class ActionContext:
    """决策的**身份上下文**——admission 的必要输入，不是可选元数据（评审 §1）。
    一条 action 想拿到物理执行资格，必须先能证明三件事，各占一个字段、互不重叠：
    release_id = 谁产生的（provenance），epoch = 属于哪个生命周期（lifecycle authority），
    sequence_id = 这个生命周期里的第几个（ordering）。正文那张"身份刻度"表在这里落到执行前契约上。"""
    release_id: str
    epoch: int
    sequence_id: int


class CommandAdmission:
    """唯一的命令入口（正文那张图里的那道门）：authority / release / epoch 三道闸收在一处，
    全过之后再交给 ActionBuffer 做序号 / 时窗接纳。任何决策——policy 同步产出、异步推理晚到、
    active 在 shadow 里落地——都只能走 admit()，拿到执行资格才可能进 sink。
    关键：release 是**必要输入**（由 ctx 显式携带），没有 `getattr(..., 当前值)` 的 fallback
    （评审 §1）——不带身份、或身份对不上的决策一律拒。"""

    def __init__(self, action_buffer, active_release_id, epoch):
        self.action_buffer = action_buffer
        self.active_release_id = active_release_id
        self.epoch = epoch
        self.authority_open = True

    def _gates_ok(self, ctx, action) -> bool:
        """纯判定：authority → release → epoch，并要求 ctx 与它要放行的 action 自洽
        （防"贴了新标签的旧动作"）。无副作用，可被 prepare/commit 分别调用。"""
        if not self.authority_open:                                   # ① 停发新命令（回滚第一步）
            return False
        if ctx.release_id != self.active_release_id:                  # ② 发布身份屏障（provenance）
            return False
        if ctx.epoch != self.epoch:                                   # ③ 生命周期屏障（lifecycle authority）
            return False
        if ctx.epoch != action.epoch or ctx.sequence_id != action.sequence_id:
            return False                                              # ④ 身份上下文必须描述这条 action
        return True

    def prepare(self, ctx, action) -> bool:
        """只判闸、不入队（无副作用）。I17 用它复现 check 与 commit 之间的 TOCTOU 窗口。"""
        return self._gates_ok(ctx, action)

    def commit(self, ctx, action, now) -> bool:
        """generation-checked enqueue：在真正入队这一刻**重新过一遍闸**（评审 §12/§13）。
        多线程里 prepare 与 commit 之间可能被回滚插队——重判保证不存在 rollback/admission 的线性化窗口。"""
        if not self._gates_ok(ctx, action):
            return False
        return self.action_buffer.put(action, now)                    # 过闸后仍受 buffer 的序号 / 时窗约束

    def admit(self, ctx, action, now) -> bool:
        """同步产出的决策走这里：prepare 通过后立即 commit（单线程下二者等价）。"""
        if not self.prepare(ctx, action):
            return False
        return self.commit(ctx, action, now)


class ShadowRunner:
    """同一份状态喂两个 policy，candidate 的输出永不进 sink——影子模式的判定：
    只记账，不落地。而落地的那一条**也必须先过统一命令入口**（评审 §2 方案 A）：active 的
    决策同样先 action.admit(ctx, a, now)，过闸拿到执行资格后才 safety.check → sink.submit，
    与回滚共享同一个 CommandAdmission。这样正文"所有决策只能从这道门进"对 active 也成立，
    不再是"candidate 的安全边界比 active 的命令边界更完整"。
    candidate 侧依旧只走 preview，不进 admission、不进 sink。
    范围声明：candidate 进不了 sink 仍是**当前实现路径**性质（没有任何一条代码路径拿 candidate
    的返回值去调 admit / sink.submit），不是类型系统或 capability token 层的强制。Python 动态类型不
    拦得住未来某次重构里的一次误接线。生产版该补的是：CandidateAction 与 Action 拆成不同类型、
    sink.submit 签名只收后者；或引入 authority token，token 不匹配时 sink 拒绝。见正文三层词表。"""

    def __init__(self, active, candidate, safety, sink, divergence_eps=1e-6, admission=None):
        self.active = active
        self.candidate = candidate
        self.safety = safety
        self.sink = sink
        self.eps = divergence_eps
        self.stats = ShadowStats()
        self._last = None                # (active_seq, candidate_seq)：接管节奏比对基线
        # 默认给一条只跑 active、authority 常开、release 固定为影子标的的入口；回滚测试会注入自己的。
        self.admission = admission if admission is not None else \
            CommandAdmission(ActionBuffer(epoch=1), _SHADOW_RELEASE, 1)

    def tick(self, state, now) -> bool:
        a = self.active.act(state, now)
        c = self.candidate.act(state, now)
        d = Divergence()
        d.value = max(abs(x - y) for x, y in zip(a.values, c.values)) > self.eps
        chunk_a, chunk_c = a.horizon * a.dt, c.horizon * c.dt
        # 两轴解耦：validity = 生效时刻（时间对齐），horizon = chunk 时长（覆盖窗口）。
        # 合成一轴会让 horizon=True 必然推出 validity=True，维度就不再独立了（见正文）。
        d.validity = abs(a.valid_from - c.valid_from) > self.eps
        d.horizon = abs(chunk_a - chunk_c) > self.eps
        # sequence 契约（本文选定）：sequence_id 是 policy-local——active/candidate 各有独立
        # SequenceAllocator，绝对值天然对不齐，只能比步进 delta。若你的语义是"全局共享序号"，
        # 这里应换成 a.sequence_id != c.sequence_id 的绝对判定——两种契约不可混用（见正文）。
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
        # active 落地也先过统一命令入口：ctx 由这条 action 的身份构成，admit 全绿才 clamp+submit
        ctx = ActionContext(release_id=_SHADOW_RELEASE, epoch=a.epoch, sequence_id=a.sequence_id)
        if self.admission.admit(ctx, a, now):
            self.sink.submit(self.safety.check(state, (a.values[0],)))   # 只有过闸的 active 落地
        return d.value


def promote_allowed(stats: ShadowStats, policy: ReleasePolicy) -> bool:
    """晋升门槛是数据不是勇气：门槛本身住在 manifest 的 policy 段里。
    范围：这里只实现两个最小门槛——样本量 min_ticks + aggregate 分歧率 divergence_budget。
    安全硬门槛（限幅/急停频次）、任务分层覆盖、成功率与 P95 劣化、fleet 级 policy evaluation
    都属于下一层调度器，不在本最小状态机内（见正文范围声明）。"""
    return stats.ticks >= policy.min_ticks and stats.divergence_rate <= policy.divergence_budget


# ---------------------------------------------------------------- 配置热加载：prepare → validate → commit

class ConfigRejected(Exception):
    pass


class ConfigManager:
    """阈值也是发布物：改一个阈值走一遍状态机，不许绕过发布通道直写现场参数。
    validate 不过 → current 一字不动 + 留事件；生效时点标注为下一个控制边界。
    范围：commit() 里的 self.current = staged 是**单线程假实现下的对象引用替换**。要真正保证
    "控制环读不到半新半旧"，生产需 immutable ConfigSnapshot + 原子指针 swap / generation 号 /
    控制边界 latch（或复用文中的 epoch 屏障），而不是靠一次赋值——本 fake 不证明并发原子性（见正文）。"""

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
        """编排三层校验，返回第一个失败原因（评审 §9：别让读者把"版本没越界"读成"配置兼容"）。
        真正的字段/schema 与语义层在这里只是**留了位子的 fake**，生产版才填真逻辑。"""
        c = self._staged
        if c is None:
            return "nothing_staged"
        return self.validate_version(c) or self.validate_schema(c) or self.validate_semantics(c)

    def validate_version(self, c) -> Optional[str]:
        """① 版本层面的兼容闸：config_version 是否整体落在 runtime 的 accepts_config 区间内。"""
        v = parse_version(c["config_version"])
        if v > parse_version(self.runtime.accepts_config[1]):
            return "config_too_new"
        if v < parse_version(self.runtime.accepts_config[0]):
            return "config_too_old"                     # 前向兼容同样要声明，不是默认成立
        return None

    def validate_schema(self, c) -> Optional[str]:
        """② 字段 / schema 层：改字段名、加/删字段、换 key——版本闸管不到这层（fake，生产补真逻辑）。"""
        return None

    def validate_semantics(self, c) -> Optional[str]:
        """③ 语义 / 物理量层：阈值是不是合法物理量、单位与默认值对不对——同样独立于版本（fake 只做最小检查）。"""
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
    ⑤ 切指针 + 完整 reset_episode（本 fake 只置一个 reset 标志；生产在 reset 里清
       policy 隐藏状态 / estimator 滤波器 / tracking / 种子——见正文），回 VERIFYING 重走证据链
    epoch 的唯一发布者是 RuntimeCore：supervisor 与台账只消费镜像、不自行递增。"""
    # epoch 语义（本文定义：每个生命周期边界一格）——一次 rollback 经过两个边界，因此 +2：
    #   epoch N     active execution          （正在跑、要被撤销的旧世界）
    #   epoch N+1   rollback invalidation barrier（② bump_epoch：旧在途决策结构性失权）
    #   epoch N+2   installed / reset lifecycle  （⑤ install_manifest 再 bump：新世界开跑，
    #                                            序号重新起算）
    # 这不是重复操作：第一格是撤销屏障，第二格是重启屏障（见正文）。
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

然后是十七个 pytest，我不按"测了哪个函数"排，而按它钉住的**不变量**排——这样它和 9/17 的"契约 + invariant"风格对得上：

| # | 不变量 | 钉住它的测试 |
| --- | --- | --- |
| I1 | 缺发布身份 / schema·config 越界 / 观测指纹漂移 → 拒绝装载（结构校验，不重算 artifact、不验签） | `test_i1_boot_rejects_missing_identity_and_observation_drift` |
| I2 | 兼容性是包含判定，不是相交判定 | `test_i2_compatibility_is_containment_not_intersection` |
| I3 | 状态机不得跳过证据阶段 | `test_i3_state_machine_rejects_illegal_transition` |
| I4 | SAFE_STOP 后必须重新验证，不许"重启大法" | `test_i4_safe_stop_forces_reverification` |
| I5 | candidate 产物不进 sink（当前实现路径性质，非类型强制） | `test_i5_shadow_never_lands_candidate` |
| I6 | 晋升必须满足 evidence budget | `test_i6_promotion_gate_requires_evidence` |
| I7 | epoch 屏障拒绝旧 epoch 决策（缓冲原语） | `test_i7_action_buffer_rejects_stale_epoch` |
| I8 | 回滚执行后：旧 epoch 新决策被拒 + **已排队未派发**的在途 chunk 也派发不出来（不止 `current()` 取不到，缓冲内部 active/scheduled 双槽同时清空——把证据从"没吐出来"往"无可派发"推一格）+ 完整 reset 留痕 + authority/release 随事务翻转 | `test_i8_rollback_executes_and_invalidates_queued_inflight` |
| I9 | 限幅差异可观察、可计数 | `test_i9_clamp_divergence_detected` |
| I10 | 数值一致但时间语义分歧要被抓到 | `test_i10_temporal_divergence_without_value_divergence` |
| I11 | 故障按严重度分流，不一律急停 | `test_i11_fault_severity_routing` |
| I12 | 每条合法转移都留下完整证据 | `test_i12_events_form_evidence_chain` |
| I13 | 配置热加载：提交原子、非法候选在生效前被拒（不声称并发原子/无 motion） | `test_i13_config_reload_is_atomic_and_rejects_invalid` |
| I14 | 旧版本 ≠ 可回滚版本：回滚要重过闸 | `test_i14_rollback_must_reverify_old_manifest` |
| I15 | 数值等价的两份 policy 在同一 episode 下逐位同输出（不等于换 policy 保 runtime seam——那要断言接口结构） | `test_i15_equivalent_policy_swap_preserves_episode_output` |
| I16 | 统一命令入口（`ActionContext` 必填身份）：回滚后晚到的旧 release 异步结果被拒——epoch 对上也没用，release 身份屏障兜底；**缺失 / 空 `release_id` 的决策同样被拒**（这一版没有 getattr 兜底那条路） | `test_i16_async_late_result_denied_after_rollback` |
| I17 | admission 与 rollback 无竞态窗口（确定性交错）：`prepare()` 放行一条旧动作 → `rollback()` 撤销 → `commit()` 被 generation 复核挡回，缓冲双槽仍空（demo 证确定性交错，真多线程仍是生产债） | `test_i17_admission_and_rollback_have_no_race_window` |

```python
# tests/test_deploy.py —— 部署与运维最小闭环：把 9/18 正文的每条不变量钉成一个可跑断言
# 与 9/17 同一套纪律：时钟注入、确定性 policy、epoch 屏障、单写者状态机。
# 范围：这些 pytest 验证的是"发布状态机的阶段约束 + 证据结构"，不模拟 fleet 级 canary 调度器。
import pytest

from deploy_fakes import (ActionContext, ArtifactIdentity, CommandAdmission,
                         CompatibilityContract, ConfigManager,
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
    """给 9/17 的 SafetyLimiter 补一个**结构上无副作用**的 preview：把纯判定从提交动作里拆出来。
    评审 P0：旧写法 preview() 直接复用 check()，"影子不产生副作用"只是**恰好**成立——它依赖
    check() 目前是纯函数这个偶然性质。真实 SafetyGate 常带状态突变 / 计数 / watchdog / 限流记账，
    一旦 check() 有了副作用，candidate 的 preview 就会偷偷改到 active 的门内状态。
    修法：把纯判定收敛成唯一来源 evaluate()，preview 只读 evaluate()、永不碰提交态；
    check（active/生产落地路径）= evaluate + commit。这样"candidate 拿不到 authority"从
    代码路径性质升级成接口性质。"""

    def __init__(self, max_velocity=5.0, dt=DT):
        super().__init__(max_velocity, dt)
        self.last_command = None        # 提交态：只有 check/commit 会写
        self.clamp_count = 0            # 提交态：限幅计数，preview 绝不触碰

    def evaluate(self, state, cmd):
        """唯一真源，纯判定：返回 (approved_cmd, clamped)，不写任何 self.* 状态。"""
        pos, = cmd
        measured, = state.proprio
        max_delta = self.max_velocity * self._dt
        delta = max(-max_delta, min(max_delta, pos - measured))
        approved = (measured + delta,)
        return approved, approved != cmd

    def preview(self, state, cmd):
        """影子 / candidate 路径：只读纯 evaluate()，不 commit——所以门内状态纹丝不动。"""
        return self.evaluate(state, cmd)

    def check(self, state, cmd):
        """active / 生产落地路径：evaluate 之后才 commit 副作用（记账 + 限幅计数）。"""
        approved, clamped = self.evaluate(state, cmd)
        self.last_command = approved
        if clamped:
            self.clamp_count += 1
        return approved


# 只为回滚接线准备的最小 runtime 替身：真实现里 RuntimeCore 本身就持有这些引用。
class RollbackCore:
    def __init__(self, active_manifest, previous_manifest, version, clock):
        self._epoch = 1
        self.active_manifest = active_manifest
        self.previous_manifest = previous_manifest
        self.version = version
        # 统一命令入口：authority / release / epoch 三道闸 + ActionBuffer 收进 CommandAdmission，
        # 与 ShadowRunner 共用同一个类——不再是回滚专用的一份私有实现（评审 §2）。
        self.admission = CommandAdmission(ActionBuffer(epoch=self._epoch),
                                          active_manifest.release_id, self._epoch)
        self.reset_called = False        # 观测钩子：证明 reset_episode 真的跑过

    @property
    def epoch(self):
        return self._epoch

    @property
    def action_buffer(self):
        return self.admission.action_buffer

    @property
    def authority_open(self):
        return self.admission.authority_open

    @property
    def active_release_id(self):
        return self.admission.active_release_id

    def admit(self, ctx, action, now) -> bool:
        """直接委托给共享的 CommandAdmission：release 由 ctx 显式携带、**无 fallback**（评审 §1）。"""
        return self.admission.admit(ctx, action, now)

    def stop_new_commands(self):
        self.admission.authority_open = False     # ① 停发新命令

    def bump_epoch(self):
        self._epoch += 1                          # ② 旧在途决策结构性失权
        self.admission.epoch = self._epoch
        self.admission.action_buffer.sync_epoch(self._epoch)

    def reset_episode(self):
        # 生产在这里清 policy 隐藏状态 / estimator 滤波器 / tracking / 种子；
        # 本 fake 只需留一个可断言的痕，好让 I8 能证明"完整 reset"不是空话（见正文）。
        self.reset_called = True

    def install_manifest(self, manifest):
        self.active_manifest = manifest           # ⑤ 切指针
        self.admission.active_release_id = manifest.release_id   # 权威 release 随之切换
        self._epoch += 1                          # 新世界生命周期：epoch 再进一格
        self.admission.epoch = self._epoch
        self.admission.action_buffer.sync_epoch(self._epoch)    # 清 active/scheduled + 序号重算
        self.reset_episode()                      # 完整 reset：本 fake 置标志，生产清上述状态
        self.admission.authority_open = True


# ================================================================ I1
def test_i1_boot_rejects_missing_identity_and_observation_drift():
    # 范围：boot_check 只拦"缺身份 + 观测漂移 + schema/config 越界"三类结构问题；
    # artifact 字节重算与验签不在此闸，属 registry / artifact-loader 边界（见 ArtifactIdentity 说明）。
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


# ================================================================ I8（评审 #13：真正执行 rollback + 队到未派发也失权）
def test_i8_rollback_executes_and_invalidates_queued_inflight():
    clock = FakeClock()
    core = RollbackCore(_manifest(release_id="r7"),
                        _manifest(release_id="r6", code_version="1.3.9"),
                        _runtime(), clock)
    sup = ReleaseSupervisor(clock, "r7")
    sup.request_transition("VERIFYING", "boot")
    sup.request_transition("CANARY", "canary_start")
    sup.note_epoch(core.epoch)
    sup.request_transition("ACTIVE", "promote")
    # 回滚第一步（STOP）是被测不变量，不只是散文描述：authority 初始开、指向 r7。
    assert core.authority_open is True
    assert core.active_release_id == "r7"
    # 回滚前：旧 epoch 的决策可入队；故意排成"已入队、尚未派发"（future valid_from），
    # 停在 scheduled 槽——这才是回滚 safety 真正要钉住的那类在途决策。
    inflight = Action(values=(0.5,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                      generated_at=0.0, valid_from=0.3, valid_until=0.62,
                      state_stamp=0.0, epoch=1, sequence_id=1)
    assert core.action_buffer.put(inflight, 0.0) is True     # 0.0 < valid_from → 进 scheduled
    assert rollback(sup, core) is True
    assert sup.state == "VERIFYING"                          # 停在证据链起点，不直接回 ACTIVE
    assert core.active_manifest.release_id == "r6"
    assert core.active_release_id == "r6"                    # STOP→...→install：权威 release 切到 r6
    assert core.authority_open is True                       # authority 关闭是回滚的**中途态**，
    #  完整五步跑完后新世界重新开门（停发只在事务内生效，见 rollback 的 ①/⑤ 两步）
    assert core.epoch == 3                                   # 1 →(bump 屏障) 2 →(reset) 3
    assert core.reset_called                                 # "完整 reset" 真的发生了，不是空话
    # 关键：回滚前就排进缓冲、尚未派发的那个 chunk，到点后也绝不能再被派发执行
    assert core.action_buffer.current(0.5) is None
    # §6 加强：dispatch 侧真正读的是缓冲内部态——两个槽都必须空，才叫"下游 sink 永不会发出它"，
    # 而不仅仅"这一刻没吐出来"。生产里 buffer 与 MCU 之间还隔着 dispatcher / controller queue，
    # 这条断言把失权证明落到"没有东西可供派发"这一层（见正文物理边界）。
    assert core.action_buffer._active is None and core.action_buffer._scheduled is None
    # 新插入的旧 epoch 决策同样被屏障拒绝（结构失权的另一面）
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


# ================================================================ I13（评审 #14/#15：证据止于"原子提交 + 拒绝非法候选"）
def test_i13_config_reload_is_atomic_and_rejects_invalid():
    # 名字从 fail-before-motion 降级为 atomic-and-rejects-invalid：本 fake 没有 motion、
    # 也没有控制环并发读 config，能证明的是"非法候选提交失败、current 一字不动"（见正文）。
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


# ================================================================ I15（评审 #18：名字别大于内容）
def test_i15_equivalent_policy_swap_preserves_episode_output():
    # 范围：这测的是"数值等价的两份实现，在同一 episode 下 sent/safe_entries 逐位相同"。
    # 它没有、也不能证明"换 policy 不改 runtime seam/contract shape"——那要断言的是
    # PolicyProtocol / Action / StateContract / reset() / act() 这套接口结构本身（见正文）。
    ctrl_a, safety_a = run_episode(FakeClock(), n_steps=20, policy=SinePolicy())
    ctrl_b, safety_b = run_episode(FakeClock(), n_steps=20, policy=OffsetPolicy(0.0))
    assert ctrl_a.sent == ctrl_b.sent                        # 同一 episode 下直接比，不另跑一次
    assert safety_a.safe_entries == safety_b.safe_entries


# ================================================================ I16（评审 §1/§16/§18：异步晚到结果不得重回命令通道，且 release 是必要输入）
def test_i16_async_late_result_denied_after_rollback():
    clock = FakeClock()
    core = RollbackCore(_manifest(release_id="r7"),
                        _manifest(release_id="r6", code_version="1.3.9"),
                        _runtime(), clock)
    sup = ReleaseSupervisor(clock, "r7")
    sup.request_transition("VERIFYING", "boot")
    sup.request_transition("CANARY", "canary_start")
    sup.request_transition("ACTIVE", "promote")
    # 回滚前：一条当前 release / epoch 的同步决策带身份正常过闸，并把序号基线推到 10。
    ok = Action(values=(0.3,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                generated_at=0.0, valid_from=0.0, valid_until=0.32,
                state_stamp=0.0, epoch=1, sequence_id=10)
    assert core.admit(ActionContext("r7", 1, 10), ok, 0.0) is True
    assert rollback(sup, core) is True
    # 晚到结果 A：旧世界 epoch=1 → 生命周期屏障先拒（release 与 epoch 都还是旧的）。
    late_a = Action(values=(0.7,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                    generated_at=0.0, valid_from=0.0, valid_until=0.32,
                    state_stamp=0.0, epoch=1, sequence_id=11)
    assert core.admit(ActionContext("r7", 1, 11), late_a, 0.0) is False
    # 晚到结果 B：epoch 碰巧等于新世界的 3，但 ctx 仍带旧 release r7 →
    # 只有"发布身份屏障"能拦住它。这正是 epoch 单独承重时的漏网鱼（评审 §5）。
    late_b = Action(values=(0.9,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                    generated_at=0.0, valid_from=0.0, valid_until=0.32,
                    state_stamp=0.0, epoch=core.epoch, sequence_id=99)
    assert core.admit(ActionContext("r7", core.epoch, 99), late_b, 0.0) is False
    # §1 反例：一条**不带 release 身份**的决策（release_id 缺失/空），即使 epoch 恰好等于当前
    # 世界，也必须在门口被拒——release 是必要输入，没有"缺省成当前 release"的 fallback。
    no_release = Action(values=(0.8,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                        generated_at=0.0, valid_from=0.0, valid_until=0.32,
                        state_stamp=0.0, epoch=core.epoch, sequence_id=100)
    assert core.admit(ActionContext("", core.epoch, 100), no_release, 0.0) is False
    # 三道屏障都没放行 → 命令通道里始终是空的：sink 侧不会收到任何旧 release / 无身份的派发。
    assert core.action_buffer.current(0.1) is None


# ================================================================ I17（评审 §12/§13：admission 与 rollback 不存在线性化窗口）
def test_i17_admission_and_rollback_have_no_race_window():
    clock = FakeClock()
    core = RollbackCore(_manifest(release_id="r7"),
                        _manifest(release_id="r6", code_version="1.3.9"),
                        _runtime(), clock)
    sup = ReleaseSupervisor(clock, "r7")
    sup.request_transition("VERIFYING", "boot")
    sup.request_transition("CANARY", "canary_start")
    sup.request_transition("ACTIVE", "promote")
    act = core.admission
    a = Action(values=(0.4,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
               generated_at=0.0, valid_from=0.0, valid_until=0.32,
               state_stamp=0.0, epoch=1, sequence_id=7)
    ctx = ActionContext("r7", 1, 7)
    # Thread A：对旧世界做 prepare，三道闸此刻全绿（authority 开、r7、epoch 1）。
    assert act.prepare(ctx, a) is True
    # 窗口里 Thread B 发起回滚：authority 关闭后重开、epoch +2、权威 release 翻到 r6。
    assert rollback(sup, core) is True
    # Thread A 在屏障之后才 commit 同一条旧 ctx。真正的竞态 bug 会信 prepare 的一次性"是"；
    # 正确的实现是 generation-checked enqueue——commit 当场重判，旧的 r7 / epoch 1 再也过不了。
    assert act.commit(ctx, a, 0.0) is False
    # 关键：没有"prepare 时放行、commit 时才落地"的漏网动作残留在通道里，两个缓冲槽都空。
    assert act.action_buffer._active is None and act.action_buffer._scheduled is None
```

`python -m pytest tests/test_deploy.py -q` 直接通过（本文付印前实跑：17 passed）。先把这句话的分量说准：**17 passed 证明的是"这套 fake runtime 里定义的不变量可以被重复执行验证"，不是"生产系统里的 deployment safety 已经成立"**——它没做 hardware-in-the-loop、没注入故障、没测分布式回滚的正确性，也没做真验签 / artifact 字节复核 / fleet canary 调度 / 持久化的防篡改证据库；至于异步推理竞态，这一版只由 I17 钉了**确定性交错**（prepare 判闸 → rollback 撤销 → commit 复核被挡），真正的**多线程 / 跨进程**线性化仍是没碰的生产债。把"设计意图"当成"代码已保证"，是这类文章最容易翻的地方，这一篇宁可把线画丑，也不替 demo 吹成生产——至于这些不变量之外那三层用词纪律（demo 保证 / 架构要求 / 生产必须补）的完整展开，见上一篇概念稿的总结。前面几版陆续补齐的是评审一眼能看的"说了没做"：`clamp_diffs` 真的在数（I9）、candidate 的零落地由 runner 内部唯一路径保证而非测试替它调 `submit`（I5，注意这只是当前实现路径性质、不是类型强制）、回滚测的是 `rollback()` 本身而不是缓冲原语、config 热加载有了能跑的 `ConfigManager`（I13，证据止于生效前拒绝）、`release_id` 进了事件与状态机（I12）。而这一轮真正往前挪的一步，是把"唯一命令入口"这句话从**正文比代码强**修成了**代码真兑现**：第一，发布身份成了准入的**必填输入**——决策随一个显式 `ActionContext(release_id, epoch, sequence_id)` 进门，删掉了 `getattr(action, "release_id", 当前版本)` 那条兜底，于是缺 `release_id` 或带空串的旧命令不再被悄悄归成"当前 release"（I16 补了这条反例）；第二，**连 shadow 里 active 决策的落地也走同一道 `CommandAdmission`**，不再抄 `safety.check → sink` 的近路，candidate 与 active 从此过的是同一批闸；第三，`prepare()` / `commit()` 拆开后，用 generation 复核关掉"判闸"与"入队"之间的 TOCTOU 窗口（I17）。这一篇的心脏是 I4、I8、I14、I16、I17 五块：I4 拒绝"重启大法"——SAFE_STOP 之后唯一出路是回到证据起点重走一遍；I8 把回滚落成一次真正的失权，旧版本的在途决策（含排队未派发的）被 epoch 屏障结构性拒绝，而不是靠"等一下让它执行完"；I14 补上最反直觉的一刀——回滚的目标版本也得先过兼容性闸，"旧"不等于"可回滚"；I16 把异步世界的迟到者挡在门外，连"没带身份"的也不放行；I17 则钉住 admission 与 rollback 之间不留竞态窗口——回滚这道撤销屏障一旦落下，之前在闸口排好队、还没 commit 的旧动作也休想补交进来。


## 收个尾

到这儿，概念篇那条 `身份 → 兼容 → shadow → 灰度 → epoch 屏障 → 回滚` 的链，每一环都在上面落成了一个能跑、能复验的断言。但请记住这套绿线证的只是最小模型的局部自洽：真验签、artifact 字节复核、物理停稳的确认闭环、准入闸的跨进程并发加固，仍是明写的生产债。至于这些代码**为什么**这么长、哪些地方刻意留白、以及部署层那六个最常踩的反模式，回到 [9/22 概念篇](/zh/articles/2026-09-22-agent-deployment-rollback/) 接着读；再往前，骨架与 fakes 的来路在 [9/19 架构篇](/zh/articles/2026-09-19-embodied-agent-architecture/)。三步连起来，才是"换路线、换传感器、换机器人"从一句口号变成一条有账、有闸、有证据的工程线。
