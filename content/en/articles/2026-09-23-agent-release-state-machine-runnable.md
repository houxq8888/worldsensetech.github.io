---
title: "Running the Release State Machine: A Pure-stdlib Minimal Closed Loop and Seventeen Invariants"
slug: "2026-09-23-agent-release-state-machine-runnable"
date: 2026-09-23
draft: false
categories: ["Embodied AI", "Tutorial"]
tags: ["Embodied AI", "Software Architecture", "Robotics", "Deployment & Ops", "VLA", "Python", "System Design", "Engineering", "Testing"]
description: "The 9/22 concept piece cast deployment and rollback as an evidence-carrying state machine but gave only the design, not the code. This piece delivers on that half-sentence: a full deploy_fakes.py, a test_deploy.py that pins I1-I17 one assertion at a time, and a real run (17 passed). Prerequisite: it reuses 9/19's fakes.py (ActionBuffer, the epoch barrier, injected clock, single-writer CommandSink); concepts and boundaries live in the 9/22 concept piece."
toc: true
related_articles:
  - 2026-09-22-agent-deployment-rollback
  - 2026-09-19-embodied-agent-architecture
  - 2026-09-16-policy-side-evaluation
  - 2026-09-12-sim-to-real-evaluation-protocol
---

[The 9/22 piece](/en/articles/2026-09-22-agent-deployment-rollback/) cast deployment and rollback as an evidence-carrying state machine -- release identity, the compatibility grid, shadow comparison, gated canary promotion, rollback backed by the epoch barrier, and the authority/release/epoch gates collapsed into a single command admission -- but it gave only the design, leaving the half-sentence "and at the end actually run the release state machine with a pure-stdlib fake implementation" to this piece. Here is where that promise is kept: the full `deploy_fakes.py`, a `test_deploy.py` that pins each invariant as one runnable assertion, and a real run (17 passed).

Let me set the prerequisite straight so nobody gets stuck on "can't run it": this piece **reuses [9/19's fakes.py](/en/articles/2026-09-19-embodied-agent-architecture/)** -- `from fakes import ActionBuffer`, and in the tests `Action / ControllerSink / FakeClock` and the rest all come from the same fake set at the end of that article; injected clock, deterministic policy, the epoch barrier, and the single-writer CommandSink are all present and unchanged, and this piece does not touch them. As for **why** it is designed this way -- the three boundaries people keep missing, the six deployment-layer anti-patterns, and the three-layer word discipline running through the whole series -- that is all in [the 9/22 concept piece](/en/articles/2026-09-22-agent-deployment-rollback/), which now keeps only the design and no longer spreads the code out. This piece does one thing: set the state machine up so it runs, then pin it for you one invariant at a time.

One word-discipline up front: the seventeen green pytest below prove only that "the invariants defined on this stdlib minimal model are repeatably executable and locally self-consistent" -- **not that production deployment safety is established**. Which of them the demo really runs out, which are only architecture requirements, and which still have to be filled in by production are labeled item by item below.

## What This State Machine's Minimal Closed Loop Looks Like

Turn the chain from the concept piece into its minimal runnable-and-testable form -- still no torch, no GPU, reusing 9/19's fakes (injected clock, deterministic policy, epoch barrier, CommandSink single-writer all present, unchanged) and adding only a few things: the layered `Manifest` / `RuntimeVersion` (release identity and compatibility ranges), a `ReleaseSupervisor` with an evidence chain (the sole writer, every transition landing as one `ReleaseEvent` line), `Fault`s routed by nature, a `ShadowRunner` that compares on the same state across five dimensions with observable clamping, and `boot_check` / `promote_allowed` / `ConfigManager` / `rollback` (the four gates from install to retreat). First the skeleton code:

```python
# tests/deploy_fakes.py -- the minimal closed loop for deployment and ops: pure stdlib, runs directly under pytest
# Same discipline as 9/17's fakes.py: injected clock, deterministic policy, epoch barrier, single-writer state machine.
# Scope note: this file verifies only the phase constraints and evidence structure of the release state machine;
# it does not simulate a fleet-level canary scheduler (machine selection, task mix, observation-window auto-promotion are all fleet registry concerns; see the end of the article).
from dataclasses import dataclass, field
from typing import Optional, Tuple

from fakes import ActionBuffer

_SHADOW_RELEASE = "shadow-active"        # the fixed release tag an active command lands under in a pure shadow comparison


def parse_version(s: str) -> Tuple[int, int, int]:
    """MAJOR.MINOR.PATCH minimal parsing: accepts 1~3 numeric segments, right-pads to three, so
    "2.0" equals "2.0.0" and (1,2) is no longer tuple-compared as smaller than (1,2,0). More than 3
    segments, or any non-digit segment, raises ValueError immediately -- so out-of-range inputs like
    "1.2.3.4" and "-1.2.0" are no longer silently accepted (review §3: if the doc says three segments,
    the code must not swallow a fourth or a minus sign). Deliberately not called semver -- it does not
    handle -rc1 / +build7 prerelease metadata. Name the thing right; don't cargo-cult the spec."""
    parts = s.split(".")
    if not (1 <= len(parts) <= 3):
        raise ValueError("version must have 1..3 numeric segments")
    if not all(p.isdigit() for p in parts):
        raise ValueError("version segments must be numeric")
    nums = [int(p) for p in parts] + [0, 0, 0]
    return (nums[0], nums[1], nums[2])


def compat_within(declared: Tuple[str, str], accepted: Tuple[str, str]) -> bool:
    """Containment, not intersection: the entire declared range an artifact supports must lie fully within the runtime's accepted range.
    Intersection would wrongly pass the half-segment the manifest claims to support but the runtime never accepts (e.g. in
    manifest [1.2,2.0] ∩ runtime [1.8,3.0], the 1.2~1.8 part). The continuous semantic-version assumption is discussed in the body:
    versions are modeled here as monotonic continuous intervals -- real discrete capability negotiation should be swapped for a frozenset intersection (see the body)."""
    return (parse_version(accepted[0]) <= parse_version(declared[0])
            and parse_version(declared[1]) <= parse_version(accepted[1]))


# ---------------------------------------------------------------- Release identity and manifest

@dataclass(frozen=True)
class ArtifactIdentity:
    """The manifest's **field carriage** of content identity + provenance identity. Scope: this fake
    only carries these two fields -- it does not recompute the artifact bytes and does no cryptographic
    verification. Canonicalization + signature verification, and the sha256(actual_artifact)==artifact_hash
    re-check, are all the production registry / artifact-loader boundary's job and must complete before the
    manifest enters the state machine (see the body)."""
    artifact_hash: str           # content identity: catches same-name, different-thing (this fake carries it, does not recompute)
    signature: str               # provenance identity: the signature should cover canonical(manifest - signature),
                                 # guarding against "hash unchanged, compat tampered" (this fake does not verify, only models the field and its intended semantics)


@dataclass(frozen=True)
class CompatibilityContract:
    """The three sections of a compatibility declaration. Two boundaries need nailing down first, so they
    aren't over-read:
    ① config_range is a **version-level** compatibility gate -- it only checks "the version falls inside the
       interval". Field-level schema / semantic compatibility (a field rename, a unit change, a new default)
       is the config validator's job -- a different layer; do not read "the version is in range" as "the
       config really is compatible" (review §7).
    ② obs_fingerprint is an **evidence identifier, not a correctness proof** -- equality only proves "the
       preprocessing definition is identical", not "the implementation is correct": two transforms that are
       both wrong still produce the same fp-v7. The 'implementation actually matches' half is caught by the
       numeric comparison against the golden vectors (9/17), not by this string equality (review §8)."""
    schema_range: Tuple[str, str]    # must lie entirely within the runtime's accepts_schema (containment, not intersection)
    config_range: Tuple[str, str]    # both min and max take a stance: too-old config can equally be incompatible
    obs_fingerprint: str             # combined hash of the semantic fingerprint + golden vectors (9/17)


@dataclass(frozen=True)
class ReleasePolicy:
    min_ticks: int                   # promotion gate ①: sample size
    divergence_budget: float         # promotion gate ②: upper bound on divergence rate


@dataclass(frozen=True)
class Manifest:
    """The in-memory form of deploy/release_manifest.yaml: a one-to-one match with the YAML's release/compat/policy sections.
    The manifest is the input to the release state machine -- promotion gates are read from the policy fields, not constants scattered in code."""
    release_id: str                  # release-event identity: the alignment axis of the event stream; monotonicity
                                     # is enforced by the registry/CI issuer, this machine does not check r7>r6 (see the body)
    code_version: str
    artifact: ArtifactIdentity
    compat: CompatibilityContract
    policy: ReleasePolicy


@dataclass(frozen=True)
class RuntimeVersion:
    """Say clearly what this runtime version on the robot will accept.
    Note: code_version here is only an identity carrier -- this fake's boot_check does not use code_version
    for a compatibility judgement; the actual three compatibility axes are accepts_schema / accepts_config /
    expected_fingerprint. Code identity is carried by artifact_hash + code_version (see the body's paragraph
    on whether code is an independent axis)."""
    code_version: str
    accepts_schema: Tuple[str, str]
    accepts_config: Tuple[str, str]
    expected_fingerprint: str


def boot_check(manifest: Manifest, runtime: RuntimeVersion) -> Optional[str]:
    """One gate shared by boot and rollback, **structural only**: checks whether the manifest has a
    release_id, whether the schema/config ranges are fully covered by the runtime, and whether
    obs_fingerprint matches; returns a reason on failure, None on pass. Rather refuse than run with a fault.
    Scope: it does not recompute the artifact bytes and does not verify a signature -- artifact digest /
    signature verification is a registry / artifact-loader pre-step in production (see the body and ArtifactIdentity).
    For release_id it also **only checks presence**: syntax / issuer / uniqueness / monotonicity
    (the r7 > r6 sort of thing) all belong to registry admission, not this local state machine (review §13)."""
    if not manifest.release_id:
        return "missing_release_id"          # an identity-less event stream can't be reviewed; reject outright
    if not compat_within(manifest.compat.schema_range, runtime.accepts_schema):
        return "schema_incompatible"
    if manifest.compat.obs_fingerprint != runtime.expected_fingerprint:
        return "fingerprint_mismatch"
    if not compat_within(manifest.compat.config_range, runtime.accepts_config):
        return "config_incompatible"
    return None


# ---------------------------------------------------------------- Release state machine and evidence chain

TRANSITIONS = {
    "INSTALLED": {"VERIFYING"},
    "VERIFYING": {"SHADOW", "SAFE_STOP"},
    "SHADOW":    {"CANARY", "SAFE_STOP"},
    "CANARY":    {"ACTIVE", "SAFE_STOP"},
    "ACTIVE":    {"DEGRADED", "SAFE_STOP", "ROLLING_BACK"},
    "DEGRADED":  {"ACTIVE", "SAFE_STOP", "ROLLING_BACK"},
    "ROLLING_BACK": {"VERIFYING", "SAFE_STOP"},   # a mid-way failure may fall back to SAFE_STOP
    "SAFE_STOP": {"VERIFYING"},         # after a fault/rollback, rewalk the evidence chain; no going straight back to ACTIVE
}


class Fault:
    SAFETY = "safety"                  # over-limit / stale / controller timeout / e-stop -> SAFE_STOP
    RELEASE = "release"                # divergence rate climbing / success rate slipping / P95 degrading -> DEGRADED + rollback budget
    OBSERVABILITY = "observability"    # metrics gap -> freeze promotion, leave the current execution state alone
    # Scope: this fake lumps OBSERVABILITY into one cell. In production, EVIDENCE_LOSS should be split out of OBSERVABILITY:
    # a ledger write failure means the system temporarily loses the ability to "prove itself safe", and its handling may go
    # beyond freezing promotion to quarantining the candidate. Whether it routes to OBSERVABILITY_DEGRADED or EVIDENCE_LOSS
    # is defined explicitly by the deployment policy (see the body).


@dataclass(frozen=True)
class ReleaseEvent:
    """The smallest unit of the evidence chain: every transition carries identity, time, reason, and epoch --
    when an audit asks "which version entered which state, when, and why," this line is the answer.
    Scope (review §14): in this demo, `supervisor.events` is only a plain list -- this is an **event log**,
    not yet a **tamper-evident evidence chain**. A production registry / audit log has to add event_id,
    previous_event_hash chained into a hash link, append-only persistent storage, and a signer identity
    before the record can be called non-repudiable; none of those live in this state machine."""
    release_id: str
    timestamp: float                   # injected clock: the same clock discipline as 9/17
    from_state: str
    to_state: str
    reason: str                        # operator intent or fault event: "rollback" / "policy_timeout"...
    epoch: int                         # the runtime's epoch at the time: strings the action stream together in review
    evidence_ref: Optional[str] = None # a credential reference to the comparison ledger / CI run


class ReleaseSupervisor:
    """The sole writer of the release state machine (the deployment form of 9/17's Supervisor discipline):
    other components may only request_transition / report_fault, never change state themselves.
    Every legal transition lands as one ReleaseEvent line; illegal transitions are rejected and not recorded.

    A boundary that must be nailed down (review §9/§10; this article takes option A, "keep the model minimal"):
    what this state machine judges is **transition-graph legality** -- `CANARY -> ACTIVE` and
    `DEGRADED -> ACTIVE` are **legal edges**, but a legal edge is not the same as the **business guard** for
    walking that edge being satisfied. The real promotion criterion (`promote_allowed`: sample size +
    divergence budget) and the recovery criterion (health back, divergence back within budget, shadow evidence
    still fresh) are **upper-layer promotion / recovery controller guards**; this demo **deliberately does not
    wire them into the state machine**: `request_transition("ACTIVE", "promote")` looks only at the edge, not
    at the guard. Do not read "the state machine let it through" as "the evidence was enough"."""

    def __init__(self, clock, release_id, epoch=0):
        self._clock = clock
        self.release_id = release_id
        self._epoch = epoch            # read-only mirror: the sole publisher of epoch is RuntimeCore (see rollback)
        self.state = "INSTALLED"
        self.events = []

    def note_epoch(self, epoch):
        self._epoch = epoch            # sync the mirror after RuntimeCore increments it, for the next event

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
        """Faults are routed by nature, not all the way to SAFE_STOP:
        safety touches physical limits and must stop; release is a quality reading, DEGRADED first then the rollback budget;
        observability is just going blind -- freeze promotion, don't interrupt a safely-running control loop, but keep a trace."""
        if kind == Fault.SAFETY:
            self._move("SAFE_STOP", reason, evidence_ref)
        elif kind == Fault.RELEASE:
            self._move("DEGRADED", reason, evidence_ref)
        else:                          # OBSERVABILITY: state unchanged, event still recorded
            self.events.append(ReleaseEvent(
                self.release_id, self._clock.monotonic(), self.state, self.state,
                reason, self._epoch, evidence_ref))
        return self.state


# ---------------------------------------------------------------- Shadow comparison: the divergence ledger

@dataclass
class Divergence:
    """Divergence is not one-dimensional: identical numbers != identical behavior.
    0.50 vs 0.52 may be noise, but valid_from 100ms late is another chunk -- the danger hides in the time semantics."""
    value: bool = False                    # action value exceeds epsilon
    validity: bool = False                 # valid_from takeover-moment misaligned (the time-alignment axis)
    horizon: bool = False                  # chunk duration (horizon*dt) inconsistent (the coverage-window axis)
    sequence: bool = False                 # takeover rhythm (sequence_id step) inconsistent
    safety: bool = False                   # clamping difference: active passes, candidate is clipped


@dataclass
class ShadowStats:
    ticks: int = 0
    divergences: int = 0
    clamp_diffs: int = 0                 # count on the safety dimension: direct evidence the new policy is more aggressive
    validity_diffs: int = 0
    ledger: list = field(default_factory=list)   # per-tick (now, Divergence): the ledger precedes the aggregate

    @property
    def divergence_rate(self) -> float:
        # Scope: aggregate -- any of the five dims firing counts as one divergence, compressing
        # value/validity/horizon/sequence/safety into the same reading, which loses risk weighting
        # (one clamp difference != one 1e-5 numeric difference). Production promotion criteria should
        # at least split safety/clamp as a hard gate and numeric as a soft gate, or better compute a rate per dimension (see the body).
        return self.divergences / self.ticks if self.ticks else 0.0


@dataclass(frozen=True)
class ActionContext:
    """The decision's **identity context** -- a required input to admission, not optional metadata (review §1).
    For an action to earn the right to physical execution it must first prove three things, each on its own field,
    none overlapping: release_id = who produced it (provenance), epoch = which lifecycle it belongs to (lifecycle
    authority), sequence_id = which decision within that lifecycle (ordering). The body's "identity scale" table
    lands on the pre-execution contract right here."""
    release_id: str
    epoch: int
    sequence_id: int


class CommandAdmission:
    """The single command entry (that one door in the body's diagram): the authority / release / epoch gates are
    gathered here, and only after they all pass is it handed to the ActionBuffer for sequence / time-window
    acceptance. Every decision -- a synchronously produced policy action, an async late result, active landing in
    shadow -- may only enter through admit(); only once it earns execution rights can it possibly reach the sink.
    Key: release is a **required input** (carried explicitly on ctx), with no `getattr(..., current)` fallback
    (review §1) -- a decision that carries no identity, or whose identity does not match, is rejected outright."""

    def __init__(self, action_buffer, active_release_id, epoch):
        self.action_buffer = action_buffer
        self.active_release_id = active_release_id
        self.epoch = epoch
        self.authority_open = True

    def _gates_ok(self, ctx, action) -> bool:
        """Pure judgement: authority -> release -> epoch, plus requiring ctx to be self-consistent with the action
        it wants to release (guarding against "an old action relabeled with a new tag"). No side effects, callable
        independently by prepare/commit."""
        if not self.authority_open:                                   # (1) stop issuing new commands (rollback step one)
            return False
        if ctx.release_id != self.active_release_id:                  # (2) release-identity barrier (provenance)
            return False
        if ctx.epoch != self.epoch:                                   # (3) lifecycle barrier (lifecycle authority)
            return False
        if ctx.epoch != action.epoch or ctx.sequence_id != action.sequence_id:
            return False                                              # (4) the identity context must describe this action
        return True

    def prepare(self, ctx, action) -> bool:
        """Only judges the gates, does not enqueue (no side effects). I17 uses it to reproduce the TOCTOU window between check and commit."""
        return self._gates_ok(ctx, action)

    def commit(self, ctx, action, now) -> bool:
        """generation-checked enqueue: re-run the full gate judgement at the exact moment of enqueueing
        (review §12/§13). In a multithreaded setting a rollback can slip in between prepare and commit --
        the re-judgement guarantees there is no linearization window between rollback and admission."""
        if not self._gates_ok(ctx, action):
            return False
        return self.action_buffer.put(action, now)                    # still subject to the buffer's sequence / time-window constraint after passing the gates

    def admit(self, ctx, action, now) -> bool:
        """A synchronously produced decision comes through here: commit immediately after prepare passes (single-threaded, the two are equivalent)."""
        if not self.prepare(ctx, action):
            return False
        return self.commit(ctx, action, now)


class ShadowRunner:
    """Feed the same state to two policies; the candidate's output never enters the sink -- the defining test of shadow mode:
    bookkeeping only, no landing. And the one path that does land **must also pass through the single command entry** (review
    §2 option A): the active decision likewise first does admission.admit(ctx, a, now), and only after passing the gates and
    earning execution rights does it go safety.check -> sink.submit, sharing the very same CommandAdmission as rollback. That
    way the body's "every decision may only enter through this one door" also holds for active, and no longer reads as
    "candidate's safety boundary is more complete than active's command boundary."
    The candidate side still only previews -- it never enters admission or the sink.
    Scope: candidate not reaching the sink remains a property of the **current implementation path** (no code path anywhere
    takes candidate's return value to call admit / sink.submit), not a type-system or capability-token enforcement. Python's
    dynamic typing cannot stop a future refactor from mis-wiring once. What the production version should add is: split
    CandidateAction and Action into distinct types with sink.submit's signature accepting only the latter; or introduce an
    authority token, and the sink rejects when the token does not match. See the body's three-layer vocabulary."""

    def __init__(self, active, candidate, safety, sink, divergence_eps=1e-6, admission=None):
        self.active = active
        self.candidate = candidate
        self.safety = safety
        self.sink = sink
        self.eps = divergence_eps
        self.stats = ShadowStats()
        self._last = None                # (active_seq, candidate_seq): baseline for the takeover-rhythm comparison
        # Default to an entry that runs only active, keeps authority open, and pins release to the shadow tag; rollback tests inject their own.
        self.admission = admission if admission is not None else \
            CommandAdmission(ActionBuffer(epoch=1), _SHADOW_RELEASE, 1)

    def tick(self, state, now) -> bool:
        a = self.active.act(state, now)
        c = self.candidate.act(state, now)
        d = Divergence()
        d.value = max(abs(x - y) for x, y in zip(a.values, c.values)) > self.eps
        chunk_a, chunk_c = a.horizon * a.dt, c.horizon * c.dt
        # Two axes decoupled: validity = takeover moment (time alignment), horizon = chunk duration
        # (coverage window). Merging them into one axis would make horizon=True necessarily imply
        # validity=True, and the dimensions would stop being independent (see the body).
        d.validity = abs(a.valid_from - c.valid_from) > self.eps
        d.horizon = abs(chunk_a - chunk_c) > self.eps
        # sequence contract (as chosen in this article): sequence_id is policy-local -- active/candidate
        # each have their own SequenceAllocator, absolute values never align, so only the step delta is
        # comparable. If your semantics is a global shared sequence, switch this to an absolute
        # a.sequence_id != c.sequence_id check -- the two contracts must not be mixed (see the body).
        d.sequence = (self._last is not None
                      and (a.sequence_id - self._last[0]) != (c.sequence_id - self._last[1]))
        self._last = (a.sequence_id, c.sequence_id)
        _, a_clamp = self.safety.preview(state, (a.values[0],))   # preview only judges, never sends
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
        # Active landing also passes the single command entry first: ctx is built from this action's identity; only fully-green does it clamp+submit
        ctx = ActionContext(release_id=_SHADOW_RELEASE, epoch=a.epoch, sequence_id=a.sequence_id)
        if self.admission.admit(ctx, a, now):
            self.sink.submit(self.safety.check(state, (a.values[0],)))   # only an active that passed the gates lands
        return d.value


def promote_allowed(stats: ShadowStats, policy: ReleasePolicy) -> bool:
    """The promotion gate is data, not courage: the gate itself lives in the manifest's policy section.
    Scope: this implements only two minimal gates -- sample size min_ticks + aggregate divergence rate
    divergence_budget. The safety hard gate (clamping / e-stop frequency), task-stratified coverage,
    success-rate and P95 degradation, and fleet-level policy evaluation all belong to the next scheduler
    layer and are not in this minimal state machine (see the body's scope note)."""
    return stats.ticks >= policy.min_ticks and stats.divergence_rate <= policy.divergence_budget


# ---------------------------------------------------------------- Config hot reload: prepare -> validate -> commit

class ConfigRejected(Exception):
    pass


class ConfigManager:
    """Thresholds are release artifacts too: changing one threshold walks the state machine; you may not bypass the release channel to write on-site parameters directly.
    If validate fails, current stays byte-for-byte unchanged plus an event is kept; the effective point is marked as the next control boundary.
    Scope: commit()'s self.current = staged is an **object reference swap under a single-threaded fake**. To truly guarantee the control loop
    never reads half-new half-old, production needs an immutable ConfigSnapshot + atomic pointer swap / generation number / control-boundary
    latch (or reuse the article's epoch barrier), not one assignment -- this fake does not prove concurrency atomicity (see the body)."""

    def __init__(self, current, runtime: RuntimeVersion):
        self.current = current
        self.runtime = runtime
        self._staged = None
        self.events = []

    def prepare(self, candidate):
        if self._staged is not None:
            raise ConfigRejected("already_staged")      # only one in-flight change at a time
        self._staged = candidate

    def validate(self) -> Optional[str]:
        """Orchestrate the three validation layers, returning the first failure reason (review §9: don't let readers
        read "the version didn't go out of range" as "the config is compatible"). The real field/schema and semantic
        layers here are only **placeholder fakes**; the production version fills in the real logic."""
        c = self._staged
        if c is None:
            return "nothing_staged"
        return self.validate_version(c) or self.validate_schema(c) or self.validate_semantics(c)

    def validate_version(self, c) -> Optional[str]:
        """(1) Version-level compatibility gate: does config_version lie wholly inside the runtime's accepts_config range."""
        v = parse_version(c["config_version"])
        if v > parse_version(self.runtime.accepts_config[1]):
            return "config_too_new"
        if v < parse_version(self.runtime.accepts_config[0]):
            return "config_too_old"                     # forward compatibility must be declared too, it is not assumed
        return None

    def validate_schema(self, c) -> Optional[str]:
        """(2) Field / schema layer: renamed fields, added/removed fields, changed keys -- the version gate does not reach this layer (fake, fill in real logic in production)."""
        return None

    def validate_semantics(self, c) -> Optional[str]:
        """(3) Semantic / physical-quantity layer: is the threshold a legal physical quantity, are the units and defaults right -- likewise independent of version (fake does only a minimal check)."""
        if float(c["max_velocity"]) <= 0:               # a threshold must be a legal physical quantity
            return "unsafe_threshold"
        return None

    def commit(self, now: float) -> bool:
        reason = self.validate()
        staged, self._staged = self._staged, None
        if reason is not None:
            self.events.append(("rejected", staged["config_version"], reason, now))
            return False                                # on failure: current is unchanged, never half-new half-old
        self.events.append(("committed", staged["config_version"], None, now))
        self.current = staged                           # effective point: the next control tick reads the new value
        return True


# ---------------------------------------------------------------- Rollback: first disenfranchise, then flip the pointer, then re-verify

class RollbackRejected(Exception):
    pass


def rollback(supervisor: ReleaseSupervisor, runtime_core) -> bool:
    """Rollback is not flipping a pointer. Order is semantics; not one step may be swapped:
    ① STOP: new commands stop being sent (authority closed)
    ② invalidate: epoch+1 -- the old version's in-flight decisions are structurally disenfranchised, without waiting for them to "finish"
    ③ SAFE_STOP: enter the safe state, the robot is not left hanging mid-air
    ④ the old manifest must pass boot_check again -- "an old version" is not "a rollback-able version"
    ⑤ flip the pointer + a full reset_episode (this fake only sets a reset flag; production clears
       policy hidden state / estimator filters / tracking / seeds inside the reset -- see the body), back to VERIFYING to rewalk the evidence chain
    the sole publisher of epoch is RuntimeCore: the supervisor and ledger only consume the mirror, never increment it themselves."""
    # epoch semantics (as defined here: one cell per lifecycle boundary) -- one rollback crosses two boundaries, hence +2:
    #   epoch N     active execution            (the old world running, about to be invalidated)
    #   epoch N+1   rollback invalidation barrier (step ② bump_epoch: old in-flight decisions structurally disenfranchised)
    #   epoch N+2   installed / reset lifecycle   (step ⑤ install_manifest bumps again: the new world starts,
    #                                            the sequence restarts from 0)
    # This is not a repeated operation: the first cell is the invalidation barrier, the second is the restart barrier (see the body).
    supervisor.request_transition("ROLLING_BACK", "rollback")
    runtime_core.stop_new_commands()                    # ①
    runtime_core.bump_epoch()                           # ②
    supervisor.note_epoch(runtime_core.epoch)
    supervisor.request_transition("SAFE_STOP", "rollback_safe")   # ③
    reason = boot_check(runtime_core.previous_manifest, runtime_core.version)   # ④
    if reason is not None:
        raise RollbackRejected(reason)                  # old version incompatible: stay in SAFE_STOP and ask for help, never hard-flip
    runtime_core.install_manifest(runtime_core.previous_manifest)  # ⑤ flip the pointer + a full reset
    supervisor.note_epoch(runtime_core.epoch)
    supervisor.request_transition("VERIFYING", "rollback_complete")
    return True
```

Then come seventeen pytest, which I do not order by "which function they tested" but by the **invariant** each one pins -- so it matches 9/17's "contract + invariant" style:

| # | Invariant | Test that pins it |
| --- | --- | --- |
| I1 | Missing release identity / schema·config out of bounds / observation-fingerprint drift -> refuse to load (structural check, no artifact recompute, no signature verify) | `test_i1_boot_rejects_missing_identity_and_observation_drift` |
| I2 | Compatibility is a containment check, not an intersection check | `test_i2_compatibility_is_containment_not_intersection` |
| I3 | The state machine must not skip evidence phases | `test_i3_state_machine_rejects_illegal_transition` |
| I4 | After SAFE_STOP you must re-verify; no "just restart it" | `test_i4_safe_stop_forces_reverification` |
| I5 | The candidate's product does not enter the sink (a property of the current implementation path, not a type enforcement) | `test_i5_shadow_never_lands_candidate` |
| I6 | Promotion must satisfy the evidence budget | `test_i6_promotion_gate_requires_evidence` |
| I7 | The epoch barrier rejects stale-epoch decisions (buffer primitive) | `test_i7_action_buffer_rejects_stale_epoch` |
| I8 | After rollback runs: new old-epoch decisions rejected + a chunk already queued-but-not-dispatched also cannot be dispatched (not just `current()` returning None -- both internal ActionBuffer slots, active and scheduled, are cleared at once, pushing the evidence one step from "didn't emit" toward "nothing dispatchable") + full reset leaves a trace + authority/release flip with the transaction | `test_i8_rollback_executes_and_invalidates_queued_inflight` |
| I9 | Clamping differences are observable and countable | `test_i9_clamp_divergence_detected` |
| I10 | Value-identical but time-semantics divergence must be caught | `test_i10_temporal_divergence_without_value_divergence` |
| I11 | Faults are routed by severity, not all e-stopped | `test_i11_fault_severity_routing` |
| I12 | Every legal transition leaves complete evidence | `test_i12_events_form_evidence_chain` |
| I13 | Config hot reload: commit is atomic, an illegal candidate is rejected before it takes effect (does not claim concurrency atomicity or motion) | `test_i13_config_reload_is_atomic_and_rejects_invalid` |
| I14 | An old version != a rollback-able version: rollback must pass the gate again | `test_i14_rollback_must_reverify_old_manifest` |
| I15 | Two numerically-equivalent policies produce bit-identical output within the same episode (not "swapping a policy preserves the runtime seam" -- that would assert the interface structure) | `test_i15_equivalent_policy_swap_preserves_episode_output` |
| I16 | Unified command entry (`ActionContext` carries a required identity): after rollback, a late-arriving async result from the old release is denied -- even when the epoch matches, the release-identity barrier backs it up; **a decision with a missing / empty `release_id` is denied too** (this version has no getattr fallback path) | `test_i16_async_late_result_denied_after_rollback` |
| I17 | No race window between admission and rollback (deterministic interleaving): `prepare()` releases an old action -> `rollback()` invalidates -> `commit()` is turned back by the generation re-check, both buffer slots still empty (the demo proves the deterministic interleaving; real multithreading remains production debt) | `test_i17_admission_and_rollback_have_no_race_window` |

```python
# tests/test_deploy.py -- the minimal deployment-and-ops loop: pin each invariant from the 9/18 body into a runnable assertion
# Same discipline as 9/17: injected clock, deterministic policy, epoch barrier, single-writer state machine.
# Scope: these pytest verify the phase constraints + evidence structure of the release state machine; they do not simulate a fleet-level canary scheduler.
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


# ---------------------------------------------------------------- construction helpers

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
    """A shadow candidate differing from SinePolicy by a constant offset: used to manufacture deterministic divergence."""

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
    """Gives 9/17's SafetyLimiter a **structurally side-effect-free** preview: the pure judgement is split out
    of the commit action. Review P0: the old writing had preview() reuse check() directly, so "the shadow
    produces no side effect" held only **by accident** -- it depended on the contingent fact that check() was
    currently a pure function. A real SafetyGate usually carries state mutation / counters / watchdogs /
    rate-limit bookkeeping; once check() has side effects, a candidate's preview would quietly mutate the
    gate's active-side state. Fix: collapse the pure judgement into a single source evaluate() -- preview only
    reads evaluate() and never touches commit state; check (the active/production landing path) = evaluate +
    commit. That is how "candidate can't get authority" is upgraded from a code-path property to an interface
    property."""

    def __init__(self, max_velocity=5.0, dt=DT):
        super().__init__(max_velocity, dt)
        self.last_command = None        # commit state: only check/commit writes it
        self.clamp_count = 0            # commit state: the clamp counter; preview must never touch it

    def evaluate(self, state, cmd):
        """The single source of truth, pure judgement: returns (approved_cmd, clamped) and writes no self.* state."""
        pos, = cmd
        measured, = state.proprio
        max_delta = self.max_velocity * self._dt
        delta = max(-max_delta, min(max_delta, pos - measured))
        approved = (measured + delta,)
        return approved, approved != cmd

    def preview(self, state, cmd):
        """Shadow / candidate path: reads only the pure evaluate(), never commits -- so the gate's state stays untouched."""
        return self.evaluate(state, cmd)

    def check(self, state, cmd):
        """Active / production landing path: commit the side effects only after evaluate (bookkeeping + clamp count)."""
        approved, clamped = self.evaluate(state, cmd)
        self.last_command = approved
        if clamped:
            self.clamp_count += 1
        return approved


# A minimal runtime stand-in only for wiring up rollback: in a real implementation RuntimeCore itself holds these references.
class RollbackCore:
    def __init__(self, active_manifest, previous_manifest, version, clock):
        self._epoch = 1
        self.active_manifest = active_manifest
        self.previous_manifest = previous_manifest
        self.version = version
        # The unified command entry: the authority / release / epoch gates + ActionBuffer are folded into a
        # CommandAdmission, sharing the very same class as ShadowRunner -- no longer a private rollback-only
        # implementation (review §2).
        self.admission = CommandAdmission(ActionBuffer(epoch=self._epoch),
                                          active_manifest.release_id, self._epoch)
        self.reset_called = False        # observation hook: proves reset_episode really ran

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
        """Delegate straight to the shared CommandAdmission: release is carried explicitly on ctx, with **no fallback** (review §1)."""
        return self.admission.admit(ctx, action, now)

    def stop_new_commands(self):
        self.admission.authority_open = False     # ① stop sending new commands

    def bump_epoch(self):
        self._epoch += 1                          # ② old in-flight decisions structurally disenfranchised
        self.admission.epoch = self._epoch
        self.admission.action_buffer.sync_epoch(self._epoch)

    def reset_episode(self):
        # Production clears policy hidden state / estimator filters / tracking / seeds here;
        # this fake only needs to leave an assertable trace, so I8 can prove "a full reset" is not empty talk (see the body).
        self.reset_called = True

    def install_manifest(self, manifest):
        self.active_manifest = manifest           # ⑤ flip the pointer
        self.admission.active_release_id = manifest.release_id   # the authoritative release switches with it
        self._epoch += 1                          # new-world lifecycle: epoch advances one more cell
        self.admission.epoch = self._epoch
        self.admission.action_buffer.sync_epoch(self._epoch)    # clear active/scheduled + sequence restart
        self.reset_episode()                      # full reset: this fake sets a flag, production clears the above state
        self.admission.authority_open = True


# ================================================================ I1
def test_i1_boot_rejects_missing_identity_and_observation_drift():
    # Scope: boot_check only intercepts the three structural classes -- missing identity + observation drift
    # + schema/config out of bounds; recomputing artifact bytes and verifying signatures are not in this gate,
    # they belong to the registry / artifact-loader boundary (see the ArtifactIdentity note).
    # everything conforms: pass
    assert boot_check(_manifest(), _runtime()) is None
    # missing release_id: the event stream can't be reviewed; reject outright
    assert boot_check(_manifest(release_id=""), _runtime()) == "missing_release_id"
    # obs semantic fingerprint drift: the most common silent bug when swapping a ckpt
    bad_fp = _manifest(compat=CompatibilityContract(
        schema_range=("1.8", "2.0"), config_range=("3.1.0", "3.2.0"),
        obs_fingerprint="fp-v6"))
    assert boot_check(bad_fp, _runtime()) == "fingerprint_mismatch"
    # schema falls outside the runtime's accepted range
    bad_schema = _manifest(compat=CompatibilityContract(
        schema_range=("3.1", "4.0"), config_range=("3.1.0", "3.2.0"),
        obs_fingerprint="fp-v7"))
    assert boot_check(bad_schema, _runtime()) == "schema_incompatible"
    # config too new
    bad_cfg = _manifest(compat=CompatibilityContract(
        schema_range=("1.8", "2.0"), config_range=("5.0.0", "6.0.0"),
        obs_fingerprint="fp-v7"))
    assert boot_check(bad_cfg, _runtime()) == "config_incompatible"


# ================================================================ I2 (review #2: containment != intersection)
def test_i2_compatibility_is_containment_not_intersection():
    # manifest claims support 1.2~2.0, runtime only eats 1.8~3.0: they intersect but half the range isn't covered; must reject
    partial = _manifest(compat=CompatibilityContract(
        schema_range=("1.2", "2.0"), config_range=("3.1.0", "3.2.0"),
        obs_fingerprint="fp-v7"))
    narrow_runtime = _runtime(accepts_schema=("1.8", "3.0"))
    assert boot_check(partial, narrow_runtime) == "schema_incompatible"
    # pin the predicate directly: containment is true, intersection doesn't count
    assert compat_within(("1.8", "2.0"), ("1.2", "3.0")) is True
    assert compat_within(("1.2", "2.0"), ("1.8", "3.0")) is False
    assert parse_version("1.4.0") == (1, 4, 0)


# ================================================================ I3
def test_i3_state_machine_rejects_illegal_transition():
    sup = ReleaseSupervisor(FakeClock(), "r7")
    assert sup.request_transition("VERIFYING", "boot")
    assert sup.request_transition("SHADOW", "shadow_start")
    assert not sup.request_transition("ACTIVE", "skip")     # skip-level promotion: illegal
    assert sup.state == "SHADOW"
    assert sup.request_transition("CANARY", "canary_start")
    assert sup.request_transition("ACTIVE", "promote")
    assert sup.request_transition("DEGRADED", "health_dip")
    assert sup.request_transition("ACTIVE", "recover")      # recovery also takes a legal edge


# ================================================================ I4
def test_i4_safe_stop_forces_reverification():
    sup = ReleaseSupervisor(FakeClock(), "r7")
    sup.request_transition("VERIFYING", "boot")
    sup.request_transition("SHADOW", "shadow_start")
    assert sup.report_fault(Fault.SAFETY, "policy_timeout") == "SAFE_STOP"
    assert not sup.request_transition("ACTIVE", "reboot_magic")  # "just restart it" is rejected
    assert sup.request_transition("VERIFYING", "reverify")       # the only way out: return to the start of the evidence chain


# ================================================================ I5 (review #1: candidate structurally cannot reach the sink)
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
        runner.tick(state, now)          # landing happens only inside tick: active->safety.check->sink
    assert runner.stats.ticks == 20
    assert runner.stats.divergences == 20                    # constant offset: value divergence every tick
    assert len(sink._controller.sent) == 20                  # every landed tick comes from the active branch
    # the candidate's product never shows up in the sink: it only entered the ledger. The test never calls submit on the runner's behalf.


# ================================================================ I6
def test_i6_promotion_gate_requires_evidence():
    policy = ReleasePolicy(min_ticks=100, divergence_budget=0.02)
    few = ShadowStats(ticks=50)                              # not enough samples
    assert promote_allowed(few, policy) is False
    ok = ShadowStats(ticks=200, divergences=2)               # 1% < 2%
    assert promote_allowed(ok, policy) is True
    hot = ShadowStats(ticks=200, divergences=30)             # 15% > 2%
    assert promote_allowed(hot, policy) is False


# ================================================================ I7 (review #13: split the buffer primitive from the real rollback)
def test_i7_action_buffer_rejects_stale_epoch():
    buf = ActionBuffer(epoch=2)
    stale = Action(values=(1.0,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                   generated_at=0.2, valid_from=0.2, valid_until=0.52,
                   state_stamp=0.2, epoch=1, sequence_id=99)
    assert buf.put(stale, 0.2) is False                      # in-flight decision from epoch 1: the barrier rejects it
    assert ("rejected_stale_epoch", 99) in buf.events


# ================================================================ I8 (review #13: actually execute rollback + queued-but-not-dispatched is also invalidated)
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
    # Rollback step one (STOP) is a tested invariant, not just prose: authority starts open, pointing at r7.
    assert core.authority_open is True
    assert core.active_release_id == "r7"
    # Before rollback: an old-epoch decision can be enqueued; deliberately enqueue it as a FUTURE
    # action (valid_from in the future) so it sits in the scheduled slot -- queued but not yet dispatched.
    # This is exactly the class of in-flight decision that rollback safety must pin down.
    inflight = Action(values=(0.5,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                      generated_at=0.0, valid_from=0.3, valid_until=0.62,
                      state_stamp=0.0, epoch=1, sequence_id=1)
    assert core.action_buffer.put(inflight, 0.0) is True     # 0.0 < valid_from -> goes into scheduled
    assert rollback(sup, core) is True
    assert sup.state == "VERIFYING"                          # stops at the start of the evidence chain, not straight back to ACTIVE
    assert core.active_manifest.release_id == "r6"
    assert core.active_release_id == "r6"                    # STOP->...->install: the authoritative release flips to r6
    assert core.authority_open is True                       # closing authority is a **mid-transaction** state of rollback;
    #  once the full five steps finish, the new world reopens the door (the stop only holds inside the transaction -- see rollback's steps ①/⑤)
    assert core.epoch == 3                                   # 1 ->(bump barrier) 2 ->(reset) 3
    assert core.reset_called                                 # "a full reset" really happened, not empty talk
    # Key: the chunk that was already queued in the buffer before rollback and not yet dispatched
    # must NEVER be able to dispatch and execute after the rollback either.
    assert core.action_buffer.current(0.5) is None
    # §6 strengthening: what dispatch really reads is the buffer's internal state -- both slots must be empty
    # before you can call it "the downstream sink will never emit it," and not merely "at this instant it
    # didn't come out." In production there is still a dispatcher / controller queue between buffer and MCU;
    # this assertion lands the disenfranchisement proof at the "there is nothing to dispatch" level (see the body's physical boundary).
    assert core.action_buffer._active is None and core.action_buffer._scheduled is None
    # A newly inserted old-epoch decision is likewise rejected by the barrier (the other face of structural disenfranchisement)
    stale = Action(values=(0.5,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                   generated_at=0.0, valid_from=0.0, valid_until=0.32,
                   state_stamp=0.0, epoch=2, sequence_id=5)
    assert core.action_buffer.put(stale, 0.0) is False       # in-flight decision from the rollback-era epoch: already disenfranchised


# ================================================================ I9 (review #9: clamp_diffs really counts)
def test_i9_clamp_divergence_detected():
    clock = FakeClock()
    active = OffsetPolicy(0.0)                               # ~0.02, does not trigger clamping
    candidate = OffsetPolicy(6.0)                            # exceeds max_velocity*dt, gets clipped
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
    assert runner.stats.clamp_diffs == 5                     # every tick active passes, candidate is clipped
    assert all(d.safety for _, d in runner.stats.ledger)
    # the candidate sees the clamping verdict, but its product never lands: what the sink receives is still active's approved value
    assert all(cmd[0] <= 5.0 * DT + 1e-9 for cmd in sink._controller.sent)


# ================================================================ I10 (review #8: time-semantics divergence)
def test_i10_temporal_divergence_without_value_divergence():
    clock = FakeClock()
    active = SinePolicy()
    candidate = SlowPolicy(latency=0.15)                     # same values, effective moment 0.15s later
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
    assert value_diverged is False                           # value dimension: zero divergence
    assert runner.stats.validity_diffs == 5                  # time-semantics dimension: diverges every tick
    assert runner.stats.divergences == 5                     # if it only compared values, this state machine would go "all green" and miss it
    assert all(d.validity and not d.value for _, d in runner.stats.ledger)


# ================================================================ I11 (review #5: fault grading, not a blanket e-stop)
def test_i11_fault_severity_routing():
    clock = FakeClock()
    sup = ReleaseSupervisor(clock, "r7")
    sup.request_transition("VERIFYING", "boot")
    sup.request_transition("SHADOW", "shadow_start")
    sup.request_transition("CANARY", "canary_start")
    sup.request_transition("ACTIVE", "promote")
    # observability fault: metrics gap -- freeze promotion, don't interrupt a safely-running control loop, but keep a trace
    assert sup.report_fault(Fault.OBSERVABILITY, "metrics_gap") == "ACTIVE"
    assert sup.events[-1].to_state == "ACTIVE"
    assert sup.events[-1].reason == "metrics_gap"
    # release-health degradation: divergence rate climbing -- go to DEGRADED, not SAFE_STOP
    assert sup.report_fault(Fault.RELEASE, "divergence_rise") == "DEGRADED"
    # safety fault: touches physical limits -- must go SAFE_STOP
    assert sup.report_fault(Fault.SAFETY, "command_over_limit") == "SAFE_STOP"


# ================================================================ I12 (review #4/#17/#18: the evidence chain is complete)
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
    assert e1.timestamp > e0.timestamp                       # injected clock: time is reviewable
    assert e1.reason == "shadow_start" and e1.from_state == "VERIFYING"
    # illegal transitions aren't recorded: the evidence chain records only moves that really happened
    n_before = len(sup.events)
    assert not sup.request_transition("ACTIVE", "illegal")
    assert len(sup.events) == n_before


# ================================================================ I13 (review #14/#15: the evidence reaches "atomic commit + rejects invalid candidate")
def test_i13_config_reload_is_atomic_and_rejects_invalid():
    # The name is downgraded from fail-before-motion to atomic-and-rejects-invalid: this fake has no motion
    # and no control loop concurrently reading config; what it can prove is "an illegal candidate's commit
    # fails and current stays byte-for-byte unchanged" (see the body).
    runtime = _runtime()
    cfg = ConfigManager({"config_version": "3.1.0", "max_velocity": 5.0}, runtime)
    cfg.prepare({"config_version": "3.1.5", "max_velocity": 4.0})
    assert cfg.validate() is None
    assert cfg.commit(now=1.0) is True
    assert cfg.current["config_version"] == "3.1.5"          # legal: the next tick reads the new value
    # illegal threshold: prepare->validate fails -> current stays byte-for-byte unchanged
    cfg.prepare({"config_version": "3.1.6", "max_velocity": -1.0})
    assert cfg.commit(now=2.0) is False
    assert cfg.current["max_velocity"] == 4.0                # never half-new half-old
    assert ("rejected", "3.1.6", "unsafe_threshold", 2.0) in cfg.events
    # forward compatibility must be declared too: too-old config is also rejected (not assumed fine)
    cfg.prepare({"config_version": "2.0.0", "max_velocity": 3.0})
    assert cfg.validate() == "config_too_old"
    cfg.commit(now=3.0)
    # only one in-flight change at a time
    cfg2 = ConfigManager({"config_version": "3.1.0", "max_velocity": 5.0}, runtime)
    cfg2.prepare({"config_version": "3.1.1", "max_velocity": 5.0})
    with pytest.raises(ConfigRejected):
        cfg2.prepare({"config_version": "3.1.2", "max_velocity": 5.0})


# ================================================================ I14 (review #20: an old version != a rollback-able version)
def test_i14_rollback_must_reverify_old_manifest():
    clock = FakeClock()
    # the previous release r6's obs fingerprint doesn't match the current runtime: no hard rollback
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
    assert sup.state == "SAFE_STOP"                          # stays in the safe state asking for help, never flips back with a fault
    assert core.active_manifest.release_id == "r7"           # the pointer wasn't polluted


# ================================================================ I15 (review #18: don't let the name outgrow the content)
def test_i15_equivalent_policy_swap_preserves_episode_output():
    # Scope: what this tests is "two numerically-equivalent implementations produce bit-identical
    # sent/safe_entries within the same episode". It does not, and cannot, prove "swapping a policy
    # leaves the runtime seam / contract shape unchanged" -- that would require asserting the interface
    # structure itself: PolicyProtocol / Action / StateContract / reset() / act() (see the body).
    ctrl_a, safety_a = run_episode(FakeClock(), n_steps=20, policy=SinePolicy())
    ctrl_b, safety_b = run_episode(FakeClock(), n_steps=20, policy=OffsetPolicy(0.0))
    assert ctrl_a.sent == ctrl_b.sent                        # compare directly within the same episode, no extra run
    assert safety_a.safe_entries == safety_b.safe_entries


# ================================================================ I16 (review §1/§16/§18: an async late result must not re-enter the command channel, and release is a required input)
def test_i16_async_late_result_denied_after_rollback():
    clock = FakeClock()
    core = RollbackCore(_manifest(release_id="r7"),
                        _manifest(release_id="r6", code_version="1.3.9"),
                        _runtime(), clock)
    sup = ReleaseSupervisor(clock, "r7")
    sup.request_transition("VERIFYING", "boot")
    sup.request_transition("CANARY", "canary_start")
    sup.request_transition("ACTIVE", "promote")
    # Before rollback: a synchronous decision on the current release / epoch carries its identity and passes the gate normally, pushing the sequence baseline to 10.
    ok = Action(values=(0.3,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                generated_at=0.0, valid_from=0.0, valid_until=0.32,
                state_stamp=0.0, epoch=1, sequence_id=10)
    assert core.admit(ActionContext("r7", 1, 10), ok, 0.0) is True
    assert rollback(sup, core) is True
    # Late arrival A: old-world epoch=1 -> the lifecycle barrier rejects it first (both release and epoch are still old).
    late_a = Action(values=(0.7,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                    generated_at=0.0, valid_from=0.0, valid_until=0.32,
                    state_stamp=0.0, epoch=1, sequence_id=11)
    assert core.admit(ActionContext("r7", 1, 11), late_a, 0.0) is False
    # Late arrival B: the epoch happens to equal the new world's 3, but the ctx still carries the old release r7 ->
    # only the "release-identity barrier" can stop it. This is exactly the fish that slips through when the
    # epoch barrier carries the load alone (review §5).
    late_b = Action(values=(0.9,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                    generated_at=0.0, valid_from=0.0, valid_until=0.32,
                    state_stamp=0.0, epoch=core.epoch, sequence_id=99)
    assert core.admit(ActionContext("r7", core.epoch, 99), late_b, 0.0) is False
    # §1 counterexample: a decision that carries **no release identity** (release_id missing / empty), even
    # when its epoch happens to equal the current world, must be rejected at the door -- release is a required
    # input, there is no "default to the current release" fallback.
    no_release = Action(values=(0.8,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
                        generated_at=0.0, valid_from=0.0, valid_until=0.32,
                        state_stamp=0.0, epoch=core.epoch, sequence_id=100)
    assert core.admit(ActionContext("", core.epoch, 100), no_release, 0.0) is False
    # None of the barriers let anything through -> the command channel stays empty: the sink never receives any dispatch from an old release / an unidentified decision.
    assert core.action_buffer.current(0.1) is None


# ================================================================ I17 (review §12/§13: no linearization window between admission and rollback)
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
    # Thread A: prepare against the old world -- all three gates green right now (authority open, r7, epoch 1).
    assert act.prepare(ctx, a) is True
    # In the window, Thread B launches a rollback: authority closes then reopens, epoch +2, authoritative release flips to r6.
    assert rollback(sup, core) is True
    # Thread A only commits the same old ctx after the barrier. A real race bug would trust prepare's one-shot "yes";
    # the correct implementation is a generation-checked enqueue -- commit re-judges on the spot, and the stale r7 / epoch 1 no longer passes.
    assert act.commit(ctx, a, 0.0) is False
    # Key: no "released at prepare, landed at commit" straggler is left in the channel -- both buffer slots are empty.
    assert act.action_buffer._active is None and act.action_buffer._scheduled is None
```

`python -m pytest tests/test_deploy.py -q` passes directly (run before this piece went to print: 17 passed). First let me state the weight of that sentence accurately: **17 passed proves that "the invariants defined inside this fake runtime can be repeatably executed and verified," not that "production deployment safety has been established"** -- it does not run hardware-in-the-loop, inject faults, or verify the correctness of a distributed rollback, and it never touches signature verification / artifact-byte re-check / a fleet canary scheduler / a persistent tamper-evident evidence store; as for async-inference races, this version pins only a **deterministic interleave** via I17 (prepare checks the gates → rollback revokes → commit's re-check is turned back), while true **multithreaded / cross-process** linearization remains untouched production debt. Reading "design intent" as "already guaranteed by code" is the easiest place for this kind of article to stumble, so this piece would rather draw the line ugly than inflate the demo into production -- and for the full three-layer word discipline (what the demo guarantees / what the architecture requires / what production must fill in), see the summary of the concept piece before it. What earlier versions gradually filled in were the "said but didn't do" items a review catches at a glance: `clamp_diffs` really counts (I9), the candidate's zero-landing is guaranteed by a single internal path in the runner rather than the test calling `submit` for it (I5, note this is only a current-implementation-path property, not a type enforcement), rollback tests `rollback()` itself rather than the buffer primitive, config hot reload now has a runnable `ConfigManager` (I13, the evidence reaches rejecting before it takes effect), and `release_id` enters the events and the state machine (I12). And the step this round really moved forward is fixing the sentence "a single command entry" from **prose stronger than code** to **code that actually delivers it**: first, release identity became a **required admission input** -- each decision enters with an explicit `ActionContext(release_id, epoch, sequence_id)`, and the `getattr(action, "release_id", current_version)` fallback is deleted, so an old command missing `release_id` or carrying an empty string is no longer silently relabeled as "the current release" (I16 adds this counterexample); second, **even active landing in shadow now goes through the same `CommandAdmission`**, no longer taking a `safety.check → sink` shortcut, so candidate and active pass the very same gates from now on; third, once `prepare()` / `commit()` are split apart, a generation re-check closes the TOCTOU window between "checking the gates" and "enqueueing" (I17). The heart of this article is the five blocks I4, I8, I14, I16, and I17: I4 rejects "just restart it" -- after SAFE_STOP the only way out is to return to the start of the evidence chain and walk it again; I8 turns rollback into a real disenfranchisement, with the old version's in-flight decisions (including those queued but not yet dispatched) structurally rejected by the epoch barrier rather than by "wait a moment and let it finish"; I14 adds the most counterintuitive cut -- the rollback target version must first pass the compatibility gate, "old" does not mean "rollback-able"; I16 keeps the async world's late-comers outside the door, and does not let even the "carries no identity" ones through; I17 pins that no race window is left between admission and rollback -- once the rollback revocation barrier falls, an old action that had already queued at the gate but not yet committed cannot smuggle itself in afterward.


## Wrapping Up

At this point every link in the concept piece's chain -- `identity -> compatibility -> shadow -> canary -> epoch barrier -> rollback` -- has landed above as a runnable, re-verifiable assertion. But remember these green lights prove only the local self-consistency of a minimal model: real signature verification, artifact-byte re-check, the physically-confirmed stop closed loop, and cross-process concurrent hardening of the admission gate are all still explicitly stated production debt. As for **why** this code is this long, where it deliberately leaves gaps, and the six most common anti-patterns at the deployment layer, go back and read [the 9/22 concept piece](/en/articles/2026-09-22-agent-deployment-rollback/); further back, the skeleton and the origin of the fakes are in [the 9/19 architecture piece](/en/articles/2026-09-19-embodied-agent-architecture/). Read the three together, and "swap the route, the sensor, the robot" turns from a slogan into one engineering line with a ledger, a barrier, and evidence at every step.
