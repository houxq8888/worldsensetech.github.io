---
title: "Deployment and Ops for Embodied AI: Swapping a Component Is Not Releasing a Version, It Is Feeding a State Machine"
slug: "2026-09-22-agent-deployment-rollback"
date: 2026-09-22
draft: false
categories: ["Embodied AI", "Tutorial"]
tags: ["Embodied AI", "Software Architecture", "Robotics", "Deployment", "VLA", "Python", "System Design", "Engineering Architecture", "Sim-to-Real"]
description: "The 9/17 piece gave a skeleton that runs, tests, and swaps components; this one answers the next question: what catches you the instant a component comes out. A web service rollback is an undo, but a robot rollback disenfranchises old decisions. It covers release identity (a layered manifest plus signature semantics), a code x schema x config compatibility grid built on containment rather than intersection, a five-dimensional shadow divergence ledger with clamping differences, canary with promotion gates, a five-step rollback backed by the epoch barrier, fail-before-motion hot config reload, and a pure-stdlib minimal release state machine pinned by fifteen invariants."
toc: true
related_articles:
  - 2026-09-19-embodied-agent-architecture
  - 2026-09-16-policy-side-evaluation
  - 2026-09-12-sim-to-real-evaluation-protocol
  - 2026-09-15-policy-side-interface
  - 2026-09-09-robot-data-scaling
---

A web service rollback is an undo -- route the traffic back to the old version and, once the error rate drops, you are done. A robot rollback is disenfranchising old decisions -- you flip the manifest pointer back, but that version still has a string of in-flight chunks, a queued async inference, and un-zeroed hidden state clutching control and refusing to let go. This article is about exactly that gap.

The previous article stood the skeleton up: [the six-layer runtime stack, the three core contracts, and a minimal closed loop that runs and tests](/en/articles/2026-09-19-embodied-agent-architecture/), with the conclusion that "swapping the route, the sensor, or the robot is demoted from open surgery to plugging in a component." But between "it can be plugged and unplugged" and "you dare to pull it out and plug it back in" there is still a whole unwritten layer -- **the moment it comes out, who catches you?**

Swap a board and a bad solder joint can be reworked; swap a set of policy weights while the robot is carrying a glass across the room. It is told in the same style as 9/17 -- get the concepts to a discussable state first, then at the end actually run the release state machine with a pure-stdlib fake implementation (fifteen invariants; run before this piece went to print: 15 passed). What sets this apart from ordinary DevOps can be said in one line: **it treats deployment as part of the robot's control safety boundary**, so the whole article is really one chain --

```text
release_id -> compatibility -> shadow evidence -> canary -> epoch barrier -> rollback
   identity       can it install   does it move once installed   whom to run   old decisions lose power   retreat if it breaks
```

If identity doesn't line up, there is no reviewable ledger behind it; if compatibility is never tested, installing is running with a fault; without a shadow ledger, promotion runs on nerve; without the epoch barrier, a rollback is the old and new versions fighting over the same robot. Each section takes one link.

## Set One Principle First: Deployment Is Not an Action, It Is a State Machine with Evidence

Deploying a web backend is roughly "put the new code up, watch the error rate." Transplanting that intuition onto a robot is dangerous, and dangerous on three levels. First, **the rollback window is not symmetric**: a web rollback undoes; a robot rollback recovers from a physical state in which a wrong action has already been executed -- the glass has already fallen. Second, **the observation surface is not symmetric**: error rate and latency curves are centralized, but whether a robot is "right" is scattered across every episode on every machine, and by the time you see the on-site incident the samples were long since consumed. Third -- and this is what the article keeps returning to -- **one deployment changes more than code**: bump the ckpt by one version and the normalization statistics, the ObsTransform, the calibration, and the control thresholds are often all in motion; these "invisible versions" raise not a single line of error, they just let behavior quietly drift.

So the correct model of deployment is not an action but a **state machine plus an evidence chain**: every version move must leave a reviewable credential (who built it, against which schema, which checks it passed, whether it was compared head-to-head with the old version and diverged), and state may only move along legal edges -- illegal moves are structurally rejected rather than left to procedural self-discipline. Rings a bell? This is exactly the rule 9/17 set for the runtime -- command authority has a single writer, the failure state machine has a single owner, the epoch barrier -- now carried, unchanged, onto the release channel. The better the seams the architecture article designs, the cheaper the operations in this one; that is the real coupling between the two.

## Release Identity: Every Field in the Manifest Is a Fool-Proof Switch

9/17 gave a `deploy/artifact_manifest.yaml`: checkpoint, code_commit, schema version, normalization statistics, calibration file, container digest. Back then its job was "reproduce one deployment"; here it takes on a second job -- **serving as the input to the release state machine**. The fields themselves are nothing new; what's new is the gate configured for each of them:

```yaml
# deploy/release_manifest.yaml -- the ops-hardened form of artifact_manifest
release:
  release_id: 2026.09-r7        # monotonic release sequence: all event streams align on it (see below)
  checkpoint: ckpt/diffusion_v17.pt
  artifact_sha256: 9b41...      # weight hash: tags change, hashes don't
  code_commit: 8f3a2c1
  signature: minisig:r7.ok      # provenance signature: covers canonical(manifest - signature), not just the artifact
compat:
  state_schema: [v3, v4]        # declared supported schema interval: must lie entirely within the runtime's accepted range (containment, not intersection)
  action_schema: [v2]
  obs_fingerprint: fp-v7        # combined hash of the semantic fingerprint + golden vectors (9/17)
  config_range: [3.1.0, 3.2.0]  # forward + backward compatibility interval: both min/max take a stance, too-old can also be incompatible
runtime:
  container_digest: sha256:71c0...
  python: "3.11"
policy:                          # release policy enters the manifest: promotion gates are data, not constants scattered in code
  baseline_release: 2026.09-r6  # compare head-to-head with whom
  divergence_budget: 0.02       # promotion gate: upper bound on divergence rate
  min_ticks: 10000              # promotion gate: minimum sample size
```

In code this manifest is not one flat pile of fields but **split into three layers, a one-to-one match with the YAML's three sections**: `ArtifactIdentity` (hash + signature) handles "which part is this and who built it", `CompatibilityContract` (schema range + config range + obs fingerprint) handles "can it run together with the current runtime", and `ReleasePolicy` (min_ticks + divergence_budget) handles "how much evidence must accumulate before promotion is allowed." The layering is not fastidiousness -- it lets `boot_check`, `promote_allowed`, and `rollback` each read only the section they should, and the promotion gates are taken from the manifest's policy fields rather than three magic constants scattered through the state machine.

The three identity fields each block a different class of accident, and it matters to see that they are **not the same thing**. `artifact_sha256` is the **content identity**, guarding against "right name, wrong thing": `diffusion_v17.pt` is a name, not an identity, and same-name-different-content ckpts are a regular at every "what on earth, the behavior is different" crime scene. `signature` is the **provenance identity** (authenticity / provenance), guarding against "the content wasn't swapped, but it's unclear who signed it" -- the key point is that what the signature should cover is the entire canonical byte string `canonical(manifest - signature)`, not merely the artifact's hash; otherwise you get the bypass where "the hash didn't change, but someone quietly edited the schema range in compat." `release_id` is the **release-event identity**, guarding against misaligned event streams: operational events (deployment, comparison, canary, rollback) span three clock domains -- CI, the registry, and every robot -- and without a monotonic sequence number there is no reviewable ledger. `obs_fingerprint` guards against definition drift: in 9/17 it watches train/serve skew, and here it watches release/serve skew -- a new ckpt wired to old preprocessing, not one field missing yet the whole behavior inverted, is caught on the spot by per-element numerical comparison. The traceability requirement of the evaluation protocol ([9/12](/en/articles/2026-09-12-sim-to-real-evaluation-protocol/)), when it lands in the deployment layer, looks exactly like this one manifest plus these gates.

There is another set of easily-confused identity scales worth nailing down once here, because each solves a different alignment problem: `release_id` (which release), `robot_id` (which machine), `episode_id` (which job), `epoch` (which barrier-delimited segment of this job's lifecycle), `sequence_id` (which decision within the segment). In particular, be clear that **`release_id != epoch`**: one release can span many episodes, and a single episode may bump the epoch several times because of a rollback. During an incident review they are nested coordinates -- `release r7 / episode 42 / epoch 8 / sequence 183` -- and once you roll back it jumps to `release r6 / episode 43 / epoch 9 / sequence 0`. Miss any one layer and you cannot precisely point in the logs at "which decision knocked the glass off."

## The Compatibility Grid: Not Backward Compatibility, but "Which Cells Can Run Together"

The most common wrong assumption in version management is linearity: new code is compatible with old data, old code with new data, backward all the way. An embodied system's versions move along at least three independent axes, and they are not orthogonal to each other:

```text
            schema  v3      v4      v5
code        ─────────────────────────────
  1.4.x      ✓        ✓      ✗      ✗
  1.5.x      ✗        ✓      ✓      ✗
config      ─────────────────────────────
  3.1        ✓        ✓      ✗      ✗
  3.2        ✗        ✓      ✓      ✗
             ↑ each cell = one real boot check that actually ran; an untested cell defaults to ✗
```

Three disciplines. **One, the grid records only combinations that genuinely existed.** Teams that insist on maintaining a full compatibility matrix end up testing none of it. The typical truth of a robot fleet is: on any one machine, the runtime lagging the registry by a version is the norm -- so the compatibility interval must be modeled explicitly (the `[v3, v4]` under `compat`), not left to "should be fine, right?" **Two, an untested cell defaults to ✗.** Compatibility is a property you test into existence, not one you declare; behind every ✓ there must be a real `boot_check` that ran plus a round of shadow comparison. This sentence needs a follow-up about a real-world pit: **whose head does that ✓ actually sit on?** If the ✓ is just a boolean someone hand-typed into YAML in the registry, it will sooner or later drift from reality. The trustworthy approach is to have the ✓ carry an evidence reference (which CI run, which shadow round, which test suite ran), written by the pipeline and never edited by hand -- and this is exactly the interface the next article on fleet version governance will pick up. **Three, config too must be forward-compatible.** During a canary rollback, the old runtime often has to load config the new version already changed (episode 42); a field renamed or its semantics shifted crashes the old code on the spot -- the easiest path to patching in a second fault the very night you roll back. So config changes follow the standard expand-migrate-contract: only add, never delete; give new fields defaults; put deletion after every machine has upgraded; the `config_range` in the manifest, with both a min and a max, is the enforcing gate on this discipline -- too old and out of bounds are equally suspect.

One modeling confession: `compat_within` treats versions as **monotonic continuous semantic intervals** and does a containment check (the entire declared range must lie within the runtime's accepted range), which holds when the schema really evolves linearly. But in the real world versions are often **discrete capabilities** -- supporting `{v3, v4}` does not mean supporting some pseudoversion in between with different fields. For that case you would swap this for an intersection over capability sets, not intervals. This article chose the interval model for a minimal closed loop and left that seam open in a code comment; if your schema is a set of discrete IDs, please replace "interval containment" with "set containment." Note that I deliberately used **containment, not intersection**: the manifest declares support for `[1.2, 2.0]`, the runtime only eats `[1.8, 3.0]`, they intersect but must never be judged compatible -- because the runtime doesn't recognize the `1.2~1.8` half the manifest promises, and an intersection test would let it through the gate.

The startup validation order is also worth nailing down, from cheap to expensive: **first structure** -- are the manifest fields complete, is there a `release_id`, is the schema range fully covered by the runtime, does the config version cross the bounds; pure comparison, milliseconds. **Then numbers** -- re-run the shared `ObsTransform` on the golden vectors in the release manifest and compare semantic fingerprints (9/17's technique, reused at the deployment point). **Last, behavior** -- enter shadow mode for a head-to-head comparison (next section). Rather refuse to boot than run with a fault: fail-before-motion has been said ten thousand times, and every defeat comes down to "let's run it and see."

## Shadow Mode: Compare Decisions, Never Touch the Ground

The cheapest insurance in a release system is to let the new policy **see everything and touch nothing**. On the same `StateContract` stream, the old and new policies each produce an `Action`: the old one goes, as usual, through SafetyGate -> CommandSink to real execution; the new one goes into the shadow ledger:

```text
StateContract stream ──→ active policy ──→ ActionBuffer → SafetyGate → CommandSink → real robot
        └──────────→ candidate policy ─→ divergence ledger (bookkeeping only, no path of its own in the sink)
```

The implementation discipline of shadow mode is a reuse of two of 9/17's old rules. First, **authority is unique**: the candidate's output has no path to the RobotInterface -- precisely, under a dynamic type system like Python this is not "it can't get in at the type level" but "there is no such call path structurally": inside `ShadowRunner.tick()` the return value of `candidate.act()` flows only into the divergence ledger, and the only place that calls `sink.submit()` is fed the product of `active` through `safety.check()`. Type-level enforcement (splitting `CandidateAction` and `Action` into distinct types and letting the sink accept only the latter) is what the production version should add; we do not claim it is already done here. Second, **comparison must be same-state, same-clock**: both are fed the same state object from `latest_valid(now)` on the same tick, under the same injected clock; divergence computed by a shadow run on a stale state is all noise.

The ledger should not record just one number. Divergence is a matter of more than one dimension: **identical numbers != identical behavior.** The real danger is often not `0.50` vs `0.52` (that may be noise) but the same action's `valid_from` being 100ms late -- that is **another chunk**, taking over at the wrong physical moment. So `Divergence` is split into five dimensions: `value` (numerical value over ε), `validity` (valid_from / chunk duration misaligned), `horizon` (chunk coverage inconsistent), `sequence` (takeover rhythm, whether the sequence_id step matches), and `safety` (clamping difference). The one most easily missed is the last, and the one most worth watching: the **clamping difference** -- for the same command, SafetyGate releases the old policy but clips the new one; even if the task success rate shows nothing yet, that is direct evidence the new policy is more aggressive. In implementation, both active and candidate pass through `safety.preview()` (judge only, don't send), and by comparing the two `clamped` flags `clamp_diffs` really counts; the candidate sees the safety gate's verdict yet still cannot reach the sink -- which demonstrates "can observe, cannot execute" at the same time. Divergence rate is the risk reading; a canary that ignores it is running bare.

The shadow stage has two failure conditions to guard against in advance as well. **The candidate must not touch state**: the shadow policy is forbidden from writing anything back to the estimator / StateBuffer, otherwise "observe only" becomes co-decision. **Comparison must pick the boundary cases**: on a uniform timeline the two policies agree most of the time, of course -- the divergence ledger must be stratified by the task labels declared in the manifest (contact, occlusion, calibration drift); a comparison window that fails to cover the boundary cases is a comparison that didn't happen.

## Canary and Promotion Gates: The Divergence Ledger Has the Say

Only after the head-to-head comparison passes does real execution come into play. A canary in the robotics setting is a different thing from a percentage of web traffic -- **the natural slices of traffic are not users but tasks, robots, and time windows**: shadow first, then a single task on a single machine, then a same-model fleet, then cross-model; every step is "real execution + a fault budget," differing only in blast radius.

The rules of promotion must be set up front: **the gate is data, not courage.** The two fields in `release_manifest.yaml` -- `min_ticks` and `divergence_budget` -- are the hinge of the promotion gate: not enough samples, no promotion (even if the divergence rate is zero); divergence over budget, no promotion (even if the sample is large enough). The most taboo shape is "let's keep watching": observation without a quantified gate always slides toward Friday-afternoon "looks fine, ship it." The rollback criterion works the same way, and moreover must be graded by hardness: frequent safety-gate intervention, divergence crossing a hard threshold, is **immediate rollback**, no discussion; slowly degrading coverage or success rate is **in-budget rollback**, counted per window. Both kinds of criterion must be written as computable expressions before the release, not arrived at by meeting for consensus afterward.

One scope note, so readers aren't misled: the pytest suite in this article verifies the **phase constraints and evidence structure of the release state machine** -- which states may legally transition between, whether promotion requires enough evidence, whether rollback must pass the gate again. It does **not simulate a fleet-level canary scheduler**: machine selection, task mix, and auto-promotion at the end of an observation window are the domain of the fleet registry and belong to the next article. The `CANARY` that appears here is only a phase name guarded by the state machine, not a scheduler.

## Rollback: 9/17's Epoch Barrier Goes On Duty a Second Time

Rollback is hard, and not because of switching the config. The config flips in a second; what floats out afterward is the real trouble -- **in-flight things**: chunks produced by the old version still in the ActionBuffer (a copy still lying in the scheduled slot), an async inference queued in the policy service, a trajectory segment mid-execution, and the policy's hidden state. A rollback that only flips the manifest pointer is equivalent to letting the old and new versions co-drive the same robot -- the old version's last decision is still queuing in the buffer while the new version's first decision has already entered, and their sequence_ids are each counting on their own.

So rollback is not a one-line assignment but a small transaction in which **order is semantics**, and not one of the five steps may be swapped: ① stop sending new commands, close command authority; ② `epoch + 1`, which **structurally disenfranchises** the old version's in-flight decisions -- without waiting for "them to finish," and without letting them override the post-rollback version; ③ enter `SAFE_STOP`, so the robot is not left hanging in a middle state; ④ the old manifest **must pass `boot_check` once more**; ⑤ only then flip the pointer + run a full `reset_episode` (clear the policy's hidden state, filters, and tracking), and go back to `VERIFYING` to rewalk the evidence chain. The epoch barrier 9/17 designed was prepared for step ②: after the epoch is incremented and synced to both the policy and the ActionBuffer, every in-flight decision of the old version -- however large its sequence_id, however far from taking effect -- is rejected uniformly at the `put()` epoch barrier; the sequence space restarts, so there is no cross-version sequence entanglement. "Wait for the old chunk to finish, then switch" is neither needed nor allowed: the barrier disenfranchises the old version, and this is not about being fast, it is about being clean.

Step ④ is the single most easily missed and most lethal line in the article: **"an old version" is not "a rollback-able version."** The release you roll back to -- its schema, config, and obs_fingerprint may no longer be compatible with the **current** runtime, especially if you already upgraded the runtime first. Re-run a compatibility check against the old manifest before going back; if it fails, stop in `SAFE_STOP` and ask for help, never hard-flip: force-switching to an incompatible old version is just replacing an old bug with a new one. In the code this step raises `RollbackRejected` on failure, the state machine rests firmly in `SAFE_STOP`, and the pointer does not move a single word.

This also draws out an architectural rule running through both articles: **the sole publisher of epoch is RuntimeCore.** The supervisor, policy, and ActionBuffer only **accept** epoch and sync a mirror; no one may do `epoch += 1` themselves. Otherwise you get the half-synchronized hell where the supervisor sees 7, the policy is still at 7, and the buffer is at 6 -- and rollback is precisely the moment most prone to crash inside such inconsistency. In `rollback()` the only thing that increments epoch is `runtime_core`; the supervisor takes a mirror afterward via `note_epoch()` and writes it into the event, which is this single-owner discipline made concrete.

For auditability, I also added a `ROLLING_BACK` state to the machine: `ACTIVE -> SAFE_STOP` alone cannot tell apart "a real fault," "a manual E-stop," "a rollback," "a watchdog," and "a controller timeout." With an explicit rollback state (and a mid-way failure that can fall back to `SAFE_STOP`), plus an event on every transition carrying a `reason`, you can afterward ask "what exactly did this machine go through last night." And the counterintuitive discipline that **the rollback path itself must be tested** -- a release plan whose rollback has never been executed is equivalent to having no rollback -- is, at the end of the article, just one line of pytest.

## Hot Reload of Config and Thresholds: Changing a Threshold Is Not Changing Code, but It's More Dangerous Than Changing Code

Deployment swaps artifacts; what operations swap is often just thresholds: velocity ceilings, validity windows, control frequency. Hot config reload is tempting because it bypasses the entire release channel; it is dangerous because **the thresholds are the safety parameters.** Three rules: the config version goes into the manifest, and changing a threshold counts as a release too (through the same state machine, no shortcuts); hot reload goes through **prepare -> validate -> commit**, and on validation failure keeps the old value and records an event -- a half-new, half-old state must never be read by the control loop; every threshold is annotated with its effective point (next tick / next episode), and a parameter switch mid-control must happen only at a boundary, like an ActionBuffer takeover. The words "small change" show up absurdly often in incident postmortems. This one, too, is turned into a runnable assertion at the end of the article with a `ConfigManager`: after an illegal threshold's commit, `current` must stay byte-for-byte unchanged.

## Patch It Up to Runnable: A Minimal Closed Loop for a Release State Machine

Everything above is design. This section turns the release channel into its minimal runnable-and-testable form -- still no torch, no GPU, reusing 9/17's fakes (injected clock, deterministic policy, epoch barrier, CommandSink single-writer all present, unchanged) and adding only a few things: the layered `Manifest` / `RuntimeVersion` (release identity and compatibility ranges), a `ReleaseSupervisor` with an evidence chain (the sole writer, every transition landing as one `ReleaseEvent` line), `Fault`s routed by nature, a `ShadowRunner` that compares on the same state across five dimensions with observable clamping, and `boot_check` / `promote_allowed` / `ConfigManager` / `rollback` (the four gates from install to retreat). First the skeleton code:

```python
# tests/deploy_fakes.py -- the minimal closed loop for deployment and ops: pure stdlib, runs directly under pytest
# Same discipline as 9/17's fakes.py: injected clock, deterministic policy, epoch barrier, single-writer state machine.
# Scope note: this file verifies only the phase constraints and evidence structure of the release state machine;
# it does not simulate a fleet-level canary scheduler (machine selection, task mix, observation-window auto-promotion are all fleet registry concerns; see the end of the article).
from dataclasses import dataclass, field
from typing import Optional, Tuple


def parse_version(s: str) -> Tuple[int, ...]:
    """Minimal int.int.int version parsing: deliberately not called semver -- it does not handle -rc1 / +build7
    prerelease metadata. Name the thing right; don't cargo-cult the spec."""
    return tuple(int(x) for x in s.split("."))


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
    artifact_hash: str           # content identity: catches same-name, different-thing
    signature: str               # provenance identity: the signature covers canonical(manifest - signature),
                                 # guarding against "hash unchanged, compat tampered" -- hash trusts content, signature trusts the publisher


@dataclass(frozen=True)
class CompatibilityContract:
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
    release_id: str                  # monotonic release sequence: the alignment axis of the event stream
    code_version: str
    artifact: ArtifactIdentity
    compat: CompatibilityContract
    policy: ReleasePolicy


@dataclass(frozen=True)
class RuntimeVersion:
    """Say clearly what this runtime version on the robot will accept."""
    code_version: str
    accepts_schema: Tuple[str, str]
    accepts_config: Tuple[str, str]
    expected_fingerprint: str


def boot_check(manifest: Manifest, runtime: RuntimeVersion) -> Optional[str]:
    """One gate shared by boot and rollback: returns a reason on failure, None on pass. Rather refuse than run with a fault."""
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
    OBSERVABILITY = "observability"    # metrics gap / ledger write failure -> freeze promotion, leave the current execution state alone


@dataclass(frozen=True)
class ReleaseEvent:
    """The smallest unit of the evidence chain: every transition carries identity, time, reason, and epoch --
    when an audit asks "which version entered which state, when, and why," this line is the answer."""
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
    Every legal transition lands as one ReleaseEvent line; illegal transitions are rejected and not recorded."""

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
    validity: bool = False                 # valid_from / chunk duration misaligned
    horizon: bool = False                  # chunk duration (horizon*dt) inconsistent
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
        return self.divergences / self.ticks if self.ticks else 0.0


class ShadowRunner:
    """Feed the same state to two policies; the candidate's output never enters the sink -- the defining test of shadow mode:
    bookkeeping only, no landing. The one path that does land goes safety.check -> sink.submit, the same authority path as production.
    Note: Python's dynamic typing does not structurally bar the candidate from the sink; this implementation relies on the
    "single call path" structure -- candidate.act()'s return value flows only to the ledger; type-level enforcement is left to the production version (see the body)."""

    def __init__(self, active, candidate, safety, sink, divergence_eps=1e-6):
        self.active = active
        self.candidate = candidate
        self.safety = safety
        self.sink = sink
        self.eps = divergence_eps
        self.stats = ShadowStats()
        self._last = None                # (active_seq, candidate_seq): baseline for the takeover-rhythm comparison

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
        self.sink.submit(self.safety.check(state, (a.values[0],)))   # only active lands
        return d.value


def promote_allowed(stats: ShadowStats, policy: ReleasePolicy) -> bool:
    """The promotion gate is data, not courage: the gate itself lives in the manifest's policy section."""
    return stats.ticks >= policy.min_ticks and stats.divergence_rate <= policy.divergence_budget


# ---------------------------------------------------------------- Config hot reload: prepare -> validate -> commit

class ConfigRejected(Exception):
    pass


class ConfigManager:
    """Thresholds are release artifacts too: changing one threshold walks the state machine; you may not bypass the release channel to write on-site parameters directly.
    If validate fails, current stays byte-for-byte unchanged plus an event is kept; the effective point is marked as the next control boundary."""

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
        c = self._staged
        if c is None:
            return "nothing_staged"
        v = parse_version(c["config_version"])
        if v > parse_version(self.runtime.accepts_config[1]):
            return "config_too_new"
        if v < parse_version(self.runtime.accepts_config[0]):
            return "config_too_old"                     # forward compatibility must be declared too, it is not assumed
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
    ⑤ flip the pointer + a full reset_episode (clear hidden state/filters/tracking), back to VERIFYING to rewalk the evidence chain
    the sole publisher of epoch is RuntimeCore: the supervisor and ledger only consume the mirror, never increment it themselves."""
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

Then come fifteen pytest, which I do not order by "which function they tested" but by the **invariant** each one pins -- so it matches 9/17's "contract + invariant" style:

| # | Invariant | Test that pins it |
| --- | --- | --- |
| I1 | An incompatible / identity-less artifact must not enter the motion path | `test_i1_boot_rejects_identity_and_drift` |
| I2 | Compatibility is a containment check, not an intersection check | `test_i2_compatibility_is_containment_not_intersection` |
| I3 | The state machine must not skip evidence phases | `test_i3_state_machine_rejects_illegal_transition` |
| I4 | After SAFE_STOP you must re-verify; no "just restart it" | `test_i4_safe_stop_forces_reverification` |
| I5 | The candidate can never get command authority | `test_i5_shadow_never_lands_candidate` |
| I6 | Promotion must satisfy the evidence budget | `test_i6_promotion_gate_requires_evidence` |
| I7 | The epoch barrier rejects stale-epoch decisions (buffer primitive) | `test_i7_action_buffer_rejects_stale_epoch` |
| I8 | After rollback runs, the old in-flight decisions are permanently disenfranchised | `test_i8_rollback_executes_and_invalidates_inflight` |
| I9 | Clamping differences are observable and countable | `test_i9_clamp_divergence_detected` |
| I10 | Value-identical but time-semantics divergence must be caught | `test_i10_temporal_divergence_without_value_divergence` |
| I11 | Faults are routed by severity, not all e-stopped | `test_i11_fault_severity_routing` |
| I12 | Every legal transition leaves complete evidence | `test_i12_events_form_evidence_chain` |
| I13 | Config hot reload is fail-before-motion | `test_i13_config_reload_is_fail_before_motion` |
| I14 | An old version != a rollback-able version: rollback must pass the gate again | `test_i14_rollback_must_reverify_old_manifest` |
| I15 | Swapping the policy does not change the runtime seam | `test_i15_swap_policy_keeps_contract_shape` |

```python
# tests/test_deploy.py -- the minimal deployment-and-ops loop: pin each invariant from the 9/18 body into a runnable assertion
# Same discipline as 9/17: injected clock, deterministic policy, epoch barrier, single-writer state machine.
# Scope: these pytest verify the phase constraints + evidence structure of the release state machine; they do not simulate a fleet-level canary scheduler.
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
    """Adds a preview to 9/17's SafetyLimiter: it only judges, never sends, so the shadow can observe clamping differences.
    check() (which lands) is inherited from the parent; the sole deploy-side landing path still goes check->sink.submit."""

    def preview(self, state, cmd):
        approved = self.check(state, cmd)
        return approved, approved != cmd


# A minimal runtime stand-in only for wiring up rollback: in a real implementation RuntimeCore itself holds these references.
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
        self.authority_open = False          # ① stop sending new commands

    def bump_epoch(self):
        self._epoch += 1                     # ② old in-flight decisions structurally disenfranchised
        self.action_buffer.sync_epoch(self._epoch)

    def install_manifest(self, manifest):
        self.active_manifest = manifest      # ⑤ flip the pointer
        self._epoch += 1                     # full reset: epoch advances once more, hidden state cleared
        self.action_buffer.sync_epoch(self._epoch)
        self.authority_open = True


# ================================================================ I1
def test_i1_boot_rejects_identity_and_drift():
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


# ================================================================ I8 (review #13: actually execute rollback)
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
    assert core.action_buffer.put(inflight, 0.0) is True     # before rollback: an old-epoch decision can be enqueued
    assert rollback(sup, core) is True
    assert sup.state == "VERIFYING"                          # stops at the start of the evidence chain, not straight back to ACTIVE
    assert core.active_manifest.release_id == "r6"
    assert core.epoch == 3                                   # 1 ->(bump) 2 ->(reset) 3
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


# ================================================================ I13 (review #11: hot reload is fail-before-motion)
def test_i13_config_reload_is_fail_before_motion():
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


# ================================================================ I15 (review #12: compare A/B directly in the same episode)
def test_i15_swap_policy_keeps_contract_shape():
    # swap in an implementation that's numerically identical but a different class: the seam and the sent stream are pinned by the contract, replay is bit-for-bit identical
    ctrl_a, safety_a = run_episode(FakeClock(), n_steps=20, policy=SinePolicy())
    ctrl_b, safety_b = run_episode(FakeClock(), n_steps=20, policy=OffsetPolicy(0.0))
    assert ctrl_a.sent == ctrl_b.sent                        # compare directly within the same episode, no extra run
    assert safety_a.safe_entries == safety_b.safe_entries
```

`python -m pytest tests/test_deploy.py -q` passes directly (run before this piece went to print: 15 passed). What this version does beyond the first draft is precisely to fill in the "said but didn't do" that a review would catch at a glance: `clamp_diffs` really counts (I9), the candidate's zero-landing is guaranteed by a single internal path in the runner rather than the test calling `submit` for it (I5), rollback tests `rollback()` itself rather than the buffer primitive (I7/I8 split apart), config hot reload now has a runnable `ConfigManager` (I13), and `release_id` enters the events and the state machine (I12). The heart of this article is I4, I8, and I14: I4 rejects "just restart it" -- after SAFE_STOP the only way out is to return to the start of the evidence chain and walk it again; I8 turns rollback into a real disenfranchisement, with the old version's in-flight decisions structurally rejected by the epoch barrier rather than by "wait a moment and let it finish"; I14 adds the most counterintuitive cut -- the rollback target version must first pass the compatibility gate, "old" does not mean "rollback-able."

## Six Anti-Patterns at the Deployment Layer

Continuing 9/17's list, here only the pits on the release channel, still ordered by frequency of appearance:

1. **Same-name, different-thing artifacts**: matching versions by filename; `v17_final_v2.pt`-style naming is the number-one breeding ground for skew. The fix: a hash for content identity, a signature for provenance identity (and the signature must cover the entire canonicalized manifest), a release_id for event identity -- do not conflate the three.
2. **Rollback that only flips config**: the pointer flipped, but the old version's in-flight chunks and async inference are still in the air. The fix: rollback = stop commands + epoch disenfranchisement + pass boot_check again + flip the pointer + a full reset -- the five-step order is the semantics.
3. **"Promote because it looks fine"**: observation without a quantified gate always slides toward luck. The fix: write `min_ticks` + `divergence_budget` into the manifest's policy section; the promotion gate recognizes only these two numbers.
4. **The rollback path never tested, the rollback target never checked for compatibility**: running the rollback script for the first time on the night of the incident, only to find after switching back that the old version is incompatible with the current runtime. The fix: put the rollback drill in the pipeline, and re-run boot_check on the old manifest before rolling back -- an old version is not a rollback-able version.
5. **Hot-changing thresholds by shortcut**: editing on-site parameters directly, bypassing the release channel. The fix: config version into the manifest, hot reload through prepare -> validate -> commit, old value stays byte-for-byte unchanged if validation fails, and changing a threshold counts as a release too.
6. **A compatibility matrix built on declarations**: "supports v3 and above" with v5 never tested, and the ✓ hand-typed into YAML. The fix: a cell that has not run a boot check defaults to ✗, every ✓ carries an evidence reference and is written by the pipeline; compatibility is a property you test, not one you declare.

## Summary

9/17 said the skeleton must "be able to swap components"; what this article adds is the layer for the swapping: **release identity** (a layered manifest -- content hash, provenance signature, event sequence, plus the identity scales of release_id and epoch), **the compatibility grid** (code × schema × config, containment rather than intersection, with untested cells defaulting to ✗ and every ✓ carrying evidence), **shadow comparison** (bookkeeping only, no landing; divergence split into the five dimensions of value/time/amplitude/sequence/safety, with clamping differences really counted), **canary promotion with gates** (data has the say, and it is made explicit that this article does not simulate a fleet scheduler), **rollback backed by the epoch barrier** (stop commands first, then disenfranchise the old decisions, pass the compatibility gate again, only then flip the pointer, and the rollback path itself must be drilled), and **fail-before-motion hot config reload**. Fifteen invariants pin this state machine to a runnable degree.

Looking back at the division of labor in this series: the evaluation protocol (9/12) defined what counts as evidence, the architecture article (9/17) defined which seam the evidence grows out of, and this article defined how evidence gates the next release. Put the three layers together, and "swap the route, the sensor, the robot" for the first time becomes a sentence with engineering meaning -- pull it out and there is a ledger, plug it back and there is a barrier, every step with its evidence.

For the next step I lean toward the second direction: **version governance for a multi-machine fleet**. Because this article has already naturally grown the basic primitives a fleet registry needs -- `release_id + compatibility matrix + evidence + epoch + rollback`. Promoting them into a registry / desired-state / reconciliation-loop (N robots × the three version axes of code/schema/config, how the registry records the evidence behind each ✓ and how it converges drift) closes the architectural line of the whole series. The first direction (expanding shadow into a full release pipeline: comparison sample mix, boundary cases, the ledger schema) is still on the candidate list. Whichever one you want, tell me in the comments.
