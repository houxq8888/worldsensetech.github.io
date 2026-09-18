---
title: 'Architecture Is the Design of Seams: An Embodied Agent Skeleton That Runs, Tests, and Swaps Components'
slug: "2026-09-19-embodied-agent-architecture"
date: 2026-09-19
draft: false
categories: ["Embodied AI", "Tutorial"]
tags: ["Embodied AI", "Software Architecture", "Robotics", "VLA", "World Model", "Python", "System Design", "Engineering Architecture", "Sim-to-Real"]
description: 'The previous articles turned "embodied AI lacks interfaces, not models" into contracts and evaluation protocols; this one lands in engineering — how the code of an embodied agent is actually organized. It gives the layering of the runtime stack and the training stack, the three core contracts StateContract / Action / Env, temporalized actions with the active/scheduled dual-slot, epoch-barrier ActionBuffer scheduling protocol, a runtime loop with three threads and dual-rate decoupling, the Temporal & Failure Contract (safety gate and command authority, failure state machine with FAULT_LATCHED, latency budget and action coverage, Supervisor), and a pure-stdlib minimal closed loop at the end that actually runs and can be tested.'
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

The previous articles pushed the claim "embodied AI lacks interfaces, not models" all the way to contract objects and evaluation protocols: multimodal fusion must deliver a [Structured State Contract](/en/articles/2026-09-14-multimodal-fusion-interface/), the policy side must consume it under a [Consumer Contract](/en/articles/2026-09-15-policy-side-interface/), and making it real takes [four kinds of compliance evidence](/en/articles/2026-09-16-policy-side-evaluation/). The theory is closed at this point, but the question in the comments keeps coming back to the same one:

**The reasoning is clear. What does the code look like?**

This article answers that question. Not framework selection — that's another article. The skeleton: how an embodied agent's code is layered, what seams sit between the modules, and how training code and deployment code share one definition.

First, the positioning: **what I'm giving you is not a demo of some model, but an engineering-oriented architectural skeleton.** The sample code deliberately compresses implementation details (schematic: imports, type definitions, and error handling omitted); the focus is the seams between components, the time semantics, and the failure boundaries. At the end, a pure-stdlib minimal closed loop runs the smallest slice of it for real. Getting that loop running is still a whole calibration, driver, and hardware layer away from production — but the shape of the seams is serious. And the boundary should be stated up front: the Protocols in this article define semantic boundaries between software components — they are not equivalent to real-time guarantees, safety certification, or driver-level assurance. Hard real-time and functional safety live in the controller and hardware layer; the software skeleton only has to not get in their way.

## Principle First: Architecture Is the Design of Seams

The most common mistake when writing an embodied agent is to start from the model and stack code downward: you begin with a big policy network, then cram the camera driver, action post-processing, and safety clamping into the training script. Three months later the code is a blob — swapping a camera touches the policy, swapping a robot touches the data pipeline, and nobody dares to refactor.

The right starting point is the reverse: **design the seams first, then fill in the implementations.** An embodied system naturally splits into two stacks:

| Stack | Responsibility | When it runs | Key constraints |
|---|---|---|---|
| Runtime stack | perception → state estimation → decision → action → control → feedback | on the robot, real-time | latency determinism, safety, degradability |
| Training stack | data collection → learning (policy / world model) → evaluation → deployment | offline / sim cluster | throughput, reproducibility, data quality |

The two stacks share the same **core data definitions** (the schema of state, action, and observation), but their execution rhythms, failure modes, and resource profiles are completely different. The first lesson of engineering architecture: **do not try to serve both stacks with one process and one codebase.** The skeleton in this article is therefore told in two halves — but their seams — `StateContract`, `Action`, `Env` — are one and the same contract definition. The whole map looks like this:

```text
┌───────────── Training Stack (offline) ─────────────┐
│ Dataset → Preprocess → Policy/WM → Eval → Artifact │
└──────────────────────┬───────────────────────┘
      Shared contracts: State / Action Schema · ObsTransform · Artifact Manifest
┌──────────────────────┴───────────────────────┐
│                Runtime Stack (real-time)             │
│ Sensor → Estimator → StateContract → Policy → Action │
│                       ↓ buffer / interpolate / hold  │
│               Safety Gate → Controller → Robot       │
└──────────────────────────────────────────────┘
```

## The Runtime Stack: Six Layers, Each Doing Exactly One Thing

Slice the chain from perception to action, and each layer's responsibility fits in one sentence. The layer you cannot state in one sentence is the mud ball in your codebase.

| Layer | Responsibility (one sentence) | Input → Output | Typical implementation |
|---|---|---|---|
| 1. Sensing | Read the physical world into raw observations | hardware timestamp → `RawObservation` | camera drivers, force/tactile drivers, proprioception |
| 2. State Estimation | Fuse multi-stream raw observations into structured state | `RawObservation` → `StateContract` | multimodal estimator (cf. the 9/14 piece) |
| 3. Policy | Decide "what to do" from state | `StateContract` → `Action` | VLA / Diffusion / MPC |
| 4. Action Interface | Buffer policy output (chunk), hold via interpolation, translate into controller commands | `Action` → `ControlCommand` | action buffer, chunk scheduling, semantic conversion |
| 5. Control | Drive the commands down to motors / joints | `ControlCommand` → motor torques | vendor SDK, ROS2 controller, force-control loop |
| 6. Feedback | Read back execution results, close the loop | proprioception → state update | shares the driver layer with Layer 1 |

Two engineering facts that are easy to overlook, pinned down here:

**First, policy frequency ≠ control frequency.** A VLA emitting decisions at 5–10 Hz is common, while a joint force-control loop must run at 200 Hz–1 kHz. Between Layer 3 and Layer 5 there must be a layer (Layer 4) doing "decision holding + interpolation + safety takeover" — otherwise one stalled policy frame leaves the robot frozen mid-air. The VLA deep dive stressed that "different frequency bases cannot be compared directly"; in engineering terms, this layer is exactly what that statement corresponds to. A structural conclusion follows: the runtime loop must be **dual-rate** — the policy thread produces chunks at its own pace, the control thread consumes them at a fixed dt, and the latter never blocks on the former. The minimal loop below is exactly this shape.

**Second, safety is not an if-branch inside the policy — it is an independent layer.** Layer 4 enforces hard constraints (velocity clamping, workspace boundaries, force limits), and above Layer 5 there must also be a watchdog: on heartbeat loss, stale state, or quality dropping below threshold, the robot enters a predefined **safe state** — note, a safe state, not specifically "hold position": an arm may hold, a drone should land, a car should brake; the concrete motion is defined by the robot's own SafetyPolicy (expanded in the "Temporal & Failure Contract" section below). Burying safety inside the policy means letting "a learning system that makes mistakes" double as the safety arbiter. The boundary must also be drawn: this layer is one independent defense on the software control chain — hardware limits, driver protection, and the independent e-stop chain are out of its reach; if the Python process dies, the hardware must still hold the robot.

## The Training Stack: Data In, Model Out, Sharing the Schema with Runtime

The training stack has a simpler shape than the runtime stack; the traps are all in the details:

| Stage | Responsibility | Key engineering questions |
|---|---|---|
| Data collection | Accumulate episodes from real robot / sim / teleoperation | unified episode format, timestamp alignment, don't drop failure cases |
| Data management | Storage, versioning, dedup, replay | data is an asset — it needs lineage and a split protocol |
| Learning | Train policy / world model | share preprocessing with deployment (details below) |
| Evaluation | Offline metrics + sim rollout + HIL | evaluation protocol in the 9/12 piece |
| Deployment | Export checkpoint + config | artifact versioning, configs in the same repo as code |

Buried here is one of the most expensive bugs in embodied engineering: **train/serve skew** — training uses one preprocessing pipeline (resize, normalize, coordinate transforms) while deployment uses another, the numbers don't match, and model performance silently degrades. The fix is to force both stacks to import the same code:

```python
# agent/state/obs_transform.py
# Both training and deployment import from this single module; separate implementations are forbidden
import numpy as np

class ObsTransform:
    """The single authoritative preprocessing for camera observations, shared by train/deploy."""

    IMG_SIZE = (224, 224)

    def __call__(self, raw_bgr: np.ndarray) -> np.ndarray:
        img = raw_bgr[..., ::-1]              # BGR -> RGB
        img = resize(img, self.IMG_SIZE)      # bilinear, aligned with the training config
        return (img / 127.5 - 1.0).astype(np.float32)  # consistent with the checkpoint
```

Even easier to miss than preprocessing is the **artifact manifest**. The filename `checkpoint_v17.pt` alone is not enough to reproduce a deployment — the same checkpoint paired with a different state schema, different normalization statistics, or a different robot calibration behaves completely differently, and train/serve skew sneaks back in through exactly this crack. Bind everything needed for reproduction into one versioned manifest, kept in the same repo and versioned together with the checkpoint:

```yaml
# deploy/artifact_manifest.yaml — the complete manifest needed to reproduce one deployment
artifact:
  checkpoint: ckpt/diffusion_v17.pt
  code_commit: 8f3a2c1           # commit of the training code
  state_schema: v3              # StateContract field-definition version
  action_schema: v2             # ActionSpace / coordinate frame / dt convention version
  obs_transform: v5             # version of the preprocessing shared by train/deploy
  normalization: stats/norm_v17.npz   # normalization statistics
  robot_model: calib/ur5e_2026-08.yaml  # robot calibration
runtime:
  container_digest: sha256:71c0...  # digest of the inference image, not a tag
  python: "3.11"
  pytorch: "2.4"
  cuda: "12.4"
compat:                        # the schema range this ckpt is compatible with; validated at startup
  state_schema: [v3, v4]
  action_schema: [v2]
```

Two easy-to-miss extensions: `runtime` pins down "which code runs in which environment" — tags are mutable, digests are not; reproducing across machines starts with matching the digest. `compat` is the firewall against schema evolution: when a policy checkpoint loads, the current schema is first validated against the compatibility range — **incompatible means fail-before-motion**, not discovering midway through a run that the fields don't line up. The traceability requirement in the evaluation protocol (9/12), when it lands in engineering, mostly looks like exactly this one manifest.

Version validation cannot stop at "are the fields still there." The compatibility criterion of a schema is **semantic compatibility**: the units changed (rad→deg), the coordinate frame changed (base→tool), the joint order changed, the normalization statistics changed, the time basis changed — not a single field is missing at the field level, yet every one of them is breaking at the behavior level. A concrete example: v3 orders the proprio tuple as `[j1, j2, j3]`, v4 as `[j2, j1, j3]` — the field names and types are exactly the same, but the per-element semantics are entirely flipped; any "field existence" check will wave this change through. So the compat range is written as semantic versions, and the load-time validation must compare units / frame / joint order / normalization fingerprint together — a mismatch is fail-before-motion, and there is no "let's run it and see" option.

The fingerprint governs "definitions are identical"; one more layer is still missing — **golden vectors**. Take a fixed set of inputs (golden input), run the shared `ObsTransform` once, and store the output together with the environment info (numpy version, resize backend, dtype conversion path) as a reference vector (golden output); re-run and compare at deployment startup and in CI — any numerical drift fails the comparison and the load is refused. The same technique applies to tokenizer round-trips, action denormalization, and coordinate transforms: even when train and serve import the same code, different numpy versions and different resize backends can still manufacture skew. At this point, train/serve skew stops being merely the organizational discipline of "don't copy the code twice" and becomes a verifiable piece of contract evidence.

## Directory Layout: Skeleton Before Flesh

Following the layering above, a maintainable repo looks like this:

```text
embodied_agent/
├── agent/
│   ├── core/            # loop, clock, lifecycle management
│   ├── perception/      # Layer 1: Sensor drivers (read → RawObservation)
│   ├── state/           # Layer 2: StateContract + StateEstimator
│   ├── policy/          # Layer 3: Protocol + implementations per route
│   ├── action/          # Layer 4: tokenizer / chunking / safety gate
│   ├── control/         # Layer 5: robot SDK adapters
│   └── safety/          # watchdog, clamping, safe poses
├── train/
│   ├── data/            # episode format, dataset, replay
│   ├── sim/             # sim backend (same RobotInterface as the real robot)
│   └── learn/           # training entry, loss, checkpoint export
├── configs/             # one yaml per robot / per task
├── tests/               # layered tests (see below)
└── deploy/              # exported artifacts, version manifest
```

Note three deliberate design choices: `agent/core` imports no concrete implementation; `configs/` lives in the same repo as the code — any config that affects model inputs, action semantics, control parameters, safety boundaries, or experiment results is treated as a versioned artifact and goes through the same review, not a runtime environment variable tweaked at will; `tests/` mirrors the `agent/` hierarchy directory by directory.

The chain of authority for configuration must also be pinned down: the yaml in the repo is the immutable source; at process startup it is parsed into one **runtime configuration** (default expansion, environment overrides, path resolution), and its hash is printed. Any component that wants to change a parameter at runtime may only modify its own in-memory copy — the source never changes — so "what parameters did this experiment actually use" can always be answered after the fact.

## Core Contracts and Seams

The skeleton has more than one seam — Sensor, StateEstimator, Controller, and Safety each have their own Protocol — but the load-bearing core contracts are three: the shape of state (StateContract), the shape of action (Action), and the shape of the environment (Env). Policy is the policy seam hanging between state and action, turning the three routes into swappable implementations. All of them are Python Protocols (structural typing / duck typing): they force no inheritance and take no implementation hostage, but they define the shape of the seams. Beyond the data contracts, the runtime has one more layer of **temporal and failure contracts** (scheduling protocol, safety gate, failure state machine) — the first three govern "what shape components exchange," the last governs "how the system winds down when components miss the beat or fail to behave," and it is unfolded separately after the testing strategy section.

### Contract 1: StateContract

This is the direct code form of the "Structured State Contract" from the 9/14 article: state is not a blob of tensors but a structured object carrying assumptions, provenance, freshness, and quality. In the anti-patterns below we criticize "bare dicts everywhere," so the core contracts themselves cannot still be string keys + objects — observation, proprioception, and quality must all be written as schema:

```python
# agent/state/contract.py
from dataclasses import dataclass
from enum import Enum

class Hypothesis(Enum):
    """A multimodal estimator may return multiple hypotheses; silently collapsing them is forbidden (cf. 9/15 Failure 1)."""
    SINGLE = "single"
    MULTI = "multi"          # requires mode_select on the policy side

@dataclass(frozen=True)
class Provenance:
    source: str              # which sensor / estimator produced it
    stamp: float             # physical-world sampling time (same clock domain as monotonic())
    frame_id: str            # coordinate frame

@dataclass(frozen=True)
class ObservationField:
    """One observation in schema-typed packaging: beyond the value, the semantic metadata is complete."""
    name: str                # "image" / "depth" / "tactile" / ...
    value: object
    dtype: str
    shape: tuple
    unit: str | None
    frame_id: str | None
    stamp: float

@dataclass(frozen=True)
class Observation:
    """Open multimodal container: adding a sensor = adding a field; the contract doesn't move."""
    fields: tuple            # tuple[ObservationField, ...]

    def get(self, name: str) -> ObservationField | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None          # an absent sensor is normal; the policy must handle it

@dataclass(frozen=True)
class ProprioState:
    """Proprioception: field names are the contract; order is not; units are also the contract.
    Convention: position=rad, velocity=rad/s, effort=Nm; changing units goes through
    a schema version, never a verbal notice."""
    joint_names: tuple       # ("shoulder_pan", "shoulder_lift", ...)
    position: tuple
    velocity: tuple
    effort: tuple | None
    stamp: float

@dataclass(frozen=True)
class StateQuality:
    """State usability, not probability. Given per source; Safety gates per task definition."""
    proprio: float
    vision: float
    localization: float
    temporal: float          # multi-sensor alignment quality (see below)

    def min(self) -> float:
        return min(self.proprio, self.vision, self.localization, self.temporal)

@dataclass(frozen=True)
class StateContract:
    """The only state object a policy is allowed to consume (cf. 9/15: the policy is a contract consumer)."""
    observation: Observation
    proprio: ProprioState
    task: object                       # instruction / goal representation (None for MPC-style routes without language input)
    hypothesis: Hypothesis
    provenance: Provenance
    validity_sec: float                # validity window of this state (expired states must be gated out)
    quality: StateQuality
    schema_version: str                # field-definition version; validated at startup against the
                                       # artifact manifest's state_schema: the object must know which version it is

    def age_sec(self, now: float) -> float:
        """Data age = now - physical sampling time. If the estimator spent 100 ms computing, the age grows by 100 ms.
        `now` is passed in explicitly by the caller: the contract does not read the wall clock internally (see the Clock Protocol below)."""
        return now - self.provenance.stamp

    def is_fresh(self, now: float) -> bool:
        return self.age_sec(now) < self.validity_sec
```

`frozen=True` is not neatness fetish: once a state object is constructed it must not be rewritten downstream — otherwise "who modified the state at which stage" becomes a bug source you cannot localize. But the semantic boundary of `frozen` must be pinned down precisely: what it guarantees is **that field references cannot be rebound**, not deep immutability — if an `ndarray` sits inside `ObservationField.value`, `field.value[0] = 123` still takes effect. True downstream immutability has to be supplemented with read-only views, ownership rules, or copy-on-write; miss this, and `frozen=True` merely gives the illusion of safety.

Also part of the minimal skeleton is the single number `validity_sec`: it is enough for a closed loop, but in a multi-modal runtime the legitimate age of vision at 30 Hz, IMU at 500 Hz, and joints at 1 kHz is naturally different (tens of milliseconds versus a few), and production systems usually need per-field freshness or a modality-specific validity policy — otherwise one blanket `is_fresh(now)` gets misread as "the whole state is fresh" when in reality only the proprio stream still is.

**Time semantics are the easiest thing to get wrong here.** Freshness must be computed from the **physical sampling time**, not the object creation time — otherwise, if the estimator spent 100 ms computing, `is_fresh` would consider the state "fresh off the press" when it actually describes the world as it was 100 ms ago. What matters for a robot is data age, not the age of a software object. The precondition is that all timestamps live in the same clock domain as `monotonic()`; if sensor timestamps come from ROS / the system wall clock, align the clock domains before comparing.

From this follows one discipline: **the clock is a dependency, not a global variable.** Neither the contract, the buffer, nor the scheduling layer may read `time.monotonic()` on its own — every `now` flows in from a single injected Clock. Production injects `SystemClock`, tests inject `FakeClock`; without an injected clock, deterministic replay is off the table:

```python
# agent/core/clock.py
from typing import Protocol
import time

class Clock(Protocol):
    """The single source of time for the whole system. Every spot in business
    code that scatter-reads time.monotonic() is a time crack that tests cannot freeze."""
    def monotonic(self) -> float: ...

class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()
```

**Multi-sensor synchronization is the other hard half.** A camera at 30 Hz, an IMU at 500 Hz, joints at 1 kHz — their timestamps are naturally different — and a StateEstimator that actually does the work must state: the alignment window, the interpolation/extrapolation policy, and the timeout-drop rule. That is why `RawObservation` makes the timestamp a first-class citizen, and the estimator reports the quality of its output via `quality.temporal`:

```python
@dataclass(frozen=True)
class RawObservation:
    """Raw reading from a single sensor."""
    sensor_id: str
    sensor_timestamp: float    # hardware sampling time (note which clock domain it belongs to)
    receive_timestamp: float   # when it entered the software
    sequence_id: int
    clock_domain: str          # "monotonic" / "ros_wall" / ...
    calibration_version: str
    payload: object
```

Multiple clock domains must become an explicit pipeline, not an implicit convention buried inside the estimator — without normalization, timestamps from two domains are simply not comparable, and `data age` cannot even be discussed:

```text
Raw sensor timestamps (each in its own clock domain)
    ↓ clock normalization: converted into the monotonic domain, conversion version recorded
Sample buffer with unified timestamps
    ↓ temporal alignment: window-based alignment, interpolation/extrapolation, timeout drop
StateContract (provenance.stamp already in the monotonic domain)
```

Another seam that is easy to blur together is "read" versus "estimate." Layers 1 and 2 should be two separate Protocols, not two methods on one perception object:

```python
# agent/perception/sensor_buffer.py, agent/state/estimator.py
@dataclass(frozen=True)
class SensorSnapshot:
    """The set of latest samples per source at time now: the input to the estimator's
    temporal alignment, not a single-sensor read."""
    samples: tuple          # tuple[RawObservation, ...], at most one per source
    window_sec: float       # alignment window: samples older than now - window must not join

class SensorBuffer(Protocol):
    """Layer 1: the driver thread keeps writing timestamped samples into a ring buffer;
    consumers take the latest non-blockingly. The production form is never a blocking
    read() — a runtime thread has no business blocking on a sensor."""
    def latest(self, now: float) -> SensorSnapshot: ...

class StateEstimator(Protocol):
    """Layer 2: the only place in the system allowed to produce a StateContract.
    Responsibilities: clock-domain normalization, temporal alignment, interpolation/
    extrapolation, drop policy, max sync window."""
    def estimate(self, snapshot: SensorSnapshot, now: float) -> StateContract: ...
```

### Contract 2: Action

A VLA spits out joint deltas, MPC spits out target poses, Diffusion spits out action chunks — these outputs differ in **semantics, units, coordinate frame, time interval, and execution duration**. For a seam to genuinely carry these differences, Action must carry its own metadata; otherwise "the three routes all emit the same Action" is only type-level unity, not a real contract.

More missing than metadata is the **time basis**. Suppose the state is sampled at t=10.000, and the VLA — 150 ms of inference later — emits a chunk with dt=0.02 and horizon=10. Which world-time span does it correspond to: `[10.150, 10.350]`, or "execute for 200 ms starting now," or "the 200 ms of future after state 10.000"? These three are not the same thing, and since the policy runs on an async thread, a vague time basis makes downstream buffering impossible to define strictly. So Action must answer five questions: **which world-time moment this decision is for, when it becomes valid, when it expires, which episode (epoch) it belongs to, and its serial number within that episode:**

```python
# agent/action/contract.py
class ActionSpace(Enum):
    JOINT_POS = "joint_pos"
    JOINT_DELTA = "joint_delta"
    EE_POSE_DELTA = "ee_pose_delta"   # end-effector pose delta
    # ...

@dataclass(frozen=True)
class ActionSchema:
    """Spatial semantics and units: space defines semantics, representation defines
    the encoding; units and coordinate frames are contract."""
    space: ActionSpace
    representation: str         # encoding convention: joint_rad / ee_se3_log / ee_6d_rot / ...
    translation_unit: str|None  # translation unit (None for joint actions); mandatory for EE actions
    rotation_unit: str|None     # rotation representation: se3_log / quat / euler_xyz / 6d ...
    frame_id: str               # coordinate frame: joint / base / ee ...

@dataclass(frozen=True)
class Action:
    """Policy output is not a bare blob of tensor: semantics, frame, and time basis are all written into the object."""
    values: object        # action values: a chunk (H, D) or a single step (D,)
    schema: ActionSchema  # spatial semantics, encoding, units, frame
    schema_version: str   # schema fingerprint: consumers fail-before-motion based on it
    dt: float             # time interval between adjacent action points
    horizon: int          # chunk length; 1 for a single-step action

    generated_at: float   # when inference finished and the chunk was handed to Layer 4
    valid_from: float     # validity start
    valid_until: float    # hard expiry: past this, stale and must not be consumed
    state_stamp: float    # physical sampling time of the state this decision is based on
    epoch: int            # lifecycle identity: which episode / runtime epoch it belongs to
    sequence_id: int      # decision serial within the epoch: async commits must not arrive out of order (see the scheduling protocol)
```

The same `EE_POSE_DELTA` with rotation expressed as an se(3) logarithm, a quaternion, or a 6D rotation matrix has completely different value shapes and interpolation semantics — that is exactly why the `representation` field exists.

With these five temporal fields plus the epoch identity, "preemption timing, expiry drop, decision traceability" all gain a basis: `state_stamp` pins the decision back to the world it saw, `valid_until` gives the consumer a hard deadline, `sequence_id` orders async commits within the epoch, and `epoch` permanently strips late-arriving old decisions of their right to override after a reset — the last decision of episode 41, however late it arrives, cannot enter episode 42's buffer.

The five fields also imply an invariant worth stating explicitly: **`state_stamp <= generated_at <= valid_from < valid_until`**. It splits apart two latencies that are constantly conflated — decision latency `generated_at - state_stamp` (the time from the policy seeing the world to committing the decision; the VLA's 150 ms is spent here) and scheduling lead time `valid_from - generated_at` (the buffer between committing the decision and being allowed to take effect, reserved for transport and queuing). Beyond the invariant there is the handling protocol for late decisions: when a new chunk arrives with `valid_from` already passed but `valid_until` not yet reached, do you truncate the head and immediately execute the remainder, drop the whole chunk, or re-lay it ASAP from the current moment? The three choices correspond to three action-continuity semantics; they must be declared explicitly in the scheduling protocol, not left for the implementation to improvise.

Another convention that papers take for granted and implementations must pin down: **a chunk is a zero-order hold, not point samples**. `values[i]` applies to the world-time interval `[valid_from + i*dt, valid_from + (i+1)*dt)` — left-closed, right-open. So a chunk with `dt=0.02, horizon=16` actually covers `[valid_from, valid_from + 0.32)`; the control loop reading point 15 at the tail of the interval and executing it through to the end is the natural consequence of this semantics, not an off-by-one. Misreading a chunk as "target values at a sequence of instants" is the most common implementation bug of this protocol. The sampling index is `int((now - valid_from) / dt)` — floor, not round: at t=0.011 with dt=0.02, round would mis-assign the 0.55 beat to point 1, while under the left-closed right-open definition it still belongs to point 0.

Beyond the time semantics, a **trajectory continuity** contract sits at chunk handoffs: A(t_end⁻) ≈ B(t_start⁺) — the start of the new chunk must connect to the end of the old chunk in position, and ideally in velocity and acceleration too. This contract is out of the ActionBuffer's reach: **the ActionBuffer is responsible for time selection** (which point of which beat takes effect); **trajectory feasibility / continuity belongs to the Safety and Controller contract**. Equally worth pinning down is the semantic boundary of rate-limiting: it only guarantees `Δq ≤ vmax·dt` per tick (a position-increment ceiling) — it does not generally guarantee a velocity ceiling, let alone acceleration and jerk. Calling "rate limiting" a "trajectory-continuity guarantee" is the most common confusion of these two layers' responsibilities.

### Seam: Policy

```python
# agent/policy/base.py
from typing import Protocol

class Policy(Protocol):
    """The seam of Layer 3. VLA / Diffusion / MPC are merely its implementations."""

    def reset(self, epoch: int, alloc: "SequenceAllocator") -> None:
        """Episode boundary: clear KV cache / hidden states / planning cache,
        and collect this epoch's identity and decision serial — (epoch, sequence_id) is the
        decision's true identity. A global next_seq()-style process-level mutable state
        pollutes replay and reset, and is forbidden."""
        ...

    def act(self, state: StateContract, now: float) -> Action:
        """Given a state and a world-time moment, emit an action (usually an action chunk). `now` is passed in explicitly by the Clock."""
        ...
```

The three implementations are each short — short because the skeleton pushes all the dirty work beyond "eat state, spit action" to other layers. They are all schematic, with imports, error handling, and their respective inference details omitted — read them for the seams only:

```python
# agent/policy/vla.py (illustrative; framework details omitted)
class VLAPolicy:
    def __init__(self, ckpt, action_tokenizer, device="cuda"):
        self.model = load_vla(ckpt).to(device).eval()
        self.tok = action_tokenizer
        self._past = None

    def reset(self, epoch=0, alloc=None):
        self._past = None     # the KV cache belongs to the epoch: cleared together with reset

    @torch.no_grad()
    def act(self, state: StateContract, now: float) -> Action:
        image = state.observation.get("image")   # -> ObservationField | None; absence must degrade gracefully
        ids = self.model.generate(
            image=None if image is None else image.value,
            instruction=state.task,
            past=self._past,
        )
        # schematic: whether the KV cache can be reused across frames depends on the model architecture
        # (whether visual tokens are replaced each frame, positional encoding, causal attention structure) — never assume it
        self._past = ids.kv_cache
        return self.tok.decode(ids.action_bins)  # -> Action(chunk); Layer 4 releases it on schedule
```

```python
# agent/policy/mpc.py (world model in the loop, echoing the "hybrid architecture" trend)
class MPCPolicy:
    """Learns no explicit policy; instead rolls out with the world model to pick actions — the runtime form of the TD-MPC route."""

    def __init__(self, world_model, candidate_sampler, horizon=5, n_candidates=32):
        self.wm = world_model
        self.sampler = candidate_sampler
        self.horizon = horizon
        self.n = n_candidates

    def reset(self, epoch=0, alloc=None):
        pass  # no cross-frame state: re-encode from the current observation every frame (stateless re-plan)

    def act(self, state: StateContract, now: float) -> Action:
        z = self.wm.encode(state)
        cands = self.sampler(z, n=self.n, horizon=self.horizon)
        # schematic: compressing dynamics rollout / cost / constraint / uncertainty
        # into estimate_return() only to highlight the Policy seam; a real MPC must not hide
        # those semantics inside an unconstrained scalar scorer
        scores = [self.wm.estimate_return(z, c) for c in cands]
        best = cands[int(np.argmax(scores))]
        u0 = best[0]  # execute only the first step, replan next frame (receding horizon)
        return Action(values=u0, horizon=1, ...)   # Policy.act returns an Action, not a bare control vector
```

```python
# agent/policy/diffusion.py (illustrative: timestep / noise schedule / CFG and other sampling details omitted)
class DiffusionPolicy:
    """Generative policy: denoise an action chunk over multiple steps, naturally multi-hypothesis (cf. 9/15 Failure 1)."""

    def __init__(self, denoiser, n_steps=10, clock=None):
        self.den = denoiser
        self.n_steps = n_steps
        self.clock = clock if clock is not None else SystemClock()   # Clock Protocol

    def reset(self):
        pass  # stateless, or clear the conditioning queue depending on implementation

    def act(self, state: StateContract) -> Action:
        x = torch.randn(1, CHUNK_LEN, ACTION_DIM)   # denoising starts from noise; the endpoint is not
        image = state.observation.get("image")
        for _ in range(self.n_steps):
            x = self.den(x, None if image is None else image.value, state.task)
        values = self.denormalize(x[0])   # model output -> denormalization -> action values
        now = self.clock.monotonic()      # the single source of time: no scatter-reads of time.monotonic()
        return Action(values=values,
                      schema=ActionSchema(space=ActionSpace.JOINT_POS,
                                          representation="joint_rad",
                                          translation_unit=None,
                                          rotation_unit=None,   # joint space has no rotation unit: that belongs to SE(3) actions
                                          frame_id="joint"),
                      schema_version="joint_pos@v2",
                      dt=0.02, horizon=CHUNK_LEN,
                      generated_at=now, valid_from=now,
                      valid_until=now + CHUNK_LEN * 0.02,
                      state_stamp=state.provenance.stamp,
                      sequence_id=next_seq())  # the chunk goes to Layer 4 for timed release
```

Every link of this chain must be auditable: **model output → denormalization → action-schema conversion → limits/projection → ActionBuffer → Safety**. The statistics used for denormalization must be the exact same ones used in training — train/serve consistency of normalization is gated on this chain, and if any link swaps its implementation, skew comes back through it. A side lesson in schema design: `ActionSchema` crams joint actions and SE(3) actions into the same pair of translation/rotation unit fields, so a joint action can only fill both with `None` as a fallback — as it evolves, it should be split into `JointActionSchema` / `CartesianActionSchema` (or `units: tuple[str, ...]`); don't make the type system paper over holes by convention.

Put the three implementations side by side and the 9/15 grid of "conditioning representation × action head" lands on the ground: they eat the same state and spit the same action; the only differences are "how the state enters the network" and "how the action is generated." **The route debate is demoted, at the skeleton level, to a swappable implementation** — that is exactly the value of interface design.

One theoretical qualification: in the software abstraction of this article, any module that takes the current state and produces the next control decision is uniformly treated as a policy-like decision module — so MPC shares the same slot as learned policies at the interface level. But that does not make them the same kind of object in control theory. MPC is more precisely an online receding-horizon decision-maker; "plugging it into the Policy slot" describes the engineering seam, not theoretical equivalence.

### The Layer-4 Scheduling Protocol (ActionBuffer)

The Action's temporal fields hand the hardest question to Layer 4: **when a new chunk arrives, what exactly happens?** Chunk A covers `[0.00, 0.20]`; chunk B (with a larger sequence_id) arrives at `t=0.10` and covers `[0.10, 0.30]` — does A yield, does A finish before B takes over, is it immediate preemption, or a mix? For a robot this is an **action-continuity** problem, not an ordinary caching problem. The answer must be written down as an explicit scheduling protocol:

```text
ActionBuffer semantics (active + scheduled dual slots, epoch barrier):
- one serial space per epoch: put() only accepts chunks of the current epoch with sequence_id > last_accepted;
  old-epoch / out-of-order late arrivals are rejected outright — after a reset, a late old decision permanently loses the right to override
- stale chunk (valid_until passed): drop — never carry an old decision forward
- future chunk (valid_from in the future): park in the scheduled slot; take over at the control boundary when valid_from arrives;
  in the meantime only a chunk with a larger sequence_id may replace it
- active chunk: a later arrival with a larger sequence_id and an already-reached valid_from takes over the active slot immediately
- takeover happens only at control boundaries — never switch mid-tick;
  while A is executing and B has not reached its effective point, keep executing A — no execution gap is allowed
- missing chunk (no decision available): enter the safe state, not silent reuse
- action discontinuity (A→B jump too large): project / rate-limit before release
```

There is one more rule, and it is the discipline of async systems: **policy output must not be committed out of order.** Swap in an async inference service and request #42 may finish before #41 — if the buffer accepts everything, a late old decision will overwrite the new one. So `put(action, now)` first applies the epoch-barrier check (anything with `action.epoch != current_epoch` is rejected outright), then only accepts chunks with `sequence_id > last_accepted_sequence`; everything else is dropped on arrival. This also answers an important boundary of "swapping components": any implementation that plugs into the Policy slot must obey the same commit discipline.

This layer must be thin — thin enough to contain no hidden policy: **no cross-chunk fusion by default**; an A→B jump is handed over as-is, and continuity is covered by the SafetyGate's project / rate-limit. Replanning is allowed at any time — if B arrives at `t=0.10` with an effective point of `t=0.20`, it takes over at the `t=0.20` control boundary, the remainder of A is void, and there is no queue semantics of "finish A first"; if B's effective point has already passed, it takes over the active slot immediately. The name says buffer, but the behavior is an **active + scheduled dual-slot register**: the FIFO intuitions (queuing, overflow, drop-oldest) all fail to hold at this layer — the barrier check and the overwrite in `put()` are the entire semantics.

### Contract 3: Env — Split into "Robot Interface" and "Task Environment"

First, be clear about what it can and cannot do. The unified interface kills **interface differences that humans manufactured** — two codebases, two sets of field names, two sets of coordinate conventions. It cannot kill the **sim-to-real gap itself** — dynamics mismatch, actuator latency, friction, sensor noise, calibration error; those are addressed by domain randomization, latency/noise modeling, system identification, and validation on the real robot. Treating interface unification as the cure for the gap is the most common overestimation in architectures of this kind.

But "unified interface" itself hides a common design error: the `reset() -> RawObservation` / `step() -> (RawObservation, reward, done, info)` trio is natural for an RL simulator, yet it crams **three roles** into one interface — the robot interface, the RL environment, and task evaluation. A real robot does not naturally own `reward` and `done`: those are task definitions, not hardware properties. Split them apart, and the sim-to-real boundary actually gets cleaner:

```python
# agent/control/robot.py, train/sim/task_env.py
class RobotInterface(Protocol):
    """Real robot and sim backends implement the same interface: only observe, send, reset."""
    def observe(self) -> RawObservation: ...
    def send(self, cmd: ControlCommand) -> None: ...
    def reset_robot(self) -> None: ...

class TaskEnv(Protocol):
    """Task definition: strips reward / done / success criteria out of the hardware."""
    def reset(self) -> StateContract: ...
    def step(self, action: Action) -> Transition: ...   # only Transition carries reward/done
```

```python
class SimBackend:
    """MuJoCo / Isaac adapter implementing RobotInterface. Proprio field names strictly identical to the real-robot driver."""
    ...

class RealRobot:
    """Real-robot adapter implementing RobotInterface. All blocking IO moves to background threads; observe() only takes the latest."""
    ...

class PickPlaceTask:
    """One kind of TaskEnv: wraps a RobotInterface or SimBackend and is responsible for scoring reward."""
    ...
```

In training, `TaskEnv` wraps a `SimBackend`; in deployment there is no `TaskEnv` and no reward — the policy consumes the `StateContract` directly. The two worlds align through the single seam of `RobotInterface`, and all that sim-to-real work has left to manage is this one implementation difference.

Also, `send() -> None` is a schematic minimal synchronous interface: a `send()` call neither means the controller accepted the command nor that the actuators executed it — production adapters usually also need command acknowledgement, controller mode, health and fault status, modeling submitted / accepted / executing / rejected separately. This minimality is intentional: align the seam first, and fill in the reliability semantics robot by robot.

## The Minimal Runtime Loop: Three Threads, Dual-Rate Decoupling

Put the pieces together. The runtime is **not a single loop**: the estimator thread keeps publishing state at its own high rate, the policy thread produces chunks at 5–20 Hz, and the control thread consumes them at 200–1000 Hz — the two ends are decoupled by two registers: state goes into the StateBuffer, action into the ActionBuffer. A 150 ms VLA inference only slows down the policy thread; the control loop doesn't feel it:

```python
# agent/core/loop.py (schematic)
CONTROL_DT = 1.0 / 200     # command update cadence 200 Hz; the policy thread runs at the model's own pace
POLICY_DT = 1.0 / 20       # upper cadence of policy production: slowness is the model's call, overshooting is not allowed
MIN_QUALITY = 0.5          # state-usability floor: below this, no new decision is produced

class AgentLoop:
    def __init__(self, sensor, estimator, policy, state_buffer,
                 action_buffer, safety, sink, clock):
        self.sensor = sensor                # Sensor Protocol (Layer 1)
        self.estimator = estimator          # StateEstimator Protocol (Layer 2)
        self.policy = policy                # Policy Protocol (Layer 3)
        self.state_buffer = state_buffer    # Layer 2 exit: state register
        self.action_buffer = action_buffer  # Layer 4: chunk register + scheduling protocol
        self.safety = safety                # independent safety layer (SafetyGate)
        self.sink = sink                    # the sole exit: ApprovedCommandSink
        self.clock = clock                  # Clock Protocol: no scatter-reads of time.monotonic()

    def _estimator_thread(self):
        while self._running:
            now = self.clock.monotonic()
            snapshot = self.sensor.latest(now)   # non-blocking: takes the latest sample set from each source
            state = self.estimator.estimate(snapshot, now)
            self.state_buffer.publish(state)     # sole writer; consumers each take the latest-and-fresh

    def _policy_thread(self):
        while self._running:
            now = self.clock.monotonic()
            state = self.state_buffer.latest_valid(now)  # decoupled from data production
            if state is None or state.quality.min() < MIN_QUALITY:
                self.safety.enter_safe_state("stale_state")   # stale data / insufficient quality
                self._sleep_until(now + POLICY_DT)   # the error path keeps the cadence too — no busy-waiting
                continue
            action = self.policy.act(state, now)   # 100 ms+ is fine — it carries no control deadline
            self.action_buffer.put(action, now)    # the chunk goes to Layer 4 for release per the scheduling protocol

    def _control_thread(self):
        next_tick = self.clock.monotonic()
        while self._running:
            now = self.clock.monotonic()
            state = self.state_buffer.latest_valid(now)   # the control side independently re-judges freshness
            if state is None:
                self.safety.enter_safe_state("stale_state")
            else:
                cmd = self.action_buffer.current(now)  # returns None when no decision is available (protocol-defined)
                if cmd is None:
                    self.safety.enter_safe_state("no_action")
                else:
                    decision = self.safety.evaluate(state=state, action=cmd,
                                                    context={"now": now})  # production form; see the next section
                    if decision.allow:
                        self.sink.submit(decision.command)  # the sole exit: only approved commands enter the RobotInterface
                    else:
                        self.safety.enter_safe_state(decision.reason)
            next_tick += CONTROL_DT      # cadence advance is decoupled from per-tick cost — no drift
            if now > next_tick:
                self.safety.record_deadline_miss(now - next_tick)  # breach bookkeeping, handled per miss policy
                next_tick = now          # realign: skip tick / degraded / safe stop
            self._sleep_until(next_tick)

    def _sleep_until(self, deadline):
        # align to the cadence with a monotonic clock; on timeout, record a jitter metric instead of swallowing it silently
        ...
```

A few deliberate designs: `safety.evaluate` sits after the action buffer and before the controller — no branch of the software path can get around it — but it is **one independent defense on the software control chain**, not the final defense of the whole system: hardware limits, driver protection, and the independent e-stop chain must exist independently. Two upgrades over the first draft are worth spelling out: the state threshold went from a single `confidence` scalar to `quality.min()`, because folding "vision 0.9, localization 0.2" into one scalar is itself a semantic cheat; and `hold_position()` was replaced by `enter_safe_state(reason)`, because hold is only the arm's safe state — a drone's is land, a car's is brake — and the motion is defined by the robot's own SafetyPolicy. Cadence alignment is its own function, and a timeout becomes a metric — latency problems in embodied systems usually show up first as "growing jitter," and turning that into an observable signal beats catching it in a postmortem stack trace.

Two more seam upgrades deserve a name-check. State no longer passes through a cross-thread private cache like `self._last_state`: the estimator thread is the StateBuffer's sole writer, and the policy side and the control side each call `latest_valid(now)` to judge freshness independently — the policy-side stale gate and the control-side stale gate keep their own books, and neither can decide for the other. Command authority also has exactly one path: a policy's `Action` is only a proposal, and **only a Safety-approved ControlCommand may enter the RobotInterface** — the normal authority chain is `SafetyGate → CommandSink` (see the safety gate section); the e-stop belongs to another, independent hardware authority that does not pass through any layer of the software chain.

Two cadence disciplines deserve to be named separately, because both are small pits of runtime correctness: the error path must not busy-wait — before `continue` it must likewise sleep until the next tick point, otherwise the moment a sensor dies, the policy thread hammers the CPU in a `latest_valid → safe → continue` loop; and cadence advance uses `next_tick += CONTROL_DT` instead of re-reading `t0 + CONTROL_DT` each round — with the latter, once one tick overruns, the sleep baseline is pushed back and the drift rolls into every later tick. A timeout must not be swallowed silently: record the deadline miss and handle it per the miss policy (skip tick / degraded / safe stop) — this is the same thing as the statement "the control deadline is a hard contract."

Finally, pin down the real-time boundary: the 200–1000 Hz in this article refers to the **update rate of the control interface**; the Python layer carries 200 Hz-level command scheduling and supervision; the true hard-real-time servo loop (1 kHz and above) is carried by the RT controller, RTOS, drivers, or firmware — **the Python runtime does not act as a hard-real-time safety loop**.

## Making It Run: A Minimal Pure-stdlib Closed Loop

Everything above is schematic. This section patches the seams up to something that actually runs and actually tests: no torch, no GPU — one set of fake implementations plus a RuntimeCore and nine pytest tests; `python -m pytest tests/test_loop.py -q` passes directly (verified before publication: 9 passed). Production and tests share the same loop body — `RuntimeCore` exposes only three entry points, `prime(now)`, `policy_tick(now)`, and `control_tick(now)`: in production, three threads drive them at their own paces; in the minimal form, the estimator is folded into the control tick, publishing into the StateBuffer before consuming each tick, with the seams unchanged; in tests, a FakeScheduler drives them tick by tick with an injected clock — without clock injection, deterministic replay is off the table:

```python
# tests/fakes.py — fake implementations in pure stdlib; pytest runs them directly
# contract fields identical in shape to agent/state/contract.py and agent/action/contract.py;
# to stay minimal, fields such as quality / SafeState are omitted — only the seams this test group pins down are kept.
from dataclasses import dataclass
from typing import Optional, Tuple
import math

CHUNK_LEN = 16
DT = 0.02


@dataclass(frozen=True)
class Provenance:
    source: str
    stamp: float                 # physical sampling time (same clock domain as the injected clock)
    frame_id: str


@dataclass(frozen=True)
class StateContract:
    proprio: Tuple[float, ...]
    provenance: Provenance
    validity_sec: float

    def is_fresh(self, now: float) -> bool:
        return (now - self.provenance.stamp) < self.validity_sec


class SequenceAllocator:
    """Decision serial allocator: one per runtime epoch, monotonic within the epoch, zeroed on reset.
    A global next_seq() pollutes replay and reset — (epoch, sequence_id) is the decision's true identity."""

    def __init__(self):
        self._next = 0

    def next(self) -> int:
        self._next += 1
        return self._next


@dataclass(frozen=True)
class Action:
    values: Tuple[float, ...]    # action chunk (CHUNK_LEN,)
    dt: float
    horizon: int
    generated_at: float          # when inference finished and the chunk was handed to Layer 4
    valid_from: float
    valid_until: float           # hard expiry: past this, the chunk is stale and must not be consumed
    state_stamp: float           # physical sampling time of the state this decision is based on
    epoch: int                   # lifecycle identity: which episode / runtime epoch it belongs to
    sequence_id: int             # decision serial: monotonic within the epoch; async commits must not arrive out of order


class FakeClock:
    """Clock injected for tests (production counterpart SystemClock; both implement the same Clock Protocol)."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeSensor:
    """Minimal form of the Layer-1 buffer: latest(now) takes only the newest sample and never blocks the caller."""

    def __init__(self, clock):
        self.clock = clock

    def latest(self, now):
        t = now
        return {"angle": math.sin(t), "stamp": t, "frame_id": "joint"}


class DelayedSensor(FakeSensor):
    """Simulates a sensor delayed by 0.2 s: data age exceeding validity should trigger the safe state."""

    def latest(self, now):
        raw = super().latest(now)
        return {**raw, "stamp": raw["stamp"] - 0.2}


class FailingSensor(FakeSensor):
    """Simulates a sensor that dies at fail_at: the stamp freezes on the last frame before death, and data age keeps growing."""

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
    """Minimal form of the cross-thread seam: the estimator publishes the newest state, and the policy / control sides each take the 'newest and still fresh' one.

    Each consumer judges freshness on its own: the policy-side gate only guarantees 'the state entering the decision is fresh'; the control side must re-judge — the state a decision was based on may already be expired by the time it executes.
    """

    def __init__(self):
        self._state = None

    def publish(self, state):
        self._state = state

    def latest_valid(self, now):
        s = self._state
        if s is None or not s.is_fresh(now):
            return None        # expired/missing: hand it to Safety to enter the safe state — never silently reuse
        return s


class SinePolicy:
    """deterministic policy: the output depends only on the state, so episode replay is bit-for-bit identical."""

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
    """Inference latency injection: the decision sees the state at the moment it is initiated, and is committed only latency seconds later —
    the gap before it takes effect is exactly where the 'action coverage' gap comes from."""

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
    """Minimal Layer 4: active + scheduled dual slots + epoch barrier.
    The precise semantics of latest-wins: a future chunk is parked until its takeover point — never creating an execution gap."""

    def __init__(self, epoch=1):
        self._epoch = epoch
        self._active: Optional[Action] = None
        self._scheduled: Optional[Action] = None
        self._last_accepted_sequence = 0
        self._active_sequence = 0
        self.events = []         # evidence hook: accepted/activated/preempted/expired/rejected_*

    def sync_epoch(self, epoch):
        """The barrier of reset_episode: decisions of the old epoch are voided immediately, and the serial restarts."""
        self._epoch = epoch
        self._active = None
        self._scheduled = None
        self._last_accepted_sequence = 0
        self._active_sequence = 0
        self.events.clear()

    def put(self, action, now) -> bool:
        """Only accepts decisions of the current epoch that are newer; old-epoch / out-of-order late arrivals are rejected outright."""
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
            self._active = action            # takes effect immediately: latest-wins
            self._active_sequence = action.sequence_id
            self._scheduled = None
            self.events.append(("activated", action.sequence_id))
        else:
            self._scheduled = action         # future: parked, takes over at the control boundary when valid_from arrives
        return True

    def current(self, now: float) -> Optional[float]:
        """Index the chunk point by time; the scheduled one is promoted at its takeover point, expired ones are voided — never reused."""
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
            self._active = None              # expired decision voided, not reused
            return None
        idx = min(int((now - a.valid_from) / a.dt), a.horizon - 1)   # ZOH: floor, not round
        return a.values[idx]


class SafetyLimiter:
    """Minimal safety layer: a velocity ceiling (a physical quantity, rad/s) + safe-state entry logging.
    The limiting is based on the measured position, not the previous command: sending 1.0 does not mean the robot reached 1.0."""

    def __init__(self, max_velocity=5.0, dt=DT):
        self.max_velocity = max_velocity     # what is limited is velocity, not position
        self._dt = dt
        self.safe_entries = []

    def check(self, state, cmd):
        pos, = cmd
        measured, = state.proprio            # whether the velocity is legal must be judged against the measured state, not the last command
        max_delta = self.max_velocity * self._dt   # velocity ceiling converted into a per-tick position increment
        delta = max(-max_delta, min(max_delta, pos - measured))
        return (measured + delta,)           # a legal position ≠ a legal velocity

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
    """Test version of the CommandSink: the only object allowed to call controller.send().
    RuntimeCore only gets the sink — writing to the controller while bypassing the SafetyLimiter is structurally impossible,
    the same authority constraint as in production (see the safety gate section)."""

    def __init__(self, controller):
        self._controller = controller

    def submit(self, cmd):
        self._controller.send(cmd)


class RuntimeCore:
    """The same loop body as production: the two ticks are driven by a scheduler, not by threads.
    The epoch is owned by the core: reset_episode increments it and syncs it to the policy and the ActionBuffer."""

    def __init__(self, sensor, estimator, policy, action_buffer, safety, sink,
                 state_buffer):
        self.sensor = sensor
        self.estimator = estimator
        self.policy = policy
        self.action_buffer = action_buffer
        self.safety = safety
        self.sink = sink   # the sole exit: approved commands reach the controller through the CommandSink
        self.state_buffer = state_buffer
        self.epoch = 0

    def reset_episode(self):
        self.epoch += 1
        self.policy.reset(self.epoch, SequenceAllocator())
        self.safety.reset()
        self.action_buffer.sync_epoch(self.epoch)
        # production also resets: estimator filters, StateBuffer, controller tracking, seeds

    def prime(self, now):
        """Fill the state buffer at the episode start so the policy has state to consume on its first tick."""
        state = self.estimator.estimate(self.sensor.latest(now), now)
        if state.is_fresh(now):
            self.state_buffer.publish(state)

    def policy_tick(self, now):
        state = self.state_buffer.latest_valid(now)
        if state is None:
            self.safety.enter_safe_state("stale_state")  # policy-side freshness gate
            return
        self.action_buffer.put(self.policy.act(state, now), now)

    def control_tick(self, now):
        # in production the estimator runs in its own high-frequency loop publishing to the StateBuffer;
        # the minimal loop folds it into the control tick, and the two judge latest_valid by exactly the same criterion.
        state = self.estimator.estimate(self.sensor.latest(now), now)
        if state.is_fresh(now):
            self.state_buffer.publish(state)
        state = self.state_buffer.latest_valid(now)
        if state is None:
            self.safety.enter_safe_state("stale_state")  # control-side freshness gate, independent of the policy side
            return
        point = self.action_buffer.current(now)
        if point is None:
            self.safety.enter_safe_state("no_action")   # missing: no silent reuse
            return
        self.sink.submit(self.safety.check(state, (point,)))   # the sole exit: approved command goes through the CommandSink


def run_episode(clock, n_steps, hz=50, sensor=None, policy=None, policy_every=5):
    """FakeScheduler: the policy cadence is approximated with policy_every ticks, clock injected."""
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

The nine tests pin down nine things respectively: deterministic replay, the dual staleness gates, rate limiting, out-of-order rejection, stale state never executing a buffered decision, real latency injection, the epoch barrier, scheduled takeover at its valid_from, and floor indexing:

```python
# tests/test_loop.py
import pytest

from fakes import (CHUNK_LEN, DT, Action, ActionBuffer, FakeClock, FakeSensor,
                   DelayedSensor, FailingSensor, NaiveEstimator, Provenance,
                   SafetyLimiter, SinePolicy, SlowPolicy, StateContract, run_episode)


def _state(angle):
    """Build a state whose measured joint position is `angle`: the input to the rate-limit tests."""
    return StateContract(proprio=(angle,),
                         provenance=Provenance("test", 0.0, "joint"),
                         validity_sec=0.05)


def test_episode_replay_bitwise_identical():
    a, _ = run_episode(FakeClock(), n_steps=100)
    b, _ = run_episode(FakeClock(), n_steps=100)
    assert a.sent == b.sent   # deterministic policy + injected clock => bit-for-bit identical


def test_stale_state_enters_safe_state():
    clock = FakeClock()
    _, safety = run_episode(clock, n_steps=10, sensor=DelayedSensor(clock),
                            policy_every=1)
    # the two gates keep their own books: the policy-side freshness gate fires 10 times (never publishes),
    # the control-side latest_valid fires 10 times (the register is empty); stale is reported before missing
    assert safety.safe_entries.count("stale_state") == 20
    assert safety.safe_entries.count("no_action") == 0


def test_safety_limits_against_measured_state():
    # velocity limiting must be done against the measured position, not the previous command:
    # sending 1.0 does not mean the robot reached 1.0 — computing Δq against the command systematically underestimates the true velocity
    safety = SafetyLimiter(max_velocity=5.0, dt=DT)   # 5 rad/s @ 50 Hz => per-tick ceiling 0.1 rad
    assert safety.check(_state(0.7), (5.0,)) == pytest.approx((0.8,))    # target 5.0: Δq=4.3, clamped to 0.7+0.1
    assert safety.check(_state(0.75), (5.0,)) == pytest.approx((0.85,))  # measured only reached 0.75: re-limit against the measurement


def test_out_of_order_commit_rejected():
    buf = ActionBuffer(epoch=1)
    clock = FakeClock()
    p = SinePolicy(); p.reset(epoch=1)
    state = NaiveEstimator().estimate(FakeSensor(clock).latest(0.0), 0.0)
    older = p.act(state, 0.0)
    newer = p.act(state, 0.0)     # simulate request #42 finishing before #41
    assert buf.put(newer, 0.0)
    assert not buf.put(older, 0.0)                # a late old decision must be rejected
    assert buf.current(0.0) == newer.values[0]


def test_stale_state_cannot_execute_buffered_action():
    # t=0.0 the state is fresh and the policy produces A (valid_until=0.32); the sensor dies at t=0.1.
    # during t=0.16~0.30 A has not expired yet, but the state is already stale — the control side must refuse to execute A.
    clock = FakeClock()
    ctrl, safety = run_episode(clock, n_steps=20, sensor=FailingSensor(clock, fail_at=0.10),
                               policy_every=100)
    assert len(ctrl.sent) == 8                 # t=0.00~0.14: state fresh, normal execution
    # t=0.16~0.38: state stale, A must never execute even though it is still unexpired in the buffer
    assert safety.safe_entries.count("stale_state") == 12
    assert safety.safe_entries.count("no_action") == 0


def test_slow_policy_yields_coverage_gap_then_recovers():
    # real latency injection: the decision is initiated at t and takes effect only at t+150ms. The control cadence is not
    # dragged down by inference, but there is no executable decision in the gap before it takes effect — the coverage gap lands explicitly in the safe state.
    clock = FakeClock()
    ctrl, safety = run_episode(clock, n_steps=40, policy_every=10,
                               policy=SlowPolicy(latency=0.15))
    assert len(ctrl.sent) == 32                # the first 8 ticks are the gap: no decision during the 150ms latency
    assert safety.safe_entries.count("no_action") == 8
    assert ctrl.sent                            # after inference completes, the control loop resumes execution


def test_epoch_barrier_rejects_stale_episode_action():
    # a late decision of Episode 41 (with a larger seq) must never override the current decision of Episode 42
    buf = ActionBuffer(epoch=1)
    clock = FakeClock()
    p41 = SinePolicy(); p41.reset(epoch=1)
    state = NaiveEstimator().estimate(FakeSensor(clock).latest(0.0), 0.0)
    p41.act(state, 0.0)
    late = p41.act(state, 0.0)          # decision #2 of episode 41, arriving late
    buf.sync_epoch(2)                    # reset_episode: episode 42 begins
    p42 = SinePolicy(); p42.reset(epoch=2)
    fresh = p42.act(state, 0.0)          # decision #1 of episode 42
    assert buf.put(fresh, 0.0)
    assert not buf.put(late, 0.0)        # old epoch arriving late: rejected even with a larger seq
    assert buf.current(0.0) == fresh.values[0]
    assert ("rejected_stale_epoch", late.sequence_id) in buf.events


def test_scheduled_chunk_takes_over_at_its_valid_from():
    # while A is executing, B (larger seq) arrives early but its valid_from is in the future:
    # A must keep executing during 0.10~0.20, and B takes over at the control boundary at 0.20 — no gap is allowed in between
    buf = ActionBuffer(epoch=1)
    a = Action(values=tuple(float(i) for i in range(CHUNK_LEN)), dt=DT, horizon=CHUNK_LEN,
               generated_at=0.0, valid_from=0.0, valid_until=0.32,
               state_stamp=0.0, epoch=1, sequence_id=10)
    b = Action(values=(2.0,) * CHUNK_LEN, dt=DT, horizon=CHUNK_LEN,
               generated_at=0.10, valid_from=0.20, valid_until=0.52,
               state_stamp=0.10, epoch=1, sequence_id=11)
    assert buf.put(a, 0.10)
    assert buf.current(0.10) == 5.0      # point 5 of A: floor((0.10-0)/0.02)
    assert buf.put(b, 0.10)
    assert buf.current(0.10) == 5.0      # B not effective yet: keep executing A, must not return None
    assert buf.current(0.19) == 9.0      # floor(9.5)=9: still A
    assert buf.current(0.20) == 2.0      # takeover point: B takes over at the control boundary
    assert buf.current(0.21) == 2.0      # floor(0.5)=0: point 0 of B


def test_chunk_indexing_uses_floor_not_round():
    # ZOH: values[i] covers [valid_from + i*dt, valid_from + (i+1)*dt), left-closed right-open
    buf = ActionBuffer(epoch=1)
    a = Action(values=tuple(float(i) for i in range(CHUNK_LEN)), dt=DT, horizon=CHUNK_LEN,
               generated_at=0.0, valid_from=0.0, valid_until=0.32,
               state_stamp=0.0, epoch=1, sequence_id=1)
    buf.put(a, 0.0)
    assert buf.current(0.015) == 0.0     # 0.75 beats: round would mis-pick 1, floor stays at 0
    assert buf.current(0.020) == 1.0     # interval boundary: enters the next point
```

Note that the assertion strength of the first test is "bit-for-bit identical" — because `SinePolicy` is deterministic. Swap in a sampling policy and the same assertion must be demoted, per the grading rules in the testing strategy section, to a tolerance-level or distribution-level check. What two hundred-plus lines of fake implementations buy you: every time you change a contract, these nine tests watch the seams for you — especially the last four: out-of-order commits, real latency injection, stale state never executing a buffered decision, and the epoch barrier are the first four things that bite once an async inference service and frequent resets are wired in.

## Testing Strategy: A Pyramid, Layer by Layer

Testing cost in embodied systems rises exponentially from simulation to real hardware, so bugs should die at the cheapest layer possible:

| Layer | What to test | How |
|---|---|---|
| Unit | StateContract serialization / expiry logic, action bounds, tokenizer round-trip consistency | plain pytest, millisecond level |
| Component | Single-layer substitution: fake observations into policy, fake policy into controller | fake Protocol implementations, run in memory |
| Sim integration | The full loop runs episodes in SimEnv; episode-replay regression | fixed seed, graded assertions (see below) |
| Runtime fault injection | policy latency / timeout / exceptions, buffer empty / expired / out-of-order, clock jumps | fake scheduler + clock injection, assert deadline invariants |
| HIL | Real robot + safe pose + speed limits; verify interface timing and watchdog triggers | small scale, human in the loop |

Of these, "record an episode and replay it" is the one piece of infrastructure most worth investing in: store a frame of state, feed it to the policy over and over, and assert by policy type — a deterministic policy must be bit-for-bit identical; a sampling policy (diffusion, most VLAs) gets numerical tolerances (atol/rtol) with a fixed random seed, or is checked with distribution-level or trajectory-level regression. GPU kernel non-determinism, quantization, and compilation optimizations all introduce numerical differences, so "always bit-identical" does not hold for modern policies. This one piece of infrastructure catches a large class of "the code didn't change, but the behavior did" regressions.

Of all fault injection, the first one most worth doing is the **policy latency sweep**: with an injected clock, drag the policy's production cadence through 0 / 50 / 100 / 150 / 500 ms and beyond, to timeout; at every level, assert the same invariants — the control loop's deadline is not exceeded, expired chunks are not silently reused, and missing chunks enter the safe state. VLA inference latency is the norm in production, not an exception, and this sweep turns "does the dual-rate decoupling actually hold" from confidence into a regression test. The last test in the minimal closed loop above is a simplified level of exactly this sweep.

## Temporal & Failure Contract: Where the Real Boundary of a Robot Runtime Lies

The first three contracts govern "what shape components exchange," and the Action's temporal fields plus the scheduling protocol govern "at what rhythm they exchange." This section handles the last two questions: **who makes the safety decision, and based on what**, and **how the system winds down when something fails**. These two questions are nearly free in simulation and cost real money in production — they are exactly what separates "the demo runs" from "the system runs long-term."

Step back and raise this layer's contract one more notch: an `Action` (and a `StateContract`) crossing a system boundary must carry all **four identities** — **semantics** (what space, what units: `ActionSchema`), **time** (a decision about which world moment, when it takes effect and when it expires), **coordinates** (in which frame it is expressed: `frame_id`), and **version** (the `schema_version` fingerprint — mismatch, and consumption is refused). Miss any one, and the other side of the seam has to fill the gap by guessing — and guessing is another name for a bug. The first three contracts each pinned down some of these; this section completes the remaining ones, together with the failure semantics.

### The Safety Gate: A Runtime Gate, Not an if Inside the Policy

Layer 4's routine "trim the numbers a bit" is not safety. Consider a concrete example: target position `q_target = 1.0`, current `q_current = 0.1`, control tick `dt = 0.001 s` — the position itself is entirely inside the workspace, but the implied velocity of this step is **900 rad/s**. A legal position does not mean a legal velocity, and whether a velocity is legal depends on the **last committed command**, dt, and the robot's current health (temperature, torque margin). So the input of a safety decision is naturally the `(state, action, context)` triple, not the action alone:

```python
# agent/safety/gate.py
@dataclass(frozen=True)
class SafetyDecision:
    allow: bool
    command: ControlCommand | None   # when allowed, released (possibly after project/rate-limit)
    reason: str                      # on rejection, the reason code for entering the safe state

class SoftwareSafeAction(Enum):
    """Software safe states: defined by the robot itself — an arm holding is safe; a drone landing is."""
    HOLD = "hold"                    # arm: hold the current pose
    BRAKE = "brake"                  # wheeled base: brake
    ZERO_TORQUE = "zero_torque"      # collaborative arm / quadruped: release torque
    SIT = "sit"                      # quadruped: lie down
    LAND = "land"                    # drone: land
    RETURN_HOME = "return_home"
    # note: EMERGENCY_STOP is not here — the e-stop is a hardware authority; see HardwareSafety

class HardwareSafety(Protocol):
    """Hardware authority: the e-stop chain, driver STO, independent limit switches. It does not belong to the software stack —
    it is online when the SafetyGate passes everything, and it still holds the robot when all software is dead."""
    def assert_emergency_stop(self) -> None: ...

class SafetyGate(Protocol):
    """An independent decision layer on the software control chain: no branch can get around it."""
    def evaluate(self, state: StateContract, action: Action,
                 context: dict) -> SafetyDecision: ...
    def enter_safe_state(self, reason: str) -> None: ...
```

The mapping from `SoftwareSafeAction` to concrete motions is implemented by each robot's `SafetyPolicy` — get this abstraction right, and the same skeleton moves from an arm to a quadruped to a drone. Two boundaries must be pinned down: implementing the e-stop as one value in a software enum amounts to assuming the Python process lives forever, so `EMERGENCY_STOP` belongs to `HardwareSafety`; and the `SafetyGate` is a runtime safety-decision layer, not functional safety in the certified sense (the ISO 13849 / IEC 62061 line) — a certified safety loop must exist independently of the software, and the software layer only performs runtime adjudication.

There is one layer even easier to overlook than "where the decision layer sits": **the uniqueness of the authority's writer**. A Protocol defines the shape of the seam, not the authority itself — if every component in the loop can get hold of the controller handle, then any future line `controller.send(raw)` quietly bypasses the SafetyGate, and "no branch can get around it" degenerates into a code-review convention. So the sole software owner of command authority is Safety: approved commands enter the RobotInterface through a `CommandSink`, and no component in the software stack (including the policy and the controller itself) holds a direct handle to the `RobotInterface`:

```python
# agent/control/sink.py
class CommandSink(Protocol):
    """The single entry of command authority: Safety-approved commands enter the RobotInterface
    through here. No component in the software stack (including policy and controller)
    holds a direct handle to the RobotInterface."""
    def submit(self, cmd: ControlCommand) -> None: ...
```

`Protocol ≠ authority enforcement`: the former is expressed by the type system, the latter is guaranteed by the structure of "only one writer." The chain is `Policy → Action → ActionBuffer → SafetyGate → CommandSink → RobotInterface`; without the sink link, everything upstream of Safety can still be bypassed.

**Fail-closed is the default semantics of this chain.** If the safety layer itself dies, the heartbeat is lost, or the state goes stale, the system's default behavior must be "forbid commands," not "keep executing the last command" — the latter hands fate to whatever old decision happens to sit in the buffer. But the exact meaning of fail-closed must be pinned down: what it revokes is **command authority** — forbidding new commands from entering the RobotInterface — **not zeroing the actuators**. A drone hovering or descending gently is safe, while stopping the motors abruptly could crash it; an arm holding its pose is safe, while zero torque could let it fall under gravity. The "closed" in fail-closed falls on "who still has the right to command," not on "what the command content is." This principle has one corollary: any component's failure must be able to land on a predefined safe state — if the Python process dies, the hardware limits and the independent e-stop chain still hold the robot.

### The Failure State Machine: Not Every Exception Is a hold_position()

"Stop everything and hold" is enough for a demo, but in production it locks the system into one crude behavior. A production system needs an explicit failure state machine, where every state has clear entry conditions, exit conditions, and an owner:

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
                 ┌─────────────┐   fault persists / relapses   ┌──────────────┐
                 │  SAFE_STOP  │ ───────────────────────────→ │ FAULT_LATCHED │
                 └──────┬──────┘                              └──────┬───────┘
                        │ operator / recovery                       │ manual reset only
                        ↓                                           ↓
                 ┌─────────────┐                              （manual recovery
                 │   RECOVER   │                               no auto-exit）  
                 └─────────────┘
```

Three semantics must be pinned down: **DEGRADED → SAFE_STOP is the clock of "the fault persists past timeout," not a "try again" loop** — the degraded state is a window left for recovery, not for retry momentum; **SAFE_STOP → RECOVER → RUNNING must go through the operator or an explicit recovery flow** — unattended auto-return to RUNNING is forbidden, otherwise one intermittent fault can keep the system oscillating forever in "fault — recover — fault again"; and **SAFE_STOP → FAULT_LATCHED is one-way** — the fault persists past timeout or keeps relapsing, and the system latches until manual reset. The difference between a latch and a safe stop is exactly "can still recover automatically" versus "a human must reset it first": the former leaves a window, the latter requires a human to see what happened before anything moves again.

### Latency Budget and Action Coverage: Turning "Fast Enough" into a Measurable Account

Beyond failure semantics, the Temporal Contract keeps a more everyday account: latency. The end-to-end budget from sensor to actuator is

```text
L_total = L_sensor + L_transport + L_sync + L_estimator
        + L_policy + L_queue + L_safety + L_controller  <  L_budget
```

where `L_budget` is the control period (5 ms at 200 Hz). Every term must be measurable and alarmable on its own — "the system is lagging" is not an actionable diagnosis; "L_policy p99 went from 80 ms to 210 ms" is. When the budget is breached, this diagram is the answer to which ring should raise the alarm.

A metric worth watching even more than policy frame rate is the **action coverage**: every control tick logs `valid_until - now` — how far the currently consumed decision is from expiring. Its distribution (p05 in particular) predicts incidents better than policy FPS: a high FPS with short chunks can still run coverage into the floor; coverage hugging zero for a long time means the policy's production cadence is already running right against the consumer, and any jitter turns straight into missing. Alongside it, distinguish the hardness of two kinds of deadlines: **the policy's deadline is a soft contract** — exceeding it means the decision grows old and quality degrades, covered by the staleness gate; **the control deadline is a hard contract** — exceeding it is a direct breach, and the safe state is its breach handler. Conflating the two is the most common objective-function misalignment when "optimizing policy latency."

### The Supervisor: A Cross-Cutting Layer Beyond the Six

The state machine above does not run itself. Lifecycle, health heartbeats, deadline monitoring, failover, metrics, logging, artifact validation — these concerns belong to no single layer yet cut across all of them; they belong to one independent **Runtime Supervisor**:

```text
┌──────────────── Runtime Supervisor ────────────────┐
│ lifecycle · heartbeat · deadline · fault · metrics │
└──┬─────────┬──────────┬─────────┬──────────┬───────┘
 Sensor    Estimator   Policy   ActionBuffer Controller
```

It answers questions of this kind: the policy thread has produced nothing for 200 ms — is the VLA just slow, or is it deadlocked? Has the estimator's quality degradation been recorded as a metric? After restarting a component, do the artifact manifest and the config hash still match? It is not drawn in the dual-rate loop diagram above, not because it isn't needed, but because drawing it in would clutter the picture — cross-cutting concerns and layered architecture are two orthogonal things. The scope must also be narrowed: the Supervisor only watches, it does not decide — it detects that the policy has timed out and moves the system into DEGRADED, but it never decides behavior for any layer; decision rights stay inside the layers forever. Turn it into a "central brain" and you have merely reinvented the big ball of mud in another form. But the failure state machine itself must have a single owner — otherwise the policy thread writes DEGRADED, the control thread writes SAFE_STOP, and the Supervisor writes FAULT_LATCHED, and three writers land right back in the old concurrent-authority problem: **the Supervisor is the sole state owner of RUNNING / DEGRADED / SAFE_STOP / FAULT_LATCHED; other components may only `report_fault(event)`** (`POLICY_TIMEOUT` / `STATE_STALE` / `CONTROLLER_FAULT` / `HEARTBEAT_LOST`) — they cannot `set_state(...)` on their own.

One last easily overlooked piece is the **episode lifecycle**. The boundary of a formal experiment is not the start and end of a `for` loop but explicit state transitions: `initialize → start → reset_episode → run → stop → fault/recover → shutdown`. Among them, `reset_episode` has a checklist that is easy to leak: **the policy's hidden state, the estimator's filters, the action buffer, the controller's tracking state, and the random seed**. Miss any one of them, and residue from the previous episode contaminates the next episode's decisions — when replay doesn't reproduce, this is usually where it broke first.

## Six Common Engineering Anti-Patterns

All are pits that recur in real projects, ordered by frequency of appearance:

1. **The big-ball-of-mud import**: modules import each other's implementations, and swapping any one component pulls the whole body. The fix: force every layer to depend only on Protocols, with imports flowing one way.
2. **Bare dicts / bare tensors everywhere**: state has no schema and field names rely on verbal agreement. There is only one fix — StateContract, validated at the boundaries.
3. **Two preprocessing pipelines for training and deployment**: the root of skew. The fix was given above: share the `ObsTransform` module; copying the implementation is forbidden.
4. **Safety logic scattered inside the policy**: safety decisions are handed to a learning system. The fix: an independent safety layer + watchdog, so the policy can always be bypassed — but above the software safety layer there are still hardware limits and an independent e-stop; the two must not replace each other.
5. **Stuffing policy and control into one loop**: one 150 ms VLA inference drags down the entire 200 Hz control loop. The fix: separate threads, separate frequencies — the policy produces chunks into the action buffer, the control loop only consumes the buffer, and the two frequencies are declared separately in config.
6. **Separate environment code for simulation and real robot**: interface differences are artificially amplified. The fix: one RobotInterface, with proprio field names generated from a single schema — it manages interface unification; the gap itself is handled separately via randomization and real-robot validation (see the Env section).

## Summary

This article turned "embodied AI lacks interfaces" from a judgment into a skeleton: a six-layer runtime stack + one training pipeline, joined by four groups of contract seams — **State** (the shape of state: observation, proprioception, quality scores), **Action** (temporalized actions: the ActionSchema semantics fingerprint, valid_from / valid_until / epoch / sequence_id, and the active/scheduled dual-slot ActionBuffer scheduling protocol with the epoch barrier), **Env** (RobotInterface + TaskEnv, splitting sim from task), and **Temporal & Failure** (the four identities, the safety gate and command authority, fail-closed, the failure state machine with FAULT_LATCHED, latency budget and action coverage, the Supervisor, the episode lifecycle). VLA, Diffusion, and MPC are just three implementation classes of the same Policy slot in the same codebase. The value of the skeleton is not the code itself — it is that it demotes "swapping route, sensor, or robot" from open surgery to plugging in a component. And that is precisely the precondition for the evaluation protocol series (9/12–9/16) to land: only with clear seams does compliance have pluggable measurement points.

Next steps can go in two directions. One: turn one Protocol in this skeleton into a complete runnable implementation (say the MPC branch, with world-model rollouts). Two: follow the 9/12 evaluation protocol and wire evidence collection into every layer. Tell me in the comments which one you want.