---
title: 'Stacking Sensors Is Not Fusing Them: Multimodal Robotics Lacks an Interface, Not a Model'
slug: "2026-09-12-multimodal-fusion-interface"
date: 2026-09-12
draft: false
categories: ["Embodied AI", "Multimodal Perception"]
tags: ["Embodied AI", "Multimodal Fusion", "Visuo-Tactile", "Force/Torque", "Proprioception", "Representation Interface", "Multimodal State Estimation", "Registration", "Cross-attention", "Modality Dropout", "VLA", "World Model", "Frame Alignment", "Time Alignment", "Uncertainty"]
description: 'Multimodal fusion is usually presented as a question of "which attention architecture", but in robotics the real bottleneck is upstream: four streams (Vision / Tactile / Force-torque / Proprioception) have never agreed on temporal and spatial registration, or on task-relevant semantic projection. This piece splits "fusion" into *perception → registration → semantic projection → multimodal state estimation → structured state interface*, and argues that vision won not because its encoders are cleverer but because it locked down those steps first (as a family of highly interoperable task-level conventions, not a single unified schema) alongside a whole ecosystem — GPUs, cheap sensors, internet-scale data, annotation, standard file formats. It does not oppose cross-attention; it limits its job. Attention can compose already-aligned representations, but it cannot substitute for explicit registration and state estimation. The concrete route proposed: a contact-set-centric state interface for contact-rich manipulation (not "compress all robot data into contact sets"), with sensor-specific perception, and slot records that carry uncertainty, provenance, timestamps, and age. Policy, world model, and controller all consume the same interface. Missing modalities and cross-modal contradiction become first-class citizens in both training and benchmark.'
toc: true
related_articles:
  - 2026-09-11-tactile-force-sensing
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-09-robot-data-scaling
  - 2026-09-07-vla-world-models
  - 2026-09-03-vla-deep-dive
  - 2026-08-26-world-model-in-robotics
---

> Continues [The Hand Robots Don't Have: Tactile and Force Sensing in Embodied AI](/en/articles/2026-09-11-tactile-force-sensing/): that article pinned three tags — Action-conditioned observation · Contact-state representation · Closed-loop value — and left an obvious question mark behind: **if tactile, force, and proprioception each carry real value, why hasn't any of them formed a reusable common representation the way vision has?** This article answers that question head-on. The answer is not model architecture, **the answer is interface**.

Picture a bimanual robot fitted with an RGB camera, GelSight fingertips, wrist six-axis force/torque sensors, and joint encoders on every link. The hardware bill of materials looks genuinely "multimodal". Yet when you hand that rig to a policy, most published work tells you "just fuse them with cross-attention and you're done". That runs in a demo; **in real deployment it will almost certainly fail in one of four ways**: a modality drops frames, a sensor dies, coordinate frames drift, or the model quietly converges to one dominant modality and treats the rest as noise. These four are not engineering footnotes. **They all trace back to the same root cause: the modalities never agreed on an interface consumable by policy, world model, and controller alike — an interface with timestamps, frames, uncertainty, and provenance baked into its schema.**

This piece takes on the easiest topic to hand-wave in embodied AI: **multimodal fusion**. It will not sell you a specific network, and it will not argue against cross-attention. What it *does* argue against is a **single policy network being asked to handle state estimation, sensor fusion, and control at once**. The article digs fusion down to its three foundations — time, frame, task semantics — and uses them to explain **why vision was able to grow a common data interface while tactile / force-torque / proprioception have not, and what a minimum usable state interface looks like if you want to start building today**.

## 0. Framework: from "a fusion layer" to "multimodal state estimation + state interface"

Put the whole article's frame up front. Every section returns to this picture.

```text
                     ┌────────────────────────────────────┐
                     │         Raw sensor streams         │
                     │  V · T · F · P  (four channels)    │
                     └──────────────────┬─────────────────┘
                                        │
                                        ▼
                        Modality-specific perception
                        (sensor-native encoder、raw → 局部结构观测)
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
      Temporal registration     Spatial registration     Semantic projection
        (clock / timestamp        (SE(3)、hand-eye、
         alias / event bus)        sensor-mount、wrench transform)
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        ▼
                     ┌────────────────────────────────────┐
                     │  Multimodal state estimator         │
                     │  (Bayesian fusion over $C_t$,       │
                     │   wrench_ext, robot_state, belief)  │
                     └──────────────────┬─────────────────┘
                                        ▼
                     ┌────────────────────────────────────┐
                     │  Structured state interface (API)   │
                     │  contacts[]  · wrench  · robot_state│
                     │  task_context · belief              │
                     │  + uncertainty · provenance         │
                     │  + timestamp · age                  │
                     └──────────────────┬─────────────────┘
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
                   Policy          World Model         Controller
                     │                  │                  │
                     └──────────────────┼──────────────────┘
                                        ▼
                                   Action a_t
```

Three things this diagram is trying to say, and the three architectural claims of the article:

1. **Fusion is not a network module; it is a whole "state-estimation + interface contract"**. Letting a policy network also handle state estimation is the deepest risk in most current multimodal-robotics papers.
2. **Alignment is only a sub-section before state estimation** (temporal registration + spatial registration + semantic projection), **not fusion itself**. Stuffing semantic projection into "alignment" dilutes a problem that admits closed-form solutions into one that is really representation learning.
3. **The interface layer (Structured State API) is not a learned embedding — it is a contract**. Every field has a unit, a frame, a timestamp, an uncertainty, and a provenance. Its optimisation target is **interoperability**, not "friendly to one model".

A reviewer-defensible way to state the core thesis:

> **Multimodal fusion should *compose already-aligned modality representations*, not *discover the alignment contract from raw observations*.**

If 9/11's three tags were "the three hard problems tactile faces on its own", this article pins three for multimodal fusion:

> **Alignment before fusion · Interface before architecture · Missing modalities by design**

Every section elaborates on these three.

## 1. "Multimodal" is not "many sensors"

The easiest way to derail the topic is to conflate "many sensors" with "many modalities" — and many paper introductions do exactly that. This conflation does not hold.

**"Modality" itself has no unique physical definition; it is a product of the analytical frame you commit to**. For this article's taxonomy, a modality is fixed by four jointly-specified properties: **measurement space, physical origin, noise model, and update semantics**. Different authors and tasks may draw the boundary a little wider or narrower — what matters is that the choice is explicit at the top and consistent throughout.

Under this taxonomy, three cases that are commonly mislabelled "multimodal" are actually not:

- Wrist camera + overhead camera → **same modality, multiple viewpoints**, not two modalities. Measurement space, physical origin, and noise model essentially match; only extrinsics differ; one encoder handles both.
- Four GelSight fingertip cameras → **an internal structure of one tactile modality**, not four visual modalities. The measurement space is "surface deformation field of an elastomer", not scene RGB. Even if a CNN is involved, downstream semantics are contact geometry, not object detection.
- Joint encoder + motor current → in this article's taxonomy, treat them as **different observation channels within a single robot-state modality** (one is a direct position measurement, the other an indirect torque inference, sharing the same latent physical state). Fusing them as two independent modalities only teaches attention redundancy.

Conversely, these are **genuinely different modalities under this taxonomy**:

- **Vision** (RGB / RGB-D / event camera) — light-radiance field, camera frame, effective observation rate typically 15–60 Hz.
- **Tactile** (GelSight / GelSlim / TacTip / 9DTact / taxel arrays) — contact-interface deformation or force distribution, sensor frame; effective observation rate varies by sensor family from 30 Hz to kHz — camera-based variants are frame-rate-limited, taxel arrays are scan-limited, event-based variants are event-density-limited.
- **Force/torque** (wrist F/T, six-axis wrench) — **aggregate** contact wrench, sensor frame + tool frame, typically 500 Hz – 1 kHz.
- **Proprioception** (joint angle q, velocity q̇, joint torque τ, end-effector pose) — the robot's own state, base / world frame; typical range from hundreds of Hz to kHz; industrial servo loops higher, teaching-arm loops often in the hundreds of Hz.

Some works add audio, thermal, gas, or ultrasound as fifth / sixth modalities. The classification logic is the same.

**Where this section lands**: before talking about fusion, write out the modality list and pin the four defining properties *under an explicit taxonomy*. Skip this and every downstream fusion architecture is built on air. 9/11 §2.1 splits contact sensing into Tactile / Force-torque / Proprioceptive branches; this article keeps that same three-way split and confines the discussion to **V + T + F + P**.

## 2. Three pre-fusion steps: registration and semantic projection

This is the most technical section and the easiest to skim past. There is also an **important terminology tightening** here — the previous draft called all three bases "alignment", which a reviewer correctly noted conflates two different kinds of work. Going forward this article separates:

```text
Pre-fusion pipeline
├── Registration        (hard constraints, closed-form or standard algorithms)
│   ├── Temporal registration
│   └── Spatial registration
└── Semantic projection (learning task-relevant state representations)
    └── Raw observation → task-relevant state
```

**Registration** is a measurement-layer problem: standardisable, calibratable, unit-testable. **Semantic projection** is closer to perception and state estimation, and is what §6's state interface cares about. Both are prerequisites to composition, but their natures differ, and calling them both "alignment" hides that.

### 2.1 Temporal registration

The default time constants of the four streams differ by orders of magnitude:

```text
Vision (image-based)      ~15–60 Hz            Δt ≈ 17–67 ms
Tactile (image-based)     ~30–200 Hz           Δt ≈ 5–33 ms
Tactile (taxel / array)   ~500 Hz – kHz        Δt ≈ 1 ms
Force/torque              ~500 Hz – 1 kHz      Δt ≈ 1–2 ms
Proprioception            ~hundreds of Hz – kHz  Δt ≈ 0.5–2 ms
```

One thing easy to conflate: these are **effective observation rates**, not internal sensor sampling rates. A camera-based tactile sensor advertising "kHz taxel sampling" but delivering 30 Hz images gives you an effective rate of 30 Hz.

A common but overly simplistic baseline is to **resample all streams onto the slowest policy clock, typically the vision rate**. For contact-rich control this compresses some high-rate events into invisible aliasing — slip detection, transient contact force peaks, joint impacts often happen *between* two vision frames, and downsampling to 30 Hz literally throws them away.

What real systems actually do more often, and more robustly, is **multi-rate coexistence**:

```text
                    ┌── high-rate state estimator / controller
                    │       (1 kHz or higher, dictated by robot servo hardware)
                    │            ↑
                    │       latest proprio · latest F/T
                    │       tactile history · latest vision
                    │
   multi-rate observations
     ↑        ↑        ↑        ↑
  camera   tactile    F/T     proprio
```

The fusion layer must **acknowledge different time constants and choose the alignment layer per task sensitivity**. Vision policy can run 30 Hz; force/tactile-driven compliance must stay at native rate; events need their own bus.

Three details that must live in the interface:

- **Timestamp semantics**: sensor timestamp (the exposure or sample instant) vs arrival timestamp (when software actually sees it) vs host timestamp (the fusion clock). The three can differ by 5–20 ms — enough to shift a "cup contact moment" outside the finger-closure window.
- **Clock drift**: cameras run their own crystal, robot controllers run another, host PCs a third. Tens of milliseconds drift in minutes, enough over a night to teach a fusion model wrong lead/lag correlations. NTP/PTP only fixes things at the system layer; at the sensor layer you still need hard triggers.
- **Event-driven vs polled**: tactile slip detection is a sparse event stream, not a periodic sample. Stuff it into a uniform time buffer and you lose the density signal.

**The rule**: **treat different time constants as a modelling assumption; do not paper over them with downsampling**. This is the same judgment as 9/11 §4.1's refusal to write down specific ms numbers.

### 2.2 Spatial registration

The four streams natively live in four different frames:

```text
Vision           camera frame (or: RGB in pixels + depth in camera)
Tactile          sensor frame (local frame attached to fingertip surface)
Force/torque     sensor frame (usually wrist-mounted, must be transformed to tool/base)
Proprioception   base frame / world frame
```

To place them in one fusion layer, at minimum three things must be true:

- **Hand-eye calibration**: $T^{cam}_{base}$ is SE(3); a single collision can nudge it by fractions of a degree or a few degrees. Probably fine for a coarse grasp; immediately catastrophic for fine contact manipulation.
- **Sensor-mount calibration**: fingertip sensor frame → link frame, $T^{sensor}_{link}$. This is often quietly waved away with "we assume it was mounted perfectly", but optical tactile sensors like GelSight are extremely sensitive to it.
- **Wrench transformation and gravity compensation**: an F/T sensor reads wrench in its own frame; you must transform to base or world and subtract tool gravity plus inertial terms. Any step wrong and every downstream normal/shear decomposition is broken.

**A common symptom of a bad spatial registration**: the trained model works at the training pose, training camera placement, and training tool — swap any of them and accuracy collapses. The model was actually learning correlations in a particular sensor frame, not the task itself.

A subtler issue is **relative vs absolute pose**: contact physics fundamentally depends only on who-is-touching-whom, but many policies feed absolute end-effector pose in world frame, forcing the model to learn degrees of freedom it should not need. This mirrors 9/11 §3.3: what a policy needs to be consistent about is the transition of contact modes, not the absolute value of every physical parameter.

### 2.3 Semantic projection

This step is not really "registration"; it is mapping observations onto a shared task-semantic coordinate:

```text
Vision           ──► object / scene semantics (what, where)
Tactile          ──► local contact semantics (who is touching, how, slipping?)
Force/torque     ──► aggregate contact semantics (net wrench right now)
Proprioception   ──► self-state semantics (my posture, my velocities)
```

9/11 §2.1 split signal → meaning into nine layers:

```
Sensor → Calibration → Raw obs
       → Contact perception
       → Contact geometry & wrench
       → Contact mode & physical state
       → Task-relevant belief
       → Policy / controller
       → Action → New contact
```

**The central multimodal-fusion question is: at which layer do you compose?** A common mistake is composing at the raw layer — concatenate four tensors and throw them into a network. That forces the policy to learn all of §2.1 / §2.2 / §2.3 from scratch — enormously expensive and sample-inefficient. A more reasonable choice is to **compose at the slot layer**: let each modality finish its own raw → perception → contact-geometry chain, and only compose at "contact events, external wrench, robot state, task belief", where semantics have already converged. That's §6's state interface.

**Landing**: temporal and spatial registration are hard measurement-layer constraints; semantic projection is perception and state estimation. Hand-wave any one of them and the fanciest fusion architecture is just paying back that debt later.

## 3. Genealogy of fusion paradigms: mechanisms on different axes, not mutually exclusive alternatives

When the literature talks about "multimodal fusion", it usually means **how to organise signals at composition**. This section walks the mainstream designs by *when* fusion happens, and calls out failure modes. The classic Baltrusaitis et al. survey [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) taxonomises multimodal ML into representation / learning / feature selection / fusion / application; this section borrows the fusion sub-taxonomy.

**Positioning first**: these "paradigms" are not mutually exclusive architectures — they are mechanisms on **different axes**, and mature systems usually combine several:

| Mechanism | What it solves | Where it sits in §0's pipeline |
| --- | --- | --- |
| Modality-specific encoder | raw → local observation | Perception |
| Temporal / spatial registration | clocks, SE(3), frames | Registration |
| Cross-attention | learned composition | Mainly inside or after state interface |
| Shared latent / VLT-style | representation alignment | Semantic projection |
| Late / decision fusion | decision-level composition | Policy / controller |
| Structured state interface | schema and contract | State interface |

So the article's stance is not "against attention". It is:

> **Cross-attention cannot substitute for explicit temporal / spatial / semantic registration.**

This is the pivot sentence for §3 and for the whole article.

### 3.1 Early concat: raw / embedding-level concatenation

The most naive pattern: encode each stream, concatenate into one big state vector, throw it into an MLP or Transformer.

```python
state_t = concat([
    enc_v(img_t),
    enc_tac(tac_t),
    enc_ft(wrench_t),
    proprio_t           # (q, q̇, τ, EE pose)
])
a_t = policy(state_t)
```

Upside: no design overhead, and with enough data the network learns *some* alignment implicitly.

Downside — and this is nearly universal:

- Time-constant differences are hidden by concat and the model learns fake lead/lag "at the moment a vision frame arrives, what were the other signals doing".
- Any dropped frame must be zero-filled or held last-value, both out-of-distribution.
- No modality-specific prior — tactile and vision encoders have very different inductive biases; flattening them pollutes gradients.
- Any missing-modality combination not seen in training collapses the policy.

**The VLA paragraph needs care**. An earlier draft labelled RT-2 / OpenVLA / π0 as "vision + proprio + language concat", which is inaccurate:

- **RT-2** [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) represents robot actions as **text tokens** co-trained with the VLM — not a state-vector concat.
- **π0** [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) is VLM backbone + a **proprioception token** (linearly projected into the transformer embedding space) + a **noisy action chunk**, decoded through flow matching.
- **OpenVLA** [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) explicitly supports multi-camera, depth, and proprioceptive state encoding configurations; not a naive V+L concat either.

A more accurate description:

> The common thread across mainstream VLAs is not "simple concat", but "**vision-language pretraining as the scale entry, with robot state injected as an additional representation**". They differ a lot in input organisation and action representation. What they share is that **tactile / force-torque have not yet become an input modality of the same scale and standardisation as vision**.

This absence is not oversight — adding tactile reopens §2's three steps wholesale, and tactile/F/T lack the equivalent pretraining ecosystem. This is expanded in §9.1.

### 3.2 Cross-attention / Transformer fusion

One step further: treat each stream as a token (or short token sequence), then run self- or cross-attention on top. This is the template used by most of the vision-language-action lineage. Tsai et al.'s Multimodal Transformer [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) is a canonical starting point, explicitly designed for temporally unaligned modalities.

Upside: the architecture acknowledges that cross-modal alignment is *learned*, giving soft alignment room to emerge. It can also ingest irregular timestamps. And a well-built multimodal transformer can absolutely combine:

- **timestamp / positional embeddings** for time offsets
- **relative temporal encoding** for lead/lag
- **modality embeddings** to distinguish channel identity
- **frame-aware features** (SE(3) transforms as input, not something to be "learned")
- **modality-specific encoders** preserving per-channel inductive biases
- **modality dropout / masking** to forbid single-channel dependence
- **auxiliary per-modality losses** to keep each stream individually informative
- **attention masks** to explicitly handle missing modalities

So the real question is not "attention crashes". It is: **if you do not pin down the above items as part of the interface contract, attention will "learn" registration from the data distribution — and what it learns will be a coincidence of that distribution, not a transferable interface**.

In a mature multimodal system, cross-attention's job is: **perception and registration hand off aligned representations into the state interface, and attention composes *within* that interface**.

### 3.3 Late / decision-level fusion

Each modality proposes its own sub-policy (or sub-value, or sub-action), and a decision layer merges them via weights or confidence gating. This is closer to traditional robotics: vision provides coarse reach, F/T provides force-limited approach, tactile provides slip recovery, proprioception tracks the nominal trajectory.

Upside: **engineering-robust**. Losing a modality loses only one proposal instead of crashing everything. Safety layers and interpretability are easier.

Downside:

- Low-level cross-modal coupling is lost. Many signals are physically cross-modal (e.g. "tactile + F/T jointly say this contact is sliding, not sticking"), and it is hard to reconstruct them once proposals are separated.
- Weighting rules are either hand-written (does not scale) or learned (back to §3.1 problems).
- Poor fit for tasks that need a *joint* belief supported by multiple modalities.

**This family is in fact the most common in industrial contact-rich manipulation** — it is usually called "controller hierarchy" instead of "fusion", which is why it barely appears in papers. The impedance + visual coarse pose + tactile slip-recovery stack described in 9/11 is exactly this late-fusion pattern.

### 3.4 Shared latent / VLT-style alignment

Contrastive, distillation, or CLIP-style objectives pull modalities into one latent space. Recent examples include:

- TVL / Binding Touch to Everything [arXiv:2402.13232](https://arxiv.org/abs/2402.13232) (Zhao et al., ICML 2024) — aligns tactile with vision-language at the representation level; about 44K vision-touch pairs; hard to avoid in any "tactile foundation model" discussion.
- AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) (Feng et al., 2025) — a unified static-dynamic representation across **heterogeneous visuo-tactile sensors**; effectively a "tactile-internal" shared latent.
- 3D-ViTac [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) (Huang et al., CoRL 2024) — visuo-tactile joint representation for fine-grained insertion; abstract explicitly reports visuo-tactile gains over vision-only.
- Lee et al.'s Making Sense of Vision and Touch [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) (ICRA 2019) — an early anchor for visuotactile self-supervised alignment.

Upside: representation is reusable, with a high ceiling across tasks and embodiments.

Downside: **where does the supervision signal come from?** Vision-language contrastive learning has billions of web image-text pairs. For a four-way vision-tactile-force-proprioception alignment there is no natural supervision source. Current mitigations are (a) treat "close in time" in teleoperation logs as weak alignment supervision — Calandra et al.'s More Than a Feeling [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) is an early tactile-grasp instance of exactly this idea — or (b) route via language/vision as a bridge and pull tactile into the V-L space (TVL does this). Both are far from scale.

**A note on inference**: the previous draft chained "shared latent → structured interface needed" as though the literature had concluded that. That is **this article's own inference**, not the conclusion of those papers. Restated more carefully:

> Those works show that shared latent representations are viable within specific task families, but their representations are policy- or dataset-specific. This article argues that if we want these representations to compose across policies, world models, controllers, diagnostic tools, and sensors, we need a **state interface beneath the latent**, with units, frames, timestamps, and uncertainty. This layer is the article's central proposal, not a conclusion of the papers cited above.

9/11 §3.5 split representation into "interface" and "learned representation". Shared latent belongs to the latter branch; this article is pushing the former.

### 3.5 Structured state interface: the route this article recommends

The last option: **do not fuse at raw and do not fuse inside a latent — compose at an explicitly designed intermediate representation**. That representation is not a learned embedding; it is a set of **slots whose semantics have already converged and which carry uncertainty, provenance, and timestamps**. This is the direct answer to 9/11 §3.4's observation that vision has a common data interface while tactile does not — with the scope tightened to contact-rich manipulation.

The specific schema is in §6; here is the shape:

```text
state_slots = {
    contact_set      : list of contact records (§6.1 details)
    wrench_ext       : 6D + covariance
    robot_state      : (q, q̇, τ, EE pose)
    task_context     : language token / goal embedding
    belief           : task-relevant latent / explicit
}
```

Each of the four streams fills the keys it can; "fusion" reduces to "writing into the same dict"; policy and world model both read from the dict. Advantages:

- Losing a stream only affects the keys it writes to; other keys stay clean. Modality dropout becomes natural.
- Time base, frame base, and semantic base are all resolved **inside the slot definition**; the composition architecture can then be swapped freely.
- Directly compatible with 9/11 §5.2.1's feedback-value ablation — arms A/B/C/D literally mean "write fewer keys into the dict".

**Downside**: designing the slot is heavier than "throw a Transformer at it". If the slot fidelity is poor, downstream policy cannot exceed it. **And the slot does not subsume vision's non-contact semantics** — object identity / geometry / free-space / occlusion / scene context all live elsewhere in the state interface. This article recommends a state interface for **contact-rich manipulation**, not a universal multimodal interface. That scope tightening matters; §6 makes it explicit.

**Landing**: within the genealogy, early concat / attention / late fusion / shared latent each have a place, but almost all of their observed failure modes trace back to §2's three pre-fusion steps. This article moves engineering attention from "which attention to pick" back to "define the state interface first".

## 4. Why vision got there first

A natural follow-up: §2's three steps exist for vision too, so why did vision still manage to grow a common data interface? A historical-engineering answer, also useful as a mirror for tactile.

**First a scope correction**: this is not single-causal. Vision's rise combines locked-down registration and semantics with **GPU/CNN scaling laws, cheap CMOS sensors, internet-scale data, an annotation ecosystem, and standardised file formats (JPEG / PNG / MP4 / HDF5)**. This article highlights the three steps because they are precisely what tactile currently lacks, **not** because it claims "three steps solved = vision wins".

**On temporal registration**: video is frame-indexed by nature. 30 Hz or 60 Hz. Every downstream task — classification, detection, segmentation, SLAM — has agreed "frames are the unit". This sounds trivial but it **eliminates the hardest class of problems in fusion discussion**: everything image-based lives on the same time axis.

**On spatial registration**: pinhole model + camera intrinsics + extrinsics turn "image pixel ↔ world point" into one formula, $s \cdot m = K [R | t] \cdot M$. Every 3D vision, SLAM, NeRF, 3D Gaussian Splatting, and multi-view stereo method grows on this convention. Calibration errors exist, of course, but **what they look like is predictable and reproducible**.

**On semantic projection**: raw RGB has no semantics, but the vision community spent the last decade using COCO / ImageNet / ADE20K / LVIS to form **a set of highly interoperable task-level representation conventions** — object class, bounding box, instance mask, depth map, affordance, caption. To be more precise, these datasets do **not** constitute a single unified semantic schema: ImageNet is a classification ontology, COCO is detection + instance + caption, ADE20K is scene parsing, LVIS is a long-tail instance distribution. Four separate definitions that happen to be composable. **"Highly interoperable task-level conventions" is closer to the fact than "a unified semantic schema", and it also explains why CV models interoperate so quickly.**

Contrast with tactile:

- Temporal registration: 30 Hz to kHz across sensor families; event- and poll-triggered streams mixed; no consensus.
- Spatial registration: GelSight is pixel + elastomer deformation; 9DTact is pixel + 3D deformation field; taxel arrays are 1D/2D force distributions; optical waveguides (AnySkin / DigiTact) are a different topology entirely. There is no agreed shape for "one tactile reading".
- Semantic projection: contact point, normal, shear, slip, mode appear across papers, **but no cross-dataset, interoperable task-level conventions**. This is also a good moment to correct a previous-draft analogy — "9DTact calls it 6D force, GelSight calls it shear map, arrays call it taxel load, all the same physical quantity viewed differently" is too strong. GelSight's shear / deformation map is closer to a **raw local deformation observation**, while 9DTact's 6D force is a **model-inverted global wrench estimate**. They contain overlapping physical information but at different levels — **observation layer vs estimation layer**. This distinction directly shapes the slot schema in §6.

**Landing**: vision won not because its encoders are cleverer, but because it locked down §2's three steps first, together with a whole ecosystem — every subsequent fusion discussion inherits a shared floor. Tactile, F/T, and proprio need to do the same before any fusion ecosystem at scale becomes possible. Conventions first, architecture second.

## 5. Per-modality fusion pain points

Walk each stream's concrete difficulties before designing the interface in §6.

### 5.1 Tactile: images, taxels, optics, capacitive — not unified at the raw layer

Tactile is the most "fragmented" of the four. The same physical quantity — local contact-interface deformation or force distribution — has at least these **structurally distinct** implementations:

```
Image-based      GelSight  · 3 colored LEDs + camera → RGB deformation image
                 9DTact    · annular colored LEDs + camera → multi-view field [arXiv:2308.14277]
                 GelSlim   · 3 cameras + speckle elastomer → 3 optical flow fields
                 TacTip    · silicone cone + fibers + camera → tip pose

Taxel array      BioTac · arrays of pressure/temperature/EDS channels
                 1D/2D capacitive arrays · local normal-force distribution
                 Piezoresistive arrays · local stress map

Optical waveguide AnySkin · DigiTact · light propagation in elastomer, edge-triggered contacts

Proprioceptive-  F/T + kinematics back-out ("soft tactile")
 inferred
```

Before entering the fusion layer, each of these needs its own encoder — and the encoders differ structurally: CNN for images, MLP for taxel arrays, spline models for waveguides, IK for proprioceptively-inferred signals.

A common misconception: "then build a tactile foundation model that unifies all sensors". That is precisely what AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) is trying to do, and the direction is right — **but it unifies the learned-representation layer, not the raw layer, and not the state-interface layer**. Even with AnyTouch, its output is still an embedding; you still need an explicit "embedding → contact slot" convention to enter state interface. 9/11 §3.5 split "interface" from "learned representation" specifically to keep these layers from collapsing into one another.

**Fusion pain points**:

- Different sensor families differ in time constant and trigger semantics; temporal registration must be handled per case.
- Sensor-frame definitions disagree (GelSight is pixel, arrays are index, waveguides are topology).
- Output semantics disagree (image / force field / 3D deformation) — and note this is **observation-level vs estimate-level**, not "same quantity, different projection".

**Mitigation**: introduce a **sensor-specific decoder** between raw and slot that unifies heterogeneous tactile outputs into §6.1's slot schema. It can be analytic or learned, but it must be **part of the interface**, not buried inside the policy.

### 5.2 Force/torque: looks the "cleanest", hides the most traps

F/T outputs a 6D wrench. Format is far more standardised than tactile, but the **semantics** are messier than they appear:

```
raw_wrench_sensor = contact_wrench + gravity_wrench
                  + inertial_wrench + friction_wrench + bias
```

Extracting "external contact wrench" requires at minimum:

- **Bias compensation** — sensor bias drifts with temperature and age; a tare is standard but rarely perfect on real hardware.
- **Gravity compensation** — $\tau_g = g(q)$ depends on tool and gripper mass distribution. Change the tool, the entire curve changes. Formulas look different in base frame vs sensor frame; easy to get wrong.
- **Inertial compensation** — $\tau_i = M(q)\ddot{q} + C(q,\dot{q})\dot{q}$. Non-negligible at high speed, sometimes ignorable at low speed.
- **Frame transformation** — sensor-frame wrench to tool / base / world requires SE(3) adjoints. Every step is a place to lose precision.

There is also a more essential limit: F/T gives an **aggregate** wrench, not a spatially distributed contact. This aggregate-vs-field distinction is why 9/11 §2.1's taxonomy puts tactile and force/torque in separate branches, and it has direct consequences for state-interface design (see §6.2).

**Fusion pain point**: F/T has "clean" semantics (six numbers) but *relies on extensive compensation upstream*, and that compensation depends on tool models and dynamics. Change the tool and the downstream fusion distribution changes with it.

**Mitigation**: F/T should produce **two** things at the slot layer — (a) the compensated external wrench (for force-aware policies), and (b) residual magnitude (for drift detection and unexpected-collision flags). The second is often dropped.

### 5.3 Proprioception: the "background" channel that actually dominates

Among the four, proprioception is by default the fastest and the **only one the robot natively carries in full**. But it has an awkward status: **many policies concatenate (q, q̇, τ) directly into state without ever calling it a "modality"**, which is why proprio is often missing from multimodal-fusion discussions. That is a mistake.

Proprio plays two roles in fusion:

- **As contact-hypothesis input**: given a commanded end-effector trajectory and joint torques, external contact force can be recovered via $J^T \hat{F}_{ext} = \tau_{residual}$ (contact inference — a classic tool from the Hogan impedance lineage). This lets proprio contribute to the contact slot, not just to its own channel.
- **As time and frame anchor**: fusion needs a stable frame. Proprio runs at whatever rate the servo hardware allows (hundreds of Hz to kHz), making it the most "clock-like" of the four; EE pose can serve as an anchor for camera and sensor frames alike.

**Fusion pain point**: proprio is "too clean" — it measures the robot's own state without external-world uncertainty. Models tend to **over-rely** on proprio, learning it as a shortcut. Result: fine in non-contact scenes, but under-using tactile / F/T exactly when contact matters. This is precisely what 9/11 §5.2.1 feedback-value ablation guards against — the evaluation must measure "with proprio already given, how much does tactile/F/T add", not "tactile-only vs vision-only".

### 5.4 Different time constants are themselves modelling assumptions

Combining §5.1–5.3 with §2.1:

> **Different modalities running at different effective observation rates is not an engineering "how do we fuse them" problem, but a modelling assumption about "what control bandwidth the task actually needs".**

Two examples:

- **Wiping / polishing**: force-control bandwidth is at least 100–500 Hz, vision at 30 Hz is fine, tactile and F/T must stay native, proprio needs torque-level rates. Fusion cannot be downsampled to 30 Hz.
- **Pick-and-place / insertion**: vision dominates, contact events are sparse; downsampling tactile to vision rate is an acceptable approximation.

Concrete rates vary with hardware, task, and controller; the numbers above are illustrative, not normative. Same judgment as 9/11 §4.1.

## 6. A minimum viable state interface

Time to answer "what should we actually build". This section proposes an interface that is implementable, benchmarkable, and incrementally evolvable.

**Scope tightened upfront**: this is **not** a claim that "all robot multimodal data should be compressed into contact sets". It is a claim that **for contact-rich manipulation, a contact set should become the core structured object of the cross-modal state interface**. Vision's object identity / geometry / free-space / occlusion / scene context do not fit inside a contact set — they live in other slots (e.g. `task_context`, `scene_state`, `world_model_state`). The full state interface §6.1 lists them.

### 6.1 Contact set as the core structured object — but not the whole interface

Define the cross-modality shared **contact record set** (upgraded to carry uncertainty, provenance, timestamps):

$$
C_t = \big\{\, \big(\, \mathrm{id}_i,\; p_i,\; n_i,\; f_i^{\perp},\; f_i^{\parallel},\; \phi_i^{\text{slip}},\; m_i,\; \Sigma_i,\; s_i,\; t_i^{\text{obs}},\; a_i \,\big) \,\big\}_{i=1}^{N_t}
$$

Every field:

```python
contact = {
    # identity & geometry
    "id":                int,           # tracking ID, persistent across frames
    "p":                 Vector3,       # contact position, base frame
    "n":                 Vector3,       # unit normal
    # physics
    "f_perp":            float,         # normal force
    "f_parallel":        Vector2,       # tangential force, in the tangent plane
    "slip_probability":  float,         # φ ∈ [0, 1]
    "mode":              enum,          # free / touch / sticking / sliding / rolling / separating
    # uncertainty (must be explicit — the biggest gap in the previous draft)
    "covariance":        Matrix,         # Σ, joint covariance over position / force / mode
    "confidence":        float,          # optional scalar summary
    # provenance
    "source":            enum,          # tactile / FT / proprio / vision / fused
    "evidence_mask":     [V, T, F, P],  # which channels contributed
    # time
    "timestamp":         float,         # observation instant on host clock
    "age":               float,         # now − timestamp
    "valid_from":        float,         # if interval-quantified (e.g. vision prediction)
    "valid_until":       float,
}
```

Four properties this schema is designed to hold:

- **Sensor-agnostic**: whatever is downstream — GelSight, taxel array, F/T — if it can fill these keys, it enters the state interface.
- **Physically interpretable with explicit units**: every field carries a dimensional quantity and a frame.
- **Uncertainty-aware**: covariance gives Bayesian fusion, safety layers, and fallback a first-class input. **A slot without uncertainty is not an interface — it is an assertion, and real robots never satisfy one.**
- **Provenance-bearing**: each cell records who contributed it and how fallback works when a stream is missing. This is what makes §7.5 modality dropout and §8.6 cross-modal contradiction expressible at the schema level, not the runtime level.

Beyond the contact set, the state interface at minimum also holds:

```text
wrench_ext          : 6D + covariance              # aggregate F/T, independent of C_t
robot_state         : (q, q̇, τ, EE pose)          # proprioception
task_context        : language token / goal embed  # high-level instruction, object identity
scene_state         : free-space / occlusion / …   # vision's non-contact semantics, optional
belief              : task-relevant latent / explicit
```

**Contact set is the core structured object of this interface, not its entire content** — this is the single most important scope tightening in the article.

### 6.2 Per-modality write paths — the key fix: F/T is a constraint, not a detector

The previous draft said "F/T decomposes onto existing $C_t$ entries; if there is no tactile, emit one aggregate contact entry." That is easily read as *"a 6D wrench can recover contact geometry"*, which is false. Concretely:

$$
w = \sum_{i=1}^{N} \begin{bmatrix} f_i \\ (p_i - p_0) \times f_i \end{bmatrix}
$$

Different sets of $\{p_i, f_i\}$ across multiple contacts can produce the **same** resultant wrench $w$. This is a classic **inverse problem, not a deterministic decoder**.

The correct slot semantics is that each modality contributes a **likelihood / constraint / proposal**, not a completed "fact":

```text
Tactile         → local contact observations       (direct measurement, sensor noise)
F/T             → global wrench constraint         (aggregate-level constraint, no localisation)
Proprioception  → kinematic / dynamic constraint   (J^T back-out, model-accuracy-limited)
Vision          → geometric prior                  (predicted contact hypothesis, not observation)
```

Combining them becomes a Bayesian problem:

$$
p(C_t \mid V, T, F, P) \;\propto\; p(V \mid C_t)\, p(T \mid C_t)\, p(F \mid C_t)\, p(P \mid C_t)\, p(C_t)
$$

Where:

- $p(T \mid C_t)$ is the **local observation likelihood** (tactile sensor model).
- $p(F \mid C_t)$ is the **global wrench consistency likelihood** (wrench reconstruction).
- $p(P \mid C_t)$ is the **dynamic consistency likelihood** (residual $\tau_{residual} = J^T F$).
- $p(V \mid C_t)$ is the **geometric / visual prediction likelihood** (affordance, pose estimation).
- $p(C_t)$ is a prior (e.g. contacts should lie on object surfaces).

This formulation is what grounds §6.3: **evidence hierarchy is hypothesis-dependent, not global**.

**A reviewer-perspective consequence**: the state interface stores not just "facts" but **beliefs with uncertainty**. The covariance per record in $C_t$ is the quantitative form of that.

### 6.3 Evidence ranking: hypothesis-dependent, not a fixed order

The previous draft's fixed order "tactile > F/T > proprio > vision" was wrong. The correct statement:

> **Evidence ranking should be hypothesis-dependent.**

For different quantities under discussion, the four streams have different directness:

```text
Contact location:    Tactile > Vision > F/T ≈ Proprio
Global wrench:       F/T > Tactile > Proprio > Vision
Object pose:         Vision > Tactile > F/T ≈ Proprio
Joint state / τ_res: Proprio >> others
Contact mode:        Tactile > F/T ≈ Proprio > Vision
Slip probability:    Tactile > F/T (rate) > Proprio (residual) > Vision
```

**Different fields within the same $C_t$ record can have entirely different dominant modalities** — which is why the interface needs `evidence_mask`. This upgrade turns the interface from a rule system into a **probabilistic evidence fusion**, directly corresponding to §6.2's Bayesian form.

### 6.4 Interface ≠ learned latent representation (new subsection)

This subsection carries the article's core architectural claim.

The previous draft said structured slots and shared latent are two branches, but did not spell out why they differ. Doing so now:

```text
Latent representation             State interface
─────────────────────────         ─────────────────────────
Optimisation target: downstream   Optimisation target: interoperability
                                   across many consumers
Units: implicit                   Units: every field has explicit SI / local dimension
Frames: implicit in data flow     Frames: declared, each field carries frame_id
Time: implicit                    Time: explicit timestamp + age
Uncertainty: implicit or in       Uncertainty: field-level covariance / confidence
  training loss only
Provenance: none                  Provenance: source + evidence_mask
Consumers: one model              Consumers: policy / world model / controller /
                                    diagnostic tool / safety layer / another sensor
```

A formula worth keeping:

$$
\text{Interface} \;=\; \text{Semantics} + \text{Units} + \text{Frame} + \text{Time} + \text{Uncertainty} + \text{Provenance}
$$

**A latent can be excellent for one policy while being unfit as an input for a world model, controller, diagnostic tool, or another sensor**. That gap is precisely what an interface exists to fill, and what §3.4's shared-latent line is currently missing. CLIP / ImageNet / COCO did not succeed because their embeddings are clever; they succeeded because they made semantics, units, frames, and time into a contract every downstream consumer can read. Tactile, F/T, and proprio need the same — **not a bigger model**.

### 6.5 When to compose: not at raw, not at decision, **at the state interface**

With §6.1's slot schema, fusion becomes three steps:

```python
# Step 1: per-modality perception + registration
#         temporal / spatial / semantic all resolved inside this step
raw_v  → enc_v → pred_contact_hypothesis      # vision prediction, with confidence
raw_t  → enc_t → detected_contact             # tactile observation, with covariance
raw_ft → enc_ft → external_wrench + residual  # compensated wrench, with covariance
raw_p  → proprio_state → J^T · τ_res → inferred_contact  # with model error

# Step 2: multimodal state estimation → written into state interface
#         §6.2 Bayesian fusion, §6.3 hypothesis-dependent ranking
C_t         = state_estimator(V, T, F, P)   # output is belief, not hard fact
wrench_ext  = compensate_ft(raw_ft)
robot_state = (q, q̇, τ, EE_pose)
belief_t    = belief_update(C_t, wrench_ext, robot_state, task_ctx)

# Step 3: policy / world model / controller consume state interface
a_t = policy(state_interface)
```

The key claim: **the output schema of Step 2 is the actual interface of the whole system**. Encoders in Step 1 and policies in Step 3 can be swapped freely as long as Step 2's schema is stable.

### 6.6 Missing-modality fallback: dropouts are part of the training distribution, not exceptions

Real deployments drop sensors as a matter of course. The interface must handle it explicitly:

```python
if tactile_missing:
    # in §6.2, p(T | C_t) collapses to uniform likelihood
    C_t = state_estimator(V, F, P)           # no T
    for r in C_t:
        r.covariance *= inflation_T_missing   # inflate
        r.evidence_mask.T = False
        r.provenance  = "no-tactile"
```

**This is not a runtime patch. It is a training-time constraint.** See §7.5 modality dropout and §8.6 cross-modal contradiction.

## 7. Fusion failure modes and diagnostics

Reviewer perspective: these five failure classes are the most common in robot multimodal papers, and the easiest to brush past with "we used cross-attention so we're robust". For each class, **symptom / diagnostic / mitigation**.

### 7.1 Modality collapse: attention quietly becomes vision-only

**Symptom**: training loss normal, validation success normal. But at eval time, masking out vision barely changes performance — meaning other modalities are effectively unused.

**Diagnostic**: visualise attention weight aggregated by modality across time; or, more decisively, run **modality ablation gain** (§8.1): if $\Delta_{\text{modality}}$ is near zero, that stream is dead weight to the policy. The 3D-ViTac [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) ablation matters precisely because it establishes this diagnostic as a serious reference.

**Mitigation**: (a) explicit modality dropout at training time — see §7.5; (b) per-modality auxiliary supervision (e.g. tactile encoder must independently predict slip events, not just feed the policy); (c) evaluate with 9/11 §5.2.1's feedback-value benchmark rather than raw success rate.

### 7.2 Temporal smearing: everything interpolated to 30 Hz

**Symptom**: high-frequency contact phenomena — slip, impact, transient force peaks — never learned.

**Diagnostic**: run the policy at 30-Hz-fused vs native-rate-fused and measure the delta; or look at $\Delta_{\text{tail}}$ restricted to difficult contact conditions.

**Mitigation**: follow §5.4 — treat time constants as modelling assumptions. Use different rates at different layers; route events on their own bus; never downsample high-rate tactile/F/T into a common slot.

### 7.3 Frame confusion: model learned spurious frame correlations

**Symptom**: model excels at the training pose, camera placement, and tool; nudges any of them and accuracy collapses.

**Diagnostic**: perturb frames at eval time — rotate camera extrinsics a few degrees, shift tool CoM a few grams, translate base a few centimetres; measure how far the policy falls.

**Mitigation**: state interface explicitly carries `frame_id`; consumers must transform before reading. Include frame perturbations in domain randomisation during training (Tobin et al. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)).

### 7.4 Semantic leakage: raw pixels contaminate slot semantics

**Symptom**: a slot defined as "contact geometry" behaves as if it depended on object appearance — swapping backgrounds drops performance.

**Diagnostic**: swap the visual contribution to a slot with a "minimal sufficient" synthetic slot (e.g. oracle contact prediction from simulator) and measure the delta.

**Mitigation**: enforce **single-path** encoding — raw pixels reach the policy only through the slot; no direct bypass. Aligns with 9/11 §3.5's "interface vs learned representation" split.

### 7.5 Missing / degraded modalities: dies on the first dropout, or silently degrades

**Symptom**: training sees all modalities every step; deployment drops one frame of tactile and the policy behaves wildly differently.

**Diagnostic**: run **modality masking / dropout tests** — at eval time, drop each modality independently with probability p, plot success-rate curve. Modality masking / dropout is now **a common family of training strategies** for missing-modality robustness; see Maiga et al. MMP [arXiv:2410.03010](https://arxiv.org/abs/2410.03010) and the missing-modality robustness literature.

**But dropout ≠ sensor failure**. Real degradation is almost never Bernoulli(p) independent dropping. Much more common:

```text
Modality missing     ── sensor unplugged
Modality stale       ── value present, but from tens of ms ago
Modality delayed     ── timestamp correct, arrival late
Modality corrupted   ── single-frame outlier (camera blur, spurious tactile signal)
Modality biased      ── systematic drift (F/T tare failure, IMU temperature)
Modality noisy       ── Gaussian or heavy-tailed noise increase
```

The interface must express all six, not a binary "present / absent" flag. §6.1's `timestamp / age / covariance / provenance` is what makes stale / delayed / biased detectable at the data-format level.

## 8. Benchmark and evaluation

Consolidate the §7 diagnostics into an executable benchmark skeleton. The design follows 9/11 §5.2.1's feedback-value stance — **the benchmark should measure "how much signal this modality uniquely contributes", not "how well the model memorised the training distribution"**.

### 8.1 Modality ablation gain

$$
\Delta_{\text{mod}} = S(\text{base} + \text{mod}) - S(\text{base})
$$

$S$ is task success rate; base is a fixed policy without that modality. Report $\Delta_{\text{mod}}$ per task family — small $\Delta_{\text{tactile}}$ for stable grasping is expected; large $\Delta_{\text{tactile}}$ for slip recovery, fine insertion, and wiping is *required*.

### 8.2 Missing-modality robustness

$$
R_{\text{rob}}(p) = \mathbb{E}_{\mathcal{D}}\big[\, S \,\big|\, \text{each modality dropped with prob } p \,\big]
$$

Sweep $p \in \{0, 0.1, 0.25, 0.5\}$ and plot per-modality dropout curves. **A healthy multimodal system's curves should be relatively flat**.

### 8.3 Degradation modes beyond dropout

Following §7.5, extend tests beyond Bernoulli dropout:

```text
missing      ── stream unplugged
stale        ── value retained but timestamp frozen
delayed      ── timestamp correct, arrival 20/50/100 ms late
corrupted    ── single-frame outlier injection
biased       ── fixed offset or slow drift added
noisy        ── σ progressively increased
```

Report per-class success-rate curves. **This is the direct test of whether uncertainty and provenance in the interface are actually being used**.

### 8.4 Temporal and frame perturbation

Add ±5 ms timing jitter to vision, ±20 ms to tactile, ±5 ms to F/T; observe $\Delta_{\text{tail}}$ (restricted to hard contact conditions). Frame perturbations work the same way. This is the direct quantitative counterpart to §7.2/7.3.

### 8.5 Slot fidelity measured independently

Since this article argues the interface is core, **slot accuracy should be measured independently of the policy**:

- contact position IoU / distance error against simulator ground truth;
- slip detection AUROC;
- contact-mode classification macro-F1;
- **covariance calibration** (e.g. reliability diagram — does the predicted Σ match empirical errors).

These are "quality before composition" metrics; if they are poor, high policy success rate almost certainly means overfitting.

### 8.6 Cross-modal contradiction test (new; the benchmark this article recommends most strongly)

If the whole point of an interface is **to let the system disagree productively when modalities disagree** — and to correctly inflate uncertainty rather than average into a confident wrong answer — then the benchmark should actively manufacture cross-modal contradictions.

$$
\mathcal{L}_{consistency} = d\big(\hat{C}^{T}, \hat{C}^{F}, \hat{C}^{V}, \hat{C}^{P}\big)
$$

Three contradiction classes to inject:

```text
Spatial disagreement   ── vision says contact at A, tactile says at B (5 / 10 / 20 mm apart)
Temporal disagreement  ── F/T wrench has dropped, tactile still reports sliding
Semantic disagreement  ── vision classifies as "rigid object", tactile local deformation matches "compliant"
```

Then measure four things:

1. **Contradiction detection** — did the system flag that "these channels disagree"?
2. **Uncertainty calibration** — does reported uncertainty scale with actual discrepancy?
3. **Source attribution** — post-hoc, can it identify which channel was wrong?
4. **Recovery** — how far does success rate fall after handling?

This benchmark is arguably **a better fit for the "interface" thesis than modality dropout**, because it directly stresses the Uncertainty and Provenance terms in §6.4's formula Interface = Semantics + Units + Frame + Time + Uncertainty + Provenance. For an interface built the way this article suggests, **§8.6 should be a first-class metric, ahead of §8.2**.

### 8.7 Fit with existing benchmarks

RoboCasa, LIBERO, ManiSkill3, and BEHAVIOR are largely vision-centric with tactile absent. This section makes no sweeping verdict but **suggests** adding a feedback-value plugin to existing benchmarks — the six metric families above (ablation / dropout / degradation modes / temporal+frame perturbation / slot fidelity / contradiction) — and borrowing 9/11 §5.2.1's A/B/C/D arms directly.

## 9. Relation to VLA and world models

### 9.1 VLA today: what's missing is a same-tier pretraining ecosystem, not a concat channel

§3.1 already corrected the record: describing RT-2 / OpenVLA / π0 as "vision + proprio + language concat" is inaccurate. Their input structures differ meaningfully — π0 has explicit proprio token and action chunk, OpenVLA supports multi-camera / depth / state encoding, RT-2 represents actions as text tokens.

A more accurate framing:

> **Mainstream VLAs scale around vision-language observation and robot state / action; tactile and F/T have not yet formed a pretraining ecosystem of comparable scale or standardisation.**

The absence is not oversight. Adding tactile reopens §2 wholesale, and tactile / F/T lack both ImageNet-scale datasets and a §6.1-style state-interface schema. The real open question is: **does the next VLA wave stack more vision + language, or finally add tactile / F/T slots?** This article bets on the latter — but not by adding tactile as "just another channel"; by inserting §6's state interface into the VLA input layer. Qi et al.'s T-Dex [arXiv:2309.09979](https://arxiv.org/abs/2309.09979) (CoRL 2023) and Lee et al. [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) can be read as early sketches of that route.

### 9.2 World models: don't smear contact events into a single continuous latent

RSSM / DreamerV3 [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) latent dynamics typically assume smooth, differentiable transitions. 9/11 §5.1 already noted that contact is essentially a hybrid-dynamics mode switch.

**A more careful framing** (the previous draft's "RSSM inevitably smooths mode switches" is too absolute; modern world models can use discrete latent / categorical latent / hybrid state / event-conditioned dynamics / multiple latent heads / mode-conditioned transition):

> **If a world model represents contact state with a single continuous latent dynamics and lacks an explicit mode / event variable, then contact-induced discontinuities are more likely to be represented as smooth latent-space evolution.**

Conditional, not deterministic. The mitigation is exactly §6's state interface:

- **World-model layer keeps structured contact slots**: mode is discrete, $C_t$ is a set, $f^{\perp}/f^{\parallel}$ are real. Do not stuff these three into a single continuous latent.
- **Policy layer may consume a learned representation**: embedding the slot for policy use is fine; but that embedding is a *downstream consumer*, not an *upstream data format*.
- **Interface between them is §6**: world model predicts the next-timestep slot, policy reads the current slot.

This mirrors 9/11 §5.1's belief-update framing — the same structure, generalised across modalities.

### 9.3 One sentence to bind both

**VLAs are missing tactile state interfaces; world models are missing contact structure. These are the same underlying failure — everyone has been hoping that piling on data will teach the model the interface, so nobody bothers designing it.**

## 10. Conclusion

This article started from the common misconception that multimodal fusion is a "model architecture" question, and pulled it back to foundations: the real difficulty is not **which attention to pick**, but **whether there is a structured state interface, consumable by policy, world model, and controller alike, with timestamps, frames, uncertainty, and provenance in its schema**. Vision grew its common data interface not because its encoders are cleverer, but because it locked down §2's registration and semantic-projection steps first — and layered on top an ecosystem of GPUs, cheap sensors, internet-scale data, annotation, and standard formats. What it produced is **a family of highly interoperable task-level representation conventions, not a single unified schema**. Tactile, F/T, and proprio are still not converged on those steps; every cross-attention or shared-latent design will keep hitting the failure classes in §7.

This article is not against cross-attention, and not against shared latent. It is against **one policy network being asked to do state estimation, sensor fusion, and control at the same time**. The minimum viable route proposed: **a contact-set-centric structured state interface for contact-rich manipulation, sensor-specific perception feeding raw → slot, the interface itself carrying uncertainty / provenance / timestamp / age, policy and world model composed only at the interface layer, and modality masking / dropout plus cross-modal contradiction tests treated as first-class citizens in training and benchmark**.

**This route is not a final answer — it is a direct response to the open question 9/11 left behind**: "tactile has no reusable intermediate representation". Design the interface first; then talk about architecture.

If 9/11's three tags were *Action-conditioned observation · Contact-state representation · Closed-loop value*, this article pins three:

> **Alignment before fusion · Interface before architecture · Missing modalities by design**

One sentence tying them together: **before fusing, do registration and semantic projection; before choosing architecture, define the state interface; in training and benchmark, treat missing modalities and cross-modal contradictions as part of the distribution, not exceptions**.

Next up (9/13) we descend one level from "interface": **dexterous hands and in-hand manipulation** — how these slots actually look in T-Dex / DextrAH / LEAP settings, and why "many robots have hands, few do real dexterity" reflects a cost structure, not an algorithm gap.

## Sources

Citations here are grouped by **thesis** rather than "more sensor papers"; no padding.

### A · Cross-modal representation and fusion paradigms (supporting §3 / §4)

- Baltrusaitis, Ahuja, Morency, *Multimodal Machine Learning: A Survey and Taxonomy*, TPAMI 2019 · [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) (the classic multimodal-fusion taxonomy · reference frame for §3)
- Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language Sequences*, ACL 2019 · [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) (an early explicit attempt at temporal misalignment · §3.2)

### B · The visuo-tactile empirical line (supporting §4 / §7.1 / §9.1)

- Calandra et al., *More Than a Feeling: Learning to Grasp and Regrasp using Vision and Touch*, RA-L 2018 · [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) (one of the earliest visuo-tactile regrasp experiments; the original "tactile pays off in long-tail cases" evidence)
- Lee et al., *Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks*, ICRA 2019 · [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) (early visuotactile self-supervised representation; an anchor for §3.4 shared latent)
- Huang et al., *3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing*, CoRL 2024 · [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) (explicit "vision + tactile > vision-only" evidence · reference point for §7.1 modality-collapse diagnostic)
- Qi et al., *General In-Hand Object Rotation with Vision and Touch* (T-Dex), CoRL 2023 · [arXiv:2309.09979](https://arxiv.org/abs/2309.09979) (active tactile exploration + visuotactile fusion · a concrete shape of the §9.1 VLA-plus-tactile route)

### C · Cross-sensor / cross-modal unified representations (supporting §3.4 / §5.1)

- Zhao et al., *A Touch, Vision, and Language Dataset for Multimodal Alignment* (TVL / Binding Touch to Everything), ICML 2024 · [arXiv:2402.13232](https://arxiv.org/abs/2402.13232) (about 44K vision-touch pairs; tactile-VL alignment; representative of the "tactile CLIP" route)
- Feng et al., *AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors*, 2025 · [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) (cross-heterogeneous-tactile unified representation · contrast to §5.1 raw-layer non-uniformity)
- Lin et al., *9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation*, ICRA 2023 · [arXiv:2308.14277](https://arxiv.org/abs/2308.14277) (a concrete tactile modality; 3D shape reconstruction and 6D force sit at the **estimation layer** · evidence for §5.1 heterogeneity)

### D · Robustness and modality masking / dropout (supporting §7.5 / §8.2 / §8.3)

- Maiga et al., *MMP: Towards Robust Multi-Modal Learning with Masked Modality Prior Fine-Tuning*, 2024 · [arXiv:2410.03010](https://arxiv.org/abs/2410.03010) (modality masking as a training strategy; one of several common approaches to missing-modality robustness · §7.5 mitigation reference)
- *Robust Multimodal Learning with Missing Modalities via Parameter Projection*, 2023 · [arXiv:2310.03986](https://arxiv.org/abs/2310.03986) (representation alignment under missing modalities · reference for §8.2 robustness metric)

### E · VLA and world models (supporting §9)

- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) (VLM backbone + proprio token + noisy action chunk + flow matching · corrected description for §3.1 and §9.1)
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) (open-source VLA baseline; public configs include multi-camera / depth / proprio state encoding, not a naive V+L concat)
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) (actions expressed as text tokens co-trained with the VLM · corrected description in §3.1)
- Hafner et al., *Mastering Diverse Control Tasks through World Models* (DreamerV3), Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) (a latent-dynamics world-model representative; the contrast surface for §9.2's conditional claim)

### F · Sim-to-Real / domain randomisation background (supporting §7.3)

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) (classical recipe for treating frame perturbation as domain randomisation)

### G · Continued from 9/11 · Contact state and impedance (background)

- Already cited in 9/11 and reused here: Hogan's impedance trilogy, Posa-Cantu-Tedrake IJRR 2014 (hybrid contact-mode trajectory optimisation), Lee 1810.10191, Qi 2309.09979, Huang 2410.24091, Zhao 2402.13232, Feng 2502.12191. This article will not re-list their links; see 9/11's Sources for exact references.

---

> **Related reading**
>
> - [The Hand Robots Don't Have: Tactile and Force Sensing in Embodied AI](/en/articles/2026-09-11-tactile-force-sensing/) — the predecessor; tactile and force sensing in isolation
> - [Sim-to-Real Methodology](/en/articles/2026-09-10-sim-to-real-methodology/) — frame perturbation (§7.3) and temporal perturbation (§8.4) borrow its domain-randomisation lens
> - [Why Robot Data Is Harder than LLM Data](/en/articles/2026-09-09-robot-data-scaling/) — the "where does supervision come from" problem in §3.4 shared latent is really a data-scaling problem
> - [VLA and World Models](/en/articles/2026-09-07-vla-world-models/) — §9 is a concrete side of that broader comparison: VLA missing tactile state interface, world models missing contact structure
