---
title: 'Stacking Sensors Is Not Fusing Them: Multimodal Robotics Lacks an Interface, Not a Model'
slug: "2026-09-12-multimodal-fusion-interface"
date: 2026-09-12
draft: false
categories: ["Embodied AI", "Multimodal Perception"]
tags: ["Embodied AI", "Multimodal Fusion", "Visuo-Tactile", "Force/Torque", "Proprioception", "Representation Interface", "Cross-attention", "Modality Dropout", "VLA", "World Model", "Frame Alignment", "Time Alignment"]
description: 'Multimodal fusion is usually presented as a question of "which attention architecture", but in robotics the real bottleneck is upstream: four streams (Vision / Tactile / Force-torque / Proprioception) have never agreed on a time base, a frame base, or a semantic base. This piece splits "fusion" into *alignment* and *composition*, argues that vision won precisely because it locked down all three bases, and shows why tactile / force-torque / proprioception keep failing in cross-attention and shared-latent designs — until we stop pretending that concatenation is a substitute for an interface. A concrete minimum viable route is proposed: use a contact set as the cross-modal representation interface, keep per-modality raw→slot decoding inside sensor-specific front-ends, let policy and world model compose only at the slot layer, and treat modality dropout as a first-class citizen in both training and benchmarking.'
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

Picture a bimanual robot fitted with an RGB camera, GelSight fingertips, wrist six-axis force/torque sensors, and joint encoders on every link. The hardware bill of materials looks genuinely "multimodal". Yet when you hand that rig to a policy, most published work tells you "just fuse them with cross-attention and you're done". That runs in a demo; **in real deployment it will almost certainly fail in one of four ways**: a modality drops frames, a sensor dies, coordinate frames drift, or the model quietly converges to one dominant modality and treats the rest as noise. These four are not engineering footnotes. **They all trace back to the same root cause: the modalities never agreed on an interface that is alignable, composable, and consumable by a real-time controller.**

This piece takes on the easiest topic to hand-wave in embodied AI: **multimodal fusion**. It will not sell you a specific network. It will instead dig fusion down to its three foundations — **time, frame, semantics** — and use them to explain **why vision was able to grow a common data interface while tactile / force-torque / proprioception have not, and what a minimum usable interface looks like if you want to start building today**.

## 0. Framework: Split "fusion" into alignment and composition

Let's put the whole article's frame up front. Every section returns to this picture.

```text
                    ┌───────────────────────────────┐
                    │         Raw sensor streams    │
                    │  V · T · F · P  (4 channels)  │
                    └──────────────┬────────────────┘
                                   │  ①  alignment
                       ┌───────────┼───────────┐
                       ▼           ▼           ▼
                    Time base   Frame base  Semantic base
                   (t, Δt, τ)  (SE(3) +   (raw → contact
                                camera FK)  → mode → belief)
                       └───────────┼───────────┘
                                   │  ②  composition
                                   ▼
                       Structured interface  (slots)
                     {contact_set, wrench, robot_state,
                      task_context, belief}
                                   │
                       ┌───────────┼───────────┐
                       ▼           ▼           ▼
                     Policy    World Model   Controller
                       │           │           │
                       └───────────┼───────────┘
                                   ▼
                              Action a_t
```

The top half — **alignment** (time / frame / semantics) — is the **protagonist** of this article, and is precisely the step most multimodal papers skip. The bottom half — **composition** (attention, shared latent, structured slots) — is comparatively mature. The core judgment of this article is: **the reason robot multimodality has not yet produced a general paradigm is the top half, not the bottom half**.

If the three tags pinned in 9/11 were "the three hard problems tactile faces on its own", this article pins three for multimodal fusion:

> **Alignment before fusion · Structured contact slots · Modality dropout as first-class**

Every section elaborates on these three.

## 1. "Multimodal" is not "many sensors"

The easiest way to derail the topic is to conflate "many sensors" with "many modalities" — and many paper introductions do exactly that. This conflation does not hold.

**A modality is properly defined by locking down four things together**: measurement space, physical origin, noise model, and update semantics. Only when all four agree can two data streams be treated as the same kind of thing.

A few counterexamples:

- Wrist camera + overhead camera → **same modality, multiple viewpoints**, not two modalities. Measurement space, physical origin, and noise model essentially match; only extrinsics differ; one encoder handles both.
- Four GelSight fingertip cameras → **an internal structure of one tactile modality**, not four visual modalities. The measurement space is "surface deformation field of an elastomer", not scene RGB. Even if a CNN is involved, downstream semantics are contact geometry, not object detection.
- Joint encoder + motor current → both are essentially **one modality seen from two points** (one directly measures robot state, the other infers it). Treating them as two modalities to be fused only teaches attention redundancy.

Conversely, these are **genuinely different modalities**:

- Vision (RGB / RGB-D / event camera) — light-radiance field, camera frame, 15–60 Hz.
- Tactile (GelSight / GelSlim / TacTip / 9DTact / taxel arrays) — contact-interface deformation or force distribution, sensor frame, 100 Hz – kHz.
- Force/torque (wrist F/T, six-axis wrench) — **aggregate** contact wrench, sensor frame + tool frame, 500 Hz – 1 kHz.
- Proprioception (joint angle q, velocity q̇, torque τ, end-effector pose) — the robot's own state, base / world frame, 1 kHz+.

Some works add audio (microphone arrays), thermal, gas, or ultrasound as fifth / sixth modalities. The classification logic is the same.

**Where this section lands**: before talking about fusion, write out the modality list and pin the four defining properties for each. Skip this and every downstream fusion architecture is built on air. The taxonomy tree in 9/11 §2.1 splits contact sensing into Tactile / Force-torque / Proprioceptive branches — this article keeps that same three-way split and confines the discussion to **V + T + F + P**.

## 2. Three foundations before fusion: time, frame, semantics

This is the most technical section and the easiest to skim past. But **if any one of these three is undefined, everything downstream is a castle in the air**.

### 2.1 Time base

The default time constants of the four streams are separated by **multiple orders of magnitude**:

```text
Vision           ~30 Hz       Δt ≈ 33 ms      frame triggered / polled
Tactile (image)  30–200 Hz    Δt ≈ 5–33 ms    frame triggered
Tactile (taxel)  500 Hz – kHz Δt ≈ 1 ms       scan triggered
Force/torque     500 Hz–1 kHz Δt ≈ 1–2 ms     polled
Proprioception   1 kHz+       Δt ≈ 0.5–1 ms   hard real-time
```

The standard engineering move is to interpolate everything to the lowest rate — usually vision. **This looks harmless but silently erases the high-frequency physics of tactile, force, and proprioception.** 9/11 already discussed the point: slip detection, transient contact forces, joint impacts — these are event-like signals that often fire between two vision frames. Interpolating to 30 Hz throws them away.

Three further wrinkles:

- **Timestamp semantics**: sensor timestamp (the exposure or sample instant) vs arrival timestamp (when software actually sees it) vs host timestamp (the fusion clock). The three can differ by 5–20 ms — enough to shift a "cup contact moment" outside the finger-closure window.
- **Clock drift**: cameras run their own crystal, robot controllers run another, host PCs a third. Tens of milliseconds drift in minutes, enough over a night to teach a fusion model wrong lead/lag correlations. NTP/PTP only fixes things at the system layer; at the sensor layer you still need hard triggers.
- **Event-driven vs polled**: tactile slip detection is a sparse event stream, not a periodic sample. Stuff it into a uniform time buffer and you lose the density signal.

**The judgment**: the fusion layer must **acknowledge different time constants and choose the alignment layer per task sensitivity**. Vision policy can run 30 Hz; force/tactile-driven compliance must stay at native rate; events need their own bus. Freezing all of these to one unified time base is a **modeling assumption**, not an engineering detail.

### 2.2 Frame base

The four streams natively live in four different frames:

```text
Vision           camera frame (RGB in pixels + depth in camera)
Tactile          sensor frame (local frame attached to fingertip surface)
Force/torque     sensor frame (usually wrist-mounted, must be transformed to tool/base)
Proprioception   base frame / world frame
```

To place them in one fusion layer, at minimum three things must be true:

- **Hand-eye calibration**: $T^{cam}_{base}$ is SE(3); a single collision can nudge it by fractions of a degree or a few degrees. Probably fine for a coarse grasp; immediately catastrophic for fine contact manipulation.
- **Sensor-mount calibration**: fingertip sensor frame → link frame, $T^{sensor}_{link}$. This is often quietly waved away with "we assume it was mounted perfectly", but optical tactile sensors like GelSight are extremely sensitive to it.
- **Wrench transformation and gravity compensation**: an F/T sensor reads wrench in its own frame; you must transform to base or world and subtract tool gravity plus inertial terms. Any step wrong and every downstream normal/shear decomposition is broken.

**A common symptom of a bad frame base**: the trained model works at the training pose, training camera placement, and training tool — swap any of them and accuracy collapses. The model was actually learning correlations in a particular sensor frame, not the task itself.

A subtler issue is **relative vs absolute pose**: contact physics fundamentally depends only on who-is-touching-whom, but many policies feed absolute end-effector pose in world frame, forcing the model to learn degrees of freedom it should not need. This is the same judgment as 9/11 §3.3: what a policy needs to be consistent about is the transition of contact modes, not the absolute value of every physical parameter.

### 2.3 Semantic base

The four streams are **semantically misaligned by nature**:

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

**The central multimodal-fusion question is: at which layer do you fuse?** A common mistake is fusing at the raw layer — concatenate four tensors and throw them into a network. That forces the policy to learn all three bases from scratch — enormously expensive and sample-inefficient. A more reasonable choice is to **fuse at the slot layer**: let each modality finish its own raw → perception → contact-geometry chain, and only compose at "contact events, external wrench, robot state, task belief", where the semantic base has already converged.

**Landing**: time base, frame base, and semantic base must be resolved **before fusion**. Hand-wave any one of them and the fanciest fusion architecture is just paying back that debt later.

## 3. The genealogy of existing fusion paradigms

When the literature talks about "multimodal fusion", it usually means **how to organize signals at the composition step**. This section walks the mainstream designs by *when* fusion happens, and calls out failure modes along the way. The classic Baltrusaitis et al. survey [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) taxonomises multimodal ML into representation / learning / feature selection / fusion / application; this section borrows the fusion sub-taxonomy.

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

Early concat is still the default inside most modern VLAs — except most of them only concat vision + proprioception + language and quietly skip tactile / F/T. π0 [arXiv:2410.24164](https://arxiv.org/abs/2410.24164), OpenVLA [arXiv:2406.09246](https://arxiv.org/abs/2406.09246), and RT-2 [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) are all "vision + proprio + language" concat. That fact alone tells you what's going on: **mainstream VLA still hasn't seriously added tactile / F/T**, and it's not because authors forgot — adding them forces the whole §2 discussion onto the table.

### 3.2 Cross-attention / Transformer fusion

One step further: treat each stream as a token (or short token sequence), then run self- or cross-attention on top. This is the template used by most of the vision-language-action lineage. Tsai et al.'s Multimodal Transformer [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) is a canonical starting point, explicitly designed for temporally unaligned modalities.

Upside: the architecture acknowledges that cross-modal alignment is *learned*, giving soft alignment room to emerge. It can also ingest irregular timestamps.

Downside:

- Implicit attention alignment under data scarcity collapses into **dominant-modality collapse** — weights concentrate on vision, other modalities become regularisation noise. Both 3D-ViTac and TVL observe this explicitly (see §7).
- Sample hungry. Vision pretraining carries the vision token; tactile and F/T do not yet have comparable pretrained encoders, so attention gets no stable modality-specific representation to work with.
- The time-alignment problem is partially hidden by learning, **but does not disappear** — it just shifts from modelling to training. Engineering cost goes up.

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

- TVL / Binding Touch to Everything [arXiv:2402.13232](https://arxiv.org/abs/2402.13232) (Zhao et al., ICML 2024) — aligns tactile with vision-language at the representation level; hard to avoid in any "tactile foundation model" discussion.
- AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) (Feng et al., 2025) — attempts a unified static-dynamic representation across **heterogeneous tactile sensors**, effectively a "tactile-internal" shared latent.
- 3D-ViTac [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) (Huang et al., CoRL 2024) — visuo-tactile joint representation for fine-grained insertion; provides a clear "vision + tactile > vision-only" empirical evidence.
- Lee et al.'s Making Sense of Vision and Touch [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) (ICRA 2019) — an early anchor for visuotactile self-supervised alignment.

Upside: representation is reusable, with a high ceiling across tasks and embodiments.

Downside: **where does the supervision signal come from?** Vision-language contrastive learning has billions of web image-text pairs. For a four-way vision-tactile-force-proprioception alignment there is no natural supervision source. Current mitigations are (a) treat "close in time" in teleoperation logs as weak alignment supervision — Calandra et al.'s More Than a Feeling [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) is an early tactile-grasp instance of exactly this idea — or (b) route via language/vision as a bridge and pull tactile into the V-L space (TVL does this). Both are far from scale.

**9/11 §3.5 already split representation into "interface" and "learned representation".** Shared latent belongs to the latter branch, and it only stabilises once the former already exists. §6 makes that concrete.

### 3.5 Structured contact slots: the interface route this article recommends

The last option: **do not fuse at raw and do not fuse inside a latent — compose at an explicitly designed intermediate representation**. That representation is not a learned embedding; it is a set of **slots whose semantics have already converged**. This is the direct answer to 9/11 §3.4's observation that vision has a common data interface while tactile does not.

The specific slot design is laid out in §6; here is the rough shape:

```text
state_slots = {
    contact_set      : list of { p, n, f⊥, f∥, φ_slip, mode }   # produced jointly by tactile + F/T
    wrench_ext       : 6D                                       # produced by F/T alone
    robot_state      : (q, q̇, τ, EE pose)                       # produced by proprioception
    task_context     : language token / goal embedding           # high-level instruction
    belief           : task-relevant latent / explicit           # state estimation
}
```

Each of the four streams fills the keys it can; "fusion" reduces to "writing into the same dict"; policy and world model both read from the dict. Advantages:

- Losing a stream only affects the keys it writes to; other keys stay clean. Modality dropout becomes natural.
- Time base, frame base, and semantic base are all resolved **inside the slot definition**; the composition architecture can then be swapped freely.
- Directly compatible with 9/11 §5.2.1's feedback-value ablation — arms A/B/C/D literally mean "write fewer keys into the dict".

**Downside**: designing the slot is heavier than "throw a Transformer at it". If the slot fidelity is poor, downstream policy cannot exceed it.

**Landing**: within the genealogy, early concat / attention / late fusion / shared latent each have a place, but **almost all of their observed failure modes can be traced back to §2's three bases**. This article's stance is to move engineering attention from "which attention to pick" back to "define the slot first".

## 4. Why vision got there first

A natural follow-up: §2's three bases exist for vision too, so why was vision able to grow a common data interface anyway? A historical-engineering answer, also useful as a mirror for tactile.

**On the time base**: video is frame-indexed by nature. 30 Hz or 60 Hz. Every downstream task — classification, detection, segmentation, SLAM — has agreed "frames are the unit". This sounds trivial but it **eliminates the hardest class of problems in fusion discussion**: everything image-based lives on the same time axis.

**On the frame base**: pinhole model + camera intrinsics + extrinsics turn "image pixel ↔ world point" into one formula, $s \cdot m = K [R | t] \cdot M$. Every 3D vision, SLAM, NeRF, 3D Gaussian Splatting, and multi-view stereo method grows on this convention. Calibration errors exist, of course, but **what they look like is predictable and reproducible**.

**On the semantic base**: raw RGB has no semantics, but the vision community spent the last decade using COCO / ImageNet / ADE20K / LVIS to lock down the "semantic slots": object class, bounding box, instance mask, depth map, affordance, caption. These are not learned — **they are community conventions**. That's why any new vision model interoperates with existing ones from day one.

Contrast with tactile:

- Time base: 30 Hz to kHz across sensor families; event- and poll-triggered streams mixed; no consensus.
- Frame base: GelSight is pixel + elastomer deformation; 9DTact is pixel + 3D deformation field; taxel arrays are 1D/2D force distributions; optical waveguides (AnySkin / DigiTact) are a different topology entirely. There is no agreed shape for "one tactile reading".
- Semantic base: contact point, normal, shear, slip, mode appear across papers, **but no cross-dataset unified schema**. 9DTact calls it "6D force", GelSight calls it "shear map", arrays call it "taxel load" — three projections of the same physical quantity.

**Landing**: vision won not because its encoders are cleverer, but because it **already resolved §2's three bases** — every subsequent fusion discussion inherits a shared floor. Tactile, F/T, and proprio need to do the same before any fusion ecosystem at scale becomes possible. That means conventions first, architecture second.

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

A common misconception: "then build a tactile foundation model that unifies all sensors". That is precisely what AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) is trying to do, and the direction is right — **but it unifies the learned-representation layer, not the raw layer**. Even with AnyTouch, its output is still an embedding; you still need an explicit "embedding → contact slot" convention to enter fusion. 9/11 §3.5 split "interface" from "learned representation" specifically to keep these layers from collapsing into one another.

**Fusion pain points**:

- Different sensor families differ in time constant and trigger semantics; time base must be handled per case.
- Sensor-frame definitions disagree (GelSight is pixel, arrays are index, waveguides are topology).
- Output semantics disagree (image / force field / 3D deformation).

**Mitigation**: introduce a **sensor-specific decoder** between raw and slot that unifies heterogeneous tactile outputs into $(p, n, f_\perp, f_\parallel, \phi_{\text{slip}}, \text{mode})$. It can be analytic or learned, but it must be **part of the interface**, not buried inside the policy.

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

There is also a subtle but important distinction: F/T gives an **aggregate** wrench, not a spatially distributed contact. A single grasp might have three contact points with different normal/shear; F/T only reports the sum. This aggregate-vs-field distinction is exactly why 9/11 §2.1's taxonomy puts tactile and force/torque in separate branches.

**Fusion pain point**: F/T has "clean" semantics (six numbers) but *relies on extensive compensation upstream*, and that compensation depends on tool models and dynamics. Change the tool and the downstream fusion distribution changes with it.

**Mitigation**: F/T should produce **two** things at the slot layer — (a) the compensated external wrench (for force-aware policies), and (b) residual magnitude (for drift detection and unexpected-collision flags). The second is often dropped.

### 5.3 Proprioception: the "background" channel that actually dominates

Among the four, proprioception is the fastest and the **only one the robot natively carries in full**. But it has an awkward status: **many policies concatenate (q, q̇, τ) directly into state without ever calling it a "modality"**, which is why proprio is often missing from multimodal-fusion discussions. That is a mistake.

Proprio plays two roles in fusion:

- **As contact-hypothesis input**: given a commanded end-effector trajectory and joint torques, external contact force can be recovered via $J^T \hat{F}_{ext} = \tau_{residual}$ (contact inference — a classic tool from the Hogan impedance lineage). This lets proprio contribute to the contact slot, not just to its own channel.
- **As time and frame anchor**: fusion needs a stable frame. Proprio runs 1 kHz hard real-time, making it the most "clock-like" of the four; EE pose can serve as an anchor for camera and sensor frames alike.

**Fusion pain point**: proprio is "too clean" — it measures the robot's own state without external-world uncertainty. Models tend to **over-rely** on proprio, learning it as a shortcut. Result: fine in non-contact scenes, but under-using tactile / F/T exactly when contact matters. This is precisely what 9/11 §5.2.1 feedback-value ablation guards against — the evaluation must measure "with proprio already given, how much does tactile/F/T add", not "tactile-only vs vision-only".

### 5.4 Different time constants are themselves modelling assumptions

Combining §5.1–5.3 with §2.1:

> **Different modalities running at different time constants is not an engineering "how do we fuse them" problem, but a modeling assumption about "what control bandwidth the task actually needs".**

Two examples:

- **Wiping / polishing**: force-control bandwidth is at least 100–500 Hz, vision at 30 Hz is fine, tactile and F/T must stay native, proprio needs torque-level rates. Fusion cannot be downsampled to 30 Hz.
- **Pick-and-place / insertion**: vision dominates, contact events are sparse; downsampling tactile to vision rate is an acceptable approximation.

This is the same judgment as 9/11 §4.1's refusal to write down specific ms numbers: "concrete rates depend on policy architecture, hardware servo, controller, and compute budget".

## 6. A minimum viable fusion interface

Time to answer "what should we actually build". This section proposes an interface that is implementable, benchmarkable, and incrementally evolvable.

### 6.1 A contact-set-centric intermediate

Define a cross-modality shared **contact-event set**:

$$
C_t = \big\{\, \big(\, p_i,\; n_i,\; f_i^{\perp},\; f_i^{\parallel},\; \phi_i^{\text{slip}},\; m_i \,\big) \,\big\}_{i=1}^{N_t}
$$

Here $p_i$ is contact position (in base frame), $n_i$ is surface normal, $f_i^{\perp}$ / $f_i^{\parallel}$ are normal and tangential force magnitudes, $\phi_i^{\text{slip}} \in [0,1]$ is slip probability, and $m_i$ is discrete contact mode (free / touch / sticking / sliding / rolling / separating). $N_t$ is the current number of active contacts and varies with the task.

The definition deliberately achieves three properties:

- **Sensor-agnostic**: whether downstream is GelSight, taxel array, or F/T, as long as it can fill these keys it enters fusion.
- **Physically interpretable**: each of the six numbers has a clear physical meaning — useful for diagnostics and safety layers.
- **Directly consumable by controllers**: impedance, hybrid force-position, grasp-force optimisation, and QP-based whole-body controllers can all read $C_t$ without an extra decoder.

### 6.2 How each stream writes into the slot

```text
Vision          ──► predicted contact hypothesis (soft, from affordance + grasp planner)
                      fills the "predicted" slot, low confidence
Tactile         ──► detected contact instance (hard, via sensor-specific decoder)
                      fills the "detected" slot, high confidence, position accuracy sensor-dependent
Force/torque    ──► aggregate wrench → distributed over existing C_t
                      if only F/T is available, emit one "aggregate" entry
Proprioception  ──► back-out external contact via J^T
                      fills the "inferred" slot, accuracy depends on dynamics model
```

Multiple streams write into the same slot. When they disagree, merge by **evidence hierarchy**: (tactile detected) > (F/T-derived) > (proprio inferred) > (vision predicted). Not a physical law, but an explicit and explainable default.

### 6.3 When to fuse: not at raw, not at decision, **at the slot**

With §6.1 defined, fusion becomes three steps:

```python
# Step 1: per-modality preprocessing → sensor-specific encoder
#         (time / frame / semantics are resolved inside this step)
raw_v  → enc_v   → pred_contact_hypothesis
raw_t  → enc_t   → detected_contact
raw_ft → enc_ft  → external_wrench + residual
raw_p  → proprio_state → J^T · τ_res → inferred_contact

# Step 2: composition into slots (structured slots, this article's recommendation)
C_t        = merge(pred, detected, wrench_split, inferred)
wrench_ext = raw_ft.compensated
robot_state= (q, q̇, τ, EE_pose)
belief_t   = belief_update(C_t, wrench_ext, robot_state, task_ctx)

# Step 3: policy / controller consumes the slot layer
a_t = policy(C_t, wrench_ext, robot_state, belief_t, task_ctx)
```

The key claim: **the output schema of Step 2 is the actual interface of the whole system**. Encoders in Step 1 and policies in Step 3 can be swapped freely as long as Step 2's schema is stable.

### 6.4 Missing-modality fallback: dropouts are part of the training distribution, not exceptions

Real deployments drop sensors as a matter of course. The interface must handle it explicitly:

```python
if tactile_missing:
    C_t = C_t without detected_slot
         + fill_from_ft_and_proprio()
         + inflate_uncertainty()
```

**This is not a runtime patch. It is a training-time constraint.** See §7.5 modality dropout.

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

**Mitigation**: use relative quantities at the slot layer ($p_i$ in EE frame rather than world, wrench in tool frame rather than sensor frame). Include frame perturbations in domain randomisation during training (Tobin et al. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)).

### 7.4 Semantic leakage: raw pixels contaminate slot semantics

**Symptom**: a slot defined as "contact geometry" behaves as if it depended on object appearance — swapping backgrounds drops performance.

**Diagnostic**: swap the visual contribution to a slot with a "minimal sufficient" synthetic slot (e.g. oracle contact prediction from simulator) and measure the delta.

**Mitigation**: enforce **single-path** encoding — raw pixels reach the policy only through the slot; no direct bypass. Aligns with 9/11 §3.5's "interface vs learned representation" split.

### 7.5 Missing-modality brittleness: dies on the first dropout

**Symptom**: training sees all modalities every step; deployment drops one frame of tactile and the policy behaves wildly differently.

**Diagnostic**: run **modality dropout tests** — at eval time, drop each modality independently with probability p, plot success-rate curve. This is already standard in robust multimodal learning; see Maiga et al. MMP [arXiv:2410.03010](https://arxiv.org/abs/2410.03010) and the missing-modality robustness literature.

**Mitigation**: make modality dropout a first-class training-time citizen; carry an explicit uncertainty flag at the interface (§6.4); report "missing-modality recovery" as its own benchmark column (§8.2).

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

### 8.3 Temporal and frame perturbation

Add ±5 ms timing jitter to vision, ±20 ms to tactile, ±5 ms to F/T; observe $\Delta_{\text{tail}}$ (restricted to hard contact conditions). Frame perturbations work the same way. This section is the direct quantitative counterpart to §7.2/7.3.

### 8.4 Slot fidelity measured independently

Since this article argues that the slot is the interface, **slot accuracy should be measured independently of the policy**:

- contact position IoU / distance error against simulator ground truth;
- slip detection AUROC;
- contact-mode classification macro-F1.

These are "quality before fusion" metrics; if they are poor, high policy success rate almost certainly means overfitting.

### 8.5 Fit with existing benchmarks

RoboCasa, LIBERO, ManiSkill3, and BEHAVIOR are largely vision-centric with tactile absent. This section makes no sweeping verdict but **suggests** adding a feedback-value plugin to existing benchmarks — the four metric families above (ablation / dropout / temporal / slot fidelity) — and borrowing 9/11 §5.2.1's A/B/C/D arms directly.

## 9. Relation to VLA and world models

### 9.1 VLA today: architecturally absent, not accidentally omitted

§3.1 already said it: mainstream VLAs (π0 [arXiv:2410.24164](https://arxiv.org/abs/2410.24164), OpenVLA [arXiv:2406.09246](https://arxiv.org/abs/2406.09246), RT-2 [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)) are essentially **vision + proprioception + language** concat, with tactile and F/T **absent**. The absence is not oversight — adding them reopens all of §2, and VLA's scaling advantage rests on data that *can* be collected at web-scale. Tactile and F/T collection cost, and their missing unified schema, are what keep them out.

The real open question is: **does the next VLA wave stack more vision+language, or finally add tactile/F/T slots?** This article bets on the latter — not by adding tactile as "just another channel", but by inserting §6's slot interface into the VLA input layer. Qi et al.'s T-Dex [arXiv:2309.09979](https://arxiv.org/abs/2309.09979) (CoRL 2023) and Lee et al. [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) can be read as early sketches of that route.

### 9.2 World models: don't smear contact events into a latent

RSSM / DreamerV3 [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) latent dynamics assume smooth, differentiable transitions. 9/11 §5.1 already noted that contact is essentially a hybrid-dynamics mode switch — shove mode switches into a smooth RSSM latent and the model will learn a continuous approximation, which by definition smooths out the very events that matter.

The suggested architectural split:

- **World-model layer keeps structured contact slots**: mode is discrete, $C_t$ is a set, $f^{\perp}/f^{\parallel}$ are real; do not stuff these three into one latent vector.
- **Policy layer may consume a learned representation**: embedding the slot for policy use is fine; but that embedding is a *downstream consumer*, not an *upstream data format*.
- **Interface between them is §6**: world model predicts the next-timestep slot, policy reads the current slot.

This mirrors 9/11 §5.1's belief-update framing — the same structure, generalised across modalities.

### 9.3 One sentence to bind both

**VLAs are missing tactile slots; world models are missing contact structure. These are the same underlying failure — everyone has been hoping that piling on data will teach the model the interface, so nobody bothers designing it.**

## 10. Conclusion

This article started from the common misconception that multimodal fusion is a "model architecture" question, and pulled it back to foundations: the real difficulty is the **three bases** — time, frame, semantics — and no fusion architecture replaces them. Vision grew its common data interface not because its encoders are cleverer, but because it locked down those three bases first. Tactile, F/T, and proprio are still not converged on those bases; every cross-attention or shared-latent design will keep hitting the failure classes in §7.

The minimum viable route proposed here: **a contact-set-centric structured slot interface, sensor-specific raw→slot decoders, policy and world model composed only at the slot layer, and modality dropout treated as a first-class citizen in both training and benchmark**. This is not a final answer — **it is a direct response to the open question 9/11 left behind**: "tactile has no reusable intermediate representation". Design the interface first; then talk about architecture.

If 9/11's three tags were *Action-conditioned observation · Contact-state representation · Closed-loop value*, this article pins three:

> **Alignment before fusion · Structured contact slots · Modality dropout as first-class**

One sentence tying them together: **before fusing, align time/frame/semantics; after aligning, design slots before choosing architecture; on top of slots, treat missing modalities as part of the distribution, not exceptions**.

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

- Zhao et al., *A Touch, Vision, and Language Dataset for Multimodal Alignment* (TVL / Binding Touch to Everything), ICML 2024 · [arXiv:2402.13232](https://arxiv.org/abs/2402.13232) (tactile-VL alignment; representative of the "tactile CLIP" route)
- Feng et al., *AnyTouch: Learning Unified Static-Dynamic Representation across Multimodal Tactile Sensors*, 2025 · [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) (cross-heterogeneous-tactile unified representation · contrast to §5.1 raw-layer non-uniformity)
- Lin et al., *9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation*, ICRA 2023 · [arXiv:2308.14277](https://arxiv.org/abs/2308.14277) (one concrete tactile modality · evidence for §5.1 heterogeneity)

### D · Robustness and modality dropout (supporting §7.5 / §8.2)

- Maiga et al., *MMP: Towards Robust Multi-Modal Learning with Masked Modality Prior Fine-Tuning*, 2024 · [arXiv:2410.03010](https://arxiv.org/abs/2410.03010) (modality dropout as first-class; recent mitigation reference for §7.5)
- *Robust Multimodal Learning with Missing Modalities via Parameter Projection*, 2023 · [arXiv:2310.03986](https://arxiv.org/abs/2310.03986) (representation alignment under missing modalities · reference for §8.2 robustness metric)

### E · VLA and world models (supporting §9)

- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) (empirical evidence that VLA currently omits tactile)
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) (open-source VLA baseline; input side likewise vision+proprio+language)
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) (founding work of the VLA line; the architectural starting point of the tactile absence)
- Hafner et al., *Mastering Diverse Control Tasks through World Models* (DreamerV3), Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) (a latent-dynamics world-model representative; the contrast surface for §9.2 "don't smear contact events into a latent")

### F · Sim-to-Real / domain randomisation background (supporting §7.3)

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) (classical recipe for treating frame perturbation as domain randomisation)

### G · Continued from 9/11 · Contact state and impedance (background)

- Already cited in 9/11 and reused here: Hogan's impedance trilogy, Posa-Cantu-Tedrake IJRR 2014 (hybrid contact-mode trajectory optimisation), Lee 1810.10191, Qi 2309.09979, Huang 2410.24091, Zhao 2402.13232, Feng 2502.12191. This article will not re-list their links; see 9/11's Sources for exact references.

---

> **Related reading**
>
> - [The Hand Robots Don't Have: Tactile and Force Sensing in Embodied AI](/en/articles/2026-09-11-tactile-force-sensing/) — the predecessor; tactile and force sensing in isolation
> - [Sim-to-Real Methodology](/en/articles/2026-09-10-sim-to-real-methodology/) — frame perturbation (§7.3) and temporal perturbation (§8.3) borrow its domain-randomisation lens
> - [Why Robot Data Is Harder than LLM Data](/en/articles/2026-09-09-robot-data-scaling/) — the "where does supervision come from" problem in §3.4 shared latent is really a data-scaling problem
> - [VLA and World Models](/en/articles/2026-09-07-vla-world-models/) — §9 is a concrete side of that broader comparison: VLA missing tactile, world models missing contact structure
