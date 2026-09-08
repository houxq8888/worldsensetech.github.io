---
title: 'Stacking Sensors Is Not Fusing Them: Multimodal Robotics Lacks an Interface, Not a Model'
slug: "2026-09-12-multimodal-fusion-interface"
date: 2026-09-12
draft: false
categories: ["Embodied AI", "Multimodal Perception"]
tags: ["Embodied AI", "Multimodal Fusion", "Visuo-Tactile", "Force/Torque", "Proprioception", "Representation Interface", "Multimodal State Estimation", "Hybrid State Estimator", "Structured Belief", "Registration", "Cross-attention", "Modality Dropout", "VLA", "World Model", "Frame Alignment", "Time Alignment", "Uncertainty"]
description: 'Multimodal fusion is usually presented as a question of "which attention architecture", but in robotics the real bottleneck is upstream: Vision / Tactile / Force-torque / Proprioception have never agreed on temporal and spatial registration, task-relevant semantic projection, or on validity, provenance, and uncertainty. This piece splits fusion into perception → registration → semantic projection → multimodal (hybrid) state estimation → structured belief interface, and argues that what is missing is not a better fusion operator but a structured belief state sitting between heterogeneous observations and downstream consumers. A concrete minimum usable interface is proposed: a contact-set-centric slot for contact-rich manipulation (not "compress all robot data into contact sets"), with geometry typed as point / patch / region, continuous covariance and categorical probability kept separate, track_id as an inference hypothesis rather than an intrinsic identity, and validity / availability / lifecycle as first-class fields. Policy, world model, and controller interoperate through the interface — but the interface need not become the only information path. Benchmark suite adds oracle-slot vs estimated-slot vs end-to-end baselines, a consistency graph, and an interface-swap test. Three tags: Register before compose · Expose belief at the interface · Design for disagreement.'
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

Picture a bimanual robot fitted with an RGB camera, GelSight fingertips, wrist six-axis force/torque sensors, and joint encoders on every link. The hardware bill of materials looks genuinely "multimodal". Yet when you hand that rig to a policy, most published work tells you "just fuse them with cross-attention and you're done". That runs in a demo; **in real deployment it will almost certainly fail in one of four ways**: a modality drops frames, a sensor dies, coordinate frames drift, or the model quietly converges to one dominant modality and treats the rest as noise. These four are not engineering footnotes. **They all trace back to the same root cause: the modalities never agreed on a structured belief interface — explicitly expressing time, frame, semantics, uncertainty, provenance, and validity — that policy, world model, and controller can consume in common.**

This piece takes on the easiest topic to hand-wave in embodied AI: **multimodal fusion**. It will not sell you a specific network, will not argue against cross-attention, and — importantly — will not argue against end-to-end learning. What it *does* argue against is **letting state estimation, cross-modal composition, and control all happen implicitly inside a single, undiagnosable, non-reusable latent interface**. The article digs fusion down to its three foundations — time, frame, task semantics — and uses them to explain **why vision was able to grow a common data interface while tactile / force-torque / proprioception have not, and what a minimum usable belief interface looks like if you want to start building today**.

## 0. Framework: from "a fusion layer" to "hybrid state estimation + structured belief interface"

Put the whole article's frame up front. Every section returns to this picture.

```text
                     ┌────────────────────────────────────┐
                     │         Raw sensor streams         │
                     │  V · T · F · P  (four channels)    │
                     └──────────────────┬─────────────────┘
                                        │
                                        ▼
                        Modality-specific perception
                        (sensor-native encoder, raw → local structured observation)
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
      Temporal registration     Spatial registration     Semantic projection
        (clock / timestamp        (SE(3), hand-eye,
         alias / event bus)        sensor-mount, wrench transform)
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        ▼
                     ┌────────────────────────────────────┐
                     │  Hybrid multimodal state estimator │
                     │  x_t = (q_t, q̇_t, C_t, w_t, m_t, g_t) │
                     │  continuous · set-valued · discrete  │
                     │  mode · task goal                    │
                     └──────────────────┬─────────────────┘
                                        ▼
                     ┌────────────────────────────────────┐
                     │  Structured belief interface       │
                     │  Value · Semantics · Frame · Time  │
                     │  Uncertainty · Provenance ·        │
                     │  Validity / Availability / Lifecycle│
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

Three things this diagram wants to insist on, and the architectural thesis of the entire article:

1. **Fusion is not a network module. It is a complete "hybrid state estimation + interface contract" stack.** Letting the policy network implicitly handle state estimation on the side is the deepest hidden cost in most multimodal robot learning papers today. "Hybrid" here is deliberate: the state to be estimated naturally mixes continuous variables $(q, \dot{q})$, a set-valued contact state $C_t$, a discrete mode $m_t$, and a task goal $g_t$.
2. **Registration and semantic projection are two preparation steps before state estimation, not one.** Registration is measurable, calibratable, and unit-testable; semantic projection is task-conditioned representation learning. Collapsing them into a single "alignment" makes the hard-constraint part of the problem disappear into a soft-learning part that never converges cleanly.
3. **The interface is a contract, not a learned embedding.** Every field has a unit, a frame, a timestamp, an uncertainty, a source, and a validity state. Its optimization target is **interoperability across consumers**, not "friendliness to one specific downstream model".

One sentence that carries the central claim:

> **The missing abstraction is not a better fusion operator, but a structured belief state for heterogeneous, multi-rate observations.**

Or, as the boxed architectural thesis of this article:

$$
\boxed{\;\text{Raw multimodal observations} \;\rightarrow\; \text{structured belief state} \;\rightarrow\; \text{multiple consumers}\;}
$$

Once framed this way, VLA, world models, tactile foundation models, modality dropout, cross-modal contradiction, F/T constraints, and contact-set slots all become **different instances of the same thesis, not six parallel talking points**.

If the three tags from 9/11 were "three hard problems tactile has to solve on its own", this article pins three more technical tags on multimodal fusion:

> **Register before compose · Expose belief at the interface · Design for disagreement**

Every section below returns to these three.

## 1. "Multimodal" is not "Multiple Sensors"

The easiest way to derail this topic from the very first paragraph is to equate "a few more sensors" with "multimodal". That equation does not hold.

**"Modality" itself has no unique physics definition — it is a product of the analytic lens**. For this article's taxonomy, we fix four elements: **measurement space, physical origin, noise model, and update semantics** (triggering / sampling / clock). Together these four decide whether a data stream can be treated as "the same kind of thing". Different authors, different tasks can draw the boundary a bit looser or a bit tighter, but as long as it is declared up front, subsequent discussion will not drift.

Under this taxonomy, several commonly mislabelled examples: a wrist camera plus a head camera are **same-modality multi-view**, not two modalities; the four tiny cameras inside a GelSight fingertip are **internal structure of a single tactile modality** whose measurement space is "elastomer surface deformation field", not scene RGB — even with a CNN in the middle, the downstream semantics are contact geometry, not object detection; joint encoders plus motor current are two **observation channels of a single robot-state modality** (one direct position measurement, one indirect torque inference, both sharing the same latent physical state) — treating them as two independent modalities and fusing them will teach attention nothing but redundancy.

The four genuine modalities this article reasons about: **Vision** (RGB / RGB-D / event cameras; light-radiance field; camera frame; effective observation rate typically 15–60 Hz), **Tactile** (GelSight / GelSlim / TacTip / 9DTact / taxel arrays; contact-interface deformation or force distribution; sensor frame; 30 Hz to kHz depending on sensor family), **Force / torque** (wrist F/T; **aggregate** wrench at the contact interface; sensor + tool frames; 500 Hz – 1 kHz), **Proprioception** ($q, \dot{q}, \tau$, EE pose; base / world frame; hundreds of Hz to kHz). Some work adds audio, thermal, gas, ultrasound as fifth or sixth modalities; the classification logic is the same.

**Take-away**: before talking about fusion, list the modalities explicitly against the four-element taxonomy of this article. Without this list, every "fusion architecture" downstream is a castle in the air. 9/11 §2.1 used a taxonomy tree to split contact sensing into Tactile / Force-torque / Proprioceptive branches — this article keeps that three-way split and restricts the discussion to V + T + F + P.

## 2. The Three Preparations Before Fusion: Registration and Semantic Projection

The most technical, most-skipped section. First, a **terminology tightening**: **"registration" and "semantic projection" have different characters**, and collapsing them into "alignment" makes the reader think a math transform is enough. This article uses:

```text
Pre-fusion pipeline
├── Registration        (hard constraints, closed-form or standard algorithms)
│   ├── Temporal registration
│   └── Spatial registration
└── Semantic projection (learn task-relevant state representation)
    └── Raw observation → task-relevant state
```

**Registration** lives at the measurement layer — it is standardized, calibratable, unit-testable. **Semantic projection** is closer to perception and state estimation, and is the protagonist of §6 below. Both happen **before** fusion, but they are different activities.

### 2.1 Temporal Registration

The default time constants differ by orders of magnitude: vision (image-based) ~15–60 Hz; image-based tactile ~30–200 Hz; taxel / array tactile ~500 Hz – kHz; F/T ~500 Hz – 1 kHz; proprioception ~hundreds of Hz to kHz. Beware a common confusion: these are **effective observation rates**, not internal sensor sampling rates. A "kHz-taxel" GelSight-like sensor whose *output image* is only 30 Hz still delivers 30 Hz to the fusion layer.

A common but overly simple baseline is to resample every modality onto a shared low-rate policy clock (usually vision's). **For contact-rich control, that turns high-frequency events into invisible alias** — slip detection, transient contact-force peaks, joint impacts typically live between two vision frames; downsampling erases them.

A more honest architecture is **multi-rate coexistence**: a visual policy at 30 Hz, force/tactile-driven compliant control at native rate, and event signals on a dedicated event bus. Three details that must be written into the interface layer:

- **Timestamp semantics**: sensor timestamp (exposure / sampling instant) vs arrival timestamp (software receives) vs host timestamp (fusion instant). The gap is 5–20 ms, easily enough to shift the "moment of making contact with the cup" to before finger closure.
- **Clock drift**: camera crystal, robot controller clock, and host clock are three independent oscillators; drift over minutes is tens of milliseconds, and overnight drift is more than enough to teach the fusion model a wrong lead/lag correlation.
- **Event vs periodic**: tactile event signals (e.g. slip triggers) are sparse events, not periodic samples. Stuffing them into an evenly-spaced buffer destroys event density information.

**Design rule**: **write time-constant differences down as modeling assumptions; do not hide them behind downsampling**. This is the same rule as 9/11 §4.1 — "concrete rates are determined by policy architecture, hardware servo, controller, and compute budget; do not hard-code numbers".

### 2.2 Spatial Registration

Four signals natively hang on four coordinate frames: Vision on camera frame, Tactile on sensor frame (a local frame stuck on the fingertip surface), F/T on sensor frame + tool frame, Proprio on base / world frame. Putting them in one fusion layer requires at least three things: **hand-eye calibration** ($T^{cam}_{base} \in SE(3)$, drifts by fractions of a degree after any collision); **sensor-mount calibration** (fingertip sensor frame → link frame offset $T^{\text{sens}}_{\text{link}}$); and **wrench transformation plus gravity / inertial compensation** (bring sensor-frame wrench to base / world, subtract tool gravity and inertial terms). If any of those steps is wrong, every downstream "normal vs tangential force" decomposition inherits the error.

**Typical symptom of poor registration**: a trained model loses points immediately when you swap the tool, change the camera angle, or nudge the base. The model actually learned correlations specific to a training-time sensor frame, not the task. A subtler issue is **relative vs absolute pose**: contact physics depends only on who touches whom, but many policies feed the raw end-effector pose in world frame — the model happily learns degrees of freedom it should not have.

### 2.3 Semantic Projection

This is not registration; it is **mapping observations onto a shared task-semantic coordinate**: Vision → object / scene semantics; Tactile → local contact semantics; F/T → aggregate contact semantics; Proprio → self-state semantics. 9/11 §2.1 laid out a nine-layer signal-to-meaning chain (Sensor → Calibration → Raw obs → Contact perception → Contact geometry & wrench → Contact mode & physical state → Task-relevant belief → Policy / controller → Action → New contact). **The core multimodal-fusion question is: at which layer do you compose?** The common mistake is composing at the raw layer — concatenating four tensors and feeding them to a network — which forces the policy to implicitly learn everything in §2.1 / §2.2 / §2.3, at enormous sample cost. The more reasonable split is: **each modality runs through raw → perception → contact geometry on its own, then composition happens at the four semantic points where meaning has already converged (contact events, wrenches, robot state, task belief)** — this is the structured belief interface in §6.

**Take-away**: temporal and spatial registration are hard measurement-layer constraints; semantic projection is the split of responsibility between perception and state estimation. Cut any corner here and every beautiful fusion architecture downstream is a patch over the hole.

## 3. A Genealogy of Existing Fusion Paradigms: Different Mechanisms, Different Problems

When academia says "multimodal fusion", it usually means "how to organize signals at the composition step". Baltrusaitis et al.'s classic survey [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) organizes this space into five layers — representation, learning, feature selection, fusion, application — this section borrows the fusion layer. **Important positioning**: the mechanisms below are **not mutually exclusive architectural choices, but different axes of mechanisms**; mature systems typically combine several:

| Mechanism | Problem it addresses | Which layer of §0 it lives in |
| --- | --- | --- |
| Modality-specific encoder | raw → local observation | Perception |
| Temporal / spatial registration | clock, SE(3), coordinate frames | Registration |
| Cross-attention | learned composition | Composition inside belief interface |
| Shared latent / VLT-style | representation-level alignment | Semantic projection |
| Late / decision fusion | decision-level combination | Policy / controller |
| Structured belief interface | schema and contract | Belief interface |

That is, this article does **not** oppose cross-attention; it **scopes its job**: **cross-attention cannot substitute for explicit temporal / spatial / semantic registration**. Attention can compose already-aligned representations; it cannot discover registration out of raw data. That one sentence is the pivot of this section and the article.

### 3.1 Early concat: raw / embedding-level stacking

The most naive approach: run each modality through an encoder, concatenate, feed an MLP or Transformer. Pros: low barrier to entry; with enough samples, a deep network will "learn" some alignment. Cons (near-certain failure modes): time-constant differences get masked by concat, encouraging spurious lead/lag; any dropped frame means zero-fill or hold-last-value, both out of distribution; the visual and tactile encoders have very different inductive biases and gradients pollute each other; unseen combinations of missing modalities cause immediate collapse.

**This is where we must be careful about VLA**. Classifying RT-2 / OpenVLA / π0 as "vision + proprio + language concat" is inaccurate. **RT-2** [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) expresses actions directly as **text tokens** fine-tuned jointly with the VLM — not a state-vector concat. **π0** [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) is a VLM backbone + **proprioception tokens** + **noisy action chunks** emitting actions via flow matching. **OpenVLA** [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) explicitly supports multi-camera, depth, and proprioceptive state encoding branches — also not a simple V+L concat. A more accurate characterization:

> **Looking at RT-2, OpenVLA, and π0 as representative systems, we observe that the common thread in modern VLAs is "vision-language pretraining as the primary scaling surface, with robot state injected as additional representations into the policy". Tactile and force-torque have not yet formed a public, cross-task, cross-embodiment large-scale pretraining ecosystem comparable to vision-language.**

This absence is not an oversight. Adding tactile / F/T forces §2 to be re-derived from scratch and there is no pretraining corpus behind those modalities at the moment. §9.1 unpacks this further.

### 3.2 Cross-attention / Transformer fusion

One step beyond concat: treat each modality as a token (or short token sequence) and run self-attention / cross-attention on top. Most vision-language-action work uses this template. Tsai et al.'s Multimodal Transformer [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) is a classic starting point that explicitly handles "different modalities run at different time granularities". Structurally, it acknowledges that alignment is **learned** and gives soft alignment some room. Modern multi-modal Transformers already come with the tools: timestamp / positional embeddings, relative temporal encoding, modality embeddings, frame-aware features, modality-specific encoders, modality dropout / masking, auxiliary per-modality losses, attention masks for missing modalities. So the real failure is not "attention breaks" — it is: **if you do not write these into the interface contract, attention will happily learn registration from the data distribution as well, and what it learns is a coincidence, not a portable interface**. In a mature multimodal system, cross-attention belongs strictly **inside the state interface, once perception and registration have already deposited aligned inputs**.

### 3.3 Late / decision-level fusion

Each modality produces its own sub-policy (or sub-value / sub-action proposal); a decision layer votes or gates by confidence. This is closer to classical robotics: vision coarse-aligns, F/T fine-aligns, tactile handles slip recovery, proprio tracks the nominal trajectory. Pros: **engineering-robust** — dropping one modality only removes one proposal; safety layers fit naturally; explainable. Cons: bottom-level coupling is lost — many signals are inherently cross-modal ("from tactile + F/T jointly, is this a stable slide or stick-slip?"), and reconstruction at the decision layer is hard; weighting rules either hand-written (does not scale) or learned (back to §3.1's problems). **This branch is actually the most common in industrial contact-rich manipulation** — it is usually called "layered control", not "fusion", so papers under-report it. 9/11's combination of impedance control + visual coarse localization + tactile slip detection is essentially this late fusion.

### 3.4 Shared latent / VLT-style alignment

Contrastive learning / distillation / CLIP-style objectives pull different modalities into one latent space. Recent examples: **TVL / Binding Touch to Everything** [arXiv:2402.13232](https://arxiv.org/abs/2402.13232) (Zhao et al., ICML 2024) — aligns tactile with vision-language in latent space via ~44K vision-touch pairs, using language as the bridge; **AnyTouch** [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) (Feng et al., 2025) — a unified static-dynamic representation across heterogeneous visuo-tactile sensors, essentially an intra-tactile latent fusion; **3D-ViTac** [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) (Huang et al., CoRL 2024) — joint visuo-tactile representation for fine-grained insertion; **Lee et al. Making Sense of Vision and Touch** [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) (ICRA 2019) — one of the earliest self-supervised visuotactile alignment works. Pros: reusable representations; theoretical ceiling for cross-task / cross-embodiment. Cons: **where does supervision come from?** Vision-language contrastive learning can eat the internet's image-text pairs; there is no natural supervision source for a vision-tactile-force-proprio quadruple. Two current workarounds: (a) force co-collection from robot teleop logs and treat "close timestamps" as weak alignment supervision (Calandra et al. *More Than a Feeling* [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) is an early tactile-grasp instantiation of this idea); (b) use language / vision as the bridge and pull tactile into the V-L space (which is what TVL does).

**Explicit caveat**: the above works demonstrate that **shared latent alignment is feasible on specific task families**; their representations remain policy-specific or dataset-specific. This article proposes, as **its own conclusion (not theirs)**: if you want these representations to be reused across policy / world model / controller / sensor, you need another layer *below* the latent — a **structured belief interface with explicit units, frames, timestamps, uncertainty, provenance, and validity**. That layer is this article's thesis, not a finding of the cited papers.

### 3.5 Structured belief interface: the route this article recommends

The last category: **don't fuse at raw, don't fuse at latent — fuse on an explicitly specified intermediate representation**. That intermediate is not a learned embedding, it is a set of **semantically converged belief slots carrying uncertainty, provenance, timestamps, and validity**. This directly extends 9/11 §3.4 "vision has a common data interface, tactile does not", but scoped to contact-rich manipulation. Schema design is in §6; here is a skeleton:

```text
state_slots = {
    contact_set      : list of contact records (see §6.1)
    wrench_ext       : 6D + covariance + validity (§6.7)
    robot_state      : (q, q̇, τ, EE pose) + availability
    task_context     : language token / goal embedding
    scene_state      : free-space / occlusion / … (optional)
    belief           : task-relevant latent / explicit
}
```

Pros: missing one modality only touches its own key; time / frame / semantics / validity are all converged inside slot definitions; once slots are pinned down, fusion structure can be swapped freely. Cons: designing the slots is heavier than "just stack a Transformer"; if slots are inaccurate, downstream policy cannot exceed them; **and slots do not subsume the object identity / geometry / free-space / occlusion / scene-context semantics that vision provides** — this article's recommendation is scoped to **contact-rich manipulation**, not general multimodal robotics.

**Take-away**: within the genealogy, early concat / attention / late fusion / shared latent each have their place, but their failure modes almost all trace back to §2's three preparations. This article prefers shifting engineering gravity from "which attention" back to "pin down the state interface first".

## 4. Why Vision Got a Reusable Data-and-Representation Ecosystem First

A natural question: §2's three preparations apply to vision as well — why did vision alone grow a common data interface? This section gives a history-flavored engineering answer, and holds up a mirror for tactile / force-torque / proprioception.

**Correction first**: **this is not single-cause**. Vision's win is three-bases fixed + ecosystem factors combined: standardized hardware (CMOS sensor + unified lens mount), standardized file formats (JPEG / PNG / MP4 / HDF5), a coordinate model (pinhole + SE(3)), internet-scale data, mature annotation tasks, public benchmarks (ImageNet / COCO / ADE20K / LVIS), mature encoders, GPU scaling laws, and available compute. This article emphasizes the "three bases" axis only because that axis is exactly what tactile currently lacks. **We are not claiming "fix the three bases ⇒ vision succeeded" or "vision solved §2 first and therefore became a foundation model."**

On **temporal registration**: video is frame-indexed by definition, at 30 or 60 Hz; every downstream task (classification / detection / segmentation / SLAM) tacitly agrees "one frame is one unit". This convention kills the most painful class of fusion problems — every image-based downstream algorithm shares one timeline. On **spatial registration**: pinhole camera model + intrinsics / extrinsics turn "image pixel ↔ world point" into one formula ($s \cdot m = K [R | t] \cdot M$), on top of which 3D vision, SLAM, NeRF, 3D Gaussian Splatting, and multi-view stereo all grow. Calibration errors exist, but "what an error looks like" is predictable and reproducible. On **semantic projection**: RGB has no semantics by itself, but the vision community has — through COCO / ImageNet / ADE20K / LVIS and years of large-scale annotation — **formed a family of highly interoperable task-level representation conventions**: object class, bounding box, instance mask, depth, affordance, caption. More precisely, these datasets **do not define one unified semantic schema**: ImageNet is a classification ontology, COCO is detection + instance + caption, ADE20K is scene parsing, LVIS is long-tail instance distribution. They define independently and compose across. "Highly interoperable task-level conventions" is closer to fact than "unified schema", and explains why one CV model plugs into another with so little friction.

Contrast tactile: on temporal registration, different sensor families span 30 Hz to kHz, with event and polling mixed, and no consensus; on spatial registration, GelSight is pixel + elastomer deformation, 9DTact is pixel + 3D deformation field, taxel arrays are 1D / 2D force distributions, optical tactile (AnySkin / DigiTact) is yet another representation — "what shape one tactile reading takes" is not agreed; on semantic projection, contact points, normals, tangential forces, slip, and mode all appear in papers, but there are **no cross-dataset task-level conventions**. It is also worth correcting a common shortcut — "9DTact's 6D force, GelSight's shear map, and array taxel load are all the same physical quantity viewed differently" — **they overlap but sit at different layers of abstraction**: GelSight's shear / deformation map is closer to raw local deformation observation, whereas 9DTact's 6D force is a global wrench **estimate** obtained via model inversion. They are not two projections of one physical quantity; they are the **observation layer vs the estimation layer**. This distinction directly shapes the slot schema in §6.

**Take-away**: vision did not win because its encoder is cleverer. It won because it nailed §2's three preparations **and** grew an ecosystem on top of them, so all downstream fusion discussion runs on one foundation. For tactile / force-torque / proprioception to grow a same-scale fusion ecosystem, the first move is **to agree on an interface**, not to compete on architectures.

## 5. Per-Modality Fusion Difficulties

This section walks the modality-specific pain points that set up §6's interface design. **Note**: time / frame / semantics were already established in §2; this section only adds what is unique to each sensor and does not re-derive generalities.

### 5.1 Tactile: image / taxel / optical / capacitive — even the raw layer is not unified

Tactile is the most "fractured" channel. One physical quantity (deformation or force distribution at a local contact interface) has at least these **structurally different** realizations:

```
Image-based      GelSight  · 3 colored LEDs + camera → RGB deformation image
                 9DTact    · ring of colored LEDs + camera → multi-view image [arXiv:2308.14277]
                 GelSlim   · 3 cameras + speckle in elastomer → 3 optical-flow fields
                 TacTip    · silicone tip + optical fibers + camera → tip pose

Taxel array      BioTac · array pressure / temperature / EDS
                 1D/2D capacitive arrays · local normal-force distribution
                 Piezoresistive arrays · local stress image

Optical waveguide AnySkin · DigiTact · light guided in elastomer, edge scattering marks contact

Proprioceptive-  F/T + kinematics → inferred contact ("soft tactile")
 inferred
```

Before any fusion, each requires a different encoder — CNN for images, MLP for taxel arrays, spline models for waveguides, IK for proprioception-inferred. A common reaction is "then build a tactile foundation model to unify all sensors" — that is exactly what AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) is doing. The direction is right, but **it unifies the learned-representation layer, not the raw layer, and not the state-interface layer**. Even with AnyTouch, its output is still an embedding; you still need an explicit "embedding → contact slot" contract before it enters the state interface. 9/11 §3.5 splits representation into "interface" vs "learned representation" precisely to keep these two layers from being collapsed. **Mitigation**: insert a **sensor-specific decoder** between raw and slot, mapping heterogeneous tactile outputs into §6.1's slot schema; the decoder may be analytic or learned, but it must be **part of the interface**, not buried inside the policy.

### 5.2 Force / Torque: looks the most "standard", hides the most traps

F/T outputs a 6D wrench in a unified format (much friendlier than tactile), but its **semantics** are more complex:

$$
w_{\text{raw}} \;=\; w_{\text{contact}} + w_{\text{gravity}} + w_{\text{inertial}} + w_{\text{friction}} + b_{\text{bias}}
$$

Extracting external contact wrench requires: zero-drift compensation ($b_{\text{bias}}$, tare, never perfect on real robots), gravity compensation ($w_g = g(q)$, includes tool + gripper mass distribution; any tool swap changes every curve), inertial compensation, and coordinate transformation. **Formula tightening on inertial compensation**: the joint-space dynamics term $M(q)\ddot{q} + C(q,\dot{q})\dot{q}$ is *not* the wrench an F/T sensor sees. The sensor-frame inertial wrench requires mapping through the Jacobian and spatial dynamics transform:

$$
w_{\text{inertial}}^{\text{sens}} \;=\; \mathrm{Ad}^{\top}_{T_{\text{tool}\rightarrow\text{sens}}}\, J^{\top}(q)\,\big[M(q)\ddot{q} + C(q,\dot{q})\dot{q} + \tau_{\text{drive}} - \tau_{\text{meas}}\big]
$$

At slow speeds this term is often approximated as negligible; at high speeds it is not. **Any slip in this step corrupts every downstream normal / tangential decomposition**. F/T has a more fundamental limitation: **it reports the aggregate wrench at the interface, not local contacts**. This "aggregate vs field-level" split is why 9/11 §2.1 puts tactile and force/torque on separate taxonomy branches, and why §6.2 treats F/T as a constraint rather than a detector. **Mitigation**: F/T should emit **two** things into the slot layer — (a) compensated external wrench (for force-aware policy); (b) residual magnitude (to detect "drifted again" or "hit something we shouldn't"). The second is often dropped.

### 5.3 Proprioception: treated as "background signal", actually load-bearing

Proprio is typically the fastest of the four and is **usually the most direct, most stable, and most readily available internal-state observation on a robot**. That phrasing needs qualification — "the complete built-in channel" is inaccurate: many robots have no direct torque sensing, $\dot{q}$ is often numerical differentiation, EE pose is often an FK estimate, motor current and joint torque are separated by friction / backlash / transmission-ratio maps, and on soft or low-cost platforms proprio can be severely incomplete. So the safer statement is: **it is the easiest channel to standardize, and the natural anchor for coordinate frames and time bases**.

Proprio plays two roles in fusion: **as input to a contact hypothesis** — given a desired end-effector trajectory and joint-torque feedback, one can invert $J^T \hat{F}_{ext} = \tau_{\text{residual}}$ to infer external contact force (contact inference, a classical tool going back to Hogan's impedance line); **as anchor for time base and coordinate frames** — fusion needs a stable reference frame, proprio natively runs at whatever hardware servo rate is available, and EE pose can serve as anchor for camera / sensor frames. **Fusion difficulty**: proprio is too "clean" — it is the robot measuring itself, uncontaminated by external-world uncertainty. The model will over-rely on proprio as a shortcut, doing well in non-contact scenes and never learning to use tactile / F/T when contact matters. This is exactly what 9/11 §5.2.1's feedback-value ablation is designed to catch.

### 5.4 Time-constant difference is itself a modeling assumption

Combining §5.1–5.3 with §2.1: **cross-modality differences in effective observation rate are not a "handle it in the fusion layer" engineering question but a "what control bandwidth does the task need" modeling assumption**. Wiping / polishing needs force-control bandwidth of at least 100–500 Hz, vision 30 Hz is plenty, tactile and F/T must run at native rate, proprio needs torque level — those tasks cannot downsample everything to 30 Hz. Pick-and-place / assembly is vision-dominant with sparse contact events, and downsampling tactile to vision rate is an acceptable approximation. Concrete numbers vary by hardware, task, and controller; do not hard-code them. This is the same judgment as 9/11 §4.1.

## 6. A Minimum Usable Structured Belief Interface

Now to answer "what to do concretely". This section gives a **deployable, benchmarkable, incrementally-evolvable** minimum interface.

**Scope tightening first**: what this section proposes is **not** "all robot multimodal information should be compressed into a contact set". It is: **for contact-rich manipulation, the contact set should be the core of the cross-modal structured belief state**. Vision's object identity / geometry / free-space / occlusion / scene context do not go into the contact set — they have their own slots (e.g. task_context, scene_state, world_model_state).

### 6.1 Contact-set-centric, but not only contact set

Define the shared contact record (upgraded version, carrying uncertainty / provenance / timestamps / validity):

$$
C_t = \big\{\, \big(\, \mathrm{track\_id}_i,\; G_i,\; \mathcal{F}_i,\; m_i,\; \mathcal{U}_i,\; s_i,\; \mathcal{I}_i,\; \mathcal{L}_i \,\big) \,\big\}_{i=1}^{N_t}
$$

where $G$ is geometry, $\mathcal{F}$ is a wrench, $m$ is mode, $\mathcal{U}$ is uncertainty (split below), $s$ is source, $\mathcal{I}$ is evidence mask, $\mathcal{L}$ is lifecycle. Expanded:

```python
contact = {
    # Identity — cross-frame consistency is a hypothesis, not intrinsic
    "track_id":            Optional[int],   # a hypothesis maintained by temporal
                                            # association; see §6.1.2

    # Geometry — do not silently assume point contact
    "geometry": {
        "type":            enum,            # point | patch | region
        "position":        Vector3,         # representative point or centroid, base frame
        "normal":          Vector3,         # mean normal (if applicable)
        "extent":          Optional[...],   # if type ≠ point: patch size / area /
                                            # principal axes / center of pressure
    },

    # Wrench (canonical point / patch summary)
    "wrench": {
        "f_perp":          float,           # normal force
        "f_parallel":      Vector2,         # tangential force in tangent plane
        "moment":          Optional[Vector3],  # if type ≠ point: torsional / rolling moment
    },

    # Mode & events
    "slip_probability":    float,           # φ ∈ [0, 1]
    "mode":                enum,            # free / touch / sticking / sliding / rolling / separating

    # Uncertainty: continuous and categorical must not be merged into one matrix
    "uncertainty": {
        "pose_covariance":      Matrix,     # Σ_pose on SE(3) or ℝ³
        "force_covariance":     Matrix,     # Σ_wrench matching wrench
        "slip_probability":     float,      # a scalar probability, not a covariance
        "mode_probability":     Vector,     # categorical distribution over the mode enum
        "calibration_quality":  enum,       # nominal / degraded / unknown
    },

    # Provenance
    "source":              enum,            # tactile / FT / proprio / vision / fused
    "evidence_mask":       [V, T, F, P],    # which channels contributed to this record

    # Time & lifecycle (§6.1.1)
    "timestamp":           float,           # observation instant, host clock
    "age":                 float,           # time since observation
    "valid_from":          float,
    "valid_until":         float,
    "availability":        enum,            # present / stale / delayed / unavailable / corrupt
    "lifecycle":           enum,            # new / tracked / occluded / lost / merged / split
}
```

This definition is designed for five properties: **sensor-agnostic** (GelSight, taxel array, F/T — anything that can fill these keys can enter the state interface); **physically interpretable with explicit units** (every number has a dimensional unit and a frame); **carries uncertainty** (continuous covariance and categorical probabilities are stored separately, giving downstream Bayesian fusion, safety layers, and fallback something consumable; **a slot without uncertainty is not an interface, it is "asserted fact", and it is never true on real hardware**); **carries provenance** (who contributed this record; how to degrade when a channel is missing — directly linked to §7.5 modality dropout and §8.6 cross-modal contradiction); and **carries validity / availability / lifecycle** (detailed in the next two subsections).

Beyond the contact set, the state interface also contains at least:

```text
wrench_ext          : 6D + covariance + validity       # F/T independent aggregate observation
robot_state         : (q, q̇, τ, EE pose) + availability
task_context        : language token / goal embed
scene_state         : free-space / occlusion / …       # non-contact vision semantics (optional)
belief              : task-relevant latent / explicit
```

**The contact set is the core structured object of this interface, not its entire content**.

#### 6.1.1 Validity / Availability / Lifecycle are first-class

The same slot value can correspond to three very different states:

```text
value plausible, age large   ── valid but stale
value fresh, covariance huge ── fresh but uncertain
value missing                ── unavailable
```

These three require different controller behavior: stale → hold + covariance inflation + fallback; fresh-uncertain → keep consuming but down-weight; unavailable → route to explicit §6.6 degradation path. If the interface layer can only say "value / no value", downstream has to guess. **This is not a benchmark trick — it is a requirement on interface design itself**.

#### 6.1.2 track_id is a hypothesis, not an intrinsic property

Writing `id` as "tracking ID, cross-frame consistent" was too optimistic. A contact set is fundamentally a **permutation-invariant set** — it has no intrinsic identity. Multi-finger manipulation, rolling contact, contact patch splitting / merging, occlusion, and transient contact all turn identity assignment into an inference problem in its own right. This article therefore rewrites the field as `track_id: Optional[int]` and states:

> **track_id is a hypothesis maintained by temporal association, not an intrinsic physical property of a contact.**

Slots are still valid without a track_id (single-frame estimation, or when identity is genuinely unreliable). When present, track_id is a consumable, confidence-carrying hypothesis. Without this sentence, the entire "cross-frame consistency" of the interface is an empty promissory note.

#### 6.1.3 Point contact is a modeling assumption, not physical fact

The classical quartet $(p, n, f_\perp, f_\parallel)$ implicitly encodes "point contact, or a local patch reduced to a representative point". Many tactile tasks are not point contact. Finger-pad-to-object is really a contact patch $\mathcal{A}_i$ with a pressure distribution, shear distribution, normal-force distribution, center of pressure, and torsional moment. For GelSight-class sensors, the raw observation is more naturally contact-patch geometry, not a single $(p, n)$. Hence the interface explicitly carries `geometry.type ∈ {point, patch, region}` with `extent` and `moment` where applicable. At minimum, refer to $p, n, f_\perp, f_\parallel$ as a **"canonical point/patch summary"**, so the schema does not pretend to be a complete physical description of a contact.

### 6.2 How each signal maps to slots · F/T is a constraint, not a detector

Say the F/T → contact mapping precisely: "F/T decomposes onto an existing $C_t$; without tactile, synthesize a single global contact entry" can be misread as "a 6D wrench suffices to recover contact geometry". In reality:

$$
w \;=\; \sum_{i=1}^{N} \begin{bmatrix} f_i \\ (p_i - p_0) \times f_i \end{bmatrix}
$$

Infinitely many $\{p_i, f_i\}$ configurations can produce the **same** resultant wrench $w$. This is a classic **inverse problem, not a deterministic decoder**. The correct slot semantics are: **each channel contributes one of likelihood / constraint / proposal — none of them fills a fact by itself**:

```text
Tactile         → local contact observations       (direct, sensor-noisy)
F/T             → global wrench constraint         (aggregate, no localization / decomposition)
Proprioception  → kinematic / dynamic constraint   (via J^T, model-accuracy dependent)
Vision          → geometric prior                  (predicted contact hypothesis, not observation)
```

Combining four channels becomes a Bayesian problem:

$$
p(C_t \mid V, T, F, P) \;\propto\; p(V \mid C_t)\, p(T \mid C_t)\, p(F \mid C_t)\, p(P \mid C_t)\, p(C_t)
$$

$p(T \mid C_t)$ is the local observation likelihood (tactile sensor model); $p(F \mid C_t)$ is the wrench-consistency likelihood; $p(P \mid C_t)$ is the dynamic-consistency likelihood via $\tau_{\text{residual}} = J^T F$; $p(V \mid C_t)$ is the geometric / predictive likelihood; $p(C_t)$ is the prior (e.g. contacts lie on object surfaces).

**Important caveat**: this factorization **implicitly assumes conditional independence**, $p(V, T, F, P \mid C) = p(V|C) p(T|C) p(F|C) p(P|C)$. Real robot systems contain heavy cross-modal coupling — robot pose, object pose, contact geometry, dynamics, calibration, and actuator state are all shared variables. Concretely, F/T and proprio jointly depend on $(q, \dot{q}, \tau)$. **We adopt the conditionally independent factorization purely for expository clarity; real systems must explicitly model cross-modal correlations** — via a joint Gaussian likelihood with a shared covariance, or a graphical model / message passing over a coupled factor graph. Write this caveat into the article; otherwise "the Bayesian step" is elegant but over-idealized.

### 6.3 Evidence Ranking: hypothesis-dependent, not a fixed order

A fixed ranking "tactile > F/T > proprio > vision" is wrong. The correct framing is **evidence ranking should be hypothesis-dependent**:

```text
Contact location:    Tactile > Vision > F/T ≈ Proprio
Global wrench:       F/T > Tactile > Proprio > Vision
Object pose:         Vision > Tactile > F/T ≈ Proprio
Joint state / τ_res: Proprio >> others
Contact mode:        Tactile > F/T ≈ Proprio > Vision
Slip probability:    Tactile > F/T (rate) > Proprio (residual) > Vision
```

**Different fields of the same $C_t$ record can have entirely different dominant modalities** — which is exactly why the interface carries `evidence_mask`. This upgrade moves the interface from "rule system" to "probabilistic evidence fusion", directly matching the Bayesian formulation in §6.2.

### 6.4 Interface ≠ learned latent representation

The single most important subsection of the article.

```text
Latent representation              Structured belief interface
─────────────────────────          ─────────────────────────────
Optimized for downstream model     Optimized for cross-consumer interoperability
No units                            Explicit SI or local unit per field
No frame (implicit in data flow)    Frame declared per field, with frame_id
No time                             Explicit timestamp + age
No validity                         availability / lifecycle / valid_until
No uncertainty (only in loss)       Continuous covariance + categorical distribution
                                    + calibration metadata
No provenance                       source + evidence_mask
Single consumer                     Multiple consumers: policy / world model / controller
                                    / diagnostic tool / safety layer / another sensor
```

A formula that survives the reviewer's pen (compared with v2, now explicitly includes **Value and Validity**):

$$
\text{Interface} \;=\; \big(\, \text{Value},\;\text{Semantics},\;\text{Frame},\;\text{Time},\;\text{Uncertainty},\;\text{Provenance},\;\text{Validity} \,\big)
$$

**A latent can be perfectly suited to a policy and still unsuited as the input contract for a world model, a controller, a diagnostic tool, or a replacement sensor.** That gap is exactly why an interface has to exist — and it is the missing piece in §3.4's shared-latent line. CLIP / ImageNet / COCO do not power the vision ecosystem because their embeddings are best; they power it because they turned semantics, units, frames, time, and validity into a contract that every downstream consumer can read. Tactile, force-torque, and proprioception need that contract, **not another larger model**.

### 6.5 When to compose: not at raw, not at decision — **at the belief interface**

With §6.1's schema in place, fusion is three steps:

```python
# Step 1: per-modality perception + registration
raw_v  → enc_v → pred_contact_hypothesis      (visual contact prediction, with confidence)
raw_t  → enc_t → detected_contact             (tactile observation, with pose_covariance)
raw_ft → enc_ft → external_wrench + residual  (compensated wrench, with force_covariance)
raw_p  → proprio_state → J^T · τ_res → inferred_contact (with model error)

# Step 2: hybrid state estimation → write to belief interface
C_t         = state_estimator(V, T, F, P)   # output is belief, not hard fact
wrench_ext  = compensate_ft(raw_ft)
robot_state = (q, q̇, τ, EE_pose) + availability
belief_t    = belief_update(C_t, wrench_ext, robot_state, task_ctx)

# Step 3: policy / world model / controller consume the belief interface
a_t = policy(belief_interface)
```

The key claim is: **Step 2's interface schema is the multimodal system's actual interface**. Step 1's encoders can be swapped at will; Step 3's policy architectures can be swapped at will; as long as Step 2's schema is stable, everything interoperates. And note: **the structured belief interface is the stable interoperability boundary of the system, not necessarily the only information path for every downstream model**. A policy can still consume extra raw visual features or language tokens on top of structured slots. What must hold is that **the state shared across consumers flows through this contract**, not that "every bit of information must pass through it".

### 6.6 Fallback under missing modalities: making "one channel gone" part of the training distribution

Real deployment has dead sensors, dropped frames, timeouts as the norm. The interface must support "no data on channel X" explicitly:

```python
if tactile_missing:
    C_t = state_estimator(V, F, P)         # no T, replace p(T|C_t) with uniform likelihood
    for r in C_t:
        r.uncertainty.pose_covariance  *= inflation_T_missing
        r.uncertainty.mode_probability  = uniform_over_modes
        r.evidence_mask.T = False
        r.source           = "no-tactile"
        r.availability     = "unavailable"
```

**This is not an engineering patch. It is a core training-time constraint.** See §7.5 modality dropout and §8.6 cross-modal contradiction.

### 6.7 wrench_ext vs contacts[]: aggregate observation + consistency constraint, not redundancy

A reviewer will immediately ask: if `contacts[]` already carries per-contact force and moment, why do we still need a global wrench? The answer is elegant:

$$
w_{\text{ext}} \;\neq\; \sum_i w_i^{\text{reconstructed}}
$$

`wrench_ext` is an **independent aggregate measurement** (what the F/T sensor directly reports). `contacts[]` is a **structured hypothesis** (the joint inference of all perception channels through the state estimator). These are two different observations, and they constrain each other:

$$
\mathcal{C}_{\text{wrench}} \;=\; \big\| w_{\text{ext}} - \textstyle\sum_i \big[f_i;\, (p_i - p_0) \times f_i\big] \big\|_{\Sigma^{-1}}
$$

The larger this residual, the less consistent the contact hypothesis is with the F/T observation, and the system should either raise a consistency alarm or push `mode_probability` toward uniform. **So `wrench_ext` is not a redundant field alongside `contacts[]`, it is an independent aggregate observation that serves as a consistency constraint on contact-set reconstruction.** This is also the physical basis for the F ↔ T, F ↔ V, F ↔ P edges in §8.7's consistency graph.

### 6.8 Schema evolution and interface compatibility

If the interface really exists, it must answer one very concrete engineering question: **can an old policy consume a new sensor version?**

```text
v1  contact: p, n, force
v2  contact: p, n, force, slip_probability, covariance
v3  contact: geometry{type, pose, extent}, wrench{f, m}, mode_distribution
```

An interface is an API, so it must handle API versioning. The minimum set of conventions:

```text
Interface compatibility
├── version tag            # every slot carries a schema version
├── optional fields        # old consumers ignore new fields;
│                          # new consumers handle missing with defaults
├── backward compatibility # v2 consumers read v1 slots; v1 safely degrades
│                          # when reading v2
├── unit / frame validation # schema checker, asserted at startup
└── graceful degradation   # inflation / default / alarm policy when a field is missing
```

Without this layer, the analogy "state interface = API" does not hold. This layer is also the prerequisite for §8.9's interface-swap benchmark.

## 7. Fusion Failure Modes and Diagnostics

Five failure classes a robotics reviewer will reach for first. Each has symptom / diagnostic / mitigation. §2 and §6 already establish the shared vocabulary; this section does not re-derive.

### 7.1 Modality collapse: attention quietly reverts to vision dominance

**Symptom**: training loss looks fine, validation success looks fine; cover vision at eval and behavior barely changes — the other modalities were never really used. **Diagnostic**: the primary metric should be **modality ablation gain** (§8.1) — if $\Delta_{\text{mod}}$ is near zero, that channel is effectively dead inside the policy. 3D-ViTac [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) matters for exactly this reason: it provides an explicit reference for the metric. Attention-weight visualization is fine as **auxiliary diagnostic visualization**, but **must not be used as evidence of modality contribution** — attention ≠ causal importance. A low-attention modality can still be essential; a high-attention modality can still be removable. This is a classic interpretability trap. **Mitigation**: (a) train with modality dropout, see §7.5; (b) auxiliary supervision per modality (tactile encoder must independently predict slip events, not only feed the policy); (c) evaluate with 9/11 §5.2.1's feedback-value benchmark rather than raw success rate.

### 7.2 Temporal smearing: everything force-fit onto 30 Hz

**Symptom**: policies do not learn slip detection, transient contact, or impact tasks — because §2.1's downsampling has already erased them. **Diagnostic**: compare "30 Hz fusion" against "native-rate fusion" success rate; or use $\Delta_{\text{tail}}$ restricted to hard contact conditions. **Mitigation**: see §5.4 — treat time-constant differences as modeling assumptions; use different rates on different layers; put events on a bus; never downsample high-frequency signals at the interface layer.

### 7.3 Frame confusion: model learns a coordinate-frame coincidence

**Symptom**: model is great at training-time pose, camera placement, and tool model; nudged and it collapses. **Diagnostic**: eval set applies **coordinate-frame perturbations** (split into four classes in §8.4). **Mitigation**: state interface carries `frame_id` explicitly, downstream consumers must transform to read; training treats frame perturbation as part of domain randomization (Tobin et al. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)).

### 7.4 Semantic leakage: raw visual path bypasses slot-defined contact semantics

**Symptom**: slots are defined as "contact geometry", yet policy behavior depends heavily on visual appearance (same object with a new background loses points). **Diagnostic**: replace the visual contribution to the slot with a "minimally sufficient" synthetic slot (e.g. substitute predicted contact with ground-truth contact) and observe the change. **Mitigation**: the previous version was too absolute here. This article does **not** argue "raw pixels must not enter the policy" — that would immediately draw a VLA reviewer's rebuttal, since modern VLAs routinely go image → vision encoder → token representation → policy. What this article argues is: **for the contact-related state defined by the state interface, no raw-visual shortcut should bypass the slot encoder**. In other words, the sole authorized path for contact semantics is the slot encoder — not that visual features are disallowed from the policy. This aligns with 9/11 §3.5's "representation interface vs learned representation", and is more implementable.

### 7.5 Missing / degraded modalities: crash on failure, or silently degrade without alarm

**Symptom**: training always has all modalities present; a single tactile frame drop shifts policy behavior dramatically. **Diagnostic**: run **modality masking / dropout tests** — drop each modality at probability p, plot success rate curves. Modality masking / dropout has become a **common training strategy** for missing-modality robustness — see Maiga et al. MMP [arXiv:2410.03010](https://arxiv.org/abs/2410.03010) and related work. **But dropout ≠ sensor failure**. Real degradations are not i.i.d. Bernoulli(p):

```text
Modality missing     ── sensor unplugged
Modality stale       ── value present, timestamp not refreshed
Modality delayed     ── timestamp correct, arrival late
Modality corrupted   ── single-frame outliers (camera blur, stray signal outside the contact patch)
Modality biased      ── systematic drift (F/T tare fails, IMU thermal drift)
Modality noisy       ── Gaussian or non-Gaussian noise amplitude rises
```

The interface must express all six states, not just "present / absent". §6.1's `timestamp / age / availability / lifecycle / uncertainty` are there exactly to make stale / delayed / biased **detectable in the data format itself** — this is a requirement of interface design, not a benchmark trick.

## 8. Benchmark and Evaluation

Consolidates §7's scattered diagnostics into an executable benchmark skeleton. The guiding principle follows 9/11 §5.2.1's feedback-value stance — **the benchmark should measure how much irreplaceable signal a modality actually provides, whether the interface can carry disagreement, and whether encoders and policies can be swapped behind the interface — not how well a model memorizes the training distribution**.

### 8.1 Modality ablation gain

$$
\Delta_{\text{mod}} = S(\text{base} + \text{mod}) - S(\text{base})
$$

$S$ is task success rate; base is a fixed policy without that modality. Report $\Delta_{\text{mod}}$ per task family. Small $\Delta_{\text{tactile}}$ on stable grasps is expected; large $\Delta_{\text{tactile}}$ on slip recovery, precision insertion, and wiping is what you should see.

### 8.2 Missing-modality robustness: information gain + graceful degradation

The previous version's "healthy systems have flat dropout curves" is too strong. If tactile is genuinely critical to the task, `all → 95%, drop tactile → 45%` may not be a health problem — it may just mean tactile is irreplaceable. The right thing to measure is **robustness against information loss**, not "flatness under dropout". Report two quantities together:

**Information gain** (upper bound on what the modality contributes):

$$
\Delta_m \;=\; S(M) - S(M \setminus m)
$$

**Graceful degradation** (whether the drop is smooth as you lose fractions of the modality):

$$
G_m(p) \;=\; \frac{S(M, p) - S(M \setminus m)}{\Delta_m}
$$

where $p$ is dropout rate. A good system can have **large $\Delta_m$ (the modality matters a lot) but $G_m(p)$ decreases smoothly from 1 to 0** — the modality is important, yet partial loss does not cliff the system. That is a much more discriminating view than "flat is healthy", and it matches realistic sensor importance distributions.

### 8.3 Degradation modes beyond dropout

Corresponding to §7.5, extend beyond Bernoulli dropout to: missing / stale / delayed / corrupted / biased / noisy. Report a success-rate curve per degradation class. **This is the direct test of whether the interface actually handles uncertainty / provenance / validity**.

### 8.4 Temporal & frame perturbation (relativize, do not hard-code)

The previous version wrote "vision ±5 ms, tactile ±20 ms, F/T ±5 ms". This looks engineering-concrete but camera exposure, tactile latency, network delay, controller cycle, and timestamp uncertainty differ completely across systems — hard-coded numbers kill portability. Use **relative perturbation**:

$$
\delta t \;\in\; \{0.25\,\Delta t,\; 0.5\,\Delta t,\; \Delta t,\; 2\,\Delta t\}
$$

where $\Delta t$ is the **task-relevant temporal bandwidth** (e.g. contact-mode transition characteristic time, or policy control period). Equivalently: "perturbation magnitude = fraction of task-relevant temporal bandwidth".

**Frame perturbation** should also split into four classes rather than a single "rotate a few degrees, translate a few centimeters":

```text
Registration robustness
├── calibration noise        ── T̂ = T · ΔT, ΔT is a small Gaussian / uniform jitter
├── calibration drift        ── slow long-horizon drift (thermal, mechanical creep)
├── sensor remounting        ── new tool / new camera / new sensor mount, geometry redefined
└── frame convention mismatch── base ↔ world, sensor ↔ tool, extrinsic sign flip
```

The first three are "parameter errors"; the fourth is a "convention error", and downstream failure modes look entirely different. This is much more informative than plain domain randomization.

### 8.5 Slot-layer fidelity, measured independently of any policy

If the state interface is the thesis, **slot accuracy must be measured independently of the policy**: contact position IoU / distance error against ground truth; slip detection AUROC; contact-mode macro-F1; **uncertainty calibration** — reliability diagram or negative log-likelihood for continuous parts, Brier score or expected calibration error (ECE) for categorical parts; on patch-type slots also extent error and center-of-pressure error; on track_id also identity preservation metrics (ID-switch count, MOTA / IDF1). These are "quality before fusion" metrics — if they fail, high policy success rate is likely overfitting.

### 8.6 Cross-modal contradiction test (the most recommended benchmark axis)

If the interface exists to help the system be **correctly uncertain when modalities disagree** (rather than averaging into a confident wrong answer), then the benchmark must inject cross-modal contradictions deliberately. The previous version's example — "vision classifies the object as rigid, tactile local deformation is consistent with a compliant object" — is **not actually a contradiction**: a metal object with a rubber coating can be globally rigid and locally compliant at the same time. Both readings may be correct.

Real benchmarkable contradictions must be **on a single verifiable variable**:

```text
Spatial disagreement    vision reports contact at A, tactile reports at B (5 / 10 / 20 mm apart)
Temporal disagreement   F/T reports contact start at t, tactile reports at t + Δ
Force disagreement      tactile predicts 2 N normal, F/T-consistent reconstruction requires 8 N
Kinematic disagreement  proprio reports EE pose X, vision registration reports Y
```

Then measure four things: (1) **Contradiction detection** — does the model flag "these channels disagree"; (2) **Uncertainty calibration** — is the reported uncertainty proportional to actual disagreement; (3) **Source attribution** — after the fact, can you localize which channel was at fault; (4) **Recovery** — how much success-rate drop after handling. **On the recommended interface, §8.6 and §8.7 — not §8.2 — are the first-class citizens.**

### 8.7 Consistency graph: from "is there a contradiction" to "which edge is suspicious"

Pushing further, $\mathcal{L}_{consistency}$ should not be a scalar distance. It should be a graph:

```text
           Vision
          /      \
      Tactile --- F/T
          \      /
         Proprio
```

Each edge $(i, j)$ carries a consistency constraint $\mathcal{C}_{ij}$ — $\mathcal{C}_{VT}$ is a residual on contact location between vision and tactile; $\mathcal{C}_{TF}$ is wrench consistency between tactile and F/T; $\mathcal{C}_{FP}$ is the $J^T F$ residual between F/T and proprio; $\mathcal{C}_{VP}$ is EE-pose consistency between proprio and vision. Total loss:

$$
\mathcal{L}_{\text{consistency}} \;=\; \sum_{(i,j)} w_{ij}\, \mathcal{C}_{ij}
$$

This way the system not only says "there is disagreement", but also answers **"which observation-graph edge is most suspicious"** — rank $w_{ij} \mathcal{C}_{ij}$ per edge and the top one points at the most likely sensor fault. This connects naturally to `provenance`, `diagnostics`, and `sensor fault isolation`. It upgrades §6.4's 7-tuple interface formula so that **Provenance and Validity become benchmarkable**, not just declared.

### 8.8 Oracle-slot / Estimated-slot / End-to-end: three-baseline comparison

To actually test the claim "state interface matters more than end-to-end fusion", you need **three baselines side by side**:

```text
A · Oracle interface   : ground-truth contact slots → policy
B · Estimated interface: sensor → state estimator → estimated slots → policy
C · End-to-end fusion  : sensor → fusion policy (no explicit interface)
```

Then compare:

$$
\underbrace{S_A}_{\text{interface ceiling}} \quad \underbrace{S_B}_{\text{interface + estimation}} \quad \underbrace{S_C}_{\text{end-to-end}}
$$

At minimum, this experiment distinguishes whether performance loss comes from **perception / interface** ($S_A$ high, $S_B$ drops), from the **policy** (both $S_A$ and $S_B$ drop), or whether "the interface abstraction simply did not help" ($S_C \geq S_B$). **$S_{\text{total}} \approx S_{\text{interface}} \times S_{\text{downstream}}$ — this factorization is the most direct empirical test of the article's core claim.** Without this baseline, all other benchmarks are still describing whether the interface is convenient, not whether it is *necessary*.

### 8.9 Interface swap: an interoperability benchmark

The single experiment this article would most like to see run. The core claim is "once the interface is stable, encoders and policies can be swapped". Then actually measure it:

```text
Encoder A          Encoder B
   ↓                   ↓
┌────── Slot schema (one contract) ──────┐
   ↓                   ↓                   ↓
Policy A           Policy B           World Model
```

Three concrete swaps:

1. **Sensor encoder swap**: swap tactile encoder A → encoder B (different backbone, even different sensor family). Slot schema unchanged, policy weights unchanged. Measure transfer.
2. **Policy swap**: MLP policy → Transformer policy. Slot unchanged. Measure downstream retraining cost.
3. **Consumer swap**: policy ↔ world model ↔ controller — three consumers share one slot, trained independently. Measure whether the interface is genuinely reusable.

If all three work without redesigning the middle layer, that is much stronger evidence than another success-rate number. **This is really an interface interoperability benchmark — turning "interface" from metaphor into a measurable property**. §8.9 and §6.8 form a pair: schema evolution guarantees forward compatibility, interface swap guarantees horizontal pluggability.

### 8.10 Fit with existing benchmarks

RoboCasa / LIBERO / ManiSkill3 / BEHAVIOR — mainstream manipulation benchmarks are **mostly vision-centric with tactile absent**. This section does not conclude; it only **recommends**: package the ten metric families above (ablation / information-gain + graceful-degradation / degradation modes / relative temporal + registration / slot fidelity / contradiction / consistency graph / oracle-vs-estimated-vs-e2e / interface swap) as an optional plug-in, giving existing benchmarks a "feedback-value + interface-quality" layer. 9/11 §5.2.1's A/B/C/D arms can be borrowed directly.

## 9. Relation to VLA and World Models

### 9.1 VLA today: what is missing is "a scaled pretraining ecosystem on par with vision-language"

§3.1 already corrected — treating RT-2 / OpenVLA / π0 as "concat" is inaccurate. A more accurate observation, **using RT-2, OpenVLA, and π0 as representative systems**: existing VLA scaled-pretraining ecosystems center on vision-language observation and robot state / action; tactile and F/T **have not yet formed a public, cross-task, cross-embodiment large-scale pretraining ecosystem comparable to vision-language**. This is an ecosystem-level judgment, not something a single paper can prove — the evidence chain should be "concrete systems → observation → this article's induction", not "a few papers prove the whole VLA community is this way".

This absence is not a "forgot to add it". **Data and interface are both missing** — tactile and F/T have neither an ImageNet-class public annotated corpus, nor a standard slot-schema contract like §6.1. The open question: **does the next VLA wave scale up more vision+language, or does it plug in a tactile / F/T state interface?** This article bets on the latter — but not "add tactile as another channel"; the bet is to plug §6's structured belief interface into VLA's input layer. Qi et al. T-Dex [arXiv:2309.09979](https://arxiv.org/abs/2309.09979) (CoRL 2023) and Lee et al. [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) can be seen as early forms of this route.

### 9.2 World models: if contact mode is causally relevant, keep an explicitly identifiable contact / event state

RSSM / DreamerV3 [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) style latent dynamics typically assume smooth, differentiable state transitions. 9/11 §5.1 discussed how contact events are mode switches in hybrid dynamics. **A more careful phrasing** — modern world models can and do use discrete latents, categorical latents, hybrid state, event-conditioned dynamics, multiple latent heads, mode-conditioned transitions — so this article's point is **not** "world models must predict `contact[]` directly". A world model may predict each piece separately:

$$
z_{t+1}, \quad p(m_{t+1} \mid z_t, a_t), \quad p(C_{t+1} \mid z_t, a_t)
$$

or use a hybrid latent of `z_continuous + z_discrete + contact state`. What this article argues is:

> **If contact mode is causally relevant to the task's decision-making, the world model should preserve some explicitly identifiable contact / event state, rather than force it to exist entirely inside an opaque continuous latent.**

This connects §9.2 back to §6 naturally: **the world model does not have to consume every slot field, but it should have at least one head that explicitly predicts the mode / event portion of the slot** — as a discrete latent, or as $p(m_{t+1})$. The policy layer is free to use learned representations (embedding the slot for the policy is reasonable — but that embedding is a downstream consumer, not an upstream data format). The two connect via §6's interface: the world model predicts the next-time-step slot, the policy consumes the current slot.

### 9.3 One-liner

**VLA lacks a scaled pretraining ecosystem for tactile / F/T; world models lack an explicitly identifiable contact / event state. These are the same gap: nobody has done work at the interface layer, everyone is betting that "with more data, the model will figure it out".**

## 10. Conclusion

This article began with the misconception that "multimodal fusion is a question of architecture" and pulled the discussion back to the floor. What robotics actually lacks is not a stronger fusion operator, but a **structured belief interface sitting between heterogeneous observations and downstream models**, one that explicitly carries **value / semantics / frame / time / uncertainty / provenance / validity**. Vision supports a common data interface not because its encoders are cleverer, but because it nailed §2's three preparations first and grew a whole ecosystem — GPUs, cheap sensors, internet-scale data, annotation — on top. What vision actually formed is **a family of highly interoperable task-level representation conventions, not one unique schema**. Tactile / force-torque / proprioception's effective observation rates, frames, raw representations, and semantic conventions have not converged, so all cross-attention and shared-latent effort collapses on §7's five failure modes.

This article does **not** oppose cross-attention, does **not** oppose shared latent, and **does not oppose end-to-end learning**. What it opposes is **letting state estimation, cross-modal composition, and control all happen implicitly inside one undiagnosable, non-reusable latent interface**. The minimum route proposed: a contact-set-centric structured belief interface; raw→slot mapping in sensor-specific perception; slots carrying **point / patch / region geometry**, with **continuous covariance and categorical probability kept separate**, **track_id as a hypothesis rather than intrinsic identity**, and **uncertainty / provenance / timestamp / availability / lifecycle as interface parts**; `wrench_ext` as an independent aggregate observation serving as a consistency constraint on contact-set reconstruction; interface itself versioned with graceful degradation; policy, world model, and controller interoperating through the interface, but the interface need not be the only information path; and modality dropout, cross-modal contradiction, consistency graph, oracle-vs-estimated-vs-end-to-end baselines, and interface swap written into training and benchmark as first-class citizens.

**This route is not the final answer; it is a head-on response to 9/11's open question "tactile lacks a reusable intermediate representation" — agree on the interface first, then talk about architecture**. If 9/11's three tags are *Action-conditioned observation · Contact-state representation · Closed-loop value*, this article's three tags are:

> **Register before compose · Expose belief at the interface · Design for disagreement**

One sentence tying them together: **before composing, register; at the interface, expose belief / uncertainty / provenance / validity, not just values; design the system to handle not only "did not see" but also "saw different things".**

Next article (9/13) walks downstream of the interface: **dexterous hands and in-hand manipulation** — put this article's state interface against T-Dex / DextrAH / LEAP and see whether it can support their architectures, and what the cost structure behind "many robots can hold a hand, few are actually doing dexterous" really is.

## Sources

Citations here follow the four thesis lines, not "stack sensor papers".

### A · Cross-modal representation and fusion paradigms (§3 / §4)

- Baltrusaitis, Ahuja, Morency, *Multimodal Machine Learning: A Survey and Taxonomy*, TPAMI 2019 · [arXiv:1705.09406](https://arxiv.org/abs/1705.09406)
- Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language Sequences*, ACL 2019 · [arXiv:1906.00295](https://arxiv.org/abs/1906.00295)

### B · Empirical visuo-tactile fusion line (§4 / §7.1 / §9.1)

- Calandra et al., *More Than a Feeling: Learning to Grasp and Regrasp using Vision and Touch*, RA-L 2018 · [arXiv:1805.11085](https://arxiv.org/abs/1805.11085)
- Lee et al., *Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks*, ICRA 2019 · [arXiv:1810.10191](https://arxiv.org/abs/1810.10191)
- Huang et al., *3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing*, CoRL 2024 · [arXiv:2410.24091](https://arxiv.org/abs/2410.24091)
- Qi et al., *General In-Hand Object Rotation with Vision and Touch* (T-Dex), CoRL 2023 · [arXiv:2309.09979](https://arxiv.org/abs/2309.09979)

### C · Cross-sensor / cross-modal unified representation (§3.4 / §5.1)

- Zhao et al., *A Touch, Vision, and Language Dataset for Multimodal Alignment* (TVL / Binding Touch to Everything), ICML 2024 · [arXiv:2402.13232](https://arxiv.org/abs/2402.13232)
- Feng et al., *AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors*, 2025 · [arXiv:2502.12191](https://arxiv.org/abs/2502.12191)
- Lin et al., *9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation*, ICRA 2023 · [arXiv:2308.14277](https://arxiv.org/abs/2308.14277)

### D · Robustness with modality masking / dropout (§7.5 / §8.2 / §8.3)

- Maiga et al., *MMP: Towards Robust Multi-Modal Learning with Masked Modality Prior Fine-Tuning*, 2024 · [arXiv:2410.03010](https://arxiv.org/abs/2410.03010)
- *Robust Multimodal Learning with Missing Modalities via Parameter Projection*, 2023 · [arXiv:2310.03986](https://arxiv.org/abs/2310.03986)

### E · VLA and World Models (§9)

- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
- Hafner et al., *Mastering Diverse Control Tasks through World Models* (DreamerV3), Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

### F · Sim-to-Real / Domain Randomization background (§7.3 / §8.4)

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)

### G · Continuation of 9/11 · Contact state and impedance (background)

Works already cited in 9/11 and reused by this article without repeating links: Hogan's impedance trilogy; Posa-Cantu-Tedrake IJRR 2014 (hybrid contact-mode trajectory optimization); Lee 1810.10191; Qi 2309.09979; Huang 2410.24091; Zhao 2402.13232; Feng 2502.12191. For precise sources see 9/11's Sources section.

---

> **Related reading**
>
> - [The Hand Robots Don't Have: Tactile and Force Sensing in Embodied AI](/en/articles/2026-09-11-tactile-force-sensing/) — predecessor; tactile and force control treated on their own
> - [Sim-to-Real Methodology](/en/articles/2026-09-10-sim-to-real-methodology/) — §7.3 frame perturbations and §8.4 temporal / registration perturbations borrow its domain-randomization lens
> - [Why Robot Data Is Harder than LLM Data](/en/articles/2026-09-09-robot-data-scaling/) — the "where does supervision come from" question in §3.4's shared-latent line is fundamentally a data-scaling question
> - [VLA vs World Models](/en/articles/2026-09-07-vla-world-models/) — §9 is one specific angle: VLA lacks a scaled tactile / F/T pretraining ecosystem; world models lack an explicitly identifiable contact / event state
