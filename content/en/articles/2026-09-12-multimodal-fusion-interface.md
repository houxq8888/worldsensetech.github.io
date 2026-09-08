---
title: 'Stacking Sensors Is Not Fusing Them: Multimodal Robotics Lacks an Interface, Not a Model'
slug: "2026-09-12-multimodal-fusion-interface"
date: 2026-09-12
draft: false
categories: ["Embodied AI", "Multimodal Perception"]
tags: ["Embodied AI", "Multimodal Fusion", "Visuo-Tactile", "Force/Torque", "Proprioception", "Representation Interface", "Multimodal State Estimation", "Hybrid State Estimator", "Structured Belief", "Registration", "Cross-attention", "Modality Dropout", "VLA", "World Model", "Frame Alignment", "Time Alignment", "Uncertainty"]
description: 'Multimodal fusion is usually presented as a question of "which attention architecture", but in robotics the real bottleneck is upstream: Vision / Tactile / Force-torque / Proprioception have never agreed on temporal and spatial registration, task-relevant semantic projection, or on validity, provenance, and uncertainty. This piece splits fusion into perception → registration → semantic projection → hybrid multimodal state estimation → structured belief interface, and argues that what is missing is not a better fusion operator but a structured belief state — carrying value, semantics, frame, time, uncertainty, provenance, and validity — sitting between heterogeneous observations and downstream policy / world model / controller / diagnostics. Contact set is an instance of this interface for contact-rich manipulation, not the definition of the interface itself; the interface does not prescribe a Bayesian inference algorithm — belief just means uncertainty-aware state, and EKF / factor graph / learned filter are all valid implementations. In the benchmark section, interface swap moves to the first primary experiment, joined by oracle-slot vs estimated-slot vs end-to-end baselines, a consistency graph, and a formal cross-modal contradiction measure. Three design principles: Register before compose · Expose belief at the interface · Design for disagreement.'
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

Picture a bimanual robot fitted with an RGB camera, GelSight fingertips, wrist six-axis force/torque sensors, and joint encoders on every link. The hardware bill of materials looks genuinely "multimodal". Yet when you hand that rig to a policy, much of the published literature tells you "just fuse them with cross-attention and you're done". That runs in a demo; **in real deployment you typically hit one of four failure modes**: a modality drops frames, a sensor dies, coordinate frames drift, or the model quietly converges to one dominant modality and treats the rest as noise. These four are not engineering footnotes. **They all trace back to the same root cause: the modalities never agreed on a structured belief interface — explicitly expressing time, frame, semantics, uncertainty, provenance, and validity — that policy, world model, controller, and diagnostics can consume in common.**

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
              ┌─────────────┬───────────┼───────────┬─────────────┐
              ▼             ▼           ▼           ▼             ▼
           Policy      World Model  Controller  Diagnostics  Safety layer
              │             │           │           │             │
              └─────────────┴───────────┼───────────┴─────────────┘
                                        ▼
                                   Action a_t
```

Three things this diagram insists on, and the architectural thesis of the entire article: **fusion is not a network module — it is a complete hybrid-state-estimation-plus-interface-contract stack** ("hybrid" is deliberate, because the state to be estimated $x_t$ naturally mixes continuous $(q, \dot{q})$, a set-valued contact state $C_t$, a discrete mode $m_t$, and a task goal $g_t$); **registration and semantic projection are two different preparation steps before state estimation** — registration is a measurement-layer problem described by explicit temporal / geometric / calibration variables and should be solved by calibration or estimation rather than silently pushed into a downstream fusion network, whereas semantic projection is task-conditioned representation learning; **the interface is a contract, not a learned embedding** — every field carries a unit, a frame, a timestamp, an uncertainty, a source, and a validity state, and its optimization target is **interoperability across consumers**, not "friendliness to one specific downstream model".

One sentence that carries the central claim:

> **The missing abstraction is not a better fusion operator, but a structured belief state for heterogeneous, multi-rate observations.**

Or, as the boxed architectural thesis of this article:

$$
\boxed{\;\text{Sensor-specific observations} \;\rightarrow\; \underbrace{\text{Structured Belief}}_{\text{stable interface}} \;\rightarrow\; \{\text{Policy, World Model, Controller, Diagnostics}\}\;}
$$

Once framed this way, VLA, world models, tactile foundation models, modality dropout, cross-modal contradiction, F/T constraints, and contact-set slots all become **different instances of the same main line, not six parallel talking points**.

**This article does not propose a new fusion operator**. It proposes three system-level design principles, and every section below is a concrete unfolding of them:

1. **Register before compose** — pull temporal / spatial registration and semantic projection out of learned fusion, and expose them as calibratable, estimable, unit-testable system variables.
2. **Expose belief at the interface** — use a structured state carrying semantics, frame, time, uncertainty, provenance, and validity as the stable boundary between multiple consumers; "belief" here just means *uncertainty-aware state*, and Bayesian posterior inference is one implementation option, not a required property of the schema.
3. **Design for disagreement** — treat missing modalities, cross-modal contradictions, and consistency reasoning as interface-design goals, not as post-deployment exceptions.

**Contact set is an instance of this interface for contact-rich manipulation, not the definition of the interface itself** — this sentence runs through the whole article, and it is also why this piece positions itself as an *architecture position paper* rather than a *tactile survey*.

## 1. "Multimodal" is not "Multiple Sensors"

The easiest way to derail this topic from the very first paragraph is to equate "a few more sensors" with "multimodal". That equation does not hold.

**"Modality" itself has no unique physics definition — it is a product of the analytic lens.** For this article's taxonomy we fix four elements: **measurement space, physical origin, noise model, and update semantics** (triggering / sampling / clock). Together these four decide whether a data stream can be treated as "the same kind of thing". Different authors can draw the boundary a bit looser or a bit tighter, but as long as it is declared up front, subsequent discussion will not drift.

Under this taxonomy, several commonly mislabelled examples: a wrist camera plus a head camera are **same-modality multi-view**, not two modalities; the four tiny cameras inside a GelSight fingertip are **internal structure of a single tactile modality** whose measurement space is "elastomer surface deformation field" — even with a CNN in the middle, the downstream semantics are contact geometry, not object detection; joint encoders plus motor current are two **observation channels of a single robot-state modality** sharing the same latent physical state — treating them as two independent modalities and fusing them will teach attention nothing but redundancy. Conversely, **Vision / Tactile / Force-torque / Proprioception** are four genuinely distinct modalities under this taxonomy; their measurement spaces, coordinate frames, and effective observation rates are summarized in the §5 table. Some works add audio, thermal, gas, or ultrasound as fifth / sixth modalities, and the same classification logic applies. 9/11 §2.1 already split contact sensing into Tactile / Force-torque / Proprioceptive using a taxonomy tree; this article keeps that same three-way partition and restricts the discussion to V + T + F + P.

## 2. Three Steps Before Fusion: Registration and Semantic Projection

This is the most technical — and most often skipped — section in the article. First, a **terminology tightening**: **"registration" and "semantic projection" are different in kind**, and collapsing both into a single "alignment" makes readers think they can be solved by pure math transforms alone.

```text
Pre-fusion pipeline
├── Registration        (measurement-layer; explicit time / geometry / calibration variables)
│   ├── Temporal registration
│   └── Spatial registration
└── Semantic projection (learning task-relevant state representations)
    └── Raw observation → task-relevant state
```

**Registration is primarily a measurement-layer problem: it should be described by explicit temporal, geometric, and calibration variables as much as possible, and should be solved via calibration or estimation rather than silently handed to a downstream fusion network.** That is this article's formal definition, and it is a notch weaker than "hard constraint / closed-form" — real robots routinely have registration as an *estimation* problem (online calibration, time-varying extrinsics, compliant sensor mounting, tactile elastomer deformation, thermal drift, synchronization / latency estimation) — but its *nature* remains measurement-layer explicit variables, not task semantics. **Semantic projection** sits closer to perception and state estimation, and it is the main character of the §6 structured belief interface that follows. Both are *before* fusion; they differ in nature and in who owns them.

### 2.1 Temporal Registration

The default time constants of the four streams differ by orders of magnitude: vision ~15–60 Hz, tactile (image-based) ~30–200 Hz, tactile (taxel / array) ~500 Hz – kHz, F/T ~500 Hz – 1 kHz, proprio ~ hundreds of Hz – kHz. A commonly confused point: these are **effective observation rates**, not internal sensor sampling rates — a "kHz-taxel-sampling" camera-based tactile sensor whose output image is still 30 Hz gives the fusion layer an effective rate of 30 Hz.

Resampling everything onto a common low-rate policy clock (usually the vision one) is a standard but risky baseline. **For contact-rich control, this compresses part of the high-frequency events into invisible aliases** — slip detection, transient contact force peaks, joint impacts — these "event-like" signals often happen between two vision frames; downsampling them is equivalent to throwing them away. A more robust architecture is **multi-rate coexistence**: the vision policy can run at 30 Hz, force / tactile-driven compliant control keeps its native rate, and event signals go through a dedicated event bus. Three time details must be pinned down in the interface layer: **timestamp semantics** (sensor timestamp vs arrival timestamp vs host timestamp — a 5–20 ms gap is enough to shift "the moment the finger closes on the cup") and **clock drift** (camera, controller, and host each run on their own crystal; a few tens of milliseconds per minute is common, and one overnight run can teach a fusion model a spurious lead/lag correlation) and **event-based vs periodic** (a slip trigger is a sparse event, not a periodic sample — stuffing it into a uniformly indexed buffer destroys the event-density signal). **Decision rule: express the time-constant difference as an explicit modeling assumption, do not hide it behind downsampling.** This is the same judgment as 9/11 §4.1.

### 2.2 Spatial Registration

The four streams naturally live in four different frames: Vision — camera frame; Tactile — sensor frame; F/T — sensor frame + tool frame; Proprio — base / world frame. Getting them into a common fusion layer requires at least three things: **hand-eye calibration** ($T^{\text{cam}}_{\text{base}} \in SE(3)$ — any collision can knock it off by a fraction of a degree to several degrees); **sensor-mount calibration** (fingertip sensor frame → link frame offset $T^{\text{sens}}_{\text{link}}$); and **wrench transformation with gravity / inertial compensation** (bring the sensor-frame wrench to base or world; the exact formula is in §5.2).

A common symptom of poor spatial registration: model performance drops as soon as you swap the tool, reposition the camera, or bump the robot — because what the model learned is a *correlation under a particular sensor frame*, not the task. A subtler issue is **relative pose vs absolute pose**: contact physics fundamentally depends only on relative geometry ("who touches whom"), yet many policies feed in the end-effector's absolute world-frame position and end up learning a bundle of degrees of freedom they should not have.

### 2.3 Semantic Projection

Strictly speaking this step is not "registration" — it is "mapping observations into a common task-semantic coordinate". Vision → object / scene semantics; tactile → local-contact semantics; F/T → aggregate-contact semantics; proprio → self-state semantics. 9/11 §2.1 split signal → meaning into nine layers (Sensor → Calibration → Raw obs → Contact perception → Contact geometry & wrench → Contact mode & physical state → Task-relevant belief → Policy / controller → Action → New contact). **The key multimodal-fusion question is: on which layer should composition happen?** A common mistake is composing at the raw layer (concatenating four tensors and feeding them to a network) — that forces the policy to learn §2.1 / §2.2 / §2.3 all by itself, at enormous sample cost. A more reasonable pattern is **letting each stream complete raw → perception → contact geometry independently, and composing only on the four semantically converged variables "contact events, wrench, robot state, task belief"** — that is the §6 structured belief interface.

**Section takeaway**: temporal / spatial registration are measurement-layer explicit variables; semantic projection is the shared responsibility of perception and state estimation. Skimp on any one of the three, and no downstream fusion architecture is fancy enough to compensate.

## 3. The Landscape of Existing Fusion Paradigms: Different Mechanisms, Different Layers

Baltrusaitis, Ahuja, and Morency's classic survey [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) organizes multimodal ML into representation / learning / feature selection / fusion / application layers; this section borrows the fusion layer's classification. **Important positioning**: the paradigms below are **not mutually exclusive architectural choices, but mechanisms on different axes** — a mature system often uses several of them simultaneously.

| Mechanism | What it addresses | Which §0 layer it belongs to |
| --- | --- | --- |
| Modality-specific encoder | Each stream's raw → local observation | Perception |
| Temporal / spatial registration | Clock, SE(3), coordinate frame | Registration |
| Cross-attention | Learned composition | Mostly inside the state interface |
| Shared latent / VLT-style | Cross-modal representation alignment | Semantic projection |
| Late / decision fusion | Decision-level composition | Policy / controller |
| Structured state interface | Semantic schema and contract | Belief interface |

This article does not oppose cross-attention; it **limits its scope**: **cross-attention is not guaranteed to recover explicit temporal / spatial / semantic registration from raw data**. Attention can compose already-aligned representations, but without explicit constraints it usually cannot reliably "learn" registration along the way. This sentence is the hinge of §3 and the entire article.

**Early concat (raw / embedding layer)**: concatenate the four encoders' outputs into a state vector, feed to an MLP or Transformer. Low barrier to entry; with enough data a deep network will pick up some alignment correlations on its own. Downsides: concat masks the time-constant differences and easily learns spurious lead / lag; any dropped frame becomes zero or hold-last-value, both out of distribution; tactile and visual encoders are structurally very different, so gradients pollute one another; and the policy breaks whenever a missing-modality combination not seen during training appears.

**A paragraph that needs care — VLA**. Grouping RT-2 / OpenVLA / π0 all as "vision + proprio + language concat" is inaccurate. **RT-2** [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) expresses the action as **text tokens** and jointly fine-tunes with the VLM; **π0** [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) is a VLM backbone plus a **proprioception token** plus a **noisy action chunk**, decoding actions via flow matching; **OpenVLA** [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) explicitly documents multi-camera, depth, and proprioceptive-state encoding branches in its public configuration. **Looking at these representative public systems, we observe**: what existing general VLAs have in common is "vision-language pretraining as the primary scaling entry point, injecting robot state as an additional representation into the policy"; **tactile / force-torque has not yet formed a public, cross-task, cross-embodiment scaled pretraining ecosystem comparable to vision-language**. This absence is not "they forgot to add it" — once added, all three steps in §2 have to be reopened, and tactile / F/T currently lack a corresponding pretraining data scale. This judgment is expanded again in §9.1.

**Cross-attention / Transformer fusion**: treat each stream as a token sequence, then run self / cross-attention above. Tsai et al.'s Multimodal Transformer [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) is the classic starting point on this line, explicitly handling "different modalities have different time granularities". A modern multi-modal transformer can readily be combined with timestamp / positional embeddings, relative temporal encoding, modality embeddings, frame-aware features, modality-specific encoders, modality dropout / masking, per-modality auxiliary losses, and attention masks. **The real problem is not "attention breaks", but that if these ingredients are not written down as part of the interface contract, attention will happily "learn" registration along the way, and what it learns is a coincidence of the data distribution rather than a portable interface.** In a mature system, cross-attention should sit **after perception and registration have already delivered input onto the state interface, and only do composition inside the state interface**.

**Late / decision-level fusion**: each stream produces a sub-policy (or a sub-value / sub-action proposal), which the decision layer combines via weighted voting or confidence gating. This is closer to traditional robotics — vision for coarse, F/T for fine, tactile for slip recovery, proprio for nominal trajectory tracking. **A common engineering pattern** is that any single stream going down only drops one proposal rather than collapsing the whole system, making it easier to add a safety layer. Downsides: low-level coupling information is lost — cross-modal joint facts ("viewed together, is this contact a stable slide or stick-slip?") are hard to reconstruct at the decision layer; weighting rules are either hand-designed and unscalable, or learned, which pushes you back into the early-concat issues. 9/11's combination of impedance control + coarse visual localization + tactile slip detection is, at heart, exactly this kind of late fusion.

**Shared latent / VLT-style alignment**: contrastive / distillation / CLIP-style objectives pull representations from different modalities into a common latent space. Representative works: **TVL / Binding Touch to Everything** [arXiv:2402.13232](https://arxiv.org/abs/2402.13232) (Zhao et al., ICML 2024) builds cross-modal alignment via language over ~44K vision–touch pairs; **AnyTouch** [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) (Feng et al., 2025) learns a unified static-dynamic representation across **heterogeneous visuo-tactile sensors**; **3D-ViTac** [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) (Huang et al., CoRL 2024) reports in its own experiments a significant visuo-tactile vs vision-only gain; and Lee et al.'s *Making Sense of Vision and Touch* [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) (ICRA 2019) is one of the earliest self-supervised anchors on this line. The obvious open issue: **where does the training supervision come from?** Vision–language contrastive learning can eat massive web image–text pairs; a four-tuple vision–tactile–force–proprio alignment has no natural supervision source. The two common paths are (a) teleoperation logs force-collect the four streams together and treat "close timestamps" as weak alignment supervision (Calandra et al.'s *More Than a Feeling* [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) is an early instantiation of this on tactile-grasp), or (b) language / vision as a bridge, pulling tactile into a V-L space (TVL is on this route).

**State this explicitly**: the works above show that a shared latent *can* work on particular task families, but their representations are still policy-specific or dataset-specific. This article draws an inference: if these representations are meant to be reused across policies / world models / controllers / sensors, one more layer is needed under the latent — a **structured belief interface** with units, frames, timestamps, uncertainties, provenance, and validity. This layer is *this article's claim*, not the papers'.

**Structured belief interface** (the route recommended here): do not fuse at the raw layer, do not fuse at the latent layer; compose on an explicitly agreed intermediate representation. That representation is not a learned embedding but a set of belief slots whose semantics have converged and that carry uncertainty / provenance / timestamp / validity. The concrete schema is unfolded in §6. Advantages: a missing stream only affects its own key; time, frame, semantics, and validity all converge in the slot definition; once slots are fixed, the fusion structure can be changed freely. Disadvantages: slot design is heavier than "just wrap a Transformer"; if slot fidelity is poor, downstream policy cannot exceed it; and **the slot does not by itself cover the "non-contact semantics" vision provides — object identity, geometry, free-space, occlusion, scene context**. This article recommends a state interface for *contact-rich manipulation*, not a general multimodal interface.

**Section takeaway**: along the fusion paradigm spectrum, early concat / attention / late fusion / shared latent all have their place, but most of their failure modes trace back to §2's three steps not being done properly. This article leans toward shifting the engineering weight from "pick an attention" back to "define the state interface first".

## 4. Why Vision Was the First to Grow a Reusable Data and Representation Ecosystem

A natural question: the three steps in §2 apply to vision as well — why did vision manage to grow a common data interface?

**Correction first**: **this is not single-cause**. It is the three bases plus a whole set of ecosystem conditions — standardized hardware (CMOS sensors + a unified lens mount), unified file formats (JPEG / PNG / MP4 / HDF5), a coordinate model (pinhole + SE(3)), internet-scale data, mature annotation tasks, public benchmarks (ImageNet / COCO / ADE20K / LVIS), mature encoders, GPU scaling laws, available compute. This article emphasizes only the three bases, because that happens to be exactly what tactile lacks most — **it does not claim "three bases fixed = visual success", and it does not claim "vision solved these three first, therefore it could become a foundation model"**.

**On the time base**, video is naturally frame-indexed at 30 or 60 Hz, and every downstream task (classification / detection / segmentation / SLAM) agrees to think "in frames". This convention looks trivial, but it **eliminates the hardest class of problems in the fusion discussion** — all image-based downstream algorithms share one time axis. **On the coordinate base**, pinhole + intrinsics / extrinsics turn image-pixel ↔ world-point into a formula ($s \cdot m = K [R | t] \cdot M$) — 3D vision, SLAM, NeRF, 3D Gaussian Splatting, and multi-view stereo all grow on this convention; calibration errors exist, but "what the error looks like" is predictable and reproducible. **On the semantic base**, RGB itself carries no meaning; the visual community built a set of **highly interoperable task-level representation conventions** — object class / bounding box / instance mask / depth / affordance / caption — via COCO / ImageNet / ADE20K / LVIS. **More precisely, these datasets did *not* produce a unified semantic schema**: ImageNet is a classification ontology, COCO defines detection + instance + caption, ADE20K defines scene parsing, LVIS defines a long-tail instance distribution — each is defined separately, only *composable* with each other.

Comparing tactile: on time, different sensor families range from 30 Hz to kHz with a mix of event-triggered and polled semantics, with no consensus; on coordinates, GelSight is pixel + elastomer deformation, 9DTact is pixel + 3D deformation field, taxel arrays are 1D / 2D force distributions, optical tactile is yet another representation — even "what shape a tactile reading is" has not converged; on semantics, contact point, normal, tangential, slip, and mode all appear in papers, but there is **no cross-dataset unified task-level convention**. Related correction: it is tempting to say "9DTact calls it 6D force, GelSight calls it shear map, arrays call it taxel load — same physical quantity, different projections". **They contain overlapping but hierarchically different physical information**: GelSight's shear / deformation map is closer to raw local deformation observation; 9DTact's 6D force is a global wrench *estimate* obtained through model inversion. The two are not the same quantity projected differently — they are **observation-layer vs estimation-layer** differences. This directly affects §6's slot schema.

**Section takeaway**: the visual ecosystem's success is not that ImageNet / COCO provide a unified multimodal state contract (they do not), but that they gradually formed a set of **highly interoperable task-level representation conventions**; camera models, calibration practices, image formats, timelines, and mature benchmarks further lowered interface friction across systems. If tactile / force / proprio want to grow an ecosystem of comparable scale, the first thing to do is **agree on an interface**, not compete on model architecture.

## 5. Per-Modality Fusion Difficulties

This section sets aside the time / frame / semantic bases already established in §2 and focuses on sensor-specific modeling assumptions, to prepare §6's interface design.

| Modality | Native evidence | Main ambiguity | Canonical slot output |
| --- | --- | --- | --- |
| Vision | Scene geometry, appearance, semantics | Occlusion, view-dependent pose, identity ambiguity | Geometric prior / contact hypothesis |
| Tactile | Local deformation / force distribution at contact interface | Sensor-family heterogeneity, patch vs point, mounting offset | Local contact observation (patch-aware) |
| Force/torque | Aggregate 6D wrench at sensor frame | Contact decomposition is under-determined; requires model-based compensation | Aggregate wrench constraint + residual |
| Proprioception | Robot internal state $(q, \dot{q}, \tau)$, FK-based EE pose | Model mismatch, friction / backlash, no external-world signal | Kinematic / dynamic constraint |

### 5.1 Tactile: Raw Layer Has Not Converged

For essentially the same physical quantity (local deformation or force distribution at the contact interface), there are currently at least four implementations with **completely different representations**:

```
Image-based      GelSight / 9DTact [arXiv:2308.14277] / GelSlim / TacTip
                 → RGB deformation map / multi-view images / optical flow / end-pose

Taxel array      BioTac / 1D-2D capacitive arrays / piezoresistive arrays
                 → array pressure / local normal-force distribution / local stress map

Optical waveguide AnySkin / DigiTact
                 → light propagation in elastomer, edge-triggered contact points

Proprioceptive-inferred  F/T + kinematics
                 → contact inference ("soft tactile")
```

The four encoder families differ structurally: CNNs for images, MLPs for taxel arrays, spline models for waveguides, IK for proprioception-inferred. A common misconception is "make a tactile foundation model to unify all tactile sensors" — that is exactly what AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) is doing, and the direction is right. **But it unifies the learned-representation layer, not the raw layer, and it does not unify the state-interface layer either**. Even with AnyTouch, the output is still an embedding, and one more explicit "embedding → contact slot" convention is needed before entering the state interface. **Mitigation**: introduce a **sensor-specific decoder** between raw and slot, uniformly decoding heterogeneous tactile outputs to §6.1's slot schema — analytic or learned, but it must be part of the interface, not hidden inside the policy.

### 5.2 Force/Torque: Aggregate, Compensation-Heavy, and Cannot Invert Geometry

F/T outputs a 6D wrench in a uniform format, but its **semantics** are much richer:

$$
w_{\text{raw}} \;=\; w_{\text{contact}} + w_{\text{gravity}} + w_{\text{inertial}} + w_{\text{friction}} + b_{\text{bias}}
$$

Extracting the "external contact wrench" requires at least: bias / tare compensation, gravity compensation $w_g = g(q)$ (containing tool + gripper mass distribution — swap the tool and the whole curve changes), inertial compensation, and coordinate transformation (bring the sensor-frame wrench to base or world).

**An important formula tightening on inertial compensation** (v3 conflated joint-space dynamics with sensor-frame wrench in a single formula, and had the mapping direction reversed; the reviewer correctly pointed out that $\tau = J^T F$ maps wrench → joint torque, so running it in reverse needs a pseudoinverse and additional rank / model assumptions):

- **The joint-side estimation chain** first computes a joint residual torque
$$\tau_{\mathrm{res}} \;=\; \tau_{\mathrm{meas}} - \hat{\tau}_{\mathrm{model}}(q,\dot q,\ddot q)$$
  where $\hat{\tau}_{\mathrm{model}}$ typically includes $M(q)\ddot{q} + C(q,\dot q)\dot{q} + g(q) + \tau_{\mathrm{friction}}$. Under reasonable model, full-rank joint space, and null-space torque not interfering, an external wrench hypothesis is
$$\hat{F}_{\mathrm{ext}} \;=\; (J^T)^{\dagger}\, \tau_{\mathrm{res}}$$
  with $(\cdot)^{\dagger}$ a weighted pseudoinverse; the solution is not unique and requires an additional minimum-norm or task-space weighting assumption.
- **The wrist F/T side is a separate chain**: the sensor directly outputs $w_{\text{raw}}$, which is processed through SE(3) adjoint transforms plus sensor-frame bias / gravity / inertial compensation to obtain $w_{\text{ext}}^{\text{sens}}$, and then transformed to base or tool frame. **Do not** fuse the two chains into a single formula — joint-space dynamics terms and the inertial term seen by the wrist sensor are different physical quantities with different mapping directions.

For slow manipulation, the joint-side residual is often enough; for high-speed motion, inertial and actuator dynamics both become non-negligible. **If anything goes wrong in this layer, the downstream "normal / tangential force decomposition" is entirely wrong.**

There is a more fundamental limit to F/T: **it gives an *aggregate* wrench, not a *local* contact**. This judgment is best stated as a boxed principle:

$$
\boxed{\;\text{Force/torque measures an aggregate wrench; it does not, on its own, uniquely decompose into individual contacts.}\;}
$$

$$
w \;=\; \sum_{i=1}^{N} \begin{bmatrix} f_i \\ (p_i - p_0) \times f_i \end{bmatrix}
$$

Infinitely many $\{p_i, f_i\}$ yield the same $w$. **Qualified**: given known contact geometry, robot kinematics, object geometry, and a contact model, F/T *can* indirectly yield strong localization constraints (e.g., "with only one candidate contact point, F/T can pin the force to that point"). The issue is not "F/T cannot localize" but **"F/T alone does not uniquely decompose into individual contacts"**. This directly affects §6.2's slot semantics.

**Mitigation**: at the slot layer F/T should output two things — (a) the compensated external wrench (for a force-aware policy), and (b) a residual magnitude (to detect "did it drift again / did it hit something it shouldn't"). The second is often ignored.

### 5.3 Proprioception: The Most Direct Stream, and the Easiest to Over-Trust

Proprio is the fastest default time constant among the four, and it is **usually the most direct, most stable, and easiest-to-acquire internal state available in a robot system** — with an important qualifier: "complete channels" is not accurate. Many robots have no direct torque sensing, $\dot q$ is often numerically differentiated, EE pose is FK-based, motor current and joint torque are mediated by friction / backlash / transmission ratio, and on soft robots or low-cost platforms proprio may be very partial. A more accurate claim is — **proprio is the easiest stream to standardize and the most natural anchor for frame and time base**.

Proprio plays two roles in fusion: **as an input to contact hypothesis** — given a desired EE trajectory plus joint-torque feedback, §5.2's joint-side chain can invert an external wrench hypothesis (Hogan's impedance-control lineage classic tool); and **as an anchor for the time base and coordinate frame** — the fusion layer needs a stable frame, proprio runs at hardware-servo-allowed rates, and EE pose can serve as the anchor for other frames. **Fusion difficulty**: proprio's information is "too clean" — it is the robot measuring itself, without external-world uncertainty. Models easily **over-rely on proprio** as a shortcut, doing well in no-contact scenes but never learning to use tactile / F/T when contact happens. This is exactly what 9/11 §5.2.1's feedback-value ablation is trying to avoid.

### 5.4 Time-Constant Differences Are Themselves a Modeling Assumption

Combining §5.1–5.3 with §2.1, a judgment crystallizes: **the differences in effective observation rates across modalities are not an engineering question of "should the fusion layer do something" but a modeling assumption of "what control bandwidth the task requires"**. Wiping / polishing need force-control bandwidth of at least 100–500 Hz; vision at 30 Hz is fine; tactile and F/T must run at native rate; proprio must reach the torque layer — for such tasks the fusion layer cannot downsample everything to 30 Hz. Pick-and-place / assembly are vision-dominant with sparse contact events, and downsampling tactile to vision rate is an acceptable approximation. Concrete numbers depend on hardware, task, and controller architecture; we do not fix them here.

## 6. A Minimum Usable Structured Belief Interface

Time to answer "what to actually do". This section proposes a **deployable, benchmarkable, incrementally extensible** minimum interface.

**Scope tightening first**: what this section proposes is *not* "all robot multimodal information should be compressed into a contact set", nor "structured belief interface ≈ contact set". **Contact set is a core instance of this interface in contact-rich manipulation, not the definition of the interface itself** — this sentence determines whether readers doing locomotion / navigation / whole-body manipulation can plug their state into the same contract. The interface itself contains at least:

```text
Structured belief interface
├── robot_state      ── (q, q̇, τ, EE pose) + availability
├── scene_state      ── free-space / object pose / occlusion … (optional)
├── task_context     ── language token / goal embedding
├── wrench_ext       ── 6D + covariance + validity
├── contact_set      ── core instance for contact-rich manipulation, detailed in §6.1
└── belief (optional)── task-relevant latent / explicit
```

For tasks without contact semantics (e.g., pure navigation), the `contact_set` field can be entirely absent and the interface still stands. **Contact set is an instance; the interface is a contract.**

### 6.1 Contact Slot Schema

Define the cross-modality shared **contact event record** (upgraded: with uncertainty / provenance / timestamp / validity):

$$
C_t = \big\{\, \big(\, \mathrm{track\_id}_i,\; G_i,\; \mathcal{F}_i,\; m_i,\; \mathcal{U}_i,\; \mathcal{P}_i,\; \mathcal{I}_i,\; \mathcal{L}_i \,\big) \,\big\}_{i=1}^{N_t}
$$

$G$ geometry, $\mathcal{F}$ wrench, $m$ mode, $\mathcal{U}$ uncertainty (written out), $\mathcal{P}$ provenance (objectified), $\mathcal{I}$ evidence mask, $\mathcal{L}$ lifecycle. Field by field:

```python
contact = {
    # Identity — cross-frame consistency is a hypothesis, not an intrinsic property
    "track_id":            Optional[int],
    "track_confidence":    float,           # ∈ [0, 1]; "track_id=17" and
                                            # "track_id=17 + confidence=0.97"
                                            # mean different things at the interface level

    # Geometry — not always a point contact
    "geometry": {
        "type":            enum,            # point | patch | region
        "position":        Quantity,        # {value: Vector3, unit: "m",
                                            #  frame_id: "base"}
        "normal":          Quantity,        # {value: Vector3, unit: "-",
                                            #  frame_id: "base"}
        "extent":          Optional[Quantity],  # present when type ≠ point
        "center_of_pressure": Optional[Quantity],
    },

    # Wrench (canonical point / patch summary)
    "wrench": {
        "f_perp":          Quantity,        # {value: float, unit: "N", frame_id}
        "f_parallel":      Quantity,        # {value: Vector2, unit: "N", frame_id}
        "moment":          Optional[Quantity],  # present when type ≠ point
    },

    # Modality and events
    "slip_probability":    float,           # ∈ [0, 1]
    "mode":                enum,            # free / touch / sticking /
                                            # sliding / rolling / separating

    # Uncertainty — continuous and categorical must be kept separate
    "uncertainty": {
        "pose_covariance":      Matrix,     # Σ on ℝ³ or SE(3)
        "force_covariance":     Matrix,
        "slip_probability":     float,      # scalar, not a covariance
        "mode_probability":     Vector,     # categorical distribution
        "calibration_quality":  enum,       # nominal / degraded / unknown
        # Note: a covariance alone does not distinguish measurement noise
        # from model uncertainty or calibration uncertainty. If finer
        # attribution is needed, explicitly split aleatoric / epistemic /
        # calibration, or at least track σ_sensor, σ_model, σ_calibration.
    },

    # Provenance (objectified; no longer a single enum)
    "provenance": {
        "primary_sources":      ["tactile", "force_torque"],   # dominant contributors
        "contributing_mask":    [V, T, F, P] → [0/1, 0/1, 0/1, 0/1],
        "estimator":            "contact_estimator_v2",
        "calibration_version":  "...",
    },

    # Time and lifecycle (§6.1.1)
    "timestamp":           float,           # host clock, seconds
    "age":                 float,           # seconds
    "valid_from":          float,
    "valid_until":         float,
    "availability":        enum,            # present / stale / delayed /
                                            # unavailable / corrupt
    "lifecycle":           enum,            # new / tracked / occluded /
                                            # lost / merged / split
}
```

Every numeric field in the schema has type `Quantity = {value, unit, frame_id, ...}`, with unit in SI canonical form (meter, newton, newton-meter, second, radian) and frame_id referencing a registered frame namespace. This is not just "semantically we should have units" — it is "the schema can be machine-validated for units and frames", and §6.8's runtime validator asserts every slot's unit / frame pair.

The schema deliberately aims for five properties: **sensor-agnostic** (GelSight, taxel arrays, F/T — anything that can fill these keys can enter the interface); **physically interpretable with explicit units** (every number has an SI dimension and a declared frame); **uncertainty-aware** (continuous covariance and categorical probability are kept separate, with a further aleatoric / epistemic / calibration split preserved as an extension point; **a slot without uncertainty is not an interface — it is a "fact", and facts never hold on real hardware**); **provenance-aware** (`primary_sources` + `contributing_mask` + `estimator` + `calibration_version`, directly connected to §7.5 modality dropout, §8.6 contradiction, and §8.7 consistency graph); and **validity / availability / lifecycle as first-class** (unfolded in the next three sub-sections).

#### 6.1.1 Why Validity / Availability / Lifecycle Are First-Class

The same slot value can correspond to three completely different states:

```text
value similar, age large   ── historically valid + temporally stale
value fresh, covariance large ── fresh + uncertain
value missing              ── unavailable
```

The controller should treat these three differently: stale → hold + covariance inflation + fallback; fresh-uncertain → keep consuming with lower weight; unavailable → §6.6's explicit degraded path. v3's wording can be a bit tighter here — strictly, "a stale observation may remain valid with respect to a *past* state, but not necessarily with respect to the current one", so this article standardizes on **"staleness is not invalidity; it is validity relative to a past timestamp"**. If the interface can only express "value present / value absent", the downstream has to guess; **this layer is a design requirement of the interface itself, not a benchmark trick**.

#### 6.1.2 track_id Is a Hypothesis, Not an Intrinsic Property

v2's "tracking ID, cross-frame consistent" was too optimistic. A contact set is essentially **permutation-invariant** and has no natural stable identity: multi-finger manipulation, rolling contact, patch splitting / merging, occlusion, and transient contact all make identity assignment itself an inference problem. Since v3 the field has been rewritten as `track_id: Optional[int]` + `track_confidence: float`, with an explicit:

> **track_id is a hypothesis maintained by temporal association, not an intrinsic physical property of a contact.**

Slots remain valid without a track_id; with a track_id, it is a downstream-consumable hypothesis with confidence attached. `track_confidence` can also be partially covered by lifecycle (e.g., `lifecycle = "split"` should cause both children's track_confidence to be reduced).

#### 6.1.3 Point Contact Is a Modeling Assumption, Not a Physical Fact

The classic $p$ / $n$ / $f_\perp$ / $f_\parallel$ set is actually a **canonical summary of a point contact, or of a local contact patch collapsed to a representative point**. Many tactile tasks are not point contacts — finger pad ↔ object is really a contact patch $\mathcal{A}_i$ with pressure, shear, and normal-force distributions, a center of pressure, and a torsional moment. GelSight's raw observation is more naturally a contact-patch geometry than a single $p + n$. The interface layer carries `geometry.type ∈ {point, patch, region}` + `extent` + `center_of_pressure` + `moment` explicitly. At minimum, $p, n, f_\perp, f_\parallel$ should be labeled as a **"canonical point / patch summary"**, not as the complete physical description of contact.

### 6.2 How Each Stream Maps to Slots · F/T Is a Constraint, Not a Detector

Cross-modal signal → slot semantics:

```text
Tactile         → local contact observations   (direct observation, with sensor noise)
F/T             → global wrench constraint     (aggregate, no localization, no decomposition)
Proprioception  → kinematic / dynamic constraint (via (J^T)†, model-dependent)
Vision          → geometric prior              (predicted contact hypothesis, not observation)
```

A natural formulation is Bayesian:

$$
p(C_t \mid V, T, F, P) \;\propto\; p(V \mid C_t)\, p(T \mid C_t)\, p(F \mid C_t)\, p(P \mid C_t)\, p(C_t)
$$

$p(T|C_t)$ is the local-observation likelihood, $p(F|C_t)$ the wrench-consistency likelihood, $p(P|C_t)$ the dynamics-consistency likelihood (via the $\tau_{\mathrm{res}}$ residual), $p(V|C_t)$ the geometric / visual-prediction likelihood, $p(C_t)$ the prior.

**Two caveats that must be written down** —

**(i) Conditional independence is a simplification.** The factorization above implicitly assumes $p(V,T,F,P|C) = p(V|C)p(T|C)p(F|C)p(P|C)$. Real robots have extensive shared variables across vision / proprio / F/T / tactile: robot pose, object pose, contact geometry, dynamics, calibration, actuator state (for example F/T and proprio both depend on $(q,\dot q,\tau)$). **We adopt conditional independence here only to keep the evidence-fusion structure legible; practical systems must model cross-modal correlations explicitly** — a joint Gaussian likelihood with shared covariance, or a graphical model / message passing over a coupled factor graph.

**(ii) Bayesian inference is an implementation choice, not an interface requirement.** **The interface does not prescribe a Bayesian inference algorithm. "Belief" here denotes uncertainty-aware state information; Bayesian posterior inference is one implementation, not a requirement of the schema.** The same §6.1 slot schema can be produced by an EKF / UKF, a factor graph, a particle filter, a learned filter, a diffusion state estimator, a Transformer state estimator, or a hybrid neural-symbolic estimator — as long as the output satisfies §6.4's 7-tuple (Value / Semantics / Frame / Time / Uncertainty / Provenance / Validity). This sentence turns the proposal from "a probabilistic architecture" into a genuine **interface proposal**.

### 6.3 Evidence Ranking Is Hypothesis-Dependent, Not Fixed

A fixed ordering like "tactile > F/T > proprio > vision" is wrong. The correct statement is **evidence ranking should be hypothesis-dependent**:

```text
Contact location:    Tactile > Vision > F/T ≈ Proprio
Global wrench:       F/T > Tactile > Proprio > Vision
Object pose:         Vision > Tactile > F/T ≈ Proprio
Joint state / τ_res: Proprio >> others
Contact mode:        Tactile > F/T ≈ Proprio > Vision
Slip probability:    Tactile > F/T (rate) > Proprio (residual) > Vision
```

**Within the same $C_t$ record, different fields can have completely different "dominant modalities"** — that is also why §6.1's provenance tracks both `primary_sources` and `contributing_mask`.

### 6.4 Interface ≠ Learned Latent Representation

This is the most architecturally central subsection.

```text
Latent representation             Structured belief interface
─────────────────────────         ─────────────────────────────
Optimization target: friendly to downstream model   Optimization target: interoperable across consumers
Units: implicit                   Units: explicit canonical unit in Quantity
Frame: implicit in data flow      Frame: explicit declaration + frame_id per field
Time: implicit                    Time: explicit timestamp + age
Validity: none                    Validity: availability / lifecycle / valid_until
Uncertainty: implicit or loss-only Uncertainty: continuous covariance + categorical distribution + calibration metadata
Provenance: none                  Provenance: primary_sources + contributing_mask + estimator
Consumers: one model              Consumers: policy / world model / controller /
                                     diagnostic tool / safety layer / another sensor
```

One formula worth keeping:

$$
\text{Interface} \;=\; \big(\, \text{Value},\;\text{Semantics},\;\text{Frame},\;\text{Time},\;\text{Uncertainty},\;\text{Provenance},\;\text{Validity} \,\big)
$$

**A latent can be extremely good for one policy and unsuitable as an input interface for a world model, controller, diagnostic tool, or another sensor** — that is the necessity of the interface, and the missing piece in §3.4's shared-latent works. **The visual ecosystem enables cross-system reuse not because ImageNet / COCO provided a complete multimodal state contract (they did not define SI units, frames, timestamps, covariances, or validity lifecycle), but because they formed a set of highly interoperable task-level representation conventions, and camera models / calibration practice / file formats / timeline conventions further drove cross-system interface friction down very low**. What tactile, force, and proprio need now is one step beyond that layer of conventions — an explicit state contract **across sensor families, across embodiments, and across downstream consumers**.

### 6.5 Where to Fuse: Not at Raw, Not at Decision — **On the Structured Belief Interface**

With §6.1's schema, fusion becomes three steps:

```python
# Step 1: per-modality perception + registration
raw_v  → enc_v → pred_contact_hypothesis      (visual contact prediction, with confidence)
raw_t  → enc_t → detected_contact             (tactile observation, with pose_covariance)
raw_ft → enc_ft → external_wrench + residual  (compensated wrench, with force_covariance)
raw_p  → proprio_state → τ_res → (J^T)† τ_res → inferred_contact  (with model-error term)

# Step 2: hybrid state estimation → write to belief interface
# (Bayesian / EKF / factor graph / learned filter all valid implementations)
C_t         = state_estimator(V, T, F, P)   # belief, not hard fact
wrench_ext  = compensate_ft(raw_ft)
robot_state = (q, q̇, τ, EE_pose) + availability
belief_t    = belief_update(C_t, wrench_ext, robot_state, task_ctx)

# Step 3: policy / world model / controller / diagnostics consume belief interface
a_t = policy(belief_interface)
```

The key judgment — **Step 2's interface schema is the interface of the whole multimodal system**. In Step 1 each modality can swap encoders freely; in Step 3 the policy can swap architectures freely; what has to be stable is Step 2. And — **the structured belief interface is the system's stable interoperability boundary, but it need not become the only information path for every downstream model**: a policy can absolutely take structured slots plus additional raw visual features / language tokens / task embeddings. What must be preserved is that **the state shared across consumers** flows through this contract, not that "all information must pass through here".

### 6.6 Missing-Modality Fallback: Treat "One Stream Gone" as Part of the Training Distribution

In real deployment, sensor dropouts, frame loss, and timeouts are the norm. The interface layer must explicitly support "this stream has no data for now":

```python
if tactile_missing:
    C_t = state_estimator(V, F, P)         # no T; p(T|C_t) → uniform likelihood
    for r in C_t:
        r.uncertainty.pose_covariance  *= inflation_T_missing
        r.uncertainty.mode_probability  = uniform_over_modes
        r.provenance.contributing_mask.T = False
        r.provenance.primary_sources.remove("tactile")
        r.availability     = "unavailable"
```

**This is not an engineering patch; it is a core training constraint**. See §7.5 modality dropout, §8.5 degradation modes, §8.6 cross-modal contradiction.

### 6.7 wrench_ext and contacts[]: Aggregate Observation + Consistency Constraint, Not Redundancy

A reviewer will immediately ask: if `contacts[]` already contains each contact's force and moment, why keep a global wrench? Answer —

$$
w_{\text{ext}} \;\neq\; \sum_i w_i^{\text{reconstructed}}
$$

`wrench_ext` is an **independent aggregate measurement** (what the F/T sensor directly reports), while `contacts[]` is a **structured hypothesis** (the joint product of each perception stream and the state estimator). They are two different observations that constrain each other:

$$
\mathcal{C}_{\text{wrench}} \;=\; \big\| w_{\text{ext}} - \textstyle\sum_i \big[f_i;\, (p_i - p_0) \times f_i\big] \big\|_{\Sigma^{-1}}
$$

The larger this residual, the more contact hypotheses disagree with the F/T observation, and the system should raise a consistency alarm or pull `mode_probability` toward uniform. **`wrench_ext` is not a redundant field on `contacts[]` — it is an independent aggregate observation that acts as a consistency constraint on contact-set reconstruction**, which is also the physical foundation of the F↔T, F↔V, and F↔P edges in the §8.7 consistency graph.

### 6.8 Schema Evolution / Interface Compatibility (Where the Unit / Frame Validator Lives)

If an interface really exists, it must answer a very practical engineering question: **can an old policy consume a new sensor version?**

```text
v1  contact: p, n, force
v2  contact: p, n, force, slip_probability, covariance
v3  contact: geometry{type, pose, extent}, wrench{f, m}, mode_distribution
```

If the interface is an API, it must handle API versioning. Minimum set:

```text
Interface compatibility
├── version tag            # every slot carries a schema version
├── optional fields        # old consumers ignore new fields;
│                          # new consumers fill defaults for missing ones
├── backward compatibility # v2 consumers can read v1 slots;
│                          # v1 consumers can safely degrade on v2 slots
├── unit / frame validator # schema metadata defines each field's canonical unit
│                          # and frame namespace; runtime assert. Quantity
│                          # {value, unit, frame_id} is this validator's landing point
└── graceful degradation   # inflation / default / alarm strategy on missing fields
```

Without this layer, "state interface = API" is only an analogy. This layer is also a precondition of §8.1's interface-swap benchmark.

## 7. Fusion Failure Modes and Diagnostics

From a reviewer's perspective, the five failure modes below are the most common and the easiest to gloss over with "we used cross-attention so we're robust". Each section follows a uniform four-part pattern: **Failure mode → Observable symptom → Diagnostic test → Mitigation**.

### 7.1 Modality Collapse: Policy Silently Falls Back to Vision / Proprio

- **Failure mode**: some modalities become de facto dead inputs during training.
- **Observable symptom**: training loss and validation success rate both look normal; mask out vision at eval and policy performance barely changes.
- **Diagnostic test**: primary metric is **modality ablation gain** (§8.4) — if $\Delta_{\text{mod}} \approx 0$, that stream is not being consumed by the policy. 3D-ViTac [arXiv:2410.24091](https://arxiv.org/abs/2410.24091)'s ablation matters because it provides an existence proof for this gain (the visuo-tactile > vision-only delta is **that paper's experimental result, not a universal law**). Attention-weight visualization is only **auxiliary diagnostic** — **attention ≠ causal importance**; low weight on a modality does not mean it is unused, high weight does not mean removing it will hurt. This is a classic interpretability trap.
- **Mitigation**: (a) explicit modality dropout in training, see §7.5; (b) per-modality auxiliary supervision (e.g., tactile encoder independently predicts slip events, not just feeds the policy); (c) use 9/11 §5.2.1's feedback-value benchmark instead of pure success rate to force real contributions out.

### 7.2 Temporal Smearing: Forcing Everything to 30 Hz

- **Failure mode**: high-frequency physical events are erased by downsampling.
- **Observable symptom**: policies for slip detection, transient contact, or impact tasks simply do not learn.
- **Diagnostic test**: compare "30 Hz fusion" against "native-rate fusion" success rate; or use $\Delta_{\text{tail}}$ to look only at hard-contact conditions.
- **Mitigation**: see §5.4 — treat time-constant differences as **modeling assumptions**, run different layers at different rates, and put events on an event bus; do not downsample high-rate signals at the interface layer.

### 7.3 Frame Confusion: Model Learns a Coordinate-Frame Pseudo-Correlation

- **Failure mode**: the model learns a correlation under a specific sensor frame rather than the task.
- **Observable symptom**: excellent performance at training pose, training camera placement, training tool; a small change and it collapses.
- **Diagnostic test**: deliberately apply **coordinate-frame perturbations** on the eval set (four sub-classes in §8.8).
- **Mitigation**: put `frame_id` explicitly on state interface fields; downstream consumers must transform to read them; include frame perturbation as part of training-time domain randomization (Tobin et al. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)).

### 7.4 Semantic Leakage: Raw Vision Path Bypasses the Slot Definition

- **Failure mode**: the definition of contact state is quietly rewritten by an unconstrained visual latent.
- **Observable symptom**: slot is defined as "contact geometry", yet policy performance depends strongly on appearance — swap the background of the same object's photo and it drops.
- **Diagnostic test**: replace the visual contribution in the slot with a "minimal sufficient" synthetic slot (e.g., predicted contact → ground-truth contact), observe performance change.
- **Mitigation**: v3's wording was too strong and needs tightening — this article **does not** claim "raw pixels should not reach the policy" (modern VLAs routinely go image → vision encoder → token representation → policy; feeding visual latents to a policy is normal practice). What this article does claim is: **for the contact-related state defined by the state interface, there should not exist a raw-visual shortcut that bypasses the slot encoder**. That is, "the sole path to contact semantics is through the slot encoder" — not "visual features are not allowed to reach the policy". This is consistent with, and more implementable than, 9/11 §3.5.

### 7.5 Missing / Degraded Modalities: Collapse on Failure, or Silent Degradation Without Alarm

- **Failure mode**: the training distribution always assumes every modality is present; real degradations are out of distribution.
- **Observable symptom**: one tactile frame drop causes dramatic policy behavior changes, or degradation slips by silently with no alarm.
- **Diagnostic test**: run **modality masking / dropout** (randomly dropping a stream with probability $p$). Modality masking / dropout is **one common training strategy** for missing-modality robustness; see Maiga et al.'s MMP [arXiv:2410.03010](https://arxiv.org/abs/2410.03010). **But dropout ≠ sensor failure.** Real degradations usually are not independent Bernoulli(p) drops; they are the six classes missing / stale / delayed / corrupted / biased / noisy.
- **Mitigation**: the interface must express all six, not just a present / absent flag. §6.1's `timestamp / age / availability / lifecycle / uncertainty` exist exactly so that stale / delayed / biased are detectable in the data format — **this is a design requirement of the interface, not a benchmark trick**.

## 8. Benchmarks and Evaluation

This section consolidates the diagnostics scattered in §7 into an executable benchmark skeleton. **Interface swap is placed at §8.1**, because dropout-style metrics prove *modality robustness*, whereas swap proves *interface interoperability*, which is closer to this article's core claim. The overall stance follows 9/11 §5.2.1's feedback-value argument: **a benchmark should measure whether the interface really acts as an abstraction boundary, whether each modality is really irreplaceable, and whether the system handles disagreement correctly — not whether a model can memorize the training distribution**.

### 8.1 Interface Swap (Interoperability Benchmark, the First Primary Experiment Here)

If the interface is an abstraction boundary, swapping encoders or consumers should not require redesigning the middle layer. Three concrete experiments:

1. **Sensor encoder swap**: tactile encoder A → encoder B (different backbone, even different sensor family), slot schema unchanged, policy weights unchanged, observe performance transfer.
2. **Policy swap**: MLP → Transformer, slots unchanged, observe downstream training cost.
3. **Consumer swap**: policy ↔ world model ↔ controller ↔ diagnostic tool — four consumers sharing the same slot, trained independently, observe whether the interface is truly reusable.

If all three achieve "swap encoder / swap consumer without redesigning the middle layer", the evidence is much stronger than another success-rate number — **it turns "interface" from a metaphor into a measurable property**. §8.1 and §6.8 are pair-composed: schema guarantees forward compatibility, swap guarantees horizontal pluggability.

### 8.2 Oracle-Slot / Estimated-Slot / End-to-End: Three Baselines

To test whether "state interface is necessary", these three baselines are required:

```text
A · Oracle interface   : ground-truth contact slots → policy
B · Estimated interface: sensor → state estimator → estimated slots → policy
C · End-to-end fusion  : sensor → fusion policy (no explicit interface)
```

$S_A$ = interface upper bound; $S_B$ = interface + estimation; $S_C$ = end-to-end. At minimum, the experiment distinguishes: is the loss in **perception / interface** ($S_A$ high, $S_B$ drops), in **policy** ($S_A$ and $S_B$ both drop), or does **the interface abstraction simply not help** ($S_C \geq S_B$). The factorization $S_{\text{total}} \approx S_{\text{interface}} \times S_{\text{downstream}}$ is the most direct experimental test of this article's central claim. Without this baseline, other benchmarks only describe "is the interface nice to use", not "is the interface necessary".

### 8.3 Slot Fidelity Measured Independently

**Slot accuracy itself should be measured independent of the policy**: contact-position IoU / distance to ground truth; slip detection AUROC; contact-mode macro-F1; **uncertainty calibration** — reliability diagram or negative log-likelihood for continuous parts, Brier score or ECE for categorical parts; extent / center-of-pressure error for patches; track_id identity preservation (ID-switch count, MOTA / IDF1). These are "quality-before-fusion" metrics — if they are bad, high policy success rate can only be overfitting.

### 8.4 Modality Information Gain and Graceful Degradation

**Information gain** (each modality's upper-bound contribution):

$$
\Delta_m \;=\; S(M) - S(M \setminus m)
$$

**Graceful degradation** (shape under partial loss):

$$
G_m(p) \;=\; \frac{S(M, p) - S(M \setminus m)}{\Delta_m}
$$

**Symbol definitions first**: $S(M, p)$ denotes performance when modality $m$ is independently degraded at rate $p$, and by convention $S(M, 0) = S(M)$; hence $G_m(0) = 1$, $G_m(1) = 0$, and intermediate values describe the shape of the degradation curve.

A "good system" can have $\Delta_m$ large (this modality is irreplaceable) yet $G_m(p)$ smoothly declining from 1 to 0 (partial loss does not immediately collapse) — a more realistic characterization of sensor importance than "flatter is healthier". **Also make explicit**: do not imply that a good system's $G_m(p)$ curve must be monotonic or smooth. Threshold effects are common — e.g., once tactile resolution drops below some cutoff, slip simply becomes undetectable. A more accurate statement is — **graceful degradation should be characterized rather than assumed monotonic or smooth**; report the shape, do not presume it.

### 8.5 Degradation Modes Beyond Dropout

Matching §7.5, expand from Bernoulli dropout to missing / stale / delayed / corrupted / biased / noisy. Report success-rate curves separately for each. **This is the direct test of whether the interface really handles uncertainty / provenance / validity**.

### 8.6 Cross-Modal Contradiction: Define $D_{ij}(z)$

**Good fusion $\neq$ agreement**. This is the most valuable benchmark dimension in the article, and the one most likely to become its own brand. The reason the interface exists is precisely **that when modalities disagree, the system can be "correctly uncertain" rather than "average into a wrong answer"** — so the benchmark should actively induce cross-modal disagreement.

**Definition**: a contradiction is **two modalities disagreeing on the same verifiable latent variable $z$**:

$$
D_{ij}(z) \;=\; d\!\big(p_i(z),\, p_j(z)\big)
$$

For example, $\mathcal{C}_{VT}$ measures vision vs tactile disagreement on contact position, $\mathcal{C}_{TF}$ measures tactile vs F/T disagreement on normal force, $\mathcal{C}_{VP}$ measures proprio vs vision disagreement on EE pose. Four common instances:

```text
Spatial disagreement    vision says contact at A, tactile says B (5/10/20 mm tiers)
Temporal disagreement   F/T says contact starts at t, tactile says t + Δ
Force disagreement      tactile reports 2 N normal, F/T-consistent reconstruction needs 8 N
Kinematic disagreement  proprio reports EE pose X, vision registration reports Y
```

(v2's "vision says rigid, tactile says compliant" is not a contradiction — one object can be globally rigid and locally compliant. Removed.)

Then measure four things: (1) **contradiction detection** — did the model raise a cross-modal inconsistency; (2) **uncertainty calibration** — is the reported uncertainty proportional to actual disagreement; (3) **source attribution** — can it localize which stream misbehaved; (4) **recovery** — how much success rate remains after handling.

### 8.7 Consistency Graph: From "Contradiction Detected" to "Which Edge Is Suspicious"

$\mathcal{L}_{consistency}$ is not a scalar — it is a graph:

```text
           Vision
          /      \
      Tactile --- F/T
          \      /
         Proprio
```

Each $(i, j)$ edge is a consistency constraint $\mathcal{C}_{ij} = D_{ij}(z_{ij})$: $\mathcal{C}_{VT}$ on contact position, $\mathcal{C}_{TF}$ on normal force, $\mathcal{C}_{FP}$ on the $J^T F$ residual, $\mathcal{C}_{VP}$ on EE pose. Total loss:

$$
\mathcal{L}_{\text{consistency}} \;=\; \sum_{(i,j)} w_{ij}\, \mathcal{C}_{ij}
$$

Taking one step further, define each edge's residual $r_{ij} = \mathcal{C}_{ij}$ and rank edges: **edge residual ranking → candidate faulty modality**, or the more principled $P(\text{fault} = i \mid \{r_{ij}\}_j)$. This lets the system not just say "there is a contradiction" but also "**which observation-graph edge is most suspicious**", naturally connecting to provenance + diagnostics + sensor fault isolation. This upgrade makes the **Provenance and Validity** terms of §6.4's 7-tuple formally benchmarkable.

### 8.8 Registration Perturbation (Relative, Four Classes)

Temporal perturbation is expressed relative to the task's own time scale: $\delta t \in \{0.25\,\Delta t,\; 0.5\,\Delta t,\; \Delta t,\; 2\,\Delta t\}$, where $\Delta t$ is **this task's time-relevant bandwidth** (the characteristic time of contact-mode transition, or the policy control period). No fixed millisecond numbers — otherwise the benchmark cannot transfer.

Spatial perturbation splits into four classes:

```text
Registration robustness
├── calibration noise        ── T̂ = T · ΔT, ΔT small perturbation (Gaussian / uniform)
├── calibration drift        ── slow long-run drift (temperature / mechanical creep)
├── sensor remounting        ── tool swap / camera swap / sensor-mount swap, geometry redefined
└── frame convention mismatch── base ↔ world, sensor ↔ tool, extrinsic sign flip
```

The first three are "wrong parameters"; the fourth is "wrong convention" — downstream consumers fail differently. This is more convincing than plain domain randomization.

### 8.9 Fit with Existing Benchmarks

Most mainstream manipulation benchmarks — RoboCasa / LIBERO / ManiSkill3 / BEHAVIOR — are **vision-dominant, tactile-partial**. This section does not conclude; **but the suggestion** is: turn the eight metrics above into an optional plugin that adds a "feedback-value + interface-quality" view onto existing benchmarks; 9/11 §5.2.1's A/B/C/D four-arm ablation can be reused directly.

## 9. Relation to VLA and World Models

### 9.1 VLA: What Is Missing Is Not Another Concat Channel, but a Stable Cross-Sensor-Family State Contract

§3.1 already corrected the mischaracterization — RT-2 / OpenVLA / π0 are not simply "concat". **Looking at these representative public systems, we observe**: the scaled pretraining ecosystem of existing general VLAs is organized mostly around vision-language observation and robot state / action; **tactile and F/T have not yet formed a public, cross-task, cross-embodiment scaled pretraining ecosystem comparable to vision-language**. This is an ecosystem-level judgment; the evidence chain is "concrete systems → observation → this article's synthesis", not "several papers prove the whole VLA community is like this".

**To be clear** — this absence does **not** mean "nobody has worked on it": tactile foundation models are appearing (TVL, AnyTouch), visuo-tactile representation learning is extensive (Lee, Calandra, 3D-ViTac), some robot foundation model works do include proprio in the input, and hybrid state representations are not a blank. What has not yet formed is a **stable structured state contract across sensor families, across embodiments, and across downstream consumers**. This is a harder-to-rebut, more accurate claim.

The real open question: **does the next wave of VLA scaling extend more vision+language, or does it fill in tactile / F/T state interfaces?** This article bets on the latter — not adding tactile as "yet another channel", but placing §6's structured belief interface at the input layer of a VLA. Qi et al.'s T-Dex [arXiv:2309.09979](https://arxiv.org/abs/2309.09979) (CoRL 2023) and Lee et al. [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) can be viewed as early shapes of this route.

### 9.2 World Models: When Contact / Event State Materially Affects Task Transitions and Control, World Models Typically Benefit

RSSM / DreamerV3 [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) latent dynamics typically assume relatively smooth, differentiable transitions. 9/11 §5.1 discussed that contact events are inherently hybrid-dynamics mode switches.

**A more careful, "recommendation-not-assertion" version**: modern world models can absolutely use discrete latents / categorical latents / hybrid state / event-conditioned dynamics / multiple latent heads / mode-conditioned transitions. This article is **not** saying "world models must directly predict `contact[]`". A world model may predict separately:

$$
z_{t+1}, \quad p(m_{t+1} \mid z_t, a_t), \quad p(C_{t+1} \mid z_t, a_t)
$$

or use a hybrid latent (`z_continuous + z_discrete + contact state`). Our recommendation is — **for contact-rich tasks, when contact / event state materially changes task transitions and control behavior, we argue that world models typically benefit from retaining some explicitly identifiable contact / event representation, rather than forcing it to exist entirely implicitly inside an uninterpretable continuous latent**. This is an architectural recommendation, not a factual conclusion.

This dovetails with §6's structured belief interface: **a world model need not consume every slot field, but usually benefits from having at least one head explicitly predict the mode / event portion of the slot** (as a discrete latent or $p(m_{t+1})$); the policy layer can use a learned representation (embedding slots for policy consumption is legitimate — but this embedding is a "downstream consumer", not an "upstream data format"); the two connect through §6's interface, with the world model predicting the next-time slot and the policy consuming the current slot.

### 9.3 One-Sentence Summary

**What VLA lacks is not the tactile channel per se, but a stable state contract across sensor families / embodiments / consumers; what world models lack is not a continuous latent, but an explicitly identifiable representation of contact / event in contact-rich tasks. The common root cause is that heterogeneous observations and downstream models do not have a middle layer capable of carrying belief / uncertainty / provenance / validity — a structured interface.**

## 10. Conclusion

Starting from the misconception that "multimodal fusion is a model-architecture question", this piece drags the discussion back to the foundations: what is missing is not a stronger fusion operator, but a **structured belief interface** sitting between heterogeneous observations and downstream models that explicitly expresses value / semantics / frame / time / uncertainty / provenance / validity. Vision managed to grow a common data interface not because its encoders are cleverer, but because it first formed **highly interoperable task-level representation conventions**, and stacked on top of them a whole ecosystem of camera models / calibration practices / file formats / timeline / GPUs / cheap sensors / internet-scale data / annotation infrastructure. **What vision formed is task-level conventions, not a single schema, and not a complete multimodal state contract.** Tactile / force / proprio have not yet converged on effective observation rate, coordinate semantics, raw representation, or task-level conventions, so cross-attention / shared-latent efforts easily fall into the five failure modes of §7.

This article does not oppose cross-attention, does not oppose shared latents, and **does not oppose end-to-end learning**; what it opposes is — **letting state estimation, cross-modal composition, and control all happen implicitly inside a single, undiagnosable, non-reusable latent interface**. The proposed minimum path: a structured belief interface with contact set as a core instance (**contact set is an instance, not the definition of the interface**), raw → slot mapping placed inside sensor-specific perception, slots carrying **point / patch / region geometry**, **continuous covariance and categorical probability kept separate**, **uncertainty retaining an aleatoric / epistemic / calibration split as an extension point**, **track_id as a confidence-tagged hypothesis, not intrinsic identity**, **units and frames entering the schema through the `Quantity` type and validated at runtime**, **staleness defined as "validity relative to a past timestamp", not as invalidity**; `wrench_ext` in the interface as an aggregate observation and consistency constraint; the interface itself versioned with graceful degradation; policy / world model / controller / diagnostics interoperate via the interface without the interface becoming the only information path; **Bayesian inference being an implementation option, not a required property of the schema**; and interface swap, oracle / estimated / end-to-end baselines, cross-modal contradiction, consistency graph, modality dropout, and degradation modes all written into the benchmark as first-class.

**This route is not a final answer; it is a direct response to 9/11's diagnosis that "tactile lacks a reusable intermediate representation" — agree on the interface first, then discuss architectures.** Back to this article's three design principles: **Register before compose · Expose belief at the interface · Design for disagreement**.

End with three boxed formulas as anchors of the entire piece:

$$
\boxed{\;\text{Sensor-specific observations} \;\rightarrow\; \underbrace{\text{Structured Belief}}_{\text{stable interface}} \;\rightarrow\; \{\text{Policy, World Model, Controller, Diagnostics}\}\;}
$$

$$
\boxed{\;\text{Good fusion} \;\neq\; \text{agreement}\;}
$$

$$
\boxed{\;\text{Good fusion} \;=\; \text{evidence} + \text{uncertainty} + \text{provenance} + \text{disagreement handling}\;}
$$

The next article (9/13) will walk one step downstream from "interface" into **dexterous hands and in-hand manipulation**, testing whether this article's state interface can carry the architectures of T-Dex / DextrAH / LEAP, and why the ratio "many robots can mount a hand · very few actually do dexterous work" looks the way it does.

## Sources

Citations here are grouped by **the four main thesis-lines they support**, not by "piling up sensor papers".

### A · Cross-Modal Representation and Fusion Paradigms (supports §3 / §4)

- Baltrusaitis, Ahuja, Morency, *Multimodal Machine Learning: A Survey and Taxonomy*, TPAMI 2019 · [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) (classic taxonomy; the reference frame for §3's layer-by-layer analysis)
- Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language Sequences*, ACL 2019 · [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) (early representative that explicitly handles cross-modal temporal misalignment · §3.2)

### B · Visuo-Tactile Fusion Evidence Line (supports §4 / §7.1 / §9.1)

- Calandra et al., *More Than a Feeling: Learning to Grasp and Regrasp using Vision and Touch*, RA-L 2018 · [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) (early visuo-tactile regrasp evidence; one of the first demonstrations that tactile is disproportionately valuable in long-tail scenes)
- Lee et al., *Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks*, ICRA 2019 · [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) (visuotactile self-supervised representation; an early anchor of §3.4's shared-latent line)
- Huang et al., *3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing*, CoRL 2024 · [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) (this paper reports a substantial visuo-tactile vs vision-only gain in its own experiments · reference point for §7.1 modality-collapse diagnostics)
- Qi et al., *General In-Hand Object Rotation with Vision and Touch* (T-Dex), CoRL 2023 · [arXiv:2309.09979](https://arxiv.org/abs/2309.09979) (active tactile exploration + visuo-tactile fusion · a concrete form of §9.1's "add tactile to VLA")

### C · Cross-Sensor / Cross-Modal Unified Representation (supports §3.4 / §5.1)

- Zhao et al., *A Touch, Vision, and Language Dataset for Multimodal Alignment* (TVL / Binding Touch to Everything), ICML 2024 · [arXiv:2402.13232](https://arxiv.org/abs/2402.13232) (~44K vision-touch pairs, tactile-VL alignment, representative of the "tactile-CLIP" route)
- Feng et al., *AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors*, 2025 · [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) (unified representation across heterogeneous visuo-tactile sensors · contrast for §5.1's raw-layer non-convergence)
- Lin et al., *9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation*, ICRA 2023 · [arXiv:2308.14277](https://arxiv.org/abs/2308.14277) (a concrete form of multi-modal tactile; 3D shape reconstruction + 6D force is at the **estimation layer** · §5.1 heterogeneity evidence)

### D · Robustness and Modality Masking / Dropout (supports §7.5 / §8.4 / §8.5)

- Maiga et al., *MMP: Towards Robust Multi-Modal Learning with Masked Modality Prior Fine-Tuning*, 2024 · [arXiv:2410.03010](https://arxiv.org/abs/2410.03010) (modality masking as a training strategy; one common approach within missing-modality robustness · §7.5 mitigation reference)
- *Robust Multimodal Learning with Missing Modalities via Parameter Projection*, 2023 · [arXiv:2310.03986](https://arxiv.org/abs/2310.03986) (representation alignment under missing modalities · §8.4 robustness reference)

### E · VLA and World Models (supports §9)

- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) (VLM backbone + proprio token + noisy action chunk + flow matching · §3 and §9.1 corrected description)
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) (open-source VLA baseline; public config includes multi-camera / depth / proprioceptive state encoding, not simple V+L concat)
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) (action expressed as text tokens, jointly fine-tuned with a VLM · §3 corrected description)
- Hafner et al., *Mastering Diverse Control Tasks through World Models* (DreamerV3), Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) (latent-dynamics world-model representative · the contrast surface for §9.2 "if only continuous latents with no explicit mode / event variable, contact events are more easily smoothed")

### F · Sim-to-Real / Domain Randomization Background (supports §7.3 / §8.8)

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) (classical practice of treating frame perturbation as part of domain randomization)

### G · Follows 9/11 · Contact State and Impedance (background)

- Already cited in 9/11 and reused here: Hogan's impedance-control trilogy, Posa-Cantu-Tedrake IJRR 2014 (hybrid contact-mode trajectory optimization), Lee 1810.10191, Qi 2309.09979, Huang 2410.24091, Zhao 2402.13232, Feng 2502.12191. This article does not re-paste those links; for precise sources see 9/11's Sources.

---

> **Related reading**
>
> - [The Hand Robots Don't Have: Tactile and Force Sensing in Embodied AI](/en/articles/2026-09-11-tactile-force-sensing/) — the predecessor to this article, tactile and force control taken separately
> - [Sim-to-Real Methodology](/en/articles/2026-09-10-sim-to-real-methodology/) — §7.3 frame perturbation and §8.8 registration perturbation can borrow its domain-randomization lens
> - [Why Robot Data Is Harder Than LLM Data](/en/articles/2026-09-09-robot-data-scaling/) — §3 shared-latent's "where does the supervision come from" is really a data-scaling problem
> - [VLA and World Models](/en/articles/2026-09-07-vla-world-models/) — §9 is one concrete facet: VLA still lacks a stable state contract across sensor family / embodiment / consumer; world models on contact-rich tasks typically benefit from explicit contact / event representation
