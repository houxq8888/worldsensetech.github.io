---
title: 'Stacking Sensors Is Not Fusing Them: Multimodal Robotics Lacks an Interface, Not a Model'
slug: "2026-09-12-multimodal-fusion-interface"
date: 2026-09-12
draft: false
categories: ["Embodied AI", "Multimodal Perception"]
tags: ["Embodied AI", "Multimodal Fusion", "Visuo-Tactile", "Force/Torque", "Proprioception", "Representation Interface", "Multimodal State Estimation", "Hybrid State Estimator", "Structured State Contract", "Observability", "Identifiability", "Registration", "Cross-attention", "Modality Dropout", "VLA", "World Model", "Frame Alignment", "Time Alignment", "Uncertainty"]
description: 'Multimodal fusion is usually framed as a question of "which attention architecture", but in robotics the real bottleneck sits upstream: Vision / Tactile / Force-torque / Proprioception have never agreed on temporal base, coordinate base, task semantics, validity, or provenance. This piece splits the system into four layers — measurement layer (time / frame / calibration) → perception & semantic projection → state estimation → structured state contract — before handing anything to policy / world model / controller / diagnostics. The core claim is not "yet another fusion operator" but a stable structured state contract between heterogeneous observations and multiple downstream consumers: it specifies what information must be exposed, with what semantics, unit, frame, time, uncertainty, provenance, and validity, while leaving the inference algorithm and downstream representation learning entirely unconstrained. Interface ≠ inference algorithm ≠ latent representation; belief is only one kind of content this contract may carry. The contact set is one manipulation-specific instance, not the definition of the interface. The benchmark section gives an Interface Property Benchmark: schema/encoder/consumer swap, oracle/estimated/end-to-end controlled baselines, uncertainty-aware disagreement, a consistency graph with fault isolation, and a representation-bottleneck ablation. Three design principles: Register before compose · Expose belief at the interface · Design for disagreement.'
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

Picture a bimanual robot fitted with an RGB camera, GelSight fingertips, wrist six-axis force/torque sensors, and joint encoders on every link. The hardware bill of materials looks genuinely "multimodal". Yet when you hand that rig to a policy, most published work tells you "just fuse them with cross-attention and you're done". That runs in a demo; **in real deployment it will almost certainly fail in one of four ways**: a modality drops frames, a sensor dies, coordinate frames drift, or the model quietly converges to one dominant modality and treats the rest as noise. These four are not engineering footnotes. **They all trace back to the same root cause: the modalities never agreed on a structured state contract — explicitly expressing time, frame, semantics, uncertainty, provenance, and validity — that policy, world model, controller, and diagnostics can consume in common.**

This piece takes on the easiest topic to hand-wave in embodied AI: **multimodal fusion**. It will not sell you a specific network, will not argue against cross-attention, and — importantly — will not argue against end-to-end learning. What it *does* argue against is **letting state estimation, cross-modal composition, and control all happen implicitly inside a single, undiagnosable, non-reusable latent, with no explicit, cross-consumer state boundary**. The article digs fusion down to its three foundations — time, frame, task semantics — and uses them to explain **why vision was able to grow a common data interface while tactile / force-torque / proprioception have not, and what a minimum usable contract looks like if you want to start building today**.

## 0. Framework: a four-layer system and the structured state contract

Put the whole article's frame up front. Every section returns to this picture. **The single most important conceptual tightening in this version**: what the article wants to establish is not called "a better fusion layer", and is not quite the previous draft's "structured belief interface" either — the more accurate name is a **structured state contract**. Belief / uncertainty-aware state is only **one kind of content this contract may carry**, not the contract itself; the contract specifies "what to expose, and with what semantics", and **does not specify what algorithm downstream uses to infer, nor what representation the policy uses internally**.

```text
              Heterogeneous observations
              V / T / F / P / audio / ...   (raw sensor streams)
                              │
                              ▼
            ┌───────────────────────────────────┐
            │  Layer 1 · Measurement layer      │  ← home of registration
            │  time / frame / calibration /     │     (explicit, calibratable,
            │  latency / sensor noise model     │      estimatable)
            └─────────────────┬─────────────────┘
                              ▼
            ┌───────────────────────────────────┐
            │  Layer 2 · Perception &           │  ← semantic projection
            │  semantic projection              │     (observation →
            │  sensor-native encoder            │      task-relevant state)
            └─────────────────┬─────────────────┘
                              ▼
            ┌───────────────────────────────────┐
            │  Layer 3 · State estimation       │  ← home of the estimator
            │  hybrid / uncertain               │     (EKF / factor graph /
            │  x̂_t, Σ_t  or general B(x_t)       │      learned filter all valid)
            └─────────────────┬─────────────────┘
                              ▼
            ┌───────────────────────────────────┐
            │  Layer 4 · Structured State       │  ← where this article's
            │  Contract (interface)             │     contribution lives
            │  value·semantics·frame·time·      │     value · semantics ·
            │  uncertainty·provenance·validity  │     frame · time ·
            └───────┬────────┬────────┬─────────┘  uncertainty · provenance
                    ▼        ▼        ▼             · validity
        Policy   World Model  Controller  Diagnostics / Safety
                    │        ▼        ▼
                    └───── Action a_t ─────┘
```

Three things this diagram wants to insist on, and the architectural thesis of the entire article: **fusion is not a network module — it is a system problem spanning measurement / perception / estimation / contract, four layers**; **state estimator and state contract are two different layers** — the estimator is responsible for "computing $\hat{x}$, $\Sigma$, or a general $\mathcal{B}(x)$", the contract is responsible for "specifying that whatever is computed is exposed to multiple consumers with what semantics, unit, frame, time, uncertainty, provenance, and validity", and the two can evolve independently; **the interface layer is not a learned embedding — it is a contract**, where every field has a unit, a frame, a timestamp, an uncertainty, a source, a validity state, and the optimization target is **interoperability across consumers**, not "friendliness to one specific downstream model".

One sentence that carries the central claim (this version recasts it from "belief state" to "state contract" precisely to deflect the "isn't that just the state estimator's output?" objection):

> **The missing abstraction is not necessarily another fusion operator, but a stable structured state contract between heterogeneous observations and multiple downstream consumers. This contract should expose task-relevant state together with its semantics, frame, time, uncertainty, provenance, and validity, while leaving the inference algorithm and downstream representation learning unconstrained.**

Or, as the three boxed formulas that state the article's three most core claims:

$$
\boxed{\;\text{Sensor-specific observations} \;\rightarrow\; \underbrace{\text{Structured State Contract}}_{\text{stable interface}} \;\rightarrow\; \{\text{Policy, World Model, Controller, Diagnostics}\}\;}
$$

$$
\boxed{\;\text{Interface} \;\neq\; \text{Inference algorithm}\qquad \text{Belief} \;\neq\; \text{(necessarily) Bayesian posterior}\;}
$$

$$
\boxed{\;\text{representation compatibility} \;\neq\; \text{interface compatibility}\;}
$$

Once framed this way, VLA, world models, tactile foundation models, modality dropout, cross-modal contradiction, F/T constraints, and contact-set slots all become **different instances of the same thesis, not six or seven parallel talking points**.

**This article proposes neither a new fusion operator nor a new state estimator** — what it proposes is **a contract over estimator outputs**, plus the experimental methodology to test that contract. The contribution collapses to three items, and every section below can be read as a concretization of one of them:

1. **Abstraction** — introduce the **structured state contract** as a layer for heterogeneous robot observations: it specifies how cross-consumer shared state is exposed with explicit semantics, unit, frame, time, uncertainty, provenance, and validity, rather than leaving it as an implicit latent inside some downstream network.
2. **Representation** — give one concrete instantiation for contact-rich manipulation: contact set + wrench + robot state + uncertainty + provenance + validity + lifecycle (while stressing that the contact set is only one instance).
3. **Evaluation** — propose an **Interface Property Benchmark**: schema / encoder / consumer swap, oracle / estimated / end-to-end controlled baselines, uncertainty-aware disagreement, a consistency graph with fault isolation, a representation-bottleneck ablation, and registration perturbation.

The three design principles stay unchanged — they already cover the full measurement → representation → robustness chain, and this article **does not intend to add a fourth**:

> **Register before compose · Expose belief at the interface · Design for disagreement**

**The contact set is one instance of this interface under contact-rich manipulation, not the definition of the interface itself** — this sentence runs through the whole article, and is why the piece positions itself as an *architecture position paper* rather than a *tactile survey*.

## 1. "Multimodal" is not "Multiple Sensors"

The easiest way to derail this topic from the very first paragraph is to equate "a few more sensors" with "multimodal". That equation does not hold.

**"Modality" itself has no unique physics definition — it is a product of the analytic lens.** For this article's taxonomy we fix four elements: **measurement space, physical origin, noise model, and update semantics** (triggering / sampling / clock). Together these four decide whether a data stream can be treated as "the same kind of thing". Different authors, different tasks can draw the boundary a bit looser or a bit tighter, but as long as it is declared up front, subsequent discussion will not drift.

Under this taxonomy, several commonly mislabelled examples: a wrist camera plus a head camera are **same-modality multi-view**, not two modalities; the four tiny cameras inside a GelSight fingertip are **internal structure of a single tactile modality** whose measurement space is "elastomer surface deformation field" and whose downstream semantics are contact geometry, not object detection; joint encoders plus motor current are two **observation channels of a single robot-state modality** sharing the same latent physical state — treating them as two independent modalities and fusing them most likely teaches attention nothing but redundancy. Conversely, **Vision / Tactile / Force-torque / Proprioception** are the four genuinely distinct modalities under this article's taxonomy — each one's measurement space, coordinate frame, and effective observation rate appear in the §5 table. Some works additionally carry audio, thermal, gas, or ultrasound as a fifth / sixth modality, and the classification logic is identical. 9/11 §2.1 used a taxonomy tree to split contact sensing into Tactile / Force-torque / Proprioceptive branches; this article keeps the same three-way split and confines the discussion to V + T + F + P.

## 2. The two lowest layers: measurement layer and semantic projection

This section maps to Layer 1 (measurement layer / registration) and Layer 2 (perception + semantic projection) of the §0 diagram. First a **terminology tightening**: **"registration" and "semantic projection" are two things of different character**, and cramming both under "alignment" makes readers think a math transform is all it takes.

```text
Pre-fusion pipeline
├── Registration        (Layer 1 · measurement layer, explicit time/geometry/calibration variables)
│   ├── Temporal registration
│   └── Spatial registration
└── Semantic projection (Layer 2 · observation → task-relevant state variables)
```

**Registration belongs mainly to the measurement layer: it should be described, as far as possible, by explicit time, geometry, and calibration variables, and solved preferentially via calibration or estimation rather than being defaulted to an implicit learn-by-downstream-fusion-network step.** This is the article's formal definition of registration, deliberately one notch weaker than "hard constraint / closed-form solution" — because on real robots registration is often also an estimation problem (online calibration, time-varying extrinsics, compliant sensor mounting, tactile elastomer deformation, thermal drift, synchronization / latency estimation) — but **its character is still that of measurement-layer explicit variables, not task semantics**. **Semantic projection** is closer to perception and state estimation, and is the protagonist of this article's §6 state contract. Both sit **before fusion**, but their character and their division of labor are entirely different.

### 2.1 Temporal registration

The default time constants of the four streams differ by several orders of magnitude: vision ~15–60 Hz, tactile (camera-type) ~30–200 Hz, tactile (array-type) ~500 Hz – kHz, F/T ~500 Hz – 1 kHz, proprio ~a few hundred Hz – kHz. Here is a point that gets routinely confused: the numbers above are **effective observation rates**, not the sensors' internal sampling rates. The **effective observation rate should be defined explicitly** as "the rate at which temporally meaningful observations actually become available to a given consuming module", roughly written as

$$
f_{\text{effective}} \;\approx\; \min\!\big(\,f_{\text{sampling}},\; f_{\text{processing}},\; f_{\text{transport}},\; f_{\text{consumer}}\,\big)
$$

A camera-type tactile sensor may sample taxels at kHz, but if its output image is still only 30 Hz, then the effective rate the fusion layer can consume is 30 Hz.

Resampling every modality onto a common low-frequency policy clock (usually vision's) is the common baseline, but **for contact-rich control this compresses some high-frequency events into invisible aliases** — slip detection, transient contact-force peaks, joint impacts, these "event-like" signals often fall precisely between two vision frames, and downsampling is throwing them away. A more robust architecture is **multi-rate coexistence**: the vision policy may run at 30 Hz, force/tactile-driven compliant control stays at native rate, event signals ride a separate event bus.

The time details that must be written into the interface layer are not just timestamps — **latency belongs there too**. The moment an observation becomes consumable is

$$
t_{\text{effective}} \;=\; t_{\text{sens}} + \Delta_{\text{processing}} + \Delta_{\text{transport}} + \Delta_{\text{queue}}
$$

and **timestamp synchronization $\neq$ causal synchronization**: aligning a camera timestamp to a tactile timestamp does not mean the two describe the same physical-state moment — the processing / transport latency in between is what actually decides "which instant of the world state this observation corresponds to". Three time details that must be written into the interface: **timestamp semantics** (sensor / arrival / host timestamp; a 5–20 ms gap between them is already enough to misplace "the contact instant of grabbing a cup" to "before the fingers close"), **latency and causal alignment** (the $t_{\text{effective}}$ above, and the lead/lag it introduces), and **event-type vs periodic** (a slip trigger is essentially a sparse event; stuffing it into a uniformly spaced buffer destroys event density). **The judgment:** **write the time constants and latency differences down as modeling assumptions; do not hide them by downsampling.** This is the same judgment as 9/11 §4.1.

### 2.2 Spatial registration

The four streams naturally hang off four different frames: Vision camera frame, Tactile sensor frame, F/T sensor frame + tool frame, Proprio base / world frame. To put them into one fusion layer you need at least three things: **hand-eye calibration** ($T^{cam}_{base} \in SE(3)$; any single collision can drift it by a fraction of a degree to several degrees), **sensor-mounting calibration** (the fingertip sensor frame → link frame offset $T^{\text{sens}}_{\text{link}}$), and **wrench transformation with gravity / inertial compensation** (map the sensor-frame wrench to base or world, subtracting tool gravity and inertial terms; formulas in §5.2).

**One more distinction here**: registration parameters themselves may be **static** (factory-calibrated extrinsic constants $T$) or **dynamic** (time-varying latent variables). This is especially visible for tactile — $T_{\text{sens}\rightarrow\text{link}}(t)$ is not constant under elastomer deformation, compliant mounting, or thermal drift. **Treating dynamic registration as "calibrate once and freeze" is a common failure source**; more reasonable is to model $T_{\text{sens}\rightarrow\text{link}}(t)$ as a time-drifting estimate folded into the measurement layer's online estimation. This exactly supports the article's core judgment — **registration is measurement-layer estimation**, not "a constant you calibrate once and freeze".

Typical symptoms of poor spatial registration: the model drops points immediately after a tool change, a camera-angle change, or the robot being nudged out of alignment — because what the model actually learned was "some correlation under a specific sensor frame", not the task. A subtler problem is **relative vs absolute pose**: contact physics fundamentally depends only on the relative "who touches whom" relation, yet many policies feed the end-effector's absolute position in world frame, so the model learns a pile of degrees of freedom it should not have to.

### 2.3 Semantic projection

This part is strictly not "registration" but **"projecting observations onto task-relevant state variables"**; here "semantic" means **task-relevant state abstraction, not human-readable semantics**. Vision → object / scene state, Tactile → local contact state, F/T → aggregate wrench state, Proprio → self-state — writable as a set of concrete mappings:

```text
RGB        → object pose / scene geometry
GelSight   → contact patch
F/T        → external wrench
(q, q̇)     → kinematic state
```

9/11 §2.1 split signal → meaning into nine layers (Sensor → Calibration → Raw obs → Contact perception → Contact geometry & wrench → Contact mode & physical state → Task-relevant belief → Policy / controller → Action → New contact). **The key question of multimodal fusion is: at which layer do you do composition?** The common mistake is composing at the raw layer (concatenating four tensors into a network) — that forces the policy to learn all of §2.1 / §2.2 / §2.3 by itself, at very high cost and very low sample efficiency. The more reasonable approach is **to let each stream complete its own raw → perception → contact-geometry layers, then compose at the four positions where semantics have already converged — "contact event, wrench, robot state, task belief"** — which is this article's §6 structured state contract.

**The landing point of this section**: the two registration sub-parts (temporal / spatial) are measurement-layer explicit variables (Layer 1); semantic projection is the perception-vs-estimation division of labor (Layer 2). Fudge any one of the three and no matter how gorgeous the downstream fusion architecture, it is just patching up an upstream mistake.

## 3. The genealogy of existing fusion paradigms: different mechanisms, different layers

Baltrusaitis et al.'s classic survey [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) organizes multimodal ML into representation / learning / feature selection / fusion / application — this section borrows its fusion layer. **First an important positioning**: the paradigms below **are not mutually exclusive architecture choices but mechanisms on different axes** — a mature system typically uses several at once:

| Mechanism | What it solves | Which §0 layer |
| --- | --- | --- |
| Modality-specific encoder | each raw → local observation | Layer 2 · Perception |
| Temporal / spatial registration | clock, SE(3), frame, latency | Layer 1 · Measurement |
| Cross-attention | learned composition | can sit at Layer 2 / 3 / 4 |
| Shared latent / VLT-style | cross-modal alignment at representation level | Layer 2 · Semantic projection |
| Late / decision fusion | composition at decision level | Layer 4 downstream · policy |
| Structured state contract | semantics schema and cross-consumer boundary | Layer 4 · Contract |

This article does not oppose cross-attention; it **scopes its responsibility**: **cross-attention is not guaranteed to recover explicit temporal / spatial / semantic registration from raw data** — attention can compose already-aligned representations, but "learning registration on the side" is generally unreliable absent explicit constraints. This sentence is one of the hinges of the section and of the whole article.

**Early concat** (concatenation at the raw / embedding layer): concatenate four streams after their encoders into a state vector, throw it at an MLP or Transformer. The barrier is low, and with enough samples a deep network learns some alignment on its own. Downsides: differing time constants are masked by the concat and spurious lead/lag is learned; if any stream drops a frame you must zero-pad or hold-last-value, both out-of-distribution; tactile and vision encoders differ so much that gradients contaminate each other; when a missing-modality combination unseen in training appears, the policy collapses.

**A paragraph about VLA that has to be careful**: lumping RT-2 / OpenVLA / π0 all under "vision + proprio + language concat" is inaccurate. **RT-2** [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) centers on expressing actions directly as **text tokens** jointly fine-tuned with a VLM; **π0** [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) is a VLM backbone + **proprioception token** + **noisy action chunk**, producing actions via flow matching; **OpenVLA**'s [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) public configuration explicitly has multi-camera, depth, and proprioceptive-state encoding branches. **Taking these representative public systems as examples, one can observe that their common point is "using vision-language pretraining as the main scaling entry and injecting robot state as an additional representation into the policy."** Note carefully that two things must be distinguished: **"supporting a type of input" and "establishing a cross-consumer shared structured state contract" are different things** — OpenVLA can eat proprio input, which does not mean it defines a state contract reusable by world model / controller / diagnostics with unit / frame / validity. This judgment is expanded again in §9.1.

**Cross-attention / Transformer fusion**: treat each stream as a token sequence, run self / cross-attention above. Tsai et al.'s Multimodal Transformer [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) is the classic starting point here, explicitly handling "modalities with mismatched temporal granularity". A modern multi-modal transformer can absolutely be paired with timestamp / positional embeddings, relative temporal encodings, modality embeddings, frame-aware features, modality-specific encoders, modality dropout / masking, auxiliary per-modality losses, and attention masks — **the real problem is not "attention fails", but that if you do not write these down as an interface contract, attention will learn registration on the side, and what it learns is coincidence in the data distribution, not a portable interface**. In mature systems cross-attention is better used, **after perception and registration have delivered inputs to Layer 3 / 4, to compose around the state contract** (but this is not the only legal usage — see §6.5).

**Late / decision-level fusion**: each stream emits a sub-policy, and the decision layer re-weights or confidence-gates. Closer to traditional robotics — vision gives coarse alignment, F/T gives fine tuning, tactile gives slip recovery, proprioception gives nominal trajectory tracking. **A common engineering pattern** is that any stream going offline only drops one proposal rather than the whole system, and it is easy to add a safety layer. Downsides: low-level coupled information is lost (cross-modal joint info like "combining tactile and F/T, is this contact stable sliding or stick-slip" is hard to reconstruct at the decision layer once proposals are separated); the weighting rules are either hand-written and unscalable, or learned, which reintroduces the early-concat problems. The impedance control + visual coarse localization + tactile slip detection combo from 9/11 is essentially this late fusion.

**Shared latent / VLT-style alignment**: use contrastive / distillation / CLIP-style objectives to pull different modalities' representations into one latent space. Representative works: **TVL / Binding Touch to Everything** [arXiv:2402.13232](https://arxiv.org/abs/2402.13232) (Zhao et al., ICML 2024) builds cross-modal alignment via language using ~44K vision-touch pairs; **AnyTouch** [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) (Feng et al., 2025) learns a unified static-dynamic representation for heterogeneous visuo-tactile sensors; **3D-ViTac** [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) (Huang et al., CoRL 2024) reports, in that paper's experiments, a significant gain of visuo-tactile over vision-only representation; **Lee et al.'s Making Sense of Vision and Touch** [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) (ICRA 2019) is the earliest self-supervised anchor on this route. Downside: **where does the training supervision come from?** Vision-language contrastive learning can eat vast web image-text pairs; four-way alignment has no natural supervision source. So far it relies mainly on (a) teleoperation logs co-collecting all four streams by force (Calandra et al.'s *More Than a Feeling* [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) being an early demonstration on tactile grasping), or (b) using language / vision as a bridge to pull tactile into the V-L space (the route TVL takes).

**State this explicitly**: the works above show shared latent is feasible on specific task families, but their representations remain policy-specific or dataset-specific. On that basis this article proposes — if you want those representations to be reused across policy / world model / controller / sensor, you need a further layer on top of the latent: a **structured state contract** carrying unit, frame, timestamp, uncertainty, provenance, and validity. This layer is this article's core claim, not a conclusion of those papers.

**Structured state contract** (this article's recommended route): fuse neither at the raw layer nor at the latent layer, but compose on an explicitly agreed intermediate representation. That representation is not a learned embedding but a set of state slots whose semantics have already converged and that carry uncertainty / provenance / timestamp / validity. The concrete schema is deferred to §6. Advantages: a missing signal only affects its own key, other keys stay uncontaminated; temporal base, coordinate base, semantic base, and validity all converge inside the slot definitions; once the slots are fixed the fusion structure can change freely. Downsides: the slots must be designed first, which is heavier than "drop in a Transformer"; when slot precision is insufficient, downstream policies cannot learn beyond the slots; **and slots do not address the "non-contact semantics" vision provides — object identity / geometry / free space / occlusion / scene context**. What this article recommends is a state contract for contact-rich manipulation, not a general-purpose multimodal interface.

**Landing point of this section**: within the fusion-paradigm genealogy, early concat / attention / late fusion / shared latent all have their place, but almost all their failure modes trace back to the three sub-parts of §2 being done badly; this article prefers to shift the engineering center of gravity from "which attention to pick" back to "fix the state contract first".

## 4. Why vision formed a reusable data and representation ecosystem first

A natural question: the three sub-parts of §2 apply to vision just as much, so why could vision carry a whole common data interface?

**First a correction**: **this is not single-cause** — it is three bases solidified together with ecosystem conditions: standardized hardware (CMOS sensor + unified lens mount), unified file formats (JPEG / PNG / MP4 / HDF5), a coordinate model (pinhole + SE(3)), large-scale internet data, mature annotation tasks, public benchmarks (ImageNet / COCO / ADE20K / LVIS), mature encoders, a GPU scaling law, available compute. The article emphasizes only the three-bases dimension, because it is exactly what tactile currently lacks most.

**On the temporal base**, video is naturally frame-indexed at 30 or 60 Hz, and every downstream task conventionally proceeds "per frame" — that convention wipes out the trickiest class of problems in the fusion discussion. **On the coordinate base**, the pinhole model plus intrinsics/extrinsics turn image pixel ↔ world point into one formula ($s \cdot m = K [R | t] \cdot M$), and 3D vision, SLAM, NeRF, 3D Gaussian Splatting, and multi-view stereo all grow on that convention; calibration errors exist, but "what an error looks like" is predictable and reproducible. **On the semantic base**, RGB itself has no semantics, and the vision community formed a set of **highly interoperable task-level representation conventions** via COCO / ImageNet / ADE20K / LVIS — object class / bounding box / instance mask / depth / affordance / caption. **More precisely, this batch of datasets did not form a single unified semantic schema**: ImageNet is a classification ontology, COCO is detection + instance + caption, ADE20K is scene parsing, LVIS is a long-tail instance distribution — each defines its own thing, they merely compose with one another.

Contrast tactile: on the temporal base, different sensor families run anywhere from 30 Hz to kHz, event-triggering and polling are mixed, there is no consensus; on the coordinate base, GelSight is pixel + elastomer deformation, 9DTact is pixel + 3D deformation field, array taxels are a 1D / 2D force distribution, optical tactile is yet another representation — even "what shape a tactile reading actually has" is not unified; on the semantic base, concepts like contact point, normal, tangential, slip, mode all appear somewhere in the literature, but there is a shortage of **cross-dataset unified task-level conventions**. Let's also fix a common phrasing here — "9DTact calls it 6D force, GelSight calls it shear map, arrays call it taxel load, all the same physical quantity under different projections" — **they carry overlapping but differently-leveled physical information**: GelSight's shear / deformation map is closer to a raw local deformation observation, 9DTact's 6D force is a global wrench estimate obtained through model inversion. They are not two projections of one quantity; they are the **observation layer vs estimation layer** difference. This distinction directly affects the §6 slot schema.

**Landing point of this section (and a correction to how the argument runs)**: vision's success is not because ImageNet / COCO provided a unified multimodal state contract, **nor because "vision solved registration first"** — quite the opposite, vision still has not solved multi-camera time sync, extrinsic calibration, or semantic alignment. What actually happened is that **vision accumulated enough conventions that those unsolved registration problems degenerated into local, each-its-own-problem engineering hassles rather than a system-level representational obstacle.** For tactile / force / proprioception to build a fusion ecosystem of comparable scale, the first move is to **agree on the interface**, not to race on model architecture.

## 5. Each signal's own fusion difficulties

This section sets aside the time/frame/semantics bases already established in §2 and discusses only each sensor's specific modeling assumptions and difficulties, laying groundwork for §6's interface design.

| Modality | Native evidence | Main ambiguity | Canonical slot output |
| --- | --- | --- | --- |
| Vision | Scene geometry, appearance, semantics | Occlusion, view-dependent pose, identity ambiguity | Geometric / semantic observation → contact hypothesis |
| Tactile | Local deformation / force distribution at contact interface | Sensor-family heterogeneity, patch vs point, mounting offset | Local contact observation (patch-aware) |
| Force/torque | Aggregate 6D wrench at sensor frame + reference point | Contact decomposition is non-identifiable; requires model-based compensation | Aggregate wrench measurement + residual |
| Proprioception | Robot internal state $(q, \dot{q}, \tau)$, FK-based EE pose | Model mismatch, friction / backlash, no external-world signal | Kinematic / dynamic constraint |

### 5.1 Tactile: not even the raw layer is unified

For the same class of physical quantity (local deformation or force distribution at the contact interface) there are at least four implementations whose **representations are entirely different**:

```
Image-based      GelSight / 9DTact [arXiv:2308.14277] / GelSlim / TacTip
                 → RGB deformation image / multi-view image / optical flow field / end pose

Taxel array      BioTac / 1D-2D capacitive array / piezoresistive array
                 → array pressure / local normal-force distribution / local stress map

Optical waveguide AnySkin / DigiTact
                 → light propagates in the elastomer, edges trigger contact points

Proprioceptive-inferred  F/T + kinematics
                 → inverse-inferred contact ("soft tactile")
```

The four encoders differ enormously in structure: CNN for images, MLP for taxel arrays, spline models for waveguides, IK for proprioception-inferred. A common misconception is "make a tactile foundation model that handles all tactile sensors uniformly" — that is exactly what AnyTouch [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) is doing, the direction is right, **but it unifies the learned-representation layer, not the raw layer, and not the state-contract layer**. Even with AnyTouch the output is still an embedding, and a further explicit "embedding → contact slot" agreement is needed before it can enter the state contract. **Mitigation**: between raw and slot, introduce a layer of **sensor-specific decoders** that decode heterogeneous tactile outputs uniformly into §6.1's slot schema; this layer can be analytic or learned, but it must be **part of the interface**, not hidden inside the policy.

### 5.2 Force/torque: aggregate, dependent on a lot of prior compensation, and cannot invert geometry

The F/T sensor outputs a 6D wrench, uniform in format, but its **semantics** are far more complex:

$$
w_{\text{raw}} \;=\; w_{\text{contact}} + w_{\text{gravity}} + w_{\text{inertial}} + w_{\text{friction}} + b_{\text{bias}}
$$

To extract "external contact force" you must at least do: zero-drift compensation ($b_{\text{bias}}$, tare zeroing, hard to do perfectly on the real robot), gravity compensation ($w_g = g(q)$, including tool + gripper mass distribution — change the tool and the whole curve changes), inertial compensation, and frame transformation (map the sensor-frame wrench to base or world, subtracting tool gravity and inertial terms).

**There is one more frame-semantics trap about the wrench itself**: a wrench is not nailed down by "a unit + a frame_id". Torque depends on the **moment reference point**:

$$
\tau_{p_2} \;=\; \tau_{p_1} + (p_1 - p_2) \times f
$$

Same line of action, different reference point, different moment. So a wrench field, besides its orientation frame, must carry an explicit **reference point**, plus a **wrench element ordering** ($[f;\tau]$ or $[\tau;f]$) and an **active/passive transform convention**. §6.1's `Quantity` therefore expands `frame` into a small struct with `orientation_frame / reference_point / convention`.

**An important formula tightening about inertial compensation** (the v3 version mixed joint-space dynamics and sensor-frame wrench into one formula and got the direction backwards; the reviewer correctly noted that $\tau = J^T F$ maps from wrench to joint torque, and inverting it is ill-posed). A more general and more honest way to write it is not to hand over a pseudoinverse closed form, but to phrase the joint-side external-force estimate as **a regularized least-squares**:

- **joint-side estimation chain**: residual torque

$$\tau_{\mathrm{res}} \;=\; \tau_{\mathrm{meas}} - \hat{\tau}_{\mathrm{model}}(q,\dot q,\ddot q)$$

  where $\hat{\tau}_{\mathrm{model}}$ usually includes $M(q)\ddot{q} + C(q,\dot q)\dot{q} + g(q) + \tau_{\mathrm{friction}}$. But the real situation is

$$\tau_{\mathrm{res}} \;=\; J^T F_{\mathrm{ext}} + \tau_{\mathrm{null}} + \epsilon$$

  where $\tau_{\mathrm{null}}$ is self-motion internal force lying in the null space of $J^T$ and invisible to joint-torque observation, and $\epsilon$ aggregates unmodeled friction, actuator / transmission dynamics, compliance, and joint-torque estimation error. Therefore **the external force is not a determinate value that gets "recovered" but a hypothesis**, and it is safer to phrase it as an optimization:

$$\hat{F}_{\mathrm{ext}} \;=\; \arg\min_{F}\; \big\|\, \tau_{\mathrm{res}} - J^T F \,\big\|_{W}^{2} \;+\; \lambda\, R(F)$$

  In full-rank, well-conditioned configurations this reduces to the weighted pseudoinverse $(J^T)^{\dagger}\tau_{\mathrm{res}}$, but in redundant / ill-conditioned configurations it needs regularization or extra contact assumptions, and the solution is non-unique.

- **the wrist F/T side is a separate chain**: the sensor directly outputs $w_{\text{raw}}$; via an SE(3) adjoint transform and sensor-frame bias / gravity / inertial compensation you get the external wrench $w_{\text{ext}}^{\text{sens}}$, then transform to base or tool frame (with the reference-point transform of §5.2 above). **Do not** merge the two chains into one formula — the joint-space dynamics term and the inertial wrench a wrist sensor sees are two different physical quantities with different mapping directions.

For slow manipulation the joint-side residual is often enough; for fast manipulation inertial and actuator dynamics are not negligible. **If any step at this layer goes wrong, the downstream "normal / tangential force decomposition" is wrong wholesale**.

F/T has an even more fundamental limit — **what it gives is the "aggregate" wrench, not the "local" contact**. This article lists it as **one of three physics principles**:

$$
\boxed{\;\text{Physics principle 1 — Force/torque measures an aggregate wrench; it does not, on its own, uniquely decompose into individual contacts.}\;}
$$

$$
w \;=\; \sum_{i=1}^{N} \begin{bmatrix} f_i \\ (p_i - p_0) \times f_i \end{bmatrix}
$$

Infinitely many sets $\{p_i, f_i\}$ can give the same $w$. This is not "the estimator is not strong enough" but the fact that **the inverse problem is, from F/T alone, non-identifiable** (identifiability is expanded in §6.1.4). **Scope this**: given contact geometry, robot kinematics, object geometry, and a contact model, F/T can **indirectly** impose fairly strong localization constraints (e.g. "with only one candidate contact point, F/T pins the force magnitude to that point"); the problem is not "F/T cannot localize" but **"F/T alone does not provide a unique contact decomposition"**. This directly affects §6.2's slot semantics.

**Mitigation**: at the slot layer F/T should output two things — (a) a compensated external wrench (for force-aware policy), (b) a residual magnitude (to detect "did it drift again / did it bump something it shouldn't"). The second is often overlooked.

### 5.3 Proprioception: the most direct stream, and the most easily over-relied-upon

Proprio is, by default, the fastest time constant of the four, and **on conventional rigid robots it is usually the most readily available high-rate internal reference** — this needs qualification; "complete channel / natural master clock" would be inaccurate: many robots have no direct torque sensing, $\dot q$ is often numerical differentiation, EE pose is often an FK estimate, motor current and joint torque are mapped through friction / backlash / gear ratio, some soft robots, distributed architectures, or networked platforms even have multiple control clocks, and proprio may itself be delayed / estimated / filtered. A more robust phrasing is — **proprioceptive state is often the most readily available high-rate internal reference on conventional rigid robots**; it is **the stream most easily standardized and most often used as the coordinate and time-base anchor**, not "the natural master clock of the whole system".

Proprio usually plays two roles in fusion: **as an input to contact hypotheses** — given a desired end-effector trajectory plus joint-torque feedback, you can invert an external-wrench hypothesis via the §5.2 joint-side chain (contact inference, a classic tool going back to Hogan's impedance control); **as the anchor for the time base and coordinate frame** — the fusion layer needs a stable frame, proprio usually runs at the high rate hardware servos allow, and EE pose is often used as the anchor for other frames. **The fusion difficulty**: proprio's information is almost too "clean" — it is the robot measuring its own state, with no external-world uncertainty, so the model easily **over-relies on proprio**, treats it as a shortcut, performs well in contact-free scenes, and fails to learn to use tactile / F/T in contact scenes. This is exactly what 9/11 §5.2.1's feedback-value ablation aims to avoid.

### 5.4 The time-constant difference is itself a modeling assumption

Combining §5.1–5.3 with §2.1 lets us distill this judgment: **the difference in effective observation rates across modalities is not an engineering question of "should the fusion layer handle it" but a modeling assumption about "what control bandwidth the task needs"**. Wiping / polishing need force-control bandwidth of at least 100–500 Hz, vision at 30 Hz is entirely enough, tactile and F/T must run at native rate, proprio must reach the torque layer — for such tasks the fusion layer cannot downclock everything to 30 Hz; for pick-and-place / assembly vision dominates and contact events are sparse, so downsampling tactile to vision rate is an acceptable approximation. Specific numbers vary with hardware, task, and controller architecture, and are not written as absolutes here.

## 6. A minimum usable structured state contract

Here we can answer "what to actually do" head-on. This section gives a **deployable, benchmarkable, incrementally evolvable** minimum interface.

**First a scope tightening**: what this section proposes is neither "all multimodal robot information should be compressed into contact sets" nor "structured state contract ≈ contact set". **The contact set is one core instance of this contract under contact-rich manipulation, not the definition of the contract itself** — this sentence decides whether readers from locomotion / navigation / whole-body manipulation can put their state into the same contract. The contract itself contains at least (note that contact_state is only one manipulation-specific branch):

```text
Structured State Contract
├── robot_state      ── (q, q̇, τ, EE pose) + availability
├── scene_state      ── free-space / object pose / occlusion … (optional)
├── task_context     ── language token / goal embedding
├── events           ── contact / slip / impact / mode-switch sparse events
├── wrench           ── 6D + reference point + covariance + validity
├── contact_state    ── expanded in detail in §6.1, a manipulation-specific instance
└── uncertainty / metadata ── unit / frame / provenance / validity threading every field
```

In set language: $\text{ContactSet} \subset \text{StructuredState}$, not $\text{StructuredState} \approx \text{ContactSet}$. For tasks with no contact semantics (e.g. pure navigation), the contact_state branch can be absent wholesale and the contract still holds. **The contact set is the instance; the contract is the contract.**

### 6.1 Contact slot schema (a manipulation-specific instance)

Define first the cross-modal shared **contact-event record** (with uncertainty / provenance / timestamp / validity):

$$
C_t = \big\{\, \big(\, \mathrm{track\_id}_i,\; G_i,\; \mathcal{F}_i,\; m_i,\; \mathcal{U}_i,\; \mathcal{P}_i,\; \mathcal{I}_i,\; \mathcal{L}_i \,\big) \,\big\}_{i=1}^{N_t}
$$

$G$ geometry, $\mathcal{F}$ wrench, $m$ mode, $\mathcal{U}$ uncertainty (written out separately), $\mathcal{P}$ provenance (object-ified), $\mathcal{I}$ evidence mask, $\mathcal{L}$ lifecycle. Field by field:

```python
contact = {
    # identity (cross-frame consistency is only an "assumption", not an intrinsic property)
    "track_id":            Optional[int],
    "track_confidence":    float,           # ∈ [0, 1]; "track_id=17" and
                                            # "track_id=17 + confidence=0.97"
                                            # mean entirely different things to the interface

    # geometry: do not assume it must be a point contact
    "geometry": {
        "type":            enum,            # point | patch | region
        "position":        Quantity,         # {value: Vector3, unit: "m", frame: ...}
        "normal":          Quantity,         # {value: Vector3, unit: "-", frame: ...}
        "extent":          Optional[Quantity],  # if type ≠ point
        "center_of_pressure": Optional[Quantity],
    },

    # wrench (canonical point / patch summary) — must carry reference point + convention
    "wrench": {
        "f_perp":          Quantity,         # {value: float, unit: "N", frame: ...}
        "f_parallel":      Quantity,         # {value: Vector2, unit: "N", frame: ...}
        "moment":          Optional[Quantity],  # if type ≠ point
    },

    # mode and events
    "slip_probability":    float,           # ∈ [0, 1]
    "mode":                enum,            # free / touch / sticking /
                                            # sliding / rolling / separating

    # uncertainty: continuous and categorical must be kept separate, and match the
    # topology of the state variable
    "uncertainty": {
        "pose_covariance":      Matrix,     # Σ on ℝ³ or SE(3)
        "force_covariance":     Matrix,
        "slip_probability":     float,      # scalar, not a covariance
        "mode_probability":     Vector,     # categorical distribution
        "count_uncertainty":    Optional[Distribution],  # N_t ∈ {0,1,2,...} is
                                            # combinatorial, not expressible by a single covariance
        "calibration_quality":  enum,       # nominal / degraded / unknown
        # note: a single Gaussian covariance is insufficient to represent a
        # multimodal posterior hypothesis; covariance also does not separate
        # measurement noise / model uncertainty / calibration uncertainty. When
        # finer attribution is needed, split aleatoric / epistemic / calibration,
        # or at least record σ_sensor, σ_model, σ_calibration.
    },

    # observability: not all state is observable from the four streams
    "observability":       enum,            # observable | partially_observable
                                            # | unobservable

    # provenance: object-ified, no longer a single enum
    "provenance": {
        "primary_sources":      ["tactile", "force_torque"],   # main contributing streams
        "contributing_mask":    [V, T, F, P] → [0/1, 0/1, 0/1, 0/1],
        "negative_evidence":    ["vision"],   # streams that explicitly "ruled out a
                                            # hypothesis" are also evidence (see §6.1.3)
        "correlated_with":      ["proprio"],  # shares a latent with this slot's evidence,
                                            # guards against double counting (see §6.2)
        "estimator":            "contact_estimator_v2",
        "calibration_version":  "...",
    },

    # time and lifecycle (§6.1.1)
    "timestamp":           float,           # host clock, seconds
    "age":                 float,           # seconds, freshness
    "valid_from":          float,
    "valid_until":         float,
    "availability":        enum,            # present / stale / delayed /
                                            # unavailable / corrupt (= can I access it)
    "validity":            enum,            # valid_now / valid_for_past /
                                            # conditionally_valid / invalid (= can I trust it)
    "lifecycle":           enum,            # new / tracked / occluded /
                                            # lost / merged / split
}
```

**The `Quantity` frame expansion**: every numeric field in the schema has type `Quantity = {value, unit, frame, ...}`, with unit in SI canonical notation (metre, newton, newton-metre, second, radian). But for geometric quantities like wrench / moment a single `frame_id` is not enough; `frame` must be a small struct `{orientation_frame, reference_point, convention}` — because torque depends on the reference point (§5.2's $\tau_{p_2} = \tau_{p_1} + (p_1-p_2)\times f$) and on the $[f;\tau]$ / $[\tau;f]$ ordering convention. This layer is not "semantically requiring a unit" but "letting the schema machine-verify the unit / frame / reference-point combination" — §6.8's runtime validator asserts this metadata for every slot.

This schema deliberately achieves several things: **sensor-agnostic** (GelSight, taxel arrays, F/T — anything that can fill these keys enters the state contract); **physically interpretable + unit-explicit** (every number has a clear SI dimension and frame / reference point); **uncertainty-aware** (continuous covariance kept separate from categorical probability; the uncertainty representation must **match the topology of the state variable** — contact count is combinatorial, a multimodal hypothesis cannot be crammed into a single Gaussian, and the aleatoric / epistemic / calibration split interface is retained; **a slot without uncertainty is not an interface but a "fact", which never holds on a real robot**); **observability-aware** (§6.1.4 below); **provenance-aware** (primary_sources + contributing_mask + negative_evidence + correlated_with + estimator + calibration_version — corresponding directly to §7.5 modality dropout, §8.6 contradiction, §8.7 consistency graph); and **availability / validity / lifecycle-aware** (§6.1.1 below).

#### 6.1.1 availability and validity are two different axes

The previous version crammed staleness into a single `availability` field, which invites semantic conflict. This version splits it into three explicitly orthogonal axes:

```text
availability = can I access this observation          (present / stale / delayed / unavailable / corrupt)
validity     = under what time/model assumptions can I trust it (valid_now / valid_for_past / conditionally_valid / invalid)
age / freshness = how old is it                       (seconds)
```

So "present + stale + valid_for_past" is a self-consistent combination: **a stale observation is still accessible right now (availability=present), still valid for past states (validity=valid_for_past), just not necessarily valid for the current state**. The article's unified line — **staleness is not invalidity; it is validity relative to a past timestamp**. For the controller the three states are handled differently: stale-but-valid-for-past → hold + covariance inflation + fallback; fresh-uncertain → keep consuming but downweight; unavailable → take §6.6's explicit degradation path. **This layer is a requirement of interface design itself, not a benchmark trick.**

#### 6.1.2 track_id is a hypothesis, not an intrinsic property

v2 wrote id as "tracking ID, consistent across frames", which is over-optimistic. A contact set is essentially a **permutation-invariant set** with no intrinsic stable identity: multi-fingered manipulation, rolling contact, contact-patch splitting / merging, occlusion, and transient contacts all make identity assignment itself an inference problem. From v3 on, the field became `track_id: Optional[int]` + `track_confidence: float`, with this made explicit:

> **track_id is a hypothesis maintained by temporal association, not an intrinsic physical property of a contact.**

A slot is still valid without a track_id; with a track_id it is a confidence-bearing hypothesis consumable downstream. `track_confidence` can also be partly overridden by the lifecycle field (e.g. `lifecycle = "split"` means the two child slots' track_confidence should drop in sync).

#### 6.1.3 provenance must also express "negative evidence"

`primary_sources` records the streams that **support** a contact's existence, but when a stream explicitly **rules out a hypothesis** it is equally evidence. E.g. vision reports "no visible contact", tactile reports contact, F/T reports a wrench inconsistency — here vision is not a contributing source, but it provides negative evidence. So provenance keeps a `negative_evidence` field (conceptually; the field can be deferred). It turns "evidence" from "who agrees" into "who provided a likelihood update (positive or negative) to some hypothesis".

#### 6.1.4 observability, identifiability, and "don't turn an under-determined problem into a single point estimate"

However beautiful the interface, one fact stands: **not all state is observable from the four streams**. F/T alone makes contact decomposition unobservable / non-identifiable (§5.2); under visual occlusion contact location is only partially observable; tactile is often only weakly observable for object identity. So structured state is not just $(\hat{x}, \Sigma)$ but implicitly carries **observability / identifiability** — §6.1 records at least this with an `observability ∈ {observable, partially_observable, unobservable}` field.

Going further: **when the inverse problem is non-identifiable, the interface should let multiple hypotheses coexist rather than forcing a single contact point**. This is exactly the value of a contact set (a set, possibly multi-solution) over a "single point estimate". Write it explicitly to prevent misreading:

> **A single Gaussian covariance is not sufficient to represent multimodal posterior hypotheses; the uncertainty representation should match the topology of the state variable (continuous, categorical, combinatorial, or set-valued).**

This also reinforces the other face of §5.2's physics principle — **structured belief ≠ hallucinated point estimate**.

### 6.2 How each stream maps to a slot · F/T is a constraint, not a detector

Mapping of the four merged signals → slot semantics:

```text
Tactile         → local contact observation      (direct observation, with sensor noise)
F/T             → global wrench measurement      (aggregate independent measurement; does not localize/decompose)
Proprioception  → kinematic / dynamic constraint (via §5.2 argmin; depends on model accuracy)
Vision          → geometric / semantic observation
                  → contact hypothesis / prior    (both an observation and a prediction)
```

One **illustrative** formulation is a Bayesian factorization — but note this is only scaffolding to write the evidence-fusion relation clearly, not the algorithm this article advocates:

$$
p(C_t \mid O_{1:N}) \;\propto\; p(O_{1:N} \mid C_t)\, p(C_t)
$$

The dependency structure inside $p(O_{1:N}\mid C_t)$ can, as the system needs, be chosen as the naive factorization below, or any graphical / learned structure:

$$
p(O_{1:N}\mid C_t) \;=\; p(V\mid C_t)\,p(T\mid C_t)\,p(F\mid C_t)\,p(P\mid C_t)\quad (\text{one illustrative factorization})
$$

**Three caveats that must be written down** —

**(i) This article is not proposing Bayesian fusion.** The formulas above are just an illustrative factorization. The truly general form is $p(C_t\mid O_{1:N})\propto p(O_{1:N}\mid C_t)p(C_t)$, where $p(O_{1:N}\mid C_t)$ may adopt any graphical / learned dependency structure. Writing it as a conditionally independent product is only to make the evidence-fusion relation legible at a glance, and does not mean the article's contribution is "probabilistic contact estimation".

**(ii) Conditional independence is a simplification, and it double counts.** The factorization above implicitly assumes $p(V,T,F,P|C) = p(V|C)p(T|C)p(F|C)p(P|C)$. On real robots the four streams share a lot of latents: robot pose, object pose, contact geometry, dynamics, calibration, actuator state. **The sharpest trap**: the external wrench inferred from proprio ($\hat F_{\mathrm{ext}}$ from $\tau_{\mathrm{res}}$) and the external wrench directly measured by F/T share the same batch of physical evidence, so multiplying $p(F\mid C)\,p(P\mid C)$ directly very likely **counts the same evidence twice (double counting)**. The correct approach is to **explicitly track correlated evidence / shared latent variables** — either use a joint Gaussian likelihood with a shared covariance, or in the graphical model / factor graph connect both observations to the same wrench latent and annotate it in §6.1's provenance `correlated_with` field. This is one reason §8.7's consistency graph exists.

**(iii) The interface does not prescribe an inference algorithm.** **The interface does not prescribe a Bayesian inference algorithm. "Belief" here denotes uncertainty-aware state information; Bayesian posterior inference is one implementation, not a requirement of the schema. The contribution is not a new estimator — it is a contract over estimator outputs.** The same §6.1 slot schema can be produced by EKF / UKF, factor graph, particle filter, learned filter, diffusion state estimator, Transformer state estimator, or hybrid neural-symbolic estimator — as long as the output satisfies §6.4's 7-tuple formula (Value / Semantics / Frame / Time / Uncertainty / Provenance / Validity). This turns the interface from "some probabilistic architecture" into a real "contract proposal".

### 6.3 Evidence ranking: hypothesis-dependent, not a fixed order

A fixed order "tactile > F/T > proprio > vision" is wrong. The correct statement is that **evidence ranking should be hypothesis-dependent**:

```text
Contact location:    Tactile > Vision > F/T ≈ Proprio
Global wrench:       F/T > Tactile > Proprio > Vision
Object pose:         Vision > Tactile > F/T ≈ Proprio
Joint state / τ_res: Proprio >> others
Contact mode:        Tactile > F/T ≈ Proprio > Vision
Slip probability:    Tactile > F/T (rate) > Proprio (residual) > Vision
```

**Different fields within the same $C_t$ record can have entirely different dominant modalities** — which is why §6.1's provenance records both `primary_sources` and `contributing_mask`.

### 6.4 Interface ≠ learned latent representation

This subsection is the core of the article's architectural claim.

```text
Latent representation             Structured state contract
─────────────────────────         ─────────────────────────────
opt. target: friendly to the       opt. target: interoperable across
downstream model                   multiple consumers
units: implicit                     units: canonical unit explicit in Quantity
frames: implicit (buried in the     frames: declared explicitly, fields carry
data flow)                          frame + reference point
time: implicit                      time: explicit timestamp + age + latency
validity: none                       validity: availability / validity / lifecycle
uncertainty: implicit or only       uncertainty: continuous covariance + categorical
in the loss                          distribution + topology match + calibration metadata
provenance: none                     provenance: primary + contributing + negative + correlated
consumers: one model                 consumers: policy / world model / controller /
                                       diagnostic tool / safety layer / another sensor
```

One formula worth keeping:

$$
\text{Contract} \;=\; \big(\, \text{Value},\;\text{Semantics},\;\text{Frame},\;\text{Time},\;\text{Uncertainty},\;\text{Provenance},\;\text{Validity} \,\big)
$$

**Two important qualifications in this section** —

**(a) This is not saying "latents inherently lack uncertainty / provenance".** Today we can absolutely have probabilistic latents, uncertainty-aware latents, timestamped latents, multimodal latents. What this article distinguishes is:

$$
\text{latent} \;\not\Rightarrow\; \text{contract}\qquad(\text{not}\quad \text{latent} \Rightarrow \text{no uncertainty})
$$

The accurate statement is — **a generic learned latent does not *guarantee* these semantics unless they are explicitly encoded and standardized across producers and consumers**. A latent can be a very good fit for one policy yet not be guaranteed as a stable input interface for a world model, controller, diagnostic tool, or another sensor — that is why an "interface" is necessary, and it is exactly the missing link in the §3.4 shared-latent works.

**(b) structured ≠ symbolic-only.** The consumer of this contract can absolutely be a neural network. Structured state can be embedded and fed to a policy, or fed directly to a controller:

```text
structured state → neural embedding → policy
structured state → controller
```

Making this point explicit is to deflect the attack "are you dragging learned robotics back to hand-designed symbolic state" — **this article claims that cross-consumer shared state needs an explicit, standardized boundary, not that this boundary must be human-readable symbols**.

### 6.5 When to fuse: reusable state should be exposed at the contract (not "all fusion must happen here")

**This section is an important downgrade relative to the previous draft's line.** The previous draft wrote "the timing of fusion: not at raw, not at decision, at the structured state contract" — that was too absolute. Modern multimodal learning can absolutely fuse simultaneously at raw / latent / decision levels: learned latent fusion, neural implicit state estimators, end-to-end visuomotor policies, diffusion policies, and world-model latent fusion can all be stronger than handcrafted slots, and they need not first converge $C_t$ into explicit contact positions to get performance.

So what this article really claims is a **much weaker, yet much harder to attack**, proposition:

> **Reusable multimodal state should be exposed at the structured state contract, while learned fusion may still occur before, within, or after that interface.**

English gloss of the Chinese: **this article does not require all fusion to happen at the structured state contract; it claims that any multimodal state needing to be shared, diagnosed, checked, or reused across embodiments by multiple downstream consumers should be exposed explicitly at this layer.** With §6.1's slot schema, a typical composition is three steps:

```python
# Layer 1+2: per-modality perception + registration + semantic projection
raw_v  → enc_v → pred_contact_hypothesis      (vision's contact prediction, with confidence)
raw_t  → enc_t → detected_contact             (tactile observation, with pose_covariance)
raw_ft → enc_ft → external_wrench + residual  (compensated wrench, with force_covariance)
raw_p  → τ_res → argmin_F ‖τ_res − JᵀF‖²_W + λR(F) → inferred_contact (with model error)

# Layer 3+4: hybrid state estimation → write to the state contract
# (Bayesian / EKF / factor graph / learned filter are all optional implementations)
C_t         = state_estimator(V, T, F, P)   # outputs a belief, not a hard fact
wrench      = compensate_ft(raw_ft)
robot_state = (q, q̇, τ, EE_pose) + availability
contract_t  = assemble(C_t, wrench, robot_state, task_ctx)   # one contract

# consumers: policy / world model / controller / diagnostics consume the contract
a_t = policy(contract_t)   # the policy may still do learned fusion internally
```

The key judgment — **the Layer 4 contract schema is the cross-consumer interface of the whole multimodal system**. Layer 1–2 encoders can be swapped freely, Layer 3 estimators can be swapped freely, the consumer's policy structure can be swapped freely, as long as the Layer 4 schema stays stable. And — **the state contract is the system's stable interoperability boundary, but need not become the only information path for all downstream models**: a policy can absolutely take structured slots plus extra raw visual features / language tokens / task embeddings at the same time. What truly must be kept is that **the part of state shared across consumers** goes through this contract, not that "all fusion must pass through here".

### 6.6 Fallback when a modality is missing: treat "one stream down" as part of the training distribution

In real deployment, sensors dying, dropping frames, and timing out are the norm. The interface layer must explicitly support "one stream temporarily has no data":

```python
if tactile_missing:
    C_t = state_estimator(V, F, P)         # no T; p(T|C_t) becomes a uniform likelihood
    for r in C_t:
        r.uncertainty.pose_covariance  *= inflation_T_missing
        r.uncertainty.mode_probability  = uniform_over_modes
        r.provenance.contributing_mask.T = False
        r.provenance.primary_sources.remove("tactile")
        r.availability     = "unavailable"
        r.validity         = "invalid"        # this stream itself has no trustworthy value
```

**This is not an engineering patch — it is a core training constraint.** See §7.5 modality dropout, §8.5 degradation modes, §8.6 cross-modal contradiction test.

### 6.7 wrench (measurement) vs contacts[] (hypothesis): not redundancy, but a measurement ↔ hypothesis consistency layer

A question a reviewer will raise immediately: if `contacts[]` already has each contact's force and moment, why still need a global wrench? The answer is —

$$
w_{\text{ext}} \;\neq\; \sum_i w_i^{\text{reconstructed}}
$$

**`wrench` is an independent aggregate measurement** (the overall wrench the F/T sensor directly gives), **`contacts[]` is a structured hypothesis** (the joint inference of each perception stream + the state estimator). The two are different kinds of quantity, and produce a residual between them:

$$
\mathcal{C}_{\text{wrench}} \;=\; \big\| w_{\text{ext}} - \textstyle\sum_i \big[f_i;\, (p_i - p_0) \times f_i\big] \big\|_{\Sigma^{-1}}
$$

The larger this residual, the more the contact hypothesis disagrees with the F/T observation, and the system should raise a consistency alarm or pull `mode_probability` toward uniform. This is worth emphasizing: it is exactly this measurement ↔ hypothesis residual that turns §0's **Design for disagreement** from a slogan into a **mathematical object**. So — **`wrench` is not a redundant field of `contacts[]` but an independent aggregate measurement that can serve as a consistency constraint on contact-set reconstruction**; it is also the physical basis for the F↔T, F↔V, F↔P edges in §8.7's consistency graph.

### 6.8 Schema evolution / interface compatibility (where the unit / frame validator lands)

If the interface really exists, it must answer a very practical engineering question: **can an old policy consume a new sensor version?**

```text
v1  contact: p, n, force
v2  contact: p, n, force, slip_probability, covariance
v3  contact: geometry{type, pose, extent}, wrench{f, m}, mode_distribution
```

If the interface is an API, it has to handle API versioning. A minimal set of agreements:

```text
Interface compatibility
├── version tag            # every slot carries a schema version
├── optional fields        # old consumers ignore new fields; new consumers default missing ones
├── backward compatibility # a v2 consumer can read a v1 slot; a v1 consumer can safely
│                          # degrade-read a v2 slot
├── unit / frame validator # schema metadata defines each field's canonical unit, frame,
│                          # reference point; runtime assert; Quantity
│                          # {value, unit, frame{orientation, ref_point, conv}} is where
│                          # this validator lands
└── graceful degradation   # inflation / default / alarm strategy on missing fields
```

Without this layer the "state contract = API" analogy does not hold. It is also the precondition for §8.1's schema-swap benchmark.

## 7. Fusion failure modes and diagnostics

From a reviewer's perspective, the five classes below are the most common in robot multimodal work, and the easiest to gloss over with "we used cross-attention, so we're robust". Each class follows four beats: **Failure mode → Observable symptom → Diagnostic test → Mitigation**.

### 7.1 Modality collapse: the policy quietly reverts to vision / proprio dominance

- **Failure mode**: some modalities are marginalized during training into de-facto dead inputs.
- **Observable symptom**: training loss and validation success rate both look normal; the moment you occlude vision in eval, the policy barely changes.
- **Diagnostic test**: the primary metric is **modality marginal contribution** (§8.4) — if $\Delta_{\text{mod}} \approx 0$, this stream is effectively not consumed by the policy. The reason 3D-ViTac's [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) ablation matters is that it gives this metric a concrete reference point (**the visuo-tactile > vision-only gain reported in that paper's experiments is evidence of such a contribution's existence, not a universal law**). Attention-weight visualization is only an **auxiliary diagnostic visualization**, not a basis for judging modality contribution — **attention ≠ causal importance**: a low attention weight for a modality does not mean it contributes nothing, and a high weight does not guarantee performance drops without it; this is a classic interpretability trap.
- **Mitigation**: (a) run explicit modality dropout in training, see §7.5; (b) add auxiliary supervision per modality (e.g. the tactile encoder, besides feeding the policy, must independently predict slip events); (c) use 9/11 §5.2.1's feedback-value benchmark rather than pure success rate, to force out the true contribution.

### 7.2 Temporal smearing: all signals force-interpolated to 30 Hz

- **Failure mode**: high-frequency physical events are erased by downsampling.
- **Observable symptom**: policies for slip detection, transient contact, and impact tasks cannot be learned.
- **Diagnostic test**: compare "fuse at 30 Hz vs fuse at native rate" success-rate difference; or use $\Delta_{\text{tail}}$ to look only at hard-contact-condition performance.
- **Mitigation**: see §5.4 — treat the time-constant difference as a **modeling assumption**; use different frequencies at different layers, events on an event bus; do not downsample high-frequency signals away at the interface layer.

### 7.3 Frame confusion: the model learns coordinate-frame spurious correlations

- **Failure mode**: what the model learned is correlation under a specific sensor frame, not the task.
- **Observable symptom**: excellent at the training pose, training camera placement, and training tool model; moves slightly and collapses.
- **Diagnostic test**: the eval set actively applies **frame perturbations** (four sub-classes in §8.9).
- **Mitigation**: carry `frame` (including reference point) explicitly in the state contract, so downstream consumers must transform to read it; in training, treat frame perturbation as part of domain randomization (Tobin et al. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)).

### 7.4 Semantic leakage: the raw visual path rewrites contact semantics, bypassing the slot

- **Failure mode**: the definition of contact state is quietly overwritten by an unconstrained visual latent.
- **Observable symptom**: the slot is defined as "contact geometry", but policy performance depends heavily on visual appearance (same object, change the photo background, and it drops points).
- **Diagnostic test**: replace the visual contribution in the slot with a "minimally sufficient" synthetic slot (e.g. swap the predicted contact for ground-truth contact) and watch the performance change.
- **Mitigation**: this article **does not** claim "raw pixels must not enter the policy directly" (modern VLAs are exactly image → vision encoder → token representation → policy; feeding raw visual latents to the policy is standard); this article claims — **for the contact-related state defined by the state contract, there should be no undiagnosable raw-visual shortcut that bypasses the slot encoder**. That is, "the canonical path for contact semantics is the slot encoder", not "visual features may not enter the policy". This is consistent with, and more deployable than, 9/11 §3.5.

### 7.5 Missing / degraded modalities: collapse on one failure, or silently degrade without alarm

- **Failure mode**: training always has complete modalities; runtime degradation is out of distribution.
- **Observable symptom**: a single tactile frame drop causes a violent policy-behavior change, or silent degradation with no alarm.
- **Diagnostic test**: run a **modality masking / dropout test** (randomly drop a stream with probability (1-p)); modality masking / dropout is a **common training strategy** within missing-modality robustness, cf. Maiga et al.'s MMP [arXiv:2410.03010](https://arxiv.org/abs/2410.03010).
- **Mitigation**: the interface layer must express all six degradation classes of §8.5, not just "present/absent" two flags. §6.1's `availability / validity / age / lifecycle / uncertainty` exist precisely so these states are detectable in the data format — **a requirement of interface design itself, not a benchmark trick**.

## 8. Interface Property Benchmark and evaluation

This section integrates the diagnostics scattered through §7 into an executable benchmark skeleton. The article names it the **Interface Property Benchmark** — because it measures not "how high this fusion model's success rate is" but "whether this interface holds up as an abstraction boundary". The overall approach follows 9/11 §5.2.1's feedback-value claim: **a benchmark should test whether the interface really is an abstraction boundary, whether modalities are truly irreplaceable, whether the system handles disagreement correctly — not "whether this model can memorize the training distribution"**.

A methodological stance that runs through this section goes first: **evidence of interface value should not be only task success**. $\Delta_m$ and success rate only show "some modality is useful / some pipeline runs"; what actually proves "this is an interface, not a naming game" are four properties — **interchangeability, degradation, diagnostics, cross-consumer reuse**. §8.1 / §8.2 below map onto these.

### 8.1 Interface & schema swap (interoperability benchmark, this article's primary experiment)

If the interface is an abstraction boundary, swapping encoder / consumer should not require redesigning the middle layer. Four concrete experiments:

1. **Sensor encoder swap**: tactile encoder A → encoder B (different backbone, even different sensor family), slot schema unchanged, policy weights unchanged, watch performance transfer.
2. **Policy swap**: MLP policy → Transformer policy, slots unchanged, watch downstream training cost.
3. **Consumer swap**: policy ↔ world model ↔ controller ↔ diagnostic tool — four consumers sharing the same slots, trained independently — watch whether the interface is truly "reusable".
4. **Schema swap / evolution**: new in v5, and the most critical — not just swapping encoders but swapping **schema versions**. E.g. v1 (position + force) ↔ v2 (position + force + covariance + provenance); test both directions: **old consumer + new producer** (can an old consumer safely degrade-read the new producer's fields) and **new consumer + old producer** (can a new consumer default / inflate missing fields). This is the real **interface compatibility** test; running only encoder swap proves representation transfer, not interface compatibility.

If all four can achieve "swapping encoder / consumer / schema version does not require redesigning the middle layer", this experiment is far stronger than another success-rate run — **it turns "interface" from a metaphor into a measurable property**. §8.1 and §6.8 are a pair: schema versioning guarantees forward compatibility, swap guarantees horizontal pluggability.

### 8.2 Oracle-slot / Estimated-slot / End-to-end: three controlled baselines (make explicit what each measures)

To test "is a state contract necessary" you must run three baselines — **but you must also nail down their roles and information budgets**, or the $\Delta$ gets misread:

```text
A · Oracle interface   : ground-truth contact slots → policy
B · Estimated interface: sensor → state estimator → estimated slots → policy
C · End-to-end fusion  : sensor → fusion policy (no explicit interface)
```

**Formally define the semantics of the three quantities**: $S_A$ = "downstream policy performance conditioned on ideal state" (upper bound conditioned on ideal state); $S_B$ = the real interface pipeline (estimation + interface); $S_C$ = unconstrained end-to-end baseline. So read separately:

$$
S_A - S_B \;=\; \text{state / interface estimation gap}\qquad
S_B - S_C \;=\; \text{net value of an explicit interface over end-to-end}
$$

**Proactively plug the hole a reviewer will hit**: $\;S_B > S_C\;$ **by itself does not prove the interface is "necessary"** — it may just be that slots are easier to learn, oracle supervision is stronger, architecture capacity is unfair, or the end-to-end baseline was under-tuned. So the three-way comparison must **control the information budget** (align trainable params / training samples / supervision access as far as possible), and place the primary evidence of interface value on §8.1's interchangeability and §8.5–§8.7's degradation / diagnostics, not on a single success-rate difference. **The factorization $S_{\text{total}} \approx S_{\text{interface}} \times S_{\text{downstream}}$ is a tool for separating estimation gap from downstream capability, not a single proof that "the interface is better".**

### 8.3 Slot-level fidelity, measured separately

**Slot accuracy should be measured independently of the policy**: contact position vs ground-truth IoU / distance error; slip-detection AUROC; contact-mode macro-F1; **uncertainty calibration** — reliability diagram or negative log-likelihood for the continuous part, Brier score or expected calibration error (ECE) for the categorical part; for patch scenes add extent / center-of-pressure error; for track_id add identity preservation (ID-switch count, MOTA / IDF1). These are "quality metrics before the fusion layer"; if they fail, a high policy success rate can only be overfitting.

### 8.4 Modality marginal contribution and graceful degradation

**Marginal contribution** (the performance increment each modality brings):

$$
\Delta_m \;=\; S(M) - S(M \setminus m)
$$

**Terminology correction**: $\Delta_m$ measures **marginal performance contribution**, not information gain in the information-theoretic sense. A true information gain would be the conditional mutual information $I(X_m; Y \mid X_{-m})$; if you want to report MI, define it separately. This section names it formally **Modality marginal contribution**.

**Graceful degradation** (the shape of decline under partial loss):

$$
G_m(p) \;=\; \frac{S(M, p) - S(M \setminus m)}{\Delta_m}
$$

**Fix the symbol and the domain of applicability first** — $S(M, p)$ is "performance when modality $m$ is dropped at an independent **masking/dropout** rate $p$", with $S(M, 0) = S(M)$; hence $G_m(0) = 1$, $G_m(1) = 0$. **This normalized curve is defined only for pure masking / dropout** — it assumes $S(M, p{=}1) = S(M\setminus m)$, but stale / biased / corrupted are all not equivalent to missing, so those degradation modes are defined separately in §8.5 and not folded into this $G_m(p)$. A "good system" can have large $\Delta_m$ (this modality is irreplaceable) yet $G_m(p)$ declining smoothly from 1 to 0 (partial loss doesn't collapse it immediately). **Also make explicit**: do not imply a good system's $G_m(p)$ is necessarily monotonic or smooth — real systems often have threshold effects. The more accurate statement is **graceful degradation should be characterized rather than assumed monotonic or smooth** — report the shape, don't presuppose it.

### 8.5 Degradation modes beyond dropout: test masking ≠ missing ≠ stale ≠ latency ≠ bias ≠ corruption separately

Both $\Delta_m$ and $G_m(p)$ rest on "randomly drop one stream independently", but **real sensor degradation is far more than dropout**. This section extends the test from Bernoulli masking into an explicit hierarchy, because **they differ completely in statistical meaning to the estimator**:

$$
\text{Random masking} \;\neq\; \text{Missingness} \;\neq\; \text{Staleness} \;\neq\; \text{Latency} \;\neq\; \text{Bias} \;\neq\; \text{Corruption}
$$

The sharpest contrast: $p(y_m)=0$ (the stream is entirely gone) vs $y_m = y_{\text{true}} + b$ (the stream carries a systematic bias but looks "normal") are two completely different problems for inference. Where the interface is truly valuable is exactly that its `availability / validity / age / uncertainty / provenance` can **express these six distinctly**. Report a separate success-rate curve per degradation class. **This is the direct test of "whether the interface really handles uncertainty / provenance / validity"** (cf. §6.1.1, §7.5).

### 8.6 Cross-modal contradiction: use uncertainty-aware disagreement, not just raw distance

**Good fusion $\neq$ agreement**. This section is the part with the most original potential: this article evaluates fusion not by "whether it combines information" but by "**whether it behaves correctly when the information disagrees**".

The previous draft defined contradiction as a raw distance between two predictions, $D_{ij}(z) = d(p_i(z), p_j(z))$ — but **a mean mismatch between two posteriors is not a fault**. E.g. vision has large uncertainty, tactile small, means differ a little — that may be entirely within expected noise, not a contradiction. So this article turns disagreement into an **uncertainty-normalized measure**, which for Gaussians is the Mahalanobis form:

$$
D_{ij} \;=\; (\mu_i - \mu_j)^{\top}\,\big(\Sigma_i + \Sigma_j\big)^{-1}\,(\mu_i - \mu_j)
$$

Only when $D_{ij} \gg 1$ is it a **statistically meaningful contradiction**. For non-Gaussian / categorical cases, replace $d(\cdot,\cdot)$ with the corresponding predictive divergence (e.g. a symmetrized $\mathrm{KL}$, or Bhattacharyya / Hellinger). Three common instance types (acting on the normalized measure):

```text
Spatial disagreement    vision vs tactile at contact position (5/10/20 mm tiers, normalized by Σ)
Force disagreement      tactile reports 2 N normal, F/T-consistent reconstruction needs 8 N
Temporal disagreement   F/T reports contact start at t, tactile at t + Δ (normalized by latency)
```

("vision says rigid · tactile says compliant" is not a contradiction across abstraction layers — one object can be globally rigid + surface compliant.")

Then test four things: (1) **Contradiction detection** — did the model report "there is a statistical cross-modal disagreement here"; (2) **Uncertainty calibration** — is the reported uncertainty proportional to the actual deviation; (3) **Source attribution** — can you later localize which stream went wrong; (4) **Recovery** — how much success rate degrades after handling.

### 8.7 Consistency graph → a quantifiable fault-isolation benchmark

$\mathcal{L}_{consistency}$ is not a scalar, it is a graph:

```text
           Vision
          /      \
      Tactile --- F/T
          \      /
         Proprio
```

Each $(i, j)$ edge is a consistency constraint $\mathcal{C}_{ij} = D_{ij}(z_{ij})$ ($D$ uses §8.6's normalized measure); $\mathcal{C}_{VT}$ on contact position, $\mathcal{C}_{TF}$ on normal force, $\mathcal{C}_{FP}$ on the $\tau_{\mathrm{res}} - J^T F$ residual, $\mathcal{C}_{VP}$ on EE pose. Total loss:

$$
\mathcal{L}_{\text{consistency}} \;=\; \sum_{(i,j)} w_{ij}\, \mathcal{C}_{ij}
$$

Going further, record each edge's residual as $r_{ij} = \mathcal{C}_{ij}$ and rank by edge: **edge residual ranking → candidate faulty modality**, or a more principled $P(\text{fault} = i \mid \{r_{ij}\}_j)$. This upgrades it from "detecting contradictions" into a **genuinely quantifiable fault-isolation benchmark**, giving provenance its first quantitative metric:

```text
Fault isolation metrics
├── fault detection AUROC
├── source attribution accuracy / F1_fault
├── top-1 faulty modality / top-k coverage
├── recovery success (performance rebound after isolate + re-estimate)
└── time-to-diagnosis (delay from injection to alarm)
```

This upgrade makes **Provenance and Validity** in §6.4's 7-tuple formula benchmarkable, and turns §6.2(ii)'s correlated-evidence problem (e.g. proprio / F/T sharing a wrench latent) into a testable object.

### 8.8 Representation bottleneck ablation: is the interface working via "structure" or just "an extra bottleneck"

New in v5, and the key group answering the reviewer's "isn't the interface value just an extra bottleneck" question. Compare four raw→policy paths:

```text
A  raw → fusion → policy                         (no structure, no interface)
B  raw → learned latent → policy                 (implicit representation, no explicit contract)
C  raw → structured slots → learned embedding → policy
                                                 (interface present, embedded then given to policy)
D  raw → structured slots → controller           (interface present, given directly to a non-learned consumer)
```

Compare across six conditions: IID task success, **unseen sensor, unseen mounting, missing modality, contradiction, consumer swap**. If C / D do not lose to A / B on IID but are clearly stronger on unseen / missing / contradiction / consumer swap, then the value truly comes from "a structured contract" rather than a plain bottleneck; conversely, if IID is clearly worse and robustness does not improve, this article's recommendation should be downgraded to "only worth it when multi-consumer reuse is required".

### 8.9 Registration perturbation (relativized, four classes, parameter vs convention reported separately)

Relativize the time perturbation: $\delta t \in \{0.25\,\Delta t,\; 0.5\,\Delta t,\; \Delta t,\; 2\,\Delta t\}$, with $\Delta t$ the task's **time-relevant bandwidth**; do not write specific millisecond numbers, to keep the benchmark transferable. Split frame perturbation into four classes:

```text
Registration robustness
├── calibration noise        ── T̂ = T · ΔT, ΔT a small perturbation (Gaussian / uniform)
├── calibration drift        ── long-horizon slow drift, simulating thermal / mechanical creep
├── sensor remounting        ── tool / camera / sensor-mount swap, geometry redefined
└── frame convention mismatch── base ↔ world, sensor ↔ tool, extrinsic sign flip
```

**The first three are parameter perturbations (continuous $T \rightarrow T\,\Delta T$ perturbations); the fourth, convention violation, is not a continuous perturbation but a semantic / convention failure**, and the downstream consumer's failure mode is completely different. The two classes should be **reported separately**, not merged into one domain-randomization metric, or you conflate the average effect of "wrong parameters" with "wrong conventions".

### 8.10 Fit of existing benchmarks

Mainstream manipulation benchmarks — RoboCasa / LIBERO / ManiSkill3 / BEHAVIOR — **are largely vision-centric, tactile-absent**. This section draws no conclusion but **suggests**: turn the nine metrics above into an optional plug-in adding a "feedback-value + interface quality" lens to existing benchmarks; 9/11 §5.2.1's A/B/C/D four-arm ablation can be borrowed directly.

## 9. Relation to VLA and world models

### 9.1 VLA: what's missing is not a "concat channel" but a cross-sensor-family stable state contract

§3 has already corrected this — lumping RT-2 / OpenVLA / π0 under "concat" is inaccurate, and "supporting input" vs "establishing a contract" must be distinguished. **Taking representative public systems such as RT-2, OpenVLA, π0 as examples, one can observe**: existing general VLA scaling-pretraining ecosystems center on vision-language observation and robot state / action. On the absence of tactile / F/T, this article states it as an **ecosystem observation (a literature-positioning judgment)** rather than a fact-law confirmable or falsifiable by a few papers, and uses as defeasible phrasing as possible:

> **To our knowledge, no publicly established ecosystem currently matches vision-language-action pretraining in scale, sensor diversity, task coverage, and embodiment diversity for tactile / force observations.**

The needed qualification is that this absence is **not "nobody has done it"**: tactile foundation models are emerging (TVL, AnyTouch), visuo-tactile representation learning is already abundant (Lee, Calandra, 3D-ViTac), some robot foundation-model works do add proprio as input, and hybrid state representations are not blank either. What genuinely has not taken shape is **a stable structured state contract across sensor family, across embodiment, and across downstream consumer**; and "some VLA can eat proprio input" is still a whole §6.4 layer away from "it defines a cross-consumer reusable state contract".

The real open question: **is the VLA's next expansion more vision+language, or finally supplying the tactile / F/T state contract?** This piece bets on the latter — but not treating tactile as "just another channel", rather inserting §6's structured state contract into the VLA's input layer. Qi et al.'s T-Dex [arXiv:2309.09979](https://arxiv.org/abs/2309.09979) (CoRL 2023) and Lee et al.'s [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) can be seen as early forms of this route.

### 9.2 World models: not criticizing latent world models, only suggesting contact-rich dynamics keep an identifiable event / mode channel

RSSM / DreamerV3's [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) latent dynamics usually assume relatively smooth, differentiable state transitions. 9/11 §5.1 discussed that contact events are essentially mode switches of hybrid dynamics.

**This article is not criticizing latent world models, nor saying "continuous latent will smooth contact away"** — modern latent dynamics can absolutely use discrete latent / stochastic latent / event latent / hierarchical latent / latent mode switching / multiple heads to carry events and modes. This article's phrasing converges to a **very narrow, and therefore hard-to-refute** suggestion:

> **For contact-rich dynamics, we suggest a world model keep at least one identifiable event / mode channel, rather than forcing it to exist wholly implicitly inside an undifferentiable continuous latent.**

A world model can separately predict $z_{t+1}$, $p(m_{t+1}\mid z_t, a_t)$, $p(C_{t+1}\mid z_t, a_t)$, or use a hybrid latent (`z_continuous + z_discrete + contact state`). This is an architectural recommendation, not a factual conclusion.

This division of labor connects naturally with §6's state contract: **a world model need not consume all slot fields, but it usually warrants having some head explicitly predict the mode / event part of the slots**; the policy layer may use a learned representation (learning an embedding from slots for the policy is reasonable, but that embedding is a "downstream consumer", not an "upstream data format" — see §6.4(b)); the two connect via §6's interface, the world model predicting the next-timestep slots and the policy consuming the current slots.

### 9.3 One-sentence summary

**What VLA lacks is not the tactile channel itself but a stable state contract across sensor family / embodiment / consumer; what world models lack is not continuous latent but, on contact-rich tasks, keeping at least one identifiable event / mode channel. The common root cause of both is the absence, between heterogeneous observations and downstream models, of a structured state contract that can carry value / semantics / frame / time / uncertainty / provenance / validity.**

## 10. Conclusion

This piece set out from the misunderstanding that "multimodal fusion is usually framed as a model-architecture problem" and pulled the discussion back to the foundations. The article's real point goes one step beyond "fusion should add uncertainty": **this is not proposing a new fusion operator, nor a new state estimator, but a contract over estimator outputs** — a **structured state contract** spanning heterogeneous observations and multiple downstream consumers, able to express value / semantics / frame / time / uncertainty / provenance / validity explicitly. It deliberately only specifies "what to expose", leaving "which algorithm to infer with" (Layer 3 — Bayesian / EKF / factor graph / learned filter, all fine) and "which representation downstream consumes with" (symbols or neural embedding) entirely blank.

$$
\boxed{\;\text{Interface} \;\neq\; \text{Inference algorithm} \;\neq\; \text{Latent representation}\;}
$$

Vision carried a whole common data interface not because its encoders are smarter, nor because it "solved registration" (it did not), but because the **task-level representation conventions** it accumulated were numerous enough to press the still-unsolved registration problems down into local engineering hassles, plus a whole set of ecosystem conditions — camera model, calibration conventions, file formats, timeline, GPU, cheap sensors, internet data, annotation ecosystem. Tactile / force / proprioception each still have unconverged effective observation rates, coordinate frames, raw representations, and semantic conventions, so efforts in cross-attention / shared latent easily crash on the five failure modes of §7.

This article does not oppose cross-attention, does not oppose shared latent, **and does not oppose end-to-end learning**; what it opposes is **letting state estimation, cross-modal composition, and control all happen implicitly inside one undiagnosable, non-reusable latent**. The minimum usable route given: take a **structured state contract** as Layer 4, raw→slot mapping in sensor-specific perception (Layers 1–2), estimator freely chosen (Layer 3); the contact set is one instance of this contract under manipulation ($\text{ContactSet}\subset\text{StructuredState}$), slots carry **three geometry classes — point / patch / region**, **continuous covariance kept separate from categorical probability, with the uncertainty representation matching state topology (count is combinatorial, a multimodal hypothesis cannot be crammed into a single Gaussian)**, **retaining an aleatoric / epistemic / calibration split interface**, **track_id is a confidence-bearing hypothesis, not an intrinsic identity**, **unit / frame / reference point enter the schema via `Quantity` and are validated at runtime**, **availability (can I access) and validity (can I trust) split into two axes, staleness being "validity relative to a past timestamp", not invalidity**, **explicitly recording observability / identifiability and correlated evidence (guarding against double counting)**; `wrench`, as an independent aggregate measurement, produces a consistency residual against the `contacts[]` hypothesis, turning Design for disagreement into a mathematical object; the interface has versioning and graceful degradation; **this article does not require all fusion at the interface, only requiring that state shared / diagnosed / reused across consumers be exposed at the interface**; and writes the Interface Property Benchmark (schema / encoder / consumer swap, oracle / estimated / end-to-end controlled baselines, uncertainty-aware disagreement, consistency graph with fault isolation, representation-bottleneck ablation, registration perturbation) into the evaluation as a first-class citizen.

Back to this article's three design principles (they already cover measurement → representation → robustness, no fourth added): **Register before compose · Expose belief at the interface · Design for disagreement**.

Finally, three boxed formulas to pin to the wall, as the anchors of the whole article:

$$
\boxed{\;\text{Sensor-specific observations} \;\rightarrow\; \underbrace{\text{Structured State Contract}}_{\text{stable interface}} \;\rightarrow\; \{\text{Policy, World Model, Controller, Diagnostics}\}\;}
$$

$$
\boxed{\;\text{Good fusion} \;\neq\; \text{agreement}\;}
$$

$$
\boxed{\;\text{A good multimodal system must represent disagreement, not merely resolve it.}\;}
$$

The next piece (9/13) plans to walk one step downstream from the "interface": **dexterous hands and in-hand manipulation** — placing this piece's state contract into the concrete scenes of recent works like T-Dex / DextrAH / LEAP, to see whether it can carry their architectures, and why the cost structure behind the contrast "many robots can mount a hand, few are really doing dexterous" looks the way it does.

## Sources

This article's citations do not chase "piling up sensor papers" but group by **the evidence chains of four main thesis judgments**:

### A · Cross-modal representation and fusion paradigms (supporting §3 / §4)

- Baltrusaitis, Ahuja, Morency, *Multimodal Machine Learning: A Survey and Taxonomy*, TPAMI 2019 · [arXiv:1705.09406](https://arxiv.org/abs/1705.09406) (the classic multimodal-fusion taxonomy · the reference for §3's layered genealogy)
- Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language Sequences*, ACL 2019 · [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) (an early representative explicitly handling "cross-modal temporal misalignment" · §3.2)

### B · The visuo-tactile fusion evidence line (supporting §4 / §7.1 / §9.1)

- Calandra et al., *More Than a Feeling: Learning to Grasp and Regrasp using Vision and Touch*, RA-L 2018 · [arXiv:1805.11085](https://arxiv.org/abs/1805.11085) (an early visuo-tactile regrasp demonstration, one of the first pieces of evidence that "tactile is worth more in long-tail scenes")
- Lee et al., *Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks*, ICRA 2019 · [arXiv:1810.10191](https://arxiv.org/abs/1810.10191) (visuotactile self-supervised representation, an early anchor of §3.4's shared-latent route)
- Huang et al., *3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing*, CoRL 2024 · [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) (that paper's experiments report a significant visuo-tactile-over-vision-only gain · a reference point for §7.1's modality-collapse diagnostic)
- Qi et al., *General In-Hand Object Rotation with Vision and Touch* (T-Dex), CoRL 2023 · [arXiv:2309.09979](https://arxiv.org/abs/2309.09979) (active tactile exploration + visuo-tactile fusion · a concrete form of §9.1's VLA-supplies-tactile claim)

### C · Cross-sensor / cross-modal unified representation (supporting §3.4 / §5.1)

- Zhao et al., *A Touch, Vision, and Language Dataset for Multimodal Alignment* (TVL / Binding Touch to Everything), ICML 2024 · [arXiv:2402.13232](https://arxiv.org/abs/2402.13232) (~44K vision-touch pairs, tactile-VL alignment, the representative of the "tactile CLIP" route)
- Feng et al., *AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors*, 2025 · [arXiv:2502.12191](https://arxiv.org/abs/2502.12191) (unified representation across heterogeneous visuo-tactile sensors · the contrast for §5.1's raw-layer non-unifiability)
- Lin et al., *9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation*, ICRA 2023 · [arXiv:2308.14277](https://arxiv.org/abs/2308.14277) (a concrete form of multimodal tactile; 3D shape reconstruction + 6D force is at the **estimation layer** · §5.1 heterogeneity evidence)

### D · Robustness and modality masking / dropout (supporting §7.5 / §8.4 / §8.5)

- Maiga et al., *MMP: Towards Robust Multi-Modal Learning with Masked Modality Prior Fine-Tuning*, 2024 · [arXiv:2410.03010](https://arxiv.org/abs/2410.03010) (modality masking as a training strategy, a common practice within missing-modality robustness · §7.5 mitigation reference)
- *Robust Multimodal Learning with Missing Modalities via Parameter Projection*, 2023 · [arXiv:2310.03986](https://arxiv.org/abs/2310.03986) (representation alignment under missing modalities · §8.4 robustness-metric reference)

### E · VLA and World Model (supporting §9)

- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, 2024 · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) (VLM backbone + proprio token + noisy action chunk + flow matching · §3 and §9.1 corrected descriptions)
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) (an open-source VLA baseline · public config has multi-camera / depth / proprioceptive state encoding; but "supporting input" ≠ "establishing a cross-consumer state contract")
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023 · [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) (expressing action as text tokens jointly fine-tuned with a VLM · §3 corrected description)
- Hafner et al., *Mastering Diverse Control Tasks through World Models* (DreamerV3), Nature 2023 · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) (the world-model representative of latent dynamics · §9.2 reference for "can keep an event / mode channel, but latent world models are not themselves criticized")

### F · Sim-to-Real / Domain Randomization background (supporting §7.3 / §8.9)

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) (the classic practice of treating frame perturbation as part of domain randomization)

### G · Continuing 9/11 · Contact state and impedance (background)

- Already cited in 9/11 and reused here: Hogan's impedance-control trilogy, Posa-Cantu-Tedrake IJRR 2014 (hybrid contact-mode trajectory optimization), Lee 1810.10191, Qi 2309.09979, Huang 2410.24091, Zhao 2402.13232, Feng 2502.12191. This piece does not re-paste the links; for precise citations see 9/11's Sources directly.

---

> **Related reading**
>
> - [The Hand Robots Don't Have: Tactile and Force Sensing in Embodied AI](/en/articles/2026-09-11-tactile-force-sensing/) — the predecessor of this piece, taking tactile and force control apart separately
> - [Sim-to-Real Methodology](/en/articles/2026-09-10-sim-to-real-methodology/) — §7.3's frame perturbation and §8.9's registration perturbation can borrow its domain-randomization lens
> - [Why Robot Data Is Harder Than LLM Data](/en/articles/2026-09-09-robot-data-scaling/) — §3's shared-latent route's "where does supervision come from" is really a data-scaling problem
> - [VLA and World Models](/en/articles/2026-09-07-vla-world-models/) — §9 is one concrete facet of it: VLA still lacks a stable state contract across sensor family / embodiment / consumer; world models are advised to keep an identifiable event / mode channel on contact-rich tasks
