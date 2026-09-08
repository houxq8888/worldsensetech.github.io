---
title: 'All Eyes, No Fingertips: Why General Robots Still Lack a Sense of Touch'
slug: "2026-09-11-tactile-force-sensing"
date: 2026-09-11
draft: false
categories: ["Embodied AI", "Multimodal Perception"]
tags: ["Embodied AI", "Tactile Sensing", "Force Control", "Impedance Control", "Admittance Control", "Contact-Rich Manipulation", "Active Perception", "VLA", "Sim-to-Real", "Robot Data", "GelSight", "Contact State Estimation"]
description: 'What general-purpose robots really lack is not "one more tactile sensor," but the ability to stably turn heterogeneous contact signals into task-relevant contact states and feed them back into a control loop in real time. Using "the loop" as a through-line, this piece separates sensor / calibration / raw observation / contact perception / state estimation / task-relevant latent / policy / controller / action, sharpens the boundary between weak visual observability and direct tactile observability, reframes force control as four parallel control paradigms instead of a capability ladder, and locates tactile value both in contact-state perception per se and in the last-mile scenarios where tactile most cleanly quantifies engineering ROI.'
toc: true
related_articles:
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-09-robot-data-scaling
  - 2026-08-26-world-model-in-robotics
  - 2026-09-07-vla-world-models
  - 2026-09-03-vla-deep-dive
  - 2026-09-06-embodied-ai-landscape
---

> Picking up from [Sim-to-Real methodology](/en/articles/2026-09-10-sim-to-real-methodology/) and [robot data scaling](/en/articles/2026-09-09-robot-data-scaling/): those two pieces talked about how to budget and intervene on the mismatch between training and evaluation distributions, and what the next unit of data collection should buy us. This one drills down one more level — **the real world is not a purely visual world; a lot of the state that decides success or failure lives on the contact interface. And general-purpose robots are exactly where the "contact → feedback → adjust-action" loop is least mature.**

Even with your eyes closed, you can unscrew a half-finished water bottle, pick one egg out of a basket without crushing it — not because of vision, but because of "feel" in your hand. On many **general-purpose robots**, the opposite holds: the "eye" (the camera) is very advanced, while contact sensing on the "hand" has not yet formed the reusable data, representation, model and hardware-interface ecosystem that vision has. This is not to say robots have no touch at all — wrist-mounted 6-axis force/torque sensors, joint torque sensing, and fingertip tactile arrays are all used today. What is genuinely scarce is *a deploy-at-scale loop of "sense the contact → change the action."* This piece is about the layer in embodied AI that gets overlooked and yet matters most: **tactile sensing and force control**.

## 0. The through-line: treat the contact loop as the frame

Put the whole analytical frame up front — every later section returns to it.

```text
                        WORLD
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
            Vision                  Contact
              │                       │
              ▼                       ▼
        World state            Contact state
              │                       │
              └───────────┬───────────┘
                          ▼
                    State estimator
                          ▼
                 Task-relevant latent
                          ▼
                        Policy
                          ▼
                Motion / Force reference
                          ▼
                 High-frequency controller
                          ▼
                        Robot
                          ▼
                       Contact  ────┐
                          │         │
                    tactile / force │
                          └─────────┘  ↺

     ── a hidden cross-cutting layer beneath the whole chain ──
     Calibration · Synchronization · Registration
     timestamp alignment · coordinate-frame alignment ·
     sensor-to-robot extrinsics · tactile-to-contact-frame registration
```

Along this chain, "the tactile sensor measured something" ≠ "the robot knows what happened" ≠ "the robot changes its action accordingly." These are three different capabilities; whichever link is missing, the loop breaks there. And underneath them all lies an engineering layer that papers routinely ignore — *timestamp alignment across modalities, coordinate-frame alignment, sensor-to-robot extrinsics calibration, tactile-to-contact-frame registration*. Without it, even the prettiest tactile encoder has a hard time stably entering the policy or controller. Everything below is really an argument about **why the segment from raw signal to task-relevant latent has not yet formed a unified stack the way vision has**.

## 1. Why "seeing" is not enough

The visible progress in robotics over the past decade has been in seeing — cameras get sharper, vision models get stronger. So it becomes a default assumption: get the pixels right, and the action will follow. Once contact enters the picture, "seeing" starts to fall short.

Let's tighten the wording first. It is *not* "vision cannot see contact, only touch can" — that framing gets picked apart by robotics / control readers. *The more careful statement is*: **certain contact states are unidentifiable from vision, or can only be inferred through long time series plus physical priors; tactile and force sensing provide more direct local observations**.

Zooming in further, "vision weakly observes contact" is really a bundle of at least 5 different failure modes, and lumping them blurs the point:

- **Occlusion**: the contact patch is often blocked by the object itself or the hand.
- **Insufficient resolution**: local deformation, micro-slip, edge contact points are averaged out at pixel level.
- **Temporal aliasing**: slip, chatter, and impact happen on millisecond scales, well beyond typical camera frame rates.
- **Hidden internal force / stress**: normal force, shear, and internal stress do not live on the imaged surface.
- **Friction & material state**: wetness, oil, viscoelasticity, roughness — many only show up when you press.

So the more accurate claim is not "vision cannot measure force," but: **under occlusion, low resolution, reflective / transparent surfaces, or fast motion, vision-only estimates of contact force, slip, and contact state are more indirect and less robust**. Vision *can* estimate force indirectly through deformation, optical flow, object displacement, marker tracking, inverse dynamics — just with a longer inference chain, more assumptions, and less robustness. A few cases where tactile/force is decisive:

- **Picking up an egg.** How much grip force an egg can survive is very hard to read from appearance alone. What actually works is a small loop: vision locates the egg → establish initial contact → tactile / force read out contact area and slip → the controller nudges grip force → check whether slip has stopped. **The point is not "I sensed it" but "I can change the next action based on the sensation."**
- **Inserting a USB or a peg.** Not "vision is useless" — vision brings the end-effector into the right neighborhood, and then the task enters a very typical **contact-rich manipulation** regime: once contact happens, the environment's constraint on motion suddenly strengthens, and contact force / torque telling the robot "what I am pressing against, did it slide in" is more direct than watching pixels.
- **Twisting off a bottle cap.** How tight is "tight enough"? If it slips, should I press harder? These are questions the contact interface answers.
- **Taking one sheet from a stack.** Grabbing too many, dragging neighbors along, the sheet slipping — many of these failures barely register in a single frame but are obvious in the hand.

> *In many contact-rich tasks*, vision brings the robot to the neighborhood of contact, while tactile and force keep providing local observations after contact. This is not "vision retires after contact" — it is "vision's signal density and identifiability drop." For actions, fine manipulation in particular, the second half is often what decides success.

## 2. What the robot's "senses" are missing

### 2.1 Disentangle tactile / force / proprioception / force control

These terms get conflated easily, but strictly they are different layers. *On the sensing side*: **vision** tells you external world state; **tactile** tells you local contact state — contact location, pressure distribution, shear, slip, texture; **force / torque sensing** tells you how much external wrench the robot as a whole is receiving; **proprioception** tells you where the robot's own joints and end-effector are and how they are moving. *On the acting side*: **force control** is about actively regulating behavior using those feedbacks.

An arm with a wrist 6-axis F/T sensor directly gets "net force and moment on the end-effector." A high-resolution fingertip tactile array directly gets "contact location and pressure distribution in space." They are *not equivalent*, and they are *not mutually exclusive* — the former is wrench-level observation, the latter is field-level observation. For multi-contact points, local slip, or contact geometry reconstruction, field-level is usually richer; for overall load monitoring or collision detection, wrist F/T is often more stable, cheaper, and more mature.

So the chain this piece cares about is:

```text
Sensor
  ↓
Calibration · Synchronization · Registration
                           (timestamp alignment, coordinate-frame alignment, extrinsics)
  ↓
Raw observation            (pressure array / RGB / 6-axis wrench / joint current)
  ↓
Contact perception         (where contact is, area, normal/shear decomposition)
  ↓
State estimation           (contact geometry, slip velocity, stiffness, friction, contact mode)
  ↓
Task-relevant latent state (is the grasp stable, is the insert done, is the cap tight)
  ↓
Policy / controller
  ↓
Action
  ↓
New contact                ↺
```

**At least 7 layers of engineering / learning between "signal" and "meaning."** Treating tactile as a *sensor* is not enough; treating it as the full *signal-to-latent chain* is closer to engineering reality.

The human palm is one of the densest tactile surfaces in the body: pressure, texture, temperature, slip, and proprioception all work together. In comparison, most robot hands read a much sparser signal:

| Sensory dimension | Human hand | Typical robot |
|---|---|---|
| Position / pose | Yes (proprioception) | Given by joint encoders + motor current + torque + kinematics combined, **often more precise than humans for its own pose** |
| Joint torque | Not read directly; felt indirectly via muscle spindles, Golgi tendon organs, etc. | Readable on some platforms (joint current or dedicated torque sensors) |
| Contact pressure distribution | Very dense | Mostly absent or very coarse |
| Contact texture / local material properties | Press → feel resistance → feel rebound → feel surface friction, **acquired dynamically during contact** | Vision recognizes lots of "looks-like" material; what is missing is that **local, dynamic** material information during contact (usually **jointly inferred** from vision / force / vibration / current / motion response) |
| Slip / friction | Detected instantly | Detectable (tactile arrays, shear force, contact-point motion, vision) — hard to **estimate stably, at low latency, generalize across sensors, and feed into control** |
| Temperature | Yes | Essentially no |

One thing to stress: **human tactile and robot sensors do not map one-to-one.** A robot without a tactile array is not "totally unable to feel" — with end-effector wrench, joint current, visual deformation, vibration and other channels, a lot can be inferred. The real gap is not "do we have sensors" but:

> What robots lack is **high-density, multimodal, low-latency, broad-coverage contact sensing that also gives local contact information — plus the estimation chain that stably turns that sensing into contact state**. Locality is the dimension most easily overlooked and yet most critical — take "two fingertips together lifting a cup": *a wrist 6-axis F/T sensor directly gives net wrench at the end-effector*; *a fingertip tactile array additionally gives contact location and pressure distribution*. For multi-contact, local slip, and contact geometry reconstruction, the latter is usually richer. But be careful: *given kinematics, contact geometry hypotheses, dynamics, and actuator states, wrist F/T can also do contact localization / wrench decomposition / external force estimation* — the well-posedness and scalability of that inversion are just far weaker than reading the field directly. **The two kinds of information are not equivalent along the axis of "can this directly drive downstream decisions."**
>
> The way a robot hand currently looks, it is more like a hand wrapped in a thick glove with a few sparse receptors at isolated points: it can feel "I touched something," but hardly "where, how, is it right."

### 2.2 A harder technical takeaway

If §2 has to be compressed into a single quotable claim:

> **What robots lack is not tactile signals per se, but the ability to stably turn heterogeneous contact signals (pressure field / RGB / shear / vibration / torque / current) into task-relevant contact states, and to feed those states into a control loop in real time.**

This sentence demotes "tactile = another sensor" and promotes "the whole perception–estimation–control chain." It also explains why "just bolt a GelSight onto the fingers and we're done" is not true.

## 3. Why this has been hard for so long

### 3.1 The sensor itself: not one design space, but several

To sense contact, the tactile layer has to be thin, wear-resistant, and survive repeated pressing without breaking. But what actually gives engineers trouble is not "can we build one" — it is *that these goals conflict with each other*.

*One caveat up front*: vision-based tactile (GelSight / 9DTact — the *camera + elastomer + illumination + reconstruction* family), capacitive / resistive tactile arrays, piezoresistive skins, magnetic-flux schemes, wrist 6-axis F/T — they face *completely different* engineering constraints. The seven dimensions below only live in a shared trade-off space at the level of *cross-modality system design*; *inside a single concrete sensor design, typically only two or three of those walls are actually hit*.

At the cross-modality level, at least 7 dimensions are in tension:

```text
Spatial resolution  ↔  Force range
        ↔  Bandwidth  ↔  Latency
        ↔  Durability ↔  Calibration stability
        ↔  Mechanical compliance
        ↔  Cost
```

Directional trade-offs at this level:

- **Higher spatial resolution, higher dynamic bandwidth, and larger force range are usually not simultaneously achievable; high-rate sampling also amplifies noise, bandwidth, data-throughput, and long-term-stability constraints.**
- Soft sensing skins conform well to curved grasps, but do not always survive high payloads or industrial cycle times.
- Mounting a sensor on the fingertip changes the fingertip geometry — **adding tactile itself changes the manipulation dynamics**.
- Designs that get closer to human-hand density / softness are harder to industrialize: accurate, stable, repeatable, and without degrading the hand's original grasping performance.

A classic review worth citing: Dahiya et al., [Tactile Sensing—From Humans to Humanoids](https://ieeexplore.ieee.org/document/5339133), IEEE Transactions on Robotics, 2010. It still reads current fifteen years later — which itself is a signal of how slow this layer moves.

### 3.2 Vision scales for more than "RGB is a common representation" — tactile data is *action-conditioned observation*

To be strict, tactile data is not a blank space. There is already a body of tactile datasets, visuo-tactile datasets, tactile pretraining efforts, and early "tactile foundation model" explorations; sensors like GelSight, [GelSlim](https://github.com/Antaoyu/GelSlim_4Gen_Curvature-Based_Fingertip_Sensor), [TacTip](https://www.tacpix.com/products), and [9DTact](https://arxiv.org/abs/2308.14277) are widely used. On cross-modal alignment, [TVL / Binding Touch to Everything](https://arxiv.org/abs/2402.13232) (Zhao et al., ICML 2024) is one of the current must-read "tactile foundation model" nodes. On the manipulation side, [3D-ViTac](https://arxiv.org/abs/2410.24091) (Huang et al., CoRL 2024) gives a concrete experimental argument for "vision + tactile > vision-only" on fine-grained insertion.

**But the real problem is not "no tactile data" — it is *that scale, cross-sensor comparability, task coverage, and cross-embodiment transferability are all nowhere near what vision has*.**

Grant the intuition first — "pixels / RGB / video as a common representation absorbs differences" — it is right but *not sufficient*. Vision scales because **multiple conditions matured simultaneously**:

- **cheap, universal sensors** (a whole CMOS imaging supply chain);
- **standardized image / video formats and color conventions**;
- **internet-scale data already lying around** (nobody had to specifically collect it);
- **weak supervision / auto-labeling is easy** (alt-text, clicks, social signals);
- **spatial locality & reusable CNN / ViT architectures**;
- **a mature benchmark ecosystem** (ImageNet / COCO / Kinetics / LAION …);
- **pretrained weights transfer across tasks**.

Tactile barely has any one of those — but the deepest *structural* difference is this:

> **Vision data is largely passive observation, whereas tactile data is intrinsically action-conditioned observation.**

```text
Vision：   world  ─────►  observation

Tactile：  action  ──►  contact  ──►  observation
                          ▲              │
                          └── next action ◄┘
```

A camera can be "put there and it just collects data," whereas *tactile data only comes into existence once the robot actively makes contact*. So a tactile dataset is never just an $X$; it looks more like:

$$\mathcal{D} \;=\; \big\{\,\big(s_t,\ a_t,\ c_t,\ o^{\mathrm{tac}}_t,\ y_t\big)\,\big\}_t$$

where $c_t$ is the contact state, $o^{\mathrm{tac}}_t$ is the tactile observation, and $y_t$ is the task outcome. Worse — *the data-collection policy $\pi(a\mid o)$ itself shapes the distribution*: change the policy, and the observed contact distribution changes. This is exactly the *interaction distribution* argument from [robot data scaling](/en/articles/2026-09-09-robot-data-scaling/): *robots are not just short on "action data," they are short on action data with contact state attached and generated by a sensible collection policy*.

Following this action-conditioned thread, "tactile data heterogeneity" splits further into three axes:

- **Sensor heterogeneity**: pressure / RGB deformation / shear / vibration / wrench / current are different physical quantities, not naturally comparable;
- **Embodiment heterogeneity**: different hands, finger geometry, materials, actuators, kinematics, contact surfaces;
- **Interaction-policy heterogeneity**: *where you touch, how you touch, with what force and at what speed* itself determines *what tactile information you get*.

The third axis is what this section most wants to stress and what is most often overlooked — **it is an independent dimension beyond the first two, and it directly decides whether a tactile dataset can be reused at all**.

### 3.3 Sim-to-Real: what has to match is the contact mode, not every physical parameter

Light and collisions are relatively easy in simulation. But "contact — deformation — friction — slip" is a much harder physics stack: friction coefficients, contact stiffness, material viscoelasticity, micro-surface structure, contact area evolution, sensor noise, soft-material / skin coupling… every link can inject error.

Time to narrow the claim. It is not that "tactile Sim-to-Real is harder than visual Sim-to-Real" as a blanket — visual Sim-to-Real has plenty of unsolved problems too (lighting, material, rendering gap). More precisely: **contact-rich tasks' Sim-to-Real tends to be far more sensitive to contact dynamics, friction, material, sensor geometry, and contact geometry, so their domain gap is harder to close with simple visual domain randomization alone**.

Adding one more technical layer: for most manipulation policies, *what actually has to be right in the simulator is not every physical parameter, but the contact mode* the policy conditions on, plus its transitions and failure boundaries:

```text
free space
   ↓ touch
contact (sticking)
   ↓ shear force ↑
sliding  ⇄  sticking
   ↓ rolling
rolling
   ↓ release
separation
```

Whether $\mu = 0.42$ or $\mu = 0.38$ — many policies are not that sensitive. But *"am I in sticking or sliding right now," "when does contact separate"* — those *mode boundaries*, once the simulator misjudges them, break the policy instantly. So a sensible sim-to-real calibration target is usually not "estimate every quantity accurately" but *push the contact-mode transition boundaries into the range the policy can tolerate* — this ties straight into the "State estimation" link in §2.1 and into the policy-conditioned mismatch framing of [Sim-to-Real methodology](/en/articles/2026-09-10-sim-to-real-methodology/): **as long as the reality gap falls into a direction the policy is insensitive to, it is not a problem**.

Also be careful not to over-swing: *"sim is inaccurate" does not mean "everything must come from real data."* What engineering actually does is treat sim as a calibrable approximation — and a **loop**, not a one-shot pipeline:

```text
Simulation
    ↓
Policy
    ↓
Real robot
    ↓
Failure / residual
    ↓
Parameter estimation (system identification)
    ↓
Simulator calibration
    ↓
Retrain
    ↺
```

Combined with domain randomization ([Tobin et al., 2017](https://arxiv.org/abs/1703.06907)), real-world fine-tuning, and residual learning / action-transformation-style methods to close the residual. **Sim remains a powerful tool for training and verification**; the difference is that tactile / contact tasks need this calibration loop to run more "manually" and more iteratively than visual tasks do — essentially the same *data engine / feedback loop* argument from [Sim-to-Real methodology](/en/articles/2026-09-10-sim-to-real-methodology/).

### 3.4 Standardization: not one hdf5 schema — five layers of infrastructure

Vision's real advantage is not "everyone has the same camera" — it is that **the base representation of images is highly unified**, and around it an entire shared interface ecosystem has formed. Tactile is far messier: different sensors output pressure arrays, 3D geometric deformation, RGB images, shear forces, normal forces, 6-axis F/T, resistance / capacitance changes, high-frequency vibration signals… **even the same word "pressure" is not directly comparable across sensors.**

Splitting "missing standardization" one level deeper makes it concrete:

| Layer | Vision (has a relatively mature shared interface and benchmark ecosystem) | Tactile (current state) |
|---|---|---|
| **Hardware** | Camera interfaces, lens specs, intrinsics / extrinsics conventions | Sensor geometry, mounting, coordinate frames, sampling rates, latency, calibration protocol — everyone does their own |
| **Data** | Image / video formats, timestamps, metadata, color spaces | Raw representation, multimodal time sync, contact labels, cross-dataset metadata all non-uniform |
| **Representation** | pixels → CNN/ViT features (reused across tasks) | Contact location, normal/shear decomposition, slip, deformation, material — *these intermediate representations themselves have no accepted convention* |
| **Task** | Detection / segmentation / depth as standard tasks | Grasp stability, insertion completion, contact mode, failure state — definitions scattered |
| **Benchmark** | COCO / ImageNet / Waymo / nuScenes | Cross-sensor transfer, cross-hand transfer, sim-to-real, closed-loop success, recovery rate — each paper its own |

One caveat: **vision has not "finished" standardization either** — camera intrinsics / extrinsics / color calibration, depth-sensor heterogeneity, event cameras, multispectral, rolling shutter, varying frame rates are all still open problems. When this piece says vision is mature, *it means "a relatively mature shared interface plus benchmark ecosystem"*, not "standardization has been solved." Do not conflate the two.

Read this as a table and "the tactile ecosystem is immature" stops being a slogan and becomes "every layer is still missing something."

### 3.5 Tactile is bad at "saying what it felt" — because there is no intermediate representation chain

Do not jump to "tactile has no semantics." On the contrary, **tactile can carry very rich semantics** — material, texture, hollowness, deformation, whether the grasp is stable, all of these can be judged through touch. The real problem is: *between image and high-level semantics, there is already a very mature middle representation* (pixels → features → objects / scenes); *between raw tactile signal and high-level semantics, we still lack a widely reused, cross-sensor representation path*.

That missing chain is exactly the core of §2.1's 5-layer stack:

```text
raw tactile signal  →  contact perception  →  contact state estimation
                   →  task-relevant latent state  →  action
```

A camera naturally gives you a 2D spatial structure: here is a cup, there is a hand, this is red, that is the table — the representation is already close to semantics. Tactile, on the other hand, typically hands you a pressure blob, a time series, a shear-force transition — semantics are not absent, but the robot has to learn on its own: "this is not a blob of pressure change, this is **an object currently slipping out from between my fingertips**."

A research direction worth watching is turning this chain into a "tactile foundation model." TVL / AnyTouch / Binding Touch are early work in this line (aligning tactile into the vision-language representation space); we do not yet have a "tactile CLIP"-level common base, let alone a "tactile ImageNet"-level training corpus. *Here "tactile CLIP" is not about replicating CLIP's model architecture — it is about a public representation space that is cross-sensor, cross-embodiment, cross-task reusable*. What is really missing is a *representation standard* — the whole "raw signal → contact state → action-conditioned representation" convention — not necessarily a particular foundation-model architecture. This distinction matters; do not confuse "train a big model" with "produce a unified representation."

> **My read (conditional, not absolute)**: if the goal is "build a cross-hardware, scale-trained general tactile capability," then *data scalability, cross-sensor representation, and the intermediate representation from raw signal to task-relevant latent* are among the most foundational bottlenecks. Of course, different research routes hit different first walls: *hardware folks first hit durability and consistency, whole-robot folks first hit the hand itself, control folks first hit sensor bandwidth and actuation delay, learning folks first hit data volume and annotation cost*. Without defining "cross-hardware, scalable," these five difficulties have no objective ordering; with that definition, the "data + intermediate representation + ecosystem" bundle is what decides whether others can build on top of you.

## 4. Force control: four parallel control paradigms, not a capability ladder

Sensing is only the first step. The harder step is making the robot **adjust its action based on force**. There is a basic taxonomy worth spelling out — *these are not four rungs on a ladder from primitive to advanced; they are four parallel control paradigms*:

```text
                     ┌─ Position control     ：specifies "where the end-effector goes"
                     │
Task command ────────┼─ Force control        ：specifies "how much force / wrench"
                     │
                     ├─ Impedance control    ：achieve a desired force–motion dynamic
                     │                        relationship via the controller
                     │                        F = Mẍ + Bẋ + K(x − x_d)
                     │
                     └─ Admittance control   ：solve for desired motion from measured force
                                              ẍ = M⁻¹(F − Bẋ − K(x − x_d))
                                              then send it as position / velocity reference
```

- **Position control**: command "move the hand to this coordinate." Stiff, precise, but the moment something unexpected resists, it "argues" with the environment and can crush things.
- **Force control**: command "apply this much force." Suits tasks dominated by contact force, like "move along a surface."
- **Impedance control**: make the end-effector *exhibit a desired dynamic relationship* — not simply "input displacement, output force." The controller closes $F = M\ddot{x} + B\dot{x} + Kx$ actively, so the robot responds to disturbances with the specified mass–damping–stiffness behavior.
- **Admittance control**: measure the external force first, solve for a desired motion, and dispatch it as a position / velocity reference to a high-stiffness position / velocity controller.

One technical caveat to avoid pushback from control readers: *strictly speaking, both impedance and admittance realize a kind of "desired force–motion dynamic relationship" — the practical difference is closer to "which side is the input, which side is the measurement"*. Impedance takes motion as input, produces force, and usually needs good torque-control bandwidth; admittance takes force as input, produces motion, and is more often paired with a stiff position loop. In industrial practice the two are mixed, with the choice determined by sensor and actuator bandwidth.

The popular "make the hand soft" metaphor needs a caveat: **"soft" is not simply lowering stiffness** — it is making the robot produce **controlled compliance** in response to external disturbances, along a set force–displacement relation. Push stiffness too low and the robot becomes floppy, precision is gone. Tuning "feel" is about shaping the spring–damper curve, not switching it off. So many assembly, insertion, wiping, and fine-manipulation tasks with high contact uncertainty really require **some form of active or passive compliance** — not all tasks must be solved with active force control; high-precision positioning, dedicated fixtures, compliant mechanisms, passive compliance, and pre-defined trajectories are all common engineering answers. **Force control ≠ "the robot learned to control force by itself"** — real systems usually combine *mechanical compliance + impedance control + force feedback + policy learning*.

Beyond that, a more ambitious direction is to place force / tactile alongside vision, language, and proprioception **inside the same policy and learning framework**: not "just another sensor channel," but **adding a high-value observation about contact state** to the policy. This direction is right, and there is a body of work showing it works (e.g., [3D-ViTac](https://arxiv.org/abs/2410.24091)); what is still unsolved is **the lack of a scaled, unified, cross-hardware generalizable paradigm comparable to vision-language models**. So the more accurate statement is not "does visuo-tactile learning work," but "it works, but it is still far from unified."

### 4.1 A key time-scale fact: VLA ≠ controller

This is exactly the seam between AI readers and robotics-control readers, and where many popular AI articles slip. Plugging tactile into a VLA **does not mean the VLA directly outputs motor current or torque**. A more typical architecture looks like this:

```text
Vision / Language / Tactile / Proprioception
                    ↓
              State / Policy  (low rate, tens to hundreds of ms)
                    ↓
       ┌────────────┴────────────┐
       ↓                         ↓
 task-space command        impedance / force target
 (pose / velocity)         (stiffness / damping)
       └────────────┬────────────┘
                    ↓
       High-frequency controller (much higher rate)
                    ↓
                  Motor
                    ↓
                 Contact
                    ↓
             tactile / force
                    ↺
```

One physical fact worth calling out as its own line: **the time scale of the VLA and the time scale of contact control are not the same thing.** High-level policies typically update at low rates; low-level controllers close the loop at much higher rates. *In typical systems, the two differ by one or more orders of magnitude, with the exact gap depending on policy architecture, action chunking, inference accelerator, motor controller, and impedance loop.* That middle controller layer is the bridge that turns "decision" into "motor." This is also why, however large the VLA is, it does not automatically become a controller that survives contact disturbances. **"Tactile enters the VLA" ≠ "the VLA is the controller."**

## 5. The real value is the loop

At this point I can put a more essential claim on the table: **the value of tactile is not "sense more" — it is enabling the robot to change its action in real time based on contact outcomes.**

One thing to be careful about: it is tempting to say "tactile is the first thing that closed the loop in manipulation." Strictly, that does not hold. *Visual servoing* is itself a closed loop (see the classic tutorial by Hutchinson, Hager, and Corke, *A tutorial on visual servo control*, IEEE Transactions on Robotics and Automation, 1996 · [Semantic Scholar](https://www.semanticscholar.org/paper/A-tutorial-on-visual-servo-control-Hutchinson-Hager/4676d81d6a2477f6ea1de5723fc81e431fbaa96f)); many robot systems have long been running "see → estimate state → act → see again." What tactile really adds is not "opening a loop for the first time," but:

> **Extending the visual-dominant closed loop to cover the local "after contact" states that vision cannot reliably observe.**
>
> In other words, *the meaning of tactile is not that the robot has one more sensor bolted on — it is that once contact happens, the robot knows what happened and can change the next action accordingly*. This is the one sentence this whole piece wants to leave you with.

```text
Vision-only · a visual closed loop:
       see ──► estimate ──► act ──┐
        ▲                         │
        └───── look again ◄───────┘
       (once contact starts and is
         occluded, feedback breaks)

With contact feedback · an extended loop:
       see ──► act ──► contact
              ▲          │
              └ tactile/force · correction ◄┘
```

This is the essential difference between "can see and can feel" and "can only see." Much of the distribution shift, error accumulation, and recovery failure discussed in [Sim-to-Real methodology](/en/articles/2026-09-10-sim-to-real-methodology/) are, in the last centimeter, exactly what **"the loop is missing a contact feedback leg"** looks like.

One more convergence step: **entering the loop ≠ benefiting from the loop**. Even if a tactile signal is technically inside the observation space, if it suffers from *high latency, heavy noise, drift, poor identifiability, or weak localization*, downstream policies may not learn how to use it, and the system gain is near zero. So the harder statement is:

> **Whether tactile has value does not depend on whether it enters the observation space, but on whether it delivers sufficiently timely, identifiable, task-relevant information that lets the closed-loop policy change its action and improve system stability.**

This ties §3.1's trade-offs, §4.1's time-scale fact, and §6's ROI framing together — *"wired into the loop" is necessary, "wired in well" is sufficient*.

### 5.1 Active perception: tactile is not only about "avoiding failure" — it changes how information is acquired

Tactile is not only for "avoiding failure" — it is also for **exploration**. When you cannot see inside a cup, you reach in and feel; when you do not know how heavy something is, you lift it and estimate; when unsure whether a part is slipping, you nudge it lightly; when unsure how soft or stiff a material is, you press and rub. In robotics this has a broader name: **active perception / active sensing**.

Its scope is broader than "active tactile" — *moving the camera, changing viewpoint, walking around the object, reaching in, pushing* are all active perception; **tactile is one very important realization of it**. The core is: *the robot does not just passively sense the world through sensors; it actively acts to earn the information it wants.*

Push this one step further and we get a harder judgment: *tactile is not just another observation modality — it changes the mechanism of "how information is acquired" itself*:

```text
Vision   ：  observe  ─────►  act

Tactile  ：  act  ──►  contact  ──►  observe  ──►  act
```

In other words, *on the tactile side, action is itself a sensing operation* — exploration and manipulation start to couple, and the optimal policy no longer optimizes task reward alone but also **information gain**. This lands naturally in the classical robotics framing: **POMDP / active sensing / information gathering**, which is exactly about "when state is only partially observable, how to use actions to buy useful observations before deciding." Tactile tasks are almost a canonical POMDP: contact state is hidden until you touch, and only action can shrink the belief.

This is also where §3.2's "action-conditioned observation" argument finally closes the circle: *because the data itself is shaped by action, the collection policy and the execution policy are necessarily coupled* — a structural difference between tactile and visual datasets, not a "just add more data" issue.

A research direction worth watching: *jointly optimizing active tactile, VLA, and world models inside the same policy* — VLA maps vision + language to action; the next step is plausibly "vision + language + tactile + proprioception → contact state understanding → action closed loop." *This is a predictive judgment, not a technical fact*, but work like TVL / 3D-ViTac / AnyTouch is already doing early alignment.

### 5.2 How to actually measure the loop: an evaluation-metric layer

Every "loop" claim above has to be measurable to stand up. Here is a cross-layer metric table — *it is also the mirror of the §3.4 standardization table*: if standardization is missing at every layer, evaluation metrics are equally missing at every layer.

| Layer | Key metrics |
|---|---|
| **Sensor** | spatial resolution · force range · bandwidth · latency · drift · calibration stability · cost |
| **Contact perception** | contact detection accuracy · contact localization error · normal / shear decomposition error |
| **State estimation** | slip-detection AUROC · contact-mode classification · force / pose / stiffness estimation error |
| **Policy** | task success rate · sample efficiency · **failure recovery rate** · **time-to-recovery** |
| **Control** | wrench / pose tracking error · contact-force tracking error · overshoot on transitions |
| **System** | cycle time · long-run success rate · **contact-induced failure rate** · **success under perturbation** |
| **Generalization** | **cross-sensor transfer** · cross-hand transfer · cross-object · **performance degradation under sensor shift** |

*The bolded metrics are the ones most worth adopting as mainstream evaluation practice* — they map directly onto §6's "tactile value is in the long tail / conditional benefit" argument and are the ones that actually answer "did a loop form?" Task success rate alone averages out the stability benefits tactile is supposed to bring, and hides exactly the part §6 is trying to surface.

## 6. Where tactile most cleanly quantifies engineering ROI: the Last-Mile long tail

One caveat first: *tactile's value is not confined to long-tail / recovery*. Tactile is a first-class perception channel and directly provides:

- **Object identity / material cues** (softness, hollowness, texture)
- **Grasp state** (contact area, force distribution, stability)
- **Contact geometry** (normal vs shear decomposition, contact location)
- **Pose refinement** (in-hand reorientation to target pose)
- **Texture recognition** (surface roughness, patterns)
- **Active exploration** (reach in when vision is not enough)

All of that stands on its own as "tactile sensing value," independent of any closed loop.

**Tasks that do not really need tactile** — drawing this boundary is more useful than another paragraph on "tactile is important":

- Large-space visual localization (AGV / AMR running SLAM in a structured warehouse);
- Coarse pick-and-place in structured environments (front-facing placement, loose tolerance, single material);
- Open-space navigation (avoiding people and obstacles, no fine contact);
- Operations that never touch the object (scanning, printing, demo, pure display);
- Assembly already covered by high-precision fixtures / jigs / visual servoing.

So the sentence that should actually be written down is:

> **Tactile value scales with *contact richness*, not with "task complexity." Complex task ≠ must need tactile.**

A more operational engineering proxy:

$$\text{Tactile ROI} \;\propto\; \underbrace{\text{contact uncertainty}}_{\text{how hard the contact state is to predict}} \;\times\; \underbrace{\text{contact sensitivity}}_{\text{how sensitive performance is to it}} \;\times\; \underbrace{\text{failure cost}}_{\text{how expensive one contact failure is}}$$

*The larger this product, the higher the tactile ROI.* Fine assembly, fragile objects, deformable objects, dirty / wet / friction-variable scenes all sit in the high-value region. Conversely, even very "hard" tasks in other dimensions (long-horizon navigation, complex decision trees) can get very little benefit from tactile as long as contact uncertainty is low.

Back to Last Mile: *it is the scenario where tactile ROI is easiest to quantify.* Previous pieces ([Sim-to-Real methodology](/en/articles/2026-09-10-sim-to-real-methodology/), [robot data scaling](/en/articles/2026-09-09-robot-data-scaling/)) kept saying one thing: real deployment is about the system. The value of tactile often **does not show up as "success rate goes from 80% to 90%"** — it shows up in the **long tail of contact states** that vision cannot handle: slightly tilted, slightly slipping, jammed, contact point shifted, friction suddenly changed, tolerance pushed to the edge, the object being squeezed out of shape. These are exactly the last-mile scenarios where "each event is low probability, together they add up."

Another point, closer to engineering intuition: **tactile's benefit is often conditional**. Without tactile you might still get 95% success under normal conditions; with tactile it becomes 99%. Looked at only through demos, both robots pick the object up, and the difference is not dramatic. The real fork appears when: the object is a bit wet, the grasp position is a bit off, the friction coefficient drifted, the object starts deforming, the contact point is quietly sliding, a micro-slip has begun but the object has not dropped yet… *tactile may not make the demo more impressive, but it very plausibly shortens the failure tail* — which lands right on the core claim of [Sim-to-Real methodology](/en/articles/2026-09-10-sim-to-real-methodology/): **whether a system can be trusted is never judged by its highlight moments, but by those "individually low-probability, collectively non-trivial" moments.**

> Vision models handle the bulk of ordinary cases in "understanding the world"; tactile and force control both provide independent sensing value on the "contact state" axis *and* deliver the clearest ROI on "local anomalies after contact." What is genuinely hard for a robot is precisely those **low-probability, combinatorially complex, hard-to-enumerate-in-advance** contact states.

## 7. A harder open question

If I am allowed to compress this whole piece into one open question, it is:

> **Why, to this day, has tactile still not formed a unified technology stack that can, across sensors, across embodiments, and across tasks, stably map action-conditioned contact observations into generalizable contact states and finally into a real-time control loop?**

Once this question stands up, what this piece is really answering is no longer "**why do robots need tactile**," but a plainly harder one: *why tactile robotics has not yet had its own "visual moment."*

This decomposes directly into the four segments of the §0 diagram:

1. **Sensor → Calibration → Raw observation**: the hardware trade-off space is too wide, no "CMOS image sensor" equivalent exists as a universal form factor; cross-modal timestamp alignment, coordinate-frame registration, sensor-to-robot calibration *still lack any reusable public infrastructure* (§3.1, §2.1).
2. **Raw observation → Task-relevant latent state**: no widely reused intermediate-representation chain, so every team is reinventing contact perception and state estimation (§3.5).
3. **Task-relevant latent → Control loop**: the time scales of VLA and contact control are not yet connected; the three segments do not share benchmarks or evaluation metrics (§4.1, §3.4, §5.2).
4. **The data-acquisition policy itself is the problem**: tactile data is action-conditioned observation, so the collection policy and the execution policy are naturally coupled — you cannot just "leave a camera out there" and accumulate data (§3.2, §5.1).

*Answering those four clearly moves this article one more step from "tactile popular-science" toward "tactile technical review."*

## Summary: a four-layer architecture

Back to the §0 diagram, compressed into a four-layer spine:

```text
vision             →  world understanding (world state)
tactile / force    →  contact understanding (contact state)
force control      →  contact closed loop (constrained contact behavior)
policy learning    →  turn those local feedbacks into generalizable behavior
```

None of these four layers means "the robot is useless without it" — visual picking, visual localization, open-space navigation, many non-contact operations, all run well without tactile. But for **contact-rich, fine-manipulation, contact-state-heavy tasks** — assembly, insertion, wiping, deformable-object manipulation — missing any of these four layers is a very plausible bottleneck between "the robot can do it" and "the robot reliably does it well." **Getting from "can see" to "can work" — the missing link is usually not how big the perception model is, but whether the contact → feedback → adjust loop has been seriously built.**

So the next genuinely exciting progress in robotics may not be "seeing even better," but — the day a robot finally builds that "sense contact → adjust action" loop end to end.

---

**Further Reading / Sources**

*Tactile sensing & surveys*

- Dahiya, Meta, Schmitt, Cowley, *Tactile Sensing—From Humans to Humanoids*, IEEE Transactions on Robotics, 2010 (classic review) · <https://ieeexplore.ieee.org/document/5339133>
- GelSight high-resolution vision-based tactile sensor · <https://gelsight.com/>
- GelSlim open-source fingertip tactile sensor · <https://github.com/Antaoyu/GelSlim_4Gen_Curvature-Based_Fingertip_Sensor>
- TacTip 3D-printable tactile fingertip · <https://www.tacpix.com/products>
- Lin et al., *9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation*, ICRA 2023 · [arXiv:2308.14277](https://arxiv.org/abs/2308.14277)

*Visuotactile learning & tactile representation / foundation models*

- Huang et al., *3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing*, CoRL 2024 · [arXiv:2410.24091](https://arxiv.org/abs/2410.24091)
- Zhao et al., *A Touch, Vision, and Language Dataset for Multimodal Alignment* (TVL / Binding Touch to Everything), ICML 2024 · [arXiv:2402.13232](https://arxiv.org/abs/2402.13232)
- Feng et al., *AnyTouch: Learning Unified Static-Dynamic Representation across Multimodal Tactile Sensors*, 2025 (cross-sensor unified representation, directly relevant to §3.5's "representation standard") · [arXiv:2502.12191](https://arxiv.org/abs/2502.12191)

*Contact-rich control (classical)*

- Hogan, *Impedance Control: An Approach to Manipulation* (Parts I–III), ASME Journal of Dynamic Systems, Measurement, and Control, 1985 · [Semantic Scholar](https://www.semanticscholar.org/paper/Impedance-Control%3A-An-Approach-to-Manipulation-Hogan/f5f8a6aa4adc13070224c2bd43a255c4e0844c15)
- Hutchinson, Hager, Corke, *A tutorial on visual servo control*, IEEE Transactions on Robotics and Automation, 1996 (visual servoing as a canonical closed loop) · [Semantic Scholar](https://www.semanticscholar.org/paper/A-tutorial-on-visual-servo-control-Hutchinson-Hager/4676d81d6a2477f6ea1de5723fc81e431fbaa96f)

*Sim-to-Real*

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
