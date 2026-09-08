---
title: 'All Eyes, No Fingertips: Why General Robots Still Lack a Sense of Touch'
slug: "2026-09-11-tactile-force-sensing"
date: 2026-09-11
draft: false
categories: ["Embodied AI", "Multimodal Perception"]
tags: ["Embodied AI", "Tactile Sensing", "Force Control", "Impedance Control", "Admittance Control", "Contact-Rich Manipulation", "Active Perception", "VLA", "Sim-to-Real", "Robot Data", "GelSight"]
description: 'What general-purpose robots really lack is not "one more tactile sensor" but a scalable "perceive-contact → adjust-action" loop. This piece separates tactile / force / proprioception / force control into layers, clarifies that vision can only weakly observe many contact states, positions tactile sensing as an extension of the visual closed loop rather than the first closed loop, and explains why tactile Sim-to-Real is more manual and more calibration-loop dependent than visual Sim-to-Real.'
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

Even with your eyes closed, you can unscrew a half-finished water bottle, pick one egg out of a basket without crushing it — not because of vision, but because of "feel" in your hand. On many **general-purpose robots**, the opposite holds: the "eye" (the camera) is very advanced, while contact sensing on the "hand" is nowhere near as mature, universal, or standardized. This is not to say robots have no touch at all — wrist-mounted 6-axis force/torque sensors, joint torque sensing, and fingertip tactile arrays are all used today. What is genuinely scarce is **a deploy-at-scale loop of "sense the contact → change the action"**. This piece is about the layer in embodied AI that gets overlooked and yet matters most: **tactile sensing and force control**.

## 1. Why "seeing" is not enough

The visible progress in robotics over the past decade has been in seeing — cameras get sharper, vision models get stronger. So it becomes a default assumption: get the pixels right, and the action will follow. Once contact enters the picture, "seeing" starts to fall short.

Let's be precise first: it is not that "vision cannot see contact, only touch can." **Vision can infer contact, slip, and deformation indirectly** — optical flow, object motion estimation, and local deformation all give clues. The real difference is that **many contact states are weakly observable or even unobservable to vision, but directly observable to tactile / force sensors**. Occlusion, tiny contact regions, transparent or reflective objects, fast motion — these are exactly the everyday manipulation scenarios where visual observability drops the most.

A few cases where vision struggles but "feel" is decisive:

- **Picking up an egg.** How much grip force an egg can survive is very hard to read from appearance alone. What actually works is a small loop: vision locates the egg → establish initial contact → tactile / force read out contact area and slip → the controller nudges grip force → check whether slip has stopped. **The point is not "I sensed it" but "I can change the next action based on the sensation."**
- **Inserting a USB or a peg.** Not "vision is useless" — vision brings the end-effector into the right neighborhood, and then the task enters a very typical **contact-rich manipulation** regime: once contact happens, the environment's constraint on motion suddenly strengthens, and contact force / torque telling the robot "what I am pressing against, did it slide in" is more direct than watching pixels.
- **Twisting off a bottle cap.** How tight is "tight enough"? If it slips, should I press harder? These are questions the contact interface answers.
- **Taking one sheet from a stack.** Grabbing too many, dragging neighbors along, the sheet slipping — many of these failures barely register in a single frame but are obvious in the hand.

So the more rigorous phrasing is: **many of the critical contact cues are not directly and stably readable from a single frame.** Contact force, local deformation, friction state, whether slip is happening — these live inside the "contact" interface, and are usually only clear when you "press, feel the resistance, feel the rebound." Vision can guess indirectly, but not stably. (Side note: "hardness" itself is not something any sensor reads directly either — it has to be estimated from the force–deformation relation.)

> Rephrased: **vision is what brings the robot to the neighborhood of contact; tactile and force are what keep navigating after contact**. For actions — fine manipulation in particular — the second half is often what decides success.

## 2. What the robot's "senses" are missing

**First, disentangle tactile ≠ force ≠ force control (and don't forget proprioception).**

These terms get conflated easily, but strictly they are different layers. On the **sensing** side: **vision** tells you external world state; **tactile** tells you local contact state — contact location, pressure distribution, shear, slip, texture; **force / torque sensing** tells you how much external wrench the robot as a whole is receiving; **proprioception** tells you where the robot's own joints and end-effector are and how they are moving. On the **acting** side: **force control** is about actively regulating behavior using those feedbacks.

A robot arm with a wrist 6-axis F/T sensor knows "there is currently 8 N of downward force on the end-effector," but may not know "each fingertip is contacting where, which side is slipping." Conversely, a high-resolution tactile array senses contact distribution clearly but may not directly give precise absolute force. **Tactile is not everything the robot senses** — it is just often the most missing piece.

So the chain this piece cares about is: **tactile / force provide contact observations → state estimation turns raw signals into *contact states* like "where contact is, whether slip has started, whether it is stuck" → the controller / policy adjusts actions on those states → force control realizes "constrained contact behavior."** The middle "signal → state estimation" step is very easy to skip, but it is what turns tactile from a reading into a meaning.

The human palm is one of the densest tactile surfaces in the body: pressure, texture, temperature, slip, and proprioception (you know where your fingers are without looking) all work together. In comparison, most robot hands read a much sparser signal:

| Sensory dimension | Human hand | Typical robot |
|---|---|---|
| Position / pose | Yes (proprioception) | Given by joint encoders + motor current + torque + kinematics combined, **often more precise than humans for its own pose** |
| Joint torque | Not read directly; felt indirectly via muscle spindles, Golgi tendon organs, etc. | Readable on some platforms (joint current or dedicated torque sensors) |
| Contact pressure distribution | Very dense | Mostly absent or very coarse |
| Contact texture / local material properties | Press → feel resistance → feel rebound → feel surface friction, **acquired dynamically during contact** | Vision recognizes lots of "looks-like" material; what is missing is that **local, dynamic** material information during contact (usually has to be **jointly inferred** from vision / force / vibration / current / motion response) |
| Slip / friction | Detected instantly | Detectable (tactile arrays, shear force, contact-point motion, vision) — hard to **estimate stably, at low latency, and generalize across sensors, then feed into control** |
| Temperature | Yes | Essentially no |

One thing to stress: **human tactile and robot sensors do not map one-to-one.** A robot without a tactile array is not "totally unable to feel" — with end-effector wrench, joint current, visual deformation, vibration and other channels, a lot can be inferred. The real gap is not "do we have sensors" but:

> What robots lack is **high-density, multimodal, low-latency, broad-coverage contact sensing that also gives *local* contact information**. Locality is the dimension most easily overlooked and yet most critical — take "two fingertips together lifting a cup": *a wrist 6-axis F/T sensor might just say "net 5 N on the end-effector"*; *a tactile array will tell you "left fingertip front carries 2 N, right fingertip edge carries 3 N, and right side is starting to show shear."* For a manipulation policy, these two pieces of information are on completely different levels — the former is a resultant wrench, the latter is **contact state**.
>
> The way a robot hand currently looks, it is more like a hand wrapped in a thick glove with a few sparse receptors at isolated points: it can feel "I touched something," but hardly "where, how, is it right."

## 3. Why this has been hard for so long

### 3.1 The sensor itself is hard, harder to keep stable long-term

To sense contact, the tactile layer has to be thin, wear-resistant, and survive repeated pressing without breaking. But what actually gives engineers trouble is not "make one work" — it is **calibration, temperature drift, hysteresis, nonlinearity, mechanical wear, skin aging, per-finger consistency**. There is a saying in this community: the closer a design gets to a real human hand, the harder it becomes to productize — you need accuracy, long-term stability, reproducible calibration, and you must not destroy the hand's original grasping performance by adding a sensing layer on top.

### 3.2 Data is not "absent" — its scale and uniformity lag vision by a wide margin

To be strict, tactile data is not a blank space. The community already has a set of tactile datasets, visuo-tactile datasets, tactile pretraining efforts, and early explorations of "tactile foundation models"; sensors like [GelSight](https://gelsight.com/), [GelSlim](https://github.com/Antaoyu/GelSlim_4Gen_Curvature-Based_Fingertip_Sensor), and [TacTip](https://www.tacpix.com/products) are widely used. **The real problem is not "no tactile data" — it is that scale, uniformity, task coverage, and cross-sensor transferability are all nowhere near what vision has.**

The harder layer: **tactile data is heavily embodiment-dependent.** The same "pick up a cup" produces very different signals across robot hands, fingertip materials, and sensors. Visual data shares easily (everyone is capturing RGB pixels), while tactile data is deeply coupled with "which hand, which skin, contacting how" — very hard to accumulate into a cross-device common base like ImageNet. This ties back to [robot data scaling](/en/articles/2026-09-09-robot-data-scaling/): robots are not just short on "action data," they are short on **action data with contact state attached**.

### 3.3 Simulation is very hard to get right — the Sim-to-Real gap on contact is deeper

Light and collisions are relatively easy in simulation. But "contact — deformation — friction — slip" is a much harder physics stack: friction coefficients, contact stiffness, material viscoelasticity, micro-surface structure, contact area evolution, sensor noise, soft-material / skin coupling… every link can inject error.

But be careful not to over-swing: *"sim is inaccurate" does not mean "everything must come from real data."* What engineering actually does is treat sim as a calibrable approximation — and a **loop**, not a one-shot pipeline: **train in sim → real experiments expose the gap → estimate friction, stiffness, damping via system identification → recalibrate sim → retrain → re-verify**; combined with domain randomization ([Tobin et al., 2017](https://arxiv.org/abs/1703.06907)), real-world fine-tuning, and residual learning / action-transformation-style methods to close the residual. **Sim remains a powerful tool for training and verification**; the difference is that contact-model errors bite policy transfer particularly hard, so tactile Sim-to-Real is more manual and more dependent on that iterative calibration loop than visual Sim-to-Real — essentially the same *data engine / feedback loop* argument from [Sim-to-Real methodology](/en/articles/2026-09-10-sim-to-real-methodology/).

### 3.4 What is missing is not just a data format — but an entire unified stack

Vision's real advantage is not "everyone has the same camera" (there are still differences in resolution, lens, spectrum, and intrinsic / extrinsic parameters) — it is that **the base representation of images is highly unified**: pixels, RGB, video, coordinate frames, and around them a mature set of shared interfaces. Tactile is far messier: different sensors output pressure arrays, 3D geometric deformation, RGB images, shear forces, normal forces, 6-axis F/T, resistance / capacitance changes, high-frequency vibration signals… **even the same word "pressure" is not directly comparable across sensors.** So a more accurate statement is: what tactile lacks is not just a common data file, but **unified sensor interfaces, calibration methods, coordinate-frame conventions, data representations, task definitions, and evaluation metrics** — an entire ecosystem-level infrastructure gap, not something a single hdf5 schema can close.

### 3.5 Tactile is bad at "saying what it felt"

Do not jump to "tactile has no semantics." On the contrary, **tactile can carry very rich semantics** — material, texture, hollowness, deformation, whether the grasp is stable, all of these can be judged through touch. The real problem is: **between image and high-level semantics, there is already a very mature middle representation** (pixels → features → objects / scenes), while **between raw tactile signal and high-level semantics, we still lack a widely reused, cross-sensor representation path**.

A camera naturally gives you a 2D spatial structure: here is a cup, there is a hand, this is red, that is the table — the representation is already close to semantics. Tactile, on the other hand, typically hands you a pressure blob, a time series, a shear-force transition — semantics are not absent, but the robot has to learn on its own: "this is not a blob of pressure change, this is **an object currently slipping out from between my fingertips**."

This learning chain is itself a large problem:

```text
raw tactile signal  →  contact state  →  physical interpretation  →  action
```

It also pushes the question in [VLA × world models](/en/articles/2026-09-07-vla-world-models/) one step forward — VLA maps vision + language to action; the next step is plausibly "vision + language + tactile + proprioception → contact state understanding → action closed loop."

> **My read**: among these five, **data scale & uniformity, and "still lacking a mature generic representation between raw signal and high-level semantics"** are the most foundational. Algorithms can keep competing, but without stable comparable hardware, a cross-sensor transferable dataset, and the middle representation that pulls raw signals into "contact state," a model cannot even assemble what to learn from. Of course, different research routes hit different first walls: *hardware folks first hit durability and consistency, whole-robot folks first hit the hand itself, control folks first hit sensor bandwidth and actuation delay, learning folks first hit data volume and annotation cost*. But if the goal is "make tactile deployable at scale like vision," then **data, standardization, and hardware ecosystem are probably the non-avoidable base bottlenecks** — not because they are the hardest, but because they decide whether others can build on top of them.

## 4. Force control: it is not enough to "feel" — you must use it right

Sensing is only the first step. The harder step is making the robot **adjust its action based on force**. There is a basic taxonomy here worth spelling out.

- **Position control**: command "move the hand to this coordinate." Stiff, precise, but the moment something unexpected resists, it "argues" with the environment and can crush things.
- **Force control**: command "apply this much force." Suits tasks dominated by contact force, like "move along a surface."
- **Impedance control**: not "where" or "how hard," but making the end-effector **behave like a spring–damper system**: when position deviates from the target, generate compliance according to a set force–displacement relationship.
- **Admittance control**: the reverse — from measured external force, compute a desired position / velocity change, and track it. In industrial robotics these are distinct.

Now, the popular "make the hand soft" metaphor needs a caveat: **"soft" is not simply lowering stiffness** — it is making the robot produce **controlled compliance** in response to external disturbances, along a set force–displacement relation. Push stiffness too low and the robot becomes floppy, precision is gone. Tuning "feel" is about shaping the spring–damper curve, not switching it off. So many assembly, insertion, wiping, and fine-manipulation tasks with high contact uncertainty really require **some form of active or passive compliance** — not all tasks must be solved with active force control; high-precision positioning, dedicated fixtures, compliant mechanisms, passive compliance, and pre-defined trajectories are also common engineering answers. **Force control ≠ "the robot learned to control force by itself"** — real systems usually combine *mechanical compliance + impedance control + force feedback + policy learning*.

Beyond that, a more ambitious direction is to place force / tactile alongside vision, language, and proprioception **inside the same policy and learning framework**: not "just another sensor channel," but **adding a high-value observation about contact state** to the policy. This direction is right, and there is a body of work showing it works; what is still unsolved is **the lack of a scaled, unified, cross-hardware generalizable paradigm comparable to vision-language models**. So the more accurate statement is not "does visuo-tactile learning work," but "it works, but it is still far from unified."

```text
vision / language / proprioception / tactile · force
              │
              ▼
       High-level policy
              │
              ▼
 target pose / velocity / force reference / impedance parameters
              │
              ▼
        Low-level controller
              │
              ▼
             Motors
              │
              ▼
            Contact
              │
              ▼
     tactile / force feedback ──┐
                                │
                                └─► back to policy (closed loop)
```

> **A common confusion.**
>
> Plugging tactile into a VLA **does not mean the VLA directly outputs motor current or torque**. A more typical division of labor is: the model emits *target pose / velocity / force reference / impedance parameters*, and below it a low-level controller running at a much higher frequency executes and closes the loop. "Tactile enters the VLA" ≠ "the VLA is the controller."
>
> There is also a physical intuition worth pointing out: **the two run at very different time scales.** A high-level policy usually updates "what to want next" at a slower decision frequency; the low-level controller runs at clearly higher frequency, executing commands and closing the control loop steadily. That middle layer is the bridge that turns "decision" into "motor." This is also why, however large the VLA is, it does not automatically become a controller that survives contact disturbances.

## 5. The real value is the loop

At this point I can put a more essential claim on the table: **the value of tactile is not "sense more" — it is enabling the robot to change its action in real time based on contact outcomes.**

One thing to be careful about: it is tempting to say "tactile is the first thing that closed the loop in manipulation." Strictly, that does not hold. *Visual servoing* is itself a closed loop (see the classic tutorial by Hutchinson, Hager, and Corke, *A tutorial on visual servo control*, IEEE Transactions on Robotics and Automation, 1996 · [Semantic Scholar](https://www.semanticscholar.org/paper/A-tutorial-on-visual-servo-control-Hutchinson-Hager/4676d81d6a2477f6ea1de5723fc81e431fbaa96f)); many robot systems have long been running "see → estimate state → act → see again." What tactile really adds is not "opening a loop for the first time," but:

> **Extending the visual-dominant closed loop to cover the "after contact" states that vision cannot reliably observe.**
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

One more direction deserves a mention: tactile is not only for "avoiding failure" — it is also for **exploration**. When you cannot see inside a cup, you reach in and feel; when you do not know how heavy something is, you lift it and estimate; when unsure whether a part is slipping, you nudge it lightly; when unsure how soft or stiff a material is, you press and rub. In robotics this has a broader name: **active perception / active sensing**. Its scope is broader than "active tactile" — *moving the camera, changing viewpoint, walking around the object, reaching in, pushing* are all active perception; **tactile is one very important realization of it**. The core is: *the robot does not just passively sense the world through sensors; it actively acts to earn the information it wants.* Add this and tactile is upgraded from "a sensor" to "a link in sense — act — re-sense," and only then does it truly enter the core of embodied AI.

## 6. This is also part of the Last-Mile long tail

Previous pieces ([Sim-to-Real methodology](/en/articles/2026-09-10-sim-to-real-methodology/), [robot data scaling](/en/articles/2026-09-09-robot-data-scaling/)) kept saying one thing: real deployment is about the system. The value of tactile often **does not show up as "success rate goes from 80% to 90%"** — it shows up in the **long tail of contact states** that vision cannot handle — slightly tilted, slightly slipping, jammed, contact point shifted, friction suddenly changed, tolerance pushed to the edge, the object being squeezed out of shape. These are exactly the last-mile scenarios where "each event is low probability, together they add up."

Another point, closer to engineering intuition: **tactile's benefit is often conditional**. Without tactile you might still get 95% success under normal conditions; with tactile it becomes 99%. Looked at only through demos, both robots pick the object up, and the difference is not dramatic. The real fork appears when: the object is a bit wet, the grasp position is a bit off, the friction coefficient drifted, the object starts deforming, the contact point is quietly sliding, a micro-slip has begun but the object has not dropped yet… *tactile may not make the demo more impressive, but it very plausibly shortens the failure tail* — which lands right on the core claim of [Sim-to-Real methodology](/en/articles/2026-09-10-sim-to-real-methodology/): **whether a system can be trusted is never judged by its highlight moments, but by those "individually low-probability, collectively non-trivial" moments.**

> Vision models handle the bulk of ordinary cases in "understanding the world"; tactile and force control handle the local anomalies "after contact." What is genuinely hard for a robot is precisely those **low-probability, combinatorially complex, hard-to-enumerate-in-advance** contact states.

## Summary

Robot vision is advancing fast, yet many **general-purpose robots** still have a pair of great eyes and a pair of hands whose sensing stack has not been systematically built out. In the real world, much of success and failure hides in the moment of contact — *contact force, local deformation, whether it is slipping, how much effort to apply*. This layer is difficult because of the sensor, the data scale and uniformity that lag vision, the Sim-to-Real gap on contact, and the still-low level of standardization. None of these is something "just throw more compute at it" can cross.

If I had to compress this piece into a single technical spine:

```text
vision             →  world understanding
tactile / force    →  contact understanding
force control      →  contact closed loop
policy learning    →  turning these local feedbacks into generalizable behavior
```

None of these four layers means "the robot is useless without it" — visual picking, visual localization, open-space navigation, many non-contact operations, all run well without tactile. But for **contact-rich, fine-manipulation, contact-state-heavy tasks** — assembly, insertion, wiping, deformable-object manipulation — missing any of these four layers is a very plausible bottleneck between "the robot can do it" and "the robot reliably does it well." **Getting from "can see" to "can work" — the missing link is usually not how big the perception model is, but whether the contact → feedback → adjust loop has been seriously built.**

So the next genuinely exciting progress in robotics may not be "seeing even better," but — the day a robot finally learns to feel, and builds a whole loop around that feeling.

---

**Further Reading / Sources**

- GelSight high-resolution vision-based tactile sensor · <https://gelsight.com/>
- GelSlim open-source fingertip tactile sensor · <https://github.com/Antaoyu/GelSlim_4Gen_Curvature-Based_Fingertip_Sensor>
- TacTip 3D-printable tactile fingertip · <https://www.tacpix.com/products>
- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 · [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
- Hutchinson, Hager, Corke, *A tutorial on visual servo control*, IEEE Transactions on Robotics and Automation, 1996 (visual servoing as a canonical closed loop) · [Semantic Scholar](https://www.semanticscholar.org/paper/A-tutorial-on-visual-servo-control-Hutchinson-Hager/4676d81d6a2477f6ea1de5723fc81e431fbaa96f)
