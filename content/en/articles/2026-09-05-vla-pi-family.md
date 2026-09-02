---
title: "VLA Deep Dive (Part 2): The pi0 Family and Action Interface Evolution"
slug: "2026-09-05-vla-pi-family"
date: 2026-09-05
draft: false
categories: ["Embodied Intelligence", "Paper Analysis"]
tags: ["VLA", "pi0", "pi0.5", "pi0.7", "Flow Matching", "Action Chunking", "Physical Intelligence", "Embodied Intelligence"]
description: "Part 2 of a 3-part VLA series. pi0 uses flow matching for continuous action generation, pi0.5 introduces a discrete-continuous hybrid recipe, and pi0.7 extends generalization with context-rich steering and visual subgoals -- unpacking the core technical density of action interface evolution and proposing the Training Interface vs Execution Interface decoupling meta-axis."
toc: true
related_articles:
  - 2026-09-03-vla-deep-dive
  - 2026-09-07-vla-world-models
  - 2026-09-02-jepa-deep-dive
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
---

> **VLA Series (3 parts):** [Part 1: RT-2 to OpenVLA](/en/articles/2026-09-03-vla-deep-dive/) | Part 2 (this article) | [Part 3: VLA and World Models](/en/articles/2026-09-07-vla-world-models/)

In [Part 1](/en/articles/2026-09-03-vla-deep-dive/), we walked through RT-2, OpenVLA, and OFT, seeing the foundational architecture and six evolution axes of VLA. OFT's results revealed a key signal: the action interface itself is an important system design axis. Starting from this article, we enter the more technically dense part -- the three generations of the pi0 family.

## 1. pi0: Flow Matching + Action Chunking

### Paper Information

*pi0: A Vision-Language-Action Flow Model for General Robot Control*, Physical Intelligence, October 2024. arXiv:2410.24164.

### Physical Intelligence Background

Physical Intelligence is a San Francisco-headquartered company focused on general-purpose robot foundation models. Its pi series represents one of the important technical trajectories for continuous-action VLA today.

### Core Architecture

pi0 made a different choice from RT-2 / OpenVLA: **instead of discrete tokens, it uses flow matching to generate continuous actions.**

The architecture has two parts:

| Component | Description |
|-----------|-------------|
| VLM Backbone | PaliGemma (3B parameter vision-language model) |
| Action Expert | 300M parameter specialized network, attached to the VLM |
| **Total Parameters** | **Approximately 3.3B** |

It is important to note that **the action expert is not a traditional add-on controller.** pi0 actually adopts a structure similar to a two-expert mixture: images/text primarily use the VLM backbone's first set of weights, while robot state/action tokens use an independent set of action-expert weights; the two share context through attention. A more accurate description is: **a language-vision backbone + a continuous action generation expert, sharing conditional information through attention mechanisms.**

The inputs include image tokens, language tokens, and proprioception, which pass through the shared representation and then the action expert outputs an action chunk via flow matching.

### Three Mechanisms for Continuous Action Generation

There are three main mechanisms for continuous action generation, which need to be clearly distinguished. They are **not a sequence of replacements, but three parallel continuous action modeling approaches**:

```
             Continuous Action Modeling
                        |
          +-------------+-------------+
          |             |             |
     Regression      Diffusion    Flow Matching
          |             |             |
    direct output   denoising      vector field
                    sampling       + ODE
```

Flow matching trains a vector field / velocity field over a probability path, enabling the model to transport from a simple prior distribution to the target action distribution through ODE integration. What it does is:

- Define a probability path from a simple prior distribution to a target action distribution (linear-Gaussian probability path)
- Train a network to predict the velocity field along this path
- During inference, start from the prior and integrate along the learned vector field to obtain continuous actions

The key difference from diffusion is that their training objectives and inference forms differ. pi0 adopts flow matching, and in the specific implementation uses a small number of Euler integration steps for action generation, keeping the continuous generation process within a computational budget suitable for real-time robot execution.

### Action Chunks, Temporal Abstraction, and Planning: Three Easily Confused Concepts

pi0 each time predicts an action chunk of length **50 steps**, with each step corresponding to the current embodiment's action vector (up to 18 DoF). If calculated at a 50 Hz control frequency, a 50-step chunk corresponds to approximately 1 second in timescale -- but chunk length itself is not a time duration; the actual time span is T = H / f, and with a different embodiment or control rate, the same 50 steps would span a different duration. The paper reports that its system can achieve up to 50 Hz system control frequency on dexterous tasks.

It is important to note that flow matching inference itself still requires multi-step integration. The 50 Hz figure is the system-level control frequency including action chunk execution, not the speed of a single flow matching inference call.

Actual execution uses receding-horizon / action chunk execution: the model generates a 50-step chunk, executes the first few steps, then re-observes and generates the next chunk -- it is not open-loop execution of the entire 1-second action sequence.

**Three easily confused concepts need special clarification here:**

**Action chunking** -- outputting a_t, a_{t+1}, ..., a_{t+H-1} at once. This addresses reducing decision frequency and improving motion coherence. pi0's 50-step chunk belongs to this level.

**Temporal abstraction** -- compressing long tasks to a higher-level timescale. For example, "pick up the cup" is a high-level behavior, not 50 joint actions. pi0.5's semantic subtask belongs to this level.

**Planning** -- requires considering what state changes an action sequence will produce, and evaluating "which future is better." This is the core of world-model planning.

**Action chunk does not equal planning.** pi0 generates a 50-step action chunk, executes the first few steps, then re-observes and generates the next chunk. This is **receding-horizon policy execution**, not internally simulating multiple candidate futures and comparing them for selection. **Chunking changes the output timescale of the policy, not the policy's decision criterion** -- changing from pi(a_t|o_t) to pi(a_{t:t+H-1}|o_t) does not automatically become argmax_a E[J(z_{t+H})|z_t,a]. The latter begins to enter prediction/planning.

### Cross-Embodiment Action Space

pi0 maps the actions of different embodiments into a **common action representation of up to 18 DoF**, with insufficient parts handled through padding / masking. For example, single-arm and dual-arm systems can occupy different numbers of action dimensions respectively:

```
Different embodiments
      |
Map to a common action representation of up to 18 DoF
      |
Insufficient parts: padding / masking
```

This allows the same model to control robots of different morphologies -- this is one of pi0's important contributions, and also an early practice in the embodiment interface dimension.

### Training

It can be roughly understood as a multi-stage recipe of 'initialization + robot pretraining + post-training,' but the specific data mixing and training process is more complex than this simplified picture:

**VLM initialization stage:** Inheriting PaliGemma's internet visual-language pretraining knowledge. pi0 does not redo internet pretraining itself, but initializes from the already-pretrained PaliGemma.

**pi0 robot pre-training:**
- Open X-Embodiment 'Magic Soup' subset
- Proprietary multi-robot data: approximately **10,000 hours**, approximately **903M timesteps** (of which 106M from single-arm, 797M from dual-arm), covering **68 tasks** across **7 hardware configurations**. The maximum action/configuration dimension is **18**, corresponding to two 6-DoF arms + 2 grippers + mobile base + vertically actuated torso.

**Stage 2 -- Targeted post-training:**
- Fine-tuning on high-quality, curated task demos to acquire complex manipulation skills

### Key Results

Under the zero-shot real-world evaluation protocol reported in the paper, pi0 demonstrated capabilities that discrete token approaches did not possess on complex manipulation tasks:

| Task Type | Observations from pi0 Paper |
|-----------|---------------------------|
| Cloth folding | Demonstrated complex, multi-object, long-horizon dexterous manipulation |
| Desk clearing / bussing | Demonstrated cross-embodiment complex manipulation and language-conditioned execution |
| Novel skills | Learned through post-training tasks not covered or significantly different in pre-training |
| Comparison with discrete VLAs | Clearly stronger on these specific zero-shot protocols |

Note that these observations are highly dependent on the specific task protocol (number of trials, robot embodiment, task definition, the exact meaning of "zero-shot," etc.), and therefore should not be interpreted as a universal cross-model performance ranking. These results indicate that the continuous generation approach has potential on this type of task, but the gains cannot be attributed solely to "continuous representation" itself; action representation, generation mechanism, chunking, data scale, and training recipe are all co-varying factors.

### Understanding pi0's Performance Gains

From RT-2 to pi0, continuous action generation has indeed become an increasingly important direction. However, we cannot simply attribute the performance gains to "continuous representation being superior to discrete tokens." pi0 simultaneously introduced action chunking, flow matching, cross-embodiment training data, and larger-scale robot data. A more accurate conclusion is: **continuous action representation solves some structural problems of quantization and autoregressive output, but its ultimate benefit is strongly coupled with data scale, action chunking, and training recipe.**

### Limitations

- **Code and some model checkpoints have been open-sourced through openpi, but the training data, complete internal training system, and all production/experimental variants are not fully open**
- Some tasks requiring precise force control remain unreliable
- Generalization to entirely new physical domains (autonomous driving, aerial vehicles) is unknown
- VLM fine-tuning may lead to degradation of language/vision capabilities (catastrophic forgetting)

## 2. pi0.5 and pi0.7: From Action Generation to Policy Steering

### pi0.5: Data Composition + Hierarchical Policy

pi0.5's core contribution (April 2025, arXiv:2504.16054) is not simply "adding high-level reasoning," but introducing a **hierarchical architecture**, along with a very important technical finding: **discrete and continuous action representations can serve different roles within the same foundation model.**

pi0.5's training is divided into two stages, using different action representations and prediction targets:

```
                 pi0.5

Pre-training
+---------------------------+
| VLM + FAST autoregressive |
| action token prediction   |
| alpha = 0 (no flow loss)  |
+--------------+------------+
               |
Post-training
+---------------------------+
| Retain FAST sequence model|
| + add flow action expert  |
| alpha > 0 (flow loss on)  |
+--------------+------------+
               |
Inference
language/subtask
      |
semantic prediction (FAST AR)
      |
flow matching
      |
continuous action chunk
```

**pi0.5 does not simply replace a discrete policy with a continuous one.** The same model during pretraining has both FAST autoregressive action prediction and flow-field prediction, but alpha=0 meaning the flow loss is not active; during post-training alpha>0, adding the flow action expert while retaining the FAST sequence model. In other words, **FAST discrete modeling is not replaced -- it coexists with flow matching during the post-training stage.**

The **pretraining stage** uses **FAST action tokenizer** to discretize robot actions, enabling actions to share next-token prediction training with language and semantic subtask tokens. This allows large-scale heterogeneous data (different robots, different tasks, even web data) to be utilized within a unified sequence modeling framework. High-level **hierarchical task decomposition / semantic subtask prediction** is an important component of this stage.

The **post-training stage**, while retaining the FAST sequence model, **adds a flow-matching action head** (alpha>0) for high-frequency continuous control, targeting mobile manipulation post-training. During inference, the model first uses FAST to infer a high-level semantic subtask (e.g., "pick up the pillow"), then generates a continuous action chunk using flow matching conditioned on this subtask.

One notable data point: in pi0.5's first-stage training data, approximately **97.6% is not mobile-manipulator household data**. It gains generalization capability through mixing large amounts of heterogeneous data, then fine-tunes on a small amount of target-domain data.

pi0.5 therefore demonstrates that "discrete vs continuous" is not a simple substitution relationship. Its actual trajectory is not "discrete -> continuous" but rather **discrete for scalable multimodal pretraining + continuous for fine-grained control**.

From the design of pi0.5 and pi0-FAST, an increasingly clear engineering division of labor is: **discrete representation is better suited for foundation-model pretraining and multimodal sequence modeling, while continuous generation is better suited for final high-precision control.** If compressed into one sentence, it can be summarized as:

> **Discrete handles unification; continuous handles control.**

This is this article's summarizing interpretation of the pi0 / pi0.5 / pi0-FAST technical evolution, not a universal design principle already proven by any single paper.

### A Deeper Change: Training Interface and Execution Interface Begin to Decouple

Looking back at the evolution from RT-2 to pi0.5, there is a meta-axis deeper than the 'six axes' emerging:

```
                 Foundation Model
                       |
             +---------+---------+
             |                   |
       Training Interface   Execution Interface
             |                   |
       discrete tokens      continuous actions
       FAST / AR             flow / regression
             |                   |
       scalable learning     high-frequency control
```

Specifically:

- **RT-2**: training = discrete action tokens, execution = autoregressive action tokens
- **OpenVLA**: training = autoregressive action tokens, execution = autoregressive action tokens
- **OpenVLA-OFT**: foundation model representation -> continuous action head -> parallel execution
- **pi0**: VLM semantic representation -> continuous flow action expert -> high-frequency execution
- **pi0.5**: most interesting -- pretraining interface = FAST discrete tokens, post-training / execution interface = flow continuous action

What this suggests is not a simple 'discrete -> continuous' trajectory, but rather **the foundation-model learning interface and the robot execution interface begin to decouple**. Discrete tokens serve scalable multimodal learning; continuous generation serves high-frequency control -- the two can coexist in the same system.

This framework can thread together the entire story from RT-2 to OpenVLA to OFT to pi0 to pi0.5, and also provides a more explanatory perspective for understanding subsequent models.

pi0.5 demonstrated minute-scale long-horizon household manipulation, and discussed 10-15 minute long-task capability in the paper; its specific tasks in the quantitative real-home evaluation lasted approximately 2-5 minutes. It is important to note that "being able to execute 10-15 minute tasks" and "having a systematic benchmark on 10-15 minute tasks" are not the same thing. Error accumulation in long-horizon tasks remains the main bottleneck for VLA generalization.

### pi0-FAST: Why Discrete Tokens Have Not Disappeared?

Physical Intelligence subsequently explored **pi0-FAST**, using the FAST action tokenizer to compress continuous action chunks into discrete tokens, bringing robot actions back into the autoregressive language-modeling framework.

This shows that Physical Intelligence itself has not concluded that discrete action tokenization is a dead end. In fact, **discrete tokens still have enormous value for multimodal pretraining** -- they allow robot actions to share the same sequence modeling interface with language, vision, and semantic subtasks.

Here it is worth making an explicit comparison of the respective advantages of discrete and continuous:

**Advantages of discrete action tokens:**
- Naturally aligned with LLM/VLM autoregressive objectives
- Mixing language, vision, and actions into the same token space
- Leveraging large-scale multimodal pretraining
- Unified data format

**Advantages of continuous action (a in R^D):**
- Avoids quantization error from discrete bins
- Compatible with continuous distribution modeling methods such as diffusion / flow matching
- Can directly generate continuous action chunks
- More suitable for high-precision control

So the conclusion is not "discrete -> continuous" but rather **tokenization and continuous generation may serve different stages / different levels.**

### pi0.7: From Task Conditioning to Context-Rich Policy Steering

pi0.7 (April 2026, arXiv:2604.15483) further explores VLA's "steerable generalization."

From RT-2 to pi0.7, the policy's inputs underwent a qualitative change:

```
RT-2:   language -> action
pi0:    language + image + proprioception -> action chunk
pi0.5:  language + observation -> semantic subtask -> action chunk
pi0.7:  language + subtask + episode metadata + subgoal image + observation history + proprioception + control mode -> action chunk
```

pi0.7's true advance is not simply "more inputs," but rather: **the prompt changed from "describing what to do" to "describing how to do it."** That is, a gradual shift from **task conditioning** toward **context-rich policy steering**. This is precisely why pi0.7's title uses "Steerable Generalist."

The model no longer conditions only on language instructions, but unifies language, episode metadata, execution strategy information, visual subgoals, and observation history as multimodal context inputs to the policy.

**From a modeling perspective, pi0.7 can be understood as attempting to solve the conditional conflict problem introduced by heterogeneous data.** Different demos for the same task may exhibit completely different behavioral patterns -- fast but rough, slow but high-quality, containing errors, different strategies, different robots. If the training data only has (task, observation) -> action, then these behavioral patterns may be **mutually conflicting supervision** for the policy. pi0.7's solution is to add conditioning variables (subtask, strategy/metadata, quality, speed, subgoal image, control mode, etc.) -- **not simply adding more data, but adding conditioning variables that explain "why the data differs," making the originally mutually conflicting action supervision conditionally consistent.** This is a mechanistic explanation of pi0.7's design motivation, not a theorem proven by causal experiments.

pi0.7's model scale is approximately 5B, consisting of an approximately 4B VLM backbone, a video history encoding module (MEM-style video history encoder), and an 860M parameter action expert. **The pi0.7 VLA itself is approximately a 5B model.** The complete experimental inference stack can additionally include a high-level semantic policy and a **BAGEL-initialized subgoal-image world model** (approximately 14B parameters) for generating visual subgoals. It still follows the flow matching continuous action generation approach.

It is worth noting that **pi0.7's core VLA itself is not an action-conditioned world model**. But the complete inference system adds a **BAGEL-initialized visual generative world model** (approximately 14B parameters), external to the VLA, for generating candidate visual subgoals based on current observations, subtasks, and context, which are then provided as conditioning inputs to the low-level VLA. The entire system can be drawn as:

```
                  pi0.7 System
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
 High-level        World Model       VLA
 Semantic Policy   BAGEL-based       ~5B
        |           visual generator  |
        v              |              v
    subtask            v          action chunk
        |          subgoal image       |
        +--------------+--------------+
                       |
                  Robot Action
```

There is a very important boundary that must be pinned down here: **the world model in the pi0.7 system performs visual subgoal generation ("what the future should look like"), not action-conditioned dynamics prediction ("what will happen after executing a specific action").** Its information flow is:

```
current observation + subtask / context
       |
visual generative model
       |
candidate visual subgoal
       |
VLA conditioning -> action
```

And not:

```
current state + candidate action
       |
predicted future
       |
evaluate -> select action
```

These two functions of world models are fundamentally different. When this article discusses the relationship between VLA and world models, the distinction between "pi0.7 has a world model" and "pi0.7's VLA is itself an action-conditioned world model" must be the clearest boundary in the entire article.

Therefore, a more accurate positioning of pi0.7 is not "VLA has become a world model," but rather: **VLA has begun to use future goals produced by predictive models as policy conditioning signals.** This means the interface between policy and prediction has begun to emerge, but the two still bear different responsibilities: the world model is responsible for generating the visual target of "what the future should look like," and the VLA is responsible for learning "how to act in the current state to achieve this goal." It is not yet a unified, queryable action-conditioned dynamics model.

In a specific **pi0.7 (GC) zero-shot cross-embodiment T-shirt folding evaluation** -- using the generated subgoal / visual goal conditioning configuration, on a previously unseen dual-arm UR5e folding setting (without having trained on UR5e folding data) -- **pi0.7 (GC)** achieved 85.6% task progress and 80% success rate; 10 experienced teleoperators on the same unfamiliar UR5e dual-arm platform achieved 90.9% and 80.6% respectively. More interestingly, this was not simply copying source robot trajectories -- instead, the system exhibited the phenomenon of reorganizing behavioral strategies adapted to the target robot's kinematics.

### Visual Subgoal Is Not a World Model

With pi0.7 introducing visual subgoals, the boundary between VLA and world models begins to blur. However, it is important to note that **"using future visual subgoals" is not automatically equivalent to "having an explicit world model."** If what is being discussed is a world model for robot planning, then the key capability is typically action-conditioned prediction: given the current state and candidate actions, predict the future state or latent state. A more accurate description of pi0.7 is that it introduces future visual goals as a conditioning signal in the policy.

This distinction matters: a model "seeing a future goal" and a model "being able to predict how the world will change after executing actions" are two different capabilities.


---

**Next (Part 3)** enters the most conceptual part of the series: the VLA-world-model relationship analysis, open questions, and three judgments.

> **VLA Series:** [Part 1: RT-2 to OpenVLA](/en/articles/2026-09-03-vla-deep-dive/) | [Part 2: The pi0 Family](/en/articles/2026-09-05-vla-pi-family/) | [Part 3: VLA and World Models](/en/articles/2026-09-07-vla-world-models/)
