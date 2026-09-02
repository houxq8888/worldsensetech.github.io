---
title: "VLA Deep Dive (Part 3): VLA and World Models, Open Questions, and Three Judgments"
slug: "2026-09-07-vla-world-models"
date: 2026-09-07
draft: false
categories: ["Embodied Intelligence", "Paper Analysis"]
tags: ["VLA", "World Model", "Planning", "Embodied Intelligence", "Robot Foundation Model"]
description: "Part 3 of a 3-part VLA series. Discussing the relationship between VLA and world models -- distinguishing passive predictive, action-conditioned, and subgoal generator types -- then exploring data bottleneck, long-horizon, safety, and missing modality open questions, concluding with three judgments."
toc: true
related_articles:
  - 2026-09-03-vla-deep-dive
  - 2026-09-05-vla-pi-family
  - 2026-09-02-jepa-deep-dive
  - 2026-09-01-world-model-h2-review
  - rssm-deep-dive
  - 2026-08-25-dreamer-explained
---

> **VLA Series (3 parts):** [Part 1: RT-2 to OpenVLA](/en/articles/2026-09-03-vla-deep-dive/) | [Part 2: The pi0 Family](/en/articles/2026-09-05-vla-pi-family/) | Part 3 (this article)

In [Part 1](/en/articles/2026-09-03-vla-deep-dive/) and [Part 2](/en/articles/2026-09-05-vla-pi-family/), we walked through the complete technical evolution of VLA from RT-2 to pi0.7. This final article brings the threads together to focus on the most core question: what exactly is the relationship between VLA and world models?

## 1. VLA and World Models: Policy Learning vs Predictive Modeling

This is the part I think deserves the deepest discussion. The technical evolution above has already touched on this question multiple times -- from RT-2's model-free policy, to pi0's action chunk not equaling planning, to pi0.7's visual subgoal not equaling a world model -- now let us bring these threads together.

### What VLA Lacks Is Not "Prediction Capability" but an Explicit, Queryable Action-Conditioned Prediction Interface

A common simplification is: VLA can only do actions, world models can only do predictions. But this is not precise enough.

VLA can certainly make predictions -- a large enough autoregressive model can perfectly well predict the next frame. The real distinction is not about "whether there is prediction capability," but rather: **is prediction an explicit, queryable, action-conditioned interface of the model?**

Specifically:

- **VLA learns the action distribution**: pi(a_t | o_{<=t}, l) -- only needs to answer "what action should I take now?"
- **World model learns the future distribution**: a typical action-conditioned world model can be expressed as p_theta(z_{t+1:t+H} | z_t, a_{t:t+H-1}), or can also be a deterministic latent transition function f_theta(z_t, a_t) -> z_{t+1} -- answering "if I execute these actions, what will the future look like?"

With the latter, one can naturally form a typical form of planning:

```
Candidate action a(1) -> predicted future o^(1) -> evaluate J(a(1))
Candidate action a(2) -> predicted future o^(2) -> evaluate J(a(2))
...
Select the action sequence with the highest J
```

**This is a typical form of planning: using a predictive model to evaluate the consequences of candidate actions or trajectories, then selecting or optimizing.** Planning can also be achieved through trajectory optimization, MPC, gradient-based optimization, tree search, latent-space optimization, goal-conditioned planning, and other approaches -- it does not necessarily require explicitly generating multiple discrete candidate trajectories. A typical imitation-learning VLA does not use action-conditioned future prediction as an explicit, queryable model interface.

Note: a typical imitation-learning VLA does not explicitly learn a queryable action-conditioned dynamics model -- but the policy itself can implicitly encode dynamic priors. This is very different from "having no internal representation of the physical world at all."

### A More Accurate Distinction Framework

| Dimension | VLA / Policy | Action-conditioned World Model |
|-----------|-------------|-------------------------------|
| **Core Question** | What should I do now? | What will happen after I do it? |
| **Learning Objective** | pi(a \| o, l) | p(z_future \| z, a) or f(z, a) -> z' |
| **Data Relationship** | observation -> action | observation + action -> future |
| **Output** | Action commands | Predicted future state / latent |
| **Typical Use** | execution | prediction / planning |
| **Main Risk** | policy error / distribution shift | model bias / compounding prediction error |
| **Requires Search?** | No | Can combine with search / MPC / optimization |
| **Typical Role** | leans toward execution | leans toward prediction / planning |

Simply put: a VLA answers "what action should I take?"; a world model answers "what will the world look like after executing this action?" In terms of typical role, VLA leans more toward execution, world model leans more toward prediction/planning -- but VLA can also do implicit planning, hierarchical policy, chain-of-thought, and world models can also directly support policy learning.

### Passive World Models vs Action-conditioned World Models vs Subgoal Generators

There is another easily confused concept that needs distinguishing. **When this article refers to 'world models for robot planning,' it focuses on predictive models with a queryable action-conditioned prediction interface.** Broader definitions of world models can include passive video prediction, latent dynamics, reward prediction, object-centric models, generative simulators, and goal-conditioned models, but here we focus on types directly relevant to planning.

**Passive world models** can learn "how the world changes" using only video -- predicting o_{t+1} from o_t, without action labels.

**Action-conditioned world models** require (o_t, a_t, o_{t+1}) triplets, learning "**what results different actions will produce.**" The action here does not have to be a low-level robot motor command -- it can be an end-effector action, a semantic action, or even a high-level skill. What "action-conditioned" truly requires is that the model knows what intervention / control variable caused the state transition.

**Subgoal generators (such as pi0.7's BAGEL-based world model)** are a third type: they do not predict "what will happen after executing a specific action," but instead generate "what the future should look like" as candidate visual subgoals based on task conditions and context.

| Model Type | Conditioning | Output | Directly Answers 'Action Consequences'? |
|------------|-------------|--------|----------------------------------------|
| Passive predictive | Current state | Future state | No |
| Action-conditioned WM | Current state + action | Future state | Yes |
| Subgoal generator | Current state + task/context | Visual goal | No |
| Policy | Current state + task | Action | -- |

This table can serve as a core reference for understanding the relationship between VLA and world models.

So what truly needs action-labeled interaction data is the **world model used for action-conditioned planning**. This also explains why V-JEPA 2 (passive video prediction) and V-JEPA 2-AC (action-conditioned) need to be separated in the technology stack -- JEPA itself is a predictive representation learning method; V-JEPA 2-AC is an **action-conditioned extension** on the V-JEPA 2 series, not simply the next-generation standalone model -- it further introduces action conditioning into the prediction process, enabling it to take on the role of action-conditioned world modeling.

### The Unified Model Technical Framework

A true unified model needs to simultaneously answer two questions. A conceptual joint modeling formulation can be written as:

p(a_{t:t+H}, z_{t+1:t+H} | z_t, l, g)

That is, simultaneously learning:
- **What should I do?** (policy)
- **What will happen after I do it?** (prediction)

This is more precise than simply saying "VLA + world model" -- it defines a joint model that possesses both an action distribution and a future distribution. **Joint modeling of policy and future state is the foundation for planning, but true planning still requires an objective function, search, optimization, MPC, or other action selection mechanism.** The joint distribution itself does not equal planning.

### The Two Lines Are Converging

A common misconception needs correction: the world model line is not "without language" or "unable to do actions." V-JEPA 2 has already demonstrated the complete technology stack from web-scale video pretraining to action-conditioned latent prediction to robot planning/control, including zero-shot robot deployment and image-goal planning. World models themselves can also acquire semantic capabilities through language alignment.

Planning in the JEPA line is also not necessarily "generate multiple trajectories and select." It can be latent prediction -> goal-conditioned planning, implemented via search, optimization, or policy guidance.

So the more accurate picture is: **VLA and world models are approaching the same goal from two directions -- a robot foundation model that simultaneously possesses policy, prediction, and planning capabilities.** Future systems are more likely to be Actor + Predictor, not one or the other.

```
                 Robot Foundation Model
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       Language       Perception      Robot Data
          |              |              |
          +--------------+--------------+
                         |
                 Shared Representation
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       Semantic       Policy        Prediction
       Subtask          |              |
          |             v              v
          |       Action Chunk    Future State
          |             |              |
          |             v              v
          |       Flow Matching    World Model
          |             |              |
          +-------------+--------------+
                        |
                  Physical Action
```

**The future robot foundation model is likely neither a pure VLA nor a pure world model, but a unified system that simultaneously supports semantic conditioning, policy execution, and predictive modeling.**

### Are They Complementary?

I believe the answer is yes, but the complementarity is more subtle than "putting two modules together."

**What does a typical direct-policy VLA lack?** It does not explicitly learn a queryable action-conditioned dynamics model. When encountering novel situations not seen during training, it can only rely on the generalization capabilities learned during pretraining -- the policy can implicitly encode dynamic priors, but cannot explicitly simulate the consequences of actions the way a world model can.

**What does a world model lack?** While world models are gaining language and action capabilities, they still fall short of the VLA line in terms of the efficiency of end-to-end policy learning and the naturalness of language grounding.

So a natural idea is: **use the world model for physical prediction, and the VLA for action execution and language understanding.**

## 2. Open Questions

**Data bottleneck -- from "hours" to "effective data."** A more meaningful question may not be "how to obtain a million hours of robot data," but rather: **should robot data really continue to be measured in "hours"?** One hour of a human continuously folding 300 garments successfully, and one hour of a robot encountering 50 failures, 20 recoveries, 10 different strategies, and 5 embodiments -- the information content is completely different. The value function for future data scaling may look more like:

Data Value = f(diversity, failure, recovery, embodiment, task coverage)

rather than simply Data Value proportional to hours. This connects directly to pi0.7's exploration of heterogeneous / suboptimal data. Many mainstream VLA datasets consist primarily of successful demonstrations, with failure / recovery data being relatively scarce. **How do we move from "successful demo datasets" to experience datasets that include failures, recoveries, and policy variations?** is the more critical question. pi0 itself provides direct evidence: using only high-quality data makes the policy more fluent but prone to lacking error recovery; diverse, lower-quality pretraining data provides a recovery / correction repertoire. The next stage is not simply scaling hours, but scaling useful experience.

**Long-horizon tasks -- error propagation, not step count.** The real problem with long-horizon tasks is not H > 5 or H > 50, but the probabilistic effect of error accumulation. Under a simplified i.i.d. approximation with no recovery:

P(success over T) ≈ product_{t=1}^{T} p_t

If per-step success rate p_t = p = 0.98, then 0.98^100 ≈ 13%. This is of course not a realistic statistical model of robot task success rates -- in real tasks, decisions are not independent, recovery can alter subsequent probabilities, some errors are recoverable, and some critical actions are far more important than others -- but it illustrates the exponential intuition of error accumulation well: **the essence of long-horizon difficulty is error accumulation, not simply sequence length.** This also explains why hierarchical policy, recovery policy, replanning, world models, and memory are all natural directions for addressing long-horizon problems.

**Safety -- three levels.** VLA safety issues can be divided into three levels:

*Policy safety*: Will pi(a|o) output dangerous actions?

*Predictive safety*: p(o_future|o,a) -- will this action cause danger when executed?

*Runtime safety*: Even if the model is wrong, is there an independent safety layer to intercept?

One possible runtime safety architecture is:

```
VLA
 |
candidate action
 |
world model / safety critic
 |
constraint checker
 |
robot
```

Safety constraints cannot be treated merely as a language-level alignment problem -- they are hard engineering constraints.

**Missing modalities.** Vision and language remain the core conditioning modalities for mainstream VLAs, and proprioception has already become a standard input for many systems (both pi0 and pi0.7 use it), while coverage of modalities such as touch, force/torque, and audio in current large-scale VLA pretraining systems remains notably insufficient. Yet for fine manipulation (screwing in bolts, inserting keys, folding soft objects), these modalities may be critical information sources.

**Does VLA need a world model?** I think this question does not yet have a definitive answer. pi0.7's introduction of visual subgoals as a conditioning signal does improve generalization, but this is not the same as "having an explicit world model." True integration may require a single model to simultaneously achieve: language grounding, action-conditioned prediction, and high-frequency continuous control. **As far as the publicly representative work discussed in this article, there is not yet a system that addresses language grounding, action-conditioned prediction, and high-frequency continuous control simultaneously in a mature and unified manner on large-scale real robot tasks.**

**The final verdict on discrete vs continuous.** From RT-2's discrete tokens to OpenVLA-OFT's continuous regression to pi0's flow matching, continuous methods have demonstrated advantages in control precision and inference speed. But pi0.5 and pi0-FAST show that **discrete and continuous likely serve different roles: discrete handles "unification" -- enabling actions to share the sequence modeling interface with language, vision, and semantic subtasks; continuous handles "control" -- providing high-frequency, fine-grained continuous action output at the final execution stage.**

```
         Foundation-model pretraining
                    |
                    |
           discrete tokens
                    |
                    |
              shared LM space
                    |
                    v
       continuous action generation
                    |
                    v
           high-frequency control
```

This judgment is much stronger than "continuous will eventually replace discrete," and it better explains why pi0-FAST, pi0.5, and pi0.7 -- which might appear to be "going backwards" -- are actually exploring different model interfaces.

---

## 3. Three Judgments

Finally, let me distill the article's argument into three judgments. The first part summarizes technical trends from existing papers; the second part proposes future architecture judgments based on these trends.

**Judgment 1: VLA's progress cannot be explained by parameter scale alone; the action interface is becoming a design axis parallel in importance to backbone scaling.** RT-2 -> OpenVLA -> OFT -> pi0 demonstrate that backbone scaling + data scaling + action interface jointly determine performance. From RT-2's 55B to OpenVLA's 7B to pi0's 3.3B, parameter counts are shrinking; but from 256-bin discrete tokens to parallel continuous regression to flow matching + 50-step action chunks, the action interface is continuously evolving. OFT's results demonstrate that the action interface, decoding strategy, and temporal chunking are themselves important system design axes, not merely secondary concerns to backbone scaling.

**Judgment 2: The key bottleneck for generalist robot capability is shifting from representation scaling to effective data scaling, temporal abstraction, and recovery.** pi0.5's 97.6% non-target-domain data, pi0.7's utilization of suboptimal data, and the structural difficulty of error accumulation in long-horizon tasks collectively suggest that -- beyond model scale -- effective data scaling, temporal abstraction, and recovery are gradually becoming independent performance determinants.

**Judgment 3: The real next stage may not be "VLA or World Model," but the unification of policy, predictor, and planner.** The technology map for the future robot foundation model can be drawn as:

```
                 Robot Foundation Models
                           |
          +----------------+----------------+
          |                |                |
       Backbone       Action Interface   Temporal Structure
          |                |                |
      VLM / VLA       discrete token      action
          |                |              chunk
      semantic        continuous            |
      grounding       regression       semantic subtask
          |                |                |
          |           flow matching         |
          |                |                |
          +----------------+----------------+
                           |
                    Generalist Policy
                           |
                +----------+----------+
                |                     |
             Action              Prediction
                |                     |
                |              future state /
                |              visual subgoal
                |                     |
                +----------+----------+
                           |
                   Planning / Recovery
                           |
                           |
                    Physical Robot
```

If this technical main thread is compressed:

```
RT-2
|
+- web knowledge -> action token
|
v
OpenVLA
|
+- open multi-robot scaling
|
v
OpenVLA-OFT
|
+- action interface
|
v
pi0
|
+- continuous generative action
|
v
pi0.5
|
+- heterogeneous data
+- semantic hierarchy
|
v
pi0.7
|
+- context-rich steering
+- subgoal conditioning
+- heterogeneous / suboptimal experience
|
v
???
+- policy
+- prediction
+- planning
```

If a directional description is to be used:

> A possible form of Robot Foundation Model = combining Perception + Language + Policy + Prediction + Planning capabilities on a unified representational foundation

But two caveats must immediately follow: **today's public systems typically cover only a portion of this, and work like pi0.5/pi0.7 is more like gradually expanding this closed loop rather than having completed the unification.** Furthermore, this does not mean every robot foundation model must simultaneously include all of these capabilities -- the more likely future form is a system that can flexibly combine these capabilities on a unified representational foundation.

As I mentioned in the [world model survey](/en/articles/2026-09-01-world-model-h2-review/), "world model" is losing its singular meaning. The addition of VLA makes this picture more complex -- and more interesting.

*Next, I plan to dive deep into Sim-to-Real -- just how wide the deployment gap is from simulation to real robots, and what the current best transfer methods are.*

> **VLA Series complete:** [Part 1: RT-2 to OpenVLA](/en/articles/2026-09-03-vla-deep-dive/) | [Part 2: The pi0 Family](/en/articles/2026-09-05-vla-pi-family/) | [Part 3: VLA and World Models](/en/articles/2026-09-07-vla-world-models/)
