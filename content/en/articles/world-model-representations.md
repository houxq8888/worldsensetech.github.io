---
title: "Four Paradigms of World Model Representations: A Comparative Analysis"
slug: "world-model-representations"
date: 2026-08-10
draft: false
categories: ["World Models"]
tags: ["World Models", "Representations", "Latent State", "3D Structure", "Object-Centric", "RSSM", "Scene Graphs", "NeRF"]
description: "A systematic comparison of four world model representation paradigms — flat vectors, structured 3D, object-centric, and hybrid — analyzing their trade-offs in prediction accuracy, generalization, and downstream task performance."
toc: true
related_articles:
  - world-model-transformer
  - rssm-deep-dive
  - world-model-intro
  - 2026-08-31-world-model-future
  - vla-vs-world-model
  - mujoco-vs-isaac-sim
aliases:
  - /en/articles/world-model-representations.html
---


In a previous post, we compared MuJoCo and Isaac Sim to clarify simulator selection. But regardless of which simulator you use, world models face a more fundamental question: what exactly should be used to represent the "world"?

This may sound abstract, but it directly determines what a world model can and cannot do. It is like choosing the wrong data structure — no matter how clever your algorithms are downstream, you cannot recover.

I categorize current world model representation approaches into four main paradigms. It is worth noting that these paradigms are not strictly mutually exclusive — they correspond to different dimensions of representation: latent space compression is a choice of representation space, visual generation is a choice of output space, explicit geometric modeling is a way to represent world geometry, and object-centric structuring is a way to organize information. More precisely, these four paradigms are not competing model categories but rather four inductive biases in world model design — different representations inject different prior assumptions into the model. In practice, real systems frequently combine them. However, understanding the core logic and limitations of each paradigm is a prerequisite for making sound design choices.

## Paradigm 1: Latent State

Representatives: DreamerV3, TD-MPC2

Core idea: Compress observations (images, sensor data) into a fixed-dimensional vector, then model dynamics and train policies within this latent space.

DreamerV3's RSSM uses a recurrent neural network to maintain a deterministic state (4,096 dimensions) and combines it with discrete stochastic variables (32 categorical variables, each with 64 classes) to represent uncertainty; together they are used to predict the future. TD-MPC2 takes a more aggressive approach — a 512-dimensional latent state, with no requirement for the world model to reconstruct raw observations. Instead, it performs model-predictive control directly in a task-relevant latent space.

What problem does this paradigm solve?

Sample efficiency and computational speed. The latent state is a highly compressed representation — a 64x64x3 image is compressed into a vector of a few thousand dimensions. Running imagination training in this compressed space is orders of magnitude faster than predicting directly in pixel space. DreamerV3 has lower computational requirements compared to pixel-level world models, enabling efficient training with limited compute. TD-MPC2 has also demonstrated high training efficiency across multiple continuous control tasks.

Moreover, latent states are naturally suited for control. The policy network can directly output actions from the latent state, and the world model can perform rollouts in latent space without generating full images. For real-time control scenarios (such as robotic arm manipulation), this efficiency is essential.

Where does it hit a wall?

The latent space is a black box. It typically does not provide explicitly accessible 3D structure — the model may internally encode spatial cues such as depth and position, but this information is difficult to use directly for geometric reasoning. What it learns is "after taking this action, the latent vector changes from A to B," but what spatial information A and B actually contain remains completely opaque.

This leads to a direct consequence: long-horizon predictions tend to drift. Since latent states usually lack explicit geometric consistency constraints, each prediction step can introduce small errors. As these errors accumulate, imagined trajectories gradually diverge from true dynamics. DreamerV3 typically uses a relatively short imagination horizon (e.g., 15 steps), which is an engineering trade-off among model error accumulation, value estimation bias, and computational cost.

Another issue is transfer. If the latent space is primarily learned from single-task data, its state semantics often lack explicit structural constraints, and cross-scenario transfer may require re-adaptation.

## Paradigm 2: Pixel/Video

Representatives: NVIDIA Cosmos, 1X World Model

Core idea: Directly predict future images or video frames. Give the world model a video of the past plus an action sequence, and it generates the video of the future.

Cosmos positions itself as a foundation model for Physical AI, processing visual information through two pathways: one is an autoregressive discrete visual token approach (similar to GPT predicting the next token), and the other is a diffusion model generating continuous frames. Its goal is not only to predict video but also to acquire general physical priors through large-scale visual generation. The 1X World Model follows a similar approach, using a video generation model to predict future frames while also predicting discrete latent state codes.

What problem does this paradigm solve?

Visual fidelity and generality. By modeling directly in pixel space, this approach can leverage the vast amount of video data available on the internet for pretraining. Cosmos is reported to have been trained on millions of hours of video — traditional control-oriented latent state models typically cannot directly utilize video data at this scale, whereas the video generation paradigm is naturally suited to absorbing internet-scale visual data.

Moreover, pixel-level predictions are naturally suited for "verification" — you can directly see the future frames predicted by the world model and judge whether they look reasonable. For non-technical stakeholders (such as clients or managers), watching a predicted video is far more intuitive than observing changes in a latent vector.

Where does it hit a wall?

Slow. Generating a single frame requires a full network forward pass (diffusion models even require multiple iterative steps), making real-time control infeasible. Robotic control typically requires millisecond-level feedback, and the inference speed of current large-scale video generation models generally cannot meet the demands of high-frequency closed-loop control.

Another fundamental issue is that pixel representations typically do not explicitly encode 3D structure. They predict "what the image will look like" rather than "what the world will look like." Although video models may implicitly learn depth, motion, and other 3D information internally, this 3D knowledge is not explicitly represented and cannot be directly used for spatial reasoning: where exactly is the object? How far away is it? Will there be a collision?

There is also a hidden risk: hallucination. Video generation models may "imagine" details that do not exist. For control tasks, such hallucinations can be fatal — the world model says "there is no obstacle ahead," but there actually is one. Therefore, robotic applications need not only generative capability but also reliable uncertainty estimation — knowing what the model "is uncertain about" is just as important as knowing what the model "predicts."

## Paradigm 3: 3D Explicit Structure

Representatives: World Labs Marble, GaussianDream, GWM

Core idea: Represent the world using 3D geometric structures (point clouds, meshes, 3D Gaussian Splatting). The world model maintains an explicit 3D scene representation and predicts how actions change this 3D structure.

Marble generates 3D Gaussian Splatting scenes from images or text, which can be exported as meshes for physics simulation. It is closer to "world generation" than "dynamics prediction" — it addresses "what the world looks like" rather than "how the world will change." It is important to note that a 3D scene representation alone is not yet a complete world model. A world model needs not only to know what the world looks like but also to predict how the world responds to actions. Some recent works (such as GaussianDream and GWM) attempt to combine 3D Gaussians with dynamic prediction in world models, using 3D structure as a supervisory signal to constrain latent space learning. However, these efforts remain in early exploratory stages.

What problem does this paradigm solve?

Spatial understanding. 3D representations naturally carry depth, geometry, and spatial relationship information. The world model knows where objects are (x, y, z), how large they are, and what direction they face. This means it can perform collision detection, spatial reasoning, and trajectory planning — all fundamental requirements for manipulation tasks.

However, it is important to note that geometric representation does not equal physical representation. A 3D scene knowing where objects are does not mean it knows their mass, friction coefficients, rigid-body constraints, or affordances. There is still considerable distance between "seeing" and "manipulating."

Furthermore, 3D structures can interface directly with physics engines. Scenes generated by Marble can be imported into Isaac Sim for physics simulation without manual modeling. For the goal of "rapidly setting up new task environments," this is enormously valuable.

Where does it hit a wall?

Currently, this paradigm primarily addresses "what the world looks like" (static reconstruction) rather than "how the world will change" (dynamic prediction). Marble can generate a 3D scene from a single photo, but ask it "what happens if I push this cup," and it cannot answer. A truly robotics-oriented 3D world model needs to move from static reconstruction to dynamic scene representation — simultaneously modeling geometry, physical properties, and interaction dynamics. This is an unsolved core challenge.

The computational overhead of 3D representations is also non-trivial. Rendering and updating 3D Gaussian Splats are both significantly slower than working with latent state vectors. For scenarios requiring high-frequency control, this is a practical engineering bottleneck.

Additionally, 3D representations require 3D supervisory signals (depth maps, point clouds, camera poses, etc.). These data are not always easy to obtain in real-world scenarios, increasing data costs.

## Paradigm 4: Object-Centric

Representatives: FOCUS, multiple new works at NeurIPS 2025

Core idea: Decompose a scene into individual objects, each with its own independent state representation, and predict overall dynamics through shared or interaction models. Instead of predicting "how the entire image changes," the world model predicts "how each object changes" and how objects influence one another.

For example, recent works such as FOCUS explore representing scenes as object-level latent states and combining spatial representations to model object dynamics. NeurIPS 2025 also features multiple works using Slot Attention-style mechanisms to automatically discover objects in scenes, learn dynamic states for object-level representations, or learn interaction patterns between objects. However, most of these works are still at the proof-of-concept stage and are some distance from practical application.

What problem does this paradigm solve?

Compositional generalization. In theory, if your world model has already learned the dynamics of "a cup" and "a plate" separately, when both appear in a scene, it may be able to directly compose existing knowledge without relearning. This compositional capability is critical for task adaptation: a client's environment may contain all sorts of object combinations, and you cannot train a separate model for every possible combination.

Object-level representations also naturally support attention mechanisms — focusing only on task-relevant objects while ignoring the background. This helps with sample efficiency as well: there is no need to waste compute predicting an unchanging background.

Where does it hit a wall?

Object discovery itself is a hard problem. How do you know how many objects are in a scene? How do you separate objects from the background? How do you persistently track the same object under occlusion? These questions still lack satisfactory solutions in academia.

As the number of objects increases, complexity rises rapidly. If there are 20 objects in a scene, each with its own independent model, both computation and communication overhead become significant.

There is also a fundamental issue: some tasks are not "object-level." Consider liquid manipulation (pouring water) or deformable object manipulation (folding clothes) — the subjects of these tasks are not rigid bodies and are difficult to decompose into independent objects.

## Comparison of the Four Paradigms

| Dimension | Latent State | Pixel/Video | 3D Structure | Object-Centric |
| --- | --- | --- | --- | --- |
| Sample Efficiency | High | Low | Medium | Medium-High |
| Real-Time Control | Strong | Weak | Weak | Strong |
| Spatial Explicitness | Low | Low | Strong | Medium-Strong |
| Data Scale Advantage | Low | Strong | Weak | Medium |
| Compositional Generalization Potential | Medium | Medium | Medium | Strong |
| Physical Consistency | Medium | Low-Medium | Medium-High | Medium-High |
| Data Acquisition Difficulty | Low | Low (internet data) | High (multi-view/depth/calibration) | Medium-High |

Note: "Compositional generalization" here refers to whether the model can reuse existing object, relational, and dynamical regularities, rather than acquiring generalization ability through larger-scale training data. The generalization of the pixel paradigm comes primarily from data scale, not from structured composition.

## Trend: Hybrid Representations


Looking at this table, you will notice one thing: no single paradigm is perfect. Latent states are fast but lack spatial understanding; pixels offer generality but are too slow; 3D provides spatial precision but cannot model dynamics; object-centric offers compositional generalization but discovery is difficult.

So the recent trend is hybridization — combining the strengths of different paradigms.

Latent state + 3D structure. Works such as GaussianDream have made initial explorations: introducing 3D decoding as an auxiliary supervisory signal during training, then reverting to compact latent states at inference time. If this approach matures, it could potentially gain the training signal from 3D structure without sacrificing inference speed. However, it remains in an early validation stage.

Latent state + object decomposition. Works such as FOCUS explore decomposing scenes into object-level latent states and combining spatial representations to model object dynamics. If this approach matures, it could retain the efficiency of latent states while potentially gaining compositional generalization ability.

Pixel pretraining + latent state fine-tuning. First perform pixel-level pretraining on large-scale video data (to acquire general physical intuition), then fine-tune with latent states on specific tasks (to gain efficient control capability). This mirrors the "pretrain + fine-tune" paradigm in the LLM domain.

These hybrid directions are all essentially answering the same question: how to make the latent space no longer "flat" but instead carry meaningful structural information — 3D, object-level, physical.

## Going Further: The Vision of Hierarchical Representations


Hybrid representations naturally lead to a deeper question: is a single level of representation sufficient?

A natural source of engineering inspiration is that the perceptual systems of humans and animals typically exhibit different levels of information processing — when you see a "cup," you first recognize it as an object (semantic layer), know it is on the right side of the table 30 centimeters away from you (spatial layer), expect it to roll if pushed (dynamic layer), while your visual cortex processes color and shape (perceptual layer). Of course, this is not a direct replication of biological mechanisms but rather an engineering structural assumption — using hierarchy to reduce problem complexity.

World models may also need a similar hierarchical architecture. An intuitive vision is:

```

                 Task / Planning
                       ↑
          Object + Semantic Layer
        "What is there? What is the goal?"
                       ↑
          3D Geometry + Physics Layer
        "Where is it? How to interact?"
                       ↑
          Latent Dynamics Layer
        "How will future states change?"
                       ↑
        Sensor / Pixel Interface
        "How to observe? How to generate?"

```

Top layer: Objects and semantic structure. Responsible for object discovery, semantic understanding, and task goal decomposition. This layer answers "what is in the scene" and "what does the task require." It can be implemented with object-level representations or symbolic scene graphs.

Middle layer: 3D spatial relationships. Maintains explicit 3D geometry — where objects are, how large they are, their orientation, and whether collisions will occur. This layer provides the foundation for spatial reasoning and trajectory planning. Technologies such as 3D Gaussian Splatting and neural radiance fields may play a role at this layer.

Bottom layer: Latent dynamics. Models state transitions and policies in compressed space. This layer is responsible for efficient imagination training and real-time control. DreamerV3's RSSM and TD-MPC2's latent space models are well-suited for this layer.

Perception and generation interface layer: Pixels and video. Responsible for encoding sensor inputs into internal representations and rendering internal representations into visible images for visual supervision, human-machine interaction, or interfacing with existing vision-language models.

This hierarchical architecture is currently just a vision — academia has not yet produced a complete implementation. But it offers an interesting perspective: the four representation paradigms may not be a matter of "which one to choose" but rather "which layer to place them in." Each layer uses the most appropriate representation, and layers collaborate through information passing.

If this direction proves valid, world model design shifts from "choosing a single representation" to "designing inter-layer information flow" — which may be the core challenge of the next stage.

## Implications for Task Adaptation


Returning to our core question: enabling robots to quickly adapt to new tasks. The choice of representation directly affects "adaptation speed."

If the world model uses pure latent states (DreamerV3), switching to a new scenario means relearning the meanings of latent variables. Prior experience is difficult to transfer because the latent space lacks structured semantics.

If the world model uses object-level representations, switching scenarios only requires replacing or composing object models. Having seen the dynamics of "a cup," when moving to a "cup + plate" scenario, the cup model can potentially be reused. However, it is important to note that object-level reuse does not mean complete reuse — the same cup, empty versus full of water, has entirely different dynamics, and a plastic cup versus a ceramic cup has different interaction modes. The robot still needs to learn physical properties, interaction modes, and task-relevant affordances of objects.

If the world model has 3D structure, setting up a new environment can directly use 3D scanning or reconstruction for initialization, without needing to learn scene geometry from scratch.

Therefore, representation is not merely a technical choice — it directly impacts the potential for task adaptation. The goal is to reduce the new task adaptation cycle from days to hours, but achieving this goal still requires joint breakthroughs in representation, data, training methods, and robotic systems. Choosing the right representation direction is a necessary condition, but not a sufficient one. What truly determines a robot's generalization ability is not just model scale, but whether the model has formed a reusable, structured world representation internally.

## Summary


World model representations follow four paradigms: latent state (fast but lacking explicit structure), pixel/video (general but computationally intensive), 3D explicit structure (spatially precise but dynamic modeling is still early), and object-centric (compositional generalization potential but object discovery remains difficult). No single paradigm can solve all problems on its own.

The current trend is hybridization and hierarchization — injecting 3D structure or object-level decomposition into latent states, or even building hierarchical representation architectures where different paradigms serve their roles at different levels. This direction is still in its early stages, but it may be the key to world models achieving true generality.

Understanding the differences and complementary relationships among these representation paradigms is the foundation for understanding the technical evolution of world models. Subsequent articles will dive deeper into the specific implementations and improvement directions of each paradigm within this framework.

For robotics, the next step in the Scaling Law may not be simply increasing parameter counts, but rather enabling the model to form a more structured, composable, and transferable internal representation of the world.
