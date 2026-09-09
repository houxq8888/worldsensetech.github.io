---
title: '具身智能 Sim-to-Real 方法论（二）：四把手术刀与两条改写问题的新路线'
slug: "2026-09-11-sim-to-real-intervention-lenses"
date: 2026-09-11
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "Sim-to-Real", "System Identification", "Domain Randomization", "可微仿真", "Residual Physics", "Domain Adaptation", "Real-world Fine-tuning", "世界模型", "Co-training"]
description: '三部曲-方法篇。把 SI / DR / DA / FT 读成四个可组合的 intervention lens（Model x Data x Representation x Optimization）、不是四选一；再加 World Model 与 Sim-and-Real Co-training 两条松动 environment-generating-process 假设的新路线。每一把刀处理什么 mismatch、机制与失效边界。理论 spine 见 Part 1、评估与落地见 Part 3。'
toc: true
related_articles:
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-12-sim-to-real-evaluation-protocol
  - 2026-09-13-tactile-force-sensing
  - 2026-09-14-multimodal-fusion-interface
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
---


> **前情速览**（承上篇）：本文是 Sim-to-Real 方法论三部曲的第二篇。核心框架已在 [Part 1](/zh/articles/2026-09-10-sim-to-real-methodology/) 建立——reality gap 被重述成四元组 $(\pi, \mathcal{E}_{\mathrm{shared}}, M_{\mathrm{sim}}, M_{\mathrm{real}})$ 下的 downstream discrepancy、误差预算被写成 state-conditioned 序贯分配、决策单位是 intervention action $m_t^* = \arg\max\, Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t)$。全文符号约定：$s_t = (b_t, \pi_t, q_t, h_t)$（state）、$\mathcal{M}_t^{\mathrm{feasible}}$（feasible action set）、$MV$（efficiency readout、非 decision rule）、$\mathrm{CVU}$（one-step continuation uplift）。如未读上篇、建议先看。

## 四个 intervention lenses（可组合的分析维度）

SI / DR / DA / FT **非同一抽象层级**——SI 是 model calibration、DR 是 distribution manipulation、DA 是 representation alignment、FT 是 optimization strategy——并排成"四类方法"会误导四选一、其实是**四个可组合的 intervention lens**（本文 analytical decomposition、非领域公认 ontology）：

$$\boxed{\text{Model} \times \text{Data} \times \text{Representation} \times \text{Optimization}}$$

"$\times$" 是组合空间、非正交——DR 触及 Model / Observation / Distribution、DA 可发生在多层。

选工具标准是**"点估计 → 后验 → 鲁棒随机化"连续谱**。SI 可以做 point calibration、也可以进一步给出 posterior；下面先写 point estimate：

$$\hat\phi \;=\; \operatorname*{arg\,min}_{\phi}\; \mathcal{L}_{\mathrm{ID}}\big(D_{\mathrm{real}},\ f_{\mathrm{sim}}(\cdot\,;\,\phi)\big)$$

$\mathcal{L}_{\mathrm{ID}}$ 可取 trajectory prediction / one-step transition error / force-torque residual / likelihood——**经典 SI 的目标通常是参数估计或 transition / observation prediction error minimization、而不必显式做 trajectory-distribution matching**。SI 处理的是**可参数化的 model mismatch**：动力学残差、接触/摩擦系数、延迟、相机外参等——若 gap 落在 model class 之外（未建模 long tail、语义级视觉差），SI 就力不从心、需换 DR / DA / WM。

| mismatch 的性质 | 更自然的工具 |
| --- | --- |
| 可参数化 + 可辨识 | System Identification（point estimate $\hat\phi$） |
| 可参数化但只能给出不确定性 | Bayesian / posterior SI → posterior-guided DR |
| 可参数化但难辨识 / uncertainty 大 | Domain Randomization |
| 难以由低维物理参数充分表达、但有结构化 residual | Residual learning |
| observation / appearance mismatch | Domain Adaptation |
| policy 在目标域仍有 systematic residual | Fine-tuning |

关键：**"不能精确辨识" ≠ "完全不知道"**——拿到 $p(\phi\mid D_{\mathrm{real}})$、最自然动作 $\phi\sim p(\phi\mid D)$ 做 posterior-guided DR、**SI 与 DR 是连续谱两端**。

### Axis A — Model：system identification、可微仿真与 residual physics

这条轴处理 $\Delta_{\mathrm{model}}$、三层次常被混淆：

$$y_t \;=\; \underbrace{g_{\mathrm{physics}}(x_t,a_t;\phi)}_{\text{可参数化的物理}} \;+\; \underbrace{r_\theta\big(\psi(x_t,a_t)\big)}_{\text{残差}} \;+\; \epsilon_t$$

**这只是 representative parameterization**——$y_t$ 可为 $x_{t+1}$、contact impulse、acceleration、deformation field 或其他 observable、$\psi$ 是 residual 的 input view；additive state-transition form 是一种 parameterization assumption、部分动力学更自然的 residual 是加在 acceleration 或 latent dynamics 上、而非 observable 本身。

- **可微仿真**解决 optimization interface、不解决 model class correctness；
- **Residual physics** 保留 prior、有限修正；
- **Full-learned dynamics** 处理 physics 不适用的场景。

**可微性在 discontinuous contact / complementarity / friction cone 切换处面临 gradient instability 或 ill-defined gradients 的风险**（不一定表现为 vanishing、更常见是不连续或高方差梯度）。工程上 soft-contact / smooth relaxation 是常见的处理手段。工程判据：physics 结构基本正确、参数或边界不准时、可微仿真性价比最高。

Residual physics 一个常见的适用区间是 $f_{\mathrm{physics}}$ 已提供**结构性归纳偏置**、residual 只在目标分布上有限修正的场景。风险：sim 有 residual 补偿后看似好、到 OOD 失效——**residual model 的 valid domain 需与 deployment condition 对齐**。可微仿真在 contact-rich 场景受 contact mode switches / complementarity constraints 带来的非光滑与梯度不稳定问题掣肘；在 physics 结构基本正确、残差相对局域的条件下，可微仿真通常更值得优先评估。

### Axis B — Data distribution：domain randomization 及其家族

这条轴让 policy 对一族参数 $\{\phi\}$ 都稳健、不追求逼近最准 $p_{\mathrm{real}}$。**Tobin et al.（1703.06907）是现代深度视觉 / 机器人 sim-to-real 文献中的经典代表性起点**（domain randomization 思想本身更早、此处指其在端到端视觉 policy transfer 里的代表性位置）。

**DR 非"隐式 ensemble"**——训练的是单个共享 $\pi_\theta$、目标是：

$$\max_{\theta}\; \mathbb{E}_{\phi \sim p(\phi)}\big[J(\pi_\theta;\phi)\big]$$

更准确：**DR 是对一族环境模型做 population-level 优化**、risk-neutral average-case baseline；worst-case 可写 $\max_\theta\min_\phi$。过度 DR 让 policy 过于保守、牺牲 performance。

**DR 非选 scalar range、而是设计 joint distribution**——**当真实参数本身存在显著 joint dependency 时**、$p(\phi_1,\phi_2)\neq p(\phi_1)p(\phi_2)$、independent sampling 会把有限 sampling budget 分配到大量低 deployment relevance 或物理不一致组合；若真实参数本就近似独立、independent DR 反而是合理近似。**correlated / adversarial curriculum** 是 dependency 存在时的对策。

### Axis C — Observation / Representation：domain adaptation 与观测翻译

处理 $\Delta_{\mathrm{obs}}$。DA 可发生在 input / feature / latent / policy / dynamics 多层。机制包括 feature-level adapters、latent alignment、RCAN (1812.07252)。**不把 DA 压成 image translation**。边界：camera intrinsics / temporal sync 更适合 calibration、非 DA。

### Axis D — Optimization / adaptation：真机微调

这条轴**是 adaptation operator**：直接在目标域继续优化。可作前三轴收尾、也可作早期诊断（少量 FT 暴露哪些 mismatch 最伤 deployment）。

- **Offline / imitation：** $D_{\mathrm{real}} \to \theta$、主要成本是**采集**。
- **Online RL：** $\pi_\theta \to a \to$ 真实 transition $\to \theta'$、主要成本是**交互 + 安全 + 磨损 + 探索**。

比较不能只看最终 success rate、还要看**达目标所需真机交互预算**。粗略指标：

$$\eta_{\mathrm{real}} \;=\; \frac{\Delta J_{\mathrm{real}}}{\text{robot-hours}} \qquad \text{或}\qquad \frac{\Delta J_{\mathrm{real}}}{N_{\mathrm{real}}}$$

但只是**粗略指标**：依赖 baseline、非真 marginal efficiency。真正该看 learning curve / AULC / 每 100 条轨迹的边际收益

$$MV_{\mathrm{real}} \;\approx\; \frac{J(N+\Delta N)-J(N)}{\Delta N}$$

——这才与全文 $MV$ 框架接上。风险不止灾难性遗忘、更常见是**分布收窄**——真机数据比 sim 窄得多、微调后目标切片更好但鲁棒性反降、**generalization 换 specialization**；$MV_{\mathrm{real}}(N)$ **不保证始终为正**、**FT 本身可进入负边际收益区间**。

## 两条松动 environment-generating-process 假设的新路线

上面四条轴共享一隐含前提：经典 framing 把 simulator / real environment 视为**两个给定的 environment-generating processes**（对应分布 $p_{\mathrm{sim}}$、$p_{\mathrm{real}}$）。下面两条路线恰在松动这个前提——非"第五第六种技巧"、是整个问题的 reformulation：**前四条 lens 改变 intervention、WM 与 co-training 改变的是 intervention 所作用的 underlying training substrate**、不塞回同一 taxonomy。

### World model：不是取消 simulator，而是换掉 simulator 的来源

**本文 lens**：本节把 world model 读作"model source replacement"的 reformulation、只挑"相对 physics-sim 换掉 model 来源与 inductive bias"这个切面——不是 world model 的标准定义、也不声称这是唯一读法。

[数据 scaling 下篇](/zh/articles/2026-09-09-robot-data-scaling/)讨论过 world model 与 data utility。放进 sim-to-real 语境先纠正定位误读：**world model 不天然属于 sim-to-real**——两条路线 causal direction 不同：

```
Physics-sim route：  hand-designed dynamics  → train / optimize → deploy real
Learned-model route：interaction data → learned dynamics → imagine → optimize
```

**interaction data 可来自 real / sim 或混合**——learned-model route ≠ real-only。

需要说准：WM **并未取消 sim**、仍在做 simulation / imagination、只是 predictive model 是学出来的、更精确表述是**改变 predictive model 的来源与 inductive bias**：

$$\text{model source} = \text{physics prior} + \text{learned dynamics} + \text{data}$$

三者可 hybrid、不必是 $f_{\mathrm{hand}} \rightarrow f_{\mathrm{learned}}$ 的二元替换。Dreamer（1912.01603）、TD-MPC2（2310.16828）体现这条路——**人工 sim 的 model bias 大到不值得先修**时、WM 提供的是问题本身的改写。DayDreamer（2206.14176）常被误读成"sim 预训练 → real 微调"、更准是展示 **real-interaction-driven 实验路线**。**不依赖手工 sim ≠ model-free**、WM 仍吃假设、只是把 inductive bias 从显式 physics 移到 learned model。

诚实边界：contact-rich / long-tail 场景学到的 model 常在 OOD 给出很自信也很错的想象。**WM net value = predictive utility − model uncertainty risk**——uncertainty 必须进入一个 risk-aware decision layer、按部署需求可选 **hard feasibility gate**（$\Pr(\text{model-induced unsafe}) \le \alpha$）或 **soft risk penalty**（$U_{\mathrm{WM}} = U_{\mathrm{prediction}} - \gamma R_{\mathrm{model}}$）；只有 safety-critical deployment 才更适合前者。

### Sim-and-real co-training：把"迁移"重述成 data mixture

Maddukuri et al.（RSS 2025, 2503.24361）的 Sim-and-Real Co-Training 是务实方向。**论文报告**：sim + real 混合采样、两平台六视觉操作任务、相对**real-only baseline** 观测到**约 37.9% aggregate relative improvement**（across 6 tasks / 2 embodiments）——是**跨任务归一化的 relative lift**、非绝对百分点；引用务必带 baseline 与 aggregation 定义。不做单向迁移、而是一个 recipe 决定比例与调度。

读成 **data-mixture**——$p_{\mathrm{train}}=\alpha_{\mathrm{mix}} p_{\mathrm{sim}}+(1-\alpha_{\mathrm{mix}}) p_{\mathrm{real}}$；$\alpha_{\mathrm{sampling}} \neq \alpha_{\mathrm{effective}}$。Mechanistic 分析（Lei et al., 2604.13645）指出在该 generative robot policy 设置中 mixture 诱发 structured representation alignment——**paper-specific、不外推为 universal**。
---

> **上篇（Part 1）**：[Sim-to-Real 方法论-理论篇](/zh/articles/2026-09-10-sim-to-real-methodology/) -- reality gap 四元组定义与 L1-L5 spine。
>
> **下一篇（Part 3）**：[你怎么知道 gap 补好了](/zh/articles/2026-09-12-sim-to-real-evaluation-protocol/) -- 评估体系与最小可执行 Protocol。

*本篇是"具身智能 Sim-to-Real 方法论"三部曲-方法篇。理论框架见 Part 1、评估与落地见 Part 3。*
