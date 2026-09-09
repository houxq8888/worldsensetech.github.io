---
title: '具身智能 Sim-to-Real 方法论（三）：评估体系、决策矩阵与最小可执行 Protocol'
slug: "2026-09-12-sim-to-real-evaluation-protocol"
date: 2026-09-12
draft: false
categories: ["具身智能", "训练方法"]
tags: ["具身智能", "Sim-to-Real", "评估", "Sim Utility", "Ranking Correlation", "Selection Regret", "Allocation Protocol", "Stopping Rule", "组合决策"]
description: '三部曲-评估与落地篇。Sim fidelity 不只有一个维度——数值预测、排序准确性、选出的 policy 好不好、三维不能互替；本文给出 gap x 可建模性 x 真机预算的决策矩阵、stopping rule 三类、以及一套 6 步最小可执行 Allocation Protocol。理论 spine 见 Part 1、方法谱系见 Part 2。'
toc: true
related_articles:
  - 2026-09-10-sim-to-real-methodology
  - 2026-09-11-sim-to-real-intervention-lenses
  - 2026-09-13-tactile-force-sensing
  - 2026-09-14-multimodal-fusion-interface
  - 2026-09-09-robot-data-scaling
  - 2026-09-08-data-and-training-recipes
---


> **前情速览**（承上篇）：本文是 Sim-to-Real 方法论三部曲的第三篇。核心框架已在 [Part 1](/zh/articles/2026-09-10-sim-to-real-methodology/) 建立——reality gap 被重述成四元组 $(\pi, \mathcal{E}_{\mathrm{shared}}, M_{\mathrm{sim}}, M_{\mathrm{real}})$ 下的 downstream discrepancy、误差预算被写成 state-conditioned 序贯分配、决策单位是 intervention action $m_t^* = \arg\max\, Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t)$。全文符号约定：$s_t = (b_t, \pi_t, q_t, h_t)$（state）、$\mathcal{M}_t^{\mathrm{feasible}}$（feasible action set）、$MV$（efficiency readout、非 decision rule）、$\mathrm{CVU}$（one-step continuation uplift）。如未读上篇、建议先看。

## 评估：你怎么知道自己把 gap 补好了？

本文 claim 挂在**三级证据层**上——$\boxed{\text{A: mechanism}\quad \text{B: policy-response}\quad \text{C: deployment}}$：A 是 friction ID / calibration / latency measurement、B 是 $\hat S_k^{\mathrm{int}}$ / ablation / 有限差分 attribution、C 是真机 $\Delta J$ / $Q_{\lambda_t}$ / $MV$ / sim ranking utility。**三层是证据层级、非固定执行顺序**——诊断可循环（真机 failure → 怀疑 latency → 回测 A）；**不能互相替代**——SI 拟合属 A、不等于 C deployment 改善。

危险的做法是只在 sim benchmark 报性能。可信评估至少：

- 报 **zero-shot** 与 **few-shot / N-shot** 曲线；
- 用一组 **held-out hardware / object / contact / environmental regimes**；
- 明确声明 sim 与 real evaluation distribution 是否一致；
- **不只报均值**：mean ± CI、多 seeds、paired evaluation；
- **安全失败单独统计**：$X\sim\mathrm{Bin}(n,p),\;X=0$ 只能给 $p$ 的 UCB（Clopper–Pearson）。

顺着"sim 是真实世界的代理"、还有个比"数值对齐"更本质的问题：**sim 能否正确预测"哪个 policy 更好"？**

一个**概念性例子**（数值不代表实验结果）：

| Policy | Sim | Real |
| --- | ---: | ---: |
| A | 90 | 50 |
| B | 80 | 70 |
| C | 70 | 65 |

在 sim 上 $A > B > C$、真机却是 $B > C > A$。这时 simulator **失去 model-selection utility**——你会用它挑出最差的 policy。故 **simulator 用于 policy / model selection** 时应同时看排序相关性 $\rho_{\mathrm{rank}} = \mathrm{Spearman}(J_{\mathrm{sim}}(\pi_i), J_{\mathrm{real}}(\pi_i))$ 与 selection regret：

$$\pi_{\mathrm{sim}} = \operatorname*{arg\,max}_{\pi \in \Pi} J_{\mathrm{sim}}(\pi), \qquad R_{\mathrm{select}} = J_{\mathrm{real}}\big(\pi^{*}_{\mathrm{real}}\big) - J_{\mathrm{real}}\big(\pi_{\mathrm{sim}}\big)$$

**在更大的 policy pool 上**、即使 $\rho_{\mathrm{rank}} = 0.95$、top-1 仍可能被选错、灾难不减；反过来 $\rho_{\mathrm{rank}} = 0.7$、若 top-1 基本不出错、对"选一个能部署的 policy"就够用（**注意**：此处 $\rho_{\mathrm{rank}}$ 的直觉例子是更大 policy 集合上的相关性、而非上文 $A/B/C$ 三个 policy 的统计量——$n = 3$ 时 Spearman 只能取有限离散值、$0.95$ 那样的连续数字不适用）。**sim fidelity 是 task-of-use dependent、不是 absolute property**——换用途（pretrain / exploration / curriculum / safety filter）"哪些误差重要"整个变一遍。$\pi^*_{\mathrm{real}}$ 不可得、$R_{\mathrm{select}}$ 与 real-domain learning gap 一样都是 oracle-defined 量、实际用 $J_{\mathrm{real}}(\pi_{\mathrm{best\text{-}validated}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$ 作 **validated-best observed proxy**（$\pi_{\mathrm{best\text{-}validated}}$ 在独立 audit / held-out 切片上评估最优、不是 noisy eval 里的 argmax、避免 winner's curse）。

**更重要的一层**：真实项目通常不要求 sim 精确排序所有 policy、只要求把值得上真机的候选压到可接受集合——**top-$k$ recall**、**regret@k** 应与 ranking 同级。**警惕 adaptive selection bias**：sim 若被 adaptive filter policy、用同一批被选候选反过来评 sim 造成 self-confirming 循环。**维护两个 pool**——$\Pi_{\mathrm{adapt}}$ 参与 training / selection、$\Pi_{\mathrm{audit}}$ 只做 held-out evaluation。**held-out set 并非无限次免疫的**：长期项目应保留 audit slice 或定期 refresh evaluation set、避免 adaptive experimentation 过拟合固定真机评测集。

至此、**allocation framework 的一个 corollary**：**sim utility 不是单一属性、是三个不能互替的维度；sim 内部自洽、低 prediction loss 或高 training reward 不能单独证明 downstream utility——必须由独立 real evidence 验证**——

| Simulator utility 维度 | 典型 metric |
| --- | --- |
| 数值预测准不准（absolute error / calibration） | MAE / RMSE $\mathbb{E}\big[|J_{\mathrm{sim}}(\pi) - J_{\mathrm{real}}(\pi)|\big]$、calibration curve、prediction interval coverage |
| 排序准不准（ranking） | Spearman $\rho_{\mathrm{rank}}$、Kendall $\tau$、top-k recall、regret@k |
| 选出的 policy 好不好（decision quality） | $R_{\mathrm{select}} = J_{\mathrm{real}}(\pi^{*}_{\mathrm{real}}) - J_{\mathrm{real}}(\pi_{\mathrm{sim}})$（实际用 **best-validated proxy**、独立 audit 切片上的 argmax、非 noisy eval 上的 argmax） |

一个 sim 可校得很准却选错 policy、也可数值全错但排序稳、regret 小——三维不能互替，**三类 metric 不仅量纲不同、优化目标也不同、因此不存在一个自然的 universal scalar simulator score**。$U_{\mathrm{sim}}$ 不该写成抽象标量、应按用途**上标索引**：$U_{\mathrm{sim}}^{(u)}$、$u \in \{\text{pretrain},\ \text{selection},\ \text{exploration},\ \text{curriculum},\ \text{safety}\}$。评 fidelity 要相对**候选 policy family** 与用途：$U_{\mathrm{sim}}^{(u)}(\cdot \mid \Pi_{\mathrm{candidate}},\ p_{\mathrm{eval}}^{\mathrm{real}})$。**更关键的是**：simulator 的最终 utility 不只看 $U_{\mathrm{sim}}^{(u)}$ 本身、还看它在 downstream allocation 里引发的期望价值——一个数值预测不够准、但能稳定做 candidate screening 的 sim、其 downstream allocation utility 可能仍很高。

## 组合与决策，以及一个常被回避的问题

真实项目更有用的是 **gap × 可建模性 × 真机预算** 矩阵：

| Gap | 可参数化 / 可辨识？ | Real data | 更自然的候选（最终仍由 state-conditioned $\Delta J,\Delta C,\mathrm{CVU}$ 决定） |
| --- | --- | ---: | --- |
| low-dimensional dynamics bias | 高 | 少 | SI |
| parameterizable dynamics uncertainty | 中 | 少 | posterior-guided DR / Bayesian SI → DR |
| dynamics residual | 低（但有结构） | 中 | Residual learning |
| visual appearance | 高 | 无 / 少 | DA / DR（候选） |
| actuator latency | 高 | 少 | SI + DR |
| unobserved rare tail、可被 model family 表示 | 低 | 少 | targeted simulation / DR |
| unknown long-tail，sim 生成不可信 | 低 | 中 | real data |
| model class 不确定 | 低 | 多 | learned world model（若 real 稀缺则先 physics prior + residual / DR） |
| mixed | mixed | mixed | co-training candidate（需先验证正迁移条件） |

**限定词不能省**：model-class uncertainty 下 SI 与 DR 未必适用、得先落到 residual / WM / 真机。"co-training 兜底"与 allocation 冲突——sim 质量差时可能负迁移。

常见组合 **SI → DR → DA → co-training / fine-tune**：**箭头只是示意、非固定 workflow**、顺序由主导 gap 与边际效用决定。**当 sim 已有较强 coverage、主要未知来自 model misspecification**、real data 的高价值用途是**发现 sim 未建模的 failure mode** 让 sim 放大；**若 deployment distribution 已相当固定**、real data 也可能主要承担直接 adaptation / imitation、不必先走 discovery / amplify——

$$\text{discover real tail} \rightarrow \text{identify structure} \rightarrow \text{amplify} \rightarrow \text{real validation}$$

即 **real 发现、sim 放大、real 再验证**。**这条 chain 成立有硬前提**：发现的 failure mode 能被当前 model class / learned surrogate 以可信方式表示；否则 real discover 之后应直接转向 richer model / world model / 追加 real data、而不是强把不可表示的 tail 塞进 sim 放大（这正是 model-class uncertainty 那一段的具体化）。

**什么时候最优解其实是"不做 sim-to-real"？**
- **真机数据已便宜到 $C_{\mathrm{SI}}+C_{\mathrm{DR}} > C_{\mathrm{real}}^{\mathrm{effective}}$**（比较的是 horizon 内 cumulative value / cost）。
- **仿真器 model class 本身就差**（软体 / 流体 / 复杂接触）——不如 WM 或真机数据。
- **部署分布非常固定**——少量 targeted real FT 更划算。
- **sim 不提供 unique coverage / safety / exploration / counterfactual access**——$U_{\mathrm{sim}}^{\mathrm{downstream}} < C_{\mathrm{sim}}^{\mathrm{effective}}$。

**stopping rule 三类**：**(a) local net-value stop（economic stop）**——$\max_{m \in \mathcal{M}_t^{\mathrm{feasible}}} Q_{\lambda_t}^{\mathrm{perf+CVU}}(m \mid s_t) \le 0$（local one-step stop、非全局最优——已知强互补 portfolio 应作为 candidate 一并评估）；**(b) continuation-value stop**——**best remaining positive continuation uplift 已经接近零**：$\max_{m \in \mathcal{M}_t^{\mathrm{feasible}}} \mathrm{CVU}(m \mid s_t) \le \varepsilon$（$\varepsilon$ 为小正阈值；**因 $\mathrm{CVU}$ 可正可负、"expected CVU 接近 0" 表述会漏掉"当前最优 CVU 严重为负、必须立即停"的情形、$\max$ 而非 expectation 才是正确的停止判据**）；**(c) safety / feasibility stop**——剩余 candidate 全在 feasible 集外。任一触发即停。

## 一个最小可执行的 Sim-to-Real Allocation Protocol

框架不落到"明天项目组怎么跑"、就还是聪明的 framing。以下 6 步是**最小可执行版**、可跳过、但跳之前要说清对本项目 no-op 的原因。

**Step 1 — 固定 evaluation。** 锁死 task / initial-state 分布 / horizon / success metric / safety threshold / policy interface（obs + action schema + control freq）。**若 $\pi$ stochastic（$a_t \sim \pi_\theta(\cdot \mid o_t)$）、$J(\pi)$ 应理解成 evaluation protocol 下对 policy / reset / hardware randomness 的期望**、用 repeated runs / block evaluation 估计。**没这一步、后面 $\Delta J$ 没有共同基准**。

**Step 2 — 建 held-out real evaluation set。** 真机 eval 集与训练数据**必须分开**、覆盖 held-out hardware / calibration / object / 场景切片。用训练数据 evaluate、$\widehat{\Delta J}$ 一定 optimistic。**但 eval 结果可进入 allocator 的 belief update**：$\mathcal{D}_t$ = "allocator 在 step $t$ 可获得的全部 evidence"、包括 $D_{\mathrm{eval}}$ 反馈的 failure mode 与 uncertainty 变化；"不参与 training" 与 "参与 posterior update" 是两件事、不冲突。

**Step 3 — 列 mismatch hypotheses（可 falsify）。**

| Hypothesis | Evidence | Belief | 候选 intervention |
| --- | --- | ---: | --- |
| friction $\mu$ 偏低 | contact slip | med | SI + DR |
| actuator latency 未建模 | 高频振荡 | high | SI + timing |
| camera extrinsics 偏 | grasp offset | high | Calibration / DA |
| contact model 错 | 柔性物体 OOD 失败 | low | Residual / WM |

每条 hypothesis **必须能被具体实验否证**、写不出否证条件的先剔除。

**Step 4 — one-time initial calibration pilot**（Step 5 才进入 sequential adaptive allocation）。**对会直接改变当前 policy 的 action 估计 immediate effect distribution**（$\mu_{\Delta J,t}(m)$ 与其 spread、Bayesian 实现下即 posterior、频率派实现下即 CI）、**对 diagnosis / model-refresh action 主要估计 evidence quality / continuation uplift distribution**（其 immediate $\Delta J \equiv 0$、不存在 meaningful 的 immediate effect distribution 可估、experimental target 是 $\mathrm{CVU}$ 相关的 evidence 与后验改善量）；不预设固定样本数。**$\widehat{\Delta J}_t(m) = J_{\mathrm{real}}(\pi_t^{m}) - J_{\mathrm{real}}(\pi_t^{\mathrm{control}})$**——control 承担相同 training 步数、**相同 elapsed time（覆盖机器人温度 / 电量 / wear 等 background drift）**、相同 training seed（真机硬件扰动本身没有"seed"可对齐、只能靠 matched evaluation block 逼近），只关掉本 intervention；**diagnostic-only action 与"只更新 simulator / surrogate、暂不重新训练当前 policy 的 model-refresh action"的 $\widehat{\Delta J}_t \equiv 0$、其价值在本文一步近似中统一通过 $\mathrm{CVU}$ 汇总（unified continuation surrogate）、不再单独定义额外 reward channel**。**matched / paired / block 化评估**：同 training seed、并在可行时采用 matched evaluation blocks / hardware conditions、同 held-out slice；漂移系统记录 hardware state。**单 intervention matched control 识别的是 incremental effect relative to the current protocol、不识别高阶 interaction effect；组合 action（例如 SI+DR 或 SI+WM refresh）需作为独立 candidate 做 matched comparison**，否则 synergy / conflict 无法从数据里分离。

**Step 5 — sequential adaptive allocation**：$m_t^* = \arg\max_{m\in\mathcal{M}_t^{\mathrm{feasible}}(s_t)} Q_{\lambda_t}^{\mathrm{perf+CVU}}(m\mid s_t)$——$\lambda_t$ 是 resource-weight estimate、objective 与 local score 必须一致；cost 与预算同时 state-conditioned（$\Delta C(m\mid s_t)$、$b_{t+1} = b_t - \Delta C(m_t^*\mid s_t)$）。Execution-level safety 走 $\alpha_{\mathrm{exec}}$ gate、deployment-level safety 走 $\alpha_{\mathrm{deploy}}$ terminal 约束、都不进 cost。

**Step 6 — real evaluation → posterior update → 回到 Step 3。** 更新 $\mathcal{D}_t \rightarrow \mathcal{D}_{t+1}$、重估 $\lambda_t$、淘汰否证 hypothesis、新失败补入表。**最易跳过、最关键**——没 posterior update、流程退化为静态 checklist。

**定位**：最低落地版——小团队可合并 Step 3/4、大团队可加 portfolio opt。6 步都要写下来。
---

> **上篇（Part 2）**：[四把手术刀与两条新路线](/zh/articles/2026-09-11-sim-to-real-intervention-lenses/) -- SI / DR / DA / FT / WM / Co-training 详解。
>
> **上上篇（Part 1）**：[Sim-to-Real 方法论-理论篇](/zh/articles/2026-09-10-sim-to-real-methodology/) -- reality gap 与 allocation 框架。

*本篇是"具身智能 Sim-to-Real 方法论"三部曲-评估与落地篇。理论框架见 Part 1、方法谱系见 Part 2。*
