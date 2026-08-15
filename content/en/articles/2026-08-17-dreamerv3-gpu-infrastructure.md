---
title: "DreamerV3 GPU Infrastructure: Cloud vs Self-Built Cost Analysis"
slug: "2026-08-17-dreamerv3-gpu-infrastructure"
date: 2026-08-17
draft: false
categories: ["Infrastructure"]
tags: ["GPU", "DreamerV3", "AutoDL", "RTX 5090D", "Cost Analysis"]
description: "DreamerV3 GPU Infrastructure: Cloud vs Self-Built Cost Analysis - WorldSense Tech Blog"
toc: true
aliases:
  - /en/articles/2026-08-17-dreamerv3-gpu-infrastructure.html
---


In previous articles, we covered DreamerV3 environment setup and training tips. But there's a more fundamental question that often gets overlooked: what hardware are you running on?


This question seems simple, but it directly affects both your research pace and your wallet. I spent a month on AutoDL cloud GPUs, then a week planning a self-built workstation — and even received a ¥36,000 turnkey quote from a vendor. This article shares the entire process and cost breakdown, in the hope of giving anyone wrestling with the "rent vs. buy" decision a useful reference.

## TL;DR


If you're confident you'll sustain high GPU utilization over the long term, a self-built workstation deserves a serious cost analysis. Based on my actual usage at ¥2.58/GPU-hour, the simple static break-even point at 12 hours/day is roughly 29 months, and at 18 hours/day about 19 months. A purely financial calculation doesn't mean "buying is always cheaper than renting" — 12 hours is not a threshold where "buying immediately saves money." But in high-frequency research scenarios, the advantages of a local machine in wait times, environment stability, and data management may carry even greater value.


But "paying off the investment" isn't the only consideration. A self-built workstation has another advantage cloud GPUs can't offer: always available, data stays local, no queuing, no worrying about shutting down to save money. For research that demands high-frequency iteration, this difference in experience is enormous.

## Phase 1: Actual Spending on AutoDL Cloud GPUs


My initial choice was AutoDL — one of the most commonly used GPU cloud platforms in China, with hourly billing, a wide range of machine types, and no environment maintenance required. For getting started with DreamerV3 experiments, this was the most sensible option.

### Actual Costs


The AutoDL instance I actually used worked out to approximately ¥2.58/GPU-hour (RTX 4090D model). I occasionally encountered lower short-term promotional rates (around ¥1.67/h), but the monthly/yearly estimates below use the ¥2.58/h regular price.


Two days cost around ¥100. That doesn't sound like much, but when you extrapolate:
    GPU daily runtime Monthly cost Yearly cost     4 hours/day (light) ~¥310 ~¥3,764   8 hours/day (moderate) ~¥619 ~¥7,530   12 hours/day (heavy) ~¥929 ~¥11,299   18 hours/day (extreme) ~¥1,393 ~¥16,943    

Note this only covers GPU rental — it doesn't include storage costs (AutoDL charges separately for system and data disks) or network transfer costs.

### Real Pain Points of Cloud GPUs


Beyond the money, there are several practical experience issues:


- Fear of shutting down. Once you've occupied a cloud server, you don't dare shut it off — reopening it might mean the model you need isn't available and you have to wait in a queue. But as long as it stays on, even when you're not training, you're being charged continuously. This "holding it but not daring to use it, not using it but still burning money" situation is mentally draining.
- Friction in remote environment management. Without using `tmux`, `screen`, or task scheduling mechanisms, SSH disconnections and instance restarts can make training and environment configuration troublesome. Especially during initial CUDA/JAX setup, the feedback and debugging experience of remote operations is usually not as smooth as working directly on a local machine.
- Extra cost of environment migration. The CUDA, JAX, drivers, and Python dependencies you painstakingly configured, if you haven't created an image or locked the environment beforehand (conda environment.yml, Docker), will likely need to be reconfigured when you switch to a different cloud instance. For research environments, this migration cost is easily underestimated.
- Session limits. Some instances have runtime limits, auto-release, or renewal restrictions. Long training jobs require checking the specific rules for your current machine type in advance.
- Psychological burden. Every time you see the timer ticking, you can't help thinking "should I shut it down?" This mindset is very unfriendly to research work that requires long-running experiments.


These experience issues piled up, leading me to seriously consider building my own workstation.


That said, to be fair, some of these issues can be mitigated through engineering practices. For long-term research projects, rather than treating "local vs. cloud" as two completely different environments, it's better to make your environment portable from day one: `environment.yml` / `requirements.txt` to lock Python dependencies, Docker or container images to pin the CUDA/JAX environment, and decouple checkpoints, config files, and experiment logs from the compute instance. This way, even if you continue using cloud GPUs, the friction of switching machines drops significantly.

## Phase 2: Choosing the Self-Built Workstation


Once I decided to build, the first question was: what specs does DreamerV3 actually need?

### My Recommended Experiment Configuration


Important note: Don't interpret the following specs as DreamerV3's "minimum hardware requirements." Actual resource consumption depends on task type, model size, batch size, image input resolution, number of parallel environments, and replay configuration. The official implementation itself provides small / medium / large / xlarge model configurations of varying scale. These numbers reflect observations from "my experiment setup" rather than universal DreamerV3 requirements.
    Component Entry-level Long-term research recommendation     GPU 12GB+ 24GB VRAM   CPU 8-12 cores 16-20 cores   RAM 32-64GB 64-128GB   SSD 1TB NVMe 2TB+ NVMe   OS Linux Linux    

Specific notes:


- GPU VRAM. DreamerV3's VRAM needs depend heavily on the task, model size, batch size, image inputs, and JAX configuration. The official implementation supports scaling down the model, so you can't simply conclude "DreamerV3 requires 24GB VRAM." 12GB is enough to get started on simple tasks; if your goal is running visual tasks, larger models, and high-frequency experiments long-term, 24GB is noticeably more comfortable. In my own cartpole_balance configuration, system monitoring showed the RTX 5090D (32GB VRAM) peaking at around 25.6GB — since JAX/XLA may pre-allocate and cache memory, this number doesn't directly equal the model's actual VRAM demand. Note that this observation comes from a 32GB card and cannot be used to prove that 24GB on a 4090D is sufficient — reproducing the same configuration on 24GB may risk OOM.
- System RAM. DreamerV3's replay buffer consumes a significant amount of system memory, but actual usage depends heavily on observation type (state vectors vs. images), episode/chunk length, replay capacity, dtype, and other settings. You can't derive a universal RAM requirement from "5e6 transitions × bytes per transition." The default replay buffer capacity is `replay.size: 5e6` (5 million), storing episode/chunk structures. For visual tasks and larger replay configurations, memory can quickly become a bottleneck. In my experiments, 64GB serves as an entry-level option; if you plan to run visual tasks, multi-seed experiments, or larger replay configurations long-term, 128GB is safer.
- CPU cores. The CPU mainly affects environment sampling and data preprocessing speed, not DreamerV3's core model training speed. Actual demand depends on the number of `envs` and environment complexity. The official default is `envs: 16`, but that doesn't mean you need a 16-core CPU — some environments are very lightweight, while visual/physics environments are a different story entirely. I recommend 12-16 cores minimum; if you're running lots of parallel MuJoCo environments, around 20 cores is more comfortable.
- Storage I/O. 2TB is a comfortable starting point. If you're only running MuJoCo and a few experiments, 1TB works; if you're keeping multiple seeds, checkpoints, logs, and datasets long-term, 2TB or even 4TB is advisable. The advantage of NVMe is mainly for environment startup, checkpoint saving, data processing, and experiment logging — not because DreamerV3 specifically requires a 2TB SSD.


For the single-GPU, long-term DreamerV3 visual experiment scenario discussed in this article, if you could only upgrade one component, I'd prioritize VRAM capacity and the actual peak demand of your target task — if your target configuration has been verified to need more than 24GB VRAM, then piling on CPU/RAM is pointless; you should go straight to a GPU with more VRAM. Assuming VRAM is sufficient, system memory is the second area to focus on: with a 24GB GPU but only 64GB RAM, the replay buffer may exhaust your memory first.


Based on this analysis, my final configuration was:
    Part Model Price (¥) Rationale     GPU RTX 4090D 24GB 16,299 24GB VRAM; best value choice for a single-GPU workstation   CPU Intel i7-14700KF (20 cores, 28 threads) 2,699 20 cores / 28 threads, providing ample headroom for parallel environment sampling and data preprocessing   Motherboard MSI B760M MORTAR WiFi DDR5 899 DDR5 support, M.2×2, built-in WiFi   RAM G.Skill 64GB (32G×2) DDR5-5600 1,099 64GB actual config; 128GB recommended for long-term visual tasks or multi-seed; dual-channel   SSD Samsung 990 PRO 2TB NVMe 899 Fast I/O for training data; high endurance   Cooler Thermalright Frozen Magic 360 AIO 399 360mm AIO, keeps 14700KF from thermal throttling under full load   PSU Corsair RM1000e 1000W 80+ Gold Fully Modular 799 1000W with upgrade headroom   Case Fractal Design Pop XL Air 499 Good airflow, supports 360mm AIO, fits full-size GPUs   Total  ≈ 23,592     

The core philosophy of this build: GPU as the budget priority, but don't over-compress RAM, SSD, and PSU to save on the GPU. In this configuration, the GPU accounts for 69.1% of the total budget — this is mainly because the GPU is by far the most expensive single component, and for a workstation focused on single-GPU deep learning experiments, this budget structure is reasonable. Note that if you switch to a GPU with more VRAM, 128GB RAM, or a different platform, this ratio will change significantly. DreamerV3's replay buffer can become a RAM bottleneck, and parallel environments depend on the CPU, so you need to leave enough headroom for those components.


Note: The VRAM data cited earlier (25.6GB peak) came from an RTX 5090D instance on AutoDL. It's important to clarify that the RTX 5090D has 32GB VRAM, while the final workstation chose the RTX 4090D with 24GB. The two have different VRAM capacities, so the 25.6GB observation cannot be used to prove that 24GB on the 4090D is definitely sufficient. Choosing the 4090D was a decision based on budget and single-GPU value-for-money, not reverse-engineered from the 25.6GB experimental data. For readers aiming to exactly reproduce the experiment configuration in this article, 24GB is not guaranteed to be sufficient — before choosing the 4090D, you should verify actual peak VRAM under your target configuration, and if necessary adapt by scaling down model size, batch size, or sequence length.

## Phase 3: Vendor Quote Comparison


After finalizing the parts list, I sent it to a vendor for a quote. Their turnkey price was ¥36,000 — about ¥12,700 more than my self-sourced parts list.

### Price Difference Analysis


A component-by-component comparison of the vendor's configuration vs. my self-sourced list:
    Component My choice Vendor configuration Difference     GPU RTX 4090D 24GB RTX4090D 24G Single Blower Blower-style cooling exhausts hot air directly out of the case, better suited for multi-GPU or server scenarios; for a single-card workstation, it's not necessarily quieter than an open-frame triple-fan design   CPU i7-14700KF Boxed i7-14700KF Tray OEM tray is ¥200-300 cheaper, same performance   Motherboard MSI B760M MORTAR ASUS TUF B760M-PLUS Same tier, ASUS TUF slightly pricier   RAM G.Skill 64GB DDR5-5600 Kingston 64GB DDR5-6000 Higher frequency but looser C40 timings, similar real-world performance   Cooler Thermalright 360 AIO Deepcool Ice Blade 360 Different brand positioning; actual performance depends on specific model   PSU Corsair RM1000e Great Wall Huaxia Song 1000HX Platinum ATX3.1 Established domestic brand, platinum certified + ATX3.1, solid choice   Case Fractal Pop XL Air Deepcool Blade 360 Different model positioning and brand, core specs in a similar range    

Core specs are basically in the same tier, but some specific models and brands differ, especially the GPU cooling solution, cooler, and case. The ¥12,700 price difference covers assembly, OS installation, CUDA environment setup, full-system testing, and warranty service.

### Is It Worth It?


It depends on how you price "peace of mind":


- If the vendor offers a 3-year system warranty, clear on-site repair service, and CUDA environment support, this premium can be understood as paying for convenience and after-sales service; whether it's worth it depends on your time cost and support needs
- If it's just a standard shop warranty with no on-site service, the price is steep — you could negotiate down to ¥28,000-30,000
- If you're willing to spend half a day assembling and a day or two configuring the environment yourself, the ¥12,700 saved by sourcing parts is very real


My choice was self-sourced parts. The reason is simple: as an engineer, assembling a PC and configuring CUDA environments is part of the job — no reason to pay a premium for it.

## Simple Static Cost Break-Even


Finally, let's run the numbers: how long until a self-built workstation "pays for itself"? I'll use a simple static model — a true TCO should also account for GPU depreciation, maintenance, cooling, cloud storage, and transfer fees, but we'll start with the most basic calculation.


Formula: Simple cost break-even ≈ Self-build initial cost ÷ (Cloud GPU annual cost − Self-build annual operating cost). Self-build annual operating cost mainly includes electricity.
    GPU daily runtime GPU rental annual cost Self-build annual electricity Simple break-even     4 hours/day ¥3,764 ~¥525 ~87 months (>7 years, self-build not recommended)   8 hours/day ¥7,530 ~¥1,050 ~44 months (~3.7 years)   12 hours/day ¥11,299 ~¥1,577 ~29 months (~2.4 years)   18 hours/day ¥16,943 ~¥2,365 ~19 months    

Electricity estimate: The RTX 4090D draws about 425W under full load, but total wall power will be higher (CPU, motherboard, RAM, SSD, fans also consume power). Estimating total system average at ~0.6kW and electricity at ¥0.6/kWh, running 12 hours/day yields annual electricity of approximately 0.6 × 12 × 365 × 0.6 ≈ ¥1,577. If the workstation is in a room that needs extra air conditioning in summer, the additional AC power should be factored into TCO; this article doesn't include that for now.


If your GPU averages 12-18 hours of effective runtime per day, the simple static break-even falls between 19-29 months. Note this is "GPU runtime," not "how long a person works" — the machine can run 24 hours; a human doesn't need to watch it constantly. The break-even here only counts GPU rental costs, excluding cloud storage and transfer fees.


From a purely economic perspective, at ¥2.58/h, the payback period for a self-built workstation isn't short. 12 hours is not a threshold where "buying immediately saves money." You only enter a clear economic advantage zone with very high GPU utilization (18+ hours/day), sustained over a long period. But the value of building isn't just about saving money — experience and environment stability are equally important considerations.


Another way to look at it: under this article's assumptions (0.6kW total system, ¥0.6/kWh electricity), the wall power cost for the self-built machine is about ¥0.36/hour, while cloud GPU is ¥2.58/hour. The direct compute cost difference per GPU-hour is about ¥2.22. But ¥0.36/h is not the full TCO — it doesn't account for depreciation, repairs, air conditioning, residual hardware value, and failure risk. Building requires an upfront capital expenditure of about ¥23,592, so what truly determines the outcome isn't "who's cheaper per hour" but how long you can keep the machine at high utilization.


But paying off the investment isn't the only consideration. A self-built workstation has several advantages cloud GPUs can't provide:


- Data locality. Training data, model checkpoints, and experiment logs all stay local — no back-and-forth transfers
- No cloud queuing. When you need to start an experiment, you can run it immediately without waiting for cloud GPU resources to free up
- Predictable long-term costs. After purchasing, daily cash outflow is mainly electricity; but if calculating strict TCO, you need to factor in hardware depreciation, maintenance, and residual value
- Upgradable. You can expand RAM and SSD in the future; if you have a clear plan for dual GPUs, it's best to plan the motherboard, PSU, case, and PCIe lanes for a dual-card platform from the start

## Advice for Different Stages


Based on where you currently are, here's my recommendation:


Entry stage (still learning DreamerV3 fundamentals, running cartpole_balance): Use Google Colab or AutoDL's free/low-cost instances. At this stage your training tasks are light — no need to invest in hardware.


Intermediate stage (starting to run more complex DMC visual tasks, multiple seeds, or higher training-ratio experiments): If your weekly GPU runtime has consistently exceeded 20 hours (this is my personal threshold, not a universal economic standard), I'd start running a "rent vs. buy" cost analysis rather than buying outright. Use AutoDL for 1-2 months first, confirm you'll keep using it consistently, then order hardware.


Research stage (need high-frequency iteration, writing blog posts, running comparison experiments): If you need high-frequency iteration and a single card can cover most experiments, the convenience of a local workstation becomes very apparent; if your experiment scale has entered multi-GPU or large-scale parallel territory, cloud or cluster resources are more appropriate.

## Quick Decision Reference
    Your situation Leans toward     < 4 hours/day Cloud GPU   4-8 hours/day Cloud GPU preferred   8-12 hours/day Calculate based on actual pricing and usage cycle   12-18 hours/day, expected to continue 2+ years Seriously consider self-building   18+ hours/day, sustained high single-GPU utilization Self-build economics strengthen significantly   Occasionally need multiple GPUs Cloud GPU / cluster   Regularly need 4-8 GPUs Cloud/cluster usually more sensible   Data must stay local Self-build attractiveness increases notably   Just starting to learn DreamerV3 Rent first, don't buy    
## Summary


If you only occasionally run DreamerV3, rent cloud GPU time; if you've confirmed you'll be a long-term, high-frequency single-GPU user, then seriously consider building.


Based on this article's ¥2.58/GPU-hour cloud pricing and approximately ¥23,592 self-build cost, under a simple model considering only GPU rental and electricity, running 12 hours/day takes about 29 months and 18 hours/day about 19 months to reach static cost break-even. This means "12 hours/day" is not the economic tipping point for buying a workstation. The scenarios where building truly makes sense are typically: high GPU utilization, long expected usage period, and the no-queue, stable-environment, local-data, and fast-iteration benefits of a local machine genuinely improve your research productivity.


Conversely, if your experiments occasionally require multiple GPUs, or your GPU usage has clear peaks and valleys, the elasticity of cloud GPUs is often more valuable than self-building.


So don't ask "which is cheaper, cloud GPU or workstation" — ask three questions first: How many GPU hours do I actually use per month? How long will this usage intensity last? How much premium am I willing to pay for the low latency and stability of a local machine? The answers to these three questions matter more than any hardware configuration list.
