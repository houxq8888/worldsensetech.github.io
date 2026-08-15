---
title: "Isaac Lab 安装避坑指南：从零到跑通全流程"
slug: "isaac-lab-install-guide"
aliases:
  - /articles/isaac-lab-install-guide.html
date: 2026-08-15
draft: false
categories: ["仿真"]
tags: ["Isaac Lab", "Isaac Sim", "安装教程", "具身智能"]
description: "Isaac Lab 安装避坑指南：从零到跑通全流程 - WorldSense 技术笔记"
toc: true
---


上一篇介绍了 Isaac Lab 是什么、能做什么。但如果你真的动手装过，会发现从"克隆代码"到"跑通 example"之间，隔着不少坑。
 

这篇文章记录我在 AutoDL 云 GPU 服务器（RTX 5090D/32GB）上安装 Isaac Lab 的完整过程。每一个报错都是真实遇到的，每一个解决方案都是实际验证过的。希望能帮你省掉几个小时的排查时间。
 
## 为什么 Isaac Lab 这么难装？
 

Isaac Lab 的安装复杂度主要来自三个方面：
 

依赖链长。Isaac Lab 依赖 Isaac Sim，Isaac Sim 依赖 NVIDIA 驱动、CUDA、特定版本的 PyTorch，而 PyTorch 又依赖特定版本的 numpy、triton 等。任何一环版本不对，都会报错。
 

Python 版本要求高。Isaac Lab 要求 Python ≥ 3.12，而很多服务器和云 GPU 平台的基础镜像还在用 Python 3.10 或 3.11。版本不满足会直接安装失败。
 

包体积大。PyTorch + CUDA 的 whl 文件超过 2GB，numpy 也有几十 MB。在国内网络环境下，从 PyPI 下载经常中断，而且 pip 的断点续传并不总是可靠。
 
## 环境要求
 

在开始安装之前，确认你的环境满足以下条件：
 
 
- GPU。NVIDIA RTX 级别以上 GPU（推荐 RTX 4090 或更高） 
- 系统。Ubuntu 22.04（推荐）或 Windows 
- Python。3.12 或更高版本（这是硬要求，3.11 也不行） 
- 驱动。NVIDIA 驱动 525+ 和 CUDA 12+ 
- 内存。16GB+（推荐 32GB） 
- 磁盘。至少 50GB 可用空间（Isaac Sim 本身体积较大） 
 
## 第一步：解决 Python 版本问题
 

这是最容易踩的第一个坑。很多服务器默认的 Python 版本是 3.10 或 3.11，直接跑 `./isaaclab.sh --install` 会报：
 
```
`ModuleNotFoundError: No module named 'tomllib'`
```
 

或者更明确的版本错误：
 
```
`ERROR: Package 'isaaclab' requires a different Python: 3.11.15 not in '>=3.12'`
```
 

`tomllib` 是 Python 3.11 加入标准库的模块，但 Isaac Lab 实际要求的是 Python ≥ 3.12。所以 3.11 也不够用。
 

解决方案：用 conda 创建一个独立的 Python 3.12 环境。这样做的好处是不影响服务器上其他项目（比如你正在跑的 DreamerV3 训练）。
 
```
`# 创建 Python 3.12 环境
conda create -n isaaclab python=3.12 -y

# 激活环境
conda activate isaaclab

# 确认版本
python --version  # 应该显示 Python 3.12.x`
```
 

如果 `conda activate` 报错 `CommandNotFoundError`，说明 shell 没有初始化 conda，先运行：
 
```
`conda init bash
source ~/.bashrc`
```
 

然后再 `conda activate isaaclab`。
 
## 第二步：配置国内 pip 镜像源
 

这是第二个大坑。Isaac Lab 安装过程中需要下载 PyTorch（2GB+）、numpy、triton 等大包。从 PyPI 官方源下载，在国内网络环境下大概率会反复中断。pip 虽然有断点续传，但尝试几次之后就会放弃：
 
```
`WARNING: Connection interrupted while downloading.
WARNING: Attempting to resume incomplete download (11.0 MB/16.6 MB, attempt 5)
error: incomplete-download
× Download failed after 6 attempts`
```
 

解决方案：提前把 pip 的默认源换成国内镜像。清华源和阿里云源都比较稳定：
 
```
`# 设置清华镜像源（推荐）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或者阿里云源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple`
```
 

如果后续安装 PyTorch 时仍然需要从 NVIDIA 的源拉取 CUDA 版本，可以在命令中单独指定：
 
```
`pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128`
```
 

这条命令会绕过全局镜像源，直接从 PyTorch 官方 CUDA 源下载。如果这个也慢，可以先下载 whl 文件再本地安装。
 
## 第三步：安装 Isaac Sim
 

Isaac Sim 是 Isaac Lab 的底层仿真引擎，必须先装好。有三种方式：
 
### 方式一：pip 安装（推荐）
 

Isaac Sim 4.x 以上版本支持 pip 安装，最简单：
 
```
`pip install isaacsim`
```
 

如果 pip 源已经配置为国内镜像，这一步会比较顺利。Isaac Sim 的包比较大，耐心等待即可。
 
### 方式二：Omniverse Launcher
 

适合需要 GUI 的本地开发场景。从 NVIDIA 官网下载 Omniverse Launcher，在 Launcher 中安装 Isaac Sim。这种方式有图形界面，可以看到安装进度，但需要桌面环境。
 
### 方式三：Docker
 

适合服务器或无头环境：
 
```
`docker pull nvcr.io/nvidia/isaac-sim:latest`
```
 

Docker 方式的好处是环境隔离，不会污染宿主机。但需要配置 NVIDIA Container Toolkit，而且容器内的文件访问不如直接安装方便。
 

对于云 GPU 服务器（如 AutoDL），推荐方式一（pip）或方式三（Docker）。
 
## 第四步：克隆并安装 Isaac Lab
 

Isaac Sim 装好后，就可以安装 Isaac Lab 了：
 
```
`# 克隆仓库
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 运行安装脚本
./isaaclab.sh --install`
```
 

这个脚本会做几件事：安装 Isaac Lab 自身的 Python 包（editable mode）、安装依赖（gymnasium、robomimic 等）、配置环境变量。
 

如果前面几步都做好了，这一步通常比较顺利。但如果遇到报错，往下看。
 
## 常见报错和解决方案
 
### 1. Python 版本不满足
 
```
`ERROR: Package 'isaaclab' requires a different Python: 3.x.x not in '>=3.12'`
```
 

原因：当前 Python 版本低于 3.12。解决：回到第一步，用 conda 创建 Python 3.12 环境。
 
### 2. conda activate 失败
 
```
`CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.`
```
 

原因：shell 没有初始化 conda。解决：运行 `conda init bash` 然后 `source ~/.bashrc`。
 
### 3. pip 下载中断
 
```
`WARNING: Connection interrupted while downloading.
error: incomplete-download`
```
 

原因：从 PyPI 下载大包时网络不稳定。解决：配置国内镜像源（第二步），或者手动下载 whl 文件后 `pip install ./xxx.whl`。
 
### 4. torch CUDA 版本不匹配
 
```
`Found no NVIDIA driver on your system.`
```
 

原因：安装了 CPU 版本的 PyTorch。解决：确保用 `--index-url https://download.pytorch.org/whl/cu128` 安装 CUDA 版本的 torch。
 
### 5. Isaac Sim 导入失败
 
```
`ModuleNotFoundError: No module named 'isaacsim'`
```
 

原因：Isaac Sim 没有正确安装，或者当前 conda 环境里没有。解决：确认在正确的 conda 环境中运行 `pip install isaacsim`。
 
### 6. 非交互式 SSH 环境的问题
 

如果你通过 SSH 远程连接服务器（如 AutoDL），注意非交互式 shell 不会自动加载 `~/.bashrc`。这意味着 `conda activate` 可能不生效。
 

解决：在远程命令中使用 `bash -l -c 'command'` 模式，或者在脚本开头显式 `source ~/.bashrc`。
 
## 验证安装
 

安装完成后，运行官方 example 验证环境是否正常：
 
```
`# 在 IsaacLab 目录下
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py`
```
 

如果看到仿真窗口启动（或无头模式下没有报错），说明安装成功。
 

也可以跑一个经典的 Cartpole 训练来端到端验证：
 
```
`./isaaclab.sh -p source/standalone/workflows/rl_games/train.py \
    --task=Isaac-Cartpole-RGB-v0 \
    --headless`
```
 

如果能正常开始训练并输出 reward 曲线，恭喜，Isaac Lab 已经可以用了。
 
## 安装时间参考
 

整个安装过程的时间主要花在下载大包上。在良好的网络环境下：
 
 
- Python 3.12 环境创建：约 2-3 分钟 
- PyTorch + CUDA 下载：约 10-20 分钟（2GB+） 
- Isaac Sim 下载：约 10-30 分钟（取决于版本和包大小） 
- Isaac Lab 依赖安装：约 5-10 分钟 
 

如果网络不稳定，下载时间可能翻倍甚至更多。建议在网络状况好的时段操作，或者提前下载好 whl 文件。
 
## 小结
 

Isaac Lab 的安装确实比一般的 Python 包复杂，但核心就三件事：Python 版本要对（≥ 3.12）、pip 源要快（国内镜像）、Isaac Sim 要先装好。
 

把这三个前提搞定，后面的安装过程就比较顺了。遇到报错也不要慌，大部分问题都是版本不匹配或网络问题，排查方向很明确。
 

装好之后，Isaac Lab 的实战内容就告一段落了。下一篇我们换个角度，看看 2026 年 AI 为什么在疯狂寻找物理外壳——从眼镜到机器人，AI 正在一步步走进物理世界。
