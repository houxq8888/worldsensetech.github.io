---
title: "Isaac Lab Installation Guide: From Zero to Running on AutoDL"
slug: "isaac-lab-install-guide"
date: 2026-08-15
draft: false
categories: ["Tutorial"]
tags: ["Isaac Lab", "Installation", "AutoDL", "Isaac Sim", "Tutorial"]
description: "Isaac Lab Installation Guide: From Zero to Running on AutoDL - WorldSense Tech Blog"
toc: true
aliases:
  - /en/articles/isaac-lab-install-guide.html
---


The previous article covered what Isaac Lab is and what it can do. But if you've actually tried installing it, you know there's quite a gap between "cloning the repo" and "running the examples."

This article documents my complete process of installing Isaac Lab on an AutoDL cloud GPU server (RTX 5090D / 32GB). Every error mentioned here was genuinely encountered, and every solution was verified in practice. Hopefully this saves you a few hours of debugging.

## Why Is Isaac Lab So Hard to Install?

The installation complexity of Isaac Lab mainly comes from three aspects:

Long dependency chain. Isaac Lab depends on Isaac Sim, which depends on NVIDIA drivers, CUDA, and specific versions of PyTorch, which in turn depends on specific versions of numpy, triton, and so on. If any link in the chain has the wrong version, you'll get errors.

Strict Python version requirement. Isaac Lab requires Python >= 3.12, yet many servers and cloud GPU platforms still ship base images with Python 3.10 or 3.11. An unsatisfied version requirement will cause the installation to fail outright.

Large package sizes. The PyTorch + CUDA wheel files exceed 2GB, and numpy alone is tens of MB. In mainland China's network environment, downloads from PyPI frequently get interrupted, and pip's resume capability isn't always reliable.

## Prerequisites

Before starting the installation, make sure your environment meets the following requirements:

- GPU. NVIDIA RTX-class or above (RTX 4090 or higher recommended)
- OS. Ubuntu 22.04 (recommended) or Windows
- Python. 3.12 or higher (this is a hard requirement; 3.11 won't work either)
- Driver. NVIDIA driver 525+ and CUDA 12+
- RAM. 16GB+ (32GB recommended)
- Disk. At least 50GB of free space (Isaac Sim itself is quite large)

## Step 1: Resolve the Python Version Issue

This is the easiest pitfall to fall into. Many servers default to Python 3.10 or 3.11. Running `./isaaclab.sh --install` directly will produce:

```
`ModuleNotFoundError: No module named 'tomllib'`
```

Or a more explicit version error:

```
`ERROR: Package 'isaaclab' requires a different Python: 3.11.15 not in '>=3.12'`
```

`tomllib` was added to the standard library in Python 3.11, but Isaac Lab actually requires Python >= 3.12. So 3.11 isn't sufficient either.

Solution: Use conda to create an isolated Python 3.12 environment. The advantage is that it won't affect other projects on the server (such as your ongoing DreamerV3 training).

```
# Create a Python 3.12 environment
conda create -n isaaclab python=3.12 -y

# Activate the environment
conda activate isaaclab

# Verify the version
python --version  # Should display Python 3.12.x
```

If `conda activate` throws a `CommandNotFoundError`, it means your shell hasn't initialized conda. Run the following first:

```
conda init bash
source ~/.bashrc
```

Then run `conda activate isaaclab` again.

## Step 2: Configure a Domestic pip Mirror

This is the second major pitfall. Installing Isaac Lab requires downloading PyTorch (2GB+), numpy, triton, and other large packages. Downloading from the official PyPI index under mainland China's network conditions will most likely result in repeated interruptions. pip does support resuming downloads, but after a few failed attempts you'll want to give up:

```
`WARNING: Connection interrupted while downloading.
WARNING: Attempting to resume incomplete download (11.0 MB/16.6 MB, attempt 5)
error: incomplete-download
x Download failed after 6 attempts`
```

Solution: Switch pip's default index to a domestic mirror in advance. The Tsinghua and Alibaba Cloud mirrors are both quite stable:

```
# Set the Tsinghua mirror (recommended)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Or the Alibaba Cloud mirror
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
```

If you later need to pull the CUDA version of PyTorch from NVIDIA's index, you can specify it explicitly in the command:

```
pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

This command bypasses the global mirror and downloads directly from PyTorch's official CUDA index. If that's also slow, you can download the wheel file first and then install it locally.

## Step 3: Install Isaac Sim

Isaac Sim is the underlying simulation engine for Isaac Lab and must be installed first. There are three approaches:

### Method 1: pip Install (Recommended)

Isaac Sim 4.x and above support pip installation, which is the simplest approach:

```
pip install isaacsim
```

If you've already configured pip to use a domestic mirror, this step should go smoothly. The Isaac Sim package is quite large, so just be patient.

### Method 2: Omniverse Launcher

Suitable for local development that requires a GUI. Download the Omniverse Launcher from NVIDIA's website and install Isaac Sim through the Launcher. This method provides a graphical interface where you can monitor installation progress, but it requires a desktop environment.

### Method 3: Docker

Suitable for servers or headless environments:

```
docker pull nvcr.io/nvidia/isaac-sim:latest
```

The advantage of Docker is environment isolation — it won't pollute the host system. However, you'll need to configure the NVIDIA Container Toolkit, and file access inside the container isn't as convenient as with a direct installation.

For cloud GPU servers (such as AutoDL), Method 1 (pip) or Method 3 (Docker) is recommended.

## Step 4: Clone and Install Isaac Lab

Once Isaac Sim is installed, you can proceed to install Isaac Lab:

```
# Clone the repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Run the installation script
./isaaclab.sh --install
```

This script does several things: installs Isaac Lab's own Python packages (in editable mode), installs dependencies (gymnasium, robomimic, etc.), and configures environment variables.

If all the previous steps were completed successfully, this step should go smoothly. But if you encounter errors, read on.

## Common Errors and Solutions

### 1. Python Version Not Satisfied

```
`ERROR: Package 'isaaclab' requires a different Python: 3.x.x not in '>=3.12'`
```

Cause: The current Python version is below 3.12. Solution: Go back to Step 1 and create a Python 3.12 environment with conda.

### 2. conda activate Fails

```
`CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.`
```

Cause: The shell hasn't initialized conda. Solution: Run `conda init bash` followed by `source ~/.bashrc`.

### 3. pip Download Interrupted

```
`WARNING: Connection interrupted while downloading.
error: incomplete-download`
```

Cause: Unstable network when downloading large packages from PyPI. Solution: Configure a domestic mirror (Step 2), or manually download the wheel file and install it with `pip install ./xxx.whl`.

### 4. torch CUDA Version Mismatch

```
`Found no NVIDIA driver on your system.`
```

Cause: The CPU-only version of PyTorch was installed. Solution: Make sure to install the CUDA version of torch with `--index-url https://download.pytorch.org/whl/cu128`.

### 5. Isaac Sim Import Fails

```
`ModuleNotFoundError: No module named 'isaacsim'`
```

Cause: Isaac Sim wasn't installed correctly, or it's not present in the current conda environment. Solution: Make sure you're running `pip install isaacsim` in the correct conda environment.

### 6. Non-Interactive SSH Environment Issues

If you're connecting to a server via SSH (such as AutoDL), be aware that non-interactive shells don't automatically source `~/.bashrc`. This means `conda activate` may not take effect.

Solution: Use the `bash -l -c 'command'` pattern for remote commands, or explicitly add `source ~/.bashrc` at the beginning of your script.

## Verifying the Installation

After installation is complete, run an official example to verify that the environment works correctly:

```
# In the IsaacLab directory
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

If you see the simulation window launch (or no errors in headless mode), the installation was successful.

You can also run a classic Cartpole training job for end-to-end verification:

```
./isaaclab.sh -p source/standalone/workflows/rl_games/train.py \
    --task=Isaac-Cartpole-RGB-v0 \
    --headless
```

If training starts normally and outputs reward curves, congratulations — Isaac Lab is ready to use.

## Installation Time Reference

The total installation time is mainly spent downloading large packages. With a good network connection:

- Python 3.12 environment creation: approximately 2–3 minutes
- PyTorch + CUDA download: approximately 10–20 minutes (2GB+)
- Isaac Sim download: approximately 10–30 minutes (depending on version and package size)
- Isaac Lab dependency installation: approximately 5–10 minutes

If the network is unstable, download times can double or more. It's advisable to perform the installation during off-peak hours or pre-download the wheel files.

## Summary

Installing Isaac Lab is certainly more complex than a typical Python package, but it boils down to three things: get the Python version right (>= 3.12), make pip fast (use a domestic mirror), and install Isaac Sim first.

Once those three prerequisites are in place, the rest of the installation should go smoothly. Don't panic if you hit errors — most issues are version mismatches or network problems, and the troubleshooting direction is clear.

With the installation done, we've wrapped up the hands-on Isaac Lab setup. In the next article, we'll shift perspective and look at why AI in 2026 is frantically seeking physical embodiment — from glasses to robots, AI is steadily stepping into the physical world.
