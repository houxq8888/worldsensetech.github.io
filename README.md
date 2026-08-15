# WorldSense Blog (Hugo)

双语技术博客，聚焦世界模型、具身智能、Sim-to-Real 方向。

## 项目结构

```
worldsense-blog-hugo/
├── config/              # Hugo 配置
│   └── _default/
│       ├── config.yaml      # 主配置
│       ├── params.yaml      # 站点参数
│       └── menus.yaml       # 菜单配置
├── content/             # 文章内容
│   ├── zh/                  # 中文文章
│   │   ├── articles/
│   │   └── about.md
│   └── en/                  # 英文文章
│       ├── articles/
│       └── about.md
├── themes/              # 主题
│   └── worldsense/
│       ├── layouts/
│       ├── static/
│       └── theme.toml
├── static/              # 静态资源
│   └── images/
├── translate.py         # 翻译脚本
└── README.md
```

## 快速开始

### 1. 安装 Hugo

```bash
# macOS
brew install hugo

# Windows
choco install hugo-extended

# Linux
sudo apt install hugo
```

### 2. 本地预览

```bash
hugo server -D
```

访问 http://localhost:1313

### 3. 构建站点

```bash
hugo
```

生成的静态文件在 `public/` 目录。

## 写文章

### 中文文章

在 `content/zh/articles/` 创建 Markdown 文件：

```markdown
---
title: "文章标题"
date: 2026-08-15
draft: false
categories: ["世界模型"]
tags: ["World Model", "DreamerV3"]
description: "文章描述"
toc: true
---

正文内容...
```

### 英文文章

在 `content/en/articles/` 创建同名文件，或使用翻译脚本自动生成。

## 翻译

### 自动翻译

```bash
python translate.py
```

脚本会：
1. 扫描 `content/zh/articles/` 中的所有文章
2. 翻译未翻译或中文版本更新的文章
3. 生成英文版本到 `content/en/articles/`

### 术语表

翻译使用内置术语表，保证专业术语一致性。主要术语：

| 中文 | 英文 |
|------|------|
| 世界模型 | World Model |
| 具身智能 | Embodied AI |
| 强化学习 | Reinforcement Learning |
| 状态空间模型 | State Space Model |
| 仿真 | Simulation |
| Sim-to-Real | Sim-to-Real |

## 部署

### GitHub Pages

```bash
hugo
# 将 public/ 推送到 gh-pages 分支
```

### 阿里云服务器

```bash
hugo
# 将 public/ 上传到服务器
rsync -avz public/ user@server:/path/to/webroot/
```

### DNS 配置

在阿里云 DNS 设置：
- 海外线路 → CNAME → GitHub Pages
- 国内线路 → A 记录 → 阿里云服务器 IP

## 分类

- `world-model` - 世界模型
- `embodied-ai` - 具身智能
- `reinforcement-learning` - 强化学习
- `sim-to-real` - Sim-to-Real
- `simulation` - 仿真
- `tutorial` - 教程

## 许可证

MIT License
