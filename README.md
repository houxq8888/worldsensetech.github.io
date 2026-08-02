# WorldSense 技术笔记

个人技术博客，专注于世界模型、机器人智能和Sim-to-Real技术。

## 本地预览

直接在浏览器中打开 `index.html` 即可预览。

## 部署到 GitHub Pages

1. 在 GitHub 上创建仓库 `worldsensetech.github.io`（或任意仓库名）
2. 推送代码：
   ```bash
   git remote add origin https://github.com/worldsensetech/worldsensetech.github.io.git
   git branch -M main
   git push -u origin main
   ```
3. 在仓库 Settings → Pages 中选择部署分支
4. 如果使用自定义域名，在 Settings → Pages → Custom domain 中填入 `worldsensetech.com`

## 目录结构

```
├── index.html          # 首页（文章列表）
├── about.html          # 关于页面
├── style.css           # 样式
└── articles/           # 文章目录
    ├── world-model-intro.html    # 世界模型科普
    ├── rssm-deep-dive.html       # RSSM详解
    └── embodied-ai-guide.html    # 具身智能入门
```

## 知乎版本

`zhihu-*.md` 文件是知乎发布用的 Markdown 版本，可直接复制粘贴到知乎编辑器。
