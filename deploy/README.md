# WorldSense Blog 部署指南

## 文件说明

```
deploy/
├── nginx-worldsense.conf    # Nginx 配置文件
├── setup-server.sh          # 服务器初始化脚本
└── README.md                # 本文件
```

## 一、阿里云服务器配置

### 1. 上传部署文件到服务器

```bash
scp -r deploy/ root@116.62.212.72:/tmp/
```

### 2. SSH 登录服务器并执行初始化

```bash
ssh root@116.62.212.72
cd /tmp/deploy
chmod +x setup-server.sh
./setup-server.sh
```

### 3. 上传 SSL 证书

```bash
# 在本地执行
scp your_cert.pem root@116.62.212.72:/etc/nginx/ssl/worldsensetech.com.pem
scp your_key.key root@116.62.212.72:/etc/nginx/ssl/worldsensetech.com.key
```

### 4. 测试并重载 Nginx

```bash
# 在服务器执行
sudo nginx -t
sudo systemctl reload nginx
```

## 二、GitHub 配置

### 1. 添加 Secrets

进入 GitHub 仓库 → Settings → Secrets and variables → Actions → New repository secret

| Secret 名称 | 值 |
|------------|-----|
| `ALIYUN_SERVER_IP` | `116.62.212.72` |
| `ALIYUN_SERVER_USER` | `root` |
| `ALIYUN_SSH_KEY` | SSH 私钥内容（`cat ~/.ssh/id_rsa`） |
| `BAIDU_PUSH_TOKEN` | `MqQ1cbD8xkXheg5z` |

### 2. 获取 SSH 私钥

```bash
# 如果没有 SSH 密钥，先生成
ssh-keygen -t rsa -b 4096 -C "github-actions"

# 查看私钥内容
cat ~/.ssh/id_rsa
```

### 3. 将公钥添加到服务器

```bash
# 在本地执行
ssh-copy-id root@116.62.212.72
```

## 三、自动发布流程

### 草稿发布

在 `content/zh/articles/` 创建草稿，设置 `publishDate`：

```yaml
---
title: "新文章"
slug: "new-article"
date: 2026-08-20
publishDate: 2026-08-25T08:00:00+08:00  # 到期自动发布
draft: false
---
```

### 触发方式

1. **定时触发**：每天北京时间 08:00 自动检查
2. **手动触发**：推送代码到 main 分支
3. **手动运行**：GitHub Actions 页面点击 "Run workflow"

### 部署目标

- GitHub Pages（国际）：`worldsensetech.github.io`
- 阿里云服务器（国内）：`116.62.212.72`

## 四、DNS 配置

### 阿里云 DNS 解析

| 记录类型 | 主机记录 | 记录值 |
|---------|---------|--------|
| A | @ | 116.62.212.72 |
| CNAME | www | worldsensetech.com |

### GitHub Pages 自定义域名

在仓库 Settings → Pages → Custom domain 填写：
```
worldsensetech.com
```

勾选 "Enforce HTTPS"。

## 五、验证部署

### 1. 检查 GitHub Actions 运行状态

```
https://github.com/houxq8888/worldsensetech.github.io/actions
```

### 2. 检查国内访问

```bash
curl -I https://worldsensetech.com
```

### 3. 检查百度收录

```
https://ziyuan.baidu.com/dashboard
```

## 六、常见问题

### Q: 阿里云服务器 SSH 连接失败？

检查安全组是否开放 22 端口。

### Q: GitHub Actions 部署失败？

检查 Secrets 是否正确配置，SSH 密钥是否有效。

### Q: 国内访问慢？

确认 DNS 解析到阿里云 IP，而非 GitHub Pages。

### Q: 文章发布后没更新？

- 检查 `publishDate` 是否已到期
- 检查 GitHub Actions 是否运行成功
- 手动清除浏览器缓存
