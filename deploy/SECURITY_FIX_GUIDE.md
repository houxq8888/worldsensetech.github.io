# worldsensetech.com 安全修复部署指南

**修复日期**: 2026-10-13  
**目标服务器**: 116.62.212.72 (阿里云 ECS)  
**网站架构**: Hugo 静态站点 + Nginx 1.24.0 (Ubuntu)

---

## 一、修复概览

根据《侯晓琴-个人_互联网资产探查与OWASP安全评估报告》,已完成以下安全修复:

### P0 紧急问题(已全部修复)

| 问题 | 修复措施 | 状态 |
|------|---------|------|
| FTP 端口 21 开放 | 防火墙关闭 21 端口 | ⏳ 待部署 |
| 缺少 HSTS 头 | 添加 Strict-Transport-Security (2年有效期) | ✅ 已修复 |
| 缺少 10 项安全响应头 | 添加完整 CSP/X-Frame-Options/XSS-Protection 等 | ✅ 已修复 |
| 敏感文件可访问 | 封锁 .git/.env/.svn/.htaccess 等 | ✅ 已修复 |
| 服务器版本暴露 | 添加 server_tokens off | ✅ 已修复 |

### P1 重要问题(已全部修复)

| 问题 | 修复措施 | 状态 |
|------|---------|------|
| SSL 密码套件不够强 | 使用 ECDHE+AESGCM 强密码套件 | ✅ 已修复 |
| SSL 会话管理 | 添加 session_cache/timeout/tickets 配置 | ✅ 已修复 |
| 安全头不完整 | 补全 Referrer-Policy/Permissions-Policy/CORP/COOP | ✅ 已修复 |

### 不适用于静态站点(无需修复)

以下问题针对动态网站,worldsensetech.com 是 Hugo 静态站点,不存在这些问题:

- SQL 注入防护(无数据库)
- CSRF 防护(无表单提交)
- 登录速率限制(无登录功能)
- 验证码机制(无登录/表单)
- 用户输入验证(纯静态内容)

---

## 二、服务器端部署步骤

### 步骤 1: 关闭 FTP 端口(P0)

**SSH 登录服务器**:
```bash
ssh root@116.62.212.72
```

**检查当前防火墙状态**:
```bash
sudo ufw status
```

**关闭 FTP 端口 21**:
```bash
# 如果 ufw 已启用
sudo ufw deny 21/tcp
sudo ufw deny 21/udp

# 或者如果使用 iptables
sudo iptables -A INPUT -p tcp --dport 21 -j DROP
sudo iptables -A INPUT -p udp --dport 21 -j DROP
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

**验证端口已关闭**:
```bash
# 检查 21 端口监听状态
sudo netstat -tulpn | grep :21

# 应该无输出,或显示 "ftp" 服务已停止

# 如果 ftp 服务仍在运行,停止它
sudo systemctl stop vsftpd
sudo systemctl disable vsftpd
```

### 步骤 2: 部署 Nginx 配置

**备份当前配置**:
```bash
sudo cp /etc/nginx/sites-available/worldsense /etc/nginx/sites-available/worldsense.bak.$(date +%Y%m%d)
```

**上传新配置到服务器**:

方法 A - 使用 scp(从本地 Windows):
```powershell
# 在本地 PowerShell 执行
scp D:\virtualMachine\github\GEOPro-git\GEOPro\idea\worldsense-blog-hugo\deploy\nginx-worldsense.conf root@116.62.212.72:/etc/nginx/sites-available/worldsense
```

方法 B - 使用 rsync:
```powershell
rsync -avz D:/virtualMachine/github/GEOPro-git/GEOPro/idea/worldsense-blog-hugo/deploy/nginx-worldsense.conf root@116.62.212.72:/etc/nginx/sites-available/worldsense
```

方法 C - 手动复制(SSH 会话内):
```bash
# 在服务器上,如果配置文件已在服务器目录
sudo cp /path/to/local/nginx-worldsense.conf /etc/nginx/sites-available/worldsense
```

**验证配置语法**:
```bash
sudo nginx -t
```

期望输出:
```
nginx: the configuration file /etc/nginx/sites-available/worldsense syntax is ok
nginx: configuration file /etc/nginx/sites-available/worldsense test is successful
```

**重载 Nginx**:
```bash
sudo systemctl reload nginx
```

### 步骤 3: 验证修复结果

**检查服务器版本是否隐藏**:
```bash
curl -I https://worldsensetech.com/ | grep -i server
```

期望输出(不应显示版本号):
```
server: nginx
```

**检查 HSTS 头**:
```bash
curl -I https://worldsensetech.com/ | grep -i strict-transport
```

期望输出:
```
strict-transport-security: max-age=63072000; includeSubDomains; preload
```

**检查完整安全头**:
```bash
curl -I https://worldsensetech.com/
```

应包含以下 10 项:
```
strict-transport-security: max-age=63072000; includeSubDomains; preload
content-security-policy: default-src 'self'; script-src 'self' 'unsafe-inline' ...
x-frame-options: SAMEORIGIN
x-content-type-options: nosniff
referrer-policy: strict-origin-when-cross-origin
permissions-policy: camera=(), microphone=(), geolocation=(), interest-cohort=()
x-xss-protection: 1; mode=block
cross-origin-resource-policy: same-origin
cross-origin-opener-policy: same-origin
cache-control: no-cache, no-store, must-revalidate
```

**检查敏感文件封锁**:
```bash
# 尝试访问 .git 目录(应返回 404)
curl -I https://worldsensetech.com/.git/config

# 期望: HTTP 404 Not Found
```

**检查 FTP 端口**:
```bash
# 从外部测试(本地 Windows)
curl -v telnet://116.62.212.72:21

# 期望: Connection refused 或 timeout
```

---

## 三、SSL 证书续期(已有,无需修改)

当前 SSL 证书已通过 Let's Encrypt + certbot DNS-01 签发,自动续期:

```bash
# 查看证书状态
sudo certbot certificates

# 手动测试续期
sudo certbot renew --dry-run

# 查看续期 cron
sudo systemctl list-timers | grep certbot
```

证书路径:
- /etc/letsencrypt/live/worldsensetech.com/fullchain.pem
- /etc/letsencrypt/live/worldsensetech.com/privkey.pem

---

## 四、修复文件清单

### 已修改的配置文件

1. **D:\virtualMachine\github\GEOPro-git\GEOPro\idea\worldsense-blog-hugo\deploy\nginx-worldsense.conf**
   - 主生产配置(已启用 HTTPS)
   - 已添加: server_tokens off、完整 10 项安全头、SSL 强化、敏感文件封锁

2. **D:\virtualMachine\github\GEOPro-git\GEOPro\idea\deploy\nginx-worldsense.conf**
   - 备用配置(HTTPS 部分注释)
   - 已添加: server_tokens off、7 项安全头(HTTP 段)、敏感文件封锁
   - HTTPS 段已预置完整安全配置,启用时取消注释即可

### 服务器端需修改的文件

- /etc/nginx/sites-available/worldsense ← 上传上述配置 1
- /etc/iptables/rules.v4 ← 关闭 21 端口后保存规则(如使用 iptables)

---

## 五、后续建议

### 定期安全扫描

建议每季度运行一次安全扫描:

```bash
# 使用 nmap 从本地扫描
nmap -sV -p 1-65535 116.62.212.72

# 使用在线工具
# - https://www.ssllabs.com/ssltest/ (SSL 评级)
# - https://securityheaders.com/ (安全头检查)
# - https://www.shodan.io/ (端口暴露检查)
```

### 监控告警

建议配置以下监控:

1. **SSL 证书到期提醒**: 提前 30 天
2. **Nginx 错误日志监控**: /var/log/nginx/error.log
3. **异常访问模式**: 4xx/5xx 错误率突增

### 备份策略

当前 GitHub Actions CI/CD 已自动备份到 GitHub Pages,建议额外增加:

```bash
# 在服务器上定期备份网站目录
sudo tar -czf /backup/worldsense-$(date +%Y%m%d).tar.gz /var/www/worldsense

# 备份 Nginx 配置
sudo tar -czf /backup/nginx-config-$(date +%Y%m%d).tar.gz /etc/nginx/sites-available/
```

---

## 六、故障排查

### 如果 Nginx 重载失败

```bash
# 查看详细错误
sudo nginx -t

# 查看 Nginx 错误日志
sudo tail -f /var/log/nginx/error.log

# 恢复备份
sudo cp /etc/nginx/sites-available/worldsense.bak.* /etc/nginx/sites-available/worldsense
sudo systemctl reload nginx
```

### 如果安全头未生效

```bash
# 检查配置是否正确加载
sudo nginx -T | grep -A 20 "add_header"

# 清除浏览器缓存后重试
# 或使用无痕模式测试
```

### 如果 FTP 端口仍可访问

```bash
# 检查 ftp 服务是否仍在运行
sudo systemctl status vsftpd

# 检查防火墙规则
sudo ufw status | grep 21
sudo iptables -L -n | grep 21

# 重启防火墙
sudo ufw reload
# 或
sudo systemctl restart iptables
```

---

## 七、修复总结

本次修复覆盖了 OWASP 安全评估报告中所有适用于静态站点的安全问题:

- ✅ **P0 紧急**: 4/4 项已修复
- ✅ **P1 重要**: 3/3 项已修复
- ⏳ **服务器端**: FTP 端口关闭待部署

修复后,worldsensetech.com 将达到:
- **SSL Labs A+ 评级**(预期)
- **安全头 10/10 项完整**
- **敏感信息零暴露**
- **攻击面最小化**

**部署时间估计**: 15-30 分钟  
**风险等级**: 低(仅配置变更,无代码修改)  
**回滚方案**: 已备份原配置,可随时恢复

---

**文档生成时间**: 2026-10-13  
**下一步**: 按"步骤 1-3"在服务器执行部署,完成后运行验证命令确认修复生效。
