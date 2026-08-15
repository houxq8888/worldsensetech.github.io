#!/bin/bash
# WorldSense Blog - 阿里云服务器初始化脚本
# 在阿里云服务器上运行此脚本完成初始配置

set -e

echo "=== WorldSense Blog 服务器初始化 ==="

# 1. 创建网站目录
echo "1. 创建网站目录..."
sudo mkdir -p /var/www/worldsense
sudo chown -R www-data:www-data /var/www/worldsense
sudo chmod -R 755 /var/www/worldsense

# 2. 安装 Nginx（如果未安装）
echo "2. 检查 Nginx..."
if ! command -v nginx &> /dev/null; then
    echo "安装 Nginx..."
    sudo apt-get update
    sudo apt-get install -y nginx
fi

# 3. 配置 Nginx
echo "3. 配置 Nginx..."
# 复制配置文件
sudo cp nginx-worldsense.conf /etc/nginx/sites-available/worldsensetech.com

# 启用站点
sudo ln -sf /etc/nginx/sites-available/worldsensetech.com /etc/nginx/sites-enabled/

# 删除默认站点（可选）
# sudo rm -f /etc/nginx/sites-enabled/default

# 4. 创建 SSL 目录
echo "4. 创建 SSL 目录..."
sudo mkdir -p /etc/nginx/ssl

echo ""
echo "=== 下一步操作 ==="
echo ""
echo "1. 上传 SSL 证书到 /etc/nginx/ssl/"
echo "   sudo cp your_cert.pem /etc/nginx/ssl/worldsensetech.com.pem"
echo "   sudo cp your_key.key /etc/nginx/ssl/worldsensetech.com.key"
echo ""
echo "2. 测试 Nginx 配置"
echo "   sudo nginx -t"
echo ""
echo "3. 重载 Nginx"
echo "   sudo systemctl reload nginx"
echo ""
echo "4. 设置 GitHub Actions SSH 密钥"
echo "   - 在 GitHub 仓库 Settings → Secrets 添加 ALIYUN_SSH_KEY"
echo "   - 密钥内容：~/.ssh/id_rsa 的内容"
echo ""
echo "5. 首次部署"
echo "   - 推送代码到 main 分支，GitHub Actions 会自动部署"
echo ""
