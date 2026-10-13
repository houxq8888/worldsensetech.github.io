#!/usr/bin/env bash
# worldsensetech.com 安全修复验证脚本
# 用法: bash verify-security-fixes.sh [domain]
# 默认域名: worldsensetech.com

DOMAIN=${1:-"worldsensetech.com"}
echo "=========================================="
echo "worldsensetech.com 安全修复验证"
echo "域名: $DOMAIN"
echo "日期: $(date)"
echo "=========================================="
echo ""

PASS=0
FAIL=0

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((PASS++))
}

check_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((FAIL++))
}

check_warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
}

echo "【1/6】检查服务器版本隐藏..."
SERVER_HEADER=$(curl -sI "https://$DOMAIN/" | grep -i "^server:" | head -1)
if echo "$SERVER_HEADER" | grep -qE "nginx/[0-9]"; then
    check_fail "服务器版本暴露: $SERVER_HEADER"
elif echo "$SERVER_HEADER" | grep -qi "nginx"; then
    check_pass "服务器版本已隐藏: $SERVER_HEADER"
else
    check_warn "未检测到 Server 头或格式异常: $SERVER_HEADER"
fi
echo ""

echo "【2/6】检查 HSTS 头..."
HSTS=$(curl -sI "https://$DOMAIN/" | grep -i "^strict-transport-security:" | head -1)
if [ -n "$HSTS" ]; then
    if echo "$HSTS" | grep -q "max-age=63072000"; then
        check_pass "HSTS 已启用且有效期≥2年: $HSTS"
    else
        check_warn "HSTS 已启用但有效期较短: $HSTS"
    fi
else
    check_fail "HSTS 头缺失"
fi
echo ""

echo "【3/6】检查完整安全响应头(10项)..."
HEADERS=$(curl -sI "https://$DOMAIN/")

declare -A REQUIRED_HEADERS=(
    ["strict-transport-security"]="HSTS"
    ["content-security-policy"]="CSP"
    ["x-frame-options"]="X-Frame-Options"
    ["x-content-type-options"]="X-Content-Type-Options"
    ["referrer-policy"]="Referrer-Policy"
    ["permissions-policy"]="Permissions-Policy"
    ["x-xss-protection"]="X-XSS-Protection"
    ["cross-origin-resource-policy"]="CORP"
    ["cross-origin-opener-policy"]="COOP"
)

for header in "${!REQUIRED_HEADERS[@]}"; do
    if echo "$HEADERS" | grep -qi "^$header:"; then
        check_pass "${REQUIRED_HEADERS[$header]} 已存在"
    else
        check_fail "${REQUIRED_HEADERS[$header]} 缺失"
    fi
done
echo ""

echo "【4/6】检查敏感文件封锁..."
SENSITIVE_PATHS=(
    "/.git/config"
    "/.env"
    "/.svn/entries"
    "/.htaccess"
)

for path in "${SENSITIVE_PATHS[@]}"; do
    HTTP_CODE=$(curl -sI -o /dev/null -w "%{http_code}" "https://$DOMAIN$path")
    if [ "$HTTP_CODE" = "404" ] || [ "$HTTP_CODE" = "403" ]; then
        check_pass "敏感路径 $path 已封锁 (HTTP $HTTP_CODE)"
    elif [ "$HTTP_CODE" = "200" ]; then
        check_fail "敏感路径 $path 可公开访问 (HTTP 200)"
    else
        check_warn "敏感路径 $path 返回异常状态码: HTTP $HTTP_CODE"
    fi
done
echo ""

echo "【5/6】检查 SSL/TLS 配置..."
SSL_PROTOCOLS=$(echo | openssl s_client -connect "$DOMAIN:443" -servername "$DOMAIN" 2>/dev/null | grep "Protocol  :" | awk '{print $3}')
if [ -n "$SSL_PROTOCOLS" ]; then
    if [[ "$SSL_PROTOCOLS" == *"TLSv1.2"* ]] || [[ "$SSL_PROTOCOLS" == *"TLSv1.3"* ]]; then
        check_pass "SSL 协议: $SSL_PROTOCOLS"
    else
        check_warn "SSL 协议可能过旧: $SSL_PROTOCOLS"
    fi
else
    check_warn "无法检测 SSL 协议版本"
fi

SSL_CIPHER=$(echo | openssl s_client -connect "$DOMAIN:443" -servername "$DOMAIN" 2>/dev/null | grep "Cipher    :" | awk '{print $3}')
if [ -n "$SSL_CIPHER" ] && [ "$SSL_CIPHER" != "0000" ]; then
    check_pass "SSL 密码套件: $SSL_CIPHER"
else
    check_fail "SSL 密码套件异常或未连接"
fi
echo ""

echo "【6/6】检查 FTP 端口(21)..."
FTP_TEST=$(curl -s --connect-timeout 5 telnet://$DOMAIN:21 2>&1)
if [ $? -ne 0 ] || echo "$FTP_TEST" | grep -qi "refused\|timeout"; then
    check_pass "FTP 端口 21 已关闭"
else
    check_fail "FTP 端口 21 仍可访问"
fi
echo ""

echo "=========================================="
echo "验证结果汇总"
echo "=========================================="
echo -e "通过: ${GREEN}$PASS${NC}"
echo -e "失败: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ 所有安全检查通过!${NC}"
    exit 0
else
    echo -e "${RED}✗ 存在 $FAIL 项失败,请检查配置${NC}"
    exit 1
fi
