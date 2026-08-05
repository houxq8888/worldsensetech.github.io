# setup-auto-publish-task.ps1
# 设置博客自动发布+同步的Windows任务计划
# 以管理员身份运行此脚本

$TaskName = "BlogAutoPublish"
$ScriptPath = "D:\virtualMachine\github\GEOPro-git\GEOPro\idea\worldsense-blog\scripts\blog-auto-publish.ps1"

# 创建任务动作
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$ScriptPath`""

# 创建触发器：每天早上8:00运行（北京时间）
$Trigger = New-ScheduledTaskTrigger -Daily -At "08:00"

# 任务设置
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# 注册任务
try {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "博客自动发布 - 每天检查草稿并发布，同时同步远程更新" -Force
    Write-Host "任务创建成功：$TaskName" -ForegroundColor Green
    Write-Host "每天早上8:00自动运行" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "功能：" -ForegroundColor Yellow
    Write-Host "  1. 检查_drafts/中到期的草稿"
    Write-Host "  2. 自动发布到articles/"
    Write-Host "  3. 提交并推送到GitHub"
    Write-Host "  4. 拉取远程更新（保持本地同步）"
    Write-Host ""
    Write-Host "管理命令：" -ForegroundColor Yellow
    Write-Host "  查看任务：Get-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  手动运行：Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  删除任务：Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
} catch {
    Write-Host "创建任务失败：$_" -ForegroundColor Red
    Write-Host "请确保以管理员身份运行此脚本" -ForegroundColor Yellow
}
