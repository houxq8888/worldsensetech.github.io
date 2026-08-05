# setup-sync-task.ps1
# 设置博客自动同步的Windows任务计划
# 以管理员身份运行此脚本

$TaskName = "BlogSync"
$ScriptPath = "D:\virtualMachine\github\GEOPro-git\GEOPro\idea\worldsense-blog\scripts\sync-blog.ps1"

# 创建任务动作
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$ScriptPath`""

# 创建触发器：每小时运行一次
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)

# 任务设置
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# 注册任务
try {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "博客自动同步 - 每小时检查远程更新并拉取" -Force
    Write-Host "任务创建成功：$TaskName" -ForegroundColor Green
    Write-Host "每小时自动同步一次博客仓库" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "管理命令：" -ForegroundColor Yellow
    Write-Host "  查看任务：Get-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  手动运行：Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  删除任务：Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
} catch {
    Write-Host "创建任务失败：$_" -ForegroundColor Red
    Write-Host "请确保以管理员身份运行此脚本" -ForegroundColor Yellow
}
