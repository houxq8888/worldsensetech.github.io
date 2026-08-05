# setup-auto-publish-task.ps1
# Setup Windows scheduled task for blog auto-publish
# Run as Administrator

$TaskName = "BlogAutoPublish"
$ScriptPath = "D:\virtualMachine\github\GEOPro-git\GEOPro\idea\worldsense-blog\scripts\blog-auto-publish.ps1"

$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$ScriptPath`""
$Trigger = New-ScheduledTaskTrigger -Daily -At "08:00"
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

try {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Blog auto-publish - check drafts and sync remote" -Force
    Write-Host "Task created: $TaskName" -ForegroundColor Green
    Write-Host "Schedule: Daily at 08:00" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Functions:" -ForegroundColor Yellow
    Write-Host "  1. Check _drafts/ for due articles"
    Write-Host "  2. Publish to articles/"
    Write-Host "  3. Commit and push to GitHub"
    Write-Host "  4. Pull remote updates"
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  View: Get-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  Run: Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  Delete: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
} catch {
    Write-Host "Failed: $_" -ForegroundColor Red
    Write-Host "Please run as Administrator" -ForegroundColor Yellow
}
