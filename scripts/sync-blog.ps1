# sync-blog.ps1
# Blog auto-sync script - check remote updates and pull

param(
    [string]$RepoPath = "D:\virtualMachine\github\GEOPro-git\GEOPro\idea\worldsense-blog",
    [switch]$Verbose
)

$LogFile = Join-Path $RepoPath "sync-log.txt"
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

function Write-Log {
    param([string]$Message)
    $LogEntry = "[$Timestamp] $Message"
    if ($Verbose) { Write-Host $LogEntry }
    Add-Content -Path $LogFile -Value $LogEntry -Encoding UTF8
}

# Check if repo exists
if (-not (Test-Path $RepoPath)) {
    Write-Log "Error: Repo path not found $RepoPath"
    exit 1
}

# Change to repo directory
Set-Location $RepoPath

# Check if it's a git repo
if (-not (Test-Path ".git")) {
    Write-Log "Error: $RepoPath is not a git repo"
    exit 1
}

Write-Log "Starting sync check..."

# Fetch remote updates
try {
    $FetchOutput = git fetch origin 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Log "git fetch failed: $FetchOutput"
        exit 1
    }
} catch {
    Write-Log "git fetch exception: $_"
    exit 1
}

# Check if there are remote updates
$BehindCount = git rev-list HEAD..origin/main --count 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Log "git rev-list failed"
    exit 1
}

if ([int]$BehindCount -gt 0) {
    Write-Log "Found $BehindCount remote updates, pulling..."
    
    # Check for local uncommitted changes
    $Status = git status --porcelain 2>$null
    if ($Status) {
        Write-Log "Local changes detected, stashing..."
        git stash push -m "auto-sync-stash-$Timestamp" 2>$null
        $Stashed = $true
    }
    
    # Pull remote updates
    try {
        $PullOutput = git pull --rebase origin main 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Log "git pull failed: $PullOutput"
            if ($Stashed) {
                git stash pop 2>$null
            }
            exit 1
        }
        Write-Log "Pull success: $PullOutput"
    } catch {
        Write-Log "git pull exception: $_"
        if ($Stashed) {
            git stash pop 2>$null
        }
        exit 1
    }
    
    # Restore stashed changes
    if ($Stashed) {
        Write-Log "Restoring stashed changes..."
        $StashPopOutput = git stash pop 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Warning: stash pop failed, please run git stash pop manually"
        }
    }
    
    Write-Log "Sync completed"
} else {
    Write-Log "Already up to date, no sync needed"
}

# Clean up logs older than 30 days
if (Test-Path $LogFile) {
    $LogAge = (Get-Item $LogFile).LastWriteTime
    if (((Get-Date) - $LogAge).Days -gt 30) {
        $Lines = Get-Content $LogFile -Encoding UTF8
        $Cutoff = (Get-Date).AddDays(-30).ToString("yyyy-MM-dd")
        $Filtered = $Lines | Where-Object { $_ -match "^\[(\d{4}-\d{2}-\d{2})" -and $matches[1] -ge $Cutoff }
        $Filtered | Set-Content $LogFile -Encoding UTF8
    }
}
