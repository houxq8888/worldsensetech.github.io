# blog-auto-publish.ps1
# 博客自动发布+同步脚本
# 功能：检查草稿→发布→推送→拉取远程更新

param(
    [string]$RepoPath = "D:\virtualMachine\github\GEOPro-git\GEOPro\idea\worldsense-blog",
    [switch]$Verbose
)

$LogFile = Join-Path $RepoPath "auto-publish-log.txt"
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

Set-Location $RepoPath

if (-not (Test-Path ".git")) {
    Write-Log "Error: $RepoPath is not a git repo"
    exit 1
}

Write-Log "=== Starting auto-publish check ==="

# Step 1: Check for drafts ready to publish
$DraftsDir = Join-Path $RepoPath "_drafts"
$Today = (Get-Date).ToString("yyyy-MM-dd")
$ReadyDrafts = @()

if (Test-Path $DraftsDir) {
    $DraftFiles = Get-ChildItem -Path $DraftsDir -Filter "*.html" -ErrorAction SilentlyContinue
    foreach ($Draft in $DraftFiles) {
        # Check if filename starts with today's date or earlier
        if ($Draft.Name -match "^(\d{4}-\d{2}-\d{2})-") {
            $DraftDate = $matches[1]
            if ($DraftDate -le $Today) {
                $ReadyDrafts += $Draft
                Write-Log "Found ready draft: $($Draft.Name)"
            }
        }
    }
}

if ($ReadyDrafts.Count -gt 0) {
    Write-Log "Found $($ReadyDrafts.Count) draft(s) ready to publish"
    
    # Step 2: Run publish script
    $PublishScript = Join-Path $RepoPath "scripts\publish_drafts.py"
    if (Test-Path $PublishScript) {
        Write-Log "Running publish script..."
        $PublishOutput = python $PublishScript 2>&1
        $PublishExitCode = $LASTEXITCODE
        
        Write-Log "Publish output: $PublishOutput"
        
        if ($PublishExitCode -eq 0) {
            Write-Log "Publish script completed successfully"
            
            # Step 3: Commit and push
            Write-Log "Committing and pushing changes..."
            
            git add -A 2>&1 | Out-Null
            $Status = git status --porcelain 2>&1
            
            if ($Status) {
                git config user.name "github-actions[bot]" 2>&1 | Out-Null
                git config user.email "github-actions[bot]@users.noreply.github.com" 2>&1 | Out-Null
                git commit -m "Auto-publish scheduled draft(s)" 2>&1 | Out-Null
                
                $PushOutput = git push origin main 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Log "Push successful"
                } else {
                    Write-Log "Push failed: $PushOutput"
                }
            } else {
                Write-Log "No changes to commit"
            }
        } else {
            Write-Log "Publish script failed with exit code $PublishExitCode"
        }
    } else {
        Write-Log "Publish script not found: $PublishScript"
    }
} else {
    Write-Log "No drafts ready to publish"
}

# Step 4: Sync with remote (pull any changes from GitHub Actions)
Write-Log "Checking for remote updates..."

$FetchOutput = git fetch origin 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Log "git fetch failed: $FetchOutput"
    exit 1
}

$BehindCount = git rev-list HEAD..origin/main --count 2>$null
if ([int]$BehindCount -gt 0) {
    Write-Log "Found $BehindCount remote updates, pulling..."
    
    # Stash local changes if any
    $Status = git status --porcelain 2>$null
    if ($Status) {
        Write-Log "Stashing local changes..."
        git stash push -m "auto-publish-sync-$Timestamp" 2>&1 | Out-Null
        $Stashed = $true
    }
    
    $PullOutput = git pull --rebase origin main 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Pull successful"
    } else {
        Write-Log "Pull failed: $PullOutput"
    }
    
    if ($Stashed) {
        Write-Log "Restoring stashed changes..."
        git stash pop 2>&1 | Out-Null
    }
} else {
    Write-Log "Already up to date"
}

Write-Log "=== Auto-publish check completed ==="
