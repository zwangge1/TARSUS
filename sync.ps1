# One-click commit + push for the TARSUS repo (run inside the TARSUS folder)
# Usage: .\sync.ps1 "commit message"
param([string]$m = "update")

git add -A
git commit -m $m

$ok = $false
for ($i = 1; $i -le 5; $i++) {
    git push
    if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    Write-Host "push failed (try $i/5), retry in 3s... (github.com:443 flaky in CN)" -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}

if ($ok) {
    Write-Host "`nPushed OK. On A100 run: git pull" -ForegroundColor Green
} else {
    Write-Host "`nPush still failing. Either retry later, or set a proxy:" -ForegroundColor Red
    Write-Host "  git config --global http.proxy http://127.0.0.1:7890" -ForegroundColor Red
    Write-Host "  git config --global https.proxy http://127.0.0.1:7890" -ForegroundColor Red
}
