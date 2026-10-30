# One-time setup of the TARSUS git repo. Run ONCE inside the TARSUS folder.
# Usage: .\init_repo.ps1 https://github.com/zwangge1/TARSUS.git
# It only touches this folder. It does not touch FFY\FFY (STILT) or EvoWM.
param([Parameter(Mandatory=$true)][string]$remote)

if (Test-Path ".git") { Write-Host "Already a git repo, abort." -ForegroundColor Red; exit 1 }

git init -b main
# Commit 1: pristine upstream DeepLabCut (fa5a8c3), nothing else
git add third_party/DeepLabCut third_party/UPSTREAM.md
git commit -m "upstream: DeepLabCut fa5a8c3 (pristine copy, no modification)"
# Commit 2: TARSUS skeleton
git add -A
git commit -m "TARSUS skeleton: README, prereg seg_w1, design doc 06, sync script"
git remote add origin $remote
git push -u origin main
if ($LASTEXITCODE -eq 0) { Write-Host "`nDone. On A100: git clone $remote" -ForegroundColor Green }
else { Write-Host "`nPush failed, run: git push -u origin main (retry / proxy)" -ForegroundColor Red }
