# Build colab_bundle.zip for Google Colab (PLAN_v2). Run from anywhere:  .\scripts\make_colab_bundle.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$stage = Join-Path $root "_bundle_stage"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory $stage | Out-Null
Copy-Item -Recurse .\src, .\scripts, .\tests $stage
New-Item -ItemType Directory (Join-Path $stage "weak"), (Join-Path $stage "annotations") | Out-Null
Copy-Item .\weak\* (Join-Path $stage "weak")
Copy-Item .\annotations\weak_labels.csv (Join-Path $stage "annotations")
Copy-Item .\lexicon.csv, .\PLAN_v2.md, .\requirements.txt $stage
Get-ChildItem $stage -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
$zip = Join-Path $root "colab_bundle.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Write-Host "Compressing (this can take a few minutes) ..."
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip
Remove-Item -Recurse -Force $stage
$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "Built $zip ($mb MB) at $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
