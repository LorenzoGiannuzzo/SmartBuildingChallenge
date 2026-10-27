# Clean on-policy episodes for the world-model diagnosis.
#
# Usage (from starter-kit):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_clean_collect.ps1
#
# Steps:
#   1. one 7-day episode per official scenario (30) and per bench2 scenario
#      (15), driven by transformer_v6 alone (no exploration segments), from a
#      random start date (about 1 hour with 10 workers);
#   2. the episodes are committed and pushed (they are small, about 0.6 MB
#      each); they are used only to measure the world-model error, never for
#      training.
param([string]$Policy = "policies/transformer_v6.pt", [int]$Workers = 10, [int]$Seed = 31)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path

& $py -m sbwm.collect --scenarios smart_buildings/training_scenarios --out diag_clean/official --monthly `
    --steps 2016 --dpc-weights $Policy --clean --workers $Workers --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "collection on the official scenarios failed" }
& $py -m sbwm.collect --scenarios bench2_data/scenarios --out diag_clean/bench2 --monthly `
    --steps 2016 --dpc-weights $Policy --clean --workers $Workers --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "collection on bench2 failed" }

git add diag_clean
git commit -m "Clean transformer_v6 episodes for the world-model diagnosis"
git pull --rebase github main
git push github main
