# Control figures for trained policies (dashboards and comparisons).
#
# Usage (from starter-kit):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_inspect.ps1 -Policies policies/v10_base.pt policies/v10_long.pt
#
# Steps:
#   1. each policy (plus pid_water, transformer_v6 and A10 as references) is
#      run for 7 days on the standard scenarios and every step is recorded;
#      finished recordings are reused;
#   2. dashboards go to results/inspect/<policy>/ and comparisons to
#      results/inspect/compare/;
#   3. recordings and figures are committed and pushed.
param([string[]]$Policies = @(), [int]$Workers = 10)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path

$all = @("pid_water", "policies/transformer_v6.pt", "policies/abl_A10.pt") + $Policies
& $py -m sbwm.inspect_control --policies @all --steps 2016 --workers $Workers
if ($LASTEXITCODE -ne 0) { throw "inspection failed" }

git add results/inspect
git commit -m "Control figures for $($Policies -join ', ')"
git pull --rebase github main
git push github main
