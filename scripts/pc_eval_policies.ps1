# Evaluate ready policies on the 30 official scenarios, draw their control
# figures and push, one policy at a time.
#
# Usage (from starter-kit):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_eval_policies.ps1 `
#         -Policies policies/abl_A10_f03.pt,policies/abl_A10_f05.pt -Out results/filter_official30.json
#
# Steps, for each policy:
#   1. 7-day evaluation on the official scenarios (pid_water, transformer_v6
#      and A10 are reused from the ablation file), recording the standard
#      inspection scenarios; the summary is printed on the terminal;
#   2. control figures against pid_water, transformer_v6 and A10;
#   3. policy, results and figures are committed and pushed.
param([string[]]$Policies = @(), [string]$Out = "results/extra_official30.json", [int]$Workers = 10)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path

if (-not (Test-Path $Out)) { Copy-Item "results/ablation_official30.json" $Out }
$refs = @("pid_water", "policies/transformer_v6.pt", "policies/abl_A10.pt")
foreach ($w in $Policies) {
    & $py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies pid_water dpc `
        --weights $w --steps 2016 --workers $Workers --record --out $Out
    if ($LASTEXITCODE -ne 0) { throw "evaluation of $w failed" }
    & $py -m sbwm.inspect_control --policies @refs $w --steps 2016 --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "figures of $w failed" }
    git add $w $Out results/inspect
    git commit -m "Evaluation and control figures of ${w}"
    git pull --rebase github main
    git push github main
}
