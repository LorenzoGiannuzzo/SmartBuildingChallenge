# Evaluation of the policies trained on world model v9 (policies/v9_*.pt).
#
# Usage (from starter-kit):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_eval_v9.ps1
#
# Steps:
#   1. every policies/v9_*.pt is evaluated over 7 days on the 30 official
#      scenarios and on bench2, reusing the pid_water and transformer_v6
#      (abl_A) runs of the ablation; finished runs are skipped;
#   2. policies, training logs and results are committed and pushed.
param([int]$Workers = 10)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path

foreach ($b in "official30", "bench2") {
    if (-not (Test-Path "results/critic_$b.json")) { Copy-Item "results/ablation_$b.json" "results/critic_$b.json" }
}
foreach ($w in Get-ChildItem policies -Filter "v9_*.pt" | Sort-Object Name) {
    $wp = "policies/" + $w.Name
    & $py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies pid_water dpc `
        --weights $wp --steps 2016 --workers $Workers --record --out results/critic_official30.json
    & $py -m sbwm.eval_policy --scenarios bench2_data/scenarios --policies pid_water dpc `
        --weights $wp --steps 2016 --workers $Workers --out results/critic_bench2.json
}
foreach ($d in Get-ChildItem wm_runs -Directory -Filter "v9_*") {
    if (Test-Path "$($d.FullName)/dpc_log.json") { Copy-Item "$($d.FullName)/dpc_log.json" "results/train_log_$($d.Name).json" }
}

git add policies/v9_*.pt results/train_log_v9_*.json results/critic_official30.json results/critic_bench2.json
git commit -m "Policies on world model v9 with evaluations"
git pull --rebase github main
git push github main
