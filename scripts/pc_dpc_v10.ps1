# Policies on world model v10 (long memory), trained and evaluated on the PC.
#
# Usage (from starter-kit, after pc_wm_v10.ps1):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_dpc_v10.ps1
#
# Steps:
#   1. three policies are fine-tuned from transformer_v6 on world model v10
#      (400 iterations, start states from buildings in
#      the official size range):
#        v10_base  same recipe as transformer_v6 (6 h then 12 h, gradient decay 0.95);
#        v10_base10 the same without the comfort correction (1.0), since v10 no
#                  longer underestimates the comfort term on clean episodes;
#        v10_long  12 h then 24 h horizon, gradient decay 0.99, which failed on v9
#                  and tells whether the memory makes long horizons usable;
#   2. all are evaluated over 7 days on the 30 official scenarios and on
#      bench2, reusing the pid_water and transformer_v6 (abl_A) runs;
#   3. policies, training logs and results are committed and pushed.
# Finished steps are skipped on a rerun.
param([int]$Workers = 10, [int]$Threads = 8)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path
if (-not (Test-Path "world_models/wm_v10.pt")) { throw "world_models/wm_v10.pt missing: run pc_wm_v10.ps1 first" }

$train = @("month_data/episodes")
foreach ($d in "pc_data/episodes", "pc_data/episodes_onpolicy", "pc_data/episodes_onpolicy_v4") {
    if (Test-Path $d) { $train += $d }
}
$train += Get-ChildItem month_data -Directory -Filter "episodes_onpolicy_*" | ForEach-Object { "month_data/" + $_.Name }

$variants = [ordered]@{
    "base" = @("--lengths", "72", "144", "--grad-decay", "0.95", "--comfort-scale", "1.8");
    "base10" = @("--lengths", "72", "144", "--grad-decay", "0.95", "--comfort-scale", "1.0");
    "long" = @("--lengths", "144", "288", "--grad-decay", "0.99", "--comfort-scale", "1.8")
}
foreach ($name in $variants.Keys) {
    if (Test-Path "policies/v10_$name.pt") { Write-Host "variant $name already trained"; continue }
    $out = "wm_runs/v10_$name"
    $extra = @($variants[$name])
    & $py -m sbwm.train_dpc --wm world_models/wm_v10.pt --train @train --test month_data/episodes_test `
        --exclude-buildings month_data/buildings_list.csv --out $out --iters 400 `
        --init policies/transformer_v6.pt --eval-every 200 --threads $Threads --pool 60 --refresh 100 @extra
    if ($LASTEXITCODE -ne 0) { throw "training of variant $name failed" }
    Copy-Item "$out/policy_it00400.pt" "policies/v10_$name.pt"
    Copy-Item "$out/dpc_log.json" "results/train_log_v10_$name.json"
}

foreach ($b in "official30", "bench2") {
    if (-not (Test-Path "results/v10_$b.json")) { Copy-Item "results/ablation_$b.json" "results/v10_$b.json" }
}
foreach ($name in $variants.Keys) {
    $w = "policies/v10_$name.pt"
    & $py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies pid_water dpc `
        --weights $w --steps 2016 --workers $Workers --out results/v10_official30.json
    & $py -m sbwm.eval_policy --scenarios bench2_data/scenarios --policies pid_water dpc `
        --weights $w --steps 2016 --workers $Workers --out results/v10_bench2.json
}

git add policies/v10_*.pt results/train_log_v10_*.json results/v10_official30.json results/v10_bench2.json
git commit -m "Policies on world model v10 with evaluations"
git pull --rebase github main
git push github main
