# Architecture ablation on world model v8, trained and evaluated on the PC.
#
# Usage (from starter-kit):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_ablation.ps1
#
# Steps:
#   1. five policies are fine-tuned with DPC on world model v8 (6 h then 12 h
#      horizon, comfort correction 1.8, 400 iterations), with start states from
#      the monthly and PC episodes of buildings in the official size range:
#        A  transformer only               (from dpc_v5)
#        B  transformer + zone PI          (from hybrid_v1)
#        C  B + derivative term            (from hybrid_v1)
#        D  C + winter plant floor         (from hybrid_v1)
#        E  D + summer supply-air ceiling  (from hybrid_v1)
#   2. bench2 (15 unseen buildings, official size range and stations) is built;
#   3. pid_water, hybrid_v1, hybrid_v4 and A-E are evaluated over 7 days on the
#      30 official scenarios and on bench2;
#   4. policies and results are committed and pushed.
# Finished steps are skipped on a rerun.
param([int]$Workers = 10, [int]$Threads = 8)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path

$train = @("month_data/episodes")
foreach ($d in "pc_data/episodes", "pc_data/episodes_onpolicy", "pc_data/episodes_onpolicy_v4") {
    if (Test-Path $d) { $train += $d }
}
$train += Get-ChildItem month_data -Directory -Filter "episodes_onpolicy_*" | ForEach-Object { "month_data/" + $_.Name }

$pi0 = "margin=0.1,ki=0.02"
$variants = [ordered]@{
    "A" = @("--init", "policies/dpc_v5.pt");
    "B" = @("--init", "policies/hybrid_v1.pt", "--hybrid", "--pi", $pi0);
    "C" = @("--init", "policies/hybrid_v1.pt", "--hybrid", "--pi", "$pi0,kd=2");
    "D" = @("--init", "policies/hybrid_v1.pt", "--hybrid", "--pi", "$pi0,kd=2,plant=1,run_thr=0.7,hws_thr=0.25,water_gain=2,cool_gain=0");
    "E" = @("--init", "policies/hybrid_v1.pt", "--hybrid", "--pi", "$pi0,kd=2,plant=1,run_thr=0.7,hws_thr=0.25,water_gain=2,cool_gain=2")
}
foreach ($name in $variants.Keys) {
    if (Test-Path "policies/abl_$name.pt") { Write-Host "variant $name already trained"; continue }
    $out = "wm_runs/abl_$name"
    & $py -m sbwm.train_dpc --wm world_models/wm_v8.pt --train @train --test month_data/episodes_test `
        --exclude-buildings month_data/buildings_list.csv --out $out --iters 400 --lengths 72 144 `
        --eval-every 200 --threads $Threads --comfort-scale 1.8 --pool 60 --refresh 100 @($variants[$name])
    if ($LASTEXITCODE -ne 0) { throw "training of variant $name failed" }
    Copy-Item "$out/policy_it00400.pt" "policies/abl_$name.pt"
}

& $py -m sbwm.make_benchmark --default-every 0 --stations atlanta_1 denver_1 fairbanks_1 phoenix_1 singapore_1 `
    --per-climate 3 --start-seed 3000 --root bench2_data
foreach ($w in @("policies/hybrid_v1.pt", "policies/hybrid_v4.pt") + ($variants.Keys | ForEach-Object { "policies/abl_$_.pt" })) {
    & $py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies pid_water dpc `
        --weights $w --steps 2016 --workers $Workers --out results/ablation_official30.json
    & $py -m sbwm.eval_policy --scenarios bench2_data/scenarios --policies pid_water dpc `
        --weights $w --steps 2016 --workers $Workers --out results/ablation_bench2.json
}

git add policies/abl_*.pt results/ablation_official30.json results/ablation_bench2.json
git commit -m "Architecture ablation A-E on world model v8 with evaluations"
git pull --rebase github main
git push github main
