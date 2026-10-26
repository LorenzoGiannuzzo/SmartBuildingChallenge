# Long-horizon credit on world model v9: longer DPC horizon and a learned critic.
#
# Usage (from starter-kit, after pc_onpolicy_round.ps1 has produced wm_v9):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_critic.ps1
#
# Steps:
#   1. three policies are fine-tuned from transformer_v6 on world model v9
#      (comfort correction 1.8, 400 iterations, start states from buildings in
#      the official size range):
#        v9_base    same recipe as transformer_v6 (6 h then 12 h, gradient decay 0.95),
#                   to separate the effect of the new world model;
#        v9_long    12 h then 24 h horizon, gradient decay 0.99;
#        v9_critic  6 h then 12 h, gradient decay 0.98, critic trained on 48 h rollouts,
#                   discount 0.997 (about 28 h), 400 critic-only warm-up iterations;
#   2. the three are evaluated over 7 days on the 30 official scenarios and on
#      bench2, reusing the pid_water and transformer_v6 (abl_A) runs of the ablation;
#   3. policies, training logs and results are committed and pushed.
# Finished steps are skipped on a rerun.
param([int]$Workers = 10, [int]$Threads = 8)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path
if (-not (Test-Path "world_models/wm_v9.pt")) { throw "world_models/wm_v9.pt missing: run pc_onpolicy_round.ps1 first" }

$train = @("month_data/episodes")
foreach ($d in "pc_data/episodes", "pc_data/episodes_onpolicy", "pc_data/episodes_onpolicy_v4") {
    if (Test-Path $d) { $train += $d }
}
$train += Get-ChildItem month_data -Directory -Filter "episodes_onpolicy_*" | ForEach-Object { "month_data/" + $_.Name }

$variants = [ordered]@{
    "base" = @("--lengths", "72", "144", "--grad-decay", "0.95");
    "long" = @("--lengths", "144", "288", "--grad-decay", "0.99");
    "critic" = @("--lengths", "72", "144", "--grad-decay", "0.98", "--gamma", "0.997", "--critic", "--critic-warmup", "400", "--critic-updates", "4")
}
foreach ($name in $variants.Keys) {
    if (Test-Path "policies/v9_$name.pt") { Write-Host "variant $name already trained"; continue }
    $out = "wm_runs/v9_$name"
    $extra = @($variants[$name])
    & $py -m sbwm.train_dpc --wm world_models/wm_v9.pt --train @train --test month_data/episodes_test `
        --exclude-buildings month_data/buildings_list.csv --out $out --iters 400 --comfort-scale 1.8 `
        --init policies/transformer_v6.pt --eval-every 200 --threads $Threads --pool 60 --refresh 100 @extra
    if ($LASTEXITCODE -ne 0) { throw "training of variant $name failed" }
    Copy-Item "$out/policy_it00400.pt" "policies/v9_$name.pt"
    Copy-Item "$out/dpc_log.json" "results/train_log_v9_$name.json"
}

#Lorenzo Giannuzzo: the ablation files already hold pid_water and
#transformer_v6 (abl_A) on the same scenarios, so they are the starting point.
foreach ($b in "official30", "bench2") {
    if (-not (Test-Path "results/critic_$b.json")) { Copy-Item "results/ablation_$b.json" "results/critic_$b.json" }
}
foreach ($name in $variants.Keys) {
    $w = "policies/v9_$name.pt"
    & $py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies pid_water dpc `
        --weights $w --steps 2016 --workers $Workers --out results/critic_official30.json
    & $py -m sbwm.eval_policy --scenarios bench2_data/scenarios --policies pid_water dpc `
        --weights $w --steps 2016 --workers $Workers --out results/critic_bench2.json
}

git add policies/v9_*.pt results/train_log_v9_*.json results/critic_official30.json results/critic_bench2.json
git commit -m "Long-horizon credit on world model v9: base, long horizon and critic, with evaluations"
git pull --rebase github main
git push github main
