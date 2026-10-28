# Policies on world model v10 (long memory), trained and evaluated on the PC.
#
# Usage (from starter-kit, after pc_wm_v10.ps1):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_dpc_v10.ps1
#
# Steps:
#   1. three policies are fine-tuned from transformer_v6 on world model v10
#      (400 iterations, start states from buildings in the official size range):
#        v10_base10 the recipe of transformer_v6 without comfort correction (1.0),
#                   since v10 no longer underestimates the comfort term;
#        v10_base   the same with the comfort correction 1.8;
#        v10_long   12 h then 24 h horizon, gradient decay 0.99, which failed on v9
#                   and tells whether the memory makes long horizons usable;
#   2. after each training the policy is evaluated over 7 days on the 30
#      official scenarios (recording the standard inspection scenarios), its
#      control figures are drawn against pid_water, transformer_v6 and A10, and
#      everything is pushed;
#   3. bench2 (unseen buildings) is evaluated last for all three and pushed.
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

#Lorenzo Giannuzzo: the variant without comfort correction goes first, since
#A10 beat transformer_v6 on Kaggle and v10 no longer underestimates comfort.
$variants = [ordered]@{
    "base10" = @("--lengths", "72", "144", "--grad-decay", "0.95", "--comfort-scale", "1.0");
    "base" = @("--lengths", "72", "144", "--grad-decay", "0.95", "--comfort-scale", "1.8");
    "long" = @("--lengths", "144", "288", "--grad-decay", "0.99", "--comfort-scale", "1.8")
}
foreach ($b in "official30", "bench2") {
    if (-not (Test-Path "results/v10_$b.json")) { Copy-Item "results/ablation_$b.json" "results/v10_$b.json" }
}
$refs = @("pid_water", "policies/transformer_v6.pt", "policies/abl_A10.pt")

#Lorenzo Giannuzzo: each variant is trained, evaluated on the official
#scenarios (the ones Kaggle follows), drawn and pushed before the next one
#starts, so its results are on the terminal and in the repository as soon as
#they exist.
foreach ($name in $variants.Keys) {
    $w = "policies/v10_$name.pt"
    if (-not (Test-Path $w)) {
        $out = "wm_runs/v10_$name"
        $extra = @($variants[$name])
        & $py -m sbwm.train_dpc --wm world_models/wm_v10.pt --train @train --test month_data/episodes_test `
            --exclude-buildings month_data/buildings_list.csv --out $out --iters 400 `
            --init policies/transformer_v6.pt --eval-every 200 --threads $Threads --pool 60 --refresh 100 @extra
        if ($LASTEXITCODE -ne 0) { throw "training of variant $name failed" }
        Copy-Item "$out/policy_it00400.pt" $w
        Copy-Item "$out/dpc_log.json" "results/train_log_v10_$name.json"
    }
    & $py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies pid_water dpc `
        --weights $w --steps 2016 --workers $Workers --record --out results/v10_official30.json
    if ($LASTEXITCODE -ne 0) { throw "evaluation of variant $name failed" }
    & $py -m sbwm.inspect_control --policies @refs $w --steps 2016 --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "figures of variant $name failed" }
    git add $w results/train_log_v10_$name.json results/v10_official30.json results/inspect
    git commit -m "Policy v10_$name: official evaluation and control figures"
    git pull --rebase github main
    git push github main
}

#Lorenzo Giannuzzo: bench2 (unseen buildings) as a final check, after all
#official results are out.
foreach ($name in $variants.Keys) {
    & $py -m sbwm.eval_policy --scenarios bench2_data/scenarios --policies pid_water dpc `
        --weights "policies/v10_$name.pt" --steps 2016 --workers $Workers --out results/v10_bench2.json
}
git add results/v10_bench2.json
git commit -m "Policies on world model v10: bench2 evaluations"
git pull --rebase github main
git push github main
