# On-policy round with v10_base_f03, world model v11, new policies and 14-day evaluations.
#
# Usage (from starter-kit):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_round_v11.ps1
#
# Steps (each one is skipped when its output exists, and results are pushed
# as soon as they are ready, so the script can be rerun after an interruption):
#   1. one 14-day episode per training building (random climate) driven by
#      v10_base_f03 with exploration segments (about 7 hours);
#   2. world model v11: v10 fine-tuned with the long memory and 4-24 h unrolls,
#      half of the samples from these episodes (about 4 hours), then diagnosed
#      against v10 on the clean episodes;
#   3. three policies from transformer_v6 on v11 (comfort correction 1.8):
#        v11_base_f03   recipe of v10_base, command filter 0.3 at deployment;
#        v11_fit3       the same with the filter inside the training loop;
#        v11_long_fit3  12 h then 24 h horizon, gradient decay 0.99, filter in
#                       the loop (the long horizon against the weekend effect);
#   4. 14-day evaluation on the 30 official scenarios (each window contains a
#      Monday after a real weekend, as on Kaggle) of pid_water, A10,
#      v10_base_f03 and the three new policies, with the 7-day control figures
#      of the new ones.
param([int]$Workers = 10, [int]$Threads = 8)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path

function Push-Results([string[]]$Paths, [string]$Message) {
    git add @Paths
    git commit -m $Message
    git pull --rebase github main
    git push github main
}

# 1. on-policy collection
$onp = "month_data/episodes_onpolicy_f03"
& $py -m sbwm.collect --scenarios month_data/scenarios --out $onp --monthly --one-station `
    --steps 4032 --dpc-weights policies/v10_base_f03.pt --workers $Workers --seed 41
if ($LASTEXITCODE -ne 0) { throw "collection failed" }

# 2. world model v11
$train = @("month_data/episodes")
if (Test-Path "pc_data/episodes") { $train += "pc_data/episodes" }
$train += Get-ChildItem month_data -Directory -Filter "episodes_onpolicy_*" | ForEach-Object { "month_data/" + $_.Name }
if (-not (Test-Path "world_models/wm_v11.pt")) {
    & $py -m sbwm.train_wm --train @train --emphasis $onp --emphasis-share 0.5 `
        --test month_data/episodes_test --out wm_runs/v11 --init world_models/wm_v10.pt --mem `
        --steps 1000 --batch 16 --lr 5e-5 --lengths 48 144 288 --w-comfort 5 --pool 40 --refresh 200 `
        --threads $Threads --eval-every 250
    if ($LASTEXITCODE -ne 0) { throw "world model training failed" }
    Copy-Item wm_runs/v11/world_model_best.pt world_models/wm_v11.pt -Force
    Copy-Item wm_runs/v11/eval_log.json world_models/wm_v11_eval_log.json -Force
    & $py -m sbwm.diagnose_wm --wm world_models/wm_v10.pt world_models/wm_v11.pt --policy policies/transformer_v6.pt `
        --episodes diag_clean/official diag_clean/bench2 --threads $Threads --out results/wm_diagnosis_v10_v11.json
    if ($LASTEXITCODE -ne 0) { throw "diagnosis failed" }
    Push-Results @("world_models/wm_v11.pt", "world_models/wm_v11_eval_log.json", "results/wm_diagnosis_v10_v11.json") `
        "World model v11 (on-policy round with v10_base_f03) and diagnosis against v10"
}

# 3. policies on v11
$dpcTrain = @("month_data/episodes")
foreach ($d in "pc_data/episodes", "pc_data/episodes_onpolicy", "pc_data/episodes_onpolicy_v4") {
    if (Test-Path $d) { $dpcTrain += $d }
}
$dpcTrain += Get-ChildItem month_data -Directory -Filter "episodes_onpolicy_*" | ForEach-Object { "month_data/" + $_.Name }
$variants = [ordered]@{
    "base" = @("--lengths", "72", "144", "--grad-decay", "0.95");
    "fit3" = @("--lengths", "72", "144", "--grad-decay", "0.95", "--action-filter", "0.3");
    "long_fit3" = @("--lengths", "144", "288", "--grad-decay", "0.99", "--action-filter", "0.3")
}
foreach ($name in $variants.Keys) {
    $w = "policies/v11_$name.pt"
    if (Test-Path $w) { continue }
    $out = "wm_runs/v11_$name"
    $extra = @($variants[$name])
    & $py -m sbwm.train_dpc --wm world_models/wm_v11.pt --train @dpcTrain --test month_data/episodes_test `
        --exclude-buildings month_data/buildings_list.csv --out $out --iters 400 --comfort-scale 1.8 `
        --init policies/transformer_v6.pt --eval-every 200 --threads $Threads --pool 60 --refresh 100 @extra
    if ($LASTEXITCODE -ne 0) { throw "training of v11_$name failed" }
    Copy-Item "$out/policy_it00400.pt" $w
    Copy-Item "$out/dpc_log.json" "results/train_log_v11_$name.json"
    $paths = @($w, "results/train_log_v11_$name.json")
    if ($name -eq "base") {
        & $py -m sbwm.set_filter $w policies/v11_base_f03.pt 0.3
        $paths += "policies/v11_base_f03.pt"
    }
    Push-Results $paths "Policy v11_${name} on world model v11"
}

# 4. 14-day evaluations and control figures
$refs = @("pid_water", "policies/transformer_v6.pt", "policies/abl_A10.pt")
foreach ($w in "policies/abl_A10.pt", "policies/v10_base_f03.pt", "policies/v11_base_f03.pt",
               "policies/v11_fit3.pt", "policies/v11_long_fit3.pt") {
    & $py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies pid_water dpc `
        --weights $w --steps 4032 --workers $Workers --out results/official30_14d.json
    if ($LASTEXITCODE -ne 0) { throw "evaluation of $w failed" }
    $paths = @("results/official30_14d.json")
    if ($w -like "*v11_*") {
        & $py -m sbwm.inspect_control --policies @refs $w --steps 2016 --workers $Workers
        if ($LASTEXITCODE -ne 0) { throw "figures of $w failed" }
        $paths += "results/inspect"
    }
    Push-Results $paths "14-day official evaluation of ${w}"
}
