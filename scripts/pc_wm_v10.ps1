# World model v10: long memory per zone and long unroll curriculum.
#
# Usage (from starter-kit, after pc_clean_collect.ps1):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_wm_v10.ps1
#
# Steps:
#   1. v10 is fine-tuned from v9 with the long memory (exponential averages of
#      zone temperature, reheat and damper commands and outdoor air, from 1 to
#      24 h) and unrolls of 4, 12 and 24 hours, half of the samples from the
#      latest on-policy round (about 6 hours);
#   2. v9 and v10 are diagnosed on the clean transformer_v6 episodes
#      (replay and closed loop, errors up to 24 h);
#   3. v10 and the diagnosis are committed and pushed.
param([int]$Threads = 8)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path

$train = @("month_data/episodes")
if (Test-Path "pc_data/episodes") { $train += "pc_data/episodes" }
$train += Get-ChildItem month_data -Directory -Filter "episodes_onpolicy_*" | ForEach-Object { "month_data/" + $_.Name }
if (-not (Test-Path "world_models/wm_v10.pt")) {
    & $py -m sbwm.train_wm --train @train --emphasis month_data/episodes_onpolicy_a6 --emphasis-share 0.5 `
        --test month_data/episodes_test --out wm_runs/v10 --init world_models/wm_v9.pt --mem `
        --steps 1500 --batch 16 --lr 1e-4 --lengths 48 144 288 --w-comfort 5 --pool 40 --refresh 200 `
        --threads $Threads --eval-every 250
    if ($LASTEXITCODE -ne 0) { throw "world model training failed" }
    Copy-Item wm_runs/v10/world_model_best.pt world_models/wm_v10.pt -Force
    Copy-Item wm_runs/v10/eval_log.json world_models/wm_v10_eval_log.json -Force
}

& $py -m sbwm.diagnose_wm --wm world_models/wm_v9.pt world_models/wm_v10.pt --policy policies/transformer_v6.pt `
    --episodes diag_clean/official diag_clean/bench2 --threads $Threads --out results/wm_diagnosis_v9_v10.json
if ($LASTEXITCODE -ne 0) { throw "diagnosis failed" }

git add world_models/wm_v10.pt world_models/wm_v10_eval_log.json results/wm_diagnosis_v9_v10.json
git commit -m "World model v10 (long memory, 24 h unrolls) and diagnosis against v9"
git pull --rebase github main
git push github main
