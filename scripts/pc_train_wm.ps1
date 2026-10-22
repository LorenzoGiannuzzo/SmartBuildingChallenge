# Train world model v6 on the monthly episodes and push it to the repository.
#
# Usage (from starter-kit, after the collection has finished):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_train_wm.ps1
#
# Steps:
#   1. two episodes per climate are moved to month_data/episodes_test and held out;
#   2. world model v6 is fine-tuned from v5 on all monthly episodes plus the
#      earlier PC data, streaming 40 episodes per source in memory;
#   3. the model is copied to world_models/wm_v6.pt, committed and pushed.
param([int]$Steps = 6000, [int]$Threads = 8)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path

#Lorenzo Giannuzzo: held-out test episodes, chosen deterministically (the first
#two files of each climate in name order) and only once.
$test = "month_data/episodes_test"
if (-not (Test-Path $test)) {
    New-Item -ItemType Directory -Path $test | Out-Null
    foreach ($st in "atlanta", "denver", "fairbanks", "minneapolis", "phoenix", "singapore") {
        Get-ChildItem "month_data/episodes" -Filter "*__${st}_1__*.npz" | Sort-Object Name |
            Select-Object -First 2 | Move-Item -Destination $test
    }
}

$train = @("month_data/episodes")
if (Test-Path "pc_data/episodes") { $train += "pc_data/episodes" }
$emph = @()
foreach ($d in "pc_data/episodes_onpolicy", "pc_data/episodes_onpolicy_v4") {
    if (Test-Path $d) { $emph += $d }
}
$extra = @()
if ($emph.Count -gt 0) { $extra = @("--emphasis") + $emph + @("--emphasis-share", "0.15") }

& $py -m sbwm.train_wm --train @train @extra --test $test --out wm_runs/v6 `
    --init world_models/wm_v5.pt --steps $Steps --lr 1e-4 --lengths 12 24 48 `
    --w-comfort 5 --pool 40 --refresh 200 --threads $Threads --eval-every 500
if ($LASTEXITCODE -ne 0) { throw "world model training failed" }

Copy-Item wm_runs/v6/world_model.pt world_models/wm_v6.pt -Force
Copy-Item wm_runs/v6/eval_log.json world_models/wm_v6_eval_log.json -Force
git add world_models/wm_v6.pt world_models/wm_v6_eval_log.json
git commit -m "Add world model v6 trained on the monthly episodes"
git pull --rebase github main
git push github main
