# On-policy round: collect data with the hybrid policy, update the world model,
# check it in closed loop, and push the results.
#
# Usage (from starter-kit):
#     powershell -ExecutionPolicy Bypass -File scripts\pc_onpolicy_round.ps1
#
# Steps:
#   1. one 14-day episode per training building (random climate), driven by
#      hybrid_v1 with exploration segments (about 7-8 hours with 10 workers);
#   2. world model v7: v6 fine-tuned with half of the samples from these
#      on-policy episodes (about 1.5 hours);
#   3. closed-loop check of v6 and v7 on 12 of the new episodes;
#   4. world model v7 and the check are committed and pushed.
param([string]$Policy = "policies/hybrid_v1.pt", [int]$Workers = 10, [int]$Threads = 8)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path
$onp = "month_data/episodes_onpolicy_h1"

& $py -m sbwm.collect --scenarios month_data/scenarios --out $onp --monthly --one-station `
    --steps 4032 --dpc-weights $Policy --workers $Workers --seed 11
if ($LASTEXITCODE -ne 0) { throw "collection failed" }

$train = @("month_data/episodes")
if (Test-Path "pc_data/episodes") { $train += "pc_data/episodes" }
& $py -m sbwm.train_wm --train @train --emphasis $onp --emphasis-share 0.5 `
    --test month_data/episodes_test --out wm_runs/v7 --init world_models/wm_v6.pt `
    --steps 2000 --lr 5e-5 --lengths 24 48 --w-comfort 5 --pool 40 --refresh 200 `
    --threads $Threads --eval-every 250
if ($LASTEXITCODE -ne 0) { throw "world model training failed" }
Copy-Item wm_runs/v7/world_model_best.pt world_models/wm_v7.pt -Force
Copy-Item wm_runs/v7/eval_log.json world_models/wm_v7_eval_log.json -Force

#Lorenzo Giannuzzo: closed-loop check on two new episodes per climate, for the
#previous and the updated world model.
$diag = "month_data/diag_h1"
if (-not (Test-Path $diag)) {
    New-Item -ItemType Directory -Path $diag | Out-Null
    foreach ($st in "atlanta", "denver", "fairbanks", "minneapolis", "phoenix", "singapore") {
        Get-ChildItem $onp -Filter "*__${st}_1__*.npz" | Sort-Object Name |
            Select-Object -First 2 | Copy-Item -Destination $diag
    }
}
$report = "world_models/closed_loop_h1.txt"
"world model v6" | Out-File $report -Encoding utf8
& $py -m sbwm.closed_loop_check --wm world_models/wm_v6.pt --policy $Policy --episodes $diag |
    Out-File $report -Append -Encoding utf8
"world model v7" | Out-File $report -Append -Encoding utf8
& $py -m sbwm.closed_loop_check --wm world_models/wm_v7.pt --policy $Policy --episodes $diag |
    Out-File $report -Append -Encoding utf8

git add world_models/wm_v7.pt world_models/wm_v7_eval_log.json $report
git commit -m "Add world model v7 (on-policy round with hybrid_v1) and closed-loop check"
git pull --rebase github main
git push github main
