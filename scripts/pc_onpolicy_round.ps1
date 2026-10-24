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
param([string]$Policy = "policies/hybrid_v1.pt", [string]$Tag = "h1", [int]$Seed = 11,
      [string]$Init = "world_models/wm_v6.pt", [string]$New = "v7",
      [int]$Workers = 10, [int]$Threads = 8)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path
$onp = "month_data/episodes_onpolicy_$Tag"

& $py -m sbwm.collect --scenarios month_data/scenarios --out $onp --monthly --one-station `
    --steps 4032 --dpc-weights $Policy --workers $Workers --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "collection failed" }

$train = @("month_data/episodes")
if (Test-Path "pc_data/episodes") { $train += "pc_data/episodes" }
#Lorenzo Giannuzzo: the earlier on-policy rounds stay in the training mix.
$older = Get-ChildItem month_data -Directory -Filter "episodes_onpolicy_*" | Where-Object { $_.Name -ne "episodes_onpolicy_$Tag" } |
    ForEach-Object { "month_data/" + $_.Name }
if ($older) { $train += @($older) }
& $py -m sbwm.train_wm --train @train --emphasis $onp --emphasis-share 0.5 `
    --test month_data/episodes_test --out wm_runs/$New --init $Init `
    --steps 2000 --lr 5e-5 --lengths 24 48 --w-comfort 5 --pool 40 --refresh 200 `
    --threads $Threads --eval-every 250
if ($LASTEXITCODE -ne 0) { throw "world model training failed" }
Copy-Item wm_runs/$New/world_model_best.pt world_models/wm_$New.pt -Force
Copy-Item wm_runs/$New/eval_log.json world_models/wm_${New}_eval_log.json -Force

#Lorenzo Giannuzzo: closed-loop check on two new episodes per climate, for the
#previous and the updated world model.
$diag = "month_data/diag_$Tag"
if (-not (Test-Path $diag)) {
    New-Item -ItemType Directory -Path $diag | Out-Null
    foreach ($st in "atlanta", "denver", "fairbanks", "minneapolis", "phoenix", "singapore") {
        Get-ChildItem $onp -Filter "*__${st}_1__*.npz" | Sort-Object Name |
            Select-Object -First 2 | Copy-Item -Destination $diag
    }
}
$report = "world_models/closed_loop_$Tag.txt"
"previous world model ($Init)" | Out-File $report -Encoding utf8
& $py -m sbwm.closed_loop_check --wm $Init --policy $Policy --episodes $diag |
    Out-File $report -Append -Encoding utf8
"world model $New" | Out-File $report -Append -Encoding utf8
& $py -m sbwm.closed_loop_check --wm world_models/wm_$New.pt --policy $Policy --episodes $diag |
    Out-File $report -Append -Encoding utf8

git add world_models/wm_$New.pt world_models/wm_${New}_eval_log.json $report
git commit -m "Add world model $New (on-policy round $Tag) and closed-loop check"
git pull --rebase github main
git push github main
