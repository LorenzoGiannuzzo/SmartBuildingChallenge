# Runs the long jobs on the PC, one after the other. Every step can be
# interrupted and resumed: rerunning this script skips what is already done.
#
# Start it from the starter-kit folder with:
#   powershell -ExecutionPolicy Bypass -File scripts\pc_pipeline.ps1
# Add -Test to also run the 60-building test at the end.

param([switch]$Test, [string]$Policy = "policies/dpc_v4.pt")

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe")).Path
$workers = 10

# 1. Training buildings: 100 new buildings, every third with the default footprint.
& $py -m sbwm.make_buildings --start-seed 1000 --valid 100 --default-every 3 --root month_data

# 2. Monthly hybrid collection: one 30-day episode per building and station,
#    one station at a time (about 5 hours each).
foreach ($st in "atlanta", "denver", "fairbanks", "minneapolis", "phoenix", "singapore") {
    & $py -m sbwm.collect --scenarios month_data/scenarios --out month_data/episodes --monthly `
        --steps 8640 --hybrid --dpc-weights policies/dpc_v4.pt --workers $workers --seed 7 --stations $st
}

# 3. Optional test on 60 unseen buildings (30-day episodes). Baseline runs are
#    kept in the results file and reused for every later policy.
if ($Test) {
    & $py -m sbwm.make_validation --count 60
    $name = [System.IO.Path]::GetFileNameWithoutExtension($Policy)
    & $py -m sbwm.eval_policy --scenarios val_data/scenarios --policies dpc pid_water fixed_setpoint `
        --weights $Policy --steps 8640 --workers $workers --out results/test60.json
    git add results/test60.json
    git commit -m "Test on 60 unseen buildings ($name)"
    git pull --rebase github main
    git push github main
}
Write-Host "Pipeline finished."
