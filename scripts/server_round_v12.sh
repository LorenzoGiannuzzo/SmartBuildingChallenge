#!/usr/bin/env bash
# Round v12: probe collection, world model v12, policies with memory and holiday
# handling, 14-day evaluations (server).
#
# Usage (from the repository root, in the background):
#     COLLECT_POLICY=policies/v11_long_fit3.pt WM_INIT=world_models/wm_v11.pt \
#         nohup bash scripts/server_round_v12.sh > logs/round_v12.log 2>&1 &
#
# Steps (each skipped when its output exists, results pushed as they are ready):
#   1. one 14-day episode per training building, driven by COLLECT_POLICY with
#      plant probes in a quarter of the segments (water, supply-air temperature
#      and static pressure held at random levels);
#   2. world model v12 from WM_INIT, half of the samples from these episodes,
#      then diagnosed against WM_INIT on the clean episodes;
#   3. three policies on v12, all with the recipe of v11_long_fit3 (24 h
#      horizon, gradient decay 0.99, command filter 0.3 inside training),
#      trained in parallel:
#        v12_long_fit3      from transformer_v6, no memory (control);
#        v12_mem_long_fit3  from transformer_v6, policy memory and holidays;
#        v12_mem_ft         from v11_long_fit3, policy memory and holidays;
#   4. 14-day evaluation on the 30 official scenarios, control figures.
source "$(dirname "$0")/server_env.sh"
mkdir -p logs
COLLECT_POLICY="${COLLECT_POLICY:-policies/v11_long_fit3.pt}"
WM_INIT="${WM_INIT:-world_models/wm_v11.pt}"
[ -f "$WM_INIT" ] || WM_INIT=world_models/wm_v10.pt
echo "collection policy $COLLECT_POLICY, world model from $WM_INIT"

# 1. probe collection
ONP=month_data/episodes_onpolicy_probe_v11
py -m sbwm.collect --scenarios month_data/scenarios --out "$ONP" --monthly --one-station \
    --steps 4032 --dpc-weights "$COLLECT_POLICY" --probe --workers "$WORKERS" --seed 51

# 2. world model v12
TRAIN=(month_data/episodes)
for d in pc_data/episodes month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then TRAIN+=("$d"); fi
done
if [ ! -f world_models/wm_v12.pt ]; then
    py -m sbwm.train_wm --train "${TRAIN[@]}" --emphasis "$ONP" --emphasis-share 0.5 \
        --test month_data/episodes_test --out wm_runs/v12 --init "$WM_INIT" --mem \
        --steps 1000 --batch 16 --lr 5e-5 --lengths 48 144 288 --w-comfort 5 --pool 40 --refresh 200 \
        --threads "$THREADS" --eval-every 250
    cp wm_runs/v12/world_model_best.pt world_models/wm_v12.pt
    cp wm_runs/v12/eval_log.json world_models/wm_v12_eval_log.json
    py -m sbwm.diagnose_wm --wm "$WM_INIT" world_models/wm_v12.pt --policy policies/transformer_v6.pt \
        --episodes diag_clean/official diag_clean/bench2 --threads "$THREADS" --out results/wm_diagnosis_v12.json
    push_results "World model v12 (probe round) and diagnosis" \
        world_models/wm_v12.pt world_models/wm_v12_eval_log.json results/wm_diagnosis_v12.json
fi

# 3. policies on v12, three trainings in parallel
DPC_TRAIN=(month_data/episodes)
for d in pc_data/episodes pc_data/episodes_onpolicy pc_data/episodes_onpolicy_v4 month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then DPC_TRAIN+=("$d"); fi
done
LONG="--lengths 144 288 --grad-decay 0.99 --action-filter 0.3"
declare -A VARIANTS=(
    [long_fit3]="--init policies/transformer_v6.pt $LONG"
    [mem_long_fit3]="--init policies/transformer_v6.pt $LONG --policy-mem --holiday"
    [mem_ft]="--init policies/v11_long_fit3.pt $LONG --policy-mem --holiday"
)
NAMES_V12=(long_fit3 mem_long_fit3 mem_ft)
PAR_THREADS=$(( THREADS * 2 / 3 ))
pids=()
for name in "${NAMES_V12[@]}"; do
    if [ -f "policies/v12_$name.pt" ]; then continue; fi
    # shellcheck disable=SC2086
    py -m sbwm.train_dpc --wm world_models/wm_v12.pt --train "${DPC_TRAIN[@]}" --test month_data/episodes_test \
        --exclude-buildings month_data/buildings_list.csv --out wm_runs/v12_$name --iters 400 --comfort-scale 1.8 \
        --eval-every 200 --threads "$PAR_THREADS" --pool 60 --refresh 100 \
        ${VARIANTS[$name]} > "logs/v12_$name.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
paths=()
for name in "${NAMES_V12[@]}"; do
    w=policies/v12_$name.pt
    if [ ! -f "$w" ]; then
        cp wm_runs/v12_$name/policy_it00400.pt "$w"
        cp wm_runs/v12_$name/dpc_log.json results/train_log_v12_$name.json
        paths+=("$w" results/train_log_v12_$name.json)
    fi
done
if [ ${#paths[@]} -gt 0 ]; then push_results "Policies on world model v12" "${paths[@]}"; fi

# 4. 14-day evaluations and control figures
REFS=(pid_water policies/abl_A10.pt policies/v10_base_f03.pt policies/v11_long_fit3.pt)
for w in policies/v12_long_fit3.pt policies/v12_mem_long_fit3.pt policies/v12_mem_ft.pt; do
    py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies pid_water dpc \
        --weights "$w" --steps 4032 --workers "$WORKERS" --out results/official30_14d.json
    paths=(results/official30_14d.json)
    if [[ "$w" == *v12_* ]]; then
        py -m sbwm.inspect_control --policies "${REFS[@]}" "$w" --steps 2016 --workers "$WORKERS"
        paths+=(results/inspect)
    fi
    push_results "14-day official evaluation of $w" "${paths[@]}"
done
echo "round v12 finished"
