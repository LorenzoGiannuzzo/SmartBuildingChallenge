#!/usr/bin/env bash
# Round wm15: world model v15 with structured power, trained on new on-policy
# data with the power components of every device, then policies on it (server).
#
# Usage (from the repository root, in the background):
#     nohup bash scripts/server_round_wm15.sh > logs/round_wm15.log 2>&1 &
#
# Steps (each skipped when its output exists, results pushed as they are ready):
#   1. collection with the power components (sbwm.components):
#        on-policy v14_new2wm with probe segments, one station per building;
#        the same on every Fairbanks folder (more cold-climate data);
#        v14_new2wm alone on the 30 official scenarios, 7 days (diagnosis);
#   2. world model v15 from v14: structured power heads supervised by the
#      components, error on the hourly mean power, unrolls up to 48 h, half of
#      the batches from the new data;
#   3. diagnosis: official diagnosis (v14 and v15), error figures with the
#      hourly power profiles and the component ratios for v14 and v15;
#   4. policies on world model v15, two batches: recipe of v14_new2wm (v15 and
#      v14, 24 h) seeds 0-2, 48 h on v15 seeds 0-1; 14-day evaluation;
#   5. control figures of the best policy.
source "$(dirname "$0")/server_env.sh"
mkdir -p logs

W=policies/v14_new2wm.pt
A=month_data/episodes_onpolicy_v14c
B=month_data/episodes_onpolicy_v14cold
C=diag_clean/v14_comp

# 1. collection with components
py -m sbwm.collect --scenarios month_data/scenarios --out "$A" --monthly --one-station \
    --steps 4032 --dpc-weights "$W" --probe --workers "$WORKERS" --seed 71
py -m sbwm.collect --scenarios month_data/scenarios --out "$B" --monthly --stations fairbanks \
    --steps 4032 --dpc-weights "$W" --probe --workers "$WORKERS" --seed 72
py -m sbwm.collect --scenarios smart_buildings/training_scenarios --out "$C" --monthly \
    --steps 2016 --dpc-weights "$W" --clean --workers "$WORKERS" --seed 73
echo "collected: $(ls "$A" | wc -l) on-policy, $(ls "$B" | wc -l) cold, $(ls "$C" | wc -l) diagnosis"

# 2. world model v15
TRAIN=(month_data/episodes)
for d in pc_data/episodes month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then TRAIN+=("$d"); fi
done
if [ ! -f world_models/wm_v15.pt ]; then
    py -m sbwm.train_wm --train "${TRAIN[@]}" --emphasis "$A" "$B" --emphasis-share 0.5 \
        --test month_data/episodes_test --out wm_runs/v15 --init world_models/wm_v14.pt --mem --structured \
        --w-hourly 10 --steps 1600 --batch 16 --lr 5e-5 --lengths 48 144 288 576 --w-comfort 5 --pool 40 \
        --refresh 200 --threads "$THREADS" --eval-every 400
    cp wm_runs/v15/world_model_best.pt world_models/wm_v15.pt
    cp wm_runs/v15/eval_log.json world_models/wm_v15_eval_log.json
    push_results "World model v15 (structured power)" world_models/wm_v15.pt world_models/wm_v15_eval_log.json
fi

# 3. diagnosis
if [ ! -f results/wm_diagnosis_v15.json ]; then
    py -m sbwm.diagnose_wm --wm world_models/wm_v14.pt world_models/wm_v15.pt --policy policies/transformer_v6.pt \
        --episodes diag_clean/official diag_clean/bench2 --threads "$THREADS" --out results/wm_diagnosis_v15.json
    mkdir -p results/figures
    for m in v14 v15; do
        py -m sbwm.wm_error_plots --weights "$W" --wm "world_models/wm_$m.pt" \
            --scenarios building_train_1/fairbanks_1 building_train_2/denver_8 building_train_1/phoenix_5 \
            --steps 2016 --cache results/wm_error_episodes --out "results/figures/wm_error_$m.png"
    done
    push_results "Diagnosis of world model v15 against v14" results/wm_diagnosis_v15.json results/figures
fi

# 4. policies on world model v15
DPC_TRAIN=(month_data/episodes)
for d in pc_data/episodes pc_data/episodes_onpolicy pc_data/episodes_onpolicy_v4 month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then DPC_TRAIN+=("$d"); fi
done
LONG="--lengths 144 288 --grad-decay 0.99 --action-filter 0.3"
LONG48="--lengths 288 576 --grad-decay 0.995 --gamma 0.999 --action-filter 0.3"
MEM="--policy-mem --holiday --weekend-boost 0.3"
declare -A VARIANTS=(
    [24_s0]="--wm world_models/wm_v15.pt world_models/wm_v14.pt $LONG $MEM --seed 0"
    [24_s1]="--wm world_models/wm_v15.pt world_models/wm_v14.pt $LONG $MEM --seed 1"
    [24_s2]="--wm world_models/wm_v15.pt world_models/wm_v14.pt $LONG $MEM --seed 2"
    [48_s0]="--wm world_models/wm_v15.pt $LONG48 $MEM --seed 0"
    [48_s1]="--wm world_models/wm_v15.pt $LONG48 $MEM --seed 1"
)
BATCH1=(48_s0 24_s0 24_s1)
BATCH2=(48_s1 24_s2)
PAR_THREADS=$(( THREADS / 2 ))
EVAL() {
    py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies dpc \
        --weights "$1" --steps 4032 --workers "$WORKERS" --out results/official30_14d.json
}
run_batch() {
    local pids=() paths=() name w
    for name in "$@"; do
        if [ -f "policies/v15w_$name.pt" ]; then continue; fi
        # shellcheck disable=SC2086
        py -m sbwm.train_dpc ${VARIANTS[$name]} --train "${DPC_TRAIN[@]}" --test month_data/episodes_test \
            --exclude-buildings month_data/buildings_list.csv --out wm_runs/v15w_$name --iters 400 --comfort-scale 1.8 \
            --init policies/transformer_v6.pt --eval-every 200 --threads "$PAR_THREADS" --pool 60 --refresh 100 \
            > "logs/v15w_$name.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "WARNING: a training of $* failed"; done
    for name in "$@"; do
        w=policies/v15w_$name.pt
        if [ ! -f "$w" ] && [ -f "wm_runs/v15w_$name/policy_it00400.pt" ]; then
            cp "wm_runs/v15w_$name/policy_it00400.pt" "$w"
            cp "wm_runs/v15w_$name/dpc_log.json" "results/train_log_v15w_$name.json"
            paths+=("$w" "results/train_log_v15w_$name.json")
        fi
    done
    if [ ${#paths[@]} -gt 0 ]; then push_results "Policies on world model v15 ($*)" "${paths[@]}"; fi
    for name in "$@"; do
        w=policies/v15w_$name.pt
        if [ -f "$w" ]; then
            EVAL "$w"
            push_results "14-day official evaluation of $w" results/official30_14d.json
        fi
    done
}
run_batch "${BATCH1[@]}"
run_batch "${BATCH2[@]}"

# 5. control figures of the best policy
ALL=()
for name in "${BATCH1[@]}" "${BATCH2[@]}"; do
    if [ -f "policies/v15w_$name.pt" ]; then ALL+=("policies/v15w_$name.pt"); fi
done
read -r -a BEST <<< "$(py -m sbwm.pick_best results/official30_14d.json 1 "${ALL[@]}" policies/v14_new2wm.pt)"
py -m sbwm.inspect_control --policies pid_water policies/v14_new2wm.pt "${BEST[@]}" --steps 2016 --workers "$WORKERS"
push_results "Control figures of round wm15" results/inspect
echo "round wm15 finished, best: ${BEST[*]}"
