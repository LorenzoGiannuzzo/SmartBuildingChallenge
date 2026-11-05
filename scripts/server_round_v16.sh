#!/usr/bin/env bash
# Round v16: weight of the comfort term in the DPC objective (kappa) on the
# best recipe of round v14, and minimum on/off time of the plant (server).
# Run after round v15.
#
# Usage (from the repository root, in the background):
#     nohup bash scripts/server_round_v16.sh > logs/round_v16.log 2>&1 &
#
# All policies are evaluated with the official reward; kappa only changes the
# training objective. World model v14 overestimates discomfort (comfort ratio
# 1.22 on logged actions, 1.08 in closed loop), so kappa = 1.8 may now count
# the comfort twice.
#
# Steps (each skipped when its output exists, results pushed as they are ready):
#   1. recipe of v14_new2wm (world models v14 and v12, 24 h horizon, memory,
#      holidays, 30% weekend starts, as in round v14) with kappa 1.0 and 1.4, two seeds
#      each, and kappa 1.8 with a second seed; 14-day evaluation;
#   2. the best policy of rounds v14-v16 with a minimum on/off time of 30 min
#      for the boiler loop and the air handlers (anti-short-cycling), alone and
#      with the light frost guard, and with the plant commands (air handlers,
#      hot-water loop) updated every 15 or 30 min while the zones keep 5 min;
#      14-day evaluation and control figures.
source "$(dirname "$0")/server_env.sh"
mkdir -p logs

EVAL() {
    py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies dpc \
        --weights "$1" --steps 4032 --workers "$WORKERS" --out results/official30_14d.json
}

DPC_TRAIN=(month_data/episodes)
for d in pc_data/episodes pc_data/episodes_onpolicy pc_data/episodes_onpolicy_v4 month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then DPC_TRAIN+=("$d"); fi
done
REC="--wm world_models/wm_v14.pt world_models/wm_v12.pt --lengths 144 288 --grad-decay 0.99 --action-filter 0.3 --policy-mem --holiday --weekend-boost 0.3"
declare -A VARIANTS=(
    [k10_s0]="$REC --comfort-scale 1.0 --seed 0"
    [k10_s1]="$REC --comfort-scale 1.0 --seed 1"
    [k14_s0]="$REC --comfort-scale 1.4 --seed 0"
    [k14_s1]="$REC --comfort-scale 1.4 --seed 1"
    [k18_s1]="$REC --comfort-scale 1.8 --seed 1"
)
BATCH1=(k10_s0 k14_s0 k18_s1)
BATCH2=(k10_s1 k14_s1)
PAR_THREADS=$(( THREADS / 2 ))

run_batch() {
    local pids=() paths=() name w
    for name in "$@"; do
        if [ -f "policies/v16_$name.pt" ]; then continue; fi
        # shellcheck disable=SC2086
        py -m sbwm.train_dpc ${VARIANTS[$name]} --train "${DPC_TRAIN[@]}" --test month_data/episodes_test \
            --exclude-buildings month_data/buildings_list.csv --out wm_runs/v16_$name --iters 400 \
            --init policies/transformer_v6.pt --eval-every 200 --threads "$PAR_THREADS" --pool 60 --refresh 100 \
            > "logs/v16_$name.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "WARNING: a training of $* failed"; done
    for name in "$@"; do
        w=policies/v16_$name.pt
        if [ ! -f "$w" ] && [ -f "wm_runs/v16_$name/policy_it00400.pt" ]; then
            cp "wm_runs/v16_$name/policy_it00400.pt" "$w"
            cp "wm_runs/v16_$name/dpc_log.json" "results/train_log_v16_$name.json"
            paths+=("$w" "results/train_log_v16_$name.json")
        fi
    done
    if [ ${#paths[@]} -gt 0 ]; then push_results "Policies of round v16 ($*)" "${paths[@]}"; fi
    for name in "$@"; do
        w=policies/v16_$name.pt
        if [ -f "$w" ]; then
            EVAL "$w"
            push_results "14-day official evaluation of $w" results/official30_14d.json
        fi
    done
}
run_batch "${BATCH1[@]}"
run_batch "${BATCH2[@]}"

# 2. minimum on/off time on the best policy of rounds v14-v16
ALL=()
for w in policies/v14_*.pt policies/v15_*.pt policies/v16_*.pt; do
    case "$w" in *_fr*|*_h[0-9]*|*_mr*|*_ph*) continue ;; esac
    if [ -f "$w" ]; then ALL+=("$w"); fi
done
read -r -a BEST <<< "$(py -m sbwm.pick_best results/official30_14d.json 1 "${ALL[@]}")"
b="${BEST[0]}"
OUT=()
for v in "mr6:min_run=6" "mr6_fr5:min_run=6 frost_guard=5,8" "ph3:plant_hold=3" "ph6:plant_hold=6"; do
    tag="${v%%:*}"; opts="${v#*:}"
    w="${b%.pt}_$tag.pt"
    OUT+=("$w")
    if [ -f "$w" ]; then continue; fi
    # shellcheck disable=SC2086
    py -m sbwm.set_option "$b" "$w" $opts
    EVAL "$w"
    push_results "14-day official evaluation of $w ($opts)" "$w" results/official30_14d.json
done
REFS=(pid_water policies/v11_long_fit3.pt policies/v14_new2wm.pt)
py -m sbwm.inspect_control --policies "${REFS[@]}" "$b" "${OUT[@]}" --steps 2016 --workers "$WORKERS"
push_results "Control figures of round v16" results/inspect
echo "round v16 finished, best: $b"
