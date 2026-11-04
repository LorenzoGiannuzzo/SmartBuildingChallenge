#!/usr/bin/env bash
# Round v15: physics-based zone features (sbwm.physics) on the two recipes of
# round v14, three seeds each (server). Run after round v14.
#
# Usage (from the repository root, in the background):
#     nohup bash scripts/server_round_v15.sh > logs/round_v15.log 2>&1 &
#
# Steps (each skipped when its output exists, results pushed as they are ready):
#   1. policies with the physics features, in two batches of three trainings:
#        v15_2wm_phys_s{0,1,2}  recipe of v14_2wm (world models v11 and v12,
#                               24 h horizon, memory, holidays, weekend starts);
#        v15_48_phys_s{0,1,2}   recipe of v14_48 (world model v14, 48 h horizon,
#                               discount 0.999);
#      each policy is evaluated on 14 days as soon as its batch ends;
#   2. ranking of rounds v14 and v15 together; the two best get the light frost
#      guard (5-8 C, kept in every policy from now on, so that the building
#      never drops below about 5 C), 14-day evaluation and control figures;
#   3. slower decisions at deployment on the best policy: commands held for 10
#      and 15 min (sbwm.set_hold), 14-day evaluation.
source "$(dirname "$0")/server_env.sh"
mkdir -p logs

EVAL() {
    py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies dpc \
        --weights "$1" --steps 4032 --workers "$WORKERS" --out results/official30_14d.json
}
FROST() {
    # FROST policy.pt on off -> policy_fr<on>.pt, evaluated
    local w="${1%.pt}_fr$2.pt"
    if [ -f "$w" ]; then return; fi
    py -m sbwm.set_frost "$1" "$w" "$2" "$3"
    EVAL "$w"
    push_results "14-day official evaluation of $w (frost guard $2-$3 C)" "$w" results/official30_14d.json
}

DPC_TRAIN=(month_data/episodes)
for d in pc_data/episodes pc_data/episodes_onpolicy pc_data/episodes_onpolicy_v4 month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then DPC_TRAIN+=("$d"); fi
done
LONG="--lengths 144 288 --grad-decay 0.99 --action-filter 0.3"
LONG48="--lengths 288 576 --grad-decay 0.995 --gamma 0.999 --action-filter 0.3"
MEM="--policy-mem --holiday --weekend-boost 0.3 --phys"
declare -A VARIANTS=(
    [2wm_phys_s0]="--wm world_models/wm_v11.pt world_models/wm_v12.pt $LONG $MEM --seed 0"
    [2wm_phys_s1]="--wm world_models/wm_v11.pt world_models/wm_v12.pt $LONG $MEM --seed 1"
    [2wm_phys_s2]="--wm world_models/wm_v11.pt world_models/wm_v12.pt $LONG $MEM --seed 2"
    [48_phys_s0]="--wm world_models/wm_v14.pt $LONG48 $MEM --seed 0"
    [48_phys_s1]="--wm world_models/wm_v14.pt $LONG48 $MEM --seed 1"
    [48_phys_s2]="--wm world_models/wm_v14.pt $LONG48 $MEM --seed 2"
)
BATCH1=(48_phys_s0 2wm_phys_s0 2wm_phys_s1)
BATCH2=(48_phys_s1 48_phys_s2 2wm_phys_s2)
PAR_THREADS=$(( THREADS / 2 ))

run_batch() {
    local pids=() paths=() name w
    for name in "$@"; do
        if [ -f "policies/v15_$name.pt" ]; then continue; fi
        # shellcheck disable=SC2086
        py -m sbwm.train_dpc ${VARIANTS[$name]} --train "${DPC_TRAIN[@]}" --test month_data/episodes_test \
            --exclude-buildings month_data/buildings_list.csv --out wm_runs/v15_$name --iters 400 --comfort-scale 1.8 \
            --init policies/transformer_v6.pt --eval-every 200 --threads "$PAR_THREADS" --pool 60 --refresh 100 \
            > "logs/v15_$name.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "WARNING: a training of $* failed"; done
    for name in "$@"; do
        w=policies/v15_$name.pt
        if [ ! -f "$w" ] && [ -f "wm_runs/v15_$name/policy_it00400.pt" ]; then
            cp "wm_runs/v15_$name/policy_it00400.pt" "$w"
            cp "wm_runs/v15_$name/dpc_log.json" "results/train_log_v15_$name.json"
            paths+=("$w" "results/train_log_v15_$name.json")
        fi
    done
    if [ ${#paths[@]} -gt 0 ]; then push_results "Policies of round v15 ($*)" "${paths[@]}"; fi
    for name in "$@"; do
        w=policies/v15_$name.pt
        if [ -f "$w" ]; then
            EVAL "$w"
            push_results "14-day official evaluation of $w" results/official30_14d.json
        fi
    done
}
if [ ! -f world_models/wm_v14.pt ]; then
    echo "world model v14 missing: run round v14 first"
    exit 1
fi
run_batch "${BATCH1[@]}"
run_batch "${BATCH2[@]}"

# 2. ranking of rounds v14 and v15, light frost guard and control figures of the two best
ALL=()
for w in policies/v14_*.pt policies/v15_*.pt; do
    case "$w" in *_fr*) continue ;; esac
    if [ -f "$w" ]; then ALL+=("$w"); fi
done
read -r -a BEST <<< "$(py -m sbwm.pick_best results/official30_14d.json 2 "${ALL[@]}")"
FR=()
for w in "${BEST[@]}"; do FROST "$w" 5 8; FR+=("${w%.pt}_fr5.pt"); done
REFS=(pid_water policies/v11_long_fit3.pt policies/v13_mem_2wm.pt)
py -m sbwm.inspect_control --policies "${REFS[@]}" "${BEST[@]}" "${FR[@]}" --steps 2016 --workers "$WORKERS"
push_results "Control figures of round v15" results/inspect

# 3. commands held for 10 and 15 min on the best policy
for k in 2 3; do
    w="${BEST[0]%.pt}_h$k.pt"
    if [ -f "$w" ]; then continue; fi
    py -m sbwm.set_hold "${BEST[0]}" "$w" "$k"
    EVAL "$w"
    push_results "14-day official evaluation of $w (commands held for $((5 * k)) min)" "$w" results/official30_14d.json
done
echo "round v15 finished, best of v14 and v15: ${BEST[*]}"
