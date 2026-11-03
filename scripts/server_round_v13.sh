#!/usr/bin/env bash
# Round v13: policies with memory on world model v11, 48 h horizon, seed
# variance of the v11_long_fit3 recipe, supply-water floor (server).
#
# Usage (from the repository root, in the background):
#     nohup bash scripts/server_round_v13.sh > logs/round_v13.log 2>&1 &
#
# Steps (each skipped when its output exists, results pushed as they are ready):
#   1. supply-water floor on v11_long_fit3 at deployment (0.0 and -0.5),
#      14-day evaluation;
#   2. four policies trained in parallel, all with the command filter 0.3
#      inside training and from transformer_v6:
#        v13_long_s1       recipe of v11_long_fit3 with another seed (how much
#                          of a difference between two runs is noise);
#        v13_mem_long      v11 world model, memory and holidays, 24 h horizon;
#        v13_mem_2wm       the same alternating world models v11 and v12;
#        v13_mem_long48    v11 world model, memory and holidays, 48 h horizon;
#   3. 14-day evaluation on the 30 official scenarios, control figures.
source "$(dirname "$0")/server_env.sh"
mkdir -p logs

# 1. supply-water floor
for f in 0.0 -0.5; do
    tag=w${f/./}
    w=policies/v11_long_fit3_${tag/-/m}.pt
    if [ -f "$w" ]; then continue; fi
    py -m sbwm.set_floor policies/v11_long_fit3.pt "$w" "$f"
    py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies dpc \
        --weights "$w" --steps 4032 --workers "$WORKERS" --out results/official30_14d.json
    push_results "14-day official evaluation of $w (supply-water floor $f)" "$w" results/official30_14d.json
done

# 2. policies, four trainings in parallel
DPC_TRAIN=(month_data/episodes)
for d in pc_data/episodes pc_data/episodes_onpolicy pc_data/episodes_onpolicy_v4 month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then DPC_TRAIN+=("$d"); fi
done
LONG="--lengths 144 288 --grad-decay 0.99 --action-filter 0.3"
MEM="--policy-mem --holiday"
declare -A VARIANTS=(
    [long_s1]="--wm world_models/wm_v11.pt $LONG --seed 1"
    [mem_long]="--wm world_models/wm_v11.pt $LONG $MEM"
    [mem_2wm]="--wm world_models/wm_v11.pt world_models/wm_v12.pt $LONG $MEM"
    [mem_long48]="--wm world_models/wm_v11.pt --lengths 288 576 --grad-decay 0.995 --action-filter 0.3 $MEM"
)
NAMES_V13=(long_s1 mem_long mem_2wm mem_long48)
PAR_THREADS=$(( THREADS / 2 ))
pids=()
for name in "${NAMES_V13[@]}"; do
    if [ -f "policies/v13_$name.pt" ]; then continue; fi
    # shellcheck disable=SC2086
    py -m sbwm.train_dpc ${VARIANTS[$name]} --train "${DPC_TRAIN[@]}" --test month_data/episodes_test \
        --exclude-buildings month_data/buildings_list.csv --out wm_runs/v13_$name --iters 400 --comfort-scale 1.8 \
        --init policies/transformer_v6.pt --eval-every 200 --threads "$PAR_THREADS" --pool 60 --refresh 100 \
        > "logs/v13_$name.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
paths=()
for name in "${NAMES_V13[@]}"; do
    w=policies/v13_$name.pt
    if [ ! -f "$w" ]; then
        cp wm_runs/v13_$name/policy_it00400.pt "$w"
        cp wm_runs/v13_$name/dpc_log.json results/train_log_v13_$name.json
        paths+=("$w" results/train_log_v13_$name.json)
    fi
done
if [ ${#paths[@]} -gt 0 ]; then push_results "Policies of round v13" "${paths[@]}"; fi

# 3. 14-day evaluations and control figures
REFS=(pid_water policies/v10_base_f03.pt policies/v11_long_fit3.pt)
for name in "${NAMES_V13[@]}"; do
    w=policies/v13_$name.pt
    py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies dpc \
        --weights "$w" --steps 4032 --workers "$WORKERS" --out results/official30_14d.json
    py -m sbwm.inspect_control --policies "${REFS[@]}" "$w" --steps 2016 --workers "$WORKERS"
    push_results "14-day official evaluation of $w" results/official30_14d.json results/inspect
done
echo "round v13 finished"
