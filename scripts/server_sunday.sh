#!/usr/bin/env bash
# Sunday run (server), after round wm15; the server is free from Saturday 22:00.
#
# Usage (from the repository root, in the background; it waits for round wm15
# and for Saturday 22:00 by itself when started with the command in the docs):
#     nohup bash scripts/server_sunday.sh > logs/sunday.log 2>&1 &
#
# Steps (each skipped when its output exists, results pushed as they are ready):
#   A. second batch of policies on world model v15 (48 h seed 1, 24 h seed 2)
#      in parallel with a second world model v15 (seed 1, bootstrap resample of
#      the episodes), for the planner ensemble; 14-day evaluation;
#   B. planner tests with the structured world model: naive (v15) and safe
#      (v15, v15b and v14 must agree, online correction, safety gate) on
#      Fairbanks 1, Denver 8 and Phoenix 5, 7 days;
#   C. 30-day evaluation (as on Kaggle) of v11_long_fit3, v14_new2wm and the
#      best policy on world model v15;
#   D. oracle bound on Atlanta 2 and Singapore 8 with the best policy, to cover
#      the five climates;
#   E. control figures of the best policy.
source "$(dirname "$0")/server_env.sh"
mkdir -p logs results/wm_planner

EVAL() {
    py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies dpc \
        --weights "$1" --steps 4032 --workers "$WORKERS" --out results/official30_14d.json
}
DPC_TRAIN=(month_data/episodes)
for d in pc_data/episodes pc_data/episodes_onpolicy pc_data/episodes_onpolicy_v4 month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then DPC_TRAIN+=("$d"); fi
done
TRAIN=(month_data/episodes)
for d in pc_data/episodes month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then TRAIN+=("$d"); fi
done
LONG="--lengths 144 288 --grad-decay 0.99 --action-filter 0.3"
LONG48="--lengths 288 576 --grad-decay 0.995 --gamma 0.999 --action-filter 0.3"
MEM="--policy-mem --holiday --weekend-boost 0.3"
declare -A VARIANTS=(
    [48_s1]="--wm world_models/wm_v15.pt $LONG48 $MEM --seed 1"
    [24_s2]="--wm world_models/wm_v15.pt world_models/wm_v14.pt $LONG $MEM --seed 2"
)
PAR_THREADS=$(( THREADS / 2 ))
if [ ! -f world_models/wm_v15.pt ]; then
    echo "world model v15 missing: round wm15 did not finish"
    exit 1
fi

# A. policies and second world model in parallel
pids=()
for name in 48_s1 24_s2; do
    if [ -f "policies/v15w_$name.pt" ]; then continue; fi
    # shellcheck disable=SC2086
    py -m sbwm.train_dpc ${VARIANTS[$name]} --train "${DPC_TRAIN[@]}" --test month_data/episodes_test \
        --exclude-buildings month_data/buildings_list.csv --out wm_runs/v15w_$name --iters 400 --comfort-scale 1.8 \
        --init policies/transformer_v6.pt --eval-every 200 --threads "$PAR_THREADS" --pool 60 --refresh 100 \
        > "logs/v15w_$name.log" 2>&1 &
    pids+=($!)
done
if [ ! -f world_models/wm_v15b.pt ]; then
    py -m sbwm.train_wm --train "${TRAIN[@]}" --emphasis month_data/episodes_onpolicy_v14c month_data/episodes_onpolicy_v14cold \
        --emphasis-share 0.5 --test month_data/episodes_test --out wm_runs/v15b --init world_models/wm_v14.pt --mem \
        --structured --w-hourly 10 --steps 1600 --batch 16 --lr 5e-5 --lengths 48 144 288 576 --w-comfort 5 \
        --pool 40 --refresh 200 --threads "$PAR_THREADS" --eval-every 400 --seed 1 --bootstrap > logs/wm_v15b.log 2>&1 &
    pids+=($!)
fi
for p in "${pids[@]}"; do wait "$p" || echo "WARNING: a job of step A failed"; done
paths=()
for name in 48_s1 24_s2; do
    w=policies/v15w_$name.pt
    if [ ! -f "$w" ] && [ -f "wm_runs/v15w_$name/policy_it00400.pt" ]; then
        cp "wm_runs/v15w_$name/policy_it00400.pt" "$w"
        cp "wm_runs/v15w_$name/dpc_log.json" "results/train_log_v15w_$name.json"
        paths+=("$w" "results/train_log_v15w_$name.json")
    fi
done
if [ ! -f world_models/wm_v15b.pt ] && [ -f wm_runs/v15b/world_model_best.pt ]; then
    cp wm_runs/v15b/world_model_best.pt world_models/wm_v15b.pt
    paths+=(world_models/wm_v15b.pt)
fi
if [ ${#paths[@]} -gt 0 ]; then push_results "Sunday: second batch on world model v15 and world model v15b" "${paths[@]}"; fi
for name in 48_s1 24_s2; do
    w=policies/v15w_$name.pt
    if [ -f "$w" ]; then
        EVAL "$w"
        push_results "14-day official evaluation of $w" results/official30_14d.json
    fi
done

ALL=(policies/v14_new2wm.pt)
for w in policies/v15w_*.pt; do
    case "$w" in *_fr*|*_h[0-9]*|*_mr*|*_ph*) continue ;; esac
    if [ -f "$w" ]; then ALL+=("$w"); fi
done
read -r -a BEST <<< "$(py -m sbwm.pick_best results/official30_14d.json 1 "${ALL[@]}")"
b="${BEST[0]}"
echo "best policy: $b"

# B. planner tests with the structured world model
SAFE_WM=(world_models/wm_v15.pt world_models/wm_v14.pt)
if [ -f world_models/wm_v15b.pt ]; then SAFE_WM=(world_models/wm_v15.pt world_models/wm_v15b.pt world_models/wm_v14.pt); fi
pids=()
for s in building_train_1/fairbanks_1 building_train_2/denver_8 building_train_1/phoenix_5; do
    n=$(basename "$s")
    [ -f "results/wm_planner/v15naive__$n.json" ] || {
        py -m sbwm.wm_planner --weights "$b" --wm world_models/wm_v15.pt --scenarios "$s" --steps 2016 \
            --horizon 288 --candidates 31 --threads 4 --out "results/wm_planner/v15naive__$n.json" \
            > "logs/planner_v15naive_$n.log" 2>&1 &
        pids+=($!); }
    [ -f "results/wm_planner/v15safe__$n.json" ] || {
        py -m sbwm.wm_planner --weights "$b" --wm "${SAFE_WM[@]}" --scenarios "$s" --steps 2016 --horizon 144 \
            --candidates 31 --threads 4 --agree --correct --gate 1.0 --out "results/wm_planner/v15safe__$n.json" \
            > "logs/planner_v15safe_$n.log" 2>&1 &
        pids+=($!); }
done
for p in "${pids[@]}"; do wait "$p" || echo "WARNING: a planner run failed"; done
push_results "Sunday: planner tests with world model v15" results/wm_planner

# C. 30-day evaluation, as on Kaggle
for w in policies/v11_long_fit3.pt policies/v14_new2wm.pt "$b"; do
    py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies dpc \
        --weights "$w" --steps 8640 --workers "$WORKERS" --out results/official30_30d.json
    push_results "30-day official evaluation of $w" results/official30_30d.json
done

# D. oracle on the two remaining climates
py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies dpc --weights "$b" \
    --filter building_train_1/atlanta_2 building_train_1/singapore_8 --steps 2016 --workers 2 \
    --out results/oracle_base.json
for s in building_train_1/atlanta_2 building_train_1/singapore_8; do
    py -m sbwm.oracle --weights "$b" --scenarios "$s" --steps 2016 --every 12 --horizon 288 \
        --candidates 31 --workers 31 --out results/oracle.json
    push_results "Oracle bound on $s with $b" results/oracle.json results/oracle_base.json
done

# E. control figures
py -m sbwm.inspect_control --policies pid_water policies/v14_new2wm.pt "$b" --steps 2016 --workers "$WORKERS"
push_results "Sunday: control figures" results/inspect
echo "sunday run finished"
