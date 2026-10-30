#!/usr/bin/env bash
# On-policy round with v10_base_f03, world model v11, new policies and 14-day
# evaluations, on the server (same steps as scripts/pc_round_v11.ps1).
#
# Usage (from the repository root, in the background):
#     nohup bash scripts/server_round_v11.sh > logs/round_v11.log 2>&1 &
#     tail -f logs/round_v11.log
#
# Every step is skipped when its output exists and results are pushed as soon
# as they are ready, so the script can be rerun after an interruption.
source "$(dirname "$0")/server_env.sh"
mkdir -p logs

# 1. on-policy collection (one 14-day episode per training building)
ONP=month_data/episodes_onpolicy_f03
py -m sbwm.collect --scenarios month_data/scenarios --out "$ONP" --monthly --one-station \
    --steps 4032 --dpc-weights policies/v10_base_f03.pt --workers "$WORKERS" --seed 41

# 2. world model v11 and its diagnosis against v10
TRAIN=(month_data/episodes)
for d in pc_data/episodes month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then TRAIN+=("$d"); fi
done
if [ ! -f world_models/wm_v11.pt ]; then
    py -m sbwm.train_wm --train "${TRAIN[@]}" --emphasis "$ONP" --emphasis-share 0.5 \
        --test month_data/episodes_test --out wm_runs/v11 --init world_models/wm_v10.pt --mem \
        --steps 1000 --batch 16 --lr 5e-5 --lengths 48 144 288 --w-comfort 5 --pool 40 --refresh 200 \
        --threads "$THREADS" --eval-every 250
    cp wm_runs/v11/world_model_best.pt world_models/wm_v11.pt
    cp wm_runs/v11/eval_log.json world_models/wm_v11_eval_log.json
    py -m sbwm.diagnose_wm --wm world_models/wm_v10.pt world_models/wm_v11.pt --policy policies/transformer_v6.pt \
        --episodes diag_clean/official diag_clean/bench2 --threads "$THREADS" --out results/wm_diagnosis_v10_v11.json
    push_results "World model v11 (on-policy round with v10_base_f03) and diagnosis against v10" \
        world_models/wm_v11.pt world_models/wm_v11_eval_log.json results/wm_diagnosis_v10_v11.json
fi

# 3. policies on v11
DPC_TRAIN=(month_data/episodes)
for d in pc_data/episodes pc_data/episodes_onpolicy pc_data/episodes_onpolicy_v4 month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then DPC_TRAIN+=("$d"); fi
done
declare -A VARIANTS=(
    [base]="--lengths 72 144 --grad-decay 0.95"
    [fit3]="--lengths 72 144 --grad-decay 0.95 --action-filter 0.3"
    [long_fit3]="--lengths 144 288 --grad-decay 0.99 --action-filter 0.3"
)
for name in base fit3 long_fit3; do
    w=policies/v11_$name.pt
    if [ -f "$w" ]; then continue; fi
    # shellcheck disable=SC2086
    py -m sbwm.train_dpc --wm world_models/wm_v11.pt --train "${DPC_TRAIN[@]}" --test month_data/episodes_test \
        --exclude-buildings month_data/buildings_list.csv --out wm_runs/v11_$name --iters 400 --comfort-scale 1.8 \
        --init policies/transformer_v6.pt --eval-every 200 --threads "$THREADS" --pool 60 --refresh 100 ${VARIANTS[$name]}
    cp wm_runs/v11_$name/policy_it00400.pt "$w"
    cp wm_runs/v11_$name/dpc_log.json results/train_log_v11_$name.json
    paths=("$w" results/train_log_v11_$name.json)
    if [ "$name" = base ]; then
        py -m sbwm.set_filter "$w" policies/v11_base_f03.pt 0.3
        paths+=(policies/v11_base_f03.pt)
    fi
    push_results "Policy v11_$name on world model v11" "${paths[@]}"
done

# 4. 14-day evaluations on the official scenarios, with control figures
REFS=(pid_water policies/transformer_v6.pt policies/abl_A10.pt)
for w in policies/abl_A10.pt policies/v10_base_f03.pt policies/v11_base_f03.pt policies/v11_fit3.pt policies/v11_long_fit3.pt; do
    py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies pid_water dpc \
        --weights "$w" --steps 4032 --workers "$WORKERS" --out results/official30_14d.json
    paths=(results/official30_14d.json)
    if [[ "$w" == *v11_* ]]; then
        py -m sbwm.inspect_control --policies "${REFS[@]}" "$w" --steps 2016 --workers "$WORKERS"
        paths+=(results/inspect)
    fi
    push_results "14-day official evaluation of $w" "${paths[@]}"
done
echo "round v11 finished"
