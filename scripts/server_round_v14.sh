#!/usr/bin/env bash
# Round v14: world model trained to 48 h, policies with memory, holidays and
# more weekend start states on several world models, three seeds per recipe,
# frost guard at deployment (server).
#
# Usage (from the repository root, in the background):
#     nohup bash scripts/server_round_v14.sh > logs/round_v14.log 2>&1 &
#
# Steps (each skipped when its output exists, results pushed as they are ready):
#   1. frost guard on the two best policies so far, 14-day evaluation (quick,
#      gives the cost of the guard early); the guard acts outside the comfort
#      hours when the 25th percentile of the zone temperatures drops below
#      10 C (until back above 13 C), or, in a lighter version, below 5 C
#      (until back above 8 C);
#   2. world model v14 from v11 with unrolls of 4, 12, 24 and 48 h, diagnosis;
#   3. policies, in two batches of trainings in parallel:
#        v14_2wm_s{0,1,2}   recipe of v13_mem_2wm (world models v11 and v12,
#                           24 h horizon) with 30% of the start states on
#                           Fridays and Saturdays, three seeds;
#        v14_48_s{0,1,2}    world model v14, 48 h horizon with discount 0.999
#                           (0.995 halves the weight every 11.5 h), memory, holidays and
#                           weekend start states, three seeds;
#        v14_new2wm         world models v14 and v12, 24 h horizon, as v14_2wm;
#      each policy is evaluated on 14 days as soon as its batch ends;
#   4. both frost guards on the two best policies of the round, 14-day evaluation,
#      control figures.
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

# 1. frost guard on the best policies so far
for w in policies/v11_long_fit3.pt policies/v13_mem_2wm.pt; do
    FROST "$w" 10 13
    FROST "$w" 5 8
done

# 2. world model v14, trained up to 48 h
TRAIN=(month_data/episodes)
for d in pc_data/episodes month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then TRAIN+=("$d"); fi
done
ONP=(month_data/episodes_onpolicy_probe_v11)
if [ ! -f world_models/wm_v14.pt ]; then
    py -m sbwm.train_wm --train "${TRAIN[@]}" --emphasis "${ONP[@]}" --emphasis-share 0.3 \
        --test month_data/episodes_test --out wm_runs/v14 --init world_models/wm_v11.pt --mem \
        --steps 1600 --batch 16 --lr 5e-5 --lengths 48 144 288 576 --w-comfort 5 --pool 40 --refresh 200 \
        --threads "$THREADS" --eval-every 400
    cp wm_runs/v14/world_model_best.pt world_models/wm_v14.pt
    cp wm_runs/v14/eval_log.json world_models/wm_v14_eval_log.json
    py -m sbwm.diagnose_wm --wm world_models/wm_v11.pt world_models/wm_v14.pt --policy policies/transformer_v6.pt \
        --episodes diag_clean/official diag_clean/bench2 --threads "$THREADS" --out results/wm_diagnosis_v14.json
    push_results "World model v14 (48 h unrolls) and diagnosis" \
        world_models/wm_v14.pt world_models/wm_v14_eval_log.json results/wm_diagnosis_v14.json
fi

# 3. policies
DPC_TRAIN=(month_data/episodes)
for d in pc_data/episodes pc_data/episodes_onpolicy pc_data/episodes_onpolicy_v4 month_data/episodes_onpolicy_*; do
    if [ -d "$d" ]; then DPC_TRAIN+=("$d"); fi
done
LONG="--lengths 144 288 --grad-decay 0.99 --action-filter 0.3"
LONG48="--lengths 288 576 --grad-decay 0.995 --gamma 0.999 --action-filter 0.3"
MEM="--policy-mem --holiday --weekend-boost 0.3"
declare -A VARIANTS=(
    [2wm_s0]="--wm world_models/wm_v11.pt world_models/wm_v12.pt $LONG $MEM --seed 0"
    [2wm_s1]="--wm world_models/wm_v11.pt world_models/wm_v12.pt $LONG $MEM --seed 1"
    [2wm_s2]="--wm world_models/wm_v11.pt world_models/wm_v12.pt $LONG $MEM --seed 2"
    [48_s0]="--wm world_models/wm_v14.pt $LONG48 $MEM --seed 0"
    [48_s1]="--wm world_models/wm_v14.pt $LONG48 $MEM --seed 1"
    [48_s2]="--wm world_models/wm_v14.pt $LONG48 $MEM --seed 2"
    [new2wm]="--wm world_models/wm_v14.pt world_models/wm_v12.pt $LONG $MEM --seed 0"
)
BATCH1=(48_s0 2wm_s0 48_s1 2wm_s1)
BATCH2=(48_s2 2wm_s2 new2wm)
PAR_THREADS=$(( THREADS / 2 ))

run_batch() {
    local pids=() paths=() name w
    for name in "$@"; do
        if [ -f "policies/v14_$name.pt" ]; then continue; fi
        # shellcheck disable=SC2086
        py -m sbwm.train_dpc ${VARIANTS[$name]} --train "${DPC_TRAIN[@]}" --test month_data/episodes_test \
            --exclude-buildings month_data/buildings_list.csv --out wm_runs/v14_$name --iters 400 --comfort-scale 1.8 \
            --init policies/transformer_v6.pt --eval-every 200 --threads "$PAR_THREADS" --pool 60 --refresh 100 \
            > "logs/v14_$name.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "WARNING: a training of $* failed"; done
    for name in "$@"; do
        w=policies/v14_$name.pt
        if [ ! -f "$w" ] && [ -f "wm_runs/v14_$name/policy_it00400.pt" ]; then
            cp "wm_runs/v14_$name/policy_it00400.pt" "$w"
            cp "wm_runs/v14_$name/dpc_log.json" "results/train_log_v14_$name.json"
            paths+=("$w" "results/train_log_v14_$name.json")
        fi
    done
    if [ ${#paths[@]} -gt 0 ]; then push_results "Policies of round v14 ($*)" "${paths[@]}"; fi
    for name in "$@"; do
        w=policies/v14_$name.pt
        if [ -f "$w" ]; then
            EVAL "$w"
            push_results "14-day official evaluation of $w" results/official30_14d.json
        fi
    done
}
run_batch "${BATCH1[@]}"
run_batch "${BATCH2[@]}"

# 4. frost guard on the two best policies of the round, control figures
ALL=()
for name in "${BATCH1[@]}" "${BATCH2[@]}"; do
    if [ -f "policies/v14_$name.pt" ]; then ALL+=("policies/v14_$name.pt"); fi
done
read -r -a BEST <<< "$(py -m sbwm.pick_best results/official30_14d.json 2 "${ALL[@]}")"
for w in "${BEST[@]}"; do FROST "$w" 10 13; FROST "$w" 5 8; done
REFS=(pid_water policies/v11_long_fit3.pt policies/v13_mem_2wm.pt)
FR=()
for w in "${BEST[@]}"; do FR+=("${w%.pt}_fr10.pt"); done
py -m sbwm.inspect_control --policies "${REFS[@]}" "${BEST[@]}" "${FR[@]}" --steps 2016 --workers "$WORKERS"
push_results "Control figures of round v14" results/inspect
echo "round v14 finished, best: ${BEST[*]}"
