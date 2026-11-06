#!/usr/bin/env bash
# Planning at deployment with the world model, first tests (server).
#
# Usage (from the repository root, in the background):
#     nohup bash scripts/server_planner.sh > logs/planner.log 2>&1 &
#
# The best policy (v14_new2wm) runs 7 days on the three scenarios of the oracle
# (Fairbanks 1, Denver 8, Phoenix 5), with two planners that, every hour, try
# 31 modifications of the policy commands on the world model and apply the
# best; both are given the true exogenous future (weather, setpoints, prices),
# to isolate the effect of the world-model errors:
#   naive  world model v14, 24 h horizon;
#   safe   world models v14, v12 and v11, 12 h horizon, a modification applied
#          only when all three models agree, online per-zone drift correction
#          and a safety gate (back to the policy above 1 K of 1-hour error).
# The six runs go in parallel (4 threads each, about 1-2 h); compare with
# results/oracle.json and results/oracle_base.json.
source "$(dirname "$0")/server_env.sh"
mkdir -p logs results/wm_planner

W=policies/v14_new2wm.pt
SCEN=(building_train_1/fairbanks_1 building_train_2/denver_8 building_train_1/phoenix_5)
pids=()
for s in "${SCEN[@]}"; do
    n=$(basename "$s")
    py -m sbwm.wm_planner --weights "$W" --wm world_models/wm_v14.pt --scenarios "$s" --steps 2016 \
        --horizon 288 --candidates 31 --threads 4 --out "results/wm_planner/naive__$n.json" \
        > "logs/planner_naive_$n.log" 2>&1 &
    pids+=($!)
    py -m sbwm.wm_planner --weights "$W" --wm world_models/wm_v14.pt world_models/wm_v12.pt world_models/wm_v11.pt \
        --scenarios "$s" --steps 2016 --horizon 144 --candidates 31 --threads 4 --agree --correct --gate 1.0 \
        --out "results/wm_planner/safe__$n.json" > "logs/planner_safe_$n.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "WARNING: a planner run failed"; done
push_results "Planner tests with the world model (naive and safe) on three scenarios" results/wm_planner
echo "planner tests finished"
