#!/usr/bin/env bash
# Oracle bound with the true simulator (server). Run after round v16.
#
# Usage (from the repository root, in the background):
#     nohup bash scripts/server_oracle.sh > logs/oracle.log 2>&1 &
#
# The best policy of rounds v14-v16 is run for 7 days on three scenarios
# (subarctic, hot-dry and mild climate), first alone and then inside the oracle
# (sbwm.oracle): every hour 31 candidate modifications of its commands are
# tried on forked copies of the simulator over the next 24 h, and the best is
# applied. The gap between the two scores bounds what better short-term
# decisions can still gain (about 2 h per scenario on 31 cores).
source "$(dirname "$0")/server_env.sh"
mkdir -p logs

SCEN=(building_train_1/fairbanks_1 building_train_1/phoenix_5 building_train_2/denver_8)
ALL=()
for w in policies/v14_*.pt policies/v15_*.pt policies/v16_*.pt; do
    case "$w" in *_fr*|*_h[0-9]*|*_mr*|*_ph*) continue ;; esac
    if [ -f "$w" ]; then ALL+=("$w"); fi
done
read -r -a BEST <<< "$(py -m sbwm.pick_best results/official30_14d.json 1 "${ALL[@]}")"
b="${BEST[0]}"
echo "oracle on $b"

# 1. the policy alone, 7 days
F=()
for s in "${SCEN[@]}"; do F+=("$s"); done
py -m sbwm.eval_policy --scenarios smart_buildings/training_scenarios --policies dpc --weights "$b" \
    --filter "${F[@]}" --steps 2016 --workers 3 --out results/oracle_base.json
push_results "Oracle reference: $b alone on 7 days" results/oracle_base.json

# 2. the oracle, one scenario at a time (results pushed after each)
for s in "${SCEN[@]}"; do
    py -m sbwm.oracle --weights "$b" --scenarios "$s" --steps 2016 --every 12 --horizon 288 \
        --candidates 31 --workers 31 --out results/oracle.json
    push_results "Oracle bound on $s with $b" results/oracle.json
done
echo "oracle finished"
