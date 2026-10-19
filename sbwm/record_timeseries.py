"""Record step-by-step trajectories of two controllers on one scenario.

Both runs use the scenario's official start date, weather and seeded
occupancy, so the only difference is the controller.

Usage:
    python -m sbwm.record_timeseries --scenario building_train_8/phoenix_4 \
        --weights policies/dpc_v3.pt --steps 864
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch


def record(scenario: Path, policy_name: str, weights: str, steps: int) -> list[dict]:
    from sbwm.collect import _load_env
    from sbwm.policy import DPCPolicy
    from sbwm.schema import BuildingSchema
    from smart_buildings.evaluate import POLICY_MAP
    from smart_buildings.smart_control.utils import conversion_utils

    start = re.search(r"^start_timestamp = '(.*)'$", (scenario / "config.gin").read_text(), re.M).group(1)
    env, meta = _load_env(scenario, start)
    s = BuildingSchema.from_meta(meta)
    if policy_name == "dpc":
        pol = DPCPolicy(meta, env.action_spec(), weights=weights)
    else:
        pol = POLICY_MAP[policy_name](meta, env.action_spec())
    cap = {}
    original = env.reward_function.compute_reward

    def hook(info):
        resp = original(info)
        cap["info"], cap["resp"] = info, resp
        return resp

    env.reward_function.compute_reward = hook
    obs, _ = env.reset()
    rows = []
    for _ in range(steps):
        a = pol.get_action(obs)
        obs, r, term, trunc, _ = env.step(a)
        info, resp = cap["info"], cap["resp"]
        zri = info.zone_reward_infos
        temp = np.array([zri[z].zone_air_temperature for z in s.zone_rooms]) - 273.15
        hsp = np.array([zri[z].heating_setpoint_temperature for z in s.zone_rooms]) - 273.15
        csp = np.array([zri[z].cooling_setpoint_temperature for z in s.zone_rooms]) - 273.15
        occ = np.array([zri[z].average_occupancy for z in s.zone_rooms])
        ahu = info.air_handler_reward_infos
        blr = info.boiler_reward_infos
        elec = sum(ahu[k].blower_electrical_energy_rate + abs(ahu[k].air_conditioning_electrical_energy_rate)
                   for k in ahu) + sum(blr[k].pump_electrical_energy_rate for k in blr)
        gas = sum(blr[k].natural_gas_heating_energy_rate for k in blr)
        t = conversion_utils.proto_to_pandas_timestamp(info.start_timestamp)
        rows.append({
            "time": str(t), "reward": r,
            "comfort": 0.6 * resp.normalized_productivity_regret,
            "energy": -0.2 * resp.normalized_energy_cost, "carbon": -0.2 * resp.normalized_carbon_emission,
            "temp_mean": temp.mean(), "temp_min": temp.min(), "temp_max": temp.max(),
            "heating_setpoint": float(np.median(hsp)), "cooling_setpoint": float(np.median(csp)),
            "occupants": occ.sum(), "outside_air": float(obs[s.ahu_obs_idx[0, -1]]) - 273.15,
            "electricity_kw": elec / 1000.0, "gas_kw": gas / 1000.0,
        })
        if term or trunc:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="building_train_8/phoenix_4")
    ap.add_argument("--weights", default="policies/dpc_v3.pt")
    ap.add_argument("--baseline", default="pid_water")
    ap.add_argument("--steps", type=int, default=864)
    ap.add_argument("--out", default="evaluations")
    args = ap.parse_args()
    torch.set_num_threads(1)
    scen = Path("smart_buildings/training_scenarios") / args.scenario
    for name in ("dpc", args.baseline):
        rows = record(scen, name, args.weights, args.steps)
        out = Path(args.out) / f"timeseries_{args.scenario.replace('/', '__')}_{name}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{name}: mean reward {np.mean([r['reward'] for r in rows]):.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
