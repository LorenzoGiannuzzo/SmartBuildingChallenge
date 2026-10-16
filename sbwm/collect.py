"""Collect simulator transitions for world-model training.

Each episode runs one scenario from a random start date with an exploration
policy that switches between realistic control (``pid_water``), noisy control,
random held setpoints and extreme plant states. Per step it stores the policy
observation, the action, and the physical quantities needed to rebuild the
competition reward exactly (zone temperatures, setpoints, occupancy, electric
and gas power, time-varying cost and carbon coefficients).

Usage (from the repository root):
    python -m sbwm.collect --scenarios wm_data/scenarios --out wm_data/episodes \
        --episodes 60 --steps 2016 --workers 2 --seed 0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODES = ("pid", "pid_noise", "random_hold", "off", "max_heat", "max_cool")
MODE_WEIGHTS = np.array([0.30, 0.30, 0.25, 0.05, 0.05, 0.05])


def _load_env(scenario_dir: Path, start: str):
    """Load a scenario with its start timestamp replaced."""
    from smart_buildings import evaluate as ev

    text = (scenario_dir / "config.gin").read_text()
    text = re.sub(r"^start_timestamp = '.*'$", f"start_timestamp = '{start}'", text, flags=re.M)
    with tempfile.NamedTemporaryFile("w", suffix=".gin", delete=False) as fh:
        fh.write(text)
        tmp = Path(fh.name)
    try:
        env = ev.load_hybrid_environment(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    meta = ev.create_building_meta_data(env, scenario_dir)
    return env, meta


class ExplorationPolicy:
    """Switches between behaviour modes on random segments of 0.5 to 6 hours."""

    def __init__(self, meta, action_spec, schema, rng: np.random.Generator):
        from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy

        self.pid = PIDWaterResetPolicy(meta, action_spec)
        self.s = schema
        self.rng = rng
        self.mode = "pid"
        self.left = 0
        self.hold_c = None
        self.hold_d = None

    def _new_segment(self):
        self.mode = MODES[self.rng.choice(len(MODES), p=MODE_WEIGHTS)]
        self.left = int(self.rng.integers(6, 73))
        s = self.s
        if self.mode == "random_hold":
            self.hold_c = self.rng.uniform(-1, 1, s.n_continuous).astype(np.float32)
            #Lorenzo Giannuzzo: AHUs and pump are switched on with high
            #probability so that the held setpoints actually act on the plant.
            self.hold_d = (self.rng.random(s.n_discrete) < 0.75).astype(np.int32)

    def _extreme(self, heat: bool):
        s = self.s
        c = np.full(s.n_continuous, -1.0, np.float32)
        d = np.ones(s.n_discrete, np.int32)
        c[s.ahu_temp_act] = 1.0 if heat else -1.0
        c[s.ahu_pressure_act] = self.rng.uniform(0.0, 1.0)
        c[s.ahu_recirc_act] = 1.0 if heat else self.rng.uniform(-1.0, 1.0)
        valid = s.zone_damper_act >= 0
        c[s.zone_damper_act[valid]] = self.rng.uniform(0.0, 1.0, valid.sum())
        valid = s.zone_reheat_act >= 0
        c[s.zone_reheat_act[valid]] = 1.0 if heat else -1.0
        if s.hws_temp_act >= 0:
            c[s.hws_temp_act] = 1.0 if heat else -1.0
        if s.hws_pressure_act >= 0:
            c[s.hws_pressure_act] = 1.0 if heat else -1.0
        if not heat:
            d[s.hws_run_act] = 0
        return c, d

    def __call__(self, obs):
        #Lorenzo Giannuzzo: the PID is stepped every time to keep its internal
        #integrators consistent, even when its action is not used.
        base = self.pid.get_action(obs)
        if self.left <= 0:
            self._new_segment()
            self.ext = self._extreme(self.mode == "max_heat") if self.mode.startswith("max") else None
        self.left -= 1
        c = np.asarray(base["continuous_action"], np.float32).copy()
        d = np.asarray(base["discrete_action"], np.int32).copy()
        if self.mode == "pid_noise":
            c = np.clip(c + self.rng.normal(0, 0.35, c.shape), -1, 1).astype(np.float32)
            flip = self.rng.random(d.shape) < 0.05
            d = np.where(flip, 1 - d, d).astype(np.int32)
        elif self.mode == "random_hold":
            jitter = self.rng.normal(0, 0.05, c.shape)
            c = np.clip(self.hold_c + jitter, -1, 1).astype(np.float32)
            d = self.hold_d.copy()
        elif self.mode == "off":
            c = np.full_like(c, -1.0)
            d = np.zeros_like(d)
        elif self.mode in ("max_heat", "max_cool"):
            c, d = self.ext[0].copy(), self.ext[1].copy()
        return c, d, MODES.index(self.mode)


def run_episode(job: dict) -> str:
    from smart_buildings.smart_control.utils import conversion_utils
    from sbwm.schema import BuildingSchema

    out = Path(job["out"])
    if out.exists():
        return f"skip {out.name}"
    rng = np.random.default_rng(job["seed"])
    env, meta = _load_env(Path(job["scenario"]), job["start"])
    schema = BuildingSchema.from_meta(meta)
    policy = ExplorationPolicy(meta, env.action_spec(), schema, rng)

    #Lorenzo Giannuzzo: the reward function is wrapped so that the exact
    #RewardInfo used by the environment is captured without calling the
    #simulator a second time (occupancy is stochastic).
    captured = {}
    reward_fn = env.reward_function
    original = reward_fn.compute_reward

    def hook(reward_info):
        response = original(reward_info)
        captured["info"] = reward_info
        captured["resp"] = response
        return response

    reward_fn.compute_reward = hook
    elec_model = reward_fn._electricity_energy_cost
    gas_model = reward_fn._natural_gas_energy_cost

    T, V = job["steps"], schema.num_zones
    obs, _ = env.reset()
    rec = {k: [] for k in (
        "act_c", "act_d", "mode", "zone_temp", "zone_hsp", "zone_csp", "zone_flow",
        "zone_flow_sp", "zone_occ", "blower_w", "ac_w", "gas_w", "pump_w",
        "elec_cost_w", "gas_cost_w", "elec_co2_w", "gas_co2_w", "reward",
        "regret", "norm_energy", "norm_carbon", "t_start")}
    obs_list = [np.asarray(obs, np.float32)]
    t0 = time.time()
    for _ in range(T):
        c, d, mode = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(
            {"continuous_action": c, "discrete_action": d})
        info, resp = captured["info"], captured["resp"]
        zri = info.zone_reward_infos
        rec["zone_temp"].append([zri[r].zone_air_temperature for r in schema.zone_rooms])
        rec["zone_hsp"].append([zri[r].heating_setpoint_temperature for r in schema.zone_rooms])
        rec["zone_csp"].append([zri[r].cooling_setpoint_temperature for r in schema.zone_rooms])
        rec["zone_flow"].append([zri[r].air_flow_rate for r in schema.zone_rooms])
        rec["zone_flow_sp"].append([zri[r].air_flow_rate_setpoint for r in schema.zone_rooms])
        rec["zone_occ"].append([zri[r].average_occupancy for r in schema.zone_rooms])
        ahu = info.air_handler_reward_infos
        rec["blower_w"].append(sum(ahu[k].blower_electrical_energy_rate for k in ahu))
        rec["ac_w"].append(sum(ahu[k].air_conditioning_electrical_energy_rate for k in ahu))
        blr = info.boiler_reward_infos
        rec["gas_w"].append(sum(blr[k].natural_gas_heating_energy_rate for k in blr))
        rec["pump_w"].append(sum(blr[k].pump_electrical_energy_rate for k in blr))
        start = conversion_utils.proto_to_pandas_timestamp(info.start_timestamp)
        end = conversion_utils.proto_to_pandas_timestamp(info.end_timestamp)
        #Lorenzo Giannuzzo: cost and carbon are linear in the energy rate, so
        #the per-watt coefficients of this step rebuild them exactly.
        rec["elec_cost_w"].append(elec_model.cost(start_time=start, end_time=end, energy_rate=1.0))
        rec["gas_cost_w"].append(gas_model.cost(start_time=start, end_time=end, energy_rate=1.0))
        rec["elec_co2_w"].append(elec_model.carbon(start_time=start, end_time=end, energy_rate=1.0))
        rec["gas_co2_w"].append(gas_model.carbon(start_time=start, end_time=end, energy_rate=1.0))
        rec["reward"].append(reward)
        rec["regret"].append(resp.normalized_productivity_regret)
        rec["norm_energy"].append(resp.normalized_energy_cost)
        rec["norm_carbon"].append(resp.normalized_carbon_emission)
        rec["t_start"].append(int(start.value // 10**9))
        rec["act_c"].append(c)
        rec["act_d"].append(d)
        rec["mode"].append(mode)
        obs_list.append(np.asarray(obs, np.float32))
        if terminated or truncated:
            break

    arrays = {k: np.asarray(v) for k, v in rec.items()}
    for k, v in arrays.items():
        if v.dtype == np.float64:
            arrays[k] = v.astype(np.float32)
    arrays["obs"] = np.stack(obs_list)
    #Lorenzo Giannuzzo: geometry features are stored with the episode so that
    #files collected on another machine do not need the scenario folders.
    from sbwm.geometry import zone_static_features
    arrays["zone_static"] = zone_static_features(meta, schema)
    header = {
        "scenario": job["scenario"],
        "start": job["start"],
        "seconds": time.time() - t0,
        "schema": schema.to_dict(),
        "meta": {
            k: meta[k] for k in (
                "observation_field_names", "morning_start_hour", "evening_start_hour",
                "comfort_temp_window", "eco_temp_window", "time_zone",
                "observation_normalization_metadata", "num_floors")
        },
        "reward": reward_fn.json_metadata,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, header=np.array(json.dumps(header, default=str)), **arrays)
    os.replace(tmp, out)
    return f"done {out.name} steps={len(rec['reward'])} {time.time() - t0:.0f}s"


def build_jobs(scen_root: Path, out_root: Path, n: int, steps: int, seed: int,
               stations=None) -> list[dict]:
    rng = np.random.default_rng(seed)
    scen = sorted(p for p in scen_root.glob("*/*") if (p / "config.gin").exists())
    if stations:
        scen = [p for p in scen if p.name.split("_")[0] in stations]
    days = steps * 300 / 86400.0
    jobs = []
    for i in range(n):
        sc = scen[rng.integers(len(scen))]
        #Lorenzo Giannuzzo: random start within 2023, leaving room for the
        #episode and for the simulator's internal lookahead.
        first = pd.Timestamp("2023-01-02")
        last = pd.Timestamp("2023-12-31") - pd.Timedelta(days=days + 2)
        day = first + pd.Timedelta(days=int(rng.integers(0, (last - first).days)))
        start = f"{day:%Y-%m-%d} 06:00"
        name = f"{sc.parent.name}__{sc.name}__{day:%m%d}__s{seed}_{i:04d}.npz"
        jobs.append({"scenario": str(sc), "start": start, "steps": steps,
                     "seed": int(seed * 100000 + i), "out": str(out_root / name)})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2016)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stations", nargs="*")
    args = ap.parse_args()
    jobs = build_jobs(Path(args.scenarios), Path(args.out), args.episodes,
                      args.steps, args.seed, args.stations)
    if args.workers == 1:
        for job in jobs:
            print(run_episode(job), flush=True)
    else:
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            for msg in pool.imap_unordered(run_episode, jobs):
                print(msg, flush=True)


if __name__ == "__main__":
    main()
