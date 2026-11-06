"""Planning at deployment with the world model (first test, privileged exogenous data).

Same receding-horizon search as sbwm.oracle, but the candidates are scored by
the world model instead of the true simulator: at every decision hour, N
candidate modifications of the policy commands for the coming hour are rolled
out in one batch through the world model over the horizon, the unmodified
policy continuing afterwards, and the best candidate is applied to the real
building. The pure policy is always a candidate.

This first test gives the planner the exogenous future of the scenario
(weather, setpoints, occupancy, prices) from a run of the policy alone, so it
measures how much of the oracle gap survives the world-model errors; the
deployable version will replace them with forecasts.

Usage:
    python -m sbwm.wm_planner --weights policies/v14_new2wm.pt --wm world_models/wm_v14.pt \
        --scenarios building_train_2/denver_8 --steps 2016 --out results/wm_planner.json
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from sbwm.oracle import REHEAT, SUPPLY, WATER, PLANT, candidate_set, modify


class Recorder:
    """Records a run in the format of sbwm.collect, so that it loads as an Episode."""

    KEYS = ("act_c", "act_d", "zone_temp", "zone_hsp", "zone_csp", "zone_flow", "zone_flow_sp", "zone_occ",
            "blower_w", "ac_w", "gas_w", "pump_w", "elec_cost_w", "gas_cost_w", "elec_co2_w", "gas_co2_w",
            "reward", "regret", "norm_energy", "norm_carbon", "t_start", "mode")

    def __init__(self, env, schema):
        self.schema = schema
        self.rec = {k: [] for k in self.KEYS}
        self.obs = []
        rf = env.reward_function
        self.elec, self.gas = rf._electricity_energy_cost, rf._natural_gas_energy_cost
        self.reward_meta = rf.json_metadata

    def add_obs(self, obs):
        self.obs.append(np.asarray(obs, np.float32))

    def add_step(self, c, d, info, resp, reward):
        from smart_buildings.smart_control.utils import conversion_utils

        s, r = self.schema, self.rec
        zri = info.zone_reward_infos
        r["zone_temp"].append([zri[z].zone_air_temperature for z in s.zone_rooms])
        r["zone_hsp"].append([zri[z].heating_setpoint_temperature for z in s.zone_rooms])
        r["zone_csp"].append([zri[z].cooling_setpoint_temperature for z in s.zone_rooms])
        r["zone_flow"].append([zri[z].air_flow_rate for z in s.zone_rooms])
        r["zone_flow_sp"].append([zri[z].air_flow_rate_setpoint for z in s.zone_rooms])
        r["zone_occ"].append([zri[z].average_occupancy for z in s.zone_rooms])
        ahu = info.air_handler_reward_infos
        r["blower_w"].append(sum(ahu[k].blower_electrical_energy_rate for k in ahu))
        r["ac_w"].append(sum(ahu[k].air_conditioning_electrical_energy_rate for k in ahu))
        blr = info.boiler_reward_infos
        r["gas_w"].append(sum(blr[k].natural_gas_heating_energy_rate for k in blr))
        r["pump_w"].append(sum(blr[k].pump_electrical_energy_rate for k in blr))
        start = conversion_utils.proto_to_pandas_timestamp(info.start_timestamp)
        end = conversion_utils.proto_to_pandas_timestamp(info.end_timestamp)
        r["elec_cost_w"].append(self.elec.cost(start_time=start, end_time=end, energy_rate=1.0))
        r["gas_cost_w"].append(self.gas.cost(start_time=start, end_time=end, energy_rate=1.0))
        r["elec_co2_w"].append(self.elec.carbon(start_time=start, end_time=end, energy_rate=1.0))
        r["gas_co2_w"].append(self.gas.carbon(start_time=start, end_time=end, energy_rate=1.0))
        r["reward"].append(reward)
        r["regret"].append(resp.normalized_productivity_regret)
        r["norm_energy"].append(resp.normalized_energy_cost)
        r["norm_carbon"].append(resp.normalized_carbon_emission)
        r["t_start"].append(int(start.value // 10**9))
        r["act_c"].append(c)
        r["act_d"].append(d)
        r["mode"].append(0)

    def episode(self, scenario: Path, start: str, meta):
        from sbwm.geometry import zone_static_features
        from sbwm.wm_data import load_episode

        arrays = {k: np.asarray(v) for k, v in self.rec.items()}
        for k, v in arrays.items():
            if v.dtype == np.float64:
                arrays[k] = v.astype(np.float32)
        arrays["obs"] = np.stack(self.obs)
        arrays["zone_static"] = zone_static_features(meta, self.schema)
        header = {"scenario": str(scenario), "start": start, "seconds": 0.0, "schema": self.schema.to_dict(),
                  "meta": {k: meta[k] for k in (
                      "observation_field_names", "morning_start_hour", "evening_start_hour", "comfort_temp_window",
                      "eco_temp_window", "time_zone", "observation_normalization_metadata", "num_floors")},
                  "reward": self.reward_meta}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ep.npz"
            np.savez(p, header=np.array(json.dumps(header, default=str)), **arrays)
            return load_episode(p)


def _hooked(env):
    captured = {}
    rf = env.reward_function
    original = rf.compute_reward

    def hook(info):
        resp = original(info)
        captured["info"], captured["resp"] = info, resp
        return resp

    rf.compute_reward = hook
    return captured


def _act(pol, obs):
    a = pol.get_action(obs)
    return np.asarray(a["continuous_action"], np.float32), np.asarray(a["discrete_action"], np.int32)


def prerun(scenario: Path, start: str, weights: str, steps: int):
    """The policy alone: score and the logged episode (exogenous future for the planner)."""
    from sbwm.collect import _load_env
    from sbwm.policy import DPCPolicy
    from sbwm.schema import BuildingSchema

    env, meta = _load_env(scenario, start)
    pol = DPCPolicy(meta, env.action_spec(), weights=weights)
    s = BuildingSchema.from_meta(meta)
    cap = _hooked(env)
    rec = Recorder(env, s)
    obs, _ = env.reset()
    rec.add_obs(obs)
    for _ in range(steps):
        c, d = _act(pol, obs)
        obs, r, term, trunc, _ = env.step({"continuous_action": c, "discrete_action": d})
        rec.add_step(c, d, cap["info"], cap["resp"], r)
        rec.add_obs(obs)
        if term or trunc:
            break
    rec.meta = meta
    return rec.episode(scenario, start, meta), rec


def _modify_policy_space(zact, aact, hact, cands):
    """Candidate modifications (one per batch element) in the policy action space."""
    z, a, h = zact.clone(), aact.clone(), hact.clone()
    for i, (plant, water, supply, reheat) in enumerate(cands):
        if plant == "off":
            a[i, :, 0] = -1.0
            h[i, 0] = -1.0
        elif plant == "on":
            a[i, :, 0] = 1.0
            h[i, 0] = 1.0
        h[i, 1] = (h[i, 1] + water).clamp(-1, 1)
        a[i, :, 1] = (a[i, :, 1] + supply).clamp(-1, 1)
        z[i, :, 1] = (z[i, :, 1] + reheat).clamp(-1, 1)
    return z, a, h


@torch.no_grad()
def _apply_bias(state, new_temp, bias):
    """Adds the per-zone drift correction (K per step) to a predicted step."""
    new_temp = new_temp + bias
    state["temp_hist"] = torch.cat([new_temp.unsqueeze(1), state["temp_hist"][:, 1:]], 1)
    return new_temp


@torch.no_grad()
def score_candidates(net, models, norm, ep, t0: int, cands, every: int, horizon: int, biases=None) -> np.ndarray:
    """Reward summed over the horizon for each candidate (M, N), one row per world model."""
    from sbwm.train_dpc import policy_step
    from sbwm.train_wm import reward_from
    from sbwm.world_model import Rollout

    n = len(cands)
    h = min(horizon, ep.length - t0)
    out_all = []
    for mi, m in enumerate(models):
        ro = Rollout(m, norm, ep)
        tt = torch.full((n,), t0, dtype=torch.long)
        state = ro.init_state(tt)
        total = torch.zeros(n)
        for k in range(h):
            t = tt + k
            carry = {}
            zact, aact, hact, _ = policy_step(net, ro, state, ep, t, norm, hard=True, carry=carry)
            if k < every:
                zact, aact, hact = _modify_policy_space(zact, aact, hact, cands)
            state, out, new_temp = ro.step(state, t, zact, aact, hact)
            state.update(carry)
            if biases is not None:
                new_temp = _apply_bias(state, new_temp, biases[mi])
            total += reward_from(ep, t, new_temp, out["power"])
        out_all.append(total)
    return torch.stack(out_all).numpy()


@torch.no_grad()
def replay_error(models, norm, ep, t: int, steps: int, biases=None) -> list[torch.Tensor]:
    """Per-zone error (truth minus prediction, (V,) per model) of a replay of the
    last ``steps`` logged actions from t - steps, with the current corrections."""
    from sbwm.world_model import Rollout

    errs = []
    for mi, m in enumerate(models):
        ro = Rollout(m, norm, ep)
        tt = torch.tensor([t - steps])
        state = ro.init_state(tt)
        for k in range(steps):
            i = tt + k
            state, out, new_temp = ro.step(state, i, ep.zact[i], ep.aact[i], ep.hact[i])
            if biases is not None:
                new_temp = _apply_bias(state, new_temp, biases[mi])
        errs.append(ep.temp[t - 1] - new_temp[0])
    return errs


def choose(scores: np.ndarray, agree: bool) -> int:
    """Index of the candidate to apply; 0 is the policy.

    With ``agree`` a candidate replaces the policy only when every world model
    predicts it to be better than the policy, and among those the one with the
    largest mean gain is taken.
    """
    if not agree or scores.shape[0] == 1:
        return int(np.argmax(scores.mean(0)))
    adv = scores - scores[:, :1]
    ok = adv.min(0) > 0
    if not ok.any():
        return 0
    mean = np.where(ok, adv.mean(0), -np.inf)
    return int(np.argmax(mean))


def _fill_row(ep, t: int, obs_row: np.ndarray, schema):
    """Write the observation of decision t into the episode tensors."""
    v = schema.num_zones

    def take(idx):
        idx = np.asarray(idx)
        out = np.zeros(idx.shape, np.float32)
        ok = idx >= 0
        out[ok] = obs_row[idx[ok]]
        return torch.as_tensor(out)

    ep.zobs[t] = take(schema.zone_obs_idx)
    ep.aobs[t] = take(schema.ahu_obs_idx)
    ep.hobs[t] = take(schema.hws_obs_idx)


def _fill_step(ep, t: int, c: np.ndarray, d: np.ndarray, temp: list, schema):
    """Write the action and rewarded temperature of step t into the episode tensors."""
    s = schema
    ep.temp[t] = torch.as_tensor(np.asarray(temp, np.float32))
    ep.zact[t] = torch.as_tensor(np.stack([
        np.where(s.zone_damper_act >= 0, c[s.zone_damper_act], -1),
        np.where(s.zone_reheat_act >= 0, c[s.zone_reheat_act], -1)], -1).astype(np.float32))
    ep.aact[t] = torch.as_tensor(np.stack([
        d[s.ahu_run_act] * 2 - 1, c[s.ahu_temp_act], c[s.ahu_pressure_act], c[s.ahu_recirc_act]], -1).astype(np.float32))
    ep.hact[t] = torch.as_tensor(np.array([
        d[s.hws_run_act] * 2 - 1,
        c[s.hws_temp_act] if s.hws_temp_act >= 0 else -1.0,
        c[s.hws_pressure_act] if s.hws_pressure_act >= 0 else -1.0], np.float32))
    for key in ("_wm_mem", "_phys"):
        ep.__dict__.pop(key, None)


def run(scenario: Path, weights: str, wm_paths: list[str], steps: int, every: int, horizon: int,
        n_cand: int, seed: int, agree: bool = False, correct: bool = False, gate: float = 0.0) -> dict:
    from sbwm.collect import _load_env
    from sbwm.policy import DPCPolicy
    from sbwm.schema import BuildingSchema
    from sbwm.train_dpc import load_models

    torch.set_num_threads(max(1, torch.get_num_threads()))
    start = re.search(r"^start_timestamp = '(.*)'$", (scenario / "config.gin").read_text(), re.M).group(1)
    t0 = time.time()
    ep, rec0 = prerun(scenario, start, weights, steps)
    base = -float(np.mean(rec0.rec["reward"]))
    t_pre = time.time() - t0
    models, norm = load_models(wm_paths)

    env, meta = _load_env(scenario, start)
    pol = DPCPolicy(meta, env.action_spec(), weights=weights)
    net = pol.net
    ck = torch.load(weights, weights_only=False)
    if ck.get("filter_in_training"):
        net.action_filter = ck.get("action_filter")
    s = BuildingSchema.from_meta(meta)
    cap = _hooked(env)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    rewards, comps, chosen, plan_s = [], [], [], 0.0
    #Lorenzo Giannuzzo: online correction of each world model on this building:
    #every hour the last hour is replayed with the logged actions and the
    #per-zone error updates a drift added to every predicted step; its size
    #also drives the safety gate, which hands control back to the policy
    #when the corrected model still misses the last hour by more than ``gate``.
    v = s.num_zones
    biases = [torch.zeros(v) for _ in models] if correct else None
    mae = None
    gated = 0
    t = 0
    while t < steps:
        _fill_row(ep, t, np.asarray(obs, np.float32), s)
        cands = candidate_set(n_cand, rng)
        best = 0
        if t >= 2:
            tp = time.time()
            if (correct or gate > 0) and t >= every + 4:
                errs = replay_error(models, norm, ep, t, every, biases)
                if correct:
                    biases = [(b + 0.3 * e / every).clamp(-0.05, 0.05) for b, e in zip(biases, errs)]
                m_now = float(np.mean([e.abs().mean() for e in errs]))
                mae = m_now if mae is None else 0.8 * mae + 0.2 * m_now
            if gate > 0 and mae is not None and mae > gate:
                gated += 1
            else:
                scores = score_candidates(net, models, norm, ep, t, cands, every, horizon, biases)
                best = choose(scores, agree)
            plan_s += time.time() - tp
        chosen.append(cands[best])
        for _ in range(min(every, steps - t)):
            _fill_row(ep, t, np.asarray(obs, np.float32), s)
            c, d = _act(pol, obs)
            a = modify({"continuous_action": c, "discrete_action": d}, cands[best], s)
            obs, r, term, trunc, _ = env.step(a)
            info, resp = cap["info"], cap["resp"]
            _fill_step(ep, t, a["continuous_action"], a["discrete_action"],
                       [info.zone_reward_infos[z].zone_air_temperature for z in s.zone_rooms], s)
            rewards.append(float(r))
            comps.append((resp.normalized_productivity_regret, resp.normalized_energy_cost,
                          resp.normalized_carbon_emission))
            t += 1
            if term or trunc:
                t = steps
                break
        if len(chosen) % 24 == 0:
            print(f"{scenario.name} {t}/{steps} planner {-np.mean(rewards):.4f} (policy alone {base:.4f}) "
                  f"| planning {plan_s:.0f}s", flush=True)
    c = np.asarray(comps)
    counts = {}
    for k, name in enumerate(("plant", "water", "supply", "reheat")):
        vals = [str(x[k]) for x in chosen]
        counts[name] = {v: vals.count(v) for v in sorted(set(vals))}
    return {"scenario": scenario.as_posix(), "weights": weights, "wm": wm_paths, "steps": len(rewards),
            "every": every, "horizon": horizon, "candidates": n_cand, "policy_alone": base,
            "mean_reward": float(np.mean(rewards)), "comfort": float(0.6 * c[:, 0].mean()),
            "energy": float(-0.2 * c[:, 1].mean()), "carbon": float(-0.2 * c[:, 2].mean()),
            "chosen": counts, "planning_seconds": plan_s, "prerun_seconds": t_pre,
            "agree": agree, "correct": correct, "gate": gate, "gated_decisions": gated,
            "replay_mae_1h": mae,
            "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--wm", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--root", default="smart_buildings/training_scenarios")
    ap.add_argument("--steps", type=int, default=2016)
    ap.add_argument("--every", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=288)
    ap.add_argument("--candidates", type=int, default=31)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agree", action="store_true", help="change the policy only when all world models agree")
    ap.add_argument("--correct", action="store_true", help="online per-zone drift correction of the world models")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="hand control back to the policy above this 1-hour replay error [K] (0 = off)")
    ap.add_argument("--out", default="results/wm_planner.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    res = json.loads(out.read_text()) if out.exists() else []
    for sc in args.scenarios:
        r = run(Path(args.root) / sc, args.weights, args.wm, args.steps, args.every, args.horizon,
                args.candidates, args.seed, args.agree, args.correct, args.gate)
        print(json.dumps({k: r[k] for k in ("scenario", "policy_alone", "mean_reward", "comfort", "energy",
                                             "carbon", "planning_seconds", "seconds")}), flush=True)
        res.append(r)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
