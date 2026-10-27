"""World-model diagnosis on clean on-policy episodes.

The episodes are driven by one policy alone (``sbwm.collect --clean``), so two
comparisons are fair:

* replay: the world model is rolled out from real states with the logged
  actions, which are the policy's own; this is the pure model error on the
  states the policy visits;
* closed loop: the policy acts inside the world model from the same states;
  this adds the feedback of the policy on the model errors, which is what DPC
  optimizes through.

For each, the zone-temperature error is reported against the horizon (1, 6,
12 and 24 hours), together with the ratio of predicted to true comfort and
energy terms of the reward, broken down by climate, by occupancy and by the
outdoor temperature at the start.

Usage:
    python -m sbwm.diagnose_wm --wm world_models/wm_v8.pt world_models/wm_v9.pt \
        --policy policies/transformer_v6.pt --episodes diag_clean/official diag_clean/bench2 \
        --out results/wm_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sbwm.policy import PI_WARMUP, PolicyNet
from sbwm.train_dpc import load_models, policy_step
from sbwm.train_wm import comfort_term, reward_from
from sbwm.wm_data import AHU_OAT, HIST, load_episode
from sbwm.world_model import Rollout

HORIZONS = {"1h": 12, "6h": 72, "12h": 144, "24h": 288}


@torch.no_grad()
def rollout(model, norm, ep, t0, horizon, policy=None):
    """Predicted temperatures (H, B, V), comfort and energy terms (H, B)."""
    ro = Rollout(model, norm, ep)
    st = ro.init_state(t0)
    temps, comfort, energy = [], [], []
    for i in range(horizon):
        t = t0 + i
        if policy is None:
            z, a, h = ep.zact[t], ep.aact[t], ep.hact[t]
            pi_int = None
        else:
            z, a, h, pi_int = policy_step(policy, ro, st, ep, t, norm)
        st, out, nt = ro.step(st, t, z, a, h)
        if pi_int is not None:
            st["pi_int"] = pi_int
        temps.append(nt)
        c = comfort_term(ep, t, nt)
        comfort.append(c)
        energy.append(reward_from(ep, t, nt, out["power"]) - c)
    return torch.stack(temps), torch.stack(comfort), torch.stack(energy)


def truth(ep, t0, horizon):
    idx = t0.unsqueeze(0) + torch.arange(horizon).unsqueeze(1)  # (H, B)
    temps = ep.temp[idx]
    comfort = torch.stack([comfort_term(ep, idx[i], temps[i]) for i in range(horizon)])
    energy = torch.stack([reward_from(ep, idx[i], temps[i], ep.power[idx[i]]) for i in range(horizon)]) - comfort
    return temps, comfort, energy


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-start records into errors per horizon and term ratios."""
    if not rows:
        return {}
    out = {"starts": len(rows)}
    for name in HORIZONS:
        out[f"temp_mae_{name}"] = round(float(np.mean([r["mae"][name] for r in rows])), 3)
        out[f"temp_bias_{name}"] = round(float(np.mean([r["bias"][name] for r in rows])), 3)
    #Lorenzo Giannuzzo: ratios of sums over all starts, so that starts with
    #almost no discomfort do not dominate as they would in a mean of ratios.
    for term in ("comfort", "energy"):
        for name in ("12h", "24h"):
            p = sum(r[f"{term}_pred"][name] for r in rows)
            t = sum(r[f"{term}_true"][name] for r in rows)
            out[f"{term}_ratio_{name}"] = round(p / t, 3) if abs(t) > 1e-9 else None
    return out


def diagnose(model, norm, ep, policy, horizon, stride):
    hi = ep.length - horizon - 2
    lo = HIST + 2 + PI_WARMUP
    if hi <= lo:
        return []
    t0 = torch.arange(lo, hi, stride)
    true_t, true_c, true_e = truth(ep, t0, horizon)
    oat = ((ep.aobs[t0, :, AHU_OAT]).mean(-1) - 273.15).tolist()
    occupied = (ep.occ[t0].sum(-1) > 0).tolist()
    rows = []
    for mode, pol in (("replay", None), ("closed_loop", policy)):
        pt, pc, pe = rollout(model, norm, ep, t0, horizon, pol)
        err = pt - true_t
        for b in range(len(t0)):
            rec = {"mode": mode, "episode": ep.name, "station": ep.name.split("__")[1].split("_")[0],
                   "oat_c": oat[b], "occupied": bool(occupied[b]), "mae": {}, "bias": {}}
            for name, k in HORIZONS.items():
                if k <= horizon:
                    rec["mae"][name] = err[k - 1, b].abs().mean().item()
                    rec["bias"][name] = err[k - 1, b].mean().item()
            for term, p, t in (("comfort", pc, true_c), ("energy", pe, true_e)):
                rec[f"{term}_pred"] = {n: p[:k, b].sum().item() for n, k in HORIZONS.items() if k <= horizon}
                rec[f"{term}_true"] = {n: t[:k, b].sum().item() for n, k in HORIZONS.items() if k <= horizon}
            rows.append(rec)
    return rows


def report(rows: list[dict]) -> dict:
    res = {}
    for mode in ("replay", "closed_loop"):
        m = [r for r in rows if r["mode"] == mode]
        res[mode] = {"all": summarize(m)}
        res[mode]["by_station"] = {s: summarize([r for r in m if r["station"] == s])
                                   for s in sorted({r["station"] for r in m})}
        res[mode]["by_occupancy"] = {k: summarize([r for r in m if r["occupied"] == v])
                                     for k, v in (("occupied", True), ("empty", False))}
        bins = [(-60, -10), (-10, 5), (5, 20), (20, 30), (30, 60)]
        res[mode]["by_outdoor_c"] = {f"{a}..{b}": summarize([r for r in m if a <= r["oat_c"] < b]) for a, b in bins}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="+", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--horizon", type=int, default=288)
    ap.add_argument("--stride", type=int, default=72, help="steps between rollout starts")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/wm_diagnosis.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ck = torch.load(args.policy, weights_only=False)
    policy = PolicyNet(**ck["policy_args"])
    policy.load_state_dict(ck["policy"])
    policy.eval()
    files = sorted(f for d in args.episodes for f in Path(d).glob("*.npz"))
    episodes = [load_episode(f) for f in files]
    print(f"{len(episodes)} episodes", flush=True)
    results = {}
    for w in args.wm:
        models, norm = load_models([w])
        rows = []
        for ep in episodes:
            rows += diagnose(models[0], norm, ep, policy, args.horizon, args.stride)
        results[Path(w).stem] = report(rows)
        for mode in ("replay", "closed_loop"):
            a = results[Path(w).stem][mode]["all"]
            print(f"{Path(w).stem:10s} {mode:11s} MAE 1h {a['temp_mae_1h']:.2f} 6h {a['temp_mae_6h']:.2f} "
                  f"12h {a['temp_mae_12h']:.2f} 24h {a['temp_mae_24h']:.2f} K | bias 24h {a['temp_bias_24h']:+.2f} | "
                  f"comfort pred/true 12h {a['comfort_ratio_12h']} 24h {a['comfort_ratio_24h']} | "
                  f"energy 24h {a['energy_ratio_24h']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
