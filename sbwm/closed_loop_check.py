"""Closed-loop calibration of a world model on trajectories of a policy.

The policy is run inside the world model from real states of an on-policy
episode collected in the simulator with the same policy, and the predicted
temperatures, comfort and power are compared with what really happened. This
is the error that matters for DPC: the model's accuracy in the states the
policy itself visits.

Usage:
    python -m sbwm.closed_loop_check --wm wm_runs/v3/world_model.pt \
        --policy policies/dpc_v2.pt --episodes wm_runs/diag
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sbwm.policy import PolicyNet
from sbwm.train_dpc import load_models, policy_step
from sbwm.train_wm import comfort_term
from sbwm.wm_data import load_episode
from sbwm.world_model import Rollout


@torch.no_grad()
def check(model, norm, policy, ep, horizon=144, stride=30):
    t0 = torch.arange(20, ep.length - horizon - 2, stride)
    ro = Rollout(model, norm, ep)
    st = ro.init_state(t0)
    pred_t, true_t, cp, ct, pp, pt = [], [], 0.0, 0.0, [], []
    for i in range(horizon):
        t = t0 + i
        z, a, h, pi_int = policy_step(policy, ro, st, ep, t, norm)
        st, out, nt = ro.step(st, t, z, a, h)
        if pi_int is not None:
            st["pi_int"] = pi_int
        pred_t.append(nt)
        true_t.append(ep.temp[t])
        cp += comfort_term(ep, t, nt).sum().item()
        ct += comfort_term(ep, t, ep.temp[t]).sum().item()
        pp.append(out["power"])
        pt.append(ep.power[t])
    pred_t, true_t = torch.stack(pred_t), torch.stack(true_t)
    err = (pred_t - true_t).abs()
    ratio = torch.stack(pp).sum((0, 1)) / torch.stack(pt).sum((0, 1)).clamp_min(1e-9)
    return {
        "temp_mae_1h_6h_12h": [round(err[k].mean().item(), 2) for k in (11, 71, horizon - 1)],
        "mean_temp_bias": round((pred_t - true_t).mean().item(), 2),
        "comfort_pred_true": [round(cp, 1), round(ct, 1)],
        "power_ratio_elec_gas": [round(x, 2) for x in ratio.tolist()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", default="wm_runs/diag")
    args = ap.parse_args()
    torch.set_num_threads(2)
    models, norm = load_models([args.wm])
    ck = torch.load(args.policy, weights_only=False)
    policy = PolicyNet(**ck["policy_args"])
    policy.load_state_dict(ck["policy"])
    policy.eval()
    policy.action_filter = ck.get("action_filter") if ck.get("filter_in_training") else None
    for f in sorted(Path(args.episodes).glob("*.npz")):
        print(f.name, check(models[0], norm, policy, load_episode(f)), flush=True)


if __name__ == "__main__":
    main()
