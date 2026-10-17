"""Differentiable Predictive Control: train the policy through the world model.

From start states drawn from collected episodes, the policy acts, the world
model predicts the next state, and the competition reward is rebuilt from the
predicted temperatures and power. The discounted reward over the horizon is
maximized by backpropagation through the whole closed loop. Exogenous inputs
(weather, occupancy, comfort schedule, tariffs) come from the episode.

Usage:
    python -m sbwm.train_dpc --wm wm_runs/v2_s0/world_model.pt \
        --train wm_data/episodes_train pc_data/episodes --test wm_data/episodes_test \
        --out wm_runs/policy --iters 3000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sbwm.policy import N_OBS_HIST, PolicyNet, build_inputs
from sbwm.train_wm import reward_from
from sbwm.wm_data import HIST, Normalizer, load_dir
from sbwm.world_model import Rollout, WorldModel

assert HIST + 1 >= N_OBS_HIST + 1


def load_models(paths):
    models, norm = [], None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = WorldModel(d=a.get("d", 128), layers=a.get("layers", 4))
        m.load_state_dict(ck["model"])
        m.eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
        models.append(m)
        norm = norm or Normalizer.from_state_dict(ck["norm"])
    return models, norm


def policy_step(policy, ro, state, ep, t, norm, hard=True):
    obs_hist = state["temp_hist"][:, 1:1 + N_OBS_HIST]
    zin, ain, hin, gin = build_inputs(
        state["zobs"], state["aobs"], state["hobs"], ep.gobs[t], obs_hist, ep.static,
        state["zact_prev"], state["aact_prev"], state["hact_prev"], ep.sched, norm.dtemp_std)
    return policy(zin, ain, hin, gin, ro.rel, hard=hard)


def _damp(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Same value, gradient through the recurrence scaled by lam."""
    return x.detach() + lam * (x - x.detach())


def rollout_return(policy, model, norm, ep, t0, length, gamma, hard=True, grad_decay=1.0):
    """Discounted predicted reward (B,) and per-step rewards (L, B)."""
    ro = Rollout(model, norm, ep)
    state = ro.init_state(t0)
    rewards = []
    for i in range(length):
        t = t0 + i
        zact, aact, hact = policy_step(policy, ro, state, ep, t, norm, hard=hard)
        state, out, new_temp = ro.step(state, t, zact, aact, hact)
        rewards.append(reward_from(ep, t, new_temp, out["power"]))
        #Lorenzo Giannuzzo: backpropagation through many world-model steps can
        #explode; the gradient carried by the state is damped geometrically,
        #so credit from h steps later is weighted by grad_decay**h.
        if grad_decay < 1.0:
            state = {k: _damp(v, grad_decay) for k, v in state.items()}
    r = torch.stack(rewards)
    disc = gamma ** torch.arange(length, dtype=r.dtype).unsqueeze(1)
    return (disc * r).sum(0), r


@torch.no_grad()
def evaluate(policy, models, norm, episodes, horizon=144, starts=4, seed=0):
    """Mean predicted per-step reward of the policy versus the logged actions."""
    rng = np.random.default_rng(seed)
    pol, logged_pred, logged_true = [], [], []
    for ep in episodes:
        hi = ep.length - horizon - 2
        t0 = torch.as_tensor(rng.integers(HIST + 2, hi, starts))
        for m in models:
            _, r = rollout_return(policy, m, norm, ep, t0, horizon, 1.0)
            pol.append(r.mean().item())
            ro = Rollout(m, norm, ep)
            state = ro.init_state(t0)
            rl = []
            for i in range(horizon):
                t = t0 + i
                state, out, nt = ro.step(state, t, ep.zact[t], ep.aact[t], ep.hact[t])
                rl.append(reward_from(ep, t, nt, out["power"]))
            logged_pred.append(torch.stack(rl).mean().item())
        idx = (t0.unsqueeze(1) + torch.arange(horizon)).reshape(-1)
        logged_true.append(ep.reward[idx].mean().item())
    return {"policy_pred": float(np.mean(pol)), "logged_pred": float(np.mean(logged_pred)),
            "logged_true": float(np.mean(logged_true))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", nargs="+", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-decay", type=float, default=0.95)
    ap.add_argument("--init", default=None, help="policy checkpoint to continue from")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    models, norm = load_models(args.wm)
    train = [e for d in args.train for e in load_dir(Path(d))]
    test = load_dir(Path(args.test))
    policy_args = {"d": args.d, "layers": args.layers}
    policy = PolicyNet(**policy_args)
    if args.init:
        policy.load_state_dict(torch.load(args.init, weights_only=False)["policy"])
    print(f"policy params {sum(p.numel() for p in policy.parameters())} | train eps {len(train)} "
          f"| models {len(models)}", flush=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    log = []
    best = -1e9
    t_start = time.time()
    for it in range(1, args.iters + 1):
        #Lorenzo Giannuzzo: the horizon grows from 2 to 6 hours, long enough to
        #see the effect of preconditioning before the comfort period.
        frac = it / args.iters
        length = 24 if frac < 0.3 else 48 if frac < 0.7 else 72
        ep = train[rng.integers(len(train))]
        model = models[rng.integers(len(models))]
        t0 = torch.as_tensor(rng.integers(HIST + 2, ep.length - length - 2, args.batch))
        ret, r = rollout_return(policy, model, norm, ep, t0, length, args.gamma,
                                grad_decay=args.grad_decay)
        loss = -ret.mean() / length
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 25 == 0:
            print(f"it {it} L={length} step_reward {r.mean().item():.4f} grad {gnorm:.3f} "
                  f"| {time.time() - t_start:.0f}s", flush=True)
        if it % args.eval_every == 0 or it == args.iters:
            res = evaluate(policy, models, norm, test)
            res["it"] = it
            log.append(res)
            print(json.dumps(res), flush=True)
            ck = {"policy": policy.state_dict(), "policy_args": policy_args,
                  "norm": norm.state_dict(), "wm": args.wm, "it": it, "eval": res}
            torch.save(ck, out / "policy_last.pt")
            if res["policy_pred"] > best:
                best = res["policy_pred"]
                torch.save(ck, out / "policy.pt")
            (out / "dpc_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
