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


def _closed_loop_step(policy, ro, ep, norm, hard, t, state, comfort_scale=1.0, anchor=None):
    zact, aact, hact = policy_step(policy, ro, state, ep, t, norm, hard=hard)
    #Lorenzo Giannuzzo: optional trust region, the squared distance from the
    #actions an anchor policy (the one that collected the latest on-policy
    #data) would take in the same state, which keeps each update inside the
    #region where the world model has just been corrected.
    if anchor is not None:
        with torch.no_grad():
            za, aa, ha = policy_step(anchor, ro, state, ep, t, norm, hard=hard)
        dev = ((zact - za) ** 2).mean((1, 2)) + ((aact - aa) ** 2).mean((1, 2)) + ((hact - ha) ** 2).mean(1)
    else:
        dev = torch.zeros(zact.shape[0])
    new_state, out, new_temp = ro.step(state, t, zact, aact, hact)
    return new_state, reward_from(ep, t, new_temp, out["power"], comfort_scale), dev


def needs_checkpoint(ep, length: int) -> bool:
    """Checkpoint only when the stored activations would be large.

    Activation memory grows with the horizon and roughly with N * (N + 64)
    for N tokens (feed-forward plus attention); the threshold keeps the run
    within a few GB on the largest generated buildings.
    """
    n = ep.schema.num_zones + ep.schema.num_ahus + 2
    return length * n * (n + 64) > 500_000


def rollout_return(policy, model, norm, ep, t0, length, gamma, hard=True, grad_decay=1.0,
                   checkpoint=None, comfort_scale=1.0, anchor=None, return_dev=False):
    """Discounted predicted reward (B,) and per-step rewards (L, B).

    With ``return_dev`` the mean squared action distance from ``anchor`` over
    the rollout is returned as a third value.
    """
    from torch.utils.checkpoint import checkpoint as ckpt

    if checkpoint is None:
        checkpoint = needs_checkpoint(ep, length)
    ro = Rollout(model, norm, ep)
    state = ro.init_state(t0)
    rewards, devs = [], []
    for i in range(length):
        t = t0 + i
        #Lorenzo Giannuzzo: each closed-loop step is recomputed during the
        #backward pass instead of being stored, so memory no longer grows with
        #the activations of every world-model step over the horizon.
        if checkpoint and torch.is_grad_enabled():
            state, r, dev = ckpt(_closed_loop_step, policy, ro, ep, norm, hard, t, state,
                                 comfort_scale, anchor, use_reentrant=False)
        else:
            state, r, dev = _closed_loop_step(policy, ro, ep, norm, hard, t, state,
                                              comfort_scale, anchor)
        rewards.append(r)
        devs.append(dev)
        #Lorenzo Giannuzzo: backpropagation through many world-model steps can
        #explode; the gradient carried by the state is damped geometrically,
        #so credit from h steps later is weighted by grad_decay**h.
        if grad_decay < 1.0:
            state = {k: _damp(v, grad_decay) for k, v in state.items()}
    r = torch.stack(rewards)
    disc = gamma ** torch.arange(length, dtype=r.dtype).unsqueeze(1)
    if return_dev:
        return (disc * r).sum(0), r, torch.stack(devs).mean()
    return (disc * r).sum(0), r


@torch.no_grad()
def evaluate(policy, models, norm, episodes, horizon=144, starts=3, seed=0, cache=None):
    """Mean predicted per-step reward of the policy versus the logged actions.

    The logged-action reference does not depend on the policy, so it is
    computed once and kept in ``cache``.
    """
    rng = np.random.default_rng(seed)
    pol = []
    first = cache is not None and not cache
    for ep in episodes:
        hi = ep.length - horizon - 2
        t0 = torch.as_tensor(rng.integers(HIST + 2, hi, starts))
        for m in models:
            _, r = rollout_return(policy, m, norm, ep, t0, horizon, 1.0)
            pol.append(r.mean().item())
            if first:
                ro = Rollout(m, norm, ep)
                state = ro.init_state(t0)
                rl = []
                for i in range(horizon):
                    t = t0 + i
                    state, out, nt = ro.step(state, t, ep.zact[t], ep.aact[t], ep.hact[t])
                    rl.append(reward_from(ep, t, nt, out["power"]))
                cache.setdefault("pred", []).append(torch.stack(rl).mean().item())
        if first:
            idx = (t0.unsqueeze(1) + torch.arange(horizon)).reshape(-1)
            cache.setdefault("true", []).append(ep.reward[idx].mean().item())
    res = {"policy_pred": float(np.mean(pol))}
    if cache:
        res["logged_pred"] = float(np.mean(cache["pred"]))
        res["logged_true"] = float(np.mean(cache["true"]))
    return res


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
    ap.add_argument("--lengths", type=int, nargs="+", default=[24, 48, 72],
                    help="horizon curriculum, in equal shares of the iterations")
    ap.add_argument("--comfort-scale", type=float, default=1.0,
                    help="multiplier on the predicted comfort term, to offset world-model optimism")
    ap.add_argument("--init", default=None, help="policy checkpoint to continue from")
    ap.add_argument("--anchor", default=None, help="policy checkpoint used as trust-region anchor")
    ap.add_argument("--anchor-weight", type=float, default=0.0)
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
    anchor = None
    if args.anchor:
        ack = torch.load(args.anchor, weights_only=False)
        anchor = PolicyNet(**ack["policy_args"])
        anchor.load_state_dict(ack["policy"])
        anchor.eval()
        for prm in anchor.parameters():
            prm.requires_grad_(False)
    print(f"policy params {sum(p.numel() for p in policy.parameters())} | train eps {len(train)} "
          f"| models {len(models)}", flush=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    log = []
    best = -1e9
    eval_cache = {}
    t_start = time.time()
    for it in range(1, args.iters + 1):
        #Lorenzo Giannuzzo: the horizon grows along the curriculum (by default
        #2, 4 and 6 hours), long enough to see the effect of preconditioning.
        frac = (it - 1) / args.iters
        length = args.lengths[min(int(frac * len(args.lengths)), len(args.lengths) - 1)]
        ep = train[rng.integers(len(train))]
        model = models[rng.integers(len(models))]
        t0 = torch.as_tensor(rng.integers(HIST + 2, ep.length - length - 2, args.batch))
        ret, r, dev = rollout_return(policy, model, norm, ep, t0, length, args.gamma,
                                     grad_decay=args.grad_decay, comfort_scale=args.comfort_scale,
                                     anchor=anchor, return_dev=True)
        loss = -ret.mean() / length + args.anchor_weight * dev
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 25 == 0:
            print(f"it {it} L={length} step_reward {r.mean().item():.4f} dev {dev.item():.4f} grad {gnorm:.3f} "
                  f"| {time.time() - t_start:.0f}s", flush=True)
        if it % args.eval_every == 0 or it == args.iters:
            res = evaluate(policy, models, norm, test, cache=eval_cache)
            res["it"] = it
            log.append(res)
            print(json.dumps(res), flush=True)
            ck = {"policy": policy.state_dict(), "policy_args": policy_args,
                  "norm": norm.state_dict(), "wm": args.wm, "it": it, "eval": res}
            torch.save(ck, out / "policy_last.pt")
            torch.save(ck, out / f"policy_it{it:05d}.pt")
            if res["policy_pred"] > best:
                best = res["policy_pred"]
                torch.save(ck, out / "policy.pt")
            (out / "dpc_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
