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
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as ckpt

from sbwm.critic import VALUE_SCALE, CriticNet, critic_loss, critic_y, lambda_returns, to_sum
from sbwm.policy import (N_OBS_HIST, PI_WARMUP, PolicyNet, band_excess, build_inputs, filter_actions, hybrid_plant,
                         hybrid_zone_actions)
from sbwm.train_wm import reward_from
from sbwm.wm_data import HIST, EpisodePool, Normalizer, list_files, load_dir
from sbwm.world_model import Rollout, WorldModel

assert HIST + 1 >= N_OBS_HIST + 1


def load_models(paths):
    models, norm = [], None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = WorldModel(d=a.get("d", 128), layers=a.get("layers", 4), mem=a.get("mem", False))
        m.load_state_dict(ck["model"])
        m.eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
        models.append(m)
        norm = norm or Normalizer.from_state_dict(ck["norm"])
    return models, norm


def policy_step(policy, ro, state, ep, t, norm, hard=True):
    """Actions (zone, AHU, HWS) and, for the hybrid policy, the new PI integrals."""
    obs_hist = state["temp_hist"][:, 1:1 + N_OBS_HIST]
    hybrid = getattr(policy, "hybrid", False)
    pi_int = state.get("pi_int") if hybrid else None
    if hybrid and pi_int is None:
        pi_int = torch.zeros(state["zobs"].shape[:2])
    zin, ain, hin, gin = build_inputs(
        state["zobs"], state["aobs"], state["hobs"], ep.gobs[t], obs_hist, ep.static,
        state["zact_prev"], state["aact_prev"], state["hact_prev"], ep.sched, norm.dtemp_std,
        pi_int)
    zact, aact, hact = policy(zin, ain, hin, gin, ro.rel, hard=hard)
    if hybrid:
        zact, pi_int, u = hybrid_zone_actions(zact, state["zobs"], pi_int, policy.pi,
                                              obs_hist[:, 0] - obs_hist[:, 1])
        aact, hact = hybrid_plant(aact, hact, u, torch.as_tensor(ep.schema.zone_ahu),
                                  ep.schema.num_ahus, policy.pi, band_excess(state["zobs"]))
    alpha = getattr(policy, "action_filter", None)
    if alpha is not None:
        #Lorenzo Giannuzzo: the commands reach the world model through the same
        #filter used at deployment, so the policy learns with it in the loop.
        zact, aact, hact = filter_actions(alpha, zact, aact, hact,
                                          state["zact_prev"], state["aact_prev"], state["hact_prev"])
    return zact, aact, hact, pi_int


def _damp(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Same value, gradient through the recurrence scaled by lam."""
    return x.detach() + lam * (x - x.detach())


def _closed_loop_step(policy, ro, ep, norm, hard, t, state, comfort_scale=1.0, anchor=None, smooth=0.0):
    zact, aact, hact, pi_int = policy_step(policy, ro, state, ep, t, norm, hard=hard)
    #Lorenzo Giannuzzo: optional trust region, the squared distance from the
    #actions an anchor policy (the one that collected the latest on-policy
    #data) would take in the same state, which keeps each update inside the
    #region where the world model has just been corrected.
    if anchor is not None:
        with torch.no_grad():
            za, aa, ha, _ = policy_step(anchor, ro, state, ep, t, norm, hard=hard)
        dev = ((zact - za) ** 2).mean((1, 2)) + ((aact - aa) ** 2).mean((1, 2)) + ((hact - ha) ** 2).mean(1)
    else:
        dev = torch.zeros(zact.shape[0])
    new_state, out, new_temp = ro.step(state, t, zact, aact, hact)
    if pi_int is not None:
        new_state["pi_int"] = pi_int
    r = reward_from(ep, t, new_temp, out["power"], comfort_scale)
    if smooth > 0:
        #Lorenzo Giannuzzo: training-only penalty on step-to-step changes of the
        #continuous commands (zone dampers and reheat, supply air, pressure and
        #recirculation, water temperature and pressure), against the chattering
        #seen in the control figures; on/off decisions are left free.
        dz = ((zact - state["zact_prev"]) ** 2).mean((1, 2))
        da = ((aact[..., 1:] - state["aact_prev"][..., 1:]) ** 2).mean((1, 2))
        dh = ((hact[:, 1:] - state["hact_prev"][:, 1:]) ** 2).mean(1)
        r = r - smooth * (dz + da + dh)
    return new_state, r, dev


def init_policy(policy, sd: dict) -> None:
    """Load a checkpoint, also from a direct-action policy into a hybrid one.

    The trunk and the AHU and hot-water heads are copied. For a hybrid target
    the extra zone input starts with zero weights and the zone head keeps its
    fresh initialization (tuned PI margins, unit gain, no residual), so the
    initial policy is the source policy's plant logic on top of the zone PI.
    """
    own = policy.state_dict()
    for k, v in sd.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k] = v
        elif k == "enc_zone.0.weight" and own[k].shape[1] == v.shape[1] + 1:
            own[k] = torch.cat([v, torch.zeros(v.shape[0], 1)], 1)
        elif k.startswith("head_zone.") and v.shape[0] == 5 and own[k].shape[0] == 6:
            #Lorenzo Giannuzzo: a hybrid head gains the derivative output, which
            #keeps its fresh (near zero) initialization.
            own[k][:5] = v
    policy.load_state_dict(own)


def needs_checkpoint(ep, length: int) -> bool:
    """Checkpoint only when the stored activations would be large.

    Activation memory grows with the horizon and roughly with N * (N + 64)
    for N tokens (feed-forward plus attention); the threshold keeps the run
    within a few GB on the largest generated buildings.
    """
    n = ep.schema.num_zones + ep.schema.num_ahus + 2
    return length * n * (n + 64) > 500_000


def rollout_return(policy, model, norm, ep, t0, length, gamma, hard=True, grad_decay=1.0,
                   checkpoint=None, comfort_scale=1.0, anchor=None, return_dev=False, keep_states=False,
                   smooth=0.0):
    """Discounted predicted reward (B,) and per-step rewards (L, B).

    With ``return_dev`` the mean squared action distance from ``anchor`` over
    the rollout is returned as a third value. With ``keep_states`` the states
    s_0..s_L, their decision indices and the rollout are appended (for the
    critic).
    """
    from torch.utils.checkpoint import checkpoint as ckpt

    if checkpoint is None:
        checkpoint = needs_checkpoint(ep, length)
    ro = Rollout(model, norm, ep)
    #Lorenzo Giannuzzo: the hybrid policy carries PI integrals, which at
    #deployment have been accumulating for hours; a short closed-loop warm-up
    #without gradient replaces the cold start before the scored horizon.
    warm = PI_WARMUP if getattr(policy, "hybrid", False) else 0
    state = ro.init_state(t0 - warm)
    if warm:
        with torch.no_grad():
            for i in range(warm):
                state, _, _ = _closed_loop_step(policy, ro, ep, norm, hard, t0 - warm + i, state)
    rewards, devs = [], []
    states, ts = [state], [t0]
    for i in range(length):
        t = t0 + i
        #Lorenzo Giannuzzo: each closed-loop step is recomputed during the
        #backward pass instead of being stored, so memory no longer grows with
        #the activations of every world-model step over the horizon.
        if checkpoint and torch.is_grad_enabled():
            state, r, dev = ckpt(_closed_loop_step, policy, ro, ep, norm, hard, t, state,
                                 comfort_scale, anchor, smooth, use_reentrant=False)
        else:
            state, r, dev = _closed_loop_step(policy, ro, ep, norm, hard, t, state,
                                              comfort_scale, anchor, smooth)
        rewards.append(r)
        devs.append(dev)
        #Lorenzo Giannuzzo: backpropagation through many world-model steps can
        #explode; the gradient carried by the state is damped geometrically,
        #so credit from h steps later is weighted by grad_decay**h.
        if grad_decay < 1.0:
            state = {k: _damp(v, grad_decay) for k, v in state.items()}
        if keep_states:
            states.append(state)
            ts.append(t + 1)
    r = torch.stack(rewards)
    disc = gamma ** torch.arange(length, dtype=r.dtype).unsqueeze(1)
    out = ((disc * r).sum(0), r)
    if return_dev:
        out += (torch.stack(devs).mean(),)
    if keep_states:
        out += (states, ts, ro)
    return out


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
        t0 = torch.as_tensor(rng.integers(HIST + 2 + PI_WARMUP, hi, starts))
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


@torch.no_grad()
def critic_check(policy, critic, model, norm, episodes, gamma, comfort_scale=1.0, horizon=288, starts=3, seed=1,
                 rclip=0.0):
    """Calibration of the critic against longer imagined rollouts.

    From test states, the value predicted at the start is compared with the
    discounted reward of a ``horizon``-step rollout of the policy plus the
    critic's value at its end; both are reported per step. A calibrated
    critic has small bias and high correlation.
    """
    rng = np.random.default_rng(seed)
    pred, mc = [], []
    for ep in episodes:
        hi = ep.length - horizon - 2
        if hi <= HIST + 2 + PI_WARMUP:
            continue
        t0 = torch.as_tensor(rng.integers(HIST + 2 + PI_WARMUP, hi, starts))
        _, r, states, ts, ro = rollout_return(policy, model, norm, ep, t0, horizon, gamma,
                                              comfort_scale=comfort_scale, keep_states=True)
        v = to_sum(critic_y(critic, [states[0], states[-1]], [ts[0], ts[-1]], ep, norm, ro.rel).mean(0), gamma)
        if rclip > 0:
            r = r.clamp(min=-rclip)  # the same floor as in the critic targets
        disc = gamma ** torch.arange(horizon, dtype=r.dtype).unsqueeze(1)
        pred.append((1 - gamma) * v[0])
        mc.append((1 - gamma) * ((disc * r).sum(0) + gamma ** horizon * v[1]))
    pred, mc = torch.cat(pred), torch.cat(mc)
    corr = torch.corrcoef(torch.stack([pred, mc]))[0, 1].item() if len(pred) > 2 else float("nan")
    return {"critic_bias": round((pred - mc).mean().item(), 5), "critic_corr": round(corr, 3),
            "critic_mc_mean": round(mc.mean().item(), 5)}


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
    ap.add_argument("--pool", type=int, default=0, help="start-state episodes in memory (0: all)")
    ap.add_argument("--refresh", type=int, default=100, help="iterations between pool refreshes")
    ap.add_argument("--anchor", default=None, help="policy checkpoint used as trust-region anchor")
    ap.add_argument("--anchor-weight", type=float, default=0.0)
    ap.add_argument("--action-filter", type=float, default=None,
                    help="train with the low-pass command filter in the loop (weight of the new command)")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="weight of the penalty on step-to-step changes of the continuous commands")
    ap.add_argument("--exclude-buildings", nargs="*", default=[],
                    help="building lists (csv with a footprint column); default-footprint buildings are skipped")
    ap.add_argument("--hybrid", action="store_true",
                    help="transformer supervises a differentiable PI per zone")
    ap.add_argument("--pi", default="", help="PI constants, e.g. margin=0.5,ki=0.02,full=2,decay=0.9")
    ap.add_argument("--critic", action="store_true",
                    help="bootstrap the rollouts with a learned value function (lambda-returns)")
    ap.add_argument("--critic-d", type=int, default=64)
    ap.add_argument("--critic-layers", type=int, default=3)
    ap.add_argument("--critic-ens", type=int, default=2, help="independent value networks in the ensemble")
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--critic-horizon", type=int, default=576,
                    help="steps of the gradient-free rollouts the critic learns from (576: 48 h)")
    ap.add_argument("--critic-batch", type=int, default=8, help="start states per critic rollout")
    ap.add_argument("--critic-updates", type=int, default=2, help="critic steps per critic rollout")
    ap.add_argument("--critic-warmup", type=int, default=200,
                    help="critic-only iterations before the policy starts to move")
    ap.add_argument("--critic-lambda", type=float, default=1.0,
                    help="lambda of the critic targets (the world model is deterministic, so 1 adds no variance)")
    ap.add_argument("--actor-lambda", type=float, default=0.95, help="lambda of the policy objective")
    ap.add_argument("--critic-tau", type=float, default=0.05, help="target-critic update rate")
    ap.add_argument("--critic-sample", type=float, default=0.25,
                    help="share of rollout steps used as critic regression samples")
    ap.add_argument("--critic-pessimism", type=int, default=1,
                    help="1: the policy sees the ensemble minimum, 0: the ensemble mean")
    ap.add_argument("--critic-init", default=None, help="checkpoint whose critic is loaded")
    ap.add_argument("--critic-chunk", type=int, default=12, help="states per checkpointed critic call")
    ap.add_argument("--critic-version", type=int, default=2,
                    help="2: physical state only (no commands, readbacks or previous actions); 1: policy inputs")
    ap.add_argument("--actor-mode", choices=["terminal", "lambda"], default="terminal",
                    help="terminal: rollout return plus the weighted value of the final state; "
                         "lambda: lambda-returns with the value at every step")
    ap.add_argument("--critic-beta", type=float, default=0.5, help="weight of the terminal value")
    ap.add_argument("--critic-rclip", type=float, default=0.1,
                    help="per-step rewards floored at -rclip in the critic targets (0: no floor)")
    ap.add_argument("--critic-noise", type=float, default=0.05,
                    help="state noise (normalized units) on the critic regression inputs")
    ap.add_argument("--critic-every", type=int, default=1, help="policy iterations per critic rollout")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    models, norm = load_models(args.wm)
    files = list_files(args.train)
    #Lorenzo Giannuzzo: start states can be restricted to buildings in the size
    #range of the official ones, dropping the default (larger) footprints.
    excluded = set()
    for csv_path in args.exclude_buildings:
        for line in Path(csv_path).read_text().splitlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[-1] == "default":
                excluded.add(parts[0])
    if excluded:
        files = [f for f in files if f.name.split("__")[0] not in excluded]
        print(f"excluded {len(excluded)} large buildings, {len(files)} start-state files left", flush=True)
    train = EpisodePool(files, args.pool, rng, args.refresh)
    test = load_dir(Path(args.test))
    policy_args = {"d": args.d, "layers": args.layers}
    if args.hybrid:
        from sbwm.zone_pi import parse_params

        policy_args.update(hybrid=True, pi={k: float(v) for k, v in parse_params(args.pi).items()})
    policy = PolicyNet(**policy_args)
    if args.init:
        init_policy(policy, torch.load(args.init, weights_only=False)["policy"])
    policy.action_filter = args.action_filter
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
    critic = target = copt = None
    critic_args = {"d": args.critic_d, "layers": args.critic_layers, "ens": args.critic_ens,
                   "version": args.critic_version}
    if args.critic:
        critic = CriticNet(**critic_args)
        if args.critic_init:
            cck = torch.load(args.critic_init, weights_only=False)
            if "critic" in cck:
                critic.load_state_dict(cck["critic"])
                print(f"critic loaded from {args.critic_init}", flush=True)
        target = copy.deepcopy(critic)
        for prm in target.parameters():
            prm.requires_grad_(False)
        copt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=1e-4)
        print(f"critic params {sum(p.numel() for p in critic.parameters())}", flush=True)

    #Lorenzo Giannuzzo: with a discount of 0.997 the value looks about 28 h
    #ahead; on the 6-12 h rollouts of the policy update its targets would be
    #mostly its own bootstrap and the value would spread across states very
    #slowly. Rollouts of 48 h without gradient make most of each target real
    #(predicted) reward, at a fraction of the cost of the policy update.
    def critic_round(ep, model, first=False):
        """One gradient-free long rollout of the current policy and critic steps on it."""
        length = args.critic_horizon
        t0 = torch.as_tensor(rng.integers(HIST + 2 + PI_WARMUP, max(HIST + 3 + PI_WARMUP, ep.length - length - 2),
                                          args.critic_batch))
        length = min(length, ep.length - 2 - int(t0.max()))
        with torch.no_grad():
            _, r, states, ts, ro = rollout_return(policy, model, norm, ep, t0, length, args.gamma,
                                                  comfort_scale=args.comfort_scale, keep_states=True)
            if first:
                #Lorenzo Giannuzzo: the value starts at the mean reward of the
                #first rollouts instead of an arbitrary level.
                r0 = r.clamp(min=-args.critic_rclip) if args.critic_rclip > 0 else r
                for net in (critic, target):
                    for m in net.members:
                        m.head_glob.bias.fill_(r0.mean().item() / VALUE_SCALE)
        for _ in range(args.critic_updates):
            closs, ret = critic_loss(critic, target, states, ts, r, ep, norm, ro.rel, args.gamma,
                                     args.critic_lambda, args.critic_sample, rng,
                                     rclip=args.critic_rclip, noise=args.critic_noise)
            copt.zero_grad()
            closs.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            copt.step()
        with torch.no_grad():
            for pt, pc in zip(target.parameters(), critic.parameters()):
                pt.lerp_(pc, args.critic_tau)
        return closs.item(), ret, r

    #Lorenzo Giannuzzo: the critic first learns the value of the initial policy,
    #so the policy is never pushed by an untrained value function.
    t_start = time.time()
    if critic is not None and not args.critic_init:
        for it in range(1, args.critic_warmup + 1):
            train.maybe_refresh()
            closs, ret, r = critic_round(train.sample(), models[rng.integers(len(models))], first=it == 1)
            if it % 25 == 0:
                print(f"critic warm-up {it} loss {closs:.4f} return/step {(1 - args.gamma) * ret[0].mean().item():.4f} "
                      f"reward {r.mean().item():.4f} | {time.time() - t_start:.0f}s", flush=True)
    log = []
    best = -1e9
    eval_cache = {}
    skipped = 0
    for it in range(1, args.iters + 1):
        #Lorenzo Giannuzzo: the horizon grows along the curriculum (by default
        #2, 4 and 6 hours), long enough to see the effect of preconditioning.
        frac = (it - 1) / args.iters
        length = args.lengths[min(int(frac * len(args.lengths)), len(args.lengths) - 1)]
        train.maybe_refresh()
        ep = train.sample()
        model = models[rng.integers(len(models))]
        t0 = torch.as_tensor(rng.integers(HIST + 2 + PI_WARMUP, ep.length - length - 2, args.batch))
        res_ro = rollout_return(policy, model, norm, ep, t0, length, args.gamma,
                                grad_decay=args.grad_decay, comfort_scale=args.comfort_scale,
                                anchor=anchor, return_dev=True, keep_states=critic is not None,
                                smooth=args.smooth)
        ret, r, dev = res_ro[:3]
        if critic is None:
            loss = -ret.mean() / length + args.anchor_weight * dev
        else:
            #Lorenzo Giannuzzo: the policy maximizes the lambda-returns of the
            #rollout, each bootstrapped with the critic's value of the state
            #reached; the value is differentiable with respect to that state, so
            #its gradient flows back through the world model to the actions.
            states, ts, ro = res_ro[3:]
            if args.actor_mode == "terminal":
                #Lorenzo Giannuzzo: the DPC return stays the main objective and
                #the critic only adds the (weighted) value of the state reached
                #at the end of the rollout, as in SHAC; its gradient reaches the
                #actions only through the world model.
                y = critic_y(critic, [states[-1]], [ts[-1]], ep, norm, ro.rel)[:, 0]
                y = y.min(0).values if args.critic_pessimism else y.mean(0)
                term = args.critic_beta * args.gamma ** length * to_sum(y, args.gamma)
                loss = -(ret + term).mean() / length + args.anchor_weight * dev
            else:
                ys = [ckpt(critic_y, critic, states[c:c + args.critic_chunk], ts[c:c + args.critic_chunk],
                           ep, norm, ro.rel, use_reentrant=False)
                      for c in range(1, len(states), args.critic_chunk)]
                y = torch.cat(ys, 1)
                y = y.min(0).values if args.critic_pessimism else y.mean(0)
                lam_ret = lambda_returns(r, to_sum(y, args.gamma), args.gamma, args.actor_lambda)
                loss = -(1 - args.gamma) * lam_ret.mean() + args.anchor_weight * dev
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        #Lorenzo Giannuzzo: a rollout that blows up in the world model gives an
        #infinite or NaN gradient; that update is skipped, since one of them is
        #enough to turn the weights into NaN (v10_long on the PC).
        if torch.isfinite(loss) and torch.isfinite(gnorm):
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
            print(f"it {it}: non-finite loss or gradient, update skipped ({skipped} so far)", flush=True)
        sched.step()
        closs = None
        if critic is not None and it % args.critic_every == 0:
            closs, _, _ = critic_round(ep, model)
        if it % 25 == 0:
            extra = "" if closs is None else f" critic_loss {closs:.4f} objective {-loss.item():.4f}"
            print(f"it {it} L={length} step_reward {r.mean().item():.4f} dev {dev.item():.4f} grad {gnorm:.3f}"
                  f"{extra} | {time.time() - t_start:.0f}s", flush=True)
        if it % args.eval_every == 0 or it == args.iters:
            res = evaluate(policy, models, norm, test, cache=eval_cache)
            if critic is not None:
                res.update(critic_check(policy, critic, models[0], norm, test, args.gamma, args.comfort_scale,
                                        rclip=args.critic_rclip))
            res["it"] = it
            log.append(res)
            print(json.dumps(res), flush=True)
            ck = {"policy": policy.state_dict(), "policy_args": policy_args,
                  "norm": norm.state_dict(), "wm": args.wm, "it": it, "eval": res}
            if critic is not None:
                ck.update(critic=critic.state_dict(), critic_args=critic_args)
            if args.action_filter is not None:
                ck.update(action_filter=args.action_filter, filter_in_training=True)
            torch.save(ck, out / "policy_last.pt")
            torch.save(ck, out / f"policy_it{it:05d}.pt")
            if res["policy_pred"] > best:
                best = res["policy_pred"]
                torch.save(ck, out / "policy.pt")
            (out / "dpc_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
