"""Train and evaluate the entity-token world model.

Usage:
    python -m sbwm.train_wm --train wm_data/episodes_train pc_data/episodes \
        --test wm_data/episodes_test --out wm_runs/v2_s0 --steps 6000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sbwm.wm_data import AHU_DYN, HIST, ZONE_DYN, Normalizer, load_dir
from sbwm.world_model import Rollout, WorldModel

HORIZONS = (1, 12, 72, 144)  # 5 min, 1 h, 6 h, 12 h


def comfort_term(ep, t, temp):
    """Weighted comfort part of the reward, 0.6 * regret, per batch element."""
    rm = ep.reward_meta
    w = rm["productivity_deviation_ref"]
    dev = torch.clamp(torch.maximum(temp - ep.csp[t], ep.hsp[t] - temp), min=0.0)
    shortfall = torch.where(dev <= w, dev / w, 1 + ((dev - w) / w) ** 2)
    occ = ep.occ[t]
    tot = occ.sum(-1)
    maxp, minp = rm["max_productivity_personhour_usd"], rm["min_productivity_personhour_usd"]
    act = (maxp * (1 - shortfall) * occ).sum(-1)
    regret = torch.where(tot > 0, (act - minp * tot) / ((maxp - minp) * tot).clamp_min(1e-9) - 1, 0.0)
    return rm["productivity_weight"] * regret


def unroll_loss(model, norm, ep, t0, length, w_power=10.0, w_comfort=1.0, zone_boost=3.0):
    ro = Rollout(model, norm, ep)
    state = ro.init_state(t0)
    total = 0.0
    parts = {"temp": 0.0, "comfort": 0.0, "zone": 0.0, "ahu": 0.0, "hws": 0.0, "power": 0.0}
    for i in range(length):
        t = t0 + i
        state, out, new_temp = ro.step(state, t, ep.zact[t], ep.aact[t], ep.hact[t])
        #Lorenzo Giannuzzo: zones that are occupied and outside, or within 1 K
        #of, their comfort band weigh more, since that is where the reward is
        #sensitive to temperature errors and where v1 smoothed excursions.
        true_t = ep.temp[t]
        signed = torch.maximum(true_t - ep.csp[t], ep.hsp[t] - true_t)
        weight = 1.0 + zone_boost * ((ep.occ[t] > 0) & (signed > -1.0)).float()
        err2 = ((new_temp - true_t) / norm.dtemp_std) ** 2
        l_temp = (weight * err2).sum() / weight.sum()
        #Lorenzo Giannuzzo: the comfort term of the reward is also matched
        #directly, so that the error is measured in the units the DPC loss uses.
        l_comf = (((comfort_term(ep, t, new_temp) - comfort_term(ep, t, true_t)) / 0.1) ** 2).mean()
        zt = ((ep.zobs[t + 1] - norm.z_mean) / norm.z_std)[..., list(ZONE_DYN)]
        at = ((ep.aobs[t + 1] - norm.a_mean) / norm.a_std)[..., list(AHU_DYN)]
        ht = (ep.hobs[t + 1] - norm.h_mean) / norm.h_std
        l_zone = ((out["zone_dyn"] - zt) ** 2).mean()
        l_ahu = ((out["ahu_dyn"] - at) ** 2).mean()
        l_hws = ((out["hws"] - ht) ** 2).mean()
        l_pow = ((out["power"] - ep.power[t]) ** 2).mean()
        loss = l_temp + w_comfort * l_comf + 0.3 * (l_zone + l_ahu + l_hws) + w_power * l_pow
        total = total + loss / length
        for k, v in zip(parts, (l_temp, l_comf, l_zone, l_ahu, l_hws, l_pow)):
            parts[k] += v.item() / length
    return total, parts


@torch.no_grad()
def evaluate(model, norm, episodes, starts_per_ep=6, horizon=144, seed=0):
    """Closed-loop rollout errors on held-out episodes, versus persistence."""
    model.eval()
    rng = np.random.default_rng(seed)
    err = {h: [] for h in HORIZONS}
    pers = {h: [] for h in HORIZONS}
    occ_err = {h: [] for h in HORIZONS}
    pow_rel, rew_pred, rew_true, comf_pred, comf_true = [], [], [], [], []
    for ep in episodes:
        hi = ep.length - horizon - 2
        if hi <= HIST + 2:
            continue
        t0 = torch.as_tensor(rng.integers(HIST + 2, hi, starts_per_ep))
        ro = Rollout(model, norm, ep)
        state = ro.init_state(t0)
        last_true = ep.temp[t0 - 1]
        pe, pt = [], []
        for i in range(horizon):
            t = t0 + i
            state, out, new_temp = ro.step(state, t, ep.zact[t], ep.aact[t], ep.hact[t])
            if i + 1 in err:
                e = (new_temp - ep.temp[t]).abs()
                err[i + 1].append(e.mean().item())
                pers[i + 1].append((last_true - ep.temp[t]).abs().mean().item())
                occ = ep.occ[t] > 0
                if occ.any():
                    occ_err[i + 1].append(e[occ].mean().item())
            pe.append(out["power"])
            pt.append(ep.power[t])
            rew_pred.append(reward_from(ep, t, new_temp, out["power"]))
            rew_true.append(ep.reward[t])
            comf_pred.append(comfort_term(ep, t, new_temp))
            comf_true.append(comfort_term(ep, t, ep.temp[t]))
        #Lorenzo Giannuzzo: power error is aggregated over all steps, since
        #per-window ratios explode on windows where the plant is off.
        pow_rel.append((torch.stack(pe).reshape(-1, 2), torch.stack(pt).reshape(-1, 2)))
    model.train()
    rp, rt = torch.cat(rew_pred), torch.cat(rew_true)
    pe = torch.cat([a for a, _ in pow_rel])
    pt = torch.cat([b for _, b in pow_rel])
    cp, ct = torch.cat(comf_pred), torch.cat(comf_true)
    return {
        "comfort_sum_pred_vs_true": [float(cp.sum()), float(ct.sum())],
        "comfort_corr": float(np.corrcoef(cp.numpy(), ct.numpy())[0, 1]),
        "power_bias_elec_gas": (pe.sum(0) / pt.sum(0).clamp_min(1e-9)).tolist(),
        "temp_mae": {h: float(np.mean(v)) for h, v in err.items()},
        "temp_mae_occupied": {h: float(np.mean(v)) if v else None for h, v in occ_err.items()},
        "persistence_mae": {h: float(np.mean(v)) for h, v in pers.items()},
        "power_rel_err_elec_gas": ((pe - pt).abs().sum(0) / pt.abs().sum(0).clamp_min(1e-9)).tolist(),
        "reward_sum_pred_vs_true": [float(rp.sum()), float(rt.sum())],
        "reward_step_mae": float((rp - rt).abs().mean()),
        "reward_corr": float(np.corrcoef(rp.numpy(), rt.numpy())[0, 1]),
    }


def reward_from(ep, t, temp, power, comfort_scale=1.0):
    """Competition reward of step t rebuilt from temperature and power.

    ``comfort_scale`` multiplies the comfort term; 1.0 is the exact reward.
    """
    rm = ep.reward_meta
    k = ep.schema.num_ahus
    elec_w = power[..., 0] * 80000.0 * k
    gas_w = power[..., 1] * 205000.0 * k
    c = ep.coef[t]
    ne = (elec_w * c[..., 0] + gas_w * c[..., 1]) / (rm["max_electricity_rate"] * c[..., 0] + rm["max_natural_gas_rate"] * c[..., 1])
    nc = (elec_w * c[..., 2] + gas_w * c[..., 3]) / (rm["max_electricity_rate"] * c[..., 2] + rm["max_natural_gas_rate"] * c[..., 3])
    return comfort_scale * comfort_term(ep, t, temp) - rm["energy_cost_weight"] * ne - rm["carbon_emission_weight"] * nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, nargs="+")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", action="store_true",
                    help="train on a bootstrap resample of the episodes (ensemble members)")
    ap.add_argument("--zone-boost", type=float, default=3.0)
    ap.add_argument("--init", default=None,
                    help="fine-tune from this checkpoint, keeping its normalizer")
    ap.add_argument("--emphasis", nargs="*", default=[],
                    help="directories sampled with --emphasis-share probability (e.g. on-policy data)")
    ap.add_argument("--emphasis-share", type=float, default=0.5)
    ap.add_argument("--lengths", type=int, nargs="+", default=None,
                    help="unroll-length curriculum in equal shares (default 1, 6, 12, 24 schedule)")
    ap.add_argument("--w-comfort", type=float, default=1.0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = [e for d in args.train for e in load_dir(Path(d))]
    focus = [e for d in args.emphasis for e in load_dir(Path(d))]
    test = load_dir(Path(args.test))
    #Lorenzo Giannuzzo: the normalizer is fitted on all training episodes
    #before resampling, so that every ensemble member shares one state scale.
    #When fine-tuning, the checkpoint's normalizer is kept, so that a policy
    #trained on the previous world model sees inputs on the same scale.
    init = torch.load(args.init, weights_only=False) if args.init else None
    norm = Normalizer.from_state_dict(init["norm"]) if init else Normalizer.fit(train + focus)
    rng = np.random.default_rng(args.seed)
    if args.bootstrap:
        train = [train[i] for i in rng.integers(0, len(train), len(train))]
    print(f"train episodes {len(train)} + emphasis {len(focus)} | test {len(test)}", flush=True)

    model = WorldModel(d=args.d, layers=args.layers)
    if init:
        model.load_state_dict(init["model"])
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    log = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        #Lorenzo Giannuzzo: curriculum on the unroll length, from one-step
        #prediction to 24-step (2 h) closed-loop rollouts.
        frac = step / args.steps
        if args.lengths:
            length = args.lengths[min(int((step - 1) / args.steps * len(args.lengths)), len(args.lengths) - 1)]
        else:
            length = 1 if frac < 0.15 else 6 if frac < 0.4 else 12 if frac < 0.7 else 24
        pool = focus if focus and rng.random() < args.emphasis_share else train
        ep = pool[rng.integers(len(pool))]
        hi = ep.length - length - 2
        t0 = torch.as_tensor(rng.integers(HIST + 2, hi, args.batch))
        loss, parts = unroll_loss(model, norm, ep, t0, length,
                                  w_comfort=args.w_comfort, zone_boost=args.zone_boost)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step} L={length} loss {loss.item():.4f} " + " ".join(f"{k}={v:.4f}" for k, v in parts.items())
                  + f" | {time.time() - t_start:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate(model, norm, test)
            res["step"] = step
            log.append(res)
            print(json.dumps(res), flush=True)
            torch.save({"model": model.state_dict(), "norm": norm.state_dict(),
                        "args": vars(args)}, out / "world_model.pt")
            (out / "eval_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
