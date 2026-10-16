"""Train and evaluate the entity-token world model.

Usage:
    python -m sbwm.train_wm --train wm_data/episodes_train --test wm_data/episodes_test \
        --out wm_runs/v1 --steps 6000
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


def unroll_loss(model, norm, ep, t0, length, w_power=10.0):
    ro = Rollout(model, norm, ep)
    state = ro.init_state(t0)
    total, parts = 0.0, {"temp": 0.0, "zone": 0.0, "ahu": 0.0, "hws": 0.0, "power": 0.0}
    for i in range(length):
        t = t0 + i
        state, out, new_temp = ro.step(state, t, ep.zact[t], ep.aact[t], ep.hact[t])
        l_temp = (((new_temp - ep.temp[t]) / norm.dtemp_std) ** 2).mean()
        zt = ((ep.zobs[t + 1] - norm.z_mean) / norm.z_std)[..., list(ZONE_DYN)]
        at = ((ep.aobs[t + 1] - norm.a_mean) / norm.a_std)[..., list(AHU_DYN)]
        ht = (ep.hobs[t + 1] - norm.h_mean) / norm.h_std
        l_zone = ((out["zone_dyn"] - zt) ** 2).mean()
        l_ahu = ((out["ahu_dyn"] - at) ** 2).mean()
        l_hws = ((out["hws"] - ht) ** 2).mean()
        l_pow = ((out["power"] - ep.power[t]) ** 2).mean()
        loss = l_temp + 0.3 * (l_zone + l_ahu + l_hws) + w_power * l_pow
        total = total + loss / length
        for k, v in zip(parts, (l_temp, l_zone, l_ahu, l_hws, l_pow)):
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
    pow_rel, rew_pred, rew_true = [], [], []
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
        pe, pt = torch.stack(pe), torch.stack(pt)
        pow_rel.append(((pe - pt).abs().sum(0) / pt.abs().sum(0).clamp_min(1e-6)).mean(0))
    model.train()
    rp, rt = torch.cat(rew_pred), torch.cat(rew_true)
    return {
        "temp_mae": {h: float(np.mean(v)) for h, v in err.items()},
        "temp_mae_occupied": {h: float(np.mean(v)) if v else None for h, v in occ_err.items()},
        "persistence_mae": {h: float(np.mean(v)) for h, v in pers.items()},
        "power_rel_err_elec_gas": torch.stack(pow_rel).mean(0).tolist(),
        "reward_sum_pred_vs_true": [float(rp.sum()), float(rt.sum())],
        "reward_step_mae": float((rp - rt).abs().mean()),
        "reward_corr": float(np.corrcoef(rp.numpy(), rt.numpy())[0, 1]),
    }


def reward_from(ep, t, temp, power):
    """Competition reward of step t rebuilt from temperature and power."""
    rm = ep.reward_meta
    w = rm["productivity_deviation_ref"]
    dev = torch.clamp(torch.maximum(temp - ep.csp[t], ep.hsp[t] - temp), min=0.0)
    shortfall = torch.where(dev <= w, dev / w, 1 + ((dev - w) / w) ** 2)
    occ = ep.occ[t]
    tot = occ.sum(-1)
    maxp, minp = rm["max_productivity_personhour_usd"], rm["min_productivity_personhour_usd"]
    act = (maxp * (1 - shortfall) * occ).sum(-1)
    regret = torch.where(tot > 0, (act - minp * tot) / ((maxp - minp) * tot).clamp_min(1e-9) - 1, 0.0)
    k = ep.schema.num_ahus
    elec_w = power[..., 0] * 80000.0 * k
    gas_w = power[..., 1] * 205000.0 * k
    c = ep.coef[t]
    ne = (elec_w * c[..., 0] + gas_w * c[..., 1]) / (rm["max_electricity_rate"] * c[..., 0] + rm["max_natural_gas_rate"] * c[..., 1])
    nc = (elec_w * c[..., 2] + gas_w * c[..., 3]) / (rm["max_electricity_rate"] * c[..., 2] + rm["max_natural_gas_rate"] * c[..., 3])
    return rm["productivity_weight"] * regret - rm["energy_cost_weight"] * ne - rm["carbon_emission_weight"] * nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = load_dir(Path(args.train))
    test = load_dir(Path(args.test))
    norm = Normalizer.fit(train)
    print(f"train episodes {len(train)} steps {sum(e.length for e in train)} | test {len(test)}", flush=True)

    model = WorldModel(d=args.d, layers=args.layers)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    log = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        #Lorenzo Giannuzzo: curriculum on the unroll length, from one-step
        #prediction to 24-step (2 h) closed-loop rollouts.
        frac = step / args.steps
        length = 1 if frac < 0.15 else 6 if frac < 0.4 else 12 if frac < 0.7 else 24
        ep = train[rng.integers(len(train))]
        hi = ep.length - length - 2
        t0 = torch.as_tensor(rng.integers(HIST + 2, hi, args.batch))
        loss, parts = unroll_loss(model, norm, ep, t0, length)
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
