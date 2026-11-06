"""Figures of the world-model error with the policy in control.

For each scenario the policy runs in the true simulator for a few days (the
truth); from start points every 6 hours the world model then predicts the next
24 hours, either replaying the actions the policy really took (Replay) or with
the policy acting inside the world model (Closed Loop, as in the DPC training
and in the planner). The figure shows the truth with the predicted 24-hour
segments on top, for the 10th-percentile zone temperature and the heating and
electric power, and the mean absolute error of the zone temperatures against
the prediction horizon.

Usage:
    python -m sbwm.wm_error_plots --weights policies/v14_new2wm.pt --wm world_models/wm_v14.pt \
        --scenarios building_train_1/fairbanks_1 building_train_2/denver_8 building_train_1/phoenix_5 \
        --steps 2016 --out results/figures/wm_error.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

COL = {"truth": "black", "replay": "#9aa5b1", "closed": "#1F4E79"}


@torch.no_grad()
def predict(net, model, norm, ep, starts, horizon, closed: bool):
    """Predicted zone temperatures (B, H, V) and power (B, H, 2) [W] from each start."""
    from sbwm.train_dpc import policy_step
    from sbwm.world_model import Rollout

    ro = Rollout(model, norm, ep)
    t0 = torch.as_tensor(starts, dtype=torch.long)
    state = ro.init_state(t0)
    temps, powers, comps = [], [], []
    k = ep.schema.num_ahus
    for i in range(horizon):
        t = t0 + i
        if closed:
            carry = {}
            z, a, h, _ = policy_step(net, ro, state, ep, t, norm, hard=True, carry=carry)
        else:
            z, a, h, carry = ep.zact[t], ep.aact[t], ep.hact[t], {}
        state, out, new_temp = ro.step(state, t, z, a, h)
        state.update(carry)
        temps.append(new_temp)
        p = out["power"]
        powers.append(torch.stack([p[..., 0] * 80000.0 * k, p[..., 1] * 205000.0 * k], -1))
        if "comp" in out:
            c = out["comp"]
            comps.append(torch.stack([c["reheat"].sum(-1) * 205000.0, c["fan"].sum(-1) * 80000.0,
                                      c["cool"].sum(-1) * 80000.0, c["pump"] * 80000.0, c["loss"] * 205000.0,
                                      c["tank"] * 205000.0], -1))
    predict.last_comp = torch.stack(comps, 1).numpy() if comps else None
    return torch.stack(temps, 1).numpy(), torch.stack(powers, 1).numpy()


COMP_NAMES = ("reheat", "fan", "cool", "pump", "loss", "tank")


def true_components(ep):
    """Simulated components (T, 6) [W], or None for episodes collected without them."""
    if ep.comp is None:
        return None
    c = ep.comp
    return np.stack([c["reheat"].sum(-1).numpy() * 205000.0, c["fan"].sum(-1).numpy() * 80000.0,
                     c["cool"].sum(-1).numpy() * 80000.0, c["pump"].numpy() * 80000.0,
                     c["loss"].numpy() * 205000.0, c["tank"].numpy() * 205000.0], -1)


def hour_of_day(ep) -> np.ndarray:
    g = ep.gobs.numpy()
    return (np.mod(np.arctan2(g[:, 3], g[:, 4]), 2 * np.pi) * 24 / (2 * np.pi)).astype(int) % 24


def true_power(ep):
    from sbwm.wm_data import ELEC_SCALE_PER_AHU, GAS_SCALE_PER_AHU

    k = ep.schema.num_ahus
    p = ep.power.numpy()
    return np.stack([p[:, 0] * ELEC_SCALE_PER_AHU * k, p[:, 1] * GAS_SCALE_PER_AHU * k], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--root", default="smart_buildings/training_scenarios")
    ap.add_argument("--steps", type=int, default=2016)
    ap.add_argument("--every", type=int, default=72, help="steps between prediction starts (72 = 6 h)")
    ap.add_argument("--horizon", type=int, default=288)
    ap.add_argument("--cache", default="results/wm_error_episodes")
    ap.add_argument("--out", default="results/figures/wm_error.png")
    args = ap.parse_args()

    from sbwm.policy import DPCPolicy
    from sbwm.train_dpc import load_models
    from sbwm.wm_data import load_episode
    from sbwm.wm_planner import prerun

    (model,), norm = load_models([args.wm])
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9})
    n = len(args.scenarios)
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.9 * n), gridspec_kw={"width_ratios": [2.2, 2.2, 1]},
                             squeeze=False)
    summary = {}
    fig2, axes2 = plt.subplots(n, 2, figsize=(9, 2.6 * n), squeeze=False)
    report = {}
    for row, sc in enumerate(args.scenarios):
        scen = Path(args.root) / sc
        f = cache / (sc.replace("/", "__") + f"__{Path(args.weights).stem}__{args.steps}.npz")
        if f.exists():
            ep = load_episode(f)
        else:
            start = re.search(r"^start_timestamp = '(.*)'$", (scen / "config.gin").read_text(), re.M).group(1)
            ep, rec = prerun(scen, start, args.weights, args.steps)
            rec_ep_path = f
            _save(rec, scen, start, rec_ep_path)
        ck = torch.load(args.weights, weights_only=False)
        from sbwm.policy import PolicyNet

        net = PolicyNet(**ck["policy_args"])
        net.load_state_dict(ck["policy"])
        net.eval()
        if ck.get("filter_in_training"):
            net.action_filter = ck.get("action_filter")
        T = ep.length
        starts = np.arange(288, T - args.horizon, args.every)
        res = {}
        for mode in ("replay", "closed"):
            res[mode] = predict(net, model, norm, ep, starts, args.horizon, mode == "closed")
        true_t = ep.temp.numpy() - 273.15
        true_p = true_power(ep)
        hours = np.arange(T) / 12.0
        heat = 1 if true_p[:, 1].mean() > 0.2 * true_p[:, 0].mean() else 0
        ax_t, ax_p, ax_e = axes[row]
        ax_t.plot(hours, np.percentile(true_t, 10, axis=1), color=COL["truth"], lw=1.0, label="Simulator (Truth)")
        ax_p.plot(hours, true_p[:, heat] / 1000.0, color=COL["truth"], lw=0.8, label="Simulator (Truth)")
        for mode, lab in (("replay", "World Model, Replayed Actions"), ("closed", "World Model, Policy In The Loop")):
            pt, pp = res[mode]
            for j, s0 in enumerate(starts):
                hh = hours[s0:s0 + args.horizon]
                ax_t.plot(hh, np.percentile(pt[j] - 273.15, 10, axis=1), color=COL[mode], lw=0.8, alpha=0.8,
                          label=lab if j == 0 else None)
                ax_p.plot(hh, pp[j, :, heat] / 1000.0, color=COL[mode], lw=0.6, alpha=0.6,
                          label=lab if j == 0 else None)
            idx = starts[:, None] + np.arange(args.horizon)
            hod = hour_of_day(ep)[idx]
            for col, lab2 in ((0, "Electric"), (1, "Natural Gas")):
                prof_p = np.array([pp[..., col][hod == h].mean() if (hod == h).any() else np.nan for h in range(24)])
                prof_t = np.array([true_p[idx][..., col][hod == h].mean() if (hod == h).any() else np.nan
                                   for h in range(24)])
                ax2 = axes2[row, col]
                if mode == "replay":
                    ax2.plot(range(24), prof_t / 1000.0, color=COL["truth"], lw=1.2, marker="o", ms=2.5,
                             label="Simulator (Truth)")
                ax2.plot(range(24), prof_p / 1000.0, color=COL[mode], lw=1.2, marker="o", ms=2.5,
                         label="World Model, Replayed Actions" if mode == "replay" else "World Model, Policy In The Loop")
                ax2.set_ylabel(f"{sc.split('/')[-1].replace('_', ' ').title()}\n{lab2} Power [kW]" if col == 0
                               else f"{lab2} Power [kW]")
                ax2.set_xticks(range(0, 24, 3))
                ax2.grid(lw=0.4, alpha=0.5)
                if row == n - 1:
                    ax2.set_xlabel("Hour Of The Day [h]")
                report.setdefault(sc, {})[f"{mode}_{lab2.lower().replace(' ', '_')}_ratio"] = float(
                    pp[..., col].sum() / max(true_p[idx][..., col].sum(), 1e-9))
            tc = true_components(ep)
            if mode == "replay" and tc is not None and predict.last_comp is not None:
                pc = predict.last_comp
                report[sc]["components_pred_over_true"] = {
                    nm: float(pc[..., i].sum() / tc[idx][..., i].sum()) if abs(tc[idx][..., i].sum()) > 1e-6 else None
                    for i, nm in enumerate(COMP_NAMES)}
            err = np.abs(pt - ep.temp.numpy()[starts[:, None] + np.arange(args.horizon)]).mean((0, 2))
            ax_e.plot(np.arange(1, args.horizon + 1) / 12.0, err, color=COL[mode], lw=1.2)
            summary[(sc, mode)] = {h: float(err[h * 12 - 1]) for h in (1, 6, 12, 24) if h * 12 <= len(err)}
        name = sc.split("/")[-1].replace("_", " ").title()
        ax_t.set_ylabel(f"{name}\nZone Temperature,\n10th Percentile [°C]")
        ax_p.set_ylabel(("Natural Gas" if heat else "Electric") + " Power [kW]")
        ax_e.set_ylabel("Zone Temperature\nMean Absolute Error [K]")
        for ax in (ax_t, ax_p):
            ax.set_xlim(0, hours[-1])
            ax.set_xticks(np.arange(0, hours[-1] + 1, 24))
            ax.grid(lw=0.4, alpha=0.5)
        ax_e.set_xlim(0, args.horizon / 12)
        ax_e.set_xticks([0, 6, 12, 18, 24])
        ax_e.grid(lw=0.4, alpha=0.5)
        if row == n - 1:
            ax_t.set_xlabel("Time From Episode Start [h]")
            ax_p.set_xlabel("Time From Episode Start [h]")
            ax_e.set_xlabel("Prediction Horizon [h]")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1 - 0.35 / (2.9 * n)), frameon=True,
               fancybox=False, edgecolor="black")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / (2.9 * n)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    for (sc, mode), v in summary.items():
        print(sc, mode, {k: round(x, 2) for k, x in v.items()})
    h2, l2 = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(h2, l2, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1 - 0.35 / (2.6 * n)), frameon=True,
                fancybox=False, edgecolor="black", fontsize=8)
    fig2.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (2.6 * n)))
    fig2.savefig(out.with_name(out.stem + "_profile.png"), dpi=170)
    import json

    for sc in report:
        report[sc]["temp_mae"] = {mode: summary[(sc, mode)] for mode in ("replay", "closed")}
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def _save(rec, scen, start, path):
    import json

    from sbwm.geometry import zone_static_features

    arrays = {k: np.asarray(v) for k, v in rec.rec.items()}
    for k, v in arrays.items():
        if v.dtype == np.float64:
            arrays[k] = v.astype(np.float32)
    arrays["obs"] = np.stack(rec.obs)
    meta = rec.meta
    arrays["zone_static"] = zone_static_features(meta, rec.schema)
    header = {"scenario": str(scen), "start": start, "seconds": 0.0, "schema": rec.schema.to_dict(),
              "meta": {k: meta[k] for k in (
                  "observation_field_names", "morning_start_hour", "evening_start_hour", "comfort_temp_window",
                  "eco_temp_window", "time_zone", "observation_normalization_metadata", "num_floors")},
              "reward": rec.reward_meta}
    np.savez_compressed(path, header=np.array(json.dumps(header, default=str)), **arrays)


if __name__ == "__main__":
    main()
