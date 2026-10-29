"""Detailed view of how a controller drives a building, step by step.

Each controller is run in the simulator on an official scenario from its
configured start date (the same window as ``sbwm.eval_policy``), and every
step is recorded: zone temperatures against their setpoints, zone commands,
air-handler and hot-water readings, power and the three reward terms. Two
kinds of figures are drawn from the recordings:

* a dashboard per controller and scenario (outdoor air, zone temperatures,
  per-zone deviation from the comfort band, reheat and damper commands,
  supply air and water temperatures, power, reward terms);
* a comparison per scenario across controllers (mean and worst zone, power,
  cumulative reward).

Recordings are kept in results/inspect/recordings (finished ones are reused;
``sbwm.eval_policy --record`` writes them during the evaluation, so a policy
that has been evaluated needs no extra simulation),
dashboards in results/inspect/<policy>/ and comparisons in
results/inspect/compare/.

Usage (the default scenarios are the standard set below):
    python -m sbwm.inspect_control --policies pid_water policies/transformer_v6.pt policies/abl_A10.pt \
        --steps 2016 --workers 2
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np

OUT = Path("results/inspect")
REC = OUT / "recordings"
#Lorenzo Giannuzzo: the standard set covers the hardest official scenarios
#(Fairbanks and Denver in winter, Phoenix in summer) and a mild one.
STANDARD = ["building_train_1/fairbanks_1", "building_train_2/fairbanks_2", "building_train_1/phoenix_5",
            "building_train_2/denver_8", "building_train_8/atlanta_1"]
NAMES = {"pid_water": "PID With Water Reset", "transformer_v6": "Transformer v6 (Comfort Correction 1.8)",
         "abl_A10": "Transformer A10 (No Comfort Correction)", "v10_base": "Transformer On World Model v10",
         "v10_base10": "Transformer On World Model v10 (No Comfort Correction)",
         "v10_long": "Transformer On World Model v10 (24 h Horizon)",
         "v10_smooth2": "Transformer On World Model v10 (Smoothness 0.002)",
         "v10_smooth5": "Transformer On World Model v10 (Smoothness 0.005)"}


def label(policy: str) -> str:
    stem = Path(policy).stem
    return NAMES.get(stem, stem.replace("_", " ").title())


class Recorder:
    """Per-step record of one simulated run, in the layout the figures read.

    Used by ``record`` below and by ``sbwm.eval_policy --record``, so that an
    evaluation leaves its trajectories behind instead of only their mean.
    """

    KEYS = ("temp", "hsp", "csp", "occ", "zobs", "aobs", "hobs", "elec_kw", "gas_kw",
            "comfort", "energy", "carbon", "reward", "hour")

    def __init__(self, schema, start: str):
        self.s, self.start = schema, start
        self.rec = {k: [] for k in self.KEYS}

    @staticmethod
    def _take(o, idx):
        idx = np.asarray(idx)
        v = np.full(idx.shape, np.nan, np.float32)
        v[idx >= 0] = o[idx[idx >= 0]]
        return v

    def add(self, obs, info, resp, reward):
        s, rec = self.s, self.rec
        zri = info.zone_reward_infos
        rec["temp"].append([zri[z].zone_air_temperature - 273.15 for z in s.zone_rooms])
        rec["hsp"].append([zri[z].heating_setpoint_temperature - 273.15 for z in s.zone_rooms])
        rec["csp"].append([zri[z].cooling_setpoint_temperature - 273.15 for z in s.zone_rooms])
        rec["occ"].append([zri[z].average_occupancy for z in s.zone_rooms])
        rec["zobs"].append(self._take(obs, s.zone_obs_idx))
        rec["aobs"].append(self._take(obs, s.ahu_obs_idx))
        rec["hobs"].append(self._take(obs, s.hws_obs_idx))
        ahu, blr = info.air_handler_reward_infos, info.boiler_reward_infos
        rec["elec_kw"].append((sum(ahu[k].blower_electrical_energy_rate + abs(ahu[k].air_conditioning_electrical_energy_rate)
                                   for k in ahu) + sum(blr[k].pump_electrical_energy_rate for k in blr)) / 1000.0)
        rec["gas_kw"].append(sum(blr[k].natural_gas_heating_energy_rate for k in blr) / 1000.0)
        rec["comfort"].append(0.6 * resp.normalized_productivity_regret)
        rec["energy"].append(-0.2 * resp.normalized_energy_cost)
        rec["carbon"].append(-0.2 * resp.normalized_carbon_emission)
        rec["reward"].append(reward)
        rec["hour"].append(len(rec["hour"]) / 12.0)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, start=self.start, zone_ahu=self.s.zone_ahu,
                            **{k: np.asarray(v, np.float32) for k, v in self.rec.items()})
        tmp.replace(path)


def recording_path(scenario, policy: str) -> Path:
    """results/inspect/recordings/<building>__<station>__<policy>.npz"""
    parts = Path(scenario).parts[-2:]
    return REC / f"{parts[0]}__{parts[1]}__{Path(policy).stem}.npz"


def record(job: dict) -> str:
    import torch

    from sbwm.collect import _load_env
    from sbwm.schema import BuildingSchema

    torch.set_num_threads(1)
    scen = Path("smart_buildings/training_scenarios") / job["scenario"]
    out = recording_path(job["scenario"], job["policy"])
    if out.exists():
        return f"skip {out.name}"
    start = re.search(r"^start_timestamp = '(.*)'$", (scen / "config.gin").read_text(), re.M).group(1)
    env, meta = _load_env(scen, start)
    s = BuildingSchema.from_meta(meta)
    if job["policy"].endswith(".pt"):
        from sbwm.policy import DPCPolicy

        pol = DPCPolicy(meta, env.action_spec(), weights=job["policy"])
    else:
        from smart_buildings.evaluate import POLICY_MAP

        pol = POLICY_MAP[job["policy"]](meta, env.action_spec())
    cap = {}
    original = env.reward_function.compute_reward

    def hook(info):
        resp = original(info)
        cap["info"], cap["resp"] = info, resp
        return resp

    env.reward_function.compute_reward = hook
    obs, _ = env.reset()
    rec = Recorder(s, start)
    for _ in range(job["steps"]):
        a = pol.get_action(obs)
        obs, r, term, trunc, _ = env.step({
            "continuous_action": np.asarray(a["continuous_action"], np.float32),
            "discrete_action": np.asarray(a["discrete_action"], np.int32)})
        rec.add(obs, cap["info"], cap["resp"], r)
        if term or trunc:
            break
    rec.save(out)
    return f"done {out.name}: mean reward {np.mean(rec.rec['reward']):.4f}"


# --------------------------------------------------------------------------- figures

def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e4e3df", "grid.linewidth": 0.7, "axes.axisbelow": True,
        "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False,
        "legend.framealpha": 1.0, "legend.fontsize": 8, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.formatter.use_locale": False,
    })
    return plt


#Lorenzo Giannuzzo: fixed categorical order (blue, orange, aqua, violet) so a
#controller keeps its color in every figure; grays for context.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8f5fd0", "#c9a227"]
INK, MUTED, BAND, OCC = "#0b0b0b", "#b5b3ad", "#dcebfb", "#f3f2ee"


def _legend(ax, ncol):
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, borderpad=0.4,
                    handlelength=2.2, columnspacing=1.2)
    leg.get_frame().set_linewidth(0.8)


def _occupied(ax, d, days):
    occ = d["occ"].sum(1) > 0
    edges = np.flatnonzero(np.diff(np.r_[0, occ.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(days[a], days[min(b, len(days) - 1)], color=OCC, lw=0, zorder=0)


def _pct(x):
    return (x + 1.0) * 50.0  # commands in [-1, 1] mapped to [0, 100] %


def dashboard(plt, d, title, path):
    from matplotlib.colors import TwoSlopeNorm

    days = d["hour"] / 24.0
    temp, hsp, csp = d["temp"], d["hsp"], d["csp"]
    dev = np.where(temp < hsp, temp - hsp, np.where(temp > csp, temp - csp, 0.0))
    occupied = d["occ"] > 0
    fig, axes = plt.subplots(9, 1, figsize=(11, 22), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6, 1.4, 1.2, 1.2, 1, 1, 1, 1]})
    ax = axes[0]
    oat = np.nanmean(d["aobs"][:, :, 14], 1) - 273.15
    ax.plot(days, oat, color=COLORS[0], lw=1.5)
    ax.set_ylabel("Outdoor Air [°C]")
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8)

    ax = axes[1]
    _occupied(ax, d, days)
    ax.fill_between(days, np.median(hsp, 1), np.median(csp, 1), color=BAND, lw=0, label="Comfort Band (Median Zone)")
    for v in range(temp.shape[1]):
        ax.plot(days, temp[:, v], color=MUTED, lw=0.5, alpha=0.7)
    ax.plot([], [], color=MUTED, lw=1, label="Single Zones")
    ax.plot(days, temp.mean(1), color=COLORS[0], lw=1.8, label="Mean Of Zones")
    ax.plot(days, temp.min(1), color=COLORS[1], lw=1.0, label="Coldest Zone")
    ax.set_ylabel("Zone Temperature [°C]")
    _legend(ax, 4)

    ax = axes[2]
    lim = max(0.5, float(np.nanpercentile(np.abs(dev), 99.5)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    shown = np.where(occupied, dev, np.nan)
    im = ax.imshow(shown.T, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest",
                   extent=[days[0], days[-1], temp.shape[1] - 0.5, -0.5])
    ax.set_ylabel("Zone [-]")
    ax.grid(False)
    cb = fig.colorbar(im, cax=ax.inset_axes([1.01, 0.0, 0.012, 1.0]))
    cb.set_label("Deviation When Occupied [K]")
    cb.outline.set_edgecolor("black")

    for ax, col, name in ((axes[3], 4, "Reheat Valve Command [%]"), (axes[4], 3, "Damper Command [%]")):
        im = ax.imshow(_pct(d["zobs"][:, :, col]).T, aspect="auto", cmap="Blues", vmin=0, vmax=100,
                       interpolation="nearest", extent=[days[0], days[-1], temp.shape[1] - 0.5, -0.5])
        ax.set_ylabel("Zone [-]")
        ax.grid(False)
        #Lorenzo Giannuzzo: colorbars sit outside the axes, so that the heatmaps
        #keep the same width and time axis as the line panels.
        cb = fig.colorbar(im, cax=ax.inset_axes([1.01, 0.0, 0.012, 1.0]))
        cb.set_label(name)
        cb.outline.set_edgecolor("black")

    ax = axes[5]
    k = d["aobs"].shape[1]
    for j in range(k):
        c = COLORS[j % len(COLORS)]
        ax.plot(days, d["aobs"][:, j, 0] - 273.15, color=c, lw=1.2, label=f"Air Handler {j + 1}, Measured")
        ax.plot(days, d["aobs"][:, j, 1] - 273.15, color=c, lw=0.8, ls="--", label=f"Air Handler {j + 1}, Setpoint")
    ax.set_ylabel("Supply Air [°C]")
    _legend(ax, min(4, 2 * k))

    ax = axes[6]
    for j in range(k):
        ax.plot(days, d["aobs"][:, j, 4], color=COLORS[j % len(COLORS)], lw=1.2, label=f"Air Handler {j + 1}")
    ax.set_ylabel("Supply Air Flow [m³/s]")
    _legend(ax, min(4, k))

    ax = axes[7]
    ax.plot(days, d["hobs"][:, 0] - 273.15, color=COLORS[0], lw=1.2, label="Measured")
    ax.plot(days, d["hobs"][:, 1] - 273.15, color=COLORS[1], lw=0.8, ls="--", label="Setpoint")
    ax.set_ylabel("Supply Water [°C]")
    _legend(ax, 2)

    ax = axes[8]
    ax.plot(days, d["elec_kw"], color=COLORS[0], lw=1.0, label="Electricity")
    ax.plot(days, d["gas_kw"], color=COLORS[1], lw=1.0, label="Natural Gas")
    ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Time Since Start [Day]")
    _legend(ax, 2)
    fig.align_ylabels(axes)
    fig.savefig(path)
    plt.close(fig)


def reward_terms(plt, runs, title, path):
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.4 * len(runs) + 0.6), sharex=True, squeeze=False)
    for ax, (name, d) in zip(axes[:, 0], runs.items()):
        days = d["hour"] / 24.0
        for c, key, lab in ((COLORS[1], "comfort", "Comfort"), (COLORS[0], "energy", "Energy Cost"),
                            (COLORS[2], "carbon", "Carbon")):
            ax.plot(days, -d[key], color=c, lw=0.9, label=f"{lab} ({-d[key].mean():.4f})")
        ax.set_ylabel("Loss Per Step [-]")
        ax.set_title(f"{name}, Score {abs(d['reward'].mean()):.4f}", loc="left", fontsize=9, color=INK, pad=22)
        _legend(ax, 3)
    axes[-1, 0].set_xlabel("Time Since Start [Day]")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path)
    plt.close(fig)


def comparison(plt, runs, title, path):
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    first = next(iter(runs.values()))
    days = first["hour"] / 24.0
    _occupied(axes[0], first, days)
    axes[0].fill_between(days, np.median(first["hsp"], 1), np.median(first["csp"], 1), color=BAND, lw=0,
                         label="Comfort Band")
    for i, (name, d) in enumerate(runs.items()):
        c = COLORS[i % len(COLORS)]
        dd = d["hour"] / 24.0
        temp, hsp, csp = d["temp"], d["hsp"], d["csp"]
        dev = np.where(temp < hsp, hsp - temp, np.where(temp > csp, temp - csp, 0.0))
        dev = np.where(d["occ"] > 0, dev, 0.0)
        axes[0].plot(dd, temp.mean(1), color=c, lw=1.3, label=name)
        axes[1].plot(dd, dev.max(1), color=c, lw=1.0, label=name)
        axes[2].plot(dd, d["elec_kw"], color=c, lw=0.9, label=name)
        axes[3].plot(dd, d["gas_kw"], color=c, lw=0.9, label=name)
        axes[4].plot(dd, np.cumsum(-d["reward"]), color=c, lw=1.6, label=f"{name} ({abs(d['reward'].mean()):.4f})")
    axes[0].set_ylabel("Mean Zone Temperature [°C]")
    axes[1].set_ylabel("Worst Occupied Deviation [K]")
    axes[2].set_ylabel("Electricity [kW]")
    axes[3].set_ylabel("Natural Gas [kW]")
    axes[4].set_ylabel("Cumulative Loss [-]")
    axes[4].set_xlabel("Time Since Start [Day]")
    for ax in axes:
        _legend(ax, min(4, len(runs) + (ax is axes[0])))
    axes[0].set_title(title, loc="left", color=INK, fontsize=11, pad=34)
    fig.align_ylabels(axes)
    fig.savefig(path)
    plt.close(fig)


def figures(scenarios, policies):
    """Dashboards in results/inspect/<policy>/, comparisons in results/inspect/compare/."""
    plt = _style()
    for sc in scenarios:
        runs = {}
        tag = sc.replace("/", "__")
        nice = sc.replace("building_train_", "Building ").replace("/", ", ").replace("_", " ").title()
        for p in policies:
            f = recording_path(sc, p)
            if not f.exists():
                continue
            d = dict(np.load(f))
            runs[label(p)] = d
            folder = OUT / Path(p).stem
            folder.mkdir(parents=True, exist_ok=True)
            dashboard(plt, d, f"{label(p)}, {nice}, From {d['start']}", folder / f"dashboard__{tag}.png")
        if runs:
            (OUT / "compare").mkdir(parents=True, exist_ok=True)
            comparison(plt, runs, f"Controllers Compared, {nice}", OUT / "compare" / f"{tag}.png")
            reward_terms(plt, runs, f"Reward Terms Per Step, {nice}", OUT / "compare" / f"terms__{tag}.png")
            print("figures for", sc, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=STANDARD)
    ap.add_argument("--policies", nargs="+", default=["pid_water", "policies/transformer_v6.pt"])
    ap.add_argument("--steps", type=int, default=2016)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    if not args.figures_only:
        jobs = [{"scenario": s, "policy": p, "steps": args.steps} for s in args.scenarios for p in args.policies]
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            for msg in pool.imap_unordered(record, jobs):
                print(msg, flush=True)
    figures(args.scenarios, args.policies)


if __name__ == "__main__":
    main()
