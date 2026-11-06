"""Figures of the Round 1 paper, built from the evaluation results in results/.

Usage (from the repository root):
    python paper/make_paper_figures.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("paper/figures")
RES = Path("results/official30_14d.json")
REC = Path("results/inspect/recordings")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.formatter.use_locale": False, "pdf.fonttype": 42,
})
CLIMATES = {"atlanta": "Atlanta", "denver": "Denver", "fairbanks": "Fairbanks", "phoenix": "Phoenix",
            "singapore": "Singapore"}
COLORS = {"atlanta": "#1b9e77", "denver": "#d95f02", "fairbanks": "#3b5bdb", "phoenix": "#e7298a",
          "singapore": "#66a61e"}
C_OLD, C_NEW, C_KAGGLE = "#9aa5b1", "#1f4e79", "#c0504d"


def legend_top(ax_or_fig, handles=None, labels=None, ncol=2, y=1.0, fig=False):
    #Lorenzo Giannuzzo: legend centered above the plot, inside a black frame.
    kw = dict(loc="lower center", ncol=ncol, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0)
    if fig:
        leg = ax_or_fig.legend(handles, labels, bbox_to_anchor=(0.5, y), **kw)
    elif handles is None:
        leg = ax_or_fig.legend(bbox_to_anchor=(0.5, y), **kw)
    else:
        leg = ax_or_fig.legend(handles, labels, bbox_to_anchor=(0.5, y), **kw)
    leg.get_frame().set_linewidth(0.8)
    return leg


STAGES = [("pid_water", "PID\nBaseline", None),
          ("abl_A10", "Entity-Token\nPolicy", 0.019),
          ("v10_base_f03", "+ Memory\nWorld Model\n& Filter", 0.018),
          ("v11_base_f03", "+ Calibrated\nWorld Model", None),
          ("v11_long_fit3", "+ 12-24 h Horizon\n& Filter In\nTraining", 0.013),
          ("v14_new2wm", "+ 48 h World\nModel & Two\nWorld Models", None)]
C_COMFORT, C_ENERGY = "#f57c00", "#3a1458"


def scores(key="mean_reward"):
    d = json.loads(RES.read_text())
    out = collections.defaultdict(dict)
    for r in d:
        if r["steps"] != 4032:
            continue
        name = "pid_water" if r["policy"] == "pid_water" else Path(r["weights"]).stem
        scen = r["scenario"].split("training_scenarios/")[-1]
        out[name][scen] = abs(r[key])
    return out


def fig_impact():
    #Lorenzo Giannuzzo: what each stage changes, as the comfort and the energy &
    #carbon terms of the 30-scenario score (top) and as the score of each
    #climate (bottom), with the columns of the two panels aligned.
    tot, com = scores(), scores("comfort")
    keys = [k for k, _, _ in STAGES]
    total = np.array([np.mean(list(tot[k].values())) for k in keys])
    comfort = np.array([np.mean(list(com[k].values())) for k in keys])
    x = np.arange(len(keys))
    fig = plt.figure(figsize=(5.6, 2.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    b1 = ax.bar(x, comfort, 0.56, color=C_COMFORT, edgecolor="black", lw=0.4, label="Comfort")
    b2 = ax.bar(x, total - comfort, 0.56, bottom=comfort, color=C_ENERGY, edgecolor="black", lw=0.4,
                label="Energy & Carbon")
    for i, t in enumerate(total):
        ax.text(x[i], t + 0.0012, f"{t:.4f}", ha="center", va="bottom", fontsize=7)
    kx = [i for i, (_, _, k) in enumerate(STAGES) if k]
    hk = ax.scatter([x[i] + 0.36 for i in kx], [STAGES[i][2] for i in kx], marker="D", s=16, color="#2166ac",
                    edgecolor="black", lw=0.4, zorder=4, label="Kaggle Public Leaderboard")
    for i in kx:
        ax.text(x[i] + 0.36, STAGES[i][2] + 0.0022, f"{STAGES[i][2]:.3f}", ha="center", va="bottom", fontsize=6.5,
                color="#2166ac")
    ax.set_ylabel("Mean Absolute Step\nReward [-]")
    ax.set_ylim(0, 0.068)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_xticks(x, [])
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    legend_top(ax, [b1, b2, hk], ["Comfort", "Energy & Carbon", "Kaggle Public Leaderboard"], ncol=3, y=1.02)
    #Lorenzo Giannuzzo: mean score of the six scenarios of each climate, colored
    #by its ratio to the PID baseline in the same climate.
    axh = fig.add_subplot(gs[1])
    clim = list(CLIMATES)
    m = np.array([[np.mean([v for sc, v in tot[k].items() if f"/{c}_" in sc]) for k in keys] for c in clim])
    ratio = m / m[:, :1]
    #Lorenzo Giannuzzo: the dark part of the magma scale up to a strong orange
    #in the middle, ending on a light orange.
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    mag = plt.get_cmap("magma")
    cmap = LinearSegmentedColormap.from_list("magma_orange", [(0.0, mag(0.0)), (0.25, mag(0.3)),
                                                              (0.5, to_rgb("#f57c00")),
                                                              (1.0, to_rgb("#ffd59e"))])
    im = axh.imshow(ratio, cmap=cmap, aspect="auto", vmin=0.0, vmax=0.6,
                    extent=(-0.5, len(keys) - 0.5, len(clim) - 0.5, -0.5))
    for i in range(len(clim)):
        for j in range(len(keys)):
            r, g, b, _ = cmap(min(ratio[i, j] / 0.6, 1.0))
            axh.text(j, i, f"{m[i, j]:.4f}", ha="center", va="center", fontsize=6.5,
                     color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black")
    axh.set_yticks(range(len(clim)), [CLIMATES[c] for c in clim], fontsize=7.5)
    axh.set_xticks(x, [lab for _, lab, _ in STAGES], fontsize=7)
    axh.tick_params(length=0)
    cb = fig.colorbar(im, cax=axh.inset_axes([1.015, 0.0, 0.02, 1.0]), extend="max")
    cb.set_label("Ratio To PID [-]", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_edgecolor("black")
    fig.savefig(OUT / "impact.pdf", bbox_inches="tight")
    plt.close(fig)
    return dict(zip(keys, np.round(total, 4))), {c: dict(zip(keys, np.round(m[i], 4))) for i, c in enumerate(clim)}


def _diag(name):
    for f in ("wm_diagnosis_v14.json", "wm_diagnosis_v12.json"):
        d = json.loads((Path("results") / f).read_text())
        if name in d:
            return d[name]["closed_loop"]["by_station"]
    raise KeyError(name)


def fig_analysis():
    #Lorenzo Giannuzzo: (a) 14-day score of every training seed of the three
    #recipes run with more than one seed; (b, c) closed-loop calibration of the
    #three world models on the two reward terms, by climate.
    tot = scores()
    mean = {k: np.mean(list(v.values())) for k, v in tot.items()}
    recipes = [("World\nModels\n2 & 1\n12-24 h", ["v14_2wm_s0", "v14_2wm_s1", "v14_2wm_s2"]),
               ("World\nModel\n3\n24-48 h", ["v14_48_s0", "v14_48_s1", "v14_48_s2"]),
               ("World\nModels\n3 & 2\n12-24 h", ["v14_new2wm", "v16_k18_s1"])]
    wms = [("wm_v11", "World Model 1", C_OLD), ("wm_v12", "World Model 2", C_KAGGLE),
           ("wm_v14", "World Model 3", C_NEW)]
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.8), gridspec_kw={"width_ratios": [1.05, 1.2, 1.2]})
    ax = axes[0]
    for i, (lab, ks) in enumerate(recipes):
        v = [mean[k] for k in ks]
        ax.scatter(np.full(len(v), i) + np.linspace(-0.13, 0.13, len(v)), v, s=20, color="white",
                   edgecolor="black", lw=0.7, zorder=3)
    hf = ax.scatter([2 - 0.13], [mean["v14_new2wm"]], s=55, marker="*", color=C_NEW, edgecolor="black", lw=0.4,
                    zorder=4)
    ax.annotate("Final Policy", (2 - 0.13, mean["v14_new2wm"]), (0.6, 0.026), fontsize=6.5,
                arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_xticks(range(len(recipes)), [r for r, _ in recipes], fontsize=6)
    ax.set_xlim(-0.5, len(recipes) - 0.5)
    ax.set_ylim(0, 0.038)
    ax.set_yticks([0, 0.01, 0.02, 0.03])
    ax.set_ylabel("14-Day Score [-]", fontsize=8)
    ax.set_title("(a) Training Seeds", fontsize=8)
    out = {"seeds": {lab.replace("\n", " "): [round(mean[k], 4) for k in ks] for lab, ks in recipes}}
    clim = list(CLIMATES)
    w = 0.26
    for ax, key, title in ((axes[1], "comfort_ratio_24h", "(b) Comfort, Predicted / True"),
                           (axes[2], "energy_ratio_24h", "(c) Energy, Predicted / True")):
        for j, (name, lab, col) in enumerate(wms):
            st = _diag(name)
            v = [st[c][key] for c in clim]
            ax.bar(np.arange(len(clim)) + (j - 1) * w, v, w, color=col, edgecolor="black", lw=0.4, label=lab)
            out[(name, key)] = dict(zip(clim, np.round(v, 2)))
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_xticks(range(len(clim)), [CLIMATES[c] for c in clim], fontsize=6.5, rotation=35, ha="right",
                      rotation_mode="anchor")
        ax.set_ylim(0, 1.7)
        ax.set_yticks([0, 0.5, 1.0, 1.5])
        ax.set_ylabel("Ratio [-]", fontsize=8)
        ax.set_title(title, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        ax.set_axisbelow(True)
    h, l = axes[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=0.6)
    legend_top(fig, h, l, ncol=3, y=0.87, fig=True)
    fig.savefig(OUT / "analysis.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def fig_headroom():
    #Lorenzo Giannuzzo: 7-day score of the final policy, of the oracle with the
    #true simulator and of the same search scored by world models.
    base = {Path(r["scenario"]).name: abs(r["mean_reward"]) for r in json.loads(Path("results/oracle_base.json").read_text())}
    orc = {Path(r["scenario"]).name: abs(r["mean_reward"]) for r in json.loads(Path("results/oracle.json").read_text())}
    pl = {}
    for f in Path("results/wm_planner").glob("*.json"):
        r = json.loads(f.read_text())
        r = r[-1] if isinstance(r, list) else r
        mode, sc = f.stem.split("__")
        pl[(mode, sc)] = abs(r["mean_reward"])
    scen = [("fairbanks_1", "Fairbanks 1"), ("denver_8", "Denver 8"), ("phoenix_5", "Phoenix 5")]
    #Lorenzo Giannuzzo: scores relative to the final policy in the same
    #scenario, so that the three climates share one scale.
    series = [("Oracle With Simulator", "#7b2d8e", [orc[s] / base[s] for s, _ in scen]),
              ("Planner With World Model 3", "#ff8c00", [pl[("naive", s)] / base[s] for s, _ in scen]),
              ("Planner With World Models 1-3", "#2166ac", [pl[("safe", s)] / base[s] for s, _ in scen])]
    fig, ax = plt.subplots(figsize=(5.6, 1.5))
    w = 0.24
    for j, (lab, col, v) in enumerate(series):
        bars = ax.bar(np.arange(len(scen)) + (j - 1) * w, v, w, color=col, edgecolor="black", lw=0.4, label=lab)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center",
                    va="bottom", fontsize=6)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Final Policy")
    ax.set_ylim(0, 1.4)
    ax.set_xticks(range(len(scen)), [n for _, n in scen])
    ax.set_ylabel("Score Relative To\nFinal Policy [-]", fontsize=7.5)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    leg = legend_top(ax, ncol=4, y=1.02)
    for t in leg.get_texts():
        t.set_fontsize(6.5)
    fig.tight_layout()
    fig.savefig(OUT / "headroom.pdf", bbox_inches="tight")
    plt.close(fig)
    return {lab: dict(zip([n for _, n in scen], np.round(v, 4))) for lab, _, v in series}


def fig_examples():
    #Lorenzo Giannuzzo: first week of a subarctic and of a mild winter scenario,
    #PID baseline and the same two policies, shared legend above the two columns.
    names = {"pid_water": ("PID With Water Reset", C_KAGGLE),
             "v10_base_f03": ("Memory World Model & Filter", C_OLD),
             "v14_new2wm": ("Final Policy", C_NEW)}
    cases = [("building_train_1__fairbanks_1", "Fairbanks 1, Building 1"),
             ("building_train_2__denver_8", "Denver 8, Building 2")]
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 4.0), sharex="col",
                             gridspec_kw={"height_ratios": [0.75, 1.35, 1.05]})
    out = {}
    for c, (scen, title) in enumerate(cases):
        recs = {k: np.load(REC / f"{scen}__{k}.npz", allow_pickle=True) for k in names}
        d0 = recs["v14_new2wm"]
        #Lorenzo Giannuzzo: hours since midnight of the start day, so that the
        #ticks fall on real clock times (the recordings start at 06:00).
        start_h = float(str(d0["start"]).split()[1].split(":")[0])
        days = start_h + np.arange(len(d0["reward"])) * 5.0 / 60.0
        axes[0, c].plot(days, np.nanmean(d0["aobs"][:, :, 14], 1) - 273.15, color="black", lw=1.0)
        axes[0, c].set_title(title, fontsize=9)
        handles = []
        for k, (lab, col) in names.items():
            d = recs[k]
            h, = axes[1, c].plot(days, np.percentile(d["temp"], 10, axis=1), color=col, lw=1.0)
            axes[2, c].plot(days, d["gas_kw"], color=col, lw=0.9)
            handles.append(h)
            out[(scen, k)] = round(float(d["reward"].mean()), 4)
        hs, = axes[1, c].plot(days, np.median(d0["hsp"], 1), color="black", lw=0.8, ls="--")
        #Lorenzo Giannuzzo: numbered markers on the behaviors discussed in the
        #text, explained in the caption.
        def mark(ax, x, y, n, dy):
            ax.annotate(str(n), (x, y), (x, y + dy), fontsize=6, ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="black", lw=0.6),
                        arrowprops=dict(arrowstyle="-", lw=0.6), zorder=6)
        if c == 0:
            t10 = np.percentile(recs["v10_base_f03"]["temp"], 10, axis=1)
            i = np.argmin(np.where(days > 144, t10, np.inf))
            mark(axes[1, c], days[i] - 4, t10[i - 48], 1, 12)
            g = d0["gas_kw"]
            j = np.argmax((days > 144) & (g > 50))
            mark(axes[2, c], days[j], g[j], 2, 900)
            gp = recs["pid_water"]["gas_kw"]
            k = np.argmin(np.abs(days - 99))
            mark(axes[2, c], days[k], gp[k], 3, 700)
        else:
            tp = np.percentile(recs["pid_water"]["temp"], 10, axis=1)
            k = np.argmin(np.abs(days - 81))
            mark(axes[1, c], days[k], tp[k], 4, -3.5)
        axes[2, c].set_xlabel("Hour Of The Day [h] & Day Of The Week")
        axes[2, c].set_xlim(days[0], days[-1])
        names_d = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        ticks = np.arange(24, days[-1] + 1, 12)
        axes[2, c].set_xticks(ticks, [f"0\n{names_d[int(h // 24)]}" if h % 24 == 0 else "12\n" for h in ticks],
                              fontsize=6.5)
    axes[0, 0].set_ylabel("Outdoor\nAir [°C]", fontsize=7)
    axes[1, 0].set_ylabel("Zone Air, 10th\nPercentile [°C]", fontsize=7)
    axes[2, 0].set_ylabel("Gas\nPower [kW]", fontsize=7)
    fig.align_ylabels(axes[:, 0])
    for ax in axes.flat:
        ax.grid(lw=0.4, alpha=0.5)
        ax.set_axisbelow(True)
    labels = [lab for lab, _ in names.values()] + ["Median Heating Setpoint"]
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    legend_top(fig, handles + [hs], labels, ncol=4, y=0.945, fig=True)
    fig.savefig(OUT / "examples.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("impact", fig_impact())
    print("analysis", fig_analysis())
    print("headroom", fig_headroom())
    print("examples", fig_examples())
