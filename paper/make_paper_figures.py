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


def scores():
    d = json.loads(RES.read_text())
    out = collections.defaultdict(dict)
    for r in d:
        if r["steps"] != 4032:
            continue
        name = "pid_water" if r["policy"] == "pid_water" else Path(r["weights"]).stem
        scen = r["scenario"].split("training_scenarios/")[-1]
        out[name][scen] = abs(r["mean_reward"])
    return out


def fig_progression(s):
    #Lorenzo Giannuzzo: local 14-day score of each stage of the method next to
    #its Kaggle public leaderboard score, where one was submitted.
    stages = [("pid_water", "PID\nBaseline", None),
              ("abl_A10", "Entity-\nToken\nPolicy", 0.019),
              ("v10_base_f03", "+ Memory\nWorld Model,\nFilter", 0.018),
              ("v11_base_f03", "+ Calibrated\nWorld\nModel", None),
              ("v11_long_fit3", "+ 12-24 h\nHorizon, Filter\nIn Training", 0.013),
              ("v14_new2wm", "+ 48 h World\nModel, Two\nWorld Models", None)]
    local = [np.mean(list(s[k].values())) for k, _, _ in stages]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(5.2, 2.45))
    w = 0.38
    b1 = ax.bar(x - w / 2, local, w, color=C_NEW, label="Local, 30 Scenarios, 14 Days")
    kag = [k for _, _, k in stages]
    b2 = ax.bar(x[[i for i, k in enumerate(kag) if k]] + w / 2, [k for k in kag if k], w, color=C_KAGGLE,
                label="Kaggle Public Leaderboard")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008, f"{b.get_height():.4f}".rstrip("0"),
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [lab for _, lab, _ in stages], fontsize=7)
    ax.set_ylabel("Mean Absolute Step Reward [-]")
    ax.set_ylim(0, 0.068)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    legend_top(ax, ncol=2, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "progression.pdf", bbox_inches="tight")
    plt.close(fig)
    return dict(zip([k for k, _, _ in stages], local))


def fig_scenarios(s):
    a, b = s["v10_base_f03"], s["v14_new2wm"]
    fig, ax = plt.subplots(figsize=(2.75, 2.65))
    for c, name in CLIMATES.items():
        keys = [k for k in a if f"/{c}_" in k]
        ax.scatter([a[k] for k in keys], [b[k] for k in keys], s=18, color=COLORS[c], label=name,
                   edgecolor="black", linewidth=0.3, zorder=3)
    lim = (0.0015, 0.12)
    ax.plot(lim, lim, color="black", lw=0.8, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Memory World Model, Filter [-]")
    ax.set_ylabel("Final Policy [-]")
    ax.grid(lw=0.4, alpha=0.5, which="major")
    ax.set_axisbelow(True)
    legend_top(ax, ncol=3, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "scenarios.pdf", bbox_inches="tight")
    plt.close(fig)
    return sum(b[k] < a[k] for k in a), len(a)


def fig_examples():
    #Lorenzo Giannuzzo: first week of a subarctic and of a mild winter scenario,
    #PID baseline and the same two policies, shared legend above the two columns.
    names = {"pid_water": ("PID With Water Reset", C_KAGGLE),
             "v10_base_f03": ("Memory World Model, Filter", C_OLD),
             "v14_new2wm": ("Final Policy", C_NEW)}
    cases = [("building_train_1__fairbanks_1", "Fairbanks 1, Building 1"),
             ("building_train_2__denver_8", "Denver 8, Building 2")]
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 2.75), sharex="col",
                             gridspec_kw={"height_ratios": [0.8, 1.35, 0.95]})
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
        axes[2, c].set_xlabel("Hour Of The Day [h] & Day Of The Week")
        axes[2, c].set_xlim(days[0], days[-1])
        names_d = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        ticks = np.arange(24, days[-1] + 1, 12)
        axes[2, c].set_xticks(ticks, [f"0\n{names_d[int(h // 24)]}" if h % 24 == 0 else "12\n" for h in ticks],
                              fontsize=6.5)
    axes[0, 0].set_ylabel("Outdoor\nAir [°C]", fontsize=7.5)
    axes[1, 0].set_ylabel("Zone Temperature,\n10th Percentile [°C]", fontsize=7.5)
    axes[2, 0].set_ylabel("Natural Gas\nPower [kW]", fontsize=7.5)
    fig.align_ylabels(axes[:, 0])
    for ax in axes.flat:
        ax.grid(lw=0.4, alpha=0.5)
        ax.set_axisbelow(True)
    labels = [lab for lab, _ in names.values()] + ["Median Heating Setpoint"]
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    legend_top(fig, handles + [hs], labels, ncol=4, y=0.92, fig=True)
    fig.savefig(OUT / "examples.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    s = scores()
    print("progression", fig_progression(s))
    print("scenarios won", fig_scenarios(s))
    print("examples", fig_examples())
