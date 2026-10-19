"""Draw the result figures (PNG and PDF) into ``evaluations/figures``.

Reads the CSV tables written by ``sbwm.report_tables`` and the trajectories
written by ``sbwm.record_timeseries``.

Usage (from the repository root):
    python -m sbwm.make_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EVAL = Path("evaluations")
OUT = EVAL / "figures"

#Lorenzo Giannuzzo: validated categorical slots (blue, orange, aqua) for the
#series, a muted gray for de-emphasized bars, recessive gray for the grid.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED, GRID, INK, INK2 = "#b5b3ad", "#e4e3df", "#0b0b0b", "#52514e"
BAND = "#ecebe7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False,
    "legend.framealpha": 1.0, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _legend(ax, ncol):
    leg = ax.legend(loc="upper center", ncol=ncol, borderpad=0.5, handlelength=2.6)
    leg.get_frame().set_linewidth(0.8)
    return leg


def _default_saver(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    print("wrote", OUT / f"{name}.png")


#Lorenzo Giannuzzo: the saver can be replaced by other scripts (for example
#sbwm.build_results, which writes PDF and JPG into separate folders).
SAVER = _default_saver


def _save(fig, name):
    SAVER(fig, name)
    plt.close(fig)


def official_scores():
    s = pd.read_csv(EVAL / "official_summary.csv").sort_values("Score", ascending=False)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    colors = [BLUE if "Ours" in c else MUTED for c in s["Controller"]]
    ax.barh(s["Controller"], s["Score"], height=0.55, color=colors)
    for y, v in enumerate(s["Score"]):
        ax.text(v + 0.003, y, f"{v:.4f}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Score, Mean Absolute Reward Per Step [-]")
    ax.set_xlim(0, s["Score"].max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.set_title("Official Evaluation (30 Scenarios, 30 Days Each)", loc="left", color=INK, fontsize=11)
    _save(fig, "fig1_official_scores")


def improvement_heatmap():
    t = pd.read_csv(EVAL / "official_per_scenario.csv")
    col = "Improvement Versus PID With Water Reset [%]"
    piv = t.pivot(index="Building", columns="Climate", values=col)
    piv = piv.reindex(sorted(piv.index, key=lambda b: int(b.split()[-1])))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=10,
                    color="white" if v > 45 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Improvement Versus PID With Water Reset [%]")
    cb.outline.set_edgecolor("black")
    ax.set_title("Reduction Of The Loss On Every Unseen Building And Climate", loc="left", color=INK, fontsize=11)
    _save(fig, "fig2_improvement_heatmap")


def per_scenario():
    t = pd.read_csv(EVAL / "official_per_scenario.csv")
    t["Label"] = t["Building"] + ", " + t["Climate"]
    t = t.sort_values("PID With Water Reset")
    fig, ax = plt.subplots(figsize=(7.0, 8.2))
    y = np.arange(len(t))
    for yi, a, b in zip(y, t["DPC Transformer"], t["PID With Water Reset"]):
        ax.plot([a, b], [yi, yi], color=MUTED, lw=1.5, zorder=1)
    ax.scatter(t["PID With Water Reset"], y, s=46, color=ORANGE, edgecolor="white", linewidth=1.5,
               zorder=2, label="PID With Water Reset")
    ax.scatter(t["DPC Transformer"], y, s=46, color=BLUE, edgecolor="white", linewidth=1.5,
               zorder=3, label="DPC Transformer (Ours)")
    ax.set_yticks(y, t["Label"])
    ax.set_xscale("log")
    ax.set_xlabel("Score, Mean Absolute Reward Per Step [-] (Log Scale)")
    ax.set_ylim(-1, len(t) + 2.2)
    ax.grid(axis="y", visible=False)
    wins = int((t["DPC Transformer"] < t["PID With Water Reset"]).sum())
    ax.set_title(f"Per-Scenario Score, Lower Is Better (Better In {wins} Of {len(t)} Scenarios)",
                 loc="left", color=INK, fontsize=11)
    _legend(ax, 2)
    _save(fig, "fig3_per_scenario")


def loop_progress():
    p = EVAL / "loop_benchmark_6_scenarios_2_days.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    pid = t[t["Policy"] == "PID With Water Reset"]["Score"].iloc[0]
    d = t[t["Policy"] != "PID With Water Reset"]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(d))
    ax.axhline(pid, color=ORANGE, lw=2, label="PID With Water Reset")
    ax.plot(x, d["Score"], color=BLUE, lw=2, marker="o", ms=7, mec="white", mew=1.5,
            label="DPC Transformer (Ours)")
    for xi, (v, wm) in enumerate(zip(d["Score"], d["World Model"])):
        ax.text(xi, v - 0.0035, f"{v:.4f}", ha="center", va="top", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{p}\n({w})" for p, w in zip(d["Policy"], d["World Model"])], fontsize=8.5)
    ax.set_ylabel("Score [-]")
    ax.set_ylim(0, pid * 1.35)
    ax.set_title("Model-Based Loop: Policy Improvement In The Real Simulator", loc="left", color=INK, fontsize=11)
    _legend(ax, 2)
    _save(fig, "fig4_loop_progress")


def world_model_accuracy():
    p = EVAL / "world_model_accuracy.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    hours = [5 / 60, 1, 6, 12]
    labels = ["5 Min", "1 H", "6 H", "12 H"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    last = t.iloc[-1]
    first = t.iloc[0]
    ax.plot(hours, [last[f"Persistence MAE {l} [K]"] for l in labels], color=ORANGE, lw=2,
            marker="o", ms=7, mec="white", mew=1.5, label="Persistence")
    ax.plot(hours, [first[f"Temperature MAE {l} [K]"] for l in labels], color=AQUA, lw=2,
            marker="o", ms=7, mec="white", mew=1.5, label=first["World Model"])
    ax.plot(hours, [last[f"Temperature MAE {l} [K]"] for l in labels], color=BLUE, lw=2,
            marker="o", ms=7, mec="white", mew=1.5, label=last["World Model"])
    ax.set_xlabel("Prediction Horizon [h]")
    ax.set_ylabel("Zone Temperature MAE [K]")
    ax.set_ylim(0, max(last[f"Persistence MAE 12 H [K]"], 1) * 1.35)
    ax.set_title("World Model Accuracy On Unseen Buildings", loc="left", color=INK, fontsize=11)
    _legend(ax, 3)
    _save(fig, "fig5_world_model_accuracy")


def timeseries(scenario: str, title: str, name: str):
    a = EVAL / f"timeseries_{scenario}_dpc.csv"
    b = EVAL / f"timeseries_{scenario}_pid_water.csv"
    if not (a.exists() and b.exists()):
        return
    #Lorenzo Giannuzzo: 5-minute zone temperatures chatter, so the figure shows
    #hourly means of the building-average temperature and of the zone extremes.
    def hourly(df):
        g = df.groupby(np.arange(len(df)) // 12)
        return pd.DataFrame({
            "temp_mean": g["temp_mean"].mean(), "temp_min": g["temp_min"].mean(),
            "temp_max": g["temp_max"].mean(), "heating_setpoint": g["heating_setpoint"].min(),
            "cooling_setpoint": g["cooling_setpoint"].max(), "occupants": g["occupants"].mean(),
            "reward": g["reward"].sum()})
    d0, p0 = pd.read_csv(a), pd.read_csv(b)
    d, p = hourly(d0), hourly(p0)
    h = np.arange(len(d)) + 0.5
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    occupied = d["occupants"].values > 0.5
    for ax in (ax1, ax2):
        edges = np.flatnonzero(np.diff(np.r_[0, occupied.astype(int), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2]):
            ax.axvspan(s0, s1, color="#f7f3e6", zorder=0, lw=0)
    ax1.fill_between(np.arange(len(d) + 1), np.r_[d["heating_setpoint"], d["heating_setpoint"].iloc[-1]],
                     np.r_[d["cooling_setpoint"], d["cooling_setpoint"].iloc[-1]], step="post",
                     color=BAND, edgecolor=MUTED, lw=1, zorder=1, label="Comfort Band")
    for df, color, label in ((p, ORANGE, "PID With Water Reset"), (d, BLUE, "DPC Transformer (Ours)")):
        ax1.fill_between(h, df["temp_min"], df["temp_max"], color=color, alpha=0.12, lw=0, zorder=2)
        ax1.plot(h, df["temp_mean"], color=color, lw=2, zorder=3, label=label)
    lo = min(d["temp_min"].min(), p["temp_min"].min(), d["heating_setpoint"].min())
    hi = max(d["temp_max"].max(), p["temp_max"].max(), d["cooling_setpoint"].max())
    top = hi + (hi - lo) * 0.45
    ax1.set_ylim(lo - 1, top)
    #Lorenzo Giannuzzo: no tick labels behind the legend box.
    ticks = [t for t in ax1.get_yticks() if lo - 1 <= t <= hi + 0.5]
    ax1.set_yticks(ticks)
    ax1.set_ylabel("Zone Temperature [°C]")
    _legend(ax1, 3)
    ax1.set_title(title, loc="left", color=INK, fontsize=11)
    for df, raw, color, label in ((p, p0, ORANGE, "PID With Water Reset"), (d, d0, BLUE, "DPC Transformer (Ours)")):
        ax2.plot(np.arange(len(raw)) * 5 / 60, raw["reward"].cumsum(), color=color, lw=2, label=label)
        ax2.text(len(raw) * 5 / 60 + 0.8, raw["reward"].sum(), f"{raw['reward'].sum():.1f}",
                 va="center", fontsize=9, color=INK)
    ax2.set_ylabel("Cumulative Reward [-]")
    ax2.set_xlabel("Time [h]")
    ax2.set_xlim(0, len(d) + 7)
    fig.text(0.125, 0.015, "Shaded Columns: Occupied Hours. Lines: Hourly Building Mean; Bands: Zone Range.",
             fontsize=8, color=INK2)
    _save(fig, name)


def main():
    official_scores()
    improvement_heatmap()
    per_scenario()
    loop_progress()
    world_model_accuracy()
    timeseries("building_train_8__phoenix_4", "Building 8, Phoenix: Three Days Under Both Controllers",
               "fig6_timeseries_phoenix")
    timeseries("building_train_1__fairbanks_1", "Building 1, Fairbanks: Three Days Under Both Controllers",
               "fig7_timeseries_fairbanks")


if __name__ == "__main__":
    main()
