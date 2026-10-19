"""Build the result tables (CSV) in ``evaluations/``.

Sources:
  * evaluations/official_eval_dpc_v3.txt  official 30-scenario, 30-day run
  * baseline_returns.json                  shipped baseline returns
  * wm_runs/*/real_*.json                  short-window benchmark runs (optional)
  * wm_runs/*/train.log                    world-model evaluation logs (optional)

Usage (from the repository root):
    python -m sbwm.report_tables
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

EVAL = Path("evaluations")
CLIMATES = {"atlanta": "Atlanta", "denver": "Denver", "fairbanks": "Fairbanks",
            "phoenix": "Phoenix", "singapore": "Singapore", "minneapolis": "Minneapolis"}
BASELINES = {"bang_bang": "Bang-Bang", "fixed_setpoint": "Fixed Setpoint", "pid": "PID",
             "pid_water": "PID With Water Reset"}

#Lorenzo Giannuzzo: checkpoints of the DPC loop evaluated on the short-window
#benchmark, in chronological order, with the world model each was trained on.
LOOP = [
    ("DPC Iteration 250", "World Model v2", "wm_runs/policy/real_it250.json"),
    ("DPC v1", "World Model v2", "wm_runs/policy2/real_it00750.json"),
    ("DPC v2", "World Model v2", "wm_runs/policy3/real_it00800.json"),
    ("DPC v3", "World Model v3", "wm_runs/policy4/real_it00400.json"),
    ("DPC v4", "World Model v4", "wm_runs/policy5/real_it00200.json"),
]
WM_LOGS = [("World Model v1", "wm_runs/v1/train.log"), ("World Model v2", "wm_runs/v2_s0/train.log"),
           ("World Model v3", "wm_runs/v3/train.log"), ("World Model v4", "wm_runs/v4b/train.log")]


def _scenario_key(path: str) -> str:
    return "/".join(Path(path).parts[-2:])


def _split(key: str) -> tuple[str, str]:
    building, weather = key.split("/")
    return "Building " + building.split("_")[-1], CLIMATES[weather.split("_")[0]]


def official_table(text_file: Path, out: Path) -> None:
    pat = re.compile(r"^(building_train_\d+/\w+)\s+dpc\s+reward (-?[\d.]+) \(comfort (-?[\d.]+) "
                     r"energy (-?[\d.]+) carbon (-?[\d.]+)\)", re.M)
    dpc = {m.group(1): [float(m.group(i)) for i in range(2, 6)] for m in pat.finditer(text_file.read_text())}
    base = json.loads(Path("baseline_returns.json").read_text())
    rows = []
    for key in sorted(dpc):
        building, climate = _split(key)
        b = {name: abs(base[key][k]) / 8640 for k, name in BASELINES.items()}
        best_name = min(b, key=b.get)
        ours = abs(dpc[key][0])
        rows.append({
            "Scenario": key, "Building": building, "Climate": climate,
            "DPC Transformer": round(ours, 5),
            **{name: round(v, 5) for name, v in b.items()},
            "Best Baseline": best_name, "Best Baseline Score": round(b[best_name], 5),
            "Improvement Versus PID With Water Reset [%]": round(100 * (1 - ours / b["PID With Water Reset"]), 1),
            "Improvement Versus Best Baseline [%]": round(100 * (1 - ours / b[best_name]), 1),
            "DPC Comfort Term": round(dpc[key][1], 5), "DPC Energy Term": round(dpc[key][2], 5),
            "DPC Carbon Term": round(dpc[key][3], 5),
        })
    with open(out / "official_per_scenario.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    summary = [{"Controller": "DPC Transformer (Ours)",
                "Score": round(sum(r["DPC Transformer"] for r in rows) / n, 5)}]
    for name in BASELINES.values():
        summary.append({"Controller": name, "Score": round(sum(r[name] for r in rows) / n, 5)})
    summary.append({"Controller": "Best Baseline Per Scenario",
                    "Score": round(sum(r["Best Baseline Score"] for r in rows) / n, 5)})
    wins = sum(r["DPC Transformer"] < r["Best Baseline Score"] for r in rows)
    with open(out / "official_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Controller", "Score"])
        w.writeheader()
        w.writerows(summary)
    print(f"official: {n} scenarios, DPC wins against the best baseline in {wins}")


def loop_table(out: Path) -> None:
    rows = []
    pid = {}
    for label, wm, path in LOOP:
        p = Path(path)
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        for r in res:
            if r["policy"] == "pid_water":
                pid[_scenario_key(r["scenario"])] = abs(r["mean_reward"])
        dpc = {_scenario_key(r["scenario"]): abs(r["mean_reward"]) for r in res if r["policy"] == "dpc"}
        rows.append({"Policy": label, "World Model": wm, **{k: round(v, 5) for k, v in sorted(dpc.items())},
                     "Score": round(sum(dpc.values()) / len(dpc), 5)})
    if not rows:
        return
    if pid:
        rows.insert(0, {"Policy": "PID With Water Reset", "World Model": "-",
                        **{k: round(v, 5) for k, v in sorted(pid.items())},
                        "Score": round(sum(pid.values()) / len(pid), 5)})
    fields = list(rows[-1])
    with open(out / "loop_benchmark_6_scenarios_2_days.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def world_model_table(out: Path) -> None:
    rows = []
    for name, log in WM_LOGS:
        p = Path(log)
        if not p.exists():
            continue
        evals = [json.loads(line) for line in p.read_text().splitlines() if line.startswith("{")]
        if not evals:
            continue
        e = evals[-1]
        row = {"World Model": name}
        for h, label in (("1", "5 Min"), ("12", "1 H"), ("72", "6 H"), ("144", "12 H")):
            row[f"Temperature MAE {label} [K]"] = round(e["temp_mae"][h], 3)
        for h, label in (("1", "5 Min"), ("12", "1 H"), ("72", "6 H"), ("144", "12 H")):
            row[f"Persistence MAE {label} [K]"] = round(e["persistence_mae"][h], 3)
        if "comfort_sum_pred_vs_true" in e:
            cp, ct = e["comfort_sum_pred_vs_true"]
            row["Comfort Captured [%]"] = round(100 * cp / ct, 1)
        row["Reward Correlation"] = round(e["reward_corr"], 3)
        rows.append(row)
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "world_model_accuracy.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    EVAL.mkdir(exist_ok=True)
    official_table(EVAL / "official_eval_dpc_v3.txt", EVAL)
    loop_table(EVAL)
    world_model_table(EVAL)
    for f in sorted(EVAL.glob("*.csv")):
        print("wrote", f)


if __name__ == "__main__":
    main()
