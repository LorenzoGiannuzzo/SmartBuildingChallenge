"""Evaluate policies in the real simulator on the official scenario windows.

Each scenario is run from its own configured start date for ``--steps`` steps
(8640 is the full official episode). Occupancy is seeded in the configs, so
every policy sees the same occupants and weather. The score is the
competition metric: mean absolute per-step reward, averaged over scenarios.

Usage:
    python -m sbwm.eval_policy --policies dpc pid_water --weights wm_runs/policy/policy.pt \
        --scenarios smart_buildings/training_scenarios --steps 2016 --workers 2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import numpy as np


def run(job: dict) -> dict:
    try:
        return _run(job)
    except Exception:  # noqa: BLE001
        import traceback

        return {**job, "error": traceback.format_exc()}


def _run(job: dict) -> dict:
    import torch

    from sbwm.collect import _load_env

    torch.set_num_threads(1)
    scen = Path(job["scenario"])
    start = re.search(r"^start_timestamp = '(.*)'$", (scen / "config.gin").read_text(), re.M).group(1)
    env, meta = _load_env(scen, start)
    spec = env.action_spec()
    if job["policy"] == "dpc":
        from sbwm.policy import DPCPolicy

        pol = DPCPolicy(meta, spec, weights=job["weights"])
    elif job["policy"] == "zonepi":
        from sbwm.zone_pi import ZonePIPolicy

        pol = ZonePIPolicy(meta, spec, params=job.get("params"))
    else:
        from smart_buildings.evaluate import POLICY_MAP

        pol = POLICY_MAP[job["policy"]](meta, spec)
    reward_fn = env.reward_function
    original = reward_fn.compute_reward
    comps = []

    def hook(info):
        resp = original(info)
        comps.append((resp.normalized_productivity_regret, resp.normalized_energy_cost,
                      resp.normalized_carbon_emission))
        return resp

    reward_fn.compute_reward = hook
    obs, _ = env.reset()
    rewards = []
    t0 = time.time()
    for _ in range(job["steps"]):
        a = pol.get_action(obs)
        obs, r, term, trunc, _ = env.step({
            "continuous_action": np.asarray(a["continuous_action"], np.float32),
            "discrete_action": np.asarray(a["discrete_action"], np.int32)})
        rewards.append(r)
        if term or trunc:
            break
    c = np.asarray(comps[-len(rewards):])
    return {**job, "mean_reward": float(np.mean(rewards)), "steps_done": len(rewards),
            "comfort": float(0.6 * c[:, 0].mean()), "energy": float(-0.2 * c[:, 1].mean()),
            "carbon": float(-0.2 * c[:, 2].mean()), "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["dpc", "pid_water"])
    ap.add_argument("--weights", default="wm_runs/policy/policy.pt")
    ap.add_argument("--scenarios", default="smart_buildings/training_scenarios")
    ap.add_argument("--filter", nargs="*", help="substrings a scenario path must contain")
    ap.add_argument("--steps", type=int, default=2016)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--params", default="", help="zonepi parameters, e.g. margin=0.5,ki=0.02")
    args = ap.parse_args()
    scen = sorted(p for p in Path(args.scenarios).glob("*/*") if (p / "config.gin").exists())
    if args.filter:
        scen = [p for p in scen if any(f in str(p) for f in args.filter)]
    jobs = [{"scenario": str(s), "policy": p, "steps": args.steps, "weights": args.weights,
             "params": args.params} for s in scen for p in args.policies]
    results = []
    #Lorenzo Giannuzzo: results are saved after every run and reused on a
    #restart, so a long evaluation can be interrupted and resumed.
    if args.out and Path(args.out).exists():
        results = json.loads(Path(args.out).read_text())
        #Lorenzo Giannuzzo: baselines do not depend on the policy weights, so their
        #runs are reused whatever --weights is.
        def key(r):
            w = r["weights"] if r["policy"] == "dpc" else r.get("params") if r["policy"] == "zonepi" else None
            return Path(r["scenario"]).as_posix(), r["policy"], r["steps"], w

        done = {key(r) for r in results}
        jobs = [j for j in jobs if key(j) not in done]
        print(f"resuming: {len(results)} runs already done, {len(jobs)} to go", flush=True)
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(run, jobs):
            if "error" in res:
                print(f"FAILED {res['scenario']} {res['policy']}\n{res['error']}", flush=True)
                continue
            results.append(res)
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(results, indent=1))
            name = "/".join(Path(res["scenario"]).parts[-2:])
            print(f"{name:34s} {res['policy']:10s} reward {res['mean_reward']:.4f} "
                  f"(comfort {res['comfort']:.4f} energy {res['energy']:.4f} carbon {res['carbon']:.4f}) "
                  f"{res['seconds']:.0f}s", flush=True)
    #Lorenzo Giannuzzo: one file can hold several DPC weights, so DPC runs are
    #labelled by their weights file and only scenarios of this call are scored.
    wanted = {Path(s).as_posix() for s in map(str, scen)}
    results = [r for r in results if Path(r["scenario"]).as_posix() in wanted and r["steps"] == args.steps]

    def label(r):
        if r["policy"] == "zonepi":
            return f"zonepi:{r.get('params', '')}"
        if r["policy"] != "dpc":
            return r["policy"]
        #Lorenzo Giannuzzo: checkpoints of different runs share file names, so
        #the run folder is part of the label.
        w = Path(r["weights"])
        return f"dpc:{w.stem}" if w.parent.name == "policies" else f"dpc:{w.parent.name}/{w.stem}"

    by = {}
    for r in results:
        by.setdefault(label(r), {})[Path(r["scenario"]).as_posix()] = abs(r["mean_reward"])
    print("\nscore (mean |reward| over scenarios, lower is better)")
    for name, rs in sorted(by.items()):
        print(f"  {name:24s} {np.mean(list(rs.values())):.4f}  over {len(rs)} scenarios")
    #Lorenzo Giannuzzo: the paired comparison removes most of the variance due
    #to weather and building, since every policy runs on the same scenarios.
    ref = by.get("pid_water")
    if ref:
        print("\npaired improvement over pid_water (same scenarios)")
        for name, rs in sorted(by.items()):
            common = sorted(set(rs) & set(ref))
            if name == "pid_water" or not common:
                continue
            imp = np.array([1 - rs[s] / ref[s] for s in common])
            print(f"  {name:24s} mean {100 * imp.mean():5.1f}%  median {100 * np.median(imp):5.1f}%  "
                  f"worst {100 * imp.min():6.1f}%  wins {int((imp > 0).sum())}/{len(common)}")
    #Lorenzo Giannuzzo: on full official episodes the shipped baseline returns
    #are directly comparable, so they are printed on the same scenarios.
    ref = Path("baseline_returns.json")
    if args.steps == 8640 and ref.exists():
        base = json.loads(ref.read_text())
        keys = sorted({"/".join(Path(r["scenario"]).parts[-2:]) for r in results} & set(base))
        if keys:
            print("  official baselines on the same scenarios:")
            for b in ("bang_bang", "fixed_setpoint", "pid", "pid_water"):
                print(f"  {b:14s} {np.mean([abs(base[k][b]) / 8640 for k in keys]):.4f}")


if __name__ == "__main__":
    main()
