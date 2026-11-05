"""Oracle bound: receding-horizon search with the true simulator.

At every decision hour the simulator and the base policy are forked into
worker processes (copy-on-write, so the building state is cloned exactly);
each worker applies one candidate modification of the policy commands for the
coming hour, lets the unmodified policy continue for the rest of the horizon,
and returns the official reward summed over the horizon. The best candidate is
applied to the real run. The pure policy is always among the candidates, so the
oracle is never worse than the policy over each horizon (a rollout algorithm
with a perfect model): the gap between the two scores bounds what better
short-term decisions can still gain.

Candidates combine, for the coming hour:
    plant      as the policy / forced off (air handlers and boiler loop) / forced on
    water      supply-water command offset (normalized): -0.5, 0, +0.5, +1
    supply air air-handler supply temperature offset: -0.5, 0, +0.5
    reheat     zone reheat valve offset: -0.3, 0, +0.3

Usage:
    python -m sbwm.oracle --weights policies/v14_new2wm.pt \
        --scenarios building_train_1/fairbanks_1 building_train_2/denver_8 \
        --steps 2016 --horizon 144 --candidates 31 --workers 31 --out results/oracle.json
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import numpy as np

PLANT = ("policy", "off", "on")
WATER = (0.0, -0.5, 0.5, 1.0)
SUPPLY = (0.0, -0.5, 0.5)
REHEAT = (0.0, -0.3, 0.3)

#Lorenzo Giannuzzo: state shared with the forked workers (copy-on-write).
_G: dict = {}


def _share_superlu():
    #Lorenzo Giannuzzo: the thermal solver keeps a sparse LU factorization of a
    #constant matrix, which cannot be pickled; deep copies share it.
    from scipy.sparse.linalg import SuperLU

    copy._deepcopy_dispatch[SuperLU] = lambda x, memo: x


def modify(a: dict, cand: tuple, s) -> dict:
    """Policy action a with candidate cand = (plant, water, supply, reheat) applied."""
    plant, water, supply, reheat = cand
    c = a["continuous_action"].astype(np.float32).copy()
    d = a["discrete_action"].astype(np.int32).copy()
    if plant == "off":
        d[s.ahu_run_act] = 0
        d[s.hws_run_act] = 0
    elif plant == "on":
        d[s.ahu_run_act] = 1
        d[s.hws_run_act] = 1
    if water and s.hws_temp_act >= 0:
        c[s.hws_temp_act] += water
    if supply:
        c[s.ahu_temp_act] += supply
    if reheat:
        ok = s.zone_reheat_act >= 0
        c[s.zone_reheat_act[ok]] += reheat
    return {"continuous_action": np.clip(c, -1, 1), "discrete_action": d}


def _act(pol, obs):
    a = pol.get_action(obs)
    return {"continuous_action": np.asarray(a["continuous_action"], np.float32),
            "discrete_action": np.asarray(a["discrete_action"], np.int32)}


def _evaluate(i: int) -> float:
    """Reward summed over the horizon for candidate i, from the forked state."""
    env, pol, obs, s = _G["env"], _G["pol"], _G["obs"], _G["schema"]
    cand, every, horizon = _G["cands"][i], _G["every"], _G["horizon"]
    total = 0.0
    for k in range(horizon):
        a = _act(pol, obs)
        if k < every:
            a = modify(a, cand, s)
        obs, r, term, trunc, _ = env.step(a)
        total += float(r)
        if term or trunc:
            break
    return total


def candidate_set(n: int, rng: np.random.Generator) -> list[tuple]:
    full = list(itertools.product(PLANT, WATER, SUPPLY, REHEAT))
    base = ("policy", 0.0, 0.0, 0.0)
    rest = [c for c in full if c != base]
    pick = rng.choice(len(rest), size=min(n - 1, len(rest)), replace=False)
    return [base] + [rest[j] for j in pick]


def run(scenario: Path, weights: str, steps: int, every: int, horizon: int, n_cand: int, workers: int,
        seed: int) -> dict:
    from sbwm.collect import _load_env
    from sbwm.policy import DPCPolicy
    from sbwm.schema import BuildingSchema

    start = re.search(r"^start_timestamp = '(.*)'$", (scenario / "config.gin").read_text(), re.M).group(1)
    env, meta = _load_env(scenario, start)
    pol = DPCPolicy(meta, env.action_spec(), weights=weights)
    s = BuildingSchema.from_meta(meta)
    reward_fn = env.reward_function
    original = reward_fn.compute_reward
    comps = []

    def hook(info):
        resp = original(info)
        comps.append((resp.normalized_productivity_regret, resp.normalized_energy_cost,
                      resp.normalized_carbon_emission))
        return resp

    reward_fn.compute_reward = hook
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    rewards, chosen, gains = [], [], []
    ctx = mp.get_context("fork")
    t0 = time.time()
    t = 0
    while t < steps:
        h = min(horizon, steps - t)
        cands = candidate_set(n_cand, rng)
        _G.update(env=env, pol=pol, obs=obs, schema=s, cands=cands, every=every, horizon=h)
        with ctx.Pool(min(workers, len(cands))) as pool:
            scores = pool.map(_evaluate, range(len(cands)))
        best = int(np.argmax(scores))
        chosen.append(cands[best])
        gains.append(scores[best] - scores[0])
        for _ in range(min(every, steps - t)):
            a = modify(_act(pol, obs), cands[best], s)
            obs, r, term, trunc, _ = env.step(a)
            rewards.append(float(r))
            t += 1
            if term or trunc:
                t = steps
                break
        if len(chosen) % 12 == 0:
            print(f"{scenario.name} {t}/{steps} mean reward {np.mean(rewards):.4f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
    c = np.asarray(comps[-len(rewards):])
    counts = {}
    for k, name in enumerate(("plant", "water", "supply", "reheat")):
        vals = [str(x[k]) for x in chosen]
        counts[name] = {v: vals.count(v) for v in sorted(set(vals))}
    return {"scenario": scenario.as_posix(), "weights": weights, "steps": len(rewards),
            "every": every, "horizon": horizon, "candidates": n_cand,
            "mean_reward": float(np.mean(rewards)), "comfort": float(0.6 * c[:, 0].mean()),
            "energy": float(-0.2 * c[:, 1].mean()), "carbon": float(-0.2 * c[:, 2].mean()),
            "chosen": counts, "mean_gain_per_decision": float(np.mean(gains)),
            "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True,
                    help="building/station folders under smart_buildings/training_scenarios")
    ap.add_argument("--root", default="smart_buildings/training_scenarios")
    ap.add_argument("--steps", type=int, default=2016)
    ap.add_argument("--every", type=int, default=12, help="steps between decisions (12 = 1 h)")
    ap.add_argument("--horizon", type=int, default=144, help="look-ahead steps (144 = 12 h)")
    ap.add_argument("--candidates", type=int, default=31)
    ap.add_argument("--workers", type=int, default=31)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/oracle.json")
    args = ap.parse_args()
    _share_superlu()
    out = Path(args.out)
    res = json.loads(out.read_text()) if out.exists() else []
    done = {(r["scenario"], r["weights"], r["steps"], r["horizon"], r["candidates"]) for r in res}
    for sc in args.scenarios:
        scen = Path(args.root) / sc
        key = (scen.as_posix(), args.weights, args.steps, args.horizon, args.candidates)
        if key in done:
            continue
        r = run(scen, args.weights, args.steps, args.every, args.horizon, args.candidates, args.workers, args.seed)
        print(json.dumps({k: r[k] for k in ("scenario", "mean_reward", "comfort", "energy", "carbon",
                                             "mean_gain_per_decision", "seconds")}), flush=True)
        res.append(r)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
