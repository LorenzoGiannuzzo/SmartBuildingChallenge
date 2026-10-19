"""Generate the fixed benchmark used to compare policies during development.

The benchmark uses buildings never seen in training and never used by the
final test set (seeds from 2000 onwards, while training and test buildings use
seeds below 1300). Every climate gets two buildings, one starting in the first
half of the year and one in the second half, so that both heating and cooling
seasons are covered. Every third valid building keeps
the generator's default (larger) footprint, as in the training and test sets.
Episodes last seven days, so every one of them contains a weekend and a Monday
morning start-up. The start date is written into the scenario config, which
is where ``sbwm.eval_policy`` reads it from.

Usage (from the repository root):
    python -m sbwm.make_benchmark
    python -m sbwm.eval_policy --scenarios bench_data/scenarios --policies pid_water dpc \
        --weights policies/dpc_v4.pt --steps 2016 --workers 2 --out results/benchmark.json
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STATIONS = ("atlanta_1", "denver_1", "fairbanks_1", "minneapolis_1", "phoenix_1", "singapore_1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-seed", type=int, default=2000)
    ap.add_argument("--per-climate", type=int, default=2)
    ap.add_argument("--default-every", type=int, default=3)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--root", default="bench_data")
    args = ap.parse_args()
    root = Path(args.root)
    bdir, sdir = root / "buildings", root / "scenarios"
    bdir.mkdir(parents=True, exist_ok=True)

    needed = args.per_climate * len(STATIONS)
    seed, made, rows = args.start_seed, 0, []
    while made < needed and seed < args.start_seed + 10 * needed:
        yml = bdir / f"building_{seed:04d}.yaml"
        if not yml.exists():
            subprocess.run([sys.executable, "-m", "smart_buildings.run_floor_generator", "--seeds", str(seed),
                            "--no-pdf", "--output-dir", bdir.as_posix()], check=True, stdout=subprocess.DEVNULL)
        cfg = yaml.safe_load(yml.read_text())
        default = made % args.default_every == args.default_every - 1
        if not default:
            rng = np.random.default_rng(cfg["seed"])
            cfg["floor_plan"]["width"] = int(rng.uniform(310, 440))
            cfg["floor_plan"]["height"] = int(rng.uniform(185, 305))
            yml.write_text(yaml.safe_dump(cfg, sort_keys=False))
        #Lorenzo Giannuzzo: consecutive buildings share a climate and split the
        #year into equal segments (January-June and July-December with two).
        station = STATIONS[(made // args.per_climate) % len(STATIONS)]
        half = made % args.per_climate
        target = sdir / yml.stem
        gin = target / station / "config.gin"
        if not gin.exists():
            res = subprocess.run([sys.executable, "-m", "smart_buildings.run_scenario_generator",
                                  "--buildings", yml.as_posix(), "--weathers", station,
                                  "--output-dir", sdir.as_posix()],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode != 0 or not gin.exists():
                shutil.rmtree(target, ignore_errors=True)
                print(f"{yml.stem} invalid floor plan, skipped", flush=True)
                seed += 1
                continue
        rng = np.random.default_rng([seed, half])
        segment = 362 // args.per_climate
        lo = pd.Timestamp("2023-01-02") + pd.Timedelta(days=segment * half)
        span = segment - args.days - 1
        day = lo + pd.Timedelta(days=int(rng.integers(0, span)))
        start = f"{day:%Y-%m-%d} 00:00"
        text = re.sub(r"^start_timestamp = '.*'$", f"start_timestamp = '{start}'", gin.read_text(), flags=re.M)
        gin.write_text(text)
        size = "default" if default else "shipped"
        rows.append(f"{yml.stem},{station},{start},{size}")
        print(f"{yml.stem} {station} from {start} ({size} footprint)", flush=True)
        made += 1
        seed += 1
    (root / "benchmark_set.csv").write_text("building,weather,start,footprint\n" + "\n".join(rows) + "\n")
    print(f"benchmark scenarios: {made}")


if __name__ == "__main__":
    main()
