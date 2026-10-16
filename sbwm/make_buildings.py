"""Generate training buildings and their scenarios for data collection.

Buildings come from the official generator; their footprint is rescaled to the
range of the six shipped buildings (310-440 ft by 185-305 ft) with a generator
seeded by the building seed. Envelope materials are stratified by the official
generator across the whole batch, so use a seed range of your own on each
machine: every machine then contributes different buildings, which only adds
diversity. Seeds whose floor plan fails the generator's validity checks are
skipped, and existing scenario folders are left untouched.

Usage (from the repository root):
    python -m sbwm.make_buildings --start-seed 600 --count 60
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

STATIONS = ("atlanta_1", "denver_1", "fairbanks_1", "minneapolis_1", "phoenix_1", "singapore_1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-seed", type=int, default=600)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--root", default="wm_data")
    args = ap.parse_args()
    root = Path(args.root)
    bdir, sdir = root / "buildings", root / "scenarios"
    bdir.mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, "-m", "smart_buildings.run_floor_generator",
                    "--start-seed", str(args.start_seed), "--count", str(args.count),
                    "--no-pdf", "--output-dir", str(bdir)],
                   check=True, stdout=subprocess.DEVNULL)

    valid = 0
    for seed in range(args.start_seed, args.start_seed + args.count):
        yml = bdir / f"building_{seed:04d}.yaml"
        cfg = yaml.safe_load(yml.read_text())
        rng = np.random.default_rng(cfg["seed"])
        cfg["floor_plan"]["width"] = int(rng.uniform(310, 440))
        cfg["floor_plan"]["height"] = int(rng.uniform(185, 305))
        yml.write_text(yaml.safe_dump(cfg, sort_keys=False))

        target = sdir / yml.stem
        if all((target / s / "config.gin").exists() for s in STATIONS):
            valid += 1
            print(f"{yml.stem} exists", flush=True)
            continue
        res = subprocess.run([sys.executable, "-m", "smart_buildings.run_scenario_generator",
                              "--buildings", str(yml), "--weathers", *STATIONS,
                              "--output-dir", str(sdir)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0:
            valid += 1
            print(f"{yml.stem} ok", flush=True)
        else:
            shutil.rmtree(target, ignore_errors=True)
            print(f"{yml.stem} invalid floor plan, skipped", flush=True)
    print(f"valid buildings: {valid}")


if __name__ == "__main__":
    main()
