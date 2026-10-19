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
    python -m sbwm.make_buildings --start-seed 1000 --valid 100 --default-every 3 --root month_data
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
    ap.add_argument("--count", type=int, default=60, help="seeds to try (ignored with --valid)")
    ap.add_argument("--valid", type=int, default=None, help="stop after this many valid buildings")
    ap.add_argument("--default-every", type=int, default=0,
                    help="every n-th valid building keeps the generator's default (larger) footprint")
    ap.add_argument("--root", default="wm_data")
    args = ap.parse_args()
    root = Path(args.root)
    bdir, sdir = root / "buildings", root / "scenarios"
    bdir.mkdir(parents=True, exist_ok=True)

    target_valid = args.valid if args.valid is not None else None
    last_seed = args.start_seed + (10 * target_valid if target_valid else args.count)
    valid, seed, rows = 0, args.start_seed, []
    while seed < last_seed and (target_valid is None or valid < target_valid):
        yml = bdir / f"building_{seed:04d}.yaml"
        if not yml.exists():
            subprocess.run([sys.executable, "-m", "smart_buildings.run_floor_generator", "--seeds", str(seed),
                            "--no-pdf", "--output-dir", bdir.as_posix()],
                           check=True, stdout=subprocess.DEVNULL)
        cfg = yaml.safe_load(yml.read_text())
        #Lorenzo Giannuzzo: the position in the list of valid buildings decides
        #the footprint, so rerunning the script gives the same set.
        default = args.default_every > 0 and valid % args.default_every == args.default_every - 1
        if not default:
            rng = np.random.default_rng(cfg["seed"])
            cfg["floor_plan"]["width"] = int(rng.uniform(310, 440))
            cfg["floor_plan"]["height"] = int(rng.uniform(185, 305))
            yml.write_text(yaml.safe_dump(cfg, sort_keys=False))

        target = sdir / yml.stem
        if not all((target / s / "config.gin").exists() for s in STATIONS):
            res = subprocess.run([sys.executable, "-m", "smart_buildings.run_scenario_generator",
                                  "--buildings", yml.as_posix(), "--weathers", *STATIONS,
                                  "--output-dir", sdir.as_posix()],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode != 0:
                shutil.rmtree(target, ignore_errors=True)
                print(f"{yml.stem} invalid floor plan, skipped", flush=True)
                seed += 1
                continue
        valid += 1
        size = "default" if default else "shipped"
        rows.append(f"{yml.stem},{size}")
        print(f"{yml.stem} ok ({size} footprint)", flush=True)
        seed += 1
    (root / "buildings_list.csv").write_text("building,footprint\n" + "\n".join(rows) + "\n")
    print(f"valid buildings: {valid}")


if __name__ == "__main__":
    main()
