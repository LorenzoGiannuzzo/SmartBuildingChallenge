"""Generate an independent validation set of unseen buildings.

Buildings use seeds never used for training (default 700 onwards). The first
``--small`` valid buildings are rescaled to the footprint range of the six
shipped buildings; the next ``--large`` keep the generator's default
footprint (400-700 ft by 300-500 ft), which yields larger buildings than any
seen in training. Each building is paired with one official weather strip,
cycling over the five official stations, so the set mirrors the official
evaluation (30-day episodes from the strip's start date).

Usage (from the repository root):
    python -m sbwm.make_validation --small 20 --large 10
    python -m sbwm.eval_policy --scenarios val_data/scenarios \
        --policies dpc pid_water fixed_setpoint --weights policies/dpc_v3.pt \
        --steps 8640 --workers 10 --out results/validation_dpc_v3.json
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

STATIONS = ("atlanta", "denver", "fairbanks", "phoenix", "singapore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-seed", type=int, default=700)
    ap.add_argument("--small", type=int, default=20)
    ap.add_argument("--large", type=int, default=10)
    ap.add_argument("--root", default="val_data")
    args = ap.parse_args()
    root = Path(args.root)
    bdir, sdir = root / "buildings", root / "scenarios"
    bdir.mkdir(parents=True, exist_ok=True)

    needed = args.small + args.large
    seed, made, rows = args.start_seed, 0, []
    while made < needed and seed < args.start_seed + 10 * needed:
        yml = bdir / f"building_{seed:04d}.yaml"
        if not yml.exists():
            subprocess.run([sys.executable, "-m", "smart_buildings.run_floor_generator", "--seeds", str(seed),
                            "--no-pdf", "--output-dir", bdir.as_posix()], check=True, stdout=subprocess.DEVNULL)
        cfg = yaml.safe_load(yml.read_text())
        size = "shipped" if made < args.small else "default"
        if size == "shipped":
            #Lorenzo Giannuzzo: same rescaling as the training buildings, so the
            #small half is directly comparable with the official six.
            rng = np.random.default_rng(cfg["seed"])
            cfg["floor_plan"]["width"] = int(rng.uniform(310, 440))
            cfg["floor_plan"]["height"] = int(rng.uniform(185, 305))
            yml.write_text(yaml.safe_dump(cfg, sort_keys=False))
        station = STATIONS[made % len(STATIONS)]
        strip = f"{station}_{(made // len(STATIONS)) % 8 + 1}"
        target = sdir / yml.stem
        if not (target / strip / "config.gin").exists():
            res = subprocess.run([sys.executable, "-m", "smart_buildings.run_scenario_generator",
                                  "--buildings", yml.as_posix(), "--weathers", strip,
                                  "--output-dir", sdir.as_posix()],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode != 0:
                shutil.rmtree(target, ignore_errors=True)
                print(f"{yml.stem} invalid floor plan, skipped", flush=True)
                seed += 1
                continue
        rows.append(f"{yml.stem},{strip},{size}")
        print(f"{yml.stem} {strip} ({size} footprint)", flush=True)
        made += 1
        seed += 1
    (root / "validation_set.csv").write_text("building,weather,footprint\n" + "\n".join(rows) + "\n")
    print(f"validation scenarios: {made}")


if __name__ == "__main__":
    main()
