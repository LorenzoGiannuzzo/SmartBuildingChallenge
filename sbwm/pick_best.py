"""Print the best policies of a list by their 14-day official score.

Usage:
    python -m sbwm.pick_best results/official30_14d.json 2 policies/a.pt policies/b.pt ...

The score is the mean |reward| over the scenarios (lower is better); only
weights evaluated on all 30 scenarios with 4032 steps are ranked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main():
    path, k, wanted = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    res = json.loads(Path(path).read_text())
    by: dict[str, dict[str, float]] = {}
    for r in res:
        if r["policy"] == "dpc" and r["steps"] == 4032 and r["weights"] in wanted:
            by.setdefault(r["weights"], {})[r["scenario"]] = abs(r["mean_reward"])
    scores = sorted((np.mean(list(v.values())), w) for w, v in by.items() if len(v) >= 30)
    for s, w in scores:
        print(f"{w} {s:.4f}", file=sys.stderr)
    print(" ".join(w for _, w in scores[:k]))


if __name__ == "__main__":
    main()
