"""Copy a policy checkpoint with a frost guard at deployment.

When the Q quantile of the zone temperatures (default 0.25) drops below ON
degrees Celsius the hot-water loop runs,
the air handlers recirculate and the zones below OFF open their reheat valves,
until that quantile is back above OFF. The policy inputs are not changed.

Usage:
    python -m sbwm.set_frost policies/v11_long_fit3.pt policies/v11_long_fit3_fr10.pt 10 13 [0.25]
"""

from __future__ import annotations

import sys

import torch


def main():
    src, dst, on, off = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    q = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25
    ck = torch.load(src, weights_only=False)
    ck["frost_guard"] = [on, off, q]
    torch.save(ck, dst)
    print(f"{dst}: frost guard on below {on} C, off above {off} C (zone quantile {q})")


if __name__ == "__main__":
    main()
