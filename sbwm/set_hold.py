"""Copy a policy checkpoint with an action hold at deployment.

The commands are recomputed every K decisions (5 min each) and repeated in
between; the policy state still advances at every step.

Usage:
    python -m sbwm.set_hold policies/v11_long_fit3.pt policies/v11_long_fit3_h3.pt 3
"""

from __future__ import annotations

import sys

import torch


def main():
    src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ck = torch.load(src, weights_only=False)
    ck["action_hold"] = k
    torch.save(ck, dst)
    print(f"{dst}: commands held for {k} decisions ({5 * k} min)")


if __name__ == "__main__":
    main()
