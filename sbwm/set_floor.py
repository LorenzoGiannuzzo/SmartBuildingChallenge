"""Copy a policy checkpoint with a floor on the supply-water command.

The floor is in the normalized command range (-1 minimum, 1 maximum water
temperature) and acts at deployment only.

Usage:
    python -m sbwm.set_floor policies/v11_long_fit3.pt policies/v11_long_fit3_w0.pt 0.0
"""

from __future__ import annotations

import sys

import torch


def main():
    src, dst, floor = sys.argv[1], sys.argv[2], float(sys.argv[3])
    ck = torch.load(src, weights_only=False)
    ck["water_floor"] = floor
    torch.save(ck, dst)
    print(f"{dst}: supply-water command floor {floor} at deployment")


if __name__ == "__main__":
    main()
