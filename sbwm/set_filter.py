"""Copy a policy checkpoint with a deployment command filter attached.

Usage:
    python -m sbwm.set_filter policies/v11_base.pt policies/v11_base_f03.pt 0.3
"""

from __future__ import annotations

import sys

import torch


def main():
    src, dst, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
    ck = torch.load(src, weights_only=False)
    ck["action_filter"] = alpha
    ck.pop("filter_in_training", None)
    torch.save(ck, dst)
    print(f"{dst}: command filter {alpha} at deployment")


if __name__ == "__main__":
    main()
