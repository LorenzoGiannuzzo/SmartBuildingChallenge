"""Copy a policy checkpoint with deployment options set.

Options: water_floor=F, frost_guard=ON,OFF[,Q], action_hold=K, min_run=K,
action_filter=A (see sbwm.policy.DPCPolicy).

Usage:
    python -m sbwm.set_option policies/v14_new2wm.pt policies/v14_new2wm_mr6_fr5.pt min_run=6 frost_guard=5,8
"""

from __future__ import annotations

import sys

import torch

INT = {"action_hold", "min_run"}


def main():
    src, dst, opts = sys.argv[1], sys.argv[2], sys.argv[3:]
    ck = torch.load(src, weights_only=False)
    for o in opts:
        k, v = o.split("=", 1)
        if k == "frost_guard":
            ck[k] = [float(x) for x in v.split(",")]
        elif k in INT:
            ck[k] = int(v)
        else:
            ck[k] = float(v)
    torch.save(ck, dst)
    print(f"{dst}: {' '.join(opts)}")


if __name__ == "__main__":
    main()
