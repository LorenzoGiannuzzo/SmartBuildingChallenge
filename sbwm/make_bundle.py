"""Build the Kaggle bundle of a policy checkpoint.

The bundle holds the checkpoint as policy.pt, the marker file read by the
submission notebook and the modules the deployment wrapper imports.

Usage:
    python -m sbwm.make_bundle policies/v13_mem_2wm.pt kaggle/sbwm-policy-v13_mem_2wm.zip
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

MODULES = ("__init__.py", "geometry.py", "schema.py", "policy.py", "physics.py", "world_model.py", "wm_data.py")


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    pkg = Path(__file__).resolve().parent
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(src, "policy.pt")
        z.writestr("SBWM_POLICY_BUNDLE", f"weights: {src.name}\n")
        for m in MODULES:
            z.write(pkg / m, f"sbwm/{m}")
    print(f"{dst}: {src.name} with {len(MODULES)} modules")


if __name__ == "__main__":
    main()
