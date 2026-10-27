"""Build the Kaggle submission: a dataset folder with the policy code and weights,
and the notebook that serves the policy to the competition gateway.

Usage (from the repository root):
    python scripts/build_kaggle_bundle.py --weights policies/dpc_v4.pt

Outputs in kaggle/:
    sbwm-policy/          upload as a private Kaggle dataset (or sbwm-policy.zip)
    submission.ipynb      import as a Kaggle notebook, attach the dataset, submit
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MODULES = ("__init__.py", "policy.py", "world_model.py", "wm_data.py", "schema.py", "geometry.py")

CELL = '''"""Entity-token transformer policy trained with Differentiable Predictive Control
through a learned world model (Lorenzo Giannuzzo, Politecnico di Torino).
"""

import glob
import os
import sys

COMPETITION_DATA_FOLDER = '/kaggle/input/competitions/neurips-2026-smart-buildings/'
sys.path.append(os.path.join(COMPETITION_DATA_FOLDER, 'kaggle_evaluation'))
sys.path.append(COMPETITION_DATA_FOLDER)

#Lorenzo Giannuzzo: the dataset mount point depends on the Kaggle layout, so
#the bundle is located by its marker file instead of a fixed path. A dataset
#can hold several bundles, so the marker must also name the wanted weights.
WANTED = '__WANTED__'
marker = [m for m in glob.glob('/kaggle/input/**/SBWM_POLICY_BUNDLE', recursive=True)
          if open(m).read().strip() == 'weights: ' + WANTED]
assert len(marker) == 1, f'expected one bundle with {WANTED}, found {marker}'
BUNDLE = os.path.dirname(marker[0])
print('bundle:', BUNDLE, 'weights:', WANTED, flush=True)
sys.path.insert(0, BUNDLE)

import numpy as np
from kaggle_evaluation import smart_buildings_inference_server
from sbwm.policy import DPCPolicy

WEIGHTS = os.path.join(BUNDLE, 'policy.pt')


class TransferPolicy(smart_buildings_inference_server.Policy):
    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.inner = DPCPolicy(building_meta_data, action_spec, weights=WEIGHTS)
        c, d = action_spec['continuous_action'], action_spec['discrete_action']
        self.c_low, self.c_high = c.low, c.high
        self.d_low, self.d_high = d.low, d.high
        print(f'Controlling a building with {self.inner.v} zones', flush=True)

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        a = self.inner.get_action(obs)
        #Lorenzo Giannuzzo: actions are clipped into the spec bounds, since the
        #gateway rejects out-of-range actions.
        return {
            'continuous_action': np.clip(np.asarray(a['continuous_action'], np.float32),
                                         self.c_low, self.c_high).astype(np.float32),
            'discrete_action': np.clip(np.asarray(a['discrete_action']),
                                       self.d_low, self.d_high).astype(np.int32),
        }


smart_buildings_inference_server.SmartBuildingsInferenceServer(TransferPolicy).run(
    competition_data_folder=COMPETITION_DATA_FOLDER,
)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="policies/dpc_v4.pt")
    ap.add_argument("--out", default="kaggle")
    args = ap.parse_args()
    out = Path(args.out)
    bundle = out / "sbwm-policy"
    shutil.rmtree(bundle, ignore_errors=True)
    (bundle / "sbwm").mkdir(parents=True)
    for m in MODULES:
        shutil.copy(Path("sbwm") / m, bundle / "sbwm" / m)
    shutil.copy(args.weights, bundle / "policy.pt")
    (bundle / "SBWM_POLICY_BUNDLE").write_text(f"weights: {Path(args.weights).name}\n")
    shutil.make_archive(str(out / "sbwm-policy"), "zip", bundle)
    #Lorenzo Giannuzzo: a copy named after the weights, so that each upload
    #gets its own folder in the dataset.
    shutil.copy(out / "sbwm-policy.zip", out / f"sbwm-policy-{Path(args.weights).stem}.zip")
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "id": "sbwm-policy", "metadata": {},
                   "outputs": [], "source": CELL.replace("__WANTED__", Path(args.weights).name)
                   .splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (out / "submission.ipynb").write_text(json.dumps(nb, indent=1))
    print(f"bundle: {bundle} (+ .zip), notebook: {out / 'submission.ipynb'}, weights: {args.weights}")


if __name__ == "__main__":
    main()
