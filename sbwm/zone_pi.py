"""Zone-level PI controller on an inner comfort band (no learning).

Built on the shipped ``pid_water`` baseline, which already commands every VAV
from its own zone temperature but only acts once a zone has left the band, so
zones settle on the band edges and slip out of it. Three changes:

  * each zone is judged against its own setpoint sensors (the band the reward
    uses) shrunk by ``margin`` kelvin on both sides, so the controller acts
    before the reward starts counting;
  * each zone keeps an integral of its error, so a zone that a proportional
    command cannot hold (for example one with a large exterior wall) receives
    progressively more air or reheat;
  * the air handler loop follows the mean zone error on the same inner band
    (``ahu="mean"``) or its worst zone (``ahu="worst"``).

It is the floor the learned hybrid policy has to beat.

Parameters are passed as a comma-separated string, for example
``"margin=0.5,ki=0.02,ahu=mean"``, so that ``sbwm.eval_policy`` can sweep them.
"""

from __future__ import annotations

import numpy as np

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy


def parse_params(text: str | None) -> dict:
    out = {}
    for part in (text or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class ZonePIPolicy(PIDWaterResetPolicy):
    USE_SENSOR_BAND = True

    def __init__(self, building_meta_data, action_spec, params: str | None = None):
        p = parse_params(params)
        self.margin = float(p.get("margin", 0.5))
        self.zone_ki = float(p.get("ki", 0.02))
        self.decay = float(p.get("decay", 0.9))
        self.ahu_mode = p.get("ahu", "mean")
        super().__init__(building_meta_data, action_spec)
        self.zone_full_k = float(p.get("full", 2.0))
        self.zone_damper_full_k = self.zone_full_k
        self.REHEAT_FULL_K = self.zone_full_k
        self._zone_int: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "zone_pi"

    def _zone_band(self, obs, vav_id, low, high):
        lo, hi = super()._zone_band(obs, vav_id, low, high)
        #Lorenzo Giannuzzo: the band is shrunk only while it stays at least
        #0.5 K wide, so a narrow band is never inverted.
        m = min(self.margin, max(0.0, (hi - lo - 0.5) / 2))
        return lo + m, hi - m

    def _command_zones(self, continuous, zones, obs, low, high) -> None:
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, low, high)
            err = self._deadband_error(self._denormalize_zone_temp(obs[obs_idx]), lo, hi)
            i = self._zone_int.get(vav_id, 0.0)
            #Lorenzo Giannuzzo: positive error means the zone is too cold. The
            #integral grows while the zone is outside the inner band and decays
            #inside it, so it cannot wind up during long eco periods.
            i = float(np.clip(i + self.zone_ki * err, -1.0, 1.0)) if err != 0.0 else i * self.decay
            self._zone_int[vav_id] = i
            u = float(np.clip(err / self.zone_full_k + i, -1.0, 1.0))
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * abs(u) - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 2.0 * max(0.0, u) - 1.0

    def _ahu_error(self, obs, ctrl, low, high) -> float:
        errs = []
        for vav_id, obs_idx, _ in ctrl["zones"]:
            lo, hi = self._zone_band(obs, vav_id, low, high)
            errs.append(self._deadband_error(self._denormalize_zone_temp(obs[obs_idx]), lo, hi))
        if not errs:
            return 0.0
        if self.ahu_mode == "worst":
            return float(max(errs, key=abs))
        return float(np.mean(errs))
