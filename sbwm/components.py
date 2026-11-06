"""Power components of the simulated building, read during data collection.

The reward only reports building totals (blowers, air conditioning, boiler gas,
pump). The simulator computes them device by device, so the collection reads
the parts, which the structured power heads of the world model learn:

    zone_reheat_w   (V,)  heat of each zone's reheat coil [W] (drawn from the boiler)
    ahu_fan_w       (K,)  supply and exhaust fans of each air handler [W]
    ahu_cool_w      (K,)  cooling electricity of each air handler [W]
    boiler_loss_w   ()    heat lost by the boiler tank to the outside [W]

The values are captured by wrapping the device methods the simulator calls
during the step and the reward computation, so nothing is computed twice.
"""

from __future__ import annotations

import numpy as np


class ComponentProbe:
    def __init__(self, env, schema):
        h = env.building.simulator._hvac
        system = h.air_handler
        ahus = system.ahus if hasattr(system, "ahus") else [system]
        by_id = {a.device_id(): a for a in ahus}
        self.ahus = [by_id[i] for i in schema.ahu_ids]
        self.vavs = [h.vavs.get(z) for z in schema.zone_ids]
        self.reheat = np.zeros(len(self.vavs), np.float32)
        self.cool = np.zeros(len(self.ahus), np.float32)
        self.loss = 0.0
        for i, v in enumerate(self.vavs):
            if v is not None:
                v.compute_reheat_energy_rate = self._wrap(v.compute_reheat_energy_rate, "reheat", i)
        for i, a in enumerate(self.ahus):
            a.compute_cooling_electrical_energy_rate = self._wrap(a.compute_cooling_electrical_energy_rate, "cool", i)
        src = h.hot_water_system._heat_source
        src.compute_thermal_dissipation_rate = self._wrap(src.compute_thermal_dissipation_rate, "loss", None)

    def _wrap(self, fn, name, i):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            if name == "loss":
                self.loss = float(out)
            else:
                getattr(self, name)[i] = float(out)
            return out

        return wrapped

    def read(self) -> dict:
        """Components of the step just simulated."""
        fan = np.array([a.compute_intake_fan_energy_rate() + a.compute_exhaust_fan_energy_rate() for a in self.ahus],
                       np.float32)
        return {"zone_reheat_w": self.reheat.copy(), "ahu_fan_w": fan, "ahu_cool_w": self.cool.copy(),
                "boiler_loss_w": np.float32(self.loss)}
