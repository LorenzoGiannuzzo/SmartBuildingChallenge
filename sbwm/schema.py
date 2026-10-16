"""Entity layout of one building, derived only from the policy metadata.

Everything here is built from ``building_meta_data`` (the dict the policy
receives at construction time on Kaggle), so the same code serves data
collection, world-model training and the final submitted policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

#Lorenzo Giannuzzo: per-zone observation suffixes, in the order used to build
#zone tokens. Missing fields are filled with NaN and masked downstream.
ZONE_OBS_FIELDS = (
    "zone_air_temperature_sensor",
    "zone_air_heating_temperature_setpoint_sensor",
    "zone_air_cooling_temperature_setpoint_sensor",
    "supply_air_damper_percentage_command",
    "reheat_valve_percentage_command",
    "supply_air_flowrate_setpoint",
)

AHU_OBS_FIELDS = (
    "supply_air_temperature_sensor",
    "supply_air_temperature_setpoint",
    "supply_air_static_pressure_sensor",
    "supply_air_static_pressure_setpoint",
    "supply_air_flowrate_sensor",
    "supply_air_flowrate_setpoint",
    "outside_air_flowrate_sensor",
    "recirculation_percentage_command",
    "cooling_request_count",
    "supply_fan_run_command",
    "supply_fan_speed_percentage_command",
    "discharge_fan_speed_percentage_command",
    "exhaust_fan_run_command",
    "supervisor_run_command",
    "outside_air_temperature_sensor",
)

HWS_OBS_FIELDS = (
    "supply_water_temperature_sensor",
    "supply_water_temperature_setpoint",
    "differential_pressure_sensor",
    "differential_pressure_setpoint",
    "heating_request_count",
    "run_status",
    "supervisor_run_command",
)

GLOBAL_OBS_FIELDS = ("comfort_mode_now", "comfort_mode_soon", "num_occupants")


def _natural_key(name: str) -> tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name))


@dataclass
class BuildingSchema:
    """Index tables mapping the flat observation/action vectors to entities."""

    ahu_ids: list[str]
    zone_ids: list[str]  # VAV ids, grouped by AHU
    zone_rooms: list[str]  # matching room ids used in reward info
    zone_ahu: np.ndarray  # (V,) AHU index of each zone
    zone_obs_idx: np.ndarray  # (V, len(ZONE_OBS_FIELDS)), -1 when missing
    ahu_obs_idx: np.ndarray  # (K, len(AHU_OBS_FIELDS)), -1 when missing
    hws_obs_idx: np.ndarray  # (len(HWS_OBS_FIELDS),), -1 when missing
    global_obs_idx: np.ndarray  # (len(GLOBAL_OBS_FIELDS),)
    hod_sin_idx: int
    hod_cos_idx: int
    dow_sin_idx: int
    dow_cos_idx: int
    # continuous action indices
    ahu_temp_act: np.ndarray  # (K,)
    ahu_pressure_act: np.ndarray  # (K,)
    ahu_recirc_act: np.ndarray  # (K,)
    zone_damper_act: np.ndarray  # (V,)
    zone_reheat_act: np.ndarray  # (V,)
    hws_temp_act: int
    hws_pressure_act: int
    # discrete action indices
    ahu_run_act: np.ndarray  # (K,)
    hws_run_act: int
    n_continuous: int
    n_discrete: int
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def num_ahus(self) -> int:
        return len(self.ahu_ids)

    @property
    def num_zones(self) -> int:
        return len(self.zone_ids)

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any]) -> "BuildingSchema":
        names = list(meta["observation_field_names"])
        pos = {n: i for i, n in enumerate(names)}

        assignments = meta["ahu_zone_assignments"]
        ahu_ids = sorted(assignments.keys(), key=_natural_key)
        zone_ids, zone_rooms, zone_ahu = [], [], []
        for a, ahu in enumerate(ahu_ids):
            for room in sorted(assignments[ahu], key=_natural_key):
                zone_rooms.append(room)
                zone_ids.append(room.replace("room_", "VAV_"))
                zone_ahu.append(a)

        zone_obs_idx = np.full((len(zone_ids), len(ZONE_OBS_FIELDS)), -1, np.int64)
        for z, vav in enumerate(zone_ids):
            for f, suffix in enumerate(ZONE_OBS_FIELDS):
                zone_obs_idx[z, f] = pos.get(f"{vav}_{suffix}", -1)

        ahu_obs_idx = np.full((len(ahu_ids), len(AHU_OBS_FIELDS)), -1, np.int64)
        for a, ahu in enumerate(ahu_ids):
            for f, suffix in enumerate(AHU_OBS_FIELDS):
                ahu_obs_idx[a, f] = pos.get(f"ahs_{ahu}_{suffix}", -1)

        hws_obs_idx = np.array([pos.get(f"hws_{s}", -1) for s in HWS_OBS_FIELDS], np.int64)
        global_obs_idx = np.array([pos.get(s, -1) for s in GLOBAL_OBS_FIELDS], np.int64)

        #Lorenzo Giannuzzo: the action layout is read from the metadata index
        #tables. pressure_indices carries the HWS differential pressure as its
        #last entry, after the K air-handler pressures.
        k = len(ahu_ids)
        pressure = list(meta["pressure_indices"])
        zone_actions = meta["zone_action_indices"]
        damper = [zone_actions.get(v, {}).get("supply_air_damper_percentage_command", -1) for v in zone_ids]
        reheat = [zone_actions.get(v, {}).get("reheat_valve_percentage_command", -1) for v in zone_ids]

        n_discrete = int(meta["n_discrete_action"])
        return cls(
            ahu_ids=ahu_ids,
            zone_ids=zone_ids,
            zone_rooms=zone_rooms,
            zone_ahu=np.array(zone_ahu, np.int64),
            zone_obs_idx=zone_obs_idx,
            ahu_obs_idx=ahu_obs_idx,
            hws_obs_idx=hws_obs_idx,
            global_obs_idx=global_obs_idx,
            hod_sin_idx=pos["hod_sin_000"],
            hod_cos_idx=pos["hod_cos_000"],
            dow_sin_idx=pos["dow_sin_000"],
            dow_cos_idx=pos["dow_cos_000"],
            ahu_temp_act=np.array(meta["temp_indices"][:k], np.int64),
            ahu_pressure_act=np.array(pressure[:k], np.int64),
            ahu_recirc_act=np.array(meta["recirculation_indices"][:k], np.int64),
            zone_damper_act=np.array(damper, np.int64),
            zone_reheat_act=np.array(reheat, np.int64),
            hws_temp_act=int(meta["water_indices"][0]) if meta["water_indices"] else -1,
            hws_pressure_act=int(pressure[k]) if len(pressure) > k else -1,
            ahu_run_act=np.arange(k, dtype=np.int64),
            hws_run_act=n_discrete - 1,
            n_continuous=int(meta["n_continuous_action"]),
            n_discrete=n_discrete,
        )

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for key, value in self.__dict__.items():
            out[key] = value.tolist() if isinstance(value, np.ndarray) else value
        return out
