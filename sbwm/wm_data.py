"""Episode loading and tensor layout for world-model training.

Timing convention (verified on collected data): ``obs[k]`` is the observation
returned by step ``k`` (``obs[0]`` from reset) and ``zone_temp[k]`` is the zone
temperature used by the reward of step ``k + 1``. The observed zone temperature
lags the rewarded one by one step: ``obs[k]`` zone temperature equals
``zone_temp[k - 2]``.

For a decision at index ``t`` (action ``act[t]``, applied by step ``t + 1``):
  * true pre-action zone temperature: ``zone_temp[t - 1]``
  * post-action zone temperature:     ``zone_temp[t]``  (scored by the reward)
  * plant observation before acting:  ``obs[t]``
  * plant observation after acting:   ``obs[t + 1]``
  * energy rates of the step:         ``*_w[t]``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sbwm.geometry import zone_static_features
from sbwm.schema import AHU_OBS_FIELDS, BuildingSchema

#Lorenzo Giannuzzo: exogenous fields are taken from data during rollouts; the
#world model predicts every other field of the next observation.
ZONE_DYN = (3, 4, 5)  # damper, reheat, airflow setpoint (indices in ZONE_OBS_FIELDS)
ZONE_EXO = (1, 2)  # heating and cooling setpoint sensors
AHU_OAT = AHU_OBS_FIELDS.index("outside_air_temperature_sensor")
AHU_DYN = tuple(i for i in range(len(AHU_OBS_FIELDS)) if i != AHU_OAT)
HIST = 3  # past zone-temperature differences fed to the model

ELEC_SCALE_PER_AHU = 80000.0
GAS_SCALE_PER_AHU = 205000.0


@dataclass
class Episode:
    name: str
    schema: BuildingSchema
    static: torch.Tensor  # (V, S)
    zobs: torch.Tensor  # (T+1, V, 6) raw zone observation fields
    aobs: torch.Tensor  # (T+1, K, 15)
    hobs: torch.Tensor  # (T+1, 7)
    gobs: torch.Tensor  # (T+1, G) comfort now/soon, occupants per zone, time features
    temp: torch.Tensor  # (T, V) rewarded zone temperature [K]
    occ: torch.Tensor  # (T, V)
    hsp: torch.Tensor  # (T, V)
    csp: torch.Tensor  # (T, V)
    zact: torch.Tensor  # (T, V, 2) damper, reheat command
    aact: torch.Tensor  # (T, K, 4) run, supply temp, pressure, recirculation
    hact: torch.Tensor  # (T, 3) run, water temp, differential pressure
    power: torch.Tensor  # (T, 2) electric and gas rate, scaled per AHU
    coef: torch.Tensor  # (T, 4) elec cost, gas cost, elec co2, gas co2 per W
    reward: torch.Tensor  # (T,)
    reward_meta: dict

    @property
    def length(self) -> int:
        return self.temp.shape[0]


def _static_for(scenario: Path, meta: dict, schema: BuildingSchema) -> np.ndarray:
    cache = scenario / "zone_static.npy"
    if cache.exists():
        return np.load(cache)
    meta = dict(meta)
    meta["floor_plan"] = np.load(scenario / "floor_plan.npy", allow_pickle=True)
    meta["zone_map"] = np.load(scenario / "zone_map.npy", allow_pickle=True)
    feats = zone_static_features(meta, schema)
    np.save(cache, feats)
    return feats


def load_episode(path: Path) -> Episode:
    z = np.load(path)
    header = json.loads(str(z["header"]))
    sd = header["schema"]
    schema = BuildingSchema(**{k: (np.asarray(v) if isinstance(v, list) and k not in (
        "ahu_ids", "zone_ids", "zone_rooms") else v) for k, v in sd.items()})
    meta = header["meta"]
    meta["ahu_zone_assignments"] = None
    static = _static_for(Path(header["scenario"]), meta, schema)

    obs = z["obs"].astype(np.float32)
    k, v = schema.num_ahus, schema.num_zones

    def take(idx):
        idx = np.asarray(idx)
        out = np.full(obs.shape[:1] + idx.shape, np.nan, np.float32)
        ok = idx >= 0
        out[:, ok] = obs[:, idx[ok]]
        return out

    zobs = take(schema.zone_obs_idx)
    aobs = take(schema.ahu_obs_idx)
    hobs = take(schema.hws_obs_idx)
    g = take(schema.global_obs_idx)
    g[:, 2] = g[:, 2] / max(v, 1)
    hod = np.stack([obs[:, schema.hod_sin_idx], obs[:, schema.hod_cos_idx],
                    obs[:, schema.dow_sin_idx], obs[:, schema.dow_cos_idx]], -1)
    gobs = np.concatenate([g, hod], -1)

    ac, ad = z["act_c"].astype(np.float32), z["act_d"].astype(np.float32)
    zact = np.stack([np.where(schema.zone_damper_act >= 0, ac[:, schema.zone_damper_act], -1),
                     np.where(schema.zone_reheat_act >= 0, ac[:, schema.zone_reheat_act], -1)], -1)
    aact = np.stack([ad[:, schema.ahu_run_act] * 2 - 1, ac[:, schema.ahu_temp_act],
                     ac[:, schema.ahu_pressure_act], ac[:, schema.ahu_recirc_act]], -1)
    hact = np.stack([ad[:, schema.hws_run_act] * 2 - 1,
                     ac[:, schema.hws_temp_act] if schema.hws_temp_act >= 0 else -np.ones(len(ac)),
                     ac[:, schema.hws_pressure_act] if schema.hws_pressure_act >= 0 else -np.ones(len(ac))], -1)

    elec = z["blower_w"] + np.abs(z["ac_w"]) + z["pump_w"]
    power = np.stack([elec / (ELEC_SCALE_PER_AHU * k), z["gas_w"] / (GAS_SCALE_PER_AHU * k)], -1)
    coef = np.stack([z["elec_cost_w"], z["gas_cost_w"], z["elec_co2_w"], z["gas_co2_w"]], -1)

    t = lambda a: torch.as_tensor(np.nan_to_num(np.asarray(a, np.float32)))
    return Episode(
        name=path.name, schema=schema, static=t(static), zobs=t(zobs), aobs=t(aobs),
        hobs=t(hobs), gobs=t(gobs), temp=t(z["zone_temp"]), occ=t(z["zone_occ"]),
        hsp=t(z["zone_hsp"]), csp=t(z["zone_csp"]), zact=t(zact), aact=t(aact),
        hact=t(hact), power=t(power), coef=t(coef.astype(np.float64) * 1e6),
        reward=t(z["reward"]), reward_meta=header["reward"])


@dataclass
class Normalizer:
    """Per-field affine scaling fitted on training episodes."""

    z_mean: torch.Tensor
    z_std: torch.Tensor
    a_mean: torch.Tensor
    a_std: torch.Tensor
    h_mean: torch.Tensor
    h_std: torch.Tensor
    dtemp_std: float

    @classmethod
    def fit(cls, episodes: list[Episode]) -> "Normalizer":
        def stats(x):
            m, s = x.mean(0), x.std(0)
            return m, torch.where(s < 1e-6, torch.ones_like(s), s)

        zs = torch.cat([e.zobs.reshape(-1, e.zobs.shape[-1]) for e in episodes])
        a_s = torch.cat([e.aobs.reshape(-1, e.aobs.shape[-1]) for e in episodes])
        hs = torch.cat([e.hobs for e in episodes])
        dt = torch.cat([(e.temp[1:] - e.temp[:-1]).reshape(-1) for e in episodes])
        zm, zsd = stats(zs)
        #Lorenzo Giannuzzo: all temperatures share one scale so that
        #differences between them keep their physical meaning.
        zm[:3], zsd[:3] = 295.0, 5.0
        zm[0], zsd[0] = 0.0, 1.0  # zone temperature is already normalized in obs
        am, asd = stats(a_s)
        hm, hsd = stats(hs)
        return cls(zm, zsd, am, asd, hm, hsd, float(dt.std()))

    def state_dict(self) -> dict:
        return {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        return cls(**{k: (torch.tensor(v) if isinstance(v, list) else v) for k, v in d.items()})


def load_dir(path: Path, limit: int | None = None) -> list[Episode]:
    files = sorted(Path(path).glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    return [load_episode(f) for f in files[:limit]]
