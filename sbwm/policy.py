"""Entity-token transformer policy and its deployment wrapper.

The same feature construction is used in DPC training (inputs taken from the
world-model state) and at deployment (inputs taken from the raw observation),
so the policy never sees anything at training time that it will not see on
Kaggle. In particular the rewarded zone temperature of the current step is
hidden: the policy only receives observed zone temperatures, which lag by one
step.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sbwm.world_model import Block, relation_matrix

N_OBS_HIST = 3  # observed zone temperatures kept (current plus two past)
ZONE_IN = 6 + (N_OBS_HIST - 1) + 7 + 2
AHU_IN = 15 + 4
HWS_IN = 7 + 3
GLOB_IN = 7 + 3


def schedule_features(gobs: torch.Tensor, morning: float, evening: float) -> torch.Tensor:
    """Hours to the next comfort start, hours left in comfort, weekday flag.

    Decoded from the cyclic time features and the schedule hours that are part
    of the building metadata. Holidays are not visible here; the comfort-mode
    flags in the observation cover them.
    """
    hs, hc, ds, dc = gobs[..., 3], gobs[..., 4], gobs[..., 5], gobs[..., 6]
    hour = torch.remainder(torch.atan2(hs, hc), 2 * math.pi) * 24.0 / (2 * math.pi)
    day = torch.floor(torch.remainder(torch.atan2(ds, dc), 2 * math.pi) * 7.0 / (2 * math.pi) + 1e-4)
    weekday = (day < 5).float()
    to_start = torch.full_like(hour, 24.0 * 7)
    for k in range(8):
        cand = torch.remainder(day + k, 7)
        ok = (cand < 5) & ((k > 0) | (hour < morning))
        hours = k * 24.0 + morning - hour
        to_start = torch.where(ok, torch.minimum(to_start, hours), to_start)
    in_comfort = (weekday > 0) & (hour >= morning) & (hour < evening)
    left = torch.where(in_comfort, evening - hour, torch.zeros_like(hour))
    return torch.stack([torch.clamp(to_start, max=72.0) / 24.0, left / 12.0, weekday], -1)


#Lorenzo Giannuzzo: default constants of the zone PI layer of the hybrid
#policy (margin and proportional band in K, integral gain per step, integral
#decay per step inside the band).
PI_DEFAULTS = {"margin": 0.5, "full": 2.0, "ki": 0.02, "decay": 0.9}
PI_WARMUP = 12  # steps run before a DPC rollout so the integrals are not cold


def _inv_softplus(y: float) -> float:
    return float(math.log(math.expm1(y)))


def hybrid_zone_actions(raw: torch.Tensor, zobs_n: torch.Tensor, pi_int: torch.Tensor, pi: dict):
    """Zone damper and reheat commands from a differentiable PI per zone.

    ``raw`` (B, V, 5) are the transformer's zone outputs: margins of the inner
    band from the heating and from the cooling setpoint, gain, and residuals
    on damper and reheat. ``zobs_n`` are the normalized zone observations
    (temperature and both setpoint sensors share the 295 K / 5 K scale) and
    ``pi_int`` (B, V) the integral states. Returns the zone actions (B, V, 2)
    in [-1, 1] and the new integrals.
    """
    temp = zobs_n[..., 0] * 5.0 + 295.0
    hsp = zobs_n[..., 1] * 5.0 + 295.0
    csp = zobs_n[..., 2] * 5.0 + 295.0
    width = torch.clamp(csp - hsp, min=0.5)
    #Lorenzo Giannuzzo: the margins are in kelvin and start at the tuned PI
    #margin; each is capped so that the inner band never inverts.
    cap = (width - 0.5) / 2
    c0 = _inv_softplus(pi["margin"]) if pi["margin"] > 0 else -6.0
    m_lo = torch.minimum(F.softplus(raw[..., 0] + c0), cap)
    m_hi = torch.minimum(F.softplus(raw[..., 1] + c0), cap)
    lo, hi = hsp + m_lo, csp - m_hi
    #Lorenzo Giannuzzo: positive error means too cold, as in the baselines;
    #it is zero inside the inner band.
    err = F.relu(lo - temp) - F.relu(temp - hi)
    inside = (err == 0).float()
    new_int = inside * pi_int * pi["decay"] + (1 - inside) * torch.clamp(pi_int + pi["ki"] * err, -1.0, 1.0)
    gain = torch.exp(torch.tanh(raw[..., 2]))
    u = torch.clamp(err * gain / pi["full"] + new_int, -1.0, 1.0)
    damper = 2.0 * u.abs() - 1.0
    reheat = 2.0 * F.relu(u) - 1.0
    zact = torch.stack([damper, reheat], -1) + torch.tanh(raw[..., 3:5])
    return torch.clamp(zact, -1.0, 1.0), new_int


class PolicyNet(nn.Module):
    def __init__(self, d: int = 64, layers: int = 3, heads: int = 4, ff: int = 128,
                 hybrid: bool = False, pi: dict | None = None):
        super().__init__()
        #Lorenzo Giannuzzo: the hybrid variant sees each zone's PI integral as
        #one extra input and outputs the PI supervision instead of the
        #actuator commands.
        self.hybrid = hybrid
        self.pi = {**PI_DEFAULTS, **(pi or {})}
        self.enc_zone = nn.Sequential(nn.Linear(ZONE_IN + int(hybrid), d), nn.GELU(), nn.Linear(d, d))
        self.enc_ahu = nn.Sequential(nn.Linear(AHU_IN, d), nn.GELU(), nn.Linear(d, d))
        self.enc_hws = nn.Sequential(nn.Linear(HWS_IN, d), nn.GELU(), nn.Linear(d, d))
        self.enc_glob = nn.Sequential(nn.Linear(GLOB_IN, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.zeros(4, d))
        self.blocks = nn.ModuleList(Block(d, heads, ff) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head_zone = nn.Linear(d, 5 if hybrid else 2)  # PI supervision, or damper and reheat
        self.head_ahu = nn.Linear(d, 4)  # run logit, supply temp, pressure, recirculation
        self.head_hws = nn.Linear(d, 3)  # run logit, water temp, differential pressure
        for head in (self.head_zone, self.head_ahu, self.head_hws):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
        #Lorenzo Giannuzzo: plant switched on at initialization, so that the
        #first rollouts start from a conditioned building rather than an idle
        #one and the gradients are informative from the first iteration.
        with torch.no_grad():
            self.head_ahu.bias[0] = 2.0
            self.head_hws.bias[0] = 2.0

    def forward(self, zin, ain, hin, gin, rel, hard: bool = True):
        v, k = zin.shape[1], ain.shape[1]
        x = torch.cat([
            self.enc_zone(zin) + self.type_emb[0],
            self.enc_ahu(ain) + self.type_emb[1],
            (self.enc_hws(hin) + self.type_emb[2]).unsqueeze(1),
            (self.enc_glob(gin) + self.type_emb[3]).unsqueeze(1),
        ], 1)
        for blk in self.blocks:
            x = blk(x, rel)
        x = self.ln(x)
        oz = self.head_zone(x[:, :v]) if self.hybrid else torch.tanh(self.head_zone(x[:, :v]))
        oa = self.head_ahu(x[:, v:v + k])
        oh = self.head_hws(x[:, v + k])

        def run(logit):
            #Lorenzo Giannuzzo: straight-through estimator, the forward pass
            #uses the hard on/off decision and the gradient flows through the
            #sigmoid probability.
            p = torch.sigmoid(logit)
            if not hard:
                return 2 * p - 1
            h = (p > 0.5).float()
            return 2 * (h + p - p.detach()) - 1

        aact = torch.cat([run(oa[..., :1]), torch.tanh(oa[..., 1:])], -1)
        hact = torch.cat([run(oh[:, :1]), torch.tanh(oh[:, 1:])], -1)
        return oz, aact, hact


def build_inputs(zobs_n, aobs_n, hobs_n, gobs, obs_hist, static, zact_prev, aact_prev,
                 hact_prev, sched, dtemp_std, pi_int=None):
    """Token inputs. obs_hist is (B, N_OBS_HIST, V) observed zone temperatures [K]."""
    b = zobs_n.shape[0]
    diffs = (obs_hist[:, :-1] - obs_hist[:, 1:]) / dtemp_std
    parts = [zobs_n, diffs.permute(0, 2, 1), static.expand(b, -1, -1), zact_prev]
    if pi_int is not None:
        parts.append(pi_int.unsqueeze(-1))
    zin = torch.cat(parts, -1)
    ain = torch.cat([aobs_n, aact_prev], -1)
    hin = torch.cat([hobs_n, hact_prev], -1)
    gin = torch.cat([gobs, schedule_features(gobs, *sched)], -1)
    return zin, ain, hin, gin


class DPCPolicy:
    """Competition interface: ``__init__(building_meta_data, action_spec)``,
    ``get_action(obs)``. Weights are read from ``SBWM_POLICY`` or the default
    path next to this file."""

    name = "dpc"
    DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "wm_runs" / "policy" / "policy.pt"

    def __init__(self, building_meta_data: Mapping[str, Any], action_spec: Any = None,
                 weights: str | None = None) -> None:
        import os

        from sbwm.geometry import zone_static_features
        from sbwm.schema import BuildingSchema
        from sbwm.wm_data import Normalizer

        torch.set_num_threads(1)
        path = Path(weights or os.environ.get("SBWM_POLICY", self.DEFAULT_WEIGHTS))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = PolicyNet(**ck["policy_args"])
        self.net.load_state_dict(ck["policy"])
        self.net.eval()
        self.norm = Normalizer.from_state_dict(ck["norm"])
        meta = building_meta_data
        self.s = BuildingSchema.from_meta(meta)
        self.static = torch.as_tensor(zone_static_features(meta, self.s)).unsqueeze(0)
        self.rel = relation_matrix(torch.as_tensor(self.s.zone_ahu), self.s.num_ahus)
        self.sched = (float(meta["morning_start_hour"]), float(meta["evening_start_hour"]))
        self.v = self.s.num_zones
        self.hist = None
        self.zact_prev = -torch.ones(1, self.v, 2)
        self.aact_prev = -torch.ones(1, self.s.num_ahus, 4)
        self.hact_prev = -torch.ones(1, 3)
        self.pi_int = torch.zeros(1, self.v) if self.net.hybrid else None

    def _take(self, obs, idx):
        idx = np.asarray(idx)
        out = np.zeros(idx.shape, np.float32)
        ok = idx >= 0
        out[ok] = obs[idx[ok]]
        return torch.as_tensor(out).unsqueeze(0)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        s, n = self.s, self.norm
        obs = np.nan_to_num(np.asarray(obs, np.float32))
        zobs = self._take(obs, s.zone_obs_idx)
        aobs = self._take(obs, s.ahu_obs_idx)
        hobs = self._take(obs, s.hws_obs_idx)
        g = self._take(obs, s.global_obs_idx)
        g[..., 2] = g[..., 2] / max(self.v, 1)
        hod = torch.tensor([[obs[s.hod_sin_idx], obs[s.hod_cos_idx], obs[s.dow_sin_idx],
                             obs[s.dow_cos_idx]]])
        gobs = torch.cat([g, hod], -1)
        temp_k = zobs[..., 0] * 5.0 + 295.0
        if self.hist is None:
            self.hist = temp_k.unsqueeze(1).repeat(1, N_OBS_HIST, 1)
        else:
            self.hist = torch.cat([temp_k.unsqueeze(1), self.hist[:, :-1]], 1)
        zobs_n = (zobs - n.z_mean) / n.z_std
        zin, ain, hin, gin = build_inputs(
            zobs_n, (aobs - n.a_mean) / n.a_std, (hobs - n.h_mean) / n.h_std,
            gobs, self.hist, self.static, self.zact_prev, self.aact_prev, self.hact_prev,
            self.sched, n.dtemp_std, self.pi_int)
        zact, aact, hact = self.net(zin, ain, hin, gin, self.rel)
        if self.net.hybrid:
            zact, self.pi_int = hybrid_zone_actions(zact, zobs_n, self.pi_int, self.net.pi)
        self.zact_prev, self.aact_prev, self.hact_prev = zact, aact, hact
        return to_env_action(s, zact[0].numpy(), aact[0].numpy(), hact[0].numpy())


def to_env_action(s, zact, aact, hact) -> dict[str, np.ndarray]:
    c = np.full(s.n_continuous, -1.0, np.float32)
    d = np.zeros(s.n_discrete, np.int32)
    ok = s.zone_damper_act >= 0
    c[s.zone_damper_act[ok]] = zact[ok, 0]
    ok = s.zone_reheat_act >= 0
    c[s.zone_reheat_act[ok]] = zact[ok, 1]
    d[s.ahu_run_act] = (aact[:, 0] > 0).astype(np.int32)
    c[s.ahu_temp_act] = aact[:, 1]
    c[s.ahu_pressure_act] = aact[:, 2]
    c[s.ahu_recirc_act] = aact[:, 3]
    d[s.hws_run_act] = int(hact[0] > 0)
    if s.hws_temp_act >= 0:
        c[s.hws_temp_act] = hact[1]
    if s.hws_pressure_act >= 0:
        c[s.hws_pressure_act] = hact[2]
    return {"continuous_action": np.clip(c, -1, 1), "discrete_action": d}
