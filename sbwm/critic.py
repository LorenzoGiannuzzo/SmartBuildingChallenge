"""Value function (critic) for DPC with long-horizon credit.

Backpropagation through the world model only gives credit a few hours ahead,
while the hardest scenarios are decided over a day or more (keeping the
thermal mass warm over a cold weekend, pre-cooling before a hot afternoon).
The critic estimates the discounted reward beyond the end of each imagined
rollout, so the policy is trained on rollout reward plus the value of the
state it leaves behind (lambda-returns, as in Dreamer and SHAC).

The critic is only used during training, so it is privileged: besides the
policy's inputs it sees the true current zone temperatures and summaries of
the future exogenous inputs of the episode (outdoor air, occupancy,
setpoints, tariffs up to two days ahead), which the policy cannot observe.
This removes most of the variance due to weather and schedule that a critic
limited to the policy's observations would have to average out.

Values are in units of mean per-step reward: the critic output ``y`` maps to
the discounted sum ``V = VALUE_SCALE * y / (1 - gamma)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sbwm.policy import AHU_IN, GLOB_IN, HWS_IN, N_OBS_HIST, ZONE_IN, build_inputs
from sbwm.wm_data import AHU_OAT
from sbwm.world_model import Block

FUT_WINDOWS = (12, 36, 72, 144, 288, 576)  # steps ahead: 1, 3, 6, 12, 24 and 48 hours
FUT_SERIES = 6  # outdoor air, occupied share, heating and cooling setpoints, electricity cost and carbon
ZONE_FUT_WINDOWS = (12, 72, 288)
GLOB_FUT = FUT_SERIES * len(FUT_WINDOWS)
ZONE_FUT = 1 + len(ZONE_FUT_WINDOWS)
ZONE_IN_C = ZONE_IN + 1 + ZONE_FUT + 1  # policy inputs, PI integral slot, future occupancy, true temperature
VALUE_SCALE = 0.01


def _window_means(x: torch.Tensor, windows) -> torch.Tensor:
    """Mean of x[t:t+w] for every t (clipped at the end), stacked on the last axis.

    x is (T, ...); the result is (T + 1, ..., len(windows)), the last row
    repeating the previous one so that indices up to T are valid.
    """
    t_len = x.shape[0]
    s = torch.cat([torch.zeros_like(x[:1]), torch.cumsum(x, 0)])
    idx = torch.arange(t_len)
    out = []
    for w in windows:
        end = torch.clamp(idx + w, max=t_len)
        m = (s[end] - s[idx]) / (end - idx).view(-1, *([1] * (x.dim() - 1))).to(x.dtype)
        out.append(torch.cat([m, m[-1:]]))
    return torch.stack(out, -1)


def future_features(ep, norm) -> tuple[torch.Tensor, torch.Tensor]:
    """(T + 1, GLOB_FUT) global and (T + 1, V, ZONE_FUT) zone features, cached on the episode."""
    cached = ep.__dict__.get("_critic_fut")
    if cached is not None:
        return cached
    t_len = ep.length
    oat = ((ep.aobs[:t_len, :, AHU_OAT] - norm.a_mean[AHU_OAT]) / norm.a_std[AHU_OAT]).mean(-1)
    occ = (ep.occ > 0).float()
    coef = ep.coef.float()
    cost = coef[:, 0] / coef[:, 0].mean().clamp_min(1e-12)
    co2 = coef[:, 2] / coef[:, 2].mean().clamp_min(1e-12)
    series = torch.stack([oat.float(), occ.mean(-1), ((ep.hsp - 295.0) / 5.0).mean(-1).float(),
                          ((ep.csp - 295.0) / 5.0).mean(-1).float(), cost - 1.0, co2 - 1.0], -1)
    glob = _window_means(series, FUT_WINDOWS).reshape(t_len + 1, -1)
    zone = torch.cat([torch.cat([occ, occ[-1:]]).unsqueeze(-1), _window_means(occ, ZONE_FUT_WINDOWS)], -1)
    ep.__dict__["_critic_fut"] = (glob, zone)
    return glob, zone


def critic_inputs(state: dict, ep, t: torch.Tensor, norm):
    """Token inputs of the critic for the world-model state before the decision at t."""
    obs_hist = state["temp_hist"][:, 1:1 + N_OBS_HIST]
    zin, ain, hin, gin = build_inputs(
        state["zobs"], state["aobs"], state["hobs"], ep.gobs[t], obs_hist, ep.static,
        state["zact_prev"], state["aact_prev"], state["hact_prev"], ep.sched, norm.dtemp_std)
    glob_fut, zone_fut = future_features(ep, norm)
    pi_int = state.get("pi_int")
    if pi_int is None:
        pi_int = torch.zeros(zin.shape[:2])
    true_now = (state["temp_hist"][:, 0] - 295.0) / 5.0
    zin = torch.cat([zin, pi_int.unsqueeze(-1), zone_fut[t], true_now.unsqueeze(-1)], -1)
    gin = torch.cat([gin, glob_fut[t]], -1)
    return zin, ain, hin, gin


class _Trunk(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, ff: int):
        super().__init__()
        self.enc_zone = nn.Sequential(nn.Linear(ZONE_IN_C, d), nn.GELU(), nn.Linear(d, d))
        self.enc_ahu = nn.Sequential(nn.Linear(AHU_IN, d), nn.GELU(), nn.Linear(d, d))
        self.enc_hws = nn.Sequential(nn.Linear(HWS_IN, d), nn.GELU(), nn.Linear(d, d))
        self.enc_glob = nn.Sequential(nn.Linear(GLOB_IN + GLOB_FUT, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.zeros(4, d))
        self.blocks = nn.ModuleList(Block(d, heads, ff) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        #Lorenzo Giannuzzo: the value is a building-level quantity (the reward is
        #normalized per building), read out from the global token plus the mean
        #of per-zone contributions, so it does not grow with the zone count.
        self.head_glob = nn.Linear(d, 1)
        self.head_zone = nn.Linear(d, 1)
        for head in (self.head_glob, self.head_zone):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.head_glob.bias.fill_(-2.0)  # about -0.02 per step, a typical reward

    def forward(self, zin, ain, hin, gin, rel):
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
        return self.head_glob(x[:, -1]).squeeze(-1) + self.head_zone(x[:, :v]).squeeze(-1).mean(1)


class CriticNet(nn.Module):
    """Ensemble of independent entity-token value networks; output (E, B)."""

    def __init__(self, d: int = 64, layers: int = 3, heads: int = 4, ff: int = 128, ens: int = 2):
        super().__init__()
        self.members = nn.ModuleList(_Trunk(d, layers, heads, ff) for _ in range(ens))

    def forward(self, zin, ain, hin, gin, rel):
        return torch.stack([m(zin, ain, hin, gin, rel) for m in self.members])


def critic_y(critic, states: list[dict], ts: list[torch.Tensor], ep, norm, rel) -> torch.Tensor:
    """Critic outputs (E, n, B) for n states of the same episode, batched together."""
    b = ts[0].shape[0]
    keys = states[0].keys()
    st = {k: torch.cat([s[k] for s in states]) for k in keys}
    t = torch.cat(ts)
    y = critic(*critic_inputs(st, ep, t, norm), rel)
    return y.view(y.shape[0], len(states), b)


def lambda_returns(r: torch.Tensor, v: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """Lambda-returns (H, B) from rewards r (H, B) and values v (H, B) of the next states."""
    out = []
    nxt = v[-1]
    for i in range(r.shape[0] - 1, -1, -1):
        boot = v[i] if i == r.shape[0] - 1 else (1 - lam) * v[i] + lam * nxt
        nxt = r[i] + gamma * boot
        out.append(nxt)
    return torch.stack(out[::-1])


def to_sum(y: torch.Tensor, gamma: float) -> torch.Tensor:
    return VALUE_SCALE * y / (1.0 - gamma)


def to_y(v: torch.Tensor, gamma: float) -> torch.Tensor:
    return v * (1.0 - gamma) / VALUE_SCALE


def critic_loss(critic, target, states, ts, r, ep, norm, rel, gamma, lam, sample, rng):
    """Regression of the critic on lambda-return targets from the target critic.

    ``states`` holds s_0..s_H of a rollout and ``r`` (H, B) its rewards;
    gradients do not reach the world model or the policy.
    """
    h = r.shape[0]
    with torch.no_grad():
        det = [{k: x.detach() for k, x in s.items()} for s in states]
        v_next = to_sum(critic_y(target, det[1:], ts[1:], ep, norm, rel).mean(0), gamma)
        ret = lambda_returns(r.detach(), v_next, gamma, lam)
    n = max(1, int(round(sample * h)))
    idx = sorted(rng.choice(h, n, replace=False).tolist())
    y = critic_y(critic, [det[i] for i in idx], [ts[i] for i in idx], ep, norm, rel)
    tgt = to_y(ret[idx], gamma).unsqueeze(0).expand_as(y)
    return F.smooth_l1_loss(y, tgt), ret
