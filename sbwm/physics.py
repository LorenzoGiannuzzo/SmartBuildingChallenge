"""Physics-based zone features of the policy.

Each zone carries a small state, updated at every decision from the observed
zone temperature, the outdoor air, the heating setpoint and the previous
commands, both inside the DPC rollouts (temperatures predicted by the world
model) and at deployment (observed temperatures):

    ts          observed temperature smoothed over about 15 min [K], against
                the sensor noise (0.3-0.6 K) of the official scenarios
    buffer      last 48 smoothed temperatures (most recent first) [K]
    hourly      smoothed temperature sampled every hour, last 49 samples [K]
    n           decisions taken so far (sets the hourly sampling)
    inv_tau     estimated inverse time constant of the free-floating zone [1/h]
    rec         estimated recovery rate while heating at full reheat [K/h]
    hsp         highest heating setpoint seen, the comfort setpoint [K]
    free, heat  consecutive steps in free floating and in recovery

From this state and the current setpoints the policy receives, per zone
(PHYS_IN = 14):

    1. indoor-outdoor difference (T - T_out) / 20
    2-4. temperature change over 30 min, 1 h and 4 h [K] / 1, 2, 4
    5-7. temperature change over 12, 24 and 48 h [K] / 6, 8, 10, from the
       hourly samples (the lag is exact to within one hour)
    8. inv_tau * 50 (about 1 for a 50 h time constant)
    9. rec / 3
    10. (T_ff - hsp) / 5, with T_ff the free-floating temperature expected at
        the next comfort start, T_out + (T - T_out) exp(-dt * inv_tau)
    11. preheat slack (dt - max(hsp - T_ff, 0) / rec) / 24, in [-1, 3]
    12. (current heating setpoint - T) / 5
    13. (current cooling setpoint - T) / 5
    14. (hsp - T_out) / 20, the outdoor difference at the comfort setpoint

Features 12-14 follow Brandi et al. (Energy and Buildings, 2020), where
setpoint-referenced temperature differences generalized better across
setpoints and schedules than absolute temperatures.

where T is the smoothed temperature. The estimates start from priors at the
beginning of an episode and are refined with exponential averages over valid
hours (free floating: reheat closed and air handler off for at least one hour;
recovery: zone more than 0.5 K below the setpoint with reheat open and air
handler on for one hour). The state carries no gradient; the features depend
on the current temperature, through which the gradient of the DPC objective
flows.
"""

from __future__ import annotations

import math

import numpy as np
import torch

PHYS_IN = 14
BUF = 49  # current plus 48 past smoothed temperatures (4 h)
LAGS = (6, 12, 48)  # 30 min, 1 h, 4 h
SCALES = (1.0, 2.0, 4.0)
HBUF = 49  # hourly samples (48 h)
HLAGS = (12, 24, 48)  # hours
HSCALES = (6.0, 8.0, 10.0)
PRIOR_INV_TAU = 0.02  # 1/h, a time constant of 50 h
PRIOR_REC = 3.0  # K/h
PRIOR_HSP = 294.15  # K, used when no heating setpoint is observed
PRIOR_CSP = 297.15  # K, used when no cooling setpoint is observed
A_TS = math.exp(-1.0 / 3)  # smoothing of the observed temperature (15 min)
A_EST = math.exp(-1.0 / 288)  # estimates average about one day of valid samples
KEYS = ("phys_ts", "phys_buf", "phys_hbuf", "phys_n", "phys_inv_tau", "phys_rec", "phys_hsp", "phys_free",
        "phys_heat")


def _valid_hsp(hsp):
    return torch.where(hsp > 250.0, hsp, torch.full_like(hsp, PRIOR_HSP))


def phys_init(temp: torch.Tensor, hsp: torch.Tensor) -> dict:
    """State before the first decision of an episode, from (B, V) temperature and setpoint [K]."""
    b, v = temp.shape
    temp = temp.detach()
    return {
        "phys_ts": temp.clone(),
        "phys_buf": temp.unsqueeze(1).repeat(1, BUF - 1, 1),
        "phys_hbuf": temp.unsqueeze(1).repeat(1, HBUF, 1),
        "phys_n": torch.zeros(b),
        "phys_inv_tau": torch.full((b, v), PRIOR_INV_TAU),
        "phys_rec": torch.full((b, v), PRIOR_REC),
        "phys_hsp": _valid_hsp(hsp.detach()),
        "phys_free": torch.zeros(b, v),
        "phys_heat": torch.zeros(b, v),
    }


def phys_update(p: dict, temp: torch.Tensor, oat: torch.Tensor, hsp: torch.Tensor,
                reheat_prev: torch.Tensor, ahu_on_prev: torch.Tensor, csp: torch.Tensor | None = None) -> dict:
    """State after observing the decision step.

    temp, hsp, reheat_prev (normalized command) and ahu_on_prev (0/1 for each
    zone's air handler) are (B, V); oat is (B,) [K]. The returned dict also
    holds the full buffer ("_buf", current smoothed temperature first) and the
    smoothed temperature with gradient ("_ts"), used by phys_features.
    """
    ts_g = A_TS * p["phys_ts"] + (1 - A_TS) * temp
    ts = ts_g.detach()
    hsp = _valid_hsp(hsp.detach())
    buf = torch.cat([ts.unsqueeze(1), p["phys_buf"]], 1)  # (B, 49, V)
    d1h = ts - buf[:, 12]  # K per hour
    dout = ts - oat.unsqueeze(-1)
    free_now = (reheat_prev <= -0.9) & (ahu_on_prev < 0.5) & (dout > 5.0)
    heat_now = (ts < hsp - 0.5) & (reheat_prev >= 0.5) & (ahu_on_prev > 0.5)
    free = torch.where(free_now, p["phys_free"] + 1, torch.zeros_like(ts))
    heat = torch.where(heat_now, p["phys_heat"] + 1, torch.zeros_like(ts))
    r_tau = (-d1h / dout.clamp(min=5.0)).clamp(0.0, 0.2)
    inv_tau = torch.where(free >= 12, A_EST * p["phys_inv_tau"] + (1 - A_EST) * r_tau, p["phys_inv_tau"])
    rec = torch.where(heat >= 12, A_EST * p["phys_rec"] + (1 - A_EST) * d1h.clamp(0.0, 15.0), p["phys_rec"])
    #Lorenzo Giannuzzo: one hourly sample every 12 decisions, starting with the
    #first one, keeps 48 h of history at a twelfth of the memory.
    push = (torch.remainder(p["phys_n"], 12) == 0).view(-1, 1, 1)
    hbuf = torch.where(push, torch.cat([ts.unsqueeze(1), p["phys_hbuf"][:, :-1]], 1), p["phys_hbuf"])
    return {
        "phys_ts": ts,
        "phys_buf": buf[:, :BUF - 1],
        "phys_hbuf": hbuf,
        "phys_n": p["phys_n"] + 1,
        "phys_inv_tau": inv_tau,
        "phys_rec": rec,
        "phys_hsp": torch.maximum(p["phys_hsp"], hsp),
        "phys_free": free,
        "phys_heat": heat,
        "_buf": buf,
        "_ts": ts_g,
        "_hsp": hsp,
        "_csp": torch.full_like(hsp, PRIOR_CSP) if csp is None else
        torch.where(csp.detach() > 250.0, csp.detach(), torch.full_like(hsp, PRIOR_CSP)),
    }


def phys_features(p: dict, oat: torch.Tensor, hours_to_start: torch.Tensor) -> torch.Tensor:
    """(B, V, PHYS_IN) features from the updated state (phys_update output),
    the outdoor air (B,) [K] and the hours to the next comfort start (B,)."""
    buf, temp = p["_buf"], p["_ts"]
    oat = oat.unsqueeze(-1)
    dout = temp - oat
    diffs = [(temp - buf[:, lag]) / s for lag, s in zip(LAGS, SCALES)]
    diffs += [(temp - p["phys_hbuf"][:, lag]) / s for lag, s in zip(HLAGS, HSCALES)]
    dt = hours_to_start.unsqueeze(-1)
    t_ff = oat + dout * torch.exp(-dt * p["phys_inv_tau"])
    need = (p["phys_hsp"] - t_ff).clamp(min=0.0) / p["phys_rec"].clamp(min=0.5)
    slack = ((dt - need) / 24.0).clamp(-1.0, 3.0)
    return torch.stack([dout / 20.0, *diffs, p["phys_inv_tau"] * 50.0, p["phys_rec"] / 3.0,
                        (t_ff - p["phys_hsp"]) / 5.0, slack, (p["_hsp"] - temp) / 5.0, (p["_csp"] - temp) / 5.0,
                        (p["phys_hsp"] - oat) / 20.0], -1)


def phys_state(p: dict) -> dict:
    """The part of an updated state carried to the next step."""
    return {k: p[k] for k in KEYS}


def _episode_inputs(ep):
    from sbwm.wm_data import AHU_OAT

    #Lorenzo Giannuzzo: inside the rollouts the policy reads the observed
    #temperature from the true (noise-free) history, two steps back at the
    #decision (train_dpc.policy_step), so the logged state is built from the
    #same quantity; the first two decisions use the observation.
    obs = torch.cat([ep.zobs[:2, :, 0] * 5.0 + 295.0, ep.temp], 0)[:ep.length]
    hsp = ep.zobs[:ep.length, :, 1]
    oat = ep.aobs[:ep.length, :, AHU_OAT].mean(-1)
    za = torch.as_tensor(ep.schema.zone_ahu).long()
    reheat_prev = torch.cat([-torch.ones(1, obs.shape[1]), ep.zact[:ep.length - 1, :, 1]], 0)
    ahu_prev = torch.cat([torch.zeros(1, obs.shape[1]), (ep.aact[:ep.length - 1, za, 0] > 0).float()], 0)
    return obs, hsp, oat, reheat_prev, ahu_prev


def episode_phys(ep) -> dict:
    """State after each decision of a logged episode, cached on the episode.

    The recursion runs in numpy (the same arithmetic as phys_update, in float32)
    because torch dispatch costs about 2 ms per step on these small arrays; the
    buffer is not stored, it is rebuilt from the smoothed temperatures.
    """
    cached = ep.__dict__.get("_phys")
    if cached is not None:
        return cached
    obs, hsp, oat, reheat_prev, ahu_prev = (x.numpy().astype(np.float32) for x in _episode_inputs(ep))
    n, v = obs.shape
    hsp = np.where(hsp > 250.0, hsp, np.float32(PRIOR_HSP)).astype(np.float32)
    a_ts, a_est = np.float32(A_TS), np.float32(A_EST)
    out = {k: np.zeros((n, v), np.float32) for k in ("ts", "inv_tau", "rec", "hsp", "free", "heat")}
    ts = obs[0].copy()
    hist = np.repeat(ts[None], BUF - 1, 0)  # most recent first
    inv_tau = np.full(v, PRIOR_INV_TAU, np.float32)
    rec = np.full(v, PRIOR_REC, np.float32)
    hmax = hsp[0].copy()
    free = np.zeros(v, np.float32)
    heat = np.zeros(v, np.float32)
    for t in range(n):
        ts = (a_ts * ts + (1 - a_ts) * obs[t]).astype(np.float32)
        d1h = ts - hist[11]  # hist[11] is 12 steps before the current one
        dout = ts - oat[t]
        free_now = (reheat_prev[t] <= -0.9) & (ahu_prev[t] < 0.5) & (dout > 5.0)
        heat_now = (ts < hsp[t] - 0.5) & (reheat_prev[t] >= 0.5) & (ahu_prev[t] > 0.5)
        free = np.where(free_now, free + 1, 0).astype(np.float32)
        heat = np.where(heat_now, heat + 1, 0).astype(np.float32)
        r_tau = np.clip(-d1h / np.maximum(dout, 5.0), 0.0, 0.2)
        inv_tau = np.where(free >= 12, a_est * inv_tau + (1 - a_est) * r_tau, inv_tau).astype(np.float32)
        rec = np.where(heat >= 12, a_est * rec + (1 - a_est) * np.clip(d1h, 0.0, 15.0), rec).astype(np.float32)
        hmax = np.maximum(hmax, hsp[t])
        hist = np.concatenate([ts[None], hist[:-1]], 0)
        for k, x in (("ts", ts), ("inv_tau", inv_tau), ("rec", rec), ("hsp", hmax), ("free", free),
                     ("heat", heat)):
            out[k][t] = x
    res = {k: torch.from_numpy(x) for k, x in out.items()}
    res["obs"] = torch.from_numpy(obs)
    res["hsp_obs"] = torch.from_numpy(hsp)
    ep.__dict__["_phys"] = res
    return res


def phys_at(ep, t0: torch.Tensor) -> dict:
    """State before the decisions at t0 (B,) of a logged episode, as a policy
    would have it at deployment after running from the episode start."""
    e = episode_phys(ep)
    prev = (t0 - 1).clamp(min=0)
    idx = torch.stack([(t0 - 1 - i).clamp(min=0) for i in range(BUF - 1)], 1)  # (B, 48)
    #Lorenzo Giannuzzo: hourly samples were taken at the decisions 0, 12, 24...;
    #before the first one the buffer holds the first observation, as ts[0].
    last = torch.div(t0 - 1, 12, rounding_mode="floor") * 12  # latest sampled decision
    hidx = torch.stack([(last - 12 * k).clamp(min=0) for k in range(HBUF)], 1)  # (B, 49)
    p = {"phys_ts": e["ts"][prev], "phys_buf": e["ts"][idx], "phys_hbuf": e["ts"][hidx],
         "phys_n": t0.float()}
    for k in ("inv_tau", "rec", "hsp", "free", "heat"):
        p["phys_" + k] = e[k][prev]
    first = t0 == 0
    if bool(first.any()):
        #Lorenzo Giannuzzo: at the first decision nothing has been observed yet,
        #as at deployment, where the state starts from the first observation.
        init = phys_init(e["obs"][t0], e["hsp_obs"][t0])
        for k in KEYS:
            mask = first.view(-1, *([1] * (p[k].dim() - 1)))
            p[k] = torch.where(mask, init[k], p[k])
    return p
