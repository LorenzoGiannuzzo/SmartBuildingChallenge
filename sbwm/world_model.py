"""Entity-token transformer world model.

One token per zone, per air handler, one for the hot-water system and one
global token. Attention is biased by learned per-head scalars that depend only
on the structural relation between two tokens (same air handler, zone to its
own air handler, and so on), so the model is independent of building size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sbwm.wm_data import AHU_DYN, AHU_OAT, HIST, ZONE_DYN, ZONE_EXO, Episode, Normalizer

ZONE_IN = 6 + 1 + HIST + 7 + 2 + 2
AHU_IN = 15 + 4 + 4
HWS_IN = 7 + 3 + 3
GLOB_IN = 7
N_REL = 10


def relation_matrix(zone_ahu: torch.Tensor, k: int) -> torch.Tensor:
    """(N, N) relation ids for token order [zones, ahus, hws, global]."""
    v = zone_ahu.shape[0]
    n = v + k + 2
    rel = torch.zeros(n, n, dtype=torch.long)
    za = zone_ahu
    same = za[:, None] == za[None, :]
    rel[:v, :v] = torch.where(same, 0, 1)
    own = za[:, None] == torch.arange(k)[None, :]
    rel[:v, v:v + k] = torch.where(own, 2, 3)
    rel[v:v + k, :v] = torch.where(own.T, 4, 5)
    rel[v:v + k, v:v + k] = 6
    rel[:, v + k] = 7
    rel[v + k, :] = 7
    rel[:, v + k + 1] = 8
    rel[v + k + 1, :] = 8
    rel[torch.arange(n), torch.arange(n)] = 9
    return rel


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.rel_bias = nn.Embedding(N_REL, heads)
        nn.init.zeros_(self.rel_bias.weight)
        self.drop = dropout

    def forward(self, x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, d // self.heads).unbind(2)
        bias = self.rel_bias(rel).permute(2, 0, 1).unsqueeze(0)  # (1, H, N, N)
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias,
            dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(att.transpose(1, 2).reshape(b, n, d))
        return x + self.ff(self.ln2(x))


class WorldModel(nn.Module):
    def __init__(self, d: int = 128, layers: int = 4, heads: int = 4, ff: int = 256):
        super().__init__()
        self.enc_zone = nn.Sequential(nn.Linear(ZONE_IN, d), nn.GELU(), nn.Linear(d, d))
        self.enc_ahu = nn.Sequential(nn.Linear(AHU_IN, d), nn.GELU(), nn.Linear(d, d))
        self.enc_hws = nn.Sequential(nn.Linear(HWS_IN, d), nn.GELU(), nn.Linear(d, d))
        self.enc_glob = nn.Sequential(nn.Linear(GLOB_IN, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.zeros(4, d))
        self.blocks = nn.ModuleList(Block(d, heads, ff) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        #Lorenzo Giannuzzo: zone head predicts the temperature change, the next
        #zone actuator readbacks and a gas share; AHU head predicts its next
        #sensors and an electric share; HWS head its next sensors plus pump
        #electricity and boiler standby gas.
        self.head_zone = nn.Linear(d, 1 + len(ZONE_DYN) + 1)
        self.head_ahu = nn.Linear(d, len(AHU_DYN) + 1)
        self.head_hws = nn.Linear(d, 7 + 2)

    def forward(self, zin, ain, hin, gin, rel):
        """Inputs are (B, V, ZONE_IN), (B, K, AHU_IN), (B, HWS_IN), (B, GLOB_IN)."""
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
        oz = self.head_zone(x[:, :v])
        oa = self.head_ahu(x[:, v:v + k])
        oh = self.head_hws(x[:, v + k])
        elec = (F.softplus(oa[..., -1]).sum(1) + F.softplus(oh[:, -2])) / k
        gas = (F.softplus(oz[..., -1]).sum(1) + F.softplus(oh[:, -1])) / k
        return {
            "dtemp": oz[..., 0],
            "zone_dyn": oz[..., 1:1 + len(ZONE_DYN)],
            "ahu_dyn": oa[..., :-1],
            "hws": oh[:, :7],
            "power": torch.stack([elec, gas], -1),
        }


class Rollout:
    """Closed-loop multi-step prediction on one episode with given actions.

    Exogenous quantities (setpoints, outdoor air, time, comfort flags) are
    read from the episode; everything else is fed back from the model.
    """

    def __init__(self, model: WorldModel, norm: Normalizer, ep: Episode):
        self.m, self.n, self.ep = model, norm, ep
        self.rel = relation_matrix(torch.as_tensor(ep.schema.zone_ahu), ep.schema.num_ahus)
        self.static = ep.static

    def init_state(self, t0: torch.Tensor) -> dict:
        """State before the decision at index t0 (B,), using true data."""
        ep, n = self.ep, self.n
        hist = torch.stack([ep.temp[t0 - 1 - i] for i in range(HIST + 1)], 1)  # (B, H+1, V)
        return {
            "temp_hist": hist,  # most recent first, true rewarded temperatures
            "zobs": (ep.zobs[t0] - n.z_mean) / n.z_std,
            "aobs": (ep.aobs[t0] - n.a_mean) / n.a_std,
            "hobs": (ep.hobs[t0] - n.h_mean) / n.h_std,
            "zact_prev": ep.zact[t0 - 1],
            "aact_prev": ep.aact[t0 - 1],
            "hact_prev": ep.hact[t0 - 1],
        }

    def step(self, state: dict, t: torch.Tensor, zact, aact, hact):
        ep, n = self.ep, self.n
        b = t.shape[0]
        hist = state["temp_hist"]
        diffs = (hist[:, :-1] - hist[:, 1:]) / n.dtemp_std  # (B, H, V)
        tprev = (hist[:, 0] - 295.0) / 5.0
        zin = torch.cat([
            state["zobs"], tprev.unsqueeze(-1), diffs.permute(0, 2, 1),
            self.static.expand(b, -1, -1), zact, state["zact_prev"]], -1)
        ain = torch.cat([state["aobs"], aact, state["aact_prev"]], -1)
        hin = torch.cat([state["hobs"], hact, state["hact_prev"]], -1)
        gin = ep.gobs[t]
        out = self.m(zin, ain, hin, gin, self.rel)

        new_temp = hist[:, 0] + out["dtemp"] * n.dtemp_std
        nxt = t + 1
        zobs = state["zobs"].clone()
        #Lorenzo Giannuzzo: the next observation shows the pre-action true
        #temperature (one-step sensor delay), already normalized by 295/5.
        zobs[..., 0] = tprev
        zobs[..., list(ZONE_EXO)] = ((ep.zobs[nxt] - n.z_mean) / n.z_std)[..., list(ZONE_EXO)]
        zobs[..., list(ZONE_DYN)] = out["zone_dyn"]
        aobs = state["aobs"].clone()
        aobs[..., list(AHU_DYN)] = out["ahu_dyn"]
        aobs[..., AHU_OAT] = ((ep.aobs[nxt] - n.a_mean) / n.a_std)[..., AHU_OAT]
        new_state = {
            "temp_hist": torch.cat([new_temp.unsqueeze(1), hist[:, :-1]], 1),
            "zobs": zobs, "aobs": aobs, "hobs": out["hws"],
            "zact_prev": zact, "aact_prev": aact, "hact_prev": hact,
        }
        return new_state, out, new_temp
