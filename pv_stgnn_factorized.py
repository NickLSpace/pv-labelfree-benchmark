"""Section 16: factorized multimodal STGNN (STGNN-fmm).

Principle (factorization of multimodal data into DESCRIPTIVE and GENERATIVE factors, after the
multimodal factorization model of Tsai et al., ICLR 2019, https://arxiv.org/abs/1806.06176),
adapted to unsupervised PV diagnostics and to a graph of cross-modal relations:

* **Modalities** (node = PV plant, hourly, causal window of ``window`` hours):
  context (always present: P_exp, GHI, sun elevation, beam share, hour / day of year) and seven
  groups that may be missing — inverter (sibling ratio, DC-off), sub-hourly (5-min zeros),
  phase (imbalance), grid (AC over-voltage margin, frequency deviation), metering (site meter vs
  sum of inverters), **shading** (generated mask of neighbouring buildings and terrain,
  ``pv_shading``) and **Kalman** (causal predictive sd of P_exp, ``pv_kalman_uncertainty``).
* **Cross-modal graph** (domain knowledge): one node per modality and directed relations
  ``PRIOR_EDGES`` (e.g. grid -> inverter: over-voltage trips the inverter; inverter -> sub-hourly:
  a trip shows up as 5-min zeros; shading -> power; metering <-> inverter).  Messages between
  modality embeddings flow with learned weights ``w_ij = sigmoid(l_ij) * (prior_ij + eps (1 - prior_ij))``;
  edges outside the prior are allowed but L1-penalised.  The learned weights are reported as an
  interpretable output (which relations the model uses).
* **Descriptive factor** ``z_d``: attention pooling over the AVAILABLE modality nodes; trained to
  explain the deviation of the plant from the physics, ``log((P + c) / (P_exp + c))`` over the window,
  from the modalities and the context only (the measured power is not an input of this part).
  The attention weights per hour say which modality explains the deviation.
* **Generative factors**: for every group a generator ``G_g(z_d, h_g, h_spatial_g, noise)``
  reproduces the window of a missing modality from the descriptive factor, the cross-modal message
  into the node of g and the same modality at neighbouring plants of the spatial PV graph.
  Training is ADVERSARIAL: a conditional discriminator per group (LSGAN) separates real windows
  from generated ones, the generator minimises L1 on artificially hidden real windows + the
  adversarial term.  At test time a missing modality (23 stations without inverter files, new
  plants without a shading mask, modalities removed for the robustness test) is replaced by the
  generated one, with the availability mask kept as an input.
* **Detector**: the multimodal STGNN of section 15 (same losses, calibration, S4 detector, frozen
  weights) with the completed modal channels, the availability masks, ``z_d`` and the explained
  deviation as extra node inputs, plus the modal graph edges.  Modality dropout during STGNN
  training swaps observed channels for their generated version.

Training is two-stage: (A) the factorized module (encoders, cross-modal graph, descriptive head,
generators, discriminators) is co-trained over all modalities on the training period; (B) it is
frozen and the STGNN is trained on its outputs.  Checkpoint
``<cache_dir>/stgnn_mm_mm_fact_<seed>.pt`` holds both parts, the shading masks and the Kalman
parameters (all fitted on the training period only); the SHA-256 fingerprint covers both networks.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

import pv_modal_data as md
import pv_semisynth_eval as sse
import pv_stgnn_multimodal as smm
import pv_shading as psh
import pv_kalman_uncertainty as pku

VARIANT = "mm_fact"
LABEL = "STGNN-fmm (factorized: descriptive + adversarial generative, cross-modal graph)"
GROUPS = {"inverter": ("sib", "dc_off"), "subhourly": ("n_zero",), "phase": ("imb",), "grid": ("vac", "f_dev"),
          "metering": ("meter",), "shading": ("s_bin", "s_eff"), "kalman": ("k_sd_par", "k_sd_tot")}
GROUP_LABEL = dict(md.MODALITY_LABEL, shading="Shading mask (generated)", kalman="Kalman sd of P_exp")
GNAMES = list(GROUPS)
CHANNELS = [c for g in GNAMES for c in GROUPS[g]]
COND = ("p_exp", "ghi", "sin_elev", "beam", "hour_sin", "hour_cos", "doy_sin", "doy_cos")
MOD_NODES = ["context"] + GNAMES
PRIOR_EDGES = [  # (source, target): domain knowledge, used in both directions unless stated
    ("context", "shading"), ("context", "kalman"), ("context", "grid"),
    ("shading", "context"), ("kalman", "context"),
    ("grid", "inverter"), ("inverter", "subhourly"), ("subhourly", "inverter"),
    ("phase", "inverter"), ("inverter", "phase"), ("metering", "inverter"), ("inverter", "metering"),
    ("grid", "context"), ("inverter", "context"), ("subhourly", "context"), ("phase", "context"),
    ("metering", "context"), ("context", "inverter"),
]
_TR = {"sib": lambda x: np.clip(1.0 - x, -1, 1), "dc_off": lambda x: np.clip(x, 0, 1),
       "n_zero": lambda x: np.clip(x / 12.0, 0, 1), "imb": lambda x: np.clip(x, 0, 2),
       "vac": lambda x: np.clip(x / 5.0, -2, 3), "f_dev": lambda x: np.clip(x / 0.1, 0, 5),
       "meter": lambda x: 2.0 * np.clip(x, -1, 1), "s_bin": lambda x: np.clip(x, 0, 1), "s_eff": lambda x: np.clip(x, 0, 1),
       "k_sd_par": lambda x: np.clip(20.0 * x, 0, 3), "k_sd_tot": lambda x: np.clip(10.0 * x, 0, 3)}


def _log(*a):
    print(time.strftime("%H:%M:%S"), "[stgnn-fmm]", *a, flush=True)


@dataclass
class FMMConfig:
    window: int = 24
    d: int = 32
    dz: int = 4
    dn: int = 4
    layers: int = 2
    eps_off: float = 0.2           # weight scale of edges outside the prior
    l1_off: float = 1e-2
    lam_rec: float = 1.0
    lam_adv: float = 0.05
    lam_desc: float = 1.0
    p_hide: float = 0.3            # hide an available group of a node
    p_hide_all: float = 0.1        # hide a group at all nodes
    epochs: int = 8
    batch: int = 32
    stride: int = 3
    lr: float = 1e-3
    day_ghi: float = 50.0
    p_group: float = 0.15          # STGNN modality dropout (swap observed -> generated)
    p_all: float = 0.15


# ============================================================================ inputs
@dataclass
class FMMInputs:
    X: np.ndarray        # (T, n, C) modal channels, 0 where missing
    A: np.ndarray        # (T, n, K) step availability per group
    cond: np.ndarray     # (T, n, len(COND))
    rho: np.ndarray      # (T, n) target deviation log((y + c) / (a + c))
    rho_m: np.ndarray    # (T, n) target mask
    day: np.ndarray      # (T, n)


def build_inputs(D: dict, C: dict, M: "md.ModalData", S: "psh.ShadingMask", K: "pku.KalmanUncertainty",
                 df: Optional[pd.DataFrame] = None, phys: Optional[dict] = None, modal=None,
                 drop: Sequence[str] = (), cfg: Optional[FMMConfig] = None) -> FMMInputs:
    cfg = cfg or FMMConfig()
    df = D["df"] if df is None else df; phys = D["phys"] if phys is None else phys; modal = modal or M
    pv = list(D["pv_cols"]); n = len(pv); idx = pd.DatetimeIndex(df.index); T = len(idx)
    rated = np.array([max(D["rated"].get(c) or 1.0, 1e-3) * 1000.0 for c in pv])
    a = np.asarray(phys["p_expected"], float)[:, :n] / rated
    ghi = np.nan_to_num(df[md.IRR].to_numpy(float))
    act = C["valid"].reindex(index=idx, columns=pv, fill_value=False).to_numpy(bool) if C.get("valid") is not None else np.ones((T, n), bool)
    day = (a > 0.01) & (ghi[:, None] > cfg.day_ghi) & act          # daylight hours of an active plant
    F = modal.site_arrays(pv, idx, site_p=df[pv].to_numpy(float), drop=())
    has = np.nan_to_num(F["has"]).astype(bool)
    sh = S.channels(D, pv, df=df) if S is not None else {"s_bin": np.full((T, n), np.nan), "s_eff": np.full((T, n), np.nan),
                                                         "has": np.zeros((T, n))}
    kc = K.channels(D, C, pv, df=df, phys=phys) if K is not None else {"k_sd_par": np.full((T, n), np.nan),
                                                                       "k_sd_tot": np.full((T, n), np.nan), "has": np.zeros((T, n))}
    src = {**{f: F[f] for f in ("sib", "dc_off", "n_zero", "imb", "vac", "f_dev", "meter")},
           "s_bin": sh["s_bin"], "s_eff": sh["s_eff"], "k_sd_par": kc["k_sd_par"], "k_sd_tot": kc["k_sd_tot"]}
    X = np.zeros((T, n, len(CHANNELS)), np.float32); Av = np.zeros((T, n, len(GNAMES)), np.float32)
    ci = 0
    for gi, g in enumerate(GNAMES):
        base = (sh["has"] > 0) if g == "shading" else (kc["has"] > 0) if g == "kalman" else has
        av = base & day
        for f in GROUPS[g]:
            v = np.asarray(src[f], float)
            if g in ("inverter",):
                av_f = av & (np.isfinite(v) | (f == "sib"))
            else:
                av_f = av & np.isfinite(v)
            X[:, :, ci] = np.where(av_f, np.nan_to_num(_TR[f](v)), 0.0); ci += 1
        fin = np.zeros((T, n), bool)
        for f in GROUPS[g]:
            fin |= np.isfinite(np.asarray(src[f], float))
        Av[:, :, gi] = (av & fin & (g not in drop)).astype(np.float32)
    # dropped groups: zero values too
    ci = 0
    for gi, g in enumerate(GNAMES):
        for _ in GROUPS[g]:
            X[:, :, ci] *= Av[:, :, gi]; ci += 1
    sun = psh.sun_frame(D, idx)
    se = np.clip(np.sin(np.deg2rad(sun["elev"].to_numpy())), 0, 1)
    bs = np.clip(psh.erbs_beam_share(psh._kt(D, df)) / 0.835, 0, 1) * (se > 0)
    h = idx.hour.to_numpy() + idx.minute.to_numpy() / 60; dy = idx.dayofyear.to_numpy()
    glob = np.stack([ghi / 1000.0, se, bs, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * dy / 365.25), np.cos(2 * np.pi * dy / 365.25)], 1)
    cond = np.concatenate([np.clip(np.nan_to_num(a), 0, 2)[:, :, None], np.repeat(glob[:, None, :], n, 1)], 2).astype(np.float32)
    y = df[pv].to_numpy(float) / rated
    with np.errstate(all="ignore"):
        rho = np.clip(np.log((np.nan_to_num(y) + 0.02) / (np.nan_to_num(a) + 0.02)), -3, 1)
    rho_m = day & act & np.isfinite(y) & (a > 0.02)
    return FMMInputs(X, Av, cond, np.where(rho_m, rho, 0).astype(np.float32), rho_m.astype(np.float32), day)


def spatial_adjacency(graph, n: int) -> np.ndarray:
    ei = np.asarray(graph.edge_index); w = np.asarray(graph.edge_weight, float)
    A = np.zeros((n, n), np.float32)
    m = (ei[0] < n) & (ei[1] < n) & (ei[0] != ei[1])
    for s_, d_, ww in zip(ei[0][m], ei[1][m], w[m]):
        A[d_, s_] += float(ww)
    return A


def prior_matrix() -> np.ndarray:
    P = np.zeros((len(MOD_NODES), len(MOD_NODES)), np.float32)
    for s_, t_ in PRIOR_EDGES:
        P[MOD_NODES.index(t_), MOD_NODES.index(s_)] = 1.0       # row = target, column = source
    return P


# ============================================================================ network
def _nets(cfg: FMMConfig):
    import torch
    import torch.nn as nn

    class TempEnc(nn.Module):
        def __init__(self, cin, d):
            super().__init__()
            self.c1 = nn.Conv1d(cin, d, 3, padding=1); self.c2 = nn.Conv1d(d, d, 3, padding=1)
            self.out = nn.Linear(2 * d, d); self.act = nn.GELU()

        def forward(self, x):                     # (B, cin, W)
            h = self.act(self.c2(self.act(self.c1(x))))
            return self.out(torch.cat([h.mean(-1), h[..., -1]], -1))

    class FMMNet(nn.Module):
        def __init__(self):
            super().__init__()
            W, d = cfg.window, cfg.d
            self.chs = [len(GROUPS[g]) for g in GNAMES]; K = len(GNAMES)
            self.enc = nn.ModuleList([TempEnc(c + 1, d) for c in self.chs])
            self.cenc = TempEnc(len(COND), d)
            self.register_buffer("prior", torch.tensor(prior_matrix()))
            self.edge_logit = nn.Parameter(torch.where(self.prior > 0, torch.tensor(1.0), torch.tensor(-2.0)))
            self.msg = nn.ModuleList([nn.Linear(d, d) for _ in range(cfg.layers)])
            self.upd = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(cfg.layers)])
            self.miss = nn.Parameter(torch.zeros(K, d))
            self.sp = nn.Linear(d, d)
            self.pool = nn.Linear(d, 1)
            self.to_z = nn.Linear(d, cfg.dz)
            self.desc = nn.Sequential(nn.Linear(cfg.dz + d, d), nn.GELU(), nn.Linear(d, W))
            self.gen = nn.ModuleList([nn.Sequential(nn.Linear(cfg.dz + 2 * d + cfg.dn, 2 * d), nn.GELU(),
                                                    nn.Linear(2 * d, c * W)) for c in self.chs])
            self.act = nn.GELU()

        def edge_weights(self):
            K1 = self.prior.shape[0]
            w = torch.sigmoid(self.edge_logit) * (self.prior + cfg.eps_off * (1 - self.prior))
            return w * (1 - torch.eye(K1, device=w.device))

        def forward(self, xs, ms, vis, cond, Adj, noise=True):
            """xs[g]: (B, n, c_g, W); ms[g]: (B, n, W); vis: (B, n, K) bool; cond: (B, n, Cc, W); Adj: (n, n)."""
            B, n = vis.shape[:2]; K = len(GNAMES); W = cfg.window
            hs = []
            for g in range(K):
                x = torch.cat([xs[g] * ms[g][:, :, None], ms[g][:, :, None]], 2).reshape(B * n, -1, W)
                hs.append(self.enc[g](x).reshape(B, n, -1))
            H = torch.stack(hs, 2)                                      # (B, n, K, d)
            v = vis.float()
            # spatial context of each modality: neighbours with the modality visible
            num = torch.einsum("ij,bjkd->bikd", Adj, H * v[..., None])
            den = torch.einsum("ij,bjk->bik", Adj, v)[..., None]
            Hsp = self.sp(num / den.clamp_min(1e-6)) * (den > 0).float()
            H = torch.where(vis[..., None], H, self.miss[None, None] + Hsp)
            hc = self.cenc(cond.reshape(B * n, -1, W)).reshape(B, n, 1, -1)
            Hn = torch.cat([hc, H], 2)                                  # (B, n, K+1, d)
            av = torch.cat([torch.ones(B, n, 1, device=v.device), v], 2)
            Wt = self.edge_weights()                                    # (K+1, K+1) target x source
            # sources: available modalities carry full weight, missing ones (spatial estimate) half
            src_w = torch.cat([torch.ones(B, n, 1, device=v.device), 0.5 + 0.5 * v], 2)
            for l in range(cfg.layers):
                m = self.msg[l](Hn)
                Aw = Wt[None, None] * src_w[:, :, None, :]
                agg = torch.einsum("bnts,bnsd->bntd", Aw, m) / Aw.sum(-1, keepdim=True).clamp_min(1e-6)
                Hn = Hn + self.act(self.upd[l](torch.cat([Hn, agg], -1)))
            sc = self.pool(Hn).squeeze(-1).masked_fill(av < 0.5, -1e4)
            alpha = torch.softmax(sc, -1)                               # (B, n, K+1)
            z = self.to_z((alpha[..., None] * Hn).sum(2))              # descriptive factor
            rho_hat = self.desc(torch.cat([z, Hn[:, :, 0]], -1))        # (B, n, W)
            outs = []
            for g in range(K):
                eps = torch.randn(B, n, cfg.dn, device=v.device) if noise else torch.zeros(B, n, cfg.dn, device=v.device)
                o = self.gen[g](torch.cat([z, Hn[:, :, 1 + g], Hsp[:, :, g], eps], -1))
                outs.append(o.reshape(B, n, self.chs[g], W))
            return dict(z=z, rho_hat=rho_hat, gen=outs, alpha=alpha)

    class Disc(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = nn.ModuleList([nn.Sequential(TempEnc(c + len(COND), cfg.d), nn.GELU(), nn.Linear(cfg.d, 1))
                                    for c in [len(GROUPS[g]) for g in GNAMES]])

        def forward(self, g, x, cond):                 # x: (M, c, W), cond: (M, Cc, W)
            return self.d[g](torch.cat([x, cond], 1)).squeeze(-1)

    return FMMNet, Disc


def _windows(arr: np.ndarray, t: np.ndarray, W: int) -> np.ndarray:
    """arr (T, n, ...) -> (B, n, ..., W) windows ending at t (zero padded at the start)."""
    T = arr.shape[0]
    ii = t[:, None] + np.arange(-W + 1, 1)[None, :]
    ok = ii >= 0
    out = arr[np.clip(ii, 0, T - 1)] * ok.reshape(ok.shape + (1,) * (arr.ndim - 1))
    return np.moveaxis(out, 1, -1)                     # (B, n, ..., W)


def _batch(I: FMMInputs, t: np.ndarray, W: int, dev):
    import torch
    X = _windows(I.X, t, W)          # (B, n, C, W)
    A = _windows(I.A, t, W)          # (B, n, K, W)
    cond = torch.tensor(_windows(I.cond, t, W), device=dev)
    xs, ms, ci = [], [], 0
    for gi, g in enumerate(GNAMES):
        c = len(GROUPS[g])
        xs.append(torch.tensor(X[:, :, ci:ci + c], device=dev)); ci += c
        ms.append(torch.tensor(A[:, :, gi], device=dev))
    vis = torch.tensor(A.max(-1) > 0, device=dev)     # group observed somewhere in the window
    rho = torch.tensor(_windows(I.rho, t, W), device=dev); rm = torch.tensor(_windows(I.rho_m, t, W), device=dev)
    return xs, ms, vis, cond, rho, rm


class FactorizedModal:
    """Stage A: factorized descriptive / generative module with the cross-modal graph."""

    def __init__(self, n: int, Adj: np.ndarray, cfg: Optional[FMMConfig] = None, seed: int = 0, device=None):
        import torch
        self.cfg = cfg or FMMConfig(); self.n = n; self.seed = seed
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        Net, Disc = _nets(self.cfg)
        self.net, self.disc = Net().to(self.dev), Disc().to(self.dev)
        A = Adj / np.maximum(Adj.sum(1, keepdims=True), 1e-6)
        self.Adj = torch.tensor(A, dtype=torch.float32, device=self.dev)
        self.history = []

    # ------------------------------------------------------------------ training
    def fit(self, I: FMMInputs, train_mask: np.ndarray, verbose: bool = True):
        import torch
        cfg = self.cfg; W = cfg.window
        rng = np.random.default_rng(self.seed)
        t_all = np.flatnonzero(train_mask & I.day.any(1))
        t_all = t_all[t_all >= W - 1][::cfg.stride]
        optG = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        optD = torch.optim.Adam(self.disc.parameters(), lr=cfg.lr)
        prior = self.net.prior
        for ep in range(cfg.epochs):
            perm = rng.permutation(t_all); stats = []
            self.net.train(); self.disc.train()
            for b0 in range(0, len(perm), cfg.batch):
                t = np.sort(perm[b0:b0 + cfg.batch])
                xs, ms, vis, cond, rho, rm = _batch(I, t, W, self.dev)
                B, n = vis.shape[:2]
                hide = (torch.rand(B, n, len(GNAMES), device=self.dev) < cfg.p_hide)
                hide |= (torch.rand(B, 1, len(GNAMES), device=self.dev) < cfg.p_hide_all)
                hide &= vis
                vis_in = vis & ~hide
                ms_in = [ms[g] * vis_in[:, :, g, None].float() for g in range(len(GNAMES))]
                out = self.net(xs, ms_in, vis_in, cond, self.Adj)
                # ---- discriminator
                lossD = 0.0; nD = 0
                for g in range(len(GNAMES)):
                    real = vis[:, :, g]; fake = ~vis_in[:, :, g] & (cond[:, :, 0].amax(-1) > 0.01)
                    if real.any() and fake.any():
                        cr = cond[real]; cf = cond[fake]
                        dr = self.disc(g, xs[g][real], cr); df_ = self.disc(g, out["gen"][g][fake].detach(), cf)
                        lossD = lossD + ((dr - 1) ** 2).mean() + (df_ ** 2).mean(); nD += 1
                if nD:
                    optD.zero_grad(); (lossD / nD).backward(); optD.step()
                # ---- generator / encoders / descriptive head
                rec = 0.0; adv = 0.0; ng = 0
                for g in range(len(GNAMES)):
                    h = hide[:, :, g]
                    if h.any():
                        m = ms[g][h][:, None, :]
                        rec = rec + (torch.abs(out["gen"][g][h] - xs[g][h]) * m).sum() / m.sum().clamp_min(1.0) / xs[g].shape[2]
                    fake = ~vis_in[:, :, g] & (cond[:, :, 0].amax(-1) > 0.01)
                    if fake.any():
                        adv = adv + ((self.disc(g, out["gen"][g][fake], cond[fake]) - 1) ** 2).mean(); ng += 1
                desc = (torch.nn.functional.smooth_l1_loss(out["rho_hat"], rho, reduction="none") * rm).sum() / rm.sum().clamp_min(1.0)
                l1 = (torch.sigmoid(self.net.edge_logit) * (1 - prior)).mean()
                loss = cfg.lam_rec * rec + cfg.lam_adv * (adv / max(ng, 1)) + cfg.lam_desc * desc + cfg.l1_off * l1
                optG.zero_grad(); loss.backward(); optG.step()
                stats.append([float(rec), float(adv) / max(ng, 1), float(desc), float(lossD) / max(nD, 1) if nD else np.nan])
            s = np.nanmean(np.array(stats, float), 0) if stats else np.full(4, np.nan)
            self.history.append(dict(epoch=ep + 1, rec=s[0], adv=s[1], desc=s[2], disc=s[3]))
            if verbose:
                _log(f"epoch {ep + 1}/{cfg.epochs}: L1 rec {s[0]:.4f} | adv {s[1]:.3f} | descriptive {s[2]:.4f} | D {s[3]:.3f}")
        self.net.eval(); self.disc.eval()
        for p in list(self.net.parameters()) + list(self.disc.parameters()):
            p.requires_grad_(False)
        return self

    # ------------------------------------------------------------------ inference
    def complete(self, I: FMMInputs, batch: int = 128, per_group: bool = True) -> Dict[str, np.ndarray]:
        """Causal outputs at the last step of every window: completed channels, generated channels
        (each group generated with that group hidden), z_d, explained deviation, attention."""
        import torch
        cfg = self.cfg; W = cfg.window; T, n = I.X.shape[:2]
        K = len(GNAMES)
        comp = np.zeros((T, n, len(CHANNELS)), np.float32); gen = np.zeros_like(comp)
        z = np.zeros((T, n, cfg.dz), np.float32); rh = np.zeros((T, n), np.float32)
        alpha = np.zeros((T, n, K + 1), np.float16)
        offs = np.cumsum([0] + [len(GROUPS[g]) for g in GNAMES])
        need = np.flatnonzero(I.day.any(1))
        with torch.no_grad():
            for b0 in range(0, len(need), batch):
                t = need[b0:b0 + batch]
                xs, ms, vis, cond, _, _ = _batch(I, t, W, self.dev)
                out = self.net(xs, ms, vis, cond, self.Adj, noise=False)
                z[t] = out["z"].cpu().numpy(); rh[t] = out["rho_hat"][..., -1].cpu().numpy()
                alpha[t] = out["alpha"].cpu().numpy().astype(np.float16)
                for g in range(K):
                    sl = slice(offs[g], offs[g + 1])
                    gl = out["gen"][g][..., -1].cpu().numpy()                     # (B, n, c)
                    obs = I.A[t, :, g] > 0
                    comp[t, :, sl] = np.where(obs[..., None], I.X[t, :, sl], gl)
                    gen[t, :, sl] = gl
                if per_group:
                    for g in range(K):
                        if not vis[:, :, g].any():
                            continue
                        v2 = vis.clone(); v2[:, :, g] = False
                        ms2 = list(ms); ms2[g] = ms[g] * 0
                        o2 = self.net(xs, ms2, v2, cond, self.Adj, noise=False)
                        sl = slice(offs[g], offs[g + 1])
                        gen[t, :, sl] = o2["gen"][g][..., -1].cpu().numpy()
        day = I.day[..., None]
        return dict(completed=comp * day, generated=gen * day, z=z * day, rho_hat=rh * I.day, alpha=alpha)

    def node_features(self, I: FMMInputs, n_nodes: int, out: Optional[dict] = None):
        """(T, N_nodes, F) completed features and the generated-swap version for modality dropout."""
        out = out or self.complete(I)
        T, n = I.X.shape[:2]
        mask = (I.A > 0).astype(np.float32)
        def pack(ch):
            Fv = np.concatenate([ch, mask, out["z"], out["rho_hat"][..., None]], 2)
            Z = np.zeros((T, n_nodes, Fv.shape[2]), np.float32); Z[:, :n] = Fv
            return Z
        nf = pack(out["completed"])
        gen_nf = np.concatenate([out["generated"], np.zeros_like(mask), out["z"], out["rho_hat"][..., None]], 2)
        G = np.zeros_like(nf); G[:, :n] = gen_nf
        return nf, G, out

    # ------------------------------------------------------------------ state / interpretability
    def state(self) -> dict:
        return dict(net={k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()},
                    disc={k: v.detach().cpu().clone() for k, v in self.disc.state_dict().items()},
                    cfg=asdict(self.cfg), history=list(self.history))

    @classmethod
    def from_state(cls, st: dict, n: int, Adj: np.ndarray, seed: int = 0, device=None):
        obj = cls(n, Adj, FMMConfig(**st["cfg"]), seed, device)
        obj.net.load_state_dict({k: v.to(obj.dev) for k, v in st["net"].items()})
        obj.disc.load_state_dict({k: v.to(obj.dev) for k, v in st["disc"].items()})
        obj.history = list(st.get("history", []))
        obj.net.eval(); obj.disc.eval()
        for p in list(obj.net.parameters()) + list(obj.disc.parameters()):
            p.requires_grad_(False)
        return obj

    def crossmodal_table(self) -> pd.DataFrame:
        import torch
        with torch.no_grad():
            W = self.net.edge_weights().cpu().numpy()
        P = prior_matrix(); rows = []
        lab = {"context": "Context (P_exp, weather)", **GROUP_LABEL}
        for i, t in enumerate(MOD_NODES):
            for j, s_ in enumerate(MOD_NODES):
                if i != j:
                    rows.append({"Source": lab[s_], "Target": lab[t], "Prior edge": bool(P[i, j]), "Learned weight": float(W[i, j])})
        return pd.DataFrame(rows).sort_values("Learned weight", ascending=False).reset_index(drop=True)


# ============================================================================ STGNN stage (B)
def _swap_dataset(ds, nf: np.ndarray, gen_nf: np.ndarray):
    from pv_physics_graph import PIGraphDataset
    offs = np.cumsum([0] + [len(GROUPS[g]) for g in GNAMES]); C = len(CHANNELS)

    class FMMGraphDataset(PIGraphDataset):
        def __getitem__(self, i):
            d = PIGraphDataset.__getitem__(self, i)
            t0 = self.idx[i]
            x = self.node_feat[t0:t0 + self.window].transpose(1, 2, 0).copy()            # (N, F, W)
            if self.p_group > 0 or self.p_all > 0:
                rng = np.random.default_rng()
                g_ = self.gen_feat[t0:t0 + self.window].transpose(1, 2, 0)
                N = x.shape[0]
                allm = rng.random(N) < self.p_all
                for gi in range(len(GNAMES)):
                    m = (rng.random(N) < self.p_group) | allm
                    if m.any():
                        sl = list(range(offs[gi], offs[gi + 1]))
                        x[np.ix_(m, sl)] = g_[np.ix_(m, sl)]
                        x[m, C + gi] = 0.0
            d["x"] = np.concatenate([d["x"], x], 1).astype(np.float32)
            return d

    if not isinstance(ds, PIGraphDataset) or getattr(ds, "_fmm", False):
        return ds
    ds.__class__ = FMMGraphDataset
    ds.node_feat, ds.gen_feat = nf, gen_nf; ds.p_group = 0.0; ds.p_all = 0.0; ds._fmm = True
    ds.in_ch = ds.in_ch + nf.shape[2]
    return ds


def make_pipe(df, graph, phys, nf, gen_nf, p_group=0.15, p_all=0.15, **kw):
    import torch
    from pv_stgnn_unsupervised import UnsupervisedSTGNN
    from pv_physics_graph import PhysicsInformedSTGNN

    class FactorizedSTGNN(UnsupervisedSTGNN):
        def split(self, boundary, val_fraction: float = 0.1):
            UnsupervisedSTGNN.split(self, boundary, val_fraction)
            for ds in (self.ds_full, self.ds_tr, self.ds_te):
                _swap_dataset(ds, self.node_feat, self.gen_feat)
            return self

        def fit(self, *a, **k):
            self.ds_tr.p_group, self.ds_tr.p_all = self.p_group, self.p_all
            try:
                return UnsupervisedSTGNN.fit(self, *a, **k)
            finally:
                self.ds_tr.p_group = self.ds_tr.p_all = 0.0

    pipe = FactorizedSTGNN(df, graph, phys, **kw)
    pipe.node_feat, pipe.gen_feat = nf, gen_nf; pipe.p_group, pipe.p_all = p_group, p_all
    torch.manual_seed(kw.get("seed", 0))
    in_ch = 5 + (0 if kw.get("context") is None else int(np.asarray(kw["context"]).shape[1])) + nf.shape[2]
    pipe.model = PhysicsInformedSTGNN(graph, in_ch=in_ch, hidden=kw.get("hidden", 64), tcn_layers=kw.get("tcn_layers", 4),
                                      gat_layers=kw.get("gat_layers", 2), heads=kw.get("heads", 4),
                                      window=kw.get("window", 48), horizon=kw.get("horizon", 1)).to(pipe.dev)
    pipe.opt = torch.optim.Adam(pipe.model.parameters(), lr=1e-3, weight_decay=1e-5)
    pipe.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(pipe.opt, factor=0.5, patience=3)
    return pipe


class FrozenFMM(smm.FrozenMM):
    """Frozen STGNN-fmm (factorized module + STGNN).  ``build(df, phys, modal, drop)``: ``drop`` may
    contain any of ``GROUPS`` (the generator then reproduces the dropped modality)."""

    def __init__(self, ckpt, D, C, M, window=48, hidden=32, seed=0, device=None, source=""):
        self.S, self.K = ckpt["shading"], ckpt["kunc"]
        self.graph_fmm = smm.modal_graph(C["graph_mv"], M, D)
        n = len(D["pv_cols"])
        self.fmm = FactorizedModal.from_state(ckpt["fmm"], n, spatial_adjacency(self.graph_fmm, n), seed, device)
        self.alpha_log: Dict[str, dict] = {}
        super().__init__(ckpt, D, C, M, "mm_graph", window, hidden, seed, device, source)
        self.mm_variant = VARIANT

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for part in ("model",):
            for k in sorted(self.ckpt[part]):
                h.update(k.encode()); h.update(self.ckpt[part][k].numpy().tobytes())
        for part in ("net", "disc"):
            for k in sorted(self.ckpt["fmm"][part]):
                h.update(k.encode()); h.update(self.ckpt["fmm"][part][k].numpy().tobytes())
        return h.hexdigest()[:16]

    def unchanged(self) -> bool:
        live = hashlib.sha256()
        for k in sorted(self.ckpt["model"]):
            live.update(k.encode()); live.update(self.ckpt["model"][k].numpy().tobytes())
        sd = {"net": self.fmm.net.state_dict(), "disc": self.fmm.disc.state_dict()}
        for part in ("net", "disc"):
            for k in sorted(sd[part]):
                live.update(k.encode()); live.update(sd[part][k].detach().cpu().numpy().tobytes())
        return live.hexdigest()[:16] == self._fp and self.fingerprint() == self._fp

    def _log_alpha(self, df, I, out, drop=(), log_key=None):
        """Keep the attention weights (all builds without dropped modalities) and, for the clean
        frame, the real and generated channels (generation quality)."""
        if drop:
            return
        key = log_key or ("clean" if df is self.D["df"] else f"id{id(df)}")
        L = dict(alpha=out["alpha"], ts=pd.DatetimeIndex(df.index), day=I.day)
        if key == "clean":
            L.update(A=I.A.astype(np.float16), gen=out["generated"].astype(np.float16), X=I.X.astype(np.float16))
        self.alpha_log[key] = L

    def explain(self, df, phys, modal=None, log_key=None):
        """Factorized module only (no STGNN): attention, z_d, generated channels of a frame."""
        I = self.inputs(df, phys, modal)
        out = self.fmm.complete(I)
        self._log_alpha(df, I, out, (), log_key)
        return out

    def inputs(self, df, phys, modal=None, drop: Sequence[str] = ()):
        return build_inputs(self.D, self.C, self.M, self.S, self.K, df, phys, modal, drop, self.fmm.cfg)

    def build(self, df, phys, modal=None, drop: Sequence[str] = (), log_key: Optional[str] = None):
        kw = sse.pipe_kwargs(self.D, self.C, "multiview", self.window, self.hidden, self.seed, self.device)
        kw.pop("graph")
        I = self.inputs(df, phys, modal, drop)
        nf, gnf, out = self.fmm.node_features(I, len(self.graph.node_names))
        self._log_alpha(df, I, out, drop, log_key)
        with contextlib.redirect_stdout(io.StringIO()):
            pipe = make_pipe(df, self.graph, phys, nf, gnf, 0.0, 0.0, **kw)
            pipe.split(sse.BOUNDARY, val_fraction=0.1)
        pipe.model.load_state_dict({k: v.to(pipe.dev) for k, v in self.ckpt["model"].items()})
        pipe.stats = {k: v.clone().to(pipe.dev) for k, v in self.ckpt["stats"].items()}
        for a in sse.CALIB_ATTRS:
            if self.ckpt.get(a) is not None or a in ("regime_thresholds", "cluster_threshold_prior"):
                setattr(pipe, a, copy.deepcopy(self.ckpt.get(a)))
        pipe.model.eval()
        for p in pipe.model.parameters():
            p.requires_grad_(False)
        pipe.fit = sse._blocked; pipe.online_fine_tune = sse._blocked; pipe.opt = None
        return pipe

    @classmethod
    def train(cls, D, C, M, seed=0, window=48, hidden=32, epochs=15, train_stride=1, device=None, save_to=None,
              fmm_cfg: Optional[FMMConfig] = None, S=None, K=None, verbose=False, **_):
        from pv_physics_graph import data_signature
        fcfg = fmm_cfg or FMMConfig()
        S = S or psh.fit_shading(D, C, verbose=verbose)
        K = K or pku.fit_uncertainty(D, C, verbose=verbose)
        g = smm.modal_graph(C["graph_mv"], M, D)
        n = len(D["pv_cols"])
        I = build_inputs(D, C, M, S, K, cfg=fcfg)
        fm = FactorizedModal(n, spatial_adjacency(g, n), fcfg, seed, device)
        _log(f"seed {seed}: stage A (factorized module, adversarial generation, {fcfg.epochs} epochs)")
        fm.fit(I, np.asarray(D["df"].index < pd.Timestamp(sse.BOUNDARY)), verbose=True)
        nf, gnf, _ = fm.node_features(I, len(g.node_names))
        _log(f"seed {seed}: stage B (STGNN on completed modalities, {epochs} epochs, stride {train_stride})")
        kw = sse.pipe_kwargs(D, C, "multiview", window, hidden, seed, device); kw.pop("graph")
        pipe = make_pipe(D["df"], g, D["phys"], nf, gnf, fcfg.p_group, fcfg.p_all, **kw)
        pipe.split(sse.BOUNDARY, val_fraction=0.1)
        pipe.fit(epochs=epochs, batch_size=64, patience=3, train_stride=train_stride, verbose=verbose)
        pipe.calibrate(q=0.995, cal_stride=2)
        ck = sse.make_checkpoint(pipe, dict(variant="multiview", mm_variant=VARIANT, seed=seed, window=window,
                                            hidden=hidden, epochs=epochs, train_stride=train_stride,
                                            modal_sha=smm.modal_signature(M), shading_sha=S.signature(),
                                            kunc_sha=K.signature(), fmm_cfg=asdict(fcfg), **data_signature(D, sse.BOUNDARY)))
        ck["fmm"] = fm.state(); ck["shading"] = S; ck["kunc"] = K
        if save_to:
            sse.save_checkpoint(ck, save_to)
        return cls(ck, D, C, M, window, hidden, seed, device, source="trained factorized multimodal model")


def frozen_fmm_models(D, C, M, seeds=(0,), cache_dir="results/cache", window=48, hidden=32, epochs=15, train_stride=1,
                      device=None, train_missing=True, fmm_cfg: Optional[FMMConfig] = None, S=None, K=None) -> Dict[int, FrozenFMM]:
    """Checkpoint (same data, same modal data) > training once, then freezing."""
    import torch
    out = {}
    for s in seeds:
        p = smm.checkpoint_path(cache_dir, VARIANT, s); mdl = None
        if os.path.exists(p):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            status, why = sse.check_checkpoint(ck, D, C, "multiview")
            m = ck.get("meta", {})
            if status == "ok" and m.get("modal_sha") == smm.modal_signature(M) and m.get("mm_variant") == VARIANT and "fmm" in ck:
                mdl = FrozenFMM(ck, D, C, M, m.get("window", window), m.get("hidden", hidden), s, device,
                                source=f"checkpoint {os.path.basename(p)}")
            else:
                _log(f"{os.path.basename(p)} not used ({why or 'modal data differ'}); retraining")
        if mdl is None:
            if not train_missing:
                raise FileNotFoundError(p)
            _log(f"training {VARIANT} seed {s}, then freezing")
            mdl = FrozenFMM.train(D, C, M, s, window, hidden, epochs, train_stride, device, p, fmm_cfg, S, K)
        out[s] = mdl
        _log(f"{VARIANT} seed {s}: {mdl.source}, weights sha256[:16] = {mdl.fingerprint()}")
    return out


# ============================================================================ interpretability / generation quality
def generation_quality(mdl: FrozenFMM, boundary: str = "2023-01-01") -> pd.DataFrame:
    """Each group generated with that group hidden vs the real channel (test period, available hours),
    against two baselines: the plant's training median and the mean of the spatial neighbours."""
    L = mdl.alpha_log.get("clean")
    if L is None or "X" not in L:
        mdl.explain(mdl.D["df"], mdl.D["phys"], log_key="clean"); L = mdl.alpha_log["clean"]
    ts, A, X, G = L["ts"], L["A"].astype(float), L["X"].astype(float), L["gen"].astype(float)
    test = np.asarray(ts >= pd.Timestamp(boundary)); train = ~test
    Adj = spatial_adjacency(mdl.graph_fmm, X.shape[1])
    rows = []; ci = 0
    for gi, g in enumerate(GNAMES):
        for f in GROUPS[g]:
            av = A[:, :, gi] > 0; m = av & test[:, None]
            if m.sum() < 50:
                ci += 1; continue
            x = X[:, :, ci]; gen = G[:, :, ci]
            med = np.array([np.median(x[train & av[:, j], j]) if (train & av[:, j]).sum() else 0.0 for j in range(x.shape[1])])
            num = (x * av) @ Adj.T; den = av.astype(float) @ Adj.T
            nb = np.where(den > 0, num / np.maximum(den, 1e-9), med[None, :])
            e = lambda p: float(np.mean(np.abs(p[m] - x[m])))
            def corr(p):
                a_, b_ = p[m], x[m]
                return float(np.corrcoef(a_, b_)[0, 1]) if np.std(a_) > 1e-9 and np.std(b_) > 1e-9 else np.nan
            rows.append({"Modality": GROUP_LABEL[g], "Channel": f, "Test hours": int(m.sum()),
                         "MAE generated": e(gen), "MAE training median": e(np.broadcast_to(med, x.shape)),
                         "MAE neighbour mean": e(nb), "Corr generated": corr(gen), "Corr neighbour mean": corr(nb)})
            ci += 1
    return pd.DataFrame(rows)


def attention_by_scenario(mdl: FrozenFMM, reals: dict, keys: Optional[dict] = None) -> pd.DataFrame:
    """Attention of the descriptive factor over the modality nodes at the event hours of each scenario
    MINUS the attention of the same plant at the same hours in the clean frame (so that the difference
    reflects the event, not which plants have which modalities)."""
    rows = []
    if "clean" not in mdl.alpha_log:
        mdl.explain(mdl.D["df"], mdl.D["phys"], log_key="clean")
    Lc = mdl.alpha_log["clean"]; ac = Lc["alpha"].astype(float)
    pv = list(mdl.D["pv_cols"])
    for k, real in reals.items():
        key = (keys or {}).get(k, f"id{id(real.df)}")
        if key not in mdl.alpha_log:
            mdl.explain(real.df, real.phys, getattr(real, "modal", None), log_key=key)
        L = mdl.alpha_log[key]; ts = L["ts"]; al = L["alpha"].astype(float)
        pos = Lc["ts"].get_indexer(ts)
        for r in real.labels.itertuples():
            if r.column not in pv:
                continue
            j = pv.index(r.column)
            w = np.asarray((ts >= r.start) & (ts <= r.end)) & L["day"][:, j] & (pos >= 0)
            if w.sum() == 0:
                continue
            ev = al[w, j].mean(0); cl = ac[pos[w], j].mean(0)
            rows.append(dict(scenario=r.scenario, **{MOD_NODES[i]: ev[i] - cl[i] for i in range(len(MOD_NODES))}))
    if not rows:
        return pd.DataFrame()
    X = pd.DataFrame(rows); n = X.groupby("scenario").size()
    T = X.groupby("scenario")[MOD_NODES].mean()
    lab = {"context": "Context (P_exp, weather)", **GROUP_LABEL}
    T.columns = [f"Delta attention: {lab[c]}" for c in T.columns]
    import pv_modal_eval as me
    T.insert(0, "Events", n)
    T.index = [me.SCN_SHORT.get(i, i) for i in T.index]
    return T.reset_index().rename(columns={"index": "Scenario"})


def crossmodal_mean(mdls: Dict[int, FrozenFMM]) -> pd.DataFrame:
    """Prior vs learned cross-modal edge weights, mean (and range) over model seeds."""
    T = [m.fmm.crossmodal_table().assign(seed=s) for s, m in mdls.items()]
    X = pd.concat(T).groupby(["Source", "Target", "Prior edge"])["Learned weight"].agg(["mean", "min", "max"]).reset_index()
    X = X.rename(columns={"mean": "Learned weight (mean over seeds)", "min": "Min over seeds", "max": "Max over seeds"})
    return X.sort_values(["Prior edge", "Learned weight (mean over seeds)"], ascending=[False, False]).reset_index(drop=True)


def training_history(mdls: Dict[int, FrozenFMM]) -> pd.DataFrame:
    rows = []
    for s, m in mdls.items():
        for h in m.fmm.history:
            rows.append({"Seed": s, "Epoch": h["epoch"], "L1 reconstruction (hidden groups)": h["rec"],
                         "Adversarial (generator)": h["adv"], "Descriptive (Huber)": h["desc"], "Discriminator": h["disc"]})
    return pd.DataFrame(rows)


def plot_crossmodal(mdls: Dict[int, FrozenFMM], ax=None):
    W = np.mean([m.fmm.net.edge_weights().detach().cpu().numpy() for m in mdls.values()], 0)
    return plot_crossmodal_matrix(W, ax)


def plot_crossmodal_matrix(W: np.ndarray, ax=None):
    import matplotlib.pyplot as plt
    P = prior_matrix()
    lab = {"context": "Context", "inverter": "Inverter", "subhourly": "5-min", "phase": "Phase", "grid": "Grid V/f",
           "metering": "Meter", "shading": "Shading", "kalman": "Kalman sd"}
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
    else:
        fig = ax.figure
    im = ax.imshow(W, cmap="viridis", vmin=0, vmax=max(float(W.max()), 1e-3))
    names = [lab[m] for m in MOD_NODES]
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            if P[i, j] > 0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="white", lw=1.2))
    ax.set_xlabel("source modality"); ax.set_ylabel("target modality")
    ax.set_title("Cross-modal graph: learned edge weights\n(white frame = prior edge from domain knowledge)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
