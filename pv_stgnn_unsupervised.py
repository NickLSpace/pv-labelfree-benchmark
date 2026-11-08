"""
Unsupervised spatio-temporal GNN for the physically-informed PV graph.

`UnsupervisedSTGNN` is a single notebook-friendly class that wraps the whole
label-free pipeline on top of the graph built by `pv_physics_graph`:

    pipe = UnsupervisedSTGNN(df, graph, phys, window=48, horizon=1)
    pipe.split("2023-01-01").fit(epochs=15).calibrate()
    res = pipe.score()                      # anomaly scores, no labels used
    pipe.plot_training(); pipe.plot_score_map()
    events = pipe.events(min_len=3)          # contiguous flagged intervals per plant
    pipe.degradation_trend()                 # annual dP/dGHI loss per plant

Self-supervision has three signals, none of which needs an anomaly label:
  1. forecast      - predict t+1..t+H from the causal past (temporal pretext task)
  2. reconstruction - rebuild the input window (spatial-temporal autoencoding)
  3. physical residual - eta_hat * P_expected must match P_measured, where
     P_expected comes from the PVWatts/Faiman model and eta_hat is the network's
     predicted loss coefficient.  This anchors the representation to physics and
     makes the residual itself an interpretable, unsupervised health signal.

Scoring combines the normalised reconstruction error, the forecast deviation and the
physical residual; the threshold is set from the training distribution only
(extreme-value / quantile), so no anomalies need to be known in advance.

References: see pv_physics_graph.REFERENCES (Faiman 2008; Dobos 2014; King et al. 2004;
Wu et al. 2019 Graph WaveNet; Deng & Hooi 2021 GDN; Zhao et al. 2020 MTAD-GAT;
Raissi et al. 2019 PINN; Siffer et al. 2017 SPOT/POT thresholding).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pv_cohorts import short
from pv_physics_graph import (PIGraphDataset, PVGraph, PhysicsInformedSTGNN,
                             split_datasets)


@dataclass
class ScoreResult:
    """Container for the unsupervised scores of one split."""
    timestamps: pd.DatetimeIndex          # target time of each window
    node_names: List[str]
    recon_z: np.ndarray                   # (S, N) normalised reconstruction error
    forecast_z: np.ndarray                # (S, N) normalised forecast deviation
    physics_z: np.ndarray                 # (S, n_pv) normalised physical residual
    combined: np.ndarray                  # (S, N) fused score
    fleet: np.ndarray                     # (S,) fleet-level score
    eta: np.ndarray                       # (S, n_pv) predicted loss coefficient
    active: np.ndarray                    # (S, N) target-time activity
    threshold: float

    def frame(self, which: str = "combined") -> pd.DataFrame:
        a = getattr(self, which)
        cols = self.node_names[: a.shape[1]]
        return pd.DataFrame(a, index=self.timestamps, columns=cols)

    def ranking(self) -> pd.Series:
        return self.frame().mean(skipna=True).sort_values(ascending=False)


def borrow_error_stats(mu: torch.Tensor, sd: torch.Tensor, has: np.ndarray, n_pv: int,
                       cluster_labels: Optional[pd.Series], pv_cols: Sequence[str]):
    """Error normalisation for PV nodes without train data (cold start, e.g. cohort 3).

    mu / sd of such a node = median over PV nodes of the same technical cluster that have personal
    statistics; if there are none (or no cluster labels), median over all PV nodes with statistics.
    Exogenous nodes and nodes with personal statistics are unchanged.
    Returns (mu, sd, source) with source[j] in {"personal", "cluster", "fleet", "default"}."""
    mu, sd = mu.clone(), sd.clone()
    n = min(int(n_pv), len(mu))
    src = np.where(has, "personal", "default").astype(object)
    ref = np.flatnonzero(has[:n])
    if not len(ref):
        return mu, sd, src
    lab = (cluster_labels.reindex(list(pv_cols)[:n]).to_numpy() if cluster_labels is not None
           else np.full(n, np.nan))
    for j in range(n):
        if has[j]:
            continue
        same = ref[(lab[ref] == lab[j])] if pd.notna(lab[j]) else ref[:0]
        use, src[j] = (same, "cluster") if len(same) else (ref, "fleet")
        mu[j] = mu[use].median(); sd[j] = sd[use].median()
    return mu, sd, src


class UnsupervisedSTGNN:
    """End-to-end unsupervised trainer / scorer for the physics-informed PV graph."""

    def __init__(self, df: pd.DataFrame, graph: PVGraph, phys: Dict[str, np.ndarray],
                 window: int = 48, horizon: int = 1,
                 hidden: int = 64, tcn_layers: int = 4, gat_layers: int = 2, heads: int = 4,
                 alpha_recon: float = 1.0, alpha_phys: float = 0.5,
                 alpha_smooth: float = 0.05, phys_weight: float = 100.0,
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 device: Optional[str] = None, seed: int = 0,
                 activity_mask: Optional[pd.DataFrame] = None,
                 cold_start_steps: Optional[int] = None,
                 calibration_steps: int = 168,
                 observed: Optional[pd.DataFrame] = None,
                 context: Optional[np.ndarray] = None,
                 regime: Optional[pd.Series] = None,
                 cluster_labels: Optional[pd.Series] = None,
                 min_regime_count: int = 50):
        """
        observed       : (T x node) validity of measured values (filled gaps -> False)
        context        : (T x K) weather-regime one-hot, appended as input channels
        regime         : (T,) integer weather regime -> regime-specific thresholds
        cluster_labels : pv column -> technical cluster; new plants inherit the
                         cluster-median threshold (cold start without a personal
                         calibration period)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.observed, self.context = observed, context
        self.regime = None if regime is None else regime.reindex(df.index).fillna(0).astype(int).to_numpy()
        self.cluster_labels = cluster_labels
        self.min_stats_count = 20          # active reference windows needed for personal z-stats
        self.stats_source: Dict[str, np.ndarray] = {}   # "personal" / "cluster" / "fleet" per node
        self.min_regime_count = int(min_regime_count)
        self.regime_thresholds = None
        self.cluster_threshold_prior = None
        self.node_delay = None
        self.df, self.graph, self.phys = df, graph, phys
        self.window, self.horizon = window, horizon
        self.ar, self.ap, self.asm, self.pw = alpha_recon, alpha_phys, alpha_smooth, phys_weight
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_pv = int(graph.meta["n_pv"])
        self.node_names = list(graph.node_names)
        self.activity_mask = activity_mask
        self.cold_start_steps = window if cold_start_steps is None else int(cold_start_steps)
        self.calibration_steps = int(calibration_steps)

        in_ch = 5 + (0 if context is None else int(np.asarray(context).shape[1]))
        self.model = PhysicsInformedSTGNN(graph, in_ch=in_ch, hidden=hidden, tcn_layers=tcn_layers,
                                         gat_layers=gat_layers, heads=heads,
                                         window=window, horizon=horizon).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=3)

        self.ds_full = PIGraphDataset(df, graph, phys, window=window, horizon=horizon,
                                      activity_mask=activity_mask, observed=observed, context=context)
        self.ds_tr: Optional[PIGraphDataset] = None
        self.ds_te: Optional[PIGraphDataset] = None
        self.history: List[dict] = []
        self.stats: Dict[str, torch.Tensor] = {}
        self._p_scale = torch.as_tensor(self.ds_full.p_scale).to(self.dev).unsqueeze(-1)

    # ------------------------------------------------------------------ data
    def split(self, boundary, val_fraction: float = 0.1) -> "UnsupervisedSTGNN":
        """Chronological, leakage-safe split with train-only normalisation."""
        boundary = pd.Timestamp(boundary)
        cut = int(self.df.index.searchsorted(boundary, side="left"))
        if cut <= self.window + self.horizon or cut >= len(self.df):
            raise ValueError("boundary leaves an empty train or test split")
        fit_end = self.df.index[cut - 1]
        self.ds_full = PIGraphDataset(self.df, self.graph, self.phys, window=self.window,
                                      horizon=self.horizon, activity_mask=self.activity_mask,
                                      norm_fit_end=fit_end, observed=self.observed,
                                      context=self.context)
        target_pos = self.ds_full.idx + self.window + self.horizon - 1
        tr_idx = self.ds_full.idx[target_pos < cut]
        te_idx = self.ds_full.idx[target_pos >= cut]
        self.ds_tr = copy.copy(self.ds_full); self.ds_tr.idx = tr_idx
        self.ds_te = copy.copy(self.ds_full); self.ds_te.idx = te_idx
        n_val = int(len(tr_idx) * val_fraction)
        self._val_idx = tr_idx[-n_val:] if n_val else np.array([], int)
        self._fit_idx = tr_idx[:-n_val] if n_val else tr_idx
        self._p_scale = torch.as_tensor(self.ds_full.p_scale).to(self.dev).unsqueeze(-1)
        print(f"train windows: {len(self._fit_idx)} | val: {len(self._val_idx)} | test: {len(te_idx)}")
        return self

    def _loader(self, ds: PIGraphDataset, idx: Optional[np.ndarray] = None,
                batch_size: int = 32, shuffle: bool = False):
        """DataLoader over an optional subset of windows.

        A shallow copy keeps the subset index fixed for the loader's lifetime; mutating
        and restoring `ds.idx` in place would silently expand the subset back to the
        full set, because the DataLoader reads `idx` lazily at iteration time.
        """
        if idx is not None:
            sub = copy.copy(ds)
            sub.idx = np.asarray(idx)
            ds = sub
        return ds.torch_loader(batch_size=batch_size, shuffle=shuffle)

    def _timestamps(self, ds: PIGraphDataset, sub: pd.DataFrame) -> pd.DatetimeIndex:
        return sub.index[ds.idx + self.window + self.horizon - 1]

    # ------------------------------------------------- self-supervised losses
    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(x.dtype)
        return (x * mask).sum() / mask.sum().clamp_min(1.0)

    def _forward_losses(self, b) -> Tuple[torch.Tensor, dict, dict]:
        b = {k: v.to(self.dev) for k, v in b.items()}
        node_now = b["edge_mask"]
        o = self.model(b["x"], node_mask=node_now)
        fore_el = F.smooth_l1_loss(o["forecast"], b["y"], reduction="none")
        rec_el = F.smooth_l1_loss(o["recon"], b["recon_target"], reduction="none")
        l_fore = self._masked_mean(fore_el, b["target_mask"])
        l_rec = self._masked_mean(rec_el, b["node_mask"])
        eta = o["eta"][:, :self.n_pv]
        pmask = b["target_mask"][:, :self.n_pv]
        resid = (eta * b["p_exp"] - b["p_meas"][:, :self.n_pv]) / self._p_scale
        l_phys = self._masked_mean(resid ** 2, pmask) * self.pw
        l_smooth = self._masked_mean((eta - 1.0) ** 2, pmask)
        if eta.shape[-1] > 1:
            l_smooth = l_smooth + self._masked_mean(eta.diff(dim=-1) ** 2, pmask[..., 1:])
        total = l_fore + self.ar * l_rec + self.ap * l_phys + self.asm * l_smooth
        active_eta = eta[pmask.bool()]
        parts = dict(total=total.item(), forecast=l_fore.item(), recon=l_rec.item(),
                     physics=l_phys.item(), eta_reg=l_smooth.item(),
                     eta_mean=active_eta.mean().item() if active_eta.numel() else float("nan"))
        return total, parts, o

    # ------------------------------------------------------------------ train
    def fit(self, epochs: int = 15, batch_size: int = 32, clip: float = 5.0,
            patience: int = 5, verbose: bool = True, visualize_graph: bool = False,
            graph_timestamp=None, graph_every: int = 1, train_stride: int = 1) -> "UnsupervisedSTGNN":
        """train_stride > 1 subsamples the (highly overlapping) training windows for speed."""
        assert self.ds_tr is not None, "call .split(...) first"
        train = self._loader(self.ds_tr, self._fit_idx[::max(int(train_stride), 1)], batch_size, shuffle=True)
        val = self._loader(self.ds_tr, self._val_idx, batch_size) if len(self._val_idx) else None
        best, best_state, bad = np.inf, None, 0

        for ep in range(1, epochs + 1):
            self.model.train()
            acc = {}
            for b in train:
                self.opt.zero_grad()
                loss, parts, _ = self._forward_losses(b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.opt.step()
                for k, v in parts.items():
                    acc[k] = acc.get(k, 0.0) + v
            acc = {k: v / max(len(train), 1) for k, v in acc.items()}

            if val is not None:
                self.model.eval()
                with torch.no_grad():
                    vl = np.mean([self._forward_losses(b)[1]["total"] for b in val])
                acc["val"] = float(vl)
                self.sched.step(vl)
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
            self.history.append(acc)
            if verbose:
                print(f"epoch {ep:3d} | " + "  ".join(f"{k}={v:.5f}" for k, v in acc.items()))
            if visualize_graph and ep % max(graph_every, 1) == 0:
                import matplotlib.pyplot as plt
                self.plot_dynamic_graph(graph_timestamp or self.df.index[self._fit_idx[-1] + self.window])
                plt.show()
            if val is not None and bad >= patience:
                if verbose:
                    print(f"early stop at epoch {ep} (best val {best:.5f})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # -------------------------------------------------------------- calibrate
    @torch.no_grad()
    def calibrate(self, batch_size: int = 64, q: float = 0.995, cal_stride: int = 1) -> "UnsupervisedSTGNN":
        """Per-node baseline from active reference windows only (single forward pass)."""
        self.model.eval()
        self._cal_idx = np.asarray(self._fit_idx)[::max(int(cal_stride), 1)]
        loader = self._loader(self.ds_tr, self._cal_idx, batch_size)
        raw = self._collect_raw(loader)
        for k in ("rec", "fore", "phy"):
            x = torch.as_tensor(raw[k])
            m = torch.as_tensor(raw["active"][:, :self.n_pv] if k == "phy" else raw["active"])
            mu = torch.zeros(x.shape[1]); sd = torch.ones(x.shape[1])
            has = np.zeros(x.shape[1], bool)
            for j in range(x.shape[1]):
                v = x[:, j][m[:, j]]
                if v.numel() >= self.min_stats_count:
                    mu[j] = v.mean(); sd[j] = v.std(unbiased=False).clamp_min(1e-6); has[j] = True
            # nodes without active reference windows (plants commissioned after the train period):
            # borrow the error scale from the same technical cluster, so that their z-scores are on
            # the same scale as the (cluster-prior) thresholds
            mu, sd, src = borrow_error_stats(mu, sd, has, self.n_pv, self.cluster_labels,
                                             self.graph.meta["pv_cols"])
            self.stats_source[k] = src
            self.stats[f"{k}_mu"], self.stats[f"{k}_sd"] = mu.to(self.dev), sd.to(self.dev)
        zc = self._combine_raw(raw)
        train_scores = zc["fleet"]
        valid = train_scores[np.isfinite(train_scores)]
        self.threshold = float(np.quantile(valid, q))
        self.threshold_pot = self._pot_threshold(valid, q)
        cal = zc["combined"][:, :self.n_pv]
        self.q = q
        self.node_thresholds = np.array([
            np.quantile(cal[np.isfinite(cal[:, j]), j], q) if np.isfinite(cal[:, j]).sum() >= 20 else np.nan
            for j in range(self.n_pv)])
        # --- regime-specific per-node thresholds (weather-regime context) ------
        self.regime_thresholds = None
        if self.regime is not None:
            pos = np.asarray(self._cal_idx) + self.window + self.horizon - 1
            rg = self.regime[pos]
            R = int(self.regime.max()) + 1
            thr = np.full((R, self.n_pv), np.nan)
            for r in range(R):
                sel = rg == r
                for j in range(self.n_pv):
                    v = cal[sel, j]
                    v = v[np.isfinite(v)]
                    if len(v) >= self.min_regime_count:
                        thr[r, j] = np.quantile(v, q)
            self.regime_thresholds = thr
        # --- cold-start prior from technical clusters -------------------------
        self.cluster_threshold_prior = None
        if self.cluster_labels is not None:
            cols = self.graph.meta["pv_cols"]
            lab = self.cluster_labels.reindex(cols).to_numpy()
            prior = np.full(self.n_pv, np.nan)
            for j in range(self.n_pv):
                same = (lab == lab[j]) & np.isfinite(self.node_thresholds)
                pool = self.node_thresholds[same] if same.any() else self.node_thresholds[np.isfinite(self.node_thresholds)]
                prior[j] = np.nanmedian(pool) if len(pool) else np.nan
            self.cluster_threshold_prior = prior
            if self.regime_thresholds is not None:
                rt = self.regime_thresholds
                for j in range(self.n_pv):
                    if not np.isfinite(rt[:, j]).any():
                        same = (lab == lab[j]) & np.isfinite(rt).any(0)
                        if same.any():
                            rt[:, j] = np.nanmedian(rt[:, same], axis=1)
        print(f"threshold (q={q}): {self.threshold:.3f} | POT-refined: {self.threshold_pot:.3f}")
        return self

    def threshold_matrix(self, pos: np.ndarray) -> np.ndarray:
        """(S, n_pv) threshold per scored window: regime-specific if available, else the
        personal node threshold, else the cluster prior, else the fleet POT threshold."""
        base = np.where(np.isfinite(self.node_thresholds), self.node_thresholds,
                        self.cluster_threshold_prior if self.cluster_threshold_prior is not None
                        else np.nan)
        base = np.where(np.isfinite(base), base, getattr(self, "threshold_pot", 3.0))
        T = np.broadcast_to(base, (len(pos), self.n_pv)).copy()
        if self.regime_thresholds is not None:
            rt = self.regime_thresholds[self.regime[pos]]
            T = np.where(np.isfinite(rt), rt, T)
        return T

    @staticmethod
    def _pot_threshold(x: np.ndarray, q: float = 0.995, risk: float = 1e-4) -> float:
        """Peaks-over-threshold (generalised Pareto) refinement, cf. Siffer et al. 2017."""
        t = np.quantile(x, q)
        exc = x[x > t] - t
        if exc.size < 20:
            return float(t)
        m, v = exc.mean(), exc.var() + 1e-12
        xi = 0.5 * (1 - m ** 2 / v)                 # method of moments
        beta = 0.5 * m * (m ** 2 / v + 1)
        if abs(xi) < 1e-6:
            return float(t + beta * np.log(exc.size / (risk * x.size)))
        return float(t + beta / xi * ((risk * x.size / exc.size) ** (-xi) - 1))

    # ------------------------------------------------------------------ score
    @torch.no_grad()
    def _raw_errors(self, b) -> Dict[str, torch.Tensor]:
        b = {k: v.to(self.dev) for k, v in b.items()}
        o = self.model(b["x"], node_mask=b["edge_mask"])
        rec = (o["recon"] - b["recon_target"]).abs().mean(-1)
        fore = (o["forecast"] - b["y"]).abs().mean(-1)
        eta = o["eta"][:, :self.n_pv]
        phy = ((eta * b["p_exp"] - b["p_meas"][:, :self.n_pv]) / self._p_scale).abs().mean(-1)
        active = b["target_mask"].all(-1).bool()
        return dict(rec=rec, fore=fore, phy=phy, eta=eta.mean(-1), active=active)

    def _z(self, x: torch.Tensor, key: str) -> np.ndarray:
        mu, sd = self.stats[f"{key}_mu"], self.stats[f"{key}_sd"]
        return ((x - mu) / sd).cpu().numpy()

    @torch.no_grad()
    def _collect_raw(self, loader) -> Dict[str, np.ndarray]:
        self.model.eval()
        out = {k: [] for k in ("rec", "fore", "phy", "eta", "active")}
        for b in loader:
            e = self._raw_errors(b)
            for k in out:
                out[k].append(e[k].cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    def _combine_raw(self, raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        z = {}
        for k in ("rec", "fore", "phy"):
            mu = self.stats[f"{k}_mu"].cpu().numpy(); sd = self.stats[f"{k}_sd"].cpu().numpy()
            z[k] = (raw[k] - mu) / sd
        z["eta"] = raw["eta"].copy()
        z["active"] = raw["active"].copy()
        comb = z["rec"].copy()
        comb[:, :self.n_pv] = z["rec"][:, :self.n_pv] + .5*z["fore"][:, :self.n_pv] + .5*z["phy"]
        comb[~z["active"]] = np.nan
        z["phy"][~z["active"][:, :self.n_pv]] = np.nan
        z["eta"][~z["active"][:, :self.n_pv]] = np.nan
        z["combined"] = comb
        with np.errstate(all="ignore"):
            fleet = np.nanmax(comb[:, :self.n_pv], axis=1)
        fleet[~z["active"][:, :self.n_pv].any(1)] = np.nan
        z["fleet"] = fleet
        return z

    @torch.no_grad()
    def _combine_from_loader(self, loader) -> Dict[str, np.ndarray]:
        return self._combine_raw(self._collect_raw(loader))

    def score(self, split: str = "test", batch_size: int = 64) -> ScoreResult:
        assert self.stats, "call .calibrate() first"
        if split == "test":
            ds, idx = self.ds_te, self.ds_te.idx
        elif split == "train":
            ds, idx = self.ds_tr, self._fit_idx
        else:
            ds, idx = self.ds_full, self.ds_full.idx
        loader = self._loader(ds, idx, batch_size)
        z = self._combine_from_loader(loader)
        pos = np.asarray(idx) + self.window + self.horizon - 1
        ts = self.df.index[pos]
        delay = self.cold_start_steps + self.calibration_steps
        # with a technical-cluster prior a new node needs no personal calibration period
        node_delay = np.full(self.n_pv, delay)
        if self.cluster_threshold_prior is not None:
            node_delay[~np.isfinite(self.node_thresholds)] = self.cold_start_steps
        self.node_delay = node_delay
        if self.activity_mask is not None and delay > 0:
            am = self.activity_mask.reindex(index=self.df.index, columns=self.graph.meta["pv_cols"], fill_value=False).to_numpy(bool)
            ready = np.zeros_like(am)
            run = np.zeros(self.n_pv, int)
            for t in range(len(am)):
                run = np.where(am[t], run + 1, 0); ready[t] = run >= node_delay
            # Causally calibrate nodes absent from the training reference on their
            # first healthy post-window interval; never use future observations.
            for j in range(self.n_pv):
                if not np.isfinite(self.node_thresholds[j]) and self.cluster_threshold_prior is None:
                    run_at_pos = np.zeros(len(pos), int)
                    runj = 0
                    for q, t in enumerate(pos):
                        runj = runj + 1 if am[t, j] else 0
                        run_at_pos[q] = runj
                    cal = (run_at_pos >= self.cold_start_steps) & (run_at_pos < delay)
                    v = z["combined"][cal, j]
                    v = v[np.isfinite(v)]
                    if len(v): self.node_thresholds[j] = float(np.quantile(v, .995))
            not_ready = ~ready[pos]
            z["combined"][:, :self.n_pv][not_ready] = np.nan
            z["phy"][not_ready] = np.nan; z["eta"][not_ready] = np.nan
            with np.errstate(all="ignore"):
                z["fleet"] = np.nanmax(z["combined"][:, :self.n_pv], axis=1)
            z["fleet"][~np.isfinite(z["combined"][:, :self.n_pv]).any(1)] = np.nan
        self.last_pos = pos
        self.last_thr = self.threshold_matrix(pos)
        self.last = ScoreResult(timestamps=pd.DatetimeIndex(ts), node_names=self.node_names,
            recon_z=z["rec"], forecast_z=z["fore"], physics_z=z["phy"],
            combined=z["combined"], fleet=z["fleet"], eta=z["eta"], active=z["active"],
            threshold=getattr(self, "threshold_pot", 3.0))
        return self.last

    def ds_te_timestamps(self) -> pd.DatetimeIndex:
        n = len(self.ds_te)
        start = len(self.df) - (n + self.window + self.horizon - 1)
        sub = self.df.index[start:]
        return sub[self.ds_te.idx + self.window + self.horizon - 1]

    def online_fine_tune(self, recent_idx: Sequence[int], replay_idx: Sequence[int] = (),
                         epochs: int = 1, batch_size: int = 32, lr: Optional[float] = None):
        """Incremental adaptation on recent windows mixed with replay windows."""
        idx = np.unique(np.r_[recent_idx, replay_idx]).astype(int)
        if not len(idx):
            return self
        if lr is not None:
            for group in self.opt.param_groups: group["lr"] = lr
        loader = self._loader(self.ds_full, idx, batch_size, shuffle=True)
        for _ in range(epochs):
            self.model.train()
            for b in loader:
                self.opt.zero_grad(); loss, _, _ = self._forward_losses(b)
                loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 5.0); self.opt.step()
        return self

    def plot_dynamic_graph(self, timestamp, ax=None, seed: int = 0):
        """Visualise the induced active graph at a timestamp."""
        import matplotlib.pyplot as plt
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("install networkx for graph visualisation") from exc
        t = self.df.index.get_indexer([pd.Timestamp(timestamp)], method="nearest")[0]
        active = self.ds_full.activity[t]
        ei, et = self.graph.edge_index, self.graph.edge_type
        keep = active[ei[0]] & active[ei[1]]
        G = nx.DiGraph()
        for i, name in enumerate(self.node_names):
            if active[i]: G.add_node(i, label=name, weather=self.graph.node_type[i] == 1)
        for k in np.flatnonzero(keep): G.add_edge(int(ei[0,k]), int(ei[1,k]), family=int(et[k]))
        ax = ax or plt.subplots(figsize=(12, 8))[1]
        pos = nx.spring_layout(G, seed=seed)
        colors = ["#e76f51" if not G.nodes[n]["weather"] else "#2a9d8f" for n in G]
        nx.draw_networkx(G, pos, ax=ax, node_color=colors, node_size=140, width=.45,
                         arrowsize=7, labels={n: G.nodes[n]["label"].replace("power(W)_", "") for n in G},
                         font_size=5)
        ax.set_title(f"Active induced graph: {self.df.index[t]}"); ax.axis("off")
        return ax

    # ----------------------------------------------------------- diagnostics
    def events(self, res: Optional[ScoreResult] = None, min_len: int = 3,
               thresh: Optional[float] = None) -> pd.DataFrame:
        """Contiguous above-threshold intervals per plant - the unsupervised detections."""
        res = res or self.last
        if thresh is not None:
            thr = thresh
        elif getattr(self, "last_thr", None) is not None and len(self.last_thr) == len(res.timestamps):
            thr = self.last_thr
        else:
            learned = getattr(self, "node_thresholds", None)
            thr = learned if learned is not None else res.threshold
        rows = []
        with np.errstate(invalid="ignore"):
            M = res.combined[:, : self.n_pv] > np.asarray(thr)
        for j, name in enumerate(self.graph.meta["pv_cols"]):
            m = M[:, j]
            if not m.any():
                continue
            d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
            for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                if b - a >= min_len:
                    rows.append(dict(plant=name.replace("power(W)_", ""),
                                     start=res.timestamps[a], end=res.timestamps[b - 1],
                                     n_windows=int(b - a),
                                     peak_score=float(res.combined[a:b, j].max()),
                                     mean_physics_z=float(res.physics_z[a:b, j].mean()),
                                     mean_eta=float(res.eta[a:b, j].mean())
                                     if res.eta.ndim > 1 else np.nan))
        out = pd.DataFrame(rows)
        return out.sort_values("peak_score", ascending=False).reset_index(drop=True) if len(out) else out

    def degradation_trend(self, freq: str = "YE") -> pd.DataFrame:
        """Unsupervised degradation proxy: the trend of the learned loss coefficient eta
        and of the measured/expected power ratio, per plant and per period."""
        cols = self.graph.meta["pv_cols"]
        meas = self.df[cols].to_numpy(float)
        exp = self.phys["p_expected"]
        day = self.phys["ghi"] > 200.0
        pr = np.where(day[:, None] & (exp > 1.0), meas / np.maximum(exp, 1e-6), np.nan)
        prf = pd.DataFrame(pr, index=self.df.index,
                           columns=[c.replace("power(W)_", "") for c in cols])
        per = prf.resample(freq).median()
        yr = (per.index - per.index[0]).days / 365.25
        slope = {}
        for c in per.columns:
            y = per[c].to_numpy(float)
            ok = np.isfinite(y)
            slope[c] = (np.polyfit(yr[ok], y[ok], 1)[0] / max(np.nanmean(y[ok]), 1e-9) * 100
                        if ok.sum() > 1 else np.nan)
        out = per.T.copy()
        out.columns = [str(c.year) for c in per.index]
        out["degradation_%_per_year"] = pd.Series(slope)
        return out.sort_values("degradation_%_per_year")

    @torch.no_grad()
    def prior_strength(self) -> pd.DataFrame:
        """How much of the physics prior each attention layer retained (lambda per family)."""
        rows = []
        for i, g in enumerate(self.model.gats):
            lam = F.softplus(g.lam).detach().cpu().numpy()
            for t, fam in enumerate(("topology", "peer", "G->P")):
                rows.append(dict(layer=i, family=fam, lam_mean=lam[t].mean(),
                                 lam_min=lam[t].min(), lam_max=lam[t].max()))
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- plotting
    def plot_training(self, ax=None):
        import matplotlib.pyplot as plt
        h = pd.DataFrame(self.history)
        ax = ax or plt.subplots(figsize=(9, 3.4))[1]
        for k in [c for c in ("forecast", "recon", "physics", "val") if c in h]:
            ax.plot(h.index + 1, h[k], marker="o", ms=3, label=k)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.legend(); ax.set_title("Unsupervised objectives"); plt.tight_layout()
        return ax

    def plot_score_map(self, res: Optional[ScoreResult] = None, vmax: float = 6.0):
        import matplotlib.pyplot as plt
        res = res or self.last
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(res.combined[:, : self.n_pv].T, aspect="auto", cmap="magma",
                       vmin=0, vmax=vmax,
                       extent=[0, res.combined.shape[0], self.n_pv, 0])
        ax.set_yticks(np.arange(self.n_pv) + 0.5)
        ax.set_yticklabels([c.replace("power(W)_", "") for c in self.graph.meta["pv_cols"]],
                           fontsize=6)
        ax.set_xlabel("window"); ax.set_title("Unsupervised anomaly score [sigma]")
        fig.colorbar(im, ax=ax, label="combined score")
        plt.tight_layout()
        return ax

    def plot_plant(self, plant: str, res: Optional[ScoreResult] = None, span=None):
        """Measured vs. physical expectation, with the score and eta underneath."""
        import matplotlib.pyplot as plt
        res = res or self.last
        cols = self.graph.meta["pv_cols"]
        col = plant if plant in cols else f"power(W)_{plant}"
        j = cols.index(col)
        sl = span or slice(0, min(24 * 21, len(res.timestamps)))
        t = res.timestamps[sl]
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1]))
        ax[0].plot(t, self.df[col].reindex(t).values, lw=1, label="measured")
        exp = pd.Series(self.phys["p_expected"][:, j], index=self.df.index).reindex(t).values
        ax[0].plot(t, exp, "--", lw=1, label="physics expectation")
        ax[0].set_ylabel("W"); ax[0].legend(); ax[0].set_title(col.replace("power(W)_", ""))
        ax[1].plot(t, res.combined[sl, j], lw=1, label="score")
        ax[1].axhline(res.threshold, color="r", ls=":", label="threshold")
        ax[1].plot(t, res.eta[sl, j] * res.threshold, lw=1, alpha=.6,
                   label=r"$\hat\eta$ (scaled)")
        ax[1].set_ylabel("sigma"); ax[1].legend(fontsize=8); plt.tight_layout()
        return ax

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def node_errors(self, split: str = "test", idx: Optional[np.ndarray] = None,
                    batch_size: int = 64) -> Dict[str, np.ndarray]:
        """Per window x PV node absolute errors in units of RATED power (comparable
        across plants of different size) and the corresponding masks.

        forecast : |P_hat(t+1..t+H) - P(t+1..t+H)| / P_rated     (target_mask)
        recon    : mean_t |P_hat_rec - P| / P_rated over observed input steps
        physics  : |eta_hat * P_exp - P| / P_rated                (target_mask)
        """
        self.model.eval()
        ds = self.ds_te if split == "test" else (self.ds_tr if split == "train" else self.ds_full)
        if idx is None:
            idx = ds.idx if split != "train" else self._fit_idx
        loader = self._loader(ds, idx, batch_size)
        n = self.n_pv
        sd = torch.as_tensor(ds.sd[0, :n], device=self.dev)[None, :, None]
        scale = torch.as_tensor(ds.p_scale[0], device=self.dev)[None, :, None]
        F_, R_, P_, MF, MR = [], [], [], [], []
        for b in loader:
            b = {k: v.to(self.dev) for k, v in b.items()}
            o = self.model(b["x"], node_mask=b["edge_mask"])
            fe = ((o["forecast"][:, :n] - b["y"][:, :n]).abs() * sd / scale)          # (B, n, H)
            re = ((o["recon"][:, :n] - b["recon_target"][:, :n]).abs() * sd / scale)  # (B, n, T)
            eta = o["eta"][:, :n]
            pe = ((eta * b["p_exp"] - b["p_meas"][:, :n]).abs() / scale)
            tm = b["target_mask"][:, :n]
            rm = b["node_mask"][:, :n]
            F_.append(((fe * tm).sum(-1) / tm.sum(-1).clamp_min(1)).cpu().numpy())
            P_.append(((pe * tm).sum(-1) / tm.sum(-1).clamp_min(1)).cpu().numpy())
            R_.append(((re * rm).sum(-1) / rm.sum(-1).clamp_min(1)).cpu().numpy())
            MF.append((tm.sum(-1) > 0).cpu().numpy())
            MR.append((rm.sum(-1) > 0).cpu().numpy())
        pos = np.asarray(idx) + self.window + self.horizon - 1
        return dict(forecast=np.concatenate(F_), recon=np.concatenate(R_), physics=np.concatenate(P_),
                    mask_f=np.concatenate(MF), mask_r=np.concatenate(MR),
                    timestamps=self.df.index[pos], pos=pos)

    @staticmethod
    def summarise_errors(err: Dict[str, np.ndarray], cols: Sequence[str],
                         groups: Optional[Dict[str, Sequence[str]]] = None) -> pd.DataFrame:
        """MAE per group of plants (e.g. cohorts); errors in % of rated power."""
        groups = groups or {"all": list(cols)}
        rows = []
        for g, members in groups.items():
            jj = [list(cols).index(c) for c in members if c in cols]
            if not jj:
                continue
            r = dict(group=g, n_plants=len(jj))
            for k, mk in (("forecast", "mask_f"), ("recon", "mask_r"), ("physics", "mask_f")):
                e, m = err[k][:, jj], err[mk][:, jj]
                r[f"{k}_MAE_pct_rated"] = float(e[m].mean() * 100) if m.any() else np.nan
                r[f"n_{k}"] = int(m.sum())
            rows.append(r)
        return pd.DataFrame(rows)

    def cold_start_hours(self, err: Dict[str, np.ndarray], cols: Sequence[str], new_cols: Sequence[str],
                         ref_cols: Sequence[str], roll: int = 72, tol: float = 1.1) -> pd.DataFrame:
        """Time (h) from the first scored window of a new plant until its rolling
        forecast MAE (``roll`` h, daylight windows) first falls below ``tol`` x the
        median MAE of reference (old) plants over the same period; also reports the
        design time-to-first-valid-anomaly-score (window + calibration delay)."""
        F, M = err["forecast"], err["mask_f"]
        jj_ref = [list(cols).index(c) for c in ref_cols if c in cols]
        ref = np.nanmedian(np.where(M[:, jj_ref], F[:, jj_ref], np.nan), axis=1)
        ref_level = float(np.nanmedian(ref)) if np.isfinite(ref).any() else np.nan
        rows = []
        for c in new_cols:
            if c not in cols:
                continue
            j = list(cols).index(c)
            s = pd.Series(np.where(M[:, j], F[:, j], np.nan))
            first = s.first_valid_index()
            if first is None:
                rows.append(dict(plant=short(c), cold_start_h=np.nan)); continue
            r = s.rolling(roll, min_periods=roll // 3).mean()
            ok = np.flatnonzero((r.to_numpy() <= tol * ref_level) & (np.arange(len(r)) >= first))
            h = float(ok[0] - first) if len(ok) else np.nan
            delay = int(self.node_delay[j]) if getattr(self, "node_delay", None) is not None else np.nan
            rows.append(dict(plant=short(c), cold_start_h=h, time_to_valid_score_h=delay,
                             steady_MAE_pct=float(np.nanmean(s.to_numpy()[-24 * 30:]) * 100),
                             ref_MAE_pct=ref_level * 100))
        return pd.DataFrame(rows)

    # ----------------------------------------------------------- persistence
    def save(self, path: str):
        torch.save(dict(model=self.model.state_dict(),
                        stats={k: v.cpu() for k, v in self.stats.items()},
                        threshold=getattr(self, "threshold", None),
                        threshold_pot=getattr(self, "threshold_pot", None),
                        node_thresholds=getattr(self, "node_thresholds", None),
                        history=self.history), path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.stats = {k: v.to(self.dev) for k, v in ck["stats"].items()}
        self.threshold = ck["threshold"]; self.threshold_pot = ck["threshold_pot"]
        self.node_thresholds = ck.get("node_thresholds")
        self.history = ck["history"]
        return self
