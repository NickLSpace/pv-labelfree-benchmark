"""Section 15, stage 4: multimodal STGNN (inverter channels as node inputs + modal graph edges).

Independent extension of ``pv_stgnn_unsupervised.UnsupervisedSTGNN``: the original class,
the checkpoints of sections 5-14 and the SPS / Kalman comparison are not changed.

Two variants (both multi-view, i.e. weather-regime context and technical clusters as in
section 5):

* ``mm_early``  early fusion: every PV node receives ``len(NN_FEATS)`` extra input channels
  (site-level inverter channels of ``pv_modal_data``: availability mask, metering ratio,
  inverter-sibling ratio, AC over-voltage margin, frequency deviation, phase imbalance,
  5-min zeros, DC-off share).  Missing modality = zeros + mask 0 (the 23 stations without
  inverter data and the night hours).  Modality dropout during training (each modality group
  of a node is removed with probability ``p_group``, all modal channels with ``p_all``) makes the
  model usable when a modality is absent at test time.
* ``mm_graph``  ``mm_early`` + modal graph edges: AC-voltage correlation (sites on the same
  feeder, training period; edge type ``max + 1``) and the Brick / name hierarchy (sites of the
  same building complex; edge type ``max + 2``).

Losses, calibration and the detector are exactly those of the unimodal model, so the anomaly
score keeps its meaning (forecast + reconstruction of the site power + physics residual).

Frozen checkpoints: ``<cache_dir>/stgnn_mm_<variant>_<seed>.pt`` with the data signature of
``pv_physics_graph.data_signature`` plus a signature of the modal data.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import os
import re
import time
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

import pv_modal_data as md
import pv_semisynth_eval as sse

NN_FEATS = ("has", "meter", "sib", "vac", "f_dev", "imb", "n_zero", "dc_off")
GROUP_CH = {g: [NN_FEATS.index(f) for f in fs] for g, fs in md.MODALITIES.items()}
MM_VARIANTS = ("mm_early", "mm_graph")
MM_LABEL = {"mm_early": "STGNN-mm (early fusion)", "mm_graph": "STGNN-mm (early fusion + modal edges)"}


def _log(*a):
    print(time.strftime("%H:%M:%S"), "[stgnn-mm]", *a, flush=True)


# ============================================================================ node features
def node_features(M: "md.ModalData", df: pd.DataFrame, graph, drop: Sequence[str] = ()) -> np.ndarray:
    """(T, N_nodes, F) float32 modal input channels (zeros for weather nodes / missing data)."""
    pv = list(graph.meta["pv_cols"]); n = len(pv)
    ts = pd.DatetimeIndex(df.index)
    A = M.site_arrays(pv, ts, site_p=df[pv].to_numpy(float), drop=drop)
    has = np.nan_to_num(A["has"]).astype(bool)
    tr = {"has": lambda x: x,
          "meter": lambda x: 2.0 * np.clip(x, -1, 1),
          "sib": lambda x: np.clip(1.0 - x, -1, 1),
          "vac": lambda x: np.clip(x / 5.0, -2, 3),
          "f_dev": lambda x: np.clip(x / 0.1, 0, 5),
          "imb": lambda x: np.clip(x, 0, 2),
          "n_zero": lambda x: np.clip(x / 12.0, 0, 1),
          "dc_off": lambda x: np.clip(x, 0, 1)}
    X = np.zeros((len(ts), len(graph.node_names), len(NN_FEATS)), np.float32)
    for i, f in enumerate(NN_FEATS):
        v = tr[f](A[f].astype(float))
        v = np.where(has, np.nan_to_num(v), 0.0)
        X[:, :n, i] = v
    return X


def modal_signature(M: "md.ModalData") -> str:
    h = hashlib.sha256()
    for k in sorted(M.inv):
        h.update(k.encode()); h.update(np.nan_to_num(M.inv[k]["P"].to_numpy(np.float64)).tobytes())
    return h.hexdigest()[:12]


# ============================================================================ graph
def _complex(name: str) -> str:
    s = md._short(name)
    s = re.sub(r"\(.*?\)", "", s)
    m = re.match(r"^([A-Za-z]+(?:\s+[A-Za-z]+)?)", s.strip())
    return (m.group(1) if m else s).strip().lower()


def modal_graph(graph, M: "md.ModalData", D: dict, k: int = 2, min_corr: float = 0.8, boundary: str = "2023-01-01"):
    """Copy of ``graph`` with AC-voltage-correlation edges (type t+1) and hierarchy edges (type t+2)."""
    g = copy.deepcopy(graph)
    pv = list(g.meta["pv_cols"]); idx = {c: g.node_names.index(c) for c in pv}
    t_v, t_h = int(np.max(g.edge_type)) + 1, int(np.max(g.edge_type)) + 2
    src, dst, et, w = [], [], [], []
    sites = [c for c in M.sites() if c in idx]
    b = pd.Timestamp(boundary)
    V = pd.DataFrame({c: pd.concat([M.inv[kk]["vac"] for kk in M.inverters_of(c)], axis=1).mean(1) for c in sites})
    Vt = V[V.index < b]
    if Vt.notna().sum().min() < 200 if len(Vt.columns) else True:
        Vt = V
    R = Vt.corr(min_periods=200) if len(sites) >= 2 else pd.DataFrame()
    for c in sites:
        if c not in R:
            continue
        r = R[c].drop(c).dropna().sort_values(ascending=False)
        for o, v in r[r >= min_corr].head(k).items():
            src.append(idx[o]); dst.append(idx[c]); et.append(t_v); w.append(float(v))
    loc = {}
    brick, mapping = D.get("brick"), D.get("mapping") or {}
    for c in pv:
        s = brick.sites.get(mapping.get(c)) if brick is not None and hasattr(brick, "sites") else None
        loc[c] = (getattr(s, "location", None) or _complex(c)) if s is not None else _complex(c)
    for c in pv:
        for o in pv:
            if o != c and loc[o] == loc[c]:
                src.append(idx[o]); dst.append(idx[c]); et.append(t_h); w.append(0.5)
    if src:
        g.edge_index = np.hstack([g.edge_index, np.vstack([src, dst]).astype(np.int64)])
        g.edge_type = np.concatenate([g.edge_type, np.asarray(et)])
        wn = np.clip(np.asarray(w, float), 1e-6, None)
        g.edge_weight = np.concatenate([g.edge_weight, wn])
        g.edge_phys_logit = np.concatenate([np.asarray(g.edge_phys_logit, float), np.log(wn)]).astype(np.asarray(graph.edge_phys_logit).dtype)
    g.meta = dict(g.meta, modal_edges=dict(voltage=int(np.sum(np.asarray(et) == t_v)), hierarchy=int(np.sum(np.asarray(et) == t_h))))
    return g


# ============================================================================ dataset / pipe
def _make_modal_dataset(ds, node_feat: np.ndarray):
    """Turn a PIGraphDataset instance into the modal subclass in place (shared arrays)."""
    from pv_physics_graph import PIGraphDataset

    class ModalGraphDataset(PIGraphDataset):
        def __getitem__(self, i):
            d = PIGraphDataset.__getitem__(self, i)
            t0 = self.idx[i]
            nf = self.node_feat[t0:t0 + self.window].transpose(1, 2, 0).copy()      # (N, F, W)
            if self.p_group > 0 or self.p_all > 0:
                rng = np.random.default_rng()
                N = nf.shape[0]
                for g, ch in GROUP_CH.items():
                    m = np.flatnonzero(rng.random(N) < self.p_group)
                    if len(m):
                        nf[np.ix_(m, ch)] = 0.0
                m = rng.random(N) < self.p_all
                nf[m] = 0.0
            d["x"] = np.concatenate([d["x"], nf], 1).astype(np.float32)
            return d

    if not isinstance(ds, PIGraphDataset) or getattr(ds, "_modal", False):
        return ds
    ds.__class__ = ModalGraphDataset
    ds.node_feat = node_feat; ds.p_group = 0.0; ds.p_all = 0.0; ds._modal = True
    ds.in_ch = ds.in_ch + node_feat.shape[2]
    return ds


def make_pipe(df, graph, phys, node_feat: np.ndarray, p_group: float = 0.15, p_all: float = 0.15, **kw):
    """UnsupervisedSTGNN with ``node_feat.shape[2]`` extra node channels and modality dropout."""
    import torch
    from pv_stgnn_unsupervised import UnsupervisedSTGNN
    from pv_physics_graph import PhysicsInformedSTGNN

    class MultimodalSTGNN(UnsupervisedSTGNN):
        def split(self, boundary, val_fraction: float = 0.1):
            UnsupervisedSTGNN.split(self, boundary, val_fraction)
            for ds in (self.ds_full, self.ds_tr, self.ds_te):
                _make_modal_dataset(ds, self.node_feat)
            return self

        def fit(self, *a, **k):
            for ds in (self.ds_tr,):
                ds.p_group, ds.p_all = self.p_group, self.p_all
            try:
                return UnsupervisedSTGNN.fit(self, *a, **k)
            finally:
                self.ds_tr.p_group = self.ds_tr.p_all = 0.0

    pipe = MultimodalSTGNN(df, graph, phys, **kw)
    pipe.node_feat = node_feat; pipe.p_group, pipe.p_all = p_group, p_all
    torch.manual_seed(kw.get("seed", 0))
    in_ch = 5 + (0 if kw.get("context") is None else int(np.asarray(kw["context"]).shape[1])) + node_feat.shape[2]
    pipe.model = PhysicsInformedSTGNN(graph, in_ch=in_ch, hidden=kw.get("hidden", 64), tcn_layers=kw.get("tcn_layers", 4),
                                      gat_layers=kw.get("gat_layers", 2), heads=kw.get("heads", 4),
                                      window=kw.get("window", 48), horizon=kw.get("horizon", 1)).to(pipe.dev)
    pipe.opt = torch.optim.Adam(pipe.model.parameters(), lr=1e-3, weight_decay=1e-5)
    pipe.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(pipe.opt, factor=0.5, patience=3)
    return pipe


# ============================================================================ frozen multimodal model
class FrozenMM(sse.FrozenSTGNN):
    """Frozen multimodal STGNN.  ``build(df, phys, modal=..., drop=...)`` uses the modal channels of
    a (semi-synthetic) realisation; ``drop`` removes modality groups at test time."""

    def __init__(self, ckpt, D, C, M, mm_variant="mm_graph", window=48, hidden=32, seed=0, device=None, source=""):
        self.M, self.mm_variant = M, mm_variant
        self.graph = modal_graph(C["graph_mv"], M, D) if mm_variant == "mm_graph" else C["graph_mv"]
        super().__init__(ckpt, D, C, "multiview", window, hidden, seed, device, source)

    def build(self, df, phys, modal=None, drop: Sequence[str] = ()):
        kw = sse.pipe_kwargs(self.D, self.C, "multiview", self.window, self.hidden, self.seed, self.device)
        kw.pop("graph")
        nf = node_features(modal or self.M, df, self.graph, drop)
        with contextlib.redirect_stdout(io.StringIO()):
            pipe = make_pipe(df, self.graph, phys, nf, 0.0, 0.0, **kw)
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
    def train(cls, D, C, M, seed=0, mm_variant="mm_graph", window=48, hidden=32, epochs=15, train_stride=1,
              device=None, save_to=None, p_group=0.15, p_all=0.15, verbose=False):
        from pv_physics_graph import data_signature
        obj = cls.__new__(cls)
        obj.M, obj.mm_variant = M, mm_variant
        g = modal_graph(C["graph_mv"], M, D) if mm_variant == "mm_graph" else C["graph_mv"]
        kw = sse.pipe_kwargs(D, C, "multiview", window, hidden, seed, device); kw.pop("graph")
        nf = node_features(M, D["df"], g)
        pipe = make_pipe(D["df"], g, D["phys"], nf, p_group, p_all, **kw)
        pipe.split(sse.BOUNDARY, val_fraction=0.1)
        pipe.fit(epochs=epochs, batch_size=64, patience=3, train_stride=train_stride, verbose=verbose)
        pipe.calibrate(q=0.995, cal_stride=2)
        ck = sse.make_checkpoint(pipe, dict(variant="multiview", mm_variant=mm_variant, seed=seed, window=window,
                                            hidden=hidden, epochs=epochs, train_stride=train_stride,
                                            modal_sha=modal_signature(M), p_group=p_group, p_all=p_all,
                                            **data_signature(D, sse.BOUNDARY)))
        if save_to:
            sse.save_checkpoint(ck, save_to)
        return cls(ck, D, C, M, mm_variant, window, hidden, seed, device, source="trained multimodal model")


def checkpoint_path(cache_dir: str, mm_variant: str, seed: int) -> str:
    return os.path.join(cache_dir, f"stgnn_mm_{mm_variant}_{seed}.pt")


def frozen_mm_models(D, C, M, seeds=(0,), mm_variant="mm_graph", cache_dir="results/cache", window=48, hidden=32,
                     epochs=15, train_stride=1, device=None, train_missing=True) -> Dict[int, FrozenMM]:
    """Checkpoint (same data + same modal data) > training once, then freezing."""
    import torch
    out = {}
    for s in seeds:
        p = checkpoint_path(cache_dir, mm_variant, s); mdl = None
        if os.path.exists(p):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            status, why = sse.check_checkpoint(ck, D, C, "multiview")
            m = ck.get("meta", {})
            if status == "ok" and m.get("modal_sha") == modal_signature(M) and m.get("mm_variant") == mm_variant:
                mdl = FrozenMM(ck, D, C, M, mm_variant, m.get("window", window), m.get("hidden", hidden), s, device,
                               source=f"checkpoint {os.path.basename(p)}")
            else:
                _log(f"{os.path.basename(p)} not used ({why or 'modal data differ'}); retraining")
        if mdl is None:
            if not train_missing:
                raise FileNotFoundError(p)
            _log(f"training {mm_variant} seed {s} ({epochs} epochs, stride {train_stride}), then freezing")
            mdl = FrozenMM.train(D, C, M, s, mm_variant, window, hidden, epochs, train_stride, device, p)
        out[s] = mdl
        _log(f"{mm_variant} seed {s}: {mdl.source}, weights sha256[:16] = {mdl.fingerprint()}")
    return out


def collect_mm(mdl: FrozenMM, D: dict, reals: Dict[object, object], batch_size: int = 64,
               drop: Sequence[str] = (), clean_drop: Optional[Sequence[str]] = None):
    """``sss.ModelRaw`` of a frozen multimodal model: calibration (train, stride 2), validation,
    clean test and every realisation (with its own modal channels)."""
    import pv_semisynth_sensitivity as sss
    fp0 = mdl.fingerprint()
    pipe = mdl.build(D["df"], D["phys"], drop=clean_drop if clean_drop is not None else drop)
    cal = sss._collect(pipe, pipe.ds_tr, np.asarray(pipe._fit_idx)[::2], batch_size)
    clean = sss._collect(pipe, pipe.ds_te, pipe.ds_te.idx, batch_size)
    nr = sss._not_ready(pipe, clean["pos"])
    vi = np.asarray(pipe._val_idx)
    if not len(vi):
        vi = np.asarray(pipe._fit_idx)[-max(len(pipe._fit_idx) // 10, 1):]
    val = sss._collect(pipe, pipe.ds_tr, vi, batch_size)
    val_nr = sss._not_ready(pipe, val["pos"])
    calib = dict(stats={k: v.detach().cpu().numpy() for k, v in pipe.stats.items()},
                 q=float(getattr(pipe, "q", 0.995) or 0.995),
                 node_thresholds=np.asarray(pipe.node_thresholds, float).copy(),
                 regime_thresholds=None if pipe.regime_thresholds is None else np.asarray(pipe.regime_thresholds).copy(),
                 cluster_threshold_prior=None if pipe.cluster_threshold_prior is None
                 else np.asarray(pipe.cluster_threshold_prior).copy(),
                 threshold_pot=float(getattr(pipe, "threshold_pot", 3.0)),
                 regime=None if pipe.regime is None else np.asarray(pipe.regime),
                 cluster_labels=pipe.cluster_labels, min_regime_count=int(pipe.min_regime_count),
                 min_stats_count=int(getattr(pipe, "min_stats_count", 20)), n_pv=int(pipe.n_pv))
    pv = list(pipe.graph.meta["pv_cols"])
    del pipe
    inj = {}
    for k, real in reals.items():
        p = mdl.build(real.df, real.phys, modal=getattr(real, "modal", None), drop=drop)
        inj[k] = sss._collect(p, p.ds_te, p.ds_te.idx, batch_size)
        del p
    assert mdl.fingerprint() == fp0, "weights changed"
    R = sss.ModelRaw(mdl.seed, cal, clean, inj, nr, calib, pv, fp0)
    R.val, R.val_not_ready = val, val_nr
    return R
