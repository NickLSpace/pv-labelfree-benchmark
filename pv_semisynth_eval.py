"""
Semi-synthetic validation of the trained SS-PI-STGNN with FROZEN weights (independent module).

Workflow (see the last section of the notebook):

    import pv_semisynth_gen as ssg, pv_semisynth_eval as sse
    models = sse.frozen_models(D, C, seeds=[0, 1], cache_dir="results/cache")   # load / train once, freeze
    out = sse.run_semisynthetic(D, C, models, placement_seeds=[0, 1, 2],
                                estimators=("sps", "kalman"), A=A)
    sse.show(out); sse.export_latex(out, "results/semisynth")

Frozen weights
--------------
``FrozenSTGNN`` keeps one immutable checkpoint (state_dict + train-period calibration:
z-statistics, node / regime / cluster thresholds).  For every (semi-synthetic) data set
a fresh ``UnsupervisedSTGNN`` is built on the modified frame, the checkpoint is loaded,
all parameters get ``requires_grad=False``, ``fit`` / ``online_fine_tune`` are disabled,
and only ``score('test')`` is called.  A SHA-256 fingerprint of the weights is checked
before and after the validation.  Injections are restricted to the test period, so the
train-only normalisation of the dataset is identical to the clean run.

Metrics
-------
Every injected run is compared with the CLEAN run of the same frozen model (paired
design: real anomalies already present in the data are not counted).

* event-level recall / precision / F1 (detections = contiguous exceedances, min_len=3;
  a detection is credited to an injected event if it contains at least one NEW
  exceedance — present in the injected run, absent in the clean run — inside
  [effective start, effective end + tolerance]);
* detection delay (h) from the first effective injected hour;
* false alarms on control plants (new detections per 1000 scored hours) and the real
  alarm rate of the clean run on the same plants;
* minimum detectable loss: the smallest strength with recall >= 0.5 (and log-linear
  interpolation between strengths);
* response of eta_hat and of the physical residual z: paired mean difference
  injected - clean over the effective hours;
* change of b1 (SPS and Kalman): quarterly b1 injected vs clean against the exact
  expected change (ratio of sums of P / P_exp over the valid hours of the quarter);
  degradation: train/test b1, region disjointness and the fused evidence;
* fuzzy inference: share of detected events whose largest membership is the injected
  type (``pv_fuzzy``), plus the crisp rules and the K-Means anomaly clusters for reference.

The module imports the existing project modules read-only; nothing in the existing
pipeline imports it.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import io
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import pv_semisynth_gen as ssg

BOUNDARY = "2023-01-01"
CALIB_ATTRS = ("threshold", "threshold_pot", "node_thresholds", "regime_thresholds",
               "cluster_threshold_prior", "q")
EST_LABELS = {"sps": "SPS", "kalman": "Kalman"}


def _log(*a):
    print(time.strftime("%H:%M:%S"), "[semisynth]", *a, flush=True)


# ============================================================================ frozen model
def checkpoint_path(cache_dir: str, variant: str, seed: int) -> str:
    return os.path.join(cache_dir, f"stgnn_frozen_{variant}_{seed}.pt")


def _repair_legacy_stats(ck: dict, D: dict, C: dict) -> int:
    """Checkpoints saved before the cold-start fix keep the default error normalisation
    (mu = 0, sd = 1) for PV nodes without train data.  Apply the same technical-cluster borrowing
    as UnsupervisedSTGNN.calibrate; weights and thresholds are not touched.  Returns #nodes fixed."""
    if ck.get("stats_source") is not None:
        return 0
    from pv_stgnn_unsupervised import borrow_error_stats
    pv = list(D["pv_cols"]); n = len(pv)
    tech = C.get("tech"); lab = getattr(tech, "labels", None)
    fixed, src_all = 0, {}
    for k in ("rec", "fore", "phy"):
        mu, sd = ck["stats"].get(f"{k}_mu"), ck["stats"].get(f"{k}_sd")
        if mu is None:
            continue
        has = ~((mu == 0) & (sd == 1)).cpu().numpy()
        has[n:] = True
        mu2, sd2, src = borrow_error_stats(mu.cpu(), sd.cpu(), has, n, lab, pv)
        ck["stats"][f"{k}_mu"], ck["stats"][f"{k}_sd"] = mu2, sd2
        fixed = max(fixed, int((~has[:n]).sum())); src_all[k] = src
    ck["stats_source"] = src_all
    if fixed:
        _log(f"legacy checkpoint: error normalisation of {fixed} node(s) without train data borrowed "
            f"from the technical cluster")
    return fixed


class CheckpointMismatch(ValueError):
    """The checkpoint was trained on other data (other plants or other train-period values)."""


def check_checkpoint(ck: dict, D: dict, C: dict, variant: str = "multiview") -> Tuple[str, str]:
    """('ok' | 'legacy' | 'data' | 'incompatible', reason).

    incompatible : other plants / node count — the weights cannot describe these data;
    data         : same plants, other train-period values (other fleet realisation or preprocessing);
    legacy       : checkpoint without a data signature (written before this check) — accepted."""
    pv = list(D["pv_cols"]); m = ck.get("meta", {}) or {}
    g = C["graph_mv"] if variant == "multiview" else D["graph_base"]
    n_nodes = len(g.node_names)
    st = ck.get("stats", {})
    for k, n in (("rec_mu", n_nodes), ("fore_mu", n_nodes), ("phy_mu", len(pv))):
        if k in st and int(st[k].shape[0]) != n:
            return "incompatible", f"{k} has {int(st[k].shape[0])} entries, the data have {n}"
    if m.get("pv_cols") is not None and list(m["pv_cols"]) != pv:
        return "incompatible", f"checkpoint plants ({len(m['pv_cols'])}) differ from the data plants ({len(pv)})"
    if m.get("train_sha") is None:
        return "legacy", "no data signature in the checkpoint"
    from pv_physics_graph import data_signature
    sha = data_signature(D, BOUNDARY)["train_sha"]
    if m["train_sha"] != sha:
        return "data", f"train-period data differ (checkpoint {m['train_sha']}, data {sha})"
    return "ok", ""


def make_checkpoint(pipe, meta: Optional[dict] = None) -> dict:
    """Immutable snapshot of trained weights + train-period calibration of a pipe."""
    ck = dict(model={k: v.detach().cpu().clone() for k, v in pipe.model.state_dict().items()},
              stats={k: v.detach().cpu().clone() for k, v in pipe.stats.items()},
              meta=dict(meta or {}))
    for a in CALIB_ATTRS:
        v = getattr(pipe, a, None)
        ck[a] = copy.deepcopy(v)
    ck["stats_source"] = copy.deepcopy(getattr(pipe, "stats_source", None)) or None
    return ck


def save_checkpoint(ck: dict, path: str):
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(ck, path)


def pipe_kwargs(D: dict, C: dict, variant: str = "multiview", window: int = 48, hidden: int = 32,
                seed: int = 0, device: Optional[str] = None) -> dict:
    """Constructor arguments identical to ``run_pipeline.run_variant``."""
    from pv_physics_graph import node_activity_mask
    mv = variant == "multiview"
    graph = C["graph_mv"] if mv else D["graph_base"]
    act = node_activity_mask(D["df"], graph, D["comm"])
    obs = D["observed"].reindex(columns=graph.node_names, fill_value=True)
    return dict(graph=graph, window=window, horizon=1, hidden=hidden, tcn_layers=4, gat_layers=2, heads=4,
                seed=seed, activity_mask=act, observed=obs, cold_start_steps=window, calibration_steps=168,
                context=C["wr"].onehot if mv else None, regime=C["wr"].regime if mv else None,
                cluster_labels=C["tech"].labels if mv else None, device=device)


def _blocked(*_a, **_k):
    raise RuntimeError("FrozenSTGNN: weights are frozen — training / fine-tuning is disabled")


@dataclass
class ScoreBundle:
    timestamps: pd.DatetimeIndex
    combined: np.ndarray     # (S, n_pv)
    thr: np.ndarray          # (S, n_pv)
    exceed: np.ndarray       # (S, n_pv) bool
    event_mask: np.ndarray   # (S, n_pv) bool — points inside detections (min_len)
    physics_z: np.ndarray
    eta: np.ndarray
    events: pd.DataFrame
    pv_cols: List[str]

    def rank(self) -> pd.Series:
        with np.errstate(all="ignore"):
            r = np.nanmean(self.combined, axis=0)
        return pd.Series(r, index=[ssg._short(c) for c in self.pv_cols])

    def exceed_share(self) -> pd.Series:
        fin = np.isfinite(self.combined)
        return pd.Series(self.exceed.sum(0) / np.maximum(fin.sum(0), 1), index=[ssg._short(c) for c in self.pv_cols])


def _run_mask(M: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only runs of True of length >= min_len (column-wise)."""
    out = np.zeros_like(M, bool)
    for j in range(M.shape[1]):
        m = M[:, j]
        if not m.any():
            continue
        d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a >= min_len:
                out[a:b, j] = True
    return out


class FrozenSTGNN:
    """Trained STGNN with frozen weights; scores arbitrary (semi-synthetic) frames."""

    def __init__(self, ckpt: dict, D: dict, C: dict, variant: str = "multiview", window: int = 48,
                 hidden: int = 32, seed: int = 0, device: Optional[str] = None, source: str = ""):
        self.ckpt, self.D, self.C = ckpt, D, C
        self.variant, self.window, self.hidden, self.seed = variant, window, hidden, seed
        self.device, self.source = device, source
        self.stats_repaired = _repair_legacy_stats(ckpt, D, C)
        self._fp = self.fingerprint()

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_pipe(cls, pipe, D, C, variant="multiview", seed=0, source="notebook pipe"):
        ck = make_checkpoint(pipe, dict(variant=variant, seed=seed, window=pipe.window,
                                        hidden=pipe.model.fc_fore[0].in_features))
        return cls(ck, D, C, variant, pipe.window, pipe.model.fc_fore[0].in_features, seed, pipe.dev, source)

    @classmethod
    def load(cls, path, D, C, device=None, variant="multiview", seed=0, window=48, hidden=32, check=True):
        """check=True: raise CheckpointMismatch if the checkpoint belongs to other data."""
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = ck.get("meta", {})
        if check:
            status, why = check_checkpoint(ck, D, C, m.get("variant", variant))
            if status in ("incompatible", "data"):
                raise CheckpointMismatch(f"{os.path.basename(path)}: {why}")
            if status == "legacy":
                _log(f"{os.path.basename(path)}: {why} (accepted; written before the data check)")
        return cls(ck, D, C, m.get("variant", variant), m.get("window", window), m.get("hidden", hidden),
                   m.get("seed", seed), device, source=f"checkpoint {os.path.basename(path)}")

    @classmethod
    def train(cls, D, C, seed=0, variant="multiview", window=48, hidden=32, epochs=15, train_stride=1,
              device=None, save_to: Optional[str] = None, verbose=False):
        """Train exactly like ``run_pipeline.run_variant`` (fit + calibrate), then freeze."""
        from pv_stgnn_unsupervised import UnsupervisedSTGNN
        kw = pipe_kwargs(D, C, variant, window, hidden, seed, device)
        g = kw.pop("graph")
        pipe = UnsupervisedSTGNN(D["df"], g, D["phys"], **kw)
        pipe.split(BOUNDARY, val_fraction=0.1)
        pipe.fit(epochs=epochs, batch_size=64, patience=3, train_stride=train_stride, verbose=verbose)
        pipe.calibrate(q=0.995, cal_stride=2)
        from pv_physics_graph import data_signature
        ck = make_checkpoint(pipe, dict(variant=variant, seed=seed, window=window, hidden=hidden,
                                        epochs=epochs, train_stride=train_stride, **data_signature(D, BOUNDARY)))
        if save_to:
            save_checkpoint(ck, save_to)
        return cls(ck, D, C, variant, window, hidden, seed, device, source="trained for semi-synthetic test")

    # ---------------------------------------------------------------- integrity
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.ckpt["model"]):
            h.update(k.encode()); h.update(self.ckpt["model"][k].numpy().tobytes())
        return h.hexdigest()[:16]

    def unchanged(self) -> bool:
        return self.fingerprint() == self._fp

    # ---------------------------------------------------------------- scoring
    def build(self, df: pd.DataFrame, phys: Dict[str, np.ndarray]):
        import torch
        from pv_stgnn_unsupervised import UnsupervisedSTGNN
        kw = pipe_kwargs(self.D, self.C, self.variant, self.window, self.hidden, self.seed, self.device)
        g = kw.pop("graph")
        with contextlib.redirect_stdout(io.StringIO()):
            pipe = UnsupervisedSTGNN(df, g, phys, **kw)
            pipe.split(BOUNDARY, val_fraction=0.1)
        pipe.model.load_state_dict({k: v.to(pipe.dev) for k, v in self.ckpt["model"].items()})
        pipe.stats = {k: v.clone().to(pipe.dev) for k, v in self.ckpt["stats"].items()}
        for a in CALIB_ATTRS:
            if self.ckpt.get(a) is not None or a in ("regime_thresholds", "cluster_threshold_prior"):
                setattr(pipe, a, copy.deepcopy(self.ckpt.get(a)))
        pipe.model.eval()
        for p in pipe.model.parameters():
            p.requires_grad_(False)
        pipe.fit = _blocked; pipe.online_fine_tune = _blocked; pipe.opt = None
        return pipe

    def score(self, df: pd.DataFrame, phys: Dict[str, np.ndarray], min_len: int = 3,
              batch_size: int = 64) -> ScoreBundle:
        pipe = self.build(df, phys)
        with contextlib.redirect_stdout(io.StringIO()):
            res = pipe.score("test", batch_size=batch_size)
        n = pipe.n_pv
        comb = res.combined[:, :n]
        thr = np.asarray(pipe.last_thr)
        with np.errstate(invalid="ignore"):
            exc = np.nan_to_num(comb, nan=-np.inf) > thr
        ev = pipe.events(res, min_len=min_len)
        out = ScoreBundle(pd.DatetimeIndex(res.timestamps), comb, thr, exc, _run_mask(exc, min_len),
                          res.physics_z[:, :n], res.eta[:, :n], ev, list(pipe.graph.meta["pv_cols"]))
        del pipe
        return out


def frozen_models(D, C, seeds: Sequence[int] = (0,), variant: str = "multiview", cache_dir: str = "results/cache",
                  window: int = 48, hidden: int = 32, epochs: int = 15, train_stride: int = 1,
                  device: Optional[str] = None, existing: Optional[Dict[int, object]] = None,
                  train_missing: bool = True) -> Dict[int, FrozenSTGNN]:
    """One frozen model per seed: notebook pipe (``existing``) > checkpoint > training.

    Checkpoints ``stgnn_frozen_<variant>_<seed>.pt`` are written by run_pipeline (before
    the online fine-tuning) or by this function after training."""
    out = {}
    for s in seeds:
        if existing and s in existing and existing[s] is not None:
            out[s] = FrozenSTGNN.from_pipe(existing[s], D, C, variant, s)
        else:
            p = checkpoint_path(cache_dir, variant, s)
            loaded = None
            if os.path.exists(p):
                try:
                    loaded = FrozenSTGNN.load(p, D, C, device, variant, s, window, hidden)
                except CheckpointMismatch as exc:
                    if not train_missing:
                        raise
                    _log(f"checkpoint not used — {exc}; retraining and overwriting")
            if loaded is not None:
                out[s] = loaded
            elif train_missing:
                _log(f"no checkpoint for seed {s}: training {variant} once ({epochs} epochs), then freezing")
                out[s] = FrozenSTGNN.train(D, C, s, variant, window, hidden, epochs, train_stride, device, p)
            else:
                raise FileNotFoundError(p)
        _log(f"model seed {s}: {out[s].source}, weights sha256[:16] = {out[s].fingerprint()}")
    return out


# ============================================================================ detection metrics
@dataclass
class EvalConfig:
    tol_after_h: int = 48            # detections up to window length after the end still count
    neighbourhood_days: int = 7      # FP attributed to an event cell within +/- this many days
    fleet_share: float = 0.2         # fleet event detected if >= this share of plants detect it
    min_len: int = 3
    recall_target: float = 0.5       # minimum detectable loss criterion
    fuzzy_config: Optional[object] = None


def _pos_range(ts: pd.DatetimeIndex, a, b) -> np.ndarray:
    return np.flatnonzero((ts >= a) & (ts <= b))


def _classify(feat: pd.DataFrame, A: Optional[dict], fz_cfg) -> pd.DataFrame:
    """Fuzzy (argmax membership), crisp rules and K-Means anomaly-cluster class."""
    import pv_clustering as clu
    import pv_fuzzy as fz
    out = pd.DataFrame(index=feat.index)
    if not len(feat):
        return out.assign(fuzzy=[], fuzzy_mu=[], crisp=[], kmeans=[])
    mu = fz.fuzzy_classify_batch(feat, fz_cfg or fz.FuzzyConfig())
    out["fuzzy"] = mu.idxmax(1)
    out["fuzzy_mu"] = mu.max(1)
    out["crisp"] = feat.apply(clu._label_center, axis=1).values
    anom = (A or {}).get("anom")
    if anom is not None and getattr(anom, "model", None) is not None:
        X0 = feat[clu.ANOMALY_FEATURES].astype(float).fillna(0.0)
        X0["duration_h"] = np.log1p(X0["duration_h"])
        lab = anom.model.predict(anom.scaler.transform(X0.to_numpy(float)))
        out["kmeans"] = [anom.extra["class_names"].get(int(l), "") for l in lab]
    else:
        out["kmeans"] = np.nan
    return out


def evaluate_detection(D, C, real: "ssg.Realisation", clean: ScoreBundle, inj: ScoreBundle,
                       cfg: Optional[EvalConfig] = None, A: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """Per-event detection rows, per-detection rows and control-plant false alarms."""
    import pv_clustering as clu
    cfg = cfg or EvalConfig()
    ts = inj.timestamps
    pv = inj.pv_cols
    new = inj.event_mask & ~clean.exceed
    rawnew = inj.exceed & ~clean.exceed
    roles = real.roles.reindex(pv)
    lab = real.labels[real.labels["effective"] & (real.labels["level"] != "trend")].copy()
    tol = pd.Timedelta(hours=cfg.tol_after_h)
    nb = pd.Timedelta(days=cfg.neighbourhood_days)
    regime = C["wr"].regime
    rows, det_feat, oracle_feat = [], [], []

    # detections of the injected run that contain new points
    ev = inj.events.copy()
    if len(ev):
        col_of = {ssg._short(c): j for j, c in enumerate(pv)}
        ev["j"] = ev["plant"].map(col_of)
        ev["new"] = [bool(new[_pos_range(ts, r.start, r.end), int(r.j)].any()) for r in ev.itertuples()]
        ev["assigned"] = -1; ev["cell_fp"] = -1
    else:
        ev = pd.DataFrame(columns=["plant", "start", "end", "j", "new", "assigned", "cell_fp", "peak_score",
                                   "mean_physics_z"])

    for r in lab.itertuples():
        a, b = r.eff_start, r.eff_end
        W = _pos_range(ts, a, b + tol)
        E = _pos_range(ts, a, b)
        base = dict(event_id=r.event_id, scenario=r.scenario, level=r.level, plant=r.plant, strength=r.strength,
                    duration=r.duration, weather=r.weather, tech_cluster=r.tech_cluster, n_eff_h=r.n_eff_h,
                    true_loss_pct=r.true_loss_pct, pexp_bias_pct=r.pexp_bias_pct, expected_class=r.expected_class,
                    start=a, end=b, cohort=getattr(r, "cohort", np.nan),
                    days_since_commissioning=getattr(r, "days_since_commissioning", np.nan))
        if r.level == "plant":
            j = pv.index(r.column)
            hit = new[W, j]
            det = bool(hit.any())
            delay = (ts[W[np.argmax(hit)]] - a) / pd.Timedelta(hours=1) if det else np.nan
            em = real.inj_mask[r.column].reindex(ts[E]).to_numpy(bool) if len(E) else np.zeros(0, bool)
            Ee = E[em] if em.any() else E
            with np.errstate(all="ignore"):
                d_eta = float(np.nanmean(inj.eta[Ee, j] - clean.eta[Ee, j])) if len(Ee) else np.nan
                d_phy = float(np.nanmean(inj.physics_z[Ee, j] - clean.physics_z[Ee, j])) if len(Ee) else np.nan
                d_sc = float(np.nanmean(inj.combined[Ee, j] - clean.combined[Ee, j])) if len(Ee) else np.nan
            # credited detections
            if len(ev):
                m = (ev["j"] == j) & ev["new"] & (ev["start"] <= b + tol) & (ev["end"] >= a)
                ev.loc[m & (ev["assigned"] < 0), "assigned"] = r.event_id
                cand = ev[m]
                if det and len(cand):
                    ov = [(min(e.end, b + tol) - max(e.start, a)) for e in cand.itertuples()]
                    e = cand.iloc[int(np.argmax(ov))]
                    det_feat.append(dict(event_id=r.event_id, plant=r.plant, start=e["start"], end=e["end"],
                                         peak_score=e["peak_score"], mean_physics_z=e["mean_physics_z"]))
            if len(E):
                with np.errstate(all="ignore"):
                    oracle_feat.append(dict(event_id=r.event_id, plant=r.plant, start=a, end=b,
                                            peak_score=float(np.nanmax(np.r_[inj.combined[E, j], -np.inf])),
                                            mean_physics_z=float(np.nan_to_num(np.nanmean(inj.physics_z[E, j])))))
            rows.append(dict(base, detected=det, raw_hit=bool(rawnew[W, j].any()), delay_h=delay,
                             d_eta=d_eta, d_physics_z=d_phy, d_score=d_sc, fleet_share_detected=np.nan,
                             n_plants=1))
        else:   # fleet event: every plant with a valid score in W
            act = np.isfinite(inj.combined[W]).any(0) if len(W) else np.zeros(len(pv), bool)
            act &= (roles != "degradation").to_numpy()
            js = np.flatnonzero(act)
            firsts = []
            for j in js:
                h = new[W, j]
                if h.any():
                    firsts.append(ts[W[np.argmax(h)]])
            share = len(firsts) / max(len(js), 1)
            k = int(np.ceil(cfg.fleet_share * max(len(js), 1)))
            det = share >= cfg.fleet_share and len(firsts) >= k
            delay = (sorted(firsts)[k - 1] - a) / pd.Timedelta(hours=1) if det else np.nan
            with np.errstate(all="ignore"):
                d_eta = float(np.nanmean(inj.eta[np.ix_(E, js)] - clean.eta[np.ix_(E, js)])) if len(E) and len(js) else np.nan
                d_phy = float(np.nanmean(inj.physics_z[np.ix_(E, js)] - clean.physics_z[np.ix_(E, js)])) if len(E) and len(js) else np.nan
                d_sc = float(np.nanmean(inj.combined[np.ix_(E, js)] - clean.combined[np.ix_(E, js)])) if len(E) and len(js) else np.nan
            if len(ev):
                m = ev["new"] & (ev["start"] <= b + tol) & (ev["end"] >= a) & ev["j"].isin(js)
                ev.loc[m & (ev["assigned"] < 0), "assigned"] = r.event_id
                for e in ev[m].itertuples():
                    det_feat.append(dict(event_id=r.event_id, plant=e.plant, start=e.start, end=e.end,
                                         peak_score=e.peak_score, mean_physics_z=e.mean_physics_z))
            rows.append(dict(base, detected=bool(det), raw_hit=bool(rawnew[np.ix_(W, js)].any()) if len(js) else False,
                             delay_h=delay, d_eta=d_eta, d_physics_z=d_phy, d_score=d_sc,
                             fleet_share_detected=share, n_plants=len(js)))
    E = pd.DataFrame(rows)

    # false positives: new detections not credited to any injected event
    plant_lab = lab[lab["level"] == "plant"]
    fleet_lab = lab[lab["level"] == "fleet"]
    fp_cell = {}
    bg_rows = []
    if len(ev):
        for i, e in ev[ev["new"] & (ev["assigned"] < 0)].iterrows():
            c = pv[int(e["j"])]
            role = roles.get(c, "unused")
            if role == "degradation":
                continue
            in_fleet = ((fleet_lab["eff_start"] - nb <= e["end"]) & (fleet_lab["eff_end"] + nb >= e["start"]))
            near = plant_lab[(plant_lab["column"] == c) & (plant_lab["eff_start"] - nb <= e["end"]) &
                             (plant_lab["eff_end"] + nb >= e["start"])]
            if len(near):
                fp_cell[int(near.iloc[0]["event_id"])] = fp_cell.get(int(near.iloc[0]["event_id"]), 0) + 1
            elif in_fleet.any():
                eid = int(fleet_lab[in_fleet].iloc[0]["event_id"])
                fp_cell[eid] = fp_cell.get(eid, 0) + 1
            else:
                bg_rows.append(dict(plant=e["plant"], role=role, start=e["start"], end=e["end"]))
    tp_det = ev[ev["assigned"] >= 0].groupby("assigned").size() if len(ev) else pd.Series(dtype=int)
    if len(E):
        E["n_tp_detections"] = E["event_id"].map(tp_det).fillna(0).astype(int)
        E["n_fp_near"] = E["event_id"].map(fp_cell).fillna(0).astype(int)

    # control plants: alarm rates per 1000 scored hours outside fleet events
    fleet_hours = np.zeros(len(ts), bool)
    for r in fleet_lab.itertuples():
        fleet_hours[_pos_range(ts, r.eff_start - nb, r.eff_end + nb)] = True
    fa = []
    for c in [c for c in pv if roles.get(c) == "control"]:
        j = pv.index(c)
        ok = np.isfinite(clean.combined[:, j]) & ~fleet_hours
        h = int(ok.sum())
        def _n_events(M):
            m = M[:, j] & ok
            d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
            return int((d == 1).sum())
        n_new = len([1 for b_ in bg_rows if b_["plant"] == ssg._short(c)])
        coh = (int(real.plants.loc[c, "cohort"]) if getattr(real, "plants", None) is not None
               and c in real.plants.index else np.nan)
        fa.append(dict(plant=ssg._short(c), cohort=coh, hours=h, clean_alarms=_n_events(clean.event_mask),
                       injected_alarms=_n_events(inj.event_mask), induced_alarms=n_new))
    FA = pd.DataFrame(fa)
    BG = pd.DataFrame(bg_rows)

    # classification of detected (and of the true) intervals
    cls = pd.DataFrame()
    if det_feat or oracle_feat:
        parts = []
        for kind, rowsf in (("detected", det_feat), ("oracle", oracle_feat)):
            if not rowsf:
                continue
            evf = pd.DataFrame(rowsf)
            f = clu.anomaly_event_features(evf, real.df, real.phys, pv, regime)
            if not len(f):
                continue
            f["event_id"] = evf["event_id"].values[f["event_id"].values] if "event_id" in f else evf["event_id"].values
            c_ = _classify(f, A, cfg.fuzzy_config)
            keep = ["event_id", "plant", "duration_h", "mean_ratio", "zero_frac", "flatness", "mean_ghi",
                    "mean_hour"] + [c for c in getattr(clu, "SHAPE_FEATURES", []) if c in f]
            parts.append(pd.concat([f[keep].reset_index(drop=True),
                                    c_.reset_index(drop=True)], axis=1).assign(kind=kind))
        cls = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return dict(events=E, detections=ev, false_alarms=FA, background=BG, classes=cls)


# ============================================================================ b1 (SPS / Kalman)
def _cloud(D):
    return (1.0 - D["kt"].clip(0, 1.2)).fillna(1.0).to_numpy()


def _est_one(name: str, D: dict, C: dict, df: pd.DataFrame, phys: dict, col: str, kalman_cfg=None,
             t_cell: Optional[np.ndarray] = None):
    """Periods + split (+ Kalman daily track) for ONE plant (estimators are per-plant)."""
    j = list(D["pv_cols"]).index(col)
    ph = dict(p_expected=phys["p_expected"][:, [j]], ghi=phys["ghi"])
    tc_all = D["t_cell"] if t_cell is None else t_cell
    tc = tc_all[:, [j]] if np.ndim(tc_all) == 2 else tc_all
    valid = C["valid"][[col]]
    args = (df, ph, [col], D["rated"], valid, tc, _cloud(D))
    if name == "sps":
        import pv_sps as m
        per = m.sps_periods(*args, freq="QS")
        spl = m.sps_split(*args, BOUNDARY)
        return per, spl, None
    import pv_kalman as m
    cfg = kalman_cfg or m.KalmanConfig()
    tr = m.kalman_track(*args, cfg)
    per = m.kalman_periods(*args, cfg=cfg, tracks=tr)
    spl = m.kalman_split(*args, BOUNDARY, cfg=cfg, tracks=tr)
    daily = m.tracks_frame(tr, cfg.smoother)
    return per, spl, daily


def _ratio_change(D, C, df1, phys1, col, a, b, df0=None, phys0=None) -> float:
    """Exact expected relative change of b1 between clean and modified data:
    (sum P1 / sum Pexp1) / (sum P0 / sum Pexp0) - 1 over the valid hours in [a, b]."""
    df0 = D["df"] if df0 is None else df0
    phys0 = D["phys"] if phys0 is None else phys0
    j = list(D["pv_cols"]).index(col)
    m = C["valid"][col].to_numpy(bool) & np.asarray((df0.index >= a) & (df0.index <= b))
    if m.sum() < 10:
        return np.nan
    P0, P1 = df0[col].to_numpy(float)[m], df1[col].to_numpy(float)[m]
    E0, E1 = phys0["p_expected"][m, j], phys1["p_expected"][m, j]
    ok = np.isfinite(P0) & np.isfinite(P1)
    r0 = np.sum(P0[ok]) / max(np.sum(E0[ok]), 1e-9)
    r1 = np.sum(P1[ok]) / max(np.sum(E1[ok]), 1e-9)
    return (r1 / r0 - 1) * 100 if r0 > 0 else np.nan


class EstimatorCache:
    """Clean estimator output per (estimator, plant) is computed once and reused."""

    def __init__(self, kalman_cfg=None):
        self.clean: Dict[Tuple[str, str], tuple] = {}
        self.kalman_cfg = kalman_cfg

    def get_clean(self, name, D, C, col):
        k = (name, col)
        if k not in self.clean:
            self.clean[k] = _est_one(name, D, C, D["df"], D["phys"], col, self.kalman_cfg)
        return self.clean[k]


def evaluate_b1(D, C, plan: "ssg.Plan", real_plant: "ssg.Realisation", real_ghi: Optional["ssg.Realisation"],
                estimators: Sequence[str], cache: EstimatorCache,
                bundles: Optional[Dict[int, Tuple[ScoreBundle, ScoreBundle]]] = None) -> Dict[str, pd.DataFrame]:
    """Quarterly b1 response for plant events, degradation trend and fleet GHI shift."""
    lab = real_plant.labels
    rows, deg_rows = [], []
    for name in estimators:
        import pv_sps
        fuse = pv_sps.fuse_evidence
        cols = sorted(set(lab.loc[lab["level"].isin(["plant", "trend"]), "column"].dropna()))
        for col in cols:
            per0, spl0, d0 = cache.get_clean(name, D, C, col)
            per1, spl1, d1 = _est_one(name, D, C, real_plant.df, real_plant.phys, col, cache.kalman_cfg)
            p0 = per0.set_index("period")["theta_b1_Pexp"] if len(per0) else pd.Series(dtype=float)
            p1 = per1.set_index("period")["theta_b1_Pexp"] if len(per1) else pd.Series(dtype=float)
            for r in lab[(lab["column"] == col) & (lab["level"] == "plant") & lab["effective"]].itertuples():
                q = pd.Period(r.quarter)
                qs, qe = q.start_time, q.end_time
                est = (p1.get(qs, np.nan) / p0.get(qs, np.nan) - 1) * 100
                exp_ = _ratio_change(D, C, real_plant.df, real_plant.phys, col, qs, qe)
                loc_est = loc_exp = np.nan
                if d0 is not None and r.duration != "short":
                    a, b = r.eff_start.normalize(), r.eff_end.normalize()
                    s0 = d0.set_index("day")["b1_Pexp"]; s1 = d1.set_index("day")["b1_Pexp"]
                    m0 = s0[(s0.index >= a) & (s0.index <= b)]; m1 = s1.reindex(m0.index)
                    if len(m0):
                        loc_est = (m1.mean() / m0.mean() - 1) * 100
                        loc_exp = _ratio_change(D, C, real_plant.df, real_plant.phys, col, a,
                                                b + pd.Timedelta(hours=23))
                disj = np.nan
                if len(per1):
                    g = per1.set_index("period")
                    if qs in g.index and len(per0) and qs in per0.set_index("period").index:
                        g0 = per0.set_index("period").loc[qs]
                        disj = bool(g.loc[qs, "b1_hi"] < g0["b1_lo"] or g.loc[qs, "b1_lo"] > g0["b1_hi"])
                rows.append(dict(estimator=name, event_id=r.event_id, scenario=r.scenario, strength=r.strength,
                                 duration=r.duration, plant=r.plant, quarter=r.quarter, expected_b1_pct=exp_,
                                 estimated_b1_pct=est, local_expected_pct=loc_exp, local_estimated_pct=loc_est,
                                 region_vs_clean_disjoint=disj,
                                 shared_quarter=bool(getattr(r, "shared_quarter", False) is True),
                                 cohort=getattr(r, "cohort", np.nan)))
            for r in lab[(lab["column"] == col) & (lab["level"] == "trend")].itertuples():
                s0 = spl0.set_index("plant").iloc[0]; s1 = spl1.set_index("plant").iloc[0]
                exp_ = _ratio_change(D, C, real_plant.df, real_plant.phys, col, r.start, r.end)
                est = (s1["test_b1"] / s0["test_b1"] - 1) * 100
                rec = dict(estimator=name, event_id=r.event_id, plant=r.plant, strength=r.strength,
                           cohort=getattr(r, "cohort", np.nan),
                           final_loss_pct=r.strength * 100, expected_b1_pct=exp_, estimated_b1_pct=est,
                           change_pct_clean=s0["b1_change_pct"], change_pct_injected=s1["b1_change_pct"],
                           disjoint_injected=s1["regions_disjoint"], disjoint_clean=s0["regions_disjoint"],
                           test_lo=s1["test_b1_lo"], test_hi=s1["test_b1_hi"])
                # quarterly trajectory slope (per quarter, % of train b1)
                if len(per1):
                    g1 = per1[per1["period"] >= pd.Timestamp(BOUNDARY)].set_index("period")["theta_b1_Pexp"]
                    g0 = per0[per0["period"] >= pd.Timestamp(BOUNDARY)].set_index("period")["theta_b1_Pexp"]
                    rr = (g1 / g0.reindex(g1.index) - 1).dropna() * 100
                    rec["last_quarter_b1_pct"] = float(rr.iloc[-1]) if len(rr) else np.nan
                # fused evidence per frozen model seed (STGNN rank / exceedance from the injected run)
                for ms, (bc, bi) in (bundles or {}).items():
                    ev1 = fuse(bi.rank(), bi.exceed_share(), spl1, per1)
                    ev0 = fuse(bc.rank(), bc.exceed_share(), spl0, per0)
                    rec_s = dict(rec, model_seed=ms, evidence_injected=ev1["evidence"].iloc[0],
                                 evidence_clean=ev0["evidence"].iloc[0],
                                 exceed_share_injected=float(bi.exceed_share().get(r.plant, np.nan)),
                                 exceed_share_clean=float(bc.exceed_share().get(r.plant, np.nan)))
                    j = list(D["pv_cols"]).index(col)
                    ts = bi.timestamps
                    lastq = ts >= ts[-1] - pd.Timedelta(days=90)
                    with np.errstate(all="ignore"):
                        rec_s["d_eta_last_quarter"] = float(np.nanmean(bi.eta[lastq, j] - bc.eta[lastq, j]))
                        rec_s["d_physics_z_last_quarter"] = float(np.nanmean(bi.physics_z[lastq, j] - bc.physics_z[lastq, j]))
                    deg_rows.append(rec_s)
                if not bundles:
                    deg_rows.append(rec)
        # fleet GHI shift: control plants, quarters that contain shifted hours
        if real_ghi is not None and real_ghi.fleet_mask.any():
            ctrl = [c for c, r in real_ghi.roles.items() if r == "control"]
            qs_all = sorted(set(real_ghi.fleet_mask[real_ghi.fleet_mask].index.to_period("Q")))
            for col in ctrl:
                per0, _, _ = cache.get_clean(name, D, C, col)
                per1, _, _ = _est_one(name, D, C, real_ghi.df, real_ghi.phys, col, cache.kalman_cfg,
                                      t_cell=real_ghi.t_cell)
                if not len(per0) or not len(per1):
                    continue
                p0 = per0.set_index("period")["theta_b1_Pexp"]; p1 = per1.set_index("period")["theta_b1_Pexp"]
                for q in qs_all:
                    qs, qe = q.start_time, q.end_time
                    rows.append(dict(estimator=name, event_id=-1, scenario="ghi_shift", strength=np.nan,
                                     duration="all in quarter", plant=ssg._short(col), quarter=str(q),
                                     expected_b1_pct=_ratio_change(D, C, real_ghi.df, real_ghi.phys, col, qs, qe),
                                     estimated_b1_pct=(p1.get(qs, np.nan) / p0.get(qs, np.nan) - 1) * 100,
                                     local_expected_pct=np.nan, local_estimated_pct=np.nan,
                                     region_vs_clean_disjoint=np.nan))
    return dict(b1=pd.DataFrame(rows), degradation=pd.DataFrame(deg_rows))


# ============================================================================ orchestration
@dataclass
class SemiSynthResult:
    events: pd.DataFrame
    classes: pd.DataFrame
    false_alarms: pd.DataFrame
    background: pd.DataFrame
    b1: pd.DataFrame
    degradation: pd.DataFrame
    labels: pd.DataFrame
    plans: Dict[int, "ssg.Plan"]
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    examples: Dict[str, dict] = field(default_factory=dict)
    integrity: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: dict = field(default_factory=dict)


def run_semisynthetic(D: dict, C: dict, models: Dict[int, FrozenSTGNN], placement_seeds: Sequence[int] = (0, 1, 2),
                      estimators: Sequence[str] = ("sps", "kalman"), A: Optional[dict] = None,
                      gen_cfg: Optional["ssg.GenConfig"] = None, eval_cfg: Optional[EvalConfig] = None,
                      kalman_cfg=None, cache_dir: Optional[str] = "results/semisynth",
                      keep_examples: bool = True) -> SemiSynthResult:
    """Full semi-synthetic validation: placements x frozen models x estimators."""
    gen_cfg = gen_cfg or ssg.GenConfig(boundary=BOUNDARY)
    eval_cfg = eval_cfg or EvalConfig()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    clean = {}
    for ms, mdl in models.items():
        _log(f"clean scoring, model seed {ms}")
        clean[ms] = mdl.score(D["df"], D["phys"], min_len=eval_cfg.min_len)
    cache = EstimatorCache(kalman_cfg)
    E_all, C_all, FA_all, BG_all, B_all, DG_all, L_all = [], [], [], [], [], [], []
    plans, examples = {}, {}
    for ps in placement_seeds:
        plan = ssg.plan_semisynthetic(D, C, seed=ps, cfg=gen_cfg)
        plans[ps] = plan
        real = ssg.apply_plan(D, C, plan)                                     # everything (for the STGNN)
        L_all.append(real.labels.assign(placement=ps))
        _log(f"placement {ps}: {int(real.labels['effective'].sum())} effective events, "
             f"targets={int((plan.roles == 'target').sum())}, controls={int((plan.roles == 'control').sum())}")
        bundles = {}
        for ms, mdl in models.items():
            bi = mdl.score(real.df, real.phys, min_len=eval_cfg.min_len)
            bundles[ms] = (clean[ms], bi)
            r = evaluate_detection(D, C, real, clean[ms], bi, eval_cfg, A)
            tag = dict(placement=ps, model_seed=ms)
            E_all.append(r["events"].assign(**tag)); FA_all.append(r["false_alarms"].assign(**tag))
            if len(r["classes"]):
                C_all.append(r["classes"].assign(**tag))
            if len(r["background"]):
                BG_all.append(r["background"].assign(**tag))
            if keep_examples and ps == placement_seeds[0] and ms == list(models)[0]:
                examples = _collect_examples(D, real, clean[ms], bi)
        # b1: plant-level + degradation without the fleet sensor shift; fleet shift separately
        _log(f"placement {ps}: b1 estimators {list(estimators)}")
        no_ghi = tuple(s for s in gen_cfg.scenarios if s != "ghi_shift")
        real_p = ssg.apply_plan(D, C, plan, include=no_ghi)
        real_g = ssg.apply_plan(D, C, plan, include=("ghi_shift",)) if "ghi_shift" in gen_cfg.scenarios else None
        rb = evaluate_b1(D, C, plan, real_p, real_g, estimators, cache, bundles)
        B_all.append(rb["b1"].assign(placement=ps)); DG_all.append(rb["degradation"].assign(placement=ps))
    integ = pd.DataFrame([dict(model_seed=ms, source=m.source, sha256_16=m.fingerprint(), unchanged=m.unchanged())
                          for ms, m in models.items()])
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    out = SemiSynthResult(events=cat(E_all), classes=cat(C_all), false_alarms=cat(FA_all), background=cat(BG_all),
                          b1=cat(B_all), degradation=cat(DG_all), labels=cat(L_all), plans=plans,
                          examples=examples, integrity=integ,
                          meta=dict(placement_seeds=list(placement_seeds), model_seeds=list(models),
                                    estimators=list(estimators), gen_cfg=gen_cfg.__dict__, eval_cfg=eval_cfg.__dict__))
    out.tables = build_tables(out, eval_cfg)
    if cache_dir:
        with open(os.path.join(cache_dir, "semisynth_result.pkl"), "wb") as f:
            pickle.dump(out, f)
    _log("weights unchanged:", bool(integ["unchanged"].all()))
    return out


def _collect_examples(D, real, clean: ScoreBundle, inj: ScoreBundle, pad_h: int = 48) -> Dict[str, dict]:
    """One example trace per scenario (largest strength, day / week duration)."""
    ex = {}
    lab = real.labels[real.labels["effective"]]
    for scn in lab["scenario"].unique():
        g = lab[lab["scenario"] == scn].sort_values(["strength", "n_eff_h"], ascending=False)
        g2 = g[g["duration"] == "day"]
        r = (g2 if len(g2) else g).iloc[0]
        a = r["eff_start"] - pd.Timedelta(hours=pad_h)
        b = (r["eff_end"] + pd.Timedelta(hours=pad_h)) if scn != "degradation" else r["eff_start"] + pd.Timedelta(days=10)
        if scn == "degradation":
            a = D["df"].index[-1] - pd.Timedelta(days=10); b = D["df"].index[-1]
        col = r["column"] if isinstance(r["column"], str) else [c for c, ro in real.roles.items() if ro == "control"][0]
        j = list(D["pv_cols"]).index(col)
        m = (D["df"].index >= a) & (D["df"].index <= b)
        k = (inj.timestamps >= a) & (inj.timestamps <= b)
        ex[scn] = dict(label=r.to_dict(), plant=ssg._short(col), t=D["df"].index[m],
                       p_clean=D["df"].loc[m, col].to_numpy(float), p_inj=real.df.loc[m, col].to_numpy(float),
                       pexp=real.phys["p_expected"][m, j], ts=inj.timestamps[k],
                       score_clean=clean.combined[k, j], score_inj=inj.combined[k, j], thr=inj.thr[k, j],
                       eta_clean=clean.eta[k, j], eta_inj=inj.eta[k, j])
        if scn == "degradation":                 # whole test: daily ratio P / P_exp, clean vs injected
            t = D["df"].index
            pe0 = D["phys"]["p_expected"][:, j]          # clean P_exp: the GHI shift must not leak in
            mt = (t >= r["eff_start"]) & (pe0 > 0.2 * np.nanmax(pe0))
            day = t[mt].normalize()
            pe = pd.Series(pe0[mt], index=day).groupby(level=0).sum()
            ex[scn]["daily"] = pd.DataFrame(dict(
                clean=pd.Series(D["df"].loc[mt, col].to_numpy(float), index=day).groupby(level=0).sum() / pe,
                inj=pd.Series(real.df.loc[mt, col].to_numpy(float), index=day).groupby(level=0).sum() / pe))
    return ex


# ============================================================================ tables
def _mdl(g: pd.DataFrame, target: float) -> Tuple[float, float]:
    """Smallest strength with recall >= target and a log-linear interpolation."""
    t = g.groupby("strength")["detected"].mean().sort_index()
    ok = t[t >= target]
    first = float(ok.index[0]) if len(ok) else np.nan
    interp = np.nan
    s, r = t.index.to_numpy(float), t.to_numpy(float)
    for i in range(len(s)):
        if r[i] >= target:
            if i == 0:
                interp = s[0]
            else:
                x0, x1 = np.log(s[i - 1]), np.log(s[i]); y0, y1 = r[i - 1], r[i]
                interp = float(np.exp(x0 + (target - y0) / max(y1 - y0, 1e-9) * (x1 - x0)))
            break
    return first, interp


COLD_BINS = (0, 30, 90, 400)        # days since commissioning


def _prf(g: pd.DataFrame) -> Tuple[float, float, float]:
    tp = g["n_tp_detections"].sum(); fp = g["n_fp_near"].sum()
    rec = g["detected"].mean() if len(g) else np.nan
    prec = tp / (tp + fp) if tp + fp > 0 else np.nan
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and prec + rec > 0 else np.nan
    return rec, prec, f1


def _cohort_tables(res: "SemiSynthResult", cfg: EvalConfig) -> Dict[str, pd.DataFrame]:
    """Detection and false alarms per installation cohort; cold start (days since commissioning);
    plant roles per cohort; degradation bias per cohort."""
    out = {}
    title = {k: s.title for k, s in ssg.SCENARIOS.items()}
    E = res.events
    P = E[(E["level"] == "plant") & E["cohort"].notna()].copy()
    P["cohort"] = P["cohort"].astype(int)
    FA = res.false_alarms
    rows = []
    for k, g in P.groupby("cohort"):
        g2 = g[g["duration"] != "short"]; gs = g[g["duration"] == "short"]
        rec, prec, f1 = _prf(g2)
        fa = FA[FA["cohort"] == k] if len(FA) and "cohort" in FA else pd.DataFrame()
        h = fa["hours"].sum() if len(fa) else 0
        rows.append({"Cohort": ssg.COHORT_TITLES.get(k, str(k)), "cohort": k,
                     "Target plants": g["plant"].nunique(), "N (day+week)": len(g2),
                     "Recall": rec, "Precision": prec, "F1": f1,
                     "Delay, h": g2.loc[g2["detected"], "delay_h"].median(),
                     "Short events cut": 1 - gs["detected"].mean() if len(gs) else np.nan,
                     "Days after commissioning (median)": g["days_since_commissioning"].median(),
                     "Control plants": int(fa["plant"].nunique()) if len(fa) else 0,
                     "Induced false alarms / 1000 h": fa["induced_alarms"].sum() / h * 1000 if h else np.nan,
                     "Alarms in clean data / 1000 h": fa["clean_alarms"].sum() / h * 1000 if h else np.nan})
    if rows:
        out["cohorts"] = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
        g = P[P["duration"] != "short"]
        out["cohort_scenario"] = (g.pivot_table(index="scenario", columns="cohort", values="detected", aggfunc="mean")
                                  .rename(index=title, columns=lambda k: ssg.COHORT_TITLES.get(int(k), str(k)))
                                  .reset_index().rename(columns={"scenario": "Scenario"}))
        # cold start: recall vs time since commissioning (cohort 3 is scored only in its first year)
        b = list(COLD_BINS)
        g = P[(P["duration"] != "short") & P["days_since_commissioning"].notna()].copy()
        if len(g):
            lab = [f"{b[i]}–{b[i + 1]} d" for i in range(len(b) - 1)]
            g["bin"] = pd.cut(g["days_since_commissioning"], b, labels=lab, right=False)
            rows = []
            for (k, bn), h in g.groupby(["cohort", "bin"], observed=True):
                rec, prec, f1 = _prf(h)
                rows.append({"Cohort": ssg.COHORT_TITLES.get(int(k), str(k)), "After commissioning": str(bn),
                             "N": len(h), "Recall": rec, "Precision": prec, "F1": f1,
                             "Delay, h": h.loc[h["detected"], "delay_h"].median()})
            out["cold_start"] = pd.DataFrame(rows)
    # roles per cohort (all placements)
    rr = []
    for pl, plan in res.plans.items():
        if getattr(plan, "plants", None) is not None:
            rr.append(plan.plants.assign(placement=pl))
    if rr:
        R = pd.concat(rr).reset_index(drop=True)
        R["role"] = R["role"].fillna("unused").where(R["eligible"], "not admitted")
        t = pd.crosstab(R["cohort"].map(ssg.COHORT_TITLES), R["role"]) / max(len(res.plans), 1)
        t["plants in total"] = R.groupby(R["cohort"].map(ssg.COHORT_TITLES))["plant"].nunique()
        out["roles"] = t.round(1).reset_index().rename(columns={"cohort": "Cohort"})
    DG = res.degradation
    if len(DG) and "cohort" in DG:
        g = DG.dropna(subset=["cohort"]).groupby(["estimator", "cohort"])
        out["degradation_cohort"] = pd.DataFrame({
            "Estimator": [EST_LABELS.get(e, e) for e, _ in g.groups.keys()],
            "Cohort": [ssg.COHORT_TITLES.get(int(k), str(k)) for _, k in g.groups.keys()],
            "N": g.size().values,
            "Exp. Δb1 (test), %": g["expected_b1_pct"].mean().values,
            "Est. Δb1 (test), %": g["estimated_b1_pct"].mean().values,
            "Bias, p.p.": (g["estimated_b1_pct"].mean() - g["expected_b1_pct"].mean()).values,
            "Evidence «β1 decrease»": g["evidence_injected"].apply(
                lambda s_: s_.astype(str).str.contains("decrease").mean()).values
            if "evidence_injected" in DG else np.nan})
    return out


def build_tables(res: SemiSynthResult, cfg: Optional[EvalConfig] = None) -> Dict[str, pd.DataFrame]:
    cfg = cfg or EvalConfig()
    E = res.events
    T = {}
    title = {k: s.title for k, s in ssg.SCENARIOS.items()}
    dur = ssg.DURATIONS
    if len(E):
        cls = res.classes
        det_cls = cls[cls["kind"] == "detected"] if len(cls) else pd.DataFrame()
        orc_cls = cls[cls["kind"] == "oracle"] if len(cls) else pd.DataFrame()
        rows = []
        for (scn, st, du), g in E.groupby(["scenario", "strength", "duration"]):
            tp = g["n_tp_detections"].sum(); fp = g["n_fp_near"].sum()
            rec = g["detected"].mean()
            prec = tp / (tp + fp) if tp + fp > 0 else np.nan
            f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and prec + rec > 0 else (0.0 if rec == 0 else np.nan)
            key = ["placement", "model_seed", "event_id"]
            ids = g[key]
            def _share(dc):
                if not len(dc):
                    return np.nan
                m = dc.merge(ids, on=["placement", "model_seed", "event_id"])
                if not len(m):
                    return np.nan
                exp = ssg.SCENARIOS[scn].expected_name
                # per event: majority over detected plants (fleet) / the one detection (plant)
                v = m.assign(ok=m["fuzzy"] == exp).groupby(key)["ok"].mean() >= 0.5
                return float(v.mean())
            def _share_c(dc, colname):
                if not len(dc) or colname not in dc or dc[colname].isna().all():
                    return np.nan
                m = dc.merge(ids, on=key)
                if not len(m):
                    return np.nan
                exp = ssg.SCENARIOS[scn].expected_name
                return float((m.assign(ok=m[colname] == exp).groupby(key)["ok"].mean() >= 0.5).mean())
            rows.append({"scenario": scn, "Scenario": title[scn], "s, %": round(st * 100), "Duration": dur[du],
                         "duration": du, "strength": st, "N": len(g), "Recall": rec, "Precision": prec, "F1": f1,
                         "Raw hit": g["raw_hit"].mean(),
                         "Delay, h (median)": g.loc[g["detected"], "delay_h"].median(),
                         "Δη̂": g["d_eta"].mean(), "Δz_phys": g["d_physics_z"].mean(),
                         "True loss, %": g["true_loss_pct"].mean() if scn != "ghi_shift" else g["pexp_bias_pct"].mean(),
                         "Fuzzy: correct type": _share(det_cls),
                         "Fuzzy (true interval)": _share(orc_cls),
                         "Crisp rules": _share_c(det_cls, "crisp"),
                         "K-Means cluster": _share_c(det_cls, "kmeans")})
        D_ = pd.DataFrame(rows)
        order = {k: i for i, k in enumerate(ssg.SCENARIOS)}
        dorder = {"short": 0, "day": 1, "week": 2}
        D_ = D_.sort_values(["scenario", "duration", "strength"],
                            key=lambda s: s.map(order) if s.name == "scenario" else (s.map(dorder) if s.name == "duration" else s))
        T["detection"] = D_.reset_index(drop=True)
        # summary per scenario (all strengths / durations except the short noise control)
        rows = []
        for scn, g in E.groupby("scenario"):
            g2 = g[g["duration"] != "short"]
            tp = g2["n_tp_detections"].sum(); fp = g2["n_fp_near"].sum()
            rec = g2["detected"].mean() if len(g2) else np.nan
            prec = tp / (tp + fp) if tp + fp > 0 else np.nan
            f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and prec + rec > 0 else np.nan
            gs = g[g["duration"] == "short"]
            rows.append({"Scenario": title[scn], "scenario": scn, "N (day+week)": len(g2), "Recall": rec,
                         "Precision": prec, "F1": f1, "Delay, h": g2.loc[g2["detected"], "delay_h"].median(),
                         "Short events cut (1–2 steps)": 1 - gs["detected"].mean() if len(gs) else np.nan})
        T["by_scenario"] = pd.DataFrame(rows).sort_values("scenario", key=lambda s: s.map(order)).reset_index(drop=True)
        # minimum detectable loss
        rows = []
        for (scn, du), g in E[E["scenario"] != "outage"].groupby(["scenario", "duration"]):
            first, interp = _mdl(g, cfg.recall_target)
            f8, _ = _mdl(g, 0.8)
            rows.append({"Scenario": title[scn], "scenario": scn, "Duration": dur[du], "duration": du,
                         f"MDL (recall ≥ {cfg.recall_target:g}), %": first * 100 if np.isfinite(first) else np.nan,
                         "MDL interp., %": interp * 100 if np.isfinite(interp) else np.nan,
                         "MDL (recall ≥ 0.8), %": f8 * 100 if np.isfinite(f8) else np.nan})
        T["mdl"] = pd.DataFrame(rows).sort_values(["scenario", "duration"], key=lambda s: s.map(order) if s.name == "scenario" else s.map(dorder)).reset_index(drop=True)
        # weather
        g = E[E["level"] == "plant"]
        T["weather"] = (g.pivot_table(index="scenario", columns="weather", values="detected", aggfunc="mean")
                        .rename(index=title).reset_index().rename(columns={"scenario": "Scenario"}))
        T["tech_cluster"] = (g.pivot_table(index="scenario", columns="tech_cluster", values="detected", aggfunc="mean")
                             .rename(index=title).reset_index().rename(columns={"scenario": "Scenario"}))
        # fuzzy confusion (detected events, plant level)
        if len(det_cls):
            m = det_cls.merge(E[["placement", "model_seed", "event_id", "scenario"]], on=["placement", "model_seed", "event_id"])
            T["fuzzy_confusion"] = pd.crosstab(m["scenario"].map(title), m["fuzzy"], normalize="index")
    # cohorts (plants with different exploitation periods: 2021-2023 / from 2022 / only 2023)
    if len(E) and "cohort" in E:
        T.update(_cohort_tables(res, cfg))
    # false alarms
    FA = res.false_alarms
    if len(FA):
        g = FA.groupby("model_seed")[["hours", "clean_alarms", "injected_alarms", "induced_alarms"]].sum()
        T["false_alarms"] = pd.DataFrame({
            "Model seed": g.index, "Control hours": g["hours"].values,
            "Alarms in clean data / 1000 h": (g["clean_alarms"] / g["hours"] * 1000).values,
            "Alarms after injection / 1000 h": (g["injected_alarms"] / g["hours"] * 1000).values,
            "Induced false alarms / 1000 h": (g["induced_alarms"] / g["hours"] * 1000).values})
    # b1
    B = res.b1
    if len(B):
        rows = []
        for (est, scn, du), g in B.groupby(["estimator", "scenario", "duration"]):
            for st, h in (g.groupby("strength") if g["strength"].notna().any() else [(np.nan, g)]):
                ok = h["expected_b1_pct"].abs() > 0.05
                sens = (h.loc[ok, "estimated_b1_pct"] / h.loc[ok, "expected_b1_pct"]).median() if ok.any() else np.nan
                sign = (np.sign(h.loc[ok, "estimated_b1_pct"]) == np.sign(h.loc[ok, "expected_b1_pct"])).mean() if ok.any() else np.nan
                rows.append({"Estimator": EST_LABELS.get(est, est), "estimator": est, "Scenario": title.get(scn, scn),
                             "scenario": scn, "s, %": round(st * 100) if np.isfinite(st) else "mix",
                             "Duration": dur.get(du, du), "N": len(h),
                             "Exp. Δb1 q., %": h["expected_b1_pct"].mean(), "Est. Δb1 q., %": h["estimated_b1_pct"].mean(),
                             "Sensitivity": sens, "Sign correct": sign,
                             "Exp. Δb1 loc., %": h["local_expected_pct"].mean(),
                             "Est. Δb1 loc., %": h["local_estimated_pct"].mean()})
        T["b1"] = pd.DataFrame(rows)
    DG = res.degradation
    if len(DG):
        g = DG.groupby(["estimator", "strength"])
        stat_down = lambda s: s.astype(str).str.contains("decrease").mean()
        T["degradation"] = pd.DataFrame({
            "Estimator": [EST_LABELS.get(e, e) for e, _ in g.groups.keys()],
            "Loss at the end, %": [round(s * 100) for _, s in g.groups.keys()],
            "N": g.size().values,
            "Exp. Δb1 (test), %": g["expected_b1_pct"].mean().values,
            "Est. Δb1 (test), %": g["estimated_b1_pct"].mean().values,
            "Bias, p.p.": (g["estimated_b1_pct"].mean() - g["expected_b1_pct"].mean()).values,
            "Δb1 last quarter, %": g["last_quarter_b1_pct"].mean().values if "last_quarter_b1_pct" in DG else np.nan,
            "Train/test regions do not overlap": g["disjoint_injected"].apply(lambda s: s.astype(float).mean()).values,
            "Evidence «β1 decrease»": g["evidence_injected"].apply(stat_down).values if "evidence_injected" in DG else np.nan,
            "Same without injection": g["evidence_clean"].apply(stat_down).values if "evidence_clean" in DG else np.nan,
            "Δ STGNN exceedance share": (g["exceed_share_injected"].mean() - g["exceed_share_clean"].mean()).values if "exceed_share_injected" in DG else np.nan,
            "Δη̂ (last quarter)": g["d_eta_last_quarter"].mean().values if "d_eta_last_quarter" in DG else np.nan,
        })
    # estimator summary (the full model = frozen STGNN + estimator)
    if len(B) or len(DG):
        rows = []
        det = T.get("by_scenario")
        f1_all = float(det["F1"].mean()) if det is not None else np.nan
        for est in res.meta.get("estimators", []):
            b = B[(B["estimator"] == est) & (B["scenario"] != "ghi_shift")] if len(B) else pd.DataFrame()
            ok = b["expected_b1_pct"].abs() > 0.5 if len(b) else pd.Series(dtype=bool)
            d = DG[DG["estimator"] == est] if len(DG) else pd.DataFrame()
            rows.append({"Estimator": EST_LABELS.get(est, est),
                         "F1 STGNN (mean over scenarios)": f1_all,
                         "Sensitivity Δb1 (quarters)": float((b.loc[ok, "estimated_b1_pct"] / b.loc[ok, "expected_b1_pct"]).median()) if ok.any() else np.nan,
                         "MAE Δb1, p.p.": float((b["estimated_b1_pct"] - b["expected_b1_pct"]).abs().mean()) if len(b) else np.nan,
                         "MAE Δb1 of degradation, p.p.": float((d["estimated_b1_pct"] - d["expected_b1_pct"]).abs().mean()) if len(d) else np.nan,
                         "Degradation detected (evid.)": float(d["evidence_injected"].astype(str).str.contains("decrease").mean()) if "evidence_injected" in d else np.nan})
        T["estimators"] = pd.DataFrame(rows)
    T["integrity"] = res.integrity
    return T


# ============================================================================ notebook display / plots
def show(res: SemiSynthResult, fmt: str = "{:.3f}"):
    from IPython.display import display, Markdown
    names = {"by_scenario": "Summary by scenario (day + week)", "detection": "Detection: scenario × strength × duration",
             "mdl": "Minimum detectable loss (MDL)", "false_alarms": "False alarms on control plants",
             "b1": "Change of b1: SPS and Kalman filter", "degradation": "Gradual degradation: SPS and Kalman filter",
             "estimators": "Summary by estimator (frozen STGNN + SPS / Kalman)", "weather": "Recall by weather conditions",
             "tech_cluster": "Recall by technical cluster", "fuzzy_confusion": "Fuzzy inference: matrix (shares)",
             "roles": "Plant roles by cohort", "cohorts": "Detection and false alarms by cohort",
             "cohort_scenario": "Recall by scenario and cohort", "cold_start": "Cold start (days after commissioning)",
             "degradation_cohort": "Degradation by cohort",
             "integrity": "Frozen-weight check"}
    for k, nm in names.items():
        t = res.tables.get(k)
        if t is None or not len(t):
            continue
        display(Markdown(f"**{nm}**"))
        t = t.drop(columns=[c for c in ("scenario", "duration", "strength", "estimator", "cohort") if c in t.columns])
        display(t.style.format(precision=3, na_rep="—"))


def plot_all(res: SemiSynthResult, fig_dir: Optional[str] = "results/figures") -> Dict[str, object]:
    import matplotlib.pyplot as plt
    figs = {}
    T = res.tables
    title = {k: s.title for k, s in ssg.SCENARIOS.items()}
    colors = {"short": "#999999", "day": "#1f77b4", "week": "#d62728"}
    det = T.get("detection")
    if det is not None and len(det):
        scns = [s for s in ssg.SCENARIOS if s in set(det["scenario"])]
        n = len(scns); nc = 4; nr = int(np.ceil(n / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.0 * nr), squeeze=False)
        for ax, scn in zip(axes.ravel(), scns):
            g = det[det["scenario"] == scn]
            for du, h in g.groupby("duration"):
                ax.plot(h["strength"] * 100, h["Recall"], "o-", color=colors.get(du), label=ssg.DURATIONS[du])
            ax.set_ylim(-0.03, 1.03); ax.set_xscale("log"); ax.set_xticks([5, 15, 40, 100])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_title(title[scn], fontsize=9); ax.grid(alpha=.3); ax.set_xlabel("strength, %", fontsize=8)
            ax.axhline(0.5, color="k", lw=.6, ls=":")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        axes[0, 0].set_ylabel("recall"); axes[0, 0].legend(fontsize=7)
        fig.suptitle("Semi-synthetic: recall of the frozen STGNN by strength and duration"); fig.tight_layout()
        figs["recall"] = fig
    E = res.events
    if len(E):
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        g = E[E["detected"] & (E["duration"] != "short")]
        if len(g):
            data, labs = [], []
            for scn in ssg.SCENARIOS:
                for du in ("day", "week"):
                    v = g[(g["scenario"] == scn) & (g["duration"] == du)]["delay_h"].dropna()
                    if len(v):
                        data.append(v.to_numpy()); labs.append(f"{scn}\n{ssg.DURATIONS[du]}")
            if data:
                ax[0].boxplot(data, showfliers=False)
                ax[0].set_xticks(range(1, len(labs) + 1)); ax[0].set_xticklabels(labs, fontsize=6, rotation=90)
        ax[0].set_ylabel("delay, h"); ax[0].set_title("Detection delay")
        p = E[E["level"] == "plant"]
        for scn, h in p.groupby("scenario"):
            ax[1].scatter(-h["true_loss_pct"] / 100, h["d_eta"], s=12, label=scn, alpha=.7)
        lim = [-1.05, 0.5]
        ax[1].plot(lim, lim, "k:", lw=.8); ax[1].set_xlim(lim)
        ax[1].set_xlabel("true relative power change"); ax[1].set_ylabel("Δη̂ (injected − clean)")
        ax[1].set_title("Response of η̂"); ax[1].legend(fontsize=6); ax[1].grid(alpha=.3)
        for scn, h in p.groupby("scenario"):
            ax[2].scatter(h["true_loss_pct"], h["d_physics_z"], s=12, label=scn, alpha=.7)
        ax[2].set_xlabel("true loss, %"); ax[2].set_ylabel("Δz of the physics residual")
        ax[2].set_title("Response of the physics residual"); ax[2].grid(alpha=.3)
        fig.tight_layout(); figs["response"] = fig
    fc = T.get("fuzzy_confusion")
    if fc is not None and len(fc):
        try:
            from pv_clustering import ANOMALY_CLASSES as _AC
            fc = fc.reindex(columns=list(dict.fromkeys(list(_AC) + list(fc.columns))), fill_value=0.0)
        except Exception:
            pass
        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(fc.to_numpy(float), cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(len(fc.index))); ax.set_yticklabels(fc.index, fontsize=8)
        ax.set_xticks(range(len(fc.columns))); ax.set_xticklabels([f"C{i}: " + str(c).split(" (")[0] for i, c in enumerate(fc.columns)], rotation=30, ha="right", fontsize=8)
        for i in range(fc.shape[0]):
            for j in range(fc.shape[1]):
                v = fc.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > .6 else "black")
        fig.colorbar(im, ax=ax, label="share of events"); ax.set_title("Fuzzy inference: injected type → class with the highest membership")
        fig.tight_layout(); figs["fuzzy"] = fig
    B = res.b1
    if len(B):
        ests = list(dict.fromkeys(B["estimator"]))
        fig, axes = plt.subplots(1, len(ests) + 1, figsize=(5.2 * (len(ests) + 1), 4.2), squeeze=False)
        for ax, est in zip(axes[0], ests):
            g = B[B["estimator"] == est]
            for scn, h in g.groupby("scenario"):
                ax.scatter(h["expected_b1_pct"], h["estimated_b1_pct"], s=12, alpha=.7, label=scn)
            v = np.nanmax(np.abs(np.r_[g["expected_b1_pct"], g["estimated_b1_pct"], 1.0]))
            ax.plot([-v, v], [-v, v], "k:", lw=.8); ax.grid(alpha=.3)
            ax.set_xlabel("expected quarterly Δb1, %"); ax.set_ylabel("estimated Δb1, %")
            ax.set_title(f"{EST_LABELS.get(est, est)}: quarterly b1"); ax.legend(fontsize=6)
        DG = res.degradation
        ax = axes[0, -1]
        if len(DG):
            w = 0.38
            for i, est in enumerate(ests):
                g = DG[DG["estimator"] == est].groupby("strength")[["expected_b1_pct", "estimated_b1_pct"]].mean()
                x = np.arange(len(g))
                ax.bar(x + (i - 0.5) * w, g["estimated_b1_pct"], w, label=f"{EST_LABELS.get(est, est)}: estimate")
                if i == 0:
                    ax.scatter(x, g["expected_b1_pct"], color="k", zorder=3, label="expected")
                ax.set_xticks(x); ax.set_xticklabels([f"{s * 100:.0f} %" for s in g.index])
            ax.set_title("Degradation: test Δb1"); ax.set_xlabel("loss at the end of the test"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); figs["b1"] = fig
    if res.examples:
        ks = list(res.examples)
        fig, axes = plt.subplots(len(ks), 1, figsize=(13, 2.6 * len(ks)), squeeze=False)
        for ax, k in zip(axes[:, 0], ks):
            e = res.examples[k]
            L = e["label"]
            if "daily" in e:
                d = e["daily"]
                ax.plot(d.index, d["clean"], color="#888888", lw=.8, label="clean: daily P / P_exp")
                ax.plot(d.index, d["inj"], color="#c0392b", lw=.8, label="with degradation: daily P / P_exp")
                ax.plot(d.index, d["clean"].mean() * (1 - L["strength"] * np.linspace(0, 1, len(d))), "k:", lw=.8,
                        label="injected trend")
                ax.set_title(f"{ssg.SCENARIOS[k].title}: {e['plant']}, s = {L['strength'] * 100:.0f} % at the end of the test",
                             fontsize=9)
                ax.legend(fontsize=6, loc="lower left"); ax.grid(alpha=.3)
                continue
            ax.plot(e["t"], e["p_clean"] / 1000, color="#888888", lw=1, label="clean P, kW")
            ax.plot(e["t"], e["p_inj"] / 1000, color="#c0392b", lw=1, label="with injection")
            ax.plot(e["t"], e["pexp"] / 1000, color="#1f77b4", lw=.7, ls="--", label="P_exp")
            a2 = ax.twinx()
            a2.plot(e["ts"], e["score_inj"], color="#6a1b9a", lw=.8, label="score")
            a2.plot(e["ts"], e["thr"], color="#6a1b9a", lw=.6, ls=":")
            a2.set_ylabel("score (dots — threshold)", fontsize=7)
            ax.axvspan(L["eff_start"], L["eff_end"], color="orange", alpha=.15)
            ax.set_title(f"{ssg.SCENARIOS[k].title}: {e['plant']}, s = {L['strength'] * 100:.0f} %, {ssg.DURATIONS[L['duration']]}", fontsize=9)
            ax.legend(fontsize=6, loc="upper left")
        fig.tight_layout(); figs["examples"] = fig
    # cohorts: recall per scenario and cohort, cold start, false alarms
    cs, co, cst = T.get("cohort_scenario"), T.get("cohorts"), T.get("cold_start")
    if cs is not None and len(cs):
        ccol = {t: c for t, c in zip(ssg.COHORT_TITLES.values(), ["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"])}
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1.2, 1]))
        ax = axes[0]
        kc = [c for c in cs.columns if c != "Scenario"]
        x = np.arange(len(cs)); w = 0.8 / max(len(kc), 1)
        for i, c in enumerate(kc):
            v = cs[c].to_numpy(float); xx = x + (i - (len(kc) - 1) / 2) * w
            ax.bar(xx, np.nan_to_num(v), w, label=c, color=ccol.get(c, "#7f7f7f"))
            for xi, vi in zip(xx, v):           # make zero / missing bars visible
                if not np.isfinite(vi) or vi == 0:
                    ax.text(xi, 0.01, "—" if not np.isfinite(vi) else "0", ha="center", va="bottom", fontsize=7,
                            color=ccol.get(c, "#7f7f7f"))
        ax.set_xticks(x); ax.set_xticklabels([t.split(" (")[0] for t in cs["Scenario"]], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("recall (day + week)"); ax.legend(fontsize=7)
        ax.set_title("Recall by cohort (0 — not detected, — no events)", fontsize=10); ax.grid(axis="y", alpha=.3)
        ax = axes[1]
        if cst is not None and len(cst):
            b = list(COLD_BINS); order = [f"{b[i]}–{b[i + 1]} d" for i in range(len(b) - 1)]
            for k, g in cst.groupby("Cohort", sort=False):
                g = g.set_index("After commissioning").reindex(order)
                ax.plot(np.arange(len(order)), g["Recall"].to_numpy(float), "o-", label=k, color=ccol.get(k, "#7f7f7f"))
            ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(order)
            ax.set_ylim(-0.03, 1.05); ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax.set_title("Cold start: recall and time after commissioning", fontsize=10); ax.set_xlabel("after commissioning")
        ax = axes[2]
        if co is not None and len(co):
            ax.bar([t.split(" (")[0] for t in co["Cohort"]], np.nan_to_num(co["Induced false alarms / 1000 h"].to_numpy(float)),
                   color=[ccol.get(t, "#7f7f7f") for t in co["Cohort"]])
        ax.set_title("Induced false alarms / 1000 h", fontsize=10); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); figs["cohorts"] = fig
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        for k, f in figs.items():
            f.savefig(os.path.join(fig_dir, f"fig_semisynth_{k}.png"), dpi=200, bbox_inches="tight")
    return figs


# ============================================================================ LaTeX
def _tex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                 ("η̂", "$\\hat\\eta$"), ("Δ", "$\\Delta$"), ("≥", "$\\geq$"), ("→", "$\\to$"), ("–", "--"),
                 ("«", "<<"), ("»", ">>"), ("×", "$\\times$"), ("↔", "--"),
                 ("β", "$\\beta$"), ("η", "$\\eta$"), ("σ", "$\\sigma$"), ("≤", "$\\leq$"),
                 ("±", "$\\pm$"), ("—", "---"), ("≈", "$\\approx$"), ("·", "$\\cdot$")):
        s = s.replace(a, b)
    return s


def _tex_head(c, width: int = 12) -> str:
    """Multi-line header (\\shortstack, no extra packages) for long column names."""
    words, lines, cur = str(c).split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    lines = [_tex_escape(x) for x in lines]
    return lines[0] if len(lines) == 1 else "\\shortstack{" + " \\\\ ".join(lines) + "}"


def _tex_caption(text) -> str:
    """Escapes _ % & # in a caption outside $...$ (already escaped characters are kept)."""
    import re as _re
    parts = str(text).split("$")
    for k in range(0, len(parts), 2):                       # even parts are outside math
        parts[k] = _re.sub(r"(?<!\\)([_%&#])", r"\\\1", parts[k])
    return "$".join(parts)


def df_to_latex(df: pd.DataFrame, caption: str, label: str, digits: int = 2, long_rows: int = 40) -> str:
    """booktabs table; tables longer than ``long_rows`` become ``longtable`` (needs \\usepackage{longtable})."""
    df = df.drop(columns=[c for c in ("scenario", "duration", "strength", "estimator", "cohort") if c in df.columns])
    cols = list(df.columns)
    def _al(c):
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            return "r"
        if len(df) > long_rows and df[c].astype(str).str.len().max() > 22:   # long text in long tables -> wrap
            return ">{\\raggedright\\arraybackslash}p{2.5cm}"
        return "l"
    aligns = [_al(c) for c in cols]
    align = "".join(aligns)
    head = " & ".join(_tex_head(c, 9 if len(df) > long_rows else 12) for c in cols) + " \\\\"
    body, prev = [], None
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            elif isinstance(v, (float, np.floating)):
                cells.append("--" if not np.isfinite(v) else
                             f"{(0.0 if abs(v) < 0.5 * 10 ** -digits else v):.{digits}f}")
            else:
                cells.append(_tex_escape(v))
        if prev is not None and cols and r[cols[0]] != prev and aligns[0] != "r":
            body.append("\\midrule")
        prev = r[cols[0]]
        body.append(" & ".join(cells) + " \\\\")
    caption = _tex_caption(caption)
    if len(df) > long_rows:
        lines = ["{\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}", f"\\begin{{longtable}}{{{align}}}",
                 f"\\caption{{{caption}}}\\label{{{label}}}\\\\", "\\toprule", head, "\\midrule", "\\endfirsthead",
                 "\\toprule", head, "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]
        lines += body + ["\\end{longtable}}", ""]
    else:
        wide = len(cols) > 7                      # only wide tables are scaled down to the text width
        lines = ["\\begin{table}[ht]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                 ("\\resizebox{\\textwidth}{!}{%" if wide else "{%"), f"\\begin{{tabular}}{{{align}}}",
                 "\\toprule", head, "\\midrule"]
        lines += body + ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


TEX_CAPTIONS = {
    "by_scenario": "Semi-synthetic test: summary by scenario (duration day and week; frozen STGNN weights)",
    "detection": "Semi-synthetic test: precision, recall, $F_1$, delay, response of $\\hat\\eta$ and of the physics residual, share of correct fuzzy classification",
    "mdl": "Minimum detectable loss (MDL)",
    "false_alarms": "False alarms on control plants (per 1000 h)",
    "b1": "Change of $b_1$ (SPS and Kalman filter): expected and estimated",
    "degradation": "Gradual degradation: $b_1$ estimates by SPS and the Kalman filter and final evidence",
    "estimators": "Final comparison of the model with SPS and with the Kalman filter on semi-synthetic data",
    "weather": "Recall by weather conditions",
    "tech_cluster": "Recall by technical cluster of plants",
    "cohorts": "Detection and false alarms by plant cohort (different operating periods)",
    "cohort_scenario": "Recall by scenario and cohort (day and week)",
    "cold_start": "Cold start: detection as a function of time after plant commissioning",
    "roles": "Plant roles in the semi-synthetic test by cohort (mean per placement)",
    "degradation_cohort": "Gradual degradation by cohort: bias of the $\\Delta b_1$ estimate",
    "fuzzy_confusion": ("Fuzzy inference: shares of the highest membership by injected scenario "
                        "(C0 --- outage, C1 --- clipping, C2 --- partial shading, C3 --- gradual loss, "
                        "C4 --- overgeneration, C5 --- low irradiance)"),
    "integrity": "Frozen STGNN weight check (SHA-256 before and after the test)",
}


def _fuzzy_for_tex(t: pd.DataFrame) -> pd.DataFrame:
    """Confusion matrix with short class headers K0..K5 (full names are in the caption legend)."""
    try:
        from pv_clustering import ANOMALY_CLASSES
        names = list(ANOMALY_CLASSES.values()) if isinstance(ANOMALY_CLASSES, dict) else list(ANOMALY_CLASSES)
    except Exception:
        names = []
    if names:
        t = t.reindex(columns=list(dict.fromkeys(names + [c for c in t.columns if c not in names])), fill_value=0.0)
    ren = {c: (f"C{names.index(c)}" if c in names else str(c)) for c in t.columns}
    out = t.rename(columns=ren)
    out = out[sorted(out.columns, key=str)]
    out.insert(0, "Injected scenario", out.index.astype(str))
    return out.reset_index(drop=True)


def export_latex(res: SemiSynthResult, out_dir: str = "results/semisynth") -> Dict[str, str]:
    """One .tex per table + ``semisynth_tables.tex`` with all of them (booktabs)."""
    os.makedirs(out_dir, exist_ok=True)
    paths, allp = {}, ["% requires \\usepackage{booktabs,graphicx,longtable,array}; Cyrillic: \\usepackage[T2A]{fontenc}, "
                       "\\usepackage[utf8]{inputenc}, \\usepackage[russian]{babel}", ""]
    for k, cap in TEX_CAPTIONS.items():
        t = res.tables.get(k)
        if t is None or not len(t):
            continue
        if k == "fuzzy_confusion":
            t = _fuzzy_for_tex(t)
        if k == "detection":
            key = [c for c in ("Scenario", "s, %", "Duration", "N") if c in t.columns]
            resp = [c for c in t.columns if c not in key and any(x in str(c) for x in
                    ("Δ", "True", "Fuzzy", "Crisp", "K-Means"))]
            det = [c for c in t.columns if c not in resp]
            tex = (df_to_latex(t[det], cap.split(", response")[0] + " (frozen STGNN weights)",
                               "tab:semisynth_detection") + "\n" +
                   df_to_latex(t[key + resp], "Response of $\\hat\\eta$ and of the physics residual and classification "
                               "of detected events (share of the correct type: fuzzy inference, crisp rules, K-Means)",
                               "tab:semisynth_response"))
        else:
            tex = df_to_latex(t, cap, f"tab:semisynth_{k}")
        p = os.path.join(out_dir, f"semisynth_{k}.tex")
        with open(p, "w", encoding="utf-8") as f:
            f.write(tex)
        paths[k] = p; allp.append(tex)
    p = os.path.join(out_dir, "semisynth_tables.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(allp))
    paths["all"] = p
    for k, t in res.tables.items():
        if t is not None and len(t):
            t.to_csv(os.path.join(out_dir, f"semisynth_{k}.csv"), index=k in ("fuzzy_confusion",))
    return paths


def download_link(path: str, text: Optional[str] = None):
    from IPython.display import HTML
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    name = os.path.basename(path)
    mime = "application/zip" if name.endswith(".zip") else "text/x-tex"
    return HTML(f'<a download="{name}" href="data:{mime};base64,{b64}" target="_blank" '
                f'style="font-weight:bold">{text or "Download " + name}</a>')


def zip_outputs(out_dir: str = "results/semisynth", fig_dir: str = "results/figures",
                name: str = "semisynth_results.zip") -> str:
    import zipfile
    p = os.path.join(out_dir, name)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out_dir)):
            if f.endswith((".tex", ".csv")):
                z.write(os.path.join(out_dir, f), f)
        if fig_dir and os.path.isdir(fig_dir):
            for f in sorted(os.listdir(fig_dir)):
                if f.startswith("fig_semisynth_"):
                    z.write(os.path.join(fig_dir, f), os.path.join("figures", f))
    return p


# ============================================================================ CLI
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Semi-synthetic validation with frozen STGNN weights")
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--out", default="results/semisynth")
    ap.add_argument("--model_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--placements", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--estimators", nargs="+", default=["sps", "kalman"])
    ap.add_argument("--no_train", action="store_true", help="fail if no frozen checkpoint exists")
    a = ap.parse_args()
    D = pickle.load(open(os.path.join(a.cache, "data.pkl"), "rb"))
    C = pickle.load(open(os.path.join(a.cache, "clusters.pkl"), "rb"))
    fA = os.path.join(a.cache, "anomaly_sps.pkl")
    A = pickle.load(open(fA, "rb")) if os.path.exists(fA) else None
    models = frozen_models(D, C, a.model_seeds, cache_dir=a.cache, train_missing=not a.no_train)
    res = run_semisynthetic(D, C, models, a.placements, a.estimators, A=A, cache_dir=a.out)
    plot_all(res, os.path.join(os.path.dirname(a.out) or ".", "figures"))
    print(export_latex(res, a.out))


if __name__ == "__main__":
    main()
