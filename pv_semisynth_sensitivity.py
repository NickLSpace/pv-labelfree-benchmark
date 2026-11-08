"""
Sensitivity of the semi-synthetic detection to the alarm rule and to the source of eta_hat
(independent module; FROZEN STGNN weights, nothing in the existing pipeline imports it).

Questions
---------
1. Threshold quantile q (0.95 ... 0.999) and the minimum run length ``min_len`` of a detection:
   how do recall, precision, F1, delay and the false-alarm rate trade off?
2. eta_hat from the CURRENT input window (as the model does) "explains" an injected anomaly:
   the window already contains the anomalous hours, eta_hat drops and the physical residual
   |eta_hat * P_exp - P| stays small.  Alternatives evaluated with the same frozen network:

   ``current``  eta_hat of the current window (the model, baseline)
   ``lag24``    eta_hat predicted 24 h earlier (same hour of the previous day)
   ``lag168``   eta_hat predicted 168 h earlier (one week)
   ``med7d``    median of the daylight eta_hat over the 7 days that end 24 h before t
   ``train``    one value per plant: median daylight eta_hat of the train-period calibration
                windows (cold-start plants: median of the technical cluster, else of the fleet)

What is recomputed / what is not
--------------------------------
One forward pass of the frozen network per data set (train calibration windows, clean test,
every injected placement); the raw reconstruction / forecast errors, eta_hat, P_exp / P_rated
and P / P_rated are stored.  Everything else is post-processing:

* physical residual  phy = mean_h |eta_v * P_exp - P| / P_rated  for the eta_hat variant v;
* its z-normalisation (mu, sd per node) from the train calibration windows with the same
  variant (cold-start nodes: technical-cluster median, ``borrow_error_stats``); reconstruction
  and forecast z-statistics are those of the checkpoint;
* combined score  z_rec + 0.5 z_fore + 0.5 z_phy  and the thresholds for quantile q from the
  train calibration windows exactly as ``UnsupervisedSTGNN.calibrate`` (node, weather-regime,
  technical-cluster prior, POT fallback);
* detections = runs of exceedances of length >= min_len; detection metrics with
  ``pv_semisynth_eval.evaluate_detection`` (paired with the clean run of the same variant).

``current`` with the checkpoint quantile reproduces ``pv_semisynth_eval`` exactly (checked in the
``reproduction`` table).  Weights are never modified (SHA-256 check before / after).

Caveat: choosing q / min_len / the eta_hat variant from these semi-synthetic results is a
selection on the evaluation data.  For the paper, fix the operating point by a false-alarm
budget on the clean data (``fa_matched`` table) and report the curves as sensitivity analysis.

Usage (notebook section 11)::

    import pv_semisynth_sensitivity as sss
    sens = sss.run_sensitivity(D, C, ss_models, ss_res=ss_res)        # same placements as section 10
    sss.show(sens); sss.plot_all(sens); sss.export_latex(sens)
"""

from __future__ import annotations

import contextlib
import io
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import pv_semisynth_gen as ssg
import pv_semisynth_eval as sse

ETA_MODES = {
    "current": "current window (model)",
    "lag24": "lag 24 h",
    "lag168": "lag 168 h",
    "med7d": "7-day median, lag 24 h",
    "train": "training period",
}
SCN_ORDER = ["outage", "clipping", "shading", "partial_loss", "overgen", "lowlight", "ghi_shift"]


@dataclass
class SensConfig:
    qs: Sequence[float] = (0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999)
    min_lens: Sequence[int] = (1, 2, 3, 4, 6)
    eta_modes: Sequence[str] = tuple(ETA_MODES)
    q_ref: Optional[float] = None        # None -> quantile of the checkpoint (0.995)
    min_len_ref: int = 3
    grid: str = "cross"                  # "cross": q-curve at min_len_ref + min_len-curve at q_ref; "full"
    day_frac: float = 0.05               # daylight for eta_hat medians: P_exp > day_frac * P_rated
    med_window: str = "7D"
    med_lag_h: int = 24
    batch_size: int = 64


def _log(*a):
    print(time.strftime("%H:%M:%S"), "[sensitivity]", *a, flush=True)


# ============================================================================ raw collection
def _collect(pipe, ds, idx, batch_size: int = 64) -> Dict[str, np.ndarray]:
    """One forward pass: raw errors + eta_hat + P_exp / P_rated + P / P_rated (PV nodes)."""
    import torch
    n = pipe.n_pv
    out = {k: [] for k in ("rec", "fore", "eta", "a", "y", "active")}
    pipe.model.eval()
    with torch.no_grad():
        for b in pipe._loader(ds, idx, batch_size):
            b = {k: v.to(pipe.dev) for k, v in b.items()}
            o = pipe.model(b["x"], node_mask=b["edge_mask"])
            out["rec"].append((o["recon"] - b["recon_target"]).abs().mean(-1).cpu().numpy())
            out["fore"].append((o["forecast"] - b["y"]).abs().mean(-1).cpu().numpy())
            out["eta"].append(o["eta"][:, :n].cpu().numpy())
            out["a"].append((b["p_exp"] / pipe._p_scale).cpu().numpy())
            out["y"].append((b["p_meas"][:, :n] / pipe._p_scale).cpu().numpy())
            out["active"].append(b["target_mask"].all(-1).bool().cpu().numpy())
    r = {k: np.concatenate(v).astype(np.float32 if k != "active" else bool) for k, v in out.items()}
    pos = np.asarray(idx) + pipe.window + pipe.horizon - 1
    r["pos"] = pos
    r["ts"] = pd.DatetimeIndex(pipe.df.index[pos])
    return r


def _not_ready(pipe, pos: np.ndarray) -> np.ndarray:
    """Cold-start mask of ``UnsupervisedSTGNN.score`` (activity run shorter than the node delay)."""
    n = pipe.n_pv
    delay = pipe.cold_start_steps + pipe.calibration_steps
    node_delay = np.full(n, delay)
    if pipe.cluster_threshold_prior is not None:
        node_delay[~np.isfinite(pipe.node_thresholds)] = pipe.cold_start_steps
    if pipe.activity_mask is None or delay <= 0:
        return np.zeros((len(pos), n), bool)
    am = pipe.activity_mask.reindex(index=pipe.df.index, columns=pipe.graph.meta["pv_cols"],
                                    fill_value=False).to_numpy(bool)
    ready = np.zeros_like(am)
    run = np.zeros(n, int)
    for t in range(len(am)):
        run = np.where(am[t], run + 1, 0); ready[t] = run >= node_delay
    return ~ready[pos]


@dataclass
class ModelRaw:
    seed: int
    cal: Dict[str, np.ndarray]
    clean: Dict[str, np.ndarray]
    inj: Dict[int, Dict[str, np.ndarray]]
    not_ready: np.ndarray
    calib: dict                     # checkpoint calibration (stats, thresholds, regime, clusters, ...)
    pv_cols: List[str]
    fingerprint: str


def collect_model(mdl: "sse.FrozenSTGNN", D: dict, reals: Dict[int, "ssg.Realisation"],
                  batch_size: int = 64) -> ModelRaw:
    fp0 = mdl.fingerprint()
    pipe = mdl.build(D["df"], D["phys"])
    cal_idx = np.asarray(pipe._fit_idx)[::2]               # = calibrate(cal_stride=2)
    cal = _collect(pipe, pipe.ds_tr, cal_idx, batch_size)
    clean = _collect(pipe, pipe.ds_te, pipe.ds_te.idx, batch_size)
    nr = _not_ready(pipe, clean["pos"])
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
    for ps, real in reals.items():
        p = mdl.build(real.df, real.phys)
        inj[ps] = _collect(p, p.ds_te, p.ds_te.idx, batch_size)
        del p
    assert mdl.fingerprint() == fp0, "weights changed"
    return ModelRaw(mdl.seed, cal, clean, inj, nr, calib, pv, fp0)


# ============================================================================ eta_hat variants
def _train_eta(raw_cal: dict, calib: dict, pv: List[str], day_frac: float) -> np.ndarray:
    eta = raw_cal["eta"].mean(-1); a = raw_cal["a"].mean(-1)
    n = eta.shape[1]
    ok = raw_cal["active"][:, :n] & (a > day_frac)
    med = np.array([np.nanmedian(eta[ok[:, j], j]) if ok[:, j].sum() >= calib["min_stats_count"] else np.nan
                    for j in range(n)])
    lab = calib["cluster_labels"]
    lab = lab.reindex(pv).to_numpy() if lab is not None else np.full(n, np.nan)
    have = np.isfinite(med)
    fleet = np.nanmedian(med) if have.any() else 1.0
    for j in np.flatnonzero(~have):
        same = have & (lab == lab[j]) if pd.notna(lab[j]) else np.zeros(n, bool)
        med[j] = np.nanmedian(med[same]) if same.any() else fleet
    return med


def eta_variant(mode: str, raw: dict, train_med: np.ndarray, cfg: SensConfig) -> np.ndarray:
    """(S, n_pv, H) eta_hat for the variant; causal (only the run's own past)."""
    eta = raw["eta"]
    if mode == "current":
        return eta
    S, n, H = eta.shape
    if mode == "train":
        return np.broadcast_to(train_med[None, :, None], eta.shape).astype(np.float32)
    e = pd.DataFrame(eta.mean(-1), index=raw["ts"])
    if mode.startswith("lag"):
        L = int(mode[3:])
        v = e.shift(freq=pd.Timedelta(hours=L)).reindex(raw["ts"]).to_numpy()
    elif mode == "med7d":
        a = raw["a"].mean(-1)
        day = pd.DataFrame(np.where(raw["active"][:, :n] & (a > cfg.day_frac), e.to_numpy(), np.nan), index=raw["ts"])
        m = day.rolling(cfg.med_window, min_periods=6).median()
        v = m.shift(freq=pd.Timedelta(hours=cfg.med_lag_h)).reindex(raw["ts"]).to_numpy()
    else:
        raise ValueError(mode)
    v = np.where(np.isfinite(v), v, train_med[None, :])
    return np.repeat(v[:, :, None], H, axis=2).astype(np.float32)


def _phy(raw: dict, eta_v: np.ndarray) -> np.ndarray:
    return np.abs(eta_v * raw["a"] - raw["y"]).mean(-1)


def _phy_stats(phy_cal: np.ndarray, active: np.ndarray, calib: dict, pv: List[str]):
    import torch
    from pv_stgnn_unsupervised import borrow_error_stats
    n = phy_cal.shape[1]
    mu = torch.zeros(n); sd = torch.ones(n); has = np.zeros(n, bool)
    for j in range(n):
        v = phy_cal[active[:, j], j]
        if len(v) >= calib["min_stats_count"]:
            mu[j] = float(v.mean()); sd[j] = max(float(v.std()), 1e-6); has[j] = True
    mu, sd, _ = borrow_error_stats(mu, sd, has, n, calib["cluster_labels"], pv)
    return mu.numpy(), sd.numpy()


def _combined(raw: dict, phy: np.ndarray, calib: dict, phy_mu, phy_sd, not_ready=None):
    st, n = calib["stats"], calib["n_pv"]
    zr = ((raw["rec"] - st["rec_mu"]) / st["rec_sd"])[:, :n]
    zf = ((raw["fore"] - st["fore_mu"]) / st["fore_sd"])[:, :n]
    zp = (phy - phy_mu) / phy_sd
    comb = zr + .5 * zf + .5 * zp
    bad = ~raw["active"][:, :n]
    if not_ready is not None:
        bad = bad | not_ready
    comb[bad] = np.nan; zp = zp.copy(); zp[bad] = np.nan
    return comb.astype(np.float64), zp


# ============================================================================ thresholds
def thresholds_from_cal(cal_comb: np.ndarray, cal_pos: np.ndarray, q: float, calib: dict, pv: List[str]) -> dict:
    """Same rules as ``UnsupervisedSTGNN.calibrate`` for an arbitrary quantile q."""
    n = calib["n_pv"]
    fleet = np.full(len(cal_comb), np.nan)
    fin = np.isfinite(cal_comb)
    with np.errstate(all="ignore"):
        fleet[fin.any(1)] = np.nanmax(cal_comb[fin.any(1)], axis=1)
    valid = fleet[np.isfinite(fleet)]
    from pv_stgnn_unsupervised import UnsupervisedSTGNN
    pot = UnsupervisedSTGNN._pot_threshold(valid, q) if len(valid) else 3.0
    node = np.array([np.quantile(cal_comb[fin[:, j], j], q) if fin[:, j].sum() >= 20 else np.nan for j in range(n)])
    reg = None
    if calib["regime"] is not None:
        rg = calib["regime"][cal_pos]
        R = int(calib["regime"].max()) + 1
        reg = np.full((R, n), np.nan)
        for r in range(R):
            sel = rg == r
            for j in range(n):
                v = cal_comb[sel, j]; v = v[np.isfinite(v)]
                if len(v) >= calib["min_regime_count"]:
                    reg[r, j] = np.quantile(v, q)
    prior = None
    if calib["cluster_labels"] is not None:
        lab = calib["cluster_labels"].reindex(pv).to_numpy()
        prior = np.full(n, np.nan)
        for j in range(n):
            same = (lab == lab[j]) & np.isfinite(node)
            pool = node[same] if same.any() else node[np.isfinite(node)]
            prior[j] = np.nanmedian(pool) if len(pool) else np.nan
        if reg is not None:
            for j in range(n):
                if not np.isfinite(reg[:, j]).any():
                    same = (lab == lab[j]) & np.isfinite(reg).any(0)
                    if same.any():
                        reg[:, j] = np.nanmedian(reg[:, same], axis=1)
    return dict(node_thresholds=node, regime_thresholds=reg, cluster_threshold_prior=prior, threshold_pot=pot)


def threshold_matrix(th: dict, pos: np.ndarray, calib: dict) -> np.ndarray:
    node, prior = th["node_thresholds"], th["cluster_threshold_prior"]
    base = np.where(np.isfinite(node), node, prior if prior is not None else np.nan)
    base = np.where(np.isfinite(base), base, th["threshold_pot"])
    T = np.broadcast_to(base, (len(pos), len(base))).copy()
    if th["regime_thresholds"] is not None and calib["regime"] is not None:
        rt = th["regime_thresholds"][calib["regime"][pos]]
        T = np.where(np.isfinite(rt), rt, T)
    return T


# ============================================================================ bundles / evaluation
def _events(comb, thr, physics_z, eta, ts, pv, min_len) -> pd.DataFrame:
    with np.errstate(invalid="ignore"):
        M = np.nan_to_num(comb, nan=-np.inf) > thr
    rows = []
    for j, name in enumerate(pv):
        m = M[:, j]
        if not m.any():
            continue
        d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a >= min_len:
                rows.append(dict(plant=ssg._short(name), start=ts[a], end=ts[b - 1], n_windows=int(b - a),
                                 peak_score=float(comb[a:b, j].max()),
                                 mean_physics_z=float(np.nanmean(physics_z[a:b, j])) if np.isfinite(physics_z[a:b, j]).any() else np.nan,
                                 mean_eta=float(np.nanmean(eta[a:b, j])) if np.isfinite(eta[a:b, j]).any() else np.nan))
    out = pd.DataFrame(rows)
    return out.sort_values("peak_score", ascending=False).reset_index(drop=True) if len(out) else out


def _bundle(comb, thr, zp, eta_v, raw, pv, min_len) -> "sse.ScoreBundle":
    with np.errstate(invalid="ignore"):
        exc = np.nan_to_num(comb, nan=-np.inf) > thr
    eta2 = eta_v.mean(-1).astype(np.float64).copy()
    eta2[~np.isfinite(comb)] = np.nan
    ev = _events(comb, thr, zp, eta2, raw["ts"], pv, min_len)
    return sse.ScoreBundle(raw["ts"], comb, thr, exc, sse._run_mask(exc, min_len), zp, eta2, ev, pv)


@contextlib.contextmanager
def _no_classification():
    """evaluate_detection without the (q-independent, slow) fuzzy / K-Means classification."""
    import pv_clustering as clu
    f0 = clu.anomaly_event_features
    clu.anomaly_event_features = lambda *a, **k: pd.DataFrame()
    try:
        yield
    finally:
        clu.anomaly_event_features = f0


def _bg_rate(b: "sse.ScoreBundle") -> float:
    """Clean-run detections per 1000 scored hours over ALL plants (real alarm budget)."""
    h = int(np.isfinite(b.combined).sum())
    return 1000.0 * len(b.events) / h if h else np.nan


def summarise(E: pd.DataFrame, FA: pd.DataFrame, bg: Sequence[float]) -> dict:
    pl = E[E["level"].isin(["plant", "fleet"])]
    g = pl[pl["duration"] != "short"]
    sh = pl[(pl["duration"] == "short") & (pl["strength"] <= 0.15)]
    tp, fp = g["n_tp_detections"].sum(), g["n_fp_near"].sum()
    rec = g["detected"].mean() if len(g) else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    h = FA["hours"].sum() if len(FA) else 0
    r = dict(recall=rec, precision=prec,
             F1=2 * rec * prec / (rec + prec) if np.isfinite(rec) and np.isfinite(prec) and rec + prec else np.nan,
             raw_hit=g["raw_hit"].mean() if len(g) else np.nan,
             delay_h=g.loc[g["detected"], "delay_h"].median() if g["detected"].any() else np.nan,
             cut_short=1 - sh["detected"].mean() if len(sh) else np.nan,
             clean_fa=1000 * FA["clean_alarms"].sum() / h if h else np.nan,
             induced_fa=1000 * FA["induced_alarms"].sum() / h if h else np.nan,
             bg_all=float(np.nanmean(bg)) if len(bg) else np.nan)
    for st in sorted(g["strength"].unique()):
        r[f"R_{int(round(st * 100))}"] = g.loc[g["strength"] == st, "detected"].mean()
    for scn in SCN_ORDER:
        x = g[g["scenario"] == scn]
        if len(x):
            r[f"R_{scn}"] = x["detected"].mean()
    o = g[g["scenario"] == "outage"]
    r["outage_d_eta"] = o["d_eta"].mean() if len(o) else np.nan
    r["outage_d_phy"] = o["d_physics_z"].mean() if len(o) else np.nan
    return r


# ============================================================================ orchestration
@dataclass
class SensResult:
    grid: pd.DataFrame
    events: pd.DataFrame
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _combos(cfg: SensConfig, q_ref: float):
    qs = sorted(set(map(float, cfg.qs)) | {float(q_ref)})
    if cfg.grid == "full":
        return [(m, q, L) for m in cfg.eta_modes for q in qs for L in cfg.min_lens]
    L0 = cfg.min_len_ref
    out = [(m, q, L0) for m in cfg.eta_modes for q in qs]
    out += [(m, q_ref, L) for m in cfg.eta_modes for L in cfg.min_lens if L != L0]
    return out


def run_sensitivity(D: dict, C: dict, models: Dict[int, "sse.FrozenSTGNN"], ss_res=None,
                    placement_seeds: Sequence[int] = (0, 1, 2), cfg: Optional[SensConfig] = None,
                    gen_cfg=None, cache_dir: Optional[str] = "results/sensitivity", use_cache: bool = True) -> SensResult:
    """q x min_len x eta_hat-variant grid on the SAME placements as section 10 (``ss_res.plans``)."""
    cfg = cfg or SensConfig()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    if ss_res is not None and getattr(ss_res, "plans", None):
        plans = dict(ss_res.plans)
        _log(f"placements from section 10: {sorted(plans)}")
    else:
        gen_cfg = gen_cfg or ssg.GenConfig(boundary=sse.BOUNDARY)
        plans = {ps: ssg.plan_semisynthetic(D, C, seed=ps, cfg=gen_cfg) for ps in placement_seeds}
    reals = {ps: ssg.apply_plan(D, C, p) for ps, p in plans.items()}

    raws: Dict[int, ModelRaw] = {}
    for ms, mdl in models.items():
        f = os.path.join(cache_dir, f"raw_{ms}_{mdl.fingerprint()}.pkl") if cache_dir else None
        if f and use_cache and os.path.exists(f):
            with open(f, "rb") as fh:
                R = pickle.load(fh)
            if sorted(R.inj) == sorted(reals):
                raws[ms] = R; _log(f"model {ms}: raw outputs from cache"); continue
        _log(f"model {ms}: forward passes (calibration, clean test, {len(reals)} placements)")
        raws[ms] = collect_model(mdl, D, reals, cfg.batch_size)
        if f:
            with open(f, "wb") as fh:
                pickle.dump(raws[ms], fh)

    q_ref = float(cfg.q_ref if cfg.q_ref is not None else next(iter(raws.values())).calib["q"])
    combos = _combos(cfg, q_ref)
    ev_cfg = {L: sse.EvalConfig(min_len=L) for L in set(L for _, _, L in combos)}
    rows, E_keep = [], []
    t0 = time.time()
    for ms, R in raws.items():
        calib, pv = R.calib, R.pv_cols
        tmed = _train_eta(R.cal, calib, pv, cfg.day_frac)
        for mode in cfg.eta_modes:
            ecal = eta_variant(mode, R.cal, tmed, cfg)
            phy_cal = _phy(R.cal, ecal)
            if mode == "current":
                mu, sd = calib["stats"]["phy_mu"], calib["stats"]["phy_sd"]
            else:
                mu, sd = _phy_stats(phy_cal, R.cal["active"][:, :calib["n_pv"]], calib, pv)
            cal_comb, _ = _combined(R.cal, phy_cal, calib, mu, sd)
            sets = {"clean": R.clean, **{ps: r for ps, r in R.inj.items()}}
            Z = {}
            for k, r in sets.items():
                ev_ = eta_variant(mode, r, tmed, cfg)
                Z[k] = (ev_,) + _combined(r, _phy(r, ev_), calib, mu, sd, R.not_ready)
            for q in sorted(set(q for m_, q, _ in combos if m_ == mode)):
                if mode == "current" and abs(q - calib["q"]) < 1e-12:
                    th = {k: calib[k] for k in ("node_thresholds", "regime_thresholds", "cluster_threshold_prior",
                                                "threshold_pot")}
                else:
                    th = thresholds_from_cal(cal_comb, R.cal["pos"], q, calib, pv)
                thr = threshold_matrix(th, R.clean["pos"], calib)
                for L in sorted(set(L for m_, q_, L in combos if m_ == mode and q_ == q)):
                    ev0, comb0, zp0 = Z["clean"]
                    cb = _bundle(comb0, thr, zp0, ev0, R.clean, pv, L)
                    E, FA = [], []
                    for ps, real in reals.items():
                        evi, combi, zpi = Z[ps]
                        ib = _bundle(combi, thr, zpi, evi, R.inj[ps], pv, L)
                        with _no_classification(), contextlib.redirect_stdout(io.StringIO()):
                            out = sse.evaluate_detection(D, C, real, cb, ib, ev_cfg[L], None)
                        tag = dict(eta_mode=mode, q=q, min_len=L, model_seed=ms, placement=ps)
                        E.append(out["events"].assign(**tag)); FA.append(out["false_alarms"].assign(**tag))
                    E = pd.concat(E, ignore_index=True); FA = pd.concat(FA, ignore_index=True)
                    rows.append(dict(eta_mode=mode, q=q, min_len=L, model_seed=ms, **summarise(E, FA, [_bg_rate(cb)])))
                    E_keep.append(E)
        _log(f"model {ms}: grid done ({time.time() - t0:.0f}s)")
    G = pd.DataFrame(rows)
    out = SensResult(grid=G, events=pd.concat(E_keep, ignore_index=True) if E_keep else pd.DataFrame(),
                     meta=dict(q_ref=q_ref, min_len_ref=cfg.min_len_ref, cfg=cfg.__dict__, model_seeds=list(raws),
                               placements=sorted(reals), fingerprints={ms: R.fingerprint for ms, R in raws.items()},
                               weights_unchanged=all(m.unchanged() for m in models.values())))
    out.tables = build_tables(out, ss_res)
    if cache_dir:
        with open(os.path.join(cache_dir, "sensitivity_result.pkl"), "wb") as f:
            pickle.dump(out, f)
    _log("weights unchanged:", out.meta["weights_unchanged"])
    return out


# ============================================================================ tables
_COLS = {"recall": "Recall", "precision": "Precision", "F1": "F1", "raw_hit": "Raw hit",
         "delay_h": "Delay, h", "cut_short": "Short events cut", "clean_fa": "FA controls, /1000 h",
         "induced_fa": "Induced FA, /1000 h", "bg_all": "Background all plants, /1000 h",
         "R_5": "R 5 %", "R_15": "R 15 %", "R_40": "R 40 %",
         "outage_d_eta": "Δη̂ outage", "outage_d_phy": "Δz_phys outage",
         "recall_at_budget": "Recall at budget (interp.)", "q_at_budget": "q at budget (interp.)"}
_SCN_T = {k: s.title for k, s in ssg.SCENARIOS.items()}


def _agg(G: pd.DataFrame) -> pd.DataFrame:
    """Mean over model seeds (+ std of recall)."""
    keys = ["eta_mode", "q", "min_len"]
    num = [c for c in G.columns if c not in keys + ["model_seed"]]
    A = G.groupby(keys, sort=False)[num].mean()
    A["recall_sd"] = G.groupby(keys, sort=False)["recall"].std()
    return A.reset_index()


def _pretty(T: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame({"η̂": T["eta_mode"].map(ETA_MODES), "q": T["q"], "min_len": T["min_len"]})
    for c in cols:
        if c in T:
            out[_COLS.get(c, _SCN_T.get(c[2:], c) if c.startswith("R_") else c)] = T[c].values
    return out.reset_index(drop=True)


def build_tables(res: SensResult, ss_res=None) -> Dict[str, pd.DataFrame]:
    G = res.grid
    if not len(G):
        return {}
    A = _agg(G)
    q0, L0 = res.meta["q_ref"], res.meta["min_len_ref"]
    main = ["recall", "precision", "F1", "raw_hit", "delay_h", "cut_short", "clean_fa", "induced_fa", "bg_all"]
    T = {}
    T["q_curve"] = _pretty(A[A["min_len"] == L0].sort_values(["eta_mode", "q"], key=_mode_key), main + ["R_5", "R_15", "R_40"])
    T["minlen_curve"] = _pretty(A[np.isclose(A["q"], q0)].sort_values(["eta_mode", "min_len"], key=_mode_key), main)
    scn = [f"R_{s}" for s in SCN_ORDER if f"R_{s}" in A]
    T["scenario"] = _pretty(A[np.isclose(A["q"], q0) & (A["min_len"] == L0)].sort_values("eta_mode", key=_mode_key),
                            ["recall"] + scn + ["outage_d_eta", "outage_d_phy"])
    # operating point at the false-alarm budget of the baseline (clean alarms on control plants):
    # every metric is linearly interpolated along the q-curve of the variant at exactly that budget
    base = A[(A["eta_mode"] == "current") & np.isclose(A["q"], q0) & (A["min_len"] == L0)]
    if len(base):
        budget = float(base["clean_fa"].iloc[0])
        cols = ["q", "recall", "precision", "F1", "bg_all", "delay_h"] + scn
        rows = []
        for m, g in A[A["min_len"] == L0].groupby("eta_mode", sort=False):
            c = g[np.isfinite(g["clean_fa"])].sort_values(["clean_fa", "q"])
            fa = c["clean_fa"].to_numpy(float)
            r = dict(eta_mode=m, min_len=L0, clean_fa=budget)
            fa_u, iu = np.unique(fa, return_index=True)
            inside = len(fa_u) >= 2 and fa_u[0] <= budget <= fa_u[-1]
            for k in cols:
                v = c[k].to_numpy(float)[iu]
                ok = np.isfinite(v)
                r[k] = (float(np.interp(budget, fa_u[ok], v[ok])) if inside and ok.sum() >= 2
                        else float(c.iloc[int(np.argmin(np.abs(fa - budget)))][k]))
            r["interpolated"] = bool(inside)
            rows.append(r)
        F = pd.DataFrame(rows).sort_values("eta_mode", key=_mode_key)
        T["fa_matched"] = _pretty(F, ["recall", "precision", "F1", "bg_all", "delay_h"] + scn)
        T["fa_matched"].insert(3, "FA budget, /1000 h", budget)
        T["fa_matched"].insert(4, "Interpolated", F["interpolated"].to_numpy())
    # reproduction of section 10 (current eta, checkpoint q, min_len 3)
    if ss_res is not None and len(getattr(ss_res, "events", [])):
        Es = ss_res.events
        Es = Es[Es["level"].isin(["plant", "fleet"]) & (Es["duration"] != "short")]
        mine = res.events[(res.events["eta_mode"] == "current") & np.isclose(res.events["q"], q0) &
                          (res.events["min_len"] == L0)]
        mine = mine[mine["level"].isin(["plant", "fleet"]) & (mine["duration"] != "short")]
        if len(mine):
            a = Es.groupby("scenario")["detected"].mean(); b = mine.groupby("scenario")["detected"].mean()
            R = pd.DataFrame({"Section 10": a, "Recomputed (current, q_ref, min_len_ref)": b})
            R.index = [_SCN_T.get(i, i) for i in R.index]
            R.loc["All"] = [Es["detected"].mean(), mine["detected"].mean()]
            T["reproduction"] = R.reset_index().rename(columns={"index": "Scenario"})
    return T


def _mode_key(s: pd.Series):
    order = {m: i for i, m in enumerate(ETA_MODES)}
    return s.map(order) if s.name == "eta_mode" else s


def show(res: SensResult, digits: int = 3):
    from IPython.display import display, Markdown
    titles = {"reproduction": "Reproduction of section 10 (recomputation check)",
              "q_curve": f"Curve over q (min_len = {res.meta['min_len_ref']})",
              "minlen_curve": f"Curve over min_len (q = {res.meta['q_ref']})",
              "scenario": f"Recall by scenario at q = {res.meta['q_ref']}, min_len = {res.meta['min_len_ref']}",
              "fa_matched": "Operating point at the false-alarm budget of the reference variant"}
    for k, t in titles.items():
        if k in res.tables:
            display(Markdown(f"**{t}**")); display(res.tables[k].round(digits))


# ============================================================================ figures
def plot_all(res: SensResult, fig_dir: Optional[str] = "results/figures") -> Dict[str, object]:
    import matplotlib.pyplot as plt
    A = _agg(res.grid)
    q0, L0 = res.meta["q_ref"], res.meta["min_len_ref"]
    figs = {}
    modes = [m for m in ETA_MODES if m in set(A["eta_mode"])]
    col = dict(zip(ETA_MODES, ["#444444", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]))
    # 1. q-curves
    f, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for m in modes:
        g = A[(A["eta_mode"] == m) & (A["min_len"] == L0)].sort_values("q")
        lab = ETA_MODES[m]
        ax[0].plot(g["q"], g["recall"], "o-", c=col[m], label=lab)
        ax[1].plot(g["q"], g["clean_fa"], "o-", c=col[m], label=lab)
        ax[2].plot(g["clean_fa"], g["recall"], "o-", c=col[m], label=lab)
        for _, r in g.iterrows():
            if m == "current":
                ax[2].annotate(f"q={r['q']:g}", (r["clean_fa"], r["recall"]), fontsize=7, xytext=(4, -10),
                               textcoords="offset points")
    ax[0].set(xlabel="q", ylabel="recall (day + week)", title=f"Recall, min_len = {L0}")
    ax[1].set(xlabel="q", ylabel="false alarms controls, /1000 h", title="Background alarms (clean data)")
    ax[1].set_yscale("symlog", linthresh=0.1); ax[2].set_xscale("symlog", linthresh=0.1)
    ax[2].set(xlabel="false alarms controls, /1000 h", ylabel="recall", title="Trade-off recall — false alarms")
    from matplotlib.ticker import FixedLocator, FuncFormatter
    tk = [0, 0.1, 0.3, 1, 3, 10, 30, 100]
    for a_, axis in ((ax[1], "y"), (ax[2], "x")):
        ax_ = a_.yaxis if axis == "y" else a_.xaxis
        ax_.set_major_locator(FixedLocator(tk)); ax_.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for a in ax:
        a.axvline(q0, c="grey", ls=":", lw=.8) if a is not ax[2] else None
        a.grid(alpha=.3)
    ax[0].legend(fontsize=7)
    f.tight_layout(); figs["q"] = f
    # 2. min_len curves
    f, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for m in modes:
        g = A[(A["eta_mode"] == m) & np.isclose(A["q"], q0)].sort_values("min_len")
        ax[0].plot(g["min_len"], g["recall"], "o-", c=col[m], label=ETA_MODES[m])
        ax[1].plot(g["min_len"], g["clean_fa"], "o-", c=col[m])
        ax[2].plot(g["min_len"], g["cut_short"], "o-", c=col[m])
    ax[0].set(xlabel="min_len, steps", ylabel="recall", title=f"Recall, q = {q0}")
    ax[1].set(xlabel="min_len, steps", ylabel="false alarms controls, /1000 h", title="Background alarms")
    ax[2].set(xlabel="min_len, steps", ylabel="share", title="Short events cut (1–2 steps, s ≤ 15 %)")
    for a in ax:
        a.grid(alpha=.3); a.set_xticks(sorted(set(A["min_len"])))
    ax[0].legend(fontsize=7)
    f.tight_layout(); figs["minlen"] = f
    # 3. scenarios at the FA-matched operating point
    if "fa_matched" in res.tables:
        F = res.tables["fa_matched"]
        scn = [c for c in F.columns if c in _SCN_T.values()]
        f, ax = plt.subplots(figsize=(13, 4.3))
        w = 0.8 / max(len(F), 1)
        x = np.arange(len(scn))
        for i, (_, r) in enumerate(F.iterrows()):
            m = [k for k, v in ETA_MODES.items() if v == r["η̂"]][0]
            ax.bar(x + i * w - 0.4 + w / 2, [r[s] for s in scn], w, color=col[m], label=f"{r['η̂']} (q ≈ {r['q']:.4f})")
        ax.set_xticks(x); ax.set_xticklabels(scn, rotation=15, ha="right", fontsize=8)
        ax.set(ylabel="recall", ylim=(0, 1),
               title=f"Recall by scenario at an equal false-alarm budget ({F['FA budget, /1000 h'].iloc[0]:.2f} /1000 h on control plants), min_len = {L0}")
        ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
        f.tight_layout(); figs["scenario"] = f
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        for k, fg in figs.items():
            fg.savefig(os.path.join(fig_dir, f"fig_sensitivity_{k}.png"), dpi=150, bbox_inches="tight")
    return figs


# ============================================================================ LaTeX / export
TEX_CAPTIONS = {
    "reproduction": "Recomputation check: recall of section 10 and of the recomputation at the model $\\hat\\eta$, $q_{\\mathrm{ref}}$ and min\\_len$_{\\mathrm{ref}}$",
    "q_curve": "Detection sensitivity to the threshold quantile $q$ and the source of $\\hat\\eta$ (frozen STGNN weights, mean over seeds)",
    "minlen_curve": "Detection sensitivity to the minimum event length min\\_len",
    "scenario": "Recall by scenario for the $\\hat\\eta$ variants and response to a full outage",
    "fa_matched": "$\\hat\\eta$ variants at the same false-alarm budget on control plants",
}


def export_latex(res: SensResult, out_dir: str = "results/sensitivity") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths, allp = {}, ["% requires \\usepackage{booktabs,graphicx,longtable,array}; Cyrillic: \\usepackage[T2A]{fontenc}, "
                       "\\usepackage[utf8]{inputenc}, \\usepackage[russian]{babel}", ""]
    for k, cap in TEX_CAPTIONS.items():
        t = res.tables.get(k)
        if t is None or not len(t):
            continue
        tex = sse.df_to_latex(t, cap, f"tab:sens_{k}", digits=3)
        p = os.path.join(out_dir, f"sensitivity_{k}.tex")
        with open(p, "w", encoding="utf-8") as f:
            f.write(tex)
        paths[k] = p; allp.append(tex)
        t.to_csv(os.path.join(out_dir, f"sensitivity_{k}.csv"), index=False)
    res.grid.to_csv(os.path.join(out_dir, "sensitivity_grid_by_seed.csv"), index=False)
    p = os.path.join(out_dir, "sensitivity_tables.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(allp))
    paths["all"] = p
    return paths


def zip_outputs(out_dir: str = "results/sensitivity", fig_dir: str = "results/figures",
                name: str = "sensitivity_results.zip") -> str:
    import zipfile
    p = os.path.join(out_dir, name)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out_dir)):
            if f.endswith((".tex", ".csv")):
                z.write(os.path.join(out_dir, f), f)
        if fig_dir and os.path.isdir(fig_dir):
            for f in sorted(os.listdir(fig_dir)):
                if f.startswith("fig_sensitivity_"):
                    z.write(os.path.join(fig_dir, f), os.path.join("figures", f))
    return p


download_link = sse.download_link


def main():
    import argparse
    ap = argparse.ArgumentParser(description="q / min_len / eta_hat sensitivity of the semi-synthetic detection")
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--semisynth", default="results/semisynth/semisynth_result.pkl")
    ap.add_argument("--out", default="results/sensitivity")
    ap.add_argument("--model_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--placements", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--grid", default="cross", choices=["cross", "full"])
    a = ap.parse_args()
    D = pickle.load(open(os.path.join(a.cache, "data.pkl"), "rb"))
    C = pickle.load(open(os.path.join(a.cache, "clusters.pkl"), "rb"))
    ss = pickle.load(open(a.semisynth, "rb")) if os.path.exists(a.semisynth) else None
    models = sse.frozen_models(D, C, a.model_seeds, cache_dir=a.cache, train_missing=False)
    res = run_sensitivity(D, C, models, ss, a.placements, SensConfig(grid=a.grid), cache_dir=a.out)
    plot_all(res, os.path.join(os.path.dirname(a.out) or ".", "figures"))
    print(export_latex(res, a.out))


if __name__ == "__main__":
    main()
