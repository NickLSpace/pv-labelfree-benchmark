"""
End-to-end pipeline: cohorts -> multi-view clustering -> dynamic supergraph -> STGNN
(baseline vs multi-view, 5 seeds) -> anomaly clusters -> SPS / Kalman evidence -> tables/figures.

    python run_pipeline.py --xlsx data.xlsx --ttl PV-generation-system-metadata.ttl \
                           --out results --epochs 6 --seeds 0 1 2 3 4 [--estimator kalman]

``--estimator sps`` (default) uses Sign-Perturbed Sums (pv_sps.py); ``--estimator kalman``
uses the random-walk Kalman filter / RTS smoother (pv_kalman.py) with the same output schema.

Stages are cached in <out>/cache/*.pkl so reporting can be re-run without retraining
(`--stage report`).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict

# Ensure the script directory (and its parent on Kaggle) is on sys.path
# so sibling modules (pv_stgnn_unsupervised, pv_physics_graph, etc.) are importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [_script_dir, os.path.dirname(_script_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from pv_physics_graph import (DEFAULT_EXO, BrickPVModel, PVGraphBuilder, PhysicsPrior,
                              build_from_dataframe, node_activity_mask, scenario_commissioning)
from pv_cohorts import (analyse_cohorts, clearness_index, harmonise_irradiance,
                        irradiance_sensor_check, short)
import pv_clustering as clu
import pv_sps as sps_mod

BOUNDARY = "2023-01-01"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _find_file(name, extra_paths=None):
    """Search for a file in common locations (Kaggle, local, home, recursive)."""
    candidates = [
        name,
        os.path.join('/kaggle/input', 'datasets', 'skaldfire', 'fuzzy-stgnn-lib', name),
        os.path.join(os.path.expanduser('~'), name),
    ]
    if extra_paths:
        candidates = extra_paths + candidates
    for p in candidates:
        if os.path.exists(p):
            return p
    for root in ['/kaggle/input', '.', '..']:
        if os.path.isdir(root):
            for d, _, files in os.walk(root):
                if name in files:
                    return os.path.join(d, name)
    return None


# ============================================================================ stage 1: data
def stage_data(df_raw: pd.DataFrame, ttl_path: str = None) -> Dict[str, object]:
    """Stage 1: build graph and physics model from a DataFrame.

    Always accepts a pre-loaded DataFrame (raw or preprocessed).
    File loading (xlsx/pickle) should be done by the caller — main() for
    CLI, or directly in a notebook cell.
    """
    if ttl_path is None:
        ttl_path = _find_file('PV-generation-system-metadata.ttl')
    if ttl_path is None:
        raise FileNotFoundError(
            "TTL file not found. Pass ttl_path explicitly: "
            "stage_data(df_raw, ttl_path='/your/path/PV-generation-system-metadata.ttl')")
    raw = df_raw
    df0, brick, builder, graph0, phys0, _ = build_from_dataframe(raw, ttl_path, exo_cols=DEFAULT_EXO,
                                                                peer_fit_end=BOUNDARY)
    observed = builder.observed
    pv_cols = graph0.meta["pv_cols"]
    mapping = graph0.meta["mapping"]
    rated = {c: brick.sites[mapping[c]].rated_kw or 0.0 for c in pv_cols}
    comm = scenario_commissioning(df0, brick, pv_cols, scenario="dynamic_actual")

    # --- systemic-change test of the irradiance sensor (PV independent) -------
    sensor = irradiance_sensor_check(df0)
    log("sensor check:", {k: v for k, v in sensor.items() if k in ("change_point", "shift_pct", "factor_after_cp", "invalid_months")})
    # cohort analysis on RAW irradiance (to document the synchronous 2023 drop)
    act0 = node_activity_mask(df0, graph0, comm)
    ca_raw = analyse_cohorts(df0, phys0, pv_cols, observed, act0)
    # harmonised irradiance -> physics recomputed
    df = harmonise_irradiance(df0, sensor)
    phys = builder.physics_tensors(df, graph0)
    graph_base = builder.build(df, pv_cols, exo_cols=graph0.meta["exo_cols"], peer_fit_end=BOUNDARY,
                               observed=observed)
    act = node_activity_mask(df, graph_base, comm)
    ca = analyse_cohorts(df, phys, pv_cols, observed, act, irr_valid=sensor["irr_valid"])
    log("cohorts:", ca.cohorts["cohort"].value_counts().to_dict())

    # per-plant cell temperature / POA for behavioural features and SPS
    prior = builder.prior
    times = pd.DatetimeIndex(df.index)
    age = (times - times[0]).days.to_numpy(float) / 365.25
    t_cell = np.zeros((len(df), len(pv_cols))); poa = np.zeros_like(t_cell)
    for j, c in enumerate(pv_cols):
        _, aux = prior.expected_power(df["Irradiance (W/m2)"].to_numpy(float), times,
                                      df["Temp (Degree Celsius)"].to_numpy(float),
                                      df["wind_speed"].to_numpy(float), brick.sites[mapping[c]], 8.0, age)
        t_cell[:, j], poa[:, j] = aux["t_cell"], aux["poa"]
    kt = clearness_index(df)
    return dict(df=df, df_raw_irr=df0, brick=brick, builder=builder, graph_base=graph_base,
                phys=phys, phys_raw=phys0, observed=observed, pv_cols=pv_cols, mapping=mapping,
                rated=rated, comm=comm, act=act, sensor=sensor, ca=ca, ca_raw=ca_raw,
                t_cell=t_cell, poa=poa, kt=kt)


# ============================================================================ stage 2: clustering
def stage_clusters(D, args) -> Dict[str, object]:
    df, phys, pv_cols = D["df"], D["phys"], D["pv_cols"]
    from pv_cohorts import coverage_mask
    valid = coverage_mask(df, phys, pv_cols, D["observed"], D["act"]) & np.asarray(D["sensor"]["irr_valid"])[:, None]
    tech = clu.technical_clusters(D["brick"], D["mapping"], pv_cols)
    log("technical:", tech.metrics_row())
    F = clu.behavioural_features(df, phys, pv_cols, D["rated"], valid, BOUNDARY, brick=D["brick"],
                                 mapping=D["mapping"], t_cell=D["t_cell"], poa=D["poa"])
    beh = clu.behavioural_clusters(F)
    log("behavioural:", beh.metrics_row())
    S = clu.behavioural_similarity(beh)
    W = clu.weather_features(df, kt=D["kt"])
    wr = clu.weather_regimes(W, BOUNDARY)
    log("weather regimes:", wr.result.metrics_row(), wr.names)
    base_peer = {(D["graph_base"].node_names[s], D["graph_base"].node_names[d])
                 for s, d, t in zip(*D["graph_base"].edge_index, D["graph_base"].edge_type) if t == 1}
    b_edges = clu.behavioural_edges(S, k=3, existing=base_peer)
    prior = clu.peer_prior_from_clusters(tech.labels, across=0.5)
    graph_mv = D["builder"].build(df, pv_cols, exo_cols=D["graph_base"].meta["exo_cols"],
                                  peer_fit_end=BOUNDARY, observed=D["observed"], peer_prior=prior,
                                  extra_edges=b_edges)
    graph_mv.meta["tech_cluster"] = tech.labels.to_dict()
    graph_mv.meta["behav_cluster"] = beh.labels.to_dict()
    log("multi-view graph edges per type:", np.bincount(graph_mv.edge_type).tolist())
    cold = clu.cold_start_baseline(F, tech.labels)
    return dict(valid=valid, tech=tech, F=F, beh=beh, S=S, W=W, wr=wr, b_edges=b_edges,
                peer_prior=prior, graph_mv=graph_mv, cold=cold)


# ============================================================================ stage 3: STGNN
def run_variant(D, C, variant: str, seed: int, args) -> Dict[str, object]:
    import torch
    from pv_stgnn_unsupervised import UnsupervisedSTGNN
    torch.set_num_threads(max(os.cpu_count() or 1, 1))
    df, phys, pv_cols = D["df"], D["phys"], D["pv_cols"]
    mv = variant == "multiview"
    graph = C["graph_mv"] if mv else D["graph_base"]
    act = node_activity_mask(df, graph, D["comm"])
    obs = D["observed"].reindex(columns=graph.node_names, fill_value=True)
    pipe = UnsupervisedSTGNN(df, graph, phys, window=args.window, horizon=1, hidden=args.hidden,
                             tcn_layers=4, gat_layers=2, heads=4, seed=seed, activity_mask=act,
                             observed=obs, cold_start_steps=args.window, calibration_steps=168,
                             context=C["wr"].onehot if mv else None,
                             regime=C["wr"].regime if mv else None,
                             cluster_labels=C["tech"].labels if mv else None,
                             device=args.device)
    pipe.split(BOUNDARY, val_fraction=0.1)
    t0 = time.time()
    pipe.fit(epochs=args.epochs, batch_size=64, patience=3, train_stride=args.train_stride, verbose=True)
    train_time = time.time() - t0
    pipe.calibrate(q=0.995, cal_stride=2)
    # frozen snapshot BEFORE the online fine-tuning (used e.g. by semi-synthetic validation)
    if getattr(args, "ckpt_path", None):
        from pv_physics_graph import data_signature
        save_frozen_checkpoint(pipe, args.ckpt_path, dict(variant=variant, seed=seed, window=args.window,
                                                          hidden=args.hidden, epochs=args.epochs,
                                                          train_stride=args.train_stride,
                                                          **data_signature(D, BOUNDARY)))
    res = pipe.score("test")
    err = pipe.node_errors("test")
    coh = D["ca"].cohorts["cohort"]
    groups = {"all": pv_cols}
    for k in (1, 2, 3):
        groups[f"cohort {k}"] = [c for c in pv_cols if coh.get(c) == k]
    summ = pipe.summarise_errors(err, pv_cols, groups)
    new_cols = groups["cohort 3"]
    old_cols = groups["cohort 1"]
    cs = pipe.cold_start_hours(err, pv_cols, new_cols, old_cols)
    events = pipe.events(res, min_len=3)
    # exceedance share per plant (regime-aware thresholds when available)
    with np.errstate(invalid="ignore"):
        exc = res.combined[:, :len(pv_cols)] > pipe.last_thr
    fin = np.isfinite(res.combined[:, :len(pv_cols)])
    exceed_share = pd.Series(exc.sum(0) / np.maximum(fin.sum(0), 1), index=[short(c) for c in pv_cols])
    rank = res.frame().iloc[:, :len(pv_cols)].mean()
    rank.index = [short(c) for c in rank.index]
    # --- online fine-tuning: first 30 days of 2023 (recent) + replay of train windows
    pos = err["pos"]
    split_t = pd.Timestamp(BOUNDARY) + pd.Timedelta(days=args.ft_days)
    after = np.asarray(df.index[pos] >= split_t)
    before = pipe.summarise_errors({k: (v[after] if isinstance(v, np.ndarray) and v.ndim == 2 else v)
                                    for k, v in err.items()}, pv_cols, groups)
    recent = pipe.ds_te.idx[~after]
    rng = np.random.default_rng(seed)
    replay = rng.choice(pipe._fit_idx, size=min(len(recent), len(pipe._fit_idx)), replace=False)
    t1 = time.time()
    pipe.online_fine_tune(recent, replay, epochs=1, batch_size=64, lr=1e-4)
    ft_time = time.time() - t1
    err_ft = pipe.node_errors("test", idx=pipe.ds_te.idx[after])
    after_ft = pipe.summarise_errors(err_ft, pv_cols, groups)
    hist = pd.DataFrame(pipe.history)
    return dict(variant=variant, seed=seed, summary=summ, cold_start=cs, events=events,
                exceed_share=exceed_share, rank=rank, fixed_after=before, ft_after=after_ft,
                train_time_s=train_time, ft_time_s=ft_time, history=hist,
                prior_strength=pipe.prior_strength(), n_params=sum(p.numel() for p in pipe.model.parameters()),
                node_delay=pipe.node_delay, threshold_pot=pipe.threshold_pot,
                eta_mean=np.nanmean(res.eta, axis=0))


def save_frozen_checkpoint(pipe, path, meta=None):
    """Weights + train-period calibration (z-stats, node / regime / cluster thresholds)
    of a trained, calibrated pipe.  Optional artefact; failures never stop the pipeline."""
    try:
        import copy as _copy
        import torch
        ck = dict(model={k: v.detach().cpu().clone() for k, v in pipe.model.state_dict().items()},
                  stats={k: v.detach().cpu().clone() for k, v in pipe.stats.items()}, meta=dict(meta or {}))
        for a in ("threshold", "threshold_pot", "node_thresholds", "regime_thresholds",
                  "cluster_threshold_prior", "q"):
            ck[a] = _copy.deepcopy(getattr(pipe, a, None))
        ck["stats_source"] = _copy.deepcopy(getattr(pipe, "stats_source", None)) or None
        torch.save(ck, path)
    except Exception as exc:  # pragma: no cover
        log(f"checkpoint not saved ({exc})")


def stage_stgnn(D, C, args, cache_dir):
    out = []
    for variant in args.variants:
        for seed in args.seeds:
            f = os.path.join(cache_dir, f"run_{variant}_{seed}.pkl")
            if os.path.exists(f):
                out.append(pickle.load(open(f, "rb"))); continue
            log(f"=== STGNN {variant} seed={seed}")
            args.ckpt_path = os.path.join(cache_dir, f"stgnn_frozen_{variant}_{seed}.pt")
            r = run_variant(D, C, variant, seed, args)
            args.ckpt_path = None
            pickle.dump(r, open(f, "wb"))
            out.append(r)
            log(r["summary"].to_string())
    return out


# ============================================================================ stage 4: anomalies + SPS / Kalman
def anomaly_cache_name(estimator: str = "sps") -> str:
    """anomaly_sps.pkl is kept for SPS (backward compatible with existing caches)."""
    return "anomaly_sps.pkl" if estimator == "sps" else f"anomaly_{estimator}.pkl"


def stage_anomaly_sps(D, C, runs, args, estimator: str = None):
    """Anomaly clusters + statistical evidence for b1.

    ``estimator`` = "sps" (Sign-Perturbed Sums, exact finite-sample regions) or
    "kalman" (random-walk Kalman filter + RTS smoother).  Output keys are identical
    (``sps_periods`` / ``sps_split`` / ``sps_cluster`` hold the estimator output, the
    name is kept for compatibility with pv_report); ``A["estimator"]`` tells which one."""
    estimator = (estimator or getattr(args, "estimator", "sps") or "sps").lower()
    df, phys, pv_cols = D["df"], D["phys"], D["pv_cols"]
    mv = [r for r in runs if r["variant"] == "multiview"] or runs
    ref = mv[0]
    feat0 = clu.anomaly_event_features(ref["events"], df, phys, pv_cols, C["wr"].regime)
    anom = clu.anomaly_clusters(feat0) if len(feat0) >= 10 else None
    per_seed = {}
    if anom is not None:
        mp = clu.most_probable_class(feat0, anom, pv_cols)
        names = anom.extra["class_names"]
        for r in mv:
            f = clu.anomaly_event_features(r["events"], df, phys, pv_cols, C["wr"].regime)
            if len(f) == 0:
                continue
            X0 = f[clu.ANOMALY_FEATURES].astype(float).fillna(0.0)
            X0["duration_h"] = np.log1p(X0["duration_h"])
            lab = anom.model.predict(anom.scaler.transform(X0.to_numpy(float)))
            tmp = clu.ClusterResult("anomaly", pd.Series(lab, index=f.index), anom.k, anom.algorithm,
                                    X0, anom.scores, extra=dict(class_names=names))
            per_seed[r["seed"]] = clu.most_probable_class(f, tmp, pv_cols)["most_probable_class"]
        if per_seed:
            P = pd.DataFrame(per_seed)
            mode = P.mode(axis=1)[0]
            mp["class_mode_5seeds"] = mode
            mp["class_agreement_5seeds"] = (P.eq(mode, axis=0)).mean(1)
    else:
        mp = pd.DataFrame(index=[short(c) for c in pv_cols])
        mp["most_probable_class"] = "Normal"; mp["class_probability"] = 1.0
    cloud = (1.0 - D["kt"].clip(0, 1.2)).fillna(1.0).to_numpy()
    valid = C["valid"]
    daily = None
    if estimator == "kalman":
        import pv_kalman as kf_mod
        out = kf_mod.kalman_all(df, phys, pv_cols, D["rated"], valid, D["t_cell"], cloud, BOUNDARY,
                                C["tech"].labels, cfg=getattr(args, "kalman_cfg", None))
        per, split, clus, daily = out["periods"], out["split"], out["cluster"], out["daily"]
        fuse = kf_mod.fuse_evidence
    elif estimator == "sps":
        per = sps_mod.sps_periods(df, phys, pv_cols, D["rated"], valid, D["t_cell"], cloud, freq="QS")
        split = sps_mod.sps_split(df, phys, pv_cols, D["rated"], valid, D["t_cell"], cloud, BOUNDARY)
        clus = sps_mod.sps_cluster(df, phys, pv_cols, D["rated"], valid, D["t_cell"], cloud,
                                   C["tech"].labels, BOUNDARY)
        fuse = sps_mod.fuse_evidence
    else:
        raise ValueError(f"unknown estimator {estimator!r}: use 'sps' or 'kalman'")
    exceed = pd.concat([r["exceed_share"] for r in mv], axis=1).mean(1)
    rank = pd.concat([r["rank"] for r in mv], axis=1).mean(1)
    evid = fuse(rank, exceed, split, per)
    log(f"estimator: {estimator}; plants with disjoint train/test b1:",
        int((split["regions_disjoint"] == True).sum()))  # noqa: E712
    return dict(anom_feat=feat0, anom=anom, most_probable=mp, sps_periods=per, sps_split=split,
                sps_cluster=clus, evidence=evid, estimator=estimator, est_daily=daily)


# ============================================================================ notebook API
def run_from_dataframe(df_raw, ttl_path=None,
                      out_dir="results", epochs=6, hidden=32, window=48,
                      train_stride=4, ft_days=30, seeds=None, variants=None,
                      device=None, stage="all", estimator="sps", kalman_cfg=None):
    """Run the full pipeline from a DataFrame (notebook mode).

    Replaces the CLI ``main()`` that reads from ``output.xlsx``.
    Returns the data dict ``D``, clusters dict ``C``, runs list, and anomaly/SPS dict ``A``.
    ``estimator`` = "sps" (default) or "kalman" selects the b1 estimator of stage 4;
    ``kalman_cfg`` is an optional ``pv_kalman.KalmanConfig``.

    Usage in a notebook:
        from run_pipeline import run_from_dataframe
        D, C, runs, A = run_from_dataframe(df_raw, ttl_path="...", epochs=15)
    """
    # Resolve TTL path: use provided, or search common locations
    if ttl_path is None:
        ttl_path = _find_file('PV-generation-system-metadata.ttl')
    if ttl_path is None:
        raise FileNotFoundError(
            "TTL file not found. Pass ttl_path explicitly: "
            "run_from_dataframe(df_raw, ttl_path='/your/path/PV-generation-system-metadata.ttl')")

    class _Args:
        pass
    args = _Args()
    args.ttl = ttl_path
    args.out = out_dir
    args.epochs = epochs
    args.hidden = hidden
    args.window = window
    args.train_stride = train_stride
    args.ft_days = ft_days
    args.seeds = seeds or [0, 1, 2]
    args.variants = variants or ["baseline", "multiview"]
    args.device = device
    args.stage = stage
    args.estimator = estimator
    args.kalman_cfg = kalman_cfg

    cache = os.path.join(out_dir, "cache")
    os.makedirs(cache, exist_ok=True)
    fD = os.path.join(cache, "data.pkl")
    fC = os.path.join(cache, "clusters.pkl")

    # Stage 1: data (always rebuild from DataFrame)
    D = stage_data(df_raw, ttl_path)
    pickle.dump(D, open(fD, "wb"))

    # Stage 2: clusters
    if os.path.exists(fC):
        C = pickle.load(open(fC, "rb"))
    else:
        C = stage_clusters(D, args)
        pickle.dump(C, open(fC, "wb"))

    if stage == "prep":
        return D, C, None, None

    # Stage 3: STGNN
    runs = stage_stgnn(D, C, args, cache) if stage in ("all", "stgnn") else \
        [pickle.load(open(os.path.join(cache, f), "rb")) for f in sorted(os.listdir(cache)) if f.startswith("run_")]

    if stage == "stgnn":
        return D, C, runs, None

    # Stage 4: anomalies + SPS / Kalman
    fA = os.path.join(cache, anomaly_cache_name(estimator))
    if os.path.exists(fA):
        A = pickle.load(open(fA, "rb"))
    else:
        A = stage_anomaly_sps(D, C, runs, args, estimator=estimator)
        pickle.dump(A, open(fA, "wb"))

    from pv_report import build_report
    build_report(D, C, runs, A, out_dir)
    log("done ->", out_dir)
    return D, C, runs, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=None,
                    help="Path to xlsx/pickle file. If omitted, reads from --df_pickled.")
    ap.add_argument("--df_pickled", default=None,
                    help="Path to pickled DataFrame (alternative to --xlsx).")
    ap.add_argument("--ttl", default="PV-generation-system-metadata.ttl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--train_stride", type=int, default=4)
    ap.add_argument("--ft_days", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=["baseline", "multiview"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--stage", default="all", choices=["all", "prep", "stgnn", "report"])
    ap.add_argument("--estimator", default="sps", choices=["sps", "kalman"],
                    help="b1 estimator for stage 4: Sign-Perturbed Sums or Kalman filter")
    args = ap.parse_args()
    cache = os.path.join(args.out, "cache")
    os.makedirs(cache, exist_ok=True)
    fD, fC = os.path.join(cache, "data.pkl"), os.path.join(cache, "clusters.pkl")
    if os.path.exists(fD):
        D = pickle.load(open(fD, "rb"))
    else:
        # Load DataFrame from file, then pass to stage_data
        if args.xlsx:
            raw = pd.read_excel(args.xlsx, sheet_name="Data") if args.xlsx.endswith("xlsx") else pd.read_pickle(args.xlsx)
        elif args.df_pickled:
            raw = pd.read_pickle(args.df_pickled)
        else:
            ap.error("Provide --xlsx or --df_pickled to load data")
        D = stage_data(raw, args.ttl); pickle.dump(D, open(fD, "wb"))
    if os.path.exists(fC):
        C = pickle.load(open(fC, "rb"))
    else:
        C = stage_clusters(D, args); pickle.dump(C, open(fC, "wb"))
    if args.stage == "prep":
        return
    runs = stage_stgnn(D, C, args, cache) if args.stage in ("all", "stgnn") else \
        [pickle.load(open(os.path.join(cache, f), "rb")) for f in sorted(os.listdir(cache)) if f.startswith("run_")]
    if args.stage == "stgnn":
        return
    fA = os.path.join(cache, anomaly_cache_name(args.estimator))
    if os.path.exists(fA):
        A = pickle.load(open(fA, "rb"))
    else:
        A = stage_anomaly_sps(D, C, runs, args, estimator=args.estimator); pickle.dump(A, open(fA, "wb"))
    from pv_report import build_report
    build_report(D, C, runs, A, args.out)
    log("done ->", args.out)


#if __name__ == "__main__":
#    main()
