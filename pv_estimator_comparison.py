"""
SPS vs Kalman filter: comparison on the FINAL output of the pipeline.

Both estimators feed the same downstream stages (STGNN scores -> evidence fusion,
plant-level diagnostic clustering, anomaly-event clustering).  Everything except the
estimator of b1 is held fixed (same STGNN runs, same events, same seeds), so the
difference in the metrics is attributable to the estimator.

Three levels of the final result are evaluated:

| level      | objects | partition                                              | features                                        |
|------------|---------|--------------------------------------------------------|-------------------------------------------------|
| evidence   | plants  | rule-based evidence label of ``fuse_evidence``         | STGNN score/exceedance + estimator b1 summary   |
| plants     | plants  | K-Means on the fused plant features (k by silhouette)  | same as above                                   |
| events     | events  | anomaly clusters (K-Means), features + local b1 ratio  | ANOMALY_FEATURES + b1(event)/b1(train)          |

Metrics (same as ``pv_clustering``):
    Silhouette (higher is better), Davies-Bouldin (lower), Calinski-Harabasz (higher),
    ARI — stability of the partition over the STGNN training seeds (mean pairwise ARI;
    for the event level the per-plant most probable class is compared, because events
    differ between seeds).  ARI over K-Means initialisations (``ARI_init``, as
    ``stability_ARI_5seeds`` in pv_clustering) is reported as well, and the agreement
    between the SPS and Kalman partitions (ARI SPS<->Kalman) per level.

Note: Silhouette / DB / CH are internal indices computed in the estimator-specific
feature space; they measure how well separated the final groups are, not which
estimator is 'true'.  ARI over seeds measures robustness of the final result.
"""

from __future__ import annotations

import os
import pickle
from itertools import combinations
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

import pv_clustering as clu
from pv_cohorts import short

ESTIMATORS = ("sps", "kalman")
LABELS = {"sps": "SPS", "kalman": "Kalman"}
LEVEL_LABELS = {"evidence": "Final evidence (plants)",
                "plants": "Diagnostic plant clusters",
                "events": "Anomaly event clusters"}
LEVEL_LABELS_TEX = {"evidence": "Evidence", "plants": "Plant clusters",
                    "events": "Event clusters"}
PLANT_FEATURES = ["stgnn_mean_score", "stgnn_exceed_share", "train_b1", "test_b1", "b1_change_pct",
                  "test_b1_width", "mean_b1_width", "n_regime_changes"]


# ----------------------------------------------------------------------------- estimators
def get_estimator(name: str):
    name = name.lower()
    if name == "sps":
        import pv_sps as m
    elif name in ("kalman", "kf"):
        import pv_kalman as m
    else:
        raise ValueError(f"unknown estimator {name!r}: use 'sps' or 'kalman'")
    return m


def run_estimator(name: str, D: dict, C: dict, boundary: str = "2023-01-01",
                  kalman_cfg=None, sps_kw: Optional[dict] = None) -> Dict[str, object]:
    """periods / split / cluster (+ daily tracks for Kalman) on the data of ``stage_data``."""
    df, phys, pv_cols = D["df"], D["phys"], D["pv_cols"]
    cloud = (1.0 - D["kt"].clip(0, 1.2)).fillna(1.0).to_numpy()
    valid = C["valid"]
    if name == "sps":
        import pv_sps as m
        kw = sps_kw or {}
        per = m.sps_periods(df, phys, pv_cols, D["rated"], valid, D["t_cell"], cloud, freq="QS", **kw)
        split = m.sps_split(df, phys, pv_cols, D["rated"], valid, D["t_cell"], cloud, boundary, **kw)
        clus = m.sps_cluster(df, phys, pv_cols, D["rated"], valid, D["t_cell"], cloud, C["tech"].labels, boundary)
        return dict(estimator="sps", periods=per, split=split, cluster=clus, daily=None)
    import pv_kalman as m
    out = m.kalman_all(df, phys, pv_cols, D["rated"], valid, D["t_cell"], cloud, boundary,
                       C["tech"].labels, cfg=kalman_cfg)
    out.pop("tracks", None)
    out["estimator"] = "kalman"
    return out


def fuse(name: str, est: dict, rank: pd.Series, exceed: pd.Series) -> pd.DataFrame:
    """Evidence table with GENERIC column names (est_*) for either estimator."""
    import pv_sps
    ev = pv_sps.fuse_evidence(rank, exceed, est["split"], est["periods"])
    if name == "kalman":
        ev["evidence"] = ev["evidence"].str.replace("SPS", LABELS["kalman"], regex=False)
    return ev.rename(columns={c: c.replace("sps_", "est_", 1) for c in ev.columns})


# ----------------------------------------------------------------------------- features
def plant_features(est: dict, rank: pd.Series, exceed: pd.Series) -> pd.DataFrame:
    s = est["split"].set_index("plant")
    per = est["periods"]
    F = pd.DataFrame(index=s.index)
    F["stgnn_mean_score"] = rank.reindex(F.index)
    F["stgnn_exceed_share"] = exceed.reindex(F.index)
    F["train_b1"] = s["train_b1"]
    F["test_b1"] = s["test_b1"]
    F["b1_change_pct"] = s["b1_change_pct"]
    F["test_b1_width"] = s["test_b1_hi"] - s["test_b1_lo"]
    if len(per):
        g = per.groupby("plant")
        F["mean_b1_width"] = g["b1_width"].mean().reindex(F.index)
        F["n_regime_changes"] = g["disjoint_from_previous"].sum().reindex(F.index)
    else:
        F["mean_b1_width"] = np.nan; F["n_regime_changes"] = np.nan
    F = F.astype(float)
    # plants without a train period (cohort 3): no change can be estimated -> neutral 0 %
    F["train_b1"] = F["train_b1"].fillna(F["test_b1"])
    F["b1_change_pct"] = F["b1_change_pct"].fillna(0.0)
    return F.fillna(F.median()).fillna(0.0)


def local_b1_ratio(est: dict, feat: pd.DataFrame) -> np.ndarray:
    """b1 at the time of each event divided by the train b1 of the plant.

    Kalman: smoothed daily b1 on the event start day (nearest available day).
    SPS: b1 of the quarter that contains the event start (point estimate of the region)."""
    split = est["split"].set_index("plant")
    tr = split["train_b1"].fillna(split["test_b1"])
    out = np.full(len(feat), np.nan)
    if est.get("daily") is not None and len(est["daily"]):
        dly = est["daily"]
        for p, g in feat.groupby("plant"):
            d = dly[dly.plant == p]
            if not len(d):
                continue
            s = pd.Series(d["b1_Pexp"].to_numpy(), index=pd.DatetimeIndex(d["day"])).sort_index()
            t = pd.DatetimeIndex(pd.to_datetime(g["start"])).normalize()
            pos = np.clip(s.index.searchsorted(t), 0, len(s) - 1)
            out[np.flatnonzero(feat.index.isin(g.index))] = s.to_numpy()[pos]
    else:
        per = est["periods"]
        for p, g in feat.groupby("plant"):
            d = per[per.plant == p]
            if not len(d):
                continue
            s = pd.Series(d["theta_b1_Pexp"].to_numpy(), index=pd.DatetimeIndex(d["period"])).sort_index()
            t = pd.DatetimeIndex(pd.to_datetime(g["start"]))
            pos = np.clip(s.index.searchsorted(t, side="right") - 1, 0, len(s) - 1)
            out[np.flatnonzero(feat.index.isin(g.index))] = s.to_numpy()[pos]
    den = feat["plant"].map(tr).to_numpy(float)
    r = out / np.where(np.abs(den) > 1e-6, den, np.nan)
    return np.nan_to_num(np.clip(r, 0, 3), nan=1.0)


# ----------------------------------------------------------------------------- metrics helpers
def _internal(X: np.ndarray, lab: np.ndarray, sample: int = 4000, seed: int = 0):
    lab = np.asarray(lab)
    if len(np.unique(lab)) < 2 or len(np.unique(lab)) >= len(lab):
        return np.nan, np.nan, np.nan
    if len(X) > sample:
        i = np.random.default_rng(seed).choice(len(X), sample, replace=False)
        sil = silhouette_score(X[i], lab[i]) if len(np.unique(lab[i])) > 1 else np.nan
    else:
        sil = silhouette_score(X, lab)
    return float(sil), float(davies_bouldin_score(X, lab)), float(calinski_harabasz_score(X, lab))


def _mean_pairwise_ari(parts: Sequence[pd.Series]) -> float:
    vals = []
    for a, b in combinations(parts, 2):
        a2, b2 = a.align(b, join="inner")
        if len(a2) > 1:
            vals.append(adjusted_rand_score(a2.astype(str), b2.astype(str)))
    return float(np.mean(vals)) if vals else np.nan


def _agg(rows: Iterable[dict], keys=("silhouette", "davies_bouldin", "calinski_harabasz")) -> dict:
    df = pd.DataFrame(list(rows))
    out = {}
    for k in keys:
        v = df[k].astype(float) if k in df else pd.Series(dtype=float)
        out[k] = float(v.mean()) if v.notna().any() else np.nan
        out[f"{k}_std"] = float(v.std(ddof=0)) if v.notna().sum() > 1 else 0.0
    return out


# ----------------------------------------------------------------------------- per-level evaluation
def base_event_features(D: dict, C: dict, runs: list, variant: str = "multiview") -> Dict[int, pd.DataFrame]:
    """Estimator-independent event descriptors per STGNN seed (computed once, reused)."""
    mv = [r for r in runs if r["variant"] == variant] or list(runs)
    out = {}
    for r in mv:
        f = clu.anomaly_event_features(r["events"], D["df"], D["phys"], D["pv_cols"], C["wr"].regime)
        if len(f):
            out[r["seed"]] = f
    return out


def evaluate_estimator(name: str, est: dict, D: dict, C: dict, runs: list, variant: str = "multiview",
                       ks=range(2, 7), k_plants: Optional[int] = None, seed: int = 0,
                       event_base: Optional[Dict[int, pd.DataFrame]] = None) -> Dict[str, object]:
    df, phys, pv_cols = D["df"], D["phys"], D["pv_cols"]
    mv = [r for r in runs if r["variant"] == variant] or list(runs)
    if not mv:
        raise ValueError("no STGNN runs available")
    seeds = [r["seed"] for r in mv]

    # ---------- plant level: evidence + K-Means on fused features, one partition per STGNN seed
    feats, evid = {}, {}
    for r in mv:
        F = plant_features(est, r["rank"], r["exceed_share"])
        feats[r["seed"]] = F
        evid[r["seed"]] = fuse(name, est, r["rank"], r["exceed_share"])["evidence"].reindex(F.index)
    ref = mv[0]["seed"]
    sc = StandardScaler().fit(feats[ref][PLANT_FEATURES].to_numpy(float))
    Xr = sc.transform(feats[ref][PLANT_FEATURES].to_numpy(float))
    if k_plants is None:
        grid, alg, k, _m, _lab = clu.select_clustering(Xr, [k for k in ks if k < len(Xr)],
                                                       algorithms=("kmeans",), seed=seed)
    else:
        k = int(k_plants)
    rows_ev, rows_pl, part_ev, part_pl = [], [], [], []
    for s_ in seeds:
        X = StandardScaler().fit_transform(feats[s_][PLANT_FEATURES].to_numpy(float))
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        sil, db, ch = _internal(X, lab)
        rows_pl.append(dict(silhouette=sil, davies_bouldin=db, calinski_harabasz=ch))
        part_pl.append(pd.Series(lab, index=feats[s_].index))
        e = evid[s_].astype(str)
        sil, db, ch = _internal(X, pd.factorize(e)[0])
        rows_ev.append(dict(silhouette=sil, davies_bouldin=db, calinski_harabasz=ch))
        part_ev.append(e)
    ari_init_pl = clu.stability_ari(Xr, k)
    plants = dict(level="plants", k=k, n_objects=len(Xr), ari_seeds=_mean_pairwise_ari(part_pl),
                  ari_init=ari_init_pl, **_agg(rows_pl))
    n_ev_cls = int(pd.concat(part_ev).nunique())
    evidence = dict(level="evidence", k=n_ev_cls, n_objects=len(Xr), ari_seeds=_mean_pairwise_ari(part_ev),
                    ari_init=np.nan, **_agg(rows_ev))

    # ---------- event level: anomaly clusters with the estimator's local b1 ratio as extra feature
    fcols = clu.ANOMALY_FEATURES + ["est_b1_ratio"]
    if event_base is None:
        event_base = base_event_features(D, C, mv, variant)
    ev_feat = {}
    for s_, f in event_base.items():
        f = f.copy(); f["est_b1_ratio"] = local_b1_ratio(est, f)
        ev_feat[s_] = f
    events, part_cls, ev_labels = None, [], {}
    if ref in ev_feat and len(ev_feat[ref]) >= 10:
        f0 = ev_feat[ref]
        X0 = f0[fcols].astype(float).fillna(0.0); X0["duration_h"] = np.log1p(X0["duration_h"])
        scE = StandardScaler().fit(X0.to_numpy(float))
        XE = scE.transform(X0.to_numpy(float))
        _g, algE, kE, mE, labE = clu.select_clustering(XE, [k for k in ks if k < len(XE)],
                                                       algorithms=("kmeans",), seed=seed, sample=4000)
        names = {int(i): clu._label_center(c) for i, c in f0[clu.ANOMALY_FEATURES].astype(float)
                 .fillna(0.0).groupby(labE).median().iterrows()}
        rows_e = []
        for s_, f in ev_feat.items():
            Xs = f[fcols].astype(float).fillna(0.0); Xs["duration_h"] = np.log1p(Xs["duration_h"])
            Xs = scE.transform(Xs.to_numpy(float))
            lab = mE.predict(Xs)
            ev_labels[s_] = pd.Series(lab, index=f.index)
            sil, db, ch = _internal(Xs, lab, seed=seed)
            rows_e.append(dict(silhouette=sil, davies_bouldin=db, calinski_harabasz=ch))
            tmp = clu.ClusterResult("anomaly", pd.Series(lab, index=f.index), kE, algE, pd.DataFrame(Xs),
                                    pd.DataFrame(), extra=dict(class_names=names))
            part_cls.append(clu.most_probable_class(f, tmp, pv_cols)["most_probable_class"])
        events = dict(level="events", k=kE, n_objects=len(XE), ari_seeds=_mean_pairwise_ari(part_cls),
                      ari_init=clu.stability_ari(XE[:4000], kE), **_agg(rows_e))
    levels = [evidence, plants] + ([events] if events else [])
    table = pd.DataFrame(levels)
    table.insert(0, "estimator", name)
    return dict(table=table, partitions=dict(evidence=part_ev, plants=part_pl, events=part_cls),
                plant_features=feats, evidence=evid, event_features=ev_feat, event_labels=ev_labels,
                seeds=seeds)


def compare_estimators(D: dict, C: dict, runs: list, estimators: Sequence[str] = ESTIMATORS,
                       precomputed: Optional[Dict[str, dict]] = None, cache_dir: Optional[str] = "results/cache",
                       variant: str = "multiview", k_plants: Optional[int] = None, kalman_cfg=None,
                       boundary: str = "2023-01-01", verbose: bool = True) -> Dict[str, object]:
    """Run every estimator (or reuse cached / precomputed output) and compare final results.

    Returns dict(table=long metric table, wide=paper-style table, cross_ari=..., raw=...)."""
    precomputed = dict(precomputed or {})
    raw, tables = {}, []
    event_base = base_event_features(D, C, runs, variant)
    for name in estimators:
        est = precomputed.get(name)
        f = os.path.join(cache_dir, f"estimator_{name}.pkl") if cache_dir else None
        if est is None and f and os.path.exists(f):
            est = pickle.load(open(f, "rb"))
        if est is None:
            if verbose:
                print(f"[compare] running estimator: {name}")
            est = run_estimator(name, D, C, boundary=boundary, kalman_cfg=kalman_cfg)
            if f:
                os.makedirs(cache_dir, exist_ok=True)
                pickle.dump(est, open(f, "wb"))
        res = evaluate_estimator(name, est, D, C, runs, variant=variant, k_plants=k_plants, event_base=event_base)
        raw[name] = dict(est=est, **res)
        tables.append(res["table"])
    table = pd.concat(tables, ignore_index=True)
    # agreement of the final partitions between estimators (same seeds)
    cross = []
    if len(estimators) == 2:
        a, b = estimators
        for lvl in ("evidence", "plants", "events"):
            pa, pb = raw[a]["partitions"][lvl], raw[b]["partitions"][lvl]
            vals = []
            for x, y in zip(pa, pb):
                x2, y2 = x.align(y, join="inner")
                if len(x2) > 1:
                    vals.append(adjusted_rand_score(x2.astype(str), y2.astype(str)))
            cross.append(dict(level=lvl, ari_between=float(np.mean(vals)) if vals else np.nan))
    cross = pd.DataFrame(cross)
    if len(cross):
        table = table.merge(cross, on="level", how="left")
    return dict(table=table, wide=paper_table(table), cross_ari=cross, raw=raw)


# ----------------------------------------------------------------------------- presentation
METRICS = [("silhouette", "Silhouette", "max"), ("davies_bouldin", "Davies–Bouldin", "min"),
           ("calinski_harabasz", "Calinski–Harabasz", "max"), ("ari_seeds", "ARI (seeds)", "max")]


def paper_table(table: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Readable table (one row per level x estimator) for the notebook."""
    t = table.copy()
    out = pd.DataFrame({"Level": t["level"].map(LEVEL_LABELS), "Estimator": t["estimator"].map(LABELS),
                        "k": t["k"].astype(int), "N": t["n_objects"].astype(int)})
    for key, lab, _ in METRICS:
        out[lab] = t[key].round(digits)
    out["ARI (init)"] = t["ari_init"].round(digits)
    if "ari_between" in t:
        out["ARI SPS↔Kalman"] = t["ari_between"].round(digits)
    return out


def _fmt(v, sd=None, d=3, bold=False):
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:.{d}f}" if abs(v) < 1000 else f"{v:.1f}"
    if sd is not None and np.isfinite(sd) and sd > 0:
        s += f" \\pm {sd:.{d}f}" if abs(v) < 1000 else f" \\pm {sd:.1f}"
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def to_latex(table: pd.DataFrame, path: Optional[str] = None, with_std: bool = True,
             caption: str = "Comparison of SPS and the Kalman filter on the final model output "
                            "(Silhouette $\\uparrow$, Davies--Bouldin $\\downarrow$, "
                            "Calinski--Harabasz $\\uparrow$, ARI $\\uparrow$; mean $\\pm$ std over STGNN seeds)",
             label: str = "tab:sps_vs_kalman", digits: int = 3) -> str:
    """booktabs LaTeX table (table*, full text width in two-column layouts); the best estimator per level
    and metric is in bold."""
    has_cross = "ari_between" in table and table["ari_between"].notna().any()
    cols = "llrr" + "c" * len(METRICS) + ("c" if has_cross else "")
    up, down = "$\\uparrow$", "$\\downarrow$"
    head = ["Level", "Estimator", "$k$", "$N$"] + [
        lab.replace("–", "--") + " " + (up if how == "max" else down) for _, lab, how in METRICS]
    if has_cross:
        head.append("ARI SPS--Kalman")
    lines = ["% requires \\usepackage{booktabs}; Cyrillic: \\usepackage[T2A]{fontenc}, \\usepackage[utf8]{inputenc}, \\usepackage[russian]{babel}",
             "\\begin{table*}[!t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             " & ".join(head) + " \\\\", "\\midrule"]
    levels = [l for l in ("evidence", "plants", "events") if l in set(table["level"])]
    for li, lvl in enumerate(levels):
        g = table[table["level"] == lvl]
        best = {}
        for key, _, how in METRICS:
            v = g[key].astype(float)
            if v.notna().sum() >= 2 and v.nunique() > 1:
                best[key] = v.idxmax() if how == "max" else v.idxmin()
        for ri, (idx, r) in enumerate(g.iterrows()):
            cells = [LEVEL_LABELS_TEX[lvl] if ri == 0 else "", LABELS.get(r["estimator"], r["estimator"]),
                     str(int(r["k"])), str(int(r["n_objects"]))]
            for key, _, _ in METRICS:
                sd = r.get(f"{key}_std") if with_std else None
                cells.append(_fmt(r[key], sd, digits, bold=best.get(key) == idx))
            if has_cross:
                cells.append(_fmt(r["ari_between"], None, digits) if ri == 0 else "")
            lines.append(" & ".join(cells) + " \\\\")
        if li < len(levels) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    tex = "\n".join(lines) + "\n"
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(tex)
    return tex


def download_link(path: str, text: Optional[str] = None):
    """IPython object with a download link for a file (works in Jupyter, Kaggle, Colab)."""
    import base64
    from IPython.display import HTML
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    name = os.path.basename(path)
    text = text or f"Download {name}"
    return HTML(f'<a download="{name}" href="data:text/x-tex;base64,{b64}" target="_blank" '
                f'style="font-weight:bold">{text}</a>')


def plot_comparison(table: pd.DataFrame, path: Optional[str] = None):
    """Grouped bars: metric x level for both estimators."""
    import matplotlib.pyplot as plt
    levels = [l for l in ("evidence", "plants", "events") if l in set(table["level"])]
    ests = list(dict.fromkeys(table["estimator"]))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.8))
    w = 0.8 / max(len(ests), 1)
    colors = {"sps": "#1f5f8b", "kalman": "#c0392b"}
    for ax, (key, lab, how) in zip(axes, METRICS):
        for i, e in enumerate(ests):
            g = table[table.estimator == e].set_index("level").reindex(levels)
            ax.bar(np.arange(len(levels)) + i * w - 0.4 + w / 2, g[key], w,
                   yerr=g.get(f"{key}_std"), capsize=3, color=colors.get(e, None), label=LABELS.get(e, e))
        ax.set_xticks(range(len(levels)))
        short_lv = {"evidence": "evidence", "plants": "plants", "events": "events"}
        ax.set_xticklabels([short_lv[l] for l in levels], fontsize=8)
        ax.set_title(f"{lab} ({'↑' if how == 'max' else '↓'})", fontsize=10)
        ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("SPS vs Kalman filter: metrics of the final output")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig
