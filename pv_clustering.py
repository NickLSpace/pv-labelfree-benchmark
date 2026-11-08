"""
Multi-view clustering for the PV fleet.

| Grouping              | Objects              | Changes in time | Role                                   |
|-----------------------|----------------------|-----------------|----------------------------------------|
| Technical clusters    | PV installations     | rarely          | comparable peer nodes (edge prior)     |
| Behavioural clusters  | PV installations     | possibly        | baseline, cold start, thresholds       |
| Weather regimes       | time windows w_t     | yes             | forecast / detection context           |
| Anomaly clusters      | deviating windows    | yes             | interpretation of fault types          |

    metadata (coords, rated power, tilt, azimuth, panel, inverter)
        -> technical clusters -> prior for peer edges + cold-start baseline
    real TRAIN profiles, dP/dGHI, P/P_expected
        -> behavioural similarity matrix -> additional behavioural peer edges
        -> dynamic supergraph -> STGNN  <- current weather regime (context feature,
                                            regime-specific threshold)

All behavioural features are computed ONLY on the normal part of the training
period (no leakage from the test period):  dP/dGHI, median P/P_exp, normalised
daily profile, temperature sensitivity, variability of the physics residual,
median efficiency eta, correlation with neighbouring plants.

Cluster quality: Silhouette (primary), Davies-Bouldin (secondary),
Calinski-Harabasz (additional); stability as ARI over five seeds.

References: Rousseeuw (1987) J. Comput. Appl. Math. 20:53-65 (silhouette);
Davies & Bouldin (1979) IEEE TPAMI 1:224-227; Calinski & Harabasz (1974)
Commun. Stat. 3:1-27; Hubert & Arabie (1985) J. Classif. 2:193-218 (ARI);
Bickel & Scheffer (2004) ICDM "Multi-view clustering".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

from pv_cohorts import short


# ----------------------------------------------------------------------------- generic
@dataclass
class ClusterResult:
    view: str
    labels: pd.Series                       # index = object id
    k: int
    algorithm: str
    features: pd.DataFrame                  # standardised features actually clustered
    scores: pd.DataFrame                    # one row per (algorithm, k)
    centers: Optional[pd.DataFrame] = None
    model: object = None
    scaler: object = None
    stability_ari: float = np.nan
    extra: Dict[str, object] = field(default_factory=dict)

    def metrics_row(self) -> dict:
        r = self.scores[(self.scores.algorithm == self.algorithm) & (self.scores.k == self.k)].iloc[0]
        return dict(view=self.view, algorithm=self.algorithm, k=self.k, n_objects=len(self.labels),
                    silhouette=r.silhouette, davies_bouldin=r.davies_bouldin,
                    calinski_harabasz=r.calinski_harabasz, stability_ARI_5seeds=self.stability_ari)


def _scores(X: np.ndarray, lab: np.ndarray, sample: Optional[int] = None, seed: int = 0):
    if len(np.unique(lab)) < 2:
        return np.nan, np.nan, np.nan
    if sample and len(X) > sample:
        rng = np.random.default_rng(seed)
        i = rng.choice(len(X), sample, replace=False)
        sil = silhouette_score(X[i], lab[i])
    else:
        sil = silhouette_score(X, lab)
    return float(sil), float(davies_bouldin_score(X, lab)), float(calinski_harabasz_score(X, lab))


def select_clustering(X: np.ndarray, ks: Sequence[int], algorithms=("kmeans", "agglomerative"),
                      seed: int = 0, sample: Optional[int] = None, min_cluster: int = 2):
    """Grid over algorithms x k; choose max silhouette, ties -> lower Davies-Bouldin."""
    rows, fitted = [], {}
    n = len(X)
    for alg in algorithms:
        if alg == "agglomerative" and n > 6000:
            continue
        for k in ks:
            if k >= n:
                continue
            if alg == "kmeans":
                m = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
                lab = m.labels_
            else:
                m = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
                lab = m.labels_
            sizes = np.bincount(lab)
            sil, db, ch = _scores(X, lab, sample, seed)
            rows.append(dict(algorithm=alg, k=k, silhouette=sil, davies_bouldin=db,
                             calinski_harabasz=ch, min_size=int(sizes.min())))
            fitted[(alg, k)] = (m, lab)
    sc = pd.DataFrame(rows)
    ok = sc[sc.min_size >= min_cluster]
    ok = ok if len(ok) else sc
    best = ok.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
    m, lab = fitted[(best.algorithm, int(best.k))]
    return sc, best.algorithm, int(best.k), m, lab


def stability_ari(X: np.ndarray, k: int, seeds=(0, 1, 2, 3, 4)) -> float:
    labs = [KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(X) for s in seeds]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return float(np.mean(aris)) if aris else np.nan


def _centroids(X: np.ndarray, lab: np.ndarray) -> np.ndarray:
    return np.vstack([X[lab == c].mean(0) for c in np.unique(lab)])


# ----------------------------------------------------------------------------- 1. technical
def technical_features(brick, mapping: Dict[str, str], pv_cols: Sequence[str]) -> pd.DataFrame:
    rows = {}
    for c in pv_cols:
        s = brick.sites[mapping[c]]
        az = np.deg2rad(s.azimuth_deg or 180.0)
        rows[c] = dict(lat=s.lat, lon=s.lon, log_rated_kw=np.log1p(s.rated_kw or 0.0),
                       tilt=s.tilt_deg, az_sin=np.sin(az), az_cos=np.cos(az),
                       multi_orientation=float(s.multi_orientation),
                       n_inverters=float(s.n_inverters), module_eff=s.module_eff,
                       inverter_brand=s.inverter_brand or "NA", panel_brand=s.panel_brand or "NA",
                       panel_type=s.panel_type or "NA",
                       flexible=float("flexible" in c.lower()))
    f = pd.DataFrame(rows).T
    num = f[["lat", "lon", "log_rated_kw", "tilt", "az_sin", "az_cos", "multi_orientation",
             "n_inverters", "module_eff", "flexible"]].astype(float)
    num = num.fillna(num.median())
    cat = pd.get_dummies(f[["inverter_brand", "panel_brand", "panel_type"]].astype(str), dtype=float)
    return pd.concat([num, cat], axis=1)


def technical_clusters(brick, mapping, pv_cols, ks=range(2, 11), seed=0, k_min: int = 3,
                       min_cluster: int = 3) -> ClusterResult:
    """k is chosen by silhouette among k >= ``k_min`` (k = 2 only separates the
    flexible thin-film arrays from the rest, which is useless as a peer prior);
    the full grid, including k = 2, is kept in ``scores`` for reporting."""
    F = technical_features(brick, mapping, pv_cols)
    sc = StandardScaler().fit(F.to_numpy(float))
    X = sc.transform(F.to_numpy(float))
    # one-hot blocks are down-weighted so each categorical attribute counts as ~1 feature
    w = np.ones(X.shape[1])
    for pre in ("inverter_brand_", "panel_brand_", "panel_type_"):
        idx = [i for i, c in enumerate(F.columns) if c.startswith(pre)]
        if idx:
            w[idx] = 1.0 / np.sqrt(len(idx))
    X = X * w
    scores, _, _, _, _ = select_clustering(X, ks, seed=seed, min_cluster=1)
    _, alg, k, m, lab = select_clustering(X, [k for k in ks if k >= k_min], seed=seed, min_cluster=min_cluster)
    res = ClusterResult("technical", pd.Series(lab, index=list(pv_cols), name="tech_cluster"), k, alg,
                        pd.DataFrame(X, index=list(pv_cols), columns=F.columns), scores,
                        centers=pd.DataFrame(_centroids(X, lab), columns=F.columns), model=m, scaler=sc)
    res.stability_ari = stability_ari(X, k)
    res.extra["raw_features"] = F
    return res


def peer_prior_from_clusters(labels: pd.Series, across: float = 0.5) -> pd.DataFrame:
    """Multiplicative prior for peer edges: 1 within a technical cluster, ``across`` otherwise."""
    L = labels.to_numpy()
    M = np.where(L[:, None] == L[None, :], 1.0, across)
    return pd.DataFrame(M, index=labels.index, columns=labels.index)


# ----------------------------------------------------------------------------- 2. behavioural
def _neighbours(brick, mapping, pv_cols, k=5):
    from pv_physics_graph import PVGraphBuilder
    n = len(pv_cols)
    D = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = PVGraphBuilder._haversine_km(brick.sites[mapping[pv_cols[i]]],
                                                       brick.sites[mapping[pv_cols[j]]])
    # ties at identical coordinates are frequent -> break by index for determinism
    return {pv_cols[i]: [pv_cols[j] for j in np.lexsort((np.arange(n), D[i]))[:k]] for i in range(n)}


def behavioural_features(df: pd.DataFrame, phys: Dict[str, np.ndarray], pv_cols: Sequence[str],
                         rated_kw: Dict[str, float], valid: pd.DataFrame, train_end,
                         brick=None, mapping=None, t_cell: Optional[np.ndarray] = None,
                         poa: Optional[np.ndarray] = None, warmup_days: int = 30,
                         mad_k: float = 4.0) -> pd.DataFrame:
    """Behavioural descriptors from the NORMAL training data only.

    For plants that have no training data (commissioned after ``train_end``) the
    features are computed on their first ``warmup_days`` of valid operation (causal
    warm-up) and flagged in column ``warmup``.
    """
    train_end = pd.Timestamp(train_end)
    ghi = phys["ghi"]
    pe = phys["p_expected"]
    hours = df.index.hour.to_numpy()
    tr = df.index < train_end
    rows = {}
    neigh = _neighbours(brick, mapping, list(pv_cols)) if brick is not None else {}
    ratio_all = {}
    for j, c in enumerate(pv_cols):
        P = df[c].to_numpy(float)
        v = valid[c].to_numpy(bool)
        m = v & tr
        warm = False
        if m.sum() < 200:                                  # new plant -> causal warm-up window
            first = df.index[v].min() if v.any() else None
            if first is None:
                rows[c] = dict(warmup=True); continue
            m = v & (df.index >= first) & (df.index < first + pd.Timedelta(days=warmup_days))
            warm = True
        kw = max(rated_kw.get(c) or 1.0, 1e-3)
        r = P[m] / np.maximum(pe[m, j], 1e-6)
        med, mad = np.median(r), np.median(np.abs(r - np.median(r))) + 1e-9
        normal = np.abs(r - med) < mad_k * 1.4826 * mad         # exclude outages / spikes
        mm = np.flatnonzero(m)[normal]
        r = r[normal]
        ratio_all[c] = pd.Series(r, index=df.index[mm])
        y = P[mm] / (kw * 1000.0)
        g = ghi[mm]
        slope = np.polyfit(g, y, 1)[0] * 1000.0 if len(g) > 10 else np.nan  # (P/Prated) per kW/m2
        # temperature sensitivity of the physics-normalised output (per K)
        if t_cell is not None and len(mm) > 10:
            tc = t_cell[:, j] if np.ndim(t_cell) == 2 else t_cell
            tsens = np.polyfit(tc[mm] - 25.0, r, 1)[0]
        else:
            tsens = np.nan
        eta = (P[mm] / (kw * 1000.0 * np.maximum(poa[mm, j], 1.0) / 1000.0)) if poa is not None else r
        # normalised daily profile (all valid daylight hours, not only GHI>200)
        mp = (df.index < train_end) if not warm else (df.index >= df.index[mm].min()) & \
             (df.index < df.index[mm].min() + pd.Timedelta(days=warmup_days))
        prof = pd.Series(P[mp & (ghi > 20)], index=hours[mp & (ghi > 20)]).groupby(level=0).mean()
        prof = prof.reindex(range(6, 20)).fillna(0.0)
        prof = prof / max(prof.max(), 1e-9)
        hrs = np.arange(6, 20)
        cen = float((prof * hrs).sum() / max(prof.sum(), 1e-9))
        width = float(np.sqrt((prof * (hrs - cen) ** 2).sum() / max(prof.sum(), 1e-9)))
        asym = float(prof[prof.index < 12].sum() - prof[prof.index > 12].sum()) / max(prof.sum(), 1e-9)
        rows[c] = dict(dPdGHI_norm=slope, median_P_over_Pexp=float(np.median(r)),
                       profile_centroid_h=cen, profile_width_h=width, profile_asymmetry=asym,
                       temp_sensitivity_per_K=tsens,
                       residual_iqr=float(np.subtract(*np.quantile(r, [.75, .25]))),
                       median_eta=float(np.median(eta)), warmup=warm, n_hours=int(len(mm)))
        for h in range(6, 20):
            rows[c][f"prof_{h:02d}"] = float(prof.get(h, 0.0))
    F = pd.DataFrame(rows).T
    # correlation with the k nearest neighbours (hourly P/P_exp, overlapping hours)
    corr = {}
    for c in pv_cols:
        if c not in ratio_all:
            corr[c] = np.nan; continue
        vals = []
        for nb in neigh.get(c, []):
            if nb in ratio_all:
                a, b = ratio_all[c].align(ratio_all[nb], join="inner")
                if len(a) > 50:
                    vals.append(np.corrcoef(a, b)[0, 1])
        corr[c] = float(np.nanmean(vals)) if vals else np.nan
    F["neighbour_corr"] = pd.Series(corr)
    return F


BEHAV_COLS = ["dPdGHI_norm", "median_P_over_Pexp", "profile_centroid_h", "profile_width_h",
              "profile_asymmetry", "temp_sensitivity_per_K", "residual_iqr", "median_eta",
              "neighbour_corr"]


def behavioural_clusters(F: pd.DataFrame, ks=range(2, 9), seed=0) -> ClusterResult:
    """Fit on plants with TRAIN data; warm-up (new) plants are assigned to the nearest
    centroid (cold start)."""
    F = F.copy()
    cols = [c for c in BEHAV_COLS if c in F]
    X_all = F[cols].astype(float)
    X_all = X_all.fillna(X_all.median())
    fit_idx = F.index[~F["warmup"].astype(bool)]
    sc = StandardScaler().fit(X_all.loc[fit_idx].to_numpy(float))
    Xf = sc.transform(X_all.loc[fit_idx].to_numpy(float))
    scores, alg, k, m, lab = select_clustering(Xf, ks, seed=seed)
    cen = _centroids(Xf, lab)
    Xa = sc.transform(X_all.to_numpy(float))
    d = ((Xa[:, None, :] - cen[None]) ** 2).sum(-1)
    lab_all = pd.Series(np.argmin(d, 1), index=F.index, name="behav_cluster")
    lab_all.loc[fit_idx] = lab
    res = ClusterResult("behavioural", lab_all, k, alg, pd.DataFrame(Xa, index=F.index, columns=cols),
                        scores, centers=pd.DataFrame(cen, columns=cols), model=m, scaler=sc)
    res.stability_ari = stability_ari(Xf, k)
    res.extra["fit_index"] = list(fit_idx)
    return res


def behavioural_similarity(res: ClusterResult, gamma: Optional[float] = None) -> pd.DataFrame:
    X = res.features.to_numpy(float)
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    g = gamma or 1.0 / max(np.median(d2[d2 > 0]), 1e-9)
    S = np.exp(-g * d2)
    np.fill_diagonal(S, 0.0)
    return pd.DataFrame(S, index=res.features.index, columns=res.features.index)


def behavioural_edges(S: pd.DataFrame, k: int = 3, existing: Optional[set] = None,
                      min_sim: float = 0.3, edge_type: int = 3) -> List[Tuple[str, str, float, int]]:
    """Top-k behavioural neighbours (directed src -> dst), skipping pairs already
    connected by a peer edge."""
    out = []
    existing = existing or set()
    for dst in S.index:
        for src in S.loc[dst].sort_values(ascending=False).index[:k]:
            w = float(S.loc[dst, src])
            if w >= min_sim and (src, dst) not in existing:
                out.append((src, dst, w, edge_type))
    return out


def cold_start_baseline(F: pd.DataFrame, tech: pd.Series, col: str = "median_P_over_Pexp") -> pd.Series:
    """Baseline P/P_exp for every plant from the median of its technical cluster
    (computed on plants with training data only)."""
    fit = F.index[~F["warmup"].astype(bool)]
    med = F.loc[fit, col].astype(float).groupby(tech.loc[fit]).median()
    return tech.map(med).rename("cold_start_baseline_ratio")


# ----------------------------------------------------------------------------- 3. weather regimes
def weather_features(df: pd.DataFrame, irr_col="Irradiance (W/m2)", temp_col="Temp (Degree Celsius)",
                     wind_col="wind_speed", kt: Optional[pd.Series] = None, roll: int = 3) -> pd.DataFrame:
    """w_t = [GHI, dGHI, sigma(GHI), T, wind, cloudiness, RH, visibility, rain] with CAUSAL
    rolling statistics (past ``roll`` hours only)."""
    g = df[irr_col]
    out = pd.DataFrame(index=df.index)
    out["GHI"] = g
    out["dGHI"] = g.diff().fillna(0.0)
    out["sigma_GHI"] = g.rolling(roll, min_periods=1).std().fillna(0.0)
    out["T"] = df[temp_col]
    out["wind"] = df[wind_col]
    if kt is not None:
        out["cloudiness"] = (1.0 - kt.clip(0, 1.2)).fillna(1.0)
    for c, n in (("RH (%)", "RH"), ("Vis (km)", "Vis"), ("rainfall_log", "rain")):
        if c in df:
            out[n] = df[c]
    return out


@dataclass
class WeatherRegimes:
    result: ClusterResult
    regime: pd.Series                 # 0 = night, 1..K = daylight regimes
    onehot: np.ndarray                # (T, K+1)
    profile: pd.DataFrame             # mean raw feature per regime
    names: Dict[int, str]


def weather_regimes(W: pd.DataFrame, train_end, ks=range(2, 8), day_thr: float = 20.0,
                    seed: int = 0, sample: int = 4000) -> WeatherRegimes:
    train_end = pd.Timestamp(train_end)
    day = W["GHI"] > day_thr
    fit = day & (W.index < train_end)
    sc = StandardScaler().fit(W.loc[fit].to_numpy(float))
    Xf = sc.transform(W.loc[fit].to_numpy(float))
    scores, alg, k, m, lab = select_clustering(Xf, ks, algorithms=("kmeans",), seed=seed, sample=sample)
    reg = pd.Series(0, index=W.index, name="weather_regime")
    Xd = sc.transform(W.loc[day].to_numpy(float))
    reg.loc[day] = m.predict(Xd) + 1
    # order daylight regimes by mean cloudiness / GHI so labels are interpretable
    prof = W.groupby(reg).mean()
    key = prof.drop(index=0)["GHI"].sort_values(ascending=False).index.tolist()
    remap = {old: new + 1 for new, old in enumerate(key)}
    remap[0] = 0
    reg = reg.map(remap)
    prof = W.groupby(reg).mean()
    prof["share_train"] = reg[W.index < train_end].value_counts(normalize=True).reindex(prof.index)
    names = {0: "night"}
    for r in prof.index[1:]:
        p = prof.loc[r]
        if ("rain" in p and p["rain"] > 0.3) or ("RH" in p and p["RH"] > 92):
            nm = "rain / low visibility"
        elif "cloudiness" in p and p["cloudiness"] < 0.6:
            nm = "clear / high irradiance"
        elif p["sigma_GHI"] > prof.loc[prof.index != 0, "sigma_GHI"].median() and p.get("cloudiness", 1) < 0.85:
            nm = "variable cloudiness"
        else:
            nm = "cloudy"
        names[int(r)] = f"R{int(r)}: {nm}"
    onehot = np.eye(k + 1, dtype=np.float32)[reg.to_numpy()]
    res = ClusterResult("weather_regimes", reg, k, alg, pd.DataFrame(Xf), scores,
                        centers=pd.DataFrame(m.cluster_centers_, columns=W.columns), model=m, scaler=sc)
    # stability on a subsample
    rng = np.random.default_rng(seed)
    sub = Xf[rng.choice(len(Xf), min(len(Xf), sample), replace=False)]
    res.stability_ari = stability_ari(sub, k)
    return WeatherRegimes(res, reg, onehot, prof, names)


# ----------------------------------------------------------------------------- 4. anomaly clusters
ANOMALY_FEATURES = ["duration_h", "mean_ratio", "zero_frac", "mean_ghi", "mean_hour",
                    "hour_spread", "peak_score", "mean_physics_z", "flatness", "clear_share"]


def anomaly_event_features(events: pd.DataFrame, df: pd.DataFrame, phys: Dict[str, np.ndarray],
                           pv_cols: Sequence[str], regimes: Optional[pd.Series] = None) -> pd.DataFrame:
    """Descriptors of every detected event (contiguous above-threshold windows)."""
    rows = []
    col_of = {short(c): c for c in pv_cols}
    for i, e in events.iterrows():
        c = col_of.get(e["plant"])
        if c is None:
            continue
        j = list(pv_cols).index(c)
        sl = (df.index >= e["start"]) & (df.index <= e["end"])
        P = df.loc[sl, c].to_numpy(float)
        pe = phys["p_expected"][sl, j]
        g = phys["ghi"][sl]
        day = g > 50
        ratio = np.median(P[day] / np.maximum(pe[day], 1.0)) if day.any() else np.nan
        hrs = df.index[sl].hour.to_numpy()
        pmax = np.nanquantile(df[c].to_numpy(float), 0.99) or 1.0
        flat = float(np.mean(P[day] > 0.95 * pmax)) if day.any() else 0.0
        clear = float(np.mean(regimes[sl].to_numpy() == 1)) if regimes is not None else np.nan
        rows.append(dict(event_id=i, plant=e["plant"], start=e["start"], end=e["end"],
                         duration_h=float(sl.sum()), mean_ratio=float(np.clip(ratio, 0, 3)) if np.isfinite(ratio) else 1.0,
                         zero_frac=float(np.mean(P[day] <= 0)) if day.any() else 0.0,
                         mean_ghi=float(g.mean()), mean_hour=float(hrs.mean()),
                         hour_spread=float(hrs.std()), peak_score=float(e["peak_score"]),
                         mean_physics_z=float(np.nan_to_num(e.get("mean_physics_z", 0.0))),
                         flatness=flat, clear_share=clear,
                         **event_shape_features(df, phys, c, j, e["start"], e["end"])))
    return pd.DataFrame(rows)


# Shape descriptors used by the CALIBRATED fuzzy rule base (pv_fuzzy rule_set="calibrated").
# They are computed on the full calendar days touched by the event (the detected interval is
# often only the strongest hours) and normalised by the plant's own causal baseline (the
# ``baseline_days`` before the event), so that a plant with PR ~ 0.8 is not "below" all the time
# and a physics-model bias at low irradiance cancels out.
SHAPE_FEATURES = ["rel_ratio", "edge_gap", "clip_score", "lowlight_gap"]


def event_shape_features(df: pd.DataFrame, phys: Dict[str, np.ndarray], c: str, j: int,
                         start, end, baseline_days: int = 30, ghi_min: float = 50.0,
                         ghi_bright: float = 250.0, mid_hours=(10, 15)) -> Dict[str, float]:
    """rel_ratio    : median(P/P_exp) on the event hours / the same median in the baseline period
                      (GHI > ghi_min, P > 0).  1 = plant's usual level; partial loss ~ 1 - loss.
    edge_gap       : (median ratio at midday - median ratio at morning/evening) / baseline, on the
                      event days.  Morning / evening shading -> positive; uniform loss -> ~ 0.
    clip_score     : share of productive hours (P >= 0.5 cap) where the expected power, scaled by
                      the median ratio of the event days, exceeds the cap by >= 5 % while P stays
                      within 3 % of the cap (flat top that the physics does not explain; a uniform
                      loss cancels in the scaling).  cap = max P on the event days.
    lowlight_gap   : relative ratio in bright hours (GHI >= ghi_bright) minus relative ratio in dim
                      hours (ghi_min < GHI < ghi_bright), each normalised by its own baseline bin.
                      Deficit only under low light -> positive (weather-driven); uniform loss -> ~ 0."""
    idx = df.index
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    i0, i1 = idx.searchsorted(t0), idx.searchsorted(t1, side="right")
    d0 = idx.searchsorted(t0.normalize())
    d1 = idx.searchsorted(t1.normalize() + pd.Timedelta(days=1))
    b0 = idx.searchsorted(t0 - pd.Timedelta(days=baseline_days))
    P_all = df[c].to_numpy(float)
    pe_all = phys["p_expected"][:, j]
    g_all = phys["ghi"]
    hrs_all = idx.hour.to_numpy()

    def _sel(a, b):
        P, pe, g = P_all[a:b], pe_all[a:b], g_all[a:b]
        m = (g > ghi_min) & (pe > 1.0) & np.isfinite(P)
        return dict(r=P[m] / np.maximum(pe[m], 1.0), h=hrs_all[a:b][m], P=P[m], pe=pe[m], g=g[m])

    def _med(x, m=None, n=5):
        v = x if m is None else x[m]
        return float(np.median(v)) if v.size >= n else np.nan

    B = _sel(b0, d0)
    pos = B["P"] > 0
    if pos.sum() < 10:                                  # no usable history: whole series
        B = _sel(0, len(idx)); pos = B["P"] > 0
    base = _med(B["r"], pos, 1)
    base = base if np.isfinite(base) and base > 0 else 1.0
    base_dim = _med(B["r"], pos & (B["g"] < ghi_bright)); base_dim = base_dim if np.isfinite(base_dim) and base_dim > 0 else base
    base_br = _med(B["r"], pos & (B["g"] >= ghi_bright)); base_br = base_br if np.isfinite(base_br) and base_br > 0 else base
    E = _sel(i0, i1)
    ev = _med(E["r"], None, 1)
    rel = ev / base if np.isfinite(ev) else 1.0
    Dd = _sel(d0, d1)
    gap, clip, llg = 0.0, 0.0, 0.0
    if Dd["r"].size >= 4:
        mid = (Dd["h"] >= mid_hours[0]) & (Dd["h"] < mid_hours[1])
        if mid.any() and (~mid).any():
            gap = (np.median(Dd["r"][mid]) - np.median(Dd["r"][~mid])) / base
        cap = float(np.max(Dd["P"]))
        if cap > 0:
            prod = Dd["P"] >= 0.5 * cap
            lvl = float(np.median(Dd["r"]))           # own level on the event days (uniform loss cancels)
            hit = (Dd["pe"] * lvl >= 1.05 * cap) & (Dd["P"] >= 0.97 * cap)
            clip = float(hit[prod].mean()) if prod.any() else 0.0
        dim = _med(Dd["r"], Dd["g"] < ghi_bright, 2)
        br = _med(Dd["r"], Dd["g"] >= ghi_bright, 2)
        if np.isfinite(dim):
            llg = (br / base_br if np.isfinite(br) else 1.0) - dim / base_dim
    return dict(rel_ratio=float(np.clip(rel, 0, 3)), edge_gap=float(np.clip(gap, -1, 1)),
                clip_score=float(clip), lowlight_gap=float(np.clip(llg, -1, 1)))


ANOMALY_CLASSES = ["Outage / zero generation", "Power limitation (clipping / curtailment)",
                   "Partial shading (time of day)", "Gradual loss (soiling / degradation)",
                   "Overgeneration / data spike (P > P_exp)", "Weather-driven deviation (low irradiance)"]


def _label_center(c: pd.Series) -> str:
    if c["zero_frac"] > 0.5:
        return ANOMALY_CLASSES[0]
    if c["mean_ratio"] > 1.25:
        return ANOMALY_CLASSES[4]
    if c["flatness"] > 0.3 and c["mean_ghi"] > 400 and c["mean_ratio"] < 0.95:
        return ANOMALY_CLASSES[1]
    if c["mean_ghi"] < 100 or abs(c["mean_ratio"] - 1.0) < 0.1:
        return ANOMALY_CLASSES[5]              # low light / ratio unchanged -> weather-driven
    if (c["mean_hour"] < 10 or c["mean_hour"] > 14.5) and c["duration_h"] < 12 and c["mean_ratio"] < 0.9:
        return ANOMALY_CLASSES[2]
    if c["mean_ratio"] < 0.95 and c["duration_h"] >= 6:
        return ANOMALY_CLASSES[3]
    return ANOMALY_CLASSES[5]


def anomaly_clusters(feat: pd.DataFrame, ks=range(2, 7), seed=0) -> ClusterResult:
    X0 = feat[ANOMALY_FEATURES].astype(float).fillna(0.0)
    X0["duration_h"] = np.log1p(X0["duration_h"])
    sc = StandardScaler().fit(X0.to_numpy(float))
    X = sc.transform(X0.to_numpy(float))
    scores, alg, k, m, lab = select_clustering(X, [k for k in ks if k < len(X)], algorithms=("kmeans",),
                                               seed=seed, sample=4000)
    raw = feat[ANOMALY_FEATURES].astype(float).fillna(0.0)
    centers_raw = raw.groupby(lab).median()
    names = {int(i): _label_center(r) for i, r in centers_raw.iterrows()}
    res = ClusterResult("anomaly", pd.Series(lab, index=feat.index, name="anomaly_cluster"), k, alg,
                        pd.DataFrame(X, index=feat.index, columns=ANOMALY_FEATURES), scores,
                        centers=centers_raw, model=m, scaler=sc)
    res.stability_ari = stability_ari(X, k)
    res.extra["class_names"] = names
    return res


def most_probable_class(feat: pd.DataFrame, res: ClusterResult, all_plants: Sequence[str]) -> pd.DataFrame:
    """Per plant: duration-weighted posterior share of each anomaly class; the argmax is
    reported as the most probable class (``Normal`` if the plant has no events)."""
    names = res.extra["class_names"]
    f = feat.assign(cls=res.labels.map(names).values)
    tab = f.pivot_table(index="plant", columns="cls", values="duration_h", aggfunc="sum").fillna(0.0)
    share = tab.div(tab.sum(1), axis=0)
    out = pd.DataFrame(index=[short(p) for p in all_plants])
    out["n_events"] = f.groupby("plant").size().reindex(out.index).fillna(0).astype(int)
    out["anomalous_hours"] = f.groupby("plant")["duration_h"].sum().reindex(out.index).fillna(0.0)
    out["most_probable_class"] = share.idxmax(1).reindex(out.index).fillna("Normal")
    out["class_probability"] = share.max(1).reindex(out.index).fillna(1.0)
    for c in share.columns:
        out[f"p[{c}]"] = share[c].reindex(out.index).fillna(0.0)
    return out
