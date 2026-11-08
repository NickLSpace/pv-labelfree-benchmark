"""
Publication-ready outputs: xlsx tables and figures (PNG 300 dpi + vector PDF),
including the visualisation of the dynamic supergraph at the GNN input.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
if "ipykernel" not in __import__("sys").modules:   # keep inline plots in notebooks
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

from pv_cohorts import COHORT_NAMES, cohort_summary, short

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})
COH_COL = {1: "#1f5f8b", 2: "#e08e0b", 3: "#3a9a5b", 0: "#9e9e9e"}
FAM = {0: ("topology (Brick)", "#8d99ae"), 1: ("peer (geo + correlation)", "#2a6f97"),
       2: ("G → P (dP/dGHI)", "#e76f51"), 3: ("behavioural peer", "#6a4c93")}
VAR_NAME = {"baseline": "Baseline STGNN", "multiview": "Multi-view STGNN"}
VAR_COL = {"baseline": "#8d99ae", "multiview": "#1f5f8b"}


def _save(fig, out, name):
    d = os.path.join(out, "figures")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================ figures
def fig_coverage(D, out):
    ca = D["ca"]
    order = ca.cohorts.sort_values(["cohort", "plant"]).index
    order = [c for c in order if ca.cohorts.loc[c, "cohort"] > 0] + \
            [c for c in order if ca.cohorts.loc[c, "cohort"] == 0]
    M = ca.cov_m[order].T
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(M.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short(c) for c in order], fontsize=6)
    for t, lab in zip(ax.get_yticklabels(), order):
        t.set_color(COH_COL[int(ca.cohorts.loc[lab, "cohort"])])
    xt = [i for i, t in enumerate(M.columns) if t.month in (1, 7)]
    ax.set_xticks(xt); ax.set_xticklabels([M.columns[i].strftime("%Y-%m") for i in xt], rotation=45)
    cs = ca.cohorts.loc[order, "cohort"].to_numpy()
    for i in np.flatnonzero(np.diff(cs) != 0):
        ax.axhline(i + 0.5, color="white", lw=1.5)
    ax.contour((M.to_numpy(float) >= 0.8).astype(float), levels=[0.5], colors="red", linewidths=0.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("share of valid daytime measurements per month")
    ax.set_title("Data coverage (GHI > 200 W/m², P_exp > 0, no filled gaps, after commissioning);\n"
                 "red frame — 80 % threshold; label colour — cohort")
    handles = [Patch(color=COH_COL[k], label=COHORT_NAMES[k]) for k in (1, 2, 3, 0)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.12, 1.0), frameon=False)
    _save(fig, out, "fig01_coverage_heatmap")


def fig_sensor(D, out):
    s = D["sensor"]; ca, car = D["ca"], D["ca_raw"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    k = s["kt_quantile"]
    ax[0].plot(k.index, k.values, "o-", ms=3, color="#333")
    if pd.notna(s["change_point"]):
        ax[0].axvline(s["change_point"], color="r", ls="--", lw=1)
        ax[0].text(s["change_point"] - pd.Timedelta(days=15), 0.25, f"change point {s['change_point']:%Y-%m}\n"
                   f"shift {s['shift_pct']:+.1f} %, p = {s['p_value']:.1e}\ncoefficient {s['factor_after_cp']:.3f}",
                   va="center", ha="right", fontsize=8, color="r", bbox=dict(fc="white", ec="r", lw=.5))
    for m in s["invalid_months"]:
        ax[0].axvspan(m, m + pd.offsets.MonthEnd(1), color="grey", alpha=.3)
    ax[0].set_ylabel("98th percentile of k_t")
    ax[0].set_title("a) Pyranometer check independent of PV: clearness index k_t = GHI / GHI_extra (grey — sensor failure)")
    f_raw = car.systemic["fleet_median_pr"]; f_h = ca.systemic["fleet_median_pr"]
    ax[1].plot(f_raw.index, f_raw.values, "o-", ms=3, color="#c0392b", label="original GHI")
    ax[1].plot(f_h.index, f_h.values, "o-", ms=3, color="#1f5f8b", label="harmonised GHI")
    ax[1].set_ylabel("median monthly PR of cohort 1")
    ax[1].set_title(f"b) Synchronous 2023 drop: share of plants with a coinciding change point "
                    f"{car.systemic['synchronous_fraction']:.0%} → {ca.systemic['synchronous_fraction']:.0%} after correction")
    ax[1].legend()
    fig.tight_layout()
    _save(fig, out, "fig02_systemic_2023_check")


def fig_pr_cohorts(D, out):
    ca = D["ca"]
    cov = ca.cov_m >= 0.8
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for k in (1, 2, 3):
        cols = ca.cohorts.index[ca.cohorts["cohort"] == k]
        if not len(cols):
            continue
        X = ca.pr_m[cols].where(cov[cols])
        med, lo, hi = X.median(1), X.quantile(.25, axis=1), X.quantile(.75, axis=1)
        ax.plot(med.index, med, "-o", ms=2.5, color=COH_COL[k], label=f"{COHORT_NAMES[k]} (n={len(cols)})")
        ax.fill_between(med.index, lo, hi, color=COH_COL[k], alpha=.15)
    ax.set_ylabel("PR = ΣP_meas / ΣP_exp")
    ax.set_title("Monthly temperature-corrected PR by cohort (median and IQR; covered months only)")
    ax.legend(); fig.tight_layout()
    _save(fig, out, "fig03_pr_by_cohort")


def fig_degradation(D, out):
    deg = D["ca"].degradation
    fig, ax = plt.subplots(1, 2, figsize=(11, 7), gridspec_kw=dict(width_ratios=[1.3, 1]))
    d1 = deg[deg.cohort == 1].sort_values("Rd_yoy_pct_per_yr")
    y = np.arange(len(d1))
    ax[0].errorbar(d1["Rd_yoy_pct_per_yr"], y, xerr=[d1["Rd_yoy_pct_per_yr"] - d1["Rd_yoy_lo"],
                   d1["Rd_yoy_hi"] - d1["Rd_yoy_pct_per_yr"]], fmt="o", ms=3, color=COH_COL[1],
                   ecolor="#9fb7cc", label="YoY (harmonised GHI), 95 % bootstrap")
    ax[0].plot(d1["Rd_yoy_fleetnorm_pct_per_yr"], y, "x", color="#c0392b", ms=4, label="YoY, fleet-normalised")
    ax[0].axvline(0, color="k", lw=.6); ax[0].axvline(-0.5, color="grey", ls=":", lw=.8)
    ax[0].set_yticks(y); ax[0].set_yticklabels(d1["plant"], fontsize=6)
    ax[0].set_xlabel("Rd, %/yr"); ax[0].set_xlim(-30, 30)
    ax[0].set_title("a) Cohort 1: three-year degradation trend")
    ax[0].legend(loc="lower right", fontsize=7)
    d2 = deg[deg.cohort == 2].sort_values("dPR_2023_2022_pct")
    y2 = np.arange(len(d2))
    ax[1].errorbar(d2["dPR_2023_2022_pct"], y2, xerr=[d2["dPR_2023_2022_pct"] - d2["dPR_2023_2022_lo"],
                   d2["dPR_2023_2022_hi"] - d2["dPR_2023_2022_pct"]], fmt="o", ms=3, color=COH_COL[2],
                   ecolor="#f3c98b", label="ΔPR 2023/2022, 95 % bootstrap")
    ax[1].plot(d2["dPR_2023_2022_fleetnorm_pct"], y2, "x", color="#c0392b", ms=4, label="fleet-normalised")
    ax[1].axvline(0, color="k", lw=.6)
    ax[1].set_yticks(y2); ax[1].set_yticklabels(d2["plant"], fontsize=7)
    ax[1].set_xlabel("ΔPR 2023/2022, %  (not long-term degradation)")
    ax[1].set_title("b) Cohort 2: change in 2023 relative to 2022")
    ax[1].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    _save(fig, out, "fig04_degradation_by_cohort")


def fig_cluster_metrics(C, A, out):
    views = [("Technical", C["tech"].scores), ("Behavioural", C["beh"].scores),
             ("Weather regimes", C["wr"].result.scores)]
    if A.get("anom") is not None:
        views.append(("Anomaly clusters", A["anom"].scores))
    fig, ax = plt.subplots(3, len(views), figsize=(3.2 * len(views), 7), sharex="col")
    chosen = {"Technical": C["tech"], "Behavioural": C["beh"], "Weather regimes": C["wr"].result,
              "Anomaly clusters": A.get("anom")}
    for j, (nm, sc) in enumerate(views):
        for i, (met, lab) in enumerate((("silhouette", "Silhouette ↑"), ("davies_bouldin", "Davies–Bouldin ↓"),
                                        ("calinski_harabasz", "Calinski–Harabasz ↑"))):
            for alg, g in sc.groupby("algorithm"):
                ax[i, j].plot(g.k, g[met], "o-", ms=3, label=alg)
            r = chosen[nm]
            ax[i, j].axvline(r.k, color="r", ls=":", lw=1)
            if j == 0:
                ax[i, j].set_ylabel(lab)
        extra = "; k ≥ 3, ≥ 3 objects per cluster" if nm == "Technical" else ""
        ax[0, j].set_title(f"{nm}\n(selected k={chosen[nm].k}, {chosen[nm].algorithm}{extra})", fontsize=9)
        ax[2, j].set_xlabel("k")
    ax[0, 0].legend()
    fig.tight_layout()
    _save(fig, out, "fig05_clustering_metrics")


def fig_clusters(D, C, out):
    tech, beh = C["tech"], C["beh"]
    coh = D["ca"].cohorts["cohort"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    Z = PCA(2, random_state=0).fit_transform(tech.features.to_numpy(float))
    cmap = plt.get_cmap("tab10")
    for c in np.unique(tech.labels):
        m = tech.labels.to_numpy() == c
        ax[0].scatter(Z[m, 0], Z[m, 1], s=30, color=cmap(c), label=f"T{c}", edgecolor="k", lw=.3)
    for i, n in enumerate(tech.labels.index):
        ax[0].annotate(short(n), Z[i], fontsize=4.5, alpha=.7)
    ax[0].set_title(f"a) Technical clusters (PCA of metadata), k={tech.k}"); ax[0].legend(fontsize=7)
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    Zb = PCA(2, random_state=0).fit_transform(beh.features.to_numpy(float))
    warm = C["F"]["warmup"].astype(bool).reindex(beh.labels.index).to_numpy()
    for c in np.unique(beh.labels):
        m = beh.labels.to_numpy() == c
        ax[1].scatter(Zb[m & ~warm, 0], Zb[m & ~warm, 1], s=30, color=cmap(c + 5), label=f"B{c}", edgecolor="k", lw=.3)
        ax[1].scatter(Zb[m & warm, 0], Zb[m & warm, 1], s=40, color=cmap(c + 5), marker="^", edgecolor="k", lw=.5)
    for i, n in enumerate(beh.labels.index):
        ax[1].annotate(short(n), Zb[i], fontsize=4.5, alpha=.7)
    ax[1].scatter([], [], marker="^", color="grey", label="cold start (warm-up)")
    ax[1].set_title(f"b) Behavioural clusters (train), k={beh.k}"); ax[1].legend(fontsize=7)
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    prof_cols = [c for c in C["F"].columns if c.startswith("prof_")]
    hrs = [int(c[5:]) for c in prof_cols]
    for c in np.unique(beh.labels):
        idx = beh.labels.index[beh.labels == c]
        P = C["F"].loc[idx, prof_cols].astype(float)
        ax[2].plot(hrs, P.median(), "-o", ms=3, color=cmap(c + 5), label=f"B{c} (n={len(idx)})")
        ax[2].fill_between(hrs, P.quantile(.25), P.quantile(.75), color=cmap(c + 5), alpha=.15)
    ax[2].set_xlabel("hour (local time)"); ax[2].set_ylabel("P / max P")
    ax[2].set_title("c) Normalised daily profiles (train)"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out, "fig06_technical_behavioural_clusters")


def fig_weather(C, out):
    wr = C["wr"]
    prof = wr.profile.drop(columns=["share_train"], errors="ignore")
    Zp = (prof - prof.mean()) / prof.std()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    im = ax[0].imshow(Zp.to_numpy(float), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax[0].set_yticks(range(len(Zp))); ax[0].set_yticklabels([wr.names[int(i)] for i in Zp.index])
    ax[0].set_xticks(range(Zp.shape[1])); ax[0].set_xticklabels(Zp.columns, rotation=45)
    for i in range(Zp.shape[0]):
        for j in range(Zp.shape[1]):
            ax[0].text(j, i, f"{prof.iloc[i, j]:.0f}" if abs(prof.iloc[i, j]) >= 10 else f"{prof.iloc[i, j]:.2f}",
                       ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax[0], fraction=.03, label="z-score of the mean")
    ax[0].set_title("a) Weather regime profiles w_t (mean values)")
    reg = wr.regime[wr.regime > 0]
    sh = pd.crosstab(reg.index.to_period("M").to_timestamp(), reg, normalize="index")
    bottom = np.zeros(len(sh))
    cm = plt.get_cmap("Set2")
    for i, r in enumerate(sh.columns):
        ax[1].bar(sh.index, sh[r], bottom=bottom, width=25, color=cm(i), label=wr.names[int(r)])
        bottom += sh[r].to_numpy()
    ax[1].set_ylabel("share of daytime hours"); ax[1].set_title("b) Monthly share of regimes")
    import matplotlib.dates as mdates
    ax[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax[1].tick_params(axis="x", rotation=45)
    ax[1].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    _save(fig, out, "fig07_weather_regimes")


def _layout(D, graph, C):
    """Geographic layout: plants of one coordinate group on a circle around the site."""
    brick, mapping = D["brick"], D["mapping"]
    pv = graph.meta["pv_cols"]
    coords = {}
    for c in pv:
        s = brick.sites[mapping[c]]
        coords[c] = (s.lon, s.lat)
    groups: Dict[tuple, List[str]] = {}
    for c, xy in coords.items():
        groups.setdefault(xy, []).append(c)
    lon = np.array([k[0] for k in groups]); lat = np.array([k[1] for k in groups])
    cx, cy = lon.mean(), lat.mean()
    span = max(np.ptp(lon), np.ptp(lat), 1e-3)
    pos = {}
    tech = C["tech"].labels
    for (x, y), members in groups.items():
        members = sorted(members, key=lambda c: (tech.get(c, 0), c))
        r = 0.035 * span * (1 + len(members) / 12)
        for i, c in enumerate(members):
            a = 2 * np.pi * i / max(len(members), 1)
            pos[c] = ((x - cx) / span + r / span * np.cos(a) * 3.2, (y - cy) / span + r / span * np.sin(a) * 3.2)
    ys = [p[1] for p in pos.values()]
    xs = [p[0] for p in pos.values()]
    ex = [n for n in graph.node_names if n not in pv]
    for i, n in enumerate(ex):
        pos[n] = (max(xs) + 0.35, max(ys) - i * (max(ys) - min(ys)) / max(len(ex) - 1, 1))
    return pos


def _circular_layout(D, graph, C):
    """PV nodes on a circle ordered by technical cluster (then cohort), weather nodes inside."""
    pv = graph.meta["pv_cols"]
    tech, coh = C["tech"].labels, D["ca"].cohorts["cohort"]
    order = sorted(pv, key=lambda c: (tech.get(c, 0), coh.get(c, 0), c))
    pos, ang = {}, {}
    n = len(order)
    for i, c in enumerate(order):
        a = np.pi / 2 - 2 * np.pi * i / n
        pos[c] = (np.cos(a), np.sin(a)); ang[c] = a
    ex = [x for x in graph.node_names if x not in pv]
    for i, e in enumerate(ex):
        a = np.pi / 2 - 2 * np.pi * i / max(len(ex), 1)
        pos[e] = (0.28 * np.cos(a), 0.28 * np.sin(a))
    return pos, ang


def fig_graph(D, C, out):
    graph = C["graph_mv"]
    pos, ang = _circular_layout(D, graph, C)
    names = graph.node_names
    pv = set(graph.meta["pv_cols"])
    coh = D["ca"].cohorts["cohort"]
    tech = C["tech"].labels
    rated = D["rated"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    ei, et, ew = graph.edge_index, graph.edge_type, graph.edge_weight
    from matplotlib.collections import LineCollection
    is_pv = np.array([n in pv for n in names])
    self_loop = ei[0] == ei[1]
    fams = {  # key: (mask, colour, lw, alpha, ls, label)
        "exo": ((et == 1) & ~is_pv[ei[0]] & ~self_loop, "#9ad1c9", .35, .35, "-", "weather → PV (type 1)"),
        1: ((et == 1) & is_pv[ei[0]] & ~self_loop, FAM[1][1], .6, .55, "-", "PV–PV peer: geo + correlation (type 1)"),
        2: ((et == 2) & ~self_loop, FAM[2][1], .45, .4, "-", "GHI → PV, weight dP/dGHI (type 2)"),
        3: ((et == 3) & ~self_loop, FAM[3][1], .9, .85, "--", "behavioural peer (type 3)"),
    }
    for key in ("exo", 1, 2, 3):
        m, col, lw, al, ls, _ = fams[key]
        segs = [(pos[names[s_]], pos[names[d_]]) for s_, d_ in zip(ei[0, m], ei[1, m])]
        ax.add_collection(LineCollection(segs, colors=col, linewidths=lw, alpha=al, linestyles=ls))
    cmap = plt.get_cmap("tab10")
    marker = {1: "o", 2: "s", 3: "^", 0: "X"}
    for n in names:
        x, y = pos[n]
        if n in pv:
            ax.scatter(x, y, s=25 + 3 * np.sqrt(rated.get(n, 10) or 10) * 6, color=cmap(int(tech.get(n, 0))),
                       marker=marker[int(coh.get(n, 0))], edgecolor="k", lw=.4, zorder=3)
            a = ang[n]; deg = np.degrees(a)
            flip = np.cos(a) < 0
            ax.text(1.07 * np.cos(a), 1.07 * np.sin(a), short(n), fontsize=7.5,
                    rotation=deg + 180 if flip else deg, rotation_mode="anchor",
                    ha="right" if flip else "left", va="center", zorder=4)
        else:
            ax.scatter(x, y, s=160, color="#2a9d8f", marker="D", edgecolor="k", lw=.5, zorder=5)
            ax.text(x, y - 0.06, n.split(" (")[0], fontsize=8, ha="center", va="top", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=.8, pad=.5))
    # cluster arcs
    pv_order = sorted(graph.meta["pv_cols"], key=lambda c: (tech.get(c, 0), coh.get(c, 0), c))
    for cl in sorted(tech.unique()):
        mem = [c for c in pv_order if tech.get(c) == cl]
        a0, a1 = ang[mem[0]], ang[mem[-1]]
        tt = np.linspace(a0, a1, 50)
        ax.plot(1.035 * np.cos(tt), 1.035 * np.sin(tt), color=cmap(int(cl)), lw=4, solid_capstyle="butt")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.6, 1.5)
    ax.set_aspect("equal"); ax.axis("off")
    h1 = [Line2D([0], [0], color=FAM[0][1], lw=2,
                 label=f"Brick topology (type 0, {int(((et == 0) & ~self_loop).sum())}) — panel b")]
    h1 += [Line2D([0], [0], color=fams[k][1], lw=2, ls=fams[k][4], label=f"{fams[k][5]} ({int(fams[k][0].sum())})")
           for k in ("exo", 1, 2, 3)]
    h2 = [Line2D([0], [0], marker=marker[k], color="w", markerfacecolor="lightgrey", markeredgecolor="k",
                 ms=7, label=COHORT_NAMES[k]) for k in (1, 2, 3, 0)]
    h3 = [Patch(color=cmap(c), label=f"technical cluster T{c}") for c in sorted(tech.unique())]
    h4 = [Line2D([0], [0], marker="D", color="w", markerfacecolor="#2a9d8f", markeredgecolor="k", ms=7,
                 label="weather nodes (GHI, T, RH, wind, SLP, Vis)")]
    ax.legend(handles=h1 + h2 + h3 + h4, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8,
              frameon=False, ncol=3)
    ax.set_title(f"a) Dynamic supergraph at the STGNN input: {len(names)} nodes, {len(et)} edges (incl. self-loops).\n"
                 f"Edges of types 1–3 are shown (topological — in panel b). PV nodes are ordered by technical "
                 f"clusters (arcs), weather nodes in the centre; size ∝ √P_rated", fontsize=10)
    # adjacency by family ordered by technical cluster
    ax = axes[1]
    order = sorted(graph.meta["pv_cols"], key=lambda c: (tech.get(c, 0), coh.get(c, 0), c))
    idx = {n: i for i, n in enumerate(order)}
    A = np.zeros((len(order), len(order)))
    for s, d, t, w in zip(ei[0], ei[1], et, ew):
        a, b = names[s], names[d]
        if a in idx and b in idx and a != b:
            A[idx[b], idx[a]] = max(A[idx[b], idx[a]], {0: 1, 1: 2, 3: 3}.get(int(t), 0))
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(["white", FAM[0][1], FAM[1][1], FAM[3][1]])
    ax.imshow(A, cmap=cm, vmin=-.5, vmax=3.5, interpolation="nearest")
    b = np.flatnonzero(np.diff([tech.get(c, 0) for c in order])) + .5
    for v in b:
        ax.axhline(v, color="k", lw=.5); ax.axvline(v, color="k", lw=.5)
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels([short(c) for c in order], rotation=90, fontsize=6)
    ax.set_yticklabels([short(c) for c in order], fontsize=6)
    ax.set_title("b) PV–PV adjacency matrix (row = receiver), blocks = technical clusters;\n"
                 "colour: topology / peer / behavioural (last one has priority)")
    fig.tight_layout()
    _save(fig, out, "fig08_graph_gnn_input")


def fig_dynamic_graph(D, C, out, dates=("2021-06-15 12:00", "2022-06-15 12:00", "2023-06-15 12:00")):
    graph = C["graph_mv"]
    pos = _layout(D, graph, C)
    names = graph.node_names
    from pv_physics_graph import node_activity_mask
    act = node_activity_mask(D["df"], graph, D["comm"])
    fig, axes = plt.subplots(1, len(dates), figsize=(5.2 * len(dates), 5))
    ei, et = graph.edge_index, graph.edge_type
    from matplotlib.collections import LineCollection
    for ax, dt in zip(axes, dates):
        t = act.index.get_indexer([pd.Timestamp(dt)], method="nearest")[0]
        a = act.iloc[t].reindex(names).fillna(True).to_numpy(bool)
        keep = a[ei[0]] & a[ei[1]] & (ei[0] != ei[1]) & (et != 0)
        src_pv = graph.node_type[ei[0]] == 0
        for m, col in ((keep & (et == 1) & ~src_pv, "#9ad1c9"), (keep & (et == 1) & src_pv, FAM[1][1]),
                       (keep & (et == 2), FAM[2][1]), (keep & (et == 3), FAM[3][1])):
            segs = [(pos[names[s]], pos[names[d]]) for s, d in zip(ei[0, m], ei[1, m])]
            ax.add_collection(LineCollection(segs, colors=col, linewidths=.5, alpha=.5))
        xy = np.array([pos[n] for n in names])
        ax.scatter(xy[~a, 0], xy[~a, 1], s=14, facecolor="none", edgecolor="#bbb", lw=.6)
        ax.scatter(xy[a, 0], xy[a, 1], s=18, color=["#e76f51" if graph.node_type[i] == 0 else "#2a9d8f"
                                                    for i in np.flatnonzero(a)], edgecolor="k", lw=.3)
        ax.set_aspect("equal"); ax.axis("off"); ax.autoscale()
        n_act = int(a[graph.node_type == 0].sum())
        ax.set_title(f"{pd.Timestamp(dt):%Y-%m-%d}: active {n_act} PV nodes, {int(keep.sum())} edges")
    fig.suptitle("Induced active subgraphs (commission/decommission masks); hollow circles — inactive nodes")
    fig.tight_layout()
    _save(fig, out, "fig09_dynamic_induced_graphs")


def _stgnn_tables(runs):
    rows = []
    for r in runs:
        s = r["summary"].copy()
        s["variant"], s["seed"] = r["variant"], r["seed"]
        s["train_time_s"] = r["train_time_s"]
        rows.append(s)
    per = pd.concat(rows, ignore_index=True)
    ft = []
    for r in runs:
        a = r["fixed_after"].assign(mode="fixed weights")
        b = r["ft_after"].assign(mode="online fine-tuning")
        for x in (a, b):
            x["variant"], x["seed"] = r["variant"], r["seed"]
        ft += [a, b]
    ft = pd.concat(ft, ignore_index=True)
    cs = pd.concat([r["cold_start"].assign(variant=r["variant"], seed=r["seed"]) for r in runs], ignore_index=True)
    return per, ft, cs


def _agg(per, keys, cols):
    g = per.groupby(keys)[cols]
    m, s = g.mean(), g.std(ddof=1)
    out = m.copy()
    for c in cols:
        out[c + "_std"] = s[c]
        out[c + "_CV_%"] = s[c] / m[c].abs() * 100
    return out.reset_index()


def fig_stgnn(runs, out):
    per, ft, cs = _stgnn_tables(runs)
    mets = [("forecast_MAE_pct_rated", "Forecast MAE, % P_rated"), ("recon_MAE_pct_rated", "Reconstruction MAE, % P_rated"),
            ("physics_MAE_pct_rated", "Physics residual, % P_rated")]
    groups = [g for g in ["all", "cohort 1", "cohort 2", "cohort 3"] if g in set(per.group)]
    variants = [v for v in ("baseline", "multiview") if v in set(per.variant)]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    w = 0.8 / max(len(variants), 1)
    for i, (m, lab) in enumerate(mets):
        for k, v in enumerate(variants):
            g = per[per.variant == v].groupby("group")[m]
            mu, sd = g.mean().reindex(groups), g.std(ddof=1).reindex(groups)
            x = np.arange(len(groups)) + (k - (len(variants) - 1) / 2) * w
            ax[i].bar(x, mu, w, yerr=sd, capsize=2, color=VAR_COL[v], label=VAR_NAME[v])
            for xx, yy, ss in zip(x, mu, sd.fillna(0)):
                ax[i].text(xx, yy + ss + 0.03, f"{yy:.2f}", ha="center", va="bottom", fontsize=6)
        ax[i].set_xticks(range(len(groups))); ax[i].set_xticklabels(groups)
        ax[i].set_ylabel(lab)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    n_seeds = per.seed.nunique()
    fig.suptitle(f"Downstream metrics on the 2023 test by cohort (mean ± std over {n_seeds} seeds)")
    fig.tight_layout()
    _save(fig, out, "fig10_stgnn_metrics_by_cohort")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    sub = ft[ft.group.isin(["cohort 1", "cohort 2"])]
    cats = [(v, g) for v in variants for g in ("cohort 1", "cohort 2")]
    for k, (mode, col) in enumerate((("fixed weights", "#8d99ae"), ("online fine-tuning", "#e76f51"))):
        mu, sd = [], []
        for v, gname in cats:
            x = sub[(sub.variant == v) & (sub.group == gname) & (sub["mode"] == mode)]["forecast_MAE_pct_rated"]
            mu.append(x.mean()); sd.append(x.std(ddof=1))
        xs = np.arange(len(cats)) + (k - 0.5) * 0.38
        ax[0].bar(xs, mu, 0.38, yerr=sd, capsize=2, color=col, label=mode)
        for xx, yy in zip(xs, mu):
            ax[0].text(xx, yy + 0.05, f"{yy:.2f}", ha="center", va="bottom", fontsize=6)
    ax[0].set_xticks(range(len(cats)))
    ax[0].set_xticklabels([f"{VAR_NAME[v]}\n{g}" for v, g in cats], fontsize=7)
    ax[0].set_ylabel("Forecast MAE, % P_rated"); ax[0].legend(fontsize=7, loc="lower right")
    ax[0].set_title("a) Old plants: before and after online fine-tuning (30 d + replay)")
    if len(cs):
        c2 = cs.groupby(["variant", "plant"])["cold_start_h"].agg(["mean", "std"]).reset_index()
        plants = sorted(c2.plant.unique())
        for k, v in enumerate(variants):
            d = c2[c2.variant == v].set_index("plant").reindex(plants)
            x = np.arange(len(plants)) + (k - (len(variants) - 1) / 2) * w
            ax[1].bar(x, d["mean"], w, yerr=d["std"], color=VAR_COL[v], capsize=2, label=VAR_NAME[v])
        ax[1].set_xticks(range(len(plants))); ax[1].set_xticklabels(plants, rotation=30, fontsize=7)
        ax[1].set_ylabel("cold-start time, h")
        ax[1].set_title("b) New plants (cohort 3): time to MAE convergence")
        ax[1].legend()
    fig.tight_layout()
    _save(fig, out, "fig11_finetune_coldstart")

    fig, ax = plt.subplots(1, len(variants), figsize=(5.5 * len(variants), 3.3), squeeze=False)
    for k, v in enumerate(variants):
        for r in runs:
            if r["variant"] != v:
                continue
            h = r["history"]
            ax[0, k].plot(h.index + 1, h["total"], "-", color=VAR_COL[v], alpha=.6)
            if "val" in h:
                ax[0, k].plot(h.index + 1, h["val"], "--", color="#e76f51", alpha=.6)
        ax[0, k].set_yscale("log"); ax[0, k].set_xlabel("epoch"); ax[0, k].set_ylabel("loss")
        ax[0, k].set_title(f"{VAR_NAME[v]}: train (solid) / val (dashed), all seeds")
    fig.tight_layout()
    _save(fig, out, "fig12_training_curves")
    return per, ft, cs


def fig_anomaly(A, out):
    mp = A["most_probable"]
    pc = [c for c in mp.columns if c.startswith("p[")]
    if not pc:
        return
    d = mp[mp.n_events > 0].sort_values("anomalous_hours", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    bottom = np.zeros(len(d))
    cm = plt.get_cmap("Set1")
    for i, c in enumerate(pc):
        ax.bar(range(len(d)), d[c] * d["anomalous_hours"], bottom=bottom, color=cm(i), label=c[2:-1])
        bottom += (d[c] * d["anomalous_hours"]).to_numpy()
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d.index, rotation=90, fontsize=6)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("hours in anomaly events (test 2023), symlog")
    ax.set_title("Anomaly clusters: event composition by plant (most probable class — largest share)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out, "fig13_anomaly_classes")


def _est(A):
    """(label, file tag) of the b1 estimator used in stage 4 (SPS by default)."""
    e = A.get("estimator", "sps") if isinstance(A, dict) else "sps"
    return ("Kalman", "kalman") if e == "kalman" else ("SPS", "sps")


def fig_sps(A, D, out, n_show: int = 8):
    per, split = A["sps_periods"], A["sps_split"]
    lab, tag = _est(A)
    if not len(per):
        return
    ev = A["evidence"]
    pick = list(ev[ev.evidence.str.startswith("strong")].index[:4]) + \
           list(ev.sort_values("sps_b1_change_pct").index[:n_show])
    pick = list(dict.fromkeys([p for p in pick if p in set(per.plant)]))[:n_show]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for ax, p in zip(axes.ravel(), pick):
        g = per[per.plant == p]
        x = g["period"]
        ax.fill_between(x, g["b1_lo"], g["b1_hi"], step="mid", alpha=.3, color="#1f5f8b", label=f"{lab} 95 %")
        ax.plot(x, g["theta_b1_Pexp"], "o-", ms=3, color="#1f5f8b", label="OLS β1" if tag == "sps" else "smoothed β1")
        dis = g[g["disjoint_from_previous"] == True]  # noqa: E712
        ax.scatter(dis["period"], dis["theta_b1_Pexp"], s=40, facecolor="none", edgecolor="r", lw=1.2,
                   label="no overlap with previous")
        ax.axhline(1.0, color="k", lw=.6, ls=":")
        ax.axvline(pd.Timestamp("2023-01-01"), color="grey", lw=.6)
        ax.set_title(p, fontsize=9); ax.tick_params(axis="x", rotation=45, labelsize=7)
    axes[0, 0].set_ylabel("β1"); axes[1, 0].set_ylabel("β1"); axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{lab}: β1 confidence regions by quarter (model P = β0 + β1·P_exp,cal + β2·T_cell + β3·w)")
    fig.tight_layout()
    _save(fig, out, "fig14_sps_b1_trajectories" if tag == "sps" else "fig14_kalman_b1_trajectories")

    s = split.dropna(subset=["train_b1", "test_b1"]).sort_values("b1_change_pct")
    fig, ax = plt.subplots(figsize=(7, 9))
    y = np.arange(len(s))
    ax.errorbar(s["train_b1"], y + .15, xerr=[s["train_b1"] - s["train_b1_lo"], s["train_b1_hi"] - s["train_b1"]],
                fmt="o", ms=3, color="#8d99ae", label="train 2021–2022")
    col = np.where(s["regions_disjoint"] == True, "#c0392b", "#1f5f8b")  # noqa: E712
    ax.errorbar(s["test_b1"], y - .15, xerr=[s["test_b1"] - s["test_b1_lo"], s["test_b1_hi"] - s["test_b1"]],
                fmt="none", ecolor=col)
    ax.scatter(s["test_b1"], y - .15, c=col, s=10, label="test 2023 (red — regions do not overlap)")
    ax.axvline(1, color="k", lw=.6, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(s["plant"], fontsize=6)
    ax.set_xlabel(f"β1 ({lab} 95 %)"); ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"{lab}: β1 on train and test for plants with data in both periods")
    fig.tight_layout()
    _save(fig, out, "fig15_sps_train_vs_test" if tag == "sps" else "fig15_kalman_train_vs_test")


# ============================================================================ xlsx
def _plant_summary(D, C, A, runs):
    ca, deg = D["ca"], D["ca"].degradation
    rows = pd.DataFrame(index=D["pv_cols"])
    rows["Plant"] = [short(c) for c in rows.index]
    rows["Cohort"] = ca.cohorts["cohort"]
    rows["Cohort (name)"] = ca.cohorts["cohort_name"]
    rows["Cohort (task list)"] = ca.cohorts["reference_cohort"]
    rows["P_rated, kW"] = pd.Series(D["rated"])
    rows["Technical cluster"] = C["tech"].labels.map(lambda v: f"T{v}")
    rows["Behavioural cluster"] = C["beh"].labels.map(lambda v: f"B{v}")
    rows["Cold-start (warm-up)"] = C["F"]["warmup"].astype(bool)
    rows["Cold-start baseline P/P_exp"] = C["cold"]
    for y in (2021, 2022, 2023):
        rows[f"PR {y}"] = deg[f"PR_{y}"]
    rows["Degradation / change, %"] = deg["degradation_pct_headline"]
    rows["Indicator type"] = deg["metric_type"]
    rows["Rd YoY, %/yr"] = deg.get("Rd_yoy_pct_per_yr")
    rows["Rd YoY 95% lower"] = deg.get("Rd_yoy_lo")
    rows["Rd YoY 95% upper"] = deg.get("Rd_yoy_hi")
    rows["Rd YoY fleet-norm., %/yr"] = deg.get("Rd_yoy_fleetnorm_pct_per_yr")
    rows["ΔPR 2023/2022, %"] = deg["dPR_2023_2022_pct"]
    rows["ΔPR 95% lower"] = deg["dPR_2023_2022_lo"]
    rows["ΔPR 95% upper"] = deg["dPR_2023_2022_hi"]
    rows["ΔPR fleet-norm., %"] = deg["dPR_2023_2022_fleetnorm_pct"]
    mp = A["most_probable"].reindex(rows["Plant"].values)
    rows["Most probable class"] = mp["most_probable_class"].values
    rows["Class probability"] = mp["class_probability"].values
    if "class_agreement_5seeds" in mp:
        rows["Class agreement over 5 seeds"] = mp["class_agreement_5seeds"].values
    rows["Events (test)"] = mp.get("n_events", pd.Series(np.nan, index=mp.index)).values
    ev = A["evidence"].reindex(rows["Plant"].values)
    L, _tag = _est(A)
    for c_src, c_dst in (("stgnn_mean_score", "STGNN mean score, σ"), ("stgnn_exceed_share", "STGNN exceedance share"),
                         ("sps_train_b1", f"{L} β1 train"), ("sps_test_b1", f"{L} β1 test"),
                         ("sps_test_b1_lo", f"{L} β1 test lower"), ("sps_test_b1_hi", f"{L} β1 test upper"),
                         ("sps_b1_change_pct", "Δβ1, %"), ("sps_regions_disjoint", f"{L} regions do not overlap"),
                         ("sps_n_regime_changes", f"{L} regime changes (quarters)"), ("evidence", "Final evidence")):
        if c_src in ev:
            rows[c_dst] = ev[c_src].values
    rows["Mean monthly coverage"] = deg["mean_monthly_coverage"]
    return rows.reset_index(drop=True).sort_values(["Cohort", "Plant"])


README_ROWS = [
    ("Plant_summary", "Main table for the paper: cohort, clusters, PR by year, degradation / change percentage "
     "(indicator matches the cohort), SPS estimates of β1, most probable anomaly class and final evidence."),
    ("Cohort_summary", "Median and IQR of degradation indicators by cohort."),
    ("STGNN_summary", "Downstream metrics (forecast MAE, reconstruction MAE, physics residual — in % of rated power) "
     "by model variant and cohort: mean, std and CV over five seeds."),
    ("STGNN_by_seed", "The same metrics for each run (variant × seed × cohort)."),
    ("Fine_tuning", "MAE of old plants after online fine-tuning (first 30 days of 2023 + replay) vs fixed weights "
     "on the same windows (after day 30)."),
    ("Cold_start", "Cold-start time of new plants: hours until the rolling (72 h) MAE converges to 1.1 × the median MAE of cohort 1; "
     "time_to_valid_score_h — delay until the first valid anomaly score (window + calibration)."),
    ("Clustering_metrics", "Selected solutions: Silhouette (primary), Davies–Bouldin (secondary), Calinski–Harabasz (additional), ARI over 5 seeds."),
    ("Clustering_grids", "Full algorithm × k grids for all views."),
    ("Coverage_by_month", "Share of valid daytime measurements (GHI>200, P_exp>0, no filled gaps, after commissioning)."),
    ("Cohorts", "Data-driven cohort (first year with ≥5 covered months at the 80 % threshold) and check against the task list."),
    ("Degradation", "Full table: PR by year, YoY Rd with bootstrap CI, seasonal OLS estimate, fleet-normalised variants, ΔPR."),
    ("Systemic_shift_2023", "Check of the synchronous 2023 drop: pyranometer k_t, median PR of cohort 1 before/after harmonisation."),
    ("Technical_features", "Brick metadata (coordinates, power, tilt, azimuth, inverter, panel) and technical cluster."),
    ("Behavioural_features", "Features from the normal train part only: dP/dGHI, median P/P_exp, profile, "
     "temperature sensitivity, IQR of the physics residual, median η, correlation with neighbours."),
    ("Weather_regimes", "Regime profiles w_t = [GHI, ΔGHI, σ(GHI), T, wind, cloudiness, RH, Vis, precipitation]."),
    ("Anomaly_events", "STGNN events (multi-view, seed 0) with features and cluster/class."),
    ("Anomalies_by_plant", "Class distribution by plant, most probable class and its stability over 5 seeds."),
    ("SPS_quarters", "SPS 95 % intervals of β1 by quarter, flag of no overlap with the previous quarter."),
    ("SPS_train_test", "β1 on train (2021–2022) and test (2023), non-overlap of regions."),
    ("SPS_clusters", "Cluster-level (pooled) regressions on train by technical cluster."),
    ("Graph", "Supergraph edge composition by type for both model variants."),
    ("Prior_strength", "Retained strength of the physics prior (λ) in the attention layers by edge family."),
]


def _write_xlsx(path, sheets: Dict[str, pd.DataFrame]):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            d = df.copy()
            d.columns = pd.Index([str(c) for c in d.columns])
            if d.columns.duplicated().any():
                d = d.loc[:, ~d.columns.duplicated()]
            for c in d.columns:
                if pd.api.types.is_datetime64_any_dtype(d[c]):
                    d[c] = d[c].dt.tz_localize(None) if getattr(d[c].dt, "tz", None) else d[c]
            d.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "B2"
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F5F8B")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            ws.row_dimensions[1].height = 42
            for j, c in enumerate(d.columns, 1):
                vals = [str(v) for v in d[c].head(200).tolist()]
                width = min(max([len(str(c)) * 0.8] + [len(v) for v in vals]) + 2, 60)
                ws.column_dimensions[get_column_letter(j)].width = max(width, 8)
                if pd.api.types.is_float_dtype(d[c]):
                    for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                        for cell in row:
                            cell.number_format = "0.000"
            if name.startswith("Description"):
                ws.column_dimensions["B"].width = 120
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")


def build_report(D, C, runs, A, out):
    os.makedirs(out, exist_ok=True)
    fig_coverage(D, out); fig_sensor(D, out); fig_pr_cohorts(D, out); fig_degradation(D, out)
    fig_cluster_metrics(C, A, out); fig_clusters(D, C, out); fig_weather(C, out)
    fig_graph(D, C, out); fig_dynamic_graph(D, C, out)
    per, ft, cs = fig_stgnn(runs, out) if runs else (pd.DataFrame(),) * 3
    fig_anomaly(A, out); fig_sps(A, D, out)

    ca = D["ca"]
    summary = _plant_summary(D, C, A, runs)
    mcols = ["forecast_MAE_pct_rated", "recon_MAE_pct_rated", "physics_MAE_pct_rated"]
    stg = _agg(per, ["variant", "group"], mcols) if len(per) else pd.DataFrame()
    ft_s = _agg(ft, ["variant", "group", "mode"], mcols) if len(ft) else pd.DataFrame()
    cs_s = cs.groupby(["variant", "plant"]).agg(cold_start_h_mean=("cold_start_h", "mean"),
                                                cold_start_h_std=("cold_start_h", "std"),
                                                time_to_valid_score_h=("time_to_valid_score_h", "first"),
                                                steady_MAE_pct=("steady_MAE_pct", "mean")).reset_index() if len(cs) else pd.DataFrame()
    cm = [C["tech"].metrics_row(), C["beh"].metrics_row(), C["wr"].result.metrics_row()]
    if A.get("anom") is not None:
        cm.append(A["anom"].metrics_row())
    grids = pd.concat([C["tech"].scores.assign(view="technical"), C["beh"].scores.assign(view="behavioural"),
                       C["wr"].result.scores.assign(view="weather_regimes")] +
                      ([A["anom"].scores.assign(view="anomaly")] if A.get("anom") is not None else []))
    s = D["sensor"]
    sysd = pd.DataFrame({"month": s["kt_quantile"].index, "k_t 98th percentile": s["kt_quantile"].values})
    sysd["median PR of cohort 1 (original GHI)"] = D["ca_raw"].systemic["fleet_median_pr"].reindex(s["kt_quantile"].index).values
    sysd["median PR of cohort 1 (harmonised GHI)"] = ca.systemic["fleet_median_pr"].reindex(s["kt_quantile"].index).values
    key = pd.DataFrame([
        ("Pyranometer scale change point", str(s["change_point"])),
        ("k_t shift, %", s["shift_pct"]), ("p-value (Mann–Whitney)", s["p_value"]),
        ("GHI harmonisation factor after the change point", s["factor_after_cp"]),
        ("Pyranometer failure months", ", ".join(str(m.date()) for m in s["invalid_months"])),
        ("Synchronous share of cohort 1 plants (original GHI)", D["ca_raw"].systemic["synchronous_fraction"]),
        ("Shift of the median PR of cohort 1, % (original GHI)", D["ca_raw"].systemic["fleet_shift_pct"]),
        ("Conclusion (original GHI)", D["ca_raw"].systemic["note"]),
        ("Synchronous share after harmonisation", ca.systemic["synchronous_fraction"]),
        ("Conclusion after harmonisation", ca.systemic["note"]),
        ("Annual GHI, kWh/m² (original)", "; ".join(f"{int(i)}: {v:.0f}" for i, v in
                                                         D["ca_raw"].systemic["ghi_year"]["annual_kWh_m2"].items())),
    ], columns=["Indicator", "Value"])
    key["Value"] = key["Value"].astype(str)
    tf = C["tech"].extra["raw_features"].copy()
    tf.insert(0, "Plant", [short(c) for c in tf.index]); tf["Technical cluster"] = C["tech"].labels.values
    bf = C["F"].copy()
    bf.insert(0, "Plant", [short(c) for c in bf.index]); bf["Behavioural cluster"] = C["beh"].labels.reindex(bf.index).values
    wp = C["wr"].profile.copy(); wp.insert(0, "Regime", [C["wr"].names[int(i)] for i in wp.index])
    af = A["anom_feat"].copy()
    if A.get("anom") is not None and len(af):
        af["Cluster"] = A["anom"].labels.values
        af["Class"] = A["anom"].labels.map(A["anom"].extra["class_names"]).values
    mp = A["most_probable"].copy(); mp.insert(0, "Plant", mp.index)
    gr = []
    for nm, g in (("baseline", D["graph_base"]), ("multiview", C["graph_mv"])):
        cnt = pd.Series(g.edge_type).value_counts().sort_index()
        for t, n in cnt.items():
            gr.append(dict(variant=nm, type=int(t), family=FAM.get(int(t), ("self-loops/other",))[0], edges=int(n)))
    ps = pd.concat([r["prior_strength"].assign(variant=r["variant"], seed=r["seed"]) for r in runs]) if runs else pd.DataFrame()
    readme = pd.DataFrame(README_ROWS, columns=["Sheet", "Contents"])
    cov = ca.cov_m.T.copy(); cov.columns = [c.strftime("%Y-%m") for c in cov.columns]
    cov.insert(0, "Plant", [short(c) for c in cov.index]); cov.insert(1, "Cohort", ca.cohorts["cohort"].values)
    sheets = {
        "Description": readme, "Plant_summary": summary, "Cohort_summary": cohort_summary(ca.degradation),
        "STGNN_summary": stg, "STGNN_by_seed": per, "Fine_tuning": ft_s, "Cold_start": cs_s,
        "Clustering_metrics": pd.DataFrame(cm), "Clustering_grids": grids,
        "Coverage_by_month": cov, "Cohorts": ca.cohorts.reset_index(drop=True),
        "Degradation": ca.degradation.reset_index(drop=True),
        "Systemic_shift_2023": pd.concat([key, pd.DataFrame([{}]), sysd.astype(object)], ignore_index=True),
        "Technical_features": tf, "Behavioural_features": bf, "Weather_regimes": wp,
        "Anomaly_events": af, "Anomalies_by_plant": mp,
        f"{_est(A)[0]}_quarters": A["sps_periods"], f"{_est(A)[0]}_train_test": A["sps_split"].reset_index(drop=True),
        f"{_est(A)[0]}_clusters": A["sps_cluster"], "Graph": pd.DataFrame(gr), "Prior_strength": ps,
    }
    if A.get("est_daily") is not None and len(A["est_daily"]):
        sheets["Kalman_by_day"] = A["est_daily"]
    _write_xlsx(os.path.join(out, "PV_results_tables.xlsx"), sheets)
    summary.to_csv(os.path.join(out, "plant_summary.csv"), index=False)
    return sheets
