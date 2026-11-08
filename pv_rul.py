"""
Remaining useful life (RUL) of the PV fleet from fleet-adjusted degradation rates.

Independent, optional module (notebook section 19).  It only reads objects that the
notebook has already computed:

    ca   CohortAnalysis from pv_cohorts.analyse_cohorts (monthly PR, coverage, cohorts)
    per  per plant x quarter b1 table from pv_sps.sps_periods (or pv_kalman.kalman_periods)

and does not modify them.  Nothing else in the project depends on this module.

Rules
-----
1. The rate is NOT taken from the raw PR median.  Two rate sources are reported:
     "PR, fleet-adjusted"  monthly PR divided by the leave-one-out median PR of the
                           cohort-1 plants in the same month (removes the common
                           fleet shift, e.g. the 2023 irradiance-sensor change);
     "SPS b1"              quarterly b1 of the physical regression (pv_sps), which is
                           estimated on the irradiance-harmonised data; optionally also
                           fleet-adjusted, and "Kalman b1" if the Kalman table is given;
     "STGNN eta"           monthly median of the daylight loss coefficient eta_hat of the
                           frozen physics-informed STGNN (eta_hat * P_exp ~ P), averaged over
                           the available seeds.  eta_hat = sigmoid(.) < 1, so a plant whose
                           P / P_exp exceeds 1 is saturated and mild degradation is hidden;
                           the years 2021-2022 are in-sample for the network.
   The fleet-adjusted rate is a rate RELATIVE to the fleet: a degradation common to all
   plants is removed together with the sensor shift, so this RUL is optimistic if the
   whole fleet degrades.  The b1 rate keeps the common trend.  The two sources therefore
   bracket the plant-specific and the total degradation.
2. Rate = median of year-on-year ratios (month- or quarter-matched, RdTools style), %/yr.
   Uncertainty: nonparametric bootstrap that resamples the YoY pairs (rate) and the
   months of the reference and of the last year (current level) jointly; the RUL interval
   is the percentile interval of the bootstrap RUL, not only the median.
3. RUL = (c - theta) / |r|, c = current level relative to the reference (first) year,
   theta = end-of-life level (default 0.80 of the reference performance).
4. Censoring: plants without significant degradation (upper CI bound of the rate >= 0)
   and plants whose point RUL exceeds the horizon H are reported as RIGHT-CENSORED,
   "RUL > H", and are kept in the table and the figure (arrow to the right).
   If the rate is significant but the upper RUL bound exceeds H, the interval is
   right-open ("estimated, upper bound > H").
5. Cohort 2 (first reliable year 2022) has only one year pair, 2023/2022.  Its RUL is
   either not computed (cohort2="skip") or reported as a CONDITIONAL scenario
   "if the 2023/2022 change persists" (cohort2="conditional", default), with its own
   status and marker.  Cohort 3 (2023 only) is not estimable.

References: Jordan & Kurtz (2013) Prog. Photovolt. 21:12-29; Jordan et al. (2018)
IEEE J. Photovolt. 8:525-531 (YoY method); IEC 61724-1:2021 (PR).
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

COHORT_LABEL = {1: "Cohort 1 (2021-2023)", 2: "Cohort 2 (from 2022)", 3: "Cohort 3 (2023 only)",
                0: "Insufficient data"}

ST_EST = "estimated"
ST_OPEN = "estimated, upper bound > H"
ST_CENS_NS = "censored: no significant degradation"
ST_CENS_H = "censored: RUL > H"
ST_BELOW = "below threshold"
ST_COND = "conditional"            # prefix for cohort 2 ("conditional: ...")
ST_NA = "not estimable"

COND_NOTE = "conditional: if the 2023/2022 change persists"

TITLES = {
    "plants": "Per-plant RUL (bootstrap intervals; censored plants kept)",
    "summary": "RUL status by rate source and cohort",
    "agreement": "Agreement of RUL status between rate sources (cohort 1)",
}


@dataclass
class RULConfig:
    threshold: float = 0.80          # end-of-life level relative to the reference year
    horizon: float = 30.0            # H, years; RUL beyond H is right-censored
    n_boot: int = 2000
    ci: float = 0.95
    min_pairs_pr: int = 6            # monthly YoY pairs
    min_pairs_b1: int = 3            # quarterly YoY pairs
    min_ref: int = 3                 # reference plants needed for the fleet median
    min_cov: float = 0.8             # monthly coverage for a valid PR month
    cohort2: str = "conditional"     # "conditional" or "skip"
    b1_fleet_adjust: bool = False    # also divide b1 by the fleet median
    eta_fleet_adjust: bool = False   # also divide STGNN eta by the fleet median
    day_frac: float = 0.05           # daylight for eta: P_exp > day_frac * P_rated
    min_eta_hours: int = 50          # daylight hours for a valid eta month
    seed: int = 0


@dataclass
class RULResult:
    plants: pd.DataFrame
    summary: pd.DataFrame
    agreement: pd.DataFrame
    series: Dict[str, pd.DataFrame] = field(default_factory=dict)
    cfg: RULConfig = field(default_factory=RULConfig)
    meta: Dict[str, object] = field(default_factory=dict)

    def tables(self) -> Dict[str, pd.DataFrame]:
        return {"plants": self.plants, "summary": self.summary, "agreement": self.agreement}


# ----------------------------------------------------------------------------- series
def _short(c: str) -> str:
    try:
        from pv_cohorts import short
        return short(c)
    except Exception:
        return str(c).split("/")[-1].replace("_", " ")


def fleet_adjust(S: pd.DataFrame, ref_cols: Sequence[str], min_ref: int = 3) -> pd.DataFrame:
    """S_i(t) / median_{k in ref, k != i} S_k(t)  (leave-one-out fleet median).

    Periods with fewer than ``min_ref`` valid reference plants become NaN."""
    ref_cols = [c for c in ref_cols if c in S.columns]
    R = S[ref_cols]
    out = pd.DataFrame(np.nan, index=S.index, columns=S.columns)
    n_ok = R.notna().sum(axis=1)
    med_all = R.median(axis=1)
    for c in S.columns:
        if c in ref_cols:
            med = R.drop(columns=c).median(axis=1)
            ok = (n_ok - R[c].notna().astype(int)) >= min_ref
        else:
            med, ok = med_all, n_ok >= min_ref
        out[c] = (S[c] / med).where(ok)
    return out


def pr_series(ca, cfg: RULConfig, adjust: bool = True) -> pd.DataFrame:
    """Monthly PR on covered months (optionally fleet-adjusted)."""
    covered = ca.cov_m >= cfg.min_cov
    S = ca.pr_m.where(covered)
    S = S.where(S > 0)
    if adjust:
        ref = ca.cohorts.index[ca.cohorts["cohort"] == 1].tolist()
        S = fleet_adjust(S, ref, cfg.min_ref)
    return S


def b1_series(per: pd.DataFrame, cohorts: pd.DataFrame, cfg: RULConfig,
              adjust: bool = False) -> pd.DataFrame:
    """Quarterly b1 (LS point inside the SPS / Kalman interval) as plant columns."""
    if per is None or not len(per):
        return pd.DataFrame()
    col = "theta_b1_Pexp" if "theta_b1_Pexp" in per else "b1"
    S = per.pivot_table(index="period", columns="column", values=col, aggfunc="mean")
    S.index = pd.DatetimeIndex(S.index)
    S = S.where(S > 0).sort_index()
    S = S.reindex(pd.date_range(S.index.min(), S.index.max(), freq="QS"))
    if adjust:
        ref = [c for c in cohorts.index[cohorts["cohort"] == 1] if c in S.columns]
        S = fleet_adjust(S, ref, cfg.min_ref)
    return S


def stgnn_eta_hourly(models, df: pd.DataFrame, phys: Dict[str, np.ndarray], cfg: Optional["RULConfig"] = None,
                     batch_size: int = 64) -> Dict[str, pd.DataFrame]:
    """Hourly daylight eta_hat of every frozen model over the WHOLE record (train + test windows).

    ``models``: {seed: pv_semisynth_eval.FrozenSTGNN}.  Returns {"eta": mean over seeds,
    "n_seeds": ...}.  The weights are checked to be unchanged."""
    from pv_semisynth_sensitivity import _collect
    cfg = cfg or RULConfig()
    frames = []
    for seed, mdl in (models or {}).items():
        fp = mdl.fingerprint()
        pipe = mdl.build(df, phys)
        raw = _collect(pipe, pipe.ds_full, pipe.ds_full.idx, batch_size)
        pv = list(pipe.graph.meta["pv_cols"])
        n = len(pv)
        eta, a = raw["eta"].mean(-1)[:, :n], raw["a"].mean(-1)[:, :n]
        ok = raw["active"][:, :n] & (a > cfg.day_frac) & np.isfinite(eta)
        frames.append(pd.DataFrame(np.where(ok, eta, np.nan), index=raw["ts"], columns=pv))
        del pipe
        assert mdl.fingerprint() == fp, "STGNN weights changed"
    if not frames:
        return {}
    E = sum(f for f in frames) / len(frames) if len(frames) > 1 else frames[0]
    E = E[~E.index.duplicated()].sort_index()
    return {"eta": E, "n_seeds": len(frames)}


def eta_series(eta_hourly: pd.DataFrame, cohorts: pd.DataFrame, cfg: "RULConfig",
               adjust: bool = False) -> pd.DataFrame:
    """Monthly median daylight eta_hat (months with < ``min_eta_hours`` hours -> NaN)."""
    if eta_hourly is None or not len(eta_hourly):
        return pd.DataFrame()
    g = eta_hourly.resample("MS")
    S = g.median().where(g.count() >= cfg.min_eta_hours)
    S = S.where(S > 0)
    if adjust:
        ref = [c for c in cohorts.index[cohorts["cohort"] == 1] if c in S.columns]
        S = fleet_adjust(S, ref, cfg.min_ref)
    return S


def source_series(ca, per_sps=None, per_kalman=None, eta_hourly=None, cfg: Optional["RULConfig"] = None,
                  pr_m: Optional[pd.DataFrame] = None) -> Dict[str, tuple]:
    """{source: (series, lag, min_pairs)} for all available rate sources."""
    cfg = cfg or RULConfig()
    coh = ca.cohorts
    if pr_m is not None:
        ca = _with_pr(ca, pr_m)
    out: Dict[str, tuple] = {"PR, fleet-adjusted": (pr_series(ca, cfg, adjust=True), 12, cfg.min_pairs_pr)}
    if per_sps is not None and len(per_sps):
        out["SPS b1"] = (b1_series(per_sps, coh, cfg, adjust=False), 4, cfg.min_pairs_b1)
        if cfg.b1_fleet_adjust:
            out["SPS b1, fleet-adjusted"] = (b1_series(per_sps, coh, cfg, adjust=True), 4, cfg.min_pairs_b1)
    if per_kalman is not None and len(per_kalman):
        out["Kalman b1"] = (b1_series(per_kalman, coh, cfg, adjust=False), 4, cfg.min_pairs_b1)
    if eta_hourly is not None and len(eta_hourly):
        out["STGNN eta"] = (eta_series(eta_hourly, coh, cfg, adjust=False), 12, cfg.min_pairs_pr)
        if cfg.eta_fleet_adjust:
            out["STGNN eta, fleet-adjusted"] = (eta_series(eta_hourly, coh, cfg, adjust=True), 12, cfg.min_pairs_pr)
    return out


class _with_pr:
    """CohortAnalysis view with another monthly PR table (semi-synthetic data)."""

    def __init__(self, ca, pr_m):
        self.pr_m, self.cov_m, self.cohorts = pr_m, ca.cov_m, ca.cohorts


# ----------------------------------------------------------------------------- per plant
def _yoy(s: pd.Series, lag: int):
    """Year-on-year ratios (b/a - 1) with the year pair label of each ratio."""
    s = s.dropna()
    vals, pairs = [], []
    idx = set(s.index)
    for t, a in s.items():
        t2 = t + pd.DateOffset(years=1)
        if t2 in idx and a > 0:
            vals.append(s[t2] / a - 1.0)
            pairs.append(f"{t2.year}/{t.year}")
    return np.asarray(vals, float), pairs


def _level(s: pd.Series):
    """Reference (first year with data) and last-year samples of the series."""
    s = s.dropna()
    if not len(s):
        return np.array([]), np.array([]), None, None
    y0, y1 = int(s.index.year.min()), int(s.index.year.max())
    return (s[s.index.year == y0].to_numpy(float), s[s.index.year == y1].to_numpy(float), y0, y1)


def plant_rul(s: pd.Series, lag: int, cfg: RULConfig, min_pairs: int, rng: np.random.Generator):
    """Rate (%/yr), current level, RUL and bootstrap intervals for one plant."""
    r_vals, pairs = _yoy(s, lag)
    ref, last, y0, y1 = _level(s)
    out = dict(n_pairs=len(r_vals), year_pairs=",".join(sorted(set(pairs))), ref_year=y0, last_year=y1,
               rate_pct_per_yr=np.nan, rate_lo=np.nan, rate_hi=np.nan, level=np.nan, level_lo=np.nan,
               level_hi=np.nan, rul=np.nan, rul_lo=np.nan, rul_hi=np.nan, p_rate_neg=np.nan)
    if len(r_vals) < min_pairs or len(ref) == 0 or len(last) == 0 or y0 == y1:
        out["status"] = ST_NA
        return out
    a = (1 - cfg.ci) / 2
    th, H = cfg.threshold, cfg.horizon
    r = float(np.median(r_vals))
    c = float(np.median(last) / np.median(ref))
    B = cfg.n_boot
    rb = np.median(rng.choice(r_vals, (B, len(r_vals)), replace=True), axis=1)
    cb = (np.median(rng.choice(last, (B, len(last)), replace=True), axis=1)
          / np.median(rng.choice(ref, (B, len(ref)), replace=True), axis=1))

    def _rul(cv, rv):
        rv = np.asarray(rv, float); cv = np.asarray(cv, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(rv < 0, (cv - th) / (-rv), np.inf)
        return np.where(cv <= th, 0.0, x)

    Rb = _rul(cb, rb)
    out.update(rate_pct_per_yr=r * 100, rate_lo=float(np.quantile(rb, a)) * 100,
               rate_hi=float(np.quantile(rb, 1 - a)) * 100, level=c,
               level_lo=float(np.quantile(cb, a)), level_hi=float(np.quantile(cb, 1 - a)),
               p_rate_neg=float(np.mean(rb < 0)))
    point = float(_rul(c, r))
    Rq = np.where(np.isfinite(Rb), Rb, 1e9)              # inf-safe percentiles
    lo, hi = (float(v) if v < 1e8 else np.inf for v in np.quantile(Rq, [a, 1 - a]))
    out.update(rul=point, rul_lo=lo, rul_hi=hi)
    if c <= th:
        out["status"] = ST_BELOW
    elif out["rate_hi"] >= 0:
        out["status"] = ST_CENS_NS
    elif point > H:
        out["status"] = ST_CENS_H
    elif hi > H:
        out["status"] = ST_OPEN
    else:
        out["status"] = ST_EST
    return out


def rul_table(S: pd.DataFrame, lag: int, cohorts: pd.DataFrame, source: str, cfg: RULConfig,
              min_pairs: int) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for c in cohorts.index:
        coh = int(cohorts.loc[c, "cohort"])
        base = dict(column=c, plant=_short(c), cohort=coh, source=source)
        if coh in (0, 3) or c not in S.columns:
            reason = "2023 only" if coh == 3 else ("insufficient data" if coh == 0 else "no series")
            rows.append(dict(base, status=f"{ST_NA} ({reason})", n_pairs=0))
            continue
        if coh == 2 and cfg.cohort2 == "skip":
            rows.append(dict(base, status=f"{ST_NA} (cohort 2: one year pair)", n_pairs=0))
            continue
        r = plant_rul(S[c], lag, cfg, min_pairs, rng)
        if coh == 2 and not r["status"].startswith(ST_NA):
            r["status"] = f"{COND_NOTE} | {r['status']}"
        rows.append(dict(base, **r))
    out = pd.DataFrame(rows)
    out["censored"] = out["status"].str.contains("censored")
    out["conditional"] = out["status"].str.startswith(ST_COND)
    return out


# ----------------------------------------------------------------------------- run
def run_rul(ca, per_sps: Optional[pd.DataFrame] = None, per_kalman: Optional[pd.DataFrame] = None,
            cfg: Optional[RULConfig] = None, out_dir: Optional[str] = "results/rul",
            eta_hourly: Optional[pd.DataFrame] = None, models=None, df=None, phys=None) -> RULResult:
    """RUL for every plant and every available rate source.

    STGNN source: pass ``eta_hourly`` (from :func:`stgnn_eta_hourly`) or ``models`` + ``df`` + ``phys``."""
    cfg = cfg or RULConfig()
    if cfg.cohort2 not in ("conditional", "skip"):
        raise ValueError("cohort2 must be 'conditional' or 'skip'")
    coh = ca.cohorts
    n_seeds = 0
    if eta_hourly is None and models and df is not None and phys is not None:
        e = stgnn_eta_hourly(models, df, phys, cfg)
        eta_hourly, n_seeds = e.get("eta"), e.get("n_seeds", 0)
    srcs = source_series(ca, per_sps, per_kalman, eta_hourly, cfg)
    plants = pd.concat([rul_table(S, lag, coh, name, cfg, mp) for name, (S, lag, mp) in srcs.items()],
                       ignore_index=True)
    res = RULResult(plants=plants, summary=summary_table(plants, cfg),
                    agreement=agreement_table(plants), series={k: v[0] for k, v in srcs.items()},
                    cfg=cfg, meta=dict(sources=list(srcs), threshold=cfg.threshold, horizon=cfg.horizon,
                                       cohort2=cfg.cohort2, n_boot=cfg.n_boot, stgnn_seeds=n_seeds))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for k, t in res.tables().items():
            t.to_csv(os.path.join(out_dir, f"rul_{k}.csv"), index=False)
    return res


def _base_status(s: str) -> str:
    return s.split(" | ")[-1] if " | " in s else s


def summary_table(plants: pd.DataFrame, cfg: RULConfig) -> pd.DataFrame:
    rows = []
    for (src, coh), g in plants.groupby(["source", "cohort"]):
        st = g["status"].map(_base_status)
        fin = g[st.isin([ST_EST, ST_OPEN])]
        rows.append({"Rate source": src, "Cohort": COHORT_LABEL.get(int(coh), coh), "n": len(g),
                     "Estimated": int((st == ST_EST).sum()), "Upper bound > H": int((st == ST_OPEN).sum()),
                     "Censored, n.s.": int((st == ST_CENS_NS).sum()),
                     f"Censored, RUL > {cfg.horizon:g} yr": int((st == ST_CENS_H).sum()),
                     "Below threshold": int((st == ST_BELOW).sum()),
                     "Not estimable": int(st.str.startswith(ST_NA).sum()),
                     "Conditional": "yes" if int(coh) == 2 and cfg.cohort2 == "conditional" else "",
                     "Median RUL of estimated, yr": float(fin["rul"].median()) if len(fin) else np.nan,
                     "Median rate, %/yr": float(g["rate_pct_per_yr"].median()) if g["rate_pct_per_yr"].notna().any() else np.nan})
    return pd.DataFrame(rows)


def agreement_table(plants: pd.DataFrame) -> pd.DataFrame:
    """Cohort 1: same status class (finite / censored / other) under two rate sources."""
    p = plants[plants["cohort"] == 1].copy()
    p["cls"] = p["status"].map(_base_status).map(
        lambda s: "finite" if s in (ST_EST, ST_OPEN, ST_BELOW) else ("censored" if "censored" in s else "n/a"))
    W = p.pivot_table(index="column", columns="source", values="cls", aggfunc="first")
    rows, srcs = [], list(W.columns)
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            a, b = W[srcs[i]], W[srcs[j]]
            ok = (a != "n/a") & (b != "n/a") & a.notna() & b.notna()
            rows.append({"Source A": srcs[i], "Source B": srcs[j], "n": int(ok.sum()),
                         "Same class": int((a[ok] == b[ok]).sum()),
                         "Finite in A only": int(((a == "finite") & (b == "censored") & ok).sum()),
                         "Finite in B only": int(((a == "censored") & (b == "finite") & ok).sum()),
                         "Share same": float((a[ok] == b[ok]).mean()) if ok.any() else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- display
def _fmt_rul(r, H) -> str:
    s = _base_status(str(r["status"]))
    if s.startswith(ST_NA):
        return "n/a"
    if s == ST_BELOW:
        return "0 (reached)"
    lo = r["rul_lo"]
    lo_s = f"{lo:.1f}" if np.isfinite(lo) and lo <= H else f">{H:g}"
    if "censored" in s:
        return f"> {H:g}" if not np.isfinite(lo) or lo > H else f"> {H:g} (lower {lo_s})"
    hi = r["rul_hi"]
    hi_s = f"{hi:.1f}" if np.isfinite(hi) and hi <= H else f">{H:g}"
    return f"{r['rul']:.1f} [{lo_s}, {hi_s}]"


def plant_view(res: RULResult) -> pd.DataFrame:
    H = res.cfg.horizon
    p = res.plants.copy()
    out = pd.DataFrame({
        "Plant": p["plant"], "Cohort": p["cohort"], "Rate source": p["source"], "YoY pairs": p["n_pairs"],
        "Rate, %/yr (95% CI)": [f"{a:.2f} [{b:.2f}, {c:.2f}]" if np.isfinite(a) else "n/a"
                                for a, b, c in zip(p["rate_pct_per_yr"], p["rate_lo"], p["rate_hi"])],
        "Current level": [f"{x:.3f}" if np.isfinite(x) else "n/a" for x in p["level"]],
        "RUL, yr (95% CI)": [_fmt_rul(r, H) for _, r in p.iterrows()],
        "Status": p["status"]})
    return out.sort_values(["Rate source", "Cohort", "Plant"]).reset_index(drop=True)


def show(res: RULResult):
    try:
        from IPython.display import display
    except Exception:                                   # pragma: no cover
        display = print
    print(f"Threshold: {res.cfg.threshold:.2f} of the reference-year level; horizon H = {res.cfg.horizon:g} yr; "
          f"bootstrap B = {res.cfg.n_boot}; cohort 2: {res.cfg.cohort2}")
    print(TITLES["summary"]); display(res.summary)
    print(TITLES["agreement"]); display(res.agreement)
    print(TITLES["plants"]); display(plant_view(res))


# ----------------------------------------------------------------------------- figure
def plot_forest(res: RULResult, path: Optional[str] = None, sources: Optional[Sequence[str]] = None):
    """Per-plant RUL: point with bootstrap interval; censored plants as arrows to the right."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    H, cfg = res.cfg.horizon, res.cfg
    sources = list(sources or res.meta["sources"])
    P = res.plants[res.plants["cohort"].isin([1, 2])]
    P = P[~P["status"].map(_base_status).str.startswith(ST_NA)]
    if not len(P):
        return None
    # common plant order: cohort, then RUL of the first source (censored last)
    first = P[P["source"] == sources[0]].set_index("column")
    key = first["rul"].where(~first["censored"], np.inf).fillna(np.inf).clip(upper=1e6)
    order = (P.drop_duplicates("column").set_index("column")
             .assign(k=lambda d: key.reindex(d.index).fillna(np.inf).to_numpy())
             .sort_values(["cohort", "k"], ascending=[True, True]))
    cols = order.index.tolist()
    ypos = {c: i for i, c in enumerate(cols)}
    xmax = H * 1.18
    fig, axes = plt.subplots(1, len(sources), figsize=(4.3 * len(sources) + 1.6, 0.22 * len(cols) + 1.9),
                             sharey=True, squeeze=False)
    col_c1, col_c2, col_cens = "#1f4e79", "#c55a11", "#7f7f7f"
    for ax, src in zip(axes[0], sources):
        g = P[P["source"] == src].set_index("column")
        ax.axvspan(H, xmax, color="#f2f2f2", zorder=0)
        ax.axvline(H, color="#bfbfbf", lw=0.8, zorder=1)
        for c, r in g.iterrows():
            y = ypos[c]
            st = _base_status(r["status"])
            cond = bool(r["conditional"])
            colr = col_c2 if cond else col_c1
            mk = "s" if cond else "o"
            lo = r["rul_lo"] if np.isfinite(r["rul_lo"]) else H
            if "censored" in st:
                x0 = min(max(lo, 0.0), H)
                ax.annotate("", xy=(xmax * 0.995, y), xytext=(x0, y),
                            arrowprops=dict(arrowstyle="-|>", color=colr if cond else col_cens, lw=1.1,
                                            mutation_scale=9), zorder=3)
                ax.plot(x0, y, marker="|", color=colr if cond else col_cens, ms=7, zorder=3)
            elif st == ST_BELOW:
                ax.plot(0, y, marker="x", color="#a50021", ms=6, zorder=4)
            else:
                hi = r["rul_hi"]
                if np.isfinite(hi) and hi <= H:
                    ax.plot([lo, hi], [y, y], color=colr, lw=1.3, zorder=2)
                else:
                    ax.annotate("", xy=(xmax * 0.995, y), xytext=(lo, y),
                                arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.3, mutation_scale=9), zorder=2)
                ax.plot(min(r["rul"], H), y, marker=mk, color=colr, ms=5,
                        mfc="white" if cond else colr, zorder=4)
        n_c1 = int((order["cohort"] == 1).sum())
        if 0 < n_c1 < len(cols):
            ax.axhline(n_c1 - 0.5, color="black", lw=0.6, ls="--")
        ax.set_xlim(0, xmax)
        ax.set_title(src, fontsize=10)
        ax.set_xlabel("RUL, years")
        ax.text(H + (xmax - H) / 2, 0.995, f"> {H:g} yr", ha="center", va="top", fontsize=8,
                color="#595959", transform=ax.get_xaxis_transform())
        ax.grid(axis="x", alpha=0.3)
    ax0 = axes[0][0]
    ax0.set_yticks(range(len(cols)))
    ax0.set_yticklabels([order.loc[c, "plant"] for c in cols], fontsize=7)
    ax0.set_ylim(len(cols) - 0.4, -1.3)                 # cohort 1 on top
    for lab, c in zip(ax0.get_yticklabels(), cols):
        lab.set_color(col_c2 if int(order.loc[c, "cohort"]) == 2 else "black")
    handles = [Line2D([], [], color=col_c1, marker="o", lw=1.3, label="Cohort 1: RUL, 95% bootstrap CI"),
               Line2D([], [], color=col_cens, marker="|", lw=1.1, label=f"Censored: RUL > {H:g} yr / no significant degradation"),
               Line2D([], [], color="#a50021", marker="x", lw=0, label="Below threshold")]
    if (P["conditional"]).any():
        handles.insert(1, Line2D([], [], color=col_c2, marker="s", mfc="white", lw=1.3,
                                 label="Cohort 2: conditional, if the 2023/2022 change persists"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"Remaining useful life to {cfg.threshold:.0%} of the reference-year performance",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.08 if len(cols) > 20 else 0.12, 1, 0.97))
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------- export
def _latex(df: pd.DataFrame, caption: str, label: str) -> str:
    try:
        import pv_semisynth_eval as sse
        return sse.df_to_latex(df, caption, label)
    except Exception:
        body = df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.2f}")
        return ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
                "\\footnotesize\n" + body + "\\end{table}\n")


def export_latex(res: RULResult, out_dir: str = "results/rul") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    H = res.cfg.horizon
    note = (f" Threshold {res.cfg.threshold:.2f} of the reference-year level, horizon $H={H:g}$~yr, "
            f"{res.cfg.n_boot} bootstrap resamples. Censored plants are kept.")
    items = {
        "summary": (res.summary, TITLES["summary"] + "." + note, "tab:rul_summary"),
        "agreement": (res.agreement, TITLES["agreement"] + ".", "tab:rul_agreement"),
        "plants": (plant_view(res), TITLES["plants"] + "." + note +
                   (" Cohort 2: " + COND_NOTE + "." if res.cfg.cohort2 == "conditional" else ""), "tab:rul_plants"),
    }
    paths = {}
    for k, (t, cap, lab) in items.items():
        p = os.path.join(out_dir, f"rul_{k}.tex")
        tt = t.copy()
        for c in tt.columns:
            if not pd.api.types.is_numeric_dtype(tt[c]):
                tt[c] = (tt[c].astype(str).str.replace(" | ", "; ", regex=False)
                         .str.replace(">", "$>$", regex=False))   # text-mode '>' is not printable in OT1
        tt.columns = [str(c).replace(">", "$>$") for c in tt.columns]
        tex = _latex(tt, cap.replace(">", "$>$"), lab)
        with open(p, "w", encoding="utf-8") as f:
            f.write(tex)
        t.to_csv(os.path.join(out_dir, f"rul_{k}_table.csv"), index=False)
        paths[k] = p
    return paths


def zip_outputs(out_dir: str = "results/rul", fig_paths: Sequence[str] = (), name: str = "rul_results.zip") -> str:
    z = os.path.join(out_dir, name)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(out_dir)):
            if f.endswith((".tex", ".csv")):
                zf.write(os.path.join(out_dir, f), f)
        for f in fig_paths:
            if f and os.path.exists(f):
                zf.write(f, os.path.basename(f))
    return z
