"""
Semi-synthetic validation of the RUL algorithms (notebook section 20).

Independent, optional module: it reads the notebook objects D, C (pv_semisynth_gen format)
and, if available, the frozen STGNN models; nothing in sections 1-19 depends on it and no
other module is modified.  STGNN weights are frozen (checked by fingerprint).

Design
------
* Known degradation is injected into the REAL measured power of the plants with a
  multi-year record (cohort 1; cohort 2 is evaluated separately as the conditional case):
      P'_i(t) = P_i(t) * (1 + r_i * tau(t)) * s(t),      tau = years since the record start,
  r_i = plant-specific rate (a share of the plants, the others are controls with r_i = 0)
  plus, depending on the scenario, a common fleet rate or a fleet-wide step s(t):
      "plant"          plant-specific rates only;
      "plant+step"     + a fleet-wide step (default -5 % from 2023-01-01) that is NOT
                       degradation (equivalent to an irradiance-sensor change for ratio methods);
      "plant+common"   + a common degradation of all plants (default -0.5 %/yr) that IS
                       degradation and is part of the truth.
  P_exp, GHI and the validity masks are not changed.
* Every rate source of pv_rul is computed on the clean and on the modified data with the
  same code: PR fleet-adjusted, SPS b1, Kalman b1, STGNN eta (frozen weights, one forward
  pass of the whole record per realisation and seed).
* Paired protocol: the RUL of the INJECTED component is estimated from the ratio series
  S_modified / S_clean of each source (this cancels the unknown real degradation of the
  plant, as in the b1 check of section 10), with pv_rul.plant_rul (YoY rate, bootstrap
  interval, censoring).  The truth is computed from the injected factor without the step.
* Metrics per source and scenario: rate MAE / bias (p.p./yr), RUL MAE (years, finite truth
  and estimate), 95 % interval coverage, sensitivity (finite RUL when the true RUL <= H),
  specificity (censored when the true RUL > H, including controls), balanced accuracy.
  Paired comparisons of the sources: cluster bootstrap over plants + sign-flip permutation,
  Holm correction.
"""

from __future__ import annotations

import os
import time
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import pv_rul as rul

_SHORT_SC = {
    "Plant-specific degradation": "Plant only",
    "Plant-specific degradation + fleet-wide step (not degradation)": "Plant +\nfleet step",
    "Plant-specific + common fleet degradation": "Plant +\ncommon rate",
}

SCENARIO_TITLES = {
    "plant": "Plant-specific degradation",
    "plant+step": "Plant-specific degradation + fleet-wide step (not degradation)",
    "plant+common": "Plant-specific + common fleet degradation",
}
TITLES = {
    "overall": "Comparison of RUL algorithms on semi-synthetic data (cohort 1, all scenarios)",
    "comparison": "Comparison of RUL algorithms by scenario (cohort 1)",
    "by_rate": "Rate recovery by injected rate (cohort 1)",
    "cohort2": "Cohort 2 (conditional, one year pair): comparison of RUL algorithms",
    "paired": "Paired comparisons of RUL algorithms (cohort 1; Holm-adjusted)",
    "design": "Semi-synthetic RUL design",
}
PAIR_METRICS = {"abs_rate_err": ("absolute rate error, p.p./yr", "lower"),
                "correct": ("correct status (finite vs censored)", "higher")}


@dataclass
class RULSSConfig:
    rates: Sequence[float] = (-0.5, -1.0, -2.0, -4.0)    # %/yr, plant-specific (degraded plants)
    share_degraded: float = 0.4                         # degraded share of eligible plants per realisation
    scenarios: Sequence[str] = ("plant", "plant+step", "plant+common")
    step: float = -0.05
    step_start: str = "2023-01-01"
    common_rate: float = -0.5                           # %/yr
    cohorts: Sequence[int] = (1, 2)
    estimators: Sequence[str] = ("sps", "kalman")
    rul_boot: int = 500                                 # bootstrap of each plant RUL
    n_boot: int = 2000                                  # bootstrap of the comparison metrics
    n_perm: int = 2000
    alpha: float = 0.05
    seed_offset: int = 53000


@dataclass
class RULSSResult:
    rows: pd.DataFrame
    tables: Dict[str, pd.DataFrame]
    cfg: RULSSConfig
    rcfg: rul.RULConfig
    meta: Dict[str, object] = field(default_factory=dict)


# ----------------------------------------------------------------------------- design
def _cohorts(D) -> pd.DataFrame:
    ca = D.get("ca")
    if ca is None or not hasattr(ca, "cohorts"):
        raise ValueError("D['ca'] (pv_cohorts.CohortAnalysis) is required")
    return ca.cohorts


def plan_roles(D, cfg: RULSSConfig, placement: int) -> pd.DataFrame:
    """Plant-specific injected rate of every eligible plant for one placement seed."""
    coh = _cohorts(D)
    rng = np.random.default_rng(cfg.seed_offset + placement)
    rows = []
    for k in cfg.cohorts:
        cols = [c for c in coh.index[coh["cohort"] == k] if c in D["pv_cols"]]
        cols = list(rng.permutation(cols))
        n_deg = int(round(cfg.share_degraded * len(cols)))
        rates = [cfg.rates[i % len(cfg.rates)] for i in range(n_deg)]
        rng.shuffle(rates)
        for i, c in enumerate(cols):
            r = rates[i] if i < n_deg else 0.0
            rows.append(dict(placement=placement, column=c, plant=rul._short(c), cohort=k,
                             role="degraded" if i < n_deg else "control", r_plant=r))
    return pd.DataFrame(rows)


def _tau(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx - idx[0]).total_seconds().to_numpy() / (365.25 * 86400)


def factors(D, roles: pd.DataFrame, scenario: str, cfg: RULSSConfig):
    """(total factor incl. step, degradation-only factor) per modified plant: {col: (f, f_deg)}."""
    idx = D["df"].index
    tau = _tau(idx)
    common = cfg.common_rate / 100 if scenario == "plant+common" else 0.0
    step = np.where(idx >= pd.Timestamp(cfg.step_start), 1 + cfg.step, 1.0) if scenario == "plant+step" else 1.0
    out = {}
    for _, r in roles.iterrows():
        f_deg = 1 + (r["r_plant"] / 100 + common) * tau
        out[r["column"]] = (f_deg * step, f_deg)
    return out


def apply(D, fac: Dict[str, tuple]) -> pd.DataFrame:
    df = D["df"].copy()
    for c, (f, _) in fac.items():
        if not np.allclose(f, 1.0):
            df[c] = df[c].to_numpy(float) * f
    return df


def truth(D, roles: pd.DataFrame, fac: Dict[str, tuple], valid_cov: pd.DataFrame, covered: pd.DataFrame,
          rcfg: rul.RULConfig) -> pd.DataFrame:
    """True rate (YoY of the degradation factor, %/yr), level and RUL of the injected component."""
    rows = []
    for _, r in roles.iterrows():
        c = r["column"]
        f_deg = fac[c][1]
        m = pd.Series(np.where(valid_cov[c].to_numpy(bool), f_deg, np.nan), index=D["df"].index)
        fm = m.resample("MS").mean().where(covered[c].reindex(m.resample("MS").mean().index, fill_value=False))
        yo, _ = rul._yoy(fm, 12)
        ref, last, y0, y1 = rul._level(fm)
        if not len(yo) or not len(ref) or not len(last):
            rows.append(dict(column=c, r_true=np.nan, c_true=np.nan, rul_true=np.nan)); continue
        rt = float(np.median(yo))
        ct = float(np.median(last) / np.median(ref))
        rt_ = rt if abs(rt) > 1e-9 else 0.0
        rl = (ct - rcfg.threshold) / (-rt_) if rt_ < 0 else np.inf
        rows.append(dict(column=c, r_true=rt * 100, c_true=ct, rul_true=max(rl, 0.0)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- estimation
class _Estimators:
    """Clean estimator outputs are computed once; modified ones per realisation."""

    def __init__(self, D, C, models, cfg: RULSSConfig, rcfg: rul.RULConfig, kalman_cfg=None):
        import pv_cohorts as pvc
        self.D, self.C, self.models, self.cfg, self.rcfg, self.kcfg = D, C, models, cfg, rcfg, kalman_cfg
        pv = list(D["pv_cols"])
        self.valid_cov = pvc.coverage_mask(D["df"], D["phys"], pv, D.get("observed"), D.get("act"))
        self.ca = D["ca"]
        self.covered = self.ca.cov_m >= rcfg.min_cov
        self._b1_clean: Dict[tuple, pd.DataFrame] = {}
        self.clean = self.sources(D["df"], changed=None)

    def _b1(self, name, df, col):
        from pv_semisynth_eval import _est_one
        per = _est_one(name, self.D, self.C, df, self.D["phys"], col, self.kcfg)[0]
        return per

    def per_table(self, name, df, changed) -> pd.DataFrame:
        parts = []
        for col in self.D["pv_cols"]:
            if col not in self.ca.cohorts.index or int(self.ca.cohorts.loc[col, "cohort"]) not in (1, 2):
                continue
            k = (name, col)
            if changed is None or col not in changed:
                if k not in self._b1_clean:
                    self._b1_clean[k] = self._b1(name, self.D["df"], col)
                parts.append(self._b1_clean[k])
            else:
                parts.append(self._b1(name, df, col))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def sources(self, df, changed: Optional[set]) -> Dict[str, tuple]:
        import pv_cohorts as pvc
        pr_m = pvc.monthly_pr(df, self.D["phys"], self.valid_cov)
        per_sps = self.per_table("sps", df, changed) if "sps" in self.cfg.estimators else None
        per_kf = self.per_table("kalman", df, changed) if "kalman" in self.cfg.estimators else None
        eta = None
        if self.models:
            eta = rul.stgnn_eta_hourly(self.models, df, self.D["phys"], self.rcfg).get("eta")
        return rul.source_series(self.ca, per_sps, per_kf, eta, self.rcfg, pr_m=pr_m)


def _ratio(S1: pd.DataFrame, S0: pd.DataFrame) -> pd.DataFrame:
    S1, S0 = S1.align(S0, join="inner")
    return (S1 / S0).where((S0 > 0) & (S1 > 0))


def _cls(status: str) -> str:
    s = rul._base_status(str(status))
    if s.startswith(rul.ST_NA):
        return "n/a"
    return "censored" if "censored" in s else "finite"


def run_rul_semisynth(D, C, models=None, placement_seeds: Sequence[int] = (0, 1, 2),
                      cfg: Optional[RULSSConfig] = None, rcfg: Optional[rul.RULConfig] = None,
                      kalman_cfg=None, out_dir: Optional[str] = "results/rul_semisynth",
                      verbose: bool = True) -> RULSSResult:
    cfg = cfg or RULSSConfig()
    rcfg = rcfg or rul.RULConfig(n_boot=cfg.rul_boot)
    t0 = time.time()
    fps = {s: m.fingerprint() for s, m in (models or {}).items()}
    df0 = D["df"].copy()
    est = _Estimators(D, C, models, cfg, rcfg, kalman_cfg)
    rng = np.random.default_rng(cfg.seed_offset)
    rows = []
    for ps in placement_seeds:
        roles = plan_roles(D, cfg, ps)
        for sc in cfg.scenarios:
            fac = factors(D, roles, sc, cfg)
            changed = {c for c, (f, _) in fac.items() if not np.allclose(f, 1.0)}
            df1 = apply(D, fac)
            inj = est.sources(df1, changed)
            tr = truth(D, roles, fac, est.valid_cov, est.covered, rcfg).set_index("column")
            for src, (S1, lag, mp) in inj.items():
                S0 = est.clean[src][0]
                R = _ratio(S1, S0)
                for _, r in roles.iterrows():
                    c = r["column"]
                    base = dict(placement=ps, scenario=sc, source=src, column=c, plant=r["plant"],
                                cohort=int(r["cohort"]), role=r["role"], r_plant=r["r_plant"],
                                **tr.loc[c].to_dict())
                    if c not in R.columns:
                        rows.append(dict(base, status=rul.ST_NA)); continue
                    o = rul.plant_rul(R[c], lag, rcfg, mp, rng)
                    rows.append(dict(base, **{k: o[k] for k in ("status", "n_pairs", "rate_pct_per_yr", "rate_lo",
                                                                  "rate_hi", "level", "rul", "rul_lo", "rul_hi")}))
            if verbose:
                print(time.strftime("%H:%M:%S"), f"[rul-semisynth] placement {ps}, {sc}: {len(changed)} plants changed, "
                      f"{len(inj)} sources", flush=True)
    assert D["df"].equals(df0), "clean data modified"
    for s, m in (models or {}).items():
        assert m.fingerprint() == fps[s], "STGNN weights changed"
    R = _derive(pd.DataFrame(rows), rcfg)
    res = RULSSResult(rows=R, tables={}, cfg=cfg, rcfg=rcfg,
                      meta=dict(placements=list(placement_seeds), sources=list(est.clean),
                                stgnn_seeds=list(fps), weights_unchanged=True, seconds=round(time.time() - t0)))
    res.tables = build_tables(res)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        R.to_csv(os.path.join(out_dir, "rul_ss_rows.csv"), index=False)
        for k, t in res.tables.items():
            t.to_csv(os.path.join(out_dir, f"rul_ss_{k}.csv"), index=False)
    return res


def _derive(R: pd.DataFrame, rcfg: rul.RULConfig) -> pd.DataFrame:
    H = rcfg.horizon
    R = R.copy()
    R["truth_cls"] = np.where(R["rul_true"] <= H, "finite", "censored")
    R["est_cls"] = R["status"].map(_cls)
    R = R[R["est_cls"] != "n/a"].copy()
    R["rate_err"] = R["rate_pct_per_yr"] - R["r_true"]
    R["abs_rate_err"] = R["rate_err"].abs()
    R["correct"] = (R["truth_cls"] == R["est_cls"]).astype(float)
    both = (R["truth_cls"] == "finite") & (R["est_cls"] == "finite")
    R["abs_rul_err"] = np.where(both, (R["rul"].clip(upper=H) - R["rul_true"]).abs(), np.nan)
    lo, hi = R["rul_lo"].fillna(np.inf), R["rul_hi"].fillna(np.inf)
    R["covered"] = np.where(R["truth_cls"] == "finite", (lo <= R["rul_true"]) & (R["rul_true"] <= hi), hi > H).astype(float)
    return R


# ----------------------------------------------------------------------------- tables
def _boot_ci(g: pd.DataFrame, col: str, n_boot: int, seed: int = 0, cl: str = "column"):
    v = g[[cl, col]].dropna()
    if not len(v):
        return np.nan, np.nan, np.nan
    grp = v.groupby(cl)[col].agg(["sum", "count"])
    s, n = grp["sum"].to_numpy(), grp["count"].to_numpy()
    rng = np.random.default_rng(seed)
    k = len(s)
    idx = rng.integers(0, k, (n_boot, k))
    b = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(s.sum() / n.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def _metrics(g: pd.DataFrame, n_boot: int) -> dict:
    fin, cen = g[g["truth_cls"] == "finite"], g[g["truth_cls"] == "censored"]
    m, lo, hi = _boot_ci(g, "abs_rate_err", n_boot)
    sens = float((fin["est_cls"] == "finite").mean()) if len(fin) else np.nan
    spec = float((cen["est_cls"] == "censored").mean()) if len(cen) else np.nan
    return {"n": len(g), "Rate MAE, p.p./yr": m, "Rate MAE CI": f"[{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else "",
            "Rate bias, p.p./yr": float(g["rate_err"].mean()),
            "RUL MAE, yr": float(g["abs_rul_err"].mean()) if g["abs_rul_err"].notna().any() else np.nan,
            "CI coverage": float(g["covered"].mean()),
            "Sensitivity": sens, "Specificity": spec,
            "Balanced accuracy": float(np.nanmean([sens, spec])) if np.isfinite([sens, spec]).any() else np.nan}


def _paired(A: pd.DataFrame, B: pd.DataFrame, metric: str, n_boot: int, n_perm: int, seed: int = 0) -> dict:
    key = ["placement", "scenario", "column"]
    j = A.set_index(key)[[metric]].join(B.set_index(key)[[metric]], lsuffix="_a", rsuffix="_b", how="inner").dropna()
    if not len(j):
        return dict(n=0)
    d = (j[f"{metric}_a"] - j[f"{metric}_b"]).groupby(level="column").agg(["sum", "count"])
    s, n = d["sum"].to_numpy(), d["count"].to_numpy()
    est = float(s.sum() / n.sum())
    rng = np.random.default_rng(seed)
    k = len(s)
    idx = rng.integers(0, k, (n_boot, k))
    b = s[idx].sum(1) / n[idx].sum(1)
    sg = rng.choice([-1.0, 1.0], (n_perm, k))
    perm = (sg * s).sum(1) / n.sum()
    p = float((np.sum(np.abs(perm) >= abs(est) - 1e-12) + 1) / (n_perm + 1))
    return dict(n=int(n.sum()), plants=k, diff=est, lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975)), p=p)


def build_tables(res: RULSSResult) -> Dict[str, pd.DataFrame]:
    R, cfg = res.rows, res.cfg
    c1, c2 = R[R["cohort"] == 1], R[R["cohort"] == 2]
    T = {}
    T["design"] = pd.DataFrame([
        {"Item": "Scenarios", "Value": "; ".join(SCENARIO_TITLES[s] for s in cfg.scenarios)},
        {"Item": "Plant-specific rates, %/yr", "Value": ", ".join(f"{r:g}" for r in cfg.rates) + " (controls: 0)"},
        {"Item": "Degraded share of plants", "Value": f"{cfg.share_degraded:.0%}"},
        {"Item": "Fleet-wide step (not degradation)", "Value": f"{cfg.step:+.0%} from {cfg.step_start}"},
        {"Item": "Common fleet rate", "Value": f"{cfg.common_rate:g} %/yr"},
        {"Item": "Threshold / horizon", "Value": f"{res.rcfg.threshold:.2f} / {res.rcfg.horizon:g} yr"},
        {"Item": "Placements", "Value": ", ".join(map(str, res.meta.get("placements", [])))},
        {"Item": "Rate sources", "Value": ", ".join(res.meta.get("sources", []))},
    ])
    T["overall"] = pd.DataFrame([dict(Source=s, **_metrics(g, cfg.n_boot)) for s, g in c1.groupby("source", sort=False)])
    T["comparison"] = pd.DataFrame([dict(Source=s, Scenario=SCENARIO_TITLES.get(sc, sc), **_metrics(g, cfg.n_boot))
                                    for (s, sc), g in c1.groupby(["source", "scenario"], sort=False)])
    rows = []
    for (s, r), g in c1.groupby(["source", "r_plant"], sort=False):
        rows.append({"Source": s, "Injected plant rate, %/yr": r, "n": len(g),
                     "True rate (median), %/yr": float(g["r_true"].median()),
                     "Estimated rate (median), %/yr": float(g["rate_pct_per_yr"].median()),
                     "Rate bias, p.p./yr": float(g["rate_err"].mean()),
                     "Rate MAE, p.p./yr": float(g["abs_rate_err"].mean()),
                     "Share finite RUL": float((g["est_cls"] == "finite").mean())})
    T["by_rate"] = pd.DataFrame(rows).sort_values(["Source", "Injected plant rate, %/yr"]).reset_index(drop=True)
    T["cohort2"] = (pd.DataFrame([dict(Source=s, **_metrics(g, cfg.n_boot)) for s, g in c2.groupby("source", sort=False)])
                    if len(c2) else pd.DataFrame())
    # paired
    srcs = list(c1["source"].unique())
    prs = []
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            A, B = c1[c1["source"] == srcs[i]], c1[c1["source"] == srcs[j]]
            for met, (lab, better) in PAIR_METRICS.items():
                t = _paired(A, B, met, cfg.n_boot, cfg.n_perm)
                if t.get("n"):
                    prs.append({"A": srcs[i], "B": srcs[j], "Metric": lab, "better": better, "n": t["n"],
                                "Plants": t["plants"], "A - B": t["diff"], "CI low": t["lo"], "CI high": t["hi"], "p": t["p"]})
    P = pd.DataFrame(prs)
    if len(P):
        try:
            from pv_metrics_stats import holm
        except Exception:                                               # pragma: no cover
            def holm(p):
                v = p.sort_values(); m = len(v)
                adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * v.to_numpy()))
                return pd.Series(adj, index=v.index).reindex(p.index)
        P["p (Holm)"] = holm(P["p"])

        def verdict(r):
            if not (r["p (Holm)"] < cfg.alpha) or (r["CI low"] <= 0 <= r["CI high"]):
                return "no difference shown"
            a_better = (r["A - B"] < 0) if r["better"] == "lower" else (r["A - B"] > 0)
            return "A better" if a_better else "B better"
        P["Verdict"] = P.apply(verdict, axis=1)
        P = P.drop(columns="better")
    T["paired"] = P
    return T


# ----------------------------------------------------------------------------- display / figures / export
def show(res: RULSSResult):
    try:
        from IPython.display import display
    except Exception:                                   # pragma: no cover
        display = print
    print(f"placements: {res.meta['placements']} | sources: {res.meta['sources']} | "
          f"STGNN seeds: {res.meta['stgnn_seeds']} | weights unchanged: {res.meta['weights_unchanged']} | "
          f"{res.meta['seconds']} s")
    for k in ("design", "overall", "comparison", "by_rate", "cohort2", "paired"):
        if len(res.tables.get(k, [])):
            print(TITLES[k]); display(res.tables[k].round(3))


def plot_all(res: RULSSResult, fig_dir: Optional[str] = "results/figures") -> Dict[str, object]:
    import matplotlib.pyplot as plt
    R = res.rows[res.rows["cohort"] == 1]
    srcs = list(R["source"].unique())
    figs = {}
    col = {"plant": "#1f4e79", "plant+step": "#c55a11", "plant+common": "#548235"}
    fig, axes = plt.subplots(1, len(srcs), figsize=(3.3 * len(srcs), 3.4), sharex=True, sharey=True, squeeze=False)
    lim = min(-1.0, float(np.nanmin(R[["r_true", "rate_pct_per_yr"]].to_numpy())) - 0.5)
    hi = max(1.0, float(np.nanmax(R[["r_true", "rate_pct_per_yr"]].to_numpy())) + 0.5)
    for ax, s in zip(axes[0], srcs):
        g = R[R["source"] == s]
        for sc, gg in g.groupby("scenario"):
            ax.errorbar(gg["r_true"], gg["rate_pct_per_yr"],
                        yerr=[(gg["rate_pct_per_yr"] - gg["rate_lo"]).clip(lower=0), (gg["rate_hi"] - gg["rate_pct_per_yr"]).clip(lower=0)],
                        fmt="o", ms=3, lw=0.6, alpha=0.7, color=col.get(sc, "grey"), label=SCENARIO_TITLES.get(sc, sc))
        ax.plot([lim, hi], [lim, hi], color="black", lw=0.8, ls="--")
        ax.set_title(s, fontsize=9); ax.set_xlabel("True rate, %/yr"); ax.grid(alpha=0.3)
        ax.set_xlim(lim, hi); ax.set_ylim(lim, hi)
    axes[0][0].set_ylabel("Estimated rate, %/yr")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    figs["rates"] = fig
    C = res.tables["comparison"]
    mets = ["Rate MAE, p.p./yr", "CI coverage", "Balanced accuracy"]
    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 3.4))
    scs = list(C["Scenario"].unique())
    w = 0.8 / max(len(srcs), 1)
    pal = ["#1f4e79", "#c55a11", "#548235", "#7030a0", "#7f7f7f"]
    for a, m in zip(ax2, mets):
        for i, s in enumerate(srcs):
            v = [C[(C["Source"] == s) & (C["Scenario"] == sc)][m].mean() for sc in scs]
            a.bar(np.arange(len(scs)) + i * w - 0.4 + w / 2, v, w, label=s, color=pal[i % len(pal)])
        a.set_xticks(range(len(scs)))
        a.set_xticklabels([_SHORT_SC.get(sc, sc) for sc in scs], fontsize=8)
        a.set_title(m, fontsize=9); a.grid(axis="y", alpha=0.3)
        if m != "Rate MAE, p.p./yr":
            a.set_ylim(0, 1.05)
            if m == "CI coverage":
                a.axhline(0.95, color="black", lw=0.7, ls="--")
    h, l = ax2[0].get_legend_handles_labels()
    fig2.legend(h, l, loc="lower center", ncol=len(srcs), fontsize=8, frameon=False)
    fig2.tight_layout(rect=(0, 0.1, 1, 1))
    figs["summary"] = fig2
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        for k, f in figs.items():
            f.savefig(os.path.join(fig_dir, f"fig_rul_ss_{k}.png"), dpi=200, bbox_inches="tight")
    return figs


def export_latex(res: RULSSResult, out_dir: str = "results/rul_semisynth") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for k, t in res.tables.items():
        if not len(t):
            continue
        tt = t.copy()
        for c in tt.columns:
            if not pd.api.types.is_numeric_dtype(tt[c]):
                tt[c] = tt[c].astype(str).str.replace(">", "$>$", regex=False)
        tex = rul._latex(tt, TITLES[k] + ".", f"tab:rul_ss_{k}")
        p = os.path.join(out_dir, f"rul_ss_{k}.tex")
        with open(p, "w", encoding="utf-8") as f:
            f.write(tex)
        paths[k] = p
    return paths


def zip_outputs(out_dir: str = "results/rul_semisynth", fig_dir: str = "results/figures",
                name: str = "rul_semisynth_results.zip") -> str:
    z = os.path.join(out_dir, name)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(out_dir)):
            if f.endswith((".tex", ".csv")):
                zf.write(os.path.join(out_dir, f), f)
        if fig_dir and os.path.isdir(fig_dir):
            for f in sorted(os.listdir(fig_dir)):
                if f.startswith("fig_rul_ss_"):
                    zf.write(os.path.join(fig_dir, f), f)
    return z
