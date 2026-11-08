"""Section 18: semi-synthetic CONFIGURATION CHANGES — permanent disconnection of part of the
modules and permanent addition of new modules (independent generator + evaluation).

Sections 1-17 are untouched: this module only reads ``D``, ``C``, the optional inverter data
``M`` (section 15) and the optional frozen STGNN checkpoints (section 10); it writes its own
realisations, caches and tables.  The notebook runs without it.

Scenarios (applied to a COPY of the real HKUST data, test period only, from the change time t0
to the end of the data; the plant metadata — rated power, P_exp — are NOT updated, as in real
monitoring before the operator edits the asset register):

===========  ======  =====  ==================================================================
scenario     class   alarm  what changes
===========  ======  =====  ==================================================================
module_loss  plant   yes    a fraction f of the DC capacity is disconnected (strings / modules
                            removed or failed open): affected inverters P_k x (1 - g); site P
                            reduced by the same amount; DC voltage unchanged (parallel strings)
module_add   config  yes    new strings with a DC capacity f are connected to the existing
                            inverters: P_k x (1 + g), limited by the inverter AC limit
                            L_k = headroom x q99.9(P_k, training) -> more midday clipping
===========  ======  =====  ==================================================================

f in {5, 15, 40 %} of the plant DC capacity.  On a multi-inverter site the change is placed on
the smallest number of inverters k such that each of them changes by g = f n / k <= 50 % of its
strings (string work is done per inverter).  Sites without inverter data: the site power is
scaled (the AC limit is estimated from the site power).

"alarm = yes" for module_add means "a configuration-change notice is expected": P > P_exp for
the rest of the life of the plant because the metadata are stale; it is NOT a fault.

Detection is credited only in the ONSET window [t0, t0 + onset_days] (+ pad); afterwards the
module reports the alarm-hour share in the windows 7-30 d and 30-90 d after the change minus
the share on the clean data (same hours): a detector with a fixed reference (PR) keeps alarming,
a detector with a rolling reference (PR+, STGNN with eta_hat med7d) adapts.

b1: the quarterly SPS and Kalman estimates on the modified data vs the clean data, compared
with the exact expected change (energy ratio, ``pv_semisynth_eval._ratio_change``); for the
Kalman filter also the adaptation lag of the FILTERED daily b1 (days until 90 % of the step).

Usage::

    import pv_semisynth_config as psc
    cres = psc.run_config(D, C, M=M_inv, models=ss_models)      # models / M optional
    psc.show(cres); psc.plot_all(cres); psc.export_latex(cres)
"""
from __future__ import annotations

import math
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

IRR = "Irradiance (W/m2)"
CONFIG_SCENARIOS = {
    "module_loss": dict(cls="plant", alarm=True, title="Disconnection of part of the modules (permanent)"),
    "module_add": dict(cls="config", alarm=True,
                       title="Addition of new modules (permanent; rated power not updated)"),
}
METHOD_LABEL = {"pr": "PR", "prp": "PR+", "prp_modal": "PR+ & modal rules", "stgnn": "STGNN (S4, daylight)"}
COLORS = {"pr": "#ff7f0e", "prp": "#9467bd", "prp_modal": "#8c564b", "stgnn": "#2ca02c"}
EST_LABEL = {"sps": "SPS", "kalman": "Kalman filter"}
TITLES = {
    "scenarios": "Configuration-change scenarios",
    "events": "Generated configuration changes",
    "detection": "Detection of configuration changes in the onset window (reference budget)",
    "persistence": "Alarm-hour share after the change minus clean data (reference budget)",
    "cause": "Attributed cause of detected configuration changes (reference budget)",
    "paired": "Paired comparisons of detectors on the same configuration changes",
    "b1": "Quarterly b1 response to configuration changes: SPS vs Kalman filter",
    "b1_paired": "Paired comparison of the b1 estimators on the same configuration changes",
    "budget": "Recall in the onset window vs clean alarm budget",
}


def _log(*a):
    print(time.strftime("%H:%M:%S"), "[config]", *a, flush=True)


def _short(c: str) -> str:
    return str(c).replace("power(W)_", "")


# ============================================================================ generator
@dataclass
class ConfigGenConfig:
    boundary: str = "2023-01-01"
    strengths: Tuple[float, ...] = (0.05, 0.15, 0.40)    # fraction of the plant DC capacity
    n_per_scenario: int = 6                             # per placement seed
    pre_days: int = 35                                  # scored days before t0 (> PR+ rolling window)
    post_days: int = 90                                 # data days after t0 (persistence windows)
    onset_days: int = 7                                 # detection is credited in [t0, t0 + onset]
    warmup_days: int = 10
    day_ghi: float = 50.0
    inv_headroom: float = 1.10                          # AC limit = headroom x q99.9 of training P
    max_inv_share: float = 0.5                          # <= 50 % of the strings of one inverter
    max_cluster_share: float = 0.2                      # changed plants per cluster and realisation
                                                        # <= max(1, floor(share x cluster size))
    seed_offset: int = 47_000


@dataclass
class ConfigPlan:
    labels: pd.DataFrame
    cfg: ConfigGenConfig
    seed: int


@dataclass
class ConfigRealisation:
    """Same interface as ``pv_semisynth_modal_gen.ModalRealisation`` (df, phys, modal, labels)."""
    df: pd.DataFrame
    phys: Dict[str, np.ndarray]
    modal: "object"
    labels: pd.DataFrame
    inj_mask: pd.DataFrame
    drift: Optional[dict] = None
    seed: int = 0


def _gen_cfg_modal(cfg: ConfigGenConfig):
    import pv_semisynth_modal_gen as smg
    return smg.ModalGenConfig(boundary=cfg.boundary, warmup_days=cfg.warmup_days, day_ghi=cfg.day_ghi)


def _cluster_labels(C: dict, pv: Sequence[str]) -> np.ndarray:
    t = C.get("tech") if C else None
    if t is not None and getattr(t, "labels", None) is not None:
        return t.labels.reindex(pv).fillna(-1).to_numpy()
    return np.zeros(len(pv))


def plan_config(D: dict, C: dict, M=None, seed: int = 0, cfg: Optional[ConfigGenConfig] = None) -> ConfigPlan:
    """One permanent change per plant; plants of different technical clusters share a realisation
    (``group``) with at most max(1, floor(max_cluster_share x cluster size)) changed plants per cluster."""
    import pv_semisynth_modal_gen as smg
    cfg = cfg or ConfigGenConfig()
    rng = np.random.default_rng(cfg.seed_offset + seed)
    mcfg = _gen_cfg_modal(cfg)
    pv = list(D["pv_cols"]); lab = dict(zip(pv, _cluster_labels(C, pv)))
    sites = set(M.sites()) if M is not None else set()
    days = {}
    for c in pv:
        ds = smg._candidate_days(D, M if c in sites else None, c, mcfg)
        if len(ds) < 2:
            continue
        lo = ds[0] + pd.Timedelta(days=cfg.pre_days); hi = ds[-1] - pd.Timedelta(days=cfg.post_days)
        ok = [d for d in ds if lo <= d <= hi]
        if ok:
            days[c] = ok
    pool = sorted(days)
    order = [(s, f) for f in cfg.strengths for s in CONFIG_SCENARIOS]       # balanced scenario x strength
    jobs = [order[i % len(order)] for i in range(cfg.n_per_scenario * len(CONFIG_SCENARIOS))]
    rows = []
    for eid, (scn, f) in enumerate(jobs):
        if not pool:
            break
        c = pool.pop(int(rng.integers(len(pool))))
        d0 = days[c][int(rng.integers(len(days[c])))]
        rows.append(dict(event_id=eid, scenario=scn, column=c, plant=_short(c), strength=float(f), start=d0,
                         cluster=lab.get(c, -1), has_inverters=c in sites))
    L = pd.DataFrame(rows, columns=["event_id", "scenario", "column", "plant", "strength", "start", "cluster",
                                    "has_inverters"])
    # groups (realisations): few changed plants per technical cluster, so that the cluster
    # peers used by PR+ stay (mostly) unchanged
    size = pd.Series(list(lab.values())).value_counts()
    cap = {k: max(1, int(np.floor(cfg.max_cluster_share * v))) for k, v in size.items()}
    grp, used = [], []
    for r in L.itertuples():
        g = 0
        while g < len(used) and used[g].get(r.cluster, 0) >= cap.get(r.cluster, 1):
            g += 1
        if g == len(used):
            used.append({})
        used[g][r.cluster] = used[g].get(r.cluster, 0) + 1; grp.append(g)
    L["group"] = grp
    L["cls"] = L["scenario"].map(lambda s: CONFIG_SCENARIOS[s]["cls"])
    L["expected_alarm"] = L["scenario"].map(lambda s: CONFIG_SCENARIOS[s]["alarm"])
    return ConfigPlan(labels=L, cfg=cfg, seed=seed)


def _limit(x: np.ndarray, train: np.ndarray, headroom: float) -> float:
    v = x[train & np.isfinite(x)]
    return headroom * float(np.quantile(v, 0.999)) if len(v) >= 50 else np.inf


def _scale(x: np.ndarray, m: np.ndarray, scn: str, g: float, L: float) -> np.ndarray:
    y = x.copy(); ok = m & np.isfinite(x)
    if scn == "module_loss":
        y[ok] = x[ok] * (1 - g)
    else:
        y[ok] = np.minimum(x[ok] * (1 + g), np.maximum(L, x[ok]))   # never cut an existing value
    return y


def apply_config(D: dict, M, plan: ConfigPlan, group: int = 0) -> ConfigRealisation:
    """Inject the planned changes of one group into copies of the site frame and inverter channels."""
    cfg = plan.cfg
    rng = np.random.default_rng(cfg.seed_offset + 1000 + 97 * plan.seed + group)
    df = D["df"].copy(); M2 = M.copy() if M is not None else None
    idx = df.index; pv = list(D["pv_cols"]); P0 = D["df"][pv].to_numpy(float)
    train = np.asarray(idx < pd.Timestamp(cfg.boundary))
    lab = plan.labels[plan.labels["group"] == group].copy()
    rows = []
    for r in lab.itertuples():
        c = r.column; m = np.asarray(idx >= r.start)
        before = df[c].to_numpy(float).copy()
        ks = M2.inverters_of(c) if M2 is not None else []
        n_aff, g = 0, r.strength
        if ks:
            n = len(ks)
            k_aff = min(n, max(1, int(math.ceil(r.strength * n / cfg.max_inv_share - 1e-9))))
            g = min(r.strength * n / k_aff, 0.95)
            aff = [ks[i] for i in rng.choice(n, k_aff, replace=False)]
            delta = np.zeros(len(idx))
            for k in aff:
                F = M2.inv[k]; pk = F["P"].reindex(idx).to_numpy(float)
                new = _scale(pk, m, r.scenario, g, _limit(pk, train, cfg.inv_headroom))
                d = np.nan_to_num(new - pk); delta += d
                h = m & np.isfinite(pk)
                F.loc[idx[h], "P"] = new[h]
            n_aff = k_aff
            ok = m & np.isfinite(before)
            df.loc[idx[ok], c] = np.maximum(before[ok] + delta[ok], 0.0)
        else:
            df[c] = _scale(before, m, r.scenario, g, _limit(before, train, cfg.inv_headroom))
        after = df[c].to_numpy(float)
        ch = np.flatnonzero(np.abs(np.nan_to_num(after - before)) > 1e-6)
        w30 = ch[idx[ch] < r.start + pd.Timedelta(days=30)] if len(ch) else ch
        s0, s1 = np.nansum(before[w30]), np.nansum(after[w30])
        eff = idx[ch[0]] if len(ch) else r.start
        rows.append(dict(event_id=r.event_id, eff_start=eff, inj_end=idx[ch[-1]] if len(ch) else r.start,
                         end=eff.normalize() + pd.Timedelta(days=cfg.onset_days) - pd.Timedelta(hours=1),
                         n_eff_h=int(len(ch)), n_inv_affected=n_aff, inv_change=g,
                         change_30d_pct=100 * (s1 / s0 - 1) if s0 > 0 else np.nan))
    lab = lab.merge(pd.DataFrame(rows, columns=["event_id", "eff_start", "inj_end", "end", "n_eff_h",
                                                "n_inv_affected", "inv_change", "change_30d_pct"]),
                    on="event_id", how="left")
    lab["effective"] = lab["n_eff_h"].fillna(0) > 0
    inj = np.abs(np.nan_to_num(df[pv].to_numpy(float) - P0)) > 1e-6
    return ConfigRealisation(df=df, phys=D["phys"], modal=M2, labels=lab.reset_index(drop=True),
                             inj_mask=pd.DataFrame(inj, index=idx, columns=pv), seed=plan.seed)


def realisations(D, C, M=None, placement_seeds: Sequence[int] = (0, 1, 2),
                 cfg: Optional[ConfigGenConfig] = None) -> Tuple[Dict[int, ConfigPlan], Dict[str, ConfigRealisation]]:
    plans = {ps: plan_config(D, C, M, ps, cfg) for ps in placement_seeds}
    reals = {}
    for ps, p in plans.items():
        for g in sorted(p.labels["group"].unique()):
            reals[f"c{ps}g{g}"] = apply_config(D, M, p, int(g))
    return plans, reals


def scenario_table() -> pd.DataFrame:
    return pd.DataFrame([dict(Scenario=k, Title=v["title"], Class=v["cls"],
                              **{"Alarm / notice expected": "yes" if v["alarm"] else "no"})
                         for k, v in CONFIG_SCENARIOS.items()])


# ============================================================================ sets without inverter data
def _sets_plain(D: dict, reals: dict, boundary: str, R=None) -> dict:
    """Sets of ``pv_modal_eval`` without modal features (when M is None)."""
    import pv_modal_eval as me
    pv = list(D["pv_cols"])
    rw = np.array([max(float(D["rated"].get(c, 0) or 0) * 1000.0, 1.0) for c in pv]) if isinstance(D.get("rated"), dict) \
        else np.nanmax(np.r_[D["df"][pv].to_numpy(float), np.ones((1, len(pv)))], 0)
    out = {}
    if R is None:
        idx = D["df"].index; b = pd.Timestamp(boundary); act = me._active_frame(D, pv)
        for key, df, phys in [("cal", D["df"], D["phys"]), ("clean", D["df"], D["phys"])] + \
                             [(k, r.df, r.phys) for k, r in reals.items()]:
            rows = np.asarray(idx < b) if key == "cal" else np.asarray(idx >= b)
            y = df[pv].to_numpy(float)[rows] / rw; a = np.asarray(phys["p_expected"], float)[rows] / rw
            ok = act[rows] & np.isfinite(y) & np.isfinite(a)
            out[key] = dict(ts=idx[rows], y=np.nan_to_num(y), a=np.nan_to_num(a), active=ok,
                            ghi=np.nan_to_num(df[IRR].to_numpy(float)[rows]), feats=None, raw=None, nr=None)
        return out
    n = R.calib["n_pv"]
    src = [("cal", R.cal, None, D["df"]), ("val", R.val, getattr(R, "val_not_ready", None), D["df"]),
           ("clean", R.clean, R.not_ready, D["df"])] + [(k, R.inj[k], R.not_ready, reals[k].df) for k in reals]
    for key, raw, nr, df in src:
        ts = pd.DatetimeIndex(raw["ts"]); nr2 = None if nr is None else np.asarray(nr)[:, :n]
        y = raw["y"].mean(-1)[:, :n].astype(float); a = raw["a"].mean(-1)[:, :n].astype(float)
        act = raw["active"][:, :n] & np.isfinite(y) & np.isfinite(a)
        if nr2 is not None:
            act &= ~nr2
        out[key] = dict(ts=ts, y=np.nan_to_num(y), a=np.nan_to_num(a), active=act,
                        ghi=np.nan_to_num(df[IRR].reindex(ts).to_numpy(float)), feats=None, raw=raw, nr=nr2)
    return out


# ============================================================================ evaluation config / result
@dataclass
class ConfigEvalConfig:
    budgets: Sequence[float] = (0.5, 1.0, 2.0)          # clean alarm events per 1000 scored hours
    ref_budget: float = 1.0
    windows: Tuple[Tuple[int, int], ...] = ((7, 30), (30, 90))
    pad_h: int = 24
    n_boot: int = 2000
    n_perm: int = 2000
    seed: int = 0
    estimators: Tuple[str, ...] = ("sps", "kalman")
    lag_share: float = 0.9


@dataclass
class ConfigResult:
    events: pd.DataFrame          # detector x budget x model seed x event
    points: pd.DataFrame          # operating points (q, clean rate)
    b1: pd.DataFrame              # estimator x event x quarter
    lags: pd.DataFrame            # Kalman adaptation lag per event
    labels: pd.DataFrame
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    examples: Dict[str, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


# ============================================================================ detectors
def _det_rows(det, key, q, real: ConfigRealisation, pv: List[str], tag: dict, cfg: ConfigEvalConfig) -> List[dict]:
    import pv_modal_eval as me
    S = det.sets[key]; ts = S["ts"]; em = det.mask(key, q); sc = det.scored(key)
    Sc = det.sets["clean"]; emc = det.mask("clean", q); scc = det.scored("clean")
    rows = []
    for r in real.labels.itertuples():
        if r.column not in pv or not r.effective:
            continue
        j = pv.index(r.column)
        w = (ts >= r.eff_start) & (ts <= r.end + pd.Timedelta(hours=cfg.pad_h))
        pos = np.flatnonzero(w)
        rec = dict(tag, event_id=r.event_id, scenario=r.scenario, strength=r.strength, column=r.column,
                   plant=r.plant, has_inverters=bool(r.has_inverters), cls=r.cls)
        scored = bool(len(pos) and sc[pos, j].any())
        hit = em[pos, j] if len(pos) else np.array([], bool)
        det_ = bool(scored and hit.any())
        rec.update(scored=scored, detected=det_,
                   ttd_h=float((ts[pos[np.argmax(hit)]] - r.eff_start) / pd.Timedelta(hours=1)) if det_ else np.nan,
                   pred=(me._attrib(det.rules, key, pos, j, em) if det.modal else "plant") if det_ else "none")
        for a, b in cfg.windows:
            ww = (ts >= r.eff_start + pd.Timedelta(days=a)) & (ts < r.eff_start + pd.Timedelta(days=b)) & \
                 (ts <= r.inj_end)
            p = np.flatnonzero(ww & sc[:, j])
            pc = Sc["ts"].get_indexer(ts[p]); pc = pc[pc >= 0]
            pc = pc[scc[pc, j]]
            rec[f"share_{a}_{b}"] = float(em[p, j].mean()) if len(p) else np.nan
            rec[f"share_clean_{a}_{b}"] = float(emc[pc, j].mean()) if len(pc) else np.nan
            rec[f"excess_{a}_{b}"] = rec[f"share_{a}_{b}"] - rec[f"share_clean_{a}_{b}"]
        rows.append(rec)
    return rows


def _detectors(D, C, M, models, reals, out_dir, use_cache, boundary, mcfg):
    """[(seed, {method: Det})] — rule-based methods on the hourly frame without models."""
    import pv_modal_eval as me
    pv = list(D["pv_cols"]); blocks = []
    if models:
        import pv_fixes_eval as fx
        raws = fx.collect(models, D, reals, os.path.join(out_dir, "raw"), None, 64, use_cache)
        for s, R in raws.items():
            ss = me.sets_from_raw(R, D, M, reals) if M is not None else _sets_plain(D, reals, boundary, R)
            blocks.append((s, ss, me.stgnn_fam(R, ss, mcfg)))
    else:
        ss = me.sets_from_frame(D, M, reals, boundary) if M is not None else _sets_plain(D, reals, boundary)
        blocks.append(("-", ss, None))
    out = []
    for s, ss, st in blocks:
        dets = {"pr": me.Det("pr", me.pr_fam(ss, C, pv, mcfg), None, ss, mcfg, False),
                "prp": me.Det("prp", me.prp_fam(ss, C, pv, mcfg), None, ss, mcfg, False)}
        if M is not None:
            dets["prp_modal"] = me.Det("prp_modal", dets["prp"].fam, me.make_rules(ss, mcfg, pv), ss, mcfg, True)
        if st is not None:
            dets["stgnn"] = me.Det("stgnn", st, None, ss, mcfg, False)
        out.append((s, dets))
    return out


# ============================================================================ b1 (SPS / Kalman)
def _est(name: str, D, C, df, phys, col, kalman_cfg=None):
    import pv_semisynth_eval as sse
    if name == "sps":
        per, _, _ = sse._est_one("sps", D, C, df, phys, col)
        return per, None
    import pv_kalman as pk
    j = list(D["pv_cols"]).index(col)
    ph = dict(p_expected=phys["p_expected"][:, [j]], ghi=phys["ghi"])
    tc = D["t_cell"][:, [j]] if np.ndim(D["t_cell"]) == 2 else D["t_cell"]
    args = (df, ph, [col], D["rated"], C["valid"][[col]], tc, sse._cloud(D))
    kc = kalman_cfg or pk.KalmanConfig()
    tr = pk.kalman_track(*args, kc)
    per = pk.kalman_periods(*args, cfg=kc, tracks=tr)
    daily = pk.tracks_frame(tr, False)                       # FILTERED (causal) daily b1
    return per, daily


def _b1_rows(D, C, real: ConfigRealisation, key, estimators, cache: dict, cfg: ConfigEvalConfig, kalman_cfg=None):
    import pv_semisynth_eval as sse
    rows, lags = [], []
    for r in real.labels[real.labels["effective"]].itertuples():
        col = r.column
        for name in estimators:
            if (name, col) not in cache:
                cache[(name, col)] = _est(name, D, C, D["df"], D["phys"], col, kalman_cfg)
            per0, d0 = cache[(name, col)]
            per1, d1 = _est(name, D, C, real.df, real.phys, col, kalman_cfg)
            if not len(per0) or not len(per1):
                continue
            p0 = per0.set_index("period")["theta_b1_Pexp"]; p1 = per1.set_index("period")["theta_b1_Pexp"]
            q0 = pd.Period(r.eff_start, "Q")
            for qs in p1.index:
                q = pd.Period(qs, "Q")
                if q < q0 or qs not in p0.index:
                    continue
                exp_ = sse._ratio_change(D, C, real.df, real.phys, col, q.start_time, q.end_time)
                est = (p1[qs] / p0[qs] - 1) * 100 if p0[qs] else np.nan
                rows.append(dict(estimator=name, placement=key, event_id=r.event_id, scenario=r.scenario,
                                 strength=r.strength, plant=r.plant, quarter=str(q), change_quarter=bool(q == q0),
                                 expected_b1_pct=exp_, estimated_b1_pct=est, error_pp=est - exp_))
            if name == "kalman" and d1 is not None and d0 is not None and len(d1):
                s0 = d0.set_index("day")["b1_Pexp"]; s1 = d1.set_index("day")["b1_Pexp"].reindex(s0.index)
                rel = (s1 / s0 - 1) * 100
                step = sse._ratio_change(D, C, real.df, real.phys, col, r.eff_start + pd.Timedelta(days=30),
                                         r.eff_start + pd.Timedelta(days=90))
                after = rel[rel.index >= r.eff_start.normalize()].dropna()
                lag = np.nan
                if np.isfinite(step) and abs(step) > 0.5 and len(after):
                    hit = after.index[np.sign(step) * after.to_numpy() >= cfg.lag_share * abs(step)]
                    lag = float((hit[0] - r.eff_start.normalize()).days) if len(hit) else np.inf
                lags.append(dict(placement=key, event_id=r.event_id, scenario=r.scenario, strength=r.strength,
                                 plant=r.plant, step_pct=step, lag_days=lag))
    return rows, lags


# ============================================================================ main
def run_config(D: dict, C: dict, M=None, models: Optional[dict] = None, placement_seeds: Sequence[int] = (0, 1, 2),
               cfg: Optional[ConfigEvalConfig] = None, gen_cfg: Optional[ConfigGenConfig] = None,
               modal_cfg=None, kalman_cfg=None, out_dir: str = "results/config", use_cache: bool = True,
               b1: bool = True) -> ConfigResult:
    """Detection (PR, PR+, PR+ & modal rules, frozen STGNN) at equal clean alarm budgets and the b1
    response of SPS and the Kalman filter.  ``M`` (section 15) and ``models`` (section 10) are optional."""
    import pv_modal_eval as me
    cfg = cfg or ConfigEvalConfig(); gen_cfg = gen_cfg or ConfigGenConfig()
    mcfg = modal_cfg or me.ModalEvalConfig()
    t0 = time.time(); os.makedirs(out_dir, exist_ok=True)
    pv = list(D["pv_cols"])
    plans, reals = realisations(D, C, M, placement_seeds, gen_cfg)
    labels = pd.concat([r.labels.assign(placement=k) for k, r in reals.items()], ignore_index=True)
    _log(f"{len(labels)} configuration changes in {len(reals)} realisations "
         f"({int(labels['effective'].sum())} effective)")
    budgets = sorted(set(float(b) for b in cfg.budgets) | {float(cfg.ref_budget)})
    grid = me._grid(mcfg)
    fps = {s: m.fingerprint() for s, m in (models or {}).items()}
    EV, PT = [], []
    for seed, dets in _detectors(D, C, M, models, reals, out_dir, use_cache, gen_cfg.boundary, mcfg):
        for m, det in dets.items():
            cache = {}
            for bud in budgets:
                q, rate = me.match_q(det, bud, grid, cache)
                tag = dict(method=m, model_seed=seed, budget=bud)
                PT.append(dict(tag, q=q, clean_rate=rate))
                for k, real in reals.items():
                    EV += _det_rows(det, k, q, real, pv, dict(tag, placement=k), cfg)
        _log(f"seed {seed}: {len(dets)} detectors ({time.time() - t0:.0f}s)")
    for s, m in (models or {}).items():
        assert m.fingerprint() == fps[s] and m.unchanged(), "weights changed"
    B1, LG = [], []
    if b1 and cfg.estimators:
        est_cache = {}
        for k, real in reals.items():
            r_, l_ = _b1_rows(D, C, real, k, cfg.estimators, est_cache, cfg, kalman_cfg)
            B1 += r_; LG += l_
        _log(f"b1 estimators {list(cfg.estimators)} done ({time.time() - t0:.0f}s)")
    res = ConfigResult(events=pd.DataFrame(EV), points=pd.DataFrame(PT), b1=pd.DataFrame(B1), lags=pd.DataFrame(LG),
                       labels=labels,
                       meta=dict(budgets=budgets, ref_budget=float(cfg.ref_budget), placements=list(reals),
                                 placement_seeds=list(placement_seeds), seeds=sorted(fps) or ["-"],
                                 neural=bool(models), modal=M is not None, n_boot=cfg.n_boot, n_perm=cfg.n_perm,
                                 windows=[list(w) for w in cfg.windows], onset_days=gen_cfg.onset_days,
                                 weights_unchanged=True, gen_cfg=gen_cfg.__dict__))
    res.examples = _examples(D, C, reals, labels)
    res.tables = build_tables(res, cfg)
    with open(os.path.join(out_dir, "config_result.pkl"), "wb") as f:
        pickle.dump(res, f)
    _log(f"done in {time.time() - t0:.0f}s")
    return res


def _examples(D, C, reals, labels) -> Dict[str, dict]:
    """Daily P / P_exp (clean vs modified) for the strongest change of each scenario."""
    out = {}
    pv = list(D["pv_cols"])
    for scn in CONFIG_SCENARIOS:
        L = labels[(labels["scenario"] == scn) & labels["effective"]]
        if not len(L):
            continue
        r = L.sort_values(["strength", "has_inverters"], ascending=False).iloc[0]
        real = reals[r["placement"]]; j = pv.index(r["column"])
        a, b = r["eff_start"] - pd.Timedelta(days=45), r["eff_start"] + pd.Timedelta(days=120)
        idx = D["df"].index; m = np.asarray((idx >= a) & (idx <= b))
        pe = np.asarray(D["phys"]["p_expected"], float)[m, j]
        day = pe > 0.05 * np.nanmax(np.r_[pe, 1.0])
        f = pd.DataFrame({"clean": np.where(day, D["df"][r["column"]].to_numpy(float)[m], np.nan),
                          "mod": np.where(day, real.df[r["column"]].to_numpy(float)[m], np.nan),
                          "pexp": np.where(day, pe, np.nan)}, index=idx[m])
        d = f.resample("D").sum(min_count=3)
        out[scn] = dict(plant=r["plant"], strength=float(r["strength"]), t0=r["eff_start"],
                        ratio_clean=(d["clean"] / d["pexp"]), ratio_mod=(d["mod"] / d["pexp"]),
                        placement=r["placement"], event_id=int(r["event_id"]))
    return out


# ============================================================================ statistics
def _per_event(X: pd.DataFrame, col: str) -> pd.Series:
    """Mean over model seeds per event cluster (realisation, event)."""
    return X.groupby(["placement", "event_id"])[col].mean()


def _boot(v: np.ndarray, n_boot: int, rng) -> Tuple[float, float]:
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan
    I = rng.integers(0, len(v), size=(n_boot, len(v)))
    return tuple(float(x) for x in np.quantile(v[I].mean(1), [0.025, 0.975]))


def _ci_str(est, lo, hi, d=3) -> str:
    return f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if np.isfinite(est) and np.isfinite(lo) else \
        (f"{est:.{d}f}" if np.isfinite(est) else "-")


def paired_diff(a: pd.Series, b: pd.Series, n_boot: int = 2000, n_perm: int = 2000, seed: int = 0) -> dict:
    """a - b on the same event clusters: bootstrap CI and a sign-flip (swap within event) p-value."""
    j = a.index.intersection(b.index)
    d = (a.loc[j] - b.loc[j]).to_numpy(float); d = d[np.isfinite(d)]
    if not len(d):
        return dict(n=0, diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    rng = np.random.default_rng(seed)
    lo, hi = _boot(d, n_boot, rng)
    obs = abs(d.mean())
    S = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    p = (1 + int((np.abs((S * d).mean(1)) >= obs - 1e-12).sum())) / (n_perm + 1)
    return dict(n=int(len(d)), a=float(a.loc[j].mean()), b=float(b.loc[j].mean()), diff=float(d.mean()),
                lo=lo, hi=hi, p=float(p))


def _holm(p: pd.Series) -> pd.Series:
    import pv_metrics_stats as pst
    return pst.holm(p)


def _verdict(lo, hi, ph, lower_better=False) -> str:
    if not (np.isfinite(lo) and np.isfinite(ph)):
        return "no data"
    if ph < 0.05 and (lo > 0 or hi < 0):
        a_better = (hi < 0) if lower_better else (lo > 0)
        return "A better" if a_better else "B better"
    return "no difference shown"


PAIRS = [("stgnn", "prp"), ("stgnn", "pr"), ("prp", "pr"), ("prp_modal", "prp")]


def build_tables(res: ConfigResult, cfg: Optional[ConfigEvalConfig] = None) -> Dict[str, pd.DataFrame]:
    cfg = cfg or ConfigEvalConfig()
    rng = np.random.default_rng(cfg.seed)
    T = {"scenarios": scenario_table()}
    L = res.labels
    if len(L):
        T["events"] = (L.groupby(["scenario", "strength"])
                       .agg(Events=("event_id", "size"), Effective=("effective", "sum"),
                            **{"With inverter data": ("has_inverters", "sum"),
                               "Median change in first 30 d, %": ("change_30d_pct", "median"),
                               "Median inverters affected": ("n_inv_affected", "median")})
                       .reset_index().rename(columns={"scenario": "Scenario", "strength": "Strength (DC share)"}))
    E = res.events
    ref = res.meta.get("ref_budget", cfg.ref_budget)
    if len(E):
        E = E[E["scored"]]
        X = E[E["budget"] == ref]
        rows, prow, crow = [], [], []
        for m in [k for k in METHOD_LABEL if k in set(X["method"])]:
            Xm = X[X["method"] == m]
            for scn in list(CONFIG_SCENARIOS) + ["all"]:
                Xs = Xm if scn == "all" else Xm[Xm["scenario"] == scn]
                for st in sorted(Xs["strength"].unique()) + ["all"]:
                    Y = Xs if st == "all" else Xs[Xs["strength"] == st]
                    if not len(Y) or (scn == "all" and st != "all"):
                        continue
                    v = _per_event(Y, "detected").to_numpy(float)
                    lo, hi = _boot(v, cfg.n_boot, rng)
                    rows.append({"Detector": METHOD_LABEL[m], "Scenario": scn, "Strength": st, "Events": len(v),
                                 "Recall (95% CI)": _ci_str(v.mean(), lo, hi),
                                 "Median TTD, h": float(Y.loc[Y["detected"], "ttd_h"].median())
                                 if Y["detected"].any() else np.nan})
            for scn in CONFIG_SCENARIOS:
                Y = Xm[Xm["scenario"] == scn]
                if not len(Y):
                    continue
                r = {"Detector": METHOD_LABEL[m], "Scenario": scn}
                for a, b in cfg.windows:
                    v = _per_event(Y, f"excess_{a}_{b}").to_numpy(float)
                    lo, hi = _boot(v, cfg.n_boot, rng)
                    r[f"Share {a}-{b} d"] = float(np.nanmean(_per_event(Y, f"share_{a}_{b}"))) if len(v) else np.nan
                    r[f"Excess {a}-{b} d (95% CI)"] = _ci_str(float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                                                             lo, hi)
                prow.append(r)
                D_ = Y[Y["detected"]]
                if len(D_) and m == "prp_modal":
                    vc = D_["pred"].value_counts(normalize=True)
                    crow.append({"Detector": METHOD_LABEL[m], "Scenario": scn, "Detected": len(D_),
                                 **{f"Share {c}": float(vc.get(c, 0.0)) for c in ("plant", "inverter", "grid", "data")}})
        T["detection"] = pd.DataFrame(rows)
        T["persistence"] = pd.DataFrame(prow)
        if crow:
            T["cause"] = pd.DataFrame(crow)
        # paired comparisons (pre-specified): recall in the onset window and excess alarms 30-90 d
        P = []
        a_, b_ = cfg.windows[-1]
        for A, B in PAIRS:
            if A not in set(X["method"]) or B not in set(X["method"]):
                continue
            for scn in list(CONFIG_SCENARIOS) + ["all"]:
                XA = X[(X["method"] == A) & ((X["scenario"] == scn) | (scn == "all"))]
                XB = X[(X["method"] == B) & ((X["scenario"] == scn) | (scn == "all"))]
                for metric, col, lb in (("Recall (onset)", "detected", False),
                                        (f"Excess alarm share {a_}-{b_} d", f"excess_{a_}_{b_}", True)):
                    d = paired_diff(_per_event(XA, col), _per_event(XB, col), cfg.n_boot, cfg.n_perm, cfg.seed)
                    if not d["n"]:
                        continue
                    P.append(dict(Comparison=f"{METHOD_LABEL[A]} vs {METHOD_LABEL[B]}", Scenario=scn, Metric=metric,
                                  Events=d["n"], A=d["a"], B=d["b"], **{"Delta (A - B)": d["diff"],
                                  "95% CI": f"[{d['lo']:.3f}, {d['hi']:.3f}]", "p": d["p"]}, _lo=d["lo"],
                                  _hi=d["hi"], _lb=lb))
        if P:
            Pd = pd.DataFrame(P); Pd["p (Holm)"] = _holm(Pd["p"]).to_numpy()
            Pd["Result"] = [_verdict(r._lo, r._hi, r["p (Holm)"], r._lb) for _, r in Pd.iterrows()]
            T["paired"] = Pd.drop(columns=["_lo", "_hi", "_lb"])
        # recall vs budget
        BR = []
        for (m, bud), Y in E.groupby(["method", "budget"]):
            for scn in CONFIG_SCENARIOS:
                v = _per_event(Y[Y["scenario"] == scn], "detected").to_numpy(float)
                if len(v):
                    BR.append(dict(Detector=METHOD_LABEL[m], method=m, Scenario=scn, Budget=bud, Recall=float(v.mean()),
                                   Events=len(v)))
        T["budget"] = pd.DataFrame(BR)
    B1 = res.b1
    if len(B1):
        rows = []
        for (name, scn, st), Y in B1.groupby(["estimator", "scenario", "strength"]):
            for cq in (False, True):
                Z = Y[Y["change_quarter"] == cq]
                if not len(Z):
                    continue
                v = Z.groupby(["placement", "event_id"])["error_pp"].apply(lambda s: np.nanmean(np.abs(s))).to_numpy()
                lo, hi = _boot(v, cfg.n_boot, rng)
                r = {"Estimator": EST_LABEL.get(name, name), "Scenario": scn, "Strength": st,
                     "Quarters": "change quarter" if cq else "full quarters after the change", "Events": len(v),
                     "Mean expected Δb1, %": float(Z["expected_b1_pct"].mean()),
                     "Mean estimated Δb1, %": float(Z["estimated_b1_pct"].mean()),
                     "MAE, p.p. (95% CI)": _ci_str(float(np.nanmean(v)), lo, hi, 2)}
                if name == "kalman" and len(res.lags) and not cq:
                    lg = res.lags[(res.lags["scenario"] == scn) & (res.lags["strength"] == st)]["lag_days"]
                    r["Median lag to 90% of step, d (Kalman)"] = float(lg.replace(np.inf, np.nan).median()) \
                        if lg.notna().any() else np.nan
                rows.append(r)
        T["b1"] = pd.DataFrame(rows)
        if set(B1["estimator"]) >= {"sps", "kalman"}:
            P = []
            for scn in list(CONFIG_SCENARIOS) + ["all"]:
                for cq in (False, True):
                    Z = B1[(B1["change_quarter"] == cq) & ((B1["scenario"] == scn) | (scn == "all"))]
                    if not len(Z):
                        continue
                    ae = lambda n: Z[Z["estimator"] == n].groupby(["placement", "event_id"])["error_pp"] \
                        .apply(lambda s: np.nanmean(np.abs(s)))
                    d = paired_diff(ae("kalman"), ae("sps"), cfg.n_boot, cfg.n_perm, cfg.seed)
                    if d["n"]:
                        P.append(dict(Comparison="Kalman filter vs SPS", Scenario=scn,
                                      Quarters="change quarter" if cq else "full quarters after the change",
                                      Metric="MAE Δb1, p.p.", Events=d["n"], A=d["a"], B=d["b"],
                                      **{"Delta (A - B)": d["diff"], "95% CI": f"[{d['lo']:.2f}, {d['hi']:.2f}]",
                                         "p": d["p"]}, _lo=d["lo"], _hi=d["hi"]))
            if P:
                Pd = pd.DataFrame(P); Pd["p (Holm)"] = _holm(Pd["p"]).to_numpy()
                Pd["Result"] = [_verdict(r._lo, r._hi, r["p (Holm)"], True) for _, r in Pd.iterrows()]
                T["b1_paired"] = Pd.drop(columns=["_lo", "_hi"])
    return T


# ============================================================================ output
def show(res: ConfigResult, digits: int = 3):
    try:
        from IPython.display import display
    except Exception:                                    # pragma: no cover
        display = print
    for k, t in TITLES.items():
        X = res.tables.get(k)
        if X is None or not len(X) or k == "budget":
            continue
        print(t)
        display(X.round(digits))


def plot_all(res: ConfigResult, fig_dir: Optional[str] = "results/figures") -> Dict[str, object]:
    import matplotlib.pyplot as plt
    figs = {}
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
    BR = res.tables.get("budget")
    if BR is not None and len(BR):
        fig, axs = plt.subplots(1, len(CONFIG_SCENARIOS), figsize=(11, 3.8), sharey=True)
        for ax, scn in zip(np.atleast_1d(axs), CONFIG_SCENARIOS):
            for m in METHOD_LABEL:
                Y = BR[(BR["method"] == m) & (BR["Scenario"] == scn)].groupby("Budget")["Recall"].mean()
                if len(Y):
                    ax.plot(Y.index, Y.values, "o-", color=COLORS[m], label=METHOD_LABEL[m])
            ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
            ax.set_title(CONFIG_SCENARIOS[scn]["title"], fontsize=9)
            ax.set_xlabel("Clean alarm budget, events / 1000 h")
        np.atleast_1d(axs)[0].set_ylabel(f"Recall in the onset window ({res.meta.get('onset_days', 7)} d)")
        np.atleast_1d(axs)[0].legend(fontsize=8)
        fig.tight_layout(); figs["recall_budget"] = fig
    E = res.events
    if len(E):
        X = E[(E["budget"] == res.meta["ref_budget"]) & E["scored"]]
        wins = res.meta.get("windows", [[7, 30], [30, 90]])
        fig, axs = plt.subplots(1, len(CONFIG_SCENARIOS), figsize=(11, 3.8), sharey=True)
        meths = [m for m in METHOD_LABEL if m in set(X["method"])]
        w = 0.8 / max(len(meths), 1)
        for ax, scn in zip(np.atleast_1d(axs), CONFIG_SCENARIOS):
            for i, m in enumerate(meths):
                Y = X[(X["method"] == m) & (X["scenario"] == scn)]
                v = [np.nanmean(_per_event(Y, f"share_{a}_{b}")) if len(Y) else np.nan for a, b in wins]
                v0 = [np.nanmean(_per_event(Y, "detected")) if len(Y) else np.nan]
                ax.bar(np.arange(len(wins) + 1) + i * w, v0 + v, w, color=COLORS[m], label=METHOD_LABEL[m])
            ax.set_xticks(np.arange(len(wins) + 1) + w * (len(meths) - 1) / 2)
            ax.set_xticklabels(["Recall, onset"] + [f"Alarm share {a}-{b} d" for a, b in wins], fontsize=8)
            ax.set_title(CONFIG_SCENARIOS[scn]["title"], fontsize=9); ax.grid(alpha=0.3, axis="y")
        np.atleast_1d(axs)[0].legend(fontsize=8); np.atleast_1d(axs)[0].set_ylabel("Share")
        fig.tight_layout(); figs["persistence"] = fig
    B1 = res.b1
    if len(B1):
        Z = B1[~B1["change_quarter"]]
        fig, axs = plt.subplots(1, len(CONFIG_SCENARIOS), figsize=(11, 3.8), sharey=False)
        for ax, scn in zip(np.atleast_1d(axs), CONFIG_SCENARIOS):
            Y = Z[Z["scenario"] == scn]
            for name, mk in (("sps", "s"), ("kalman", "o")):
                Q = Y[Y["estimator"] == name]
                if len(Q):
                    ax.scatter(Q["expected_b1_pct"], Q["estimated_b1_pct"], marker=mk, alpha=0.7,
                               label=EST_LABEL[name])
            lim = np.nanmax(np.abs(np.r_[Y["expected_b1_pct"], Y["estimated_b1_pct"], 1.0])) * 1.1
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel("Expected Δb1, %"); ax.set_title(CONFIG_SCENARIOS[scn]["title"], fontsize=9)
            ax.grid(alpha=0.3)
        np.atleast_1d(axs)[0].set_ylabel("Estimated Δb1, % (full quarters after the change)")
        np.atleast_1d(axs)[0].legend(fontsize=8)
        fig.tight_layout(); figs["b1"] = fig
    if res.examples:
        fig, axs = plt.subplots(1, len(res.examples), figsize=(11, 3.6))
        for ax, (scn, ex) in zip(np.atleast_1d(axs), res.examples.items()):
            ax.plot(ex["ratio_clean"].index, ex["ratio_clean"].values, color="grey", lw=1, label="clean")
            ax.plot(ex["ratio_mod"].index, ex["ratio_mod"].values, color="C3", lw=1, label="modified")
            ax.axvline(ex["t0"], color="k", ls="--", lw=0.8)
            ax.set_title(f"{scn}: plant {ex['plant']}, f = {ex['strength']:.0%}", fontsize=9)
            ax.set_ylabel("Daily P / P_exp"); ax.grid(alpha=0.3); ax.tick_params(axis="x", labelsize=7, rotation=30)
        np.atleast_1d(axs)[0].legend(fontsize=8)
        fig.tight_layout(); figs["examples"] = fig
    if fig_dir:
        for k, f in figs.items():
            f.savefig(os.path.join(fig_dir, f"fig_config_{k}.png"), dpi=150, bbox_inches="tight")
    return figs


def export_latex(res: ConfigResult, out_dir: str = "results/config") -> Dict[str, str]:
    import pv_semisynth_eval as sse
    os.makedirs(out_dir, exist_ok=True); out = {}
    for k, t in TITLES.items():
        X = res.tables.get(k)
        if X is None or not len(X):
            continue
        X = X.drop(columns=[c for c in ("method",) if c in X])
        name = f"config_{k}"
        X.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
        p = os.path.join(out_dir, name + ".tex")
        with open(p, "w") as f:
            f.write(sse.df_to_latex(X, t, f"tab:{name}", digits=3))
        out[k] = p
    return out


def zip_outputs(out_dir: str = "results/config", fig_dir: str = "results/figures",
                name: str = "config_results.zip") -> str:
    import zipfile
    z = os.path.join(out_dir, name)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(out_dir)):
            if fn.endswith((".tex", ".csv")):
                zf.write(os.path.join(out_dir, fn), fn)
        if fig_dir and os.path.isdir(fig_dir):
            for fn in sorted(os.listdir(fig_dir)):
                if fn.startswith("fig_config_"):
                    zf.write(os.path.join(fig_dir, fn), os.path.join("figures", fn))
    return z
