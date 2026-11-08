"""
Section 12: fixes of the detector, before / after comparison and simple baselines
(semi-synthetic protocol of sections 10-11, frozen STGNN weights).

Causes found in sections 10-11 (HKUST run) and the fixes evaluated here
------------------------------------------------------------------------
1. eta_hat absorbs the anomaly (the efficiency estimate of the current window follows the drop,
   so the physics residual vanishes)   ->  eta_hat from history: rolling 7-day median with 24 h
   lag ("med7d"; alternatives "lag24", "lag168", "train" - see section 11).
2. Threshold too high / min_len too long  ->  min_len = 2 and the quantile q chosen on the
   VALIDATION part of the training period (no test, no semi-synthetic labels) so that the alarm
   rate equals the budget of the original configuration.
3. Full outages are partly invisible   ->  rule "P ~ 0 at GHI > 200 W/m2" fused (OR) with the
   STGNN detections (requires NaN-aware aggregation, pv_preprocess, so that zero-filled gaps are
   not mistaken for outages: unobserved hours are never flagged).
4. Fuzzy rules for clipping / shading / partial loss  ->  calibrated rule set
   (pv_fuzzy_calibration; parameters fitted on disjoint calibration placements).

Comparison
----------
* steps S0 -> S4 (cumulative): S0 = section 10 configuration, S4 = all fixes;
* simple baselines on the same placements: PR threshold rule, PR+ (hour x month baseline, cluster
  peers, causal 30-day rolling median; pv_prplus, same definition as sections 15, 16, 18),
  rule "P ~ 0 at GHI > 200", Isolation Forest;
* every method at (a) its operating point chosen on validation at the same false-alarm budget
  and (b) recall interpolated along its threshold curve at exactly the false-alarm budget of S0
  on the clean test data (control plants).

The module only reads frozen checkpoints (fingerprints are checked); it does not modify
pv_semisynth_eval / pv_semisynth_sensitivity results.  All tables and figures are in English.
"""

from __future__ import annotations

import contextlib
import io
import os
import pickle
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import pv_semisynth_gen as ssg
import pv_semisynth_eval as sse
import pv_semisynth_sensitivity as sss

IRR = ssg.IRR_COL
SCN_ORDER = sss.SCN_ORDER
SCN_TITLE = {k: s.title for k, s in ssg.SCENARIOS.items()}


def _log(*a):
    print(time.strftime("%H:%M:%S"), "[fixes]", *a, flush=True)


# ============================================================================ configuration
@dataclass
class DetectorConfig:
    key: str
    label: str
    eta_mode: str = "current"
    q: Optional[float] = None          # None -> checkpoint q; "val" selection via select_q=True
    select_q: bool = False             # choose q on validation at the false-alarm budget
    min_len: int = 3
    zero_rule: bool = False
    fuzzy: str = "full"                # "full" (original rules) | "calibrated"


def default_steps(eta_mode: str = "med7d", min_len: int = 2) -> List[DetectorConfig]:
    return [
        DetectorConfig("S0", "S0 before (section 10)"),
        DetectorConfig("S1", f"S1 + eta_hat {eta_mode}", eta_mode=eta_mode),
        DetectorConfig("S2", f"S2 + min_len {min_len}, q on validation", eta_mode=eta_mode, select_q=True,
                       min_len=min_len),
        DetectorConfig("S3", "S3 + rule P~0 at GHI>200", eta_mode=eta_mode, select_q=True, min_len=min_len,
                       zero_rule=True),
        DetectorConfig("S4", "S4 after: + calibrated fuzzy rules", eta_mode=eta_mode, select_q=True,
                       min_len=min_len, zero_rule=True, fuzzy="calibrated"),
    ]


@dataclass
class FixesConfig:
    eta_mode: str = "med7d"
    min_len_after: int = 2
    qs: Sequence[float] = (0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995)
    q_select_grid: int = 41            # fine grid (logit spaced, 0.9 .. 0.9999) for the validation choice
    zero_eps: float = 0.02             # P / P_rated below this = "zero"
    zero_ghi: float = 200.0            # W/m2
    zero_min_pexp: float = 0.05        # P_exp / P_rated above this (sun high enough)
    zero_min_h: int = 2                # consecutive hours
    zero_min_h_grid: Sequence[int] = (1, 2, 3, 4, 6)
    min_val_budget_events: float = 1.0  # budget floor: this many alarms over all validation hours
    day_frac: float = 0.05             # daylight for PR / IF: P_exp > day_frac * P_rated and GHI > day_ghi
    day_ghi: float = 50.0
    if_n_estimators: int = 200
    if_max_rows: int = 60000
    seed: int = 0
    baselines: Sequence[str] = ("pr_rule", "prp_rule", "zero_rule", "iforest")
    batch_size: int = 64


BASELINE_LABEL = {"pr_rule": "Baseline: PR threshold rule", "prp_rule": "Baseline: PR+ (hour x month, peers, rolling)",
                  "zero_rule": "Baseline: rule P~0 at GHI>200",
                  "iforest": "Baseline: Isolation Forest"}


# ============================================================================ raw collection
def collect(models: Dict[int, "sse.FrozenSTGNN"], D: dict, reals: Dict[int, "ssg.Realisation"],
            cache_dir: Optional[str] = "results/fixes", sens_cache_dir: Optional[str] = "results/sensitivity",
            batch_size: int = 64, use_cache: bool = True) -> Dict[int, "sss.ModelRaw"]:
    """Forward passes of the frozen models (calibration, validation, clean test, placements).
    Reuses the raw outputs of section 11 when available (same fingerprints / placements)."""
    out = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    for ms, mdl in models.items():
        fp = mdl.fingerprint()
        f = os.path.join(cache_dir, f"raw_fixes_{ms}_{fp}.pkl") if cache_dir else None
        if f and use_cache and os.path.exists(f):
            with open(f, "rb") as fh:
                R = pickle.load(fh)
            if sorted(R.inj) == sorted(reals) and getattr(R, "val", None) is not None:
                out[ms] = R; _log(f"model {ms}: raw outputs from cache"); continue
        R = None
        g = os.path.join(sens_cache_dir, f"raw_{ms}_{fp}.pkl") if sens_cache_dir else None
        if g and use_cache and os.path.exists(g):
            with open(g, "rb") as fh:
                R0 = pickle.load(fh)
            if sorted(R0.inj) == sorted(reals):
                R = R0; _log(f"model {ms}: raw outputs of section 11 reused")
        if R is None:
            _log(f"model {ms}: forward passes (calibration, clean test, {len(reals)} placements)")
            R = sss.collect_model(mdl, D, reals, batch_size)
        pipe = mdl.build(D["df"], D["phys"])
        vi = np.asarray(pipe._val_idx)
        if not len(vi):                                  # no validation split: last 10 % of calibration
            vi = np.asarray(pipe._fit_idx)[-max(len(pipe._fit_idx) // 10, 1):]
        R.val = sss._collect(pipe, pipe.ds_tr, vi, batch_size)
        R.val_not_ready = sss._not_ready(pipe, R.val["pos"])
        del pipe
        assert mdl.fingerprint() == fp, "weights changed"
        if f:
            with open(f, "wb") as fh:
                pickle.dump(R, fh)
        out[ms] = R
    return out


def _ghi_at(df: pd.DataFrame, ts: pd.DatetimeIndex) -> np.ndarray:
    return df[IRR].reindex(ts).to_numpy(float)


# ============================================================================ helpers
def _bad(raw: dict, n: int, nr: Optional[np.ndarray]) -> np.ndarray:
    bad = ~raw["active"][:, :n]
    return bad | nr if nr is not None else bad


def zero_rule_mask(raw: dict, ghi: np.ndarray, n: int, nr: Optional[np.ndarray], cfg: FixesConfig,
                   min_h: Optional[int] = None) -> np.ndarray:
    """(S, n) True inside runs (>= min_h hours) with P/P_rated < eps while GHI > zero_ghi and
    P_exp/P_rated > zero_min_pexp, on observed & active & ready hours only."""
    y = raw["y"].mean(-1)[:, :n]; a = raw["a"].mean(-1)[:, :n]
    M = (y < cfg.zero_eps) & (np.nan_to_num(ghi)[:, None] > cfg.zero_ghi) & (a > cfg.zero_min_pexp)
    M &= ~_bad(raw, n, nr)
    return sse._run_mask(M, int(min_h or cfg.zero_min_h))


def _events_from_mask(em: np.ndarray, comb: np.ndarray, zp: np.ndarray, eta: np.ndarray,
                      ts: pd.DatetimeIndex, pv: List[str]) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(pv):
        m = em[:, j]
        if not m.any():
            continue
        d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            c = comb[a:b, j]
            with np.errstate(all="ignore"):
                rows.append(dict(plant=ssg._short(name), start=ts[a], end=ts[b - 1], n_windows=int(b - a),
                                 peak_score=float(np.nanmax(c)) if np.isfinite(c).any() else np.nan,
                                 mean_physics_z=float(np.nanmean(zp[a:b, j])) if np.isfinite(zp[a:b, j]).any() else np.nan,
                                 mean_eta=float(np.nanmean(eta[a:b, j])) if np.isfinite(eta[a:b, j]).any() else np.nan))
    out = pd.DataFrame(rows)
    return out.sort_values("peak_score", ascending=False, na_position="last").reset_index(drop=True) if len(out) else out


def make_bundle(comb: np.ndarray, thr: np.ndarray, zp: np.ndarray, eta_v: np.ndarray, raw: dict, pv: List[str],
                min_len: int, zero: Optional[np.ndarray] = None) -> "sse.ScoreBundle":
    """STGNN-type bundle (threshold + min_len), optionally OR-fused with a rule mask."""
    with np.errstate(invalid="ignore"):
        exc = np.nan_to_num(comb, nan=-np.inf) > thr
    eta2 = eta_v.mean(-1).astype(np.float64).copy() if eta_v.ndim == 3 else eta_v.astype(np.float64).copy()
    eta2[~np.isfinite(comb)] = np.nan
    em = sse._run_mask(exc, min_len)
    if zero is not None:
        exc = exc | zero; em = em | zero
    ev = _events_from_mask(em, comb, zp, eta2, raw["ts"], pv)
    return sse.ScoreBundle(raw["ts"], comb, thr, exc, em, zp, eta2, ev, pv)


def _rate(b: "sse.ScoreBundle") -> float:
    return sss._bg_rate(b)


def _rate_fast(comb: np.ndarray, thr: np.ndarray, min_len: int, zero: Optional[np.ndarray] = None) -> float:
    """Events per 1000 scored hours without building the event table (validation selection)."""
    with np.errstate(invalid="ignore"):
        exc = np.nan_to_num(comb, nan=-np.inf) > thr
    em = sse._run_mask(exc, min_len)
    if zero is not None:
        em = em | zero
    d = np.diff(np.concatenate([np.zeros((1, em.shape[1]), np.int8), em.astype(np.int8),
                                np.zeros((1, em.shape[1]), np.int8)]), axis=0)
    h = int(np.isfinite(comb).sum())
    return 1000.0 * int((d == 1).sum()) / h if h else np.nan


def _q_grid(cfg: FixesConfig) -> np.ndarray:
    lo, hi = np.log(0.9 / 0.1), np.log(0.9999 / 0.0001)
    z = np.linspace(lo, hi, cfg.q_select_grid)
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================================ score families
@dataclass
class Family:
    """Scores of one method on all sets + how to build thresholds for a quantile q."""
    name: str
    sets: Dict[object, dict]                 # key -> dict(raw, comb, zp, eta, nr)
    thr_fn: object                           # q -> {key: thr matrix}
    model_seed: object = "-"
    q_default: Optional[float] = None


def stgnn_family(R: "sss.ModelRaw", eta_mode: str, scfg: Optional["sss.SensConfig"] = None) -> Family:
    scfg = scfg or sss.SensConfig()
    calib, pv = R.calib, R.pv_cols
    n = calib["n_pv"]
    tmed = sss._train_eta(R.cal, calib, pv, scfg.day_frac)
    ecal = sss.eta_variant(eta_mode, R.cal, tmed, scfg)
    phy_cal = sss._phy(R.cal, ecal)
    if eta_mode == "current":
        mu, sd = calib["stats"]["phy_mu"], calib["stats"]["phy_sd"]
    else:
        mu, sd = sss._phy_stats(phy_cal, R.cal["active"][:, :n], calib, pv)
    cal_comb, _ = sss._combined(R.cal, phy_cal, calib, mu, sd)
    sets = {}
    src = [("val", R.val, R.val_not_ready), ("clean", R.clean, R.not_ready)] + \
          [(ps, r, R.not_ready) for ps, r in R.inj.items()]
    for key, raw, nr in src:
        ev = sss.eta_variant(eta_mode, raw, tmed, scfg)
        comb, zp = sss._combined(raw, sss._phy(raw, ev), calib, mu, sd, nr)
        sets[key] = dict(raw=raw, comb=comb, zp=zp, eta=ev, nr=nr)
    cache = {}

    def thr_fn(q):
        q = float(q)
        if q not in cache:
            if eta_mode == "current" and abs(q - calib["q"]) < 1e-12:
                th = {k: calib[k] for k in ("node_thresholds", "regime_thresholds", "cluster_threshold_prior",
                                            "threshold_pot")}
            else:
                th = sss.thresholds_from_cal(cal_comb, R.cal["pos"], q, calib, pv)
            cache[q] = {k: sss.threshold_matrix(th, s["raw"]["pos"], calib) for k, s in sets.items()}
        return cache[q]
    return Family(f"stgnn_{eta_mode}", sets, thr_fn, R.seed, calib["q"])


def _daylight(raw: dict, ghi: np.ndarray, n: int, cfg: FixesConfig) -> np.ndarray:
    a = raw["a"].mean(-1)[:, :n]
    return (a > cfg.day_frac) & (np.nan_to_num(ghi)[:, None] > cfg.day_ghi) & raw["active"][:, :n]


def _plant_level(R: "sss.ModelRaw", ghi_cal: np.ndarray, cfg: FixesConfig) -> np.ndarray:
    """Per-plant median P/P_exp in the calibration (training) period; cold-start plants borrow
    the median of their technical cluster (fleet median as last resort)."""
    n = R.calib["n_pv"]
    y = R.cal["y"].mean(-1)[:, :n]; a = R.cal["a"].mean(-1)[:, :n]
    day = _daylight(R.cal, ghi_cal, n, cfg) & (y > 0)
    r0 = np.array([np.median(y[day[:, j], j] / a[day[:, j], j]) if day[:, j].sum() >= 20 else np.nan
                   for j in range(n)])
    lab = R.calib["cluster_labels"]
    lab = lab.reindex(R.pv_cols).to_numpy() if lab is not None else np.full(n, np.nan)
    have = np.isfinite(r0)
    fleet = np.nanmedian(r0) if have.any() else 1.0
    for j in np.flatnonzero(~have):
        same = have & (lab == lab[j]) if pd.notna(lab[j]) else np.zeros(n, bool)
        r0[j] = np.nanmedian(r0[same]) if same.any() else fleet
    return r0


def _node_quantile_thr(cal_s: np.ndarray, cal_day: np.ndarray, q: float, min_n: int = 20) -> np.ndarray:
    n = cal_s.shape[1]
    allv = cal_s[cal_day & np.isfinite(cal_s)]
    fleet = float(np.quantile(allv, q)) if allv.size else np.inf
    return np.array([np.quantile(cal_s[cal_day[:, j], j], q) if cal_day[:, j].sum() >= min_n else fleet
                     for j in range(n)])


def _raw_sets(R: "sss.ModelRaw", ghis: Dict[object, np.ndarray]):
    return [("val", R.val, R.val_not_ready), ("clean", R.clean, R.not_ready)] + \
           [(ps, r, R.not_ready) for ps, r in R.inj.items()]


def pr_family(R: "sss.ModelRaw", ghis: Dict[object, np.ndarray], cfg: FixesConfig) -> Family:
    """PR threshold rule: s = |(P/P_exp) / r0 - 1| on daylight hours (two-sided, so both losses
    and over-generation are flagged); per-plant threshold = quantile q of the training scores."""
    n = R.calib["n_pv"]
    r0 = _plant_level(R, ghis["cal"], cfg)

    def score(raw, ghi, nr):
        y = raw["y"].mean(-1)[:, :n]; a = raw["a"].mean(-1)[:, :n]
        day = _daylight(raw, ghi, n, cfg)
        with np.errstate(all="ignore"):
            s = np.where(day, np.abs(y / np.maximum(a, 1e-6) / r0[None, :] - 1.0), 0.0)
        s = s.astype(np.float64); s[_bad(raw, n, nr)] = np.nan
        return s, day

    cs, cday = score(R.cal, ghis["cal"], None)
    sets = {}
    for key, raw, nr in _raw_sets(R, ghis):
        s, _ = score(raw, ghis[key], nr)
        sets[key] = dict(raw=raw, comb=s, zp=np.zeros_like(s), eta=raw["eta"], nr=nr)

    def thr_fn(q):
        t = _node_quantile_thr(cs, cday, q)
        return {k: np.broadcast_to(t, v["comb"].shape) for k, v in sets.items()}
    return Family("pr_rule", sets, thr_fn)


def _if_features(raw: dict, ghi: np.ndarray, r0: np.ndarray, n: int, cfg: FixesConfig):
    y = raw["y"].mean(-1)[:, :n]; a = raw["a"].mean(-1)[:, :n]
    day = _daylight(raw, ghi, n, cfg)
    with np.errstate(all="ignore"):
        rn = np.where(day, y / np.maximum(a, 1e-6) / r0[None, :], np.nan)
    ts = raw["ts"]
    roll = pd.DataFrame(rn, index=ts).rolling("3h", min_periods=1).mean().to_numpy()
    with np.errstate(all="ignore"):
        peer = rn - np.nanmedian(np.where(np.isfinite(rn), rn, np.nan), axis=1, keepdims=True)
    h = ts.hour.to_numpy()
    F = np.stack([np.nan_to_num(rn, nan=1.0), y, a, np.repeat((np.nan_to_num(ghi) / 1000.0)[:, None], n, 1),
                  np.repeat(np.sin(2 * np.pi * h / 24)[:, None], n, 1), np.repeat(np.cos(2 * np.pi * h / 24)[:, None], n, 1),
                  np.nan_to_num(roll, nan=1.0), np.nan_to_num(peer, nan=0.0)], axis=-1)
    return F, day


def iforest_family(R: "sss.ModelRaw", ghis: Dict[object, np.ndarray], cfg: FixesConfig) -> Family:
    """Isolation Forest on hourly per-plant features (normalised ratio, P, P_exp, GHI, hour,
    3-h causal mean of the ratio, deviation from the fleet median) trained on the training period."""
    from sklearn.ensemble import IsolationForest
    n = R.calib["n_pv"]
    r0 = _plant_level(R, ghis["cal"], cfg)
    Fc, dc = _if_features(R.cal, ghis["cal"], r0, n, cfg)
    X = Fc[dc]
    rng = np.random.default_rng(cfg.seed)
    if len(X) > cfg.if_max_rows:
        X = X[rng.choice(len(X), cfg.if_max_rows, replace=False)]
    model = IsolationForest(n_estimators=cfg.if_n_estimators, random_state=cfg.seed).fit(X)

    def score(F, day, raw, nr):
        s = np.zeros(day.shape, np.float64)
        if day.any():
            s[day] = -model.score_samples(F[day])
        return s, day

    cs, _ = score(Fc, dc, R.cal, None)
    floor = float(np.min(cs[dc])) if dc.any() else 0.0
    cs = np.where(dc, cs, floor)
    sets = {}
    for key, raw, nr in _raw_sets(R, ghis):
        F, d = _if_features(raw, ghis[key], r0, n, cfg)
        s, _ = score(F, d, raw, nr)
        s = np.where(d, s, floor); s[_bad(raw, n, nr)] = np.nan
        sets[key] = dict(raw=raw, comb=s, zp=np.zeros_like(s), eta=raw["eta"], nr=nr)

    def thr_fn(q):
        t = _node_quantile_thr(cs, dc, q)
        return {k: np.broadcast_to(t, v["comb"].shape) for k, v in sets.items()}
    return Family("iforest", sets, thr_fn)


# ============================================================================ evaluation
def _evaluate(D, C, reals, clean_b, inj_b: Dict[int, "sse.ScoreBundle"], min_len: int, tag: dict,
              fz_cfg=None, classify: bool = False):
    E, FA, CL = [], [], []
    ecfg = sse.EvalConfig(min_len=min_len, fuzzy_config=fz_cfg)
    for ps, real in reals.items():
        with contextlib.redirect_stdout(io.StringIO()):
            if classify:
                out = sse.evaluate_detection(D, C, real, clean_b, inj_b[ps], ecfg, None)
            else:
                with sss._no_classification():
                    out = sse.evaluate_detection(D, C, real, clean_b, inj_b[ps], ecfg, None)
        t = dict(tag, placement=ps)
        E.append(out["events"].assign(**t)); FA.append(out["false_alarms"].assign(**t))
        if classify and len(out["classes"]):
            CL.append(out["classes"].assign(**t))
    E = pd.concat(E, ignore_index=True); FA = pd.concat(FA, ignore_index=True)
    CL = pd.concat(CL, ignore_index=True) if CL else pd.DataFrame()
    return sss.summarise(E, FA, [_rate(clean_b)]), E, FA, CL


def _select_q(fam: Family, min_len: int, budget: float, cfg: FixesConfig,
              zeros: Optional[Dict[object, np.ndarray]] = None) -> Tuple[float, float]:
    """Smallest q of the fine grid whose VALIDATION alarm rate does not exceed the budget."""
    best = None
    v = fam.sets["val"]
    for q in _q_grid(cfg):
        r = _rate_fast(v["comb"], fam.thr_fn(q)["val"], min_len, zeros.get("val") if zeros else None)
        if np.isfinite(r) and r <= budget:
            best = (float(q), float(r)); break
    if best is None:
        q = float(_q_grid(cfg)[-1])
        best = (q, _rate_fast(v["comb"], fam.thr_fn(q)["val"], min_len, zeros.get("val") if zeros else None))
    return best


@dataclass
class FixesResult:
    points: pd.DataFrame                      # operating points (per model seed)
    curves: pd.DataFrame                      # threshold curves (per model seed)
    events: pd.DataFrame
    classes: pd.DataFrame
    oracle: pd.DataFrame
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def run_fixes(D: dict, C: dict, models: Dict[int, "sse.FrozenSTGNN"], ss_res=None,
              placement_seeds: Sequence[int] = (0, 1, 2), cfg: Optional[FixesConfig] = None,
              steps: Optional[List[DetectorConfig]] = None, fuzzy_calibrated=None, gen_cfg=None,
              cache_dir: Optional[str] = "results/fixes", sens_cache_dir: Optional[str] = "results/sensitivity",
              use_cache: bool = True, aggregation_table: Optional[pd.DataFrame] = None) -> FixesResult:
    """Before / after + baselines on the SAME placements as section 10 (``ss_res.plans``)."""
    import pv_fuzzy as fz
    import pv_fuzzy_calibration as fzc
    cfg = cfg or FixesConfig()
    steps = steps or default_steps(cfg.eta_mode, cfg.min_len_after)
    t0 = time.time()
    if ss_res is not None and getattr(ss_res, "plans", None):
        plans = dict(ss_res.plans)
    else:
        gen_cfg = gen_cfg or ssg.GenConfig(boundary=sse.BOUNDARY)
        plans = {ps: ssg.plan_semisynthetic(D, C, seed=ps, cfg=gen_cfg) for ps in placement_seeds}
    reals = {ps: ssg.apply_plan(D, C, p) for ps, p in plans.items()}
    _log(f"placements: {sorted(plans)}")

    # fuzzy calibration on disjoint placements
    if fuzzy_calibrated is None:
        fuzzy_calibrated, _ = fzc.fit(D, C, fzc.CAL_PLACEMENTS, gen_cfg or ssg.GenConfig(boundary=sse.BOUNDARY),
                                      eval_placements=sorted(plans))
    fz_cfg = {"full": fz.FuzzyConfig(), "calibrated": fuzzy_calibrated}

    raws = collect(models, D, reals, cache_dir, sens_cache_dir, cfg.batch_size, use_cache)
    any_R = next(iter(raws.values()))
    ghis = {"cal": _ghi_at(D["df"], any_R.cal["ts"]), "val": _ghi_at(D["df"], any_R.val["ts"]),
            "clean": _ghi_at(D["df"], any_R.clean["ts"])}
    for ps, real in reals.items():
        ghis[ps] = _ghi_at(real.df, any_R.inj[ps]["ts"])

    P, CV, EV, CL = [], [], [], []

    def _zeros(fam, R, min_h=None):
        return {k: zero_rule_mask(s["raw"], ghis[k], R.calib["n_pv"], s["nr"], cfg, min_h) for k, s in fam.sets.items()}

    def _point(fam, R, q, min_len, zeros, tag, fzk, classify, val_rate=None):
        thr = fam.thr_fn(q)
        B = {k: make_bundle(s["comb"], thr[k], s["zp"], s["eta"], s["raw"], R.pv_cols, min_len,
                            zeros.get(k) if zeros else None) for k, s in fam.sets.items()}
        summ, E, FA, cl = _evaluate(D, C, reals, B["clean"], {ps: B[ps] for ps in reals}, min_len, tag,
                                    fz_cfg[fzk], classify)
        vr = val_rate if val_rate is not None else _rate(B["val"])
        return dict(tag, q=q, min_len=min_len, val_fa=vr, **summ), E, cl

    budget_val, budgets = None, []
    for ms, R in raws.items():
        fams = {}
        for st in steps:
            if st.eta_mode not in fams:
                fams[st.eta_mode] = stgnn_family(R, st.eta_mode)
        for st in steps:
            fam = fams[st.eta_mode]
            zeros = _zeros(fam, R) if st.zero_rule else None
            if st.select_q:
                q, vr = _select_q(fam, st.min_len, budget_val, cfg, zeros)
            else:
                q, vr = float(st.q or R.calib["q"]), None
            tag = dict(method=st.key, label=st.label, model_seed=ms, eta_mode=st.eta_mode, zero_rule=st.zero_rule,
                       fuzzy_rules=st.fuzzy)
            row, E, cl = _point(fam, R, q, st.min_len, zeros, tag, st.fuzzy, classify=True, val_rate=vr)
            if st.key == "S0":
                h_val = int(np.isfinite(fam.sets["val"]["comb"]).sum())
                floor = 1000.0 * cfg.min_val_budget_events / h_val if h_val else 0.0
                budget_val = max(float(row["val_fa"]), floor)
                row["val_budget"] = budget_val
                budgets.append(budget_val)
            P.append(row); EV.append(E); CL.append(cl)
            _log(f"model {ms} {st.key}: q={q:.4f} val FA={row['val_fa']:.2f} test FA={row['clean_fa']:.2f} "
                 f"recall={row['recall']:.3f} ({time.time() - t0:.0f}s)")
        # threshold curves for the equal-budget comparison: S0 and S3/S4 detection
        for key, st in (("S0", steps[0]), ("S4", steps[-1])):
            fam = fams[st.eta_mode]
            zeros = _zeros(fam, R) if st.zero_rule else None
            for q in sorted(set(map(float, cfg.qs)) | {float(R.calib["q"])}):
                tag = dict(method=key, label=st.label, model_seed=ms)
                row, _, _ = _point(fam, R, q, st.min_len, zeros, tag, "full", classify=False)
                CV.append(row)

    # baselines (independent of the STGNN seed: computed once on the raw data, budget = mean S0 budget)
    R = any_R
    budget_val = float(np.mean(budgets)) if budgets else budget_val
    for b in cfg.baselines:
        lab = BASELINE_LABEL[b]
        if b == "zero_rule":
            fam = pr_family(R, ghis, cfg)            # only for the set structure / masks
            for mh in cfg.zero_min_h_grid:
                zeros = _zeros(fam, R, mh)
                tag = dict(method=b, label=lab, model_seed="-")
                sets0 = {k: dict(s, comb=np.where(np.isfinite(s["comb"]), 0.0, np.nan)) for k, s in fam.sets.items()}
                famz = Family(b, sets0, lambda q: {k: np.full(v["comb"].shape, 0.5) for k, v in sets0.items()})
                row, E, cl = _point(famz, R, 0.5, 1, zeros, dict(tag, param=f"min_h={mh}"), "calibrated",
                                    classify=(mh == cfg.zero_min_h))
                row["q"] = np.nan; row["min_len"] = mh
                CV.append(row)
                if mh == cfg.zero_min_h:
                    P.append(row); EV.append(E); CL.append(cl)
            continue
        if b == "pr_rule":
            fam = pr_family(R, ghis, cfg)
        elif b == "prp_rule":
            import pv_prplus as prp
            fam = prp.prp_family(R, ghis, cfg)
        else:
            fam = iforest_family(R, ghis, cfg)
        q, vr = _select_q(fam, cfg.min_len_after, budget_val, cfg)
        tag = dict(method=b, label=lab, model_seed="-")
        row, E, cl = _point(fam, R, q, cfg.min_len_after, None, tag, "calibrated", classify=True, val_rate=vr)
        P.append(row); EV.append(E); CL.append(cl)
        for qq in cfg.qs:
            r2, _, _ = _point(fam, R, float(qq), cfg.min_len_after, None, tag, "full", classify=False)
            CV.append(r2)
        _log(f"{b}: q={q:.4f} val FA={row['val_fa']:.2f} test FA={row['clean_fa']:.2f} recall={row['recall']:.3f}")

    # classification of the TRUE intervals (detector-independent): original vs calibrated rules
    oracle = fzc.oracle_features(D, C, plans, reals)
    res = FixesResult(points=pd.DataFrame(P), curves=pd.DataFrame(CV),
                      events=pd.concat(EV, ignore_index=True), classes=pd.concat([c for c in CL if len(c)], ignore_index=True)
                      if any(len(c) for c in CL) else pd.DataFrame(), oracle=oracle,
                      meta=dict(cfg=cfg.__dict__, steps=[s.__dict__ for s in steps], placements=sorted(plans),
                                model_seeds=list(raws), fingerprints={ms: r.fingerprint for ms, r in raws.items()},
                                weights_unchanged=all(m.unchanged() for m in models.values()),
                                fuzzy_calibration=fuzzy_calibrated.calibration,
                                calibration_placements=list(fzc.CAL_PLACEMENTS)))
    res.tables = build_tables(res, fz_cfg)
    if aggregation_table is not None:
        res.tables["aggregation"] = aggregation_table
    if cache_dir:
        with open(os.path.join(cache_dir, "fixes_result.pkl"), "wb") as f:
            pickle.dump(res, f)
    _log(f"done in {time.time() - t0:.0f}s; weights unchanged: {res.meta['weights_unchanged']}")
    return res


# ============================================================================ tables
_COLS = {"q": "q", "min_len": "min_len", "val_fa": "Val. FA /1000 h", "clean_fa": "Test FA (controls) /1000 h",
         "bg_all": "Test FA (all plants) /1000 h", "recall": "Recall", "precision": "Precision", "F1": "F1",
         "delay_h": "Delay, h (median)", "raw_hit": "Raw hit", "cut_short": "Short events cut",
         "outage_d_eta": "Δη̂ outage", "outage_d_phy": "Δz_phys outage"}
ORDER = ["S0", "S1", "S2", "S3", "S4", "pr_rule", "prp_rule", "zero_rule", "iforest"]


def _mean_by_method(Pt: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in Pt.columns if c not in ("method", "label", "model_seed", "eta_mode", "zero_rule", "fuzzy_rules",
                                               "param", "placement") and pd.api.types.is_numeric_dtype(Pt[c])]
    A = Pt.groupby(["method", "label"], sort=False)[num].mean().reset_index()
    A["recall_sd"] = Pt.groupby(["method", "label"], sort=False)["recall"].std().values
    A["_o"] = A["method"].map({m: i for i, m in enumerate(ORDER)})
    return A.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def _class_share(cls: pd.DataFrame, E: pd.DataFrame, col: str = "fuzzy") -> pd.Series:
    """Per scenario: share of detected events whose class (majority over detections) is right."""
    if not len(cls) or col not in cls:
        return pd.Series(dtype=float)
    key = ["placement", "model_seed", "event_id", "method"]
    d = cls[cls["kind"] == "detected"].merge(E[key + ["scenario"]].drop_duplicates(), on=key)
    if not len(d):
        return pd.Series(dtype=float)
    d = d.assign(ok=[r == ssg.SCENARIOS[s].expected_name for r, s in zip(d[col], d["scenario"])])
    v = d.groupby(key + ["scenario"])["ok"].mean() >= 0.5
    out = v.groupby(level="scenario").mean()
    out["all"] = v.mean()
    return out


def build_tables(res: FixesResult, fz_cfg: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    import pv_fuzzy as fz
    import pv_fuzzy_calibration as fzc
    T = {}
    A = _mean_by_method(res.points)
    scn = [f"R_{s}" for s in SCN_ORDER if f"R_{s}" in A]
    main = ["q", "min_len", "val_fa", "clean_fa", "bg_all", "recall", "precision", "F1", "delay_h"]
    t = A[["label"] + [c for c in main if c in A]].rename(columns={"label": "Method", **_COLS})
    T["operating_points"] = t
    s = A[["label", "recall"] + scn + [c for c in ("outage_d_eta", "outage_d_phy") if c in A]].rename(
        columns={"label": "Method", "recall": "Recall (all)", **{f"R_{k}": SCN_TITLE[k] for k in SCN_ORDER}, **_COLS})
    T["scenario_recall"] = s
    # equal false-alarm budget on the clean test data (budget = S0 at its operating point)
    S0 = A[A["method"] == "S0"]
    if len(S0) and len(res.curves):
        budget = float(S0["clean_fa"].iloc[0])
        Cv = res.curves.copy()
        Cv["model_seed"] = Cv["model_seed"].astype(str)
        rows = []
        for (m, ms), g in Cv.groupby(["method", "model_seed"], sort=False):
            g = g[np.isfinite(g["clean_fa"])].sort_values(["clean_fa", "recall"])
            fa = g["clean_fa"].to_numpy(float)
            if not len(fa):
                continue
            fa_u, iu = np.unique(fa, return_index=True)
            inside = len(fa_u) >= 2 and fa_u[0] <= budget <= fa_u[-1]
            r = dict(method=m, label=g["label"].iloc[0], model_seed=ms)
            for k in ["recall", "precision", "F1", "bg_all"] + [c for c in scn if c in g]:
                v = g[k].to_numpy(float)[iu]; ok = np.isfinite(v)
                r[k] = (float(np.interp(budget, fa_u[ok], v[ok])) if inside and ok.sum() >= 2
                        else float(g.iloc[int(np.argmin(np.abs(fa - budget)))][k]))
            r["interpolated"] = bool(inside)
            r["fa_range"] = f"{fa_u[0]:.2f}-{fa_u[-1]:.2f}"
            rows.append(r)
        F = pd.DataFrame(rows)
        num = [c for c in F.columns if c not in ("method", "label", "model_seed", "interpolated", "fa_range")]
        G = F.groupby(["method", "label"], sort=False)[num].mean().reset_index()
        G["interpolated"] = F.groupby(["method", "label"], sort=False)["interpolated"].all().values
        G["fa_range"] = F.groupby(["method", "label"], sort=False)["fa_range"].first().values
        G["_o"] = G["method"].map({m: i for i, m in enumerate(ORDER)})
        G = G.sort_values("_o").drop(columns="_o")
        lab = {"S0": "STGNN before (S0 detection)", "S4": "STGNN after (S3/S4 detection)"}
        G["label"] = [lab.get(m, l) for m, l in zip(G["method"], G["label"])]
        out = G[["label"] + [c for c in ["recall", "precision", "F1", "bg_all"] + scn if c in G] +
                ["interpolated", "fa_range"]].rename(columns={
                    "label": "Method", "interpolated": "Interpolated", "fa_range": "FA range of the curve",
                    **{f"R_{k}": SCN_TITLE[k] for k in SCN_ORDER}, **_COLS})
        out.insert(1, "FA budget (controls) /1000 h", budget)
        T["equal_budget"] = out.reset_index(drop=True)
    # classification accuracy
    rows = []
    if len(res.oracle):
        cfgs = fz_cfg or {"full": fz.FuzzyConfig()}
        acc = fzc.accuracy_table(res.oracle, {"Oracle intervals, original rules": cfgs["full"],
                                              **({"Oracle intervals, calibrated rules": cfgs["calibrated"]}
                                                 if "calibrated" in cfgs else {})})
        acc = acc.set_index("scenario")
    else:
        acc = pd.DataFrame()
    det = {}
    for m, lbl in (("S0", "Detected, S0 (original rules)"), ("S4", "Detected, S4 (calibrated rules)")):
        if len(res.classes):
            det[lbl] = _class_share(res.classes[res.classes["method"] == m], res.events[res.events["method"] == m])
    if len(acc) or det:
        idx = [SCN_TITLE[s] for s in SCN_ORDER if s in SCN_TITLE]
        Tc = pd.DataFrame(index=list(acc.index) if len(acc) else idx + ["All plant-level events"])
        for c in acc.columns:
            Tc[c] = acc[c]
        for lbl, sr in det.items():
            sr = sr.rename(index={**SCN_TITLE, "all": "All plant-level events"})
            Tc[lbl] = sr.reindex(Tc.index)
        T["classification"] = Tc.reset_index().rename(columns={"index": "Scenario", "scenario": "Scenario"})
    if res.meta.get("fuzzy_calibration"):
        T["fuzzy_calibration"] = fzc.calibration_table(res.meta["fuzzy_calibration"])
    # recall by cohort at the operating points
    E = res.events
    if "cohort" in E and E["cohort"].notna().any():
        g = E[E["level"].isin(["plant"]) & (E["duration"] != "short")]
        tab = g.groupby(["method", "cohort"])["detected"].mean().unstack()
        tab = tab.reindex([m for m in ORDER if m in tab.index])
        tab.columns = [ssg.COHORT_TITLES.get(int(c), f"Cohort {c}") for c in tab.columns]
        labs = A.set_index("method")["label"]
        tab.insert(0, "Method", [labs.get(m, m) for m in tab.index])
        T["cohort_recall"] = tab.reset_index(drop=True)
    return T


def show(res: FixesResult, digits: int = 3):
    from IPython.display import display, Markdown
    titles = {"aggregation": "Hourly aggregation: zero fill vs NaN-aware rule (min_count)",
              "operating_points": "Operating points (q chosen on validation at the false-alarm budget of S0)",
              "scenario_recall": "Recall by scenario at the operating points",
              "equal_budget": "All methods at the same false-alarm budget on the clean test data (interpolated)",
              "classification": "Fuzzy classification: share of events with the correct type",
              "fuzzy_calibration": "Calibrated membership parameters (calibration placements disjoint from evaluation)",
              "cohort_recall": "Recall by commissioning cohort at the operating points"}
    for k, t in titles.items():
        if k in res.tables and len(res.tables[k]):
            display(Markdown(f"**{t}**")); display(res.tables[k].round(digits))


# ============================================================================ figures
COLORS = {"S0": "#444444", "S1": "#8c8c8c", "S2": "#1f77b4", "S3": "#17becf", "S4": "#2ca02c",
          "pr_rule": "#ff7f0e", "prp_rule": "#8c564b", "zero_rule": "#d62728", "iforest": "#9467bd"}


def plot_all(res: FixesResult, fig_dir: Optional[str] = "results/figures") -> Dict[str, object]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter
    figs = {}
    Cv = res.curves.copy()
    num = [c for c in Cv.columns if pd.api.types.is_numeric_dtype(Cv[c]) and c not in ("q", "min_len")]
    Cv["x"] = Cv["q"].fillna(Cv["min_len"])
    A = Cv.groupby(["method", "label", "x"], sort=False)[num].mean().reset_index()
    lab = {"S0": "STGNN before (S0)", "S4": "STGNN after (S3/S4 detection)"}
    P = _mean_by_method(res.points)
    budget = float(P.loc[P["method"] == "S0", "clean_fa"].iloc[0]) if (P["method"] == "S0").any() else np.nan
    f, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for m in [m for m in ORDER if m in set(A["method"])]:
        g = A[A["method"] == m].sort_values("clean_fa")
        name = lab.get(m, BASELINE_LABEL.get(m, m).replace("Baseline: ", ""))
        ax[0].plot(g["clean_fa"], g["recall"], "o-", c=COLORS.get(m), label=name, ms=4)
        ax[1].plot(g["clean_fa"], g["F1"], "o-", c=COLORS.get(m), label=name, ms=4)
    for m in [m for m in ORDER if m in set(P["method"])]:
        r = P[P["method"] == m].iloc[0]
        ax[0].scatter([r["clean_fa"]], [r["recall"]], marker="*", s=140, c=COLORS.get(m), edgecolor="k", zorder=5)
    for a in ax:
        a.axvline(budget, c="grey", ls=":", lw=1)
        a.set_xscale("symlog", linthresh=0.1)
        a.set_xlim(left=0)
        a.xaxis.set_major_locator(FixedLocator([0, 0.1, 0.3, 1, 3, 10, 30, 100]))
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.set_xlabel("false alarms on control plants, per 1000 h (clean test)")
        a.grid(alpha=.3)
    ax[0].set(ylabel="recall (day + week events)", title="Recall vs false alarms (stars: operating points)")
    ax[1].set(ylabel="F1", title="F1 vs false alarms (dotted: budget of S0)")
    ax[0].legend(fontsize=7, loc="lower right")
    f.tight_layout(); figs["tradeoff"] = f
    # per-scenario recall at the operating points
    S = res.tables.get("scenario_recall")
    if S is not None and len(S):
        cols = [SCN_TITLE[k] for k in SCN_ORDER if SCN_TITLE[k] in S]
        sel = [m for m in ("S0", "S4", "pr_rule", "prp_rule", "zero_rule", "iforest") if m in set(P["method"])]
        f, ax = plt.subplots(figsize=(14, 4.6))
        w = 0.8 / max(len(sel), 1); x = np.arange(len(cols))
        for i, m in enumerate(sel):
            lbl = P.loc[P["method"] == m, "label"].iloc[0]
            r = S[S["Method"] == lbl].iloc[0]
            ax.bar(x + i * w - 0.4 + w / 2, [r[c] for c in cols], w, color=COLORS.get(m), label=lbl)
        ax.set_xticks(x); ax.set_xticklabels(cols, rotation=15, ha="right", fontsize=8)
        ax.set(ylabel="recall", ylim=(0, 1.05), title="Recall by scenario at the operating points "
                                                         "(same validation false-alarm budget)")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3, axis="y")
        f.tight_layout(); figs["scenarios"] = f
    Tc = res.tables.get("classification")
    if Tc is not None and len(Tc):
        cols = [c for c in Tc.columns if c not in ("Scenario", "N")]
        f, ax = plt.subplots(figsize=(14, 4.6))
        w = 0.8 / max(len(cols), 1); x = np.arange(len(Tc))
        pal = ["#bbbbbb", "#2ca02c", "#444444", "#98df8a"]
        for i, c in enumerate(cols):
            ax.bar(x + i * w - 0.4 + w / 2, Tc[c].astype(float).fillna(0).values, w, label=c, color=pal[i % len(pal)])
        ax.set_xticks(x); ax.set_xticklabels(Tc["Scenario"], rotation=15, ha="right", fontsize=8)
        ax.set(ylabel="share with the correct type", ylim=(0, 1.05), title="Fuzzy classification before / after calibration")
        ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
        f.tight_layout(); figs["classification"] = f
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        for k, fg in figs.items():
            fg.savefig(os.path.join(fig_dir, f"fig_fixes_{k}.png"), dpi=150, bbox_inches="tight")
    return figs


# ============================================================================ LaTeX / export
TEX_CAPTIONS = {
    "aggregation": "Hourly aggregation of PV power: zero fill before resampling vs NaN-aware rule with minimum coverage",
    "operating_points": "Detector before (S0) and after the fixes (S1--S4) and simple baselines at operating points chosen on validation at the false-alarm budget of S0 (frozen STGNN weights, semi-synthetic test)",
    "scenario_recall": "Recall by scenario at the operating points",
    "equal_budget": "Methods compared at the same false-alarm budget on the clean test data (linear interpolation along the threshold curve)",
    "classification": "Share of events with the correct fuzzy class: original vs calibrated rules",
    "fuzzy_calibration": "Calibrated membership parameters (calibration placements disjoint from the evaluation placements)",
    "cohort_recall": "Recall by commissioning cohort at the operating points",
}


def export_latex(res: FixesResult, out_dir: str = "results/fixes") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths, allp = {}, ["% requires \\usepackage{booktabs,graphicx,longtable,array}", ""]
    for k, cap in TEX_CAPTIONS.items():
        t = res.tables.get(k)
        if t is None or not len(t):
            continue
        tex = sse.df_to_latex(t, cap, f"tab:fixes_{k}", digits=3)
        p = os.path.join(out_dir, f"fixes_{k}.tex")
        with open(p, "w", encoding="utf-8") as f:
            f.write(tex)
        paths[k] = p; allp.append(tex)
        t.to_csv(os.path.join(out_dir, f"fixes_{k}.csv"), index=False)
    res.points.to_csv(os.path.join(out_dir, "fixes_points_by_seed.csv"), index=False)
    res.curves.to_csv(os.path.join(out_dir, "fixes_curves_by_seed.csv"), index=False)
    p = os.path.join(out_dir, "fixes_tables.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(allp))
    paths["all"] = p
    return paths


def zip_outputs(out_dir: str = "results/fixes", fig_dir: str = "results/figures", name: str = "fixes_results.zip") -> str:
    import zipfile
    p = os.path.join(out_dir, name)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out_dir)):
            if f.endswith((".tex", ".csv")):
                z.write(os.path.join(out_dir, f), f)
        if fig_dir and os.path.isdir(fig_dir):
            for f in sorted(os.listdir(fig_dir)):
                if f.startswith("fig_fixes_"):
                    z.write(os.path.join(fig_dir, f), os.path.join("figures", f))
    return p


download_link = sse.download_link
