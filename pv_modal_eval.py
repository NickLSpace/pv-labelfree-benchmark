"""Section 15: inverter modalities of HKUST — stages 1, 2, 4 and 5 of the validation plan.

Stage 0 (data audit) is in ``pv_modal_data``; stage 3 (semi-synthetic scenarios consistent
across channels) in ``pv_semisynth_modal_gen``; the multimodal STGNN in ``pv_stgnn_multimodal``.

* **Stage 1 — does a modality reduce the residual spread?**  Robust spread (1.4826 MAD) of
  log(P/P_exp) on clear training hours: one-plane vs multi-plane P_exp (multi-orientation
  sites), site-peer ratio and inverter-sibling ratio; share of "normal" hours that hide
  5-min outages; lift of the AC over-voltage among low-PR hours; by season and sky condition.
* **Stage 2 — what explains the real alarms?**  PR and PR+ alarms (and STGNN alarms if the
  frozen models are given) on the clean 2023 test, attributed with the modal channels to
  metering / grid / inverter / no modal cause / no inverter data.
* **Stage 4 — equal alarm budget on the modal scenarios.**  Methods (``METHODS``): PR rule,
  PR+ (hour x month baseline, peer normalisation inside the technical cluster, causal rolling
  median; P_exp is already the multi-plane model), modal rules alone, PR+ with modal decision
  fusion (inverter rules OR, grid / metering suppression, cause attribution), STGNN (S4 detector
  of section 12, daylight), STGNN with modal decision fusion, STGNN-mm early fusion,
  STGNN-mm with modal edges, and the last one with decision fusion.  Equal budget = the same
  number of alarm events per 1000 scored hours on the clean test of ALL plants.
  Metrics: recall of the fault scenarios, nuisance-alarm share of the non-fault scenarios
  (grid curtailment, metering fault), excess alarms of the other plants during the pyranometer
  drift, macro-F1 of the cause (plant / inverter / grid / data), time to detection; clean
  alarm rate of sites with / without inverter data; leave-one-modality-out; STGNN-mm with the
  modal channels removed at test time.
* **Stage 5 — pre-registered criteria** (``verdict_table``): a modality is useful if
  (a) the recall gain has a 95 % CI above 0, or (b) nuisance alarms fall by >= 20 % with recall
  not lower (CI of the difference >= -0.02), or (c) macro-F1 rises by >= 0.10.  STGNN-mm has to
  beat PR+ with the same modalities (recall CI > 0).

Without frozen models the rule-based part (PR, PR+, modal rules) runs on the hourly frame.
"""
from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import pv_modal_data as md
import pv_semisynth_modal_gen as smg

IRR = md.IRR
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
           9: "SON", 10: "SON", 11: "SON"}
METHODS = {
    "pr": dict(fam="pr", modal=False, label="PR threshold rule"),
    "prp": dict(fam="prp", modal=False, label="PR+ (hour x month, peers, rolling)"),
    "rules": dict(fam="rules", modal=True, label="Modal rules only"),
    "prp_m": dict(fam="prp", modal=True, label="PR+ with modal decision fusion"),
    "stgnn": dict(fam="stgnn", modal=False, label="STGNN (S4, daylight)"),
    "stgnn_m": dict(fam="stgnn", modal=True, label="STGNN with modal decision fusion"),
    "mm_early": dict(fam="mm_early", modal=False, label="STGNN-mm early fusion"),
    "mm_graph": dict(fam="mm_graph", modal=False, label="STGNN-mm early fusion + modal edges"),
    "mm_graph_m": dict(fam="mm_graph", modal=True, label="STGNN-mm + modal edges + decision fusion"),
    # section 16 (present only when shading / Kalman uncertainty or the factorized model are given)
    "prp_sk": dict(fam="prp_sk", modal=False, label="PR+ with shading mask and Kalman sd"),
    "prp_skm": dict(fam="prp_sk", modal=True, label="PR+ with shading mask, Kalman sd and modal decision fusion"),
    "mm_fact": dict(fam="mm_fact", modal=False, label="STGNN-fmm (factorized, generative imputation, cross-modal graph)"),
    "mm_fact_m": dict(fam="mm_fact", modal=True, label="STGNN-fmm + modal decision fusion"),
}
SHORT = {"pr": "PR", "prp": "PR+", "rules": "Rules", "prp_m": "PR+ & modal", "stgnn": "STGNN",
         "stgnn_m": "STGNN & modal", "mm_early": "STGNN-mm", "mm_graph": "STGNN-mm+edges",
         "mm_graph_m": "STGNN-mm+edges & modal", "prp_sk": "PR+ (shade, sd)", "prp_skm": "PR+ (shade, sd) & modal",
         "mm_fact": "STGNN-fmm", "mm_fact_m": "STGNN-fmm & modal"}
COLORS = {"pr": "#ff7f0e", "prp": "#d62728", "rules": "#8c564b", "prp_m": "#9467bd", "stgnn": "#98df8a",
          "stgnn_m": "#2ca02c", "mm_early": "#17becf", "mm_graph": "#1f77b4", "mm_graph_m": "#08306b",
          "prp_sk": "#e377c2", "prp_skm": "#7f3b08", "mm_fact": "#bcbd22", "mm_fact_m": "#636363"}
ROBUST_DROPS = {"mm_graph": [("no modal channels", tuple(md.MODALITIES))],
                "mm_fact": [("no modal channels", tuple(md.MODALITIES)), ("no shading", ("shading",)),
                            ("no Kalman sd", ("kalman",))]}
ROBUST_LABEL = {"mm_graph": "STGNN-mm + modal edges", "mm_fact": "STGNN-fmm"}
SCN_SHORT = {"inverter_trip": "Inverter trip", "short_outage": "Short outages", "phase_loss": "Phase loss",
             "grid_curtail": "Grid curtailment", "metering": "Meter fault", "drift_fault": "Drift + plant fault",
             "new_obstruction": "New obstruction"}


def _log(*a):
    print(time.strftime("%H:%M:%S"), "[modal-eval]", *a, flush=True)


@dataclass
class ModalEvalConfig:
    budgets: Sequence[float] = (0.5, 1.0, 2.0)          # clean alarm events per 1000 scored hours (all plants)
    ref_budget: float = 1.0
    min_len: int = 2
    q_grid: int = 81
    day_frac: float = 0.05
    day_ghi: float = 50.0
    # PR+
    hm_min_n: int = 20
    min_peers: int = 3
    roll_days: int = 30
    roll_min: int = 50
    # modal rules (pre-registered; per-site thresholds never looser than these values)
    rule_q: float = 0.999            # per-site calibration quantile of the inverter channels (training)
    sib_thr: float = 0.6
    zero_min: float = 2.0            # 5-min zero samples in an hour (>= 10 min)
    zero_ghi: float = 200.0
    imb_thr: float = 0.6
    dc_off_thr: float = 0.5
    meter_thr: float = 0.15          # |log(site / sum inverters)|
    vac_margin: float = 0.0          # V above the per-inverter over-voltage threshold
    f_limit: float = 0.2
    pad_h: int = 2
    n_boot: int = 2000
    seed: int = 0
    eta_mode: str = "med7d"


# ============================================================================ array sets
def _rated(M: "md.ModalData", pv: Sequence[str]) -> np.ndarray:
    return np.array([max(M.rated_w.get(c) or 1.0, 1.0) for c in pv])


def _active_frame(D: dict, pv: Sequence[str]) -> np.ndarray:
    idx = D["df"].index
    a = D["observed"].reindex(index=idx, columns=pv, fill_value=True).to_numpy(bool).copy() if "observed" in D \
        else np.ones((len(idx), len(pv)), bool)
    if D.get("act") is not None:
        a &= D["act"].reindex(index=idx, columns=pv, fill_value=False).to_numpy(bool)
    return a


def _mk_set(ts, y, a, active, ghi, M, pv, site_p, nr=None, raw=None):
    feats = M.site_arrays(pv, ts, site_p=site_p)
    act = active & np.isfinite(y) & np.isfinite(a)
    if nr is not None:
        act &= ~nr
    return dict(ts=pd.DatetimeIndex(ts), y=np.nan_to_num(y), a=np.nan_to_num(a), active=act, ghi=np.nan_to_num(ghi),
                feats=feats, raw=raw, nr=nr)


def sets_from_frame(D: dict, M: "md.ModalData", reals: Dict[object, "smg.ModalRealisation"], boundary: str) -> dict:
    """Hourly sets for the rule-based methods without a neural model (cal = training, clean = test)."""
    pv = list(D["pv_cols"]); idx = D["df"].index; b = pd.Timestamp(boundary)
    rw = _rated(M, pv); act = _active_frame(D, pv)
    out = {}
    for key, df, phys, MM in [("cal", D["df"], D["phys"], M), ("clean", D["df"], D["phys"], M)] + \
                             [(k, r.df, r.phys, r.modal) for k, r in reals.items()]:
        rows = np.asarray(idx < b) if key == "cal" else np.asarray(idx >= b)
        P = df[pv].to_numpy(float)[rows]
        y = P / rw; a = np.asarray(phys["p_expected"], float)[rows] / rw
        out[key] = _mk_set(idx[rows], y, a, act[rows], df[IRR].to_numpy(float)[rows], MM, pv, P)
    return out


def sets_from_raw(R, D: dict, M: "md.ModalData", reals: Dict[object, "smg.ModalRealisation"]) -> dict:
    """Sets from the raw outputs of a frozen (multimodal) STGNN: same scored hours as the model."""
    n = R.calib["n_pv"]; pv = list(R.pv_cols[:n])
    out = {}
    src = [("cal", R.cal, None, D["df"], M), ("val", R.val, getattr(R, "val_not_ready", None), D["df"], M),
           ("clean", R.clean, R.not_ready, D["df"], M)] + \
          [(k, R.inj[k], R.not_ready, reals[k].df, reals[k].modal) for k in reals if k in R.inj]
    for key, raw, nr, df, MM in src:
        ts = pd.DatetimeIndex(raw["ts"])
        y = raw["y"].mean(-1)[:, :n].astype(float); a = raw["a"].mean(-1)[:, :n].astype(float)
        P = df[pv].reindex(ts).to_numpy(float)
        out[key] = _mk_set(ts, y, a, raw["active"][:, :n], df[IRR].reindex(ts).to_numpy(float), MM, pv, P,
                           nr=None if nr is None else np.asarray(nr)[:, :n], raw=raw)
    return out


def _day(S: dict, cfg: ModalEvalConfig) -> np.ndarray:
    return (S["a"] > cfg.day_frac) & (S["ghi"][:, None] > cfg.day_ghi) & S["active"]


# ============================================================================ score families
@dataclass
class Fam:
    name: str
    comb: Dict[object, np.ndarray]               # key -> (S, n) score (NaN = not scored)
    thr: object                                  # q -> {key: (S, n) or (n,) threshold}
    zero: Dict[object, np.ndarray] = field(default_factory=dict)   # extra alarm masks (zero rule)


def _fallback(v: np.ndarray, labels: Optional[np.ndarray]) -> np.ndarray:
    have = np.isfinite(v); fleet = np.nanmedian(v) if have.any() else 1.0
    out = v.copy()
    for j in np.flatnonzero(~have):
        same = have & (labels == labels[j]) if labels is not None and pd.notna(labels[j]) else np.zeros(len(v), bool)
        out[j] = np.nanmedian(v[same]) if same.any() else fleet
    return out


def _labels(C: dict, pv: Sequence[str]) -> Optional[np.ndarray]:
    t = C.get("tech") if C else None
    return t.labels.reindex(pv).to_numpy() if t is not None and getattr(t, "labels", None) is not None else None


def _qthr(cs: np.ndarray, cday: np.ndarray, q: float, min_n: int = 20) -> np.ndarray:
    allv = cs[cday & np.isfinite(cs)]
    fleet = float(np.quantile(allv, q)) if allv.size else np.inf
    return np.array([np.quantile(cs[cday[:, j] & np.isfinite(cs[:, j]), j], q)
                     if (cday[:, j] & np.isfinite(cs[:, j])).sum() >= min_n else fleet for j in range(cs.shape[1])])


def _finish(sets, score_fn, cfg, name) -> Fam:
    comb, days = {}, {}
    for k, S in sets.items():
        s, d = score_fn(S, k)
        s = np.where(d, s, 0.0).astype(float); s[~S["active"]] = np.nan
        comb[k] = s; days[k] = d
    cache = {}

    def thr(q):
        q = float(q)
        if q not in cache:
            t = _qthr(comb["cal"], days["cal"], q)
            cache[q] = {k: t for k in sets}
        return cache[q]
    return Fam(name, comb, thr)


def pr_fam(sets: dict, C: dict, pv: Sequence[str], cfg: ModalEvalConfig) -> Fam:
    cal = sets["cal"]; d = _day(cal, cfg) & (cal["y"] > 0)
    with np.errstate(all="ignore"):
        r = cal["y"] / np.maximum(cal["a"], 1e-6)
    r0 = np.array([np.median(r[d[:, j], j]) if d[:, j].sum() >= 20 else np.nan for j in range(r.shape[1])])
    r0 = _fallback(r0, _labels(C, pv))

    def score(S, k):
        with np.errstate(all="ignore"):
            return np.abs(S["y"] / np.maximum(S["a"], 1e-6) / r0[None, :] - 1.0), _day(S, cfg)
    return _finish(sets, score, cfg, "pr")


def _peer_norm(rho: np.ndarray, day: np.ndarray, labels: Optional[np.ndarray], min_peers: int) -> np.ndarray:
    import warnings
    n = rho.shape[1]; X = np.where(day, rho, np.nan); out = np.ones_like(rho)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for j in range(n):
            others = np.ones(n, bool); others[j] = False
            grp = others & (labels == labels[j]) if labels is not None and pd.notna(labels[j]) else others
            med = np.full(len(rho), np.nan)
            cnt = np.isfinite(X[:, grp]).sum(1)
            if grp.any():
                med = np.where(cnt >= min_peers, np.nanmedian(X[:, grp], 1), np.nan)
            cnt_all = np.isfinite(X[:, others]).sum(1)
            fleet = np.where(cnt_all >= min_peers, np.nanmedian(X[:, others], 1), np.nan)
            med = np.where(np.isfinite(med), med, fleet)
            out[:, j] = np.where(np.isfinite(med) & (med > 0), med, 1.0)
    return rho / out


def _rolling(rho: np.ndarray, day: np.ndarray, ts: pd.DatetimeIndex, days: int, min_n: int) -> np.ndarray:
    X = pd.DataFrame(np.where(day, rho, np.nan), index=ts)
    med = X.rolling(f"{days}D", closed="left", min_periods=min_n).median().to_numpy()
    return rho / np.where(np.isfinite(med) & (med > 0), med, 1.0)


def prp_fam(sets: dict, C: dict, pv: Sequence[str], cfg: ModalEvalConfig) -> Fam:
    """PR+: r / r0(plant, hour, month) -> / median of the cluster peers -> / causal 30-day rolling median."""
    cal = sets["cal"]; n = cal["y"].shape[1]; lab = _labels(C, pv)
    d = _day(cal, cfg) & (cal["y"] > 0)
    with np.errstate(all="ignore"):
        r = cal["y"] / np.maximum(cal["a"], 1e-6)
    r0 = _fallback(np.array([np.median(r[d[:, j], j]) if d[:, j].sum() >= 20 else np.nan for j in range(n)]), lab)
    H, Mo = cal["ts"].hour.to_numpy(), cal["ts"].month.to_numpy()
    base = np.ones((n, 24, 13))
    for j in range(n):
        base[j] *= r0[j]
        dj = d[:, j]
        if dj.sum() < 20:
            continue
        df_ = pd.DataFrame({"h": H[dj], "m": Mo[dj], "r": r[dj, j]})
        gh = df_.groupby("h")["r"].agg(["median", "size"])
        for h, row in gh.iterrows():
            if row["size"] >= cfg.hm_min_n:
                base[j, int(h), :] = row["median"]
        g = df_.groupby(["h", "m"])["r"].agg(["median", "size"])
        for (h, m), row in g.iterrows():
            if row["size"] >= cfg.hm_min_n:
                base[j, int(h), int(m)] = row["median"]

    def score(S, k):
        dd = _day(S, cfg)
        with np.errstate(all="ignore"):
            rr = S["y"] / np.maximum(S["a"], 1e-6)
        b = base[:, S["ts"].hour.to_numpy(), S["ts"].month.to_numpy()].T          # (S, n)
        rho = rr / np.maximum(b, 1e-6)
        rho = _peer_norm(rho, dd, lab, cfg.min_peers)
        rho = _rolling(rho, dd, S["ts"], cfg.roll_days, cfg.roll_min)
        sc = np.abs(rho - 1.0)
        if S.get("kfac") is not None:            # section 16: score in units of the Kalman predictive sd
            sc = sc / S["kfac"]
        return sc, dd
    return _finish(sets, score, cfg, "prp")


def stgnn_fam(R, sets: dict, cfg: ModalEvalConfig) -> Fam:
    """S4 detector of section 12 (eta_hat med7d + rule "P ~ 0 at GHI > 200"), alarms on daylight hours."""
    import pv_fixes_eval as fx
    cfx = fx.FixesConfig(eta_mode=cfg.eta_mode, min_len_after=cfg.min_len)
    base = fx.stgnn_family(R, cfg.eta_mode)
    n = R.calib["n_pv"]
    comb, zero = {}, {}
    for k, s in base.sets.items():
        if k not in sets:
            continue
        S = sets[k]; c = s["comb"].astype(float).copy()
        day = _day(S, cfg)
        c = np.where(day | ~np.isfinite(c), c, -1e6)
        comb[k] = c
        zero[k] = fx.zero_rule_mask(s["raw"], S["ghi"], n, s["nr"], cfx)
    thr = lambda q: {k: v for k, v in base.thr_fn(q).items() if k in comb}
    return Fam(base.name, comb, thr, zero)


# ============================================================================ modal rules
@dataclass
class Rules:
    inv: Dict[object, np.ndarray]            # (S, n) inverter-side fault flags (alarm)
    grid: Dict[object, np.ndarray]           # grid-side cause (suppress, notice "grid")
    meter: Dict[object, np.ndarray]          # metering cause (suppress, notice "data")
    thr: pd.DataFrame                        # per-site thresholds (calibrated on training)


def _rule_thresholds(cal: dict, cfg: ModalEvalConfig, pv: Sequence[str]) -> pd.DataFrame:
    F = cal["feats"]; d = _day(cal, cfg)
    rows = []
    for j, c in enumerate(pv):
        m = d[:, j] & (F["has"][:, j] > 0)
        sib = F["sib"][m, j]; z = F["n_zero"][m & (cal["ghi"] > cfg.zero_ghi), j]; im = F["imb"][m, j]
        q = cfg.rule_q
        rows.append(dict(column=c,
                         sib=min(cfg.sib_thr, float(np.nanquantile(sib, 1 - q))) if np.isfinite(sib).sum() >= 50 else cfg.sib_thr,
                         n_zero=max(cfg.zero_min, float(np.nanquantile(z, q)) + 1) if np.isfinite(z).sum() >= 50 else cfg.zero_min,
                         imb=max(cfg.imb_thr, float(np.nanquantile(im, q))) if np.isfinite(im).sum() >= 50 else cfg.imb_thr))
    return pd.DataFrame(rows).set_index("column")


def make_rules(sets: dict, cfg: ModalEvalConfig, pv: Sequence[str], drop: Sequence[str] = ()) -> Rules:
    T = _rule_thresholds(sets["cal"], cfg, pv)
    inv, grid, meter = {}, {}, {}
    for k, S in sets.items():
        F = S["feats"]; d = _day(S, cfg) & (F["has"] > 0); sun = d & (S["ghi"][:, None] > cfg.zero_ghi)
        with np.errstate(invalid="ignore"):
            fl = np.zeros_like(d)
            if "inverter" not in drop:
                fl |= d & (np.nan_to_num(F["sib"], nan=9) < T["sib"].to_numpy()[None, :])
                fl |= sun & (np.nan_to_num(F["dc_off"]) >= cfg.dc_off_thr)
            if "subhourly" not in drop:
                fl |= sun & (np.nan_to_num(F["n_zero"]) >= T["n_zero"].to_numpy()[None, :])
            if "phase" not in drop:
                fl |= d & (np.nan_to_num(F["imb"]) > T["imb"].to_numpy()[None, :])
            g = np.zeros_like(d); mt = np.zeros_like(d)
            if "grid" not in drop:
                g = d & ((np.nan_to_num(F["vac"], nan=-99) > cfg.vac_margin) | (np.nan_to_num(F["f_dev"]) > cfg.f_limit))
            if "metering" not in drop:
                mt = d & (np.abs(np.nan_to_num(F["meter"])) > cfg.meter_thr)
        inv[k], grid[k], meter[k] = fl, g, mt
    return Rules(inv, grid, meter, T)


# ============================================================================ detector
def _run_mask(M: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1:
        return M.copy()
    out = np.zeros_like(M, bool)
    for j in range(M.shape[1]):
        m = M[:, j]
        if not m.any():
            continue
        dd = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)):
            if b - a >= min_len:
                out[a:b, j] = True
    return out


def _n_events(em: np.ndarray) -> int:
    z = np.zeros((1, em.shape[1]), np.int8)
    return int((np.diff(np.concatenate([z, em.astype(np.int8), z]), axis=0) == 1).sum())


class Det:
    def __init__(self, method: str, fam: Optional[Fam], rules: Optional[Rules], sets: dict, cfg: ModalEvalConfig,
                 modal: bool):
        self.method, self.fam, self.rules, self.sets, self.cfg, self.modal = method, fam, rules, sets, cfg, modal

    def scored(self, key) -> np.ndarray:
        if self.fam is not None:
            return np.isfinite(self.fam.comb[key])
        return self.sets[key]["active"].copy()

    def mask(self, key, q) -> np.ndarray:
        if self.fam is not None and q is not None:
            c = self.fam.comb[key]; t = self.fam.thr(q)[key]
            with np.errstate(invalid="ignore"):
                exc = np.nan_to_num(c, nan=-np.inf) > (t if np.ndim(t) == 2 else t[None, :])
            em = _run_mask(exc, self.cfg.min_len)
            if key in self.fam.zero:
                em |= self.fam.zero[key]
        else:
            em = np.zeros_like(self.sets[key]["active"])
        if self.modal and self.rules is not None:
            em = em | self.rules.inv[key]
            em = em & ~(self.rules.grid[key] | self.rules.meter[key])
        return em & self.scored(key)

    def rate(self, key, q) -> float:
        em = self.mask(key, q); h = int(self.scored(key).sum())
        return 1000.0 * _n_events(em) / h if h else np.nan


def _grid(cfg: ModalEvalConfig) -> np.ndarray:
    z = np.linspace(np.log(0.9 / 0.1), np.log(0.99999 / 0.00001), cfg.q_grid)
    return 1.0 / (1.0 + np.exp(-z))


def match_q(det: Det, budget: float, grid: np.ndarray, cache: dict) -> Tuple[Optional[float], float]:
    if det.fam is None:
        return None, det.rate("clean", None)
    for q in grid:
        q = float(q)
        if q not in cache:
            cache[q] = det.rate("clean", q)
        if np.isfinite(cache[q]) and cache[q] <= budget:
            return q, cache[q]
    q = float(grid[-1])
    return q, cache.get(q, det.rate("clean", q))


# ============================================================================ evaluation of one detector
def _attrib(rules: Rules, key, pos: np.ndarray, j: int, alarm: np.ndarray) -> str:
    if rules is None:
        return "plant"
    a = pos[alarm[pos, j]] if alarm is not None else pos
    if not len(a):
        return "plant"
    if rules.meter[key][a, j].mean() > 0.5:
        return "data"
    if rules.grid[key][a, j].mean() > 0.5:
        return "grid"
    return "inverter" if rules.inv[key][a, j].any() else "plant"


def eval_events(det: Det, key, q, real: "smg.ModalRealisation", pv: List[str], tag: dict) -> Tuple[List[dict], dict]:
    S = det.sets[key]; ts = S["ts"]; em = det.mask(key, q)
    rows = []
    for r in real.labels.itertuples():
        if r.column not in pv:
            continue
        j = pv.index(r.column)
        st = r.eff_start if pd.notna(getattr(r, "eff_start", pd.NaT)) else r.start
        w = (ts >= r.start) & (ts <= r.end + pd.Timedelta(hours=det.cfg.pad_h))
        pos = np.flatnonzero(w)
        sc = det.scored(key)[pos, j] if len(pos) else np.array([], bool)
        if not len(pos) or not sc.any():
            rows.append(dict(tag, event_id=r.event_id, scenario=r.scenario, cls=r.cls, expected_alarm=bool(r.expected_alarm),
                             column=r.column, scored=False, detected=False, ttd_h=np.nan, pred="none")); continue
        hit = em[pos, j]
        detected = bool(hit.any())
        ttd = float((ts[pos[np.argmax(hit)]] - st) / pd.Timedelta(hours=1)) if detected else np.nan
        if detected:
            pred = _attrib(det.rules, key, pos, j, em) if det.modal else "plant"
        elif det.modal and det.rules is not None:
            g, m = det.rules.grid[key][pos, j].sum(), det.rules.meter[key][pos, j].sum()
            pred = "data" if m > 0 and m >= g else ("grid" if g > 0 else "none")
        else:
            pred = "none"
        rows.append(dict(tag, event_id=r.event_id, scenario=r.scenario, cls=r.cls, expected_alarm=bool(r.expected_alarm),
                         column=r.column, scored=True, detected=detected, ttd_h=ttd, pred=pred))
    extra = {}
    if real.drift is not None:          # alarms of the non-faulty plants during the drift window vs clean
        tgt = set(real.labels.loc[real.labels["scenario"] == "drift_fault", "column"])
        others = np.array([c not in tgt for c in pv])
        w = (ts >= real.drift["start"]) & (ts <= real.drift["end"])
        def _rate(k):
            E = det.mask(k, q); Sk = det.sets[k]["ts"]
            ww = (Sk >= real.drift["start"]) & (Sk <= real.drift["end"])
            h = int(det.scored(k)[ww][:, others].sum())
            return 1000.0 * _n_events(E[ww][:, others]) / h if h else np.nan
        extra = dict(drift_rate=_rate(key), drift_rate_clean=_rate("clean"))
    return rows, extra


# ============================================================================ stage 1 / stage 2 (real data)
def _rob(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) >= 30 else np.nan


def _one_plane_pexp(D: dict, col: str) -> Optional[np.ndarray]:
    brick, mapping, builder = D.get("brick"), D.get("mapping") or {}, D.get("builder")
    if brick is None or builder is None or not hasattr(builder, "prior"):
        return None
    import copy as _c
    s = brick.sites.get(mapping.get(col))
    if s is None or len(set(zip(s.tilts_deg or [], s.azimuths_deg or []))) < 2:
        return None
    s1 = _c.copy(s)
    az = np.deg2rad(np.asarray(s.azimuths_deg, float))
    s1.tilts_deg = [float(np.mean(s.tilts_deg))]
    s1.azimuths_deg = [float(np.rad2deg(np.arctan2(np.sin(az).mean(), np.cos(az).mean())) % 360)]
    df = D["df"]; t = pd.DatetimeIndex(df.index); age = (t - t[0]).days.to_numpy(float) / 365.25
    p, _ = builder.prior.expected_power(df[IRR].to_numpy(float), t, df["Temp (Degree Celsius)"].to_numpy(float),
                                        df["wind_speed"].to_numpy(float), s1, 8.0, age)
    return p


def stage1(D: dict, C: dict, M: "md.ModalData", cfg: Optional[ModalEvalConfig] = None,
           boundary: str = "2023-01-01") -> Dict[str, pd.DataFrame]:
    cfg = cfg or ModalEvalConfig()
    pv = list(D["pv_cols"]); df = D["df"]; idx = df.index; b = pd.Timestamp(boundary)
    S = sets_from_frame(D, M, {}, boundary)
    full = {k: np.concatenate([S["cal"][k], S["clean"][k]]) for k in ("y", "a", "active", "ghi")}
    ts = S["cal"]["ts"].append(S["clean"]["ts"])
    feats = {f: np.concatenate([S["cal"]["feats"][f], S["clean"]["feats"][f]]) for f in md.SITE_FEATS}
    clear = (full["ghi"][:, None] > 200) & full["active"] & (full["a"] > cfg.day_frac) & (full["y"] > 0)
    with np.errstate(all="ignore"):
        r = np.log(full["y"] / np.maximum(full["a"], 1e-6))
    r = np.where(clear, r, np.nan)
    lab = _labels(C, pv)
    peer = np.log(_peer_norm(np.exp(np.nan_to_num(r)), clear, lab, cfg.min_peers))
    peer = np.where(clear, peer, np.nan)
    train = np.asarray(ts < b)
    kt = D.get("kt")
    ktv = kt.reindex(ts).to_numpy(float) if kt is not None else np.full(len(ts), np.nan)
    sky = np.where(ktv > 0.6, "clear sky (kt > 0.6)", np.where(ktv > 0.3, "partly cloudy", "overcast (kt <= 0.3)"))
    season = np.array([SEASONS[m] for m in ts.month])
    # --- (a) one-plane vs multi-plane P_exp
    rows = []
    for j, c in enumerate(pv):
        p1 = _one_plane_pexp(D, c)
        if p1 is None:
            continue
        p_multi = np.asarray(D["phys"]["p_expected"], float)[:, j]
        P = df[c].to_numpy(float)
        m = np.asarray(idx < b) & (df[IRR].to_numpy(float) > 200) & _active_frame(D, [c])[:, 0] & (P > 0) & (p_multi > 0) & (p1 > 0)
        h = idx.hour.to_numpy()
        def spread(pe):
            lr = np.log(P[m] / pe[m]); med_h = pd.Series(lr).groupby(h[m]).median()
            return _rob(lr), float(med_h.max() - med_h.min()) if len(med_h) else np.nan
        s1, d1 = spread(p1); s2, d2 = spread(p_multi)
        st_ = D["brick"].sites[D["mapping"][c]]
        rows.append({"Site": md._short(c), "Orientations": len(set(zip(st_.tilts_deg, st_.azimuths_deg))),
                     "Spread log(P/P_exp), one plane": s1, "Spread log(P/P_exp), multi-plane": s2,
                     "Hour-of-day bias range, one plane": d1, "Hour-of-day bias range, multi-plane": d2})
    if not rows:   # HKUST Brick model: every site has a single tilt/azimuth
        rows = [{"Site": "(no multi-orientation site in the Brick model)", "Orientations": 1,
                 "Spread log(P/P_exp), one plane": np.nan, "Spread log(P/P_exp), multi-plane": np.nan,
                 "Hour-of-day bias range, one plane": np.nan, "Hour-of-day bias range, multi-plane": np.nan}]
    T = {"orientation": pd.DataFrame(rows)}
    # --- (b) spread of site ratio vs site-peer ratio vs inverter-sibling ratio
    rows = []
    sib = np.where(clear, np.log(np.clip(feats["sib"], 1e-3, None)), np.nan)
    for gname, G in [("All", np.full(len(ts), "all"))] + [("Season", season), ("Sky", sky)]:
        for v in pd.unique(G):
            g = (G == v)[:, None]
            for scope, cols in (("sites with inverter data", np.array([c in M.sites() for c in pv])),
                                ("sites with >= 2 inverters", np.array([len(M.inverters_of(c)) >= 2 for c in pv]))):
                mm = g & cols[None, :]
                rows.append({"Stratum": gname if gname == "All" else f"{gname}: {v}", "Sites": scope,
                             "Spread log(P/P_exp)": _rob(r[mm]), "Spread log(site / cluster peers)": _rob(peer[mm]),
                             "Spread log(inverter / siblings)": _rob(sib[mm]) if scope.startswith("sites with >=") else np.nan,
                             "Hours": int(np.isfinite(r[mm]).sum())})
    T["spread"] = pd.DataFrame(rows)
    # --- (c) short outages hidden in normal hours, (d) over-voltage among low-PR hours
    rows = []
    r0 = np.nanmedian(np.where(train[:, None], r, np.nan), 0)
    rel = r - r0[None, :]
    for j, c in enumerate(pv):
        if c not in M.sites():
            continue
        has = feats["has"][:, j] > 0
        normal = clear[:, j] & has & (rel[:, j] > np.log(0.8))
        low = clear[:, j] & has & (rel[:, j] < np.log(0.8)) & (full["ghi"] > 300)
        prod = clear[:, j] & has
        ov = np.nan_to_num(feats["vac"][:, j], nan=-99) > cfg.vac_margin
        base = ov[prod].mean() if prod.any() else np.nan
        rows.append({"Site": md._short(c), "Normal clear hours": int(normal.sum()),
                     "Normal hours with >= 1 zero 5-min sample, %": 100 * float((feats["n_zero"][normal, j] >= 1).mean()) if normal.any() else np.nan,
                     "Normal hours with >= 2 zero 5-min samples, %": 100 * float((feats["n_zero"][normal, j] >= 2).mean()) if normal.any() else np.nan,
                     "Low-PR hours (< 0.8 r0, GHI > 300)": int(low.sum()),
                     "Over-voltage share, low-PR hours, %": 100 * float(ov[low].mean()) if low.any() else np.nan,
                     "Over-voltage share, all producing hours, %": 100 * float(base) if np.isfinite(base) else np.nan})
    X = pd.DataFrame(rows)
    if len(X):
        X["Lift of over-voltage in low-PR hours"] = X["Over-voltage share, low-PR hours, %"] / X["Over-voltage share, all producing hours, %"].replace(0, np.nan)
    T["subhourly_grid"] = X
    return T


def stage2(D: dict, C: dict, M: "md.ModalData", cfg: Optional[ModalEvalConfig] = None, boundary: str = "2023-01-01",
           budget: Optional[float] = None, extra: Optional[Dict[str, Tuple[Fam, dict]]] = None) -> Dict[str, pd.DataFrame]:
    """Attribution of the real (clean 2023) alarms of PR and PR+ (and optional extra detectors)."""
    cfg = cfg or ModalEvalConfig(); budget = budget or cfg.ref_budget
    pv = list(D["pv_cols"]); sets = sets_from_frame(D, M, {}, boundary)
    rules = make_rules(sets, cfg, pv)
    fams = {"pr": (pr_fam(sets, C, pv, cfg), sets), "prp": (prp_fam(sets, C, pv, cfg), sets)}
    fams.update(extra or {})
    rows, ev = [], []
    for name, (fam, ss) in fams.items():
        rl = rules if ss is sets else make_rules(ss, cfg, pv)
        det = Det(name, fam, rl, ss, cfg, modal=False)
        q, rate = match_q(det, budget, _grid(cfg), {})
        em = det.mask("clean", q); ts = ss["clean"]["ts"]
        for j, c in enumerate(pv):
            m = em[:, j]
            if not m.any():
                continue
            dd = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
            for a, b2 in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)):
                pos = np.arange(a, b2)
                has = ss["clean"]["feats"]["has"][pos, j] > 0
                if c not in M.sites():
                    cause = "no inverter data (site without optimisers)"
                elif not has.any():
                    cause = "inverter data missing"
                else:
                    k = _attrib(rl, "clean", pos[has], j, None)
                    cause = {"data": "metering mismatch", "grid": "grid over-voltage / frequency",
                             "inverter": "inverter state / siblings / 5-min outages / phase", "plant": "no modal cause (PV side)"}[k]
                ev.append(dict(method=name, plant=md._short(c), start=ts[a], end=ts[b2 - 1], hours=int(b2 - a), cause=cause))
        rows.append(dict(method=name, q=q, clean_rate=rate))
    E = pd.DataFrame(ev, columns=["method", "plant", "start", "end", "hours", "cause"])
    if len(E):
        share = E.groupby(["method", "cause"]).size().unstack(fill_value=0)
        share = (100 * share.div(share.sum(1), axis=0)).round(1)
        share.insert(0, "Alarm events", E.groupby("method").size())
        share.index = [METHODS.get(i, {}).get("label", i) for i in share.index]
    else:
        share = pd.DataFrame()
    # fleet-frequency events: hours where >= 30 % of sites with inverter data alarm at once
    fleet = []
    if len(E):
        for name, g in E.groupby("method"):
            hrs = pd.concat([pd.Series(pd.date_range(r.start, r.end, freq="h")) for r in g.itertuples()])
            cnt = hrs.value_counts()
            fleet.append(dict(method=METHODS.get(name, {}).get("label", name),
                              **{"Hours with alarms at >= 30 % of the sites": int((cnt >= 0.3 * len(pv)).sum())}))
    return {"attribution": share.reset_index().rename(columns={"index": "Method"}) if len(share) else share,
            "alarms": E, "operating_points": pd.DataFrame(rows), "fleet": pd.DataFrame(fleet)}


# ============================================================================ stage 4 orchestration
@dataclass
class ModalResult:
    events: pd.DataFrame
    points: pd.DataFrame
    drift: pd.DataFrame
    strata: pd.DataFrame
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    stage0: Dict[str, pd.DataFrame] = field(default_factory=dict)
    stage1: Dict[str, pd.DataFrame] = field(default_factory=dict)
    stage2: Dict[str, pd.DataFrame] = field(default_factory=dict)
    stage16: Dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def sk_sets(sets: dict, D: dict, C: dict, reals: dict, shading=None, kunc=None, boundary: str = "2023-01-01") -> dict:
    """Section 16 copies of the hourly sets: ``a`` -> a * (1 - s_eff) (generated shading mask) and
    ``kfac`` = clip(Kalman sd / training median of the plant, 0.5, 3); ``s_eff`` kept for strata."""
    pv = list(D["pv_cols"]); cache = {}
    def full(df, phys):
        if id(df) not in cache:
            T, n = len(df), len(pv)
            se = np.nan_to_num(shading.channels(D, pv, df=df)["s_eff"]) if shading is not None else np.zeros((T, n))
            if kunc is not None:
                kc = kunc.channels(D, C, pv, df=df, phys=phys)
                sd = np.where(kc["has"] > 0, kc["k_sd_tot"], np.nan)
            else:
                sd = np.full((T, n), np.nan)
            cache[id(df)] = (pd.DatetimeIndex(df.index), se, sd)
        return cache[id(df)]
    idx0, _, sd0 = full(D["df"], D["phys"])
    tr = np.asarray(idx0 < pd.Timestamp(boundary))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(sd0[tr], 0)
    out = {}
    for k, S in sets.items():
        df, phys = (reals[k].df, reals[k].phys) if k in reals else (D["df"], D["phys"])
        idx, se, sd = full(df, phys)
        pos = idx.get_indexer(S["ts"]); ok = pos >= 0
        s_eff = np.zeros(S["a"].shape); s_eff[ok] = se[pos[ok]]
        kf = np.ones(S["a"].shape)
        if kunc is not None:
            with np.errstate(all="ignore"):
                r = sd[pos[ok]] / med[None, :]
            kf[ok] = np.where(np.isfinite(r), np.clip(r, 0.5, 3.0), 1.0)
        out[k] = dict(S, a=S["a"] * (1.0 - np.clip(s_eff, 0, 1)), kfac=kf, s_eff=s_eff)
    return out


def _loo_methods() -> List[str]:
    return [f"prp_m-{g}" for g in md.MODALITIES]


def run_modal(D: dict, C: dict, M: "md.ModalData", models: Optional[dict] = None, mm_models: Optional[Dict[str, dict]] = None,
              placement_seeds: Sequence[int] = (0, 1, 2), cfg: Optional[ModalEvalConfig] = None,
              gen_cfg: Optional["smg.ModalGenConfig"] = None, out_dir: str = "results/modal", use_cache: bool = True,
              robust: bool = True, stages_012: bool = True, boundary: str = "2023-01-01",
              shading=None, kunc=None) -> ModalResult:
    """Stages 1, 2 (real data), 3 (scenarios) and 4 (equal budget).  ``models``: {seed: FrozenSTGNN}
    of the multi-view variant; ``mm_models``: {"mm_early"/"mm_graph": {seed: FrozenMM}}.  Without
    models only the rule-based methods run (on the hourly frame).  Section 16: ``shading``
    (pv_shading.ShadingMask) and ``kunc`` (pv_kalman_uncertainty.KalmanUncertainty) add PR+ with the
    shading mask and the Kalman sd; ``mm_models["mm_fact"]`` = {seed: pv_stgnn_factorized.FrozenFMM}."""
    import pv_stgnn_multimodal as smm
    cfg = cfg or ModalEvalConfig(); gen_cfg = gen_cfg or smg.ModalGenConfig(boundary=boundary)
    t0 = time.time(); os.makedirs(out_dir, exist_ok=True)
    pv = list(D["pv_cols"])
    plans = {ps: smg.plan_modal(D, C, M, ps, gen_cfg) for ps in placement_seeds}
    reals = {f"m{ps}": smg.apply_modal(D, M, p) for ps, p in plans.items()}
    _log(f"{sum(len(r.labels) for r in reals.values())} modal events in {len(reals)} placements")
    grid = _grid(cfg)
    budgets = sorted(set(float(b) for b in cfg.budgets) | {float(cfg.ref_budget)})
    # --- per model seed: sets + families
    blocks = []                                 # (seed, {method: (fam, sets)})
    if models:
        import pv_fixes_eval as fx
        raws = fx.collect(models, D, reals, os.path.join(out_dir, "raw"), None, 64, use_cache)
        # raw outputs of the multimodal models are kept on disk and loaded one seed at a time, so the
        # memory does not grow with the number of seeds (results identical to loading all at once)
        mm_paths = {}
        for var, mods in (mm_models or {}).items():
            for s, mdl in mods.items():
                f = os.path.join(out_dir, "raw", f"raw_{var}_{s}_{mdl.fingerprint()}_{_sig(reals)}.pkl")
                if not (use_cache and os.path.exists(f)):
                    _log(f"{var} seed {s}: forward passes")
                    pickle.dump(smm.collect_mm(mdl, D, reals), open(f, "wb"))
                mm_paths[(var, s)] = f
                for tag_, drop_ in (ROBUST_DROPS.get(var, []) if robust else []):
                    f2 = f.replace(".pkl", "_nomodal.pkl" if tag_ == "no modal channels" else f"_{tag_.replace(' ', '_')}.pkl")
                    if not (use_cache and os.path.exists(f2)):
                        _log(f"{var} seed {s}: forward passes, {tag_}")
                        pickle.dump(smm.collect_mm(mdl, D, reals, drop=drop_), open(f2, "wb"))
                    mm_paths[(var + "|" + tag_, s)] = f2

        def _block(s, R=None):
            R = raws[s] if R is None else R
            ss = sets_from_raw(R, D, M, reals)
            fams = {"pr": (pr_fam(ss, C, pv, cfg), ss), "prp": (prp_fam(ss, C, pv, cfg), ss),
                    "stgnn": (stgnn_fam(R, ss, cfg), ss), "rules": (None, ss)}
            if shading is not None or kunc is not None:
                sk = sk_sets(ss, D, C, reals, shading, kunc, boundary); fams["prp_sk"] = (prp_fam(sk, C, pv, cfg), sk)
            for (var, s2), fp in mm_paths.items():
                if s2 == s:
                    Rm = pickle.load(open(fp, "rb"))
                    sm = sets_from_raw(Rm, D, M, reals)
                    fams[var] = (stgnn_fam(Rm, sm, cfg), sm)
                    del Rm
            return fams
        blocks = [(s, _block) for s in raws]
    else:
        ss = sets_from_frame(D, M, reals, boundary)
        fams0 = {"pr": (pr_fam(ss, C, pv, cfg), ss), "prp": (prp_fam(ss, C, pv, cfg), ss), "rules": (None, ss)}
        if shading is not None or kunc is not None:
            sk = sk_sets(ss, D, C, reals, shading, kunc, boundary); fams0["prp_sk"] = (prp_fam(sk, C, pv, cfg), sk)
        blocks = [("-", lambda s: fams0)]
    has_inv = np.array([c in M.sites() for c in pv])
    EV, PT, DR, ST = [], [], [], []
    import gc
    for seed, build in blocks:
        fams = build(seed)
        sk_shade = {}                        # per seed (keys are object ids of this seed's sets)
        rules_cache = {}
        def rules_for(ss, drop=()):
            k = (id(ss), tuple(drop))
            if k not in rules_cache:
                rules_cache[k] = make_rules(ss, cfg, pv, drop)
            return rules_cache[k]
        specs = []
        for m, spec in METHODS.items():
            if spec["fam"] in fams:
                specs.append((m, spec["fam"], spec["modal"], ()))
        if "prp" in fams:
            specs += [(f"prp_m-{g}", "prp", True, (g,)) for g in md.MODALITIES]
        for var in [k for k in fams if "|" in k]:
            specs.append((var, var, False, ()))
        for m, fk, modal, drop in specs:
            fam, ss = fams[fk]
            det = Det(m, fam, rules_for(ss, drop), ss, cfg, modal)
            cache = {}
            for bud in budgets:
                if "|" in fk:                                  # q of the model WITH all channels (same seed)
                    bv = fk.split("|")[0]
                    base = Det(bv, fams[bv][0], None, fams[bv][1], cfg, False)
                    q, _ = match_q(base, bud, grid, {}); rate = det.rate("clean", q)
                else:
                    q, rate = match_q(det, bud, grid, cache)
                em = det.mask("clean", q); sc = det.scored("clean")
                def r_(cols):
                    h = int(sc[:, cols].sum())
                    return 1000.0 * _n_events(em[:, cols]) / h if h else np.nan
                tag = dict(method=m, model_seed=seed, budget=bud)
                pt = dict(tag, q=q, clean_rate=rate, rate_with_inv=r_(has_inv), rate_without_inv=r_(~has_inv))
                if shading is not None:                      # clean alarms in hours with a known shadow
                    se = sk_shade.setdefault(id(ss["clean"]), sk_sets({"clean": ss["clean"]}, D, C, {}, shading, None,
                                                                         boundary)["clean"]["s_eff"]) > 0.1
                    hh = int((sc & se).sum())
                    pt.update(shaded_hours=hh, rate_shaded=1000.0 * _n_events(em & se) / hh if hh else np.nan,
                              rate_unshaded=1000.0 * _n_events(em & ~se) / max(int((sc & ~se).sum()), 1))
                PT.append(pt)
                for k, real in reals.items():
                    rows, extra = eval_events(det, k, q, real, pv, dict(tag, placement=k))
                    EV += rows
                    if extra:
                        DR.append(dict(tag, placement=k, **extra))
                if bud == cfg.ref_budget:
                    sea = np.array([SEASONS[mo] for mo in ss["clean"]["ts"].month])
                    for v in pd.unique(sea):
                        mrow = sea == v
                        for grp, cols in (("with inverter data", has_inv), ("without inverter data", ~has_inv)):
                            h = int(sc[mrow][:, cols].sum())
                            ST.append(dict(tag, season=v, sites=grp, hours=h, alarms=_n_events(em[mrow][:, cols])))
        _log(f"seed {seed}: {len(specs)} detectors evaluated ({time.time() - t0:.0f}s)")
        del fams, rules_cache, sk_shade
        gc.collect()
    if models:
        for mdl in models.values():
            assert mdl.unchanged(), "weights changed"
        for mods in (mm_models or {}).values():
            for mdl in mods.values():
                assert mdl.unchanged(), "weights changed"
    res = ModalResult(events=pd.DataFrame(EV), points=pd.DataFrame(PT), drift=pd.DataFrame(DR), strata=pd.DataFrame(ST),
                      meta=dict(budgets=budgets, ref_budget=cfg.ref_budget, seeds=[b[0] for b in blocks],
                                placements=list(reals), n_boot=cfg.n_boot, neural=bool(models),
                                mm_variants=sorted((mm_models or {}).keys()), modal_source=M.source,
                                n_events=int(sum(len(r.labels) for r in reals.values())),
                                section16=bool(shading is not None or kunc is not None or "mm_fact" in (mm_models or {}))))
    res.meta["labels"] = pd.concat([r.labels.assign(placement=k) for k, r in reals.items()], ignore_index=True)
    if stages_012:
        res.stage0 = md.audit(D, M)
        res.stage1 = stage1(D, C, M, cfg, boundary)
        res.stage2 = stage2(D, C, M, cfg, boundary)
    if shading is not None or kunc is not None or "mm_fact" in (mm_models or {}):
        res.stage16 = section16_tables(D, C, shading, kunc, (mm_models or {}).get("mm_fact"), reals, boundary)
    res.tables = build_tables(res, cfg)
    with open(os.path.join(out_dir, "modal_result.pkl"), "wb") as f:
        pickle.dump(res, f)
    _log(f"done in {time.time() - t0:.0f}s")
    return res


def section16_tables(D, C, shading=None, kunc=None, fmm_models=None, reals=None, boundary="2023-01-01") -> Dict[str, pd.DataFrame]:
    """Section 16 diagnostics of the new modalities and of the factorized model (English tables)."""
    out = {}
    if shading is not None:
        import pv_shading as psh
        for k, v in psh.evaluate(D, C, shading).items():
            out[f"shading_{k}"] = v
    if kunc is not None:
        import pv_kalman_uncertainty as pku
        for k, v in pku.evaluate(D, C, kunc).items():
            out[f"kalman_{k}"] = v
    if fmm_models:
        import pv_stgnn_factorized as pfm
        m0 = fmm_models[sorted(fmm_models)[0]]
        out["fmm_crossmodal"] = pfm.crossmodal_mean(fmm_models)
        W = np.mean([m.fmm.net.edge_weights().detach().cpu().numpy() for m in fmm_models.values()], 0)
        out["fmm_edge_matrix"] = pd.DataFrame(W, index=pfm.MOD_NODES, columns=pfm.MOD_NODES)
        out["fmm_history"] = pfm.training_history(fmm_models)
        out["fmm_generation"] = pfm.generation_quality(m0, boundary)
        out["fmm_attention"] = pfm.attention_by_scenario(m0, reals or {})
        for m in fmm_models.values():
            m.alpha_log.clear()
    return out


S16_TITLES = {"shading_masks": "Section 16: shading masks (training period)",
              "shading_summary": "Section 16: shading correction on clear hours (median over sites; test = out of sample)",
              "kalman_calibration": "Section 16: causal Kalman predictive interval of the normalised power (daylight, valid hours)",
              "fmm_crossmodal": "Section 16: cross-modal graph of STGNN-fmm, prior vs learned edge weights",
              "fmm_history": "Section 16: training of the factorized module (stage A)",
              "fmm_generation": "Section 16: generative factor, modality generated with that modality hidden (test period)",
              "fmm_attention": "Section 16: change of the attention of the descriptive factor at event hours (event - clean, same plant and hours)"}


def _sig(reals) -> str:
    import hashlib
    h = hashlib.sha256()
    for k in sorted(reals):
        L = reals[k].labels[["event_id", "column", "scenario", "strength", "start", "end"]]
        h.update(str(k).encode()); h.update(pd.util.hash_pandas_object(L.astype(str), index=False).values.tobytes())
    return h.hexdigest()[:12]


# ============================================================================ statistics / tables
def _macro_f1(true: np.ndarray, pred: np.ndarray) -> float:
    f = []
    for c in smg.CLASSES:
        tp = np.sum((pred == c) & (true == c)); fp = np.sum((pred == c) & (true != c)); fn = np.sum((pred != c) & (true == c))
        if tp + fn == 0:
            continue
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f)) if f else np.nan


def _metrics(X: pd.DataFrame) -> dict:
    X = X[X["scored"]]
    fa = X[X["expected_alarm"]]; nf = X[~X["expected_alarm"]]
    return dict(recall=float(fa["detected"].mean()) if len(fa) else np.nan,
                nuisance=float(nf["detected"].mean()) if len(nf) else np.nan,
                macro_f1=_macro_f1(X["cls"].to_numpy(), X["pred"].to_numpy()) if len(X) else np.nan,
                ttd_h=float(fa.loc[fa["detected"], "ttd_h"].median()) if fa["detected"].any() else np.nan,
                n_fault=int(len(fa)), n_nonfault=int(len(nf)))


def paired(E: pd.DataFrame, a: str, b: str, budget: float, n_boot: int = 2000, seed: int = 0) -> dict:
    """Differences a - b on the same events (seed, placement, event): recall, nuisance, macro-F1; 95 % CI
    from an event-level cluster bootstrap (seeds of one event are resampled together)."""
    k = ["model_seed", "placement", "event_id"]
    X = E[(E["budget"] == budget) & E["scored"]]
    A = X[X["method"] == a].set_index(k); B = X[X["method"] == b].set_index(k)
    j = A.index.intersection(B.index)
    if not len(j):
        return dict(n=0)
    A, B = A.loc[j], B.loc[j]
    fa = A["expected_alarm"].to_numpy(bool)
    da, db = A["detected"].to_numpy(float), B["detected"].to_numpy(float)
    ca, cb, ct = A["pred"].to_numpy(), B["pred"].to_numpy(), A["cls"].to_numpy()
    # event-level cluster bootstrap: all model seeds of one (placement, event) are resampled together
    codes, _ = pd.factorize(pd.MultiIndex.from_arrays([j.get_level_values("placement"), j.get_level_values("event_id")]))
    groups = [np.flatnonzero(codes == c) for c in range(int(codes.max()) + 1)]
    rng = np.random.default_rng(seed)
    G = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    dr, dn, df1 = [], [], []
    for gg in G:
        ii = np.concatenate([groups[g] for g in gg])
        f = fa[ii]
        dr.append(da[ii][f].mean() - db[ii][f].mean() if f.any() else np.nan)
        dn.append(da[ii][~f].mean() - db[ii][~f].mean() if (~f).any() else np.nan)
        df1.append(_macro_f1(ct[ii], ca[ii]) - _macro_f1(ct[ii], cb[ii]))
    q = lambda v: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))) if np.isfinite(v).any() else (np.nan, np.nan)
    na, nb = (da[~fa].mean() if (~fa).any() else np.nan), (db[~fa].mean() if (~fa).any() else np.nan)
    return dict(n=int(len(j)), recall_a=float(da[fa].mean()) if fa.any() else np.nan,
                recall_b=float(db[fa].mean()) if fa.any() else np.nan,
                d_recall=float(da[fa].mean() - db[fa].mean()) if fa.any() else np.nan, recall_ci=q(np.array(dr)),
                nuisance_a=na, nuisance_b=nb, d_nuisance=na - nb, nuisance_ci=q(np.array(dn)),
                f1_a=_macro_f1(ct, ca), f1_b=_macro_f1(ct, cb), d_f1=_macro_f1(ct, ca) - _macro_f1(ct, cb), f1_ci=q(np.array(df1)))


def _crit(p: dict, nuis_drop: float = 0.2, f1_gain: float = 0.10, recall_floor: float = -0.02) -> Tuple[str, str]:
    if not p or not p.get("n"):
        return "no events", ""
    hits = []
    lo = p["recall_ci"][0]
    if np.isfinite(lo) and lo > 0:
        hits.append("(a) recall gain, CI > 0")
    nb, na = p["nuisance_b"], p["nuisance_a"]
    if np.isfinite(nb) and nb > 0 and np.isfinite(na) and (nb - na) / nb >= nuis_drop and np.isfinite(lo) and lo >= recall_floor:
        hits.append(f"(b) nuisance alarms -{100 * (nb - na) / nb:.0f} % at no lower recall")
    if np.isfinite(p["d_f1"]) and p["d_f1"] >= f1_gain and np.isfinite(p["f1_ci"][0]) and p["f1_ci"][0] > 0:
        hits.append(f"(c) macro-F1 +{p['d_f1']:.2f}")
    return ("useful" if hits else "not shown"), "; ".join(hits)


def verdict_table(res: ModalResult, cfg: Optional[ModalEvalConfig] = None) -> pd.DataFrame:
    cfg = cfg or ModalEvalConfig(); E = res.events; b = res.meta["ref_budget"]
    have = set(E["method"]) if len(E) else set()
    comps = [("All modalities with PR+", "prp_m", "prp")]
    comps += [(f"Modality: {md.MODALITY_LABEL[g]}", "prp_m", f"prp_m-{g}") for g in md.MODALITIES]
    comps += [("All modalities with STGNN (decision fusion)", "stgnn_m", "stgnn"),
              ("STGNN-mm (early fusion) vs STGNN", "mm_early", "stgnn"),
              ("STGNN-mm + modal edges vs STGNN-mm", "mm_graph", "mm_early"),
              ("STGNN-mm + modal edges vs STGNN", "mm_graph", "stgnn"),
              ("STGNN-mm + edges & modal vs PR+ & modal (same modalities)", "mm_graph_m", "prp_m"),
              ("PR+ with shading mask + Kalman sd vs PR+", "prp_sk", "prp"),
              ("New modalities (shading, Kalman sd) with PR+ & modal", "prp_skm", "prp_m"),
              ("STGNN-fmm vs STGNN-mm + modal edges", "mm_fact", "mm_graph"),
              ("STGNN-fmm vs STGNN", "mm_fact", "stgnn"),
              ("STGNN-fmm & modal vs PR+ (shade, sd) & modal (same information)", "mm_fact_m", "prp_skm")]
    rows = []
    for name, a, bb in comps:
        if a not in have or bb not in have:
            continue
        p = paired(E, a, bb, b, cfg.n_boot, cfg.seed)
        if name.startswith("STGNN-mm + edges & modal vs PR+") or name.startswith("STGNN-fmm & modal vs PR+"):
            lo = p["recall_ci"][0] if p.get("n") else np.nan
            v = ("beats PR+ (recall CI > 0)" if np.isfinite(lo) and lo > 0 else "does not beat PR+"), ""
        else:
            v = _crit(p)
        rows.append({"Comparison": name, "A": SHORT.get(a, a), "B": SHORT.get(bb, bb), "Events": p.get("n", 0),
                     "Recall A": p.get("recall_a"), "Recall B": p.get("recall_b"), "Delta recall": p.get("d_recall"),
                     "CI low": p.get("recall_ci", (np.nan,))[0], "CI high": p.get("recall_ci", (np.nan, np.nan))[1],
                     "Nuisance A": p.get("nuisance_a"), "Nuisance B": p.get("nuisance_b"),
                     "Macro-F1 A": p.get("f1_a"), "Macro-F1 B": p.get("f1_b"), "Verdict": v[0], "Criteria met": v[1]})
    return pd.DataFrame(rows)


def build_tables(res: ModalResult, cfg: Optional[ModalEvalConfig] = None) -> Dict[str, pd.DataFrame]:
    cfg = cfg or ModalEvalConfig(); T = {}
    E, P, b = res.events, res.points, res.meta["ref_budget"]
    if not len(E):
        return T
    rows = []
    for (m, bud), g in E.groupby(["method", "budget"]):
        pts = P[(P["method"] == m) & (P["budget"] == bud)]
        rows.append({"Method": METHODS.get(m.split("|")[0], {}).get("label", m) + (" (" + m.split("|")[1] + ")" if "|" in m else "")
                     if m in METHODS or "|" in m else f"PR+ & modal without {md.MODALITY_LABEL[m.split('-', 1)[1]]}",
                     "key": m, "Budget /1000 h": bud, "Clean FA /1000 h": pts["clean_rate"].mean(),
                     "Clean FA, sites with inverters": pts["rate_with_inv"].mean(),
                     "Clean FA, sites without inverters": pts["rate_without_inv"].mean(), **_metrics(g)})
    A = pd.DataFrame(rows)
    T["equal_budget"] = A[~A["key"].str.contains("-")].sort_values(["Budget /1000 h", "key"]).drop(columns="key").reset_index(drop=True)
    X = E[(E["budget"] == b) & E["scored"]]
    rec = X[X["expected_alarm"]].pivot_table(index="method", columns="scenario", values="detected", aggfunc="mean")
    nui = X[~X["expected_alarm"]].pivot_table(index="method", columns="scenario", values="detected", aggfunc="mean")
    S = pd.concat([rec.rename(columns=lambda c: f"Recall: {SCN_SHORT.get(c, c)}"),
                   nui.rename(columns=lambda c: f"Nuisance: {SCN_SHORT.get(c, c)}")], axis=1)
    S = S.loc[[m for m in list(METHODS) + sorted(set(S.index) - set(METHODS)) if m in S.index]]
    S.index = [SHORT.get(i, i) for i in S.index]
    T["by_scenario"] = S.reset_index().rename(columns={"index": "Method"})
    if len(res.drift):
        Dd = res.drift[res.drift["budget"] == b].groupby("method")[["drift_rate", "drift_rate_clean"]].mean()
        Dd["Excess alarms /1000 h"] = Dd["drift_rate"] - Dd["drift_rate_clean"]
        Dd.index = [SHORT.get(i, i) for i in Dd.index]
        T["drift"] = Dd.rename(columns={"drift_rate": "Other plants, drift window /1000 h",
                                        "drift_rate_clean": "Same window, clean /1000 h"}).reset_index().rename(columns={"index": "Method"})
    loo = []
    for g in md.MODALITIES:
        if f"prp_m-{g}" in set(E["method"]):
            p = paired(E, "prp_m", f"prp_m-{g}", b, cfg.n_boot, cfg.seed)
            loo.append({"Modality removed": md.MODALITY_LABEL[g], "Delta recall (full - without)": p.get("d_recall"),
                        "CI low": p.get("recall_ci", (np.nan,))[0], "CI high": p.get("recall_ci", (np.nan, np.nan))[1],
                        "Delta nuisance": p.get("d_nuisance"), "Delta macro-F1": p.get("d_f1")})
    T["leave_one_out"] = pd.DataFrame(loo)
    if len(res.strata):
        St = res.strata[res.strata["budget"] == b].groupby(["method", "season", "sites"])[["hours", "alarms"]].sum()
        St["FA /1000 h"] = 1000 * St["alarms"] / St["hours"].replace(0, np.nan)
        St = St["FA /1000 h"].unstack(["sites", "season"])
        St.columns = [f"{s}, {se}" for s, se in St.columns]
        St.index = [SHORT.get(i, i) for i in St.index]
        T["fa_strata"] = St.reset_index().rename(columns={"index": "Method"})
    rb = [m for m in set(E["method"]) if "|" in m]
    if rb:
        rows = []
        for bv in [v for v in ROBUST_DROPS if any(m.startswith(v + "|") for m in rb)]:
            for m in [bv] + sorted(m for m in rb if m.startswith(bv + "|")):
                g = X[X["method"] == m]; pts = P[(P["method"] == m) & (P["budget"] == b)]
                rows.append({"Model": ROBUST_LABEL[bv] + (f" ({m.split('|')[1]})" if "|" in m else " (all channels)"),
                             "Clean FA /1000 h (q fixed)": pts["clean_rate"].mean(), **_metrics(g)})
        T["robustness"] = pd.DataFrame(rows)
    if "rate_shaded" in P.columns:
        Z = P[(P["budget"] == b) & P["rate_shaded"].notna()].groupby("method")[["shaded_hours", "rate_shaded", "rate_unshaded"]].mean()
        Z = Z.loc[[m for m in list(METHODS) if m in Z.index]]
        Z.index = [SHORT.get(i, i) for i in Z.index]
        T["fa_shaded"] = Z.rename(columns={"shaded_hours": "Scored hours with s_eff > 0.1",
                                           "rate_shaded": "Clean FA /1000 h, shaded hours",
                                           "rate_unshaded": "Clean FA /1000 h, other hours"}).reset_index().rename(columns={"index": "Method"})
    T["verdict"] = verdict_table(res, cfg)
    return T


TITLES = {"equal_budget": "Stage 4: methods at equal clean alarm budgets (modal scenarios)",
          "by_scenario": "Stage 4: recall (fault scenarios) and nuisance-alarm share (non-fault scenarios) at the reference budget",
          "drift": "Stage 4: alarms of the other plants during the pyranometer drift",
          "leave_one_out": "Stage 4: leave-one-modality-out (PR+ with modal decision fusion)",
          "fa_strata": "Stage 4: clean false-alarm rate by season and inverter-data availability",
          "robustness": "Stage 4: multimodal STGNNs with modalities removed at test time (q of the full model)",
          "fa_shaded": "Section 16: clean false alarms in hours with a known shadow (s_eff > 0.1) vs other hours",
          "verdict": "Stage 5: pre-registered criteria"}
S1_TITLES = {"orientation": "Stage 1: one-plane vs multi-plane P_exp (multi-orientation sites, training, GHI > 200)",
             "spread": "Stage 1: robust spread of site ratio, site-peer ratio and inverter-sibling ratio",
             "subhourly_grid": "Stage 1: short outages hidden in normal hours; over-voltage among low-PR hours"}
S2_TITLES = {"attribution": "Stage 2: causes of the real 2023 alarms (share of alarm events, %)",
             "operating_points": "Stage 2: operating points (clean 2023 test)",
             "fleet": "Stage 2: fleet-wide alarm hours"}


def show(res: ModalResult, digits: int = 3):
    from IPython.display import display
    for grp, titles in ((res.stage0, md.TITLES), (res.stage1, S1_TITLES), (res.stage2, S2_TITLES), (res.tables, TITLES),
                        (getattr(res, "stage16", {}) or {}, S16_TITLES)):
        for k, t in titles.items():
            if k in grp and isinstance(grp[k], pd.DataFrame) and len(grp[k]):
                print(t); display(grp[k].round(digits) if hasattr(grp[k], "round") else grp[k])


def plot_all(res: ModalResult, fig_dir: Optional[str] = "results/figures") -> Dict[str, object]:
    import matplotlib.pyplot as plt
    figs = {}
    E, b = res.events, res.meta["ref_budget"]
    if len(E):
        X = E[(E["budget"] == b) & E["scored"]]
        ms = [m for m in METHODS if m in set(X["method"])]
        scn = [s for s in smg.ALL_SCENARIOS if s in set(X["scenario"])]
        fig, ax = plt.subplots(figsize=(12, 4.8))
        w = 0.8 / max(len(ms), 1)
        for i, m in enumerate(ms):
            v = [X[(X["method"] == m) & (X["scenario"] == s)]["detected"].mean() for s in scn]
            ax.bar(np.arange(len(scn)) + i * w, v, w, color=COLORS[m], label=SHORT[m])
        ax.set_xticks(np.arange(len(scn)) + 0.4 - w / 2)
        ax.set_xticklabels([SCN_SHORT[s] + ("\n(alarm = nuisance)" if not smg.ALL_SCENARIOS[s]["alarm"] else "") for s in scn])
        ax.set_ylabel("share of events with an alarm"); ax.set_ylim(0, 1.05)
        ax.set_title(f"Modal scenarios at an equal alarm budget ({b:g} per 1000 clean hours)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(len(ms), 5), frameon=False)
        fig.tight_layout(); figs["scenarios"] = fig
        if res.meta.get("section16"):
            fig.set_size_inches(14, 5.2)
        V = res.tables.get("verdict")
        if V is not None and len(V):
            fig, ax = plt.subplots(figsize=(9, 0.5 * len(V) + 1.5))
            y = np.arange(len(V))[::-1]
            ax.errorbar(V["Delta recall"], y, xerr=[V["Delta recall"] - V["CI low"], V["CI high"] - V["Delta recall"]],
                        fmt="o", color="#1f77b4", capsize=3)
            ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(V["Comparison"])
            ax.set_xlabel("difference in recall of the fault scenarios (A - B), 95 % CI")
            ax.set_title("Stage 5: recall differences (paired bootstrap)")
            fig.tight_layout(); figs["verdict"] = fig
    s1 = res.stage1.get("spread") if res.stage1 else None
    if s1 is not None and len(s1):
        Z = s1[s1["Sites"] == "sites with >= 2 inverters"].set_index("Stratum")
        cols = ["Spread log(P/P_exp)", "Spread log(site / cluster peers)", "Spread log(inverter / siblings)"]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        Z = Z.rename(index=lambda t: t.replace("Season: ", "").replace("Sky: ", "").replace(" (kt <= 0.3)", "").replace(" (kt > 0.6)", ""))
        Z[cols].plot.bar(ax=ax, color=["#ff7f0e", "#9467bd", "#1f77b4"], width=0.8)
        ax.set_ylabel("robust spread (1.4826 MAD of the log ratio)"); ax.set_xlabel("")
        ax.set_title("Stage 1: residual spread at sites with >= 2 inverters (clear hours, GHI > 200)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
        plt.setp(ax.get_xticklabels(), rotation=0)
        fig.tight_layout(); figs["stage1"] = fig
    s2 = res.stage2.get("attribution") if res.stage2 else None
    if s2 is not None and len(s2):
        Z = s2.set_index(s2.columns[0]).drop(columns=["Alarm events"])
        fig, ax = plt.subplots(figsize=(10, 3.8))
        Z.plot.barh(stacked=True, ax=ax, colormap="tab20")
        ax.set_xlabel("share of alarm events, %"); ax.set_ylabel("")
        ax.set_title("Stage 2: causes of the real 2023 alarms (modal attribution)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
        fig.tight_layout(); figs["stage2"] = fig
    Wm = (getattr(res, "stage16", {}) or {}).get("fmm_edge_matrix")
    if Wm is not None:
        import pv_stgnn_factorized as pfm
        figs["fmm_crossmodal"] = pfm.plot_crossmodal_matrix(Wm.to_numpy())
    A = (getattr(res, "stage16", {}) or {}).get("fmm_attention")
    if A is not None and len(A):
        Z = A.set_index("Scenario").drop(columns=["Events"], errors="ignore")
        Z.columns = [c.replace("Delta attention: ", "") for c in Z.columns]
        v = float(np.nanmax(np.abs(Z.to_numpy()))) or 1e-3
        fig, ax = plt.subplots(figsize=(11, 0.5 * len(Z) + 2.6))
        im = ax.imshow(Z.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(Z.shape[1])); ax.set_xticklabels(Z.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(Z))); ax.set_yticklabels(Z.index)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ax.text(j, i, f"{Z.iat[i, j]:+.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("STGNN-fmm: change of attention at event hours vs the same plant and hours without the event")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout(); figs["fmm_attention"] = fig
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        for k, f in figs.items():
            f.savefig(os.path.join(fig_dir, f"fig_modal_{k}.png"), dpi=150, bbox_inches="tight")
    return figs


def export_latex(res: ModalResult, out_dir: str = "results/modal") -> Dict[str, str]:
    import pv_semisynth_eval as sse
    os.makedirs(out_dir, exist_ok=True); out = {}
    for grp, titles, pre in ((res.stage0, md.TITLES, "s0"), (res.stage1, S1_TITLES, "s1"), (res.stage2, S2_TITLES, "s2"),
                             (res.tables, TITLES, "s4"), (getattr(res, "stage16", {}) or {}, S16_TITLES, "s16")):
        for k, t in titles.items():
            X = grp.get(k) if grp else None
            if X is None or not isinstance(X, pd.DataFrame) or not len(X):
                continue
            name = f"modal_{pre}_{k}"
            X.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
            p = os.path.join(out_dir, name + ".tex")
            with open(p, "w") as f:
                f.write(sse.df_to_latex(X, t, f"tab:{name}", digits=3))
            out[k if pre in ("s4", "s16") else f"{pre}_{k}"] = p
    return out


def zip_outputs(out_dir: str = "results/modal", fig_dir: str = "results/figures", name: str = "modal_results.zip") -> str:
    import zipfile
    z = os.path.join(out_dir, name)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(out_dir)):
            if fn.endswith((".tex", ".csv")):
                zf.write(os.path.join(out_dir, fn), fn)
        if fig_dir and os.path.isdir(fig_dir):
            for fn in sorted(os.listdir(fig_dir)):
                if fn.startswith("fig_modal_"):
                    zf.write(os.path.join(fig_dir, fn), os.path.join("figures", fn))
    return z
