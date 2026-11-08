"""
pv_prplus.py -- PR+ baseline as a score family for the semi-synthetic protocols of sections 12 and 14.

PR+ is the same detector as in sections 15, 16 and 18 (``pv_modal_eval.prp_fam``):

    rho = (P / P_exp) / r0(plant, hour, month)       hour x month baseline of the training period
    rho = rho / median(rho of the technical-cluster peers)          (fleet median as fallback)
    rho = rho / causal 30-day rolling median of rho                 (closed='left', >= 50 daylight hours)
    s   = |rho - 1| on daylight hours,  per-plant threshold = quantile q of the training scores

The peer normalisation and the rolling median are imported from ``pv_modal_eval`` so that the definition
is identical in every protocol.  The only addition is a warm-up of the rolling median: every scored set
(validation, clean test, injected placements) is preceded by the last ``roll_days`` days of the clean
set that ends before it (training -> validation -> test), so the first test weeks are not scored without
a rolling reference.  The warm-up hours are never scored.

The module only reads the raw outputs of the frozen models (``pv_semisynth_sensitivity.ModelRaw``);
no model is retrained and no existing result is modified.  All labels are in English.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

LABEL = "Baseline: PR+ (hour x month, peers, rolling)"
SHORT = "PR+"


@dataclass
class PRPlusConfig:
    hm_min_n: int = 20          # min daylight hours per (hour) and (hour, month) cell of the baseline
    min_peers: int = 3          # min scored peers for the cluster median
    roll_days: int = 30         # causal rolling window, days
    roll_min: int = 50          # min daylight hours in the rolling window
    warmup: bool = True         # prepend the tail of the preceding clean set to the rolling median


def _labels(R) -> Optional[np.ndarray]:
    lab = R.calib.get("cluster_labels") if isinstance(R.calib, dict) else None
    n = R.calib["n_pv"]
    return lab.reindex(R.pv_cols[:n]).to_numpy() if lab is not None else None


def _base_table(y: np.ndarray, a: np.ndarray, ts: pd.DatetimeIndex, day: np.ndarray, r0: np.ndarray,
                cfg: PRPlusConfig) -> np.ndarray:
    """(n, 24, 13) median P/P_exp per plant, hour and month; falls back to hour, then to r0."""
    n = y.shape[1]
    with np.errstate(all="ignore"):
        r = y / np.maximum(a, 1e-6)
    H, Mo = ts.hour.to_numpy(), ts.month.to_numpy()
    base = np.ones((n, 24, 13)) * r0[:, None, None]
    for j in range(n):
        dj = day[:, j] & (y[:, j] > 0)
        if dj.sum() < 20:
            continue
        d = pd.DataFrame({"h": H[dj], "m": Mo[dj], "r": r[dj, j]})
        for h, row in d.groupby("h")["r"].agg(["median", "size"]).iterrows():
            if row["size"] >= cfg.hm_min_n:
                base[j, int(h), :] = row["median"]
        for (h, m), row in d.groupby(["h", "m"])["r"].agg(["median", "size"]).iterrows():
            if row["size"] >= cfg.hm_min_n:
                base[j, int(h), int(m)] = row["median"]
    return base


def prp_family(R, ghis: Dict[object, np.ndarray], cfx=None, cfg: Optional[PRPlusConfig] = None):
    """PR+ score family with the structure of ``pv_fixes_eval.Family`` (sets: val, clean, placements)."""
    import pv_fixes_eval as fx
    import pv_modal_eval as me
    cfx = cfx or fx.FixesConfig(); cfg = cfg or PRPlusConfig()
    n = R.calib["n_pv"]; lab = _labels(R)
    r0 = fx._plant_level(R, ghis["cal"], cfx)

    def yad(raw, ghi):
        y = raw["y"].mean(-1)[:, :n].astype(float); a = raw["a"].mean(-1)[:, :n].astype(float)
        return y, a, fx._daylight(raw, ghi, n, cfx)

    yc, ac, dc = yad(R.cal, ghis["cal"])
    base = _base_table(yc, ac, pd.DatetimeIndex(R.cal["ts"]), dc, r0, cfg)

    def rho_of(raw, ghi):
        y, a, day = yad(raw, ghi)
        ts = pd.DatetimeIndex(raw["ts"])
        with np.errstate(all="ignore"):
            rr = y / np.maximum(a, 1e-6)
        b = base[:, ts.hour.to_numpy(), ts.month.to_numpy()].T
        rho = me._peer_norm(rr / np.maximum(b, 1e-6), day, lab, cfg.min_peers)
        return rho, day, ts

    hist = {"cal": rho_of(R.cal, ghis["cal"]), "val": rho_of(R.val, ghis["val"])}

    def warm(ts0: pd.Timestamp):
        """Tail (roll_days) of the latest clean set that ends before ts0."""
        best = None
        for k, (rho, day, ts) in hist.items():
            if len(ts) and ts[-1] < ts0 and (best is None or ts[-1] > best[2][-1]):
                best = (rho, day, ts)
        if best is None:
            return None
        rho, day, ts = best
        m = ts > ts0 - pd.Timedelta(days=cfg.roll_days)
        return rho[m], day[m], ts[m]

    def score(raw, ghi, nr, allow_warm=True):
        rho, day, ts = rho_of(raw, ghi)
        w = warm(ts[0]) if (cfg.warmup and allow_warm and len(ts)) else None
        if w is not None and len(w[2]):
            rr = me._rolling(np.vstack([w[0], rho]), np.vstack([w[1], day]), w[2].append(ts),
                             cfg.roll_days, cfg.roll_min)[len(w[2]):]
        else:
            rr = me._rolling(rho, day, ts, cfg.roll_days, cfg.roll_min)
        s = np.where(day, np.abs(rr - 1.0), 0.0).astype(np.float64)
        s[fx._bad(raw, n, nr)] = np.nan
        return s, day

    cs, cday = score(R.cal, ghis["cal"], None, allow_warm=False)
    cs = np.where(np.isfinite(cs), cs, 0.0)
    sets = {}
    for key, raw, nr in fx._raw_sets(R, ghis):
        s, _ = score(raw, ghis[key], nr)
        sets[key] = dict(raw=raw, comb=s, zp=np.zeros_like(s), eta=raw["eta"], nr=nr)
    cache = {}

    def thr_fn(q):
        q = float(q)
        if q not in cache:
            t = fx._node_quantile_thr(cs, cday, q)
            cache[q] = {k: np.broadcast_to(t, v["comb"].shape) for k, v in sets.items()}
        return cache[q]
    return fx.Family("prp_rule", sets, thr_fn)
