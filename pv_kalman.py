"""
Kalman-filter alternative to SPS for the interpretable per-plant power model.

The regression is the same as in ``pv_sps.py`` (all powers normalised by rated power,
P_exp calibrated on the plant's own commissioning period):

    y_t = P_meas_{i,t} = b0_i + b1_i * P_exp,cal_{i,t} + b2_i * T_cell_t + b3_i * w_t + eps_{i,t}

but the coefficients are treated as a slowly varying STATE (random walk at the daily
time scale) instead of constants inside a period:

    theta_d     = theta_{d-1} + eta_d,         eta_d ~ N(0, Q * Delta_d)        (state)
    y_{d,h}     = phi_{d,h}' theta_d + eps,     eps ~ N(0, R_eff)                (observation)

* d is the calendar day; all valid hourly observations of the same day update the state
  in one step (information-form batch update).  Gaps of several days increase the
  prediction covariance by Q * Delta_d, so uncertainty grows where data are missing.
* R is estimated robustly from the OLS residual (1.4826 * MAD)^2 and inflated by a
  design effect computed from day-block residual sums.  This is the Gaussian analogue
  of the BLOCK sign perturbations used by SPS: hourly residuals within a day are
  correlated, day blocks are treated as independent.
* Q can be fixed or tuned per plant by maximising the innovation log-likelihood over a
  multiplicative grid (``tune_q=True``).
* A Rauch-Tung-Striebel smoother gives the retrospective trajectory b1_d | all data.

Outputs have exactly the same schema as the SPS functions (``sps_periods``,
``sps_split``, ``sps_cluster``), so ``fuse_evidence``, the report and the comparison
module can use either estimator.  Interpretation of b1 is unchanged:

    b1 ~ 1                       measurement agrees with the physical model
    sustained decrease of b1     possible degradation, soiling, shading or curtailment
    widening interval            lack of data or unstable regime
    disjoint consecutive periods possible regime change

Differences to SPS that matter for interpretation:
* SPS gives an EXACT finite-sample region under symmetric independent (block) noise;
  the Kalman interval is Gaussian / asymptotic and depends on Q and R.
* The Kalman filter provides a daily trajectory b1_d (``kalman_track``), so changes
  inside a quarter are visible and events can be matched to the local b1.

References
----------
R. E. Kalman (1960). "A New Approach to Linear Filtering and Prediction Problems."
  J. Basic Eng. 82(1):35-45. https://doi.org/10.1115/1.3662552
H. E. Rauch, F. Tung, C. T. Striebel (1965). "Maximum likelihood estimates of linear
  dynamic systems." AIAA J. 3(8):1445-1450. https://doi.org/10.2514/3.3166
A. C. Harvey (1990). Forecasting, Structural Time Series Models and the Kalman Filter.
  Cambridge University Press. https://doi.org/10.1017/CBO9781107049994
J. Durbin, S. J. Koopman (2012). Time Series Analysis by State Space Methods, 2nd ed.
  Oxford University Press. https://doi.org/10.1093/acprof:oso/9780199641178.001.0001
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from pv_cohorts import short
from pv_sps import PARAM_NAMES, reference_calibration, regression_frame

ESTIMATOR_NAME = "kalman"
ESTIMATOR_LABEL = "Kalman"


# ----------------------------------------------------------------------------- configuration
@dataclass
class KalmanConfig:
    """Hyper-parameters of the random-walk state-space model.

    q_daily  : per-day standard deviation of the random walk of [b0, b1, b2, b3].
               Default b1: 0.002 / day  ->  ~0.04 / year (i.e. a few % per year drift is
               tracked, hourly noise is not).  b2 (temperature) and b3 (cloudiness) are
               physical sensitivities and are kept almost constant: T_cell and P_exp are
               strongly correlated, so a freely drifting b2 would absorb the loss of b1.
    p0_sd    : prior standard deviation of the initial state around ``theta0``.
    theta0   : prior mean (b1 = 1 means 'as at commissioning' after calibration).
    tune_q   : choose a multiplicative scale of Q per plant by maximum innovation
               likelihood over ``q_grid``.
    deff_max : upper bound of the within-day design effect applied to R.
    smoother : use RTS-smoothed (True) or filtered (False) states in summaries.
    confidence : nominal level of the reported intervals (0.95 = SPS default 1-q/m).
    """
    q_daily: Sequence[float] = (2e-4, 2e-3, 2e-5, 2e-5)
    p0_sd: Sequence[float] = (0.5, 0.5, 0.5, 0.5)
    theta0: Sequence[float] = (0.0, 1.0, 0.0, 0.0)
    tune_q: bool = False
    q_grid: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0)
    deff_max: float = 24.0
    smoother: bool = True
    confidence: float = 0.95
    min_obs_day: int = 1

    @property
    def z(self) -> float:
        return float(norm.ppf(0.5 + self.confidence / 2.0))


@dataclass
class KalmanTrack:
    """Daily state trajectory of one plant."""
    column: str
    days: pd.DatetimeIndex
    n_obs: np.ndarray            # observations per day
    theta_f: np.ndarray          # filtered mean   (D, 4)
    P_f: np.ndarray              # filtered cov    (D, 4, 4)
    theta_s: np.ndarray          # smoothed mean   (D, 4)
    P_s: np.ndarray              # smoothed cov    (D, 4, 4)
    r: float                     # effective observation variance
    deff: float                  # design effect applied to R
    q_scale: float               # selected Q multiplier
    loglik: float
    calib_k: float
    theta_ols: np.ndarray = field(default_factory=lambda: np.full(4, np.nan))
    C: Optional[np.ndarray] = None   # RTS gains (D, 4, 4) for lag covariances

    def mean_b1(self, mask: np.ndarray, smoother: bool = True, idx: int = 1):
        """Mean of b1 over the selected days and the EXACT sd of that mean.

        With the RTS smoother Cov(x_k, x_j | Y) = C_k Cov(x_{k+1}, x_j | Y) (j > k), so the
        variance of a sum of smoothed states is accumulated by a backward recursion in O(D)."""
        mask = np.asarray(mask, bool)
        if not mask.any():
            return np.nan, np.nan
        th = self.theta_s if smoother else self.theta_f
        P = self.P_s if smoother else self.P_f
        m = float(th[mask, idx].mean())
        if not smoother or self.C is None:
            return m, float(np.sqrt(np.clip(P[mask, idx, idx], 0, None)).mean())
        D, d = th.shape
        G = np.zeros((d, d))
        tot = 0.0
        for k in range(D - 1, -1, -1):
            if k < D - 1:
                G = self.C[k] @ (G + (P[k + 1] if mask[k + 1] else 0.0))
            if mask[k]:
                tot += P[k, idx, idx] + 2.0 * G[idx, idx]
        n = mask.sum()
        return m, float(np.sqrt(max(tot, 0.0)) / n)

    def frame(self, smoother: bool = True) -> pd.DataFrame:
        th = self.theta_s if smoother else self.theta_f
        P = self.P_s if smoother else self.P_f
        out = pd.DataFrame({nm: th[:, k] for k, nm in enumerate(PARAM_NAMES)}, index=self.days)
        for k, nm in enumerate(PARAM_NAMES):
            out[f"sd_{nm}"] = np.sqrt(np.clip(P[:, k, k], 0, None))
        out["n"] = self.n_obs
        out.index.name = "day"
        return out


# ----------------------------------------------------------------------------- core filter
def _robust_r_and_deff(Phi: np.ndarray, y: np.ndarray, day_id: np.ndarray, deff_max: float):
    """Robust observation variance and within-day design effect from OLS residuals."""
    theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
    e = y - Phi @ theta
    mad = np.median(np.abs(e - np.median(e)))
    r = float((1.4826 * mad) ** 2) or float(np.var(e)) or 1e-6
    _, inv = np.unique(day_id, return_inverse=True)
    s = np.bincount(inv, weights=e)                     # residual sum per day
    n = np.bincount(inv)
    var_e = float(np.var(e)) or r
    deff = float(np.sum(s ** 2) / max(np.sum(n) * var_e, 1e-12))
    return r, float(np.clip(deff, 1.0, deff_max)), theta


def _filter(Phi, y, day_id, days, r, Q, theta0, P0, smoother=True):
    """Daily random-walk Kalman filter with batch (information-form) updates.

    Returns filtered/smoothed means & covariances and the innovation log-likelihood."""
    d = Phi.shape[1]
    D = len(days)
    th_f = np.zeros((D, d)); P_f = np.zeros((D, d, d))
    th_p = np.zeros((D, d)); P_p = np.zeros((D, d, d))
    th, P = np.asarray(theta0, float).copy(), np.asarray(P0, float).copy()
    order = np.argsort(day_id, kind="stable")
    Phi, y, day_id = Phi[order], y[order], day_id[order]
    bounds = np.searchsorted(day_id, days.asi8, side="left")
    bounds = np.append(bounds, len(day_id))
    ll = 0.0
    prev_day = None
    for k in range(D):
        # --- predict (random walk; step proportional to elapsed days)
        if prev_day is not None:
            gap = max((days[k] - prev_day).days, 1)
            P = P + Q * gap
        th_p[k], P_p[k] = th, P
        H = Phi[bounds[k]:bounds[k + 1]]
        z = y[bounds[k]:bounds[k + 1]]
        n = len(z)
        if n:
            # innovation likelihood via Woodbury (S = H P H' + r I)
            e = z - H @ th
            HtH = H.T @ H
            Pinv = np.linalg.inv(P)
            A = Pinv + HtH / r                                # posterior information
            Ainv = np.linalg.inv(A)
            Hte = H.T @ e
            quad = (e @ e) / r - (Hte @ Ainv @ Hte) / r ** 2
            sign, logdet_IA = np.linalg.slogdet(np.eye(d) + P @ HtH / r)
            ll += -0.5 * (n * np.log(2 * np.pi * r) + logdet_IA + quad)
            # update
            th = Ainv @ (Pinv @ th + H.T @ z / r)
            P = 0.5 * (Ainv + Ainv.T)
        th_f[k], P_f[k] = th, P
        prev_day = days[k]
    th_s, P_s = th_f.copy(), P_f.copy()
    Cs = np.zeros((D, d, d))
    if smoother and D > 1:
        for k in range(D - 2, -1, -1):
            C = P_f[k] @ np.linalg.inv(P_p[k + 1])
            Cs[k] = C
            th_s[k] = th_f[k] + C @ (th_s[k + 1] - th_p[k + 1])
            P_s[k] = P_f[k] + C @ (P_s[k + 1] - P_p[k + 1]) @ C.T
    return th_f, P_f, th_s, P_s, float(ll), Cs


def kalman_fit(R: pd.DataFrame, cfg: Optional[KalmanConfig] = None, column: str = "",
               calib_k: float = 1.0) -> Optional[KalmanTrack]:
    """Run the filter on a regression frame produced by ``pv_sps.regression_frame``."""
    cfg = cfg or KalmanConfig()
    if len(R) < 10:
        return None
    Phi = R[PARAM_NAMES].to_numpy(float)
    y = R["y"].to_numpy(float)
    ok = np.isfinite(Phi).all(1) & np.isfinite(y)
    R = R[ok]; Phi = Phi[ok]; y = y[ok]
    day_idx = R.index.normalize()
    day_id = day_idx.asi8
    cnt = pd.Series(1, index=day_idx).groupby(level=0).size()
    cnt = cnt[cnt >= cfg.min_obs_day]
    keep = np.isin(day_id, cnt.index.asi8)
    Phi, y, day_id = Phi[keep], y[keep], day_id[keep]
    days = pd.DatetimeIndex(cnt.index)
    if len(days) < 2:
        return None
    r, deff, th_ols = _robust_r_and_deff(Phi, y, day_id, cfg.deff_max)
    r_eff = r * deff
    Q0 = np.diag(np.asarray(cfg.q_daily, float) ** 2)
    P0 = np.diag(np.asarray(cfg.p0_sd, float) ** 2)
    theta0 = np.asarray(cfg.theta0, float)
    scales = list(cfg.q_grid) if cfg.tune_q else [1.0]
    best = None
    for s in scales:
        out = _filter(Phi, y, day_id, days, r_eff, Q0 * s, theta0, P0, smoother=cfg.smoother)
        if best is None or out[4] > best[1][4]:
            best = (s, out)
    s, (th_f, P_f, th_s, P_s, ll, Cs) = best
    return KalmanTrack(column, days, cnt.to_numpy(int), th_f, P_f, th_s, P_s, r_eff, deff, s, ll,
                       calib_k, th_ols, Cs if cfg.smoother else None)


def kalman_track(df: pd.DataFrame, phys: Dict[str, np.ndarray], pv_cols: Sequence[str],
                 rated_kw: Dict[str, float], valid: pd.DataFrame, t_cell: np.ndarray,
                 cloud: np.ndarray, cfg: Optional[KalmanConfig] = None,
                 subsample: int = 2) -> Dict[str, KalmanTrack]:
    """Daily b-trajectories for every plant (used by periods / split / comparison)."""
    cfg = cfg or KalmanConfig()
    tracks = {}
    for j, c in enumerate(pv_cols):
        k = reference_calibration(df, phys, c, j, valid[c])
        R = regression_frame(df, phys, c, j, rated_kw.get(c) or 1.0, valid[c], t_cell, cloud, calib=k)
        R = R.iloc[::subsample]
        tr = kalman_fit(R, cfg, column=c, calib_k=k)
        if tr is not None:
            tracks[c] = tr
    return tracks


def tracks_frame(tracks: Dict[str, KalmanTrack], smoother: bool = True) -> pd.DataFrame:
    """Long table of daily states (plant x day) — handy for plotting and event matching."""
    parts = []
    for c, tr in tracks.items():
        f = tr.frame(smoother).reset_index()
        f.insert(0, "plant", short(c)); f.insert(0, "column", c)
        parts.append(f)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ----------------------------------------------------------------------------- SPS-compatible API
def _summ(tr: "KalmanTrack", mask: np.ndarray, z: float, smoother: bool = True):
    """Mean state over the selected days and the interval for the mean b1.

    With the smoother the sd of the period mean is exact (lag covariances); with the
    filter only, the mean of daily sd is used (conservative)."""
    th_all = tr.theta_s if smoother else tr.theta_f
    th = th_all[mask].mean(0)
    m, sd = tr.mean_b1(mask, smoother)
    return th, m - z * sd, m + z * sd, sd


def kalman_periods(df, phys, pv_cols, rated_kw, valid, t_cell, cloud, freq: str = "QS",
                   cfg: Optional[KalmanConfig] = None, min_n: int = 150, subsample: int = 2,
                   tracks: Optional[Dict[str, KalmanTrack]] = None, **_) -> pd.DataFrame:
    """Per plant x period b1 interval — same columns as ``pv_sps.sps_periods``."""
    cfg = cfg or KalmanConfig()
    tracks = tracks if tracks is not None else kalman_track(df, phys, pv_cols, rated_kw, valid, t_cell,
                                                            cloud, cfg, subsample)
    z = cfg.z
    periods = pd.date_range(df.index[0].to_period("Q").to_timestamp(), df.index[-1], freq=freq)
    rows = []
    for c in pv_cols:
        tr = tracks.get(c)
        if tr is None:
            continue
        for p0 in periods:
            p1 = p0 + pd.tseries.frequencies.to_offset(freq)
            mask = np.asarray((tr.days >= p0) & (tr.days < p1))
            if tr.n_obs[mask].sum() < min_n:
                continue
            th, lo, hi, sd = _summ(tr, mask, z, cfg.smoother)
            rows.append(dict(column=c, plant=short(c), period=p0, n=int(tr.n_obs[mask].sum()), n_days=int(mask.sum()),
                             calib_k=tr.calib_k, **{f"theta_{nm}": v for nm, v in zip(PARAM_NAMES, th)},
                             b1_lo=lo, b1_hi=hi, b1_width=hi - lo,
                             b1_eq_1_in_region=bool(lo <= 1.0 <= hi), confidence=cfg.confidence,
                             b1_sd=sd, kf_q_scale=tr.q_scale, kf_deff=tr.deff))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["column", "period"]).reset_index(drop=True)
        prev_hi = out.groupby("column")["b1_hi"].shift()
        prev_lo = out.groupby("column")["b1_lo"].shift()
        out["disjoint_from_previous"] = (out["b1_lo"] > prev_hi) | (out["b1_hi"] < prev_lo)
        out["shift_direction"] = np.where(out["b1_hi"] < prev_lo, "down",
                                          np.where(out["b1_lo"] > prev_hi, "up", ""))
    return out


def kalman_split(df, phys, pv_cols, rated_kw, valid, t_cell, cloud, boundary,
                 cfg: Optional[KalmanConfig] = None, subsample: int = 2, min_n: int = 150,
                 tracks: Optional[Dict[str, KalmanTrack]] = None, **_) -> pd.DataFrame:
    """b1 on the train period vs the test period — same columns as ``pv_sps.sps_split``.

    The smoothed daily trajectory is averaged over each part; the interval is for this
    mean (exact smoothed covariance, see ``KalmanTrack.mean_b1``)."""
    cfg = cfg or KalmanConfig()
    tracks = tracks if tracks is not None else kalman_track(df, phys, pv_cols, rated_kw, valid, t_cell,
                                                            cloud, cfg, subsample)
    boundary = pd.Timestamp(boundary)
    z = cfg.z
    rows = []
    for c in pv_cols:
        tr = tracks.get(c)
        r = dict(column=c, plant=short(c), calib_k=tr.calib_k if tr else np.nan)
        for nm in ("train", "test"):
            if tr is None:
                mask = np.zeros(0, bool); n = 0
            else:
                mask = np.asarray(tr.days < boundary) if nm == "train" else np.asarray(tr.days >= boundary)
                n = int(tr.n_obs[mask].sum())
            if n < min_n:
                r.update({f"{nm}_b1": np.nan, f"{nm}_b1_lo": np.nan, f"{nm}_b1_hi": np.nan,
                          f"{nm}_n": n, f"{nm}_b1_eq_1": np.nan})
                continue
            th, lo, hi, _sd = _summ(tr, mask, z, cfg.smoother)
            r.update({f"{nm}_b1": th[1], f"{nm}_b1_lo": lo, f"{nm}_b1_hi": hi,
                      f"{nm}_n": n, f"{nm}_b1_eq_1": bool(lo <= 1.0 <= hi)})
        if np.isfinite(r.get("train_b1", np.nan)) and np.isfinite(r.get("test_b1", np.nan)):
            r["regions_disjoint"] = bool(r["test_b1_hi"] < r["train_b1_lo"] or r["test_b1_lo"] > r["train_b1_hi"])
            r["b1_change_pct"] = (r["test_b1"] / r["train_b1"] - 1) * 100
        else:
            r["regions_disjoint"] = np.nan
            r["b1_change_pct"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("column")


def kalman_cluster(df, phys, pv_cols, rated_kw, valid, t_cell, cloud, labels: pd.Series, train_end,
                   cfg: Optional[KalmanConfig] = None, subsample: int = 4, **_) -> pd.DataFrame:
    """Pooled (cluster-level) state-space regression on the training period.

    Observations of all member plants on the same day update the common state together;
    summary = mean smoothed state over the training period (same columns as sps_cluster)."""
    cfg = cfg or KalmanConfig()
    rows = []
    train_end = pd.Timestamp(train_end)
    z = cfg.z
    for cl, members in labels.groupby(labels):
        parts = []
        for c in members.index:
            j = list(pv_cols).index(c)
            k = reference_calibration(df, phys, c, j, valid[c])
            R = regression_frame(df, phys, c, j, rated_kw.get(c) or 1.0, valid[c], t_cell, cloud, calib=k)
            parts.append(R[R.index < train_end].iloc[::subsample])
        R = pd.concat(parts).sort_index() if parts else pd.DataFrame()
        if len(R) < 200:
            continue
        tr = kalman_fit(R, cfg, column=f"cluster_{cl}")
        if tr is None:
            continue
        th, lo, hi, _sd = _summ(tr, np.ones(len(tr.days), bool), z, cfg.smoother)
        rows.append(dict(cluster=cl, n_plants=len(members), n=int(tr.n_obs.sum()),
                         **{f"theta_{nm}": v for nm, v in zip(PARAM_NAMES, th)},
                         b1_lo=lo, b1_hi=hi, b1_eq_1_in_region=bool(lo <= 1.0 <= hi)))
    return pd.DataFrame(rows)


def fuse_evidence(stgnn_rank: pd.Series, exceed_share: pd.Series, split: pd.DataFrame,
                  periods: pd.DataFrame, exceed_thr: float = 0.02, min_change_pct: float = 5.0,
                  prefix: str = "sps") -> pd.DataFrame:
    """Same fusion rule as ``pv_sps.fuse_evidence`` applied to Kalman intervals.

    By default the column names are kept identical to SPS (``sps_*``) so that the report
    and downstream code work without changes; pass ``prefix="kf"`` to rename them.
    The evidence labels state the estimator explicitly ("Kalman")."""
    from pv_sps import fuse_evidence as _fuse
    out = _fuse(stgnn_rank, exceed_share, split, periods, exceed_thr=exceed_thr, min_change_pct=min_change_pct)
    out["evidence"] = out["evidence"].str.replace("SPS", ESTIMATOR_LABEL, regex=False)
    if prefix != "sps":
        out = out.rename(columns={c: c.replace("sps_", f"{prefix}_", 1) for c in out.columns if c.startswith("sps_")})
    return out


def kalman_all(df, phys, pv_cols, rated_kw, valid, t_cell, cloud, boundary, tech_labels: pd.Series,
               cfg: Optional[KalmanConfig] = None, subsample: int = 2, freq: str = "QS") -> Dict[str, object]:
    """Convenience: one pass of the filter -> periods, split, cluster and daily tracks."""
    cfg = cfg or KalmanConfig()
    tracks = kalman_track(df, phys, pv_cols, rated_kw, valid, t_cell, cloud, cfg, subsample)
    per = kalman_periods(df, phys, pv_cols, rated_kw, valid, t_cell, cloud, freq=freq, cfg=cfg, tracks=tracks)
    split = kalman_split(df, phys, pv_cols, rated_kw, valid, t_cell, cloud, boundary, cfg=cfg, tracks=tracks)
    clus = kalman_cluster(df, phys, pv_cols, rated_kw, valid, t_cell, cloud, tech_labels, boundary, cfg=cfg)
    return dict(periods=per, split=split, cluster=clus, tracks=tracks, daily=tracks_frame(tracks, cfg.smoother))


# aliases so that ``import pv_kalman as est_mod`` can replace ``import pv_sps as est_mod``
sps_periods = kalman_periods
sps_split = kalman_split
sps_cluster = kalman_cluster
