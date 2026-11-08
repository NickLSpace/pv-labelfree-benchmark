"""Section 16: modality "uncertainty of P_exp" = predictive variance of the Kalman filter.

Uses the state-space model of ``pv_kalman`` (same regression as SPS / Kalman in sections 1-9:
``y = b0 + b1 * P_exp,cal + b2 * (T_cell - 25) / 10 + b3 * cloud``, daily random walk of the
coefficients), but strictly CAUSALLY, so that the channel can be an input of a detector:

* the observation variance ``r`` and the within-day design effect are estimated on the TRAINING
  period only (robust OLS residual, as in ``pv_kalman``); ``Q`` = ``KalmanConfig.q_daily``;
* the filter runs over every calendar day; for day d the PREDICTED state ``theta_d|d-1`` and
  covariance ``P_d|d-1 = P_d-1|d-1 + Q`` (before the observations of day d) are used, i.e. the
  channel of an hour never depends on that hour or later hours;
* hourly predictive variance of the normalised power
  ``var_par = phi_t' P_d|d-1 phi_t``   (uncertainty of the calibrated P_exp model)
  ``var_tot = var_par + r_eff``        (plus observation noise).

Channels (per unit of rated power): ``k_sd_par``, ``k_sd_tot``, ``k_rel`` = ``k_sd_tot`` /
max(expected power, 0.05) and ``has``.  The covariance of a linear Kalman filter depends only on
the regressors and on which days have data, not on the measured values; injected faults therefore
change the channel only through the data gaps and, for the pyranometer drift, through P_exp.
This is the property that makes it an uncertainty modality and not another detector.

This is independent of the SPS / Kalman switch of sections 1-9: the estimator comparison is not
changed and the channel is computed even when SPS is the selected estimator.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

import pv_kalman as kf
from pv_sps import PARAM_NAMES, reference_calibration, regression_frame


def _log(*a):
    import time
    print(time.strftime("%H:%M:%S"), "[kalman-unc]", *a, flush=True)


@dataclass
class KalmanUncertainty:
    params: Dict[str, dict]                     # column -> {k, r_eff, deff, theta_ols}
    cfg: kf.KalmanConfig = field(default_factory=kf.KalmanConfig)
    boundary: str = "2023-01-01"

    def signature(self) -> str:
        h = hashlib.sha256()
        for c in sorted(self.params):
            p = self.params[c]; h.update(c.encode()); h.update(np.array([p["k"], p["r_eff"]], float).tobytes())
        return h.hexdigest()[:12]

    # ------------------------------------------------------------------ channels
    def channels(self, D: dict, C: dict, pv_cols: Sequence[str], df: Optional[pd.DataFrame] = None,
                 phys: Optional[dict] = None) -> Dict[str, np.ndarray]:
        df = D["df"] if df is None else df
        phys = D["phys"] if phys is None else phys
        idx = df.index; T, n = len(idx), len(pv_cols)
        valid = _valid(D, C, df, pv_cols)
        cloud = (1.0 - _kt_series(D, df).clip(0, 1.2)).fillna(1.0).to_numpy()
        out = {k: np.full((T, n), np.nan) for k in ("k_sd_par", "k_sd_tot", "k_rel", "k_pred")}
        Q = np.diag(np.asarray(self.cfg.q_daily, float) ** 2)
        P0 = np.diag(np.asarray(self.cfg.p0_sd, float) ** 2)
        th0 = np.asarray(self.cfg.theta0, float)
        days_all = idx.normalize()
        ud, day_of = np.unique(days_all.asi8, return_inverse=True)
        for j, c in enumerate(pv_cols):
            p = self.params.get(c)
            if p is None:
                continue
            kw = max(D["rated"].get(c) or 1.0, 1e-3) * 1000.0
            Phi = np.column_stack([np.ones(T), p["k"] * np.asarray(phys["p_expected"], float)[:, j] / kw,
                                   ((D["t_cell"][:, j] if np.ndim(D["t_cell"]) == 2 else D["t_cell"]) - 25.0) / 10.0, cloud])
            y = df[c].to_numpy(float) / kw
            ok = valid[:, j] & np.isfinite(Phi).all(1) & np.isfinite(y)
            r = p["r_eff"]
            th, P = th0.copy(), P0.copy()
            started = False
            Ppred = np.full((len(ud), 4, 4), np.nan); thpred = np.full((len(ud), 4), np.nan)
            order = np.argsort(day_of, kind="stable"); bounds = np.searchsorted(day_of[order], np.arange(len(ud) + 1))
            for d in range(len(ud)):
                rows = order[bounds[d]:bounds[d + 1]]
                if started:
                    P = P + Q
                Ppred[d], thpred[d] = P, th
                rr = rows[ok[rows]]
                if len(rr):
                    H, z = Phi[rr], y[rr]
                    Pinv = np.linalg.inv(P)
                    A = np.linalg.inv(Pinv + H.T @ H / r)
                    th = A @ (Pinv @ th + H.T @ z / r); P = 0.5 * (A + A.T)
                    started = True
            if not started:
                continue
            Pt = Ppred[day_of]; tht = thpred[day_of]
            var_par = np.einsum("ti,tij,tj->t", Phi, Pt, Phi)
            fin = np.isfinite(Phi).all(1)
            sd_par = np.where(fin, np.sqrt(np.clip(var_par, 0, None)), np.nan)
            sd_tot = np.where(fin, np.sqrt(np.clip(var_par, 0, None) + r), np.nan)
            pred = np.where(fin, np.einsum("ti,ti->t", Phi, tht), np.nan)
            out["k_sd_par"][:, j] = sd_par; out["k_sd_tot"][:, j] = sd_tot; out["k_pred"][:, j] = pred
            out["k_rel"][:, j] = sd_tot / np.maximum(np.abs(np.nan_to_num(pred)), 0.05)
        a = np.asarray(phys["p_expected"], float)[:, :n] / np.array([max(D["rated"].get(c) or 1.0, 1e-3) * 1000 for c in pv_cols])
        day = a > 0.02
        out["has"] = (np.isfinite(out["k_sd_tot"]) & day).astype(float)
        return out


def _kt_series(D: dict, df: pd.DataFrame) -> pd.Series:
    import pv_shading as sh
    return pd.Series(sh._kt(D, df), index=df.index)


def _valid(D: dict, C: dict, df: pd.DataFrame, pv: Sequence[str]) -> np.ndarray:
    v = C["valid"].reindex(index=df.index, columns=pv, fill_value=False).to_numpy(bool).copy()
    return v & np.isfinite(df[pv].to_numpy(float))


def fit_uncertainty(D: dict, C: dict, cfg: Optional[kf.KalmanConfig] = None, boundary: str = "2023-01-01",
                    subsample: int = 2, verbose: bool = True) -> KalmanUncertainty:
    """r_eff and the calibration factor per plant from the TRAINING period only."""
    cfg = cfg or kf.KalmanConfig()
    df, phys, pv = D["df"], D["phys"], list(D["pv_cols"])
    b = pd.Timestamp(boundary)
    cloud = (1.0 - _kt_series(D, df).clip(0, 1.2)).fillna(1.0).to_numpy()
    tr = pd.Series(np.asarray(df.index < b), index=df.index)
    params = {}
    for j, c in enumerate(pv):
        v = C["valid"][c].reindex(df.index, fill_value=False) & tr
        if v.sum() < 50:
            continue
        k = reference_calibration(df, phys, c, j, v)
        R = regression_frame(df, phys, c, j, D["rated"].get(c) or 1.0, v, D["t_cell"], cloud, calib=k).iloc[::subsample]
        Phi = R[PARAM_NAMES].to_numpy(float); y = R["y"].to_numpy(float)
        ok = np.isfinite(Phi).all(1) & np.isfinite(y)
        if ok.sum() < 50:
            continue
        r, deff, th = kf._robust_r_and_deff(Phi[ok], y[ok], R.index[ok].normalize().asi8, cfg.deff_max)
        params[c] = dict(k=float(k), r_eff=float(r * deff), r=float(r), deff=float(deff), theta_ols=th)
    if verbose:
        _log(f"observation variance fitted on the training period for {len(params)} / {len(pv)} plants")
    return KalmanUncertainty(params, cfg, boundary)


def evaluate(D: dict, C: dict, K: KalmanUncertainty, ch: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, pd.DataFrame]:
    """Calibration of the causal predictive interval on daylight valid hours (training / test)."""
    from scipy.stats import spearmanr
    pv = list(D["pv_cols"]); df = D["df"]; idx = df.index; b = pd.Timestamp(K.boundary)
    ch = ch or K.channels(D, C, pv)
    val = _valid(D, C, df, pv)
    rows = []
    for period, sel in (("training", np.asarray(idx < b)), ("test", np.asarray(idx >= b))):
        cov, rho, rel, n = [], [], [], 0
        for j, c in enumerate(pv):
            kw = max(D["rated"].get(c) or 1.0, 1e-3) * 1000.0
            y = df[c].to_numpy(float) / kw
            m = sel & val[:, j] & (ch["has"][:, j] > 0) & np.isfinite(ch["k_pred"][:, j])
            if m.sum() < 50:
                continue
            e = np.abs(y[m] - ch["k_pred"][m, j]); s = ch["k_sd_tot"][m, j]
            cov.append(float(np.mean(e <= 1.96 * s))); rho.append(float(spearmanr(s, e).statistic)); rel.append(float(np.median(ch["k_rel"][m, j])))
            n += 1
        rows.append({"Period": period, "Plants": n, "Coverage of the 95 % interval (median)": np.median(cov) if cov else np.nan,
                     "Spearman(sd, |error|) (median)": np.median(rho) if rho else np.nan,
                     "Relative sd (median)": np.median(rel) if rel else np.nan})
    return {"calibration": pd.DataFrame(rows)}


TITLES = {"calibration": "Section 16: causal Kalman predictive interval of the normalised power (daylight, valid hours)"}
