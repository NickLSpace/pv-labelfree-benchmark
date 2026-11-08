"""
Sign-Perturbed Sums (SPS) confidence regions for an interpretable per-plant power model.

SPS is NOT applied to the neural network.  It is applied to the low-dimensional
physical regression of one installation (or one cluster):

    P_meas_{i,t} = b0_i + b1_i * P_exp_{i,t} + b2_i * T_cell_t + b3_i * w_t + eps_{i,t}

(all powers normalised by the rated power; T_cell in (T-25)/10 units; w_t = cloudiness
1 - k_t).  b1 is the key coefficient:
    b1 ~ 1                    measurement agrees with the physical model
    sustained decrease of b1  possible degradation, soiling, shading or curtailment
    widening region           lack of data or unstable regime
    disjoint consecutive regions  possible regime change

SPS (Csaji, Campi & Weyer 2015) builds an EXACT finite-sample confidence region
under the assumptions that the noise is independent and symmetric about zero,
without knowing its distribution:

    S_0(th) = R^{-1/2} (1/n) sum_t        phi_t (y_t - phi_t' th)
    S_j(th) = R^{-1/2} (1/n) sum_t a_jt   phi_t (y_t - phi_t' th),   a_jt in {-1,+1}
    th in region  <=>  rank of ||S_0||^2 among {||S_j||^2}_{j=0..m-1}  <= m - q
    P(th* in region) = 1 - q/m  exactly.

Hourly PV residuals are autocorrelated within a day.  We therefore use BLOCK sign
perturbations (one sign per day), which keeps the exactness argument valid when
day-blocks are independent and symmetric (Kolumbán 2016 discusses block/dependent
extensions); this is stated as a modelling assumption.

Projection of the region onto b1 is obtained by profiling: for each candidate b1
the remaining parameters minimise ||S_0||^2 (closed form); the candidate is kept if
this profile point is in the region.  This gives the SPS interval for b1 used in
tables and plots.

References
----------
B. Cs. Csáji, M. C. Campi, E. Weyer (2015). "Sign-Perturbed Sums: A New System
  Identification Approach for Constructing Exact Non-Asymptotic Confidence Regions in
  Linear Regression Models." IEEE Trans. Signal Processing 63(1):169-181.
  https://doi.org/10.1109/TSP.2014.2369000
S. Kolumbán (2016). "System Identification in Highly Non-Informative Environment."
  PhD thesis, VU Amsterdam / BME (SPS extensions).
M. C. Campi, E. Weyer (2005). "Guaranteed non-asymptotic confidence regions in system
  identification." Automatica 41(10):1751-1764.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from pv_cohorts import short

PARAM_NAMES = ["b0", "b1_Pexp", "b2_Tcell", "b3_cloud"]


@dataclass
class SPSRegion:
    theta_ls: np.ndarray
    b1_lo: float
    b1_hi: float
    contains_b1_eq_1: bool
    n: int
    n_blocks: int
    confidence: float
    grid: np.ndarray
    accepted: np.ndarray


class SPS:
    """Exact SPS test for linear regression with (optionally) block sign perturbations."""

    def __init__(self, m: int = 100, q: int = 5, seed: int = 0):
        assert 0 < q < m
        self.m, self.q, self.seed = m, q, seed

    def fit(self, Phi: np.ndarray, y: np.ndarray, blocks: Optional[np.ndarray] = None):
        Phi = np.asarray(Phi, float)
        y = np.asarray(y, float)
        n, d = Phi.shape
        rng = np.random.default_rng(self.seed)
        if blocks is None:
            blocks = np.arange(n)
        ub, inv = np.unique(blocks, return_inverse=True)
        nb = len(ub)
        alpha = np.ones((self.m, n))
        alpha[1:] = rng.choice([-1.0, 1.0], size=(self.m - 1, nb))[:, inv]
        R = Phi.T @ Phi / n
        w, V = np.linalg.eigh(R)
        Rmh = V @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-12, None))) @ V.T      # R^{-1/2}
        # S_j(th) = Rmh (b_j - A_j th) / n  with  b_j = Phi' (a_j * y), A_j = Phi' diag(a_j) Phi
        self.b = np.einsum("jt,td,t->jd", alpha, Phi, y) / n                      # (m, d)
        self.A = np.einsum("jt,td,te->jde", alpha, Phi, Phi) / n                  # (m, d, d)
        self.Rmh, self.n, self.nb, self.d = Rmh, n, nb, d
        self.tie = rng.permutation(self.m)                                       # random tie-breaking
        self.theta_ls = np.linalg.lstsq(Phi, y, rcond=None)[0]
        return self

    def norms(self, theta: np.ndarray) -> np.ndarray:
        """||S_j(theta)||^2 for all j; theta (..., d) -> (..., m)."""
        th = np.atleast_2d(theta)
        r = self.b[None] - np.einsum("jde,ke->kjd", self.A, th)                  # (K, m, d)
        s = np.einsum("de,kje->kjd", self.Rmh, r)
        return (s ** 2).sum(-1)

    def accept(self, theta: np.ndarray) -> np.ndarray:
        z = self.norms(theta)                                                    # (K, m)
        z0 = z[:, :1]
        # rank of Z0 among all (strict order, ties broken by random permutation)
        greater = (z0 > z[:, 1:]) | ((z0 == z[:, 1:]) & (self.tie[0] > self.tie[1:]))
        rank = 1 + greater.sum(1)
        return rank <= self.m - self.q

    def profile_b1(self, b1_grid: np.ndarray, idx: int = 1) -> np.ndarray:
        """For each b1 value, parameters minimising ||S_0||^2 with b1 fixed."""
        A0, b0 = self.A[0], self.b[0]
        M = self.Rmh @ A0
        c = self.Rmh @ b0
        others = [k for k in range(self.d) if k != idx]
        Mo, Mi = M[:, others], M[:, idx]
        th = np.zeros((len(b1_grid), self.d))
        pinv = np.linalg.pinv(Mo)
        for g, b1 in enumerate(b1_grid):
            th_o = pinv @ (c - Mi * b1)
            th[g, others] = th_o
            th[g, idx] = b1
        return th

    def region_b1(self, width: float = 1.0, n_grid: int = 401, idx: int = 1) -> SPSRegion:
        c = self.theta_ls[idx]
        grid = np.linspace(c - width, c + width, n_grid)
        acc = self.accept(self.profile_b1(grid, idx))
        # widen once if the region touches the grid border
        if acc.any() and (acc[0] or acc[-1]):
            grid = np.linspace(c - 4 * width, c + 4 * width, n_grid)
            acc = self.accept(self.profile_b1(grid, idx))
        if acc.any():
            lo, hi = grid[acc].min(), grid[acc].max()
        else:                                   # region smaller than grid step
            lo = hi = c
        one = bool(self.accept(self.profile_b1(np.array([1.0]), idx))[0])
        return SPSRegion(self.theta_ls, float(lo), float(hi), one, self.n, self.nb,
                         1 - self.q / self.m, grid, acc)


# ----------------------------------------------------------------------------- data assembly
def regression_frame(df: pd.DataFrame, phys: Dict[str, np.ndarray], col: str, j: int,
                     rated_kw: float, valid: pd.Series, t_cell: np.ndarray, cloud: np.ndarray,
                     calib: float = 1.0) -> pd.DataFrame:
    kw = max(rated_kw, 1e-3) * 1000.0
    m = valid.to_numpy(bool)
    out = pd.DataFrame({
        "y": df[col].to_numpy(float)[m] / kw,
        "b0": 1.0,
        "b1_Pexp": calib * phys["p_expected"][m, j] / kw,
        "b2_Tcell": ((t_cell[:, j] if np.ndim(t_cell) == 2 else t_cell)[m] - 25.0) / 10.0,
        "b3_cloud": cloud[m],
    }, index=df.index[m])
    return out


def reference_calibration(df, phys, col, j, valid: pd.Series, days: int = 90) -> float:
    """k_i = median(P_meas / P_exp) over the first ``days`` of valid operation.

    Rated power / transposition metadata errors make the absolute ratio differ from 1
    for many HKUST plants; calibrating on the plant's own commissioning period makes
    b1 = 1 mean 'as at commissioning', so a sustained drop of b1 is interpretable as
    loss of performance."""
    v = valid.to_numpy(bool)
    if not v.any():
        return 1.0
    t0 = df.index[v].min()
    m = v & (df.index < t0 + pd.Timedelta(days=days))
    r = df[col].to_numpy(float)[m] / np.maximum(phys["p_expected"][m, j], 1e-6)
    r = r[np.isfinite(r) & (r > 0)]
    return float(np.median(r)) if len(r) else 1.0   # P_exp,cal = k * P_exp


def sps_periods(df: pd.DataFrame, phys: Dict[str, np.ndarray], pv_cols: Sequence[str],
                rated_kw: Dict[str, float], valid: pd.DataFrame, t_cell: np.ndarray,
                cloud: np.ndarray, freq: str = "QS", m: int = 100, q: int = 5,
                min_n: int = 150, subsample: int = 2, seed: int = 0) -> pd.DataFrame:
    """Per plant x period SPS regions of b1 (block signs = days)."""
    rows = []
    periods = pd.date_range(df.index[0].to_period("Q").to_timestamp(), df.index[-1], freq=freq)
    for j, c in enumerate(pv_cols):
        k = reference_calibration(df, phys, c, j, valid[c])
        R = regression_frame(df, phys, c, j, rated_kw.get(c) or 1.0, valid[c], t_cell, cloud, calib=k)
        R = R.iloc[::subsample]
        for p0 in periods:
            p1 = p0 + pd.tseries.frequencies.to_offset(freq)
            Rp = R[(R.index >= p0) & (R.index < p1)]
            if len(Rp) < min_n:
                continue
            sps = SPS(m, q, seed).fit(Rp[PARAM_NAMES].to_numpy(), Rp["y"].to_numpy(),
                                      blocks=Rp.index.normalize().asi8)
            reg = sps.region_b1(width=0.5)
            rows.append(dict(column=c, plant=short(c), period=p0, n=reg.n, n_days=reg.n_blocks,
                             calib_k=k, **{f"theta_{nm}": v for nm, v in zip(PARAM_NAMES, reg.theta_ls)},
                             b1_lo=reg.b1_lo, b1_hi=reg.b1_hi, b1_width=reg.b1_hi - reg.b1_lo,
                             b1_eq_1_in_region=reg.contains_b1_eq_1, confidence=reg.confidence))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["column", "period"]).reset_index(drop=True)
        prev_hi = out.groupby("column")["b1_hi"].shift()
        prev_lo = out.groupby("column")["b1_lo"].shift()
        out["disjoint_from_previous"] = (out["b1_lo"] > prev_hi) | (out["b1_hi"] < prev_lo)
        out["shift_direction"] = np.where(out["b1_hi"] < prev_lo, "down",
                                          np.where(out["b1_lo"] > prev_hi, "up", ""))
    return out


def sps_split(df, phys, pv_cols, rated_kw, valid, t_cell, cloud, boundary,
              m=100, q=5, subsample=2, seed=0, min_n=150) -> pd.DataFrame:
    """b1 region on the train period vs the test period for every plant."""
    boundary = pd.Timestamp(boundary)
    rows = []
    for j, c in enumerate(pv_cols):
        k = reference_calibration(df, phys, c, j, valid[c])
        R = regression_frame(df, phys, c, j, rated_kw.get(c) or 1.0, valid[c], t_cell, cloud, calib=k)
        R = R.iloc[::subsample]
        r = dict(column=c, plant=short(c), calib_k=k)
        for nm, part in (("train", R[R.index < boundary]), ("test", R[R.index >= boundary])):
            if len(part) < min_n:
                r.update({f"{nm}_b1": np.nan, f"{nm}_b1_lo": np.nan, f"{nm}_b1_hi": np.nan,
                          f"{nm}_n": len(part), f"{nm}_b1_eq_1": np.nan})
                continue
            sps = SPS(m, q, seed).fit(part[PARAM_NAMES].to_numpy(), part["y"].to_numpy(),
                                      blocks=part.index.normalize().asi8)
            reg = sps.region_b1(width=0.5)
            r.update({f"{nm}_b1": reg.theta_ls[1], f"{nm}_b1_lo": reg.b1_lo, f"{nm}_b1_hi": reg.b1_hi,
                      f"{nm}_n": reg.n, f"{nm}_b1_eq_1": reg.contains_b1_eq_1})
        if np.isfinite(r.get("train_b1", np.nan)) and np.isfinite(r.get("test_b1", np.nan)):
            r["regions_disjoint"] = bool(r["test_b1_hi"] < r["train_b1_lo"] or r["test_b1_lo"] > r["train_b1_hi"])
            r["b1_change_pct"] = (r["test_b1"] / r["train_b1"] - 1) * 100
        else:
            r["regions_disjoint"] = np.nan
            r["b1_change_pct"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("column")


def sps_cluster(df, phys, pv_cols, rated_kw, valid, t_cell, cloud, labels: pd.Series, train_end,
                m=100, q=5, subsample=4, seed=0) -> pd.DataFrame:
    """Pooled (cluster-level) regression on the training period."""
    rows = []
    train_end = pd.Timestamp(train_end)
    for cl, members in labels.groupby(labels):
        parts = []
        for c in members.index:
            j = list(pv_cols).index(c)
            k = reference_calibration(df, phys, c, j, valid[c])
            R = regression_frame(df, phys, c, j, rated_kw.get(c) or 1.0, valid[c], t_cell, cloud, calib=k)
            R = R[R.index < train_end].iloc[::subsample]
            R["block"] = R.index.normalize().asi8 + hash(c) % 997      # plant-day blocks
            parts.append(R)
        R = pd.concat(parts) if parts else pd.DataFrame()
        if len(R) < 200:
            continue
        sps = SPS(m, q, seed).fit(R[PARAM_NAMES].to_numpy(), R["y"].to_numpy(), blocks=R["block"].to_numpy())
        reg = sps.region_b1(width=0.3)
        rows.append(dict(cluster=cl, n_plants=len(members), n=reg.n,
                         **{f"theta_{nm}": v for nm, v in zip(PARAM_NAMES, reg.theta_ls)},
                         b1_lo=reg.b1_lo, b1_hi=reg.b1_hi, b1_eq_1_in_region=reg.contains_b1_eq_1))
    return pd.DataFrame(rows)


def fuse_evidence(stgnn_rank: pd.Series, exceed_share: pd.Series, split: pd.DataFrame,
                  periods: pd.DataFrame, exceed_thr: float = 0.02, min_change_pct: float = 5.0) -> pd.DataFrame:
    """STGNN score + SPS statistical evidence -> one evidence level per plant.

    With thousands of hourly points SPS regions become very narrow, so statistical
    non-overlap is combined with a practical-significance threshold |Δβ1| >= min_change_pct."""
    out = pd.DataFrame(index=split["plant"].values)
    out["stgnn_mean_score"] = stgnn_rank.reindex(out.index)
    out["stgnn_exceed_share"] = exceed_share.reindex(out.index)
    s = split.set_index("plant")
    out["sps_train_b1"] = s["train_b1"]
    out["sps_test_b1"] = s["test_b1"]
    out["sps_test_b1_lo"] = s["test_b1_lo"]
    out["sps_test_b1_hi"] = s["test_b1_hi"]
    out["sps_b1_change_pct"] = s["b1_change_pct"]
    out["sps_regions_disjoint"] = s["regions_disjoint"]
    out["sps_test_excludes_1"] = s["test_b1_eq_1"].map(lambda v: (not v) if isinstance(v, (bool, np.bool_)) else np.nan)
    if len(periods):
        g = periods.groupby("plant")
        out["sps_n_regime_changes"] = g["disjoint_from_previous"].sum().reindex(out.index)
        out["sps_mean_b1_width"] = g["b1_width"].mean().reindex(out.index)
    model = out["stgnn_exceed_share"].fillna(0) > exceed_thr
    material = out["sps_b1_change_pct"].abs() >= min_change_pct
    stat_down = (out["sps_regions_disjoint"] == True) & (out["sps_b1_change_pct"] < 0) & material  # noqa: E712
    stat_any = (out["sps_regions_disjoint"] == True) & material  # noqa: E712
    out["evidence"] = np.select(
        [model & stat_down, stat_down, model & stat_any, model, stat_any],
        ["strong: STGNN + SPS (β1 decrease)", "statistical: β1 decrease (SPS)",
         "STGNN + SPS (β1 change)", "model-based: STGNN only", "statistical: β1 change (SPS)"],
        default="no")
    return out
