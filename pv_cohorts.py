"""
Data coverage, installation cohorts and cohort-aware degradation metrics.

Why cohorts
-----------
A single common analysis period for the whole fleet is not meaningful: plants were
commissioned in 2021, 2022 and 2023.  Installations are therefore split into
cohorts by the year from which they have reliable data:

    cohort 1 ("old")   reliable data in 2021-2023  -> three-year trend (%/yr)
    cohort 2 ("2022")  first reliable year 2022    -> Delta PR 2023/2022 (NOT long-term degradation)
    cohort 3 ("2023")  data only in 2023           -> state vs physics model / peers, no degradation

Coverage definition (per plant, per hour)
-----------------------------------------
An hour counts as a valid measurement if
    GHI > 200 W/m2,  P_expected > 0 and finite,  P_measured finite,
    the value was not artificially filled (see pv_physics_graph.infer_filled_mask),
    the plant is commissioned (exploitation mask).
A month is covered if valid / expected daylight hours >= 0.8 (expected = hours with
GHI > 200 and positive P_expected in that month).

The sharp synchronous 2023 drop of the old cohort is tested as a possible systemic
change (sensor / data pipeline / irradiance) BEFORE interpreting it as degradation:
synchronicity of plant-level change points, fleet-normalised PR and irradiance
statistics are reported.

References: Jordan & Kurtz (2013) Prog. Photovolt. 21:12-29 (degradation rates);
Jordan et al. (2018) IEEE J. Photovolt. 8:525-531 (RdTools year-on-year method);
IEC 61724-1:2021 (performance ratio, data filtering).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Cohort lists given in the task specification (used for cross-checking only;
# the operative assignment is data-driven from the coverage criterion).
REFERENCE_COHORTS: Dict[int, List[str]] = {
    1: ["LSK South", "LSK North", "UG Hall2 RF", "UG Hall2 2F", "UG Hall3", "UG Hall4",
        "UG Hall6", "UG Hall7", "Zone A1", "Zone A4", "CYT (Station 1)", "CYT (Station 2)",
        "Zone L1 (Station 1)", "Zone L1(Station 2)"] +
       [f"SQ{i}" for i in (1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)] + ["SQ567"],
    2: ["Zone J2", "Zone A5", "Zone A6", "Zone A7", "SQ Block R", "SQ Block S", "Library",
        "Tower A", "Indoor Sports Centre", "SQ567 Flexible PV", "SQ Apartment37-48 Flexible PV",
        "UG Hall4 Flexible PV", "UG Hall7 Flexible PV"],
    3: ["SQ Block Q", "SQ Block Q (Phrase 2)", "UG Hall8", "UG Hall9", "Zone L2"],
}
COHORT_NAMES = {1: "Cohort 1 (2021–2023)", 2: "Cohort 2 (from 2022)", 3: "Cohort 3 (2023 only)",
                0: "Insufficient data"}


def short(c: str) -> str:
    return c.replace("power(W)_", "")


def reference_cohort(name: str) -> Optional[int]:
    for k, v in REFERENCE_COHORTS.items():
        if short(name) in v:
            return k
    return None


# ----------------------------------------------------------------------------- coverage
def coverage_mask(df: pd.DataFrame, phys: Dict[str, np.ndarray], pv_cols: Sequence[str],
                  observed: Optional[pd.DataFrame] = None,
                  activity: Optional[pd.DataFrame] = None,
                  ghi_thr: float = 200.0) -> pd.DataFrame:
    """(T x plant) boolean: hour is a valid measurement for PR / degradation."""
    ghi = phys["ghi"]
    pe = phys["p_expected"]
    P = df[list(pv_cols)].to_numpy(float)
    ok = (ghi[:, None] > ghi_thr) & np.isfinite(pe) & (pe > 0) & np.isfinite(P)
    if observed is not None:
        ok &= observed.reindex(index=df.index, columns=pv_cols, fill_value=False).to_numpy(bool)
    if activity is not None:
        ok &= activity.reindex(index=df.index, columns=pv_cols, fill_value=False).to_numpy(bool)
    return pd.DataFrame(ok, index=df.index, columns=list(pv_cols))


def expected_mask(df: pd.DataFrame, phys: Dict[str, np.ndarray], pv_cols: Sequence[str],
                  ghi_thr: float = 200.0) -> pd.DataFrame:
    ghi = phys["ghi"]
    pe = phys["p_expected"]
    ok = (ghi[:, None] > ghi_thr) & np.isfinite(pe) & (pe > 0)
    return pd.DataFrame(ok, index=df.index, columns=list(pv_cols))


def monthly_coverage(valid: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    v = valid.resample("MS").sum()
    e = expected.resample("MS").sum().replace(0, np.nan)
    return (v / e).clip(upper=1.0)


def monthly_pr(df: pd.DataFrame, phys: Dict[str, np.ndarray], valid: pd.DataFrame) -> pd.DataFrame:
    """Energy-weighted monthly performance index PR = sum(P_meas) / sum(P_expected) over
    valid hours (IEC 61724-1 style; P_expected already contains POA transposition and
    the Faiman temperature correction, i.e. a temperature-corrected PR)."""
    cols = list(valid.columns)
    P = np.where(valid.to_numpy(bool), df[cols].to_numpy(float), 0.0)
    E = np.where(valid.to_numpy(bool), phys["p_expected"], 0.0)
    num = pd.DataFrame(P, index=df.index, columns=cols).resample("MS").sum()
    den = pd.DataFrame(E, index=df.index, columns=cols).resample("MS").sum()
    return num / den.replace(0, np.nan)


# ----------------------------------------------------------------------------- cohorts
def assign_cohorts(cov_m: pd.DataFrame, min_cov: float = 0.8, min_months_year: int = 5,
                   years=(2021, 2022, 2023)) -> pd.DataFrame:
    """Data-driven cohort: first year with >= ``min_months_year`` covered months."""
    covered = cov_m >= min_cov
    rows = []
    for c in cov_m.columns:
        per_year = {y: int(covered.loc[covered.index.year == y, c].sum()) for y in years}
        rel = [y for y in years if per_year[y] >= min_months_year]
        first = rel[0] if rel else None
        if first is None:
            coh = 0
        elif first == years[0] and len(rel) >= 2:
            coh = 1
        elif first == years[0]:
            coh = 1 if per_year[years[-1]] >= min_months_year else 0
        elif first == years[1]:
            coh = 2
        else:
            coh = 3
        ref = reference_cohort(c)
        first_m = covered.index[covered[c].to_numpy()].min() if covered[c].any() else pd.NaT
        rows.append(dict(plant=short(c), column=c, cohort=coh, cohort_name=COHORT_NAMES[coh],
                         reference_cohort=ref, agrees_with_reference=(ref is None or ref == coh),
                         first_covered_month=first_m,
                         **{f"covered_months_{y}": per_year[y] for y in years}))
    return pd.DataFrame(rows).set_index("column")


# ----------------------------------------------------------------------------- degradation
def _bootstrap_median(x: np.ndarray, n: int = 2000, seed: int = 0, ci: float = 0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(x, (n, len(x)), replace=True), axis=1)
    a = (1 - ci) / 2
    return float(np.median(x)), float(np.quantile(b, a)), float(np.quantile(b, 1 - a))


def yoy_degradation(pr_m: pd.Series, covered: pd.Series, years: Sequence[int]):
    """Year-on-year (RdTools-style) rate from month-matched monthly PR pairs, %/yr."""
    s = pr_m.where(covered)
    vals = []
    for y in years[:-1]:
        for m in range(1, 13):
            a = s.get(pd.Timestamp(year=y, month=m, day=1), np.nan)
            b = s.get(pd.Timestamp(year=y + 1, month=m, day=1), np.nan)
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                vals.append((b / a - 1.0) * 100.0)
    med, lo, hi = _bootstrap_median(np.array(vals))
    return med, lo, hi, len(vals)


def seasonal_trend(pr_m: pd.Series, covered: pd.Series):
    """OLS PR ~ month-of-year dummies + t ; slope as % of mean PR per year."""
    s = pr_m.where(covered).dropna()
    if s.index.year.nunique() < 2 or len(s) < 8:
        return np.nan, np.nan
    t = (s.index - s.index[0]).days.to_numpy(float) / 365.25
    M = pd.get_dummies(s.index.month).to_numpy(float)
    X = np.column_stack([M, t])
    beta, *_ = np.linalg.lstsq(X, s.to_numpy(float), rcond=None)
    resid = s.to_numpy(float) - X @ beta
    dof = max(len(s) - X.shape[1], 1)
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid) / dof
    mean = float(s.mean())
    return float(beta[-1] / mean * 100.0), float(np.sqrt(cov[-1, -1]) / mean * 100.0)


def matched_delta(pr_m: pd.Series, covered: pd.Series, y0: int = 2022, y1: int = 2023):
    """Delta PR_{y1/y0} = (PR_y1 / PR_y0 - 1) * 100 over months covered in BOTH years."""
    s = pr_m.where(covered)
    a, b = [], []
    for m in range(1, 13):
        x = s.get(pd.Timestamp(year=y0, month=m, day=1), np.nan)
        y = s.get(pd.Timestamp(year=y1, month=m, day=1), np.nan)
        if np.isfinite(x) and np.isfinite(y):
            a.append(x); b.append(y)
    if not a:
        return np.nan, np.nan, np.nan, 0
    a, b = np.array(a), np.array(b)
    point = (np.mean(b) / np.mean(a) - 1.0) * 100.0
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        k = rng.integers(0, len(a), len(a))
        boots.append((b[k].mean() / a[k].mean() - 1.0) * 100.0)
    return float(point), float(np.quantile(boots, .025)), float(np.quantile(boots, .975)), len(a)


@dataclass
class CohortAnalysis:
    cov_m: pd.DataFrame
    pr_m: pd.DataFrame
    cohorts: pd.DataFrame
    degradation: pd.DataFrame
    systemic: Dict[str, object]
    pr_year: pd.DataFrame
    fleet_norm_pr_m: pd.DataFrame


def fleet_normalised(pr_m: pd.DataFrame, covered: pd.DataFrame, ref_cols: Sequence[str]) -> pd.DataFrame:
    """PR_i(month) / median PR of the reference (old-cohort) plants in the same month.
    Removes fleet-wide (systemic) shifts such as irradiance-sensor changes."""
    ref = pr_m[list(ref_cols)].where(covered[list(ref_cols)]).median(axis=1)
    return pr_m.div(ref, axis=0)


def change_point(series: pd.Series):
    """Single mean-shift change point (least squares); returns (timestamp, shift, p)."""
    s = series.dropna()
    if len(s) < 8:
        return pd.NaT, np.nan, np.nan
    x = s.to_numpy(float)
    best, k_best = np.inf, None
    for k in range(3, len(x) - 3):
        c = ((x[:k] - x[:k].mean()) ** 2).sum() + ((x[k:] - x[k:].mean()) ** 2).sum()
        if c < best:
            best, k_best = c, k
    from scipy.stats import mannwhitneyu
    p = mannwhitneyu(x[:k_best], x[k_best:]).pvalue
    return s.index[k_best], float(x[k_best:].mean() / x[:k_best].mean() - 1.0) * 100.0, float(p)


def systemic_2023_check(pr_m: pd.DataFrame, covered: pd.DataFrame, cohorts: pd.DataFrame,
                        df: pd.DataFrame, irr_col: str = "Irradiance (W/m2)",
                        window_months: int = 2) -> Dict[str, object]:
    old = cohorts.index[cohorts["cohort"] == 1].tolist()
    if not old:
        return dict(flag=False, note="no cohort 1 plants")
    fleet = pr_m[old].where(covered[old]).median(axis=1)
    cp_fleet, shift_fleet, p_fleet = change_point(fleet)
    plant_cp = {}
    for c in old:
        t, sh, p = change_point(pr_m[c].where(covered[c]))
        plant_cp[c] = (t, sh, p)
    sync = [c for c, (t, sh, p) in plant_cp.items()
            if pd.notna(t) and pd.notna(cp_fleet) and abs((t - cp_fleet).days) <= 31 * window_months
            and sh < 0]
    frac = len(sync) / max(len(old), 1)
    ghi = df[irr_col]
    ghi_year = ghi[ghi > 200].groupby(ghi[ghi > 200].index.year).agg(["mean", "count"])
    ghi_year["annual_kWh_m2"] = ghi.groupby(ghi.index.year).sum() / 1000.0
    flag = bool(frac >= 0.6 and pd.notna(cp_fleet) and shift_fleet < -3 and p_fleet < 0.05)
    return dict(flag=flag, fleet_change_point=cp_fleet, fleet_shift_pct=shift_fleet,
                fleet_p_value=p_fleet, synchronous_fraction=frac, n_old=len(old),
                plant_change_points=plant_cp, fleet_median_pr=fleet, ghi_year=ghi_year,
                note=("synchronous shift of most cohort 1 plants — probably a systemic change "
                      "(sensor / data processing / pyranometer); degradation is also assessed with the "
                      "fleet-normalised PR and on the 2021–2022 period") if flag else
                     "synchronous systemic shift not confirmed")


def analyse_cohorts(df: pd.DataFrame, phys: Dict[str, np.ndarray], pv_cols: Sequence[str],
                    observed: Optional[pd.DataFrame] = None, activity: Optional[pd.DataFrame] = None,
                    min_cov: float = 0.8, years=(2021, 2022, 2023),
                    irr_valid: Optional[np.ndarray] = None,
                    min_months_year: int = 5) -> CohortAnalysis:
    valid = coverage_mask(df, phys, pv_cols, observed, activity)
    if irr_valid is not None:                      # months with a broken pyranometer
        valid = valid & np.asarray(irr_valid, bool)[:, None]
    expected = expected_mask(df, phys, pv_cols)
    cov_m = monthly_coverage(valid, expected)
    pr_m = monthly_pr(df, phys, valid)
    covered = cov_m >= min_cov
    cohorts = assign_cohorts(cov_m, min_cov=min_cov, years=years, min_months_year=min_months_year)
    old = cohorts.index[cohorts["cohort"] == 1].tolist()
    fn = fleet_normalised(pr_m, covered, old) if old else pr_m * np.nan
    sysc = systemic_2023_check(pr_m, covered, cohorts, df)

    pr_year = pd.DataFrame({y: pr_m.where(covered)[pr_m.index.year == y].median() for y in years})
    rows = []
    for c in pv_cols:
        coh = int(cohorts.loc[c, "cohort"])
        r = dict(plant=short(c), column=c, cohort=coh)
        for y in years:
            r[f"PR_{y}"] = pr_year.loc[c, y]
        d, dlo, dhi, n = matched_delta(pr_m[c], covered[c], 2022, 2023)
        r.update(dPR_2023_2022_pct=d, dPR_2023_2022_lo=dlo, dPR_2023_2022_hi=dhi, n_matched_months=n)
        dn, dnlo, dnhi, _ = matched_delta(fn[c], covered[c], 2022, 2023)
        r.update(dPR_2023_2022_fleetnorm_pct=dn, dPR_2023_2022_fleetnorm_lo=dnlo,
                 dPR_2023_2022_fleetnorm_hi=dnhi)
        if coh == 1:
            yo, ylo, yhi, ny = yoy_degradation(pr_m[c], covered[c], years)
            st, se = seasonal_trend(pr_m[c], covered[c])
            yo2, ylo2, yhi2, _ = yoy_degradation(pr_m[c], covered[c], years[:2])
            fyo, fylo, fyhi, _ = yoy_degradation(fn[c], covered[c], years)
            d21, _, _, _ = matched_delta(pr_m[c], covered[c], 2021, 2022)
            r.update(Rd_yoy_pct_per_yr=yo, Rd_yoy_lo=ylo, Rd_yoy_hi=yhi, n_yoy_pairs=ny,
                     Rd_seasonal_ols_pct_per_yr=st, Rd_seasonal_ols_se=se,
                     Rd_yoy_2021_2022_pct_per_yr=yo2,
                     Rd_yoy_fleetnorm_pct_per_yr=fyo, Rd_yoy_fleetnorm_lo=fylo, Rd_yoy_fleetnorm_hi=fyhi,
                     dPR_2022_2021_pct=d21)
            # headline figure: if the 2023 drop is systemic use the fleet-normalised rate
            r["degradation_pct_headline"] = fyo if sysc.get("flag") else yo
            r["metric_type"] = ("Rd, %/yr (YoY, fleet-norm.)" if sysc.get("flag") else "Rd, %/yr (YoY)")
        elif coh == 2:
            r["degradation_pct_headline"] = dn if sysc.get("flag") else d
            r["metric_type"] = ("ΔPR 2023/2022, % (fleet-norm.; not long-term degradation)" if sysc.get("flag")
                                else "ΔPR 2023/2022, % (not long-term degradation)")
        elif coh == 3:
            r["degradation_pct_headline"] = np.nan
            r["metric_type"] = "degradation not computed (2023 only)"
        else:
            r["degradation_pct_headline"] = np.nan
            r["metric_type"] = "insufficient data"
        r["mean_monthly_coverage"] = float(cov_m[c].mean())
        rows.append(r)
    deg = pd.DataFrame(rows).set_index("column")
    return CohortAnalysis(cov_m=cov_m, pr_m=pr_m, cohorts=cohorts, degradation=deg,
                          systemic=sysc, pr_year=pr_year, fleet_norm_pr_m=fn)


def cohort_summary(deg: pd.DataFrame) -> pd.DataFrame:
    """Median / IQR of the cohort-appropriate metric for the paper table."""
    out = []
    for coh, g in deg.groupby("cohort"):
        if coh == 1:
            cols = ["Rd_yoy_pct_per_yr", "Rd_yoy_fleetnorm_pct_per_yr", "Rd_seasonal_ols_pct_per_yr",
                    "Rd_yoy_2021_2022_pct_per_yr", "dPR_2023_2022_pct", "dPR_2023_2022_fleetnorm_pct"]
        elif coh == 2:
            cols = ["dPR_2023_2022_pct", "dPR_2023_2022_fleetnorm_pct"]
        else:
            cols = ["PR_2023"]
        for c in cols:
            if c in g:
                x = g[c].astype(float).dropna()
                out.append(dict(cohort=coh, cohort_name=COHORT_NAMES.get(coh), metric=c, n=len(x),
                                median=x.median() if len(x) else np.nan,
                                q25=x.quantile(.25) if len(x) else np.nan,
                                q75=x.quantile(.75) if len(x) else np.nan,
                                mean=x.mean() if len(x) else np.nan))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- irradiance sensor
def clearness_index(df: pd.DataFrame, lat: float = 22.3363, lon: float = 114.2634,
                    irr_col: str = "Irradiance (W/m2)", tz: float = 8.0) -> pd.Series:
    """Hourly clearness index k_t = GHI / (G_sc * E0 * cos z) for cos z > 0.3."""
    from pv_physics_graph import PhysicsPrior
    t = pd.DatetimeIndex(df.index)
    z, _ = PhysicsPrior.solar_position(t, lat, lon, tz)
    cz = np.cos(np.deg2rad(z))
    doy = t.dayofyear.to_numpy(float)
    ext = 1367.0 * (1 + 0.033 * np.cos(np.deg2rad(360 * doy / 365.0))) * np.clip(cz, 0, None)
    kt = np.where(cz > 0.3, df[irr_col].to_numpy(float) / np.maximum(ext, 1.0), np.nan)
    return pd.Series(kt, index=df.index, name="kt")


def irradiance_sensor_check(df: pd.DataFrame, irr_col: str = "Irradiance (W/m2)",
                            q: float = 0.98, min_kt: float = 0.45, **kw) -> Dict[str, object]:
    """PV-independent test of the pyranometer: the monthly upper quantile of k_t
    (clear-sky hours) must be stable and physically plausible (~0.65-0.80 for hourly
    data).  A step change indicates a sensor replacement / re-scaling, i.e. a
    SYSTEMIC change, not degradation.  Months with q-quantile < ``min_kt`` are
    treated as sensor failure (irradiance invalid)."""
    kt = clearness_index(df, irr_col=irr_col, **kw)
    ktq = kt.resample("MS").quantile(q)
    bad_months = ktq.index[ktq < min_kt]
    good = ktq.drop(bad_months)
    cp, shift, p = change_point(good)
    if pd.notna(cp) and abs(shift) > 10 and p < 0.01:
        before = good[good.index < cp].tail(12).median()
        after = good[good.index >= cp].head(12).median()
        factor = float(before / after)
    else:
        cp, factor = pd.NaT, 1.0
    month = pd.DatetimeIndex(df.index).to_period("M").to_timestamp()
    irr_valid = ~np.isin(month, bad_months)
    return dict(kt_quantile=ktq, change_point=cp, shift_pct=shift, p_value=p,
                factor_after_cp=factor, invalid_months=list(bad_months), irr_valid=irr_valid,
                systemic=bool(pd.notna(cp)))


def harmonise_irradiance(df: pd.DataFrame, info: Dict[str, object],
                         irr_col: str = "Irradiance (W/m2)") -> pd.DataFrame:
    """Rescale GHI after the detected sensor change point to the pre-change level."""
    out = df.copy()
    cp = info.get("change_point")
    if pd.notna(cp) and info.get("factor_after_cp", 1.0) != 1.0:
        m = out.index >= cp
        out.loc[m, irr_col] = out.loc[m, irr_col] * float(info["factor_after_cp"])
    return out
