"""Section 15, stage 0: inverter-level modalities of the HKUST dataset (loader, hourly
aggregation, site-level modal channels and the data audit).

The HKUST release contains, besides the site-level power used everywhere else in the project,
44 inverter files (5-min) for the 37 stations with panel-level optimisers:

    Time, dcVoltage(V), totalActivePower(W),
    L{1,2,3}_acCurrent(A), L{1,2,3}_acFrequency(Hz), L{1,2,3}_acVoltage(V),
    L{1,2,3}_activePower(W), L{1,2,3}_reactivePower(W)

File names are ``<Site>_Inverter.csv`` or ``<Site>_Inverter_<k>.csv``; ``<Site>`` is the
stem of the site-level file, so the column of the site in the project frame is
``power(W)_<Site>``.  The 23 stations without optimisers have no inverter data.

This module is independent of the rest of the pipeline (it only needs the hourly frame
``D["df"]`` and the PV column names); nothing here changes sections 1-14.

Hourly aggregation follows the rule of the site-level preprocessing (NaN kept until the
aggregation, an hour is valid when enough 5-min samples are present: ``min_count`` of 12).

Per inverter and hour (``INV_FEATS``):
    P       mean active power, W
    n_obs   number of valid 5-min samples
    n_zero  5-min samples with P below ``zero_w`` (short outages invisible in the hourly mean)
    cv      coefficient of variation of the 5-min power inside the hour
    dcV     median DC voltage, V
    dc_on   share of 5-min samples with dcV above ``dc_on_frac`` x the typical producing dcV
    vac_max maximum over the hour and the three phases of the AC voltage, V
    vac     mean phase voltage, V
    f_dev   maximum |f - 50 Hz|
    imb     mean phase imbalance (max - min) / mean of L1..L3 active power (producing samples)
    q       mean total reactive power, W

Site-level modal channels (``SITE_FEATS``; computed for any set of timestamps):
    has     inverter data present in this hour
    meter   log(site P / sum of inverter P) — metering consistency (both producing)
    sib     min_k (P_k / sum P) / share_k — inverter-sibling ratio (sites with >= 2 inverters)
    vac     max over the site's inverters of (vac_max - over-voltage threshold), V (> 0: over-voltage)
    f_dev   max frequency deviation, Hz
    imb     max phase imbalance
    n_zero  max number of zero 5-min samples
    dc_off  share of the site's inverters with dc_on < 0.5 in a producing hour
Modality groups (for dropout / leave-one-modality-out): ``MODALITIES``.
"""
from __future__ import annotations

import copy
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

IRR = "Irradiance (W/m2)"
INV_FEATS = ("P", "n_obs", "n_zero", "cv", "dcV", "dc_on", "vac_max", "vac", "f_dev", "imb", "q")
SITE_FEATS = ("has", "meter", "sib", "vac", "f_dev", "imb", "n_zero", "dc_off")
MODALITIES = {                       # modality group -> site channels it provides
    "inverter": ("sib", "dc_off"),   # inverter siblings + DC state
    "subhourly": ("n_zero",),        # 5-min power inside the hour
    "phase": ("imb",),               # per-phase power
    "grid": ("vac", "f_dev"),        # AC voltage / frequency (grid side)
    "metering": ("meter",),          # site meter vs sum of inverters
}
MODALITY_LABEL = {"inverter": "Inverter siblings + DC state", "subhourly": "Sub-hourly (5-min) power",
                  "phase": "Per-phase power", "grid": "AC voltage / frequency", "metering": "Site meter vs inverter sum"}


def _log(*a):
    print(time.strftime("%H:%M:%S"), "[modal]", *a, flush=True)


@dataclass
class ModalConfig:
    min_count: int = 9                 # valid 5-min samples per hour (9 of 12 = "3 of 4")
    zero_w: float = 20.0               # W, a 5-min sample below this (and below 0.5 % of p99) = zero
    dc_on_frac: float = 0.5            # dcV above this share of the typical producing dcV = DC side on
    nominal_v: float = 220.0           # HK low-voltage supply, phase-to-neutral
    vac_limit: float = 233.2           # 220 V + 6 %: lower bound of the over-voltage threshold (assumption)
    vac_q: float = 0.995               # per-inverter over-voltage threshold = this quantile of training vac_max
    f_limit: float = 0.2               # Hz, frequency excursion flag
    f_nominal: float = 50.0
    meter_min_frac: float = 0.05       # metering check only when both P > this share of rated
    sib_min_frac: float = 0.10         # sibling ratio only when the site sum > this share of rated
    share_min_n: int = 50              # producing training hours to estimate inverter shares
    boundary: str = "2023-01-01"


@dataclass
class ModalData:
    """Hourly inverter channels aligned with the project's hourly index."""
    index: pd.DatetimeIndex
    inv: Dict[str, pd.DataFrame]               # inverter id -> hourly frame (INV_FEATS)
    site_of: Dict[str, str]                    # inverter id -> pv column
    rated_w: Dict[str, float]                  # pv column -> rated W (for normalisation)
    cfg: ModalConfig = field(default_factory=ModalConfig)
    share: Dict[str, float] = field(default_factory=dict)   # inverter id -> training share of the site sum
    vac_thr: Dict[str, float] = field(default_factory=dict) # inverter id -> AC over-voltage threshold, V
    source: str = ""
    raw_stats: Optional[pd.DataFrame] = None   # per inverter: coverage of the 5-min file

    # ---------------------------------------------------------------- structure
    def sites(self) -> List[str]:
        return sorted(set(self.site_of.values()))

    def inverters_of(self, col: str) -> List[str]:
        return sorted(k for k, s in self.site_of.items() if s == col)

    def copy(self) -> "ModalData":
        out = copy.copy(self)
        out.inv = {k: v.copy() for k, v in self.inv.items()}
        return out

    def fit_shares(self):
        """Training-period references: median share of every inverter in its site's inverter sum and
        the inverter's AC over-voltage threshold = max(vac_limit, q99.5 of its producing-hour vac_max).
        A per-inverter threshold is needed because the local voltage level differs by > 10 V between
        HKUST buildings (several feeders run above 220 V + 6 % most of the time)."""
        b = pd.Timestamp(self.cfg.boundary)
        for k, F in self.inv.items():
            c = self.site_of[k]
            prod = F["P"] > 0.05 * self.rated_w.get(c, 1.0) / max(len(self.inverters_of(c)), 1)
            v = F.loc[prod & (F.index < b), "vac_max"].dropna()
            if len(v) < self.cfg.share_min_n:
                v = F.loc[prod, "vac_max"].dropna()
            self.vac_thr[k] = float(max(self.cfg.vac_limit, v.quantile(self.cfg.vac_q))) if len(v) else self.cfg.vac_limit
        for s in self.sites():
            ks = self.inverters_of(s)
            if len(ks) < 2:
                for k in ks:
                    self.share[k] = 1.0
                continue
            P = pd.concat({k: self.inv[k]["P"] for k in ks}, axis=1)
            tot = P.sum(1, min_count=len(ks))
            ok = (P.index < b) & (tot > self.cfg.sib_min_frac * self.rated_w.get(s, tot.max() or 1.0))
            if ok.sum() < self.cfg.share_min_n:          # cold start: all available producing hours
                ok = tot > self.cfg.sib_min_frac * self.rated_w.get(s, tot.max() or 1.0)
            for k in ks:
                v = (P.loc[ok, k] / tot[ok]).median() if ok.any() else np.nan
                self.share[k] = float(v) if np.isfinite(v) and v > 0 else 1.0 / len(ks)
        return self

    # ---------------------------------------------------------------- site channels
    def site_arrays(self, pv_cols: Sequence[str], ts: pd.DatetimeIndex, site_p: Optional[np.ndarray] = None,
                    drop: Sequence[str] = ()) -> Dict[str, np.ndarray]:
        """(S, n) arrays of SITE_FEATS at timestamps ``ts``; NaN where not available.
        ``site_p`` (S, n) site power in W (for the metering channel).  ``drop``: modality groups to
        remove (leave-one-modality-out / missing-modality robustness)."""
        S, n = len(ts), len(pv_cols)
        out = {f: np.full((S, n), np.nan) for f in SITE_FEATS}
        out["has"] = np.zeros((S, n))
        cfg = self.cfg
        for j, c in enumerate(pv_cols):
            ks = self.inverters_of(c)
            if not ks:
                continue
            F = {k: self.inv[k].reindex(ts) for k in ks}
            P = np.stack([F[k]["P"].to_numpy(float) for k in ks], 1)
            ok = np.isfinite(P).all(1)
            out["has"][:, j] = ok.astype(float)
            tot = np.where(ok, P.sum(1), np.nan)
            rw = self.rated_w.get(c) or np.nanmax(np.r_[tot, 1.0])
            if len(ks) >= 2:
                sh = np.array([self.share.get(k, 1.0 / len(ks)) for k in ks])
                import warnings
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    rel = (P / tot[:, None]) / sh[None, :]
                    out["sib"][:, j] = np.where(ok & (tot > cfg.sib_min_frac * rw), np.nanmin(rel, 1), np.nan)
            def agg(f, how):
                A = np.stack([F[k][f].to_numpy(float) for k in ks], 1)
                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        return getattr(np, how)(A, 1)
            V = np.stack([F[k]["vac_max"].to_numpy(float) - self.vac_thr.get(k, cfg.vac_limit) for k in ks], 1)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    out["vac"][:, j] = np.nanmax(V, 1)          # margin above the over-voltage threshold, V
            out["f_dev"][:, j] = agg("f_dev", "nanmax")
            out["imb"][:, j] = agg("imb", "nanmax")
            out["n_zero"][:, j] = agg("n_zero", "nanmax")
            dc = np.stack([F[k]["dc_on"].to_numpy(float) for k in ks], 1)
            prod = tot > cfg.sib_min_frac * rw if len(ks) >= 2 else np.ones(S, bool)
            with np.errstate(all="ignore"):
                out["dc_off"][:, j] = np.where(np.isfinite(dc).all(1), (dc < 0.5).mean(1), np.nan)
            if site_p is not None:
                sp = site_p[:, j]
                both = ok & (np.nan_to_num(sp) > cfg.meter_min_frac * rw) & (np.nan_to_num(tot) > cfg.meter_min_frac * rw)
                with np.errstate(all="ignore"):
                    out["meter"][:, j] = np.where(both, np.log(sp / tot), np.nan)
            for f in SITE_FEATS:
                if f != "has":
                    out[f][~ok, j] = np.nan
        for g in drop:
            for f in MODALITIES[g]:
                out[f][:] = np.nan
        return out


# ============================================================================ loading
def _inv_id(stem: str):
    m = re.match(r"^(.*)_Inverter(?:_(\d+))?$", stem)
    if not m:
        return None, None
    return m.group(1), int(m.group(2) or 1)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def site_column_map(pv_cols: Sequence[str], site_names: Sequence[str]) -> Dict[str, Optional[str]]:
    """Site stem of an inverter file -> pv column ``power(W)_<stem>`` (normalised match)."""
    by = {_norm(c.replace("power(W)_", "")): c for c in pv_cols}
    return {s: by.get(_norm(s)) for s in site_names}


def hourly_from_5min(raw: pd.DataFrame, index: pd.DatetimeIndex, cfg: ModalConfig) -> pd.DataFrame:
    """Hourly INV_FEATS of one inverter from its 5-min frame (DatetimeIndex)."""
    g = lambda c: pd.to_numeric(raw[c], errors="coerce") if c in raw else pd.Series(np.nan, index=raw.index)
    P = g("totalActivePower(W)").clip(lower=0)
    ph = pd.concat([g(f"L{i}_activePower(W)").clip(lower=0) for i in (1, 2, 3)], axis=1)
    V = pd.concat([g(f"L{i}_acVoltage(V)") for i in (1, 2, 3)], axis=1)
    Fq = pd.concat([g(f"L{i}_acFrequency(Hz)") for i in (1, 2, 3)], axis=1)
    Q = pd.concat([g(f"L{i}_reactivePower(W)") for i in (1, 2, 3)], axis=1).sum(1, min_count=1)
    dcV = g("dcVoltage(V)")
    V = V.where((V > 50) & (V < 400)); Fq = Fq.where((Fq > 40) & (Fq < 60))
    p99 = float(np.nanquantile(P, 0.99)) if P.notna().any() else 0.0
    zero_thr = max(cfg.zero_w, 0.005 * p99)
    prod = P > max(0.05 * p99, cfg.zero_w)
    dc_typ = float(np.nanmedian(dcV[prod])) if prod.any() and dcV[prod].notna().any() else np.nan
    with np.errstate(all="ignore"):
        imb = (ph.max(1) - ph.min(1)) / ph.mean(1)
    imb = imb.where(prod)
    frame = pd.DataFrame({"P": P, "is_zero": (P < zero_thr).where(P.notna()).astype(float),
                          "dcV": dcV, "dc_on": (dcV > cfg.dc_on_frac * dc_typ).where(dcV.notna()).astype(float)
                          if np.isfinite(dc_typ) else np.nan,
                          "vmax": V.max(1), "vmean": V.mean(1), "fdev": (Fq - cfg.f_nominal).abs().max(1),
                          "imb": imb, "q": Q})
    r = frame.resample("1h")
    n_obs = r["P"].count()
    out = pd.DataFrame({"P": r["P"].mean(), "n_obs": n_obs, "n_zero": r["is_zero"].sum(min_count=1),
                        "cv": r["P"].std() / r["P"].mean().replace(0, np.nan), "dcV": r["dcV"].median(),
                        "dc_on": r["dc_on"].mean(), "vac_max": r["vmax"].max(), "vac": r["vmean"].mean(),
                        "f_dev": r["fdev"].max(), "imb": r["imb"].mean(), "q": r["q"].mean()})
    bad = out["n_obs"] < cfg.min_count
    out.loc[bad, [c for c in out.columns if c != "n_obs"]] = np.nan
    out = out.reindex(index)
    out["n_obs"] = out["n_obs"].fillna(0)
    return out[list(INV_FEATS)]


def load_inverters(inv_dir: str, D: dict, cfg: Optional[ModalConfig] = None, files: Optional[Sequence[str]] = None,
                   encoding: str = "cp1251", verbose: bool = True) -> ModalData:
    """Read all ``*_Inverter*.csv`` files and aggregate them to the hourly index of ``D["df"]``."""
    cfg = cfg or ModalConfig()
    index = pd.DatetimeIndex(D["df"].index)
    pv = list(D["pv_cols"])
    names = sorted(f for f in os.listdir(inv_dir) if f.lower().endswith(".csv")) if files is None else list(files)
    stems = {f: os.path.splitext(f)[0] for f in names}
    smap = site_column_map(pv, {(_inv_id(s)[0]) for s in stems.values() if _inv_id(s)[0]})
    inv, site_of, stats = {}, {}, []
    t0 = time.time()
    for f in names:
        site, k = _inv_id(stems[f])
        col = smap.get(site) if site else None
        if col is None:
            stats.append(dict(inverter=stems[f], site=site, column=None, note="no matching site column")); continue
        raw = pd.read_csv(os.path.join(inv_dir, f), encoding=encoding)
        raw.columns = [c.strip() for c in raw.columns]
        t = pd.to_datetime(raw[raw.columns[0]], errors="coerce")
        raw = raw.loc[t.notna()].set_index(t[t.notna()]); raw = raw[~raw.index.duplicated(keep="last")]
        raw = raw.loc[(raw.index >= index[0]) & (raw.index < index[-1] + pd.Timedelta(hours=1))]
        key = f"{site} #{k}"
        inv[key] = hourly_from_5min(raw, index, cfg); site_of[key] = col
        P = pd.to_numeric(raw.get("totalActivePower(W)"), errors="coerce")
        # the files only cover the recording window of the day (about 06:00-18:00): expected samples =
        # days x 5-min slots of the day that are recorded on at least 10 % of the days
        if len(raw):
            slot = raw.index.hour * 12 + raw.index.minute // 5
            days = raw.index.normalize().nunique()
            n_slots = int((pd.Series(slot).value_counts() >= 0.1 * days).sum())
            span_days = (raw.index.max().normalize() - raw.index.min().normalize()).days + 1
            expected = span_days * n_slots
        else:
            n_slots, expected = 0, 0
        stats.append(dict(inverter=key, site=site, column=col, first=raw.index.min(), last=raw.index.max(),
                          slots_per_day=n_slots, samples_5min=int(len(raw)), expected_5min=int(expected),
                          missing_pct=100 * (1 - P.notna().sum() / max(expected, 1)),
                          note=""))
    rated = {c: float((D.get("rated") or {}).get(c) or 0.0) * 1000.0 for c in pv}
    for c in pv:
        if not rated[c] > 0:
            x = D["df"][c].to_numpy(float)
            rated[c] = float(np.nanquantile(x, 0.995)) if np.isfinite(x).any() else 1.0
    M = ModalData(index=index, inv=inv, site_of=site_of, rated_w=rated, cfg=cfg,
                  source=f"HKUST inverter files ({len(inv)} inverters)", raw_stats=pd.DataFrame(stats)).fit_shares()
    if verbose:
        _log(f"{len(inv)} inverters at {len(M.sites())} sites loaded in {time.time() - t0:.0f}s")
    return M


def load_site_energy(site_dir: str, D: dict, encoding: str = "cp1251") -> Dict[str, pd.DataFrame]:
    """15-min site files (generation kWh, power W) of the optimiser stations for the energy check."""
    pv = list(D["pv_cols"]); out = {}
    for f in sorted(os.listdir(site_dir)):
        if not f.lower().endswith(".csv"):
            continue
        col = site_column_map(pv, [os.path.splitext(f)[0]]).get(os.path.splitext(f)[0])
        if col is None:
            continue
        r = pd.read_csv(os.path.join(site_dir, f), encoding=encoding)
        t = pd.to_datetime(r[r.columns[0]], errors="coerce")
        r = r.loc[t.notna()].set_index(t[t.notna()])
        r = r[~r.index.duplicated(keep="last")]
        gen = [c for c in r.columns if "generation" in c.lower()]
        pw = [c for c in r.columns if "power" in c.lower()]
        out[col] = pd.DataFrame({"gen_kwh": pd.to_numeric(r[gen[0]], errors="coerce") if gen else np.nan,
                                 "power_w": pd.to_numeric(r[pw[0]], errors="coerce") if pw else np.nan})
    return out


# ============================================================================ audit (stage 0)
def _short(c: str) -> str:
    return str(c).replace("power(W)_", "")


def audit(D: dict, M: ModalData, site_energy: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
    """Stage-0 tables (English): mapping / coverage, Brick consistency, energy consistency,
    DC-voltage states, AC voltage and phase imbalance."""
    cfg = M.cfg; df = D["df"]; pv = list(D["pv_cols"])
    brick = D.get("brick"); mapping = D.get("mapping") or {}
    T = {}
    # --- mapping and coverage
    rows = []
    for c in M.sites():
        ks = M.inverters_of(c)
        site = brick.sites.get(mapping.get(c)) if brick is not None and hasattr(brick, "sites") else None
        tilts = list(getattr(site, "tilts_deg", []) or []) if site is not None else []
        azs = list(getattr(site, "azimuths_deg", []) or []) if site is not None else []
        st = M.raw_stats.set_index("inverter").reindex(ks) if M.raw_stats is not None and len(M.raw_stats) else None
        hourly_cov = np.mean([M.inv[k]["P"].notna().mean() for k in ks])
        rows.append({"Site": _short(c), "Inverter files": len(ks),
                     "Inverters in Brick": int(getattr(site, "n_inverters", 0) or 0) if site is not None else np.nan,
                     "Orientations in Brick": max(len(set(zip(tilts, azs))), 1 if site is not None else 0),
                     "Mixed orientation (Brick)": bool(site is not None and "mixed" in str(getattr(site, "tilt_raw", "")).lower()
                                                       or site is not None and "mixed" in str(getattr(site, "azimuth_raw", "")).lower()),
                     "Rated kW": round(M.rated_w.get(c, np.nan) / 1000, 1),
                     "First record": None if st is None else st["first"].min(),
                     "Missing 5-min samples, %": np.nan if st is None or "missing_pct" not in st else round(float(st["missing_pct"].mean()), 2),
                     "Valid hours, %": round(100 * hourly_cov, 2)})
    T["coverage"] = pd.DataFrame(rows)
    none = [c for c in pv if c not in M.sites()]
    T["summary"] = pd.DataFrame([
        {"Item": "Inverters with data", "Value": len(M.inv)},
        {"Item": "Sites with inverter data", "Value": len(M.sites())},
        {"Item": "Sites with >= 2 inverters", "Value": sum(len(M.inverters_of(c)) >= 2 for c in M.sites())},
        {"Item": "Sites without inverter data", "Value": len(none)},
        {"Item": "Unmatched inverter files", "Value": int((M.raw_stats["column"].isna()).sum()) if M.raw_stats is not None and len(M.raw_stats) else 0},
        {"Item": "Mean missing 5-min samples, %", "Value": round(float(M.raw_stats["missing_pct"].mean()), 2)
         if M.raw_stats is not None and "missing_pct" in M.raw_stats else np.nan}])
    # --- energy consistency: site meter vs sum of inverters (daily), generation vs integral of power
    rows = []
    for c in M.sites():
        ks = M.inverters_of(c)
        Pi = pd.concat([M.inv[k]["P"] for k in ks], axis=1).sum(1, min_count=len(ks))
        Ps = df[c].where(D["observed"][c]) if "observed" in D and c in D["observed"] else df[c]
        both = Pi.notna() & Ps.notna()
        e_inv = (Pi.where(both)).resample("D").sum(min_count=8) / 1000      # inverter files: ~06-18 h only
        e_site = (Ps.where(both)).resample("D").sum(min_count=8) / 1000
        ok = e_inv.notna() & e_site.notna() & (e_site > 1)
        rel = (e_site[ok] - e_inv[ok]) / e_site[ok]
        r = {"Site": _short(c), "Days compared": int(ok.sum()),
             "Median |site - sum inv| / site, %": round(100 * float(rel.abs().median()), 3) if ok.any() else np.nan,
             "Days with |diff| > 5 %, %": round(100 * float((rel.abs() > 0.05).mean()), 2) if ok.any() else np.nan}
        if site_energy and c in site_energy:
            se = site_energy[c]
            okp = se["gen_kwh"].notna() & se["power_w"].notna()          # 15-min samples with both values
            g = se["gen_kwh"].where(okp).resample("D").sum(min_count=40)
            p = (se["power_w"].where(okp) * 0.25 / 1000).resample("D").sum(min_count=40)
            k2 = g.notna() & p.notna() & (g > 1)
            rr = (g[k2] - p[k2]) / g[k2]
            r["Median |generation - integral of power| / generation, %"] = round(100 * float(rr.abs().median()), 3) if k2.any() else np.nan
        if site_energy and c in site_energy and k2.any():
            r["Median signed (generation - integral) / generation, %"] = round(100 * float(rr.median()), 3)
        rows.append(r)
    T["energy"] = pd.DataFrame(rows)
    # --- DC voltage states, AC voltage, imbalance, frequency
    rows = []
    for k, F in M.inv.items():
        c = M.site_of[k]
        prod = F["P"] > 0.05 * M.rated_w.get(c, 1.0) / max(len(M.inverters_of(c)), 1)
        night = F["P"] < cfg.zero_w
        v = F["vac_max"]
        rows.append({"Inverter": k,
                     "dcV median producing, V": round(float(F.loc[prod, "dcV"].median()), 1) if prod.any() else np.nan,
                     "dcV IQR producing, V": round(float(F.loc[prod, "dcV"].quantile(.75) - F.loc[prod, "dcV"].quantile(.25)), 1) if prod.any() else np.nan,
                     "dcV median not producing, V": round(float(F.loc[night, "dcV"].median()), 1) if night.any() else np.nan,
                     "Producing hours with DC off, %": round(100 * float((F.loc[prod, "dc_on"] < 0.5).mean()), 2) if prod.any() else np.nan,
                     "AC V p50, V": round(float(v.median()), 1) if v.notna().any() else np.nan,
                     "AC V p99, V": round(float(v.quantile(.99)), 1) if v.notna().any() else np.nan,
                     f"Hours with AC V > {cfg.vac_limit:g} V, %": round(100 * float((v[prod] > cfg.vac_limit).mean()), 2) if prod.any() else np.nan,
                     "Over-voltage threshold, V": round(M.vac_thr.get(k, np.nan), 1),
                     "Max |f - 50| p99, Hz": round(float(F["f_dev"].quantile(.99)), 3) if F["f_dev"].notna().any() else np.nan,
                     "Phase imbalance p50": round(float(F.loc[prod, "imb"].median()), 3) if prod.any() else np.nan,
                     "Phase imbalance p99": round(float(F.loc[prod, "imb"].quantile(.99)), 3) if prod.any() else np.nan,
                     "Producing hours with >= 1 zero 5-min sample, %": round(100 * float((F.loc[prod, "n_zero"] >= 1).mean()), 2) if prod.any() else np.nan})
    T["channels"] = pd.DataFrame(rows)
    return T


TITLES = {"summary": "Stage 0: inverter modality — summary",
          "coverage": "Stage 0: inverter files mapped to sites (Brick consistency, coverage)",
          "energy": "Stage 0: energy consistency (site meter vs sum of inverters; generation vs integral of power)",
          "channels": "Stage 0: DC voltage states, AC voltage, frequency, phase imbalance, 5-min zeros"}
