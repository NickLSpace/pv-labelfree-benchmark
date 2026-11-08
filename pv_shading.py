"""Section 16: generated modality "shading mask from neighbouring buildings and terrain".

The HKUST Brick metadata give every site the same default orientation (tilt 10 deg, azimuth
180 deg) and coordinates with two decimals, and there is no 3D model of the campus in the
dataset.  A geometric shading mask therefore cannot be computed from the dataset alone.  The
mask is GENERATED instead, in one of two ways (a site uses the geometric mask if it has one):

1. **Geometric (optional)** — a horizon profile per site (``horizon_csv``: ``column`` or
   ``site``, ``azimuth_deg`` (0 = north, clockwise), ``horizon_deg``), e.g. from a 3D city model /
   LiDAR (buildings) or a DEM horizon such as the PVGIS horizon service (terrain), or obstacle
   rows (``azimuth_deg``, ``width_deg``, ``distance_m``, ``height_m`` above the array) that are
   converted to a horizon.  Beam irradiance is blocked when the sun is below the horizon.
2. **Data-driven (default)** — the obstruction map of each site is estimated on the TRAINING
   period only (no look-ahead): on clear hours the ratio ``P / P_exp`` is binned by sun azimuth
   x sun elevation; the loss of a bin relative to the site's high-sun reference is the shading
   loss of that sun position.  The loss is made non-increasing with elevation for each azimuth
   (an obstacle that shades the sun at elevation e also shades it below e) and smoothed.  This
   recovers fixed obstructions (neighbouring buildings, parapets, terrain) and also absorbs the
   systematic low-sun error of the P_exp model at that site — both are fixed functions of the
   sun position and should not raise alarms.

Hourly channels (``ShadingMask.channels``):

* ``s_bin``  shading loss of the beam at the current sun position (0 = unshaded, 1 = fully
  blocked; NaN = sun position not seen in training / sun down);
* ``s_eff``  expected loss of the measured power = ``s_bin`` x beam share of the hour
  (Erbs decomposition of the clearness index, normalised by the clear-sky beam share), i.e. the
  shading matters on clear hours and vanishes under overcast skies;
* ``has``    availability (site has a mask and the sun position is covered).

The mask is fixed in time: a NEW obstruction after training is not in the mask and must still
raise an alarm.

References: Erbs, Klein, Duffie (1982), Solar Energy 28(4):293-302,
https://doi.org/10.1016/0038-092X(82)90302-4; PVGIS horizon API,
https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

IRR = "Irradiance (W/m2)"
HKUST_LATLON = (22.3375, 114.2630)


def _log(*a):
    import time
    print(time.strftime("%H:%M:%S"), "[shading]", *a, flush=True)


# ============================================================================ sun geometry
def solar_position(times: pd.DatetimeIndex, lat: float, lon: float, tz_offset_hours: float = 8.0):
    """NOAA low-precision algorithm, identical to ``pv_physics_graph`` (zenith, azimuth 0 = north)."""
    times = pd.DatetimeIndex(times)
    doy = times.dayofyear.to_numpy(float)
    hour = times.hour.to_numpy(float) + times.minute.to_numpy(float) / 60.0
    b = np.deg2rad(360.0 / 365.0 * (doy - 81.0))
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    lst = hour + (4.0 * (lon - 15.0 * tz_offset_hours) + eot) / 60.0
    hra = np.deg2rad(15.0 * (lst - 12.0))
    decl = np.deg2rad(23.45) * np.sin(b)
    phi = np.deg2rad(lat)
    cos_z = np.clip(np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hra), -1, 1)
    zen = np.rad2deg(np.arccos(cos_z))
    az = np.rad2deg(np.arctan2(np.sin(hra), np.cos(hra) * np.sin(phi) - np.tan(decl) * np.cos(phi))) + 180.0
    return zen, az % 360.0


def erbs_beam_share(kt: np.ndarray) -> np.ndarray:
    """1 - diffuse fraction (Erbs 1982) for the clearness index kt."""
    kt = np.clip(np.nan_to_num(np.asarray(kt, float)), 0, 1.2)
    kd = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4, 0.165))
    return np.clip(1 - kd, 0, 1)


def site_latlon(D: dict) -> tuple:
    """Coordinates used by the physics model (median of the Brick sites) or the HKUST campus."""
    brick, mapping = D.get("brick"), D.get("mapping") or {}
    if brick is not None and hasattr(brick, "sites"):
        ll = [(s.lat, s.lon) for c in D["pv_cols"] for s in [brick.sites.get(mapping.get(c))]
              if s is not None and s.lat is not None and s.lon is not None]
        if ll:
            a = np.asarray(ll, float)
            return float(np.median(a[:, 0])), float(np.median(a[:, 1]))
    return HKUST_LATLON


def sun_frame(D: dict, index: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    idx = pd.DatetimeIndex(D["df"].index if index is None else index)
    lat, lon = site_latlon(D)
    tz = float(getattr(D.get("builder"), "tz", 8.0) or 8.0)
    zen, az = solar_position(idx, lat, lon, tz)
    return pd.DataFrame({"elev": 90.0 - zen, "az": az}, index=idx)


# ============================================================================ configuration / result
@dataclass
class ShadingConfig:
    az_step: float = 10.0
    el_step: float = 5.0
    min_n: int = 6                  # clear training hours per (azimuth, elevation) bin
    kt_clear: float = 0.55          # clear hours: clearness index above ...
    ghi_min: float = 120.0
    el_min: float = 3.0
    ref_el: float = 35.0            # high-sun reference: bins at or above this elevation (or the top 25 %)
    min_loss: float = 0.05          # losses below this are noise (set to 0)
    max_loss: float = 0.95
    min_hours: int = 150            # clear training hours needed for a data-driven mask
    beam_clear: float = 0.835       # Erbs beam share for kt >= 0.8
    boundary: str = "2023-01-01"


@dataclass
class ShadingMask:
    az_edges: np.ndarray
    el_edges: np.ndarray
    loss: Dict[str, np.ndarray]                  # column -> (n_az, n_el) loss, NaN = not covered
    source: Dict[str, str]                       # column -> "geometric" | "data-driven" | "none"
    n_hours: Dict[str, int] = field(default_factory=dict)
    cfg: ShadingConfig = field(default_factory=ShadingConfig)

    def signature(self) -> str:
        h = hashlib.sha256()
        for c in sorted(self.loss):
            h.update(c.encode()); h.update(np.nan_to_num(self.loss[c], nan=-1).tobytes())
        return h.hexdigest()[:12]

    def lookup(self, col: str, elev: np.ndarray, az: np.ndarray) -> np.ndarray:
        L = self.loss.get(col)
        out = np.full(len(elev), np.nan)
        if L is None:
            return out
        up = elev > 0
        ia = np.clip(np.digitize(az, self.az_edges) - 1, 0, L.shape[0] - 1)
        ie = np.clip(np.digitize(np.clip(elev, 0, 89.999), self.el_edges) - 1, 0, L.shape[1] - 1)
        out[up] = L[ia[up], ie[up]]
        return out

    def channels(self, D: dict, pv_cols: Sequence[str], df: Optional[pd.DataFrame] = None,
                 kt: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """(T, n) arrays s_bin, s_eff, has for the frame ``df`` (default D["df"])."""
        df = D["df"] if df is None else df
        sun = sun_frame(D, df.index)
        if kt is None:
            kt = _kt(D, df)
        bs = erbs_beam_share(kt) / self.cfg.beam_clear
        T, n = len(df), len(pv_cols)
        out = {k: np.full((T, n), np.nan) for k in ("s_bin", "s_eff")}
        el, az = sun["elev"].to_numpy(), sun["az"].to_numpy()
        for j, c in enumerate(pv_cols):
            s = self.lookup(c, el, az)
            out["s_bin"][:, j] = s
            out["s_eff"][:, j] = s * np.clip(bs, 0, 1)
        out["has"] = np.isfinite(out["s_bin"]).astype(float)
        return out

    def table(self) -> pd.DataFrame:
        """Per-site summary (English)."""
        rows = []
        for c, L in self.loss.items():
            v = L[np.isfinite(L)]
            rows.append({"Site": _short(c), "Mask": self.source.get(c, "none"), "Clear training hours": self.n_hours.get(c, 0),
                         "Sun positions covered": int(v.size), "Shaded positions (loss > 0)": int((v > 0).sum()),
                         "Mean loss over shaded positions": float(v[v > 0].mean()) if (v > 0).any() else 0.0,
                         "Max loss": float(v.max()) if v.size else np.nan})
        return pd.DataFrame(rows)


def _short(c: str) -> str:
    try:
        from pv_cohorts import short
        return short(c)
    except Exception:  # pragma: no cover
        return str(c)


def _kt(D: dict, df: pd.DataFrame) -> np.ndarray:
    """Clearness index of the frame (D["kt"] for the clean frame; recomputed if the GHI differs)."""
    kt0 = D.get("kt")
    if kt0 is not None and df is D["df"]:
        return np.asarray(pd.Series(kt0).reindex(df.index), float)
    sun = sun_frame(D, df.index)
    doy = df.index.dayofyear.to_numpy(float)
    g0 = 1367.0 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)) * np.clip(np.sin(np.deg2rad(sun["elev"].to_numpy())), 0, None)
    with np.errstate(all="ignore"):
        kt = np.where(g0 > 20, df[IRR].to_numpy(float) / g0, np.nan)
    return np.clip(kt, 0, 1.2)


# ============================================================================ estimation
def _edges(cfg: ShadingConfig):
    return np.arange(0, 360 + cfg.az_step, cfg.az_step), np.arange(0, 90 + cfg.el_step, cfg.el_step)


def _pav_nonincreasing(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic regression (pool adjacent violators), non-increasing in the index."""
    y = list(-v); ww = list(w); blocks = [[i] for i in range(len(v))]
    i = 0
    while i < len(y) - 1:
        if y[i] > y[i + 1]:
            nw = ww[i] + ww[i + 1]; ny = (y[i] * ww[i] + y[i + 1] * ww[i + 1]) / nw
            y[i:i + 2] = [ny]; ww[i:i + 2] = [nw]; blocks[i:i + 2] = [blocks[i] + blocks[i + 1]]
            i = max(i - 1, 0)
        else:
            i += 1
    out = np.empty(len(v))
    for b_, yy in zip(blocks, y):
        out[b_] = -yy
    return out


def _monotone_smooth(L: np.ndarray, N: np.ndarray, cfg: ShadingConfig) -> np.ndarray:
    """3-bin azimuth median filter, then a count-weighted isotonic fit (loss non-increasing with
    elevation for every azimuth: an obstacle that blocks the sun at elevation e blocks it below e)."""
    import warnings
    sm = L.copy()
    for i in range(L.shape[0]):
        nb = L[[(i - 1) % L.shape[0], i, (i + 1) % L.shape[0]]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmedian(nb, 0)
        sm[i] = np.where(np.isfinite(L[i]), m, np.nan)
    out = np.full_like(L, np.nan)
    for i in range(L.shape[0]):
        f = np.isfinite(sm[i])
        if f.sum() >= 2:
            out[i, f] = _pav_nonincreasing(sm[i, f], np.maximum(N[i, f], 1))
        elif f.any():
            out[i, f] = sm[i, f]
    out = np.where(out < cfg.min_loss, 0.0, np.clip(out, 0, cfg.max_loss))
    return np.where(np.isfinite(L), out, np.nan)


def horizon_from_obstacles(obst: pd.DataFrame, az_step: float = 1.0) -> pd.DataFrame:
    """Obstacle rows (azimuth_deg, width_deg, distance_m, height_m) -> horizon profile per site."""
    rows = []
    key = "column" if "column" in obst else "site"
    for k, g in obst.groupby(key):
        az = np.arange(0, 360, az_step); hz = np.zeros_like(az)
        for r in g.itertuples():
            e = np.rad2deg(np.arctan2(float(r.height_m), max(float(r.distance_m), 0.1)))
            d = (az - float(r.azimuth_deg) + 180) % 360 - 180
            hz = np.where(np.abs(d) <= float(r.width_deg) / 2, np.maximum(hz, e), hz)
        rows += [{key: k, "azimuth_deg": a, "horizon_deg": h} for a, h in zip(az, hz)]
    return pd.DataFrame(rows)


def _geometric(hz: pd.DataFrame, cfg: ShadingConfig) -> np.ndarray:
    ae, ee = _edges(cfg)
    ac, ec = 0.5 * (ae[:-1] + ae[1:]), 0.5 * (ee[:-1] + ee[1:])
    h = np.interp(ac, hz["azimuth_deg"].to_numpy(float), hz["horizon_deg"].to_numpy(float), period=360)
    return (ec[None, :] < h[:, None]).astype(float)


def fit_shading(D: dict, C: Optional[dict] = None, cfg: Optional[ShadingConfig] = None,
                horizon: Optional[pd.DataFrame] = None, obstacles: Optional[pd.DataFrame] = None,
                verbose: bool = True) -> ShadingMask:
    """Estimate the shading masks on the training period (clean data only)."""
    cfg = cfg or ShadingConfig()
    df = D["df"]; pv = list(D["pv_cols"]); idx = df.index; b = pd.Timestamp(cfg.boundary)
    ae, ee = _edges(cfg)
    sun = sun_frame(D); el, az = sun["elev"].to_numpy(), sun["az"].to_numpy()
    kt = _kt(D, df)
    pexp = np.asarray(D["phys"]["p_expected"], float)
    act = np.ones((len(idx), len(pv)), bool)
    if C is not None and C.get("valid") is not None:
        act &= C["valid"].reindex(index=idx, columns=pv, fill_value=False).to_numpy(bool)
    elif D.get("observed") is not None:
        act &= D["observed"].reindex(index=idx, columns=pv, fill_value=True).to_numpy(bool)
    clear = np.asarray(idx < b) & (np.nan_to_num(kt) > cfg.kt_clear) & (df[IRR].to_numpy(float) > cfg.ghi_min) & (el > cfg.el_min)
    ia = np.clip(np.digitize(az, ae) - 1, 0, len(ae) - 2); ie = np.clip(np.digitize(np.clip(el, 0, 89.999), ee) - 1, 0, len(ee) - 2)
    geo = {}
    if obstacles is not None and len(obstacles):
        horizon = pd.concat([horizon, horizon_from_obstacles(obstacles)]) if horizon is not None else horizon_from_obstacles(obstacles)
    if horizon is not None and len(horizon):
        key = "column" if "column" in horizon else "site"
        for k, g in horizon.groupby(key):
            col = k if k in pv else next((c for c in pv if _short(c) == k), None)
            if col is not None:
                geo[col] = _geometric(g, cfg)
    loss, src, nh = {}, {}, {}
    for j, c in enumerate(pv):
        if c in geo:
            loss[c], src[c] = geo[c], "geometric"; nh[c] = 0
            continue
        P = df[c].to_numpy(float)
        m = clear & act[:, j] & np.isfinite(P) & (pexp[:, j] > 0) & (P >= 0)
        nh[c] = int(m.sum())
        if m.sum() < cfg.min_hours:
            loss[c] = np.full((len(ae) - 1, len(ee) - 1), np.nan); src[c] = "none"
            continue
        r = P[m] / pexp[m, j]
        X = pd.DataFrame({"a": ia[m], "e": ie[m], "r": r})
        g = X.groupby(["a", "e"])["r"].agg(["median", "size"])
        g = g[g["size"] >= cfg.min_n]
        L = np.full((len(ae) - 1, len(ee) - 1), np.nan); N = np.zeros_like(L)
        if len(g):
            hi = g[g.index.get_level_values("e") * cfg.el_step >= cfg.ref_el]
            if len(hi) < 3:
                top = g.index.get_level_values("e") >= np.quantile(g.index.get_level_values("e"), 0.75)
                hi = g[top]
            ref = float(np.median(hi["median"])) if len(hi) else float(np.median(g["median"]))
            for (a_, e_), row in g.iterrows():
                L[int(a_), int(e_)] = 1.0 - row["median"] / max(ref, 1e-6); N[int(a_), int(e_)] = row["size"]
            L = _monotone_smooth(L, N, cfg)
        loss[c], src[c] = L, "data-driven"
    M = ShadingMask(ae, ee, loss, src, nh, cfg)
    if verbose:
        t = M.table()
        _log(f"{(t['Mask'] != 'none').sum()} masks ({(t['Mask'] == 'geometric').sum()} geometric); "
             f"sites with shaded sun positions: {(t['Shaded positions (loss > 0)'] > 0).sum()}")
    return M


# ============================================================================ real-data check (no neural network)
def evaluate(D: dict, C: Optional[dict], S: ShadingMask, boundary: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Out-of-sample check on the test period (clear hours): spread of log(P/P_exp) and range of the
    hour-of-day bias with / without the shading correction P_exp * (1 - s_eff)."""
    boundary = boundary or S.cfg.boundary
    df = D["df"]; pv = list(D["pv_cols"]); idx = df.index; b = pd.Timestamp(boundary)
    ch = S.channels(D, pv)
    kt = _kt(D, df); sun = sun_frame(D)
    pexp = np.asarray(D["phys"]["p_expected"], float)
    act = C["valid"].reindex(index=idx, columns=pv, fill_value=False).to_numpy(bool) if C is not None and C.get("valid") is not None \
        else np.ones((len(idx), len(pv)), bool)
    rows = []
    for period, sel in (("training", np.asarray(idx < b)), ("test", np.asarray(idx >= b))):
        clear = sel & (np.nan_to_num(kt) > S.cfg.kt_clear) & (df[IRR].to_numpy(float) > S.cfg.ghi_min) & (sun["elev"].to_numpy() > S.cfg.el_min)
        for j, c in enumerate(pv):
            if S.source.get(c) == "none":
                continue
            P = df[c].to_numpy(float)
            m = clear & act[:, j] & np.isfinite(P) & (P > 0) & (pexp[:, j] > 0) & np.isfinite(ch["s_eff"][:, j])
            if m.sum() < 50:
                continue
            h = idx.hour.to_numpy()[m]
            for corr in (False, True):
                pe = pexp[m, j] * ((1 - ch["s_eff"][m, j]) if corr else 1.0)
                lr = np.log(P[m] / np.maximum(pe, 1e-6))
                mh = pd.Series(lr).groupby(h).median()
                rows.append(dict(period=period, column=c, corrected=corr, n=int(m.sum()),
                                 spread=float(1.4826 * np.median(np.abs(lr - np.median(lr)))),
                                 hour_bias_range=float(mh.max() - mh.min()) if len(mh) else np.nan))
    R = pd.DataFrame(rows)
    out = {"masks": S.table()}
    if len(R):
        A = R.pivot_table(index=["period", "column"], columns="corrected", values=["spread", "hour_bias_range"])
        A.columns = [f"{v}_{'corrected' if k else 'raw'}" for v, k in A.columns]
        A = A.reset_index()
        shaded = {c for c, L in S.loss.items() if np.isfinite(L).any() and np.nanmax(L) > 0}
        A["shaded site"] = A["column"].isin(shaded)
        summ = []
        for (p, sh), g in A.groupby(["period", "shaded site"]):
            summ.append({"Period": p, "Sites": "with shaded sun positions" if sh else "without", "N sites": len(g),
                         "Spread log(P/P_exp), raw": g["spread_raw"].median(),
                         "Spread log(P/P_exp), shading-corrected": g["spread_corrected"].median(),
                         "Hour-of-day bias range, raw": g["hour_bias_range_raw"].median(),
                         "Hour-of-day bias range, shading-corrected": g["hour_bias_range_corrected"].median(),
                         "Sites with lower test spread": int((g["spread_corrected"] < g["spread_raw"] - 1e-9).sum())})
        out["summary"] = pd.DataFrame(summ)
        A.insert(1, "Site", A.pop("column").map(_short))
        out["per_site"] = A.rename(columns={"period": "Period", "spread_raw": "Spread, raw", "spread_corrected": "Spread, corrected",
                                            "hour_bias_range_raw": "Hour bias range, raw",
                                            "hour_bias_range_corrected": "Hour bias range, corrected"})
    return out


TITLES = {"masks": "Section 16: shading masks (training period)",
          "summary": "Section 16: shading correction on clear hours (median over sites; test = out of sample)",
          "per_site": "Section 16: shading correction per site"}
