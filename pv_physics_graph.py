"""
Physically-informed spatio-temporal graph for PV fleet forecasting + anomaly detection.

Contents
--------
BrickPVModel          : reads the Brick/TTL metadata (with auto-repair of malformed statements)
PhysicsPrior          : POA transposition (HDKR / Hay-Davies) + Faiman cell temperature,
                        analytic dP/dGHI sensitivity
PVGraphBuilder        : builds nodes + three edge families (topology, peer-neighborhood, G -> P)
PIGraphDataset        : windowed tensors for joint forecast / reconstruction training
PhysicsInformedSTGNN  : dilated causal TCN  ->  physics-biased graph attention  ->  heads
PIGraphTrainer        : joint loss (forecast + reconstruction + physics residual), anomaly score

Literature basis is documented in REFERENCES at the bottom of this file.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # torch optional for pure graph construction
    torch = None

# ----------------------------------------------------------------------------------
# 1. Brick model reader
# ----------------------------------------------------------------------------------

BRICK = "https://brickschema.org/schema/Brick#"
EXT = "https://www.HKUST_PV_System.com/schema/BrickExtension#"


@dataclass
class PVSite:
    """One Brick :PV_Generation_System with the physical attributes we need."""
    uri: str
    name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    altitude: Optional[float] = None
    rated_kw: Optional[float] = None
    tilt_deg: Optional[float] = None
    azimuth_deg: Optional[float] = None          # 0 = north, 180 = south (pvlib convention)
    tilt_raw: Optional[str] = None
    azimuth_raw: Optional[str] = None
    location: Optional[str] = None
    contractor: Optional[str] = None
    brand: Optional[str] = None
    module: Optional[str] = None
    parts: List[str] = field(default_factory=list)   # inverters / strings
    connection_date: Optional[str] = None
    gamma_pdc: float = -0.0035                       # 1/K, temperature coefficient of Pmp
    first_year_degradation: Optional[float] = None
    linear_degradation: Optional[float] = None
    # --- equipment read from the hasPart chain  system -> inverter -> PV_Panel ---
    inverter_brand: Optional[str] = None
    inverter_model: Optional[str] = None
    n_inverters: int = 0
    panel_brand: Optional[str] = None
    panel_model: Optional[str] = None
    panel_type: Optional[str] = None
    n_panels: Optional[float] = None
    module_eff: Optional[float] = None
    panel_rated_w: Optional[float] = None
    # --- all sub-array orientations (pvlib convention, 0 = N, 180 = S) ---
    tilts_deg: List[float] = field(default_factory=list)
    azimuths_deg: List[float] = field(default_factory=list)
    multi_orientation: bool = False


class BrickPVModel:
    """Parses the PV Brick model and exposes site-level physical metadata.

    The provided TTL contains one malformed statement (a block terminated with '.'
    and then continued with further predicates).  `_repair_turtle` patches that
    class of error before handing the text to rdflib, so the file can be read as-is.
    """

    def __init__(self, ttl_path: str, verbose: bool = True):
        import rdflib

        self.rdflib = rdflib
        raw = open(ttl_path, encoding="utf-8").read()
        text, n_fixed = self._repair_turtle(raw)
        if verbose and n_fixed:
            print(f"[BrickPVModel] repaired {n_fixed} malformed Turtle statement(s)")
        self.graph = rdflib.Graph()
        self.graph.parse(data=text, format="turtle")
        self.B = rdflib.Namespace(BRICK)
        self.E = rdflib.Namespace(EXT)
        self.sites: Dict[str, PVSite] = self._read_sites()
        if verbose:
            print(f"[BrickPVModel] {len(self.graph)} triples, {len(self.sites)} PV systems")

    # -- turtle hygiene -------------------------------------------------------
    @staticmethod
    def _repair_turtle(text: str) -> Tuple[str, int]:
        lines = text.replace("\r\n", "\n").split("\n")
        pred = re.compile(r"^\s*(ext|brick|rdfs|skos|owl|sh):[\w\-]+\s")
        fixed = 0
        for i in range(len(lines) - 1):
            cur = lines[i].rstrip()
            if cur.endswith(".") and not cur.endswith("..") and pred.match(lines[i + 1]):
                lines[i] = cur[:-1].rstrip() + " ;"
                fixed += 1
        return "\n".join(lines), fixed

    # -- helpers --------------------------------------------------------------
    def _bval(self, node, prop="value"):
        for o in self.graph.objects(node, getattr(self.B, prop)):
            return o
        return None

    def _scalar(self, subj, ns, pred, prop="value", cast=float):
        for b in self.graph.objects(subj, getattr(ns, pred)):
            v = self._bval(b, prop)
            if v is None:
                continue
            try:
                return cast(str(v))
            except (TypeError, ValueError):
                return str(v)
        return None

    @staticmethod
    def parse_angle(raw: Optional[str], default: float) -> Optional[float]:
        """'10deg' -> 10 ; '0/11deg' -> 5.5 ; '90/-90deg' -> vector mean ; 'Mixed' -> default."""
        if raw is None:
            return default
        s = str(raw).strip().lower().replace("deg", "")
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]
        if not nums:
            return default                     # 'mixed' / unparseable
        if len(nums) == 1:
            return nums[0]
        # circular mean keeps East/West split arrays honest (90/-90 -> ~0 net offset)
        rad = np.deg2rad(nums)
        return float(np.rad2deg(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))

    @staticmethod
    def parse_angle_list(raw: Optional[str]) -> List[float]:
        """'0/11deg' -> [0, 11]; 'Mixed' / None -> []."""
        if raw is None:
            return []
        s = str(raw).strip().lower().replace("deg", "")
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]

    @staticmethod
    def brick_to_pvlib_azimuth(a: float) -> float:
        """The HKUST Brick extension defines azimuth with south = 0, east = -90,
        west = +90 (see ext:azimuth skos:definition in the TTL).  pvlib / the solar
        geometry below use north = 0, east = 90, south = 180, west = 270."""
        return float((180.0 + a) % 360.0)

    def _first_literal(self, node, pred_ns, pred, cast=str):
        for b in self.graph.objects(node, getattr(pred_ns, pred)):
            v = self._bval(b)
            if v is not None:
                try:
                    return cast(str(v))
                except (TypeError, ValueError):
                    return str(v)
        return None

    def _read_equipment(self, site: "PVSite", s) -> None:
        """Walk system -> inverter(s) -> PV_Panel and copy the equipment metadata."""
        inv_brand, inv_model, pan = [], [], []
        n_inv = 0
        for inv in self.graph.objects(s, self.B.hasPart):
            n_inv += 1
            b = self._first_literal(inv, self.E, "brand")
            m = self._first_literal(inv, self.E, "module")
            if b: inv_brand.append(b)
            if m: inv_model.append(m)
            for p in self.graph.objects(inv, self.B.hasPart):
                pan.append(dict(
                    brand=self._first_literal(p, self.E, "brand"),
                    model=self._first_literal(p, self.E, "module"),
                    ptype=self._first_literal(p, self.E, "specificType"),
                    n=self._first_literal(p, self.E, "number", float),
                    eff=self._first_literal(p, self.E, "ratedModuleConversionEfficiency", float),
                    w=self._first_literal(p, self.E, "ratedPowerOutput", float),
                    d1=self._first_literal(p, self.E, "panelDegradationFirstYear", float),
                    dl=self._first_literal(p, self.E, "panelLinearDegradation", float)))
        site.n_inverters = n_inv
        site.inverter_brand = inv_brand[0] if inv_brand else None
        site.inverter_model = inv_model[0] if inv_model else None
        if pan:
            first = lambda k: next((q[k] for q in pan if q[k] is not None), None)
            site.panel_brand, site.panel_model, site.panel_type = first("brand"), first("model"), first("ptype")
            ns = [q["n"] for q in pan if isinstance(q["n"], float)]
            site.n_panels = float(np.sum(ns)) if ns else None
            site.module_eff = first("eff")
            site.panel_rated_w = first("w")
            if site.first_year_degradation is None:
                site.first_year_degradation = first("d1")
            if site.linear_degradation is None:
                site.linear_degradation = first("dl")

    # -- main extraction ------------------------------------------------------
    def _read_sites(self) -> Dict[str, PVSite]:
        rdflib = self.rdflib
        out: Dict[str, PVSite] = {}
        for s in self.graph.subjects(rdflib.RDF.type, self.B.PV_Generation_System):
            name = str(s).split("#")[-1]
            site = PVSite(uri=str(s), name=name)
            for c in self.graph.objects(s, self.B.coordinates):
                lat, lon = self._bval(c, "latitude"), self._bval(c, "longitude")
                site.lat = float(lat) if lat is not None else None
                site.lon = float(lon) if lon is not None else None
            site.altitude = self._scalar(s, self.E, "altitude")
            site.rated_kw = self._scalar(s, self.E, "ratedPowerOutput")
            site.tilt_raw = self._scalar(s, self.E, "tiltAngle", cast=str)
            site.azimuth_raw = self._scalar(s, self.E, "azimuth", cast=str)
            site.location = self._scalar(s, self.B, "hasLocation", cast=str)
            site.contractor = self._scalar(s, self.E, "contractor", cast=str)
            site.brand = self._scalar(s, self.E, "brand", cast=str)
            site.module = self._scalar(s, self.E, "module", cast=str)
            site.connection_date = self._scalar(s, self.E, "connectionDate", cast=str)
            site.first_year_degradation = self._scalar(s, self.E, "panelDegradationFirstYear")
            site.linear_degradation = self._scalar(s, self.E, "panelLinearDegradation")
            site.parts = [str(p).split("#")[-1] for p in self.graph.objects(s, self.B.hasPart)]
            self._read_equipment(site, s)
            # Orientation.  Brick azimuth: south = 0, east = -90, west = +90.
            # Split arrays ('90/-90deg', '0/11deg') are kept as a list of sub-arrays and
            # the transposition factor is averaged over them (no circular mean: an
            # East/West array is NOT a north-facing array).  'Mixed' -> fleet default
            # (10 deg tilt, due south).
            tilts = self.parse_angle_list(site.tilt_raw) or [10.0]
            azs = [self.brick_to_pvlib_azimuth(a) for a in self.parse_angle_list(site.azimuth_raw)] or [180.0]
            if len(tilts) != len(azs):                      # pair every tilt with every azimuth
                pairs = [(t, a) for t in tilts for a in azs]
                tilts, azs = [p[0] for p in pairs], [p[1] for p in pairs]
            site.tilts_deg, site.azimuths_deg = tilts, azs
            site.multi_orientation = len(set(zip(tilts, azs))) > 1
            site.tilt_deg = float(np.mean(tilts))
            rad = np.deg2rad(azs)
            site.azimuth_deg = float(np.rad2deg(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360.0)
            out[name] = site
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(s) for s in self.sites.values()]).set_index("name")

    def __getstate__(self):                       # picklable (module handle dropped)
        st = self.__dict__.copy()
        st.pop("rdflib", None)
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        import rdflib
        self.rdflib = rdflib


# ----------------------------------------------------------------------------------
# 2. Physics: POA transposition, Faiman temperature, dP/dGHI
# ----------------------------------------------------------------------------------

class PhysicsPrior:
    """Irradiance -> power model and its analytic sensitivity.

    Power model (PVWatts / Sandia style, Dobos 2014; King et al. 2004):

        P = P_dc0 * (G_poa / G_ref) * [1 + gamma * (T_cell - 25)] * eta_sys

    Transposition (Hay-Davies / HDKR, Duffie & Beckman ch. 2):

        G_poa = G_b,n*cos(theta) + G_d*[A_i*Rb + (1-A_i)*(1+cos(beta))/2] + G*rho*(1-cos(beta))/2
        R_tr  = G_poa / GHI                      ("transposition factor")

    Cell temperature (Faiman 2008):

        T_cell = T_air + G_poa / (U0 + U1 * w)

    Sensitivity actually used as the G -> P edge weight (W per W/m^2):

        dP/dGHI = P_dc0 * eta * R_tr / G_ref
                  * [ 1 + gamma*(T_cell - 25) + gamma * G_poa / (U0 + U1*w) ]

    The bracket's last term is the thermal feedback: more irradiance heats the module,
    which partly cancels the linear gain.  Irradiance is split into beam/diffuse with
    the Erbs correlation when only GHI is measured, as in this dataset.
    """

    G_REF = 1000.0

    def __init__(self, u0: float = 25.0, u1: float = 6.84, albedo: float = 0.2,
                 eta_system: float = 0.86):
        self.u0, self.u1, self.albedo, self.eta_system = u0, u1, albedo, eta_system

    # -- solar geometry (NOAA low-precision algorithm) ------------------------
    @staticmethod
    def solar_position(times: pd.DatetimeIndex, lat: float, lon: float,
                       tz_offset_hours: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
        doy = times.dayofyear.to_numpy(float)
        hour = times.hour.to_numpy(float) + times.minute.to_numpy(float) / 60.0
        b = np.deg2rad(360.0 / 365.0 * (doy - 81.0))
        eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)      # minutes
        lstm = 15.0 * tz_offset_hours
        tc = 4.0 * (lon - lstm) + eot
        lst = hour + tc / 60.0
        hra = np.deg2rad(15.0 * (lst - 12.0))
        decl = np.deg2rad(23.45) * np.sin(b)
        phi = np.deg2rad(lat)
        cos_z = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hra)
        cos_z = np.clip(cos_z, -1, 1)
        zen = np.arccos(cos_z)
        az = np.arctan2(np.sin(hra),
                        np.cos(hra) * np.sin(phi) - np.tan(decl) * np.cos(phi))
        az = np.rad2deg(az) + 180.0                                          # 0 = north
        return np.rad2deg(zen), az

    @staticmethod
    def erbs_split(ghi: np.ndarray, zenith_deg: np.ndarray, doy: np.ndarray):
        """Erbs (1982) diffuse fraction from the clearness index."""
        cos_z = np.clip(np.cos(np.deg2rad(zenith_deg)), 1e-3, None)
        e0 = 1 + 0.033 * np.cos(np.deg2rad(360 * doy / 365.0))
        ghi_et = np.maximum(1367.0 * e0 * cos_z, 1e-6)
        kt = np.clip(ghi / ghi_et, 0, 1.2)
        df = np.where(kt <= 0.22, 1 - 0.09 * kt,
             np.where(kt <= 0.80,
                      0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
                      0.165))
        df = np.clip(df, 0.0, 1.0)
        dhi = ghi * df
        dni = np.where(cos_z > 1e-3, (ghi - dhi) / cos_z, 0.0)
        return np.clip(dni, 0, 1400), dhi, ghi_et

    def transposition_factor(self, ghi: np.ndarray, times: pd.DatetimeIndex,
                             lat: float, lon: float, tilt: float, azimuth: float,
                             tz_offset_hours: float = 8.0) -> np.ndarray:
        """HDKR POA / GHI ratio; returns 0 where the sun is down."""
        zen, saz = self.solar_position(times, lat, lon, tz_offset_hours)
        doy = times.dayofyear.to_numpy(float)
        dni, dhi, ghi_et = self.erbs_split(np.asarray(ghi, float), zen, doy)
        b, z = np.deg2rad(tilt), np.deg2rad(zen)
        cos_theta = np.clip(
            np.cos(z) * np.cos(b)
            + np.sin(z) * np.sin(b) * np.cos(np.deg2rad(saz - azimuth)), 0, 1)
        rb = cos_theta / np.clip(np.cos(z), 1e-3, None)
        ai = np.clip(dni / np.maximum(ghi_et, 1e-6), 0, 1)              # anisotropy index
        sky = dhi * (ai * rb + (1 - ai) * (1 + np.cos(b)) / 2.0)
        grd = np.asarray(ghi, float) * self.albedo * (1 - np.cos(b)) / 2.0
        poa = dni * cos_theta + sky + grd
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(np.asarray(ghi, float) > 5.0, poa / np.maximum(ghi, 1e-6), 0.0)
        return np.clip(np.nan_to_num(r), 0.0, 2.5)

    def cell_temperature(self, poa: np.ndarray, t_air: np.ndarray, wind: np.ndarray) -> np.ndarray:
        return t_air + poa / (self.u0 + self.u1 * np.maximum(wind, 0.0))

    def temperature_factor(self, t_cell: np.ndarray, gamma: float) -> np.ndarray:
        return 1.0 + gamma * (t_cell - 25.0)

    def expected_power(self, ghi, times, t_air, wind, site: PVSite,
                       tz_offset_hours: float = 8.0, age_years: np.ndarray | float = 0.0):
        """Clear-of-faults expected AC power in W, plus the intermediates.

        For split arrays the transposition factor is the (equal-capacity) mean over
        the sub-array orientations listed in the Brick model."""
        orients = list(zip(site.tilts_deg, site.azimuths_deg)) or [(site.tilt_deg, site.azimuth_deg)]
        r_tr = np.mean([self.transposition_factor(ghi, times, site.lat, site.lon, t, a, tz_offset_hours)
                        for t, a in orients], axis=0)
        poa = np.asarray(ghi, float) * r_tr
        t_cell = self.cell_temperature(poa, np.asarray(t_air, float), np.asarray(wind, float))
        f_t = self.temperature_factor(t_cell, site.gamma_pdc)
        p0 = (site.rated_kw or 0.0) * 1000.0
        deg = self._degradation(site, age_years)
        p_exp = p0 * (poa / self.G_REF) * f_t * self.eta_system * deg
        return np.clip(p_exp, 0, None), dict(r_tr=r_tr, poa=poa, t_cell=t_cell, f_t=f_t, deg=deg)

    @staticmethod
    def _degradation(site: PVSite, age_years):
        d1 = (site.first_year_degradation or 0.0) / 100.0
        dl = (site.linear_degradation or 0.0) / 100.0
        age = np.asarray(age_years, float)
        return np.clip(1.0 - d1 - dl * np.maximum(age - 1.0, 0.0), 0.5, 1.0)

    def dP_dGHI(self, ghi, times, t_air, wind, site: PVSite,
                tz_offset_hours: float = 8.0, age_years: np.ndarray | float = 0.0):
        """Analytic partial derivative used as the physical G -> PV edge weight."""
        p_exp, aux = self.expected_power(ghi, times, t_air, wind, site,
                                        tz_offset_hours, age_years)
        p0 = (site.rated_kw or 0.0) * 1000.0
        u = self.u0 + self.u1 * np.maximum(np.asarray(wind, float), 0.0)
        thermal = site.gamma_pdc * aux["poa"] / u          # d(f_T)/dG * G  term
        d = (p0 * self.eta_system * aux["deg"] * aux["r_tr"] / self.G_REF) \
            * (aux["f_t"] + thermal)
        return np.clip(d, 0.0, None), p_exp, aux


# ----------------------------------------------------------------------------------
# 3. Graph construction
# ----------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Commissioning / decommissioning and exploitation masks
# ------------------------------------------------------------------------------

def commissioning_frame(df: pd.DataFrame, brick: BrickPVModel,
                         power_cols: Sequence[str]) -> pd.DataFrame:
    """Derive per-plant commissioning/decommissioning dates from the DataFrame.

    The heuristic is simple and conservative:
    - commissioning_date from Brick if available;
    - otherwise, first day with sustained positive power on daylight hours;
    - decommissioning as the last day with sustained positive power, if it is
      earlier than the end of the record.

    The resulting frame has columns:
        commissioning_date, decommissioning_date, n_active_days.
    """
    idx = pd.DatetimeIndex(df.index)
    day = df.index.normalize()
    daylight = df["Irradiance (W/m2)"] > 200.0 if "Irradiance (W/m2)" in df.columns else None
    records = []

    for col in power_cols:
        name = col.replace("power(W)_", "")
        site = brick.sites.get(name)
        s = df[col].to_numpy(float)
        if daylight is not None:
            active = (s > 0.0) & daylight.to_numpy(bool)
        else:
            active = s > 0.0
        if active.any():
            days = day[active]
            first = days.min()
            last = days.max()
        else:
            first = pd.NaT
            last = pd.NaT
        # prefer Brick connection_date if present and parsable
        if site is not None and site.connection_date:
            try:
                cd = pd.to_datetime(site.connection_date).normalize()
                if pd.notna(first):
                    first = min(first, cd)
                else:
                    first = cd
            except Exception:
                pass
        records.append(dict(plant=col,
                            commissioning_date=first,
                            decommissioning_date=last,
                            n_active_days=int(len(set(days)) if active.any() else 0)))
    return pd.DataFrame(records).set_index("plant")


def exploitation_mask(df: pd.DataFrame, comm: pd.DataFrame) -> pd.DataFrame:
    """Causal half-open exploitation mask: commission <= t < decommission.

    A missing commission means never active; a missing decommission means active
    until the record ends. This differs deliberately from treating the last
    positive measurement as an inclusive observation.
    """
    ts = pd.DatetimeIndex(df.index)
    out = {}
    for plant, row in comm.iterrows():
        c, d = pd.to_datetime(row.get("commissioning_date")), pd.to_datetime(row.get("decommissioning_date"))
        if pd.isna(c):
            out[plant] = np.zeros(len(ts), bool)
            continue
        m = ts >= c
        if pd.notna(d):
            m &= ts < d
        out[plant] = np.asarray(m, bool)
    return pd.DataFrame(out, index=df.index)


def scenario_commissioning(df: pd.DataFrame, brick: BrickPVModel,
                           power_cols: Sequence[str], scenario: str = "dynamic_actual",
                           overrides: Optional[Dict[str, Tuple[object, object]]] = None,
                           min_positive_steps: int = 6,
                           irr_col: str = "Irradiance (W/m2)") -> pd.DataFrame:
    """Create static, observed-history, or simulated commission/decommission dates.

    Observed dates use the first/last *sustained* daylight-positive run. Overrides
    are explicit scenario parameters and therefore do not claim historical truth.
    """
    if scenario not in {"static_full", "dynamic_actual", "dynamic_simulated"}:
        raise ValueError(f"unknown scenario: {scenario}")
    start, stop = df.index[0], df.index[-1] + (df.index[1] - df.index[0] if len(df) > 1 else pd.Timedelta(seconds=1))
    if scenario == "static_full":
        return pd.DataFrame({"commissioning_date": start, "decommissioning_date": stop,
                             "n_active_steps": len(df), "source": "static_full"}, index=list(power_cols))
    daylight = df[irr_col].to_numpy(float) > 50 if irr_col in df else np.ones(len(df), bool)
    rows = []
    for col in power_cols:
        positive = (df[col].to_numpy(float) > 0) & daylight
        run = np.convolve(positive.astype(int), np.ones(min_positive_steps, int), mode="same") >= min_positive_steps
        pos = np.flatnonzero(run)
        c = df.index[pos[0]] if len(pos) else pd.NaT
        d = (df.index[pos[-1]] + (df.index[1] - df.index[0])) if len(pos) and len(df) > 1 else pd.NaT
        site = brick.sites.get(col.replace("power(W)_", ""))
        if site is not None and site.connection_date:
            try:
                bc = pd.Timestamp(site.connection_date)
                c = min(c, bc) if pd.notna(c) else bc
            except (TypeError, ValueError):
                pass
        if scenario == "dynamic_simulated":
            if not overrides or col not in overrides:
                raise ValueError(f"dynamic_simulated requires override for {col}")
            c, d = map(pd.Timestamp, overrides[col])
        rows.append((col, c, d, int(positive.sum()), scenario))
    return pd.DataFrame(rows, columns=["plant", "commissioning_date", "decommissioning_date",
                                       "n_active_steps", "source"]).set_index("plant")


def node_activity_mask(df: pd.DataFrame, graph: "PVGraph", comm: pd.DataFrame) -> pd.DataFrame:
    """Time x node mask; weather/exogenous nodes are permanently active."""
    pv = exploitation_mask(df, comm).reindex(columns=graph.meta["pv_cols"], fill_value=False)
    out = pd.DataFrame(True, index=df.index, columns=graph.node_names)
    out.loc[:, graph.meta["pv_cols"]] = pv.to_numpy(bool)
    return out


@dataclass
class PVGraph:
    node_names: List[str]
    node_type: np.ndarray                 # 0 = PV, 1 = exogenous/weather
    edge_index: np.ndarray                # (2, E) source -> target
    edge_type: np.ndarray                 # 0 topology, 1 peer, 2 G->P
    edge_weight: np.ndarray               # normalised prior in (0, 1]
    edge_phys_logit: np.ndarray           # log-prior added to the attention logit
    site_index: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)


class PVGraphBuilder:
    """Builds the physically-informed spatio-temporal graph.

    Edge families
    -------------
    topology (type 0) : from the Brick model — sites sharing a building/location,
                        an inverter vendor, or a parent asset are electrically and
                        thermally coupled (shading, feeder, curtailment).
    peer     (type 1) : k-nearest geographic neighbours (haversine) reinforced by
                        clear-sky-normalised power correlation, i.e. the classic
                        peer-comparison baseline for PV fault detection.
    G -> P   (type 2) : one directed edge from the irradiance node to every PV node,
                        weighted by the median daytime dP/dGHI of PhysicsPrior.
                        The attention logit receives log(w) additively, so the
                        network can rescale but not discard the physics.
    """

    def __init__(self, brick: BrickPVModel, prior: Optional[PhysicsPrior] = None,
                 k_peers: int = 6, tz_offset_hours: float = 8.0):
        self.brick = brick
        self.prior = prior or PhysicsPrior()
        self.k_peers = k_peers
        self.tz = tz_offset_hours

    # -- column <-> Brick name alignment -------------------------------------
    @staticmethod
    def _norm(s: str) -> str:
        s = s.replace("power(W)_", "")
        s = re.sub(r"[^a-z0-9]", "", s.lower())
        return (s.replace("centre", "center")
                 .replace("sports", "sport")
                 .replace("phrase", "phase"))

    def align_columns(self, power_cols: Sequence[str]) -> Dict[str, str]:
        lut = {self._norm(n): n for n in self.brick.sites}
        mapping, missing = {}, []
        for c in power_cols:
            key = self._norm(c)
            hit = lut.get(key)
            if hit is None:                                   # substring fallback
                cand = [v for k, v in lut.items() if k.startswith(key) or key.startswith(k)]
                hit = cand[0] if len(cand) == 1 else None
            if hit:
                mapping[c] = hit
            else:
                missing.append(c)
        if missing:
            print(f"[PVGraphBuilder] no Brick match for {len(missing)} column(s): {missing}")
        return mapping

    @staticmethod
    def _haversine_km(a: PVSite, b: PVSite) -> float:
        if None in (a.lat, a.lon, b.lat, b.lon):
            return np.inf
        p1, p2 = math.radians(a.lat), math.radians(b.lat)
        dp, dl = p2 - p1, math.radians(b.lon - a.lon)
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 6371.0 * 2 * math.asin(math.sqrt(h))

    # -- main -----------------------------------------------------------------
    def build(self, df: pd.DataFrame, power_cols: Sequence[str],
              irr_col: str = "Irradiance (W/m2)",
              temp_col: str = "Temp (Degree Celsius)",
              wind_col: str = "wind_speed",
              exo_cols: Sequence[str] = (),
              peer_fit_end: Optional[object] = None,
              observed: Optional[pd.DataFrame] = None,
              peer_prior: Optional[pd.DataFrame] = None,
              extra_edges: Optional[Sequence[Tuple[str, str, float, int]]] = None) -> PVGraph:
        """Build the supergraph.

        observed    : (T x pv) validity mask; invalid values are excluded from the
                      peer correlation (no ffill/zero artefacts in the affinity).
        peer_prior  : (pv x pv) multiplicative prior for peer edges, e.g. from
                      technical clusters (1 inside the cluster, <1 across clusters).
        extra_edges : additional (src_col, dst_col, weight, edge_type) tuples, e.g.
                      behavioural peer edges (edge_type 3).
        """
        times = pd.DatetimeIndex(df.index)
        mapping = self.align_columns(power_cols)
        pv_cols = [c for c in power_cols if c in mapping]

        node_names = list(pv_cols) + [irr_col] + list(exo_cols)
        node_type = np.array([0] * len(pv_cols) + [1] * (1 + len(exo_cols)))
        idx = {n: i for i, n in enumerate(node_names)}
        g_idx = idx[irr_col]

        ghi = df[irr_col].to_numpy(float)
        t_air = df[temp_col].to_numpy(float)
        wind = df[wind_col].to_numpy(float)
        age = (times - times[0]).days.to_numpy(float) / 365.25

        src, dst, etype, w = [], [], [], []
        sens_summary, phys = {}, {}

        # --- G -> P edges: the physical sensitivity ---------------------------
        for c in pv_cols:
            site = self.brick.sites[mapping[c]]
            d, p_exp, aux = self.prior.dP_dGHI(ghi, times, t_air, wind, site, self.tz, age)
            day = ghi > 50.0
            s = float(np.median(d[day])) if day.any() else 0.0
            sens_summary[c] = s
            phys[c] = dict(sensitivity_W_per_Wm2=s,
                           median_R_tr=float(np.median(aux["r_tr"][day])) if day.any() else 0.0,
                           median_f_T=float(np.median(aux["f_t"][day])) if day.any() else 1.0,
                           rated_kw=site.rated_kw, tilt=site.tilt_deg, azimuth=site.azimuth_deg)
            src.append(g_idx); dst.append(idx[c]); etype.append(2); w.append(max(s, 1e-6))

        # --- topology edges from Brick ---------------------------------------
        seen = set()
        for i, ci in enumerate(pv_cols):
            si = self.brick.sites[mapping[ci]]
            for cj in pv_cols[i + 1:]:
                sj = self.brick.sites[mapping[cj]]
                score = 0.0
                if si.location and si.location == sj.location:
                    score += 1.0
                if si.contractor and si.contractor == sj.contractor:
                    score += 0.3
                pre = lambda n: re.match(r"[A-Za-z]+", n).group(0) if re.match(r"[A-Za-z]+", n) else n
                if pre(si.name) == pre(sj.name):
                    score += 0.3                     # same asset family / feeder group
                if set(si.parts) & set(sj.parts):
                    score += 1.0                     # shared inverter
                if score > 0 and (ci, cj) not in seen:
                    seen.add((ci, cj))
                    for a, b in ((ci, cj), (cj, ci)):
                        src.append(idx[a]); dst.append(idx[b]); etype.append(0); w.append(score)

        # --- peer-neighborhood edges (fit on past/reference data only) --------
        peer_rows = np.ones(len(df), dtype=bool)
        if peer_fit_end is not None:
            peer_rows = times <= pd.Timestamp(peer_fit_end)
            if not peer_rows.any():
                raise ValueError("peer_fit_end precedes all observations")
        P = df[pv_cols].to_numpy(float)[peer_rows]
        if observed is not None:
            obs = observed.reindex(index=df.index, columns=pv_cols, fill_value=False).to_numpy(bool)[peer_rows]
            P = np.where(obs, P, np.nan)
        peer_ghi = ghi[peer_rows]
        denom = np.maximum(peer_ghi, 1.0)[:, None]
        pr = np.where(peer_ghi[:, None] > 50, P / denom, np.nan)          # performance ratio proxy
        corr = pd.DataFrame(pr, columns=pv_cols).corr().to_numpy()
        corr = np.nan_to_num(corr, nan=0.0)
        n = len(pv_cols)
        dist = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i, j] = self._haversine_km(self.brick.sites[mapping[pv_cols[i]]],
                                                    self.brick.sites[mapping[pv_cols[j]]])
        scale = np.median(dist[np.isfinite(dist) & (dist > 0)]) or 1.0
        aff = np.exp(-dist / scale) * np.clip(corr, 0, 1)
        if peer_prior is not None:
            pp = peer_prior.reindex(index=pv_cols, columns=pv_cols).fillna(1.0).to_numpy(float)
            aff = aff * pp
        np.fill_diagonal(aff, 0.0)
        for i in range(n):
            for j in np.argsort(-aff[i])[: self.k_peers]:
                if aff[i, j] <= 0:
                    continue
                src.append(idx[pv_cols[j]]); dst.append(idx[pv_cols[i]])
                etype.append(1); w.append(float(aff[i, j]))

        # --- exogenous weather -> PV (learnable, no physical prior) -----------
        for e in exo_cols:
            for c in pv_cols:
                src.append(idx[e]); dst.append(idx[c]); etype.append(1); w.append(0.1)

        # --- additional (e.g. behavioural) edges --------------------------------
        for a, b, wt, t in (extra_edges or []):
            if a in idx and b in idx and a != b:
                src.append(idx[a]); dst.append(idx[b]); etype.append(int(t)); w.append(float(wt))

        # --- self loops -------------------------------------------------------
        for nm in node_names:
            src.append(idx[nm]); dst.append(idx[nm]); etype.append(0); w.append(1.0)

        edge_index = np.vstack([src, dst]).astype(np.int64)
        etype = np.asarray(etype)
        w = np.asarray(w, float)

        # normalise each family to (0, 1] so log-priors are comparable in scale
        wn = w.copy()
        for t in np.unique(etype):
            m = etype == t
            if m.any():
                wn[m] = w[m] / max(w[m].max(), 1e-9)
        wn = np.clip(wn, 1e-6, None)

        return PVGraph(node_names=node_names, node_type=node_type, edge_index=edge_index,
                       edge_type=etype, edge_weight=wn, edge_phys_logit=np.log(wn),
                       site_index={c: idx[c] for c in pv_cols},
                       meta=dict(mapping=mapping, pv_cols=pv_cols, irr_col=irr_col,
                                 exo_cols=list(exo_cols), physics=phys,
                                 sensitivity=sens_summary, n_pv=len(pv_cols),
                                 peer_fit_end=None if peer_fit_end is None else str(pd.Timestamp(peer_fit_end))))

    # -- per-timestep physics tensors used by the loss ------------------------
    def physics_tensors(self, df: pd.DataFrame, graph: PVGraph,
                        irr_col="Irradiance (W/m2)", temp_col="Temp (Degree Celsius)",
                        wind_col="wind_speed") -> Dict[str, np.ndarray]:
        times = pd.DatetimeIndex(df.index)
        ghi = df[irr_col].to_numpy(float)
        t_air = df[temp_col].to_numpy(float)
        wind = df[wind_col].to_numpy(float)
        age = (times - times[0]).days.to_numpy(float) / 365.25
        cols = graph.meta["pv_cols"]
        p_exp = np.zeros((len(df), len(cols)))
        sens = np.zeros_like(p_exp)
        for j, c in enumerate(cols):
            site = self.brick.sites[graph.meta["mapping"][c]]
            d, pe, _ = self.prior.dP_dGHI(ghi, times, t_air, wind, site, self.tz, age)
            p_exp[:, j] = pe
            sens[:, j] = d
        return dict(p_expected=p_exp, dP_dGHI=sens, ghi=ghi)


# ----------------------------------------------------------------------------------
# 4. Dataset
# ----------------------------------------------------------------------------------

class PIGraphDataset:
    """Sliding windows: X (N, F, T_in) -> forecast horizon H and reconstruction of X."""

    def __init__(self, df: pd.DataFrame, graph: PVGraph, phys: Dict[str, np.ndarray],
                 feature_cols: Optional[Sequence[str]] = None,
                 window: int = 48, horizon: int = 1, stride: int = 1,
                 activity_mask: Optional[pd.DataFrame] = None,
                 norm_fit_end: Optional[object] = None,
                 observed: Optional[pd.DataFrame] = None,
                 context: Optional[np.ndarray] = None):
        """
        observed : (T x node) validity of each measured value (see infer_filled_mask);
                   unobserved values are zeroed in the input and removed from every
                   loss / score mask, so filled gaps never act as targets.
        context  : (T x K) exogenous context (e.g. one-hot weather regime); appended
                   to every node as K extra channels.
        """
        self.graph, self.window, self.horizon = graph, window, horizon
        names = graph.node_names
        self.signal = df[names].to_numpy(np.float32)                      # (T, N)
        if observed is None:
            self.observed = np.ones((len(df), len(names)), bool)
        else:
            self.observed = observed.reindex(index=df.index, columns=names, fill_value=True).to_numpy(bool).copy()
        if activity_mask is None:
            self.activity = np.ones((len(df), len(names)), dtype=bool)
        else:
            self.activity = activity_mask.reindex(index=df.index, columns=names, fill_value=False).to_numpy(bool).copy()
        self.activity[:, graph.node_type == 1] = True
        fit_rows = np.ones(len(df), bool) if norm_fit_end is None else (df.index <= pd.Timestamp(norm_fit_end))
        if not np.asarray(fit_rows).any():
            raise ValueError("norm_fit_end precedes all observations")
        valid = self.observed & self.activity
        fit_signal = np.where(valid, self.signal, np.nan)[np.asarray(fit_rows)]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.mu = np.nanmean(fit_signal, axis=0, keepdims=True)
                self.sd = np.nanstd(fit_signal, axis=0, keepdims=True)
        # nodes never observed in the reference period (new plants): fall back to
        # the whole active history - they are only used after commissioning anyway
        miss = ~np.isfinite(self.mu[0]) | ~np.isfinite(self.sd[0]) | (self.sd[0] < 1e-6)
        if miss.any():
            allv = np.where(valid, self.signal, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.mu[0, miss] = np.nan_to_num(np.nanmean(allv[:, miss], 0))
                self.sd[0, miss] = np.nan_to_num(np.nanstd(allv[:, miss], 0), nan=1.0)
        self.mu = np.nan_to_num(self.mu).astype(np.float32)
        self.sd = (np.nan_to_num(self.sd, nan=1.0) + 1e-6).astype(np.float32)
        self.z = np.nan_to_num((self.signal - self.mu) / self.sd).astype(np.float32)
        tod = df.index.hour.to_numpy(np.float32) / 24.0
        doy = df.index.dayofyear.to_numpy(np.float32) / 366.0
        tf = [np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
              np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)]
        self.time_feat = np.stack(tf, 1).astype(np.float32)
        self.context = None if context is None else np.asarray(context, np.float32)
        self.n_ctx = 0 if context is None else self.context.shape[1]
        self.in_ch = 1 + 4 + self.n_ctx
        self.p_expected = phys["p_expected"].astype(np.float32)
        # per-site normaliser (installed W from Brick) so every plant contributes
        # comparably to the physics residual regardless of its size
        rated = []
        for c in graph.meta["pv_cols"]:
            kw = graph.meta["physics"][c].get("rated_kw") or 0.0
            rated.append(max(kw * 1000.0, 1e3))
        self.p_scale = np.asarray(rated, np.float32)[None, :]          # (1, n_pv)
        self.idx = np.arange(0, len(df) - window - horizon + 1, stride)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t0 = self.idx[i]
        t1, t2 = t0 + self.window, t0 + self.window + self.horizon
        x = self.z[t0:t1].T[:, None, :]
        feats = self.time_feat[t0:t1]
        if self.context is not None:
            feats = np.concatenate([feats, self.context[t0:t1]], 1)
        tf = np.broadcast_to(feats.T[None], (x.shape[0], feats.shape[1], self.window))
        x = np.concatenate([x, tf], 1).astype(np.float32)
        obs_in = self.observed[t0:t1].T
        obs_out = self.observed[t1:t2].T
        input_mask = (self.activity[t0:t1].T & obs_in).astype(np.float32)
        recon_mask = input_mask.copy()
        target_mask = (self.activity[t1:t2].T & obs_out).astype(np.float32)
        full_history = self.activity[t0:t1].all(0)
        n_pv = int(self.graph.meta["n_pv"])
        target_mask[:n_pv] *= full_history[:n_pv, None]
        recon_mask[:n_pv] *= full_history[:n_pv, None]
        x[:, 0, :] *= input_mask
        return dict(x=x,
                    y=self.z[t1:t2].T.astype(np.float32),
                    recon_target=self.z[t0:t1].T.astype(np.float32),
                    p_meas=self.signal[t1:t2].T.astype(np.float32),
                    p_exp=self.p_expected[t1:t2].T.astype(np.float32),
                    node_mask=recon_mask,
                    target_mask=target_mask,
                    edge_mask=self.activity[t1:t2].all(0).astype(np.float32))

    def torch_loader(self, batch_size=32, shuffle=True, **kw):
        from torch.utils.data import DataLoader, Dataset

        ds = self

        class _W(Dataset):
            def __len__(self): return len(ds)
            def __getitem__(self, i):
                return {k: torch.from_numpy(np.ascontiguousarray(v))
                        for k, v in ds[i].items()}
        return DataLoader(_W(), batch_size=batch_size, shuffle=shuffle, **kw)


# ----------------------------------------------------------------------------------
# 5. Model
# ----------------------------------------------------------------------------------

if torch is not None:

    class DilatedCausalTCN(nn.Module):
        """Gated dilated causal convolutions (WaveNet / Graph WaveNet style)."""

        def __init__(self, in_ch: int, hidden: int, layers: int = 4, kernel: int = 2):
            super().__init__()
            self.inp = nn.Conv2d(in_ch, hidden, 1)
            self.filt, self.gate, self.res, self.skip, self.bn = (nn.ModuleList() for _ in range(5))
            for l in range(layers):
                d = 2 ** l
                self.filt.append(nn.Conv2d(hidden, hidden, (1, kernel), dilation=(1, d)))
                self.gate.append(nn.Conv2d(hidden, hidden, (1, kernel), dilation=(1, d)))
                self.res.append(nn.Conv2d(hidden, hidden, 1))
                self.skip.append(nn.Conv2d(hidden, hidden, 1))
                self.bn.append(nn.BatchNorm2d(hidden))
            self.kernel, self.layers = kernel, layers

        def forward(self, x):                       # x: (B, C, N, T)
            h = self.inp(x)
            skip = 0.0
            for i in range(self.layers):
                pad = (self.kernel - 1) * 2 ** i
                hp = F.pad(h, (pad, 0))             # left pad only => causal
                z = torch.tanh(self.filt[i](hp)) * torch.sigmoid(self.gate[i](hp))
                skip = skip + self.skip[i](z)
                h = self.bn[i](self.res[i](z) + h)
            return F.relu(skip), h                  # (B, H, N, T)

    class PhysicsBiasedGAT(nn.Module):
        """Masked graph attention whose logit gets log(physical prior) added.

        alpha_ij = softmax_j( a^T [W h_i || W h_j] / sqrt(d) + lambda_r * log w_ij + b_r )

        With lambda_r initialised at 1 the layer starts as the physics prior; gradients
        may rescale it per edge family but the log term is never removable.
        """

        def __init__(self, dim: int, heads: int = 4, n_types: int = 3, dropout: float = 0.1):
            super().__init__()
            assert dim % heads == 0
            self.h, self.dk = heads, dim // heads
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
            self.out = nn.Linear(dim, dim)
            self.lam = nn.Parameter(torch.ones(n_types, heads))     # prior strength
            self.bias = nn.Parameter(torch.zeros(n_types, heads))
            self.drop = nn.Dropout(dropout)

        def forward(self, h, edge_index, phys_logit, edge_type, node_mask=None):
            # h: (B, N, D)
            B, N, D = h.shape
            s, d = edge_index[0], edge_index[1]
            q = self.q(h).view(B, N, self.h, self.dk)
            k = self.k(h).view(B, N, self.h, self.dk)
            v = self.v(h).view(B, N, self.h, self.dk)
            logit = (q[:, d] * k[:, s]).sum(-1) / math.sqrt(self.dk)          # (B, E, heads)
            prior = phys_logit[None, :, None] * F.softplus(self.lam)[edge_type][None]
            logit = logit + prior + self.bias[edge_type][None]
            if node_mask is not None:
                active_edge = node_mask[:, s].bool() & node_mask[:, d].bool()
                logit = logit.masked_fill(~active_edge[..., None], -torch.inf)
            # segment softmax over incoming edges of each destination node
            m = torch.full((B, N, self.h), -1e30, device=h.device)
            m = m.index_reduce(1, d, logit, "amax", include_self=True)
            e = torch.exp(logit - m[:, d])
            e = torch.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
            den = torch.zeros(B, N, self.h, device=h.device).index_add(1, d, e) + 1e-9
            a = self.drop(e / den[:, d])
            msg = a.unsqueeze(-1) * v[:, s]
            agg = torch.zeros(B, N, self.h, self.dk, device=h.device).index_add(1, d, msg)
            return self.out(agg.reshape(B, N, D)), a

    class PhysicsInformedSTGNN(nn.Module):
        """Temporal TCN -> physics-biased graph attention -> forecast, reconstruction,
        and a per-node loss coefficient eta in (0, 1] used by the physics loss."""

        def __init__(self, graph: PVGraph, in_ch: int = 5, hidden: int = 64,
                     tcn_layers: int = 4, gat_layers: int = 2, heads: int = 4,
                     window: int = 48, horizon: int = 1):
            super().__init__()
            self.graph, self.window, self.horizon = graph, window, horizon
            self.tcn = DilatedCausalTCN(in_ch, hidden, tcn_layers)
            n_types = int(np.max(graph.edge_type)) + 1 if len(graph.edge_type) else 3
            self.gats = nn.ModuleList([PhysicsBiasedGAT(hidden, heads, n_types=max(n_types, 3))
                                       for _ in range(gat_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(gat_layers)])
            self.node_emb = nn.Parameter(torch.randn(len(graph.node_names), hidden) * 0.02)
            self.fc_fore = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                         nn.Linear(hidden, horizon))
            self.fc_rec = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, window))
            self.fc_eta = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                        nn.Linear(hidden // 2, horizon))
            self.register_buffer("edge_index", torch.as_tensor(graph.edge_index))
            self.register_buffer("edge_type", torch.as_tensor(graph.edge_type, dtype=torch.long))
            self.register_buffer("phys_logit",
                                 torch.as_tensor(graph.edge_phys_logit, dtype=torch.float32))
            self.register_buffer("is_pv",
                                 torch.as_tensor(graph.node_type == 0, dtype=torch.bool))

        def forward(self, x, node_mask=None):       # x: (B, N, C, T)
            x = x.permute(0, 2, 1, 3)               # (B, C, N, T)
            skip, _ = self.tcn(x)
            h = skip[..., -1].permute(0, 2, 1)      # (B, N, hidden) causal summary
            h = h + self.node_emb[None]
            attn = None
            for gat, ln in zip(self.gats, self.norms):
                z, attn = gat(h, self.edge_index, self.phys_logit, self.edge_type, node_mask=node_mask)
                h = ln(h + F.relu(z))
            return dict(forecast=self.fc_fore(h),               # (B, N, H)
                        recon=self.fc_rec(h),                   # (B, N, T)
                        eta=torch.sigmoid(self.fc_eta(h)),      # (B, N, H) loss coefficient
                        attn=attn, embed=h)

    class PIGraphTrainer:
        """Joint forecast + reconstruction + physics-residual training and scoring.

        L = L_forecast + a_r * L_recon + a_p * L_physics
        L_physics = mean( ( eta_hat * P_expected - P_measured ) / P_scale )^2

        eta_hat is the network's predicted loss coefficient (soiling, shading, clipping,
        derate).  The term forces the physical power model and the learned loss factor to
        jointly reproduce the measurement, so the graph cannot explain power away with
        purely statistical correlations.
        """

        def __init__(self, model, dataset: PIGraphDataset, lr=1e-3,
                     alpha_recon=1.0, alpha_phys=0.5, device=None):
            self.m = model
            self.ds = dataset
            self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.m.to(self.dev)
            self.opt = torch.optim.Adam(self.m.parameters(), lr=lr, weight_decay=1e-5)
            self.ar, self.ap = alpha_recon, alpha_phys
            self.n_pv = int(dataset.graph.meta["n_pv"])
            self.p_scale = torch.as_tensor(dataset.p_scale).to(self.dev).unsqueeze(-1)
            self.err_mu = None
            self.err_sd = None

        def _losses(self, b):
            b = {k: v.to(self.dev) for k, v in b.items()}
            o = self.m(b["x"], node_mask=b.get("edge_mask"))
            tm, hm = b.get("target_mask", torch.ones_like(b["y"])), b.get("node_mask", torch.ones_like(b["recon_target"]))
            lf = F.smooth_l1_loss(o["forecast"], b["y"], reduction="none")
            lr = F.smooth_l1_loss(o["recon"], b["recon_target"], reduction="none")
            l_f = (lf * tm).sum() / tm.sum().clamp_min(1)
            l_r = (lr * hm).sum() / hm.sum().clamp_min(1)
            eta = o["eta"][:, : self.n_pv]
            pm = tm[:, :self.n_pv]
            resid = (eta * b["p_exp"] - b["p_meas"][:, : self.n_pv]) / self.p_scale
            l_p = ((resid ** 2) * pm).sum() / pm.sum().clamp_min(1) * 100.0          # keep the term on par with the others
            return l_f + self.ar * l_r + self.ap * l_p, dict(
                forecast=l_f.item(), recon=l_r.item(), physics=l_p.item()), o

        def fit(self, loader, epochs=10, log_every=1):
            hist = []
            for ep in range(1, epochs + 1):
                self.m.train()
                acc = {}
                for b in loader:
                    self.opt.zero_grad()
                    loss, parts, _ = self._losses(b)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.m.parameters(), 5.0)
                    self.opt.step()
                    for k, v in parts.items():
                        acc[k] = acc.get(k, 0.0) + v
                acc = {k: v / max(len(loader), 1) for k, v in acc.items()}
                hist.append(acc)
                if ep % log_every == 0:
                    print(f"epoch {ep:3d} | " + " ".join(f"{k}={v:.5f}" for k, v in acc.items()))
            return hist

        @torch.no_grad()
        def calibrate(self, loader):
            """Fit per-node mean/std of the reconstruction error on clean data."""
            self.m.eval()
            s, ss, n = 0.0, 0.0, 0
            for b in loader:
                b = {k: v.to(self.dev) for k, v in b.items()}
                o = self.m(b["x"])
                e = (o["recon"] - b["recon_target"]).abs().mean(-1)        # (B, N)
                s = s + e.sum(0); ss = ss + (e ** 2).sum(0); n += e.shape[0]
            self.err_mu = s / n
            self.err_sd = ((ss / n - self.err_mu ** 2).clamp_min(1e-12)).sqrt()
            return self.err_mu.cpu().numpy(), self.err_sd.cpu().numpy()

        @torch.no_grad()
        def anomaly_score(self, loader, aggregate="max"):
            """Normalised reconstruction error per node, plus fleet score."""
            self.m.eval()
            if self.err_mu is None:
                self.calibrate(loader)
            out, fore, phys = [], [], []
            for b in loader:
                b = {k: v.to(self.dev) for k, v in b.items()}
                o = self.m(b["x"])
                e = (o["recon"] - b["recon_target"]).abs().mean(-1)
                out.append(((e - self.err_mu) / self.err_sd).cpu().numpy())
                fore.append((o["forecast"] - b["y"]).abs().mean(-1).cpu().numpy())
                eta = o["eta"][:, : self.n_pv]
                phys.append((((eta * b["p_exp"] - b["p_meas"][:, : self.n_pv])
                              / self.p_scale).abs().mean(-1)).cpu().numpy())
            node = np.concatenate(out)                                    # (S, N)
            agg = node.max(1) if aggregate == "max" else node.mean(1)
            return dict(node_score=node, fleet_score=agg,
                        forecast_error=np.concatenate(fore),
                        physics_residual=np.concatenate(phys),
                        node_names=self.ds.graph.node_names)

else:  # pragma: no cover
    DilatedCausalTCN = PhysicsBiasedGAT = PhysicsInformedSTGNN = PIGraphTrainer = None


# ----------------------------------------------------------------------------------
# 6. One-call convenience wrapper
# ----------------------------------------------------------------------------------

DEFAULT_EXO = ("Temp (Degree Celsius)", "RH (%)", "wind_speed", "SLP (hPa)", "Vis (km)")


def _to_time_indexed(df: pd.DataFrame, time_col: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    if time_col is None and not isinstance(out.index, pd.DatetimeIndex):
        cand = [c for c in out.columns
                if str(c).lower() in ("timestamp", "time", "datetime", "date", "index")
                or str(c).startswith("Unnamed")]
        time_col = cand[0] if cand else out.columns[0]
    if time_col is not None:
        out = out.set_index(pd.DatetimeIndex(pd.to_datetime(out[time_col]))) \
                 .drop(columns=[time_col])
    else:
        out.index = pd.DatetimeIndex(out.index)
    out.index.name = "timestamp"
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out.apply(pd.to_numeric, errors="coerce")


def prepare_dataframe(df: pd.DataFrame, time_col: Optional[str] = None,
                      return_mask: bool = False, power_fill: str = "zero"):
    """Make an already-loaded frame usable: DatetimeIndex, sorted, numeric, gap-filled.

    IMPORTANT (data-availability fix).  The availability mask ``finite before any
    filling`` is computed FIRST and returned when ``return_mask=True``; afterwards
    weather channels are forward-filled and power channels are filled with 0 only
    so that tensors are finite.  Without the mask a real zero generation cannot be
    told apart from a gap, which biases coverage, PR and degradation estimates.

    The timestamp may be the index, a named column ('timestamp'/'time'/'datetime'/'date'),
    or the first unnamed column (what pd.read_excel produces for this file).
    """
    out = _to_time_indexed(df, time_col)
    avail = out.notna()                                   # <- preserved BEFORE filling
    pcols = [c for c in out.columns if str(c).startswith("power(W)_")]
    wcols = [c for c in out.columns if c not in pcols]
    out[wcols] = out[wcols].ffill().bfill()
    if power_fill == "ffill":                             # legacy behaviour, not recommended
        out[pcols] = out[pcols].ffill()
    out = out.fillna(0.0)
    if return_mask:
        return out, avail
    return out


def infer_filled_mask(df: pd.DataFrame, power_cols: Sequence[str], rated_kw: Dict[str, float],
                      irr_col: str = "Irradiance (W/m2)", avail: Optional[pd.DataFrame] = None,
                      min_const_run: int = 3, max_ratio: float = 1.3,
                      ghi_day: float = 200.0, zero_day_rule: str = "auto") -> pd.DataFrame:
    """Observation-validity mask for power columns of an ALREADY-filled file.

    When the upstream export has been ffill-ed / zero-filled (no NaN left, as in
    output.xlsx) the original gaps must be reconstructed heuristically.  A value is
    marked NOT observed if any of:
      * it was NaN in the raw frame (``avail``);
      * it belongs to a run of >= ``min_const_run`` identical non-zero daylight
        values (signature of forward filling);
      * the whole day is exactly zero although GHI > ``ghi_day`` occurred that day
        (a zero-filled gap; a real inverter trip rarely spans a full sunny day and,
        for degradation, is equally unusable);
      * P > ``max_ratio`` * P_rated (physically impossible spike).
    Partial-day zeros are kept as observations (they are potential outages that the
    anomaly detector should see).

    zero_day_rule : "auto" (default) - the whole-day-zero heuristic is applied only to columns
                    whose raw frame contains NO NaN (i.e. the file was gap-filled upstream).  When
                    gaps are kept as NaN until aggregation (pv_preprocess, min_count rule), a zero
                    on a sunny day is a real measurement and stays observed, so full-day outages
                    reach the detector and the rule "P ~ 0 at GHI > 200".
                    "always" - legacy behaviour; "never" - disable.
    """
    idx = df.index
    ghi = df[irr_col].to_numpy(float) if irr_col in df else np.full(len(df), 1000.0)
    day_key = idx.normalize()
    sunny_day = pd.Series(ghi > ghi_day, index=idx).groupby(day_key).transform("any").to_numpy()
    out = {}
    for c in power_cols:
        p = df[c].to_numpy(float)
        ok = np.isfinite(p)
        if avail is not None and c in avail:
            ok &= avail[c].to_numpy(bool)
        # forward-fill signature
        same = np.r_[False, np.isclose(p[1:], p[:-1])] & (p != 0) & (ghi > 50)
        run_id = np.cumsum(~same)
        run_len = pd.Series(1, index=run_id).groupby(level=0).transform("size").to_numpy()
        const = (run_len >= min_const_run) & (pd.Series(same).groupby(run_id).transform("max").to_numpy())
        ok &= ~const
        # whole-day zero on a sunny day (only for gap-filled inputs, see zero_day_rule)
        has_nan = bool(avail is not None and c in avail and not avail[c].to_numpy(bool).all())
        if zero_day_rule == "always" or (zero_day_rule == "auto" and not has_nan):
            dmax = pd.Series(np.nan_to_num(p), index=idx).groupby(day_key).transform("max").to_numpy()
            ok &= ~((dmax <= 0) & sunny_day)
        # impossible spikes
        kw = rated_kw.get(c) or 0.0
        if kw > 0:
            ok &= ~(p > max_ratio * kw * 1000.0)
        out[c] = ok
    return pd.DataFrame(out, index=idx)


def build_from_dataframe(df: pd.DataFrame, ttl_path: str, *,
                         time_col: Optional[str] = None,
                         power_cols: Optional[Sequence[str]] = None,
                         irr_col: str = "Irradiance (W/m2)",
                         temp_col: str = "Temp (Degree Celsius)",
                         wind_col: str = "wind_speed",
                         exo_cols: Sequence[str] = DEFAULT_EXO,
                         window: int = 48, horizon: int = 1, k_peers: int = 6,
                         prior: Optional[PhysicsPrior] = None,
                         tz_offset_hours: float = 8.0,
                         peer_fit_end: Optional[object] = None,
                         activity_mask: Optional[pd.DataFrame] = None,
                         norm_fit_end: Optional[object] = None,
                         zero_day_rule: str = "auto"):
    """Same as build_from_files, but for a DataFrame you already hold in memory.

    zero_day_rule : passed to infer_filled_mask ("auto" | "always" | "never")."""
    df, raw_avail = prepare_dataframe(df, time_col, return_mask=True)
    if power_cols is None:
        power_cols = [c for c in df.columns if c.startswith("power(W)_")]
        if not power_cols:              # fallback: everything that is not a weather channel
            power_cols = [c for c in df.columns
                          if c not in set(exo_cols) | {irr_col, temp_col, wind_col}]
    exo_cols = [c for c in exo_cols if c in df.columns and c != irr_col]
    brick = BrickPVModel(ttl_path)
    builder = PVGraphBuilder(brick, prior or PhysicsPrior(), k_peers=k_peers,
                             tz_offset_hours=tz_offset_hours)
    rated = {c: (brick.sites[m].rated_kw or 0.0) for c, m in builder.align_columns(power_cols).items()}
    observed = infer_filled_mask(df, power_cols, rated, irr_col=irr_col, avail=raw_avail,
                                 zero_day_rule=zero_day_rule)
    builder.raw_availability = raw_avail
    builder.observed = observed                     # validity of each power value
    graph = builder.build(df, power_cols, irr_col=irr_col, temp_col=temp_col,
                          wind_col=wind_col, exo_cols=exo_cols, peer_fit_end=peer_fit_end,
                          observed=observed)
    phys = builder.physics_tensors(df, graph, irr_col=irr_col, temp_col=temp_col,
                                   wind_col=wind_col)
    ds = PIGraphDataset(df, graph, phys, window=window, horizon=horizon,
                        activity_mask=activity_mask, norm_fit_end=norm_fit_end)
    return df, brick, builder, graph, phys, ds


def split_datasets(df: pd.DataFrame, graph: PVGraph, phys: Dict[str, np.ndarray],
                   split, window: int = 48, horizon: int = 1):
    """Chronological train/test split with the physics tensors kept aligned.

    The test slice starts `window + horizon` steps early so the first test window is
    complete without leaking any target into training.
    """
    i = int(df.index.searchsorted(pd.Timestamp(split)))
    slices = (slice(0, i), slice(max(i - window - horizon, 0), len(df)))
    out = []
    for s in slices:
        sub = {k: (v[s] if isinstance(v, np.ndarray) and v.shape[0] == len(df) else v)
               for k, v in phys.items()}
        out.append(PIGraphDataset(df.iloc[s], graph, sub, window=window, horizon=horizon))
    return out[0], out[1]


def build_from_files(xlsx_path: str, ttl_path: str, sheet: str = "Data",
                     window: int = 48, horizon: int = 1, k_peers: int = 6,
                     exo_cols: Sequence[str] = DEFAULT_EXO):
    """Convenience path when the Excel file has not been read yet."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    return build_from_dataframe(df, ttl_path, window=window, horizon=horizon,
                                k_peers=k_peers, exo_cols=exo_cols)


REFERENCES = """
Physical model (irradiance -> PV power) and its sensitivity
  1. D. Faiman (2008). "Assessing the outdoor operating temperature of photovoltaic
     modules." Progress in Photovoltaics 16(4):307-315.  T_cell = T_air + G_poa/(U0 + U1*w);
     the U0 = 25.0 W/m2K, U1 = 6.84 W/m3sK defaults are the IEC 61853-2 / Sandia values.
     https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/module-temperature/faiman-module-temperature-model/
  2. A. P. Dobos (2014). "PVWatts Version 5 Manual", NREL/TP-6A20-62641 - the
     P = Pdc0*(Gpoa/1000)*(1+gamma*(Tcell-25)) form and the system derate.
     https://docs.nrel.gov/docs/fy14osti/62641.pdf  (record: https://www.osti.gov/biblio/1158421)
  3. D. L. King, W. E. Boyson, J. A. Kratochvil (2004). "Photovoltaic Array Performance
     Model", SAND2004-3535 - Sandia array performance model and temperature coefficients.
     https://pvpmc.sandia.gov/modeling-guide/
  4. J. E. Hay, J. A. Davies (1980) sky-diffuse transposition and the HDKR variant in
     J. A. Duffie & W. A. Beckman, "Solar Engineering of Thermal Processes" (ch. 2) -
     the POA/GHI transposition factor R_tr at a given tilt and azimuth.
     https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/plane-of-array-poa-irradiance/calculating-poa-irradiance/poa-sky-diffuse/hay-and-davies-sky-diffuse-model/
  5. D. G. Erbs, S. A. Klein, J. A. Duffie (1982). Diffuse-fraction correlation used to
     split the measured GHI into DNI/DHI. Solar Energy 28(4):293-302.
  6. W. F. Holmgren, C. W. Hansen, M. A. Mikofski (2018). "pvlib python: a python package
     for modeling solar energy systems." JOSS 3(29):884 - reference implementations of
     (1)-(5); this module mirrors its formulas so pvlib stays an optional dependency.
     https://pvlib-python.readthedocs.io/en/stable/reference/irradiance/transposition.html

Semantic metadata
  7. B. Balaji et al. (2016). "Brick: Towards a Unified Metadata Schema For Buildings."
     ACM BuildSys 2016.  https://brickschema.org/papers/Brick-BuildSys-2016-Balaji.pdf
     (DOI: https://dl.acm.org/doi/10.1145/2993422.2993577) - source of tilt, azimuth,
     rated power, coordinates, location and hasPart topology.

Spatio-temporal architecture
  8. Z. Wu, S. Pan, G. Long, J. Jiang, C. Zhang (2019). "Graph WaveNet for Deep
     Spatial-Temporal Graph Modeling." IJCAI 2019 - dilated causal convolutions before
     graph message passing.  https://ar5iv.labs.arxiv.org/html/1906.00121
  9. S. Bai, J. Z. Kolter, V. Koltun (2018). "An Empirical Evaluation of Generic
     Convolutional and Recurrent Networks for Sequence Modeling." arXiv:1803.01271 - TCN.
 10. P. Velickovic et al. (2018). "Graph Attention Networks." ICLR 2018 - the attention
     logit that the log-physics prior is added to.

Joint forecasting / reconstruction anomaly detection
 11. A. Deng, B. Hooi (2021). "Graph Neural Network-Based Anomaly Detection in
     Multivariate Time Series." AAAI 2021 - forecast-deviation graph anomaly scores.
     https://arxiv.org/abs/2106.06947
 12. H. Zhao et al. (2020). "Multivariate Time-series Anomaly Detection via Graph
     Attention Network" (MTAD-GAT), ICDM 2020 - joint forecast + reconstruction objective
     and the normalised per-node error score used here.

Physics-informed learning
 13. M. Raissi, P. Perdikaris, G. E. Karniadakis (2019). "Physics-informed neural
     networks." J. Comput. Physics 378:686-707 - the residual-as-loss-term principle
     behind L_physics = (eta_hat * P_expected - P_measured)^2.
"""


# ============================================================================ data signature
def data_signature(D: dict, boundary="2023-01-01") -> dict:
    """Identity of the training data of a checkpoint: PV columns and a SHA-256 of the
    train-period values (float32, NaN -> sentinel).  Stored in checkpoint meta so that a model
    trained on another fleet / another preprocessing is never reused silently."""
    import hashlib
    df = D["df"]
    tr = df.loc[df.index < pd.Timestamp(boundary)]
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(f"{len(tr)}|{tr.index[0] if len(tr) else ''}|{tr.index[-1] if len(tr) else ''}".encode())
    h.update(np.nan_to_num(tr.to_numpy(np.float32), nan=-9999.0).tobytes())
    return dict(pv_cols=list(D["pv_cols"]), n_pv=len(D["pv_cols"]), train_sha=h.hexdigest()[:16])
