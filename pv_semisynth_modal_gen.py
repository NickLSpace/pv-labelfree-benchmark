"""Section 15, stage 3: semi-synthetic scenarios that are physically consistent across the
site meter and the inverter channels (independent generator; sections 10-14 are untouched).

Every scenario modifies a COPY of the real hourly site power ``D["df"]`` and of the hourly
inverter channels (``pv_modal_data.ModalData``) so that the channels stay consistent:

==============  ========  =====  =======================================================
scenario        class     alarm  what changes
==============  ========  =====  =======================================================
inverter_trip   inverter  yes    one inverter of a multi-inverter site stops for 1-3 days:
                                 its P = 0, DC side off, 5-min zeros = 12; site P -= its P
short_outage    inverter  yes    2-4 daylight hours of one day with a 10-30 min stop of one
                                 inverter: P_k and site P reduced by m/60, 5-min zeros += m/5
phase_loss      inverter  yes    one AC phase lost for 1-2 days: P_k x 2/3, site P -= P_k/3,
                                 phase imbalance 1.5
grid_curtail    grid      no     grid over-voltage on a clear day, 2-5 midday hours: AC V above
                                 the inverter threshold by 1-5 V, all inverters of the site x 0.4-0.8
metering        data      no     site meter reads 50-80 % for 1-3 days; inverters unchanged
drift_fault     plant     yes    fleet-wide pyranometer drift (measured GHI down 6-12 % over
                                 3 weeks, P_exp recomputed) + a real 30 % loss of one plant for
                                 2 days inside the drift window (inverters scaled, no modal cause)
==============  ========  =====  =======================================================

Alarm = "a maintenance alarm is expected".  ``grid`` and ``data`` events are NOT plant faults:
the right output is a notice of the cause, an alarm is a nuisance alarm.  Communication loss
(all channels NaN) is not generated: unobserved hours are never scored by any detector.

Usage::

    import pv_semisynth_modal_gen as smg
    plan = smg.plan_modal(D, C, M, seed=0)
    real = smg.apply_modal(D, M, plan)        # -> ModalRealisation(df, phys, modal, labels, inj_mask)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

IRR = "Irradiance (W/m2)"
CLASSES = ("plant", "inverter", "grid", "data")
MODAL_SCENARIOS = {
    "inverter_trip": dict(cls="inverter", alarm=True, title="Single inverter trip (multi-inverter site)"),
    "short_outage": dict(cls="inverter", alarm=True, title="Short inverter outages, 10-30 min"),
    "phase_loss": dict(cls="inverter", alarm=True, title="Loss of one AC phase"),
    "grid_curtail": dict(cls="grid", alarm=False, title="Grid over-voltage curtailment (no fault)"),
    "metering": dict(cls="data", alarm=False, title="Site meter fault (inverters unchanged)"),
    "drift_fault": dict(cls="plant", alarm=True, title="Pyranometer drift + plant fault"),
}
# optional scenarios (section 16; NOT in the default list, so section 15 is unchanged).  Events of an
# optional scenario are planned after the default ones, so the default events of a placement seed are
# identical with and without them.
EXTRA_SCENARIOS = {
    "new_obstruction": dict(cls="plant", alarm=True,
                            title="New obstruction (scaffold / new building): beam blocked at new sun positions"),
}
ALL_SCENARIOS = {**MODAL_SCENARIOS, **EXTRA_SCENARIOS}


@dataclass
class ModalGenConfig:
    boundary: str = "2023-01-01"
    n_per_scenario: int = 6
    warmup_days: int = 10              # STGNN window (48 h) + calibration (168 h) after the first active hour
    min_gap_days: int = 4              # between two events of the same site
    trip_days: Tuple[int, ...] = (1, 2, 3)
    short_minutes: Tuple[int, ...] = (10, 15, 20, 30)
    short_hours: Tuple[int, int] = (2, 4)
    phase_days: Tuple[int, ...] = (1, 2)
    curtail_factor: Tuple[float, float] = (0.4, 0.8)
    curtail_dv: Tuple[float, float] = (1.0, 5.0)
    curtail_hours: Tuple[int, int] = (2, 5)
    metering_factor: Tuple[float, float] = (0.5, 0.8)
    metering_days: Tuple[int, ...] = (1, 2, 3)
    drift: Tuple[float, float] = (0.06, 0.12)
    drift_days: int = 21
    fault_loss: float = 0.30
    fault_days: int = 2
    day_ghi: float = 50.0
    seed_offset: int = 31_000
    scenarios: Optional[Tuple[str, ...]] = None          # None = MODAL_SCENARIOS (section 15)
    obstruction_days: Tuple[int, ...] = (7, 10, 14)
    obstruction_block: Tuple[float, float] = (0.6, 0.9)  # share of the beam blocked
    obstruction_width: float = 60.0                      # azimuth sector, deg
    obstruction_height: float = 15.0                     # deg above the sun elevation of the anchor hour


@dataclass
class ModalPlan:
    labels: pd.DataFrame
    drift: Optional[dict]
    cfg: ModalGenConfig
    seed: int


@dataclass
class ModalRealisation:
    df: pd.DataFrame
    phys: Dict[str, np.ndarray]
    modal: "object"                    # pv_modal_data.ModalData
    labels: pd.DataFrame
    inj_mask: pd.DataFrame
    drift: Optional[dict] = None
    seed: int = 0


def _short(c: str) -> str:
    return str(c).replace("power(W)_", "")


def _candidate_days(D: dict, M, col: str, cfg: ModalGenConfig, ghi_min: float = 400.0) -> List[pd.Timestamp]:
    """Test days with observed site power AND inverter data in >= 6 daylight hours and max GHI >= ghi_min."""
    df = D["df"]; idx = df.index
    b = pd.Timestamp(cfg.boundary)
    obs = D["observed"][col].to_numpy(bool) if "observed" in D and col in D["observed"] else np.ones(len(idx), bool)
    act = D["act"][col].to_numpy(bool) if D.get("act") is not None and col in D["act"] else np.ones(len(idx), bool)
    first = idx[np.argmax(act & obs)] if (act & obs).any() else idx[-1]
    start = max(b, first.normalize() + pd.Timedelta(days=cfg.warmup_days))
    ghi = df[IRR].to_numpy(float)
    ks = M.inverters_of(col) if M is not None else []
    inv_ok = np.ones(len(idx), bool)
    for k in ks:
        inv_ok &= M.inv[k]["P"].reindex(idx).notna().to_numpy()
    day = (ghi > cfg.day_ghi) & obs & act & (inv_ok if ks else True)
    d = pd.DataFrame({"day": day, "g": ghi}, index=idx)
    d = d[idx >= start].groupby(d.index[idx >= start].normalize()).agg(n=("day", "sum"), gmax=("g", "max"))
    return list(d.index[(d["n"] >= 6) & (d["gmax"] >= ghi_min)])


def plan_modal(D: dict, C: dict, M, seed: int = 0, cfg: Optional[ModalGenConfig] = None,
               scenarios: Optional[Sequence[str]] = None) -> ModalPlan:
    """Random placement of the modal scenarios on sites with inverter data (exact labels)."""
    cfg = cfg or ModalGenConfig()
    rng = np.random.default_rng(cfg.seed_offset + seed)
    scenarios = list(scenarios or cfg.scenarios or MODAL_SCENARIOS)
    scenarios = [x for x in MODAL_SCENARIOS if x in scenarios] + [x for x in EXTRA_SCENARIOS if x in scenarios]
    sites = [c for c in M.sites() if c in D["pv_cols"]]
    multi = [c for c in sites if len(M.inverters_of(c)) >= 2]
    cand = {c: _candidate_days(D, M, c, cfg) for c in sites}
    clear = {c: _candidate_days(D, M, c, cfg, ghi_min=650.0) for c in sites}
    busy: Dict[str, List[pd.Timestamp]] = {c: [] for c in D["pv_cols"]}
    rows, eid = [], 0

    def free(c, d0, nd):
        return all(abs((d0 - d).days) >= cfg.min_gap_days + nd for d in busy[c])

    def pick(pool, days_of, nd):
        for _ in range(200):
            if not pool:
                return None
            c = pool[int(rng.integers(len(pool)))]
            ds = days_of.get(c) or []
            if not ds:
                continue
            d0 = ds[int(rng.integers(len(ds)))]
            if free(c, d0, nd):
                return c, d0
        return None

    drift = None
    if "drift_fault" in scenarios:
        all_days = sorted(set().union(*[set(v) for v in cand.values()])) if cand else []
        if all_days:
            ws = [d for d in all_days if d + pd.Timedelta(days=cfg.drift_days) <= all_days[-1]]
            if ws:
                d0 = ws[int(rng.integers(len(ws)))]
                d1 = d0 + pd.Timedelta(days=cfg.drift_days) - pd.Timedelta(hours=1)
                drift = dict(start=d0, end=d1, strength=float(rng.uniform(*cfg.drift)))
                fault_from = d0 + pd.Timedelta(days=cfg.drift_days // 2)
                pool = [c for c in sites if any(fault_from <= d <= d1 - pd.Timedelta(days=cfg.fault_days)
                                                for d in cand[c])]
                for _ in range(min(cfg.n_per_scenario, len(pool))):
                    days_in = {c: [d for d in cand[c] if fault_from <= d <= d1 - pd.Timedelta(days=cfg.fault_days)]
                               for c in pool}
                    p = pick(pool, days_in, cfg.fault_days)
                    if p is None:
                        break
                    c, d = p; pool.remove(c)
                    busy[c].append(d)
                    rows.append(dict(event_id=eid, scenario="drift_fault", column=c, inverter=None, start=d,
                                     end=d + pd.Timedelta(days=cfg.fault_days) - pd.Timedelta(hours=1),
                                     strength=cfg.fault_loss)); eid += 1
    for scn in scenarios:
        if scn == "drift_fault":
            continue
        for _ in range(cfg.n_per_scenario):
            if scn == "inverter_trip":
                nd = int(rng.choice(cfg.trip_days)); p = pick(list(multi), cand, nd); st = float(nd)
            elif scn == "phase_loss":
                nd = int(rng.choice(cfg.phase_days)); p = pick(list(sites), cand, nd); st = float(nd)
            elif scn == "short_outage":
                nd = 1; p = pick(list(sites), cand, nd); st = float(rng.choice(cfg.short_minutes))
            elif scn == "grid_curtail":
                nd = 1; p = pick(list(sites), clear, nd); st = float(rng.uniform(*cfg.curtail_factor))
            elif scn == "metering":
                nd = int(rng.choice(cfg.metering_days)); p = pick(list(sites), cand, nd)
                st = float(rng.uniform(*cfg.metering_factor))
            elif scn == "new_obstruction":       # any plant (with or without inverter data)
                for c in D["pv_cols"]:
                    if c not in cand:
                        cand[c] = _candidate_days(D, None, c, cfg)
                nd = int(rng.choice(cfg.obstruction_days)); p = pick(list(D["pv_cols"]), cand, nd)
                st = float(rng.uniform(*cfg.obstruction_block))
            else:
                raise ValueError(scn)
            if p is None:
                break
            c, d = p
            busy[c].append(d)
            ks = M.inverters_of(c)
            k = ks[int(rng.integers(len(ks)))] if scn in ("inverter_trip", "short_outage", "phase_loss") else None
            rows.append(dict(event_id=eid, scenario=scn, column=c, inverter=k, start=d,
                             end=d + pd.Timedelta(days=nd) - pd.Timedelta(hours=1), strength=st)); eid += 1
    L = pd.DataFrame(rows, columns=["event_id", "scenario", "column", "inverter", "start", "end", "strength"])
    L["cls"] = L["scenario"].map(lambda s: ALL_SCENARIOS[s]["cls"])
    L["expected_alarm"] = L["scenario"].map(lambda s: ALL_SCENARIOS[s]["alarm"])
    L["plant"] = L["column"].map(_short)
    L["has_inverters"] = L["column"].map(lambda c: bool(M.inverters_of(c)))
    check_strengths(L, cfg)
    return ModalPlan(labels=L, drift=drift, cfg=cfg, seed=seed)


def strength_ranges(cfg: ModalGenConfig) -> Dict[str, Tuple[float, float]]:
    """Configured range of the labelled ``strength`` for every scenario (closed interval)."""
    return {"inverter_trip": (min(cfg.trip_days), max(cfg.trip_days)),
            "phase_loss": (min(cfg.phase_days), max(cfg.phase_days)),
            "short_outage": (min(cfg.short_minutes), max(cfg.short_minutes)),
            "grid_curtail": tuple(cfg.curtail_factor),
            "metering": tuple(cfg.metering_factor),
            "new_obstruction": tuple(cfg.obstruction_block),
            "drift_fault": (cfg.fault_loss, cfg.fault_loss)}


def check_strengths(L: pd.DataFrame, cfg: ModalGenConfig) -> None:
    """Every labelled strength must lie in the range configured for its own scenario."""
    R = strength_ranges(cfg)
    for scn, g in L.groupby("scenario"):
        lo, hi = R[scn]
        bad = g[(g["strength"] < lo - 1e-9) | (g["strength"] > hi + 1e-9)]
        if len(bad):
            raise ValueError(f"{scn}: labelled strength outside [{lo}, {hi}] for events "
                             f"{bad['event_id'].tolist()}")


def apply_modal(D: dict, M, plan: ModalPlan) -> ModalRealisation:
    """Inject the planned events into copies of the hourly site frame and the inverter channels."""
    cfg = plan.cfg
    rng = np.random.default_rng(cfg.seed_offset + 1000 + plan.seed)
    df = D["df"].copy(); M2 = M.copy(); idx = df.index; pv = list(D["pv_cols"])
    P0 = D["df"][pv].to_numpy(float)
    ghi = df[IRR].to_numpy(float)
    phys = D["phys"]
    drift = plan.drift
    if drift is not None:                      # measured GHI drifts; P (true irradiance) unchanged
        m = (idx >= drift["start"]) & (idx <= drift["end"])
        n = m.sum(); ramp = np.minimum(np.arange(n) / max(n / 3.0, 1.0), 1.0)
        g = ghi.copy(); g[m] = ghi[m] * (1 - drift["strength"] * ramp)
        df[IRR] = g
        graph = D["graph_base"]
        phys = D["builder"].physics_tensors(df, graph)
    rows = []
    for r in plan.labels.itertuples():
        c = r.column; j = pv.index(c)
        sl = (idx >= r.start) & (idx <= r.end)
        day = sl & (ghi > cfg.day_ghi)
        ks = M2.inverters_of(c)
        before = df[c].to_numpy(float).copy()
        if r.scenario == "inverter_trip":
            F = M2.inv[r.inverter]; pk = F["P"].reindex(idx).to_numpy(float)
            h = day & np.isfinite(pk)
            df.loc[h, c] = np.maximum(df.loc[h, c].to_numpy(float) - pk[h], 0.0)
            F.loc[h, ["P", "imb", "cv"]] = [0.0, np.nan, 0.0]
            F.loc[h, "n_zero"] = F.loc[h, "n_obs"]; F.loc[h, "dc_on"] = 0.0
            F.loc[h, "dcV"] = float(F["dcV"].quantile(0.05)) if F["dcV"].notna().any() else 0.0
        elif r.scenario == "short_outage":
            F = M2.inv[r.inverter]; pk = F["P"].reindex(idx).to_numpy(float)
            cand_h = np.flatnonzero(sl & (ghi > 200) & np.isfinite(pk) & (pk > 0))
            nh = min(int(rng.integers(cfg.short_hours[0], cfg.short_hours[1] + 1)), len(cand_h))
            hh = np.sort(rng.choice(cand_h, nh, replace=False)) if nh else np.array([], int)
            fr = r.strength / 60.0
            for p in hh:
                t = idx[p]; d = pk[p] * fr
                df.iat[p, df.columns.get_loc(c)] = max(df.iat[p, df.columns.get_loc(c)] - d, 0.0)
                F.at[t, "P"] = pk[p] - d
                F.at[t, "n_zero"] = min(np.nan_to_num(F.at[t, "n_zero"]) + r.strength / 5, F.at[t, "n_obs"] or 12)
                F.at[t, "cv"] = max(np.nan_to_num(F.at[t, "cv"]), np.sqrt(fr * (1 - fr)) / max(1 - fr, 1e-3))
                F.at[t, "dc_on"] = np.nan_to_num(F.at[t, "dc_on"], nan=1.0) * (1 - fr)
        elif r.scenario == "phase_loss":
            F = M2.inv[r.inverter]; pk = F["P"].reindex(idx).to_numpy(float)
            h = day & np.isfinite(pk)
            df.loc[h, c] = np.maximum(df.loc[h, c].to_numpy(float) - pk[h] / 3.0, 0.0)
            F.loc[h, "P"] = pk[h] * 2.0 / 3.0
            F.loc[h & (pk > 0), "imb"] = 1.5
        elif r.scenario == "grid_curtail":
            cand_h = np.flatnonzero(sl & (ghi > 300))
            if len(cand_h):
                L = int(rng.integers(cfg.curtail_hours[0], cfg.curtail_hours[1] + 1))
                mid = cand_h[len(cand_h) // 2]; a = max(cand_h[0], mid - L // 2); hh = [p for p in range(a, a + L) if p in set(cand_h)]
                dv = float(rng.uniform(*cfg.curtail_dv)); f = r.strength
                for p in hh:
                    t = idx[p]; tot_d = 0.0
                    for k in ks:
                        F = M2.inv[k]
                        if not np.isfinite(F.at[t, "P"]):
                            continue
                        d = F.at[t, "P"] * (1 - f); tot_d += d
                        F.at[t, "P"] -= d
                        thr = M2.vac_thr.get(k, M2.cfg.vac_limit)
                        F.at[t, "vac_max"] = max(np.nan_to_num(F.at[t, "vac_max"]), thr + dv)
                        F.at[t, "vac"] = max(np.nan_to_num(F.at[t, "vac"]), thr + dv - 3.0)
                    df.iat[p, df.columns.get_loc(c)] = max(df.iat[p, df.columns.get_loc(c)] - tot_d, 0.0)
        elif r.scenario == "metering":
            df.loc[day, c] = df.loc[day, c].to_numpy(float) * r.strength
        elif r.scenario == "new_obstruction":
            import pv_shading as psh
            sun = psh.sun_frame(D, idx); el, az = sun["elev"].to_numpy(), sun["az"].to_numpy()
            cand_h = np.flatnonzero(sl & (ghi > 200) & (el > 5))
            if len(cand_h):
                p0 = cand_h[int(rng.integers(len(cand_h)))]
                a0, h0 = az[p0], el[p0] + cfg.obstruction_height
                dz = (az - a0 + 180) % 360 - 180
                hit = day & (np.abs(dz) <= cfg.obstruction_width / 2) & (el > 0) & (el < h0)
                bs = np.clip(psh.erbs_beam_share(psh._kt(D, df)) / 0.835, 0, 1)
                f = 1.0 - r.strength * bs
                df.loc[hit, c] = df.loc[hit, c].to_numpy(float) * f[hit]
                for k in ks:
                    F = M2.inv[k]; h = hit & F["P"].reindex(idx).notna().to_numpy()
                    F.loc[h, "P"] = F.loc[h, "P"].to_numpy(float) * f[h]
        elif r.scenario == "drift_fault":
            df.loc[day, c] = df.loc[day, c].to_numpy(float) * (1 - r.strength)
            for k in ks:
                F = M2.inv[k]; h = day & F["P"].reindex(idx).notna().to_numpy()
                F.loc[h, "P"] = F.loc[h, "P"] * (1 - r.strength)
        after = df[c].to_numpy(float)
        ch = np.flatnonzero(np.abs(np.nan_to_num(after - before)) > 1e-6)
        s0, s1 = np.nansum(before[ch]), np.nansum(after[ch])
        rows.append(dict(event_id=r.event_id, eff_start=idx[ch[0]] if len(ch) else r.start,
                         eff_end=idx[ch[-1]] if len(ch) else r.end, n_eff_h=int(len(ch)),
                         site_loss_pct=100 * (1 - s1 / s0) if s0 > 0 else 0.0))
    lab = plan.labels.merge(pd.DataFrame(rows, columns=["event_id", "eff_start", "eff_end", "n_eff_h", "site_loss_pct"]),
                            on="event_id", how="left")
    # an event without any modified hour can still be visible in the modal channels (grid / phase)
    lab["effective"] = (lab["n_eff_h"].fillna(0) > 0) | lab["scenario"].isin(["phase_loss", "grid_curtail"])
    inj = np.abs(np.nan_to_num(df[pv].to_numpy(float) - P0)) > 1e-6
    return ModalRealisation(df=df, phys=phys, modal=M2, labels=lab, inj_mask=pd.DataFrame(inj, index=idx, columns=pv),
                            drift=drift, seed=plan.seed)


def scenario_table(extra: bool = False) -> pd.DataFrame:
    return pd.DataFrame([dict(Scenario=k, Title=v["title"], Class=v["cls"], **{"Alarm expected": "yes" if v["alarm"] else "no"})
                         for k, v in (ALL_SCENARIOS if extra else MODAL_SCENARIOS).items()])
