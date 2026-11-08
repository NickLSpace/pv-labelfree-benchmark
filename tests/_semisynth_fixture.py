"""Small synthetic fleet with a REAL (tiny) STGNN for testing the semi-synthetic modules.
Not HKUST data: weather is simulated, but the pipeline objects (D, C, graph, builder) have
the same structure as run_pipeline.stage_data / stage_clusters."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from types import SimpleNamespace
from pv_physics_graph import PVGraph

IRR, TMP, WND = "Irradiance (W/m2)", "Temp (Degree Celsius)", "wind_speed"

# Real-world effects that the physics model P_exp (MockBuilder) does NOT know about.
#   pyranometer_drift : the measured GHI loses sensitivity linearly (dome soiling / ageing);
#                       P follows the TRUE irradiance, P_exp the measured one -> fleet-wide P/P_exp drift
#   pexp_error        : plant-specific model error: orientation (hour-dependent gain), low-light
#                       efficiency loss, morning / afternoon shading by neighbouring buildings
#   seasonal_bias     : plant-specific seasonal gain (dry-season soiling, washed off by summer rain);
#                       shading also lasts longer in winter (low sun)
ALL_EFFECTS = ("pyranometer_drift", "pexp_error", "seasonal_bias")
EFFECT_DEFAULTS = dict(drift_per_year=-0.03,          # -3 %/year of pyranometer sensitivity
                       orient_amp=0.12,               # |gain error| at 6 h from noon, up to 12 %
                       lowlight_amp=(0.05, 0.15),     # efficiency loss at very low GHI
                       shaded_share=0.35, shade_depth=(0.3, 0.6),
                       seasonal_amp=(0.02, 0.08))     # seasonal gain amplitude per plant


class MockBuilder:
    def __init__(self, rated): self.rated = rated
    def physics_tensors(self, df, graph):
        G = df[IRR].to_numpy(float); T = df[TMP].to_numpy(float)
        cols = graph.meta["pv_cols"]
        pe = np.stack([self.rated[c] * 1000 * G / 1000 * 0.8 * (1 - 0.004 * (T + 0.03 * G - 25)) for c in cols], 1)
        return dict(p_expected=np.clip(pe, 0, None), dP_dGHI=np.ones_like(pe), ghi=G)


def _effect_factors(effects, idx, ghi_true, n_pv, rng, prm):
    """Multiplicative gain of the TRUE power w.r.t. the physics model, per hour and plant,
    and the gain of the pyranometer.  Returns (plant_gain [T, n], sensor_gain [T], meta DataFrame)."""
    T = len(idx); hour = idx.hour.to_numpy() + 0.5; doy = idx.dayofyear.to_numpy()
    years = (idx - idx[0]).total_seconds().to_numpy() / (365.25 * 86400)
    gain = np.ones((T, n_pv)); meta = pd.DataFrame(index=range(n_pv))
    sensor = np.ones(T)
    if "pyranometer_drift" in effects:
        sensor = 1.0 + prm["drift_per_year"] * years
    winter = 0.5 * (1 + np.cos(2 * np.pi * (doy - 355) / 365.25))          # 1 in late December, 0 in June
    if "pexp_error" in effects:
        a = rng.uniform(-prm["orient_amp"], prm["orient_amp"], n_pv)          # east (<0) / west (>0) bias
        c = rng.uniform(*prm["lowlight_amp"], n_pv)
        shaded = rng.random(n_pv) < prm["shaded_share"]
        side = rng.choice(["morning", "afternoon"], n_pv)
        depth = rng.uniform(*prm["shade_depth"], n_pv)
        x = np.clip((hour - 12.0) / 6.0, -1, 1)
        for j in range(n_pv):
            g = (1 + a[j] * x[:]) * (1 - c[j] * np.exp(-ghi_true / 150.0))
            if shaded[j]:
                extra = 1.5 * winter if "seasonal_bias" in effects else 0.0
                on = hour < 8.0 + extra if side[j] == "morning" else hour > 16.0 - extra
                g = g * np.where(on, 1 - depth[j], 1.0)
            gain[:, j] *= g
        meta["orient_gain"] = a; meta["lowlight_loss"] = c
        meta["shading"] = np.where(shaded, side, "none"); meta["shade_depth"] = np.where(shaded, depth, 0.0)
    if "seasonal_bias" in effects:
        amp = rng.uniform(*prm["seasonal_amp"], n_pv)
        phase = rng.normal(45, 20, n_pv)                                       # soiling peak ~ mid-February
        for j in range(n_pv):
            gain[:, j] *= 1 - amp[j] * 0.5 * (1 + np.cos(2 * np.pi * (doy - phase[j]) / 365.25))
        meta["seasonal_amp"] = amp; meta["seasonal_peak_doy"] = phase
    return gain, sensor, meta


def make(n_pv=14, start="2022-07-01", end="2023-12-31 23:00", seed=3, cohorts=False, effects=None, effect_params=None):
    """cohorts=True: 2021-2023 fleet with three commissioning cohorts like HKUST
    (2021 / 2022 / 2023) and one plant decommissioned in 2023; no data outside exploitation;
    cohorts assigned by pv_cohorts.analyse_cohorts as in run_pipeline.stage_data.

    effects: subset of ALL_EFFECTS (see top of the file); default: none for the plain fleet,
    all three for ``cohorts=True``.  ``D["truth"]`` keeps the true GHI, the sensor gain and
    the plant-level parameters of the effects (never used by the pipeline)."""
    if cohorts:
        return _make_cohorts(n_pv=n_pv, seed=seed, effects=ALL_EFFECTS if effects is None else effects,
                             effect_params=effect_params)
    effects = tuple(effects or ())
    unknown = set(effects) - set(ALL_EFFECTS)
    assert not unknown, unknown
    prm = {**EFFECT_DEFAULTS, **(effect_params or {})}
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h"); T = len(idx)
    hour = idx.hour.to_numpy(); doy = idx.dayofyear.to_numpy()
    clear = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * (850 + 150 * np.cos(2 * np.pi * (doy - 172) / 365))
    nd = T // 24 + 1
    dcl = np.repeat(rng.choice([0.95, 0.6, 0.25], nd, p=[.45, .35, .2]) * rng.uniform(.85, 1.05, nd), 24)[:T]
    ghi = clear * np.clip(dcl + rng.normal(0, .05, T), 0.05, 1.1)             # TRUE irradiance
    temp = 22 + 6 * np.sin(2 * np.pi * (doy - 100) / 365) + 4 * np.sin(np.pi * (hour - 8) / 12) + rng.normal(0, .5, T)
    wind = np.abs(rng.normal(2, 1, T))
    cols = [f"power(W)_P{i:02d}" for i in range(n_pv)]
    rated = {c: float(rng.choice([8.0, 10.0, 15.0])) for c in cols}
    eff_rng = np.random.default_rng(seed + 100)        # separate stream: effects do not change the weather
    gain, sensor, eff_meta = _effect_factors(effects, idx, ghi, n_pv, eff_rng, prm)
    ghi_meas = ghi * sensor
    df = pd.DataFrame({IRR: ghi, TMP: temp, WND: wind}, index=idx)          # true weather (for P)
    cl = np.arange(n_pv) % 3
    names = cols + [IRR, TMP, WND]
    src, dst, et = [], [], []
    for i in range(n_pv):
        for k in range(n_pv):
            if i != k and cl[i] == cl[k]:
                src.append(i); dst.append(k); et.append(1)
        src.append(n_pv); dst.append(i); et.append(2)
        src.append(n_pv + 1); dst.append(i); et.append(0)
    g = PVGraph(node_names=names, node_type=np.array([0] * n_pv + [1, 1, 1]),
                edge_index=np.array([src, dst]), edge_type=np.array(et), edge_weight=np.ones(len(et)),
                edge_phys_logit=np.zeros(len(et), np.float32),
                meta=dict(n_pv=n_pv, pv_cols=cols, mapping={c: c for c in cols},
                          physics={c: {"rated_kw": rated[c]} for c in cols}, exo_cols=[IRR, TMP, WND]))
    b = MockBuilder(rated)
    phys_true = b.physics_tensors(df, g)
    regional = np.repeat(rng.normal(0, .04, (nd, 3)), 24, 0)[:T]
    for j, c in enumerate(cols):
        P = phys_true["p_expected"][:, j] * gain[:, j] * (1 + regional[:, cl[j]] + rng.normal(0, .02, T)) \
            + rng.normal(0, 50, T)
        df[c] = np.clip(P, 0, None)
    df[IRR] = ghi_meas                                  # the pipeline only sees the measured GHI
    df = df[names]
    phys = b.physics_tensors(df, g) if effects else phys_true
    ghi_true, ghi = ghi, ghi_meas
    kt = pd.Series(np.where(clear > 20, ghi / np.maximum(clear, 1), np.nan), index=idx)
    t_cell = np.repeat((temp + 0.03 * ghi)[:, None], n_pv, 1)
    observed = pd.DataFrame(True, index=idx, columns=names)
    comm = pd.DataFrame({"commissioning_date": [idx[0]] * n_pv, "decommissioning_date": [pd.NaT] * n_pv}, index=cols)
    D = dict(df=df, phys=phys, pv_cols=cols, rated=rated, kt=kt, t_cell=t_cell, observed=observed, comm=comm,
             graph_base=g, builder=b)
    eff_meta.index = cols
    D["truth"] = dict(effects=effects, params=prm, ghi_true=pd.Series(ghi_true, index=idx), sensor_gain=pd.Series(sensor, index=idx),
                      plant_gain=pd.DataFrame(gain, index=idx, columns=cols), plants=eff_meta)
    valid = pd.DataFrame({c: ghi > 200 for c in cols}, index=idx)
    regime = pd.Series(np.where(ghi > 20, np.where(kt.fillna(0) > .6, 1, 2), 0), index=idx)
    onehot = np.eye(3, dtype=np.float32)[regime.to_numpy()]
    wr = SimpleNamespace(regime=regime, onehot=onehot, names={0: "night", 1: "R1: clear", 2: "R2: cloudy"})
    C = dict(valid=valid, tech=SimpleNamespace(labels=pd.Series(cl, index=cols)), wr=wr, graph_mv=g)
    return D, C


def _make_cohorts(n_pv=30, seed=3, effects=ALL_EFFECTS, effect_params=None):
    from pv_physics_graph import exploitation_mask
    from pv_cohorts import coverage_mask, analyse_cohorts
    D, C = make(n_pv=n_pv, start="2021-01-01", end="2023-12-31 23:00", seed=seed, effects=effects,
                effect_params=effect_params)
    cols, idx = D["pv_cols"], D["df"].index
    rng = np.random.default_rng(seed + 1)
    comm = D["comm"].copy()
    k1, k2 = int(round(n_pv * 0.45)), int(round(n_pv * 0.75))
    for i, c in enumerate(cols):
        if i < k1:                       # cohort 1: whole record
            comm.loc[c, "commissioning_date"] = idx[0]
        elif i < k2:                     # cohort 2: commissioned in the first half of 2022
            comm.loc[c, "commissioning_date"] = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 150)))
        else:                            # cohort 3: commissioned in early 2023 (no train data)
            comm.loc[c, "commissioning_date"] = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 75)))
    comm.loc[cols[1], "decommissioning_date"] = pd.Timestamp("2023-09-01")   # cohort-1 plant leaves in 2023
    D["comm"] = comm
    act = exploitation_mask(D["df"], comm).reindex(columns=cols, fill_value=False)
    for c in cols:
        D["df"].loc[~act[c].to_numpy(), c] = 0.0       # as pv_physics_graph.load: gaps -> 0, mask in observed
        D["observed"].loc[~act[c].to_numpy(), c] = False
    D["act"] = act
    ca = analyse_cohorts(D["df"], D["phys"], cols, D["observed"], act)
    D["ca"] = ca
    C["valid"] = coverage_mask(D["df"], D["phys"], cols, D["observed"], act)
    return D, C


# ============================================================================ inverter-level (modal) fixture
def make_modal(D, seed=11, frac=0.6, n_multi=4):
    """Synthetic inverter-level channels (pv_modal_data.ModalData) for the fixture fleet.

    About ``frac`` of the plants get inverter data; ``n_multi`` of them have 2-3 inverters with
    consistent sums (sum of inverters = site power up to a 0.5 % meter noise).  AC voltage
    ~ per-site level 224-233 V + noise and a mild power-dependent rise, natural short curtailment
    events (cap of P and vac above the threshold), dcV ~ 750 V while producing, balanced phases,
    rare zero 5-min samples.  Hours without site data have no inverter data either."""
    import pv_modal_data as md
    rng = np.random.default_rng(seed)
    df = D["df"]; idx = pd.DatetimeIndex(df.index); pv = list(D["pv_cols"]); T = len(idx)
    sel = sorted(rng.choice(len(pv), max(2, int(round(frac * len(pv)))), replace=False).tolist())
    multi = set(sel[:n_multi])
    inv, site_of = {}, {}
    rated = {c: float(D["rated"][c]) * 1000.0 for c in pv}
    for j in sel:
        c = pv[j]; P = df[c].to_numpy(float)
        ni = int(rng.choice([2, 3])) if j in multi else 1
        sh = rng.dirichlet(np.full(ni, 8.0)) if ni > 1 else np.ones(1)
        v0 = rng.uniform(224, 233)
        for k in range(ni):
            noise = np.exp(rng.normal(0, 0.02, T)) if ni > 1 else np.ones(T)
            Pk = P * sh[k] * noise
            prod = np.nan_to_num(Pk) > 0.02 * rated[c] * sh[k]
            vac = v0 + 2.0 * np.nan_to_num(P) / rated[c] + rng.normal(0, 0.8, T)
            F = pd.DataFrame({
                "P": Pk, "n_obs": np.where(np.isfinite(P), 12, 0).astype(float),
                "n_zero": np.where(prod, rng.binomial(1, 0.003, T), np.where(np.isfinite(P), 12, np.nan)).astype(float),
                "cv": np.where(prod, np.abs(rng.normal(0.05, 0.02, T)), np.nan),
                "dcV": np.where(prod, 750 + rng.normal(0, 0.5, T), 0.0),
                "dc_on": np.where(prod, 1.0, 0.0),
                "vac_max": vac + np.abs(rng.normal(0.8, 0.3, T)), "vac": vac,
                "f_dev": np.abs(rng.normal(0, 0.03, T)),
                "imb": np.where(prod, np.abs(rng.normal(0.01, 0.005, T)), np.nan),
                "q": rng.normal(0, 0.01, T) * np.nan_to_num(Pk)}, index=idx)
            F.loc[~np.isfinite(P), [f for f in md.INV_FEATS if f != "n_obs"]] = np.nan
            key = f"P{j:02d} #{k + 1}"
            inv[key] = F[list(md.INV_FEATS)]; site_of[key] = c
        # natural curtailment events: 1-3 h, about 6 per year, P capped at 60 %, voltage above threshold
        day = np.flatnonzero((idx.hour >= 11) & (idx.hour <= 13) & np.isfinite(P) & (P > 0.5 * rated[c]))
        for t0 in rng.choice(day, min(len(day), 18), replace=False) if len(day) else []:
            L = int(rng.integers(1, 4))
            for k in range(ni):
                F = inv[f"P{j:02d} #{k + 1}"]
                F.iloc[t0:t0 + L, F.columns.get_loc("vac_max")] = v0 + 12 + rng.normal(0, 1)
            df.iloc[t0:t0 + L, df.columns.get_loc(c)] = np.minimum(P[t0:t0 + L], 0.6 * rated[c])
            for k in range(ni):
                F = inv[f"P{j:02d} #{k + 1}"]
                F.iloc[t0:t0 + L, F.columns.get_loc("P")] = df[c].to_numpy(float)[t0:t0 + L] * sh[k]
    M = md.ModalData(index=idx, inv=inv, site_of=site_of, rated_w=rated,
                     source=f"synthetic fixture ({len(inv)} inverters)").fit_shares()
    return M


def add_shading(D, n_sites=4, seed=21, depth=(0.5, 0.8), max_elev=(20.0, 30.0)):
    """Fixed obstruction (neighbouring building) at ``n_sites`` plants: beam blocked at low sun in an
    azimuth sector (east or west).  Modifies D["df"] in place; returns {column: (az0, az1, elev, depth)}."""
    import pv_shading as psh
    rng = np.random.default_rng(seed)
    df = D["df"]; pv = list(D["pv_cols"])
    sun = psh.sun_frame(D, df.index); el, az = sun["elev"].to_numpy(), sun["az"].to_numpy()
    bs = np.clip(psh.erbs_beam_share(psh._kt(D, df)) / 0.835, 0, 1)
    out = {}
    for j in sorted(rng.choice(len(pv), n_sites, replace=False).tolist()):
        c = pv[j]
        a0 = float(rng.choice([80.0, 230.0])); a1 = a0 + 50.0
        h = float(rng.uniform(*max_elev)); d = float(rng.uniform(*depth))
        hit = (az >= a0) & (az <= a1) & (el > 0) & (el < h)
        f = 1.0 - d * bs
        v = df[c].to_numpy(float).copy(); v[hit] = v[hit] * f[hit]
        df[c] = v
        out[c] = (a0, a1, h, d)
    return out
