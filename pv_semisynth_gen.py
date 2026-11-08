"""
Semi-synthetic data generator for the SS-PI-STGNN pipeline (independent module).

Real weather, real measurement noise and the real coupling between plants are kept:
anomalies are INJECTED into the measured power (or into the irradiance sensor) of the
real HKUST data in the TEST period only, so the training normalisation and the frozen
model are untouched.  Every injected event has an exact label (plant, start, end,
effective hours, true energy loss, expected anomaly class).

Scenarios (one per anomaly class of the model, ``pv_clustering.ANOMALY_CLASSES`` /
``pv_fuzzy.CLASS_NAMES``) + the fleet-level sensor shift and the gradual degradation:

| key           | what is injected                                         | expected class |
|---------------|----------------------------------------------------------|----------------|
| outage        | P = 0 (full daily shutdown)                              | 0 Outage       |
| clipping      | P = min(P, (1-s) * P99)                                  | 1 Clipping     |
| shading       | P *= (1-s) in a morning / evening hour window             | 2 Shading      |
| partial_loss  | P *= (1-s), one plant                                    | 3 Loss         |
| overgen       | P *= (1+s) (meter error / data outlier)                  | 4 Overgen.     |
| lowlight      | P *= (1-s) only in overcast hours (kt < 0.45)            | 5 Weather      |
| ghi_shift     | GHI_meas *= (1-s) for the whole fleet (pyranometer soiling) | 4 (P > P_exp)|
| degradation   | P *= 1 - s * (t - t0)/(t1 - t0) over the test period     | 3 Loss         |

Strengths s ~ 5, 15, 40 % (outage is 100 % by definition), durations: 1-2 steps
(noise cut-off control), one day, one week or longer.  Placement: plants from
different technical clusters, days of different weather (clear / variable / overcast
by the daily clearness index), several random placements (``seed``).  Each target
plant receives at most ONE event per quarter so that the quarterly b1 of SPS / Kalman
can be attributed to a single event.

This file only uses numpy / pandas and the dictionaries produced by
``run_pipeline.stage_data`` (D) and ``stage_clusters`` (C); nothing in the existing
pipeline imports it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:                                    # class names of the model (single source of truth)
    from pv_fuzzy import CLASS_NAMES as _CLASSES
except Exception:                       # pragma: no cover
    _CLASSES = ["Outage / zero generation", "Power limitation (clipping / curtailment)",
                "Partial shading (time of day)", "Gradual loss (soiling / degradation)",
                "Overgeneration / data spike (P > P_exp)",
                "Weather-driven deviation (low irradiance)"]
ANOMALY_CLASSES = list(_CLASSES)
IRR_COL = "Irradiance (W/m2)"


# ----------------------------------------------------------------------------- scenarios
@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    level: str                    # "plant" | "fleet" | "trend"
    expected_class: int           # index in ANOMALY_CLASSES
    full_strength_only: bool = False
    weather: Optional[str] = None  # required day category, None = any (diversified)

    @property
    def expected_name(self) -> str:
        return ANOMALY_CLASSES[self.expected_class]


SCENARIOS: Dict[str, Scenario] = {
    "outage": Scenario("outage", "Full-day outage", "plant", 0, full_strength_only=True),
    "clipping": Scenario("clipping", "Power limitation (clipping)", "plant", 1, weather="clear"),
    "shading": Scenario("shading", "Partial shading (morning / evening)", "plant", 2),
    "partial_loss": Scenario("partial_loss", "Partial power loss", "plant", 3),
    "overgen": Scenario("overgen", "Overgeneration / data spike", "plant", 4),
    "lowlight": Scenario("lowlight", "Deviation at low irradiance", "plant", 5, weather="overcast"),
    "ghi_shift": Scenario("ghi_shift", "Fleet-wide GHI shift", "fleet", 4),
    "degradation": Scenario("degradation", "Gradual degradation", "trend", 3),
}
DURATIONS = {"short": "1–2 steps", "day": "day", "week": "week+", "trend": "whole test"}
WEATHER_CATS = ("clear", "variable", "overcast")


@dataclass
class GenConfig:
    scenarios: Tuple[str, ...] = tuple(SCENARIOS)
    strengths: Tuple[float, ...] = (0.05, 0.15, 0.40)
    durations: Tuple[str, ...] = ("short", "day", "week")
    week_days: int = 7
    boundary: str = "2023-01-01"
    n_controls: int = 6
    n_degradation: int = 3
    buffer_days: int = 3            # clean days around every fleet event
    min_valid_share: float = 0.6    # plant daylight coverage on the event days
    min_plant_cov: float = 0.5      # coverage of the plant's OWN active daylight hours (train / test)
    min_test_days: int = 30         # active, scored test days required for a target / control
    warmup_h: int = 48 + 168        # STGNN cold start (window) + node calibration after commissioning
    by_cohort: bool = True          # stratify roles by (cohort, technical cluster)
    degradation_cohorts: Tuple[int, ...] = (1, 2)   # cohort 3 has no train period -> no train/test b1
    shading_windows: Tuple[Tuple[int, int], ...] = ((7, 10), (15, 18))
    lowlight_kt: float = 0.45
    clip_quantile: float = 0.99
    effect_tol: float = 0.005       # |dP| > tol * rated  ->  effective hour
    kt_clear: float = 0.6
    kt_overcast: float = 0.35
    max_tries: int = 80


@dataclass
class Plan:
    labels: pd.DataFrame
    roles: pd.Series                 # column -> target / control / degradation / unused
    cfg: GenConfig
    seed: int
    caps: Dict[str, float] = field(default_factory=dict)
    plants: Optional[pd.DataFrame] = None   # cohort, active window, coverage, eligibility (plant_info)

    def summary(self) -> pd.DataFrame:
        return (self.labels.groupby(["scenario", "duration"]).size().unstack(fill_value=0))

    def roles_by_cohort(self) -> pd.DataFrame:
        """Number of plants per cohort and role (eligible plants only)."""
        if self.plants is None:
            return pd.DataFrame()
        t = pd.crosstab(self.plants["cohort"].map(COHORT_TITLES), self.plants["role"])
        return t.rename_axis(index="Cohort", columns="Role")


@dataclass
class Realisation:
    df: pd.DataFrame
    phys: Dict[str, np.ndarray]
    labels: pd.DataFrame             # plan labels + effective interval / true loss
    roles: pd.Series
    inj_mask: pd.DataFrame           # (T x pv) True where the power was modified
    fleet_mask: pd.Series            # (T,) True where the irradiance sensor was modified
    included: Tuple[str, ...]
    t_cell: Optional[np.ndarray] = None   # recomputed cell temperature (sensor shift), else None
    plants: Optional[pd.DataFrame] = None  # plant_info of the plan (cohort, active window, role)


# ----------------------------------------------------------------------------- helpers
def _short(c: str) -> str:
    return c.replace("power(W)_", "")


def _rated_w(D: dict, c: str, P: np.ndarray) -> float:
    kw = (D.get("rated") or {}).get(c) or 0.0
    if kw > 0:
        return kw * 1000.0
    v = P[np.isfinite(P)]
    return float(np.quantile(v, 0.99)) if len(v) else 1000.0


def daily_weather(D: dict, cfg: GenConfig) -> pd.DataFrame:
    """Daily clearness (mean kt over daylight) and a weather category per day."""
    kt = D["kt"]
    ghi = pd.Series(D["phys"]["ghi"], index=D["df"].index)
    day = ghi > 50
    k = kt.where(day).groupby(kt.index.normalize()).mean()
    cat = np.where(k >= cfg.kt_clear, "clear", np.where(k < cfg.kt_overcast, "overcast", "variable"))
    return pd.DataFrame({"kt": k, "category": np.where(k.isna(), "no data", cat)})


def _modify(scn: str, P: np.ndarray, s: float, hours: np.ndarray, kt: np.ndarray, ghi: np.ndarray,
            cap: float, window: Optional[Tuple[int, int]] = None, frac: Optional[np.ndarray] = None) -> np.ndarray:
    """Return the modified copy of a power slice for one plant-level scenario."""
    out = P.copy()
    if scn == "partial_loss":
        out = P * (1 - s)
    elif scn == "outage":
        out = np.where(np.isfinite(P), 0.0, P)
    elif scn == "clipping":
        out = np.where(np.isfinite(P), np.minimum(P, (1 - s) * cap), P)
    elif scn == "shading":
        a, b = window
        m = (hours >= a) & (hours < b)
        out = np.where(m, P * (1 - s), P)
    elif scn == "overgen":
        out = P * (1 + s)
    elif scn == "lowlight":
        m = (np.nan_to_num(kt, nan=1.0) < 0.45) & (ghi > 50)
        out = np.where(m, P * (1 - s), P)
    elif scn == "degradation":
        out = P * (1 - s * frac)
    else:
        raise ValueError(scn)
    return out


# ----------------------------------------------------------------------------- planning
COHORT_TITLES = {1: "Cohort 1 (2021–2023)", 2: "Cohort 2 (from 2022)", 3: "Cohort 3 (2023 only)",
                 0: "Insufficient data"}


def plant_cohorts(D: dict, C: dict, pv: Sequence[str]) -> pd.Series:
    """Cohort of every plant: pipeline's data-driven assignment ``D["ca"].cohorts`` (pv_cohorts);
    fallback — first calendar year with valid daylight data (1 = first year, 2 = second, 3 = later)."""
    ca = D.get("ca")
    if ca is not None and hasattr(ca, "cohorts") and "cohort" in ca.cohorts:
        return ca.cohorts["cohort"].reindex(pv).fillna(0).astype(int)
    valid = C["valid"]
    years = sorted(set(valid.index.year))
    out = {}
    for c in pv:
        yv = valid[c].groupby(valid.index.year).sum()
        first = [y for y in years if yv.get(y, 0) >= 200]
        out[c] = 0 if not first else min(years.index(first[0]) + 1, 3)
    return pd.Series(out, dtype=int)


def plant_info(D: dict, C: dict, cfg: Optional[GenConfig] = None) -> pd.DataFrame:
    """Per plant: cohort, exploitation window (commissioning / decommissioning from ``D["comm"]``,
    else first / last valid hour), first scored hour of the test period (after the STGNN cold start +
    calibration), coverage of the plant's own active daylight hours in train / test and eligibility.
    Coverage is relative to the active period, so plants commissioned or decommissioned within the
    record (cohorts 2 and 3) are not discarded."""
    cfg = cfg or GenConfig()
    df, pv = D["df"], list(D["pv_cols"])
    idx = df.index
    b = pd.Timestamp(cfg.boundary)
    valid = C["valid"]
    day = np.asarray(D["phys"]["ghi"]) > 200
    test = np.asarray(idx >= b)
    coh = plant_cohorts(D, C, pv)
    act = None
    if D.get("comm") is not None:
        try:
            from pv_physics_graph import exploitation_mask
            act = exploitation_mask(df, D["comm"]).reindex(columns=pv, fill_value=False)
        except Exception:
            act = None
    rows = []
    for c in pv:
        v = valid[c].to_numpy(bool)
        a = act[c].to_numpy(bool) if act is not None else v.copy()
        if not a.any():
            a = v.copy()
        if a.any():
            first, last = idx[np.argmax(a)], idx[len(a) - 1 - np.argmax(a[::-1])]
        else:
            first = last = pd.NaT
        scored_from = max(b, first + pd.Timedelta(hours=cfg.warmup_h)) if pd.notna(first) else pd.NaT
        tr_d, te_d = a & day & ~test, a & day & test
        tr = v[tr_d].mean() if tr_d.any() else np.nan
        te = v[te_d].mean() if te_d.any() else np.nan
        sc = np.asarray(idx >= scored_from) & a if pd.notna(scored_from) else np.zeros(len(idx), bool)
        n_days = int(len(set(idx[sc & day & v].normalize())))
        ok = bool(np.nan_to_num(te) >= cfg.min_plant_cov and n_days >= cfg.min_test_days)
        has_train = bool(tr_d.sum() >= 24 * 30 * 0.4 and np.nan_to_num(tr) >= cfg.min_plant_cov)
        rows.append(dict(column=c, plant=_short(c), cohort=int(coh.get(c, 0)), active_from=first, active_to=last,
                         scored_from=scored_from, train_cov=tr, test_cov=te, scored_test_days=n_days,
                         has_train=has_train, eligible=ok,
                         degradation_ok=bool(ok and has_train and int(coh.get(c, 0)) in cfg.degradation_cohorts)))
    return pd.DataFrame(rows).set_index("column")


def eligible_plants(D: dict, C: dict, cfg: GenConfig) -> List[str]:
    info = plant_info(D, C, cfg)
    return info.index[info["eligible"]].tolist()


def _round_robin(cols: Sequence[str], clusters: pd.Series, rng) -> List[str]:
    groups = {}
    for c in cols:
        groups.setdefault(clusters.get(c, -1), []).append(c)
    for g in groups.values():
        rng.shuffle(g)
    keys = list(groups); rng.shuffle(keys)
    out = []
    while any(groups[k] for k in keys):
        for k in keys:
            if groups[k]:
                out.append(groups[k].pop())
    return out


def plan_semisynthetic(D: dict, C: dict, seed: int = 0, cfg: Optional[GenConfig] = None) -> Plan:
    """Random placement of all scenario x strength x duration events (exact labels)."""
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(seed)
    df, pv = D["df"], list(D["pv_cols"])
    idx = df.index
    b = pd.Timestamp(cfg.boundary)
    test_days = pd.DatetimeIndex(sorted(set(idx[idx >= b].normalize())))
    if len(test_days) < 30:
        raise ValueError("test period too short for semi-synthetic injection")
    W = daily_weather(D, cfg)
    ghi = D["phys"]["ghi"]; kt = D["kt"].reindex(idx).to_numpy(float)
    hours = idx.hour.to_numpy()
    valid = C["valid"]
    tech = C["tech"].labels if hasattr(C.get("tech"), "labels") else pd.Series(0, index=pv)
    day_pos = {d: np.flatnonzero(idx.normalize() == d) for d in test_days}

    # ---- plant roles: stratified by (cohort, technical cluster)
    info = plant_info(D, C, cfg)
    elig = info.index[info["eligible"]].tolist()
    strata = (pd.Series({c: (int(info.loc[c, "cohort"]), int(tech.get(c, -1))) for c in elig})
              if cfg.by_cohort else tech)
    order = _round_robin(elig, strata, rng)
    plant_scn = [s for s in cfg.scenarios if SCENARIOS[s].level == "plant"]
    specs = []
    for s in plant_scn:
        for st in ((1.0,) if SCENARIOS[s].full_strength_only else cfg.strengths):
            for du in cfg.durations:
                specs.append((s, float(st), du))
    quarters = sorted(set(test_days.to_period("Q")))
    quarters = [q for q in quarters if sum(test_days.to_period("Q") == q) >= 20]
    n_deg = cfg.n_degradation if "degradation" in cfg.scenarios else 0
    n_tg = int(np.ceil(len(specs) / max(len(quarters), 1)))
    n_ctrl = cfg.n_controls
    if len(order) < n_deg + n_tg + n_ctrl:
        n_ctrl = max(2, len(order) - n_deg - n_tg)
    if len(order) < n_deg + n_tg + n_ctrl:
        n_tg = max(1, len(order) - n_deg - n_ctrl)
    if cfg.by_cohort:                   # at least one control plant per cohort
        n_coh = info.loc[elig, "cohort"].nunique() if elig else 0
        if n_ctrl < n_coh:
            n_ctrl = n_coh
            n_tg = max(1, min(n_tg, len(order) - n_deg - n_ctrl))
    deg_pool = [c for c in order if info.loc[c, "degradation_ok"]]
    deg_cols = _round_robin(deg_pool, info.loc[deg_pool, "cohort"], rng)[:n_deg] if deg_pool else []
    rest = [c for c in order if c not in deg_cols]
    if cfg.by_cohort:                   # every cohort gets control plants (false alarms per cohort)
        coh_rest = info.loc[rest, "cohort"]
        groups = {k: [c for c in rest if coh_rest[c] == k] for k in sorted(set(coh_rest))}
        ctrl_cols = []
        while len(ctrl_cols) < n_ctrl and any(len(g) > 1 for g in groups.values()):
            for k in groups:            # keep >= 1 plant per cohort for the targets
                if len(ctrl_cols) < n_ctrl and len(groups[k]) > 1:
                    ctrl_cols.append(groups[k].pop(0))
    else:
        ctrl_cols = rest[:n_ctrl]
    tg_cols = [c for c in rest if c not in ctrl_cols][:n_tg]
    roles = pd.Series("unused", index=pv, name="role")
    roles[deg_cols] = "degradation"; roles[ctrl_cols] = "control"; roles[tg_cols] = "target"

    caps = {}
    test_mask = idx >= b
    for c in pv:
        P = df[c].to_numpy(float)[~test_mask]
        P = P[np.isfinite(P) & (P > 0)]
        if len(P) < 200:                    # cohort 3: no train data -> first scored test month
            r = info.loc[c]
            m = np.asarray((idx >= r["scored_from"]) & (idx < r["scored_from"] + pd.Timedelta(days=30))) \
                if pd.notna(r["scored_from"]) else np.zeros(len(idx), bool)
            P = df[c].to_numpy(float)[m]; P = P[np.isfinite(P) & (P > 0)]
        caps[c] = float(np.quantile(P, cfg.clip_quantile)) if len(P) else 0.0

    rows = []
    eid = 0
    blocked = np.zeros(len(idx), bool)        # hours reserved by fleet events (+ buffer)

    def fleet_ok(days: pd.DatetimeIndex) -> bool:
        pos = np.concatenate([day_pos[d] for d in days if d in day_pos]) if len(days) else np.array([], int)
        if len(pos) == 0 or blocked[pos].any():
            return False
        dl = ghi[pos] > 200
        return dl.sum() >= 2

    # ---- fleet events (sensor shift)
    if "ghi_shift" in cfg.scenarios:
        fspecs = [(float(st), du) for st in cfg.strengths for du in cfg.durations]
        rng.shuffle(fspecs)
        for k, (st, du) in enumerate(fspecs):
            want = WEATHER_CATS[k % 3]
            ndays = 1 if du in ("short", "day") else cfg.week_days
            for _ in range(cfg.max_tries * 2):
                d0 = test_days[rng.integers(0, max(len(test_days) - ndays, 1))]
                days = pd.date_range(d0, periods=ndays, freq="D")
                if not all(d in day_pos for d in days):
                    continue
                if W["category"].get(d0, "") != want and _ < cfg.max_tries:
                    continue
                if not fleet_ok(days):
                    continue
                pos = np.concatenate([day_pos[d] for d in days])
                if du == "short":
                    cand = pos[(ghi[pos] > 300)]
                    if not len(cand):
                        continue
                    st_pos = int(rng.choice(cand)); L = int(rng.integers(1, 3))
                    p0, p1 = st_pos, min(st_pos + L - 1, len(idx) - 1)
                else:
                    p0, p1 = int(pos[0]), int(pos[-1])
                bd = pd.Timedelta(days=cfg.buffer_days)
                blk = (idx >= idx[p0].normalize() - bd) & (idx <= idx[p1].normalize() + bd + pd.Timedelta(hours=23))
                blocked |= np.asarray(blk)
                rows.append(dict(event_id=eid, scenario="ghi_shift", level="fleet", column=None, plant="*all*",
                                 strength=st, duration=du, start=idx[p0], end=idx[p1], n_steps=p1 - p0 + 1,
                                 weather=W["category"].get(idx[p0].normalize(), "no data"),
                                 tech_cluster=-1, window=None, cap=np.nan,
                                 quarter=str(idx[p0].to_period("Q")), cohort=-1))
                eid += 1
                break

    # ---- plant-level events: one per (target plant, quarter)
    def _scored_days(c):
        r = info.loc[c]
        return [d for d in test_days if d >= r["scored_from"].normalize() + pd.Timedelta(days=1)
                and d + pd.Timedelta(days=1) <= r["active_to"]]
    sdays = {c: _scored_days(c) for c in tg_cols}
    slots = [(c, q) for c in tg_cols for q in quarters
             if sum(d.to_period("Q") == q for d in sdays[c]) >= 20]
    rng.shuffle(slots)
    rng.shuffle(specs)
    shared = [False] * len(slots)
    while len(slots) < len(specs):         # not enough (plant, quarter) slots -> reuse, flagged
        extra = slots[: len(specs) - len(slots)]
        slots = slots + extra; shared = shared + [True] * len(extra)
    occupied = {c: np.zeros(len(idx), bool) for c in tg_cols}    # event days +- buffer per plant
    bd = pd.Timedelta(days=cfg.buffer_days)
    for (scn, st, du), (c, q), sh in zip(specs, slots, shared):
        j = pv.index(c)
        sc = SCENARIOS[scn]
        P = df[c].to_numpy(float)
        rated = _rated_w(D, c, P)
        qdays = [d for d in sdays[c] if d.to_period("Q") == q]
        ndays = 1 if du in ("short", "day") else cfg.week_days
        want = sc.weather or WEATHER_CATS[int(rng.integers(0, 3))]
        win = cfg.shading_windows[int(rng.integers(0, len(cfg.shading_windows)))] if scn == "shading" else None
        placed = False
        for t in range(cfg.max_tries * 3):
            if len(qdays) <= ndays:
                break
            d0 = qdays[int(rng.integers(0, len(qdays) - ndays))]
            days = pd.date_range(d0, periods=ndays, freq="D")
            if not all(d in day_pos for d in days):
                continue
            pos = np.concatenate([day_pos[d] for d in days])
            if blocked[pos].any() or occupied[c][pos].any():
                continue
            if t < cfg.max_tries and W["category"].get(d0, "") != want:
                continue
            dl = ghi[pos] > 200
            if dl.sum() == 0 or valid[c].to_numpy(bool)[pos][dl].mean() < cfg.min_valid_share:
                continue
            new = _modify(scn, P[pos], st, hours[pos], kt[pos], ghi[pos], caps[c], win)
            eff = np.abs(np.nan_to_num(new - P[pos])) > cfg.effect_tol * rated
            if du == "short":
                cand = pos[eff]
                if not len(cand):
                    continue
                st_pos = int(rng.choice(cand)); L = int(rng.integers(1, 3))
                p0, p1 = st_pos, min(st_pos + L - 1, len(idx) - 1)
            else:
                if eff.sum() < 2:
                    continue
                p0, p1 = int(pos[0]), int(pos[-1])
            rows.append(dict(event_id=eid, scenario=scn, level="plant", column=c, plant=_short(c), strength=st,
                             duration=du, start=idx[p0], end=idx[p1], n_steps=p1 - p0 + 1,
                             weather=W["category"].get(idx[p0].normalize(), "no data"),
                             tech_cluster=int(tech.get(c, -1)), window=win, cap=caps[c],
                             quarter=str(q), shared_quarter=sh, cohort=int(info.loc[c, "cohort"]),
                             days_since_commissioning=(idx[p0] - info.loc[c, "active_from"]) / pd.Timedelta(days=1)))
            occupied[c] |= np.asarray((idx >= days[0] - bd) & (idx <= days[-1] + bd + pd.Timedelta(hours=23)))
            eid += 1
            placed = True
            break
        if not placed:
            rows.append(dict(event_id=eid, scenario=scn, level="plant", column=c, plant=_short(c), strength=st,
                             duration=du, start=pd.NaT, end=pd.NaT, n_steps=0, weather="not placed",
                             tech_cluster=int(tech.get(c, -1)), window=win, cap=caps[c], quarter=str(q),
                             shared_quarter=sh, cohort=int(info.loc[c, "cohort"])))
            eid += 1

    # ---- gradual degradation over the whole test period
    if n_deg:
        sts = list(cfg.strengths) * int(np.ceil(n_deg / len(cfg.strengths)))
        rng.shuffle(sts)
        for c, st in zip(deg_cols, sts):
            # the ramp reaches s at the plant's LAST active hour (decommissioning inside the test)
            t0 = max(idx[test_mask][0], info.loc[c, "active_from"])
            t1 = min(idx[-1], info.loc[c, "active_to"])
            rows.append(dict(event_id=eid, scenario="degradation", level="trend", column=c, plant=_short(c),
                             strength=float(st), duration="trend", start=t0, end=t1,
                             n_steps=int(((idx >= t0) & (idx <= t1)).sum()), weather="all",
                             tech_cluster=int(tech.get(c, -1)), window=None, cap=caps[c], quarter="whole test",
                             cohort=int(info.loc[c, "cohort"]),
                             days_since_commissioning=(t0 - info.loc[c, "active_from"]) / pd.Timedelta(days=1)))
            eid += 1
    labels = pd.DataFrame(rows)
    labels["scenario_title"] = labels["scenario"].map(lambda s: SCENARIOS[s].title)
    labels["expected_class"] = labels["scenario"].map(lambda s: SCENARIOS[s].expected_name)
    labels["seed"] = seed
    labels["cohort_title"] = labels["cohort"].map(lambda k: COHORT_TITLES.get(int(k), "whole fleet")
                                                 if pd.notna(k) else "whole fleet")
    info = info.assign(role=roles.reindex(info.index).values)
    return Plan(labels=labels, roles=roles, cfg=cfg, seed=seed, caps=caps, plants=info)


# ----------------------------------------------------------------------------- injection
def apply_plan(D: dict, C: dict, plan: Plan, include: Optional[Iterable[str]] = None,
               recompute_physics: bool = True) -> Realisation:
    """Inject the planned events into a copy of the real data.

    include : scenario keys to inject (default: all).  Use e.g. all but "ghi_shift" to
              evaluate b1 estimators without the fleet-level sensor shift.
    Physics (P_exp) is recomputed with ``D["builder"].physics_tensors`` only when the
    irradiance sensor is modified."""
    cfg = plan.cfg
    include = tuple(include) if include is not None else tuple(cfg.scenarios)
    df0 = D["df"]
    pv = list(D["pv_cols"])
    idx = df0.index
    df = df0.copy()
    P0 = df0[pv].to_numpy(float)
    P = P0.copy()
    G0 = df0[IRR_COL].to_numpy(float) if IRR_COL in df0 else np.asarray(D["phys"]["ghi"], float)
    G = G0.copy()
    ghi = D["phys"]["ghi"]; kt = D["kt"].reindex(idx).to_numpy(float)
    hours = idx.hour.to_numpy()
    lab = plan.labels[plan.labels["scenario"].isin(include) & plan.labels["start"].notna()].copy()
    fleet = np.zeros(len(idx), bool)
    for _, r in lab.iterrows():
        p0 = idx.get_loc(r["start"]); p1 = idx.get_loc(r["end"])
        sl = slice(p0, p1 + 1)
        if r["scenario"] == "ghi_shift":
            G[sl] = G[sl] * (1 - r["strength"])
            fleet[sl] = True
            continue
        j = pv.index(r["column"])
        frac = None
        if r["scenario"] == "degradation":
            tt = idx[sl]
            frac = np.asarray((tt - tt[0]) / max(tt[-1] - tt[0], pd.Timedelta(hours=1)), float)
        P[sl, j] = _modify(r["scenario"], P[sl, j], r["strength"], hours[sl], kt[sl], ghi[sl],
                           plan.caps.get(r["column"], np.nanmax(P0[:, j])), r["window"], frac)
    df[pv] = P
    phys = D["phys"]
    t_cell = None
    if fleet.any():
        if IRR_COL in df:
            df[IRR_COL] = G
        if recompute_physics and D.get("builder") is not None:
            graph = C.get("graph_mv") or D.get("graph_base")
            phys = D["builder"].physics_tensors(df, graph)
            t_cell = _recompute_t_cell(D, df)
        else:
            phys = dict(D["phys"]); phys["ghi"] = G
    inj = np.abs(np.nan_to_num(P - P0)) > 0
    # effective interval and true loss for every label
    rated = np.array([_rated_w(D, c, P0[:, j]) for j, c in enumerate(pv)])
    pe0, pe1 = D["phys"]["p_expected"], phys["p_expected"]
    eff_rows = []
    for _, r in lab.iterrows():
        p0 = idx.get_loc(r["start"]); p1 = idx.get_loc(r["end"])
        sl = np.arange(p0, p1 + 1)
        if r["scenario"] == "ghi_shift":
            m = (G0[sl] > 20) & (np.abs(G[sl] - G0[sl]) > 0)
            pos = sl[m]
            if len(pos):
                with np.errstate(all="ignore"):
                    bias = (np.nansum(pe1[pos]) / max(np.nansum(pe0[pos]), 1e-9) - 1) * 100
            else:
                bias = np.nan
            eff_rows.append(dict(event_id=r["event_id"], eff_start=idx[pos[0]] if len(pos) else pd.NaT,
                                 eff_end=idx[pos[-1]] if len(pos) else pd.NaT, n_eff_h=int(len(pos)),
                                 true_loss_pct=np.nan, pexp_bias_pct=bias))
            continue
        j = pv.index(r["column"])
        m = np.abs(np.nan_to_num(P[sl, j] - P0[sl, j])) > cfg.effect_tol * rated[j]
        pos = sl[m]
        s0, s1 = np.nansum(P0[pos, j]), np.nansum(P[pos, j])
        eff_rows.append(dict(event_id=r["event_id"], eff_start=idx[pos[0]] if len(pos) else pd.NaT,
                             eff_end=idx[pos[-1]] if len(pos) else pd.NaT, n_eff_h=int(len(pos)),
                             true_loss_pct=(1 - s1 / s0) * 100 if s0 > 0 else np.nan, pexp_bias_pct=0.0))
    lab = lab.merge(pd.DataFrame(eff_rows), on="event_id", how="left") if eff_rows else lab
    lab["effective"] = lab.get("n_eff_h", pd.Series(0, index=lab.index)).fillna(0) > 0
    return Realisation(df=df, phys=phys, labels=lab.reset_index(drop=True), roles=plan.roles,
                       inj_mask=pd.DataFrame(inj, index=idx, columns=pv),
                       fleet_mask=pd.Series(fleet, index=idx), included=include, t_cell=t_cell,
                       plants=plan.plants)


def _recompute_t_cell(D: dict, df: pd.DataFrame) -> Optional[np.ndarray]:
    """Cell temperature from the (shifted) irradiance, exactly as in run_pipeline.stage_data;
    None if the physical prior / Brick model are not available."""
    try:
        prior = D["builder"].prior
        brick, mapping = D["brick"], D["mapping"]
    except (KeyError, AttributeError):
        return None
    times = pd.DatetimeIndex(df.index)
    age = (times - times[0]).days.to_numpy(float) / 365.25
    out = np.zeros((len(df), len(D["pv_cols"])))
    for j, c in enumerate(D["pv_cols"]):
        _, aux = prior.expected_power(df[IRR_COL].to_numpy(float), times, df["Temp (Degree Celsius)"].to_numpy(float),
                                      df["wind_speed"].to_numpy(float), brick.sites[mapping[c]], 8.0, age)
        out[:, j] = aux["t_cell"]
    return out


def scenario_table() -> pd.DataFrame:
    """Human-readable description of the scenarios (for the notebook / paper)."""
    desc = {
        "outage": "P = 0 during the event",
        "clipping": "P = min(P, (1−s)·P99) — limitation of the upper part of the daily profile",
        "shading": "P ·= (1−s) in the 7–10 h or 15–18 h window",
        "partial_loss": "P ·= (1−s) for one plant",
        "overgen": "P ·= (1+s) — meter error / data spike",
        "lowlight": "P ·= (1−s) only in overcast hours (kt < 0.45)",
        "ghi_shift": "GHI_meas ·= (1−s) for the whole fleet — pyranometer soiling",
        "degradation": "P ·= 1 − s·(t − t₀)/(t₁ − t₀) — linear growth of losses up to s at the end of the test",
    }
    return pd.DataFrame([dict(Scenario=s.title, Key=k, Level={"plant": "plant", "fleet": "fleet",
                                                                  "trend": "trend"}[s.level],
                              Injection=desc[k], **{"Expected class": s.expected_name})
                         for k, s in SCENARIOS.items()])
