"""Section 14: does the STGNN add value over the PR threshold rule?  Checks of conditions 2-4.

Condition 1 (section 12): STGNN better than the PR rule at an equal false-alarm budget.
This module checks the remaining conditions on the SAME frozen models and the same placements:

* **Condition 2 — fusion.**  "PR rule OR STGNN" (each thresholded at its own per-plant
  quantile q, the same q for both; pre-specified, no tuning on the test) against the PR rule
  alone at an equal alarm budget; paired bootstrap of the recall difference over events.
* **Condition 3 — where the alarms go.**  At an equal TOTAL alarm budget on the clean
  (non-injected) test of all plants, alarm rates by weather regime, season, cohort and by the
  plant's physics-model mismatch (IQR of P/P_exp on the training period).  A method is better
  here if it spends fewer alarms in the difficult strata (overcast, winter, poorly modelled plants).
* **Condition 4 — cases where P/P_exp is weak.**
  (a) cold start: recall by cohort and by days after commissioning;
  (b) additional scenarios injected on the same target plants / windows as section 10
      (independent generator, ``HARD_SCENARIOS``): time shift of the power profile (energy
      preserved), stuck meter (flat line), small sustained loss (3-8 %), multiplicative noise
      (mean preserved);
  (c) sensor vs plant: during the fleet-wide GHI shift (sensor fault) — share of plants that
      alarm; for plant-level events — share wrongly flagged as systemic (>= 50 % of plants).

Main comparison (added): **STGNN vs PR+ at an equal alarm budget**.  ``prp`` = PR+ (pv_prplus; the
same detector as in sections 15, 16 and 18) and ``fusion_prp`` = PR+ OR STGNN-daylight are evaluated on
the same placements, budgets and events; conditions 1+-4+ repeat the pre-specified criteria with PR+
as the reference.

Methods: ``pr`` (PR rule of section 12), ``stgnn`` (AFTER detector of section 12: eta_hat
med7d, min_len 2, rule "P ~ 0 at GHI > 200"), ``fusion`` (OR of both).  Equal budget = the same
number of alarm events per 1000 scored hours on the clean test of ALL plants (computed for a
fine q grid without labels; the q with the largest rate not above the budget is used).

Independent of the main pipeline; uses the raw outputs cached by section 12.

Usage (notebook, after sections 10 and 12)::

    import pv_stgnn_value as sv
    val_res = sv.run_value_checks(D, C, ss_models, ss_res=ss_res, fixes_res=fx_res)
    sv.show(val_res); sv.plot_all(val_res); sv.export_latex(val_res)
"""
from __future__ import annotations

import copy
import hashlib
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import pv_semisynth_eval as sse
import pv_semisynth_gen as ssg
import pv_semisynth_sensitivity as sss
import pv_fixes_eval as fx
import pv_prplus as prp

METHODS = ("pr", "prp", "stgnn", "stgnn_day", "fusion", "fusion_prp")     # shown in all tables
ALL_METHODS = METHODS + ("fusion_25", "fusion_75", "fusion_prp_25", "fusion_prp_75")   # budget-split sensitivity
# fusion -> (rule component, share of the budget given to STGNN-daylight)
FUSION_ALPHA = {"fusion": ("pr", 0.5), "fusion_25": ("pr", 0.25), "fusion_75": ("pr", 0.75),
                "fusion_prp": ("prp", 0.5), "fusion_prp_25": ("prp", 0.25), "fusion_prp_75": ("prp", 0.75)}
RULES = ("pr", "prp")
METHOD_LABEL = {"pr": "PR threshold rule", "prp": "PR+ (hour x month, peers, rolling)",
                "stgnn": "STGNN (S4 detector)", "stgnn_day": "STGNN (S4, daylight alarms only)",
                "fusion": "Fusion: PR OR STGNN-daylight (budget 50/50)",
                "fusion_25": "Fusion: PR OR STGNN-daylight (budget 75/25)",
                "fusion_75": "Fusion: PR OR STGNN-daylight (budget 25/75)",
                "fusion_prp": "Fusion: PR+ OR STGNN-daylight (budget 50/50)",
                "fusion_prp_25": "Fusion: PR+ OR STGNN-daylight (budget 75/25)",
                "fusion_prp_75": "Fusion: PR+ OR STGNN-daylight (budget 25/75)"}
SHORT = {"pr": "PR", "prp": "PR+", "stgnn": "STGNN", "stgnn_day": "STGNN-day", "fusion": "Fusion",
         "fusion_25": "Fusion 75/25", "fusion_75": "Fusion 25/75", "fusion_prp": "Fusion+",
         "fusion_prp_25": "Fusion+ 75/25", "fusion_prp_75": "Fusion+ 25/75"}
COLORS = {"pr": "#ff7f0e", "prp": "#8c564b", "stgnn": "#98df8a", "stgnn_day": "#2ca02c", "fusion": "#1f77b4",
          "fusion_prp": "#9467bd"}

HARD_SCENARIOS = {
    "time_shift": dict(title="Time shift of the profile (energy preserved)", strengths=(1.0, 2.0),
                       unit="h", desc="P(t) := P(t − s h) — timestamp / clock error"),
    "stuck": dict(title="Stuck meter (flat line)", strengths=(1.0,), unit="-",
                  desc="P held at its value of the first sunny hour (GHI > 200) of the event"),
    "small_loss": dict(title="Small sustained loss (3–8 %)", strengths=(0.03, 0.05, 0.08), unit="fraction",
                       desc="P ·= (1 − s) — below the hourly scatter of P/P_exp"),
    "noise": dict(title="Multiplicative noise (mean preserved)", strengths=(0.10, 0.20), unit="fraction",
                  desc="P ·= 1 + s·ε, ε ~ N(0, 1) clipped to ±2.5 — unstable inverter / MPPT"),
}
SEASONS = {12: "DJF (winter)", 1: "DJF (winter)", 2: "DJF (winter)", 3: "MAM (spring)", 4: "MAM (spring)",
           5: "MAM (spring)", 6: "JJA (summer)", 7: "JJA (summer)", 8: "JJA (summer)", 9: "SON (autumn)",
           10: "SON (autumn)", 11: "SON (autumn)"}
DIFFICULT = r"rain|rainy|cloud|overcast|DJF|^high P/P_exp"   # word boundaries: "rain" must not match "training"
AGE_BINS = [(-np.inf, 30, "0–30 d"), (30, 90, "30–90 d"), (90, 365, "90–365 d"), (365, np.inf, "> 365 d")]


def _log(*a):
    print(time.strftime("%H:%M:%S"), "[value]", *a, flush=True)


# ============================================================================ hard scenarios
def plan_hard(base_plan: "ssg.Plan", seed: int = 0) -> pd.DataFrame:
    """Re-use the plant-level day / week events of a section-10 plan (same targets, windows,
    eligibility, cohorts) and assign the additional scenarios round-robin."""
    lab = base_plan.labels
    lab = lab[(lab["level"] == "plant") & lab["duration"].isin(["day", "week"]) & lab["start"].notna()].copy()
    lab = lab.sort_values("event_id").reset_index(drop=True)
    rng = np.random.default_rng(10_000 + seed)
    keys = list(HARD_SCENARIOS)
    scn, st = [], []
    for i in range(len(lab)):
        k = keys[i % len(keys)]
        scn.append(k); st.append(float(rng.choice(HARD_SCENARIOS[k]["strengths"])))
    lab["scenario"] = scn; lab["strength"] = st
    lab["expected_class"] = np.nan
    lab["window"] = None
    return lab


def apply_hard(D: dict, base_plan: "ssg.Plan", labels: pd.DataFrame, seed: int = 0) -> "ssg.Realisation":
    df0 = D["df"]; pv = list(D["pv_cols"]); idx = df0.index
    P0 = df0[pv].to_numpy(float); P = P0.copy()
    G = df0[ssg.IRR_COL].to_numpy(float) if ssg.IRR_COL in df0 else np.asarray(D["phys"]["ghi"], float)
    rng = np.random.default_rng(20_000 + seed)
    for r in labels.itertuples():
        j = pv.index(r.column); p0 = idx.get_loc(r.start); p1 = idx.get_loc(r.end)
        sl = slice(p0, p1 + 1); x = P0[sl, j]
        if r.scenario == "time_shift":
            d = int(round(r.strength)); src = P0[max(p0 - d, 0):p1 + 1 - d, j]
            y = x.copy(); y[len(x) - len(src):] = src
        elif r.scenario == "stuck":
            g = G[sl]; ok = np.flatnonzero((np.nan_to_num(g) > 200) & np.isfinite(x))
            y = x.copy()
            if len(ok):
                v = x[ok[0]]; y[ok[0]:] = np.where(np.isfinite(x[ok[0]:]), v, x[ok[0]:])
        elif r.scenario == "small_loss":
            y = x * (1 - r.strength)
        elif r.scenario == "noise":
            e = np.clip(rng.standard_normal(len(x)), -2.5, 2.5)
            y = np.maximum(x * (1 + r.strength * e), 0.0)
        else:
            raise ValueError(r.scenario)
        P[sl, j] = y
    df = df0.copy(); df[pv] = P
    rated = np.array([ssg._rated_w(D, c, P0[:, j]) for j, c in enumerate(pv)])
    tol = base_plan.cfg.effect_tol
    rows = []
    for r in labels.itertuples():
        j = pv.index(r.column); p0 = idx.get_loc(r.start); p1 = idx.get_loc(r.end)
        sl = np.arange(p0, p1 + 1)
        m = np.abs(np.nan_to_num(P[sl, j] - P0[sl, j])) > tol * rated[j]
        pos = sl[m]; s0, s1 = np.nansum(P0[pos, j]), np.nansum(P[pos, j])
        rows.append(dict(event_id=r.event_id, eff_start=idx[pos[0]] if len(pos) else pd.NaT,
                         eff_end=idx[pos[-1]] if len(pos) else pd.NaT, n_eff_h=int(len(pos)),
                         true_loss_pct=(1 - s1 / s0) * 100 if s0 > 0 else np.nan, pexp_bias_pct=0.0))
    lab = labels.drop(columns=[c for c in ("eff_start", "eff_end", "n_eff_h", "true_loss_pct", "pexp_bias_pct",
                                           "effective") if c in labels]).merge(pd.DataFrame(rows), on="event_id")
    lab["effective"] = lab["n_eff_h"].fillna(0) > 0
    inj = np.abs(np.nan_to_num(P - P0)) > 0
    return ssg.Realisation(df=df, phys=D["phys"], labels=lab.reset_index(drop=True), roles=base_plan.roles,
                           inj_mask=pd.DataFrame(inj, index=idx, columns=pv),
                           fleet_mask=pd.Series(False, index=idx), included=tuple(HARD_SCENARIOS),
                           t_cell=None, plants=base_plan.plants)


def _labels_sig(reals: Dict[object, "ssg.Realisation"]) -> str:
    h = hashlib.sha256()
    for k in sorted(reals, key=str):
        L = reals[k].labels[["event_id", "column", "scenario", "strength", "start", "end"]]
        h.update(str(k).encode()); h.update(pd.util.hash_pandas_object(L.astype(str), index=False).values.tobytes())
    return h.hexdigest()[:12]


def collect_hard(models: Dict[int, "sse.FrozenSTGNN"], reals_h: Dict[str, "ssg.Realisation"], cache_dir: str,
                 batch_size: int = 64, use_cache: bool = True) -> Dict[int, Dict[str, dict]]:
    """Forward passes of the frozen models on the additional realisations only."""
    os.makedirs(cache_dir, exist_ok=True)
    sig = _labels_sig(reals_h)
    out = {}
    for ms, mdl in models.items():
        fp = mdl.fingerprint()
        f = os.path.join(cache_dir, f"raw_hard_{ms}_{fp}_{sig}.pkl")
        if use_cache and os.path.exists(f):
            with open(f, "rb") as fh:
                out[ms] = pickle.load(fh)
            _log(f"model {ms}: additional scenarios from cache"); continue
        _log(f"model {ms}: forward passes on {len(reals_h)} additional realisations")
        inj = {}
        for k, real in reals_h.items():
            p = mdl.build(real.df, real.phys)
            inj[k] = sss._collect(p, p.ds_te, p.ds_te.idx, batch_size)
            del p
        assert mdl.fingerprint() == fp, "weights changed"
        with open(f, "wb") as fh:
            pickle.dump(inj, fh)
        out[ms] = inj
    return out


# ============================================================================ detectors
class Detector:
    """Scores + thresholds of one method on every set (val, clean, placements, hard placements).

    ``stgnn``      AFTER detector of section 12 (all hours);
    ``stgnn_day``  the same, alarms restricted to daylight hours (same definition as the PR rule);
    ``pr``         PR threshold rule;
    ``prp``        PR+ (pv_prplus: hour x month baseline, cluster peers, causal rolling median);
    ``fusion*``    rule OR STGNN-daylight (rule = PR for ``fusion*``, PR+ for ``fusion_prp*``); ``q`` is a
                   pair (q_stgnn, q_rule) — the budget is split between the two components in a
                   pre-specified ratio (``FUSION_ALPHA``)."""

    def __init__(self, method: str, st: "fx.Family", rules: Dict[str, "fx.Family"], zeros: Dict[object, np.ndarray],
                 days: Dict[object, np.ndarray], min_len: int):
        if not isinstance(rules, dict):                     # backwards compatible: a single PR family
            rules = {"pr": rules}
        self.method, self.st, self.rules, self.zeros, self.days, self.min_len = method, st, rules, zeros, days, min_len
        self.pr = rules.get("pr")

    def _st(self, key, q, day_only):
        c = self.st.sets[key]["comb"]
        if day_only:
            c = np.where(self.days[key] | ~np.isfinite(c), c, -1e6)   # finite: night hours stay in the denominator
        return c, self.st.thr_fn(q)[key]

    def comb_thr(self, key, q):
        if self.method in ("stgnn", "stgnn_day"):
            c, t = self._st(key, q, self.method == "stgnn_day")
            return c, t, self.zeros.get(key)
        if self.method in RULES:
            fam = self.rules[self.method]
            return fam.sets[key]["comb"], fam.thr_fn(q)[key], None
        rule = self.rules[FUSION_ALPHA[self.method][0]]
        q_st, q_pr = q
        c1, t1 = self._st(key, q_st, True)
        a = c1 - t1
        b = rule.sets[key]["comb"] - rule.thr_fn(q_pr)[key]
        with np.errstate(invalid="ignore"):
            c = np.fmax(a, b)
        c[~np.isfinite(self.st.sets[key]["comb"]) & ~np.isfinite(rule.sets[key]["comb"])] = np.nan
        return c, np.zeros_like(c), self.zeros.get(key)

    def rate(self, key, q) -> float:
        c, t, z = self.comb_thr(key, q)
        return fx._rate_fast(c, t, self.min_len, z)

    def bundle(self, key, q, pv) -> "sse.ScoreBundle":
        c, t, z = self.comb_thr(key, q)
        s = self.st.sets[key]
        c = np.where(np.isneginf(c), -1e6, c)
        return fx.make_bundle(c, t, s["zp"], s["eta"], s["raw"], pv, self.min_len, z)


def _match_q(det: Detector, budget: float, grid: np.ndarray, cache: dict) -> Tuple[object, float]:
    """q of the fine grid with the largest clean (all-plant) alarm rate not above the budget.
    For the fusion each component gets its share of the budget; the fused rate is <= the sum."""
    if det.method in FUSION_ALPHA:
        rk, al = FUSION_ALPHA[det.method]
        dst = Detector("stgnn_day", det.st, det.rules, {}, det.days, det.min_len)
        dpr = Detector(rk, det.st, det.rules, {}, det.days, det.min_len)
        q1, _ = _match_q(dst, al * budget, grid, cache.setdefault("st", {}))
        q2, _ = _match_q(dpr, (1 - al) * budget, grid, cache.setdefault(rk, {}))
        return (q1, q2), det.rate("clean", (q1, q2))
    best = None
    for q in grid:
        q = float(q)
        if q not in cache:
            cache[q] = det.rate("clean", q)
        r = cache[q]
        if np.isfinite(r) and r <= budget:
            best = (q, r); break
    if best is None:
        q = float(grid[-1]); best = (q, cache.get(q, det.rate("clean", q)))
    return best


# ============================================================================ configuration / result
@dataclass
class ValueConfig:
    budgets: Sequence[float] = (0.5, 1.0, 2.0, 5.0)     # clean alarm events per 1000 scored hours (all plants)
    ref_budget: Optional[float] = None                 # default: S4 of section 12 (bg_all), else 1.0
    n_boot: int = 2000
    systemic_share: float = 0.5
    hard: bool = True
    seed: int = 0


@dataclass
class ValueResult:
    points: pd.DataFrame          # method x budget x model seed: q, clean rate, summary metrics
    events: pd.DataFrame          # base events with detection flags (method, budget, seed, placement)
    hard_events: pd.DataFrame
    strata: pd.DataFrame          # condition 3: alarm counts by stratum at the reference budget
    systemic: pd.DataFrame
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


# ============================================================================ strata (condition 3)
def _plant_mismatch(R: "sss.ModelRaw", ghi_cal: np.ndarray, cfx: "fx.FixesConfig") -> pd.Series:
    n = R.calib["n_pv"]
    y = R.cal["y"].mean(-1)[:, :n]; a = R.cal["a"].mean(-1)[:, :n]
    day = fx._daylight(R.cal, ghi_cal, n, cfx) & (y > 0)
    v = {}
    for j, c in enumerate(R.pv_cols[:n]):
        r = y[day[:, j], j] / a[day[:, j], j]
        v[c] = float(np.subtract(*np.quantile(r, [0.75, 0.25])) / max(np.median(r), 1e-6)) if len(r) >= 200 else np.nan
    s = pd.Series(v)
    out = pd.Series("no training data", index=s.index, dtype=object)
    ok = s.notna()
    if ok.sum() >= 3:
        t = s[ok].quantile([1 / 3, 2 / 3]).to_numpy()
        out[ok] = np.where(s[ok] <= t[0], "low P/P_exp scatter (train)",
                           np.where(s[ok] <= t[1], "medium P/P_exp scatter (train)", "high P/P_exp scatter (train)"))
    elif ok.any():
        out[ok] = "medium P/P_exp scatter (train)"
    return out


def _strata_rows(b: "sse.ScoreBundle", D: dict, C: dict, R, mismatch: pd.Series, cohorts: pd.Series, tag: dict):
    em = b.event_mask; comb = b.combined; ts = b.timestamps
    n = em.shape[1]
    scored = np.isfinite(comb)
    d = np.diff(np.concatenate([np.zeros((1, n), np.int8), em.astype(np.int8)]), axis=0) == 1
    wr = C["wr"]
    reg = wr.regime.reindex(ts).fillna(0).astype(int).to_numpy()
    names = getattr(wr, "names", {}) or {}
    rname = np.array([str(names.get(r, f"R{r}")) for r in reg], dtype=object)
    season = np.array([SEASONS[m] for m in ts.month], dtype=object)
    pv = R.pv_cols[:n]
    groups = {"Weather regime": (rname[:, None].repeat(n, 1)),
              "Season": (season[:, None].repeat(n, 1)),
              "Physics-model mismatch of the plant": np.array([mismatch.get(c, "no training data") for c in pv],
                                                              dtype=object)[None, :].repeat(len(ts), 0),
              "Cohort": np.array([ssg.COHORT_TITLES.get(int(cohorts.get(c, 0)), "Insufficient data") for c in pv],
                                 dtype=object)[None, :].repeat(len(ts), 0)}
    rows = []
    for g, lab in groups.items():
        for v in pd.unique(lab[scored]):
            m = (lab == v) & scored
            rows.append(dict(tag, group=g, stratum=v, hours=int(m.sum()), alarms=int((d & m).sum())))
    return rows


# ============================================================================ main
def _systemic_rows(E: pd.DataFrame, b: "sse.ScoreBundle", share: float, tag: dict) -> List[dict]:
    em, comb, ts = b.event_mask, b.combined, b.timestamps
    sc = np.isfinite(comb)
    with np.errstate(invalid="ignore"):
        frac = (em & sc).sum(1) / np.maximum(sc.sum(1), 1)
    rows = []
    for r in E.itertuples():
        if r.level != "plant" or not r.detected or pd.isna(r.start):
            continue
        p = (ts >= r.start) & (ts <= r.end + pd.Timedelta(hours=24))
        rows.append(dict(tag, event_id=r.event_id, systemic=bool(p.any() and frac[p].max() >= share)))
    return rows


def run_value_checks(D: dict, C: dict, models: Dict[int, "sse.FrozenSTGNN"], ss_res=None, fixes_res=None,
                     placement_seeds: Sequence[int] = (0, 1, 2), cfg: Optional[ValueConfig] = None,
                     cfx: Optional["fx.FixesConfig"] = None, gen_cfg=None, raw_cache_dir: str = "results/fixes",
                     out_dir: str = "results/value", use_cache: bool = True) -> ValueResult:
    cfg = cfg or ValueConfig(); cfx = cfx or fx.FixesConfig()
    t0 = time.time()
    if ss_res is not None and getattr(ss_res, "plans", None):
        plans = dict(ss_res.plans)
    else:
        gen_cfg = gen_cfg or ssg.GenConfig(boundary=sse.BOUNDARY)
        plans = {ps: ssg.plan_semisynthetic(D, C, seed=ps, cfg=gen_cfg) for ps in placement_seeds}
    reals = {ps: ssg.apply_plan(D, C, p) for ps, p in plans.items()}
    raws = fx.collect(models, D, reals, raw_cache_dir, None, cfx.batch_size, use_cache)
    reals_h, hraw = {}, {}
    if cfg.hard:
        for ps, p in plans.items():
            reals_h[f"hard{ps}"] = apply_hard(D, p, plan_hard(p, ps), ps)
        hraw = collect_hard(models, reals_h, out_dir, cfx.batch_size, use_cache)
    ref = cfg.ref_budget
    if ref is None and fixes_res is not None:
        s4 = fixes_res.points[fixes_res.points["method"] == "S4"]
        ref = float(s4["bg_all"].mean()) if len(s4) and "bg_all" in s4 else None
    ref = float(ref) if ref is not None and np.isfinite(ref) else 1.0
    budgets = sorted(set(float(b) for b in cfg.budgets) | {round(ref, 3)})
    _log(f"reference alarm budget (clean test, all plants): {ref:.3f} /1000 h; budgets {budgets}")
    grid = fx._q_grid(fx.FixesConfig(q_select_grid=81))
    cohorts = ssg.plant_cohorts(D, C, list(D["pv_cols"]))

    PTS, EV, EH, ST, SY = [], [], [], [], []
    for ms, R0 in raws.items():
        R = copy.copy(R0); R.inj = dict(R0.inj); R.inj.update(hraw.get(ms, {}))
        ghis = {"cal": fx._ghi_at(D["df"], R.cal["ts"]), "val": fx._ghi_at(D["df"], R.val["ts"]),
                "clean": fx._ghi_at(D["df"], R.clean["ts"])}
        allreals = dict(reals); allreals.update(reals_h)
        for k, real in allreals.items():
            ghis[k] = fx._ghi_at(real.df, R.inj[k]["ts"])
        st = fx.stgnn_family(R, cfx.eta_mode)
        rules = {"pr": fx.pr_family(R, ghis, cfx), "prp": prp.prp_family(R, ghis, cfx)}
        n = R.calib["n_pv"]
        zeros = {k: fx.zero_rule_mask(s["raw"], ghis[k], n, s["nr"], cfx) for k, s in st.sets.items()}
        days = {k: fx._daylight(s["raw"], ghis[k], n, cfx) for k, s in st.sets.items()}
        mism = _plant_mismatch(R, ghis["cal"], cfx)
        for m in ALL_METHODS:
            det = Detector(m, st, rules, zeros, days, cfx.min_len_after)
            cache = {}
            for b in budgets:
                q, rate = _match_q(det, b, grid, cache)
                B = {k: det.bundle(k, q, R.pv_cols) for k in ["clean"] + list(allreals)}
                tag = dict(method=m, label=METHOD_LABEL[m], model_seed=ms, budget=b)
                summ, E, _, _ = fx._evaluate(D, C, reals, B["clean"], {ps: B[ps] for ps in reals}, cfx.min_len_after,
                                             tag, None, False)
                PTS.append(dict(tag, q=q if not isinstance(q, tuple) else q[0], q_pr=q[1] if isinstance(q, tuple) else np.nan,
                                clean_rate=rate, **{k: summ.get(k) for k in
                                                              ("recall", "precision", "F1", "clean_fa", "bg_all")}))
                EV.append(E)
                if reals_h and m in METHODS:
                    _, Eh, _, _ = fx._evaluate(D, C, reals_h, B["clean"], {k: B[k] for k in reals_h},
                                               cfx.min_len_after, tag, None, False)
                    EH.append(Eh)
                if abs(b - round(ref, 3)) < 1e-9 and m in METHODS:
                    ST += _strata_rows(B["clean"], D, C, R, mism, cohorts, tag)
                    for ps in reals:
                        SY += _systemic_rows(E[E["placement"] == ps], B[ps], cfg.systemic_share, dict(tag, placement=ps))
            _log(f"seed {ms} {m}: done ({time.time() - t0:.0f}s)")
    for mdl in models.values():
        assert mdl.unchanged(), "weights changed"
    res = ValueResult(points=pd.DataFrame(PTS), events=pd.concat(EV, ignore_index=True),
                      hard_events=pd.concat(EH, ignore_index=True) if EH else pd.DataFrame(),
                      strata=pd.DataFrame(ST), systemic=pd.DataFrame(SY),
                      meta=dict(ref_budget=round(ref, 3), budgets=budgets, seeds=sorted(models), placements=sorted(plans),
                                n_boot=cfg.n_boot, weights_unchanged=True, systemic_share=cfg.systemic_share,
                                seed=cfg.seed))
    res.tables = build_tables(res)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "value_result.pkl"), "wb") as f:
            pickle.dump(res, f)
    _log(f"done in {time.time() - t0:.0f}s")
    return res


# ============================================================================ statistics / tables
def _main_events(E: pd.DataFrame) -> pd.DataFrame:
    return E[E["level"].isin(["plant", "fleet"]) & (E["duration"] != "short")]


def paired_bootstrap(E: pd.DataFrame, a: str, b: str, budget: float, n_boot: int = 2000, seed: int = 0,
                     subset=None) -> dict:
    """Recall(a) − recall(b) on the same events (model seed, placement, event id); 95 % percentile CI
    from an event-level cluster bootstrap (seeds of one event are resampled together)."""
    k = ["model_seed", "placement", "event_id"]
    X = E[(E["budget"] == budget)]
    if subset is not None:
        X = X[subset(X)]
    A = X[X["method"] == a].set_index(k)["detected"].astype(float)
    B = X[X["method"] == b].set_index(k)["detected"].astype(float)
    j = A.index.intersection(B.index)
    if not len(j):
        return dict(n=0, diff=np.nan, lo=np.nan, hi=np.nan, only_a=0, only_b=0)
    da, db = A.loc[j].to_numpy(), B.loc[j].to_numpy()
    # event-level cluster bootstrap: all model seeds of one (placement, event) are resampled together
    cl = pd.MultiIndex.from_arrays([j.get_level_values("placement"), j.get_level_values("event_id")])
    codes, _ = pd.factorize(cl)
    nc = int(codes.max()) + 1
    sa, sb = np.bincount(codes, da, nc), np.bincount(codes, db, nc)
    cnt = np.bincount(codes, minlength=nc).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nc, size=(n_boot, nc))
    w = cnt[idx].sum(1)
    diffs = sa[idx].sum(1) / w - sb[idx].sum(1) / w
    return dict(n=int(len(j)), recall_a=float(da.mean()), recall_b=float(db.mean()), diff=float(da.mean() - db.mean()),
                lo=float(np.quantile(diffs, 0.025)), hi=float(np.quantile(diffs, 0.975)),
                only_a=int(((da == 1) & (db == 0)).sum()), only_b=int(((db == 1) & (da == 0)).sum()))


def _verdict(lo: float, hi: float) -> str:
    if not np.isfinite(lo):
        return "no events"
    return "better (CI > 0)" if lo > 0 else ("worse (CI < 0)" if hi < 0 else "no significant difference")


def _age_bin(r) -> str:
    d = r["days_since_commissioning"]
    if not np.isfinite(d):
        return "unknown"
    for lo, hi, lab in AGE_BINS:
        if lo < d <= hi:
            return lab
    return "unknown"


def build_tables(res: ValueResult) -> Dict[str, pd.DataFrame]:
    T = {}
    P, E = res.points, _main_events(res.events)
    ref, nb, sd = res.meta["ref_budget"], res.meta["n_boot"], res.meta["seed"]
    order = {m: i for i, m in enumerate(ALL_METHODS)}
    # ---- condition 2: equal budget + fusion bootstrap
    A = P.groupby(["budget", "method", "label"], sort=False)[["q", "q_pr", "clean_rate", "clean_fa", "recall", "precision",
                                                              "F1"]].mean().reset_index()
    A["recall_sd"] = P.groupby(["budget", "method", "label"], sort=False)["recall"].std().values
    A = A.assign(_o=A["method"].map(order)).sort_values(["budget", "_o"]).drop(columns=["_o", "method"])
    T["equal_budget"] = A.rename(columns={"budget": "Alarm budget (clean, all plants) /1000 h", "label": "Method",
                                          "q": "q (STGNN / single)", "q_pr": "q (PR part of fusion)",
                                          "clean_rate": "Matched clean rate /1000 h",
                                          "clean_fa": "FA on control plants /1000 h", "recall": "Recall",
                                          "recall_sd": "Recall SD (seeds)", "precision": "Precision"}).reset_index(drop=True)
    rows = []
    for b in res.meta["budgets"]:
        for a, c in (("fusion", "pr"), ("fusion_25", "pr"), ("fusion_75", "pr"), ("stgnn", "pr"),
                     ("stgnn_day", "pr"), ("fusion", "stgnn_day"),
                     ("stgnn", "prp"), ("stgnn_day", "prp"), ("fusion_prp", "prp"), ("fusion_prp_25", "prp"),
                     ("fusion_prp_75", "prp"), ("fusion_prp", "stgnn_day"), ("prp", "pr")):
            r = paired_bootstrap(E, a, c, b, nb, sd)
            rows.append({"Alarm budget /1000 h": b, "Comparison": f"{SHORT[a]} − {SHORT[c]}",
                         "Events": r["n"], "ΔRecall": r["diff"], "95% CI low": r["lo"], "95% CI high": r["hi"],
                         "Only first detects": r["only_a"], "Only second detects": r["only_b"],
                         "Result": _verdict(r["lo"], r["hi"])})
    T["fusion_bootstrap"] = pd.DataFrame(rows)
    # ---- condition 3: alarms by stratum at the reference budget
    S = res.strata
    if len(S):
        G = S.groupby(["group", "stratum", "method"], sort=False)[["hours", "alarms"]].sum().reset_index()
        G["rate"] = 1000 * G["alarms"] / G["hours"].clip(lower=1)
        tot = G.groupby(["group", "method"])["alarms"].transform("sum")
        G["share"] = G["alarms"] / tot.clip(lower=1)
        W = G.pivot_table(index=["group", "stratum"], columns="method", values=["rate", "share"], sort=False)
        H = G[G["method"] == G["method"].iloc[0]].set_index(["group", "stratum"])["hours"]
        out = pd.DataFrame({"Group": [i[0] for i in W.index], "Stratum": [i[1] for i in W.index],
                            "Scored plant-hours": H.reindex(W.index).to_numpy() // max(len(res.meta["seeds"]), 1)})
        for m in METHODS:
            if ("rate", m) in W:
                out[f"Alarms /1000 h: {SHORT[m]}"] = W[("rate", m)].to_numpy()
        for m in METHODS:
            if ("share", m) in W:
                out[f"Share of alarms: {SHORT[m]}"] = W[("share", m)].to_numpy()
        if ("rate", "stgnn_day") in W and ("rate", "pr") in W:
            with np.errstate(all="ignore"):
                out["Rate ratio STGNN-day / PR"] = W[("rate", "stgnn_day")].to_numpy() / W[("rate", "pr")].to_numpy()
        if ("rate", "stgnn_day") in W and ("rate", "prp") in W:
            with np.errstate(all="ignore"):
                out["Rate ratio STGNN-day / PR+"] = W[("rate", "stgnn_day")].to_numpy() / W[("rate", "prp")].to_numpy()
        T["alarms_by_stratum"] = out.reset_index(drop=True)
    # ---- condition 4a: cold start
    X = E[E["budget"] == ref].copy()
    if len(X):
        X["Cohort"] = X["cohort"].map(lambda c: ssg.COHORT_TITLES.get(int(c), "Insufficient data") if pd.notna(c) else "unknown")
        X["Days after commissioning"] = X.apply(_age_bin, axis=1)
        X = X[X["level"] == "plant"]
        rows = []
        for gcol in ("Cohort", "Days after commissioning"):
            for v, g in X.groupby(gcol):
                r = {"Group": gcol, "Stratum": v, "Events": int(len(g[g["method"] == "pr"]))}
                for m in METHODS:
                    r[f"Recall: {SHORT[m]}"] = g.loc[g["method"] == m, "detected"].mean()
                bt = paired_bootstrap(E, "stgnn_day", "pr", ref, nb, sd,
                                      subset=lambda Z, gcol=gcol, v=v: _strat_col(Z, gcol) == v)
                r.update({"ΔRecall STGNN-day − PR": bt["diff"], "95% CI low": bt["lo"], "95% CI high": bt["hi"],
                          "Result": _verdict(bt["lo"], bt["hi"])})
                bp = paired_bootstrap(E, "stgnn_day", "prp", ref, nb, sd,
                                      subset=lambda Z, gcol=gcol, v=v: _strat_col(Z, gcol) == v)
                r.update({"ΔRecall STGNN-day − PR+": bp["diff"], "95% CI low (PR+)": bp["lo"],
                          "95% CI high (PR+)": bp["hi"], "Result vs PR+": _verdict(bp["lo"], bp["hi"])})
                rows.append(r)
        T["cold_start"] = pd.DataFrame(rows)
    # ---- condition 4b: additional scenarios
    H = res.hard_events
    if len(H):
        H = H[(H["budget"] == ref) & (H["level"] == "plant")]
        rows = []
        for (scn, s), g in H.groupby(["scenario", "strength"], sort=False):
            r = {"Scenario": HARD_SCENARIOS[scn]["title"], "Strength": s,
                 "Unit": HARD_SCENARIOS[scn]["unit"], "Events": int(len(g[g["method"] == "pr"]))}
            for m in METHODS:
                r[f"Recall: {SHORT[m]}"] = g.loc[g["method"] == m, "detected"].mean()
            rows.append(r)
        for scn, g in H.groupby("scenario", sort=False):
            bt = paired_bootstrap(H, "stgnn_day", "pr", ref, nb, sd, subset=lambda Z, scn=scn: Z["scenario"] == scn)
            bf = paired_bootstrap(H, "fusion", "pr", ref, nb, sd, subset=lambda Z, scn=scn: Z["scenario"] == scn)
            r = {"Scenario": HARD_SCENARIOS[scn]["title"], "Strength": "all", "Unit": "",
                 "Events": bt["n"]}
            for m in METHODS:
                r[f"Recall: {SHORT[m]}"] = g.loc[g["method"] == m, "detected"].mean()
            bp = paired_bootstrap(H, "stgnn_day", "prp", ref, nb, sd, subset=lambda Z, scn=scn: Z["scenario"] == scn)
            r.update({"ΔRecall STGNN-day − PR": bt["diff"], "95% CI low": bt["lo"], "95% CI high": bt["hi"],
                      "Result": _verdict(bt["lo"], bt["hi"]), "ΔRecall fusion − PR": bf["diff"],
                      "Fusion result": _verdict(bf["lo"], bf["hi"]),
                      "ΔRecall STGNN-day − PR+": bp["diff"], "95% CI low (PR+)": bp["lo"],
                      "95% CI high (PR+)": bp["hi"], "Result vs PR+": _verdict(bp["lo"], bp["hi"])})
            rows.append(r)
        Th = pd.DataFrame(rows)
        Th["_o"] = Th["Strength"].astype(str).eq("all").astype(int)
        Th["_s"] = Th["Scenario"].map({v["title"]: i for i, v in enumerate(HARD_SCENARIOS.values())})
        T["hard_scenarios"] = Th.sort_values(["_s", "_o"]).drop(columns=["_o", "_s"]).reset_index(drop=True)
    # ---- condition 4c: sensor vs plant
    rows = []
    F = res.events[(res.events["budget"] == ref) & (res.events["scenario"] == "ghi_shift")]
    for m in METHODS:
        r = {"Method": METHOD_LABEL[m]}
        g = F[F["method"] == m]
        r["Fleet GHI shift: share of plants alarming"] = g["fleet_share_detected"].mean() if len(g) else np.nan
        r["Fleet GHI shift: flagged as systemic"] = g["detected"].mean() if len(g) else np.nan
        y = res.systemic[res.systemic["method"] == m] if len(res.systemic) else pd.DataFrame()
        r["Plant events wrongly flagged as systemic"] = y["systemic"].mean() if len(y) else np.nan
        rows.append(r)
    T["sensor_vs_plant"] = pd.DataFrame(rows)
    T["verdict"] = verdict_table(T, ref)
    return T


def _strat_col(Z: pd.DataFrame, gcol: str) -> pd.Series:
    if gcol == "Cohort":
        return Z["cohort"].map(lambda c: ssg.COHORT_TITLES.get(int(c), "Insufficient data") if pd.notna(c) else "unknown")
    return Z.apply(_age_bin, axis=1) if len(Z) else pd.Series([], dtype=object)


def verdict_table(T: Dict[str, pd.DataFrame], ref: float) -> pd.DataFrame:
    rows = []
    fb = T.get("fusion_bootstrap")
    if fb is not None and len(fb):
        r = fb[(fb["Alarm budget /1000 h"] == ref) & (fb["Comparison"] == f"{SHORT['fusion']} − {SHORT['pr']}")]
        o = fb[(fb["Alarm budget /1000 h"] == ref) & fb["Comparison"].str.startswith("Fusion")
               & fb["Comparison"].str.endswith(f" − {SHORT['pr']}")]
        if len(r):
            r = r.iloc[0]
            extra = "; budget-split sensitivity: " + ", ".join(
                f"{x['Comparison'].split(' − ')[0]} {x['ΔRecall']:+.3f}" for _, x in o.iterrows()) if len(o) else ""
            rows.append(dict(Condition="2. Fusion PR OR STGNN beats the PR rule at equal budget",
                             Criterion="95% CI of ΔRecall (fusion 50/50 − PR) above 0 (pre-specified split)",
                             Evidence=f"ΔRecall = {r['ΔRecall']:+.3f} [{r['95% CI low']:.3f}, {r['95% CI high']:.3f}], "
                                      f"n = {r['Events']}" + extra, Met=bool(r["95% CI low"] > 0)))
    st = T.get("alarms_by_stratum")
    if st is not None and len(st) and "Rate ratio STGNN-day / PR" in st:
        hard = st[st["Stratum"].astype(str).str.contains(DIFFICULT, case=False, regex=True)]
        if len(hard):
            ratio = (hard["Alarms /1000 h: STGNN-day"] * hard["Scored plant-hours"]).sum() / \
                max((hard["Alarms /1000 h: PR"] * hard["Scored plant-hours"]).sum(), 1e-9)
            rows.append(dict(Condition="3. Fewer STGNN alarms in difficult conditions at equal total budget",
                             Criterion="alarm-rate ratio STGNN-day / PR < 0.8 pooled over cloudy / rain, winter and high-mismatch strata",
                             Evidence=f"pooled ratio = {ratio:.2f}; " + "; ".join(
                                 f"{x['Stratum']}: {x['Rate ratio STGNN-day / PR']:.2f}" for _, x in hard.iterrows()),
                             Met=bool(ratio < 0.8)))
    cs, hs = T.get("cold_start"), T.get("hard_scenarios")
    ev, met = [], False
    if cs is not None and len(cs):
        new = cs[cs["Stratum"].astype(str).str.contains("Cohort 3|0–30 d|30–90 d")]
        for _, r in new.iterrows():
            ev.append(f"{r['Stratum']}: Δ={r['ΔRecall STGNN-day − PR']:+.3f}")
            met |= bool(r["95% CI low"] > 0)
    if hs is not None and len(hs):
        for _, r in hs[hs["Strength"].astype(str) == "all"].iterrows():
            ev.append(f"{r['Scenario']}: Δ={r['ΔRecall STGNN-day − PR']:+.3f}")
            met |= bool(r["95% CI low"] > 0)
    sp = T.get("sensor_vs_plant")
    if sp is not None and len(sp):
        a = sp.set_index("Method")["Plant events wrongly flagged as systemic"]
        ev.append(f"false systemic flags STGNN-day {a.get(METHOD_LABEL['stgnn_day'], np.nan):.2f} vs PR {a.get(METHOD_LABEL['pr'], np.nan):.2f}")
    if ev:
        rows.append(dict(Condition="4. STGNN better for new plants or where P/P_exp is weak",
                         Criterion="95% CI of ΔRecall (STGNN-day − PR) above 0 for cohort 3 / ≤ 90 d or any additional scenario",
                         Evidence="; ".join(ev), Met=bool(met)))
    rows += _verdict_prp(T, ref)
    return pd.DataFrame(rows)


def _verdict_prp(T: Dict[str, pd.DataFrame], ref: float) -> List[dict]:
    """The same pre-specified criteria with PR+ as the reference (main comparison STGNN vs PR+)."""
    rows = []
    fb = T.get("fusion_bootstrap")
    if fb is not None and len(fb):
        for cond, a in (("1+. STGNN-daylight beats PR+ at equal budget", "stgnn_day"),
                        ("2+. Fusion PR+ OR STGNN beats PR+ at equal budget", "fusion_prp")):
            r = fb[(fb["Alarm budget /1000 h"] == ref) & (fb["Comparison"] == f"{SHORT[a]} − {SHORT['prp']}")]
            if len(r):
                r = r.iloc[0]
                other = fb[(fb["Comparison"] == f"{SHORT[a]} − {SHORT['prp']}") & (fb["Alarm budget /1000 h"] != ref)]
                extra = "; other budgets: " + ", ".join(f"{x['Alarm budget /1000 h']:g}: {x['ΔRecall']:+.3f}"
                                                         for _, x in other.iterrows()) if len(other) else ""
                rows.append(dict(Condition=cond, Criterion=f"95% CI of ΔRecall ({SHORT[a]} − PR+) above 0",
                                 Evidence=f"ΔRecall = {r['ΔRecall']:+.3f} [{r['95% CI low']:.3f}, {r['95% CI high']:.3f}], "
                                          f"n = {r['Events']}" + extra, Met=bool(r["95% CI low"] > 0)))
    st = T.get("alarms_by_stratum")
    if st is not None and len(st) and "Rate ratio STGNN-day / PR+" in st:
        hard = st[st["Stratum"].astype(str).str.contains(DIFFICULT, case=False, regex=True)]
        if len(hard):
            ratio = (hard["Alarms /1000 h: STGNN-day"] * hard["Scored plant-hours"]).sum() / \
                max((hard["Alarms /1000 h: PR+"] * hard["Scored plant-hours"]).sum(), 1e-9)
            rows.append(dict(Condition="3+. Fewer STGNN alarms than PR+ in difficult conditions",
                             Criterion="alarm-rate ratio STGNN-day / PR+ < 0.8 pooled over cloudy / rain, winter and high-mismatch strata",
                             Evidence=f"pooled ratio = {ratio:.2f}", Met=bool(ratio < 0.8)))
    cs, hs = T.get("cold_start"), T.get("hard_scenarios")
    ev, met = [], False
    if cs is not None and len(cs) and "ΔRecall STGNN-day − PR+" in cs:
        for _, r in cs[cs["Stratum"].astype(str).str.contains("Cohort 3|0–30 d|30–90 d")].iterrows():
            ev.append(f"{r['Stratum']}: Δ={r['ΔRecall STGNN-day − PR+']:+.3f}")
            met |= bool(r["95% CI low (PR+)"] > 0)
    if hs is not None and len(hs) and "ΔRecall STGNN-day − PR+" in hs:
        for _, r in hs[hs["Strength"].astype(str) == "all"].iterrows():
            ev.append(f"{r['Scenario']}: Δ={r['ΔRecall STGNN-day − PR+']:+.3f}")
            met |= bool(r["95% CI low (PR+)"] > 0)
    if ev:
        rows.append(dict(Condition="4+. STGNN better than PR+ for new plants or where P/P_exp is weak",
                         Criterion="95% CI of ΔRecall (STGNN-day − PR+) above 0 for cohort 3 / ≤ 90 d or any additional scenario",
                         Evidence="; ".join(ev), Met=bool(met)))
    return rows


# ============================================================================ output
TITLES = {"equal_budget": "Conditions 1+ and 2: PR rule, PR+, STGNN and fusions at equal alarm budgets",
          "fusion_bootstrap": "Paired bootstrap of the recall difference (vs PR and PR+)",
          "alarms_by_stratum": "Condition 3: where the alarms go (clean test, all plants, reference budget)",
          "cold_start": "Condition 4a: recall of new plants (reference budget)",
          "hard_scenarios": "Condition 4b: additional scenarios where P/P_exp is weak (reference budget)",
          "sensor_vs_plant": "Condition 4c: sensor fault vs plant fault",
          "verdict": "Summary of conditions 2-4 (vs PR) and 1+-4+ (vs PR+)"}


def show(res: ValueResult, digits: int = 3):
    try:
        from IPython.display import display
    except Exception:                                   # pragma: no cover
        display = print
    print(f"Reference alarm budget: {res.meta['ref_budget']} /1000 h (clean test, all plants); "
          f"weights unchanged: {res.meta['weights_unchanged']}")
    for k, t in res.tables.items():
        print(f"\n=== {TITLES.get(k, k)}")
        display(t.round(digits) if hasattr(t, "round") else t)


def plot_all(res: ValueResult, fig_dir: Optional[str] = "results/figures") -> Dict[str, object]:
    import matplotlib.pyplot as plt
    figs = {}
    P = res.points[res.points["method"].isin(METHODS)].groupby(["method", "budget"], sort=False)[
        ["clean_rate", "recall", "F1"]].mean().reset_index()
    f, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for m in METHODS:
        g = P[P["method"] == m].sort_values("clean_rate")
        ax[0].plot(g["clean_rate"], g["recall"], "o-", c=COLORS[m], label=METHOD_LABEL[m])
        ax[1].plot(g["clean_rate"], g["F1"], "o-", c=COLORS[m], label=METHOD_LABEL[m])
    for a in ax:
        a.axvline(res.meta["ref_budget"], c="grey", ls=":", lw=1)
        a.set_xscale("log"); a.grid(alpha=.3)
        a.set_xlabel("alarm events per 1000 scored hours, clean test, all plants (log)")
    ax[0].set(ylabel="recall (day + week events)", title="Condition 2: recall at equal alarm budget")
    ax[1].set(ylabel="F1", title="Condition 2: F1 at equal alarm budget (dotted: reference)")
    ax[0].legend(fontsize=8, loc="lower right")
    f.tight_layout(); figs["fusion"] = f
    S = res.tables.get("alarms_by_stratum")
    if S is not None and len(S):
        groups = [g for g in S["Group"].unique()]
        f, axs = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 4.6), squeeze=False)
        for a, g in zip(axs[0], groups):
            s = S[S["Group"] == g]
            x = np.arange(len(s)); w = 0.8 / len(METHODS)
            for i, m in enumerate(METHODS):
                col = f"Alarms /1000 h: {SHORT[m]}"
                if col in s:
                    a.bar(x + (i - (len(METHODS) - 1) / 2) * w, s[col], w, color=COLORS[m], label=METHOD_LABEL[m])
            a.set_xticks(x); a.set_xticklabels([str(v) for v in s["Stratum"]], rotation=30, ha="right", fontsize=8)
            a.set_title(g, fontsize=10); a.grid(alpha=.3, axis="y")
        axs[0][0].set_ylabel("alarm events per 1000 h (clean test)")
        h, l = axs[0][0].get_legend_handles_labels()
        f.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
        f.suptitle(f"Condition 3: alarms by stratum at an equal total budget ({res.meta['ref_budget']:.2f} /1000 h)",
                   fontsize=11)
        f.tight_layout(rect=(0, 0.1, 1, 1)); figs["strata"] = f
    Hs = res.tables.get("hard_scenarios"); Cs = res.tables.get("cold_start")
    if (Hs is not None and len(Hs)) or (Cs is not None and len(Cs)):
        f, axs = plt.subplots(1, 2, figsize=(14, 4.6))
        for a, t, lab, ttl in ((axs[0], Hs, "Scenario", "Condition 4b: additional scenarios"),
                               (axs[1], Cs, "Stratum", "Condition 4a: new plants")):
            if t is None or not len(t):
                a.axis("off"); continue
            t = t[t["Strength"].astype(str) == "all"] if "Strength" in t else t
            x = np.arange(len(t)); w = 0.8 / len(METHODS)
            for i, m in enumerate(METHODS):
                if f"Recall: {SHORT[m]}" in t:
                    a.bar(x + (i - (len(METHODS) - 1) / 2) * w, t[f"Recall: {SHORT[m]}"], w, color=COLORS[m],
                          label=METHOD_LABEL[m])
            a.set_xticks(x); a.set_xticklabels([str(v) for v in t[lab]], rotation=20, ha="right", fontsize=8)
            a.set(ylabel="recall", ylim=(0, 1.05), title=f"{ttl} (budget {res.meta['ref_budget']:.2f} /1000 h)")
            a.grid(alpha=.3, axis="y")
        h, l = axs[0].get_legend_handles_labels()
        f.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
        f.tight_layout(rect=(0, 0.1, 1, 1)); figs["hard"] = f
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        for k, fg in figs.items():
            fg.savefig(os.path.join(fig_dir, f"fig_value_{k}.png"), dpi=150, bbox_inches="tight")
    return figs


TEX_CAPTIONS = {
    "equal_budget": "PR threshold rule, PR+, STGNN and their fusions at equal alarm budgets (semi-synthetic, frozen weights)",
    "fusion_bootstrap": "Paired bootstrap of the recall difference at equal alarm budgets (STGNN and fusions vs PR and PR+)",
    "alarms_by_stratum": "Alarm rates by weather regime, season, physics-model mismatch and cohort at an equal total budget (condition 3)",
    "cold_start": "Recall of new plants by cohort and days after commissioning (condition 4a)",
    "hard_scenarios": "Additional scenarios where the ratio $P/P^{\\mathrm{exp}}$ is weak (condition 4b)",
    "sensor_vs_plant": "Sensor fault versus plant fault (condition 4c)",
    "verdict": "Does the STGNN add value over the PR rule and PR+? Summary of the pre-specified conditions",
}


def export_latex(res: ValueResult, out_dir: str = "results/value") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths, allp = {}, ["% requires \\usepackage{booktabs,graphicx,longtable,array}", ""]
    for k, cap in TEX_CAPTIONS.items():
        t = res.tables.get(k)
        if t is None or not len(t):
            continue
        tex = sse.df_to_latex(t, cap, f"tab:value_{k}", digits=3)
        p = os.path.join(out_dir, f"value_{k}.tex")
        with open(p, "w", encoding="utf-8") as f:
            f.write(tex)
        t.to_csv(os.path.join(out_dir, f"value_{k}.csv"), index=False)
        paths[k] = p; allp.append(tex)
    res.points.to_csv(os.path.join(out_dir, "value_points_by_seed.csv"), index=False)
    p = os.path.join(out_dir, "value_tables.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(allp))
    paths["all"] = p
    return paths


def zip_outputs(out_dir: str = "results/value", fig_dir: str = "results/figures", name: str = "value_results.zip") -> str:
    import zipfile
    z = os.path.join(out_dir, name)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(out_dir)):
            if fn.endswith((".tex", ".csv")):
                zf.write(os.path.join(out_dir, fn), fn)
        if fig_dir and os.path.isdir(fig_dir):
            for fn in sorted(os.listdir(fig_dir)):
                if fn.startswith("fig_value_"):
                    zf.write(os.path.join(fig_dir, fn), os.path.join("figures", fn))
    return z
