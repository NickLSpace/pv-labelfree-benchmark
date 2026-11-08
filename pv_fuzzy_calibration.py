"""
Calibration of the fuzzy rules for clipping, shading and partial loss (semi-synthetic protocol).

Problem (section 10 / HKUST run): the original rule base uses the raw ratio P/P_exp (a plant with
PR ~ 0.85 looks "below" all the time), "flatness" relative to the plant's whole-series P99 (a
clip at 0.85*P99 is not flat by that definition) and requires "long"/"medium" duration for
partial loss.  Clipping, shading and partial loss are therefore mostly mapped to the
weather-driven class.

Fix: four plant-normalised shape descriptors computed on the event days
(pv_clustering.event_shape_features: rel_ratio, edge_gap, clip_score, lowlight_gap) and the rule set
``pv_fuzzy.FuzzyConfig(rule_set="calibrated")``.  The term parameters are fitted here on
CALIBRATION placements whose seeds are disjoint from the evaluation placements of sections
10-12 (no leakage): the true intervals of the injected events are used (oracle features), the
detector is not involved.

Fitting rule (transparent, no optimisation of the evaluation metric):
  * "boundary" terms are one-sided shoulders: the peak is the median of the positive class,
    sigma = distance between that median and the reference level / 2, so a value at the
    reference level gets mu ~ exp(-2) ~ 0.14 and the midpoint mu ~ 0.6;
  * reference levels: rel_ratio 1 (plant-normalised), edge_gap and clip_score = median over
    the non-target loss classes;
  * sigma floors avoid degenerate terms.
The calibrated rules are tuned to the generator; on real data they must be confirmed by an
expert sample (stated in the notebook).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import pv_semisynth_gen as ssg
import pv_semisynth_eval as sse

CAL_PLACEMENTS = (101, 102, 103)        # disjoint from the evaluation placements (0, 1, 2)
TARGET_SCN = ("clipping", "shading", "partial_loss")


def oracle_features(D: dict, C: dict, plans: Dict[int, "ssg.Plan"], reals: Optional[Dict[int, "ssg.Realisation"]] = None
                    ) -> pd.DataFrame:
    """Event descriptors on the TRUE intervals of the plant-level injected events."""
    import pv_clustering as clu
    out = []
    for ps, plan in plans.items():
        real = (reals or {}).get(ps) or ssg.apply_plan(D, C, plan)
        lab = real.labels[real.labels["effective"] & (real.labels["level"] == "plant")]
        if not len(lab):
            continue
        evf = pd.DataFrame(dict(plant=[ssg._short(c) for c in lab["column"]], start=lab["eff_start"].values,
                                end=lab["eff_end"].values, peak_score=0.0, mean_physics_z=0.0))
        f = clu.anomaly_event_features(evf, real.df, real.phys, D["pv_cols"], C["wr"].regime)
        f["scenario"] = lab["scenario"].values[f["event_id"].values]
        f["strength"] = lab["strength"].values[f["event_id"].values]
        f["duration"] = lab["duration"].values[f["event_id"].values]
        f["expected"] = [ssg.SCENARIOS[s].expected_name for s in f["scenario"]]
        f["placement"] = ps
        out.append(f)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _q(x, q, default):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else default


def calibrate_fuzzy(feat: pd.DataFrame, floor: float = 0.02) -> dict:
    """Fit the term parameters of rel_ratio / edge_gap / clip_score (see module docstring)."""
    f = feat
    by = {s: f[f["scenario"] == s] for s in f["scenario"].unique()}
    g = lambda s, c: by[s][c] if s in by else pd.Series(dtype=float)
    loss = f[f["scenario"].isin(["shading", "partial_loss"])]
    below_pk = min(_q(loss["rel_ratio"], 0.5, 0.9), 0.97)
    above_pk = max(_q(g("overgen", "rel_ratio"), 0.5, 1.2), 1.03)
    sh_pk = _q(g("shading", "edge_gap"), 0.5, 0.2)
    eg_ref = _q(f[f["scenario"].isin(["partial_loss", "clipping"])]["edge_gap"], 0.5, 0.0)
    sh_pk = max(sh_pk, eg_ref + 2 * floor)
    cl_ref = _q(f[f["scenario"].isin(["partial_loss", "shading"])]["clip_score"], 0.5, 0.0)
    cl_pk = max(_q(g("clipping", "clip_score"), 0.5, 0.4), cl_ref + 2 * floor)
    ll_ref = _q(f[f["scenario"].isin(["partial_loss", "clipping", "shading"])]["lowlight_gap"], 0.5, 0.0)
    ll_pk = max(_q(g("lowlight", "lowlight_gap"), 0.5, 0.15), ll_ref + 2 * floor)
    s_below = max((1.0 - below_pk) / 2, floor)
    s_above = max((above_pk - 1.0) / 2, floor)
    s_near = max(min(s_below, s_above) / 1.5, floor / 2)
    s_eg = max((sh_pk - eg_ref) / 2, floor)
    s_cl = max((cl_pk - cl_ref) / 2, floor)
    s_ll = max((ll_pk - ll_ref) / 2, floor)
    vars_ = {
        "rel_ratio": {"vmin": 0.0, "vmax": 3.0, "terms": {"below": (below_pk, s_below), "near1": (1.0, s_near),
                                                          "above": (above_pk, s_above)},
                      "shoulders": {"below": "left", "above": "right"}},
        "edge_gap": {"vmin": -1.0, "vmax": 1.0, "terms": {"low": (eg_ref, s_eg), "high": (sh_pk, s_eg)},
                     "shoulders": {"high": "right"}},
        "clip_score": {"vmin": 0.0, "vmax": 1.0, "terms": {"low": (cl_ref, s_cl), "high": (cl_pk, s_cl)},
                       "shoulders": {"high": "right"}},
        "lowlight_gap": {"vmin": -1.0, "vmax": 1.0, "terms": {"low": (ll_ref, s_ll), "high": (ll_pk, s_ll)},
                         "shoulders": {"high": "right"}},
    }
    return {"vars": vars_, "n_events": int(len(f)), "placements": sorted(map(int, f["placement"].unique()))
            if "placement" in f else []}


def calibration_table(cal: dict) -> pd.DataFrame:
    rows = []
    for v, spec in cal["vars"].items():
        for t, (c, s) in spec["terms"].items():
            rows.append({"variable": v, "term": t, "centre / peak": c, "sigma": s,
                         "shoulder": spec.get("shoulders", {}).get(t, "-")})
    return pd.DataFrame(rows)


def classify(feat: pd.DataFrame, cfg) -> pd.Series:
    import pv_fuzzy as fz
    if not len(feat):
        return pd.Series(dtype=object)
    return fz.fuzzy_classify_batch(feat, cfg).idxmax(1)


def accuracy_table(feat: pd.DataFrame, configs: Dict[str, object]) -> pd.DataFrame:
    """Share of events whose argmax fuzzy class equals the expected class, per scenario."""
    if not len(feat):
        return pd.DataFrame()
    rows = []
    pred = {k: classify(feat, c).to_numpy() for k, c in configs.items()}
    for scn in [s for s in ssg.SCENARIOS if s in set(feat["scenario"])] + ["all"]:
        m = (feat["scenario"] == scn).to_numpy() if scn != "all" else np.ones(len(feat), bool)
        r = {"scenario": ssg.SCENARIOS[scn].title if scn != "all" else "All plant-level events", "N": int(m.sum())}
        for k, p in pred.items():
            r[k] = float(np.mean(p[m] == feat["expected"].to_numpy()[m])) if m.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def confusion(feat: pd.DataFrame, cfg) -> pd.DataFrame:
    import pv_fuzzy as fz
    p = classify(feat, cfg)
    short = {n: n.split(" (")[0] for n in fz.CLASS_NAMES}
    t = pd.crosstab(feat["scenario"].map(lambda s: ssg.SCENARIOS[s].title), p.map(short), normalize="index")
    return t.reindex(columns=[short[n] for n in fz.CLASS_NAMES], fill_value=0.0)


def fit(D: dict, C: dict, placements: Sequence[int] = CAL_PLACEMENTS, gen_cfg=None,
        eval_placements: Iterable[int] = (0, 1, 2)):
    """Generate calibration placements (disjoint seeds), compute oracle features, fit the rules."""
    import pv_fuzzy as fz
    bad = set(placements) & set(eval_placements)
    if bad:
        raise ValueError(f"calibration placements overlap the evaluation placements: {sorted(bad)}")
    gen_cfg = gen_cfg or ssg.GenConfig(boundary=sse.BOUNDARY)
    plans = {ps: ssg.plan_semisynthetic(D, C, seed=ps, cfg=gen_cfg) for ps in placements}
    feat = oracle_features(D, C, plans)
    cal = calibrate_fuzzy(feat)
    return fz.FuzzyConfig(rule_set="calibrated", calibration=cal), feat
