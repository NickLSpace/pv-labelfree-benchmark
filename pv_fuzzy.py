"""
Fuzzy-logic anomaly characterization for the physics-informed PV pipeline.

Replaces the crisp rule-based ``_label_center`` in pv_clustering.py with a
Mamdani (or Takagi-Sugeno) inference system that outputs membership degrees
for all six anomaly classes simultaneously, instead of a single hard label.

IMPORTANT: the entire pipeline is unsupervised.  No labelled anomaly data is
used anywhere — not for the STGNN (self-supervised: forecast + reconstruction
+ physics residual), not for the threshold (extreme-value / quantile of the
training distribution), and not for the rule-based characterization below.
The crisp and fuzzy methods are both domain-knowledge rule systems applied
on top of unsupervised detection outputs.

Linguistic variables (same features used by the crisp rules):
    zero_frac   0..1     low / high   (high is a right shoulder: mu = 1 for zero_frac >= 0.55)
    mean_ratio  0..3     below / near1 / above
    flatness    0..1     smooth / flat
    mean_ghi    0..1000  low / mid / high
    mean_hour   6..20    morning / midday / evening
    duration_h  0..200   short / medium / long

Six anomaly classes (same as ANOMALY_CLASSES in pv_clustering.py):
    0  Outage / zero generation
    1  Power limitation (clipping / curtailment)
    2  Partial shading (time of day)
    3  Gradual loss (soiling / degradation)
    4  Overgeneration / data spike (P > P_exp)
    5  Weather-driven deviation (low irradiance)

Ablation knobs (all post-processing, no training):
    membership  = "gauss" | "tri"         (A1)
    n_vars      = 6 | 3                   (A2: 3 = zero_frac, mean_ratio, mean_ghi only)
    inference   = "mamdani" | "sugeno"    (A3)
    defuzz      = "centroid" | "mom"      (A4)
    rule_set    = "full" | "reduced"      (A5: drop low-weight / overlapping rules)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── class names (must match pv_clustering.ANOMALY_CLASSES) ──────────────────
CLASS_NAMES = [
    "Outage / zero generation",
    "Power limitation (clipping / curtailment)",
    "Partial shading (time of day)",
    "Gradual loss (soiling / degradation)",
    "Overgeneration / data spike (P > P_exp)",
    "Weather-driven deviation (low irradiance)",
]
N_CLASSES = len(CLASS_NAMES)


# ── membership functions ───────────────────────────────────────────────────

def _gauss(x, c, sig):
    return np.exp(-0.5 * ((x - c) / sig) ** 2)


def _tri(x, a, b, c):
    """Triangular membership: a=left foot, b=peak, c=right foot."""
    return np.maximum(0, np.minimum((x - a) / max(b - a, 1e-12),
                                     (c - x) / max(c - b, 1e-12)))


# ── linguistic variable definitions ────────────────────────────────────────

@dataclass
class LinguisticVar:
    name: str
    vmin: float
    vmax: float
    terms: Dict[str, Tuple[float, ...]]  # term_name -> params
    # one-sided (shoulder) terms: "right" -> mu = 1 for x >= peak, "left" -> mu = 1 for x <= peak
    shoulders: Dict[str, str] = field(default_factory=dict)


# Default (uncalibrated) parameters of the shape variables used by rule_set="calibrated";
# pv_fuzzy_calibration.calibrate_fuzzy() replaces them by values fitted on calibration
# placements that are disjoint from the evaluation placements.
CALIBRATED_VARS_DEFAULT = {
    "rel_ratio": {"vmin": 0.0, "vmax": 3.0, "terms": {"below": (0.85, 0.08), "near1": (1.0, 0.05),
                                                      "above": (1.2, 0.08)},
                  "shoulders": {"below": "left", "above": "right"}},
    "edge_gap": {"vmin": -1.0, "vmax": 1.0, "terms": {"low": (0.0, 0.06), "high": (0.25, 0.08)},
                 "shoulders": {"high": "right"}},
    "clip_score": {"vmin": 0.0, "vmax": 1.0, "terms": {"low": (0.0, 0.08), "high": (0.4, 0.12)},
                   "shoulders": {"high": "right"}},
    "lowlight_gap": {"vmin": -1.0, "vmax": 1.0, "terms": {"low": (0.0, 0.05), "high": (0.15, 0.06)},
                     "shoulders": {"high": "right"}},
}


def _to_tri(c: float, sig: float) -> Tuple[float, float, float]:
    return (c - 2.0 * sig, c, c + 2.0 * sig)


def _build_vars(membership: str = "gauss", calibration: Optional[dict] = None) -> Dict[str, LinguisticVar]:
    """Build the linguistic variables.  membership='gauss' or 'tri'.

    calibration : optional {"vars": {name: {"vmin","vmax","terms": {term: (centre, sigma)},
                  "shoulders": {...}}}} - adds / overrides variables (used by rule_set="calibrated").
                  Gaussian (centre, sigma) parameters are converted to triangles for 'tri'."""
    base = _build_vars_base(membership)
    if calibration is None:
        return base
    spec = dict(CALIBRATED_VARS_DEFAULT)
    spec.update(calibration.get("vars", {}))
    for name, v in spec.items():
        terms = {t: (tuple(map(float, p)) if membership == "gauss" else _to_tri(*map(float, p)))
                 for t, p in v["terms"].items()}
        base[name] = LinguisticVar(name, float(v["vmin"]), float(v["vmax"]), terms,
                                   shoulders=dict(v.get("shoulders", {})))
    return base


def _build_vars_base(membership: str = "gauss") -> Dict[str, LinguisticVar]:
    if membership == "gauss":
        return {
            "zero_frac": LinguisticVar("zero_frac", 0, 1, {
                "low":  (0.0, 0.18),
                "high": (0.55, 0.18),
            }, shoulders={"high": "right"}),   # full outage (zero_frac = 1) -> mu_high = 1
            "mean_ratio": LinguisticVar("mean_ratio", 0, 3, {
                "below": (0.80, 0.12),
                "near1": (1.00, 0.10),
                "above": (1.30, 0.15),
            }),
            "flatness": LinguisticVar("flatness", 0, 1, {
                "smooth": (0.0, 0.12),
                "flat":   (0.35, 0.15),
            }),
            "mean_ghi": LinguisticVar("mean_ghi", 0, 1200, {
                "low":  (100, 60),
                "mid":  (400, 120),
                "high": (650, 120),
            }),
            "mean_hour": LinguisticVar("mean_hour", 5, 21, {
                "morning": (8.5, 1.5),
                "midday":  (12.5, 1.5),
                "evening": (16.0, 1.5),
            }),
            "duration_h": LinguisticVar("duration_h", 0, 200, {
                "short":  (6, 3),
                "medium": (14, 5),
                "long":   (30, 12),
            }),
        }
    else:  # triangular
        return {
            "zero_frac": LinguisticVar("zero_frac", 0, 1, {
                "low":  (-0.2, 0.0, 0.3),
                "high": (0.3, 0.55, 1.2),
            }, shoulders={"high": "right"}),   # full outage (zero_frac = 1) -> mu_high = 1
            "mean_ratio": LinguisticVar("mean_ratio", 0, 3, {
                "below": (0.4, 0.8, 1.0),
                "near1": (0.85, 1.0, 1.15),
                "above": (1.1, 1.3, 1.8),
            }),
            "flatness": LinguisticVar("flatness", 0, 1, {
                "smooth": (-0.1, 0.0, 0.25),
                "flat":   (0.15, 0.35, 0.7),
            }),
            "mean_ghi": LinguisticVar("mean_ghi", 0, 1200, {
                "low":  (-50, 100, 250),
                "mid":  (250, 400, 550),
                "high": (450, 650, 900),
            }),
            "mean_hour": LinguisticVar("mean_hour", 5, 21, {
                "morning": (5, 8.5, 11),
                "midday":  (10, 12.5, 15),
                "evening": (14, 16, 20),
            }),
            "duration_h": LinguisticVar("duration_h", 0, 200, {
                "short":  (-2, 6, 14),
                "medium": (8, 14, 24),
                "long":   (18, 30, 60),
            }),
        }


# ── rule base ──────────────────────────────────────────────────────────────

@dataclass
class FuzzyRule:
    """A single fuzzy rule mapping antecedent memberships to a class.

    antecedent: list of (var_name, term_name)  — combined by AND
    consequent: class index 0..5
    weight:     rule strength (0..1)
    """
    antecedent: List[Tuple[str, str]]
    consequent: int
    weight: float = 1.0


def _build_rules_calibrated(calibration: Optional[dict] = None) -> List[FuzzyRule]:
    """Calibrated rule base (fixes for clipping, shading and partial loss).

    * outage           : zero_frac high
    * over-generation  : rel_ratio above            (plant-normalised instead of raw P/P_exp)
    * clipping         : clip_score high & GHI not low  (flat top unexplained by the physics)
    * shading          : edge_gap high & rel_ratio below   (morning / evening deficit, any duration)
    * partial loss     : rel_ratio below & edge_gap low & clip_score low & lowlight_gap low
                         (uniform deficit, any duration - the old rules required "long"/"medium")
    * weather-driven   : lowlight_gap high (deficit only under low light) or rel_ratio near 1
                         (low GHI alone no longer overrides a clear plant-normalised deficit)
    Rule weights can be overridden by calibration["weights"] (list in this order)."""
    rules = [
        FuzzyRule([("zero_frac", "high")], 0, 1.0),
        FuzzyRule([("rel_ratio", "above")], 4, 1.0),
        FuzzyRule([("clip_score", "high"), ("mean_ghi", "high")], 1, 1.0),
        FuzzyRule([("clip_score", "high"), ("mean_ghi", "mid")], 1, 0.8),
        FuzzyRule([("edge_gap", "high"), ("rel_ratio", "below")], 2, 0.9),
        FuzzyRule([("rel_ratio", "below"), ("edge_gap", "low"), ("clip_score", "low"),
                   ("lowlight_gap", "low")], 3, 0.9),
        FuzzyRule([("lowlight_gap", "high"), ("edge_gap", "low")], 5, 0.9),
        FuzzyRule([("rel_ratio", "near1")], 5, 0.7),
    ]
    w = (calibration or {}).get("weights")
    if w is not None:
        for r, wi in zip(rules, w):
            r.weight = float(wi)
    return rules


def _build_rules(rule_set: str = "full", calibration: Optional[dict] = None) -> List[FuzzyRule]:
    """Full rule base mirrors the crisp _label_center logic."""
    if rule_set == "calibrated":
        return _build_rules_calibrated(calibration)
    rules = [
        # zero_frac > 0.5 → Outage
        FuzzyRule([("zero_frac", "high")], 0, 1.0),
        # mean_ratio > 1.25 → Overgeneration
        FuzzyRule([("mean_ratio", "above")], 4, 1.0),
        # flatness > 0.3 & mean_ghi > 400 & mean_ratio < 0.95 → Curtailment
        FuzzyRule([("flatness", "flat"), ("mean_ghi", "high"), ("mean_ratio", "below")], 1, 0.9),
        # mean_ghi < 100 → Weather
        FuzzyRule([("mean_ghi", "low")], 5, 0.8),
        # abs(mean_ratio - 1) < 0.1 → Weather
        FuzzyRule([("mean_ratio", "near1")], 5, 0.7),
        # (morning OR evening) & short & below → Shading
        FuzzyRule([("mean_hour", "morning"), ("duration_h", "short"), ("mean_ratio", "below")], 2, 0.8),
        FuzzyRule([("mean_hour", "evening"), ("duration_h", "short"), ("mean_ratio", "below")], 2, 0.8),
        # mean_ratio < 0.95 & long → Degradation
        FuzzyRule([("mean_ratio", "below"), ("duration_h", "long")], 3, 0.8),
        # mean_ratio < 0.95 & medium → Degradation (weaker)
        FuzzyRule([("mean_ratio", "below"), ("duration_h", "medium")], 3, 0.5),
    ]
    if rule_set == "reduced":
        # Drop the two weakest / most overlapping rules
        rules = [r for r in rules if r.weight >= 0.8 and r.consequent not in (3,)]
        # Keep degradation only via long duration, drop medium
        rules = [r for r in rules if not (r.consequent == 3 and r.weight < 0.8)]
    return rules


# ── fuzzy inference engine ─────────────────────────────────────────────────

@dataclass
class FuzzyConfig:
    membership: str = "gauss"      # A1: "gauss" | "tri"
    n_vars: int = 6               # A2: 6 | 3
    inference: str = "mamdani"    # A3: "mamdani" | "sugeno"
    defuzz: str = "centroid"     # A4: "centroid" | "mom"
    rule_set: str = "full"        # A5: "full" | "reduced" | "calibrated"
    calibration: Optional[dict] = None   # parameters for rule_set="calibrated" (pv_fuzzy_calibration)

    @property
    def label(self) -> str:
        parts = [f"mf={self.membership}"]
        if self.n_vars < 6:
            parts.append(f"vars={self.n_vars}")
        parts.append(f"inf={self.inference}")
        parts.append(f"df={self.defuzz}")
        if self.rule_set != "full":
            parts.append(f"rules={self.rule_set}")
        return " | ".join(parts)


def _term_membership(x: float, var: LinguisticVar, term: str, membership: str) -> float:
    params = var.terms[term]
    x_clipped = np.clip(x, var.vmin, var.vmax)
    peak = params[0] if membership == "gauss" else params[1]
    side = var.shoulders.get(term)
    if (side == "right" and x_clipped >= peak) or (side == "left" and x_clipped <= peak):
        return 1.0
    if membership == "gauss":
        c, sig = params
        return float(_gauss(x_clipped, c, sig))
    else:
        a, b, c = params
        return float(_tri(x_clipped, a, b, c))


def _eval_rule(rule: FuzzyRule, features: Dict[str, float],
               vars: Dict[str, LinguisticVar], membership: str) -> float:
    """Evaluate one rule's firing strength via AND (min for Mamdani, product for Sugeno)."""
    # OR within antecedent handled by splitting into separate rules (done in _build_rules)
    mus = []
    for var_name, term_name in rule.antecedent:
        v = vars.get(var_name)
        if v is None:
            return 0.0  # variable not available (n_vars reduction)
        if term_name not in v.terms:
            return 0.0
        mu = _term_membership(features.get(var_name, 0.0), v, term_name, membership)
        mus.append(mu)
    if not mus:
        return 0.0
    # AND combination
    if membership == "gauss":
        # Mamdani uses min; Sugeno uses product
        return min(mus) if True else 1.0  # placeholder, fixed below
    return min(mus)


def _eval_rule_strength(rule: FuzzyRule, features: Dict[str, float],
                        vars: Dict[str, LinguisticVar], membership: str,
                        inference: str) -> float:
    """Firing strength of a rule: AND via min (Mamdani) or product (Sugeno)."""
    mus = []
    for var_name, term_name in rule.antecedent:
        v = vars.get(var_name)
        if v is None or term_name not in v.terms:
            return 0.0
        mu = _term_membership(features.get(var_name, 0.0), v, term_name, membership)
        mus.append(mu)
    if not mus:
        return 0.0
    if inference == "sugeno":
        return float(np.prod(mus)) * rule.weight
    else:  # mamdani
        return float(min(mus)) * rule.weight


def fuzzy_classify(features_row: pd.Series, config: FuzzyConfig,
                   vars: Optional[Dict[str, LinguisticVar]] = None,
                   rules: Optional[List[FuzzyRule]] = None) -> np.ndarray:
    """Classify a single event.  Returns membership degrees for all 6 classes.

    For defuzz='centroid': normalised soft memberships (sum=1).
    For defuzz='mom': hard one-hot from argmax of firing strengths.
    """
    calib = config.calibration if config.rule_set == "calibrated" else None
    if config.rule_set == "calibrated" and calib is None:
        calib = {}
    if vars is None:
        vars = _build_vars(config.membership, calib)
    if rules is None:
        rules = _build_rules(config.rule_set, calib)

    # Reduce variables if n_vars < 6
    if config.n_vars < 6:
        keep = {"zero_frac", "mean_ratio", "mean_ghi"}
        vars = {k: v for k, v in vars.items() if k in keep}

    _defaults = {"rel_ratio": float(features_row.get("mean_ratio", 1.0))}
    feat = {k: float(features_row.get(k, _defaults.get(k, 0.0))) for k in vars}

    # Evaluate all rules
    class_strengths = np.zeros(N_CLASSES)
    for rule in rules:
        strength = _eval_rule_strength(rule, feat, vars, config.membership, config.inference)
        cls = rule.consequent
        # Aggregate: max (OR) across rules for same class
        class_strengths[cls] = max(class_strengths[cls], strength)

    # Default class (Weather) gets a baseline so it's never zero
    if class_strengths[5] < 0.1:
        class_strengths[5] = 0.1

    # Normalize to sum=1 (acts as soft probabilities)
    total = class_strengths.sum()
    if total > 0:
        soft = class_strengths / total
    else:
        soft = class_strengths

    if config.defuzz == "mom":
        # Mean-of-max: hard decision from the fuzzy system
        out = np.zeros(N_CLASSES)
        out[int(np.argmax(class_strengths))] = 1.0
        return out
    return soft


def fuzzy_classify_batch(feat_df: pd.DataFrame, config: FuzzyConfig) -> pd.DataFrame:
    """Classify all events.  Returns DataFrame with one column per class (membership 0..1)."""
    calib = (config.calibration or {}) if config.rule_set == "calibrated" else None
    vars = _build_vars(config.membership, calib)
    rules = _build_rules(config.rule_set, calib)
    if config.n_vars < 6:
        keep = {"zero_frac", "mean_ratio", "mean_ghi"}
        vars = {k: v for k, v in vars.items() if k in keep}

    rows = []
    for _, row in feat_df.iterrows():
        mu = fuzzy_classify(row, config, vars, rules)
        rows.append(mu)
    arr = np.array(rows)
    return pd.DataFrame(arr, index=feat_df.index, columns=CLASS_NAMES)


# ── per-plant aggregation (mirrors most_probable_class) ────────────────────

def fuzzy_plant_summary(feat_df: pd.DataFrame, memberships: pd.DataFrame,
                        all_plants: Sequence[str]) -> pd.DataFrame:
    """Per-plant: duration-weighted fuzzy membership per class.

    Similar to most_probable_class but uses fuzzy degrees instead of
    duration shares of hard cluster labels.
    """
    f = feat_df[["plant", "duration_h"]].copy()
    for c in CLASS_NAMES:
        f[c] = memberships[c].values

    # Duration-weighted average per plant
    tab = f.groupby("plant").apply(
        lambda g: pd.Series({
            c: np.average(g[c], weights=g["duration_h"] + 1e-9) if len(g) > 0 else 0.0
            for c in CLASS_NAMES
        })
    ).fillna(0.0)

    out = pd.DataFrame(index=[p for p in all_plants])
    # Map plant names to short form if needed
    from pv_cohorts import short
    out.index = [short(p) for p in all_plants]
    tab.index = [short(p) for p in tab.index]

    for c in CLASS_NAMES:
        out[c] = tab[c].reindex(out.index).fillna(0.0)

    out["n_events"] = f.groupby("plant").size().reindex(tab.index).reindex(out.index).fillna(0).astype(int)
    out["anomalous_hours"] = f.groupby("plant")["duration_h"].sum().reindex(tab.index).reindex(out.index).fillna(0.0)
    out["most_probable_class"] = out[CLASS_NAMES].idxmax(1)
    out["class_membership"] = out[CLASS_NAMES].max(1)
    out["class_entropy"] = out[CLASS_NAMES].apply(
        lambda r: -np.sum(r * np.log(r + 1e-12)) if r.sum() > 0 else 0.0, axis=1)
    return out


# ── crisp baseline (reimplemented for fair comparison) ────────────────────

def crisp_classify_batch(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Hard labels using the same _label_center logic, one-hot encoded."""
    from pv_clustering import _label_center, ANOMALY_CLASSES
    labels = feat_df.apply(_label_center, axis=1)
    onehot = np.zeros((len(feat_df), N_CLASSES))
    for i, lbl in enumerate(labels):
        idx = ANOMALY_CLASSES.index(lbl) if lbl in ANOMALY_CLASSES else 5
        onehot[i, idx] = 1.0
    return pd.DataFrame(onehot, index=feat_df.index, columns=CLASS_NAMES)


# ── ablation configurations ───────────────────────────────────────────────

def get_ablation_configs() -> Dict[str, FuzzyConfig]:
    """All ablation configurations to evaluate."""
    return {
        "crisp": None,  # special: use crisp_classify_batch
        "fuzzy_full": FuzzyConfig(),  # baseline fuzzy
        "A1_tri": FuzzyConfig(membership="tri"),
        "A2_vars3": FuzzyConfig(n_vars=3),
        "A3_sugeno": FuzzyConfig(inference="sugeno"),
        "A4_mom": FuzzyConfig(defuzz="mom"),
        "A5_reduced": FuzzyConfig(rule_set="reduced"),
    }


# ── metrics for comparison ─────────────────────────────────────────────────

def classification_entropy(memberships: pd.DataFrame) -> float:
    """Mean Shannon entropy of membership distributions (higher = more uncertain)."""
    p = memberships.values
    p = np.clip(p, 1e-12, 1.0)
    h = -np.sum(p * np.log(p), axis=1)
    return float(np.mean(h))


def classification_agreement(m1: pd.DataFrame, m2: pd.DataFrame) -> float:
    """Fraction of events where both methods agree on the most probable class."""
    c1 = m1.values.argmax(1)
    c2 = m2.values.argmax(1)
    return float(np.mean(c1 == c2))


def cross_seed_stability(plant_summaries: List[pd.DataFrame]) -> float:
    """Mean pairwise agreement of most-probable-class per plant across seeds.

    Compares at plant level (not event level) because different seeds produce
    different numbers of events.
    """
    if len(plant_summaries) < 2:
        return 1.0
    # Align on common plant index
    common_idx = plant_summaries[0].index
    for ps in plant_summaries[1:]:
        common_idx = common_idx.intersection(ps.index)
    if len(common_idx) == 0:
        return np.nan
    classes = [ps.loc[common_idx, "most_probable_class"].values for ps in plant_summaries]
    agreements = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            agreements.append(np.mean(classes[i] == classes[j]))
    return float(np.mean(agreements))


# ── top-level runner: extract features from cached events ─────────────────

def load_and_extract_features(cache_dir: str = "results/cache",
                              variant: str = "multiview",
                              seed: int = 0,
                              data: Dict[str, object] = None,
                              clusters: Dict[str, object] = None,
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached model events, extract anomaly features.

    If ``data`` and ``clusters`` dicts are provided (e.g. from
    ``run_pipeline.stage_data``), uses them directly instead of loading
    from ``data.pkl`` / ``clusters.pkl``.  This allows running from a
    preprocessed DataFrame without cache files.

    Returns (feat_df, events_df).
    """
    import pickle, sys
    sys.path.insert(0, "pv_project")
    from pv_clustering import anomaly_event_features
    from pv_cohorts import short

    # Load model results (always from cache — these are trained STGNN outputs)
    run_path = f"{cache_dir}/run_{variant}_{seed}.pkl"
    with open(run_path, "rb") as f:
        run = pickle.load(f)
    events = run["events"]

    # Load shared data: use provided dicts or fall back to cache
    if data is not None:
        df = data["df"]
        phys = data["phys"]
        pv_cols = data["pv_cols"]
    else:
        with open(f"{cache_dir}/data.pkl", "rb") as f:
            data = pickle.load(f)
        df = data["df"]
        phys = data["phys"]
        pv_cols = data["pv_cols"]

    if clusters is not None:
        regimes = clusters["wr"].result.labels if hasattr(clusters["wr"], "result") else clusters["wr"].labels
    else:
        with open(f"{cache_dir}/clusters.pkl", "rb") as f:
            clusters = pickle.load(f)
        regimes = clusters["wr"].result.labels if hasattr(clusters["wr"], "result") else clusters["wr"].labels

    feat = anomaly_event_features(events, df, phys, pv_cols, regimes)
    return feat, events
